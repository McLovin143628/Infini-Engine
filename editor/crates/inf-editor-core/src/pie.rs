//! Editor-side PIE session management (Spike D).
//!
//! Spawns `inf-player --pie` as a subprocess, hands it the cooked snapshot
//! over its stdin, and consumes its event stream. The subprocess boundary
//! is the crash-isolation guarantee: a panicking script kills the *player*,
//! and the editor observes it as [`SessionHealth::Exited`] with the panic
//! text captured from stderr — unsaved editor state is never at risk.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::Receiver;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use inf_runtime::pie::{
    read_msg, write_msg, EditorToPlayer, PlayerToEditor, ScenePayload, ViewportRectMsg,
    PIE_PROTOCOL_VERSION,
};
use inf_runtime::CookedSnapshot;

use uuid::Uuid;

use inf_blueprint::BlueprintClass;
use inf_ecs::components::{ActorClass, AnimPlayer, AnimStateMachine, SkeletalMesh};

use crate::scene::{serialize, SceneDoc};

#[derive(Debug, thiserror::Error)]
pub enum PieError {
    #[error("io error talking to the player: {0}")]
    Io(#[from] std::io::Error),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("player exited during handshake ({0})")]
    HandshakeExit(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionHealth {
    Running,
    /// The player process is gone. `code` is `None` if the OS reported no
    /// exit code (e.g. killed by signal).
    Exited {
        code: Option<i32>,
    },
}

/// A live play-in-editor session (one player subprocess).
pub struct PieSession {
    child: Child,
    stdin: ChildStdin,
    events: Receiver<PlayerToEditor>,
    stderr_lines: Arc<Mutex<Vec<String>>>,
    /// Signalled once by the stderr reader when it reaches **EOF** — i.e. when
    /// every byte the player ever wrote is in `stderr_lines`.
    ///
    /// Process exit and pipe EOF are *different events*, and this is the whole
    /// reason the field exists. `Child::try_wait` reports the first; a reader
    /// thread blocked in `read` observes the second some scheduling quantum
    /// later. Synchronizing on exit and then reading `stderr_lines()` — which is
    /// what this module used to do — is a race whose losing side is an empty or
    /// half-written panic message. It loses rarely, on a loaded CI machine, which
    /// is the worst possible failure rate for a test.
    stderr_eof: Receiver<()>,
    /// `true` once `stderr_eof` has fired. Sticky, because the channel yields
    /// its single message only once.
    stderr_drained: bool,
}

/// Pushed into the captured stderr when the reader could not reach EOF inside a
/// caller's deadline, so **any** assertion that prints the captured output says
/// so instead of quietly comparing against a truncated string.
pub const STDERR_TRUNCATED_MARKER: &str =
    "<inf: the player's stderr did not reach EOF within the deadline — output below is PARTIAL>";

impl PieSession {
    /// Spawn `player_bin --pie`, wire the reader threads, and complete the
    /// version-checked `Ready` handshake (no content sent yet).
    fn spawn_ready(player_bin: &Path) -> Result<Self, PieError> {
        let mut child = Command::new(player_bin)
            .arg("--pie")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        // Protocol frames: reader thread → channel; ends on EOF (player
        // exit) by dropping the sender.
        let (tx, events) = mpsc::channel();
        std::thread::spawn(move || {
            let mut stdout = BufReader::new(stdout);
            while let Ok(msg) = read_msg::<PlayerToEditor>(&mut stdout) {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });

        // Player logs (and panic messages) line-buffered off stderr. The reader
        // signals `eof_tx` when the pipe closes, which is the only moment the
        // capture is known to be complete.
        let stderr_lines = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&stderr_lines);
        let (eof_tx, stderr_eof) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                sink.lock().expect("stderr sink poisoned").push(line);
            }
            // Sending after the loop is what makes this a happens-before edge:
            // every push above is visible to whoever receives this.
            let _ = eof_tx.send(());
        });

        let session = PieSession {
            child,
            stdin,
            events,
            stderr_lines,
            stderr_eof,
            stderr_drained: false,
        };

        match session.next_event(Duration::from_secs(10)) {
            Some(PlayerToEditor::Ready { protocol }) if protocol == PIE_PROTOCOL_VERSION => {
                Ok(session)
            }
            Some(PlayerToEditor::Ready { protocol }) => Err(PieError::Protocol(format!(
                "player speaks protocol {protocol}, editor speaks {PIE_PROTOCOL_VERSION}"
            ))),
            Some(other) => Err(PieError::Protocol(format!("expected Ready, got {other:?}"))),
            None => {
                let mut s = session;
                Err(PieError::HandshakeExit(s.describe_exit()))
            }
        }
    }

    /// Wait for the `Loaded` acknowledgement after a content frame.
    fn await_loaded(mut self) -> Result<Self, PieError> {
        match self.next_event(Duration::from_secs(10)) {
            Some(PlayerToEditor::Loaded { .. }) => Ok(self),
            Some(PlayerToEditor::Error { message }) => {
                Err(PieError::Protocol(format!("player load error: {message}")))
            }
            Some(other) => Err(PieError::Protocol(format!(
                "expected Loaded, got {other:?}"
            ))),
            None => Err(PieError::HandshakeExit(self.describe_exit())),
        }
    }

    /// Spawn `player_bin --pie`, complete the `Ready` handshake, send the toy
    /// [`CookedSnapshot`] (Spike D crash/pause/determinism drills), and wait for
    /// `Loaded`. `tick_hz` paces the toy stream.
    pub fn spawn(
        player_bin: &Path,
        snapshot: &CookedSnapshot,
        _tick_hz: u32,
    ) -> Result<Self, PieError> {
        let mut session = Self::spawn_ready(player_bin)?;
        session.send(&EditorToPlayer::Load(snapshot.clone()))?;
        session.await_loaded()
    }

    /// Spawn `player_bin --pie`, complete the handshake, hand over the **real**
    /// live scene ([`ScenePayload`]: v3 `.inf_lvl` bytes + bound classes), and
    /// wait for `Loaded`. The player builds the world exactly like the shipping
    /// pack path — the PIE == shipping guarantee.
    pub fn spawn_scene(player_bin: &Path, payload: &ScenePayload) -> Result<Self, PieError> {
        let mut session = Self::spawn_ready(player_bin)?;
        session.send(&EditorToPlayer::LoadScene(Box::new(payload.clone())))?;
        session.await_loaded()
    }

    pub fn send(&mut self, msg: &EditorToPlayer) -> Result<(), PieError> {
        write_msg(&mut self.stdin, msg)?;
        Ok(())
    }

    /// Pause the running player.
    pub fn pause(&mut self) -> Result<(), PieError> {
        self.send(&EditorToPlayer::Pause)
    }

    /// Resume a paused player.
    pub fn resume(&mut self) -> Result<(), PieError> {
        self.send(&EditorToPlayer::Resume)
    }

    /// Advance exactly `count` fixed steps (works while paused). The player
    /// streams one `Frame` per step.
    pub fn step(&mut self, count: u32) -> Result<(), PieError> {
        self.send(&EditorToPlayer::Step { count })
    }

    /// Release input possession back to the editor (v1 Eject semantics). The
    /// player keeps running.
    pub fn eject(&mut self) -> Result<(), PieError> {
        self.send(&EditorToPlayer::Eject)
    }

    /// Forward a viewport rect change to an embedded player (physical pixels).
    pub fn set_viewport(
        &mut self,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    ) -> Result<(), PieError> {
        self.send(&EditorToPlayer::SetViewport(ViewportRectMsg {
            x,
            y,
            width,
            height,
        }))
    }

    /// Next player event, or `None` if the stream ended / nothing arrived
    /// within `timeout`.
    pub fn next_event(&self, timeout: Duration) -> Option<PlayerToEditor> {
        self.events.recv_timeout(timeout).ok()
    }

    /// Drain events until `matches` returns true; `None` on timeout or
    /// stream end. Non-matching events are consumed.
    pub fn wait_for(
        &self,
        timeout: Duration,
        mut matches: impl FnMut(&PlayerToEditor) -> bool,
    ) -> Option<PlayerToEditor> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.checked_duration_since(Instant::now())?;
            let event = self.next_event(remaining)?;
            if matches(&event) {
                return Some(event);
            }
        }
    }

    pub fn health(&mut self) -> SessionHealth {
        match self.child.try_wait() {
            Ok(Some(status)) => SessionHealth::Exited {
                code: status.code(),
            },
            Ok(None) => SessionHealth::Running,
            Err(_) => SessionHealth::Exited { code: None },
        }
    }

    /// Poll-wait for the player to exit, **then drain its stderr to EOF** (used
    /// after Stop / a crash).
    ///
    /// Both halves are load-bearing. Returning on process exit alone leaves the
    /// reader thread mid-`read`, so a caller that immediately asks for
    /// [`stderr_lines`](Self::stderr_lines) — to assert on a panic message, say —
    /// races it. The drain uses whatever is left of the same deadline; if it
    /// expires, [`STDERR_TRUNCATED_MARKER`] is pushed into the capture so the
    /// truncation is impossible to miss in a failure message, and
    /// [`stderr_complete`](Self::stderr_complete) reports `false`.
    pub fn wait_exit(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + timeout;
        let status = loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                break Some(status);
            }
            if Instant::now() >= deadline {
                break None;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        self.drain_stderr(deadline);
        status
    }

    /// Block until the stderr reader reports EOF, or `deadline` passes.
    ///
    /// Idempotent and sticky: the channel carries exactly one message, so the
    /// flag is what later calls read.
    fn drain_stderr(&mut self, deadline: Instant) {
        if self.stderr_drained {
            return;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        // A zero-length remaining deadline still gets one non-blocking look, so
        // an already-finished reader is never reported as truncated.
        match self
            .stderr_eof
            .recv_timeout(left.max(Duration::from_millis(1)))
        {
            Ok(()) => self.stderr_drained = true,
            Err(_) => {
                let mut sink = self.stderr_lines.lock().expect("stderr sink poisoned");
                if sink.last().map(String::as_str) != Some(STDERR_TRUNCATED_MARKER) {
                    sink.push(STDERR_TRUNCATED_MARKER.to_string());
                }
            }
        }
    }

    /// `true` when the player's stderr has been read all the way to EOF — i.e.
    /// [`stderr_lines`](Self::stderr_lines) is complete rather than a snapshot.
    pub fn stderr_complete(&self) -> bool {
        self.stderr_drained
    }

    /// Everything the player wrote to stderr so far (its logs; after a
    /// crash, the panic message).
    pub fn stderr_lines(&self) -> Vec<String> {
        self.stderr_lines
            .lock()
            .expect("stderr sink poisoned")
            .clone()
    }

    fn describe_exit(&mut self) -> String {
        let health = self.health();
        format!("{health:?}; stderr: {:?}", self.stderr_lines())
    }

    /// Graceful shutdown: send Stop, wait for exit (kill on timeout).
    pub fn stop(mut self, timeout: Duration) -> Result<ExitStatus, PieError> {
        // The player may already be gone; a send failure is fine.
        let _ = self.send(&EditorToPlayer::Stop);
        match self.wait_exit(timeout) {
            Some(status) => Ok(status),
            None => {
                self.child.kill()?;
                Ok(self.child.wait()?)
            }
        }
    }
}

impl Drop for PieSession {
    fn drop(&mut self) {
        // Never leave an orphaned player: if it's still running when the
        // session handle dies, kill it.
        if matches!(self.health(), SessionHealth::Running) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Build the [`ScenePayload`] handed to the player from the **live** document:
/// the v3 `.inf_lvl` bytes of the current (unsaved-included) doc, plus the bound
/// blueprint classes. Bindings resolve like [`crate::samples::bound_actors`]:
/// the scene's persisted [`ActorClass`] links first (each asset GUID resolved
/// via `resolve`), falling back to the `CharacterController2D` coyote class for
/// scenes authored before per-entity bindings. This is the point of PIE:
/// unsaved edits are previewed exactly.
///
/// # `resolve_voxel`, and the honest limit of "unsaved edits are previewed exactly"
///
/// P21.4 added the fifth resolver: every `VoxelVolume.asset` a level names is
/// resolved to its `.inf_voxel` bytes and shipped, because before it the PIE
/// player had **no voxel source at all** — a Blueprint over a carved hole read the
/// seam's `0.0` in preview and the cave floor in the shipped build, and the phase
/// gate compared two empty maps agreeing.
///
/// **PIE sees the last SAVED cave**, which is the `strip_streamed_terrain`
/// precedent one file over (PIE sees the last saved `.inf_terrain` too) — but here
/// the reason is stronger than symmetry, and it is worth stating because the
/// obvious "fix" is a law violation. Editor *Simulate* does carry unsaved carves
/// ([`crate::simulate::overlay_unsaved_carves`]), by folding the editor store's
/// **dirty** chunks over the resolved map — safe because dirty is a function of
/// the edit history and `sync_residency` refuses to evict a dirty chunk. Shipping
/// that store *as a volume* over the wire is not the same act: the store is
/// **camera-paged**, so its resident set is whatever the author happened to be
/// looking at, and a PIE session built from it would preview a cave truncated by
/// the editor viewport's position. That is precisely the dependency every seam in
/// this phase exists to forbid.
///
/// The parity fix that would work is the dirty-chunk **overlay**, not the volume:
/// ship the saved bytes here plus `(entity, chunk key, chunk bytes)` for the
/// store's dirty set, and let the player apply it with the same rule
/// `overlay_unsaved_carves` uses. That is a second wire field and a second
/// application site, it is not needed by any committed content (a sample is saved
/// by definition), and it is ledgered rather than half-built.
///
/// (Eight parameters trips clippy's arity lint. The alternative — bundling the
/// four byte-resolvers into a struct — would move thirteen call sites to hide one
/// number, and each resolver is a *different* asset kind reaching a *different*
/// store, which a struct of four identically-typed closures makes easier to
/// mis-order rather than harder. The positions are named in the `where` clause and
/// pinned by `a_payload_carries_every_referenced_asset_kind_in_its_own_field`
/// below.)
#[allow(clippy::too_many_arguments)]
pub fn build_scene_payload<F, G, H, B, V, T, M>(
    doc: &SceneDoc,
    mut resolve: F,
    mut resolve_pcg: G,
    mut resolve_anim: H,
    mut resolve_biome_set: B,
    mut resolve_voxel: V,
    mut resolve_terrain: T,
    mut resolve_mesh: M,
    tick_hz: u32,
    windowed: bool,
) -> Result<ScenePayload, PieError>
where
    F: FnMut(Uuid) -> Option<BlueprintClass>,
    G: FnMut(Uuid) -> Option<Vec<u8>>,
    H: FnMut(Uuid) -> Option<Vec<u8>>,
    B: FnMut(Uuid) -> Option<Vec<u8>>,
    V: FnMut(Uuid) -> Option<Vec<u8>>,
    T: FnMut(Uuid) -> Option<Vec<u8>>,
    M: FnMut(Uuid) -> Option<Vec<u8>>,
{
    let level_bytes = serialize::encode(&serialize::to_scene_file(doc))
        .map_err(|e| PieError::Protocol(format!("encode scene: {e}")))?;

    let encode_class = |class: &BlueprintClass| -> Result<Vec<u8>, PieError> {
        crate::samples::encode_actor(class)
            .map_err(|e| PieError::Protocol(format!("encode blueprint class: {e}")))
    };

    let mut classes: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    let world = doc.world();
    for &guid in doc.order() {
        if let Some(e) = world.entity_of(guid) {
            if let Some(ac) = world.world().get::<ActorClass>(e) {
                let asset = ac.0;
                if seen.insert(asset) {
                    if let Some(class) = resolve(asset) {
                        classes.push((asset, encode_class(&class)?));
                    }
                }
            }
        }
    }

    // Legacy fallback: no persisted bindings resolved → send the CC2D coyote
    // class under a synthetic GUID so the player's `resolve_actors` heuristic
    // has a class (mirrors `samples::bound_actors`' fallback).
    if classes.is_empty() {
        if let Some((_g, class)) = crate::samples::character_actors(doc).into_iter().next() {
            const SYNTH: Uuid = Uuid::from_u128(0xFA11_BACC_0000_0001);
            classes.push((SYNTH, encode_class(&class)?));
        }
    }

    // Referenced PCG graphs (P10.6): every `PcgVolume.graph` ref resolved to its
    // `.inf_pcg` bytes, so the PIE player evaluates scatter identically to the
    // shipping pack path (PIE == shipping for terrain/PCG content).
    let mut pcgs: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen_pcg: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for &guid in doc.order() {
        if let Some(e) = world.entity_of(guid) {
            if let Some(vol) = world.world().get::<inf_ecs::components::PcgVolume>(e) {
                if let Some(graph) = vol.graph {
                    if seen_pcg.insert(graph) {
                        if let Some(bytes) = resolve_pcg(graph) {
                            pcgs.push((graph, bytes));
                        }
                    }
                }
            }
        }
    }

    // Referenced biome sets (P19.3): every `Terrain.biome_set` ref resolved to its
    // `.inf_biomes` bytes, so the PIE player runs the biome→PCG binding against
    // the same vocabulary the shipping pack path does.
    //
    // A set's biomes name **graphs**, and the binding needs those too — so each
    // resolved set's `pcg_graph` refs are folded into `pcgs` above through the
    // same `resolve_pcg` closure and the same `seen_pcg` dedupe. That transitive
    // hop is what the cook does when it walks the level's dependency closure; the
    // PIE payload has no dependency graph to walk, so it walks the set itself.
    // Without it a biome-populated level would preview empty while shipping full.
    let mut biome_sets: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen_biome_set: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(set) = world
            .world()
            .get::<inf_ecs::components::Terrain>(e)
            .and_then(|t| t.biome_set)
        else {
            continue;
        };
        if !seen_biome_set.insert(set) {
            continue;
        }
        let Some(bytes) = resolve_biome_set(set) else {
            continue;
        };
        // A set that does not decode is the cook's advisory, not a load failure:
        // ship the bytes anyway and let the player report on them.
        if let Ok(decoded) = inf_asset::decode::<inf_terrain::BiomeSet>(&bytes) {
            for graph in decoded.biomes.iter().filter_map(|b| b.pcg_graph) {
                if seen_pcg.insert(graph) {
                    if let Some(g) = resolve_pcg(graph) {
                        pcgs.push((graph, g));
                    }
                }
            }
        }
        biome_sets.push((set, bytes));
    }

    // Referenced P11 animation assets (P11.4): the directly-referenced
    // `SkeletalMesh.skeleton` / `AnimPlayer.clip` / `AnimStateMachine.sm` GUIDs
    // resolved to their `.inf_skel` / `.inf_anim` / `.inf_sm` bytes, so the PIE
    // player resolves state machines + root-motion clips exactly like the shipping
    // pack path (PIE == shipping for animation). A machine's transitively-played
    // clips ship with the cooked pack for pose rendering (human-verified);
    // gate (c) asserts the state trace, which needs only the machine + actor vars.
    let mut skeletons: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut clips: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut machines: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen_anim: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    let collect = |guid: Uuid,
                   out: &mut Vec<(Uuid, Vec<u8>)>,
                   resolve_anim: &mut H,
                   seen: &mut std::collections::HashSet<Uuid>| {
        if seen.insert(guid) {
            if let Some(bytes) = resolve_anim(guid) {
                out.push((guid, bytes));
            }
        }
    };
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let w = world.world();
        if let Some(sk) = w.get::<SkeletalMesh>(e).and_then(|s| s.skeleton) {
            collect(sk, &mut skeletons, &mut resolve_anim, &mut seen_anim);
        }
        if let Some(clip) = w.get::<AnimPlayer>(e).and_then(|p| p.clip) {
            collect(clip, &mut clips, &mut resolve_anim, &mut seen_anim);
        }
        if let Some(sm) = w.get::<AnimStateMachine>(e).and_then(|s| s.sm) {
            collect(sm, &mut machines, &mut resolve_anim, &mut seen_anim);
        }
    }

    // Referenced voxel volumes (P21.4): every `VoxelVolume.asset` ref resolved to
    // its `.inf_voxel` bytes, so the PIE player seeds the SAME sim-side volume map
    // the cooked pack path seeds — the M9 debt.
    //
    // Keyed by ASSET (two entities may reference one cave at two transforms); the
    // per-entity world anchor is folded in on the player side by
    // `resolve_voxel_volumes`, exactly as it is on the editor Simulate side. A
    // volume the caller cannot serve is skipped rather than faked: an unresolvable
    // ref previews as no cave, which is what the shipped build does with the same
    // dangling reference.
    let mut voxels: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen_voxel: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(asset) = world
            .world()
            .get::<inf_ecs::components::VoxelVolume>(e)
            .and_then(|v| v.asset)
        else {
            continue;
        };
        if !seen_voxel.insert(asset) {
            continue;
        }
        if let Some(bytes) = resolve_voxel(asset) {
            voxels.push((asset, bytes));
        }
    }

    // Referenced streamed terrains (P21.4, the P16.3b2 deferral): every
    // `Terrain.asset` ref resolved to its `.inf_terrain` bytes.
    //
    // `strip_streamed_terrain` above blanked the inline working set of exactly
    // these terrains, on the rule that PIE previews what was SAVED. Until this
    // resolver existed that left the PIE player with no ground under an
    // asset-backed terrain at all — tolerable while the only casualty was
    // detail, and not tolerable since P21.2 put the **hole mask** in the asset:
    // a level whose caves have mouths could not be previewed without it.
    let mut terrains: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen_terrain: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(asset) = world
            .world()
            .get::<inf_ecs::components::Terrain>(e)
            .and_then(|t| t.asset)
        else {
            continue;
        };
        if !seen_terrain.insert(asset) {
            continue;
        }
        if let Some(bytes) = resolve_terrain(asset) {
            terrains.push((asset, bytes));
        }
    }

    // Derived fracture chunk sets (P22.3): for every `Destructible` actor, the
    // `.inf_fracture` its own `MeshRef.asset` implies.
    //
    // **Computed here rather than resolved**, and it is the only entry in this
    // payload that is. A `.inf_fracture` does not exist in the editor's content
    // root — it is derived at *cook*, from the mesh, because "is this mesh
    // destructible" is a fact about a level and not about the mesh. So PIE runs
    // the same derivation the cook runs: `inf_mesh::fracture_mesh` over the same
    // `.inf_mesh` bytes with the same `Destructible::{fracture_seed, chunk_count}`,
    // keyed by the same `derived_fracture_id`. A deterministic function of the
    // same inputs cannot give the preview a different building from the shipped
    // one — which is the whole claim, and it is why the resolver here hands back
    // MESH bytes rather than fracture bytes.
    //
    // One fracture per mesh, not per actor: the derived id is a function of the
    // mesh's, so two walls sharing a mesh share a chunk set. When two actors
    // disagree about the parameters, the first in document order wins — the same
    // rule (and the same advisory) `inf_packager::cook::plan_fractures` applies.
    // A mesh the caller cannot serve, or one the fracture refuses (too small,
    // degenerate, too few chunks), is simply absent: the actor previews as
    // indestructible, which is exactly what the shipped build does with the same
    // refusal.
    let mut fractures: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen_fracture: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let w = world.world();
        let Some(d) = w.get::<inf_ecs::components::Destructible>(e).copied() else {
            continue;
        };
        let Some(mesh_id) = w
            .get::<inf_ecs::components::MeshRef>(e)
            .and_then(|m| m.asset)
        else {
            continue;
        };
        if !seen_fracture.insert(mesh_id) {
            continue;
        }
        let Some(bytes) = resolve_mesh(mesh_id) else {
            continue;
        };
        let Ok(mesh) = inf_asset::decode::<inf_mesh::MeshAsset>(&bytes) else {
            continue;
        };
        let params = inf_mesh::FractureParams {
            seed: d.fracture_seed,
            chunk_count: d.chunk_count,
        };
        let Ok(asset) = inf_mesh::fracture_mesh(&mesh, inf_asset::AssetId(mesh_id), params) else {
            continue;
        };
        let Ok(encoded) = inf_asset::encode(&asset) else {
            continue;
        };
        fractures.push((
            inf_mesh::derived_fracture_id(inf_asset::AssetId(mesh_id)).uuid(),
            encoded,
        ));
    }

    Ok(
        ScenePayload::new(doc.title(), level_bytes, classes, tick_hz, windowed)
            .with_pcgs(pcgs)
            .with_biome_sets(biome_sets)
            .with_anim_assets(skeletons, clips, machines)
            .with_voxels(voxels)
            .with_terrains(terrains)
            .with_fractures(fractures),
    )
}

/// Locate the `inf-player` binary next to the running editor executable (dev
/// and shipped both place it in the same directory). Honours the
/// `INF_PLAYER_BIN` environment override. Returns the first existing candidate,
/// else the sibling path (so a spawn failure names the expected location).
pub fn find_player_bin() -> std::path::PathBuf {
    use std::path::PathBuf;
    if let Ok(p) = std::env::var("INF_PLAYER_BIN") {
        return PathBuf::from(p);
    }
    let exe_name = if cfg!(windows) {
        "inf-player.exe"
    } else {
        "inf-player"
    };
    let sibling = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(exe_name)));
    match sibling {
        Some(p) => p,
        None => PathBuf::from(exe_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;
    use inf_ecs::components::{PcgVolume, Terrain, VoxelVolume};

    /// **The positional pin for [`build_scene_payload`]'s five resolvers.**
    ///
    /// Each resolver reaches a different store and fills a different payload
    /// field, and every one of them has the same type — `FnMut(Uuid) ->
    /// Option<Vec<u8>>` — so a swapped pair compiles silently and ships a level's
    /// PCG graphs as its biome sets. This hands each resolver a *distinguishable*
    /// payload and asserts where it landed, which is the check a struct of four
    /// identically-typed closures would not have given us either.
    #[test]
    fn a_payload_carries_every_referenced_asset_kind_in_its_own_field() {
        const CLASS: Uuid = Uuid::from_u128(0x2104_0E01);
        const GRAPH: Uuid = Uuid::from_u128(0x2104_0E02);
        const SKEL: Uuid = Uuid::from_u128(0x2104_0E03);
        const BIOMES: Uuid = Uuid::from_u128(0x2104_0E04);
        const VOXELS: Uuid = Uuid::from_u128(0x2104_0E05);
        const TERRAIN: Uuid = Uuid::from_u128(0x2104_0E06);

        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Empty, "Everything", None);
        {
            let world = doc.world_mut();
            let id = world.entity_of(e).expect("the entity exists");
            world.world_mut().entity_mut(id).insert((
                ActorClass(CLASS),
                PcgVolume {
                    graph: Some(GRAPH),
                    ..PcgVolume::default()
                },
                Terrain {
                    biome_set: Some(BIOMES),
                    asset: Some(TERRAIN),
                    ..Terrain::default()
                },
                SkeletalMesh {
                    skeleton: Some(SKEL),
                    ..SkeletalMesh::default()
                },
                VoxelVolume::from_asset(VOXELS),
            ));
            world.mark_dirty();
        }

        let payload = build_scene_payload(
            &doc,
            |g| (g == CLASS).then(|| BlueprintClass::new("act:probe", "Probe")),
            |g| (g == GRAPH).then(|| b"PCG".to_vec()),
            |g| (g == SKEL).then(|| b"SKEL".to_vec()),
            |g| (g == BIOMES).then(|| b"BIOMES".to_vec()),
            |g| (g == VOXELS).then(|| b"VOXELS".to_vec()),
            |g| (g == TERRAIN).then(|| b"TERRAIN".to_vec()),
            // P22.3: no destructible meshes in this fixture.
            |_| None,
            60,
            false,
        )
        .expect("payload builds");

        assert_eq!(payload.classes.len(), 1);
        assert_eq!(payload.classes[0].0, CLASS);
        assert_eq!(payload.pcgs, vec![(GRAPH, b"PCG".to_vec())]);
        assert_eq!(payload.skeletons, vec![(SKEL, b"SKEL".to_vec())]);
        assert_eq!(payload.biome_sets, vec![(BIOMES, b"BIOMES".to_vec())]);
        assert_eq!(payload.voxels, vec![(VOXELS, b"VOXELS".to_vec())]);
        assert_eq!(payload.terrains, vec![(TERRAIN, b"TERRAIN".to_vec())]);
        assert_eq!(
            payload.schema_version,
            inf_runtime::pie::SCENE_PAYLOAD_VERSION
        );
    }

    /// A resolver that cannot serve an asset leaves the field **empty** rather than
    /// inventing one: an unresolvable reference previews as absent content, which
    /// is exactly what the shipped build does with the same dangling ref.
    #[test]
    fn an_unresolvable_voxel_reference_ships_nothing() {
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Empty, "Cave", None);
        {
            let world = doc.world_mut();
            let id = world.entity_of(e).expect("the entity exists");
            world
                .world_mut()
                .entity_mut(id)
                .insert(VoxelVolume::from_asset(Uuid::from_u128(0xDEAD)));
            world.mark_dirty();
        }
        let payload = build_scene_payload(
            &doc,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            // P22.3: no destructible meshes in this fixture.
            |_| None,
            60,
            false,
        )
        .expect("payload builds");
        assert!(payload.voxels.is_empty());
        assert!(payload.terrains.is_empty());
    }
}
