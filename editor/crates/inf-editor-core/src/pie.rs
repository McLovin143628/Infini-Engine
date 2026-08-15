//! Editor-side PIE session management (Spike D).
//!
//! Spawns `inf-player --pie` as a subprocess, hands it the cooked snapshot
//! over its stdin, and consumes its event stream. The subprocess boundary
//! is the crash-isolation guarantee: a panicking script kills the *player*,
//! and the editor observes it as [`SessionHealth::Exited`] with the panic
//! text captured from stderr — unsaved editor state is never at risk.

use std::collections::BTreeMap;
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
    stderr_lines: Arc<Mutex<StderrCapture>>,
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

/// Lines kept from the **start** of the player's stderr — the banner, the adapter
/// it picked, what it loaded. The half a crash report needs for context.
const STDERR_HEAD_LINES: usize = 256;
/// Lines kept from the **end**. A panic message and its backtrace are the last
/// thing a player writes, and they are the reason anybody reads this capture at
/// all, so the tail is the generous half.
const STDERR_TAIL_LINES: usize = 1024;

/// The player's captured stderr, bounded head-and-tail (Hardening D).
///
/// This grew without limit for the life of a PIE session — a chatty player, a
/// long play-test, or a log loop is unbounded editor memory. It is **not** a
/// plain ring, because the two ends carry different information: the head is the
/// session's provenance and the tail is the panic. So both are kept and the
/// middle is what goes, with an elision line synthesized at read time so a
/// failure message can never quietly present a gapped capture as a whole one.
#[derive(Debug, Default)]
struct StderrCapture {
    head: Vec<String>,
    tail: std::collections::VecDeque<String>,
    dropped: u64,
}

impl StderrCapture {
    fn push(&mut self, line: String) {
        if self.head.len() < STDERR_HEAD_LINES {
            self.head.push(line);
            return;
        }
        self.tail.push_back(line);
        while self.tail.len() > STDERR_TAIL_LINES {
            self.tail.pop_front();
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    /// `true` when the newest retained line is `s` — the "marker already pushed"
    /// test, which must read the *end* of the capture whichever half holds it.
    fn last_is(&self, s: &str) -> bool {
        self.tail
            .back()
            .or_else(|| self.head.last())
            .map(String::as_str)
            == Some(s)
    }

    /// Head + (elision line, when anything was dropped) + tail.
    fn lines(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.head.len() + self.tail.len() + 1);
        out.extend(self.head.iter().cloned());
        if self.dropped > 0 {
            out.push(format!(
                "<inf: {} line(s) of the player's stderr elided between the first {} and the \
                 last {}>",
                self.dropped,
                self.head.len(),
                self.tail.len()
            ));
        }
        out.extend(self.tail.iter().cloned());
        out
    }
}

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
        let stderr_lines = Arc::new(Mutex::new(StderrCapture::default()));
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
                if !sink.last_is(STDERR_TRUNCATED_MARKER) {
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
    ///
    /// **Bounded head-and-tail** (Hardening D): the first `STDERR_HEAD_LINES`
    /// and the last `STDERR_TAIL_LINES`, with an elision line between them
    /// naming how many went, so the panic tail is always present and a gapped
    /// capture always says it is one.
    pub fn stderr_lines(&self) -> Vec<String> {
        self.stderr_lines
            .lock()
            .expect("stderr sink poisoned")
            .lines()
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
pub fn build_scene_payload<F, G, H, B, V, T, M, A>(
    doc: &SceneDoc,
    mut resolve: F,
    mut resolve_pcg: G,
    mut resolve_anim: H,
    mut resolve_biome_set: B,
    mut resolve_voxel: V,
    mut resolve_terrain: T,
    mut resolve_mesh: M,
    // P26.3b: ONE resolver for the four kinds v8 added — `.inf_cloth`,
    // `.inf_hair`, `.inf_mat` and `.inf_tex`.
    //
    // One and not four, deliberately, and the reason is the arity note above
    // turned around: the six resolvers before it each reach a *different* store
    // with a *different* rule (a blueprint decodes to a class, a mesh is read to
    // be DERIVED from, a terrain's inline set was stripped in its favour). These
    // four are the same act — read this asset's committed bytes and ship them —
    // so four parameters would be four ways to make the same call and four
    // chances to mis-order them.
    mut resolve_bytes: A,
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
    A: FnMut(Uuid) -> Option<Vec<u8>>,
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
    // pack path (PIE == shipping for animation).
    //
    // **P24.1 closes the transitive hop.** This comment used to say a machine's
    // transitively-played clips "ship with the cooked pack for pose rendering
    // (human-verified)", and that the state trace "needs only the machine + actor
    // vars". The first half was true — `inf_packager::cook::asset_deps` has closed
    // the `state machine → clip` edge since P11.4 — and the second half stopped
    // being true the moment the machine started driving the DRAWN pose. A payload
    // that shipped the machine and not its clips would preview every state as the
    // rest pose while the shipped build animated: exactly the divergence this file
    // exists to prevent, arriving through the one asset nothing references by
    // component. Both closures now go through the SAME Ring-0 walk
    // (`inf_anim::StateMachine::clip_refs`) rather than two that agree by
    // construction.
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
    // The transitive hop: every clip the resolved machines' states play. Walked
    // over the machine BYTES that were just collected, so a machine that did not
    // resolve contributes nothing and the payload never claims a clip it has no
    // machine for. `Guid` order (the Ring-0 walk returns a `BTreeSet`), so the
    // payload is a deterministic function of the document.
    let machine_clips: Vec<Uuid> = machines
        .iter()
        .filter_map(|(guid, bytes)| {
            // Reported, not discarded (P24.1 audit B1, the same class one level
            // over): a machine whose bytes will not decode here still SHIPS — it
            // was collected above — while its clips silently do not, so PIE would
            // pose every state at rest while the cooked build animates.
            match inf_asset::decode::<inf_anim::StateMachineAsset>(bytes) {
                Ok(a) => Some(a),
                Err(e) => {
                    tracing::warn!(
                        "pie: state machine {guid} ships but its clips could not be \
                         resolved, so its states will pose at rest in PIE: {e}"
                    );
                    None
                }
            }
        })
        .flat_map(|a| crate::simulate::machine_clip_refs(&a.machine))
        .collect::<std::collections::BTreeSet<Uuid>>()
        .into_iter()
        .collect();
    for clip in machine_clips {
        collect(clip, &mut clips, &mut resolve_anim, &mut seen_anim);
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
    // Referenced skeletal meshes (P24.1): every `SkeletalMesh.mesh` ref resolved to
    // its `.inf_mesh` bytes, through the SAME `resolve_mesh` door the fracture
    // derivation below reads.
    //
    // The one part of a character that never crossed this wire. Skeletons, clips
    // and machines have ridden it since v3, so a PIE session stepped its
    // characters exactly like the shipped build — and drew every one of them as a
    // placeholder cube, because the mesh bytes the skinned pass needs were absent
    // and `SkinnedRegistry::resolve_skinned` therefore missed on rule 1.
    //
    // Carried, not computed: the bind-space rebuild that turns these bytes into a
    // `SkinnedMeshData` is a MIRROR pair between the two render stores, so doing
    // it here would put a third copy of it in the tree.
    let mut meshes: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen_mesh: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(mesh) = world.world().get::<SkeletalMesh>(e).and_then(|s| s.mesh) else {
            continue;
        };
        if !seen_mesh.insert(mesh) {
            continue;
        }
        if let Some(bytes) = resolve_mesh(mesh) {
            meshes.push((mesh, bytes));
        }
    }

    // Referenced garments and hairstyles (P26.3b · `ScenePayload` v8): every
    // `ClothSim.asset` / `HairGuides.asset` ref resolved to its `.inf_cloth` /
    // `.inf_hair` bytes.
    //
    // **The P24.4 debt.** The wire had no slot for these, so a character
    // previewed in PIE wore nothing while the same character in Simulate and in
    // a cooked pack wore both — `phase24_gate`'s trip-wire measured exactly that
    // and fired when v8 landed.
    //
    // Carried, not computed: a `.inf_cloth` is authored (the Model Editor's cloth
    // door tunes stiffness and pins hems), so re-deriving it here would throw the
    // tuning away and give the preview a different garment from the shipped
    // build.
    let mut cloths: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut hairs: Vec<(Uuid, Vec<u8>)> = Vec::new();
    // Two dedup sets, not one (P26.3b audit). `.inf_cloth` and `.inf_hair` are
    // two id spaces, and a single `seen` set makes a hairstyle whose GUID
    // collides with a garment's silently absent from the wire — an impossible
    // state that costs nothing to make structurally impossible instead of
    // arguing about.
    let mut seen_cloth: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    let mut seen_hair: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let w = world.world();
        if let Some(a) = w
            .get::<inf_ecs::components::ClothSim>(e)
            .and_then(|c| c.asset)
        {
            if seen_cloth.insert(a) {
                if let Some(bytes) = resolve_bytes(a) {
                    cloths.push((a, bytes));
                }
            }
        }
        if let Some(a) = w
            .get::<inf_ecs::components::HairGuides>(e)
            .and_then(|h| h.asset)
        {
            if seen_hair.insert(a) {
                if let Some(bytes) = resolve_bytes(a) {
                    hairs.push((a, bytes));
                }
            }
        }
    }

    // Referenced materials (P26.3b · scene v22 · `ScenePayload` v8): every
    // `Material.asset` binding resolved to its `.inf_mat` bytes and **derived**
    // through the one door, plus the `.inf_tex` containers the derived records
    // name.
    //
    // **Computed, not carried**, and for a sharper reason than the fractures one
    // rung down: the player cannot decode a `.inf_mat` at all. `MaterialAsset`
    // lives in `inf-material`, which owns the `image` decoders and the
    // naga-validating material compiler, and the P26.2 dependency inversion
    // exists to keep both out of a shipped build. So this runs the SAME
    // `inf_material::derive_material_bytes` the cook runs, keyed by the SAME
    // `derived_material_id` the cook writes under — one deterministic function
    // over one set of bytes, which is what makes PIE == shipping a property of
    // the code rather than of two collectors agreeing.
    //
    // The textures ride along because a record that names textures with no bytes
    // behind it previews untextured while the shipped build is textured, and a
    // comparison between two such worlds passes (the P21.4 hazard).
    let mut materials: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut textures: Vec<(Uuid, Vec<u8>)> = Vec::new();
    let mut seen_material: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    let mut seen_texture: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(asset) = world
            .world()
            .get::<inf_ecs::components::Material>(e)
            .and_then(|m| m.asset)
        else {
            continue;
        };
        if !seen_material.insert(asset) {
            continue;
        }
        let Some(bytes) = resolve_bytes(asset) else {
            continue;
        };
        // Reported, not discarded (the P24.1 audit's B1 shape, a third time): a
        // `.inf_mat` an older build wrote does not decode, and this path would
        // silently drop it while the COOK — reading the same file through the
        // same door — still derives one. The preview would then show a scalar
        // surface and the shipped build a textured one, with no message
        // anywhere.
        let derived = match inf_material::derive_material_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    "pie: material {asset} did not decode, so every surface bound to it \
                     previews UNTEXTURED while the cooked build still resolves it: {e}"
                );
                continue;
            }
        };
        for tex in derived.texture_dependencies() {
            let g = tex.uuid();
            if seen_texture.insert(g) {
                match resolve_bytes(g) {
                    Some(bytes) => textures.push((g, bytes)),
                    // Reported, not discarded (P26.3b audit — the same doctrine
                    // the material decode ten lines up already follows, and the
                    // hazard the `ScenePayload::textures` doc names in so many
                    // words). The record still ships, naming a texture with no
                    // bytes behind it: the preview is untextured while the
                    // COOKED build — which packs through its own closure and
                    // raises `missing_material_texture_advisory` — may not be.
                    // A comparison between two such worlds passes (P21.4).
                    None => tracing::warn!(
                        "pie: material {asset} references texture {g}, which did not \
                         resolve — this surface previews UNTEXTURED, and the cooked \
                         build is where the advisory for it is raised"
                    ),
                }
            }
        }
        let Ok(encoded) = inf_asset::encode(&derived) else {
            continue;
        };
        materials.push((
            inf_asset::derived_material_id(inf_asset::AssetId(asset)).uuid(),
            encoded,
        ));
    }

    let mut fractures: Vec<(Uuid, Vec<u8>)> = Vec::new();
    for (mesh_id, params) in destructible_mesh_params(doc) {
        let Some(bytes) = resolve_mesh(mesh_id) else {
            continue;
        };
        // Reported, not discarded (P24.1 re-audit adoption — B1's exact shape,
        // 130 lines down the same function).
        //
        // A `.inf_mesh` an older build wrote does not decode (bincode is
        // positional), so this `else { continue }` dropped the actor's fracture
        // from the payload — while the COOK, reading the same mesh through its own
        // decode, still derived one. PIE then previewed an indestructible prop and
        // the shipped build broke it, with no message anywhere. Silent hazards get
        // advisories; that is the cook-advisory doctrine, and it applies to the
        // payload builder because the payload IS the preview's content.
        let mesh = match inf_asset::decode::<inf_mesh::MeshAsset>(&bytes) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(
                    "pie: destructible mesh {mesh_id} did not decode, so this actor \
                     previews as INDESTRUCTIBLE while the cooked build still \
                     fractures it: {e}"
                );
                continue;
            }
        };
        let Some(asset) = derive_fracture(&mesh, mesh_id, params) else {
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
            .with_fractures(fractures)
            .with_meshes(meshes)
            .with_garments(cloths, hairs)
            .with_materials(materials, textures),
    )
}

/// **The one rule for "which destructible's parameters a shared mesh fractures
/// with"** (P22.3), walked in `doc.order()`.
///
/// Two actors may reference one mesh and disagree about `fracture_seed` /
/// `chunk_count`; the fracture is derived per MESH, so exactly one of them wins.
/// The cook resolves that in document order and takes the first
/// (`inf_packager::cook::plan_fractures`), and this is the same walk — factored
/// out and shared rather than restated, because it had already been restated
/// once and the copies disagreed.
///
/// The P22.3 audit's M2: the editor's Simulate seeder resolved the winner with
/// `iter_entities()`, which is **ECS ARCHETYPE order** — a function of which
/// components an entity happens to carry and in what sequence they were
/// inserted. Adding an `AlwaysLoaded` to one of two walls sharing a mesh flipped
/// which one won, so Simulate shattered a wall into 24 chunks that the shipped
/// pack shattered into 8. The comment above that seeder claimed the three paths
/// agreed "by construction rather than by agreement"; they agreed by neither.
/// Now there is one function, and construction is the truth.
///
/// Returns `mesh GUID -> params`, first-in-document-order wins, deterministic.
pub fn destructible_mesh_params(doc: &SceneDoc) -> BTreeMap<Uuid, inf_mesh::FractureParams> {
    let world = doc.world();
    let mut out: BTreeMap<Uuid, inf_mesh::FractureParams> = BTreeMap::new();
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
        out.entry(mesh_id).or_insert(inf_mesh::FractureParams {
            seed: d.fracture_seed,
            chunk_count: d.chunk_count,
        });
    }
    out
}

/// Derive one mesh's `.inf_fracture`, applying **the cook's own refusals**
/// (P22.3).
///
/// The second half is the P22.3 audit's M3. `inf_packager::cook` refuses the
/// WHOLE fracture if any chunk cannot become a collider, so a mesh that fails it
/// ships as an indestructible actor. PIE skipped that check and encoded
/// unconditionally, so the same wall broke in the preview and did not in the
/// shipped build, with one of its chunks rendering without colliding.
///
/// The re-audit then found the deeper version of the same problem: "same door"
/// was two doors that happened to be spelled the same. Both now call
/// [`inf_physics::d3::first_uncollidable_chunk`], which is in the crate the answer
/// is *about* — and `runtime/inf-player/tests/fracture_equivalence.rs` compares
/// the two paths' BYTES over a level, so "same answer" is measured rather than
/// asserted in a comment.
pub fn derive_fracture(
    mesh: &inf_mesh::MeshAsset,
    mesh_id: Uuid,
    params: inf_mesh::FractureParams,
) -> Option<inf_mesh::FractureAsset> {
    let asset = inf_mesh::fracture_mesh(mesh, inf_asset::AssetId(mesh_id), params).ok()?;
    // The cook's refusal, through the cook's door — `inf_physics::d3` is where it
    // lives precisely so this cannot become a second copy that agrees by
    // coincidence (it did once, and then stopped agreeing).
    inf_physics::d3::first_uncollidable_chunk(&asset)
        .is_none()
        .then_some(asset)
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

    /// **The positional pin for [`build_scene_payload`]'s seven resolvers.**
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
        const MESH: Uuid = Uuid::from_u128(0x2203_0E07);
        /// The SkeletalMesh's own `.inf_mesh` (P24.1, payload v7) — a different
        /// asset from the destructible's, so "the mesh resolver filled `meshes`"
        /// and "it filled `fractures`" are distinguishable.
        const SKMESH: Uuid = Uuid::from_u128(0x2401_0E08);
        /// The `.inf_sm` and the clip its only state plays. The clip is named by
        /// NO component — it lives inside the machine's payload — so it is only in
        /// the payload if the transitive walk ran.
        const SM: Uuid = Uuid::from_u128(0x2401_0E09);
        const SM_CLIP: Uuid = Uuid::from_u128(0x2401_0E0A);

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
                    mesh: Some(SKMESH),
                },
                inf_ecs::components::AnimStateMachine {
                    sm: Some(SM),
                    ..Default::default()
                },
                VoxelVolume::from_asset(VOXELS),
                // P22.3: the seventh resolver is the odd one out — it hands
                // back MESH bytes, and the payload carries the fracture
                // DERIVED from them under a different id. A fixture with no
                // `Destructible` could not tell that apart from a resolver
                // that was never called, which is exactly what the first cut
                // of this pin did: mutating `.with_fractures(fractures)` to
                // `Vec::new()` left every test in the tree green.
                inf_ecs::components::Destructible {
                    chunk_count: 6,
                    ..inf_ecs::components::Destructible::default()
                },
                inf_ecs::components::MeshRef {
                    asset: Some(MESH),
                    ..Default::default()
                },
            ));
            world.mark_dirty();
        }

        let payload = build_scene_payload(
            &doc,
            |g| (g == CLASS).then(|| BlueprintClass::new("act:probe", "Probe")),
            |g| (g == GRAPH).then(|| b"PCG".to_vec()),
            |g| {
                if g == SKEL {
                    Some(b"SKEL".to_vec())
                } else if g == SM {
                    Some(
                        inf_asset::encode(&inf_anim::StateMachineAsset::new(
                            inf_anim::StateMachine {
                                states: vec![inf_anim::SmState::clip("idle", *SM_CLIP.as_bytes())],
                                transitions: vec![],
                                entry: 0,
                                ..Default::default()
                            },
                            None,
                        ))
                        .expect("encodes"),
                    )
                } else if g == SM_CLIP {
                    Some(b"SM_CLIP".to_vec())
                } else {
                    None
                }
            },
            |g| (g == BIOMES).then(|| b"BIOMES".to_vec()),
            |g| (g == VOXELS).then(|| b"VOXELS".to_vec()),
            |g| (g == TERRAIN).then(|| b"TERRAIN".to_vec()),
            |g| {
                if g == MESH {
                    Some(inf_asset::encode(&fracture_cube()).expect("encodes"))
                } else if g == SKMESH {
                    Some(b"SKMESH".to_vec())
                } else {
                    None
                }
            },
            // P26.3b: the fourth-kind byte resolver (cloth / hair / material /
            // texture). `None` here is the honest fixture default — this arm authors
            // none of the four.
            |_| None,
            60,
            false,
        )
        .expect("payload builds");

        assert_eq!(payload.classes.len(), 1);
        assert_eq!(payload.classes[0].0, CLASS);
        assert_eq!(payload.pcgs, vec![(GRAPH, b"PCG".to_vec())]);
        assert_eq!(payload.skeletons, vec![(SKEL, b"SKEL".to_vec())]);
        // P24.1 (v7): the SkeletalMesh's own `.inf_mesh`, through the SAME resolver
        // the fracture derivation reads — and it must not be the destructible's.
        assert_eq!(
            payload.meshes,
            vec![(SKMESH, b"SKMESH".to_vec())],
            "the payload carries no skeletal mesh — every character in a windowed \
             PIE session would draw as a placeholder cube"
        );
        // P24.1: the machine, and the clip its only state plays. The clip is named
        // by no component at all; it is in the payload only because the transitive
        // `state machine → clip` walk ran. Without it the shipped build animates
        // and the preview stands still.
        assert_eq!(payload.machines.len(), 1);
        assert_eq!(payload.machines[0].0, SM);
        assert!(
            payload.clips.iter().any(|(g, _)| *g == SM_CLIP),
            "the payload dropped the clip the machine's state plays: {:?}",
            payload.clips.iter().map(|(g, _)| *g).collect::<Vec<_>>()
        );
        assert_eq!(payload.biome_sets, vec![(BIOMES, b"BIOMES".to_vec())]);
        assert_eq!(payload.voxels, vec![(VOXELS, b"VOXELS".to_vec())]);
        assert_eq!(payload.terrains, vec![(TERRAIN, b"TERRAIN".to_vec())]);
        // The fracture is keyed by the DERIVED id, not by the mesh's — the same
        // id the cooked pack stores it under, so the player has one lookup rule.
        assert_eq!(
            payload.fractures.len(),
            1,
            "the payload carries no fracture"
        );
        assert_eq!(
            payload.fractures[0].0,
            inf_mesh::derived_fracture_id(inf_asset::AssetId(MESH)).uuid()
        );
        assert_ne!(
            payload.fractures[0].0, MESH,
            "the fracture is keyed by the mesh's own id — the player would look it \
             up under the derived one and find nothing"
        );
        // …and the bytes really are a chunk set with the AUTHORED chunk count's
        // worth of pieces, not an empty vector that happens to decode.
        let decoded = inf_asset::decode::<inf_mesh::FractureAsset>(&payload.fractures[0].1)
            .expect("the payload's fracture decodes");
        assert_eq!(decoded.requested_chunks, 6);
        assert!(decoded.chunks.len() >= 2, "the fracture has no chunks");
        assert_eq!(
            payload.schema_version,
            inf_runtime::pie::SCENE_PAYLOAD_VERSION
        );
    }

    /// **A stale `.inf_mesh` is refused BY NAME, not dropped** (P24.1 re-audit
    /// adoption).
    ///
    /// The payload builder's mesh decode was `let Ok(m) = … else { continue }`.
    /// A `.inf_mesh` an older build wrote does not decode — bincode is positional
    /// — so the actor's fracture silently left the payload while the cook, reading
    /// the same file, still derived one: PIE previews an indestructible prop and
    /// the shipped build breaks it. Exactly B1's shape, 130 lines down the same
    /// function.
    ///
    /// What this pins is the **named** error the warning now carries, on the same
    /// door every asset kind goes through, plus the honest outcome: the payload
    /// ships no fracture for that mesh rather than a wrong one.
    #[test]
    fn a_stale_destructible_mesh_is_named_and_ships_no_fracture() {
        const MESH: Uuid = Uuid::from_u128(0x2401_C0DE);

        // A `MeshAsset` payload from a build before its current tail — modelled
        // the way it happens, by encoding a shorter shape.
        #[derive(serde::Serialize)]
        struct MeshAssetV1 {
            schema_version: u32,
        }
        let stale = bincode::serde::encode_to_vec(
            &MeshAssetV1 { schema_version: 1 },
            inf_asset::bincode_config(),
        )
        .unwrap();

        // The refusal is named, and it is the one the warning renders.
        let err = inf_asset::decode::<inf_mesh::MeshAsset>(&stale)
            .expect_err("a truncated mesh payload must not decode");
        assert!(
            matches!(
                err,
                inf_asset::AssetError::SchemaTooOld { kind: "mesh", .. }
                    | inf_asset::AssetError::Decode(_)
            ),
            "expected a named refusal, got {err:?}"
        );

        // …and the payload ships no fracture for it, rather than a wrong one.
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Empty, "Wall", None);
        {
            let world = doc.world_mut();
            let id = world.entity_of(e).expect("the entity exists");
            world.world_mut().entity_mut(id).insert((
                inf_ecs::components::Destructible {
                    chunk_count: 6,
                    ..Default::default()
                },
                inf_ecs::components::MeshRef {
                    asset: Some(MESH),
                    ..Default::default()
                },
            ));
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
            |g| (g == MESH).then(|| stale.clone()),
            // P26.3b: the fourth-kind byte resolver (cloth / hair / material /
            // texture). `None` here is the honest fixture default — this arm authors
            // none of the four.
            |_| None,
            60,
            false,
        )
        .expect("the payload still builds — one stale mesh must not stop Play");
        assert!(
            payload.fractures.is_empty(),
            "a mesh that did not decode must contribute no fracture"
        );
        // The control: the SAME fixture with a readable mesh does ship one, so the
        // emptiness above is the refusal and not a broken fixture.
        let good = inf_asset::encode(&fracture_cube()).expect("encodes");
        let payload = build_scene_payload(
            &doc,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            |g| (g == MESH).then(|| good.clone()),
            // P26.3b: the fourth-kind byte resolver (cloth / hair / material /
            // texture). `None` here is the honest fixture default — this arm authors
            // none of the four.
            |_| None,
            60,
            false,
        )
        .expect("payload builds");
        assert_eq!(payload.fractures.len(), 1);
    }

    /// **…and the warning itself is pinned** (P24.2 chore).
    ///
    /// The test above proves the payload ships no fracture for a stale mesh —
    /// which is exactly what it did *before* the adoption, silently. Deleting the
    /// `tracing::warn!` leaves that test green, so nothing in the repo stopped
    /// the advisory from being removed; the cook-advisory doctrine says a silent
    /// hazard gets an advisory, and an advisory nothing asserts is one edit from
    /// gone.
    ///
    /// Read off the module's own source (line endings normalized — a `.rs` is
    /// checked out CRLF under `core.autocrlf`, the P22 law). What is pinned is
    /// the *sentence*, not the formatting: the two clauses that make the message
    /// worth reading are "previews as INDESTRUCTIBLE" and the fact that the
    /// cooked build still fractures it, which together name the divergence.
    #[test]
    fn the_stale_mesh_advisory_is_pinned_to_what_it_says() {
        let whole = include_str!("pie.rs").replace("\r\n", "\n");
        // **Only the production half.** This test quotes the phrases it looks
        // for, so searching the whole file would find its own source and pass on
        // an empty module — the vacuity that makes a source gate worthless.
        let cut = whole
            .find("\n#[cfg(test)]\n")
            .expect("this module has a test section");
        let src = whole[..cut].to_string();
        // Search the source with the `\`-continuations collapsed, so re-wrapping
        // the literal is free and only the WORDS are load-bearing.
        let flat: String = {
            let mut out = String::with_capacity(src.len());
            let mut chars = src.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '\\' && chars.peek() == Some(&'\n') {
                    chars.next();
                    while chars.peek().is_some_and(|c| *c == ' ') {
                        chars.next();
                    }
                    continue;
                }
                out.push(c);
            }
            out
        };
        for phrase in [
            "did not decode, so this actor previews as INDESTRUCTIBLE",
            "while the cooked build still fractures it",
        ] {
            assert!(
                flat.contains(phrase),
                "the destructible-mesh advisory no longer says {phrase:?} — the \
                 stale-mesh test stays green without it, so this is the only \
                 thing keeping the warning honest"
            );
        }
        // Not vacuous: the phrases live inside a `tracing::warn!`, not a comment.
        let at = flat
            .find("did not decode, so this actor previews as INDESTRUCTIBLE")
            .expect("just asserted");
        let before = &flat[..at];
        assert!(
            before
                .rfind("tracing::warn!")
                .is_some_and(|w| before[w..].matches('\n').count() <= 2),
            "the advisory text is no longer inside a `tracing::warn!`"
        );
    }

    /// A 2 m cube, encoded — the mesh the fracture pin derives from.
    fn fracture_cube() -> inf_mesh::MeshAsset {
        let c = |x: f32, y: f32, z: f32| inf_mesh::MeshVertex {
            position: [x, y, z],
            ..Default::default()
        };
        inf_mesh::MeshAsset::new(
            vec![inf_mesh::SubMesh {
                name: "wall".into(),
                vertices: vec![
                    c(-1.0, -1.0, -1.0),
                    c(1.0, -1.0, -1.0),
                    c(1.0, 1.0, -1.0),
                    c(-1.0, 1.0, -1.0),
                    c(-1.0, -1.0, 1.0),
                    c(1.0, -1.0, 1.0),
                    c(1.0, 1.0, 1.0),
                    c(-1.0, 1.0, 1.0),
                ],
                indices: vec![
                    0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3,
                    7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
                ],
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["Concrete".into()],
        )
    }

    /// **THE SHARED RULE, both halves.** Two actors referencing one mesh with
    /// different parameters: the winner is decided in DOCUMENT order, not in ECS
    /// archetype order — which is what the P22.3 audit's M2 found diverging
    /// between the PIE payload and the editor's Simulate seeder.
    ///
    /// The `AlwaysLoaded` on the second actor is the reproduction, not decoration:
    /// it moves that entity into a different archetype, which is precisely what
    /// flipped the old `iter_entities()` walk's answer.
    #[test]
    fn two_actors_sharing_a_mesh_resolve_in_document_order() {
        const MESH: Uuid = Uuid::from_u128(0x2203_0F01);
        let mut doc = SceneDoc::new();
        let first = doc.create(SpawnKind::Empty, "First", None);
        let second = doc.create(SpawnKind::Empty, "Second", None);
        {
            let world = doc.world_mut();
            // **The components are inserted SECOND-ACTOR-FIRST on purpose.** An
            // archetype is created when its component set is first seen, and
            // `iter_entities()` walks archetypes in creation order — so giving
            // the document's *second* actor the *first* archetype is what makes
            // the two orders disagree. Without this the two walks happen to
            // coincide and the test certifies nothing (verified: the archetype
            // mutation passed against the naive fixture).
            let b = world.entity_of(second).expect("second");
            world.world_mut().entity_mut(b).insert((
                inf_ecs::components::Destructible {
                    chunk_count: 24,
                    ..inf_ecs::components::Destructible::default()
                },
                inf_ecs::components::MeshRef {
                    asset: Some(MESH),
                    ..Default::default()
                },
                inf_ecs::components::AlwaysLoaded,
            ));
            let a = world.entity_of(first).expect("first");
            world.world_mut().entity_mut(a).insert((
                inf_ecs::components::Destructible {
                    chunk_count: 8,
                    ..inf_ecs::components::Destructible::default()
                },
                inf_ecs::components::MeshRef {
                    asset: Some(MESH),
                    ..Default::default()
                },
            ));
            world.mark_dirty();
        }
        // The fixture is only a reproduction if the two walks really disagree.
        let archetype_first = doc
            .world()
            .world()
            .iter_entities()
            .find_map(|e| {
                e.get::<inf_ecs::components::Destructible>()
                    .map(|d| d.chunk_count)
            })
            .expect("a destructible");
        assert_eq!(
            archetype_first, 24,
            "the fixture no longer reproduces the archetype/document split — an \
             archetype walk visits the same actor first, so this test would pass \
             against the very bug it exists to catch"
        );
        let params = destructible_mesh_params(&doc);
        assert_eq!(params.len(), 1, "one mesh, one fracture");
        assert_eq!(
            params[&MESH].chunk_count, 8,
            "the FIRST actor in document order must win — an archetype-order walk \
             answers 24 here, and Simulate would shatter a wall the shipped pack \
             does not"
        );

        // …and the payload agrees with it, end to end.
        let payload = build_scene_payload(
            &doc,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            |_| None,
            |g| (g == MESH).then(|| inf_asset::encode(&fracture_cube()).expect("encodes")),
            // P26.3b: the fourth-kind byte resolver (cloth / hair / material /
            // texture). `None` here is the honest fixture default — this arm authors
            // none of the four.
            |_| None,
            60,
            false,
        )
        .expect("payload builds");
        let decoded =
            inf_asset::decode::<inf_mesh::FractureAsset>(&payload.fractures[0].1).expect("decodes");
        assert_eq!(decoded.requested_chunks, 8);
    }

    /// **M3: PIE applies the cook's refusal, and the refusal LOOP is what does
    /// it.**
    ///
    /// A mesh whose chunks cannot all become colliders is refused *whole* by
    /// `inf_packager::cook`, so it ships indestructible — and PIE must agree, or
    /// the preview breaks a wall the shipped build cannot.
    ///
    /// # The fixture is a 0.2 mm slab, and that is the point
    ///
    /// The first cut used three coplanar vertices, which never reached the check
    /// at all: `fracture_mesh` refused the mesh upstream (`cloud.len() < 4`), so
    /// deleting the entire collidability loop left the test green. A gate that
    /// dies before the thing it gates is a gate on nothing.
    ///
    /// A **thin slab** is the real case: it fractures perfectly happily into eight
    /// chunks, each with a volume five orders of magnitude ABOVE
    /// `MIN_HULL_VOLUME_M3`, and one of them is nonetheless near-coplanar enough
    /// that `parry` cannot build a solid hull from it. That is exactly the failure
    /// the producer gate exists for — a chunk that looks fine by volume and cannot
    /// become a collider — and it is reachable only through the loop.
    #[test]
    fn a_fracture_the_cook_would_refuse_is_refused_here_too() {
        const MESH: Uuid = Uuid::from_u128(0x2203_0F02);
        let params = inf_mesh::FractureParams {
            seed: 5,
            chunk_count: 8,
        };
        let slab = thin_slab(1.0e-4);

        // The fixture really does get past `fracture_mesh` — otherwise the refusal
        // below would be the mesh being rejected, not the chunks.
        let raw = inf_mesh::fracture_mesh(&slab, inf_asset::AssetId(MESH), params)
            .expect("the slab fractures");
        assert!(
            raw.chunks.len() >= 2,
            "the slab produced no chunks to check"
        );
        let (chunk, volume_m3) = inf_physics::d3::first_uncollidable_chunk(&raw)
            .expect("the slab must have an uncollidable chunk for this test to test anything");
        assert!(
            volume_m3 > 1.0e-9,
            "chunk {chunk} was refused for its VOLUME ({volume_m3} m³), not its \
             shape — the fixture is testing the wrong refusal"
        );

        assert!(
            derive_fracture(&slab, MESH, params).is_none(),
            "PIE derived a fracture the cook refuses — the preview would break a \
             wall the shipped build cannot"
        );
        // The positive control: a real solid IS derived, so the assertion above is
        // about the refusal and not about `derive_fracture` returning None always.
        assert!(derive_fracture(&fracture_cube(), MESH, params).is_some());
    }

    /// A 2 x 2 m plate of half-thickness `t` — thin enough that one of its chunks
    /// cannot bound a hull, thick enough that the fracture itself succeeds.
    fn thin_slab(t: f32) -> inf_mesh::MeshAsset {
        let c = |x: f32, y: f32, z: f32| inf_mesh::MeshVertex {
            position: [x, y, z],
            ..Default::default()
        };
        inf_mesh::MeshAsset::new(
            vec![inf_mesh::SubMesh {
                name: "slab".into(),
                vertices: vec![
                    c(-1.0, -t, -1.0),
                    c(1.0, -t, -1.0),
                    c(1.0, t, -1.0),
                    c(-1.0, t, -1.0),
                    c(-1.0, -t, 1.0),
                    c(1.0, -t, 1.0),
                    c(1.0, t, 1.0),
                    c(-1.0, t, 1.0),
                ],
                indices: vec![
                    0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3,
                    7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
                ],
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["Concrete".into()],
        )
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
            // P26.3b: the fourth-kind byte resolver (cloth / hair / material /
            // texture). `None` here is the honest fixture default — this arm authors
            // none of the four.
            |_| None,
            60,
            false,
        )
        .expect("payload builds");
        assert!(payload.voxels.is_empty());
        assert!(payload.terrains.is_empty());
    }
}
