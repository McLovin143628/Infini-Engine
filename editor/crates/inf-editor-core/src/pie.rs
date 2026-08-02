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

        // Player logs (and panic messages) line-buffered off stderr.
        let stderr_lines = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&stderr_lines);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                sink.lock().expect("stderr sink poisoned").push(line);
            }
        });

        let session = PieSession {
            child,
            stdin,
            events,
            stderr_lines,
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
        session.send(&EditorToPlayer::LoadScene(payload.clone()))?;
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

    /// Poll-wait for the player to exit (used after Stop / a crash).
    pub fn wait_exit(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                return Some(status);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
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
pub fn build_scene_payload<F, G, H, B>(
    doc: &SceneDoc,
    mut resolve: F,
    mut resolve_pcg: G,
    mut resolve_anim: H,
    mut resolve_biome_set: B,
    tick_hz: u32,
    windowed: bool,
) -> Result<ScenePayload, PieError>
where
    F: FnMut(Uuid) -> Option<BlueprintClass>,
    G: FnMut(Uuid) -> Option<Vec<u8>>,
    H: FnMut(Uuid) -> Option<Vec<u8>>,
    B: FnMut(Uuid) -> Option<Vec<u8>>,
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

    Ok(
        ScenePayload::new(doc.title(), level_bytes, classes, tick_hz, windowed)
            .with_pcgs(pcgs)
            .with_biome_sets(biome_sets)
            .with_anim_assets(skeletons, clips, machines),
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
