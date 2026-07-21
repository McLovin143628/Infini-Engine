//! The PIE (play-in-editor) wire protocol between the editor and an
//! `inf-player` subprocess.
//!
//! Transport: the player's **stdin/stdout pipes** — the "local channel" of
//! the roadmap. In `--pie` mode stdout carries *only* protocol frames
//! (human logs go to stderr). Subprocess-first is the whole spike: a
//! panicking script kills the player process, never the editor.
//!
//! # Frame framing (versioned, little-endian) — P9.4
//!
//! Each frame is a small **versioned little-endian header** followed by a
//! bincode payload:
//!
//! ```text
//! ┌────────────┬──────────────┬────────────┬──────────────┐
//! │ magic u32  │ frame_ver u16 │ len u32     │ payload[len]  │
//! │ 0x0050_4945│ PIE_FRAME_VER │ (bincode)   │  bincode      │
//! └────────────┴──────────────┴────────────┴──────────────┘
//! ```
//!
//! The magic + version self-describe the stream so a desynced or mismatched
//! peer fails cleanly (a wrong first byte is an error, not garbage bincode).
//! A cleanly closed pipe (peer exit) surfaces as `UnexpectedEof` on the first
//! header read, which the reader loops treat as end-of-stream.
//!
//! # Content handoff (P9.4)
//!
//! Spike D handed over a toy [`CookedSnapshot`]. P9.4 adds [`ScenePayload`]:
//! the editor streams the **real** live scene as v3 `.inf_lvl` bytes plus the
//! bound blueprint classes as `(guid, json)`, and the player builds the world
//! exactly like the shipping pack path (`InfSceneWorldBuilder::with_bindings`)
//! — so previewing never diverges from shipping.

use std::io::{self, Read, Write};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::sim::ActorState;
use crate::snapshot::CookedSnapshot;

/// Protocol version, checked at handshake. Bumped to 2 in P9.4 (the real
/// [`ScenePayload`] content handoff + Step/Eject/SetViewport control frames +
/// Window/State reports on top of the Spike D v1 surface).
pub const PIE_PROTOCOL_VERSION: u32 = 2;

/// Frame-header magic (little-endian `b"PIE\0"`).
pub const PIE_FRAME_MAGIC: u32 = 0x0050_4945;

/// Frame-header version (the framing itself, distinct from the protocol
/// version carried in the `Ready` handshake).
pub const PIE_FRAME_VERSION: u16 = 1;

/// Schema version of a [`ScenePayload`] (its own migratable envelope; the
/// `.inf_lvl` bytes inside carry the scene schema version independently).
///
/// * v1 — level bytes + bound blueprint classes.
/// * v2 — appended `pcgs`: the `.inf_pcg` graph payloads a v4 level's
///   [`PcgVolume`]s reference, so the PIE player evaluates scatter exactly like
///   the shipping pack path.
pub const SCENE_PAYLOAD_VERSION: u32 = 2;

/// Upper bound on a single frame; anything larger means a desynced or
/// corrupt stream and is treated as an error rather than an allocation. A
/// live scene payload (level bytes + blueprint JSON) fits comfortably.
pub const MAX_FRAME_LEN: usize = 256 * 1024 * 1024;

/// The real content the editor streams to the player: the live scene as v3
/// `.inf_lvl` bytes plus the set of bound blueprint classes. The player builds
/// its world from these exactly like the cooked-pack path, so PIE == shipping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePayload {
    /// Envelope schema version ([`SCENE_PAYLOAD_VERSION`]).
    pub schema_version: u32,
    /// Human label for logs / the window title.
    pub label: String,
    /// The v3 `.inf_lvl` bincode payload of the *live* (unsaved-included) doc.
    pub level_bytes: Vec<u8>,
    /// Bound blueprint classes: `(asset guid, .inf_act JSON bytes)`. The guid
    /// keys the level's persisted `ActorClass` bindings.
    pub classes: Vec<(Uuid, Vec<u8>)>,
    /// Referenced PCG graphs: `(asset guid, .inf_pcg bincode bytes)`. The guid
    /// keys a v4 level's `PcgVolume.graph` refs; the player evaluates scatter from
    /// these on load (schema v2). `#[serde(default)]` so a v1 payload decodes.
    #[serde(default)]
    pub pcgs: Vec<(Uuid, Vec<u8>)>,
    /// Fixed update rate (Hz) the player ticks at.
    pub tick_hz: u32,
    /// Open a real window (`true`, the embedded / new-window PIE path) vs run
    /// headless + step-driven (`false`, the CI / determinism path).
    pub windowed: bool,
}

impl ScenePayload {
    /// Build a payload from encoded scene bytes + bound classes.
    pub fn new(
        label: impl Into<String>,
        level_bytes: Vec<u8>,
        classes: Vec<(Uuid, Vec<u8>)>,
        tick_hz: u32,
        windowed: bool,
    ) -> Self {
        Self {
            schema_version: SCENE_PAYLOAD_VERSION,
            label: label.into(),
            level_bytes,
            classes,
            pcgs: Vec::new(),
            tick_hz,
            windowed,
        }
    }

    /// Attach the referenced `.inf_pcg` graph payloads (`(asset guid, bytes)`).
    /// Builder-style so [`Self::new`]'s signature stays stable.
    pub fn with_pcgs(mut self, pcgs: Vec<(Uuid, Vec<u8>)>) -> Self {
        self.pcgs = pcgs;
        self
    }
}

/// A viewport rectangle forwarded to an embedded player (physical pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewportRectMsg {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditorToPlayer {
    /// Hand over the Spike-D toy snapshot (kept for the determinism smoke +
    /// the crash/pause/reparent drills that don't need real content).
    Load(CookedSnapshot),
    /// Hand over the **real** live scene (P9.4). Sent once after `Ready`.
    LoadScene(ScenePayload),
    Pause,
    Resume,
    /// Advance exactly `count` fixed steps (works while paused) — the
    /// deterministic step control the PIE==shipping test drives.
    Step {
        count: u32,
    },
    /// Graceful shutdown; the player answers `Stopped` and exits 0.
    Stop,
    /// Release input possession back to the editor (v1: stops the player
    /// possessing input; camera possession is a documented follow-up). The
    /// player keeps running until `Stop`.
    Eject,
    /// Forward a viewport rect change to an embedded player.
    SetViewport(ViewportRectMsg),
    /// Test/QA hook: make the player panic on its main loop — the
    /// crash-isolation drill.
    InjectPanic,
}

/// A running/paused status report the player pushes to the editor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlayerState {
    pub running: bool,
    pub paused: bool,
    pub frame: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlayerToEditor {
    /// First message after spawn; the editor refuses a version mismatch.
    Ready {
        protocol: u32,
    },
    Loaded {
        level: String,
        actor_count: usize,
    },
    /// The player created its window; `handle` is the native handle (HWND as
    /// `isize` on Windows, `0` where not applicable) the editor reparents into
    /// the viewport slot for embedded PIE.
    Window {
        handle: i64,
    },
    Frame {
        frame: u64,
        state_hash: u64,
        actors: Vec<ActorState>,
    },
    /// A running/paused status report (frame count, last error).
    State(PlayerState),
    Paused,
    Resumed,
    Stopped,
    Ejected,
    /// A recoverable error the editor surfaces (a fatal one is a process exit).
    Error {
        message: String,
    },
}

/// Write one length-prefixed, versioned bincode frame.
pub fn write_msg<T: Serialize>(writer: &mut impl Write, msg: &T) -> io::Result<()> {
    let bytes = bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    writer.write_all(&PIE_FRAME_MAGIC.to_le_bytes())?;
    writer.write_all(&PIE_FRAME_VERSION.to_le_bytes())?;
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}

/// Read one versioned bincode frame. `Err(UnexpectedEof)` on a cleanly closed
/// pipe (the peer exited) — surfaced from the very first header read so reader
/// loops end cleanly.
pub fn read_msg<T: DeserializeOwned>(reader: &mut impl Read) -> io::Result<T> {
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    if u32::from_le_bytes(magic_bytes) != PIE_FRAME_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad PIE frame magic (desynced stream)",
        ));
    }
    let mut ver_bytes = [0u8; 2];
    reader.read_exact(&mut ver_bytes)?;
    let ver = u16::from_le_bytes(ver_bytes);
    if ver != PIE_FRAME_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("PIE frame version {ver} unsupported (host speaks {PIE_FRAME_VERSION})"),
        ));
    }
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds MAX_FRAME_LEN"),
        ));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let (msg, _): (T, usize) = bincode::serde::decode_from_slice(&buf, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_round_trip() {
        let messages = vec![
            EditorToPlayer::Load(CookedSnapshot::demo()),
            EditorToPlayer::LoadScene(ScenePayload::new(
                "level",
                vec![1, 2, 3, 4],
                vec![(Uuid::from_u128(0xAC), vec![b'{', b'}'])],
                60,
                false,
            )),
            EditorToPlayer::Pause,
            EditorToPlayer::Resume,
            EditorToPlayer::Step { count: 3 },
            EditorToPlayer::Eject,
            EditorToPlayer::SetViewport(ViewportRectMsg {
                x: 1,
                y: 2,
                width: 320,
                height: 240,
            }),
            EditorToPlayer::Stop,
            EditorToPlayer::InjectPanic,
        ];
        let mut wire = Vec::new();
        for msg in &messages {
            write_msg(&mut wire, msg).unwrap();
        }
        let mut cursor = std::io::Cursor::new(wire);
        for expected in &messages {
            let got: EditorToPlayer = read_msg(&mut cursor).unwrap();
            assert_eq!(&got, expected);
        }
        // Stream exhausted → clean EOF on the first header read.
        let eof = read_msg::<EditorToPlayer>(&mut cursor).unwrap_err();
        assert_eq!(eof.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn player_to_editor_frames_round_trip() {
        let messages = vec![
            PlayerToEditor::Ready {
                protocol: PIE_PROTOCOL_VERSION,
            },
            PlayerToEditor::Loaded {
                level: "L".into(),
                actor_count: 2,
            },
            PlayerToEditor::Window { handle: 0x1234 },
            PlayerToEditor::State(PlayerState {
                running: true,
                paused: false,
                frame: 7,
                last_error: None,
            }),
            PlayerToEditor::Ejected,
            PlayerToEditor::Error {
                message: "oops".into(),
            },
        ];
        let mut wire = Vec::new();
        for msg in &messages {
            write_msg(&mut wire, msg).unwrap();
        }
        let mut cursor = std::io::Cursor::new(wire);
        for expected in &messages {
            let got: PlayerToEditor = read_msg(&mut cursor).unwrap();
            assert_eq!(&got, expected);
        }
    }

    #[test]
    fn oversized_frame_is_rejected_without_allocating() {
        let mut wire = Vec::new();
        wire.extend_from_slice(&PIE_FRAME_MAGIC.to_le_bytes());
        wire.extend_from_slice(&PIE_FRAME_VERSION.to_le_bytes());
        wire.extend_from_slice(&(u32::MAX).to_le_bytes());
        let err = read_msg::<EditorToPlayer>(&mut std::io::Cursor::new(wire)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut wire = Vec::new();
        wire.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let err = read_msg::<EditorToPlayer>(&mut std::io::Cursor::new(wire)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
