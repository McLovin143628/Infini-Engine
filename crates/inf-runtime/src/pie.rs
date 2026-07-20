//! The PIE (play-in-editor) wire protocol between the editor and an
//! `inf-player` subprocess.
//!
//! Transport: the player's **stdin/stdout pipes** — the "local channel" of
//! the roadmap. In `--pie` mode stdout carries *only* protocol frames
//! (human logs go to stderr); each frame is a `u32` little-endian length
//! followed by that many bytes of bincode. Subprocess-first is the whole
//! spike: a panicking script kills the player process, never the editor.

use std::io::{self, Read, Write};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::sim::ActorState;
use crate::snapshot::CookedSnapshot;

pub const PIE_PROTOCOL_VERSION: u32 = 1;

/// Upper bound on a single frame; anything larger means a desynced or
/// corrupt stream and is treated as an error rather than an allocation.
pub const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditorToPlayer {
    /// Hand over the cooked level. Sent once right after spawn.
    Load(CookedSnapshot),
    Pause,
    Resume,
    /// Graceful shutdown; the player answers `Stopped` and exits 0.
    Stop,
    /// Test/QA hook: make the player panic on its main loop — the
    /// crash-isolation drill.
    InjectPanic,
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
    Frame {
        frame: u64,
        state_hash: u64,
        actors: Vec<ActorState>,
    },
    Paused,
    Resumed,
    Stopped,
}

/// Write one length-prefixed bincode frame.
pub fn write_msg<T: Serialize>(writer: &mut impl Write, msg: &T) -> io::Result<()> {
    let bytes = bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}

/// Read one length-prefixed bincode frame. `Err(UnexpectedEof)` on a cleanly
/// closed pipe (the peer exited).
pub fn read_msg<T: DeserializeOwned>(reader: &mut impl Read) -> io::Result<T> {
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
            EditorToPlayer::Pause,
            EditorToPlayer::Resume,
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
        // Stream exhausted → clean EOF.
        let eof = read_msg::<EditorToPlayer>(&mut cursor).unwrap_err();
        assert_eq!(eof.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn oversized_frame_is_rejected_without_allocating() {
        let mut wire = Vec::new();
        wire.extend_from_slice(&(u32::MAX).to_le_bytes());
        let err = read_msg::<EditorToPlayer>(&mut std::io::Cursor::new(wire)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
