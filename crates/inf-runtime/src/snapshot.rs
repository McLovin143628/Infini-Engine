//! The cooked snapshot: what the editor (or `inf-packager`, later) hands a
//! player process to run. Spike D keeps it to a level name and actor spawn
//! list; the point is that PIE exercises the *same* cooked-bytes path a
//! shipped game uses, so previewing never diverges from shipping.

use serde::{Deserialize, Serialize};

use crate::sim::Actor;

pub const SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("failed to encode snapshot: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    #[error("failed to decode snapshot: {0}")]
    Decode(#[from] bincode::error::DecodeError),
    #[error("snapshot schema version {found} unsupported (host speaks {SNAPSHOT_SCHEMA_VERSION})")]
    SchemaVersion { found: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CookedSnapshot {
    pub schema_version: u32,
    pub level: String,
    pub actors: Vec<Actor>,
}

impl CookedSnapshot {
    pub fn new(level: impl Into<String>, actors: Vec<Actor>) -> Self {
        CookedSnapshot {
            schema_version: SNAPSHOT_SCHEMA_VERSION,
            level: level.into(),
            actors,
        }
    }

    /// The fixed demo level used by `--headless` CI smoke runs and tests.
    pub fn demo() -> Self {
        CookedSnapshot::new(
            "demo",
            vec![
                Actor {
                    name: "alpha".into(),
                    pos: [0.0, 1.0, 0.0],
                    vel: [3.0, 0.25, -1.5],
                },
                Actor {
                    name: "beta".into(),
                    pos: [-40.0, 2.0, 10.0],
                    vel: [12.0, -0.5, 4.0],
                },
                Actor {
                    name: "gamma".into(),
                    pos: [99.5, 0.0, -99.5],
                    vel: [8.0, 0.0, -8.0],
                },
            ],
        )
    }

    pub fn encode(&self) -> Result<Vec<u8>, SnapshotError> {
        Ok(bincode::serde::encode_to_vec(
            self,
            bincode::config::standard(),
        )?)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, SnapshotError> {
        let (snapshot, _): (CookedSnapshot, usize) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION {
            return Err(SnapshotError::SchemaVersion {
                found: snapshot.schema_version,
            });
        }
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        let snapshot = CookedSnapshot::demo();
        let decoded = CookedSnapshot::decode(&snapshot.encode().unwrap()).unwrap();
        assert_eq!(snapshot, decoded);
    }

    #[test]
    fn wrong_schema_version_is_rejected() {
        let mut snapshot = CookedSnapshot::demo();
        snapshot.schema_version = 999;
        let bytes = bincode::serde::encode_to_vec(&snapshot, bincode::config::standard()).unwrap();
        assert!(matches!(
            CookedSnapshot::decode(&bytes),
            Err(SnapshotError::SchemaVersion { found: 999 })
        ));
    }
}
