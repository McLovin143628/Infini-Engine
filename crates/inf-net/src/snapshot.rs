//! Transform-replication snapshot encoding.
//!
//! A [`NetSnapshot`] is a Guid-keyed set of entity transforms — the replicated
//! slice of world state. It is a **pure wire type**: it knows nothing about the
//! ECS (the `inf-runtime` glue reads/writes it against `EcsWorld`), which keeps
//! this crate free of a `bevy_ecs` dependency and keeps the encoding independently
//! testable.
//!
//! ## Delta-against-baseline
//!
//! Sending a full snapshot every tick wastes bandwidth on unchanged entities.
//! [`NetSnapshot::delta`] diffs a snapshot against the last one the peer
//! acknowledged (the *baseline*), emitting only changed/added transforms plus the
//! ids that were removed. [`NetSnapshot::apply_delta`] reconstructs the full
//! snapshot on the receive side. Entity ids are `u128` (a Guid's raw bytes), so
//! the encoding never names `uuid` on the wire and stays bincode-clean.
//!
//! Quantization is deliberately **off in v1** (transforms are sent as raw `f64`)
//! — correctness first; a quantized/compressed mode is a documented follow-up in
//! the net memo.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A replicated entity id: the raw 128 bits of the entity's Guid. Ordered, so
/// snapshots serialize deterministically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NetId(pub u128);

/// One entity's replicated local transform (translation / euler-degree rotation /
/// scale — mirrors `inf_ecs::Transform`). Raw `f64`, quantization off (v1).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformState {
    pub translation: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
}

impl TransformState {
    pub const IDENTITY: TransformState = TransformState {
        translation: [0.0; 3],
        rotation: [0.0; 3],
        scale: [1.0; 3],
    };

    /// Bit-exact equality (replication must not treat `-0.0`/`0.0` as different,
    /// and must treat any real change as changed — so we compare bit patterns).
    fn bit_eq(&self, other: &TransformState) -> bool {
        let a = |v: &[f64; 3]| [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
        a(&self.translation) == a(&other.translation)
            && a(&self.rotation) == a(&other.rotation)
            && a(&self.scale) == a(&other.scale)
    }
}

/// A full snapshot: every replicated entity's transform at a given sim tick.
/// `BTreeMap` keeps it order-deterministic (byte-identical encoding for identical
/// state — the engine law).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct NetSnapshot {
    pub tick: u64,
    pub transforms: BTreeMap<NetId, TransformState>,
}

/// The compact diff of one snapshot against a baseline.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SnapshotDelta {
    pub tick: u64,
    /// Entities added or changed since the baseline.
    pub changed: Vec<(NetId, TransformState)>,
    /// Entities present in the baseline but gone now.
    pub removed: Vec<NetId>,
}

impl NetSnapshot {
    pub fn new(tick: u64) -> Self {
        Self {
            tick,
            transforms: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, id: NetId, t: TransformState) {
        self.transforms.insert(id, t);
    }

    /// Full encode (used for the first/baseline snapshot).
    pub fn encode(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("a NetSnapshot is always encodable")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, crate::NetError> {
        let (snap, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(snap)
    }

    /// Diff `self` against `baseline`, emitting only changed/added transforms and
    /// removed ids.
    pub fn delta(&self, baseline: &NetSnapshot) -> SnapshotDelta {
        let mut changed = Vec::new();
        for (id, t) in &self.transforms {
            match baseline.transforms.get(id) {
                Some(prev) if prev.bit_eq(t) => {}
                _ => changed.push((*id, *t)),
            }
        }
        let removed = baseline
            .transforms
            .keys()
            .filter(|id| !self.transforms.contains_key(id))
            .copied()
            .collect();
        SnapshotDelta {
            tick: self.tick,
            changed,
            removed,
        }
    }

    /// Reconstruct a full snapshot by applying `delta` on top of `self`
    /// (`self` = the baseline). Returns the new snapshot.
    pub fn apply_delta(&self, delta: &SnapshotDelta) -> NetSnapshot {
        let mut out = self.clone();
        out.tick = delta.tick;
        for (id, t) in &delta.changed {
            out.transforms.insert(*id, *t);
        }
        for id in &delta.removed {
            out.transforms.remove(id);
        }
        out
    }
}

impl SnapshotDelta {
    pub fn encode(&self) -> Vec<u8> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .expect("a SnapshotDelta is always encodable")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, crate::NetError> {
        let (delta, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(x: f64) -> TransformState {
        TransformState {
            translation: [x, 0.0, 0.0],
            rotation: [0.0, x * 2.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn full_round_trip() {
        let mut s = NetSnapshot::new(3);
        s.insert(NetId(1), t(1.0));
        s.insert(NetId(2), t(2.0));
        let bytes = s.encode();
        assert_eq!(NetSnapshot::decode(&bytes).unwrap(), s);
    }

    #[test]
    fn encoding_is_deterministic() {
        let mut a = NetSnapshot::new(3);
        let mut b = NetSnapshot::new(3);
        // Insert in different orders; BTreeMap normalizes.
        a.insert(NetId(2), t(2.0));
        a.insert(NetId(1), t(1.0));
        b.insert(NetId(1), t(1.0));
        b.insert(NetId(2), t(2.0));
        assert_eq!(a.encode(), b.encode());
    }

    #[test]
    fn delta_only_carries_changes() {
        let mut base = NetSnapshot::new(1);
        base.insert(NetId(1), t(1.0));
        base.insert(NetId(2), t(2.0));
        base.insert(NetId(3), t(3.0));

        let mut next = NetSnapshot::new(2);
        next.insert(NetId(1), t(1.0)); // unchanged
        next.insert(NetId(2), t(9.0)); // changed
        next.insert(NetId(4), t(4.0)); // added
                                       // NetId(3) removed

        let d = next.delta(&base);
        assert_eq!(d.tick, 2);
        let mut changed: Vec<_> = d.changed.iter().map(|(id, _)| id.0).collect();
        changed.sort();
        assert_eq!(changed, vec![2, 4]);
        assert_eq!(d.removed, vec![NetId(3)]);

        // Applying the delta on the baseline reproduces the full next snapshot.
        assert_eq!(base.apply_delta(&d), next);
    }

    #[test]
    fn delta_of_identical_is_empty() {
        let mut base = NetSnapshot::new(1);
        base.insert(NetId(1), t(1.0));
        let mut same = NetSnapshot::new(2);
        same.insert(NetId(1), t(1.0));
        let d = same.delta(&base);
        assert!(d.changed.is_empty());
        assert!(d.removed.is_empty());
    }
}
