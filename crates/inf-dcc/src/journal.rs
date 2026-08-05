//! The op journal: undo/redo, crash-survivable persistence, and the crate's
//! determinism gate — all the same mechanism.
//!
//! # Why not `GraphJournal`'s whole-snapshot design
//!
//! `inf_graph::GraphJournal` stores a complete `Graph` per edit group and undo is
//! "restore the previous one". That is right for a node graph — a few hundred
//! nodes of `BTreeMap` + `Vec` — and wrong for a mesh, where a bevel on a
//! 100k-vertex model would snapshot megabytes per click.
//!
//! So a [`MeshSession`] stores a **base mesh plus a `Vec<Op>`**, and undo is
//! *replay from the nearest checkpoint*. Three things fall out at once:
//!
//! * **Bounded memory** — ops are tens of bytes, checkpoints are capped.
//! * **A property-test harness for free** — "replay is a pure function of the
//!   ops" is a statement a generator can hammer, and it is the statement that
//!   catches an op which mutates without journalling.
//! * **Persistence** — [`SessionSave`] is the whole session as data, in bincode
//!   *and* in a self-describing format (architecture rule 4). No struct here uses
//!   `skip_serializing_if`: bincode is positional, so a skipped field desyncs the
//!   stream (the P10 law, caught three times).
//!
//! # Checkpoint cadence, and its honest worst case
//!
//! A full mesh snapshot is taken every [`CHECKPOINT_INTERVAL`] ops, and at most
//! [`MAX_CHECKPOINTS`] are retained — the ones **nearest the cursor**, because
//! that is where the next history move will be. So:
//!
//! * memory is `O(MAX_CHECKPOINTS × |mesh|)`, not `O(ops × |mesh|)`;
//! * a single undo replays at most `CHECKPOINT_INTERVAL − 1` ops;
//! * an undo that lands on a checkpoint boundary **stores the mesh it just
//!   computed**, so walking backwards through a long session stays cheap instead
//!   of re-replaying from the base every step;
//! * the worst case is real and worth stating: undoing to a point more than
//!   `MAX_CHECKPOINTS × CHECKPOINT_INTERVAL` ops behind, in one jump, after the
//!   nearby checkpoints have been evicted, replays from the base. That is
//!   `O(session)` once, and the step after it is cheap again.
//!
//! # `meshopt` is not reachable from here
//!
//! No [`Op`] routes through it, so replaying a journal on another machine
//! produces the same mesh. That is the P18 law's consequence, and the reason the
//! optimize pass lives behind a flag on [`crate::export`] instead.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ops::{self, Op, OpError, OpOutcome};
use crate::topo::Mesh;

/// A full mesh snapshot is stored every this many ops.
pub const CHECKPOINT_INTERVAL: usize = 32;
/// At most this many snapshots are retained, nearest the cursor first.
pub const MAX_CHECKPOINTS: usize = 8;

/// A whole edit session as data: the mesh it started from, every op applied, and
/// where the cursor sits. Checkpoints are **not** persisted — they are derived,
/// and re-deriving them is exactly what `restore` does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSave {
    pub base: Mesh,
    pub ops: Vec<Op>,
    /// How many of `ops` are applied. `ops[cursor..]` is the redo tail.
    pub cursor: usize,
}

/// A mesh plus its op journal.
#[derive(Debug, Clone)]
pub struct MeshSession {
    base: Mesh,
    ops: Vec<Op>,
    cursor: usize,
    current: Mesh,
    checkpoints: BTreeMap<usize, Mesh>,
    generation: u64,
}

impl MeshSession {
    /// Start a session from a base mesh.
    pub fn new(base: Mesh) -> Self {
        Self {
            current: base.clone(),
            base,
            ops: Vec::new(),
            cursor: 0,
            checkpoints: BTreeMap::new(),
            generation: 0,
        }
    }

    /// The mesh as it stands.
    pub fn mesh(&self) -> &Mesh {
        &self.current
    }
    /// The mesh the journal starts from.
    pub fn base(&self) -> &Mesh {
        &self.base
    }
    /// The whole timeline, including the redo tail past [`MeshSession::cursor`].
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }
    /// How many ops are applied.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// A monotone stamp bumped by every successful mutation and every history
    /// move.
    ///
    /// It exists because of the id-reuse rule ([`crate::topo`]): a structural op
    /// rebuilds a local patch, so half-edge and face ids that were valid a moment
    /// ago may now name something else — and they will not be *dead*, so nothing
    /// would catch a stale one. **A consumer caching ids (P23.4's selection) must
    /// discard the cache when this changes.**
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// How many snapshots are currently held — an implementation detail the
    /// bounded-memory test asserts on.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    pub fn can_undo(&self) -> bool {
        self.cursor > 0
    }
    pub fn can_redo(&self) -> bool {
        self.cursor < self.ops.len()
    }

    /// Apply an op and record it.
    ///
    /// A refusal is a value: the mesh is untouched (byte-identical) and
    /// **nothing is journalled**, so a rejected edit cannot appear in a replay.
    /// A successful op forks history — the redo tail is discarded.
    pub fn apply(&mut self, op: Op) -> Result<OpOutcome, OpError> {
        let outcome = ops::apply(&mut self.current, &op)?;
        self.ops.truncate(self.cursor);
        self.checkpoints.retain(|&k, _| k <= self.cursor);
        self.ops.push(op);
        self.cursor += 1;
        self.generation += 1;
        if self.cursor.is_multiple_of(CHECKPOINT_INTERVAL) {
            self.checkpoints.insert(self.cursor, self.current.clone());
        }
        self.evict();
        Ok(outcome)
    }

    /// Step back one op. `false` when there is nothing to undo.
    pub fn undo(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.seek(self.cursor - 1);
        true
    }

    /// Step forward one op. `false` when there is nothing to redo.
    pub fn redo(&mut self) -> bool {
        if !self.can_redo() {
            return false;
        }
        let op = self.ops[self.cursor].clone();
        // A journalled op cannot refuse on replay — the mesh it is replayed onto
        // is the one it was recorded against, byte for byte. `restore` proves the
        // same for a session read off disk, which is the only way an op sequence
        // this type did not produce itself can get in.
        ops::apply(&mut self.current, &op).expect("a journalled op replays");
        self.cursor += 1;
        self.generation += 1;
        if self.cursor.is_multiple_of(CHECKPOINT_INTERVAL) {
            self.checkpoints.insert(self.cursor, self.current.clone());
        }
        self.evict();
        true
    }

    /// Move the cursor to `target`, rebuilding the mesh from the nearest
    /// checkpoint at or before it.
    fn seek(&mut self, target: usize) {
        let (from, mut mesh) = match self.checkpoints.range(..=target).next_back() {
            Some((&k, m)) => (k, m.clone()),
            None => (0, self.base.clone()),
        };
        for op in &self.ops[from..target] {
            ops::apply(&mut mesh, op).expect("a journalled op replays");
        }
        self.current = mesh;
        self.cursor = target;
        self.generation += 1;
        if target > 0 && target.is_multiple_of(CHECKPOINT_INTERVAL) {
            self.checkpoints.insert(target, self.current.clone());
        }
        self.evict();
    }

    /// Retain the snapshots nearest the cursor; drop the farthest, breaking ties
    /// toward the older one.
    fn evict(&mut self) {
        while self.checkpoints.len() > MAX_CHECKPOINTS {
            let cursor = self.cursor;
            let victim = self
                .checkpoints
                .keys()
                .copied()
                .max_by_key(|&k| (k.abs_diff(cursor), std::cmp::Reverse(k)))
                .expect("non-empty while over the cap");
            self.checkpoints.remove(&victim);
        }
    }

    /// Replay an op sequence onto a base mesh. **The definition of the journal**:
    /// `replay(base, &session.ops()[..session.cursor()])` is byte-identical to
    /// `session.mesh()`, and that is property-tested.
    pub fn replay(base: &Mesh, ops: &[Op]) -> Result<Mesh, OpError> {
        let mut mesh = base.clone();
        for op in ops {
            ops::apply(&mut mesh, op)?;
        }
        Ok(mesh)
    }

    /// The session as persistable data.
    pub fn save(&self) -> SessionSave {
        SessionSave {
            base: self.base.clone(),
            ops: self.ops.clone(),
            cursor: self.cursor,
        }
    }

    /// Rebuild a session from a save.
    ///
    /// Every op is replayed — the applied prefix to produce the live mesh, and
    /// the redo tail to prove it *can* be redone. That is what makes
    /// [`MeshSession::redo`] and [`MeshSession::undo`] infallible afterwards: a
    /// save is the only way an op sequence this type did not itself produce can
    /// enter, so it is the only place the check is needed.
    pub fn restore(save: SessionSave) -> Result<Self, OpError> {
        let cursor = save.cursor.min(save.ops.len());
        let current = Self::replay(&save.base, &save.ops[..cursor])?;
        // Prove the tail too, on a throwaway, before promising redo cannot fail.
        Self::replay(&current, &save.ops[cursor..])?;
        let mut session = Self {
            base: save.base,
            ops: save.ops,
            cursor,
            current,
            checkpoints: BTreeMap::new(),
            generation: 0,
        };
        session.rebuild_checkpoints();
        Ok(session)
    }

    /// Re-derive the checkpoints around the cursor after a restore.
    fn rebuild_checkpoints(&mut self) {
        let mut mesh = self.base.clone();
        for (i, op) in self.ops[..self.cursor].iter().enumerate() {
            ops::apply(&mut mesh, op).expect("the prefix was replayed by `restore`");
            let count = i + 1;
            if count.is_multiple_of(CHECKPOINT_INTERVAL) {
                self.checkpoints.insert(count, mesh.clone());
            }
        }
        self.evict();
    }
}

impl PartialEq for MeshSession {
    /// Sessions compare by what they *are* — base, ops and cursor — not by which
    /// checkpoints happen to be cached. Used by tests; `restore` of a `save` is
    /// equal to the session it came from.
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base && self.ops == other.ops && self.cursor == other.cursor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, plane};
    use crate::topo::{CornerData, HalfId, VertId};
    use crate::validate::validate;

    /// A deterministic, mildly adversarial edit script over a cube.
    fn script(session: &mut MeshSession) {
        for round in 0..20u32 {
            let halfs: Vec<HalfId> = session.mesh().half_ids().collect();
            let h = halfs[(round as usize * 7) % halfs.len()];
            let _ = session.apply(Op::SplitEdge {
                half: h,
                t: 0.25 + 0.5 * (round % 3) as f64 / 3.0,
            });
            let verts: Vec<VertId> = session.mesh().vert_ids().collect();
            let _ = session.apply(Op::TranslateVerts {
                verts: vec![verts[(round as usize * 3) % verts.len()]],
                delta: [0.01, -0.02, 0.03],
            });
            let corners: Vec<HalfId> = session
                .mesh()
                .half_ids()
                .filter(|&h| session.mesh().is_boundary(h) == Some(false))
                .collect();
            let _ = session.apply(Op::SetCornerUv {
                half: corners[(round as usize * 5) % corners.len()],
                uv: [round as f64 * 0.01, 0.5],
            });
        }
    }

    #[test]
    fn replay_reproduces_the_session_byte_for_byte() {
        let mut s = MeshSession::new(cube(1.0));
        script(&mut s);
        assert!(s.cursor() >= 60, "the script must actually do work");
        let replayed = MeshSession::replay(s.base(), &s.ops()[..s.cursor()]).unwrap();
        assert_eq!(replayed.encoded(), s.mesh().encoded());
        assert_eq!(validate(&replayed), Ok(()));
    }

    #[test]
    fn two_runs_of_the_same_script_are_byte_identical() {
        let mut a = MeshSession::new(cube(1.0));
        let mut b = MeshSession::new(cube(1.0));
        script(&mut a);
        script(&mut b);
        assert_eq!(a.mesh().encoded(), b.mesh().encoded());
        assert_eq!(a.ops(), b.ops());
    }

    #[test]
    fn undo_then_redo_returns_the_same_bytes() {
        let mut s = MeshSession::new(cube(1.0));
        script(&mut s);
        let after = s.mesh().encoded();
        let steps = s.cursor();
        for _ in 0..steps {
            assert!(s.undo());
        }
        assert!(!s.undo());
        assert_eq!(s.mesh().encoded(), cube(1.0).encoded(), "back to the base");
        for _ in 0..steps {
            assert!(s.redo());
        }
        assert!(!s.redo());
        assert_eq!(s.mesh().encoded(), after);
    }

    #[test]
    fn every_intermediate_state_of_an_undo_walk_is_valid() {
        let mut s = MeshSession::new(cube(1.0));
        script(&mut s);
        while s.undo() {
            assert_eq!(validate(s.mesh()), Ok(()), "at cursor {}", s.cursor());
        }
    }

    #[test]
    fn a_refused_op_is_not_journalled_and_does_not_move_the_mesh() {
        let mut s = MeshSession::new(plane(2.0));
        let before = s.mesh().encoded();
        let err = s
            .apply(Op::SplitEdge {
                half: HalfId(9_999),
                t: 0.5,
            })
            .unwrap_err();
        assert_eq!(err, OpError::NoSuchHalf(HalfId(9_999)));
        assert_eq!(s.ops().len(), 0, "a refusal leaves no record");
        assert_eq!(s.cursor(), 0);
        assert_eq!(s.generation(), 0);
        assert_eq!(s.mesh().encoded(), before);
    }

    #[test]
    fn a_new_op_forks_history_and_discards_the_redo_tail() {
        let mut s = MeshSession::new(plane(2.0));
        let h = s
            .mesh()
            .half_ids()
            .find(|&h| s.mesh().is_boundary(h) == Some(false))
            .unwrap();
        s.apply(Op::SplitEdge { half: h, t: 0.5 }).unwrap();
        s.apply(Op::TranslateVerts {
            verts: s.mesh().vert_ids().collect(),
            delta: [1.0, 0.0, 0.0],
        })
        .unwrap();
        assert!(s.undo());
        assert!(s.can_redo());
        s.apply(Op::TranslateVerts {
            verts: vec![],
            delta: [0.0, 0.0, 0.0],
        })
        .unwrap();
        assert!(!s.can_redo(), "the tail is gone");
        assert_eq!(s.ops().len(), 2);
    }

    #[test]
    fn checkpoints_stay_bounded_over_a_long_session() {
        let mut s = MeshSession::new(plane(2.0));
        for i in 0..(CHECKPOINT_INTERVAL * MAX_CHECKPOINTS * 3) {
            s.apply(Op::TranslateVerts {
                verts: vec![],
                delta: [i as f64, 0.0, 0.0],
            })
            .unwrap();
        }
        assert!(
            s.checkpoint_count() <= MAX_CHECKPOINTS,
            "held {}",
            s.checkpoint_count()
        );
        assert!(s.checkpoint_count() > 0, "and it does checkpoint at all");
    }

    #[test]
    fn a_session_survives_both_serialization_formats() {
        let mut s = MeshSession::new(cube(1.0));
        script(&mut s);
        s.undo();
        s.undo();
        let save = s.save();

        let cfg = bincode::config::standard();
        let bin = bincode::serde::encode_to_vec(&save, cfg).unwrap();
        let (from_bin, _): (SessionSave, _) = bincode::serde::decode_from_slice(&bin, cfg).unwrap();
        assert_eq!(from_bin, save);
        assert_eq!(
            bincode::serde::encode_to_vec(&save, cfg).unwrap(),
            bin,
            "encoding is deterministic"
        );

        let json = serde_json::to_string(&save).unwrap();
        let from_json: SessionSave = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, save);

        let restored = MeshSession::restore(from_bin).unwrap();
        assert_eq!(restored.mesh().encoded(), s.mesh().encoded());
        assert_eq!(restored.cursor(), s.cursor());
        assert_eq!(restored.ops(), s.ops());
    }

    #[test]
    fn restore_refuses_a_save_whose_ops_do_not_apply() {
        let save = SessionSave {
            base: plane(2.0),
            ops: vec![Op::AddFace {
                verts: vec![VertId(0), VertId(1), VertId(400)],
                corners: vec![CornerData::default(); 3],
                slot: None,
            }],
            cursor: 1,
        };
        assert_eq!(
            MeshSession::restore(save),
            Err(OpError::NoSuchVert(VertId(400)))
        );
    }

    #[test]
    fn restore_also_proves_the_redo_tail() {
        // A save whose applied prefix is fine but whose tail is not: `restore`
        // must refuse it rather than hand back a session whose `redo` panics.
        let mut s = MeshSession::new(plane(2.0));
        s.apply(Op::AddVertex {
            position: [9.0, 0.0, 0.0],
        })
        .unwrap();
        let mut save = s.save();
        save.ops.push(Op::RemoveVertex {
            vert: VertId(4_000),
        });
        assert_eq!(
            MeshSession::restore(save),
            Err(OpError::NoSuchVert(VertId(4_000)))
        );
    }

    #[test]
    fn undoing_across_a_checkpoint_boundary_lands_on_the_same_bytes() {
        let mut s = MeshSession::new(plane(2.0));
        let mut snapshots = Vec::new();
        for i in 0..(CHECKPOINT_INTERVAL * 3) {
            s.apply(Op::AddVertex {
                position: [i as f64, 0.0, 0.0],
            })
            .unwrap();
            snapshots.push(s.mesh().encoded());
        }
        for i in (0..snapshots.len()).rev() {
            assert_eq!(s.mesh().encoded(), snapshots[i], "at cursor {}", s.cursor());
            s.undo();
        }
    }
}
