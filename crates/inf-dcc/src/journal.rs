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
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::ops::{self, Op, OpError, OpOutcome};
use crate::topo::Mesh;
use crate::validate::{validate, Violation};

/// A full mesh snapshot is stored every this many ops.
pub const CHECKPOINT_INTERVAL: usize = 32;
/// At most this many snapshots are retained, nearest the cursor first.
pub const MAX_CHECKPOINTS: usize = 8;

/// A whole edit session as data: the version it was written under, the mesh it
/// started from, every op applied, and where the cursor sits. Checkpoints are
/// **not** persisted — they are derived, and re-deriving them is exactly what
/// [`MeshSession::restore`] does.
///
/// # Why `schema_version` is the FIRST field
///
/// bincode is **positional**: it writes fields in declaration order with no
/// names and no framing. A reader that expects a leading `u32` and gets a `Mesh`
/// does not fail, it *mis-parses*. So the version has to be the first thing on
/// the wire, or it cannot guard the thing after it — and it has to exist from
/// the very first release, because adding it later would make every already-
/// written save decode its old leading field (here, the `Mesh`'s vertex arena
/// slot count) as a version number.
///
/// This crate has written zero sessions to disk, which is precisely the moment
/// the ladder is free. `inf_mesh::MeshAsset` has had one since v1 and
/// [`crate::build::from_mesh_asset`] honours it; a reader that enforces a
/// version ladder on its *input* while writing its own *output* without one is
/// the asymmetry this field closes.
///
/// # And the enum discriminants underneath it
///
/// The version guards the *shape* of this struct. It does **not** guard
/// [`Op`]'s discriminants, because a mis-numbered variant produces a
/// structurally valid save of the wrong edit. That is pinned separately by the
/// frozen-discriminant test (the P19 wire-enum law): `Op` is **append-only**,
/// and any insertion or reorder must bump [`SessionSave::CURRENT_VERSION`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSave {
    /// Must be the first field — see the type docs.
    pub schema_version: u32,
    pub base: Mesh,
    pub ops: Vec<Op>,
    /// How many of `ops` are applied. `ops[cursor..]` is the redo tail.
    pub cursor: usize,
}

impl SessionSave {
    /// v1: `{schema_version, base, ops, cursor}` with the [`Op`] discriminants
    /// frozen by `op_discriminants_are_frozen`. Bump on any change to either.
    pub const CURRENT_VERSION: u32 = 1;
}

/// Why a [`SessionSave`] could not become a session.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SessionError {
    #[error("session schema v{found} is not v{current}; this build cannot read it")]
    UnsupportedSchema { found: u32, current: u32 },
    /// The base mesh is not a valid mesh. Replaying ops onto it would produce
    /// nonsense at best and, because the internal accessors assert the kernel's
    /// own invariants rather than validating input, a panic at worst.
    #[error("the save's base mesh is invalid: {} violation(s), first {:?}", .0.len(), .0.first())]
    InvalidBase(Vec<Violation>),
    #[error("replaying the save's ops failed: {0}")]
    Op(OpError),
    /// `cursor` points past the end of `ops`.
    ///
    /// Refused rather than clamped. A truncated write loses trailing ops and
    /// leaves a cursor that outruns them; clamping produces a *fully consistent*
    /// session that has silently lost work, and the next `save()` writes the
    /// shortened history back as if it were authoritative — the corruption
    /// laundered by the function documented as the trust boundary. The version
    /// and the base get a loud refusal; so does this.
    #[error("the save's cursor is {cursor} but it carries only {ops} ops")]
    CursorOutOfRange { cursor: usize, ops: usize },
    /// Replaying the save's ops onto its (valid) base produced an invalid mesh.
    ///
    /// If this fires, some op did not preserve the invariants — a bug in this
    /// crate, not in the save. It is still a refusal rather than a debug
    /// assertion, because a save is read by a *different build* than wrote it,
    /// and "the op set changed under an old journal" is exactly the situation
    /// where a silent, subtly-broken mesh is worst.
    #[error("replaying the save produced an invalid mesh: {} violation(s)", .0.len())]
    InvalidResult(Vec<Violation>),
}

/// A process-wide source of [`MeshSession::generation`] stamps.
///
/// Monotone across *every* session in the process, including ones built by
/// [`MeshSession::restore`]. Starting a restored session back at 0 would let a
/// consumer's cached ids from an earlier session compare equal to a completely
/// different document's — which is the exact stale-id hazard the stamp exists
/// to catch, reintroduced by the door meant to be safest.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

fn fresh_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

/// A mesh plus its op journal.
#[derive(Debug)]
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
    ///
    /// The base is **not** validated here: it comes from this process — a
    /// primitive, [`crate::build::from_mesh_asset`], or another session — and
    /// every one of those doors either constructs a valid mesh or refuses.
    /// [`MeshSession::restore`] is the trust boundary, and it does validate.
    pub fn new(base: Mesh) -> Self {
        Self {
            current: base.clone(),
            base,
            ops: Vec::new(),
            cursor: 0,
            checkpoints: BTreeMap::new(),
            generation: fresh_generation(),
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
    /// move — and **never repeated anywhere in the process**, including across
    /// [`MeshSession::restore`] and across different sessions.
    ///
    /// It exists because of the id-reuse rule ([`crate::topo`]): a structural op
    /// rebuilds a local patch, so half-edge and face ids that were valid a moment
    /// ago may now name something else — and they will not be *dead*, so nothing
    /// would catch a stale one. **A consumer caching ids (P23.4's selection) must
    /// discard the cache when this changes**, and must compare stamps rather than
    /// assume any particular starting value: a fresh session does not start at 0,
    /// precisely so a cache held across a restore cannot match by accident.
    ///
    /// A **refused** op does not bump it: nothing moved, so no cache is stale.
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
        self.generation = fresh_generation();
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
        self.generation = fresh_generation();
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
        self.generation = fresh_generation();
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

    /// The session as persistable data, stamped with the current schema version.
    pub fn save(&self) -> SessionSave {
        SessionSave {
            schema_version: SessionSave::CURRENT_VERSION,
            base: self.base.clone(),
            ops: self.ops.clone(),
            cursor: self.cursor,
        }
    }

    /// Rebuild a session from a save. **This is the crate's trust boundary for
    /// journals**, and it checks three things in this order:
    ///
    /// 1. **The version.** A mismatch is a loud refusal, not a best-effort read.
    /// 2. **The base mesh.** `validate` in full, because everything downstream
    ///    assumes a valid mesh: the internal accessors are *assertions of the
    ///    kernel's own invariants*, not input validation, so a base whose `twin`
    ///    names a dead slot does not refuse — it panics inside the first
    ///    structural op, in code whose own contract says a refusal is a value.
    ///    Validating here is what makes that contract true, and it is cheap
    ///    against the replay that follows.
    /// 3. **The cursor**, which is refused when it outruns `ops` rather than
    ///    clamped — a clamp turns a truncated file into a consistent session
    ///    that has quietly lost edits, and the next `save()` writes the loss
    ///    back as history.
    /// 4. **Every op** — the applied prefix to produce the live mesh, and the
    ///    redo tail on a throwaway to prove it *can* be redone. That is what
    ///    makes [`MeshSession::redo`] and [`MeshSession::undo`] infallible
    ///    afterwards.
    /// 5. **The replayed mesh**, because a valid base plus a replay is only a
    ///    valid mesh if every op in *this build* preserves the invariants.
    pub fn restore(save: SessionSave) -> Result<Self, SessionError> {
        if save.schema_version != SessionSave::CURRENT_VERSION {
            return Err(SessionError::UnsupportedSchema {
                found: save.schema_version,
                current: SessionSave::CURRENT_VERSION,
            });
        }
        validate(&save.base).map_err(SessionError::InvalidBase)?;
        if save.cursor > save.ops.len() {
            return Err(SessionError::CursorOutOfRange {
                cursor: save.cursor,
                ops: save.ops.len(),
            });
        }
        let cursor = save.cursor;
        let current = Self::replay(&save.base, &save.ops[..cursor]).map_err(SessionError::Op)?;
        // A valid base plus a replay is only a valid MESH if every op in *this
        // build* preserves the invariants — which is a property of the code doing
        // the replaying, not of the save. One more O(|mesh|) pass on a path that
        // just replayed the whole history.
        validate(&current).map_err(SessionError::InvalidResult)?;
        Self::replay(&current, &save.ops[cursor..]).map_err(SessionError::Op)?;
        let mut session = Self {
            base: save.base,
            ops: save.ops,
            cursor,
            current,
            checkpoints: BTreeMap::new(),
            generation: fresh_generation(),
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

impl Clone for MeshSession {
    /// Everything is copied except the generation stamp, which is **drawn
    /// fresh**.
    ///
    /// A derived `Clone` copies the stamp verbatim, and then two independent
    /// documents answer `generation()` with the same number — which is precisely
    /// the collision the stamp exists to prevent, manufactured by the one
    /// operation whose whole job is to produce a second document.
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            ops: self.ops.clone(),
            cursor: self.cursor,
            current: self.current.clone(),
            checkpoints: self.checkpoints.clone(),
            generation: fresh_generation(),
        }
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
    use crate::model::{KnifePoint, MergeTarget, MirrorAxis};
    use crate::topo::{CornerData, FaceId, HalfId, VertId};
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
        let gen_before = s.generation();
        let err = s
            .apply(Op::SplitEdge {
                half: HalfId(9_999),
                t: 0.5,
            })
            .unwrap_err();
        assert_eq!(err, OpError::NoSuchHalf(HalfId(9_999)));
        assert_eq!(s.ops().len(), 0, "a refusal leaves no record");
        assert_eq!(s.cursor(), 0);
        assert_eq!(
            s.generation(),
            gen_before,
            "nothing moved, so no consumer's id cache went stale"
        );
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
            schema_version: SessionSave::CURRENT_VERSION,
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
            Err(SessionError::Op(OpError::NoSuchVert(VertId(400))))
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
            Err(SessionError::Op(OpError::NoSuchVert(VertId(4_000))))
        );
    }

    // ── B2: the version ladder and the discriminant freeze ──────────────────

    /// The frozen wire position of every [`Op`] variant.
    ///
    /// This is a `match`, not a table, on purpose: **adding a variant stops this
    /// file compiling** until an author has consciously given it the next index.
    /// A table would silently accept an insertion, which is the whole defect —
    /// bincode writes an externally-tagged enum as a varint discriminant, so
    /// inserting a variant at index 5 turns every recorded `CollapseEdge` into
    /// whatever now sits at 5, structurally valid and semantically a different
    /// edit. Append only; anything else bumps `SessionSave::CURRENT_VERSION`.
    fn frozen_discriminant(op: &Op) -> u8 {
        match op {
            Op::AddVertex { .. } => 0,
            Op::RemoveVertex { .. } => 1,
            Op::AddFace { .. } => 2,
            Op::RemoveFace { .. } => 3,
            Op::SplitEdge { .. } => 4,
            Op::CollapseEdge { .. } => 5,
            Op::SplitFace { .. } => 6,
            Op::WeldVerts { .. } => 7,
            Op::TranslateVerts { .. } => 8,
            Op::SetCornerUv { .. } => 9,
            Op::SetCornerNormal { .. } => 10,
            Op::SetEdgeSharp { .. } => 11,
            Op::SetFaceSlot { .. } => 12,
            // P23.4 appended these NINE, in this order, at the next free indices.
            // Nothing above moved, so `CollapseEdge{7}` is still `[5, 7]` and every
            // session ever written still decodes as what it said.
            Op::ExtrudeFaces { .. } => 13,
            Op::ExtrudeEdges { .. } => 14,
            Op::InsetFaces { .. } => 15,
            Op::BevelEdges { .. } => 16,
            Op::LoopCut { .. } => 17,
            Op::Knife { .. } => 18,
            Op::MergeVerts { .. } => 19,
            Op::SubdivideFaces { .. } => 20,
            Op::Mirror { .. } => 21,
        }
    }

    /// The frozen wire position of every variant of the enums nested **inside**
    /// an [`Op`].
    ///
    /// Same reasoning one level down, and it is not academic: `MergeTarget` is
    /// the entire difference between "fuse these at their centre" and "fuse these
    /// onto that one", and a swap would replay a saved session as a *different
    /// edit* with no decode error anywhere. Three `match`es, no wildcards.
    fn frozen_nested(op: &Op) -> Vec<u8> {
        match op {
            Op::MergeVerts { target, .. } => vec![match target {
                MergeTarget::Center => 0,
                MergeTarget::Last => 1,
            }],
            Op::Mirror { axis, .. } => vec![match axis {
                MirrorAxis::X => 0,
                MirrorAxis::Y => 1,
                MirrorAxis::Z => 2,
            }],
            Op::Knife { path } => path
                .iter()
                .map(|p| match p {
                    KnifePoint::Vertex(_) => 0,
                    KnifePoint::Edge { .. } => 1,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn op_discriminants_are_frozen() {
        let every: Vec<Op> = vec![
            Op::AddVertex { position: [0.0; 3] },
            Op::RemoveVertex { vert: VertId(0) },
            Op::AddFace {
                verts: vec![],
                corners: vec![],
                slot: None,
            },
            Op::RemoveFace { face: FaceId(0) },
            Op::SplitEdge {
                half: HalfId(0),
                t: 0.5,
            },
            Op::CollapseEdge { half: HalfId(7) },
            Op::SplitFace {
                from: HalfId(0),
                to: HalfId(1),
            },
            Op::WeldVerts {
                keep: VertId(0),
                merge: VertId(1),
            },
            Op::TranslateVerts {
                verts: vec![],
                delta: [0.0; 3],
            },
            Op::SetCornerUv {
                half: HalfId(0),
                uv: [0.0; 2],
            },
            Op::SetCornerNormal {
                half: HalfId(0),
                normal: None,
            },
            Op::SetEdgeSharp {
                half: HalfId(0),
                sharp: true,
            },
            Op::SetFaceSlot {
                face: FaceId(0),
                slot: None,
            },
            Op::ExtrudeFaces {
                faces: vec![],
                distance: 1.0,
            },
            Op::ExtrudeEdges {
                edges: vec![],
                delta: [0.0; 3],
            },
            Op::InsetFaces {
                faces: vec![],
                amount: 0.1,
                individual: false,
            },
            Op::BevelEdges {
                edges: vec![],
                amount: 0.1,
            },
            Op::LoopCut {
                half: HalfId(0),
                cuts: 1,
            },
            Op::Knife { path: vec![] },
            Op::MergeVerts {
                verts: vec![],
                target: MergeTarget::Center,
            },
            Op::SubdivideFaces { faces: vec![] },
            Op::Mirror {
                axis: MirrorAxis::X,
                coord: 0.0,
            },
        ];
        assert_eq!(every.len(), 22, "one sample per variant");
        let cfg = bincode::config::standard();
        for op in &every {
            let bytes = bincode::serde::encode_to_vec(op, cfg).unwrap();
            assert_eq!(
                bytes[0],
                frozen_discriminant(op),
                "{op:?} moved on the wire"
            );
        }
        // The concrete byte string the P23.3 audit measured, **unchanged** by
        // P23.4's nine appended variants — which is the whole claim of
        // "append-only", stated as bytes rather than as intent.
        assert_eq!(
            bincode::serde::encode_to_vec(Op::CollapseEdge { half: HalfId(7) }, cfg).unwrap(),
            vec![5, 7],
        );
        // And the version ladder did NOT move: appending leaves every already-
        // written session decoding as exactly what it said.
        assert_eq!(SessionSave::CURRENT_VERSION, 1);
    }

    #[test]
    fn the_enums_nested_inside_an_op_are_frozen_too() {
        let cfg = bincode::config::standard();
        // `MergeVerts` writes: [19, len(verts)=0, target]. `Mirror`: [21, axis, f64].
        for (op, want) in [
            (
                Op::MergeVerts {
                    verts: vec![],
                    target: MergeTarget::Center,
                },
                vec![19u8, 0, 0],
            ),
            (
                Op::MergeVerts {
                    verts: vec![],
                    target: MergeTarget::Last,
                },
                vec![19, 0, 1],
            ),
        ] {
            assert_eq!(
                bincode::serde::encode_to_vec(&op, cfg).unwrap(),
                want,
                "{op:?} moved on the wire"
            );
            assert_eq!(frozen_nested(&op), vec![want[2]]);
        }
        for (axis, tag) in [(MirrorAxis::X, 0u8), (MirrorAxis::Y, 1), (MirrorAxis::Z, 2)] {
            let op = Op::Mirror { axis, coord: 0.0 };
            let bytes = bincode::serde::encode_to_vec(&op, cfg).unwrap();
            assert_eq!((bytes[0], bytes[1]), (21, tag), "{op:?} moved on the wire");
            assert_eq!(frozen_nested(&op), vec![tag]);
        }
        let knife = Op::Knife {
            path: vec![
                KnifePoint::Vertex(VertId(3)),
                KnifePoint::Edge {
                    half: HalfId(4),
                    t: 0.5,
                },
            ],
        };
        let bytes = bincode::serde::encode_to_vec(&knife, cfg).unwrap();
        assert_eq!((bytes[0], bytes[1], bytes[2]), (18, 2, 0), "Knife header");
        assert_eq!(frozen_nested(&knife), vec![0, 1]);
    }

    #[test]
    fn a_session_of_modelling_ops_round_trips_through_both_codecs() {
        // The nine new variants are journal entries like any other: they replay,
        // they undo, and they survive both encoders (architecture rule 4).
        let mut s = MeshSession::new(cube(2.0));
        let top = s.mesh().face_ids().next().unwrap();
        s.apply(Op::ExtrudeFaces {
            faces: vec![top],
            distance: 0.5,
        })
        .unwrap();
        s.apply(Op::InsetFaces {
            faces: s.mesh().face_ids().take(1).collect(),
            amount: 0.1,
            individual: false,
        })
        .unwrap();
        s.apply(Op::SubdivideFaces {
            faces: s.mesh().face_ids().take(2).collect(),
        })
        .unwrap();
        let head = s.mesh().encoded();

        let cfg = bincode::config::standard();
        let save = s.save();
        let bin = bincode::serde::encode_to_vec(&save, cfg).unwrap();
        let (back, _): (SessionSave, _) = bincode::serde::decode_from_slice(&bin, cfg).unwrap();
        let json: SessionSave =
            serde_json::from_str(&serde_json::to_string(&save).unwrap()).unwrap();
        assert_eq!(back, save);
        assert_eq!(json, save);

        let restored = MeshSession::restore(back).unwrap();
        assert_eq!(restored.mesh().encoded(), head);
        // And undo walks all the way back to the base.
        let mut s = restored;
        while s.undo() {
            assert_eq!(validate(s.mesh()), Ok(()));
        }
        assert_eq!(s.mesh().encoded(), cube(2.0).encoded());
    }

    #[test]
    fn a_save_carries_its_version_first_on_the_wire() {
        let s = MeshSession::new(plane(2.0));
        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(s.save(), cfg).unwrap();
        assert_eq!(
            bytes[0],
            SessionSave::CURRENT_VERSION as u8,
            "the version must be the first byte, or it cannot guard what follows"
        );
    }

    #[test]
    fn restore_refuses_a_save_from_another_schema() {
        let s = MeshSession::new(plane(2.0));
        let mut save = s.save();
        save.schema_version = SessionSave::CURRENT_VERSION + 1;
        assert_eq!(
            MeshSession::restore(save.clone()),
            Err(SessionError::UnsupportedSchema {
                found: SessionSave::CURRENT_VERSION + 1,
                current: SessionSave::CURRENT_VERSION,
            })
        );
        save.schema_version = 0;
        assert!(matches!(
            MeshSession::restore(save),
            Err(SessionError::UnsupportedSchema { .. })
        ));
    }

    // ── B3: the base mesh is validated at the trust boundary ────────────────

    #[test]
    fn restore_refuses_a_corrupt_base_instead_of_panicking() {
        // The audit's two corruptions. Before this check, the first restored
        // `Ok` and then failed `validate` on demand; the second PANICKED inside
        // `split_edge`'s `expect("live half-edge id")` chain — in an op whose own
        // contract says a refusal is a value.
        for mangle in [
            |m: &mut Mesh| m.halfs.get_mut(0).unwrap().next = HalfId(99),
            |m: &mut Mesh| m.halfs.get_mut(0).unwrap().twin = HalfId(99),
        ] {
            let mut base = plane(2.0);
            mangle(&mut base);
            let save = SessionSave {
                schema_version: SessionSave::CURRENT_VERSION,
                base,
                ops: vec![Op::SplitEdge {
                    half: HalfId(0),
                    t: 0.5,
                }],
                cursor: 1,
            };
            match MeshSession::restore(save) {
                Err(SessionError::InvalidBase(v)) => assert!(!v.is_empty()),
                other => panic!("expected InvalidBase, got {other:?}"),
            }
        }
    }

    #[test]
    fn restore_refuses_a_cursor_that_outruns_its_ops() {
        // A truncated write: the tail of `ops` is lost, the cursor is not.
        // Clamping produced a fully consistent session that had silently dropped
        // two edits, and the next `save()` wrote the shortened history back as
        // authoritative — corruption laundered by the function documented as the
        // trust boundary.
        let mut s = MeshSession::new(plane(2.0));
        for i in 0..5 {
            s.apply(Op::AddVertex {
                position: [i as f64, 0.0, 0.0],
            })
            .unwrap();
        }
        let mut save = s.save();
        assert_eq!(save.cursor, 5);
        save.ops.truncate(3); // the write was cut short

        assert_eq!(
            MeshSession::restore(save.clone()),
            Err(SessionError::CursorOutOfRange { cursor: 5, ops: 3 })
        );

        // A cursor exactly at the end is the normal "nothing to redo" case and
        // must still be accepted — the refusal is for `>`, not `>=`.
        save.cursor = 3;
        let ok = MeshSession::restore(save).expect("a cursor at the end is legal");
        assert_eq!(ok.cursor(), 3);
        assert!(!ok.can_redo());
    }

    #[test]
    fn restore_accepts_a_sound_base() {
        // The other half of the gate: validation must not reject real saves.
        let mut s = MeshSession::new(cube(1.0));
        script(&mut s);
        let restored = MeshSession::restore(s.save()).expect("a real save restores");
        assert_eq!(restored.mesh().encoded(), s.mesh().encoded());
    }

    // ── M4: the generation stamp is monotone across every door ──────────────

    /// Every stamp this test can reach, from every door, must be distinct.
    ///
    /// The previous version applied exactly one op and never compared two live
    /// sessions — the single count at which a local `+= 1` and a global draw
    /// happen to agree. There were in fact TWO schemes interleaved: `new` and
    /// `restore` drew from the process counter while every mutation incremented
    /// locally, so two live sessions collided after a single edit and a restored
    /// long session handed out stamps it had already used. The named consumer is
    /// P23.4's selection cache keyed `(generation, HalfId)`, which would then
    /// accept one document's stamp for another document's ids.
    #[test]
    fn generation_never_repeats_across_sessions_edits_clones_or_restores() {
        let mut seen: Vec<u64> = Vec::new();
        let mut note = |g: u64, what: &str| {
            assert!(!seen.contains(&g), "stamp {g} reused by {what}: {seen:?}");
            seen.push(g);
        };

        // Two LIVE sessions, edited in lockstep. This is the interleaving that
        // collided: A=2, B=3, one op on A → A=3 == B.
        let mut a = MeshSession::new(plane(2.0));
        note(a.generation(), "session A");
        let mut b = MeshSession::new(plane(2.0));
        note(b.generation(), "session B");
        for i in 0..5 {
            a.apply(Op::AddVertex {
                position: [i as f64, 0.0, 0.0],
            })
            .unwrap();
            note(a.generation(), "an edit on A");
            b.apply(Op::AddVertex {
                position: [i as f64, 1.0, 0.0],
            })
            .unwrap();
            note(b.generation(), "an edit on B");
        }

        // History moves are mutations too.
        assert!(a.undo());
        note(a.generation(), "an undo on A");
        assert!(a.redo());
        note(a.generation(), "a redo on A");

        // A clone is a second document; it must not answer with the first's stamp.
        let cloned = a.clone();
        note(cloned.generation(), "a clone of A");

        // And a restore of a LONG session must not hand back stamps that session
        // already used — the case a local counter reset to 0 got wrong outright.
        let long = MeshSession::restore(a.save()).unwrap();
        note(long.generation(), "a restore of A");
        assert!(
            long.generation() > *seen[..seen.len() - 1].iter().max().unwrap(),
            "a restored session must start beyond every stamp already issued"
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
