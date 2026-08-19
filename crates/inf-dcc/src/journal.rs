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
/// The version guards the *shape* of this struct **and of everything it
/// contains** — including [`Mesh`], whose `HalfEdge` grew a `seam` flag at P23.5
/// and took the version from 1 to 2 with it. It does **not** guard
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
    /// * **v1** (P23.3): `{schema_version, base, ops, cursor}`, `Op` frozen by
    ///   `op_discriminants_are_frozen`.
    /// * **v2** (P23.5): `Mesh`'s `HalfEdge` grew a `seam: bool`, so the `base`
    ///   this struct carries is a **different shape** on the wire. bincode is
    ///   positional, so a v1 save read as v2 mis-parses everything after the
    ///   first half-edge — which is precisely why the version is the first field
    ///   and why this had to move.
    ///
    /// * **v3** (P24.2): the **skin channel**. `Vert` gained a `skin:
    ///   VertWeights` tail field and `Mesh` gained a `skin:
    ///   Option<SkinBinding>` tail field, so — positional encoding again — a v2
    ///   save read as v3 runs out of bytes at the first vertex and a v3 save read
    ///   as v2 leaves the whole channel as trailing garbage. The three new `Op`
    ///   variants (27/28/29) did **not** need this bump; they are appended, and
    ///   `op_discriminants_are_frozen` is what says so in bytes.
    ///
    /// * **v4** (Wave D): [`Op::BevelEdges`] grew a `segments: u32` field — the
    ///   first and only time this crate has changed an **existing** variant's
    ///   shape. bincode is positional, so a v3 `BevelEdges` read as v4 runs off
    ///   the end of the stream at the next op and a v4 read as v3 leaves four
    ///   bytes of garbage where the next discriminant should be. The
    ///   *discriminant* did not move (16 is still 16), which is why the
    ///   frozen-discriminant match is silent about this and the version is not:
    ///   the two gates see different halves of the wire and both are needed.
    ///   Wave D's **seven** appended variants (31..=37) did **not** need this
    ///   bump — see `frozen_discriminant`, which lists all seven.
    ///
    /// **No migration, and that is still the honest answer**: `MeshSession` has
    /// only ever lived in `DccState` for the life of a process, so **zero v1 or
    /// v2 sessions exist** on any disk. Writing a migration would be writing a
    /// decoder for a file that has never been produced and then claiming it
    /// works. Both directions are refused loudly by [`MeshSession::restore`] —
    /// and the refusal is a *named* error carrying a remedy
    /// ([`SessionError::SchemaTooOld`] / [`SessionError::SchemaTooNew`]), which
    /// is the `inf_asset` doctrine applied to the one format in this crate that
    /// is not an `AssetPayload`.
    ///
    /// Bump on any change to this struct's shape, to `Mesh`'s, or to the `Op`
    /// discriminants — the last of which is pinned separately, because a
    /// mis-numbered variant produces a structurally valid save of the wrong edit.
    pub const CURRENT_VERSION: u32 = 4;

    /// What to do about a session this build is too new to read.
    ///
    /// An **instruction**, not a restatement — the bar `SkeletonAsset`'s
    /// `UPGRADE_REMEDY` set at P24.1. It can say "nothing is lost" and mean it:
    /// a session is in-progress edit *history*, and the `.inf_mesh` it was opened
    /// from is a separate file this refusal does not touch.
    pub const TOO_OLD_REMEDY: &'static str =
        "re-open the mesh (Content Drawer ▸ double-click) — the asset itself is unaffected, only this session's undo history is discarded";

    /// …and one this build is too old to read.
    pub const TOO_NEW_REMEDY: &'static str =
        "update Infinity Engine, or re-open the mesh (Content Drawer ▸ double-click) to start a fresh history from the asset on disk";
}

/// Why a [`SessionSave`] could not become a session.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SessionError {
    /// The save was written by an **older** build and cannot be read by this
    /// one.
    ///
    /// Split from [`SessionError::SchemaTooNew`] at P24.2, on the doctrine
    /// `inf_asset` established for `AssetPayload`: the two directions are
    /// different problems with different remedies, and one message that covers
    /// both tells a user nothing they can act on. This crate's format is not an
    /// `AssetPayload` (it is a session, not an asset), so it carries its own
    /// remedy rather than inheriting one.
    #[error("this edit session was written by an older build (schema v{found}, this build reads v{current}); {remedy}")]
    SchemaTooOld {
        found: u32,
        current: u32,
        remedy: &'static str,
    },
    /// The save was written by a **newer** build.
    #[error("this edit session was written by a newer build (schema v{found}, this build reads v{current}); {remedy}")]
    SchemaTooNew {
        found: u32,
        current: u32,
        remedy: &'static str,
    },
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

    /// **Which** snapshots are held, ascending (Hardening D).
    ///
    /// The count alone cannot see *which* ones survived, and the eviction rule is
    /// "nearest the cursor" — so a restore that evicted in a different order than
    /// the incremental path would leave the same NUMBER of checkpoints in the
    /// wrong places, replaying more ops per undo and reporting nothing.
    pub fn checkpoint_keys(&self) -> Vec<usize> {
        self.checkpoints.keys().copied().collect()
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
        evict_checkpoints(&mut self.checkpoints, self.cursor);
    }

    /// **Re-parameterize an op that is already in the past**, and re-derive
    /// everything after it.
    ///
    /// The wave's signature capability and the journal finally earning its
    /// shape: `ops[index] = op`, then replay. See [the `amend` module](mod@crate::amend)
    /// for the two gates this runs and why neither is trusted alone.
    ///
    /// **Every refusal is inert** — the session is byte-identical, the same law
    /// the ops themselves obey. Nothing is written until the last check passes,
    /// because everything is computed on clones.
    ///
    /// On success the cursor does not move: the author asked to change an edit,
    /// not to travel to it. The generation stamp **does** move, because the mesh
    /// changed and every cached id must be re-fetched (the id-reuse rule) — even
    /// though this particular change is guaranteed not to have renumbered
    /// anything, because that guarantee is exactly what
    /// [`crate::amend::Topology`] just checked and a consumer has no way to know
    /// that from the stamp.
    pub fn amend(&mut self, index: usize, op: Op) -> Result<(), crate::amend::AmendError> {
        use crate::amend::{amend_shape_ok, topology, AmendError};

        if index >= self.cursor {
            return Err(AmendError::OutOfRange {
                index,
                cursor: self.cursor,
                ops: self.ops.len(),
            });
        }
        amend_shape_ok(&self.ops[index], &op)?;

        // The prefix is shared by both branches, so it is replayed once. It
        // cannot refuse: it is the journal's own history, and `apply` only ever
        // recorded ops that succeeded.
        let prefix =
            Self::replay(&self.base, &self.ops[..index]).expect("a journalled prefix replays");
        let mut old = prefix.clone();
        ops::apply(&mut old, &self.ops[index]).expect("a journalled op replays");
        let mut new = prefix;
        ops::apply(&mut new, &op).map_err(AmendError::Refused)?;
        // **Only when there is something after it.** The topology check exists to
        // protect the ops that name ids the amended one minted; the LAST op in a
        // journal has none, so amending it is free even when it restructures.
        // That is Blender's F9 redo panel as a special case of the general
        // feature, rather than as a second mechanism.
        if index + 1 < self.ops.len() && topology(&old) != topology(&new) {
            return Err(AmendError::IdAllocationChanged { index: index + 1 });
        }

        // **The whole tail, including the redo half.** A redo the author has not
        // pressed is still their work; leaving it un-replayed would mean an
        // amendment could silently make a future redo apply to different
        // geometry, which is the exact hazard this function exists to close.
        let mut current: Option<Mesh> = None;
        for (i, tail) in self.ops[index + 1..].iter().enumerate() {
            let at = index + 1 + i;
            if at == self.cursor {
                current = Some(new.clone());
            }
            ops::apply(&mut old, tail).expect("a journalled op replays");
            ops::apply(&mut new, tail)
                .map_err(|source| AmendError::TailRefused { index: at, source })?;
            // Compared at EVERY step, not only at the end: an intermediate
            // divergence that happens to converge again would leave the ops in
            // between having landed on the wrong polygons, and a final
            // comparison cannot see that.
            if topology(&old) != topology(&new) {
                return Err(AmendError::IdAllocationChanged { index: at });
            }
        }
        let current = current.unwrap_or_else(|| new.clone());
        validate(&current).map_err(AmendError::InvalidResult)?;

        // Nothing above wrote to `self`. Commit.
        self.ops[index] = op;
        self.current = current;
        // Every snapshot at or after the amended op describes a mesh that no
        // longer exists. The ones before it are still exactly right, which is
        // why this is a `retain` and not a `clear`.
        self.checkpoints.retain(|&k, _| k <= index);
        self.generation = fresh_generation();
        self.evict();
        Ok(())
    }

    /// Replay an op sequence onto a base mesh (`evict_checkpoints` carries the
    /// bound the two insertion points share). **The definition of the journal**:
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
        // Both directions, each by its own name (P24.2). The remedies differ
        // because the situations do: an old session is *stale work* the user can
        // abandon safely (the asset on disk is the authority and is untouched),
        // and a new one means the tool is behind the file.
        match save.schema_version.cmp(&SessionSave::CURRENT_VERSION) {
            std::cmp::Ordering::Less => {
                return Err(SessionError::SchemaTooOld {
                    found: save.schema_version,
                    current: SessionSave::CURRENT_VERSION,
                    remedy: SessionSave::TOO_OLD_REMEDY,
                })
            }
            std::cmp::Ordering::Greater => {
                return Err(SessionError::SchemaTooNew {
                    found: save.schema_version,
                    current: SessionSave::CURRENT_VERSION,
                    remedy: SessionSave::TOO_NEW_REMEDY,
                })
            }
            std::cmp::Ordering::Equal => {}
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
    ///
    /// **`evict` runs inside the loop, not after it** (Hardening D). The module
    /// bounds itself to `MAX_CHECKPOINTS` whole-mesh snapshots; evicting only at
    /// the end held `cursor / CHECKPOINT_INTERVAL` of them at once — one per
    /// interval of the restored history, each a full `Mesh` — so restoring a long
    /// session peaked at a multiple of the bound the module claims. Every other
    /// insertion point (`seek`) already evicts on the spot, and the eviction rule
    /// is cursor-relative, so applying it per insertion is not merely cheaper: it
    /// is the same answer, reached without the peak.
    fn rebuild_checkpoints(&mut self) {
        let mut mesh = self.base.clone();
        // The free function rather than `self.evict()`: the loop holds `&self.ops`,
        // and the eviction rule needs nothing but the map and the cursor.
        let cursor = self.cursor;
        for (i, op) in self.ops[..cursor].iter().enumerate() {
            ops::apply(&mut mesh, op).expect("the prefix was replayed by `restore`");
            let count = i + 1;
            if count.is_multiple_of(CHECKPOINT_INTERVAL) {
                self.checkpoints.insert(count, mesh.clone());
                evict_checkpoints(&mut self.checkpoints, cursor);
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

/// Retain the `MAX_CHECKPOINTS` snapshots nearest `cursor`; drop the farthest,
/// breaking ties toward the older one.
///
/// A free function so it can be applied while `&self.ops` is borrowed — see
/// [`MeshSession::rebuild_checkpoints`], where evicting per insertion rather
/// than once at the end is the difference between holding the module's bound and
/// holding `cursor / CHECKPOINT_INTERVAL` whole meshes at once.
fn evict_checkpoints(checkpoints: &mut BTreeMap<usize, Mesh>, cursor: usize) {
    while checkpoints.len() > MAX_CHECKPOINTS {
        let victim = checkpoints
            .keys()
            .copied()
            .max_by_key(|&k| (k.abs_diff(cursor), std::cmp::Reverse(k)))
            .expect("non-empty while over the cap");
        checkpoints.remove(&victim);
    }
}

/// **One sample of every [`Op`] variant**, in wire order.
///
/// Shared by the frozen-discriminant test and by [`crate::amend`]'s classifier
/// tests, because two lists of "every op" is one list that goes stale. It is
/// also why the discriminant freeze and the amendability classifier cannot
/// disagree about how many variants exist.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;
    use crate::model::{MergeTarget, MirrorAxis};
    use crate::sculpt::{SculptFalloff, SculptMode};
    use crate::topo::{FaceId, HalfId, VertId};

    pub(crate) fn one_of_every_op() -> Vec<Op> {
        vec![
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
                segments: 1,
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
            Op::Sculpt {
                mode: SculptMode::Draw,
                dabs: vec![],
                radius: 1.0,
                strength: 0.1,
                falloff: SculptFalloff::Smooth,
            },
            Op::RotateVerts {
                verts: vec![],
                pivot: [0.0; 3],
                axis: [0.0, 1.0, 0.0],
                radians: 0.0,
            },
            Op::ScaleVerts {
                verts: vec![],
                pivot: [0.0; 3],
                factor: [1.0; 3],
            },
            Op::SetEdgeSeam {
                half: HalfId(0),
                seam: true,
            },
            Op::Unwrap { corners: vec![] },
            Op::BindSkin {
                skeleton: None,
                joints: 1,
            },
            Op::AssignWeights { weights: vec![] },
            Op::ClearSkin,
            Op::AddMaterialSlots {
                names: vec!["Default".into()],
            },
            Op::MoveVerts { moves: vec![] },
            Op::SetEdgesSharp {
                halfs: vec![],
                sharp: true,
            },
            Op::FlipFaces { faces: vec![] },
            Op::DissolveEdges { edges: vec![] },
            Op::BridgeLoops { pairs: vec![] },
            Op::SetEdgesSeam {
                halfs: vec![],
                seam: true,
            },
            Op::MoveUvs { corners: vec![] },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, plane};
    use crate::model::{KnifePoint, MergeTarget, MirrorAxis};
    use crate::sculpt::{SculptFalloff, SculptMode};
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

    /// **A restore reaches the same checkpoint set the live session had**
    /// (Hardening D), which is the load-bearing half of moving `evict` inside
    /// `rebuild_checkpoints`' loop: the eviction rule is cursor-relative, the
    /// cursor does not move during a rebuild, and the keys are inserted in
    /// increasing order *toward* it — so the farthest is always the earliest
    /// inserted and evicting as we go drops exactly what evicting at the end
    /// would have. Asserted rather than argued, because "the answer is the same"
    /// is precisely the claim a peak-memory fix must not get wrong.
    #[test]
    fn a_restore_keeps_the_checkpoints_a_live_session_kept() {
        let mut live = MeshSession::new(plane(2.0));
        for i in 0..(CHECKPOINT_INTERVAL * MAX_CHECKPOINTS * 3) {
            live.apply(Op::TranslateVerts {
                verts: vec![],
                delta: [i as f64, 0.0, 0.0],
            })
            .unwrap();
        }
        let restored = MeshSession::restore(live.save()).expect("a live session restores");
        assert_eq!(
            restored.checkpoint_keys(),
            live.checkpoint_keys(),
            "the rebuild must reach the same snapshots, not merely the same count"
        );
        assert!(
            restored.checkpoint_count() <= MAX_CHECKPOINTS,
            "held {}",
            restored.checkpoint_count()
        );
        assert_eq!(restored.mesh(), live.mesh(), "and the same mesh");
    }

    /// **The peak, which no value can see** (Hardening D).
    ///
    /// `rebuild_checkpoints` used to evict once, after its loop, so restoring a
    /// session held one whole `Mesh` per `CHECKPOINT_INTERVAL` of replayed
    /// history at once — `cursor / 32` snapshots against a documented bound of
    /// 8. Nothing observable survives that moment: the *final* count was always
    /// correct, which is why `checkpoints_stay_bounded_over_a_long_session`
    /// passed throughout. So the enforcement is a source pin on the placement.
    #[test]
    fn the_rebuild_evicts_inside_its_loop() {
        // The P22 CRLF law: normalize before searching a `.rs`.
        let src = include_str!("journal.rs").replace("\r\n", "\n");
        let start = src
            .find("fn rebuild_checkpoints(&mut self) {")
            .expect("`rebuild_checkpoints` occurs nowhere — was it renamed?");
        let body = &src[start..];
        let end = body
            .find("\n    }\n")
            .expect("`rebuild_checkpoints` does not terminate at method indentation");
        let body = &body[..end];
        let insert = body
            .find("self.checkpoints.insert(count, mesh.clone());")
            .expect("the rebuild no longer inserts a checkpoint");
        let evict = body
            .find("evict_checkpoints(&mut self.checkpoints, cursor);")
            .expect(
                "`rebuild_checkpoints` no longer evicts inside its loop — restoring a long \
                 session peaks at one whole Mesh per CHECKPOINT_INTERVAL of history, which \
                 is a multiple of the bound this module documents and which no value can see",
            );
        assert!(
            evict > insert,
            "the eviction must follow the insertion it bounds"
        );
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
            // P23.5 appended THREE more, in this order, at the next free indices.
            // Same rule and the same reason: nothing above moved, so every byte
            // string pinned by the earlier batches is unchanged and
            // `SessionSave::CURRENT_VERSION` does not have to move for this.
            Op::Sculpt { .. } => 22,
            Op::RotateVerts { .. } => 23,
            Op::ScaleVerts { .. } => 24,
            // P23.5's UV half, appended after them.
            Op::SetEdgeSeam { .. } => 25,
            Op::Unwrap { .. } => 26,
            // P24.2's THREE skin ops, appended after them. Fourth batch, same
            // rule: nothing above moved, so `CollapseEdge{7}` is still `[5, 7]`
            // and every byte string pinned by an earlier batch is unchanged.
            // `SessionSave::CURRENT_VERSION` moved to 3 in this batch — for the
            // `Mesh`'s new shape, NOT for these, which is exactly the
            // distinction this match exists to keep legible.
            Op::BindSkin { .. } => 27,
            Op::AssignWeights { .. } => 28,
            Op::ClearSkin => 29,
            // P24.3's slot-table op, appended at the next free index. Fifth
            // batch, same rule: nothing above moved, `CollapseEdge{7}` is still
            // `[5, 7]`, and `SessionSave::CURRENT_VERSION` does NOT move — `Mesh`
            // gained no field, only entries in a `Vec<String>` it already had.
            Op::AddMaterialSlots { .. } => 30,
            // Wave D's SEVEN, appended at the next free indices (31..=37 —
            // the last two land below, after the batch's own seam write).
            // Sixth batch, same rule: nothing above moved, `CollapseEdge{7}` is
            // still `[5, 7]`. `SessionSave::CURRENT_VERSION` DID move in this batch,
            // and **not for these** — `Op::BevelEdges` grew a `segments` field,
            // which is a change to variant 16's shape while leaving its
            // discriminant at 16. That is precisely the change this match cannot
            // see, and precisely what the version exists to guard.
            Op::MoveVerts { .. } => 31,
            Op::SetEdgesSharp { .. } => 32,
            Op::FlipFaces { .. } => 33,
            Op::DissolveEdges { .. } => 34,
            Op::BridgeLoops { .. } => 35,
            // …and the batch seam write, appended after them. Same rule; the
            // version does NOT move for an appended variant.
            Op::SetEdgesSeam { .. } => 36,
            Op::MoveUvs { .. } => 37,
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
            // P23.5's stroke carries TWO nested enums, and both of them change
            // what a replayed session does rather than whether it decodes: swap
            // `Draw` and `Grab` and a saved stroke becomes a different edit with
            // no error anywhere, exactly like `MergeTarget`.
            Op::Sculpt { mode, falloff, .. } => vec![
                match mode {
                    SculptMode::Draw => 0,
                    SculptMode::Smooth => 1,
                    SculptMode::Flatten => 2,
                    SculptMode::Grab => 3,
                },
                match falloff {
                    SculptFalloff::Smooth => 0,
                    SculptFalloff::Sphere => 1,
                    SculptFalloff::Linear => 2,
                    SculptFalloff::Sharp => 3,
                },
            ],
            // **Exhaustive, no wildcard.** A `_ =>` here reads as "these carry no
            // nested enum" and stays true only until one of them does — at which
            // point this function returns an empty vector for it and the freeze
            // test passes with zero coverage of a discriminant that is now on the
            // wire. Listed out, a new nested enum stops the crate compiling.
            Op::AddVertex { .. }
            | Op::RemoveVertex { .. }
            | Op::AddFace { .. }
            | Op::RemoveFace { .. }
            | Op::SplitEdge { .. }
            | Op::CollapseEdge { .. }
            | Op::SplitFace { .. }
            | Op::WeldVerts { .. }
            | Op::TranslateVerts { .. }
            | Op::SetCornerUv { .. }
            | Op::SetCornerNormal { .. }
            | Op::SetEdgeSharp { .. }
            | Op::SetFaceSlot { .. }
            | Op::ExtrudeFaces { .. }
            | Op::ExtrudeEdges { .. }
            | Op::InsetFaces { .. }
            | Op::BevelEdges { .. }
            | Op::LoopCut { .. }
            | Op::SubdivideFaces { .. }
            | Op::RotateVerts { .. }
            | Op::ScaleVerts { .. }
            | Op::SetEdgeSeam { .. }
            | Op::Unwrap { .. }
            // P24.2's three carry no nested enum: a joint index is a number, a
            // weight is a float, and the skeleton is an `Option<[u8; 16]>` whose
            // two tags are serde's own — the same shape as every other `Option`
            // in this file and not a domain enum anyone can renumber.
            | Op::BindSkin { .. }
            | Op::AssignWeights { .. }
            | Op::ClearSkin
            // P24.3's carries a `Vec<String>` — strings, not a domain enum
            // anyone can renumber.
            | Op::AddMaterialSlots { .. }
            // Wave D's five carry ids, floats and a bool. No domain enum among
            // them, and `Op::BevelEdges`'s new `segments` is a `u32` — which is
            // why it is invisible here and visible to the version.
            | Op::MoveVerts { .. }
            | Op::SetEdgesSharp { .. }
            | Op::FlipFaces { .. }
            | Op::DissolveEdges { .. }
            | Op::BridgeLoops { .. }
            | Op::SetEdgesSeam { .. }
            | Op::MoveUvs { .. } => Vec::new(),
        }
    }

    #[test]
    fn op_discriminants_are_frozen() {
        let every: Vec<Op> = tests_support::one_of_every_op();
        assert_eq!(every.len(), 38, "one sample per variant");
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
        // The op discriminants are append-only and every byte string above is
        // unchanged. The version moved at P23.5, at P24.2 and again in Wave D,
        // and none of the three times for the discriminants: `Mesh` grew a
        // `seam` field, then a skin channel, and then `Op::BevelEdges` grew a
        // `segments` field — the first two change the shape of the `base` this
        // struct carries, the third changes the shape of a variant whose
        // discriminant this match still pins at 16. See
        // `SessionSave::CURRENT_VERSION`.
        assert_eq!(SessionSave::CURRENT_VERSION, 4);
        // …and the shape change itself, as bytes. `[16, 1, 7, <8 f64 bytes>, 2]`
        // is discriminant 16, one edge (`HalfId(7)`), the amount, and **then the
        // segment count** — the four trailing bytes a v3 reader does not know are
        // there. This is the assertion that fails if anyone ever "tidies"
        // `segments` back out of the variant or moves it in front of `amount`.
        {
            let bytes = bincode::serde::encode_to_vec(
                Op::BevelEdges {
                    edges: vec![HalfId(7)],
                    amount: 0.5,
                    segments: 2,
                },
                cfg,
            )
            .unwrap();
            assert_eq!(bytes[0], 16, "the bevel discriminant did not move");
            assert_eq!(bytes[1..3], [1, 7], "one edge, HalfId(7)");
            assert_eq!(
                *bytes.last().expect("a non-empty encoding"),
                2,
                "the segment count is the LAST field on the wire"
            );
        }
        // `ClearSkin` is a UNIT variant, so its whole encoding is its
        // discriminant — one byte, and the last free index. Pinned as bytes
        // because a unit variant is the easiest kind to renumber by accident
        // (nothing about it mentions a field).
        assert_eq!(
            bincode::serde::encode_to_vec(Op::ClearSkin, cfg).unwrap(),
            vec![29]
        );
        // P24.3's op, pinned as bytes for the same reason: `[30, 1, 1, 65]` is
        // discriminant 30, one name, one byte long, `"A"`. Unchanged by Wave D's
        // five appended variants — which is the append-only claim, in bytes.
        assert_eq!(
            bincode::serde::encode_to_vec(
                Op::AddMaterialSlots {
                    names: vec!["A".into()]
                },
                cfg
            )
            .unwrap(),
            vec![30, 1, 1, 65]
        );
        // Wave D's last free index, pinned the same way: `[35, 1, 3, 4]` is
        // discriminant 35, one pair, `HalfId(3)` and `HalfId(4)`.
        assert_eq!(
            bincode::serde::encode_to_vec(
                Op::BridgeLoops {
                    pairs: vec![(HalfId(3), HalfId(4))]
                },
                cfg
            )
            .unwrap(),
            vec![35, 1, 3, 4]
        );
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

        // P23.5's stroke: `[22, mode, len(dabs)=0, radius…, strength…, falloff]`.
        // The mode is the second byte and the falloff is the last one, so both
        // ends of the record are pinned without asserting on the float encoding
        // in between.
        for (mode, tag) in [
            (SculptMode::Draw, 0u8),
            (SculptMode::Smooth, 1),
            (SculptMode::Flatten, 2),
            (SculptMode::Grab, 3),
        ] {
            for (falloff, ftag) in [
                (SculptFalloff::Smooth, 0u8),
                (SculptFalloff::Sphere, 1),
                (SculptFalloff::Linear, 2),
                (SculptFalloff::Sharp, 3),
            ] {
                let op = Op::Sculpt {
                    mode,
                    dabs: vec![],
                    radius: 1.0,
                    strength: 0.5,
                    falloff,
                };
                let b = bincode::serde::encode_to_vec(&op, cfg).unwrap();
                assert_eq!((b[0], b[1], b[2]), (22, tag, 0), "{op:?} moved on the wire");
                assert_eq!(*b.last().expect("a non-empty record"), ftag);
                assert_eq!(frozen_nested(&op), vec![tag, ftag]);
            }
        }
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

    /// **Both directions are named refusals that say what to do** (P24.2).
    ///
    /// Not one `UnsupportedSchema` for both: a session from an older build and
    /// one from a newer build are different problems with different remedies,
    /// and the SchemaTooOld doctrine is that the message a user reads has to be
    /// an instruction. The asserts below are on the rendered strings as well as
    /// on the variants, because the string is the part a user sees.
    #[test]
    fn restore_refuses_a_save_from_another_schema_in_both_directions() {
        let s = MeshSession::new(plane(2.0));
        let mut save = s.save();

        save.schema_version = SessionSave::CURRENT_VERSION + 1;
        assert_eq!(
            MeshSession::restore(save.clone()),
            Err(SessionError::SchemaTooNew {
                found: SessionSave::CURRENT_VERSION + 1,
                current: SessionSave::CURRENT_VERSION,
                remedy: SessionSave::TOO_NEW_REMEDY,
            })
        );

        save.schema_version = SessionSave::CURRENT_VERSION - 1;
        assert_eq!(
            MeshSession::restore(save.clone()),
            Err(SessionError::SchemaTooOld {
                found: SessionSave::CURRENT_VERSION - 1,
                current: SessionSave::CURRENT_VERSION,
                remedy: SessionSave::TOO_OLD_REMEDY,
            })
        );
        save.schema_version = 0;
        assert!(matches!(
            MeshSession::restore(save),
            Err(SessionError::SchemaTooOld { .. })
        ));

        // The messages are instructions, and they are readable — the run-of-
        // spaces guard the P24.1 F4 sweep put on `SkeletonAsset`'s remedy, on
        // this format's two.
        for remedy in [SessionSave::TOO_OLD_REMEDY, SessionSave::TOO_NEW_REMEDY] {
            assert!(remedy.contains("Content Drawer"), "{remedy}");
            assert!(
                !remedy.contains("  "),
                "a remedy carries a run of spaces — a line continuation was \
                 eaten: {remedy:?}"
            );
        }
        let mut save = s.save();
        save.schema_version = 2;
        let msg = MeshSession::restore(save).unwrap_err().to_string();
        assert!(msg.contains("older build"), "{msg}");
        assert!(msg.contains(SessionSave::TOO_OLD_REMEDY), "{msg}");
    }

    /// **Appending a tail field without bumping must FAIL** (P24.2 audit M-BUMP).
    ///
    /// The shape pin next door is *symmetric*: it decodes today's save and checks
    /// every byte is consumed. That catches a field REMOVED and it does not catch
    /// a field ADDED — the added bytes are simply trailing, and
    /// `decode_from_slice` ignores trailing bytes by contract. So nothing in the
    /// crate forced the NEXT bump, on the very format this batch just bumped.
    ///
    /// This is the asymmetric half, and it is the M6 lesson applied to the format
    /// M6 was learned on: a **shadow v4** — today's struct plus one tail field,
    /// exactly what an author appending a field would write — must not be
    /// readable as a v3. Two claims, and the second is the one with teeth:
    ///
    ///  1. the v4 bytes are LONGER, so a field really was added;
    ///  2. decoding them as `SessionSave` leaves bytes on the floor, which is the
    ///     signature the shape pin next door refuses. If a future append ever
    ///     came out byte-compatible, this test says so before the ladder is
    ///     quietly wrong.
    #[test]
    fn a_tail_field_appended_without_a_bump_is_caught_by_the_shape_pin() {
        #[derive(Serialize)]
        struct V4Save<'a> {
            schema_version: u32,
            base: &'a Mesh,
            ops: &'a [Op],
            cursor: usize,
            /// The append an author would make. Any type; the point is the bytes.
            tail: Vec<u32>,
        }
        let s = MeshSession::new(cube(1.0));
        let save = s.save();
        let cfg = bincode::config::standard();
        let v3 = bincode::serde::encode_to_vec(&save, cfg).unwrap();
        let v4 = bincode::serde::encode_to_vec(
            &V4Save {
                schema_version: save.schema_version,
                base: &save.base,
                ops: &save.ops,
                cursor: save.cursor,
                tail: vec![7, 8, 9],
            },
            cfg,
        )
        .unwrap();

        assert!(
            v4.len() > v3.len(),
            "the shadow v4 is not longer than v3, so nothing was appended and \
             this test is measuring nothing"
        );
        // The asymmetric claim: v4 bytes decode as a v3 `SessionSave` — that is
        // what makes an un-bumped append silent — and the decoder does NOT
        // consume them all, which is what the shape pin refuses.
        let (back, consumed): (SessionSave, usize) =
            bincode::serde::decode_from_slice(&v4, cfg).expect("a longer save still decodes");
        assert_eq!(
            back, save,
            "the appended field changed the prefix; then it was not an append"
        );
        assert!(
            consumed < v4.len(),
            "an appended tail field left NO trailing bytes, so \
             `a_v4_save_decodes_consuming_every_byte` could not see it either — \
             the ladder has no forcing function at all"
        );
        assert_eq!(
            v4.len() - consumed,
            v4.len() - v3.len(),
            "the unconsumed tail is not exactly the appended field"
        );
    }

    /// **A real v2 save, as a shadow struct** — the M6 lesson.
    ///
    /// The cheap version of this test encodes today's `SessionSave` and lops
    /// bytes off the end, which pins nothing: it is the *current* encoder's
    /// output either way, so the day the current encoder changes the "old" bytes
    /// change with it and the test still passes. What a version ladder needs is
    /// bytes shaped like the OLD struct, written by a declaration of the old
    /// struct — so the shadow types below are P23.5's `Vert` / `Mesh` /
    /// `SessionSave`, frozen, with no `skin` anywhere.
    ///
    /// What it proves, precisely: those bytes carry a leading `2`, this build
    /// refuses them **by name and with a remedy**, and — the part that makes the
    /// bump necessary rather than decorative — they do not decode as a v3
    /// `SessionSave` at all, because `Vert` grew a tail and bincode is
    /// positional.
    #[test]
    fn a_v2_save_is_refused_by_name_and_could_not_have_been_read_anyway() {
        #[derive(Serialize)]
        struct V2Vert {
            position: [f64; 3],
            out: Vec<u32>,
        }
        #[derive(Serialize)]
        struct V2Arena<T> {
            slots: Vec<Option<T>>,
            free: Vec<u32>,
        }
        #[derive(Serialize)]
        struct V2Half {
            origin: u32,
            twin: u32,
            next: u32,
            prev: u32,
            face: Option<u32>,
            uv: [f64; 2],
            normal: Option<[f64; 3]>,
            sharp: bool,
            seam: bool,
        }
        #[derive(Serialize)]
        struct V2Face {
            half: u32,
            material_slot: Option<u32>,
        }
        #[derive(Serialize)]
        struct V2Mesh {
            verts: V2Arena<V2Vert>,
            halfs: V2Arena<V2Half>,
            faces: V2Arena<V2Face>,
            material_slots: Vec<String>,
            slot_names: Vec<(Option<u32>, String)>,
        }
        #[derive(Serialize)]
        struct V2Save {
            schema_version: u32,
            base: V2Mesh,
            ops: Vec<Op>,
            cursor: usize,
        }

        // A one-triangle v2 mesh: three vertices, three twin pairs, one face.
        let v2 = V2Save {
            schema_version: 2,
            base: V2Mesh {
                verts: V2Arena {
                    slots: (0..3)
                        .map(|i| {
                            Some(V2Vert {
                                position: [i as f64, 0.0, 0.0],
                                out: vec![2 * i],
                            })
                        })
                        .collect(),
                    free: Vec::new(),
                },
                halfs: V2Arena {
                    slots: (0..6)
                        .map(|i| {
                            Some(V2Half {
                                origin: i / 2,
                                twin: i ^ 1,
                                next: (i + 2) % 6,
                                prev: (i + 4) % 6,
                                face: (i % 2 == 0).then_some(0),
                                uv: [0.0, 0.0],
                                normal: None,
                                sharp: false,
                                seam: false,
                            })
                        })
                        .collect(),
                    free: Vec::new(),
                },
                faces: V2Arena {
                    slots: vec![Some(V2Face {
                        half: 0,
                        material_slot: None,
                    })],
                    free: Vec::new(),
                },
                material_slots: Vec::new(),
                slot_names: Vec::new(),
            },
            ops: vec![Op::CollapseEdge { half: HalfId(7) }],
            cursor: 0,
        };
        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&v2, cfg).unwrap();
        assert_eq!(bytes[0], 2, "the version is still the first byte at v2");

        // (a) THE REASON THE BUMP WAS NECESSARY: these bytes are not a v3
        //     `SessionSave` at all. `Vert` grew a tail, bincode is positional, so
        //     reading a v2 stream as v3 walks off the end of the vertex arena.
        //     Measured, not assumed — this is the claim a version ladder is FOR,
        //     and a ladder whose old bytes happen to decode is a ladder that was
        //     not needed.
        assert!(
            bincode::serde::decode_from_slice::<SessionSave, _>(&bytes, cfg).is_err(),
            "v2 bytes decoded as a v3 SessionSave — then the shape did not \
             change and this bump has no justification"
        );

        // (b) And the ladder that a v2 save DOES reach — a session handed over
        //     as a value, which is how `DccState` restores one — refuses it by
        //     name, with the remedy.
        let mut v2_shaped = MeshSession::new(plane(2.0)).save();
        v2_shaped.schema_version = 2;
        assert_eq!(
            MeshSession::restore(v2_shaped),
            Err(SessionError::SchemaTooOld {
                found: 2,
                current: SessionSave::CURRENT_VERSION,
                remedy: SessionSave::TOO_OLD_REMEDY,
            }),
            "a v2 save must be refused by name"
        );
    }

    /// **The v3 shape, pinned by FULL BYTE CONSUMPTION.**
    ///
    /// `decode_from_slice` ignores trailing bytes, so "it decoded" is not
    /// "the shape is right": a save with an extra field appended decodes
    /// perfectly as the shorter struct and silently drops it. Asserting that the
    /// decoder consumed **every** byte is what turns a successful decode into a
    /// statement about the shape.
    #[test]
    fn a_v4_save_decodes_consuming_every_byte() {
        let mut s = MeshSession::new(cube(1.0));
        s.apply(Op::BindSkin {
            skeleton: Some([3; 16]),
            joints: 5,
        })
        .unwrap();
        let first = s.mesh().vert_ids().next().unwrap();
        s.apply(Op::AssignWeights {
            weights: vec![(
                first,
                crate::skin::VertWeights::from_pairs(&[(1, 0.25), (4, 0.75)]),
            )],
        })
        .unwrap();

        let cfg = bincode::config::standard();
        let save = s.save();
        let bytes = bincode::serde::encode_to_vec(&save, cfg).unwrap();
        assert_eq!(bytes[0], 4, "the version is the first byte");
        let (back, consumed): (SessionSave, usize) =
            bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(
            consumed,
            bytes.len(),
            "the decoder left bytes on the floor — the struct on the wire is not \
             the struct this build declares"
        );
        assert_eq!(back, save);
        // …and the channel survived the round trip, values and all.
        let restored = MeshSession::restore(back).unwrap();
        assert_eq!(
            restored.mesh().skin_binding(),
            Some(crate::skin::SkinBinding {
                skeleton: Some([3; 16]),
                joints: 5
            })
        );
        assert_eq!(restored.mesh().canonical(), s.mesh().canonical());
        // The dual-format rule (architecture rule 4): the same save through a
        // self-describing encoder is the same value.
        let json: SessionSave =
            serde_json::from_str(&serde_json::to_string(&save).unwrap()).unwrap();
        assert_eq!(json, save);
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

    /// **A hostile v4 stream is a refusal, never a panic** (audit).
    ///
    /// The wave appended seven `Op` variants and bumped the version for an
    /// eighth reason, and the ladder's two directions are both refused by name
    /// — but the version is only the *first byte*. Everything after it is a
    /// positional bincode stream, and the two ways a real file goes wrong are a
    /// **truncation** (a write cut short) and a **flipped byte** (a bad sector,
    /// a bad transfer). Neither may reach a panic, because `restore` is
    /// documented as this crate's trust boundary and a trust boundary that
    /// aborts is not one.
    ///
    /// Driven over the LAST appended variant (`MoveUvs`, 37), so the arm covers
    /// the newest bytes on the wire rather than the oldest.
    #[test]
    fn a_hostile_v4_stream_refuses_as_a_value_and_never_panics() {
        let mut s = MeshSession::new(cube(1.0));
        let h = s
            .mesh()
            .half_ids()
            .find(|&h| s.mesh().is_boundary(h) == Some(false))
            .expect("an interior corner");
        s.apply(Op::MoveUvs {
            corners: vec![(h, [0.25, 0.75])],
        })
        .expect("a uv move");
        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(s.save(), cfg).expect("encodes");
        assert_eq!(bytes[0], 4, "the fixture is a v4 stream");
        assert!(bytes.contains(&37), "the fixture carries the last variant");

        // (a) **Every truncation.** A prefix that happens to decode is still
        //     handed to `restore`, which is the boundary the claim is about.
        let mut decoded_prefixes = 0usize;
        for cut in 0..bytes.len() {
            if let Ok((save, _)) =
                bincode::serde::decode_from_slice::<SessionSave, _>(&bytes[..cut], cfg)
            {
                decoded_prefixes += 1;
                let _ = MeshSession::restore(save);
            }
        }
        // Not vacuous in the other direction either: the WHOLE stream decodes.
        assert!(
            bincode::serde::decode_from_slice::<SessionSave, _>(&bytes, cfg).is_ok(),
            "the fixture does not decode at all, so the truncations proved nothing"
        );
        let _ = decoded_prefixes;

        // (b) **Every single-byte corruption**, both saturated and cleared —
        //     which between them reach an out-of-range discriminant, a bogus
        //     container length and a NaN float.
        for i in 0..bytes.len() {
            for fill in [0x00u8, 0xff] {
                let mut bad = bytes.clone();
                bad[i] = fill;
                if let Ok((save, _)) =
                    bincode::serde::decode_from_slice::<SessionSave, _>(&bad, cfg)
                {
                    let _ = MeshSession::restore(save);
                }
            }
        }

        // (c) …and a discriminant past the last variant is a refusal rather
        //     than a match on something that does not exist.
        let bogus = bincode::serde::encode_to_vec((4u32, 200u8), cfg).expect("encodes");
        assert!(
            bincode::serde::decode_from_slice::<SessionSave, _>(&bogus, cfg).is_err(),
            "a stream naming variant 200 decoded"
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
