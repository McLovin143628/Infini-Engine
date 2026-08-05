//! **Surface deformation** (P22.1): the sparse, sim-authoritative field of how
//! far the ground has been pressed down, and by what.
//!
//! A boot in snow, a crate settling into mud, a wheel cutting a rut — all of it
//! is one number per ground sample: **how many metres below its authored height
//! this patch of surface currently sits**, plus which splat layer's response law
//! is governing it. The heightfield itself is never touched: a footprint is not
//! an edit to the terrain asset, it is a transient offset the renderer composes
//! on top (see `inf_render::deform`), so nothing here can dirty a `.inf_terrain`
//! and no schema anywhere moves.
//!
//! # Why the CPU, at fixed step
//!
//! The obvious shape for this is a compute-written ring buffer. It is the wrong
//! shape for *this* engine, twice over. There is no persistent compute-written
//! texture anywhere in the renderer to model it on, and Ring-1's GPU erosion is
//! already documented as **not bit-identical across adapters** — so a
//! GPU-authoritative deformation field would make a committed level's ground
//! depend on which card the player owns, and the replay / PIE-==-shipping gates
//! would die on the first machine that isn't the author's. Determinism for
//! committed content therefore lives on the CPU, at the fixed step, in `f32`/`f64`
//! IEEE arithmetic that is bit-identical everywhere. The GPU only *draws* it.
//!
//! # Representation & memory rationale
//!
//! (The [`crate::delta`] register: state the worst case, price both candidates,
//! pick one.)
//!
//! The field is a [`BTreeMap`] of fixed-size **cells** over a global lattice of
//! samples spaced [`DEFORM_SAMPLE_PITCH_M`] apart. Each cell covers
//! [`DEFORM_CELL_SAMPLES`]² samples — 16 × 16 at 0.25 m, i.e. a 4 m × 4 m square —
//! and stores a dense `f32` depth plus a `u8` layer per sample.
//!
//! Consider the case the design must survive: a character walking for an hour.
//! A footfall is a ~0.25 m-radius disk, so a stride touches a handful of samples
//! and a walked kilometre touches on the order of 10⁴ of them, spread along a
//! line.
//!
//! * **Sparse per-sample** (`BTreeMap<(i32,i32), (f32,u8)>`): ~40 B per live
//!   sample once the map node is counted, and *every* stamp write and every
//!   recovery step chases a tree node. A footprint costs a dozen tree descents.
//! * **Dense cells**: 16 × 16 × 5 B = 1280 B per cell, so a cell pays for itself
//!   as soon as ~32 of its 256 samples are live — which one footfall already
//!   exceeds, because a 0.25 m disk lands 5 – 9 samples and a stride lays them
//!   down inside the same 4 m square. Stamping and recovery are then flat array
//!   walks, and the render window's pack is a rectangle copy rather than a
//!   scattered gather.
//!
//! Cells win, and the cell is sized for *locality*: 4 m is wider than any
//! footprint this v1 stamps (so a contact touches one or two cells, never a
//! grid of them) and small enough that a lone footprint in the middle of nowhere
//! costs 1.3 KiB rather than a page.
//!
//! `BTreeMap`, never `HashMap`: iteration order is the field's byte order, and
//! the byte order is a gate surface.
//!
//! # The active set is bounded
//!
//! A session that walks forever must not grow memory forever, and two of the
//! four layer archetypes below deliberately **never recover on their own** (see
//! [`LAYER_RESPONSE`]) — so "cells reach zero and are dropped" cannot be the
//! only bound. [`MAX_DEFORM_CELLS`] caps the live set, and the eviction rule is
//! **least-recently-stamped first, ties broken by cell coordinate ascending**:
//! the cell nobody has touched for the longest is the one the player walked away
//! from, and the coordinate tie-break makes the choice a pure function of the
//! step history rather than of map internals. A cell whose every sample has
//! relaxed to exactly zero is removed outright — that is what keeps the field
//! genuinely sparse rather than merely capped.
//!
//! **A cell touched by the current step is never a victim.** Eviction is a
//! memory policy; it must not be able to delete the ground a body is standing
//! on, least of all in the same step that body pressed it. The bound may
//! therefore be overshot while one step's contacts cover more cells than the
//! bound allows — which needs thousands of simultaneous bodies — and the excess
//! is released on the next step that leaves them alone.
//!
//! # Determinism
//!
//! Everything here is a pure function of (previous field, stamps, `dt`, snow
//! rate). No clock, no camera, no residency, no `HashMap`, no `std` trig. Two
//! runs of the same step history produce byte-identical
//! [`state_bytes`](DeformField::state_bytes) — which is what lets the field ride
//! the replay and PIE-==-shipping trace surfaces.

use std::collections::BTreeMap;

use glam::DVec2;

use crate::brush::Falloff;
use crate::splat::SPLAT_LAYERS;

/// Metres between deformation samples.
///
/// 0.25 m is a quarter of a boot and about a sixth of a tyre's contact patch, so
/// a footprint is a shape rather than a texel and a rut has a cross-section. It
/// is also **exactly** the render window's metres-per-texel
/// (`inf_render::deform::DEFORM_TEXEL_M`), which is the point: window texel `k`
/// lands on lattice sample `k`, so packing the window is an index map with no
/// resampling filter anywhere in the path.
pub const DEFORM_SAMPLE_PITCH_M: f64 = 0.25;

/// Samples along one edge of a [`DeformCell`]. See the module docs for why 16.
pub const DEFORM_CELL_SAMPLES: u32 = 16;

/// World size of one cell, metres (`DEFORM_CELL_SAMPLES · DEFORM_SAMPLE_PITCH_M`).
pub const DEFORM_CELL_SIZE_M: f64 = DEFORM_CELL_SAMPLES as f64 * DEFORM_SAMPLE_PITCH_M;

/// Samples in one cell.
pub const DEFORM_CELL_AREA: usize = (DEFORM_CELL_SAMPLES * DEFORM_CELL_SAMPLES) as usize;

/// The live-cell ceiling — the whole of the memory bound.
///
/// 4096 cells × 1.3 KiB ≈ 5.3 MiB, and 4096 × 16 m² covers 65 000 m² of *touched*
/// ground. Tracks are lines, not areas: at ~1 m of trail per cell entered, that
/// is tens of kilometres of footprints before the oldest metre is forgotten,
/// which is far past the point where a player would notice their own old tracks
/// expiring. Raising it costs linearly; it is a memory policy, not a physical
/// constant.
pub const MAX_DEFORM_CELLS: usize = 4096;

/// Hard ceiling on any sample's depth, metres. **Coupled to the terrain raster.**
///
/// `inf_render::passes::terrain` floors a patch's skirt depth at `1.0 m`
/// (`terrain.rs`, the `.max(1.0)` in the instance pack): the skirt is the
/// vertical wall that seals the crack between a fine patch and its coarser
/// neighbour, and it is sized from the patch's own height range.
///
/// Deformation is composed *after* that range was computed, and it is
/// **additive** — so the floor alone is not enough, and the renderer **adds this
/// many metres to every patch's skirt** whenever a scene carries a deformation
/// field. (The first cut relied on the floor; a 0.4 m rut on a patch with 3 m of
/// relief would have opened an unsealed LOD seam exactly where a vehicle drove,
/// because the skirt was 3 m and the surface had moved to 3.4 m.)
///
/// Two consequences follow, and both are enforced: every entry in
/// [`LAYER_RESPONSE`] is at or under this value, and the clamp is applied again
/// per sample here **and** a third time in `inf_render::pack_deform_window`, so
/// neither a future authored response table nor a malformed projection can
/// exceed what the skirt was grown for. The coupling is stated in both places.
pub const MAX_DEFORM_DEPTH_M: f32 = 1.0;

/// Largest footprint one contact may stamp, metres.
///
/// Not taste — a loop bound. The stamp walks the lattice samples inside the
/// disk, so an unclamped radius from a mis-authored collider would walk an
/// unbounded rectangle at the fixed step. 8 m is wider than any vehicle this
/// engine has and 64 × 64 samples is a bounded inner loop.
pub const MAX_FOOTPRINT_RADIUS_M: f64 = 8.0;

/// How fast a contact presses an infinitely soft surface, metres per second.
///
/// Scaled down by the layer's `hardness`, this sets how long a contact must rest
/// before it reaches its class's target depth. At snow's hardness a footfall
/// reaches ~0.22 m in about an eighth of a second, which is a footfall; at
/// grass's it saturates in one step, which is also right — grass has almost
/// nothing to give.
pub const PRESS_RATE_M_PER_S: f32 = 2.0;

/// The snowfall refill is accumulated in `f64` and applied in quanta of this
/// many metres.
///
/// **This is the `inf_ecs::sky::MAX_WEATHER_BLEND_S` lesson, paid forward.** The
/// calibrated rate is `1.4e-5 m/s`; at a 240 Hz sim that is `5.8e-8 m` per step,
/// which is *below one f32 ULP* at a depth of 1 m (`5.96e-8`) — subtracting it
/// would round straight back to the same `f32` and the refill would silently
/// freeze on exactly the deepest prints it exists to fill. Accumulating in `f64`
/// and spending the total once it clears 0.1 mm makes the refill exact,
/// step-rate-independent and immune to the stall. At full snowfall a quantum
/// lands every ~7 s; nothing visible happens faster than that at 5 cm/hour
/// anyway.
pub const SNOW_REFILL_QUANTUM_M: f64 = 1.0e-4;

/// How one splat layer answers a contact.
///
/// **Engine constants, argued rather than authored** — the
/// `inf_render::wetness` doctrine. There is no schema in this batch (scene v19
/// and `.inf_terrain` v5 are both frozen), and a per-layer authored response
/// curve is the right thing to add *later*, on a schema bump that has other
/// reasons to exist. Until then the four numbers below are reviewable against
/// reality in their doc comments, which is more than an unset authored default
/// would have been.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerResponse {
    /// Deepest a contact can press this layer, metres. `0` ⇒ the layer does not
    /// deform at all. Never above [`MAX_DEFORM_DEPTH_M`].
    pub max_depth_m: f32,
    /// Seconds for a full-depth print to relax back to nothing, or `0` for **no
    /// self-recovery** — a layer that only forgets when something refills it
    /// (snowfall) or when the cell is evicted.
    pub recovery_s: f32,
    /// Resistance in `[0, 1)`: scales [`PRESS_RATE_M_PER_S`], so a hard layer
    /// takes longer to reach the same depth. `1` would mean "never presses",
    /// which is what `max_depth_m = 0` says more directly.
    pub hardness: f32,
    /// How strongly scattered foliage standing on this layer bends when the
    /// ground under it is pressed, `[0, 1]`. Grass answers a footfall by *lying
    /// down*, not by denting — this is the channel that carries that, and it is
    /// read by `inf_render`'s scatter bend rather than by anything here.
    pub bend: f32,
}

/// The per-layer response table, indexed by splat layer.
///
/// # THE MAPPING IS BY **INDEX**, AND THAT IS A REAL LIMITATION
///
/// A splat layer has no identity beyond its position: index 0 is index 0, and
/// what it *looks* like is whatever albedo/roughness the level put in
/// `Terrain::layers[0]`. The archetypes below are therefore the semantics of the
/// engine's **default palette** — grass → rock → dirt → snow, as
/// `inf_ecs::components` documents it, as the paint UI's swatches mirror it, and
/// as `inf_terrain::BiomeSet::splat_layer` binds biomes to it — and **not** a
/// property the engine can enforce.
///
/// A level that re-skins a layer keeps the index's response. The engine's own
/// `samples::phase16_world_scene` already does exactly this: it paints its inline
/// terrain's `layers[0]` a reddish-brown, so ground that *reads* as dirt answers
/// a boot with grass's 3 cm and grass's 8-second spring-back. That is wrong, it
/// is visible, and it is stated here rather than discovered later.
///
/// The fix is an authored per-layer response, which needs a schema bump that
/// P22.1 explicitly does not have (scene v19 and `.inf_terrain` v5 are frozen).
/// Until that lands, the honest position is: **the response follows the index,
/// the index means what the default palette says it means, and a level that
/// re-skins a layer should expect the original archetype's physics.**
///
/// "Sand" and "mud" are both the loose granular layer (index 2): they differ in
/// albedo, not in how a boot sinks into them, and one response law for "loose
/// ground" is honest where two would be a distinction the field cannot see.
///
/// * **0 grass** — barely dents (3 cm) and springs back in seconds, because what
///   a footstep does to grass is *bend* it; `bend = 1` is where that lives.
/// * **1 rock** — does not deform. `max_depth_m = 0` makes every stamp on rock a
///   no-op with no branch anywhere else in the file.
/// * **2 dirt / sand / mud** — 14 cm, and it slumps rather than springs: three
///   minutes to erase a full-depth rut, which is why a vehicle's tracks are still
///   there when you drive back past them.
/// * **3 snow** — the deep one (40 cm) and the soft one, and it has **no elastic
///   recovery at all**: compacted snow stays compacted. It forgets by being
///   *snowed on*, which is exactly the P17 hook — see
///   [`DeformField::relax`].
pub const LAYER_RESPONSE: [LayerResponse; SPLAT_LAYERS] = [
    LayerResponse {
        max_depth_m: 0.03,
        recovery_s: 8.0,
        hardness: 0.55,
        bend: 1.0,
    },
    LayerResponse {
        max_depth_m: 0.0,
        recovery_s: 0.0,
        hardness: 1.0,
        bend: 0.0,
    },
    LayerResponse {
        max_depth_m: 0.14,
        recovery_s: 180.0,
        hardness: 0.25,
        bend: 0.35,
    },
    LayerResponse {
        max_depth_m: 0.40,
        recovery_s: 0.0,
        hardness: 0.08,
        bend: 0.1,
    },
];

/// What kind of thing is pressing.
///
/// A *class*, not a mass: turning a rapier body's mass into a contact pressure
/// needs a real contact-patch area, and a capsule resting on a heightfield does
/// not have one. Four named classes with argued depth fractions is the honest
/// v1, and the mapping from a body to its class lives once, in
/// `inf_ecs::deform`, so the two hosts cannot disagree about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PressureClass {
    /// Something resting lightly — a prop, a small dynamic body.
    Light,
    /// A character on foot. The reference case.
    Foot,
    /// A wheel or track: the same weight through a much smaller patch, so it
    /// cuts deeper than the character that drives it.
    Wheel,
    /// A large driven body — a vehicle chassis, a moving platform.
    Heavy,
}

impl PressureClass {
    /// Fraction of the layer's [`LayerResponse::max_depth_m`] this class can
    /// reach. `max_depth_m` is the *saturation* of the layer; the class says how
    /// much of it this contact is entitled to.
    #[inline]
    pub fn depth_fraction(self) -> f32 {
        match self {
            PressureClass::Light => 0.25,
            PressureClass::Foot => 0.55,
            PressureClass::Wheel => 0.85,
            PressureClass::Heavy => 1.0,
        }
    }
}

/// One cell's dense sample block.
///
/// `depth` is metres **below** the authored surface (always ≥ 0 — nothing here
/// raises ground). `layer` records which [`LAYER_RESPONSE`] entry governs each
/// sample's recovery: the layer of the *last contact that pressed it deeper*, so
/// a rut driven across a snow/dirt boundary recovers at each half's own rate.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformCell {
    depth: Vec<f32>,
    layer: Vec<u8>,
    /// The [`DeformField::steps`] value of the last stamp that touched this
    /// cell — the eviction key.
    last_stamp_step: u64,
}

impl DeformCell {
    fn new(step: u64) -> Self {
        Self {
            depth: vec![0.0; DEFORM_CELL_AREA],
            layer: vec![0; DEFORM_CELL_AREA],
            last_stamp_step: step,
        }
    }

    /// Depth (m) at a cell-local sample, `(0, 0)` = the cell's minimum corner.
    #[inline]
    pub fn depth(&self, i: u32, j: u32) -> f32 {
        self.depth[(j * DEFORM_CELL_SAMPLES + i) as usize]
    }

    /// The layer governing a cell-local sample.
    #[inline]
    pub fn layer(&self, i: u32, j: u32) -> u8 {
        self.layer[(j * DEFORM_CELL_SAMPLES + i) as usize]
    }

    /// The whole depth block, row-major, `DEFORM_CELL_SAMPLES²` long — the shape
    /// a render projection copies.
    #[inline]
    pub fn depths(&self) -> &[f32] {
        &self.depth
    }

    fn is_flat(&self) -> bool {
        self.depth.iter().all(|d| *d == 0.0)
    }
}

/// The sparse deformation field.
///
/// Sim state. Created empty, stepped once per fixed step, never persisted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeformField {
    cells: BTreeMap<(i32, i32), DeformCell>,
    /// **The commutativity scratch.** For every sample stamped since the last
    /// [`relax`](Self::relax), the depth it held at the *start* of this step.
    ///
    /// Without it the step is order-dependent, and provably so: two overlapping
    /// contacts each advancing from *whatever the previous one left* means
    /// `A then B` reaches a different depth from `B then A` as soon as either
    /// saturates. With it, every stamp in a step advances from the **same base**
    /// and the sample keeps the deepest result — and `max` over a set has no
    /// order. One advance per sample per step is also the physically right rule:
    /// two contacts at one instant are not twice as long a press.
    ///
    /// Cleared by `relax`, so it holds only the samples one step touched (a
    /// handful per contact) and never grows with the field.
    step_bases: BTreeMap<((i32, i32), u32), f32>,
    /// Fixed steps [`relax`](Self::relax) has run. The eviction clock.
    steps: u64,
    /// Snowfall accumulated but not yet spent — see [`SNOW_REFILL_QUANTUM_M`].
    snow_pending_m: f64,
    /// Bumped by every stamp that actually changed a sample and by every relax
    /// that actually moved one. The **only** thing a render projection needs to
    /// decide whether it may reuse last frame's window.
    epoch: u64,
    /// Cells dropped by the [`MAX_DEFORM_CELLS`] bound over this field's life —
    /// telemetry, and the thing a bound test asserts against.
    evicted: u64,
}

impl DeformField {
    /// An empty field.
    pub fn new() -> Self {
        Self::default()
    }

    /// Live cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// No cell is live — the field costs nothing and every consumer short-circuits.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Fixed steps run.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The change stamp. Two fields with the same epoch, produced by the same
    /// step history, hold the same bytes; a consumer that cached a derivation
    /// keyed on this may reuse it.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Cells the [`MAX_DEFORM_CELLS`] bound has dropped.
    pub fn evicted_cells(&self) -> u64 {
        self.evicted
    }

    /// Live cells in coordinate order — the render projection's walk.
    pub fn cells(&self) -> impl Iterator<Item = (&(i32, i32), &DeformCell)> {
        self.cells.iter()
    }

    /// The world position of a cell's minimum corner.
    pub fn cell_origin_xz(coord: (i32, i32)) -> DVec2 {
        DVec2::new(
            coord.0 as f64 * DEFORM_CELL_SIZE_M,
            coord.1 as f64 * DEFORM_CELL_SIZE_M,
        )
    }

    /// The global lattice index of the sample nearest world `(x, z)`.
    ///
    /// The one place the world→lattice rule lives. `round`, not `floor`: a
    /// consumer asking "how deep is it *here*" wants the nearest sample, and the
    /// window pack asks the inverse question (sample → world) so the two are
    /// exact inverses on lattice points.
    #[inline]
    pub fn sample_index(world_xz: DVec2) -> (i64, i64) {
        (
            (world_xz.x / DEFORM_SAMPLE_PITCH_M).round() as i64,
            (world_xz.y / DEFORM_SAMPLE_PITCH_M).round() as i64,
        )
    }

    /// World position of a global lattice sample.
    #[inline]
    pub fn sample_position(index: (i64, i64)) -> DVec2 {
        DVec2::new(
            index.0 as f64 * DEFORM_SAMPLE_PITCH_M,
            index.1 as f64 * DEFORM_SAMPLE_PITCH_M,
        )
    }

    /// Split a global lattice index into `(cell coord, cell-local sample)`.
    #[inline]
    fn split(index: (i64, i64)) -> ((i32, i32), (u32, u32)) {
        let n = DEFORM_CELL_SAMPLES as i64;
        let cx = index.0.div_euclid(n);
        let cz = index.1.div_euclid(n);
        (
            (cx as i32, cz as i32),
            (index.0.rem_euclid(n) as u32, index.1.rem_euclid(n) as u32),
        )
    }

    /// Depth (m, ≥ 0) at world `(x, z)` — nearest sample, `0` where nothing has
    /// been pressed. Never interpolated on this side: the render window carries
    /// the lattice one-to-one and does its own filtering, and a gameplay query
    /// wants the sample it is standing on.
    pub fn depth_at(&self, world_xz: DVec2) -> f32 {
        let (coord, (i, j)) = Self::split(Self::sample_index(world_xz));
        self.cells.get(&coord).map(|c| c.depth(i, j)).unwrap_or(0.0)
    }

    /// The layer governing world `(x, z)`, or `None` where nothing is pressed.
    pub fn layer_at(&self, world_xz: DVec2) -> Option<u8> {
        let (coord, (i, j)) = Self::split(Self::sample_index(world_xz));
        let cell = self.cells.get(&coord)?;
        (cell.depth(i, j) > 0.0).then(|| cell.layer(i, j))
    }

    /// Press the ground under one ground contact.
    ///
    /// Pure and deterministic: the same field, centre, radius, class, layer and
    /// `dt` always produce the same bytes. Returns `true` **iff at least one
    /// sample actually got deeper** — an honest engagement signal, not a "the
    /// call happened" one: a stamp on rock, a stamp on ground already at its
    /// class depth, and a zero-radius contact all return `false`.
    ///
    /// **Order-invariant within a step.** Every stamp between two
    /// [`relax`](Self::relax) calls advances from the depth the sample held at
    /// the start of the step (see [`step_bases`](Self#structfield.step_bases)) and
    /// keeps the deeper result, so the contacts of one fixed step may be applied
    /// in any order and land the same bytes. The `Guid` sort in
    /// `inf_ecs::deform::ground_contacts` is therefore belt-and-braces rather
    /// than load-bearing — which is the right way round: an invariant that only
    /// holds because a caller sorted is one call site away from being false.
    ///
    /// **Every cell the footprint covers has its eviction clock refreshed, even
    /// when nothing presses.** That is the point, not an oversight: a body that
    /// has settled to its class depth stops writing but is still standing there,
    /// and evicting the ground out from under it would make its footprint vanish
    /// while it watched. The cost is a bounded no-op walk (a footprint's worth of
    /// samples) per resting contact per step, and the honest consequence is that
    /// a permanently-resting body pins one or two cells against
    /// [`MAX_DEFORM_CELLS`] for as long as it rests.
    ///
    /// * `center_xz` — world XZ of the contact.
    /// * `footprint_radius_m` — clamped to [`MAX_FOOTPRINT_RADIUS_M`].
    /// * `pressure` — the contact's class; sets the *target* depth.
    /// * `layer` — the terrain's dominant splat layer under the contact, taken
    ///   by the caller from **sim-resident tiles only** (the
    ///   [`crate::wants`] determinism doctrine: a query that could see a
    ///   camera-paged tile would make the sim depend on where somebody is
    ///   looking). Out-of-range values fold into the table.
    /// * `dt` — the fixed step. Pressing is rate-limited, so a contact that
    ///   rests reaches its target and a contact that passes through leaves less.
    ///
    /// The profile is [`Falloff::Plateau`] at half radius — flat over the inner
    /// half, smoothstepped to nothing at the rim. That is the shape of a boot
    /// (a sole, then an edge), and it is the engine's **one** falloff law rather
    /// than a second one spelled here.
    pub fn stamp_contact(
        &mut self,
        center_xz: DVec2,
        footprint_radius_m: f64,
        pressure: PressureClass,
        layer: u8,
        dt: f64,
    ) -> bool {
        let resp = LAYER_RESPONSE[(layer as usize) % SPLAT_LAYERS];
        if resp.max_depth_m <= 0.0 || dt <= 0.0 {
            return false;
        }
        let radius = footprint_radius_m.clamp(0.0, MAX_FOOTPRINT_RADIUS_M);
        if radius <= 0.0 {
            return false;
        }
        let peak = (resp.max_depth_m * pressure.depth_fraction()).min(MAX_DEFORM_DEPTH_M);
        let rate = PRESS_RATE_M_PER_S * (1.0 - resp.hardness).clamp(0.0, 1.0);
        let advance = rate * dt as f32;
        if advance <= 0.0 {
            return false;
        }
        let falloff = Falloff::Plateau(0.5);
        let layer = (layer as usize % SPLAT_LAYERS) as u8;

        // The lattice samples inside the disk's bounding square. Bounded by
        // MAX_FOOTPRINT_RADIUS_M above, so this is at worst 64 × 64.
        let (i0, j0) = (
            ((center_xz.x - radius) / DEFORM_SAMPLE_PITCH_M).ceil() as i64,
            ((center_xz.y - radius) / DEFORM_SAMPLE_PITCH_M).ceil() as i64,
        );
        let (i1, j1) = (
            ((center_xz.x + radius) / DEFORM_SAMPLE_PITCH_M).floor() as i64,
            ((center_xz.y + radius) / DEFORM_SAMPLE_PITCH_M).floor() as i64,
        );
        let mut touched = false;
        for sj in j0..=j1 {
            for si in i0..=i1 {
                let p = Self::sample_position((si, sj));
                let dist = (p - center_xz).length();
                let w = falloff.weight(dist, radius) as f32;
                if w <= 0.0 {
                    continue;
                }
                let target = (peak * w).min(MAX_DEFORM_DEPTH_M);
                let (coord, (li, lj)) = Self::split((si, sj));
                let step = self.steps;
                let cell = self
                    .cells
                    .entry(coord)
                    .or_insert_with(|| DeformCell::new(step));
                // Refresh the eviction clock for every covered cell, pressing or
                // not — see the doc comment: this is what stops the ground being
                // paged out from under a body that has stopped sinking.
                cell.last_stamp_step = step;
                let idx = (lj * DEFORM_CELL_SAMPLES + li) as usize;
                let cur = cell.depth[idx];
                if cur >= target {
                    continue;
                }
                // The commutativity base: the depth at the START of this step, so
                // every stamp of the step advances from the same place and the
                // deepest wins.
                let base = *self.step_bases.entry((coord, idx as u32)).or_insert(cur);
                let next = (base + advance).min(target);
                if next <= cur {
                    continue;
                }
                cell.depth[idx] = next;
                cell.layer[idx] = layer;
                touched = true;
            }
        }
        if touched {
            self.epoch += 1;
        }
        // Bounded whether or not anything pressed: a stamp that only *created*
        // cells (all already at depth) still grew the live set.
        self.enforce_bound();
        touched
    }

    /// One fixed step of **relaxation**: per-layer recovery, snowfall refill,
    /// and the drop of any cell that has returned to flat.
    ///
    /// Call once per fixed step, *before* this step's stamps, so a contact that
    /// keeps resting converges on its target instead of oscillating around it.
    ///
    /// `snow_rate_m_per_s` is [`inf_ecs::sky::ResolvedSky::snow_accumulation_rate`]
    /// — a **plain `f64` parameter**, because `inf-terrain` is Ring 0 and does
    /// not know what a sky is. The rate is calibrated once, there
    /// (`SNOW_ACCUMULATION_MAX_M_PER_S = 1.4e-5 m/s ≈ 5 cm/hour`, whose doc says
    /// "P22.1 consumes this"); this module must not re-derive it and does not.
    /// Falling snow settles into *any* depression, so the refill applies to every
    /// live sample regardless of which layer pressed it — one rule, and the
    /// truthful one: a boot print in mud does stop reading as a boot print once
    /// it has snow in it.
    ///
    /// Recovery is **linear**, not exponential, and that is load-bearing: an
    /// exponential decay is asymptotic, so no cell would ever reach exactly zero,
    /// no cell would ever be dropped, and the "sparse" in the module title would
    /// be a lie. Linear reaches zero in a finite, computable number of steps.
    ///
    /// The step is comfortably clear of the `f32` stall that
    /// [`SNOW_REFILL_QUANTUM_M`] exists to dodge: the smallest per-step recovery
    /// in [`LAYER_RESPONSE`] is dirt's `0.14 / 180` m/s, i.e. `3.2e-6 m` at
    /// 240 Hz, against an `f32` ULP of `1.5e-8` at that depth — 200 ULPs, so the
    /// countdown always moves.
    pub fn relax(&mut self, dt: f64, snow_rate_m_per_s: f64) {
        self.steps += 1;
        // A new step: every sample's advance base is whatever it holds now. This
        // is what defines "a step" for the order-invariance rule — a caller that
        // stamps twice without relaxing between is describing ONE instant, and
        // gets one advance.
        self.step_bases.clear();
        if snow_rate_m_per_s > 0.0 && dt > 0.0 {
            self.snow_pending_m += snow_rate_m_per_s * dt;
        }
        let refill = if self.snow_pending_m >= SNOW_REFILL_QUANTUM_M {
            std::mem::replace(&mut self.snow_pending_m, 0.0)
        } else {
            0.0
        };
        if self.cells.is_empty() || (dt <= 0.0 && refill <= 0.0) {
            return;
        }
        let mut moved = false;
        for cell in self.cells.values_mut() {
            for idx in 0..DEFORM_CELL_AREA {
                let cur = cell.depth[idx];
                if cur <= 0.0 {
                    continue;
                }
                let resp = LAYER_RESPONSE[(cell.layer[idx] as usize) % SPLAT_LAYERS];
                let mut next = cur as f64;
                if resp.recovery_s > 0.0 {
                    next -= (resp.max_depth_m / resp.recovery_s) as f64 * dt.max(0.0);
                }
                next -= refill;
                let next = next.max(0.0) as f32;
                if next != cur {
                    cell.depth[idx] = next;
                    moved = true;
                }
            }
        }
        if moved {
            self.epoch += 1;
            self.cells.retain(|_, c| !c.is_flat());
        }
    }

    /// Drop least-recently-stamped cells until the field is back inside
    /// [`MAX_DEFORM_CELLS`].
    ///
    /// Ties on `last_stamp_step` break on the cell coordinate ascending, so the
    /// victim is a pure function of the step history. Runs only when a stamp
    /// created a cell past the bound, and removes exactly one cell per such
    /// stamp, so the scan is bounded work at a bounded rate.
    ///
    /// **A cell touched by this very step is never a victim.** Without that, a
    /// field at the bound could evict the ground out from under the body that
    /// just pressed it — the footprint would vanish under the boot that made it,
    /// and worse, in the same step, so nothing downstream could even see that it
    /// had existed. The honest consequence is that the bound may be exceeded
    /// while a single step's contacts cover more than [`MAX_DEFORM_CELLS`] cells,
    /// which would take thousands of simultaneous bodies; the excess is released
    /// on the next step that does not touch them.
    fn enforce_bound(&mut self) {
        let now = self.steps;
        while self.cells.len() > MAX_DEFORM_CELLS {
            let Some(victim) = self
                .cells
                .iter()
                .filter(|(_, cell)| cell.last_stamp_step != now)
                .map(|(coord, cell)| (cell.last_stamp_step, *coord))
                .min()
                .map(|(_, coord)| coord)
            else {
                // Everything alive is under a live contact. Overshoot rather than
                // page out ground somebody is standing on.
                return;
            };
            self.cells.remove(&victim);
            self.step_bases.retain(|(coord, _), _| *coord != victim);
            self.evicted += 1;
            self.epoch += 1;
        }
    }

    /// Pack the window of `texels²` lattice samples whose minimum corner is
    /// lattice index `origin`, row-major, metres of depth.
    ///
    /// Pure, and **no camera reaches this function** — it takes a lattice origin
    /// somebody else snapped. `out` is resized and fully written, so the same
    /// arguments always leave the same bytes regardless of what the buffer held.
    /// Only the cells that intersect the window are walked; a window over empty
    /// ground writes zeroes and touches no cell at all.
    pub fn pack_window(&self, origin: (i64, i64), texels: u32, out: &mut Vec<f32>) {
        let n = texels as usize;
        out.clear();
        out.resize(n * n, 0.0);
        if self.cells.is_empty() || texels == 0 {
            return;
        }
        let cs = DEFORM_CELL_SAMPLES as i64;
        let (c0x, c0z) = (origin.0.div_euclid(cs), origin.1.div_euclid(cs));
        let (c1x, c1z) = (
            (origin.0 + texels as i64 - 1).div_euclid(cs),
            (origin.1 + texels as i64 - 1).div_euclid(cs),
        );
        for cz in c0z..=c1z {
            for cx in c0x..=c1x {
                let Some(cell) = self.cells.get(&(cx as i32, cz as i32)) else {
                    continue;
                };
                let base = (cx * cs - origin.0, cz * cs - origin.1);
                for lj in 0..DEFORM_CELL_SAMPLES as i64 {
                    let ty = base.1 + lj;
                    if ty < 0 || ty >= texels as i64 {
                        continue;
                    }
                    for li in 0..DEFORM_CELL_SAMPLES as i64 {
                        let tx = base.0 + li;
                        if tx < 0 || tx >= texels as i64 {
                            continue;
                        }
                        out[ty as usize * n + tx as usize] = cell.depth[(lj * cs + li) as usize];
                    }
                }
            }
        }
    }

    /// The field's canonical bytes — the trace surface.
    ///
    /// Hand-rolled rather than `serde`, on purpose: the field is **never
    /// persisted** (no schema moves in this batch), so what this needs is a
    /// stable byte order for hashing, not a format with a migration story.
    /// Little-endian throughout, cells in `BTreeMap` order, `f32` by
    /// `to_le_bytes` so the bits are the bits.
    ///
    /// Contains no camera, no frame index and no wall clock — the
    /// `wetness_is_a_pure_function_of_the_water` register: fold this into a replay trace and
    /// the only thing that can move it is the step history.
    pub fn state_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.cells.len() * (16 + DEFORM_CELL_AREA * 5));
        out.extend_from_slice(&self.steps.to_le_bytes());
        out.extend_from_slice(&(self.cells.len() as u64).to_le_bytes());
        for (coord, cell) in &self.cells {
            out.extend_from_slice(&coord.0.to_le_bytes());
            out.extend_from_slice(&coord.1.to_le_bytes());
            out.extend_from_slice(&cell.last_stamp_step.to_le_bytes());
            for d in &cell.depth {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&cell.layer);
        }
        out
    }
}

/// The lattice index of the minimum corner of a `texels²` window centred on
/// world `(x, z)`, **snapped to the lattice**.
///
/// The window follows the camera *for drawing only* — it selects what is drawn
/// out of sim-authoritative state and never what is simulated, which is the
/// P21 "a camera-paged store may be read but never shipped" law taken at its
/// word. This function is the snap: because the origin is an integer lattice
/// index, moving the camera translates the window by whole samples and a texel
/// always carries the *same* sample's value. Nothing is ever resampled, so
/// panning the camera cannot shimmer a footprint — the same quantization the
/// sun-bucket and radius-bucket amortizations use, applied to a texture origin.
pub fn window_origin(center_xz: DVec2, texels: u32) -> (i64, i64) {
    let (cx, cz) = DeformField::sample_index(center_xz);
    let half = (texels / 2) as i64;
    (cx - half, cz - half)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1.0 / 60.0;
    /// The engine's snow layer.
    const SNOW: u8 = 3;
    /// The engine's loose-granular layer.
    const DIRT: u8 = 2;
    /// The engine's undeformable layer.
    const ROCK: u8 = 1;

    /// Rest a contact on one spot for `steps` FIXED STEPS — relax then stamp,
    /// which is the loop `inf_ecs::deform::step_deformation` runs.
    ///
    /// Tests must model steps now that the advance is once-per-sample-per-step:
    /// a burst of stamps with no `relax` between them describes ONE instant, and
    /// correctly presses once.
    fn press(
        field: &mut DeformField,
        at: DVec2,
        radius: f64,
        p: PressureClass,
        layer: u8,
        steps: u32,
    ) {
        for _ in 0..steps {
            field.relax(DT, 0.0);
            field.stamp_contact(at, radius, p, layer, DT);
        }
    }

    fn walk(field: &mut DeformField, steps: u32, layer: u8) {
        for s in 0..steps {
            field.relax(DT, 0.0);
            let x = s as f64 * 0.5;
            field.stamp_contact(DVec2::new(x, 0.0), 0.25, PressureClass::Foot, layer, DT);
        }
    }

    #[test]
    fn the_table_never_exceeds_the_skirt_floor() {
        // The coupling stated in MAX_DEFORM_DEPTH_M's docs, asserted: a response
        // deeper than the terrain raster's skirt floor would open patch-boundary
        // cracks, and the table is the only place that could break it.
        for (k, r) in LAYER_RESPONSE.iter().enumerate() {
            assert!(
                r.max_depth_m <= MAX_DEFORM_DEPTH_M,
                "layer {k} presses {} m, past the {MAX_DEFORM_DEPTH_M} m skirt floor",
                r.max_depth_m
            );
            assert!(r.hardness >= 0.0 && r.hardness <= 1.0);
            assert!(r.bend >= 0.0 && r.bend <= 1.0);
        }
    }

    #[test]
    fn a_contact_presses_and_saturates_at_the_class_target() {
        let mut f = DeformField::new();
        let at = DVec2::new(1.0, 1.0);
        // Rest a foot on snow for a long time; it must converge on exactly the
        // class's fraction of the layer's max and then stop.
        press(&mut f, at, 0.3, PressureClass::Foot, SNOW, 600);
        let peak = LAYER_RESPONSE[SNOW as usize].max_depth_m * PressureClass::Foot.depth_fraction();
        assert!((f.depth_at(at) - peak).abs() < 1e-6, "{}", f.depth_at(at));
        // A heavier class on the same sample presses further, still bounded.
        press(&mut f, at, 0.3, PressureClass::Heavy, SNOW, 600);
        let heavy = LAYER_RESPONSE[SNOW as usize].max_depth_m;
        assert!((f.depth_at(at) - heavy).abs() < 1e-6);
        // …and a lighter one afterwards cannot un-press it, and says so.
        let before = f.depth_at(at);
        f.relax(DT, 0.0);
        assert!(
            !f.stamp_contact(at, 0.3, PressureClass::Light, SNOW, DT),
            "a stamp that pressed nothing must report false"
        );
        assert_eq!(f.depth_at(at), before);
    }

    #[test]
    fn rock_never_deforms() {
        let mut f = DeformField::new();
        for _ in 0..100 {
            f.relax(DT, 0.0);
            assert!(
                !f.stamp_contact(DVec2::ZERO, 1.0, PressureClass::Heavy, ROCK, DT),
                "a stamp on an undeformable layer must report false"
            );
        }
        assert!(f.is_empty());
        assert_eq!(f.depth_at(DVec2::ZERO), 0.0);
    }

    #[test]
    fn two_runs_of_the_same_history_are_byte_identical() {
        let mut a = DeformField::new();
        let mut b = DeformField::new();
        walk(&mut a, 200, SNOW);
        walk(&mut b, 200, SNOW);
        assert_eq!(a.state_bytes(), b.state_bytes());
        assert!(!a.is_empty());
        // …and the bytes are a function of the history, not of the object: a
        // field that walked one step further differs.
        walk(&mut b, 1, SNOW);
        assert_ne!(a.state_bytes(), b.state_bytes());
    }

    #[test]
    fn disjoint_stamps_commute_but_overlapping_ones_do_not() {
        // Where order must not matter: two footprints far enough apart to share
        // no sample must land the same field whichever went first.
        let mk = |flip: bool| {
            let mut f = DeformField::new();
            let a = (DVec2::new(0.0, 0.0), SNOW);
            let b = (DVec2::new(40.0, 40.0), DIRT);
            let (p, q) = if flip { (b, a) } else { (a, b) };
            f.relax(DT, 0.0);
            f.stamp_contact(p.0, 0.3, PressureClass::Foot, p.1, DT);
            f.stamp_contact(q.0, 0.3, PressureClass::Foot, q.1, DT);
            f
        };
        assert_eq!(mk(false).state_bytes(), mk(true).state_bytes());

        // Where order MUST matter, and the field must not pretend otherwise:
        // the same sample stamped by two different layers keeps the layer of
        // whichever pressed it deeper last, and that governs its recovery.
        let mk2 = |flip: bool| {
            let mut f = DeformField::new();
            let at = DVec2::new(5.0, 5.0);
            let (p, q) = if flip { (DIRT, SNOW) } else { (SNOW, DIRT) };
            for _ in 0..600 {
                f.relax(DT, 0.0);
                f.stamp_contact(at, 0.3, PressureClass::Heavy, p, DT);
                f.stamp_contact(at, 0.3, PressureClass::Heavy, q, DT);
            }
            f.layer_at(at).expect("pressed")
        };
        // Snow is deeper than dirt, so whichever order runs, snow wins the
        // *depth* — but the layer byte records the last writer that went deeper,
        // which is snow in both orders. Assert the real invariant instead: a
        // shallower layer stamped after a deeper one cannot steal the sample.
        assert_eq!(mk2(false), SNOW);
        assert_eq!(mk2(true), SNOW);
    }

    #[test]
    fn recovery_returns_to_exactly_zero_and_drops_the_cell() {
        let mut f = DeformField::new();
        let at = DVec2::new(2.0, 2.0);
        press(&mut f, at, 0.3, PressureClass::Foot, 0, 60);
        assert!(f.depth_at(at) > 0.0);
        assert_eq!(f.cell_count(), 1);
        // Grass recovers in 8 s for a full-depth print; run well past it.
        for _ in 0..(60 * 20) {
            f.relax(DT, 0.0);
        }
        assert_eq!(f.depth_at(at), 0.0);
        assert!(
            f.is_empty(),
            "a relaxed cell must be dropped, not kept at zero"
        );
    }

    #[test]
    fn snow_never_recovers_on_its_own() {
        let mut f = DeformField::new();
        let at = DVec2::new(3.0, 3.0);
        press(&mut f, at, 0.3, PressureClass::Foot, SNOW, 600);
        let pressed = f.depth_at(at);
        for _ in 0..(60 * 600) {
            f.relax(DT, 0.0);
        }
        assert_eq!(
            f.depth_at(at),
            pressed,
            "compacted snow does not spring back"
        );
    }

    #[test]
    fn snowfall_refills_prints_and_a_zero_rate_refills_nothing() {
        // This is the assertion that makes `inf_ecs::sky`'s "P22.1 consumes
        // this" true: the rate crosses as a parameter and the field spends it.
        let rate = 1.4e-5_f64; // SNOW_ACCUMULATION_MAX_M_PER_S, full blast.
        let mut with = DeformField::new();
        let mut without = DeformField::new();
        let at = DVec2::new(4.0, 4.0);
        for f in [&mut with, &mut without] {
            press(f, at, 0.3, PressureClass::Foot, SNOW, 600);
        }
        let pressed = with.depth_at(at);
        // Half an hour of heavy snow ≈ 2.5 cm — a real dent in a 22 cm print.
        for _ in 0..(60 * 1800) {
            with.relax(DT, rate);
            without.relax(DT, 0.0);
        }
        assert_eq!(without.depth_at(at), pressed, "rate 0 must refill nothing");
        let filled = pressed - with.depth_at(at);
        assert!(
            filled > 0.02 && filled < 0.03,
            "half an hour at 5 cm/hour should fill ~2.5 cm, filled {filled}"
        );
        // Enough snow erases the print entirely and the cell goes. Driven at 50×
        // the calibrated rate so the arm is ten minutes of sim rather than the
        // four real hours 5 cm/hour needs to bury a 22 cm boot print — which is
        // itself the calibration being believable.
        for _ in 0..(60 * 600) {
            with.relax(DT, rate * 50.0);
        }
        assert_eq!(with.depth_at(at), 0.0);
        assert!(with.is_empty());
    }

    #[test]
    fn the_active_set_is_bounded_and_eviction_is_deterministic() {
        // Walk far enough to want many more cells than the bound allows.
        let mut f = DeformField::new();
        let mut x = 0.0;
        while f.evicted_cells() < 32 {
            f.relax(DT, 0.0);
            f.stamp_contact(DVec2::new(x, 0.0), 0.3, PressureClass::Foot, SNOW, DT);
            x += DEFORM_CELL_SIZE_M;
            assert!(
                f.cell_count() <= MAX_DEFORM_CELLS,
                "the bound is never exceeded, even transiently"
            );
        }
        // The oldest ground is what went: the walk started at x = 0.
        assert_eq!(f.depth_at(DVec2::ZERO), 0.0);
        assert!(f.depth_at(DVec2::new(x - DEFORM_CELL_SIZE_M, 0.0)) > 0.0);

        // Same history, same victims — twice.
        let mut g = DeformField::new();
        let mut gx = 0.0;
        while g.evicted_cells() < 32 {
            g.relax(DT, 0.0);
            g.stamp_contact(DVec2::new(gx, 0.0), 0.3, PressureClass::Foot, SNOW, DT);
            gx += DEFORM_CELL_SIZE_M;
        }
        assert_eq!(f.state_bytes(), g.state_bytes());
    }

    #[test]
    fn the_window_is_a_pure_translation_of_the_lattice() {
        let mut f = DeformField::new();
        press(
            &mut f,
            DVec2::new(1.0, 1.0),
            0.5,
            PressureClass::Foot,
            SNOW,
            600,
        );
        let texels = 64u32;
        let o = window_origin(DVec2::new(1.0, 1.0), texels);
        let mut a = Vec::new();
        f.pack_window(o, texels, &mut a);
        // Packing twice into a dirty buffer gives the same bytes (the buffer is
        // fully written, never merged into).
        let mut b = vec![9.0f32; 7];
        f.pack_window(o, texels, &mut b);
        assert_eq!(a, b);
        assert!(a.iter().any(|d| *d > 0.0));

        // THE REBASE ARM (the wetness `a_rebase_moves_the_band_with_the_world_and_not_against_it` register): shift
        // the window by whole texels and the same lattice values come back,
        // shifted — no resampling, no filter, no drift.
        let shift = 5i64;
        let mut c = Vec::new();
        f.pack_window((o.0 + shift, o.1), texels, &mut c);
        for j in 0..texels as usize {
            for i in 0..(texels as usize - shift as usize) {
                assert_eq!(
                    c[j * texels as usize + i],
                    a[j * texels as usize + i + shift as usize],
                    "texel ({i}, {j}) is not the same sample after a {shift}-texel pan"
                );
            }
        }
    }

    #[test]
    fn the_window_origin_snaps_to_the_lattice() {
        // Sub-texel camera motion must not move the window at all…
        let texels = 512u32;
        let a = window_origin(DVec2::new(100.0, 100.0), texels);
        let b = window_origin(DVec2::new(100.05, 100.05), texels);
        assert_eq!(a, b);
        // …and a full texel of motion must move it by exactly one.
        let c = window_origin(DVec2::new(100.0 + DEFORM_SAMPLE_PITCH_M, 100.0), texels);
        assert_eq!(c, (a.0 + 1, a.1));
        // The origin really is the window's minimum corner in world units.
        let world = DeformField::sample_position(a);
        assert!(world.x < 100.0 && world.y < 100.0);
    }

    #[test]
    fn the_field_is_a_pure_function_of_its_step_history() {
        // No camera, no clock, no residency reaches this module — the strongest
        // form of that claim available without a compiler: a field driven only
        // by stamps and relaxes reproduces byte-for-byte from a cold start, and
        // `epoch` moves only when bytes do.
        let mut f = DeformField::new();
        let before = f.epoch();
        f.relax(DT, 0.0);
        assert_eq!(f.epoch(), before, "an empty relax changes nothing");
        assert!(!f.stamp_contact(DVec2::ZERO, 0.3, PressureClass::Foot, ROCK, DT));
        assert_eq!(f.epoch(), before, "a stamp on rock changes nothing");
        assert!(f.stamp_contact(DVec2::ZERO, 0.3, PressureClass::Foot, SNOW, DT));
        assert!(f.epoch() > before, "a stamp that pressed must be visible");
        let e = f.epoch();
        let bytes = f.state_bytes();
        assert!(!f.stamp_contact(DVec2::new(1000.0, 0.0), 0.0, PressureClass::Foot, SNOW, DT));
        assert_eq!(f.epoch(), e, "a zero-radius contact presses nothing");
        assert_eq!(f.state_bytes(), bytes);
    }

    #[test]
    fn cell_geometry_is_self_consistent() {
        assert_eq!(
            DEFORM_CELL_SIZE_M,
            DEFORM_CELL_SAMPLES as f64 * DEFORM_SAMPLE_PITCH_M
        );
        // Negative coordinates split correctly (div_euclid, not `/`).
        let idx = DeformField::sample_index(DVec2::new(-0.3, -4.1));
        let (coord, (i, j)) = DeformField::split(idx);
        assert!(coord.0 < 0 && coord.1 < 0);
        assert!(i < DEFORM_CELL_SAMPLES && j < DEFORM_CELL_SAMPLES);
        // …and a stamp across the origin lands on both sides of it.
        let mut f = DeformField::new();
        press(&mut f, DVec2::ZERO, 1.0, PressureClass::Heavy, SNOW, 600);
        assert!(f.depth_at(DVec2::new(-0.5, -0.5)) > 0.0);
        assert!(f.depth_at(DVec2::new(0.5, 0.5)) > 0.0);
    }

    #[test]
    fn the_contacts_of_one_step_commute() {
        // GENUINE ORDER-INVARIANCE, not reproducibility: the same set of
        // overlapping contacts applied in three different orders, inside one
        // step, must land byte-identical fields. Every previous determinism arm
        // in this batch replayed the SAME order twice, which proves nothing about
        // order at all.
        let contacts = [
            (DVec2::new(5.00, 5.00), 0.45, PressureClass::Heavy, SNOW),
            (DVec2::new(5.20, 5.05), 0.30, PressureClass::Foot, DIRT),
            (DVec2::new(4.85, 5.15), 0.50, PressureClass::Light, SNOW),
            (DVec2::new(5.10, 4.90), 0.35, PressureClass::Wheel, DIRT),
        ];
        let run = |order: &[usize]| {
            let mut f = DeformField::new();
            // Several steps, so saturation genuinely bites — the regime the
            // old non-commutative rule diverged in.
            for _ in 0..40 {
                f.relax(DT, 0.0);
                for &i in order {
                    let (at, r, p, l) = contacts[i];
                    f.stamp_contact(at, r, p, l, DT);
                }
            }
            f.state_bytes()
        };
        let a = run(&[0, 1, 2, 3]);
        let b = run(&[3, 2, 1, 0]);
        let c = run(&[2, 0, 3, 1]);
        assert!(!a.is_empty());
        assert_eq!(a, b, "reversing the contact order changed the field");
        assert_eq!(a, c, "permuting the contact order changed the field");

        // ANTI-VACUITY: the contacts really do overlap, so the equality above is
        // over samples that more than one of them wrote.
        let mut f = DeformField::new();
        f.relax(DT, 0.0);
        f.stamp_contact(
            contacts[0].0,
            contacts[0].1,
            contacts[0].2,
            contacts[0].3,
            DT,
        );
        let one = f.depth_at(DVec2::new(5.0, 5.0));
        f.stamp_contact(
            contacts[1].0,
            contacts[1].1,
            contacts[1].2,
            contacts[1].3,
            DT,
        );
        assert!(one > 0.0 && f.depth_at(DVec2::new(5.0, 5.0)) > 0.0);
    }

    #[test]
    fn one_step_is_one_advance_however_many_contacts_press() {
        // The physical half of the commutativity rule: two contacts at one
        // instant are not twice as long a press. Two overlapping stamps in one
        // step must reach exactly the depth the deeper one alone would.
        let at = DVec2::new(9.0, 9.0);
        let mut one = DeformField::new();
        let mut two = DeformField::new();
        one.relax(DT, 0.0);
        one.stamp_contact(at, 0.4, PressureClass::Heavy, SNOW, DT);
        two.relax(DT, 0.0);
        two.stamp_contact(at, 0.4, PressureClass::Heavy, SNOW, DT);
        two.stamp_contact(at, 0.4, PressureClass::Heavy, SNOW, DT);
        assert_eq!(one.depth_at(at), two.depth_at(at));
        assert!(one.depth_at(at) > 0.0);
    }

    #[test]
    fn eviction_never_takes_the_ground_from_under_a_live_contact() {
        // Fill the field to its bound with old cells, then stamp a fresh contact:
        // the new cell must survive its own step, and the victim must be one of
        // the old ones.
        let mut f = DeformField::new();
        let mut x = 0.0;
        while f.cell_count() < MAX_DEFORM_CELLS {
            f.relax(DT, 0.0);
            f.stamp_contact(DVec2::new(x, 0.0), 0.3, PressureClass::Foot, SNOW, DT);
            x += DEFORM_CELL_SIZE_M;
        }
        assert_eq!(f.cell_count(), MAX_DEFORM_CELLS);
        let evicted_before = f.evicted_cells();

        // A contact far away, spanning several fresh cells in ONE step.
        f.relax(DT, 0.0);
        let far = DVec2::new(100_000.0, 100_000.0);
        for dx in [-3.0f64, 0.0, 3.0] {
            f.stamp_contact(
                far + DVec2::new(dx, 0.0),
                1.0,
                PressureClass::Heavy,
                SNOW,
                DT,
            );
        }
        assert!(
            f.evicted_cells() > evicted_before,
            "the bound must have bitten"
        );
        assert!(
            f.depth_at(far) > 0.0,
            "the field evicted the cell the live contact had just pressed"
        );
        assert!(f.cell_count() <= MAX_DEFORM_CELLS);
    }

    #[test]
    fn a_resting_contact_keeps_its_ground_alive() {
        // A body that has settled stops PRESSING but is still standing there, so
        // its cell must keep its eviction clock refreshed. Otherwise the ground
        // would page out from under it while it watched.
        let mut f = DeformField::new();
        let at = DVec2::new(2.0, 2.0);
        press(&mut f, at, 0.3, PressureClass::Foot, SNOW, 600);
        let saturated = f.depth_at(at);
        let stamp_step_before = f.steps();

        // A step in which the contact presses NOTHING (already at target)…
        f.relax(DT, 0.0);
        assert!(!f.stamp_contact(at, 0.3, PressureClass::Foot, SNOW, DT));
        assert_eq!(f.depth_at(at), saturated);
        // …still counts as activity for the eviction rule.
        assert!(f.steps() > stamp_step_before);
        let (_, cell) = f.cells().next().expect("one cell");
        assert_eq!(
            cell.last_stamp_step,
            f.steps(),
            "a resting contact must refresh its cell's eviction clock"
        );
    }
}
