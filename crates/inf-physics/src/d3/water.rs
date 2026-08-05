//! **Water volumes, buoyancy, drag and swim mode** (P20.2) — the fixed-step half
//! of the water system.
//!
//! # The sim never reads render state
//!
//! Everything here evaluates from `(WaterBody params, body pose, level clock)`
//! through [`inf_water::WaterSurface::height_at`] — pure `f64`, allocation-free,
//! bit-portable, camera-free, frame-free. That is the P16.3 sim/render split, and
//! it is why a boat floats identically in the editor's Simulate world, in a PIE
//! subprocess and in a shipped build: the *only* input that varies is the level's
//! own clock, which is sim state (`ResolvedSky::cloud_time_s`).
//!
//! There is **one** wave model. The Gerstner sum the renderer uploads and the one
//! sampled here are the same [`inf_water::WaveField`], derived by the same
//! `from_spec`, because both hosts build it from the same `WaterBody` fields.
//! `runtime/inf-player/tests/water_physics.rs` pins that as *bits*, not as
//! duplicated source text.
//!
//! # Where this runs in the fixed step
//!
//! **THE ORDERING LAW.** The force pass runs strictly between the ECS→physics
//! sync and the solver step:
//!
//! ```text
//!   1.  bridge.sync_from_world(world)     ECS → physics; also gathers the water
//!   1b. bridge.apply_water_forces(dt)     ← HERE: buoyancy + drag, events armed
//!   2.  bridge.step(dt)                   the solver consumes the forces
//!   3.  bridge.drain_water_events()       enter / exit / splash, in the same slot
//!                                         as the collision drain
//!   4.  bridge.write_back_into(world)     poses → ECS
//! ```
//!
//! It must be after the sync because a body that moved this step has to be
//! sampled where it *is*, and before the step because rapier clears force
//! accumulators every step — a force applied after one is a force applied to the
//! next. Events are computed during the pass (from pre-step poses, exactly like a
//! contact is detected during the step) and drained in the collision slot, so the
//! fixed step has **one** event slot rather than two.
//!
//! # The off-path discipline
//!
//! A level with no buoyant body pays **one `is_empty()` branch** per fixed step,
//! not one test per rigid body. The gather that feeds it rides inside
//! `sync_from_world`'s existing entity walk, so it adds no second pass over the
//! world either. A furnished town is ~13 000 static colliders and zero `Buoyancy`
//! components, and the water pass never enumerates one of them.

use std::collections::BTreeMap;

use glam::{DAffine3, DQuat, DVec2, DVec3};
use inf_ecs::components::{
    Buoyancy, Spline, SplineInterp as SceneSplineInterp, WaterBody, WaterKind,
};
use inf_water::{RiverPath, RiverProfile, WaterSurface, WaveField, WaveSpec};
use uuid::Uuid;

use super::world::ColliderShape3D;

/// How many points a body is sampled at. **Fixed**, never adaptive: the
/// operation count — and therefore the answer — has to be identical on every
/// machine and at every body size, which is the same reason the Gerstner inverse
/// runs a fixed six iterations instead of a convergence test.
///
/// Four is the smallest count that can express a *tilt*: one point gives a force
/// with no lever arm and therefore no righting moment, and two can only right
/// about one axis.
pub const BUOYANCY_SAMPLES: usize = 4;

/// Vertical speed, m/s, at or above which a surface crossing also reports a
/// [`WaterEventKind3D::Splash`].
///
/// 2 m/s is the speed a body reaches after falling ~20 cm: low enough that a
/// dropped crate splashes, high enough that a boat riding a swell does not. The
/// speed is measured **normal to gravity's up**, so a body skimming the surface
/// horizontally at 30 m/s does not splash — it has not crossed anything.
pub const SPLASH_SPEED_M_S: f64 = 2.0;

/// Submerged fraction at which a character controller starts **swimming**.
///
/// 0.6 means "the water is past your chest": below it a character is wading and
/// should keep its feet, above it there is not enough of it in contact with the
/// ground for walking to mean anything.
pub const SWIM_ENTER_FRACTION: f64 = 0.6;

/// Submerged fraction at which a swimming character goes back to walking.
/// Strictly below [`SWIM_ENTER_FRACTION`] — the hysteresis band is what stops a
/// character standing exactly at chest depth in a rippling lake from flickering
/// between two locomotion modes every fixed step.
pub const SWIM_EXIT_FRACTION: f64 = 0.45;

/// The submerged fraction a swimming character is pulled toward: 0.8, i.e. head
/// out of the water. This is the "mild buoyancy balance" that replaces gravity —
/// a swimmer neither sinks nor pops out.
pub const SWIM_FLOAT_SUBMERSION: f64 = 0.8;

/// How hard a swimmer is pulled to [`SWIM_FLOAT_SUBMERSION`], m/s per unit of
/// fraction error. At fully submerged (`f == 1`) that is a 0.8 m/s rise — a
/// bob back to the surface over about a second, not a cork's pop.
pub const SWIM_FLOAT_RATE_M_S: f64 = 4.0;

/// Horizontal speed cap while swimming, m/s. A fast human swim is ~2 m/s; 2.5
/// leaves room for a game to feel good without letting a character keep its
/// running speed in deep water.
pub const SWIM_SPEED_MAX_M_S: f64 = 2.5;

/// Vertical speed a swimmer can ask for, m/s — the character's own control
/// authority, on top of the buoyancy balance.
pub const SWIM_VERT_MAX_M_S: f64 = 2.0;

/// How much of a *downward* vertical request survives, `[0, 1]`.
///
/// This asymmetry is what "gravity is replaced" means concretely, and it exists
/// because the host cannot tell a deliberate dive from an accumulated fall: a
/// character controller integrates gravity into its own velocity every step and
/// has no way to know the water should have stopped it, so it arrives at
/// `move_and_slide` asking to go *down* forever. Honouring that at full strength
/// would drive every swimmer to the bed. At a quarter strength the buoyancy
/// balance wins by default — a character that only integrates gravity floats —
/// while a player who really is holding "dive" still sinks, which is the
/// behaviour both halves want.
pub const SWIM_SINK_AUTHORITY: f64 = 0.25;

/// The fraction of a body's own height by which it must clear the surface before
/// it counts as *out* of the water (never less than [`MIN_EXIT_HYSTERESIS_M`]).
/// Depth hysteresis rather than fraction hysteresis, because a body that floats
/// always has its underside wet — asking "is it 2 % submerged" would report a
/// bobbing cork as dry.
const EXIT_HYSTERESIS_FRACTION: f64 = 0.05;

/// Floor on the exit hysteresis, metres — so a pebble still needs a centimetre of
/// clear air rather than a micrometre.
const MIN_EXIT_HYSTERESIS_M: f64 = 0.01;

/// The largest per-step velocity change a drag term is allowed to ask for, as a
/// fraction of the current relative velocity.
///
/// Linear drag integrated explicitly is stable while `k·dt < 1` and reverses the
/// velocity above `k·dt > 2`. Authoring `linear_drag = 1000` is a mistake, not an
/// instruction to explode the sim, so the coefficient is clamped to `0.9/dt`.
/// Deterministic (it is a function of `dt`, which is fixed), and it only ever
/// engages on values that would have been wrong anyway.
const DRAG_STABILITY_LIMIT: f64 = 0.9;

/// Target cell size of the water grid index, metres. A lake is tens of metres
/// across and an ocean is unbounded, so 64 m puts a typical lake in one cell and
/// a large one in a handful.
const GRID_TARGET_CELL_M: f64 = 64.0;

/// Cap on the grid's side length, so the index is `O(1)` memory no matter how far
/// apart two lakes are authored. At 48 the grid is at most 2 304 cells.
const GRID_MAX_DIM: u32 = 48;

// ─────────────────────────────────────────────────────────────────────────────
// Events
// ─────────────────────────────────────────────────────────────────────────────

/// What a [`WaterEvent3D`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WaterEventKind3D {
    /// The body's deepest point went under a water surface.
    Enter,
    /// The body cleared the surface by its exit hysteresis.
    Exit,
    /// A crossing (in either direction) fast enough to throw water — fired
    /// **in addition to** the `Enter`/`Exit` it accompanies, never instead of it,
    /// so a handler that only cares about wet/dry never has to know about it.
    Splash,
}

/// A body crossing a water surface: which body, which water, which way, how fast.
///
/// Reported in the owning body's `Guid` order (the bridge discipline), and within
/// one body `Enter`/`Exit` always precedes its `Splash`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterEvent3D {
    /// The entity that crossed.
    pub body: Uuid,
    /// The water body it crossed into or out of.
    pub water: Uuid,
    /// Which way, and whether it was fast.
    pub kind: WaterEventKind3D,
    /// Speed along gravity's up-axis at the crossing, m/s, always `>= 0`.
    pub speed_m_s: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe
// ─────────────────────────────────────────────────────────────────────────────

/// What the water looks like from one body's point of view — the answer the
/// `water.*` Blueprint nodes and the swim latch are both built on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterProbe {
    /// The water body the deepest sample found.
    pub water: Uuid,
    /// Surface elevation over the body's origin, metres of world Y.
    pub surface_y: f64,
    /// How much of the body is under water, `[0, 1]`.
    pub fraction: f64,
    /// How far the body's **lowest** point is below the surface, metres —
    /// negative when the body is clear of the water, `-inf` when there is no water
    /// under it at all.
    ///
    /// Measured at the underside rather than at the sample plane on purpose: a
    /// body that floats always has its bottom wet, so "is it in the water" has to
    /// be a question about the deepest point clearing the surface. Measuring at
    /// the mid-plane would report a cork — which rides with its centre *above*
    /// the waterline — as dry.
    pub depth_m: f64,
    /// The water's own velocity at the body's origin, m/s in world XZ. Zero for
    /// an ocean or a lake (see [`inf_water::WaterSurface::flow_at`] on why a
    /// Gerstner orbit is not a current); the tangent flow for a river.
    pub flow: DVec2,
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample geometry
// ─────────────────────────────────────────────────────────────────────────────

/// Where a collider is sampled, in its own body frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleGeometry {
    /// The shape's vertical half-extent in the body frame, metres — the length
    /// the per-sample submerged fraction is measured against.
    pub half_y: f64,
    /// The sample offsets, body frame. Placed on the shape's mid-plane at the
    /// quadrant midpoints, so a sample sitting exactly at the still-water line
    /// reads as half submerged.
    pub offsets: [DVec3; BUOYANCY_SAMPLES],
}

/// The sample layout for a collider shape.
///
/// **Exact primitives, AABB fallback.** A box, a sphere and a capsule state their
/// own vertical half-extent; a trimesh is reduced to the AABB of its vertices —
/// which is honest rather than lazy, because rapier cannot give a trimesh
/// well-defined mass properties either, so a trimesh body is static in practice
/// and never reaches this path with anything to float.
///
/// A **convex hull** (P22.2) takes the same AABB reduction, but here the
/// approximation is real and worth naming: a hull body *can* be dynamic and *can*
/// float, and its AABB over-states the waterplane of anything that is not a box.
/// The sampled draught is therefore a little optimistic for a wedge-shaped
/// fracture chunk. That is the same class of v1 simplification the linear-drag
/// model already carries, and the exact fix (sample the hull's own waterplane
/// section) belongs with whatever first needs floating debris, not here.
pub fn sample_geometry(shape: &ColliderShape3D, local_translation: DVec3) -> SampleGeometry {
    /// The AABB half-extents + centre of a point cloud in the body frame, or the
    /// half-metre default for an empty/non-finite one.
    fn cloud_extents(points: &[DVec3]) -> (f64, f64, f64, DVec3) {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        for v in points {
            min = min.min(*v);
            max = max.max(*v);
        }
        if !min.is_finite() || !max.is_finite() {
            (0.5, 0.5, 0.5, DVec3::ZERO)
        } else {
            let half = (max - min) * 0.5;
            (half.y, half.x, half.z, (max + min) * 0.5)
        }
    }
    let (half_y, hx, hz, centre) = match shape {
        ColliderShape3D::Box { half_extents } => (
            half_extents.y.abs(),
            half_extents.x.abs(),
            half_extents.z.abs(),
            DVec3::ZERO,
        ),
        ColliderShape3D::Sphere { radius } => {
            let r = radius.abs();
            (r, r, r, DVec3::ZERO)
        }
        ColliderShape3D::Capsule {
            half_height,
            radius,
        } => {
            let r = radius.abs();
            (half_height.abs() + r, r, r, DVec3::ZERO)
        }
        ColliderShape3D::Trimesh { vertices, .. } => cloud_extents(vertices),
        ColliderShape3D::ConvexHull { points } => cloud_extents(points),
        // A height field (P22.3) is ground. It is static by construction — it has
        // no mass (`ColliderShape3D::volume_m3` is `None` for it, exactly as for a
        // trimesh), so it can never carry a `Buoyancy` that reaches this function,
        // and the water pass skips non-dynamic bodies before it gets here anyway.
        // The half-metre default is the same "nothing sensible to report" answer
        // an empty point cloud gets, and it is unreachable rather than
        // approximate: a terrain that floated would be a bug two layers up.
        ColliderShape3D::Heightfield { .. } => (0.5, 0.5, 0.5, DVec3::ZERO),
    };
    // A fixed quadrant order, so two runs place the same force at the same point.
    let base = local_translation + centre;
    SampleGeometry {
        half_y: half_y.max(1e-6),
        offsets: [
            base + DVec3::new(-hx * 0.5, 0.0, -hz * 0.5),
            base + DVec3::new(-hx * 0.5, 0.0, hz * 0.5),
            base + DVec3::new(hx * 0.5, 0.0, -hz * 0.5),
            base + DVec3::new(hx * 0.5, 0.0, hz * 0.5),
        ],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The index
// ─────────────────────────────────────────────────────────────────────────────

/// One water body as the fixed step sees it.
#[derive(Clone, Debug)]
pub struct WaterEntry {
    /// The authoring entity's stable identity — what a water event reports.
    pub guid: Uuid,
    /// Its surface, built from the component the same way both scene projectors
    /// build the renderer's.
    pub surface: WaterSurface,
    /// World-XZ bounds, or `None` for an **unbounded** ocean.
    bounds: Option<[DVec2; 2]>,
}

/// A deterministic spatial index over a level's water.
///
/// Without it, a town with one lake would test all 13 000 of its colliders
/// against every water body every step; with it, the cost is one cell lookup per
/// *buoyant* body — and the town has none, so the whole pass is one branch.
///
/// The structure is a **uniform grid over the union of the bounded bodies**, plus
/// a separate list of unbounded ones (an ocean is over every point, so it belongs
/// in no cell). Both lists are kept in ascending body index, and
/// [`highest_surface_at`](Self::highest_surface_at) **merges** them in that order
/// rather than concatenating, so its answer is identical to scanning every body —
/// including the tie rule (`inf_water::highest_surface`: topmost wins, ties to the
/// earlier body). A spatial index that changed the answer would be a bug you could
/// only see in a big level.
#[derive(Default)]
pub struct WaterIndex {
    entries: Vec<WaterEntry>,
    /// Indices of unbounded bodies, ascending.
    unbounded: Vec<u32>,
    /// Union of the bounded bodies' XZ bounds; `None` if there are none.
    union: Option<[DVec2; 2]>,
    cols: u32,
    rows: u32,
    cell: DVec2,
    /// `cols * rows` buckets, each an ascending list of body indices.
    grid: Vec<Vec<u32>>,
}

impl WaterIndex {
    /// Rebuild from a `Guid`-sorted list of bodies. The caller owns the change
    /// detection — this is the expensive half (a river resamples its arc length
    /// here) and it must not run at 60 Hz on unchanged water.
    pub fn rebuild(&mut self, entries: Vec<WaterEntry>) {
        self.entries = entries;
        self.unbounded.clear();
        self.grid.clear();
        self.union = None;
        self.cols = 0;
        self.rows = 0;
        self.cell = DVec2::ZERO;

        let mut union: Option<[DVec2; 2]> = None;
        for (i, e) in self.entries.iter().enumerate() {
            match e.bounds {
                None => self.unbounded.push(i as u32),
                Some([lo, hi]) => {
                    union = Some(match union {
                        None => [lo, hi],
                        Some([ulo, uhi]) => [ulo.min(lo), uhi.max(hi)],
                    });
                }
            }
        }
        let Some([lo, hi]) = union else {
            return;
        };
        let extent = (hi - lo).max(DVec2::ZERO);
        let cols = grid_dim(extent.x);
        let rows = grid_dim(extent.y);
        // A zero-extent axis gets a cell of 1 m so the divide is never by zero;
        // every point then lands in column 0, which is correct for a degenerate
        // (single-point) lake.
        let cell = DVec2::new(
            if extent.x > 0.0 {
                extent.x / cols as f64
            } else {
                1.0
            },
            if extent.y > 0.0 {
                extent.y / rows as f64
            } else {
                1.0
            },
        );
        self.union = Some([lo, hi]);
        self.cols = cols;
        self.rows = rows;
        self.cell = cell;
        self.grid = vec![Vec::new(); (cols * rows) as usize];
        for (i, e) in self.entries.iter().enumerate() {
            let Some([blo, bhi]) = e.bounds else { continue };
            let (x0, x1) = span(blo.x, bhi.x, lo.x, cell.x, cols);
            let (z0, z1) = span(blo.y, bhi.y, lo.y, cell.y, rows);
            for z in z0..=z1 {
                for x in x0..=x1 {
                    self.grid[(z * cols + x) as usize].push(i as u32);
                }
            }
        }
    }

    /// Forget every body (a level unload, or a scene whose water was deleted).
    pub fn clear(&mut self) {
        self.rebuild(Vec::new());
    }

    /// Are there no water bodies at all?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many bodies the index holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The indexed bodies, in `Guid` order.
    pub fn entries(&self) -> &[WaterEntry] {
        &self.entries
    }

    /// The bodies that could possibly cover `p`, in **ascending index order** —
    /// the unbounded list merged with the cell's list, never concatenated (see the
    /// type docs on why the order is load-bearing).
    pub fn candidates(&self, p: DVec2) -> Vec<u32> {
        let cell = self.cell_of(p);
        let bucket: &[u32] = match cell {
            Some(c) => &self.grid[c],
            None => &[],
        };
        let mut out = Vec::with_capacity(self.unbounded.len() + bucket.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.unbounded.len() && j < bucket.len() {
            if self.unbounded[i] <= bucket[j] {
                out.push(self.unbounded[i]);
                i += 1;
            } else {
                out.push(bucket[j]);
                j += 1;
            }
        }
        out.extend_from_slice(&self.unbounded[i..]);
        out.extend_from_slice(&bucket[j..]);
        out
    }

    /// The **highest** water surface over `p` at time `t`, and the body it came
    /// from. Identical to `inf_water::highest_surface` over every body — the index
    /// only skips bodies that would have answered `None`.
    pub fn highest_surface_at(&self, p: DVec2, t: f64) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for i in self.candidates(p) {
            let e = &self.entries[i as usize];
            if let Some(h) = e.surface.height_at(p, t) {
                if best.is_none_or(|(_, bh)| h > bh) {
                    best = Some((i as usize, h));
                }
            }
        }
        best
    }

    /// The flow of the topmost water over `p`, m/s in world XZ — zero where the
    /// water is still, `None` where there is no water.
    pub fn flow_at(&self, p: DVec2, t: f64) -> Option<DVec2> {
        let (i, _) = self.highest_surface_at(p, t)?;
        self.entries[i].surface.flow_at(p)
    }

    fn cell_of(&self, p: DVec2) -> Option<usize> {
        let [lo, hi] = self.union?;
        if self.cols == 0 || self.rows == 0 {
            return None;
        }
        // Outside the union is not an error — an ocean still answers, and a point
        // far from every lake simply has no bucket to look in. Tested on the
        // CLOSED bounds so a point exactly on an edge still lands in a cell; the
        // clamp below is what keeps `p.x == hi.x` out of column `cols`.
        if p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y {
            return None;
        }
        let x = ((p.x - lo.x) / self.cell.x)
            .floor()
            .clamp(0.0, (self.cols - 1) as f64) as u32;
        let z = ((p.y - lo.y) / self.cell.y)
            .floor()
            .clamp(0.0, (self.rows - 1) as f64) as u32;
        Some((z * self.cols + x) as usize)
    }
}

fn grid_dim(extent: f64) -> u32 {
    // `<=` rather than `!(> 0)`: a NaN extent is impossible here (the bounds come
    // from finite component fields) and one column is the honest answer for a
    // degenerate axis either way.
    if extent <= 0.0 {
        return 1;
    }
    ((extent / GRID_TARGET_CELL_M).ceil() as u32).clamp(1, GRID_MAX_DIM)
}

/// The inclusive cell-index span an interval covers, clamped into the grid.
fn span(lo: f64, hi: f64, origin: f64, cell: f64, dim: u32) -> (u32, u32) {
    let a = ((lo - origin) / cell).floor().clamp(0.0, (dim - 1) as f64) as u32;
    let b = ((hi - origin) / cell).floor().clamp(0.0, (dim - 1) as f64) as u32;
    (a.min(b), a.max(b))
}

// ─────────────────────────────────────────────────────────────────────────────
// Component → surface
// ─────────────────────────────────────────────────────────────────────────────

/// Build the fixed step's [`WaterSurface`] from a scene [`WaterBody`] (plus the
/// [`Spline`] on the **same entity**, for a river).
///
/// **This is Ring 0 on purpose.** The renderer's equivalent (`project_water`) is
/// written twice in Ring 2 and gated character-for-character, because neither
/// `inf-render` nor `inf-ecs` can host it. The *simulation's* projector has no
/// such problem — `inf-physics` already names `inf-ecs` — so it is written once,
/// here, and both hosts call it. The two projections are then pinned to each other
/// numerically (`water_physics.rs` compares the derived Gerstner components as
/// bits), which is a stronger statement than two copies of the same source.
///
/// `env` is `(level clock in seconds, weather wind in m/s)` — resolved once per
/// sync from [`inf_ecs::sky::water_environment`], never from a wall clock.
pub fn water_surface_of(
    water: &WaterBody,
    spline: Option<&Spline>,
    affine: &DAffine3,
    env: (f64, (f64, f64)),
) -> Option<WaterSurface> {
    let (_t, weather_wind) = env;
    let (wind_x, wind_z) = water.effective_wind(weather_wind);
    let river = water.kind == WaterKind::River;
    // A river's ripple travels DOWNSTREAM: its wave frame is (arc length,
    // lateral), so its "wind" is +1 along the river rather than a world direction.
    // Byte-for-byte the rule `project_water` applies.
    let spec = WaveSpec {
        amplitude_m: water.wave_amplitude_m,
        wavelength_m: water.wave_length_m,
        steepness: water.wave_steepness,
        wind_x: if river { 1.0 } else { wind_x },
        wind_z: if river { 0.0 } else { wind_z },
        spread_rad: water.wave_spread_deg.to_radians(),
        seed: water.wave_seed,
        count: water.wave_count,
    };
    let waves = WaveField::from_spec(&spec);
    match water.kind {
        WaterKind::Ocean => Some(WaterSurface::Ocean {
            level_m: water.level_m,
            waves,
        }),
        WaterKind::Lake => Some(WaterSurface::Lake {
            level_m: water.level_m,
            center: DVec2::new(affine.translation.x, affine.translation.z),
            half_extent: DVec2::new(water.extent.x.max(0.0), water.extent.y.max(0.0)),
            waves,
        }),
        WaterKind::River => {
            // No spline ⇒ no centreline ⇒ nothing to float on. An authoring
            // state, not an error — exactly as the renderer treats it.
            let sp = spline?;
            if sp.points.len() < 2 {
                return None;
            }
            let points: Vec<DVec3> = sp
                .points
                .iter()
                .map(|p| affine.transform_point3(p.to_dvec3()))
                .collect();
            let interp = match sp.interp {
                SceneSplineInterp::Linear => inf_math::spline::SplineInterp::Linear,
                SceneSplineInterp::CatmullRom => inf_math::spline::SplineInterp::CatmullRom,
            };
            // ONE sanitizer, in Ring 0 (P20.4) — see `RiverProfile::authored`.
            let profile = RiverProfile::authored(
                water.river_width_start_m,
                water.river_width_end_m,
                water.river_depth_start_m,
                water.river_depth_end_m,
                water.river_flow_m_s,
            );
            let path = RiverPath::from_points(&points, sp.closed, interp, &profile);
            if path.frames.is_empty() {
                return None;
            }
            Some(WaterSurface::River { path, waves })
        }
    }
}

/// The world-XZ bounds of a surface, or `None` when it is unbounded.
///
/// Conservative on purpose: a river's bound is the AABB of its frame centres
/// grown by the widest half-width, which always contains the ribbon and is cheap
/// to compute. An index that under-reported a bound would lose a body that really
/// was under the water.
fn surface_bounds(surface: &WaterSurface) -> Option<[DVec2; 2]> {
    match surface {
        WaterSurface::Ocean { .. } => None,
        WaterSurface::Lake {
            center,
            half_extent,
            ..
        } => {
            let h = half_extent.max(DVec2::ZERO);
            Some([*center - h, *center + h])
        }
        WaterSurface::River { path, .. } => {
            let mut lo = DVec2::splat(f64::INFINITY);
            let mut hi = DVec2::splat(f64::NEG_INFINITY);
            let mut widest: f64 = 0.0;
            for f in &path.frames {
                let c = DVec2::new(f.center.x, f.center.z);
                lo = lo.min(c);
                hi = hi.max(c);
                widest = widest.max(f.width_m.abs() * 0.5);
            }
            if !lo.is_finite() || !hi.is_finite() {
                return Some([DVec2::ZERO, DVec2::ZERO]);
            }
            let pad = DVec2::splat(widest);
            Some([lo - pad, hi + pad])
        }
    }
}

/// A [`WaterEntry`] from an already-built surface.
pub fn water_entry(guid: Uuid, surface: WaterSurface) -> WaterEntry {
    let bounds = surface_bounds(&surface);
    WaterEntry {
        guid,
        surface,
        bounds,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The change stamp
// ─────────────────────────────────────────────────────────────────────────────

/// Everything a water surface is derived from, in one comparable value.
///
/// The P19.5 change-stamp pattern, applied to water for the same reason it was
/// applied to PCG structures: a river's `RiverPath` resamples its spline at even
/// **arc length**, which is an `O(points × subdivisions)` walk, and doing it at
/// 60 Hz to learn that nobody moved the river is a per-step cost a load-time
/// budget never sees. The clock is deliberately **not** in the stamp — a wave
/// field is a function of the wind, and time is an argument to `height_at`, not a
/// parameter of the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterStamp {
    guid: Uuid,
    body: WaterBody,
    /// The spline folded to 64 bits rather than cloned — a river's points are the
    /// only unbounded part of this, and a Vec clone per step is a Vec clone per
    /// step.
    spline: u64,
    affine: DAffine3,
}

impl WaterStamp {
    /// Stamp one water entity.
    pub fn new(guid: Uuid, body: WaterBody, spline: Option<&Spline>, affine: DAffine3) -> Self {
        Self {
            guid,
            body,
            spline: spline.map(spline_hash).unwrap_or(0),
            affine,
        }
    }

    /// The entity this stamp belongs to.
    pub fn guid(&self) -> Uuid {
        self.guid
    }
}

/// A 64-bit fold of a spline's identity. Integer mixing over IEEE bits, so it is
/// bit-portable and has no float comparison in it; collisions would mean a moved
/// river went unnoticed, which is why the multiplier is a full 64-bit odd
/// constant rather than a small prime.
fn spline_hash(sp: &Spline) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
    };
    mix(sp.points.len() as u64);
    mix(sp.closed as u64);
    mix(match sp.interp {
        SceneSplineInterp::Linear => 1,
        SceneSplineInterp::CatmullRom => 2,
    });
    for p in &sp.points {
        mix(p.x.to_bits());
        mix(p.y.to_bits());
        mix(p.z.to_bits());
    }
    // A zero hash means "no spline", so a spline that hashes to zero is nudged.
    if h == 0 {
        1
    } else {
        h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-body state
// ─────────────────────────────────────────────────────────────────────────────

/// What the water pass remembers about one buoyant body between steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BuoyantState {
    /// Whether the body was reported in the water at the last pass (the latch the
    /// enter/exit hysteresis flips).
    pub in_water: bool,
    /// The water it was last in, so an `Exit` can name the body it left even
    /// after the surface has moved out from under it.
    pub water: Option<Uuid>,
}

/// A body's buoyancy tuning as the pass sees it — the facade-local shape of the
/// [`Buoyancy`] component, mirroring [`super::BodyDesc3D`]'s relationship to
/// `RigidBody3D`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuoyancyDesc3D {
    /// The body's density for flotation, kg/m³.
    pub density_kg_m3: f64,
    /// The fluid's density, kg/m³.
    pub fluid_density_kg_m3: f64,
    /// Linear drag, s⁻¹.
    pub linear_drag: f64,
    /// Angular drag, s⁻¹.
    pub angular_drag: f64,
}

impl BuoyancyDesc3D {
    /// Map a scene [`Buoyancy`], or `None` when it is switched off.
    pub fn from_component(b: &Buoyancy) -> Option<Self> {
        if !b.enabled {
            return None;
        }
        Some(Self {
            density_kg_m3: b.density_kg_m3,
            fluid_density_kg_m3: b.fluid_density_kg_m3,
            linear_drag: b.linear_drag.max(0.0),
            angular_drag: b.angular_drag.max(0.0),
        })
    }

    /// Whether this body can float at all — a non-positive density has no
    /// displaced volume to speak of and would divide by zero.
    pub fn is_usable(&self) -> bool {
        self.density_kg_m3 > 0.0 && self.fluid_density_kg_m3 > 0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The force solve (pure)
// ─────────────────────────────────────────────────────────────────────────────

/// One body's water forces for one step — computed as a **pure function** of the
/// pose, the velocities, the mass and the water, then applied by the bridge.
///
/// Split out so the whole model is testable without a rapier world, and so the
/// bridge's apply pass is a loop with no arithmetic in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterForces {
    /// Buoyant force per sample point, and the world point to apply it at. The
    /// samples carry the torque: a body tipped on a wave has more of itself under
    /// water on one side, and applying the force where it is generated is what
    /// turns that into a righting moment rather than a number nobody uses.
    pub samples: [(DVec3, DVec3); BUOYANCY_SAMPLES],
    /// Linear drag, applied at the centre of mass.
    pub drag: DVec3,
    /// Angular drag torque.
    pub torque: DVec3,
    /// The probe that produced them.
    pub probe: WaterProbe,
}

/// Everything the solve needs about the body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyState {
    pub translation: DVec3,
    pub rotation: DQuat,
    pub linvel: DVec3,
    pub angvel: DVec3,
    pub mass: f64,
}

/// Probe the water under a body: how deep its lowest point is, how much of it is
/// submerged, the surface height over it, and the water's flow.
///
/// The submerged fraction is the mean of the per-column fractions — see
/// [`column_fraction`] for the model and its named error.
pub fn probe(index: &WaterIndex, t: f64, state: &BodyState, geo: &SampleGeometry) -> WaterProbe {
    let mut sum = 0.0;
    let mut deepest = f64::NEG_INFINITY;
    let mut water: Option<Uuid> = None;
    for off in geo.offsets {
        let p = state.translation + state.rotation * off;
        match index.highest_surface_at(DVec2::new(p.x, p.z), t) {
            Some((i, h)) => {
                // How deep this column's *underside* is: the sample sits on the
                // shape's mid-plane, so the bottom is `half_y` below it.
                let wet = (h - p.y) + geo.half_y;
                if wet > deepest {
                    deepest = wet;
                    water = Some(index.entries[i].guid);
                }
                sum += column_fraction(wet, geo.half_y);
            }
            None => {
                // No water over this column: it displaces nothing, and it cannot
                // be the deepest sample either.
            }
        }
    }
    let centre = DVec2::new(state.translation.x, state.translation.z);
    let surface_y = index
        .highest_surface_at(centre, t)
        .map(|(_, h)| h)
        .unwrap_or(f64::NEG_INFINITY);
    WaterProbe {
        water: water.unwrap_or(Uuid::nil()),
        surface_y,
        fraction: sum / BUOYANCY_SAMPLES as f64,
        depth_m: if deepest.is_finite() {
            deepest
        } else {
            f64::NEG_INFINITY
        },
        flow: index.flow_at(centre, t).unwrap_or(DVec2::ZERO),
    }
}

/// How much of one sample column is under water, `[0, 1]`, from how deep its
/// underside is.
///
/// **Linear in depth over the shape's vertical extent.** Exact for a box at any
/// depth, and exact for a sphere or a capsule at the symmetric half-submerged
/// point (which is the equilibrium the statics tests assert); a shape factor away
/// from exact elsewhere. Stated here rather than implied, because "approximate"
/// with a named error is engineering and "approximate" without one is a guess.
#[inline]
fn column_fraction(wet_m: f64, half_y: f64) -> f64 {
    (wet_m / (2.0 * half_y)).clamp(0.0, 1.0)
}

/// Solve one body's buoyancy + drag.
///
/// **The model.** Buoyancy is Archimedes: `submerged_fraction × displaced_volume
/// × ρ_fluid × g`, opposing gravity, split evenly across the sample points and
/// weighted by each point's own submersion. The displaced volume is
/// `mass / density_kg_m3` — rapier's **exact per-shape** volume, read back
/// through the mass it already computed, rather than a second hand-written volume
/// table that could disagree with the one the solver uses.
///
/// **Drag is linear**, in the velocity relative to the *water's* flow, scaled by
/// the submerged fraction, with the coefficient in s⁻¹ so it means exactly what
/// `RigidBody3D::linear_damping` means. Quadratic drag is the physically right
/// law for a hull and is deferred: it needs a reference area and a shape-dependent
/// drag coefficient, neither of which v1 has anywhere honest to get.
///
/// **The water's velocity is its flow, not its wave orbit.** `flow_at` is zero for
/// an ocean or a lake by an explicit P20.1 decision (a Gerstner particle's orbit
/// averages to no net transport, so reporting the instantaneous orbital velocity
/// as a current would push a boat across the sea at wave speed). Using the orbital
/// velocity as the *drag reference* rather than as a current is defensible and is
/// deferred: it would need a new `WaveField` seam and it oscillates at wave
/// frequency, which is a stiff term at a 60 Hz step. Still-water drag, stated as
/// the v1 it is.
pub fn solve(
    index: &WaterIndex,
    t: f64,
    state: &BodyState,
    geo: &SampleGeometry,
    desc: &BuoyancyDesc3D,
    gravity: DVec3,
    dt: f64,
) -> WaterForces {
    let probe = probe(index, t, state, geo);
    let g_mag = gravity.length();
    let up = if g_mag > 0.0 {
        -gravity / g_mag
    } else {
        DVec3::Y
    };
    let mut out = WaterForces {
        samples: [(DVec3::ZERO, DVec3::ZERO); BUOYANCY_SAMPLES],
        drag: DVec3::ZERO,
        torque: DVec3::ZERO,
        probe,
    };
    if !desc.is_usable() || state.mass <= 0.0 || g_mag <= 0.0 || probe.fraction <= 0.0 {
        // Still fill the sample points so a caller that wants to draw them has
        // them; the forces stay zero.
        for (i, off) in geo.offsets.iter().enumerate() {
            out.samples[i].1 = state.translation + state.rotation * *off;
        }
        return out;
    }
    // Displaced volume per sample: the body's volume, split evenly, weighted by
    // that sample's own submersion.
    let volume = state.mass / desc.density_kg_m3;
    let per_sample = desc.fluid_density_kg_m3 * g_mag * volume / BUOYANCY_SAMPLES as f64;
    for (i, off) in geo.offsets.iter().enumerate() {
        let p = state.translation + state.rotation * *off;
        let f_i = match index.highest_surface_at(DVec2::new(p.x, p.z), t) {
            Some((_, h)) => column_fraction((h - p.y) + geo.half_y, geo.half_y),
            None => 0.0,
        };
        out.samples[i] = (up * (per_sample * f_i), p);
    }

    // Drag, against the water's own flow.
    let v_water = DVec3::new(probe.flow.x, 0.0, probe.flow.y);
    let v_rel = state.linvel - v_water;
    let limit = if dt > 0.0 {
        DRAG_STABILITY_LIMIT / dt
    } else {
        f64::INFINITY
    };
    let k_lin = (desc.linear_drag * probe.fraction).min(limit);
    out.drag = -v_rel * (k_lin * state.mass);

    // Angular drag needs a moment of inertia. Rather than reach for rapier's
    // (which is a local-frame tensor that would have to be rotated into world
    // space every step), the sample points supply one: `m · mean(|r|²)` is the
    // second moment of the mass placed at the points the buoyancy already acts
    // through. It is isotropic — exact for a sphere, an approximation for a long
    // hull — and it makes the coefficient a plain s⁻¹ rate like the linear one.
    let mut r2 = 0.0;
    for off in geo.offsets {
        r2 += (state.rotation * off).length_squared();
    }
    let inertia = state.mass * (r2 / BUOYANCY_SAMPLES as f64);
    let k_ang = (desc.angular_drag * probe.fraction).min(limit);
    out.torque = -state.angvel * (k_ang * inertia);
    out
}

/// The exit hysteresis for a body of this geometry, metres.
pub fn exit_hysteresis_m(geo: &SampleGeometry) -> f64 {
    (EXIT_HYSTERESIS_FRACTION * 2.0 * geo.half_y).max(MIN_EXIT_HYSTERESIS_M)
}

/// Flip a body's in-water latch and report the events the crossing produced.
///
/// The latch is depth hysteresis, not fraction hysteresis: a floating body always
/// has its underside wet, so "is it in the water" has to be a question about the
/// deepest point clearing the surface, not about a fraction that would report a
/// bobbing cork as dry. `speed` is measured along gravity's up-axis, so a body
/// skimming the surface horizontally does not splash — it has not crossed
/// anything.
pub(crate) fn crossing_events(
    guid: Uuid,
    state: &mut BuoyantState,
    probe: &WaterProbe,
    hysteresis_m: f64,
    up_speed: f64,
    out: &mut Vec<WaterEvent3D>,
) {
    let now = if state.in_water {
        probe.depth_m > -hysteresis_m
    } else {
        probe.depth_m > 0.0
    };
    if now == state.in_water {
        if now {
            state.water = Some(probe.water);
        }
        return;
    }
    let water = if now {
        probe.water
    } else {
        state.water.unwrap_or(probe.water)
    };
    let kind = if now {
        WaterEventKind3D::Enter
    } else {
        WaterEventKind3D::Exit
    };
    let speed = up_speed.abs();
    out.push(WaterEvent3D {
        body: guid,
        water,
        kind,
        speed_m_s: speed,
    });
    if speed >= SPLASH_SPEED_M_S {
        out.push(WaterEvent3D {
            body: guid,
            water,
            kind: WaterEventKind3D::Splash,
            speed_m_s: speed,
        });
    }
    state.in_water = now;
    state.water = now.then_some(water);
}

// ─────────────────────────────────────────────────────────────────────────────
// Swim
// ─────────────────────────────────────────────────────────────────────────────

/// Flip a character's swim latch from its submerged fraction, with hysteresis.
///
/// Pure, so both hosts (the editor's Simulate loop and the shipped runtime) get
/// the *same* answer from the *same* function rather than from two copies of a
/// threshold — which is exactly the drift the projector MIRROR gate exists to
/// catch elsewhere, avoided here by there being nothing to mirror.
pub fn swim_latch(was_swimming: bool, fraction: f64) -> bool {
    if was_swimming {
        fraction > SWIM_EXIT_FRACTION
    } else {
        fraction >= SWIM_ENTER_FRACTION
    }
}

/// Transform a `move_and_slide` motion for a swimming character.
///
/// Three things happen, and they are the whole of "swim mode":
///
/// 1. **Gravity is replaced.** The caller's vertical motion is read as a
///    *rate* — an intent — clamped to [`SWIM_VERT_MAX_M_S`] and, downward, scaled
///    by [`SWIM_SINK_AUTHORITY`]; a buoyancy-balance term then drives the swimmer
///    toward [`SWIM_FLOAT_SUBMERSION`]. Because the balance outweighs a
///    full-strength downward request, a character that is doing nothing but
///    integrating gravity **surfaces**, which is the whole point: it has no way to
///    know the water should have stopped it.
/// 2. **Vertical input still works.** Asking to rise is honoured in full, up to
///    the swim rate; asking to dive is honoured at a quarter, which reads as the
///    water pushing back.
/// 3. **Horizontal speed is capped** to [`SWIM_SPEED_MAX_M_S`], because a
///    character's running speed in deep water is the single clearest tell that a
///    game has no swim mode.
///
/// Pure and `dt`-explicit, so both hosts get the same answer from the same
/// function.
pub fn swim_motion(motion: DVec3, fraction: f64, dt: f64) -> DVec3 {
    if dt <= 0.0 {
        return motion;
    }
    let intent = (motion.y / dt).clamp(-SWIM_VERT_MAX_M_S, SWIM_VERT_MAX_M_S);
    let intent = if intent < 0.0 {
        intent * SWIM_SINK_AUTHORITY
    } else {
        intent
    };
    let balance = SWIM_FLOAT_RATE_M_S * (fraction - SWIM_FLOAT_SUBMERSION);
    let y = (intent + balance) * dt;
    let horizontal = DVec2::new(motion.x, motion.z);
    let cap = SWIM_SPEED_MAX_M_S * dt;
    let len = horizontal.length();
    let horizontal = if len > cap && len > 0.0 {
        horizontal * (cap / len)
    } else {
        horizontal
    };
    DVec3::new(horizontal.x, y, horizontal.y)
}

/// The per-entity buoyancy + water state the bridge keeps.
pub(crate) type BuoyantMap = BTreeMap<Uuid, (BuoyancyDesc3D, BuoyantState)>;

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::math::Vec2d;

    fn still_lake(guid: u128, level: f64, centre: DVec2, half: DVec2) -> WaterEntry {
        water_entry(
            Uuid::from_u128(guid),
            WaterSurface::Lake {
                level_m: level,
                center: centre,
                half_extent: half,
                waves: WaveField::from_spec(&WaveSpec {
                    amplitude_m: 0.0,
                    ..WaveSpec::default()
                }),
            },
        )
    }

    fn ocean(guid: u128, level: f64) -> WaterEntry {
        water_entry(
            Uuid::from_u128(guid),
            WaterSurface::Ocean {
                level_m: level,
                waves: WaveField::from_spec(&WaveSpec {
                    amplitude_m: 0.0,
                    ..WaveSpec::default()
                }),
            },
        )
    }

    fn unit_box() -> SampleGeometry {
        sample_geometry(
            &ColliderShape3D::Box {
                half_extents: DVec3::splat(0.5),
            },
            DVec3::ZERO,
        )
    }

    #[test]
    fn an_empty_index_answers_nothing_and_costs_nothing() {
        let idx = WaterIndex::default();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.highest_surface_at(DVec2::ZERO, 0.0).is_none());
        assert!(idx.flow_at(DVec2::ZERO, 0.0).is_none());
        assert!(idx.candidates(DVec2::new(1e9, -1e9)).is_empty());
    }

    /// The load-bearing property: the index must give the SAME answer a full scan
    /// would, including the tie rule. A spatial structure that changed the answer
    /// would only be visible in a level too big to debug.
    #[test]
    fn the_index_answers_exactly_what_a_full_scan_would() {
        let entries = vec![
            still_lake(1, 5.0, DVec2::new(-200.0, 0.0), DVec2::splat(30.0)),
            ocean(2, 4.0),
            still_lake(3, 5.0, DVec2::new(-200.0, 0.0), DVec2::splat(30.0)),
            still_lake(4, 9.0, DVec2::new(300.0, 400.0), DVec2::splat(50.0)),
            still_lake(5, 1.0, DVec2::new(0.0, 0.0), DVec2::splat(10.0)),
        ];
        let surfaces: Vec<WaterSurface> = entries.iter().map(|e| e.surface.clone()).collect();
        let mut idx = WaterIndex::default();
        idx.rebuild(entries);
        for i in -60..60 {
            for j in -60..60 {
                let p = DVec2::new(i as f64 * 12.0, j as f64 * 12.0);
                let scan = inf_water::highest_surface(&surfaces, p, 3.0);
                let indexed = idx.highest_surface_at(p, 3.0);
                assert_eq!(scan, indexed, "at {p:?}");
            }
        }
    }

    #[test]
    fn candidates_are_ascending_and_include_every_ocean() {
        let mut idx = WaterIndex::default();
        idx.rebuild(vec![
            still_lake(1, 1.0, DVec2::ZERO, DVec2::splat(10.0)),
            ocean(2, 0.0),
            still_lake(3, 1.0, DVec2::new(500.0, 0.0), DVec2::splat(10.0)),
        ]);
        let c = idx.candidates(DVec2::ZERO);
        assert!(c.windows(2).all(|w| w[0] < w[1]), "{c:?}");
        assert!(c.contains(&1), "the ocean is over every point");
        assert!(c.contains(&0));
        assert!(!c.contains(&2), "the far lake is not a candidate here");
    }

    /// The town statement: bodies far from every lake reach only the oceans, and
    /// with no ocean they reach nothing at all.
    #[test]
    fn a_body_outside_every_bounded_body_is_untouched() {
        let mut idx = WaterIndex::default();
        idx.rebuild(vec![still_lake(1, 1.0, DVec2::ZERO, DVec2::splat(20.0))]);
        for p in [
            DVec2::new(10_000.0, 0.0),
            DVec2::new(-5_000.0, 5_000.0),
            DVec2::new(0.0, 21.0),
        ] {
            assert!(idx.candidates(p).is_empty(), "at {p:?}");
            assert!(idx.highest_surface_at(p, 0.0).is_none());
        }
    }

    #[test]
    fn sample_geometry_is_exact_for_primitives_and_aabb_for_a_trimesh() {
        let b = sample_geometry(
            &ColliderShape3D::Box {
                half_extents: DVec3::new(2.0, 0.5, 3.0),
            },
            DVec3::ZERO,
        );
        assert_eq!(b.half_y, 0.5);
        assert_eq!(b.offsets[0], DVec3::new(-1.0, 0.0, -1.5));
        assert_eq!(b.offsets[3], DVec3::new(1.0, 0.0, 1.5));

        let s = sample_geometry(&ColliderShape3D::Sphere { radius: 2.0 }, DVec3::ZERO);
        assert_eq!(s.half_y, 2.0);

        let c = sample_geometry(
            &ColliderShape3D::Capsule {
                half_height: 0.9,
                radius: 0.3,
            },
            DVec3::ZERO,
        );
        assert_eq!(c.half_y, 1.2, "a capsule is its cylinder plus both caps");

        // The AABB fallback.
        let t = sample_geometry(
            &ColliderShape3D::Trimesh {
                vertices: vec![
                    DVec3::new(-1.0, -2.0, -3.0),
                    DVec3::new(1.0, 2.0, 3.0),
                    DVec3::ZERO,
                ],
                indices: vec![[0, 1, 2]],
            },
            DVec3::ZERO,
        );
        assert_eq!(t.half_y, 2.0);

        // The collider offset rides along.
        let off = sample_geometry(
            &ColliderShape3D::Sphere { radius: 1.0 },
            DVec3::new(0.0, 5.0, 0.0),
        );
        assert!(off.offsets.iter().all(|o| o.y == 5.0));
    }

    #[test]
    fn a_body_at_the_waterline_is_half_submerged() {
        let mut idx = WaterIndex::default();
        idx.rebuild(vec![still_lake(1, 10.0, DVec2::ZERO, DVec2::splat(50.0))]);
        let geo = unit_box();
        let state = BodyState {
            translation: DVec3::new(0.0, 10.0, 0.0),
            rotation: DQuat::IDENTITY,
            linvel: DVec3::ZERO,
            angvel: DVec3::ZERO,
            mass: 1.0,
        };
        let p = probe(&idx, 0.0, &state, &geo);
        assert!((p.fraction - 0.5).abs() < 1e-12, "{}", p.fraction);
        assert!(
            (p.depth_m - 0.5).abs() < 1e-12,
            "the UNDERSIDE of a half-submerged unit box is 0.5 m down, not 0"
        );
        assert!((p.surface_y - 10.0).abs() < 1e-12);

        // Fully under, and fully clear.
        let under = BodyState {
            translation: DVec3::new(0.0, 5.0, 0.0),
            ..state
        };
        assert_eq!(probe(&idx, 0.0, &under, &geo).fraction, 1.0);
        let clear = BodyState {
            translation: DVec3::new(0.0, 20.0, 0.0),
            ..state
        };
        let cp = probe(&idx, 0.0, &clear, &geo);
        assert_eq!(cp.fraction, 0.0);
        assert!(
            cp.depth_m < 0.0,
            "clear of the water reads as negative depth"
        );
    }

    /// The analytic statement, solved rather than simulated: at the equilibrium
    /// fraction the buoyant force is exactly the body's weight.
    #[test]
    fn buoyancy_balances_weight_at_the_equilibrium_fraction() {
        let mut idx = WaterIndex::default();
        idx.rebuild(vec![still_lake(1, 0.0, DVec2::ZERO, DVec2::splat(50.0))]);
        let geo = unit_box();
        let g = DVec3::new(0.0, -9.81, 0.0);
        for density in [200.0, 500.0, 750.0, 1000.0] {
            let desc = BuoyancyDesc3D {
                density_kg_m3: density,
                fluid_density_kg_m3: 1000.0,
                linear_drag: 0.0,
                angular_drag: 0.0,
            };
            let equilibrium = density / 1000.0;
            // Place the box so exactly `equilibrium` of it is under water: the
            // mid-plane sits `(0.5 - equilibrium)` box-heights above the surface.
            let y = (0.5 - equilibrium) * 2.0 * geo.half_y;
            let mass = density * 1.0; // a 1 m³ box
            let state = BodyState {
                translation: DVec3::new(0.0, y, 0.0),
                rotation: DQuat::IDENTITY,
                linvel: DVec3::ZERO,
                angvel: DVec3::ZERO,
                mass,
            };
            let f = solve(&idx, 0.0, &state, &geo, &desc, g, 1.0 / 60.0);
            let total: DVec3 = f.samples.iter().map(|(v, _)| *v).sum();
            let weight = mass * 9.81;
            assert!(
                (total.y - weight).abs() < 1e-9,
                "density {density}: lift {} vs weight {weight}",
                total.y
            );
        }
    }

    #[test]
    fn drag_opposes_the_velocity_relative_to_the_water_and_is_stability_clamped() {
        let mut idx = WaterIndex::default();
        idx.rebuild(vec![still_lake(1, 10.0, DVec2::ZERO, DVec2::splat(50.0))]);
        let geo = unit_box();
        let g = DVec3::new(0.0, -9.81, 0.0);
        let desc = BuoyancyDesc3D {
            density_kg_m3: 500.0,
            fluid_density_kg_m3: 1000.0,
            linear_drag: 2.0,
            angular_drag: 3.0,
        };
        let state = BodyState {
            translation: DVec3::new(0.0, 5.0, 0.0), // fully submerged
            rotation: DQuat::IDENTITY,
            linvel: DVec3::new(4.0, -2.0, 0.0),
            angvel: DVec3::new(0.0, 5.0, 0.0),
            mass: 500.0,
        };
        let dt = 1.0 / 60.0;
        let f = solve(&idx, 0.0, &state, &geo, &desc, g, dt);
        assert_eq!(f.probe.fraction, 1.0);
        // Direction: exactly antiparallel to the relative velocity.
        assert!(f.drag.dot(state.linvel) < 0.0);
        assert!(f.drag.normalize().dot(state.linvel.normalize()) < -0.999_999);
        // Magnitude: k · f · m · |v| with k = 2, f = 1.
        let expected = 2.0 * 500.0 * state.linvel.length();
        assert!((f.drag.length() - expected).abs() < 1e-9);
        // Torque opposes the spin.
        assert!(f.torque.dot(state.angvel) < 0.0);

        // A hostile coefficient is clamped rather than allowed to explode: the
        // per-step velocity change never exceeds 90 % of the velocity.
        let hostile = BuoyancyDesc3D {
            linear_drag: 10_000.0,
            ..desc
        };
        let hf = solve(&idx, 0.0, &state, &geo, &hostile, g, dt);
        let dv = hf.drag * (dt / state.mass);
        assert!(
            dv.length() <= state.linvel.length() * 0.9 + 1e-9,
            "clamped Δv {} vs |v| {}",
            dv.length(),
            state.linvel.length()
        );
    }

    #[test]
    fn a_dry_body_gets_no_force_at_all() {
        let mut idx = WaterIndex::default();
        idx.rebuild(vec![still_lake(1, 0.0, DVec2::ZERO, DVec2::splat(5.0))]);
        let geo = unit_box();
        let desc = BuoyancyDesc3D {
            density_kg_m3: 500.0,
            fluid_density_kg_m3: 1000.0,
            linear_drag: 2.0,
            angular_drag: 2.0,
        };
        let state = BodyState {
            translation: DVec3::new(500.0, 0.0, 0.0), // far outside the lake
            rotation: DQuat::IDENTITY,
            linvel: DVec3::new(10.0, 0.0, 0.0),
            angvel: DVec3::new(1.0, 0.0, 0.0),
            mass: 500.0,
        };
        let f = solve(
            &idx,
            0.0,
            &state,
            &geo,
            &desc,
            DVec3::new(0.0, -9.81, 0.0),
            1.0 / 60.0,
        );
        assert_eq!(f.drag, DVec3::ZERO);
        assert_eq!(f.torque, DVec3::ZERO);
        assert!(f.samples.iter().all(|(v, _)| *v == DVec3::ZERO));
    }

    /// A tilted body must be righted: the deeper side gets the bigger force, and
    /// applying each force where it is generated turns that into a moment.
    #[test]
    fn a_tilted_body_gets_a_righting_moment() {
        let mut idx = WaterIndex::default();
        idx.rebuild(vec![still_lake(1, 0.0, DVec2::ZERO, DVec2::splat(50.0))]);
        let geo = sample_geometry(
            &ColliderShape3D::Box {
                half_extents: DVec3::new(2.0, 0.25, 2.0),
            },
            DVec3::ZERO,
        );
        let desc = BuoyancyDesc3D {
            density_kg_m3: 500.0,
            fluid_density_kg_m3: 1000.0,
            linear_drag: 0.0,
            angular_drag: 0.0,
        };
        // Roll 25° about +Z: the -X side goes down, so it should be pushed up
        // harder, producing a torque about +Z that rolls it back.
        let rot = DQuat::from_rotation_z(25f64.to_radians());
        let state = BodyState {
            translation: DVec3::ZERO,
            rotation: rot,
            linvel: DVec3::ZERO,
            angvel: DVec3::ZERO,
            mass: 500.0 * 4.0,
        };
        let f = solve(
            &idx,
            0.0,
            &state,
            &geo,
            &desc,
            DVec3::new(0.0, -9.81, 0.0),
            1.0 / 60.0,
        );
        let torque: DVec3 = f
            .samples
            .iter()
            .map(|(force, p)| (*p - state.translation).cross(*force))
            .sum();
        assert!(
            torque.z < -1e-6,
            "a raft rolled +25° about Z must be pushed back: {torque:?}"
        );
        // Level, the same raft gets no moment at all.
        let level = BodyState {
            rotation: DQuat::IDENTITY,
            ..state
        };
        let lf = solve(
            &idx,
            0.0,
            &level,
            &geo,
            &desc,
            DVec3::new(0.0, -9.81, 0.0),
            1.0 / 60.0,
        );
        let lt: DVec3 = lf
            .samples
            .iter()
            .map(|(force, p)| (*p - level.translation).cross(*force))
            .sum();
        assert!(lt.length() < 1e-9, "a level raft must not rotate: {lt:?}");
    }

    #[test]
    fn crossings_latch_with_hysteresis_and_splash_above_the_threshold() {
        let guid = Uuid::from_u128(7);
        let mut state = BuoyantState::default();
        let geo = unit_box();
        let hyst = exit_hysteresis_m(&geo);
        assert_eq!(hyst, 0.05, "5 % of a 1 m box");
        let mut out = Vec::new();
        let p = |depth: f64| WaterProbe {
            water: Uuid::from_u128(99),
            surface_y: 0.0,
            fraction: 0.5,
            depth_m: depth,
            flow: DVec2::ZERO,
        };

        // Gentle entry: Enter, no splash.
        crossing_events(guid, &mut state, &p(0.01), hyst, 0.5, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, WaterEventKind3D::Enter);
        assert_eq!(out[0].water, Uuid::from_u128(99));
        out.clear();

        // Bobbing inside the hysteresis band fires nothing.
        for d in [-0.01, -0.04, 0.02, -0.049] {
            crossing_events(guid, &mut state, &p(d), hyst, 0.1, &mut out);
        }
        assert!(out.is_empty(), "the band must not chatter: {out:?}");

        // Lifted clear: Exit, and fast enough to splash.
        crossing_events(guid, &mut state, &p(-0.2), hyst, 3.0, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].kind, WaterEventKind3D::Exit);
        assert_eq!(out[1].kind, WaterEventKind3D::Splash);
        assert_eq!(out[1].speed_m_s, 3.0);
        out.clear();

        // Dropped back in fast: Enter + Splash.
        crossing_events(guid, &mut state, &p(0.3), hyst, -6.0, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].kind, WaterEventKind3D::Enter);
        assert_eq!(
            out[1].speed_m_s, 6.0,
            "speed is reported as a magnitude, whichever way it crossed"
        );
    }

    #[test]
    fn the_swim_latch_has_a_hysteresis_band() {
        assert!(!swim_latch(false, 0.5), "wading is not swimming");
        assert!(swim_latch(false, 0.6));
        assert!(swim_latch(true, 0.5), "still swimming inside the band");
        assert!(!swim_latch(true, 0.44));
        // The band itself, spelled out through `swim_latch` rather than as a
        // comparison of two constants (which clippy rightly folds away): there
        // must exist a fraction at which the answer depends on where you came
        // from, and that is what a hysteresis band IS.
        let band = SWIM_EXIT_FRACTION + (SWIM_ENTER_FRACTION - SWIM_EXIT_FRACTION) * 0.5;
        assert!(
            swim_latch(true, band) && !swim_latch(false, band),
            "the swim hysteresis band collapsed"
        );
    }

    #[test]
    fn swim_motion_beats_free_fall_caps_speed_and_balances_buoyancy() {
        let dt = 1.0 / 60.0;
        // THE load-bearing case: a character doing nothing but integrating
        // gravity arrives asking to fall at 20 m/s, and must SURFACE anyway.
        let falling = swim_motion(DVec3::new(0.0, -20.0 * dt, 0.0), 1.0, dt);
        assert!(
            falling.y > 0.0,
            "a fully submerged character must rise even while asking to fall: {}",
            falling.y / dt
        );
        // At the float depth, with no vertical asked for, nothing happens.
        let neutral = swim_motion(DVec3::ZERO, SWIM_FLOAT_SUBMERSION, dt);
        assert!(neutral.y.abs() < 1e-12);
        // Deliberate diving still works — it is just weaker than rising.
        let dive = swim_motion(DVec3::new(0.0, -2.0 * dt, 0.0), SWIM_FLOAT_SUBMERSION, dt);
        let rise = swim_motion(DVec3::new(0.0, 2.0 * dt, 0.0), SWIM_FLOAT_SUBMERSION, dt);
        assert!(dive.y < 0.0, "a swimmer must be able to dive");
        assert!(rise.y > 0.0);
        assert!(
            rise.y > -dive.y,
            "rising must have more authority than sinking: {} vs {}",
            rise.y,
            -dive.y
        );
        assert!((rise.y / dt - SWIM_VERT_MAX_M_S).abs() < 1e-12);
        // Horizontal is capped.
        let sprint = swim_motion(DVec3::new(9.0 * dt, 0.0, 12.0 * dt), 0.9, dt);
        let speed = DVec2::new(sprint.x, sprint.z).length() / dt;
        assert!(
            (speed - SWIM_SPEED_MAX_M_S).abs() < 1e-9,
            "capped to {SWIM_SPEED_MAX_M_S}, got {speed}"
        );
        // A slow swimmer is not sped up.
        let dawdle = swim_motion(DVec3::new(0.5 * dt, 0.0, 0.0), 0.9, dt);
        assert!((dawdle.x - 0.5 * dt).abs() < 1e-12);
        // A degenerate step is the identity rather than a divide by zero.
        let m = DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(swim_motion(m, 1.0, 0.0), m);
    }

    #[test]
    fn the_stamp_notices_every_thing_a_surface_is_built_from() {
        let guid = Uuid::from_u128(3);
        let body = WaterBody::lake(4.0, Vec2d::splat(20.0));
        let base = WaterStamp::new(guid, body, None, DAffine3::IDENTITY);
        assert_eq!(base, WaterStamp::new(guid, body, None, DAffine3::IDENTITY));
        assert_eq!(base.guid(), guid);

        let moved = WaterStamp::new(
            guid,
            body,
            None,
            DAffine3::from_translation(DVec3::new(1.0, 0.0, 0.0)),
        );
        assert_ne!(base, moved, "a moved lake is a different lake");

        let raised = WaterStamp::new(
            guid,
            WaterBody {
                level_m: 4.5,
                ..body
            },
            None,
            DAffine3::IDENTITY,
        );
        assert_ne!(base, raised);

        let sp = Spline {
            points: vec![
                inf_ecs::math::Vec3d::ZERO,
                inf_ecs::math::Vec3d::new(10.0, 0.0, 0.0),
            ],
            ..Spline::default()
        };
        let with = WaterStamp::new(guid, body, Some(&sp), DAffine3::IDENTITY);
        assert_ne!(base, with);
        let mut moved_points = sp.clone();
        moved_points.points[1] = inf_ecs::math::Vec3d::new(10.0, 0.0, 1e-9);
        assert_ne!(
            with,
            WaterStamp::new(guid, body, Some(&moved_points), DAffine3::IDENTITY),
            "a nanometre of spline movement is still movement"
        );
    }

    #[test]
    fn the_component_maps_onto_the_facade_descriptor() {
        let on = Buoyancy::of_density(500.0);
        let d = BuoyancyDesc3D::from_component(&on).unwrap();
        assert_eq!(d.density_kg_m3, 500.0);
        assert_eq!(d.fluid_density_kg_m3, 1000.0);
        assert!(d.is_usable());

        let off = Buoyancy {
            enabled: false,
            ..on
        };
        assert!(BuoyancyDesc3D::from_component(&off).is_none());

        // Negative drag is clamped to zero rather than accelerating the body.
        let negative = Buoyancy {
            linear_drag: -5.0,
            angular_drag: -1.0,
            ..on
        };
        let nd = BuoyancyDesc3D::from_component(&negative).unwrap();
        assert_eq!(nd.linear_drag, 0.0);
        assert_eq!(nd.angular_drag, 0.0);

        let vacuum = Buoyancy {
            density_kg_m3: 0.0,
            ..on
        };
        assert!(!BuoyancyDesc3D::from_component(&vacuum).unwrap().is_usable());
    }
}
