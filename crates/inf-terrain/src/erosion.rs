//! **Hydraulic + thermal erosion** — the Gaea-heritage terrain differentiator's
//! CPU reference (P10.3a).
//!
//! This is the *virtual-pipes shallow-water* model (Mei / Decaudin / Hu style):
//! the industry-standard interactive-erosion formulation, chosen because every
//! pass maps 1:1 onto a GPU compute shader over textures — so the P10.3b WGSL
//! port reproduces these exact semantics. This module is the **golden reference**
//! the shader is validated against.
//!
//! Erosion operates on a [`HeightRegion`] — the same dense-grid CPU/GPU seam the
//! sculpt brushes use ([`region`](crate::region)). [`erode_terrain`] wraps the
//! extract → erode → write-back round-trip so a stroke of erosion is an
//! undo-compatible [`HeightDelta`] for free.
//!
//! # The simulation state
//!
//! Per cell (dense `nx · nz` `f32` grids sized to the region):
//!
//! | field | symbol | meaning |
//! |-------|--------|---------|
//! | `b`   | terrain height (region offset, metres) |
//! | `d`   | water depth |
//! | `s`   | suspended sediment |
//! | `fl,fr,ft,fb` | outflow flux to the ±x / ±z neighbour (volume · s⁻¹) |
//! | `u,v` | water velocity |
//!
//! # One step (fixed `dt`, order is load-bearing — the WGSL port must match)
//!
//! 1. **Rain** — `d += dt · rain · R(x,z)`, `R` a seeded world-anchored fBm factor
//!    (uniform when `rain_variation == 0`). Applied to authored cells only.
//! 2. **Flux** — for each of the 4 pipes,
//!    `f ← max(0, f + dt · A · g · Δh / l)` where `Δh` is the water-surface
//!    (`b + d`) difference across the pipe, `A` the pipe cross-section, `g`
//!    gravity, `l` the cell spacing. Then a scale
//!    `K = min(1, d · l² / ((fl+fr+ft+fb) · dt))` guarantees a cell never ships
//!    more water than it holds (⇒ `d ≥ 0` always — the closed-box conservation
//!    guard).
//! 3. **Water** — `d += dt · (Σ inflow − Σ outflow) / l²`.
//! 4. **Velocity** — from the net flux through each axis divided by the mean water
//!    column (`u = ΔWx / (l · d̄)`), guarded against the dry-cell singularity.
//! 5. **Erosion / deposition** — capacity `C = Kc · sin(tilt) · |vel|` (tilt
//!    clamped to `min_tilt` so dead-flat fast flow still carries a little). Under
//!    capacity, erode `Ks · (C − s)` (capped by `max_erode_depth` per step — the
//!    hard per-step floor); over capacity, deposit `Kd · (s − C)`.
//! 6. **Sediment advection** — semi-Lagrangian backtrace `p = cell − vel · dt / l`,
//!    bilinear sample of `s`, clamped to the grid.
//! 7. **Evaporation** — `d ·= 1 − Ke · dt`.
//!
//! **Thermal erosion** (step 8, every `thermal_every` steps): for slopes steeper
//! than the talus angle, material is moved to lower 8-neighbours proportional to
//! each neighbour's slope excess. It is computed into a separate delta buffer and
//! applied in one pass, so it is **mass-conserving and order-independent** by
//! construction.
//!
//! # Boundary policy
//!
//! * **Region outer edge is open**: an off-grid neighbour is a dry ghost whose
//!   water surface equals the edge cell's *terrain* (`Δh = d`), so standing water
//!   drains out of the region and is counted in [`ErosionStats::water_out`]; no
//!   water flows *in* from outside.
//! * **Unauthored (hole) cells are walls**: no flux crosses between an authored
//!   cell and a hole in either direction, and thermal never moves material into a
//!   hole. Holes are never read as terrain nor written back (matching
//!   [`HeightRegion::write_back`]). Surrounding an authored patch with a ring of
//!   holes therefore makes a *closed box* (the water-conservation test).
//!
//! # Determinism
//!
//! [`erode`] is a pure function of `(region, params, steps)`: fixed row-major
//! iteration order, `f32` state (GPU-parity), no wall-clock, no hashing container;
//! the only randomness is the seeded, world-anchored [`fbm_signed`] rain factor.
//! Same input ⇒ byte-identical output (the determinism test hashes both runs).
//!
//! # WGSL-port map (for the P10.3b implementer)
//!
//! Each CPU field becomes a storage texture; the passes become compute
//! dispatches. Use **ping-pong pairs** for every field a pass both reads a
//! neighbourhood of and writes:
//!
//! | CPU field | texture | format | ping-pong? |
//! |-----------|---------|--------|-----------|
//! | `b` | `terrain` | `r32float` | yes (thermal + erode/deposit) |
//! | `d` | `water` | `r32float` | yes (water/evap read old for velocity) |
//! | `s` | `sediment` | `r32float` | yes (advection reads a neighbourhood) |
//! | `fl,fr,ft,fb` | `flux` | `rgba32float` (one texel = 4 pipes) | in-place OK (flux update reads only `b,d`) |
//! | `u,v` | `velocity` | `rg32float` | in-place OK |
//! | `authored` | `mask` | `r8uint` | read-only |
//!
//! Suggested layout: one `@compute @workgroup_size(8, 8)` entry point per numbered
//! pass, dispatched `ceil(nx/8) × ceil(nz/8)`; the mask sampled per texel gates
//! walls; boundary reads clamp to edge and apply the open-ghost rule. Tiling for
//! terrains larger than a single dispatch adds a 1-cell (thermal: also 1-cell)
//! halo per tile — the flux/thermal stencils are one-ring. The scalar reductions
//! that produce [`ErosionStats`] become an atomic-add or a follow-up reduction
//! pass; they are diagnostics and need not be bit-exact on GPU.

use serde::{Deserialize, Serialize};

use crate::data::TerrainData;
use crate::delta::HeightDelta;
use crate::noise::fbm_signed;
use crate::region::HeightRegion;
use glam::DVec2;

/// The 8 neighbour offsets `(dx, dz)` used by thermal erosion, in a fixed order
/// (orthogonal then diagonal) so the pass is deterministic.
const N8: [(i32, i32); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

/// Tunable parameters for [`erode`]. All rates are per **step** unless noted;
/// defaults are a sane, stable starting point for 1 m-spaced terrain. `serde` +
/// [`Default`] so a terrain-graph erosion node can persist and diff them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErosionParams {
    /// Integration time step. Smaller is more stable; the flux clamp keeps the
    /// sim bounded regardless. Range ~`0.005..0.05`. Default `0.02`.
    pub dt: f32,
    /// Water added per cell per unit time (before the `dt` factor). Range
    /// `0.0..0.1`. Default `0.02`.
    pub rain_rate: f32,
    /// Fractional fBm modulation of the rain (`0` = perfectly uniform rain,
    /// `0.5` = ±50 % spatial variation). Range `0.0..1.0`. Default `0.0`.
    pub rain_variation: f32,
    /// Seed for the rain-variation fBm (world-anchored, so rain is reproducible).
    /// Default `0`.
    pub rain_seed: u64,
    /// Base spatial frequency of the rain fBm (cycles per world metre). Default
    /// `0.02`.
    pub rain_frequency: f64,
    /// Octaves of the rain fBm. Default `3`.
    pub rain_octaves: u32,
    /// Evaporation coefficient `Ke` (`d *= 1 − Ke·dt` each step). Range
    /// `0.0..0.1`. Default `0.015`.
    pub evaporation: f32,
    /// Gravity `g` driving the pipe flux. Default `9.81`.
    pub gravity: f32,
    /// Virtual-pipe cross-section area `A`. Larger = faster water response. Range
    /// `0.5..20`. Default `1.0`.
    pub pipe_area: f32,
    /// Sediment capacity constant `Kc` (`C = Kc·sin(tilt)·|vel|`). Range
    /// `0.1..4`. Default `1.0`.
    pub sediment_capacity: f32,
    /// Dissolving (erosion) rate `Ks` — how fast terrain dissolves toward
    /// capacity. Range `0.0..1.0`. Default `0.5`.
    pub dissolving: f32,
    /// Deposition rate `Kd` — how fast surplus sediment settles. Range
    /// `0.0..1.0`. Default `1.0`.
    pub deposition: f32,
    /// Minimum `sin(tilt)` used in the capacity term, so fast flow over flat
    /// ground still transports (avoids a dead-flat stall). Range `0.001..0.1`.
    /// Default `0.01`.
    pub min_tilt: f32,
    /// Hard cap on how much terrain a single step may erode from one cell (the
    /// per-step floor). Range `0.01..2`. Default `0.5`.
    pub max_erode_depth: f32,
    /// Talus angle in **degrees** — thermal erosion relaxes slopes toward this
    /// (the material angle of repose, ~30–40°). Default `33.0`.
    pub thermal_talus_deg: f32,
    /// Fraction of the local height excess thermal erosion moves per step. Range
    /// `0.0..1.0` (`0` disables thermal). Default `0.5`.
    pub thermal_rate: f32,
    /// Hard cap on thermal material moved from one cell per step. Default `0.5`.
    pub max_thermal_depth: f32,
    /// Run the thermal pass every N steps (`1` = every step). Default `1`.
    pub thermal_every: u32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            dt: 0.02,
            rain_rate: 0.02,
            rain_variation: 0.0,
            rain_seed: 0,
            rain_frequency: 0.02,
            rain_octaves: 3,
            evaporation: 0.015,
            gravity: 9.81,
            pipe_area: 1.0,
            sediment_capacity: 1.0,
            dissolving: 0.5,
            deposition: 1.0,
            min_tilt: 0.01,
            max_erode_depth: 0.5,
            thermal_talus_deg: 33.0,
            thermal_rate: 0.5,
            max_thermal_depth: 0.5,
            thermal_every: 1,
        }
    }
}

/// Mass-accounting diagnostics returned by [`erode`]. Volumes are in world³
/// (metres³): a per-cell depth times the cell area `l²`. Determinism note: these
/// accumulate in `f64`, so they are stable across runs but not the *definition*
/// of the result (the terrain heights are).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErosionStats {
    /// Total water volume added by rain over all steps.
    pub water_in: f64,
    /// Total water volume that left the system: open-boundary outflow plus
    /// evaporation. In a closed box (hole-walled) with no evaporation this is `0`.
    pub water_out: f64,
    /// Cumulative volume of terrain eroded into suspension over all steps (`≥ 0`)
    /// — the erosion-activity measure.
    pub sediment_moved: f64,
    /// Net terrain volume change `Σ(b_final − b_init)·l²`. Negative = the surface
    /// net-eroded; `≈ 0` for a pure thermal pass (thermal conserves terrain mass).
    pub mass_delta: f64,
    /// Water volume still present in the region at the end (`Σ d · l²`). For the
    /// closed-box conservation test this equals [`water_in`](Self::water_in).
    pub water_present: f64,
}

/// The dense working state of one erosion run. Kept private; the public surface is
/// [`erode`] / [`erode_with`] / [`erode_terrain`].
struct Sim {
    nx: u32,
    nz: u32,
    /// Cell spacing in world metres (== the terrain's metres-per-sample).
    l: f32,
    /// Cell area `l²` (world metres²), as `f64` for volume accounting.
    area: f64,
    /// World XZ of cell `(0, 0)` and the sample stride, for the world-anchored
    /// rain fBm.
    wx0: f64,
    wz0: f64,
    mps: f64,
    b: Vec<f32>,
    d: Vec<f32>,
    s: Vec<f32>,
    fl: Vec<f32>,
    fr: Vec<f32>,
    ft: Vec<f32>,
    fb: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    authored: Vec<bool>,
    // scratch buffers reused across steps (avoid per-step allocation)
    new_d: Vec<f32>,
    new_s: Vec<f32>,
    /// Snapshot of `b` at the start of the erode/deposit pass, so the tilt
    /// gradient reads pre-pass heights (ping-pong parity with the WGSL port).
    pre_b: Vec<f32>,
    // accounting
    b_init_sum: f64,
    water_in: f64,
    water_out: f64,
    sediment_moved: f64,
}

impl Sim {
    fn from_region(region: &HeightRegion) -> Self {
        let (nx, nz) = region.dims();
        let n = (nx as usize) * (nz as usize);
        // Cell spacing from the region's world mapping (no direct mps accessor).
        let l = if nx >= 2 {
            (region.world_xz(1, 0).x - region.world_xz(0, 0).x).abs()
        } else if nz >= 2 {
            (region.world_xz(0, 1).y - region.world_xz(0, 0).y).abs()
        } else {
            1.0
        };
        let l = if l > 0.0 { l } else { 1.0 };
        let origin = region.world_xz(0, 0);

        let mut b = vec![0.0f32; n];
        let mut authored = vec![false; n];
        let mut b_init_sum = 0.0f64;
        for r in 0..nz {
            for c in 0..nx {
                let i = (r * nx + c) as usize;
                if region.is_authored(c, r) {
                    let h = region.height(c, r);
                    b[i] = h;
                    authored[i] = true;
                    b_init_sum += h as f64;
                }
            }
        }

        Self {
            nx,
            nz,
            l: l as f32,
            area: l * l,
            wx0: origin.x,
            wz0: origin.y,
            mps: l,
            b,
            d: vec![0.0; n],
            s: vec![0.0; n],
            fl: vec![0.0; n],
            fr: vec![0.0; n],
            ft: vec![0.0; n],
            fb: vec![0.0; n],
            u: vec![0.0; n],
            v: vec![0.0; n],
            authored,
            new_d: vec![0.0; n],
            new_s: vec![0.0; n],
            pre_b: vec![0.0; n],
            b_init_sum,
            water_in: 0.0,
            water_out: 0.0,
            sediment_moved: 0.0,
        }
    }

    #[inline]
    fn idx(&self, c: u32, r: u32) -> usize {
        (r * self.nx + c) as usize
    }

    /// One full step: hydraulic passes 1–7, then thermal when it is due.
    fn step(&mut self, p: &ErosionParams, step_index: u32) {
        self.rain(p);
        self.flux(p);
        self.water_and_velocity(p);
        self.erode_deposit(p);
        self.advect_sediment(p);
        self.evaporate(p);
        if p.thermal_rate > 0.0 && step_index.is_multiple_of(p.thermal_every.max(1)) {
            self.thermal(p);
        }
    }

    // 1. rain
    fn rain(&mut self, p: &ErosionParams) {
        let add = p.dt * p.rain_rate;
        if add == 0.0 {
            return;
        }
        for r in 0..self.nz {
            for c in 0..self.nx {
                let i = self.idx(c, r);
                if !self.authored[i] {
                    continue;
                }
                let factor = if p.rain_variation != 0.0 {
                    let wx = self.wx0 + c as f64 * self.mps;
                    let wz = self.wz0 + r as f64 * self.mps;
                    let f = fbm_signed(p.rain_seed, p.rain_frequency, p.rain_octaves, wx, wz);
                    (1.0 + p.rain_variation as f64 * f).max(0.0) as f32
                } else {
                    1.0
                };
                let amt = add * factor;
                self.d[i] += amt;
                self.water_in += amt as f64 * self.area;
            }
        }
    }

    // 2. flux update (in place: reads only b, d and its own old flux)
    fn flux(&mut self, p: &ErosionParams) {
        let (nx, nz) = (self.nx, self.nz);
        let scale = p.dt * p.pipe_area * p.gravity / self.l;
        for r in 0..nz {
            for c in 0..nx {
                let i = self.idx(c, r);
                if !self.authored[i] {
                    // holes never carry flux (walls)
                    self.fl[i] = 0.0;
                    self.fr[i] = 0.0;
                    self.ft[i] = 0.0;
                    self.fb[i] = 0.0;
                    continue;
                }
                let self_surf = self.b[i] + self.d[i];

                // Δh per pipe: None ⇒ wall (hole neighbour), Some(open ghost = d)
                // for an off-grid neighbour, Some(surface diff) otherwise.
                let dh = |sim: &Sim, wall_open: bool, at_edge: bool, j: usize| -> Option<f32> {
                    if at_edge {
                        Some(sim.d[i]) // open boundary: dry ghost at terrain level
                    } else if wall_open {
                        Some(self_surf - (sim.b[j] + sim.d[j]))
                    } else {
                        None // hole wall
                    }
                };

                let nfl = self.pipe(self.fl[i], scale, {
                    if c == 0 {
                        dh(self, false, true, 0)
                    } else {
                        let j = i - 1;
                        dh(self, self.authored[j], false, j)
                    }
                });
                let nfr = self.pipe(self.fr[i], scale, {
                    if c == nx - 1 {
                        dh(self, false, true, 0)
                    } else {
                        let j = i + 1;
                        dh(self, self.authored[j], false, j)
                    }
                });
                let nft = self.pipe(self.ft[i], scale, {
                    if r == 0 {
                        dh(self, false, true, 0)
                    } else {
                        let j = i - nx as usize;
                        dh(self, self.authored[j], false, j)
                    }
                });
                let nfb = self.pipe(self.fb[i], scale, {
                    if r == nz - 1 {
                        dh(self, false, true, 0)
                    } else {
                        let j = i + nx as usize;
                        dh(self, self.authored[j], false, j)
                    }
                });

                let total = nfl + nfr + nft + nfb;
                if total > 0.0 {
                    // never ship more than the cell holds this step
                    let avail = self.d[i] * self.l * self.l;
                    let k = (avail / (total * p.dt)).min(1.0);
                    self.fl[i] = nfl * k;
                    self.fr[i] = nfr * k;
                    self.ft[i] = nft * k;
                    self.fb[i] = nfb * k;
                } else {
                    self.fl[i] = 0.0;
                    self.fr[i] = 0.0;
                    self.ft[i] = 0.0;
                    self.fb[i] = 0.0;
                }
            }
        }
    }

    /// New non-negative flux for one pipe (`0` for a wall).
    #[inline]
    fn pipe(&self, old: f32, scale: f32, dh: Option<f32>) -> f32 {
        match dh {
            Some(dh) => (old + scale * dh).max(0.0),
            None => 0.0,
        }
    }

    // 3 + 4. water update and velocity field
    fn water_and_velocity(&mut self, p: &ErosionParams) {
        let (nx, nz) = (self.nx, self.nz);
        for r in 0..nz {
            for c in 0..nx {
                let i = self.idx(c, r);
                if !self.authored[i] {
                    self.new_d[i] = 0.0;
                    continue;
                }
                let fr_left = if c > 0 { self.fr[i - 1] } else { 0.0 };
                let fl_right = if c < nx - 1 { self.fl[i + 1] } else { 0.0 };
                let fb_top = if r > 0 { self.fb[i - nx as usize] } else { 0.0 };
                let ft_bottom = if r < nz - 1 {
                    self.ft[i + nx as usize]
                } else {
                    0.0
                };

                let inflow = fr_left + fl_right + fb_top + ft_bottom;
                let outflow = self.fl[i] + self.fr[i] + self.ft[i] + self.fb[i];
                let d_new = (self.d[i] + p.dt * (inflow - outflow) / (self.l * self.l)).max(0.0);
                self.new_d[i] = d_new;

                // water that left the region through an open edge this step
                let mut leaving = 0.0f32;
                if c == 0 {
                    leaving += self.fl[i];
                }
                if c == nx - 1 {
                    leaving += self.fr[i];
                }
                if r == 0 {
                    leaving += self.ft[i];
                }
                if r == nz - 1 {
                    leaving += self.fb[i];
                }
                // flux is a volume rate (the K clamp measures it against d·l²),
                // so the volume that left this step is dt · flux.
                self.water_out += (p.dt * leaving) as f64;

                // velocity from the net axial flux over the mean water column
                let d_mean = 0.5 * (self.d[i] + d_new);
                if d_mean > 1e-6 {
                    let dwx = 0.5 * ((fr_left - self.fl[i]) + (self.fr[i] - fl_right));
                    let dwz = 0.5 * ((fb_top - self.ft[i]) + (self.fb[i] - ft_bottom));
                    self.u[i] = dwx / (self.l * d_mean);
                    self.v[i] = dwz / (self.l * d_mean);
                } else {
                    self.u[i] = 0.0;
                    self.v[i] = 0.0;
                }
            }
        }
        // commit the new depths
        std::mem::swap(&mut self.d, &mut self.new_d);
    }

    // 5. erosion / deposition against sediment capacity
    fn erode_deposit(&mut self, p: &ErosionParams) {
        let (nx, nz) = (self.nx, self.nz);
        // Read the tilt gradient from a pre-pass snapshot so the pass is
        // order-independent (each cell only writes its own b[i]).
        self.pre_b.copy_from_slice(&self.b);
        for r in 0..nz {
            for c in 0..nx {
                let i = self.idx(c, r);
                if !self.authored[i] {
                    continue;
                }
                let sin_tilt = self.sin_tilt(c, r).max(p.min_tilt);
                let speed = (self.u[i] * self.u[i] + self.v[i] * self.v[i]).sqrt();
                let capacity = p.sediment_capacity * sin_tilt * speed;
                let s = self.s[i];
                if capacity > s {
                    // erode toward capacity, capped by the per-step floor
                    let amt = (p.dissolving * (capacity - s))
                        .min(p.max_erode_depth)
                        .max(0.0);
                    self.b[i] -= amt;
                    self.s[i] = s + amt;
                    self.sediment_moved += amt as f64 * self.area;
                } else {
                    // deposit the surplus (never more than is suspended)
                    let amt = (p.deposition * (s - capacity)).min(s).max(0.0);
                    self.b[i] += amt;
                    self.s[i] = s - amt;
                }
            }
        }
    }

    /// `sin` of the local terrain tilt from a central-difference gradient over the
    /// `pre_b` snapshot (one-sided at edges / next to holes).
    #[inline]
    fn sin_tilt(&self, c: u32, r: u32) -> f32 {
        let i = self.idx(c, r);
        let here = self.pre_b[i];
        let bx0 = self.neighbour_b(c, r, -1, 0).unwrap_or(here);
        let bx1 = self.neighbour_b(c, r, 1, 0).unwrap_or(here);
        let bz0 = self.neighbour_b(c, r, 0, -1).unwrap_or(here);
        let bz1 = self.neighbour_b(c, r, 0, 1).unwrap_or(here);
        let gx = (bx1 - bx0) / (2.0 * self.l);
        let gz = (bz1 - bz0) / (2.0 * self.l);
        let slope = (gx * gx + gz * gz).sqrt();
        slope / (1.0 + slope * slope).sqrt()
    }

    /// Terrain height (from the `pre_b` snapshot) of the authored neighbour at
    /// `(c+dx, r+dz)`, or `None` when off-grid or a hole.
    #[inline]
    fn neighbour_b(&self, c: u32, r: u32, dx: i32, dz: i32) -> Option<f32> {
        let nc = c as i32 + dx;
        let nr = r as i32 + dz;
        if nc < 0 || nr < 0 || nc >= self.nx as i32 || nr >= self.nz as i32 {
            return None;
        }
        let j = self.idx(nc as u32, nr as u32);
        self.authored[j].then_some(self.pre_b[j])
    }

    // 6. semi-Lagrangian sediment advection
    fn advect_sediment(&mut self, p: &ErosionParams) {
        let (nx, nz) = (self.nx, self.nz);
        for r in 0..nz {
            for c in 0..nx {
                let i = self.idx(c, r);
                if !self.authored[i] {
                    self.new_s[i] = 0.0;
                    continue;
                }
                let px = c as f32 - self.u[i] * p.dt / self.l;
                let pz = r as f32 - self.v[i] * p.dt / self.l;
                self.new_s[i] = self.sample_s(px, pz);
            }
        }
        std::mem::swap(&mut self.s, &mut self.new_s);
    }

    /// Bilinear sample of the sediment field at fractional grid coords, clamped to
    /// the grid. Hole cells contribute `0` (their `s` is always `0`).
    #[inline]
    fn sample_s(&self, px: f32, pz: f32) -> f32 {
        let px = px.clamp(0.0, (self.nx - 1) as f32);
        let pz = pz.clamp(0.0, (self.nz - 1) as f32);
        let x0 = px.floor() as u32;
        let z0 = pz.floor() as u32;
        let x1 = (x0 + 1).min(self.nx - 1);
        let z1 = (z0 + 1).min(self.nz - 1);
        let fx = px - x0 as f32;
        let fz = pz - z0 as f32;
        let s00 = self.s[self.idx(x0, z0)];
        let s10 = self.s[self.idx(x1, z0)];
        let s01 = self.s[self.idx(x0, z1)];
        let s11 = self.s[self.idx(x1, z1)];
        let sx0 = s00 + (s10 - s00) * fx;
        let sx1 = s01 + (s11 - s01) * fx;
        sx0 + (sx1 - sx0) * fz
    }

    // 7. evaporation
    fn evaporate(&mut self, p: &ErosionParams) {
        let keep = 1.0 - p.evaporation * p.dt;
        if keep >= 1.0 {
            return;
        }
        for r in 0..self.nz {
            for c in 0..self.nx {
                let i = self.idx(c, r);
                if !self.authored[i] {
                    continue;
                }
                let before = self.d[i];
                let after = (before * keep).max(0.0);
                self.d[i] = after;
                self.water_out += (before - after) as f64 * self.area;
            }
        }
    }

    // 8. thermal erosion (mass-conserving; computed into a delta, then applied)
    fn thermal(&mut self, p: &ErosionParams) {
        let (nx, nz) = (self.nx, self.nz);
        let talus_tan = (p.thermal_talus_deg as f64).to_radians().tan();
        let l = self.l as f64;
        let sqrt2 = std::f32::consts::SQRT_2 as f64;
        let mut delta = vec![0.0f64; self.b.len()];

        for r in 0..nz {
            for c in 0..nx {
                let i = self.idx(c, r);
                if !self.authored[i] {
                    continue;
                }
                let bi = self.b[i] as f64;
                let mut excess = [0.0f64; 8];
                let mut sum_excess = 0.0f64;
                let mut max_diff = 0.0f64;
                for (k, &(dx, dz)) in N8.iter().enumerate() {
                    let nc = c as i32 + dx;
                    let nr = r as i32 + dz;
                    if nc < 0 || nr < 0 || nc >= nx as i32 || nr >= nz as i32 {
                        continue;
                    }
                    let j = self.idx(nc as u32, nr as u32);
                    if !self.authored[j] {
                        continue; // hole wall: never move material into a hole
                    }
                    let diff = bi - self.b[j] as f64;
                    if diff <= 0.0 {
                        continue;
                    }
                    let dist = if dx == 0 || dz == 0 { l } else { l * sqrt2 };
                    if diff / dist > talus_tan {
                        let e = diff - talus_tan * dist; // height above the talus plane
                        if e > 0.0 {
                            excess[k] = e;
                            sum_excess += e;
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }
                    }
                }
                if sum_excess <= 0.0 {
                    continue;
                }
                let move_total = (p.thermal_rate as f64 * 0.5 * max_diff)
                    .min(p.max_thermal_depth as f64)
                    .max(0.0);
                if move_total <= 0.0 {
                    continue;
                }
                for (k, &(dx, dz)) in N8.iter().enumerate() {
                    if excess[k] <= 0.0 {
                        continue;
                    }
                    let j = self.idx((c as i32 + dx) as u32, (r as i32 + dz) as u32);
                    let give = move_total * excess[k] / sum_excess;
                    delta[j] += give;
                    delta[i] -= give;
                }
            }
        }

        for (i, &dl) in delta.iter().enumerate() {
            if self.authored[i] {
                self.b[i] = (self.b[i] as f64 + dl) as f32;
            }
        }
    }

    /// Write the current terrain heights back into the region (authored cells
    /// only — holes are never touched, matching `write_back`).
    fn flush_terrain(&self, region: &mut HeightRegion) {
        for r in 0..self.nz {
            for c in 0..self.nx {
                let i = self.idx(c, r);
                if self.authored[i] {
                    region.set_height(c, r, self.b[i]);
                }
            }
        }
    }

    fn finish_stats(&self) -> ErosionStats {
        let mut b_now = 0.0f64;
        let mut water = 0.0f64;
        for i in 0..self.b.len() {
            if self.authored[i] {
                b_now += self.b[i] as f64;
                water += self.d[i] as f64;
            }
        }
        ErosionStats {
            water_in: self.water_in,
            water_out: self.water_out,
            sediment_moved: self.sediment_moved,
            mass_delta: (b_now - self.b_init_sum) * self.area,
            water_present: water * self.area,
        }
    }
}

/// Run `steps` of hydraulic + thermal erosion on `region` in place, returning the
/// mass-accounting [`ErosionStats`]. Pure and deterministic: identical inputs
/// produce byte-identical terrain (see the module docs).
pub fn erode(region: &mut HeightRegion, params: &ErosionParams, steps: u32) -> ErosionStats {
    let mut sim = Sim::from_region(region);
    for step in 1..=steps {
        sim.step(params, step);
    }
    sim.flush_terrain(region);
    sim.finish_stats()
}

/// Like [`erode`], but calls `on_step(step, region)` after each step with the
/// terrain flushed into `region` — the seam a future interactive preview drives
/// (watch channels carve, scrub the step count). Slower than [`erode`] because it
/// flushes every step.
pub fn erode_with<F: FnMut(u32, &HeightRegion)>(
    region: &mut HeightRegion,
    params: &ErosionParams,
    steps: u32,
    mut on_step: F,
) -> ErosionStats {
    let mut sim = Sim::from_region(region);
    for step in 1..=steps {
        sim.step(params, step);
        sim.flush_terrain(region);
        on_step(step, region);
    }
    if steps == 0 {
        sim.flush_terrain(region);
    }
    sim.finish_stats()
}

/// Extract the world AABB `[min, max]` (expanded by `margin` samples), erode it,
/// and write it back through the [`HeightRegion`] seam — yielding an
/// undo-compatible [`HeightDelta`] alongside the [`ErosionStats`]. This is the
/// editor-facing entry point (drag an erosion region, get one undo step).
///
/// `margin` gives the simulation a border so open-boundary draining happens
/// outside the region of interest; a few samples is plenty.
pub fn erode_terrain(
    data: &mut TerrainData,
    min: DVec2,
    max: DVec2,
    margin: u32,
    params: &ErosionParams,
    steps: u32,
) -> (HeightDelta, ErosionStats) {
    let mut stats = ErosionStats::default();
    let delta = data.edit_region(min, max, margin, |region| {
        stats = erode(region, params, steps);
    });
    (delta, stats)
}
