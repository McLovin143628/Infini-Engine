//! **The many-lights substrate** (wave PAR0): the frame's light list as a
//! storage buffer, a camera-fitted froxel grid built on the GPU, and the pure
//! CPU plan that decides which lights reach the buffer.
//!
//! # What it replaced
//!
//! Until PAR0 every lit pass owned a private 1 040 B *uniform* holding the
//! first sixteen lights in scene order (`passes::mesh::MAX_LIGHTS`), and six
//! WGSL files each looped over it with a private copy of the BRDF. A nightlife
//! strip that built 99 fixtures kept twelve (the content layer pre-truncated
//! to fit the sixteen), and a 21:00 city could light nothing it built.
//!
//! # The shape now
//!
//! * [`plan_lights`] — a **pure function** of the scene's lights, the view and
//!   the [`LightSettings`]: frustum cull, energy cull (a light is bounded by the
//!   radius past which it delivers less than [`LIGHT_ILLUMINANCE_FLOOR`]), and
//!   — only past [`LIGHTS_PER_FRAME_CEILING`] — a priority cut by on-screen
//!   footprint. Directional lights first, then the surviving local lights, both
//!   in **scene order**, so the list is identical on every host that projects
//!   the same scene from the same camera.
//! * [`LightGrid`] — the two GPU buffers every lit shader reads (a 96 B header
//!   uniform + one storage buffer holding the records, the per-froxel counts and
//!   the per-froxel index lists), the froxel-build compute
//!   (`shaders/light_cluster.wgsl`, one invocation per froxel, list order in →
//!   list order out, no atomics), and the census readback a gate uses to count
//!   what the SHADER saw rather than what the CPU submitted.
//! * `shaders/lights.wgsl` — the ONE shared include: the BRDF, the froxel
//!   lookup and the light loops, composed into every lit shader.
//!
//! # The local-light door PAR1 inherits
//!
//! A headlamp, a porch lamp and a room fixture are all one [`RenderLight`]:
//! a point or a spot (`inner_cos`/`outer_cos` = the cone), a `range`, a
//! `cast_shadows` request the [`shadow_policy`] grants or refuses per frame
//! (the shadow slot), and an `intensity` the host computes from the level clock
//! through `inf_ecs::sky::fixture_schedule` (the schedule). Nothing here knows
//! what kind of fixture it is drawing.

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec3};

use crate::camera::RenderView;
use crate::gpu::GpuContext;
use crate::scene::{LightBound, LightKind, RenderLight, RenderScene};

/// Froxel grid columns (screen x).
pub const CLUSTER_X: u32 = 16;
/// Froxel grid rows (screen y).
pub const CLUSTER_Y: u32 = 9;
/// Froxel depth slices: slice 0 is `[0, CLUSTER_NEAR_M)`, the last is
/// `[CLUSTER_FAR_M, ∞)`, and the 22 between are log-spaced.
pub const CLUSTER_Z: u32 = 24;
/// Froxels in the grid.
pub const CLUSTER_COUNT: u32 = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
/// The end of the first depth slice, metres.
pub const CLUSTER_NEAR_M: f32 = 1.0;
/// The start of the last depth slice, metres.
pub const CLUSTER_FAR_M: f32 = 1000.0;
/// The fractional depth pad each froxel's box is grown by, so a fragment whose
/// `log2` rounds across a slice boundary still finds every light reaching it.
pub const CLUSTER_DEPTH_PAD: f32 = 0.02;

/// **How many local lights one froxel can list.** A froxel's stored count is
/// the number that REACHED it; past this the fragment walks only the first
/// `CLUSTER_MAX_LIGHTS` (in list order) and the census reports the overflow.
/// A multiple of four, so no two froxels' index lists share a `vec4`.
pub const CLUSTER_MAX_LIGHTS: u32 = 256;

/// **The frame's light ceiling** — records the storage buffer holds, and the
/// most lights [`plan_lights`] ever emits. Read by name by
/// `par0_lights_gate` and by the instrument's census.
///
/// Minted from the island's own arithmetic, not a round number: a lit Harbour
/// City at 21:00 wants ≈1 670 lights and the whole island ≈5 000 (the parity
/// memo's CP-B10, over measured content); 8 192 holds the island with 1.6×
/// headroom at 512 KiB of records, and the light-loop curve in the PAR0 ledger
/// measures the frame at 5 000.
pub const LIGHTS_PER_FRAME_CEILING: usize = 8192;

/// **The energy floor**: the illuminance (light units × 1/m², at exposure 1)
/// below which a light's contribution is under one 8-bit step of a white
/// diffuse surface after the sRGB encode (1/255/12.92 ≈ 3.0e-4 of linear white,
/// × π for a Lambertian albedo of 1 → 9.5e-4). A light delivers at most
/// `I · max(rgb) / d²`, so past `sqrt(I · max(rgb) / floor)` it lights nothing
/// a pixel can show — that radius is its cull sphere (or its `range`, if
/// shorter). Divided by the frame's exposure, so a brighter exposure lowers it.
pub const LIGHT_ILLUMINANCE_FLOOR: f32 = 9.5e-4;

/// The camera lattice the [`shadow_policy`] evaluates on, metres. The policy is
/// a pure function of the SNAPPED camera, so the shadowed set (and with it the
/// virtual-shadow address space, which is rebuilt when the set changes) moves
/// only when the camera crosses a cell, never every frame.
pub const SHADOW_POLICY_CELL_M: f64 = 8.0;

/// How near the camera a shadow-asking local light's cull sphere must come
/// to be a candidate for the [`shadow_policy`], metres. Forty-eight is the
/// distance at which a one-metre interior caster's shadow from a 9 m-range
/// room fixture is under a pixel at 1080p / 60° (1 m / 48 m × 935 px ≈ 19 px
/// for the caster, its penumbra a few) — the brightest-first ranking decides
/// within it, the reach only stops a far fixture from holding a page tree.
pub const SHADOW_POLICY_REACH_M: f64 = 48.0;

/// **How many local lights get a page tree** in one frame — `None` lets the
/// policy fill whatever the projection cap leaves after the directional lights
/// (`VSM_MAX_PROJECTIONS` = 64: the sun's clipmap levels, then six projections
/// a point light, one a spot).
pub const LOCAL_SHADOW_BUDGET: usize = 8;

/// Renderer-side light knobs. Never persisted (`RenderSettings` is not a wire
/// type), so none of these can move the scene schema.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSettings {
    /// Frustum + energy culling before the buffer. Off = every light in the
    /// scene reaches the buffer (up to the ceiling) — the arm that proves a
    /// culled light's removal changes no pixel compares the two.
    pub cull: bool,
    /// Upload local (point/spot) lights at all. Off = directional only — the
    /// instrument's WITHOUT row for the light-loop delta.
    pub local_lights: bool,
    /// The cluster-grid debug view: lit passes paint each froxel's light count.
    pub debug_view: bool,
    /// Grant `cast_shadows` to at most this many local lights a frame (the
    /// [`shadow_policy`]). 0 = no local light is shadowed.
    pub local_shadow_budget: usize,
}

impl Default for LightSettings {
    fn default() -> Self {
        Self {
            cull: true,
            local_lights: true,
            debug_view: false,
            local_shadow_budget: LOCAL_SHADOW_BUDGET,
        }
    }
}

/// One GPU light, std140-friendly (four `vec4`, 64 B). `pos_dir.w` = kind
/// (0 directional, 1 point, 2 spot, 3 a point clipped to a ROOM's box, 4 a
/// point clipped to an exterior box — PAR1a + its audit); for a directional light `pos_dir.xyz` is
/// the unit direction toward it, for point/spot the render-local position.
/// `color.a` = intensity. `params` = (range, spot inner_cos, spot outer_cos,
/// virtual-shadow slot + 1). `spot_dir.xyz` = the beam's emission axis (spot
/// only), `spot_dir.w` = the cull radius the froxel build tests (local only).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GpuLight {
    pub color: [f32; 4],
    pub pos_dir: [f32; 4],
    pub params: [f32; 4],
    pub spot_dir: [f32; 4],
}

/// The header uniform `lights.wgsl` and `light_cluster.wgsl` share (144 B).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct LightHeader {
    /// x = records, y = directional records (the first y), z = 1 when the scene
    /// carried no light at all, w = [`CLUSTER_MAX_LIGHTS`].
    pub counts: [u32; 4],
    /// xyz = froxel dims, w = froxel count.
    pub grid: [u32; 4],
    /// x = vec4 index of the froxel counts, y = vec4 index of the index lists,
    /// z = the debug view flag.
    pub bases: [u32; 4],
    /// x = near, y = far, z = the log2 slice scale, w = the depth pad.
    pub zparams: [f32; 4],
    /// xyz = camera forward (render-local).
    pub fwd: [f32; 4],
    /// xyz = render-local eye.
    pub eye: [f32; 4],
    /// xyz = camera right (render-local) — view-space x.
    pub right: [f32; 4],
    /// xyz = camera up, orthogonalised against `fwd` — view-space y.
    pub up: [f32; 4],
    /// x, y = the tile tangent scale (perspective: `tan(fov/2)` × aspect and
    /// `tan(fov/2)`; orthographic: the half extents in metres), z = 1 when
    /// orthographic, w = the far end of the last depth slice (the deepest
    /// point any listed light's sphere reaches, never less than
    /// [`CLUSTER_FAR_M`]).
    pub proj: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuLight>() == 64);
const _: () = assert!(std::mem::size_of::<LightHeader>() == 144);
const _: () = assert!(CLUSTER_MAX_LIGHTS.is_multiple_of(4));
const _: () = assert!(CLUSTER_COUNT.is_multiple_of(4));

/// `vec4` index of the first froxel count in the storage buffer.
pub const COUNT_BASE_VEC4: u32 = (LIGHTS_PER_FRAME_CEILING as u32) * 4;
/// `vec4` index of the first froxel index list.
pub const INDEX_BASE_VEC4: u32 = COUNT_BASE_VEC4 + CLUSTER_COUNT / 4;
/// Bytes of the light half of the one storage buffer.
pub const LIGHT_DATA_BYTES: u64 =
    (INDEX_BASE_VEC4 as u64 + (CLUSTER_COUNT as u64 * CLUSTER_MAX_LIGHTS as u64) / 4) * 16;

/// **Where the GI probe records live: in THIS buffer** (wave PAR0b, clause 5).
///
/// PAR0 put the light list in the environment group as its fifth fragment
/// storage buffer, which with the visibility resolve's four meshlet pools made
/// nine — one past `wgpu::Limits::default()` — and an 8-limit adapter lost the
/// meshlet tier. The GI probe records (`gi_sh`, the environment group's other
/// storage buffer that every lit fragment reads) now ride behind the light data
/// at a 256-byte-aligned offset (the storage-offset alignment every backend
/// grants): the probe march binds that sub-range read-write as its own
/// `array<vec4<f32>>`, the lit passes read it through the light words they
/// already bind (`gi_probe_word` in `lights.wgsl`). Back to eight.
pub const GI_PROBE_OFFSET_BYTES: u64 = LIGHT_DATA_BYTES.div_ceil(256) * 256;
/// The probe region's first `vec4` in the light words.
pub const GI_PROBE_BASE_VEC4: u32 = (GI_PROBE_OFFSET_BYTES / 16) as u32;
/// Bytes of the probe region — sized for the HIGHEST GI tier (the most probes),
/// so a tier change never reallocates the buffer the environment group binds.
pub const GI_PROBE_REGION_BYTES: u64 =
    crate::gi::probe_count() as u64 * crate::gi::PROBE_STRIDE_VEC4 as u64 * 16;
/// The whole buffer: the light words, the alignment pad, the probe records.
pub const LIGHT_BUFFER_BYTES: u64 = GI_PROBE_OFFSET_BYTES + GI_PROBE_REGION_BYTES;

/// **The frame's light list** — what [`plan_lights`] decided, before upload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LightPlan {
    /// The records, directional first, then local, both in scene order.
    pub records: Vec<GpuLight>,
    /// Per record, the scene light index it came from.
    pub scene_index: Vec<u32>,
    /// How many of `records` are directional (the first ones).
    pub directional: u32,
    /// The scene carried no light at all (the fallback editor sun).
    pub fallback_sun: bool,
    /// Local lights the frustum cull removed.
    pub culled_frustum: u32,
    /// Local lights whose energy sphere could not reach the frustum — counted
    /// inside `culled_frustum`'s test, reported apart for the census.
    pub culled_energy: u32,
    /// Local lights the ceiling's priority cut removed.
    pub culled_ceiling: u32,
    /// Local lights skipped because [`LightSettings::local_lights`] is off.
    pub skipped_local: u32,
    /// **Local lights past their own draw distance** (wave PAR1a): a room
    /// fixture whose sphere's nearest point is farther from the eye than its
    /// [`LightBound::draw_m`].
    pub culled_distance: u32,
    /// Local records emitted as a CLIPPED point (kind 3, wave PAR1a).
    pub clipped: u32,
}

impl LightPlan {
    /// The list as bytes — records then scene indices — for the
    /// PIE-vs-shipping byte compare.
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.records.len() * 68 + 8);
        out.extend_from_slice(&self.directional.to_le_bytes());
        out.extend_from_slice(&u32::from(self.fallback_sun).to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(&self.records));
        for i in &self.scene_index {
            out.extend_from_slice(&i.to_le_bytes());
        }
        out
    }

    /// Local records in the list.
    pub fn local(&self) -> u32 {
        self.records.len() as u32 - self.directional
    }
}

/// The radius past which `light` lights nothing a pixel can show, at
/// `exposure`: its `range` or its energy radius, whichever is shorter.
pub fn cull_radius(light: &RenderLight, exposure: f32) -> f32 {
    let peak = light.intensity.max(0.0) * light.color.iter().fold(0.0f32, |a, &c| a.max(c));
    let floor = LIGHT_ILLUMINANCE_FLOOR / exposure.max(1e-6);
    let energy = (peak / floor).sqrt();
    if light.range > 0.0 {
        light.range.min(energy)
    } else {
        energy
    }
}

/// The camera facts the CPU plan needs, in f64 world space.
#[derive(Debug, Clone, Copy)]
struct Viewer {
    eye: DVec3,
    fwd: DVec3,
    right: DVec3,
    up: DVec3,
    /// tan(half fov) on x and y (perspective), or half extents (ortho).
    half: (f64, f64),
    ortho: bool,
    /// Pixels per unit of tangent (perspective) or per metre (ortho), vertical.
    focal_px: f64,
}

impl Viewer {
    fn of(view: &RenderView) -> Self {
        let fwd = view.forward.as_dvec3().normalize_or(DVec3::NEG_Z);
        let right = fwd.cross(view.up.as_dvec3()).normalize_or(DVec3::X);
        let up = right.cross(fwd);
        let aspect = view.width.max(1) as f64 / view.height.max(1) as f64;
        match view.ortho {
            Some(o) => {
                let hh = o.half_height as f64;
                Self {
                    eye: view.eye_world,
                    fwd,
                    right,
                    up,
                    half: (hh * aspect, hh),
                    ortho: true,
                    focal_px: view.height.max(1) as f64 * 0.5 / hh.max(1e-6),
                }
            }
            None => {
                let ty = (view.fov_y as f64 * 0.5).tan();
                Self {
                    eye: view.eye_world,
                    fwd,
                    right,
                    up,
                    half: (ty * aspect, ty),
                    ortho: false,
                    focal_px: view.height.max(1) as f64 * 0.5 / ty.max(1e-6),
                }
            }
        }
    }

    /// Whether a sphere touches the view volume (conservative).
    fn sees(&self, c: DVec3, r: f64) -> bool {
        let d = c - self.eye;
        let z = d.dot(self.fwd);
        let x = d.dot(self.right);
        let y = d.dot(self.up);
        if z < -r {
            return false;
        }
        if self.ortho {
            return x.abs() - self.half.0 <= r && y.abs() - self.half.1 <= r;
        }
        // Distance to each side plane through the eye.
        let side = |a: f64, t: f64| (a.abs() - z * t) / (1.0 + t * t).sqrt();
        side(x, self.half.0) <= r && side(y, self.half.1) <= r
    }

    /// The sphere's on-screen radius in pixels (infinite with the eye inside).
    fn footprint_px(&self, c: DVec3, r: f64) -> f64 {
        if self.ortho {
            return r * self.focal_px;
        }
        let dist = (c - self.eye).length();
        if dist <= r {
            return f64::INFINITY;
        }
        r / dist * self.focal_px
    }
}

/// **The frame's light plan** — a pure function of the scene's lights, the
/// view, the per-scene-light virtual-shadow slots and the settings.
///
/// 1. Directional lights always pass, in scene order.
/// 2. A local light is culled when its cull sphere ([`cull_radius`]: range, or
///    energy radius at `exposure`) cannot touch the view volume — an exact-safe
///    cull, since every visible fragment is inside the volume.
/// 3. Past [`LIGHTS_PER_FRAME_CEILING`], the local lights with the largest
///    on-screen footprint win (ties by scene index), and the winners are then
///    re-emitted in scene order.
pub fn plan_lights(
    scene: &RenderScene,
    view: &RenderView,
    vsm_slots: &[u32],
    settings: &LightSettings,
    exposure: f32,
) -> LightPlan {
    let viewer = Viewer::of(view);
    let origin = view.origin.origin();
    let mut plan = LightPlan {
        fallback_sun: scene.lights.is_empty(),
        ..LightPlan::default()
    };
    let mut locals: Vec<(u32, f32, f64)> = Vec::new();
    for (i, l) in scene.lights.iter().enumerate() {
        if l.kind == LightKind::Directional {
            continue;
        }
        if !settings.local_lights {
            plan.skipped_local += 1;
            continue;
        }
        let r = cull_radius(l, exposure);
        if settings.cull {
            if r <= 0.0 {
                plan.culled_energy += 1;
                continue;
            }
            // PAR1a: a fixture's own draw distance, measured to the nearest
            // point of its sphere.
            if let Some(b) = bound_of(&scene.light_bounds, i as u32) {
                if b.draw_m > 0.0 && (l.position - viewer.eye).length() - r as f64 > b.draw_m as f64
                {
                    plan.culled_distance += 1;
                    continue;
                }
            }
            if !viewer.sees(l.position, r as f64) {
                if l.range <= 0.0 || r < l.range {
                    // Would the RANGE sphere have been seen? Then it was the
                    // energy bound that culled it.
                    if viewer.sees(l.position, l.range.max(0.0) as f64) || l.range <= 0.0 {
                        plan.culled_energy += 1;
                        continue;
                    }
                }
                plan.culled_frustum += 1;
                continue;
            }
        }
        let r = if settings.cull {
            r
        } else if l.range > 0.0 {
            l.range
        } else {
            r
        };
        locals.push((i as u32, r, viewer.footprint_px(l.position, r as f64)));
    }
    let dirs: Vec<u32> = scene
        .lights
        .iter()
        .enumerate()
        .filter(|(_, l)| l.kind == LightKind::Directional)
        .map(|(i, _)| i as u32)
        .collect();
    let room = LIGHTS_PER_FRAME_CEILING.saturating_sub(dirs.len());
    if locals.len() > room {
        let mut ranked = locals.clone();
        ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        ranked.truncate(room);
        ranked.sort_by_key(|e| e.0);
        plan.culled_ceiling = (locals.len() - room) as u32;
        locals = ranked;
    }
    for &i in dirs.iter().take(LIGHTS_PER_FRAME_CEILING) {
        let l = &scene.lights[i as usize];
        let d = l.direction.normalize_or_zero();
        plan.records.push(GpuLight {
            color: [l.color[0], l.color[1], l.color[2], l.intensity],
            pos_dir: [d.x, d.y, d.z, 0.0],
            params: [0.0, 0.0, 0.0, slot(vsm_slots, i)],
            spot_dir: [0.0; 4],
        });
        plan.scene_index.push(i);
    }
    plan.directional = plan.records.len() as u32;
    for (i, r, _) in locals {
        let l = &scene.lights[i as usize];
        let p = (l.position - origin).as_vec3();
        // PAR1b.2 THE DRAW-DISTANCE FADE: see [`LIGHT_FADE_M`].
        let fade = if settings.cull {
            bound_of(&scene.light_bounds, i).map_or(1.0, |b| {
                draw_fade(b.draw_m, (l.position - viewer.eye).length() - r as f64)
            })
        } else {
            1.0
        };
        let mut g = GpuLight {
            color: [l.color[0], l.color[1], l.color[2], l.intensity * fade],
            pos_dir: [p.x, p.y, p.z, 1.0],
            params: [l.range, 0.0, 0.0, slot(vsm_slots, i)],
            spot_dir: [0.0, 0.0, 0.0, r],
        };
        if l.kind == LightKind::Spot {
            let emit = (-l.direction).normalize_or_zero();
            g.pos_dir[3] = 2.0;
            g.params[1] = l.inner_cos;
            g.params[2] = l.outer_cos;
            g.spot_dir = [emit.x, emit.y, emit.z, r];
        } else if let Some(clip) = bound_of(&scene.light_bounds, i).and_then(|b| b.clip) {
            encode_clip(&mut g, l.position, &clip);
            plan.clipped += 1;
        }
        plan.records.push(g);
        plan.scene_index.push(i);
    }
    plan
}

/// **Over how many metres a local light fades out before its draw distance**
/// (wave PAR1b.2). [`plan_lights`] drops a light whose sphere's nearest point
/// is past [`LightBound::draw_m`]; before this the light went from its whole
/// intensity to nothing across that one metre, and a street lamp's pool popped
/// onto the asphalt at ~126 m (`draw_m` 110 + its 16 m sphere — the PAR1b
/// audit). Now its intensity ramps linearly to zero over the last
/// `LIGHT_FADE_M` before the cut, so the record that is dropped was already
/// dark. Applied HERE, in the one plan every lit pass reads its local lights
/// from (mesh, terrain, scatter, water, GI inject), so every bounded light —
/// PAR1a's room fixtures, PAR1b's lamps and signal heads, PAR1c's headlamps —
/// inherits it; a multiply on the record, not a new light. A light with no
/// bound (`draw_m` 0) is unchanged.
pub const LIGHT_FADE_M: f32 = 24.0;

/// The fade multiplier for a light whose sphere's nearest point is `near_m`
/// from the eye against its draw distance `draw_m` — `1` inside
/// `draw_m - LIGHT_FADE_M`, `0` at `draw_m`, linear between; `1` with no draw
/// distance.
pub fn draw_fade(draw_m: f32, near_m: f64) -> f32 {
    if draw_m <= 0.0 || !near_m.is_finite() {
        return 1.0;
    }
    ((f64::from(draw_m) - near_m) / f64::from(LIGHT_FADE_M)).clamp(0.0, 1.0) as f32
}

/// The bound a scene light carries, if any — `bounds` is in light order, so a
/// binary search.
fn bound_of(bounds: &[LightBound], i: u32) -> Option<&LightBound> {
    bounds
        .binary_search_by_key(&i, |b| b.light)
        .ok()
        .map(|k| &bounds[k])
}

/// IEEE half-precision bits of `x` (truncating; finite range clamped) — the
/// packing `unpack2x16float` reads back in `lights.wgsl`.
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x007f_ffff;
    if exp >= 31 {
        return sign | 0x7bff;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        return sign | ((man | 0x0080_0000) >> (14 - exp)) as u16;
    }
    let mut h = u32::from(sign) | ((exp as u32) << 10) | (man >> 13);
    if man & 0x1000 != 0 {
        h += 1;
    }
    h as u16
}

/// Two values as one `pack2x16float` word, stored in an `f32` lane by bits.
fn pack2(lo: f32, hi: f32) -> f32 {
    f32::from_bits(u32::from(f16_bits(lo)) | (u32::from(f16_bits(hi)) << 16))
}

/// **A CLIPPED POINT LIGHT** (wave PAR1a): kind `3`, its box in the lanes a
/// point light leaves free — `spot_dir.xyz` hold three `pack2x16float` words
/// (the box centre's offset from the light along `u` and `v`, the box's half
/// extents along `u` and `v`, and `u` itself as `(x, z)`), `params.yz` the box's
/// floor and ceiling relative to the light. `spot_dir.w` keeps the cull radius.
fn encode_clip(g: &mut GpuLight, light: DVec3, clip: &crate::scene::LightClip) {
    let d = (clip.center - light).as_vec3();
    let (ux, uz) = (clip.u[0], clip.u[1]);
    // v = (-u.z, u.x): the lot frame's own convention (`LotFrame::v`).
    let off_u = d.x * ux + d.z * uz;
    let off_v = -d.x * uz + d.z * ux;
    // 3: a room's box (terrain and water skip it); 4: an exterior box.
    g.pos_dir[3] = if clip.interior { 3.0 } else { 4.0 };
    g.params[1] = d.y - clip.half.y;
    g.params[2] = d.y + clip.half.y;
    g.spot_dir[0] = pack2(off_u, off_v);
    g.spot_dir[1] = pack2(clip.half.x, clip.half.z);
    g.spot_dir[2] = pack2(ux, uz);
}

/// The CPU mirror of `lights.wgsl`'s `light_clip_holds`: whether world point
/// `p` is inside a clipped light's box (for arms that ask without a GPU).
pub fn clip_holds(clip: &crate::scene::LightClip, p: DVec3) -> bool {
    let d = (p - clip.center).as_vec3();
    let (ux, uz) = (clip.u[0], clip.u[1]);
    let du = d.x * ux + d.z * uz;
    let dv = -d.x * uz + d.z * ux;
    du.abs() <= clip.half.x && dv.abs() <= clip.half.z && d.y.abs() <= clip.half.y
}

fn slot(vsm_slots: &[u32], i: u32) -> f32 {
    vsm_slots.get(i as usize).copied().unwrap_or(0) as f32
}

/// **The local-light shadow policy** (wave PAR0 clause 2): per scene light,
/// whether it may have a virtual-shadow page tree this frame.
///
/// Directional lights keep their authored `cast_shadows`. Among the local
/// lights that ASKED (`cast_shadows`) and whose cull sphere comes within
/// [`SHADOW_POLICY_REACH_M`] of the camera, the `budget` brightest at the
/// camera win — brightness = `I · max(rgb) / max(d², 1)` from the camera
/// SNAPPED to [`SHADOW_POLICY_CELL_M`], ties by scene index. A pure function
/// of the scene and the snapped camera POSITION, so the shadowed set changes
/// only when the camera crosses a lattice cell, and never when it turns.
///
/// **Not the view frustum, measured**: the first cut admitted only lights
/// inside the frustum, and on the island's 1 670-light strip with the camera
/// turning 3° a frame the set changed nearly every frame — each change
/// rebuilds the virtual-shadow address space and re-rasters its pages — so
/// two shadowed lights cost MORE than eight (vsm segments 64.3 ms vs 39.5 ms,
/// GPU 92.3 vs 68.3 ms, the PAR0 ledger's run 2). A shadowed light behind the
/// camera costs little: pages are marked from VISIBLE receivers.
pub fn shadow_policy(
    scene: &RenderScene,
    view: &RenderView,
    budget: usize,
    exposure: f32,
) -> Vec<bool> {
    let snap = |v: f64| (v / SHADOW_POLICY_CELL_M).round() * SHADOW_POLICY_CELL_M;
    let cam = DVec3::new(
        snap(view.eye_world.x),
        snap(view.eye_world.y),
        snap(view.eye_world.z),
    );
    let mut out: Vec<bool> = scene
        .lights
        .iter()
        .map(|l| l.kind == LightKind::Directional && l.cast_shadows)
        .collect();
    let mut cands: Vec<(u32, f64)> = scene
        .lights
        .iter()
        .enumerate()
        .filter(|(_, l)| l.kind != LightKind::Directional && l.cast_shadows)
        .filter(|(_, l)| {
            let r = cull_radius(l, exposure);
            r > 0.0 && (l.position - cam).length() - r as f64 <= SHADOW_POLICY_REACH_M
        })
        .map(|(i, l)| {
            let peak =
                l.intensity.max(0.0) as f64 * l.color.iter().fold(0.0f32, |a, &c| a.max(c)) as f64;
            (
                i as u32,
                peak / (l.position - cam).length_squared().max(1.0),
            )
        })
        .collect();
    cands.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    for (i, _) in cands.into_iter().take(budget) {
        out[i as usize] = true;
    }
    out
}

/// **The froxel a fragment falls in** — the CPU mirror of `lights.wgsl`'s
/// `light_cluster_of`, for a probe that picks a pixel and asks the read-back
/// grid what its froxel lists. `pixel` is in the view's physical pixels,
/// `world` the fragment's world position.
pub fn froxel_of(view: &RenderView, pixel: (f32, f32), world: DVec3) -> u32 {
    let tx = ((pixel.0.max(0.0) / view.width.max(1) as f32 * CLUSTER_X as f32) as u32)
        .min(CLUSTER_X - 1);
    let ty = ((pixel.1.max(0.0) / view.height.max(1) as f32 * CLUSTER_Y as f32) as u32)
        .min(CLUSTER_Y - 1);
    let fwd = view.forward.normalize_or(Vec3::NEG_Z);
    let p = (world - view.origin.origin()).as_vec3();
    let depth = (p - view.eye_local()).dot(fwd);
    let tz = if depth < CLUSTER_NEAR_M {
        0
    } else if depth >= CLUSTER_FAR_M {
        CLUSTER_Z - 1
    } else {
        let scale = (CLUSTER_Z - 2) as f32 / (CLUSTER_FAR_M / CLUSTER_NEAR_M).log2();
        (1 + ((depth / CLUSTER_NEAR_M).log2() * scale).floor().max(0.0) as u32)
            .clamp(1, CLUSTER_Z - 2)
    };
    (tz * CLUSTER_Y + ty) * CLUSTER_X + tx
}

/// What the froxel build actually produced, read back from the GPU buffer —
/// the number a gate may call "lights the shader saw".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LightCensus {
    /// Records the header told the shader about.
    pub records: u32,
    /// Directional records among them.
    pub directional: u32,
    /// Distinct local lights listed by at least one froxel.
    pub distinct_local: u32,
    /// Froxels listing at least one light.
    pub nonempty_froxels: u32,
    /// The longest froxel list (before the capacity clamp).
    pub max_per_froxel: u32,
    /// Froxels whose list overflowed [`CLUSTER_MAX_LIGHTS`].
    pub overflowed_froxels: u32,
    /// Sum of every froxel's (clamped) list length.
    pub entries: u64,
    /// **What a TILED grid would have listed** (the mini-scout's comparison):
    /// for each screen tile, the distinct lights across all its depth slices —
    /// exactly the set a tile frustum with no depth bounds intersects, since
    /// the slices' boxes tile it. Summed over tiles; compare with `entries`.
    pub tiled_entries: u64,
    /// The longest of those tile lists — what the worst fragment of a tiled
    /// grid would walk (compare with `max_per_froxel`).
    pub tiled_max: u32,
    /// Per froxel, the stored count (unclamped).
    pub counts: Vec<u32>,
}

impl LightCensus {
    /// Every light the shader could shade this frame: directional + distinct local.
    pub fn lights_in_shader(&self) -> u32 {
        self.directional + self.distinct_local
    }
}

/// **The GPU half of the substrate**: the header + the storage buffer every
/// lit pass binds, the froxel-build compute, and the census readback.
pub struct LightGrid {
    header: wgpu::Buffer,
    data: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    compute_bg: wgpu::BindGroup,
    bare_bgl: wgpu::BindGroupLayout,
    bare_bg: wgpu::BindGroup,
    last: LightHeader,
    plan: LightPlan,
    frames: u64,
}

/// The two layout entries every consumer of the light list declares, at
/// `hdr` (the header uniform) and `data` (the read-only storage words).
pub fn light_bgl_entries(hdr: u32, data: u32) -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: hdr,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: data,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
}

impl LightGrid {
    /// Build the buffers + the froxel-build compute.
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;
        let header = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights-header"),
            size: std::mem::size_of::<LightHeader>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights-data"),
            // PAR0b: the light words AND the GI probe records (see
            // `GI_PROBE_OFFSET_BYTES`).
            size: LIGHT_BUFFER_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let compute_bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light-cluster-lights"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light-cluster-lights"),
            layout: &compute_bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.as_entire_binding(),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("light-cluster"),
            source: wgpu::ShaderSource::Wgsl(crate::passes::light_cluster_source().into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light-cluster"),
            bind_group_layouts: &[Some(&compute_bgl1)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("light-cluster"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cs_cluster"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bare_entries = light_bgl_entries(0, 1);
        let bare_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lights-bare"),
            entries: &bare_entries,
        });
        let bare_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lights-bare"),
            layout: &bare_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: header.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.as_entire_binding(),
                },
            ],
        });
        Self {
            header,
            data,
            pipeline,
            compute_bg,
            bare_bgl,
            bare_bg,
            last: LightHeader::zeroed(),
            plan: LightPlan::default(),
            frames: 0,
        }
    }

    /// Stage this frame's plan + header on the queue.
    pub fn upload(&mut self, gpu: &GpuContext, plan: LightPlan, view: &RenderView, debug: bool) {
        let n = plan.records.len().min(LIGHTS_PER_FRAME_CEILING);
        let fwd = view.forward.normalize_or(Vec3::NEG_Z);
        let right = fwd.cross(view.up).normalize_or(Vec3::X);
        let up = right.cross(fwd);
        let eye = view.eye_local();
        let scale = (CLUSTER_Z - 2) as f32 / (CLUSTER_FAR_M / CLUSTER_NEAR_M).log2();
        let aspect = view.width.max(1) as f32 / view.height.max(1) as f32;
        let (half, ortho) = match view.ortho {
            Some(o) => ((o.half_height * aspect, o.half_height), 1.0),
            None => {
                let t = (view.fov_y * 0.5).tan();
                ((t * aspect, t), 0.0)
            }
        };
        // The last slice ends where the deepest listed light's sphere does.
        let deepest = plan.records[plan.directional as usize..n.max(plan.directional as usize)]
            .iter()
            .map(|g| {
                let p = Vec3::new(g.pos_dir[0], g.pos_dir[1], g.pos_dir[2]);
                (p - eye).dot(fwd) + g.spot_dir[3]
            })
            .fold(CLUSTER_FAR_M, f32::max);
        let header = LightHeader {
            counts: [
                n as u32,
                plan.directional.min(n as u32),
                u32::from(plan.fallback_sun),
                CLUSTER_MAX_LIGHTS,
            ],
            grid: [CLUSTER_X, CLUSTER_Y, CLUSTER_Z, CLUSTER_COUNT],
            bases: [COUNT_BASE_VEC4, INDEX_BASE_VEC4, u32::from(debug), 0],
            zparams: [CLUSTER_NEAR_M, CLUSTER_FAR_M, scale, CLUSTER_DEPTH_PAD],
            fwd: [fwd.x, fwd.y, fwd.z, 0.0],
            eye: [eye.x, eye.y, eye.z, 0.0],
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            proj: [half.0, half.1, ortho, deepest],
        };
        gpu.queue
            .write_buffer(&self.header, 0, bytemuck::bytes_of(&header));
        if n > 0 {
            gpu.queue
                .write_buffer(&self.data, 0, bytemuck::cast_slice(&plan.records[..n]));
        }
        self.last = header;
        self.plan = plan;
    }

    /// Record the froxel build: one compute pass, one dispatch.
    pub fn record_cluster(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("light-cluster"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.compute_bg, &[]);
        pass.dispatch_workgroups(CLUSTER_COUNT.div_ceil(64), 1, 1);
        self.frames += 1;
    }

    /// The header uniform (for the environment group's binding).
    pub fn header_buffer(&self) -> &wgpu::Buffer {
        &self.header
    }

    /// The storage words (for the environment group's binding).
    pub fn data_buffer(&self) -> &wgpu::Buffer {
        &self.data
    }

    /// The layout of the standalone group the voxel + water passes bind.
    pub fn bare_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.bare_bgl
    }

    /// The standalone group over the same two buffers.
    pub fn bare_bg(&self) -> &wgpu::BindGroup {
        &self.bare_bg
    }

    /// The last uploaded plan.
    pub fn plan(&self) -> &LightPlan {
        &self.plan
    }

    /// The last uploaded header.
    pub fn header(&self) -> LightHeader {
        self.last
    }

    /// Frames that recorded the froxel build — the engagement counter.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// **Read back what the froxel build wrote** — blocking; a gate's and the
    /// instrument's door, never the shipping frame path. Reads the GPU buffer
    /// after the last submitted frame.
    pub fn read_census(&self, gpu: &GpuContext) -> Result<LightCensus, String> {
        let words = self.read_words(gpu)?;
        let count_base = COUNT_BASE_VEC4 as usize * 4;
        let index_base = INDEX_BASE_VEC4 as usize * 4;
        let mut census = LightCensus {
            records: self.last.counts[0],
            directional: self.last.counts[1],
            counts: Vec::with_capacity(CLUSTER_COUNT as usize),
            ..LightCensus::default()
        };
        let mut seen = std::collections::BTreeSet::new();
        for c in 0..CLUSTER_COUNT as usize {
            let n = words[count_base + c];
            census.counts.push(n);
            if n > 0 {
                census.nonempty_froxels += 1;
            }
            census.max_per_froxel = census.max_per_froxel.max(n);
            if n > CLUSTER_MAX_LIGHTS {
                census.overflowed_froxels += 1;
            }
            let len = n.min(CLUSTER_MAX_LIGHTS) as usize;
            census.entries += len as u64;
            let at = index_base + c * CLUSTER_MAX_LIGHTS as usize;
            for &i in &words[at..at + len] {
                seen.insert(i);
            }
        }
        census.distinct_local = seen.len() as u32;
        for t in 0..(CLUSTER_X * CLUSTER_Y) as usize {
            let mut tile = std::collections::BTreeSet::new();
            for z in 0..CLUSTER_Z as usize {
                let c = z * (CLUSTER_X * CLUSTER_Y) as usize + t;
                let len = census.counts[c].min(CLUSTER_MAX_LIGHTS) as usize;
                let at = index_base + c * CLUSTER_MAX_LIGHTS as usize;
                tile.extend(words[at..at + len].iter().copied());
            }
            census.tiled_entries += tile.len() as u64;
            census.tiled_max = census.tiled_max.max(tile.len() as u32);
        }
        Ok(census)
    }

    /// The froxel index list of froxel `c`, read back (a gate's door).
    pub fn read_froxel(&self, gpu: &GpuContext, c: u32) -> Result<Vec<u32>, String> {
        let words = self.read_words(gpu)?;
        let n = words[COUNT_BASE_VEC4 as usize * 4 + c as usize].min(CLUSTER_MAX_LIGHTS) as usize;
        let at = INDEX_BASE_VEC4 as usize * 4 + c as usize * CLUSTER_MAX_LIGHTS as usize;
        Ok(words[at..at + n].to_vec())
    }

    fn read_words(&self, gpu: &GpuContext) -> Result<Vec<u32>, String> {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights-census"),
            size: LIGHT_DATA_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lights-census"),
            });
        encoder.copy_buffer_to_buffer(&self.data, 0, &staging, 0, LIGHT_DATA_BYTES);
        gpu.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("poll: {e}"))?;
        rx.recv()
            .map_err(|e| format!("map_async dropped: {e}"))?
            .map_err(|e| format!("map_async: {e}"))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| format!("get_mapped_range: {e}"))?;
        let words: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(words)
    }
}

#[cfg(test)]
mod fade_tests {
    use super::*;

    /// **A bounded light fades to nothing at its draw distance** (wave
    /// PAR1b.2): the multiplier is 1 inside `draw - LIGHT_FADE_M`, 0 at the
    /// draw distance, linear between; a light with no draw distance is
    /// untouched. Mutation: `LIGHT_FADE_M` 0.001 (the old hard cut) -> the
    /// midpoint reads 0, red.
    #[test]
    fn a_bounded_light_fades_out_over_the_last_metres_of_its_draw_distance() {
        assert_eq!(draw_fade(110.0, 50.0), 1.0);
        assert_eq!(draw_fade(110.0, 110.0), 0.0);
        assert_eq!(draw_fade(110.0, 130.0), 0.0);
        let mid = draw_fade(110.0, 110.0 - f64::from(LIGHT_FADE_M) * 0.5);
        assert!((mid - 0.5).abs() < 1e-6, "midpoint {mid}");
        assert_eq!(draw_fade(0.0, 5_000.0), 1.0);
    }
}
