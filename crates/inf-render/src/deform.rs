//! **Drawing surface deformation** (P22.1): the camera-following window through
//! which the shaders see the sim's footprint field.
//!
//! The field itself is CPU-side and sim-authoritative (`inf_terrain::deform`):
//! a sparse map of how far the ground has been pressed down, stepped at the
//! fixed step, byte-identical on every machine. This module is the *view* of it
//! — one fixed-size R32Float texture the terrain vertex/fragment stages and the
//! scatter bend read, plus the uniform that says where in the world it is.
//!
//! # The window follows the camera, and that is legal
//!
//! P21's law is that a **camera-paged store may be read but never shipped or
//! simulated**. This window is exactly the permitted half: it selects *what is
//! drawn* out of state the sim already decided, and nothing downstream of it can
//! feed back into a step. Move the camera and no footprint changes, appears or
//! disappears — the same footprints are simply carried by different texels.
//! [`window_origin_texels`] is the mechanism: the origin is an **integer lattice
//! index**, so panning translates the window by whole texels and a texel always
//! carries the same field sample. Nothing is resampled, so a footprint cannot
//! shimmer as you walk past it. (The sun-bucket and radius-bucket quantizations
//! are the same trick applied to other continuously-varying inputs.)
//!
//! # One window, sized once
//!
//! [`DEFORM_WINDOW_TEXELS`]² at [`DEFORM_TEXEL_M`] per texel = a 128 m square
//! centred on the camera. 128 m is comfortably past the distance at which a
//! 25 cm-wide dent in the ground is a visible shape rather than a shading tint,
//! and 512² × 4 B = 1 MiB is a rounding error against the shadow atlas.
//!
//! **The size is a constant, not a setting**, and that is load-bearing: it is why
//! [`DeformResources`] is created once in `EngineRenderer::new`, never
//! recreated, and therefore does **not** add a fourth component to
//! `passes::ResourceKey`. The rule is quoted verbatim from
//! [`crate::wetness::WetnessResources`]: *"The key exists for resources that get
//! **recreated** — a resize, a quality clamp — because a bind group holding the
//! old view keeps sampling it, silently. A buffer allocated in
//! `EngineRenderer::new` and only ever `write_buffer`-ed cannot go stale."* A
//! resizable deformation window would need a generation and a key component; it
//! was deliberately not built.
//!
//! # Precision: why R32Float
//!
//! Max depth is 1 m (the terrain skirt floor — see
//! `inf_terrain::deform::MAX_DEFORM_DEPTH_M`), and what the eye actually reads
//! is not the depth but the **normal**, which the terrain fragment recomputes by
//! central difference over a 0.25 m step.
//!
//! * **R8Unorm scaled by 1 m** — a 3.9 mm quantum. Over a 0.25 m difference that
//!   is a 1.6 % slope step, and a central-difference normal turns it into
//!   visible terracing across the flat interior of every print. Rejected.
//! * **R16Float** — 0.24 mm at 0.4 m, which is plenty, and half the bytes. It
//!   also needs a half-float encoder on the pack path, and the thing being
//!   halved is a *once-allocated* megabyte whose per-frame traffic is a sparse
//!   dirty rect rather than the whole surface. Rejected as the wrong trade.
//! * **R32Float** — carries the field's own `f32` exactly, so the pack is a copy
//!   with no encode step to get wrong, and matches the pass's existing
//!   unfilterable-float doctrine: sampled with `textureLoad` + manual bilinear
//!   like the height texture beside it, so no `FLOAT32_FILTERABLE` feature and
//!   no sampler is needed anywhere and headless CI adapters stay happy. Chosen.
//!
//! # Uploads are gated, and the gate is countable
//!
//! [`DeformResources::update`] uploads **nothing** when neither the field nor the
//! window origin has moved, and counts every upload it does perform. That is an
//! engagement counter in the P20.3 sense — the claim "an idle frame costs
//! nothing" is a statement about the command stream, and a pixel comparison
//! cannot make it.

use bytemuck::{Pod, Zeroable};
use inf_math::FloatingOrigin;
use std::sync::atomic::{AtomicU64, Ordering};

/// World metres per window texel.
///
/// **Exactly** `inf_terrain::deform::DEFORM_SAMPLE_PITCH_M`. Kept local — the
/// renderer never names `inf-terrain`, the same discipline `DEFAULT_WEIGHT` and
/// `UNASSIGNED_BIOME_COLOR` keep in `passes::terrain` — and pinned by
/// `the_window_geometry_matches_the_field` in the host projector tests. Equal
/// pitch is what makes the pack an index map: texel `k` *is* field sample `k`,
/// with no resampling filter anywhere in the path.
pub const DEFORM_TEXEL_M: f64 = 0.25;

/// Window edge, texels. 512 × 0.25 m = a 128 m square. See the module docs.
pub const DEFORM_WINDOW_TEXELS: u32 = 512;

/// Window edge, metres.
pub const DEFORM_WINDOW_M: f64 = DEFORM_WINDOW_TEXELS as f64 * DEFORM_TEXEL_M;

/// The deepest value the window can carry, metres — the mirror of
/// `inf_terrain::deform::MAX_DEFORM_DEPTH_M`, which is itself floored by the
/// terrain patch skirt (`passes::terrain`'s `.max(1.0)`). Clamped again here so
/// a projection carrying a deeper value cannot pull a patch boundary below its
/// crack-sealing skirt.
pub const DEFORM_MAX_DEPTH_M: f32 = 1.0;

/// One cell of the projected field: a dense `cell_samples²` block of depths at
/// a cell coordinate on the global lattice.
///
/// The projection ships **cells, not a window**, so the host does no camera work
/// and the renderer does no field work. A walking character keeps tens of cells
/// live, so this is kilobytes; the field's own bound
/// (`inf_terrain::deform::MAX_DEFORM_CELLS`) caps it absolutely.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderDeformCell {
    /// Cell coordinate on the global lattice.
    pub coord: (i32, i32),
    /// `cell_samples²` depths in metres, row-major, `≥ 0` = pressed down.
    pub depths: Vec<f32>,
}

/// The projected deformation field.
///
/// Carried behind an `Arc` on [`crate::RenderScene`] so a host can hand the same
/// projection to consecutive frames unchanged — the field only moves when
/// something walked, and rebuilding this every frame for a standing character
/// would be the copy the sparse representation exists to avoid.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderDeform {
    /// Samples along one cell edge (`inf_terrain::deform::DEFORM_CELL_SAMPLES`).
    pub cell_samples: u32,
    /// Metres between samples — must equal [`DEFORM_TEXEL_M`].
    pub texel_m: f64,
    /// The field's change stamp. Two projections with the same epoch describe the
    /// same field; the upload gate is a compare on this.
    pub epoch: u64,
    /// Live cells, in the field's own (`BTreeMap`) order.
    pub cells: Vec<RenderDeformCell>,
}

impl RenderDeform {
    /// Whether this projection can move a pixel. An empty cell list is the
    /// off-path case and every consumer short-circuits on it.
    pub fn drawable(&self) -> bool {
        !self.cells.is_empty() && self.cell_samples > 0
    }
}

/// The lattice index of the minimum corner of the window centred on world
/// `(x, z)`.
///
/// The **snap**. Mirrors `inf_terrain::deform::window_origin` exactly — same
/// rounding, same half-width — because the two must agree about which sample a
/// texel carries. Pinned by `the_snap_matches_the_field`.
pub fn window_origin_texels(center_x: f64, center_z: f64) -> (i64, i64) {
    let cx = (center_x / DEFORM_TEXEL_M).round() as i64;
    let cz = (center_z / DEFORM_TEXEL_M).round() as i64;
    let half = (DEFORM_WINDOW_TEXELS / 2) as i64;
    (cx - half, cz - half)
}

/// Pack the projected cells into a `DEFORM_WINDOW_TEXELS²` row-major depth
/// buffer whose minimum corner is lattice index `origin`.
///
/// Pure: no camera (the caller already snapped one into `origin`), no clock, no
/// frame index. `out` is resized and fully written, so the same arguments always
/// leave the same bytes whatever the buffer held. Only cells intersecting the
/// window are walked; depths are clamped to [`DEFORM_MAX_DEPTH_M`] here so the
/// skirt coupling holds even against a malformed projection.
///
/// Returns the **dirty rect** `(x0, y0, w, h)` of texels the cells actually
/// wrote, or `None` when nothing intersected — which is what lets the upload be
/// a sparse region rather than a megabyte.
pub fn pack_deform_window(
    deform: &RenderDeform,
    origin: (i64, i64),
    out: &mut Vec<f32>,
) -> Option<(u32, u32, u32, u32)> {
    let n = DEFORM_WINDOW_TEXELS as usize;
    out.clear();
    out.resize(n * n, 0.0);
    if !deform.drawable() {
        return None;
    }
    let cs = deform.cell_samples as i64;
    let expect = (cs * cs) as usize;
    let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for cell in &deform.cells {
        // A malformed block is skipped whole rather than written partially — the
        // defensive rule the tile weights and biome ids take, for the same
        // reason: half a block is deformation nobody pressed.
        if cell.depths.len() != expect {
            continue;
        }
        let base = (
            cell.coord.0 as i64 * cs - origin.0,
            cell.coord.1 as i64 * cs - origin.1,
        );
        for lj in 0..cs {
            let ty = base.1 + lj;
            if ty < 0 || ty >= n as i64 {
                continue;
            }
            for li in 0..cs {
                let tx = base.0 + li;
                if tx < 0 || tx >= n as i64 {
                    continue;
                }
                let d = cell.depths[(lj * cs + li) as usize].clamp(0.0, DEFORM_MAX_DEPTH_M);
                out[ty as usize * n + tx as usize] = d;
                x0 = x0.min(tx);
                y0 = y0.min(ty);
                x1 = x1.max(tx);
                y1 = y1.max(ty);
            }
        }
    }
    (x1 >= x0).then(|| {
        (
            x0 as u32,
            y0 as u32,
            (x1 - x0 + 1) as u32,
            (y1 - y0 + 1) as u32,
        )
    })
}

/// The deformation uniform. Mirrors `struct Deform` in `deform.wgsl`; the pair is
/// pinned field-for-field by `the_uniform_matches_the_shader_struct`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct DeformUniform {
    /// x = render-local window min X, y = render-local window min Z,
    /// z = 1 / texel metres, w = window texels.
    pub window: [f32; 4],
    /// x = window enabled (1/0), y = max depth (m), z = foliage bend gain,
    /// w = foliage wind enabled (1/0).
    pub params: [f32; 4],
    /// x/y = wind direction XZ (unit, or zero), z = wavenumber (rad/m),
    /// w = **the whole phase offset, already reduced** (rad).
    ///
    /// The reduction is the point. A shader computing `speed · t · k` in `f32`
    /// with `t` = `ResolvedSky::cloud_time_s` (up to 86 400 s) loses every bit
    /// of the fractional phase long before the day is out, and the sway freezes
    /// — the same class of failure `inf_ecs::sky::MAX_WEATHER_BLEND_S` documents
    /// one crate over. And the world→render-local offset has to be folded in
    /// here too, in `f64`, or the phase would **re-roll on every floating-origin
    /// rebase**: the grass would flick to a new position of its sway cycle
    /// because the camera crossed a 10 m boundary.
    pub wind: [f32; 4],
}

impl Default for DeformUniform {
    fn default() -> Self {
        Self {
            window: [
                0.0,
                0.0,
                (1.0 / DEFORM_TEXEL_M) as f32,
                DEFORM_WINDOW_TEXELS as f32,
            ],
            params: [0.0, DEFORM_MAX_DEPTH_M, DEFORM_BEND_GAIN, 0.0],
            wind: [0.0; 4],
        }
    }
}

impl DeformUniform {
    /// `false` ⇒ every shader branch that reads the window is present-but-false
    /// and the arithmetic around it runs instruction-for-instruction unchanged,
    /// so every pre-P22.1 golden stays byte-stable.
    pub fn enabled(&self) -> bool {
        self.params[0] > 0.5
    }

    /// `false` ⇒ foliage does not sway. **Independent of
    /// [`enabled`](Self::enabled)**, and driven by
    /// [`crate::ScatterSettings::foliage_wind`] rather than by whether anything
    /// happens to have walked on the ground — the first cut gated sway on "is
    /// there a live deform cell anywhere", which made an ambient effect flick on
    /// and off with a footprint expiring half a kilometre away.
    pub fn wind_enabled(&self) -> bool {
        self.params[3] > 0.5
    }
}

/// The wind phase offset for a frame, **reduced in `f64` to `[0, 2π)`**.
///
/// `origin_xz` is the floating origin's XZ (so the shader's render-local `dot`
/// plus this equals the world-space `dot`, making the phase origin-invariant),
/// `wind_xz` the unit wind direction, `speed` m/s, `clock_s` the level clock.
///
/// Pure, `f64` throughout, and unit-tested for both properties — origin
/// invariance and non-freezing at a day-long clock.
pub fn wind_phase(
    origin_xz: (f64, f64),
    wind_xz: (f64, f64),
    speed: f64,
    clock_s: f64,
    wavenumber: f64,
) -> f32 {
    let tau = std::f64::consts::TAU;
    let raw = (origin_xz.0 * wind_xz.0 + origin_xz.1 * wind_xz.1) * wavenumber
        - speed * clock_s * wavenumber;
    (raw.rem_euclid(tau)) as f32
}

/// How far a fully-pressed sample bends the foliage standing in it, as a
/// fraction of the instance's own height.
///
/// Grass answers a boot by lying down, not by denting — 0.85 takes a blade in a
/// full-depth print almost flat while leaving it recognisably a blade. Scaled
/// per layer by `inf_terrain::deform::LayerResponse::bend` on the sim side and by
/// the sampled depth here, so grass over rock (bend 0) never moves.
pub const DEFORM_BEND_GAIN: f32 = 0.85;

/// Texels over which the window's effect fades to nothing at its edge.
///
/// **Not polish — a discontinuity fix.** The window is 128 m across, so its rim
/// sits 64 m from the camera, which is *inside* the scatter's 120 m full-mesh
/// band and far inside any terrain draw distance. Without a fade, a rut would
/// stop dead at a straight line 64 m out: the terrain's shading would step, and
/// a row of grass would snap upright mid-field. 32 texels is 8 m — long enough
/// that the ramp is below the noise of the ground it is fading across, short
/// enough that it costs almost none of the window.
pub const DEFORM_RIM_TEXELS: f32 = 32.0;

/// The window's edge fade at a texel coordinate, `[0, 1]`. Shared by the shader
/// (`deform.wgsl`) and its CPU mirror; unit-tested here so the pair cannot drift.
pub fn rim_fade(px: f32, pz: f32, texels: f32) -> f32 {
    let edge = (px.min(pz)).min((texels - 1.0 - px).min(texels - 1.0 - pz));
    (edge / DEFORM_RIM_TEXELS).clamp(0.0, 1.0)
}

/// Peak wind sway as a fraction of an instance's height.
///
/// Small on purpose: this is the ambient shimmer that stops a field of grass
/// reading as plastic, not a storm. The storm is the weather system's job.
pub const WIND_SWAY: f32 = 0.12;

/// Wind sway wavelength, metres — how far apart two blades are before they are
/// out of phase. 4 m gives the travelling-gust look at grass scale.
pub const WIND_WAVELENGTH_M: f32 = 4.0;

/// The GPU-side home of the window: one texture and one uniform, created once
/// and never resized.
///
/// **That is load-bearing** — see the module docs for the `ResourceKey`
/// exemption this inherits verbatim from [`crate::wetness::WetnessResources`].
pub struct DeformResources {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub uniform: wgpu::Buffer,
    /// CPU mirror of the texture, so an upload can be skipped when nothing
    /// changed and clipped to a dirty rect when something did.
    shadow: Vec<f32>,
    /// The lattice origin `shadow` was packed at, and the field epoch it was
    /// packed from. The upload gate is a compare on this pair.
    packed: Option<((i64, i64), u64)>,
    /// The dirty rect of the **previous** pack — the texels that currently hold
    /// non-zero depth.
    ///
    /// Load-bearing, and the first cut did not have it: a rect covers what the
    /// cells wrote, not what they **stopped** writing. A field that shrinks (a
    /// cell relaxes to zero and is dropped) produces a *smaller* rect, or `None`
    /// when the last cell goes — and with the camera parked, `None` meant zero
    /// uploads, which meant last frame's ruts stayed burned into the window for
    /// ever. Unioning with this clears exactly the vacated texels and nothing
    /// else.
    prev_rect: Option<(u32, u32, u32, u32)>,
    /// `write_texture` calls issued over this renderer's life — the engagement
    /// counter. An unchanged field over N frames must not move it.
    uploads: AtomicU64,
}

impl DeformResources {
    pub fn new(gpu: &crate::gpu::GpuContext) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("deform-window"),
            size: wgpu::Extent3d {
                width: DEFORM_WINDOW_TEXELS,
                height: DEFORM_WINDOW_TEXELS,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deform"),
            size: std::mem::size_of::<DeformUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            texture,
            view,
            uniform,
            shadow: Vec::new(),
            packed: None,
            prev_rect: None,
            uploads: AtomicU64::new(0),
        }
    }

    /// `write_texture` calls issued so far — the engagement counter.
    pub fn uploads(&self) -> u64 {
        self.uploads.load(Ordering::Relaxed)
    }

    /// Bring the window up to date for this frame and return the uniform to
    /// write.
    ///
    /// The gate, in order:
    /// 1. no projection (or an undrawable one) ⇒ **disabled uniform, zero
    ///    uploads**, and the shadow is forgotten so the next real projection
    ///    re-uploads in full rather than diffing against a stale window;
    /// 2. same origin *and* same epoch ⇒ the texture already holds the answer,
    ///    **zero uploads**;
    /// 3. the origin moved ⇒ every texel may be a different sample, so the whole
    ///    window uploads (one call);
    /// 4. only the field moved ⇒ upload the rect the cells wrote **unioned with
    ///    the rect they wrote last time**, so texels a shrinking field vacated
    ///    are cleared rather than left burned in.
    ///
    /// `eye_world_xz` is the camera's world XZ — the only camera this whole
    /// subsystem touches, and it decides what is *drawn*, never what is
    /// simulated. `sky` is `(level clock seconds, wind X, wind Z)`, taken off the
    /// cloud block both projectors already publish; `wind_on` is
    /// [`crate::ScatterSettings::foliage_wind`].
    pub fn update(
        &mut self,
        gpu: &crate::gpu::GpuContext,
        deform: Option<&RenderDeform>,
        origin: &FloatingOrigin,
        eye_world_xz: (f64, f64),
        sky: (f64, f32, f32),
        wind_on: bool,
    ) -> DeformUniform {
        let (eye_world_x, eye_world_z) = eye_world_xz;
        let (clock_s, wind_xz) = (sky.0, (sky.1 as f64, sky.2 as f64));
        let o = origin.origin();

        // The wind half is packed whether or not there is a field: sway is an
        // ambient property of foliage, not a consequence of somebody's boots.
        let speed = (wind_xz.0 * wind_xz.0 + wind_xz.1 * wind_xz.1).sqrt();
        let dir = if speed > 1e-4 {
            (wind_xz.0 / speed, wind_xz.1 / speed)
        } else {
            (0.0, 0.0)
        };
        let k = std::f64::consts::TAU / WIND_WAVELENGTH_M as f64;
        let wind_live = wind_on && speed > 1e-4;
        let wind = [
            dir.0 as f32,
            dir.1 as f32,
            k as f32,
            wind_phase((o.x, o.z), dir, speed, clock_s, k),
        ];

        let Some(deform) = deform.filter(|d| d.drawable()) else {
            self.packed = None;
            self.prev_rect = None;
            return DeformUniform {
                params: [
                    0.0,
                    DEFORM_MAX_DEPTH_M,
                    DEFORM_BEND_GAIN,
                    if wind_live { 1.0 } else { 0.0 },
                ],
                wind,
                ..DeformUniform::default()
            };
        };
        let texels = window_origin_texels(eye_world_x, eye_world_z);
        let world_min_x = texels.0 as f64 * DEFORM_TEXEL_M;
        let world_min_z = texels.1 as f64 * DEFORM_TEXEL_M;

        let key = (texels, deform.epoch);
        let moved = self.packed.map(|(t, _)| t) != Some(texels);
        if self.packed != Some(key) {
            let dirty = pack_deform_window(deform, texels, &mut self.shadow);
            let full = (0, 0, DEFORM_WINDOW_TEXELS, DEFORM_WINDOW_TEXELS);
            // A window that moved must be uploaded whole: every texel may now
            // carry a different sample, including the ones the cells did not
            // touch (they became zero, and zero is an answer).
            let rect = if moved {
                Some(full)
            } else {
                union_rect(dirty, self.prev_rect)
            };
            if let Some((x0, y0, w, h)) = rect {
                self.upload_rect(gpu, x0, y0, w, h);
            }
            self.packed = Some(key);
            self.prev_rect = dirty;
        }

        DeformUniform {
            window: [
                (world_min_x - o.x) as f32,
                (world_min_z - o.z) as f32,
                (1.0 / DEFORM_TEXEL_M) as f32,
                DEFORM_WINDOW_TEXELS as f32,
            ],
            params: [
                1.0,
                DEFORM_MAX_DEPTH_M,
                DEFORM_BEND_GAIN,
                if wind_live { 1.0 } else { 0.0 },
            ],
            wind,
        }
    }

    fn upload_rect(&self, gpu: &crate::gpu::GpuContext, x0: u32, y0: u32, w: u32, h: u32) {
        // Full-width rects (the moved case) hand the shadow straight over; a
        // narrower rect is repacked, because `write_texture` wants the sub-rect's
        // rows contiguous.
        if x0 == 0 && w == DEFORM_WINDOW_TEXELS {
            let n = DEFORM_WINDOW_TEXELS as usize;
            self.write(gpu, 0, y0, w, h, &self.shadow[y0 as usize * n..]);
            return;
        }
        let scratch = extract_rect(&self.shadow, x0, y0, w, h);
        self.write(gpu, x0, y0, w, h, &scratch);
    }

    fn write(&self, gpu: &crate::gpu::GpuContext, x0: u32, y0: u32, w: u32, h: u32, data: &[f32]) {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data[..(w * h) as usize]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.uploads.fetch_add(1, Ordering::Relaxed);
    }
}

/// The smallest rect containing both, or whichever exists.
///
/// The vacated-texel fix in one function, so it is testable without a GPU.
pub fn union_rect(
    a: Option<(u32, u32, u32, u32)>,
    b: Option<(u32, u32, u32, u32)>,
) -> Option<(u32, u32, u32, u32)> {
    match (a, b) {
        (None, None) => None,
        (Some(r), None) | (None, Some(r)) => Some(r),
        (Some(a), Some(b)) => {
            let x0 = a.0.min(b.0);
            let y0 = a.1.min(b.1);
            let x1 = (a.0 + a.2).max(b.0 + b.2);
            let y1 = (a.1 + a.3).max(b.1 + b.3);
            Some((x0, y0, x1 - x0, y1 - y0))
        }
    }
}

/// Copy the `w × h` sub-rect at `(x0, y0)` out of a `DEFORM_WINDOW_TEXELS`-wide
/// row-major buffer into a contiguous one — what `write_texture` needs for a
/// partial upload.
///
/// A free function, and public, precisely so **the partial-upload path is
/// reachable from a test**: it is the module's headline optimization and it was
/// previously covered by nothing at all (a `panic!` planted inside it stayed
/// green through the whole suite).
pub fn extract_rect(shadow: &[f32], x0: u32, y0: u32, w: u32, h: u32) -> Vec<f32> {
    let n = DEFORM_WINDOW_TEXELS as usize;
    let mut out = Vec::with_capacity((w * h) as usize);
    for j in 0..h as usize {
        let row = (y0 as usize + j) * n + x0 as usize;
        out.extend_from_slice(&shadow[row..row + w as usize]);
    }
    out
}

/// CPU mirror of the shader's window sample: bilinear over the deformation
/// texture, in **render-local** space, `0` outside the window.
///
/// Kept in sync with `deform.wgsl`'s `deform_depth`; unit-tested, because the
/// shader itself needs a GPU. `texels` is the packed window buffer.
pub fn deform_depth_reference(
    u: &DeformUniform,
    texels: &[f32],
    local_x: f32,
    local_z: f32,
) -> f32 {
    if !u.enabled() {
        return 0.0;
    }
    let n = u.window[3] as i32;
    let p = (
        (local_x - u.window[0]) * u.window[2],
        (local_z - u.window[1]) * u.window[2],
    );
    if p.0 < 0.0 || p.1 < 0.0 || p.0 > (n - 1) as f32 || p.1 > (n - 1) as f32 {
        return 0.0;
    }
    let (i0, j0) = (p.0.floor() as i32, p.1.floor() as i32);
    let (fx, fz) = (p.0 - i0 as f32, p.1 - j0 as f32);
    let load = |i: i32, j: i32| -> f32 {
        let i = i.clamp(0, n - 1) as usize;
        let j = j.clamp(0, n - 1) as usize;
        texels.get(j * n as usize + i).copied().unwrap_or(0.0)
    };
    let a = load(i0, j0) + (load(i0 + 1, j0) - load(i0, j0)) * fx;
    let b = load(i0, j0 + 1) + (load(i0 + 1, j0 + 1) - load(i0, j0 + 1)) * fx;
    (a + (b - a) * fz) * rim_fade(p.0, p.1, n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn cell(coord: (i32, i32), samples: u32, fill: f32) -> RenderDeformCell {
        RenderDeformCell {
            coord,
            depths: vec![fill; (samples * samples) as usize],
        }
    }

    fn projection(cells: Vec<RenderDeformCell>, epoch: u64) -> RenderDeform {
        RenderDeform {
            cell_samples: 16,
            texel_m: DEFORM_TEXEL_M,
            epoch,
            cells,
        }
    }

    #[test]
    fn the_window_geometry_is_self_consistent() {
        assert_eq!(
            DEFORM_WINDOW_M,
            DEFORM_WINDOW_TEXELS as f64 * DEFORM_TEXEL_M
        );
        // The window is centred on the camera to within one texel.
        let (x, z) = window_origin_texels(1000.0, -1000.0);
        let half = (DEFORM_WINDOW_TEXELS / 2) as f64 * DEFORM_TEXEL_M;
        assert!(((x as f64 * DEFORM_TEXEL_M) - (1000.0 - half)).abs() <= DEFORM_TEXEL_M);
        assert!(((z as f64 * DEFORM_TEXEL_M) - (-1000.0 - half)).abs() <= DEFORM_TEXEL_M);
    }

    #[test]
    fn the_snap_quantizes_camera_motion() {
        let a = window_origin_texels(100.0, 100.0);
        // Sub-texel motion must not move the window at all…
        assert_eq!(window_origin_texels(100.05, 100.05), a);
        // …and one texel of motion must move it by exactly one.
        assert_eq!(
            window_origin_texels(100.0 + DEFORM_TEXEL_M, 100.0),
            (a.0 + 1, a.1)
        );
    }

    #[test]
    fn the_pack_is_pure_and_writes_the_whole_buffer() {
        let d = projection(vec![cell((0, 0), 16, 0.3)], 1);
        let mut a = Vec::new();
        let ra = pack_deform_window(&d, (0, 0), &mut a);
        let mut b = vec![9.0f32; 5];
        let rb = pack_deform_window(&d, (0, 0), &mut b);
        assert_eq!(a, b);
        assert_eq!(ra, rb);
        assert_eq!(
            a.len(),
            (DEFORM_WINDOW_TEXELS * DEFORM_WINDOW_TEXELS) as usize
        );
        assert_eq!(ra, Some((0, 0, 16, 16)));
        assert_eq!(a[0], 0.3);
        assert_eq!(a[17 * DEFORM_WINDOW_TEXELS as usize], 0.0);
    }

    #[test]
    fn a_pan_translates_the_same_samples() {
        // THE REBASE / PAN ARM: the window is an index map, so shifting the
        // origin must move the values without changing any of them.
        let d = projection(vec![cell((4, 4), 16, 0.2)], 1);
        let n = DEFORM_WINDOW_TEXELS as usize;
        let mut a = Vec::new();
        pack_deform_window(&d, (0, 0), &mut a);
        let mut b = Vec::new();
        pack_deform_window(&d, (7, 0), &mut b);
        for j in 0..n {
            for i in 0..n - 7 {
                assert_eq!(b[j * n + i], a[j * n + i + 7], "texel ({i}, {j}) resampled");
            }
        }
    }

    #[test]
    fn a_malformed_cell_is_skipped_whole() {
        let mut d = projection(vec![cell((0, 0), 16, 0.3)], 1);
        d.cells[0].depths.truncate(10);
        let mut out = Vec::new();
        assert_eq!(pack_deform_window(&d, (0, 0), &mut out), None);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn depths_are_clamped_to_the_skirt_floor() {
        // The coupling with `passes::terrain`'s `.max(1.0)` skirt: a projection
        // carrying a deeper value must not reach the vertex stage.
        let d = projection(vec![cell((0, 0), 16, 12.0)], 1);
        let mut out = Vec::new();
        pack_deform_window(&d, (0, 0), &mut out);
        assert_eq!(out[0], DEFORM_MAX_DEPTH_M);
    }

    #[test]
    fn an_empty_projection_draws_nothing() {
        let d = projection(Vec::new(), 1);
        assert!(!d.drawable());
        let mut out = Vec::new();
        assert_eq!(pack_deform_window(&d, (0, 0), &mut out), None);
        assert!(out.iter().all(|v| *v == 0.0));
        assert!(!DeformUniform::default().enabled());
    }

    #[test]
    fn the_reference_sampler_is_bilinear_and_bounded() {
        let mut u = DeformUniform {
            params: [1.0, DEFORM_MAX_DEPTH_M, DEFORM_BEND_GAIN, 0.0],
            ..Default::default()
        };
        u.window[0] = 0.0;
        u.window[1] = 0.0;
        let n = DEFORM_WINDOW_TEXELS as usize;
        let mut texels = vec![0.0f32; n * n];
        // Probe well inside the rim fade, or the fade (not the filter) would be
        // what the assertions below measured.
        let c = n / 2;
        texels[c * n + c] = 1.0;
        let at = |i: usize| (i as f64 * DEFORM_TEXEL_M) as f32;
        // Exact at the sample…
        assert_eq!(deform_depth_reference(&u, &texels, at(c), at(c)), 1.0);
        // …halfway along x is the mean of the two…
        let half =
            deform_depth_reference(&u, &texels, at(c) + (DEFORM_TEXEL_M / 2.0) as f32, at(c));
        assert!((half - 0.5).abs() < 1e-6, "{half}");
        // …and outside the window it is zero, never a clamped edge smear.
        assert_eq!(deform_depth_reference(&u, &texels, -1.0, at(c)), 0.0);
        assert_eq!(
            deform_depth_reference(&u, &texels, (DEFORM_WINDOW_M + 1.0) as f32, at(c)),
            0.0
        );
        // A disabled uniform never reads the texture at all.
        let off = DeformUniform::default();
        assert_eq!(deform_depth_reference(&off, &texels, at(c), at(c)), 0.0);
    }

    #[test]
    fn the_window_uniform_is_render_local() {
        // The window's world position is fixed by the camera; the uniform's is
        // that minus the floating origin, so a rebase moves the numbers and not
        // the footprints.
        let far = FloatingOrigin::new(DVec3::new(100_000.0, 0.0, 100_000.0));
        let texels = window_origin_texels(100_000.0, 100_000.0);
        let local_x = (texels.0 as f64 * DEFORM_TEXEL_M - far.origin().x) as f32;
        assert!(
            local_x.abs() < DEFORM_WINDOW_M as f32,
            "a render-local window min must stay small: {local_x}"
        );
    }

    // ── THE CITED GATES ──────────────────────────────────────────────────────
    //
    // The three tests below are named in doc comments elsewhere in this module
    // and in `deform.wgsl`. The first cut cited all three and defined none of
    // them, which is the P20 law ("a cited gate that doesn't exist is worse than
    // no claim") broken three times over. They exist now.

    /// Cited by [`DEFORM_TEXEL_M`]. The renderer's window geometry and the Ring-0
    /// field's lattice must be the SAME numbers, or a texel would carry a
    /// different sample than the one packed into it.
    #[test]
    fn the_window_geometry_matches_the_field() {
        assert_eq!(
            DEFORM_TEXEL_M,
            inf_terrain::deform::DEFORM_SAMPLE_PITCH_M,
            "a window texel must be exactly one field sample — otherwise the pack \
             is a resample and the whole no-filter argument is false"
        );
        assert_eq!(
            DEFORM_MAX_DEPTH_M,
            inf_terrain::deform::MAX_DEFORM_DEPTH_M,
            "the renderer's clamp and the field's clamp are the same skirt \
             coupling; they cannot differ"
        );
        // The window must be a whole number of cells across, or a cell would
        // straddle the rim in a way the pack's cell walk does not model.
        assert_eq!(
            DEFORM_WINDOW_TEXELS % inf_terrain::deform::DEFORM_CELL_SAMPLES,
            0
        );
    }

    /// Cited by [`window_origin_texels`]. `inf-terrain` and `inf-render` each
    /// have their own snap — Ring 0 cannot name the renderer and the renderer
    /// cannot name Ring 0 — so the ONLY thing keeping them the same function is
    /// this test. Until it existed, both were dead code as far as the other was
    /// concerned.
    #[test]
    fn the_snap_matches_the_field() {
        for &(x, z) in &[
            (0.0, 0.0),
            (0.1, -0.1),
            (123.456, -987.654),
            (-0.125, 0.125),
            (1e6, -1e6),
        ] {
            assert_eq!(
                window_origin_texels(x, z),
                inf_terrain::deform::window_origin(glam::DVec2::new(x, z), DEFORM_WINDOW_TEXELS),
                "the two snaps disagree at ({x}, {z})"
            );
        }
    }

    /// The two **packs** agree too, texel for texel: `DeformField::pack_window`
    /// (Ring 0, used by nothing but its own tests) and `pack_deform_window` (the
    /// renderer's, used by every frame). Two independent implementations of one
    /// rule, now exercised against each other.
    #[test]
    fn the_two_window_packs_agree() {
        use inf_terrain::deform::{DeformField, PressureClass, DEFORM_CELL_SAMPLES};
        let mut field = DeformField::new();
        for i in 0..24 {
            field.relax(1.0 / 60.0, 0.0);
            field.stamp_contact(
                glam::DVec2::new(3.0 + i as f64 * 0.4, 4.0),
                0.35,
                PressureClass::Heavy,
                3,
                1.0 / 60.0,
            );
        }
        assert!(!field.is_empty());
        let origin = window_origin_texels(5.0, 4.0);

        let mut ring0 = Vec::new();
        field.pack_window(origin, DEFORM_WINDOW_TEXELS, &mut ring0);

        let projection = RenderDeform {
            cell_samples: DEFORM_CELL_SAMPLES,
            texel_m: DEFORM_TEXEL_M,
            epoch: field.epoch(),
            cells: field
                .cells()
                .map(|(coord, cell)| RenderDeformCell {
                    coord: *coord,
                    depths: cell.depths().to_vec(),
                })
                .collect(),
        };
        let mut renderer = Vec::new();
        pack_deform_window(&projection, origin, &mut renderer);

        assert_eq!(ring0.len(), renderer.len());
        assert_eq!(
            ring0, renderer,
            "the Ring-0 pack and the renderer's pack disagree — one of the two \
             window implementations is wrong and neither test would have said so"
        );
        assert!(
            renderer.iter().any(|d| *d > 0.0),
            "the fixture must be non-empty"
        );
    }

    /// Cited by [`DeformUniform`]. The Rust struct and the WGSL `struct Deform`
    /// must agree field for field, in order and in size — a `vec4` that drifted
    /// would silently re-interpret the window origin as the wind direction.
    #[test]
    fn the_uniform_matches_the_shader_struct() {
        // Three `vec4<f32>`, std140-clean, no padding.
        assert_eq!(std::mem::size_of::<DeformUniform>(), 3 * 16);
        assert_eq!(std::mem::align_of::<DeformUniform>(), 4);

        let src = include_str!("shaders/deform.wgsl");
        let body = src
            .split_once("struct Deform {")
            .expect("the shader declares struct Deform")
            .1
            .split_once("};")
            .expect("…and closes it")
            .0;
        let fields: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
            .collect();
        assert_eq!(
            fields,
            vec![
                "window: vec4<f32>,",
                "params: vec4<f32>,",
                "wind: vec4<f32>,",
            ],
            "the WGSL `Deform` no longer matches `DeformUniform`'s three vec4s"
        );
        // And the shader really does read the two flags out of the slots this
        // side writes them into.
        assert!(src.contains("dfm.params.x > 0.5"));
        assert!(src.contains("dfm.params.w > 0.5"));
    }

    // ── the vacated-texel fix, and the partial-upload path ──────────────────

    #[test]
    fn a_shrinking_field_clears_the_texels_it_vacated() {
        // The union rule in isolation: last frame's rect must be re-uploaded so
        // the texels it wrote are overwritten with the zeroes they now hold.
        assert_eq!(union_rect(None, None), None);
        assert_eq!(union_rect(Some((4, 4, 2, 2)), None), Some((4, 4, 2, 2)));
        assert_eq!(union_rect(None, Some((4, 4, 2, 2))), Some((4, 4, 2, 2)));
        assert_eq!(
            union_rect(Some((0, 0, 2, 2)), Some((6, 8, 1, 1))),
            Some((0, 0, 7, 9))
        );
        // The case that matters: the field shrank to nothing, so the current
        // dirty rect is None and the PREVIOUS one is what must be cleared. A
        // `None` here was last frame's ruts burned into the window for ever.
        assert_eq!(union_rect(None, Some((10, 10, 4, 4))), Some((10, 10, 4, 4)));
    }

    #[test]
    fn the_partial_upload_extracts_the_right_texels() {
        // The module's headline optimization, which was covered by NOTHING: a
        // `panic!` planted in the sub-rect repack stayed green through the entire
        // suite because no test ever took that branch.
        let n = DEFORM_WINDOW_TEXELS as usize;
        let mut shadow = vec![0.0f32; n * n];
        // A recognisable pattern: value = x + 1000 * y.
        for y in 0..n {
            for x in 0..n {
                shadow[y * n + x] = x as f32 + 1000.0 * y as f32;
            }
        }
        let (x0, y0, w, h) = (7u32, 11u32, 5u32, 3u32);
        let rect = extract_rect(&shadow, x0, y0, w, h);
        assert_eq!(rect.len(), (w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                assert_eq!(
                    rect[(j * w + i) as usize],
                    (x0 + i) as f32 + 1000.0 * (y0 + j) as f32,
                    "sub-rect texel ({i}, {j}) came from the wrong place"
                );
            }
        }
        // A full-width rect is the identity on its rows (the fast path's premise).
        let full = extract_rect(&shadow, 0, 2, DEFORM_WINDOW_TEXELS, 2);
        assert_eq!(&full[..], &shadow[2 * n..4 * n]);
    }

    // ── the window rim ──────────────────────────────────────────────────────

    #[test]
    fn the_window_rim_fades_instead_of_cutting_off() {
        let n = DEFORM_WINDOW_TEXELS as f32;
        // Dead centre: untouched.
        assert_eq!(rim_fade(n * 0.5, n * 0.5, n), 1.0);
        // On the rim: nothing.
        assert_eq!(rim_fade(0.0, n * 0.5, n), 0.0);
        assert_eq!(rim_fade(n - 1.0, n * 0.5, n), 0.0);
        assert_eq!(rim_fade(n * 0.5, 0.0, n), 0.0);
        // Continuous and monotone across the ramp — the property that stops the
        // straight line across the ground at 64 m.
        let mut prev = 0.0;
        for k in 0..=DEFORM_RIM_TEXELS as u32 {
            let f = rim_fade(k as f32, n * 0.5, n);
            assert!(f >= prev - 1e-6, "the rim fade is not monotone at {k}");
            assert!((0.0..=1.0).contains(&f));
            prev = f;
        }
        assert_eq!(rim_fade(DEFORM_RIM_TEXELS, n * 0.5, n), 1.0);
        // …and the sampler applies it, so a depth at the rim really is gone.
        let mut u = DeformUniform {
            params: [1.0, DEFORM_MAX_DEPTH_M, DEFORM_BEND_GAIN, 0.0],
            ..Default::default()
        };
        u.window[0] = 0.0;
        u.window[1] = 0.0;
        let n_i = DEFORM_WINDOW_TEXELS as usize;
        let texels = vec![0.5f32; n_i * n_i];
        let middle = (DEFORM_WINDOW_M / 2.0) as f32;
        assert!((deform_depth_reference(&u, &texels, middle, middle) - 0.5).abs() < 1e-6);
        assert_eq!(deform_depth_reference(&u, &texels, 0.0, middle), 0.0);
    }

    // ── wind ────────────────────────────────────────────────────────────────

    #[test]
    fn the_wind_phase_is_origin_invariant_and_never_freezes() {
        let k = std::f64::consts::TAU / WIND_WAVELENGTH_M as f64;
        let dir = (0.9486832980505138, 0.31622776601683794); // unit (3, 1)
        let speed = 6.32455532;
        let clock = 4_321.5;

        // ORIGIN INVARIANCE: the phase the shader ends up evaluating at a fixed
        // WORLD point must not depend on where the floating origin happens to be.
        // shader phase = dot(local, dir) * k + offset, and local = world - origin.
        let world = (1234.5, -678.25);
        let total = |origin: (f64, f64)| {
            let local = (world.0 - origin.0, world.1 - origin.1);
            let offset = wind_phase(origin, dir, speed, clock, k) as f64;
            ((local.0 * dir.0 + local.1 * dir.1) * k + offset).rem_euclid(std::f64::consts::TAU)
        };
        let a = total((0.0, 0.0));
        for origin in [(10.0, 0.0), (-1230.0, 670.0), (100_000.0, -100_000.0)] {
            let b = total(origin);
            assert!(
                (a - b).abs() < 1e-3 || (a - b).abs() > std::f64::consts::TAU - 1e-3,
                "a rebase to {origin:?} moved the wind phase from {a} to {b} — the \
                 grass would flick to a new point of its cycle when the camera \
                 crossed a 10 m boundary"
            );
        }

        // NEVER FREEZES: a day-long clock still advances the phase between
        // consecutive frames. This is the f32 stall the reduction exists to dodge
        // — `speed * 86400 * k` in f32 has no room left for a 16 ms step.
        let day = 86_400.0;
        let p0 = wind_phase((0.0, 0.0), dir, speed, day, k);
        let p1 = wind_phase((0.0, 0.0), dir, speed, day + 1.0 / 60.0, k);
        assert_ne!(p0, p1, "the wind phase froze at a day-long clock");
        assert!(p0.is_finite() && (0.0..std::f32::consts::TAU).contains(&p0));

        // Zero wind ⇒ the phase is a constant, and the direction is zero, so the
        // sway term vanishes identically rather than jittering around zero.
        assert_eq!(wind_phase((5.0, 5.0), (0.0, 0.0), 0.0, 99.0, k), 0.0);
    }

    #[test]
    fn wind_and_the_window_are_separate_switches() {
        // The uniform's two flags are independent — which is the whole point of
        // splitting them. A scene with no field may still sway; a scene with a
        // field may still be windless.
        let off = DeformUniform::default();
        assert!(!off.enabled());
        assert!(!off.wind_enabled());
        let windy = DeformUniform {
            params: [0.0, DEFORM_MAX_DEPTH_M, DEFORM_BEND_GAIN, 1.0],
            ..Default::default()
        };
        assert!(!windy.enabled());
        assert!(windy.wind_enabled());
        let pressed = DeformUniform {
            params: [1.0, DEFORM_MAX_DEPTH_M, DEFORM_BEND_GAIN, 0.0],
            ..Default::default()
        };
        assert!(pressed.enabled());
        assert!(!pressed.wind_enabled());
    }
}
