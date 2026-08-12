//! **The virtual-texture streaming loop** (P26.4): the analytic want floor, the
//! GPU coverage feedback that refines it, and the pop-in instruments.
//!
//! One call per frame, at the renderer's sync point, doing five things in a fixed
//! order:
//!
//! 1. gather this frame's **coverage** — every drawn surface that names a virtual
//!    texture, as a render-local bounding sphere;
//! 2. compute the **analytic floor** from `(camera, bounds, registry)`;
//! 3. read the feedback mask from **frame F − 2** (or nothing) and decode it into
//!    refinement wants in virtual-address order;
//! 4. apply `floor ∪ refinement` to the residency and upload the pages;
//! 5. record the next feedback pass and hand its buffer to the ring.
//!
//! # The floor is a floor
//!
//! Residency can never fall below the analytic floor because of anything the
//! feedback does or fails to do, and that is structural rather than a policy:
//!
//! * every floor want is [`VT_PRIORITY_FLOOR`](inf_vt::VT_PRIORITY_FLOOR) and
//!   every refinement is [`VT_PRIORITY_FEEDBACK`](inf_vt::VT_PRIORITY_FEEDBACK),
//!   and `VtResidency::apply_wants` sorts on priority first and never evicts a
//!   page it has already touched **this transaction** — so a refinement cannot
//!   take a floor tile's slot;
//! * a late, dropped or never-arriving mask contributes **zero** refinements, so
//!   the transaction is exactly the floor's. A gate can therefore assert that a
//!   frame with feedback disabled produces the same trace as a frame whose ring
//!   missed, which is what "degrades deterministically" has to mean.
//!
//! # What the floor is, and what it costs
//!
//! `VtTextures::want_floor`'s three coarsest levels are **camera-free** and
//! bounded at 21 pages per texture — and on a 4096² texture those three levels
//! are 4×4, 2×2 and 1×1 *texels*. That is "visibly textured, not sharp", exactly
//! as the P26.3 ledger says. The analytic floor adds, per **visible** surface,
//! the finest level whose tile count fits [`VT_FLOOR_MAX_TILES`] and that is no
//! finer than the surface's screen footprint justifies. So the floor's cost is
//! bounded by `visible surfaces × VT_FLOOR_MAX_TILES` and its quality tracks the
//! camera; the feedback then asks for the level the footprint *really* justifies,
//! uncapped, and the budget decides how much of it is served.
//!
//! Both halves compute the level from the same rule. That is deliberate: the
//! floor is the answer the CPU can always give, the feedback is the same question
//! asked where the whole draw set is already on the GPU and where P27/P28 extend
//! it. See `docs/memos/p26-4-feedback-mechanism.md`.

use std::collections::BTreeMap;

use glam::{Mat4, Vec3};
use inf_vt::{
    TileCoord, VtFeedbackLayout, VtTextureHandle, VtWant, VT_PRIORITY_FEEDBACK, VT_PRIORITY_FLOOR,
};

use crate::camera::RenderView;
use crate::readback::ReadbackRing;
use crate::scene::{RenderScene, VtTextureSet};
use crate::vt_library::VtTextures;

/// The most tiles the **analytic floor** claims for one visible surface texture.
///
/// Sixteen: a 4×4 tile grid is 512² payload texels, which is a legible surface at
/// any framing a floor has to survive, and sixteen BC1 pages is 148 KiB. The
/// number bounds the floor's whole cost — `visible surfaces × 16` — which is what
/// lets it be claimed unconditionally, before any feedback, on every frame.
///
/// It is **not** the sharpness ceiling: the GPU feedback marks the level the
/// footprint justifies with no such cap, and the budget decides how much of that
/// is served. This is the floor below which residency never falls.
pub const VT_FLOOR_MAX_TILES: u32 = 16;

/// The most tiles **one feedback request** may mark, handed to the shader.
///
/// A bound on the compute pass's inner loop rather than on quality: a level with
/// more tiles than this is refused in favour of the next coarser one, so a
/// single 8192² texture filling the screen asks for a level that can plausibly
/// be served instead of for four thousand pages that will all be deferred. 256
/// tiles is 4 Mtexels of unique detail for one surface.
pub const VT_FEEDBACK_MAX_TILES: u32 = 256;

/// One drawn surface that names a virtual texture (P26.4).
///
/// Deliberately a **bounding sphere and a texture set**, not a mesh: what the
/// floor and the feedback both need is "how much of the screen does this surface
/// cover", and every path — rigid primitive, meshlet asset, skinned character —
/// can answer that from its transform and its bounds without the projector
/// learning anything about virtual texturing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VtCoverage {
    /// Render-local centre (metres).
    pub centre: Vec3,
    /// Bounding-sphere radius (metres).
    pub radius: f32,
    /// The three slots this surface samples (`handle + 1`, 0 = none).
    pub set: VtTextureSet,
}

/// Pop-in instrumentation (P26.4, clause 5) — **per tile class**, because the two
/// classes fail differently and one number hides which.
///
/// A *floor* tile at fallback is a budget failure: the level's conservative,
/// bounded claim did not fit, so the surface is blurrier than the floor promises.
/// A *refinement* tile at fallback is the ordinary steady state on the way to
/// sharp — it is what pop-in **is** — and the interesting reading is how long it
/// lasts.
///
/// All counters are frame-summed `(frame, want)` pairs, so "how many frames did
/// this scene spend at fallback" is a ratio against [`wants`](Self::floor_wants).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VtPopIn {
    /// Floor wants offered, summed over frames.
    pub floor_wants: u64,
    /// …of which resolved to an ancestor rather than to themselves.
    pub floor_fallback: u64,
    /// Refinement wants offered, summed over frames.
    pub refine_wants: u64,
    /// …of which resolved to an ancestor rather than to themselves.
    pub refine_fallback: u64,
    /// Frames in which a feedback mask arrived from the ring.
    pub feedback_frames: u64,
    /// Frames in which it did not, and the floor stood alone.
    pub feedback_misses: u64,
    /// Wants the residency could not seat, summed over frames.
    pub deferred: u64,
    /// Pages admitted, summed over frames — the **anti-vacuity** number: a
    /// streaming loop that never admits anything also never reports a fallback.
    pub admits: u64,
    /// Frames this streamer has run.
    pub frames: u64,
}

impl VtPopIn {
    /// A one-line human summary, in the shape `VtStats::summary` and
    /// `TerrainStreamStats::summary` already ship so three streamers read alike
    /// in one log.
    pub fn summary(&self) -> String {
        format!(
            "vt streaming: {} frames, {} admits, {} deferred, floor {}/{} at fallback, \
             refine {}/{} at fallback, feedback {} landed / {} missed",
            self.frames,
            self.admits,
            self.deferred,
            self.floor_fallback,
            self.floor_wants,
            self.refine_fallback,
            self.refine_wants,
            self.feedback_frames,
            self.feedback_misses,
        )
    }
}

/// **Which mip level a surface's screen footprint justifies** — the ONE rule, on
/// the CPU, mirrored by `vt_feedback.wgsl`.
///
/// `extent` is the texture's mip-0 long side in texels, `screen_px` the diameter
/// of the surface's projected bounding sphere in pixels. A level of `extent /
/// 2^L` texels covering `screen_px` pixels is right when its texels land about
/// one per pixel, so `L = ceil(log2(extent / screen_px))`, clamped into the
/// pyramid.
///
/// `ceil` and not `round`: erring coarse costs blur, erring fine costs pages
/// that will be deferred and a want set that never converges.
pub fn justified_mip(extent: u32, screen_px: f32, mip_count: u32) -> u32 {
    if mip_count == 0 {
        return 0;
    }
    let want = (extent.max(1) as f32 / screen_px.max(1.0)).log2().ceil();
    want.clamp(0.0, (mip_count - 1) as f32) as u32
}

/// The projected diameter, in pixels, of a bounding sphere at `centre` — the
/// quantity both the floor and the shader derive their level from.
///
/// `proj_scale` is pixels per world unit at one metre
/// (`0.5 · height / tan(fov_y / 2)`); an orthographic view has no perspective
/// divide, so it uses its own constant scale and the distance term drops out.
pub fn screen_diameter_px(centre: Vec3, radius: f32, eye: Vec3, proj_scale: f32) -> f32 {
    let dist = (centre - eye).length().max(1e-4);
    2.0 * radius * proj_scale / dist
}

/// Pixels per world unit at one metre for `view`.
pub fn projection_scale(view: &RenderView) -> f32 {
    match view.ortho {
        // Orthographic: the whole frame is `2 · half_height` metres tall, so the
        // scale is constant and the caller's distance term is inert (it divides
        // by `dist` and this multiplies by it — see `screen_diameter_px`, where
        // an ortho view is handled by making the product distance-free).
        Some(o) => 0.5 * view.height as f32 / o.half_height.max(1e-4),
        None => 0.5 * view.height as f32 / (view.fov_y * 0.5).tan().max(1e-4),
    }
}

/// **Is this surface on screen?** — the camera term of the analytic floor, and
/// the CPU twin of the shader's frustum test.
///
/// A conservative sphere-vs-frustum test in clip space: behind the eye is out,
/// and outside the NDC box by more than the sphere's own projected margin is
/// out. Conservative in the "keeps too much" direction, which is the correct
/// direction for a floor.
pub fn on_screen(view_proj: &Mat4, centre: Vec3, radius: f32, ndc_margin: f32) -> bool {
    let clip = *view_proj * centre.extend(1.0);
    if clip.w <= 0.0 {
        // Behind the eye — unless the sphere still straddles it.
        return radius > -clip.w;
    }
    let ndc = clip.truncate() / clip.w;
    ndc.x.abs() <= 1.0 + ndc_margin && ndc.y.abs() <= 1.0 + ndc_margin
}

/// **Gather every drawn surface that names a virtual texture** (P26.4).
///
/// Reads the render scene the frame is about to draw — rigid instances, meshlet
/// instances and skinned characters — so the coverage is a function of what is
/// actually being submitted rather than of what the document contains. A surface
/// whose set is [`VtTextureSet::NONE`] contributes nothing, so a textureless
/// scene produces an empty list and the whole loop below is skipped.
///
/// Bounds are the honest conservative ones each path can offer: a built-in
/// primitive is a unit shape, so its sphere is `√3/2 · max(scale)`; a meshlet
/// asset carries its own bounding sphere in its header (read without paging
/// anything); a skinned character has no cheap bound, so it takes the primitive
/// rule against its instance scale — which is why a character's floor is
/// conservative rather than tight.
pub fn scene_coverage(scene: &RenderScene, origin: &inf_math::FloatingOrigin) -> Vec<VtCoverage> {
    /// Half the diagonal of a unit cube — the bounding sphere of every built-in
    /// primitive at unit scale.
    const UNIT_RADIUS: f32 = 0.866_025_4;

    let mut out = Vec::new();
    for i in &scene.instances {
        if i.vt.is_none() {
            continue;
        }
        out.push(VtCoverage {
            centre: origin.to_render(i.translation),
            radius: UNIT_RADIUS * i.scale.abs().max_element().max(1e-4),
            set: i.vt,
        });
    }
    let bounds: BTreeMap<u128, f32> = scene
        .vgeom_assets
        .iter()
        .map(|a| (a.id, a.bounds().1))
        .collect();
    for i in &scene.vgeom_instances {
        if i.vt.is_none() {
            continue;
        }
        let r = bounds.get(&i.asset).copied().unwrap_or(UNIT_RADIUS);
        out.push(VtCoverage {
            centre: origin.to_render(i.translation),
            radius: r * i.scale.abs().max_element().max(1e-4),
            set: i.vt,
        });
    }
    for i in &scene.skinned {
        if i.vt.is_none() {
            continue;
        }
        out.push(VtCoverage {
            centre: origin.to_render(i.translation),
            radius: UNIT_RADIUS * i.scale.abs().max_element().max(1e-4),
            set: i.vt,
        });
    }
    out
}

/// **The analytic want-set floor** (P26.4, clause 4): `(camera, bounds, registry)`
/// → a conservative tile set, computed every frame.
///
/// Two parts, and the first is the reason the second can be bounded:
///
/// * the **camera-free** part is `VtTextures::want_floor` — the three coarsest
///   levels of every registered texture, at most 21 pages each however large the
///   texture. It is what guarantees a sample is never a hole even for a surface
///   that is off screen, behind the camera, or not drawn this frame at all.
/// * the **camera-driven** part adds, for every *visible* surface, the finest
///   level of each of its three maps that fits [`VT_FLOOR_MAX_TILES`] and is no
///   finer than [`justified_mip`] allows. Bounded by construction, so the floor's
///   cost is `visible surfaces × 16` pages and cannot be made unbounded by a
///   scene.
///
/// Every want is [`VT_PRIORITY_FLOOR`], which is what makes it a floor: the
/// residency serves the whole class before a refinement is offered a slot.
///
/// **Deterministic in committed input.** Nothing here reads a clock, a frame
/// counter or a previous frame's state — two runs of one scripted camera path
/// over one scene produce one want sequence, which is what the phase gate pins.
pub fn analytic_floor(lib: &VtTextures, view: &RenderView, coverage: &[VtCoverage]) -> Vec<VtWant> {
    let mut out = lib.want_floor();
    if coverage.is_empty() {
        return out;
    }
    let view_proj = view.view_proj();
    let proj_scale = projection_scale(view);
    let eye = view.eye_local();
    let half_h = (view.height as f32 * 0.5).max(1.0);
    for c in coverage {
        let px = screen_diameter_px(c.centre, c.radius, eye, proj_scale);
        if !on_screen(&view_proj, c.centre, c.radius, px / half_h) {
            continue;
        }
        for slot in c.set.slots() {
            if slot == 0 {
                continue;
            }
            let handle = VtTextureHandle(slot - 1);
            let Some(desc) = lib.residency().desc(handle) else {
                continue;
            };
            let extent = desc.mips[0].width.max(desc.mips[0].height);
            let mut lv = justified_mip(extent, px, desc.mip_count());
            // Coarser until the level fits the floor's per-surface cap.
            while lv + 1 < desc.mip_count()
                && desc.mips[lv as usize].tile_count() > VT_FLOOR_MAX_TILES
            {
                lv += 1;
            }
            let m = desc.mips[lv as usize];
            for y in 0..m.tiles_y {
                for x in 0..m.tiles_x {
                    out.push(
                        VtWant::new(handle, TileCoord::new(lv, x, y))
                            .with_priority(VT_PRIORITY_FLOOR),
                    );
                }
            }
        }
    }
    out
}

/// One feedback request as the compute shader reads it — 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FeedbackRequest {
    /// xyz = render-local centre, w = radius (metres).
    pub centre: [f32; 4],
    /// x = word offset of the texture's block in the table, y = its first bit in
    /// the mask, zw = reserved.
    pub tex: [u32; 4],
}

/// The uniform the feedback pass reads — 96 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FeedbackParams {
    pub view_proj: [f32; 16],
    /// xyz = render-local eye, w = pixels per world unit at one metre.
    pub eye: [f32; 4],
    /// x = request count, y = mask words, z = max tiles per request, w = half
    /// the viewport height in pixels.
    pub counts: [u32; 4],
}

/// **Build the feedback request list** for one frame — the same coverage the
/// analytic floor used, expanded to one entry per (surface × bound slot) and
/// resolved into table offsets on the CPU.
///
/// The block offset and the base bit are computed here rather than in the shader
/// because both are already known exactly: `VtResidency::table_block` gives the
/// first, [`VtFeedbackLayout::texture_base`] the second, and a shader that
/// re-derived either would be a second copy of the layout.
///
/// Ordered by `(slot, index)` — a stable function of the coverage list, which is
/// itself a stable function of the scene. The mask does not care (OR is
/// order-independent), and a deterministic upload keeps the *command stream*
/// comparable between runs, which the golden harness does care about.
pub fn feedback_requests(
    lib: &VtTextures,
    layout: &VtFeedbackLayout,
    coverage: &[VtCoverage],
) -> Vec<FeedbackRequest> {
    let mut out = Vec::new();
    for c in coverage {
        for slot in c.set.slots() {
            if slot == 0 {
                continue;
            }
            let handle = VtTextureHandle(slot - 1);
            let (Some((block, _)), Some(base)) = (
                lib.residency().table_block(handle),
                layout.texture_base(handle),
            ) else {
                continue;
            };
            out.push(FeedbackRequest {
                centre: [c.centre.x, c.centre.y, c.centre.z, c.radius],
                tex: [block as u32, base, 0, 0],
            });
        }
    }
    out
}

/// Split a want list into the floor and refinement classes — what the pop-in
/// counters are summed over.
pub(crate) fn count_fallbacks(lib: &VtTextures, wants: &[VtWant], stats: &mut VtPopIn) {
    for w in wants {
        let at_fallback = !lib.residency().is_resident(w.texture, w.tile);
        if w.priority == VT_PRIORITY_FEEDBACK {
            stats.refine_wants += 1;
            stats.refine_fallback += u64::from(at_fallback);
        } else {
            stats.floor_wants += 1;
            stats.floor_fallback += u64::from(at_fallback);
        }
    }
}

/// The GPU half of the loop: the coverage bitmask, its compute pass, and the
/// readback ring that reads it at a pinned latency.
pub struct VtFeedback {
    layout: VtFeedbackLayout,
    mask: wgpu::Buffer,
    params: wgpu::Buffer,
    requests: wgpu::Buffer,
    request_cap: u32,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    bind: Option<(u64, wgpu::BindGroup)>,
    ring: ReadbackRing,
}

impl VtFeedback {
    /// A feedback pass sized for `layout`.
    pub fn new(device: &wgpu::Device, layout: VtFeedbackLayout, request_cap: u32) -> Self {
        let words = layout.words() as u64;
        let mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt-feedback-mask"),
            size: words * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt-feedback-params"),
            size: std::mem::size_of::<FeedbackParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let requests = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt-feedback-requests"),
            size: (request_cap.max(1) as u64) * std::mem::size_of::<FeedbackRequest>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vt-feedback"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vt_feedback.wgsl").into()),
        });
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vt-feedback"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vt-feedback"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vt-feedback"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_feedback"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            mask,
            params,
            requests,
            request_cap: request_cap.max(1),
            pipeline,
            bgl,
            bind: None,
            ring: ReadbackRing::new(device, "vt-feedback", words * 4),
        }
    }

    /// The bitmask layout this pass writes.
    #[inline]
    pub fn layout(&self) -> &VtFeedbackLayout {
        &self.layout
    }

    /// The ring, for a host or a gate that wants its hit/miss counts.
    #[inline]
    pub fn ring(&self) -> &ReadbackRing {
        &self.ring
    }

    /// **Read frame `frame`'s feedback** — the mask recorded at `frame − 2`, or
    /// `None`. Never an adjacent frame: see [`crate::readback`].
    pub fn take_wants(
        &mut self,
        device: &wgpu::Device,
        lib: &VtTextures,
        frame: u64,
    ) -> Option<Vec<VtWant>> {
        self.ring.poll(device);
        let layout = self.layout.clone();
        let residency = lib.residency();
        self.ring.take(frame, |bytes| {
            let words: &[u32] = bytemuck::cast_slice(bytes);
            layout.wants(residency, words)
        })
    }

    /// **Record the next feedback pass** into `encoder` and hand its buffer to
    /// the ring.
    ///
    /// The mask is cleared first, in the same encoder, so a frame's coverage is
    /// this frame's and never an OR with the last one — which would make the
    /// signal depend on the frame history, the exact property the pinned ring
    /// exists to avoid.
    ///
    /// Returns the number of requests dispatched (0 = nothing recorded, and the
    /// ring gets no copy, so the read two frames later misses and the floor
    /// stands — the same degradation as a late mask).
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        table: &wgpu::Buffer,
        table_generation: u64,
        view: &RenderView,
        requests: &[FeedbackRequest],
        frame: u64,
    ) -> u32 {
        let count = (requests.len() as u32).min(self.request_cap);
        if count == 0 {
            return 0;
        }
        queue.write_buffer(
            &self.requests,
            0,
            bytemuck::cast_slice(&requests[..count as usize]),
        );
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&FeedbackParams {
                view_proj: view.view_proj().to_cols_array(),
                eye: {
                    let e = view.eye_local();
                    [e.x, e.y, e.z, projection_scale(view)]
                },
                counts: [
                    count,
                    self.layout.words() as u32,
                    VT_FEEDBACK_MAX_TILES,
                    (view.height / 2).max(1),
                ],
            }),
        );
        // The indirection buffer is RE-CREATED on a registration, so a bind group
        // cached across one keeps the old allocation alive and the pass would mark
        // bits against a table from before the new texture existed — the same
        // hazard `passes::ResourceKey`'s `table_generation` component exists for,
        // one pass over.
        if self.bind.as_ref().map(|(g, _)| *g) != Some(table_generation) {
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vt-feedback"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.requests.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.mask.as_entire_binding(),
                    },
                ],
            });
            self.bind = Some((table_generation, bind));
        }
        encoder.clear_buffer(&self.mask, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vt-feedback"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind.as_ref().expect("just built").1, &[]);
            pass.dispatch_workgroups(count.div_ceil(64).max(1), 1, 1);
        }
        self.ring.record(encoder, &self.mask, frame);
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The level rule: one texel per pixel, erring **coarse**, clamped into the
    /// pyramid at both ends.
    ///
    /// This is the function `vt_feedback.wgsl` mirrors, so it is pinned as a
    /// table rather than described — a drift between the two shows up as a floor
    /// and a feedback that disagree about what a frame needs, which reads as
    /// permanent pop-in rather than as a wrong number.
    #[test]
    fn the_justified_level_is_one_texel_per_pixel_rounded_coarse() {
        // A 1024-texel texture on a 1024-pixel surface: mip 0.
        assert_eq!(justified_mip(1024, 1024.0, 11), 0);
        // Half the screen size: one level coarser.
        assert_eq!(justified_mip(1024, 512.0, 11), 1);
        assert_eq!(justified_mip(1024, 256.0, 11), 2);
        // A sliver: clamped at the coarsest level rather than running off.
        assert_eq!(justified_mip(1024, 0.5, 11), 10);
        // Bigger on screen than it is in texels: mip 0, never negative.
        assert_eq!(justified_mip(1024, 4096.0, 11), 0);
        // Erring coarse: 700 pixels of a 1024 texture is level 1, not level 0.
        assert_eq!(justified_mip(1024, 700.0, 11), 1);
        // A one-level pyramid has one answer.
        assert_eq!(justified_mip(1024, 1.0, 1), 0);
        assert_eq!(justified_mip(1024, 1.0, 0), 0);
    }

    /// The projected diameter falls off as 1/distance and scales with the
    /// radius — the two properties the floor's level depends on.
    #[test]
    fn the_projected_diameter_tracks_distance_and_size() {
        let eye = Vec3::ZERO;
        let near = screen_diameter_px(Vec3::new(0.0, 0.0, -5.0), 1.0, eye, 500.0);
        let far = screen_diameter_px(Vec3::new(0.0, 0.0, -10.0), 1.0, eye, 500.0);
        assert!((near - 200.0).abs() < 1e-3, "{near}");
        assert!((far - near * 0.5).abs() < 1e-3, "{far} vs {near}");
        let big = screen_diameter_px(Vec3::new(0.0, 0.0, -5.0), 2.0, eye, 500.0);
        assert!((big - 2.0 * near).abs() < 1e-3);
        // A surface at the eye does not divide by zero.
        assert!(screen_diameter_px(eye, 1.0, eye, 500.0).is_finite());
    }
}
