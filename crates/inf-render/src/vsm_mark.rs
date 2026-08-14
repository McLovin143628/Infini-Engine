//! **The virtual-shadow-map marking loop** (P27.1): the screen-driven page
//! marking pass, the readback ring that reads it at a pinned latency, and the
//! residency step that turns its bits into allocations.
//!
//! One call per frame at the renderer's sync point plus one recording after the
//! opaque geometry, doing four things in a fixed order:
//!
//! 1. read the needed-page mask from **frame F − 2** (or nothing) and decode it
//!    into wants in entry order;
//! 2. apply those wants to the residency and publish the result to the mirror;
//! 3. build this frame's per-(light × face) projection list;
//! 4. after the graph has written depth, record the marking pass over that depth
//!    and hand its buffer to the ring.
//!
//! The order between 2 and 4 is load-bearing and it is `vt_stream`'s: the pass
//! marks against a table the frame has already been given, so a bit and the
//! entry it names describe the same page.
//!
//! # What P27.1 does NOT do
//!
//! **Nothing is rasterized into the atlas.** An admit here is an *allocation*:
//! the slot is published in the indirection table and the page's depth is
//! whatever the atlas last held. P27.2 is the caster pass that fills it, and
//! P27.4 is the receiver that reads it — until then no shader samples the atlas,
//! which is why this whole loop is inert on every scene by default
//! ([`VsmSettings::enabled`](crate::settings::VsmSettings::enabled) is `false`)
//! and why no golden moves.

use inf_vsm::{VsmLightDesc, VsmLightHandle, VsmMarkLayout, VsmResidency, VsmTransaction, VsmWant};

use crate::camera::RenderView;
use crate::gpu::GpuContext;
use crate::readback::ReadbackRing;
use crate::scene::RenderScene;
use crate::settings::VsmSettings;
use crate::vsm::{vsm_light_trees, vsm_projections, VsmMarkParams, VsmProjection};
use crate::vsm_atlas::VsmPools;
use crate::vt_stream::projection_scale;

/// How many (light × face) projections one frame may mark through — the buffer
/// is allocated once at this size, so it is a VRAM number (64 × 96 B = 6 KiB)
/// rather than a quality one. Matches [`crate::vsm::VSM_MAX_PROJECTIONS`], which
/// is what the CPU list is capped at, so the two cannot disagree.
pub const VSM_PROJECTION_CAP: u32 = crate::vsm::VSM_MAX_PROJECTIONS as u32;

/// Instrumentation for the marking loop — the numbers a gate asserts on.
///
/// `admits` is the **anti-vacuity** number: a marking pass that marks nothing
/// and a scene with no shadows produce the same (empty) trace, and only a
/// counter tells them apart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VsmStreamStats {
    /// Frames this loop has run.
    pub frames: u64,
    /// Pages allocated, summed over frames.
    pub admits: u64,
    /// Pages evicted, summed over frames.
    pub evicts: u64,
    /// Wants the residency could not seat, summed over frames.
    pub deferred: u64,
    /// Marked wants offered, summed over frames.
    pub marked_wants: u64,
    /// Frames in which a mask arrived from the ring.
    pub mark_frames: u64,
    /// Frames in which it did not, and nothing new was asked for.
    pub mark_misses: u64,
    /// Projections dispatched, summed over frames — **the second anti-vacuity
    /// number**: a loop that never dispatches a projection also never marks a
    /// page, and both look like a scene with no shadow-casting lights.
    pub projections: u64,
    /// Times a light's clipmap levels moved on the world lattice (P27.3), summed
    /// over frames. **A camera that has not travelled a page does not move one**,
    /// which is the property the whole caching clause rests on — so this is the
    /// counter that says whether the snapping is doing anything, and the one an
    /// arm reads to prove a coarse level held still while a fine one did not.
    pub level_shifts: u64,
    /// **Marking threads dispatched**, summed over frames (P27.5) — the tier
    /// knob's own engagement counter.
    ///
    /// The page set a stride produces is a *containment* claim, and a good
    /// stride loses nothing on a fixture whose casters are metres across — so
    /// the page set alone cannot tell a stride that reached the dispatch from
    /// one that was written down and never read. This can: it is
    /// `workgroups × 64`, and it falls as `1/s²`.
    pub mark_threads: u64,
}

impl VsmStreamStats {
    /// A one-line human summary, in the shape `VtPopIn::summary` and
    /// `VsmStats::summary` already ship so the streamers read alike in one log.
    pub fn summary(&self) -> String {
        format!(
            "vsm marking: {} frames, {} projections, {} admits / {} evicts, {} deferred, \
             {} marked wants, mask {} landed / {} missed, {} level shifts, \
             {} marking threads",
            self.frames,
            self.projections,
            self.admits,
            self.evicts,
            self.deferred,
            self.marked_wants,
            self.mark_frames,
            self.mark_misses,
            self.level_shifts,
            self.mark_threads,
        )
    }
}

/// The marking pass, its needed-page bitmask, and the ring that reads it.
pub struct VsmMarker {
    layout: VsmMarkLayout,
    mask: wgpu::Buffer,
    params: wgpu::Buffer,
    projections: wgpu::Buffer,
    projection_cap: u32,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// `(table generation, depth-target generation)` → the bind group built for
    /// them. **Both** are needed: the table buffer is re-created on a light
    /// registration and the depth view on a resize, and a bind group cached
    /// across either keeps a dead resource alive while `wgpu` validates nothing.
    bind: Option<(u64, u64, wgpu::BindGroup)>,
    ring: ReadbackRing,
}

impl VsmMarker {
    /// A marking pass sized for `layout`.
    pub fn new(device: &wgpu::Device, layout: VsmMarkLayout, projection_cap: u32) -> Self {
        let words = layout.words() as u64;
        let mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm-mark-mask"),
            size: words * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm-mark-params"),
            size: std::mem::size_of::<VsmMarkParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let projections = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm-mark-projections"),
            size: (projection_cap.max(1) as u64) * std::mem::size_of::<VsmProjection>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vsm-mark"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vsm_mark.wgsl").into()),
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
            label: Some("vsm-mark"),
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
                // The live MSAA scene depth: multisampled and unfilterable, it is
                // `textureLoad`ed at an integer texel and never sampled — the
                // same binding shape `passes::cloud` already uses for it.
                entry(
                    4,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                ),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vsm-mark"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vsm-mark"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_mark"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            mask,
            params,
            projections,
            projection_cap: projection_cap.max(1),
            pipeline,
            bgl,
            bind: None,
            ring: ReadbackRing::new(device, "vsm-mark", words * 4),
        }
    }

    /// The bitmask layout this pass writes.
    #[inline]
    pub fn layout(&self) -> &VsmMarkLayout {
        &self.layout
    }

    /// The ring, for a host or a gate that wants its hit/miss counts.
    #[inline]
    pub fn ring(&self) -> &ReadbackRing {
        &self.ring
    }

    /// **The projection list on the GPU** — the buffer the marking pass reads
    /// and, since P27.4, the buffer the *receiver* reads out of the shared
    /// environment bind group.
    ///
    /// One derivation, three readers (`vsm_projections` on the CPU, this pass,
    /// and every lit fragment), which is what makes "the page a fragment samples
    /// is the page its own depth marked" a property of the code rather than of
    /// two matrices agreeing.
    #[inline]
    pub fn projection_buffer(&self) -> &wgpu::Buffer {
        &self.projections
    }

    /// **Upload this frame's projections**, at the frame's SYNC POINT.
    ///
    /// It used to happen inside [`record`](Self::record), which runs *after* the
    /// graph — fine while the only consumer was the marking pass, and wrong the
    /// moment a lit pass reads the same buffer: the receiver would have sampled
    /// through the **previous** frame's matrices, one frame stale on top of the
    /// marking ring's pinned two. Staged here, `write_buffer` executes before the
    /// commands of the frame's encoder, so the receiver, the caster raster and
    /// the marker all describe one frame.
    ///
    /// Returns the number of projections written (the list is capped).
    pub fn upload_projections(
        &mut self,
        queue: &wgpu::Queue,
        projections: &[VsmProjection],
    ) -> u32 {
        let count = (projections.len() as u32).min(self.projection_cap);
        if count == 0 {
            return 0;
        }
        queue.write_buffer(
            &self.projections,
            0,
            bytemuck::cast_slice(&projections[..count as usize]),
        );
        count
    }

    /// **Read frame `frame`'s mask** — the one recorded at `frame − 2`, or
    /// `None`. Never an adjacent frame: see [`crate::readback`].
    pub fn take_wants(
        &mut self,
        device: &wgpu::Device,
        residency: &VsmResidency,
        frame: u64,
    ) -> Option<Vec<VsmWant>> {
        self.ring.poll(device);
        let layout = self.layout.clone();
        self.ring.take(frame, |bytes| {
            let words: &[u32] = bytemuck::cast_slice(bytes);
            layout.wants(residency, words)
        })
    }

    /// **Record the marking pass** into `encoder` and hand its buffer to the ring.
    ///
    /// Call **after** the opaque geometry has written `depth` in this same
    /// encoder, and before the frame's submit. Returns the number of projections
    /// dispatched (0 = nothing recorded, and the ring gets no copy, so the read
    /// two frames later misses — the same degradation as a late mask).
    ///
    /// The mask is cleared first, in the same encoder, so a frame's coverage is
    /// this frame's and never an OR with the last one — which would make the
    /// signal depend on the frame history, the exact property the pinned ring
    /// exists to avoid.
    ///
    /// `inv_view_proj` is passed rather than taken from `view` because the two
    /// are **not the same matrix when TAA is on**: the frame's depth was
    /// rasterized with the sub-pixel-jittered projection, and reconstructing a
    /// world position from it with the unjittered inverse puts every point up to
    /// half a pixel out. The error is small at page granularity and it is still
    /// wrong — a reconstruction has to use the matrix its depth was drawn with,
    /// or a page-allocation trace acquires a dependence on the Halton cursor.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        depth_generation: u64,
        table: &wgpu::Buffer,
        table_generation: u64,
        view: &RenderView,
        inv_view_proj: glam::Mat4,
        projections: &[VsmProjection],
        frame: u64,
        stride: u32,
    ) -> (u32, u64) {
        let count = (projections.len() as u32).min(self.projection_cap);
        let (w, h) = (view.width.max(1), view.height.max(1));
        let stride = stride.max(1);
        if count == 0 {
            return (0, 0);
        }
        // The projection BYTES are not written here: `upload_projections` staged
        // them at the frame's sync point, because the receiver reads the same
        // buffer from inside the graph and this recording happens after it.
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&VsmMarkParams {
                inv_view_proj: inv_view_proj.to_cols_array(),
                eye: {
                    let e = view.eye_local();
                    [e.x, e.y, e.z, projection_scale(view)]
                },
                counts: [count, self.layout.words() as u32, w, h],
                stride: [stride, 0, 0, 0],
            }),
        );
        if self.bind.as_ref().map(|(t, d, _)| (*t, *d))
            != Some((table_generation, depth_generation))
        {
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vsm-mark"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.projections.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.mask.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(depth),
                    },
                ],
            });
            self.bind = Some((table_generation, depth_generation, bind));
        }
        encoder.clear_buffer(&self.mask, 0, None);
        let (groups_x, groups_y) = (w.div_ceil(8 * stride), h.div_ceil(8 * stride));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vsm-mark"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind.as_ref().expect("just built").2, &[]);
            // **The stride is in the dispatch, not only in the shader** (P27.5):
            // one thread per `stride × stride` block, so the tier knob buys
            // threads back rather than making them return early. At `stride = 1`
            // this is `w.div_ceil(8)`, character for character what P27.1
            // dispatched.
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.ring.record(encoder, &self.mask, frame);
        (count, u64::from(groups_x) * u64::from(groups_y) * 64)
    }
}

/// Everything one level's virtual shadow maps need: the residency, its GPU
/// mirror, the mask layout and the marking pass.
///
/// Built from the scene's **shadow-casting light set** rather than handed over by
/// a host, because that set is scene state that changes as lights are added,
/// removed and toggled — unlike a virtual-texture registry, which is level load
/// state. [`signature`](Self::signature) is what decides when it has to be built
/// again.
pub struct VsmSystem {
    residency: VsmResidency,
    pools: VsmPools,
    marker: VsmMarker,
    trees: Vec<VsmLightDesc>,
    /// Table word offset per light, in handle order.
    blocks: Vec<u32>,
    /// First mask bit per light, in handle order.
    bases: Vec<u32>,
    /// This frame's projection list, built at the sync point and consumed by the
    /// recording after the graph — one derivation, two readers.
    projections: Vec<VsmProjection>,
    /// This frame's per-light clipmap layout (P27.3): where each level's grid sits
    /// on the world lattice, and the NDC offset that says so. One entry per
    /// **light**, unlike [`projections`](Self::projections), which is per face.
    layouts: Vec<crate::vsm::ClipmapLayout>,
    /// Index of each light's **first** projection in [`projections`](Self::projections),
    /// in handle order. A cube light owns six consecutive entries, so a page's
    /// face indexes off this base rather than off its handle — the one place the
    /// (light × face) list is turned back into a per-light one.
    proj_base: Vec<u32>,
    /// The P27.2 caster pass.
    raster: crate::vsm_raster::VsmRaster,
    signature: VsmSignature,
    stats: VsmStreamStats,
}

/// What a [`VsmSystem`] is a function of. Rebuilt when this changes and not
/// otherwise — a light moving is not a new system, a light *appearing* is.
#[derive(Debug, Clone, PartialEq)]
struct VsmSignature {
    trees: crate::vsm::VsmTreeSet,
    budget_bytes: u64,
}

impl VsmSystem {
    /// Build the system for `scene`'s shadow-casting lights, or `None` when
    /// there are none — which is every scene by default and every golden.
    pub fn for_scene(
        gpu: &GpuContext,
        scene: &RenderScene,
        settings: &VsmSettings,
    ) -> Option<Self> {
        let asked = vsm_light_trees(scene, settings);
        // **Never a silent cap** (P27.2's doctrine, extended to the ceiling the
        // P27.4 audit found). Logged once at construction rather than per frame,
        // because a system is rebuilt when its tree list changes and this number
        // is part of that list's identity.
        if asked.refused_past_shader_ceiling > 0 {
            tracing::warn!(
                "inf-render: {} shadow-casting light(s) sit past scene index {} — \
                 the lights uniform's array — so they are not shaded at all and \
                 get no page tree",
                asked.refused_past_shader_ceiling,
                crate::passes::mesh::MAX_LIGHTS,
            );
        }
        if asked.refused_past_projection_cap > 0 {
            tracing::warn!(
                "inf-render: {} shadow-casting light(s) did not fit the {} \
                 marking projections a frame may hold; they keep the cascaded \
                 shadow map",
                asked.refused_past_projection_cap,
                crate::vsm::VSM_MAX_PROJECTIONS,
            );
        }
        let trees = asked.trees.clone();
        if trees.is_empty() {
            return None;
        }
        let (mut residency, advisories) = VsmResidency::new(inf_vsm::VsmAtlasConfig {
            budget_bytes: settings.budget_bytes,
            ..Default::default()
        });
        for a in &advisories {
            // Reported, not swallowed — the P18.2 streaming-report discipline. A
            // silently empty atlas reads as "the shadows are just blurry".
            tracing::warn!("inf-render: virtual shadow atlas: {a}");
        }
        let mut blocks = Vec::with_capacity(trees.len());
        let mut kept: Vec<VsmLightDesc> = Vec::with_capacity(trees.len());
        for tree in &trees {
            // A refusal TRUNCATES rather than skips, for `vsm_light_trees`'s
            // reason and it is the same invariant: handle `n` is the `n`-th
            // shadow-casting light in scene order, and `vsm_projections` rebuilds
            // that by re-walking `scene.lights`. Dropping a light out of the
            // middle would hand every light after it another light's table block
            // and bit range — silently.
            match residency.register_light(tree.clone()) {
                Ok(_) => kept.push(tree.clone()),
                Err(e) => {
                    tracing::warn!(
                        "inf-render: shadow light {} was refused ({e}); it and every \
                         shadow-caster after it keep the cascaded shadow map",
                        kept.len()
                    );
                    break;
                }
            }
        }
        if kept.is_empty() {
            return None;
        }
        for l in 0..kept.len() {
            let (base, _) = residency
                .table_block(VsmLightHandle(l as u32))
                .expect("registered");
            blocks.push(base as u32);
        }
        let layout = VsmMarkLayout::for_residency(&residency);
        let bases: Vec<u32> = (0..kept.len())
            .map(|l| {
                layout
                    .light_base(VsmLightHandle(l as u32))
                    .expect("registered")
            })
            .collect();
        let pools = VsmPools::new(&gpu.device, &gpu.queue, &residency);
        let marker = VsmMarker::new(&gpu.device, layout, VSM_PROJECTION_CAP);
        // Where each light's projections begin. `vsm_projections` emits one entry
        // per face in tree order, so this is a running sum of `faces()` and it is
        // built ONCE — it is a function of the tree list, which is exactly what
        // `signature` says this system is rebuilt on.
        let mut proj_base = Vec::with_capacity(kept.len());
        let mut base = 0u32;
        for tree in &kept {
            proj_base.push(base);
            base += tree.faces();
        }
        Some(Self {
            signature: VsmSignature {
                // The list `vsm_light_trees` produced, **not** the truncated one:
                // `matches` compares against that function's output, so storing
                // the truncation here would make a system with a refused light
                // rebuild itself — and re-allocate an atlas — every frame.
                trees: asked,
                budget_bytes: settings.budget_bytes,
            },
            residency,
            pools,
            marker,
            trees: kept,
            blocks,
            bases,
            projections: Vec::new(),
            layouts: Vec::new(),
            proj_base,
            raster: crate::vsm_raster::VsmRaster::new(gpu),
            stats: VsmStreamStats::default(),
        })
    }

    /// Whether this system still describes `scene` under `settings`.
    pub fn matches(&self, scene: &RenderScene, settings: &VsmSettings) -> bool {
        self.signature.budget_bytes == settings.budget_bytes
            && self.signature.trees == vsm_light_trees(scene, settings)
    }

    /// The live residency, for a host or a gate that wants to assert the WORLD
    /// (the resident page set) rather than a report.
    #[inline]
    pub fn residency(&self) -> &VsmResidency {
        &self.residency
    }
    #[inline]
    pub fn pools(&self) -> &VsmPools {
        &self.pools
    }
    #[inline]
    pub fn marker(&self) -> &VsmMarker {
        &self.marker
    }
    #[inline]
    pub fn stats(&self) -> VsmStreamStats {
        self.stats
    }
    /// The trees this system registered, in handle order.
    #[inline]
    pub fn trees(&self) -> &[VsmLightDesc] {
        &self.trees
    }

    /// **What the two ceilings refused** (P27.5) — the shader ceiling and the
    /// projection cap, as counts rather than as a log line a test cannot read.
    ///
    /// This is what makes *"every rasterized page is sampleable"* an assertion:
    /// a tree exists here only for a light some lit shader can shade, so a page
    /// belonging to it has a slot that reaches a fragment.
    #[inline]
    pub fn tree_refusals(&self) -> &crate::vsm::VsmTreeSet {
        &self.signature.trees
    }
    /// The projection list the last [`sync`](Self::sync) built.
    #[inline]
    pub fn projections(&self) -> &[VsmProjection] {
        &self.projections
    }

    /// The per-light clipmap layouts the last [`sync`](Self::sync) built (P27.3).
    #[inline]
    pub fn layouts(&self) -> &[crate::vsm::ClipmapLayout] {
        &self.layouts
    }

    /// Index of each light's first projection, in handle order (P27.4's door
    /// onto what was a private field). A cube light owns six consecutive
    /// entries.
    #[inline]
    pub fn proj_base(&self) -> &[u32] {
        &self.proj_base
    }

    /// **Which projection each of `scene`'s lights reads, + 1** — the mapping
    /// `GpuLight::params.w` carries (P27.4).
    ///
    /// Re-walks `scene.lights` exactly as [`vsm_projections`] does, through the
    /// one rule in [`crate::vsm_receiver::receiver_slots`], so the lights uniform
    /// and this system cannot disagree about which tree a light owns — including
    /// about the tail of a **truncated** light list, where every light past the
    /// refusal must have *no* slot rather than another light's.
    pub fn receiver_slots(&self, scene: &RenderScene) -> Vec<u32> {
        crate::vsm_receiver::receiver_slots(
            scene.lights.iter().map(|l| l.cast_shadows),
            self.trees.len(),
            &self.proj_base,
        )
    }

    /// **The page matrix of one resident page**, through the shipped door — the
    /// light's own projection, its level's snapped offset, and the page's
    /// sub-rectangle.
    ///
    /// A door rather than a convenience: the offset is per level and per frame, so
    /// a caller that composed `vsm_page_matrix` itself out of `projections()[0]`
    /// would be reproducing the layout rather than reading it — and P27.4's
    /// receiver has to ask exactly this question.
    pub fn page_matrix(&self, light: VsmLightHandle, page: inf_vsm::VsmPage) -> Option<glam::Mat4> {
        let desc = self.residency.desc(light)?;
        let g = desc.levels.get(page.level as usize)?;
        let base = *self.proj_base.get(light.index())?;
        let proj = self.projections.get((base + page.face) as usize)?;
        let offset = self
            .layouts
            .get(light.index())
            .map_or([0.0, 0.0], |l| l.offset(page.level));
        Some(crate::vsm::vsm_page_matrix(
            glam::Mat4::from_cols_array(&proj.view_proj),
            desc.kind,
            page.level,
            g.pages_x,
            g.pages_y,
            page.x,
            page.y,
            offset,
        ))
    }

    /// **The sync point** (steps 1–3): build this frame's projections, re-point
    /// the clipmap levels they snapped, read frame `F − 2`'s mask, allocate what
    /// it asked for and publish the table.
    ///
    /// The projections come **first** since P27.3: each clipmap level snaps to its
    /// own page stride, so the projection list is what decides where the levels
    /// sit, and the residency's parent rule is a function of that. Applying wants
    /// against last frame's offsets would propagate a fallback chain the frame
    /// does not have.
    pub fn sync(
        &mut self,
        gpu: &GpuContext,
        scene: &RenderScene,
        view: &RenderView,
        settings: &VsmSettings,
        frame: u64,
    ) -> VsmTransaction {
        let (projections, layouts) = vsm_projections(
            scene,
            view,
            settings,
            &self.trees,
            &self.blocks,
            &self.bases,
        );
        self.projections = projections;
        self.layouts = layouts;
        // On the GPU **now**, not after the graph — see
        // `VsmMarker::upload_projections`.
        self.marker
            .upload_projections(&gpu.queue, &self.projections);
        // A light whose levels moved has a changed indirection block even when no
        // page was admitted, because the fallback chain was recomputed against the
        // new parent rule. Merged into the transaction's own list rather than
        // written separately, so the mirror still applies **one** transaction.
        let mut moved: Vec<VsmLightHandle> = Vec::new();
        for (i, layout) in self.layouts.iter().enumerate() {
            let h = VsmLightHandle(i as u32);
            if self.residency.set_clip_origins(h, &layout.clip_origins) {
                moved.push(h);
            }
        }
        self.stats.level_shifts += moved.len() as u64;

        let wants = match self.marker.take_wants(&gpu.device, &self.residency, frame) {
            Some(w) => {
                self.stats.mark_frames += 1;
                w
            }
            None => {
                self.stats.mark_misses += 1;
                Vec::new()
            }
        };
        self.stats.marked_wants += wants.len() as u64;
        let mut txn = self.residency.apply_wants(&wants);
        if !moved.is_empty() {
            txn.tables.extend(moved);
            txn.tables.sort_unstable();
            txn.tables.dedup();
        }
        self.pools
            .apply(&gpu.device, &gpu.queue, &self.residency, &txn);
        self.stats.admits += txn.admits.len() as u64;
        self.stats.evicts += txn.evicts.len() as u64;
        self.stats.deferred += u64::from(txn.deferred);
        self.stats.frames += 1;
        txn
    }

    /// The caster pass's engagement counters (P27.2).
    #[inline]
    pub fn raster_stats(&self) -> crate::vsm_raster::VsmRasterStats {
        self.raster.stats()
    }

    /// The caster pass itself — for a gate that wants the **cull's own verdict**
    /// (`read_draw_counts`) rather than a counter over it.
    #[inline]
    pub fn raster_state(&self) -> &crate::vsm_raster::VsmRaster {
        &self.raster
    }

    /// **Step 3b (P27.2)**: rasterize this frame's resident pages.
    ///
    /// Recorded **before** the graph, not after it like the marking pass, and the
    /// two placements answer different questions. The marker consumes the frame's
    /// depth buffer, so it has to follow every pass that writes it. The raster
    /// produces the atlas P27.4's receivers will sample from *inside* the graph, so
    /// it has to precede them — putting it after would hand the lit passes an atlas
    /// one frame stale on top of the marking ring's pinned two, and P27.4 would
    /// have to move it as its first act.
    ///
    /// Returns the number of page rectangles rasterized; 0 means the encoder was
    /// not touched.
    pub fn raster(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        scene: &RenderScene,
        view: &RenderView,
        settings: &crate::settings::RenderSettings,
    ) -> u32 {
        self.raster.record(
            gpu,
            encoder,
            &self.residency,
            crate::vsm_raster::PageGeometry {
                projections: &self.projections,
                proj_base: &self.proj_base,
                layouts: &self.layouts,
            },
            self.pools.atlas_view(),
            scene,
            view,
            settings,
        )
    }

    /// **Throw the page cache away** — a gate's door (P27.3), on
    /// `read_draw_counts`'s precedent.
    ///
    /// Nothing in the shipping path calls it: an invalidation is supposed to be a
    /// consequence of a content stamp moving, and a flush is the blunt instrument
    /// that proves the fine one is honest. Two arms need it — "a cached page's
    /// texels are byte-for-byte what a fresh raster produces" has to be able to
    /// *force* the fresh raster, and "two routes to one residency make the same
    /// decisions" has to be able to build the second route.
    pub fn flush_page_cache(&mut self) {
        self.raster.flush_cache();
    }

    /// **Step 4**: record the marking pass over `depth`, which the graph has
    /// already written this frame. `inv_view_proj` must invert the projection
    /// that depth was **rasterized** with — the jittered one when TAA is on; see
    /// [`VsmMarker::record`].
    #[allow(clippy::too_many_arguments)]
    pub fn mark(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        depth_generation: u64,
        view: &RenderView,
        inv_view_proj: glam::Mat4,
        frame: u64,
        stride: u32,
    ) -> u32 {
        let table_generation = self.pools.table_generation();
        let (n, threads) = self.marker.record(
            &gpu.device,
            &gpu.queue,
            encoder,
            depth,
            depth_generation,
            self.pools.table(),
            table_generation,
            view,
            inv_view_proj,
            &self.projections,
            frame,
            stride,
        );
        self.stats.projections += u64::from(n);
        self.stats.mark_threads += threads;
        n
    }

    /// The one line a host logs about virtual shadow maps — the residency's half
    /// (what is in the atlas right now) and the loop's half (what this session
    /// has been through), because a full atlas and a thrashing loop look
    /// identical in either one alone.
    pub fn summary(&self) -> String {
        format!(
            "{}; {}; {}",
            self.residency.stats().summary(),
            self.stats.summary(),
            self.raster.stats().summary()
        )
    }
}
