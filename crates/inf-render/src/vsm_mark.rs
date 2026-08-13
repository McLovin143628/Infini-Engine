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
}

impl VsmStreamStats {
    /// A one-line human summary, in the shape `VtPopIn::summary` and
    /// `VsmStats::summary` already ship so the streamers read alike in one log.
    pub fn summary(&self) -> String {
        format!(
            "vsm marking: {} frames, {} projections, {} admits / {} evicts, {} deferred, \
             {} marked wants, mask {} landed / {} missed",
            self.frames,
            self.projections,
            self.admits,
            self.evicts,
            self.deferred,
            self.marked_wants,
            self.mark_frames,
            self.mark_misses,
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
        projections: &[VsmProjection],
        frame: u64,
    ) -> u32 {
        let count = (projections.len() as u32).min(self.projection_cap);
        let (w, h) = (view.width.max(1), view.height.max(1));
        if count == 0 {
            return 0;
        }
        queue.write_buffer(
            &self.projections,
            0,
            bytemuck::cast_slice(&projections[..count as usize]),
        );
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&VsmMarkParams {
                inv_view_proj: view.view_proj().inverse().to_cols_array(),
                eye: {
                    let e = view.eye_local();
                    [e.x, e.y, e.z, projection_scale(view)]
                },
                counts: [count, self.layout.words() as u32, w, h],
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
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vsm-mark"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind.as_ref().expect("just built").2, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        self.ring.record(encoder, &self.mask, frame);
        count
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
    signature: VsmSignature,
    stats: VsmStreamStats,
}

/// What a [`VsmSystem`] is a function of. Rebuilt when this changes and not
/// otherwise — a light moving is not a new system, a light *appearing* is.
#[derive(Debug, Clone, PartialEq)]
struct VsmSignature {
    trees: Vec<VsmLightDesc>,
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
        let trees = vsm_light_trees(scene, settings);
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
        Some(Self {
            signature: VsmSignature {
                // The list `vsm_light_trees` produced, **not** the truncated one:
                // `matches` compares against that function's output, so storing
                // the truncation here would make a system with a refused light
                // rebuild itself — and re-allocate an atlas — every frame.
                trees,
                budget_bytes: settings.budget_bytes,
            },
            residency,
            pools,
            marker,
            trees: kept,
            blocks,
            bases,
            projections: Vec::new(),
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
    /// The projection list the last [`sync`](Self::sync) built.
    #[inline]
    pub fn projections(&self) -> &[VsmProjection] {
        &self.projections
    }

    /// **The sync point** (steps 1–3): read frame `F − 2`'s mask, allocate what
    /// it asked for, publish the table, and build this frame's projections.
    pub fn sync(
        &mut self,
        gpu: &GpuContext,
        scene: &RenderScene,
        view: &RenderView,
        settings: &VsmSettings,
        frame: u64,
    ) -> VsmTransaction {
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
        let txn = self.residency.apply_wants(&wants);
        self.pools
            .apply(&gpu.device, &gpu.queue, &self.residency, &txn);
        self.stats.admits += txn.admits.len() as u64;
        self.stats.evicts += txn.evicts.len() as u64;
        self.stats.deferred += u64::from(txn.deferred);
        self.stats.frames += 1;
        self.projections = vsm_projections(
            scene,
            view,
            settings,
            &self.trees,
            &self.blocks,
            &self.bases,
        );
        txn
    }

    /// **Step 4**: record the marking pass over `depth`, which the graph has
    /// already written this frame.
    pub fn mark(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        depth_generation: u64,
        view: &RenderView,
        frame: u64,
    ) -> u32 {
        let table_generation = self.pools.table_generation();
        let n = self.marker.record(
            &gpu.device,
            &gpu.queue,
            encoder,
            depth,
            depth_generation,
            self.pools.table(),
            table_generation,
            view,
            &self.projections,
            frame,
        );
        self.stats.projections += u64::from(n);
        n
    }

    /// The one line a host logs about virtual shadow maps — the residency's half
    /// (what is in the atlas right now) and the loop's half (what this session
    /// has been through), because a full atlas and a thrashing loop look
    /// identical in either one alone.
    pub fn summary(&self) -> String {
        format!(
            "{}; {}",
            self.residency.stats().summary(),
            self.stats.summary()
        )
    }
}
