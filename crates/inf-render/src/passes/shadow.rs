//! Cascaded shadow-map pass (P13.3b): renders the first directional light's three
//! view-frustum cascades into a `Depth32Float` array, and publishes the per-cascade
//! matrices + bias constants in the shared [`ShadowResources`] uniform the lit
//! passes sample (see [`crate::passes::EnvBinding`] + `shaders/mesh.wgsl`).
//!
//! The cascade split scheme, sphere fit, and texel snapping are pure functions in
//! [`crate::csm`]. This node owns the caster pipeline (a depth-only forward-Z
//! render, mirroring [`crate::passes::depth_prepass`]) and re-packs its own cube
//! instance buffer, so it is self-contained.
//!
//! ## Scope (v1, documented)
//!
//! * **Casters:** rigid [`MeshInstance`] geometry only (the golden is boxes on a
//!   plane). Terrain-patch + skinned casters are a follow-up.
//! * **Receivers:** mesh / skinned / terrain all *sample* the cascades (shadows
//!   land on all of them); only casting is scoped.
//! * **Off path:** when [`ShadowSettings::enabled`] is false the node still writes
//!   the shared uniform (with `enabled = 0`) so receivers read a valid flag, then
//!   renders nothing — receivers take the byte-stable un-shadowed instruction path.

use crate::camera::DEPTH_FORMAT;
use crate::csm::{
    bounding_sphere, cascade_matrix, cascade_splits, frustum_slice_corners, SHADOW_CASCADES,
    SHADOW_RESOLUTION,
};
use std::ops::Range;

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{pack_bucketed, vertex_layouts, InstanceRaw, EMPTY_RANGES};
use crate::primitives::PrimGpu;
use crate::renderer::FrameData;
use crate::scene::LightKind;

/// The scatter half of the caster-pack key (P18.5): the re-pack eye bucket, the
/// shadow range's bits, the scatter band stamp, the batches' content fold and
/// **the floating origin the pack was packed against**. Present only while the
/// scene carries scatter — these are inputs to the *scatter* caster pack and to
/// nothing else, so folding them in unconditionally would re-pack every rigid
/// caster in a foliage-free level each time the camera crossed an 8 m lattice
/// cell.
///
/// Widened by island wave I4b with the batches' own **content fold**, which is
/// what lets the pack be *cached*: before it, the key leaned on
/// `RenderScene::version` to notice a changed batch, and a version that moves
/// with every pose re-packed eleven thousand casters a frame.
///
/// # …and the origin, because the pack is RENDER-LOCAL (the I4b audit)
///
/// [`pack_fallback`](super::scatter::pack_fallback) turns each instance's world
/// position into a **render-local model matrix** through the
/// [`FloatingOrigin`](inf_math::FloatingOrigin) it is handed, so the packed bytes
/// are a function of the origin as well as of the content. I4b's cache keyed on
/// everything *except* that, and the two lattices do not line up: an origin
/// rebase moves the origin by `REBASE_DISTANCE` (1 024 m) while
/// [`eye_bucket`](super::scatter::eye_bucket) quantizes the **world** eye onto an
/// 8 m lattice, so the overwhelmingly likely frame in which a rebase fires is one
/// where the bucket did *not* tick — and the cached pack was then re-merged and
/// re-uploaded with every scatter caster a kilometre out of place, until the
/// camera happened to leave its bucket. The whole-key `CasterKey` noticed the
/// rebase and re-uploaded; what it re-uploaded was stale.
///
/// The bits rather than the `DVec3` so the key stays `Eq`, and because an origin
/// is snapped to `ORIGIN_SNAP` and can never legitimately be `NaN` — a door that
/// compares by bits refuses one instead of matching everything against it.
type ScatterCasterKey = ([i64; 3], u32, u64, u128, [u64; 3]);

/// What the packed caster buffer is a function of: the **packed rigid bytes**,
/// their per-kind ranges, the floating origin, and the scatter terms if any.
///
/// The first term used to be `scene.version`, which is a *guess* at the rigid
/// caster content and moves for reasons that have nothing to do with it. Hashing
/// the bytes the pack just produced is exact, costs `O(scene.instances)`, and is
/// zero on a level whose casters are all scatter — see [`ShadowNode::sync`].
type CasterKey = (u128, [Range<u32>; 5], glam::DVec3, Option<ScatterCasterKey>);

/// **Everything the cached scatter caster pack is a function of**, in one place
/// (the I4b audit).
///
/// A free function rather than a closure inside
/// [`ShadowNode::sync`](ShadowNode::sync) so the question "does this key move
/// when the pack would" is one a unit test can ask without a GPU, a `FrameData`
/// or a surface — which is what
/// `the_scatter_caster_key_moves_when_the_pack_it_caches_would` does, and what
/// nothing in the tree could do while the derivation lived inside `sync`.
///
/// `None` on a scatter-free scene: see [`ScatterCasterKey`] for why the terms
/// stay out of the key entirely rather than entering it as zeroes.
fn scatter_caster_key(
    batches: &[crate::scene::ScatterBatch],
    origin: &inf_math::FloatingOrigin,
    scatter_eye: glam::DVec3,
    max_distance: f32,
    caster_stamp: u64,
) -> Option<ScatterCasterKey> {
    (!batches.is_empty()).then(|| {
        let o = origin.origin();
        (
            super::scatter::eye_bucket(scatter_eye),
            max_distance.to_bits(),
            caster_stamp,
            super::scatter::scatter_caster_fold(batches),
            [o.x.to_bits(), o.y.to_bits(), o.z.to_bits()],
        )
    })
}

/// Forward-Z shadow depth: nearest caster wins (clear to 1.0 = far, keep smaller).
///
/// **Public since P27.1, and unchanged by it.** The virtual-shadow-map ruling
/// (`crate::vsm`) adopts the camera's reverse-Z for its pages and asserts that
/// this path did NOT move with it — the CSM stays exactly as it is until P27.5
/// demotes it, and a constant nothing outside this module could read would make
/// that a comment rather than a check.
pub const SHADOW_DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::LessEqual;
/// See [`SHADOW_DEPTH_COMPARE`].
pub const SHADOW_DEPTH_CLEAR: f32 = 1.0;

/// The shared shadow uniform block (`std140`), written by [`ShadowNode`] and read
/// by every lit pass through [`crate::passes::EnvBinding`]. Mirrors `struct
/// ShadowData` in the receiver shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowDataGpu {
    /// Per-cascade forward-Z light `view_proj`.
    pub cascade_vp: [[f32; 16]; SHADOW_CASCADES],
    /// Cascade far distances (x,y,z); **w = the P18.4 cascade blend fraction**
    /// ([`crate::ShadowSettings::cascade_blend`]) — the slot that was spare.
    pub splits: [f32; 4],
    /// Per-cascade world texel size (x,y,z); w unused (drives normal-offset bias).
    pub texel_world: [f32; 4],
    /// x = enabled, y = depth_bias (NDC), z = normal_bias (texels),
    /// w = cascade count.
    pub params: [f32; 4],
}

impl ShadowDataGpu {
    fn disabled() -> Self {
        Self {
            cascade_vp: [[0.0; 16]; SHADOW_CASCADES],
            splits: [0.0; 4],
            texel_world: [0.0; 4],
            params: [0.0, 0.0, 0.0, SHADOW_CASCADES as f32],
        }
    }
}

/// Renderer-owned shadow GPU resources, created once (independent of viewport size)
/// and shared with the lit passes via [`FrameData`]. The [`ShadowNode`] renders the
/// cascades into [`layer_views`](Self::layer_views) and writes [`uniform`](Self::uniform);
/// receivers sample [`array_view`](Self::array_view) + read the uniform.
pub struct ShadowResources {
    _map: wgpu::Texture,
    /// Full depth array view (receiver sampling).
    pub array_view: wgpu::TextureView,
    /// Per-cascade single-layer views (caster rendering).
    pub layer_views: Vec<wgpu::TextureView>,
    /// Shared `ShadowData` uniform (written by the node, read by receivers).
    pub uniform: wgpu::Buffer,
    /// Stable generation for receiver bind-group caching (resources never resize).
    pub generation: u64,
}

impl ShadowResources {
    pub fn new(gpu: &GpuContext) -> Self {
        let map = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-map"),
            size: wgpu::Extent3d {
                width: SHADOW_RESOLUTION,
                height: SHADOW_RESOLUTION,
                depth_or_array_layers: SHADOW_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = map.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow-map-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views = (0..SHADOW_CASCADES as u32)
            .map(|layer| {
                map.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow-cascade"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-data"),
            size: std::mem::size_of::<ShadowDataGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed with the disabled state so a receiver bound before the first
        // ShadowNode run still reads enabled = 0.
        gpu.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&ShadowDataGpu::disabled()));
        Self {
            _map: map,
            array_view,
            layer_views,
            uniform,
            generation: 0,
        }
    }
}

/// Per-cascade caster uniform (`struct Cascade` in `shadow_depth.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CascadeGpu {
    view_proj: [f32; 16],
}

pub struct ShadowNode {
    pipeline: wgpu::RenderPipeline,
    /// R-P5 masked caster variant (`vs_masked` + `fs_masked`, alpha-test discard)
    /// so a masked object casts a cut-out shadow. Used only when the scene carries
    /// masked instances; otherwise the fragment-less `pipeline` draws
    /// (byte-identical to the pre-R-P5 shadow path).
    pipeline_masked: wgpu::RenderPipeline,
    prim: PrimGpu,
    /// One tiny uniform + bind group per cascade (distinct buffers so the
    /// per-cascade writes don't collide before the passes run).
    cascade_bufs: Vec<wgpu::Buffer>,
    cascade_bgs: Vec<wgpu::BindGroup>,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    instance_count: u32,
    ranges: [Range<u32>; 5],
    /// See [`CasterKey`]. `None` until the first pack.
    uploaded_version: Option<CasterKey>,
    /// The cached P18.5 scatter caster pack, and the key it was built for
    /// (island wave I4b). Held so a frame in which only the rigid casters moved
    /// re-merges rather than re-packs; `None` on every scatter-free level, where
    /// it costs one `Option` and no allocation.
    scatter_pack: Option<(Vec<InstanceRaw>, [Range<u32>; 5])>,
    scatter_key: Option<ScatterCasterKey>,
    /// Whether the disabled-shadows uniform is already published to the (stable,
    /// created-once) `frame.shadow.uniform`. Gates the constant re-write while
    /// shadows stay off; the enabled path clears it so a later disable re-publishes.
    published_disabled: bool,
}

impl ShadowNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadow-depth"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/shadow_depth.wgsl").into(),
                ),
            });

        let cascade_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow-cascade"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let prim = PrimGpu::new(gpu, "shadow");

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow-depth"),
                bind_group_layouts: &[Some(&cascade_bgl)],
                immediate_size: 0,
            });
        let primitive = wgpu::PrimitiveState {
            // Front-face culling reduces peter-panning/acne on the classic box
            // casters (shadow depth from back faces).
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        };
        let depth_stencil = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(SHADOW_DEPTH_COMPARE),
            stencil: Default::default(),
            // A slope-scaled hardware depth bias further reduces acne.
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        };
        let make = |label: &str, vs: &str, fs: Option<wgpu::FragmentState>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vs),
                        compilation_options: Default::default(),
                        buffers: &vertex_layouts(),
                    },
                    fragment: fs,
                    primitive,
                    depth_stencil: Some(depth_stencil.clone()),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let pipeline = make("shadow-depth", "vs", None);
        let pipeline_masked = make(
            "shadow-depth-masked",
            "vs_masked",
            Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_masked"),
                compilation_options: Default::default(),
                targets: &[], // depth-only discard; no colour target
            }),
        );

        let cascade_bufs: Vec<wgpu::Buffer> = (0..SHADOW_CASCADES)
            .map(|c| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("shadow-cascade-{c}")),
                    size: std::mem::size_of::<CascadeGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let cascade_bgs = cascade_bufs
            .iter()
            .map(|buf| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow-cascade"),
                    layout: &cascade_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
            })
            .collect();

        Self {
            pipeline,
            pipeline_masked,
            prim,
            cascade_bufs,
            cascade_bgs,
            instances: None,
            instance_capacity: 0,
            instance_count: 0,
            ranges: EMPTY_RANGES,
            uploaded_version: None,
            scatter_pack: None,
            scatter_key: None,
            published_disabled: false,
        }
    }

    /// Re-pack the caster instance buffer when the caster CONTENT changed or the
    /// origin rebased (mirrors [`crate::passes::depth_prepass`]).
    ///
    /// # The key is the content, not the scene's version (island wave I4b)
    ///
    /// This used to be keyed on `RenderScene::version`, which moves whenever
    /// anything in the scene does — a pose, a streamed terrain tile, an
    /// interpolated actor. On the phase-30 city that made the scatter caster pack
    /// run **every frame** over eleven thousand instances: wave I4b's per-pass
    /// record clock measured this node at **3.149 ms of CPU against 0.344 ms of
    /// GPU**, the second-dearest thing to record in a lit frame and one nobody
    /// had reason to suspect, because a shadow pass that draws nothing looks free.
    ///
    /// Both halves are keyed on what they actually read:
    ///
    /// * the **rigid** half on the packed bytes themselves, which is exact and
    ///   costs `O(scene.instances)` — and *zero* on a level whose casters are all
    ///   scatter, which is what the phase-30 city is;
    /// * the **scatter** half on
    ///   [`scatter_caster_fold`](super::scatter::scatter_caster_fold) plus the
    ///   camera bucket, the shadow range and the caster settings — everything
    ///   `pack_fallback` reads and nothing else. Its result is **cached**, so a
    ///   frame in which only the rigid half moved re-merges rather than re-packs.
    ///
    /// The scatter terms enter the key only when the scene carries scatter, so a
    /// level with no foliage at all pays nothing for them.
    ///
    /// # …and what the exact key COSTS, which the first write-up did not say
    ///
    /// The old key was `scene.version`, a `u64`, compared **before**
    /// `pack_bucketed` ran — so a frame in which nothing changed left this node
    /// at `O(1)`. The content key has to be computed from the content, so a
    /// frame in which nothing changed now packs every rigid instance and hashes
    /// the result: **the cache HIT went from `O(1)` to `O(scene.instances)`**,
    /// and only the MISS is the same order it always was. That is the trade the
    /// wave took and it is the right one on the scene it was taken for — the
    /// city's casters are all scatter, `scene.instances` is the character and a
    /// few props, and the node measured 3.149 → 0.157 ms — but a level with tens
    /// of thousands of rigid casters and a still camera pays a pack and a hash
    /// per frame where it used to pay a comparison. Priced, carried, not hidden;
    /// the cheap way out (keeping `version` as a *negative* pre-filter, which is
    /// sound because a version that has not moved cannot hide a change) is
    /// refused here because it puts back the over-approximating coupling this
    /// change exists to remove, and nothing in the tree could arm it.
    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        let scatter_eye = frame.view.origin.to_world(frame.view.eye_local());
        let scatter_key = scatter_caster_key(
            &frame.scene.scatter,
            &frame.view.origin,
            scatter_eye,
            frame.settings.shadows.max_distance,
            frame.settings.scatter.caster_stamp(),
        );
        // Opaque+masked casters only (translucent geometry doesn't cast; folding
        // translucent shadows in is a documented follow-up).
        let (raw, ranges, _translucent) = pack_bucketed(&frame.view.origin, &frame.scene.instances);
        let rigid_fold = xxhash_rust::xxh3::xxh3_128(bytemuck::cast_slice(&raw));
        let key = (
            rigid_fold,
            ranges.clone(),
            frame.view.origin.origin(),
            scatter_key,
        );
        if self.uploaded_version.as_ref() == Some(&key) {
            return;
        }
        if self.scatter_key != scatter_key {
            self.scatter_pack = scatter_key.map(|_| self.pack_scatter_casters(frame, scatter_eye));
            self.scatter_key = scatter_key;
        }
        let cached = self
            .scatter_pack
            .clone()
            .unwrap_or((Vec::new(), EMPTY_RANGES));
        frame
            .scatter_audit
            .record_shadow_casters(cached.0.len() as u32);
        let (raw, ranges) = super::scatter::merge_bucketed((raw, ranges), cached);
        self.ranges = ranges;
        self.instance_count = raw.len() as u32;
        if !raw.is_empty() {
            if self.instances.is_none() || self.instance_capacity < raw.len() {
                let capacity = raw.len().next_power_of_two().max(64);
                self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("shadow-instances"),
                    size: (capacity * std::mem::size_of::<InstanceRaw>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.instance_capacity = capacity;
            }
            gpu.queue.write_buffer(
                self.instances.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&raw),
            );
        }
        self.uploaded_version = Some(key);
    }

    /// The P18.5 scatter caster pack — the half of [`sync`](Self::sync) whose
    /// result is cached on `scatter_key`.
    ///
    /// Before P18.5 a PCG volume's and a foliage entity's instances WERE
    /// `scene.instances`, so they cast shadows for free; moving them onto the GPU
    /// scatter path would have silently deleted every blade of grass's shadow,
    /// which is the worst shape a regression can have — invisible in a compile,
    /// invisible in a unit test, and obvious the moment someone looks at the
    /// ground. They are packed here through the same `pack_fallback` the CPU
    /// fallback uses, under `shadow_caster_settings` — so the caster band is
    /// exactly `min(mesh_distance_m, cull_distance_m, shadows.max_distance ×
    /// 1.5)`, every term of it a clamp DOWN against a knob the host already set
    /// (the tier's ceilings, the authored draw distance), and the impostor band
    /// is dropped entirely because a camera-facing card is not a silhouette from
    /// the sun's point of view. The eye is the same bucketed lattice the fallback
    /// uses, so the caster set is a pure function of the key rather than of every
    /// camera micro-motion, and the pack is bounded by
    /// `MAX_CPU_SCATTER_INSTANCES`.
    fn pack_scatter_casters(
        &self,
        frame: &FrameData,
        scatter_eye: glam::DVec3,
    ) -> (Vec<InstanceRaw>, [std::ops::Range<u32>; 5]) {
        let caster_settings = super::scatter::shadow_caster_settings(
            &frame.settings.scatter,
            frame.settings.shadows.max_distance,
        );
        let pack = super::scatter::pack_fallback(
            &frame.view.origin,
            &frame.scene.scatter,
            super::scatter::bucket_center(scatter_eye),
            &caster_settings,
            super::scatter::MAX_CPU_SCATTER_INSTANCES,
            super::scatter::PackPurpose::Casters,
        );
        if pack.clamped {
            // Reported, not swallowed — the P18.2 streaming-report discipline. A
            // silently truncated caster set reads as "the distant grass stopped
            // casting", which is indistinguishable from the band clamp doing its
            // job unless somebody says which one happened.
            tracing::warn!(
                "inf-render: scatter shadow casters clamped to {} of {} inside the \
                 shadow range — distant instances stop casting (nearest-first)",
                super::scatter::MAX_CPU_SCATTER_INSTANCES,
                pack.considered,
            );
        }
        (pack.instances, pack.ranges)
    }
}

impl RenderNode for ShadowNode {
    fn name(&self) -> &'static str {
        "shadow"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let s = &frame.settings.shadows;
        if !s.enabled {
            // Publish the disabled uniform (valid enabled=0 flag) once and render
            // nothing; the buffer is created once and only this node writes it, so
            // re-publishing the same constant every frame is redundant.
            if !self.published_disabled {
                gpu.queue.write_buffer(
                    &frame.shadow.uniform,
                    0,
                    bytemuck::bytes_of(&ShadowDataGpu::disabled()),
                );
                self.published_disabled = true;
            }
            return;
        }
        // The enabled path overwrites the uniform below, so a later disable must
        // re-publish the disabled block.
        self.published_disabled = false;

        // Shadow caster: the first directional light that **casts shadows** (R-P3
        // honours `cast_shadows` for directional CSM; point/spot shadow maps are
        // deferred, so their flag is inert). Fallbacks:
        //  * no directional light at all → the scene's **projected** sun (P17.1;
        //    a point/spot-only scene still gets grounded shadows, matching the
        //    receiver shaders' `view.sun_dir` fallback — and both now follow the
        //    time of day);
        //  * directional lights exist but none casts → no CSM this frame (publish
        //    the disabled uniform so receivers read a valid `enabled = 0`, render
        //    nothing).
        let caster = frame
            .scene
            .lights
            .iter()
            .find(|l| l.kind == LightKind::Directional && l.cast_shadows)
            .map(|l| l.direction.normalize_or_zero())
            .filter(|d| d.length_squared() > 1e-6);
        let any_directional = frame
            .scene
            .lights
            .iter()
            .any(|l| l.kind == LightKind::Directional);
        let light_dir_to = match caster {
            Some(d) => d,
            None if !any_directional => frame.scene.sun.unit_direction(),
            None => {
                gpu.queue.write_buffer(
                    &frame.shadow.uniform,
                    0,
                    bytemuck::bytes_of(&ShadowDataGpu::disabled()),
                );
                self.published_disabled = true;
                return;
            }
        };

        self.sync(gpu, frame);

        // Only pay for the alpha-test fragment when the scene has masked casters.
        let caster_pipeline = if frame.scene.instances.iter().any(|i| i.blend == 1) {
            &self.pipeline_masked
        } else {
            &self.pipeline
        };

        // Cascade splits across the shadow range.
        let near = frame.view.near.max(0.05);
        let far = s.max_distance.max(near + 1.0);
        let splits = cascade_splits(near, far, s.lambda);

        let eye = frame.view.eye_local();
        let fwd = frame.view.forward;
        let up = frame.view.up;
        let fov = frame.view.fov_y;
        let aspect = frame.view.aspect();

        let mut data = ShadowDataGpu::disabled();
        data.params = [1.0, s.depth_bias, s.normal_bias, SHADOW_CASCADES as f32];
        for c in 0..SHADOW_CASCADES {
            let d0 = if c == 0 { near } else { splits[c - 1] };
            let d1 = splits[c];
            let corners = frustum_slice_corners(eye, fwd, up, fov, aspect, d0, d1);
            let (center, radius) = bounding_sphere(&corners);
            let (vp, texel) = cascade_matrix(light_dir_to, center, radius, SHADOW_RESOLUTION);
            data.cascade_vp[c] = vp.to_cols_array();
            data.splits[c] = d1;
            data.texel_world[c] = texel;
            gpu.queue.write_buffer(
                &self.cascade_bufs[c],
                0,
                bytemuck::bytes_of(&CascadeGpu {
                    view_proj: vp.to_cols_array(),
                }),
            );
        }
        // P18.4 cascade blending: the receiver lerps into the next cascade across
        // the last `blend × range` of this one. `0` restores the hard switch
        // exactly (the shader branch is not taken and the second PCF never issues).
        data.splits[3] = s.cascade_blend.clamp(0.0, 0.5);
        gpu.queue
            .write_buffer(&frame.shadow.uniform, 0, bytemuck::bytes_of(&data));

        // One depth-only render per cascade (always run so the layer is cleared to
        // far even with no casters → everything reads as lit).
        for c in 0..SHADOW_CASCADES {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow-cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &frame.shadow.layer_views[c],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(SHADOW_DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.instance_count == 0 {
                continue;
            }
            let Some(instances) = self.instances.as_ref() else {
                continue;
            };
            pass.set_pipeline(caster_pipeline);
            pass.set_bind_group(0, &self.cascade_bgs[c], &[]);
            self.prim.draw(&mut pass, instances, &self.ranges);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{ScatterBatch, ScatterData, ScatterInstance};
    use glam::{DVec3, Quat, Vec3};
    use inf_math::FloatingOrigin;
    use std::sync::Arc;

    fn batch(anchor: DVec3) -> ScatterBatch {
        ScatterBatch::lit(
            Arc::new(ScatterData::build(
                crate::primitives::PrimMesh::Cube,
                anchor,
                [ScatterInstance {
                    position: anchor + DVec3::new(3.0, 0.0, -2.0),
                    rotation: Quat::IDENTITY,
                    scale: Vec3::new(20.0, 30.0, 7.4),
                    color: [1.0; 4],
                }],
            )),
            anchor,
            0.5,
            7,
        )
    }

    /// **A CACHED PACK IS A PACK AT AN ORIGIN** (the I4b audit).
    ///
    /// `pack_fallback` writes render-local model matrices, so the bytes it
    /// produces are a function of the [`FloatingOrigin`] it was handed as much as
    /// of the batches. Island wave I4b cached that pack under a key of the eye
    /// bucket, the shadow range, the caster stamp and the content fold — and *not*
    /// the origin. The two lattices do not line up: a rebase moves the origin by
    /// `REBASE_DISTANCE` (1 024 m) and the eye bucket quantizes the **world** eye
    /// onto 8 m, so the frame a rebase fires in is almost always one where the
    /// bucket did not tick. The whole-pack key saw the rebase and re-uploaded a
    /// merge of the STALE cached scatter half.
    ///
    /// Built to falsify in both directions: the same batches at two origins must
    /// give two keys (or the cache serves the wrong bytes), and the same batches
    /// at one origin must give one key (or the cache never hits and the 3.15 ms
    /// this wave removed comes straight back).
    #[test]
    fn the_scatter_caster_key_moves_when_the_pack_it_caches_would() {
        let batches = [batch(DVec3::new(120.0, 0.0, -40.0))];
        let eye = DVec3::new(100.0, 2.0, -30.0);
        let a = FloatingOrigin::new(DVec3::ZERO);
        let mut b = FloatingOrigin::new(DVec3::ZERO);
        assert!(
            b.maybe_rebase(eye + DVec3::new(inf_math::REBASE_DISTANCE + 50.0, 0.0, 0.0)),
            "the fixture must actually rebase, or this arm compares one origin \
             with itself"
        );
        assert_ne!(a.origin(), b.origin(), "the two origins must differ");

        let ka = scatter_caster_key(&batches, &a, eye, 250.0, 11);
        let kb = scatter_caster_key(&batches, &b, eye, 250.0, 11);
        assert_eq!(
            ka.map(|k| k.0),
            kb.map(|k| k.0),
            "the eye bucket must be the SAME across the rebase — otherwise this \
             arm is passing because the bucket ticked, which is the case the \
             defect does NOT occur in"
        );
        assert_ne!(
            ka, kb,
            "the same scatter batches at two floating origins produced one cache \
             key, so a rebase serves a cached caster pack whose model matrices are \
             a kilometre out of place — every scatter shadow in the frame is \
             displaced until the camera leaves its 8 m eye bucket"
        );

        // …and the pack really is different, which is what makes the key's job
        // real rather than defensive.
        let settings = crate::settings::ScatterSettings::default();
        let cs = super::super::scatter::shadow_caster_settings(&settings, 250.0);
        let pa = super::super::scatter::pack_fallback(
            &a,
            &batches,
            super::super::scatter::bucket_center(eye),
            &cs,
            super::super::scatter::MAX_CPU_SCATTER_INSTANCES,
            super::super::scatter::PackPurpose::Casters,
        );
        let pb = super::super::scatter::pack_fallback(
            &b,
            &batches,
            super::super::scatter::bucket_center(eye),
            &cs,
            super::super::scatter::MAX_CPU_SCATTER_INSTANCES,
            super::super::scatter::PackPurpose::Casters,
        );
        assert_eq!(pa.instances.len(), 1, "the fixture packs its one instance");
        assert_ne!(
            bytemuck::cast_slice::<_, u8>(&pa.instances),
            bytemuck::cast_slice::<_, u8>(&pb.instances),
            "the packed caster bytes did not move with the origin, so this arm \
             cannot say the key needs to"
        );

        // The control: nothing moved, so the key must not move either.
        assert_eq!(
            scatter_caster_key(&batches, &a, eye, 250.0, 11),
            ka,
            "an unchanged scene produced two different keys — the cache would \
             never hit"
        );
        assert_eq!(
            scatter_caster_key(&[], &a, eye, 250.0, 11),
            None,
            "a scatter-free scene must keep the scatter terms out of the key"
        );
    }

    /// **THE FOLD IS PINNED FIELD BY FIELD**, which is the P22 allowlist law
    /// applied to a cache key (the I4b audit).
    ///
    /// [`super::scatter::scatter_caster_fold`] exists to fold "everything
    /// `pack_fallback` reads off the batches themselves". A ban enumerates what
    /// somebody thought of; an allowlist enumerates what is allowed — so this
    /// walks **every field of `ScatterBatch`**, changes it, and requires the fold
    /// to move. A field that stops being folded is a cached shadow caster pack
    /// that never notices the change: a batch whose anchor moved keeps its old
    /// world positions, a batch whose bands moved keeps the wrong instance set,
    /// a batch whose material moved casts with the wrong `InstanceRaw`.
    ///
    /// `ScatterBatch` has nine fields and nine rows below. The day a tenth is
    /// added, `..Default::default()` does not exist on this type — the struct
    /// literal in `mutate` stops compiling, which is the point of writing it as a
    /// literal rather than a `let mut b = base; b.x = …`.
    ///
    /// **And satisfying the compiler is not satisfying this arm** (island wave
    /// I8b audit). Wave I8b added the ninth field, `casts_shadows`; the literals
    /// duly stopped compiling and were repaired by writing `casts_shadows: true`
    /// into all eight of them — which is exactly what an allowlist must not be
    /// able to absorb. `pack_fallback` reads that field, so the fold has to see
    /// it or a cached caster pack serves the pre-opt-out instance set for ever.
    /// It is folded now, and this is its row.
    #[test]
    fn the_scatter_caster_fold_moves_for_every_field_of_a_batch() {
        use crate::passes::scatter::scatter_caster_fold;
        let anchor = DVec3::new(120.0, 0.0, -40.0);
        let base = batch(anchor);
        let baseline = scatter_caster_fold(std::slice::from_ref(&base));

        // Every field, one at a time, spelled out as a whole literal so a new
        // field cannot be added without this list refusing to compile.
        let variants: [(&str, ScatterBatch); 9] = [
            (
                "data (the payload's own content key)",
                ScatterBatch {
                    data: Arc::new(ScatterData::build(
                        crate::primitives::PrimMesh::Cube,
                        anchor,
                        [ScatterInstance {
                            // one instance moved by a metre
                            position: anchor + DVec3::new(4.0, 0.0, -2.0),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::new(20.0, 30.0, 7.4),
                            color: [1.0; 4],
                        }],
                    )),
                    anchor: base.anchor,
                    metallic: base.metallic,
                    roughness: base.roughness,
                    emissive: base.emissive,
                    id: base.id,
                    draw_distance: base.draw_distance,
                    near_distance: base.near_distance,
                    casts_shadows: true,
                },
            ),
            (
                "anchor",
                ScatterBatch {
                    data: base.data.clone(),
                    anchor: anchor + DVec3::new(0.0, 0.0, 1.0),
                    metallic: base.metallic,
                    roughness: base.roughness,
                    emissive: base.emissive,
                    id: base.id,
                    draw_distance: base.draw_distance,
                    near_distance: base.near_distance,
                    casts_shadows: true,
                },
            ),
            (
                "metallic",
                ScatterBatch {
                    data: base.data.clone(),
                    anchor: base.anchor,
                    metallic: 1.0,
                    roughness: base.roughness,
                    emissive: base.emissive,
                    id: base.id,
                    draw_distance: base.draw_distance,
                    near_distance: base.near_distance,
                    casts_shadows: true,
                },
            ),
            (
                "roughness",
                ScatterBatch {
                    data: base.data.clone(),
                    anchor: base.anchor,
                    metallic: base.metallic,
                    roughness: 0.25,
                    emissive: base.emissive,
                    id: base.id,
                    draw_distance: base.draw_distance,
                    near_distance: base.near_distance,
                    casts_shadows: true,
                },
            ),
            (
                "emissive",
                ScatterBatch {
                    data: base.data.clone(),
                    anchor: base.anchor,
                    metallic: base.metallic,
                    roughness: base.roughness,
                    emissive: [0.0, 0.0, 1.0],
                    id: base.id,
                    draw_distance: base.draw_distance,
                    near_distance: base.near_distance,
                    casts_shadows: true,
                },
            ),
            (
                "id (the pick id every InstanceRaw carries)",
                ScatterBatch {
                    data: base.data.clone(),
                    anchor: base.anchor,
                    metallic: base.metallic,
                    roughness: base.roughness,
                    emissive: base.emissive,
                    id: base.id + 1,
                    draw_distance: base.draw_distance,
                    near_distance: base.near_distance,
                    casts_shadows: true,
                },
            ),
            (
                "draw_distance (the outer band)",
                ScatterBatch {
                    data: base.data.clone(),
                    anchor: base.anchor,
                    metallic: base.metallic,
                    roughness: base.roughness,
                    emissive: base.emissive,
                    id: base.id,
                    draw_distance: 500.0,
                    near_distance: base.near_distance,
                    casts_shadows: true,
                },
            ),
            (
                "near_distance (the IB-2b inner cut)",
                ScatterBatch {
                    data: base.data.clone(),
                    anchor: base.anchor,
                    metallic: base.metallic,
                    roughness: base.roughness,
                    emissive: base.emissive,
                    id: base.id,
                    draw_distance: base.draw_distance,
                    near_distance: 192.0,
                    casts_shadows: true,
                },
            ),
            (
                "casts_shadows (the I8b opt-out `pack_fallback` skips on)",
                ScatterBatch {
                    data: base.data.clone(),
                    anchor: base.anchor,
                    metallic: base.metallic,
                    roughness: base.roughness,
                    emissive: base.emissive,
                    id: base.id,
                    draw_distance: base.draw_distance,
                    near_distance: base.near_distance,
                    casts_shadows: false,
                },
            ),
        ];

        for (field, v) in &variants {
            assert_ne!(
                scatter_caster_fold(std::slice::from_ref(v)),
                baseline,
                "changing `ScatterBatch::{field}` left the caster fold where it \
                 was, so a shadow caster pack cached on it never notices — the \
                 fold has stopped covering a field `pack_fallback` reads"
            );
        }
        println!(
            "the caster fold separates all {} fields of a ScatterBatch",
            variants.len()
        );

        // …and the anti-vacuity control: two batches instead of one, and an empty
        // list, are not the same fold either — a fold that hashed nothing would
        // pass every row above by answering a different constant each call.
        assert_ne!(
            scatter_caster_fold(&[base.clone(), base.clone()]),
            baseline,
            "two copies of a batch fold to the same key as one — the fold is not \
             reading the list"
        );
        assert_eq!(
            scatter_caster_fold(std::slice::from_ref(&base)),
            baseline,
            "the fold is not a pure function of its input"
        );
    }
}
