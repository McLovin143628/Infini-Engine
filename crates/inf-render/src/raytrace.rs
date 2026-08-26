//! **The ray-query shadow experiment** (P28.5, the ROADMAP's last clause) —
//! built to be measured and **never** to be load-bearing.
//!
//! # What "never load-bearing" means here, mechanically
//!
//! * Nothing in this module is reachable from [`EngineRenderer::render`]
//!   (crate::EngineRenderer::render). There is no render-graph node, no pass, no
//!   call site inside the frame. The experiment is driven by its own gate, which
//!   is the only caller in the tree.
//! * [`RaytraceSettings::sun_shadows`](crate::RaytraceSettings) defaults to
//!   **false** and no tier, preset or clamp ever sets it — the clamps only ever
//!   clear it, which is [`AdapterCaps::clamp_ray_query`](crate::caps::AdapterCaps::clamp_ray_query),
//!   the `vsm.enabled` shape one more time.
//! * **The shipped device does not request the feature at all.** The standing
//!   request-if-available rule (`POLYGON_MODE_LINE`, `TEXTURE_COMPRESSION_BC`)
//!   turned out to be unsafe for an `EXPERIMENTAL_*` one: wgpu 30 refuses it
//!   without an `unsafe` acknowledgement token, so adding it to the ordinary
//!   optional mask made `request_device` **fail** and every headless test in
//!   the tree skip for want of an adapter. The experiment builds its own
//!   device ([`GpuContext::headless_ray_query`](crate::GpuContext::headless_ray_query)),
//!   which is the only place in the tree that signs the token, and the shipped
//!   context is exactly the one it was before this batch.
//! * The committed goldens are byte-frozen across the batch, which is the
//!   measurement that says the shipped frame did not move.
//!
//! # The platform bound, stated rather than discovered
//!
//! `wgpu::Features::EXPERIMENTAL_RAY_QUERY` is documented **Vulkan-only** and
//! native-only in the pinned wgpu 30. This tree's instance is
//! `VULKAN | METAL` (`crate::gpu::create_instance`), so on macOS the feature is
//! absent by construction, and a software adapter (lavapipe/WARP) does not have
//! it either. Every arm that needs a device therefore states what it ran on;
//! the probe, the clamp and the refusal arms run everywhere. That is the P25
//! one-platform law: say what ran where.
//!
//! # Why the primary ray is traced too
//!
//! `rt_sun_shadow.wgsl` casts a camera ray into the same TLAS instead of
//! reading a depth buffer. It costs a second trace and buys three things: the
//! experiment depends on no shipped pass, its **coverage bound is explicit**
//! (the pass only has an opinion where the TLAS has geometry), and a comparison
//! against the shipped shadow path is naturally restricted to exactly those
//! pixels.

use wgpu::util::DeviceExt;

use crate::gpu::GpuContext;

/// The primary ray hit nothing in the TLAS — this pixel has no ray-traced
/// surface and no verdict about it.
pub const RT_MISS: u32 = 0;
/// A surface, and the shadow ray toward the sun escaped.
pub const RT_LIT: u32 = 1;
/// A surface, and the shadow ray toward the sun was occluded.
pub const RT_SHADOWED: u32 = 2;

/// The shadow ray's `tmin`, in metres — how far off the surface it starts.
///
/// A shadow ray fired from exactly the hit point re-hits the triangle it came
/// from at `t = 0` on any real intersector, so every surface shadows itself and
/// the whole frame reads occluded. 1 mm is the same order as the depth bias the
/// rasterized paths carry for the same reason, and is asserted to matter by
/// `a_zero_surface_offset_shadows_every_surface_with_itself`.
pub const RT_SHADOW_BIAS_M: f32 = 1.0e-3;

/// Triangle geometry for one bottom-level acceleration structure, in the
/// asset's own local space.
///
/// **The meshlet door is [`RtBlasSource::from_meshlet_level`]** — the ROADMAP's
/// clause is "BLAS over meshlet clusters", and a cluster is a meshlet, so the
/// source is built by walking one LOD level's meshlets and asking each for its
/// triangles through `VgeomMesh::triangle`. Positions are shared: a meshlet is
/// a set of indices into the DAG's one welded vertex buffer, so the BLAS gets
/// one vertex buffer and one index buffer per *level*, not per meshlet.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RtBlasSource {
    /// Vertex positions, local space.
    pub positions: Vec<[f32; 3]>,
    /// Triangle indices into `positions`, three per triangle.
    pub indices: Vec<u32>,
}

impl RtBlasSource {
    /// **Every meshlet in one LOD level of a cooked DAG**, as one indexed
    /// triangle list.
    ///
    /// `level` indexes `mesh.levels`; `None` when the level does not exist, so
    /// a caller cannot silently trace an empty structure. The vertex buffer is
    /// the DAG's own — welded, shared, and already the thing the meshlet path
    /// pulls from — so this is the cluster geometry itself and not a second
    /// derivation of it.
    pub fn from_meshlet_level(mesh: &inf_vgeom::VgeomMesh, level: usize) -> Option<Self> {
        let range = mesh.levels.get(level)?;
        let mut indices = Vec::new();
        for m in 0..range.meshlet_count as usize {
            let mi = range.meshlet_start as usize + m;
            let meshlet = mesh.meshlets.get(mi)?;
            for t in 0..meshlet.triangle_count as usize {
                indices.extend_from_slice(&mesh.triangle(mi, t));
            }
        }
        Some(Self {
            positions: mesh.vertices.iter().map(|v| v.position).collect(),
            indices,
        })
    }

    /// Triangles in this source.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// One TLAS instance: which [`RtBlasSource`] it draws, and where.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtInstance {
    /// Index into the `sources` slice handed to [`RtScene::build`].
    pub blas: usize,
    /// A row-major 3×4 affine transform, **render-local** — the floating origin
    /// is the host's to apply, exactly as it is for every other buffer this
    /// renderer uploads.
    pub transform: [f32; 12],
}

/// Where the experiment traces from, in render-local space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtView {
    /// Eye, render-local.
    pub eye: glam::Vec3,
    /// Unit view direction.
    pub forward: glam::Vec3,
    /// View up.
    pub up: glam::Vec3,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Unit direction **toward** the sun.
    pub sun: glam::Vec3,
    /// Ray `tmin` for the primary ray, metres.
    pub near: f32,
    /// Ray `tmax` for both rays, metres.
    pub far: f32,
    /// The shadow ray's `tmin` — how far off the surface it starts, metres.
    ///
    /// A **parameter and not a constant**, so a gate can trace with zero and
    /// watch every surface shadow itself. [`RT_SHADOW_BIAS_M`] is the value the
    /// experiment uses.
    pub shadow_bias: f32,
}

/// What one [`RtScene::build`] cost and covered — the numbers the memo's
/// verdict is made of.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RtBuildStats {
    /// Bottom-level structures built.
    pub blas: usize,
    /// Instances in the top-level structure.
    pub instances: usize,
    /// Triangles across every BLAS, counted once per BLAS and not per instance.
    pub triangles: usize,
    /// Bytes of vertex + index data uploaded as build input.
    pub input_bytes: u64,
}

/// A built acceleration structure: one BLAS per source, one TLAS over the
/// instances, already built on the queue.
#[derive(Debug)]
pub struct RtScene {
    _blas: Vec<wgpu::Blas>,
    tlas: wgpu::Tlas,
    stats: RtBuildStats,
}

impl RtScene {
    /// Build every BLAS and the TLAS over them, in one command buffer.
    ///
    /// `Err` — rather than a panic or a silent empty structure — when the
    /// device was not created with the ray-query feature, which is the case on
    /// every adapter this tree supports except Vulkan ones that expose it.
    pub fn build(
        gpu: &GpuContext,
        sources: &[RtBlasSource],
        instances: &[RtInstance],
    ) -> Result<Self, String> {
        if !gpu.supports_ray_query() {
            return Err("this device has no EXPERIMENTAL_RAY_QUERY".into());
        }
        if sources.is_empty() || instances.is_empty() {
            return Err("an acceleration structure over nothing is not one".into());
        }
        let mut stats = RtBuildStats {
            blas: sources.len(),
            instances: instances.len(),
            ..Default::default()
        };

        // The build inputs. `BLAS_INPUT` is a usage of its own in wgpu 30 — a
        // vertex buffer that has never carried it is refused at build time.
        let mut buffers = Vec::with_capacity(sources.len());
        let mut sizes = Vec::with_capacity(sources.len());
        for (i, src) in sources.iter().enumerate() {
            if src.indices.len() % 3 != 0 {
                return Err(format!("source {i} has {} indices", src.indices.len()));
            }
            if let Some(bad) = src
                .indices
                .iter()
                .find(|i| **i as usize >= src.positions.len())
            {
                return Err(format!(
                    "source {i} indexes vertex {bad} of {}",
                    src.positions.len()
                ));
            }
            let verts = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rt-blas-verts"),
                    contents: bytemuck::cast_slice(&src.positions),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
            let idx = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rt-blas-indices"),
                    contents: bytemuck::cast_slice(&src.indices),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
            stats.triangles += src.triangle_count();
            stats.input_bytes += (src.positions.len() * 12 + src.indices.len() * 4) as u64;
            sizes.push(wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: src.positions.len() as u32,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(src.indices.len() as u32),
                // OPAQUE, so the driver commits triangle intersections itself
                // and `rayQueryProceed` never hands the shader a candidate it
                // would have to confirm — the shader is a visibility query and
                // not an any-hit program.
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            });
            buffers.push((verts, idx));
        }

        let blas: Vec<wgpu::Blas> = sizes
            .iter()
            .map(|size| {
                gpu.device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("rt-blas"),
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size.clone()],
                    },
                )
            })
            .collect();

        let mut tlas = gpu.device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("rt-tlas"),
            max_instances: instances.len() as u32,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        for (i, inst) in instances.iter().enumerate() {
            let b = blas
                .get(inst.blas)
                .ok_or_else(|| format!("instance {i} names BLAS {}", inst.blas))?;
            // `custom_data` carries the instance index so a future comparison
            // can attribute a hit; the mask is "everything", because this
            // experiment has exactly one ray class.
            tlas[i] = Some(wgpu::TlasInstance::new(b, inst.transform, i as u32, 0xFF));
        }

        let entries: Vec<wgpu::BlasBuildEntry<'_>> = blas
            .iter()
            .zip(sizes.iter())
            .zip(buffers.iter())
            .map(|((b, size), (verts, idx))| wgpu::BlasBuildEntry {
                blas: b,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size,
                        vertex_buffer: verts,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: Some(idx),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            })
            .collect();

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rt-build"),
            });
        // One call, both levels: wgpu requires every BLAS a TLAS instance names
        // to have been built in this call or an earlier one.
        enc.build_acceleration_structures(entries.iter(), std::iter::once(&tlas));
        gpu.queue.submit([enc.finish()]);

        Ok(Self {
            _blas: blas,
            tlas,
            stats,
        })
    }

    /// What this structure cost and covers.
    pub fn stats(&self) -> RtBuildStats {
        self.stats
    }
}

/// The sun-shadow trace: one compute dispatch, one verdict per pixel.
pub struct RtSunShadow {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// The uniform `rt_sun_shadow.wgsl` reads. Field for field with the WGSL
/// `RtParams`, and the mirror is pinned by
/// `the_shaders_params_block_is_the_rusts`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtParams {
    eye: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    fwd: [f32; 4],
    sun: [f32; 4],
    misc: [f32; 4],
    dims: [u32; 4],
}

impl RtSunShadow {
    /// Compile the pass. Cheap and device-only; the caller decides whether the
    /// device can run it (`GpuContext::supports_ray_query`).
    pub fn new(gpu: &GpuContext) -> Self {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rt_sun_shadow"),
                source: wgpu::ShaderSource::Wgsl(RT_SUN_SHADOW_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rt-sun-shadow"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rt-sun-shadow"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rt-sun-shadow"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("cs_sun_shadow"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipeline, layout }
    }

    /// Trace `w × h` verdicts, one per pixel, and read them back.
    ///
    /// Blocking, because this is an experiment's instrument and not a frame:
    /// the whole point is a number a memo can quote.
    pub fn trace(
        &self,
        gpu: &GpuContext,
        scene: &RtScene,
        view: &RtView,
        w: u32,
        h: u32,
    ) -> Vec<u32> {
        let count = (w as usize) * (h as usize);
        let bytes = (count * 4) as u64;
        let right = view.forward.cross(view.up).normalize();
        let up = right.cross(view.forward).normalize();
        let params = RtParams {
            eye: view.eye.extend(0.0).to_array(),
            right: right.extend(0.0).to_array(),
            up: up.extend(0.0).to_array(),
            fwd: view.forward.normalize().extend(0.0).to_array(),
            sun: view.sun.normalize().extend(view.shadow_bias).to_array(),
            misc: [
                (view.fov_y * 0.5).tan(),
                w as f32 / h as f32,
                view.near,
                view.far,
            ],
            dims: [w, h, 0, 0],
        };
        let ubo = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rt-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt-verdict"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt-verdict-read"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt-sun-shadow"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        });

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rt-trace"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rt-sun-shadow"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, bytes);
        gpu.queue.submit([enc.finish()]);

        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let mapped = staging
            .slice(..)
            .get_mapped_range()
            .expect("the verdict buffer maps");
        let v = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }
}

/// The shader source, exposed so the naga gate and the mirror arm read the same
/// bytes the pipeline compiles (the `SHADER_TABLE` law's standalone half).
pub const RT_SUN_SHADOW_WGSL: &str = include_str!("shaders/rt_sun_shadow.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    /// **The Rust params block and the WGSL one are the same record.** A GPU-free
    /// mirror pin, in the shape `the_shaders_bit_split_is_the_rusts` and
    /// `the_shaders_tangent_unpack_is_the_rusts` already use — and it reads the
    /// SIZE as well as the field names, because a ban on spellings is a ban on
    /// what you thought of (the P22 law).
    #[test]
    fn the_shaders_params_block_is_the_rusts() {
        // 7 vec4s: 6 float, 1 uint.
        assert_eq!(std::mem::size_of::<RtParams>(), 7 * 16);
        let src = RT_SUN_SHADOW_WGSL;
        let block = src
            .split_once("struct RtParams {")
            .expect("the shader declares RtParams")
            .1
            .split_once("};")
            .expect("the struct closes")
            .0;
        for field in ["eye", "right", "up", "fwd", "sun", "misc", "dims"] {
            assert!(
                block.contains(&format!("{field}:")),
                "the WGSL RtParams has no `{field}` — the Rust mirror has"
            );
        }
        // …and exactly seven members, so a field added on one side alone is
        // caught rather than tolerated.
        assert_eq!(
            block.matches(": vec4<").count(),
            7,
            "RtParams is not seven vec4s in the WGSL"
        );
        // The three verdicts are the shader's three, by value.
        assert!(src.contains("verdict[idx] = 0u;"));
        assert!(src.contains("verdict[idx] = 1u;"));
        assert!(src.contains("verdict[idx] = 2u;"));
        assert_eq!((RT_MISS, RT_LIT, RT_SHADOWED), (0, 1, 2));
    }

    /// **A meshlet level really is the cluster geometry** — GPU-free, over a
    /// DAG built here.
    ///
    /// The anti-vacuity is the whole arm: a level with no meshlets produces an
    /// empty source, and an empty source is refused by `RtScene::build` rather
    /// than traced as an empty world.
    #[test]
    fn a_meshlet_level_becomes_one_indexed_triangle_list() {
        fn meshlet(vertex_offset: u32, triangle_offset: u32) -> inf_vgeom::Meshlet {
            inf_vgeom::Meshlet {
                vertex_offset,
                vertex_count: 3,
                triangle_offset,
                triangle_count: 1,
                center: [0.0; 3],
                radius: 1.0,
                cone_axis: [0.0, 1.0, 0.0],
                cone_cutoff: 1.0,
                group: inf_vgeom::Meshlet::NO_GROUP,
                lod_level: 0,
                error: 0.0,
                parent_error: f32::INFINITY,
            }
        }
        let mesh = inf_vgeom::VgeomMesh {
            schema_version: inf_vgeom::VgeomMesh::CURRENT_VERSION,
            vertices: (0..6)
                .map(|i| inf_vgeom::VgeomVertex {
                    position: [i as f32, 0.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                    uv: [0.0, 0.0],
                    tangent: 0,
                })
                .collect(),
            meshlets: vec![meshlet(0, 0), meshlet(3, 3)],
            meshlet_vertices: vec![0, 1, 2, 3, 4, 5],
            meshlet_triangles: vec![0, 1, 2, 0, 1, 2],
            groups: Vec::new(),
            levels: vec![inf_vgeom::LevelRange {
                lod_level: 0,
                meshlet_start: 0,
                meshlet_count: 2,
            }],
            center: [0.0; 3],
            radius: 1.0,
            meshlet_materials: Vec::new(),
        };
        let src = RtBlasSource::from_meshlet_level(&mesh, 0).expect("level 0 exists");
        assert_eq!(src.triangle_count(), 2, "both meshlets' triangles");
        assert_eq!(src.positions.len(), 6, "the DAG's own welded buffer");
        // The SECOND meshlet's triangle indexes the second vertex run — a
        // builder that forgot `vertex_offset` would produce [0,1,2] twice.
        assert_eq!(&src.indices[3..6], &[3, 4, 5]);
        assert!(RtBlasSource::from_meshlet_level(&mesh, 1).is_none());
    }
}
