//! Heightfield terrain pass (P10.1): geometry-clipmap LOD.
//!
//! Each authored terrain tile becomes one draw of a unit grid patch, displaced
//! in the vertex shader by that tile's cached R32Float height texture. The patch
//! mesh density falls with the tile's LOD (a power-of-two vertex decimation
//! chosen by camera distance → concentric LOD rings around the camera); LOD
//! morphing blends toward the coarser grid to kill popping; a skirt ring on each
//! patch drops its boundary down to seal cracks between neighbouring LODs.
//!
//! The CPU-side ring/patch assembly, LOD selection, morph, and cull are pure
//! functions ([`assemble_patches`] & friends) so they unit-test without a GPU.
//! Height textures are cached per tile in a `BTreeMap` (deterministic), uploaded
//! version-gated like the sprite texture cache. When `scene.terrain` is `None`
//! the node encodes nothing — pre-P10.1 scenes stay byte-identical.

use std::collections::BTreeMap;

use glam::{DVec3, Vec3};
use inf_math::FloatingOrigin;
use inf_render_2d::aabb_visible;

use crate::camera::{RenderView, DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{RenderTerrain, RenderTerrainTile};

/// Number of discrete LOD levels (finest = 0).
pub const TERRAIN_LOD_COUNT: u32 = 4;
/// Grid cells per patch side at LOD 0 (halved each coarser level, min 1). The
/// patch mesh density — decoupled from the (usually finer) height-texture
/// resolution, which is sampled for the actual displacement.
pub const TERRAIN_BASE_CELLS: u32 = 32;
/// LOD-0 ring radius as a multiple of the tile span; each coarser ring doubles.
pub const TERRAIN_LOD_SCALE: f64 = 1.5;
/// Fraction of a LOD band (nearest the far edge) over which the morph ramps 0→1.
pub const TERRAIN_MORPH_REGION: f64 = 0.35;

/// Grid cells per side for a patch at `lod` (`TERRAIN_BASE_CELLS >> lod`, min 1).
pub fn cells_at_lod(lod: u32) -> u32 {
    (TERRAIN_BASE_CELLS >> lod.min(TERRAIN_LOD_COUNT - 1)).max(1)
}

/// Camera-distance thresholds between LOD bands (`TERRAIN_LOD_COUNT − 1` of them),
/// scaled by the tile span. A tile centre nearer than `thresholds[0]` is LOD 0.
pub fn lod_thresholds(span: f64) -> [f64; (TERRAIN_LOD_COUNT - 1) as usize] {
    let mut t = [0.0; (TERRAIN_LOD_COUNT - 1) as usize];
    for (k, tk) in t.iter_mut().enumerate() {
        *tk = span * TERRAIN_LOD_SCALE * (1u64 << k) as f64;
    }
    t
}

/// LOD level for a tile centre at horizontal distance `dist` from the eye: the
/// number of thresholds it has passed, capped at the coarsest level.
pub fn lod_for_distance(dist: f64, thresholds: &[f64]) -> u32 {
    let n = thresholds.iter().filter(|&&t| dist >= t).count() as u32;
    n.min(TERRAIN_LOD_COUNT - 1)
}

/// Morph factor (0 → use the fine grid, 1 → snapped to the coarser grid) for a
/// tile at `dist` in LOD `lod`: ramps up over the last [`TERRAIN_MORPH_REGION`]
/// of the band before the next threshold. The coarsest LOD never morphs (0).
pub fn morph_factor(dist: f64, lod: u32, thresholds: &[f64]) -> f32 {
    if lod as usize >= thresholds.len() {
        return 0.0; // coarsest level: nothing coarser to morph toward
    }
    let upper = thresholds[lod as usize];
    let lower = if lod == 0 {
        0.0
    } else {
        thresholds[lod as usize - 1]
    };
    let start = upper - TERRAIN_MORPH_REGION * (upper - lower);
    if dist <= start {
        0.0
    } else if dist >= upper {
        1.0
    } else {
        let t = (dist - start) / (upper - start).max(f64::MIN_POSITIVE);
        // smoothstep
        (t * t * (3.0 - 2.0 * t)) as f32
    }
}

/// Render-local AABB (`min`, `max`) of a terrain tile (planar span + its height
/// bounds), rebased through the floating origin. Used for frustum culling.
pub fn tile_aabb_local(
    tile: &RenderTerrainTile,
    span: f64,
    origin: &FloatingOrigin,
) -> (Vec3, Vec3) {
    let (hmin, hmax) = tile.height_bounds;
    let wmin = DVec3::new(tile.origin.x, tile.origin.y + hmin as f64, tile.origin.z);
    let wmax = DVec3::new(
        tile.origin.x + span,
        tile.origin.y + hmax as f64,
        tile.origin.z + span,
    );
    (origin.to_render(wmin), origin.to_render(wmax))
}

/// One assembled patch: which tile, at what LOD, with how much morph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainPatch {
    pub coord: (i32, i32),
    pub lod: u32,
    pub morph: f32,
}

/// Assemble the visible LOD patches for `terrain` under `view`: cull each tile
/// against the frustum, pick its LOD from the camera's horizontal distance to the
/// tile centre, and compute its morph. Deterministic: tiles arrive sorted by
/// coordinate, so the output is too (stable draw order).
pub fn assemble_patches(
    terrain: &RenderTerrain,
    view: &RenderView,
    origin: &FloatingOrigin,
) -> Vec<TerrainPatch> {
    let span = terrain.tile_span();
    let thresholds = lod_thresholds(span);
    let view_proj = view.view_proj();
    let eye = view.eye_world;
    let mut out = Vec::with_capacity(terrain.tiles.len());
    for tile in &terrain.tiles {
        let (lo, hi) = tile_aabb_local(tile, span, origin);
        // A small margin so a tile straddling a screen edge isn't popped early.
        let margin = Vec3::splat((span * 0.02) as f32);
        if !aabb_visible(lo - margin, hi + margin, &view_proj) {
            continue;
        }
        let cx = tile.origin.x + span * 0.5;
        let cz = tile.origin.z + span * 0.5;
        let (dx, dz) = (cx - eye.x, cz - eye.z);
        let dist = (dx * dx + dz * dz).sqrt();
        let lod = lod_for_distance(dist, &thresholds);
        let morph = morph_factor(dist, lod, &thresholds);
        out.push(TerrainPatch {
            coord: tile.coord,
            lod,
            morph,
        });
    }
    out
}

// ── GPU node ────────────────────────────────────────────────────────────────

/// Per-patch instance data (matches `@location(1..=2)` in terrain.wgsl).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PatchRaw {
    /// origin_local.xyz, world tile span.
    o_span: [f32; 4],
    /// morph, cells-at-lod, resolution, skirt depth (m).
    params: [f32; 4],
}

/// A unit grid patch mesh (positions in `[0,1]²`, `z` = skirt flag) + indices for
/// one LOD, including a skirt ring.
struct LodMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

/// Build a `cells × cells` unit grid in `[0,1]²` (vertex = `[u, v, skirt]`) plus a
/// skirt ring on all four edges (duplicate boundary verts flagged `skirt = 1`,
/// stitched into vertical quads). Returns `(vertices, indices)`.
fn build_lod_geometry(cells: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let n = cells.max(1);
    let stride = n + 1;
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let inv = 1.0 / n as f32;
    for j in 0..=n {
        for i in 0..=n {
            verts.push([i as f32 * inv, j as f32 * inv, 0.0]);
        }
    }
    let vid = |i: u32, j: u32| j * stride + i;
    for j in 0..n {
        for i in 0..n {
            let (a, b, c, d) = (vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1));
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    // Skirts: one vertical quad strip per boundary edge.
    let edges: [Vec<u32>; 4] = [
        (0..=n).map(|i| vid(i, 0)).collect(), // bottom (v = 0)
        (0..=n).map(|i| vid(i, n)).collect(), // top    (v = 1)
        (0..=n).map(|j| vid(0, j)).collect(), // left   (u = 0)
        (0..=n).map(|j| vid(n, j)).collect(), // right  (u = 1)
    ];
    for edge in edges {
        let mut sk = Vec::with_capacity(edge.len());
        for &bi in &edge {
            let p = verts[bi as usize];
            sk.push(verts.len() as u32);
            verts.push([p[0], p[1], 1.0]);
        }
        for w in 0..edge.len() - 1 {
            let (t0, t1, s0, s1) = (edge[w], edge[w + 1], sk[w], sk[w + 1]);
            indices.extend_from_slice(&[t0, t1, s1, t0, s1, s0]);
        }
    }
    (verts, indices)
}

/// A cached per-tile pair of textures — R32Float height + Rgba8Unorm splat
/// weights — sharing one bind group (`@group(1)`).
struct TileTextures {
    _height: wgpu::Texture,
    _weights: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    resolution: u32,
}

/// The terrain splat material uniform (`@group(2)`), std140-friendly: four
/// layers' albedo + params (`x` = roughness, `y` = tex_scale) and the macro
/// variation amplitude. Mirrors `TerrainMaterial` in terrain.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialRaw {
    albedo: [[f32; 4]; 4],
    params: [[f32; 4]; 4],
    macro_amp: [f32; 4],
}

impl MaterialRaw {
    fn from_terrain(terrain: &RenderTerrain) -> Self {
        let mut albedo = [[0.0f32; 4]; 4];
        let mut params = [[0.0f32; 4]; 4];
        for (k, layer) in terrain.layers.iter().enumerate() {
            albedo[k] = layer.albedo;
            params[k] = [layer.roughness, layer.tex_scale.max(1e-3), 0.0, 0.0];
        }
        MaterialRaw {
            albedo,
            params,
            macro_amp: [terrain.macro_variation, 0.0, 0.0, 0.0],
        }
    }
}

pub struct TerrainNode {
    pipeline: wgpu::RenderPipeline,
    tile_bgl: wgpu::BindGroupLayout,
    /// One grid mesh per LOD (index 0 = finest).
    lod_meshes: Vec<LodMesh>,
    /// Per-tile height + weight textures, keyed by tile coord (deterministic).
    textures: BTreeMap<(i32, i32), TileTextures>,
    /// Tile coord → index into `terrain.tiles`, rebuilt once per terrain version
    /// (replaces a per-patch linear scan of the tile list). First match wins,
    /// matching the previous `find`.
    tile_index: BTreeMap<(i32, i32), usize>,
    /// Terrain splat material uniform (`@group(2)`) + its bind group.
    material_buffer: wgpu::Buffer,
    material_bg: wgpu::BindGroup,
    /// AO + shadows + GI env bind at `@group(3)` (P13.3b).
    env: super::EnvBinding,
    /// Version of the terrain the texture/material caches reflect.
    uploaded_version: Option<u64>,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
}

/// Reference (CPU) mirror of the shader's triplanar axis weights: normalized
/// `pow(|n|, sharpness)` over the three world planes (YZ→x, XZ→y, XY→z). Kept in
/// sync with `terrain.wgsl`; unit-tested so the blend law is pinned even though
/// the shader itself needs a GPU. Returns `(w_x, w_y, w_z)` summing to 1.
pub fn triplanar_axis_weights(normal: Vec3, sharpness: f32) -> [f32; 3] {
    let mut w = [
        normal.x.abs().powf(sharpness.max(1e-3)),
        normal.y.abs().powf(sharpness.max(1e-3)),
        normal.z.abs().powf(sharpness.max(1e-3)),
    ];
    let sum = w[0] + w[1] + w[2];
    if sum > 0.0 {
        w[0] /= sum;
        w[1] /= sum;
        w[2] /= sum;
    } else {
        w = [0.0, 1.0, 0.0]; // degenerate normal → treat as flat (top plane)
    }
    w
}

/// Reference (CPU) mirror of the shader's macro albedo modulation:
/// `albedo · (1 + amp · fbm)` where `fbm ∈ [-1, 1]`. Kept in sync with
/// `terrain.wgsl`; unit-tested.
pub fn macro_modulate(albedo: Vec3, fbm: f32, amp: f32) -> Vec3 {
    albedo * (1.0 + amp * fbm)
}

impl TerrainNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("terrain"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("terrain").into()),
            });

        // Per-tile bind group (@group(1)): the R32Float height texture (binding 0)
        // + the Rgba8Unorm splat-weight texture (binding 1), both unfilterable-
        // float and sampled via textureLoad (no sampler, no FLOAT32_FILTERABLE).
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let tile_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain-tile"),
                entries: &[tex_entry(0), tex_entry(1)],
            });

        // Splat material uniform bind group (@group(2)).
        let material_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain-material"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let material_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-material"),
            size: std::mem::size_of::<MaterialRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-material"),
            layout: &material_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            }],
        });

        let env = super::EnvBinding::new(gpu);
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain"),
                bind_group_layouts: &[
                    Some(view_bgl),
                    Some(&tile_bgl),
                    Some(&material_bgl),
                    Some(&env.bgl),
                ],
                immediate_size: 0,
            });

        let vertex_attrs = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        }];
        let instance_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 2,
            },
        ];

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("terrain"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<[f32; 3]>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &vertex_attrs,
                        },
                        wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<PatchRaw>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &instance_attrs,
                        },
                    ]
                    .map(Some),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    // Skirts are vertical walls of either winding; drop back-face
                    // culling so they always seal (terrain is opaque + depth-tested).
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(DEPTH_COMPARE),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: SCENE_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });

        let lod_meshes = (0..TERRAIN_LOD_COUNT)
            .map(|lod| {
                let (verts, idx) = build_lod_geometry(cells_at_lod(lod));
                let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("terrain-lod-verts"),
                    size: std::mem::size_of_val(verts.as_slice()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
                let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("terrain-lod-indices"),
                    size: std::mem::size_of_val(idx.as_slice()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&indices, 0, bytemuck::cast_slice(&idx));
                LodMesh {
                    vertices,
                    indices,
                    index_count: idx.len() as u32,
                }
            })
            .collect();

        Self {
            pipeline,
            tile_bgl,
            lod_meshes,
            textures: BTreeMap::new(),
            tile_index: BTreeMap::new(),
            material_buffer,
            material_bg,
            env,
            uploaded_version: None,
            instances: None,
            instance_capacity: 0,
        }
    }

    /// Upload / refresh the per-tile height + weight textures and the splat
    /// material uniform when the terrain changed (version-gated). Drops textures
    /// whose tiles vanished.
    fn sync_textures(&mut self, gpu: &GpuContext, terrain: &RenderTerrain) {
        if self.uploaded_version == Some(terrain.version) {
            return;
        }
        self.uploaded_version = Some(terrain.version);

        // Rebuild the coord→index map alongside the texture cache (same version
        // gate), so the per-patch tile lookup in `run` is O(log n) instead of a
        // linear scan per patch. First insert wins, matching the previous `find`.
        self.tile_index.clear();
        for (i, tile) in terrain.tiles.iter().enumerate() {
            self.tile_index.entry(tile.coord).or_insert(i);
        }

        let res = terrain.tile_resolution.max(2);
        let expect = (res * res) as usize;

        // Splat material uniform (terrain-wide; re-upload on the same version gate).
        gpu.queue.write_buffer(
            &self.material_buffer,
            0,
            bytemuck::bytes_of(&MaterialRaw::from_terrain(terrain)),
        );

        // Remove textures for tiles no longer present.
        let live: std::collections::BTreeSet<(i32, i32)> =
            terrain.tiles.iter().map(|t| t.coord).collect();
        self.textures.retain(|k, _| live.contains(k));

        for tile in &terrain.tiles {
            if tile.heights.len() != expect {
                continue; // malformed — skip rather than panic mid-frame
            }
            let reuse = self
                .textures
                .get(&tile.coord)
                .is_some_and(|t| t.resolution == res);
            if !reuse {
                self.textures
                    .insert(tile.coord, create_tile_textures(gpu, &self.tile_bgl, res));
            }
            let tex = &self.textures[&tile.coord];
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex._height,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&tile.heights),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(res * 4),
                    rows_per_image: Some(res),
                },
                wgpu::Extent3d {
                    width: res,
                    height: res,
                    depth_or_array_layers: 1,
                },
            );

            // Splat weights: fall back to a uniform default when a tile projected
            // no (or a malformed) weight buffer, so shading stays layer 0.
            // Uniform default = 100% layer 0 (mirrors inf_terrain::DEFAULT_WEIGHT;
            // kept local so the renderer stays independent of inf-terrain).
            const DEFAULT_WEIGHT: [u8; 4] = [255, 0, 0, 0];
            let default_weights;
            let weights: &[[u8; 4]] = if tile.weights.len() == expect {
                &tile.weights
            } else {
                default_weights = vec![DEFAULT_WEIGHT; expect];
                &default_weights
            };
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex._weights,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(weights),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(res * 4),
                    rows_per_image: Some(res),
                },
                wgpu::Extent3d {
                    width: res,
                    height: res,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

/// Create an empty `res × res` R32Float height texture + Rgba8Unorm weight
/// texture, sharing one bind group.
fn create_tile_textures(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    res: u32,
) -> TileTextures {
    let size = wgpu::Extent3d {
        width: res,
        height: res,
        depth_or_array_layers: 1,
    };
    let make = |label: &str, format: wgpu::TextureFormat| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let height = make("terrain-height", wgpu::TextureFormat::R32Float);
    let weights = make("terrain-weights", wgpu::TextureFormat::Rgba8Unorm);
    let hview = height.create_view(&wgpu::TextureViewDescriptor::default());
    let wview = weights.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain-tile"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hview),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&wview),
            },
        ],
    });
    TileTextures {
        _height: height,
        _weights: weights,
        bind_group,
        resolution: res,
    }
}

impl RenderNode for TerrainNode {
    fn name(&self) -> &'static str {
        "terrain"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let Some(terrain) = frame.scene.terrain.as_ref() else {
            return; // no terrain → byte-identical to pre-P10.1
        };
        if terrain.tiles.is_empty() {
            return;
        }
        self.sync_textures(gpu, terrain);

        let patches = assemble_patches(terrain, frame.view, &frame.view.origin);
        if patches.is_empty() {
            return;
        }

        // Pack per-patch instance data (one per visible patch, in patch order).
        let span = terrain.tile_span() as f32;
        let res = terrain.tile_resolution.max(2) as f32;
        let origin = &frame.view.origin;
        let mut raw: Vec<PatchRaw> = Vec::with_capacity(patches.len());
        for p in &patches {
            let Some(tile) = self.tile_index.get(&p.coord).map(|&i| &terrain.tiles[i]) else {
                continue;
            };
            let o = origin.to_render(tile.origin);
            let (hmin, hmax) = tile.height_bounds;
            let skirt = (hmax - hmin).abs().max(span * 0.05).max(1.0);
            raw.push(PatchRaw {
                o_span: [o.x, o.y, o.z, span],
                params: [p.morph, cells_at_lod(p.lod) as f32, res, skirt],
            });
        }
        if raw.is_empty() {
            return;
        }

        if self.instances.is_none() || self.instance_capacity < raw.len() {
            let capacity = raw.len().next_power_of_two().max(16);
            self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("terrain-instances"),
                size: (capacity * std::mem::size_of::<PatchRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_capacity = capacity;
        }
        let inst_buf = self.instances.as_ref().unwrap();
        gpu.queue
            .write_buffer(inst_buf, 0, bytemuck::cast_slice(&raw));
        let env_bg = self.env.bind_group(gpu, frame).clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(2, &self.material_bg, &[]);
        pass.set_bind_group(3, &env_bg, &[]);
        pass.set_vertex_buffer(1, inst_buf.slice(..));

        for (idx, patch) in patches.iter().enumerate() {
            let Some(tex) = self.textures.get(&patch.coord) else {
                continue;
            };
            let mesh = &self.lod_meshes[patch.lod.min(TERRAIN_LOD_COUNT - 1) as usize];
            pass.set_bind_group(1, &tex.bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            let i = idx as u32;
            pass.draw_indexed(0..mesh.index_count, 0, i..i + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn flat_terrain(ntiles: i32) -> RenderTerrain {
        let res = 16u32;
        let mps = 1.0;
        let span = (res - 1) as f64 * mps;
        let mut tiles = Vec::new();
        // Tuple-sorted order (tx outer, tz inner) — matches the BTreeMap the host
        // projects from, so the assembled patch order is `(i32,i32)`-sorted.
        for tx in 0..ntiles {
            for tz in 0..ntiles {
                tiles.push(RenderTerrainTile {
                    coord: (tx, tz),
                    origin: DVec3::new(tx as f64 * span, 0.0, tz as f64 * span),
                    heights: vec![0.0; (res * res) as usize],
                    weights: Vec::new(),
                    height_bounds: (0.0, 0.0),
                });
            }
        }
        RenderTerrain {
            tile_resolution: res,
            meters_per_sample: mps,
            tiles,
            layers: Default::default(),
            macro_variation: 0.0,
            version: 1,
        }
    }

    #[test]
    fn cells_halve_per_lod() {
        assert_eq!(cells_at_lod(0), 32);
        assert_eq!(cells_at_lod(1), 16);
        assert_eq!(cells_at_lod(2), 8);
        assert_eq!(cells_at_lod(3), 4);
        // Beyond the coarsest level clamps.
        assert_eq!(cells_at_lod(99), 4);
    }

    #[test]
    fn lod_rises_with_distance() {
        let span = 100.0;
        let t = lod_thresholds(span); // [150, 300, 600]
        assert_eq!(lod_for_distance(0.0, &t), 0);
        assert_eq!(lod_for_distance(149.0, &t), 0);
        assert_eq!(lod_for_distance(151.0, &t), 1);
        assert_eq!(lod_for_distance(301.0, &t), 2);
        assert_eq!(lod_for_distance(10_000.0, &t), TERRAIN_LOD_COUNT - 1);
    }

    #[test]
    fn morph_ramps_within_band_and_zero_at_coarsest() {
        let span = 100.0;
        let t = lod_thresholds(span); // [150, 300, 600]
                                      // Deep inside LOD-0 band: no morph.
        assert_eq!(morph_factor(10.0, 0, &t), 0.0);
        // Just below the LOD-0→1 threshold: fully morphed.
        assert!(morph_factor(149.9, 0, &t) > 0.99);
        // Mid-band monotonic 0→1.
        let a = morph_factor(120.0, 0, &t);
        let b = morph_factor(140.0, 0, &t);
        assert!((0.0..=1.0).contains(&a) && a <= b);
        // Coarsest LOD never morphs.
        assert_eq!(morph_factor(10_000.0, TERRAIN_LOD_COUNT - 1, &t), 0.0);
    }

    #[test]
    fn assembly_is_deterministic_and_sorted() {
        let terrain = flat_terrain(3);
        let view = top_view();
        let origin = FloatingOrigin::new(DVec3::ZERO);
        let a = assemble_patches(&terrain, &view, &origin);
        let b = assemble_patches(&terrain, &view, &origin);
        assert_eq!(a, b, "assembly must be deterministic");
        // Coordinates are non-decreasing (tiles arrive sorted).
        for w in a.windows(2) {
            assert!(w[0].coord <= w[1].coord);
        }
        assert!(!a.is_empty());
    }

    #[test]
    fn nearest_tile_is_finest_lod() {
        let terrain = flat_terrain(3);
        let view = top_view(); // eye over tile (0,0)
        let origin = FloatingOrigin::new(DVec3::ZERO);
        let patches = assemble_patches(&terrain, &view, &origin);
        let near = patches.iter().find(|p| p.coord == (0, 0)).unwrap();
        assert_eq!(near.lod, 0, "the tile under the camera should be LOD 0");
    }

    #[test]
    fn offscreen_tiles_are_culled() {
        // A single tile far behind the camera is culled out.
        let res = 16u32;
        let terrain = RenderTerrain {
            tile_resolution: res,
            meters_per_sample: 1.0,
            tiles: vec![RenderTerrainTile {
                coord: (0, 0),
                origin: DVec3::new(0.0, 0.0, 1000.0), // way behind a -Z camera
                heights: vec![0.0; (res * res) as usize],
                weights: Vec::new(),
                height_bounds: (0.0, 0.0),
            }],
            layers: Default::default(),
            macro_variation: 0.0,
            version: 1,
        };
        let view = RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(7.0, 30.0, -20.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
            ortho: None,
        };
        assert!(assemble_patches(&terrain, &view, &FloatingOrigin::new(DVec3::ZERO)).is_empty());
    }

    #[test]
    fn triplanar_weights_normalize_and_favour_dominant_axis() {
        // A flat-top normal (+Y) → the XZ (y) plane dominates and the weights sum
        // to 1.
        let w = triplanar_axis_weights(Vec3::Y, 4.0);
        let sum = w[0] + w[1] + w[2];
        assert!((sum - 1.0).abs() < 1e-5, "weights must sum to 1: {w:?}");
        assert!(
            w[1] > w[0] && w[1] > w[2],
            "top plane should dominate: {w:?}"
        );

        // A 45° X/Y normal with higher sharpness biases harder toward the larger
        // component than a low sharpness does.
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let soft = triplanar_axis_weights(n, 1.0);
        let hard = triplanar_axis_weights(n, 8.0);
        assert!((soft[0] - soft[1]).abs() < 1e-5, "equal axes stay equal");
        assert!((hard[0] - hard[1]).abs() < 1e-5);
        // A steep (mostly-X) normal: sharpness pushes weight onto the YZ (x) plane.
        let steep = Vec3::new(0.9, 0.2, 0.386).normalize();
        let s1 = triplanar_axis_weights(steep, 1.0);
        let s8 = triplanar_axis_weights(steep, 8.0);
        assert!(
            s8[0] > s1[0],
            "sharper blend concentrates on the dominant axis"
        );
    }

    #[test]
    fn triplanar_degenerate_normal_falls_back_to_top() {
        let w = triplanar_axis_weights(Vec3::ZERO, 4.0);
        assert_eq!(w, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn macro_modulation_scales_albedo() {
        let a = Vec3::new(0.4, 0.5, 0.6);
        // No fBm → unchanged.
        assert_eq!(macro_modulate(a, 0.0, 0.3), a);
        // Positive fBm brightens, negative darkens, symmetrically.
        let up = macro_modulate(a, 1.0, 0.2);
        let dn = macro_modulate(a, -1.0, 0.2);
        assert!(up.x > a.x && dn.x < a.x);
        assert!(
            (up + dn - a * 2.0).length() < 1e-6,
            "modulation is symmetric"
        );
    }

    #[test]
    fn material_raw_packs_layers_and_macro() {
        let mut terrain = flat_terrain(1);
        terrain.layers[2].albedo = [0.1, 0.2, 0.3, 1.0];
        terrain.layers[2].roughness = 0.25;
        terrain.layers[2].tex_scale = 3.0;
        terrain.macro_variation = 0.7;
        let raw = MaterialRaw::from_terrain(&terrain);
        assert_eq!(raw.albedo[2], [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(raw.params[2][0], 0.25);
        assert_eq!(raw.params[2][1], 3.0);
        assert_eq!(raw.macro_amp[0], 0.7);
    }

    #[test]
    fn lod_geometry_has_skirts() {
        // A 2-cell patch: 9 interior verts + 4 skirt strips of (3 verts each).
        let (verts, indices) = build_lod_geometry(2);
        // Interior 3×3 = 9, plus 4 edges × 3 skirt copies = 12 → 21.
        assert_eq!(verts.len(), 9 + 12);
        // Interior tris: 2×2 cells × 2 = 8 tris (24 idx). Skirts: each edge has 2
        // segments → 2 quads → 4 tris → 24 idx; ×4 edges = 48. Total 72.
        assert_eq!(indices.len(), 24 + 48);
        // Skirt verts carry the flag; interior verts don't.
        assert!(verts[..9].iter().all(|v| v[2] == 0.0));
        assert!(verts[9..].iter().all(|v| v[2] == 1.0));
    }

    /// A camera looking down over the origin tile.
    fn top_view() -> RenderView {
        RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(7.5, 25.0, 7.5),
            forward: Vec3::new(0.0, -1.0, 0.001).normalize(),
            up: Vec3::Z,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
            ortho: None,
        }
    }
}
