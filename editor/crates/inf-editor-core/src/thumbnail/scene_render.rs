//! A small, self-contained headless render pipeline for asset thumbnails.
//!
//! This is deliberately separate from `inf-render`'s `EngineRenderer` (which is
//! built for the interactive viewport's instanced primitives + reverse-Z depth).
//! A thumbnail is a one-shot lit render of a single mesh (or a procedural sphere
//! for materials) into an offscreen target, so a minimal forward pipeline with a
//! plain perspective camera is simpler and fully isolated.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
use inf_mesh::{Aabb, MeshVertex};
use inf_render::{GpuContext, HeadlessTarget, HEADLESS_FORMAT};
use wgpu::util::DeviceExt;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    light_dir: [f32; 4],
    base_color: [f32; 4],
}

const SHADER: &str = r#"
struct U {
  view_proj: mat4x4<f32>,
  normal_matrix: mat4x4<f32>,
  light_dir: vec4<f32>,
  base_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) normal: vec3<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>) -> VOut {
  var o: VOut;
  o.clip = u.view_proj * vec4<f32>(pos, 1.0);
  o.normal = (u.normal_matrix * vec4<f32>(nrm, 0.0)).xyz;
  return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
  let n = normalize(i.normal);
  let l = normalize(u.light_dir.xyz);
  let diff = max(dot(n, l), 0.0);
  let ambient = 0.28;
  let rim = pow(1.0 - max(dot(n, vec3<f32>(0.0, 0.0, 1.0)), 0.0), 3.0) * 0.15;
  let lit = u.base_color.rgb * (ambient + diff * 0.85) + vec3<f32>(rim);
  return vec4<f32>(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

/// Render a single mesh (interleaved [`MeshVertex`] + indices), framed by its
/// bounds, into a `size×size` RGBA8 image. Returns tightly-packed rows.
pub fn render_mesh(
    gpu: &GpuContext,
    size: u32,
    vertices: &[MeshVertex],
    indices: &[u32],
    bounds: Aabb,
    base_color: [f32; 4],
) -> Result<Vec<u8>, String> {
    if vertices.is_empty() || indices.is_empty() {
        return Err("empty mesh".into());
    }
    render(gpu, size, vertices, indices, bounds, base_color)
}

/// Render a procedural UV sphere shaded with `base_color` — the material preview.
pub fn render_sphere(gpu: &GpuContext, size: u32, base_color: [f32; 4]) -> Result<Vec<u8>, String> {
    let (verts, indices) = unit_sphere(48, 32);
    let bounds = Aabb {
        min: [-1.0; 3],
        max: [1.0; 3],
    };
    render(gpu, size, &verts, &indices, bounds, base_color)
}

fn render(
    gpu: &GpuContext,
    size: u32,
    vertices: &[MeshVertex],
    indices: &[u32],
    bounds: Aabb,
    base_color: [f32; 4],
) -> Result<Vec<u8>, String> {
    let device = &gpu.device;
    let target = HeadlessTarget::new(gpu, size, size);
    let depth = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("thumb-depth"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Camera: orbit the bounding sphere from a fixed 3/4 angle.
    let center = Vec3::from(bounds.center());
    let radius = bounds.radius().max(1e-3);
    let dir = Vec3::new(0.6, 0.45, 1.0).normalize();
    let eye = center + dir * radius * 2.6;
    let view = look_at_rh(eye, center, Vec3::Y);
    let proj = perspective_rh(45f32.to_radians(), 1.0, radius * 0.05, radius * 10.0);
    let view_proj = proj * view;
    let normal_matrix = Mat4::from_mat3(Mat3::from_mat4(view).inverse().transpose());

    let uniforms = Uniforms {
        view_proj: view_proj.to_cols_array_2d(),
        normal_matrix: normal_matrix.to_cols_array_2d(),
        // Light in view space (headlight-ish, from upper right).
        light_dir: [0.4, 0.7, 1.0, 0.0],
        base_color,
    };

    let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thumb-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thumb-vertices"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("thumb-indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("thumb-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("thumb-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("thumb-bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ubo.as_entire_binding(),
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("thumb-layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<MeshVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ],
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("thumb-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[Some(vertex_layout)],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: HEADLESS_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("thumb"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("thumb-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // Neutral studio-grey backdrop.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.12,
                        g: 0.12,
                        b: 0.14,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, vbo.slice(..));
        pass.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
    }
    gpu.queue.submit([encoder.finish()]);

    target.read_rgba(gpu)
}

// ── Material graph live preview (P7.2.3) ────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PreviewCam {
    view_proj: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    cam_pos: [f32; 4],
}

/// The preview vertex + PBR fragment that wraps a material graph's generated
/// `material_surface` fn (defined in the embedded `surface_wgsl`). Lit by a fixed
/// studio key + fill + hemispheric ambient and ACES-tonemapped — a compact
/// echo of the P7.1 scene BRDF so the preview reads like the real thing.
const PREVIEW_WRAPPER: &str = r#"
struct PvCam {
  view_proj: mat4x4<f32>,
  normal_matrix: mat4x4<f32>,
  cam_pos: vec4<f32>,
};
@group(0) @binding(0) var<uniform> pv_cam: PvCam;

struct PvOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) world_pos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>, @location(2) uv: vec2<f32>) -> PvOut {
  var o: PvOut;
  o.clip = pv_cam.view_proj * vec4<f32>(pos, 1.0);
  o.world_pos = pos;
  o.normal = (pv_cam.normal_matrix * vec4<f32>(nrm, 0.0)).xyz;
  o.uv = uv;
  return o;
}

const PV_PI: f32 = 3.14159265359;
fn pv_ggx(nh: f32, r: f32) -> f32 { let a = r*r; let a2 = a*a; let d = nh*nh*(a2-1.0)+1.0; return a2/max(PV_PI*d*d, 1e-7); }
fn pv_smith(nv: f32, nl: f32, r: f32) -> f32 { let k = (r+1.0)*(r+1.0)/8.0; let gv = nv/(nv*(1.0-k)+k); let gl = nl/(nl*(1.0-k)+k); return gv*gl; }
fn pv_fresnel(c: f32, f0: vec3<f32>) -> vec3<f32> { return f0 + (vec3<f32>(1.0)-f0)*pow(clamp(1.0-c,0.0,1.0),5.0); }
fn pv_aces(x: vec3<f32>) -> vec3<f32> { return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), vec3<f32>(0.0), vec3<f32>(1.0)); }

fn pv_light(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>, albedo: vec3<f32>, metallic: f32, rough: f32, f0: vec3<f32>) -> vec3<f32> {
  let nl = max(dot(n, l), 0.0);
  if (nl <= 0.0) { return vec3<f32>(0.0); }
  let h = normalize(v + l);
  let nv = max(dot(n, v), 1e-4);
  let nh = max(dot(n, h), 0.0);
  let vh = max(dot(v, h), 0.0);
  let d = pv_ggx(nh, rough);
  let g = pv_smith(nv, nl, rough);
  let f = pv_fresnel(vh, f0);
  let spec = (d*g)*f / max(4.0*nv*nl, 1e-4);
  let kd = (vec3<f32>(1.0)-f)*(1.0-metallic);
  return (kd*albedo/PV_PI + spec) * radiance * nl;
}

@fragment
fn fs(i: PvOut) -> @location(0) vec4<f32> {
  var mi: MatIn;
  mi.uv = i.uv;
  mi.normal = normalize(i.normal);
  mi.world_pos = i.world_pos;
  mi.time = 0.0;
  let s = material_surface(mi);

  let n = normalize(i.normal);
  let v = normalize(pv_cam.cam_pos.xyz - i.world_pos);
  let albedo = s.base_color;
  let metallic = clamp(s.metallic, 0.0, 1.0);
  let rough = clamp(s.roughness, 0.04, 1.0);
  let f0 = mix(vec3<f32>(0.04), albedo, metallic);

  var lo = vec3<f32>(0.0);
  lo += pv_light(n, v, normalize(vec3<f32>(0.5, 0.8, 0.6)), vec3<f32>(3.0), albedo, metallic, rough, f0);
  lo += pv_light(n, v, normalize(vec3<f32>(-0.6, 0.2, 0.4)), vec3<f32>(0.6, 0.7, 0.9), albedo, metallic, rough, f0);
  let up = clamp(n.y*0.5+0.5, 0.0, 1.0);
  let amb = mix(vec3<f32>(0.03,0.03,0.035), vec3<f32>(0.10,0.13,0.18), up);
  lo += amb*albedo*(1.0-metallic) + amb*f0*0.5 + s.emissive;

  return vec4<f32>(pv_aces(lo), 1.0);
}
"#;

/// Where the preview camera is looking from — **the only thing a mouse-orbit
/// changes between frames** (P23.2a).
///
/// Split out because that is the whole economics of [`PreviewSession`]: a
/// camera move must not rebuild a target, a depth buffer, a shader module or a
/// pipeline. Angles are degrees and `distance` is in sphere radii (the preview
/// mesh is a unit sphere), so a caller drags in units it can reason about.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreviewView {
    /// Rotation about +Y, degrees. 0 looks down −Z.
    pub yaw_deg: f32,
    /// Elevation, degrees. Clamped to ±89 so the up-vector never degenerates.
    pub pitch_deg: f32,
    /// Eye distance from the origin, in world units.
    pub distance: f32,
}

impl Default for PreviewView {
    /// The fixed 3/4 framing every material preview used before it could orbit —
    /// `dir = normalize(0.5, 0.35, 1.0)` at 2.8 units, expressed as angles.
    fn default() -> Self {
        let dir = Vec3::new(0.5, 0.35, 1.0).normalize();
        Self {
            yaw_deg: dir.x.atan2(dir.z).to_degrees(),
            pitch_deg: dir.y.asin().to_degrees(),
            distance: 2.8,
        }
    }
}

impl PreviewView {
    /// The eye position this view looks from (target is always the origin).
    fn eye(&self) -> Vec3 {
        let yaw = self.yaw_deg.to_radians();
        let pitch = self.pitch_deg.clamp(-89.0, 89.0).to_radians();
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy) * self.distance.max(1e-3)
    }
}

/// One compiled preview program: the pipeline plus, when the surface samples
/// textures, the white-bound `group(2)` resources it was laid out for.
///
/// Held behind an `Arc` so the cache can hand the same one out repeatedly and a
/// test can prove it did (`Arc::ptr_eq` — the `GenCache` precedent in
/// `inf_render::passes`).
pub(crate) struct PreviewProgram {
    pipeline: wgpu::RenderPipeline,
    /// `(texture, sampler, bind group)` — kept alive because the bind group
    /// references the first two. `None` when the surface samples nothing.
    textures: Option<(wgpu::Texture, wgpu::Sampler, wgpu::BindGroup)>,
}

/// How many distinct surfaces a session keeps compiled.
///
/// Bounded on purpose: a material graph recompiles on every edit, so an
/// unbounded cache would hold one pipeline per keystroke for the life of the
/// session. Eight covers "the graph I am editing, and the last few I looked at"
/// — which is what the cache is for — and evicts least-recently-used beyond it.
const PIPELINE_CACHE_CAP: usize = 8;

/// **A reusable, cached-pipeline offscreen preview** (P23.2a).
///
/// The seam an interactive Model-Editor panel drives with a mouse-orbit, and the
/// reason the P23.1 memo can rule that the first DCC preview is an offscreen
/// PNG rather than a second native viewport.
///
/// The free [`render_material_preview`] rebuilt **everything** per call — render
/// target, depth texture, vertex/index/uniform buffers, shader module, bind
/// group layouts, pipeline — which is right for a thumbnail rendered once and
/// cached to disk by content hash, and hopeless at interactive rates. A session
/// owns all of it and rebuilds only what actually changed:
///
/// * a **camera** move writes 144 bytes into an existing uniform buffer;
/// * a **surface** change compiles one shader and one pipeline, and finds it
///   already compiled if that surface has been seen (bounded by
///   [`PIPELINE_CACHE_CAP`]);
/// * a **size** change is a new session (the caller's business — `size` is
///   fixed for the session's life so the target and depth never reallocate).
///
/// It is deliberately NOT `Send`: wgpu resources are tied to their device and
/// the editor drives this from the command thread that owns the `Thumbnailer`.
pub struct PreviewSession {
    size: u32,
    target: HeadlessTarget,
    depth: wgpu::TextureView,
    vbo: wgpu::Buffer,
    ibo: wgpu::Buffer,
    index_count: u32,
    ubo: wgpu::Buffer,
    cam_bgl: wgpu::BindGroupLayout,
    cam_bg: wgpu::BindGroup,
    empty_bgl: wgpu::BindGroupLayout,
    empty_bg: wgpu::BindGroup,
    /// MRU-first: `(shader hash, tex_count, source, program)`.
    ///
    /// The source is stored beside its hash and compared on a hit. A 64-bit
    /// collision is vanishingly unlikely and its consequence is not — it would
    /// draw a *different material* with total confidence, which is exactly the
    /// class of failure this codebase keeps paying for. The compare only runs on
    /// a hash hit, so it costs nothing on a miss.
    programs: Vec<(u64, u32, String, Arc<PreviewProgram>)>,
}

/// Hash a surface for the pipeline cache key.
fn surface_hash(surface_wgsl: &str, tex_count: u32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    surface_wgsl.hash(&mut h);
    tex_count.hash(&mut h);
    h.finish()
}

impl PreviewSession {
    /// Build a session rendering `size × size` frames on `gpu`.
    ///
    /// Everything size- and mesh-dependent is allocated here, once. The sphere
    /// is the same 64×48 UV sphere the one-shot preview used, so a session
    /// render and a one-shot render of the same surface are the same image.
    pub fn new(gpu: &GpuContext, size: u32) -> Self {
        let device = &gpu.device;
        let (verts, indices) = unit_sphere(64, 48);

        let target = HeadlessTarget::new(gpu, size, size);
        let depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("preview-depth"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());

        // COPY_DST, unlike the one-shot's init-only buffer: the camera is
        // rewritten per frame and that is the warm path's entire cost.
        let ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview-cam"),
            size: std::mem::size_of::<PreviewCam>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview-vtx"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview-idx"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview-cam-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview-cam-bg"),
            layout: &cam_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubo.as_entire_binding(),
            }],
        });
        // group(1) is unused by the shader but must exist (and be bound) to
        // reach group(2) — an empty layout + empty bind group.
        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview-empty-bgl"),
            entries: &[],
        });
        let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview-empty-bg"),
            layout: &empty_bgl,
            entries: &[],
        });

        Self {
            size,
            target,
            depth,
            vbo,
            ibo,
            index_count: indices.len() as u32,
            ubo,
            cam_bgl,
            cam_bg,
            empty_bgl,
            empty_bg,
            programs: Vec::new(),
        }
    }

    /// The edge length this session renders at. A caller wanting another size
    /// builds another session (the target and depth are fixed for its life).
    pub fn size(&self) -> u32 {
        self.size
    }

    /// How many distinct surfaces are currently compiled — the cache's
    /// observable state, and what its bound is asserted against.
    pub fn cached_programs(&self) -> usize {
        self.programs.len()
    }

    /// The compiled program for `surface_wgsl`, building it on a miss and moving
    /// it to the front of the MRU list on a hit.
    pub(crate) fn program(
        &mut self,
        gpu: &GpuContext,
        surface_wgsl: &str,
        tex_count: u32,
    ) -> Arc<PreviewProgram> {
        let key = surface_hash(surface_wgsl, tex_count);
        if let Some(i) = self
            .programs
            .iter()
            .position(|(k, t, src, _)| *k == key && *t == tex_count && src == surface_wgsl)
        {
            let entry = self.programs.remove(i);
            let program = entry.3.clone();
            self.programs.insert(0, entry);
            return program;
        }
        let program = Arc::new(self.build_program(gpu, surface_wgsl, tex_count));
        self.programs.insert(
            0,
            (key, tex_count, surface_wgsl.to_string(), program.clone()),
        );
        self.programs.truncate(PIPELINE_CACHE_CAP);
        program
    }

    fn build_program(
        &self,
        gpu: &GpuContext,
        surface_wgsl: &str,
        tex_count: u32,
    ) -> PreviewProgram {
        let device = &gpu.device;
        let source = format!("{surface_wgsl}\n{PREVIEW_WRAPPER}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preview-shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        // Texture group(2): a shared white 1×1 + sampler bound to every slot.
        let textures = (tex_count > 0).then(|| build_white_textures(gpu, tex_count));
        let tex_bgl = textures.as_ref().map(|(_, _, _, bgl)| bgl);
        let layout_bgls: Vec<Option<&wgpu::BindGroupLayout>> = match tex_bgl {
            Some(bgl) => vec![Some(&self.cam_bgl), Some(&self.empty_bgl), Some(bgl)],
            None => vec![Some(&self.cam_bgl)],
        };
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview-layout"),
            bind_group_layouts: &layout_bgls,
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("preview-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Some(preview_vertex_layout())],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HEADLESS_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        PreviewProgram {
            pipeline,
            textures: textures.map(|(t, s, bg, _)| (t, s, bg)),
        }
    }

    /// Render one frame of `surface_wgsl` from `view`, returning tightly-packed
    /// RGBA8 rows.
    ///
    /// `tex_count` texture slots are bound to a shared white 1×1 (binding real
    /// referenced textures is the standing P7 follow-up). The caller must have
    /// naga-validated the surface (see `inf_material::emit_wgsl`).
    ///
    /// **Deterministic within a process**: the same `(surface, tex_count, view)`
    /// produces byte-identical output, which is what makes an orbit's frames
    /// comparable and a cache hit provable. It is NOT a cross-machine claim —
    /// this is a preview image, never committed content, so the P14 trig law
    /// does not reach it (nothing here is serialized, hashed into an asset, or
    /// compared between two machines).
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        surface_wgsl: &str,
        tex_count: u32,
        view: PreviewView,
    ) -> Result<Vec<u8>, String> {
        let program = self.program(gpu, surface_wgsl, tex_count);

        let eye = view.eye();
        let look = look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let proj = perspective_rh(40f32.to_radians(), 1.0, 0.05, 20.0);
        let cam = PreviewCam {
            view_proj: (proj * look).to_cols_array_2d(),
            normal_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, 1.0],
        };
        gpu.queue
            .write_buffer(&self.ubo, 0, bytemuck::bytes_of(&cam));

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("preview"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("preview-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.10,
                            g: 0.10,
                            b: 0.12,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&program.pipeline);
            pass.set_bind_group(0, &self.cam_bg, &[]);
            if let Some((_, _, tex_bg)) = &program.textures {
                pass.set_bind_group(1, &self.empty_bg, &[]);
                pass.set_bind_group(2, tex_bg, &[]);
            }
            pass.set_vertex_buffer(0, self.vbo.slice(..));
            pass.set_index_buffer(self.ibo.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);
        self.target.read_rgba(gpu)
    }
}

/// The preview sphere's interleaved vertex layout (position / normal / uv).
fn preview_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<MeshVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 24,
                shader_location: 2,
            },
        ],
    }
}

// The free `render_material_preview` that used to live here is GONE (P23.2a).
// It rebuilt the target, the depth texture, the sphere buffers, the shader
// module, three bind-group layouts and the pipeline on **every call** — right
// for a one-shot thumbnail, hopeless for a panel that orbits. Its one caller
// (`Thumbnailer::render_material_preview`) now drives a persistent
// [`PreviewSession`]; the image is unchanged, and
// `the_default_view_reproduces_the_historical_framing` pins that claim without
// needing a GPU.

/// Build the shared white 1×1 texture + sampler and a bind group binding them to
/// every `tex.sample` slot (`2k` = texture, `2k+1` = sampler).
fn build_white_textures(
    gpu: &GpuContext,
    tex_count: u32,
) -> (
    wgpu::Texture,
    wgpu::Sampler,
    wgpu::BindGroup,
    wgpu::BindGroupLayout,
) {
    let device = &gpu.device;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("matprev-white"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255u8, 255, 255, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("matprev-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let mut entries = Vec::new();
    let mut bg_entries = Vec::new();
    for k in 0..tex_count {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: k * 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: k * 2 + 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matprev-tex-bgl"),
        entries: &entries,
    });
    for k in 0..tex_count {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: k * 2,
            resource: wgpu::BindingResource::TextureView(&view),
        });
        bg_entries.push(wgpu::BindGroupEntry {
            binding: k * 2 + 1,
            resource: wgpu::BindingResource::Sampler(&sampler),
        });
    }
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matprev-tex-bg"),
        layout: &bgl,
        entries: &bg_entries,
    });
    (tex, sampler, bg, bgl)
}

// ── Texture graph bake (P7.3) ───────────────────────────────────────────────

/// Run a material/texture graph's generated `compute_wgsl` (entry `cs_bake`) over
/// a `size × size` storage texture and read back tightly-packed RGBA8 rows.
/// `tex_count` `tex.sample` slots are bound to a shared white 1×1. The compute
/// module must have been naga-validated (`inf_material::emit_texture_compute`).
pub fn bake_texture(
    gpu: &GpuContext,
    size: u32,
    compute_wgsl: &str,
    tex_count: u32,
) -> Result<Vec<u8>, String> {
    let device = &gpu.device;

    let out_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bake-out"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bake-shader"),
        source: wgpu::ShaderSource::Wgsl(compute_wgsl.into()),
    });

    let out_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bake-out-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }],
    });
    let out_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bake-out-bg"),
        layout: &out_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&out_view),
        }],
    });

    let tex_resources = (tex_count > 0).then(|| build_white_textures(gpu, tex_count));
    let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bake-empty-bgl"),
        entries: &[],
    });
    let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bake-empty-bg"),
        layout: &empty_bgl,
        entries: &[],
    });
    let layout_bgls: Vec<Option<&wgpu::BindGroupLayout>> = match &tex_resources {
        Some((_, _, _, tex_bgl)) => vec![Some(&out_bgl), Some(&empty_bgl), Some(tex_bgl)],
        None => vec![Some(&out_bgl)],
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bake-layout"),
        bind_group_layouts: &layout_bgls,
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bake-pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("cs_bake"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bake"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bake-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &out_bg, &[]);
        if let Some((_, _, tex_bg, _)) = &tex_resources {
            pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, tex_bg, &[]);
        }
        let groups = size.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
    }

    // Read back the storage texture (256-aligned rows), unpadding into RGBA8.
    let padded = align_up(size * 4, 256);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bake-readback"),
        size: (padded * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
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
    let row = (size * 4) as usize;
    let mut out = Vec::with_capacity(row * size as usize);
    for chunk in data.chunks(padded as usize) {
        out.extend_from_slice(&chunk[..row]);
    }
    drop(data);
    readback.unmap();
    Ok(out)
}

fn align_up(v: u32, align: u32) -> u32 {
    v.div_ceil(align) * align
}

/// Right-handed look-at (glam's built-in is deprecated in 0.33; we hand-roll to
/// stay `-D warnings` clean and keep the depth convention explicit).
fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols(
        Vec4::new(s.x, u.x, -f.x, 0.0),
        Vec4::new(s.y, u.y, -f.y, 0.0),
        Vec4::new(s.z, u.z, -f.z, 0.0),
        Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// Right-handed perspective with a `[0, 1]` clip-space depth (wgpu convention),
/// looking down −Z.
fn perspective_rh(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let g = 1.0 / (fovy * 0.5).tan();
    Mat4::from_cols(
        Vec4::new(g / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, g, 0.0, 0.0),
        Vec4::new(0.0, 0.0, far / (near - far), -1.0),
        Vec4::new(0.0, 0.0, near * far / (near - far), 0.0),
    )
}

/// A unit UV sphere (radius 1), returned as interleaved vertices + indices.
fn unit_sphere(sectors: u32, stacks: u32) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    for i in 0..=stacks {
        let phi = std::f32::consts::PI * i as f32 / stacks as f32;
        let (sp, cp) = phi.sin_cos();
        for j in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
            let (st, ct) = theta.sin_cos();
            let n = [sp * ct, cp, sp * st];
            verts.push(MeshVertex {
                position: n,
                normal: n,
                uv: [j as f32 / sectors as f32, i as f32 / stacks as f32],
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
        }
    }
    let mut indices = Vec::new();
    let row = sectors + 1;
    for i in 0..stacks {
        for j in 0..sectors {
            let a = i * row + j;
            let b = a + row;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (verts, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_SURFACE: &str = "\
struct MatIn { uv: vec2<f32>, normal: vec3<f32>, world_pos: vec3<f32>, time: f32 };
struct Surface { base_color: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32> };
fn material_surface(mi: MatIn) -> Surface {
    var surf: Surface;
    surf.base_color = vec3<f32>(mi.uv, 0.4);
    surf.metallic = 0.1;
    surf.roughness = 0.5;
    surf.emissive = vec3<f32>(0.0);
    return surf;
}
";

    const TEX_SURFACE: &str = "\
struct MatIn { uv: vec2<f32>, normal: vec3<f32>, world_pos: vec3<f32>, time: f32 };
struct Surface { base_color: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32> };
@group(2) @binding(0) var mat_tex_0: texture_2d<f32>;
@group(2) @binding(1) var mat_samp_0: sampler;
fn material_surface(mi: MatIn) -> Surface {
    var surf: Surface;
    surf.base_color = textureSampleLevel(mat_tex_0, mat_samp_0, mi.uv, 0.0).rgb;
    surf.metallic = 0.0;
    surf.roughness = 0.6;
    surf.emissive = vec3<f32>(0.0);
    return surf;
}
";

    // A P13.2 layered-slab surface: `material_surface` resolves a MatSlab tree
    // and unpacks it into the four channels — exactly what inf-material emits for
    // a `slab.*` graph. Verifies the preview wrapper embeds the MatSlab struct +
    // helper cleanly (the `material_surface -> Surface` interface is unchanged).
    const SLAB_SURFACE: &str = "\
struct MatIn { uv: vec2<f32>, normal: vec3<f32>, world_pos: vec3<f32>, time: f32 };
struct Surface { base_color: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32> };
struct MatSlab { albedo: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32> };
fn mat_slab_mix(a: MatSlab, b: MatSlab, t: f32) -> MatSlab {
    return MatSlab(mix(a.albedo, b.albedo, t), mix(a.metallic, b.metallic, t), mix(a.roughness, b.roughness, t), mix(a.emissive, b.emissive, t));
}
fn material_surface(mi: MatIn) -> Surface {
    let v0: MatSlab = mat_slab_mix(MatSlab(vec3<f32>(0.9, 0.1, 0.1), 0.0, 0.3, vec3<f32>(0.0)), MatSlab(vec3<f32>(0.1, 0.1, 0.9), 1.0, 0.7, vec3<f32>(0.0)), mi.uv.x);
    var surf: Surface;
    surf.base_color = v0.albedo;
    surf.metallic = v0.metallic;
    surf.roughness = v0.roughness;
    surf.emissive = v0.emissive;
    return surf;
}
";

    fn gpu_or_skip() -> Option<GpuContext> {
        GpuContext::headless().ok()
    }

    /// Render one preview frame at the default framing — the shape the three
    /// legacy assertions below were written against, now through the session.
    fn preview(gpu: &GpuContext, size: u32, surface: &str, tex_count: u32) -> Vec<u8> {
        PreviewSession::new(gpu, size)
            .render(gpu, surface, tex_count, PreviewView::default())
            .expect("preview render")
    }

    #[test]
    fn material_preview_renders_without_textures() {
        let Some(gpu) = gpu_or_skip() else { return };
        let img = preview(&gpu, 32, MIN_SURFACE, 0);
        assert_eq!(img.len(), 32 * 32 * 4);
        // Not entirely the background clear color (the sphere is drawn).
        assert!(img.chunks(4).any(|p| p[0] > 20 && p[1] > 20));
    }

    #[test]
    fn material_preview_renders_with_texture_binding() {
        let Some(gpu) = gpu_or_skip() else { return };
        // Exercises the empty-group(1) + white-texture-group(2) path.
        let img = preview(&gpu, 32, TEX_SURFACE, 1);
        assert_eq!(img.len(), 32 * 32 * 4);
    }

    #[test]
    fn material_preview_renders_layered_slab_surface() {
        let Some(gpu) = gpu_or_skip() else { return };
        // P13.2: a slab-based surface embeds cleanly in the preview wrapper.
        let img = preview(&gpu, 32, SLAB_SURFACE, 0);
        assert_eq!(img.len(), 32 * 32 * 4);
        assert!(img.chunks(4).any(|p| p[0] > 20 || p[2] > 20));
    }

    const BAKE_COMPUTE: &str = "\
struct MatIn { uv: vec2<f32>, normal: vec3<f32>, world_pos: vec3<f32>, time: f32 };
struct Surface { base_color: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32> };
fn material_surface(mi: MatIn) -> Surface {
    var surf: Surface;
    surf.base_color = vec3<f32>(mi.uv.x, mi.uv.y, 0.5);
    surf.metallic = 0.0;
    surf.roughness = 0.5;
    surf.emissive = vec3<f32>(0.0);
    return surf;
}
@group(0) @binding(0) var bake_out: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn cs_bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(bake_out);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(dims.x), f32(dims.y));
    var mi: MatIn;
    mi.uv = uv;
    mi.normal = vec3<f32>(0.0, 0.0, 1.0);
    mi.world_pos = vec3<f32>(uv, 0.0);
    mi.time = 0.0;
    let s = material_surface(mi);
    textureStore(bake_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(s.base_color, 1.0));
}
";

    // ── PreviewSession (P23.2a) ─────────────────────────────────────────────

    /// **Determinism**: the same `(surface, tex_count, view)` renders the same
    /// bytes, and a *different* view renders different ones.
    ///
    /// Both halves matter together. The first is what makes an orbit's frames
    /// comparable at all and what proves the cached pipeline is the same
    /// program; the second is the non-vacuity guard — a session that returned a
    /// stale readback, or ignored the camera entirely, would pass the first
    /// claim perfectly and be a preview that does not move.
    #[test]
    fn the_preview_session_is_deterministic_and_the_camera_moves_it() {
        let Some(gpu) = gpu_or_skip() else { return };
        let mut session = PreviewSession::new(&gpu, 64);
        let a = PreviewView::default();

        let first = session.render(&gpu, MIN_SURFACE, 0, a).expect("first");
        let again = session.render(&gpu, MIN_SURFACE, 0, a).expect("second");
        assert_eq!(first, again, "the same parameters rendered different bytes");
        assert_eq!(first.len(), 64 * 64 * 4);

        let orbited = session
            .render(
                &gpu,
                MIN_SURFACE,
                0,
                PreviewView {
                    yaw_deg: a.yaw_deg + 90.0,
                    ..a
                },
            )
            .expect("orbited");
        assert_ne!(
            first, orbited,
            "a 90° yaw produced an identical image — the camera is not reaching \
             the shader, so every determinism claim above is vacuous"
        );
    }

    /// **The extraction changed the cost, not the picture.**
    ///
    /// The pre-P23.2a preview looked from a hard-coded
    /// `normalize(0.5, 0.35, 1.0) * 2.8`; the session takes yaw/pitch/distance
    /// so a panel can orbit. `PreviewView::default` is that same eye, and this
    /// pins it — on the CPU, so it runs on every CI leg (there is no adapter
    /// there, which is exactly where a silently re-framed preview would ship).
    ///
    /// Tolerance is f32 round-trip through `atan2`/`asin` and back, not
    /// "roughly the same view": 1e-5 is about six significant figures.
    #[test]
    fn the_default_view_reproduces_the_historical_framing() {
        let historical = Vec3::new(0.5, 0.35, 1.0).normalize() * 2.8;
        let eye = PreviewView::default().eye();
        assert!(
            (eye - historical).length() < 1e-5,
            "the default preview camera moved: {eye:?} vs the historical \
             {historical:?} — every material thumbnail in every project would \
             re-render from a different angle"
        );
    }

    /// **The pipeline cache is a cache** — pointer identity, on the `GenCache`
    /// precedent (`inf_render::passes::the_cloud_bake_key_rebuilds_on_a_generation_bump`).
    ///
    /// A `render` that quietly recompiled every frame would produce identical
    /// pixels and pass every other test in this file while being exactly the
    /// thing `PreviewSession` exists to avoid. `Arc::ptr_eq` is the only claim
    /// that can tell the two apart.
    #[test]
    fn the_pipeline_cache_hands_back_the_same_program() {
        let Some(gpu) = gpu_or_skip() else { return };
        let mut session = PreviewSession::new(&gpu, 32);

        let first = session.program(&gpu, MIN_SURFACE, 0);
        let again = session.program(&gpu, MIN_SURFACE, 0);
        assert!(
            Arc::ptr_eq(&first, &again),
            "an unchanged surface recompiled its pipeline"
        );
        assert_eq!(session.cached_programs(), 1);

        // A changed surface must NOT reuse it — the other direction, and the
        // failure that would draw the previous material with total confidence.
        let other = session.program(&gpu, SLAB_SURFACE, 0);
        assert!(!Arc::ptr_eq(&other, &first));
        assert_eq!(session.cached_programs(), 2);

        // …and the same surface at a different `tex_count` is a different
        // pipeline LAYOUT, so it may not share either.
        let textured = session.program(&gpu, TEX_SURFACE, 1);
        assert!(!Arc::ptr_eq(&textured, &first));

        // The original is still cached (MRU, not overwritten).
        assert!(Arc::ptr_eq(&session.program(&gpu, MIN_SURFACE, 0), &first));
    }

    /// The cache is BOUNDED: a material graph recompiles on every keystroke, so
    /// an unbounded one would hold a pipeline per edit for the whole session.
    #[test]
    fn the_pipeline_cache_evicts_beyond_its_cap() {
        let Some(gpu) = gpu_or_skip() else { return };
        let mut session = PreviewSession::new(&gpu, 32);
        // Distinct surfaces: a trailing comment changes the source (and so the
        // hash) without changing what is drawn.
        for i in 0..(PIPELINE_CACHE_CAP + 3) {
            let src = format!("{MIN_SURFACE}\n// variant {i}\n");
            let _ = session.program(&gpu, &src, 0);
        }
        assert_eq!(session.cached_programs(), PIPELINE_CACHE_CAP);
    }

    /// **THE MEASUREMENT** (P23.1's "measured latency"; P23.2a deliverable 4).
    ///
    /// Cold = a fresh session's first frame (target + depth + buffers + shader +
    /// pipeline). Warm = a re-render with only the camera moved, which is what a
    /// mouse-orbit does. Both include the readback, because the offscreen path's
    /// whole cost is "GPU work + map + copy to a PNG the webview can show".
    ///
    /// Asserts only what is *structurally* true (warm is not slower than cold by
    /// construction — it does strictly less work) and prints the numbers, which
    /// is what the memo quotes. A hard millisecond bound here would be a budget
    /// test on unknown hardware, and this file has no adapter on CI at all.
    ///
    /// **The first render in a PROCESS is not a cold session render**, and
    /// conflating them is how a measurement lies: the very first draw pays for
    /// driver warm-up, the shader toolchain and the first allocation of every
    /// pool, none of which recur. So the GPU is primed first and discarded, and
    /// the process-cold figure is reported separately rather than folded in.
    #[test]
    fn preview_session_cold_versus_warm_latency() {
        let Some(gpu) = gpu_or_skip() else { return };

        // Process-cold: the first render this process has ever done. Reported
        // once, and never counted as a session cost.
        let t0 = std::time::Instant::now();
        let _ = PreviewSession::new(&gpu, 256)
            .render(&gpu, MIN_SURFACE, 0, PreviewView::default())
            .expect("process-cold render");
        eprintln!(
            "PREVIEW LATENCY process-cold 256x256: {:.2} ms (driver + shader \
             toolchain warm-up; paid once per editor session)",
            t0.elapsed().as_secs_f64() * 1e3
        );

        for size in [256u32, 512] {
            // Session-cold: a NEW session on a warm process — a fresh target,
            // depth, buffers, shader and pipeline. What opening a Model Editor
            // on a new asset costs.
            let t0 = std::time::Instant::now();
            let mut session = PreviewSession::new(&gpu, size);
            let _ = session
                .render(&gpu, MIN_SURFACE, 0, PreviewView::default())
                .expect("cold render");
            let cold = t0.elapsed();

            // Warm: camera only, five frames, best-of (an orbit is judged by the
            // frame it usually delivers, not by the one that hit a scheduler).
            let mut warm = std::time::Duration::MAX;
            for i in 1..=5 {
                let view = PreviewView {
                    yaw_deg: 30.0 + i as f32 * 7.0,
                    ..PreviewView::default()
                };
                let t = std::time::Instant::now();
                let _ = session.render(&gpu, MIN_SURFACE, 0, view).expect("warm");
                warm = warm.min(t.elapsed());
            }
            // …and the rest of the round trip the panel actually pays, because
            // an offscreen preview reaches the webview as a PNG data URL. A
            // number that stopped at `read_rgba` would be measuring the half of
            // the loop that is already fast and calling it the loop.
            let frame = session
                .render(&gpu, MIN_SURFACE, 0, PreviewView::default())
                .expect("frame to encode");
            let mut encode = std::time::Duration::MAX;
            let mut encode_fast = std::time::Duration::MAX;
            let mut bytes = 0usize;
            let mut bytes_fast = 0usize;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let png = crate::thumbnail::encode_png(size, &frame).expect("png");
                encode = encode.min(t.elapsed());
                bytes = png.len();

                let t = std::time::Instant::now();
                let png = crate::thumbnail::encode_png_fast(size, &frame).expect("png fast");
                encode_fast = encode_fast.min(t.elapsed());
                bytes_fast = png.len();
            }
            eprintln!(
                "PREVIEW LATENCY {size}x{size}: session-cold {:.2} ms, warm {:.2} ms, \
                 png {:.2} ms ({bytes} B) / png-fast {:.2} ms ({bytes_fast} B) \
                 => orbit frame {:.2} ms",
                cold.as_secs_f64() * 1e3,
                warm.as_secs_f64() * 1e3,
                encode.as_secs_f64() * 1e3,
                encode_fast.as_secs_f64() * 1e3,
                (warm + encode_fast).as_secs_f64() * 1e3
            );
            assert!(
                warm <= cold,
                "warm re-render ({warm:?}) was slower than the cold first frame \
                 ({cold:?}) at {size}x{size} — the session is rebuilding something \
                 it was supposed to keep"
            );
        }
    }

    #[test]
    fn bake_texture_writes_a_gradient() {
        let Some(gpu) = gpu_or_skip() else { return };
        let img = bake_texture(&gpu, 16, BAKE_COMPUTE, 0).expect("bake");
        assert_eq!(img.len(), 16 * 16 * 4);
        // The R channel grows across the row (uv.x gradient): first texel < last.
        let first_r = img[0];
        let last_r = img[(15 * 4) as usize];
        assert!(last_r > first_r, "expected a horizontal gradient in R");
    }
}
