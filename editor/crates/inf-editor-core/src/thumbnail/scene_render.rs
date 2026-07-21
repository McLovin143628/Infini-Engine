//! A small, self-contained headless render pipeline for asset thumbnails.
//!
//! This is deliberately separate from `inf-render`'s `EngineRenderer` (which is
//! built for the interactive viewport's instanced primitives + reverse-Z depth).
//! A thumbnail is a one-shot lit render of a single mesh (or a procedural sphere
//! for materials) into an offscreen target, so a minimal forward pipeline with a
//! plain perspective camera is simpler and fully isolated.

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

/// Render a material graph's generated `surface_wgsl` on a lit preview sphere,
/// returning tightly-packed RGBA8 rows. `tex_count` texture slots are bound to a
/// shared white 1×1 (binding real referenced textures is a follow-up). The
/// caller must have naga-validated the surface (see `inf_material::emit_wgsl`).
pub fn render_material_preview(
    gpu: &GpuContext,
    size: u32,
    surface_wgsl: &str,
    tex_count: u32,
) -> Result<Vec<u8>, String> {
    let device = &gpu.device;
    let (verts, indices) = unit_sphere(64, 48);

    let target = HeadlessTarget::new(gpu, size, size);
    let depth = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("matprev-depth"),
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

    let center = Vec3::ZERO;
    let dir = Vec3::new(0.5, 0.35, 1.0).normalize();
    let eye = center + dir * 2.8;
    let view = look_at_rh(eye, center, Vec3::Y);
    let proj = perspective_rh(40f32.to_radians(), 1.0, 0.05, 20.0);
    let cam = PreviewCam {
        view_proj: (proj * view).to_cols_array_2d(),
        normal_matrix: Mat4::IDENTITY.to_cols_array_2d(),
        cam_pos: [eye.x, eye.y, eye.z, 1.0],
    };

    let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matprev-cam"),
        contents: bytemuck::bytes_of(&cam),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matprev-vtx"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matprev-idx"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let source = format!("{surface_wgsl}\n{PREVIEW_WRAPPER}");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matprev-shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matprev-cam-bgl"),
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
        label: Some("matprev-cam-bg"),
        layout: &cam_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ubo.as_entire_binding(),
        }],
    });

    // Texture group(2): a shared white 1×1 + sampler bound to every slot.
    let tex_resources = (tex_count > 0).then(|| build_white_textures(gpu, tex_count));
    // group(1) is unused by the shader but must exist (and be bound) to reach
    // group(2) — an empty layout + empty bind group.
    let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matprev-empty-bgl"),
        entries: &[],
    });
    let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matprev-empty-bg"),
        layout: &empty_bgl,
        entries: &[],
    });
    let layout_bgls: Vec<Option<&wgpu::BindGroupLayout>> = match &tex_resources {
        Some((_, _, _, tex_bgl)) => vec![Some(&cam_bgl), Some(&empty_bgl), Some(tex_bgl)],
        None => vec![Some(&cam_bgl)],
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("matprev-layout"),
        bind_group_layouts: &layout_bgls,
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
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 24,
                shader_location: 2,
            },
        ],
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("matprev-pipeline"),
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
        label: Some("matprev"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("matprev-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
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
        pass.set_bind_group(0, &cam_bg, &[]);
        if let Some((_, _, tex_bg, _)) = &tex_resources {
            pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, tex_bg, &[]);
        }
        pass.set_vertex_buffer(0, vbo.slice(..));
        pass.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
    }
    gpu.queue.submit([encoder.finish()]);
    target.read_rgba(gpu)
}

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

    fn gpu_or_skip() -> Option<GpuContext> {
        GpuContext::headless().ok()
    }

    #[test]
    fn material_preview_renders_without_textures() {
        let Some(gpu) = gpu_or_skip() else { return };
        let img = render_material_preview(&gpu, 32, MIN_SURFACE, 0).expect("preview render");
        assert_eq!(img.len(), 32 * 32 * 4);
        // Not entirely the background clear color (the sphere is drawn).
        assert!(img.chunks(4).any(|p| p[0] > 20 && p[1] > 20));
    }

    #[test]
    fn material_preview_renders_with_texture_binding() {
        let Some(gpu) = gpu_or_skip() else { return };
        // Exercises the empty-group(1) + white-texture-group(2) path.
        let img = render_material_preview(&gpu, 32, TEX_SURFACE, 1).expect("textured preview");
        assert_eq!(img.len(), 32 * 32 * 4);
    }
}
