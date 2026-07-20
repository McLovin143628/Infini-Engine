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
