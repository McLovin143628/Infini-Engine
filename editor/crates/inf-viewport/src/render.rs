//! Spike-A renderer: wgpu surface on a raw HWND drawing an infinite ground
//! grid plus a spinning triangle, viewed through a flycam. Proves surface
//! lifecycle, resize, vsync pacing, and camera plumbing; replaced by
//! `inf-render` integration in Phase 2.

use std::time::Instant;

use glam::{Mat4, Vec3};

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz = camera position (render-local space), w unused.
    cam_pos: vec4<f32>,
    // x = triangle spin angle (radians), yzw unused.
    misc: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

// ── infinite ground grid: fullscreen triangle, ray-vs-y=0 plane ──

struct GridOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_grid(@builtin(vertex_index) i: u32) -> GridOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    var out: GridOut;
    out.pos = vec4<f32>(p[i], 0.0, 1.0);
    out.ndc = p[i];
    return out;
}

fn unproject(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let p = u.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return p.xyz / p.w;
}

@fragment
fn fs_grid(in: GridOut) -> @location(0) vec4<f32> {
    let near = unproject(in.ndc, 0.0);
    let far = unproject(in.ndc, 1.0);
    let dir = far - near;
    let t = -near.y / dir.y;
    if (t <= 0.0) {
        return vec4<f32>(0.0);
    }
    let wp = near + dir * t;

    // Anti-aliased line mask at 1 m and 10 m spacing (UE-style dual grid).
    let coord = wp.xz;
    let d = max(fwidth(coord), vec2<f32>(1e-6));
    let g1 = abs(fract(coord - 0.5) - 0.5) / d;
    let l1 = 1.0 - min(min(g1.x, g1.y), 1.0);
    let g10 = abs(fract(coord / 10.0 - 0.5) - 0.5) / (d / 10.0);
    let l10 = 1.0 - min(min(g10.x, g10.y), 1.0);

    let dist = length(wp - u.cam_pos.xyz);
    let fade = exp(-dist * 0.018);

    var col = vec3<f32>(0.30, 0.33, 0.38);
    var a = max(l1 * 0.30, l10 * 0.75) * fade;

    // World axes through the origin: X in red, Z in blue.
    if (abs(wp.z) < d.y * 1.5) {
        col = vec3<f32>(0.95, 0.28, 0.30);
        a = max(a, 0.85 * fade);
    }
    if (abs(wp.x) < d.x * 1.5) {
        col = vec3<f32>(0.25, 0.45, 1.0);
        a = max(a, 0.85 * fade);
    }
    // Premultiplied alpha.
    return vec4<f32>(col * a, a);
}

// ── spinning triangle standing at the world origin ──

@vertex
fn vs_tri(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec3<f32>, 3>(
        vec3<f32>(0.0, 1.05, 0.0),
        vec3<f32>(-0.55, 0.0, 0.0),
        vec3<f32>(0.55, 0.0, 0.0),
    );
    let a = u.misc.x;
    let c = cos(a);
    let s = sin(a);
    let v = p[i];
    let w = vec3<f32>(v.x * c, v.y, -v.x * s);
    return u.view_proj * vec4<f32>(w, 1.0);
}

@fragment
fn fs_tri() -> @location(0) vec4<f32> {
    // Editor accent blue (#3d8bff), linearized-ish.
    return vec4<f32>(0.05, 0.26, 1.0, 1.0);
}
"#;

/// Flycam state owned by the viewport thread; converted to matrices here.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub pos: Vec3,
    /// Radians around +Y; 0 looks down -Z.
    pub yaw: f32,
    /// Radians; positive looks up. Clamped by the caller.
    pub pitch: f32,
}

impl Default for Camera {
    fn default() -> Self {
        // Perched behind-right of the origin, looking at the triangle.
        Self {
            pos: Vec3::new(3.2, 2.2, 4.5),
            yaw: -0.64,
            pitch: -0.39,
        }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp)
    }

    // Used by the Windows flycam; macOS input handling lands later.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, sy)
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        // rh + directx projection = right-handed view, 0..1 clip depth (wgpu).
        let view = glam::camera::rh::view::look_to_mat4(self.pos, self.forward(), Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(
            60f32.to_radians(),
            aspect.max(1e-3),
            0.05,
            2000.0,
        );
        proj * view
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    grid_pipeline: wgpu::RenderPipeline,
    tri_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    start: Instant,
}

impl Renderer {
    pub fn new(target: crate::SurfaceTarget, width: u32, height: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // Vulkan/Metal for the spike (dx12 feature is compiled out —
            // windows-crate version skew vs tauri; see Spike A memo and root
            // Cargo.toml).
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = unsafe { target.create_surface(&instance) }?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| format!("request_adapter: {e}"))?;
        tracing::info!("inf-viewport adapter: {:?}", adapter.get_info());

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| format!("request_device: {e}"))?;

        let mut config = surface
            .get_default_config(&adapter, width.max(1), height.max(1))
            .ok_or_else(|| "surface not supported by adapter".to_string())?;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spike-a-scene"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spike-a-uniforms"),
            size: UNIFORM_FLOATS as u64 * 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let make_pipeline = |label: &str, vs: &str, fs: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let grid_pipeline = make_pipeline(
            "spike-a-grid",
            "vs_grid",
            "fs_grid",
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        );
        let tri_pipeline = make_pipeline("spike-a-tri", "vs_tri", "fs_tri", None);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            grid_pipeline,
            tri_pipeline,
            uniforms,
            bind_group,
            start: Instant::now(),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    pub fn render(&mut self, camera: &Camera) {
        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match self.surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => f,
            Cst::Outdated | Cst::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Cst::Timeout | Cst::Occluded => return,
            Cst::Validation => {
                tracing::warn!("inf-viewport: surface validation error");
                return;
            }
        };

        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let vp = camera.view_proj(aspect);
        let inv = vp.inverse();
        let angle = self.start.elapsed().as_secs_f32() * 0.8;

        let mut data = [0f32; UNIFORM_FLOATS];
        data[0..16].copy_from_slice(&vp.to_cols_array());
        data[16..32].copy_from_slice(&inv.to_cols_array());
        data[32..35].copy_from_slice(&camera.pos.to_array());
        data[36] = angle;
        self.queue.write_buffer(&self.uniforms, 0, f32_bytes(&data));

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Editor --ink-bg-0 (#0d0f12), linearized.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0032,
                            g: 0.0037,
                            b: 0.0055,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.grid_pipeline);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.tri_pipeline);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

const UNIFORM_FLOATS: usize = 40; // 2×mat4 + 2×vec4

/// Minimal f32-slice → bytes cast (avoids a bytemuck dependency for the spike).
fn f32_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}
