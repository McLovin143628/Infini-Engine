//! **Auto-exposure node** (wave VIS1b): a luminance histogram over `post_hdr`,
//! reduced to a log-average and adapted toward at a rate the level authors.
//!
//! Two compute dispatches and a buffer clear, in front of the bloom node and
//! therefore in front of the tonemap — which is the whole of the ordering
//! decision, because from this wave on the bloom **threshold is exposure
//! relative** (see [`crate::passes::bloom`]). Both readers take their multiplier
//! from [`crate::exposure::ExposureResources::state`], so there is one exposure
//! per frame and no way for the threshold and the multiply to disagree about it.
//!
//! # Manual mode records nothing
//!
//! At [`ExposureMode::Manual`] — the default, and what every level authored
//! before this wave carries — the node writes sixteen bytes and returns before it
//! touches the encoder. No histogram, no dispatch, no barrier: the multiplier is
//! [`RenderSettings::exposure`](crate::RenderSettings::exposure) with the
//! compensation folded in, which at the default compensation of zero is that
//! scalar **bit for bit**. That is what keeps all fifty-five goldens byte
//! identical across this wave.
//!
//! And it writes only when the value has *changed*, so a manual frame after the
//! first one records nothing at all.

use crate::exposure::ExposureState;
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::FrameData;
use crate::settings::{manual_exposure_multiplier, ExposureMode, EXPOSURE_BINS};

/// Every second texel in each axis — a quarter of the frame.
///
/// The histogram is a *statistic*, and a quarter of two million pixels is half a
/// million samples: the log-average's standard error at that count is far below
/// the 0.078-stop width of a bin, so the fourfold saving costs nothing the rule
/// can see. A regular lattice rather than a jittered set, because a jitter would
/// make the frame a function of something other than the scene.
const SAMPLE_STRIDE: u32 = 2;

/// The largest level-clock delta one frame may adapt over, seconds.
///
/// Not a smoothing constant — a **discontinuity guard**. `cloud_time_s` is the
/// document's clock, and an author scrubbing time of day or a level rolling over
/// a year hands this node a jump of hours. Ten seconds is far above any real
/// frame at any clock rate (a level running at 60× real time advances one second
/// per frame at 60 fps) and far below a scrub.
const MAX_STEP_S: f64 = 10.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExposureParams {
    /// x = source width, y = source height (px), z = sample stride, w unused.
    source: [u32; 4],
    /// x = min luminance, y = max luminance, z = adaptation speed (stops/s),
    /// w = compensation (stops).
    control: [f32; 4],
    /// x = level-clock delta (s), y = history valid (>0.5), zw unused.
    step: [f32; 4],
}

pub struct ExposureNode {
    bgl: wgpu::BindGroupLayout,
    histogram: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    /// The previous frame's level clock, for the adaptation's `dt`. `None` until
    /// auto exposure has run once, which is also what makes the first auto frame
    /// **snap** to its target rather than crawl toward it from wherever the
    /// buffer happened to be.
    prev_clock: Option<f64>,
    /// The target generation the state was adapted against. A resize changes the
    /// histogram's source and invalidates the average with it.
    generation: u64,
    /// The manual-mode value already in the buffer, so an unchanged manual frame
    /// records nothing.
    last_manual: Option<f32>,
}

impl ExposureNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("exposure"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/exposure.wgsl").into()),
            });
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("exposure"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    storage(2),
                    storage(3),
                ],
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("exposure"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let make = |entry: &str| {
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("exposure"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure-params"),
            size: std::mem::size_of::<ExposureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            histogram: make("cs_histogram"),
            resolve: make("cs_resolve"),
            bgl,
            params_buf,
            prev_clock: None,
            generation: 0,
            last_manual: None,
        }
    }
}

impl RenderNode for ExposureNode {
    fn name(&self) -> &'static str {
        "exposure"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let ctl = frame.settings.exposure_control;

        if ctl.mode != ExposureMode::Auto {
            // The adaptation must re-snap if auto ever comes back: the stored EV
            // describes a frame nobody measured while manual held the buffer.
            self.prev_clock = None;
            let m = manual_exposure_multiplier(frame.settings.exposure, ctl.compensation_ev);
            if self.last_manual != Some(m) {
                gpu.queue.write_buffer(
                    &frame.exposure.state,
                    0,
                    bytemuck::bytes_of(&ExposureState {
                        multiplier: m,
                        ev: 0.0,
                        avg_luminance: 0.0,
                        valid: 0.0,
                    }),
                );
                self.last_manual = Some(m);
            }
            return;
        }
        self.last_manual = None;

        let clock = frame.scene.atmosphere.clouds.time_s;
        let same_size = self.generation == frame.targets.generation;
        let dt = match self.prev_clock {
            Some(prev) if same_size => (clock - prev).clamp(0.0, MAX_STEP_S) as f32,
            _ => 0.0,
        };
        let valid = self.prev_clock.is_some() && same_size;
        self.prev_clock = Some(clock);
        self.generation = frame.targets.generation;

        let (w, h) = frame.targets.size;
        gpu.queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::bytes_of(&ExposureParams {
                source: [w, h, SAMPLE_STRIDE, 0],
                control: [
                    ctl.min_luminance,
                    ctl.max_luminance,
                    ctl.adaptation_speed,
                    ctl.compensation_ev,
                ],
                step: [dt, if valid { 1.0 } else { 0.0 }, 0.0, 0.0],
            }),
        );

        // `post_hdr` alternates with TAA's ping-pong, so the bind group is
        // rebuilt per frame — the tonemap's reason, one node earlier.
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exposure"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(frame.post_hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame.exposure.histogram.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: frame.exposure.state.as_entire_binding(),
                },
            ],
        });

        encoder.clear_buffer(&frame.exposure.histogram, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("exposure-histogram"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram);
            pass.set_bind_group(0, &bind_group, &[]);
            let gx = w.div_ceil(SAMPLE_STRIDE).div_ceil(16).max(1);
            let gy = h.div_ceil(SAMPLE_STRIDE).div_ceil(16).max(1);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("exposure-resolve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup of exactly `EXPOSURE_BINS` lanes: the reduction is a
            // fixed halving tree over the bins, so there is nothing to dispatch
            // twice.
            debug_assert_eq!(EXPOSURE_BINS, 256);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}
