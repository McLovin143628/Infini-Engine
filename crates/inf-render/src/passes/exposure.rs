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
//! scalar **bit for bit**. That is what keeps every committed golden byte
//! identical across this wave.
//!
//! And it writes only when the value has *changed*, so a manual frame after the
//! first one records nothing at all.
//!
//! # The clock, and the level that does not have one
//!
//! `dt` is a **level-clock** delta (`inf_ecs::sky::ResolvedSky::cloud_time_s`) and
//! never a wall clock or a frame index — that is what makes the adaptation a pure
//! function of the document, and it is what the PIE-vs-shipping arm rests on.
//!
//! But `cloud_time_s` is derived from `TimeOfDay` alone, and **`TimeOfDay::rate`
//! defaults to `0.0`** — so on a level that never authored a running clock (which
//! is most of them, and every level authored before wave P17) the delta is zero on
//! every frame after the first. Read naively that makes the eye adapt once and
//! then freeze at whatever the first frame happened to look like, for ever: a
//! player walking out of a lit courtyard into a cellar would see **no** adaptation
//! at all. The VIS1b audit measured exactly that.
//!
//! So the rule has two halves, and `ExposureNode::clock_ran` is which half is in
//! force:
//!
//! * **the clock has never moved** ⇒ there is no rate for a ramp to be expressed
//!   in, so the eye *tracks* — every frame snaps to its own target. Unsmoothed,
//!   and correct rather than frozen;
//! * **the clock has moved at least once** ⇒ it is a running clock, and a frame
//!   whose delta is zero is either a paused world or a second render of one
//!   simulation step. Both must hold the previous value, which is what makes
//!   "a paused sim is a frozen eye" true and what makes three renders per sim step
//!   land on the same exposure as one.
//!
//! Both halves are a pure function of the document's own clock sequence, so both
//! hosts still produce the same trace. What is *not* fixed is that the first half
//! has no smoothing: a ramped eye on a static-clock level needs a simulation clock
//! in the scene projection, carried as **VIS-C1d**.

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
/// a year hands this node a jump of hours; adapting over that would snap the eye
/// on the frame the slider moved.
///
/// **What it costs, stated rather than hidden.** Ten seconds of clock per frame
/// is `TimeOfDay::rate == 600` at 60 fps — a whole day in 2.4 real minutes. Above
/// that rate the guard, not `adaptation_speed`, governs how fast the eye moves,
/// and the authored number stops meaning what it says. Every plausible game clock
/// is far below it (`rate == 60` is a day in 24 minutes) and the wave's own
/// PIE-vs-shipping arm, which runs at `rate == 30 000` to make the sun set inside
/// twenty-four frames, is *above* it — deliberately, and it is the only thing in
/// the tree that is.
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
    /// **Has this level's clock ever actually moved?** (VIS1b audit.)
    ///
    /// See the module doc. `false` ⇒ every frame snaps to its own target, because
    /// a level whose `TimeOfDay::rate` is `0.0` — the default — hands this node a
    /// zero delta for ever, and adapting by zero for ever is an eye frozen on the
    /// first frame it ever saw rather than an eye. `true` ⇒ the clock is a running
    /// one and a zero delta means a paused world or a repeated render of one
    /// simulation step, both of which must hold.
    ///
    /// A function of the clock **sequence** alone, so both hosts compute it
    /// identically and the PIE-vs-shipping trace is unaffected.
    clock_ran: bool,
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
            clock_ran: false,
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
            // describes a frame nobody measured while manual held the buffer. And
            // `clock_ran` goes with it — what the clock did during a manual stretch
            // says nothing about the trace auto exposure is about to start.
            self.prev_clock = None;
            self.clock_ran = false;
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
        // A clock that has moved once is a running clock for the rest of the
        // session; see the module doc for why the two halves are not the same
        // rule. Read from THIS frame's delta before `valid` consumes it, so the
        // first moving frame already adapts rather than snapping twice.
        if dt > 0.0 {
            self.clock_ran = true;
        }
        let valid = self.prev_clock.is_some() && same_size && self.clock_ran;
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
