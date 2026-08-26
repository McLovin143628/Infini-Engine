//! **Auto exposure** (wave VIS1b) — the shared buffers the histogram writes and
//! the bloom prefilter and the tonemap read.
//!
//! The pair lives here rather than inside [`crate::passes::exposure`] for the
//! reason `ShadowResources` and `GiResources` do (both `renderer.rs`): a
//! node cannot reach another node's buffers, and the exposure multiplier has
//! **three** readers at three depths of the frame — the histogram node writes it,
//! the bloom prefilter thresholds against it, and the tonemap multiplies by it.
//!
//! # One buffer, two binding types
//!
//! [`ExposureResources::state`] is declared `STORAGE | UNIFORM`, so the compute
//! pass writes it as a storage buffer and the two fragment passes read the very
//! same bytes as a uniform. The alternative was a storage buffer plus a
//! `copy_buffer_to_buffer` into a uniform twin, which is one more copy, one more
//! allocation and one more chance for the two to describe different frames.
//!
//! # Why the multiplier is on the GPU at all
//!
//! It is the average scene luminance that has to be measured, and measuring it
//! on the CPU means a readback. A readback's latency is a function of the
//! driver's scheduling, so an adaptation stepped by it would be a function of
//! how busy the machine was — which is exactly the property
//! [`crate::settings::adapt_exposure_ev`]'s level-clock `dt` exists to refuse.
//! Keeping the state on the GPU costs sixteen bytes and buys a rule that two runs
//! of one document agree about.

use crate::gpu::GpuContext;

/// Histogram bins — [`crate::settings::EXPOSURE_BINS`], as bytes.
const HISTOGRAM_BYTES: u64 = crate::settings::EXPOSURE_BINS as u64 * 4;

/// What the exposure state buffer holds, in the shader's layout.
///
/// Four floats, and every one of them is read by something: `multiplier` by the
/// bloom prefilter and the tonemap, `ev` by the next frame's adaptation step,
/// `avg_luminance` by [`crate::EngineRenderer::read_exposure`] (which is how a
/// gate sees what the histogram measured without a screenshot), and `valid` by
/// the resolve pass, which snaps instead of adapting when it is zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExposureState {
    /// The linear exposure multiplier, compensation already folded in.
    pub multiplier: f32,
    /// The adapted exposure in **stops**, without compensation.
    pub ev: f32,
    /// The average scene luminance the histogram measured. `0.0` in manual mode,
    /// where no histogram is built.
    pub avg_luminance: f32,
    /// `1.0` once [`ev`](Self::ev) is meaningful.
    pub valid: f32,
}

/// The shared auto-exposure buffers, created once with the renderer.
pub struct ExposureResources {
    /// 256 `u32` bins. Cleared and rebuilt every frame auto exposure runs.
    pub histogram: wgpu::Buffer,
    /// The 16-byte [`ExposureState`]. **Persistent across frames** — that is what
    /// makes the adaptation an adaptation rather than a per-frame snap.
    pub state: wgpu::Buffer,
    /// Staging for [`crate::EngineRenderer::read_exposure`]. Sixteen bytes, and
    /// nothing copies into it on a shipped frame.
    pub readback: wgpu::Buffer,
}

impl ExposureResources {
    pub fn new(gpu: &GpuContext) -> Self {
        let histogram = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure-histogram"),
            size: HISTOGRAM_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let state = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure-state"),
            size: std::mem::size_of::<ExposureState>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exposure-readback"),
            size: std::mem::size_of::<ExposureState>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A renderer that has never run the exposure node still has two readers
        // of this buffer, so it starts at the identity rather than at the zeroes
        // a fresh allocation holds — a multiplier of 0 would render black.
        gpu.queue.write_buffer(
            &state,
            0,
            bytemuck::bytes_of(&ExposureState {
                multiplier: 1.0,
                ev: 0.0,
                avg_luminance: 0.0,
                valid: 0.0,
            }),
        );
        Self {
            histogram,
            state,
            readback,
        }
    }

    /// Read the live state back, blocking.
    ///
    /// Blocking on purpose, and for the reason the picker and the golden harness
    /// block: this is asked for by a *gate*, once, outside the frame loop, where
    /// there is no next frame to overlap with. Nothing on a shipped path calls
    /// it.
    pub fn read(&self, gpu: &GpuContext) -> Result<ExposureState, String> {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("exposure-readback"),
            });
        encoder.copy_buffer_to_buffer(
            &self.state,
            0,
            &self.readback,
            0,
            std::mem::size_of::<ExposureState>() as u64,
        );
        gpu.queue.submit(Some(encoder.finish()));

        let slice = self.readback.slice(..);
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
            .map_err(|e| format!("map exposure readback: {e}"))?;
        let state = *bytemuck::from_bytes::<ExposureState>(&data);
        drop(data);
        self.readback.unmap();
        Ok(state)
    }
}
