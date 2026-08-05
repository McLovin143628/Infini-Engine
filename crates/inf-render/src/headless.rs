//! Offscreen render target + CPU readback, for golden-image tests and (later)
//! the asset thumbnailer.

use crate::gpu::GpuContext;

/// Fixed-format offscreen color target readable back to the CPU.
pub struct HeadlessTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    /// **Persistent readback staging** (P23.2a audit).
    ///
    /// `read_rgba` used to allocate a fresh `MAP_READ` buffer per call — right
    /// for a one-shot golden, and a megabyte of allocate-map-drop per frame for
    /// the interactive preview at 512² (`PreviewSession` reads back every orbit
    /// frame). The size is fixed by the target's own extent, so it belongs to
    /// the target and is created once. Mapping and unmapping the same buffer
    /// repeatedly is exactly what `map_async`/`unmap` are for.
    readback: wgpu::Buffer,
}

/// All headless rendering happens in linear→sRGB, matching the on-screen path.
pub const HEADLESS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Bytes per row of an RGBA8 readback, rounded up to wgpu's copy alignment.
/// One definition, used by both the buffer's size and the unpad walk — they
/// disagreeing is a silently sheared image.
fn padded_row(width: u32) -> usize {
    (width as usize * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize
}

impl HeadlessTarget {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HEADLESS_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless-readback"),
            size: (padded_row(width) * height as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            texture,
            view,
            width,
            height,
            readback,
        }
    }

    /// Blocking readback of the full target as tightly-packed RGBA8 rows.
    ///
    /// Reuses the target's own staging buffer (see [`readback`](Self::readback)),
    /// so an interactive caller reading every frame allocates nothing.
    pub fn read_rgba(&self, gpu: &GpuContext) -> Result<Vec<u8>, String> {
        let unpadded = self.width as usize * 4;
        let padded = padded_row(self.width);
        let buffer = &self.readback;

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless-readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
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
        let mut out = Vec::with_capacity(unpadded * self.height as usize);
        for row in data.chunks(padded) {
            out.extend_from_slice(&row[..unpadded]);
        }
        drop(data);
        buffer.unmap();
        Ok(out)
    }
}
