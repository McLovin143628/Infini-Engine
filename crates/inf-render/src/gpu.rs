//! Device/queue/adapter management with crash-safe device-lost detection.
//!
//! One [`GpuContext`] per render host (the editor viewport thread, a headless
//! golden-image test, a thumbnailer). The context flags itself lost when the
//! driver dies; the host is expected to drop everything GPU-side and rebuild
//! from a fresh context (see `inf-viewport`'s render loop).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Create the shared wgpu instance. Vulkan/Metal only: the dx12 backend is
/// compiled out until tauri catches up to windows-crate 0.62 (Spike A memo,
/// root Cargo.toml).
pub fn create_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    })
}

/// Owned GPU handles plus a lost flag the host polls once per frame.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    lost: Arc<AtomicBool>,
}

impl GpuContext {
    /// Context for presenting to `surface` (picks a compatible adapter).
    pub fn for_surface(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> Result<Self, String> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(surface),
            ..Default::default()
        }))
        .map_err(|e| format!("request_adapter: {e}"))?;
        Self::from_adapter(instance, adapter)
    }

    /// Headless context (golden tests, thumbnails). Falls back to a software
    /// adapter (WARP / lavapipe) when no hardware one exists, and reports a
    /// descriptive error when there is no adapter at all so callers can skip.
    pub fn headless() -> Result<Self, String> {
        let instance = create_instance();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            ..Default::default()
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                force_fallback_adapter: true,
                compatible_surface: None,
                ..Default::default()
            }))
        })
        .map_err(|e| format!("no adapter (hardware or fallback): {e}"))?;
        Self::from_adapter(instance, adapter)
    }

    fn from_adapter(instance: wgpu::Instance, adapter: wgpu::Adapter) -> Result<Self, String> {
        tracing::info!("inf-render adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| format!("request_device: {e}"))?;

        let lost = Arc::new(AtomicBool::new(false));
        let lost_flag = lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            // `Destroyed` is the orderly-teardown path (context dropped);
            // anything else is a real loss the host must recover from.
            if !matches!(reason, wgpu::DeviceLostReason::Destroyed) {
                tracing::error!("GPU device lost ({reason:?}): {message}");
                lost_flag.store(true, Ordering::Release);
            }
        });

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            lost,
        })
    }

    /// True once the driver reported the device lost. The host must rebuild
    /// the whole GPU stack from a fresh context.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }
}
