//! Swapchain lifecycle: sRGB format selection, debounced reconfigure during
//! interactive resizes, and outdated/lost recovery.
//!
//! Production rect-sync policy (ROADMAP P2.1): the scene is always rendered
//! offscreen at the *requested* size, but the swapchain itself reconfigures at
//! most once per [`RECONFIGURE_DEBOUNCE`] while a splitter drag streams new
//! rects. Between reconfigures the blit pass stretches the scene into the
//! stale-size swapchain (imperceptible for the few frames involved); the final
//! rect always lands because the debounce window expires.

use std::time::{Duration, Instant};

use crate::gpu::GpuContext;

/// Minimum interval between swapchain reconfigures while sizes are changing.
pub const RECONFIGURE_DEBOUNCE: Duration = Duration::from_millis(80);

pub struct SurfaceChain {
    /// `None` only between [`release`](SurfaceChain::release) and the chain's
    /// replacement — see that method for the one caller and the reason.
    surface: Option<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    /// Latest size asked for by the host (physical px). May differ from
    /// `config` while the debounce window is open.
    requested: (u32, u32),
    /// Render-target format for everything that ends up on screen: the sRGB
    /// view of the swapchain format.
    target_format: wgpu::TextureFormat,
    last_configure: Instant,
}

impl SurfaceChain {
    pub fn new(
        gpu: &GpuContext,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let mut config = surface
            .get_default_config(&gpu.adapter, width.max(1), height.max(1))
            .ok_or_else(|| "surface not supported by adapter".to_string())?;
        config.present_mode = wgpu::PresentMode::AutoVsync;

        // Render in gamma-correct sRGB regardless of the swapchain's native
        // format: add an sRGB view format and create render views with it.
        let target_format = config.format.add_srgb_suffix();
        if target_format != config.format && !config.view_formats.contains(&target_format) {
            config.view_formats.push(target_format);
        }

        surface.configure(&gpu.device, &config);
        Ok(Self {
            surface: Some(surface),
            requested: (config.width, config.height),
            target_format,
            config,
            last_configure: Instant::now(),
        })
    }

    /// Format render pipelines targeting this surface must use.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Size the swapchain is currently configured at.
    pub fn configured_size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Latest size requested by the host — the size the scene should render at.
    pub fn requested_size(&self) -> (u32, u32) {
        self.requested
    }

    /// Record a new target size. Cheap; the actual reconfigure happens inside
    /// [`Self::acquire`] under the debounce policy.
    pub fn request_resize(&mut self, width: u32, height: u32) {
        self.requested = (width.max(1), height.max(1));
    }

    /// **Drop the swapchain now, without dropping the chain** (Hardening D).
    ///
    /// Both device-loss rebuilds in this tree used to construct a *second*
    /// `wgpu::Instance` + `Surface` **for the same window while the old one was
    /// still alive** — the old chain is only dropped at the assignment that
    /// replaces it, which happens after the new one exists. That is usually
    /// benign (the old device is already lost) but two live swapchains on one
    /// HWND/CAMetalLayer is driver-dependent state, and there is nothing to be
    /// gained by holding the dead one across the rebuild.
    ///
    /// The chain is unusable afterwards: [`acquire`](Self::acquire) answers
    /// `None`, which the callers already treat as "no image this frame". The one
    /// legitimate caller replaces `self` on the next line.
    pub fn release(&mut self) {
        self.surface = None;
    }

    fn configure(&mut self, gpu: &GpuContext) {
        let Some(surface) = self.surface.as_ref() else {
            return;
        };
        self.config.width = self.requested.0;
        self.config.height = self.requested.1;
        surface.configure(&gpu.device, &self.config);
        self.last_configure = Instant::now();
    }

    /// Acquire the next swapchain texture, applying the debounced-reconfigure
    /// policy and recovering from outdated/lost surfaces. `None` means "skip
    /// this frame" (timeout/occluded/validation — all transient).
    pub fn acquire(&mut self, gpu: &GpuContext) -> Option<wgpu::SurfaceTexture> {
        if self.requested != self.configured_size()
            && self.last_configure.elapsed() >= RECONFIGURE_DEBOUNCE
        {
            self.configure(gpu);
        }

        use wgpu::CurrentSurfaceTexture as Cst;
        // A released chain has no image, which is the same answer an occluded
        // window gives and is handled by every caller. See `release`.
        let Some(surface) = self.surface.as_ref() else {
            return None;
        };
        match surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => Some(f),
            Cst::Outdated | Cst::Lost => {
                // The window system invalidated the swapchain — reconfigure at
                // the requested size immediately (no debounce) and retry once.
                self.configure(gpu);
                match self
                    .surface
                    .as_ref()
                    .map(|s| s.get_current_texture())
                    .unwrap_or(Cst::Outdated)
                {
                    Cst::Success(f) | Cst::Suboptimal(f) => Some(f),
                    _ => None,
                }
            }
            Cst::Timeout | Cst::Occluded => None,
            Cst::Validation => {
                tracing::warn!("inf-render: surface validation error");
                None
            }
        }
    }

    /// View of `frame` in the sRGB target format.
    pub fn target_view(&self, frame: &wgpu::SurfaceTexture) -> wgpu::TextureView {
        frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.target_format),
            ..Default::default()
        })
    }
}
