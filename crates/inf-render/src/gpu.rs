//! Device/queue/adapter management with crash-safe device-lost detection.
//!
//! One [`GpuContext`] per render host (the editor viewport thread, a headless
//! golden-image test, a thumbnailer). The context flags itself lost when the
//! driver dies; the host is expected to drop everything GPU-side and rebuild
//! from a fresh context (see `inf-viewport`'s render loop).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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
    /// Count of uncaptured (validation/OOM/internal) GPU errors observed since a
    /// lenient error handler was installed. Stays `0` in headless/test contexts,
    /// which deliberately keep wgpu's fatal default so CI fails hard on
    /// validation bugs (see [`GpuContext::install_lenient_error_handler`] and
    /// [`GpuContext::headless`]). Interactive hosts can poll it to surface a
    /// "renderer degraded" state.
    uncaptured_errors: Arc<AtomicU32>,
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
    ///
    /// This path intentionally does **not** install a lenient uncaptured-error
    /// handler: it keeps wgpu's fatal default so any validation/OOM error aborts
    /// the process. That is what CI wants — a shader/pipeline regression must
    /// fail the build loudly, not degrade silently. Interactive editor hosts opt
    /// into leniency via [`GpuContext::install_lenient_error_handler`].
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

    /// **A headless context for the P28.5 ray-query experiment, and for nothing
    /// else.**
    ///
    /// Identical to [`headless`](GpuContext::headless) except that the device
    /// additionally requests `EXPERIMENTAL_RAY_QUERY` — which in wgpu 30
    /// requires the caller to sign `ExperimentalFeatures::enabled()`, an
    /// `unsafe` token acknowledging that the implementation of an experimental
    /// feature may contain undefined behaviour.
    ///
    /// **That signature is why this is a separate constructor.** It cannot go
    /// on the shipped path: a token accepting possible UB is not something a
    /// player's device should carry for a feature it never uses, and — measured
    /// in this batch — putting the feature into the ordinary optional mask
    /// makes `request_device` *fail* on a machine that has ray queries but no
    /// token, taking every headless test in the tree down with it.
    ///
    /// `Err` when there is no adapter, and `Err` when the adapter has no ray
    /// queries — which is every non-Vulkan backend and every software adapter.
    /// The caller is the experiment's gate, and it skips on either.
    pub fn headless_ray_query() -> Result<Self, String> {
        let instance = create_instance();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("no adapter: {e}"))?;
        if !adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        {
            let info = adapter.get_info();
            return Err(format!(
                "{} ({:?}, {:?}) has no EXPERIMENTAL_RAY_QUERY",
                info.name, info.backend, info.device_type
            ));
        }
        let features = adapter.features()
            & (wgpu::Features::POLYGON_MODE_LINE
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        // The four acceleration-structure limits are **0** in
        // `Limits::default()` — a device that requested the feature and not the
        // limits fails at `create_blas` with "Limit `max_blas_geometry_count`
        // is 0" and at `create_bind_group_layout` with "Too many bindings of
        // type AccelerationStructures … limit is 0", which are two more ways
        // the naive wiring would have looked like a driver problem and been a
        // descriptor one. Taken from the adapter, because the experiment's size
        // is its fixture's and not a policy.
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_blas_primitive_count: adapter_limits.max_blas_primitive_count,
            max_blas_geometry_count: adapter_limits.max_blas_geometry_count,
            max_tlas_instance_count: adapter_limits.max_tlas_instance_count,
            max_acceleration_structures_per_shader_stage: adapter_limits
                .max_acceleration_structures_per_shader_stage,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: limits,
            // SAFETY: the acknowledgement wgpu 30 asks for. This device is
            // built by the P28.5 experiment's gate and by nothing else; no
            // shipped host constructs one, and the experiment traces a
            // read-only acceleration structure it built itself.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))
        .map_err(|e| format!("request_device (ray query): {e}"))?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            lost: Arc::new(AtomicBool::new(false)),
            uncaptured_errors: Arc::new(AtomicU32::new(0)),
        })
    }

    fn from_adapter(instance: wgpu::Instance, adapter: wgpu::Adapter) -> Result<Self, String> {
        tracing::info!("inf-render adapter: {:?}", adapter.get_info());
        // Request the optional features *only where the adapter exposes them*.
        // Request-if-available is the standing rule: device creation NEVER fails
        // for lack of one, so headless/test/downlevel contexts keep working
        // unchanged and every consumer has a documented fallback.
        //
        //  * `POLYGON_MODE_LINE` (R-P2 wireframe view mode) — absent, the
        //    renderer degrades wireframe to unlit.
        //  * `TEXTURE_COMPRESSION_BC` (P26.1) — absent, virtual-texture tiles are
        //    CPU-transcoded to RGBA8 pages before upload, through the same
        //    residency door at 8× the page bytes for BC1 and 4× for BC3.
        //    `caps::AdapterCaps` probes it and `clamp_bc_tiles` turns the setting
        //    off; no tier ever turns it on.
        //
        // **`EXPERIMENTAL_RAY_QUERY` is deliberately NOT in this mask** (P28.5),
        // and the reason was measured rather than assumed. wgpu 30 refuses any
        // `EXPERIMENTAL_*` feature unless the descriptor also carries
        // `ExperimentalFeatures::enabled()`, which is an **`unsafe`** token the
        // caller signs to accept possible undefined behaviour. Adding the
        // feature to this mask the request-if-available way made
        // `request_device` fail outright — *"Some experimental features,
        // EXPERIMENTAL_RAY_QUERY, were requested, but experimental features are
        // not enabled"* — so every headless test in the tree skipped for want
        // of an adapter. Request-if-available is not a safe rule for a feature
        // whose request can be refused for a reason unrelated to the adapter.
        //
        // The experiment therefore builds its own device
        // ([`GpuContext::headless_ray_query`]) and the shipped one is exactly
        // the device it was before this batch.
        //  * `TIMESTAMP_QUERY | TIMESTAMP_QUERY_INSIDE_ENCODERS` (I4) — the
        //    per-pass GPU clock the fps instrument reads. Requested as a **pair**,
        //    because a timestamp written between two encoder commands is the only
        //    kind this engine writes ([`crate::timing`]): the graph's one seam is
        //    `RenderGraph::run`, and marking there costs no pass its own
        //    descriptor. An adapter with only half the pair gets neither, so
        //    `supports_timestamp_query` is a single question with a single answer.
        //    Absent, `EngineRenderer::gpu_timings` returns `None` and the
        //    instrument reports CPU frame time alone.
        let timestamps = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let mut optional_features = adapter.features()
            & (wgpu::Features::POLYGON_MODE_LINE | wgpu::Features::TEXTURE_COMPRESSION_BC);
        if adapter.features().contains(timestamps) {
            optional_features |= timestamps;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: optional_features,
            ..Default::default()
        }))
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
            uncaptured_errors: Arc::new(AtomicU32::new(0)),
        })
    }

    /// True once the driver reported the device lost. The host must rebuild
    /// the whole GPU stack from a fresh context.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Whether line-polygon-mode raster is available on this device (R-P2), i.e.
    /// the `POLYGON_MODE_LINE` feature was requested-and-granted at creation. The
    /// renderer/host query this to build the wireframe pipeline variant (and to
    /// clamp a Wireframe view-mode request to Unlit when it's absent).
    pub fn supports_polygon_mode_line(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
    }

    /// Whether BC1/BC2/BC3/BC4/BC5/BC6H/BC7 texture formats can be created and
    /// sampled on this device (P26.1), i.e. `TEXTURE_COMPRESSION_BC` was
    /// requested-and-granted at creation.
    ///
    /// Read at the **device**, not at the adapter, on purpose: what matters is
    /// what this device was created with, and a context built elsewhere (a
    /// thumbnailer, a headless test) must be able to answer for itself.
    pub fn supports_texture_compression_bc(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
    }

    /// Whether ray queries can be issued from a shader on this device (P28.5),
    /// i.e. `EXPERIMENTAL_RAY_QUERY` was requested-and-granted at creation.
    ///
    /// Read at the **device** and not at the adapter, for
    /// [`supports_texture_compression_bc`](GpuContext::supports_texture_compression_bc)'s
    /// reason: what matters is what this device was built with.
    ///
    /// `false` on every non-Vulkan backend by construction — wgpu 30 documents
    /// the feature as Vulkan-only and native-only — and on any Vulkan adapter
    /// without `VK_KHR_ray_query`, which includes every software rasterizer.
    /// **Nothing on the shipped path reads this**: it gates an experiment
    /// (`crate::raytrace`), and virtual shadow maps remain the shadow path on
    /// every tier whatever it answers.
    pub fn supports_ray_query(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
    }

    /// Whether this device can time a segment of a command encoder (I4), i.e.
    /// **both** `TIMESTAMP_QUERY` and `TIMESTAMP_QUERY_INSIDE_ENCODERS` were
    /// requested-and-granted at creation.
    ///
    /// Read at the **device** for
    /// [`supports_texture_compression_bc`](GpuContext::supports_texture_compression_bc)'s
    /// reason. It is one question rather than two because `from_adapter` grants
    /// the pair or neither: a timestamp this engine writes always sits *between*
    /// two encoder commands, so half the pair would be a capability nothing can
    /// use.
    ///
    /// Nothing on the shipped path reads this — [`crate::timing`] is off unless a
    /// harness turns it on, and a frame with it off records byte-identical
    /// commands to a frame that never heard of it.
    pub fn supports_timestamp_query(&self) -> bool {
        let f = self.device.features();
        f.contains(wgpu::Features::TIMESTAMP_QUERY)
            && f.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    }

    /// Install a lenient uncaptured-error handler for interactive/editor
    /// contexts. wgpu's default handler **aborts the process** on any
    /// validation, out-of-memory, or internal error; that is correct for CI and
    /// the headless golden/thumbnail paths (a validation bug must fail hard —
    /// see [`GpuContext::headless`]), but wrong for a live editor viewport,
    /// where a single bad draw should DEGRADE — get logged and counted, with
    /// rendering continuing — rather than kill the whole editor.
    ///
    /// Errors are counted (read via [`GpuContext::uncaptured_error_count`]) and
    /// logged via `tracing::error`, rate-limited so a bad pipeline that repeats
    /// every frame doesn't flood the log. Call this exactly once, on the editor
    /// host's context; never call it from [`GpuContext::headless`].
    pub fn install_lenient_error_handler(&self) {
        let count = self.uncaptured_errors.clone();
        self.device
            .on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
                let n = count.fetch_add(1, Ordering::Relaxed);
                // Log the first few, then every 256th, to bound log spam when a
                // bad draw recurs each frame.
                if n < 4 || n.is_multiple_of(256) {
                    tracing::error!("inf-render uncaptured GPU error (#{}): {err}", n + 1);
                }
            }));
    }

    /// Number of uncaptured GPU errors seen since the lenient handler was
    /// installed. Always `0` in headless/test contexts (which keep wgpu's fatal
    /// default and therefore never reach this counter).
    pub fn uncaptured_error_count(&self) -> u32 {
        self.uncaptured_errors.load(Ordering::Acquire)
    }
}
