//! The windowed player: a winit window + wgpu surface running the fixed-step
//! [`RuntimeSim`] with interpolated rendering (P9.3 item 1).
//!
//! Reuses `inf-render` exactly as the editor viewport does — the surface is made
//! straight from the winit window (no per-OS code; winit + raw-window-handle
//! abstract it), then [`PlayerRenderHost`] owns the floating-origin, reverse-Z
//! forward renderer and all existing passes. The loop drives a **variable** frame
//! time into the fixed-step accumulator and renders with the interpolation
//! `alpha`. ESC (or the close button) quits.
//!
//! This path needs a GPU + display, so — like every GPU path in this repo — it is
//! human-verified, not exercised in CI. CI covers it with `cargo check` (it must
//! compile) and covers the gameplay/determinism headlessly.

use std::io::Write;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
// std::time::Instant panics on wasm32-unknown-unknown; web-time reads
// performance.now() so the fixed-step accumulator ticks in the browser.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use glam::{DVec3, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use inf_input::{InputEvent, InputMap, InputState};
use inf_render::{create_instance, GpuContext, OrthoParams, RenderView};
use inf_runtime::pie::{write_msg, EditorToPlayer, PlayerToEditor};

use crate::input;
use crate::render::PlayerRenderHost;
use crate::runtime_sim::RuntimeSim;
use crate::vmesh::VmeshRegistry;

/// Play-in-editor control channel + report sink attached to a windowed player.
/// Present only for the PIE window path; a standalone game leaves it `None`.
struct PieLink {
    control: Receiver<EditorToPlayer>,
    out: Box<dyn Write>,
    hwnd_reported: bool,
}

/// The native window handle as an `i64` for the PIE `Window` report (HWND on
/// Windows; `0` where not applicable). Lets the editor reparent the window into
/// the viewport slot.
#[cfg(windows)]
fn window_handle_i64(window: &Window) -> i64 {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Win32(h)) => h.hwnd.get() as i64,
        _ => 0,
    }
}

#[cfg(not(windows))]
fn window_handle_i64(_window: &Window) -> i64 {
    0
}

/// Orthographic near/far for the 2D-style player camera (reverse-Z inside the
/// renderer; these are positive distances from the eye).
const CAM_NEAR: f32 = 0.1;
const CAM_FAR: f32 = 2000.0;
/// Half the visible world height (world units) — the camera zoom.
const CAM_HALF_HEIGHT: f64 = 4.0;

/// Live state that exists only once the window is created.
struct Live {
    window: Arc<Window>,
    host: PlayerRenderHost,
    last: Instant,
    pending: Vec<InputEvent>,
}

/// The winit application: owns the sim + input, and (once resumed) the window.
pub struct PlayerApp {
    title: String,
    width: u32,
    height: u32,
    sim: RuntimeSim,
    input_state: InputState,
    live: Option<Live>,
    /// Play-in-editor link (control frames + reports); `None` for standalone.
    pie: Option<PieLink>,
    /// PIE pause state (ignored when `pie` is `None`).
    paused: bool,
    /// Cook-derived vmesh DAGs a `MeshRef.asset` resolves to (P13.4); attached to
    /// the render host so asset meshes render real geometry. Empty for
    /// primitive-only / PIE worlds.
    vmeshes: Arc<VmeshRegistry>,
    /// The loaded level's scene-persisted render block (R-P4); the render host
    /// maps it onto the live `RenderSettings` at build (and device-loss rebuild).
    /// `default` for content with no authored block (PIE / web / android v1).
    render: inf_scene::RenderSettingsRecord,
    /// Seconds since the last terrain-streaming diagnostics line (P16.3b2). The
    /// counters go to `tracing` — the existing Output Log / log-file path — once
    /// a second, and only while something actually streams, so a non-streaming
    /// world logs nothing at all.
    stats_accum: f64,
    /// The HTML canvas the winit window binds to (web only). Set by [`run_web`];
    /// applied to the window attributes in `resumed`.
    #[cfg(target_arch = "wasm32")]
    canvas: Option<web_sys::HtmlCanvasElement>,
    /// On-screen touch controls (a virtual gamepad). Present on touch platforms
    /// (web/Android); `None` on desktop. Winit `Touch` events route through it.
    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    touch: inf_input::TouchControls,
}

impl PlayerApp {
    #[allow(clippy::too_many_arguments)]
    fn new(
        title: String,
        width: u32,
        height: u32,
        sim: RuntimeSim,
        map: InputMap,
        vmeshes: Arc<VmeshRegistry>,
        render: inf_scene::RenderSettingsRecord,
    ) -> Self {
        Self {
            title,
            width,
            height,
            sim,
            input_state: InputState::new(map),
            live: None,
            pie: None,
            stats_accum: 0.0,
            paused: false,
            vmeshes,
            render,
            #[cfg(target_arch = "wasm32")]
            canvas: None,
            #[cfg(any(target_arch = "wasm32", target_os = "android"))]
            touch: crate::input::default_touch_controls(),
        }
    }

    /// Drain any pending PIE control frames. Returns `true` to request exit
    /// (Stop / channel closed).
    fn drain_pie_control(&mut self) -> bool {
        let Some(pie) = self.pie.as_mut() else {
            return false;
        };
        loop {
            match pie.control.try_recv() {
                Ok(EditorToPlayer::Pause) => {
                    self.paused = true;
                    let _ = write_msg(&mut pie.out, &PlayerToEditor::Paused);
                }
                Ok(EditorToPlayer::Resume) => {
                    self.paused = false;
                    let _ = write_msg(&mut pie.out, &PlayerToEditor::Resumed);
                }
                Ok(EditorToPlayer::Step { count }) => {
                    // A single-step while paused: advance the fixed sim directly.
                    for _ in 0..count {
                        self.sim
                            .step_once(crate::runtime_sim::RuntimeInput::default());
                    }
                }
                Ok(EditorToPlayer::SetViewport(r)) => {
                    // The editor owns our rect via the parent window; still resize
                    // the surface so the swapchain matches.
                    self.width = r.width.max(1);
                    self.height = r.height.max(1);
                    if let Some(live) = self.live.as_mut() {
                        live.host.resize(self.width, self.height);
                    }
                }
                Ok(EditorToPlayer::Eject) => {
                    let _ = write_msg(&mut pie.out, &PlayerToEditor::Ejected);
                }
                Ok(EditorToPlayer::Stop) => {
                    let _ = write_msg(&mut pie.out, &PlayerToEditor::Stopped);
                    return true;
                }
                Ok(EditorToPlayer::InjectPanic) => {
                    panic!("deliberate PIE panic (injected by editor)");
                }
                // Already loaded; ignore a second content frame.
                Ok(EditorToPlayer::Load(_)) | Ok(EditorToPlayer::LoadScene(_)) => {}
                Err(std::sync::mpsc::TryRecvError::Empty) => return false,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return true,
            }
        }
    }

    /// Build (or rebuild, on device loss) the GPU stack from the window. `render`
    /// is the level's scene-persisted render block (R-P4), applied to the fresh
    /// renderer so a device-loss rebuild keeps the authored look.
    fn build_host(
        window: &Arc<Window>,
        width: u32,
        height: u32,
        render: inf_scene::RenderSettingsRecord,
    ) -> Result<PlayerRenderHost, String> {
        let instance = create_instance();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create_surface: {e}"))?;
        let gpu = GpuContext::for_surface(instance, &surface)?;
        PlayerRenderHost::new(gpu, surface, width.max(1), height.max(1), render)
    }

    /// Build the render view for the current sim + surface (an ortho follow-cam).
    fn view(&self) -> Option<RenderView> {
        let live = self.live.as_ref()?;
        let (w, h) = live.host.size();
        let focus = self.sim.camera_focus();
        Some(RenderView {
            origin: live.host.origin(),
            eye_world: focus + DVec3::new(0.0, 0.0, 100.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: w,
            height: h,
            ortho: Some(OrthoParams {
                half_height: CAM_HALF_HEIGHT as f32,
                near: CAM_NEAR,
                far: CAM_FAR,
            }),
        })
    }

    /// Emit the terrain-streaming counters once a second (P16.3b2).
    ///
    /// The diagnostics seam is deliberately the existing one — `tracing`, which
    /// already tees to the log file and (in the editor) the Output Log — rather
    /// than a new overlay or IPC channel.
    fn log_stream_stats(&mut self, dt: f64) {
        if self.sim.terrain_streaming().is_empty() {
            return;
        }
        self.stats_accum += dt;
        if self.stats_accum < 1.0 {
            return;
        }
        self.stats_accum = 0.0;
        tracing::info!(
            "inf-player: {}",
            self.sim.terrain_streaming().stats().summary()
        );
    }

    /// One frame: fold input, advance the sim by the elapsed time, project, draw.
    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        // Elapsed time + drain this frame's input into the resolved state.
        let (dt, events) = {
            let Some(live) = self.live.as_mut() else {
                return;
            };
            let now = Instant::now();
            let dt = now.duration_since(live.last).as_secs_f64();
            live.last = now;
            (dt, std::mem::take(&mut live.pending))
        };
        self.input_state.apply(&events);
        let held = input::held_actions(&self.input_state);
        // PIE pause freezes the sim but keeps rendering the last frame.
        if !self.paused {
            self.sim.run_frame(dt, held);
        }

        let alpha = self.sim.alpha();
        let view = self.view();
        // THE RENDER-SYNC POINT (P16.3b2): advance every streamed terrain's
        // camera-driven cut exactly once per frame, here — after the fixed steps
        // and before the projection, so the sim can never observe it. The headless
        // harness drives `RuntimeSim::sync_render_terrain` at the same place in its
        // loop, which is what makes a scripted camera path reproduce the same
        // resident-set trace with and without a window.
        if let Some(v) = view.as_ref() {
            self.sim.sync_render_terrain(v.eye_world);
        }
        self.log_stream_stats(dt);
        let Some(live) = self.live.as_mut() else {
            return;
        };
        if live.host.is_lost() {
            tracing::warn!("inf-player: device lost — rebuilding GPU stack");
            match Self::build_host(&live.window, self.width, self.height, self.render) {
                Ok(mut host) => {
                    host.set_vmeshes(self.vmeshes.clone());
                    live.host = host;
                }
                Err(e) => {
                    tracing::error!("inf-player: GPU rebuild failed: {e}");
                    event_loop.exit();
                    return;
                }
            }
        }
        live.host.project(&self.sim, alpha);
        if let Some(view) = view {
            live.host.render(&view);
        }
        live.window.request_redraw();
    }
}

impl ApplicationHandler for PlayerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.live.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(PhysicalSize::new(self.width, self.height));
        // Web: bind the winit window to the page's <canvas> (the WebGPU surface
        // is then created from it exactly like a native window).
        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use winit::platform::web::WindowAttributesExtWebSys;
            attrs.with_canvas(self.canvas.clone())
        };
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!("inf-player: create_window failed: {e}");
                event_loop.exit();
                return;
            }
        };
        let size = window.inner_size();
        self.width = size.width.max(1);
        self.height = size.height.max(1);
        match Self::build_host(&window, self.width, self.height, self.render) {
            Ok(mut host) => {
                // Attach the vmesh registry so `MeshRef.asset` entities render real
                // geometry (meshlet path / classic fallback per the auto-tier).
                host.set_vmeshes(self.vmeshes.clone());
                // Report our native window handle so the editor can reparent us
                // into the viewport slot (embedded PIE).
                if let Some(pie) = self.pie.as_mut() {
                    if !pie.hwnd_reported {
                        let handle = window_handle_i64(&window);
                        let _ = write_msg(&mut pie.out, &PlayerToEditor::Window { handle });
                        pie.hwnd_reported = true;
                    }
                }
                window.request_redraw();
                self.live = Some(Live {
                    window,
                    host,
                    last: Instant::now(),
                    pending: Vec::new(),
                });
            }
            Err(e) => {
                tracing::error!("inf-player: GPU init failed: {e}");
                event_loop.exit();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.drain_pie_control() {
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.width = size.width.max(1);
                self.height = size.height.max(1);
                if let Some(live) = self.live.as_mut() {
                    live.host.resize(self.width, self.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == KeyCode::Escape && event.state == ElementState::Pressed {
                        event_loop.exit();
                        return;
                    }
                    if let Some(name) = input::keycode_to_code(code) {
                        if let Some(live) = self.live.as_mut() {
                            live.pending.push(InputEvent::Key {
                                code: name.to_string(),
                                pressed: event.state == ElementState::Pressed,
                            });
                        }
                    }
                }
            }
            // Touch platforms (web / Android): route winit touches through the
            // on-screen virtual gamepad, which emits synthetic gamepad events the
            // InputMap resolves — so touch drives the same actions/axes as a pad.
            #[cfg(any(target_arch = "wasm32", target_os = "android"))]
            WindowEvent::Touch(t) => {
                use winit::event::TouchPhase as WinitTouchPhase;
                let phase = match t.phase {
                    WinitTouchPhase::Started => inf_input::TouchPhase::Started,
                    WinitTouchPhase::Moved => inf_input::TouchPhase::Moved,
                    WinitTouchPhase::Ended => inf_input::TouchPhase::Ended,
                    WinitTouchPhase::Cancelled => inf_input::TouchPhase::Cancelled,
                };
                let touch_ev = InputEvent::Touch {
                    id: t.id,
                    phase,
                    position: [t.location.x as f32, t.location.y as f32],
                };
                let synth = self.touch.process(&[touch_ev]);
                if let Some(live) = self.live.as_mut() {
                    live.pending.extend(synth);
                }
            }
            WindowEvent::RedrawRequested => self.frame(event_loop),
            _ => {}
        }
    }
}

/// Run the windowed player over `sim` with input bindings `map`. Blocks until the
/// window closes; returns an error only if the event loop fails to start.
pub fn run(
    title: String,
    width: u32,
    height: u32,
    sim: RuntimeSim,
    map: InputMap,
    vmeshes: Arc<VmeshRegistry>,
    render: inf_scene::RenderSettingsRecord,
) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = PlayerApp::new(title, width, height, sim, map, vmeshes, render);
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("run_app: {e}"))
}

/// Run the windowed player as a **play-in-editor** window: identical to [`run`]
/// but wired to a PIE control channel (Pause/Resume/Step/Stop/Eject/SetViewport)
/// and a report sink (`Window` handle for reparenting, `Paused`/`Resumed`/…).
/// Blocks until Stop or the window closes.
pub fn run_pie(
    title: String,
    width: u32,
    height: u32,
    sim: RuntimeSim,
    map: InputMap,
    control: Receiver<EditorToPlayer>,
    out: Box<dyn Write>,
) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    // PIE streams no vmesh assets yet (a documented follow-up); asset meshes render
    // as placeholder cubes in PIE until the payload carries the vmesh index.
    // PIE also starts from the default render block (the streamed scene payload
    // carries no settings yet — a documented follow-up mirroring the vmesh gap).
    let mut app = PlayerApp::new(
        title,
        width,
        height,
        sim,
        map,
        Arc::new(VmeshRegistry::new()),
        inf_scene::RenderSettingsRecord::default(),
    );
    app.pie = Some(PieLink {
        control,
        out,
        hwnd_reported: false,
    });
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("run_app: {e}"))
}

/// Run the player on Android (P14.1): build the winit event loop from the
/// `AndroidApp` handed to `android_main`, then run the shared [`PlayerApp`]
/// (touch routing + mobile render tier are cfg'd on for `target_os = "android"`).
/// Needs the NDK to build (`cargo-ndk`); like every GPU/device path it is
/// structured for compilation and device-verified, not run in CI. See
/// `docs/android-player.md`.
#[cfg(target_os = "android")]
pub fn run_android(
    android_app: winit::platform::android::activity::AndroidApp,
    title: String,
    sim: RuntimeSim,
    map: InputMap,
) -> Result<(), String> {
    use winit::platform::android::EventLoopBuilderExtAndroid;
    let event_loop = EventLoop::builder()
        .with_android_app(android_app)
        .build()
        .map_err(|e| format!("event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    // Portrait-ish default; the surface reconfigures to the real window size on
    // the first `Resized`.
    let mut app = PlayerApp::new(
        title,
        1080,
        1920,
        sim,
        map,
        Arc::new(VmeshRegistry::new()),
        inf_scene::RenderSettingsRecord::default(),
    );
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("run_app: {e}"))
}

/// Run the player in the browser (P14.2): bind to `canvas`, then hand the app to
/// winit's web event loop. Unlike the desktop [`run`], `spawn_app` **returns
/// immediately** — the loop is driven by the browser's `requestAnimationFrame`,
/// so the caller (the wasm entry) must not block afterwards. Needs WebGPU; like
/// every GPU path it is compile-checked in CI and human-verified in a real
/// browser.
#[cfg(target_arch = "wasm32")]
pub fn run_web(
    title: String,
    canvas: web_sys::HtmlCanvasElement,
    sim: RuntimeSim,
    map: InputMap,
) -> Result<(), String> {
    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let width = canvas.width().max(1);
    let height = canvas.height().max(1);
    let mut app = PlayerApp::new(
        title,
        width,
        height,
        sim,
        map,
        Arc::new(VmeshRegistry::new()),
        inf_scene::RenderSettingsRecord::default(),
    );
    app.canvas = Some(canvas);
    event_loop.spawn_app(app);
    Ok(())
}
