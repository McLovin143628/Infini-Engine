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

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use inf_input::{InputEvent, InputMap, InputState};
use inf_render::{create_instance, GpuContext, OrthoParams, RenderView};

use crate::input;
use crate::render::PlayerRenderHost;
use crate::runtime_sim::RuntimeSim;

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
}

impl PlayerApp {
    fn new(title: String, width: u32, height: u32, sim: RuntimeSim, map: InputMap) -> Self {
        Self {
            title,
            width,
            height,
            sim,
            input_state: InputState::new(map),
            live: None,
        }
    }

    /// Build (or rebuild, on device loss) the GPU stack from the window.
    fn build_host(
        window: &Arc<Window>,
        width: u32,
        height: u32,
    ) -> Result<PlayerRenderHost, String> {
        let instance = create_instance();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create_surface: {e}"))?;
        let gpu = GpuContext::for_surface(instance, &surface)?;
        PlayerRenderHost::new(gpu, surface, width.max(1), height.max(1))
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
        self.sim.run_frame(dt, held);

        let alpha = self.sim.alpha();
        let view = self.view();
        let Some(live) = self.live.as_mut() else {
            return;
        };
        if live.host.is_lost() {
            tracing::warn!("inf-player: device lost — rebuilding GPU stack");
            match Self::build_host(&live.window, self.width, self.height) {
                Ok(host) => live.host = host,
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
        match Self::build_host(&window, self.width, self.height) {
            Ok(host) => {
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
) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = PlayerApp::new(title, width, height, sim, map);
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("run_app: {e}"))
}
