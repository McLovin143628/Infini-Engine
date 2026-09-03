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
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use inf_input::{InputEvent, InputMap, InputState};
use inf_render::{create_instance, GpuContext, OrthoParams, RenderView};
use inf_runtime::pie::{write_msg, EditorToPlayer, PlayerToEditor};

use crate::input;
use crate::render::PlayerRenderHost;
use crate::runtime_sim::RuntimeSim;
use crate::skinned::SkinnedRegistry;
use crate::vmesh::VmeshRegistry;
use crate::voxel::VoxelRegistry;

/// Play-in-editor control channel + report sink attached to a windowed player.
/// Present only for the PIE window path; a standalone game leaves it `None`.
struct PieLink {
    control: Receiver<EditorToPlayer>,
    out: Box<dyn Write>,
    hwnd_reported: bool,
    /// How many times the window-handle report has been tried and failed
    /// (round-2 finding B7). Bounds the retry — see
    /// [`PieLink::report_window_handle`].
    hwnd_attempts: u32,
}

/// How many frames the window-handle report may be re-attempted before the
/// player gives up on embedded PIE (round-2 finding **B7**).
///
/// ~2 seconds at 60 Hz. A failed write means the editor has not read its end of
/// our stdout yet, or has closed it; the first is transient and the second is
/// permanent, and nothing on this side can tell them apart except by trying
/// again. Retrying for ever would write into a broken pipe sixty times a
/// second for the life of the session, which is why this is bounded and says
/// so once when it runs out.
const MAX_HWND_ATTEMPTS: u32 = 120;

impl PieLink {
    /// Report the native window handle to the editor, **once**, on the first
    /// write that succeeds.
    ///
    /// # The retry the warning promised did not exist (round-2 finding B7)
    ///
    /// C4-44 made the latch success-only, which is right: a dropped `Err` used
    /// to mark the handle reported when it was not, and embedded PIE stayed a
    /// blank hole for the session. The warning it wrote said *"retrying on the
    /// next frame"* — but the whole block lived inside
    /// `ApplicationHandler::resumed`, whose first statement is `if
    /// self.live.is_some() { return; }` and whose same arm ends by setting
    /// `self.live`. `resumed` therefore runs its body exactly once per process.
    /// There was no next frame for it, and `hwnd_reported` had exactly one
    /// reader — itself. So the fix half-landed: the false latch was gone and
    /// the outcome was identical.
    ///
    /// This is that retry, called from `frame()` as well, and bounded by
    /// [`MAX_HWND_ATTEMPTS`] so a genuinely closed pipe is not written to sixty
    /// times a second for ever.
    ///
    /// Returns `true` while the report is still outstanding — i.e. the caller
    /// should try again next frame.
    fn report_window_handle(&mut self, handle: i64) -> bool {
        if self.hwnd_reported || self.hwnd_attempts >= MAX_HWND_ATTEMPTS {
            return false;
        }
        match write_msg(&mut self.out, &PlayerToEditor::Window { handle }) {
            Ok(()) => {
                self.hwnd_reported = true;
                false
            }
            Err(e) => {
                self.hwnd_attempts += 1;
                if self.hwnd_attempts == 1 {
                    tracing::warn!(
                        "PIE: could not report the window handle ({e}); \
                         retrying on the next frame"
                    );
                } else if self.hwnd_attempts >= MAX_HWND_ATTEMPTS {
                    tracing::error!(
                        "PIE: the window handle could not be reported after \
                         {MAX_HWND_ATTEMPTS} attempts ({e}); the editor cannot \
                         reparent this window, so embedded PIE will stay blank"
                    );
                    return false;
                }
                true
            }
        }
    }
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
    /// **The in-game UI session** (island wave I5): the settings dialog, the
    /// toasts and the interaction prompt. Present in the shipped player **and**
    /// in a windowed PIE preview, because a preview that could not open the menu
    /// would be previewing a different game.
    ui: crate::ui::PlayerUi,
    /// Cook-derived vmesh DAGs a `MeshRef.asset` resolves to (P13.4); attached to
    /// the render host so asset meshes render real geometry. Empty for
    /// primitive-only / PIE worlds.
    vmeshes: Arc<VmeshRegistry>,
    /// The authored meshes scattered content draws (wave TER2b) — set from the
    /// loaded pack / dev-dir, empty for PIE and the browser worlds, where every
    /// scattered instance falls back to its placeholder primitive.
    scatter_meshes: Arc<inf_render::ScatterMeshes>,
    /// Skeletal render assets a `SkeletalMesh` resolves to (the P18.3 follow-up);
    /// attached to the render host so a bound character draws its real posed
    /// geometry. Inert for primitive-only / browser worlds, which is why
    /// [`PlayerApp::new`] starts it empty and [`run`] / [`run_pie`] replace it.
    ///
    /// **PIE is no longer inert** (P24.1): `ScenePayload` v7 carries the
    /// `.inf_mesh` bytes, so [`run_pie`] hands over a real store and a windowed
    /// preview draws the same character the shipped build does. It drew a
    /// placeholder cube for three phases, because skeletons, clips and machines
    /// crossed the wire and the mesh did not.
    skinned: Arc<SkinnedRegistry>,
    /// Where a `VoxelVolume.asset` finds its `.inf_voxel` bytes (P21.1) — the
    /// cooked pack, a dev content directory, or (P21.4) the PIE payload. Attached
    /// to the render host so a placed cave draws its real carved surface. Inert
    /// for primitive-only / browser worlds, like `skinned` and for the same
    /// reason, which is why [`PlayerApp::new`] starts it empty and [`run`] /
    /// [`run_pie`] replace it.
    voxel_assets: Arc<VoxelRegistry>,
    /// The level's material bindings + their `.inf_tex` containers (P26.4).
    /// Attached to the render host so a bound `.inf_mat`'s maps reach the
    /// surfaces that name it — the shipped half of clause 0, and the same shape,
    /// for the same reason, as `voxel_assets` (P21.4) and `skinned` (P24.1):
    /// passed in rather than defaulted, so a windowed session cannot quietly
    /// render something the headless one does not.
    materials: Arc<crate::MaterialContent>,
    /// The loaded level's scene-persisted render block (R-P4); the render host
    /// maps it onto the live `RenderSettings` at build (and device-loss rebuild).
    /// `default` for content with no authored block (PIE / web / android v1).
    render: inf_scene::RenderSettingsRecord,
    /// Seconds since the last terrain-streaming diagnostics line (P16.3b2). The
    /// counters go to `tracing` — the existing Output Log / log-file path — once
    /// a second, and only while something actually streams, so a non-streaming
    /// world logs nothing at all.
    stats_accum: f64,
    /// Draw the world-partition cell overlay (`--debug-cells`, P16.5).
    debug_cells: bool,
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
        mut sim: RuntimeSim,
        map: InputMap,
        vmeshes: Arc<VmeshRegistry>,
        render: inf_scene::RenderSettingsRecord,
    ) -> Self {
        // **The player's own settings**, and the map they produce (island wave
        // I5). The `map` argument is the LEVEL's table — `input.toml` beside a
        // dev level, or the shipped default for a cooked pack — and the player's
        // overrides are applied on top of it, so a rebinding survives a level
        // change and a level that ships its own table still gets one.
        let (ui, map) = crate::ui::PlayerUi::open(crate::ui::settings_dir(), map);
        if let Some(e) = &ui.load_error {
            tracing::warn!("inf-player: {e}");
        }
        ui.apply_to_sim(&mut sim);
        Self {
            title,
            width,
            height,
            sim,
            ui,
            input_state: InputState::new(map),
            live: None,
            pie: None,
            stats_accum: 0.0,
            debug_cells: false,
            paused: false,
            vmeshes,
            scatter_meshes: Arc::new(inf_render::ScatterMeshes::new()),
            skinned: Arc::new(SkinnedRegistry::new()),
            voxel_assets: Arc::new(VoxelRegistry::new()),
            materials: Arc::new(crate::MaterialContent::default()),
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

    /// Build the render view for the current sim + surface.
    ///
    /// **Two cameras, and which one runs is decided by the content** (P29.6). A
    /// level with a player-controlled character gets the locomotion camera — a
    /// real perspective third/first-person view, sitting where
    /// `inf_physics::d3::camera` put it after its own collision sweep. Every
    /// other level keeps the ortho follow-cam it has had since P9.3, so no
    /// committed sample's picture moves by a pixel.
    fn view(&self) -> Option<RenderView> {
        let live = self.live.as_ref()?;
        let (w, h) = live.host.size();
        if let Some(pose) = self.sim.camera_pose() {
            let (_r, up, forward) = inf_ecs::camera::basis(pose.yaw_deg, pose.pitch_deg);
            return Some(RenderView {
                origin: live.host.origin(),
                eye_world: pose.position.to_dvec3(),
                forward: forward.as_vec3(),
                up: up.as_vec3(),
                fov_y: (pose.fov_deg as f32).to_radians(),
                near: 0.05,
                width: w,
                height: h,
                // Perspective, deliberately: an ortho locomotion camera is a
                // different game.
                ortho: None,
            });
        }
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

    /// Emit the streaming counters once a second (terrain P16.3b2, cells P16.5).
    ///
    /// The diagnostics seam is deliberately the existing one — `tracing`, which
    /// already tees to the log file and (in the editor) the Output Log — rather
    /// than a new panel or IPC channel.
    fn log_stream_stats(&mut self, dt: f64) {
        // THE GATE COMES FIRST (Hardening Wave E). Every probe below is a
        // *diagnostic* and three of them are expensive: `vsm_summary` formats
        // four `String`s, `vt_summary` two, `predict_summary` runs the
        // dead-reckoner and formats, and `stream_summary` computes a whole
        // `stream_report()` — a mutex acquisition, a report clone and an
        // `inf_stream::arbitrate` — *unconditionally, even on a level with
        // nothing streaming at all*. All four used to run before the
        // one-second accumulator, sixty times a second, purely to decide
        // whether to run them again a second later. They are now behind it.
        //
        // The accumulator resets whenever the second elapses, whether or not
        // there was anything to say — which is the one behavioural difference,
        // and it is unobservable: a level with no subjects logged nothing then
        // and logs nothing now.
        self.stats_accum += dt;
        if self.stats_accum < 1.0 {
            return;
        }
        self.stats_accum = 0.0;
        let terrain = !self.sim.terrain_streaming().is_empty();
        let cells = !self.sim.cell_streaming().is_empty();
        // P27.5: the shadow-page streamer joins the two that were already here,
        // through the same seam (`tracing`, which already tees to the log file
        // and, in the editor, to the Output Log) and on the same cadence. A
        // level with virtual shadows off produces no line at all rather than a
        // line of zeros — `vsm_summary` is `None` there.
        let shadows = self
            .live
            .as_ref()
            .is_some_and(|l| l.host.vsm_summary().is_some());
        // P28.5: and the three lines that had none. `vt_summary`,
        // `stream_summary` and `predict_summary` each ship with the doc comment
        // "the one line a host logs about …" and, until this batch, no host
        // logged any of them — the "a line nothing logs" class the P28.4 audit
        // inherited to the last batch of the plan. Same seam, same cadence,
        // same rule: a subject that is absent produces no line at all rather
        // than a line of zeros.
        let paging = self.live.as_ref().is_some_and(|l| {
            l.host.vt_summary().is_some()
                || l.host.stream_summary().is_some()
                || l.host.predict_summary().is_some()
        });
        if !terrain && !cells && !shadows && !paging {
            return;
        }
        if terrain {
            tracing::info!(
                "inf-player: {}",
                self.sim.terrain_streaming().stats().summary()
            );
        }
        if cells {
            tracing::info!(
                "inf-player: {}",
                self.sim.cell_streaming().stats().summary()
            );
        }
        if let Some(line) = self.live.as_ref().and_then(|l| l.host.vsm_summary()) {
            tracing::info!("inf-player: {line}");
        }
        if let Some(line) = self.live.as_ref().and_then(|l| l.host.vt_summary()) {
            tracing::info!("inf-player: {line}");
        }
        if let Some(line) = self.live.as_ref().and_then(|l| l.host.stream_summary()) {
            tracing::info!("inf-player: {line}");
        }
        if let Some(line) = self.live.as_ref().and_then(|l| l.host.predict_summary()) {
            tracing::info!("inf-player: {line}");
        }
    }

    /// **What the E key is about right now, and where to draw it** (island wave
    /// I5).
    ///
    /// Asked every frame through the **same** `inf_physics::d3::interact::resolve`
    /// the `press_interact` edge is honoured by, so what the player is told and
    /// what the press does cannot come apart. `None` on a level with no
    /// character, with nothing in reach, or with the target behind the camera.
    ///
    /// `O(candidates)`, and `O(1)` on a level with no vehicles and no
    /// interactables — which is every level committed before this wave.
    fn interaction_prompt(
        sim: &RuntimeSim,
        view: &inf_render::RenderView,
        input: &InputState,
    ) -> Option<(glam::Vec2, String)> {
        let world = sim.world();
        let actor = inf_ecs::movement::camera_subject(world)?;
        let entity = world.entity_of(actor)?;
        let cm = world
            .world()
            .get::<inf_ecs::components::CharacterMovement>(entity)?;
        let t = world
            .world()
            .get::<inf_ecs::components::Transform>(entity)?;
        let radius = world
            .world()
            .get::<inf_ecs::components::Collider3D>(entity)
            .map(|c| c.radius)
            .unwrap_or(0.3);
        let feet = t.translation.to_dvec3() - DVec3::Y * (cm.half_height_for(cm.mode) + radius);
        let seated = cm
            .runtime
            .seat
            .is_seated()
            .then_some(cm.runtime.seat.vehicle);
        let exclude = crate::ui::PlayerUi::exclude(actor, seated);
        let hit = inf_physics::d3::interact::resolve(
            world,
            sim.bridge3d(),
            feet,
            cm.runtime.aim_yaw_deg,
            &exclude,
        )?;
        // The key the prompt names is the one the player has BOUND, read out of
        // the live map — so a rebinding changes the prompt on the next frame
        // rather than leaving it advertising a key that no longer does it.
        let key = match input
            .map()
            .desk_source(inf_ecs::movement::actions::INTERACT)
        {
            Some(inf_input::ActionSource::Key(c)) => c.clone(),
            Some(other) => inf_ui::bindings::token_of(other),
            None => String::new(),
        };
        let text = inf_ecs::interact::prompt_text(hit.verb, &hit.label, &key);
        let at = crate::ui::project_to_screen(view, hit.position)?;
        Some((at, text))
    }

    /// **The driver's readout** (wave VEH2b) — speed and gear for whoever is at
    /// a wheel, or `None`.
    ///
    /// The numbers are the sim's own: the speed is the chassis velocity
    /// `step_driving` writes onto the driver every step, and the gear is
    /// `Vehicle::gear`. The FORMATTING is `inf_ecs::vehicle::drive_readout`,
    /// in Ring 0, because this function cannot be tested and that one can.
    ///
    /// It closes half of VEH2a's carried item 8: the gear is drawn now. The rev
    /// counter and the aids' intervention are still published and still read by
    /// nobody.
    /// **…and since wave VEH2c it is whatever the craft is.** The dispatch is
    /// `inf_ecs::vehicle::craft_readout`, which chooses off the RIG'"'"'S OWN
    /// PARTS — the same facts the class was chosen from — so a boat cannot be
    /// handed a gearbox'"'"'s readout by a host that forgot which seat the player
    /// is in. An aircraft'"'"'s height is the one number Ring 0 cannot supply and
    /// this can.
    fn drive_readout(sim: &mut RuntimeSim) -> Option<String> {
        let subject = inf_ecs::movement::camera_subject(sim.world())?;
        let (vehicle, speed) = {
            let world = sim.world();
            let entity = world.entity_of(subject)?;
            let cm = world
                .world()
                .get::<inf_ecs::components::CharacterMovement>(entity)?;
            if cm.mode != inf_ecs::components::MovementMode::Driving || !cm.runtime.seat.is_seated()
            {
                return None;
            }
            (
                cm.runtime.seat.vehicle,
                cm.runtime.velocity.to_dvec3().length(),
            )
        };
        let (rig, gear, at, airborne) = {
            let v = sim.bridge3d().vehicle_of(vehicle)?;
            let rig = v.rig().clone();
            let airborne = rig
                .parts_of(inf_ecs::vehicle::PartKind::Rotor)
                .next()
                .is_some();
            let at = sim.world().entity_of(vehicle).and_then(|e| {
                sim.world()
                    .world()
                    .get::<inf_ecs::components::Transform>(e)
                    .map(|t| t.translation.to_dvec3())
            })?;
            (rig, v.gear(), at, airborne)
        };
        // The ground query is a `&mut` call on the sim (it may page a tile in),
        // so it is made only for the one craft that draws a height.
        let height = airborne.then(|| at.y - sim.terrain_height_at(at.x, at.z));
        Some(inf_ecs::vehicle::craft_readout(&rig, speed, gear, height))
    }

    /// **The shooter's readout** (wave WPN1) — the magazine and the reserve for
    /// whoever has a weapon loaded, or `None`.
    ///
    /// The numbers are the sim's own `WeaponState`, which is the ammunition
    /// clock the fixed step advances; the FORMATTING is
    /// `inf_ecs::weapon::ammo_readout`, in Ring 0, because this function cannot
    /// be tested and that one can — `drive_readout`'s own split, verbatim.
    ///
    /// A character with no `WeaponState` has nothing to say, which is every
    /// character that has not equipped a weapon and has not thrown a punch.
    ///
    /// **And neither has one carrying only its fists** (WPN1 audit): the melee
    /// clock is a real `WeaponState` with `MAX_MAGAZINE` rounds in it, so before
    /// `carries_ammunition` this read `"9999 / 10000"` at the bottom of the
    /// screen for the rest of a level after one punch. The question is asked in
    /// Ring 0 for this function's own reason.
    fn ammo_readout(sim: &RuntimeSim) -> Option<String> {
        let world = sim.world();
        let entity = world.entity_of(inf_ecs::movement::camera_subject(world)?)?;
        let s = world.world().get::<inf_ecs::weapon::WeaponState>(entity)?;
        inf_ecs::weapon::carries_ammunition(s)
            .then(|| inf_ecs::weapon::ammo_readout(s.magazine, s.reserve))
    }

    /// **How wanted is the camera subject?** (wave EMS3) — the star row's
    /// condition, and the whole of the host's share of it.
    ///
    /// One Ring-0 call, exactly as [`Self::ammo_readout`] is: the ladder, the
    /// thresholds and the decision that a rating exists at all are sim state,
    /// and a host that divided a heat by three for itself would be a second
    /// opinion about `inf_ecs::crime::WANTED_STARS`.
    fn wanted_readout(sim: &RuntimeSim) -> Option<(u8, u8)> {
        let world = sim.world();
        inf_ecs::crime::wanted_readout(world, inf_ecs::movement::camera_subject(world)?)
    }

    /// **Is the camera subject pointing a weapon?** — the reticle's condition.
    ///
    /// Two halves, and both are needed: the aim mode (RMB), because a carried
    /// rifle hangs where the animation puts it and only an aimed one is on the
    /// line the shot leaves along; and an ammunition clock, because a reticle on
    /// an empty-handed character is a crosshair for a weapon that is not there.
    ///
    /// **The fists are the empty-handed case wearing a clock** (WPN1 audit), so
    /// the second half asks `carries_ammunition` rather than "is there a
    /// `WeaponState`": a punch reaches 1.2 m through an arc and there is no line
    /// for a crosshair to be true about.
    fn is_aiming(sim: &RuntimeSim) -> bool {
        let world = sim.world();
        let Some(entity) =
            inf_ecs::movement::camera_subject(world).and_then(|g| world.entity_of(g))
        else {
            return false;
        };
        let aiming = world
            .world()
            .get::<inf_ecs::components::CharacterMovement>(entity)
            .is_some_and(|cm| {
                cm.rotation_mode == inf_ecs::components::RotationMode::Aiming
                    && !cm.runtime.seat.is_seated()
            });
        aiming
            && world
                .world()
                .get::<inf_ecs::weapon::WeaponState>(entity)
                .is_some_and(inf_ecs::weapon::carries_ammunition)
    }

    /// **What the camera subject is carrying**, as the panel renders it
    /// (island wave I6).
    ///
    /// The projection `inf-ui` cannot make for itself: a UI crate does not
    /// depend on the world model, so the host walks the character's own
    /// `Inventory` and the item catalogue once a frame and hands over a
    /// snapshot. `None` on a level with no character or no bag produces an empty
    /// panel rather than a missing one, which is what a player with nothing sees.
    fn bag_of(sim: &RuntimeSim) -> inf_ui::InventoryView {
        let world = sim.world();
        let Some(actor) = inf_ecs::movement::camera_subject(world) else {
            return inf_ui::InventoryView::default();
        };
        let Some(inv) = inf_ecs::item::inventory_of(world, actor) else {
            return inf_ui::InventoryView::default();
        };
        let defs = inf_ecs::item::item_defs(world);
        inf_ui::InventoryView {
            slots: inv
                .slots
                .iter()
                .enumerate()
                .map(|(i, slot)| match slot {
                    Some(s) => {
                        let def = defs.and_then(|d| d.get(&s.id));
                        inf_ui::InventorySlot {
                            label: def.map(|d| d.label.clone()).unwrap_or_else(|| s.id.clone()),
                            count: s.count,
                            equipped: inv.equipped == Some(i),
                            equippable: def.is_some_and(|d| d.is_weapon()),
                        }
                    }
                    None => inf_ui::InventorySlot::default(),
                })
                .collect(),
        }
    }

    /// One frame: fold input, advance the sim by the elapsed time, project, draw.
    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        // **The window-handle re-attempt** (round-2 finding B7). `resumed`
        // runs its body once per process, so a failed report there had nothing
        // to retry it despite saying it would. Behind the latch, so a
        // successful report costs one `bool` per frame; bounded, so a closed
        // pipe is not written to sixty times a second for ever.
        if let (Some(pie), Some(live)) = (self.pie.as_mut(), self.live.as_ref()) {
            pie.report_window_handle(window_handle_i64(&live.window));
        }
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
        // **`apply_dt`, not `apply`** (island wave I5): the frame time is what
        // makes `InputState::hold_s` a duration rather than a zero, and that is
        // the in-game UI's clock — a menu's key repeat and a rebinding capture
        // both happen while the simulation is paused, so neither can be timed by
        // a sim step. Gameplay durations are a different clock and are
        // accumulated by `RuntimeSim` on its fixed step; see
        // `inf_input::HoldClock` for why the two must not be the same number.
        self.input_state.apply_dt(&events, dt);
        // ── the in-game menu (island wave I5) ──
        //
        //    The `menu` action's edge is read HERE, from the resolved state,
        //    rather than in the key handler: the key that opens it is a
        //    *binding*, so a player who rebound the menu to F1 opens it with F1.
        //    The handler's job is only to keep the keys away from the game once
        //    it is open.
        if self.input_state.just_pressed(inf_input::actions::MENU) {
            self.ui.toggle();
        }
        // ── the inventory panel (island wave I6) ──
        //
        //    The `inventory` action's edge, read from the resolved state for
        //    exactly the reason the menu's is: the key that opens it is a
        //    *binding*, so a player who rebound it opens it with what they
        //    bound. It is the last of the four controls I5 bound against
        //    consumers that did not exist.
        if self.input_state.just_pressed(inf_input::actions::INVENTORY) {
            self.ui.toggle_inventory();
        }
        //    What the panel is showing, and what it decided. The projection is
        //    one way and the verbs the other, and the verbs are applied on the
        //    SIM's step rather than here — see `PlayerUi::pending`.
        self.ui.set_bag(Self::bag_of(&self.sim));
        for verb in self.ui.take_inventory_verbs() {
            self.sim.apply_inventory_verb(verb);
        }
        //    Tab pauses the single-player simulation, and the pause is on the
        //    SIM rather than on this host — see `inf_ui::menu`'s ruling and
        //    `RuntimeSim::set_sim_paused`.
        self.sim.set_sim_paused(self.ui.pauses_sim());
        self.ui.report_unconsumed(&self.input_state);
        let held = input::held_actions(&self.input_state, dt);
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
        let steps = self.sim.steps();
        let Some(live) = self.live.as_mut() else {
            return;
        };
        // **The predictor's committed sample** (P28.4), at the same sync point
        // and for the same reason the terrain's cut advances here: after the
        // fixed steps and before the projection, keyed on the SIM's step count
        // rather than on a frame index. A frame that ran no fixed step commits
        // nothing and is refused, which is exactly right — the horizon is
        // measured in committed ticks, and a frame is not one.
        if let Some(v) = view.as_ref() {
            live.host.commit_camera(steps, v);
        }
        if live.host.is_lost() {
            tracing::warn!("inf-player: device lost — rebuilding GPU stack");
            // The dead device's swapchain goes first: `build_host` creates a
            // second `Instance` + `Surface` for this same window, and the old
            // chain would otherwise live until the assignment below. See
            // `PlayerRenderHost::release_surface`.
            live.host.release_surface();
            match Self::build_host(&live.window, self.width, self.height, self.render) {
                Ok(mut host) => {
                    host.set_vmeshes(self.vmeshes.clone());
                    host.set_scatter_meshes(self.scatter_meshes.clone());
                    host.set_skinned(self.skinned.clone());
                    host.set_voxel_assets(self.voxel_assets.clone());
                    // P26.4: the atlas and the indirection buffer belonged to the
                    // dead device, so the level's virtual textures are registered
                    // again here — the same call, the same order, the same pages.
                    host.set_material_content(self.materials.clone());
                    live.host = host;
                }
                Err(e) => {
                    tracing::error!("inf-player: GPU rebuild failed: {e}");
                    event_loop.exit();
                    return;
                }
            }
        }
        // The eye the render host's own camera-driven residency (P21.2 voxel
        // volumes) is measured from — the SAME point `sync_render_terrain` above
        // took, so a frame with no view (occluded/minimized) pages neither.
        let eye = view.as_ref().map(|v| v.eye_world).unwrap_or(DVec3::ZERO);
        live.host.project(&self.sim, alpha, eye);
        if self.debug_cells {
            live.host.draw_cell_overlay(&self.sim);
        }
        // I6: this step's tracers. After the cell overlay, which clears the
        // debug list — see `draw_tracers` for why a tracer is a line and not a
        // particle.
        live.host.draw_tracers(&self.sim);
        // EMS2: and the hoses, out of the same list and into the same layer.
        live.host.draw_extinguish(&self.sim);
        // ── the in-game UI (island wave I5) ──
        //
        //    BETWEEN the projection and the render, which is the only window in
        //    which both halves of the frame exist: `project_scene_full` clears
        //    and rebuilds every other field of the scene, and the UI is the one
        //    the host owns.
        //
        //    Laid out for the SURFACE's configured size rather than the
        //    window's: they differ for the frames a resize debounce is pending,
        //    and a menu laid out for a size the swap chain does not have is
        //    stretched by exactly that ratio.
        let (sw, sh) = live.host.surface_size();
        let viewport = glam::Vec2::new(sw.max(1) as f32, sh.max(1) as f32);
        self.ui.build(dt, viewport, self.input_state.map());
        if let Some(v) = view.as_ref() {
            if let Some((at, text)) = Self::interaction_prompt(&self.sim, v, &self.input_state) {
                self.ui.prompt(Some(at), &text);
            }
        }
        // VEH2b: the driver's own instruments. Absent unless somebody is at a
        // wheel, which is what makes it a readout rather than a HUD.
        //
        // WPN1 puts the ammunition beside it, in the same slot and behind an
        // `else`: the two are one line at the bottom of the screen, a character
        // cannot be at a wheel and pointing a rifle at the same time in this
        // engine (`step_driving` parks the collider and `clear_edges` eats the
        // attack), and two readouts stacked in one place would be a HUD that
        // draws over itself on the one frame both are true.
        if let Some(text) = Self::drive_readout(&mut self.sim) {
            self.ui.readout(&text);
        } else if let Some(text) = Self::ammo_readout(&self.sim) {
            self.ui.readout(&text);
        }
        // EMS3: the wanted rating, top-left. A NEW anchor rather than a third
        // tenant of the readout slot above, and the reason is that slot's own
        // stated rule: the driver's instruments and the ammunition count share
        // it behind an `else` because they cannot both be true, and a wanted
        // rating is true AT THE SAME TIME as either of them — being chased while
        // driving and being chased while shooting are the mechanic. Nothing else
        // in this engine draws in the top-left.
        //
        // **LEDGER SENTENCE: these stars will never appear in the editor's
        // Simulate viewport**, and that is a property of the architecture rather
        // than an oversight. `set_ui` has exactly one caller in the tree and it
        // is eleven lines below this one; nothing under `editor/` ever builds a
        // `UiDrawList` at all, so Simulate has no in-game HUD of any kind — no
        // prompt, no readout, no reticle and now no rating. A wanted level is
        // still fully simulated there and is still visible in a trace and in the
        // ledger; what is missing is the drawing. Embedded PIE and a new-window
        // PIE both run this host and do show it.
        if let Some((earned, slots)) = Self::wanted_readout(&self.sim) {
            self.ui.wanted(earned, slots);
        }
        // …and the reticle, which is the aim's own half. Only while aiming: see
        // `inf_ui::view::reticle` for why a permanent crosshair would be a claim
        // this engine's carried weapons do not honour.
        if Self::is_aiming(&self.sim) {
            self.ui.reticle();
        }
        live.host.set_ui(self.ui.list());
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
                // geometry (meshlet path / classic fallback per the auto-tier),
                // and the skeletal store so a `SkeletalMesh` renders its real
                // posed geometry rather than a placeholder cube.
                host.set_vmeshes(self.vmeshes.clone());
                host.set_scatter_meshes(self.scatter_meshes.clone());
                host.set_skinned(self.skinned.clone());
                // P21.1: and the `.inf_voxel` source, so a placed cave draws its
                // carved surface rather than nothing.
                host.set_voxel_assets(self.voxel_assets.clone());
                // P26.4 — clause 0. The level's bound `.inf_mat` maps become
                // virtual textures here, through the door the editor viewport
                // also calls, so a shipped build and a preview page identically.
                host.set_material_content(self.materials.clone());
                // P22.4 — THE PER-TIER DEBRIS BUDGET, and the only place in the
                // engine where a `RenderTier` becomes a `DebrisBudget`.
                //
                // It happens *here*, in the window, because this is the one place
                // that holds both a detected tier and a live session. And it is
                // asked through `debris_budget_for_session` rather than
                // `debris_budget_for`, with `self.pie.is_some()`, because **an
                // embedded PIE session is windowed** — it comes through this exact
                // path with a real host and a real tier. The first cut clamped it,
                // which meant that on any Medium or Low machine the editor's
                // Simulate ran the engine default and the PIE it had just spawned
                // ran a smaller one; the budget is read by `step_fractures`, so a
                // reclaim removes a solver body and the two were different
                // simulations. A preview must run what it previews.
                if let Some(spec) =
                    inf_render::debris_budget_for_session(host.tier(), self.pie.is_some())
                {
                    self.sim.set_debris_budget(inf_physics::d3::DebrisBudget {
                        max_live: spec.max_live,
                        lifetime_s: spec.lifetime_s,
                    });
                }
                // P24.4 — THE PER-TIER HAIR DETAIL, the second `RenderTier` → budget
                // mapping and the only one that reaches animation.
                //
                // Asked through `hair_detail_for` and **not** through a
                // `…_for_session(tier, pie)` twin, which is the deliberate
                // difference from the line above: the debris budget is read by
                // `step_fractures` and therefore *changes the simulation*, so an
                // embedded PIE must run the same budget the editor's Simulate does.
                // Ribbon geometry changes nothing in `state_bytes` (asserted by
                // `inf-ecs`' `the_detail_draws_differently_and_traces_identically`),
                // so here the right answer is the other one: PIE draws what the
                // shipped player would draw on this machine.
                let hair = inf_render::hair_detail_for(host.tier());
                self.sim.set_hair_detail(inf_anim::HairDetail {
                    guide_stride: hair.guide_stride,
                    strands_per_guide: hair.strands_per_guide,
                });
                // Report our native window handle so the editor can reparent us
                // into the viewport slot (embedded PIE).
                // C4-44 / unit U4 ("failure latched as applied") made this
                // latch success-only; round-2 finding B7 gave it the retry the
                // warning already claimed — `frame()` calls the same door, and
                // this arm runs once per process. See
                // `PieLink::report_window_handle`.
                if let Some(pie) = self.pie.as_mut() {
                    pie.report_window_handle(window_handle_i64(&window));
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

    /// Raw **device** motion (P29.3) — the unaccelerated delta, which is what a
    /// look control needs. The window-cursor delta reported by
    /// `WindowEvent::CursorMoved` is clipped by the screen edge, so a player
    /// turning right stops turning when the pointer reaches the monitor's edge.
    /// That is the oldest bug in first-person games and it is avoided by reading
    /// the device instead.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            if let Some(live) = self.live.as_mut() {
                live.pending.push(InputEvent::MouseMotion {
                    delta: [delta.0 as f32, delta.1 as f32],
                });
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
                    let pressed = event.state == ElementState::Pressed;
                    // **Escape leaves the game — unless the dialog wants it**
                    // (island wave I5). Escape is the dialog's cancel and its
                    // close, and a build that quit on it would make the settings
                    // screen unusable.
                    if code == KeyCode::Escape && pressed && !self.ui.menu.open {
                        event_loop.exit();
                        return;
                    }
                    if let Some(name) = input::keycode_to_code(code) {
                        // **The dialog gets it first, and what it takes never
                        // reaches the game.** The same press that moves the
                        // cursor would otherwise also fire a weapon — and a key
                        // being *captured* for a rebinding would fire the verb it
                        // is being taken from, which is the worst of the two.
                        let mut map = self.input_state.map().clone();
                        let verdict = self.ui.key(name, pressed, &mut map);
                        if verdict.changed() {
                            // **Rebuilt from the level's table**, not patched:
                            // the look tuning is a multiplier on what the
                            // project authored, and applying it to the live map
                            // would compound it once a frame. Re-seating keeps
                            // the raw device state (a key held across a
                            // rebinding is still held) and re-resolves against
                            // the new table.
                            self.input_state.set_map(self.ui.tuned_map());
                            self.ui.apply_to_sim(&mut self.sim);
                        }
                        if verdict.consumed {
                            return;
                        }
                        if let Some(live) = self.live.as_mut() {
                            live.pending.push(InputEvent::Key {
                                code: name.to_string(),
                                pressed,
                            });
                        }
                    }
                }
            }
            // ── P29.3 mouse ── buttons are level state; the wheel is a delta.
            //    Cursor MOTION is deliberately not read here: a window-cursor
            //    delta stops at the edge of the screen, so it is taken from
            //    `device_event` below instead.
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(button) = input::mouse_button(button) {
                    let pressed = state == ElementState::Pressed;
                    // A mouse button is a bindable source, so a running capture
                    // takes it (island wave I5). Nothing else the dialog does
                    // reads one, and a release is only interesting to the game.
                    // A release is never consumed — see `PlayerUi::key` for the
                    // stuck-key measurement that says why.
                    if pressed {
                        let mut map = self.input_state.map().clone();
                        let verdict = self.ui.mouse(button, &mut map);
                        if verdict.changed() {
                            self.input_state.set_map(self.ui.tuned_map());
                            self.ui.apply_to_sim(&mut self.sim);
                        }
                        if verdict.consumed {
                            return;
                        }
                    }
                    if let Some(live) = self.live.as_mut() {
                        live.pending
                            .push(InputEvent::MouseButton { button, pressed });
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(x, y) => [x, y],
                    // A pixel delta (trackpad) is divided by a nominal line
                    // height so both devices speak the same unit; the binding's
                    // scale then converts once.
                    MouseScrollDelta::PixelDelta(p) => [p.x as f32 / 16.0, p.y as f32 / 16.0],
                };
                if let Some(live) = self.live.as_mut() {
                    live.pending.push(InputEvent::MouseWheel { delta: d });
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
            // **Focus loss releases everything** (round-2 finding R2-9). The
            // raw device sets accumulate across events and are cleared only by
            // the matching release — and the OS sends none when the window
            // stops being focused with W held or a stick pushed. The character
            // otherwise keeps running for the rest of the session, through PIE,
            // through a level change, and into the trace a parity gate compares.
            //
            // The touch strand goes with it: a suspend delivers no `Ended`
            // either, so an in-flight virtual stick stays pushed.
            WindowEvent::Focused(false) => {
                #[cfg(any(target_arch = "wasm32", target_os = "android"))]
                {
                    let synth = self.touch.cancel_all();
                    self.input_state.apply(&synth);
                }
                self.input_state.release_all();
                if let Some(live) = self.live.as_mut() {
                    live.pending.clear();
                }
            }
            WindowEvent::RedrawRequested => self.frame(event_loop),
            _ => {}
        }
    }
}

/// Run the windowed player over `sim` with input bindings `map`. Blocks until the
/// window closes; returns an error only if the event loop fails to start.
#[allow(clippy::too_many_arguments)]
pub fn run(
    title: String,
    width: u32,
    height: u32,
    sim: RuntimeSim,
    map: InputMap,
    vmeshes: Arc<VmeshRegistry>,
    scatter_meshes: Arc<inf_render::ScatterMeshes>,
    skinned: Arc<SkinnedRegistry>,
    voxel_assets: Arc<VoxelRegistry>,
    materials: Arc<crate::MaterialContent>,
    render: inf_scene::RenderSettingsRecord,
    debug_cells: bool,
) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = PlayerApp::new(title, width, height, sim, map, vmeshes, render);
    app.scatter_meshes = scatter_meshes;
    app.skinned = skinned;
    app.voxel_assets = voxel_assets;
    app.materials = materials;
    app.debug_cells = debug_cells;
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("run_app: {e}"))
}

/// Run the windowed player as a **play-in-editor** window: identical to [`run`]
/// but wired to a PIE control channel (Pause/Resume/Step/Stop/Eject/SetViewport)
/// and a report sink (`Window` handle for reparenting, `Paused`/`Resumed`/…).
/// Blocks until Stop or the window closes.
// Nine parameters trips clippy's arity lint. Bundling them would hide the two
// that matter — `voxel_assets` was ADDED here in P21.4 because a windowed PIE
// session was binding no volumes and drawing no caves, and `skinned` in P24.1
// because it was drawing every character as a placeholder cube. A struct is
// exactly where a field like that goes back to being easy to forget to fill.
#[allow(clippy::too_many_arguments)]
pub fn run_pie(
    title: String,
    width: u32,
    height: u32,
    sim: RuntimeSim,
    map: InputMap,
    control: Receiver<EditorToPlayer>,
    out: Box<dyn Write>,
    voxel_assets: Arc<VoxelRegistry>,
    skinned: Arc<SkinnedRegistry>,
    materials: Arc<crate::MaterialContent>,
) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    // PIE streams no vmesh assets yet (a documented follow-up); a *rigid*
    // `MeshRef.asset` still renders as a placeholder cube in PIE until the payload
    // carries the vmesh index. A `SkeletalMesh` no longer does — see `skinned`
    // below and `ScenePayload::meshes` (v7).
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
    // P21.4: the payload's `.inf_voxel` bytes. Without them the windowed PIE
    // player binds no volumes, so `overlay_sim` has nothing to mirror the
    // Blueprint's carves *into* and an embedded session draws no caves at all —
    // while the headless one, which the gate drives, draws them correctly. Passed
    // in rather than defaulted so the gap cannot reopen silently.
    app.voxel_assets = voxel_assets;
    // P24.1: the payload's `.inf_mesh` / `.inf_skel` / `.inf_anim` bytes
    // (`ScenePayload` v7). Without them `resolve_skinned` misses on the mesh and
    // every character in a windowed PIE session draws as the slate placeholder
    // cube — while the headless PIE the gates drive, and the shipped build, draw
    // real posed geometry. The identical class P21.4 closed for voxel volumes,
    // and passed in rather than defaulted for the identical reason.
    app.skinned = skinned;
    // P26.4: the payload's `.inf_matd` records + `.inf_tex` containers
    // (`ScenePayload` v8). Without them a windowed PIE session renders every
    // bound surface off its scalar attributes while the shipped build textures
    // it — the identical class P21.4 closed for voxel volumes and P24.1 for
    // skeletal meshes, and passed in for the identical reason.
    app.materials = materials;
    // **AND THE FOURTH OF THE CLASS IS STILL OPEN** (wave TER2b). There is no
    // `app.scatter_meshes = …` line here, so `PlayerApp::new`'s empty table
    // stands and every scattered instance in a windowed PIE session draws the
    // placeholder cube — while a cooked boot, which fills the table in
    // `run_windowed`, draws the authored ground cover. That is the identical
    // shape as the three lines above, and it is stated here rather than only in
    // a memo because this is where the previous three were closed and where a
    // reader looks to find out which are.
    //
    // **The fix needs no schema move**: `ScenePayload::meshes` is already a
    // general `Vec<(Uuid, Vec<u8>)>` of `.inf_mesh` bytes that only
    // `SkeletalMesh.mesh` fills today, so adding a level's scatter kinds to it
    // is a *use* of a field, not a bump. `pie_equals_shipping_on_an_island_drive`
    // cannot see this: it compares SIMULATION state, and the gap is at the frame.
    app.pie = Some(PieLink {
        control,
        out,
        hwnd_reported: false,
        hwnd_attempts: 0,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error, ErrorKind};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::mpsc::channel;

    /// A writer that fails the first write of each of its first `fail_for`
    /// messages, then succeeds — the shape of an editor that has not yet
    /// drained its end of our stdout.
    ///
    /// `attempts` counts *messages started* (a `write_msg` issues several
    /// `write` calls for one frame), so the two counters below mean what their
    /// names say.
    struct Flaky {
        fail_for: usize,
        attempts: Arc<AtomicUsize>,
        delivered: Arc<AtomicUsize>,
        mid_message: bool,
    }

    impl Write for Flaky {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if !self.mid_message {
                let n = self.attempts.fetch_add(1, AtomicOrdering::Relaxed);
                if n < self.fail_for {
                    return Err(Error::new(ErrorKind::WouldBlock, "pipe is not ready"));
                }
                self.mid_message = true;
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if self.mid_message {
                self.mid_message = false;
                self.delivered.fetch_add(1, AtomicOrdering::Relaxed);
            }
            Ok(())
        }
    }

    /// `(link, attempts, delivered)`.
    fn link(fail_for: usize) -> (PieLink, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let (_tx, control) = channel();
        let attempts = Arc::new(AtomicUsize::new(0));
        let delivered = Arc::new(AtomicUsize::new(0));
        (
            PieLink {
                control,
                out: Box::new(Flaky {
                    fail_for,
                    attempts: attempts.clone(),
                    delivered: delivered.clone(),
                    mid_message: false,
                }),
                hwnd_reported: false,
                hwnd_attempts: 0,
            },
            attempts,
            delivered,
        )
    }

    /// **Round-2 finding B7.** The success-only latch was correct and the retry
    /// it promised did not exist: the whole block lived in `resumed`, which
    /// returns early once `live` is set and therefore runs its body once per
    /// process. `hwnd_reported` had exactly one reader — itself.
    #[test]
    fn a_failed_handle_report_is_re_attempted_until_it_lands() {
        let (mut pie, _, _) = link(3);

        // Three failures: each says "try me again".
        for attempt in 1..=3 {
            assert!(
                pie.report_window_handle(0x1234),
                "attempt {attempt} did not ask to be retried"
            );
            assert!(!pie.hwnd_reported, "attempt {attempt} latched on a failure");
        }
        // The fourth write succeeds and latches.
        assert!(
            !pie.report_window_handle(0x1234),
            "a successful report still asked to be retried"
        );
        assert!(pie.hwnd_reported);
    }

    /// The latch really is a latch: a reported handle is never written twice,
    /// however many frames call the door.
    #[test]
    fn a_reported_handle_is_written_exactly_once() {
        let (mut pie, _, delivered) = link(0);
        for _ in 0..50 {
            pie.report_window_handle(0x1234);
        }
        assert_eq!(
            delivered.load(AtomicOrdering::Relaxed),
            1,
            "the handle was reported more than once"
        );
    }

    /// And the retry is bounded: a permanently closed pipe is not written to
    /// sixty times a second for the life of the session.
    #[test]
    fn the_retry_gives_up_rather_than_spinning_on_a_dead_pipe() {
        let (mut pie, attempts, delivered) = link(usize::MAX);
        for _ in 0..(MAX_HWND_ATTEMPTS as usize * 3) {
            pie.report_window_handle(0x1234);
        }
        assert!(!pie.hwnd_reported);
        assert_eq!(delivered.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            attempts.load(AtomicOrdering::Relaxed),
            MAX_HWND_ATTEMPTS as usize,
            "the retry is not bounded by MAX_HWND_ATTEMPTS"
        );
        assert!(
            !pie.report_window_handle(0x1234),
            "an exhausted retry still asked to be called again"
        );
    }
}
