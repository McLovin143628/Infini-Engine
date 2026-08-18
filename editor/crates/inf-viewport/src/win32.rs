//! Win32 child-window host. A dedicated thread owns the child HWND (Win32
//! ties a window to the thread that created it), pumps its messages, drives
//! the render loop, and runs the editor camera; commands arrive over a channel.
//!
//! Input model (UE parity):
//! - RMB captures for the flycam (WASD/QE, wheel = speed, Shift = boost).
//! - Alt+LMB orbit, MMB (or Alt+MMB) pan, Alt+RMB dolly, all around the pivot.
//! - Wheel with no button dollies toward the look point.
//! - F focuses; Ctrl+1..9 store camera bookmarks, 1..9 recall them.
//! - Global-shortcut chords (Ctrl+…, F11) are forwarded to the webview so the
//!   command palette etc. still work while the 3D view holds OS focus
//!   (focus handoff, P2.3.4).

use std::cell::RefCell;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, GetKeyState, ReleaseCapture, SetCapture, SetFocus, VK_CONTROL, VK_MENU,
    VK_SHIFT,
};
use windows::Win32::UI::Input::{
    GetRawInputData, RegisterRawInputDevices, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE,
    RAWINPUTDEVICE_FLAGS, RAWINPUTHEADER, RID_INPUT, RIM_TYPEMOUSE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetCursorPos, GetWindowLongPtrW,
    GetWindowRect, IsWindow, PeekMessageW, RegisterClassW, SetCursorPos, SetParent,
    SetWindowLongPtrW, SetWindowPos, ShowCursor, ShowWindow, TranslateMessage, GWL_STYLE, HWND_TOP,
    MSG, PM_REMOVE, SWP_NOACTIVATE, SWP_NOZORDER, SW_HIDE, SW_SHOWNA, WINDOW_EX_STYLE,
    WM_CAPTURECHANGED, WM_ERASEBKGND, WM_INPUT, WM_KEYDOWN, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_RBUTTONDOWN, WM_RBUTTONUP,
    WM_SYSKEYDOWN, WNDCLASSW, WS_CHILD, WS_CLIPSIBLINGS, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

use glam::DVec2;

use crate::camera::{
    BiomeSettings, Bookmarks, Camera2D, EditorCamera, FlyInput, FoliageSettings, GizmoSpace,
    NavInput, NavMode, SculptSettings, Snap2DSettings, SnapSettings, ToolMode, ViewportMode,
    VoxelSettings, WaterSettings,
};
use crate::host::EngineHost;
use crate::{KeyChord, SharedScene, SurfaceTarget, ViewportEvent, ViewportEventSink, ViewportRect};
use inf_editor_core::ipc::{GizmoModeDto, ViewModeDto, ViewportToolStatusDto};
use inf_render::{GizmoMode, ViewMode};

/// Map the IPC gizmo-mode DTO to the renderer enum (Wave 2). Kept next to the
/// reverse map so the two stay in lockstep.
fn to_gizmo_mode(d: GizmoModeDto) -> GizmoMode {
    match d {
        GizmoModeDto::Translate => GizmoMode::Translate,
        GizmoModeDto::Rotate => GizmoMode::Rotate,
        GizmoModeDto::Scale => GizmoMode::Scale,
    }
}

/// Map the IPC view-mode DTO to the renderer enum (R-P2). The renderer clamps
/// Wireframe→Unlit when the adapter lacks `POLYGON_MODE_LINE`.
fn to_view_mode(d: ViewModeDto) -> ViewMode {
    match d {
        ViewModeDto::Lit => ViewMode::Lit,
        ViewModeDto::Unlit => ViewMode::Unlit,
        ViewModeDto::Wireframe => ViewMode::Wireframe,
        // P19.2: needs no GPU feature, so it never degrades.
        ViewModeDto::Biomes => ViewMode::Biomes,
        ViewModeDto::VtResidency => ViewMode::VtResidency,
        // P27.5: the same, one virtual system over.
        ViewModeDto::VsmPages => ViewMode::VsmPages,
    }
}

/// Map the renderer gizmo enum back to the IPC DTO for the `viewport://gizmo`
/// echo (so a W/E/R keypress updates the toolbar).
fn from_gizmo_mode(m: GizmoMode) -> GizmoModeDto {
    match m {
        GizmoMode::Translate => GizmoModeDto::Translate,
        GizmoMode::Rotate => GizmoModeDto::Rotate,
        GizmoMode::Scale => GizmoModeDto::Scale,
    }
}

/// Depth of an entity in the hierarchy (0 = root), for sorting a multi-select
/// gizmo writeback parents-first (Wave 2). Walks parent links via the ECS world.
fn hierarchy_depth(doc: &inf_editor_core::scene::SceneDoc, guid: uuid::Uuid) -> usize {
    let world = doc.world();
    let Some(mut e) = world.entity_of(guid) else {
        return 0;
    };
    let mut depth = 0;
    while let Some(p) = world.parent_of(e) {
        depth += 1;
        e = p;
    }
    depth
}

enum Cmd {
    SetRect(ViewportRect),
    SetVisible(bool),
    Drop {
        x: f32,
        y: f32,
        payload: String,
    },
    /// Switch the active projection (Perspective ↔ 2D ortho) from the toolbar.
    SetMode(ViewportMode),
    /// Replace the 2D-mode snapping configuration from the toolbar.
    SetSnap2D(Snap2DSettings),
    /// Switch the active tool (Select ↔ Sculpt) from the toolbar (P10.2b).
    SetToolMode(ToolMode),
    /// Replace the sculpt brush configuration from the toolbar (P10.2b).
    SetSculpt(SculptSettings),
    /// Replace the foliage brush configuration from the toolbar (E-P6).
    SetFoliage(FoliageSettings),
    /// Replace the biome brush configuration from the toolbar (P19.2).
    SetBiome(BiomeSettings),
    /// Replace the water-tool configuration (P20.4).
    SetWater(WaterSettings),
    /// Replace the voxel carve-tool configuration (P21.2).
    SetVoxel(VoxelSettings),
    /// Whether an in-editor Simulate session is live (P29.6). While it is, a
    /// plain LMB in the viewport captures the mouse for the GAME camera instead
    /// of picking, and the deltas are forwarded rather than steering the editor
    /// camera. Pushed by `sim_start`/`sim_stop`, backend to backend.
    SetSimRunning(bool),
    /// Push a terrain entity's resolved biome overlay palette (P19.2) — Ring 2
    /// owns the asset lookup, the viewport only draws it.
    SetBiomePalette(uuid::Uuid, Vec<[f32; 4]>),
    /// Per-terrain water-level hints by biome id (P20.4).
    SetWaterHints(uuid::Uuid, Vec<Option<f64>>),
    /// Set the transform-gizmo mode (translate/rotate/scale) from the toolbar or
    /// palette (Wave 2). The viewport echoes changes back on `viewport://gizmo`.
    SetGizmo(GizmoMode),
    /// Switch the gizmo orientation frame (World ↔ Local) from the toolbar
    /// (Wave 2).
    SetGizmoSpace(GizmoSpace),
    /// Replace the 3D transform-gizmo snap increments from the toolbar (Wave 2).
    SetSnap3D(SnapSettings),
    /// Set the shading view mode (Lit / Unlit / Wireframe) from the toolbar
    /// (R-P2). The renderer clamps Wireframe→Unlit if unsupported.
    SetViewMode(ViewMode),
    /// Point terrain streaming at a project's content root (or `None` to disable
    /// it) — the P16.4a `project://changed` wiring.
    SetContentRoot(Option<std::path::PathBuf>),
    /// Rebuild the loose `.inf_terrain` index in place (a terrain import landed).
    RefreshAssetIndex,
    /// Reopen every live terrain stream's `.inf_terrain` in place — a save just
    /// wrote sculpt/paint edits back into it (P16.4b).
    ReloadTerrainStores,
    /// Re-walk the loose `.inf_voxel` index in place — a save just folded carve
    /// edits back into the assets (P21.2). The twin of `ReloadTerrainStores`.
    ReloadVoxelStores,
    /// Release every terrain stream — the document was replaced (P16.4b).
    ClearStreams,
    /// Adopt a foreign (PIE player) window into the viewport slot: reparent it
    /// to our parent, position it at the hole, and hide our own child (embedded
    /// PIE, P9.4). The `isize` is the foreign HWND.
    EmbedForeign(isize),
    /// Release the embedded foreign window and restore our own child.
    ReleaseForeign,
    Destroy,
}

/// Cheap-to-clone handle for controlling the viewport thread.
///
/// **It holds no shared stores** (P23.2a — the hoist). The carve working set
/// (P21.2) and the Simulate fracture map (P22.3) used to be created inside
/// `spawn` and held here, which made them *per viewport*: a second viewport
/// would have minted a second carve store, and Ring 2's save path — which has
/// no host and no camera — would have had to ask a viewport which of them the
/// author's edits were in. They are created once by Ring 2
/// (`commands::SharedStores`) and handed to `spawn`, so there is exactly one of
/// each per PROCESS however many viewports exist, and every reader resolves
/// them without a viewport in hand.
pub struct ViewportHandle {
    tx: Sender<Cmd>,
}

impl ViewportHandle {
    /// Move/resize the child window (physical pixels, parent-client-relative).
    pub fn set_rect(&self, rect: ViewportRect) {
        let _ = self.tx.send(Cmd::SetRect(rect));
    }

    /// Show/hide the child window. The shell hides the viewport while an
    /// HTML overlay (menu, palette, dialog, drag ghost) is open — the native
    /// window otherwise draws OVER the webview (airspace rule), occluding
    /// any overlay that crosses the hole. P2.1 explores flash-free
    /// alternatives (window-region cutouts / last-frame freeze).
    pub fn set_visible(&self, visible: bool) {
        let _ = self.tx.send(Cmd::SetVisible(visible));
    }

    /// Hand off a drag-drop that ended over the viewport hole (Spike A stub:
    /// the webview keeps mouse capture during HTML drags, so the drop point
    /// arrives via IPC in viewport-local physical pixels).
    pub fn drop_payload(&self, x: f32, y: f32, payload: &str) {
        let _ = self.tx.send(Cmd::Drop {
            x,
            y,
            payload: payload.to_owned(),
        });
    }

    /// Switch the active viewport projection (Perspective ↔ 2D ortho).
    pub fn set_mode(&self, mode: ViewportMode) {
        let _ = self.tx.send(Cmd::SetMode(mode));
    }

    /// Replace the 2D-mode snapping configuration (grid + pixel snap).
    pub fn set_snap_2d(&self, snap: Snap2DSettings) {
        let _ = self.tx.send(Cmd::SetSnap2D(snap));
    }

    /// Switch the active viewport tool (Select ↔ Sculpt) (P10.2b).
    pub fn set_tool_mode(&self, mode: ToolMode) {
        let _ = self.tx.send(Cmd::SetToolMode(mode));
    }

    /// Replace the sculpt brush configuration (op / radius / strength / falloff).
    pub fn set_sculpt(&self, sculpt: SculptSettings) {
        let _ = self.tx.send(Cmd::SetSculpt(sculpt));
    }

    /// Replace the foliage brush configuration (radius / density / kind / …).
    pub fn set_foliage(&self, foliage: FoliageSettings) {
        let _ = self.tx.send(Cmd::SetFoliage(foliage));
    }

    /// Replace the biome brush configuration (radius / strength / falloff / id).
    pub fn set_biome(&self, biome: BiomeSettings) {
        let _ = self.tx.send(Cmd::SetBiome(biome));
    }

    /// Replace the voxel carve-tool configuration (sub-mode / radius / depth /
    /// carve-or-fill / material) — P21.2.
    pub fn set_voxel(&self, voxel: VoxelSettings) {
        let _ = self.tx.send(Cmd::SetVoxel(voxel));
    }

    /// Tell the viewport whether an in-editor Simulate session is live (P29.6).
    pub fn set_sim_running(&self, running: bool) {
        let _ = self.tx.send(Cmd::SetSimRunning(running));
    }

    /// Replace the water-tool configuration (kind / river dimensions / level
    /// offset / resolved biome hint) — P20.4.
    pub fn set_water(&self, water: WaterSettings) {
        let _ = self.tx.send(Cmd::SetWater(water));
    }

    /// Push a terrain's resolved biome overlay palette (P19.2). An empty palette
    /// clears it.
    pub fn set_biome_palette(&self, entity: uuid::Uuid, palette: Vec<[f32; 4]>) {
        let _ = self.tx.send(Cmd::SetBiomePalette(entity, palette));
    }

    /// Push a terrain's per-biome water-level hints (P20.4). An all-`None` table
    /// clears them.
    pub fn set_water_hints(&self, entity: uuid::Uuid, hints: Vec<Option<f64>>) {
        let _ = self.tx.send(Cmd::SetWaterHints(entity, hints));
    }

    /// Set the transform-gizmo mode (translate/rotate/scale) from the toolbar or
    /// command palette (Wave 2). Takes the IPC DTO so Ring 2 needn't name the
    /// platform-gated `inf_render::GizmoMode`.
    pub fn set_gizmo_mode(&self, mode: GizmoModeDto) {
        let _ = self.tx.send(Cmd::SetGizmo(to_gizmo_mode(mode)));
    }

    /// Switch the gizmo orientation frame (World ↔ Local) (Wave 2).
    pub fn set_gizmo_space(&self, space: GizmoSpace) {
        let _ = self.tx.send(Cmd::SetGizmoSpace(space));
    }

    /// Replace the 3D transform-gizmo snap increments (Wave 2).
    pub fn set_snap_3d(&self, snap: SnapSettings) {
        let _ = self.tx.send(Cmd::SetSnap3D(snap));
    }

    /// Set the shading view mode (Lit / Unlit / Wireframe) from the toolbar
    /// (R-P2). Takes the IPC DTO so Ring 2 needn't name the renderer enum.
    pub fn set_view_mode(&self, mode: ViewModeDto) {
        let _ = self.tx.send(Cmd::SetViewMode(to_view_mode(mode)));
    }

    /// Point terrain streaming at a project's content root (P16.4a). Rescans the
    /// loose `.inf_terrain` index and drops every live stream, so a project
    /// switch can never serve the previous project's pages.
    pub fn set_content_root(&self, root: Option<std::path::PathBuf>) {
        let _ = self.tx.send(Cmd::SetContentRoot(root));
    }

    /// Rebuild the loose `.inf_terrain` index without dropping live streams —
    /// pushed when a terrain import finishes (P16.4a).
    pub fn refresh_asset_index(&self) {
        let _ = self.tx.send(Cmd::RefreshAssetIndex);
    }

    /// Reopen every live terrain stream's `.inf_terrain` in place — pushed when a
    /// save wrote sculpt/paint edits back into the asset (P16.4b). Live streams
    /// keep their resident pages, so saving does not blink the terrain.
    pub fn reload_terrain_stores(&self) {
        let _ = self.tx.send(Cmd::ReloadTerrainStores);
    }

    /// Re-walk the loose `.inf_voxel` index in place — pushed when a save has
    /// folded carve edits back into the assets (P21.2). Loaded volumes keep their
    /// chunks and meshes, so saving does not blink the caves.
    pub fn reload_voxel_stores(&self) {
        let _ = self.tx.send(Cmd::ReloadVoxelStores);
    }

    /// Release every terrain stream (its pages, its edit pins and its
    /// `.inf_terrain` payload) — pushed when the open document is replaced by
    /// File ▸ Open / File ▸ New (P16.4b).
    pub fn clear_streams(&self) {
        let _ = self.tx.send(Cmd::ClearStreams);
    }

    /// Adopt a foreign (PIE player) window into the viewport slot (embedded PIE,
    /// P9.4). The foreign window is reparented under our parent, sized to the
    /// hole, and our own render child is hidden. `hwnd` is the player's HWND.
    pub fn embed_foreign(&self, hwnd: isize) {
        let _ = self.tx.send(Cmd::EmbedForeign(hwnd));
    }

    /// Release the embedded foreign window and restore our own render child.
    pub fn release_foreign(&self) {
        let _ = self.tx.send(Cmd::ReleaseForeign);
    }

    /// Tear down the viewport thread and its window.
    pub fn destroy(&self) {
        let _ = self.tx.send(Cmd::Destroy);
    }
}

/// Spawn the viewport thread: creates a `WS_CHILD` window parented to
/// `parent_hwnd`, brings up the engine on it, and renders until destroyed.
/// `sink` receives events the viewport surfaces back (forwarded key chords).
///
/// `volumes` and `fractures` are the PROCESS's shared stores, created once by
/// Ring 2 and passed in (P23.2a — the hoist). They are deliberately not created
/// here: `spawn` runs once per viewport, so creating them here made them per
/// viewport, and the save path would have had to pick one. An engine init that
/// fails simply drops its clone; the store outlives the viewport that never
/// came up, and stages exactly what was carved into it, which is nothing.
pub fn spawn(
    parent_hwnd: isize,
    sink: ViewportEventSink,
    scene: SharedScene,
    volumes: inf_editor_core::voxel_store::SharedVoxelVolumes,
    fractures: inf_editor_core::simulate::SharedFractures,
) -> ViewportHandle {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("inf-viewport".into())
        .spawn(move || thread_main(parent_hwnd, rx, sink, scene, volumes, fractures))
        .expect("failed to spawn inf-viewport thread");
    ViewportHandle { tx }
}

/// Which mouse gesture currently owns capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Capture {
    #[default]
    None,
    Fly,
    Orbit,
    Pan,
    Dolly,
    /// **Mouse-look for a running Simulate** (P29.6). Grabbed by a plain LMB
    /// while a session is live, released by `Escape` or by the session ending.
    /// It steers the GAME camera, not the editor one, so the frame loop's
    /// camera arms deliberately do nothing for it — the deltas leave over
    /// `ViewportEvent::SimLook` instead.
    SimLook,
}

/// A discrete camera action queued by `wnd_proc`, drained by the frame loop.
#[derive(Debug, Clone, Copy)]
enum Action {
    Focus,
    StoreBookmark(usize),
    RecallBookmark(usize),
    SetGizmo(GizmoMode),
}

/// Input accumulated by `wnd_proc` between frames. Thread-local is safe: the
/// wnd_proc always runs on the thread that created the window.
#[derive(Default)]
struct InputState {
    capture: Capture,
    mouse_dx: f32,
    mouse_dy: f32,
    wheel_steps: i32,
    restore_cursor: POINT,
    actions: Vec<Action>,
    chords: Vec<String>,
    /// Latest cursor position in viewport-client (physical) pixels.
    cursor: (i32, i32),
    cursor_moved: bool,
    /// Plain-LMB (no Alt) is held: select/gizmo, not orbit.
    left_down: bool,
    /// A plain-LMB press this frame: (x, y, ctrl-held) for select/gizmo-begin.
    left_press: Option<(i32, i32, bool)>,
    left_release: bool,
}

thread_local! {
    static INPUT: RefCell<InputState> = RefCell::new(InputState::default());
    /// Whether an in-editor Simulate session is live (P29.6). Read by
    /// `wnd_proc`, which runs on this same thread, and written by the frame
    /// loop when `Cmd::SetSimRunning` arrives — the two never race because
    /// there is only one thread.
    static SIM_RUNNING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn modifier(vk: windows::Win32::UI::Input::KeyboardAndMouse::VIRTUAL_KEY) -> bool {
    (unsafe { GetKeyState(vk.0 as i32) } as u16 & 0x8000) != 0
}

/// Start a capture of `kind`: grab the mouse, hide the cursor, remember where
/// to restore it. Win32 calls stay OUTSIDE any RefCell borrow (SetCapture
/// re-enters this wnd_proc synchronously via WM_CAPTURECHANGED).
fn begin_capture(hwnd: HWND, kind: Capture) {
    let start = INPUT.with(|s| {
        let mut s = s.borrow_mut();
        if s.capture != Capture::None {
            return false;
        }
        s.capture = kind;
        true
    });
    if !start {
        return;
    }
    let mut pt = POINT::default();
    unsafe {
        let _ = GetCursorPos(&mut pt);
    }
    INPUT.with(|s| s.borrow_mut().restore_cursor = pt);
    unsafe {
        SetCapture(hwnd);
        let _ = SetFocus(Some(hwnd));
        ShowCursor(false);
    }
}

/// End the capture if `kind` owns it, restoring the cursor. Returns true if it
/// released (i.e. `kind` was the active capture).
fn end_capture(kind: Capture) -> bool {
    let restore = INPUT.with(|s| {
        let mut s = s.borrow_mut();
        if s.capture == kind {
            s.capture = Capture::None;
            Some(s.restore_cursor)
        } else {
            None
        }
    });
    if let Some(pt) = restore {
        unsafe {
            let _ = ReleaseCapture();
            ShowCursor(true);
            let _ = SetCursorPos(pt.x, pt.y);
        }
        true
    } else {
        false
    }
}

/// Chord name for a virtual key, or `None` if we don't forward it.
fn vk_name(vk: u32) -> Option<String> {
    match vk {
        0x41..=0x5A => Some(((b'A' + (vk - 0x41) as u8) as char).to_string()), // A–Z
        0x30..=0x39 => Some(((b'0' + (vk - 0x30) as u8) as char).to_string()), // 0–9
        0x20 => Some("Space".into()),
        0x70..=0x7B => Some(format!("F{}", vk - 0x6F)), // F1–F12
        _ => None,
    }
}

/// Handle a key-down: bookmarks, focus, or a forwarded global-shortcut chord.
fn on_key_down(vk: u32) {
    let ctrl = modifier(VK_CONTROL);
    let alt = modifier(VK_MENU);
    let shift = modifier(VK_SHIFT);

    // Ctrl+digit stores a bookmark; plain digit recalls it.
    if let 0x31..=0x39 = vk {
        let slot = (vk - 0x30) as usize;
        let action = if ctrl {
            Action::StoreBookmark(slot)
        } else if !alt {
            Action::RecallBookmark(slot)
        } else {
            return;
        };
        INPUT.with(|s| s.borrow_mut().actions.push(action));
        return;
    }

    // Escape gives the mouse back from the P29.6 play capture. First, and
    // unconditionally: a player whose pointer is hidden inside a viewport needs
    // one key that always works.
    if vk == 0x1B {
        end_capture(Capture::SimLook);
        return;
    }

    // F focuses the selection (no modifiers).
    if vk == 0x46 && !ctrl && !alt && !shift {
        INPUT.with(|s| s.borrow_mut().actions.push(Action::Focus));
        return;
    }

    // W/E/R switch the transform-gizmo mode (UE parity), but only when not
    // flying — while RMB-captured those are WASD/QE flycam keys.
    if !ctrl && !alt && !shift {
        let mode = match vk {
            0x57 => Some(GizmoMode::Translate), // W
            0x45 => Some(GizmoMode::Rotate),    // E
            0x52 => Some(GizmoMode::Scale),     // R
            _ => None,
        };
        if let Some(mode) = mode {
            let idle = INPUT.with(|s| s.borrow().capture == Capture::None);
            if idle {
                INPUT.with(|s| s.borrow_mut().actions.push(Action::SetGizmo(mode)));
                return;
            }
        }
    }

    // Forward global-shortcut chords (Ctrl+… or F-keys) to the webview so
    // they keep working while the native viewport holds focus.
    let is_fkey = (0x70..=0x7B).contains(&vk);
    if !ctrl && !is_fkey {
        return;
    }
    let Some(name) = vk_name(vk) else { return };
    let mut parts: Vec<&str> = Vec::new();
    if ctrl {
        parts.push("Ctrl");
    }
    if alt {
        parts.push("Alt");
    }
    if shift {
        parts.push("Shift");
    }
    let mut chord = parts.join("+");
    if !chord.is_empty() {
        chord.push('+');
    }
    chord.push_str(&name);
    INPUT.with(|s| s.borrow_mut().chords.push(chord));
}

extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        // The swapchain covers every pixel — skipping the GDI background
        // erase kills the white flash during splitter resizes.
        WM_ERASEBKGND => LRESULT(1),

        // RMB: flycam, or dolly when Alt is held.
        WM_RBUTTONDOWN => {
            let kind = if modifier(VK_MENU) {
                Capture::Dolly
            } else {
                Capture::Fly
            };
            begin_capture(hwnd, kind);
            LRESULT(0)
        }
        WM_RBUTTONUP => {
            // RMB owns whichever of Fly/Dolly it started.
            end_capture(Capture::Fly);
            end_capture(Capture::Dolly);
            LRESULT(0)
        }

        // Alt+LMB orbits. Plain LMB selects / drags a gizmo handle.
        WM_LBUTTONDOWN => {
            if modifier(VK_MENU) {
                begin_capture(hwnd, Capture::Orbit);
            } else if SIM_RUNNING.with(|r| r.get()) {
                // **Play capture** (P29.6). A live Simulate turns a plain LMB in
                // the viewport into mouse-look for the GAME camera: the pointer
                // is hidden and captured exactly as a flycam gesture would, and
                // the deltas are forwarded to Ring 2 instead of steering the
                // editor camera. Escape gives it back.
                begin_capture(hwnd, Capture::SimLook);
            } else if INPUT.with(|s| {
                matches!(
                    s.borrow().capture,
                    Capture::Fly | Capture::Pan | Capture::Dolly
                )
            }) {
                // A navigation gesture (RMB fly, MMB pan, Alt+RMB dolly) already
                // owns the mouse. A plain LMB must NOT hijack it for pick/gizmo:
                // don't SetCapture, don't record a press — let the gesture
                // continue uninterrupted (L1).
            } else {
                let x = (lparam.0 & 0xffff) as i16 as i32;
                let y = ((lparam.0 >> 16) & 0xffff) as i16 as i32;
                let ctrl = modifier(VK_CONTROL);
                unsafe {
                    // Capture so we keep getting moves/up even off-window while
                    // dragging a gizmo. This is the plain-LMB path (not orbit).
                    SetCapture(hwnd);
                    let _ = SetFocus(Some(hwnd));
                }
                INPUT.with(|s| {
                    let mut s = s.borrow_mut();
                    s.left_down = true;
                    s.left_press = Some((x, y, ctrl));
                    s.cursor = (x, y);
                });
            }
            LRESULT(0)
        }
        WM_LBUTTONUP => {
            if end_capture(Capture::Orbit) {
                // was an orbit gesture
            } else if end_capture(Capture::SimLook) {
                // was the P29.6 play capture
            } else {
                let owns = INPUT.with(|s| {
                    let mut s = s.borrow_mut();
                    if s.left_down {
                        s.left_down = false;
                        s.left_release = true;
                        true
                    } else {
                        false
                    }
                });
                if owns {
                    unsafe {
                        let _ = ReleaseCapture();
                    }
                }
            }
            LRESULT(0)
        }

        WM_MOUSEMOVE => {
            let x = (lparam.0 & 0xffff) as i16 as i32;
            let y = ((lparam.0 >> 16) & 0xffff) as i16 as i32;
            INPUT.with(|s| {
                let mut s = s.borrow_mut();
                s.cursor = (x, y);
                s.cursor_moved = true;
            });
            LRESULT(0)
        }

        // MMB pans (with or without Alt).
        WM_MBUTTONDOWN => {
            begin_capture(hwnd, Capture::Pan);
            LRESULT(0)
        }
        WM_MBUTTONUP => {
            end_capture(Capture::Pan);
            LRESULT(0)
        }

        // Capture can be stolen (alt-tab, other SetCapture); un-hide cleanly
        // and drop any in-flight gizmo/selection drag.
        WM_CAPTURECHANGED => {
            // Capture only carries a hidden/moved cursor for the Fly/Orbit/Pan/
            // Dolly gestures (begin_capture stored the restore point); the plain-
            // LMB path leaves `capture == None`, so `was` gates the cursor
            // restore to exactly the gestures that need it.
            let restore = INPUT.with(|s| {
                let mut s = s.borrow_mut();
                if s.left_down {
                    s.left_down = false;
                    s.left_release = true;
                }
                let was = std::mem::replace(&mut s.capture, Capture::None) != Capture::None;
                was.then_some(s.restore_cursor)
            });
            if let Some(pt) = restore {
                // Mirror end_capture: on a stolen capture, restore cursor
                // visibility AND position to where the gesture grabbed it (L2).
                unsafe {
                    ShowCursor(true);
                    let _ = SetCursorPos(pt.x, pt.y);
                }
            }
            LRESULT(0)
        }

        WM_MOUSEWHEEL => {
            let delta = ((wparam.0 >> 16) & 0xffff) as u16 as i16 as i32 / 120;
            INPUT.with(|s| s.borrow_mut().wheel_steps += delta);
            LRESULT(0)
        }

        WM_KEYDOWN | WM_SYSKEYDOWN => {
            on_key_down(wparam.0 as u32);
            // SYSKEYDOWN must reach DefWindowProc or Alt-handling breaks.
            if msg == WM_SYSKEYDOWN {
                unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
            } else {
                LRESULT(0)
            }
        }

        WM_INPUT => {
            let mut raw = RAWINPUT::default();
            let mut size = std::mem::size_of::<RAWINPUT>() as u32;
            let read = unsafe {
                GetRawInputData(
                    HRAWINPUT(lparam.0 as *mut _),
                    RID_INPUT,
                    Some(&mut raw as *mut _ as *mut _),
                    &mut size,
                    std::mem::size_of::<RAWINPUTHEADER>() as u32,
                )
            };
            if read != u32::MAX && raw.header.dwType == RIM_TYPEMOUSE.0 {
                let mouse = unsafe { raw.data.mouse };
                // Bit 0 = MOUSE_MOVE_ABSOLUTE; we only want relative deltas.
                if mouse.usFlags.0 & 0x01 == 0 {
                    INPUT.with(|s| {
                        let mut s = s.borrow_mut();
                        if s.capture != Capture::None {
                            s.mouse_dx += mouse.lLastX as f32;
                            s.mouse_dy += mouse.lLastY as f32;
                        }
                    });
                }
            }
            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }

        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn create_child_window(parent_hwnd: isize) -> windows::core::Result<HWND> {
    unsafe {
        let hinstance = GetModuleHandleW(None)?;
        let class_name = w!("InfinityViewportClass");

        let wc = WNDCLASSW {
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinstance.into(),
            lpszClassName: class_name,
            ..Default::default()
        };
        // Returns 0 if the class already exists (second viewport); that's fine.
        RegisterClassW(&wc);

        CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            class_name,
            w!("Infinity Viewport"),
            WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS,
            0,
            0,
            64,
            64,
            Some(HWND(parent_hwnd as *mut _)),
            None,
            Some(hinstance.into()),
            None,
        )
    }
}

/// Reparent a foreign (PIE player) window into our parent window tree — the
/// proven Spike D cross-process sequence: switch it to `WS_CHILD`, `SetParent`
/// onto the editor's parent window, then position it at the hole rect. The
/// parent window is owned by the Tauri main thread (which always pumps), so the
/// child's later `DestroyWindow` `WM_PARENTNOTIFY` never deadlocks us (the
/// Spike D pump-while-teardown finding).
fn embed_foreign_window(parent_hwnd: isize, foreign: isize, rect: Option<ViewportRect>) {
    let parent = HWND(parent_hwnd as *mut _);
    let child = HWND(foreign as *mut _);
    unsafe {
        // MSDN order: WS_CHILD before SetParent (a top-level window must lose its
        // overlapped styles first).
        let style = GetWindowLongPtrW(child, GWL_STYLE);
        let child_style = (style & !(WS_OVERLAPPEDWINDOW.0 as isize)) | (WS_CHILD.0 as isize);
        SetWindowLongPtrW(child, GWL_STYLE, child_style);
        if let Err(e) = SetParent(child, Some(parent)) {
            tracing::warn!("inf-viewport: SetParent for embedded PIE failed: {e}");
            return;
        }
        if let Some(r) = rect {
            let _ = SetWindowPos(
                child,
                Some(HWND_TOP),
                r.x,
                r.y,
                r.width.max(1) as i32,
                r.height.max(1) as i32,
                SWP_NOACTIVATE,
            );
        }
        let _ = ShowWindow(child, SW_SHOWNA);
    }
}

/// Our child window's rectangle expressed in `parent`-client (physical) pixels —
/// the same coordinate space `Cmd::SetRect` uses. Used as the initial position
/// for an embedded PIE window when it is adopted before any `SetRect` has
/// arrived (L3). `GetWindowRect` yields screen coordinates, so the top-left is
/// mapped back into the parent's client area.
fn child_rect_in_parent(parent: HWND, child: HWND) -> Option<ViewportRect> {
    unsafe {
        let mut r = RECT::default();
        GetWindowRect(child, &mut r).ok()?;
        let mut tl = POINT {
            x: r.left,
            y: r.top,
        };
        let _ = ScreenToClient(parent, &mut tl);
        Some(ViewportRect {
            x: tl.x,
            y: tl.y,
            width: (r.right - r.left).max(1) as u32,
            height: (r.bottom - r.top).max(1) as u32,
        })
    }
}

fn register_raw_mouse(hwnd: HWND) {
    // Usage page 0x01 (generic desktop), usage 0x02 (mouse); deltas are
    // delivered as WM_INPUT while our window has keyboard focus.
    let rid = RAWINPUTDEVICE {
        usUsagePage: 0x01,
        usUsage: 0x02,
        dwFlags: RAWINPUTDEVICE_FLAGS(0),
        hwndTarget: hwnd,
    };
    if let Err(e) =
        unsafe { RegisterRawInputDevices(&[rid], std::mem::size_of::<RAWINPUTDEVICE>() as u32) }
    {
        tracing::warn!("inf-viewport: raw input registration failed: {e}");
    }
}

fn key_down(vk: i32) -> bool {
    (unsafe { GetAsyncKeyState(vk) } as u16 & 0x8000) != 0
}

/// Drained input for one frame.
struct FrameInput {
    capture: Capture,
    dx: f32,
    dy: f32,
    wheel: i32,
    actions: Vec<Action>,
    chords: Vec<String>,
    cursor: (i32, i32),
    cursor_moved: bool,
    left_down: bool,
    left_press: Option<(i32, i32, bool)>,
    left_release: bool,
}

fn drain_input() -> FrameInput {
    INPUT.with(|s| {
        let mut s = s.borrow_mut();
        FrameInput {
            capture: s.capture,
            dx: std::mem::take(&mut s.mouse_dx),
            dy: std::mem::take(&mut s.mouse_dy),
            wheel: std::mem::take(&mut s.wheel_steps),
            actions: std::mem::take(&mut s.actions),
            chords: std::mem::take(&mut s.chords),
            cursor: s.cursor,
            cursor_moved: std::mem::take(&mut s.cursor_moved),
            left_down: s.left_down,
            left_press: std::mem::take(&mut s.left_press),
            left_release: std::mem::take(&mut s.left_release),
        }
    })
}

/// **Settle every in-flight gesture before the render loop stops** (P21.3
/// audit).
///
/// A caught panic ends the loop but not the process: the editor, the webview and
/// the document all survive. What does not survive is the *undoability* of any
/// gesture that was mid-drag — its dabs are already in the document with no
/// `EditCommand` describing them, and the gizmo drag's open transaction would
/// additionally swallow every later edit (see
/// `EngineHost::settle_orphaned_transaction`). One lock, one call, on the way
/// out.
///
/// Failing to take the lock is not worth a second failure path here: this runs
/// while unwinding from a panic that may itself have poisoned it, and the loop
/// is stopping either way.
fn settle_before_exit(host: &mut EngineHost, scene: &SharedScene) {
    if let Ok(mut doc) = scene.lock() {
        if host.settle_all_gestures(&mut doc) {
            tracing::warn!(
                "inf-viewport: the render loop stopped mid-gesture — the edit so far was \
                 committed as one undo step so Ctrl+Z can still reach it"
            );
        }
    }
}

/// Write a viewport crash report for a caught panic and log where it landed.
/// Shared by the init and per-frame guards; the editor process survives either.
fn report_viewport_panic(location: &str, payload: &(dyn std::any::Any + Send)) {
    let msg = inf_editor_core::diagnostics::panic_message(payload);
    let report = inf_editor_core::diagnostics::CrashReport {
        app: "inf-viewport".into(),
        engine_version: env!("CARGO_PKG_VERSION").into(),
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        location: location.into(),
        message: msg.clone(),
        adapter: None,
        log_tail: Vec::new(),
    };
    let dir = std::env::temp_dir().join("InfinityEngine").join("crashes");
    match inf_editor_core::diagnostics::write_crash_report(&dir, &report) {
        Ok(path) => tracing::error!(
            "inf-viewport: {location} panicked ({msg}); crash report at {} — \
             viewport stopped, editor still running",
            path.display()
        ),
        Err(werr) => tracing::error!(
            "inf-viewport: {location} panicked ({msg}); failed to write crash report: {werr}"
        ),
    }
}

fn thread_main(
    parent_hwnd: isize,
    rx: Receiver<Cmd>,
    sink: ViewportEventSink,
    scene: SharedScene,
    volumes: inf_editor_core::voxel_store::SharedVoxelVolumes,
    fractures: inf_editor_core::simulate::SharedFractures,
) {
    let hwnd = match create_child_window(parent_hwnd) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!("inf-viewport: failed to create child window: {e}");
            return;
        }
    };

    // Keep the viewport above the WebView2 sibling so it owns its rectangle.
    unsafe {
        let _ = SetWindowPos(hwnd, Some(HWND_TOP), 0, 0, 64, 64, SWP_NOACTIVATE);
    }
    register_raw_mouse(hwnd);

    let hinstance = unsafe { GetModuleHandleW(None) }
        .map(|h| h.0 as isize)
        .unwrap_or_default();
    let target = SurfaceTarget::Win32 {
        hwnd: hwnd.0 as isize,
        hinstance,
    };
    // Engine init compiles every pass shader up-front; guard it the same way as
    // the per-frame render so a validation panic degrades to "viewport
    // unavailable" instead of silently killing this thread (the pick-shader
    // composition bug did exactly that).
    let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        EngineHost::new(target, 64, 64)
            .map(|h| h.with_voxel_volumes(volumes).with_fractures(fractures))
    }));
    let mut host = match init {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => {
            tracing::error!("inf-viewport: engine init failed: {e}");
            return;
        }
        Err(payload) => {
            report_viewport_panic("<viewport engine init>", &*payload);
            return;
        }
    };
    tracing::info!("inf-viewport: child window + engine renderer up");

    let mut camera = EditorCamera::default();
    // A separate 2D ortho camera; switching modes preserves both poses (P8.2c).
    let mut camera_2d = Camera2D::default();
    let mut bookmarks = Bookmarks::default();
    // Active smooth-focus goal, cleared once the camera settles.
    let mut focus_goal = None;
    let mut last_frame = Instant::now();

    // Embedded PIE (P9.4): the adopted foreign player window + the latest hole
    // rect, so rect changes follow the embedded window while our own child hides.
    let mut embedded: Option<HWND> = None;
    let mut last_rect: Option<ViewportRect> = None;
    // Whether our own child should be presenting. Hidden while an HTML overlay is
    // up (Cmd::SetVisible(false)) or while a PIE window is embedded; when it isn't
    // presenting, the loop sleeps instead of busy-spinning on a dead surface (M3).
    let mut visible = true;

    // Last published terrain tool-state, so the status event only fires on a
    // change (see the drain below).
    let mut last_terrain_state = (false, false, false);

    'outer: loop {
        // Recover from an embedded PIE window that vanished without a
        // ReleaseForeign (the player crashed/exited): our child is hidden behind
        // a now-dead foreign HWND. Drop the embed and re-show our child so the
        // viewport keeps working (L3).
        if let Some(foreign) = embedded {
            if !unsafe { IsWindow(Some(foreign)) }.as_bool() {
                embedded = None;
                if visible {
                    unsafe {
                        let _ = ShowWindow(hwnd, SW_SHOWNA);
                    }
                }
                tracing::info!(
                    "inf-viewport: embedded PIE window vanished — restored viewport child"
                );
            }
        }

        // 1. Apply pending commands (coalesce rect updates to the latest).
        let mut latest_rect: Option<ViewportRect> = None;
        // Drag-drops handled this frame (Wave 2, feature A): buffered here and
        // resolved in the interaction block, where the scene is locked and the
        // active render view (for the pick ray) is built.
        let mut pending_drops: Vec<(f32, f32, String)> = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(Cmd::SetRect(r)) => latest_rect = Some(r),
                Ok(Cmd::SetVisible(v)) => {
                    visible = v;
                    // While a PIE window is embedded, keep our own child hidden.
                    if embedded.is_none() {
                        unsafe {
                            // SW_SHOWNA: show without stealing focus from the webview.
                            let _ = ShowWindow(hwnd, if v { SW_SHOWNA } else { SW_HIDE });
                        }
                    }
                }
                Ok(Cmd::Drop { x, y, payload }) => {
                    // Buffer for the interaction block (needs the scene lock + the
                    // active view to raycast the world point under the cursor).
                    pending_drops.push((x, y, payload));
                }
                Ok(Cmd::SetMode(m)) => host.set_mode(m),
                Ok(Cmd::SetSnap2D(s)) => host.set_snap_2d(s),
                Ok(Cmd::SetToolMode(m)) => host.set_tool_mode(m),
                Ok(Cmd::SetSculpt(s)) => host.set_sculpt(s),
                Ok(Cmd::SetFoliage(f)) => host.set_foliage(f),
                Ok(Cmd::SetBiome(b)) => host.set_biome(b),
                Ok(Cmd::SetWater(w)) => host.set_water(w),
                Ok(Cmd::SetVoxel(v)) => host.set_voxel(v),
                Ok(Cmd::SetSimRunning(running)) => {
                    SIM_RUNNING.with(|r| r.set(running));
                    // Ending a session must give the mouse back, or the cursor
                    // stays hidden over a viewport nobody is playing in.
                    if !running {
                        end_capture(Capture::SimLook);
                    }
                }
                Ok(Cmd::SetBiomePalette(e, p)) => host.set_biome_palette(e, p),
                Ok(Cmd::SetWaterHints(e, h)) => host.set_water_hints(e, h),
                Ok(Cmd::SetGizmo(m)) => {
                    host.set_gizmo_mode(m);
                    // Echo so the toolbar reflects an IPC-driven change too.
                    sink(ViewportEvent::GizmoModeChanged(from_gizmo_mode(m)));
                }
                Ok(Cmd::SetGizmoSpace(s)) => host.set_gizmo_space(s),
                Ok(Cmd::SetSnap3D(s)) => host.set_snap_3d(s),
                Ok(Cmd::SetViewMode(m)) => host.set_view_mode(m),
                Ok(Cmd::SetContentRoot(root)) => host.set_content_root(root),
                Ok(Cmd::RefreshAssetIndex) => host.refresh_asset_index(),
                Ok(Cmd::ReloadTerrainStores) => host.reload_terrain_stores(),
                Ok(Cmd::ReloadVoxelStores) => host.reload_voxel_stores(),
                Ok(Cmd::ClearStreams) => host.clear_streams(),
                Ok(Cmd::EmbedForeign(foreign)) => {
                    // Position the foreign window at the hole immediately. If no
                    // SetRect has arrived yet, fall back to our child's current
                    // rect so the player isn't left at 0,0 until the first
                    // SetRect (L3). Subsequent SetRects follow it (target =
                    // embedded), so this only bridges the initial gap.
                    let rect = last_rect
                        .or_else(|| child_rect_in_parent(HWND(parent_hwnd as *mut _), hwnd));
                    if last_rect.is_none() {
                        last_rect = rect;
                    }
                    embed_foreign_window(parent_hwnd, foreign, rect);
                    unsafe {
                        let _ = ShowWindow(hwnd, SW_HIDE);
                    }
                    embedded = Some(HWND(foreign as *mut _));
                }
                Ok(Cmd::ReleaseForeign) => {
                    embedded = None;
                    unsafe {
                        let _ = ShowWindow(hwnd, SW_SHOWNA);
                    }
                }
                Ok(Cmd::Destroy) | Err(TryRecvError::Disconnected) => break 'outer,
                Err(TryRecvError::Empty) => break,
            }
        }
        if let Some(r) = latest_rect {
            last_rect = Some(r);
            // Move the embedded PIE window (if any) instead of our hidden child;
            // always keep our own child sized so a release restores cleanly.
            let target = embedded.unwrap_or(hwnd);
            unsafe {
                let _ = SetWindowPos(
                    target,
                    None,
                    r.x,
                    r.y,
                    r.width.max(1) as i32,
                    r.height.max(1) as i32,
                    SWP_NOACTIVATE | SWP_NOZORDER,
                );
                if embedded.is_some() {
                    let _ = SetWindowPos(
                        hwnd,
                        None,
                        r.x,
                        r.y,
                        r.width.max(1) as i32,
                        r.height.max(1) as i32,
                        SWP_NOACTIVATE | SWP_NOZORDER,
                    );
                }
            }
            host.resize(r.width.max(1), r.height.max(1));
        }

        // 2. Pump this thread's window messages.
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, Some(hwnd), 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        // 3. Camera update from accumulated input.
        let dt = last_frame.elapsed().as_secs_f32().min(0.1);
        last_frame = Instant::now();
        let input = drain_input();

        // Forward global-shortcut chords to the webview (focus handoff).
        for chord in input.chords {
            sink(ViewportEvent::Key(KeyChord { chord }));
        }

        // Active projection + surface size for this frame's camera/gizmo math.
        let two_d = host.mode == ViewportMode::TwoD;
        let (vw, vh) = host.surface_size();
        let (vwf, vhf) = (vw as f64, vh.max(1) as f64);

        // Discrete actions: gizmo mode, focus, bookmarks. Focus/recall
        // interrupt an in-flight focus animation.
        for action in input.actions {
            match action {
                Action::SetGizmo(mode) => {
                    host.set_gizmo_mode(mode);
                    // A W/E/R keypress over the viewport must update the toolbar
                    // (two-way sync, Wave 2).
                    sink(ViewportEvent::GizmoModeChanged(from_gizmo_mode(mode)));
                }
                Action::Focus => {
                    // Frame the selection, or the world origin if none.
                    let (center, radius) =
                        host.selection_focus().unwrap_or((glam::DVec3::ZERO, 4.0));
                    if two_d {
                        // Frame the selection's XY bounds instantly.
                        camera_2d.frame(
                            DVec2::new(center.x, center.y),
                            DVec2::splat(radius),
                            vwf / vhf,
                        );
                    } else {
                        focus_goal = Some(camera.focus_goal(center, radius));
                    }
                }
                Action::StoreBookmark(n) => {
                    bookmarks.store(n, camera.pose());
                    tracing::debug!("inf-viewport: stored camera bookmark {n}");
                }
                Action::RecallBookmark(n) => {
                    if let Some(pose) = bookmarks.recall(n) {
                        camera.set_pose(pose);
                        focus_goal = None;
                    }
                }
            }
        }

        // Project the shared world and run pointer interaction against it. The
        // document is the single source of truth: a pick updates its selection,
        // a gizmo drag writes transforms back as one undo entry, and both
        // signal Ring 2 (WorldChanged) so the Outliner/Details re-sync. The
        // lock is held only for this short section, never across the render.
        // Guard the pointer-interaction block (pick / gizmo / sculpt) with the
        // same crash-safety net as the render (H1). Picking drives the GPU
        // ID-buffer Picker, so a device-lost TDR between frames could surface a
        // validation panic HERE — outside the render's catch_unwind — which would
        // otherwise kill this thread and leave the scene mutex poisoned. Catch it,
        // write a crash report, and exit the loop with the same semantics as a
        // render panic. (render_frame rebuilds the picker on the fresh device, so
        // the normal recovery path stays lossless; this only backstops a panic.)
        let interaction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut world_changed = false;
            if let Ok(mut doc) = scene.lock() {
                host.sync_from_doc(&doc);

                // The render view for the ACTIVE mode drives picking + gizmo rays.
                let interact_view = if two_d {
                    host.view_2d(&camera_2d)
                } else {
                    host.view_for(&camera)
                };

                // Drag-drops (Wave 2, feature A): spawn at the world point under
                // the cursor as one undo step, regardless of the active tool.
                for (dx, dy, payload) in pending_drops.drain(..) {
                    let (px, py) = (dx.max(0.0) as u32, dy.max(0.0) as u32);
                    if host.spawn_drop(&mut doc, &interact_view, px, py, &payload) {
                        world_changed = true;
                    }
                }

                // Sculpt mode (perspective only): plain LMB paints a terrain height
                // stroke instead of selecting/gizmo-dragging. The stroke lives on the
                // viewport thread (like a gizmo drag) and commits one undo step at
                // mouse-up (P10.2b). 2D mode keeps Select regardless of the tool.
                // P19.2: the biome brush rides the SAME stroke machinery — it
                // needs the identical terrain pick, footprint paging, streamed
                // read-only gate and one-command-per-stroke commit, and only the
                // dab differs. `host.begin_sculpt` branches on the tool mode.
                let sculpting =
                    matches!(host.tool_mode(), ToolMode::Sculpt | ToolMode::Biome) && !two_d;
                let foliage = host.tool_mode() == ToolMode::Foliage && !two_d;
                // P20.4: the water tool is neither a brush nor a selection. A
                // river click appends a control point (press only); a lake drags
                // a rectangle (press → move → release). It rides in front of the
                // select/gizmo branch for the same reason the brushes do — plain
                // LMB means something else here.
                let water = host.tool_mode() == ToolMode::Water && !two_d;
                // P21.2: the voxel carve tools. The brush is the sculpt gesture
                // (press / drag / release, one undo step at mouse-up); the spline
                // tunnel is the river tool's (a click appends a waypoint,
                // Ctrl+click closes the path and carves the tube). Both ride in
                // front of the select/gizmo branch for the same reason the other
                // terrain tools do — plain LMB means something else here.
                let voxel = host.tool_mode() == ToolMode::Voxel && !two_d;
                // P21.2 audit: the branches below are gated on the ACTIVE tool, so
                // a carve stroke that was still down when the tool changed would
                // never reach its `finish_voxel` — its dabs stay in the world,
                // save like any other edit, and Ctrl+Z cannot reach them. Close it
                // here, before the branches, where the document is in hand.
                // BEFORE any settler: if the document was replaced under us
                // (File ▸ Open / New), every in-flight gesture refers to a level
                // that no longer exists — settling one would commit the OLD
                // level's edit into the new document (P21.3 audit).
                if host.abandon_gestures_on_document_swap(&doc) {
                    world_changed = true;
                }
                if host.settle_orphaned_carve(&mut doc) {
                    world_changed = true;
                }
                // P21.3 (the P21.2 audit's N2 item): the same hole, one tool
                // over. A height / splat / biome stroke still down when the tool
                // changed would never reach its `finish_sculpt` either — its dabs
                // are already in the document and Ctrl+Z could not reach them.
                if host.settle_orphaned_sculpt(&mut doc) {
                    world_changed = true;
                }
                // …and the foliage stroke, which strands an undo entry AND the
                // transaction it opened.
                if host.settle_orphaned_foliage(&mut doc) {
                    world_changed = true;
                }
                // The backstop for the one gesture with no stroke object of its
                // own: a gizmo drag opens "Move" here and commits it on release,
                // both inside the tool-gated select branch below. One unmatched
                // begin kills Ctrl+Z for the whole session (P21.3 audit ruling).
                if host.settle_orphaned_transaction(&mut doc) {
                    world_changed = true;
                }
                if sculpting {
                    if let Some((x, y, ctrl)) = input.left_press {
                        let (px, py) = (x.max(0) as u32, y.max(0) as u32);
                        host.begin_sculpt(&mut doc, &interact_view, px, py, ctrl);
                    }
                    if input.left_down && host.is_sculpting() {
                        let (x, y) = input.cursor;
                        host.update_sculpt(
                            &mut doc,
                            &interact_view,
                            x.max(0) as u32,
                            y.max(0) as u32,
                        );
                    }
                    if input.left_release && host.is_sculpting() && host.finish_sculpt(&mut doc) {
                        world_changed = true;
                    }
                    // Idle hover: the brush ring follows the cursor over terrain.
                    if !input.left_down && input.capture == Capture::None && input.cursor_moved {
                        let (x, y) = input.cursor;
                        if x >= 0 && y >= 0 {
                            host.update_sculpt_hover(&doc, &interact_view, x as u32, y as u32);
                        }
                    }
                } else if voxel {
                    if let Some((x, y, ctrl)) = input.left_press {
                        let (px, py) = (x.max(0) as u32, y.max(0) as u32);
                        // Ctrl is the tunnel's COMMIT modifier here, not the
                        // sculpt brush's invert: a tunnel needs a "that was the
                        // last waypoint" gesture, and the alternatives available
                        // on this seam (a double-click, a second button) are
                        // either not delivered or already spoken for by the
                        // camera. The hover readout says so, every frame.
                        if host.begin_voxel(&mut doc, &interact_view, px, py, ctrl) {
                            world_changed = true;
                        }
                    }
                    if input.left_down && host.is_carving() {
                        let (x, y) = input.cursor;
                        host.update_voxel(
                            &mut doc,
                            &interact_view,
                            x.max(0) as u32,
                            y.max(0) as u32,
                        );
                    }
                    if input.left_release && host.is_carving() && host.finish_voxel(&mut doc) {
                        world_changed = true;
                    }
                    // Idle hover: the cut silhouette follows the cursor, and a
                    // pending tunnel shows the segment the next click would add.
                    if !input.left_down && input.capture == Capture::None && input.cursor_moved {
                        let (x, y) = input.cursor;
                        if x >= 0 && y >= 0 {
                            host.update_voxel_hover(&doc, &interact_view, x as u32, y as u32);
                        }
                    }
                } else if water {
                    if let Some((x, y, _ctrl)) = input.left_press {
                        let (px, py) = (x.max(0) as u32, y.max(0) as u32);
                        if host.begin_water(&mut doc, &interact_view, px, py) {
                            world_changed = true;
                        }
                    }
                    if input.left_down {
                        let (x, y) = input.cursor;
                        host.update_water(&doc, &interact_view, x.max(0) as u32, y.max(0) as u32);
                    }
                    if input.left_release {
                        let (x, y) = input.cursor;
                        if host.finish_water(
                            &mut doc,
                            &interact_view,
                            x.max(0) as u32,
                            y.max(0) as u32,
                        ) {
                            world_changed = true;
                        }
                    }
                    // Idle hover: a river shows the segment the next click adds.
                    if !input.left_down && input.capture == Capture::None && input.cursor_moved {
                        let (x, y) = input.cursor;
                        if x >= 0 && y >= 0 {
                            host.update_water_hover(&doc, &interact_view, x as u32, y as u32);
                        }
                    }
                } else if foliage {
                    // Foliage mode (perspective only): plain LMB scatters (or
                    // erases) instances onto the terrain under the brush, exactly
                    // mirroring the sculpt seam — the stroke lives on the viewport
                    // thread and commits ONE undo step at mouse-up (E-P6).
                    if let Some((x, y, _ctrl)) = input.left_press {
                        let (px, py) = (x.max(0) as u32, y.max(0) as u32);
                        host.begin_foliage(&mut doc, &interact_view, px, py);
                    }
                    if input.left_down && host.is_painting_foliage() {
                        let (x, y) = input.cursor;
                        host.update_foliage(
                            &mut doc,
                            &interact_view,
                            x.max(0) as u32,
                            y.max(0) as u32,
                        );
                    }
                    if input.left_release
                        && host.is_painting_foliage()
                        && host.finish_foliage(&mut doc)
                    {
                        world_changed = true;
                    }
                    // Idle hover: the brush ring follows the cursor.
                    if !input.left_down && input.capture == Capture::None && input.cursor_moved {
                        let (x, y) = input.cursor;
                        if x >= 0 && y >= 0 {
                            host.update_foliage_hover(&doc, &interact_view, x as u32, y as u32);
                        }
                    }
                } else {
                    // Plain LMB: a handle under the cursor begins a gizmo drag (one
                    // undo transaction), otherwise it selects the picked entity.
                    if let Some((x, y, ctrl)) = input.left_press {
                        let (px, py) = (x.max(0) as u32, y.max(0) as u32);
                        if host.try_begin_gizmo(&interact_view, px, py) {
                            doc.begin_transaction("Move");
                        } else {
                            match host.pick_guid(&interact_view, px, py) {
                                Some(guid) => {
                                    doc.select(&[guid], ctrl);
                                    world_changed = true;
                                }
                                None if !ctrl => {
                                    doc.clear_selection();
                                    world_changed = true;
                                }
                                None => {}
                            }
                            host.sync_from_doc(&doc); // reflect the new selection now
                        }
                    }

                    if input.left_down && host.is_dragging_gizmo() {
                        let (x, y) = input.cursor;
                        // 3D snap increments come from the toolbar (Wave 2): every
                        // drag snaps when `always_on`, else Shift-gated (preserving
                        // the old feel). 2D translate still uses the 2D grid/pixel
                        // snap; 2D rotate/scale reuse the 3D increments.
                        let cfg = host.snap_3d();
                        let snap_on = cfg.always_on || key_down(0x10); // Shift
                        let snap = if two_d {
                            match host.gizmo_mode {
                                GizmoMode::Translate => {
                                    let s = host.snap_2d_translate();
                                    if s > 0.0 {
                                        s
                                    } else if snap_on {
                                        cfg.translate.max(0.0)
                                    } else {
                                        0.0
                                    }
                                }
                                GizmoMode::Rotate if snap_on => {
                                    cfg.rotate_deg.max(0.0).to_radians()
                                }
                                GizmoMode::Scale if snap_on => cfg.scale.max(0.0),
                                _ => 0.0,
                            }
                        } else if snap_on {
                            match host.gizmo_mode {
                                GizmoMode::Translate => cfg.translate.max(0.0),
                                GizmoMode::Rotate => cfg.rotate_deg.max(0.0).to_radians(),
                                GizmoMode::Scale => cfg.scale.max(0.0),
                            }
                        } else {
                            0.0
                        };
                        host.update_gizmo(&interact_view, x.max(0) as u32, y.max(0) as u32, snap);
                    }

                    if input.left_release {
                        if host.is_dragging_gizmo() {
                            // Write the gizmo's WORLD transforms back as parent-
                            // relative locals (Wave 2 nested-transform fix). Sort
                            // parents-first so a child composes against its
                            // parent's already-written pose (edit_set_world_transform
                            // propagates between writes).
                            let mut writes = host.selected_world_transforms();
                            writes.sort_by_key(|(guid, _)| hierarchy_depth(&doc, *guid));
                            for (guid, transform) in writes {
                                doc.edit_set_world_transform(guid, transform);
                            }
                            doc.commit_transaction();
                            world_changed = true;
                        }
                        host.end_gizmo();
                    }

                    // Hover highlight when idle (not dragging, not flying).
                    if input.cursor_moved && !input.left_down && input.capture == Capture::None {
                        let (x, y) = input.cursor;
                        if x >= 0 && y >= 0 {
                            host.set_hover(&interact_view, x as u32, y as u32);
                        }
                    }
                } // end Select-tool interaction (else of `if sculpting`)
            }
            world_changed
        }));
        let world_changed = match interaction {
            Ok(wc) => wc,
            Err(payload) => {
                report_viewport_panic("<viewport interaction>", &*payload);
                settle_before_exit(&mut host, &scene);
                break 'outer;
            }
        };
        if world_changed {
            sink(ViewportEvent::WorldChanged);
        }

        // Drain the tool-status seam (P16.4a). A rejection is one-shot; the
        // streamed flag is a standing fact, so it is published only on change —
        // an event per frame would flood the webview for no information.
        let status = host.take_tool_status();
        let terrain_state = (
            host.terrain_is_streamed(),
            host.terrain_is_editable(),
            host.terrain_has_unsaved_edits(),
        );
        if status.is_some() || terrain_state != last_terrain_state {
            last_terrain_state = terrain_state;
            sink(ViewportEvent::ToolStatus(ViewportToolStatusDto {
                message: status,
                terrain_streamed: terrain_state.0,
                terrain_editable: terrain_state.1,
                terrain_unsaved_edits: terrain_state.2,
                // EMPTY on purpose (P23.2a): the viewport thread does not know
                // its own key — Ring 2 owns the id→handle map and stamps this
                // on the way out (`commands::viewport::stamp_tool_status`).
                viewport: String::new(),
            }));
        }

        // A live navigation gesture cancels a focus animation.
        let navigating = input.capture != Capture::None || input.wheel != 0;
        if navigating {
            focus_goal = None;
        }

        if two_d {
            // 2D navigation: any drag capture pans in the plane; the wheel
            // zooms to the cursor (exponential half-height with clamps).
            match input.capture {
                Capture::None => {
                    if input.wheel != 0 {
                        let (cx, cy) = input.cursor;
                        camera_2d.zoom_at(input.wheel, cx as f64, cy as f64, vwf, vhf);
                    }
                }
                _ => camera_2d.pan(input.dx as f64, input.dy as f64, vwf, vhf),
            }
        } else {
            match input.capture {
                Capture::Fly => {
                    let fly = FlyInput {
                        mouse_dx: input.dx,
                        mouse_dy: input.dy,
                        wheel_steps: input.wheel,
                        forward: key_down(0x57), // W
                        back: key_down(0x53),    // S
                        right: key_down(0x44),   // D
                        left: key_down(0x41),    // A
                        up: key_down(0x45),      // E
                        down: key_down(0x51),    // Q
                        boost: key_down(0x10),   // Shift
                    };
                    camera.apply_fly(&fly, dt);
                }
                Capture::Orbit | Capture::Pan | Capture::Dolly => {
                    let mode = match input.capture {
                        Capture::Orbit => NavMode::Orbit,
                        Capture::Pan => NavMode::Pan,
                        _ => NavMode::Dolly,
                    };
                    let pivot = camera.pivot(None);
                    camera.apply_navigate(
                        &NavInput {
                            mode,
                            mouse_dx: input.dx,
                            mouse_dy: input.dy,
                            wheel_steps: input.wheel,
                        },
                        pivot,
                        dt,
                    );
                }
                Capture::SimLook => {
                    // **The play capture** (P29.6): the deltas belong to the
                    // GAME camera, so the editor camera is left exactly where it
                    // is and the counts are forwarded to Ring 2. Sent only when
                    // there is motion, so an idle held button costs one branch.
                    if input.dx != 0.0 || input.dy != 0.0 {
                        sink(ViewportEvent::SimLook {
                            dx: input.dx,
                            dy: input.dy,
                        });
                    }
                }
                Capture::None => {
                    // No button held: the wheel dollies toward the look point.
                    if input.wheel != 0 {
                        let pivot = camera.pivot(None);
                        camera.dolly(input.wheel as f32 * 0.12, pivot);
                    }
                }
            }
        }

        // Advance an in-flight focus animation (perspective only).
        if let Some(goal) = focus_goal {
            if camera.advance_focus(&goal, dt) {
                focus_goal = None;
            }
        }

        // 4. Rebase the floating origin on the active eye, build the view for
        //    the current mode, and render. FIFO present blocks at vsync and
        //    paces the loop. render_frame recovers from device loss internally —
        //    an error here means even the rebuild failed (driver truly gone).
        //
        //    We only render when our own child is actually presenting: hidden
        //    (an HTML overlay is up) or embedded (a PIE window covers the hole)
        //    means the surface never presents, so there is no vsync to pace us.
        //    In that case — and when a visible present nonetheless acquired no
        //    image (minimized/occluded child) — sleep ~10 ms so the loop drains
        //    commands and stays responsive without pinning a CPU core (M3). The
        //    quantum is short enough that a SetVisible(true)/SetRect wakes within
        //    one sleep.
        let should_render = visible && embedded.is_none();
        let mut presented = false;
        if should_render {
            let eye = if two_d { camera_2d.eye() } else { camera.pos };
            host.origin.maybe_rebase(eye);
            let render_view = if two_d {
                host.view_2d(&camera_2d)
            } else {
                host.view_for(&camera)
            };
            // Guard the render against a panic in engine code (P15.2). Rather than
            // let it unwind across the OS thread boundary, we catch it, write a
            // crash report, log a graceful message, and exit the render loop — the
            // editor process (and its webview) survive instead of the whole app
            // dying.
            let render_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                host.render_frame(&render_view)
            }));
            match render_result {
                Ok(Ok(p)) => presented = p,
                Ok(Err(e)) => {
                    tracing::error!("inf-viewport: unrecoverable render failure: {e}");
                    settle_before_exit(&mut host, &scene);
                    break 'outer;
                }
                Err(payload) => {
                    report_viewport_panic("<viewport render thread>", &*payload);
                    settle_before_exit(&mut host, &scene);
                    break 'outer;
                }
            }
        }
        // Nothing presented this iteration (hidden, embedded, or acquire-None):
        // throttle so the loop doesn't busy-spin at 100% CPU.
        if !presented {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    tracing::info!("inf-viewport: shutting down");
}
