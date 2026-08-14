//! Viewport commands: attach the native engine viewport to the editor window
//! and keep its rectangle in sync with the React layout hole.
//!
//! Events/commands here are the Spike A bridge (docs/ROADMAP.md §5): the
//! frontend reports CSS rects × devicePixelRatio; we forward physical pixels
//! to the `inf-viewport` thread.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use inf_editor_core::ipc::{
    BiomeSettingsDto, FoliageSettingsDto, GizmoModeDto, GizmoSpaceDto, SculptFalloffDto,
    SculptOpDto, SculptSettingsDto, Snap2DDto, Snap3DDto, SpoilModeDto, ToolModeDto, ViewModeDto,
    ViewportDrop, ViewportGizmoDto, ViewportKey, ViewportModeDto, ViewportRect,
    ViewportToolStatusDto, VoxelOpModeDto, VoxelSettingsDto, VoxelStatusDto, VoxelToolKindDto,
    WaterSettingsDto, WaterToolKindDto,
};
use inf_viewport::camera::{BiomeSettings, WaterSettings, WaterToolKind};
use inf_viewport::{
    FoliageSettings, GizmoSpace, SculptFalloff, SculptOp, SculptSettings, Snap2DSettings,
    SnapSettings, SpoilMode, ToolMode, ViewportEvent, ViewportMode, VoxelOpMode, VoxelSettings,
    VoxelToolKind,
};
use tauri::{Emitter, Manager};

use super::scene::{emit_world_delta, SceneState};

/// The well-known id of the **scene** viewport: the one the shell's centre hole
/// hosts, the one every unqualified `viewport_*` command means, and the one PIE
/// embeds its player window into (P23.2a).
pub const PRIMARY_VIEWPORT: &str = "primary";

/// Which viewport(s) a Ring-2 push addresses (P23.2a).
///
/// Every call site names one **explicitly**. That is the point of the type: with
/// a single-slot `Option<ViewportHandle>` the question never came up, and the
/// day a second viewport exists every one of the fourteen cross-module pushes
/// would have silently meant "the one that happened to be there". A content-root
/// change or an asset-index refresh is a fact about the *project* and belongs to
/// every viewport ([`All`](Target::All)); embedding a PIE window or restoring
/// visibility after it is a fact about the *scene slot* and belongs to exactly
/// one ([`Primary`](Target::Primary)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// The scene viewport ([`PRIMARY_VIEWPORT`]).
    Primary,
    /// One named viewport. `Named(PRIMARY_VIEWPORT)` and [`Primary`](Target::Primary)
    /// resolve identically — the map has one key space.
    Named(String),
    /// Every attached viewport. An empty map is a legal broadcast to nobody.
    All,
}

impl Target {
    /// A command's optional `viewport` argument → a target. **Absent ⇒ Primary**,
    /// which is what keeps the wire compatible: today's TS wrappers pass nothing.
    fn from_arg(id: Option<String>) -> Self {
        match id {
            Some(id) => Target::Named(id),
            None => Target::Primary,
        }
    }
}

/// Resolve `target` against a keyed map — the ONE rule Primary/Named/All share,
/// pulled out so it can be tested without a GPU, a window or a handle.
fn resolve<'a, T>(map: &'a BTreeMap<String, T>, target: &Target) -> Vec<&'a T> {
    match target {
        Target::Primary => map.get(PRIMARY_VIEWPORT).into_iter().collect(),
        Target::Named(id) => map.get(id.as_str()).into_iter().collect(),
        Target::All => map.values().collect(),
    }
}

/// **The process's ONE carve working set and ONE Simulate fracture map**
/// (P23.2a — the hoist).
///
/// Both used to be created inside `inf_viewport::spawn` and held by the
/// `ViewportHandle`, which made them *per viewport*. That was invisible while
/// there could only be one, and wrong the moment there can be two: the save
/// path (`scene_save` → `stage_voxels`) and the Simulate publishes
/// (`publish_sim_fractures`, `overlay_sim_carves`) have no camera and no host,
/// and would have had to pick a viewport to ask — a question with no correct
/// answer, since a carve belongs to the *document*, not to the window it was
/// made through.
///
/// So Ring 2 owns them, creates them exactly once at boot, and hands clones to
/// every `spawn`. Callers reach them through this state and never through
/// [`ViewportState`], which is what makes "resolve without asking a viewport"
/// structural rather than a convention.
///
/// **No `Option`, and no outer mutex.** With no viewport attached the stores
/// exist and are EMPTY, which is exactly the answer the old `None` stood for
/// ("no carve tool ⇒ no carve edits ⇒ nothing to save") and is now reached
/// without a second "nothing here" case for every caller to interpret. It also
/// retires the P21.2 hazard the old accessor carried in its doc comment: there
/// is no outer handle left to poison, so a panicked viewport thread can no
/// longer make unsaved carves *look absent*. The inner stores' own poisoning is
/// still reported by each caller, unchanged.
#[derive(Clone)]
pub struct SharedStores {
    volumes: inf_editor_core::voxel_store::SharedVoxelVolumes,
    fractures: inf_editor_core::simulate::SharedFractures,
}

impl Default for SharedStores {
    fn default() -> Self {
        Self {
            volumes: inf_editor_core::voxel_store::shared_volumes(),
            fractures: inf_editor_core::simulate::shared_fractures(),
        }
    }
}

impl SharedStores {
    /// The shared voxel working set every viewport carves into (P21.2).
    ///
    /// The chunks cannot come through a viewport command channel: they live
    /// behind a mutex the host and the undo history already share, and a `Cmd`
    /// has no reply path (see `inf_viewport::host::EngineHost::with_voxel_volumes`).
    pub fn voxel_volumes(&self) -> inf_editor_core::voxel_store::SharedVoxelVolumes {
        self.volumes.clone()
    }

    /// The Simulate fracture states the viewports draw broken actors from
    /// (P22.3) — the `voxel_volumes` twin, published into every fixed step.
    pub fn fracture_states(&self) -> inf_editor_core::simulate::SharedFractures {
        self.fractures.clone()
    }
}

/// Every attached native viewport, keyed by id (P23.2a).
///
/// Was `Mutex<Option<ViewportHandle>>` — a single slot with an implicit
/// identity. The map makes the identity explicit and is the whole enabler: the
/// second `EngineHost` (P23.2b) is then a second `insert`, not a redesign.
#[derive(Default)]
pub struct ViewportState(Mutex<BTreeMap<String, inf_viewport::ViewportHandle>>);

impl ViewportState {
    /// Run `f` against every handle `target` resolves to.
    ///
    /// A poisoned map is logged rather than swallowed: it means a thread
    /// panicked while attaching or detaching a viewport, and from then on every
    /// push in the editor is silently dropped.
    fn with(&self, target: Target, f: impl Fn(&inf_viewport::ViewportHandle)) {
        match self.0.lock() {
            Ok(guard) => {
                for handle in resolve(&guard, &target) {
                    f(handle);
                }
            }
            Err(_) => tracing::error!(
                "inf-studio: the viewport map is poisoned — a thread panicked while holding \
                 it, so no viewport can be reached any more (rect, visibility, tool and \
                 streaming pushes are all being dropped). Save the level and restart the \
                 editor."
            ),
        }
    }

    /// Ids of every attached viewport, in key order. The diagnostic seam for the
    /// keyed map (and what a second-viewport UI will list).
    pub fn attached(&self) -> Vec<String> {
        match self.0.lock() {
            Ok(guard) => guard.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Show/hide native viewport children (used by PIE embedding to hide the
    /// editor viewport while the player window occupies the slot).
    pub fn set_visible(&self, target: Target, visible: bool) {
        self.with(target, |h| h.set_visible(visible));
    }

    /// Adopt a foreign (PIE player) window into the viewport slot (embedded PIE).
    pub fn embed_foreign(&self, target: Target, hwnd: i64) {
        self.with(target, |h| h.embed_foreign(hwnd as isize));
    }

    /// Point the terrain streamer at a project's content root (or `None` to
    /// disable it) — pushed from the `project://changed` flow so a
    /// `Terrain.asset` authored by the import wizard resolves to a loose
    /// `.inf_terrain` and starts paging (P16.4a, the B2 seam).
    pub fn set_content_root(&self, target: Target, root: Option<std::path::PathBuf>) {
        self.with(target, |h| h.set_content_root(root.clone()));
    }

    /// Rebuild the loose `.inf_terrain` index in place, keeping live streams —
    /// pushed when a terrain import finishes (P16.4a).
    pub fn refresh_asset_index(&self, target: Target) {
        self.with(target, |h| h.refresh_asset_index());
    }

    /// Reopen every live terrain stream's `.inf_terrain` in place — pushed by the
    /// save path once it has written sculpt/paint edits back into the assets
    /// (P16.4b). Live streams keep their resident pages and published cut, so a
    /// save never blinks the terrain the user is looking at.
    pub fn reload_terrain_stores(&self, target: Target) {
        self.with(target, |h| h.reload_terrain_stores());
    }

    /// Re-walk the loose `.inf_voxel` index in place — pushed by the save path
    /// once it has folded carve edits back into the assets (P21.2).
    /// The `reload_terrain_stores` twin, and the seam the P21.2 save flow needs:
    /// a carve written into a *new* asset only resolves after a re-walk, and the
    /// loaded volumes keep their chunks so saving never blinks a cave.
    pub fn reload_voxel_stores(&self, target: Target) {
        self.with(target, |h| h.reload_voxel_stores());
    }

    /// Release every terrain stream — pushed when the open document is replaced
    /// (File ▸ Open / File ▸ New, P16.4b). The streams are keyed on the previous
    /// document's entity GUIDs, so keeping them leaks a whole `.inf_terrain`
    /// payload plus any tile it pinned for an unsaved edit.
    pub fn clear_streams(&self, target: Target) {
        self.with(target, |h| h.clear_streams());
    }

    /// Push a terrain entity's biome **overlay palette** (linear RGBA, indexed by
    /// biome id). An EMPTY palette clears the entry, which is what an unbound
    /// terrain must send — otherwise the Biomes view mode would keep tinting
    /// with a vocabulary the terrain no longer names (P19.2).
    ///
    /// Public so the terrain/asset commands can re-push after a bind or a
    /// `.inf_biomes` save without reaching into the handle themselves.
    pub fn set_biome_palette(&self, target: Target, entity: uuid::Uuid, palette: Vec<[f32; 4]>) {
        self.with(target, |h| h.set_biome_palette(entity, palette.clone()));
    }

    /// Push a terrain's id-indexed water-level hints (P20.4) — the
    /// `set_biome_palette` twin, pushed from the same place.
    pub fn set_water_hints(&self, target: Target, entity: uuid::Uuid, hints: Vec<Option<f64>>) {
        self.with(target, |h| h.set_water_hints(entity, hints.clone()));
    }

    /// Release an embedded foreign window and restore the native viewport child.
    pub fn release_foreign(&self, target: Target) {
        self.with(target, |h| h.release_foreign());
    }
}

/// Create (once per id) the native child viewport inside the calling window.
///
/// `viewport` names the slot; absent means [`PRIMARY_VIEWPORT`] (the scene
/// viewport), so the shell's existing call — which passes nothing — is
/// unchanged. Idempotent per id: React StrictMode double-mounts.
#[tauri::command]
pub async fn viewport_attach(
    window: tauri::Window,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
    scene: tauri::State<'_, SceneState>,
    stores: tauri::State<'_, SharedStores>,
) -> Result<(), String> {
    let id = viewport.unwrap_or_else(|| PRIMARY_VIEWPORT.to_string());
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if guard.contains_key(&id) {
        return Ok(());
    }

    let handle = attach_native(&window, scene.doc.clone(), &id, &stores)?;
    // The project may already be open (the viewport attaches when the workspace
    // mounts, which can be either side of a project open), so push the content
    // root now as well as from `project://changed` — P16.4a.
    if let Some(root) = window
        .app_handle()
        .try_state::<super::ProjectState>()
        .and_then(|p| p.current_content_root())
    {
        handle.set_content_root(Some(root));
    }
    guard.insert(id.clone(), handle);
    // Released before the log line, which re-locks to list the map: `attached`
    // is the one place the keyed set is visible, and a viewport that came up
    // under an unexpected id is otherwise invisible until something silently
    // fails to reach it.
    drop(guard);
    tracing::info!(
        "viewport '{id}' attached; live viewports: {:?}",
        state.attached()
    );
    Ok(())
}

/// Stamp a tool status with the viewport that raised it (P23.2a).
///
/// The viewport thread builds the DTO and does not know its own key — Ring 2
/// owns the id→handle map — so the id is written on the way out. Pulled out as
/// a function so the claim "the sink stamps it" is testable without a GPU.
fn stamp_tool_status(mut status: ViewportToolStatusDto, id: &str) -> ViewportToolStatusDto {
    status.viewport = id.to_string();
    status
}

/// Build the event sink that turns [`ViewportEvent`]s into namespaced webview
/// events, stamped with `id` (P23.2a). Forwarded key chords arrive on
/// `viewport://key` so the frontend's keybinding dispatcher can replay them
/// (focus handoff, P2.3.4).
///
/// The channels are shared by every viewport and the payloads carry the id, so
/// a listener filters rather than subscribing per viewport — one subscription
/// whatever the count, and a frontend that ignores the id keeps behaving as it
/// did while `primary` is the only sender.
fn event_sink(app: tauri::AppHandle, id: String) -> inf_viewport::ViewportEventSink {
    Arc::new(move |event: ViewportEvent| match event {
        ViewportEvent::Key(chord) => {
            let payload = ViewportKey {
                chord: chord.chord,
                viewport: id.clone(),
            };
            if let Err(e) = app.emit("viewport://key", payload) {
                tracing::warn!("viewport://key emit failed: {e}");
            }
        }
        // A pick-select or gizmo drag on the viewport thread mutated the shared
        // document; re-emit the world delta so the Outliner/Details re-sync.
        ViewportEvent::WorldChanged => {
            if let Some(state) = app.try_state::<SceneState>() {
                emit_world_delta(&app, &state);
            }
        }
        // The gizmo mode changed on the viewport side (a W/E/R keypress or an IPC
        // set); forward it so the toolbar stays in sync (two-way, Wave 2).
        ViewportEvent::GizmoModeChanged(mode) => {
            let payload = ViewportGizmoDto {
                mode,
                viewport: id.clone(),
            };
            if let Err(e) = app.emit("viewport://gizmo", payload) {
                tracing::warn!("viewport://gizmo emit failed: {e}");
            }
        }
        // A tool rejection and/or a change in whether the projected terrain
        // streams (P16.4a). The shell shows the message in the status bar and
        // greys the sculpt/paint tools out while the terrain is streamed.
        ViewportEvent::ToolStatus(status) => {
            if let Err(e) = app.emit("viewport://tool-status", stamp_tool_status(status, &id)) {
                tracing::warn!("viewport://tool-status emit failed: {e}");
            }
        }
    })
}

#[cfg(windows)]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
    id: &str,
    stores: &SharedStores,
) -> Result<inf_viewport::ViewportHandle, String> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let hwnd = match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
        RawWindowHandle::Win32(h) => h.hwnd.get(),
        _ => return Err("expected a Win32 window handle".into()),
    };
    Ok(inf_viewport::spawn(
        hwnd,
        event_sink(window.app_handle().clone(), id.to_string()),
        scene,
        stores.voxel_volumes(),
        stores.fracture_states(),
    ))
}

#[cfg(target_os = "macos")]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
    id: &str,
    stores: &SharedStores,
) -> Result<inf_viewport::ViewportHandle, String> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let ns_view = match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
        RawWindowHandle::AppKit(h) => h.ns_view.as_ptr() as isize,
        _ => return Err("expected an AppKit window handle".into()),
    };
    let sink = event_sink(window.app_handle().clone(), id.to_string());
    let volumes = stores.voxel_volumes();
    let fractures = stores.fracture_states();
    // AppKit setup must happen on the main thread; commands run on a worker.
    let (tx, rx) = std::sync::mpsc::channel();
    window
        .run_on_main_thread(move || {
            let _ = tx.send(inf_viewport::spawn(
                ns_view, sink, scene, volumes, fractures,
            ));
        })
        .map_err(|e| e.to_string())?;
    rx.recv().map_err(|e| e.to_string())
}

#[cfg(not(any(windows, target_os = "macos")))]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
    id: &str,
    stores: &SharedStores,
) -> Result<inf_viewport::ViewportHandle, String> {
    // Linux embedding (X11 reparent / Wayland streaming fallback) is a later
    // Spike A batch — see ROADMAP §5.
    Ok(inf_viewport::spawn(
        0,
        event_sink(window.app_handle().clone(), id.to_string()),
        scene,
        stores.voxel_volumes(),
        stores.fracture_states(),
    ))
}

/// Update the viewport rectangle ([`ViewportRect`]: physical pixels relative
/// to the window client area; the frontend multiplies by devicePixelRatio).
#[tauri::command]
pub async fn viewport_set_rect(
    rect: ViewportRect,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let rect = inf_viewport::ViewportRect {
        x: rect.x.round() as i32,
        y: rect.y.round() as i32,
        width: rect.width.round().max(0.0) as u32,
        height: rect.height.round().max(0.0) as u32,
    };
    state.with(Target::from_arg(viewport), |h| h.set_rect(rect));
    Ok(())
}

/// Show/hide the native viewport. The shell hides it while an HTML overlay
/// (menu, command palette, dialog, drag ghost) is open — the native child
/// otherwise draws OVER the webview (airspace rule) and occludes any
/// overlay crossing the hole. P2.1 explores flash-free alternatives.
#[tauri::command]
pub async fn viewport_set_visible(
    visible: bool,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    state.set_visible(Target::from_arg(viewport), visible);
    Ok(())
}

/// Switch the active viewport projection (Perspective ↔ 2D ortho) from the
/// viewport toolbar (P8.2c).
#[tauri::command]
pub async fn viewport_set_mode(
    mode: ViewportModeDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let mode = match mode {
        ViewportModeDto::Perspective => ViewportMode::Perspective,
        ViewportModeDto::TwoD => ViewportMode::TwoD,
    };
    state.with(Target::from_arg(viewport), |h| h.set_mode(mode));
    Ok(())
}

/// Push the 2D-mode snapping configuration (grid + pixel snap) to the viewport
/// (P8.2c). The frontend folds the per-project pixels-per-unit into this.
#[tauri::command]
pub async fn viewport_set_snap2d(
    snap: Snap2DDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let snap = Snap2DSettings {
        grid_enabled: snap.grid_enabled,
        grid_size: snap.grid_size,
        pixel_enabled: snap.pixel_enabled,
        pixels_per_unit: snap.pixels_per_unit,
    };
    state.with(Target::from_arg(viewport), |h| h.set_snap_2d(snap));
    Ok(())
}

/// Switch the active viewport tool (Select ↔ Sculpt) from the toolbar (P10.2b).
#[tauri::command]
pub async fn viewport_set_tool_mode(
    mode: ToolModeDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let mode = match mode {
        ToolModeDto::Select => ToolMode::Select,
        ToolModeDto::Sculpt => ToolMode::Sculpt,
        ToolModeDto::Foliage => ToolMode::Foliage,
        ToolModeDto::Biome => ToolMode::Biome,
        ToolModeDto::Water => ToolMode::Water,
        ToolModeDto::Voxel => ToolMode::Voxel,
    };
    state.with(Target::from_arg(viewport), |h| h.set_tool_mode(mode));
    Ok(())
}

/// Map the IPC falloff DTO to the viewport's curve enum. Shared by the sculpt and
/// biome brush pushes — the two brushes take the same curve, and a second copy of
/// this match is a place for them to drift apart.
fn to_falloff(d: SculptFalloffDto) -> SculptFalloff {
    match d {
        SculptFalloffDto::Smooth => SculptFalloff::Smooth,
        SculptFalloffDto::Linear => SculptFalloff::Linear,
        SculptFalloffDto::Sphere => SculptFalloff::Sphere,
        SculptFalloffDto::Sharp => SculptFalloff::Sharp,
    }
}

/// Push the sculpt brush configuration (op / radius / strength / falloff) to the
/// viewport (P10.2b).
#[tauri::command]
pub async fn viewport_set_sculpt(
    sculpt: SculptSettingsDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let sculpt = SculptSettings {
        op: match sculpt.op {
            SculptOpDto::Raise => SculptOp::Raise,
            SculptOpDto::Lower => SculptOp::Lower,
            SculptOpDto::Smooth => SculptOp::Smooth,
            SculptOpDto::Flatten => SculptOp::Flatten,
            SculptOpDto::Noise => SculptOp::Noise,
            SculptOpDto::Paint => SculptOp::Paint,
        },
        // One door, not a hand-written half of one (C4-35): `radius.max(0.0)`
        // used to stand here with `strength` unguarded on the next line.
        radius: sculpt.radius,
        strength: sculpt.strength,
        falloff: to_falloff(sculpt.falloff),
        paint_layer: sculpt.paint_layer,
    }
    .sanitized();
    state.with(Target::from_arg(viewport), |h| h.set_sculpt(sculpt));
    Ok(())
}

/// Push the biome brush configuration (radius / strength / falloff / id) to the
/// viewport (P19.2).
///
/// `strength` is clamped to `[0, 1]` because it is not a rate here: it selects
/// which falloff contour the painted biome's hard boundary lands on (see
/// `inf_terrain::biomepaint`), so a value outside the unit range names no
/// contour at all. `biome` needs no clamp — it is a `u8`, and every value
/// including the reserved `0` (the eraser) is meaningful.
#[tauri::command]
pub async fn viewport_set_biome(
    biome: BiomeSettingsDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let biome = BiomeSettings {
        radius: biome.radius.max(0.0),
        strength: biome.strength.clamp(0.0, 1.0),
        falloff: to_falloff(biome.falloff),
        biome: biome.biome,
    };
    state.with(Target::from_arg(viewport), |h| h.set_biome(biome));
    Ok(())
}

/// Push the water-tool configuration (kind / river dimensions / level offset) to
/// the viewport (P20.4).
///
/// There is no level field: the tool resolves one per click, from the biome
/// painted under the cursor, through the id-indexed table
/// `push_biome_palettes` pushes beside the overlay palette. A toolbar-supplied
/// hint would be a level the author never chose applied wherever they clicked.
#[tauri::command]
pub async fn viewport_set_water(
    water: WaterSettingsDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let water = WaterSettings {
        kind: match water.kind {
            WaterToolKindDto::River => WaterToolKind::River,
            WaterToolKindDto::Lake => WaterToolKind::Lake,
        },
        width_m: water.width_m.max(0.0),
        depth_m: water.depth_m.max(0.0),
        // NOT clamped: a negative flow reverses a river without re-authoring
        // its spline, which is a `WaterBody` feature, not a mistake.
        flow_m_s: water.flow_m_s,
        level_offset_m: water.level_offset_m,
    };
    state.with(Target::from_arg(viewport), |h| h.set_water(water));
    Ok(())
}

/// Push the voxel carve-tool configuration (sub-mode / radius / depth /
/// carve-or-fill / material) to the viewport (P21.2).
///
/// Both lengths are clamped non-negative — a negative radius or depth names no
/// cut, and `VoxelShape::is_valid` would reject it one layer down anyway, which
/// would read as a tool that silently does nothing. `material` needs no clamp:
/// it is a `u8` splat index and the Ring-0 op clamps it to the material count.
#[tauri::command]
pub async fn viewport_set_voxel(
    voxel: VoxelSettingsDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let voxel = VoxelSettings {
        kind: match voxel.kind {
            VoxelToolKindDto::Brush => VoxelToolKind::Brush,
            VoxelToolKindDto::Tunnel => VoxelToolKind::Tunnel,
            VoxelToolKindDto::BoxCut => VoxelToolKind::BoxCut,
            VoxelToolKindDto::Trench => VoxelToolKind::Trench,
        },
        radius_m: voxel.radius_m.max(0.0),
        depth_m: voxel.depth_m.max(0.0),
        mode: match voxel.mode {
            VoxelOpModeDto::Carve => VoxelOpMode::Carve,
            VoxelOpModeDto::Fill => VoxelOpMode::Fill,
        },
        material: voxel.material,
        dig_to_depth: voxel.dig_to_depth,
        spoil: match voxel.spoil {
            SpoilModeDto::Off => SpoilMode::Off,
            SpoilModeDto::Auto => SpoilMode::Auto,
            SpoilModeDto::Site => SpoilMode::Site,
        },
        pick_spoil_site: voxel.pick_spoil_site,
    };
    state.with(Target::from_arg(viewport), |h| h.set_voxel(voxel));
    Ok(())
}

/// The Voxel tool's live verdict (P21.2): what this level can be carved into,
/// what a surface-crossing cut would be refused for, and how much is unsaved.
///
/// Answered from the **document** and the shared voxel store, never from the
/// viewport's render cut — the document is what gets saved, so the terrain that
/// has to be able to persist a mouth is the one whose tiles are about to be
/// written. That is `SceneDoc::carve_verdict`'s rule, and this reads the same
/// doors so the toolbar and the tool cannot disagree.
///
/// Camera-independent on purpose: everything here is a fact about the level, not
/// about where the cursor is, so the readout is available before the first click
/// rather than only after the first refusal.
#[tauri::command]
pub async fn viewport_voxel_status(
    stores: tauri::State<'_, SharedStores>,
    scene: tauri::State<'_, SceneState>,
) -> Result<VoxelStatusDto, String> {
    // Lock order — document, then volumes — so this can never deadlock against
    // the viewport loop or the undo path, which both take them that way round.
    let mut level = {
        let doc = scene.doc.lock().map_err(|e| e.to_string())?;
        inf_editor_core::voxel_edit::level_status(&doc)
    };
    // With no viewport attached the store is EMPTY, so nothing is bound and
    // nothing has been carved — the honest answer rather than an error, and the
    // same one the pre-P23.2a `None` stood for, now reached without asking a
    // viewport at all (the store is the process's, not a window's).
    //
    // A **poisoned** store is not that answer, and it does not get to borrow it
    // (P21.2 audit): "0 bound, 0 unsaved" is a claim, and the one state in which
    // it is a lie is the one where the author most needs the truth. It rides the
    // advisory list, which the toolbar already sorts first.
    let volumes = stores.voxel_volumes();
    let (bound_volumes, unsaved_chunks) = match volumes.lock() {
        Ok(store) => {
            // **The replay-fault readout.** A Ctrl+Z that put back less than it
            // took leaves a world that looks entirely plausible, so the counter is
            // the only witness an author will ever see (the Ring-0 skip counts and
            // the missing-volume refusals both land on it).
            let faults = store.replay_faults();
            if faults > 0 {
                level.advisories.push(format!(
                    "{faults} carve undo/redo replay(s) could not fully land — some Ctrl+Z put \
                     back less than it took, so a cave may be part-open. Check the Output Log, \
                     and prefer re-carving over saving this level."
                ));
            }
            (store.bound_volumes().len(), store.unsaved_chunk_count())
        }
        Err(_) => {
            level.advisories.push(
                "The voxel working set is unreadable — a thread panicked while holding it. \
                 Carve counts below are unknown, carving is refused, and a save cannot write \
                 the .inf_voxel assets. Save the level and restart the editor."
                    .to_string(),
            );
            (0, 0)
        }
    };
    Ok(VoxelStatusDto {
        volumes: level.volumes as u32,
        bound_volumes: bound_volumes as u32,
        asset_backed_terrains: level.asset_backed_terrains as u32,
        // One string quoted in both places, so a refusal cannot be explained one
        // way by the toolbar and another by the tool that raised it.
        refusal: (!level.inline_terrains.is_empty())
            .then(|| inf_editor_core::scene::undo::INLINE_TERRAIN_CARVE_REFUSAL.to_string()),
        inline_terrains: level.inline_terrains,
        unsaved_chunks: unsaved_chunks as u32,
        advisories: level.advisories,
    })
}

/// Push the foliage brush configuration (radius / density / kind / …) to the
/// viewport (E-P6).
#[tauri::command]
pub async fn viewport_set_foliage(
    foliage: FoliageSettingsDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let foliage = FoliageSettings {
        radius: foliage.radius.max(0.0),
        density: foliage.density.max(0.0),
        erase: foliage.erase,
        kind: foliage.kind,
        scale_jitter: foliage.scale_jitter.max(0.0),
        align_to_normal: foliage.align_to_normal,
        seed: foliage.seed,
    };
    state.with(Target::from_arg(viewport), |h| h.set_foliage(foliage));
    Ok(())
}

/// Hand off a drag that ended over the viewport hole ([`ViewportDrop`]:
/// physical pixels relative to the hole's top-left corner). HTML drag ghosts
/// die over the native window (airspace rule), so the drop point crosses via
/// IPC and the engine side takes over — Phase 3 turns this into a pick-ray
/// spawn.
#[tauri::command]
pub async fn viewport_drop(
    drop: ViewportDrop,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    state.with(Target::from_arg(viewport), |h| {
        h.drop_payload(drop.x as f32, drop.y as f32, &drop.payload)
    });
    Ok(())
}

/// Set the transform-gizmo mode (translate/rotate/scale) from the toolbar or
/// command palette (Wave 2). The viewport echoes mode changes (including W/E/R
/// keypresses over the viewport) back on the `viewport://gizmo` event.
#[tauri::command]
pub async fn viewport_set_gizmo_mode(
    mode: GizmoModeDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    state.with(Target::from_arg(viewport), |h| h.set_gizmo_mode(mode));
    Ok(())
}

/// Switch the gizmo orientation frame (World ↔ Local) from the toolbar (Wave 2).
#[tauri::command]
pub async fn viewport_set_gizmo_space(
    space: GizmoSpaceDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let space = match space {
        GizmoSpaceDto::World => GizmoSpace::World,
        GizmoSpaceDto::Local => GizmoSpace::Local,
    };
    state.with(Target::from_arg(viewport), |h| h.set_gizmo_space(space));
    Ok(())
}

/// Set the shading view mode (Lit / Unlit / Wireframe / Biomes) from the viewport
/// toolbar (R-P2, P19.2). `Wireframe` degrades to `Unlit` in the renderer when the
/// adapter lacks `POLYGON_MODE_LINE`; `Biomes` needs no GPU feature and never
/// degrades. Editor-transient (never persisted; the player never sets it).
///
/// The DTO crosses whole — the DTO→`inf_render::ViewMode` mapping lives in
/// `inf-viewport`'s per-platform `to_view_mode`, not here.
#[tauri::command]
pub async fn viewport_set_view_mode(
    mode: ViewModeDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    state.with(Target::from_arg(viewport), |h| h.set_view_mode(mode));
    Ok(())
}

/// Push the 3D transform-gizmo snap increments (translate/rotate/scale +
/// always-on) from the toolbar (Wave 2).
#[tauri::command]
pub async fn viewport_set_snap3d(
    snap: Snap3DDto,
    viewport: Option<String>,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let snap = SnapSettings {
        translate: snap.translate.max(0.0),
        rotate_deg: snap.rotate_deg.max(0.0),
        scale: snap.scale.max(0.0),
        always_on: snap.always_on,
    };
    state.with(Target::from_arg(viewport), |h| h.set_snap_3d(snap));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_editor_core::ipc::ViewportToolStatusDto;

    /// A stand-in for `ViewportHandle`, which cannot be constructed without a
    /// real OS window. The map's resolution rule is the thing under test and it
    /// is generic over the value, which is why it was pulled out of `with`.
    fn map(ids: &[&str]) -> BTreeMap<String, u32> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| ((*id).to_string(), i as u32))
            .collect()
    }

    #[test]
    fn primary_resolves_to_the_scene_viewport_and_nothing_else() {
        let m = map(&[PRIMARY_VIEWPORT, "model", "camera"]);
        assert_eq!(resolve(&m, &Target::Primary), vec![&0]);
    }

    /// The wire-compatibility claim, directly: today's TS wrappers send no
    /// `viewport` key, and the command must therefore mean the scene viewport.
    #[test]
    fn an_absent_viewport_argument_means_primary() {
        assert_eq!(Target::from_arg(None), Target::Primary);
        assert_eq!(
            Target::from_arg(Some("model".into())),
            Target::Named("model".into())
        );
    }

    /// `Named("primary")` and `Primary` are the same slot — one key space, so a
    /// caller that spells the id out cannot address a *different* viewport than
    /// the default does.
    #[test]
    fn naming_the_primary_id_is_the_same_as_primary() {
        let m = map(&[PRIMARY_VIEWPORT, "model"]);
        assert_eq!(
            resolve(&m, &Target::Named(PRIMARY_VIEWPORT.into())),
            resolve(&m, &Target::Primary)
        );
    }

    /// **The broadcast semantics.** `All` reaches every viewport — this is what a
    /// content-root change, an asset-index refresh, a stream reload and a
    /// document swap need, and the assertion that fails if one of them is
    /// quietly narrowed to `Primary` in a future edit.
    #[test]
    fn all_reaches_every_attached_viewport() {
        let m = map(&[PRIMARY_VIEWPORT, "model", "camera"]);
        let hit = resolve(&m, &Target::All);
        assert_eq!(hit.len(), 3, "All must reach every viewport, not just one");
        assert_ne!(
            resolve(&m, &Target::All).len(),
            resolve(&m, &Target::Primary).len(),
            "with more than one viewport attached, All and Primary must differ — \
             otherwise the distinction this whole type exists for is untestable"
        );
    }

    /// A miss is empty, never a panic and never a fallback to *some* viewport:
    /// addressing a viewport that is not attached must do nothing at all.
    #[test]
    fn an_unattached_id_resolves_to_nothing() {
        let m = map(&["model"]);
        assert!(resolve(&m, &Target::Primary).is_empty());
        assert!(resolve(&m, &Target::Named("camera".into())).is_empty());
        assert!(resolve(&BTreeMap::<String, u32>::new(), &Target::All).is_empty());
    }

    /// The sink stamps the id the viewport thread cannot know (P23.2a). An
    /// unstamped status reaches the frontend as `viewport: ""`, which its
    /// Primary filter drops — so this is the difference between a status bar
    /// that speaks and one that is silent.
    #[test]
    fn the_sink_stamps_the_viewport_id_onto_a_tool_status() {
        let raw = ViewportToolStatusDto {
            message: Some("nope".into()),
            terrain_streamed: true,
            terrain_editable: false,
            terrain_unsaved_edits: false,
            viewport: String::new(),
        };
        assert_eq!(raw.viewport, "", "the viewport builds it unstamped");
        let out = stamp_tool_status(raw, PRIMARY_VIEWPORT);
        assert_eq!(out.viewport, PRIMARY_VIEWPORT);
        assert_eq!(out.message.as_deref(), Some("nope"), "nothing else moves");
        assert!(out.terrain_streamed);
    }

    /// **The exactly-one-store invariant** (the hoist). Every clone a viewport
    /// is handed points at the SAME allocation, so a second viewport cannot
    /// carve into a store the save path never looks at.
    #[test]
    fn every_viewport_is_handed_the_same_store() {
        let stores = SharedStores::default();
        let first = stores.voxel_volumes();
        let second = stores.voxel_volumes();
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "two `spawn`s must receive clones of ONE carve store"
        );
        assert!(std::sync::Arc::ptr_eq(
            &stores.fracture_states(),
            &stores.fracture_states()
        ));
        // …and a `SharedStores` clone (what `.manage` + `State` hand around) is
        // still the same store, not a second one.
        let cloned = stores.clone();
        assert!(std::sync::Arc::ptr_eq(
            &stores.voxel_volumes(),
            &cloned.voxel_volumes()
        ));
    }

    /// Two *independently constructed* store sets are genuinely different —
    /// without this the test above is vacuous (everything would be ptr-equal if
    /// `shared_volumes` handed back a process-global singleton, and the
    /// "exactly one" claim would be true for a reason nobody chose).
    #[test]
    fn independently_constructed_stores_are_not_the_same_store() {
        let a = SharedStores::default();
        let b = SharedStores::default();
        assert!(!std::sync::Arc::ptr_eq(
            &a.voxel_volumes(),
            &b.voxel_volumes()
        ));
    }
}
