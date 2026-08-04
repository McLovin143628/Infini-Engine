//! Viewport commands: attach the native engine viewport to the editor window
//! and keep its rectangle in sync with the React layout hole.
//!
//! Events/commands here are the Spike A bridge (docs/ROADMAP.md §5): the
//! frontend reports CSS rects × devicePixelRatio; we forward physical pixels
//! to the `inf-viewport` thread.

use std::sync::Arc;
use std::sync::Mutex;

use inf_editor_core::ipc::{
    BiomeSettingsDto, FoliageSettingsDto, GizmoModeDto, GizmoSpaceDto, SculptFalloffDto,
    SculptOpDto, SculptSettingsDto, Snap2DDto, Snap3DDto, ToolModeDto, ViewModeDto, ViewportDrop,
    ViewportKey, ViewportModeDto, ViewportRect, VoxelOpModeDto, VoxelSettingsDto, VoxelToolKindDto,
    WaterSettingsDto, WaterToolKindDto,
};
use inf_viewport::camera::{BiomeSettings, WaterSettings, WaterToolKind};
use inf_viewport::{
    FoliageSettings, GizmoSpace, SculptFalloff, SculptOp, SculptSettings, Snap2DSettings,
    SnapSettings, ToolMode, ViewportEvent, ViewportMode, VoxelOpMode, VoxelSettings, VoxelToolKind,
};
use tauri::{Emitter, Manager};

use super::scene::{emit_world_delta, SceneState};

#[derive(Default)]
pub struct ViewportState(Mutex<Option<inf_viewport::ViewportHandle>>);

impl ViewportState {
    /// Show/hide the native viewport child (used by PIE embedding to hide the
    /// editor viewport while the player window occupies the slot).
    pub fn set_visible(&self, visible: bool) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.set_visible(visible);
            }
        }
    }

    /// Adopt a foreign (PIE player) window into the viewport slot (embedded PIE).
    pub fn embed_foreign(&self, hwnd: i64) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.embed_foreign(hwnd as isize);
            }
        }
    }

    /// Point the viewport's terrain streamer at a project's content root (or
    /// `None` to disable it) — pushed from the `project://changed` flow so a
    /// `Terrain.asset` authored by the import wizard resolves to a loose
    /// `.inf_terrain` and starts paging (P16.4a, the B2 seam).
    pub fn set_content_root(&self, root: Option<std::path::PathBuf>) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.set_content_root(root);
            }
        }
    }

    /// Rebuild the viewport's loose `.inf_terrain` index in place, keeping live
    /// streams — pushed when a terrain import finishes (P16.4a).
    pub fn refresh_asset_index(&self) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.refresh_asset_index();
            }
        }
    }

    /// Reopen every live terrain stream's `.inf_terrain` in place — pushed by the
    /// save path once it has written sculpt/paint edits back into the assets
    /// (P16.4b). Live streams keep their resident pages and published cut, so a
    /// save never blinks the terrain the user is looking at.
    pub fn reload_terrain_stores(&self) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.reload_terrain_stores();
            }
        }
    }

    /// Re-walk the viewport's loose `.inf_voxel` index in place — pushed by the
    /// save path once it has folded carve edits back into the assets (P21.2).
    /// The `reload_terrain_stores` twin, and the seam the P21.2 save flow needs:
    /// a carve written into a *new* asset only resolves after a re-walk, and the
    /// loaded volumes keep their chunks so saving never blinks a cave.
    ///
    /// **Unwired on purpose, for exactly one commit.** Its caller is
    /// `commands/scene.rs`'s save path, beside the existing
    /// `viewport.reload_terrain_stores()` call that follows the `.inf_terrain`
    /// write-back — which the batch folding carve edits into `.inf_voxel` owns.
    /// The whole chain below this point (`ViewportHandle::reload_voxel_stores`
    /// → `Cmd::ReloadVoxelStores` on both platform pumps →
    /// `EngineHost::reload_voxel_stores`) is live, so that batch adds one line
    /// and nothing else. Deleting this instead would mean re-deriving the same
    /// four-file chain from scratch.
    #[allow(dead_code)]
    pub fn reload_voxel_stores(&self) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.reload_voxel_stores();
            }
        }
    }

    /// Release every terrain stream — pushed when the open document is replaced
    /// (File ▸ Open / File ▸ New, P16.4b). The streams are keyed on the previous
    /// document's entity GUIDs, so keeping them leaks a whole `.inf_terrain`
    /// payload plus any tile it pinned for an unsaved edit.
    pub fn clear_streams(&self) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.clear_streams();
            }
        }
    }

    /// Push a terrain entity's biome **overlay palette** (linear RGBA, indexed by
    /// biome id) to the viewport (P19.2). An EMPTY palette clears the entry, which
    /// is what an unbound terrain must send — otherwise the Biomes view mode would
    /// keep tinting with a vocabulary the terrain no longer names.
    ///
    /// Public so the terrain/asset commands can re-push after a bind or a
    /// `.inf_biomes` save without reaching into the handle themselves.
    pub fn set_biome_palette(&self, entity: uuid::Uuid, palette: Vec<[f32; 4]>) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.set_biome_palette(entity, palette);
            }
        }
    }

    /// Push a terrain's id-indexed water-level hints to the native viewport
    /// (P20.4) — the `set_biome_palette` twin, pushed from the same place.
    pub fn set_water_hints(&self, entity: uuid::Uuid, hints: Vec<Option<f64>>) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.set_water_hints(entity, hints);
            }
        }
    }

    /// Release an embedded foreign window and restore the native viewport child.
    pub fn release_foreign(&self) {
        if let Ok(guard) = self.0.lock() {
            if let Some(handle) = guard.as_ref() {
                handle.release_foreign();
            }
        }
    }
}

/// Create (once) the native child viewport inside the calling window.
#[tauri::command]
pub async fn viewport_attach(
    window: tauri::Window,
    state: tauri::State<'_, ViewportState>,
    scene: tauri::State<'_, SceneState>,
) -> Result<(), String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if guard.is_some() {
        return Ok(()); // idempotent: React StrictMode double-mounts
    }

    let handle = attach_native(&window, scene.doc.clone())?;
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
    *guard = Some(handle);
    tracing::info!("viewport attached");
    Ok(())
}

/// Build the event sink that turns [`ViewportEvent`]s into namespaced webview
/// events. Forwarded key chords arrive on `viewport://key` so the frontend's
/// keybinding dispatcher can replay them (focus handoff, P2.3.4).
fn event_sink(app: tauri::AppHandle) -> inf_viewport::ViewportEventSink {
    Arc::new(move |event: ViewportEvent| match event {
        ViewportEvent::Key(chord) => {
            if let Err(e) = app.emit("viewport://key", ViewportKey { chord: chord.chord }) {
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
            if let Err(e) = app.emit("viewport://gizmo", mode) {
                tracing::warn!("viewport://gizmo emit failed: {e}");
            }
        }
        // A tool rejection and/or a change in whether the projected terrain
        // streams (P16.4a). The shell shows the message in the status bar and
        // greys the sculpt/paint tools out while the terrain is streamed.
        ViewportEvent::ToolStatus(status) => {
            if let Err(e) = app.emit("viewport://tool-status", status) {
                tracing::warn!("viewport://tool-status emit failed: {e}");
            }
        }
    })
}

#[cfg(windows)]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
) -> Result<inf_viewport::ViewportHandle, String> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let hwnd = match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
        RawWindowHandle::Win32(h) => h.hwnd.get(),
        _ => return Err("expected a Win32 window handle".into()),
    };
    Ok(inf_viewport::spawn(
        hwnd,
        event_sink(window.app_handle().clone()),
        scene,
    ))
}

#[cfg(target_os = "macos")]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
) -> Result<inf_viewport::ViewportHandle, String> {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let ns_view = match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
        RawWindowHandle::AppKit(h) => h.ns_view.as_ptr() as isize,
        _ => return Err("expected an AppKit window handle".into()),
    };
    let sink = event_sink(window.app_handle().clone());
    // AppKit setup must happen on the main thread; commands run on a worker.
    let (tx, rx) = std::sync::mpsc::channel();
    window
        .run_on_main_thread(move || {
            let _ = tx.send(inf_viewport::spawn(ns_view, sink, scene));
        })
        .map_err(|e| e.to_string())?;
    rx.recv().map_err(|e| e.to_string())
}

#[cfg(not(any(windows, target_os = "macos")))]
fn attach_native(
    window: &tauri::Window,
    scene: inf_viewport::SharedScene,
) -> Result<inf_viewport::ViewportHandle, String> {
    // Linux embedding (X11 reparent / Wayland streaming fallback) is a later
    // Spike A batch — see ROADMAP §5.
    Ok(inf_viewport::spawn(
        0,
        event_sink(window.app_handle().clone()),
        scene,
    ))
}

/// Update the viewport rectangle ([`ViewportRect`]: physical pixels relative
/// to the window client area; the frontend multiplies by devicePixelRatio).
#[tauri::command]
pub async fn viewport_set_rect(
    rect: ViewportRect,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_rect(inf_viewport::ViewportRect {
            x: rect.x.round() as i32,
            y: rect.y.round() as i32,
            width: rect.width.round().max(0.0) as u32,
            height: rect.height.round().max(0.0) as u32,
        });
    }
    Ok(())
}

/// Show/hide the native viewport. The shell hides it while an HTML overlay
/// (menu, command palette, dialog, drag ghost) is open — the native child
/// otherwise draws OVER the webview (airspace rule) and occludes any
/// overlay crossing the hole. P2.1 explores flash-free alternatives.
#[tauri::command]
pub async fn viewport_set_visible(
    visible: bool,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_visible(visible);
    }
    Ok(())
}

/// Switch the active viewport projection (Perspective ↔ 2D ortho) from the
/// viewport toolbar (P8.2c).
#[tauri::command]
pub async fn viewport_set_mode(
    mode: ViewportModeDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_mode(match mode {
            ViewportModeDto::Perspective => ViewportMode::Perspective,
            ViewportModeDto::TwoD => ViewportMode::TwoD,
        });
    }
    Ok(())
}

/// Push the 2D-mode snapping configuration (grid + pixel snap) to the viewport
/// (P8.2c). The frontend folds the per-project pixels-per-unit into this.
#[tauri::command]
pub async fn viewport_set_snap2d(
    snap: Snap2DDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_snap_2d(Snap2DSettings {
            grid_enabled: snap.grid_enabled,
            grid_size: snap.grid_size,
            pixel_enabled: snap.pixel_enabled,
            pixels_per_unit: snap.pixels_per_unit,
        });
    }
    Ok(())
}

/// Switch the active viewport tool (Select ↔ Sculpt) from the toolbar (P10.2b).
#[tauri::command]
pub async fn viewport_set_tool_mode(
    mode: ToolModeDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_tool_mode(match mode {
            ToolModeDto::Select => ToolMode::Select,
            ToolModeDto::Sculpt => ToolMode::Sculpt,
            ToolModeDto::Foliage => ToolMode::Foliage,
            ToolModeDto::Biome => ToolMode::Biome,
            ToolModeDto::Water => ToolMode::Water,
            ToolModeDto::Voxel => ToolMode::Voxel,
        });
    }
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
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_sculpt(SculptSettings {
            op: match sculpt.op {
                SculptOpDto::Raise => SculptOp::Raise,
                SculptOpDto::Lower => SculptOp::Lower,
                SculptOpDto::Smooth => SculptOp::Smooth,
                SculptOpDto::Flatten => SculptOp::Flatten,
                SculptOpDto::Noise => SculptOp::Noise,
                SculptOpDto::Paint => SculptOp::Paint,
            },
            radius: sculpt.radius.max(0.0),
            strength: sculpt.strength,
            falloff: to_falloff(sculpt.falloff),
            paint_layer: sculpt.paint_layer.min(3),
        });
    }
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
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_biome(BiomeSettings {
            radius: biome.radius.max(0.0),
            strength: biome.strength.clamp(0.0, 1.0),
            falloff: to_falloff(biome.falloff),
            biome: biome.biome,
        });
    }
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
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_water(WaterSettings {
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
        });
    }
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
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_voxel(VoxelSettings {
            kind: match voxel.kind {
                VoxelToolKindDto::Brush => VoxelToolKind::Brush,
                VoxelToolKindDto::Tunnel => VoxelToolKind::Tunnel,
            },
            radius_m: voxel.radius_m.max(0.0),
            depth_m: voxel.depth_m.max(0.0),
            mode: match voxel.mode {
                VoxelOpModeDto::Carve => VoxelOpMode::Carve,
                VoxelOpModeDto::Fill => VoxelOpMode::Fill,
            },
            material: voxel.material,
        });
    }
    Ok(())
}

/// Push the foliage brush configuration (radius / density / kind / …) to the
/// viewport (E-P6).
#[tauri::command]
pub async fn viewport_set_foliage(
    foliage: FoliageSettingsDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_foliage(FoliageSettings {
            radius: foliage.radius.max(0.0),
            density: foliage.density.max(0.0),
            erase: foliage.erase,
            kind: foliage.kind,
            scale_jitter: foliage.scale_jitter.max(0.0),
            align_to_normal: foliage.align_to_normal,
            seed: foliage.seed,
        });
    }
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
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.drop_payload(drop.x as f32, drop.y as f32, &drop.payload);
    }
    Ok(())
}

/// Set the transform-gizmo mode (translate/rotate/scale) from the toolbar or
/// command palette (Wave 2). The viewport echoes mode changes (including W/E/R
/// keypresses over the viewport) back on the `viewport://gizmo` event.
#[tauri::command]
pub async fn viewport_set_gizmo_mode(
    mode: GizmoModeDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_gizmo_mode(mode);
    }
    Ok(())
}

/// Switch the gizmo orientation frame (World ↔ Local) from the toolbar (Wave 2).
#[tauri::command]
pub async fn viewport_set_gizmo_space(
    space: GizmoSpaceDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_gizmo_space(match space {
            GizmoSpaceDto::World => GizmoSpace::World,
            GizmoSpaceDto::Local => GizmoSpace::Local,
        });
    }
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
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_view_mode(mode);
    }
    Ok(())
}

/// Push the 3D transform-gizmo snap increments (translate/rotate/scale +
/// always-on) from the toolbar (Wave 2).
#[tauri::command]
pub async fn viewport_set_snap3d(
    snap: Snap3DDto,
    state: tauri::State<'_, ViewportState>,
) -> Result<(), String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(handle) = guard.as_ref() {
        handle.set_snap_3d(SnapSettings {
            translate: snap.translate.max(0.0),
            rotate_deg: snap.rotate_deg.max(0.0),
            scale: snap.scale.max(0.0),
            always_on: snap.always_on,
        });
    }
    Ok(())
}
