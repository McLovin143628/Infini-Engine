//! macOS viewport host: a `CAMetalLayer` sublayer added to the Tauri
//! window's layer-backed contentView, rendered by wgpu (Metal) from a
//! dedicated thread.
//!
//! Status: COMPILE-VERIFIED ONLY (cross-checked from Windows against
//! aarch64-apple-darwin; CI compiles it on real macOS). Runtime behavior —
//! retina scale, coordinate flip, cross-thread CALayer geometry writes —
//! needs a hardware pass; see the Spike A memo.
//!
//! Layer-frame updates run on the render thread inside a `CATransaction`
//! with actions disabled. Core Animation tolerates off-main-thread layer
//! property writes on standalone sublayers, but this is exactly the kind of
//! claim the hardware pass must confirm; if it flickers, the fallback is
//! dispatching frame updates to the main queue.
//!
//! Flycam input is not wired on macOS yet (Windows raw-input only for the
//! spike); the camera holds its default pose.

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

use objc2::rc::Retained;
use objc2::MainThreadMarker;
use objc2_app_kit::NSView;
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_quartz_core::{CAMetalLayer, CATransaction};

use crate::camera::{
    BiomeSettings, Camera2D, EditorCamera, FoliageSettings, GizmoSpace, SculptSettings,
    Snap2DSettings, SnapSettings, ToolMode, ViewportMode, VoxelSettings, WaterSettings,
};
use crate::host::EngineHost;
use crate::{SharedScene, SurfaceTarget, ViewportEventSink, ViewportRect};
use inf_editor_core::ipc::{GizmoModeDto, ViewModeDto};
use inf_render::{GizmoMode, ViewMode};

/// Map the IPC gizmo-mode DTO to the renderer enum (Wave 2; mirrors win32).
fn to_gizmo_mode(d: GizmoModeDto) -> GizmoMode {
    match d {
        GizmoModeDto::Translate => GizmoMode::Translate,
        GizmoModeDto::Rotate => GizmoMode::Rotate,
        GizmoModeDto::Scale => GizmoMode::Scale,
    }
}

/// Map the IPC view-mode DTO to the renderer enum (R-P2; mirrors win32).
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

enum Cmd {
    SetRect(ViewportRect),
    SetVisible(bool),
    Drop {
        x: f32,
        y: f32,
        payload: String,
    },
    SetMode(ViewportMode),
    SetSnap2D(Snap2DSettings),
    SetToolMode(ToolMode),
    SetSculpt(SculptSettings),
    SetFoliage(FoliageSettings),
    SetBiome(BiomeSettings),
    /// Replace the water-tool configuration (P20.4).
    SetWater(WaterSettings),
    /// Replace the voxel carve-tool configuration (P21.2).
    SetVoxel(VoxelSettings),
    /// Whether an in-editor Simulate session is live (P29.6). Carried so the two
    /// pumps do not drift; macOS input is unwired for the whole viewport, so
    /// there is no capture to grab and the arm records the flag and does nothing
    /// with it — the honest state of this platform rather than a missing command
    /// that would fail to compile.
    SetSimRunning(bool),
    SetBiomePalette(uuid::Uuid, Vec<[f32; 4]>),
    /// Per-terrain water-level hints by biome id (P20.4).
    SetWaterHints(uuid::Uuid, Vec<Option<f64>>),
    SetGizmo(GizmoMode),
    SetGizmoSpace(GizmoSpace),
    SetSnap3D(SnapSettings),
    SetViewMode(ViewMode),
    /// Point terrain streaming at a project's content root (P16.4a).
    SetContentRoot(Option<std::path::PathBuf>),
    /// Rebuild the loose `.inf_terrain` index in place (a terrain import landed).
    RefreshAssetIndex,
    /// Reopen every live terrain stream's `.inf_terrain` in place after a save
    /// wrote edits back (P16.4b).
    ReloadTerrainStores,
    /// Re-walk the loose `.inf_voxel` index in place — a save just folded carve
    /// edits back into the assets (P21.2). The twin of `ReloadTerrainStores`.
    ReloadVoxelStores,
    /// Release every terrain stream — the document was replaced (P16.4b).
    ClearStreams,
    Destroy,
}

/// Cheap-to-clone handle for controlling the viewport thread.
///
/// **It holds no shared stores** (P23.2a — the hoist; the win32 twin). The
/// carve working set and the Simulate fracture map are created once by Ring 2
/// (`commands::SharedStores`) and handed to `spawn`, so there is exactly one of
/// each per PROCESS however many viewports exist.
pub struct ViewportHandle {
    tx: Sender<Cmd>,
}

impl ViewportHandle {
    /// Move/resize the layer (physical pixels, top-left origin relative to
    /// the parent view — the same contract as Windows).
    pub fn set_rect(&self, rect: ViewportRect) {
        let _ = self.tx.send(Cmd::SetRect(rect));
    }

    /// Show/hide the layer (see the Windows twin for why: HTML overlays
    /// crossing the hole are otherwise occluded by the native surface).
    pub fn set_visible(&self, visible: bool) {
        let _ = self.tx.send(Cmd::SetVisible(visible));
    }

    /// Drag-drop handoff (see the Windows twin for the contract).
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

    /// Replace the 2D-mode snapping configuration.
    pub fn set_snap_2d(&self, snap: Snap2DSettings) {
        let _ = self.tx.send(Cmd::SetSnap2D(snap));
    }

    /// Switch the active viewport tool (Select ↔ Sculpt) (P10.2b). macOS input
    /// isn't wired yet, so this only sets the mode (drives the brush ring once
    /// input lands with the hardware pass).
    pub fn set_tool_mode(&self, mode: ToolMode) {
        let _ = self.tx.send(Cmd::SetToolMode(mode));
    }

    /// Replace the sculpt brush configuration.
    pub fn set_sculpt(&self, sculpt: SculptSettings) {
        let _ = self.tx.send(Cmd::SetSculpt(sculpt));
    }

    /// Replace the foliage brush configuration (E-P6). macOS input isn't wired
    /// yet, so this only sets the host state (the brush drives once input lands).
    pub fn set_foliage(&self, foliage: FoliageSettings) {
        let _ = self.tx.send(Cmd::SetFoliage(foliage));
    }

    /// Replace the biome brush configuration (P19.2).
    pub fn set_biome(&self, biome: BiomeSettings) {
        let _ = self.tx.send(Cmd::SetBiome(biome));
    }

    /// Replace the water-tool configuration (kind / river dimensions / level
    /// offset / resolved biome hint) — P20.4. macOS input isn't wired yet, so
    /// this only sets the host state (the tool authors once input lands).
    pub fn set_water(&self, water: WaterSettings) {
        let _ = self.tx.send(Cmd::SetWater(water));
    }

    /// Replace the voxel carve-tool configuration (sub-mode / radius / depth /
    /// carve-or-fill / material) — P21.2. macOS input isn't wired yet, so this
    /// only sets the host state (the tool authors once input lands).
    pub fn set_voxel(&self, voxel: VoxelSettings) {
        let _ = self.tx.send(Cmd::SetVoxel(voxel));
    }

    /// P29.6: accepted and dropped. macOS input is unwired for the whole
    /// viewport (this file has no capture state machine at all), so the play
    /// capture has nothing to hook — recorded here rather than left as a
    /// missing method that would not compile on the platform.
    pub fn set_sim_running(&self, running: bool) {
        let _ = self.tx.send(Cmd::SetSimRunning(running));
    }

    /// Push a terrain's resolved biome overlay palette (P19.2).
    pub fn set_biome_palette(&self, entity: uuid::Uuid, palette: Vec<[f32; 4]>) {
        let _ = self.tx.send(Cmd::SetBiomePalette(entity, palette));
    }

    /// Push a terrain's per-biome water-level hints (P20.4). An all-`None` table
    /// clears them.
    pub fn set_water_hints(&self, entity: uuid::Uuid, hints: Vec<Option<f64>>) {
        let _ = self.tx.send(Cmd::SetWaterHints(entity, hints));
    }

    /// Set the transform-gizmo mode (Wave 2). macOS input isn't wired yet, so
    /// this only sets the host state (the gizmo draws once input lands).
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

    /// Set the shading view mode (Lit / Unlit / Wireframe) (R-P2). macOS input
    /// isn't wired yet, but the mode still drives the renderer (it's not input).
    pub fn set_view_mode(&self, mode: ViewModeDto) {
        let _ = self.tx.send(Cmd::SetViewMode(to_view_mode(mode)));
    }

    /// Point terrain streaming at a project's content root (P16.4a). Rescans the
    /// loose `.inf_terrain` index and drops every live stream, so a project
    /// switch can never serve the previous project's pages.
    pub fn set_content_root(&self, root: Option<std::path::PathBuf>) {
        let _ = self.tx.send(Cmd::SetContentRoot(root));
    }

    /// Rebuild the loose `.inf_terrain` index without dropping live streams
    /// (P16.4a).
    pub fn refresh_asset_index(&self) {
        let _ = self.tx.send(Cmd::RefreshAssetIndex);
    }

    /// Reopen every live terrain stream's `.inf_terrain` in place after a save
    /// wrote sculpt/paint edits back into it (P16.4b).
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

    /// Adopt a foreign PIE player window (no-op on macOS: cross-process view
    /// adoption is unsupported, so PIE uses "Play in New Window" — see the
    /// Spike D memo). Kept for a uniform `ViewportHandle` surface.
    pub fn embed_foreign(&self, _hwnd: isize) {}

    /// Release an embedded foreign window (no-op on macOS).
    pub fn release_foreign(&self) {}

    /// Tear down the viewport thread and its layer.
    pub fn destroy(&self) {
        let _ = self.tx.send(Cmd::Destroy);
    }
}

/// Create the CAMetalLayer under `ns_view` (the Tauri contentView) and start
/// the render thread. MUST be called on the AppKit main thread; the Ring-2
/// caller dispatches via `run_on_main_thread`.
///
/// `volumes` and `fractures` are the PROCESS's shared stores, created once by
/// Ring 2 and passed in (P23.2a — the hoist; the win32 twin). Every early
/// return below simply drops this viewport's clones: the stores outlive a
/// viewport that never came up, and stage exactly what was carved into them,
/// which is nothing.
pub fn spawn(
    ns_view: isize,
    sink: ViewportEventSink,
    scene: SharedScene,
    volumes: inf_editor_core::voxel_store::SharedVoxelVolumes,
    fractures: inf_editor_core::simulate::SharedFractures,
) -> ViewportHandle {
    let (tx, rx) = channel();

    if MainThreadMarker::new().is_none() {
        tracing::error!("inf-viewport: macOS spawn must run on the main thread");
        return ViewportHandle { tx };
    }

    // SAFETY: the caller passes the live contentView of the editor window
    // and we are on the main thread.
    let (layer_ptr, scale) = unsafe {
        let view = &*(ns_view as *const NSView);
        view.setWantsLayer(true);
        let scale = view.window().map(|w| w.backingScaleFactor()).unwrap_or(2.0);

        let metal = CAMetalLayer::new();
        metal.setContentsScale(scale);
        metal.setFrame(CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: 64.0,
                height: 64.0,
            },
        });
        match view.layer() {
            Some(root) => root.addSublayer(&metal),
            None => {
                tracing::error!("inf-viewport: contentView has no backing layer");
                return ViewportHandle { tx };
            }
        }
        // Intentional +1 retain: the layer lives until Destroy releases it.
        (Retained::into_raw(metal) as isize, scale)
    };

    // macOS input (flycam/orbit + key forwarding) isn't wired yet — the camera
    // holds its default pose, so there are no events to surface. That also means
    // no sculpt/paint stroke can be attempted here, so the P16.4a tool-status
    // drain (`take_tool_status` / `terrain_is_streamed`) has nothing to report
    // either; the hardware pass wires the sink and both together.
    let _ = sink;

    std::thread::Builder::new()
        .name("inf-viewport".into())
        .spawn(move || thread_main(layer_ptr, scale, rx, scene, volumes, fractures))
        .expect("failed to spawn inf-viewport thread");
    ViewportHandle { tx }
}

fn apply_rect(layer_ptr: isize, scale: f64, r: ViewportRect) {
    // SAFETY: layer_ptr holds a +1 retain until Destroy.
    unsafe {
        let layer = &*(layer_ptr as *const CAMetalLayer);
        // Physical px (top-left origin) → points (bottom-left origin).
        let super_h = layer
            .superlayer()
            .map(|s| s.bounds().size.height)
            .unwrap_or(0.0);
        let w = r.width as f64 / scale;
        let h = r.height as f64 / scale;
        let x = r.x as f64 / scale;
        let y = super_h - (r.y as f64 / scale + h);
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        layer.setFrame(CGRect {
            origin: CGPoint { x, y },
            size: CGSize {
                width: w,
                height: h,
            },
        });
        CATransaction::commit();
    }
}

fn thread_main(
    layer_ptr: isize,
    scale: f64,
    rx: Receiver<Cmd>,
    scene: SharedScene,
    volumes: inf_editor_core::voxel_store::SharedVoxelVolumes,
    fractures: inf_editor_core::simulate::SharedFractures,
) {
    let target = SurfaceTarget::MetalLayer { layer: layer_ptr };
    let mut host = match EngineHost::new(target, 64, 64)
        .map(|h| h.with_voxel_volumes(volumes).with_fractures(fractures))
    {
        Ok(h) => h,
        Err(e) => {
            tracing::error!("inf-viewport: engine init failed: {e}");
            return;
        }
    };
    tracing::info!("inf-viewport: CAMetalLayer + engine renderer up");

    let camera = EditorCamera::default();
    // A default 2D camera; macOS input isn't wired yet, so 2D mode renders a
    // static top-down view (pan/zoom arrive with the macOS hardware pass).
    let camera_2d = Camera2D::default();

    'outer: loop {
        let mut latest_rect: Option<ViewportRect> = None;
        loop {
            match rx.try_recv() {
                Ok(Cmd::SetRect(r)) => latest_rect = Some(r),
                Ok(Cmd::SetVisible(v)) => unsafe {
                    // SAFETY: layer_ptr holds a +1 retain until Destroy.
                    let layer = &*(layer_ptr as *const CAMetalLayer);
                    CATransaction::begin();
                    CATransaction::setDisableActions(true);
                    layer.setHidden(!v);
                    CATransaction::commit();
                },
                Ok(Cmd::Drop { x, y, payload }) => {
                    tracing::info!(
                        "inf-viewport: drop '{payload}' at viewport-local ({x:.0}, {y:.0}) px"
                    );
                }
                Ok(Cmd::SetMode(m)) => host.set_mode(m),
                Ok(Cmd::SetSnap2D(s)) => host.set_snap_2d(s),
                Ok(Cmd::SetToolMode(m)) => host.set_tool_mode(m),
                Ok(Cmd::SetSculpt(s)) => host.set_sculpt(s),
                Ok(Cmd::SetFoliage(f)) => host.set_foliage(f),
                Ok(Cmd::SetBiome(b)) => host.set_biome(b),
                Ok(Cmd::SetWater(w)) => host.set_water(w),
                Ok(Cmd::SetVoxel(v)) => host.set_voxel(v),
                // P29.6: recorded and unused — macOS input is unwired for the
                // whole viewport, so there is no capture to grab. The arm exists
                // because Ring 2 calls the setter unconditionally and the pump
                // mirror gate holds the two platforms level.
                Ok(Cmd::SetSimRunning(_running)) => {}
                Ok(Cmd::SetBiomePalette(e, p)) => host.set_biome_palette(e, p),
                Ok(Cmd::SetWaterHints(e, h)) => host.set_water_hints(e, h),
                Ok(Cmd::SetGizmo(m)) => host.set_gizmo_mode(m),
                Ok(Cmd::SetGizmoSpace(s)) => host.set_gizmo_space(s),
                Ok(Cmd::SetSnap3D(s)) => host.set_snap_3d(s),
                Ok(Cmd::SetViewMode(m)) => host.set_view_mode(m),
                Ok(Cmd::SetContentRoot(root)) => host.set_content_root(root),
                Ok(Cmd::RefreshAssetIndex) => host.refresh_asset_index(),
                Ok(Cmd::ReloadTerrainStores) => host.reload_terrain_stores(),
                Ok(Cmd::ReloadVoxelStores) => host.reload_voxel_stores(),
                Ok(Cmd::ClearStreams) => host.clear_streams(),
                Ok(Cmd::Destroy) | Err(TryRecvError::Disconnected) => break 'outer,
                Err(TryRecvError::Empty) => break,
            }
        }
        if let Some(r) = latest_rect {
            apply_rect(layer_ptr, scale, r);
            host.resize(r.width.max(1), r.height.max(1));
        }

        // Project the shared world (read-only on macOS: no input wired yet, so
        // no picking/gizmo writeback — the editor still drives the scene).
        if let Ok(doc) = scene.lock() {
            host.sync_from_doc(&doc);
        }

        // Rebase the floating origin on the active eye, build the view for the
        // current mode, and render. FIFO present blocks at vsync.
        let two_d = host.mode == ViewportMode::TwoD;
        let eye = if two_d { camera_2d.eye() } else { camera.pos };
        host.origin.maybe_rebase(eye);
        let view = if two_d {
            host.view_2d(&camera_2d)
        } else {
            host.view_for(&camera)
        };
        if let Err(e) = host.render_frame(&view) {
            tracing::error!("inf-viewport: unrecoverable render failure: {e}");
            break;
        }
    }

    // SAFETY: reclaim the +1 retain taken in `spawn`; drop releases it.
    unsafe {
        let layer = Retained::from_raw(layer_ptr as *mut CAMetalLayer);
        if let Some(layer) = layer {
            CATransaction::begin();
            CATransaction::setDisableActions(true);
            layer.removeFromSuperlayer();
            CATransaction::commit();
        }
    }
    tracing::info!("inf-viewport: shutting down");
}
