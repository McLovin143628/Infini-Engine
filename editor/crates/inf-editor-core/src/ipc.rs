//! Shared editor↔frontend IPC types.
//!
//! Every struct/enum that crosses a `#[tauri::command]` boundary or a
//! namespaced event channel lives here, derives `serde` + `ts_rs::TS`, and is
//! exported to `editor/studio/src/bindings/` by the `bindings` test in this
//! crate (committed output; CI fails on drift). The frontend imports these
//! generated types through `src/lib/ipc.ts` — hand-written duplicates of
//! backend types are forbidden.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// The viewport hole's rectangle in PHYSICAL pixels relative to the window
/// client area (the frontend multiplies CSS px by `devicePixelRatio`; the
/// backend rounds to device pixels).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct ViewportRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// A drag that ended over the viewport hole. HTML drag ghosts die over the
/// native window (airspace rule), so the drop point crosses via IPC in
/// PHYSICAL pixels relative to the hole's top-left corner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ViewportDrop {
    pub x: f64,
    pub y: f64,
    /// Opaque payload (Phase 4 makes this an asset reference).
    pub payload: String,
}

/// A keyboard chord the native viewport forwarded to the webview on
/// `viewport://key`. When the 3D view holds OS focus, WASD/camera keys are
/// consumed natively but global shortcuts (command palette, save, …) are
/// replayed into the frontend keybinding dispatcher (focus handoff, P2.3.4).
/// `chord` matches the frontend's `chordOf` format ("Ctrl+Shift+P", "F11").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ViewportKey {
    pub chord: String,
    /// Which viewport forwarded it (P23.2a). **Stamped by Ring 2's event sink**,
    /// which owns the id→handle map; the viewport thread does not know its own
    /// key. `"primary"` is the scene viewport.
    pub viewport: String,
}

/// Log severity for the Output Log panel. Mirrors `tracing::Level`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// One structured line on the `log://line` event channel (Output Log panel).
/// Produced by the studio's tracing subscriber layer; `seq` is a per-session
/// monotonic counter so the frontend can detect dropped lines and keep a
/// stable virtual-list identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct LogLine {
    /// Exported as `number`: session-lifetime counts stay far below 2^53.
    #[ts(type = "number")]
    pub seq: u64,
    pub level: LogLevel,
    /// tracing target (module path), e.g. `inf_render::surface`.
    pub target: String,
    pub message: String,
    /// Unix epoch milliseconds (f64 for lossless JS interop).
    pub timestamp_ms: f64,
}

/// A saved dock-layout preset (`layout_list` command).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct LayoutSummary {
    pub name: String,
    /// Last-modified time, unix epoch milliseconds.
    pub modified_ms: f64,
}

// ── Scene / world binding (Phase 3) ──────────────────────────────────────
//
// The authoritative scene is an ECS world in `inf-editor-core::scene`. The
// frontend Outliner/Details never see the world directly — they consume these
// flattened, GUID-keyed DTOs over the `scene_snapshot` command (full state) and
// the `world://delta` event (incremental changes). GUIDs are stable across a
// save/reload; bevy `Entity` ids never cross the boundary.

/// One entity as the Outliner sees it. `guid` is the stable string identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SceneNode {
    pub guid: String,
    pub name: String,
    /// UE-style type column ("Static Mesh", "Point Light", "Folder", …).
    pub kind: String,
    /// This entity's own visibility toggle (the eye).
    pub visible: bool,
    /// Effective visibility (self AND every ancestor) — drives dimming.
    pub effective_visible: bool,
    /// Parent GUID, or `None` for a root.
    pub parent: Option<String>,
    /// Ordered child GUIDs.
    pub children: Vec<String>,
}

/// A full scene snapshot (`scene_snapshot` command; sent on load + resync).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SceneSnapshot {
    /// Monotonic document version — every mutation bumps it. The frontend uses
    /// it to detect a missed delta and re-fetch.
    #[ts(type = "number")]
    pub version: u64,
    /// Ordered root GUIDs.
    pub roots: Vec<String>,
    /// Every node, in no particular order (the frontend builds a map).
    pub nodes: Vec<SceneNode>,
    /// Selected GUIDs (single source of truth: viewport ↔ Outliner ↔ Details).
    pub selection: Vec<String>,
    /// Unsaved changes present.
    pub dirty: bool,
    /// Document title for the tab/status bar ("Untitled", a level name, …).
    pub title: String,
    /// Edit-menu state: whether undo/redo are available + their labels
    /// ("Undo Rename").
    pub can_undo: bool,
    pub can_redo: bool,
    pub undo_label: Option<String>,
    pub redo_label: Option<String>,
}

/// An incremental world change (`world://delta` event). Structural edits ship
/// as added/removed/updated node sets; `roots`, `selection`, and the doc meta
/// are small so they ride along every delta for a trivially-correct reducer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SceneDelta {
    #[ts(type = "number")]
    pub version: u64,
    pub added: Vec<SceneNode>,
    pub removed: Vec<String>,
    pub updated: Vec<SceneNode>,
    pub roots: Vec<String>,
    pub selection: Vec<String>,
    pub dirty: bool,
    pub title: String,
    pub can_undo: bool,
    pub can_redo: bool,
    pub undo_label: Option<String>,
    pub redo_label: Option<String>,
}

// ── Details panel (reflection-driven, P3.3) ──────────────────────────────

/// A typed property value crossing to the Details panel. The `kind` tag selects
/// the widget; only the matching payload field is meaningful.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PropValueDto {
    Bool {
        value: bool,
    },
    Number {
        value: f64,
    },
    Text {
        value: String,
    },
    Vec3 {
        value: Vec<f64>,
    },
    Color {
        value: Vec<f32>,
    },
    Enum {
        value: String,
        options: Vec<String>,
    },
    // ── E-P1 deep editing ─────────────────────────────────────────────────
    /// A homogeneous list. The ListField edits it as a whole (add/remove/reorder
    /// or per-element) and writes the entire `value` back through set-property.
    List {
        value: Vec<PropValueDto>,
    },
    /// A nested struct — rendered as indented child rows. Each child's `name` is
    /// the relative field key; the frontend joins it onto the parent path
    /// (`parent.child`) when writing a leaf.
    Struct {
        fields: Vec<PropFieldDto>,
    },
    /// A reference to another entity by GUID string (`None` → unbound). Surfaced
    /// as an entity-picker widget.
    EntityRef {
        value: Option<String>,
    },
    /// A reference to an **asset** by GUID string (`None` → unbound), with the
    /// asset kind's slug so a picker can filter by it (P26.3b).
    ///
    /// **Read-only in this batch.** It exists so the Details panel can show the
    /// `Material.asset` binding scene v22 persists — a field the reflection
    /// walker cannot reach, because it is `#[reflect(ignore)]` exactly as
    /// `MeshRef::asset` is. The asset-**picker** widget is the standing gap, the
    /// same one `skel_merge_part`'s raw-GUID text box and the Model Editor's rig
    /// field have; binding is done by dragging a `.inf_mat` onto the entity,
    /// which is `scene_apply_material`.
    AssetRef {
        value: Option<String>,
        /// [`inf_asset::AssetKind::slug`] of what may be bound here, e.g.
        /// `"material"`. Carried now so the picker, when it lands, does not need
        /// a second source of truth about which kind a row accepts.
        ///
        /// Named `asset_kind` and not `kind`: this enum is serialized
        /// **internally tagged** on `kind`, and a variant field of that name is
        /// a compile error rather than a silent collision — one of the few
        /// places serde refuses to let a wire ambiguity through.
        asset_kind: String,
    },
}

/// One editable field row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct PropFieldDto {
    /// Reflect field key (write path).
    pub name: String,
    /// Display label.
    pub label: String,
    pub value: PropValueDto,
    /// Whether every selected object shares this value (multi-edit "—").
    pub same: bool,
}

/// One component section in the Details grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ComponentDto {
    pub type_path: String,
    pub display: String,
    pub fields: Vec<PropFieldDto>,
}

/// The Details panel view of the current selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct DetailsDto {
    pub selection: Vec<String>,
    /// Header label (the object name, or "N selected").
    pub name: String,
    /// Header type (single selection only).
    pub kind: String,
    /// Component sections shared by every selected object.
    pub components: Vec<ComponentDto>,
    pub multi: bool,
}

/// One entry in the "+ Add Component" menu (E-P1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct AddableComponentDto {
    /// Stable reflect type path — the key passed to `scene_add_component`.
    pub type_path: String,
    /// Human-readable menu label.
    pub display: String,
}

/// The kind of entity to create (`scene_create` command).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum SpawnKind {
    Empty,
    Cube,
    Sphere,
    Plane,
    Cylinder,
    Cone,
    DirectionalLight,
    PointLight,
    SpotLight,
    Camera,
    // ── 2D (P8.2b) ───────────────────────────────────────────────────────
    /// A textured sprite quad.
    Sprite,
    /// A chunked 2D tilemap (painted with the Tilemap panel).
    Tilemap,
    /// A bitmap-text label.
    Text2d,
    /// A 9-slice bordered panel.
    NineSlice,
    /// A 2D radial light.
    Light2d,
    // ── 3D terrain (P10) ──────────────────────────────────────────────────
    /// A heightfield terrain (starter sine-hill; sculpt/import tooling edits it).
    Terrain,
    // ── Gameplay volumes (E-P4) ───────────────────────────────────────────
    /// An overlap-sensing trigger region: a sensor box collider + `Volume`,
    /// invisible in PIE (no mesh) but outlined in the editor.
    TriggerVolume,
    /// A movement-blocking region: a solid box collider + `Volume`, invisible
    /// in PIE (no mesh) but outlined in the editor.
    BlockingVolume,
    // ── Utility (E-P5) ────────────────────────────────────────────────────
    /// A control-point spline (camera rail / patrol route / placement path):
    /// a default `Spline` component, drawn as a polyline in the editor viewport.
    Spline,
    // ── Utility (E-P6) ────────────────────────────────────────────────────
    /// A foliage scatter (grass/rocks/trees): a `Foliage` component seeded with a
    /// 1-entry palette, populated by the viewport's foliage brush.
    Foliage,
}

// ── Asset system / Content Drawer (Phase 4) ──────────────────────────────
//
// The asset database lives in `inf-editor-core::assets` over the project's
// content root. The Content Drawer consumes these GUID-keyed DTOs via the
// `assets_snapshot` command and re-fetches on the `assets://changed` event; a
// separate `assets://import` event streams import-job progress. Thumbnails come
// as data-URLs from `asset_thumbnail` (lazy, per visible cell).

/// One asset row in the Content Drawer grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct AssetDto {
    /// Stable GUID string.
    pub id: String,
    pub name: String,
    /// Kind slug ("mesh", "texture", …) for filtering/icons.
    pub kind: String,
    /// Kind display label ("Static Mesh", …).
    pub kind_label: String,
    /// Folder path relative to the content root ("", "meshes", "props/env").
    pub folder: String,
    /// Payload path relative to the content root (forward slashes).
    pub path: String,
    /// xxh3 content hash hex (the thumbnail + change key).
    pub content_hash: String,
    /// User tags.
    pub tags: Vec<String>,
    /// Import source (relative), if imported.
    pub source: Option<String>,
    /// Number of assets this one references (outgoing deps).
    #[ts(type = "number")]
    pub dep_count: u32,
    /// Number of assets that reference this one (for the delete warning badge).
    #[ts(type = "number")]
    pub ref_count: u32,
    /// Whether this kind has a rendered thumbnail (else the UI shows an icon).
    pub previewable: bool,
}

/// A folder in the content tree (`assets_snapshot`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct AssetFolderDto {
    /// Path relative to the content root ("" = root).
    pub path: String,
    /// Leaf folder name ("" for the root).
    pub name: String,
    /// Child folder paths.
    pub children: Vec<String>,
    /// Assets directly in this folder.
    #[ts(type = "number")]
    pub asset_count: u32,
}

/// The full content snapshot (`assets_snapshot` command).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct AssetSnapshot {
    /// Monotonic version (bumped on every content change).
    #[ts(type = "number")]
    pub version: u64,
    /// Absolute content-root path, for display.
    pub root: String,
    pub assets: Vec<AssetDto>,
    pub folders: Vec<AssetFolderDto>,
}

/// A lightweight reference to another asset (delete-warning referrer list).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct AssetRefDto {
    pub id: String,
    pub name: String,
    pub kind: String,
}

/// The result of a delete request (`asset_delete`): either it happened, or it
/// was blocked by the assets still referencing the target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct DeleteResult {
    pub deleted: bool,
    /// Non-empty only when `deleted` is false — the referrers to warn about.
    pub blockers: Vec<AssetRefDto>,
}

/// One import-progress event (`assets://import`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ImportEventDto {
    #[ts(type = "number")]
    pub job: u64,
    pub source: String,
    /// "started" | "progress" | "finished" | "failed".
    pub phase: String,
    /// GUIDs produced (on "finished").
    pub produced: Vec<String>,
    pub primary: Option<String>,
    pub cached: bool,
    pub error: Option<String>,
    /// Units of work completed / total (on "progress"; terrain imports report
    /// tiles written across every LOD level). `null` for jobs with no progress
    /// model.
    #[ts(type = "number | null")]
    pub done: Option<u64>,
    #[ts(type = "number | null")]
    pub total: Option<u64>,
    /// A short stage label on "progress" ("tiles", "lod2", …).
    pub stage: Option<String>,
}

/// The viewport's tool-state notice (`viewport://tool-status`, P16.4).
///
/// One channel for every half of the status seam, because they change for the
/// same reason and land in the same corner of the UI: `message` is the one-shot
/// rejection a tool raised (drained from the host's `take_tool_status`) and goes
/// to the status bar; the three booleans are standing facts, published only when
/// one of them changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ViewportToolStatusDto {
    pub message: Option<String>,
    /// The projected terrain pages from a `.inf_terrain` asset.
    pub terrain_streamed: bool,
    /// That asset is writable, so Sculpt/Paint may edit it (P16.4b). Only
    /// meaningful together with `terrain_streamed`: *streamed && !editable* is
    /// the one case the brush tools are disabled.
    pub terrain_editable: bool,
    /// The terrain carries tiles not yet written back to its asset — the
    /// toolbar's "unsaved terrain edits" chip and the save reminder.
    pub terrain_unsaved_edits: bool,
    /// Which viewport raised it (P23.2a) — appended, so every existing field
    /// keeps its place. **Stamped by Ring 2's event sink**
    /// (`stamp_tool_status`), which owns the id→handle map: the viewport thread
    /// builds this with an EMPTY id because it does not know its own key, so an
    /// empty string on the wire is a sink that forgot to stamp rather than a
    /// viewport with no name.
    pub viewport: String,
}

/// The `viewport://gizmo` payload (P23.2a): a gizmo-mode echo stamped with the
/// viewport that produced it.
///
/// A **wrapper** rather than a field, because [`GizmoModeDto`] is an enum and an
/// enum has no tail to add to. The channel used to carry the bare mode; the id
/// is what lets a second viewport echo its own W/E/R without moving the scene
/// viewport's toolbar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct ViewportGizmoDto {
    pub mode: GizmoModeDto,
    /// `"primary"` is the scene viewport.
    pub viewport: String,
}

/// What one `scene_save` actually accomplished (P16.4b).
///
/// A level save is no longer a single all-or-nothing act: the `.inf_lvl` and the
/// `.inf_terrain` assets its streamed terrains reference are **separate files**,
/// and one can land while another does not. Returning this instead of `()` is
/// what lets the shell say "the level saved, this terrain did not" rather than
/// reporting a clean save that quietly did less than the user asked for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct SaveResultDto {
    /// Absolute path of the `.inf_lvl` payload written.
    pub path: String,
    /// How many `.inf_terrain` assets were rewritten with sculpt/paint edits.
    pub terrain_assets_written: u32,
    /// How many level-0 tiles were folded into those assets.
    pub terrain_tiles_written: u32,
    /// One line per terrain whose edits could **not** be written. Empty on a
    /// fully clean save. These edits are still in memory, still marked unsaved,
    /// and are retried by the next save — nothing was lost, but nothing was
    /// persisted either, and the user has to be told.
    pub terrain_failures: Vec<String>,
    /// How many `.inf_voxel` assets were rewritten with carve edits (P21.2).
    pub voxel_assets_written: u32,
    /// How many chunks were folded into those assets.
    pub voxel_chunks_written: u32,
    /// Everything about this save that did **not** persist a cave, in two
    /// flavours that read the same way to an author (P21.2):
    ///
    /// * a volume whose `.inf_voxel` could not be written — the terrain-failure
    ///   twin, still in memory, still retried;
    /// * an **inline-terrain hole advisory**: this document is carrying cave
    ///   mouths on a terrain whose container cannot store them, so the save just
    ///   sealed them. The carve tools refuse to create that state, so it means
    ///   the document arrived in it; the line names the terrain and the fix.
    ///
    /// Separate from `terrain_failures` because they are separate assets with
    /// separate outcomes — a save may write every tile and no chunk.
    pub voxel_warnings: Vec<String>,
}

// ── Terrain import wizard (P16.4) ────────────────────────────────────────

/// What `terrain_probe_heightmap` read out of a heightmap's **header** — no
/// pixels were decoded, so this is instant even for a 16 k × 16 k source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct HeightmapProbeDto {
    /// Absolute source path (echoed so the wizard can keep one object).
    pub path: String,
    /// "PNG" | "EXR".
    pub format: String,
    pub width: u32,
    pub height: u32,
    /// Bits per sample of the source.
    pub bit_depth: u32,
    /// `true` when the source carries floats — i.e. when float-metres mode is
    /// offered.
    pub float_samples: bool,
    /// The channel the importer will read ("gray", "Y", …).
    pub channel: String,
    /// The settings the wizard opens with for this source.
    pub suggested: TerrainImportSettingsDto,
}

/// The Terrain Import wizard's settings block. Mirrors
/// `inf_editor_core::assets::TerrainImportSettings` and is persisted verbatim
/// into the asset's sidecar, so a reimport re-runs these exact choices.
///
/// **Metric**: `meters_per_sample`, `min_height` and `max_height` are SI metres
/// (units doctrine). The wizard's kilometre readback is a display division by
/// 1000 and never a stored scale factor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct TerrainImportSettingsDto {
    pub tile_resolution: u32,
    pub meters_per_sample: f64,
    pub min_height: f64,
    pub max_height: f64,
    /// Take the decoded float as absolute metres (float sources only).
    pub float_meters: bool,
    /// Straddle the world origin instead of growing into +X/+Z.
    pub center: bool,
    pub max_pyramid_levels: u32,
    #[ts(type = "number")]
    pub min_pyramid_tiles: usize,
}

/// The world a given source + settings pair will produce, recomputed on every
/// wizard edit so the extent readback is never stale.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct TerrainImportPlanDto {
    /// Real-world span in **metres**.
    pub extent_x_m: f64,
    pub extent_z_m: f64,
    /// Level-0 tiles across / down.
    pub tiles_x: u32,
    pub tiles_z: u32,
    /// Level-0 tiles in total.
    #[ts(type = "number")]
    pub tiles: u64,
}

impl TerrainImportSettingsDto {
    /// Convert to the Ring-1 settings the import job (and the sidecar) use.
    pub fn to_settings(&self) -> crate::assets::TerrainImportSettings {
        crate::assets::TerrainImportSettings {
            tile_resolution: self.tile_resolution,
            meters_per_sample: self.meters_per_sample,
            min_height: self.min_height,
            max_height: self.max_height,
            float_meters: self.float_meters,
            center: self.center,
            max_pyramid_levels: self.max_pyramid_levels,
            min_pyramid_tiles: self.min_pyramid_tiles,
        }
    }

    /// The DTO form of a settings block.
    pub fn from_settings(s: &crate::assets::TerrainImportSettings) -> Self {
        Self {
            tile_resolution: s.tile_resolution,
            meters_per_sample: s.meters_per_sample,
            min_height: s.min_height,
            max_height: s.max_height,
            float_meters: s.float_meters,
            center: s.center,
            max_pyramid_levels: s.max_pyramid_levels,
            min_pyramid_tiles: s.min_pyramid_tiles,
        }
    }
}

/// The finished state of a terrain import — what the wizard's "Add to Scene"
/// step acts on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct TerrainImportResultDto {
    /// The `.inf_terrain` asset GUID.
    pub asset: String,
    /// Display name (file stem of the produced asset).
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub tiles_x: u32,
    pub tiles_z: u32,
    #[ts(type = "number")]
    pub tiles: u64,
    pub lod_levels: u32,
    /// Real-world span in metres.
    pub extent_x_m: f64,
    pub extent_z_m: f64,
    /// Payload size on disk, in bytes.
    #[ts(type = "number")]
    pub bytes: u64,
}

/// The `assets://changed` event payload — a version bump prompting a re-fetch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct AssetChanged {
    #[ts(type = "number")]
    pub version: u64,
}

// ── Data assets (P4.5): struct / enum / table editors ────────────────────

/// One struct field / table column, flattened for the editor. `ty` is a slug
/// ("bool"/"int"/"float"/"text"/"vec3"/"color"/"asset_ref"/"enum"); an enum
/// field also carries the referenced `.inf_enum` GUID + name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct DataFieldDto {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub enum_id: Option<String>,
    pub enum_name: Option<String>,
}

/// A struct / enum / table asset, flattened for editing. Only the fields
/// relevant to `kind` are populated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct DataAssetDto {
    pub id: String,
    /// "struct" | "enum" | "table".
    pub kind: String,
    pub name: String,
    /// Struct fields or table columns.
    pub fields: Vec<DataFieldDto>,
    /// Enum variants.
    pub variants: Vec<String>,
    /// Table rows (cells as display strings, column-aligned).
    pub rows: Vec<Vec<String>>,
}

// ── Project system (P5.5) ────────────────────────────────────────────────

/// The currently-open project, as the editor start screen / title bar sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ProjectInfoDto {
    pub name: String,
    /// Absolute project root (forward-slashed).
    pub root: String,
    /// Content root relative to the project.
    pub content_dir: String,
    /// Levels root relative to the project.
    pub levels_dir: String,
    /// Template slug the project was scaffolded from.
    pub template: String,
}

/// One entry in the recent-projects list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct RecentProjectDto {
    pub name: String,
    pub path: String,
}

/// A first-run project template (New Project dialog).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct ProjectTemplateDto {
    pub slug: String,
    pub label: String,
    pub description: String,
}

// ── IDE: file explorer / git / search (P5.4) ─────────────────────────────

/// One entry in the project file tree (`list_project_files`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct FileEntryDto {
    /// Path relative to the walked root (forward-slashed).
    pub path: String,
    /// Leaf name.
    pub name: String,
    pub is_dir: bool,
}

/// One changed file in `git_status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct GitFileDto {
    /// Path relative to the repo (forward-slashed).
    pub path: String,
    /// Short status code ("M", "A", "D", "R", "?", …).
    pub status: String,
    /// Whether the change is in the index (staged).
    pub staged: bool,
}

/// The working-tree status (`git_status`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct GitStatusDto {
    /// False when `repo` is not a git repository.
    pub is_repo: bool,
    /// Current branch (or detached HEAD label).
    pub branch: String,
    /// Commits ahead of / behind the upstream.
    #[ts(type = "number")]
    pub ahead: u32,
    #[ts(type = "number")]
    pub behind: u32,
    pub files: Vec<GitFileDto>,
}

/// Options for `search_workspace`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct SearchOptsDto {
    pub regex: bool,
    pub case_sensitive: bool,
}

// ── Sprite-sheet slicing (P8.2a) ─────────────────────────────────────────
//
// A texture asset's slice model round-trips through `texture_get_slices` /
// `texture_set_slices`, which persist it into the texture's TOML sidecar. The
// Sprite Sheet panel draws the grid overlay live (computing UVs in JS from these
// pixel params) and applies a chosen slice to the selection via
// `scene_apply_sprite_slice`. All slicing values are in texture pixels.

/// Uniform-grid slicing parameters (texture pixels). `margin_*` is a single
/// top-left offset; `padding_*` is the inter-cell gap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct SpriteGridDto {
    pub columns: u32,
    pub rows: u32,
    pub margin_x: u32,
    pub margin_y: u32,
    pub padding_x: u32,
    pub padding_y: u32,
}

/// One named manual rectangle (texture pixels, top-left origin).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct SpriteRectDto {
    pub name: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// A texture's full slice model plus its pixel dimensions (`texture_get_slices`;
/// also the payload of `texture_set_slices`, where the dimensions are ignored).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct SpriteSheetDto {
    /// The texture asset GUID this model belongs to.
    pub texture_id: String,
    /// Grid definition, or `None` when the sheet is manual-only.
    pub grid: Option<SpriteGridDto>,
    /// Named manual rectangles.
    pub manual: Vec<SpriteRectDto>,
    /// Texture width/height in pixels (read from the payload; drives the overlay).
    pub tex_width: u32,
    pub tex_height: u32,
}

// ── Sorting layers (P8.2a) ───────────────────────────────────────────────

/// One named sorting layer (`layers_get` / `layers_set`). `id` is the raw `i32`
/// the `Sprite.sorting_layer` field stores.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct SortingLayerDto {
    pub id: i32,
    pub name: String,
}

// ── Collision layers (P12.1) ─────────────────────────────────────────────

/// One named collision layer (`collision_layers_get` / `collision_layers_set`).
/// `bit` is the layer index in `0..32`; the `Collider*` masks store `1 << bit`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct CollisionLayerDto {
    pub bit: u8,
    pub name: String,
}

// ── Viewport mode + 2D snapping (P8.2c) ──────────────────────────────────

/// Active viewport projection: perspective 3D or orthographic 2D editing
/// (`viewport_set_mode`). Serializes as the tag string `"Perspective"`/`"TwoD"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum ViewportModeDto {
    Perspective,
    TwoD,
}

/// Viewport shading view mode (R-P2; `viewport_set_view_mode`). Serializes as the
/// tag string `"Lit"`/`"Unlit"`/`"Wireframe"`/`"Biomes"`/`"VtResidency"`.
/// `Wireframe` degrades to `Unlit` in the renderer when the adapter lacks
/// `POLYGON_MODE_LINE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum ViewModeDto {
    Lit,
    Unlit,
    Wireframe,
    /// Terrain tinted by its per-sample biome id, everything else unlit (P19.2).
    /// Needs no GPU feature, so it never degrades.
    Biomes,
    /// Every virtual-textured surface painted by how far behind the streamer is
    /// at that pixel — green resident, red at the analytic floor, grey unbound
    /// (P26.5). Everything else renders unlit, on the `Biomes` precedent. Needs
    /// no GPU feature; a level with no virtual textures paints uniformly grey,
    /// which is the answer rather than a failure.
    VtResidency,
}

/// 2D-mode snapping configuration pushed from the viewport toolbar
/// (`viewport_set_snap2d`). Grid snap quantizes a translate to `grid_size` world
/// units; **pixel snap** (finer) to `1/pixels_per_unit`. Pixel snap wins when
/// both are enabled.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct Snap2DDto {
    pub grid_enabled: bool,
    pub grid_size: f32,
    pub pixel_enabled: bool,
    pub pixels_per_unit: f32,
}

// ── Transform gizmo: mode + space + 3D snap (Wave 2) ─────────────────────
//
// The transform gizmo mode (translate/rotate/scale) is two-way synced with the
// native viewport: `viewport_set_gizmo_mode` pushes toolbar/keyboard changes in,
// and the viewport emits the current mode back on `viewport://gizmo` (a W/E/R
// keypress over the viewport updates the toolbar). The gizmo space toggle
// (`viewport_set_gizmo_space`) and the 3D snap increments
// (`viewport_set_snap3d`) are one-way pushes from the toolbar.

/// Transform-gizmo mode (`viewport_set_gizmo_mode`; also the `viewport://gizmo`
/// event payload). Serializes as the tag string `"Translate"`/`"Rotate"`/
/// `"Scale"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum GizmoModeDto {
    Translate,
    Rotate,
    Scale,
}

/// Gizmo orientation frame (`viewport_set_gizmo_space`). `World` aligns handles
/// to the world axes; `Local` aligns them to the selection's own rotation.
/// Serializes as the tag string `"World"`/`"Local"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum GizmoSpaceDto {
    World,
    Local,
}

/// 3D transform-gizmo snap increments (`viewport_set_snap3d`). `translate` is
/// world metres, `rotate_deg` is degrees, `scale` is a ratio step. When
/// `always_on` is false, snapping is Shift-gated (matching the pre-Wave-2 feel).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct Snap3DDto {
    pub translate: f32,
    pub rotate_deg: f32,
    pub scale: f32,
    pub always_on: bool,
}

// ── Terrain sculpt tool (P10.2b) ─────────────────────────────────────────
//
// The viewport toolbar switches between the Select tool (pick/gizmo) and the
// Sculpt tool (terrain height brush) via `viewport_set_tool_mode`, and pushes
// the brush configuration via `viewport_set_sculpt`. Sculpting is a
// perspective-only tool; 2D mode stays on Select.

/// Active viewport tool (`viewport_set_tool_mode`). Serializes as the tag string
/// `"Select"` / `"Sculpt"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum ToolModeDto {
    Select,
    Sculpt,
    /// Scatter foliage onto the terrain under a brush (E-P6). Perspective-only.
    Foliage,
    /// Paint per-sample biome ids onto the terrain under a brush (P19.2).
    /// Perspective-only.
    Biome,
    /// Place rivers and lakes against the terrain (P20.4). Perspective-only.
    ///
    /// One tool mode with two sub-modes ([`WaterToolKindDto`]) rather than two
    /// modes, because they share everything that matters: the same terrain pick,
    /// the same "which body am I editing" state and the same biome-hint
    /// defaults. The Sculpt/Paint precedent, not the Sculpt/Biome one.
    Water,
    /// Carve (and fill) a voxel volume — caves, tunnels, excavations (P21.2).
    /// Perspective-only. Two sub-modes in [`VoxelSettingsDto::kind`].
    Voxel,
}

// ── Water placement (P20.4) ──────────────────────────────────────────────

/// Which water body the [`ToolModeDto::Water`] tool places. Serializes as its tag
/// string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum WaterToolKindDto {
    /// Click to append a control point to the active river's centreline; the
    /// first click on empty space starts a new one.
    River,
    /// Drag a rectangle; the still-water level comes from the ground under the
    /// first corner (or the biome's `water_hint`, when it has one).
    Lake,
}

/// Water tool configuration pushed from the viewport toolbar
/// (`viewport_set_water`).
///
/// Units are SI throughout (architecture rule 6): metres and m/s. `level_offset_m`
/// is added to whatever level the defaults suggest, so "a lake 2 m above the
/// ground I clicked" needs no arithmetic from the author.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct WaterSettingsDto {
    pub kind: WaterToolKindDto,
    /// Full width of a new river, metres.
    pub width_m: f64,
    /// Depth to the bed of a new river, metres.
    pub depth_m: f64,
    /// Surface flow speed of a new river, m/s. Negative reverses it.
    pub flow_m_s: f64,
    /// Added to the suggested still-water level, metres.
    pub level_offset_m: f64,
}

impl Default for WaterSettingsDto {
    fn default() -> Self {
        Self {
            kind: WaterToolKindDto::River,
            width_m: 8.0,
            depth_m: 1.5,
            flow_m_s: 1.5,
            level_offset_m: 0.0,
        }
    }
}

/// What the water tool suggests for a click at a world point (`water_defaults`) —
/// the P19.2 `BiomeDef::water_hint`'s first reader.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct WaterDefaultsDto {
    /// Suggested still-water level, metres of world Y.
    pub level_m: f64,
    /// Suggested river width / depth, metres.
    pub river_width_m: f64,
    pub river_depth_m: f64,
    /// Terrain height under the point, or `null` where there is no ground.
    pub ground_m: Option<f64>,
    /// The painted biome id (`0` = unassigned) and its name (empty when none).
    pub biome_id: u8,
    pub biome_name: String,
    /// Whether `level_m` came from the biome's hint rather than from the ground.
    /// The toolbar says which, because "why is my lake at 6.5 m?" should have a
    /// visible answer.
    pub from_biome_hint: bool,
}

/// Where a still-water level lands on the terrain inside a rectangle
/// (`water_lake_preview`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct LakePreviewDto {
    pub level_m: f64,
    /// Fraction of the *known* samples at or below the level, `[0, 1]`.
    pub covered_fraction: f64,
    pub max_depth_m: f64,
    pub mean_depth_m: f64,
    /// Grid samples taken, and how many the terrain answered for. `known == 0`
    /// means "there is no ground here", which is a different statement from
    /// "the lake is empty".
    pub samples: u32,
    pub known: u32,
    /// The waterline as flat `[x0, z0, x1, z1]` world-XZ segments. Flat rather
    /// than nested so the JSON stays one array of numbers for a few thousand
    /// segments.
    pub waterline: Vec<f64>,
}

/// One stretch of a river that climbs, as the tool reports it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct RiverClimbDto {
    pub from_s: f64,
    pub to_s: f64,
    pub rise_m: f64,
    /// Rise over run, dimensionless.
    pub gradient: f64,
}

/// One stretch where a river disagrees with the ground under it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct RiverBedConflictDto {
    /// `"buried"` (terrain above the surface) or `"perched"` (terrain below the
    /// authored bed) — `BedIssue::id()`, a stable id the frontend switches on.
    pub issue: String,
    pub from_s: f64,
    pub to_s: f64,
    pub worst_m: f64,
    pub worst_x: f64,
    pub worst_z: f64,
}

/// The river tool's verdict on one river (`water_river_report`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct RiverReportDto {
    /// The river entity's GUID.
    pub entity: String,
    pub length_m: f64,
    pub points: usize,
    /// Source elevation minus mouth elevation, read the way the water flows.
    pub fall_m: f64,
    /// The two **cook** advisories, re-run here so the tool says what the build
    /// will say.
    pub surface_climbs: Vec<RiverClimbDto>,
    pub bed_climbs: Vec<RiverClimbDto>,
    /// The terrain-aware conflicts, which only the editor can produce.
    pub bed_conflicts: Vec<RiverBedConflictDto>,
    /// Frames the terrain answered for, out of the total — a report over 3 of 200
    /// frames is not a clean bill of health.
    pub sampled_frames: usize,
    pub total_frames: usize,
}

// ── Voxel carve tools (P21.2) ────────────────────────────────────────────
//
// The [`ToolModeDto::Voxel`] tool cuts a `VoxelVolume`'s SDF chunks and — where
// the cut breaks the heightfield surface — opens the terrain above it, as ONE
// undo step. A surface-crossing cut over an INLINE terrain is refused outright,
// because schema v19 pins a level's tiles at a layout with no hole mask; the
// refusal arrives on `viewport://tool-status` like every other tool verdict.

/// Which cut the [`ToolModeDto::Voxel`] tool makes. Serializes as its tag string.
///
/// Two sub-modes inside one tool mode rather than two modes — the water tool's
/// precedent — because they share the volume resolution, the surface-crossing
/// verdict, the carve/fill switch and the material. Only how the author
/// describes the path differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum VoxelToolKindDto {
    /// Drag to lay sphere dabs along the stroke, spaced by arc length so drag
    /// speed cannot change what is dug.
    Brush,
    /// Click waypoints; Ctrl+click closes the path and tube-carves the whole
    /// thing as one undo step.
    Tunnel,
    /// Press-drag a rectangle on the surface; release excavates it to
    /// `depth_m` below grade (P21.3) — the foundation pit.
    BoxCut,
    /// Click waypoints; Ctrl+click cuts a swept **rectangular** trench along
    /// them (P21.3) — the utility trench / road cut.
    Trench,
}

/// Where a dig puts the material it removes (P21.3). Serializes as its tag
/// string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum SpoilModeDto {
    /// Removed and discarded — the P21.2 behaviour, and the right one for a
    /// cave.
    Off,
    /// Piled at the deterministic default site: east of the cut, clear of its
    /// rim, on the ground there.
    Auto,
    /// Piled where the author picked in the viewport (see
    /// [`VoxelSettingsDto::pick_spoil_site`]); falls back to `Auto` until a site
    /// has been picked.
    Site,
}

/// Carve or fill — which way a voxel cut runs. Serializes as its tag string.
///
/// Not a boolean: the Ring-0 ops it maps onto are named `Carve` and `Fill`, and
/// a `carve: bool` crossing three layers of IPC is three chances to invert it
/// silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum VoxelOpModeDto {
    /// Remove material — and open the heightfield above it.
    Carve,
    /// Add material — and close the heightfield above it.
    Fill,
}

/// Voxel-tool configuration pushed from the viewport toolbar
/// (`viewport_set_voxel`).
///
/// SI throughout (architecture rule 6): both lengths are world **metres**.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct VoxelSettingsDto {
    pub kind: VoxelToolKindDto,
    /// Cut radius — the sphere of a brush dab, the tube radius of a tunnel —
    /// metres. One slider, one meaning in both sub-modes.
    pub radius_m: f64,
    /// How far below the picked surface the cut's centre sits, metres.
    ///
    /// This is what makes a tunnel a tunnel. At `0` the cut breaks the ground
    /// where the author points (a cave mouth); past the radius it hollows rock
    /// with no mouth, which is legal on any terrain because no hole is needed.
    pub depth_m: f64,
    pub mode: VoxelOpModeDto,
    /// The splat index a **fill** paints; ignored by a carve (an emptied voxel
    /// carries no material). A voxel material index IS a terrain splat index,
    /// which is what makes a cave wall shade like the hillside it opens out of.
    pub material: u8,
    /// **Dig to grade** (P21.3): a brush dab becomes a column from `depth_m`
    /// below the surface up to daylight instead of a ball at depth, so a
    /// freehand stroke leaves an open cut rather than buried bubbles. Ignored by
    /// the other three sub-modes, which are open to the sky by construction.
    pub dig_to_depth: bool,
    /// Where the excavated material goes.
    pub spoil: SpoilModeDto,
    /// While `true`, a viewport click **moves the spoil site** instead of
    /// digging — a sticky mode the toolbar turns off again, not a one-shot arm.
    pub pick_spoil_site: bool,
}

/// What the Voxel tool can and cannot do in the **open level** — the toolbar's
/// live verdict readout (`viewport_voxel_status`, P21.2).
///
/// The river verdict's shape, and for the same reason: a carve is a commit of
/// geometry, and an author who has just had one refused needs to know *why* and
/// *what to change* without reproducing the gesture. Everything here is
/// camera-independent, so it can be answered without a pick — which is what lets
/// the toolbar show it before the first click rather than after the first
/// refusal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct VoxelStatusDto {
    /// `VoxelVolume` entities in the document. Zero means there is nothing to
    /// carve at all — a different problem from a carve that would be refused,
    /// and the readout must not conflate them.
    pub volumes: u32,
    /// …of those, how many resolve to a loaded `.inf_voxel`. A volume whose
    /// asset reference is empty or unresolvable binds nothing, and a carve into
    /// it changes nothing while reporting success.
    pub bound_volumes: u32,
    /// Terrains that **can** persist a cave mouth (asset-backed).
    pub asset_backed_terrains: u32,
    /// Terrains that cannot — inline in the `.inf_lvl`, whose tiles schema v19
    /// pins at a layout with no hole mask. Names rather than GUIDs, and a list
    /// rather than a count, because in a multi-terrain world the author has to
    /// know *which* one to convert.
    pub inline_terrains: Vec<String>,
    /// Chunks carved since the last save. Carve edits live in the `.inf_voxel`
    /// and are written by an explicit save only — autosave does not touch assets
    /// — so this is the reminder that Ctrl+S is owed.
    pub unsaved_chunks: u32,
    /// The refusal a surface-crossing carve would hit, **verbatim** the sentence
    /// the viewport puts on `viewport://tool-status`
    /// (`INLINE_TERRAIN_CARVE_REFUSAL`). `None` when every terrain is
    /// asset-backed. One string, quoted in both places, so the toolbar cannot
    /// explain the refusal differently from the tool that raised it.
    pub refusal: Option<String>,
    /// The defensive advisories: this document is **already** carrying mouths it
    /// cannot save. Empty for every level the carve tools produced, and for every
    /// level written before P21.2.
    pub advisories: Vec<String>,
}

impl Default for VoxelSettingsDto {
    fn default() -> Self {
        Self {
            kind: VoxelToolKindDto::Brush,
            radius_m: 2.0,
            depth_m: 0.0,
            mode: VoxelOpModeDto::Carve,
            material: 0,
            dig_to_depth: false,
            spoil: SpoilModeDto::Off,
            pick_spoil_site: false,
        }
    }
}

/// The sculpt brush operation. Serializes as its tag string. `Paint` is the
/// P10.4 splat sub-mode (edits layer weights, targeting `paint_layer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum SculptOpDto {
    Raise,
    Lower,
    Smooth,
    Flatten,
    Noise,
    Paint,
}

/// The sculpt brush falloff curve. Serializes as its tag string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum SculptFalloffDto {
    Smooth,
    Linear,
    Sphere,
    Sharp,
}

/// Sculpt brush configuration pushed from the viewport toolbar
/// (`viewport_set_sculpt`). `radius` is world metres; `strength` is per-dab
/// metres at full weight for Raise/Lower/Noise, or a `[0,1]` blend fraction for
/// Smooth/Flatten.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct SculptSettingsDto {
    pub op: SculptOpDto,
    pub radius: f64,
    pub strength: f64,
    pub falloff: SculptFalloffDto,
    /// Target splat layer `0..=3` for the `Paint` op (P10.4). Ignored by the
    /// height ops.
    pub paint_layer: u8,
}

// ── Biome brush + biome sets (P19.2) ─────────────────────────────────────
//
// The Biome tool paints per-sample biome **ids** onto a terrain. The ids name
// entries in a `.inf_biomes` `BiomeSet` bound to the terrain entity, so the
// toolbar needs both the brush push and a read of the terrain's current
// vocabulary (`terrain_biomes`) to draw its swatches.

/// Biome-brush configuration (`viewport_set_biome`). `radius` is world metres.
///
/// `strength` is **not** a blend fraction — a biome id is categorical, so the
/// brush writes a hard boundary and `strength` selects which falloff contour that
/// boundary lands on (`1` stamps the whole disk). `biome` is the id painted; `0`
/// is the reserved *unassigned* value and erases.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct BiomeSettingsDto {
    pub radius: f64,
    pub strength: f64,
    pub falloff: SculptFalloffDto,
    pub biome: u8,
}

/// One biome definition inside a [`BiomeSetDto`].
///
/// `color` is **linear** RGBA (it is uploaded straight into the overlay palette).
/// `pcg_graph` is a GUID string (the P19.3 hook); `water_hint` is a still-water
/// level in **metres** of absolute world height; `structure_hint` names a
/// building palette (P19.5). Both hints are plain advisory data in P19.2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct BiomeDefDto {
    /// Stable id `1..=255`. `0` is reserved for *unassigned* and cannot be
    /// defined — the backend rejects a set that tries.
    pub id: u8,
    pub name: String,
    pub color: [f32; 4],
    /// Which of the terrain's four splat layers this biome shades as, if any.
    pub splat_layer: Option<u8>,
    pub pcg_graph: Option<String>,
    pub water_hint: Option<f32>,
    pub structure_hint: Option<String>,
}

/// A `.inf_biomes` asset as the inline editor sees it (`asset_get_biome_set` /
/// `asset_save_biome_set`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct BiomeSetDto {
    /// The asset GUID.
    pub id: String,
    /// Display name of the set.
    pub name: String,
    pub biomes: Vec<BiomeDefDto>,
}

/// The biome vocabulary the viewport toolbar paints with (`terrain_biomes`).
///
/// Resolved for **one terrain entity**: which set is bound (if any) and the
/// definitions it holds. `biomes` is empty when nothing is bound, which is
/// exactly when the tool has nothing to offer — the toolbar shows the picker and
/// a "bind a set" affordance rather than an empty swatch row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct TerrainBiomesDto {
    /// The terrain entity GUID these biomes belong to.
    pub entity: String,
    /// The bound `.inf_biomes` GUID, or `null`.
    pub biome_set: Option<String>,
    /// Display name of the bound set (empty when unbound).
    pub biome_set_name: String,
    pub biomes: Vec<BiomeDefDto>,
    /// Every `.inf_biomes` in the project as `[guid, name]`, for the picker.
    /// Sorted by name — deterministic, so the dropdown never reorders itself.
    pub available: Vec<(String, String)>,
}

// ── Foliage brush (E-P6) ─────────────────────────────────────────────────
//
// The viewport toolbar's Foliage tool scatters (or erases) `Foliage` instances
// onto the terrain under an LMB-drag brush, pushed via `viewport_set_foliage`.
// Perspective-only, like the sculpt tool.

/// Foliage-brush configuration (`viewport_set_foliage`). `radius` is world
/// metres; `density` is target instances per m² of brush area; `kind` selects the
/// palette slot; `scale_jitter` is the ± fractional scale spread; `seed` makes a
/// stroke's scatter deterministically reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct FoliageSettingsDto {
    pub radius: f64,
    pub density: f64,
    pub erase: bool,
    pub kind: u32,
    pub scale_jitter: f64,
    pub align_to_normal: bool,
    pub seed: u32,
}

// ── Terrain erosion bake (P10.3b) ────────────────────────────────────────
//
// The Erode dialog sends an `ErosionParamsDto` + step count to `terrain_erode`,
// which runs the GPU compute pipeline (CPU reference fallback with no adapter)
// and commits ONE undoable height delta. Mirrors `inf_terrain::ErosionParams`
// field-for-field (`rain_seed` narrowed to `u32` to stay a JS `number`).

/// Hydraulic + thermal erosion parameters (`terrain_erode`). Defaults come from
/// [`inf_terrain::ErosionParams::default`]. Rates are per simulation step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct ErosionParamsDto {
    pub dt: f32,
    pub rain_rate: f32,
    pub rain_variation: f32,
    pub rain_seed: u32,
    pub rain_frequency: f64,
    pub rain_octaves: u32,
    pub evaporation: f32,
    pub gravity: f32,
    pub pipe_area: f32,
    pub sediment_capacity: f32,
    pub dissolving: f32,
    pub deposition: f32,
    pub min_tilt: f32,
    pub max_erode_depth: f32,
    pub thermal_talus_deg: f32,
    pub thermal_rate: f32,
    pub max_thermal_depth: f32,
    pub thermal_every: u32,
}

impl ErosionParamsDto {
    /// Convert to the Ring-0 [`inf_terrain::ErosionParams`].
    pub fn to_params(&self) -> inf_terrain::ErosionParams {
        inf_terrain::ErosionParams {
            dt: self.dt,
            rain_rate: self.rain_rate,
            rain_variation: self.rain_variation,
            rain_seed: self.rain_seed as u64,
            rain_frequency: self.rain_frequency,
            rain_octaves: self.rain_octaves,
            evaporation: self.evaporation,
            gravity: self.gravity,
            pipe_area: self.pipe_area,
            sediment_capacity: self.sediment_capacity,
            dissolving: self.dissolving,
            deposition: self.deposition,
            min_tilt: self.min_tilt,
            max_erode_depth: self.max_erode_depth,
            thermal_talus_deg: self.thermal_talus_deg,
            thermal_rate: self.thermal_rate,
            max_thermal_depth: self.max_thermal_depth,
            thermal_every: self.thermal_every,
        }
    }
}

impl Default for ErosionParamsDto {
    fn default() -> Self {
        let p = inf_terrain::ErosionParams::default();
        Self {
            dt: p.dt,
            rain_rate: p.rain_rate,
            rain_variation: p.rain_variation,
            rain_seed: p.rain_seed as u32,
            rain_frequency: p.rain_frequency,
            rain_octaves: p.rain_octaves,
            evaporation: p.evaporation,
            gravity: p.gravity,
            pipe_area: p.pipe_area,
            sediment_capacity: p.sediment_capacity,
            dissolving: p.dissolving,
            deposition: p.deposition,
            min_tilt: p.min_tilt,
            max_erode_depth: p.max_erode_depth,
            thermal_talus_deg: p.thermal_talus_deg,
            thermal_rate: p.thermal_rate,
            max_thermal_depth: p.max_thermal_depth,
            thermal_every: p.thermal_every,
        }
    }
}

/// Result of an erosion bake (`terrain_erode`). `mass_delta` is the net terrain
/// volume change (world m³, negative = net-eroded), derived from the committed
/// delta so it is adapter-independent. `sediment_moved` is the CPU reference's
/// cumulative eroded volume — present only on the CPU (no-adapter) path, `None`
/// on the GPU path (GPU stat reductions are omitted; see the executor docs).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct ErosionReportDto {
    pub cells_changed: u32,
    /// Data-map samples the bake moved (P19.1) — the flow / deposition / wear
    /// accumulators it wrote. Always **at least** `cells_changed`: a height only
    /// moves through the erode/deposit pass, which writes a map in the same
    /// breath.
    pub map_cells_changed: u32,
    pub mass_delta: f64,
    pub sediment_moved: Option<f64>,
    pub used_gpu: bool,
    pub steps: u32,
}

/// Result of a data-map export (`terrain_export_data_map`, P19.1).
///
/// The written file is a **16-bit grayscale PNG** normalized over `[min, max]` —
/// the range the terrain's raw accumulators actually span, reported here so the
/// mapping is stated rather than implied. The stored data is never normalized;
/// only this view is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct DataMapExportDto {
    /// Which map was exported (`"flow"` / `"deposition"` / `"wear"`).
    pub map: String,
    /// Absolute path of the PNG written.
    pub path: String,
    pub width: u32,
    pub height: u32,
    /// Bytes written.
    pub bytes: u32,
    /// Low end of the exported range (black), in the map's own SI unit.
    pub min: f32,
    /// High end of the exported range (white), in the map's own SI unit.
    pub max: f32,
    /// SI unit of the accumulator (`"m^3"` for flow, `"m"` for the others).
    pub unit: String,
}

/// Per-project editor settings persisted under `<root>/.infinity/settings.toml`
/// (`project_settings_get` / `project_settings_set`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct ProjectSettingsDto {
    /// Pixels-per-unit for 2D pixel snapping (default 100).
    pub pixels_per_unit: f32,
}

// ── Tilemap painting (P8.2b) ─────────────────────────────────────────────
//
// The Tilemap panel reads the selected entity's tilemap via `tilemap_get` and
// paints strokes back via `tilemap_paint` (one undo step per stroke). Tile
// indices are 1-based atlas cells (`0` = empty). Coordinates are signed grid
// cells; the chunked storage means the addressable range is unbounded.

/// One painted tile cell (`tilemap_get` output row / `tilemap_paint` input).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct TilemapCellDto {
    pub x: i32,
    pub y: i32,
    /// 1-based atlas index; `0` erases.
    #[ts(type = "number")]
    pub tile: u32,
}

/// The selected entity's tilemap, projected for the paint panel (`tilemap_get`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct TilemapDto {
    /// Owning entity GUID.
    pub entity: String,
    /// Atlas texture asset GUID, if one is assigned.
    pub texture: Option<String>,
    /// World units per tile cell.
    pub tile_width: f64,
    pub tile_height: f64,
    /// Atlas grid the 1-based tile index maps into (tile→UV mapping).
    #[ts(type = "number")]
    pub atlas_cols: u32,
    #[ts(type = "number")]
    pub atlas_rows: u32,
    /// Palette dimensions: the texture's P8.2a grid slicing when present, else
    /// the atlas grid. The palette shows `palette_cols * palette_rows` numbered
    /// swatches (indices `1..=count`).
    #[ts(type = "number")]
    pub palette_cols: u32,
    #[ts(type = "number")]
    pub palette_rows: u32,
    /// Painted cells only (empty cells omitted), in deterministic chunk order.
    pub cells: Vec<TilemapCellDto>,
}

// ── Cook / Package (P9.2 item 3) ─────────────────────────────────────────
//
// The Build ▸ Package Project… dialog runs `inf_packager::cook` against the open
// project on a blocking task (`project_package` command) and renders these
// projections of `inf_packager::CookReport`. A cook failure rejects the command
// with a structured `PackageErrorDto` instead of an opaque string so the dialog
// can anchor blueprint failures to their class + handler. Start/finish is also
// broadcast on the `package://state` event (a boolean running flag) for any
// global listener. Per-stage progress is a documented follow-up — the `cook`
// API exposes no progress callback yet, so we do not fake it.

/// One per-kind asset count in a cook report (`PackageResultDto.kinds`). `kind`
/// is the asset-kind slug (`"mesh"`, `"level"`, `"blueprint"`, …).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct PackageKindCountDto {
    pub kind: String,
    #[ts(type = "number")]
    pub count: u32,
}

/// A successful cook, projected for the Package dialog (`project_package`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct PackageResultDto {
    pub project_name: String,
    pub engine_version: String,
    /// Output directory the build was written to (forward-slashed).
    pub out_dir: String,
    /// The written pack file (forward-slashed).
    pub pack_path: String,
    /// The written manifest file (forward-slashed).
    pub manifest_path: String,
    /// Total assets packed.
    #[ts(type = "number")]
    pub asset_count: u32,
    /// Per-kind counts, sorted by slug.
    pub kinds: Vec<PackageKindCountDto>,
    /// Size of the written pack in bytes.
    #[ts(type = "number")]
    pub pack_bytes: u64,
    /// Level GUIDs in the pack (sorted).
    pub levels: Vec<String>,
    /// The primary/boot level GUID (lowest), if any.
    pub root_level: Option<String>,
    /// How many blueprint assets were validated.
    #[ts(type = "number")]
    pub blueprints_validated: u32,
    /// How many levels were rewritten to the runtime schema.
    #[ts(type = "number")]
    pub levels_rewritten: u32,
    /// Non-fatal advisories (e.g. "no levels").
    pub warnings: Vec<String>,
}

/// A structured cook failure (the `Err` payload of `project_package`). `class`
/// is the error category slug; blueprint failures additionally carry the
/// blueprint class name (`blueprint_class`) and the handler/function where the
/// problem lives (`handler`) so the dialog can anchor the error. `guid` is the
/// offending asset GUID when one is known.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct PackageErrorDto {
    /// Error category: `"no_project"`, `"blueprint"`, `"scene"`, `"unknown_root"`,
    /// `"bad_root"`, `"io"`, `"project"`, `"asset"`, `"manifest"`, `"internal"`.
    pub class: String,
    /// Human-readable message.
    pub message: String,
    /// The failing blueprint's class name (blueprint failures only).
    pub blueprint_class: Option<String>,
    /// The handler/function anchor (blueprint failures only).
    pub handler: Option<String>,
    /// The offending asset GUID, when known.
    pub guid: Option<String>,
}

/// One `search_workspace` hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SearchHitDto {
    /// Path relative to the searched root (forward-slashed).
    pub path: String,
    /// 1-based line number.
    #[ts(type = "number")]
    pub line: u32,
    /// 1-based column of the match start.
    #[ts(type = "number")]
    pub column: u32,
    /// The matching line's text (trimmed to a sane length).
    pub text: String,
}

// ── Sequencer (P11.4) ────────────────────────────────────────────────────────

/// Keyframe interpolation for a sequencer track segment (mirrors
/// [`crate::sequencer::Interp`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum SeqInterpDto {
    Step,
    Linear,
}

/// One keyframe on a scalar track: a time (seconds), a scalar value, and the
/// interpolation governing the segment starting here.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct SeqKeyDto {
    pub t: f64,
    pub value: f64,
    pub interp: SeqInterpDto,
}

/// One scalar property track: the animated entity guid, the reflection scalar
/// path (`"Transform.translation.x"`), and its sorted keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SeqTrackDto {
    /// Target entity guid (string form).
    pub target: String,
    /// Reflection scalar path.
    pub path: String,
    pub keys: Vec<SeqKeyDto>,
}

/// A sequencer timeline (the panel's document view; mirrors
/// [`crate::sequencer::Sequence`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct SequenceDto {
    pub name: String,
    pub duration: f64,
    /// Authoring frame rate hint — the retime/snap grid is `1 / fps_hint`.
    #[ts(type = "number")]
    pub fps_hint: u32,
    pub tracks: Vec<SeqTrackDto>,
}

// ── World settings (R-P4 · schema v8) ────────────────────────────────────────
//
// The editable World Settings panel reads the level's [`LevelSettings`] via
// `scene_get_settings` and writes it back (debounced) via `scene_set_settings`
// (one undo step). A flat, fully-explicit DTO mirroring
// [`crate::scene::serialize::LevelSettings`]; the nested `render` object mirrors
// [`crate::scene::serialize::RenderSettingsRecord`] field-for-field. Gravity
// vectors cross as fixed `[f64; N]` arrays (→ `[number, number]` in TS).

/// The persisted renderer HDR / post / lighting settings, as the World Settings
/// panel edits them (`scene_get_settings` / `scene_set_settings`). Mirrors
/// [`crate::scene::serialize::RenderSettingsRecord`] field-for-field.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct RenderSettingsRecordDto {
    pub exposure: f32,
    pub dither: bool,
    pub bloom_enabled: bool,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    pub bloom_intensity: f32,
    pub ssao_enabled: bool,
    pub ssao_radius: f32,
    pub ssao_intensity: f32,
    pub ssao_bias: f32,
    pub taa: bool,
    pub shadows_enabled: bool,
    pub shadows_max_distance: f32,
    pub gi_enabled: bool,
    pub gi_intensity: f32,
}

impl RenderSettingsRecordDto {
    fn from_record(r: &crate::scene::serialize::RenderSettingsRecord) -> Self {
        Self {
            exposure: r.exposure,
            dither: r.dither,
            bloom_enabled: r.bloom_enabled,
            bloom_threshold: r.bloom_threshold,
            bloom_knee: r.bloom_knee,
            bloom_intensity: r.bloom_intensity,
            ssao_enabled: r.ssao_enabled,
            ssao_radius: r.ssao_radius,
            ssao_intensity: r.ssao_intensity,
            ssao_bias: r.ssao_bias,
            taa: r.taa,
            shadows_enabled: r.shadows_enabled,
            shadows_max_distance: r.shadows_max_distance,
            gi_enabled: r.gi_enabled,
            gi_intensity: r.gi_intensity,
        }
    }

    fn to_record(self) -> crate::scene::serialize::RenderSettingsRecord {
        crate::scene::serialize::RenderSettingsRecord {
            exposure: self.exposure,
            dither: self.dither,
            bloom_enabled: self.bloom_enabled,
            bloom_threshold: self.bloom_threshold,
            bloom_knee: self.bloom_knee,
            bloom_intensity: self.bloom_intensity,
            ssao_enabled: self.ssao_enabled,
            ssao_radius: self.ssao_radius,
            ssao_intensity: self.ssao_intensity,
            ssao_bias: self.ssao_bias,
            taa: self.taa,
            shadows_enabled: self.shadows_enabled,
            shadows_max_distance: self.shadows_max_distance,
            gi_enabled: self.gi_enabled,
            gi_intensity: self.gi_intensity,
        }
    }
}

/// World-partition settings, as the World Settings panel edits them (P16.5).
/// Mirrors [`crate::scene::serialize::PartitionSettings`].
///
/// **The editor stays single-document.** Turning `enabled` on does not partition
/// the open level in the editor — nothing streams while authoring. It tells the
/// *cook* to split the level into cells and the *player* (PIE and shipping) to
/// stream them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct PartitionSettingsDto {
    /// Whether this level is partitioned at cook time and streamed at runtime.
    pub enabled: bool,
    /// Square cell edge length (metres).
    pub cell_size_m: f64,
    /// How close a streaming source must come to a cell before its entities
    /// **spawn** (metres). Sim-visible: it decides what exists.
    pub activation_radius_m: f64,
    /// Extra metres within which a cell may be *loaded* ahead of need.
    /// **Not** sim-visible: a cell that reaches its activation step unloaded
    /// blocks the step, so this buys latency and never changes a result.
    pub prefetch_margin_m: f64,
}

impl PartitionSettingsDto {
    fn from_record(p: &crate::scene::serialize::PartitionSettings) -> Self {
        Self {
            enabled: p.enabled,
            cell_size_m: p.cell_size_m,
            activation_radius_m: p.activation_radius_m,
            prefetch_margin_m: p.prefetch_margin_m,
        }
    }

    fn to_record(self) -> crate::scene::serialize::PartitionSettings {
        crate::scene::serialize::PartitionSettings {
            enabled: self.enabled,
            cell_size_m: self.cell_size_m,
            activation_radius_m: self.activation_radius_m,
            prefetch_margin_m: self.prefetch_margin_m,
        }
    }
}

/// The level's **time of day** as the World Settings panel edits it (P17.1).
///
/// Unlike every other block on [`LevelSettingsDto`], this is **not** part of the
/// file-level `LevelSettings` record: the clock lives on an ECS component
/// (`inf_ecs::components::TimeOfDay`) carried by the level's *sky authority*
/// entity, because it has to persist per-entity, animate from Blueprints and key
/// from the sequencer. The panel is a view onto that component, which is why this
/// DTO carries [`present`](Self::present).
///
/// Units per architecture rule 6: `seconds` is UTC seconds since midnight,
/// angles are degrees, `rate` is a dimensionless multiplier.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct TimeOfDayDto {
    /// Whether the level actually has a clock. `false` ⇒ the other fields are the
    /// component defaults shown as a preview, and the level renders under the
    /// renderer's retired fixed sun. Writing any row creates the authority.
    pub present: bool,
    /// UTC seconds since midnight, `[0, 86400)`.
    pub seconds: f64,
    /// Day of the year, `1..=365` (no leap day; the engine's year is fixed).
    pub day_of_year: u32,
    /// Latitude in degrees, `+` north.
    pub latitude_deg: f64,
    /// Longitude in degrees, `+` east.
    pub longitude_deg: f64,
    /// Simulated clock-seconds per simulated second. `0` freezes the sun.
    pub rate: f64,
    /// Read-only sun **altitude** above the horizon, degrees — a live readback so
    /// the panel can say "the sun is 34° up" without duplicating the astronomy in
    /// TypeScript. Derived, never written back.
    pub sun_elevation_deg: f64,
    /// Read-only sun **azimuth**, degrees clockwise from north. Derived.
    pub sun_azimuth_deg: f64,
}

impl TimeOfDayDto {
    /// Project the level's clock (or, with none, the component defaults marked
    /// `present: false`).
    pub fn from_doc(doc: &crate::scene::SceneDoc) -> Self {
        let (present, tod) = match doc.time_of_day() {
            Some(t) => (true, t),
            None => (false, inf_ecs::components::TimeOfDay::default()),
        };
        let dir = inf_math::solar::sun_direction(&tod.solar_input());
        Self {
            present,
            seconds: tod.seconds,
            day_of_year: tod.day_of_year,
            latitude_deg: tod.latitude_deg,
            longitude_deg: tod.longitude_deg,
            rate: tod.rate,
            sun_elevation_deg: inf_math::solar::elevation_deg(dir),
            sun_azimuth_deg: inf_math::solar::azimuth_deg(dir),
        }
    }

    /// Convert an edited DTO back into the authoritative component, clamping
    /// every field into its documented range so a hand-crafted IPC payload
    /// cannot put the sun somewhere impossible.
    pub fn to_component(self) -> inf_ecs::components::TimeOfDay {
        inf_ecs::components::TimeOfDay {
            seconds: inf_math::solar::wrap_seconds(self.seconds),
            day_of_year: self.day_of_year.clamp(1, inf_math::solar::DAYS_PER_YEAR),
            latitude_deg: if self.latitude_deg.is_finite() {
                self.latitude_deg.clamp(-90.0, 90.0)
            } else {
                0.0
            },
            longitude_deg: if self.longitude_deg.is_finite() {
                self.longitude_deg.clamp(-180.0, 180.0)
            } else {
                0.0
            },
            rate: if self.rate.is_finite() {
                self.rate
            } else {
                0.0
            },
        }
    }
}

/// The level's **physical atmosphere** as the World Settings panel edits it
/// (P17.2) — the sky-authority entity's `inf_ecs::components::SkyAtmosphere`.
///
/// Like [`TimeOfDayDto`] this is an entity projection, not a file-settings
/// record, and it rides the same [`present`](Self::present) create flag: the two
/// components live on the same authority, so opting into either creates both.
///
/// **Deliberately numeric-and-boolean only.** `SkyAtmosphere` also carries five
/// `Color` fields (sun, moon, and the three gradient colours); those stay in the
/// reflection Details grid, which already has a colour widget, rather than
/// forcing one into the World Settings property-row kit. Nothing is lost — the
/// panel edits the block a level actually tunes, and Details edits the rest
/// through the same undo door.
///
/// Units per architecture rule 6: the fog block and the P17.3 **cloud block** are
/// **SI metres** (`m⁻¹` extinction and falloff, metre heights, m/s wind); the disc
/// sizes are **degrees of angular diameter**; everything else is a dimensionless
/// multiplier over a physical constant that lives in `inf_render::atmosphere` or
/// `inf_render::clouds`.
///
/// The cloud block follows the same numeric-and-boolean rule: `cloud_color` stays
/// in Details with the other five colours.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct SkyAtmosphereDto {
    /// Whether the level has a sky authority at all. `false` ⇒ these are the
    /// component defaults shown as a preview; writing any row creates it.
    pub present: bool,
    /// Whether the sun/moon are projected as a directional light.
    pub enabled: bool,
    /// Draw the physically-based sky (LUTs, discs, stars) instead of the
    /// three-colour gradient.
    pub physical: bool,
    /// Linear exposure multiplier on the sky's radiance.
    pub sky_intensity: f32,
    /// Aerosol density multiplier over clear air ("turbidity").
    pub turbidity: f32,
    /// Mie phase asymmetry `g`, `[-0.95, 0.95]`.
    pub mie_anisotropy: f32,
    /// Sun angular **diameter**, degrees (true ≈ 0.53).
    pub sun_disc_deg: f32,
    /// Moon angular diameter, degrees.
    pub moon_disc_deg: f32,
    /// Starfield brightness multiplier.
    pub star_intensity: f32,
    /// Blend back toward the authored gradient, `[0, 1]`.
    pub tint_strength: f32,
    /// Aerial-perspective strength on lit geometry (`1` = physical).
    pub aerial_perspective: f32,
    /// Height-fog extinction at `fog_height`, **m⁻¹**. `0` = no fog.
    pub fog_density: f32,
    /// Height-fog vertical falloff, **m⁻¹**.
    pub fog_falloff: f32,
    /// World altitude the fog density applies at, **m**.
    pub fog_height: f32,

    // ── volumetric clouds (P17.3) ────────────────────────────────────────
    /// Draw volumetric clouds. Requires [`physical`](Self::physical).
    pub clouds_enabled: bool,
    /// Fractional sky coverage, `[0, 1]`.
    pub cloud_coverage: f32,
    /// Cloud type, `[0, 1]`: 0 = stratus sheet, 1 = cumulus tower.
    pub cloud_type: f32,
    /// Bottom of the cloud layer, **m** of world altitude.
    pub cloud_bottom: f32,
    /// Top of the cloud layer, **m**.
    pub cloud_top: f32,
    /// Cloud extinction at full density, **m⁻¹**.
    pub cloud_density: f32,
    /// Erosion detail strength, `[0, 1]`.
    pub cloud_detail: f32,
    /// Field seed (low 24 bits used).
    pub cloud_seed: u32,
    /// Wind velocity in world X, **m/s**.
    pub cloud_wind_x: f32,
    /// Wind velocity in world Z, **m/s**.
    pub cloud_wind_z: f32,
    /// Forward phase asymmetry `g`, `[0, 0.95]`.
    pub cloud_phase_g: f32,
    /// How much the layer darkens the sun on the ground, `[0, 1]`.
    pub cloud_shadow: f32,
    /// Ambient multiplier inside a cloud, `[0, 4]`.
    pub cloud_ambient: f32,
}

impl SkyAtmosphereDto {
    /// Project the level's atmosphere (or, with no authority, the component
    /// defaults marked `present: false`).
    pub fn from_doc(doc: &crate::scene::SceneDoc) -> Self {
        let (present, a) = match doc.sky_atmosphere() {
            Some(a) => (true, a),
            None => (false, inf_ecs::components::SkyAtmosphere::default()),
        };
        Self {
            present,
            enabled: a.enabled,
            physical: a.physical,
            sky_intensity: a.sky_intensity,
            turbidity: a.turbidity,
            mie_anisotropy: a.mie_anisotropy,
            sun_disc_deg: a.sun_disc_deg,
            moon_disc_deg: a.moon_disc_deg,
            star_intensity: a.star_intensity,
            tint_strength: a.tint_strength,
            aerial_perspective: a.aerial_perspective,
            fog_density: a.fog_density,
            fog_falloff: a.fog_falloff,
            fog_height: a.fog_height,
            clouds_enabled: a.clouds_enabled,
            cloud_coverage: a.cloud_coverage,
            cloud_type: a.cloud_type,
            cloud_bottom: a.cloud_bottom,
            cloud_top: a.cloud_top,
            cloud_density: a.cloud_density,
            cloud_detail: a.cloud_detail,
            cloud_seed: a.cloud_seed,
            cloud_wind_x: a.cloud_wind_x,
            cloud_wind_z: a.cloud_wind_z,
            cloud_phase_g: a.cloud_phase_g,
            cloud_shadow: a.cloud_shadow,
            cloud_ambient: a.cloud_ambient,
        }
    }

    /// Overlay an edited DTO onto the level's current atmosphere, clamping every
    /// field into its documented range.
    ///
    /// It **overlays** rather than constructing from scratch because the DTO
    /// deliberately omits the five `Color` fields (see the type docs): building a
    /// fresh component here would silently reset a level's authored sun colour
    /// every time somebody nudged the fog. Non-finite input falls back to the
    /// component default rather than to zero — a `NaN` turbidity would otherwise
    /// blank the whole sky.
    pub fn to_component(
        self,
        base: inf_ecs::components::SkyAtmosphere,
    ) -> inf_ecs::components::SkyAtmosphere {
        let num = |v: f32, fallback: f32, lo: f32, hi: f32| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        };
        inf_ecs::components::SkyAtmosphere {
            enabled: self.enabled,
            physical: self.physical,
            sky_intensity: num(self.sky_intensity, 1.0, 0.0, 64.0),
            turbidity: num(self.turbidity, 1.0, 0.0, 16.0),
            mie_anisotropy: num(self.mie_anisotropy, 0.8, -0.95, 0.95),
            sun_disc_deg: num(self.sun_disc_deg, 0.545, 0.0, 90.0),
            moon_disc_deg: num(self.moon_disc_deg, 0.52, 0.0, 90.0),
            star_intensity: num(self.star_intensity, 1.0, 0.0, 64.0),
            tint_strength: num(self.tint_strength, 0.0, 0.0, 1.0),
            aerial_perspective: num(self.aerial_perspective, 1.0, 0.0, 4.0),
            // 1 m⁻¹ is already opaque within a metre; the ceiling only exists so
            // a typo cannot produce an infinity in the fog integral.
            fog_density: num(self.fog_density, 0.0, 0.0, 1.0),
            fog_falloff: num(self.fog_falloff, 0.002, 0.0, 1.0),
            fog_height: num(self.fog_height, 0.0, -1.0e7, 1.0e7),
            // ── clouds (P17.3) ──
            clouds_enabled: self.clouds_enabled,
            cloud_coverage: num(self.cloud_coverage, 0.35, 0.0, 1.0),
            cloud_type: num(self.cloud_type, 0.7, 0.0, 1.0),
            // The slab is clamped into a sane troposphere: a cloud layer below
            // sea level or above the stratosphere is not a look, it is a typo,
            // and the march would spend its whole budget on empty air.
            cloud_bottom: num(self.cloud_bottom, 1500.0, -1.0e4, 5.0e4),
            cloud_top: num(self.cloud_top, 4000.0, -1.0e4, 5.0e4),
            // 1 m⁻¹ is opaque within a metre; the ceiling exists only so a typo
            // cannot produce an infinity in the Beer-Lambert integral.
            cloud_density: num(self.cloud_density, 0.04, 0.0, 1.0),
            cloud_detail: num(self.cloud_detail, 0.6, 0.0, 1.0),
            // Masked, not clamped: the renderer carries the seed through an f32
            // uniform and only the low 24 bits survive exactly, so a larger value
            // would silently become a different sky than the one displayed.
            cloud_seed: self.cloud_seed & 0x00ff_ffff,
            // A hurricane is ~80 m/s; past that the field wraps its whole tile
            // inside a frame and reads as static.
            cloud_wind_x: num(self.cloud_wind_x, 6.0, -200.0, 200.0),
            cloud_wind_z: num(self.cloud_wind_z, 2.0, -200.0, 200.0),
            cloud_phase_g: num(self.cloud_phase_g, 0.8, 0.0, 0.95),
            cloud_shadow: num(self.cloud_shadow, 1.0, 0.0, 1.0),
            cloud_ambient: num(self.cloud_ambient, 1.0, 0.0, 4.0),
            ..base
        }
    }
}

/// The five named weather states (P17.4), as the panel's preset buttons name
/// them.
///
/// A DTO twin of [`inf_ecs::components::WeatherPreset`] rather than the type
/// itself, because Ring 0 must not derive `ts_rs::TS` — the same arrangement
/// every other enum on this boundary uses. `rename_all = "lowercase"` makes the
/// generated TypeScript a union of string literals (`"clear" | "overcast" | …`),
/// which is exactly what a row of buttons wants, and it matches
/// [`WeatherPreset::as_str`](inf_ecs::components::WeatherPreset::as_str) so a
/// Blueprint and the panel spell a preset the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
pub enum WeatherPresetDto {
    Clear,
    Overcast,
    Storm,
    Fog,
    Snow,
}

impl WeatherPresetDto {
    /// The Ring-0 preset this names.
    pub fn to_preset(self) -> inf_ecs::components::WeatherPreset {
        use inf_ecs::components::WeatherPreset as P;
        match self {
            WeatherPresetDto::Clear => P::Clear,
            WeatherPresetDto::Overcast => P::Overcast,
            WeatherPresetDto::Storm => P::Storm,
            WeatherPresetDto::Fog => P::Fog,
            WeatherPresetDto::Snow => P::Snow,
        }
    }

    /// Project a Ring-0 preset onto the DTO.
    pub fn from_preset(p: inf_ecs::components::WeatherPreset) -> Self {
        use inf_ecs::components::WeatherPreset as P;
        match p {
            P::Clear => WeatherPresetDto::Clear,
            P::Overcast => WeatherPresetDto::Overcast,
            P::Storm => WeatherPresetDto::Storm,
            P::Fog => WeatherPresetDto::Fog,
            P::Snow => WeatherPresetDto::Snow,
        }
    }
}

/// The level's **weather** block (P17.4) — the `weather_*` half of the sky
/// authority's `SkyAtmosphere`.
///
/// A block of its own rather than more fields on [`SkyAtmosphereDto`], because
/// it is a different question with a different UI: the atmosphere section is a
/// list of physical knobs, while weather is *one coherent state* picked from
/// preset buttons and then, optionally, hand-tuned. Same authority entity, same
/// [`present`](Self::present) create flag.
///
/// Numeric-and-boolean plus the preset enum: like [`SkyAtmosphereDto`], no
/// `Color` crosses here (the droplet tint is the cloud colour, edited in
/// Details).
///
/// Units per architecture rule 6: wind **m/s**, fog extinction **m⁻¹** (SI),
/// blend times **seconds**, the rest dimensionless `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct WeatherDto {
    /// Whether the level has a sky authority at all. `false` ⇒ these are the
    /// component defaults shown as a preview; writing any row creates it.
    pub present: bool,
    /// Whether the weather block **drives** the sky. `false` leaves the authored
    /// cloud/fog rows in charge and stops all precipitation.
    pub enabled: bool,
    /// The preset the live values are blending toward (and equal, once settled).
    pub preset: WeatherPresetDto,
    /// Default transition length, **seconds**, used by the preset buttons and by
    /// a `sky.set_weather` that does not say.
    pub blend_seconds: f32,
    /// Seconds left in the transition in flight; `0` = settled. Read-only in the
    /// panel — it is simulation state, not an authored value.
    pub blend_remaining: f32,
    /// Live cloud coverage `[0, 1]`.
    pub coverage: f32,
    /// Live cloud type `[0, 1]` (0 = stratus sheet, 1 = cumulus tower).
    pub cloud_type: f32,
    /// Live wind in world X, **m/s**.
    pub wind_x: f32,
    /// Live wind in world Z, **m/s**.
    pub wind_z: f32,
    /// Live height-fog extinction, **m⁻¹**.
    pub fog_density: f32,
    /// Live precipitation intensity `[0, 1]`.
    pub precipitation: f32,
    /// Live precipitation phase `[0, 1]`: 0 = rain, 1 = snow.
    pub snowiness: f32,
}

impl WeatherDto {
    /// Project the level's weather (or, with no authority, the component
    /// defaults marked `present: false`).
    pub fn from_doc(doc: &crate::scene::SceneDoc) -> Self {
        let (present, a) = match doc.sky_atmosphere() {
            Some(a) => (true, a),
            None => (false, inf_ecs::components::SkyAtmosphere::default()),
        };
        Self {
            present,
            enabled: a.weather_enabled,
            preset: WeatherPresetDto::from_preset(a.weather_target),
            blend_seconds: a.weather_blend_seconds,
            blend_remaining: a.weather_blend_remaining,
            coverage: a.weather_coverage,
            cloud_type: a.weather_cloud_type,
            wind_x: a.weather_wind_x,
            wind_z: a.weather_wind_z,
            fog_density: a.weather_fog_density,
            precipitation: a.weather_precipitation,
            snowiness: a.weather_snowiness,
        }
    }

    /// Overlay this block onto a live `SkyAtmosphere`, clamping hostile input.
    ///
    /// `..base` matters for the same reason it does on [`SkyAtmosphereDto`]: the
    /// panel sends the whole settings block on every edit, and a wind-slider drag
    /// must not reset the authored cloud colours — or, here, the *atmosphere*
    /// half of the very same component.
    pub fn to_component(
        self,
        base: inf_ecs::components::SkyAtmosphere,
    ) -> inf_ecs::components::SkyAtmosphere {
        let num = |v: f32, fallback: f32, lo: f32, hi: f32| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        };
        // The upper bound is `inf_ecs::sky::MAX_WEATHER_BLEND_S` **by reference**,
        // never a repeated `3600.0`: `sky::set_weather` is the other door into
        // these two fields and clamps to the same constant, and the bound is
        // arithmetic (past it the f32 countdown stops making progress and the
        // blend never settles), so two copies of the number would be two chances
        // to arm the blender forever.
        let max_blend = inf_ecs::sky::MAX_WEATHER_BLEND_S;
        inf_ecs::components::SkyAtmosphere {
            weather_enabled: self.enabled,
            weather_target: self.preset.to_preset(),
            // A blend of 0 is legal (snap); negative is not, and a NaN would make
            // the fixed step's `remaining > 0` test read as false forever.
            weather_blend_seconds: num(self.blend_seconds, 8.0, 0.0, max_blend),
            weather_blend_remaining: num(self.blend_remaining, 0.0, 0.0, max_blend),
            weather_coverage: num(self.coverage, 0.0, 0.0, 1.0),
            weather_cloud_type: num(self.cloud_type, 0.5, 0.0, 1.0),
            weather_wind_x: num(self.wind_x, 0.0, -200.0, 200.0),
            weather_wind_z: num(self.wind_z, 0.0, -200.0, 200.0),
            weather_fog_density: num(self.fog_density, 0.0, 0.0, 1.0),
            weather_precipitation: num(self.precipitation, 0.0, 0.0, 1.0),
            weather_snowiness: num(self.snowiness, 0.0, 0.0, 1.0),
            ..base
        }
    }

    /// The DTO for a preset **snapped**: the state a preset button produces.
    /// Keeps the button's meaning ("this preset, now") in Ring 1 where it is
    /// testable, rather than in TypeScript where it is not.
    pub fn snapped_to(self, preset: WeatherPresetDto) -> Self {
        let p = preset.to_preset().params();
        Self {
            present: true,
            enabled: true,
            preset,
            blend_remaining: 0.0,
            coverage: p.coverage,
            cloud_type: p.cloud_type,
            wind_x: p.wind_x,
            wind_z: p.wind_z,
            fog_density: p.fog_density,
            precipitation: p.precipitation,
            snowiness: p.snowiness,
            ..self
        }
    }
}

/// The level's file-level settings, as the World Settings panel edits them
/// (`scene_get_settings` / `scene_set_settings`). Mirrors
/// [`crate::scene::serialize::LevelSettings`]; `render` nests
/// [`RenderSettingsRecordDto`] and `partition` nests [`PartitionSettingsDto`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct LevelSettingsDto {
    /// 2D world gravity (m/s²) — `[x, y]`.
    pub gravity_2d: [f64; 2],
    /// 3D world gravity (m/s²) — `[x, y, z]`.
    pub gravity_3d: [f64; 3],
    /// Fixed simulation update rate (Hz).
    pub sim_hz: f64,
    /// Renderer HDR / post / lighting block.
    pub render: RenderSettingsRecordDto,
    /// World-partition / level-streaming block (P16.5).
    pub partition: PartitionSettingsDto,
    /// Time-of-day block (P17.1). Projected from the sky-authority **entity's**
    /// components, not from the file settings record — see [`TimeOfDayDto`].
    pub time_of_day: TimeOfDayDto,
    /// Physical-atmosphere block (P17.2). Same authority entity, same
    /// `present` create flag — see [`SkyAtmosphereDto`].
    pub atmosphere: SkyAtmosphereDto,
    /// Weather block (P17.4). Same authority entity again, same create flag —
    /// see [`WeatherDto`].
    pub weather: WeatherDto,
}

impl LevelSettingsDto {
    /// Project the whole document's settings into the DTO the panel reads: the
    /// file-level [`LevelSettings`](crate::scene::serialize::LevelSettings) plus
    /// the time-of-day block, which lives on an entity rather than in the record
    /// (see [`TimeOfDayDto`]).
    pub fn from_doc(doc: &crate::scene::SceneDoc) -> Self {
        let s = doc.settings();
        Self {
            gravity_2d: [s.gravity_2d.x, s.gravity_2d.y],
            gravity_3d: [s.gravity_3d.x, s.gravity_3d.y, s.gravity_3d.z],
            sim_hz: s.sim_hz,
            render: RenderSettingsRecordDto::from_record(&s.render),
            partition: PartitionSettingsDto::from_record(&s.partition),
            time_of_day: TimeOfDayDto::from_doc(doc),
            atmosphere: SkyAtmosphereDto::from_doc(doc),
            weather: WeatherDto::from_doc(doc),
        }
    }

    /// Convert an edited DTO back into the authoritative
    /// [`LevelSettings`](crate::scene::serialize::LevelSettings).
    pub fn to_settings(self) -> crate::scene::serialize::LevelSettings {
        crate::scene::serialize::LevelSettings {
            gravity_2d: inf_ecs::math::Vec2d::new(self.gravity_2d[0], self.gravity_2d[1]),
            gravity_3d: inf_ecs::math::Vec3d::new(
                self.gravity_3d[0],
                self.gravity_3d[1],
                self.gravity_3d[2],
            ),
            sim_hz: self.sim_hz,
            render: self.render.to_record(),
            partition: self.partition.to_record(),
        }
    }
}

// ── Material-instance override editor (E-P2) ─────────────────────────────────
//
// A `.inf_mati` inherits a parent material and overrides a sparse subset of its
// PBR parameters. The editor reads a `MaterialInstanceDto` (parent identity +
// the parent's resolved baseline + the current sparse overrides) via
// `asset_get_material_instance`, and writes edited overrides back via
// `asset_save_material_instance`. `resolved` is the inherited baseline shown
// grayed for each unset override; each `overrides` field is `null` when inherited.

/// Concrete resolved PBR values (the parent chain resolved) — the inherited
/// baseline the editor shows grayed under each unset override.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct MatValuesDto {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
}

impl MatValuesDto {
    /// Project the resolved parent material's PBR block.
    pub fn from_material(m: &inf_material::MaterialAsset) -> Self {
        Self {
            base_color: m.base_color,
            metallic: m.metallic,
            roughness: m.roughness,
            emissive: m.emissive,
        }
    }
}

/// The sparse overrides an instance carries (`null` = inherit the parent value).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct MatOverridesDto {
    pub base_color: Option<[f32; 4]>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
    pub emissive: Option<[f32; 3]>,
}

impl MatOverridesDto {
    /// Project the Ring-0 sparse overrides.
    pub fn from_overrides(o: &inf_material::MatOverrides) -> Self {
        Self {
            base_color: o.base_color,
            metallic: o.metallic,
            roughness: o.roughness,
            emissive: o.emissive,
        }
    }

    /// Convert back into the Ring-0 sparse overrides.
    pub fn to_overrides(self) -> inf_material::MatOverrides {
        inf_material::MatOverrides {
            base_color: self.base_color,
            metallic: self.metallic,
            roughness: self.roughness,
            emissive: self.emissive,
        }
    }
}

/// A material instance projected for the override editor
/// (`asset_get_material_instance`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct MaterialInstanceDto {
    /// Parent material/instance GUID (string form).
    pub parent: String,
    /// Parent display name (editor caption).
    pub parent_name: String,
    /// The parent's resolved PBR baseline (inherited, grayed under unset overrides).
    pub resolved: MatValuesDto,
    /// This instance's sparse overrides.
    pub overrides: MatOverridesDto,
}

// ── Named content collections (E-P8) ─────────────────────────────────────────
//
// User-defined, persisted groupings of assets (the durable successor to the
// frontend-only Favorites), stored at `<project_root>/.infinity/collections.toml`.
// The Content Drawer reads the list via `collections_list` and re-fetches on the
// `collections://changed` event; mutations go through the `collections_*`
// commands. Ids are asset GUID strings (matching `AssetDto.id`).

/// One named collection (`collections_list`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
pub struct CollectionDto {
    pub name: String,
    /// Member asset GUID strings, in insertion order (dangling ids pruned).
    pub ids: Vec<String>,
}

// ── Audio mixer editor (E-P9) ─────────────────────────────────────────────────
//
// The named-bus mixer is `inf_audio::MixerConfig` (hierarchical buses + per-bus
// volume + an effect chain), persisted at `<project_root>/.infinity/mixer.toml`.
// The Audio Mixer panel reads it via `mixer_get`, edits a draft, and writes it
// back via `mixer_save` (which validates, persists, live-applies to a running
// Simulate session, and emits `audio://mixer-changed`). These DTOs mirror the
// Ring-0 shapes faithfully so a load→edit→save round-trips any effect chain.

/// One effect in a bus's chain, mirroring [`inf_audio::Effect`]. `Gain` is
/// fully engine-side and editable (a dB trim folded into the bus's linear gain);
/// `Lowpass` is a device-side DSP effect the editor shows read-only (its cutoff
/// is modelled + folded, but audible filtering needs the cpal sub-track wiring —
/// a documented follow-up). The tag mirrors [`PropValueDto`]'s `kind` convention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MixerEffectDto {
    /// A gain trim in decibels (`0` = unity). Editable in the panel.
    Gain { db: f64 },
    /// A low-pass cutoff in hertz. Shown read-only (device-side follow-up).
    Lowpass { cutoff_hz: f64 },
}

impl MixerEffectDto {
    fn from_effect(e: &inf_audio::Effect) -> Self {
        match *e {
            inf_audio::Effect::Gain { db } => Self::Gain { db },
            inf_audio::Effect::Lowpass { cutoff_hz } => Self::Lowpass { cutoff_hz },
        }
    }

    fn to_effect(self) -> inf_audio::Effect {
        match self {
            Self::Gain { db } => inf_audio::Effect::Gain { db },
            Self::Lowpass { cutoff_hz } => inf_audio::Effect::Lowpass { cutoff_hz },
        }
    }
}

/// One mixer bus (mirrors [`inf_audio::Bus`]): a unique name, an optional parent
/// for the routing hierarchy, a linear volume, and an ordered effect chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct MixerBusDto {
    /// Unique bus name — the key an entity's free-form `AudioSource.bus` matches.
    pub name: String,
    /// Parent bus name; `None` for a root (`"master"` is the undeletable root).
    pub parent: Option<String>,
    /// Linear volume (`1.0` = unity).
    pub volume: f64,
    /// Ordered effect chain (Gain editable, Lowpass read-only).
    pub effects: Vec<MixerEffectDto>,
}

/// The project mixer configuration (mirrors [`inf_audio::MixerConfig`], minus the
/// on-disk `schema_version`, which `mixer_save` stamps with the current version).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
pub struct MixerConfigDto {
    /// The buses, in declaration order (order is preserved on save).
    pub buses: Vec<MixerBusDto>,
}

impl MixerConfigDto {
    /// Project a Ring-0 [`inf_audio::MixerConfig`] into the editor DTO.
    pub fn from_config(c: &inf_audio::MixerConfig) -> Self {
        Self {
            buses: c
                .buses
                .iter()
                .map(|b| MixerBusDto {
                    name: b.name.clone(),
                    parent: b.parent.clone(),
                    volume: b.volume,
                    effects: b.effects.iter().map(MixerEffectDto::from_effect).collect(),
                })
                .collect(),
        }
    }

    /// Convert the edited DTO back into a Ring-0 [`inf_audio::MixerConfig`],
    /// stamping the current schema version. Does NOT validate — the command layer
    /// runs [`validate_mixer`](crate::ipc::validate_mixer) before persisting.
    pub fn to_config(&self) -> inf_audio::MixerConfig {
        inf_audio::MixerConfig {
            schema_version: inf_audio::mixer::MIXER_SCHEMA_VERSION,
            buses: self
                .buses
                .iter()
                .map(|b| inf_audio::mixer::Bus {
                    name: b.name.clone(),
                    parent: b.parent.clone(),
                    volume: b.volume,
                    effects: b.effects.iter().map(|e| e.to_effect()).collect(),
                })
                .collect(),
        }
    }
}

/// Validate a mixer config before persisting it (pure; unit-tested without Tauri).
/// The rules keep the file loadable + the hierarchy sane:
///
/// 1. at least one bus;
/// 2. every name non-empty (trimmed);
/// 3. names unique;
/// 4. a bus named `"master"` is present and is a root (no parent) — it is the
///    undeletable mix root every voice ultimately folds through;
/// 5. every non-`None` parent references an existing bus;
/// 6. no routing cycles (walk each bus's parent chain with a visited set).
pub fn validate_mixer(cfg: &inf_audio::MixerConfig) -> Result<(), String> {
    use std::collections::BTreeSet;

    if cfg.buses.is_empty() {
        return Err("mixer must have at least one bus".into());
    }

    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for b in &cfg.buses {
        let name = b.name.trim();
        if name.is_empty() {
            return Err("bus names must not be empty".into());
        }
        if !seen.insert(b.name.as_str()) {
            return Err(format!("duplicate bus name: {}", b.name));
        }
    }

    let names: BTreeSet<&str> = cfg.buses.iter().map(|b| b.name.as_str()).collect();
    let master = cfg
        .buses
        .iter()
        .find(|b| b.name == "master")
        .ok_or("the master bus must exist and cannot be deleted")?;
    if master.parent.is_some() {
        return Err("the master bus must be a root (it cannot have a parent)".into());
    }

    for b in &cfg.buses {
        if let Some(p) = &b.parent {
            if !names.contains(p.as_str()) {
                return Err(format!("bus {} has an unknown parent: {p}", b.name));
            }
        }
    }

    // Cycle check: follow each bus up its parent chain; a revisit or a walk longer
    // than the bus count is a cycle.
    let index: std::collections::BTreeMap<&str, &inf_audio::mixer::Bus> =
        cfg.buses.iter().map(|b| (b.name.as_str(), b)).collect();
    for start in &cfg.buses {
        let mut visited: BTreeSet<&str> = BTreeSet::new();
        let mut cur = Some(start.name.as_str());
        while let Some(name) = cur {
            if !visited.insert(name) {
                return Err(format!("routing cycle through bus: {name}"));
            }
            cur = index
                .get(name)
                .and_then(|b| b.parent.as_deref())
                .filter(|p| index.contains_key(*p));
        }
    }

    Ok(())
}

// ── the Model Editor (P23.4) ───────────────────────────────────────────────

/// Which component kind the Model Editor is selecting in.
///
/// A mirror of `inf_dcc::SelectMode` rather than a re-export, for the reason
/// every DTO in this file is: the wire shape is the editor's contract and the
/// kernel's enum is the kernel's, and tying them together means a kernel rename
/// silently changes an IPC payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
pub enum DccModeDto {
    #[default]
    Vert,
    Edge,
    Face,
}

/// What the kernel's reader had to do to open this asset — surfaced, not hidden.
///
/// `boundaryEdges` is the one the panel puts a verdict on: a solid the author
/// believes is closed arriving with boundary edges is *fragmented*, and the exact
/// weld (tolerance zero, and it stays zero) is why. Telling them beats picking an
/// epsilon on their behalf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccImportDto {
    pub source_vertices: u32,
    pub welded_positions: u32,
    pub fan_splits: u32,
    pub degenerate_triangles_skipped: u32,
    pub sharp_edges: u32,
    pub boundary_edges: u32,
    pub non_finite_values: u32,
    /// **Welded positions where two source vertices disagreed about their
    /// skinning influences** (P24.2). Normally zero — a well-formed exporter
    /// gives every split copy of a vertex the same weights — and a non-zero
    /// reading is also the exact number that makes the export round trip
    /// inexact, because first-occurrence wins.
    pub skin_conflicts: u32,
}

/// What the writer had to do on the way out — the two unroundtrippable counters
/// plus the shape of what was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccExportDto {
    pub submeshes: u32,
    pub vertices: u32,
    pub triangles: u32,
    pub fan_fallbacks: u32,
    pub fallback_tangents: u32,
    /// Whether `meshopt` ran — the crate's one non-deterministic step, and the
    /// only field here that is a *setting* rather than a count.
    ///
    /// Reached the author for the first time at P24.2, found by the drift pin
    /// (`report_drift.rs`) while it was being written for `skin_conflicts`.
    pub optimized: bool,
    /// Kernel vertices that share a position with another. **Non-zero means the
    /// next open will not be this mesh**: the reader's exact weld fuses them.
    pub coincident_vertices: u32,
    /// Triangulation diagonals that had to repeat an existing edge. The other way
    /// a written asset comes back unreadable.
    pub reused_diagonals: u32,
    pub non_finite_written: u32,
    pub non_unit_normals_written: u32,
    /// Submeshes `optimize` was asked for and **did not run on**, because they
    /// carry a skin stream (P24.2). `inf_mesh::optimize` returns
    /// `(vertices, indices)` only, so a parallel per-vertex stream cannot follow
    /// its permutation — running it would give every vertex another vertex's
    /// weights. Skipping is the sound answer; this is the author being told.
    pub optimize_skipped_skinned: u32,
}

/// One open Model Editor document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccDocDto {
    /// `"dcc:<assetId>"` — one document per mesh asset.
    pub id: String,
    pub asset_id: String,
    pub name: String,
    pub mode: DccModeDto,
    pub verts: u32,
    pub edges: u32,
    pub faces: u32,
    /// How many components are selected **in the current mode**.
    pub selected: u32,
    pub can_undo: bool,
    pub can_redo: bool,
    /// Unsaved edits since the last `dcc_save` (or since open).
    pub dirty: bool,
    /// The journal generation. The frontend does not interpret it; it is the
    /// preview's cache key and the reason a stale image is impossible.
    #[ts(type = "number")]
    pub generation: u64,
    pub import: DccImportDto,
    /// How many waypoints the knife has collected (see `dcc_pick`).
    pub knife_points: u32,
    /// A pointer drag is in flight (P23.5). The panel reads it to keep sending
    /// moves and to know that a `dcc_drag_end` is owed — but the **backend** is
    /// what settles an orphaned one, because a panel that has already unmounted
    /// cannot.
    pub dragging: bool,
    /// How many raw path points the stroke in flight has collected. `0` for a
    /// gizmo drag or no drag. Surfaced so "is this doing anything" is answerable
    /// from the status bar.
    pub drag_points: u32,
    /// Which transform gizmo is armed, if any. Backend state, like the camera
    /// and the selection, because the handles are drawn and picked backend-side
    /// and a panel-held copy would be a second opinion about the active tool.
    pub gizmo: Option<DccGizmoModeDto>,
    /// **A content revision of the selection** — what a view keys on to know its
    /// picture is stale.
    ///
    /// Not the journal `generation` (a selection change does not move it) and not
    /// `selected` (a count: face A and face B both read `1`). Neither of those can
    /// tell two different one-face selections apart, and the UV pane keyed on the
    /// first of them and therefore never refreshed on a pick.
    #[ts(type = "number")]
    pub selection_rev: u64,
    /// Undirected edges marked as UV seams (P23.5).
    pub seams: u32,
    /// How many charts the seams cut the mesh into — the number an author checks
    /// **before** unwrapping, because it is the thing their seam marks control.
    pub charts: u32,
    /// How many joints this mesh's skin channel is bound to, or `None` when it
    /// carries no skin (P24.2).
    ///
    /// The weight brush's **bound**, and the reason the influence picker is a
    /// number box rather than a list: a `.inf_mesh` records no skeleton (the
    /// pairing lives in the scene's `SkeletalMesh`), so the kernel knows how many
    /// joints its indices address and not what any of them is called. Names
    /// arrive with P24.3's skeleton binding UI.
    pub skin_joints: Option<u32>,
}

// ── the Skeleton Editor (P24.3) ─────────────────────────────────────────────

/// One joint, as the tree view reads it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct SkelJointDto {
    /// Index into the skeleton's joints — the id every other command uses.
    pub index: u16,
    pub name: String,
    /// `None` for a root.
    pub parent: Option<u16>,
    /// Local **rest** translation (metres), rotation (`[x,y,z,w]`) and scale —
    /// the values the transform editor writes back. Not the animated pose: a
    /// skeleton asset has none.
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    /// One of the canonical humanoid nineteen. The panel marks these because
    /// renaming one costs `RetargetMap::humanoid_identity` a pairing.
    pub canonical: bool,
    /// This joint's left/right twin, when the rig has one.
    pub mirror: Option<u16>,
    /// The name says it has a side and the rig has no twin — mirroring across
    /// this rig would weight the copy to the wrong side.
    pub sided_without_twin: bool,
    /// The authored rotation limit, if any: min/max degrees about local X/Y/Z.
    /// **Absent means unlimited**, which is why these are `Option` and not a
    /// full-range default (`inf_anim::JointLimit`'s own rule).
    pub limit_min_deg: Option<[f32; 3]>,
    pub limit_max_deg: Option<[f32; 3]>,
}

/// One socket, as the socket list reads it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct SkelSocketDto {
    pub name: String,
    pub joint: u16,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// One open Skeleton Editor document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct SkelDocDto {
    /// The asset GUID this document edits — also the document key. Unlike the
    /// Model Editor there is no separate session file, so the asset **is** the
    /// document and re-opening returns the live one.
    pub id: String,
    pub name: String,
    pub joints: Vec<SkelJointDto>,
    pub sockets: Vec<SkelSocketDto>,
    /// Unsaved changes, measured against the bytes on disk rather than a flag —
    /// so undoing back to the saved state correctly reads clean.
    pub dirty: bool,
    pub can_undo: bool,
    pub can_redo: bool,
    /// Sided joints with no opposite number, by name. Non-empty means a mirror
    /// across this rig is refused; the panel shows it as a rig warning rather
    /// than waiting for the refusal.
    pub unmatched_sided: Vec<String>,
}

/// The answer to every mutating Skeleton Editor command.
///
/// One shape for all of them (the [`DccApplyDto`] pattern) so the frontend has a
/// single reducer, and a refusal can never be mistaken for a success that
/// happened to change nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct SkelApplyDto {
    pub ok: bool,
    /// Why it refused — the Ring-1 error's own text, which names the joint, the
    /// socket or the parameter. `None` when `ok`.
    pub refusal: Option<String>,
    /// What a successful edit **cost**, when that is not nothing: a rename that
    /// left the canonical vocabulary or broke a mirror pair. A warning, never a
    /// refusal — the edit happened.
    pub warning: Option<String>,
    pub doc: SkelDocDto,
}

// ── P24.5: New Character from Template ──────────────────────────────────────
//
// The wizard's whole wire. Two mirrors (`BodyParamsDto`, `GaitParamsDto`) carry
// EVERY field of the Ring-0 structs rather than the subset today's panel exposes:
// a wire that carried only the exposed knobs would have to move the day a slider
// is added, and `the_wizard_dtos_carry_every_field_of_their_models` fails the day
// either model grows a field this does not.

/// [`inf_anim::BodyParams`] on the wire. Every length is **metres** and every
/// ratio is a fraction of `height_m` (SI, units doctrine); nothing here is
/// degrees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct BodyParamsDto {
    pub height_m: f64,
    pub hip_height_ratio: f64,
    pub shoulder_height_ratio: f64,
    pub head_height_ratio: f64,
    pub spine_segments: u16,
    pub neck_segments: u16,
    pub shoulder_width_m: f64,
    pub hip_width_m: f64,
    pub upper_limb_ratio: f64,
    pub arm_length_ratio: f64,
    /// Multi-girdle plans only; a biped's torso is vertical.
    pub body_length_m: f64,
    /// Multi-girdle plans only.
    pub head_forward_m: f64,
}

/// [`inf_anim::locomotion::GaitParams`] on the wire. Angles are **degrees** (the
/// authoring boundary), rates are hertz, and the two `*_ratio` fields are
/// fractions of the rig's own hip height.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct GaitParamsDto {
    pub walk_cadence_hz: f64,
    pub run_cadence_hz: f64,
    pub hip_swing_deg: f64,
    pub knee_flex_deg: f64,
    pub arm_swing_deg: f64,
    pub run_stride_scale: f64,
    pub bob_ratio: f64,
    pub idle_period_s: f64,
    pub idle_bob_ratio: f64,
    pub idle_pitch_deg: f64,
    pub keys_per_cycle: u32,
}

/// Everything the wizard collects, in one message — so a preview and a create
/// are the *same* input and a preview cannot describe a character the create
/// would not make.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CharacterSpecDto {
    pub name: String,
    /// `"biped"` | `"quadruped"` | `"hexapod"` | `"npedal"`. A **string** for
    /// `skel_create_template`'s stated reason: the plan set is the part of this
    /// API most likely to grow, and a name that fails loudly with the list of
    /// what it does know is kinder to a stale frontend than a generated union
    /// that silently loses a variant.
    pub plan: String,
    /// Leg count for `"npedal"`; ignored by the named plans.
    pub legs: Option<u16>,
    pub params: BodyParamsDto,
    pub gait: GaitParamsDto,
    /// An existing `.inf_mesh` GUID to fit the rig to and skin. Absent → the
    /// wizard generates a blocky mannequin from the rig.
    pub mesh_asset: Option<String>,
}

/// One joint of the previewed rig — the three numbers the Skeleton Editor's SVG
/// diagram projects from, so the wizard and the editor draw the same picture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CharacterJointDto {
    pub name: String,
    pub parent: Option<u16>,
    /// Local rest translation, metres.
    pub translation: [f32; 3],
}

/// One driven leg and where it falls in the gait cycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CharacterLegDto {
    pub name: String,
    pub length_m: f64,
    /// Position in the cycle, `[0, 1)`.
    pub phase: f64,
}

/// What a spec *would* produce — recomputed on every slider drag, with nothing
/// written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CharacterRigDto {
    pub joints: Vec<CharacterJointDto>,
    pub sockets: Vec<String>,
    /// Joints carrying a rotation limit — the IK input the template emits.
    pub limits: u32,
    /// The span between the rig's lowest and highest **joint** in the bind pose
    /// (metres). **Not** the requested height and not the creature's: a template
    /// rig's topmost joint is `head`, at `head_height_ratio × height_m`, so a
    /// 1.75 m biped reads 1.6275. A fitted rig takes the mesh's, which is what
    /// makes this the fit's readout. The panel labels it "Joint span" for that
    /// reason.
    pub height_m: f64,
    pub legs: Vec<CharacterLegDto>,
    /// idle / walk / run cycle lengths, seconds.
    pub durations_s: [f32; 3],
    pub walk_speed_m_s: f64,
    pub run_speed_m_s: f64,
    pub walk_threshold_m_s: f64,
    pub run_threshold_m_s: f64,
    /// Mannequin vertex + triangle count, absent when the spec brings its own
    /// mesh.
    pub body_vertices: Option<u32>,
    pub body_triangles: Option<u32>,
}

/// The preview answer. A refusal is a **value**, not a command error: half the
/// intermediate states of a proportion drag are invalid (hips above shoulders,
/// for one), and an error toast per keystroke is not a wizard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CharacterPreviewDto {
    /// The generator's own message, naming the offending parameter. `None` when
    /// `rig` is present.
    pub refusal: Option<String>,
    pub rig: Option<CharacterRigDto>,
}

/// What a create produced. Every id is a GUID string, in the order the assets
/// were written (which is their dependency order).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CharacterCreateDto {
    pub skeleton: String,
    pub mesh: String,
    pub idle: String,
    pub walk: String,
    pub run: String,
    pub machine: String,
    /// The spawned actor's entity GUID, when the wizard was asked to add one.
    pub actor: Option<String>,
    /// Whether `mesh` is the generated mannequin.
    pub mannequin: bool,
    /// The auto-fit's readout, when one ran.
    pub fit: Option<String>,
    /// The weight solve's readout, when one ran.
    pub weights: Option<String>,
    /// Things that happened and are not refusals.
    pub warnings: Vec<String>,
}

impl BodyParamsDto {
    /// The Ring-0 params this describes.
    pub fn to_params(&self) -> inf_anim::BodyParams {
        inf_anim::BodyParams {
            height_m: self.height_m,
            hip_height_ratio: self.hip_height_ratio,
            shoulder_height_ratio: self.shoulder_height_ratio,
            head_height_ratio: self.head_height_ratio,
            spine_segments: self.spine_segments,
            neck_segments: self.neck_segments,
            shoulder_width_m: self.shoulder_width_m,
            hip_width_m: self.hip_width_m,
            upper_limb_ratio: self.upper_limb_ratio,
            arm_length_ratio: self.arm_length_ratio,
            body_length_m: self.body_length_m,
            head_forward_m: self.head_forward_m,
        }
    }

    /// The DTO form.
    pub fn from_params(p: &inf_anim::BodyParams) -> Self {
        Self {
            height_m: p.height_m,
            hip_height_ratio: p.hip_height_ratio,
            shoulder_height_ratio: p.shoulder_height_ratio,
            head_height_ratio: p.head_height_ratio,
            spine_segments: p.spine_segments,
            neck_segments: p.neck_segments,
            shoulder_width_m: p.shoulder_width_m,
            hip_width_m: p.hip_width_m,
            upper_limb_ratio: p.upper_limb_ratio,
            arm_length_ratio: p.arm_length_ratio,
            body_length_m: p.body_length_m,
            head_forward_m: p.head_forward_m,
        }
    }
}

impl GaitParamsDto {
    /// The Ring-0 gait this describes.
    pub fn to_gait(&self) -> inf_anim::locomotion::GaitParams {
        inf_anim::locomotion::GaitParams {
            walk_cadence_hz: self.walk_cadence_hz,
            run_cadence_hz: self.run_cadence_hz,
            hip_swing_deg: self.hip_swing_deg,
            knee_flex_deg: self.knee_flex_deg,
            arm_swing_deg: self.arm_swing_deg,
            run_stride_scale: self.run_stride_scale,
            bob_ratio: self.bob_ratio,
            idle_period_s: self.idle_period_s,
            idle_bob_ratio: self.idle_bob_ratio,
            idle_pitch_deg: self.idle_pitch_deg,
            keys_per_cycle: self.keys_per_cycle,
        }
    }

    /// The DTO form.
    pub fn from_gait(g: &inf_anim::locomotion::GaitParams) -> Self {
        Self {
            walk_cadence_hz: g.walk_cadence_hz,
            run_cadence_hz: g.run_cadence_hz,
            hip_swing_deg: g.hip_swing_deg,
            knee_flex_deg: g.knee_flex_deg,
            arm_swing_deg: g.arm_swing_deg,
            run_stride_scale: g.run_stride_scale,
            bob_ratio: g.bob_ratio,
            idle_period_s: g.idle_period_s,
            idle_bob_ratio: g.idle_bob_ratio,
            idle_pitch_deg: g.idle_pitch_deg,
            keys_per_cycle: g.keys_per_cycle,
        }
    }
}

impl CharacterRigDto {
    /// Project a Ring-1 preview onto the wire.
    pub fn from_preview(p: &crate::character::CharacterPreview) -> Self {
        Self {
            joints: p
                .joints
                .iter()
                .map(|j| CharacterJointDto {
                    name: j.name.clone(),
                    parent: j.parent,
                    translation: j.translation,
                })
                .collect(),
            sockets: p.sockets.clone(),
            limits: p.limits as u32,
            height_m: p.height_m,
            legs: p
                .legs
                .iter()
                .map(|(name, length_m, phase)| CharacterLegDto {
                    name: name.clone(),
                    length_m: *length_m,
                    phase: *phase,
                })
                .collect(),
            durations_s: p.durations_s,
            walk_speed_m_s: p.walk_speed_m_s,
            run_speed_m_s: p.run_speed_m_s,
            walk_threshold_m_s: p.walk_threshold_m_s,
            run_threshold_m_s: p.run_threshold_m_s,
            body_vertices: p.body.map(|(v, _)| v as u32),
            body_triangles: p.body.map(|(_, t)| t as u32),
        }
    }
}

/// The result of a tool press: what it did, or why it refused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccApplyDto {
    pub ok: bool,
    /// The kernel's typed refusal, rendered for a human. Empty when `ok`.
    pub refusal: Option<String>,
    pub doc: DccDocDto,
}

/// A rendered preview frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccPreviewDto {
    /// A PNG data-URL, or `None` on a machine with no GPU adapter (the standing
    /// degrade-to-icons case) or a shader that did not validate.
    pub image: Option<String>,
    /// Why there is no image. A value, never a crash (the P21 law).
    pub error: Option<String>,
    pub size: u32,
}

/// The verdict on a save — the P20.4 readout pattern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccSaveDto {
    pub ok: bool,
    pub export: DccExportDto,
    /// What happened to the derived `.inf_vmesh`: `"built"`, `"cached"` or
    /// `"skipped"`. Surfaced because the whole point of doing it synchronously is
    /// that the author can see it happened.
    pub vmesh: String,
    /// Human-readable advisories drawn from the export report — the two
    /// unroundtrippable counters, and anything the writer had to fall back on.
    pub advisories: Vec<String>,
}

/// **The Cloth section's knobs** (P24.4) — what the panel sends to
/// `dcc_make_garment`.
///
/// Compliance is m/N and not a 0..1 "stiffness", because XPBD compliance is
/// timestep-independent and this engine compares traces across hosts that may
/// tick at different rates (see `inf_anim::ClothMaterial`). Everything else is
/// metres, `1/s` or a count. The *operand* — which vertices are pinned — is the
/// document's own selection, resolved backend-side, exactly like a tool press.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccGarmentDto {
    /// Stretch compliance, m/N. `0` is inextensible.
    pub stretch_compliance: f32,
    /// Bend compliance, m/N. Two orders above the stretch one is a skirt.
    pub bend_compliance: f32,
    /// Velocity damping, 1/s.
    pub damping: f32,
    /// Collision thickness, metres.
    pub thickness_m: f32,
    /// Substeps per fixed step (`0` reads as `1`).
    pub substeps: u8,
    /// Constraint sweeps per substep (`0` reads as `1`).
    pub iterations: u8,
    /// Uniform body-capsule radius, metres. `0` derives none.
    pub body_radius_m: f32,
    /// The `.inf_skel` whose bones become the collision capsules, or `None` for a
    /// garment that collides against nothing.
    pub skeleton: Option<String>,
    /// Asset name for the new `.inf_cloth`.
    pub name: Option<String>,
}

/// **The Hair section's knobs** (P24.4) — what the panel sends to
/// `dcc_grow_hair`. The *operand* is the face selection: the scalp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccGroomDto {
    /// Strand length, metres.
    pub length_m: f32,
    /// Segments per strand.
    pub segments: u16,
    /// Segment compliance, m/N.
    pub segment_compliance: f32,
    /// Velocity damping, 1/s.
    pub damping: f32,
    /// Collision thickness, metres.
    pub thickness_m: f32,
    /// Substeps per fixed step (`0` reads as `1`).
    pub substeps: u8,
    /// Ribbon width at the root, metres.
    pub ribbon_width_m: f32,
    /// Clump strength, `0..=1` (dimensionless).
    pub clump_strength: f32,
    /// Clump cell size, metres. `0` puts every root in its own clump.
    pub clump_spacing_m: f32,
    /// Curl radius, metres. `0` is straight hair.
    pub curl_radius_m: f32,
    /// Curl turns over the strand's length (revolutions).
    pub curl_turns: f32,
    /// The joint a root rides when the scalp carries no skin weights.
    pub fallback_joint: u16,
    /// Uniform body-capsule radius, metres.
    pub body_radius_m: f32,
    /// The `.inf_skel` whose bones become the collision capsules.
    pub skeleton: Option<String>,
    /// Asset name for the new `.inf_hair`.
    pub name: Option<String>,
}

/// The verdict on an authoring press — the `DccSaveDto` readout pattern, for
/// cloth and hair (P24.4).
///
/// A refusal is a **value** with the builder's own words in it, never a thrown
/// error: every one of them is reachable from a mesh somebody modelled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccGroomResultDto {
    pub ok: bool,
    /// Why not, in the refusing layer's own words.
    pub refusal: Option<String>,
    /// The new asset's GUID.
    pub asset: Option<String>,
    /// Where it was written, relative to the content root.
    pub path: Option<String>,
    /// What was derived — the counters the panel prints, so an author can see
    /// that a garment has constraints and a hairstyle has strands without
    /// re-opening the file.
    pub stats: Vec<DccGroomStatDto>,
}

/// One labelled counter in a [`DccGroomResultDto`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccGroomStatDto {
    pub label: String,
    pub value: u32,
}

/// A modelling tool press. Parameters arrive from the toolbar popovers; the
/// *operands* are the document's current selection, resolved backend-side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "tool", rename_all = "camelCase")]
pub enum DccToolDto {
    /// Faces along the region normal, or boundary edges by an explicit delta.
    Extrude {
        distance: f64,
    },
    ExtrudeEdges {
        delta: [f64; 3],
    },
    Inset {
        amount: f64,
        individual: bool,
    },
    Bevel {
        amount: f64,
    },
    LoopCut {
        cuts: u32,
    },
    /// Cuts along the vertices in the order they were picked.
    Knife,
    Merge {
        center: bool,
    },
    Subdivide,
    /// `axis` is `"x"`, `"y"` or `"z"`.
    Mirror {
        axis: String,
        coord: f64,
    },
    Translate {
        delta: [f64; 3],
    },
    /// Mark (or clear) the selected edges as **UV seams** (P23.5). One op per
    /// edge, so an undo peels one mark at a time.
    Seam {
        seam: bool,
    },
    /// Soft-translate: the selection at full weight, its geodesic neighbourhood
    /// scaled by the falloff.
    SoftTranslate {
        delta: [f64; 3],
        radius: f64,
        falloff: SculptFalloffDto,
    },
    Delete,
}

/// A selection command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "action", rename_all = "camelCase")]
pub enum DccSelectDto {
    /// Switch component mode, converting what is selected.
    Mode {
        mode: DccModeDto,
    },
    All,
    None,
    Invert,
    Grow,
    Shrink,
    Linked,
    /// Edge loop / edge ring through the last-picked edge.
    Loop,
    Ring,
}

// ── P23.5: sculpt strokes and the component gizmo ──────────────────────────

/// What a brush dab does. A mirror of `inf_dcc::SculptMode`, for the reason
/// [`DccModeDto`] is a mirror of `SelectMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
pub enum DccSculptModeDto {
    #[default]
    Draw,
    Smooth,
    Flatten,
    Grab,
}

/// What a **weight** dab does to the influence under it (P24.2). A mirror of
/// `inf_dcc::PaintMode`, for the reason [`DccSculptModeDto`] is a mirror of
/// `SculptMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
pub enum DccPaintModeDto {
    #[default]
    Add,
    Subtract,
    Replace,
    Smooth,
}

/// Which transform the component gizmo is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "lowercase")]
pub enum DccGizmoModeDto {
    #[default]
    Translate,
    Rotate,
    Scale,
}

/// A pointer drag the Model Editor is about to start.
///
/// **One shape for both gestures.** A sculpt stroke and a gizmo drag are the same
/// pointer-down / move / up sequence with different arithmetic, so they share one
/// command triple (`dcc_drag_begin` / `_move` / `_end`) and one pending slot on
/// the document. Two triples would be two places for a drag to be forgotten, and
/// a forgotten drag is exactly what the orphan-settler doctrine exists to stop.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum DccDragDto {
    /// Paint along the drag path. `radius` is metres of **geodesic** reach;
    /// `strength` is metres at full weight for `draw`, a blend fraction for
    /// `smooth`/`flatten`, and a multiplier on the drag for `grab`.
    Sculpt {
        mode: DccSculptModeDto,
        radius: f64,
        strength: f64,
        falloff: SculptFalloffDto,
    },
    /// Drag a handle on the current selection. `snap` quantizes the result
    /// (metres / radians / ratio; `0` = off) and `soft_radius` turns on
    /// soft-select weighting when positive.
    Gizmo {
        mode: DccGizmoModeDto,
        snap: f64,
        /// Renamed explicitly because `rename_all` on a **tagged enum** renames
        /// its variants, not its fields — so without this the one multi-word
        /// field in this file's Dcc surface would have crossed the bridge as
        /// `soft_radius` while every neighbour is camelCase, and the mismatch
        /// would only ever show up as a silently-zero radius.
        #[serde(rename = "softRadius")]
        #[ts(rename = "softRadius")]
        soft_radius: f64,
        falloff: SculptFalloffDto,
    },
    /// Paint one **skinning influence** along the drag path (P24.2). `joint` is
    /// an index into the mesh's bound skeleton; `strength` is a weight delta in
    /// `[0, 1]` at full coverage, and `radius` is geodesic metres exactly as for
    /// a sculpt stroke.
    WeightPaint {
        joint: u32,
        mode: DccPaintModeDto,
        radius: f64,
        strength: f64,
        falloff: SculptFalloffDto,
    },
}

/// The verdict on an unwrap — the P20.4 readout pattern, and the place the
/// solver's **residual** is reported rather than hidden.
///
/// `worstResidual` is `‖Ax − b‖ / ‖b‖` for the worst chart after the fixed
/// iteration count. A fixed-count solver either converged or it did not, and the
/// only honest interface is to say which: a big number means a distorted chart,
/// not a failure, and the author can act on it (add a seam) in a way that "unwrap
/// complete" would never have prompted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccUnwrapDto {
    pub ok: bool,
    /// The kernel's typed refusal, rendered for a human. Empty when `ok`.
    pub refusal: Option<String>,
    pub charts: u32,
    pub corners: u32,
    pub seams: u32,
    /// **How much the worst chart has to stretch** — a property of the geometry.
    /// Non-zero for a shape that is not developable, however well the solve went.
    pub worst_residual: f64,
    /// **Whether the solver finished** — a property of the solve, zero iff CG
    /// converged. Split from `worstResidual` because one number gave the same
    /// reading, and therefore the same advice, to opposite causes: a *failed* flat
    /// plane read 5.7e-2 and a *converged* saddle read 4.1e-2.
    pub worst_convergence: f64,
    /// Triangles whose UV winding opposes their chart's majority — **folds**. A
    /// converged, low-distortion unwrap can still overlap itself when a chart is
    /// not a disk, and this is the only number that sees it.
    pub flipped: u32,
    /// Triangles across all charts, so `flipped` has a denominator.
    pub triangles: u32,
    pub doc: DccDocDto,
}

/// The result of starting a drag: whether the pointer actually grabbed anything.
///
/// Distinguished from [`DccApplyDto`] because "the pointer missed the model" is
/// not a refusal to report in the status bar — it is the panel's cue to orbit the
/// camera instead, which is the behaviour that makes a sculpt tool usable without
/// a modifier key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct DccDragBeginDto {
    /// `true` when a drag is now in flight. `false` means the pointer hit
    /// nothing, nothing is pending, and the panel should treat the gesture as a
    /// camera orbit.
    pub grabbed: bool,
    /// Which gizmo handle was grabbed, for the panel's cursor. `None` for a
    /// sculpt stroke or a miss.
    pub handle: Option<String>,
    /// Why the drag was **refused**, as opposed to having missed. A miss is silent
    /// — it becomes a camera orbit — but a refusal is a sentence, because an
    /// author whose brush did nothing needs to know it was the radius.
    pub refusal: Option<String>,
    pub doc: DccDocDto,
}

// ── the capture wizard (P25.4) ──────────────────────────────────────────────
//
// Photographs in, a standard asset out. Every rule lives in
// `inf_editor_core::capture`; these are the wire shapes of what it already
// computed, and nothing here decides anything. Strings where Ring 1 has enums,
// on the standing wire-enum rule: a frontend built against an older backend gets
// a name it does not know rather than a deserialization failure.

/// The assumed lens a capture is reconstructed with.
///
/// Structure from motion never refines intrinsics, so this is the one thing the
/// wizard has to be told. `focal_ratio` is a fraction of the image's longer side
/// so one setting covers a whole shoot at any resolution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CaptureCameraDto {
    /// Focal length over `max(width, height)`.
    pub focal_ratio: f64,
    /// First radial-distortion coefficient (`0` is a pinhole).
    pub k1: f64,
    /// Second radial coefficient.
    pub k2: f64,
}

/// The knobs the capture wizard exposes.
///
/// A **subset** of `CaptureConfig`, deliberately: the sparse and dense solvers'
/// constants are committed numbers whose defaults every measurement in Phase 25
/// was taken at, and a dialog that let them be typed would make every one of
/// those numbers a claim about somebody's session.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CaptureSettingsDto {
    pub camera: CaptureCameraDto,
    /// **The scale step.** Metres per reconstruction unit; `1.0` leaves the
    /// result in the reconstruction's own baseline units.
    pub metres_per_unit: f64,
    /// Triangle budget for the retopologized mesh.
    pub target_triangles: u32,
    /// Atlas side, in texels.
    pub atlas_size: u32,
    /// Ambient-occlusion rays per texel.
    pub ao_rays: u32,
    /// Drop geometry no camera photographed.
    pub trim_unseen: bool,
    /// Attempt de-lighting (it refuses on its own when the fit is not
    /// believable).
    pub delight: bool,
    /// Roughness written into the material and the ORM map.
    pub roughness: f32,
    /// Metallic, likewise.
    pub metallic: f32,
}

/// One row of the wizard's photograph table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct PhotoEntryDto {
    pub path: String,
    pub name: String,
    /// `0` when the file did not decode.
    pub width: u32,
    pub height: u32,
    /// Why it did not decode.
    pub error: Option<String>,
}

/// One finding, with the severity and stage that decide where it is shown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CaptureIssueDto {
    /// `"blocking" | "warning" | "note"`.
    pub severity: String,
    /// `"load" | "sfm" | "dense" | "finish" | "write"`.
    pub stage: String,
    /// The sentence, carrying its own remedy.
    pub message: String,
}

/// What one camera saw of the finished mesh — a row of the coverage overlay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CoverageViewDto {
    pub view: u32,
    pub photo: String,
    /// Whether it got a pose at all.
    pub registered: bool,
    pub triangles_seen: u32,
    /// That, over the finished triangle count.
    pub fraction: f64,
}

/// The coverage and overlap readout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CoverageDto {
    pub triangles: u32,
    pub views: Vec<CoverageViewDto>,
    pub seen_by_none: u32,
    pub seen_by_one: u32,
    pub seen_by_two_or_more: u32,
    pub unseen_texels: u32,
    pub covered_texels: u32,
    /// Seen by at least one camera, as a fraction.
    pub covered_fraction: f64,
    /// Seen by two or more — the redundancy the method rests on.
    pub overlap_fraction: f64,
}

/// The numbers a finished run produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CaptureResultDto {
    pub registered: u32,
    pub views: u32,
    pub points: u32,
    pub reprojection_rms_px: f64,
    pub dense_triangles: u32,
    pub voxel_size: f64,
    pub triangles: u32,
    pub vertices: u32,
    pub charts: u32,
    pub atlas_coverage: f64,
    /// The longest side in **baseline units** — what a known real-world length
    /// is divided by to get [`CaptureSettingsDto::metres_per_unit`]. It does
    /// **not** move when the scale does, so a second correction is computed
    /// against the same number as the first.
    pub extent_units: f64,
    /// The same side at the scale the run used, in metres.
    pub extent_metres: f64,
    pub coverage: CoverageDto,
    /// Wall clock per stage, in `load, sfm, dense, finish, write` order.
    #[ts(type = "number[]")]
    pub elapsed_ms: Vec<u64>,
}

/// Everything the capture panel renders.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CaptureStatusDto {
    /// `"idle" | "running" | "ready" | "imported" | "failed" | "cancelled"`.
    pub state: String,
    /// The stage in flight, when one is.
    pub stage: Option<String>,
    /// The run these events belong to.
    #[ts(type = "number")]
    pub run: u64,
    pub photos: Vec<PhotoEntryDto>,
    pub settings: CaptureSettingsDto,
    /// The pre-flight before a run, the product's findings after one.
    pub issues: Vec<CaptureIssueDto>,
    pub result: Option<CaptureResultDto>,
    /// The refusal that ended a run.
    pub error: Option<String>,
    /// Where an import will write, relative to the content root.
    pub folder: String,
}

/// One `photogrammetry://progress` event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CaptureProgressDto {
    #[ts(type = "number")]
    pub run: u64,
    /// `"load" | "sfm" | "dense" | "finish" | "write"`.
    pub stage: String,
    /// Its position in the pipeline, so a bar can show overall progress without
    /// a second copy of the stage order.
    pub stage_index: u32,
    /// How many stages there are.
    pub stages: u32,
    /// `"started" | "progress" | "finished" | "failed" | "cancelled"`.
    pub phase: String,
    #[ts(type = "number")]
    pub done: u64,
    #[ts(type = "number")]
    pub total: u64,
    pub detail: String,
    pub error: Option<String>,
}

/// The preview, as data-URL PNGs.
///
/// Two images because the offscreen path draws **geometry**: binding a real
/// base-colour texture in the preview session is the standing P7 follow-up, so
/// the atlas is shown beside the render through the CPU texture door the Content
/// Drawer already uses. Either may be absent — the geometry needs a GPU adapter
/// and the atlas does not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CapturePreviewDto {
    /// The shaded mesh, rendered offscreen.
    pub geometry: Option<String>,
    /// The baked base-colour atlas, decoded on the CPU.
    pub albedo: Option<String>,
    /// Why the geometry preview is absent.
    pub error: Option<String>,
    pub size: u32,
}

/// What an import wrote.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct CaptureImportDto {
    pub mesh: String,
    pub albedo: String,
    pub normal: String,
    pub orm: String,
    pub material: String,
    /// The content sub-folder they landed in.
    pub folder: String,
    /// The asset name they were written under.
    pub name: String,
    /// Things a caller needs to know now that the scan is on disk — the
    /// placeholder-cube note among them.
    pub notes: Vec<String>,
}

impl CaptureSettingsDto {
    /// The wire form of a Ring-1 configuration.
    pub fn from_config(cfg: &crate::capture::CaptureConfig) -> Self {
        Self {
            camera: CaptureCameraDto {
                focal_ratio: cfg.camera.focal_ratio,
                k1: cfg.camera.k1,
                k2: cfg.camera.k2,
            },
            metres_per_unit: cfg.finish.metres_per_unit,
            target_triangles: cfg.finish.target_triangles as u32,
            atlas_size: cfg.finish.bake.size,
            ao_rays: cfg.finish.bake.ao_rays as u32,
            trim_unseen: cfg.finish.trim_unseen,
            delight: cfg.finish.delight.enabled,
            roughness: cfg.finish.roughness,
            metallic: cfg.finish.metallic,
        }
    }

    /// Overlay these settings onto `base`, leaving every field the wizard does
    /// not expose exactly where it was.
    ///
    /// An overlay rather than a construction, because the solvers' committed
    /// constants are not on the wire and a `CaptureConfig::default()` here would
    /// silently reset any that a future caller had set.
    pub fn to_config(&self, base: &crate::capture::CaptureConfig) -> crate::capture::CaptureConfig {
        let mut cfg = base.clone();
        cfg.camera.focal_ratio = self.camera.focal_ratio;
        cfg.camera.k1 = self.camera.k1;
        cfg.camera.k2 = self.camera.k2;
        cfg.finish.metres_per_unit = self.metres_per_unit;
        cfg.finish.target_triangles = self.target_triangles as usize;
        cfg.finish.bake.size = self.atlas_size;
        cfg.finish.bake.ao_rays = self.ao_rays as usize;
        cfg.finish.trim_unseen = self.trim_unseen;
        cfg.finish.delight.enabled = self.delight;
        cfg.finish.roughness = self.roughness;
        cfg.finish.metallic = self.metallic;
        cfg
    }
}

impl CaptureIssueDto {
    /// The wire form of a finding — its severity, its stage and its own words.
    pub fn from_issue(issue: &crate::capture::CaptureIssue) -> Self {
        Self {
            severity: issue.severity().name().to_string(),
            stage: issue.stage().name().to_string(),
            message: issue.to_string(),
        }
    }
}

impl CoverageDto {
    /// The wire form of a coverage report.
    pub fn from_report(report: &crate::capture::CoverageReport) -> Self {
        Self {
            triangles: report.triangles as u32,
            views: report
                .views
                .iter()
                .map(|v| CoverageViewDto {
                    view: v.view,
                    photo: v.photo.clone(),
                    registered: v.registered,
                    triangles_seen: v.triangles_seen as u32,
                    fraction: v.fraction,
                })
                .collect(),
            seen_by_none: report.seen_by_none as u32,
            seen_by_one: report.seen_by_one as u32,
            seen_by_two_or_more: report.seen_by_two_or_more as u32,
            unseen_texels: report.unseen_texels as u32,
            covered_texels: report.covered_texels as u32,
            covered_fraction: report.covered_fraction(),
            overlap_fraction: report.overlap_fraction(),
        }
    }
}

impl CaptureResultDto {
    /// The wire form of a finished product, at the scale it was finished with.
    pub fn from_product(product: &crate::capture::CaptureProduct, metres_per_unit: f64) -> Self {
        let sfm = &product.reconstruction.report;
        let finish = &product.finished.report;
        Self {
            registered: sfm.registered as u32,
            views: sfm.views as u32,
            points: sfm.points as u32,
            reprojection_rms_px: sfm.reprojection_rms_px,
            dense_triangles: finish.dense_triangles as u32,
            voxel_size: finish.voxel_size,
            triangles: finish.final_triangles as u32,
            vertices: finish.final_vertices as u32,
            charts: finish.charts as u32,
            atlas_coverage: finish.atlas_coverage,
            extent_units: product.extent_units,
            extent_metres: product.extent_units * metres_per_unit,
            coverage: CoverageDto::from_report(&product.coverage),
            elapsed_ms: product.elapsed_ms.to_vec(),
        }
    }
}

impl CaptureProgressDto {
    /// The wire form of one progress event.
    pub fn from_progress(event: &crate::capture::CaptureProgress) -> Self {
        Self {
            run: event.run,
            stage: event.stage.name().to_string(),
            stage_index: event.stage.index() as u32,
            stages: crate::capture::CaptureStage::ALL.len() as u32,
            phase: event.phase.name().to_string(),
            done: event.done,
            total: event.total,
            detail: event.detail.clone(),
            error: event.error.clone(),
        }
    }
}

#[cfg(test)]
mod capture_ipc_tests {
    use super::*;
    use crate::capture::CaptureConfig;

    /// Every knob the panel offers survives the round trip, and every one it
    /// does NOT offer is left exactly where the base config had it.
    #[test]
    fn the_settings_overlay_carries_what_it_shows_and_moves_nothing_else() {
        let mut base = CaptureConfig::default();
        // A field the wizard deliberately does not expose.
        base.sfm.max_reprojection_px = 3.25;
        base.dense.min_cameras = 5;
        base.finish.seam_smoothing_passes = 7;

        let mut dto = CaptureSettingsDto::from_config(&base);
        dto.camera.focal_ratio = 0.9375;
        dto.camera.k1 = -0.09;
        dto.camera.k2 = 0.02;
        dto.metres_per_unit = 0.25;
        dto.target_triangles = 12_345;
        dto.atlas_size = 512;
        dto.ao_rays = 8;
        dto.trim_unseen = false;
        dto.delight = true;
        dto.roughness = 0.4;
        dto.metallic = 0.6;

        let cfg = dto.to_config(&base);
        assert_eq!(cfg.camera.focal_ratio, 0.9375);
        assert_eq!((cfg.camera.k1, cfg.camera.k2), (-0.09, 0.02));
        assert_eq!(cfg.finish.metres_per_unit, 0.25);
        assert_eq!(cfg.finish.target_triangles, 12_345);
        assert_eq!(cfg.finish.bake.size, 512);
        assert_eq!(cfg.finish.bake.ao_rays, 8);
        assert!(!cfg.finish.trim_unseen);
        assert!(cfg.finish.delight.enabled);
        assert_eq!((cfg.finish.roughness, cfg.finish.metallic), (0.4, 0.6));
        // The unexposed three are untouched.
        assert_eq!(cfg.sfm.max_reprojection_px, 3.25);
        assert_eq!(cfg.dense.min_cameras, 5);
        assert_eq!(cfg.finish.seam_smoothing_passes, 7);
        // …and the round trip is a fixed point.
        assert_eq!(CaptureSettingsDto::from_config(&cfg), dto);
    }

    /// A finding's wire form is its own severity, stage and sentence — three
    /// strings the panel groups by, and none of them re-derived.
    #[test]
    fn a_finding_crosses_the_wire_with_its_severity_and_its_stage() {
        let issue = crate::capture::CaptureIssue::SingleCoverage {
            triangles: 12,
            examined: 100,
        };
        let dto = CaptureIssueDto::from_issue(&issue);
        assert_eq!(dto.severity, "warning");
        assert_eq!(dto.stage, "finish");
        assert_eq!(dto.message, issue.to_string());

        let blocking = crate::capture::CaptureIssue::TooFewPhotos {
            given: 1,
            required: 3,
        };
        assert_eq!(CaptureIssueDto::from_issue(&blocking).severity, "blocking");
        assert_eq!(CaptureIssueDto::from_issue(&blocking).stage, "load");
    }

    /// The progress event carries the stage's position and the pipeline's
    /// length, so a bar does not need a second copy of the stage order.
    #[test]
    fn a_progress_event_carries_where_it_is_in_the_pipeline() {
        let event = crate::capture::CaptureProgress {
            run: 3,
            stage: crate::capture::CaptureStage::Dense,
            phase: crate::capture::CapturePhase::Progress,
            done: 2,
            total: 5,
            detail: "depth maps".into(),
            error: None,
        };
        let dto = CaptureProgressDto::from_progress(&event);
        assert_eq!(dto.stage, "dense");
        assert_eq!(dto.stage_index, 2);
        assert_eq!(dto.stages, 5);
        assert_eq!(dto.phase, "progress");
        assert_eq!((dto.done, dto.total), (2, 5));
    }
}

#[cfg(test)]
mod mixer_tests {
    use super::*;
    use inf_audio::mixer::{Bus, MIXER_SCHEMA_VERSION};
    use inf_audio::{Effect, MixerConfig};

    fn cfg(buses: Vec<Bus>) -> MixerConfig {
        MixerConfig {
            schema_version: MIXER_SCHEMA_VERSION,
            buses,
        }
    }

    #[test]
    fn default_config_validates() {
        assert!(validate_mixer(&MixerConfig::default()).is_ok());
    }

    #[test]
    fn rejects_missing_master() {
        let c = cfg(vec![Bus::new("sfx", None)]);
        assert!(validate_mixer(&c).unwrap_err().contains("master"));
    }

    #[test]
    fn rejects_master_with_parent() {
        let c = cfg(vec![
            Bus::new("root", None),
            Bus::new("master", Some("root")),
        ]);
        assert!(validate_mixer(&c).unwrap_err().contains("master"));
    }

    #[test]
    fn rejects_duplicate_name() {
        let c = cfg(vec![
            Bus::new("master", None),
            Bus::new("sfx", Some("master")),
            Bus::new("sfx", Some("master")),
        ]);
        assert!(validate_mixer(&c).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn rejects_empty_name() {
        let c = cfg(vec![
            Bus::new("master", None),
            Bus::new("   ", Some("master")),
        ]);
        assert!(validate_mixer(&c).unwrap_err().contains("empty"));
    }

    #[test]
    fn rejects_unknown_parent() {
        let c = cfg(vec![
            Bus::new("master", None),
            Bus::new("sfx", Some("ghost")),
        ]);
        assert!(validate_mixer(&c).unwrap_err().contains("unknown parent"));
    }

    #[test]
    fn rejects_cycle() {
        // a → b → a (both non-root), plus a valid master.
        let c = cfg(vec![
            Bus::new("master", None),
            Bus::new("a", Some("b")),
            Bus::new("b", Some("a")),
        ]);
        assert!(validate_mixer(&c).unwrap_err().contains("cycle"));
    }

    #[test]
    fn dto_round_trips_effects() {
        let c = cfg(vec![
            Bus::new("master", None),
            Bus {
                name: "sfx".into(),
                parent: Some("master".into()),
                volume: 0.5,
                effects: vec![
                    Effect::Gain { db: -6.0 },
                    Effect::Lowpass { cutoff_hz: 800.0 },
                ],
            },
        ]);
        let dto = MixerConfigDto::from_config(&c);
        assert_eq!(dto.to_config(), c);
    }
}

#[cfg(test)]
mod character_wire_tests {
    use super::*;

    /// **The two wizard mirrors carry every field, in the right slot.**
    ///
    /// Totality is already a *compile-time* property — both conversions build the
    /// target with an exhaustive struct literal, so a field added to either model
    /// breaks the build rather than silently defaulting. What a literal cannot
    /// catch is a **transposition**: two fields of the same type crossed on the
    /// way out and crossed back on the way in round-trip perfectly against a
    /// default value, and against any value where the two happen to agree. So
    /// every number here is distinct.
    #[test]
    fn the_wizard_dtos_carry_every_field_of_their_models() {
        let params = inf_anim::BodyParams {
            height_m: 1.11,
            hip_height_ratio: 0.22,
            shoulder_height_ratio: 0.33,
            head_height_ratio: 0.44,
            spine_segments: 5,
            neck_segments: 6,
            shoulder_width_m: 0.77,
            hip_width_m: 0.88,
            upper_limb_ratio: 0.99,
            arm_length_ratio: 0.10,
            body_length_m: 1.21,
            head_forward_m: 1.32,
        };
        assert_eq!(BodyParamsDto::from_params(&params).to_params(), params);

        let gait = inf_anim::locomotion::GaitParams {
            walk_cadence_hz: 1.1,
            run_cadence_hz: 2.2,
            hip_swing_deg: 3.3,
            knee_flex_deg: 4.4,
            arm_swing_deg: 5.5,
            run_stride_scale: 6.6,
            bob_ratio: 7.7,
            idle_period_s: 8.8,
            idle_bob_ratio: 9.9,
            idle_pitch_deg: 10.1,
            keys_per_cycle: 11,
        };
        assert_eq!(GaitParamsDto::from_gait(&gait).to_gait(), gait);
    }

    /// The preview projection carries what the panel draws, off a real generated
    /// rig rather than a hand-built one.
    #[test]
    fn the_rig_projection_describes_a_real_generated_rig() {
        let preview =
            crate::character::preview_character(&crate::character::CharacterSpec::default(), None)
                .expect("the default spec previews");
        let dto = CharacterRigDto::from_preview(&preview);
        assert_eq!(dto.joints.len(), preview.joints.len());
        assert_eq!(dto.joints[0].name, "hips");
        assert_eq!(dto.legs.len(), 2);
        assert!(dto.limits >= 4);
        assert!(dto.body_vertices.is_some_and(|v| v > 0));
        assert!(dto.durations_s.iter().all(|d| *d > 0.0));
    }
}
