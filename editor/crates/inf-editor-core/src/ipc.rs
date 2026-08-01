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
/// tag string `"Lit"`/`"Unlit"`/`"Wireframe"`. `Wireframe` degrades to `Unlit` in
/// the renderer when the adapter lacks `POLYGON_MODE_LINE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum ViewModeDto {
    Lit,
    Unlit,
    Wireframe,
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
    pub mass_delta: f64,
    pub sediment_moved: Option<f64>,
    pub used_gpu: bool,
    pub steps: u32,
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
/// Units per architecture rule 6: the fog block is **SI metres** (`m⁻¹`
/// extinction and falloff, metre height); the disc sizes are **degrees of
/// angular diameter**; everything else is a dimensionless multiplier over a
/// physical constant that lives in `inf_render::atmosphere`.
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
            ..base
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
