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
    Bool { value: bool },
    Number { value: f64 },
    Text { value: String },
    Vec3 { value: Vec<f64> },
    Color { value: Vec<f32> },
    Enum { value: String, options: Vec<String> },
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
    /// "started" | "finished" | "failed".
    pub phase: String,
    /// GUIDs produced (on "finished").
    pub produced: Vec<String>,
    pub primary: Option<String>,
    pub cached: bool,
    pub error: Option<String>,
}

/// The `assets://changed` event payload — a version bump prompting a re-fetch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
pub struct AssetChanged {
    #[ts(type = "number")]
    pub version: u64,
}
