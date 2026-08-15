//! Diagnostics: memory budgets + crash reports (P15.1 / P15.2).
//!
//! This is Ring-1 (Tauri-free) so both the editor app and headless tests can use
//! it. Two pieces:
//!
//! * [`MemoryReport`] — a per-subsystem estimate of the editor's live memory
//!   (scene document, undo stack, terrain tiles, and renderer/runtime caches).
//!   The scene-derivable fields are computed here from a [`SceneDoc`]; the
//!   renderer/runtime cache fields (`texture_cache_bytes`, `vgeom_bytes`,
//!   `audio_bytes`) are left `0` for the Ring-2 `editor_diagnostics` command to
//!   fill from the attached viewport when it can (a documented handoff — the
//!   frontend can surface these in a status-bar readout later).
//!
//! * [`CrashReport`] — a structured crash record (engine version, OS/arch,
//!   panic location + message, a tail of recent log lines) plus
//!   [`write_crash_report`], which writes a **timestamped** file into a crash
//!   directory. The editor installs a panic hook that formats + writes one of
//!   these. There is deliberately **no** upload — opt-in telemetry is a config
//!   flag + docs only (see `docs/profiling.md`), never silent.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::scene::{serialize, SceneDoc};

/// A per-subsystem estimate of the editor's live memory footprint. All sizes are
/// byte estimates (best-effort, not an allocator-accurate measurement); counts
/// are exact. Serializable so a Ring-2 `stats`/`diagnostics` command can return
/// it to the frontend (or just log it).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryReport {
    /// Entities in the active scene document.
    pub entities: u64,
    /// Estimated in-memory size of the scene document (the deterministic `.inf_lvl`
    /// encoding of the world — includes transforms, components, and persisted
    /// terrain/tilemap payloads). The authoritative document-memory figure.
    pub scene_bytes: u64,
    /// Undo-stack depth (bounded by the history limit — the "undo stack depth"
    /// budget; a monotonically growing value here would be a leak).
    pub undo_depth: u64,
    /// Redo-stack depth.
    pub redo_depth: u64,
    /// **Estimated bytes the undo + redo stacks hold** (Hardening D), summed
    /// from each record's own payload — the terrain/voxel deltas' `memory_bytes`,
    /// which existed for exactly this and had no caller.
    ///
    /// This replaces a flat 512-bytes-per-entry charge that under-reported the
    /// single largest thing the editor holds by orders of magnitude: one sculpt
    /// stroke over a 1024² patch is megabytes, and the report called it 512
    /// bytes. It is still an *estimate* — see `EditCommand::memory_bytes` for
    /// what is charged exactly and what is charged by assumption.
    pub undo_bytes: u64,
    /// Total terrain tiles across all `Terrain` entities in the scene.
    pub terrain_tiles: u64,
    /// Estimated terrain heightfield + splat-weight bytes (a subset of
    /// `scene_bytes`; reported separately as a subsystem breakdown).
    pub terrain_bytes: u64,
    /// GPU texture-cache bytes. `0` unless a Ring-2 caller fills it from the
    /// renderer (handoff).
    pub texture_cache_bytes: u64,
    /// Virtualized-geometry buffer bytes. `0` unless filled by a Ring-2 caller.
    pub vgeom_bytes: u64,
    /// Audio engine bytes. `0` unless filled by a Ring-2 caller.
    pub audio_bytes: u64,
    /// Rough total live estimate: the scene document plus the renderer/runtime
    /// caches (terrain is already inside `scene_bytes`, so it is not re-added).
    pub total_estimate_bytes: u64,
}

impl MemoryReport {
    /// Compute the scene-derivable fields from `doc`. Renderer/runtime cache
    /// fields stay `0` for a Ring-2 caller to fill (see the module docs).
    pub fn for_scene(doc: &SceneDoc) -> Self {
        let entities = doc.order().len() as u64;
        let undo_depth = doc.undo_len() as u64;
        let redo_depth = doc.redo_len() as u64;

        // Terrain: sum tiles + estimate their heightfield/splat bytes.
        let mut terrain_tiles = 0u64;
        let mut terrain_bytes = 0u64;
        for &guid in doc.order() {
            if let Some((data, _origin)) = doc.terrain_data_and_origin(guid) {
                let tiles = data.tile_count() as u64;
                let res = data.tile_resolution() as u64;
                let samples = res.saturating_mul(res);
                terrain_tiles += tiles;
                // 4 bytes/sample height (f32) + 4 bytes/sample splat weights (rgba8).
                terrain_bytes += tiles.saturating_mul(samples).saturating_mul(8);
            }
        }

        // The document's in-memory size ≈ its deterministic serialized payload.
        let scene_bytes = serialize::encode(&serialize::to_scene_file(doc))
            .map(|b| b.len() as u64)
            .unwrap_or(0);

        let mut report = Self {
            entities,
            scene_bytes,
            undo_depth,
            redo_depth,
            undo_bytes: doc.undo_bytes() as u64,
            terrain_tiles,
            terrain_bytes,
            ..Default::default()
        };
        report.recompute_total();
        report
    }

    /// Recompute [`Self::total_estimate_bytes`] from the current fields. Call after
    /// a Ring-2 caller fills the renderer/runtime cache fields.
    pub fn recompute_total(&mut self) {
        self.total_estimate_bytes = self
            .scene_bytes
            .saturating_add(self.texture_cache_bytes)
            .saturating_add(self.vgeom_bytes)
            .saturating_add(self.audio_bytes)
            .saturating_add(self.undo_bytes);
    }

    /// A one-line human summary for the Output Log / a `stats` console command.
    pub fn summary(&self) -> String {
        format!(
            "memory: {} entities · scene {} KiB · undo {}/{} ({} KiB) · terrain {} tiles ({} KiB) · \
             tex {} KiB · vgeom {} KiB · audio {} KiB · ~total {} KiB",
            self.entities,
            self.scene_bytes / 1024,
            self.undo_depth,
            self.redo_depth,
            self.undo_bytes / 1024,
            self.terrain_tiles,
            self.terrain_bytes / 1024,
            self.texture_cache_bytes / 1024,
            self.vgeom_bytes / 1024,
            self.audio_bytes / 1024,
            self.total_estimate_bytes / 1024,
        )
    }
}

/// A structured crash record. `engine_version` is passed by the caller so the
/// figure is the *app's* version (`env!("CARGO_PKG_VERSION")` at the call site),
/// not this crate's.
#[derive(Debug, Clone)]
pub struct CrashReport {
    /// Which process crashed, e.g. `"inf-studio"` / `"inf-viewport"`.
    pub app: String,
    pub engine_version: String,
    pub os: String,
    pub arch: String,
    /// `file:line:col` or `<unknown>`.
    pub location: String,
    pub message: String,
    /// Optional adapter/GPU line (`wgpu::AdapterInfo` debug), if known.
    pub adapter: Option<String>,
    /// Tail of recent log lines (most-recent last).
    pub log_tail: Vec<String>,
}

impl CrashReport {
    /// Build a report from a panic hook's info (the app supplies its name +
    /// version + optional adapter + a log tail).
    pub fn from_panic(
        app: impl Into<String>,
        engine_version: impl Into<String>,
        info: &std::panic::PanicHookInfo<'_>,
        adapter: Option<String>,
        log_tail: Vec<String>,
    ) -> Self {
        let message = panic_message(info.payload());
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".to_string());
        Self {
            app: app.into(),
            engine_version: engine_version.into(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            location,
            message,
            adapter,
            log_tail,
        }
    }

    /// Format the report as the plain-text crash file body.
    pub fn format(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("{} CRASH\n", self.app));
        out.push_str(&format!("engine  : {}\n", self.engine_version));
        out.push_str(&format!("os      : {} ({})\n", self.os, self.arch));
        if let Some(adapter) = &self.adapter {
            out.push_str(&format!("adapter : {adapter}\n"));
        }
        out.push_str(&format!("location: {}\n", self.location));
        out.push_str(&format!("message : {}\n", self.message));
        out.push_str(&format!(
            "--- last {} log line(s) ---\n",
            self.log_tail.len()
        ));
        for line in &self.log_tail {
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

/// Write `report` to a **timestamped** file inside `dir` (created if needed) and
/// return the path. Filename: `crash-<unix_millis>.txt`, so repeated crashes
/// never overwrite each other.
pub fn write_crash_report(dir: &Path, report: &CrashReport) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("crash-{millis}.txt"));
    std::fs::write(&path, report.format())?;
    Ok(path)
}

/// Extract a panic payload as a string (`&str` / `String`, else a placeholder).
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;

    #[test]
    fn memory_report_reflects_the_scene() {
        let mut doc = SceneDoc::with_demo();
        let base = MemoryReport::for_scene(&doc);
        assert!(base.entities > 0, "demo scene has entities");
        assert!(base.scene_bytes > 0, "scene has a nonzero encoded size");
        assert_eq!(
            base.total_estimate_bytes, base.scene_bytes,
            "no undo/caches yet"
        );

        // An edit grows the entity count + adds an undo entry.
        doc.edit_create(SpawnKind::Cube, "Cube", None);
        let after = MemoryReport::for_scene(&doc);
        assert_eq!(after.entities, base.entities + 1);
        assert_eq!(after.undo_depth, 1);
        assert!(
            after.total_estimate_bytes > after.scene_bytes,
            "undo counts toward total"
        );
    }

    #[test]
    fn crash_report_formats_all_fields() {
        let report = CrashReport {
            app: "inf-test".into(),
            engine_version: "0.1.0".into(),
            os: "linux".into(),
            arch: "x86_64".into(),
            location: "src/foo.rs:1:2".into(),
            message: "boom".into(),
            adapter: Some("TestGPU".into()),
            log_tail: vec!["line a".into(), "line b".into()],
        };
        let text = report.format();
        assert!(text.contains("inf-test CRASH"));
        assert!(text.contains("engine  : 0.1.0"));
        assert!(text.contains("os      : linux (x86_64)"));
        assert!(text.contains("adapter : TestGPU"));
        assert!(text.contains("location: src/foo.rs:1:2"));
        assert!(text.contains("message : boom"));
        assert!(text.contains("line a"));
        assert!(text.contains("line b"));
    }

    #[test]
    fn write_crash_report_creates_a_timestamped_file() {
        let dir = tempfile::tempdir().unwrap();
        let crash_dir = dir.path().join("crashes");
        let report = CrashReport {
            app: "inf-test".into(),
            engine_version: "0.1.0".into(),
            os: "os".into(),
            arch: "arch".into(),
            location: "<unknown>".into(),
            message: "kaboom".into(),
            adapter: None,
            log_tail: vec![],
        };
        let path = write_crash_report(&crash_dir, &report).unwrap();
        assert!(path.exists());
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(
            name.starts_with("crash-") && name.ends_with(".txt"),
            "got {name}"
        );
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("kaboom"));
    }
}
