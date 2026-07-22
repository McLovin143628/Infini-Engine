//! Audio mixer commands (E-P9): read + write the project's named-bus
//! [`inf_audio::MixerConfig`] for the Audio Mixer panel.
//!
//! The mixer is **pure data** that folds into per-bus gains (the audio
//! command-queue doctrine — see `inf_audio::mixer`), persisted at
//! `<project_root>/.infinity/mixer.toml` (the same `.infinity` directory as
//! `collections.toml`, resolved as the parent of the asset content root).
//!
//! `mixer_save` validates the edited config (unique non-empty names, a present +
//! rootless `master`, existing parents, no cycles — see
//! [`inf_editor_core::ipc::validate_mixer`]), writes the file via the Ring-0
//! `save` API, **live-applies** it to a running Simulate session
//! ([`SimState::apply_mixer`], mirroring `sim_start`'s mixer load), and emits
//! `audio://mixer-changed` so the panel re-syncs.

use std::path::PathBuf;

use inf_editor_core::ipc::{validate_mixer, MixerConfigDto};
use tauri::{AppHandle, Emitter, State};

use super::assets::AssetState;
use super::sim::SimState;

/// The project root that holds `.infinity/mixer.toml` — the parent of the asset
/// content root (mirrors `collections.rs` / `sim_start`'s mixer load). Falls back
/// to the content root itself when it has no parent.
fn project_root(assets: &AssetState) -> Result<PathBuf, String> {
    let content = assets.content_root().ok_or("no project is open")?;
    Ok(content.parent().map(|p| p.to_path_buf()).unwrap_or(content))
}

/// Load the project mixer (or the default when `mixer.toml` is absent).
#[tauri::command]
pub async fn mixer_get(assets: State<'_, AssetState>) -> Result<MixerConfigDto, String> {
    let root = project_root(&assets)?;
    let cfg = inf_audio::MixerConfig::load_or_default(&root)?;
    Ok(MixerConfigDto::from_config(&cfg))
}

/// Validate + persist the edited mixer, live-apply it to a running Simulate
/// session, and broadcast `audio://mixer-changed`.
#[tauri::command]
pub async fn mixer_save(
    app: AppHandle,
    config: MixerConfigDto,
    assets: State<'_, AssetState>,
    sim: State<'_, SimState>,
) -> Result<(), String> {
    let root = project_root(&assets)?;
    let cfg = config.to_config();
    validate_mixer(&cfg)?;
    cfg.save(&root)?;
    // Fold the new gains into a live Simulate session immediately (no-op when
    // stopped); the next `sim_start` reloads from disk regardless.
    sim.apply_mixer(cfg);
    let _ = app.emit("audio://mixer-changed", ());
    Ok(())
}
