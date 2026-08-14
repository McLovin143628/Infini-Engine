//! Sorting-layer registry commands (P8.2a).
//!
//! A small per-project map of `i32` sorting-layer ids → display names, persisted
//! under the content root (`<root>/.infinity/sorting_layers.toml`) by
//! `inf-editor-core::sorting`. The Details panel keeps showing the raw `i32` on
//! the `Sprite.sorting_layer` field; this registry drives the "Sorting Layers"
//! management dialog (a dropdown-widget upgrade in Details is a follow-up).

use inf_editor_core::ipc::SortingLayerDto;
use inf_editor_core::sorting::{SortingLayer, SortingLayers};
use tauri::State;

use super::assets::AssetState;

fn to_dtos(registry: SortingLayers) -> Vec<SortingLayerDto> {
    registry
        .layers
        .into_iter()
        .map(|l| SortingLayerDto {
            id: l.id,
            name: l.name,
        })
        .collect()
}

/// The current project's sorting-layer registry (defaults to a single
/// "Default" layer at id 0 when none has been saved).
#[tauri::command]
pub async fn layers_get(assets: State<'_, AssetState>) -> Result<Vec<SortingLayerDto>, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    // Unreadable is an error, not the default (C4-38) — see
    // `collision_layers_get`.
    Ok(to_dtos(SortingLayers::load_or_default(&root)?))
}

/// Replace the sorting-layer registry (add/rename/remove rows). Normalized +
/// written deterministically; returns the persisted (normalized) list.
#[tauri::command]
pub async fn layers_set(
    assets: State<'_, AssetState>,
    layers: Vec<SortingLayerDto>,
) -> Result<Vec<SortingLayerDto>, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    let mut registry = SortingLayers {
        layers: layers
            .into_iter()
            .map(|d| SortingLayer {
                id: d.id,
                name: d.name,
            })
            .collect(),
    };
    registry.normalize();
    registry.save(&root)?;
    Ok(to_dtos(registry))
}
