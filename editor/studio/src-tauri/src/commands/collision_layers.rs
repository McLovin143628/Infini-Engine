//! Collision-layer name-registry commands (P12.1).
//!
//! A per-project map of the 32 collision-layer bits → display names, persisted
//! under the content root (`<root>/.infinity/collision_layers.toml`) by
//! `inf-editor-core::collision_layers`. The Details panel keeps showing the raw
//! `u32` on the `Collider*.collision_memberships`/`collision_filter` fields; this
//! registry drives the "Collision Layers" management dialog (a named-bitmask
//! widget in Details is the follow-up). The exact sibling of the sorting-layer
//! commands (P8.2a).

use inf_editor_core::collision_layers::{CollisionLayer, CollisionLayers};
use inf_editor_core::ipc::CollisionLayerDto;
use tauri::State;

use super::assets::AssetState;

fn to_dtos(registry: CollisionLayers) -> Vec<CollisionLayerDto> {
    registry
        .layers
        .into_iter()
        .map(|l| CollisionLayerDto {
            bit: l.bit,
            name: l.name,
        })
        .collect()
}

/// The current project's collision-layer registry (defaults to a single
/// "Default" layer at bit 0 when none has been saved).
#[tauri::command]
pub async fn collision_layers_get(
    assets: State<'_, AssetState>,
) -> Result<Vec<CollisionLayerDto>, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    // A registry that exists but cannot be read is an error, not the default
    // (C4-38): reporting the default here is what let `collision_layers_set`
    // write it back over every named layer in the project.
    Ok(to_dtos(CollisionLayers::load_or_default(&root)?))
}

/// Replace the collision-layer registry (add/rename/remove rows). Normalized +
/// written deterministically; returns the persisted (normalized) list.
#[tauri::command]
pub async fn collision_layers_set(
    assets: State<'_, AssetState>,
    layers: Vec<CollisionLayerDto>,
) -> Result<Vec<CollisionLayerDto>, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    let mut registry = CollisionLayers {
        layers: layers
            .into_iter()
            .map(|d| CollisionLayer {
                bit: d.bit,
                name: d.name,
            })
            .collect(),
    };
    registry.normalize();
    registry.save(&root)?;
    Ok(to_dtos(registry))
}
