//! **The drag-to-viewport payload contract** (Wave E, batch B).
//!
//! A Content-Drawer drag that ends over the native viewport cannot use HTML5
//! drag-and-drop — the hole is a child window and no `dragover` will ever reach
//! it — so the drop crosses as `ViewportDrop { x, y, payload: String }` and the
//! payload is a hand-rolled string. This module is that string's parser.
//!
//! # Why it lives in Ring 1
//!
//! Verbatim the reasoning that put [`crate::render_assets`] here:
//! `inf_viewport::host` is `#[cfg(any(windows, target_os = "macos"))]`, so a
//! parser written there is invisible to the Linux CI leg. Keeping it here means
//! the cases below are exercised on all three OSes and the host keeps a call
//! site.
//!
//! # The format, and why it grew a kind
//!
//! ```text
//! spawn:<snake_case SpawnKind>      — the Place Actors drag
//! asset:<kind>:<id>:<name>          — a Content Drawer asset (current)
//! asset:<id>:<name>                 — the same, before Wave E (accepted)
//! ```
//!
//! The kind is new. Before it the viewport thread could not tell a dropped mesh
//! from a dropped texture — it has no asset database — so **every** asset drop
//! spawned a placeholder cube and `MeshRef::asset` was left unwritten, which is
//! why a dragged-in prop drew a cube for ever and had no mesh for the Model
//! Editor to open. Sending the kind costs one string segment and needs no asset
//! lookup on the viewport thread.
//!
//! The legacy two-segment form is still accepted, and told apart by whether the
//! segment after `asset:` parses as a UUID — a kind slug never does.

use uuid::Uuid;

use crate::ipc::SpawnKind;
use crate::scene::SceneDoc;

/// **Create the entity a dropped Content-Drawer asset becomes — the ONE door
/// both drop paths use** (Wave E audit, A6).
///
/// There are two of them: `EngineHost::spawn_drop` (a drag that lands on the
/// native hole) and `scene_spawn_asset` (the Ring-2 command). Wave E taught the
/// binding rule to each of them separately, and the arm that certified the fix
/// called neither — it called `SceneDoc::edit_create_mesh_asset` directly, so
/// its own comment ("delete the `mesh_asset()` branch in `spawn_drop` and the
/// second assertion fails") named a mutation it could not catch, and the two
/// doors were free to disagree. This is the P22 law: **one door for two paths.**
///
/// `mesh_asset` is `Some` only for a payload that named the `mesh` kind AND a
/// parseable id ([`DropPayload::mesh_asset`]); everything else keeps the honest
/// placeholder cube, because binding a texture's GUID to a `MeshRef` renders a
/// missing mesh instead of a cube that says "I am a placeholder".
pub fn spawn_asset_entity(
    doc: &mut SceneDoc,
    name: &str,
    mesh_asset: Option<Uuid>,
    parent: Option<Uuid>,
) -> Uuid {
    match mesh_asset {
        Some(asset) => doc.edit_create_mesh_asset(name, asset, parent),
        None => doc.edit_create(SpawnKind::Cube, name, parent),
    }
}

/// A parsed viewport drop payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropPayload<'a> {
    /// `spawn:<kind>` — place that primitive/light/camera.
    Spawn { kind: &'a str },
    /// `asset:…` — a Content-Drawer asset.
    Asset {
        /// The `AssetKind` slug ("mesh", "texture", …); empty when the payload
        /// used the pre-Wave-E form and did not carry one.
        kind: &'a str,
        /// The asset GUID, when the payload carried a parseable one.
        id: Option<Uuid>,
        /// Display name for the spawned entity ("" when absent).
        name: &'a str,
    },
}

impl DropPayload<'_> {
    /// The mesh asset this drop should bind to the entity it spawns, if any.
    ///
    /// `Some` only for a payload that names **both** the `mesh` kind and a
    /// parseable id: binding on the id alone would bind a texture's GUID to a
    /// `MeshRef`, which resolves to nothing and renders as a missing mesh
    /// instead of the honest placeholder cube.
    pub fn mesh_asset(&self) -> Option<Uuid> {
        match self {
            DropPayload::Asset {
                kind: "mesh",
                id: Some(id),
                ..
            } => Some(*id),
            _ => None,
        }
    }
}

/// Parse a drop payload. Anything unrecognized is an `Asset` with no id — the
/// pre-Wave-E fallback, which spawns a named placeholder rather than refusing.
pub fn parse_drop_payload(payload: &str) -> DropPayload<'_> {
    if let Some(kind) = payload.strip_prefix("spawn:") {
        return DropPayload::Spawn { kind };
    }
    let rest = payload.strip_prefix("asset:").unwrap_or("");
    let mut it = rest.splitn(3, ':');
    let first = it.next().unwrap_or("");
    // A kind slug never parses as a UUID, which is what tells the two forms
    // apart without a version marker.
    match Uuid::parse_str(first) {
        Ok(id) => DropPayload::Asset {
            kind: "",
            id: Some(id),
            name: it.next().unwrap_or(""),
        },
        Err(_) => {
            let id = it.next().and_then(|s| Uuid::parse_str(s).ok());
            DropPayload::Asset {
                kind: first,
                id,
                name: it.next().unwrap_or(""),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "6f1a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8";

    #[test]
    fn a_spawn_payload_carries_its_kind() {
        assert_eq!(
            parse_drop_payload("spawn:point_light"),
            DropPayload::Spawn {
                kind: "point_light"
            }
        );
    }

    #[test]
    fn the_current_form_carries_kind_id_and_name() {
        let payload = format!("asset:mesh:{ID}:Barrel");
        let p = parse_drop_payload(&payload);
        assert_eq!(
            p,
            DropPayload::Asset {
                kind: "mesh",
                id: Some(Uuid::parse_str(ID).unwrap()),
                name: "Barrel",
            }
        );
        assert_eq!(p.mesh_asset(), Some(Uuid::parse_str(ID).unwrap()));
    }

    #[test]
    fn the_pre_wave_e_form_still_parses_and_binds_nothing() {
        let payload = format!("asset:{ID}:Barrel");
        let p = parse_drop_payload(&payload);
        assert_eq!(
            p,
            DropPayload::Asset {
                kind: "",
                id: Some(Uuid::parse_str(ID).unwrap()),
                name: "Barrel",
            }
        );
        // No kind ⇒ no binding: an old payload keeps the old behaviour exactly.
        assert_eq!(p.mesh_asset(), None);
    }

    /// The distinction that matters: a texture dropped on the viewport must NOT
    /// have its GUID written into a `MeshRef`.
    #[test]
    fn only_a_mesh_binds() {
        for kind in ["texture", "material", "blueprint", "audio", ""] {
            let payload = format!("asset:{kind}:{ID}:X");
            let p = parse_drop_payload(&payload);
            assert_eq!(p.mesh_asset(), None, "{kind} must not bind");
        }
    }

    #[test]
    fn a_name_with_no_id_and_a_bare_payload_degrade_to_a_placeholder() {
        assert_eq!(
            parse_drop_payload("asset:mesh::Barrel"),
            DropPayload::Asset {
                kind: "mesh",
                id: None,
                name: "Barrel",
            }
        );
        // A payload with no recognized prefix carries nothing: it is neither a
        // spawn nor an asset reference, and the pre-Wave-E code discarded it the
        // same way (a named placeholder needed the `asset:` prefix even then).
        assert_eq!(
            parse_drop_payload("something-else"),
            DropPayload::Asset {
                kind: "",
                id: None,
                name: "",
            }
        );
    }

    /// The shared door, both branches (Wave E audit, A6). Deleting the `Some`
    /// arm's binding — the mutation the drop-gap arm claimed to catch and could
    /// not — turns this red, and both production drop paths route through here.
    #[test]
    fn the_shared_door_binds_a_mesh_and_only_a_mesh() {
        let mut doc = SceneDoc::new();
        let asset = Uuid::from_u128(0x5EED);

        let bound = spawn_asset_entity(&mut doc, "Barrel", Some(asset), None);
        assert_eq!(doc.mesh_asset_of(bound), Some(asset));

        // A non-mesh keeps the honest placeholder: a real, named, selectable
        // entity with NO asset bound.
        let placeholder = spawn_asset_entity(&mut doc, "Bricks", None, None);
        assert_eq!(doc.mesh_asset_of(placeholder), None);
        assert_eq!(doc.display_name(placeholder), "Bricks");
        assert!(doc.primitive_of(placeholder).is_some());
    }

    #[test]
    fn a_name_containing_a_colon_survives_intact() {
        // `splitn(3, ':')` gives the name the whole tail, so a name with a colon
        // in it is not silently truncated.
        let payload = format!("asset:mesh:{ID}:Barrel: rusty");
        let p = parse_drop_payload(&payload);
        assert!(matches!(
            p,
            DropPayload::Asset {
                name: "Barrel: rusty",
                ..
            }
        ));
    }
}
