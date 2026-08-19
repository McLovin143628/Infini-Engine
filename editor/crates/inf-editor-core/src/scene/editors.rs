//! **Which editors can open this object** (Wave E, batch B).
//!
//! One question, asked once, answered in Ring 1: given an entity, which assets
//! does it carry that some editor knows how to open — and when the answer is
//! *none*, **why not**.
//!
//! # Why the refusal is a field and not an absence
//!
//! The P21 law ("a gameplay refusal must be a value, not a failure") applied to
//! UX. Every alternative to a named reason produces a worse editor:
//!
//! * a dead menu item that opens an empty Model Editor;
//! * a menu item that silently does nothing, so the user concludes the feature
//!   is broken rather than inapplicable;
//! * hiding the item, which teaches nothing — the user still does not know that
//!   a built-in primitive has no mesh asset, or what to do about it.
//!
//! So [`EntityEditorsDto::no_editor_reason`] is `Some` exactly when no route
//! exists, and it names the remedy. The invariant is asserted, not just
//! described: `a_route_or_a_reason_but_never_neither`.
//!
//! # Why the frontend does not read components itself
//!
//! It cannot. `MeshRef::asset`, `SkeletalMesh::{mesh, skeleton}` and
//! `ActorClass` are `#[reflect(ignore)]` or unreflected, so the reflection
//! walker that builds the Details grid never sees them. `SceneDoc`'s four
//! Wave-E accessors are the only way through, and this module is the only
//! caller — so the routing rules live in one place and the viewport menu, the
//! outliner menu, the Details buttons and the command palette cannot drift.

use uuid::Uuid;

use crate::ipc::EntityEditorsDto;
use crate::scene::SceneDoc;

/// Answer the question for one entity.
pub fn entity_editors(doc: &SceneDoc, guid: Uuid) -> EntityEditorsDto {
    let mesh = doc.mesh_asset_of(guid);
    let skeletal = doc.skeletal_mesh_of(guid);
    let skeletal_mesh = skeletal.and_then(|(m, _)| m);
    let skeleton = skeletal.and_then(|(_, s)| s);
    let material = doc.material_asset_of(guid);
    let actor_class = doc.actor_class_of(guid);
    let primitive = doc.primitive_of(guid).map(|p| format!("{p:?}"));
    let kind = doc.kind_of_guid(guid);

    let has_route = mesh.is_some()
        || skeletal_mesh.is_some()
        || skeleton.is_some()
        || material.is_some()
        || actor_class.is_some();

    let no_editor_reason = if has_route {
        None
    } else if primitive.is_some() {
        Some(
            "This is a built-in primitive — it has no mesh asset to edit. Import or create a \
             mesh in the Content Drawer and drop it onto this actor."
                .to_string(),
        )
    } else if skeletal.is_some() {
        Some(
            "This character has no skeletal mesh or skeleton bound yet — drag a mesh and a \
             skeleton onto it from the Content Drawer."
                .to_string(),
        )
    } else {
        Some(match kind.as_str() {
            "Terrain" => "Terrain is sculpted with the viewport's terrain tools, not the Model \
                          Editor."
                .to_string(),
            "Folder" => "A folder groups other actors; it has no asset of its own.".to_string(),
            other => format!(
                "A {other} has no editable asset. Bind a blueprint class or drop a \
                              mesh onto it to give it something to open."
            ),
        })
    };

    EntityEditorsDto {
        entity: guid.to_string(),
        name: doc.display_name(guid),
        kind,
        mesh: mesh.map(|g| g.to_string()),
        skeletal_mesh: skeletal_mesh.map(|g| g.to_string()),
        skeleton: skeleton.map(|g| g.to_string()),
        material: material.map(|g| g.to_string()),
        actor_class: actor_class.map(|g| g.to_string()),
        primitive,
        no_editor_reason,
    }
}

/// The batch form — what the Ring-2 command and every context menu ask for.
///
/// Unknown guids are **skipped**, not reported as errors: a context menu built
/// from a selection that changed under it is a normal race, not a failure.
pub fn entity_editors_many(doc: &SceneDoc, guids: &[Uuid]) -> Vec<EntityEditorsDto> {
    guids
        .iter()
        .filter(|g| doc.world().entity_of(**g).is_some())
        .map(|g| entity_editors(doc, *g))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;
    use inf_ecs::components::{MeshRef, SkeletalMesh};

    fn doc_with_cube() -> (SceneDoc, Uuid) {
        let mut doc = SceneDoc::new();
        let guid = doc.edit_create(SpawnKind::Cube, "Prop", None);
        (doc, guid)
    }

    /// The vacuous-check law: assert the REASON TEXT, not merely that there are
    /// no routes. "Zero routes" is satisfied perfectly by a resolver that
    /// returns nothing for everything.
    #[test]
    fn a_bare_primitive_has_no_route_and_a_named_reason() {
        let (doc, guid) = doc_with_cube();
        let e = entity_editors(&doc, guid);
        assert_eq!(e.mesh, None);
        assert_eq!(e.skeletal_mesh, None);
        assert_eq!(e.actor_class, None);
        assert_eq!(e.primitive.as_deref(), Some("Cube"));
        let reason = e.no_editor_reason.expect("a primitive must say why");
        assert!(reason.contains("built-in primitive"), "{reason}");
        assert!(reason.contains("Content Drawer"), "{reason}");
    }

    #[test]
    fn a_dropped_mesh_asset_reports_its_mesh_and_no_reason() {
        let mut doc = SceneDoc::new();
        let asset = Uuid::from_u128(0x1234);
        let guid = doc.edit_create_mesh_asset("Barrel", asset, None);
        let e = entity_editors(&doc, guid);
        assert_eq!(e.mesh.as_deref(), Some(asset.to_string().as_str()));
        assert_eq!(e.no_editor_reason, None);
        // The placeholder primitive is still reported — the renderer falls back
        // to it — but it is not what routes.
        assert_eq!(e.primitive.as_deref(), Some("Cube"));
    }

    #[test]
    fn a_character_reports_its_rig_and_reads_as_a_skeletal_mesh() {
        let mut doc = SceneDoc::new();
        let mesh = Uuid::from_u128(0xAA);
        let skel = Uuid::from_u128(0xBB);
        let guid = doc.edit_create(SpawnKind::Empty, "Hero", None);
        let entity = doc.world_mut().entity_of(guid).unwrap();
        doc.world_mut()
            .world_mut()
            .entity_mut(entity)
            .insert(SkeletalMesh {
                mesh: Some(mesh),
                skeleton: Some(skel),
            });
        let e = entity_editors(&doc, guid);
        assert_eq!(e.skeletal_mesh.as_deref(), Some(mesh.to_string().as_str()));
        assert_eq!(e.skeleton.as_deref(), Some(skel.to_string().as_str()));
        assert_eq!(e.no_editor_reason, None);
        // `kind_of` used to report "Static Mesh" (or "Actor") for every rigged
        // character — nothing in the type column ever said "Skeletal Mesh".
        assert_eq!(e.kind, "Skeletal Mesh");
    }

    #[test]
    fn a_rigless_skeletal_mesh_says_what_is_missing() {
        let mut doc = SceneDoc::new();
        let guid = doc.edit_create(SpawnKind::Empty, "Hero", None);
        let entity = doc.world_mut().entity_of(guid).unwrap();
        doc.world_mut()
            .world_mut()
            .entity_mut(entity)
            .insert(SkeletalMesh::default());
        let reason = entity_editors(&doc, guid)
            .no_editor_reason
            .expect("a rigless character must say why");
        assert!(reason.contains("no skeletal mesh"), "{reason}");
    }

    #[test]
    fn a_bound_blueprint_class_is_a_route_on_its_own() {
        let (mut doc, guid) = doc_with_cube();
        let class = Uuid::from_u128(0xC1A55);
        assert_eq!(doc.edit_apply_actor(&[guid], class), 1);
        let e = entity_editors(&doc, guid);
        assert_eq!(e.actor_class.as_deref(), Some(class.to_string().as_str()));
        assert_eq!(e.no_editor_reason, None);
    }

    /// The invariant, over every kind the document can spawn: **a route or a
    /// reason, never neither, never both.** This is the arm that fails if a new
    /// component type is given an asset link and forgotten here.
    #[test]
    fn a_route_or_a_reason_but_never_neither() {
        let mut doc = SceneDoc::new();
        for kind in [
            SpawnKind::Empty,
            SpawnKind::Cube,
            SpawnKind::Sphere,
            SpawnKind::Plane,
            SpawnKind::Cylinder,
            SpawnKind::PointLight,
            SpawnKind::DirectionalLight,
            SpawnKind::Camera,
            SpawnKind::Terrain,
        ] {
            let guid = doc.edit_create(kind, "", None);
            let e = entity_editors(&doc, guid);
            let has_route = e.mesh.is_some()
                || e.skeletal_mesh.is_some()
                || e.skeleton.is_some()
                || e.material.is_some()
                || e.actor_class.is_some();
            assert_ne!(
                has_route,
                e.no_editor_reason.is_some(),
                "{kind:?}: routes={has_route} reason={:?}",
                e.no_editor_reason
            );
            if let Some(reason) = &e.no_editor_reason {
                assert!(!reason.trim().is_empty(), "{kind:?}: empty reason");
            }
        }
    }

    #[test]
    fn the_batch_form_skips_guids_that_no_longer_exist() {
        let (doc, guid) = doc_with_cube();
        let rows = entity_editors_many(&doc, &[guid, Uuid::from_u128(0xDEAD)]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].entity, guid.to_string());
    }

    /// A primitive that ALSO carries a mesh asset routes; the reason is the
    /// absence of every link, not the presence of a primitive.
    #[test]
    fn a_primitive_with_a_mesh_asset_still_routes() {
        let (mut doc, guid) = doc_with_cube();
        let entity = doc.world_mut().entity_of(guid).unwrap();
        doc.world_mut()
            .world_mut()
            .entity_mut(entity)
            .insert(MeshRef {
                asset: Some(Uuid::from_u128(9)),
                ..Default::default()
            });
        assert_eq!(entity_editors(&doc, guid).no_editor_reason, None);
    }
}
