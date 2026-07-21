//! The default authored scene — a small primitive+lights setup, exactly the
//! "author a primitive+lights scene" the Phase 3 gate calls for. Loaded into a
//! fresh document so the editor boots with something to select, transform, and
//! save (then round-trip byte-identically).

use glam::DVec3;
use inf_ecs::components::{Material, Transform};
use inf_ecs::math::{Color, Vec3d};

use crate::ipc::SpawnKind;
use crate::scene::SceneDoc;

/// Populate `doc` with the default scene and mark it clean (unsaved-but-pristine).
pub fn build(doc: &mut SceneDoc) {
    doc.set_title("Untitled");

    // Ground.
    let floor = doc.create(SpawnKind::Plane, "Floor", None);
    set_transform(
        doc,
        floor,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::ZERO,
            scale: Vec3d::new(20.0, 1.0, 20.0),
        },
    );
    set_color(doc, floor, Color::new(0.30, 0.32, 0.35, 1.0));

    // A little cluster of primitives under a "Props" folder.
    let props = doc.create(SpawnKind::Empty, "Props", None);
    let cube = doc.create(SpawnKind::Cube, "Cube", Some(props));
    set_transform(doc, cube, translate(-2.0, 0.5, 0.0));
    set_color(doc, cube, Color::new(0.80, 0.25, 0.22, 1.0));

    let sphere = doc.create(SpawnKind::Sphere, "Sphere", Some(props));
    set_transform(doc, sphere, translate(0.0, 0.6, -1.5));
    set_color(doc, sphere, Color::new(0.25, 0.55, 0.85, 1.0));

    let cyl = doc.create(SpawnKind::Cylinder, "Cylinder", Some(props));
    set_transform(doc, cyl, translate(2.0, 0.75, 0.5));
    set_color(doc, cyl, Color::new(0.30, 0.70, 0.35, 1.0));

    // Lighting.
    let lighting = doc.create(SpawnKind::Empty, "Lighting", None);
    let sun = doc.create(SpawnKind::DirectionalLight, "Sun", Some(lighting));
    set_transform(
        doc,
        sun,
        Transform {
            translation: Vec3d::new(0.0, 8.0, 0.0),
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::ONE,
        },
    );
    let fill = doc.create(SpawnKind::PointLight, "FillLight", Some(lighting));
    set_transform(doc, fill, translate(4.0, 3.0, 4.0));

    doc.world_mut().propagate();
    doc.mark_saved();
}

fn translate(x: f64, y: f64, z: f64) -> Transform {
    Transform::from_translation(DVec3::new(x, y, z))
}

fn set_transform(doc: &mut SceneDoc, guid: uuid::Uuid, t: Transform) {
    if let Some(e) = doc.entity_of(guid) {
        doc.world_mut().world_mut().entity_mut(e).insert(t);
        doc.world_mut().mark_dirty();
    }
}

fn set_color(doc: &mut SceneDoc, guid: uuid::Uuid, color: Color) {
    if let Some(e) = doc.entity_of(guid) {
        doc.world_mut().world_mut().entity_mut(e).insert(Material {
            base_color: color,
            ..Default::default()
        });
    }
}
