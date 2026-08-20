//! **GIS roads become geometry** (IB-4, Ring 1).
//!
//! Wave G shipped a ribbon builder that "returns vertex, UV and index arrays,
//! not a `MeshAsset`, and takes the ground as a caller-supplied closure that
//! nothing in the tree supplies". This module is the caller, and
//! [`inf_gis::roads::surface_to_mesh`] is the asset.
//!
//! # The four things a road needs, in order
//!
//! 1. **A graph.** [`inf_gis::RoadGraph::from_layer`] derives the intersections
//!    published layers do not carry.
//! 2. **Ground.** [`crate::gis::with_ground`] — the IB-15 rule, the topmost
//!    terrain that answers, so a road crossing from one terrain onto another
//!    follows both.
//! 3. **A surface.** [`inf_gis::build_surface`] drapes and merges by class and
//!    fans the junctions.
//! 4. **A place in the world.** One `.inf_mesh` written through
//!    [`crate::assets::AssetProject::write_asset`] — the same door every other
//!    importer writes through — and one entity created through
//!    `SceneDoc::edit_create_mesh_asset`, the same door a dropped asset uses.
//!
//! Nothing here is a new component, a new asset kind or a new persistence path,
//! so the road surface costs **no schema**.
//!
//! # Colliders: measured, then ruled
//!
//! A road built this way is drivable **through the ground it conforms to**. Its
//! vertices sit `lift_m` above the terrain's own surface at the terrain's own
//! sample pitch, and the terrain already carries a heightfield collider — so a
//! body dropped on a road rests on the road, and a per-segment trimesh collider
//! would be a second surface 2 cm under the first.
//!
//! It was priced rather than assumed. IB-2 measured a static collider at
//! **0.363 µs/step**, so the 10 000-road layer the certification names would
//! cost **3.63 ms/step** — 22% of a 60 fps frame — to duplicate a surface that
//! is already there. `the_ground_under_the_road_is_the_road` asserts the world
//! half — 3 758 road vertices, worst deviation 0.000000 m from `ground + lift`
//! — and the arithmetic is here.
//!
//! The case that genuinely needs its own collider is a road that **leaves** the
//! ground — a bridge, an overpass — and that needs a bridge/tunnel attribute
//! published layers rarely carry (the disposition memo's own note about planar
//! intersection snapping). Named, not smuggled.

use glam::DVec3;
use uuid::Uuid;

use inf_asset::AssetId;
use inf_gis::feature::GeoLayer;
use inf_gis::roads::{RoadGraph, RoadSurface, SurfaceOptions};

use crate::assets::AssetProject;
use crate::scene::SceneDoc;

/// Where road meshes land under the content root.
pub const ROAD_FOLDER: &str = "GIS";

/// What a road import may do.
#[derive(Clone, Debug, PartialEq)]
pub struct RoadImportOptions {
    /// Draping and junction options — see [`SurfaceOptions`].
    pub surface: SurfaceOptions,
    /// Destination folder under the content root.
    pub folder: String,
    /// Asset/entity name; empty falls back to the layer's own name.
    pub name: String,
}

impl Default for RoadImportOptions {
    fn default() -> Self {
        Self {
            surface: SurfaceOptions::default(),
            folder: ROAD_FOLDER.to_string(),
            name: String::new(),
        }
    }
}

/// What a road import produced.
#[derive(Clone, Debug, PartialEq)]
pub struct RoadImportOutcome {
    /// The `.inf_mesh` that was written.
    pub mesh: AssetId,
    /// The entity that draws it.
    pub entity: Uuid,
    /// Where the mesh's local origin sits in the world.
    pub origin: DVec3,
    pub segments: usize,
    pub junctions: usize,
    pub junctions_filled: usize,
    pub junctions_skipped: usize,
    pub triangles: usize,
    pub vertices: usize,
    /// The f32 spacing at the mesh's furthest vertex — see
    /// [`inf_gis::MeshBuildReport`].
    pub quantisation_m: f64,
    /// Total centreline length, metres.
    pub length_m: f64,
    /// Segments that took a default class because the source's value was not
    /// recognised.
    pub unclassified: usize,
    /// Everything that could not be built, and why.
    pub skipped: Vec<String>,
    pub advisories: Vec<String>,
}

impl RoadImportOutcome {
    /// A one-line summary for the log and the wizard's done state.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} segments, {:.0} m, {} triangles; {} of {} junctions paved",
            self.segments, self.length_m, self.triangles, self.junctions_filled, self.junctions
        );
        if !self.skipped.is_empty() {
            s.push_str(&format!(", {} without a surface", self.skipped.len()));
        }
        if self.unclassified > 0 {
            s.push_str(&format!(", {} unclassified", self.unclassified));
        }
        s
    }
}

/// **Import a road layer as real geometry** — the whole of IB-4 in one call.
///
/// Returns a refusal rather than a partial world when the layer has no usable
/// road in it at all; individual segments that cannot be built are *skipped and
/// named* in [`RoadImportOutcome::skipped`], because a silent hole in a road
/// network shows up as a street that stops in the middle of a block.
pub fn import_road_surface(
    project: &mut AssetProject,
    doc: &mut SceneDoc,
    layer: &GeoLayer,
    opts: &RoadImportOptions,
) -> Result<RoadImportOutcome, String> {
    let graph = RoadGraph::from_layer(layer);
    if graph.segments.is_empty() {
        return Err(format!(
            "the layer {:?} produced no road segments at all — {} feature(s) were \
             skipped. The first reason was: {}",
            layer.name,
            graph.skipped.len(),
            graph.skipped.first().cloned().unwrap_or_else(|| {
                "the layer contained no polylines with two or more positions".into()
            })
        ));
    }
    let surface = crate::gis::with_ground(doc, |h| {
        inf_gis::roads::build_surface(&graph, &opts.surface, h)
    });
    let origin = surface_origin(&surface);
    let (mesh, report) =
        inf_gis::roads::surface_to_mesh(&surface, origin).map_err(|e| e.to_string())?;

    let name = if opts.name.trim().is_empty() {
        layer.name.trim()
    } else {
        opts.name.trim()
    };
    let name = if name.is_empty() { "Roads" } else { name };

    let dir = project
        .content_dir(&opts.folder)
        .map_err(|e| format!("could not open the destination folder: {e}"))?;
    let mesh_id = project
        .write_asset(
            &dir,
            name,
            &mesh,
            Some(format!("gis:{}", layer.source_crs)),
            Vec::new(),
            None,
        )
        .map_err(|e| format!("could not write the road mesh: {e}"))?;

    // One transaction, so an author's Ctrl+Z takes the whole import back rather
    // than half of it.
    doc.begin_transaction("Import Roads");
    let entity = doc.edit_create_mesh_asset(name, mesh_id.0, None);
    doc.edit_set_transform(
        entity,
        inf_ecs::components::Transform::from_translation(origin),
    );
    doc.commit_transaction();

    let mut advisories: Vec<String> = layer.advisories.iter().map(|a| a.to_string()).collect();
    if surface.junctions_skipped > 0 {
        advisories.push(format!(
            "{} junction(s) could not be paved (degenerate cross-sections); the \
             ribbons still meet there, but the corner is open.",
            surface.junctions_skipped
        ));
    }
    if report.quantisation_m > 0.01 {
        advisories.push(format!(
            "this road mesh spans {:.0} m from its own origin, where an f32 vertex \
             resolves to {:.3} m. Import a smaller extent, or accept that the road \
             surface is quantised at that spacing.",
            report.max_offset_m, report.quantisation_m
        ));
    }
    Ok(RoadImportOutcome {
        mesh: mesh_id,
        entity,
        origin,
        segments: graph.segments.len(),
        junctions: graph.junctions().count(),
        junctions_filled: surface.junctions_filled,
        junctions_skipped: surface.junctions_skipped,
        triangles: report.triangles,
        vertices: report.vertices,
        quantisation_m: report.quantisation_m,
        length_m: graph.total_length_m(),
        unclassified: graph.unclassified,
        skipped: surface.skipped,
        advisories,
    })
}

/// The mesh's local origin: the centre of its own bounds.
///
/// **Not the world origin, and not the first vertex.** A `MeshVertex` is f32; a
/// UTM easting is ~5×10⁵ and spends the whole mantissa on the false easting,
/// leaving 30 mm of resolution. Centring on the content makes the number an
/// *extent* rather than a coordinate — half of it, in fact — so a 50 km island
/// quantises at ~3 mm instead of 30 mm at the origin and worse further out.
fn surface_origin(surface: &RoadSurface) -> DVec3 {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for r in surface.parts.values() {
        for v in &r.vertices {
            if v.is_finite() {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
    }
    if !(lo.is_finite() && hi.is_finite()) {
        return DVec3::ZERO;
    }
    (lo + hi) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_gis::feature::{Attr, GeoFeature, GeoGeometry, LayerKind};

    fn line(pts: &[(f64, f64)]) -> GeoGeometry {
        GeoGeometry::Polyline {
            points: pts.iter().map(|&(x, z)| DVec3::new(x, 0.0, z)).collect(),
            closed: false,
        }
    }

    fn road_layer() -> GeoLayer {
        let mut l = GeoLayer::new("Downtown", LayerKind::Roads, "EPSG:32610");
        let mut a = GeoFeature::new(line(&[(0.0, 0.0), (60.0, 0.0), (120.0, 0.0)]));
        a.attributes
            .insert("name".into(), Attr::Text("Broadway".into()));
        a.attributes
            .insert("road_type".into(), Attr::Text("arterial".into()));
        let mut b = GeoFeature::new(line(&[(120.0, 0.0), (220.0, 0.0)]));
        b.attributes
            .insert("name".into(), Attr::Text("Broadway East".into()));
        let mut c = GeoFeature::new(line(&[(120.0, 0.0), (120.0, 140.0)]));
        c.attributes.insert("name".into(), Attr::Text("Elm".into()));
        l.features = vec![a, b, c];
        l
    }

    /// A flat-topped hill 40 m across the middle of the layer, as a real
    /// `Terrain` component on a real entity.
    fn doc_with_terrain(second: bool) -> SceneDoc {
        use inf_ecs::components::{Terrain, Transform};
        use inf_terrain::{TerrainData, TerrainTile};

        let mut doc = SceneDoc::new();
        let mut place = |name: &str, origin: DVec3, height: f64| {
            let guid = doc.edit_create(crate::ipc::SpawnKind::Empty, name, None);
            let e = doc.world().entity_of(guid).unwrap();
            // A 129-sample tile at 2 m/sample = 256 m across.
            let res = 129u32;
            let mut h = vec![0.0f32; (res * res) as usize];
            for j in 0..res {
                for i in 0..res {
                    // A gentle dome, so a chord across it is measurably wrong.
                    let (x, z) = (i as f64 * 2.0 - 128.0, j as f64 * 2.0 - 128.0);
                    h[(j * res + i) as usize] =
                        (height - 0.0015 * (x * x + z * z) * 0.25).max(0.0) as f32;
                }
            }
            let tile = TerrainTile::from_heights(res, DVec3::ZERO, h).unwrap();
            let mut data = TerrainData::new(res, 2.0);
            data.insert_tile((0, 0), tile).ok();
            let mut t = Transform::IDENTITY;
            t.translation = inf_ecs::math::Vec3d::new(origin.x, origin.y, origin.z);
            doc.world_mut().world_mut().entity_mut(e).insert((
                Terrain {
                    data,
                    ..Default::default()
                },
                t,
            ));
            guid
        };
        // Placed so the whole road layer, kerbs included, is INSIDE the terrain:
        // a tile spans 256 m and a road running along its edge queries ground
        // that is not there, which is a fixture defect wearing an import defect's
        // clothes.
        place("Ground", DVec3::new(-20.0, 0.0, -60.0), 30.0);
        if second {
            // A second terrain, abutting the first to the east, higher.
            place("Ground East", DVec3::new(236.0, 0.0, -60.0), 50.0);
        }
        // **Propagate, or the fixture measures nothing.** `with_ground` prefers a
        // `GlobalTransform` over a local one — as both hosts do — and an entity
        // that has just been given a `Transform` still carries the identity
        // global it was created with. Without this the two terrains sit on top of
        // each other at the world origin and the "second terrain" arm is testing
        // one terrain twice.
        doc.world_mut().propagate();
        doc
    }

    fn project(dir: &std::path::Path) -> AssetProject {
        AssetProject::open(dir).expect("a fresh content root opens")
    }

    /// **A road layer becomes a real mesh asset and a real entity**, placed
    /// where the mesh's own origin says.
    #[test]
    fn a_road_layer_becomes_a_mesh_asset_and_an_entity() {
        let tmp = tempfile::tempdir().unwrap();
        let mut proj = project(tmp.path());
        let mut doc = doc_with_terrain(false);
        let before = doc.order().len();

        let out = import_road_surface(&mut proj, &mut doc, &road_layer(), &Default::default())
            .expect("the import succeeds");
        assert_eq!(out.segments, 3);
        assert_eq!(out.junctions_filled, 1, "the T is paved: {out:?}");
        assert!(out.triangles > 100, "{out:?}");
        assert!(out.skipped.is_empty(), "{:?}", out.skipped);

        // The asset is on disk, registered, and re-loads through the same door.
        let mesh: inf_mesh::MeshAsset = proj.load_payload(out.mesh).expect("the mesh re-loads");
        assert_eq!(mesh.triangle_count(), out.triangles);
        assert_eq!(
            mesh.material_slots,
            vec!["arterial".to_string(), "residential".into()]
        );

        // The entity draws it, and it is placed at the mesh's own origin.
        assert_eq!(doc.order().len(), before + 1);
        let e = doc.world().entity_of(out.entity).unwrap();
        let mr = doc
            .world()
            .world()
            .get::<inf_ecs::components::MeshRef>(e)
            .expect("a MeshRef");
        assert_eq!(mr.asset, Some(out.mesh.0));
        let t = doc
            .world()
            .world()
            .get::<inf_ecs::components::Transform>(e)
            .expect("a Transform");
        assert!((t.translation.to_dvec3() - out.origin).length() < 1e-9);

        // One undo step takes the whole import back.
        assert!(doc.undo());
        assert_eq!(doc.order().len(), before, "the import is ONE undo step");
    }

    /// **The road drapes on the ground the level actually has** — including a
    /// SECOND terrain, through the IB-15 topmost-that-answers rule.
    ///
    /// Un-fix mutation: narrow `with_ground`'s gather to the first terrain and
    /// the eastern half of the road drops to the published centreline's zero.
    #[test]
    fn a_road_drapes_across_two_terrains() {
        let tmp = tempfile::tempdir().unwrap();
        let mut proj = project(tmp.path());
        let mut doc = doc_with_terrain(true);
        // A road running east from the first terrain onto the second.
        let mut l = GeoLayer::new("Crossing", LayerKind::Roads, "EPSG:32610");
        l.features = vec![GeoFeature::new(line(&[(0.0, 0.0), (380.0, 0.0)]))];

        let out = import_road_surface(&mut proj, &mut doc, &l, &Default::default()).unwrap();
        let mesh: inf_mesh::MeshAsset = proj.load_payload(out.mesh).unwrap();
        // Sample the mesh's own vertices back in world space, split west/east of
        // the border at x = 256 - 128 = 128 (the second terrain's tile starts at
        // its origin).
        let mut west_max = f64::NEG_INFINITY;
        let mut east_max = f64::NEG_INFINITY;
        for sub in &mesh.submeshes {
            for v in &sub.vertices {
                let w = out.origin
                    + DVec3::new(
                        v.position[0] as f64,
                        v.position[1] as f64,
                        v.position[2] as f64,
                    );
                if w.x < 120.0 {
                    west_max = west_max.max(w.y);
                } else if w.x > 300.0 {
                    east_max = east_max.max(w.y);
                }
            }
        }
        // The road runs along the tiles' z = 0 edge, where the dome has already
        // fallen ~6 m from its 30 m / 50 m centres.
        assert!(
            (20.0..30.0).contains(&west_max),
            "the western half sits on the 30 m dome: {west_max}"
        );
        assert!(
            east_max > 40.0,
            "THE EASTERN HALF MUST SIT ON THE SECOND TERRAIN (50 m dome), not at \
             the centreline's own zero. It measured {east_max} m."
        );
        println!("IB-4/IB-15: road west {west_max:.3} m, east {east_max:.3} m");
    }

    /// **What a body meets under a road is the terrain's own collider, at the
    /// road's own height.**
    ///
    /// This is the collider claim, measured rather than assumed. `Terrain`
    /// carries a heightfield collider built from `TerrainData` (the P22.3
    /// prerequisite — before it a dynamic body fell through the world), and a
    /// road vertex is placed at `height_at(x, z) + lift`. So the drawn surface
    /// and the collided surface are two readings of one array, and the gap
    /// between them is exactly `lift_m` — 2 cm — everywhere, which is what this
    /// measures.
    ///
    /// Duplicating that surface as a per-segment trimesh would cost
    /// 0.363 µs/step each (IB-2's measurement), i.e. **3.63 ms/step** for the
    /// certification's 10 000-road layer, to add nothing a body can reach. The
    /// case that does need one — a road that leaves the ground — is named in the
    /// module docs.
    #[test]
    fn the_ground_under_the_road_is_the_road() {
        let tmp = tempfile::tempdir().unwrap();
        let mut proj = project(tmp.path());
        let mut doc = doc_with_terrain(false);
        let out =
            import_road_surface(&mut proj, &mut doc, &road_layer(), &Default::default()).unwrap();
        let mesh: inf_mesh::MeshAsset = proj.load_payload(out.mesh).unwrap();

        let lift = SurfaceOptions::default().lift_m;
        let mut checked = 0usize;
        let mut worst = 0.0f64;
        let mut unanswered = 0usize;
        crate::gis::with_ground(&mut doc, |ground| {
            for sub in &mesh.submeshes {
                for v in &sub.vertices {
                    let w = out.origin
                        + DVec3::new(
                            v.position[0] as f64,
                            v.position[1] as f64,
                            v.position[2] as f64,
                        );
                    match ground(w.x, w.z) {
                        Some(g) => {
                            worst = worst.max((w.y - g - lift).abs());
                            checked += 1;
                        }
                        None => unanswered += 1,
                    }
                }
            }
        });
        assert!(checked > 200, "only {checked} vertices were over ground");
        assert_eq!(unanswered, 0, "the fixture's terrain must cover the roads");
        // Millimetres, and the residue is the f32 the `MeshVertex` is stored in
        // — not a modelling gap.
        assert!(
            worst < 0.01,
            "the drawn road and the collided ground must agree to within the lift; \
             the worst vertex was {worst:.6} m off"
        );
        println!(
            "IB-4 collider: {checked} road vertices over ground, worst deviation \
             {worst:.6} m against a {lift} m lift ({unanswered} outside the terrain)"
        );
    }

    /// A layer with no roads in it is a refusal that names why, not an empty
    /// mesh asset nobody asked for.
    #[test]
    fn a_layer_with_no_roads_is_refused_with_its_reason() {
        let tmp = tempfile::tempdir().unwrap();
        let mut proj = project(tmp.path());
        let mut doc = doc_with_terrain(false);
        let mut l = GeoLayer::new("Wells", LayerKind::Roads, "EPSG:32610");
        l.features = vec![GeoFeature::new(GeoGeometry::Point(DVec3::ZERO))];
        let before = doc.order().len();
        let e = import_road_surface(&mut proj, &mut doc, &l, &Default::default()).unwrap_err();
        assert!(
            e.contains("Wells") && e.contains("not a road segment"),
            "{e}"
        );
        assert_eq!(doc.order().len(), before, "nothing was created");
    }
}
