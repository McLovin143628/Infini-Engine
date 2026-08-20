//! GIS vector layers → scene entities (Wave G, Ring 1).
//!
//! [`inf_gis`] turns a Shapefile or a GeoJSON into world-metre geometry. This
//! module is the other half: it turns that geometry into things the engine's
//! existing systems already know how to simulate and draw.
//!
//! # It spawns through the doors that already exist
//!
//! Nothing here invents a component or a persistence path. A stream becomes the
//! same `WaterBody` + `Spline` pair the hydrology tool creates by hand; a road
//! or a boundary becomes the same `Spline` a designer would draw. That is
//! deliberate and it is what makes the import *cheap*: the river validator, the
//! uphill cook advisory, the buoyancy solver, the flow-to-foam wiring and the
//! grammar's span source all work on GIS-imported water and GIS-imported roads
//! on the day they land, because none of them can tell where the spline came
//! from.
//!
//! It is also what keeps this module from being a schema event. Every spawn
//! below is an ordinary scene edit; nothing new is persisted, so nothing new is
//! versioned.
//!
//! # Flow direction comes from the vertex order
//!
//! A published stream layer digitises each watercourse **downstream** — that is
//! the near-universal convention, and it is the only flow information a polyline
//! carries. So the polyline's own order becomes the river's flow order, and
//! [`inf_gis::import::ImportOptions::reverse_flow`] exists for the layers that
//! got it backwards. An import that guessed instead would produce rivers running
//! uphill, which the P20.4 cook advisory would then report by the hundred —
//! correct, and useless.
//!
//! # Every import decision is one crate down
//!
//! Naming, the stub floor, the entity cap, the stream channel: all of it is
//! [`inf_gis::import`], and this module is the **applier**. That is what makes
//! "one import door" a thing a test can falsify rather than a thing a comment
//! claims — the `inf gis` CLI builds a [`SpawnPlan`] from the same request and
//! the two are compared as values.

use glam::DVec3;
use inf_gis::feature::GeoLayer;
use inf_gis::import::{PlannedKind, SpawnPlan};
use uuid::Uuid;

use crate::scene::SceneDoc;

pub use inf_gis::import::{
    DEFAULT_MAX_ENTITIES, DEFAULT_STREAM_DEPTH_M, DEFAULT_STREAM_FLOW_M_S, DEFAULT_STREAM_WIDTH_M,
    ISLAND_MAX_ENTITIES, MIN_FEATURE_LENGTH_M,
};

/// What to do with a layer, and how much of it.
///
/// **This is [`inf_gis::import::ImportOptions`], not a copy of it.** The cap,
/// the stub floor and the flow reversal are import decisions and they live at
/// the import door in Ring 0, where the `inf gis` CLI reads the same values from
/// the same constants. Two spellings of one cap is exactly how the editor and a
/// headless pipeline come to disagree about what a layer contains.
pub type SpawnOptions = inf_gis::import::ImportOptions;

/// What a spawn produced.
///
/// Everything except [`spawned`](SpawnReport::spawned) is copied verbatim off
/// the Ring-0 [`SpawnPlan`] — this half only turns planned entities into GUIDs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpawnReport {
    /// The entities created, in spawn order.
    pub spawned: Vec<Uuid>,
    /// Features skipped for being shorter than `min_length_m`.
    pub too_short: usize,
    /// Features skipped because they carried no usable geometry.
    pub unusable: usize,
    /// Features left unspawned because `max_entities` was reached.
    ///
    /// **Reported, never silent.** An import that quietly stops at four thousand
    /// of forty thousand roads produces a city with a hard edge and no
    /// explanation.
    pub truncated: usize,
    /// The cap that produced `truncated`, so the remedy can name the number.
    pub cap: usize,
    /// Non-fatal advisories, including the layer's own.
    pub advisories: Vec<String>,
}

impl SpawnReport {
    pub fn count(&self) -> usize {
        self.spawned.len()
    }

    /// A one-line summary for the log and the wizard's done state.
    ///
    /// Delegates to [`SpawnPlan::summary`] so the wizard, the log and the CLI
    /// print the same sentence.
    pub fn summary(&self, layer: &str) -> String {
        SpawnPlan {
            entities: Vec::new(),
            too_short: self.too_short,
            unusable: self.unusable,
            truncated: self.truncated,
            cap: self.cap,
            advisories: Vec::new(),
        }
        .summary_with_count(layer, self.spawned.len())
    }
}

/// Spawn a vector layer into the document.
///
/// A thin wrapper over [`inf_gis::import::plan_spawn`] + [`apply_plan`], kept
/// because it is the shape a caller with a layer in hand wants. **It makes no
/// import decision of its own** — see [`apply_plan`].
pub fn spawn_layer(doc: &mut SceneDoc, layer: &GeoLayer, opts: &SpawnOptions) -> SpawnReport {
    let plan = inf_gis::import::plan_spawn(layer, opts);
    apply_plan(doc, &plan)
}

/// **The applier.** Turn a Ring-0 [`SpawnPlan`] into entities.
///
/// The layer's [`kind`](GeoLayer::kind) has already decided what each feature
/// becomes; this function only knows how to build the two things a plan can ask
/// for:
///
/// | planned kind | becomes |
/// |---|---|
/// | [`PlannedKind::River`] | a `WaterBody::river` + `Spline` — the same pair the hydrology tool creates, so the river validator, the uphill advisory and the buoyancy solver all apply |
/// | [`PlannedKind::Spline`] | a `Spline` entity, which the grammar's polyline span and the road builder both consume |
///
/// Areas (`Lakes`, `Biomes`, `Buildings`, `Parcels`) arrive as their
/// **boundaries**, as closed splines. `Roads` additionally gain a real surface
/// through [`crate::gisroad`], and `Biomes` a real painted region through
/// [`crate::gisbiome`]; the closed spline is the substrate all three share.
///
/// Nothing here invents a component or a persistence path, which is what keeps
/// the import from being a schema event: every spawn below is an ordinary scene
/// edit.
pub fn apply_plan(doc: &mut SceneDoc, plan: &SpawnPlan) -> SpawnReport {
    let mut report = SpawnReport {
        too_short: plan.too_short,
        unusable: plan.unusable,
        truncated: plan.truncated,
        cap: plan.cap,
        advisories: plan.advisories.clone(),
        ..Default::default()
    };
    for e in &plan.entities {
        let guid = match e.kind {
            PlannedKind::River {
                width_m,
                depth_m,
                flow_m_s,
            } => doc.edit_create_river(&e.name, &e.points, width_m, depth_m, flow_m_s),
            PlannedKind::Spline { closed } => spawn_spline(doc, &e.name, &e.points, closed),
        };
        report.spawned.push(guid);
    }
    report
}

/// **The probe, as a wire value** — what is inside a vector source, before
/// importing it.
///
/// Ring 2 calls this rather than [`inf_gis::probe`] directly, so the studio
/// crate names no GIS type at all and the DTO conversion lives beside the DTO.
/// `source_crs` empty reads the `.prj`; `level_anchor_crs` is what the open
/// level is anchored in, carried back so the wizard can show both.
pub fn probe_dto(
    path: &std::path::Path,
    source_crs: &str,
    level_anchor_crs: Option<String>,
) -> Result<crate::ipc::GisProbeDto, String> {
    let crs = source_crs.trim();
    let p = inf_gis::probe(path, (!crs.is_empty()).then_some(crs)).map_err(|e| e.to_string())?;
    Ok(crate::ipc::GisProbeDto {
        path: p.path.display().to_string(),
        format: p.format.to_string(),
        layer_name: p.layer_name.clone(),
        features: p.features,
        points: p.points,
        polylines: p.polylines,
        polygons: p.polygons,
        dominant_kind: p.dominant_kind().to_string(),
        fields: p
            .fields
            .iter()
            .map(|f| crate::ipc::GisFieldDto {
                name: f.name.clone(),
                present: f.present,
                numeric: f.numeric,
                sample: f.sample.clone(),
            })
            .collect(),
        crs: crate::ipc::GisCrsDto {
            spec: p.crs.spec.clone(),
            origin: p.crs.origin.label().to_string(),
            name: p.crs.name.clone(),
            vertical_unit_m: p.crs.vertical_unit_m,
            prj_wkt: p.crs.prj_wkt.clone(),
        },
        centre_lat: p.centre_lat_lon.map(|(lat, _)| lat),
        centre_lon: p.centre_lat_lon.map(|(_, lon)| lon),
        suggested_anchor_epsg: p.suggested_anchor_epsg,
        level_anchor_crs,
        skipped: p.skipped.clone(),
        advisories: p.advisories.iter().map(|a| a.to_string()).collect(),
    })
}

/// **The whole import, once** — the function the wizard, the CLI's editor twin
/// and any future cook-time pipeline all call (IB-3).
///
/// One request in, one report out, and every target the door offers reached
/// through the module that owns it:
///
/// | the author asked for | this calls |
/// |---|---|
/// | any layer | [`inf_gis::import_layer`] + [`apply_plan`] |
/// | a road surface | [`crate::gisroad::import_road_surface`] (IB-4) |
/// | land cover | [`crate::gisbiome::paint_biomes_from_layer`] (IB-5a) |
/// | footprints | [`crate::gisbuild::import_footprints`] (IB-5b, IB-6) |
///
/// The extras are **additive**: every layer spawns its centrelines or its
/// boundaries first, so an author who asks for a road surface gets the splines
/// the grammar's span source consumes *as well as* the mesh, and turning the
/// surface off does not change what else the import produced.
pub fn run_import(
    project: &mut crate::assets::AssetProject,
    doc: &mut SceneDoc,
    path: &std::path::Path,
    settings: &crate::ipc::GisImportSettingsDto,
) -> Result<crate::ipc::GisImportResultDto, String> {
    let req = settings.to_request(path);
    let anchor = doc.geo().clone();
    let imported = inf_gis::import_layer(&req, &anchor).map_err(|e| e.to_string())?;

    doc.begin_transaction("Import GIS Layer");
    let report = apply_plan(doc, &imported.plan);
    doc.commit_transaction();

    let mut out = crate::ipc::GisImportResultDto {
        layer_name: imported.layer.name.clone(),
        crs: imported.crs.spec.clone(),
        spawned: report.count(),
        too_short: report.too_short,
        unusable: report.unusable,
        truncated: report.truncated,
        cap: report.cap,
        summary: report.summary(&imported.layer.name),
        digest: format!("{:016x}", imported.plan.digest()),
        advisories: report.advisories.clone(),
        ..Default::default()
    };

    if settings.road_surface {
        let opts = crate::gisroad::RoadImportOptions {
            surface: inf_gis::SurfaceOptions {
                lift_m: settings.road_lift_m,
                ground_step_m: settings.road_ground_step_m,
                fill_junctions: true,
            },
            ..Default::default()
        };
        match crate::gisroad::import_road_surface(project, doc, &imported.layer, &opts) {
            Ok(r) => {
                out.road_summary = Some(r.summary());
                out.meshes.push(r.mesh.0.to_string());
                out.advisories.extend(r.advisories);
            }
            // A refused *extra* must not lose the import that succeeded: the
            // splines are already in the document, and an author who asked for
            // a surface and got a reason is better off than one whose whole
            // import vanished.
            Err(e) => out
                .advisories
                .push(format!("no road surface was built: {e}")),
        }
    }

    let terrain = settings.biome_terrain.trim();
    if !terrain.is_empty() {
        match uuid::Uuid::parse_str(terrain) {
            Ok(guid) => {
                let names = biome_names_of(project, doc, guid);
                let opts = crate::gisbiome::BiomeImportOptions {
                    attribute: settings.biome_attribute.clone(),
                    classes: settings.biome_classes,
                    ..Default::default()
                };
                match crate::gisbiome::paint_biomes_from_layer(
                    doc,
                    guid,
                    &imported.layer,
                    &opts,
                    &names,
                ) {
                    Ok(r) => {
                        out.biome_summary = Some(r.summary(&imported.layer.name));
                        out.advisories.extend(r.advisories);
                    }
                    Err(e) => out.advisories.push(format!("no biomes were painted: {e}")),
                }
            }
            Err(_) => out
                .advisories
                .push(format!("{terrain:?} is not a terrain entity id")),
        }
    }

    if settings.buildings {
        let opts = crate::gisbuild::BuildingImportOptions {
            max_buildings: settings.max_buildings,
            furnish: settings.furnish,
            ..Default::default()
        };
        match crate::gisbuild::import_footprints(project, doc, &imported.layer, &opts) {
            Ok(r) => {
                out.building_summary = Some(r.summary(&imported.layer.name));
                out.meshes
                    .extend(r.built.iter().map(|b| b.mesh.0.to_string()));
                out.advisories.extend(r.advisories);
            }
            Err(e) => out.advisories.push(format!("no buildings were built: {e}")),
        }
    }
    Ok(out)
}

/// The `(id, name)` pairs of a terrain's own biome set, for the named-class
/// route. Empty when the terrain names no set — then only the numeric route can
/// classify, which [`crate::gisbiome`] says out loud.
fn biome_names_of(
    project: &crate::assets::AssetProject,
    doc: &SceneDoc,
    terrain: Uuid,
) -> Vec<(u8, String)> {
    let Some(set_id) = doc.terrain_biome_set(terrain) else {
        return Vec::new();
    };
    match project.load_payload::<inf_terrain::BiomeSet>(inf_asset::AssetId(set_id)) {
        Ok(set) => set.biomes.iter().map(|b| (b.id, b.name.clone())).collect(),
        Err(_) => Vec::new(),
    }
}

/// **The import's ground door.** Run `f` with a ground query over the level's
/// terrains, then hand the document back.
///
/// # One rule, not a fourth spelling of it
///
/// The rule is [`inf_voxel::ground_height_at`] — the Ring-0 function IB-15 made
/// *position-aware*, which takes the **topmost terrain that answers** rather
/// than the lowest `Guid`. The two hosts each gather their terrains for it on a
/// fixed step; this is the third gather, for the import path, and it exists
/// rather than reusing one of theirs because both of theirs answer `0.0` where
/// nothing does. An import needs `None` there: a road over ground the level has
/// not authored must keep the published centreline's own elevation, not fall to
/// sea level.
///
/// The closure shape is what makes it borrow-safe: registering the archetype
/// query needs `&mut` on the world, sampling needs `&`, and spawning what the
/// samples produced needs `&mut` again. Two phases inside one call, rather than
/// a cloned `TerrainData` per terrain, which at island scale is tens of
/// megabytes of resident tiles.
///
/// Voxel volumes are **not** consulted: at import time a road drapes on the
/// heightfield, and a road over a carved cave mouth is a named remainder rather
/// than a silent answer.
pub fn with_ground<R>(
    doc: &mut SceneDoc,
    f: impl FnOnce(&mut dyn FnMut(f64, f64) -> Option<f64>) -> R,
) -> R {
    use inf_ecs::components::{GlobalTransform, Terrain, Transform};
    use inf_ecs::{Entity, Guid};

    let mut query = doc.world_mut().world_mut().query::<(
        Entity,
        &Guid,
        &Terrain,
        Option<&GlobalTransform>,
        Option<&Transform>,
    )>();
    let mut found: Vec<(Uuid, Entity, DVec3)> = Vec::new();
    {
        let w = doc.world().world();
        for (entity, guid, t, global, local) in query.iter(w) {
            if t.data.is_empty() {
                continue;
            }
            let origin = global
                .map(|g| g.translation())
                .or_else(|| local.map(|t| t.translation.to_dvec3()))
                .unwrap_or(DVec3::ZERO);
            found.push((guid.0, entity, origin));
        }
    }
    // `Guid` order, so the rule's tie-break is a function of the level rather
    // than of a bevy archetype walk — the same ordering both hosts apply.
    found.sort_unstable_by_key(|(g, _, _)| *g);
    let w = doc.world().world();
    let terrains: Vec<(&inf_ecs::TerrainData, DVec3)> = found
        .iter()
        .filter_map(|(_, e, o)| w.get::<Terrain>(*e).map(|t| (&t.data, *o)))
        .collect();
    let empty: std::collections::BTreeMap<Uuid, inf_voxel::VoxelData> =
        std::collections::BTreeMap::new();
    let mut probe = |x: f64, z: f64| inf_voxel::ground_height_at(&terrains, &empty, x, z);
    f(&mut probe)
}

/// Spawn a bare `Spline` entity at the run's first point.
///
/// The points are stored in the entity's own frame — the same convention
/// `edit_create_river` uses — so an entity dragged in the viewport carries its
/// path with it instead of leaving it behind at the world origin.
fn spawn_spline(doc: &mut SceneDoc, name: &str, pts: &[DVec3], closed: bool) -> Uuid {
    use inf_ecs::components::{Spline, SplineInterp, Transform};
    use inf_ecs::math::Vec3d;

    let origin = pts.first().copied().unwrap_or(DVec3::ZERO);
    let guid = doc.edit_create(crate::ipc::SpawnKind::Empty, name, None);
    if let Some(entity) = doc.world().entity_of(guid) {
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(origin.x, origin.y, origin.z);
        doc.world_mut().world_mut().entity_mut(entity).insert((
            Spline {
                points: pts
                    .iter()
                    .map(|p| {
                        let l = *p - origin;
                        Vec3d::new(l.x, l.y, l.z)
                    })
                    .collect(),
                closed,
                // **Linear, not Catmull-Rom.** A surveyed centreline's vertices
                // ARE the road; smoothing through them invents curvature the
                // survey does not have and pulls the path off its own right of
                // way at every bend. A hand-drawn spline wants smoothing; an
                // imported one does not.
                interp: SplineInterp::Linear,
            },
            t,
        ));
    }
    guid
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_gis::feature::{Attr, GeoFeature, GeoGeometry, GeoLayer, LayerKind};

    fn line(pts: &[(f64, f64)]) -> GeoGeometry {
        GeoGeometry::Polyline {
            points: pts.iter().map(|&(x, z)| DVec3::new(x, 0.0, z)).collect(),
            closed: false,
        }
    }

    fn layer(kind: LayerKind, features: Vec<GeoFeature>) -> GeoLayer {
        let mut l = GeoLayer::new("Test", kind, "EPSG:32610");
        l.features = features;
        l
    }

    /// **A stream layer becomes rivers the existing hydrology understands.**
    ///
    /// The point of the arm is the last assertion: the imported entity is
    /// indistinguishable from a hand-drawn one, so `river_path_of` — the door
    /// the validator, the cook advisory and the buoyancy solver all read
    /// through — resolves it with no GIS-specific code anywhere.
    #[test]
    fn a_stream_layer_becomes_rivers_the_hydrology_tools_can_read() {
        let mut doc = SceneDoc::new();
        let mut creek = GeoFeature::new(line(&[(0.0, 0.0), (0.0, 60.0), (30.0, 120.0)]));
        creek
            .attributes
            .insert("NAME".into(), Attr::Text("Still Creek".into()));
        creek.attributes.insert("WIDTH_M".into(), Attr::Number(6.5));

        let report = spawn_layer(
            &mut doc,
            &layer(LayerKind::Streams, vec![creek]),
            &SpawnOptions::default(),
        );
        assert_eq!(report.count(), 1, "{report:?}");
        let guid = report.spawned[0];

        // It is a real river as far as the hydrology layer is concerned.
        let path = crate::hydro::river_path_of(&doc, guid)
            .expect("the imported stream resolves through the hydrology door");
        assert!(
            (path.length_m - 127.08).abs() < 1.0,
            "the river is {} m long; the polyline is ~127",
            path.length_m
        );

        // The attribute reached the component rather than the default.
        let world = doc.world();
        let e = world.entity_of(guid).unwrap();
        let body = world
            .world()
            .get::<inf_ecs::components::WaterBody>(e)
            .expect("a WaterBody");
        assert_eq!(body.kind, inf_ecs::components::WaterKind::River);
        assert!((body.river_width_start_m - 6.5).abs() < 1e-9);
        // …and the ones it did NOT carry took the documented defaults.
        assert!((body.river_depth_start_m - DEFAULT_STREAM_DEPTH_M).abs() < 1e-9);
        assert_eq!(world.name_of(e), Some("Still Creek"));
    }

    /// **Flow direction is the vertex order**, and `reverse_flow` is the escape
    /// hatch for a layer that digitised it the other way.
    ///
    /// Un-fix mutation: ignore `reverse_flow` and the two paths below start at
    /// the same place.
    #[test]
    fn flow_direction_follows_the_polyline_and_can_be_reversed() {
        let pts = &[(0.0, 0.0), (0.0, 100.0)];
        let mk = |rev: bool| {
            let mut doc = SceneDoc::new();
            let r = spawn_layer(
                &mut doc,
                &layer(LayerKind::Streams, vec![GeoFeature::new(line(pts))]),
                &SpawnOptions {
                    reverse_flow: rev,
                    ..Default::default()
                },
            );
            let path = crate::hydro::river_path_of(&doc, r.spawned[0]).expect("a river");
            path.frames[0].center
        };
        let downstream = mk(false);
        let upstream = mk(true);
        assert!(
            (downstream.z - 0.0).abs() < 1e-6,
            "the river should start at the polyline's FIRST point, got {downstream:?}"
        );
        assert!(
            (upstream.z - 100.0).abs() < 1e-6,
            "reversed, it should start at the last, got {upstream:?}"
        );
    }

    /// Roads and boundaries become splines the grammar and the road builder can
    /// consume — with **linear** interpolation, because a survey's vertices are
    /// the road.
    #[test]
    fn roads_become_linear_splines_and_rings_close() {
        let mut doc = SceneDoc::new();
        let road = GeoFeature::new(line(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]));
        let parcel = GeoFeature::new(GeoGeometry::Polygon {
            exterior: vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(40.0, 0.0, 0.0),
                DVec3::new(40.0, 0.0, 40.0),
                DVec3::new(0.0, 0.0, 40.0),
            ],
            holes: vec![],
        });
        let mut l = layer(LayerKind::Roads, vec![road]);
        l.features.push(parcel);
        let report = spawn_layer(&mut doc, &l, &SpawnOptions::default());
        assert_eq!(report.count(), 2, "{report:?}");

        let world = doc.world();
        let splines: Vec<inf_ecs::components::Spline> = report
            .spawned
            .iter()
            .map(|g| {
                world
                    .world()
                    .get::<inf_ecs::components::Spline>(world.entity_of(*g).unwrap())
                    .cloned()
                    .expect("a Spline")
            })
            .collect();
        for s in &splines {
            assert_eq!(
                s.interp,
                inf_ecs::components::SplineInterp::Linear,
                "an imported centreline must not be smoothed — its vertices ARE the road"
            );
        }
        assert!(!splines[0].closed, "an open road stays open");
        assert!(splines[1].closed, "a polygon boundary comes in as a ring");
        // Points are entity-local, so the first is the origin.
        assert_eq!(splines[0].points[0], inf_ecs::math::Vec3d::ZERO);
        assert_eq!(splines[0].points.len(), 3);
    }

    /// **Stubs are skipped, the cap truncates, and both are REPORTED.**
    ///
    /// A vector layer is full of two-metre fragments and a county road file is
    /// 10⁵ features; an import that silently drops either produces a city with a
    /// hard edge and no explanation.
    #[test]
    fn stubs_and_the_entity_cap_are_reported_rather_than_silent() {
        let mut doc = SceneDoc::new();
        let mut features = vec![
            GeoFeature::new(line(&[(0.0, 0.0), (2.0, 0.0)])), // a 2 m stub
            GeoFeature::new(GeoGeometry::Point(DVec3::ZERO)), // no run at all
        ];
        for i in 0..5 {
            features.push(GeoFeature::new(line(&[
                (0.0, i as f64 * 10.0),
                (50.0, i as f64 * 10.0),
            ])));
        }
        let report = spawn_layer(
            &mut doc,
            &layer(LayerKind::Roads, features),
            &SpawnOptions {
                max_entities: 3,
                ..Default::default()
            },
        );
        assert_eq!(report.count(), 3, "the cap holds");
        assert_eq!(report.too_short, 1);
        assert_eq!(report.unusable, 1);
        assert_eq!(report.truncated, 2, "what the cap left behind is COUNTED");
        let s = report.summary("Roads");
        assert!(s.contains("NOT IMPORTED"), "{s}");
        assert!(s.contains("too short"), "{s}");
    }

    /// **The applier applies; it does not decide.**
    ///
    /// Every count in the report is the plan's own count, and the entity list is
    /// the plan's entity list in order — so a front end that builds a plan
    /// headlessly (the `inf gis` CLI) and one that applies it (the wizard) are
    /// looking at the same import rather than two imports that agree by
    /// inspection. Un-fix mutations: dropping the cap into `apply_plan`, or
    /// re-deriving `too_short` here, both fail this.
    #[test]
    fn the_applied_report_is_the_plan_it_was_given() {
        let mut features = vec![
            GeoFeature::new(line(&[(0.0, 0.0), (2.0, 0.0)])),
            GeoFeature::new(GeoGeometry::Point(DVec3::ZERO)),
        ];
        for i in 0..5 {
            features.push(GeoFeature::new(line(&[
                (0.0, i as f64 * 10.0),
                (50.0, i as f64 * 10.0),
            ])));
        }
        let l = layer(LayerKind::Roads, features);
        let opts = SpawnOptions {
            max_entities: 3,
            ..Default::default()
        };
        let plan = inf_gis::import::plan_spawn(&l, &opts);

        let mut doc = SceneDoc::new();
        let report = apply_plan(&mut doc, &plan);
        assert_eq!(report.count(), plan.count());
        assert_eq!(
            (
                report.too_short,
                report.unusable,
                report.truncated,
                report.cap
            ),
            (plan.too_short, plan.unusable, plan.truncated, plan.cap),
            "the report restates the plan, it does not recompute it"
        );
        assert_eq!(report.advisories, plan.advisories);
        // The names came from the plan, in the plan's order.
        let world = doc.world();
        let names: Vec<String> = report
            .spawned
            .iter()
            .map(|g| {
                world
                    .name_of(world.entity_of(*g).unwrap())
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        let planned: Vec<String> = plan.entities.iter().map(|e| e.name.clone()).collect();
        assert_eq!(names, planned);
        // And `spawn_layer` is that pair, not a third path.
        let mut doc2 = SceneDoc::new();
        let via_layer = spawn_layer(&mut doc2, &l, &opts);
        assert_eq!(via_layer.count(), report.count());
        assert_eq!(via_layer.truncated, report.truncated);
    }

    /// **One door, every target.** `run_import` spawns the centrelines AND
    /// builds the road surface, and the extras are additive — turning the
    /// surface off changes nothing else the import produced.
    ///
    /// The digest it reports is the same number `inf gis plan` prints, which is
    /// what makes the CLI's cross-process comparison a comparison of *this*.
    #[test]
    fn run_import_spawns_the_layer_and_builds_the_extras_through_one_request() {
        use crate::ipc::GisImportSettingsDto;

        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("roads.geojson");
        // Two roads meeting at a shared endpoint, in the anchor's own CRS.
        std::fs::write(
            &src,
            r#"{"type":"FeatureCollection","features":[
              {"type":"Feature","properties":{"name":"Main","road_type":"arterial"},
               "geometry":{"type":"LineString","coordinates":[[491000,5459000],[491120,5459000]]}},
              {"type":"Feature","properties":{"name":"Elm"},
               "geometry":{"type":"LineString","coordinates":[[491120,5459000],[491120,5459100]]}}
            ]}"#,
        )
        .unwrap();
        let mut proj = crate::assets::AssetProject::open(tmp.path().join("Content")).unwrap();

        // A level anchored at Vancouver, with a terrain under the roads.
        let mut doc = SceneDoc::new();
        doc.edit_geo(
            inf_gis::anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "EGM2008").unwrap(),
        );
        {
            let guid = doc.edit_create(crate::ipc::SpawnKind::Empty, "Ground", None);
            let e = doc.world().entity_of(guid).unwrap();
            let res = 129u32;
            let tile = inf_terrain::TerrainTile::from_heights(
                res,
                DVec3::ZERO,
                vec![0.0; (res * res) as usize],
            )
            .unwrap();
            let mut data = inf_terrain::TerrainData::new(res, 2.0);
            data.insert_tile((0, 0), tile).ok();
            let mut t = inf_ecs::components::Transform::IDENTITY;
            t.translation = inf_ecs::math::Vec3d::new(-40.0, 0.0, -40.0);
            doc.world_mut().world_mut().entity_mut(e).insert((
                inf_ecs::components::Terrain {
                    data,
                    ..Default::default()
                },
                t,
            ));
            doc.world_mut().propagate();
        }

        let mut settings = GisImportSettingsDto {
            kind: "roads".into(),
            source_crs: "EPSG:32610".into(),
            vertical_unit_m: 0.0,
            max_entities: 4096,
            min_length_m: 8.0,
            reverse_flow: false,
            name_prefix: String::new(),
            road_surface: true,
            road_lift_m: 0.02,
            road_ground_step_m: 1.0,
            biome_terrain: String::new(),
            biome_attribute: String::new(),
            biome_classes: 8,
            buildings: false,
            max_buildings: 16,
            furnish: false,
        };

        let with_surface = run_import(&mut proj, &mut doc, &src, &settings).unwrap();
        assert_eq!(with_surface.spawned, 2, "{with_surface:?}");
        assert_eq!(with_surface.truncated, 0);
        assert!(
            with_surface.road_summary.is_some(),
            "the surface was asked for: {with_surface:?}"
        );
        assert_eq!(with_surface.meshes.len(), 1);
        assert!(
            with_surface.summary.contains("2 entities"),
            "{with_surface:?}"
        );

        // The digest is the plan's, so the CLI's cross-process comparison is a
        // comparison of THIS.
        let anchor = doc.geo().clone();
        let mut req = inf_gis::ImportRequest::new(&src, inf_gis::LayerKind::Roads);
        req.source_crs = Some("EPSG:32610".into());
        let lib = inf_gis::import_layer(&req, &anchor).unwrap();
        assert_eq!(with_surface.digest, format!("{:016x}", lib.plan.digest()));

        // Turning the extra off changes the extra and nothing else. Re-run on
        // the same document: an import's report is about that import, so the
        // comparison is valid and it also proves a second import of one file
        // reports the same plan.
        settings.road_surface = false;
        let plain = run_import(&mut proj, &mut doc, &src, &settings).unwrap();
        assert_eq!(plain.spawned, with_surface.spawned);
        assert_eq!(plain.digest, with_surface.digest);
        assert!(plain.road_summary.is_none());
        assert!(plain.meshes.is_empty());
    }

    /// A level with no geo-anchor refuses the whole import at the door, rather
    /// than spawning a city at the world origin.
    #[test]
    fn run_import_refuses_a_level_with_no_anchor() {
        use crate::ipc::GisImportSettingsDto;
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("roads.geojson");
        std::fs::write(
            &src,
            r#"{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},
               "geometry":{"type":"LineString","coordinates":[[491000,5459000],[491120,5459000]]}}]}"#,
        )
        .unwrap();
        let mut proj = crate::assets::AssetProject::open(tmp.path().join("Content")).unwrap();
        let mut doc = SceneDoc::new();
        let settings = GisImportSettingsDto::suggested(
            &crate::ipc::GisProbeDto {
                path: String::new(),
                format: "geojson".into(),
                layer_name: "roads".into(),
                features: 1,
                points: 0,
                polylines: 1,
                polygons: 0,
                dominant_kind: "polyline".into(),
                fields: Vec::new(),
                crs: crate::ipc::GisCrsDto {
                    spec: "EPSG:32610".into(),
                    origin: "stated".into(),
                    name: None,
                    vertical_unit_m: 1.0,
                    prj_wkt: None,
                },
                centre_lat: None,
                centre_lon: None,
                suggested_anchor_epsg: None,
                level_anchor_crs: None,
                skipped: Vec::new(),
                advisories: Vec::new(),
            },
            4096,
        );
        let mut settings = settings;
        settings.source_crs = "EPSG:32610".into();
        let e = run_import(&mut proj, &mut doc, &src, &settings).unwrap_err();
        assert!(e.to_ascii_lowercase().contains("anchor"), "{e}");
        assert_eq!(doc.order().len(), 0, "nothing was created");
    }

    /// A layer's own advisories ride through to the import's report — an axis
    /// warning or a datum-error note must not stop at the reader.
    #[test]
    fn layer_advisories_reach_the_spawn_report() {
        let mut doc = SceneDoc::new();
        let mut l = layer(LayerKind::Roads, vec![]);
        l.advisories
            .push(inf_gis::Advisory::new("datum.no_grid", "a 20 m offset"));
        l.skipped.push("record 4: non-finite coordinate".into());
        let report = spawn_layer(&mut doc, &l, &SpawnOptions::default());
        assert_eq!(report.count(), 0);
        assert!(
            report.advisories.iter().any(|a| a.contains("20 m offset")),
            "{:?}",
            report.advisories
        );
        assert!(
            report.advisories.iter().any(|a| a.contains("record 4")),
            "{:?}",
            report.advisories
        );
    }
}
