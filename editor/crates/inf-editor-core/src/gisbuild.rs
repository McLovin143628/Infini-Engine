//! **A GIS footprint becomes a building** (IB-5b + IB-6, Ring 1).
//!
//! The second half of the wire [`inf_gis::buildings`] opens. Wave G's G11 row
//! says there is "no code between a GIS attribute and [`BuildingParams::floors`]
//! in either direction"; this is that code, and it runs through the pieces that
//! already exist rather than beside them:
//!
//! | step | door |
//! |---|---|
//! | the footprint's own rectangle | [`inf_math::min_area_rect`] (IB-6) |
//! | its storey count and type | [`inf_gis::footprint_attrs`] (IB-5b) |
//! | the ground it stands on | [`crate::gis::with_ground`] — the IB-15 rule |
//! | the building itself | `inf_pcg::building::build_in` |
//! | the asset | [`crate::bake::bake_building_in`] |
//! | the entity | `SceneDoc::edit_create_mesh_asset` |
//!
//! # The attribute pass runs over the whole layer; the bake does not
//!
//! Reading fifty thousand attribute rows costs nothing and the *coverage* is
//! what an author needs to know ("4 831 of 5 002 took the default"), so it is
//! measured over every feature. **Baking** fifty thousand meshes is fifty
//! thousand assets on disk, so it is capped, and the cap is reported with the
//! number to raise — the IB-14 discipline, applied to the expensive half rather
//! than to the cheap one.

use glam::DVec2;
use uuid::Uuid;

use inf_asset::AssetId;
use inf_gis::buildings::{footprint_attrs, AttrCoverage, FootprintDefaults};
use inf_gis::feature::{GeoGeometry, GeoLayer};
use inf_pcg::building::{BuildingParams, LotFrame, Rect2};
use inf_pcg::ArchetypeId;

use crate::assets::AssetProject;
use crate::scene::SceneDoc;

/// Where baked buildings land under the content root.
pub const BUILDING_FOLDER: &str = "GIS";

/// How many footprints are baked into geometry by default.
///
/// **A guard on the expensive half.** The attribute wire runs over every
/// feature; each baked building is a `.inf_mesh` on disk and an entity in the
/// document, and a metropolitan footprint layer is 10⁵ of them. Reported, never
/// silent, with the number to raise in the message.
pub const DEFAULT_MAX_BUILDINGS: usize = 512;

/// A footprint smaller than this is a shed, a canopy or a digitising artefact.
pub const MIN_FOOTPRINT_AREA_M2: f64 = 24.0;

/// What a footprint import may do.
#[derive(Clone, Debug, PartialEq)]
pub struct BuildingImportOptions {
    pub defaults: FootprintDefaults,
    /// Force one archetype; `None` reads the type attribute.
    pub archetype: Option<ArchetypeId>,
    pub furnish: bool,
    pub folder: String,
    /// See [`DEFAULT_MAX_BUILDINGS`].
    pub max_buildings: usize,
    pub min_area_m2: f64,
    /// Base seed; each footprint mixes its own index in.
    pub seed: u64,
}

impl Default for BuildingImportOptions {
    fn default() -> Self {
        Self {
            defaults: FootprintDefaults::default(),
            archetype: None,
            furnish: false,
            folder: BUILDING_FOLDER.to_string(),
            max_buildings: DEFAULT_MAX_BUILDINGS,
            min_area_m2: MIN_FOOTPRINT_AREA_M2,
            seed: 0x1B05,
        }
    }
}

/// One imported building.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedBuilding {
    pub entity: Uuid,
    pub mesh: AssetId,
    pub feature: usize,
    pub floors: u32,
    pub archetype: ArchetypeId,
    /// The lot's own frame — its origin and the direction of its long side.
    pub frame: LotFrame,
    /// The lot, in that frame.
    pub lot: Rect2,
    /// The building's own height above `base_y`, metres.
    pub height_m: f64,
}

/// What a footprint import produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildingImportOutcome {
    pub built: Vec<ImportedBuilding>,
    /// Attribute coverage over the **whole** layer, not just the baked part.
    pub coverage: AttrCoverage,
    /// Footprints below `min_area_m2`.
    pub too_small: usize,
    /// Features whose geometry is not an area.
    pub not_areas: usize,
    /// Footprints past `max_buildings`.
    pub truncated: usize,
    /// The cap that produced `truncated`.
    pub cap: usize,
    /// Footprints the grammar or the bake refused, with the reason.
    pub refused: Vec<String>,
    /// Footprints standing on ground the level has not authored.
    pub off_terrain: usize,
    pub advisories: Vec<String>,
}

impl BuildingImportOutcome {
    pub fn summary(&self, layer: &str) -> String {
        let mut s = format!("{layer}: {} buildings", self.built.len());
        if self.coverage.features > 0 {
            s.push_str(&format!(
                " ({} floors from attributes, {} from heights, {} defaulted)",
                self.coverage.floors_from_attribute,
                self.coverage.floors_from_height,
                self.coverage.floors_defaulted
            ));
        }
        if self.too_small > 0 {
            s.push_str(&format!(", {} too small", self.too_small));
        }
        if self.truncated > 0 {
            s.push_str(&format!(
                ", {} NOT BUILT (the building cap is {}; raise it at the import door \
                 to build the whole layer)",
                self.truncated, self.cap
            ));
        }
        if !self.refused.is_empty() {
            s.push_str(&format!(", {} refused", self.refused.len()));
        }
        s
    }
}

/// The archetype a building's stated use names.
///
/// Deliberately a small table over the words municipal layers actually use, not
/// a taxonomy. Anything unrecognised is `None`, which the caller turns into the
/// default rather than into a silent house.
///
/// # It matches TOKENS, not substrings, and that is a measured requirement
///
/// The obvious spelling — `kind.contains("house")` — classifies a
/// **lighthouse** as a detached dwelling, and `contains("mall")` does the same
/// to anything *small*. A use code is a short phrase of whole words, so the
/// match is per word with a prefix (which is what makes `warehouse` cover
/// `warehouses` and `manufactur` cover `manufacturing`) rather than per
/// substring.
pub fn archetype_of(kind: &str) -> Option<ArchetypeId> {
    let k = kind.trim().to_ascii_lowercase();
    let tokens: Vec<&str> = k
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    let tok = |needles: &[&str]| {
        tokens
            .iter()
            .any(|t| needles.iter().any(|n| t.starts_with(n)))
    };
    if tok(&["office", "commercial", "business"]) {
        return Some(ArchetypeId::Office);
    }
    if tok(&["apartment", "multi", "condo", "tower", "flats"]) {
        return Some(ArchetypeId::Apartment);
    }
    if tok(&["industrial", "warehouse", "factory", "manufactur"]) {
        return Some(ArchetypeId::Industrial);
    }
    if tok(&["hotel", "motel", "hostel"]) {
        return Some(ArchetypeId::Hotel);
    }
    if tok(&["retail", "shop", "store", "mall", "supermarket"]) {
        return Some(ArchetypeId::Shop);
    }
    if tok(&["estate", "villa", "mansion"]) {
        return Some(ArchetypeId::Estate);
    }
    if tok(&["house", "detached", "residential", "dwelling", "yes"]) {
        return Some(ArchetypeId::House);
    }
    None
}

/// The XZ points of an area feature, or `None` when it is not an area.
fn area_points(g: &GeoGeometry) -> Option<Vec<DVec2>> {
    match g {
        GeoGeometry::Polygon { exterior, .. } if exterior.len() >= 3 => {
            Some(exterior.iter().map(|p| DVec2::new(p.x, p.z)).collect())
        }
        GeoGeometry::Polyline {
            points,
            closed: true,
        } if points.len() >= 3 => Some(points.iter().map(|p| DVec2::new(p.x, p.z)).collect()),
        _ => None,
    }
}

/// **Import a building-footprint layer as real buildings.**
///
/// One transaction, so one Ctrl+Z takes the whole import back.
pub fn import_footprints(
    project: &mut AssetProject,
    doc: &mut SceneDoc,
    layer: &GeoLayer,
    opts: &BuildingImportOptions,
) -> Result<BuildingImportOutcome, String> {
    let mut out = BuildingImportOutcome {
        cap: opts.max_buildings,
        ..Default::default()
    };

    // ── phase 1: read the layer and resolve the ground (borrows the doc) ────
    struct Planned {
        feature: usize,
        params: BuildingParams,
        frame: LotFrame,
        archetype: ArchetypeId,
        floors: u32,
        name: String,
    }
    let mut planned: Vec<Planned> = Vec::new();
    crate::gis::with_ground(doc, |ground| {
        for (i, f) in layer.features.iter().enumerate() {
            // The attribute pass runs over EVERY feature: the coverage is what
            // an author needs, and it must not be a sample of the cheap ones.
            let attrs = footprint_attrs(f, &opts.defaults);
            out.coverage.observe(&attrs);

            let Some(pts) = area_points(&f.geometry) else {
                out.not_areas += 1;
                continue;
            };
            let Some(mar) = inf_math::min_area_rect(&pts) else {
                out.not_areas += 1;
                continue;
            };
            if mar.area() < opts.min_area_m2 {
                out.too_small += 1;
                continue;
            }
            if planned.len() >= opts.max_buildings {
                out.truncated += 1;
                continue;
            }
            let centre = mar.center;
            let Some(base_y) = ground(centre.x, centre.y) else {
                out.off_terrain += 1;
                continue;
            };
            let archetype = opts
                .archetype
                .or_else(|| attrs.kind.as_deref().and_then(archetype_of))
                .unwrap_or(ArchetypeId::House);
            // The seed is a function of the FEATURE, not of the iteration, so
            // re-importing the same layer rebuilds the same city.
            let seed = opts
                .seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(i as u64);
            planned.push(Planned {
                feature: i,
                params: BuildingParams {
                    archetype,
                    footprint: Rect2 {
                        min: -mar.half,
                        max: mar.half,
                    },
                    base_y,
                    seed,
                    floors: attrs.floors,
                },
                frame: LotFrame::new(mar.center, mar.u),
                archetype,
                floors: attrs.floors,
                name: inf_gis::import::feature_name(f, &layer.name, i),
            });
        }
    });

    if planned.is_empty() {
        return Err(format!(
            "the layer {:?} produced no buildable footprints: {} feature(s) were not \
             areas, {} were smaller than {} m2, and {} stood on ground this level \
             has not authored (a building needs a terrain under it).",
            layer.name, out.not_areas, out.too_small, opts.min_area_m2, out.off_terrain
        ));
    }

    // ── phase 2: bake, write and spawn ──────────────────────────────────────
    let dir = project
        .content_dir(&opts.folder)
        .map_err(|e| format!("could not open the destination folder: {e}"))?;
    doc.begin_transaction("Import Buildings");
    for p in &planned {
        let baked =
            match crate::bake::bake_building_in(&p.params, p.frame, p.params.seed, opts.furnish) {
                Ok(b) => b,
                Err(e) => {
                    out.refused.push(format!("feature {}: {e}", p.feature));
                    continue;
                }
            };
        let height_m = f64::from(p.floors) * inf_pcg::archetype(p.archetype).floor_height;
        let mesh = match project.write_asset(
            &dir,
            &p.name,
            &baked.asset,
            Some(format!("gis:{}", layer.source_crs)),
            Vec::new(),
            None,
        ) {
            Ok(id) => id,
            Err(e) => {
                out.refused.push(format!(
                    "feature {}: could not write the mesh: {e}",
                    p.feature
                ));
                continue;
            }
        };
        let entity = doc.edit_create_mesh_asset(&p.name, mesh.0, None);
        doc.edit_set_transform(
            entity,
            inf_ecs::components::Transform::from_translation(baked.origin_world),
        );
        out.built.push(ImportedBuilding {
            entity,
            mesh,
            feature: p.feature,
            floors: p.floors,
            archetype: p.archetype,
            frame: p.frame,
            lot: p.params.footprint,
            height_m,
        });
    }
    doc.commit_transaction();

    out.advisories
        .extend(out.coverage.advisories().iter().map(|a| a.to_string()));
    out.advisories
        .extend(layer.advisories.iter().map(|a| a.to_string()));
    if out.off_terrain > 0 {
        out.advisories.push(format!(
            "{} footprint(s) stand on ground this level has not authored and were \
             skipped. Import the terrain that covers them first — a building placed \
             at a guessed elevation is a building nobody can find.",
            out.off_terrain
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use inf_gis::feature::{Attr, GeoFeature, LayerKind};
    use inf_terrain::{TerrainData, TerrainTile};

    fn doc_with_terrain() -> SceneDoc {
        let mut doc = SceneDoc::new();
        let guid = doc.edit_create(crate::ipc::SpawnKind::Empty, "Ground", None);
        let e = doc.world().entity_of(guid).unwrap();
        let res = 257u32;
        // Flat at 12 m, so a building's own height is the only thing that varies.
        let tile = TerrainTile::from_heights(
            res,
            DVec3::new(0.0, 12.0, 0.0),
            vec![0.0; (res * res) as usize],
        )
        .unwrap();
        let mut data = TerrainData::new(res, 2.0);
        data.insert_tile((0, 0), tile).ok();
        let mut t = inf_ecs::components::Transform::IDENTITY;
        t.translation = inf_ecs::math::Vec3d::new(-100.0, 12.0, -100.0);
        doc.world_mut().world_mut().entity_mut(e).insert((
            inf_ecs::components::Terrain {
                data,
                ..Default::default()
            },
            t,
        ));
        doc.world_mut().propagate();
        doc
    }

    /// A `w` x `d` footprint centred at `c`, turned by the 3-4-5 rotation.
    fn turned_footprint(c: DVec2, w: f64, d: f64, turned: bool) -> GeoGeometry {
        let (cs, sn) = if turned { (0.8f64, 0.6f64) } else { (1.0, 0.0) };
        let ring: Vec<DVec3> = [
            (-w * 0.5, -d * 0.5),
            (w * 0.5, -d * 0.5),
            (w * 0.5, d * 0.5),
            (-w * 0.5, d * 0.5),
        ]
        .iter()
        .map(|&(x, z)| {
            let p = DVec2::new(x * cs - z * sn, x * sn + z * cs) + c;
            DVec3::new(p.x, 0.0, p.y)
        })
        .collect();
        GeoGeometry::Polygon {
            exterior: ring,
            holes: Vec::new(),
        }
    }

    /// **The wire runs: a GIS attribute reaches `BuildingParams::floors`, and
    /// the WORLD is that many storeys tall.**
    ///
    /// Three footprints: one stating levels, one stating only a height, one
    /// stating neither. The arm asserts the resulting building's own height in
    /// metres, which is the thing an author sees, and the coverage counts that
    /// say which of the three routes each took.
    #[test]
    fn a_footprint_attribute_reaches_the_grammars_floor_count() {
        let tmp = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(tmp.path()).unwrap();
        let mut doc = doc_with_terrain();

        let mut l = GeoLayer::new("Downtown", LayerKind::Buildings, "EPSG:32610");
        let mut a = GeoFeature::new(turned_footprint(
            DVec2::new(-40.0, -40.0),
            24.0,
            16.0,
            false,
        ));
        a.attributes.insert("NUM_FLOORS".into(), Attr::Number(6.0));
        a.attributes
            .insert("BUILDING".into(), Attr::Text("office".into()));
        a.attributes.insert("NAME".into(), Attr::Text("Six".into()));
        let mut b = GeoFeature::new(turned_footprint(DVec2::new(20.0, -40.0), 24.0, 16.0, false));
        b.attributes.insert("HEIGHT".into(), Attr::Number(30.0));
        b.attributes
            .insert("NAME".into(), Attr::Text("Tall".into()));
        let mut c = GeoFeature::new(turned_footprint(DVec2::new(20.0, 20.0), 24.0, 16.0, false));
        c.attributes
            .insert("NAME".into(), Attr::Text("Plain".into()));
        l.features = vec![a, b, c];

        let out = import_footprints(&mut proj, &mut doc, &l, &Default::default())
            .expect("the import succeeds");
        assert_eq!(out.built.len(), 3, "{out:?}");
        assert_eq!(out.coverage.features, 3);
        assert_eq!(out.coverage.floors_from_attribute, 1);
        assert_eq!(out.coverage.floors_from_height, 1);
        assert_eq!(out.coverage.floors_defaulted, 1);

        let by_name = |n: &str| -> &ImportedBuilding {
            out.built
                .iter()
                .find(|b| {
                    doc.world()
                        .entity_of(b.entity)
                        .and_then(|e| doc.world().name_of(e))
                        .map(|s| s == n)
                        .unwrap_or(false)
                })
                .unwrap_or_else(|| panic!("{n} was not built"))
        };
        // 1. The stated count reached the grammar.
        let six = by_name("Six");
        assert_eq!(six.floors, 6);
        assert_eq!(six.archetype, ArchetypeId::Office, "the TYPE attribute too");
        // 2. …and the derived one did.
        assert_eq!(by_name("Tall").floors, 10, "30 m at 3 m a storey");
        // 3. …and the default is the default, not zero.
        assert_eq!(by_name("Plain").floors, inf_gis::buildings::DEFAULT_FLOORS);

        // **THE WORLD**: the six-storey building is six storeys of geometry tall.
        let mesh: inf_mesh::MeshAsset = proj.load_payload(six.mesh).unwrap();
        let span = mesh.bounds.max[1] - mesh.bounds.min[1];
        let expect = 6.0 * inf_pcg::archetype(ArchetypeId::Office).floor_height as f32;
        assert!(
            (span - expect).abs() < expect * 0.15,
            "a 6-storey office should be about {expect} m of geometry; the baked \
             mesh spans {span} m"
        );
        // …and the two-storey one is visibly shorter, which is the control that
        // makes the number above mean something.
        let plain: inf_mesh::MeshAsset = proj.load_payload(by_name("Plain").mesh).unwrap();
        let plain_span = plain.bounds.max[1] - plain.bounds.min[1];
        assert!(
            plain_span < span * 0.6,
            "the defaulted building is {plain_span} m against the 6-storey {span} m"
        );
        println!(
            "IB-5b: floors {}/{}/{} -> {span:.1} m and {plain_span:.1} m of geometry",
            six.floors,
            by_name("Tall").floors,
            by_name("Plain").floors
        );

        // One undo step takes all three back.
        let before = doc.order().len();
        assert!(doc.undo());
        assert_eq!(doc.order().len(), before - 3);
    }

    /// **A rotated footprint builds a rotated building** — IB-6, end to end
    /// through the import rather than through the grammar's own fixture.
    #[test]
    fn a_rotated_footprint_stands_on_its_own_lot() {
        let tmp = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(tmp.path()).unwrap();
        let mut doc = doc_with_terrain();

        let mut l = GeoLayer::new("West End", LayerKind::Buildings, "EPSG:32610");
        let mut f = GeoFeature::new(turned_footprint(DVec2::new(0.0, 0.0), 30.0, 12.0, true));
        f.attributes.insert("levels".into(), Attr::Number(3.0));
        l.features = vec![f];

        let out = import_footprints(&mut proj, &mut doc, &l, &Default::default()).unwrap();
        assert_eq!(out.built.len(), 1);
        let b = &out.built[0];
        assert!(
            (b.frame.u - DVec2::new(0.8, 0.6)).length() < 1e-9,
            "the lot's long side is the footprint's, not the compass's: {:?}",
            b.frame.u
        );
        assert!(
            (b.lot.size() - DVec2::new(30.0, 12.0)).length() < 1e-9,
            "the lot is 30 x 12 however it is turned: {:?}",
            b.lot.size()
        );
        // THE WORLD: the baked mesh is longer along the LOT's axis than along the
        // world's, which an axis-aligned collapse could never produce.
        let mesh: inf_mesh::MeshAsset = proj.load_payload(b.mesh).unwrap();
        let (mut ext_u, mut ext_v) = (0.0f64, 0.0f64);
        for sub in &mesh.submeshes {
            for v in &sub.vertices {
                let p = DVec2::new(v.position[0] as f64, v.position[2] as f64);
                ext_u = ext_u.max(p.dot(b.frame.u).abs());
                ext_v = ext_v.max(p.dot(b.frame.v()).abs());
            }
        }
        assert!(
            ext_u > 14.0 && ext_u < 16.0,
            "half the 30 m side along the lot's own axis: {ext_u}"
        );
        assert!(
            ext_v > 5.0 && ext_v < 7.0,
            "half the 12 m side across it: {ext_v}"
        );
        println!("IB-6 import: lot extents {ext_u:.2} x {ext_v:.2} m about its own axes");
    }

    /// The cap on the expensive half is reported with the number to raise, and
    /// the attribute pass still covers the WHOLE layer.
    #[test]
    fn the_building_cap_truncates_the_bake_and_not_the_attribute_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(tmp.path()).unwrap();
        let mut doc = doc_with_terrain();
        let mut l = GeoLayer::new("Blocks", LayerKind::Buildings, "EPSG:32610");
        for i in 0..8 {
            let mut f = GeoFeature::new(turned_footprint(
                DVec2::new(-60.0 + (i % 4) as f64 * 40.0, -60.0 + (i / 4) as f64 * 40.0),
                20.0,
                14.0,
                false,
            ));
            f.attributes.insert("levels".into(), Attr::Number(2.0));
            l.features.push(f);
        }
        let out = import_footprints(
            &mut proj,
            &mut doc,
            &l,
            &BuildingImportOptions {
                max_buildings: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out.built.len(), 3);
        assert_eq!(out.truncated, 5);
        assert_eq!(
            out.coverage.features, 8,
            "the ATTRIBUTE pass runs over the whole layer; only the bake is capped"
        );
        let s = out.summary("Blocks");
        assert!(
            s.contains("NOT BUILT") && s.contains("the building cap is 3"),
            "{s}"
        );
    }

    /// A footprint with no ground under it is skipped and named, not placed at a
    /// guessed elevation.
    #[test]
    fn a_footprint_off_the_terrain_is_skipped_and_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(tmp.path()).unwrap();
        let mut doc = doc_with_terrain();
        let mut l = GeoLayer::new("Far", LayerKind::Buildings, "EPSG:32610");
        let mut f = GeoFeature::new(turned_footprint(
            DVec2::new(9000.0, 9000.0),
            20.0,
            14.0,
            false,
        ));
        f.attributes.insert("levels".into(), Attr::Number(2.0));
        l.features = vec![f];
        let e = import_footprints(&mut proj, &mut doc, &l, &Default::default()).unwrap_err();
        assert!(e.contains("has not authored"), "{e}");

        // A tiny footprint is a shed, and is counted as one.
        let mut small = GeoLayer::new("Sheds", LayerKind::Buildings, "EPSG:32610");
        small.features.push(GeoFeature::new(turned_footprint(
            DVec2::ZERO,
            3.0,
            3.0,
            false,
        )));
        let e = import_footprints(&mut proj, &mut doc, &small, &Default::default()).unwrap_err();
        assert!(e.contains("smaller than"), "{e}");
    }

    /// The archetype table reads the words municipal layers use, and refuses to
    /// guess at the ones it does not know.
    #[test]
    fn the_archetype_table_reads_real_use_codes() {
        assert_eq!(archetype_of("OFFICE"), Some(ArchetypeId::Office));
        assert_eq!(archetype_of("Commercial"), Some(ArchetypeId::Office));
        assert_eq!(archetype_of("warehouse"), Some(ArchetypeId::Industrial));
        assert_eq!(archetype_of("apartments"), Some(ArchetypeId::Apartment));
        assert_eq!(archetype_of("retail"), Some(ArchetypeId::Shop));
        assert_eq!(archetype_of("hotel"), Some(ArchetypeId::Hotel));
        assert_eq!(archetype_of("detached house"), Some(ArchetypeId::House));
        assert_eq!(archetype_of("yes"), Some(ArchetypeId::House), "OSM's own");
        assert_eq!(
            archetype_of("residential_multi"),
            Some(ArchetypeId::Apartment)
        );
        assert_eq!(archetype_of("manufacturing"), Some(ArchetypeId::Industrial));
        // **A lighthouse is not a house, and a small shed is not a mall.** The
        // substring spelling of this table classified both, which is exactly the
        // kind of wrong that never looks wrong in a report.
        assert_eq!(archetype_of("lighthouse_platform"), None);
        assert_eq!(archetype_of("small storage"), None);
        assert_eq!(archetype_of(""), None);
    }
}
