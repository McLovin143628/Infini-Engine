//! **Hydrology authoring** (P20.4, Ring 1) — the terrain-aware half of the water
//! tools.
//!
//! The cook can only ask questions a `.inf_terrain` answers *structurally*,
//! because it never pages a tile in (see `uphill_rivers`' docs). The editor has
//! the terrain resident in `Terrain::data` while the author is standing on it, so
//! this is where the terrain-aware checks live:
//!
//! * [`lake_preview`] — where a still-water level lands on the ground inside a
//!   rectangle, with the waterline as drawable segments;
//! * [`river_report`] — the whole verdict on one river: its length, its fall, the
//!   cook's two terrain-free advisories re-run so the tool says the same thing the
//!   build will, and the terrain-aware [`inf_water::bed_conflicts`] the cook
//!   cannot make;
//! * [`water_defaults`] — the **biome water hint** (P19.2's `BiomeDef::water_hint`,
//!   unread until now) turned into suggested numbers for a new body.
//!
//! # The hint is a DEFAULT PROVIDER, not a generator
//!
//! `water_defaults` answers "what should this new lake's level be, given where you
//! clicked?" and nothing else. It does not place water, it does not fill basins,
//! and painting a biome does not make water appear. That is a deliberate v1
//! boundary: a hint that silently spawned geometry would be a generator wearing a
//! hint's name, and the field's own doc calls it a hint.
//!
//! # Determinism
//!
//! Every function here is a pure function of `(the document, the arguments)`. The
//! terrain authority is the **lowest-`Guid` terrain that answers**, matching
//! `inf_ecs::hydro::terrain_flow` and the runtime's `terrain_height_at`; no
//! `HashMap` iteration order, no clock, no camera. None of it reaches the fixed
//! step — these are authoring queries, and the sim's water is
//! `inf_water::WaterSurface` and only that.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    GlobalTransform, Spline, SplineInterp, Terrain, Transform, WaterBody, WaterKind,
};
use inf_terrain::BiomeSet;
use inf_water::{RiverPath, RiverProfile};

// Re-exported so Ring 2 can name the report's own types without linking
// `inf-water` itself: the commands are a projection of `RiverReport`, and a
// second dependency edge for three type names would be a worse trade than a
// re-export.
pub use inf_water::hydro::{BedConflict, BedIssue, FillPreview};
pub use inf_water::{UphillSpan, UPHILL_TOLERANCE_M};

use crate::scene::SceneDoc;

/// Tolerance the **editor** applies to the terrain-aware bed checks, metres.
///
/// Larger than the cook's `RIVER_UPHILL_TOLERANCE_M` (0.5 m) because it absorbs
/// one more source of wobble: the cook's checks read the spline alone, while
/// these sample a **bilinear heightfield** along a curve that crosses tile
/// diagonals. An author who sculpted a bank to within a metre of the water has
/// done the job, and a tool that nagged about the last few centimetres would be a
/// tool people turn off.
pub const BED_TOLERANCE_M: f64 = 1.0;

/// The suggested numbers for placing water at a world point.
///
/// Every field is a *starting value* the author can overwrite; nothing here is
/// persisted or enforced.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterDefaults {
    /// Suggested still-water level, metres of world Y.
    pub level_m: f64,
    /// Suggested river width, metres.
    pub river_width_m: f64,
    /// Suggested river depth, metres.
    pub river_depth_m: f64,
    /// The terrain height under the point, if there is terrain there.
    pub ground_m: Option<f64>,
    /// The painted biome id under the point (`0` = unassigned / no terrain).
    pub biome_id: u8,
    /// Its display name, empty when there is none.
    pub biome_name: String,
    /// Whether [`level_m`](Self::level_m) came from the biome's
    /// `water_hint` rather than from the ground.
    pub from_biome_hint: bool,
}

/// Everything the river tool reports about one river.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RiverReport {
    /// Centreline length, metres. `0` for a river with no usable spline.
    pub length_m: f64,
    /// Control points on the spline.
    pub points: usize,
    /// Elevation at the source minus elevation at the mouth, metres — positive
    /// when the river descends, read in the direction it *flows* (so a negative
    /// `river_flow_m_s` flips it).
    pub fall_m: f64,
    /// Surface stretches that climb in the flow direction — the P20.1 cook
    /// advisory, re-run here so the tool says what the build will.
    pub surface_climbs: Vec<UphillSpan>,
    /// Authored-bed stretches that climb — the P20.4 cook advisory, likewise.
    pub bed_climbs: Vec<UphillSpan>,
    /// Terrain-aware conflicts. **Only the editor can produce these**; the cook
    /// has no heightfield.
    pub bed_conflicts: Vec<BedConflict>,
    /// How many frames the terrain answered for, and how many there are — a
    /// report over 3 of 200 frames is not a clean bill of health, and the tool
    /// says so rather than implying one.
    pub sampled_frames: usize,
    /// Total frames on the centreline.
    pub total_frames: usize,
}

impl RiverReport {
    /// Whether anything at all is worth telling the author.
    pub fn is_clean(&self) -> bool {
        self.surface_climbs.is_empty()
            && self.bed_climbs.is_empty()
            && self.bed_conflicts.is_empty()
    }
}

/// The world-space terrain height under `(x, z)` — the editor twin of the
/// player's `terrain_height_at`, and the sampler every function here feeds to
/// `inf-water`.
///
/// `None` where no terrain answers (a hole, off the authored extent, or a level
/// with no terrain). It resolves terrains in ascending `Guid` order and takes the
/// first answer, which is the same rule `inf_ecs::hydro::terrain_flow` uses —
/// stated in both places because two different rules would put a river's
/// validation on a different ground from its foam.
pub fn terrain_height_at(doc: &SceneDoc, x: f64, z: f64) -> Option<f64> {
    for (origin, data) in terrains(doc) {
        if let Some(h) = data.height_at(DVec2::new(x - origin.x, z - origin.z)) {
            return Some(h + origin.y);
        }
    }
    None
}

/// `(origin, data)` for every non-empty terrain, ascending by `Guid`.
fn terrains(doc: &SceneDoc) -> Vec<(DVec3, &inf_terrain::TerrainData)> {
    let world = doc.world();
    let w = world.world();
    let mut found: Vec<(Uuid, inf_ecs::Entity)> = Vec::new();
    for guid in doc.order() {
        let Some(e) = world.entity_of(*guid) else {
            continue;
        };
        let Some(t) = w.get::<Terrain>(e) else {
            continue;
        };
        if t.data.is_empty() {
            continue;
        }
        found.push((*guid, e));
    }
    found.sort_by_key(|(g, _)| *g);
    found
        .into_iter()
        .filter_map(|(_, e)| {
            let data = &w.get::<Terrain>(e)?.data;
            let origin = w
                .get::<GlobalTransform>(e)
                .map(|g| g.translation())
                .or_else(|| w.get::<Transform>(e).map(|t| t.translation.to_dvec3()))
                .unwrap_or(DVec3::ZERO);
            Some((origin, data))
        })
        .collect()
}

/// The painted biome id under `(x, z)`, and the terrain that answered.
fn biome_at(doc: &SceneDoc, x: f64, z: f64) -> u8 {
    for (origin, data) in terrains(doc) {
        if let Some(id) = data.biome_at(DVec2::new(x - origin.x, z - origin.z)) {
            return id;
        }
    }
    inf_terrain::UNASSIGNED_BIOME
}

/// Suggested numbers for a new water body at `(x, z)` (P20.4).
///
/// `set` is the `.inf_biomes` the terrain binds, already resolved by the caller
/// (Ring 1 does not reach into the asset database from here — the Ring-2 command
/// owns that, exactly as `terrain_biomes` does).
///
/// The rule, in order:
///
/// 1. the painted biome's [`BiomeDef::water_hint`](inf_terrain::BiomeDef) if it
///    has one — **this is what P19.2 wrote the field for**, and P20.4 is its
///    first reader;
/// 2. otherwise the ground height under the point, which is where the author is
///    pointing;
/// 3. otherwise `0` — a plausible sea level and the same answer
///    `terrain.height_at` gives a terrain-less world.
///
/// River width and depth come from the component's own defaults; a biome does not
/// carry them, and inventing per-biome river dimensions out of a *level* hint
/// would be making data up.
pub fn water_defaults(doc: &SceneDoc, set: Option<&BiomeSet>, x: f64, z: f64) -> WaterDefaults {
    let ground = terrain_height_at(doc, x, z);
    let id = biome_at(doc, x, z);
    let def = set.and_then(|s| s.biomes.iter().find(|b| b.id == id));
    let hint = def.and_then(|d| d.water_hint).map(|h| h as f64);
    let defaults = WaterBody::default();
    WaterDefaults {
        level_m: hint.or(ground).unwrap_or(0.0),
        river_width_m: defaults.river_width_start_m,
        river_depth_m: defaults.river_depth_start_m,
        ground_m: ground,
        biome_id: id,
        biome_name: def.map(|d| d.name.clone()).unwrap_or_default(),
        from_biome_hint: hint.is_some(),
    }
}

/// Where a still-water level would land inside a rectangle, against the
/// document's terrain (P20.4). A thin wrapper that binds
/// [`inf_water::fill_preview`] to this document's height authority.
pub fn lake_preview(
    doc: &SceneDoc,
    center: DVec2,
    half_extent: DVec2,
    level_m: f64,
    resolution: u32,
) -> FillPreview {
    inf_water::fill_preview(
        |p| terrain_height_at(doc, p.x, p.y),
        center,
        half_extent,
        level_m,
        resolution,
    )
}

/// Build the Ring-0 [`RiverPath`] a `WaterBody::River` entity describes, in world
/// space — the same construction the two projectors and the cook perform.
///
/// `None` when the entity is not a river, has no `Spline`, or its spline is
/// degenerate. Shared by [`river_report`] and by the viewport preview so the tool
/// and the report can never be looking at two different ribbons.
pub fn river_path_of(doc: &SceneDoc, guid: Uuid) -> Option<RiverPath> {
    let world = doc.world();
    let e = world.entity_of(guid)?;
    let w = world.world();
    let body = w.get::<WaterBody>(e)?;
    if body.kind != WaterKind::River {
        return None;
    }
    let spline = w.get::<Spline>(e)?;
    if spline.points.len() < 2 {
        return None;
    }
    let affine = w
        .get::<GlobalTransform>(e)
        .map(|g| g.0)
        .unwrap_or(glam::DAffine3::IDENTITY);
    let points: Vec<DVec3> = spline
        .points
        .iter()
        .map(|p| affine.transform_point3(p.to_dvec3()))
        .collect();
    let interp = match spline.interp {
        SplineInterp::Linear => inf_math::spline::SplineInterp::Linear,
        SplineInterp::CatmullRom => inf_math::spline::SplineInterp::CatmullRom,
    };
    let profile = RiverProfile {
        width_start_m: body.river_width_start_m.max(0.0),
        width_end_m: body.river_width_end_m.max(0.0),
        depth_start_m: body.river_depth_start_m.max(0.0),
        depth_end_m: body.river_depth_end_m.max(0.0),
        flow_speed_m_s: body.river_flow_m_s,
    };
    let path = RiverPath::from_points(&points, spline.closed, interp, &profile);
    (!path.is_empty()).then_some(path)
}

/// The full verdict on one river (P20.4).
///
/// The two terrain-free checks are re-run here rather than imported from the
/// packager because Ring 1 must not depend on Ring 2 — but they call the *same*
/// `inf_water::river::uphill_spans` over the *same* profiles at the *same*
/// tolerance, so the tool and the build agree by construction rather than by
/// two people remembering the same rule.
pub fn river_report(doc: &SceneDoc, guid: Uuid, uphill_tolerance_m: f64) -> RiverReport {
    let Some(path) = river_path_of(doc, guid) else {
        return RiverReport::default();
    };
    let world = doc.world();
    let reversed = world
        .entity_of(guid)
        .and_then(|e| world.world().get::<WaterBody>(e))
        .is_some_and(|b| b.river_flow_m_s < 0.0);

    // Read both profiles the way the water actually goes — the cook's rule.
    let flip = |mut p: Vec<(f64, f64)>| {
        if reversed {
            let total = path.length_m;
            p.reverse();
            for (s, _) in p.iter_mut() {
                *s = total - *s;
            }
        }
        p
    };
    let surface = flip(path.surface_profile());
    let bed = flip(path.bed_profile_from_depth());

    // A CLOSED river is a loop: it cannot help regaining what it loses, so the
    // climb checks are skipped exactly as the cook skips them. The terrain-aware
    // conflicts still apply — a loop can perfectly well be buried.
    let (surface_climbs, bed_climbs) = if path.closed {
        (Vec::new(), Vec::new())
    } else {
        (
            inf_water::river::uphill_spans(&surface, uphill_tolerance_m),
            inf_water::river::uphill_spans(&bed, uphill_tolerance_m),
        )
    };

    let sampled = path.bed_profile(|p| terrain_height_at(doc, p.x, p.y)).len();
    let conflicts =
        inf_water::bed_conflicts(&path, |p| terrain_height_at(doc, p.x, p.y), BED_TOLERANCE_M);

    let fall_m = match (surface.first(), surface.last()) {
        (Some((_, a)), Some((_, b))) => a - b,
        _ => 0.0,
    };
    RiverReport {
        length_m: path.length_m,
        points: world
            .entity_of(guid)
            .and_then(|e| world.world().get::<Spline>(e))
            .map(|s| s.points.len())
            .unwrap_or(0),
        fall_m,
        surface_climbs,
        bed_climbs,
        bed_conflicts: conflicts,
        sampled_frames: sampled,
        total_frames: path.frames.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;
    use inf_ecs::math::Vec2d;
    use inf_terrain::BiomeDef;

    const TERRAIN: Uuid = Uuid::from_u128(0x2004_1001);
    const RIVER: Uuid = Uuid::from_u128(0x2004_1002);

    /// A doc with one terrain: a plane tilted along +X, `y = 20 - x * 0.05`, over
    /// a 256 m tile.
    fn doc_with_terrain() -> SceneDoc {
        let mut doc = SceneDoc::new();
        doc.create_with_guid(TERRAIN, SpawnKind::Empty, "Ground", None);
        let e = doc.world().entity_of(TERRAIN).unwrap();
        let mut t = Terrain {
            meters_per_sample: 4.0,
            tile_resolution: 65,
            data: inf_terrain::TerrainData::new(65, 4.0),
            ..Terrain::default()
        };
        t.data.author_tile((0, 0), |x, z| {
            let _ = z;
            20.0 - x * 0.05
        });
        doc.world_mut().world_mut().entity_mut(e).insert(t);
        doc.world_mut().mark_dirty();
        doc.world_mut().propagate();
        doc
    }

    #[test]
    fn the_height_authority_answers_and_admits_when_it_cannot() {
        let doc = doc_with_terrain();
        let h = terrain_height_at(&doc, 100.0, 50.0).expect("on the tile");
        assert!((h - (20.0 - 100.0 * 0.05)).abs() < 1e-6, "{h}");
        // Off the authored extent: `None`, not 0.
        assert!(terrain_height_at(&doc, 100_000.0, 0.0).is_none());
        // A doc with no terrain at all likewise.
        assert!(terrain_height_at(&SceneDoc::new(), 0.0, 0.0).is_none());
    }

    /// **The biome water hint's first reader** (P19.2 → P20.4).
    #[test]
    fn the_biome_water_hint_becomes_the_suggested_level() {
        let mut doc = doc_with_terrain();
        // Paint biome 3 over the whole tile.
        {
            let e = doc.world().entity_of(TERRAIN).unwrap();
            let w = doc.world_mut().world_mut();
            let mut t = w.get_mut::<Terrain>(e).unwrap();
            let res = t.tile_resolution;
            let tile = t.data.get_tile_mut((0, 0)).unwrap();
            for j in 0..res {
                for i in 0..res {
                    tile.set_biome_sample(res, i, j, 3);
                }
            }
        }
        let mut set = BiomeSet::starter();
        set.biomes = vec![BiomeDef {
            water_hint: Some(6.5),
            ..BiomeDef::new(3, "Marsh")
        }];

        let d = water_defaults(&doc, Some(&set), 100.0, 50.0);
        assert_eq!(d.biome_id, 3);
        assert_eq!(d.biome_name, "Marsh");
        assert!(d.from_biome_hint, "the hint was not consulted");
        assert!((d.level_m - 6.5).abs() < 1e-9, "{}", d.level_m);
        // The ground is still reported, so the tool can show both.
        assert!((d.ground_m.unwrap() - 15.0).abs() < 1e-6);

        // ANTI-VACUITY: with NO hint, the suggestion falls back to the ground —
        // so the assertion above is about the hint and not about a constant.
        set.biomes[0].water_hint = None;
        let d = water_defaults(&doc, Some(&set), 100.0, 50.0);
        assert!(!d.from_biome_hint);
        assert!((d.level_m - 15.0).abs() < 1e-6, "{}", d.level_m);
        // …and with no biome set at all, likewise.
        let d = water_defaults(&doc, None, 100.0, 50.0);
        assert!(!d.from_biome_hint);
        assert!((d.level_m - 15.0).abs() < 1e-6);
        assert_eq!(d.biome_name, "");
        // Off the terrain entirely: a defined answer, not a panic.
        let d = water_defaults(&doc, None, 1e6, 1e6);
        assert_eq!(d.level_m, 0.0);
        assert!(d.ground_m.is_none());
        assert_eq!(d.biome_id, inf_terrain::UNASSIGNED_BIOME);
        // The river dimensions are the component's defaults, never invented.
        assert_eq!(d.river_width_m, WaterBody::default().river_width_start_m);
    }

    #[test]
    fn a_lake_preview_reads_the_documents_terrain() {
        let doc = doc_with_terrain();
        // The tile spans x ∈ [0, 256], y = 20 − 0.05x ∈ [7.2, 20]. A level of 15
        // submerges everything past x = 100.
        let p = lake_preview(
            &doc,
            DVec2::new(128.0, 128.0),
            DVec2::splat(120.0),
            15.0,
            64,
        );
        assert!(p.has_ground());
        assert!(
            (p.covered_fraction - 0.61).abs() < 0.05,
            "covered {}",
            p.covered_fraction
        );
        assert!(!p.waterline.is_empty(), "no shore to draw");
        // The waterline runs across Z at x ≈ 100 — a straight contour on a plane
        // tilted in X alone.
        for [a, b] in &p.waterline {
            for v in [a, b] {
                assert!((v.x - 100.0).abs() < 5.0, "waterline wandered: {v:?}");
            }
        }
        // A level below the whole tile covers nothing.
        let dry = lake_preview(&doc, DVec2::new(128.0, 128.0), DVec2::splat(120.0), 0.0, 32);
        assert_eq!(dry.covered_fraction, 0.0);
    }

    fn add_river(doc: &mut SceneDoc, points: &[DVec3], depth_start: f64, depth_end: f64) {
        let guid = doc.edit_create_river(RIVER.to_string().as_str(), points, 8.0, depth_start, 1.5);
        // The tool creates with a fixed guid in production; here re-key by editing
        // the created entity's profile.
        doc.edit_set_river_profile(guid, 8.0, 8.0, depth_start, depth_end, 1.5);
    }

    #[test]
    fn a_river_report_re_runs_the_cook_checks_and_adds_the_terrain_one() {
        let mut doc = doc_with_terrain();
        // A river running along +X, following the terrain closely: the ground is
        // 20 − 0.05x, so a surface 0.5 m above it and 1 m deep is well bedded.
        let pts: Vec<DVec3> = (0..=5)
            .map(|i| {
                let x = 20.0 + i as f64 * 40.0;
                DVec3::new(x, 20.5 - x * 0.05, 128.0)
            })
            .collect();
        add_river(&mut doc, &pts, 1.0, 1.0);
        let guid = *doc.order().last().unwrap();
        let r = river_report(&doc, guid, 0.5);
        assert!(r.length_m > 190.0, "{r:?}");
        assert!(r.fall_m > 9.0, "the river should fall ~10 m: {r:?}");
        assert!(r.is_clean(), "a well-bedded river was reported: {r:?}");
        assert_eq!(
            r.sampled_frames, r.total_frames,
            "the terrain answered for every frame"
        );

        // Now BURY it: lift the terrain far above the water by dropping the
        // river 30 m. Only the editor can catch this.
        let low: Vec<DVec3> = pts
            .iter()
            .map(|p| *p - DVec3::new(0.0, 30.0, 0.0))
            .collect();
        let mut doc2 = doc_with_terrain();
        add_river(&mut doc2, &low, 1.0, 1.0);
        let g2 = *doc2.order().last().unwrap();
        let r2 = river_report(&doc2, g2, 0.5);
        assert!(!r2.is_clean());
        assert!(r2.surface_climbs.is_empty(), "the surface is fine: {r2:?}");
        assert_eq!(r2.bed_conflicts.len(), 1, "{:?}", r2.bed_conflicts);
        assert_eq!(r2.bed_conflicts[0].issue, inf_water::BedIssue::Buried);

        // And a BED climb (terrain-free) is reported too, from the depth taper
        // alone — the P20.4 cook advisory, seen in the tool.
        let mut doc3 = doc_with_terrain();
        add_river(&mut doc3, &pts, 12.0, 0.5);
        let g3 = *doc3.order().last().unwrap();
        let r3 = river_report(&doc3, g3, 0.5);
        assert!(!r3.bed_climbs.is_empty(), "{r3:?}");
        assert!(r3.surface_climbs.is_empty());
    }

    #[test]
    fn a_report_on_a_non_river_is_empty_rather_than_wrong() {
        let mut doc = doc_with_terrain();
        let lake =
            doc.edit_create_lake("Lake", DVec3::new(50.0, 0.0, 50.0), Vec2d::splat(20.0), 8.0);
        let r = river_report(&doc, lake, 0.5);
        assert_eq!(r, RiverReport::default());
        assert!(r.is_clean());
        assert!(river_path_of(&doc, lake).is_none());
        // …and on an entity that does not exist at all.
        assert_eq!(river_report(&doc, Uuid::nil(), 0.5), RiverReport::default());
    }
}
