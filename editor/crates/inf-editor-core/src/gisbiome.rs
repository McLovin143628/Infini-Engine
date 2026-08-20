//! **Land cover becomes biome ids** (IB-5a, Ring 1).
//!
//! The Wave-G disposition memo's G10 row, in full: *"there is still no path from
//! a raster to a `BiomeSet` — nothing decodes a land-cover image, nothing writes
//! biome ids, and the classifier has no caller outside its own tests."* This
//! module is the caller.
//!
//! # The two routes a land-cover layer takes, and why there are two
//!
//! A published land-cover or zoning layer states its class in one of two ways
//! and never in a third:
//!
//! * **As a code** — `NDVI`, `IMPERV_PCT`, `LC_CODE`, a canopy percentage. A
//!   *number*, whose classes are the shape of its own distribution. That is what
//!   [`inf_gis::classify_to_ids`] is for, and why the default method is Jenks
//!   natural breaks rather than equal interval: real geographic distributions
//!   are lumpy, and equal-width classes put nine tenths of a map in one biome.
//! * **As a name** — `"Coniferous Forest"`, `"Wetland"`, `"Urban"`. Then the
//!   classes are the `BiomeSet`'s own biome names and there is nothing to
//!   classify; matching them is the whole job, and a name that matches nothing
//!   is **reported**, not silently assigned to biome 1.
//!
//! # Biome id 0 is reserved, so classes start at 1
//!
//! `UNASSIGNED_BIOME` is `0` and a `BiomeDef` may not claim it. The classifier
//! numbers from zero, so every class id is shifted by one on the way in — once,
//! here, at the seam where the two numberings meet.
//!
//! # Painting never authors ground
//!
//! Like every brush in `inf_terrain::biomepaint`, a fill writes only into tiles
//! the terrain already has. A polygon over ground the level has not authored
//! paints nothing there and the count says so, rather than materialising a
//! landscape nobody asked for.

use std::collections::BTreeMap;

use glam::DVec2;
use uuid::Uuid;

use inf_gis::feature::{GeoGeometry, GeoLayer};
use inf_gis::ClassifyMethod;
use inf_terrain::BiomeFill;

use crate::scene::SceneDoc;

/// The attribute spellings a land-cover class is stored under, most specific
/// first.
/// The attribute spellings a land-cover class is stored under, most specific
/// first — **named** classes then **numeric** ones, because a layer that has
/// both wants its name.
///
/// Lookup folds case *and* separators ([`inf_gis::GeoFeature::attr`]), so
/// `CANOPY_PCT`, `canopy pct` and `CanopyPct` are all one spelling here.
pub const CLASS_FIELDS: [&str; 18] = [
    "biome",
    "landcover",
    "land_cover",
    "lc_code",
    "lccode",
    "cover",
    "class",
    "landuse",
    "land_use",
    "zoning",
    "zone",
    "type",
    // Numeric land cover: the continuous measures a portal publishes when it
    // has no class table at all, and exactly what natural breaks is for.
    "canopy_pct",
    "canopy",
    "ndvi",
    "imperv_pct",
    "impervious",
    "cover_pct",
];

/// How many classes a numeric land-cover layer is cut into by default.
///
/// Eight is the number a starter [`inf_terrain::BiomeSet`] has room for and the
/// number a reader can tell apart on a map. It is a default, not a ceiling —
/// [`inf_gis::classify::MAX_CLASSES`] is 64 and a `u8` id reaches 255.
pub const DEFAULT_CLASSES: usize = 8;

/// What a land-cover import may do.
#[derive(Clone, Debug, PartialEq)]
pub struct BiomeImportOptions {
    /// The attribute to read. Empty probes [`CLASS_FIELDS`].
    pub attribute: String,
    /// How a **numeric** attribute is cut into classes.
    pub method: ClassifyMethod,
    /// How many classes a numeric attribute is cut into.
    pub classes: usize,
    /// Biome ids to assign, in class order. Empty numbers them `1..=classes`.
    ///
    /// A caller with a real `BiomeSet` passes its own ids here, so class *k*
    /// means the biome the author named rather than the k-th one.
    pub ids: Vec<u8>,
}

impl Default for BiomeImportOptions {
    fn default() -> Self {
        Self {
            attribute: String::new(),
            method: ClassifyMethod::NaturalBreaks,
            classes: DEFAULT_CLASSES,
            ids: Vec::new(),
        }
    }
}

/// What a land-cover import wrote.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BiomeImportOutcome {
    /// The attribute that was actually read.
    pub attribute: String,
    /// `true` when the values were numbers cut by the classifier, `false` when
    /// they were names matched against the biome set.
    pub numeric: bool,
    /// The class fenceposts, for a numeric layer. Empty for a named one.
    pub breaks: Vec<f64>,
    /// How many terrain samples each biome id claimed.
    pub per_biome: BTreeMap<u8, usize>,
    /// Polygons that painted nothing — outside the terrain, or smaller than a
    /// sample.
    pub empty_polygons: usize,
    /// Features with no usable class value.
    pub unclassified: usize,
    /// Features whose geometry is not an area.
    pub not_areas: usize,
    pub advisories: Vec<String>,
}

impl BiomeImportOutcome {
    pub fn painted(&self) -> usize {
        self.per_biome.values().sum()
    }

    pub fn summary(&self, layer: &str) -> String {
        let mut s = format!(
            "{layer}: {} samples over {} biome(s) from {:?}",
            self.painted(),
            self.per_biome.len(),
            self.attribute
        );
        if self.unclassified > 0 {
            s.push_str(&format!(", {} features unclassified", self.unclassified));
        }
        if self.empty_polygons > 0 {
            s.push_str(&format!(
                ", {} polygons painted nothing (outside the terrain)",
                self.empty_polygons
            ));
        }
        s
    }
}

/// The rings of an area feature, as XZ.
fn rings_of(g: &GeoGeometry) -> Option<(Vec<DVec2>, Vec<Vec<DVec2>>)> {
    match g {
        GeoGeometry::Polygon { exterior, holes } => Some((
            exterior.iter().map(|p| DVec2::new(p.x, p.z)).collect(),
            holes
                .iter()
                .map(|h| h.iter().map(|p| DVec2::new(p.x, p.z)).collect())
                .collect(),
        )),
        // A closed polyline IS a ring; published parcel layers export both
        // spellings and refusing one of them is refusing half the files.
        GeoGeometry::Polyline {
            points,
            closed: true,
        } if points.len() >= 3 => Some((
            points.iter().map(|p| DVec2::new(p.x, p.z)).collect(),
            Vec::new(),
        )),
        _ => None,
    }
}

/// **Paint a land-cover layer's classes into a terrain's biome ids.**
///
/// One undo step for the whole import (see
/// [`SceneDoc::edit_commit_biome_fill`]). Returns a refusal when the entity is
/// not a terrain or the layer has no usable class attribute at all — the two
/// cases where painting anything would be a guess.
pub fn paint_biomes_from_layer(
    doc: &mut SceneDoc,
    terrain: Uuid,
    layer: &GeoLayer,
    opts: &BiomeImportOptions,
    biome_names: &[(u8, String)],
) -> Result<BiomeImportOutcome, String> {
    let is_terrain = doc.world().entity_of(terrain).is_some_and(|e| {
        doc.world()
            .world()
            .get::<inf_ecs::components::Terrain>(e)
            .is_some()
    });
    if !is_terrain {
        return Err(
            "the target entity is not a terrain, so it has no biome ids to paint into. \
             Import a heightmap first, or pick the terrain in the outliner."
                .to_string(),
        );
    }
    let mut out = BiomeImportOutcome::default();

    // ── which attribute, and what kind of value ─────────────────────────────
    let probe: Vec<&str> = if opts.attribute.trim().is_empty() {
        CLASS_FIELDS.to_vec()
    } else {
        vec![opts.attribute.trim()]
    };
    let mut chosen: Option<String> = None;
    for name in &probe {
        if layer.features.iter().any(|f| f.attr(name).is_some()) {
            // Report the field as the layer spells it, not as we probed for it.
            chosen = layer
                .features
                .iter()
                .find_map(|f| {
                    f.attributes
                        .keys()
                        .find(|k| f.attr(name).is_some() && k.eq_ignore_ascii_case(name))
                        .cloned()
                })
                .or_else(|| Some((*name).to_string()));
            break;
        }
    }
    let Some(attribute) = chosen else {
        return Err(format!(
            "no land-cover class attribute was found on {:?}. This engine probes \
             {} by name; state the field explicitly if it is called something else. \
             The layer's fields are: {}",
            layer.name,
            probe.join(", "),
            layer
                .features
                .first()
                .map(|f| f.attributes.keys().cloned().collect::<Vec<_>>().join(", "))
                .unwrap_or_else(|| "(the layer is empty)".into())
        ));
    };
    out.attribute = attribute.clone();

    // A layer is numeric when MOST of its stated values are numbers. Not "all":
    // a real table has a "N/A" in it, and one string must not turn a canopy
    // percentage into a name lookup.
    let stated: Vec<&inf_gis::Attr> = layer
        .features
        .iter()
        .filter_map(|f| f.attr(&attribute))
        .filter(|a| !matches!(a, inf_gis::Attr::Null))
        .collect();
    let numeric_count = stated.iter().filter(|a| a.as_number().is_some()).count();
    out.numeric = stated.len() > 0 && numeric_count * 2 > stated.len();

    // ── class id per feature ────────────────────────────────────────────────
    let ids: Vec<Option<u8>> = if out.numeric {
        // **`classify_to_ids` gets its caller.** Values in feature order, cut by
        // the chosen method, then shifted off the reserved 0.
        let values: Vec<f64> = layer
            .features
            .iter()
            .map(|f| {
                f.attr(&attribute)
                    .and_then(|a| a.as_number())
                    .unwrap_or(f64::NAN)
            })
            .collect();
        let classes = opts.classes.clamp(1, 64);
        let (raw, breaks) = inf_gis::classify_to_ids(&values, opts.method, classes, u8::MAX);
        out.breaks = breaks;
        raw.iter()
            .map(|c| {
                if *c == u8::MAX {
                    None
                } else {
                    Some(id_for_class(*c as usize, &opts.ids, classes))
                }
            })
            .collect()
    } else {
        // Named classes: match the biome set's own names, folded.
        let fold = |s: &str| -> String {
            s.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect()
        };
        let table: BTreeMap<String, u8> =
            biome_names.iter().map(|(id, n)| (fold(n), *id)).collect();
        let mut unmatched: BTreeMap<String, usize> = BTreeMap::new();
        let ids = layer
            .features
            .iter()
            .map(|f| {
                let text = f.attr(&attribute).and_then(|a| a.as_text())?;
                match table.get(&fold(text)) {
                    Some(id) => Some(*id),
                    None => {
                        *unmatched.entry(text.to_string()).or_default() += 1;
                        None
                    }
                }
            })
            .collect();
        if !unmatched.is_empty() {
            let names: Vec<String> = unmatched
                .iter()
                .take(5)
                .map(|(k, n)| format!("{k:?} x{n}"))
                .collect();
            out.advisories.push(format!(
                "{} class name(s) in {:?} match no biome in this level's biome set \
                 and were left unpainted: {}. Add them to the biome set, or import \
                 with a numeric class attribute instead.",
                unmatched.len(),
                attribute,
                names.join(", ")
            ));
        }
        ids
    };

    // ── paint ───────────────────────────────────────────────────────────────
    let mut fill = BiomeFill::new();
    let mut per_biome: BTreeMap<u8, usize> = BTreeMap::new();
    {
        let Some(entity) = doc.world().entity_of(terrain) else {
            return Err("the terrain entity disappeared mid-import".to_string());
        };
        let world = doc.world_mut().world_mut();
        let Some(mut t) = world.get_mut::<inf_ecs::components::Terrain>(entity) else {
            return Err("the target entity lost its Terrain component".to_string());
        };
        for (f, id) in layer.features.iter().zip(&ids) {
            let Some((exterior, holes)) = rings_of(&f.geometry) else {
                out.not_areas += 1;
                continue;
            };
            let Some(id) = id else {
                out.unclassified += 1;
                continue;
            };
            let n = fill.add_polygon(&mut t.data, *id, &exterior, &holes);
            if n == 0 {
                out.empty_polygons += 1;
            } else {
                *per_biome.entry(*id).or_default() += n;
            }
        }
    }
    out.per_biome = per_biome;
    if out.painted() == 0 {
        return Err(format!(
            "the layer {:?} painted no biome samples at all: {} feature(s) had no \
             usable class, {} were not areas, and {} polygons fell outside the \
             terrain's authored tiles. Painting never creates ground.",
            layer.name, out.unclassified, out.not_areas, out.empty_polygons
        ));
    }
    doc.edit_commit_biome_fill(terrain, "Import Land Cover", fill);
    out.advisories
        .extend(layer.advisories.iter().map(|a| a.to_string()));
    Ok(out)
}

/// The biome id for class `c`: the caller's own table, else `1..=classes`.
fn id_for_class(c: usize, ids: &[u8], classes: usize) -> u8 {
    if let Some(id) = ids.get(c) {
        return *id;
    }
    // Biome 0 is `UNASSIGNED_BIOME` and reserved, so classes start at 1.
    ((c + 1).min(classes.max(1)).min(255)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use inf_gis::feature::{Attr, GeoFeature, LayerKind};
    use inf_terrain::{TerrainData, TerrainTile};

    /// A 129-sample terrain at 1 m/sample covering `[0, 128]²`.
    fn doc_with_terrain() -> (SceneDoc, Uuid) {
        let mut doc = SceneDoc::new();
        let guid = doc.edit_create(crate::ipc::SpawnKind::Empty, "Ground", None);
        let e = doc.world().entity_of(guid).unwrap();
        let res = 129u32;
        let tile =
            TerrainTile::from_heights(res, DVec3::ZERO, vec![0.0; (res * res) as usize]).unwrap();
        let mut data = TerrainData::new(res, 1.0);
        data.insert_tile((0, 0), tile).ok();
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::Terrain {
                data,
                ..Default::default()
            });
        (doc, guid)
    }

    fn square(x0: f64, z0: f64, x1: f64, z1: f64) -> GeoGeometry {
        GeoGeometry::Polygon {
            exterior: vec![
                DVec3::new(x0, 0.0, z0),
                DVec3::new(x1, 0.0, z0),
                DVec3::new(x1, 0.0, z1),
                DVec3::new(x0, 0.0, z1),
            ],
            holes: Vec::new(),
        }
    }

    fn biome_at(doc: &SceneDoc, guid: Uuid, x: f64, z: f64) -> u8 {
        let e = doc.world().entity_of(guid).unwrap();
        let t = doc
            .world()
            .world()
            .get::<inf_ecs::components::Terrain>(e)
            .unwrap();
        t.data.biome_at(DVec2::new(x, z)).unwrap_or(0)
    }

    /// **The Jenks classifier finally has a caller, and the WORLD says which
    /// biome is where.**
    ///
    /// Three land-cover polygons with a numeric canopy attribute in two tight
    /// clusters: the classifier must cut them into two classes, and the terrain
    /// must come back with those two ids under those two polygons — and biome 0
    /// everywhere else, because painting never authors ground it was not given.
    #[test]
    fn a_numeric_land_cover_layer_paints_classified_biome_ids() {
        let (mut doc, terrain) = doc_with_terrain();
        let mut l = GeoLayer::new("Cover", LayerKind::Biomes, "EPSG:32610");
        for (rect, canopy) in [
            ((4.0, 4.0, 30.0, 30.0), 5.0),
            ((40.0, 4.0, 70.0, 30.0), 7.0),
            ((80.0, 4.0, 120.0, 30.0), 92.0),
        ] {
            let mut f = GeoFeature::new(square(rect.0, rect.1, rect.2, rect.3));
            f.attributes
                .insert("CANOPY_PCT".into(), Attr::Number(canopy));
            l.features.push(f);
        }

        let out = paint_biomes_from_layer(
            &mut doc,
            terrain,
            &l,
            &BiomeImportOptions {
                classes: 2,
                ..Default::default()
            },
            &[],
        )
        .expect("the import succeeds");

        assert!(out.numeric, "a percentage is a number, not a name");
        assert_eq!(out.attribute, "CANOPY_PCT");
        assert_eq!(
            out.breaks.len(),
            3,
            "2 classes = 3 fenceposts: {:?}",
            out.breaks
        );
        assert_eq!(out.per_biome.len(), 2, "{:?}", out.per_biome);
        assert_eq!(out.unclassified, 0);
        assert_eq!(out.empty_polygons, 0);

        // THE WORLD: the two sparse polygons are one biome, the dense one another.
        let a = biome_at(&doc, terrain, 10.0, 10.0);
        let b = biome_at(&doc, terrain, 50.0, 10.0);
        let c = biome_at(&doc, terrain, 100.0, 10.0);
        assert_eq!(a, b, "5% and 7% canopy are the same natural class");
        assert_ne!(
            a, c,
            "92% canopy must NOT land in the same class as 5% — that is what \
             natural breaks is for"
        );
        assert!(a >= 1 && c >= 1, "biome 0 is reserved: {a} {c}");
        // Unpainted ground stays unassigned.
        assert_eq!(biome_at(&doc, terrain, 35.0, 10.0), 0);
        assert_eq!(biome_at(&doc, terrain, 10.0, 60.0), 0);
        println!(
            "IB-5a: {} samples, biomes {:?}, breaks {:?}",
            out.painted(),
            out.per_biome,
            out.breaks
        );

        // ONE undo step takes the whole import back, to the byte.
        assert!(doc.undo());
        for (x, z) in [(10.0, 10.0), (50.0, 10.0), (100.0, 10.0)] {
            assert_eq!(biome_at(&doc, terrain, x, z), 0, "undo at ({x}, {z})");
        }
        assert!(doc.redo());
        assert_eq!(biome_at(&doc, terrain, 100.0, 10.0), c);
    }

    /// A layer whose classes are NAMES matches the level's own biome set, and a
    /// name that matches nothing is reported rather than assigned to biome 1.
    #[test]
    fn a_named_land_cover_layer_matches_the_biome_set_and_reports_what_it_cannot() {
        let (mut doc, terrain) = doc_with_terrain();
        let mut l = GeoLayer::new("Zoning", LayerKind::Biomes, "EPSG:32610");
        for (rect, name) in [
            ((4.0, 4.0, 30.0, 30.0), "Coniferous Forest"),
            ((40.0, 4.0, 70.0, 30.0), "wetland"),
            ((80.0, 4.0, 120.0, 30.0), "Lunar Regolith"),
        ] {
            let mut f = GeoFeature::new(square(rect.0, rect.1, rect.2, rect.3));
            f.attributes
                .insert("LANDCOVER".into(), Attr::Text(name.into()));
            l.features.push(f);
        }
        let set = [
            (3u8, "Coniferous forest".to_string()),
            (7u8, "Wetland".to_string()),
        ];

        let out =
            paint_biomes_from_layer(&mut doc, terrain, &l, &Default::default(), &set).unwrap();
        assert!(!out.numeric);
        assert_eq!(out.attribute, "LANDCOVER");
        assert!(out.breaks.is_empty(), "a named layer is not classified");
        assert_eq!(biome_at(&doc, terrain, 10.0, 10.0), 3);
        assert_eq!(
            biome_at(&doc, terrain, 50.0, 10.0),
            7,
            "case and spacing fold"
        );
        assert_eq!(
            biome_at(&doc, terrain, 100.0, 10.0),
            0,
            "an unmatched name must be left UNPAINTED, not assigned to biome 1"
        );
        assert_eq!(out.unclassified, 1);
        assert!(
            out.advisories.iter().any(|a| a.contains("Lunar Regolith")),
            "{:?}",
            out.advisories
        );
    }

    /// Every refusal is a value with a remedy in it.
    #[test]
    fn the_refusals_name_their_remedies() {
        let (mut doc, terrain) = doc_with_terrain();
        let empty = GeoLayer::new("Nothing", LayerKind::Biomes, "EPSG:32610");
        let e = paint_biomes_from_layer(&mut doc, terrain, &empty, &Default::default(), &[])
            .unwrap_err();
        assert!(e.contains("landcover"), "the probed names are listed: {e}");

        // A polygon entirely outside the terrain paints nothing and says so.
        let mut far = GeoLayer::new("Far", LayerKind::Biomes, "EPSG:32610");
        let mut f = GeoFeature::new(square(9000.0, 9000.0, 9100.0, 9100.0));
        f.attributes.insert("landcover".into(), Attr::Number(1.0));
        far.features.push(f);
        let e =
            paint_biomes_from_layer(&mut doc, terrain, &far, &Default::default(), &[]).unwrap_err();
        assert!(e.contains("never creates ground"), "{e}");

        // A non-terrain target is refused by name.
        let other = doc.edit_create(crate::ipc::SpawnKind::Cube, "Box", None);
        let e =
            paint_biomes_from_layer(&mut doc, other, &far, &Default::default(), &[]).unwrap_err();
        assert!(e.contains("not a terrain"), "{e}");
    }
}
