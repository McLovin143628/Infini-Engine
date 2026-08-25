//! The biome map: Jenks natural breaks where the data decides, design masks
//! where the author does.
//!
//! # Why natural breaks and not equal intervals
//!
//! Real elevation distributions are lumpy. Cutting an island's vegetated band
//! into three equal-width height classes puts nine tenths of it in one, because
//! nine tenths of the island is below a third of its peak. Fisher-Jenks
//! minimises within-class variance, so the fenceposts land in the *gaps* the
//! histogram actually has. Wave G ported the classifier for exactly this and it
//! had no caller outside its own tests; this is the caller.
//!
//! # What the classifier is NOT allowed to decide
//!
//! Three things are the author's, and each is a recipe number rather than a
//! break:
//!
//! * **the treeline** — a fact about a place, not about a histogram;
//! * **rock** — a slope, because a 45° face is bare whatever height it is at;
//! * **the beach** — a distance from the water line.
//!
//! And two biomes have no data behind them at all: **farmland** and
//! **urban-reserved** are pure design, stamped from the mask layer and from the
//! recipe's own sites. A classifier that invented a farm would be inventing a
//! farmer.

use glam::DVec2;

use crate::recipe::{BiomeSpec, IslandRecipe, SiteKind};
use crate::terrain::CoarseHeights;

/// The island's palette.
///
/// The discriminants are the ids written into `.inf_terrain` tiles and named by
/// the `.inf_biomes` set, so they are **frozen**: `0` stays
/// `inf_terrain::UNASSIGNED_BIOME` and nothing is renumbered. Appending is how a
/// palette grows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum IslandBiome {
    /// Sand and shingle inside the water line.
    Beach = 1,
    /// Low open grass.
    Plain = 2,
    /// Upland grass and scrub.
    Meadow = 3,
    /// Worked ground — design only.
    Farmland = 4,
    /// Closed canopy.
    Forest = 5,
    /// Rock, scree and everything above the treeline.
    Alpine = 6,
    /// Reserved for a settlement; nothing scatters here.
    Urban = 7,
}

impl IslandBiome {
    /// Every biome, in id order.
    pub const ALL: [IslandBiome; 7] = [
        IslandBiome::Beach,
        IslandBiome::Plain,
        IslandBiome::Meadow,
        IslandBiome::Farmland,
        IslandBiome::Forest,
        IslandBiome::Alpine,
        IslandBiome::Urban,
    ];

    /// The id written into a terrain tile.
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// The label the mask layer and the report use.
    pub const fn label(self) -> &'static str {
        match self {
            IslandBiome::Beach => "beach",
            IslandBiome::Plain => "plain",
            IslandBiome::Meadow => "meadow",
            IslandBiome::Farmland => "farmland",
            IslandBiome::Forest => "forest",
            IslandBiome::Alpine => "alpine",
            IslandBiome::Urban => "urban",
        }
    }

    /// Parse a label. `None` rather than a default, so a mask that misspells a
    /// biome is *reported* instead of quietly becoming beach.
    pub fn from_label(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.label() == s)
    }

    /// The id → biome inverse.
    pub fn from_id(id: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.id() == id)
    }

    /// A display colour for the `.inf_biomes` set, linear RGBA.
    pub const fn color(self) -> [f32; 4] {
        match self {
            IslandBiome::Beach => [0.82, 0.74, 0.52, 1.0],
            IslandBiome::Plain => [0.44, 0.62, 0.28, 1.0],
            IslandBiome::Meadow => [0.56, 0.68, 0.32, 1.0],
            IslandBiome::Farmland => [0.68, 0.60, 0.26, 1.0],
            IslandBiome::Forest => [0.16, 0.36, 0.20, 1.0],
            IslandBiome::Alpine => [0.55, 0.55, 0.58, 1.0],
            IslandBiome::Urban => [0.38, 0.38, 0.40, 1.0],
        }
    }

    /// Whether anything is scattered here at all.
    ///
    /// `Urban` answers `false` — the point of reserving it is that I8's
    /// settlement generator finds bare ground rather than a forest to clear.
    pub const fn scatters(self) -> bool {
        !matches!(self, IslandBiome::Urban)
    }
}

/// One design mask: a polygon and the biome it forces.
#[derive(Clone, Debug, PartialEq)]
pub struct BiomeMask {
    pub biome: IslandBiome,
    pub exterior: Vec<DVec2>,
    pub holes: Vec<Vec<DVec2>>,
}

/// What the classification decided.
#[derive(Clone, Debug, PartialEq)]
pub struct BiomeClassification {
    /// The height fenceposts Jenks chose for the vegetated band.
    pub breaks: Vec<f64>,
    /// How many coarse cells fell to each biome, in id order.
    pub cells: [u64; 8],
    /// Cells a **design mask** decided, whatever biome it named.
    ///
    /// It counted only `Farmland` — which is one of the two biomes the committed
    /// masks name, so the report and the build log under-stated what the design
    /// overrode by every meadow the author had painted, and there was no way to
    /// tell a meadow the classifier chose from one an author drew. Cells a *site*
    /// reservation took first are `reserved`, not this.
    pub masked: u64,
    /// Cells the recipe's own sites reserved.
    pub reserved: u64,
    /// The vegetated band the classifier saw, `(lo, hi)` metres.
    pub band_m: (f64, f64),
}

impl BiomeClassification {
    /// The fraction of **land** cells that fell to each biome, in id order.
    ///
    /// Land, not world — quoting a biome share against a map that is a third
    /// ocean makes every number a third too small and says nothing about the
    /// island.
    pub fn land_fractions(&self) -> [f64; 8] {
        let total: u64 = self.cells[1..].iter().sum();
        let mut out = [0.0; 8];
        if total == 0 {
            return out;
        }
        for (o, c) in out.iter_mut().zip(self.cells.iter()).skip(1) {
            *o = *c as f64 / total as f64;
        }
        out
    }

    /// Land cells classified.
    pub fn land_cells(&self) -> u64 {
        self.cells[1..].iter().sum()
    }
}

/// The classifier, as a value: everything it needs to answer one cell.
#[derive(Clone, Debug)]
pub struct Classifier {
    spec: BiomeSpec,
    sea_level_m: f64,
    breaks: Vec<f64>,
    /// What each natural-break class means, lowest first. Never empty, and the
    /// last entry answers for every class past its length — which is what lets a
    /// three-name ladder describe four classes by merging the top two.
    ladder: Vec<IslandBiome>,
    masks: Vec<BiomeMask>,
    /// `(centre, radius)` per urban reservation.
    reserved: Vec<(DVec2, f64)>,
}

impl Classifier {
    /// The biome at a cell, given its height and slope.
    ///
    /// The order is the priority, and it is deliberate: design beats data, and
    /// among data the *definite* facts (under water, on a cliff, above the
    /// treeline, on the shore) beat the statistical one.
    pub fn at(&self, p: DVec2, height_m: f64, slope_deg: f64) -> u8 {
        if height_m <= self.sea_level_m {
            return inf_terrain::UNASSIGNED_BIOME;
        }
        if self.reserved_at(p) {
            return IslandBiome::Urban.id();
        }
        if let Some(b) = self.mask_at(p) {
            return b.id();
        }
        if height_m - self.sea_level_m <= self.spec.beach_m {
            return IslandBiome::Beach.id();
        }
        if slope_deg >= self.spec.rock_deg || height_m >= self.spec.alpine_m {
            return IslandBiome::Alpine.id();
        }
        // The statistical half: which natural-break class this height falls in,
        // and what the RECIPE says that class means. Jenks finds the gaps; an
        // author says what grows in each band. See `BiomeSpec::class_biomes`.
        let k = inf_gis::class_of(height_m, &self.breaks).unwrap_or(usize::MAX);
        let last = self.ladder.len() - 1;
        self.ladder[k.min(last)].id()
    }

    /// The design mask covering `p`, if any.
    ///
    /// **The one place the mask test lives**, so the classifier's answer and the
    /// report's count cannot disagree about what a mask decided — which they did:
    /// the count was `id == Farmland`, and the committed masks name meadow too.
    pub fn mask_at(&self, p: DVec2) -> Option<IslandBiome> {
        self.masks
            .iter()
            .find(|m| point_in_ring(p, &m.exterior) && !m.holes.iter().any(|h| point_in_ring(p, h)))
            .map(|m| m.biome)
    }

    /// `true` where a settlement site reserves the ground. Outranks a mask.
    pub fn reserved_at(&self, p: DVec2) -> bool {
        self.reserved.iter().any(|(c, r)| (p - *c).length() <= *r)
    }

    /// The fenceposts, for the report.
    pub fn breaks(&self) -> &[f64] {
        &self.breaks
    }
}

/// Classify a coarse height grid into island biomes.
///
/// The Jenks pass runs over **land, below the treeline, off the rock and off the
/// beach** — the band the classifier is actually allowed to speak about. Feeding
/// it the whole island would spend two of three fenceposts separating sea floor
/// from mountain, which nothing downstream reads.
pub fn classify_biomes(
    recipe: &IslandRecipe,
    heights: &CoarseHeights,
    masks: &[BiomeMask],
) -> (Classifier, BiomeClassification) {
    let spec = recipe.biomes.clone();
    let sea = recipe.sea.level_m;
    let reserved: Vec<(DVec2, f64)> = recipe
        .sites
        .iter()
        .filter(|s| s.kind.reserves_urban())
        .map(|s| (DVec2::new(s.x, s.z), s.radius_m))
        .collect();

    // The band Jenks is allowed to see.
    let mut band: Vec<f64> = Vec::new();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for j in 0..heights.nz {
        for i in 0..heights.nx {
            let k = j * heights.nx + i;
            if !heights.known[k] {
                continue;
            }
            let h = f64::from(heights.h[k]);
            if h <= sea || h - sea <= spec.beach_m || h >= spec.alpine_m {
                continue;
            }
            if heights.slope_deg(i, j) >= spec.rock_deg {
                continue;
            }
            lo = lo.min(h);
            hi = hi.max(h);
            band.push(h);
        }
    }
    let breaks = inf_gis::classify_breaks(
        &band,
        inf_gis::ClassifyMethod::NaturalBreaks,
        spec.classes.max(1),
    );

    // A name the recipe's own door already refused if it were wrong; falling back
    // to forest here rather than panicking keeps a library call total.
    let ladder: Vec<IslandBiome> = {
        let v: Vec<IslandBiome> = spec
            .class_biomes
            .iter()
            .filter_map(|n| IslandBiome::from_label(n))
            .collect();
        if v.is_empty() {
            vec![IslandBiome::Forest]
        } else {
            v
        }
    };
    let c = Classifier {
        spec,
        sea_level_m: sea,
        breaks: breaks.clone(),
        ladder,
        masks: masks.to_vec(),
        reserved,
    };

    let mut cells = [0u64; 8];
    let mut masked = 0u64;
    let mut reserved_n = 0u64;
    for j in 0..heights.nz {
        for i in 0..heights.nx {
            let k = j * heights.nx + i;
            if !heights.known[k] {
                continue;
            }
            let p = heights.position(i, j);
            let id = c.at(p, f64::from(heights.h[k]), heights.slope_deg(i, j));
            cells[usize::from(id).min(7)] += 1;
            // Through the classifier's own doors, in its own priority order — a
            // second copy of the mask test here is how the count came to mean
            // "farmland" while the classifier meant "a mask decided this".
            if id == inf_terrain::UNASSIGNED_BIOME {
                // Under water: no design reached it, whatever covers it.
            } else if id == IslandBiome::Urban.id() && c.reserved_at(p) {
                reserved_n += 1;
            } else if c.mask_at(p).is_some() {
                masked += 1;
            }
        }
    }

    (
        c,
        BiomeClassification {
            breaks,
            cells,
            masked,
            reserved: reserved_n,
            band_m: if band.is_empty() {
                (0.0, 0.0)
            } else {
                (lo, hi)
            },
        },
    )
}

/// The `.inf_biomes` set this palette describes, **with its vegetation bound**.
///
/// # The binding lives on the set, and that is not this crate's choice
///
/// `inf_pcg::BiomeBinding::from_set` reads `BiomeDef::pcg_graph` — the set is the
/// one authority on "what grows in this biome", and both hosts resolve it
/// through the same door. So the set this crate writes carries the binding
/// rather than leaving a second place for it to be attached later.
///
/// `cover` is the `.inf_pcg` asset every **scattering** biome points at. Urban is
/// the one that does not, and that is the whole point of reserving it: I8's
/// settlement generator finds bare ground rather than a forest to clear.
pub fn biome_set(name: &str, cover: Option<inf_asset::AssetId>) -> inf_terrain::BiomeSet {
    let mut set = inf_terrain::BiomeSet::new(name);
    for b in IslandBiome::ALL {
        let mut def = inf_terrain::BiomeDef::new(b.id(), b.label());
        def.color = b.color();
        if b.scatters() {
            def.pcg_graph = cover.map(|a| a.0);
        }
        // The water hint the P20 hydrology tools read when an author drops a lake
        // on painted ground. Only the shore has one: everything else is dry, and
        // a hint on forest would put a lake wherever somebody clicked in a wood.
        if matches!(b, IslandBiome::Beach) {
            def.water_hint = Some(0.0);
        }
        set.biomes.push(def);
    }
    set
}

/// Half-open crossing rule — the same one `inf_terrain::BiomeFill` uses, so a
/// mask edge shared by two polygons belongs to exactly one of them.
fn point_in_ring(p: DVec2, ring: &[DVec2]) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let mut inside = false;
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        if (a.y > p.y) != (b.y > p.y) {
            let t = (p.y - a.y) / (b.y - a.y);
            if p.x < a.x + t * (b.x - a.x) {
                inside = !inside;
            }
        }
    }
    inside
}

/// The site reservations the recipe implies, for a caller that wants them
/// without running a classification.
pub fn urban_reservations(recipe: &IslandRecipe) -> Vec<(DVec2, f64)> {
    recipe
        .sites
        .iter()
        .filter(|s| s.kind.reserves_urban())
        .map(|s| (DVec2::new(s.x, s.z), s.radius_m))
        .collect()
}

/// Whether a site kind reserves ground — re-exported so a caller does not have
/// to reach into the recipe module for it.
pub fn reserves(kind: SiteKind) -> bool {
    kind.reserves_urban()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::{tests::tiny_recipe_text, Site};
    use std::path::Path;

    fn recipe() -> IslandRecipe {
        IslandRecipe::parse(&tiny_recipe_text(), Path::new("/tmp/i")).unwrap()
    }

    /// A ramp from the sea to the top of a mountain, with one steep face — so
    /// every branch of the classifier has ground that reaches it.
    fn ramp(nx: usize, nz: usize, pitch: f64) -> CoarseHeights {
        let mut h = vec![0.0f32; nx * nz];
        for j in 0..nz {
            for i in 0..nx {
                // Height climbs with i; a cliff in the middle rows.
                let mut v = i as f32 * 4.0 - 20.0;
                if (20..24).contains(&j) {
                    v += (i as f32) * 6.0;
                }
                h[j * nx + i] = v;
            }
        }
        CoarseHeights {
            min: DVec2::ZERO,
            pitch,
            nx,
            nz,
            h,
            known: vec![true; nx * nz],
        }
    }

    #[test]
    fn the_palette_is_frozen_and_round_trips_through_its_own_labels() {
        assert_eq!(IslandBiome::ALL.len(), 7);
        let ids: Vec<u8> = IslandBiome::ALL.iter().map(|b| b.id()).collect();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4, 5, 6, 7],
            "ids are frozen; 0 stays unassigned"
        );
        assert_ne!(inf_terrain::UNASSIGNED_BIOME, IslandBiome::Beach.id());
        for b in IslandBiome::ALL {
            assert_eq!(IslandBiome::from_label(b.label()), Some(b));
            assert_eq!(IslandBiome::from_id(b.id()), Some(b));
        }
        // A misspelling is None, never a default — a mask that says "forrest"
        // must be reported rather than silently becoming beach.
        assert_eq!(IslandBiome::from_label("forrest"), None);
        assert_eq!(IslandBiome::from_id(0), None);
        assert_eq!(IslandBiome::from_id(200), None);
        // Urban is the one that does not scatter.
        let non: Vec<&str> = IslandBiome::ALL
            .iter()
            .filter(|b| !b.scatters())
            .map(|b| b.label())
            .collect();
        assert_eq!(non, vec!["urban"]);
    }

    #[test]
    fn the_set_names_every_biome_and_validates() {
        let cover = inf_asset::AssetId(crate::build::cover_pcg_guid("Island"));
        let set = biome_set("Island", Some(cover));
        assert_eq!(set.biomes.len(), 7);
        set.validate()
            .expect("the island palette is a valid biome set");
        for b in IslandBiome::ALL {
            let d = set.get(b.id()).expect("every id is present");
            assert_eq!(d.name, b.label());
            assert_eq!(d.color, b.color());
            // **The vegetation binding is on the SET**, which is the one authority
            // `inf_pcg::BiomeBinding::from_set` reads — and urban is the one that
            // does not scatter, which is what reserving it is for.
            assert_eq!(
                d.pcg_graph.is_some(),
                b.scatters(),
                "{} binds a cover graph? {:?}",
                b.label(),
                d.pcg_graph
            );
            if b.scatters() {
                assert_eq!(d.pcg_graph, Some(cover.0));
            }
        }
        // The set's dependency closure names the cover asset exactly once, which
        // is what puts it in the level's sidecar.
        let deps = set.dependencies();
        assert_eq!(deps, vec![cover], "the set's dependencies are {deps:?}");
        // Only the shore carries a water hint; a hint on forest would put a lake
        // wherever an author clicked in a wood.
        let hinted: Vec<&str> = IslandBiome::ALL
            .iter()
            .filter(|b| set.get(b.id()).unwrap().water_hint.is_some())
            .map(|b| b.label())
            .collect();
        assert_eq!(hinted, vec!["beach"]);
        // With no cover asset the set is still valid and binds nothing.
        let bare = biome_set("Island", None);
        assert!(bare.biomes.iter().all(|d| d.pcg_graph.is_none()));
        assert!(bare.dependencies().is_empty());
    }

    /// The priority order is the claim, so each level of it is measured against a
    /// cell that would answer differently one level down.
    #[test]
    fn design_beats_data_and_definite_beats_statistical() {
        let mut r = recipe();
        r.biomes.alpine_m = 700.0;
        r.biomes.rock_deg = 38.0;
        r.biomes.beach_m = 25.0;
        r.sites.push(Site {
            name: "Town".into(),
            kind: SiteKind::Town,
            x: 40.0,
            z: 40.0,
            radius_m: 30.0,
        });
        let masks = vec![BiomeMask {
            biome: IslandBiome::Farmland,
            exterior: vec![
                DVec2::new(100.0, 0.0),
                DVec2::new(160.0, 0.0),
                DVec2::new(160.0, 60.0),
                DVec2::new(100.0, 60.0),
            ],
            holes: vec![],
        }];
        let h = ramp(64, 64, 8.0);
        let (c, rep) = classify_biomes(&r, &h, &masks);

        // Under water is unassigned, whatever any mask says.
        assert_eq!(
            c.at(DVec2::new(0.0, 0.0), -5.0, 0.0),
            inf_terrain::UNASSIGNED_BIOME
        );
        // The site reservation outranks the mask AND the data.
        assert_eq!(
            c.at(DVec2::new(40.0, 40.0), 300.0, 0.0),
            IslandBiome::Urban.id()
        );
        // The mask outranks the data.
        assert_eq!(
            c.at(DVec2::new(120.0, 30.0), 300.0, 0.0),
            IslandBiome::Farmland.id()
        );
        // …and outranks the beach rule, which is the "design beats data" claim
        // at the one place the two genuinely disagree.
        assert_eq!(
            c.at(DVec2::new(120.0, 30.0), 4.0, 0.0),
            IslandBiome::Farmland.id()
        );
        // Off the mask, the definite rules bite in order.
        assert_eq!(
            c.at(DVec2::new(500.0, 500.0), 4.0, 0.0),
            IslandBiome::Beach.id()
        );
        assert_eq!(
            c.at(DVec2::new(500.0, 500.0), 300.0, 45.0),
            IslandBiome::Alpine.id()
        );
        assert_eq!(
            c.at(DVec2::new(500.0, 500.0), 900.0, 0.0),
            IslandBiome::Alpine.id()
        );
        // And below all of them, the statistical one.
        let mid = c.at(DVec2::new(500.0, 500.0), 300.0, 5.0);
        assert!(
            [
                IslandBiome::Plain.id(),
                IslandBiome::Meadow.id(),
                IslandBiome::Forest.id()
            ]
            .contains(&mid),
            "the vegetated band answered {mid}"
        );

        assert!(rep.reserved > 0, "the town reserved nothing");
        assert!(rep.masked > 0, "the farmland mask stamped nothing");
        assert!(rep.land_cells() > 0);
        let f = rep.land_fractions();
        let sum: f64 = f[1..].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "the land fractions sum to {sum}");
        println!(
            "BIOMES: breaks {:?}, band {:.1}..{:.1} m, land {} cells",
            rep.breaks,
            rep.band_m.0,
            rep.band_m.1,
            rep.land_cells()
        );
        for b in IslandBiome::ALL {
            println!(
                "  {:>9}: {:>7} cells  {:>5.1} %",
                b.label(),
                rep.cells[b.id() as usize],
                f[b.id() as usize] * 100.0
            );
        }
    }

    /// **Natural breaks, not equal intervals** — the claim Wave G ported the
    /// classifier for, measured on a distribution that tells them apart.
    #[test]
    fn natural_breaks_beat_equal_intervals_on_a_lumpy_island() {
        // Nine tenths of the ground low, one tenth high — an island's own shape.
        let mut v: Vec<f64> = (0..900).map(|k| 40.0 + f64::from(k % 30) * 0.5).collect();
        v.extend((0..100).map(|k| 600.0 + f64::from(k % 30) * 0.5));
        let classes = 3;
        let jenks = inf_gis::classify_breaks(&v, inf_gis::ClassifyMethod::NaturalBreaks, classes);
        let equal = inf_gis::classify_breaks(&v, inf_gis::ClassifyMethod::EqualInterval, classes);
        let counts = |b: &[f64]| -> Vec<usize> {
            let mut c = vec![0usize; classes];
            for x in &v {
                if let Some(k) = inf_gis::class_of(*x, b) {
                    c[k.min(classes - 1)] += 1;
                }
            }
            c
        };
        let (cj, ce) = (counts(&jenks), counts(&equal));
        println!("CLASSING: jenks {cj:?} breaks {jenks:?}");
        println!("CLASSING: equal {ce:?} breaks {equal:?}");
        // **The claim, stated as a count rather than a share**: equal intervals
        // spend two of three classes on the gap between the lumps and leave one
        // EMPTY; natural breaks spend all three on ground that exists.
        assert!(
            ce.contains(&0),
            "equal intervals left no class empty on a bimodal set ({ce:?}) — the \
             fixture is not lumpy and this test measures nothing"
        );
        assert!(
            cj.iter().all(|n| *n > 0),
            "natural breaks left a class empty ({cj:?})"
        );
        // …and the biggest class is smaller under Jenks, which is the "nine
        // tenths of the map in one biome" complaint measured.
        let (bj, be) = (
            *cj.iter().max().unwrap() as f64 / v.len() as f64,
            *ce.iter().max().unwrap() as f64 / v.len() as f64,
        );
        println!("CLASSING: biggest class jenks {bj:.3}, equal {be:.3}");
        assert!(
            bj < be * 0.75,
            "Jenks's biggest class is {bj} against equal intervals' {be}"
        );
    }

    #[test]
    fn a_mask_hole_lets_the_data_back_through() {
        let r = recipe();
        let masks = vec![BiomeMask {
            biome: IslandBiome::Farmland,
            exterior: vec![
                DVec2::new(0.0, 0.0),
                DVec2::new(100.0, 0.0),
                DVec2::new(100.0, 100.0),
                DVec2::new(0.0, 100.0),
            ],
            holes: vec![vec![
                DVec2::new(40.0, 40.0),
                DVec2::new(60.0, 40.0),
                DVec2::new(60.0, 60.0),
                DVec2::new(40.0, 60.0),
            ]],
        }];
        let h = ramp(32, 32, 8.0);
        let (c, _) = classify_biomes(&r, &h, &masks);
        assert_eq!(
            c.at(DVec2::new(20.0, 20.0), 300.0, 0.0),
            IslandBiome::Farmland.id()
        );
        assert_ne!(
            c.at(DVec2::new(50.0, 50.0), 300.0, 0.0),
            IslandBiome::Farmland.id(),
            "the hole must let the classifier answer"
        );
        assert!(!c.breaks().is_empty());
        assert!(urban_reservations(&r).is_empty());
        assert!(reserves(SiteKind::City));
    }
}
