//! The island in numbers — what a build prints and what a ledger quotes.
//!
//! Every field here is a **measurement of the thing that was built**, not a
//! restatement of the recipe. That distinction is the whole reason the type
//! exists: a report that echoed its inputs would agree with itself perfectly
//! while the build did something else.

use crate::biome::IslandBiome;
use crate::hydro::StreamNetwork;
use crate::roads::RoadReport;
use crate::source::TilePlan;
use crate::terrain::SampleStats;
use crate::{Advisory, BuildStep};

/// How far a committed layer has drifted from what re-deriving it would produce.
///
/// # Why this is an advisory and not a failure
///
/// The sample step goes through the two projection modules Wave G's portability
/// gate exempts by name, so `tan`, `ln` and `atan` decide the heights, and those
/// are not bit-identical across platforms. A threshold comparison on a height —
/// which is what "is this cell a channel" is — can therefore land differently on
/// a different libm. Failing a build for that would make the island
/// un-rebuildable anywhere but the machine it was authored on; saying nothing
/// would make a *real* change to the derivation invisible. So it is measured and
/// printed, with the numbers, every time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayerDrift {
    /// Features in the committed layer.
    pub committed: usize,
    /// Features a fresh derivation produced.
    pub derived: usize,
    /// Total length or area the committed layer carries.
    pub committed_measure: f64,
    /// The same for the derivation.
    pub derived_measure: f64,
}

impl LayerDrift {
    /// The relative difference in the measure, or 0 when both are zero.
    pub fn relative(&self) -> f64 {
        let d = self.committed_measure.max(self.derived_measure);
        if d <= 0.0 {
            return 0.0;
        }
        (self.committed_measure - self.derived_measure).abs() / d
    }

    /// `true` when the two agree to within a stated fraction.
    pub fn agrees_within(&self, frac: f64) -> bool {
        self.committed == self.derived && self.relative() <= frac
    }
}

/// The island, measured.
#[derive(Clone, Debug, Default)]
pub struct IslandReport {
    pub name: String,
    /// The world square's edge, metres.
    pub extent_m: f64,
    /// The world square's area, square kilometres.
    pub map_km2: f64,
    /// Land above sea level, square kilometres.
    pub land_km2: f64,
    /// The designed shore's length, kilometres.
    pub coastline_km: f64,
    /// The highest point ON LAND, metres.
    pub peak_m: f64,
    /// The lowest carved point, metres.
    pub floor_m: f64,
    /// Level-0 tiles, and the whole catalog including the pyramid.
    pub tiles_level0: u64,
    pub tiles_total: usize,
    pub lod_levels: u32,
    /// The `.inf_terrain` payload's size in bytes.
    pub terrain_bytes: usize,
    /// The source, priced.
    pub source_tiles: usize,
    pub source_m_per_px: f64,
    pub grid_m_per_sample: f64,
    pub upsample_ratio: f64,
    /// Where the source had nothing.
    pub nodata_samples: u64,
    /// Hydrology.
    pub streams: usize,
    pub stream_km: f64,
    pub lakes: usize,
    pub lake_km2: f64,
    pub waterfalls: usize,
    pub biggest_waterfall_m: f64,
    pub max_catchment_km2: f64,
    /// Biomes, as a share of land, in id order.
    pub biome_share: [f64; 8],
    pub biome_breaks: Vec<f64>,
    /// Roads.
    pub roads: RoadReport,
    /// Which steps ran.
    pub steps: Vec<BuildStep>,
    /// Named, non-fatal findings.
    pub advisories: Vec<Advisory>,
    /// How the committed derived layers compare with a fresh derivation.
    pub stream_drift: LayerDrift,
    pub lake_drift: LayerDrift,
}

impl IslandReport {
    /// Fold in what the sample walk measured.
    pub fn with_samples(mut self, s: &SampleStats) -> Self {
        self.land_km2 = s.land_area_m2 / 1.0e6;
        self.peak_m = if s.peak_land_m.is_finite() {
            s.peak_land_m
        } else {
            0.0
        };
        self.floor_m = if s.lo_m.is_finite() { s.lo_m } else { 0.0 };
        self.nodata_samples = s.nodata;
        self
    }

    /// Fold in the source plan.
    pub fn with_plan(mut self, p: &TilePlan) -> Self {
        self.source_tiles = p.len();
        self.source_m_per_px = p.ground_m_per_px;
        self.grid_m_per_sample = p.grid_m_per_sample;
        self.upsample_ratio = p.upsample_ratio();
        self
    }

    /// Fold in the hydrology.
    pub fn with_hydrology(mut self, n: &StreamNetwork) -> Self {
        self.streams = n.streams.len();
        self.stream_km = n.total_length_m() / 1000.0;
        self.lakes = n.lakes.len();
        self.lake_km2 = n.total_lake_area_m2() / 1.0e6;
        self.waterfalls = n.waterfalls.len();
        self.biggest_waterfall_m = n.waterfalls.iter().map(|w| w.drop_m).fold(0.0f64, f64::max);
        self.max_catchment_km2 = n.max_catchment_m2 / 1.0e6;
        self
    }

    /// The multi-line summary a build prints and a ledger quotes.
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "=== {} ===", self.name);
        let _ = writeln!(
            s,
            "  map        {:.0} x {:.0} m = {:.2} km2   land {:.2} km2 ({:.1} %)",
            self.extent_m,
            self.extent_m,
            self.map_km2,
            self.land_km2,
            if self.map_km2 > 0.0 {
                100.0 * self.land_km2 / self.map_km2
            } else {
                0.0
            }
        );
        let _ = writeln!(
            s,
            "  relief     peak {:.1} m on land, sea floor {:.1} m, coastline {:.2} km",
            self.peak_m, self.floor_m, self.coastline_km
        );
        let _ = writeln!(
            s,
            "  terrain    {} level-0 tiles, {} in the catalog, {} LOD levels, {:.1} MB",
            self.tiles_level0,
            self.tiles_total,
            self.lod_levels,
            self.terrain_bytes as f64 / 1.0e6
        );
        let _ = writeln!(
            s,
            "  source     {} tiles at {:.2} m/px on a {:.2} m grid ({:.2}x upsample), \
             {} nodata samples",
            self.source_tiles,
            self.source_m_per_px,
            self.grid_m_per_sample,
            self.upsample_ratio,
            self.nodata_samples
        );
        let _ = writeln!(
            s,
            "  water      {} streams / {:.2} km, {} lakes / {:.4} km2, {} waterfalls \
             (biggest {:.1} m), max catchment {:.2} km2",
            self.streams,
            self.stream_km,
            self.lakes,
            self.lake_km2,
            self.waterfalls,
            self.biggest_waterfall_m,
            self.max_catchment_km2
        );
        let _ = write!(s, "  biomes     ");
        for b in IslandBiome::ALL {
            let _ = write!(
                s,
                "{} {:.1}%  ",
                b.label(),
                self.biome_share[b.id() as usize] * 100.0
            );
        }
        let _ = writeln!(s);
        let _ = writeln!(s, "             breaks {:?}", self.biome_breaks);
        let _ = writeln!(
            s,
            "  roads      {:.2} km over {} segments and {} junctions; worst grade \
             {:.3} against a {:.3} ceiling, {} stretches over",
            self.roads.total_km,
            self.roads.segments,
            self.roads.junctions,
            self.roads.audit.worst,
            self.roads.audit.ceiling,
            self.roads.audit.over.len()
        );
        for (k, km) in &self.roads.km_by_class {
            let _ = writeln!(s, "             {k:>12}: {km:.2} km");
        }
        let _ = writeln!(
            s,
            "  drift      streams {} vs {} ({:.2} % of length), lakes {} vs {} \
             ({:.2} % of area)",
            self.stream_drift.committed,
            self.stream_drift.derived,
            self.stream_drift.relative() * 100.0,
            self.lake_drift.committed,
            self.lake_drift.derived,
            self.lake_drift.relative() * 100.0
        );
        let _ = writeln!(
            s,
            "  steps      {}",
            self.steps
                .iter()
                .map(|x| x.label())
                .collect::<Vec<_>>()
                .join(" -> ")
        );
        for a in &self.advisories {
            let _ = writeln!(s, "  ADVISORY   {a}");
        }
        s
    }

    /// `true` when nothing in the build needs an author's attention.
    ///
    /// A blocking finding exits the CLI non-zero — the C4-40 law, met at the
    /// island's own door: an advisory printed by a pipeline whose status nobody
    /// reads is an advisory nobody reads.
    pub fn is_clean(&self) -> bool {
        self.advisories.is_empty() && self.roads.audit.is_clean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_is_relative_and_survives_two_empties() {
        let d = LayerDrift {
            committed: 12,
            derived: 12,
            committed_measure: 100.0,
            derived_measure: 99.0,
        };
        assert!((d.relative() - 0.01).abs() < 1e-12);
        assert!(d.agrees_within(0.02));
        assert!(!d.agrees_within(0.005));
        // A count difference is a disagreement whatever the measure says.
        let e = LayerDrift { derived: 13, ..d };
        assert!(!e.agrees_within(0.99));
        assert_eq!(LayerDrift::default().relative(), 0.0);
        assert!(LayerDrift::default().agrees_within(0.0));
    }

    /// The summary prints every number a ledger quotes, and it prints them from
    /// the MEASUREMENTS rather than from the recipe.
    #[test]
    fn the_summary_carries_every_number_the_ledger_needs() {
        let mut r = IslandReport {
            name: "Test Island".into(),
            extent_m: 7_168.0,
            map_km2: 51.38,
            coastline_km: 34.5,
            tiles_level0: 784,
            tiles_total: 1_049,
            lod_levels: 5,
            terrain_bytes: 280_000_000,
            biome_breaks: vec![10.0, 220.0, 480.0],
            ..Default::default()
        };
        r = r.with_samples(&SampleStats {
            land_area_m2: 31.2e6,
            peak_land_m: 948.7,
            lo_m: -60.0,
            nodata: 17,
            ..Default::default()
        });
        r.roads.total_km = 62.4;
        r.roads.segments = 9;
        r.roads.junctions = 4;
        r.roads.audit.ceiling = 0.08;
        r.roads.audit.worst = 0.079;
        r.roads.km_by_class = vec![("arterial".into(), 40.0), ("highway".into(), 22.4)];
        r.steps = BuildStep::ALL.to_vec();
        r.biome_share[IslandBiome::Forest.id() as usize] = 0.52;

        let s = r.summary();
        println!("{s}");
        for needle in [
            "Test Island",
            "51.38 km2",
            "31.20 km2",
            "948.7 m",
            "-60.0 m",
            "34.50 km",
            "784 level-0",
            "1049 in the catalog",
            "5 LOD",
            "280.0 MB",
            "17 nodata",
            "forest 52.0%",
            "62.40 km",
            "0.079",
            "0.080 ceiling",
            "arterial",
            "plan -> fetch -> sample -> carve -> hydrology -> biomes -> roads -> pyramid -> write",
        ] {
            assert!(
                s.contains(needle),
                "the summary is missing {needle:?}:\n{s}"
            );
        }
        assert!(r.is_clean());

        // …and an advisory both prints and makes the build non-clean.
        r.advisories
            .push(Advisory::new("x.y", "something to look at"));
        assert!(!r.is_clean());
        assert!(r.summary().contains("ADVISORY   [x.y]"));
    }

    #[test]
    fn a_report_folds_the_things_that_measured_them() {
        let plan = TilePlan {
            zoom: 15,
            tiles: vec![],
            lon: (0.0, 1.0),
            lat: (0.0, 1.0),
            ground_m_per_px: 3.11,
            grid_m_per_sample: 1.0,
        };
        let r = IslandReport::default().with_plan(&plan);
        assert_eq!(r.source_m_per_px, 3.11);
        assert!((r.upsample_ratio - 3.11).abs() < 1e-12);

        let net = StreamNetwork {
            streams: vec![],
            lakes: vec![],
            waterfalls: vec![],
            max_catchment_m2: 4.0e6,
            channel_cells: 0,
        };
        let r = r.with_hydrology(&net);
        assert_eq!(r.max_catchment_km2, 4.0);
        assert_eq!(r.streams, 0);
        assert_eq!(r.biggest_waterfall_m, 0.0);
    }
}
