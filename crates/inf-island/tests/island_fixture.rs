//! The CI-scale island — **every recipe step, against committed bytes**.
//!
//! # What this gate is for
//!
//! The real island is 51 km² of fetched elevation and 342 MB of terrain. CI
//! cannot build it, must not fetch it, and would learn nothing from a smaller
//! copy of the *report*. What it can do — and what this does — is run the whole
//! recipe over a 2.36 km² corner whose two source tiles are committed beside it,
//! and assert the things that would be true of any island the recipe builds.
//!
//! # The three claims that make it a gate rather than a smoke test
//!
//! * **It never reaches a network.** The plan's tile list is compared against the
//!   directory listing, both ways, so a change that needed one more tile goes red
//!   here instead of reaching for `curl` on a runner.
//! * **Every step ran.** [`BuildStep::ALL`] is enumerated and each is matched by a
//!   **count** in the log, never a substring — "a `contains` needle that is a
//!   prefix of a declaration can never fail" (the I1 audit).
//! * **The world, not the report.** The island is asserted by asking the built
//!   terrain where the ground is, at positions the fixture's own design puts on
//!   land, in the sea and on the shore.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use glam::DVec2;
use inf_island::{BuildOptions, BuildStep, IslandBiome, IslandRecipe};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture")
}

fn recipe() -> IslandRecipe {
    IslandRecipe::load(&fixture_dir().join("island.toml")).expect("the fixture recipe loads")
}

/// Build it the way a consumer does: the committed design is read, not rewritten.
fn build() -> inf_island::IslandBuild {
    inf_island::build_island(&recipe(), &BuildOptions::default()).expect("the fixture builds")
}

/// Every `.png` under a directory, as repo-relative-ish strings.
fn png_files(root: &Path) -> BTreeSet<String> {
    fn walk(p: &Path, root: &Path, out: &mut BTreeSet<String>) {
        let Ok(rd) = std::fs::read_dir(p) else { return };
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                walk(&path, root, out);
            } else if path.extension().is_some_and(|x| x == "png") {
                out.insert(
                    path.strip_prefix(root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(root, root, &mut out);
    out
}

/// **CI NEVER FETCHES.** The plan and the committed cache are compared both ways.
///
/// Un-fix mutation: delete one committed tile and `missing_in` is non-empty;
/// commit one the plan does not name and the listing comparison fails.
#[test]
fn the_fixture_builds_from_committed_bytes_and_never_reaches_a_network() {
    let r = recipe();
    let plan = inf_island::plan_tiles(&r).expect("the fixture plans");
    let cache = r.cache_dir();
    assert!(
        cache.starts_with(fixture_dir()),
        "the fixture's cache must be committed beside it, not outside the tree: {}",
        cache.display()
    );

    let missing = plan.missing_in(&cache);
    assert!(
        missing.is_empty(),
        "{} of {} source tiles are missing from the committed cache — CI cannot \
         fetch them: {missing:?}",
        missing.len(),
        plan.len()
    );

    // …and nothing is committed that the plan does not name, so the fixture
    // cannot quietly grow a megabyte of unused tiles.
    let want: BTreeSet<String> = plan
        .tiles
        .iter()
        .map(|t| format!("terrarium/{}/{}/{}.png", t.z, t.x, t.y))
        .collect();
    let got = png_files(&cache);
    assert_eq!(
        got, want,
        "the committed cache and the plan must be the same set"
    );

    let bytes: u64 = got
        .iter()
        .map(|f| {
            std::fs::metadata(cache.join(f))
                .map(|m| m.len())
                .unwrap_or(0)
        })
        .sum();
    println!(
        "FIXTURE SOURCE: {} tiles at z{}, {} bytes committed, {:.2} m/px against \
         a {:.2} m grid ({:.2}x upsample)",
        plan.len(),
        plan.zoom,
        bytes,
        plan.ground_m_per_px,
        plan.grid_m_per_sample,
        plan.upsample_ratio()
    );
    assert!(
        bytes < 512 * 1024,
        "{bytes} bytes of source is more than a fixture should cost"
    );
    assert!(
        plan.len() >= 2,
        "one tile would not exercise the mosaic's seam"
    );
}

/// **EVERY STEP RAN**, counted per step.
#[test]
fn the_build_covers_every_recipe_step_exactly_once() {
    let b = build();
    for step in BuildStep::ALL {
        assert_eq!(
            b.step_count(step),
            1,
            "step {step} ran {} times; the log is {:?}",
            b.step_count(step),
            b.steps()
        );
    }
    assert_eq!(
        b.steps(),
        BuildStep::ALL.to_vec(),
        "the steps must run in the frozen order"
    );
    assert_eq!(b.log.len(), BuildStep::ALL.len());
    // The fetch step is the one that would need a network, and it reports being
    // satisfied from the cache rather than being skipped.
    let fetch = b
        .log
        .iter()
        .find(|l| l.step == BuildStep::Fetch)
        .expect("the fetch step is logged");
    assert!(
        fetch.note.contains("nothing was fetched"),
        "the fetch step said {:?}",
        fetch.note
    );
    assert!(BuildStep::Fetch.needs_network());
    assert_eq!(
        BuildStep::ALL.iter().filter(|s| s.needs_network()).count(),
        1,
        "exactly one step needs a network"
    );
    for l in &b.log {
        println!("[{:>9}] {}", l.step.label(), l.note);
    }
}

/// **THE WORLD, NOT THE REPORT.** The built terrain is asked where the ground is.
#[test]
fn the_carve_makes_an_island_out_of_a_piece_of_a_mountain() {
    let b = build();
    let sea = b.recipe.sea.level_m;
    let t = &b.terrain;

    // Dry where the design says land: at both sites and at the world centre.
    for s in &b.recipe.sites {
        let h = t
            .height_at(DVec2::new(s.x, s.z))
            .unwrap_or_else(|| panic!("{} is off the terrain", s.name));
        assert!(
            h > sea,
            "{} is at {h} m, below the {sea} m waterline",
            s.name
        );
    }

    // Wet outside the coastline, on every side.
    let half = b.recipe.grid.half_extent_m();
    let mut wet = 0;
    for (x, z) in [
        (half - 20.0, 0.0),
        (-(half - 20.0), 0.0),
        (0.0, half - 20.0),
        (0.0, -(half - 20.0)),
    ] {
        let p = DVec2::new(x, z);
        assert!(
            !b.coast.is_land(p),
            "({x}, {z}) should be outside the designed shore"
        );
        let h = t.height_at(p).expect("inside the terrain");
        assert!(h < sea, "({x}, {z}) is at {h} m and should be sea");
        wet += 1;
    }
    assert_eq!(wet, 4, "all four edges are sea");

    // **The sea floor reaches the recipe's own shelf depth and no further**, and
    // it is a different quantity from the island's lowest point: land inside the
    // coastline whose source elevation is under the waterline stays there (a real
    // inlet, or a shore drawn across a valley), so the overall minimum is not the
    // sea's. Both are reported; this asserts the one the recipe controls.
    let want = sea - b.recipe.sea.shelf_depth_m;
    assert!(
        (b.report.sea_floor_m - want).abs() < 1.0,
        "the sea floor is {} m; the recipe says {want}",
        b.report.sea_floor_m
    );
    assert!(
        b.report.floor_m <= b.report.sea_floor_m,
        "the lowest point cannot be above the sea floor"
    );
    println!(
        "RELIEF: peak {:.1} m, sea floor {:.1} m, lowest {:.1} m, {} submerged \
         land samples of {} on land",
        b.report.peak_m,
        b.report.sea_floor_m,
        b.report.floor_m,
        b.report.submerged_land,
        b.biomes.land_cells()
    );
    // …and a point past the shelf width really is on the flat floor, measured in
    // the world rather than taken from the summary.
    let corner = DVec2::new(half - 8.0, half - 8.0);
    assert!(!b.coast.is_land(corner));
    let ch = t.height_at(corner).expect("inside the terrain");
    assert!(
        (ch - want).abs() < 1.0,
        "the far corner is at {ch} m, not the {want} m shelf floor"
    );

    // The shore really is at the waterline: walk the coastline's own vertices.
    let mut worst = 0.0f64;
    let mut probes = 0;
    for ring in b.coast.rings() {
        for v in ring {
            if let Some(h) = t.height_at(*v) {
                worst = worst.max((h - sea).abs());
                probes += 1;
            }
        }
    }
    println!(
        "SHORE: {probes} coastline vertices, worst {worst:.3} m from the {sea} m \
         waterline"
    );
    assert!(probes >= 8);
    assert!(
        worst < 2.5,
        "a coastline vertex is {worst} m from the waterline"
    );

    // Land is a real fraction of the map, and it is neither everything nor
    // nothing — the two ways a carve can look like it worked and not have.
    let frac = b.report.land_km2 / b.report.map_km2;
    println!(
        "ISLAND: map {:.3} km2, land {:.3} km2 ({:.1} %), peak {:.1} m, shore {:.2} km",
        b.report.map_km2,
        b.report.land_km2,
        frac * 100.0,
        b.report.peak_m,
        b.report.coastline_km
    );
    assert!(
        (0.35..0.9).contains(&frac),
        "{:.1} % of the map is land",
        frac * 100.0
    );
    assert!(b.report.peak_m > sea + 50.0, "the island is flat");
    assert!(b.report.coastline_km > 3.0);
}

/// The water layers are read back through the import door and carry what the
/// derivation found.
#[test]
fn the_committed_water_layers_carry_the_derivation() {
    let b = build();
    let n = &b.network;
    println!(
        "WATER: {} reaches / {:.3} km, {} lakes / {:.5} km2, {} waterfalls \
         (biggest {:.2} m), max catchment {:.4} km2",
        n.streams.len(),
        n.total_length_m() / 1000.0,
        n.lakes.len(),
        n.total_lake_area_m2() / 1.0e6,
        n.waterfalls.len(),
        b.report.biggest_waterfall_m,
        b.report.max_catchment_km2
    );
    assert!(!n.streams.is_empty(), "no streams");
    assert!(!n.lakes.is_empty(), "no lakes");
    assert!(
        !n.waterfalls.is_empty(),
        "no waterfall sites — a network read back from its committed layer with \
         none in it reads as 'this island has none', which is the defect the \
         first full build had"
    );
    for w in &n.waterfalls {
        assert!(w.drop_m > 0.0 && w.top.y > w.bottom.y);
        assert!(w.grade >= b.recipe.hydro.waterfall_grade);
        assert!(
            n.streams.get(w.stream).is_some(),
            "a waterfall names a real reach"
        );
    }
    for s in &n.streams {
        assert!(s.points.len() >= 2);
        assert!(s.width_m() > 0.0 && s.depth_m() > 0.0);
        assert!(s.catchment_m2 >= b.recipe.hydro.stream_catchment_m2 * 0.5);
    }
    for l in &n.lakes {
        assert!(l.area_m2 >= b.recipe.hydro.lake_area_m2);
        assert!(l.level_m > b.recipe.sea.level_m);
        assert_eq!(l.outline.len(), 4);
    }

    // **THE BED IS CUT.** A river laid on unmodified ground is what P20 calls a
    // basin, so the channel carve is asserted where it is: under a reach.
    let mid = &n.streams[0].points[n.streams[0].points.len() / 2];
    let on = b
        .terrain
        .height_at(DVec2::new(mid.x, mid.z))
        .expect("on the terrain");
    let off = b
        .terrain
        .height_at(DVec2::new(mid.x + 30.0, mid.z + 30.0))
        .expect("beside it");
    println!("BED: {on:.2} m on the centreline, {off:.2} m thirty metres away");
    assert!(
        on < mid.y + 0.05,
        "the bed at {on} m is not below the {} m the derivation found",
        mid.y
    );
    // …**and the channel is a channel rather than the whole island going down.**
    // `off` was measured and then discarded (`let _ = off;`), which left the arm
    // satisfied by a carve that lowered every sample on the map by 1.25 m. Thirty
    // metres away is many multiples of the widest reach's own channel, so it must
    // still be above the bed.
    assert!(
        off > on,
        "the ground {off} m thirty metres from the centreline is not above the \
         {on} m bed — the carve is not local to the reach"
    );

    // The committed layer files are small enough to be design documents.
    for f in [
        "streams.geojson",
        "lakes.geojson",
        "roads.geojson",
        "coast.geojson",
    ] {
        let p = fixture_dir().join("layers").join(f);
        let n = std::fs::metadata(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        println!("LAYER {f}: {} bytes", n.len());
        assert!(
            n.len() > 0 && n.len() < 256 * 1024,
            "{f} is {} bytes",
            n.len()
        );
    }
}

/// The road network holds its own ceiling **after** the corridor is levelled in.
#[test]
fn the_road_network_holds_the_grade_ceiling_it_was_designed_to() {
    let b = build();
    let a = &b.report.roads.audit;
    println!(
        "ROADS: {:.3} km over {} segments, {} junctions; {} of {} stretches over \
         the {:.3} ceiling ({:.2} %), worst {:.4} at ({:.0}, {:.0})",
        b.report.roads.total_km,
        b.report.roads.segments,
        b.report.roads.junctions,
        a.over.len(),
        a.samples,
        a.ceiling,
        a.over_fraction() * 100.0,
        a.worst,
        a.worst_at.x,
        a.worst_at.y
    );
    assert!(a.samples > 50, "only {} stretches were measured", a.samples);
    assert_eq!(a.off_terrain, 0, "part of a road is off the terrain");
    assert!(
        a.over_fraction() <= inf_island::report::ROAD_OVER_GRADE_CEILING,
        "{:.2} % of the network is over its ceiling",
        a.over_fraction() * 100.0
    );
    assert!(b.report.roads.total_km > 0.5);
    assert!(b.mesh.is_some(), "the road mesh was not built");

    // …and the corridor is what makes that true. The carve's own log carries the
    // count, which is zero on a build with no committed roads to level to.
    let carve = b
        .log
        .iter()
        .find(|l| l.step == BuildStep::Carve)
        .expect("the carve is logged");
    assert!(
        !carve.note.contains(" 0 corridor"),
        "the corridor levelled nothing: {:?}",
        carve.note
    );
}

/// Biomes cover the land, the masks beat the classifier, and the sites are
/// reserved.
#[test]
fn the_biome_map_covers_the_land_and_the_design_wins_where_it_speaks() {
    let b = build();
    let f = b.biomes.land_fractions();
    print!("BIOMES: ");
    for x in IslandBiome::ALL {
        print!("{} {:.1}%  ", x.label(), f[x.id() as usize] * 100.0);
    }
    println!("\nBREAKS: {:?}", b.biomes.breaks);
    let sum: f64 = f[1..].iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "the land fractions sum to {sum}");
    assert!(b.biomes.land_cells() > 1_000);
    // Every biome the design can produce is actually produced — a palette with
    // an unreachable entry is a palette that says nothing.
    for x in IslandBiome::ALL {
        assert!(
            b.biomes.cells[x.id() as usize] > 0,
            "no cell anywhere is {}",
            x.label()
        );
    }
    assert!(b.biomes.masked > 0, "the design masks stamped nothing");
    assert!(b.biomes.reserved > 0, "the sites reserved nothing");

    // **`masked` COUNTS THE MASKS, NOT ONE BIOME.** It was `id == Farmland`, and
    // both committed mask layers name meadow as well — so the report and the
    // build log under-stated what the design overrode by every meadow the author
    // had painted, and a meadow the classifier chose looked exactly like one an
    // author drew. Farmland is design-only (the classifier never produces it), so
    // its cell count is exactly the farmland masks' contribution, and the meadow
    // masks are what the difference is.
    let farmland = b.biomes.cells[IslandBiome::Farmland.id() as usize];
    println!(
        "MASKS: {} cell(s) decided by a design mask, of which {farmland} are \
         farmland — the rest are the meadow masks the old count could not see",
        b.biomes.masked
    );
    assert!(
        b.biomes.masked > farmland,
        "the mask count ({}) is exactly the farmland count ({farmland}), so it is \
         still counting one biome rather than the masks",
        b.biomes.masked
    );

    // The terrain really carries the ids: ask it at a site (urban) and inside the
    // farmland mask.
    let site = &b.recipe.sites[0];
    assert_eq!(
        b.terrain.biome_at(DVec2::new(site.x, site.z)),
        Some(IslandBiome::Urban.id()),
        "a city site must be reserved on the terrain, not just in the report"
    );
    assert_eq!(b.biome_set.biomes.len(), IslandBiome::ALL.len());
    b.biome_set.validate().expect("the biome set is valid");
}

/// **The ground is four materials, not one colour** (wave TER2a, clause 2).
///
/// Before this wave the island wrote biome ids and nothing else, so every one of
/// its samples shipped `DEFAULT_WEIGHT` — 100 % of layer 0 — and three of its
/// four declared `TerrainLayer`s were unreachable. This is the arm that says the
/// splat exists, that it *blends* rather than picking, and that its invariant
/// holds sample for sample on real ground.
///
/// # It reads the TERRAIN, not the report
///
/// The P21 law. `SplatStats` is what the build says it did; the assertions below
/// walk the built tiles and count what is actually written on them, so a stamp
/// that returned perfect statistics and wrote nothing fails here.
#[test]
fn the_ground_carries_a_real_splat_and_every_weight_sums_to_255() {
    let b = build();
    let st = b.splat;
    println!("SPLAT: {}", st.summary());
    assert!(st.samples > 0, "the splat wrote nothing");
    assert_eq!(st.sum_violations, 0, "a weight did not sum to 255");

    // Walk the terrain itself: every level-0 tile carries a real buffer, every
    // sample sums to 255, and the coverage the report claims is the coverage the
    // tiles have.
    let res = b.terrain.tile_resolution();
    let mut seen = 0u64;
    let mut default_tiles = 0u64;
    let mut dominant = [0u64; 4];
    for (_, tile) in b.terrain.tiles() {
        if tile.weights_are_default() {
            default_tiles += 1;
            continue;
        }
        for j in 0..res {
            for i in 0..res {
                let w = tile.weight_sample(res, i, j);
                let s: u32 = w.iter().map(|&v| u32::from(v)).sum();
                assert_eq!(s, 255, "a written weight sums to {s}: {w:?}");
                let mut best = 0usize;
                for k in 0..4 {
                    if w[k] > w[best] {
                        best = k;
                    }
                }
                dominant[best] += 1;
                seen += 1;
            }
        }
    }
    assert_eq!(
        default_tiles, 0,
        "{default_tiles} tiles ship the flat default"
    );
    assert_eq!(seen, st.samples, "the report counted a different terrain");
    assert_eq!(
        dominant,
        [
            st.dominant[0],
            st.dominant[1],
            st.dominant[2],
            st.dominant[3]
        ],
        "the report's coverage is not the terrain's"
    );

    // **Every one of the four layers is reached.** Three of them were declared
    // and unreachable before this wave, and a splat that only ever writes one
    // channel is the same flat colour wearing a mask.
    for (k, name) in ["grass", "rock", "forest floor", "sand"].iter().enumerate() {
        assert!(
            dominant[k] > 0,
            "no sample on the island is dominated by {name} (layer {k})"
        );
    }

    // …and it BLENDS. A paint-by-numbers mask would put every sample on one
    // channel at 255 and never feather a boundary.
    assert!(
        st.blended_fraction() > 0.02,
        "only {:.3} % of the island blends two layers — the boundaries are steps, \
         not feathers",
        st.blended_fraction() * 100.0
    );

    // The slope term did something the 8 m classification could not: this island
    // is a piece of a mountain, so faces over the rock angle exist.
    assert!(
        st.rock_by_slope > 0,
        "not one sample was pushed toward rock by its own slope"
    );

    // **The pyramid rule still holds**: weights are level-0 only, and a coarse
    // page shades off layer 0. Read off the built asset, which is what ships.
    let asset = b.asset.as_ref().expect("the fixture builds an asset");
    let reader = asset.reader();
    let mut coarse_checked = 0u64;
    let coarse_keys: Vec<_> = reader.keys().filter(|k| k.lod > 0).collect();
    for key in coarse_keys {
        let tile = reader
            .tile(key)
            .expect("a catalogued tile decodes")
            .expect("a catalogued tile is present");
        assert!(
            tile.weights_are_default(),
            "lod {} tile {:?} leaked splat weights into the pyramid",
            key.lod,
            key.coord
        );
        coarse_checked += 1;
    }
    assert!(
        coarse_checked > 0,
        "the pyramid has no coarse tiles to check"
    );
    println!(
        "SPLAT PYRAMID: {coarse_checked} coarse tiles carry no weights (the \
         existing layer-reduction rule, restated because this wave makes it \
         visible)"
    );
}

/// **Two builds of one recipe are byte-identical**, which is what makes the
/// terrain an artifact rather than a roll of the dice.
///
/// Same machine, not cross-platform: the sample step goes through the projection
/// modules the portability law exempts by name, and this repository does not
/// pretend otherwise. See `inf_island`'s own header.
#[test]
fn two_builds_of_one_recipe_produce_the_same_terrain() {
    let a = build();
    let c = build();
    let (aa, cc) = (
        a.asset.as_ref().expect("a terrain"),
        c.asset.as_ref().expect("a terrain"),
    );
    assert_eq!(
        aa.as_bytes().len(),
        cc.as_bytes().len(),
        "two builds produced different-sized terrain"
    );
    assert!(
        aa.as_bytes() == cc.as_bytes(),
        "two builds of one recipe are not byte-identical"
    );
    println!(
        "DETERMINISM: {} bytes, {} tiles, {} LOD levels — identical across two builds",
        aa.as_bytes().len(),
        aa.reader().tile_count(),
        aa.reader().lod_levels()
    );

    // The header carries the survey's own origin, which is what makes the terrain
    // land where the anchor says it does.
    let r = aa.reader();
    assert_eq!(r.origin().x, a.anchor.origin_easting_m);
    assert_eq!(r.origin().z, a.anchor.origin_northing_m);
    assert_eq!(r.tile_resolution(), a.recipe.grid.tile_resolution);
    assert_eq!(r.meters_per_sample(), a.recipe.grid.meters_per_sample);
    assert!(
        u64::try_from(r.tile_count()).unwrap() > a.recipe.grid.tile_count(),
        "the catalog must hold more than the level-0 tiles"
    );
    // And the derived asset ids are a function of the island's name.
    assert_ne!(
        inf_island::terrain_guid(&a.recipe.name),
        inf_island::road_mesh_guid(&a.recipe.name)
    );
}

/// The anchor is the one the mandate names, and the world frame agrees with the
/// compass the sky was pinned to in Phase 17.
#[test]
fn the_island_is_anchored_in_utm_zone_10n_with_north_at_minus_z() {
    let b = build();
    let a = &b.anchor;
    assert!(a.enabled);
    assert_eq!(a.crs, "EPSG:32610");
    assert!(
        (49.0..50.0).contains(&a.origin_latitude_deg)
            && (-124.0..-122.0).contains(&a.origin_longitude_deg),
        "the anchor is at {}, {}",
        a.origin_latitude_deg,
        a.origin_longitude_deg
    );
    assert_eq!(
        inf_gis::suggested_utm_epsg(a.origin_latitude_deg, a.origin_longitude_deg),
        32610
    );
    // North is -Z: a position one kilometre north of the origin has a LOWER
    // northing subtracted, i.e. a greater one in the CRS.
    let north = a.projected_from_world(glam::DVec3::new(0.0, 0.0, -1_000.0));
    assert!(
        north.1 > a.origin_northing_m,
        "walking to -Z must increase the northing; it went {} from {}",
        north.1,
        a.origin_northing_m
    );
    println!(
        "ANCHOR: {} at {:.5} N {:.5} E, convergence {:.4} deg, datum {}",
        a.crs,
        a.origin_latitude_deg,
        a.origin_longitude_deg,
        a.grid_convergence_deg,
        a.vertical_datum
    );
}

/// A **scratch copy** of the fixture: the recipe and its layers in a temp
/// directory, with `[source] cache` pointed back at the committed tiles.
///
/// Any arm that exercises a build option which *writes* has to run here, because
/// the committed fixture is source: an arm that ran `planning_pass()` against
/// `samples/island-fixture` would rewrite the repository's own committed design
/// as a side effect of `cargo test`.
fn scratch_fixture(tmp: &Path) -> PathBuf {
    let src = fixture_dir();
    std::fs::create_dir_all(tmp.join("layers")).expect("the scratch layers directory");
    for f in [
        "coast.geojson",
        "roads.geojson",
        "streams.geojson",
        "lakes.geojson",
        "biomes.geojson",
    ] {
        std::fs::copy(src.join("layers").join(f), tmp.join("layers").join(f))
            .unwrap_or_else(|e| panic!("copy layers/{f}: {e}"));
    }
    let cache = src.join("tiles");
    let text = std::fs::read_to_string(src.join("island.toml"))
        .expect("the fixture recipe reads")
        .replace(
            "cache = \"tiles\"",
            &format!("cache = {:?}", cache.display().to_string()),
        );
    let path = tmp.join("island.toml");
    std::fs::write(&path, text).expect("the scratch recipe writes");
    path
}

/// **A ROAD RE-PLAN LEAVES THE COMMITTED WATER DESIGN ALONE.**
///
/// `inf island route` runs [`BuildOptions::planning_pass`], and that pass carried
/// `rederive_layers: true` — so a verb whose subject is the road network
/// overwrote the committed **stream and lake** layers as well, silently. An
/// author who moved a reach and then re-routed the roads lost the reach. That is
/// the hazard `BuildOptions::rederive_layers`'s own doc names.
///
/// Both halves are here, because the fix must not cost the first run: an island
/// with no derived water yet still gets it written.
#[test]
fn a_road_replan_leaves_the_committed_water_design_alone() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let path = scratch_fixture(tmp.path());
    let r = IslandRecipe::load(&path).expect("the scratch recipe loads");
    let streams = tmp.path().join("layers/streams.geojson");
    let lakes = tmp.path().join("layers/lakes.geojson");

    // The author's edit: a reach they renamed by hand. Anything in the file
    // would do — the point is that it is not what a derivation would produce.
    let edited = std::fs::read_to_string(&streams)
        .expect("the scratch streams read")
        .replacen("\"Reach 0\"", "\"The Author's Own Reach\"", 1);
    assert!(
        edited.contains("The Author's Own Reach"),
        "the fixture's stream layer has no `Reach 0` to edit — this arm is \
         measuring nothing"
    );
    std::fs::write(&streams, &edited).expect("the edit writes");
    let lakes_before = std::fs::read(&lakes).expect("the scratch lakes read");

    let b = inf_island::build_island(&r, &BuildOptions::planning_pass())
        .expect("the planning pass runs");
    assert!(!b.routes.is_empty(), "the planning pass planned no road");
    println!(
        "REPLAN: {} route(s) planned, {} reach(es) read back",
        b.routes.len(),
        b.network.streams.len()
    );

    // Compared as booleans rather than with `assert_eq!`: these layers are tens
    // of kilobytes and a failure that dumps both copies buries its own message.
    let after = std::fs::read_to_string(&streams).expect("re-read");
    assert!(
        after == edited,
        "`inf island route` rewrote the committed stream layer — an author's \
         edit lasted exactly until the next re-route. The renamed reach is {} \
         after the pass ({} bytes before, {} after)",
        if after.contains("The Author's Own Reach") {
            "still named, but the bytes moved"
        } else {
            "GONE"
        },
        edited.len(),
        after.len()
    );
    assert!(
        std::fs::read(&lakes).expect("re-read") == lakes_before,
        "`inf island route` rewrote the committed lake layer ({} bytes before)",
        lakes_before.len()
    );
    // …and the road layer, which IS this verb's subject, was written.
    let roads = std::fs::read_to_string(tmp.path().join("layers/roads.geojson"))
        .expect("the roads layer reads");
    assert!(
        roads.contains("Fixture Town"),
        "{}",
        &roads[..80.min(roads.len())]
    );

    // **The first run still derives.** With no committed water at all the same
    // pass writes both layers, so the fix costs a new island nothing.
    std::fs::remove_file(&streams).expect("remove the streams");
    std::fs::remove_file(&lakes).expect("remove the lakes");
    let b2 = inf_island::build_island(&r, &BuildOptions::planning_pass())
        .expect("the planning pass runs on a fresh island");
    assert!(
        streams.exists() && lakes.exists(),
        "a fresh island got no water"
    );
    assert!(
        !b2.network.streams.is_empty() && !b2.network.lakes.is_empty(),
        "a fresh island derived {} reach(es) and {} lake(s)",
        b2.network.streams.len(),
        b2.network.lakes.len()
    );
    println!(
        "FRESH: {} reach(es) / {} lake(s) derived and written",
        b2.network.streams.len(),
        b2.network.lakes.len()
    );
}

/// **THE REPORTED START IS THE ONE THE LEVEL SPAWNS AT.**
///
/// `inf island build` prints a `start` line. It came from
/// `IslandBuild::player_start`, which read the **built terrain**; the level's own
/// hero comes from `IslandDesign::start`, which reads the **committed road
/// layer**. Two doors onto one question, and the command printed the one nothing
/// spawns at.
///
/// The road door is the one that survives: the level is authored from committed
/// design alone, so the terrain's answer is not available to it by construction.
/// This asserts the two doors are now the same answer, and prints the gap they
/// used to have — which is also a useful number in its own right, because it is
/// the corridor levelling and the channel carve having moved the ground under a
/// route since it was planned.
#[test]
fn the_reported_start_is_the_one_the_level_spawns_at() {
    let b = build();
    let reported = b.player_start();
    let design = inf_island::read_design(&b.recipe).expect("the committed design reads");
    let level = design.start(0.0);
    println!(
        "START: report ({:.3}, {:.3}, {:.3}), level ({:.3}, {:.3}, {:.3}); the \
         terrain under it is {:?}",
        reported.x,
        reported.y,
        reported.z,
        level.x,
        level.y,
        level.z,
        b.ground_under_start()
            .map(|h| (h * 1000.0).round() / 1000.0)
    );
    assert_eq!(
        reported, level,
        "the build reports one start and the level spawns at another"
    );
    // …and the gap the old door had, as a number rather than a claim.
    let ground = b
        .ground_under_start()
        .expect("the reported start is on the terrain");
    println!(
        "START GAP: the road layer plans {:.3} m and the built terrain carries \
         {:.3} m — {:.3} m apart",
        level.y,
        ground,
        (ground - level.y).abs()
    );
    // A sanity bound rather than a pin: past a few metres the corridor levelling
    // has moved the ground out from under its own route.
    assert!(
        (ground - level.y).abs() < 5.0,
        "the committed road vertex nearest the start is {:.3} m from the ground \
         the build put there",
        (ground - level.y).abs()
    );
    assert!(
        level.y > b.recipe.sea.level_m,
        "the start is under the waterline"
    );
    // The lift is added on top, and it is the only difference the level makes.
    let lifted = design.start(2.5);
    assert_eq!(lifted.y - level.y, 2.5);
    assert_eq!((lifted.x, lifted.z), (level.x, level.z));
}

/// **The detail band fills the gap under the survey, and nothing the design
/// already decided moves** (wave TER2b).
///
/// # Why this arm and not the coverage arm
///
/// The step-coverage arm counts a log line. This one measures the *ground*: it
/// asks the finished terrain whether the detail happened where it should have and
/// — the load-bearing half — whether it stayed out of the four places something
/// downstream measures.
///
/// The road half is the strongest of the four, and it is a **re-measurement**.
/// `roads::grade_audit` runs inside the Roads step, one step before the detail is
/// written, so the audit the report carries could not have seen it however wrong
/// the mask was. This arm runs the identical audit again against `b.terrain` — the
/// ground as the build finished it, detail and all — and demands the same answer
/// to the last bit. Mutation: widen the corridor mask to nothing and this fails
/// with a different worst grade.
#[test]
fn the_detail_band_fills_the_gap_under_the_survey_and_moves_nothing_designed() {
    let r = recipe();
    let b = build();
    let note = &b
        .log
        .iter()
        .find(|l| l.step == BuildStep::Detail)
        .expect("the detail step is logged")
        .note;
    println!("DETAIL: {note}");

    // 1. IT HAPPENED. A stage that silently did nothing would satisfy every
    //    exclusion below perfectly.
    assert!(
        !note.starts_with("skipped"),
        "the fixture is a {:.2}x upsample and got no detail band: {note}",
        b.plan.upsample_ratio()
    );
    let band = inf_island::DetailBand::of(b.plan.grid_m_per_sample, b.plan.ground_m_per_px)
        .expect("the fixture upsamples");
    // 2. THE BAND IS UNDER THE SURVEY AND OVER THE GRID. Both ends matter: the
    //    coarse end would overwrite surveyed relief, the fine end would alias.
    assert!(
        band.base_wavelength_m <= 2.0 * b.plan.ground_m_per_px + 1e-9,
        "the base octave is {} m, coarser than the source's own {} m Nyquist -- \
         that is overwriting survey",
        band.base_wavelength_m,
        2.0 * b.plan.ground_m_per_px
    );
    assert!(
        band.finest_wavelength_m >= 2.0 * b.plan.grid_m_per_sample - 1e-9,
        "the finest octave is {} m against a {} m grid -- that is aliasing",
        band.finest_wavelength_m,
        b.plan.grid_m_per_sample
    );

    // 3. **THE ROADS.** The same audit, re-run on the ground the build FINISHED
    //    with. `grade_audit` ran before the detail existed, so this is the only
    //    measurement that can see a corridor mask that leaks.
    let before = &b.report.roads.audit;
    let after =
        inf_island::roads::grade_audit(&b.routes, r.roads.max_grade, r.roads.grade_step_m, |p| {
            b.terrain.height_at(p)
        });
    println!(
        "ROAD GRADE after the detail band: worst {:.6} against {:.6} before, {} \
         over against {}, {} samples against {}",
        after.worst,
        before.worst,
        after.over.len(),
        before.over.len(),
        after.samples,
        before.samples
    );
    assert_eq!(
        after.samples, before.samples,
        "the audit measured a different number of stretches"
    );
    assert_eq!(after.off_terrain, before.off_terrain);
    assert_eq!(
        after.over.len(),
        before.over.len(),
        "the detail band pushed stretches over the grade ceiling"
    );
    assert_eq!(
        after.worst.to_bits(),
        before.worst.to_bits(),
        "the worst grade moved from {} to {} -- the corridor mask leaked",
        before.worst,
        after.worst
    );

    // 4. **THE SHORE.** Every coastline vertex is still at the waterline, to the
    //    same tolerance the carve arm holds it to and by a factor of hundreds.
    let sea = r.sea.level_m;
    let mut worst = 0.0f64;
    let mut probes = 0usize;
    for ring in b.coast.rings() {
        for v in ring {
            if let Some(h) = b.terrain.height_at(*v) {
                worst = worst.max((h - sea).abs());
                probes += 1;
            }
        }
    }
    assert!(probes >= 8);
    println!("SHORE after the detail band: worst {worst:.4} m over {probes} vertices");
    assert!(
        worst < 2.5,
        "a coastline vertex is {worst} m from the waterline after the detail band"
    );

    // 5. **THE AMPLITUDE.** Its own ceiling, read off the log rather than
    //    re-derived, because what the ledger prints is what a reader trusts.
    let worst_m: f64 = note
        .split("worst ")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .and_then(|s| s.parse().ok())
        .expect("the detail note prints its worst displacement");
    assert!(
        worst_m <= inf_island::detail::MAX_AMPLITUDE_M,
        "the band moved a sample {worst_m} m, past its own {} m ceiling",
        inf_island::detail::MAX_AMPLITUDE_M
    );
    assert!(
        worst_m > 0.0,
        "the band's worst displacement is zero, so nothing above measures anything"
    );
}

/// **No lake bed rises above its own water surface** (wave TER2b audit).
///
/// # The fifth exclusion that is not there
///
/// The detail stage names four exclusions and the first of them is *the sea* —
/// at and below the waterline, fading in over six metres. An **inland lake** is
/// none of those things: it is inside the coastline and its bed sits well above
/// sea level, so `is_land` is true, the shore fade is fully open, and the band
/// writes designed relief into a bed the design has already put water on top of.
///
/// The audit measured it rather than arguing it. On the fixture the one lake is
/// **1.389 m deep** and its shallowest bed sample finishes **0.572 m** under the
/// surface — so nothing pokes through today, and the margin is smaller than the
/// stage's own `MAX_AMPLITUDE_M` ceiling. A shallower lake, or a bed the
/// classifier calls alpine (which takes the full amplitude rather than 0.7 of
/// it), is one that grows an island the design never drew.
///
/// So this is the arm that says so. It is not a tripwire: it asserts the right
/// outcome and it is green. The day it is not, the answer is a fifth exclusion —
/// the lake's own outline and level, on exactly the sea's rule — and that is
/// stated in the wave's carried list rather than built here, because it moves
/// committed bytes.
#[test]
fn no_lake_bed_rises_above_its_own_water_surface() {
    /// Even–odd point-in-polygon over a lake's committed outline.
    fn inside(poly: &[glam::DVec2], p: glam::DVec2) -> bool {
        let mut hit = false;
        let n = poly.len();
        for i in 0..n {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            if (a.y > p.y) != (b.y > p.y) {
                let t = (p.y - a.y) / (b.y - a.y);
                if p.x < a.x + t * (b.x - a.x) {
                    hit = !hit;
                }
            }
        }
        hit
    }

    let b = build();
    assert!(!b.network.lakes.is_empty(), "the fixture derived no lake");
    let mps = b.terrain.meters_per_sample();
    for (n, lake) in b.network.lakes.iter().enumerate() {
        assert!(lake.max_depth_m > 0.0, "lake {n} has no depth");
        let lo = lake.centre - lake.half_extent;
        let hi = lake.centre + lake.half_extent;
        let (mut probed, mut above) = (0usize, 0usize);
        let mut worst = f64::NEG_INFINITY;
        let mut z = lo.y;
        while z <= hi.y {
            let mut x = lo.x;
            while x <= hi.x {
                let p = glam::DVec2::new(x, z);
                if inside(&lake.outline, p) {
                    if let Some(h) = b.terrain.height_at(p) {
                        probed += 1;
                        worst = worst.max(h - lake.level_m);
                        if h > lake.level_m {
                            above += 1;
                        }
                    }
                }
                x += mps;
            }
            z += mps;
        }
        println!(
            "LAKE {n}: level {:.3} m, {:.0} m2, {:.3} m deep; {probed} bed samples \
             read, {above} above the surface, shallowest {:.3} m under it",
            lake.level_m, lake.area_m2, lake.max_depth_m, -worst
        );
        // NOT VACUOUS: the walk really read the bed.
        assert!(
            probed >= 16,
            "lake {n} gave only {probed} bed samples, so the assertion below is \
             about nothing"
        );
        assert_eq!(
            above, 0,
            "{above} of lake {n}'s {probed} bed samples finish ABOVE its {:.3} m \
             surface after the detail band -- the band writes into lake beds and \
             the exclusion list does not mention lakes",
            lake.level_m
        );
    }
}
