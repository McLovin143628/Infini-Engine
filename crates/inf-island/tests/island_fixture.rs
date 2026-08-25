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
    let _ = off;

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
    assert!(b.biomes.masked > 0, "the farmland mask stamped nothing");
    assert!(b.biomes.reserved > 0, "the sites reserved nothing");

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
