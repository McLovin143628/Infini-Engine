//! An ASCII preview of a recipe's **source** elevation, before anything is
//! carved.
//!
//! # Why a probe and not a viewer
//!
//! A coastline is designed, and designing one against ground you cannot see is
//! guessing. This prints the sampled elevation over the world's own coordinate
//! frame — the frame the coastline polygon is authored in — so a shore vertex
//! can be placed where the ground actually is.
//!
//! `#[ignore]`d, because it needs a filled tile cache and this repository's CI
//! never fetches. Run it with:
//!
//! ```text
//! cargo test -p inf-island --test preview -- --ignored --nocapture
//! ```
//!
//! `INF_ISLAND_RECIPE` names the recipe; it defaults to the committed one.

use std::path::PathBuf;

use glam::{DVec2, DVec3};

fn recipe_path() -> PathBuf {
    std::env::var_os("INF_ISLAND_RECIPE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island/island.toml")
        })
}

/// One character per `PREVIEW_M` metres.
const PREVIEW_M: f64 = 128.0;

#[test]
#[ignore = "needs a filled tile cache; CI never fetches"]
fn print_the_source_elevation_over_the_world_frame() {
    let path = recipe_path();
    let recipe = match inf_island::IslandRecipe::load(&path) {
        Ok(r) => r,
        Err(e) => {
            println!("SKIP: {}: {e}", path.display());
            return;
        }
    };
    let plan = inf_island::plan_tiles(&recipe).expect("the recipe plans");
    let cache = recipe.cache_dir();
    if !plan.missing_in(&cache).is_empty() {
        println!(
            "SKIP: {} of {} tiles are missing from {} — run `inf island fetch`",
            plan.missing_in(&cache).len(),
            plan.len(),
            cache.display()
        );
        return;
    }
    let mosaic = inf_island::TileMosaic::load(&plan, &cache).expect("the cache decodes");
    let anchor = recipe.anchor().expect("the anchor builds");
    let tf = inf_gis::Transform::new("EPSG:4326", &anchor).expect("wgs84");
    let grid = inf_island::IslandGrid::of(&recipe);
    let (min, max) = grid.bounds();
    let lattice =
        inf_island::ProjectionLattice::build(&tf, plan.zoom, min, max).expect("the lattice builds");

    let nx = ((max.x - min.x) / PREVIEW_M).round() as usize;
    let nz = ((max.y - min.y) / PREVIEW_M).round() as usize;
    println!(
        "{} — source elevation, {PREVIEW_M} m a character, {nx} x {nz}",
        recipe.name
    );
    println!(
        "world x {:.0}..{:.0}, z {:.0}..{:.0} (+X east, -Z NORTH, so the TOP row \
         is north)",
        min.x, max.x, min.y, max.y
    );
    println!("source {:.3} m/px at z{}", plan.ground_m_per_px, plan.zoom);

    // The ramp, low to high. `.` is at or below sea level.
    const RAMP: &[u8] = b" .:-=+*#%@";
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut vals = vec![0.0f64; nx * nz];
    for j in 0..nz {
        for i in 0..nx {
            let p = DVec2::new(
                min.x + (i as f64 + 0.5) * PREVIEW_M,
                min.y + (j as f64 + 0.5) * PREVIEW_M,
            );
            let (gx, gy) = lattice.pixel_at(p);
            let v = mosaic.elevation_at_pixel(gx, gy).unwrap_or(f64::NAN);
            vals[j * nx + i] = v;
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    println!("source range {lo:.1}..{hi:.1} m");
    // Column ruler in world metres.
    print!("      ");
    for i in 0..nx {
        let x = min.x + (i as f64 + 0.5) * PREVIEW_M;
        print!(
            "{}",
            if (x / 1000.0).abs() % 1.0 < 0.128 {
                '|'
            } else {
                ' '
            }
        );
    }
    println!();
    for j in 0..nz {
        let z = min.y + (j as f64 + 0.5) * PREVIEW_M;
        print!("{z:>6.0}");
        for i in 0..nx {
            let v = vals[j * nx + i];
            let c = if !v.is_finite() {
                b'?'
            } else if v <= 0.0 {
                b'.'
            } else {
                let t = ((v - 0.0) / (hi.max(1.0))).clamp(0.0, 1.0);
                RAMP[((t * (RAMP.len() - 1) as f64).round() as usize).min(RAMP.len() - 1)]
            };
            print!("{}", c as char);
        }
        println!();
    }
    print!("      ");
    for i in 0..nx {
        let x = min.x + (i as f64 + 0.5) * PREVIEW_M;
        print!(
            "{}",
            if (x / 1000.0).abs() % 1.0 < 0.128 {
                '|'
            } else {
                ' '
            }
        );
    }
    println!("   x from {:.0} to {:.0}", min.x, max.x);

    // The sites, with the ground the source puts under each one.
    println!("\nSITES (world metres, and the source's own elevation there):");
    for s in &recipe.sites {
        let p = DVec2::new(s.x, s.z);
        let (gx, gy) = lattice.pixel_at(p);
        let h = mosaic.elevation_at_pixel(gx, gy);
        println!(
            "  {:<14} {:>8} x {:>7.0} z {:>7.0}  ground {:?}",
            s.name,
            s.kind.label(),
            s.x,
            s.z,
            h.map(|v| (v * 10.0).round() / 10.0)
        );
    }

    // And where the anchor is, for the record.
    let (lon, lat, _) = tf.to_source(DVec3::ZERO).unwrap();
    println!(
        "\nANCHOR world (0,0,0) = {lat:.5} N {lon:.5} E, UTM 10N {:.0} E {:.0} N",
        anchor.origin_easting_m, anchor.origin_northing_m
    );
}
