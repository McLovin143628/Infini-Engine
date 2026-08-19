//! Web-Mercator XYZ tile arithmetic (ported from GeoCanvas).
//!
//! # Provenance
//!
//! Adapted from `geocanvas-core/src/ops/terrain/tilemath.rs`, whose own header
//! reads *"Pure math — no GDAL, unit-tested below."* Ported rather than copied:
//! the types are ours, the tests are re-derived, and the doc explains what the
//! numbers mean here rather than what they meant there. What crossed unchanged is
//! the one habit worth keeping, called out below.
//!
//! # The habit worth porting
//!
//! [`tile_mercator_bounds`] computes **each edge from its own index** rather than
//! as `min + size`. That is not a style preference — it is why adjacent tiles
//! share bit-identical edges. `min + size` accumulates a different rounding error
//! per tile, so tile 10's east edge and tile 11's west edge differ in the last
//! ULP, and a terrain built from them has a hairline seam at every tile boundary
//! that no amount of vertex welding removes. The same reasoning as this engine's
//! own shared-edge tiling rule (a tile's last row *is* the next tile's first).
//!
//! # Why this is here at all
//!
//! Global elevation is published as XYZ-tiled terrarium PNGs — a keyless,
//! worldwide DEM source. This module plus [`crate::terrarium`] turns "a
//! directory of 256×256 PNGs" into "elevations in metres at known ground
//! positions", with **no new dependency** (we already decode PNG row-at-a-time)
//! and no network access in the engine itself.

/// Half the Mercator world span: `π · 6 378 137` m (the WGS84 spherical radius
/// the Web-Mercator development uses).
pub const MERC_HALF_WORLD: f64 = 20_037_508.342_789_244;

/// The full Mercator world span — the edge length of the z0 tile, in metres.
pub const MERC_WORLD: f64 = 2.0 * MERC_HALF_WORLD;

/// Tile edge in pixels. 256 is the XYZ/terrarium standard.
pub const TILE_PX: usize = 256;

/// The finest zoom worth asking for. z22 tile pixels are ~37 mm; finer source
/// data gains nothing a game world can use.
pub const MAX_NATIVE_ZOOM: u8 = 22;

/// Mercator bounds `[min_x, min_y, max_x, max_y]` of tile `(z, x, y)`.
///
/// `z0/0/0` is the whole world; `y` counts **down from the north edge** (the XYZ
/// scheme). See the module docs for why each edge is computed independently.
pub fn tile_mercator_bounds(z: u8, x: u32, y: u32) -> [f64; 4] {
    let n = (1u64 << z.min(31)) as f64;
    let ts = MERC_WORLD / n;
    [
        -MERC_HALF_WORLD + f64::from(x) * ts,
        MERC_HALF_WORLD - f64::from(y.saturating_add(1)) * ts,
        -MERC_HALF_WORLD + f64::from(x.saturating_add(1)) * ts,
        MERC_HALF_WORLD - f64::from(y) * ts,
    ]
}

/// Number of tiles across one axis at zoom `z`.
#[inline]
pub fn tiles_at_zoom(z: u8) -> u64 {
    1u64 << z.min(31)
}

/// Geodetic (longitude, latitude) in **degrees** → Web-Mercator metres.
///
/// Latitude is clamped to [`crate::crs::MAX_LATITUDE_DEG`]: the Mercator
/// northing diverges at the poles, and returning an infinity here would just
/// move the failure somewhere less informative.
///
/// A non-finite input is **refused**, not clamped: `NaN.clamp(a, b)` is `NaN`,
/// and a NaN mercator coordinate becomes a NaN tile fraction, and
/// `NaN.floor().max(0.0)` is `0.0` — because `f64::max` returns the *other*
/// operand when one is NaN. So the un-guarded version turned a NaN coordinate
/// into tile `(0, 0)`: a perfectly valid-looking tile in the Atlantic, with no
/// error anywhere. That is the crate's own headline hazard, met in the module
/// that was written to avoid the *other* one.
pub fn lonlat_to_mercator(lon_deg: f64, lat_deg: f64) -> Option<(f64, f64)> {
    if !(lon_deg.is_finite() && lat_deg.is_finite()) {
        return None;
    }
    let lon = lon_deg.clamp(-180.0, 180.0);
    let lat = lat_deg.clamp(-crate::crs::MAX_LATITUDE_DEG, crate::crs::MAX_LATITUDE_DEG);
    let x = lon * MERC_HALF_WORLD / 180.0;
    let y = ((90.0 + lat) * std::f64::consts::PI / 360.0).tan().ln() * MERC_HALF_WORLD
        / std::f64::consts::PI;
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

/// Web-Mercator metres → geodetic (longitude, latitude) in degrees.
///
/// `None` for a non-finite input, for the reason in [`lonlat_to_mercator`].
pub fn mercator_to_lonlat(x: f64, y: f64) -> Option<(f64, f64)> {
    if !(x.is_finite() && y.is_finite()) {
        return None;
    }
    let lon = x * 180.0 / MERC_HALF_WORLD;
    let lat = (2.0 * (y * std::f64::consts::PI / MERC_HALF_WORLD).exp().atan()
        - std::f64::consts::FRAC_PI_2)
        .to_degrees();
    (lon.is_finite() && lat.is_finite()).then_some((lon, lat))
}

/// The `(x, y)` tile at zoom `z` containing a geodetic position.
///
/// `None` rather than a plausible tile for a non-finite position — see
/// [`lonlat_to_mercator`].
pub fn tile_of_lonlat(z: u8, lon_deg: f64, lat_deg: f64) -> Option<(u32, u32)> {
    let n = tiles_at_zoom(z) as f64;
    let (mx, my) = lonlat_to_mercator(lon_deg, lat_deg)?;
    let fx = (mx + MERC_HALF_WORLD) / MERC_WORLD * n;
    let fy = (MERC_HALF_WORLD - my) / MERC_WORLD * n;
    if !(fx.is_finite() && fy.is_finite()) {
        return None;
    }
    let cap = (n as u32).saturating_sub(1);
    Some((
        (fx.floor().max(0.0) as u32).min(cap),
        (fy.floor().max(0.0) as u32).min(cap),
    ))
}

/// The finest zoom whose tile pixels are at least as fine as a source of
/// `px_3857` metres per pixel, clamped to `0..=MAX_NATIVE_ZOOM`.
///
/// The tiny epsilon keeps an exact power-of-two pixel size from ceiling one
/// level too far on float noise.
pub fn native_maxzoom(px_3857: f64) -> u8 {
    if !px_3857.is_finite() || px_3857 <= 0.0 {
        return MAX_NATIVE_ZOOM;
    }
    let z = (MERC_WORLD / (px_3857 * TILE_PX as f64)).log2() - 1e-9;
    z.ceil().clamp(0.0, f64::from(MAX_NATIVE_ZOOM)) as u8
}

/// Ground resolution in metres per pixel at zoom `z` and a given latitude.
///
/// **This is the number that matters when planning an import**, and it is not
/// the Mercator one: Web-Mercator "metres" are inflated by `1/cos(latitude)`, so
/// a z13 tile pixel is ~19.1 m at the equator but ~12.5 m at Vancouver's 49° N.
/// Reporting the inflated figure would tell an author their DEM is coarser than
/// it is.
///
/// A non-finite latitude yields `None` rather than a NaN metres-per-pixel that
/// then compares `false` against every threshold in [`plan_zoom`] and quietly
/// selects the finest zoom in the table.
pub fn ground_resolution_m(z: u8, lat_deg: f64) -> Option<f64> {
    if !lat_deg.is_finite() {
        return None;
    }
    let n = tiles_at_zoom(z) as f64 * TILE_PX as f64;
    let lat = lat_deg.clamp(-crate::crs::MAX_LATITUDE_DEG, crate::crs::MAX_LATITUDE_DEG);
    let m = MERC_WORLD / n * lat.to_radians().cos();
    m.is_finite().then_some(m)
}

/// The zoom whose ground resolution at `lat_deg` is closest to (and no coarser
/// than) `target_m` metres per sample — the "what zoom do I need?" planner.
pub fn plan_zoom(target_m: f64, lat_deg: f64) -> u8 {
    if !(target_m.is_finite() && target_m > 0.0) {
        return MAX_NATIVE_ZOOM;
    }
    for z in 0..=MAX_NATIVE_ZOOM {
        if ground_resolution_m(z, lat_deg).is_some_and(|m| m <= target_m) {
            return z;
        }
    }
    MAX_NATIVE_ZOOM
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z0_is_the_whole_world_and_z1_is_its_quadrants() {
        assert_eq!(
            tile_mercator_bounds(0, 0, 0),
            [
                -MERC_HALF_WORLD,
                -MERC_HALF_WORLD,
                MERC_HALF_WORLD,
                MERC_HALF_WORLD
            ]
        );
        // (0,0) is the NORTH-WEST quadrant — y counts down from the north edge.
        assert_eq!(
            tile_mercator_bounds(1, 0, 0),
            [-MERC_HALF_WORLD, 0.0, 0.0, MERC_HALF_WORLD]
        );
        assert_eq!(
            tile_mercator_bounds(1, 1, 1),
            [0.0, -MERC_HALF_WORLD, MERC_HALF_WORLD, 0.0]
        );
    }

    /// **The seam gate** — the one property this port exists to preserve.
    ///
    /// Un-fix mutation: compute the max edges as `min + ts` and this fails at
    /// most indices, because the two expressions round differently.
    #[test]
    fn adjacent_tiles_share_bit_identical_edges() {
        for z in [1u8, 5, 12, 18] {
            let n = tiles_at_zoom(z) as u32;
            for x in [0u32, 1, n / 3, n / 2, n - 2] {
                let a = tile_mercator_bounds(z, x, 12 % n);
                let b = tile_mercator_bounds(z, x + 1, 12 % n);
                assert_eq!(
                    a[2], b[0],
                    "z{z} x{x}: the east edge is not bit-identical to the next west edge"
                );
                let c = tile_mercator_bounds(z, x, 12 % n + 1);
                assert_eq!(a[1], c[3], "z{z} x{x}: south edge != next north edge");
            }
        }
    }

    #[test]
    fn mercator_round_trips_and_places_known_cities() {
        for (lon, lat) in [
            (0.0, 0.0),
            (-123.12, 49.28),
            (151.21, -33.87),
            (-157.8, 21.3),
        ] {
            let (x, y) = lonlat_to_mercator(lon, lat).expect("a finite position projects");
            let (lon2, lat2) = mercator_to_lonlat(x, y).expect("and inverts");
            assert!((lon - lon2).abs() < 1e-9, "{lon} -> {lon2}");
            assert!((lat - lat2).abs() < 1e-9, "{lat} -> {lat2}");
        }
        // The equator/prime meridian is the world centre. (`tan(pi/4).ln()` is
        // not exactly 0 in floating point, so this is "at the centre", not "bit
        // zero" — a sub-nanometre offset on a 40 000 km world.)
        let (cx, cy) = lonlat_to_mercator(0.0, 0.0).unwrap();
        assert_eq!(cx, 0.0);
        assert!(cy.abs() < 1e-6, "the equator is at y={cy}");
        // Vancouver is in the northern hemisphere and west of Greenwich.
        let (x, y) = lonlat_to_mercator(-123.12, 49.28).unwrap();
        assert!(x < 0.0 && y > 0.0);
        // The poles clamp rather than going infinite.
        assert!(lonlat_to_mercator(0.0, 90.0).unwrap().1.is_finite());
        assert!(lonlat_to_mercator(0.0, -90.0).unwrap().1.is_finite());

        // **A non-finite position is REFUSED, not turned into tile (0, 0).**
        // `NaN.clamp(a, b)` is NaN, and `NaN.floor().max(0.0)` is 0.0 because
        // `f64::max` returns the other operand against a NaN — so the unguarded
        // version answered "tile (0, 0)", a real-looking tile in the Atlantic,
        // for a coordinate that does not exist.
        //
        // Un-fix mutation: drop the finiteness guards and every assertion below
        // returns a `Some` instead of `None`.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(lonlat_to_mercator(bad, 49.0), None, "lon {bad}");
            assert_eq!(lonlat_to_mercator(-123.0, bad), None, "lat {bad}");
            assert_eq!(mercator_to_lonlat(bad, 0.0), None);
            assert_eq!(
                tile_of_lonlat(12, bad, 49.0),
                None,
                "a NaN longitude must not become a plausible tile"
            );
            assert_eq!(ground_resolution_m(12, bad), None);
        }
    }

    #[test]
    fn tile_selection_agrees_with_the_bounds_it_reports() {
        for (lon, lat) in [(-123.12, 49.28), (0.0, 0.0), (151.21, -33.87)] {
            for z in [0u8, 3, 8, 13] {
                let (tx, ty) = tile_of_lonlat(z, lon, lat).expect("a finite position has a tile");
                let b = tile_mercator_bounds(z, tx, ty);
                let (mx, my) = lonlat_to_mercator(lon, lat).unwrap();
                assert!(
                    mx >= b[0] && mx <= b[2] && my >= b[1] && my <= b[3],
                    "z{z} ({lon},{lat}) picked tile ({tx},{ty}) whose bounds {b:?} \
                     do not contain the point ({mx}, {my})"
                );
            }
        }
    }

    /// The resolution report is the TRUE ground figure, not the inflated
    /// Mercator one — the distinction an author plans an import against.
    #[test]
    fn ground_resolution_accounts_for_mercator_inflation() {
        let eq = ground_resolution_m(13, 0.0).unwrap();
        let van = ground_resolution_m(13, 49.28).unwrap();
        assert!(
            (eq - 19.1).abs() < 0.2,
            "z13 at the equator is ~19.1 m/px, got {eq}"
        );
        assert!(
            van < eq * 0.7,
            "at 49 N a z13 pixel covers much less GROUND than at the equator \
             ({van} vs {eq}) — if these are equal the cos(lat) term is missing"
        );
        // cos(49.28) = 0.6527, and that is exactly the ratio.
        assert!((van / eq - 49.28_f64.to_radians().cos()).abs() < 1e-9);

        // Planning picks a zoom that actually meets the target.
        for target in [1.0, 5.0, 30.0, 100.0] {
            let z = plan_zoom(target, 49.28);
            assert!(
                ground_resolution_m(z, 49.28).unwrap() <= target,
                "plan_zoom({target}) returned z{z}, which is {} m/px — coarser than asked",
                ground_resolution_m(z, 49.28).unwrap()
            );
            assert!(
                z == 0 || ground_resolution_m(z - 1, 49.28).unwrap() > target,
                "plan_zoom({target}) returned z{z} but z{} would also have met it",
                z - 1
            );
        }
        assert_eq!(plan_zoom(f64::NAN, 0.0), MAX_NATIVE_ZOOM);
        assert_eq!(plan_zoom(0.0, 0.0), MAX_NATIVE_ZOOM);
    }

    #[test]
    fn native_maxzoom_matches_known_resolutions_and_clamps_degenerates() {
        assert_eq!(native_maxzoom(1.0), 18);
        let z10_px = MERC_WORLD / (256.0 * 1024.0);
        assert_eq!(
            native_maxzoom(z10_px),
            10,
            "an exact power-of-two size must not ceil up"
        );
        assert_eq!(native_maxzoom(1e9), 0);
        assert_eq!(native_maxzoom(1e-6), MAX_NATIVE_ZOOM);
        assert_eq!(native_maxzoom(0.0), MAX_NATIVE_ZOOM);
        assert_eq!(native_maxzoom(f64::NAN), MAX_NATIVE_ZOOM);
    }
}
