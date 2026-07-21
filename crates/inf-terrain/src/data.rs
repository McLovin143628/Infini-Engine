//! [`TerrainData`]: a paged `f64` world-space heightfield.
//!
//! A sparse `BTreeMap<(i32, i32), TerrainTile>` of fixed-resolution height tiles
//! — only authored tiles cost memory, so a 16 km² (or a planetary patch) terrain
//! stores just the pages that carry data. Determinism is structural: the
//! `BTreeMap` fixes iteration/serialization order, and the flat serde form is
//! portable across bincode/JSON.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::tile::TerrainTile;
use crate::HeightSource;

/// Default samples per tile side (256 × 256).
pub const DEFAULT_TILE_RESOLUTION: u32 = 256;
/// Default world units between samples (1 m).
pub const DEFAULT_METERS_PER_SAMPLE: f64 = 1.0;

/// A paged heightfield in `f64` world space.
///
/// Tiles are square blocks of `tile_resolution × tile_resolution` samples spaced
/// `meters_per_sample` apart. Tile grid coordinate `(tx, tz)` places its sample
/// `(0, 0)` at world `(tx · span, ·, tz · span)` where `span = (resolution − 1) ·
/// mps` — so a tile's **last** sample row/column coincides with the **first** of
/// the next tile (shared edges), and authoring both tiles from the same world
/// function keeps them seamless (the boundary-continuity guarantee).
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainData {
    tile_resolution: u32,
    meters_per_sample: f64,
    tiles: BTreeMap<(i32, i32), TerrainTile>,
}

impl Default for TerrainData {
    fn default() -> Self {
        Self::new(DEFAULT_TILE_RESOLUTION, DEFAULT_METERS_PER_SAMPLE)
    }
}

impl TerrainData {
    /// An empty terrain with the given tile resolution (clamped to ≥ 2 so a tile
    /// spans at least one cell) and metres-per-sample (clamped to > 0).
    pub fn new(tile_resolution: u32, meters_per_sample: f64) -> Self {
        Self {
            tile_resolution: tile_resolution.max(2),
            meters_per_sample: if meters_per_sample > 0.0 {
                meters_per_sample
            } else {
                DEFAULT_METERS_PER_SAMPLE
            },
            tiles: BTreeMap::new(),
        }
    }

    /// Samples per tile side.
    #[inline]
    pub fn tile_resolution(&self) -> u32 {
        self.tile_resolution
    }

    /// World units between adjacent samples.
    #[inline]
    pub fn meters_per_sample(&self) -> f64 {
        self.meters_per_sample
    }

    /// World size of one tile edge: `(resolution − 1) · mps`. Also the world-space
    /// stride between adjacent tile origins (so edges are shared, not doubled).
    #[inline]
    pub fn tile_span(&self) -> f64 {
        (self.tile_resolution as f64 - 1.0) * self.meters_per_sample
    }

    /// Number of authored tiles.
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// `true` when no tile is authored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// World position of tile `(tx, tz)`'s sample `(0, 0)` in the XZ plane.
    #[inline]
    pub fn tile_origin_xz(&self, coord: (i32, i32)) -> DVec2 {
        let span = self.tile_span();
        DVec2::new(coord.0 as f64 * span, coord.1 as f64 * span)
    }

    /// The tile grid coordinate containing world `(x, z)` (floored). Points on a
    /// shared edge resolve to the tile on the `+X`/`+Z` side (`local` = 0).
    #[inline]
    pub fn tile_coord_of(&self, x: f64, z: f64) -> (i32, i32) {
        let span = self.tile_span();
        ((x / span).floor() as i32, (z / span).floor() as i32)
    }

    /// Iterate authored tiles in deterministic (grid-coordinate) order.
    pub fn tiles(&self) -> impl Iterator<Item = (&(i32, i32), &TerrainTile)> {
        self.tiles.iter()
    }

    pub fn get_tile(&self, coord: (i32, i32)) -> Option<&TerrainTile> {
        self.tiles.get(&coord)
    }

    pub fn get_tile_mut(&mut self, coord: (i32, i32)) -> Option<&mut TerrainTile> {
        self.tiles.get_mut(&coord)
    }

    /// True when `(tx, tz)` is authored.
    pub fn has_tile(&self, coord: (i32, i32)) -> bool {
        self.tiles.contains_key(&coord)
    }

    /// Get `(tx, tz)`, creating it flat (offsets `0`, `origin.y = 0`) if absent,
    /// and return a mutable handle. The horizontal origin is derived from the
    /// coordinate, so tiles stay grid-aligned.
    pub fn get_or_create_tile(&mut self, coord: (i32, i32)) -> &mut TerrainTile {
        let res = self.tile_resolution;
        let o = self.tile_origin_xz(coord);
        self.tiles
            .entry(coord)
            .or_insert_with(|| TerrainTile::flat(res, DVec3::new(o.x, 0.0, o.y)))
    }

    /// Insert (or replace) a tile at `coord`. Returns the tile back if its height
    /// buffer length doesn't match `resolution²` (rejected). The horizontal origin
    /// is overwritten to keep the tile grid-aligned; the caller's `origin.y` (the
    /// `f64` height anchor) is preserved.
    pub fn insert_tile(
        &mut self,
        coord: (i32, i32),
        mut tile: TerrainTile,
    ) -> Result<(), TerrainTile> {
        if tile.heights().len() != (self.tile_resolution * self.tile_resolution) as usize {
            return Err(tile);
        }
        let o = self.tile_origin_xz(coord);
        tile.origin.x = o.x;
        tile.origin.z = o.y;
        self.tiles.insert(coord, tile);
        Ok(())
    }

    /// Remove `(tx, tz)`, returning it if it existed.
    pub fn remove_tile(&mut self, coord: (i32, i32)) -> Option<TerrainTile> {
        self.tiles.remove(&coord)
    }

    /// Author an entire tile from a world-space height function `f(x, z) → metres`
    /// (creating the tile if needed). Every sample — including the edges shared
    /// with neighbours — is written from `f`, so neighbouring tiles authored the
    /// same way stay seamless. `origin.y` stays `0`, i.e. the `f32` offset carries
    /// the full height.
    pub fn author_tile<F: FnMut(f64, f64) -> f64>(&mut self, coord: (i32, i32), mut f: F) {
        let res = self.tile_resolution;
        let mps = self.meters_per_sample;
        let o = self.tile_origin_xz(coord);
        let tile = self.get_or_create_tile(coord);
        tile.origin = DVec3::new(o.x, 0.0, o.y);
        for j in 0..res {
            for i in 0..res {
                let wx = o.x + i as f64 * mps;
                let wz = o.y + j as f64 * mps;
                tile.set_sample(res, i, j, f(wx, wz) as f32);
            }
        }
    }

    /// Bulk region write — the seam P10.2 brushes build on. For every authored (or
    /// newly created) tile sample whose world XZ falls in `[min, max]`, write
    /// `offset = f(x, z) − origin.y`. Samples on a shared edge belong to two tiles
    /// and are written in both from the same `f`, so the write stays seamless.
    pub fn write_region<F: FnMut(f64, f64) -> f64>(&mut self, min: DVec2, max: DVec2, mut f: F) {
        let (min, max) = (min.min(max), min.max(max));
        let c0 = self.tile_coord_of(min.x, min.y);
        let c1 = self.tile_coord_of(max.x, max.y);
        let res = self.tile_resolution;
        let mps = self.meters_per_sample;
        for tz in c0.1..=c1.1 {
            for tx in c0.0..=c1.0 {
                let coord = (tx, tz);
                let o = self.tile_origin_xz(coord);
                let tile = self.get_or_create_tile(coord);
                let base_y = tile.origin.y;
                for j in 0..res {
                    let wz = o.y + j as f64 * mps;
                    if wz < min.y || wz > max.y {
                        continue;
                    }
                    for i in 0..res {
                        let wx = o.x + i as f64 * mps;
                        if wx < min.x || wx > max.x {
                            continue;
                        }
                        tile.set_sample(res, i, j, (f(wx, wz) - base_y) as f32);
                    }
                }
            }
        }
    }

    /// Candidate `(tile_index, local_sample)` pairs along one axis for world
    /// coordinate `w`. Normally one pair (the floored tile); exactly on a shared
    /// edge (`local == 0`) it also yields the previous tile's far edge (`res − 1`),
    /// so a query on a single tile's far edge still resolves against it.
    #[inline]
    fn axis_candidates(&self, w: f64) -> [(i32, f64); 2] {
        let span = self.tile_span();
        let mps = self.meters_per_sample;
        let res = self.tile_resolution as f64;
        let th = (w / span).floor() as i32;
        let u = (w - th as f64 * span) / mps; // in [0, res-1)
        if u <= 1e-9 {
            [(th, u), (th - 1, res - 1.0)]
        } else {
            [(th, u), (th, u)]
        }
    }

    /// Bilinearly-sampled absolute world height at world `(x, z)`, or `None` when
    /// no authored tile contains the point. Exact at sample points (including a
    /// single tile's far edge, which resolves back onto that tile).
    pub fn height_at(&self, world_xz: DVec2) -> Option<f64> {
        let res = self.tile_resolution;
        let xs = self.axis_candidates(world_xz.x);
        let zs = self.axis_candidates(world_xz.y);
        // Prefer the floored tile; fall back to the shared-edge neighbour.
        for &(tx, u) in xs.iter().take(if xs[0].0 == xs[1].0 { 1 } else { 2 }) {
            for &(tz, v) in zs.iter().take(if zs[0].0 == zs[1].0 { 1 } else { 2 }) {
                if let Some(tile) = self.tiles.get(&(tx, tz)) {
                    let u = u.clamp(0.0, (res - 1) as f64);
                    let v = v.clamp(0.0, (res - 1) as f64);
                    let i0 = u.floor() as u32;
                    let j0 = v.floor() as u32;
                    let i1 = (i0 + 1).min(res - 1);
                    let j1 = (j0 + 1).min(res - 1);
                    let fx = u - i0 as f64;
                    let fz = v - j0 as f64;
                    let h00 = tile.sample(res, i0, j0) as f64;
                    let h10 = tile.sample(res, i1, j0) as f64;
                    let h01 = tile.sample(res, i0, j1) as f64;
                    let h11 = tile.sample(res, i1, j1) as f64;
                    let hx0 = h00 + (h10 - h00) * fx;
                    let hx1 = h01 + (h11 - h01) * fx;
                    return Some(tile.origin.y + hx0 + (hx1 - hx0) * fz);
                }
            }
        }
        None
    }

    /// Surface normal (unit, +Y up) at world `(x, z)` via central differences, or
    /// `None` when the containing tile is unauthored. Missing neighbours (terrain
    /// edge) fall back to the centre height (a one-sided slope).
    pub fn normal_at(&self, world_xz: DVec2) -> Option<DVec3> {
        let e = self.meters_per_sample;
        let c = self.height_at(world_xz)?;
        let hl = self.height_at(world_xz - DVec2::new(e, 0.0)).unwrap_or(c);
        let hr = self.height_at(world_xz + DVec2::new(e, 0.0)).unwrap_or(c);
        let hd = self.height_at(world_xz - DVec2::new(0.0, e)).unwrap_or(c);
        let hu = self.height_at(world_xz + DVec2::new(0.0, e)).unwrap_or(c);
        let dhdx = (hr - hl) / (2.0 * e);
        let dhdz = (hu - hd) / (2.0 * e);
        Some(DVec3::new(-dhdx, 1.0, -dhdz).normalize())
    }
}

impl HeightSource for TerrainData {
    fn height(&self, x: f64, z: f64) -> Option<f64> {
        self.height_at(DVec2::new(x, z))
    }

    fn normal(&self, x: f64, z: f64) -> Option<DVec3> {
        self.normal_at(DVec2::new(x, z))
    }
}

// ── serde: portable flat form + length validation ───────────────────────────

/// Wire form: config scalars + a flat `(x, z, tile)` sequence (a native
/// `BTreeMap` with tuple keys doesn't encode in JSON; the flat sequence is
/// portable across bincode/JSON and deterministic via `BTreeMap` iteration).
#[derive(Serialize, Deserialize)]
struct TerrainDataRaw {
    tile_resolution: u32,
    meters_per_sample: f64,
    #[serde(default)]
    tiles: Vec<(i32, i32, TerrainTile)>,
}

impl Serialize for TerrainData {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let raw = TerrainDataRaw {
            tile_resolution: self.tile_resolution,
            meters_per_sample: self.meters_per_sample,
            tiles: self
                .tiles
                .iter()
                .map(|(&(x, z), t)| (x, z, t.clone()))
                .collect(),
        };
        raw.serialize(s)
    }
}

impl<'de> Deserialize<'de> for TerrainData {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = TerrainDataRaw::deserialize(d)?;
        let mut data = TerrainData::new(raw.tile_resolution, raw.meters_per_sample);
        let expect = (data.tile_resolution * data.tile_resolution) as usize;
        for (x, z, tile) in raw.tiles {
            if tile.heights().len() != expect {
                return Err(serde::de::Error::custom(format!(
                    "terrain tile ({x},{z}) has {} samples, expected {expect}",
                    tile.heights().len()
                )));
            }
            data.tiles.insert((x, z), tile);
        }
        Ok(data)
    }
}
