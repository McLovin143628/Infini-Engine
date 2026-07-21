//! One page of the heightfield: a square block of `f32` sample heights measured
//! against an `f64` world anchor (the precision doctrine — tile-local `f32`
//! against an `f64` origin, so a planetary-scale terrain never loses vertical
//! resolution to `f32` range).

use glam::DVec3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A single terrain tile: `resolution × resolution` height samples (row-major,
/// `j * resolution + i`, `i` along +X, `j` along +Z) stored as `f32` offsets
/// from the tile's `f64` [`origin`](TerrainTile::origin).
///
/// The tile owns its `f64` world origin (the world position of sample `(0, 0)`):
/// horizontal `x`/`z` are set from the tile grid coordinate by [`super::TerrainData`],
/// and `y` is the anchor the `f32` heights are relative to. Absolute world height
/// of sample `(i, j)` is therefore `origin.y + heights[j*res + i] as f64` — the
/// `f32` only ever carries a tile-local delta.
///
/// Serde: `origin` (as a portable `[f64; 3]`, since the workspace `glam` pin has
/// no `serde` feature) + a flat `heights` sequence. `resolution` is not stored on
/// the tile (it is a terrain-wide constant on [`super::TerrainData`]); the terrain
/// validates every tile's `heights.len() == resolution²` on load.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainTile {
    /// World position of sample `(0, 0)` (`f64` anchor for the `f32` heights).
    pub origin: DVec3,
    /// `resolution²` height offsets (metres) from `origin.y`, row-major.
    heights: Vec<f32>,
}

/// Serde wire form: `origin` as `[f64; 3]` (glam `DVec3` isn't serde-derivable
/// without enabling glam's `serde` feature workspace-wide).
#[derive(Serialize, Deserialize)]
struct TerrainTileRaw {
    origin: [f64; 3],
    heights: Vec<f32>,
}

impl Serialize for TerrainTile {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        TerrainTileRaw {
            origin: self.origin.to_array(),
            heights: self.heights.clone(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for TerrainTile {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = TerrainTileRaw::deserialize(d)?;
        Ok(Self {
            origin: DVec3::from_array(raw.origin),
            heights: raw.heights,
        })
    }
}

impl TerrainTile {
    /// A flat tile at `origin` with every sample `0.0` (height == `origin.y`).
    pub fn flat(resolution: u32, origin: DVec3) -> Self {
        Self {
            origin,
            heights: vec![0.0; (resolution * resolution) as usize],
        }
    }

    /// Build a tile from an explicit height buffer (row-major, length
    /// `resolution²`). Returns `None` on a length mismatch.
    pub fn from_heights(resolution: u32, origin: DVec3, heights: Vec<f32>) -> Option<Self> {
        (heights.len() == (resolution * resolution) as usize).then_some(Self { origin, heights })
    }

    /// The raw row-major height buffer (`f32` offsets from `origin.y`).
    #[inline]
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Mutable access to the raw buffer (bulk edits — the P10.2 brush seam).
    #[inline]
    pub fn heights_mut(&mut self) -> &mut [f32] {
        &mut self.heights
    }

    /// The `f32` height offset at sample `(i, j)` (`0`-based, `i` = column/+X,
    /// `j` = row/+Z). Out-of-range indices clamp to the edge.
    #[inline]
    pub fn sample(&self, resolution: u32, i: u32, j: u32) -> f32 {
        let r = resolution.max(1);
        let i = i.min(r - 1);
        let j = j.min(r - 1);
        self.heights[(j * r + i) as usize]
    }

    /// The absolute world height of sample `(i, j)` (`origin.y + offset`).
    #[inline]
    pub fn world_height(&self, resolution: u32, i: u32, j: u32) -> f64 {
        self.origin.y + self.sample(resolution, i, j) as f64
    }

    /// Write the `f32` offset at sample `(i, j)`. Out-of-range indices are ignored.
    #[inline]
    pub fn set_sample(&mut self, resolution: u32, i: u32, j: u32, height: f32) {
        let r = resolution.max(1);
        if i < r && j < r {
            self.heights[(j * r + i) as usize] = height;
        }
    }

    /// Inclusive `(min, max)` of the tile's `f32` offsets (for AABB culling and
    /// height-texture range normalization). An empty buffer yields `(0, 0)`.
    pub fn height_bounds(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &h in &self.heights {
            lo = lo.min(h);
            hi = hi.max(h);
        }
        if lo.is_finite() {
            (lo, hi)
        } else {
            (0.0, 0.0)
        }
    }
}
