//! [`HeightRegion`] — a dense `f32` snapshot of an AABB of the heightfield,
//! flattened across tile boundaries into one continuous grid.
//!
//! ## Why this exists
//!
//! Some sculpt operations are **neighbourhood** operations, not per-sample ones:
//! Smooth blends each sample toward the mean of its neighbours, and Flatten toward
//! the mean of the affected area. Running those directly on the paged
//! [`TerrainData`] would have to special-case tile seams on every neighbour fetch.
//! Instead we extract the affected area (plus a margin) into one dense grid where
//! a tile boundary is just an interior column — so smoothing reads *across* tile
//! edges seamlessly by construction — do the read-modify-write in that grid, then
//! [`write_back`](HeightRegion::write_back) the changes as a [`HeightDelta`].
//!
//! ## The CPU/GPU seam (P10.3)
//!
//! This is deliberately the same shape a GPU pass wants: **P10.3's hydraulic
//! erosion** will [`extract_region`](TerrainData::extract_region) an AABB, upload
//! [`heights`](HeightRegion::heights) to a storage texture, run the erosion
//! compute pipeline, read the result back into the same grid, and
//! [`write_back`](HeightRegion::write_back) — reusing this exact extract/apply seam
//! as the CPU reference. Smooth/Flatten are the CPU-side first customers.
//!
//! ## Precision note
//!
//! Region heights are `f32` offsets from a single `f64` [`anchor_y`] (the
//! `origin.y` of the tile at the region centre). This preserves the tile-local
//! `f32`/`f64`-anchor precision doctrine as long as the tiles spanned share an
//! anchor — which they always do in practice (authoring, import and
//! `get_or_create` all leave `origin.y == 0`, the height living entirely in the
//! `f32` offset). Round-trip is exact for that common case.
//!
//! [`anchor_y`]: HeightRegion::anchor_y

use glam::DVec2;

use crate::data::TerrainData;
use crate::delta::{DeltaBuilder, HeightDelta};

/// A dense `nx × nz` grid of `f32` heights sampled on the terrain's lattice,
/// spanning an AABB across however many tiles it overlaps.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightRegion {
    /// Global sample index (world `x = gx * mps`) of column 0.
    gx0: i64,
    /// Global sample index (world `z = gz * mps`) of row 0.
    gz0: i64,
    nx: u32,
    nz: u32,
    mps: f64,
    /// Terrain tile resolution (for mapping a global index → tile + local sample).
    res: u32,
    /// `f64` anchor the `f32` heights are offsets from (see module docs).
    anchor_y: f64,
    /// Row-major `nx · nz` `f32` heights (offsets from `anchor_y`).
    heights: Vec<f32>,
    /// Row-major authored mask: `false` where the sampled lattice point has no
    /// authored tile (a hole). Holes are never modified or written back.
    authored: Vec<bool>,
}

/// Map a global sample index to its owning tile index and local sample (in
/// `0..=res-2` — the shared far edge belongs to the next tile's local 0).
#[inline]
fn split_index(g: i64, res: u32) -> (i32, u32) {
    let res_1 = (res - 1) as i64;
    let t = g.div_euclid(res_1);
    let local = (g - t * res_1) as u32;
    (t as i32, local)
}

impl HeightRegion {
    /// Grid dimensions `(nx, nz)`.
    #[inline]
    pub fn dims(&self) -> (u32, u32) {
        (self.nx, self.nz)
    }

    /// The `f64` anchor the stored `f32` heights are offsets from.
    #[inline]
    pub fn anchor_y(&self) -> f64 {
        self.anchor_y
    }

    #[inline]
    fn idx(&self, c: u32, r: u32) -> usize {
        (r * self.nx + c) as usize
    }

    /// World XZ of grid cell `(c, r)`.
    #[inline]
    pub fn world_xz(&self, c: u32, r: u32) -> DVec2 {
        DVec2::new(
            (self.gx0 + c as i64) as f64 * self.mps,
            (self.gz0 + r as i64) as f64 * self.mps,
        )
    }

    /// Height offset at `(c, r)` (offset from [`anchor_y`](Self::anchor_y)).
    #[inline]
    pub fn height(&self, c: u32, r: u32) -> f32 {
        self.heights[self.idx(c, r)]
    }

    /// Overwrite the height offset at `(c, r)`.
    #[inline]
    pub fn set_height(&mut self, c: u32, r: u32, v: f32) {
        let i = self.idx(c, r);
        self.heights[i] = v;
    }

    /// Whether `(c, r)` maps to an authored tile (not a hole).
    #[inline]
    pub fn is_authored(&self, c: u32, r: u32) -> bool {
        self.authored[self.idx(c, r)]
    }

    /// The raw row-major height buffer (offsets from the anchor) — the GPU upload
    /// source for P10.3 erosion.
    #[inline]
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Mutable raw height buffer — the GPU read-back destination for P10.3.
    #[inline]
    pub fn heights_mut(&mut self) -> &mut [f32] {
        &mut self.heights
    }

    /// Write the region's current heights back into `data`, recording every
    /// changed sample into `builder`. Only authored cells are written, and each is
    /// written into **every** tile that shares it (so a sample on a tile seam stays
    /// identical in both tiles). Never creates a tile — this is why Smooth/Flatten
    /// never author outside existing space.
    pub(crate) fn write_back(&self, data: &mut TerrainData, builder: &mut DeltaBuilder) {
        let res = self.res;
        for r in 0..self.nz {
            for c in 0..self.nx {
                let idx = self.idx(c, r);
                if !self.authored[idx] {
                    continue;
                }
                let new_world = self.anchor_y + self.heights[idx] as f64;
                let gx = self.gx0 + c as i64;
                let gz = self.gz0 + r as i64;
                let (tx, i) = split_index(gx, res);
                let (tz, j) = split_index(gz, res);
                // Copies: the primary tile, plus the shared-edge neighbour(s) when
                // this sample sits on tile local 0.
                let xs: [(i32, u32); 2] = [(tx, i), (tx - 1, res - 1)];
                let zs: [(i32, u32); 2] = [(tz, j), (tz - 1, res - 1)];
                let nx_copies = if i == 0 { 2 } else { 1 };
                let nz_copies = if j == 0 { 2 } else { 1 };
                for &(ctx, ci) in xs.iter().take(nx_copies) {
                    for &(ctz, cj) in zs.iter().take(nz_copies) {
                        let coord = (ctx, ctz);
                        if let Some(tile) = data.get_tile_mut(coord) {
                            let new_off = (new_world - tile.origin.y) as f32;
                            let cur = tile.sample(res, ci, cj);
                            if new_off != cur {
                                builder.touch(coord, ci, cj, cur);
                                tile.set_sample(res, ci, cj, new_off);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl TerrainData {
    /// Extract a dense [`HeightRegion`] covering the world AABB `[min, max]`,
    /// expanded by `margin` lattice samples on every side (neighbourhood ops pass
    /// a margin so edge samples have valid neighbours to read). Lattice points with
    /// no authored tile are marked unauthored (holes).
    pub fn extract_region(&self, min: DVec2, max: DVec2, margin: u32) -> HeightRegion {
        let (min, max) = (min.min(max), min.max(max));
        let mps = self.meters_per_sample();
        let res = self.tile_resolution();
        let m = margin as i64;
        let gx0 = (min.x / mps).floor() as i64 - m;
        let gx1 = (max.x / mps).ceil() as i64 + m;
        let gz0 = (min.y / mps).floor() as i64 - m;
        let gz1 = (max.y / mps).ceil() as i64 + m;
        let nx = (gx1 - gx0 + 1).max(1) as u32;
        let nz = (gz1 - gz0 + 1).max(1) as u32;

        // Anchor on the tile at the region centre (else 0 — a flat empty region).
        let center = (min + max) * 0.5;
        let anchor_y = self
            .get_tile(self.tile_coord_of(center.x, center.y))
            .map(|t| t.origin.y)
            .unwrap_or(0.0);

        let mut heights = vec![0.0f32; (nx * nz) as usize];
        let mut authored = vec![false; (nx * nz) as usize];
        for r in 0..nz {
            let gz = gz0 + r as i64;
            let (tz, j) = split_index(gz, res);
            for c in 0..nx {
                let gx = gx0 + c as i64;
                let (tx, i) = split_index(gx, res);
                if let Some(tile) = self.get_tile((tx, tz)) {
                    let idx = (r * nx + c) as usize;
                    heights[idx] = (tile.world_height(res, i, j) - anchor_y) as f32;
                    authored[idx] = true;
                }
            }
        }
        HeightRegion {
            gx0,
            gz0,
            nx,
            nz,
            mps,
            res,
            anchor_y,
            heights,
            authored,
        }
    }

    /// Convenience: [`extract_region`](Self::extract_region) with no margin, apply
    /// a closure to the grid, then write the result back as a [`HeightDelta`]. The
    /// closure receives the mutable region to read-modify-write.
    pub fn edit_region<F: FnOnce(&mut HeightRegion)>(
        &mut self,
        min: DVec2,
        max: DVec2,
        margin: u32,
        edit: F,
    ) -> HeightDelta {
        let mut region = self.extract_region(min, max, margin);
        edit(&mut region);
        let mut builder = DeltaBuilder::default();
        region.write_back(self, &mut builder);
        builder.finalize(self)
    }
}
