//! Tilemap CPU math (P8.1b): tile → world quad, atlas cell → UV rect, chunk
//! AABB, and view-frustum culling — all pure and unit-tested, mirroring how the
//! sprite quad/UV math lives here (not in the GPU pass).
//!
//! A tilemap is expanded **per occupied chunk** into a batch of [`SpriteInstance`]
//! quads: every tile of a tilemap shares one `(layer, order, texture)`, so a
//! whole chunk becomes a single [`crate::PrebatchedRun`] that bypasses the loose
//! sprite sort (naive per-tile loose sprites would swamp the batcher's sort at
//! 100k tiles). See [`crate::batch_scene`] for how runs merge with loose sprites.

use glam::{DVec3, Mat4, Vec2, Vec3};

use crate::{SpriteInstance, TextureHandle};

/// Side length (in tiles) of one square chunk. Must match
/// `inf_ecs::CHUNK_DIM` — the ECS side stores tiles in blocks of this stride and
/// the renderer reconstructs each tile's grid coordinate from `chunk * DIM + local`.
pub const TILE_CHUNK_DIM: i32 = 32;

/// The fixed, non-tile parameters shared by every tile of one tilemap. The host
/// fills this from the ECS `Tilemap` component (see `inf-viewport`'s
/// `project_tilemap`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TilemapParams {
    /// World position of the tilemap's grid origin — the bottom-left corner of
    /// tile `(0, 0)`. Tile `(gx, gy)` occupies world
    /// `[origin + (gx, gy)·tile_size, origin + (gx+1, gy+1)·tile_size]` in the XY
    /// plane at `origin.z`.
    pub origin: DVec3,
    /// World units per tile (width, height).
    pub tile_size: Vec2,
    /// Atlas grid dimensions in cells; a 1-based tile index `i` maps to atlas
    /// cell `i - 1` at `(col, row) = ((i-1) % cols, (i-1) / cols)`. Clamped to
    /// ≥ 1 at use so a degenerate atlas can't divide by zero.
    pub atlas_cols: u32,
    pub atlas_rows: u32,
    /// GPU texture handle for the atlas (`WHITE_TEXTURE` = the white fallback).
    pub texture: TextureHandle,
    /// Linear straight-alpha tint applied to every tile.
    pub color: [f32; 4],
    /// Draw order (shared by every tile of this tilemap).
    pub sorting_layer: i32,
    pub order: i32,
}

/// One occupied chunk's tile indices (row-major, `TILE_CHUNK_DIM²` entries;
/// `0` = empty). The host copies these out of the ECS `TileChunk` on projection.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderChunk {
    /// Chunk coordinate (grid tile `(x, y)` lives in chunk `(x.div_euclid(DIM),
    /// y.div_euclid(DIM))`).
    pub coord: (i32, i32),
    /// Row-major tile indices, length `TILE_CHUNK_DIM²`.
    pub tiles: Vec<u32>,
}

/// A projected tilemap ready for per-frame cull + expansion by the sprite pass.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderTilemap {
    pub params: TilemapParams,
    pub chunks: Vec<RenderChunk>,
}

/// The number of tiles a chunk's `tiles` slice must hold.
pub const CHUNK_TILES: usize = (TILE_CHUNK_DIM * TILE_CHUNK_DIM) as usize;

/// The atlas UV sub-rect of a 1-based tile `index` for the given atlas grid.
/// Cell `index - 1` is laid out row-major; `col`/`row` wrap the grid so an
/// out-of-range index samples a valid cell rather than off the atlas. Returns
/// `(uv_min, uv_max)`; row 0 is the atlas's top row (V grows downward).
pub fn atlas_uv(index: u32, atlas_cols: u32, atlas_rows: u32) -> (Vec2, Vec2) {
    let cols = atlas_cols.max(1);
    let rows = atlas_rows.max(1);
    let cell = index.saturating_sub(1);
    let col = cell % cols;
    let row = (cell / cols) % rows;
    let cw = 1.0 / cols as f32;
    let ch = 1.0 / rows as f32;
    let min = Vec2::new(col as f32 * cw, row as f32 * ch);
    (min, Vec2::new(min.x + cw, min.y + ch))
}

/// Global grid coordinate of local tile `(lx, ly)` in `coord`.
#[inline]
fn global_tile(coord: (i32, i32), lx: i32, ly: i32) -> (i32, i32) {
    (coord.0 * TILE_CHUNK_DIM + lx, coord.1 * TILE_CHUNK_DIM + ly)
}

/// The world-space AABB (`min`, `max`) covering every tile cell of a chunk,
/// occupied or not — the conservative bound used for culling.
pub fn chunk_world_aabb(params: &TilemapParams, coord: (i32, i32)) -> (DVec3, DVec3) {
    let ts = params.tile_size.as_dvec2();
    let (gx0, gy0) = global_tile(coord, 0, 0);
    let lo = params.origin + DVec3::new(gx0 as f64 * ts.x, gy0 as f64 * ts.y, 0.0);
    let hi = lo
        + DVec3::new(
            TILE_CHUNK_DIM as f64 * ts.x,
            TILE_CHUNK_DIM as f64 * ts.y,
            0.0,
        );
    (lo, hi)
}

/// Expand one chunk's occupied tiles into [`SpriteInstance`]s appended to `out`.
/// Empty tiles (index `0`) are skipped. Each tile becomes a bottom-left-pivoted
/// quad of `tile_size`, positioned so cell `(gx, gy)` fills `[gx, gx+1)·tile_size`
/// (in world units, offset by `params.origin`). Positions are world-space `f64`
/// (the sprite pass rebases against the floating origin at pack time, like loose
/// sprites). Emission order is row-major and deterministic.
pub fn expand_chunk(params: &TilemapParams, chunk: &RenderChunk, out: &mut Vec<SpriteInstance>) {
    let ts = params.tile_size;
    for ly in 0..TILE_CHUNK_DIM {
        for lx in 0..TILE_CHUNK_DIM {
            let i = (ly * TILE_CHUNK_DIM + lx) as usize;
            let index = match chunk.tiles.get(i) {
                Some(&v) if v != 0 => v,
                _ => continue,
            };
            let (gx, gy) = global_tile(chunk.coord, lx, ly);
            let position =
                params.origin + DVec3::new(gx as f64 * ts.x as f64, gy as f64 * ts.y as f64, 0.0);
            let (uv_min, uv_max) = atlas_uv(index, params.atlas_cols, params.atlas_rows);
            out.push(SpriteInstance {
                position,
                size: ts,
                pivot: Vec2::ZERO, // bottom-left: the cell fills [g, g+1)·tile_size
                rotation: 0.0,
                uv_min,
                uv_max,
                color: params.color,
                texture: params.texture,
                sorting_layer: params.sorting_layer,
                order: params.order,
                flip_x: false,
                flip_y: false,
            });
        }
    }
}

/// Conservative frustum test for a render-local AABB against `view_proj`.
///
/// Tests the four side clip planes: if all eight corners fall outside the same
/// plane (`x < -w`, `x > w`, `y < -w`, or `y > w`), the box is fully off-screen
/// and culled (returns `false`). Homogeneous-space comparisons keep this correct
/// for corners behind the camera (`w < 0`). Near/far are left to GPU clipping —
/// a cheap correct v1. Callers add a margin by expanding the AABB before calling.
pub fn aabb_visible(min: Vec3, max: Vec3, view_proj: &Mat4) -> bool {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let clip: [glam::Vec4; 8] = corners.map(|c| *view_proj * c.extend(1.0));
    let all = |f: fn(&glam::Vec4) -> bool| clip.iter().all(f);
    if all(|c| c.x < -c.w) || all(|c| c.x > c.w) || all(|c| c.y < -c.w) || all(|c| c.y > c.w) {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> TilemapParams {
        TilemapParams {
            origin: DVec3::ZERO,
            tile_size: Vec2::new(2.0, 2.0),
            atlas_cols: 2,
            atlas_rows: 2,
            texture: 0,
            color: [1.0; 4],
            sorting_layer: 0,
            order: 0,
        }
    }

    #[test]
    fn atlas_uv_one_based_row_major() {
        // 2×2 atlas: index 1 → top-left cell, index 4 → bottom-right cell.
        let (min1, max1) = atlas_uv(1, 2, 2);
        assert_eq!(min1, Vec2::new(0.0, 0.0));
        assert_eq!(max1, Vec2::new(0.5, 0.5));
        let (min2, _) = atlas_uv(2, 2, 2);
        assert_eq!(min2, Vec2::new(0.5, 0.0)); // col 1, row 0
        let (min3, _) = atlas_uv(3, 2, 2);
        assert_eq!(min3, Vec2::new(0.0, 0.5)); // col 0, row 1
        let (min4, max4) = atlas_uv(4, 2, 2);
        assert_eq!(min4, Vec2::new(0.5, 0.5));
        assert_eq!(max4, Vec2::new(1.0, 1.0));
    }

    #[test]
    fn atlas_uv_degenerate_atlas_is_safe() {
        // Zero dims clamp to 1 (whole texture), never divide by zero.
        let (min, max) = atlas_uv(1, 0, 0);
        assert_eq!(min, Vec2::ZERO);
        assert_eq!(max, Vec2::ONE);
    }

    #[test]
    fn expand_skips_empty_and_places_quads() {
        let mut tiles = vec![0u32; CHUNK_TILES];
        // local (0,0) = index 1, local (1,0) = index 4 (in chunk (0,0)).
        tiles[0] = 1;
        tiles[1] = 4;
        let chunk = RenderChunk {
            coord: (0, 0),
            tiles,
        };
        let mut out = Vec::new();
        expand_chunk(&params(), &chunk, &mut out);
        assert_eq!(out.len(), 2); // empties skipped

        // Row-major: tile (0,0) first, then (1,0).
        assert_eq!(out[0].position, DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(out[0].pivot, Vec2::ZERO);
        assert_eq!(out[0].size, Vec2::new(2.0, 2.0));
        assert_eq!(out[1].position, DVec3::new(2.0, 0.0, 0.0)); // gx=1 → x=2
                                                                // UVs come from the atlas cell.
        assert_eq!(out[0].uv_min, Vec2::new(0.0, 0.0));
        assert_eq!(out[1].uv_min, Vec2::new(0.5, 0.5)); // index 4 → cell (1,1)
    }

    #[test]
    fn expand_uses_chunk_offset_and_origin() {
        let mut p = params();
        p.origin = DVec3::new(10.0, 20.0, -1.0);
        // A single tile at local (0,0) of chunk (1, -1).
        let mut tiles = vec![0u32; CHUNK_TILES];
        tiles[0] = 1;
        let chunk = RenderChunk {
            coord: (1, -1),
            tiles,
        };
        let mut out = Vec::new();
        expand_chunk(&p, &chunk, &mut out);
        // gx = 1*32 = 32, gy = -1*32 = -32 → world = origin + (32*2, -32*2, 0).
        assert_eq!(out[0].position, DVec3::new(10.0 + 64.0, 20.0 - 64.0, -1.0));
    }

    #[test]
    fn chunk_aabb_covers_full_chunk_extent() {
        let (min, max) = chunk_world_aabb(&params(), (0, 0));
        assert_eq!(min, DVec3::ZERO);
        // 32 tiles × 2 world units = 64.
        assert_eq!(max, DVec3::new(64.0, 64.0, 0.0));
        let (min1, _) = chunk_world_aabb(&params(), (1, 0));
        assert_eq!(min1, DVec3::new(64.0, 0.0, 0.0));
    }

    fn front_view_proj() -> Mat4 {
        // Eye at +Z looking down -Z (like the golden front view). Uses the same
        // reverse-infinite-Z projection the renderer does; the homogeneous
        // clip-plane tests in `aabb_visible` are projection-agnostic regardless.
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(
            60f32.to_radians(),
            16.0 / 9.0,
            0.1,
        );
        let view =
            glam::camera::rh::view::look_to_mat4(Vec3::new(0.0, 0.0, 6.0), Vec3::NEG_Z, Vec3::Y);
        proj * view
    }

    #[test]
    fn cull_keeps_on_screen_rejects_off_screen() {
        let vp = front_view_proj();
        // A small box at the origin is on-screen.
        assert!(aabb_visible(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            &vp
        ));
        // A box far to the right (all corners past the right plane) is culled.
        assert!(!aabb_visible(
            Vec3::new(1000.0, -1.0, 0.0),
            Vec3::new(1002.0, 1.0, 0.0),
            &vp
        ));
        // Far below screen → culled.
        assert!(!aabb_visible(
            Vec3::new(-1.0, -1000.0, 0.0),
            Vec3::new(1.0, -998.0, 0.0),
            &vp
        ));
    }
}
