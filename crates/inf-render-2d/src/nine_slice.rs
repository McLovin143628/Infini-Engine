//! 9-slice sprite CPU expansion (P8.1c): a bordered panel whose four corners
//! keep a fixed world thickness while its four edges stretch along one axis and
//! its center stretches along both — the classic scalable UI-panel primitive.
//!
//! Like tilemaps, a 9-slice expands into a small batch of [`SpriteInstance`]
//! quads (exactly nine, one per cell) sharing a single `(layer, order, texture)`,
//! so the host wraps the nine quads in one [`crate::PrebatchedRun`]. The math is
//! pure and unit-tested here; the GPU pass just packs the resulting instances
//! against the floating origin, exactly like loose sprites and tiles.

use glam::{DVec3, Vec2};

use crate::{SpriteInstance, TextureHandle};

/// The fixed parameters of one 9-slice panel. The host fills this from the ECS
/// `NineSlice` component (see `inf-viewport`'s `project_nine_slice`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NineSliceParams {
    /// World-space anchor of the panel (the `pivot` point maps here).
    pub position: DVec3,
    /// Normalized anchor in `[0,1]²`; `(0.5, 0.5)` centers the panel on
    /// `position`, `(0,0)` puts its bottom-left corner there.
    pub pivot: Vec2,
    /// Total panel extent in world units (width, height).
    pub size: Vec2,
    /// Normalized border fractions of the **texture**: `[left, right, top,
    /// bottom]`. Splits the atlas into the classic 3×3 grid of source rects.
    pub border_uv: [f32; 4],
    /// World thickness of the borders: `x` = left/right column width, `y` =
    /// top/bottom row height. Clamped so the two opposite borders never exceed
    /// the panel extent (a degenerate center collapses to zero width/height).
    pub border_world: Vec2,
    /// Linear straight-alpha tint applied to every cell.
    pub color: [f32; 4],
    /// GPU texture handle (`WHITE_TEXTURE` = the white fallback).
    pub texture: TextureHandle,
    /// Draw order shared by all nine cells.
    pub sorting_layer: i32,
    pub order: i32,
}

impl Default for NineSliceParams {
    fn default() -> Self {
        Self {
            position: DVec3::ZERO,
            pivot: Vec2::splat(0.5),
            size: Vec2::splat(2.0),
            // A centered 1/3 border on every side by default.
            border_uv: [1.0 / 3.0; 4],
            border_world: Vec2::splat(0.5),
            color: [1.0; 4],
            texture: crate::WHITE_TEXTURE,
            sorting_layer: 0,
            order: 0,
        }
    }
}

/// Expand a 9-slice panel into its **nine** cell quads (row-major, bottom row
/// first): `[r*3 + c]` for row `r` (0 = bottom, 2 = top) and column `c` (0 =
/// left, 2 = right). Corners stay at `border_world` thickness; edges stretch
/// along one axis; the center fills the remainder.
///
/// **Degenerate cells are still emitted** (as zero-area quads) so the output is
/// always exactly nine instances at fixed indices — a zero-size quad rasterizes
/// no fragments, so this is harmless, and keeping the count fixed makes the
/// layout trivially testable. When `2·border_world` exceeds the panel size the
/// border thickness is clamped to half the size, collapsing the center (and, in
/// the limit, the opposite edges) to zero extent rather than overlapping.
pub fn expand_nine_slice(p: &NineSliceParams) -> [SpriteInstance; 9] {
    let w = p.size.x.max(0.0);
    let h = p.size.y.max(0.0);
    // Clamp each border to half the extent so left+right (top+bottom) never
    // overrun the panel; the center takes whatever is left (possibly zero).
    let bw = p.border_world.x.clamp(0.0, w * 0.5);
    let bh = p.border_world.y.clamp(0.0, h * 0.5);

    // Panel bottom-left corner in world space.
    let min_x = p.position.x - p.pivot.x as f64 * w as f64;
    let min_y = p.position.y - p.pivot.y as f64 * h as f64;
    let z = p.position.z;

    // Per-column: (x-offset from min_x, width, u_min, u_max).
    let [left, right, top, bottom] = p.border_uv;
    let center_w = (w - 2.0 * bw).max(0.0);
    let cols: [(f32, f32, f32, f32); 3] = [
        (0.0, bw, 0.0, left),              // left
        (bw, center_w, left, 1.0 - right), // center
        (w - bw, bw, 1.0 - right, 1.0),    // right
    ];
    // Per-row: (y-offset from min_y, height, v_min@top, v_max@bottom). Texture V
    // grows downward, so the top world row samples the smallest V.
    let center_h = (h - 2.0 * bh).max(0.0);
    let rows: [(f32, f32, f32, f32); 3] = [
        (0.0, bh, 1.0 - bottom, 1.0),      // bottom
        (bh, center_h, top, 1.0 - bottom), // center
        (h - bh, bh, 0.0, top),            // top
    ];

    let mut out = [SpriteInstance::default(); 9];
    for (r, &(y_off, cell_h, v_min, v_max)) in rows.iter().enumerate() {
        for (c, &(x_off, cell_w, u_min, u_max)) in cols.iter().enumerate() {
            out[r * 3 + c] = SpriteInstance {
                position: DVec3::new(min_x + x_off as f64, min_y + y_off as f64, z),
                size: Vec2::new(cell_w, cell_h),
                pivot: Vec2::ZERO, // bottom-left: each cell fills its rect exactly
                rotation: 0.0,
                uv_min: Vec2::new(u_min, v_min),
                uv_max: Vec2::new(u_max, v_max),
                color: p.color,
                texture: p.texture,
                sorting_layer: p.sorting_layer,
                order: p.order,
                flip_x: false,
                flip_y: false,
                billboard: crate::BILLBOARD_NONE,
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> NineSliceParams {
        NineSliceParams {
            position: DVec3::ZERO,
            pivot: Vec2::ZERO, // bottom-left anchored → panel spans [0,size]
            size: Vec2::new(4.0, 4.0),
            border_uv: [0.25, 0.25, 0.25, 0.25],
            border_world: Vec2::new(1.0, 1.0),
            color: [1.0; 4],
            texture: 0x99,
            sorting_layer: 2,
            order: 3,
        }
    }

    #[test]
    fn nine_cells_tile_the_panel_without_gaps() {
        let cells = expand_nine_slice(&params());
        assert_eq!(cells.len(), 9);
        // Bottom-left corner cell.
        let bl = cells[0];
        assert_eq!(bl.position, DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(bl.size, Vec2::new(1.0, 1.0));
        // Center cell (row 1, col 1) fills the 2×2 interior at (1,1).
        let center = cells[4] /* row 1, col 1 */;
        assert_eq!(center.position, DVec3::new(1.0, 1.0, 0.0));
        assert_eq!(center.size, Vec2::new(2.0, 2.0));
        // Top-right corner cell.
        let tr = cells[2 * 3 + 2];
        assert_eq!(tr.position, DVec3::new(3.0, 3.0, 0.0));
        assert_eq!(tr.size, Vec2::new(1.0, 1.0));
        // Shared params propagate.
        for cell in &cells {
            assert_eq!(cell.texture, 0x99);
            assert_eq!(cell.sorting_layer, 2);
            assert_eq!(cell.order, 3);
            assert_eq!(cell.pivot, Vec2::ZERO);
        }
    }

    #[test]
    fn corner_and_center_uv_rects() {
        let cells = expand_nine_slice(&params());
        // Bottom-left corner samples the atlas's bottom-left border rect:
        // u [0,0.25], and (V grows down) the bottom row is V [0.75,1].
        let bl = cells[0];
        assert_eq!(bl.uv_min, Vec2::new(0.0, 0.75));
        assert_eq!(bl.uv_max, Vec2::new(0.25, 1.0));
        // Top-left corner: u [0,0.25], top row V [0,0.25].
        let tl = cells[2 * 3];
        assert_eq!(tl.uv_min, Vec2::new(0.0, 0.0));
        assert_eq!(tl.uv_max, Vec2::new(0.25, 0.25));
        // Center: the interior atlas rect on both axes.
        let center = cells[4] /* row 1, col 1 */;
        assert_eq!(center.uv_min, Vec2::new(0.25, 0.25));
        assert_eq!(center.uv_max, Vec2::new(0.75, 0.75));
    }

    #[test]
    fn small_size_clamps_borders_and_collapses_center() {
        // size (1,1) with a 1-unit border on each side: 2·border > size, so the
        // border clamps to 0.5 and the center collapses to zero extent.
        let mut p = params();
        p.size = Vec2::new(1.0, 1.0);
        p.border_world = Vec2::new(1.0, 1.0);
        let cells = expand_nine_slice(&p);
        // Corners are each half the panel.
        assert_eq!(cells[0].size, Vec2::new(0.5, 0.5));
        // Center cell has zero area (degenerate, still present).
        let center = cells[4] /* row 1, col 1 */;
        assert_eq!(center.size, Vec2::new(0.0, 0.0));
        // Right column starts at x = w - bw = 0.5, meeting the left column edge.
        let right = cells[2];
        assert_eq!(right.position.x, 0.5);
        assert_eq!(right.size.x, 0.5);
        // Still exactly nine instances.
        assert_eq!(cells.len(), 9);
    }

    #[test]
    fn pivot_center_places_panel_symmetrically() {
        let mut p = params();
        p.pivot = Vec2::splat(0.5);
        p.position = DVec3::new(10.0, 20.0, -1.0);
        // Panel spans [10-2, 10+2] × [20-2, 20+2].
        let cells = expand_nine_slice(&p);
        assert_eq!(cells[0].position, DVec3::new(8.0, 18.0, -1.0));
        let tr = cells[2 * 3 + 2];
        assert_eq!(tr.position, DVec3::new(11.0, 21.0, -1.0));
    }
}
