//! Bitmap-text CPU expansion (P8.1c): lay a string out as one monospace quad
//! per glyph, sampling a fixed-grid ASCII bitmap-font atlas. Newlines start a
//! new line; out-of-range codepoints are skipped (but still consume a column so
//! alignment stays stable); a per-line horizontal alignment offset centers or
//! right-justifies the row.
//!
//! Ships a **built-in 8×8 font** ([`builtin_font_rgba8`]) so text renders with
//! no asset at all — the renderer uploads it under a reserved texture handle and
//! `Text2D::font_texture = None` selects it.
//!
//! Bitmap data: **font8x8** by Daniel Hepper — public domain
//! (<https://github.com/dhepper/font8x8>), originally by Marcel Sondaar. The
//! basic-Latin table (ASCII 0x20..0x7F) is embedded below verbatim; each glyph
//! is eight rows top-to-bottom, each row a byte whose bit 0 (0x01) is the
//! leftmost pixel.

use glam::{DVec3, Vec2};

use crate::{SpriteInstance, TextureHandle};

/// Horizontal alignment of each text line about the anchor position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// The fixed parameters of one text block. The host fills this from the ECS
/// `Text2D` component (see `inf-viewport`'s `project_text`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextParams<'a> {
    /// World anchor: the top-left of the text block (Left align), the top-center
    /// (Center), or the top-right (Right). Lines grow downward from here.
    pub position: DVec3,
    /// The string to lay out (`'\n'` breaks lines).
    pub text: &'a str,
    /// Font-atlas grid dimensions in glyph cells.
    pub glyph_cols: u32,
    pub glyph_rows: u32,
    /// Codepoint of atlas cell 0 (usually 32, i.e. space).
    pub first_codepoint: u32,
    /// World size of one glyph cell (width, height).
    pub glyph_size: Vec2,
    /// Extra horizontal advance as a fraction of the glyph width (0 = tight
    /// monospace; positive spreads glyphs apart).
    pub tracking: f32,
    /// Linear straight-alpha tint multiplied with every glyph texel.
    pub color: [f32; 4],
    /// GPU texture handle of the font atlas.
    pub texture: TextureHandle,
    pub sorting_layer: i32,
    pub order: i32,
    pub halign: HAlign,
}

/// The atlas UV sub-rect of 0-based glyph `cell` in a `cols×rows` grid; row 0 is
/// the atlas top row (V grows downward). Cells wrap the grid so an out-of-grid
/// cell samples a valid rect rather than off the atlas.
fn glyph_uv(cell: u32, cols: u32, rows: u32) -> (Vec2, Vec2) {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let col = cell % cols;
    let row = (cell / cols) % rows;
    let cw = 1.0 / cols as f32;
    let ch = 1.0 / rows as f32;
    let min = Vec2::new(col as f32 * cw, row as f32 * ch);
    (min, Vec2::new(min.x + cw, min.y + ch))
}

/// Lay `params.text` out into one bottom-left-pivoted [`SpriteInstance`] per
/// in-range glyph, in reading order (line by line, left to right). Emission
/// order is deterministic; positions are world-space `f64` (packed against the
/// floating origin by the sprite pass, like every other 2D primitive).
///
/// Newlines break lines; a codepoint below `first_codepoint` or past the atlas
/// grid (`cols·rows` cells) is skipped — it draws nothing but still advances one
/// monospace column, so the rest of the line keeps its positions.
pub fn expand_text(params: &TextParams) -> Vec<SpriteInstance> {
    let gw = params.glyph_size.x;
    let gh = params.glyph_size.y;
    let advance = gw * (1.0 + params.tracking);
    let cols = params.glyph_cols.max(1);
    let rows = params.glyph_rows.max(1);
    let cell_count = cols * rows;

    let mut out = Vec::new();
    for (line_index, line) in params.text.split('\n').enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let n = chars.len();
        // Visual line width: last glyph is `gw` wide, the rest are `advance`.
        let line_w = if n > 0 {
            (n as f32 - 1.0) * advance + gw
        } else {
            0.0
        };
        let x_off = match params.halign {
            HAlign::Left => 0.0,
            HAlign::Center => -line_w * 0.5,
            HAlign::Right => -line_w,
        };
        // Each line sits just below the previous one; line 0's top is at
        // `position.y`, so its bottom edge is `position.y - gh`.
        let y_bottom = params.position.y - (line_index as f64 + 1.0) * gh as f64;

        for (col_index, ch) in chars.iter().enumerate() {
            let cp = *ch as u32;
            if cp < params.first_codepoint {
                continue; // control chars etc. — skip, but the column is consumed
            }
            let cell = cp - params.first_codepoint;
            if cell >= cell_count {
                continue; // past the atlas grid — skip
            }
            let (uv_min, uv_max) = glyph_uv(cell, cols, rows);
            let x_left = params.position.x + x_off as f64 + col_index as f64 * advance as f64;
            out.push(SpriteInstance {
                position: DVec3::new(x_left, y_bottom, params.position.z),
                size: Vec2::new(gw, gh),
                pivot: Vec2::ZERO,
                rotation: 0.0,
                uv_min,
                uv_max,
                color: params.color,
                texture: params.texture,
                sorting_layer: params.sorting_layer,
                order: params.order,
                flip_x: false,
                flip_y: false,
                billboard: crate::BILLBOARD_NONE,
            });
        }
    }
    out
}

/// Atlas grid of the built-in font: 16 columns × 6 rows = 96 cells covering
/// ASCII 32..127.
pub const BUILTIN_FONT_COLS: u32 = 16;
pub const BUILTIN_FONT_ROWS: u32 = 6;
/// Codepoint of the built-in font's cell 0 (space).
pub const BUILTIN_FONT_FIRST_CP: u32 = 32;

/// Render the built-in 8×8 font (font8x8, public domain) into a straight
/// RGBA8 atlas: 16×6 glyph cells, each 8×8 px → 128×48 px. Lit pixels are opaque
/// white; the background is fully transparent, so the tint colours the glyphs
/// and the quad edges blend cleanly. Returns `(rgba8, width, height)`.
pub fn builtin_font_rgba8() -> (Vec<u8>, u32, u32) {
    let gw = 8u32;
    let gh = 8u32;
    let width = BUILTIN_FONT_COLS * gw;
    let height = BUILTIN_FONT_ROWS * gh;
    let mut px = vec![0u8; (width * height * 4) as usize];
    for (glyph, rows) in FONT8X8.iter().enumerate() {
        let g = glyph as u32;
        let cx = (g % BUILTIN_FONT_COLS) * gw;
        let cy = (g / BUILTIN_FONT_COLS) * gh;
        for (ry, &byte) in rows.iter().enumerate() {
            for bx in 0..8u32 {
                if (byte >> bx) & 1 == 1 {
                    let x = cx + bx;
                    let y = cy + ry as u32;
                    let i = ((y * width + x) * 4) as usize;
                    px[i] = 255;
                    px[i + 1] = 255;
                    px[i + 2] = 255;
                    px[i + 3] = 255;
                }
            }
        }
    }
    (px, width, height)
}

/// font8x8 basic-Latin bitmaps, ASCII 0x20..0x7F (96 glyphs). Public domain
/// (font8x8 by Daniel Hepper / Marcel Sondaar). Row 0 is the top; bit 0 (0x01)
/// is the leftmost pixel of each row.
#[rustfmt::skip]
const FONT8X8: [[u8; 8]; 96] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0020 (space)
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00], // U+0021 (!)
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0022 (")
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00], // U+0023 (#)
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00], // U+0024 ($)
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00], // U+0025 (%)
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00], // U+0026 (&)
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0027 (')
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00], // U+0028 (()
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00], // U+0029 ())
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // U+002A (*)
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00], // U+002B (+)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06], // U+002C (,)
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00], // U+002D (-)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00], // U+002E (.)
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00], // U+002F (/)
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00], // U+0030 (0)
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00], // U+0031 (1)
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00], // U+0032 (2)
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00], // U+0033 (3)
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00], // U+0034 (4)
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00], // U+0035 (5)
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00], // U+0036 (6)
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00], // U+0037 (7)
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00], // U+0038 (8)
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00], // U+0039 (9)
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00], // U+003A (:)
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06], // U+003B (;)
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00], // U+003C (<)
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00], // U+003D (=)
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00], // U+003E (>)
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00], // U+003F (?)
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00], // U+0040 (@)
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00], // U+0041 (A)
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00], // U+0042 (B)
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00], // U+0043 (C)
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00], // U+0044 (D)
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00], // U+0045 (E)
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00], // U+0046 (F)
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00], // U+0047 (G)
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00], // U+0048 (H)
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // U+0049 (I)
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00], // U+004A (J)
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00], // U+004B (K)
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00], // U+004C (L)
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00], // U+004D (M)
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00], // U+004E (N)
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00], // U+004F (O)
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00], // U+0050 (P)
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00], // U+0051 (Q)
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00], // U+0052 (R)
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00], // U+0053 (S)
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // U+0054 (T)
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00], // U+0055 (U)
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // U+0056 (V)
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00], // U+0057 (W)
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00], // U+0058 (X)
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00], // U+0059 (Y)
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00], // U+005A (Z)
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00], // U+005B ([)
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00], // U+005C (\)
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00], // U+005D (])
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00], // U+005E (^)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], // U+005F (_)
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00], // U+0060 (`)
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00], // U+0061 (a)
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00], // U+0062 (b)
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00], // U+0063 (c)
    [0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00], // U+0064 (d)
    [0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00], // U+0065 (e)
    [0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00], // U+0066 (f)
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F], // U+0067 (g)
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00], // U+0068 (h)
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // U+0069 (i)
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E], // U+006A (j)
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00], // U+006B (k)
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // U+006C (l)
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00], // U+006D (m)
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00], // U+006E (n)
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00], // U+006F (o)
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F], // U+0070 (p)
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78], // U+0071 (q)
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00], // U+0072 (r)
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00], // U+0073 (s)
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00], // U+0074 (t)
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00], // U+0075 (u)
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // U+0076 (v)
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00], // U+0077 (w)
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00], // U+0078 (x)
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F], // U+0079 (y)
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00], // U+007A (z)
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00], // U+007B ({)
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // U+007C (|)
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00], // U+007D (})
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+007E (~)
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // U+007F (del, blank)
];

#[cfg(test)]
mod tests {
    use super::*;

    fn params<'a>(text: &'a str, halign: HAlign) -> TextParams<'a> {
        TextParams {
            position: DVec3::ZERO,
            text,
            glyph_cols: BUILTIN_FONT_COLS,
            glyph_rows: BUILTIN_FONT_ROWS,
            first_codepoint: BUILTIN_FONT_FIRST_CP,
            glyph_size: Vec2::new(1.0, 2.0),
            tracking: 0.0,
            color: [1.0; 4],
            texture: 0x55,
            sorting_layer: 0,
            order: 0,
            halign,
        }
    }

    #[test]
    fn monospace_advance_and_positions() {
        let out = expand_text(&params("AB", HAlign::Left));
        assert_eq!(out.len(), 2);
        // Left-aligned: glyph 0 left edge at x=0, glyph 1 at x=advance=gw=1.
        assert_eq!(out[0].position.x, 0.0);
        assert_eq!(out[1].position.x, 1.0);
        // Both on line 0: bottom edge at position.y - gh = -2.
        assert_eq!(out[0].position.y, -2.0);
        assert_eq!(out[0].size, Vec2::new(1.0, 2.0));
    }

    #[test]
    fn tracking_widens_advance() {
        let mut p = params("AB", HAlign::Left);
        p.tracking = 0.5; // advance = gw * 1.5 = 1.5
        let out = expand_text(&p);
        assert_eq!(out[1].position.x, 1.5);
    }

    #[test]
    fn newlines_start_new_lines() {
        let out = expand_text(&params("A\nBC", HAlign::Left));
        assert_eq!(out.len(), 3);
        // Line 0 ('A') bottom at -gh = -2; line 1 ('BC') bottom at -2*gh = -4.
        assert_eq!(out[0].position.y, -2.0);
        assert_eq!(out[1].position.y, -4.0);
        assert_eq!(out[2].position.y, -4.0);
        // 'B' at column 0, 'C' at column 1 of line 1.
        assert_eq!(out[1].position.x, 0.0);
        assert_eq!(out[2].position.x, 1.0);
    }

    #[test]
    fn center_align_offsets_line_by_half_width() {
        // "AB": line width = (2-1)*advance + gw = 1 + 1 = 2 → centered offset -1.
        let out = expand_text(&params("AB", HAlign::Center));
        assert_eq!(out[0].position.x, -1.0);
        assert_eq!(out[1].position.x, 0.0);
    }

    #[test]
    fn right_align_ends_at_anchor() {
        // "AB": width 2, right offset -2 → last glyph right edge at anchor x=0.
        let out = expand_text(&params("AB", HAlign::Right));
        assert_eq!(out[0].position.x, -2.0);
        assert_eq!(out[1].position.x, -1.0); // right edge = -1 + gw(1) = 0
    }

    #[test]
    fn codepoint_maps_to_grid_cell_uv() {
        // 'A' = 65, first_cp 32 → cell 33 → col 33%16 = 1, row 33/16 = 2.
        let out = expand_text(&params("A", HAlign::Left));
        let cw = 1.0 / BUILTIN_FONT_COLS as f32;
        let ch = 1.0 / BUILTIN_FONT_ROWS as f32;
        assert_eq!(out[0].uv_min, Vec2::new(cw, 2.0 * ch));
        assert_eq!(out[0].uv_max, Vec2::new(2.0 * cw, 3.0 * ch));
    }

    #[test]
    fn out_of_range_codepoints_skipped_but_column_consumed() {
        // 'é' (233) is past the 96-cell grid → skipped; a control char (\t=9) is
        // below first_codepoint → skipped. Both still consume a column, so the
        // trailing 'X' keeps its position.
        let out = expand_text(&params("é\tX", HAlign::Left));
        assert_eq!(out.len(), 1); // only 'X' drew
                                  // 'X' is the third character (column index 2) → x = 2 * advance = 2.
        assert_eq!(out[0].position.x, 2.0);
    }

    #[test]
    fn builtin_font_atlas_dimensions_and_content() {
        let (rgba, w, h) = builtin_font_rgba8();
        assert_eq!((w, h), (128, 48));
        assert_eq!(rgba.len(), (128 * 48 * 4) as usize);
        // Space (cell 0) is entirely transparent.
        let mut space_lit = false;
        for y in 0..8 {
            for x in 0..8 {
                if rgba[((y * w + x) * 4 + 3) as usize] != 0 {
                    space_lit = true;
                }
            }
        }
        assert!(!space_lit, "space glyph should be blank");
        // 'A' (cell 33) has at least one lit (opaque white) pixel.
        let ax = (33 % BUILTIN_FONT_COLS) * 8;
        let ay = (33 / BUILTIN_FONT_COLS) * 8;
        let mut a_lit = 0;
        for y in ay..ay + 8 {
            for x in ax..ax + 8 {
                let i = ((y * w + x) * 4) as usize;
                if rgba[i + 3] == 255 {
                    assert_eq!([rgba[i], rgba[i + 1], rgba[i + 2]], [255, 255, 255]);
                    a_lit += 1;
                }
            }
        }
        assert!(a_lit > 0, "'A' glyph should have lit pixels");
    }
}
