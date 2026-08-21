//! Screen-space rectangles, text and the draw list they land in.
//!
//! # The coordinate space, and why it is its own
//!
//! Everything here is in **virtual pixels with the origin at the top left and
//! `+y` down** — the space every UI in the world is laid out in, and the space
//! `inf-render-2d` deliberately is not: a [`SpriteInstance`] is a *world* quad in
//! the XY plane going through the game camera, depth-tested against the scene and
//! tonemapped with it. A menu drawn that way would be bloomed, would drift when
//! the camera moved, and would be behind the wall the player is standing next to.
//!
//! So the instance *shape* is reused (one vertex layout, one batcher, one font
//! atlas) and the *interpretation* is this module's: `position` is the rect's
//! top-left corner in pixels, `size` is its extent in pixels, the pivot is always
//! zero, and the node that draws it (`inf_render::passes::ui`) maps pixels onto
//! the clip cube after the tonemap.

use glam::{DVec3, Vec2};
use inf_render_2d::{SpriteInstance, TextureHandle, BUILTIN_FONT_TEXTURE, WHITE_TEXTURE};

/// The built-in font's cell, in atlas pixels. The atlas is 16 × 6 cells of 8 × 8.
pub const GLYPH_PX: f32 = 8.0;

/// A screen-space rectangle in virtual pixels, `+y` down.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width; a non-positive one draws nothing.
    pub w: f32,
    /// Height; a non-positive one draws nothing.
    pub h: f32,
}

impl Rect {
    /// A rect from its corner and extent.
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// The right edge.
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// The bottom edge.
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Shrunk by `by` on every side. A rect smaller than twice the inset
    /// collapses to zero extent rather than turning inside out — a negative
    /// width is a quad the shader would wind backwards.
    pub fn inset(&self, by: f32) -> Self {
        Self {
            x: self.x + by,
            y: self.y + by,
            w: (self.w - by * 2.0).max(0.0),
            h: (self.h - by * 2.0).max(0.0),
        }
    }

    /// Whether `p` (pixels) is inside — half-open on the far edges, so two
    /// stacked rows never both claim the pixel on their shared line.
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x && p.x < self.right() && p.y >= self.y && p.y < self.bottom()
    }

    /// Whether every field is finite. A NaN reaches the vertex shader as a quad
    /// with no position, and the whole draw list behind it becomes undefined.
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.w.is_finite() && self.h.is_finite()
    }
}

/// Straight-alpha linear RGBA, the same convention the sprite path uses.
pub type Color = [f32; 4];

/// Where a run of text sits horizontally within the box it is given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Align {
    /// Against the left edge.
    #[default]
    Left,
    /// Centred.
    Center,
    /// Against the right edge.
    Right,
}

/// **The width of `text` at `scale`, in pixels.**
///
/// The measurement the 2D pipeline never exposed: `inf_render_2d::expand_text`
/// computes a line width inside its own body and hands back only quads, so
/// nothing outside it could lay a row out around a label. Every glyph is one
/// cell wide because the built-in font is a monospace bitmap — this is exact
/// rather than approximate, and it is exact *for the font that will draw it*,
/// which is the only kind of measurement worth having.
///
/// Multi-line text measures its widest line. A non-finite scale measures zero,
/// because a NaN width propagates into every rect derived from it.
pub fn measure(text: &str, scale: f32) -> f32 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0.0;
    }
    let widest = text.lines().map(|l| l.chars().count()).max().unwrap_or(0);
    widest as f32 * GLYPH_PX * scale
}

/// The height of `text` at `scale`, in pixels — one cell per line, no leading.
pub fn measure_height(text: &str, scale: f32) -> f32 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0.0;
    }
    text.lines().count().max(1) as f32 * GLYPH_PX * scale
}

/// **The engine's in-game draw list**: screen-space quads in painter order.
///
/// A plain `Vec`, walked in the order it was pushed. There is no sorting layer
/// and no batching key because there does not need to be one: a menu is drawn
/// back to front by the code that builds it, and the only two textures it can
/// name are the 1 × 1 white fallback and the built-in font atlas. The node that
/// consumes it coalesces runs of equal texture, which on a real menu is two
/// draws.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UiDrawList {
    /// The quads, in painter order.
    pub quads: Vec<SpriteInstance>,
    /// The virtual-pixel extent this list was laid out for. The node needs it to
    /// map pixels onto clip space, and carrying it *with* the list is what stops
    /// a list built for one resolution being drawn at another — the two are one
    /// value or they are two answers to "how big is the screen".
    pub viewport: Vec2,
}

impl UiDrawList {
    /// An empty list laid out for `viewport` pixels.
    pub fn new(viewport: Vec2) -> Self {
        Self {
            quads: Vec::new(),
            viewport,
        }
    }

    /// Whether anything would be drawn. **The whole UI stack is inert on a level
    /// that opens no menu**, and this is the question the render node asks first
    /// — see its own arm for why the goldens depend on the answer.
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    /// Drop every quad, keeping the buffer and the viewport. Called once a
    /// frame by a host that rebuilds its UI from state — which is every host,
    /// because the UI is a pure function of the menu's state.
    pub fn clear(&mut self) {
        self.quads.clear();
    }

    /// Fill `rect` with `color`.
    ///
    /// A rect with a non-positive or non-finite extent pushes **nothing**: a
    /// zero-area quad is invisible and costs a vertex fetch, and a NaN one is a
    /// primitive with no position at all.
    pub fn rect(&mut self, rect: Rect, color: Color) {
        if !rect.is_finite() || rect.w <= 0.0 || rect.h <= 0.0 {
            return;
        }
        if !color.iter().all(|c| c.is_finite()) {
            return;
        }
        self.quads
            .push(quad(rect, color, WHITE_TEXTURE, Vec2::ZERO, Vec2::ONE));
    }

    /// Stroke `rect`'s outline `width` pixels thick, inside its own bounds.
    ///
    /// Four fills rather than a shader: a border is two horizontal bars and two
    /// vertical ones, the vertical pair inset by the horizontal pair's height so
    /// the corners are covered exactly once. Overdrawing the corners would be
    /// visible at any alpha below one.
    pub fn stroke(&mut self, rect: Rect, width: f32, color: Color) {
        if !rect.is_finite() || !width.is_finite() || width <= 0.0 {
            return;
        }
        let w = width.min(rect.w * 0.5).min(rect.h * 0.5);
        self.rect(Rect::new(rect.x, rect.y, rect.w, w), color);
        self.rect(Rect::new(rect.x, rect.bottom() - w, rect.w, w), color);
        self.rect(Rect::new(rect.x, rect.y + w, w, rect.h - w * 2.0), color);
        self.rect(
            Rect::new(rect.right() - w, rect.y + w, w, rect.h - w * 2.0),
            color,
        );
    }

    /// Draw `text` with its top-left at `(x, y)`, one quad per printable glyph.
    ///
    /// Returns the advanced pen position's `x`, so a caller can chain runs
    /// without measuring twice. Characters outside the atlas
    /// (`BUILTIN_FONT_FIRST_CP ..` for `COLS × ROWS` cells) draw nothing and
    /// **still advance**, which is what keeps a line of mixed text aligned with
    /// its own measurement.
    pub fn text(&mut self, x: f32, y: f32, text: &str, scale: f32, color: Color) -> f32 {
        if !scale.is_finite() || scale <= 0.0 || !x.is_finite() || !y.is_finite() {
            return x;
        }
        if !color.iter().all(|c| c.is_finite()) {
            return x;
        }
        let cell = GLYPH_PX * scale;
        let cols = inf_render_2d::BUILTIN_FONT_COLS;
        let rows = inf_render_2d::BUILTIN_FONT_ROWS;
        let first = inf_render_2d::BUILTIN_FONT_FIRST_CP;
        let (mut pen_x, mut pen_y) = (x, y);
        for ch in text.chars() {
            if ch == '\n' {
                pen_x = x;
                pen_y += cell;
                continue;
            }
            let code = ch as u32;
            if code >= first && code < first + cols * rows {
                let i = code - first;
                let (cx, cy) = (i % cols, i / cols);
                let uv_min = Vec2::new(cx as f32 / cols as f32, cy as f32 / rows as f32);
                let uv_max =
                    Vec2::new((cx + 1) as f32 / cols as f32, (cy + 1) as f32 / rows as f32);
                self.quads.push(quad(
                    Rect::new(pen_x, pen_y, cell, cell),
                    color,
                    BUILTIN_FONT_TEXTURE,
                    uv_min,
                    uv_max,
                ));
            }
            pen_x += cell;
        }
        pen_x
    }

    /// [`text`](Self::text), aligned inside `within`'s horizontal span and
    /// vertically centred in it — the shape every row in a menu wants.
    pub fn text_in(&mut self, within: Rect, pad: f32, align: Align, s: &str, scale: f32, c: Color) {
        let w = measure(s, scale);
        let x = match align {
            Align::Left => within.x + pad,
            Align::Center => within.x + (within.w - w) * 0.5,
            Align::Right => within.right() - pad - w,
        };
        let y = within.y + (within.h - measure_height(s, scale)) * 0.5;
        self.text(x, y, s, scale, c);
    }
}

/// One screen-space quad, in the [`SpriteInstance`] shape the node's vertex
/// layout already speaks.
fn quad(
    rect: Rect,
    color: Color,
    texture: TextureHandle,
    uv_min: Vec2,
    uv_max: Vec2,
) -> SpriteInstance {
    SpriteInstance {
        // Pixels, not metres: see the module header.
        position: DVec3::new(rect.x as f64, rect.y as f64, 0.0),
        size: Vec2::new(rect.w, rect.h),
        // Always zero: the rect IS the top-left corner, so the quad spans
        // `[x, x + w] x [y, y + h]` going down and right with no pivot algebra.
        pivot: Vec2::ZERO,
        rotation: 0.0,
        uv_min,
        uv_max,
        color,
        texture,
        sorting_layer: 0,
        order: 0,
        flip_x: false,
        flip_y: false,
        billboard: inf_render_2d::BILLBOARD_NONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_measurement_is_exact_for_the_font_that_draws_it() {
        // Five glyphs at scale 2 is 5 x 8 x 2.
        assert_eq!(measure("hello", 2.0), 80.0);
        assert_eq!(measure_height("hello", 2.0), 16.0);
        // Multi-line measures the WIDEST line and stacks the height.
        assert_eq!(measure("ab\ncdef", 1.0), 32.0);
        assert_eq!(measure_height("ab\ncdef", 1.0), 16.0);
        // Empty text is one line high and no pixels wide, so a row built around
        // an empty label still has a row's height.
        assert_eq!(measure("", 1.0), 0.0);
        assert_eq!(measure_height("", 1.0), 8.0);
        // …and the measurement agrees with what the emitter actually advances.
        let mut list = UiDrawList::new(Vec2::new(1920.0, 1080.0));
        let end = list.text(10.0, 0.0, "hello", 2.0, [1.0; 4]);
        assert_eq!(
            end - 10.0,
            measure("hello", 2.0),
            "the pen and the ruler disagree, so every row laid out with `measure` is wrong by the difference"
        );
        assert_eq!(list.quads.len(), 5);
        for bad in [f32::NAN, 0.0, -1.0, f32::INFINITY] {
            assert_eq!(measure("hello", bad), 0.0, "scale {bad}");
        }
    }

    /// **An unprintable character draws nothing and still advances.** Without
    /// the advance, a line containing one turns into a line one cell shorter
    /// than the ruler said, and every alignment built on it drifts.
    #[test]
    fn an_out_of_atlas_glyph_holds_its_column() {
        let mut list = UiDrawList::new(Vec2::new(800.0, 600.0));
        let end = list.text(0.0, 0.0, "a\u{2603}b", 1.0, [1.0; 4]);
        assert_eq!(
            list.quads.len(),
            2,
            "the snowman is not in a 96-glyph atlas"
        );
        assert_eq!(end, 24.0, "and it must still hold its column");
        assert_eq!(end, measure("a\u{2603}b", 1.0));
    }

    /// A newline returns the pen to the run's own left edge, not to zero.
    #[test]
    fn a_newline_returns_to_the_runs_own_left_edge() {
        let mut list = UiDrawList::new(Vec2::new(800.0, 600.0));
        list.text(100.0, 50.0, "a\nb", 1.0, [1.0; 4]);
        assert_eq!(list.quads.len(), 2);
        assert_eq!(list.quads[0].position.x, 100.0);
        assert_eq!(list.quads[1].position.x, 100.0);
        assert_eq!(list.quads[1].position.y, 58.0, "one cell down");
    }

    /// **Hostile geometry pushes nothing.** A NaN rect is a primitive with no
    /// position, and everything batched behind it in the same draw is undefined.
    #[test]
    fn a_hostile_rect_or_colour_draws_nothing() {
        let mut list = UiDrawList::new(Vec2::new(800.0, 600.0));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            list.rect(Rect::new(bad, 0.0, 10.0, 10.0), [1.0; 4]);
            list.rect(Rect::new(0.0, 0.0, bad, 10.0), [1.0; 4]);
            list.rect(Rect::new(0.0, 0.0, 10.0, 10.0), [bad, 1.0, 1.0, 1.0]);
            list.text(bad, 0.0, "x", 1.0, [1.0; 4]);
        }
        // …and so does a zero-area one.
        list.rect(Rect::new(0.0, 0.0, 0.0, 10.0), [1.0; 4]);
        list.rect(Rect::new(0.0, 0.0, 10.0, -1.0), [1.0; 4]);
        assert!(
            list.is_empty(),
            "{} hostile quads were pushed",
            list.quads.len()
        );

        // The control: a good rect does draw, so the arm is not merely refusing
        // everything.
        list.rect(Rect::new(1.0, 2.0, 3.0, 4.0), [1.0; 4]);
        assert_eq!(list.quads.len(), 1);
    }

    /// A stroke is four bars that tile the border and **do not overlap at the
    /// corners** — overdraw is visible at any alpha below one.
    #[test]
    fn a_stroke_tiles_its_border_without_overdrawing_the_corners() {
        let mut list = UiDrawList::new(Vec2::new(800.0, 600.0));
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        list.stroke(r, 2.0, [1.0; 4]);
        assert_eq!(list.quads.len(), 4);
        let area: f32 = list.quads.iter().map(|q| q.size.x * q.size.y).sum();
        // Perimeter x width, minus nothing: 2*(100*2) + 2*((50-4)*2) = 584.
        assert_eq!(
            area, 584.0,
            "the bars overlap (or leave a gap) at the corners"
        );
        // The inset is real: a stroke wider than half the rect fills it rather
        // than turning the vertical bars inside out.
        let mut list = UiDrawList::new(Vec2::new(800.0, 600.0));
        list.stroke(Rect::new(0.0, 0.0, 4.0, 4.0), 50.0, [1.0; 4]);
        assert!(list
            .quads
            .iter()
            .all(|q| q.size.x >= 0.0 && q.size.y >= 0.0));
    }

    /// The three alignments put the run where they say, measured against the
    /// ruler rather than against a hand-computed number.
    #[test]
    fn the_alignments_place_a_run_where_they_say() {
        let row = Rect::new(100.0, 0.0, 400.0, 20.0);
        let label = "abcd";
        let w = measure(label, 1.0);
        let first_x = |align| {
            let mut list = UiDrawList::new(Vec2::new(800.0, 600.0));
            list.text_in(row, 8.0, align, label, 1.0, [1.0; 4]);
            list.quads[0].position.x as f32
        };
        assert_eq!(first_x(Align::Left), 108.0);
        assert_eq!(first_x(Align::Right), 500.0 - 8.0 - w);
        assert_eq!(first_x(Align::Center), 100.0 + (400.0 - w) * 0.5);
    }
}
