//! **Drawing** the settings dialog, the toasts and the world-space interaction
//! prompt — one pure function each, over the same projections the reducers read.

use glam::Vec2;
use inf_input::InputMap;

use crate::draw::{measure, Align, Color, Rect, UiDrawList, GLYPH_PX};
use crate::menu::{self, MenuState, Page, RowKind};
use crate::settings::GameSettings;
use crate::toast::Toasts;

/// The palette, in linear straight alpha. Four colours and two greys: a settings
/// dialog is a table, and a table that needs a colour ramp is a table with too
/// much in it.
pub mod palette {
    use crate::draw::Color;
    /// The dim behind the dialog.
    pub const SCRIM: Color = [0.0, 0.0, 0.0, 0.62];
    /// The dialog's body.
    pub const PANEL: Color = [0.055, 0.06, 0.075, 0.96];
    /// Its border, and the tab strip's underline.
    pub const EDGE: Color = [0.35, 0.62, 0.95, 0.9];
    /// The focused row's fill.
    pub const FOCUS: Color = [0.16, 0.30, 0.48, 0.95];
    /// Ordinary text.
    pub const TEXT: Color = [0.88, 0.90, 0.94, 1.0];
    /// A note under a row, and an inactive tab.
    pub const MUTED: Color = [0.58, 0.61, 0.68, 1.0];
    /// A warning — a conflict, and a control with no consumer.
    pub const WARN: Color = [0.98, 0.74, 0.28, 1.0];
    /// A toast's body.
    pub const TOAST: Color = [0.09, 0.10, 0.13, 0.92];
}

/// How wide the dialog is, as a fraction of the viewport, and its bounds in
/// pixels. A menu that filled a 4K screen would be unreadable and one pinned to
/// a pixel width would be unreadable at 720p, so it scales and then clamps.
const DIALOG_FRACTION: f32 = 0.62;
const DIALOG_MIN_PX: f32 = 560.0;
const DIALOG_MAX_PX: f32 = 1100.0;
const ROW_H: f32 = 30.0;
const PAD: f32 = 18.0;

/// The text scale for a given viewport height — **integral**, because the
/// built-in font is an 8 × 8 bitmap and a fractional scale samples between
/// texels, which on a 1-pixel stem is the difference between a letter and a
/// smear.
pub fn text_scale(viewport_h: f32) -> f32 {
    if !viewport_h.is_finite() {
        return 2.0;
    }
    (viewport_h / 540.0).floor().clamp(1.0, 4.0)
}

/// How many rows fit in `viewport` — the bindings page is longer than any
/// screen, so the view scrolls it around the focused row.
fn visible_rows(viewport: Vec2) -> usize {
    let usable = (viewport.y * 0.82 - PAD * 6.0).max(ROW_H);
    ((usable / ROW_H) as usize).max(3)
}

/// The window of rows to draw so that `focus` is inside it, and the list never
/// scrolls past its own end.
fn window(total: usize, focus: usize, capacity: usize) -> (usize, usize) {
    if total <= capacity {
        return (0, total);
    }
    let half = capacity / 2;
    let start = focus.saturating_sub(half).min(total - capacity);
    (start, start + capacity)
}

/// **Draw the settings dialog into `list`.** Pure: the same state always draws
/// the same quads.
pub fn menu(list: &mut UiDrawList, state: &MenuState, s: &GameSettings, map: &InputMap) {
    if !state.open {
        return;
    }
    let vp = list.viewport;
    if !vp.x.is_finite() || !vp.y.is_finite() || vp.x <= 0.0 || vp.y <= 0.0 {
        return;
    }
    let scale = text_scale(vp.y);
    let rows = menu::rows(state, s, map);
    let capacity = visible_rows(vp);
    let (from, to) = window(rows.len(), state.focus, capacity);
    let shown = &rows[from..to];

    let w = (vp.x * DIALOG_FRACTION).clamp(DIALOG_MIN_PX.min(vp.x), DIALOG_MAX_PX);
    let header = ROW_H * 2.0 + PAD;
    let body = shown.len() as f32 * ROW_H;
    let footer = ROW_H;
    let h = (header + body + footer + PAD * 2.0).min(vp.y);
    let panel = Rect::new(
        ((vp.x - w) * 0.5).max(0.0),
        ((vp.y - h) * 0.5).max(0.0),
        w,
        h,
    );

    // The scrim first: it is what makes the game behind the dialog read as
    // *behind* it rather than as a texture the panel happens to sit on.
    list.rect(Rect::new(0.0, 0.0, vp.x, vp.y), palette::SCRIM);
    list.rect(panel, palette::PANEL);
    list.stroke(panel, 2.0, palette::EDGE);

    // ── the tab strip ──
    let mut x = panel.x + PAD;
    let tab_y = panel.y + PAD;
    for page in Page::ALL {
        let label = page.label();
        let tw = measure(label, scale);
        let colour = if page == state.page {
            palette::TEXT
        } else {
            palette::MUTED
        };
        list.text(x, tab_y, label, scale, colour);
        if page == state.page {
            list.rect(
                Rect::new(x, tab_y + GLYPH_PX * scale + 3.0, tw, 2.0),
                palette::EDGE,
            );
        }
        x += tw + GLYPH_PX * scale * 2.0;
    }

    // ── the rows ──
    let mut y = panel.y + header;
    for (i, row) in shown.iter().enumerate() {
        let index = from + i;
        let r = Rect::new(panel.x + PAD * 0.5, y, panel.w - PAD, ROW_H);
        if index == state.focus {
            list.rect(r, palette::FOCUS);
        }
        list.text_in(r, PAD, Align::Left, &row.label, scale, palette::TEXT);
        let value_colour = match row.kind {
            RowKind::Capture if row.value == "--" => palette::MUTED,
            _ => palette::TEXT,
        };
        list.text_in(r, PAD, Align::Right, &row.value, scale, value_colour);
        if let Some(note) = &row.note {
            // The note rides under the row's own band, at the small scale, so a
            // row with one is not taller than a row without one — a table whose
            // line height depended on its content would jump as the focus moved.
            let small = (scale - 1.0).max(1.0);
            list.text(
                r.x + PAD,
                r.bottom() - GLYPH_PX * small - 2.0,
                note,
                small,
                palette::WARN,
            );
        }
        y += ROW_H;
    }

    // ── the footer: the keys, because a keyboard-only dialog has to say so ──
    let hint = if matches!(state.capture, menu::Capture::Idle) {
        "[arrows] move  [enter] pick  [tab/esc] close"
    } else {
        "[press a key]  [esc] cancel"
    };
    list.text_in(
        Rect::new(panel.x, panel.bottom() - ROW_H - PAD * 0.5, panel.w, ROW_H),
        PAD,
        Align::Center,
        hint,
        (scale - 1.0).max(1.0),
        palette::MUTED,
    );
}

/// **Draw the toast stack** in the top-right corner, newest at the top.
pub fn toasts(list: &mut UiDrawList, toasts: &Toasts) {
    let vp = list.viewport;
    if !vp.x.is_finite() || !vp.y.is_finite() {
        return;
    }
    let scale = text_scale(vp.y);
    let mut y = PAD;
    for t in toasts.live() {
        let w = measure(&t.text, scale) + PAD * 2.0;
        let h = GLYPH_PX * scale + PAD;
        let r = Rect::new(vp.x - w - PAD, y, w, h);
        // The fade is in the ALPHA rather than in the colour, so a toast over a
        // bright sky and one over a dark wall fade the same way.
        let a = t.alpha();
        let mut body = palette::TOAST;
        body[3] *= a;
        let mut text = palette::WARN;
        text[3] *= a;
        list.rect(r, body);
        list.text_in(r, PAD, Align::Center, &t.text, scale, text);
        y += h + 6.0;
    }
}

/// **Draw a world-space interaction prompt** at a screen position.
///
/// `at` is where the projection put the target, in pixels. The prompt is drawn
/// *above* it by its own height, so the label sits over the thing rather than on
/// it — and it is clamped into the viewport, because a target at the edge of the
/// screen is still a target the player can reach.
pub fn prompt(list: &mut UiDrawList, at: Vec2, text: &str, colour: Color) {
    let vp = list.viewport;
    if !at.x.is_finite() || !at.y.is_finite() || text.is_empty() {
        return;
    }
    let scale = text_scale(vp.y);
    let w = measure(text, scale) + PAD;
    let h = GLYPH_PX * scale + 10.0;
    let x = (at.x - w * 0.5).clamp(0.0, (vp.x - w).max(0.0));
    let y = (at.y - h * 1.5).clamp(0.0, (vp.y - h).max(0.0));
    let r = Rect::new(x, y, w, h);
    list.rect(r, palette::TOAST);
    list.stroke(r, 1.0, palette::EDGE);
    list.text_in(r, PAD * 0.5, Align::Center, text, scale, colour);
}

/// How long each arm of the reticle is, in pixels at scale 1.
const RETICLE_ARM_PX: f32 = 7.0;
/// How wide the gap at the centre is, in pixels at scale 1 — the thing that
/// makes it a reticle rather than a plus sign, because what a player is aiming
/// at has to be visible through it.
const RETICLE_GAP_PX: f32 = 4.0;
/// How thick each arm is, in pixels at scale 1.
const RETICLE_THICK_PX: f32 = 1.0;

/// **Draw the aiming reticle** at the centre of the viewport (wave WPN1).
///
/// Four arms and a hole, in the same integral [`text_scale`] the rest of this
/// module uses — for its reason one shape along: an arm 1.5 pixels thick is
/// resolved by the rasterizer into two arms of different brightness, and a
/// crosshair that is brighter on one side than the other reads as a mis-aligned
/// crosshair.
///
/// It is drawn **only while aiming**, which is the caller's decision and not
/// this function's: a reticle on screen at all times is a claim that the
/// character is always pointing a weapon, and in this engine a carried rifle
/// hangs where the animation puts it (see
/// `inf_physics::d3::gameplay::aim_hold_point`). The one thing the reticle has
/// to be true about is that the shot goes through it, and that is only true on
/// the aim line.
pub fn reticle(list: &mut UiDrawList, colour: Color) {
    let vp = list.viewport;
    if !vp.x.is_finite() || !vp.y.is_finite() || vp.x <= 0.0 || vp.y <= 0.0 {
        return;
    }
    let s = text_scale(vp.y);
    let (arm, gap, t) = (RETICLE_ARM_PX * s, RETICLE_GAP_PX * s, RETICLE_THICK_PX * s);
    // The centre is rounded to a whole pixel, so the two horizontal arms are the
    // same length as each other on an odd-width viewport.
    let (cx, cy) = ((vp.x * 0.5).round(), (vp.y * 0.5).round());
    for r in [
        Rect::new(cx - gap - arm, cy - t * 0.5, arm, t),
        Rect::new(cx + gap, cy - t * 0.5, arm, t),
        Rect::new(cx - t * 0.5, cy - gap - arm, t, arm),
        Rect::new(cx - t * 0.5, cy + gap, t, arm),
    ] {
        list.rect(r, colour);
    }
}

/// **The wanted rating's own colour** — a warning, because that is what it is.
///
/// `palette::WARN` rather than a new entry: the palette's own note for it is
/// *"a warning — a conflict, and a control with no consumer"*, and a wanted
/// level is the same kind of statement about the world. An earned star is this
/// at full alpha and an unearned one is this at a sixth of it, so the scale is
/// always readable and the number always is too.
pub const WANTED_DIM: f32 = 0.16;

/// How wide one star is, in pixels at scale 1.
const STAR_PX: f32 = 11.0;

/// The gap between two of them, in pixels at scale 1.
const STAR_GAP_PX: f32 = 4.0;

/// How far the row sits from the top-left corner, in pixels at scale 1.
const STAR_MARGIN_PX: f32 = 10.0;

/// **A five-pointed star, in horizontal spans over an 11 x 9 grid.**
///
/// The draw list has exactly three primitives — a rect, a stroke and a run of
/// text — so a star is a raster. Each entry is `(row, x0, x1)` in grid cells,
/// and a row with two entries is the gap between the legs, which is the one
/// feature that stops a five-pointed star reading as a diamond.
const STAR_SPANS: [(u8, u8, u8); 11] = [
    (0, 5, 6),
    (1, 5, 6),
    (2, 4, 7),
    (3, 0, 11),
    (4, 1, 10),
    (5, 2, 9),
    (6, 2, 9),
    (7, 1, 5),
    (7, 6, 10),
    (8, 0, 4),
    (8, 7, 11),
];

/// The grid the spans above are expressed in.
const STAR_COLS: f32 = 11.0;
const STAR_ROWS: f32 = 9.0;

/// **DRAW THE WANTED RATING** (wave EMS3) — `earned` of `slots` stars, at the
/// top-left of the viewport.
///
/// # A new anchor, and why it is not the readout slot
///
/// The readout slot at bottom-centre is already contested: `PlayerApp::frame`
/// puts the driver's instruments and the ammunition count there behind an
/// `else`, with a stated rule that *"two readouts stacked in one place would be
/// a HUD that draws over itself on the one frame both are true"*. A wanted
/// rating is true **at the same time as both of them** — the whole mechanic is
/// being chased while driving, and being chased while shooting — so it cannot
/// share that slot without breaking the rule that governs it.
///
/// The top-right is the toast stack's (`toasts` anchors there and grows
/// downward), and putting a permanent element under a stack that can grow is
/// the same collision one corner over. So: **top-left**, which nothing else in
/// this engine draws in.
///
/// # It draws the empty slots too
///
/// An earned star is [`palette::WARN`] and an unearned one is the same colour at
/// [`WANTED_DIM`] alpha, so the row is always five wide. A rating that only drew
/// what was earned would make "one star" and "one star of five" look identical,
/// and the second is the fact a player is deciding whether to run on.
///
/// Nothing is drawn at all when `slots` is zero, which is the honest answer for
/// a game that has no wanted system in it.
pub fn wanted(list: &mut UiDrawList, earned: u8, slots: u8) {
    let vp = list.viewport;
    if !vp.x.is_finite() || !vp.y.is_finite() || vp.x <= 0.0 || vp.y <= 0.0 || slots == 0 {
        return;
    }
    let s = text_scale(vp.y);
    let (w, gap, margin) = (STAR_PX * s, STAR_GAP_PX * s, STAR_MARGIN_PX * s);
    let cell_x = w / STAR_COLS;
    let cell_y = (STAR_PX * s * STAR_ROWS / STAR_COLS) / STAR_ROWS;
    for i in 0..slots {
        let lit = i < earned;
        let mut colour = palette::WARN;
        if !lit {
            colour[3] = WANTED_DIM;
        }
        // Rounded to whole pixels, for the reticle's reason: an 11-cell raster
        // whose cells land between texels is a star with one thick arm.
        let x0 = (margin + f32::from(i) * (w + gap)).round();
        let y0 = margin.round();
        for (row, a, b) in STAR_SPANS {
            let x = x0 + f32::from(a) * cell_x;
            let y = y0 + f32::from(row) * cell_y;
            list.rect(Rect::new(x, y, f32::from(b - a) * cell_x, cell_y), colour);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menu::MenuState;

    fn vp() -> Vec2 {
        Vec2::new(1920.0, 1080.0)
    }

    /// **A closed menu draws NOTHING.** This is what the frozen goldens rest
    /// on: the UI node is a no-op on an empty list, and the list is empty on
    /// every frame nobody opened a menu on — which is every golden scene.
    #[test]
    fn a_closed_menu_draws_nothing_at_all() {
        let mut list = UiDrawList::new(vp());
        let st = MenuState::new();
        menu(
            &mut list,
            &st,
            &GameSettings::default(),
            &inf_input::default_map(),
        );
        toasts(&mut list, &Toasts::default());
        assert!(
            list.is_empty(),
            "{} quads on a closed menu",
            list.quads.len()
        );

        // The control: an open one draws a great deal, so the arm above is not
        // merely testing that the drawing is broken.
        let mut open = st.clone();
        open.set_open(true);
        menu(
            &mut list,
            &open,
            &GameSettings::default(),
            &inf_input::default_map(),
        );
        assert!(list.quads.len() > 50, "only {} quads", list.quads.len());
    }

    /// The drawing is a **pure function of the state**: two builds of the same
    /// state are identical, and a different state is not.
    #[test]
    fn the_drawing_is_a_pure_function_of_what_it_is_given() {
        let s = GameSettings::default();
        let m = inf_input::default_map();
        let mut st = MenuState::new();
        st.set_open(true);
        let build = |st: &MenuState| {
            let mut l = UiDrawList::new(vp());
            menu(&mut l, st, &s, &m);
            l
        };
        assert_eq!(build(&st), build(&st));
        let mut moved = st.clone();
        moved.focus = 2;
        assert_ne!(
            build(&st),
            build(&moved),
            "moving the focus drew the same thing"
        );
        let mut paged = st.clone();
        paged.page = Page::Bindings;
        assert_ne!(build(&st), build(&paged));
    }

    /// **Every row of the page is on the screen**, at any viewport the engine
    /// ships a resolution for — which the bindings page, at eighteen rows plus a
    /// button, is the case that fails first.
    #[test]
    fn the_focused_row_is_always_inside_the_window() {
        let mut st = MenuState::new();
        st.set_open(true);
        st.page = Page::Bindings;
        let s = GameSettings::default();
        let m = inf_input::default_map();
        let total = menu::rows(&st, &s, &m).len();
        for (w, h) in crate::settings::RESOLUTIONS {
            let cap = visible_rows(Vec2::new(w as f32, h as f32));
            for focus in 0..total {
                let (from, to) = window(total, focus, cap);
                assert!(
                    (from..to).contains(&focus),
                    "at {w}x{h} the focus {focus} of {total} is outside the window {from}..{to}"
                );
                assert!(to <= total, "the window ran past the end of the list");
            }
        }
    }

    /// A hostile viewport draws nothing rather than a quad with no position.
    #[test]
    fn a_hostile_viewport_draws_nothing() {
        let mut st = MenuState::new();
        st.set_open(true);
        for bad in [
            Vec2::new(f32::NAN, 1080.0),
            Vec2::new(1920.0, f32::NAN),
            Vec2::new(0.0, 1080.0),
            Vec2::new(1920.0, -1.0),
        ] {
            let mut l = UiDrawList::new(bad);
            menu(
                &mut l,
                &st,
                &GameSettings::default(),
                &inf_input::default_map(),
            );
            assert!(
                l.is_empty(),
                "viewport {bad:?} drew {} quads",
                l.quads.len()
            );
        }
    }

    /// A prompt stays on the screen even when its target is off the edge.
    #[test]
    fn a_prompt_is_clamped_into_the_viewport() {
        for at in [
            Vec2::new(-500.0, 500.0),
            Vec2::new(5000.0, 500.0),
            Vec2::new(500.0, -900.0),
            Vec2::new(500.0, 5000.0),
        ] {
            let mut l = UiDrawList::new(vp());
            prompt(&mut l, at, "[E] Enter vehicle", palette::TEXT);
            assert!(!l.is_empty());
            for q in &l.quads {
                let (x, y) = (q.position.x as f32, q.position.y as f32);
                assert!(
                    x >= -1.0 && y >= -1.0 && x <= vp().x + 1.0 && y <= vp().y + 1.0,
                    "a prompt quad landed at ({x}, {y}) for a target at {at:?}"
                );
            }
        }
        // A NaN target and an empty label draw nothing.
        let mut l = UiDrawList::new(vp());
        prompt(&mut l, Vec2::new(f32::NAN, 0.0), "x", palette::TEXT);
        prompt(&mut l, Vec2::new(1.0, 1.0), "", palette::TEXT);
        assert!(l.is_empty());
    }

    /// **The reticle is four arms around a HOLE**, centred, and it draws nothing
    /// on a viewport that does not exist (wave WPN1).
    ///
    /// The hole is the arm this is really for: a crosshair whose arms met would
    /// cover the thing the player is aiming at, which is the one pixel the whole
    /// element exists to point at. Measured as a gap the centre pixel is not
    /// inside any quad of.
    #[test]
    fn the_reticle_is_centred_and_leaves_its_own_middle_clear() {
        let mut l = UiDrawList::new(vp());
        reticle(&mut l, palette::TEXT);
        assert_eq!(l.quads.len(), 4, "a reticle is four arms");
        let (cx, cy) = (vp().x * 0.5, vp().y * 0.5);
        let mut left = 0;
        let mut right = 0;
        let mut above = 0;
        let mut below = 0;
        for q in &l.quads {
            let (x, y) = (q.position.x as f32, q.position.y as f32);
            let (w, h) = (q.size.x, q.size.y);
            // Nothing covers the middle.
            assert!(
                !(x <= cx && cx <= x + w && y <= cy && cy <= y + h),
                "an arm covers the centre pixel: ({x}, {y}) {w} x {h}"
            );
            if x + w < cx {
                left += 1;
            }
            if x > cx {
                right += 1;
            }
            if y + h < cy {
                above += 1;
            }
            if y > cy {
                below += 1;
            }
        }
        assert_eq!(
            (left, right, above, below),
            (1, 1, 1, 1),
            "the four arms are not one per side — a reticle that lost an arm is \
             an aiming aid that points off to one side"
        );
        // …and a viewport that is not a viewport draws nothing at all, which is
        // `menu`'s own rule and is what keeps a headless host from pushing a
        // quad at NaN.
        for bad in [
            Vec2::new(f32::NAN, 1080.0),
            Vec2::new(1920.0, 0.0),
            Vec2::new(-1.0, -1.0),
        ] {
            let mut l = UiDrawList::new(bad);
            reticle(&mut l, palette::TEXT);
            assert!(l.is_empty(), "viewport {bad:?} drew a reticle");
        }
    }

    /// **THE WANTED RATING IS ALWAYS FIVE SLOTS WIDE, IN THE TOP-LEFT**, and
    /// the earned ones are the bright ones (wave EMS3).
    ///
    /// The three claims a lazier drawer would fail: it draws the *empty* slots
    /// (so one star and one-of-five do not look identical), it draws them in
    /// the corner nothing else in this engine draws in (so it cannot collide
    /// with the toast stack or the readout slot), and the brightness order is
    /// earned-then-dim rather than the reverse.
    #[test]
    fn the_wanted_rating_draws_five_slots_and_lights_the_earned_ones() {
        for earned in 0..=5u8 {
            let mut l = UiDrawList::new(vp());
            wanted(&mut l, earned, 5);
            let per_star = STAR_SPANS.len();
            assert_eq!(
                l.quads.len(),
                per_star * 5,
                "a {earned}-star rating did not draw five slots"
            );
            let lit = l
                .quads
                .iter()
                .filter(|q| q.color[3] > WANTED_DIM + 1e-6)
                .count();
            assert_eq!(
                lit,
                per_star * usize::from(earned),
                "{earned} stars lit {lit} of {} spans",
                l.quads.len()
            );
            // The BRIGHT ones come first, left to right: a rating whose lit
            // slots were on the right would read as a different number.
            let mut lit_x: Vec<f32> = Vec::new();
            let mut dim_x: Vec<f32> = Vec::new();
            for q in &l.quads {
                let x = q.position.x as f32;
                if q.color[3] > WANTED_DIM + 1e-6 {
                    lit_x.push(x);
                } else {
                    dim_x.push(x);
                }
                // TOP-LEFT, and well clear of the two corners that are taken:
                // the toast stack anchors at `vp.x - w - PAD` and the readout
                // sits at `vp.y`.
                assert!(x < vp().x * 0.25, "a star drifted out of the top-left");
                assert!((q.position.y as f32) < vp().y * 0.25);
            }
            if let (Some(l_max), Some(d_min)) = (
                lit_x
                    .iter()
                    .cloned()
                    .fold(None::<f32>, |a, b| Some(a.map_or(b, |a: f32| a.max(b)))),
                dim_x
                    .iter()
                    .cloned()
                    .fold(None::<f32>, |a, b| Some(a.map_or(b, |a: f32| a.min(b)))),
            ) {
                assert!(
                    l_max < d_min,
                    "an earned star was drawn to the right of an empty one"
                );
            }
        }
        // A star is a STAR and not a diamond: two rows have a gap in them, which
        // is what the legs are.
        let rows: Vec<u8> = STAR_SPANS.iter().map(|(r, _, _)| *r).collect();
        let split = rows
            .iter()
            .filter(|r| rows.iter().filter(|o| o == r).count() > 1);
        assert!(
            split.count() >= 4,
            "no row of the raster has two spans — the star lost its legs"
        );
        // Nothing at all with no slots, and nothing on a viewport that is not
        // one — `reticle`'s own rule.
        let mut l = UiDrawList::new(vp());
        wanted(&mut l, 3, 0);
        assert!(l.is_empty(), "a rating with no slots drew something");
        for bad in [
            Vec2::new(f32::NAN, 1080.0),
            Vec2::new(1920.0, 0.0),
            Vec2::new(-1.0, -1.0),
        ] {
            let mut l = UiDrawList::new(bad);
            wanted(&mut l, 3, 5);
            assert!(l.is_empty(), "viewport {bad:?} drew a rating");
        }
    }

    /// The text scale is integral at every resolution the video page offers —
    /// a fractional one samples between texels of an 8 x 8 bitmap font.
    #[test]
    fn the_text_scale_is_integral_at_every_shipped_resolution() {
        for (_, h) in crate::settings::RESOLUTIONS {
            let s = text_scale(h as f32);
            assert_eq!(s, s.floor(), "{h} gave a fractional scale of {s}");
            assert!((1.0..=4.0).contains(&s), "{h} gave {s}");
        }
        assert_eq!(text_scale(f32::NAN), 2.0);
    }
}
