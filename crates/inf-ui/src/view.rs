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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menu::MenuState;

    fn vp() -> Vec2 {
        Vec2::new(1920.0, 1080.0)
    }

    /// **A closed menu draws NOTHING.** This is what the 54 frozen goldens rest
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
