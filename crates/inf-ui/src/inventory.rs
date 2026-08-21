//! **The inventory panel** (island wave I6) — the `I` key's surface.
//!
//! The same **state / reducer / projection** triple the settings dialog is built
//! from ([`crate::menu`]), for the same reason: everything here is a pure
//! function of its inputs, which is what lets a panel exist in a shipped player
//! in a repository whose CI has no GPU.
//!
//! # It does NOT pause the simulation, and that is a ruling
//!
//! The settings dialog pauses because a menu that did not would make the UI part
//! of the simulation's input — the frames a player spends reading a table would
//! be frames the sim advanced. An inventory is the other case: opening your bag
//! in the middle of a firefight is a thing that happens *in* the world, every
//! shipped game of this kind agrees, and a panel that froze the sim would make
//! the bag a safe place to stand. So `I` opens a HUD, not a modal.
//!
//! What it *does* take is the **navigation keys**, while it is open, so moving
//! the grid cursor does not also walk the character.
//!
//! # It reads a SNAPSHOT, not the world
//!
//! `inf-ui` does not depend on `inf-ecs` — a UI is not a simulation — so the
//! host projects an [`InventoryView`] out of `inf_ecs::item::Inventory` once per
//! frame and the panel renders that. The verbs go back the other way as values
//! ([`InventoryVerb`]), which the host applies **on the sim's own step**: a
//! panel that reached into the world would be a second place gameplay happens,
//! and it would happen on the frame clock instead of the fixed one.

use crate::draw::{Align, Rect, UiDrawList};

/// How many slots a row of the grid holds.
pub const GRID_COLUMNS: usize = 5;

/// **What the panel is doing.**
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InventoryState {
    /// Whether it is on screen.
    pub open: bool,
    /// The focused slot.
    pub focus: usize,
    /// The slot a move started from, if one is in progress.
    ///
    /// A move is two presses — pick up, put down — rather than a drag, because
    /// the panel is keyboard-driven (the mouse is I5's own carried remainder)
    /// and because two presses are what a gamepad can do.
    pub held: Option<usize>,
}

impl InventoryState {
    /// Open or close it, resetting the cursor.
    pub fn set_open(&mut self, open: bool) {
        self.open = open;
        self.focus = 0;
        self.held = None;
    }
}

/// **One slot, as the panel renders it** — the host's projection of an
/// `inf_ecs::item::Inventory`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InventorySlot {
    /// What the item is called. Empty for an empty slot.
    pub label: String,
    /// How many are in it. `0` for an empty slot.
    pub count: u32,
    /// Whether this is the equipped one.
    pub equipped: bool,
    /// Whether it can be equipped at all — a weapon can, a bandage cannot.
    pub equippable: bool,
}

impl InventorySlot {
    /// Whether there is anything in it.
    pub fn occupied(&self) -> bool {
        self.count > 0 && !self.label.is_empty()
    }
}

/// **What a character is carrying**, as the panel sees it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InventoryView {
    /// Every slot, in grid order.
    pub slots: Vec<InventorySlot>,
}

/// One input the panel might take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InventoryInput {
    /// A `KeyboardEvent.code`.
    Key(String),
}

/// **What the panel decided** — a value the host applies on the sim's step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryVerb {
    /// Move (or merge) one slot onto another.
    Move { from: usize, to: usize },
    /// Put a slot's contents on the floor.
    Drop(usize),
    /// Equip a slot.
    Equip(usize),
}

/// What routing one input did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InventoryOutcome {
    /// The panel took the key; the game must not see it.
    pub consumed: bool,
    /// A verb for the host to apply, if the press produced one.
    pub verb: Option<InventoryVerb>,
    /// Whether the panel closed on this press.
    pub closed: bool,
}

/// **The reducer.** One key in, one outcome out, and the state moved.
///
/// **An open panel takes every key it is given** — the `menu` reducer's own
/// rule, and for the same reason: "consumed" is about who the input belongs to,
/// not about whether anything happened. A panel that only claimed the keys it
/// used would let a player walk while tidying their bag.
///
/// `Escape` closes it, and `Escape` cannot be bound away (the I5 rule), so a
/// player always has a way out.
pub fn handle(
    state: &mut InventoryState,
    view: &InventoryView,
    input: &InventoryInput,
) -> InventoryOutcome {
    if !state.open {
        return InventoryOutcome::default();
    }
    let n = view.slots.len();
    let mut out = InventoryOutcome {
        consumed: true,
        ..Default::default()
    };
    let InventoryInput::Key(code) = input;
    if n == 0 {
        // An inventory with no slots still eats the keys — it is on screen.
        if code == "Escape" {
            state.set_open(false);
            out.closed = true;
        }
        return out;
    }
    state.focus = state.focus.min(n - 1);
    match code.as_str() {
        "Escape" => {
            state.set_open(false);
            out.closed = true;
        }
        "ArrowLeft" => state.focus = step(state.focus, n, -1),
        "ArrowRight" => state.focus = step(state.focus, n, 1),
        "ArrowUp" => state.focus = step(state.focus, n, -(GRID_COLUMNS as isize)),
        "ArrowDown" => state.focus = step(state.focus, n, GRID_COLUMNS as isize),
        "Enter" => {
            // Pick up, then put down — two presses. Putting a slot down on
            // itself cancels rather than producing a no-op move, so a player who
            // changes their mind has a way to.
            match state.held {
                Some(from) if from == state.focus => state.held = None,
                Some(from) => {
                    out.verb = Some(InventoryVerb::Move {
                        from,
                        to: state.focus,
                    });
                    state.held = None;
                }
                None if view.slots[state.focus].occupied() => state.held = Some(state.focus),
                None => {}
            }
        }
        "KeyQ" => {
            if view.slots[state.focus].occupied() {
                out.verb = Some(InventoryVerb::Drop(state.focus));
                state.held = None;
            }
        }
        "KeyF" => {
            let slot = &view.slots[state.focus];
            if slot.occupied() && slot.equippable {
                out.verb = Some(InventoryVerb::Equip(state.focus));
            }
        }
        // Every other key is taken and does nothing — see the doc above.
        _ => {}
    }
    out
}

/// Move the cursor by `by`, wrapping.
fn step(focus: usize, n: usize, by: isize) -> usize {
    if n == 0 {
        return 0;
    }
    let n_i = n as isize;
    let mut i = (focus as isize + by) % n_i;
    if i < 0 {
        i += n_i;
    }
    i as usize
}

/// The footer's key legend — one string, so the panel and a future gamepad glyph
/// row read the same sentence.
pub fn legend() -> &'static str {
    "arrows move  Enter takes/places  F equips  Q drops  Esc closes"
}

/// **Draw it.** A grid in the lower half of the screen, with the focused slot
/// outlined and the equipped one marked.
///
/// A HUD panel rather than a modal, so there is **no scrim**: the world stays
/// visible behind it, which is the whole point of not pausing.
pub fn draw(list: &mut UiDrawList, state: &InventoryState, view: &InventoryView) {
    if !state.open {
        return;
    }
    let vp = list.viewport;
    if !vp.x.is_finite() || !vp.y.is_finite() || vp.x <= 0.0 || vp.y <= 0.0 {
        return;
    }
    let scale = crate::view::text_scale(vp.y);
    let cell = (36.0 * scale).min(vp.x / (GRID_COLUMNS as f32 + 1.0));
    let pad = 8.0 * scale;
    let rows = view.slots.len().div_ceil(GRID_COLUMNS).max(1);
    let w = GRID_COLUMNS as f32 * (cell + pad) + pad;
    let h = rows as f32 * (cell + pad) + pad * 4.0;
    let x = (vp.x - w) * 0.5;
    let y = vp.y - h - pad * 3.0;
    list.rect(Rect::new(x, y, w, h), crate::view::palette::PANEL);
    list.stroke(Rect::new(x, y, w, h), 2.0, crate::view::palette::EDGE);
    for (i, slot) in view.slots.iter().enumerate() {
        let col = (i % GRID_COLUMNS) as f32;
        let row = (i / GRID_COLUMNS) as f32;
        let r = Rect::new(
            x + pad + col * (cell + pad),
            y + pad + row * (cell + pad),
            cell,
            cell,
        );
        let held = state.held == Some(i);
        let body = if held {
            crate::view::palette::WARN
        } else if slot.equipped {
            crate::view::palette::EDGE
        } else {
            crate::view::palette::TOAST
        };
        list.rect(r, body);
        if state.focus == i {
            list.stroke(r, 2.0, crate::view::palette::TEXT);
        }
        if slot.occupied() {
            // The label is clipped by the cell rather than allowed to run into
            // its neighbour: an 8 x 8 font and a 36-pixel cell is four glyphs.
            list.text_in(
                r.inset(3.0),
                2.0,
                Align::Left,
                &slot.label,
                (scale - 1.0).max(1.0),
                crate::view::palette::TEXT,
            );
            if slot.count > 1 {
                list.text_in(
                    r.inset(3.0),
                    2.0,
                    Align::Right,
                    &format!("{}", slot.count),
                    (scale - 1.0).max(1.0),
                    crate::view::palette::MUTED,
                );
            }
        }
    }
    list.text_in(
        Rect::new(x, y + h - pad * 3.0, w, pad * 3.0),
        pad,
        Align::Center,
        legend(),
        (scale - 1.0).max(1.0),
        crate::view::palette::MUTED,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> InventoryView {
        InventoryView {
            slots: vec![
                InventorySlot {
                    label: "Rifle".into(),
                    count: 1,
                    equipped: true,
                    equippable: true,
                },
                InventorySlot::default(),
                InventorySlot {
                    label: "Bandage".into(),
                    count: 3,
                    equipped: false,
                    equippable: false,
                },
                InventorySlot::default(),
                InventorySlot::default(),
                InventorySlot::default(),
            ],
        }
    }

    fn key(c: &str) -> InventoryInput {
        InventoryInput::Key(c.to_string())
    }

    /// **A closed panel takes nothing**, and an open one takes everything.
    #[test]
    fn an_open_panel_takes_every_key_and_a_closed_one_takes_none() {
        let mut st = InventoryState::default();
        let v = view();
        assert_eq!(
            handle(&mut st, &v, &key("ArrowRight")),
            InventoryOutcome::default()
        );
        st.set_open(true);
        // Even a key it has no control for: "consumed" is about ownership.
        let out = handle(&mut st, &v, &key("KeyZ"));
        assert!(out.consumed, "an open panel let a key through to the game");
        assert_eq!(out.verb, None);
        // …including the movement keys, which is what stops a player walking
        // while tidying their bag.
        assert!(handle(&mut st, &v, &key("KeyW")).consumed);
        assert!(handle(&mut st, &v, &key("Space")).consumed);
    }

    /// **The cursor wraps in both directions**, on both axes.
    #[test]
    fn the_cursor_wraps_around_the_grid() {
        let mut st = InventoryState::default();
        st.set_open(true);
        let v = view();
        assert_eq!(st.focus, 0);
        handle(&mut st, &v, &key("ArrowLeft"));
        assert_eq!(st.focus, 5, "left from the first slot did not wrap");
        handle(&mut st, &v, &key("ArrowRight"));
        assert_eq!(st.focus, 0);
        handle(&mut st, &v, &key("ArrowDown"));
        assert_eq!(st.focus, 5, "down is one row of {GRID_COLUMNS}");
        handle(&mut st, &v, &key("ArrowUp"));
        assert_eq!(st.focus, 0);
        // A grid smaller than one row still wraps rather than going out of
        // bounds — the case a bag of three slots is.
        let small = InventoryView {
            slots: vec![InventorySlot::default(); 3],
        };
        let mut s2 = InventoryState::default();
        s2.set_open(true);
        handle(&mut s2, &small, &key("ArrowDown"));
        assert!(s2.focus < 3, "{}", s2.focus);
        handle(&mut s2, &small, &key("ArrowLeft"));
        assert!(s2.focus < 3);
    }

    /// **A move is two presses, and putting it back cancels.**
    #[test]
    fn a_move_takes_two_presses_and_can_be_cancelled() {
        let mut st = InventoryState::default();
        st.set_open(true);
        let v = view();
        // An empty slot cannot be picked up.
        st.focus = 1;
        assert_eq!(handle(&mut st, &v, &key("Enter")).verb, None);
        assert_eq!(st.held, None, "an empty slot was picked up");
        // A full one can.
        st.focus = 0;
        assert_eq!(handle(&mut st, &v, &key("Enter")).verb, None);
        assert_eq!(st.held, Some(0));
        // Putting it back where it came from cancels rather than moving.
        assert_eq!(handle(&mut st, &v, &key("Enter")).verb, None);
        assert_eq!(st.held, None, "a cancel left the slot held");
        // …and somewhere else is a move.
        handle(&mut st, &v, &key("Enter"));
        st.focus = 3;
        let out = handle(&mut st, &v, &key("Enter"));
        assert_eq!(out.verb, Some(InventoryVerb::Move { from: 0, to: 3 }));
        assert_eq!(st.held, None);
    }

    /// **Drop and equip are values**, and equip refuses what cannot be equipped.
    #[test]
    fn drop_and_equip_are_verbs_and_a_bandage_is_not_a_weapon() {
        let mut st = InventoryState::default();
        st.set_open(true);
        let v = view();
        st.focus = 0;
        assert_eq!(
            handle(&mut st, &v, &key("KeyQ")).verb,
            Some(InventoryVerb::Drop(0))
        );
        assert_eq!(
            handle(&mut st, &v, &key("KeyF")).verb,
            Some(InventoryVerb::Equip(0))
        );
        // An empty slot drops nothing and equips nothing.
        st.focus = 1;
        assert_eq!(handle(&mut st, &v, &key("KeyQ")).verb, None);
        assert_eq!(handle(&mut st, &v, &key("KeyF")).verb, None);
        // A bandage drops and does not equip.
        st.focus = 2;
        assert_eq!(
            handle(&mut st, &v, &key("KeyQ")).verb,
            Some(InventoryVerb::Drop(2))
        );
        assert_eq!(
            handle(&mut st, &v, &key("KeyF")).verb,
            None,
            "a bandage was equipped as a weapon"
        );
    }

    /// **Escape always closes it** — the key I5 refuses to let a player bind
    /// away, so there is always a way out.
    #[test]
    fn escape_closes_the_panel_even_with_a_move_in_progress() {
        let mut st = InventoryState::default();
        st.set_open(true);
        let v = view();
        handle(&mut st, &v, &key("Enter"));
        assert_eq!(st.held, Some(0));
        let out = handle(&mut st, &v, &key("Escape"));
        assert!(out.closed && out.consumed);
        assert!(!st.open);
        assert_eq!(st.held, None, "a half-finished move survived the close");
        // …and a bag with no slots at all still closes.
        let empty = InventoryView::default();
        let mut s2 = InventoryState::default();
        s2.set_open(true);
        assert!(handle(&mut s2, &empty, &key("ArrowRight")).consumed);
        assert!(handle(&mut s2, &empty, &key("Escape")).closed);
        assert!(!s2.open);
    }

    /// **The draw is a pure function of its inputs**, refuses a hostile
    /// viewport, and draws nothing when it is closed.
    #[test]
    fn the_panel_draws_nothing_closed_and_something_open() {
        let mut list = UiDrawList::new(glam::Vec2::new(1920.0, 1080.0));
        let mut st = InventoryState::default();
        let v = view();
        draw(&mut list, &st, &v);
        assert!(list.is_empty(), "a closed panel drew something");
        st.set_open(true);
        draw(&mut list, &st, &v);
        let quads = list.quads.len();
        println!("an open six-slot panel is {quads} quads");
        assert!(quads > 6, "{quads}");
        // Twice is twice — a projection, not an accumulator with a memory.
        let mut a = UiDrawList::new(glam::Vec2::new(1920.0, 1080.0));
        draw(&mut a, &st, &v);
        let mut b = UiDrawList::new(glam::Vec2::new(1920.0, 1080.0));
        draw(&mut b, &st, &v);
        assert_eq!(a.quads.len(), b.quads.len());
        // A hostile viewport draws nothing rather than a NaN rectangle.
        let mut bad = UiDrawList::new(glam::Vec2::new(f32::NAN, 1080.0));
        draw(&mut bad, &st, &v);
        assert!(bad.is_empty());
        let mut zero = UiDrawList::new(glam::Vec2::new(0.0, 0.0));
        draw(&mut zero, &st, &v);
        assert!(zero.is_empty());
    }
}
