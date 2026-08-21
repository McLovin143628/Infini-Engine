//! **The in-game settings dialog**: its state, its keyboard reducer, and the
//! projection that turns both into rows.
//!
//! # It is a state machine and a pure projection, not an immediate-mode UI
//!
//! Three functions and one struct:
//!
//! * [`MenuState`] is what the player has done — open, which page, which row,
//!   whether a key capture is running.
//! * [`rows`] projects `(state, settings, map)` into the list a page shows. Pure.
//! * [`handle`] folds one key into `(state, settings, map)`. Pure.
//!
//! The *drawing* ([`crate::view`]) is a third pure function over the same
//! projection, so the thing the player sees and the thing the arrow keys move
//! through are the same list by construction rather than by two pieces of code
//! agreeing. Every arm in this module runs without a GPU, a window or a font.
//!
//! # Tab pauses the simulation, and the pause is sim state
//!
//! **The ruling this wave takes** (the brief asked for one and this is it): in a
//! single-player session, opening the menu **pauses the fixed step**. Two
//! reasons, and the second is the one that matters here.
//!
//! * A player rebinding a key while a character keeps running is being asked to
//!   read a table and stay alive at the same time, and every shipped
//!   single-player game answers this the same way.
//! * A menu that did *not* pause would make the UI part of the simulation's
//!   input: the frames spent in it are frames the sim advanced, so a trace that
//!   opened the menu would depend on how long the player took. Pausing makes the
//!   menu's cost **zero fixed steps**, which is why a parity trace can open it,
//!   rebind a key and close it again and still be byte-identical between PIE and
//!   shipping.
//!
//! The pause is therefore a fact about the *session* rather than about the
//! window, and the host applies it through the sim's own door — see
//! `RuntimeSim::set_sim_paused`. A multiplayer session cannot pause a shared
//! world and does not; the host asks [`MenuState::pauses_sim`], which answers
//! for the session it is given.

use inf_input::{InputMap, MouseButton};

use crate::bindings::{self, BindingRow};
use crate::settings::{
    GameSettings, LOOK_SENSITIVITY_RANGE, PRESS_THRESHOLD_RANGE_MS, RENDER_TIERS, RESOLUTIONS,
    VOLUME_RANGE, WINDOW_MODES,
};

/// The dialog's pages, in the order the tab strip lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Page {
    /// Window mode, resolution, render tier.
    #[default]
    Video,
    /// The mixer buses.
    Audio,
    /// Sensitivity, invert-Y, the press threshold.
    Controls,
    /// The rebinding table.
    Bindings,
}

impl Page {
    /// Every page, in tab-strip order.
    pub const ALL: [Page; 4] = [Page::Video, Page::Audio, Page::Controls, Page::Bindings];

    /// The tab's caption.
    pub fn label(self) -> &'static str {
        match self {
            Page::Video => "VIDEO",
            Page::Audio => "AUDIO",
            Page::Controls => "CONTROLS",
            Page::Bindings => "KEY BINDINGS",
        }
    }

    /// The next / previous page, wrapping.
    pub fn step(self, forward: bool) -> Page {
        let i = Page::ALL.iter().position(|p| *p == self).unwrap_or(0);
        let n = Page::ALL.len();
        Page::ALL[if forward {
            (i + 1) % n
        } else {
            (i + n - 1) % n
        }]
    }
}

/// Which setting a row is about. The reducer matches on this rather than on a
/// row index, so inserting a row cannot silently re-point the arrow keys at a
/// different control.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowId {
    /// The page selector — always row 0, on every page.
    Page,
    /// Video.
    WindowMode,
    /// Video.
    Resolution,
    /// Video.
    RenderTier,
    /// Audio.
    MasterVolume,
    /// Audio.
    SfxVolume,
    /// Audio.
    MusicVolume,
    /// Controls.
    LookSensitivity,
    /// Controls.
    InvertLookY,
    /// Controls.
    PressThreshold,
    /// One row of the bindings table, by its index in [`bindings::rows`].
    Binding(usize),
    /// Restore every binding to the shipped table.
    ResetAllBindings,
}

/// How a row responds to the arrow keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    /// Left/right cycles through a list.
    Choice,
    /// Left/right nudges a number.
    Slider,
    /// Left/right/enter flips it.
    Toggle,
    /// Enter starts a key capture.
    Capture,
    /// Enter does something once.
    Button,
}

/// One row, as the dialog draws it and as the reducer reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// What it is about.
    pub id: RowId,
    /// The left column.
    pub label: String,
    /// The right column — already formatted, so the drawing does no arithmetic.
    pub value: String,
    /// How the arrows treat it.
    pub kind: RowKind,
    /// A short line under the row, or nothing. Used for the two things a player
    /// cannot infer: that a control has no consumer yet, and what a conflict is
    /// about to take.
    pub note: Option<String>,
}

/// What the dialog is doing about a key capture.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Capture {
    /// Nothing.
    #[default]
    Idle,
    /// Waiting for the player to press the key that `row` should be bound to.
    Awaiting {
        /// Index into [`bindings::rows`].
        row: usize,
    },
    /// The key they pressed is already taken, and the dialog is asking.
    Conflict {
        /// Index into [`bindings::rows`].
        row: usize,
        /// The token they pressed.
        token: String,
        /// The row ids that already own it.
        owners: Vec<String>,
    },
}

/// The dialog's whole state.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MenuState {
    /// Whether it is showing.
    pub open: bool,
    /// Which page.
    pub page: Page,
    /// Which row has the keyboard, as an index into [`rows`]'s answer.
    pub focus: usize,
    /// The capture state machine.
    pub capture: Capture,
    /// Whether this is a session whose simulation may be paused. A host sets it
    /// once; see the module header for the ruling.
    pub single_player: bool,
}

impl MenuState {
    /// A closed dialog for a single-player session.
    pub fn new() -> Self {
        Self {
            single_player: true,
            ..Default::default()
        }
    }

    /// **Whether the fixed step should be frozen right now.**
    ///
    /// The one door a host asks. It is not `open`: a multiplayer session shares
    /// a world with other people and cannot stop it, so the menu draws over a
    /// running game there and pauses one here.
    pub fn pauses_sim(&self) -> bool {
        self.open && self.single_player
    }

    /// Open or close, resetting the focus and abandoning any capture.
    ///
    /// A capture that survived a close would fire on the next key the *game*
    /// received, which is a rebinding nobody asked for.
    pub fn set_open(&mut self, open: bool) {
        self.open = open;
        self.focus = 0;
        self.capture = Capture::Idle;
    }
}

/// A key or button the dialog was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuInput {
    /// A `KeyboardEvent.code` string.
    Key(String),
    /// A mouse button.
    Mouse(MouseButton),
}

impl MenuInput {
    /// The binding token this input would be written as, or `None` for an input
    /// that cannot be bound.
    fn token(&self) -> String {
        match self {
            MenuInput::Key(c) => c.clone(),
            MenuInput::Mouse(b) => format!("Mouse:{b:?}"),
        }
    }

    fn key(&self) -> Option<&str> {
        match self {
            MenuInput::Key(c) => Some(c.as_str()),
            MenuInput::Mouse(_) => None,
        }
    }
}

/// What one key did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MenuOutcome {
    /// Whether the dialog took the key. **A key the dialog took must not reach
    /// the game**, or the same press that moves the cursor also fires a weapon.
    pub consumed: bool,
    /// Whether [`GameSettings`] changed and should be persisted.
    pub settings_changed: bool,
    /// Whether the [`InputMap`] changed and should be persisted.
    pub bindings_changed: bool,
    /// Whether the dialog closed as a result.
    pub closed: bool,
}

/// The keys that cannot be bound to anything, because the dialog needs them.
///
/// A player who bound `Escape` to *Attack* would have no way to leave a capture.
/// Refused at the capture, as a value, with the capture left open so a real key
/// still works.
///
/// **This list is about the two keys the dialog needs while it is OPEN, and it
/// is not the rule that keeps the dialog reachable** (I5 audit, A1). The first
/// draft's doc claimed both, and only the first was built: the menu key was a
/// table row like any other, so a swap could take `Tab` for *Attack* and a left
/// arrow on the *Menu* row could clear it, and either one is a settings file a
/// keyboard player can never open the dialog to fix. That rule is
/// [`bindings::guarded`] / [`bindings::menu_is_unreachable`], and it lives there
/// rather than here because the editor's preferences panel is a **second
/// surface** that can make the same edit and must be refused by the same rule.
pub const RESERVED_KEYS: [&str; 2] = ["Escape", "Enter"];

/// **The rows a page shows**, derived from the settings and the map.
///
/// Pure and total: the same three inputs always give the same list, so the
/// drawing and the reducer cannot be looking at different rows.
pub fn rows(state: &MenuState, s: &GameSettings, map: &InputMap) -> Vec<Row> {
    let mut out = vec![Row {
        id: RowId::Page,
        label: "Page".into(),
        value: state.page.label().into(),
        kind: RowKind::Choice,
        note: None,
    }];
    match state.page {
        Page::Video => {
            // **All three are STORED and not APPLIED, and the row says so** (I5
            // audit, A5) — see [`STORED_NOT_APPLIED_NOTE`].
            out.push(stored("Window Mode", &s.window_mode, RowId::WindowMode));
            out.push(stored(
                "Resolution",
                &format!("{} x {}", s.resolution_w, s.resolution_h),
                RowId::Resolution,
            ));
            out.push(stored("Quality", &s.render_tier, RowId::RenderTier));
        }
        Page::Audio => {
            out.push(slider("Master", s.master_volume, RowId::MasterVolume));
            out.push(slider("Effects", s.sfx_volume, RowId::SfxVolume));
            out.push(slider("Music", s.music_volume, RowId::MusicVolume));
        }
        Page::Controls => {
            out.push(Row {
                id: RowId::LookSensitivity,
                label: "Look Sensitivity".into(),
                value: format!("{:.2}", s.look_sensitivity),
                kind: RowKind::Slider,
                note: None,
            });
            out.push(Row {
                id: RowId::InvertLookY,
                label: "Invert Look Y".into(),
                value: if s.invert_look_y { "ON" } else { "OFF" }.into(),
                kind: RowKind::Toggle,
                note: None,
            });
            out.push(Row {
                id: RowId::PressThreshold,
                label: "Hold Threshold".into(),
                value: format!("{} ms", s.press_threshold_ms),
                kind: RowKind::Slider,
                note: Some("how long C must be held to go prone or dive".into()),
            });
        }
        Page::Bindings => {
            let table = bindings::rows();
            for (i, row) in table.iter().enumerate() {
                out.push(binding_row(state, map, i, row));
            }
            out.push(Row {
                id: RowId::ResetAllBindings,
                label: "Reset All".into(),
                value: "[Enter]".into(),
                kind: RowKind::Button,
                note: None,
            });
        }
    }
    out
}

fn choice(label: &str, value: &str, id: RowId) -> Row {
    Row {
        id,
        label: label.into(),
        value: value.into(),
        kind: RowKind::Choice,
        note: None,
    }
}

/// What the video page's rows say about themselves (I5 audit, A5).
///
/// The three of them are written to the settings file and read back, and nothing
/// resizes a window, reconfigures a swap chain or re-detects a tier yet — that is
/// a winit/surface change on a path this repository's CI cannot run, and the
/// wave's ledger carries it as an honest remainder.
///
/// **The ledger is not where a player reads it.** This wave's own rule for a
/// control that is bound against a consumer that does not exist is to say so on
/// the row it is on ([`not_yet_note`], and the toast beside it): *a dead key is
/// indistinguishable from a broken one*. A slider that is stored and applied by
/// nothing is the same thing one level up — the ledger says as much about the
/// audio and controls pages, which were wired for exactly that reason — and these
/// three were the ones left silent. A player who changes the resolution and sees
/// nothing happen is owed the sentence.
///
/// **It does not promise a restart**, because a restart would not help either:
/// nothing reads these three at boot any more than it does at runtime, and a note
/// that sent a player to relaunch the game would be a second wrong answer on top
/// of the first. ASCII only, like every string this 8x8 basic-Latin font draws.
pub const STORED_NOT_APPLIED_NOTE: &str = "saved, and nothing applies it yet";

fn stored(label: &str, value: &str, id: RowId) -> Row {
    Row {
        note: Some(STORED_NOT_APPLIED_NOTE.to_string()),
        ..choice(label, value, id)
    }
}

fn slider(label: &str, v: f32, id: RowId) -> Row {
    Row {
        id,
        label: label.into(),
        value: format!("{:>3} %", (v * 100.0).round() as i32),
        kind: RowKind::Slider,
        note: None,
    }
}

fn binding_row(state: &MenuState, map: &InputMap, i: usize, row: &BindingRow) -> Row {
    let (value, note) = match &state.capture {
        Capture::Awaiting { row: r } if *r == i => ("[press a key]".to_string(), None),
        Capture::Conflict {
            row: r,
            token,
            owners,
        } if *r == i => (
            token.clone(),
            Some(format!(
                "already used by {} -- [Enter] to swap, [Escape] to cancel",
                owners.join(", ")
            )),
        ),
        _ => {
            let t = bindings::token_in(map, row);
            (
                if t.is_empty() { "--".to_string() } else { t },
                not_yet_note(row.name),
            )
        }
    };
    Row {
        id: RowId::Binding(i),
        label: row.label.into(),
        value,
        kind: RowKind::Capture,
        note,
    }
}

/// The honest half of a control bound against a consumer that does not exist.
fn not_yet_note(name: &str) -> Option<String> {
    inf_input::actions::NOT_YET_CONSUMED
        .contains(&name)
        .then(|| "not wired to anything yet".to_string())
}

/// Nudge a slider and report whether it moved.
fn nudge(v: &mut f32, by: f32, range: (f32, f32)) -> bool {
    let before = *v;
    *v = if v.is_finite() { *v + by } else { by };
    *v = v.clamp(range.0, range.1);
    (*v - before).abs() > f32::EPSILON
}

/// Cycle an index through `len`, wrapping in either direction.
fn cycle(i: usize, len: usize, forward: bool) -> usize {
    if len == 0 {
        return 0;
    }
    if forward {
        (i + 1) % len
    } else {
        (i + len - 1) % len
    }
}

/// **Fold one key into the dialog.** Pure over `(state, settings, map)`.
///
/// Returns what happened, so the host knows whether to persist and — the half
/// that matters for correctness — whether the key must be kept away from the
/// game.
pub fn handle(
    state: &mut MenuState,
    s: &mut GameSettings,
    map: &mut InputMap,
    input: &MenuInput,
) -> MenuOutcome {
    if !state.open {
        return MenuOutcome::default();
    }
    let mut out = MenuOutcome {
        consumed: true,
        ..Default::default()
    };

    // ── a capture eats EVERY key, which is the point of one ──
    //
    // Escape cancels and Enter confirms a swap; everything else is the key the
    // player is trying to bind. A capture that let the arrow keys through could
    // never bind an arrow key, which is a control scheme this engine ships.
    match state.capture.clone() {
        Capture::Awaiting { row } => {
            if input.key() == Some("Escape") {
                state.capture = Capture::Idle;
                return out;
            }
            let token = input.token();
            if RESERVED_KEYS.contains(&token.as_str()) {
                // Refused as a value: the dialog stays in capture and the row
                // says why on its next draw.
                return out;
            }
            let table = bindings::rows();
            let Some(target) = table.get(row) else {
                state.capture = Capture::Idle;
                return out;
            };
            let owners = bindings::conflicts(map, &token, &target.id);
            if owners.is_empty() {
                // No guard is needed here and that is a fact rather than an
                // omission: with no conflict, this only *replaces* one row's
                // desk source, so no row — the menu's included — can be left
                // with fewer keys than it had (I5 audit, A1).
                out.bindings_changed = bindings::set_row(map, target, &token);
                state.capture = Capture::Idle;
            } else {
                state.capture = Capture::Conflict { row, token, owners };
            }
            return out;
        }
        Capture::Conflict { row, token, owners } => {
            // **Only `Enter` confirms.** Anything else cancels — not just
            // Escape: a player who has been shown a warning and presses a third
            // key has not confirmed anything.
            if input.key() == Some("Enter") {
                let table = bindings::rows();
                if let Some(target) = table.get(row) {
                    // **The swap takes the KEY off the previous owners**, not
                    // their whole binding. A key bound to two rows is a key that
                    // fires two verbs, and the dialog just told the player which
                    // rows it was about to take it from — but a row with a
                    // second key keeps it, which is the honest outcome: Move
                    // Forward is W *and* Up in the shipped table, and taking W
                    // leaves Up.
                    //
                    // **And the one owner a swap may not take it from is the
                    // MENU** (I5 audit, A1): Tab is the only way back into this
                    // dialog, so a swap that took it would be the one edit a
                    // player can never undo. [`bindings::guarded`] applies the
                    // whole swap or none of it, and a refusal is a value — the
                    // dialog closes the capture and nothing moved.
                    out.bindings_changed = bindings::guarded(map, |m| {
                        let mut changed = false;
                        for id in &owners {
                            if let Some(other) = table.iter().find(|r| &r.id == id) {
                                changed |= bindings::unbind_token(m, other, &token);
                            }
                        }
                        changed | bindings::set_row(m, target, &token)
                    });
                }
            }
            state.capture = Capture::Idle;
            return out;
        }
        Capture::Idle => {}
    }

    let list = rows(state, s, map);
    let len = list.len().max(1);
    state.focus = state.focus.min(len - 1);
    // **AN OPEN DIALOG IS MODAL.** Everything from here down consumes the key,
    // including the ones the dialog has no use for: a menu that let `Space`
    // through would have the player jumping while they read the video page, and
    // one that let the mouse buttons through would fire a weapon at whatever is
    // behind it. "Consumed" is about who the input belongs to, not about whether
    // anything happened.
    let Some(key) = input.key() else {
        return out;
    };
    match key {
        "ArrowUp" | "KeyW" => state.focus = cycle(state.focus, len, false),
        "ArrowDown" | "KeyS" => state.focus = cycle(state.focus, len, true),
        "Escape" | "Tab" => {
            state.set_open(false);
            out.closed = true;
        }
        "ArrowLeft" | "ArrowRight" | "KeyA" | "KeyD" | "Enter" => {
            let forward = matches!(key, "ArrowRight" | "KeyD" | "Enter");
            let activate = key == "Enter";
            let row = &list[state.focus];
            match &row.id {
                RowId::Page => {
                    if !activate {
                        state.page = state.page.step(forward);
                        state.focus = 0;
                    }
                }
                RowId::WindowMode => {
                    let i = WINDOW_MODES
                        .iter()
                        .position(|m| *m == s.window_mode)
                        .unwrap_or(0);
                    s.window_mode = WINDOW_MODES[cycle(i, WINDOW_MODES.len(), forward)].into();
                    out.settings_changed = true;
                }
                RowId::Resolution => {
                    let i = RESOLUTIONS
                        .iter()
                        .position(|(w, h)| *w == s.resolution_w && *h == s.resolution_h)
                        .unwrap_or(0);
                    let (w, h) = RESOLUTIONS[cycle(i, RESOLUTIONS.len(), forward)];
                    s.resolution_w = w;
                    s.resolution_h = h;
                    out.settings_changed = true;
                }
                RowId::RenderTier => {
                    let i = RENDER_TIERS
                        .iter()
                        .position(|t| *t == s.render_tier)
                        .unwrap_or(0);
                    s.render_tier = RENDER_TIERS[cycle(i, RENDER_TIERS.len(), forward)].into();
                    out.settings_changed = true;
                }
                RowId::MasterVolume => {
                    let d = if forward { 0.05 } else { -0.05 };
                    out.settings_changed = nudge(&mut s.master_volume, d, VOLUME_RANGE);
                }
                RowId::SfxVolume => {
                    let d = if forward { 0.05 } else { -0.05 };
                    out.settings_changed = nudge(&mut s.sfx_volume, d, VOLUME_RANGE);
                }
                RowId::MusicVolume => {
                    let d = if forward { 0.05 } else { -0.05 };
                    out.settings_changed = nudge(&mut s.music_volume, d, VOLUME_RANGE);
                }
                RowId::LookSensitivity => {
                    let d = if forward { 0.05 } else { -0.05 };
                    out.settings_changed =
                        nudge(&mut s.look_sensitivity, d, LOOK_SENSITIVITY_RANGE);
                }
                RowId::InvertLookY => {
                    s.invert_look_y = !s.invert_look_y;
                    out.settings_changed = true;
                }
                RowId::PressThreshold => {
                    let before = s.press_threshold_ms;
                    let step = 10;
                    s.press_threshold_ms = if forward {
                        s.press_threshold_ms.saturating_add(step)
                    } else {
                        s.press_threshold_ms.saturating_sub(step)
                    }
                    .clamp(PRESS_THRESHOLD_RANGE_MS.0, PRESS_THRESHOLD_RANGE_MS.1);
                    out.settings_changed = s.press_threshold_ms != before;
                }
                RowId::Binding(i) => {
                    if activate {
                        state.capture = Capture::Awaiting { row: *i };
                    } else if !forward {
                        // Left CLEARS a binding, which is the only way to unbind
                        // one from a keyboard — a capture can never produce the
                        // empty token, because it is waiting for a key.
                        //
                        // **Except the menu's** (I5 audit, A1): clearing that
                        // row is the one edit that cannot be undone, because the
                        // control it clears is the door to the screen that would
                        // undo it. [`bindings::guarded`] refuses it as a value
                        // and the row keeps its key.
                        if let Some(target) = bindings::rows().get(*i) {
                            out.bindings_changed =
                                bindings::guarded(map, |m| bindings::set_row(m, target, ""));
                        }
                    }
                }
                RowId::ResetAllBindings => {
                    if activate {
                        let fresh = inf_input::default_map();
                        out.bindings_changed = *map != fresh;
                        *map = fresh;
                    }
                }
            }
        }
        // A key the dialog has no control for is still the dialog's — see the
        // modality note above.
        _ => {}
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (MenuState, GameSettings, InputMap) {
        let mut st = MenuState::new();
        st.set_open(true);
        (st, GameSettings::default(), inf_input::default_map())
    }

    fn key(st: &mut MenuState, s: &mut GameSettings, m: &mut InputMap, k: &str) -> MenuOutcome {
        handle(st, s, m, &MenuInput::Key(k.to_string()))
    }

    /// **A closed dialog takes nothing.** Every key it consumed while closed
    /// would be a key the game never saw.
    #[test]
    fn a_closed_dialog_consumes_nothing() {
        let (mut st, mut s, mut m) = fixture();
        st.set_open(false);
        for k in ["ArrowDown", "Enter", "Escape", "Tab", "KeyW"] {
            let out = key(&mut st, &mut s, &mut m, k);
            assert!(!out.consumed, "a closed dialog ate `{k}`");
        }
        assert_eq!(s, GameSettings::default());
    }

    /// **An open dialog is modal**: it takes every key, including the ones it
    /// has no control for.
    ///
    /// The mutation this kills: a `_ => consumed = false` fallthrough, which is
    /// what this reducer shipped with for one afternoon. It reads as "the dialog
    /// only claims what it uses" and it means the player jumps while reading the
    /// video page and fires a weapon at whatever is behind the menu.
    #[test]
    fn an_open_dialog_takes_every_key_even_the_ones_it_has_no_use_for() {
        let (mut st, mut s, mut m) = fixture();
        for k in ["Space", "KeyC", "KeyR", "F1", "Digit9", "ShiftLeft"] {
            let out = key(&mut st, &mut s, &mut m, k);
            assert!(out.consumed, "an open dialog let `{k}` through to the game");
            assert!(st.open, "`{k}` closed the dialog");
        }
        // …and a mouse button outside a capture too.
        let out = handle(
            &mut st,
            &mut s,
            &mut m,
            &MenuInput::Mouse(MouseButton::Left),
        );
        assert!(out.consumed, "the dialog let a mouse button through");
        assert!(!out.bindings_changed);
    }

    /// **The pause is a ruling, and it is about the SESSION.**
    #[test]
    fn opening_the_menu_pauses_a_single_player_session_and_not_a_shared_one() {
        let mut st = MenuState::new();
        assert!(!st.pauses_sim(), "a closed menu pauses nothing");
        st.set_open(true);
        assert!(st.pauses_sim());
        st.single_player = false;
        assert!(
            !st.pauses_sim(),
            "a shared world cannot be stopped by one player opening a menu"
        );
    }

    /// The focus wraps in both directions and never leaves the list.
    #[test]
    fn the_focus_wraps_and_stays_inside_the_page() {
        let (mut st, mut s, mut m) = fixture();
        let n = rows(&st, &s, &m).len();
        assert!(n >= 4, "the video page is {n} rows");
        key(&mut st, &mut s, &mut m, "ArrowUp");
        assert_eq!(st.focus, n - 1, "up from the top wraps to the bottom");
        key(&mut st, &mut s, &mut m, "ArrowDown");
        assert_eq!(st.focus, 0);
        for _ in 0..(n * 3) {
            key(&mut st, &mut s, &mut m, "ArrowDown");
            assert!(st.focus < n, "the focus left the page: {}", st.focus);
        }
    }

    /// Every page is reachable, and switching one resets the focus — a focus of
    /// 12 on a three-row page would point at nothing.
    #[test]
    fn every_page_is_reachable_and_switching_resets_the_focus() {
        let (mut st, mut s, mut m) = fixture();
        key(&mut st, &mut s, &mut m, "ArrowDown");
        assert_ne!(st.focus, 0);
        let mut seen = vec![st.page];
        for _ in 0..3 {
            st.focus = 0;
            key(&mut st, &mut s, &mut m, "ArrowRight");
            assert_eq!(st.focus, 0, "the focus survived a page change");
            seen.push(st.page);
        }
        for p in Page::ALL {
            assert!(seen.contains(&p), "{p:?} is unreachable");
        }
        // …and it wraps back.
        key(&mut st, &mut s, &mut m, "ArrowRight");
        assert_eq!(st.page, Page::Video);
    }

    /// Every settings row moves its own setting and no other.
    #[test]
    fn each_row_moves_exactly_the_setting_it_names() {
        let (mut st, mut s, mut m) = fixture();
        let cases: Vec<(Page, RowId)> = vec![
            (Page::Video, RowId::WindowMode),
            (Page::Video, RowId::Resolution),
            (Page::Video, RowId::RenderTier),
            (Page::Audio, RowId::MasterVolume),
            (Page::Audio, RowId::SfxVolume),
            (Page::Audio, RowId::MusicVolume),
            (Page::Controls, RowId::LookSensitivity),
            (Page::Controls, RowId::InvertLookY),
            (Page::Controls, RowId::PressThreshold),
        ];
        for (page, id) in cases {
            let before = s.clone();
            st.page = page;
            let list = rows(&st, &s, &m);
            st.focus = list.iter().position(|r| r.id == id).unwrap_or_else(|| {
                panic!("{id:?} is not on {page:?}");
            });
            // Right first, then left: several of these ship at the TOP of their
            // band (the mixer buses are all 100 %), where right saturates and
            // reports — correctly — that nothing moved. What the arm is about is
            // that the row responds to the arrows *at all*.
            let mut out = key(&mut st, &mut s, &mut m, "ArrowRight");
            if !out.settings_changed {
                out = key(&mut st, &mut s, &mut m, "ArrowLeft");
            }
            assert!(out.consumed && out.settings_changed, "{id:?} did nothing");
            assert_ne!(s, before, "{id:?} changed nothing");
            // Exactly one field moved: normalizing both and comparing every
            // other field is what says so.
            let mut probe = s.clone();
            match id {
                RowId::WindowMode => probe.window_mode = before.window_mode.clone(),
                RowId::Resolution => {
                    probe.resolution_w = before.resolution_w;
                    probe.resolution_h = before.resolution_h;
                }
                RowId::RenderTier => probe.render_tier = before.render_tier.clone(),
                RowId::MasterVolume => probe.master_volume = before.master_volume,
                RowId::SfxVolume => probe.sfx_volume = before.sfx_volume,
                RowId::MusicVolume => probe.music_volume = before.music_volume,
                RowId::LookSensitivity => probe.look_sensitivity = before.look_sensitivity,
                RowId::InvertLookY => probe.invert_look_y = before.invert_look_y,
                RowId::PressThreshold => probe.press_threshold_ms = before.press_threshold_ms,
                _ => unreachable!(),
            }
            assert_eq!(probe, before, "{id:?} moved a setting it is not about");
        }
    }

    /// A slider at its limit stops there and **says it did not move**, so a host
    /// does not write the file on every key.
    #[test]
    fn a_slider_saturates_and_reports_that_it_did_not_move() {
        let (mut st, mut s, mut m) = fixture();
        st.page = Page::Audio;
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::MasterVolume)
            .unwrap();
        for _ in 0..50 {
            key(&mut st, &mut s, &mut m, "ArrowLeft");
        }
        assert_eq!(s.master_volume, VOLUME_RANGE.0);
        let out = key(&mut st, &mut s, &mut m, "ArrowLeft");
        assert!(
            !out.settings_changed,
            "a saturated slider reported a change, so the settings file is rewritten on every key"
        );
        assert!(out.consumed, "and it still ate the key");
    }

    /// **The whole rebinding round trip, through the dialog.**
    #[test]
    fn a_capture_binds_a_free_key_and_a_conflict_names_its_owner() {
        let (mut st, mut s, mut m) = fixture();
        st.page = Page::Bindings;
        let table = bindings::rows();
        let interact = table.iter().position(|r| r.id == "interact").unwrap();
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(interact))
            .unwrap();

        // Enter starts the capture, and the row says so.
        let out = key(&mut st, &mut s, &mut m, "Enter");
        assert!(out.consumed && !out.bindings_changed);
        assert_eq!(st.capture, Capture::Awaiting { row: interact });
        let shown = rows(&st, &s, &m);
        assert_eq!(shown[st.focus].value, "[press a key]");

        // A free key binds straight away.
        let out = key(&mut st, &mut s, &mut m, "KeyJ");
        assert!(out.bindings_changed);
        assert_eq!(st.capture, Capture::Idle);
        assert_eq!(bindings::token_in(&m, &table[interact]), "KeyJ");

        // A TAKEN key stops and names the owner instead of stealing it.
        key(&mut st, &mut s, &mut m, "Enter");
        let out = key(&mut st, &mut s, &mut m, "KeyW");
        assert!(
            !out.bindings_changed,
            "a conflicting key was taken without asking"
        );
        let Capture::Conflict { owners, .. } = st.capture.clone() else {
            panic!("no conflict was raised: {:?}", st.capture);
        };
        assert_eq!(owners, ["move_y+"]);
        let shown = rows(&st, &s, &m);
        assert!(
            shown[st.focus]
                .note
                .as_deref()
                .is_some_and(|n| n.contains("move_y+")),
            "the row does not name the owner: {:?}",
            shown[st.focus].note
        );
        // Move Forward still has W while the question is open.
        let forward = table.iter().position(|r| r.id == "move_y+").unwrap();
        assert_eq!(bindings::token_in(&m, &table[forward]), "KeyW");

        // Escape cancels, and nothing moved.
        key(&mut st, &mut s, &mut m, "Escape");
        assert_eq!(st.capture, Capture::Idle);
        assert_eq!(bindings::token_in(&m, &table[interact]), "KeyJ");
        assert_eq!(bindings::token_in(&m, &table[forward]), "KeyW");

        // Confirming the swap takes it from the owner, so no key fires two verbs.
        key(&mut st, &mut s, &mut m, "Enter");
        key(&mut st, &mut s, &mut m, "KeyW");
        let out = key(&mut st, &mut s, &mut m, "Enter");
        assert!(out.bindings_changed);
        assert_eq!(bindings::token_in(&m, &table[interact]), "KeyW");
        assert!(
            bindings::conflicts(&m, "KeyW", "interact").is_empty(),
            "the swap left W bound to another row too, so one press fires two verbs: {:?}",
            bindings::conflicts(&m, "KeyW", "interact")
        );
        // **And Move Forward keeps its OTHER key.** The shipped table binds it
        // to W and to Up; a swap that took the whole row would leave a player
        // who rebound one control unable to walk.
        assert_eq!(
            bindings::token_in(&m, &table[forward]),
            "ArrowUp",
            "the swap took the row's other key as well: {:?}",
            bindings::tokens_in(&m, &table[forward])
        );
    }

    /// **A conflict reads every key a row answers to, not the one it shows.**
    ///
    /// Move Forward is W *and* Up. A check that read only the displayed cell
    /// would let a player bind Up to something else and leave two verbs on one
    /// key — the exact failure the dialog exists to prevent, reached through the
    /// cell instead of through the map.
    #[test]
    fn a_conflict_sees_a_rows_second_key() {
        let m = inf_input::default_map();
        let table = bindings::rows();
        let forward = table.iter().find(|r| r.id == "move_y+").unwrap();
        assert_eq!(bindings::token_in(&m, forward), "KeyW", "the cell shows W");
        assert!(
            bindings::tokens_in(&m, forward).contains(&"ArrowUp".to_string()),
            "the fixture has no second key, so the claim cannot be tested: {:?}",
            bindings::tokens_in(&m, forward)
        );
        assert_eq!(
            bindings::conflicts(&m, "ArrowUp", "interact"),
            ["move_y+"],
            "Up is bound to Move Forward and the conflict check did not see it"
        );
    }

    /// Per-row clear and reset-all, and the reset really restores the table.
    #[test]
    fn a_row_can_be_cleared_and_the_whole_table_reset() {
        let (mut st, mut s, mut m) = fixture();
        st.page = Page::Bindings;
        let table = bindings::rows();
        let jump = table.iter().position(|r| r.id == "jump").unwrap();
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(jump))
            .unwrap();
        let out = key(&mut st, &mut s, &mut m, "ArrowLeft");
        assert!(out.bindings_changed);
        assert_eq!(bindings::token_in(&m, &table[jump]), "");
        assert_eq!(
            rows(&st, &s, &m)[st.focus].value,
            "--",
            "an unbound row must read as unbound, not as a blank a player cannot tell from a bug"
        );

        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::ResetAllBindings)
            .unwrap();
        let out = key(&mut st, &mut s, &mut m, "Enter");
        assert!(out.bindings_changed);
        assert_eq!(m, inf_input::default_map());
        // …and a second reset reports no change.
        assert!(!key(&mut st, &mut s, &mut m, "Enter").bindings_changed);
    }

    /// **A PLAYER CANNOT LOCK THEMSELVES OUT OF THE SETTINGS DIALOG** (I5 audit,
    /// A1).
    ///
    /// Two doors reached it and both are refused now. A left arrow on the *Menu*
    /// row cleared its key outright; and a capture on any other row could press
    /// `Tab`, be told the menu owned it, press `Enter`, and take it. Either one
    /// writes a settings file that outlives the process, so the dialog that
    /// would undo it is the thing that was just unbound — the only edit on this
    /// table with no way back.
    ///
    /// **The control is what makes this a rule and not a ban**: rebinding the
    /// menu itself still works, and once it is on `KeyB`, `Tab` is an ordinary
    /// key that *Attack* may take. What is refused is the state with no menu key
    /// at all, not any particular edit.
    #[test]
    fn the_menu_key_cannot_be_cleared_or_swapped_away() {
        let table = bindings::rows();
        let menu = table.iter().position(|r| r.id == "menu").unwrap();
        let attack = table.iter().position(|r| r.id == "attack").unwrap();

        // ── the left arrow on the Menu row ──
        let (mut st, mut s, mut m) = fixture();
        st.page = Page::Bindings;
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(menu))
            .unwrap();
        let out = key(&mut st, &mut s, &mut m, "ArrowLeft");
        assert!(!out.bindings_changed, "the clear was not refused");
        assert_eq!(
            bindings::token_in(&m, &table[menu]),
            "Tab",
            "the menu row was cleared, so the dialog can never be opened again"
        );
        // …and the same key still clears an ordinary row, so the refusal is
        // about the menu rather than about the control being broken.
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(attack))
            .unwrap();
        assert!(key(&mut st, &mut s, &mut m, "ArrowLeft").bindings_changed);

        // ── the conflict swap ──
        let (mut st, mut s, mut m) = fixture();
        st.page = Page::Bindings;
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(attack))
            .unwrap();
        let before = m.clone();
        key(&mut st, &mut s, &mut m, "Enter");
        key(&mut st, &mut s, &mut m, "Tab");
        let Capture::Conflict { owners, .. } = st.capture.clone() else {
            panic!("Tab did not raise the menu as a conflict: {:?}", st.capture);
        };
        assert_eq!(owners, ["menu"]);
        let out = key(&mut st, &mut s, &mut m, "Enter");
        assert!(!out.bindings_changed, "the swap was not refused");
        assert_eq!(m, before, "a refused swap changed the table");
        assert!(
            !bindings::menu_is_unreachable(&m),
            "the menu is unreachable after a refused swap"
        );

        // ── the control: move the menu first, and Tab is then takeable ──
        let (mut st, mut s, mut m) = fixture();
        st.page = Page::Bindings;
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(menu))
            .unwrap();
        key(&mut st, &mut s, &mut m, "Enter");
        assert!(key(&mut st, &mut s, &mut m, "KeyB").bindings_changed);
        assert_eq!(bindings::token_in(&m, &table[menu]), "KeyB");
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(attack))
            .unwrap();
        key(&mut st, &mut s, &mut m, "Enter");
        let out = key(&mut st, &mut s, &mut m, "Tab");
        assert!(out.consumed);
        assert!(
            matches!(st.capture, Capture::Idle),
            "Tab was still owned by the menu after the menu moved: {:?}",
            st.capture
        );
        assert_eq!(bindings::token_in(&m, &table[attack]), "Tab");
        assert_eq!(bindings::token_in(&m, &table[menu]), "KeyB");
    }

    /// **The two keys the dialog needs cannot be bound away.**
    #[test]
    fn escape_and_enter_are_refused_as_bindings() {
        let (mut st, mut s, mut m) = fixture();
        st.page = Page::Bindings;
        let table = bindings::rows();
        let attack = table.iter().position(|r| r.id == "attack").unwrap();
        st.focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(attack))
            .unwrap();
        let before = bindings::token_in(&m, &table[attack]);

        key(&mut st, &mut s, &mut m, "Enter");
        // Escape leaves the capture (it is the cancel), and Enter is refused
        // while the capture stays open — the player can still press a real key.
        let out = key(&mut st, &mut s, &mut m, "Enter");
        assert!(!out.bindings_changed);
        assert_eq!(
            st.capture,
            Capture::Awaiting { row: attack },
            "a refused key ended the capture, so the player has to start again"
        );
        let out = key(&mut st, &mut s, &mut m, "KeyK");
        assert!(out.bindings_changed);
        assert_ne!(bindings::token_in(&m, &table[attack]), before);
    }

    /// **A capture eats the arrow keys**, or an arrow key could never be bound.
    #[test]
    fn a_capture_can_bind_the_keys_the_dialog_navigates_with() {
        let (mut st, mut s, mut m) = fixture();
        st.page = Page::Bindings;
        let table = bindings::rows();
        let roll = table.iter().position(|r| r.id == "roll").unwrap();
        let focus = rows(&st, &s, &m)
            .iter()
            .position(|r| r.id == RowId::Binding(roll))
            .unwrap();
        st.focus = focus;
        key(&mut st, &mut s, &mut m, "Enter");
        let out = key(&mut st, &mut s, &mut m, "ArrowDown");
        assert!(out.consumed);
        assert_eq!(
            st.focus, focus,
            "the arrow key moved the cursor instead of reaching the capture"
        );
        // Down is Move Back's second key in the shipped table, so the capture
        // answers with a conflict rather than a bind — which is itself the
        // proof that the key reached the capture at all.
        let Capture::Conflict { owners, .. } = st.capture.clone() else {
            panic!("the arrow key never reached the capture: {:?}", st.capture);
        };
        assert_eq!(owners, ["move_y-"]);
        let out = key(&mut st, &mut s, &mut m, "Enter");
        assert!(out.bindings_changed);
        assert_eq!(bindings::token_in(&m, &table[roll]), "ArrowDown");
        assert_eq!(
            bindings::token_in(
                &m,
                &table[table.iter().position(|r| r.id == "move_y-").unwrap()]
            ),
            "KeyS",
            "Move Back lost its primary key instead of the one the conflict named"
        );
    }

    /// The four controls with no consumer say so on their own row.
    #[test]
    fn a_control_with_no_consumer_says_so_where_the_player_can_see_it() {
        let (mut st, s, m) = fixture();
        let mut st2 = st.clone();
        st2.page = Page::Bindings;
        st.page = Page::Bindings;
        let shown = rows(&st, &s, &m);
        let noted: Vec<&str> = shown
            .iter()
            .filter(|r| r.note.as_deref() == Some("not wired to anything yet"))
            .map(|r| r.label.as_str())
            .collect();
        assert!(noted.contains(&"Attack"), "{noted:?}");
        assert!(noted.contains(&"Reload"), "{noted:?}");
        assert!(noted.contains(&"Inventory"), "{noted:?}");
        // …and a control that IS wired does not claim to be dead.
        assert!(!noted.contains(&"Jump"), "{noted:?}");
        assert_eq!(
            rows(&st2, &s, &m),
            shown,
            "the projection is not a function of anything but its inputs"
        );
    }

    /// **AND SO DOES A SETTING WITH NO CONSUMER** (I5 audit, A5).
    ///
    /// The video page's three rows are stored and applied by nothing, and until
    /// this arm they said so only in a ledger. That is the wave's own dead-key
    /// defect one level up: a player who changes the resolution and sees nothing
    /// happen cannot tell a preference from a bug, and the wave fixed exactly
    /// that for the audio and controls pages by WIRING them.
    ///
    /// The counter-claim is the half that matters: every row on the other three
    /// pages **is** live, so the note must appear on three rows and no more. The
    /// day the video half lands, this arm fails and the note comes off.
    #[test]
    fn a_setting_that_reaches_nothing_says_so_on_its_own_row() {
        let (mut st, s, m) = fixture();
        let mut noted: Vec<String> = Vec::new();
        let mut live = 0usize;
        for page in Page::ALL {
            st.page = page;
            for row in rows(&st, &s, &m) {
                if row.note.as_deref() == Some(STORED_NOT_APPLIED_NOTE) {
                    noted.push(format!("{page:?}/{}", row.label));
                } else if !matches!(row.id, RowId::Page) {
                    live += 1;
                }
            }
        }
        assert_eq!(
            noted,
            ["Video/Window Mode", "Video/Resolution", "Video/Quality"],
            "the stored-not-applied note is on the wrong set of rows"
        );
        assert!(
            live > 20,
            "only {live} rows claim to be live, so the note above is not the exception it says it is"
        );
        // ASCII, because the built-in font is 0x20..0x7F and a note with a
        // character outside it draws as a hole where a sentence should be.
        assert!(
            STORED_NOT_APPLIED_NOTE
                .chars()
                .all(|c| c.is_ascii_graphic() || c == ' '),
            "the note has a character the built-in font cannot draw"
        );
    }
}
