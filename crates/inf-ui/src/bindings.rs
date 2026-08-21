//! **The bindings table**: the rows a rebinding UI shows, the token a row's cell
//! carries, and the conflict a new token would cause.
//!
//! # A row is not an action
//!
//! UE's rebinding screen lists *Move Forward*, *Move Back*, *Move Left* and
//! *Move Right* as four rows, and this engine's `move_y` is one **axis** with a
//! `+1` key and a `-1` key. A table built on actions alone could not offer W, A,
//! S or D at all — which is most of what a player rebinds. So a row is either an
//! action or **one sign of one axis**, it has a stable id, and the id is what a
//! settings file stores.
//!
//! # The token
//!
//! A cell's value is a **name**, never an index: `"KeyW"`, `"Mouse:Left"`, or
//! the empty string for unbound. That is what makes the override table a
//! name-keyed TOML file with no version ladder — a key that this build has never
//! heard of survives a round trip instead of being renumbered into a different
//! control.
//!
//! # Gamepad bindings are not rows
//!
//! A player rebinding with a keyboard is not rebinding their pad, so every door
//! here names the *desk* source and `inf_input::InputMap` leaves the gamepad
//! sources exactly where the shipped table put them.

use std::collections::BTreeMap;

use inf_input::{ActionSource, InputMap, MouseButton};

/// What a bindings row is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    /// A digital action: `crouch`, `jump`, `interact`.
    Action,
    /// The positive half of an analog axis: `move_y` at `+1` is "forward".
    AxisPositive,
    /// The negative half: `move_y` at `-1` is "back".
    AxisNegative,
}

/// One row of the bindings table.
#[derive(Clone, Debug, PartialEq)]
pub struct BindingRow {
    /// The stable id a settings file stores: the action name, or the axis name
    /// with a trailing `+`/`-`.
    pub id: String,
    /// What the player reads.
    pub label: &'static str,
    /// The action or axis name in the input map.
    pub name: &'static str,
    /// Which half of it.
    pub kind: RowKind,
}

impl BindingRow {
    fn action(label: &'static str, name: &'static str) -> Self {
        Self {
            id: name.to_string(),
            label,
            name,
            kind: RowKind::Action,
        }
    }

    fn axis(label: &'static str, name: &'static str, positive: bool) -> Self {
        Self {
            id: format!("{name}{}", if positive { '+' } else { '-' }),
            label,
            name,
            kind: if positive {
                RowKind::AxisPositive
            } else {
                RowKind::AxisNegative
            },
        }
    }
}

/// **The table the settings dialog and the editor's preferences both render.**
///
/// One list, in the order a player reads it — movement first, then the stance
/// verbs, then the combat verbs, then the session. The order is the *table's*
/// and not the map's `BTreeMap` order, because "Move Forward, Move Back, Move
/// Left, Move Right" is a shape a player recognizes and "aim, attack, crouch,
/// dive…" is an alphabet.
pub fn rows() -> Vec<BindingRow> {
    use inf_input::actions as game;
    vec![
        BindingRow::axis("Move Forward", "move_y", true),
        BindingRow::axis("Move Back", "move_y", false),
        BindingRow::axis("Move Left", "move_x", false),
        BindingRow::axis("Move Right", "move_x", true),
        BindingRow::action("Jump", "jump"),
        BindingRow::action("Sprint", "sprint"),
        BindingRow::action("Walk", "walk"),
        BindingRow::action("Crouch / Prone", "crouch"),
        BindingRow::action("Prone (direct)", "prone"),
        BindingRow::action("Roll", "roll"),
        BindingRow::action("Dive (direct)", "dive"),
        BindingRow::action("Interact", "interact"),
        BindingRow::action("Aim", "aim"),
        // Literals, like every other movement name in this list: `attack` and
        // `reload` became **intents** in I6 and their constants moved to
        // `inf_ecs::movement::actions`, which this crate cannot name (it does
        // not depend on the world model, deliberately — a UI is not a
        // simulation). `inf-player`'s vocabulary arm holds the spellings
        // together.
        BindingRow::action("Attack", "attack"),
        BindingRow::action("Reload", "reload"),
        BindingRow::action("Inventory", game::INVENTORY),
        BindingRow::action("Fly", "fly"),
        BindingRow::action("Menu", game::MENU),
    ]
}

/// The token a source is written as in a settings file and shown as in a cell.
///
/// A **name**, so the file is readable and a build that meets an unknown one
/// leaves it alone rather than renumbering it into a different control.
pub fn token_of(source: &ActionSource) -> String {
    match source {
        ActionSource::Key(code) => code.clone(),
        ActionSource::MouseButton(b) => format!("Mouse:{b:?}"),
        ActionSource::GamepadButton(b) => format!("Pad:{b:?}"),
    }
}

/// The inverse of [`token_of`] for the two source kinds a desk rebinding can
/// produce. `None` for a blank token (which means *unbound*) and for a token
/// this build does not recognize — the second case is a file written by a newer
/// build, and reading it as "some other key" would be worse than reading it as
/// nothing.
pub fn source_of(token: &str) -> Option<ActionSource> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    if let Some(name) = token.strip_prefix("Mouse:") {
        return mouse_of(name).map(ActionSource::MouseButton);
    }
    if token.starts_with("Pad:") {
        // A pad token is legal in a file (it is what `token_of` writes for a
        // gamepad source) and is not a desk rebinding, so it is not applied.
        return None;
    }
    Some(ActionSource::Key(token.to_string()))
}

fn mouse_of(name: &str) -> Option<MouseButton> {
    Some(match name {
        "Left" => MouseButton::Left,
        "Middle" => MouseButton::Middle,
        "Right" => MouseButton::Right,
        "Back" => MouseButton::Back,
        "Forward" => MouseButton::Forward,
        _ => return None,
    })
}

/// What `row` is bound to in `map` right now, as a token. The empty string means
/// unbound, which a table shows as such rather than as a blank cell nobody can
/// tell from a missing row.
pub fn token_in(map: &InputMap, row: &BindingRow) -> String {
    match row.kind {
        RowKind::Action => map.desk_source(row.name).map(token_of).unwrap_or_default(),
        RowKind::AxisPositive => map.axis_key(row.name, true).unwrap_or("").to_string(),
        RowKind::AxisNegative => map.axis_key(row.name, false).unwrap_or("").to_string(),
    }
}

/// Bind `row` to `token` in `map`, or clear it when the token is blank.
///
/// Returns whether anything changed, so a caller can skip a write.
pub fn set_row(map: &mut InputMap, row: &BindingRow, token: &str) -> bool {
    let before = token_in(map, row);
    match (row.kind, source_of(token)) {
        (RowKind::Action, Some(s)) => {
            map.set_desk_source(row.name, s);
        }
        (RowKind::Action, None) => {
            map.clear_desk_source(row.name);
        }
        (kind, Some(ActionSource::Key(code))) => {
            map.set_axis_key(row.name, kind == RowKind::AxisPositive, code);
        }
        (kind, None) => {
            map.clear_axis_key(row.name, kind == RowKind::AxisPositive);
        }
        // An axis cannot take a mouse BUTTON through this table: a bindings row
        // for "Move Forward" offers a key, and the map's own axis door takes a
        // key code. Refused as a no-op rather than silently binding nothing.
        (_, Some(_)) => return false,
    }
    token_in(map, row) != before
}

/// **Every token `row` answers to**, in binding order.
///
/// A row *shows* its first key and can be triggered by any of them: the shipped
/// table binds Move Forward to W **and** to Up. A conflict check that read only
/// the displayed cell would let a player bind Up to something else and leave two
/// verbs on one key — which is the exact failure the conflict dialog exists to
/// prevent, arrived at through the cell instead of through the map.
pub fn tokens_in(map: &InputMap, row: &BindingRow) -> Vec<String> {
    match row.kind {
        RowKind::Action => map
            .desk_sources(row.name)
            .into_iter()
            .map(token_of)
            .collect(),
        RowKind::AxisPositive => map
            .axis_keys(row.name, true)
            .into_iter()
            .map(str::to_string)
            .collect(),
        RowKind::AxisNegative => map
            .axis_keys(row.name, false)
            .into_iter()
            .map(str::to_string)
            .collect(),
    }
}

/// **Take one exact token off `row`**, returning whether it was there.
///
/// What a conflict swap does to the previous owner: it removes the *key the
/// dialog named*, not whichever key happened to be first. A row that had two
/// keys keeps the other and shows it, which is the honest outcome — the player
/// took W from Move Forward and Move Forward still has Up.
pub fn unbind_token(map: &mut InputMap, row: &BindingRow, token: &str) -> bool {
    let Some(source) = source_of(token) else {
        return false;
    };
    match (row.kind, &source) {
        (RowKind::Action, s) => map.remove_action_source(row.name, s),
        (_, ActionSource::Key(code)) => map.remove_axis_key(row.name, code),
        _ => false,
    }
}

/// **The row a player may never be left without** — the way back into the
/// settings dialog (I5 audit, A1).
///
/// Everything else on the table can be cleared, swapped away or bound to
/// nonsense, and the player can undo it *from the dialog*. This row is the door
/// to the dialog, so unbinding it is the one edit that cannot be undone: the
/// bindings live in a settings file that outlives the process, so a cleared menu
/// key is a game that can never be reconfigured again.
pub const MENU_ROW_ID: &str = inf_input::actions::MENU;

/// **Whether `map` leaves a keyboard player no way to open the settings dialog.**
///
/// Reads the *desk* sources only, which is the point: the shipped table also
/// binds the menu to a gamepad button, and a pad the player may not own — with a
/// dialog that has no gamepad navigation yet — is not a way back in.
pub fn menu_is_unreachable(map: &InputMap) -> bool {
    let table = rows();
    let Some(row) = table.iter().find(|r| r.id == MENU_ROW_ID) else {
        // No menu row means no dialog to be locked out of.
        return false;
    };
    tokens_in(map, row).is_empty()
}

/// **Apply `edit`, and take it back if it locked the player out** (I5 audit, A1).
///
/// The one rule, so the in-game capture and the editor's preferences panel
/// cannot disagree about it — the same discipline [`apply_row`] itself carries.
/// A refusal is a **value**: nothing changes and the caller reports it, rather
/// than the edit half-landing or the dialog closing on a map with no way back.
///
/// Returns whether the edit was applied *and* changed something. A refusal is a
/// `false` with `map` exactly as it was, which is why the whole map is cloned
/// first: an edit that touches several rows (a conflict swap takes a token off
/// every owner before binding it) has to be undone as one.
pub fn guarded(map: &mut InputMap, edit: impl FnOnce(&mut InputMap) -> bool) -> bool {
    let before = map.clone();
    let changed = edit(map);
    if menu_is_unreachable(map) {
        *map = before;
        return false;
    }
    changed
}

/// **Give the menu its shipped key back when a map has none** (I5 audit, A1).
///
/// The edit doors refuse to *create* this state; this is what a host calls on
/// the map it just built from a settings file, because a file is not only
/// written by those doors — it survives a build that had no guard, it can be
/// hand-edited, and it can name a control this build spells differently.
/// Returns whether anything was restored, so a host can say so.
///
/// It restores rather than refusing to boot for the reason the whole settings
/// door does: a preferences file is not a boot config, and a player whose
/// bindings file is odd should get their game, told.
pub fn restore_menu_if_unreachable(map: &mut InputMap) -> bool {
    if !menu_is_unreachable(map) {
        return false;
    }
    let table = rows();
    let Some(row) = table.iter().find(|r| r.id == MENU_ROW_ID) else {
        return false;
    };
    let shipped = token_in(&inf_input::default_map(), row);
    if shipped.is_empty() {
        return false;
    }
    set_row(map, row, &shipped)
}

/// **Who else already uses `token`**, as row ids, excluding `except`.
///
/// The conflict a rebinding UI has to name before it takes a key. Every owner
/// is answered, not the first: a key legitimately means one thing to an axis and
/// another to an action — Space is `jump`, `handbrake` **and** `move_up` in the
/// shipped table — so a dialog that named one would quietly steal the rest.
///
/// Only rows the table *shows* are reported: `handbrake` and `move_up` share
/// Space by design and are not rows, so taking Space for something else does not
/// present the player with a conflict they have no way to resolve.
pub fn conflicts(map: &InputMap, token: &str, except: &str) -> Vec<String> {
    let Some(source) = source_of(token) else {
        return Vec::new();
    };
    let wanted = token_of(&source);
    let mut out = Vec::new();
    for row in rows() {
        if row.id == except {
            continue;
        }
        // **Every** token the row answers to, not the one it displays.
        if tokens_in(map, &row).contains(&wanted) {
            out.push(row.id);
        }
    }
    out
}

/// **The overrides `map` carries against `base`** — only the rows that differ.
///
/// Storing the difference rather than the whole map is the editor keybinding
/// door's rule and it earns its keep the same way: a file that shipped the whole
/// table would freeze a player's bindings at the build they first ran, so a
/// control added later would arrive unbound and a default corrected later would
/// never reach them.
pub fn overrides_from(base: &InputMap, map: &InputMap) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for row in rows() {
        let (a, b) = (token_in(base, &row), token_in(map, &row));
        if a != b {
            out.insert(row.id, b);
        }
    }
    out
}

/// Apply `overrides` onto `map`, ignoring rows this build does not have.
///
/// A row id from a newer build names a control that does not exist here; there
/// is nothing to bind it to and inventing one would be worse than skipping it.
/// Returns how many were applied, so a caller can report the difference between
/// "no overrides" and "none of them meant anything".
pub fn apply_overrides(map: &mut InputMap, overrides: &BTreeMap<String, String>) -> usize {
    let table = rows();
    let mut applied = 0;
    for (id, token) in overrides {
        if let Some(row) = table.iter().find(|r| &r.id == id) {
            set_row(map, row, token);
            applied += 1;
        }
    }
    applied
}

/// **Apply the controls page's look tuning** to `map` (island wave I5).
///
/// The sensitivity is a **multiplier on what the project authored**, read from
/// `base` rather than from `map`: that makes the operation idempotent (applying
/// it twice is applying it once) and it keeps a project that chose its own
/// degrees-per-count. A slider that overwrote the number would make every
/// project's feel identical at the default.
///
/// `invert_y` flips the y channel's **sign against the shipped convention**.
/// The engine reports `+y` down and a look control wants `+pitch` up, so the
/// table binds `look_y` at a negative scale; inverting means the player wants
/// the other one, whatever the project's is.
///
/// The axis NAMES are the caller's, because `inf-ui` must not name
/// `inf-ecs`'s movement vocabulary — the crate that owns both is the one that
/// passes them, which is the same discipline
/// `every_movement_action_the_intent_reads_is_bound` already runs on.
///
/// A non-finite or non-positive sensitivity reads as `1.0`: a slider is not a
/// place to author a mouse that does not move.
pub fn apply_look_tuning(
    map: &mut InputMap,
    base: &InputMap,
    x_axis: &str,
    y_axis: &str,
    sensitivity: f32,
    invert_y: bool,
) {
    let m = if sensitivity.is_finite() && sensitivity > 0.0 {
        sensitivity
    } else {
        1.0
    };
    for (axis, mouse, invert) in [
        (x_axis, inf_input::MouseAxis::X, false),
        (y_axis, inf_input::MouseAxis::Y, invert_y),
    ] {
        let Some(authored) = base.mouse_axis_scale(axis, mouse) else {
            continue;
        };
        if !authored.is_finite() {
            continue;
        }
        let sign = if (authored < 0.0) != invert {
            -1.0
        } else {
            1.0
        };
        map.set_mouse_axis_scale(axis, mouse, sign * authored.abs() * m);
    }
}

/// One row of the table as a surface renders it.
#[derive(Clone, Debug, PartialEq)]
pub struct TableRow {
    /// The stable id a settings file stores.
    pub id: String,
    /// What the player reads.
    pub label: String,
    /// What it is bound to, or empty for unbound.
    pub token: String,
    /// Whether the player has moved it off the shipped table.
    pub overridden: bool,
    /// Whether its consumer exists yet.
    pub wired: bool,
}

/// **The table, resolved** — the shipped map plus a player's overrides.
///
/// One projection, so the in-game dialog and the editor's preferences panel
/// render the same rows in the same order from the same source. A second table
/// in TypeScript is the "two copies across a language boundary" defect the P29.6
/// note beside `default_map` records; this is what stops there being one.
pub fn table(base: &InputMap, overrides: &BTreeMap<String, String>) -> Vec<TableRow> {
    let mut map = base.clone();
    apply_overrides(&mut map, overrides);
    rows()
        .into_iter()
        .map(|row| TableRow {
            token: token_in(&map, &row),
            overridden: overrides.contains_key(&row.id),
            wired: !inf_input::actions::NOT_YET_CONSUMED.contains(&row.name),
            id: row.id,
            label: row.label.to_string(),
        })
        .collect()
}

/// What [`apply_row`] decided.
#[derive(Clone, Debug, PartialEq)]
pub struct ApplyOutcome {
    /// The override set after the edit — unchanged when a conflict blocked it.
    pub overrides: BTreeMap<String, String>,
    /// The row ids that already owned the token, when one did.
    pub conflicts: Vec<String>,
}

/// **Bind `token` to `row_id`, through the same rule the in-game dialog uses.**
///
/// `swap` is the player's answer to the conflict the first call reports: with it
/// `false`, a taken token changes nothing and the owners come back so a surface
/// can name them; with it `true`, the token is taken **off those owners** — the
/// exact token, not their whole binding, so a row with a second key keeps it —
/// and bound here.
///
/// The editor's preferences panel is a second *surface*, not a second rule: it
/// calls this, and so does `crate::menu`'s capture. A key bound to two rows is a
/// key that fires two verbs, and there is one place that can produce one.
pub fn apply_row(
    base: &InputMap,
    overrides: &BTreeMap<String, String>,
    row_id: &str,
    token: &str,
    swap: bool,
) -> ApplyOutcome {
    let table = rows();
    let Some(target) = table.iter().find(|r| r.id == row_id) else {
        return ApplyOutcome {
            overrides: overrides.clone(),
            conflicts: Vec::new(),
        };
    };
    let mut map = base.clone();
    apply_overrides(&mut map, overrides);
    let conflicts = conflicts(&map, token, row_id);
    if !conflicts.is_empty() && !swap {
        return ApplyOutcome {
            overrides: overrides.clone(),
            conflicts,
        };
    }
    // **Refused if it would lock the player out of the dialog** (I5 audit, A1),
    // through the same [`guarded`] door the in-game capture uses. The refusal is
    // reported as a conflict naming [`MENU_ROW_ID`], because that is exactly what
    // it is — the menu row owns the key and this is the one owner a swap may not
    // take it from.
    let applied = guarded(&mut map, |m| {
        for id in &conflicts {
            if let Some(other) = table.iter().find(|r| &r.id == id) {
                unbind_token(m, other, token);
            }
        }
        set_row(m, target, token)
    });
    if !applied && menu_would_be_orphaned(base, overrides, &conflicts, target, token) {
        return ApplyOutcome {
            overrides: overrides.clone(),
            conflicts: vec![MENU_ROW_ID.to_string()],
        };
    }
    ApplyOutcome {
        overrides: overrides_from(base, &map),
        conflicts: Vec::new(),
    }
}

/// Whether the edit [`apply_row`] just tried would have left the menu with no
/// desk key — asked again on a throwaway map, because [`guarded`] answers
/// "nothing changed" for a refusal and for a no-op alike and the caller has to
/// tell them apart.
fn menu_would_be_orphaned(
    base: &InputMap,
    overrides: &BTreeMap<String, String>,
    conflicts: &[String],
    target: &BindingRow,
    token: &str,
) -> bool {
    let table = rows();
    let mut probe = base.clone();
    apply_overrides(&mut probe, overrides);
    for id in conflicts {
        if let Some(other) = table.iter().find(|r| &r.id == id) {
            unbind_token(&mut probe, other, token);
        }
    }
    set_row(&mut probe, target, token);
    menu_is_unreachable(&probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every row names something the shipped table has**, and the ids are
    /// unique. A row whose name is a typo would render, accept a key and bind
    /// nothing.
    #[test]
    fn every_row_names_a_real_control_and_the_ids_are_unique() {
        let map = inf_input::default_map();
        let rows = rows();
        let mut ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "two rows share an id: {ids:?}");
        for row in &rows {
            match row.kind {
                RowKind::Action => assert!(
                    map.action_names().any(|n| n == row.name),
                    "row `{}` names the action `{}`, which the shipped table does not bind",
                    row.label,
                    row.name
                ),
                _ => assert!(
                    map.axis_names().any(|n| n == row.name),
                    "row `{}` names the axis `{}`, which the shipped table does not bind",
                    row.label,
                    row.name
                ),
            }
            assert!(
                !token_in(&map, row).is_empty(),
                "row `{}` renders an empty cell on a fresh install",
                row.label
            );
        }
        // …and **every row is wired**, which is what island wave I6 bought: the
        // four controls the owner's table bound against consumers that did not
        // exist — reload, attack, inventory and the weapon wheel — all have
        // one now, so `NOT_YET_CONSUMED` is empty and no row carries the note.
        //
        // The arm is written as a **count against the rows**, not as
        // `is_empty()`: the day a later wave binds a control ahead of its
        // consumer, that control has to be a row too (a control a player cannot
        // rebind is one they cannot get out of the way either), and this is what
        // says so.
        let unwired: Vec<&str> = inf_input::actions::NOT_YET_CONSUMED.to_vec();
        println!("controls bound with no consumer: {unwired:?}");
        for a in unwired {
            assert!(
                rows.iter().any(|r| r.name == a),
                "`{a}` is bound, has no consumer, and has no row either"
            );
        }
        assert!(
            rows.iter()
                .all(|r| !inf_input::actions::NOT_YET_CONSUMED.contains(&r.name)),
            "a row is still unwired"
        );
    }

    /// **The round trip a rebinding actually takes**: edit the map, derive the
    /// overrides, apply them to a fresh map, and get the same table back.
    #[test]
    fn a_rebinding_survives_being_written_as_overrides_and_read_back() {
        let base = inf_input::default_map();
        let mut edited = base.clone();
        let rows = rows();
        let crouch = rows.iter().find(|r| r.id == "crouch").unwrap();
        let forward = rows.iter().find(|r| r.id == "move_y+").unwrap();
        assert!(set_row(&mut edited, crouch, "KeyB"));
        assert!(set_row(&mut edited, forward, "KeyT"));
        // Setting the same token again changes nothing, so a UI can skip a save.
        assert!(!set_row(&mut edited, crouch, "KeyB"));

        let overrides = overrides_from(&base, &edited);
        assert_eq!(overrides.len(), 2, "{overrides:?}");
        assert_eq!(overrides["crouch"], "KeyB");
        assert_eq!(overrides["move_y+"], "KeyT");

        let mut fresh = inf_input::default_map();
        assert_eq!(apply_overrides(&mut fresh, &overrides), 2);
        for row in &rows {
            assert_eq!(
                token_in(&fresh, row),
                token_in(&edited, row),
                "row `{}` did not survive the round trip",
                row.label
            );
        }

        // An UNBOUND row is a value: the empty token round-trips as "cleared",
        // which is the difference between a row the player cleared and one they
        // never touched.
        let mut cleared = base.clone();
        assert!(set_row(&mut cleared, crouch, ""));
        let overrides = overrides_from(&base, &cleared);
        assert_eq!(overrides["crouch"], "");
        let mut fresh = inf_input::default_map();
        apply_overrides(&mut fresh, &overrides);
        assert_eq!(token_in(&fresh, crouch), "");

        // A row id from a newer build is skipped rather than guessed at.
        let mut newer = BTreeMap::new();
        newer.insert("grapple".to_string(), "KeyG".to_string());
        let mut fresh = inf_input::default_map();
        assert_eq!(apply_overrides(&mut fresh, &newer), 0);
        assert_eq!(fresh, inf_input::default_map());
    }

    /// **A conflict names the row that already owns the key.**
    #[test]
    fn taking_a_bound_key_reports_who_had_it() {
        let map = inf_input::default_map();
        // W is Move Forward. Taking it for Interact must say so.
        let owners = conflicts(&map, "KeyW", "interact");
        assert_eq!(owners, ["move_y+"], "{owners:?}");
        // A row does not conflict with itself.
        assert!(conflicts(&map, "KeyW", "move_y+").is_empty());
        // A free key conflicts with nothing.
        assert!(conflicts(&map, "KeyJ", "interact").is_empty());
        // The mouse half works too: LMB is Attack.
        assert_eq!(conflicts(&map, "Mouse:Left", "aim"), ["attack"]);
        // A blank token is "unbind", which cannot conflict with anything.
        assert!(conflicts(&map, "", "interact").is_empty());
    }

    /// **THE EDITOR'S PANEL AND THE IN-GAME DIALOG SHARE ONE RULE** (I5).
    ///
    /// `apply_row` is what a second *surface* calls; it is not a second rule.
    /// The arm measures the three things a surface needs and the one it must not
    /// be able to do: report a conflict, refuse until asked, take the exact token
    /// off the previous owner on a swap — and never leave a key bound to two
    /// rows.
    #[test]
    fn the_editors_surface_reports_a_conflict_and_swaps_only_when_asked() {
        let base = inf_input::default_map();
        let empty = BTreeMap::new();

        // A free key binds straight away, and the override set is the DIFFERENCE.
        let out = apply_row(&base, &empty, "interact", "KeyJ", false);
        assert!(out.conflicts.is_empty());
        assert_eq!(out.overrides.len(), 1);
        assert_eq!(out.overrides["interact"], "KeyJ");

        // A taken key reports its owner and changes NOTHING.
        let out = apply_row(&base, &empty, "interact", "KeyW", false);
        assert_eq!(out.conflicts, ["move_y+"]);
        assert_eq!(out.overrides, empty, "a refused edit changed the table");

        // Asked, it swaps — and takes the exact token off the owner, leaving
        // that row its other key.
        let out = apply_row(&base, &empty, "interact", "KeyW", true);
        assert!(out.conflicts.is_empty());
        let mut after = base.clone();
        apply_overrides(&mut after, &out.overrides);
        let table = rows();
        let interact = table.iter().find(|r| r.id == "interact").unwrap();
        let forward = table.iter().find(|r| r.id == "move_y+").unwrap();
        assert_eq!(token_in(&after, interact), "KeyW");
        assert_eq!(
            token_in(&after, forward),
            "ArrowUp",
            "the swap took Move Forward's other key too: {:?}",
            tokens_in(&after, forward)
        );
        assert!(
            conflicts(&after, "KeyW", "interact").is_empty(),
            "W is still bound to two rows, so one press fires two verbs"
        );

        // **The SECOND key, which is where the two candidate rules differ.**
        // (I5 audit, A4.) The swap above takes `KeyW`, which is Move Forward's
        // *first* key — and "remove the exact token" and "remove whichever desk
        // source is first" are the same expression there, so that assertion's own
        // failure message names a defect it could not see. Taking `ArrowUp`
        // separates them: the exact rule leaves `KeyW`, the first-source rule
        // takes it.
        let out = apply_row(&base, &empty, "interact", "ArrowUp", true);
        assert!(out.conflicts.is_empty());
        let mut after = base.clone();
        apply_overrides(&mut after, &out.overrides);
        assert_eq!(token_in(&after, interact), "ArrowUp");
        assert_eq!(
            token_in(&after, forward),
            "KeyW",
            "the swap took Move Forward's FIRST key instead of the one it named: {:?}",
            tokens_in(&after, forward)
        );

        // A row id this build has never heard of is a no-op rather than a guess.
        let out = apply_row(&base, &empty, "grapple", "KeyG", true);
        assert_eq!(out.overrides, empty);
    }

    /// **THE EDITOR'S SURFACE CANNOT LOCK A PROJECT'S PLAYERS OUT EITHER** (I5
    /// audit, A1).
    ///
    /// The in-game dialog's own refusal is measured in `menu`'s arms; this is the
    /// half that says the rule lives in [`guarded`] rather than in the dialog. An
    /// author setting a project's `game_bindings` from Editor Preferences reaches
    /// the same edit through [`apply_row`], and a project shipped with no menu key
    /// is a game *nobody* can open the settings of.
    ///
    /// The refusal is a **value**: the override set is unchanged and the blocking
    /// row is named, which is the same shape a conflict takes.
    #[test]
    fn the_editors_surface_refuses_an_edit_that_orphans_the_menu() {
        let base = inf_input::default_map();
        let empty = BTreeMap::new();

        // Taking Tab for Attack, with the swap confirmed.
        let out = apply_row(&base, &empty, "attack", "Tab", true);
        assert_eq!(
            out.conflicts,
            [MENU_ROW_ID],
            "the swap was not refused: {out:?}"
        );
        assert_eq!(out.overrides, empty, "a refused edit changed the table");

        // Clearing the menu row outright.
        let out = apply_row(&base, &empty, MENU_ROW_ID, "", true);
        assert_eq!(out.conflicts, [MENU_ROW_ID]);
        assert_eq!(out.overrides, empty);

        // **The control**: moving the menu is fine, and then Tab is takeable —
        // so the rule is about the STATE, not about the key or the row.
        let moved = apply_row(&base, &empty, MENU_ROW_ID, "KeyB", false);
        assert!(moved.conflicts.is_empty());
        assert_eq!(moved.overrides[MENU_ROW_ID], "KeyB");
        let out = apply_row(&base, &moved.overrides, "attack", "Tab", true);
        assert!(out.conflicts.is_empty(), "{out:?}");
        let mut after = base.clone();
        apply_overrides(&mut after, &out.overrides);
        assert!(!menu_is_unreachable(&after));
        let table = rows();
        let attack = table.iter().find(|r| r.id == "attack").unwrap();
        assert_eq!(token_in(&after, attack), "Tab");
    }

    /// **A settings FILE that already carries the lockout is healed on load**
    /// (I5 audit, A1).
    ///
    /// The edit doors refuse to create the state; a file is the case they cannot
    /// cover, because it outlives the build that wrote it. Restoring the shipped
    /// key is the only answer that leaves the player somewhere to change their
    /// mind, and it is the settings door's own doctrine — a *preferences* file
    /// that is odd leaves the player playing, told.
    #[test]
    fn a_map_with_no_menu_key_is_given_the_shipped_one_back() {
        let base = inf_input::default_map();
        assert!(!menu_is_unreachable(&base), "the shipped table has no menu");
        assert!(
            !restore_menu_if_unreachable(&mut base.clone()),
            "a healthy map was rewritten"
        );

        // The shape a hand-edited file takes: the row id with an empty token.
        let mut overrides = BTreeMap::new();
        overrides.insert(MENU_ROW_ID.to_string(), String::new());
        let mut map = base.clone();
        apply_overrides(&mut map, &overrides);
        assert!(
            menu_is_unreachable(&map),
            "the fixture did not reproduce the lockout, so the heal below proves nothing"
        );
        assert!(restore_menu_if_unreachable(&mut map));
        let table = rows();
        let menu = table.iter().find(|r| r.id == MENU_ROW_ID).unwrap();
        assert_eq!(token_in(&map, menu), "Tab");
        // Idempotent, and it restores ONLY the menu — a player's other overrides
        // are theirs.
        assert!(!restore_menu_if_unreachable(&mut map));
        let mut both = base.clone();
        overrides.insert("crouch".to_string(), "KeyB".to_string());
        apply_overrides(&mut both, &overrides);
        restore_menu_if_unreachable(&mut both);
        let crouch = table.iter().find(|r| r.id == "crouch").unwrap();
        assert_eq!(token_in(&both, crouch), "KeyB");
    }

    /// **The look tuning multiplies what the project authored, and is
    /// idempotent** (I5).
    ///
    /// The mutation this kills: reading the *current* map instead of the base,
    /// which compounds — two frames of a held slider would square the
    /// sensitivity and a settings file re-applied on every load would run away.
    #[test]
    fn the_look_tuning_scales_the_authored_sensitivity_and_can_be_reapplied() {
        let base = inf_input::default_map();
        let shipped_x = base
            .mouse_axis_scale("look_x", inf_input::MouseAxis::X)
            .expect("the shipped table binds the mouse to look_x");
        let shipped_y = base
            .mouse_axis_scale("look_y", inf_input::MouseAxis::Y)
            .expect("…and to look_y");
        println!("the shipped look is {shipped_x} / {shipped_y} degrees per count");
        assert!(shipped_x > 0.0 && shipped_y < 0.0, "the y channel inverts");

        let mut map = base.clone();
        apply_look_tuning(&mut map, &base, "look_x", "look_y", 2.0, false);
        assert_eq!(
            map.mouse_axis_scale("look_x", inf_input::MouseAxis::X),
            Some(shipped_x * 2.0)
        );
        assert_eq!(
            map.mouse_axis_scale("look_y", inf_input::MouseAxis::Y),
            Some(shipped_y * 2.0)
        );
        // Idempotent: it reads the BASE, so a second application is the first.
        let once = map.clone();
        apply_look_tuning(&mut map, &base, "look_x", "look_y", 2.0, false);
        assert_eq!(map, once, "the tuning compounded");

        // Invert flips the y channel's sign against the shipped convention and
        // leaves x alone.
        let mut map = base.clone();
        apply_look_tuning(&mut map, &base, "look_x", "look_y", 1.0, true);
        assert_eq!(
            map.mouse_axis_scale("look_y", inf_input::MouseAxis::Y),
            Some(-shipped_y)
        );
        assert_eq!(
            map.mouse_axis_scale("look_x", inf_input::MouseAxis::X),
            Some(shipped_x)
        );

        // A hostile sensitivity reads as 1, and an axis the project did not bind
        // to the mouse is left alone rather than given one.
        for bad in [f32::NAN, 0.0, -3.0, f32::INFINITY] {
            let mut map = base.clone();
            apply_look_tuning(&mut map, &base, "look_x", "look_y", bad, false);
            assert_eq!(map, base, "sensitivity {bad} moved the table");
        }
        let mut bare = InputMap::new();
        bare.bind_axis_key("look_x", "KeyL", 1.0);
        let before = bare.clone();
        apply_look_tuning(&mut bare, &InputMap::new(), "look_x", "look_y", 4.0, true);
        assert_eq!(bare, before, "a keyboard-only look axis was given a mouse");
    }

    /// The projection a surface renders, and the two flags it needs.
    #[test]
    fn the_table_says_what_is_overridden_and_what_is_wired() {
        let base = inf_input::default_map();
        let mut overrides = BTreeMap::new();
        overrides.insert("crouch".to_string(), "KeyB".to_string());
        let t = table(&base, &overrides);
        assert_eq!(t.len(), rows().len());
        let crouch = t.iter().find(|r| r.id == "crouch").unwrap();
        assert_eq!(crouch.token, "KeyB");
        assert!(crouch.overridden);
        assert!(crouch.wired);
        let jump = t.iter().find(|r| r.id == "jump").unwrap();
        assert_eq!(jump.token, "Space");
        assert!(!jump.overridden, "an untouched row claims to be overridden");
        // **…and every row is wired as of island wave I6.** The four the owner's
        // table bound ahead of their consumers — attack, reload, inventory and
        // the weapon wheel — all have one now, so `wired` is true everywhere and
        // the flag reads what it means rather than what it was minted for.
        let unwired: Vec<&str> = t
            .iter()
            .filter(|r| !r.wired)
            .map(|r| r.id.as_str())
            .collect();
        println!("rows still claiming no consumer: {unwired:?}");
        assert!(unwired.is_empty(), "{unwired:?}");
        // The flag is still DERIVED and not hard-coded: it reads
        // `NOT_YET_CONSUMED`, so a control put back on that list shows up here
        // without this arm being touched.
        let attack = t.iter().find(|r| r.id == "attack").unwrap();
        assert_eq!(
            attack.wired,
            !inf_input::actions::NOT_YET_CONSUMED.contains(&"attack")
        );
    }

    /// Tokens are names, and the two directions agree for every source a desk
    /// rebinding can produce.
    #[test]
    fn a_token_is_a_name_and_the_two_directions_agree() {
        for s in [
            ActionSource::Key("KeyW".into()),
            ActionSource::MouseButton(MouseButton::Left),
            ActionSource::MouseButton(MouseButton::Forward),
        ] {
            let t = token_of(&s);
            assert_eq!(source_of(&t), Some(s.clone()), "token `{t}`");
        }
        // A pad source has a token (so a file can carry one) and is not a desk
        // rebinding, so it does not come back.
        let pad = ActionSource::GamepadButton(inf_input::GamepadButton::South);
        assert_eq!(token_of(&pad), "Pad:South");
        assert_eq!(source_of("Pad:South"), None);
        // Blank means unbound; an unknown mouse name means nothing rather than
        // some other button.
        assert_eq!(source_of("  "), None);
        assert_eq!(source_of("Mouse:Thumbwheel"), None);
    }
}
