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
        BindingRow::action("Attack", game::ATTACK),
        BindingRow::action("Reload", game::RELOAD),
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
        if tokens_in(map, &row).iter().any(|t| *t == wanted) {
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
    for id in &conflicts {
        if let Some(other) = table.iter().find(|r| &r.id == id) {
            unbind_token(&mut map, other, token);
        }
    }
    set_row(&mut map, target, token);
    ApplyOutcome {
        overrides: overrides_from(base, &map),
        conflicts: Vec::new(),
    }
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
        // …and the four the owner's table binds against consumers that do not
        // exist yet are all rows, because a control a player cannot rebind is
        // one they cannot get out of the way either.
        for a in inf_input::actions::NOT_YET_CONSUMED {
            assert!(
                a == inf_input::actions::WEAPON_SWITCH || rows.iter().any(|r| r.name == a),
                "`{a}` is bound and has no row"
            );
        }
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

        // A row id this build has never heard of is a no-op rather than a guess.
        let out = apply_row(&base, &empty, "grapple", "KeyG", true);
        assert_eq!(out.overrides, empty);
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
        // …and the four with no consumer say so.
        let attack = t.iter().find(|r| r.id == "attack").unwrap();
        assert!(!attack.wired, "attack claims a consumer it does not have");
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
