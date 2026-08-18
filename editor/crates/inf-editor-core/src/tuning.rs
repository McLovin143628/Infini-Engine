//! **Live tuning during Simulate** — pillar S4 (P29.5).
//!
//! §13's claim is that machine and movement parameters are editable **during**
//! Simulate, which UE cannot do for an AnimBlueprint. This module is the door,
//! and it is worth being exact about what was already true and what is new,
//! because the honest version is a smaller headline and a better mechanism.
//!
//! # What was already true
//!
//! A reflected component field written through `SceneDoc::write_prop` reaches the
//! **live** world, and during Simulate the live world *is* the document's world.
//! So a Details-panel edit to `CharacterMovement::walk_speed_mps` has changed a
//! running character since P29.3 — that is P23.6's edit-during-Simulate ruling
//! paying out, and nothing here replaces it.
//!
//! # What was not
//!
//! Three things, and each is a clause of S4:
//!
//! 1. **The step boundary was an accident, not a contract.** Ring 2 happens to
//!    hold the document lock for a whole tick, so an edit could not land
//!    mid-step — but nothing said so, and nothing would have failed if that
//!    changed. A tuned run has to remain a deterministic sequence of fixed
//!    steps or the whole S5 replay claim leaks. Tunes are therefore **queued**
//!    and drained at the top of [`SimSession::fixed_step`], before anything
//!    reads anything.
//! 2. **Nothing survived Stop.** `SimSession::exit` restores the snapshot, which
//!    is right for a *run* and wrong for a *tune*: an author who spent a minute
//!    finding the right sprint speed lost it by pressing the button that ends
//!    the test. [`TuneScope::Keep`] replays the tune onto the document after the
//!    restore, as one ordinary undoable edit.
//! 3. **A state machine's parameters had no editor door at all.** The `anim.*`
//!    kit (P29.4) is a *gameplay* door — a Blueprint's, reached from inside the
//!    running program. An author watching a machine could not set `speed` or arm
//!    a trigger from the editor to see what the graph does. [`Tune::Param`] and
//!    [`Tune::Trigger`] are that, through the same Ring-0 bridge the kit uses.
//!
//! # One door, and the player does not have it
//!
//! [`apply_tune`] is the only thing that applies one, and it lives in **Ring 1**.
//! `runtime/inf-player` links neither this module nor a queue that could hold
//! one, and `tests/player_has_no_tuning_door.rs` is the gate that keeps it that
//! way — because the PIE == shipping discipline says a tuned Simulate must be a
//! *tuned document* played identically by both hosts, never a host that can be
//! nudged while the other cannot.

use inf_ecs::PropValue;
use uuid::Uuid;

use crate::scene::SceneDoc;

/// The reflect type path of the movement component — the one a tuning UI edits
/// almost every time.
///
/// Spelled here so a caller does not have to know `bevy_reflect`'s naming, and
/// asserted against the registry by this module's own test: a renamed component
/// would otherwise turn every movement tune into a silent no-op.
pub const MOVEMENT_TYPE_PATH: &str = "inf_ecs::components::CharacterMovement";

/// One tuning edit, queued during Simulate and applied on the **next** fixed
/// step.
#[derive(Clone, Debug, PartialEq)]
pub enum Tune {
    /// A reflected component field, by `bevy_reflect` access path — the Details
    /// panel's own door, reached from the fixed step instead of from the IPC
    /// thread.
    Field {
        guid: Uuid,
        /// The component's reflect type path, e.g. [`MOVEMENT_TYPE_PATH`].
        type_path: String,
        /// A bare field name (`"walk_speed_mps"`) or a nested path
        /// (`"acceleration.at_run"`) — whatever `inf_ecs::props::write_field`
        /// accepts, because it is the same call.
        path: String,
        value: PropValue,
    },
    /// An animation **parameter**, overlaid on the actor's variables through
    /// `inf_ecs::set_anim_param`.
    Param {
        guid: Uuid,
        name: String,
        value: f64,
    },
    /// An animation **trigger**, armed through `inf_ecs::set_anim_trigger`.
    ///
    /// A separate variant and not a `Param` with a value, for the reason P29.4's
    /// kit gives: a trigger is an *event* and is edge-detected, so "set the
    /// variable to 1" fires once for two presses one step apart.
    Trigger { guid: Uuid, name: String },
}

impl Tune {
    /// The entity this tune is about.
    pub fn guid(&self) -> Uuid {
        match self {
            Tune::Field { guid, .. } | Tune::Param { guid, .. } | Tune::Trigger { guid, .. } => {
                *guid
            }
        }
    }

    /// A movement tunable, by field name.
    pub fn movement(guid: Uuid, field: &str, value: f64) -> Self {
        Tune::Field {
            guid,
            type_path: MOVEMENT_TYPE_PATH.to_string(),
            path: field.to_string(),
            value: PropValue::Number(value),
        }
    }
}

/// How long a tune's effect lasts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TuneScope {
    /// Live for this run only. `Stop` restores the snapshot and the tune is gone
    /// — UE's PIE semantics, and the right default: most tuning is a question,
    /// not an answer.
    #[default]
    Session,
    /// Live now **and** written onto the document when the session ends, as one
    /// ordinary undoable edit. The answer, kept.
    ///
    /// Only meaningful for [`Tune::Field`]: a parameter overlay and an armed
    /// trigger are runtime state (`AnimBridgeRes` is a bevy resource and reaches
    /// no file), so there is nothing about them a document could hold. A `Keep`
    /// on one of those is applied and not kept, which [`kept_edits`] says by
    /// returning nothing rather than by failing.
    Keep,
}

/// **Apply one tune to the live world.** The only place a tune is applied.
///
/// Returns whether it did anything. A refusal is a **value** — an entity that
/// despawned mid-run, a field the component does not have, a parameter on an
/// entity with no machine — because a tuning UI is a live surface over a world
/// that is changing underneath it, and taking a session down over a stale field
/// name is the wrong trade.
///
/// The field write goes through [`SceneDoc::write_prop_preview`], not
/// `edit_set_prop`: a tune is not an authoring edit, and recording one on the
/// undo stack would put a run's experiments in the author's history and mark a
/// document dirty that nobody changed.
pub fn apply_tune(doc: &mut SceneDoc, tune: &Tune) -> bool {
    match tune {
        Tune::Field {
            guid,
            type_path,
            path,
            value,
        } => doc.write_prop_preview(*guid, type_path, path, value),
        Tune::Param { guid, name, value } => {
            inf_ecs::set_anim_param(doc.world_mut(), *guid, name, *value)
        }
        Tune::Trigger { guid, name } => inf_ecs::set_anim_trigger(doc.world_mut(), *guid, name),
    }
}

/// The subset of `kept` that a document can actually hold — see
/// [`TuneScope::Keep`].
///
/// Deduplicated **last-wins** on `(guid, type_path, path)`, because tuning is a
/// drag: an author who moved a slider forty times means the fortieth value, and
/// replaying all forty onto the document would make forty undo steps out of one
/// decision.
pub fn kept_edits(kept: &[Tune]) -> Vec<&Tune> {
    let mut out: Vec<&Tune> = Vec::new();
    for t in kept {
        let Tune::Field {
            guid,
            type_path,
            path,
            ..
        } = t
        else {
            continue;
        };
        let key = (*guid, type_path.as_str(), path.as_str());
        if let Some(slot) = out.iter_mut().find(|e| match e {
            Tune::Field {
                guid: g,
                type_path: tp,
                path: p,
                ..
            } => (*g, tp.as_str(), p.as_str()) == key,
            _ => false,
        }) {
            *slot = t;
        } else {
            out.push(t);
        }
    }
    out
}

/// Write the kept tunes onto the document as ordinary undoable edits.
///
/// Called by `SimSession::exit` **after** the snapshot restore, so the value the
/// author settled on survives the button that ends the run — and `Ctrl+Z` takes
/// it back, because it is one `Edit <field>` step like any other.
pub fn commit_kept(doc: &mut SceneDoc, kept: &[Tune]) -> usize {
    let mut applied = 0;
    for t in kept_edits(kept) {
        let Tune::Field {
            guid,
            type_path,
            path,
            value,
        } = t
        else {
            continue;
        };
        if doc.edit_set_prop(*guid, type_path, path, value) {
            applied += 1;
        }
    }
    applied
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The type path really names a registered component — the whole tuning
    /// surface is a no-op if this string is wrong, and it is a string.
    #[test]
    fn the_movement_type_path_names_a_real_component() {
        let doc = SceneDoc::new();
        let registry = doc.world().registry();
        assert!(
            registry.reflect_component(MOVEMENT_TYPE_PATH).is_some(),
            "{MOVEMENT_TYPE_PATH} is not a registered component"
        );
    }

    /// Last-wins deduplication, per field — a slider drag is one decision.
    #[test]
    fn a_drag_becomes_one_kept_edit_per_field() {
        let g = Uuid::from_u128(1);
        let h = Uuid::from_u128(2);
        let kept = vec![
            Tune::movement(g, "walk_speed_mps", 1.0),
            Tune::movement(g, "walk_speed_mps", 2.0),
            Tune::movement(g, "walk_speed_mps", 3.5),
            Tune::movement(g, "run_speed_mps", 6.0),
            Tune::movement(h, "walk_speed_mps", 0.5),
            // Neither of these can be kept: they are runtime state.
            Tune::Param {
                guid: g,
                name: "speed".into(),
                value: 4.0,
            },
            Tune::Trigger {
                guid: g,
                name: "jump".into(),
            },
        ];
        let out = kept_edits(&kept);
        assert_eq!(out.len(), 3, "{out:?}");
        assert_eq!(out[0], &Tune::movement(g, "walk_speed_mps", 3.5));
        assert_eq!(out[1], &Tune::movement(g, "run_speed_mps", 6.0));
        assert_eq!(out[2], &Tune::movement(h, "walk_speed_mps", 0.5));
    }

    /// A tune on an entity that is not there is a **value**, not a panic.
    #[test]
    fn a_tune_on_a_missing_entity_is_a_refusal() {
        let mut doc = SceneDoc::new();
        let ghost = Uuid::from_u128(0xDEAD);
        assert!(!apply_tune(
            &mut doc,
            &Tune::movement(ghost, "walk_speed_mps", 3.0)
        ));
        assert!(!apply_tune(
            &mut doc,
            &Tune::Param {
                guid: ghost,
                name: "speed".into(),
                value: 1.0
            }
        ));
        assert!(!apply_tune(
            &mut doc,
            &Tune::Trigger {
                guid: ghost,
                name: "jump".into()
            }
        ));
    }
}
