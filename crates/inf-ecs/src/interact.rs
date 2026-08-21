//! **The interaction core** (island wave I5): what a thing offers to be done to
//! it, and the one rule that picks which of them the E key is about.
//!
//! # Why [`Interactable`] is not a scene component
//!
//! `RuntimeEntityGen` decodes positionally, so a new slot on it is a schema
//! bump — and this wave has no schema budget. That is the *constraint*; here is
//! the reason it is the right shape anyway.
//!
//! An interactable is **derived from what an entity already is**: a vehicle
//! offers its seat, an item offers itself. Persisting a second copy of that
//! would be a second answer to "can this be entered", and the day the two
//! disagreed the level would be right and the derivation wrong (or the other way
//! round, which is worse). So the component is a **runtime** one — the shape
//! `MovementRuntime` and P22's `DeformField` already have — derived through one
//! door, and gameplay may insert or remove one at will.
//!
//! # The resolver is ONE rule and it is pure
//!
//! [`resolve`] takes a list of candidates and answers with at most one. It has
//! no world, no physics and no clock in it, which is what lets the vehicle seat
//! (whose pose comes from the physics bridge) and an ECS item (whose pose comes
//! from its transform) be ranked by the same arithmetic — and what lets the
//! whole rule be measured without either.
//!
//! # Cost
//!
//! `O(candidates)`, and the candidate walk is `O(interactables)` rather than
//! `O(entities)`: `try_query_filtered` restricts it to archetypes that carry the
//! component, and `None` — the component never inserted in this world — is the
//! `O(1)` answer for every level that has none. Same law as
//! [`crate::movement::movement_targets`].

use std::collections::BTreeSet;

use bevy_ecs::prelude::{Component, With};
use glam::DVec3;
use uuid::Uuid;

use crate::components::{GlobalTransform, Guid};
use crate::math::Vec2d;
use crate::movement::{angle_delta_deg, planar_yaw_deg};
use crate::world::EcsWorld;

/// What the interaction *does*, in the player's words.
///
/// A small closed set rather than a free string, because the verb decides the
/// prompt and a prompt is the one thing in this system a player reads. The
/// mandate's list — interact, pickup, grab, grip, use — collapses onto these
/// five: `Grab` and `Grip` are the same gesture at two ranges and the engine has
/// no distinction to make between them yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum InteractVerb {
    /// Operate it in place: a door, a switch, a terminal.
    #[default]
    Use,
    /// Get inside it: a vehicle's seat.
    Enter,
    /// Take it: an item on the ground.
    PickUp,
    /// Hold it without taking it: a crate, a ledge, a rope.
    Grab,
    /// Speak to it.
    Talk,
}

impl InteractVerb {
    /// The word the prompt starts with.
    pub fn word(self) -> &'static str {
        match self {
            InteractVerb::Use => "Use",
            InteractVerb::Enter => "Enter",
            InteractVerb::PickUp => "Pick up",
            InteractVerb::Grab => "Grab",
            InteractVerb::Talk => "Talk to",
        }
    }
}

/// How wide a view cone counts as "no view test at all", degrees.
///
/// A cone of 360 degrees admits every direction, so the test is skipped rather
/// than computed — which matters because it is what lets the vehicle seat, whose
/// P29.7 semantics have never had a view test, migrate onto this rule **without
/// changing what it answers**.
pub const NO_VIEW_TEST_DEG: f64 = 360.0;

/// The default reach for an authored interactable, metres. Generous rather than
/// exact, for the reason `ENTER_REACH_M` is: a door is not a pixel, and a
/// refusal a player cannot see the edge of reads as a broken control.
pub const DEFAULT_REACH_M: f64 = 2.5;

/// The default cone an authored interactable is visible within, degrees (total,
/// not half). Ninety degrees is "roughly in front of me".
pub const DEFAULT_VIEW_CONE_DEG: f64 = 90.0;

/// How far past the cone's half-angle a candidate is still admitted, degrees.
///
/// **The portable `atan2`'s own error, and nothing else.** `patan2_64` is a
/// `pacos64` in disguise (the P14 law: `f32` std trig is not bit-portable, so
/// this engine ships its own), and the price of portability is a few
/// ten-thousandths of a degree. A thing at *exactly* 45 degrees of a 90-degree
/// cone must be reachable — every boundary in this repository resolves toward
/// the richer answer — and without this it would be reachable only when the
/// approximation happened to round down.
///
/// A ten-thousandth of a degree is 1.7 micrometres of arc at a metre, so nothing
/// a designer authors can tell the difference; it is arithmetic, not policy.
/// The arm beside it measures the approximation rather than trusting this
/// number.
pub const VIEW_CONE_EPSILON_DEG: f64 = 1e-4;

/// **An entity offers something to be done to it.**
///
/// A runtime component: never serialized, never in the scene record — see the
/// module header.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct Interactable {
    /// What the prompt says it does.
    pub verb: InteractVerb,
    /// What the prompt calls the thing.
    pub label: String,
    /// How close the character's feet must be, metres.
    pub range_m: f64,
    /// Whether it offers anything right now. A locked door is an interactable
    /// that answers no, which is a different thing from a door with no
    /// interaction on it — and a resolver that skipped it entirely would let the
    /// prompt fall through to whatever is behind it.
    pub enabled: bool,
    /// The total cone, in degrees, the character must be facing it within.
    /// [`NO_VIEW_TEST_DEG`] means "from any direction".
    pub view_cone_deg: f64,
}

impl Default for Interactable {
    fn default() -> Self {
        Self {
            verb: InteractVerb::Use,
            label: String::new(),
            range_m: DEFAULT_REACH_M,
            enabled: true,
            view_cone_deg: DEFAULT_VIEW_CONE_DEG,
        }
    }
}

/// One thing the resolver may pick, flattened out of whatever produced it.
///
/// The seam that lets a vehicle seat (a physics pose) and an item (an ECS
/// transform) be ranked by the same arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub struct InteractCandidate {
    /// The entity the interaction is with.
    pub guid: Uuid,
    /// What it offers.
    pub verb: InteractVerb,
    /// What to call it.
    pub label: String,
    /// Where the interaction *is* — a seat, not a chassis centre.
    pub position: DVec3,
    /// Its reach, metres.
    pub range_m: f64,
    /// Its cone, degrees.
    pub view_cone_deg: f64,
}

/// What the resolver picked.
#[derive(Clone, Debug, PartialEq)]
pub struct InteractHit {
    /// The entity.
    pub guid: Uuid,
    /// Its verb.
    pub verb: InteractVerb,
    /// Its label.
    pub label: String,
    /// Where it is.
    pub position: DVec3,
    /// How far the feet were from it, metres.
    pub distance_m: f64,
}

/// **THE interaction rule**: the nearest candidate in range and in view.
///
/// `feet` is the character's ground contact, not its centre: a seat is measured
/// from the feet in P29.7 and the reach budget includes the seat's height, so
/// measuring from the eye would silently change every distance.
///
/// `aim_yaw_deg` is where the character is looking, used only for the view test.
///
/// **Ties break by `Guid`**, and the walk is in `Guid` order with a strict `<`
/// on the distance — so two seats at exactly the same distance resolve the same
/// way in both hosts, rather than by whichever a bevy archetype yielded first.
/// That is the property P29.7's `try_enter` had and this rule inherits verbatim.
///
/// Non-finite geometry is skipped: a NaN distance compares false against
/// everything, so a candidate carrying one would be silently unreachable rather
/// than obviously wrong.
pub fn resolve(
    candidates: &[InteractCandidate],
    feet: DVec3,
    aim_yaw_deg: f64,
) -> Option<InteractHit> {
    let mut best: Option<(f64, &InteractCandidate)> = None;
    for c in candidates {
        if !c.position.is_finite() || !c.range_m.is_finite() {
            continue;
        }
        let to = c.position - feet;
        let d = to.length();
        if !d.is_finite() || d > c.range_m {
            continue;
        }
        if c.view_cone_deg < NO_VIEW_TEST_DEG {
            let planar = Vec2d::new(to.x, to.z);
            // Straight up or straight down has no direction to test; a candidate
            // directly overhead is in front of you by any reading.
            if planar.x.abs() > 1e-9 || planar.y.abs() > 1e-9 {
                // **Closed on the boundary, with a tolerance that is about the
                // arithmetic and not about the design.** A candidate at exactly
                // 45 degrees of a 90-degree cone is IN — every boundary in this
                // repo resolves toward the richer answer — and the angle comes
                // out of `inf_math::patan2_64`, which is a *portable*
                // approximation (the P14 law: std trig is not bit-portable) and
                // therefore lands a little either side of 45. Without the
                // epsilon the rule would be "in, except when the approximation
                // rounds up", which is a rule nobody can author against.
                //
                // [`VIEW_CONE_EPSILON_DEG`] is the approximation's error, not a
                // designer's tolerance — see its own docs for the measurement.
                let half = (c.view_cone_deg * 0.5).max(0.0);
                if angle_delta_deg(planar_yaw_deg(planar), aim_yaw_deg).abs()
                    > half + VIEW_CONE_EPSILON_DEG
                {
                    continue;
                }
            }
        }
        // Strict `<` over a `Guid`-ordered walk: the first of two equals wins.
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, c));
        }
    }
    best.map(|(d, c)| InteractHit {
        guid: c.guid,
        verb: c.verb,
        label: c.label.clone(),
        position: c.position,
        distance_m: d,
    })
}

/// **The ECS half of the candidate walk**: every enabled [`Interactable`] with a
/// transform, in `Guid` order, minus `exclude`.
///
/// `O(interactables)` — and `O(1)` on a level that has none, because
/// `try_query_filtered` answers `None` when the component was never inserted.
pub fn candidates_in_world(world: &EcsWorld, exclude: &BTreeSet<Uuid>) -> Vec<InteractCandidate> {
    let w = world.world();
    let Some(mut q) =
        w.try_query_filtered::<(&Guid, &Interactable, &GlobalTransform), With<Interactable>>()
    else {
        return Vec::new();
    };
    let mut out: Vec<InteractCandidate> = q
        .iter(w)
        .filter(|(g, i, _)| i.enabled && !exclude.contains(&g.0))
        .map(|(g, i, t)| InteractCandidate {
            guid: g.0,
            verb: i.verb,
            label: i.label.clone(),
            position: t.translation(),
            range_m: i.range_m,
            view_cone_deg: i.view_cone_deg,
        })
        .collect();
    // A bevy archetype walk's order is an implementation detail and must never
    // reach a result — the tie-break above is only deterministic over a sorted
    // walk.
    out.sort_by_key(|c| c.guid);
    out
}

/// **The prompt a player reads**, e.g. `[E] Enter vehicle`.
///
/// Built here so the wording exists once: the shipped player, the editor's
/// preview and any future gamepad glyph all format the same sentence, and a
/// control with no key bound reads as such rather than as `[] Enter vehicle`.
pub fn prompt_text(verb: InteractVerb, label: &str, key: &str) -> String {
    let head = if key.trim().is_empty() {
        "[unbound]".to_string()
    } else {
        format!("[{}]", key.trim())
    };
    if label.trim().is_empty() {
        format!("{head} {}", verb.word())
    } else {
        format!("{head} {} {}", verb.word(), label.trim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(guid: u128, x: f64, z: f64) -> InteractCandidate {
        InteractCandidate {
            guid: Uuid::from_u128(guid),
            verb: InteractVerb::Use,
            label: "thing".into(),
            position: DVec3::new(x, 0.0, z),
            range_m: 3.0,
            view_cone_deg: NO_VIEW_TEST_DEG,
        }
    }

    /// **The nearest one in range wins, and out of range is nothing.**
    #[test]
    fn the_resolver_picks_the_nearest_candidate_inside_its_own_reach() {
        let feet = DVec3::ZERO;
        let far = at(1, 0.0, 2.5);
        let near = at(2, 0.0, 1.0);
        let out = at(3, 0.0, 30.0);
        let hit = resolve(&[far.clone(), near.clone(), out.clone()], feet, 0.0).unwrap();
        assert_eq!(hit.guid, near.guid);
        assert!((hit.distance_m - 1.0).abs() < 1e-12);
        // Each candidate's OWN reach, not a global one: the far thing is inside
        // 3 m and the out one is not.
        assert_eq!(
            resolve(&[far.clone(), out.clone()], feet, 0.0)
                .unwrap()
                .guid,
            far.guid
        );
        assert_eq!(resolve(&[out], feet, 0.0), None);
        assert_eq!(resolve(&[], feet, 0.0), None);
    }

    /// **A tie breaks by `Guid`, not by list order.** Two seats at the same
    /// distance must resolve the same way in both hosts.
    #[test]
    fn a_tie_breaks_by_guid_whichever_order_the_walk_arrived_in() {
        let feet = DVec3::ZERO;
        let a = at(1, 0.0, 2.0);
        let b = at(2, 2.0, 0.0);
        // The walk is Guid-ordered by the caller; the rule's strict `<` keeps
        // the first, so the answer is the lower guid either way round.
        assert_eq!(
            resolve(&[a.clone(), b.clone()], feet, 0.0).unwrap().guid,
            a.guid
        );
        let mut sorted = vec![b, a.clone()];
        sorted.sort_by_key(|c| c.guid);
        assert_eq!(resolve(&sorted, feet, 0.0).unwrap().guid, a.guid);
    }

    /// **The view cone**, and the case that makes the vehicle migration safe: a
    /// candidate with [`NO_VIEW_TEST_DEG`] is reachable from behind.
    #[test]
    fn a_cone_excludes_what_is_behind_you_and_a_full_circle_does_not() {
        let feet = DVec3::ZERO;
        let mut behind = at(1, 0.0, -2.0);
        behind.view_cone_deg = 90.0;
        // Facing +z (yaw 0): a thing at -z is 180 degrees away.
        assert_eq!(resolve(&[behind.clone()], feet, 0.0), None);
        // Turn around and it is there.
        assert!(resolve(&[behind.clone()], feet, 180.0).is_some());
        // A seat carries no view test and is reachable from any side, which is
        // what P29.7's `try_enter` did and what the migration must not change.
        behind.view_cone_deg = NO_VIEW_TEST_DEG;
        assert!(resolve(&[behind], feet, 0.0).is_some());

        // The cone's own edge: 90 degrees total is 45 either side, and a thing
        // at exactly 45 is IN (closed, like every other boundary in this repo
        // that resolves toward "reachable").
        let mut edge = at(2, 1.0, 1.0);
        edge.view_cone_deg = 90.0;
        // **The approximation is measured rather than assumed.** The epsilon
        // exists to cover `patan2_64`'s error and nothing else, so the arm
        // prints the error and refuses an epsilon that is bigger than it needs
        // to be — a tolerance wide enough to hide a design mistake is not a
        // tolerance, it is a hole.
        let measured = planar_yaw_deg(Vec2d::new(1.0, 1.0));
        println!(
            "the portable atan2 puts (1, 1) at {measured} degrees, {} off 45; the epsilon is {VIEW_CONE_EPSILON_DEG}",
            (measured - 45.0).abs()
        );
        assert!(
            (measured - 45.0).abs() <= VIEW_CONE_EPSILON_DEG,
            "the epsilon does not cover the approximation's error"
        );
        assert!(
            VIEW_CONE_EPSILON_DEG < 0.01,
            "an epsilon this wide is a design tolerance pretending to be arithmetic"
        );
        assert!(
            resolve(&[edge.clone()], feet, 0.0).is_some(),
            "45 degrees is inside a 90 degree cone"
        );
        edge.view_cone_deg = 80.0;
        assert_eq!(
            resolve(&[edge], feet, 0.0),
            None,
            "and outside an 80 degree one"
        );
    }

    /// Hostile geometry is skipped rather than silently winning: a NaN distance
    /// compares false against everything, including the `>` that rejects it.
    #[test]
    fn a_candidate_with_hostile_geometry_is_skipped() {
        let feet = DVec3::ZERO;
        let good = at(2, 0.0, 1.0);
        for bad_pos in [
            DVec3::new(f64::NAN, 0.0, 0.0),
            DVec3::new(0.0, f64::INFINITY, 0.0),
        ] {
            let mut bad = at(1, 0.0, 0.0);
            bad.position = bad_pos;
            assert_eq!(
                resolve(&[bad, good.clone()], feet, 0.0).unwrap().guid,
                good.guid
            );
        }
        let mut bad = at(1, 0.0, 0.5);
        bad.range_m = f64::NAN;
        assert_eq!(
            resolve(&[bad, good.clone()], feet, 0.0).unwrap().guid,
            good.guid,
            "a NaN reach admitted a candidate that should have been skipped"
        );
    }

    /// A disabled interactable does not swallow the prompt — the walk that
    /// builds candidates drops it, so whatever is behind it is still reachable.
    #[test]
    fn a_disabled_interactable_is_not_a_candidate() {
        let mut w = EcsWorld::new();
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        for (guid, z, enabled) in [(a, 1.0, false), (b, 2.0, true)] {
            let e = w.spawn_with_guid(guid, "Thing", None);
            let mut t = crate::components::Transform::IDENTITY;
            t.translation = crate::math::Vec3d::new(0.0, 0.0, z);
            w.world_mut().entity_mut(e).insert((
                Interactable {
                    label: "thing".into(),
                    enabled,
                    view_cone_deg: NO_VIEW_TEST_DEG,
                    range_m: 5.0,
                    ..Default::default()
                },
                t,
            ));
        }
        w.mark_dirty();
        w.propagate();
        let cands = candidates_in_world(&w, &BTreeSet::new());
        assert_eq!(cands.len(), 1, "{cands:?}");
        assert_eq!(cands[0].guid, b);
        // …and the nearer, disabled one does not block it.
        assert_eq!(resolve(&cands, DVec3::ZERO, 0.0).unwrap().guid, b);

        // An excluded guid is not a candidate either — how a character avoids
        // interacting with the vehicle it is already driving.
        let mut ex = BTreeSet::new();
        ex.insert(b);
        assert!(candidates_in_world(&w, &ex).is_empty());
    }

    /// **A level with no interactables costs nothing** — the `O(1)` answer.
    #[test]
    fn a_world_with_no_interactables_walks_nothing() {
        let w = EcsWorld::new();
        assert!(candidates_in_world(&w, &BTreeSet::new()).is_empty());
    }

    /// The prompt says one sentence, and an unbound control says so.
    #[test]
    fn a_prompt_names_its_key_and_admits_when_there_is_none() {
        assert_eq!(
            prompt_text(InteractVerb::Enter, "vehicle", "E"),
            "[E] Enter vehicle"
        );
        assert_eq!(
            prompt_text(InteractVerb::PickUp, "  rifle ", "F"),
            "[F] Pick up rifle"
        );
        assert_eq!(
            prompt_text(InteractVerb::Use, "", "E"),
            "[E] Use",
            "an unlabelled thing still names its verb"
        );
        assert_eq!(
            prompt_text(InteractVerb::Enter, "vehicle", "  "),
            "[unbound] Enter vehicle",
            "a control with no key must say so rather than draw an empty bracket"
        );
    }
}
