//! **The one interaction door** (island wave I5): every candidate the E key
//! could be about, ranked by one rule.
//!
//! # Why this is here and not in `inf-ecs`
//!
//! Two of the three things a candidate needs — where it is, and whether it is
//! taken — come from different places. An authored `inf_ecs::interact::Interactable` is at its
//! `GlobalTransform`; a **vehicle seat** is at the chassis's *physics* pose plus
//! its rig's seat offset, and the physics pose is the live one rather than the
//! one the last write-back left in the ECS. Migrating the seat onto the ECS
//! transform would have moved it by one fixed step, which is a semantic change
//! to a shipped gate — so the seat keeps its source and this module is where the
//! two meet.
//!
//! What is shared is the **rule**: [`inf_ecs::interact::resolve`], pure, with no
//! world and no physics in it. Both kinds of candidate are ranked by the same
//! arithmetic and the same `Guid` tie-break, which is what makes this one door
//! rather than two that agree.
//!
//! # The migration, and what did NOT change
//!
//! P29.7's `vehicle::try_enter` was the only interaction in the engine: nearest
//! seat inside [`ENTER_REACH_M`], walked in `Guid` order with a strict `<`,
//! occupied seats skipped, no view test. Every one of those is preserved
//! *exactly* — the seat candidate carries `NO_VIEW_TEST_DEG`, so the view half
//! of the rule is skipped for it — and `try_enter` is now a call into this door
//! rather than a second implementation of it.

use std::collections::BTreeSet;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::interact::{self, InteractCandidate, InteractHit, InteractVerb, NO_VIEW_TEST_DEG};
use inf_ecs::world::EcsWorld;

use super::ecs::PhysicsBridge3D;
use super::vehicle::{seat_pose, ENTER_REACH_M};

/// What a vehicle's seat is called in a prompt.
pub const VEHICLE_LABEL: &str = "vehicle";

/// **Every vehicle seat that is free**, as candidates, in `Guid` order.
///
/// `O(vehicles)`, and `O(1)` on a level with none. The reach and the absence of
/// a view test are P29.7's, verbatim.
pub fn vehicle_candidates(
    bridge: &PhysicsBridge3D,
    occupied: &BTreeSet<Uuid>,
) -> Vec<InteractCandidate> {
    let mut out = Vec::new();
    for chassis in bridge.vehicle_guids() {
        if occupied.contains(&chassis) {
            continue;
        }
        let Some((seat, _, _)) = seat_pose(bridge, chassis) else {
            continue;
        };
        out.push(InteractCandidate {
            guid: chassis,
            verb: InteractVerb::Enter,
            label: VEHICLE_LABEL.to_string(),
            position: seat,
            range_m: ENTER_REACH_M,
            // **No view test**, exactly as `try_enter` had none: a player
            // standing beside a car with their back to it is still standing
            // beside a car, and changing that would be a semantic change to a
            // gate this wave is not allowed to move.
            view_cone_deg: NO_VIEW_TEST_DEG,
            // Getting into a car is a whole-body choreography (`SeatState`), not
            // a hand on a door handle. When a vehicle grows a door leaf this is
            // where its grip goes.
            grip: None,
        });
    }
    out
}

/// **Every candidate**, seats, doors and authored interactables together, in
/// `Guid` order.
///
/// `exclude` is what the character must not interact with — the vehicle it is
/// already driving, and itself.
///
/// `feet` is where the character is standing, and it is here for **one** reason:
/// a door's prompt says whether the lock verb is on offer, and that is a fact
/// about which face the character is on. Every other candidate ignores it. The
/// alternative — building door candidates without a side and deciding at the
/// press — is the thing I5 built one resolution site to prevent: what the player
/// is told and what the press does would be two calls.
pub fn candidates(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    feet: DVec3,
    exclude: &BTreeSet<Uuid>,
) -> Vec<InteractCandidate> {
    // **A seat with somebody in it is not an Enter candidate anywhere** (wave
    // VEH2b). It used to be the CALLER's job to say so — the movement step
    // passed `occupied_seats` and the HUD passed `{self, own car}` — and the two
    // were only ever the same set because nothing but the player ever sat down.
    // The moment an NPC drives, the prompt would read "[E] Enter vehicle" over
    // an occupied car while the press did something else entirely, which is the
    // I5 divergence ("what the player is told and what the press does cannot
    // come apart") in the one place I5 did not reach.
    //
    // Occupancy is a fact about the WORLD, not about who is asking, so it is
    // read here and unioned with whatever the caller excluded for its own
    // reasons.
    let mut occupied = super::carjack::occupied_chassis(world);
    occupied.extend(exclude.iter().copied());
    let mut out = vehicle_candidates(bridge, &occupied);
    out.extend(interact::candidates_in_world(world, exclude));
    // I6: doors. Excluded like anything else, so a door a character is somehow
    // already the subject of does not offer itself.
    out.extend(
        super::door::candidates(world, &bridge.sim_band(world), feet)
            .into_iter()
            .filter(|c| !exclude.contains(&c.guid)),
    );
    // VEH2b: the cars with somebody in them. Deliberately NOT filtered against
    // `exclude`, because at the press site `exclude` is `occupied_seats` and
    // every chassis this adds is in it — which is the whole point. See
    // `carjack::candidates`.
    out.extend(super::carjack::candidates(world, bridge, feet));
    // One sorted walk, because the rule's tie-break is only deterministic over
    // one: a seat and an item at exactly the same distance must resolve the same
    // way in both hosts.
    out.sort_by_key(|c| c.guid);
    out
}

/// **THE door.** What the E key is about, from where the character is standing.
pub fn resolve(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    feet: DVec3,
    aim_yaw_deg: f64,
    exclude: &BTreeSet<Uuid>,
) -> Option<InteractHit> {
    interact::resolve(&candidates(world, bridge, feet, exclude), feet, aim_yaw_deg)
}

/// The vehicle half on its own — what `vehicle::try_enter` answers.
///
/// Kept as a named function rather than folded into [`resolve`]'s caller because
/// the movement step asks two different questions: "is there a seat" (which is
/// what the enter path acts on) and "what is under the prompt" (which is every
/// candidate). Both go through [`interact::resolve`]; only the candidate list
/// differs.
pub fn nearest_seat(
    bridge: &PhysicsBridge3D,
    feet: DVec3,
    occupied: &BTreeSet<Uuid>,
) -> Option<Uuid> {
    // The aim is irrelevant: every seat candidate carries `NO_VIEW_TEST_DEG`, so
    // the rule skips the view half for it. Passing zero is not a guess.
    interact::resolve(&vehicle_candidates(bridge, occupied), feet, 0.0).map(|h| h.guid)
}
