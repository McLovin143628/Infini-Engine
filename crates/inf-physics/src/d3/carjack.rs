//! **The carjack** (wave VEH2b): pull whoever is in a car out of it, and get in.
//!
//! # It is the seat door, from the outside
//!
//! P29.7 built one door into a car — the interact edge resolves the nearest free
//! seat, the character warps into it over `enter_time_s`, and its collider is
//! parked for the ride. Everything about that door is right for a carjack
//! except one line: `vehicle_candidates` skips an occupied chassis, because two
//! characters must not climb into one seat.
//!
//! So this module does not build a second way into a car. It builds the
//! **candidate the first one refuses**, with a verb of its own so the prompt
//! says what the press will do, and it makes the seat free before the ordinary
//! enter runs. One press, one resolution, one warp, and the code that seats the
//! hero is the code that has always seated the hero.
//!
//! # Three conditions, and every one of them is visible to the player
//!
//! * **Somebody is in it, and that somebody is not the player.** The occupant
//!   test is `player_controlled`, which is exactly "an NPC": you cannot pull
//!   yourself out of your own car, and the one seat in this engine that is not
//!   an NPC's is the hero's own.
//! * **You are at the driver's door.** The exit places a driver at the chassis's
//!   `+X` ([`super::vehicle::EXIT_CLEARANCE_M`]'s own arithmetic), so `+X` is the
//!   driver's side, and a carjack from the passenger side is a reach across a
//!   car. `frames/steal-car/0016` is the reference: the hero stands at the
//!   driver's door.
//! * **They do not fight you off this time.** One `mix64` per attempt — see
//!   [`RESIST_CHANCE`] — so a driver who does not want to be pulled out
//!   sometimes is not, and the player presses again.
//!
//! # What happens to the person
//!
//! They land on the road at the door, in [`MovementMode::FallControlled`] —
//! staggering rather than standing, which is what `inf_ecs::movement`'s own
//! transition table was built for: *"the table PERMITS a driver to be pulled out
//! of a seat by something that is a fact about its body rather than a choice.
//! Nothing pulls it yet."* Something does now.
//!
//! Then they **flee**, and they flee by becoming an ordinary crowd agent with a
//! route away from you ([`inf_ecs::crowd::adopt`]). That is not a shortcut: a
//! person walking somewhere is exactly what the crowd is, so the victim tiers,
//! poses, collides and eventually goes `Dormant` like every other pedestrian in
//! the town — with no bespoke state machine, no flee mode and nothing to leak.

use std::collections::BTreeSet;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{CharacterMovement, MovementMode, Transform};
use inf_ecs::interact::{InteractCandidate, InteractVerb, NO_VIEW_TEST_DEG};
use inf_ecs::EcsWorld;

use super::vehicle::{seat_pose, ENTER_REACH_M};
use super::PhysicsBridge3D;

/// What a carjack candidate is called in a prompt: `"[E] Pull out driver"`.
pub const DRIVER_LABEL: &str = "driver";

/// Salts the resist draw.
pub const SALT_RESIST: u64 = 0x5245_5349_5354_0001;

/// How often a driver fights off one attempt.
///
/// A quarter, drawn per **attempt** from the victim's own guid and the sim step
/// — so it is a function of who they are and when you tried, both hosts agree,
/// and pressing again is a different draw. It is deliberately not a *state*: a
/// counter of how many times this driver has resisted would be a second copy of
/// something the seed already answers, which is the draw this module's three
/// neighbours all refused to store.
///
/// A quarter and not a half because a refusal a player cannot see the reason
/// for reads as a broken control if it happens most of the time; at a quarter,
/// two presses clear it 94 % of the time.
pub const RESIST_CHANCE: f64 = 0.25;

/// How far a pulled-out driver walks away, metres.
///
/// **Hoisted to [`inf_ecs::crowd::FLEE_M`] at wave WPN1**, when the flee gained
/// a second caller (a crowd that has heard a gunshot). Re-exported here under
/// its original name because `traffic_3d` measures against it and because "how
/// far a carjacked driver goes" is a question about this module.
pub const FLEE_M: f64 = inf_ecs::crowd::FLEE_M;

/// How fast they walk away, m/s — [`inf_ecs::crowd::FLEE_MPS`], hoisted at wave
/// WPN1 with [`FLEE_M`].
pub const FLEE_MPS: f64 = inf_ecs::crowd::FLEE_MPS;

/// **Who is sitting in this car**, or `None`.
///
/// Derived rather than stored, and it is the *inverse* of
/// `movement::occupied_seats` — which is the shape `inf_ecs::interact`'s own
/// module docs argue for: there is one answer to "is this seat taken" and it
/// lives on the character, so a second field on the vehicle would be a second
/// opinion.
///
/// `O(characters)`.
pub fn occupant_of(world: &EcsWorld, chassis: Uuid) -> Option<Uuid> {
    for guid in inf_ecs::movement::movement_targets(world) {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if world
            .world()
            .get::<CharacterMovement>(e)
            .is_some_and(|cm| cm.runtime.seat.vehicle == chassis)
        {
            return Some(guid);
        }
    }
    None
}

/// **Every chassis somebody is sitting in**, in one walk.
///
/// `occupant_of` inverted and gathered: `O(characters)` once, rather than
/// `O(vehicles x characters)` for a caller that wants the whole set. It is what
/// [`super::interact::candidates`] filters the free-seat list with, and putting
/// it here rather than in the movement step is what closed the wave's own
/// prompt/press divergence — see that function.
pub fn occupied_chassis(world: &EcsWorld) -> BTreeSet<Uuid> {
    occupants(world).into_keys().collect()
}

/// **Who is in which car**, in one walk — the map the set above is the keys of.
///
/// `O(characters)` ONCE. [`occupant_of`] answers for one chassis and is
/// `O(characters)` each time, which is right for a press and wrong for a step
/// that asks about every car it holds: the first cut of
/// `inf_physics::d3::traffic::step_traffic` called it per record, which is
/// `O(cars x characters)` sixty times a second on a settlement with three
/// hundred residents in it.
///
/// A car with two people in it answers the lower `Guid`, which cannot happen —
/// the seat is one seat — and is stated so the walk has one answer rather than
/// an insertion order.
pub fn occupants(world: &EcsWorld) -> std::collections::BTreeMap<Uuid, Uuid> {
    let mut out = std::collections::BTreeMap::new();
    for guid in inf_ecs::movement::movement_targets(world) {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if let Some(cm) = world.world().get::<CharacterMovement>(e) {
            if cm.runtime.seat.is_seated() {
                out.entry(cm.runtime.seat.vehicle).or_insert(guid);
            }
        }
    }
    out
}

/// Whether this occupant is one a carjack may pull out.
///
/// `!player_controlled`, which is the one thing that distinguishes an NPC from
/// the hero in this engine, and which means the answer to "can I carjack my own
/// car" is no without anybody having to pass an actor in.
pub fn is_ejectable(world: &EcsWorld, victim: Uuid) -> bool {
    world
        .entity_of(victim)
        .and_then(|e| world.world().get::<CharacterMovement>(e))
        .is_some_and(|cm| !cm.player_controlled && cm.mode == MovementMode::Driving)
}

/// **The point beside the driver's door** — where the victim lands and where
/// the hero has to be standing.
///
/// The exit's own arithmetic ([`super::vehicle::EXIT_CLEARANCE_M`]), read off
/// the live chassis pose, so the place a driver is thrown to is the place a
/// driver climbs out to.
pub fn door_point(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    chassis: Uuid,
    victim: Uuid,
) -> Option<DVec3> {
    let (seat, rot, _) = seat_pose(bridge, chassis)?;
    let half_width = world
        .entity_of(chassis)
        .and_then(|e| {
            world
                .world()
                .get::<inf_ecs::components::Collider3D>(e)
                .copied()
        })
        .map(|c| match c.shape_kind {
            inf_ecs::components::ColliderShape3DKind::Sphere => c.radius,
            _ => c.half_extents.x,
        })
        .unwrap_or(1.0);
    // **The exit's arithmetic, ALL of it.** The first cut dropped two terms —
    // the vertical lift that turns a seat point into a capsule CENTRE, and the
    // body's own radius in the lateral offset — so the victim was placed about
    // a metre too low and one radius too close, which is a capsule with its
    // feet under the road and its shoulder in the bodywork. The doc claimed
    // "the place a driver is thrown to is the place a driver climbs out to" and
    // it now is.
    let (half_height, radius) = world
        .entity_of(victim)
        .and_then(|e| {
            let cm = world.world().get::<CharacterMovement>(e)?;
            let r = world
                .world()
                .get::<inf_ecs::components::Collider3D>(e)
                .map(|c| c.radius)
                .unwrap_or(0.3);
            Some((cm.stand_half_height_m, r))
        })
        .unwrap_or((0.6, 0.3));
    let target = seat + DVec3::Y * (half_height + radius);
    Some(target + (rot * DVec3::X) * (half_width + super::vehicle::EXIT_CLEARANCE_M + radius))
}

/// **Is the asker at the driver's door?**
///
/// The `+X` half-space of the chassis, which is the side the exit puts a driver
/// out on. A half-space and not a cone, because a door is a side of a car
/// rather than a point on it, and a refusal a player cannot see the edge of
/// reads as a broken control ([`ENTER_REACH_M`]'s own note).
pub fn at_the_door(bridge: &PhysicsBridge3D, chassis: Uuid, feet: DVec3) -> bool {
    let Some(body) = bridge.body_of(chassis) else {
        return false;
    };
    let w = bridge.world();
    let (Some(origin), Some(rot)) = (w.body_translation(body), w.body_rotation(body)) else {
        return false;
    };
    // **Through the CHASSIS, not through the seat.** `seat_local` is offset from
    // the chassis origin, so a plane through the seat sits outboard of the
    // centreline and refuses a player standing beside the middle of the car on
    // the correct side. The car's own `+X` half is the driver's half.
    let side = rot * DVec3::X;
    let d = feet - origin;
    (d.x * side.x + d.z * side.z) > 0.0
}

/// **Every car with somebody in it that could be pulled out**, as candidates.
///
/// Deliberately **not** filtered against the caller's `exclude` set, and the
/// reason is worth stating because it looks like an omission: at the press site
/// `exclude` is `occupied_seats`, so every chassis this function offers is in
/// it — that is the whole point. A carjack candidate is defined by being
/// occupied. Filtering it out here would delete the only candidate this module
/// exists for.
///
/// Nothing else is needed to keep the asker's own car out of the list:
/// [`is_ejectable`] answers `false` for a `player_controlled` occupant, and the
/// hero is the only one there is.
pub fn candidates(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    feet: DVec3,
) -> Vec<InteractCandidate> {
    let mut out = Vec::new();
    // ONE walk over the characters, not one per vehicle. This function is on
    // the prompt path, which runs every frame.
    let seated = occupants(world);
    for chassis in bridge.vehicle_guids() {
        let Some(victim) = seated.get(&chassis).copied() else {
            continue;
        };
        if !is_ejectable(world, victim) {
            continue;
        }
        if !at_the_door(bridge, chassis, feet) {
            continue;
        }
        let Some((seat, _, _)) = seat_pose(bridge, chassis) else {
            continue;
        };
        out.push(InteractCandidate {
            guid: chassis,
            verb: InteractVerb::Carjack,
            label: DRIVER_LABEL.to_string(),
            position: seat,
            // The seat's own reach: a carjack is the enter door with somebody in
            // the way, so it must not be reachable from further off than the
            // enter is.
            range_m: ENTER_REACH_M,
            // No view test, for `vehicle_candidates`' reason verbatim: the door
            // side is already a direction test, and a second one would refuse a
            // player who is standing at the door looking at the wheel.
            view_cone_deg: NO_VIEW_TEST_DEG,
            grip: None,
        });
    }
    out
}

/// What one attempt did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carjack {
    /// The seat is free; the caller may enter it this step.
    Ejected {
        /// The car.
        chassis: Uuid,
        /// Who is now standing in the road.
        victim: Uuid,
    },
    /// They held on. Try again.
    Resisted {
        /// The car.
        chassis: Uuid,
        /// Who held on.
        victim: Uuid,
    },
}

/// **THE CARJACK DOOR** — the one place a person is taken out of a seat.
///
/// Returns `None` when there is nobody to pull out, which is the common case and
/// costs one `vehicle_guids` walk. A refusal is a value all the way down: no
/// occupant, a player-controlled one, the wrong side of the car and a lost
/// resist draw all answer without failing anything.
///
/// `overlays` is the interned overlay table the movement step already built for
/// this step; it is threaded in rather than rebuilt for `try_mantle`'s reason
/// (P29.4 A8) — a second walk over every character to serve one press.
pub fn try_carjack(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    chassis: Uuid,
    actor: Uuid,
    dt: f64,
    overlays: &inf_ecs::movement::OverlayRegistry,
) -> Option<Carjack> {
    let victim = occupant_of(world, chassis)?;
    if victim == actor || !is_ejectable(world, victim) {
        return None;
    }
    // The draw, on the victim's own guid and this step. Both hosts hold the same
    // number, and pressing again next step is a different one.
    let tick = inf_ecs::traffic::steps(world);
    if inf_ecs::crowd::agent_unit(victim, tick, SALT_RESIST) < RESIST_CHANCE {
        return Some(Carjack::Resisted { chassis, victim });
    }
    let at = door_point(world, bridge, chassis, victim)?;
    // The mode is taken rather than requested: being pulled out of a car is a
    // fact about your body and not a choice, which is the sentence
    // `inf_ecs::movement::transition_is_legal`'s own `(Driving, FallControlled)`
    // row was written for.
    if !super::movement::eject_from_seat(
        world,
        bridge,
        victim,
        at,
        MovementMode::FallControlled,
        overlays,
    ) {
        return None;
    }
    // The car is nobody's business but the player's from here.
    inf_ecs::traffic::mark_taken(world, chassis);
    // **AND THE STREET SAW IT** (wave EMS3). Raised rather than recorded,
    // because this runs in the `character move` phase and the question "who
    // could see it" needs a collision world three phases later — see
    // `inf_ecs::witness::raise_act`. At the DOOR the victim came out of, which
    // is where a witness would say it happened rather than at the chassis
    // origin, and it is the same point the ejection itself used.
    //
    // After `mark_taken`, so that when the witness pass asks what the actor was
    // driving one step later the answer is already this car.
    inf_ecs::witness::raise_act(world, inf_ecs::witness::ActKind::Carjack, actor, at);
    // …and the person walks away. An ordinary crowd agent with a route, so it
    // tiers, poses and eventually goes Dormant like every other pedestrian —
    // rather than a statue in the road with a bespoke state machine.
    flee(world, victim, at, actor, dt);
    Some(Carjack::Ejected { chassis, victim })
}

/// Give the victim somewhere to be: [`FLEE_M`] metres directly away from
/// whoever pulled them out.
///
/// A straight leg and not a plan, and the honest reason is that a plan needs a
/// graph the victim may not be standing on — a car can be carjacked anywhere a
/// car can be. The crowd's own `Full` tier steers the body through
/// `move_and_slide`, so a route that runs into a wall is a body that stops at
/// the wall rather than one that walks through it.
///
/// `RouteMode::Once`: they arrive, and then they stand. **They do not resume
/// their day** — see the wave's carried list.
///
/// # This is now the FIRST caller of one door, not the only implementation
///
/// Wave WPN1 hoisted the body of this into [`inf_ecs::crowd::flee_from`],
/// because a crowd that has heard a gunshot needs the identical behaviour and a
/// second copy would have been a second answer to *"what does a frightened
/// person in this engine do"* — including a second copy of the re-phase, which
/// is the half that is easy to leave out and impossible to see. What is left
/// here is the one thing that is genuinely about a carjack: **which point they
/// run away from**, which is whoever pulled them out.
fn flee(world: &mut EcsWorld, victim: Uuid, from: DVec3, actor: Uuid, dt: f64) {
    let away_from = world
        .entity_of(actor)
        .and_then(|e| world.world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        // No actor to run from — the door was opened by a script — so they run
        // along `-Z` from where they are, which is what `flee_from` answers for
        // a zero-length direction.
        .unwrap_or(from - DVec3::Z);
    inf_ecs::crowd::flee_from(world, victim, from, away_from, dt, FLEE_M);
}

/// Which chassis a carjack candidate names, for a caller that wants to ask
/// before it presses — the gate's own door.
pub fn carjackable(world: &EcsWorld, bridge: &PhysicsBridge3D, feet: DVec3) -> BTreeSet<Uuid> {
    candidates(world, bridge, feet)
        .into_iter()
        .map(|c| c.guid)
        .collect()
}
