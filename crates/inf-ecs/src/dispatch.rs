//! **THE DISPATCHER** (wave EMS2) — who is sent to what, and what a person sent
//! to something is allowed to do.
//!
//! The deciding half. Everything here is a pure function of sim state over
//! resources: no schema moves (scene v27 / `ScenePayload` v12 stand), nothing is
//! ever written to a file, and the applying half — the routes, the bodies, the
//! sirens — lives in `inf_physics::d3::dispatch` behind this crate's own split
//! (`inf_ecs::vehicle` decides and `inf_physics::d3::vehicle` applies; the same
//! wall, a third time).
//!
//! # This module opens with the panic exemption, and the order is the point
//!
//! [`crate::crowd::flee_from`] is the one door a frightened person goes through,
//! and until this wave it had no idea who it was frightening. `step_panic` walks
//! the *whole* population on the step a shot goes off, `flee_from` **clears the
//! schedule** of everybody it reaches, and [`crate::crowd::PanickedRes`] is never
//! released — so a gunshot at an incident would have permanently routed the
//! officers standing at it. Every later clause of this wave rests on that not
//! happening, so the exemption is the first thing in the file rather than a
//! guard bolted onto the last.
//!
//! The rule is one sentence and it is a rule rather than a filter: **a responder
//! does not rout**. It lives at the flee door so it holds for every caller —
//! the crowd panic, the carjack, and anything a later wave adds — and
//! `PanicReport::exempt` counts the times it fired so a gate can tell "the
//! officers held" from "no officer was ever in the radius".

use std::collections::BTreeSet;

use bevy_ecs::prelude::Resource;
use uuid::Uuid;

use crate::world::EcsWorld;

/// **Everybody who is on duty at something** (wave EMS2) — the named responder
/// set the panic exemption reads.
///
/// # A resource, and it is [`crate::crowd::PanickedRes`]' reason exactly
///
/// A responder is a crowd agent, and a [`Dormant`](crate::crowd::CrowdTier::Dormant)
/// one has **no entity at all** — its record is in the population, it still
/// steps, and it is exactly the agent a shot at the far edge of a panic radius
/// reaches. A marker component would have been silently absent on every one of
/// them, which is the tier-dependent-state trap `crowd_state_bytes`' own doc
/// names.
///
/// Derived, never saved, no schema moves — [`crate::item::ItemDefs`]' shape.
#[derive(Resource, Default, Debug, Clone, PartialEq, Eq)]
pub struct RespondersRes {
    /// Who is on duty, in `Guid` order.
    pub on_duty: BTreeSet<Uuid>,
}

/// **Is this person a responder?** — the named predicate the panic exemption is
/// written in terms of.
///
/// Public and named rather than folded into a `filter` inside the panic pass,
/// because the exemption is a *rule about the world* — an officer under fire
/// does not rout — and a rule nobody can ask about is a filter. The gate arm
/// `an_officer_under_fire_does_not_rout` asks it.
///
/// `O(log n)` on a level that has responders and one `get_resource` on every
/// level that does not, which is every level committed before this wave.
pub fn is_responder(world: &EcsWorld, guid: Uuid) -> bool {
    world
        .world()
        .get_resource::<RespondersRes>()
        .is_some_and(|r| r.on_duty.contains(&guid))
}

/// **Put somebody on duty**, or take them off it.
///
/// The one door, so a second producer cannot invent a second shape of the set.
/// Returns whether the set changed — an engagement counter for a caller that
/// wants to know it did something.
pub fn set_responder(world: &mut EcsWorld, guid: Uuid, on_duty: bool) -> bool {
    let mut res = world
        .world_mut()
        .remove_resource::<RespondersRes>()
        .unwrap_or_default();
    let changed = if on_duty {
        res.on_duty.insert(guid)
    } else {
        res.on_duty.remove(&guid)
    };
    world.world_mut().insert_resource(res);
    changed
}

/// Everybody on duty right now, in `Guid` order — empty on a level with no
/// responders.
pub fn responders(world: &EcsWorld) -> Vec<Uuid> {
    world
        .world()
        .get_resource::<RespondersRes>()
        .map(|r| r.on_duty.iter().copied().collect())
        .unwrap_or_default()
}

/// **Forget who was on duty** — [`crate::crowd::clear_crowd`]'s twin, for its
/// reason: an editor Simulate session must leave nothing behind in the author's
/// document, and a resource is outside the `ScenePersist::Memory` snapshot by
/// construction.
pub fn clear_dispatch(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<RespondersRes>();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    /// The set is a set: idempotent, order-free, and absent costs nothing.
    #[test]
    fn a_responder_is_on_duty_until_taken_off_it() {
        let mut w = EcsWorld::new();
        assert!(
            !is_responder(&w, guid(1)),
            "an empty world has no duty roster"
        );
        assert!(responders(&w).is_empty());

        assert!(
            set_responder(&mut w, guid(1), true),
            "the first mark changes it"
        );
        assert!(!set_responder(&mut w, guid(1), true), "the second does not");
        assert!(is_responder(&w, guid(1)));
        assert!(!is_responder(&w, guid(2)));

        set_responder(&mut w, guid(2), true);
        assert_eq!(responders(&w), vec![guid(1), guid(2)], "`Guid` order");

        assert!(set_responder(&mut w, guid(1), false));
        assert!(!is_responder(&w, guid(1)));
        assert!(is_responder(&w, guid(2)));

        clear_dispatch(&mut w);
        assert!(!is_responder(&w, guid(2)));
        assert!(responders(&w).is_empty());
    }
}
