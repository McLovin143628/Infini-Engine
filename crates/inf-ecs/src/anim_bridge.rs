//! **The animation bridge** (P29.4) — the Ring-0 doors the `anim.*` Blueprint kit
//! and the movement step go through, and the one resource they share.
//!
//! # What a bridge is for
//!
//! Before this wave the animation state machine could only be driven by the
//! actor's Blueprint **variables**: [`crate::pose::step_pose_evaluation`] resolves
//! every parameter through a `vars` closure each host builds out of its own actor
//! registry. That is a fine authoring path and a poor *gameplay* one — a system
//! that wants to say "this character just landed hard" has to know which actor
//! owns the character and what its variable is called, and a system that wants to
//! ask "is it still getting up?" has no route at all.
//!
//! So there are four questions, and they are the four nodes of the kit:
//!
//! * **set a parameter** — a named `f64` overlaid on the actor's variables;
//! * **set a trigger** — arm a declared [`inf_anim::SmParamKind::Trigger`] once,
//!   through [`inf_anim::SmRuntime::arm_trigger`] rather than by writing a level
//!   (see that method for why the difference matters);
//! * **query the state** — what the machine is in, and for how long;
//! * **consume a notify** — take one of this step's events, so exactly one
//!   consumer acts on it.
//!
//! # Why one resource and not five
//!
//! Every field here follows [`crate::pose::PoseStoreRes`]'s rules exactly — never
//! serialized, written only from the fixed step or from a gameplay door, dropped
//! by [`crate::pose::clear_poses`] — and they are all read inside the same loop of
//! the same function. Five resources would be five `get_resource` calls per step
//! and five things to remember to clear; one is one.
//!
//! It is **absent** until something writes it, so a level whose characters have no
//! bridge traffic is byte-identical to its pre-P29.4 self.
//!
//! # Determinism
//!
//! `BTreeMap` / `BTreeSet` throughout, for the reason every other sim-side map in
//! this crate is: the iteration order reaches a pose, and a pose reaches
//! `state_bytes`.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::Resource;
use inf_anim::RootMotion3D;
use uuid::Uuid;

use crate::components::AnimStateMachine;
use crate::world::EcsWorld;

/// What one entity's machine is doing, published every fixed step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimStateInfo {
    /// Index into the machine's `states`.
    pub index: usize,
    /// The state's authored name — what `anim.query_state` compares against.
    pub name: String,
    /// Seconds spent in this state.
    pub time_s: f64,
    /// Whether a transition is still blending.
    pub blending: bool,
}

/// The bridge's shared state — see the module docs.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct AnimBridgeRes {
    /// Parameters set through [`set_anim_param`], overlaid on the actor's
    /// Blueprint variables. **The overlay wins**: a name set here shadows a
    /// variable of the same name, because a gameplay system that has just said
    /// "landed hard" must not be silently outvoted by a stale variable.
    pub params: BTreeMap<Uuid, BTreeMap<String, f64>>,
    /// Triggers armed through [`set_anim_trigger`] and not yet handed to a
    /// machine. Drained by the pose step on the very next fixed step.
    pub triggers: BTreeMap<Uuid, BTreeSet<String>>,
    /// What each machine is in, published by the pose step.
    pub states: BTreeMap<Uuid, AnimStateInfo>,
    /// The root motion each machine-driven character's current state produced
    /// this step (clause 2). Published by the pose step; consumed by
    /// `inf_physics::d3::movement`.
    pub root_motion: BTreeMap<Uuid, RootMotion3D>,
    /// **The `.inf_anim` v2 curve channels** of the state each machine is in,
    /// sampled at its play-head (clause 5 and clause 7).
    ///
    /// The reason the movement step can read `Enable_FootIK_L`, `FootLock_R` and
    /// `Mask_AimOffset` at all: those are properties of a *clip*, the clip
    /// resolver is a host registry, and the pose step is the one place that has
    /// both it and the play-head. Published rather than queried, which is the
    /// command-queue shape — the physics side never reaches into the machine.
    pub curves: BTreeMap<Uuid, BTreeMap<String, f32>>,
    /// Entities whose transitions enter at the **pose-matched** frame
    /// ([`inf_anim::TransitionEntry::PoseMatched`]) rather than at zero.
    ///
    /// P29.2 shipped that entry mode off by default because it changes which
    /// frame plays, and named this wave's get-up and landing consumers as the
    /// ones that turn it on. This set is how they do.
    pub pose_matched: BTreeSet<Uuid>,
}

impl AnimBridgeRes {
    /// Whether nothing at all is stored — the condition under which the pose step
    /// removes the resource rather than writing an empty one back.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
            && self.triggers.is_empty()
            && self.states.is_empty()
            && self.root_motion.is_empty()
            && self.curves.is_empty()
            && self.pose_matched.is_empty()
    }
}

/// Whether `guid` names an entity that carries an [`AnimStateMachine`] at all.
///
/// The question every write door asks first: writing a parameter for an entity
/// with no machine is not an error worth failing a handler over, but it is worth
/// reporting, and it is the difference between "the node did nothing" and "the
/// node did nothing **and told you**".
fn has_machine(world: &EcsWorld, guid: Uuid) -> bool {
    world
        .entity_of(guid)
        .and_then(|e| world.world().get::<AnimStateMachine>(e))
        .is_some()
}

/// Read-modify-write the bridge resource, creating it if absent.
fn with_bridge<R>(world: &mut EcsWorld, f: impl FnOnce(&mut AnimBridgeRes) -> R) -> R {
    let w = world.world_mut();
    if !w.contains_resource::<AnimBridgeRes>() {
        w.insert_resource(AnimBridgeRes::default());
    }
    let mut res = w.resource_mut::<AnimBridgeRes>();
    f(&mut res)
}

/// The bridge resource, if this world has one.
pub fn bridge(world: &EcsWorld) -> Option<&AnimBridgeRes> {
    world.world().get_resource::<AnimBridgeRes>()
}

/// **Set an animation parameter** on `guid`'s machine.
///
/// Returns `false` when the entity has no machine or the value is not finite —
/// both **values**, not failures: a NaN parameter would reach a condition compare
/// and a blend-space axis, and refusing it here is cheaper than a pose nobody can
/// explain.
pub fn set_anim_param(world: &mut EcsWorld, guid: Uuid, name: &str, value: f64) -> bool {
    if name.is_empty() || !value.is_finite() || !has_machine(world, guid) {
        return false;
    }
    with_bridge(world, |b| {
        b.params
            .entry(guid)
            .or_default()
            .insert(name.to_string(), value);
    });
    true
}

/// The parameter last set through [`set_anim_param`], if any.
///
/// Deliberately **not** "the value the machine read": the machine's own answer is
/// the overlay *or* the actor's variable, and a caller asking this question wants
/// to know what it set.
pub fn anim_param(world: &EcsWorld, guid: Uuid, name: &str) -> Option<f64> {
    bridge(world)?.params.get(&guid)?.get(name).copied()
}

/// **Arm a trigger** on `guid`'s machine, consumed by the next fixed step.
///
/// Returns `false` for an entity with no machine. It does **not** report whether
/// the machine declares a trigger by that name — that is not knowable here (the
/// `.inf_sm` lives in the host's registry) and it is knowable one step later, in
/// the pose step, which is where the arm actually happens.
pub fn set_anim_trigger(world: &mut EcsWorld, guid: Uuid, name: &str) -> bool {
    if name.is_empty() || !has_machine(world, guid) {
        return false;
    }
    with_bridge(world, |b| {
        b.triggers.entry(guid).or_default().insert(name.to_string());
    });
    true
}

/// What `guid`'s machine is in, as of the last fixed step.
pub fn anim_state(world: &EcsWorld, guid: Uuid) -> Option<&AnimStateInfo> {
    bridge(world)?.states.get(&guid)
}

/// Whether `guid`'s machine is in the state named `name`.
///
/// `false` for an entity that has never stepped, which is the conservative
/// reading: a state nothing has entered is not a state anything is in.
pub fn anim_state_is(world: &EcsWorld, guid: Uuid, name: &str) -> bool {
    anim_state(world, guid).is_some_and(|s| s.name == name)
}

/// How long `guid`'s machine has been in its current state, seconds; `0` when it
/// has never stepped.
pub fn anim_state_time(world: &EcsWorld, guid: Uuid) -> f64 {
    anim_state(world, guid).map(|s| s.time_s).unwrap_or(0.0)
}

/// **Consume one of this step's notifies.**
///
/// Returns `true` exactly once per fired name per step: the name is *removed*
/// from [`crate::pose::AnimEventsRes`], so two handlers racing for one footstep
/// get one footstep. That is the difference between this and
/// [`crate::pose::anim_events`], which reads without taking.
///
/// A name that did not fire answers `false` and changes nothing.
pub fn consume_anim_notify(world: &mut EcsWorld, guid: Uuid, name: &str) -> bool {
    let w = world.world_mut();
    let Some(mut res) = w.get_resource_mut::<crate::pose::AnimEventsRes>() else {
        return false;
    };
    let Some(list) = res.0.get_mut(&guid) else {
        return false;
    };
    let Some(i) = list.iter().position(|n| n == name) else {
        return false;
    };
    list.remove(i);
    if list.is_empty() {
        res.0.remove(&guid);
    }
    let empty = res.0.is_empty();
    if empty {
        w.remove_resource::<crate::pose::AnimEventsRes>();
    }
    true
}

/// Turn [`inf_anim::TransitionEntry::PoseMatched`] on or off for `guid`.
///
/// The door P29.2 named and did not build a caller for: "off by default because
/// it changes which frame plays, and that is a content decision; P29.4's get-up
/// and landing consumers turn it on."
pub fn set_pose_match_entry(world: &mut EcsWorld, guid: Uuid, on: bool) {
    with_bridge(world, |b| {
        if on {
            b.pose_matched.insert(guid);
        } else {
            b.pose_matched.remove(&guid);
        }
    });
}

/// The root motion `guid`'s machine produced on the **last** fixed step's pose
/// evaluation, in the character's own facing frame.
///
/// # One step of latency, stated
///
/// The pose is evaluated *after* the movement step in both hosts' fixed steps
/// (movement runs before the solver; the pose runs after the write-back), so the
/// movement step reads the root motion the pose step published one step earlier.
/// Over any interval the total displacement is the same, shifted by one step of
/// 1/60 s — the same one-beat latency P22.3's structural collapse documents, and
/// for the same structural reason. Moving the pose step in front of the movement
/// step would invert the dependency (the pose reads the mode the movement step
/// decides), which is worse.
pub fn anim_root_motion(world: &EcsWorld, guid: Uuid) -> Option<RootMotion3D> {
    bridge(world)?.root_motion.get(&guid).copied()
}

/// The value of `guid`'s current clip curve `name` at this step's play-head, or
/// `fallback` when the clip has no such channel.
///
/// The reference vocabulary is [`inf_anim::channels::als`]; the names are
/// free-form `String`s by Ruling 2, so a studio's own channel reads here exactly
/// like one of the twenty-five.
pub fn anim_curve(world: &EcsWorld, guid: Uuid, name: &str, fallback: f32) -> f32 {
    bridge(world)
        .and_then(|b| b.curves.get(&guid))
        .and_then(|c| c.get(name))
        .copied()
        .unwrap_or(fallback)
}

/// **Forget every bridge entry.** Called by [`crate::pose::clear_poses`], which is
/// the one door that forgets a play session's animation state.
pub fn clear_anim_bridge(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<AnimBridgeRes>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Guid;

    fn world_with_machine() -> (EcsWorld, Uuid, Uuid) {
        let mut w = EcsWorld::new();
        let e = w.spawn("rig", None);
        let plain = w.spawn("prop", None);
        let guid = w.world().get::<Guid>(e).unwrap().0;
        let other = w.world().get::<Guid>(plain).unwrap().0;
        w.world_mut().entity_mut(e).insert(AnimStateMachine {
            sm: Some(Uuid::from_u128(9)),
            ..Default::default()
        });
        (w, guid, other)
    }

    #[test]
    fn a_write_door_refuses_by_value_and_says_so() {
        let (mut w, guid, other) = world_with_machine();
        assert!(set_anim_param(&mut w, guid, "speed", 3.5));
        assert_eq!(anim_param(&w, guid, "speed"), Some(3.5));
        // No machine on that entity.
        assert!(!set_anim_param(&mut w, other, "speed", 1.0));
        assert!(!set_anim_trigger(&mut w, other, "jump"));
        // A NaN never reaches a condition compare.
        assert!(!set_anim_param(&mut w, guid, "speed", f64::NAN));
        assert_eq!(
            anim_param(&w, guid, "speed"),
            Some(3.5),
            "the NaN did not land"
        );
        // An empty name is not a name.
        assert!(!set_anim_param(&mut w, guid, "", 1.0));
        // A world nobody has written has no resource at all.
        let clean = EcsWorld::new();
        assert!(bridge(&clean).is_none());
    }

    #[test]
    fn a_trigger_is_recorded_once_and_a_repeat_is_idempotent() {
        let (mut w, guid, _) = world_with_machine();
        assert!(set_anim_trigger(&mut w, guid, "jump"));
        assert!(set_anim_trigger(&mut w, guid, "jump"));
        assert!(set_anim_trigger(&mut w, guid, "roll"));
        let b = bridge(&w).unwrap();
        let names: Vec<&str> = b.triggers[&guid].iter().map(String::as_str).collect();
        assert_eq!(names, ["jump", "roll"], "sorted, because a BTreeSet is");
    }

    #[test]
    fn a_notify_is_consumed_exactly_once() {
        let (mut w, guid, _) = world_with_machine();
        w.world_mut().insert_resource(crate::pose::AnimEventsRes(
            [(guid, vec!["footstep_l".to_string(), "land".to_string()])]
                .into_iter()
                .collect(),
        ));
        assert!(consume_anim_notify(&mut w, guid, "footstep_l"));
        assert!(
            !consume_anim_notify(&mut w, guid, "footstep_l"),
            "twice is once"
        );
        assert_eq!(crate::pose::anim_events(&w, guid), ["land"]);
        assert!(consume_anim_notify(&mut w, guid, "land"));
        // The last one out removes the resource, so "nothing fired" has exactly
        // one representation.
        assert!(w
            .world()
            .get_resource::<crate::pose::AnimEventsRes>()
            .is_none());
        assert!(!consume_anim_notify(&mut w, guid, "land"));
    }

    #[test]
    fn the_pose_match_door_toggles_and_clearing_forgets_everything() {
        let (mut w, guid, _) = world_with_machine();
        set_pose_match_entry(&mut w, guid, true);
        assert!(bridge(&w).unwrap().pose_matched.contains(&guid));
        set_pose_match_entry(&mut w, guid, false);
        assert!(!bridge(&w).unwrap().pose_matched.contains(&guid));
        set_anim_param(&mut w, guid, "x", 1.0);
        clear_anim_bridge(&mut w);
        assert!(bridge(&w).is_none());
    }
}
