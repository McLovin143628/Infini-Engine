//! **The state machine drives the DRAWN pose** (P24.1): the one fixed-step rule
//! that turns an `AnimStateMachine`'s live state into the pose both hosts render.
//!
//! # The defect this closes
//!
//! Since P11.2 the two hosts stepped `AnimStateMachine` correctly — transitions
//! fired, cross-fades aged, the runtime state was written back — and
//! [`inf_anim::eval_pose`] had **zero production callers**. Both render stores'
//! `resolve_skinned` read only `AnimPlayer.clip + t`, so a character in a machine
//! state that was not its entry state drew its *rest pose*, forever, in the
//! editor viewport and in the shipped player alike. [`AnimStateMachine`]'s own
//! doc claimed "the state machine wins"; that was true of the sim and false of
//! the renderer, and nothing in the repo compared the two. The sim agreed with
//! itself.
//!
//! # The ruling: the pose is evaluated ONCE, at fixed step, in the SIM
//!
//! Not at projection time. Three reasons, in order of weight:
//!
//! * **Determinism.** A pose evaluated per *frame* is a function of the render
//!   clock; evaluated per *fixed step* it is a function of the step history, so
//!   it can be folded into the trace ([`pose_state_bytes`]) and compared between
//!   PIE and shipping. A pose the renderer derives is a pose no gate can see.
//! * **One evaluation, two hosts.** Both projectors read the sim world, so a
//!   sim-side home is a home both can read. Evaluating in each projector would be
//!   the same two-byte-identical-loops shape the deform doctrine replaced — and
//!   the two projectors are already a MIRROR pair maintained by hand.
//! * **The sockets consume it.** [`crate::attach::update_attachments`] needs the
//!   *animated joint*, which means it needs the pose in the same fixed step that
//!   produced it. A render-time pose is a beat late and a whole host away.
//!
//! # Where it lives, and why it is a resource
//!
//! [`PoseStoreRes`] is a bevy **resource** on the sim world — verbatim the
//! reasoning [`crate::deform::DeformFieldRes`] records:
//!
//! * not a component, so **no schema moves** for the pose itself; the scene
//!   serializer walks entities and components, never resources, so there is no
//!   path from an evaluated pose to a file — which is correct, a pose is derived
//!   state and not authored content (its *inputs* — the machine, and since v21
//!   the IK target — are components, and that is the line);
//! * not a host member, because both scene projectors read the sim world and
//!   neither reads the host struct;
//! * still sim state in every sense that matters: written only from the fixed
//!   step, a pure function of the step history.
//!
//! Since P24.2 the same slot also applies **IK** ([`inf_anim::ik`]) as a post-pass
//! over the evaluated pose, from [`IkTargetsRes`] — a second resource, for the
//! same reasons and with the schema ruling written down on the type. Since P24.3
//! it also reads the **authored** [`crate::components::IkTarget`] component
//! (scene v21) through [`authored_ik_goals`], applies each goal's authored
//! `weight` as a `pslerp` back toward the pre-solve pose, and hands the
//! skeleton's [`inf_anim::JointLimit`] table to the solver so a hinge cannot be
//! bent backwards by a target that asks.
//!
//! The resource is **absent** until a machine actually evaluates a pose, and it
//! is removed again the moment none does — so a level with no `AnimStateMachine`
//! (or one whose skeletons do not resolve) is byte-identical to its pre-P24.1
//! self, `pose_state_bytes` included.
//!
//! # Why `inf-ecs` may name `inf-anim`
//!
//! `inf-anim` depends on glam / serde / thiserror / inf-asset and on nothing
//! else, so `inf-ecs → inf-anim` is a Ring-0 → Ring-0 edge with no cycle — the
//! same shape as the existing `inf-terrain` and `inf-water` edges, taken for the
//! same reason (the rule lives once). `SmRuntimeState` used to survive that
//! dependency as a hand-copied POD mirror with a conversion in each direction;
//! P29.1 retired both — it is a **type alias** for [`inf_anim::SmRuntime`] now,
//! because the `runtime` field is `#[serde(skip)]` + `#[reflect(ignore)]` and so
//! never needed the derives the mirror existed to provide.

use std::collections::BTreeMap;

use bevy_ecs::prelude::{Entity, Resource};
use glam::Mat4;
use inf_anim::{
    AnimClip, ClipRef, Pose, PoseBlender, SkeletonAsset, SmBlendMode, SmContext, StateMachine,
};
use uuid::Uuid;

use crate::components::{AnimStateMachine, Guid, SkeletalMesh, SmRuntimeState};
use crate::math::Vec3d;
use crate::world::EcsWorld;

/// The events one entity's machine emitted this fixed step (P29.1) — the notify
/// seam P29.4's `anim.*` kit will read.
///
/// Published as a **resource** for the same three reasons [`PoseStoreRes`] is
/// (no schema moves, both hosts read the sim world, the fixed step is the only
/// place it can be a function of the step history), and it is **absent** until a
/// machine emits something and removed again the moment none does — so a level
/// whose states name no notifies is byte-identical to its pre-P29.1 self.
#[derive(Debug, Clone, Default, PartialEq, Resource)]
pub struct AnimEventsRes(pub BTreeMap<Uuid, Vec<String>>);

/// One IK goal: a chain of joints, where its tip must go, and which way it
/// bends.
///
/// Model space — the same frame [`EvaluatedPose::pose`] is evaluated in — so a
/// goal is a statement about the character's own body and not about where it is
/// standing. A caller with a world-space target converts once, at its own edge.
#[derive(Debug, Clone, PartialEq)]
pub struct IkGoal {
    /// Joint indices from the chain's root to its tip, each the parent of the
    /// next. A chain that is not a parent walk is refused by
    /// [`inf_anim::solve_chain`], as a value.
    pub chain: Vec<u16>,
    /// Where the tip must go (model space, metres).
    pub target: [f32; 3],
    /// A point the middle of the chain bends toward — a knee's forward, an
    /// elbow's back. `None` keeps the pose's existing bend.
    pub pole: Option<[f32; 3]>,
    /// How much of the solve to apply, `0..=1` (P24.3).
    ///
    /// `1.0` takes a path that never touches the pre-solve pose at all, so a
    /// caller that never lowers the weight produces exactly the bytes P24.2 did.
    /// Below 1 the chain's joints are `pslerp`ed from the pre-solve rotations
    /// toward the solved ones — portable, because this feeds `state_bytes`.
    pub weight: f32,
}

impl IkGoal {
    /// A goal applied at full strength — the P24.2 shape, so a caller that has
    /// no opinion about blending does not have to have one.
    pub fn full(chain: Vec<u16>, target: [f32; 3], pole: Option<[f32; 3]>) -> Self {
        Self {
            chain,
            target,
            pole,
            weight: 1.0,
        }
    }
}

/// **Every entity's IK goals**, keyed by [`Guid`] — a bevy resource, exactly like
/// [`PoseStoreRes`].
///
/// # THE SCHEMA RULING (P24.2), and what P24.3 did with it
///
/// P24.2 recorded that an `IkTarget` **component** was the obvious home and was
/// not available: `inf_scene::EntityRecord` is a positional bincode struct with
/// one `Option<T>` field per component type, so adding one appends a field and
/// moves the wire — which is exactly how P22.2's `Destructible` took the scene
/// from v19 to **v20**. Scene v20 was frozen for that batch, so the authored
/// component was deferred here.
///
/// **P24.3 spent the bump** ([`crate::components::IkTarget`], scene v21), and
/// this resource did not go away — it became the *runtime* half of a pair:
///
/// * the **component** is authored, saved, and re-derived from the document on
///   every fixed step by [`authored_ik_goals`];
/// * this **resource** is what a script sets ([`set_ik_goals`], which the `ik.*`
///   Blueprint kit calls) and is dropped when a session ends.
///
/// [`step_pose_evaluation`] solves the authored goals first and the runtime ones
/// after, in one list, so neither source silently disables the other.
///
/// The reason to keep the resource at all is the [`crate::deform::DeformFieldRes`]
/// doctrine, which is still the right shape for what a *scripted* IK target is:
/// **written only from the fixed step's inputs and never saved**, so a foot
/// planted on a slope this frame cannot end up in the author's document. And
/// because [`step_pose_evaluation`] reads it, **both hosts inherit IK with no
/// host-side change at all**: no signature moved, no call site moved, and the two
/// fixed steps cannot drift because neither of them knows this happened.
///
/// The resource is **absent** until something sets a goal or an authored goal
/// produces a verdict, so a level with no IK is byte-identical to its pre-P24.2
/// self, `pose_state_bytes` included.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct IkTargetsRes {
    /// The goals, by entity.
    pub goals: BTreeMap<Uuid, Vec<IkGoal>>,
    /// **What the last fixed step's solves said** — one entry per goal, in the
    /// order the goals were applied (P24.2 audit M-CALLER).
    ///
    /// The step used to spell the solve `let _ = solve_chain(..)`, which made
    /// four typed error variants and every field of [`inf_anim::IkReport`]
    /// unreachable by any layer of the engine: a chain that is not a chain, a
    /// target 40 m out of reach and a clean hit were the same silence. Landing
    /// the verdict here costs one small vector per posed entity and makes
    /// "did the foot reach the ground" answerable — by a gate, by the Details
    /// panel, and by whatever eventually drives these goals.
    ///
    /// Rebuilt from scratch each step, exactly like [`PoseStoreRes`], so a stale
    /// verdict cannot outlive the goal that produced it.
    pub last: BTreeMap<Uuid, Vec<IkOutcome>>,
}

/// What one goal's solve did, this step.
#[derive(Debug, Clone, PartialEq)]
pub enum IkOutcome {
    /// The chain solved. Carries how far the tip landed from the target and
    /// whether that counts as reached.
    Solved(inf_anim::IkReport),
    /// The solve refused, by name. The pose is untouched for this chain.
    Refused(inf_anim::IkError),
    /// The entity had a goal but published no pose (no resolvable skeleton), so
    /// nothing was solved — distinct from "solved and missed".
    NotPosed,
}

/// **The write door.** Set (or replace) an entity's IK goals.
///
/// An empty list removes the entity's entry, and emptying the last entry removes
/// the resource — so "no IK" has exactly one representation and a level that
/// stops using it stops paying for it in the trace.
pub fn set_ik_goals(world: &mut EcsWorld, guid: Uuid, goals: Vec<IkGoal>) {
    let w = world.world_mut();
    if goals.is_empty() {
        let empty = match w.get_resource_mut::<IkTargetsRes>() {
            Some(mut res) => {
                res.goals.remove(&guid);
                res.last.remove(&guid);
                res.goals.is_empty()
            }
            None => return,
        };
        if empty {
            w.remove_resource::<IkTargetsRes>();
        }
        return;
    }
    if !w.contains_resource::<IkTargetsRes>() {
        w.insert_resource(IkTargetsRes::default());
    }
    w.resource_mut::<IkTargetsRes>().goals.insert(guid, goals);
}

/// The goals set for `guid`, if any — the read door the fixed step goes through.
pub fn ik_goals(world: &EcsWorld, guid: Uuid) -> Option<&[IkGoal]> {
    world
        .world()
        .get_resource::<IkTargetsRes>()
        .and_then(|r| r.goals.get(&guid))
        .map(Vec::as_slice)
}

/// **What the last fixed step's IK solves said** for `guid` (audit M-CALLER).
///
/// The observable end of the report slot: a caller — or a gate — can ask whether
/// a foot reached the ground, or which typed refusal a chain came back with,
/// instead of the whole verdict being discarded inside the step.
pub fn ik_outcomes(world: &EcsWorld, guid: Uuid) -> Option<&[IkOutcome]> {
    world
        .world()
        .get_resource::<IkTargetsRes>()
        .and_then(|r| r.last.get(&guid))
        .map(Vec::as_slice)
}

// ── the `ik.*` node kit's Ring-0 doors (P24.3) ──────────────────────────────
//
// **The kit edits the AUTHORED component, not the runtime resource**, and that
// is a decision worth stating where it is implemented.
//
// The obvious alternative was to have the nodes call [`set_ik_goals`]. It needs
// a *chain*, which is joint indices, which a Blueprint author does not have and
// could not type — deriving one from a root/tip pair needs the skeleton, and
// neither host's Blueprint dispatch context holds a skeleton registry. Threading
// one into both would be a new field in two hand-maintained mirrors, for a
// derivation the author has already done: the chain is on the component.
//
// So the split is by *what each half knows*. The component says **which joints**
// (authored once, in the Skeleton Editor); the kit says **where the goal is**
// (per frame, from gameplay). [`set_ik_goals`] stays the door for a caller that
// genuinely has a chain — the concatenation in [`step_pose_evaluation`] is what
// keeps both live at once.
//
// Every door here returns a `bool` rather than erroring: a Blueprint node is not
// a transaction, and failing the handler would take down the rest of the Tick
// body for a goal index the author fixes by typing a smaller number (the
// `voxel.*` kit's ruling, one phase on).

/// Edit one authored goal's **world-space target**. `false` when the entity has
/// no [`crate::components::IkTarget`] or the index is past its goal list.
pub fn set_authored_goal_target(
    world: &mut EcsWorld,
    guid: Uuid,
    index: usize,
    target: Vec3d,
) -> bool {
    with_authored_goal(world, guid, index, |g| g.target = target)
}

/// Edit one authored goal's **weight** (clamped to `0..=1`; a non-finite value is
/// refused rather than stored, because it would reach `pslerp`).
pub fn set_authored_goal_weight(
    world: &mut EcsWorld,
    guid: Uuid,
    index: usize,
    weight: f32,
) -> bool {
    if !weight.is_finite() {
        return false;
    }
    with_authored_goal(world, guid, index, |g| g.weight = weight.clamp(0.0, 1.0))
}

/// Turn one authored goal on or off.
pub fn set_authored_goal_enabled(
    world: &mut EcsWorld,
    guid: Uuid,
    index: usize,
    enabled: bool,
) -> bool {
    with_authored_goal(world, guid, index, |g| g.enabled = enabled)
}

fn with_authored_goal(
    world: &mut EcsWorld,
    guid: Uuid,
    index: usize,
    edit: impl FnOnce(&mut crate::components::IkGoalRecord),
) -> bool {
    let Some(e) = world.entity_of(guid) else {
        return false;
    };
    let Some(mut t) = world.world_mut().get_mut::<crate::components::IkTarget>(e) else {
        return false;
    };
    match t.goals.get_mut(index) {
        Some(g) => {
            edit(g);
            true
        }
        None => false,
    }
}

/// **Did every goal reach its target last fixed step?**
///
/// The gameplay-readable half of [`ik_outcomes`], and the reason P24.2's
/// `IkReport` is not write-only: "is the hand on the ladder rung yet" is now a
/// question a Blueprint can ask. `false` on an entity with no goals — nothing
/// reached, because nothing was asked — and on any goal that refused or was not
/// posed, which is the conservative reading.
pub fn ik_reached(world: &EcsWorld, guid: Uuid) -> bool {
    match ik_outcomes(world, guid) {
        Some(o) if !o.is_empty() => o.iter().all(|x| match x {
            IkOutcome::Solved(r) => r.reached,
            _ => false,
        }),
        _ => false,
    }
}

/// **The worst reach error across this entity's goals last step**, metres.
///
/// `0.0` when nothing was solved — the same number a perfect solve gives, and
/// deliberately so: from gameplay's side "no goal" and "goal met" are both "the
/// limb is where it should be". [`ik_outcomes`] is where the two differ.
pub fn ik_reach_error(world: &EcsWorld, guid: Uuid) -> f32 {
    ik_outcomes(world, guid)
        .map(|o| {
            o.iter()
                .filter_map(|x| match x {
                    IkOutcome::Solved(r) => Some(r.reach_error),
                    _ => None,
                })
                .fold(0.0f32, f32::max)
        })
        .unwrap_or(0.0)
}

// ── SK1b: hands ─────────────────────────────────────────────────────────────

/// **Where one hand must go**, world metres.
///
/// World and not model space, unlike [`IkGoal`], and the difference is what the
/// two are for: an authored `IkTarget` is a statement about a character's own
/// body, and a *hand* reaches for something in the level — a ladder rung, a door
/// handle, the fore-grip of the rifle it is holding. The conversion happens once,
/// inside the step, through the same `model_to_world` door the feet go through,
/// so a character standing on a rotated platform reaches correctly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandReach {
    /// Where the wrist must land (world metres).
    pub target: Vec3d,
    /// How much of the solve to apply, `0..=1`. Zero writes nothing at all.
    pub weight: f32,
}

/// **A two-handed weapon's hold** (SK1b) — the `ik_hand_gun` convention.
///
/// One hand holds the weapon; the other is *carried by* the weapon and must land
/// on its fore-grip wherever the first hand puts it. UE's rig publishes exactly
/// one bone for that job — `ik_hand_gun`, which [`inf_anim::manny`] follows the
/// right hand with — and every ALS animation is authored against it.
///
/// The offset is in the holding hand's frame, so it describes the *weapon*: a
/// rifle's fore-grip is 30 cm along the barrel from the pistol grip whichever way
/// the character is facing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GunGrip {
    /// Which hand holds the weapon. The other one follows.
    pub holding: inf_anim::BoneSide,
    /// Where the off hand sits, in the holding hand's own frame, metres.
    pub off_hand_offset: [f32; 3],
    /// How much of the off-hand solve to apply, `0..=1`.
    pub weight: f32,
}

/// **Which grip a hand is closed on**, and how far into it.
#[derive(Debug, Clone, PartialEq)]
pub struct HandGrip {
    /// The [`inf_anim::GripAffordance`] this hand is conforming to, by name.
    pub name: String,
    /// `0` is an open hand and `1` is the grip fully taken. A release is this
    /// number falling, and at zero the fingers pose exactly the bytes an
    /// ungripped hand does.
    pub amount: f32,
}

/// **One character's hand request** for this fixed step, `[left, right]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HandIk {
    /// Where each hand must reach. `None` leaves that arm to the animation.
    pub reach: [Option<HandReach>; 2],
    /// A two-handed hold, if there is one — see [`GunGrip`].
    pub gun: Option<GunGrip>,
    /// What each hand is gripping.
    pub grip: [Option<HandGrip>; 2],
}

impl HandIk {
    /// Whether this request asks for anything at all. A request that does not is
    /// the same as no request, and both cost nothing.
    pub fn is_empty(&self) -> bool {
        self.reach.iter().all(Option::is_none)
            && self.gun.is_none()
            && self.grip.iter().all(Option::is_none)
    }

    /// `[left, right]` index for a side — the array order every field here uses.
    pub fn side_index(side: inf_anim::BoneSide) -> Option<usize> {
        match side {
            inf_anim::BoneSide::Left => Some(0),
            inf_anim::BoneSide::Right => Some(1),
            inf_anim::BoneSide::Center => None,
        }
    }
}

/// What one step's hand pass did — the engagement counters a gate reads.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HandIkReport {
    /// Each arm's solve verdict, `[left, right]`.
    pub reach: [Option<IkOutcome>; 2],
    /// The off hand's solve verdict, when a [`GunGrip`] was asked for.
    pub gun: Option<IkOutcome>,
    /// Each hand's curl, `[left, right]`.
    pub grip: [inf_anim::GripReport; 2],
    /// Bones the **correction re-drive** rewrote after the solves — see
    /// [`step_pose_evaluation`]'s writer list for why it runs at all.
    pub redriven: usize,
}

impl HandIkReport {
    /// Whether this pass **wrote a pose**, as opposed to merely running.
    ///
    /// A refusal is a value here, so "the request was there" and "a bone moved"
    /// are different facts, and the correction re-drive is gated on the second
    /// one — the SK1a audit's first decision, at the pass that inherits it.
    pub fn wrote(&self) -> bool {
        self.reach
            .iter()
            .chain(std::iter::once(&self.gun))
            .any(|o| matches!(o, Some(IkOutcome::Solved(_))))
            || self.grip.iter().any(|g| g.joints > 0)
    }
}

/// **Every character's hand request**, keyed by [`Guid`] — a bevy resource, for
/// exactly [`IkTargetsRes`]'s reasons.
///
/// **No schema moves.** SK1a's ruling for this wave was one `.inf_skel` bump and
/// nothing else, and a hand request is session state by nature: what a character
/// is holding this step is not a property of the author's document, the way an
/// `IkTarget` gizmo is. It is the [`crate::deform::DeformFieldRes`] doctrine —
/// written only from the fixed step's inputs, never saved — and it means both
/// hosts inherit hand IK with no host-side change at all, because
/// [`step_pose_evaluation`] is the only thing that reads it.
///
/// **Absent until something asks.** A level that grips nothing has no resource,
/// takes one map probe per posed character, and poses exactly the bytes SK1a did.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct HandIkRes {
    /// The requests, by entity.
    pub hands: BTreeMap<Uuid, HandIk>,
    /// What the last fixed step's hand pass did, by entity. Rebuilt from scratch
    /// each step, so a stale verdict cannot outlive the request that produced it.
    pub last: BTreeMap<Uuid, HandIkReport>,
}

/// **The write door.** Set (or replace) a character's hand request.
///
/// An empty request removes the entry and emptying the last one removes the
/// resource, so "no hands" has exactly one representation — [`set_ik_goals`]'s
/// rule, for its reason.
pub fn set_hand_ik(world: &mut EcsWorld, guid: Uuid, hands: HandIk) {
    let w = world.world_mut();
    if hands.is_empty() {
        let empty = match w.get_resource_mut::<HandIkRes>() {
            Some(mut res) => {
                res.hands.remove(&guid);
                res.last.remove(&guid);
                res.hands.is_empty()
            }
            None => return,
        };
        if empty {
            w.remove_resource::<HandIkRes>();
        }
        return;
    }
    if !w.contains_resource::<HandIkRes>() {
        w.insert_resource(HandIkRes::default());
    }
    w.resource_mut::<HandIkRes>().hands.insert(guid, hands);
}

/// The hand request set for `guid`, if any.
pub fn hand_ik(world: &EcsWorld, guid: Uuid) -> Option<&HandIk> {
    world
        .world()
        .get_resource::<HandIkRes>()
        .and_then(|r| r.hands.get(&guid))
}

/// **What the last fixed step's hand pass did** for `guid` — the observable end
/// of the report slot, and what the grip gate asserts.
pub fn hand_ik_report(world: &EcsWorld, guid: Uuid) -> Option<&HandIkReport> {
    world
        .world()
        .get_resource::<HandIkRes>()
        .and_then(|r| r.last.get(&guid))
}

/// **Forget every hand request.** Called by [`clear_poses`], like
/// [`clear_ik_goals`], and for the same reason: a hold is session state.
pub fn clear_hand_ik(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<HandIkRes>();
}

/// **Forget every IK goal.** The twin of [`clear_poses`], and called by it.
///
/// Clears the **runtime** goals only — the authored [`IkTarget`] components are
/// document state and are re-read from scratch on the next step, exactly like the
/// `AnimStateMachine` they sit beside. That asymmetry is the point of the two
/// halves: what a script set is session state and is dropped when the session
/// ends; what an author placed survives, because it is in the file.
pub fn clear_ik_goals(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<IkTargetsRes>();
}

/// **The authored goals, converted for the solver** — one entry per entity that
/// carries a non-empty, enabled [`IkTarget`] (P24.3).
///
/// This is the whole of "an authored component becomes an IK goal", and it is
/// called from inside [`step_pose_evaluation`] every fixed step rather than once
/// at session start. Both halves of that matter:
///
/// * **Every step**, because a goal that follows a moving hand-hold — or a
///   character that walks while its foot is planted — is the case IK exists for.
///   Seeded once, a target would be a constant in the character's *own* frame and
///   would drift with it.
/// * **Inside the step**, because that keeps the number of doors at one. The
///   editor's `SimSession` and the player's `RuntimeSim` both call
///   `step_pose_evaluation` and neither knows this happens, which is exactly why
///   their traces cannot disagree about it.
///
/// # The frame conversion, which is the reason this function exists
///
/// [`IkGoalRecord`] is **world** space (an author places a target with a gizmo,
/// in the world) and [`IkGoal`] is **model** space (a pose is evaluated in the
/// character's own frame). The conversion is the entity's own
/// [`GlobalTransform`], inverted. An entity whose global transform is singular —
/// a zero scale — has no inverse and its goals are **dropped**, not applied
/// through a matrix full of infinities.
///
/// `target_entity` resolves through the world's GUID index; a GUID naming nothing
/// is treated as unbound (`target` is then absolute), which leaves a chain
/// reaching for the last thing it was told rather than snapping to rest when a
/// hand-hold is deleted mid-session.
pub fn authored_ik_goals(world: &mut EcsWorld) -> BTreeMap<Uuid, Vec<IkGoal>> {
    use crate::components::{GlobalTransform, IkTarget};
    // Collected (and CLONED) first, then walked in Guid order: the result must be
    // a property of the level and not of bevy's archetype layout, and the anchor
    // lookup below reads the same world this query borrows.
    let mut rows: Vec<(Uuid, IkTarget, glam::DAffine3)> = {
        let w = world.world_mut();
        let mut q = w.query::<(&Guid, &IkTarget, &GlobalTransform)>();
        q.iter(w).map(|(g, t, gt)| (g.0, t.clone(), gt.0)).collect()
    };
    rows.sort_by_key(|(g, _, _)| *g);
    let w = world.world();
    let mut out: BTreeMap<Uuid, Vec<IkGoal>> = BTreeMap::new();
    for (guid, target, global) in rows {
        if target.goals.is_empty() {
            continue;
        }
        let det = global.matrix3.determinant();
        if !det.is_finite() || det.abs() < 1.0e-12 {
            // A singular placement has no model frame to convert into. Dropping
            // the goals is the honest answer; inverting anyway makes every
            // downstream number an infinity the solver would then refuse
            // one-by-one with a NonFinite it could not explain.
            continue;
        }
        let inv = global.inverse();
        let mut goals = Vec::new();
        for g in &target.goals {
            if !g.enabled || g.chain.len() < 2 {
                continue;
            }
            let anchor = g
                .target_entity
                .get()
                .and_then(|id| world.entity_of(id))
                .and_then(|e| w.get::<GlobalTransform>(e))
                .map(|gt| gt.0.translation)
                .unwrap_or(glam::DVec3::ZERO);
            let world_target = anchor + glam::DVec3::new(g.target.x, g.target.y, g.target.z);
            let local = inv.transform_point3(world_target);
            // A pole is a DIRECTION-defining point in the same world frame, so it
            // converts the same way — but it is deliberately NOT offset by the
            // target entity: a pole says "bend toward here", and tying it to the
            // thing the hand is reaching for would swing the elbow whenever the
            // hand-hold moved.
            let pole = g.pole.map(|p| {
                let lp = inv.transform_point3(glam::DVec3::new(p.x, p.y, p.z));
                [lp.x as f32, lp.y as f32, lp.z as f32]
            });
            goals.push(IkGoal {
                chain: g.chain.clone(),
                target: [local.x as f32, local.y as f32, local.z as f32],
                pole,
                // Clamped here rather than trusted: the field is authored, and a
                // weight of 7 or of NaN must not reach `pslerp`.
                weight: if g.weight.is_finite() {
                    g.weight.clamp(0.0, 1.0)
                } else {
                    1.0
                },
            });
        }
        if !goals.is_empty() {
            out.insert(guid, goals);
        }
    }
    out
}

/// One entity's pose as the sim evaluated it this fixed step.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedPose {
    /// The `.inf_skel` GUID the pose was evaluated **against**.
    ///
    /// Load-bearing, not bookkeeping: a projector resolves its own copy of the
    /// skeleton out of its own asset store, and applying a pose evaluated against
    /// a *different* skeleton would deform a character by another character's
    /// joint hierarchy. The projectors check this before they use the pose.
    pub skeleton: Uuid,
    /// The evaluated local pose — one TRS per joint of `skeleton`.
    pub pose: Pose,
    /// Every authored socket's **model-space** transform under `pose`, as
    /// `(name, matrix)` sorted by name ([`inf_anim::socket_transforms`]).
    ///
    /// Derived here rather than by the consumer because the consumer
    /// ([`crate::attach::update_attachments`]) has the entity and the socket name
    /// and no route at all to the `.inf_skel` — the sim host that owns the
    /// skeleton registry is the only place that can compute it.
    pub sockets: Vec<(String, Mat4)>,
}

impl EvaluatedPose {
    /// The model-space transform of the socket named `name`, if the skeleton
    /// authors one.
    pub fn socket(&self, name: &str) -> Option<Mat4> {
        self.sockets
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| *m)
    }
}

/// Every entity's evaluated pose this step, keyed by the entity's stable
/// [`Guid`]. Rebuilt from scratch each fixed step (see [`step_pose_evaluation`]),
/// so a machine that stops resolving stops posing rather than freezing.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct PoseStoreRes(pub BTreeMap<Uuid, EvaluatedPose>);

/// Every posed entity's [`PoseBlender`] — the inertialization state the `Copy`
/// [`SmRuntimeState`] cannot hold (P29.2).
///
/// A blender carries the previous two rendered poses and the live decay, which is
/// one `Vec<JointTransform>` each — far too much for a struct that is inlined into
/// an ECS component. It lives here for the same reason [`PoseStoreRes`] does: it
/// is a property of a **play session**, rebuilt from nothing, never serialized, so
/// no schema moves and [`clear_poses`] is the one door that forgets it.
///
/// The map is pruned to the entities that stepped this fixed step, so an entity
/// whose machine was unbound stops carrying a decay rather than resuming one when
/// it comes back.
#[derive(Resource, Default, Debug, Clone)]
pub struct PoseBlendRes(pub BTreeMap<Uuid, PoseBlender>);

/// How this world's state transitions blend (P29.2). **Absent means
/// [`SmBlendMode::Inertialize`]**, which is §13's amendment's default.
///
/// A *setting*, not per-step state, so — unlike [`PoseBlendRes`] — it survives
/// [`clear_poses`]: an author who switched a session to the P29.1 cross-fade did
/// not mean "until the next time Simulate stops".
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoseBlendModeRes(pub SmBlendMode);

/// Choose **this world's default** for how state transitions blend.
///
/// # It is the FALLBACK now, not the authority (`.inf_sm` v3)
///
/// P29.2 shipped this as the only selector, and said why: *"a per-transition
/// choice would be a field on `SmTransition`, and that format bumped in P29.1 and
/// does not bump again in this phase."* The island phase bumped it. So the
/// precedence is now two-deep, and it is written down in **exactly one place** —
/// [`inf_anim::PoseBlender::mode_for`]:
///
/// ```text
/// the fired transition's own `blend`  (Some)  — the author's per-edge choice
///   else this resource                        — what this function sets
///   else SmBlendMode::Inertialize
/// ```
///
/// Two authorities that each believe they decide is the P29.3 slope-limit defect;
/// this one is deliberately the *weaker* of the two and says so, and no code
/// outside `mode_for` compares them.
///
/// **It did not retire**, because retiring it would make "cross-fade this whole
/// project" a per-edge edit on every transition in every machine — and would make
/// a machine authored before anyone thought about blending indistinguishable from
/// one that chose the default deliberately.
///
/// **PIE carries it** (`ScenePayload` v11): a setting the editor can change that
/// the payload does not carry is a preview that differs from the build, which is
/// the sentence P29.2 wrote this boundary down with.
///
/// Existing blenders switch **immediately** rather than after their current decay
/// expires — a setting that takes effect at an unpredictable later moment is
/// worse than one that does not exist.
pub fn set_blend_mode(world: &mut EcsWorld, mode: SmBlendMode) {
    world.world_mut().insert_resource(PoseBlendModeRes(mode));
    if let Some(mut res) = world.world_mut().get_resource_mut::<PoseBlendRes>() {
        for b in res.0.values_mut() {
            b.mode = mode;
        }
    }
}

/// How this world blends state transitions — the default when nothing set one.
pub fn blend_mode(world: &EcsWorld) -> SmBlendMode {
    world
        .world()
        .get_resource::<PoseBlendModeRes>()
        .map(|r| r.0)
        .unwrap_or_default()
}

/// The pose the sim evaluated for `guid` this step, if any.
///
/// This is the **read door** both render projectors go through, and the reason
/// `AnimStateMachine`'s "the machine wins" is now true of the renderer too.
pub fn evaluated_pose(world: &EcsWorld, guid: Uuid) -> Option<&EvaluatedPose> {
    world
        .world()
        .get_resource::<PoseStoreRes>()
        .and_then(|r| r.0.get(&guid))
}

/// The notify names `guid`'s machine emitted **this fixed step** — its exited
/// state's `on_exit` followed by its entered state's `on_enter`, in that order.
///
/// Empty on a step where nothing happened, which is the common case: the
/// resource is absent unless something fired, so this reads as `&[]` without
/// touching the world's storage. That is the same absent-costs-nothing rule
/// [`PoseStoreRes`] follows, and it is what keeps a level whose states name no
/// notifies byte-identical to its pre-P29.1 self.
///
/// **The read door P29.4's `anim.*` kit goes through.** It exists in this batch
/// rather than that one because a resource with no reader is a resource nothing
/// can prove is written — the shape the P24.1 audit closed on
/// `inf_anim::eval_pose`, which had zero production callers while its docs
/// claimed the machine drove the pose.
pub fn anim_events(world: &EcsWorld, guid: Uuid) -> &[String] {
    world
        .world()
        .get_resource::<AnimEventsRes>()
        .and_then(|r| r.0.get(&guid))
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// **The one door from model space into the world** (P29.6).
///
/// Every consumer of an [`EvaluatedPose`] — the foot bridge, the ragdoll rig,
/// the socket attachments and both render projectors — has to answer the same
/// question, *where is this rig's origin in the world*, and before this wave each
/// of them answered it with the entity's raw `GlobalTransform`. That is right for
/// a prop and wrong for a character, because the movement step keeps a
/// character's transform at its capsule **centre** while a rig is authored with
/// its **feet at the origin** (the P29.4 audit's A12 seam; the ruling and the
/// arithmetic are on [`crate::movement::feet_offset_m`]).
///
/// So the rule lives here, once. An entity with no
/// [`CharacterMovement`](crate::components::CharacterMovement) gets its
/// `GlobalTransform` unchanged — which is every prop, every non-character
/// skeletal mesh and every committed sample in the tree today, so nothing that
/// existed before this wave moves by a micrometre.
///
/// The offset is composed on the **right**, i.e. in the entity's own frame: a rig
/// belongs to its character, so a character standing on a rotated platform drops
/// along its own down axis rather than the world's.
pub fn model_to_world(world: &EcsWorld, entity: Entity) -> glam::DAffine3 {
    let w = world.world();
    let global = w
        .get::<crate::components::GlobalTransform>(entity)
        .map(|g| g.0)
        .unwrap_or(glam::DAffine3::IDENTITY);
    let Some(cm) = w.get::<crate::components::CharacterMovement>(entity) else {
        return global;
    };
    let drop = crate::movement::feet_offset_m(cm, w.get::<crate::components::Collider3D>(entity));
    global * glam::DAffine3::from_translation(glam::DVec3::new(0.0, -drop, 0.0))
}

/// [`model_to_world`] by GUID — the door a caller outside the fixed step reaches
/// through (an attachment, a projector, a gate).
pub fn model_to_world_of(world: &EcsWorld, guid: Uuid) -> Option<glam::DAffine3> {
    world.entity_of(guid).map(|e| model_to_world(world, e))
}

/// **The same lift as a world-space offset** (P29.7) — for a caller that already
/// has a translation and cannot use the whole affine.
///
/// # The caller this exists for
///
/// The cloth and hair projectors. A simulated garment's vertices are in the
/// wearer's **model space**, which is feet-at-origin character space; the
/// translation the projectors were handed is the entity's, which for a character
/// is its capsule **centre**. So a coat was drawn `feet_offset_m` — nearly a
/// metre on a 1.8 m biped — above the character wearing it. It is the P29.6
/// foot-publish seam, at the two call sites that wave did not reach, and it was
/// carried into this one by name.
///
/// [`model_to_world`] is the door and cannot be used here: the shipped player's
/// projector passes the sim's **interpolated** actor position, not the affine's,
/// and re-deriving the translation from the affine would silently drop the
/// interpolation on every garment in the game. So the offset comes back on its
/// own and the caller adds it to whatever translation it holds.
///
/// Composed in the entity's own frame (`matrix3 × down`), exactly as
/// [`model_to_world`] composes it on the right — a character on a rotated
/// platform drops along its own down axis. `DVec3::ZERO` for an entity with no
/// [`CharacterMovement`](crate::components::CharacterMovement), which is every
/// prop in the tree, so nothing that existed before this moves.
pub fn model_offset_world(world: &EcsWorld, entity: Entity) -> glam::DVec3 {
    let w = world.world();
    let Some(cm) = w.get::<crate::components::CharacterMovement>(entity) else {
        return glam::DVec3::ZERO;
    };
    let drop = crate::movement::feet_offset_m(cm, w.get::<crate::components::Collider3D>(entity));
    let m = w
        .get::<crate::components::GlobalTransform>(entity)
        .map(|g| g.0.matrix3)
        .unwrap_or(glam::DMat3::IDENTITY);
    m * glam::DVec3::new(0.0, -drop, 0.0)
}

/// How many entities the sim posed this step (`0` on a world that has never
/// posed one).
pub fn posed_count(world: &EcsWorld) -> usize {
    world
        .world()
        .get_resource::<PoseStoreRes>()
        .map(|r| r.0.len())
        .unwrap_or(0)
}

/// **Forget every evaluated pose.** Removes the resource outright, so the world
/// is byte-for-byte one that has never posed a character.
///
/// The editor calls this at BOTH ends of a Simulate session for the reason
/// [`crate::deform::clear_deformation`] documents: `SceneDoc`'s snapshot carries
/// entities and components, and `EcsWorld::clear` despawns entities — neither
/// touches a resource, so without an explicit call a stopped session's last pose
/// would keep deforming the author's document.
///
/// Idempotent, and a no-op on a world that never posed anything.
///
/// **It clears the IK goals too** (P24.2), through this one door rather than a
/// second call every caller would have to remember. A goal is the *input* the
/// pose was produced from, so leaving it behind would let a stopped session's
/// last foot plant bend the author's character on the next Simulate — the same
/// leak `clear_deformation` exists to stop, one level up.
pub fn clear_poses(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<PoseStoreRes>();
    // …and the inertialization state (P29.2), through this same door: a decay is
    // a statement about the transition that started it, and a stopped session's
    // last transition must not still be decaying into the next one's first frame.
    world.world_mut().remove_resource::<PoseBlendRes>();
    // …and the notifies (P29.1), for the same reason and through the same door:
    // an event is a statement about a fixed step, and a stopped session's last
    // step is not this one.
    world.world_mut().remove_resource::<AnimEventsRes>();
    // …and the animation bridge (P29.4): a parameter a stopped session set is
    // not a parameter the next one meant, and a pose-matched entry is a decision
    // that belongs to the get-up that made it.
    crate::anim_bridge::clear_anim_bridge(world);
    clear_ik_goals(world);
    // …and the hands (SK1b): what a character was holding is a statement about a
    // session, and a stopped one's last grip must not close the next one's first
    // frame.
    clear_hand_ik(world);
    // …and the grabs (SK1c), for the reason directly above and one more: a grab
    // is what PRODUCES a grip request, so clearing the request and leaving the
    // grab would have the next session's first fixed step put the hand straight
    // back on a thing nobody has pressed E on.
    crate::interact::clear_grabs(world);
}

/// The evaluated poses' canonical bytes, or an empty vec when nothing is posed —
/// the shape a replay / PIE trace folds, exactly like
/// [`crate::deform::deform_state_bytes`].
///
/// Appended to the sim's `state_bytes`, which is **hashed and never decoded**, so
/// this needs no version and no reader. A level that poses nothing produces an
/// empty vec and every pre-P24.1 trace is byte-identical.
///
/// The *sockets* are deliberately absent: they are a pure function of
/// `(skeleton, pose)`, so hashing them would fold the same information twice
/// while making the trace sensitive to socket authoring, which is not sim state.
/// The skeleton GUID **is** included, because a pose evaluated against a
/// different rig is a different pose even when the numbers coincide.
pub fn pose_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(store) = world.world().get_resource::<PoseStoreRes>() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    // BTreeMap: Guid order, so the bytes are a property of the level and not of
    // bevy's archetype layout.
    for (guid, ep) in &store.0 {
        out.extend_from_slice(guid.as_bytes());
        out.extend_from_slice(ep.skeleton.as_bytes());
        out.extend_from_slice(&(ep.pose.locals.len() as u32).to_le_bytes());
        for l in &ep.pose.locals {
            for v in l
                .translation
                .iter()
                .chain(l.rotation.iter())
                .chain(l.scale.iter())
            {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

/// **The fixed-step pose slot**: advance every [`AnimStateMachine`], evaluate the
/// pose it is now in, and publish it for the projectors and the attachment
/// system.
///
/// ONE function, called from both hosts' fixed steps — the strongest form the
/// MIRROR rule takes, and the same shape [`crate::deform::step_deformation`] and
/// `crate::sky::advance_weather` use.
///
/// The four resolvers are the host's own registries, threaded in rather than
/// duplicated: `machines` yields the `.inf_sm` a component's `sm` GUID names,
/// `skeletons` the `.inf_skel` the entity's [`SkeletalMesh`] names, `clips` the
/// `.inf_anim` a state's motion plays, and `vars` the actor's Blueprint variables
/// (an entity with no actor gets an empty set, so every condition/param reads
/// `0` — the documented P11.2 rule, unchanged).
///
/// Rules, in order:
///  1. an entity with no resolvable machine advances nothing and poses nothing;
///  2. the runtime is advanced first, then the pose is evaluated **for the state
///     it landed in** — so a transition is visible on the step it fires, not one
///     late;
///  3. an entity whose `SkeletalMesh.skeleton` does not resolve still advances
///     its machine (the trace is unchanged) and publishes no pose, so the
///     projectors keep the `AnimPlayer` / rest fallback;
///  4. the store is **replaced**, not merged: an entity that stopped posing this
///     step has no entry, and a step that posed nothing removes the resource.
///
/// Deterministic: targets are collected and sorted by `Guid` before anything is
/// written, and every per-entity evaluation is independent, so the result is a
/// property of the level rather than of archetype iteration order.
///
/// **`exit_time` is LIVE since P29.1.** The context is built with
/// [`SmContext::with_clip_lengths`], so a state's motion period is derived from
/// the very clips this function already resolves in order to sample the pose —
/// which is why closing it needed no new resolver, no new argument and no change
/// in either host.
///
/// The ledgered consequence was that turning it on "would move every existing
/// machine's transition timing". Measured before it was turned on: **no committed
/// machine in this repository sets `exit_time` at all** — not the character-demo
/// sample, not the wizard's generated locomotion, not any gate fixture. The two
/// `Some(0.8)`s in the tree are round-trip fixtures in `inf_anim::asset` and
/// `commands::sm`, neither of which is ever evaluated. So the retiming is real in
/// principle and empty in fact, and it is stated that way rather than carried as
/// a reason not to fix the field. What still holds is the v1 no-deadlock
/// fallback: a machine whose clips do not resolve (an unloaded pack, a lost
/// asset) has an unknown period, and an unknown period reads as **satisfied**.
pub fn step_pose_evaluation<'c>(
    world: &mut EcsWorld,
    dt: f64,
    machines: &dyn Fn(Uuid) -> Option<&'c StateMachine>,
    skeletons: &dyn Fn(Uuid) -> Option<&'c SkeletonAsset>,
    clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    vars: &dyn Fn(Uuid) -> BTreeMap<String, f64>,
) {
    // 1. Read pass — collect targets so the write-back never overlaps the query.
    let mut targets: Vec<(Entity, Uuid, Uuid, SmRuntimeState, Option<Uuid>)> = Vec::new();
    {
        let w = world.world_mut();
        let mut q = w.query::<(Entity, &Guid, &AnimStateMachine, Option<&SkeletalMesh>)>();
        for (e, g, asm, sm) in q.iter(w) {
            if let Some(sm_guid) = asm.sm {
                targets.push((e, g.0, sm_guid, asm.runtime, sm.and_then(|s| s.skeleton)));
            }
        }
    }
    if targets.is_empty() {
        // Nothing carries a machine. Drop a store from a previous step if one
        // exists (rule 4 — a character whose machine was unbound must stop
        // posing, not freeze), but never *touch* a world that never grew one:
        // `contains_resource` keeps the no-machine level byte-identical to its
        // pre-P24.1 self.
        let w = world.world_mut();
        if w.contains_resource::<PoseStoreRes>() {
            w.remove_resource::<PoseStoreRes>();
        }
        // The blenders go with them (P29.2): an unbound machine must come back
        // with nothing in flight, and this early return is the *other* way a
        // level reaches "nothing poses" — the pruning below never runs here.
        if w.contains_resource::<PoseBlendRes>() {
            w.remove_resource::<PoseBlendRes>();
        }
        // …and the bridge's PUBLISHED half (P29.4), for the third time and the
        // same reason: a level that stopped carrying machines must stop
        // answering "what state is it in" with the answer from before it did.
        if let Some(mut b) = w.get_resource_mut::<crate::anim_bridge::AnimBridgeRes>() {
            b.states.clear();
            b.root_motion.clear();
            b.curves.clear();
            // **And the feet** (P29.4 audit, A3). This list was the one published
            // map the early return did not drop, and it is the one with a reader
            // in the *other* fixed step: `d3::movement::step_feet` asks
            // `feet_of` where the pose put a foot, so a character that stopped
            // carrying a machine went on being locked to the last position a
            // pose it no longer has had put it in. The four maps are cleared
            // together here for the same reason the main path clears them
            // together — `PoseStoreRes`'s rule 4 is about all of them or none.
            b.feet.clear();
            let empty = b.is_empty();
            if empty {
                w.remove_resource::<crate::anim_bridge::AnimBridgeRes>();
            }
        }
        return;
    }
    targets.sort_by_key(|(_, guid, _, _, _)| *guid);

    // 2. Advance + evaluate.
    //
    // The IK goals are lifted out of the world FIRST, because the write-back
    // below needs `&mut EcsWorld` and a borrow of the resource would outlive it.
    // Cloned rather than referenced for the same reason; a goal is a chain and
    // two floats, and a level has a handful.
    //
    // **The AUTHORED goals come first, then the runtime ones** (P24.3). Two
    // sources, one list, in a fixed order — rather than one winning over the
    // other, which would have made `set_ik_goals` (and therefore the `ik.*`
    // Blueprint kit) a no-op on exactly the characters an author had rigged.
    // Concatenation is also what keeps `IkTargetsRes::last` readable: the verdict
    // vector lines up index-for-index with authored-then-runtime.
    let mut goals: BTreeMap<Uuid, Vec<IkGoal>> = authored_ik_goals(world);
    let runtime: BTreeMap<Uuid, Vec<IkGoal>> = world
        .world()
        .get_resource::<IkTargetsRes>()
        .map(|r| r.goals.clone())
        .unwrap_or_default();
    for (guid, list) in runtime {
        goals.entry(guid).or_default().extend(list);
    }
    // The hand requests, lifted out for the same borrow reason and cloned for
    // the same size reason (SK1b). There is no authored half: a hold is session
    // state by nature — see [`HandIkRes`].
    let hand_requests: BTreeMap<Uuid, HandIk> = world
        .world()
        .get_resource::<HandIkRes>()
        .map(|r| r.hands.clone())
        .unwrap_or_default();
    let mut hand_reports: BTreeMap<Uuid, HandIkReport> = BTreeMap::new();
    let mut posed: BTreeMap<Uuid, EvaluatedPose> = BTreeMap::new();
    let mut verdicts: BTreeMap<Uuid, Vec<IkOutcome>> = BTreeMap::new();
    let mut fired_events: BTreeMap<Uuid, Vec<String>> = BTreeMap::new();
    // **The inertialization state** (P29.2), lifted out for the same reason the
    // IK goals are: the write-back below needs `&mut EcsWorld`. `remove_resource`
    // rather than a clone — a blender holds two poses per entity and this runs
    // every fixed step.
    let mut blenders: BTreeMap<Uuid, PoseBlender> = world
        .world_mut()
        .remove_resource::<PoseBlendRes>()
        .map(|r| r.0)
        .unwrap_or_default();
    let mut blended_this_step: Vec<Uuid> = Vec::new();
    // **The animation bridge** (P29.4), lifted out for the same reason the IK
    // goals and the blenders are: the write-back below needs `&mut EcsWorld`.
    //
    // Its two INPUT maps (`params`, `pose_matched`) survive the step; its two
    // PUBLISHED ones (`states`, `root_motion`) are cleared first and rebuilt, so
    // an entity whose machine stopped resolving stops having a state rather than
    // keeping a stale one — [`PoseStoreRes`]'s rule 4, applied to the bridge.
    let mut bridge = world
        .world_mut()
        .remove_resource::<crate::anim_bridge::AnimBridgeRes>()
        .unwrap_or_default();
    bridge.states.clear();
    bridge.root_motion.clear();
    bridge.curves.clear();
    bridge.feet.clear();
    bridge.traversal.clear();
    // Read once, so a world-level setting cannot mean two things inside one step.
    let mode = blend_mode(world);
    // **The clip-length resolver that makes `exit_time` live** (P29.1). Derived
    // from the same `clips` the pose is sampled through, so there is exactly one
    // notion of how long a clip is.
    let clip_len = |c: ClipRef| clips(c).map(|a| a.duration as f64);
    for (entity, guid, sm_guid, rt_state, skeleton_id) in targets {
        let Some(machine) = machines(sm_guid) else {
            continue;
        };
        let mut outcomes: Vec<IkOutcome> = Vec::new();
        // **The bridge's parameter overlay** (P29.4). A name set through
        // `anim.set_param` shadows an actor variable of the same name, because a
        // gameplay system that has just said "landed hard" must not be silently
        // outvoted by a variable nobody updated.
        let mut actor_vars = vars(guid);
        if let Some(over) = bridge.params.get(&guid) {
            for (k, v) in over {
                actor_vars.insert(k.clone(), *v);
            }
        }
        let mut pending_pose: Option<Pose> = None;
        let mut rt = rt_state;
        // **The bridge's armed triggers** (P29.4), handed to the machine BEFORE
        // it evaluates anything, and taken rather than read — an arm is consumed
        // by the step that delivers it, and the bit it sets survives on the
        // runtime until a transition reads it as true.
        if let Some(names) = bridge.triggers.remove(&guid) {
            for n in &names {
                rt.arm_trigger(machine, n);
            }
        }
        let state_before = rt.current;
        let time_before = rt.state_time;
        let started_before = rt.started;
        {
            let lookup = |name: &str| actor_vars.get(name).copied();
            let ctx = SmContext::with_clip_lengths(&lookup, &clip_len);
            // **The blender advances the machine AND evaluates the pose** (P29.2),
            // because inertialization is not a post-pass: it has to collapse the
            // fade the transition just set up *before* `eval_pose` runs, which is
            // the whole "one evaluation instead of two". An entity with no
            // skeleton still needs the machine stepped (rule 3), so the two calls
            // are split rather than nested — `advance_only` for that case, the
            // blender for the posed one.
            let rig = skeleton_id
                .and_then(skeletons)
                .filter(|a| !a.skeleton.is_empty());
            let step = match rig {
                None => rt.advance(machine, &ctx, dt),
                Some(asset) => {
                    blended_this_step.push(guid);
                    let blender = blenders.entry(guid).or_insert_with(|| {
                        let mut b = PoseBlender::new();
                        b.mode = mode;
                        b
                    });
                    // **The pose-matched entry** (P29.4), read from the bridge
                    // every step rather than latched: P29.2 shipped it off by
                    // default and named this wave's get-up as the consumer that
                    // turns it on, and a get-up turns it off again when it is
                    // done — a machine that entered every state at its
                    // best-matching frame would never play a state's beginning.
                    blender.entry = if bridge.pose_matched.contains(&guid) {
                        inf_anim::TransitionEntry::PoseMatched
                    } else {
                        inf_anim::TransitionEntry::Restart
                    };
                    // A rig swap invalidates the captured deviation — a delta
                    // between two different joint counts is not a delta.
                    blender.fit_rig(asset.skeleton.len());
                    let (pose, step) =
                        blender.step(machine, &mut rt, &asset.skeleton, clips, &ctx, dt);
                    pending_pose = Some(pose);
                    step
                }
            };
            // ── P29.4: what the machine IS, what its clip MOVED, and which of
            //    its clip's event markers the play-head crossed ──
            //
            // All three are properties of this step and of nothing else, so they
            // are computed here — inside the one door both hosts call — rather
            // than by two host-side loops that would have to agree.
            let mut events = step.events;
            if let Some(state) = machine.states.get(rt.current) {
                bridge.states.insert(
                    guid,
                    crate::anim_bridge::AnimStateInfo {
                        index: rt.current,
                        name: state.name.clone(),
                        time_s: rt.state_time,
                        blending: rt.prev.is_some(),
                    },
                );
                // **The traversal arc** (P29.5), published outside the
                // same-state guard below because it is a property of the STATE's
                // clip and not of the interval this step covered: a mantle needs
                // the arc on the very step it enters the state, which is a step
                // that changed state by definition.
                //
                // Only for a **one-shot** — see `TraversalArc`'s docs. After
                // P29.5 every imported clip carries a root-motion track, so the
                // `looping` gate is what keeps a level of walking NPCs from
                // paying 33 samples each per fixed step for an arc nothing warps.
                if !state.looping {
                    if let inf_anim::Motion::Clip(cref) = &state.motion {
                        if let Some(arc) = clips(*cref).and_then(traversal_arc_of) {
                            bridge.traversal.insert(guid, arc);
                        }
                    }
                }
                // The clip window this step covered. A step that CHANGED state
                // has no single window — the outgoing state's tail and the
                // incoming state's head are two different clips — so it
                // contributes no root motion and fires no markers, which is the
                // honest answer and costs one step of a transition.
                let same_state = started_before && rt.current == state_before;
                if same_state {
                    // **Where the play-head is** — the single clip a `Clip` state
                    // plays, or the LEADER of a blend space (P29.5, closing
                    // P29.4's "notifies fire from single-clip states only"
                    // remainder). `inf_anim::motion_leader` is the one place that
                    // decides, so a blend space's markers ride the same timeline
                    // its followers are already warped onto.
                    //
                    // Two calls, because an interval needs both ends — and both
                    // land on the SAME clip by construction, which is worth
                    // saying rather than guarding (P29.5 audit, A10). A blend
                    // space's weights come from `ctx`, which is this step's
                    // parameter snapshot, so `motion_leader` answers the same
                    // leader at either end of the step and a comparison between
                    // them is a branch that cannot be taken. What is genuinely
                    // not detected is a leader change *between* two steps: the
                    // weights crossing means last step ended on the other clip's
                    // warped phase, so the crossover step can repeat or skip one
                    // marker. Seeing that needs the previous step's leader
                    // remembered somewhere, and the only per-character homes are
                    // reflected components — a schema move this wave does not
                    // have. Carried, named, and bounded at one marker.
                    let head = |t: f64| {
                        inf_anim::motion_leader(&state.motion, &clips, &ctx, state.looping, t)
                    };
                    if let (Some((_, t0)), Some((clip, t1))) = (
                        head(time_before * state.speed),
                        head(rt.state_time * state.speed),
                    ) {
                        // **Root motion stays single-clip.** A blend's true
                        // root delta is the weighted blend of its clips'
                        // deltas, and the leader's alone would be wrong;
                        // blend-space root motion is `root_motion.rs`'s own
                        // documented follow-up and is not smuggled in here.
                        if let (inf_anim::Motion::Clip(_), Some(asset)) = (&state.motion, rig) {
                            let d = inf_anim::root_delta_3d(
                                clip,
                                &asset.skeleton,
                                t0,
                                t1,
                                state.looping,
                            );
                            if !d.is_zero() {
                                bridge.root_motion.insert(guid, d);
                            }
                        }
                        for m in crossed_markers(&clip.markers, t0, t1, state.looping) {
                            events.push(m.to_string());
                        }
                        // The v2 curve channels at THIS step's play-head.
                        // Published rather than queried: the clip resolver is
                        // a host registry and this is the one place that has
                        // both it and the play-head, so the movement step can
                        // read `Enable_FootIK_L` without reaching into the
                        // machine (the command-queue shape).
                        //
                        // The leader's, for a blend space — which is what
                        // `Mask_FootstepSound` has to be read off for the
                        // footstep that marker just fired to be the right
                        // loudness.
                        if !clip.curves.is_empty() {
                            let mut vals: BTreeMap<String, f32> = BTreeMap::new();
                            for c in &clip.curves {
                                if let Some(v) = c.sample(t1) {
                                    vals.insert(c.name.clone(), v);
                                }
                            }
                            if !vals.is_empty() {
                                bridge.curves.insert(guid, vals);
                            }
                        }
                    }
                }
            }
            if !events.is_empty() {
                fired_events.insert(guid, events);
            }
            // Rule 3: no skeleton ⇒ the machine still steps, nothing is posed.
            if let Some(id) = skeleton_id {
                if let Some(asset) = skeletons(id) {
                    if !asset.skeleton.is_empty() {
                        let mut pose = pending_pose
                            .take()
                            .unwrap_or_else(|| Pose::rest(&asset.skeleton));
                        // ── SK1a: THE PROCEDURAL DRIVE PASS ──
                        //
                        // The bones no clip authors: twist chains take their
                        // fraction of a neighbour's roll, and the `ik_*` handles
                        // follow their FK sources. See `inf_anim::drive` for the
                        // law, and for the bound this position costs (a twist
                        // reflects the pose the ANIMATION authored, not the pose
                        // the IK below goes on to correct).
                        //
                        // Here, immediately after the layer stack and before
                        // every pass that corrects the result, because this is
                        // pose CONSTRUCTION and those are pose corrections. It
                        // joins `sample_clip` / `blend_poses` / `apply_layers` /
                        // `solve_chain` / `apply_foot_ik` / the pelvis drop / the
                        // ragdoll blend as a pose writer, at a fixed place in
                        // that list, because the I6 law
                        // `every_trace_section_is_folded_in_its_frozen_order`
                        // makes the ORDER part of the trace rather than an
                        // implementation detail.
                        //
                        // **Absent costs nothing**: a rig with no drive tables --
                        // every `.inf_skel` older than schema v3, every imported
                        // glTF -- takes two early returns and poses the bytes it
                        // posed before this existed.
                        inf_anim::drive_pose(
                            &asset.skeleton,
                            &mut pose,
                            &asset.twists,
                            &asset.ik_follow,
                        );
                        // ── P29.5: the pelvis IK offset, APPLIED ──
                        //
                        // P29.4 computed it and wrote it down (`SetPelvisIKOffset`
                        // — when one foot is on a step below the other the whole
                        // body drops, rather than the low leg straightening past
                        // its limit) and its audit's A9 recorded that nothing
                        // consumed it: "routing it into the rig is a pose edit
                        // P29.5's authoring pass owns".
                        //
                        // Here, and BEFORE the goals below, because that is the
                        // whole point — the legs have to solve from a hip that
                        // has already come down, or the drop and the reach fight
                        // each other for a frame.
                        //
                        // **Inert by construction.** The offset is zero unless
                        // the movement step's foot IK produced one, which needs
                        // a `CharacterMovement`, a rig, feet inside ALS's
                        // ±50/45 cm envelope and ground under them at different
                        // heights. Every level that has none of that poses
                        // exactly the bytes P29.4 did.
                        // **Did anything below CORRECT this pose** — the gate on
                        // the SK1b re-drive at the bottom of this block. A
                        // counter and not a guess: "the pass ran" and "the pass
                        // wrote" are different claims, and a re-drive keyed on
                        // the first would pay a whole `global_transforms` per
                        // posed character per step for a pose nothing moved.
                        let mut corrected = false;
                        let drop = pelvis_drop(world, entity);
                        if drop != 0.0 {
                            if let Some(j) = pelvis_joint(asset) {
                                if let Some(local) = pose.locals.get_mut(j) {
                                    local.translation[1] += drop;
                                    corrected = true;
                                }
                            }
                        }
                        // ── P24.2 IK: a POST-PASS, before anything reads the
                        //    pose ──
                        //
                        // Here and not in a projector: the sockets below are
                        // derived from this pose and the trace is folded from it,
                        // so IK applied later would be a pose the attachments and
                        // every determinism gate could not see. Goals are applied
                        // in the order they were set; a refusal (a chain that is
                        // not a chain, a degenerate bone, a non-finite target) is
                        // a **value**, and the pose it refused on is untouched —
                        // so one bad goal costs its own chain and nothing else.
                        for goal in goals.get(&guid).map(Vec::as_slice).unwrap_or(&[]) {
                            // **The blend snapshot, taken only when it is
                            // needed** (P24.3). At full weight nothing is
                            // captured and nothing is blended, so a level that
                            // never lowers a weight produces exactly the bytes
                            // P24.2 did — the same "absent costs nothing"
                            // discipline the resource itself follows.
                            let before: Option<Vec<(usize, [f32; 4])>> =
                                (goal.weight < 1.0).then(|| {
                                    goal.chain
                                        .iter()
                                        .map(|&j| j as usize)
                                        .filter(|&j| j < pose.locals.len())
                                        .map(|j| (j, pose.locals[j].rotation))
                                        .collect()
                                });
                            // The verdict is KEPT (audit M-CALLER): `let _ =`
                            // here made every typed refusal and every reach
                            // number unreachable by any layer.
                            let outcome = match inf_anim::solve_chain(
                                &asset.skeleton,
                                &mut pose,
                                &goal.chain,
                                glam::Vec3::from_array(goal.target),
                                goal.pole.map(glam::Vec3::from_array),
                                // **The authored joint limits** (P24.3). The
                                // asset carries them and this is the only place
                                // that has both it and the pose, so an elbow
                                // bending backwards is now a thing the engine
                                // cannot do rather than a thing no caller
                                // happened to prevent.
                                &asset.limits,
                            ) {
                                Ok(r) => IkOutcome::Solved(r),
                                Err(e) => IkOutcome::Refused(e),
                            };
                            // Blend back toward the pre-solve pose. Applied only
                            // on a SOLVE: a refusal leaves the pose untouched by
                            // contract, so blending toward a snapshot of it would
                            // be arithmetic on two equal values — and, on the
                            // arm where the snapshot is `None`, would be a
                            // silent no-op that hid the refusal's own rule.
                            if matches!(outcome, IkOutcome::Solved(_)) {
                                corrected = true;
                            }
                            if let (Some(prev), IkOutcome::Solved(_)) = (&before, &outcome) {
                                for &(j, rot) in prev {
                                    let a = glam::Quat::from_array(rot);
                                    let b = glam::Quat::from_array(pose.locals[j].rotation);
                                    // `pslerp`, not `Quat::slerp`: the P24.2
                                    // audit's M-SLERP finding — `slerp` reaches
                                    // `acos_approx` plus three `sin`s, and this
                                    // result is folded into `state_bytes`.
                                    pose.locals[j].rotation =
                                        inf_math::pslerp(a, b, goal.weight).to_array();
                                }
                            }
                            outcomes.push(outcome);
                        }
                        // ── P29.4 clause 5: **foot IK**, applied over the
                        //    P24.2 solver from the goals the movement step put
                        //    on the bridge THIS step (it runs first in both
                        //    hosts), and the feet published for the NEXT one.
                        //
                        //    Here rather than in the movement step for the
                        //    reason the whole seam exists: a foot's position is
                        //    a pose, the chain is a skeleton, and neither is
                        //    reachable from a physics world.
                        //    **Character space** (P29.6): a rig's origin is its
                        //    FEET, and a character's entity transform is its
                        //    capsule centre, so the lift goes through the one
                        //    door that knows the difference. Identity-composed
                        //    for anything that is not a character.
                        let to_world = model_to_world(world, entity);
                        if let Some(goals) = bridge.foot_ik.get(&guid).copied() {
                            corrected |= apply_foot_ik(asset, &mut pose, &goals, to_world);
                        }
                        // ── SK1b: **hands** — the arms that reach and the
                        //    fingers that close ──
                        //
                        // After the feet because a hand is the freer end: a
                        // character's stance is decided by the ground it is on,
                        // and the hand solves against the body that stance
                        // produced. Before the re-drive below, because it is a
                        // correction and that is not.
                        let hands = hand_requests.get(&guid);
                        if let Some(request) = hands {
                            let report = apply_hand_ik(asset, &mut pose, request, to_world);
                            corrected |= report.wrote();
                            hand_reports.insert(guid, report);
                        }
                        // ── SK1b: **the correction re-drive**, and the ordering
                        //    bound SK1a routed here by name ──
                        //
                        // SK1a's drive pass runs at pose CONSTRUCTION, above,
                        // and its own docs state the cost: *"a twist reflects the
                        // pose the animation authored, not the pose the IK below
                        // goes on to correct"* — a foot IK solve that rolls an
                        // ankle 20 degrees left `calf_twist_01_l` showing the
                        // pre-solve roll. That was routed to this wave "with the
                        // measurement it needs (run the pass twice, or re-drive
                        // per chain)".
                        //
                        // **It runs twice**, and the decision is that a twist
                        // bone is a statement about the pose that is FINALLY
                        // published, so it must be the last thing computed from
                        // it — the same reading `foot_states` already has one
                        // line below. Re-driving per chain was the alternative
                        // and is worse in the way that matters: it puts the
                        // knowledge of which twists belong to which limb inside
                        // every solver, in three places, which is the shape the
                        // role table exists to retire.
                        //
                        // **Gated on a correction having happened**, so a
                        // character the passes above did not touch pays nothing
                        // and poses byte-identical bytes — and a rig with no
                        // drive tables (every `.inf_skel` older than v3, every
                        // canonical biped, every imported glTF) takes the same
                        // two early returns it always did whether the gate opens
                        // or not.
                        if corrected {
                            let n = redrive(asset, &mut pose);
                            if let Some(r) = hand_reports.get_mut(&guid) {
                                r.redriven = n;
                            }
                        }
                        let feet = foot_states(asset, &pose, to_world);
                        if feet.iter().any(Option::is_some) {
                            bridge.feet.insert(guid, feet);
                        }
                        // **The ragdoll rig, on request** (P29.4, clause 6).
                        // Model-space joint positions, lifted into the world by
                        // the entity's own transform -- this is the one place
                        // that has the skeleton, the pose AND the placement, so
                        // it is the only place the answer can be computed.
                        if bridge.ragdoll_requested.remove(&guid) {
                            // The same character-space lift the feet take — a
                            // ragdoll that spawned half a capsule above its own
                            // body would be the A12 seam wearing a different hat.
                            if let Some(bones) = rig_bones(asset, &pose, to_world) {
                                bridge.ragdoll_rig.insert(guid, bones);
                            }
                        }
                        let sockets =
                            inf_anim::socket_transforms(&asset.skeleton, &pose, &asset.sockets);
                        posed.insert(
                            guid,
                            EvaluatedPose {
                                skeleton: id,
                                pose,
                                sockets,
                            },
                        );
                    }
                }
            }
        }
        if !goals.get(&guid).map(Vec::is_empty).unwrap_or(true) {
            // A goal that produced no outcome means the entity published no pose
            // at all — a distinct answer from "solved and missed".
            if outcomes.is_empty() {
                outcomes.push(IkOutcome::NotPosed);
            }
            verdicts.insert(guid, outcomes);
        }
        if let Some(mut asm) = world.world_mut().get_mut::<AnimStateMachine>(entity) {
            asm.runtime = rt;
        }
    }

    // 3. Publish (rule 4).
    {
        // The blenders first, and **pruned to the entities that posed this
        // step**: rule 4 says the store is replaced rather than merged, and a
        // blender is store-shaped. An entity whose machine was unbound must come
        // back with no decay in flight, not resume one from before it left.
        blenders.retain(|g, _| blended_this_step.contains(g));
        let w = world.world_mut();
        if blenders.is_empty() {
            // Never *touch* a world that never posed one: `contains_resource`
            // keeps a machine-free level byte-identical to its pre-P29.2 self.
            if w.contains_resource::<PoseBlendRes>() {
                w.remove_resource::<PoseBlendRes>();
            }
        } else {
            w.insert_resource(PoseBlendRes(blenders));
        }
    }
    {
        // The hand verdicts, under rule 4: rebuilt from scratch, and never
        // created for a level that asked for nothing (SK1b).
        let w = world.world_mut();
        if let Some(mut res) = w.get_resource_mut::<HandIkRes>() {
            res.last = hand_reports;
        }
    }
    {
        let w = world.world_mut();
        match w.get_resource_mut::<IkTargetsRes>() {
            Some(mut res) => res.last = verdicts,
            // **An AUTHORED goal has no resource to land its verdict in** — the
            // resource is created by `set_ik_goals`, and P24.3's goals never go
            // through it. Created here, and only when there is something to say,
            // so a level with no IK at all still never grows one.
            None if !verdicts.is_empty() => {
                w.insert_resource(IkTargetsRes {
                    goals: BTreeMap::new(),
                    last: verdicts,
                });
            }
            None => {}
        }
    }
    let w = world.world_mut();
    // The notify seam, under rule 4 as well: a step that emitted nothing leaves
    // no resource behind, so "what fired this step" can never be a stale answer
    // from an earlier one.
    if fired_events.is_empty() {
        if w.contains_resource::<AnimEventsRes>() {
            w.remove_resource::<AnimEventsRes>();
        }
    } else {
        w.insert_resource(AnimEventsRes(fired_events));
    }
    if posed.is_empty() {
        if w.contains_resource::<PoseStoreRes>() {
            w.remove_resource::<PoseStoreRes>();
        }
    } else {
        w.insert_resource(PoseStoreRes(posed));
    }
    // The bridge, under the same rule: an empty one is no resource at all, so a
    // level with no bridge traffic is byte-identical to its pre-P29.4 self.
    if bridge.is_empty() {
        if w.contains_resource::<crate::anim_bridge::AnimBridgeRes>() {
            w.remove_resource::<crate::anim_bridge::AnimBridgeRes>();
        }
    } else {
        w.insert_resource(bridge);
    }
}

/// The **foot joints** of a skeleton, `[left, right]`, by name.
///
/// The vocabulary is deliberately generous, because a rig arrives from wherever
/// it arrives: ALS's `ik_foot_l`/`ik_foot_r`, this engine's own template
/// (`Foot.L` — see [`inf_anim::template`]), and the Mixamo/UE spellings
/// (`LeftFoot`, `foot_r`). The match is case-insensitive, requires the word
/// "foot", and takes the side from a trailing `l`/`r` or a leading `left`/`right`
/// — and the FIRST match wins, so a rig with a toe joint named `foot_l_end` does
/// not displace `foot_l`.
fn foot_joints(rig: &inf_anim::SkeletonAsset) -> [Option<u16>; 2] {
    // **The role table first** (SK1a). A rig that says which of its bones are
    // `BoneRoleKind::Foot` is answered from what it says, and no spelling can
    // move the answer -- which matters here more than anywhere else, because this
    // is the function that decides which joint foot IK drives, and the UE
    // convention ships a rig carrying BOTH `foot_l` and `ik_foot_l`. Read off the
    // name, the wrong one of those solves perfectly: the chain derived from
    // `ik_foot_l` is `[root, ik_foot_root, ik_foot_l]`, which is not a leg, and
    // the character walks with the marker instead of the limb.
    let roles = rig.role_index();
    if !roles.is_empty() {
        let by_role = [
            roles.first(inf_anim::BoneRoleKind::Foot, inf_anim::BoneSide::Left),
            roles.first(inf_anim::BoneRoleKind::Foot, inf_anim::BoneSide::Right),
        ];
        if by_role.iter().any(Option::is_some) {
            return by_role;
        }
    }
    foot_joints_by_name(&rig.skeleton)
}

/// The **legacy** foot rule, for a rig that carries no role table -- see
/// [`foot_joints`], its one caller, for what changed and why.
fn foot_joints_by_name(skeleton: &inf_anim::Skeleton) -> [Option<u16>; 2] {
    let mut out: [Option<u16>; 2] = [None, None];
    for (i, j) in skeleton.joints().iter().enumerate() {
        if i > u16::MAX as usize {
            break;
        }
        let n = j.name.to_ascii_lowercase();
        if !n.contains("foot") {
            continue;
        }
        let side = if n.starts_with("left") || n.ends_with("_l") || n.ends_with(".l") {
            0
        } else if n.starts_with("right") || n.ends_with("_r") || n.ends_with(".r") {
            1
        } else {
            continue;
        };
        if out[side].is_none() {
            out[side] = Some(i as u16);
        }
    }
    out
}

/// Where the pose put each foot, in world space.
fn foot_states(
    rig: &inf_anim::SkeletonAsset,
    pose: &Pose,
    model_to_world: glam::DAffine3,
) -> [Option<crate::anim_bridge::FootState>; 2] {
    let skeleton = &rig.skeleton;
    let joints = foot_joints(rig);
    if joints.iter().all(Option::is_none) {
        return [None, None];
    }
    let globals = inf_anim::global_transforms(skeleton, pose);
    let mut out = [None, None];
    for (side, j) in joints.iter().enumerate() {
        let Some(j) = *j else { continue };
        let Some(m) = globals.get(j as usize) else {
            continue;
        };
        let p = m.w_axis.truncate();
        let w =
            model_to_world.transform_point3(glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64));
        out[side] = Some(crate::anim_bridge::FootState {
            joint: j,
            world: Vec3d::new(w.x, w.y, w.z),
        });
    }
    out
}

/// Solve each foot toward its goal, over the P24.2 chain solver.
///
/// The chain is **derived**, not authored: a foot, its parent (the shin) and its
/// grandparent (the thigh) are the two bones every biped IK solver uses, and the
/// skeleton already says which joints those are. An author who wants a different
/// chain uses the authored [`crate::components::IkTarget`], which this pass does
/// not disturb — both run, in that order, exactly like the runtime goals.
///
/// A refusal is a **value**: a chain that is not a chain, a degenerate bone or a
/// non-finite target leaves the pose untouched and costs its own foot.
///
/// Returns whether it **wrote a pose** (SK1b) — the engagement counter the
/// correction re-drive is gated on, and the answer to "was this character's pose
/// corrected at all", which nothing could ask before.
fn apply_foot_ik(
    rig: &inf_anim::SkeletonAsset,
    pose: &mut Pose,
    goals: &[Option<crate::anim_bridge::FootGoal>; 2],
    model_to_world: glam::DAffine3,
) -> bool {
    let mut wrote = false;
    let skeleton = &rig.skeleton;
    let joints = foot_joints(rig);
    let to_model = model_to_world.inverse();
    for (side, goal) in goals.iter().enumerate() {
        let (Some(goal), Some(foot)) = (goal, joints[side]) else {
            continue;
        };
        if goal.weight.is_nan() || goal.weight <= 0.0 {
            continue;
        }
        let all = skeleton.joints();
        let Some(shin) = all.get(foot as usize).and_then(|j| j.parent) else {
            continue;
        };
        let Some(thigh) = all.get(shin as usize).and_then(|j| j.parent) else {
            continue;
        };
        let t = to_model.transform_point3(goal.target.to_dvec3());
        let target = glam::Vec3::new(t.x as f32, t.y as f32, t.z as f32);
        if !target.is_finite() {
            continue;
        }
        let before: Vec<(usize, [f32; 4])> = [thigh, shin, foot]
            .iter()
            .map(|&j| j as usize)
            .filter(|&j| j < pose.locals.len())
            .map(|j| (j, pose.locals[j].rotation))
            .collect();
        let chain = [thigh, shin, foot];
        if inf_anim::solve_chain(skeleton, pose, &chain, target, None, &[]).is_ok() {
            wrote = true;
            if goal.weight < 1.0 {
                for (j, rot) in before {
                    let a = glam::Quat::from_array(rot);
                    let b = glam::Quat::from_array(pose.locals[j].rotation);
                    pose.locals[j].rotation = inf_math::pslerp(a, b, goal.weight).to_array();
                }
            }
        }
    }
    wrote
}

/// **Re-run the drive pass over a corrected pose** (SK1b) — the second half of
/// the twist/IK ordering bound SK1a routed to this wave.
///
/// A named function rather than a second inline call, and the reason is the pin:
/// `every_pose_writer_runs_in_its_frozen_order` searches the step's body for each
/// writer's spelling and asserts their order, and two identical
/// `inf_anim::drive_pose(` occurrences are one needle that finds the first of
/// them. A writer a gate cannot name is a writer a gate cannot see move.
fn redrive(rig: &inf_anim::SkeletonAsset, pose: &mut Pose) -> usize {
    inf_anim::drive_pose(&rig.skeleton, pose, &rig.twists, &rig.ik_follow)
}

/// **The hand pass** (SK1b): the arms reach, the off hand follows the weapon, and
/// the fingers close.
///
/// Three things in one function because they are one *ordering*, and the order is
/// the content:
///
/// 1. **Each reaching hand solves** over the three-joint arm chain the role table
///    names, with a real pole — an elbow has no opinion of its own about which
///    way to fold, because the mannequin's bind pose is a T-pose of pure
///    translations and shoulder, elbow and wrist are exactly collinear in it.
/// 2. **The off hand follows the weapon**, second, because where the weapon *is*
///    is decided by the hand that holds it and that hand may have just moved. The
///    frame is `ik_hand_gun`'s when the rig publishes one — the UE convention,
///    which is the whole reason [`inf_anim::manny`] carries the bone — and the
///    holding hand's own global when it does not, so a rig without the marker
///    still holds a rifle with two hands.
/// 3. **The fingers close**, last, because a curl is expressed in each finger's
///    own bind frame and is therefore unaffected by where the arm ended up —
///    running it first would work, and would be a claim about the solver's
///    internals rather than about anatomy.
///
/// A refusal is a **value** throughout, and it costs its own hand.
fn apply_hand_ik(
    rig: &inf_anim::SkeletonAsset,
    pose: &mut Pose,
    request: &HandIk,
    model_to_world: glam::DAffine3,
) -> HandIkReport {
    use inf_anim::BoneSide;

    let mut report = HandIkReport::default();
    if request.is_empty() {
        return report;
    }
    let skeleton = &rig.skeleton;
    let roles = rig.role_index();
    let to_model = model_to_world.inverse();
    let sides = [BoneSide::Left, BoneSide::Right];
    // The arm chains, once. `None` for a side this rig has no arm on, which is
    // every quadruped and every rig whose hand is called something nobody
    // guessed — and which costs that side and nothing else.
    let chains = sides.map(|s| inf_anim::arm_chain(skeleton, roles, s));

    // Model space from world, checked rather than trusted: a target that is not
    // finite would reach `solve_chain`, which refuses it by name — but refusing
    // it here keeps the refusal about the REQUEST rather than about the chain.
    let into_model = |t: Vec3d| -> Option<glam::Vec3> {
        let p = to_model.transform_point3(t.to_dvec3());
        let v = glam::Vec3::new(p.x as f32, p.y as f32, p.z as f32);
        v.is_finite().then_some(v)
    };

    // 1. the reaches
    for (side, reach) in request.reach.iter().enumerate() {
        let (Some(reach), Some(chain)) = (reach, chains[side]) else {
            continue;
        };
        if !reach.weight.is_finite() || reach.weight <= 0.0 {
            continue;
        }
        let Some(target) = into_model(reach.target) else {
            continue;
        };
        report.reach[side] = Some(solve_arm(rig, pose, chain, target, reach.weight));
    }

    // 2. the off hand, carried by the weapon
    if let Some(gun) = request.gun {
        // **The handle first.** `ik_hand_gun` FOLLOWS `hand_r` (SK1a's
        // `IkFollow` table), and the drive pass that puts it there ran at pose
        // construction — before the reach above moved the hand it follows. Read
        // without this, the weapon's frame is where the animation left it and the
        // off hand lands a hand-span away from the fore-grip: measured 0.42 m
        // apart on a weapon that is 0.30 m long. Re-driving the handles here is
        // one global pass and it is what makes the marker mean anything.
        inf_anim::drive_ik_follow(skeleton, pose, &rig.ik_follow);
        let holding = HandIk::side_index(gun.holding);
        let off = holding.map(|h| 1 - h);
        if let (Some(holding), Some(off)) = (holding, off) {
            // Where the weapon's fore-grip is: the `ik_hand_gun` handle's frame
            // when the rig has one, else the holding hand's own.
            let handle = skeleton
                .index_of("ik_hand_gun")
                .filter(|_| gun.holding == BoneSide::Right)
                .or_else(|| chains[holding].map(|c| c[2]));
            if let (Some(handle), Some(chain)) = (handle, chains[off]) {
                let globals = inf_anim::global_transforms(skeleton, pose);
                if let Some(frame) = globals.get(handle as usize) {
                    let target =
                        frame.transform_point3(glam::Vec3::from_array(gun.off_hand_offset));
                    if target.is_finite() && gun.weight.is_finite() && gun.weight > 0.0 {
                        report.gun = Some(solve_arm(rig, pose, chain, target, gun.weight));
                    }
                }
            }
        }
    }

    // 3. the fingers
    for (side, grip) in request.grip.iter().enumerate() {
        let Some(grip) = grip else { continue };
        let Some(affordance) = rig.grips.iter().find(|g| g.name == grip.name) else {
            continue;
        };
        // The hand the affordance names, not the hand the array index implies: a
        // rig authors one affordance per hand, and the affordance's own `hand`
        // field is the authoritative answer to which one this is.
        let Some(hand) = inf_anim::hand_of(skeleton, roles, affordance.hand) else {
            continue;
        };
        report.grip[side] =
            inf_anim::apply_grip(skeleton, pose, &hand, &rig.limits, affordance, grip.amount);
    }
    report
}

/// One arm's solve — the half [`apply_hand_ik`] does twice.
///
/// Through [`inf_anim::reach`] rather than `solve_chain`, and the difference is
/// measured rather than stylistic: an elbow is a **hinge**, a pole-driven solve
/// picks a bend plane that is not the hinge's, and the clamp then throws away
/// whatever part of the bend does not lie in it. On the mannequin that cost
/// 8.3 cm of reach and iterating the pole was a fixed point. `reach` sets the
/// elbow from the distance and aims the assembly, which is exact.
///
/// The weight blend is `apply_foot_ik`'s, spelled the same way for the same
/// reason: at full weight nothing is captured and nothing is blended, so a caller
/// that never lowers it produces exactly the bytes an unweighted solve would.
fn solve_arm(
    rig: &inf_anim::SkeletonAsset,
    pose: &mut Pose,
    chain: [u16; 3],
    target: glam::Vec3,
    weight: f32,
) -> IkOutcome {
    let skeleton = &rig.skeleton;
    let before: Option<Vec<(usize, [f32; 4])>> = (weight < 1.0).then(|| {
        chain
            .iter()
            .map(|&j| j as usize)
            .filter(|&j| j < pose.locals.len())
            .map(|j| (j, pose.locals[j].rotation))
            .collect()
    });
    match inf_anim::reach(
        skeleton,
        pose,
        chain,
        target,
        // The authored limits, so an elbow cannot bend backwards to reach — and,
        // since SK1b, so a cone on any joint of the chain is applied too.
        &rig.limits,
    ) {
        Ok(r) => {
            if let Some(prev) = before {
                for (j, rot) in prev {
                    let a = glam::Quat::from_array(rot);
                    let b = glam::Quat::from_array(pose.locals[j].rotation);
                    pose.locals[j].rotation = inf_math::pslerp(a, b, weight).to_array();
                }
            }
            IkOutcome::Solved(r)
        }
        Err(e) => IkOutcome::Refused(e),
    }
}

/// The world-space bone segments of a posed skeleton (P29.4, clause 6).
///
/// Each joint contributes one bone: from its own model-space position to its
/// **first child**'s, or — for a leaf — a short stub along the direction it came
/// from, so a hand or a foot still has a body rather than a point. The result is
/// in world space, because that is where the articulated bodies are spawned.
///
/// `None` for a skeleton with no joints. A rig with no *classifiable* joint is
/// still a `Some`: which names are bones is `inf_physics::ragdoll::classify`'s
/// question, and this side does not get to answer it.
fn rig_bones(
    rig: &inf_anim::SkeletonAsset,
    pose: &Pose,
    model_to_world: glam::DAffine3,
) -> Option<Vec<crate::anim_bridge::RigBone>> {
    let skeleton = &rig.skeleton;
    let roles = rig.role_index();
    let joints = skeleton.joints();
    if joints.is_empty() {
        return None;
    }
    let globals = inf_anim::global_transforms(skeleton, pose);
    // Model space is `f32` (the pose pipeline's own precision) and the world is
    // `f64` (architecture rule 3), so the widening happens exactly once, here,
    // at the boundary between them.
    let pos = |i: usize| -> Vec3d {
        let m = globals[i].w_axis.truncate();
        let w =
            model_to_world.transform_point3(glam::DVec3::new(m.x as f64, m.y as f64, m.z as f64));
        Vec3d::new(w.x, w.y, w.z)
    };
    // **The anatomical successor, not the first child** (SK1a).
    //
    // A bone spans from itself to the next bone DOWN THE LIMB, and on a rig that
    // interleaves driven bones with deform ones those are different joints: the
    // first child of `lowerarm_l` in the mannequin's own index order is
    // `lowerarm_twist_02_l`, a driven bone one third of the way to the wrist, and
    // a forearm capsule built from it covers a third of the forearm and leaves the
    // rest of the arm bare. With a role table the tail is the first child the
    // table calls a deform bone; without one it is the first child, exactly as
    // before.
    let mut first_child: Vec<Option<usize>> = vec![None; joints.len()];
    for (i, j) in joints.iter().enumerate() {
        if let Some(p) = j.parent {
            let p = p as usize;
            if p < first_child.len() && first_child[p].is_none() {
                first_child[p] = Some(i);
            }
        }
    }
    let mut successor: Vec<Option<usize>> = first_child.clone();
    if !roles.is_empty() {
        for (i, slot) in successor.iter_mut().enumerate() {
            if let Some(c) = roles.deform_child(skeleton, i as u16) {
                *slot = Some(c as usize);
            }
        }
    }
    let mut out = Vec::with_capacity(joints.len());
    for (i, j) in joints.iter().enumerate() {
        let head = pos(i);
        let tail = match successor[i] {
            Some(c) => pos(c),
            // A leaf: a stub along the direction it came from, so the body has a
            // length. A zero-length capsule is not a capsule.
            None => match j.parent {
                Some(p) => {
                    let up = pos(p as usize);
                    Vec3d::new(
                        head.x + (head.x - up.x) * 0.5,
                        head.y + (head.y - up.y) * 0.5,
                        head.z + (head.z - up.z) * 0.5,
                    )
                }
                None => Vec3d::new(head.x, head.y + 0.1, head.z),
            },
        };
        out.push(crate::anim_bridge::RigBone {
            name: j.name.clone(),
            head,
            tail,
            parent: j.parent,
            role: roles.role_of(i as u16),
        });
    }
    Some(out)
}

/// **Resample a clip's baked root motion onto the traversal-arc grid** (P29.5).
///
/// `None` for a clip that carries no track — which is a value the mantle already
/// handles, and the reason a project with no derived traversal content behaves
/// exactly as it did before this existed.
///
/// The grid is normalized over the clip's whole timeline rather than over a warp
/// window, and that bound is stated rather than implied: a clip whose traversal
/// occupies only its middle third publishes an arc whose first and last thirds
/// are flat, and a warp over it spends those thirds correcting additively. A
/// per-clip warp window ([`inf_anim::WarpWindow`]) is where that stops being
/// true. P29.7 gave that type its first consumer — the seat warp, in
/// `inf_physics::d3::movement::step_driving` — but no *clip* carries one, so
/// this bound stands.
fn traversal_arc_of(clip: &inf_anim::AnimClip) -> Option<crate::anim_bridge::TraversalArc> {
    let track = clip.root_motion.as_ref()?;
    if track.times.len() < 2 || !clip.duration.is_finite() || clip.duration <= 0.0 {
        return None;
    }
    let n = crate::anim_bridge::TRAVERSAL_ARC_SAMPLES;
    let mut samples = Vec::with_capacity(n);
    let mut yaw_rad = Vec::with_capacity(n);
    for k in 0..n {
        let t = clip.duration * (k as f32) / (n as f32 - 1.0);
        let (p, yaw) = track.sample(t)?;
        samples.push(glam::Vec3::from_array(p));
        yaw_rad.push(yaw);
    }
    Some(crate::anim_bridge::TraversalArc { samples, yaw_rad })
}

/// **How far the hips must come down this step**, metres (P29.5).
///
/// Read off the character's own [`crate::components::MovementRuntime`], which is
/// where the movement step wrote it earlier in this same fixed step — so there
/// is no latency and no second computation. `0` for an entity with no character
/// movement at all, which is every entity in a level that has none.
fn pelvis_drop(world: &EcsWorld, entity: bevy_ecs::entity::Entity) -> f32 {
    world
        .world()
        .get::<crate::components::CharacterMovement>(entity)
        .map(|cm| cm.runtime.pelvis_offset.y as f32)
        .filter(|d| d.is_finite())
        .unwrap_or(0.0)
}

/// The joint a pelvis offset moves.
///
/// A joint named `pelvis` when the rig has one (the UEFN / ALS convention), else
/// the **root** joint — which is what [`inf_anim::build_template`] names `hips`
/// and is the same bone under a different vocabulary.
///
/// The offset is applied to that joint's **local** Y, which is model-space Y for
/// any rig whose ancestors above the pelvis are unrotated — true of every rig
/// this engine generates and of the imported convention. A rig that rotates the
/// bone above its own pelvis would tilt the drop; that bound is written down
/// rather than guarded, because guarding it needs a global-transform pass this
/// post-pass deliberately does not run.
fn pelvis_joint(rig: &inf_anim::SkeletonAsset) -> Option<usize> {
    // The role table first (SK1a): a rig that names its own pelvis is not asked
    // what that bone is called. The two name rules stay as the fallback, in the
    // order they were already in.
    if let Some(j) = rig
        .role_index()
        .first(inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
    {
        return Some(j as usize);
    }
    let skeleton = &rig.skeleton;
    skeleton
        .joints()
        .iter()
        .position(|j| j.name.eq_ignore_ascii_case("pelvis"))
        .or_else(|| inf_anim::root_joint_index(skeleton))
}

/// The **event** markers (`group` empty — see [`inf_anim::AnimMarker`]) whose
/// times the play-head crossed going from `t0` to `t1`.
///
/// Half-open at the start and closed at the end (`t0 < m <= t1`), so a marker is
/// crossed exactly once no matter how the steps land on it. When `looping` and
/// the interval wrapped past the clip's end, the window is the union of the two
/// pieces — a footstep at 0.05 s must not be lost because the step straddled the
/// loop point.
///
/// Sync-group markers are deliberately skipped: they exist to align two clips
/// ([`inf_anim::sync`]) and firing them as notifies would ring a footstep for
/// every phase alignment.
fn crossed_markers(markers: &[inf_anim::AnimMarker], t0: f32, t1: f32, looping: bool) -> Vec<&str> {
    if markers.is_empty() || t0 == t1 {
        return Vec::new();
    }
    let wrapped = looping && t1 < t0;
    markers
        .iter()
        .filter(|m| !m.is_sync() && m.time_s.is_finite())
        .filter(|m| {
            if wrapped {
                m.time_s > t0 || m.time_s <= t1
            } else {
                m.time_s > t0 && m.time_s <= t1
            }
        })
        .map(|m| m.name.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_anim::{
        AnimClip, Interpolation, Joint, JointTrack, JointTransform, QuatTrack, Skeleton, SmState,
        SmTransition, StateMachine,
    };

    const SKEL: Uuid = Uuid::from_u128(0xA11_0001);
    const SM: Uuid = Uuid::from_u128(0xA11_0002);
    const IDLE: ClipRef = [1; 16];
    const WAVE: ClipRef = [2; 16];

    /// A 2-joint chain: root, then a tip 1 m up.
    fn skeleton_asset() -> SkeletonAsset {
        let sk = Skeleton::new(vec![
            Joint {
                name: "root".into(),
                parent: None,
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            },
            Joint {
                name: "tip".into(),
                parent: Some(0),
                inverse_bind: Mat4::from_translation(glam::Vec3::new(0.0, -1.0, 0.0))
                    .to_cols_array(),
                local_bind: JointTransform::from_trs(
                    glam::Vec3::Y,
                    glam::Quat::IDENTITY,
                    glam::Vec3::ONE,
                ),
            },
        ])
        .unwrap();
        SkeletonAsset::with_sockets(sk, vec![inf_anim::Socket::new("hand_r", 1)])
    }

    /// A clip that bends the tip joint from 45° to 90° about X over one second.
    ///
    /// Its **first key is already bent**, deliberately. A transition resets
    /// `state_time` to zero, so a clip that started at the identity would be
    /// indistinguishable from the rest pose on the very step the transition
    /// fires — the assertion would then be measuring "one step later", not "the
    /// machine drives the pose".
    fn wave_clip() -> AnimClip {
        // `AnimClip::new` since `.inf_anim` v2 (P29.2): `duration` is derived
        // from the keys (1.0 here, unchanged).
        AnimClip::new(
            "wave",
            vec![JointTrack {
                joint: 1,
                translation: None,
                rotation: Some(QuatTrack::new(
                    vec![0.0, 1.0],
                    vec![
                        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_4).to_array(),
                        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2).to_array(),
                    ],
                    Interpolation::Linear,
                )),
                scale: None,
            }],
        )
    }

    /// idle → wave when `moving > 0.5`. Idle plays a clip nothing resolves, so
    /// its pose is the rest pose and the two states are distinguishable.
    fn machine() -> StateMachine {
        StateMachine {
            states: vec![SmState::clip("idle", IDLE), SmState::clip("wave", WAVE)],
            transitions: vec![SmTransition::on(
                0,
                1,
                0.0,
                "moving",
                inf_anim::CmpOp::Gt,
                0.5,
            )],
            entry: 0,
            ..Default::default()
        }
    }

    /// A world with one machine-driven skeletal character.
    fn world_with_character(guid: Uuid) -> EcsWorld {
        let mut world = EcsWorld::new();
        world.world_mut().spawn((
            Guid(guid),
            AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(9)),
                skeleton: Some(SKEL),
            },
        ));
        world.reindex_guids();
        world
    }

    struct Fixture {
        machine: StateMachine,
        skeleton: SkeletonAsset,
        clip: AnimClip,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                machine: machine(),
                skeleton: skeleton_asset(),
                clip: wave_clip(),
            }
        }

        fn step(&self, world: &mut EcsWorld, dt: f64, moving: f64) {
            let machines = |g: Uuid| (g == SM).then_some(&self.machine);
            let skeletons = |g: Uuid| (g == SKEL).then_some(&self.skeleton);
            let clips = |c: ClipRef| (c == WAVE).then_some(&self.clip);
            let vars = |_: Uuid| BTreeMap::from([("moving".to_string(), moving)]);
            step_pose_evaluation(world, dt, &machines, &skeletons, &clips, &vars);
        }
    }

    /// **The notify seam is written, readable, and does not go stale** (P29.1).
    ///
    /// Three claims, and the third is the one a resource gets wrong: the events
    /// appear on the step the transition fires, they are the exited state's
    /// `on_exit` followed by the entered state's `on_enter` in that order, and a
    /// **quiet** step reports nothing rather than repeating the last step's
    /// answer. A store that is merged rather than replaced passes the first two
    /// and fails the third, which is why the third is here.
    #[test]
    fn the_notify_seam_reports_this_step_and_only_this_step() {
        let guid = Uuid::from_u128(30);
        let mut idle = SmState::clip("idle", IDLE);
        idle.on_enter = vec!["idle_begin".into()];
        idle.on_exit = vec!["idle_end".into()];
        let mut wave = SmState::clip("wave", WAVE);
        wave.on_enter = vec!["wave_begin".into()];
        let noisy = StateMachine {
            states: vec![idle, wave],
            transitions: vec![
                SmTransition::on(0, 1, 0.0, "moving", inf_anim::CmpOp::Gt, 0.5),
                // The way back, so the closing "control: it fired" step has
                // something to fire — a control that cannot fire is a control
                // that proves nothing.
                SmTransition::on(1, 0, 0.0, "moving", inf_anim::CmpOp::Lt, 0.5),
            ],
            entry: 0,
            ..Default::default()
        };
        let skeleton = skeleton_asset();
        let clip = wave_clip();
        let mut world = world_with_character(guid);
        let step = |world: &mut EcsWorld, dt: f64, moving: f64| {
            let machines = |g: Uuid| (g == SM).then_some(&noisy);
            let skeletons = |g: Uuid| (g == SKEL).then_some(&skeleton);
            let clips = |c: ClipRef| (c == WAVE).then_some(&clip);
            let vars = |_: Uuid| BTreeMap::from([("moving".to_string(), moving)]);
            step_pose_evaluation(world, dt, &machines, &skeletons, &clips, &vars);
        };

        // Entering the entry state is itself an event -- which v1 told nobody.
        step(&mut world, 0.1, 0.0);
        assert_eq!(anim_events(&world, guid), ["idle_begin".to_string()]);

        // The transition: exit before the entry it caused.
        step(&mut world, 0.1, 1.0);
        assert_eq!(
            anim_events(&world, guid),
            ["idle_end".to_string(), "wave_begin".to_string()]
        );

        // A quiet step says NOTHING -- the store is replaced, not merged.
        step(&mut world, 0.1, 1.0);
        assert!(
            anim_events(&world, guid).is_empty(),
            "a quiet step repeated the previous step's notifies: {:?}",
            anim_events(&world, guid)
        );
        // …and an entity nobody has heard of reads empty rather than panicking.
        assert!(anim_events(&world, Uuid::from_u128(999)).is_empty());

        // The Simulate stop door forgets them, like the poses and the IK goals.
        step(&mut world, 0.1, 0.0);
        assert!(!anim_events(&world, guid).is_empty(), "control: it fired");
        clear_poses(&mut world);
        assert!(anim_events(&world, guid).is_empty());
    }

    /// **The host really passes a period resolver** (P29.1).
    ///
    /// `exit_time` was inert for four phases not because the model lacked the
    /// field but because this function built its context with `SmContext::new`.
    /// The model's own tests prove a resolved context gates correctly; only an
    /// arm HERE can prove the host supplies one — which is the difference
    /// between the fix and a fix that is written down. It is aimed at the wiring
    /// it names: the machine below is unconditional and gated at 80% of a **1 s**
    /// clip, so it fires late or it fires immediately, and "immediately" is
    /// exactly what the v1 fallback looks like.
    #[test]
    fn the_host_resolves_clip_lengths_so_exit_time_is_live() {
        let guid = Uuid::from_u128(29);
        // wave is a 1 s clip; the entry state plays it, and the only transition
        // out of it waits for 80% of a loop.
        let gated = StateMachine {
            states: vec![SmState::clip("wave", WAVE), SmState::clip("idle", IDLE)],
            transitions: vec![inf_anim::SmTransition::new(0, 1, 0.0).with_exit_time(0.8)],
            entry: 0,
            ..Default::default()
        };
        let skeleton = skeleton_asset();
        let clip = wave_clip();
        let mut world = world_with_character(guid);
        let step = |world: &mut EcsWorld, dt: f64| {
            let machines = |g: Uuid| (g == SM).then_some(&gated);
            let skeletons = |g: Uuid| (g == SKEL).then_some(&skeleton);
            let clips = |c: ClipRef| (c == WAVE).then_some(&clip);
            let vars = |_: Uuid| BTreeMap::new();
            step_pose_evaluation(world, dt, &machines, &skeletons, &clips, &vars);
        };
        let state_of = |world: &mut EcsWorld| -> usize {
            let w = world.world_mut();
            let mut q = w.query::<&AnimStateMachine>();
            q.iter(w).next().expect("the character").runtime.current
        };

        step(&mut world, 0.5);
        assert_eq!(
            state_of(&mut world),
            0,
            "0.5 s into a 1 s clip the 0.8 gate is not met — the host is still \
             building its context without a clip-length resolver, so every \
             exit_time reads as satisfied"
        );
        step(&mut world, 0.4);
        assert_eq!(state_of(&mut world), 1, "crossing 0.8 s must fire it");
    }

    /// **The headline gate.** A character in a non-entry machine state is posed
    /// DIFFERENTLY from one in the entry state — which is the whole defect: before
    /// this module the machine advanced and the drawn pose never moved.
    #[test]
    fn a_non_entry_state_poses_differently_from_the_entry_state() {
        let guid = Uuid::from_u128(1);
        let f = Fixture::new();

        // Standing still: entry state `idle`, whose clip resolves to nothing, so
        // the pose is rest.
        let mut still = world_with_character(guid);
        f.step(&mut still, 0.25, 0.0);
        let idle = evaluated_pose(&still, guid)
            .expect("a pose was published")
            .clone();
        assert_eq!(idle.skeleton, SKEL);
        assert_eq!(idle.pose.len(), 2, "one local per joint");
        assert_eq!(idle.pose, Pose::rest(&f.skeleton.skeleton));

        // Moving: the machine transitions to `wave` on the first step and the
        // pose follows it into the clip.
        let mut moving = world_with_character(guid);
        f.step(&mut moving, 0.25, 1.0);
        let waving = evaluated_pose(&moving, guid).expect("a pose was published");
        assert_eq!(
            moving
                .world()
                .get::<AnimStateMachine>(moving.entity_of(guid).unwrap())
                .unwrap()
                .runtime
                .current,
            1,
            "the machine transitioned"
        );
        assert_ne!(
            waving.pose, idle.pose,
            "the drawn pose must follow the machine's state, not the entry state"
        );
        // …and specifically the tip joint, the one the wave clip animates.
        assert_ne!(waving.pose.locals[1].rotation, idle.pose.locals[1].rotation);
    }

    /// The pose keeps advancing *within* a state — a machine that entered `wave`
    /// and stayed there must not freeze on the pose it entered with.
    #[test]
    fn the_pose_advances_inside_a_state() {
        let guid = Uuid::from_u128(2);
        let f = Fixture::new();
        let mut world = world_with_character(guid);
        f.step(&mut world, 0.1, 1.0);
        let first = evaluated_pose(&world, guid).unwrap().clone();
        f.step(&mut world, 0.1, 1.0);
        let second = evaluated_pose(&world, guid).unwrap();
        assert_ne!(first.pose, second.pose, "the play-head must advance");
    }

    /// Deterministic: two worlds stepped identically publish identical bytes.
    #[test]
    fn evaluation_is_deterministic() {
        let guid = Uuid::from_u128(3);
        let f = Fixture::new();
        let mut a = world_with_character(guid);
        let mut b = world_with_character(guid);
        for _ in 0..7 {
            f.step(&mut a, 1.0 / 60.0, 1.0);
            f.step(&mut b, 1.0 / 60.0, 1.0);
        }
        let (ba, bb) = (pose_state_bytes(&a), pose_state_bytes(&b));
        assert!(!ba.is_empty(), "a posed world must produce trace bytes");
        assert_eq!(ba, bb);
        // The bytes MOVE when the pose does (a constant would compare equal too).
        f.step(&mut a, 1.0 / 60.0, 1.0);
        assert_ne!(pose_state_bytes(&a), bb);
    }

    /// A world that poses nothing is byte-for-byte its pre-P24.1 self: no
    /// resource, no trace bytes.
    #[test]
    fn a_world_with_no_machine_never_grows_a_store() {
        let mut world = EcsWorld::new();
        world.world_mut().spawn((Guid(Uuid::from_u128(4)),));
        world.reindex_guids();
        Fixture::new().step(&mut world, 0.1, 1.0);
        assert_eq!(posed_count(&world), 0);
        assert!(pose_state_bytes(&world).is_empty());
    }

    /// Rule 3: an entity whose skeleton does not resolve still steps its machine
    /// (so the trace is unchanged) and publishes nothing (so the projectors keep
    /// their `AnimPlayer` / rest fallback).
    #[test]
    fn an_unresolvable_skeleton_steps_the_machine_and_poses_nothing() {
        let guid = Uuid::from_u128(5);
        let mut world = EcsWorld::new();
        let e = world
            .world_mut()
            .spawn((
                Guid(guid),
                AnimStateMachine {
                    sm: Some(SM),
                    ..Default::default()
                },
                SkeletalMesh {
                    mesh: Some(Uuid::from_u128(9)),
                    skeleton: Some(Uuid::from_u128(0xDEAD)),
                },
            ))
            .id();
        world.reindex_guids();
        Fixture::new().step(&mut world, 0.25, 1.0);
        assert_eq!(
            world
                .world()
                .get::<AnimStateMachine>(e)
                .unwrap()
                .runtime
                .current,
            1,
            "the machine must still advance"
        );
        assert!(evaluated_pose(&world, guid).is_none());
        assert!(pose_state_bytes(&world).is_empty());
    }

    /// Rule 4: the store is REPLACED each step, so an entity that stops posing
    /// leaves no stale pose behind — and the resource itself goes away.
    #[test]
    fn a_stopped_machine_leaves_no_stale_pose() {
        let guid = Uuid::from_u128(6);
        let f = Fixture::new();
        let mut world = world_with_character(guid);
        f.step(&mut world, 0.1, 1.0);
        assert_eq!(posed_count(&world), 1);

        // Unbind the machine; the next step must publish nothing at all.
        let e = world.entity_of(guid).unwrap();
        world.world_mut().get_mut::<AnimStateMachine>(e).unwrap().sm = None;
        f.step(&mut world, 0.1, 1.0);
        assert_eq!(
            posed_count(&world),
            0,
            "the store must be rebuilt, not merged"
        );
        assert!(pose_state_bytes(&world).is_empty());
    }

    /// The socket table rides the pose: a socket on the animated joint moves with
    /// it. This is what `update_attachments` consumes.
    #[test]
    fn the_socket_table_follows_the_animated_joint() {
        let guid = Uuid::from_u128(7);
        let f = Fixture::new();
        let mut rest = world_with_character(guid);
        f.step(&mut rest, 0.1, 0.0);
        let at_rest = evaluated_pose(&rest, guid)
            .unwrap()
            .socket("hand_r")
            .unwrap();

        let mut waving = world_with_character(guid);
        // Two steps so the clip has actually played into the bend.
        f.step(&mut waving, 0.3, 1.0);
        f.step(&mut waving, 0.3, 1.0);
        let posed = evaluated_pose(&waving, guid)
            .unwrap()
            .socket("hand_r")
            .unwrap();
        assert_ne!(at_rest.to_cols_array(), posed.to_cols_array());
        assert!(evaluated_pose(&waving, guid)
            .unwrap()
            .socket("no_such_socket")
            .is_none());
    }

    /// `clear_poses` returns the world to one that has never posed anything —
    /// the Simulate start/stop door.
    #[test]
    fn clear_poses_forgets_everything() {
        let guid = Uuid::from_u128(8);
        let f = Fixture::new();
        let mut world = world_with_character(guid);
        f.step(&mut world, 0.1, 1.0);
        assert_eq!(posed_count(&world), 1);
        clear_poses(&mut world);
        assert_eq!(posed_count(&world), 0);
        clear_poses(&mut world); // idempotent
    }

    /// **There is no conversion left to get wrong** (P29.1).
    ///
    /// This used to assert `from_anim_runtime(to_anim_runtime(s)) == s` over a
    /// hand-copied POD mirror — a round-trip that a field missing from *both*
    /// halves passes perfectly, which is the failure mode a mirror has. The
    /// mirror is a type alias now, so the property is checked by assignment: if
    /// `SmRuntimeState` ever stops being `inf_anim::SmRuntime`, this stops
    /// compiling, and that is a stronger statement than any round-trip.
    #[test]
    fn the_component_runtime_is_the_anim_runtime() {
        let s = SmRuntimeState {
            current: 3,
            prev: Some(1),
            prev_time: 0.5,
            fade_t: 0.25,
            fade_dur: 0.5,
            state_time: 1.25,
            started: true,
            ..Default::default()
        };
        let anim: inf_anim::SmRuntime = s;
        assert_eq!(anim, s);
        // …and it reaches an entity unchanged, which is what the write-back does.
        let asm = AnimStateMachine {
            runtime: s,
            ..Default::default()
        };
        assert_eq!(asm.runtime.current, 3);
        // The v2 fields exist and default to "no interruption carried, nothing
        // armed" — the state a fresh play session starts in.
        assert_eq!(SmRuntimeState::default().carry, None);
        assert_eq!(SmRuntimeState::default().triggers, 0);
    }

    // ── P24.3: the AUTHORED goals ─────────────────────────────────────────

    use crate::components::{GlobalTransform, IkGoalRecord, IkTarget, Transform};
    use crate::math::Vec3d;

    /// The same character as [`world_with_character`], plus an authored
    /// [`IkTarget`] and a real placement, so the world→model conversion has
    /// something to convert.
    fn world_with_authored_ik(guid: Uuid, at: Vec3d, goals: Vec<IkGoalRecord>) -> EcsWorld {
        let mut world = EcsWorld::new();
        world.world_mut().spawn((
            Guid(guid),
            Transform {
                translation: at,
                ..Default::default()
            },
            GlobalTransform(glam::DAffine3::from_translation(glam::DVec3::new(
                at.x, at.y, at.z,
            ))),
            AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(9)),
                skeleton: Some(SKEL),
            },
            IkTarget { goals },
        ));
        world.reindex_guids();
        world
    }

    fn authored(chain: Vec<u16>, target: Vec3d) -> IkGoalRecord {
        IkGoalRecord {
            chain,
            target,
            ..Default::default()
        }
    }

    /// **The headline gate for the authored half**: a saved `IkTarget` bends the
    /// character, through the same one door both hosts call, with nothing set at
    /// runtime.
    #[test]
    fn an_authored_ik_target_moves_the_pose() {
        let guid = Uuid::from_u128(0x24_3001);
        let f = Fixture::new();
        let mut plain = world_with_character(guid);
        f.step(&mut plain, 0.25, 0.0);
        let rest = evaluated_pose(&plain, guid).unwrap().clone();

        let mut rigged = world_with_authored_ik(
            guid,
            Vec3d::ZERO,
            vec![authored(vec![0, 1], Vec3d::new(0.7, 0.7, 0.0))],
        );
        f.step(&mut rigged, 0.25, 0.0);
        let posed = evaluated_pose(&rigged, guid).expect("a pose was published");
        assert_ne!(
            posed.pose, rest.pose,
            "an authored IK target must reach the solver"
        );
        // …and it SAYS what it did, through the read door a gate can use.
        let outcomes = ik_outcomes(&rigged, guid).expect("a verdict was published");
        assert_eq!(outcomes.len(), 1);
        assert!(
            matches!(outcomes[0], IkOutcome::Solved(_)),
            "{outcomes:?} — an authored goal must solve, not refuse"
        );
        // Nothing set a runtime goal, so the resource carries only the verdict.
        assert!(ik_goals(&rigged, guid).is_none());
    }

    /// **Anti-vacuity, and the property the PIE arm rests on**: MOVING the
    /// authored target moves the trace. A component that was read once and cached
    /// would pass the test above and fail this one.
    #[test]
    fn moving_the_authored_target_moves_the_trace() {
        let f = Fixture::new();
        let guid = Uuid::from_u128(0x24_3002);
        let bytes_for = |t: Vec3d| {
            let mut w = world_with_authored_ik(guid, Vec3d::ZERO, vec![authored(vec![0, 1], t)]);
            f.step(&mut w, 0.25, 0.0);
            pose_state_bytes(&w)
        };
        let a = bytes_for(Vec3d::new(0.7, 0.7, 0.0));
        let b = bytes_for(Vec3d::new(-0.7, 0.7, 0.0));
        assert!(!a.is_empty());
        assert_ne!(a, b, "the trace must be a function of where the target is");
        // Same target twice ⇒ same bytes (the determinism half).
        assert_eq!(a, bytes_for(Vec3d::new(0.7, 0.7, 0.0)));
    }

    /// The target is **world** space: the same authored offset on a character
    /// standing somewhere else produces the same *model-space* pose, which is the
    /// whole reason `authored_ik_goals` inverts the global transform.
    #[test]
    fn the_authored_target_is_world_space() {
        let f = Fixture::new();
        // Character at the origin reaching for world (0.7, 0.7, 0).
        let mut here = world_with_authored_ik(
            Uuid::from_u128(0x24_3003),
            Vec3d::ZERO,
            vec![authored(vec![0, 1], Vec3d::new(0.7, 0.7, 0.0))],
        );
        // The same character 100 m away, reaching for the point 100 m away — the
        // same thing relative to its own body.
        let mut there = world_with_authored_ik(
            Uuid::from_u128(0x24_3003),
            Vec3d::new(100.0, 0.0, 0.0),
            vec![authored(vec![0, 1], Vec3d::new(100.7, 0.7, 0.0))],
        );
        f.step(&mut here, 0.25, 0.0);
        f.step(&mut there, 0.25, 0.0);
        let a = evaluated_pose(&here, Uuid::from_u128(0x24_3003)).unwrap();
        let b = evaluated_pose(&there, Uuid::from_u128(0x24_3003)).unwrap();
        assert_eq!(a.pose, b.pose, "the goal must be read in the WORLD frame");

        // …and the control: the same authored number on the displaced character
        // is a different pose, so the test above is not comparing two rest poses.
        let mut naive = world_with_authored_ik(
            Uuid::from_u128(0x24_3003),
            Vec3d::new(100.0, 0.0, 0.0),
            vec![authored(vec![0, 1], Vec3d::new(0.7, 0.7, 0.0))],
        );
        f.step(&mut naive, 0.25, 0.0);
        assert_ne!(
            evaluated_pose(&naive, Uuid::from_u128(0x24_3003))
                .unwrap()
                .pose,
            a.pose
        );
    }

    /// A goal that follows another **entity** tracks it — the case a constant
    /// cannot express, and the reason `target_entity` is on the wire.
    #[test]
    fn a_goal_can_follow_another_entity() {
        let f = Fixture::new();
        let guid = Uuid::from_u128(0x24_3004);
        let hold = Uuid::from_u128(0x24_3005);
        let build = |hold_at: glam::DVec3| {
            let mut w = world_with_authored_ik(
                guid,
                Vec3d::ZERO,
                vec![IkGoalRecord {
                    chain: vec![0, 1],
                    target_entity: crate::refs::EntityRef::new(hold),
                    ..Default::default()
                }],
            );
            w.world_mut().spawn((
                Guid(hold),
                GlobalTransform(glam::DAffine3::from_translation(hold_at)),
            ));
            w.reindex_guids();
            w
        };
        let mut a = build(glam::DVec3::new(0.7, 0.7, 0.0));
        let mut b = build(glam::DVec3::new(-0.7, 0.7, 0.0));
        f.step(&mut a, 0.25, 0.0);
        f.step(&mut b, 0.25, 0.0);
        assert_ne!(
            evaluated_pose(&a, guid).unwrap().pose,
            evaluated_pose(&b, guid).unwrap().pose,
            "the chain must follow the entity it names"
        );

        // A GUID naming nothing is treated as unbound rather than as a refusal:
        // the offset is then absolute, so the goal still solves.
        let mut orphan = world_with_authored_ik(
            guid,
            Vec3d::ZERO,
            vec![IkGoalRecord {
                chain: vec![0, 1],
                target_entity: crate::refs::EntityRef::new(Uuid::from_u128(0xDEAD)),
                target: Vec3d::new(0.7, 0.7, 0.0),
                ..Default::default()
            }],
        );
        f.step(&mut orphan, 0.25, 0.0);
        assert!(matches!(
            ik_outcomes(&orphan, guid).unwrap()[0],
            IkOutcome::Solved(_)
        ));
    }

    /// `enabled: false` is authored, saved and **not solved** — and a disabled
    /// goal costs nothing at all: no verdict, no resource.
    #[test]
    fn a_disabled_goal_is_not_solved() {
        let f = Fixture::new();
        let guid = Uuid::from_u128(0x24_3006);
        let mut w = world_with_authored_ik(
            guid,
            Vec3d::ZERO,
            vec![IkGoalRecord {
                enabled: false,
                ..authored(vec![0, 1], Vec3d::new(0.7, 0.7, 0.0))
            }],
        );
        f.step(&mut w, 0.25, 0.0);
        assert!(
            ik_outcomes(&w, guid).is_none(),
            "a disabled goal says nothing"
        );
        let mut plain = world_with_character(guid);
        f.step(&mut plain, 0.25, 0.0);
        assert_eq!(
            pose_state_bytes(&w),
            pose_state_bytes(&plain),
            "a disabled goal must cost exactly nothing in the trace"
        );
    }

    /// **The weight blends**, and full weight is byte-identical to no blend at
    /// all — the claim `IkGoal::weight`'s doc makes.
    #[test]
    fn the_authored_weight_blends_and_full_weight_is_free() {
        let f = Fixture::new();
        let guid = Uuid::from_u128(0x24_3007);
        let at = |weight: f32| {
            let mut w = world_with_authored_ik(
                guid,
                Vec3d::ZERO,
                vec![IkGoalRecord {
                    weight,
                    ..authored(vec![0, 1], Vec3d::new(0.7, 0.7, 0.0))
                }],
            );
            f.step(&mut w, 0.25, 0.0);
            (pose_state_bytes(&w), w)
        };
        let (full, _) = at(1.0);
        let (half, _) = at(0.5);
        let (none, zero_world) = at(0.0);

        // A `1.0` weight takes the no-snapshot path; the bytes are the solve's.
        assert_ne!(full, none, "a full-weight solve must move the pose");
        assert_ne!(half, full, "a half weight must not be the full solve");
        assert_ne!(half, none, "…nor the rest pose");

        // Zero weight leaves the pose alone and STILL reports — a goal turned
        // down to nothing is distinguishable from an absent one.
        let mut plain = world_with_character(guid);
        f.step(&mut plain, 0.25, 0.0);
        assert_eq!(none, pose_state_bytes(&plain));
        assert!(matches!(
            ik_outcomes(&zero_world, guid).unwrap()[0],
            IkOutcome::Solved(_)
        ));
    }

    /// The authored goals and the runtime ones are **both** solved, authored
    /// first — so `set_ik_goals` (and the `ik.*` node kit) is not silently
    /// disabled on a character an author has rigged.
    #[test]
    fn authored_and_runtime_goals_are_concatenated() {
        let f = Fixture::new();
        let guid = Uuid::from_u128(0x24_3008);
        let mut w = world_with_authored_ik(
            guid,
            Vec3d::ZERO,
            vec![authored(vec![0, 1], Vec3d::new(0.7, 0.7, 0.0))],
        );
        set_ik_goals(
            &mut w,
            guid,
            vec![IkGoal::full(vec![0, 1], [-0.7, 0.7, 0.0], None)],
        );
        f.step(&mut w, 0.25, 0.0);
        let outcomes = ik_outcomes(&w, guid).expect("verdicts");
        assert_eq!(outcomes.len(), 2, "both goals ran: {outcomes:?}");
        // The runtime goal ran LAST, so the pose ends up at its target.
        let posed = evaluated_pose(&w, guid).unwrap();
        let tip = inf_anim::global_transforms(&f.skeleton.skeleton, &posed.pose)[1]
            .transform_point3(glam::Vec3::ZERO);
        assert!(
            tip.x < 0.0,
            "the LAST goal must win the tip position, got {tip:?}"
        );
    }

    /// **F7: the conversion really happens EVERY fixed step.**
    ///
    /// Every other authored-goal test steps once, and the `--pie` arm holds its
    /// target constant — so "converted per fixed step, not cached at session
    /// start" had no falsifier at all. A read-once-and-cache implementation
    /// passed all of them.
    ///
    /// Here the target ENTITY moves between steps, and the pose must follow it
    /// each time.
    ///
    /// **Two of the three claims falsify a cache**, and the third does not — the
    /// P24.3 re-audit's correction, recorded where the claim is made. The pose
    /// differing after each move, and step 3 differing from step 1, are what a
    /// read-once-and-cache implementation cannot produce. "Returning the anchor
    /// returns the pose" is satisfiable by a cache keyed on the anchor's
    /// position; it is a *determinism* claim, kept for its own sake, not counted
    /// as cache detection.
    #[test]
    fn the_authored_goal_is_reconverted_on_every_fixed_step() {
        let f = Fixture::new();
        let guid = Uuid::from_u128(0x24_300B);
        let hold = Uuid::from_u128(0x24_300C);
        let mut w = world_with_authored_ik(
            guid,
            Vec3d::ZERO,
            vec![IkGoalRecord {
                chain: vec![0, 1],
                target_entity: crate::refs::EntityRef::new(hold),
                ..Default::default()
            }],
        );
        let anchor = w
            .world_mut()
            .spawn((
                Guid(hold),
                GlobalTransform(glam::DAffine3::from_translation(glam::DVec3::new(
                    0.7, 0.7, 0.0,
                ))),
            ))
            .id();
        w.reindex_guids();

        let move_to = |w: &mut EcsWorld, at: glam::DVec3| {
            w.world_mut().get_mut::<GlobalTransform>(anchor).unwrap().0 =
                glam::DAffine3::from_translation(at);
        };

        f.step(&mut w, 1.0 / 60.0, 0.0);
        let a = pose_state_bytes(&w);
        move_to(&mut w, glam::DVec3::new(-0.7, 0.7, 0.0));
        f.step(&mut w, 1.0 / 60.0, 0.0);
        let b = pose_state_bytes(&w);
        move_to(&mut w, glam::DVec3::new(0.0, 0.99, 0.1));
        f.step(&mut w, 1.0 / 60.0, 0.0);
        let c = pose_state_bytes(&w);

        assert!(!a.is_empty(), "nothing was posed at all");
        assert_ne!(a, b, "the goal did not follow the anchor on step 2 — a read-once-and-cache implementation passes every other authored-goal test and fails here");
        assert_ne!(b, c, "the goal did not follow the anchor on step 3");
        assert_ne!(a, c, "the pose is oscillating rather than tracking");

        // …and it is a function of WHERE THE ANCHOR IS NOW, not of how it got
        // there: put it back and the pose comes back.
        move_to(&mut w, glam::DVec3::new(0.7, 0.7, 0.0));
        f.step(&mut w, 1.0 / 60.0, 0.0);
        assert_eq!(
            pose_state_bytes(&w),
            a,
            "returning the anchor did not return the pose — the solve depends on history rather than on the current conversion"
        );
    }

    /// A singular placement (zero scale) has no model frame, so its goals are
    /// dropped rather than run through a matrix of infinities.
    #[test]
    fn a_singular_placement_drops_its_goals() {
        let f = Fixture::new();
        let guid = Uuid::from_u128(0x24_3009);
        let mut w = world_with_authored_ik(
            guid,
            Vec3d::ZERO,
            vec![authored(vec![0, 1], Vec3d::new(0.7, 0.7, 0.0))],
        );
        let e = w.entity_of(guid).unwrap();
        w.world_mut().get_mut::<GlobalTransform>(e).unwrap().0 =
            glam::DAffine3::from_scale(glam::DVec3::ZERO);
        f.step(&mut w, 0.25, 0.0);
        assert!(ik_outcomes(&w, guid).is_none());
        assert!(evaluated_pose(&w, guid)
            .unwrap()
            .pose
            .locals
            .iter()
            .all(|l| l.rotation.iter().all(|c| c.is_finite())));
    }

    /// A world with an `IkTarget` but **no machine** never grows a store — the
    /// pre-P24.1 byte-identity claim, re-checked now that a second component can
    /// reach the step.
    #[test]
    fn an_ik_target_alone_never_grows_a_store() {
        let mut world = EcsWorld::new();
        world.world_mut().spawn((
            Guid(Uuid::from_u128(0x24_300A)),
            GlobalTransform::default(),
            IkTarget {
                goals: vec![authored(vec![0, 1], Vec3d::new(1.0, 0.0, 0.0))],
            },
        ));
        world.reindex_guids();
        Fixture::new().step(&mut world, 0.1, 1.0);
        assert_eq!(posed_count(&world), 0);
        assert!(pose_state_bytes(&world).is_empty());
        assert!(world.world().get_resource::<IkTargetsRes>().is_none());
    }

    /// **Inertialization is the DEFAULT for state transitions, in the one fixed
    /// step both hosts call** (P29.2, §13's catalogue amendment).
    ///
    /// The observable claim, and the reason it is observable *here* rather than
    /// only in `inf-anim`: after a transition with a real duration fires, the
    /// machine's own cross-fade is **collapsed** — `prev` is `None` and `fade_dur`
    /// is zero — so `eval_pose` samples one state and the outgoing half is the
    /// decaying deviation the blender holds. Under the P29.1 cross-fade the same
    /// step leaves `prev = Some(0)` and `fade_dur = 0.25`.
    ///
    /// Mutation check: calling `rt.advance` + `inf_anim::eval_pose` directly
    /// (the pre-P29.2 body) leaves the fade running and fails the first two
    /// assertions; dropping the resource write fails the third.
    #[test]
    fn a_transition_inertializes_rather_than_cross_fading() {
        let guid = Uuid::from_u128(0x29_2001);
        let faded = StateMachine {
            transitions: vec![SmTransition::on(
                0,
                1,
                0.25,
                "moving",
                inf_anim::CmpOp::Gt,
                0.5,
            )],
            ..machine()
        };
        let f = Fixture {
            machine: faded,
            ..Fixture::new()
        };
        let mut w = world_with_character(guid);
        // Step 1 settles into the entry state; step 2 fires the transition.
        f.step(&mut w, 1.0 / 60.0, 0.0);
        f.step(&mut w, 1.0 / 60.0, 1.0);
        let e = w.entity_of(guid).unwrap();
        let rt = w.world().get::<AnimStateMachine>(e).unwrap().runtime;
        assert_eq!(rt.current, 1, "the transition did not fire");
        assert!(
            rt.prev.is_none(),
            "the runtime is still cross-fading out of {:?}",
            rt.prev
        );
        assert_eq!(rt.fade_dur, 0.0, "the fade duration survived the collapse");
        // The decay lives in the resource, and it is live.
        let blend = w
            .world()
            .get_resource::<PoseBlendRes>()
            .expect("the blender resource was not published");
        let b = blend.0.get(&guid).expect("this entity has no blender");
        assert!(b.is_blending(), "no deviation was captured");
        assert!(
            b.decay() > 0.0 && b.decay() <= 1.0,
            "decay {} is outside [0,1]",
            b.decay()
        );
        // …and it finishes: 0.25 s at 60 Hz is 15 steps, so 20 is comfortably past.
        for _ in 0..20 {
            f.step(&mut w, 1.0 / 60.0, 1.0);
        }
        let b = &w.world().get_resource::<PoseBlendRes>().unwrap().0[&guid];
        assert!(!b.is_blending(), "the decay never finished");
    }

    /// The blender store follows [`PoseStoreRes`]'s rules: **absent costs
    /// nothing**, and it is pruned rather than merged.
    #[test]
    fn the_blender_store_is_pruned_and_never_grows_on_a_machine_free_world() {
        // A world with no machine never grows one.
        let mut bare = EcsWorld::new();
        bare.world_mut().spawn((Guid(Uuid::from_u128(0x29_2002)),));
        bare.reindex_guids();
        Fixture::new().step(&mut bare, 0.1, 1.0);
        assert!(bare.world().get_resource::<PoseBlendRes>().is_none());

        // A posed entity grows one…
        let guid = Uuid::from_u128(0x29_2003);
        let f = Fixture::new();
        let mut w = world_with_character(guid);
        f.step(&mut w, 1.0 / 60.0, 0.0);
        assert_eq!(w.world().get_resource::<PoseBlendRes>().unwrap().0.len(), 1);
        // …and unbinding its machine takes it away again, rather than leaving a
        // decay to resume from when it comes back.
        let e = w.entity_of(guid).unwrap();
        w.world_mut().get_mut::<AnimStateMachine>(e).unwrap().sm = None;
        f.step(&mut w, 1.0 / 60.0, 0.0);
        assert!(w.world().get_resource::<PoseBlendRes>().is_none());

        // `clear_poses` forgets it through the one door.
        w.world_mut().get_mut::<AnimStateMachine>(e).unwrap().sm = Some(SM);
        f.step(&mut w, 1.0 / 60.0, 0.0);
        assert!(w.world().get_resource::<PoseBlendRes>().is_some());
        clear_poses(&mut w);
        assert!(w.world().get_resource::<PoseBlendRes>().is_none());
    }
    /// **The mode is selectable, and the selection is real** (P29.2).
    ///
    /// `.inf_sm` cannot carry a per-transition choice — that format bumped in
    /// P29.1 and does not bump again this phase — so the selector is world-level.
    /// What this arm asserts is that it is not decorative: under `CrossFade` the
    /// runtime's own fade survives a fired transition, which is exactly what the
    /// default collapses.
    #[test]
    fn the_cross_fade_mode_is_selectable_and_puts_the_p29_1_path_back() {
        let guid = Uuid::from_u128(0x29_2004);
        let faded = StateMachine {
            transitions: vec![SmTransition::on(
                0,
                1,
                0.25,
                "moving",
                inf_anim::CmpOp::Gt,
                0.5,
            )],
            ..machine()
        };
        let f = Fixture {
            machine: faded,
            ..Fixture::new()
        };
        let mut w = world_with_character(guid);
        // The default, stated rather than assumed.
        assert_eq!(blend_mode(&w), inf_anim::SmBlendMode::Inertialize);
        set_blend_mode(&mut w, inf_anim::SmBlendMode::CrossFade);
        assert_eq!(blend_mode(&w), inf_anim::SmBlendMode::CrossFade);

        f.step(&mut w, 1.0 / 60.0, 0.0);
        f.step(&mut w, 1.0 / 60.0, 1.0);
        let e = w.entity_of(guid).unwrap();
        let rt = w.world().get::<AnimStateMachine>(e).unwrap().runtime;
        assert_eq!(rt.current, 1, "the transition did not fire");
        assert_eq!(
            rt.prev,
            Some(0),
            "the cross-fade was collapsed anyway — the mode did nothing"
        );
        assert!((rt.fade_dur - 0.25).abs() < 1e-12);
        // …and no deviation was captured, because there is nothing to inertialize.
        let b = &w.world().get_resource::<PoseBlendRes>().unwrap().0[&guid];
        assert!(!b.is_blending());

        // The setting SURVIVES `clear_poses`, which drops the per-step state.
        clear_poses(&mut w);
        assert_eq!(blend_mode(&w), inf_anim::SmBlendMode::CrossFade);
        // …and switching back reaches the live blenders immediately.
        f.step(&mut w, 1.0 / 60.0, 0.0);
        set_blend_mode(&mut w, inf_anim::SmBlendMode::Inertialize);
        assert!(w
            .world()
            .get_resource::<PoseBlendRes>()
            .unwrap()
            .0
            .values()
            .all(|b| b.mode == inf_anim::SmBlendMode::Inertialize));
    }

    // -- P29.4: the animation bridge, through the one door both hosts call ----

    /// A **walking** fixture: a root joint that really travels, a baked v2
    /// root-motion track, one event marker and one sync marker.
    fn walk_fixture() -> (StateMachine, SkeletonAsset, AnimClip) {
        let skeleton = skeleton_asset();
        // The root moves +2 m along +Z over one second.
        let clip = inf_anim::root_motion::straight_line_clip("walk", glam::Vec3::Z, 2.0, 1.0);
        let (rm, dist) = inf_anim::bake_root_motion(&clip, &skeleton.skeleton, 60.0).unwrap();
        let clip = clip
            .with_root_motion(rm)
            .with_distance(dist)
            .with_markers(vec![
                inf_anim::AnimMarker::event(0.25, "footstep_l"),
                inf_anim::AnimMarker::event(0.75, "footstep_r"),
                // A SYNC marker at the same instant, which must NOT ring.
                inf_anim::AnimMarker::sync(0.25, "plant_l", "foot"),
            ]);
        let machine = StateMachine {
            states: vec![SmState::clip("idle", IDLE), SmState::clip("walk", WAVE)],
            transitions: vec![SmTransition::on(0, 1, 0.0, "go", inf_anim::CmpOp::Gt, 0.5)],
            entry: 0,
            params: vec![inf_anim::SmParam::new("go", inf_anim::SmParamKind::Trigger)],
            ..Default::default()
        };
        (machine, skeleton, clip)
    }

    struct WalkFixture {
        machine: StateMachine,
        skeleton: SkeletonAsset,
        clip: AnimClip,
    }

    impl WalkFixture {
        fn new() -> Self {
            let (machine, skeleton, clip) = walk_fixture();
            Self {
                machine,
                skeleton,
                clip,
            }
        }
        fn step(&self, world: &mut EcsWorld, dt: f64) {
            let machines = |g: Uuid| (g == SM).then_some(&self.machine);
            let skeletons = |g: Uuid| (g == SKEL).then_some(&self.skeleton);
            let clips = |c: ClipRef| (c == WAVE).then_some(&self.clip);
            let vars = |_: Uuid| BTreeMap::new();
            step_pose_evaluation(world, dt, &machines, &skeletons, &clips, &vars);
        }
    }

    /// **The overlay wins over the actor's variable**, which is the whole reason
    /// `anim.set_param` exists -- and the control is the same fixture with the
    /// overlay absent, which does not transition at all.
    #[test]
    fn a_bridge_parameter_shadows_the_actors_variable() {
        let guid = Uuid::from_u128(41);
        let f = Fixture::new();

        // Control: the actor's variable says "not moving", and nothing happens.
        let mut plain = world_with_character(guid);
        f.step(&mut plain, 1.0 / 60.0, 0.0);
        f.step(&mut plain, 1.0 / 60.0, 0.0);
        let e = plain.entity_of(guid).unwrap();
        assert_eq!(
            plain
                .world()
                .get::<AnimStateMachine>(e)
                .unwrap()
                .runtime
                .current,
            0,
            "the control must NOT transition, or the arm below proves nothing"
        );

        // The same variable, overlaid by the bridge.
        let mut w = world_with_character(guid);
        assert!(crate::anim_bridge::set_anim_param(
            &mut w, guid, "moving", 1.0
        ));
        f.step(&mut w, 1.0 / 60.0, 0.0);
        f.step(&mut w, 1.0 / 60.0, 0.0);
        let e = w.entity_of(guid).unwrap();
        assert_eq!(
            w.world()
                .get::<AnimStateMachine>(e)
                .unwrap()
                .runtime
                .current,
            1,
            "the overlay did not reach the machine"
        );
        // ...and it PERSISTS: a parameter is a setting, not an event.
        assert_eq!(
            crate::anim_bridge::anim_param(&w, guid, "moving"),
            Some(1.0)
        );
    }

    /// An armed trigger fires **once**, is taken by the step that delivers it,
    /// and a second step off the same arm does nothing.
    #[test]
    fn an_armed_trigger_fires_once_and_is_taken_by_the_step() {
        let guid = Uuid::from_u128(42);
        let f = WalkFixture::new();
        let mut w = world_with_character(guid);
        f.step(&mut w, 1.0 / 60.0);
        let e = w.entity_of(guid).unwrap();
        assert_eq!(
            w.world()
                .get::<AnimStateMachine>(e)
                .unwrap()
                .runtime
                .current,
            0
        );

        assert!(crate::anim_bridge::set_anim_trigger(&mut w, guid, "go"));
        f.step(&mut w, 1.0 / 60.0);
        assert_eq!(
            w.world()
                .get::<AnimStateMachine>(e)
                .unwrap()
                .runtime
                .current,
            1,
            "the armed trigger did not reach the machine"
        );
        // The pending arm was TAKEN, not left to re-fire.
        assert!(crate::anim_bridge::bridge(&w)
            .map(|b| b.triggers.is_empty())
            .unwrap_or(true));
    }

    /// The step publishes **what the machine is in** and **what its clip moved**,
    /// and both are replaced rather than merged.
    #[test]
    fn the_step_publishes_the_state_and_the_clips_root_motion() {
        let guid = Uuid::from_u128(43);
        let f = WalkFixture::new();
        let mut w = world_with_character(guid);
        f.step(&mut w, 1.0 / 60.0);
        assert!(crate::anim_bridge::anim_state_is(&w, guid, "idle"));
        // Idle plays a clip nothing resolves, so it moves nothing.
        assert_eq!(crate::anim_bridge::anim_root_motion(&w, guid), None);

        crate::anim_bridge::set_anim_trigger(&mut w, guid, "go");
        f.step(&mut w, 1.0 / 60.0);
        assert!(crate::anim_bridge::anim_state_is(&w, guid, "walk"));
        // The step the transition fires contributes no root motion (two clips,
        // one window) -- the honest answer, and it costs one step.
        assert_eq!(crate::anim_bridge::anim_root_motion(&w, guid), None);

        f.step(&mut w, 1.0 / 60.0);
        let rm = crate::anim_bridge::anim_root_motion(&w, guid).expect("root motion");
        // 2 m/s for one 60 Hz step.
        assert!((rm.translation.z - 2.0 / 60.0).abs() < 1e-4, "{rm:?}");
        assert!((rm.distance_m - 2.0 / 60.0).abs() < 1e-4, "{rm:?}");
        assert!(crate::anim_bridge::anim_state_time(&w, guid) > 0.0);

        // Unbind the machine: the published half must not go stale.
        let e = w.entity_of(guid).unwrap();
        w.world_mut().get_mut::<AnimStateMachine>(e).unwrap().sm = None;
        f.step(&mut w, 1.0 / 60.0);
        assert!(crate::anim_bridge::anim_state(&w, guid).is_none());
        assert_eq!(crate::anim_bridge::anim_root_motion(&w, guid), None);
    }

    /// **An event marker becomes a notify, exactly once, and a sync marker does
    /// not.** The consumer takes it, and the second taker gets nothing.
    #[test]
    fn an_event_marker_rings_once_and_a_sync_marker_never_does() {
        let guid = Uuid::from_u128(44);
        let f = WalkFixture::new();
        let mut w = world_with_character(guid);
        crate::anim_bridge::set_anim_trigger(&mut w, guid, "go");
        f.step(&mut w, 1.0 / 60.0);
        f.step(&mut w, 1.0 / 60.0);
        assert!(crate::anim_bridge::anim_state_is(&w, guid, "walk"));

        // Walk the play-head past 0.25 s in 60 Hz steps and count the rings.
        let mut left = 0;
        let mut right = 0;
        let mut sync = 0;
        for _ in 0..30 {
            f.step(&mut w, 1.0 / 60.0);
            for n in anim_events(&w, guid) {
                match n.as_str() {
                    "footstep_l" => left += 1,
                    "footstep_r" => right += 1,
                    "plant_l" => sync += 1,
                    _ => {}
                }
            }
        }
        assert_eq!(left, 1, "the left footstep rang {left} times");
        assert_eq!(right, 0, "0.75 s is past the end of a 30-step walk");
        assert_eq!(sync, 0, "a sync marker is an alignment, not a notify");

        // ...and a consumer takes it: two handlers racing for one footstep get one.
        let mut w2 = world_with_character(guid);
        crate::anim_bridge::set_anim_trigger(&mut w2, guid, "go");
        f.step(&mut w2, 1.0 / 60.0);
        f.step(&mut w2, 1.0 / 60.0);
        let mut taken = 0;
        for _ in 0..30 {
            f.step(&mut w2, 1.0 / 60.0);
            if crate::anim_bridge::consume_anim_notify(&mut w2, guid, "footstep_l") {
                taken += 1;
            }
            assert!(
                !crate::anim_bridge::consume_anim_notify(&mut w2, guid, "footstep_l"),
                "a notify was consumed twice in one step"
            );
        }
        assert_eq!(taken, 1);
    }

    /// **A blend space fires its footsteps** (P29.5 — P29.4's "event-marker
    /// notifies fire from single-clip states only" remainder, closed).
    ///
    /// The mechanism is the **leader**: a blend space has no single play-head,
    /// but `inf_anim::sync` already picks a heaviest clip whose timeline every
    /// follower is warped onto, and a marker crossed on that timeline is a marker
    /// the blend crossed. Three halves:
    ///
    /// 1. a two-clip blend space fires the leader's markers over one cycle;
    /// 2. the leader's `Mask_FootstepSound` reaches the audio drain, so the
    ///    footstep is the loudness the leading clip authored and not 1.0 by
    ///    default;
    /// 3. the **control** — the same state with the weights swapped leads with
    ///    the other clip and fires ITS marker, which is what proves the answer
    ///    follows the weights rather than the entry order.
    #[test]
    fn a_blend_space_fires_the_leaders_markers_and_publishes_its_curves() {
        let guid = Uuid::from_u128(64);
        // Two one-second clips, each with one distinctly-named footstep and a
        // different authored gain.
        let clip = |name: &str, gain: f32| {
            let mut c = wave_clip();
            c.markers = vec![inf_anim::AnimMarker::event(0.5, format!("footstep_{name}"))];
            c.curves = vec![inf_anim::CurveChannel::constant(
                inf_anim::channels::als::MASK_FOOTSTEP_SOUND,
                gain,
            )];
            c
        };
        let left = clip("left", 0.25);
        let right = clip("right", 0.75);
        const L: ClipRef = [11u8; 16];
        const R: ClipRef = [22u8; 16];
        let space = inf_anim::BlendSpace1D {
            param: "moving".into(),
            entries: vec![
                inf_anim::BlendEntry1D { pos: 0.0, clip: L },
                inf_anim::BlendEntry1D { pos: 1.0, clip: R },
            ],
        };
        let machine = StateMachine {
            states: vec![SmState {
                motion: inf_anim::Motion::Blend1D(space),
                ..SmState::clip("locomotion", L)
            }],
            transitions: Vec::new(),
            entry: 0,
            ..Default::default()
        };
        let skeleton = skeleton_asset();

        let run = |moving: f64| -> (Vec<String>, f64) {
            let mut world = world_with_character(guid);
            let mut fired: Vec<String> = Vec::new();
            let mut gain = -1.0;
            for _ in 0..70 {
                let machines = |g: Uuid| (g == SM).then_some(&machine);
                let skeletons = |g: Uuid| (g == SKEL).then_some(&skeleton);
                let clips = |c: ClipRef| match c {
                    L => Some(&left),
                    R => Some(&right),
                    _ => None,
                };
                let vars = |_: Uuid| BTreeMap::from([("moving".to_string(), moving)]);
                step_pose_evaluation(&mut world, 1.0 / 60.0, &machines, &skeletons, &clips, &vars);
                for cue in crate::anim_bridge::footstep_cues(&world) {
                    fired.push(cue.name.clone());
                    gain = cue.gain;
                }
            }
            (fired, gain)
        };

        // Leaning left: the left clip leads, so its footstep rings at its gain.
        let (left_fired, left_gain) = run(0.0);
        assert_eq!(
            left_fired,
            vec!["footstep_left".to_string()],
            "{left_fired:?}"
        );
        assert!(
            (left_gain - 0.25).abs() < 1e-6,
            "the leader's Mask_FootstepSound did not reach the drain: {left_gain}"
        );

        // **The control**: lean the other way and the OTHER clip leads. Without
        // it, "fires the leader's markers" and "fires entry 0's markers" are the
        // same assertion.
        let (right_fired, right_gain) = run(1.0);
        assert_eq!(
            right_fired,
            vec!["footstep_right".to_string()],
            "{right_fired:?}"
        );
        assert!((right_gain - 0.75).abs() < 1e-6, "{right_gain}");

        // **The control that actually separates them** (P29.5 audit, A4). Both
        // probes above are at a *pure* weight, where `blend_weights_1d` returns
        // one surviving entry and "the heaviest" and "the first" are the same
        // index — so a `blend_leader` that answered `resolved[0]` passed them
        // both. In between, the list is `[(lo, 1-frac), (hi, frac)]` in POSITION
        // order, so past the midpoint the leader is the second entry and the
        // first is not. That is where the claim lives.
        let (mostly_right, gain_r) = run(0.7);
        assert_eq!(
            mostly_right,
            vec!["footstep_right".to_string()],
            "at weight 0.7 the RIGHT clip is the heaviest, so its marker is the \
             one crossed — reading entry 0 instead answers `footstep_left`: \
             {mostly_right:?}"
        );
        assert!((gain_r - 0.75).abs() < 1e-6, "{gain_r}");
        let (mostly_left, gain_l) = run(0.3);
        assert_eq!(
            mostly_left,
            vec!["footstep_left".to_string()],
            "{mostly_left:?}"
        );
        assert!((gain_l - 0.25).abs() < 1e-6, "{gain_l}");
    }

    /// **The pelvis IK offset reaches the rig** (P29.5 — P29.4 audit A9's
    /// "recorded rather than applied", closed).
    ///
    /// The audit's own bound was that the value had "no consumer to falsify
    /// against", so its arm could only assert the inert path. This is the
    /// consumer: the number the movement step records moves the posed hips, on
    /// the same fixed step, and a zero one moves nothing at all.
    ///
    /// The control is the same world with the offset left at zero — without it,
    /// "the pelvis is at −0.1" and "the pelvis was always at −0.1" are the same
    /// assertion.
    #[test]
    fn a_recorded_pelvis_offset_drops_the_posed_hips_and_a_zero_one_does_not() {
        let guid = Uuid::from_u128(66);
        let f = Fixture::new();
        let posed_root_y = |drop: f64| -> f32 {
            let mut world = world_with_character(guid);
            let e = world.entity_of(guid).unwrap();
            let cm = crate::components::CharacterMovement {
                runtime: crate::components::MovementRuntime {
                    pelvis_offset: crate::math::Vec3d::new(0.0, drop, 0.0),
                    ..Default::default()
                },
                ..Default::default()
            };
            world.world_mut().entity_mut(e).insert(cm);
            world.reindex_guids();
            f.step(&mut world, 1.0 / 60.0, 0.0);
            evaluated_pose(&world, guid).expect("a pose").pose.locals[0].translation[1]
        };
        let level = posed_root_y(0.0);
        let dropped = posed_root_y(-0.1);
        assert!(
            (dropped - (level - 0.1)).abs() < 1e-6,
            "the hips did not come down by the offset: {level} -> {dropped}"
        );
        // …and a NON-finite offset is a value, not a poisoned pose.
        assert_eq!(posed_root_y(f64::NAN), level);
    }

    /// The bridge's own resource obeys the absent-costs-nothing rule: a level
    /// with no machines never grows one, and `clear_poses` forgets it.
    #[test]
    fn a_level_with_no_bridge_traffic_never_grows_the_resource() {
        let guid = Uuid::from_u128(45);
        let mut empty = EcsWorld::new();
        let f = Fixture::new();
        f.step(&mut empty, 1.0 / 60.0, 0.0);
        assert!(crate::anim_bridge::bridge(&empty).is_none());

        // With a machine the states map is published, so the resource exists...
        let mut w = world_with_character(guid);
        f.step(&mut w, 1.0 / 60.0, 0.0);
        assert!(crate::anim_bridge::bridge(&w).is_some());
        // ...and stopping the session forgets it entirely.
        clear_poses(&mut w);
        assert!(crate::anim_bridge::bridge(&w).is_none());
    }

    /// The two helpers the marker scan rests on, probed directly: a marker is
    /// crossed exactly once however the steps land on it, including across the
    /// loop seam.
    #[test]
    fn the_marker_window_is_half_open_and_wraps() {
        let markers = vec![
            inf_anim::AnimMarker::event(0.25, "a"),
            inf_anim::AnimMarker::sync(0.25, "s", "foot"),
        ];
        assert_eq!(crossed_markers(&markers, 0.2, 0.3, false), ["a"]);
        assert_eq!(
            crossed_markers(&markers, 0.25, 0.3, false),
            Vec::<&str>::new()
        );
        assert_eq!(crossed_markers(&markers, 0.2, 0.25, false), ["a"]);
        // Across the seam of a looping clip, both pieces count.
        assert_eq!(crossed_markers(&markers, 0.9, 0.3, true), ["a"]);
        assert_eq!(
            crossed_markers(&markers, 0.9, 0.1, true),
            Vec::<&str>::new()
        );
        assert_eq!(
            crossed_markers(&markers, 0.3, 0.3, false),
            Vec::<&str>::new()
        );

        // The play-head resolver this step used to keep its own copy of is now
        // `inf_anim::clip::resolve_time`, reached through `motion_leader` — one
        // rule for "where is this clip's play-head", asserted here so a second
        // one cannot come back.
        assert_eq!(inf_anim::clip::resolve_time(1.25, 1.0, true), 0.25);
        assert_eq!(inf_anim::clip::resolve_time(1.25, 1.0, false), 1.0);
        assert_eq!(inf_anim::clip::resolve_time(-0.25, 1.0, true), 0.75);
        assert_eq!(inf_anim::clip::resolve_time(0.5, 0.0, true), 0.0);
        // A NaN play-head is NOT collapsed to zero here, and the copy this step
        // used to keep did collapse it — which is worth stating, because it is
        // the one behavioural difference between the two. It is survivable in
        // both directions and the shared rule is the honest one: `locate` is
        // total over a NaN probe (the C4-7 rule) and answers the first key, and
        // every comparison a NaN takes part in is false, so a NaN head crosses no
        // marker and reads the clip's first curve values. The collapse would have
        // made it read the first frame and cross every marker before it.
        assert!(inf_anim::clip::resolve_time(f32::NAN, 1.0, true).is_nan());
        assert_eq!(
            crossed_markers(&markers, f32::NAN, f32::NAN, true),
            Vec::<&str>::new()
        );
    }

    /// **The footstep stream is a pure function of sim state** (P29.4, clause 7
    /// under the P12 doctrine): two identical worlds, stepped identically,
    /// produce identical cue lists — the same names, the same order, the same
    /// gains — and a clip's `Mask_FootstepSound` channel is what scales them.
    #[test]
    fn the_footstep_cues_are_a_pure_function_of_sim_state() {
        let guid = Uuid::from_u128(46);
        let mut f = WalkFixture::new();
        // The animator's volume mask: this walk's footsteps are half volume.
        f.clip = f
            .clip
            .clone()
            .with_curves(vec![inf_anim::CurveChannel::constant(
                inf_anim::channels::als::MASK_FOOTSTEP_SOUND,
                0.5,
            )]);

        let run = || {
            let mut w = world_with_character(guid);
            crate::anim_bridge::set_anim_trigger(&mut w, guid, "go");
            let mut all: Vec<Vec<crate::anim_bridge::FootstepCue>> = Vec::new();
            for _ in 0..40 {
                f.step(&mut w, 1.0 / 60.0);
                all.push(crate::anim_bridge::footstep_cues(&w));
            }
            all
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "two identical worlds produced different footsteps");

        // NOT VACUOUS: something really did ring, exactly once, at the gain the
        // clip's own channel asked for.
        let fired: Vec<&crate::anim_bridge::FootstepCue> = a.iter().flatten().collect();
        assert_eq!(fired.len(), 1, "{fired:?}");
        assert_eq!(fired[0].name, "footstep_l");
        assert_eq!(fired[0].source, guid);
        assert!((fired[0].gain - 0.5).abs() < 1e-6, "{:?}", fired[0].gain);

        // A machine that never rings produces an empty stream rather than an
        // absent one — the difference a host's drain would notice.
        let clean = world_with_character(Uuid::from_u128(47));
        assert!(crate::anim_bridge::footstep_cues(&clean).is_empty());
    }

    /// The **foot matcher** takes the spellings a rig actually arrives with, and
    /// refuses the ones that are not a side.
    #[test]
    fn the_foot_matcher_reads_every_spelling_a_rig_arrives_with() {
        // A rig with NO role table, which is what every rig older than
        // `.inf_skel` v3 is and what these spellings are the whole answer for.
        fn skel(names: &[&str]) -> inf_anim::SkeletonAsset {
            let joints = names
                .iter()
                .enumerate()
                .map(|(i, n)| Joint {
                    name: (*n).into(),
                    parent: (i > 0).then(|| (i - 1) as u16),
                    inverse_bind: Mat4::IDENTITY.to_cols_array(),
                    local_bind: JointTransform::IDENTITY,
                })
                .collect();
            inf_anim::SkeletonAsset::new(inf_anim::Skeleton::new(joints).unwrap())
        }
        // This engine's own template, ALS's, and the Mixamo/UE spellings.
        for (names, want) in [
            (vec!["Hips", "Foot.L", "Foot.R"], [Some(1u16), Some(2u16)]),
            (vec!["root", "ik_foot_l", "ik_foot_r"], [Some(1), Some(2)]),
            (vec!["x", "LeftFoot", "RightFoot"], [Some(1), Some(2)]),
            (vec!["x", "foot_r", "foot_l"], [Some(2), Some(1)]),
        ] {
            assert_eq!(foot_joints(&skel(&names)), want, "{names:?}");
        }
        // A foot with no side is not a foot this pass can use, and a rig with no
        // feet at all answers nothing rather than guessing.
        assert_eq!(foot_joints(&skel(&["a", "foot"])), [None, None]);
        assert_eq!(foot_joints(&skel(&["a", "hand_l"])), [None, None]);
        // The FIRST match wins, so a toe does not displace the ankle.
        assert_eq!(
            foot_joints(&skel(&["a", "foot_l", "foot_l_toe"])),
            [Some(1), None]
        );
    }

    /// **THE DRIVE PASS RUNS INSIDE THE FIXED STEP** (SK1a), on a real mannequin,
    /// and what it produces reaches the pose store and the determinism trace.
    ///
    /// Two claims, and the second is the load-bearing one. The twist bones move
    /// when the joint that drives them does — an engagement counter in bones, not
    /// a "the function was called". And the IK handles land **on** the joints they
    /// mark after the whole rig has moved, which they only can if the pass ran
    /// after the layer stack rather than at authoring time.
    ///
    /// Mutation-relevant: deleting the `drive_pose` call leaves every twist bone
    /// at its bind (the first half fails) and every handle at the pose the clip
    /// left it in, which for `ik_hand_l` is a whole arm away (the second).
    #[test]
    fn the_drive_pass_runs_in_the_fixed_step_and_reaches_the_pose_store() {
        let guid = Uuid::from_u128(0x5_1a_d1);
        let rig =
            inf_anim::build_template(inf_anim::BodyPlan::Biped, &inf_anim::BodyParams::default())
                .unwrap();
        let sk = &rig.skeleton;
        let at = |name: &str| sk.index_of(name).unwrap();

        // A clip that rolls the LEFT WRIST — the joint `lowerarm_twist_*_l` reads —
        // and swings the whole arm at the shoulder, so the handle has somewhere to
        // travel to.
        let mut clip = inf_anim::AnimClip::new("roll", Vec::new());
        clip.duration = 1.0;
        for (joint, q) in [
            (at("hand_l"), glam::Quat::from_rotation_x(1.2)),
            (at("upperarm_l"), glam::Quat::from_rotation_z(0.7)),
        ] {
            let mut track = inf_anim::JointTrack::new(joint);
            track.rotation = Some(inf_anim::QuatTrack::new(
                vec![0.0, 1.0],
                vec![q.to_array(), q.to_array()],
                inf_anim::Interpolation::Linear,
            ));
            clip.tracks.push(track);
        }
        const C: ClipRef = [77u8; 16];
        let machine = StateMachine {
            states: vec![SmState::clip("roll", C)],
            transitions: Vec::new(),
            entry: 0,
            ..Default::default()
        };

        let mut world = world_with_character(guid);
        let machines = |g: Uuid| (g == SM).then_some(&machine);
        let skeletons = |g: Uuid| (g == SKEL).then_some(&rig);
        let clips = |c: ClipRef| (c == C).then_some(&clip);
        let vars = |_: Uuid| BTreeMap::new();
        step_pose_evaluation(&mut world, 1.0 / 60.0, &machines, &skeletons, &clips, &vars);

        let posed = evaluated_pose(&world, guid).expect("the character posed");
        assert_eq!(posed.pose.locals.len(), inf_anim::MANNY_JOINT_COUNT);

        // (1) The twists moved, by the fractions the rig authored: the bone two
        // thirds of the way to the wrist takes two thirds of the wrist's roll.
        let roll_of = |name: &str| -> f32 {
            let q = glam::Quat::from_array(posed.pose.locals[at(name) as usize].rotation);
            // The half-angle about X, read off the quaternion rather than through
            // an `acos` — this is a test of a value, not of an angle library.
            q.x.abs()
        };
        let full = glam::Quat::from_rotation_x(1.2).x;
        let two_thirds = glam::Quat::from_rotation_x(1.2 * 2.0 / 3.0).x;
        let one_third = glam::Quat::from_rotation_x(1.2 / 3.0).x;
        assert!(
            (roll_of("lowerarm_twist_01_l") - two_thirds).abs() < 1.0e-4,
            "twist_01 took {} of a full {full}",
            roll_of("lowerarm_twist_01_l")
        );
        assert!((roll_of("lowerarm_twist_02_l") - one_third).abs() < 1.0e-4);
        // The right arm did not move, so its twists are still at bind — the
        // control that stops "everything rotated" passing this.
        assert!(roll_of("lowerarm_twist_01_r") < 1.0e-6);

        // (2) The handles landed on their sources, in MODEL space, after the arm
        // swung. `ik_hand_l` hangs off `ik_hand_gun` off `ik_hand_root` off the
        // root — nowhere near the hand until something drives it.
        let globals = inf_anim::global_transforms(sk, &posed.pose);
        let at_p = |name: &str| globals[at(name) as usize].transform_point3(glam::Vec3::ZERO);
        for (handle, source) in [
            ("ik_hand_l", "hand_l"),
            ("ik_hand_r", "hand_r"),
            ("ik_foot_l", "foot_l"),
            ("ik_hand_gun", "hand_r"),
        ] {
            let d = (at_p(handle) - at_p(source)).length();
            assert!(d < 1.0e-4, "`{handle}` is {d} m from `{source}`");
        }
        // …and the hand really did travel, so the arm above is not "both at bind".
        let rest = inf_anim::global_transforms(sk, &inf_anim::Pose::rest(sk));
        let moved = (at_p("hand_l")
            - rest[at("hand_l") as usize].transform_point3(glam::Vec3::ZERO))
        .length();
        assert!(
            moved > 0.1,
            "the shoulder swing moved the hand only {moved} m"
        );

        // (3) It is in the trace: the bytes are a function of the driven pose.
        let bytes = pose_state_bytes(&world);
        assert_eq!(
            bytes.len(),
            16 + 16 + 4 + inf_anim::MANNY_JOINT_COUNT * 40,
            "36 bytes of header plus 40 per joint"
        );
    }

    /// **The role table beats the name, and the name is what it beats** (SK1a).
    ///
    /// The mannequin ships `foot_l` AND `ik_foot_l`, so the two rules can
    /// disagree, and this is the one place in the engine where being wrong about
    /// it still solves perfectly: a chain derived from `ik_foot_l` is
    /// `[root, ik_foot_root, ik_foot_l]`, three real joints that IK will happily
    /// converge on, and the leg never moves.
    ///
    /// Built so the two answers CANNOT coincide: the handles are emitted first, so
    /// the name rule's first match is the handle and the table's answer is the
    /// ankle. On the real mannequin the emission order makes them agree — that is
    /// the belt, and this is the suspenders.
    #[test]
    fn the_role_table_outranks_the_name_where_the_two_disagree() {
        let names = ["root", "ik_foot_l", "ik_foot_r", "foot_l", "foot_r"];
        let joints = names
            .iter()
            .enumerate()
            .map(|(i, n)| Joint {
                name: (*n).into(),
                parent: (i > 0).then_some(0u16),
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            })
            .collect();
        let mut rig = inf_anim::SkeletonAsset::new(inf_anim::Skeleton::new(joints).unwrap());
        // No table: the name rule answers, and it answers with the HANDLES.
        assert_eq!(foot_joints(&rig), [Some(1), Some(2)]);
        // With a table: the ankles, and nothing about the spelling changed.
        use inf_anim::{BoneRole, BoneRoleKind, BoneSide};
        rig.roles = vec![
            BoneRole::new(1, BoneRoleKind::IkTarget, BoneSide::Left),
            BoneRole::new(2, BoneRoleKind::IkTarget, BoneSide::Right),
            BoneRole::new(3, BoneRoleKind::Foot, BoneSide::Left),
            BoneRole::new(4, BoneRoleKind::Foot, BoneSide::Right),
        ];
        assert_eq!(foot_joints(&rig), [Some(3), Some(4)]);
        // A table that describes something else entirely does not silently take
        // the feet away: a rig whose table has no `Foot` row falls back.
        rig.roles = vec![BoneRole::new(0, BoneRoleKind::Root, BoneSide::Center)];
        assert_eq!(foot_joints(&rig), [Some(1), Some(2)]);
    }

    /// **The pelvis, from the table.** The old rule found a bone spelled
    /// `pelvis` and otherwise the root; the mannequin has one, so the arm that
    /// matters is the rig where the two disagree.
    #[test]
    fn the_pelvis_comes_from_the_table_before_the_spelling() {
        let joints = ["root", "hips", "pelvis"]
            .iter()
            .enumerate()
            .map(|(i, n)| Joint {
                name: (*n).into(),
                parent: (i > 0).then(|| (i - 1) as u16),
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            })
            .collect();
        let mut rig = inf_anim::SkeletonAsset::new(inf_anim::Skeleton::new(joints).unwrap());
        assert_eq!(pelvis_joint(&rig), Some(2), "the spelling, as before");
        use inf_anim::{BoneRole, BoneRoleKind, BoneSide};
        rig.roles = vec![BoneRole::new(1, BoneRoleKind::Pelvis, BoneSide::Center)];
        assert_eq!(pelvis_joint(&rig), Some(1), "the table");
        // …and a rig with neither still answers its root rather than nothing.
        rig.roles.clear();
        let bare = inf_anim::SkeletonAsset::new(
            inf_anim::Skeleton::new(vec![Joint {
                name: "b0".into(),
                parent: None,
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            }])
            .unwrap(),
        );
        assert_eq!(pelvis_joint(&bare), Some(0));
    }

    /// **The garment lift** (P29.7): the offset a projector adds to a
    /// host-interpolated translation is exactly the one `model_to_world`
    /// composes, and it is zero for anything that is not a character.
    ///
    /// This closes the P29.6 remainder by name: the cloth and hair projectors
    /// were handed the entity's translation — a capsule CENTRE — while a
    /// garment's vertices are in feet-at-origin model space, so a coat was drawn
    /// nearly a metre above its wearer. Both hosts add this now.
    #[test]
    fn the_garment_lift_is_the_one_the_model_door_composes() {
        use crate::components::{CharacterMovement, Collider3D, ColliderShape3DKind, Transform};
        let mut world = EcsWorld::new();

        // A prop: no movement component, no lift, and therefore not one micron
        // of movement for anything that existed before this function.
        let prop = world.spawn_with_guid(Uuid::from_u128(0x2907_3001), "Prop", None);
        world
            .world_mut()
            .entity_mut(prop)
            .insert(Transform::from_translation(glam::DVec3::new(3.0, 2.0, 1.0)));
        world.mark_dirty();
        world.propagate();
        assert_eq!(model_offset_world(&world, prop), glam::DVec3::ZERO);

        // A character wearing a capsule: the lift is the worn half-height plus
        // the radius, downward, and it is read off the door rather than restated
        // as a constant here.
        let cm = CharacterMovement::default();
        let radius = 0.3;
        let hero = world.spawn_with_guid(Uuid::from_u128(0x2907_3002), "Hero", None);
        world.world_mut().entity_mut(hero).insert((
            Transform::from_translation(glam::DVec3::new(
                0.0,
                cm.stand_half_height_m + radius,
                0.0,
            )),
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: crate::math::Vec3d::new(radius, cm.stand_half_height_m, radius),
                radius,
                ..Default::default()
            },
            cm.clone(),
        ));
        world.mark_dirty();
        world.propagate();
        let offset = model_offset_world(&world, hero);
        assert!(offset.y < -0.5, "the lift is {offset:?}, not a drop");
        assert!(offset.x.abs() < 1e-12 && offset.z.abs() < 1e-12);

        // …and it is the SAME offset the whole-affine door composes, which is
        // what stops the two from drifting: a projector that added this to the
        // entity's translation must land where `model_to_world` puts the pose.
        let door = model_to_world(&world, hero).translation;
        let raw = world
            .world()
            .get::<crate::components::GlobalTransform>(hero)
            .map(|g| g.0.translation)
            .expect("propagated");
        assert!(
            (door - (raw + offset)).length() < 1e-12,
            "the offset {offset:?} does not compose to the door's {door:?}"
        );
    }

    // ── SK1b: hands ─────────────────────────────────────────────────────────

    /// A world with one mannequin-rigged character at the origin.
    fn world_with_mannequin(guid: Uuid) -> EcsWorld {
        let mut world = EcsWorld::new();
        world.world_mut().spawn((
            Guid(guid),
            Transform::default(),
            GlobalTransform(glam::DAffine3::IDENTITY),
            AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(9)),
                skeleton: Some(SKEL),
            },
        ));
        world.reindex_guids();
        world
    }

    /// The mannequin, **with the catalogue it generates for itself** (SK1c).
    ///
    /// This fixture used to author `rifle_l` / `rifle_r` here, which made two
    /// hand-written catalogues in the tree — this one and the grip gate's — with
    /// different names and different apertures for the same idea. The generator
    /// is the single door now; the arms below name `inf_anim::GRIP_RIFLE` and
    /// `GRIP_RIFLE_FORE`, which is the *left* hand's affordance, so the two-hand
    /// coverage the old pair gave is kept.
    fn mannequin_rig() -> SkeletonAsset {
        inf_anim::build_manny(&inf_anim::BodyParams::default()).expect("the mannequin builds")
    }

    /// Step a mannequin character through the one fixed-step pose door.
    fn step_mannequin(world: &mut EcsWorld, rig: &SkeletonAsset, dt: f64) {
        let m = machine();
        let clip = wave_clip();
        let machines = |g: Uuid| (g == SM).then_some(&m);
        let skeletons = |g: Uuid| (g == SKEL).then_some(rig);
        let clips = |c: ClipRef| (c == WAVE).then_some(&clip);
        let vars = |_: Uuid| BTreeMap::new();
        step_pose_evaluation(world, dt, &machines, &skeletons, &clips, &vars);
    }

    /// Where a joint ended up, in model space.
    fn joint_at(posed: &EvaluatedPose, rig: &SkeletonAsset, joint: u16) -> glam::Vec3 {
        inf_anim::global_transforms(&rig.skeleton, &posed.pose)[joint as usize]
            .transform_point3(glam::Vec3::ZERO)
    }

    /// **The headline for the reaching half**: a hand set on a point in the world
    /// lands on it, through the same one door both hosts call, with no schema
    /// move and no component.
    ///
    /// The elbow's *pole* is the part that needs asserting beside the reach,
    /// because a T-pose bind has shoulder, elbow and wrist exactly collinear and
    /// a solve with no pole bends whichever way rounding decides. Backward is
    /// `-Z`, so the elbow must end up behind the line from shoulder to wrist.
    #[test]
    fn a_hand_reaches_a_point_in_the_world_and_the_elbow_bends_backwards() {
        let guid = Uuid::from_u128(0x5B_0001);
        let rig = mannequin_rig();
        let mut world = world_with_mannequin(guid);
        step_mannequin(&mut world, &rig, 0.1);
        let rest = evaluated_pose(&world, guid).expect("a pose").clone();

        let (upper, lower, hand) = {
            let c = inf_anim::arm_chain(&rig.skeleton, rig.role_index(), inf_anim::BoneSide::Right)
                .expect("a right arm");
            (c[0], c[1], c[2])
        };
        // In front of the chest and a little low — reachable, and nowhere near
        // where a T-pose puts a wrist.
        let target = Vec3d::new(0.25, 1.15, 0.45);
        set_hand_ik(
            &mut world,
            guid,
            HandIk {
                reach: [
                    None,
                    Some(HandReach {
                        target,
                        weight: 1.0,
                    }),
                ],
                ..Default::default()
            },
        );
        step_mannequin(&mut world, &rig, 0.1);
        let posed = evaluated_pose(&world, guid).expect("a pose");
        let wrist = joint_at(posed, &rig, hand);
        let miss = (wrist - glam::Vec3::new(0.25, 1.15, 0.45)).length();
        println!("the wrist landed {miss:.5} m from its target");
        assert!(miss < 0.01, "the hand missed its target by {miss} m");
        // …and it MOVED to get there, which the assertion above would also be
        // happy about if a T-pose wrist happened to be sitting on the target.
        let was = joint_at(&rest, &rig, hand);
        assert!(
            (was - wrist).length() > 0.3,
            "the wrist was already there ({was:?}) — the arm proves nothing"
        );
        // The elbow is behind the shoulder-to-wrist line, which is what the pole
        // is for. Mutation: pass `None` for the pole and this goes red.
        let (s, e) = (joint_at(posed, &rig, upper), joint_at(posed, &rig, lower));
        let along = (wrist - s).normalize();
        let off = (e - s) - along * (e - s).dot(along);
        println!(
            "the elbow sits {:.4} m off the arm line, z = {:.4}",
            off.length(),
            off.z
        );
        assert!(
            off.z < -0.02,
            "the elbow bent to {off:?}, which is not backwards"
        );
        // …and the verdict is readable, which is what a gate asserts.
        let report = hand_ik_report(&world, guid).expect("a hand verdict");
        assert!(
            matches!(report.reach[1], Some(IkOutcome::Solved(_))),
            "{:?}",
            report.reach
        );
        assert!(report.reach[0].is_none(), "the left arm was not asked");
        assert!(report.wrote());
    }

    /// **Two hands on one weapon**: the off hand lands on the fore-grip wherever
    /// the holding hand puts it — the `ik_hand_gun` convention, built.
    #[test]
    fn the_off_hand_follows_the_weapon_the_holding_hand_carries() {
        let guid = Uuid::from_u128(0x5B_0002);
        let rig = mannequin_rig();
        let left = inf_anim::arm_chain(&rig.skeleton, rig.role_index(), inf_anim::BoneSide::Left)
            .expect("a left arm")[2];
        let right = inf_anim::arm_chain(&rig.skeleton, rig.role_index(), inf_anim::BoneSide::Right)
            .expect("a right arm")[2];

        let hold = |world: &mut EcsWorld, at: Vec3d| {
            set_hand_ik(
                world,
                *world
                    .world()
                    .get_resource::<HandIkRes>()
                    .and_then(|r| r.hands.keys().next())
                    .unwrap_or(&guid),
                HandIk {
                    reach: [
                        None,
                        Some(HandReach {
                            target: at,
                            weight: 1.0,
                        }),
                    ],
                    gun: Some(GunGrip {
                        holding: inf_anim::BoneSide::Right,
                        // 30 cm along the barrel — a rifle's fore-grip.
                        off_hand_offset: [0.0, 0.0, 0.30],
                        weight: 1.0,
                    }),
                    grip: [None, None],
                },
            );
        };

        let mut world = world_with_mannequin(guid);
        hold(&mut world, Vec3d::new(0.2, 1.2, 0.35));
        step_mannequin(&mut world, &rig, 0.1);
        let a = evaluated_pose(&world, guid).expect("a pose").clone();
        let (al, ar) = (joint_at(&a, &rig, left), joint_at(&a, &rig, right));
        let apart = (al - ar).length();
        println!("the hands are {apart:.4} m apart on the weapon");
        assert!(
            (apart - 0.30).abs() < 0.06,
            "the two hands are {apart} m apart, not the 0.30 m the weapon is"
        );
        let report = hand_ik_report(&world, guid).expect("a verdict");
        assert!(
            matches!(report.gun, Some(IkOutcome::Solved(_))),
            "{report:?}"
        );

        // Move the weapon and the off hand goes with it — the claim that makes
        // this a HOLD rather than two independent reaches.
        hold(&mut world, Vec3d::new(-0.15, 1.45, 0.30));
        step_mannequin(&mut world, &rig, 0.1);
        let b = evaluated_pose(&world, guid).expect("a pose").clone();
        let (bl, br) = (joint_at(&b, &rig, left), joint_at(&b, &rig, right));
        assert!(
            (ar - br).length() > 0.15,
            "the holding hand did not move, so the off hand proves nothing"
        );
        assert!(
            (al - bl).length() > 0.10,
            "the holding hand moved {:.3} m and the off hand moved {:.3} m",
            (ar - br).length(),
            (al - bl).length()
        );
        let apart = (bl - br).length();
        assert!(
            (apart - 0.30).abs() < 0.06,
            "the hands came {apart} m apart once the weapon moved"
        );
    }

    /// **The twist/IK ordering bound, closed** — SK1a's carried item, asserted.
    ///
    /// SK1a's drive pass runs at pose construction and its docs said so: *"a
    /// twist reflects the pose the animation authored, not the pose the IK below
    /// goes on to correct."* The claim now is the opposite one, and it is
    /// checkable without a second pipeline: **re-running the drive over the
    /// published pose changes nothing**, which is true exactly when the twists
    /// already reflect the corrected arm.
    ///
    /// Two anti-vacuity halves, because the assertion above is also satisfied by
    /// a rig with no twists and by an IK solve that did nothing: the same
    /// character posed WITHOUT the reach must produce different twist bones, and
    /// the twist bones must not be the identity.
    #[test]
    fn a_twist_bone_reflects_the_arm_the_hand_ik_solved_not_the_one_the_clip_authored() {
        let guid = Uuid::from_u128(0x5B_0003);
        let rig = mannequin_rig();
        // The UPPER arm's twist, deliberately: a lower segment's twist reads
        // its distal child's roll (`lowerarm_twist_01_r` reads `hand_r`), and an
        // arm IK solve writes the shoulder and the elbow — never the wrist,
        // which is the chain's tip. So a forearm twist is invariant under this
        // correction and would make the anti-vacuity half of this arm vacuous.
        let twist = rig
            .skeleton
            .index_of("upperarm_twist_01_r")
            .expect("the upper-arm twist") as usize;

        let mut plain = world_with_mannequin(guid);
        step_mannequin(&mut plain, &rig, 0.1);
        let uncorrected = evaluated_pose(&plain, guid).expect("a pose").clone();

        let mut world = world_with_mannequin(guid);
        set_hand_ik(
            &mut world,
            guid,
            HandIk {
                reach: [
                    None,
                    Some(HandReach {
                        target: Vec3d::new(0.15, 1.30, 0.40),
                        weight: 1.0,
                    }),
                ],
                ..Default::default()
            },
        );
        step_mannequin(&mut world, &rig, 0.1);
        let posed = evaluated_pose(&world, guid).expect("a pose").clone();

        // **The claim**: the published twists are already a function of the
        // published pose. Mutation: delete the `redrive` call and this fails.
        let mut again = posed.pose.clone();
        let drove = inf_anim::drive_twists(&rig.skeleton, &mut again, &rig.twists);
        assert_eq!(
            drove,
            rig.twists.len(),
            "the rig has 16 twist bones to drive"
        );
        for (i, (a, b)) in posed
            .pose
            .locals
            .iter()
            .zip(again.locals.iter())
            .enumerate()
        {
            for (u, v) in a.rotation.iter().zip(b.rotation.iter()) {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "joint {i} `{}` is not what the FINAL pose implies — the twists \
                     were driven from the pose before the IK corrected it",
                    rig.skeleton.joints()[i].name
                );
            }
        }

        // Anti-vacuity 1: the solve really moved this twist.
        let before = uncorrected.pose.locals[twist].rotation;
        let after = posed.pose.locals[twist].rotation;
        let delta = glam::Quat::from_array(before)
            .dot(glam::Quat::from_array(after))
            .abs();
        println!("the forearm twist moved: |dot| = {delta:.6}");
        assert!(
            delta < 0.9999,
            "`lowerarm_twist_01_r` is identical with and without the reach, so the \
             re-drive is untested here"
        );
        // Anti-vacuity 2: it is a real roll, not the identity.
        assert!(
            glam::Quat::from_array(after)
                .dot(glam::Quat::IDENTITY)
                .abs()
                < 0.9999,
            "the twist bone is the identity, so nothing was driven"
        );
        // …and the report counts what the re-drive rewrote.
        let report = hand_ik_report(&world, guid).expect("a verdict");
        assert_eq!(
            report.redriven,
            rig.twists.len() + rig.ik_follow.len(),
            "the re-drive should touch every twist and every handle"
        );
    }

    /// **A grip closes the fingers through the fixed step**, and a release opens
    /// them again — the `GripAffordance` table's first runtime consumer.
    #[test]
    fn a_grip_closes_the_fingers_through_the_fixed_step_and_a_release_opens_them() {
        let guid = Uuid::from_u128(0x5B_0004);
        let rig = mannequin_rig();
        let tip = rig.skeleton.index_of("middle_03_r").expect("a fingertip");
        let wrist = rig.skeleton.index_of("hand_r").expect("a wrist");

        let grip = |world: &mut EcsWorld, amount: f32| {
            set_hand_ik(
                world,
                guid,
                HandIk {
                    grip: [
                        None,
                        Some(HandGrip {
                            name: inf_anim::GRIP_RIFLE.into(),
                            amount,
                        }),
                    ],
                    ..Default::default()
                },
            );
        };

        let mut world = world_with_mannequin(guid);
        step_mannequin(&mut world, &rig, 0.1);
        let open = evaluated_pose(&world, guid).expect("a pose").clone();
        let span = |p: &EvaluatedPose| (joint_at(p, &rig, tip) - joint_at(p, &rig, wrist)).length();
        let straight = span(&open);

        grip(&mut world, 1.0);
        step_mannequin(&mut world, &rig, 0.1);
        let closed = evaluated_pose(&world, guid).expect("a pose").clone();
        let held = span(&closed);
        println!("fingertip to wrist: {straight:.4} m open, {held:.4} m closed");
        assert!(
            held < straight * 0.75,
            "the hand did not close: {held} m against {straight} m"
        );
        let report = hand_ik_report(&world, guid).expect("a verdict");
        assert!(report.grip[1].joints >= 15, "{:?}", report.grip[1]);
        assert_eq!(report.grip[0].joints, 0, "the left hand was not asked");

        // A release returns the hand EXACTLY to where an ungripped one poses —
        // the claim `apply_grip`'s "a curl is a pose, not a delta" rests on.
        grip(&mut world, 0.0);
        step_mannequin(&mut world, &rig, 0.1);
        let released = evaluated_pose(&world, guid).expect("a pose").clone();
        assert_eq!(
            released.pose.locals, open.pose.locals,
            "a released hand did not return to the open pose"
        );
    }

    /// **Absent costs nothing**: a character nobody asked anything of poses the
    /// bytes it posed before this wave, and the resource does not appear.
    #[test]
    fn a_character_with_no_hand_request_poses_exactly_what_it_did_before() {
        let guid = Uuid::from_u128(0x5B_0005);
        let rig = mannequin_rig();
        let mut plain = world_with_mannequin(guid);
        step_mannequin(&mut plain, &rig, 0.1);
        let bytes = pose_state_bytes(&plain);
        assert!(!bytes.is_empty(), "the fixture posed nothing");
        assert!(
            plain.world().get_resource::<HandIkRes>().is_none(),
            "a level that asked for no hands grew a resource"
        );

        // Setting an EMPTY request is the same as setting none — one
        // representation, so a level that stops gripping stops paying.
        let mut same = world_with_mannequin(guid);
        set_hand_ik(&mut same, guid, HandIk::default());
        step_mannequin(&mut same, &rig, 0.1);
        assert!(same.world().get_resource::<HandIkRes>().is_none());
        assert_eq!(pose_state_bytes(&same), bytes);

        // …and a request whose every part is refusable leaves the pose alone.
        let mut refused = world_with_mannequin(guid);
        set_hand_ik(
            &mut refused,
            guid,
            HandIk {
                reach: [
                    Some(HandReach {
                        target: Vec3d::new(f64::NAN, 0.0, 0.0),
                        weight: 1.0,
                    }),
                    Some(HandReach {
                        target: Vec3d::ZERO,
                        weight: 0.0,
                    }),
                ],
                gun: None,
                grip: [
                    Some(HandGrip {
                        name: "no such grip".into(),
                        amount: 1.0,
                    }),
                    None,
                ],
            },
        );
        step_mannequin(&mut refused, &rig, 0.1);
        assert_eq!(
            pose_state_bytes(&refused),
            bytes,
            "a refusable request moved the pose"
        );
        let report = hand_ik_report(&refused, guid).expect("a verdict");
        assert!(!report.wrote(), "{report:?}");
        assert_eq!(
            report.redriven, 0,
            "the re-drive ran for a pose nothing corrected"
        );
    }
}
