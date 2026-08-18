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

/// Choose how state transitions blend, for every posed entity in this world.
///
/// The **selector** `.inf_sm` cannot carry: a per-transition choice would be a
/// field on `SmTransition`, and that format bumped in P29.1 and does not bump
/// again in this phase. So the choice is world-level and lives in a resource,
/// which is enough for a project that wants the P29.1 cross-fade back and is the
/// seam P29.4's `anim.*` kit exposes.
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
    clear_ik_goals(world);
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
        let actor_vars = vars(guid);
        let mut pending_pose: Option<Pose> = None;
        let mut rt = rt_state;
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
                    // A rig swap invalidates the captured deviation — a delta
                    // between two different joint counts is not a delta.
                    blender.fit_rig(asset.skeleton.len());
                    let (pose, step) =
                        blender.step(machine, &mut rt, &asset.skeleton, clips, &ctx, dt);
                    pending_pose = Some(pose);
                    step
                }
            };
            if !step.events.is_empty() {
                fired_events.insert(guid, step.events);
            }
            // Rule 3: no skeleton ⇒ the machine still steps, nothing is posed.
            if let Some(id) = skeleton_id {
                if let Some(asset) = skeletons(id) {
                    if !asset.skeleton.is_empty() {
                        let mut pose = pending_pose
                            .take()
                            .unwrap_or_else(|| Pose::rest(&asset.skeleton));
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
}
