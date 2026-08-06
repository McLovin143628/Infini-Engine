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
//! * not a component, so **no schema moves** (scene v20 is frozen; the scene
//!   serializer walks entities and components, never resources, so there is no
//!   path from an evaluated pose to a file — which is correct, a pose is derived
//!   state and not authored content);
//! * not a host member, because both scene projectors read the sim world and
//!   neither reads the host struct;
//! * still sim state in every sense that matters: written only from the fixed
//!   step, a pure function of the step history.
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
//! same reason (the rule lives once). [`crate::components::SmRuntimeState`]
//! survives the dependency because the *component* must derive `Reflect` +
//! serde and [`inf_anim::SmRuntime`] derives neither; [`to_anim_runtime`] /
//! [`from_anim_runtime`] are now that mirror's single conversion, instead of one
//! copy per host.

use std::collections::BTreeMap;

use bevy_ecs::prelude::{Entity, Resource};
use glam::Mat4;
use inf_anim::{AnimClip, ClipRef, Pose, SkeletonAsset, SmContext, SmRuntime, StateMachine};
use uuid::Uuid;

use crate::components::{AnimStateMachine, Guid, SkeletalMesh, SmRuntimeState};
use crate::world::EcsWorld;

/// Convert the ECS component's transient runtime POD into the anim runtime.
///
/// One copy, in Ring 0 — it used to be a private helper spelled identically in
/// `SimSession` and `RuntimeSim`, which is a mirror pair maintained by hand for
/// a struct-to-struct field copy.
pub fn to_anim_runtime(s: SmRuntimeState) -> SmRuntime {
    SmRuntime {
        current: s.current,
        prev: s.prev,
        prev_time: s.prev_time,
        fade_t: s.fade_t,
        fade_dur: s.fade_dur,
        state_time: s.state_time,
        started: s.started,
    }
}

/// Convert the advanced anim runtime back into the ECS component POD.
pub fn from_anim_runtime(r: SmRuntime) -> SmRuntimeState {
    SmRuntimeState {
        current: r.current,
        prev: r.prev,
        prev_time: r.prev_time,
        fade_t: r.fade_t,
        fade_dur: r.fade_dur,
        state_time: r.state_time,
        started: r.started,
    }
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
pub fn clear_poses(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<PoseStoreRes>();
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
/// **`exit_time` still has no period resolver** ([`SmContext::new`], not
/// `with_period`) — the documented P11.2 no-deadlock fallback. The clips needed
/// to close it are now in reach for the first time, but turning it on would move
/// every existing machine's transition timing, which is a behaviour change and
/// not a repair; it is ledgered, not smuggled in.
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
        return;
    }
    targets.sort_by_key(|(_, guid, _, _, _)| *guid);

    // 2. Advance + evaluate.
    let mut posed: BTreeMap<Uuid, EvaluatedPose> = BTreeMap::new();
    for (entity, guid, sm_guid, rt_state, skeleton_id) in targets {
        let Some(machine) = machines(sm_guid) else {
            continue;
        };
        let actor_vars = vars(guid);
        let mut rt = to_anim_runtime(rt_state);
        {
            let lookup = |name: &str| actor_vars.get(name).copied();
            let ctx = SmContext::new(&lookup);
            rt.advance(machine, &ctx, dt);
            // Rule 3: no skeleton ⇒ the machine still steps, nothing is posed.
            if let Some(id) = skeleton_id {
                if let Some(asset) = skeletons(id) {
                    if !asset.skeleton.is_empty() {
                        let pose = inf_anim::eval_pose(machine, &rt, &asset.skeleton, clips, &ctx);
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
        if let Some(mut asm) = world.world_mut().get_mut::<AnimStateMachine>(entity) {
            asm.runtime = from_anim_runtime(rt);
        }
    }

    // 3. Publish (rule 4).
    let w = world.world_mut();
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
        AnimClip {
            name: "wave".into(),
            duration: 1.0,
            tracks: vec![JointTrack {
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
        }
    }

    /// idle → wave when `moving > 0.5`. Idle plays a clip nothing resolves, so
    /// its pose is the rest pose and the two states are distinguishable.
    fn machine() -> StateMachine {
        StateMachine {
            states: vec![SmState::clip("idle", IDLE), SmState::clip("wave", WAVE)],
            transitions: vec![SmTransition {
                from: 0,
                to: 1,
                duration: 0.0,
                conditions: vec![inf_anim::SmCondition {
                    var: "moving".into(),
                    op: inf_anim::CmpOp::Gt,
                    value: 0.5,
                }],
                exit_time: None,
            }],
            entry: 0,
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

    /// The runtime POD conversion round-trips (it is the one copy now).
    #[test]
    fn runtime_state_round_trips() {
        let s = SmRuntimeState {
            current: 3,
            prev: Some(1),
            prev_time: 0.5,
            fade_t: 0.25,
            fade_dur: 0.5,
            state_time: 1.25,
            started: true,
        };
        assert_eq!(from_anim_runtime(to_anim_runtime(s)), s);
    }
}
