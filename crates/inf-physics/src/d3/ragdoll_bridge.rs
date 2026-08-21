//! **The ragdoll bridge** (P29.4, clause 6) — the physics side of the handoff,
//! under the P12 command-queue doctrine.
//!
//! # The doctrine, and what it forbids
//!
//! §13's catalogue amendment binds this seam: *blend weights are a pure function
//! of sim state, and physics never mutates the machine directly.* ALS breaks both
//! halves — its AnimBP reads the physics body for `FlailRate`, and `RagdollEnd`
//! pushes a pose snapshot into the graph imperatively (port map §4.3). Here every
//! crossing is a value in [`inf_ecs::anim_bridge`]:
//!
//! * **anim → physics**: the pose step publishes a world-space rig on request
//!   ([`inf_ecs::anim_bridge::RigBone`]); this module turns it into bodies and
//!   joints. It is the only crossing that needs a skeleton, and the skeleton
//!   lives on the other side.
//! * **physics → anim**: this module writes *numbers* — a face-up flag, a pelvis
//!   yaw, a phase and a clock — onto the character's own runtime, and sets an
//!   animation parameter and a trigger through the same Ring-0 doors the `anim.*`
//!   Blueprint kit uses. It never touches `SmRuntime`.
//!
//! The blend weight itself is [`inf_anim::ragdoll::blend_weight`], a function of
//! `(phase, clock)` — so two worlds in the same sim state cannot blend
//! differently no matter what order their steps ran in, and that is a property a
//! test asserts rather than a comment claiming it.
//!
//! # What the bodies are
//!
//! `inf_physics::ragdoll::build_ragdoll` (P12.1) has been in the tree since Phase
//! 12 with **no runtime consumer at all** — a pure builder that turns a humanoid
//! skeleton into body / collider / joint descriptors. This wave is its first.

use std::collections::BTreeSet;

use glam::{DQuat, DVec3};

use inf_ecs::components::{
    CharacterMovement, Collider3D, ColliderShape3DKind, LandingKind, MovementMode, MovementRefusal,
    Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::movement as model;
use inf_ecs::world::EcsWorld;

use super::ecs::PhysicsBridge3D;
use super::movement::MoveOutcome;
use super::{BodyId3D, ColliderShape3D, JointId3D};
use crate::ragdoll::{build_ragdoll, RagdollBone, RagdollConfig};

/// The machine parameter the bridge sets so a condition tree can pick the
/// supine get-up from the prone one: `1.0` face-up, `0.0` face-down.
pub const PARAM_FACE_UP: &str = "get_up_face_up";
/// The trigger the bridge arms when a ragdoll ends on the ground.
pub const TRIGGER_GET_UP: &str = "get_up";
/// The trigger the bridge arms when one starts.
pub const TRIGGER_RAGDOLL: &str = "ragdoll";

/// How long the physics pose blends out into the get-up, seconds.
pub const GET_UP_BLEND_S: f64 = 0.35;
/// A ragdoll is **settled** once its root body has been slower than this for
/// [`SETTLE_TIME_S`], m/s.
pub const SETTLE_SPEED_MPS: f64 = 0.35;
/// See [`SETTLE_SPEED_MPS`].
pub const SETTLE_TIME_S: f64 = 0.6;

/// **The fastest a ragdoll limb may be travelling**, m/s (island wave I5).
///
/// The magnitude of [`inf_anim::ragdoll::GRAVITY_CUTOFF_MPS`] — the speed past
/// which that module already stops a limb *accelerating*. The two are one number
/// read twice, so a limb in free fall is never clamped by this and a limb the
/// solver is feeding energy into always is.
///
/// It exists because the cutoff bounds an acceleration and nothing bounded the
/// speed. See the loop that applies it for the measurement that made it
/// necessary.
pub const MAX_LIMB_SPEED_MPS: f64 = -inf_anim::ragdoll::GRAVITY_CUTOFF_MPS;

/// How long the bridge waits for the pose step to answer a rig request before
/// concluding that **no rig is coming**, seconds (P29.4 audit, A1).
///
/// The answer normally arrives on the very next fixed step: the request is a
/// value in [`inf_ecs::anim_bridge`] and the pose step runs after the movement
/// step in both hosts. It never arrives at all for a character that has a
/// [`CharacterMovement`] and no skeleton — and that is not a hypothetical. The
/// landing classifier turns a hard enough fall into a ragdoll for *any*
/// character (`movement.rs` step 10), and `movement_parity`'s own traversal
/// fixture is exactly such a character.
///
/// Without a bound, that character stays in [`MovementMode::Ragdoll`] for ever.
/// Nothing spawns, so both exits — the settle check and the player's jump — are
/// unreachable, because both live inside the "the bodies exist" branch; the
/// capsule is written back to the same place every step with no gravity and no
/// input authority. Measured before this constant existed: six hundred steps,
/// ten seconds, never left the mode, and a held jump did not end it either.
///
/// A refusal is a value in this repository. A wait with no end is neither.
pub const RIG_WAIT_S: f64 = 0.25;

/// [`RIG_WAIT_S`], never shorter than three fixed steps — so a host running at
/// ten hertz still gives the pose step its one step of latency and a margin,
/// rather than giving up before the answer could have been written.
fn rig_wait_s(dt: f64) -> f64 {
    if dt.is_finite() && dt > 0.0 {
        RIG_WAIT_S.max(dt * 3.0)
    } else {
        RIG_WAIT_S
    }
}

/// The bodies and joints one ragdolled character owns while it is simulating.
///
/// Held by the [`PhysicsBridge3D`] rather than by the ECS, for the reason
/// `PoseStoreRes` is a resource: these are a property of a play session, they are
/// never serialized, and a stopped session must not leave a skeleton of loose
/// bodies behind.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpawnedRagdoll {
    /// The bodies, in `build_ragdoll`'s parents-first order.
    pub bodies: Vec<BodyId3D>,
    /// Their colliders — kept because the ground probe under the pelvis must be
    /// blind to the ragdoll's own limbs, and `add_collider` hands the ids back
    /// anyway. Asking the facade for "the colliders of this body" would be a new
    /// query with one caller.
    pub colliders: Vec<super::ColliderId3D>,
    /// The joints linking them.
    pub joints: Vec<JointId3D>,
    /// Which body is the pelvis — the one the capsule follows and the one whose
    /// roll decides supine versus prone.
    pub pelvis: Option<BodyId3D>,
    /// The root body whose speed drives the motor and the flail.
    pub root: Option<BodyId3D>,
    /// How long the root has been slower than [`SETTLE_SPEED_MPS`].
    pub settled_s: f64,
}

/// **Spawn** the articulated bodies for `guid` from the rig the pose step
/// published, seeded with the character's velocity.
///
/// The velocity handoff is the whole of "anim → physics": every body starts at
/// the speed the character was moving, so a ragdoll triggered mid-sprint keeps
/// going rather than dropping where it stood.
///
/// `None` when the rig produced no classifiable bones — a refusal as a value, and
/// the caller stays in `Ragdoll` with nothing spawned rather than crashing on a
/// rig that is not a humanoid.
pub fn spawn(
    bridge: &mut PhysicsBridge3D,
    rig: &[inf_ecs::anim_bridge::RigBone],
    velocity: DVec3,
) -> Option<SpawnedRagdoll> {
    let bones: Vec<RagdollBone> = rig
        .iter()
        .map(|b| RagdollBone::new(b.name.clone(), b.head.to_dvec3(), b.tail.to_dvec3()))
        .collect();
    let parts = build_ragdoll(&bones, RagdollConfig::default());
    if parts.is_empty() {
        return None;
    }
    let mut out = SpawnedRagdoll::default();
    let w = bridge.world_mut();
    for part in &parts {
        let body = w.add_body(part.body.kind, part.position, part.rotation);
        if let Some(c) = w.add_collider(body, part.collider.clone()) {
            out.colliders.push(c);
        }
        // **The handoff.** Every limb inherits the character's velocity.
        w.set_body_linvel(body, velocity);
        if part.role == crate::ragdoll::BoneRole::Hips {
            out.pelvis = Some(body);
        }
        if out.root.is_none() {
            out.root = Some(body);
        }
        out.bodies.push(body);
    }
    for (i, part) in parts.iter().enumerate() {
        if let Some(j) = part.joint {
            if let (Some(&parent), Some(&child)) = (out.bodies.get(j.parent), out.bodies.get(i)) {
                if let Some(id) = w.add_joint(parent, child, j.desc) {
                    out.joints.push(id);
                }
            }
        }
    }
    // Absent hips: the first body is the pelvis by construction, because
    // `build_ragdoll` orders parents first.
    if out.pelvis.is_none() {
        out.pelvis = out.bodies.first().copied();
    }
    Some(out)
}

/// **Despawn** them, joints first so no constraint outlives a body it names.
pub fn despawn(bridge: &mut PhysicsBridge3D, spawned: &SpawnedRagdoll) {
    let w = bridge.world_mut();
    for j in &spawned.joints {
        w.remove_joint(*j);
    }
    for b in &spawned.bodies {
        w.remove_body(*b);
    }
}

/// **Begin a ragdoll** on `cm`, in place.
///
/// Sets the mode through the one transition table (a ragdoll is legal from
/// anywhere), seeds the handoff velocity from whatever the character was doing,
/// and *asks* the pose step for a rig — which arrives one fixed step later, which
/// is when the bodies are spawned.
///
/// Returns whether the mode changed; a refusal is a value, and the only way to
/// get one is a mode the table forbids leaving.
pub fn begin(cm: &mut CharacterMovement) -> bool {
    let verdict = model::request_mode(cm.mode, MovementMode::Ragdoll, true, true);
    if verdict.refusal != MovementRefusal::None {
        cm.runtime.refusals = cm.runtime.refusals.saturating_add(1);
        return false;
    }
    cm.mode = MovementMode::Ragdoll;
    cm.runtime.time_in_mode_s = 0.0;
    cm.runtime.ragdoll.phase = inf_anim::RagdollPhase::Simulating;
    cm.runtime.ragdoll.time_in_phase_s = 0.0;
    cm.runtime.ragdoll.spawned = false;
    cm.runtime.ragdoll.settled_hint = false;
    // The velocity the limbs will be seeded with. Taken here rather than at spawn
    // time because by then a step of gravity has run and the number would be the
    // fall's, not the character's.
    cm.runtime.ragdoll.last_velocity = cm.runtime.velocity;
    cm.runtime.mantle.active = false;
    true
}

/// **The gameplay door**: put `guid` into a ragdoll.
///
/// The seam a Blueprint, an AI or a damage system uses. `false` for an entity
/// with no [`CharacterMovement`], or for a mode the table refuses to leave.
pub fn start_ragdoll(world: &mut EcsWorld, guid: uuid::Uuid) -> bool {
    let Some(entity) = world.entity_of(guid) else {
        return false;
    };
    let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(entity) else {
        return false;
    };
    if !begin(&mut cm) {
        return false;
    }
    inf_ecs::anim_bridge::request_ragdoll_rig(world, guid);
    inf_ecs::anim_bridge::set_anim_trigger(world, guid, TRIGGER_RAGDOLL);
    true
}

/// **Advance one ragdolled character a fixed step.**
///
/// In order: spawn if the rig has arrived; drive the motors from the root's own
/// speed; read the pelvis for the face-up flag and the capsule's placement; probe
/// the ground under it; and decide whether it is time to get up.
///
/// Returns the outcome the movement step reports. Every write to the world goes
/// through this one function, so a ragdoll cannot be half-applied.
pub fn step_ragdoll(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    mut cm: CharacterMovement,
    dt: f64,
) -> Option<MoveOutcome> {
    let entity = world.entity_of(guid)?;
    cm.runtime.time_in_mode_s += dt;
    cm.runtime.ragdoll.time_in_phase_s += dt;

    // ── 1. Spawn, once the rig has arrived (one fixed step after the request).
    if !cm.runtime.ragdoll.spawned {
        if let Some(rig) = inf_ecs::anim_bridge::take_ragdoll_rig(world, guid) {
            let seed = cm.runtime.ragdoll.last_velocity.to_dvec3();
            if let Some(spawned) = spawn(bridge, &rig, seed) {
                // **The capsule's collision goes off** (ALS `RagdollStart`, port
                // map §4.1): the character's kinematic capsule and the limbs it
                // just spawned occupy the same space, and a kinematic body
                // resolving that overlap shoves the ragdoll across the level
                // instead of letting it fall. Measured before this line existed:
                // the limbs never came to rest, so the get-up never fired.
                if let Some(c) = bridge.collider_of(guid) {
                    bridge.world_mut().set_collider_enabled(c, false);
                }
                bridge.set_ragdoll(guid, spawned);
                cm.runtime.ragdoll.spawned = true;
            } else {
                // A rig with no classifiable bone is not a ragdoll. Rather than
                // stand there for ever, hand the character straight back — the
                // refusal is visible as a `Ragdoll` mode that lasted one step.
                return finish(world, bridge, guid, cm, false);
            }
        } else if cm.runtime.press_jump || cm.runtime.ragdoll.time_in_phase_s > rig_wait_s(dt) {
            // **No rig is coming** (P29.4 audit, A1). See [`RIG_WAIT_S`]: the
            // branch above is "the rig arrived and was not a humanoid", and this
            // one is "the rig never arrived", which is what a character with no
            // skeleton gets. Both hand the character straight back, and this one
            // reads its own last ground answer to pick which of the two exits
            // §13's row names — a character that cannot be a ragdoll is not a
            // character that stops moving.
            cm.runtime.press_jump = false;
            let grounded = cm.runtime.grounded;
            return finish(world, bridge, guid, cm, grounded);
        }
    }

    // ── 2. Drive, read and follow.
    let mut position = {
        let w = world.world();
        w.get::<Transform>(entity)?.translation.to_dvec3()
    };
    let radius = {
        let w = world.world();
        match w.get::<Collider3D>(entity) {
            Some(c) if c.shape_kind == ColliderShape3DKind::Capsule => c.radius,
            _ => 0.3,
        }
    };
    let half = cm.half_height_for(MovementMode::Grounded);
    let mut on_ground = false;
    if let Some(spawned) = bridge.ragdoll_of(guid).cloned() {
        let root_vel = spawned
            .root
            .and_then(|b| bridge.world().body_linvel(b))
            .unwrap_or(DVec3::ZERO);
        cm.runtime.ragdoll.last_velocity = Vec3d::from_dvec3(root_vel);
        // **The velocity-scaled drive**: a fast ragdoll is stiffer and holds its
        // animated pose more; a slow one goes limp. The stiffness is ALS's number
        // and the *applied* unit is angular damping — see
        // `inf_anim::ragdoll::angular_damping` for why this facade has no motor
        // to give a stiffness to, and why the constant stays load-bearing anyway.
        let speed = root_vel.length();
        cm.runtime.ragdoll.motor_stiffness = inf_anim::motor_stiffness(speed);
        let damping = inf_anim::ragdoll::angular_damping(speed);
        for b in &spawned.bodies {
            bridge.world_mut().set_body_damping(*b, 0.0, damping);
        }
        // **Gravity cutoff**: past 40 m/s downward the ragdoll stops accelerating,
        // which is ALS's answer to a body that would otherwise tunnel the floor.
        let gravity_on = inf_anim::ragdoll::gravity_enabled(root_vel.y);
        for b in &spawned.bodies {
            bridge
                .world_mut()
                .set_body_gravity_scale(*b, if gravity_on { 1.0 } else { 0.0 });
        }
        // **…AND A CEILING ON THE SPEED ITSELF** (island wave I5).
        //
        // The cutoff above bounds an *acceleration*; nothing bounded the
        // velocity. An articulated body whose joints are seeded in a pose that
        // violates their limits does not settle — the solver feeds energy into
        // it every step and the numbers leave the world. Measured on the
        // phase-29 course: the committed run settles in 46 steps, and the same
        // ragdoll entered from **2.7 cm further along the same fall**, at a
        // bit-identical handoff velocity of `(0, -10.706, 3.750)`, reaches
        // `z = -3.85e13`. The jitter is in **both** — the pelvis moves a metre a
        // step from the first step of the committed run too — so the 46-step
        // settle was luck rather than stability.
        //
        // This is a **bound, not a cure**: the instability is upstream in the
        // joint seeding and is carried by name in the wave's ledger. What the
        // bound buys is that a simulation which cannot be trusted to settle can
        // still be trusted to stay in the level, which is the difference between
        // a ragdoll that looks wrong and a world with no finite state.
        for b in &spawned.bodies {
            let Some(v) = bridge.world().body_linvel(*b) else {
                continue;
            };
            let speed = v.length();
            if !speed.is_finite() {
                bridge.world_mut().set_body_linvel(*b, DVec3::ZERO);
            } else if speed > MAX_LIMB_SPEED_MPS {
                bridge
                    .world_mut()
                    .set_body_linvel(*b, v * (MAX_LIMB_SPEED_MPS / speed));
            }
        }
        // **The pelvis decides which way up the character is.**
        if let Some(p) = spawned.pelvis {
            let (t, r) = (
                bridge.world().body_translation(p),
                bridge.world().body_rotation(p),
            );
            if let (Some(t), Some(r)) = (t, r) {
                cm.runtime.ragdoll.pelvis = Vec3d::from_dvec3(t);
                let roll = inf_math::proll(r.as_quat()).to_degrees() as f64;
                cm.runtime.ragdoll.face_up = inf_anim::face_up_from_pelvis_roll(roll, false);
                let yaw = inf_math::pyaw(r.as_quat()).to_degrees() as f64;
                cm.runtime.ragdoll.pelvis_yaw_deg =
                    inf_anim::ragdoll::body_yaw_from_pelvis(yaw, cm.runtime.ragdoll.face_up);
                // The capsule follows the pelvis, lifted onto the floor under it.
                let mut exclude = BTreeSet::new();
                if let Some(c) = bridge.collider_of(guid) {
                    exclude.insert(c);
                }
                // The ragdoll's own limbs are not the ground.
                for c in &spawned.colliders {
                    exclude.insert(*c);
                }
                // **The probe starts ABOVE the pelvis**, and that is not a
                // detail. A settled ragdoll's hips rest a few centimetres off
                // the floor, so a sphere cast from the pelvis itself begins
                // already overlapping the ground — `started_penetrating`, whose
                // witness point parry does not stand behind. Measured before the
                // lift: a ragdoll at rest with a velocity of 0.01 m/s reported
                // `on_ground == false` for nine hundred steps and never got up.
                let probe_r = radius * 0.5;
                let lift = probe_r + 0.05;
                let drop = half + radius + 0.1;
                let hit = bridge.world_mut().cast_shape(
                    &ColliderShape3D::Sphere { radius: probe_r },
                    t + DVec3::Y * lift,
                    DQuat::IDENTITY,
                    -DVec3::Y,
                    lift + drop,
                    &exclude,
                );
                // Anything under the pelvis is ground, including something the
                // probe started inside: a pelvis in the floor is on it.
                on_ground = hit.is_some();
                let feet_y = match hit {
                    Some(h) if !h.started_penetrating => h.point.y,
                    // **The probe began inside something** (island wave I5). The
                    // line above this one says a pelvis in the floor is on it —
                    // and then the placement put the capsule's FEET a whole body
                    // below the pelvis, which on a pelvis already in the floor is
                    // a whole body below the floor. The collider is switched back
                    // on down there when the ragdoll ends and the character falls
                    // out of the world: measured on the phase-29 course at
                    // **y = -132 m and still falling**, from hips that ended
                    // 0.8 m under the ground.
                    //
                    // A pelvis at a surface has its feet AT that surface. This
                    // only ever RAISES the placement — the same direction, and
                    // the same argument, as `settle_on_spawn`'s: the wrong way
                    // round is the one that drops a body through the world and
                    // keeps it falling.
                    Some(_) => t.y,
                    None => t.y - (half + radius),
                };
                position = DVec3::new(t.x, feet_y + half + radius, t.z);
            }
        }
        // Settling: the root has to be slow for a while, not just for one step,
        // because a ragdoll at the top of its arc is momentarily slow too.
        let mut spawned = spawned;
        if root_vel.length() < SETTLE_SPEED_MPS && on_ground {
            spawned.settled_s += dt;
        } else {
            spawned.settled_s = 0.0;
        }
        let settled = spawned.settled_s >= SETTLE_TIME_S;
        cm.runtime.ragdoll.settled_hint = settled;
        bridge.set_ragdoll(guid, spawned);
        cm.runtime.ragdoll.on_ground = on_ground;
        cm.runtime.body_yaw_deg = cm.runtime.ragdoll.pelvis_yaw_deg;
        cm.runtime.target_yaw_deg = cm.runtime.body_yaw_deg;

        // ── 3. Get up, when it has settled or when the player asks.
        if settled || cm.runtime.press_jump {
            cm.runtime.press_jump = false;
            return finish(world, bridge, guid, cm, on_ground);
        }
    }

    write_back(world, bridge, guid, entity, &cm, position, half);
    let mode = cm.mode;
    Some(MoveOutcome {
        guid,
        mode,
        refusal: MovementRefusal::None,
        grounded: on_ground,
        landed: LandingKind::None,
    })
}

/// End the ragdoll: despawn the bodies and branch — **grounded gets up, airborne
/// resumes falling with the ragdoll's last velocity** (§13's catalogue row).
fn finish(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    mut cm: CharacterMovement,
    on_ground: bool,
) -> Option<MoveOutcome> {
    let entity = world.entity_of(guid)?;
    if let Some(spawned) = bridge.take_ragdoll(guid) {
        despawn(bridge, &spawned);
    }
    // …and back on, by the same hand that turned it off.
    if let Some(c) = bridge.collider_of(guid) {
        bridge.world_mut().set_collider_enabled(c, true);
    }
    cm.runtime.ragdoll.spawned = false;
    let position = {
        let w = world.world();
        w.get::<Transform>(entity)?.translation.to_dvec3()
    };
    let half = cm.half_height_for(MovementMode::Grounded);
    let to = if on_ground {
        MovementMode::Grounded
    } else {
        MovementMode::FallFree
    };
    let verdict = model::request_mode(cm.mode, to, true, true);
    cm.mode = verdict.mode;
    cm.runtime.time_in_mode_s = 0.0;
    if on_ground {
        // **The get-up.** The physics pose blends out over `GET_UP_BLEND_S`, and
        // which get-up plays is a machine decision driven by a parameter — this
        // side never picks a state.
        cm.runtime.ragdoll.phase = inf_anim::RagdollPhase::GettingUp;
        cm.runtime.ragdoll.time_in_phase_s = 0.0;
        cm.runtime.velocity = Vec3d::ZERO;
        cm.runtime.grounded = true;
    } else {
        // **The velocity handoff, outbound.** ALS's `CharacterMovement->Velocity
        // = LastRagdollVelocity`, verbatim: a ragdoll that runs out mid-air
        // resumes the fall it was already in.
        cm.runtime.ragdoll.phase = inf_anim::RagdollPhase::Inactive;
        cm.runtime.ragdoll.time_in_phase_s = 0.0;
        cm.runtime.velocity = cm.runtime.ragdoll.last_velocity;
        cm.runtime.grounded = false;
    }
    let face_up = cm.runtime.ragdoll.face_up;
    let mode = cm.mode;
    write_back(world, bridge, guid, entity, &cm, position, half);
    // The two doors the `anim.*` kit uses, and nothing else: a parameter that
    // says which way up, and a trigger that says now.
    if on_ground {
        let kind = if face_up {
            inf_anim::GetUp::Supine
        } else {
            inf_anim::GetUp::Prone
        };
        inf_ecs::anim_bridge::set_anim_param(world, guid, PARAM_FACE_UP, kind.param_value());
        inf_ecs::anim_bridge::set_anim_trigger(world, guid, TRIGGER_GET_UP);
        // **Pose matching picks the entry frame** — P29.2 built the primitive and
        // named this consumer as the one that turns it on. It is turned off again
        // when the get-up finishes, because a machine that entered every state at
        // its best-matching frame would never play a state's beginning.
        inf_ecs::anim_bridge::set_pose_match_entry(world, guid, true);
    }
    Some(MoveOutcome {
        guid,
        mode,
        refusal: MovementRefusal::None,
        grounded: on_ground,
        landed: LandingKind::None,
    })
}

/// Write the character's transform, capsule and component back.
fn write_back(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    entity: inf_ecs::Entity,
    cm: &CharacterMovement,
    position: DVec3,
    half: f64,
) {
    let body_yaw = cm.runtime.body_yaw_deg;
    {
        let w = world.world_mut();
        if let Some(mut t) = w.get_mut::<Transform>(entity) {
            t.translation.x = position.x;
            t.translation.y = position.y;
            t.translation.z = position.z;
            t.rotation.y = body_yaw;
        }
        if let Some(mut c) = w.get_mut::<Collider3D>(entity) {
            if c.shape_kind == ColliderShape3DKind::Capsule {
                c.half_extents.y = half;
            }
        }
        if let Some(mut slot) = w.get_mut::<CharacterMovement>(entity) {
            *slot = cm.clone();
        }
    }
    if let Some(body) = bridge.body_of(guid) {
        bridge.world_mut().set_body_translation(body, position);
    }
    world.mark_dirty();
}

/// **Advance the get-up blend**, for a character that is no longer ragdolling but
/// is still blending out of the pose the bodies left behind.
///
/// Called from the ordinary movement step, so a getting-up character walks,
/// turns and falls like any other — the blend is a *weight*, not a mode.
pub fn tick_get_up(world: &mut EcsWorld, guid: uuid::Uuid, cm: &mut CharacterMovement, dt: f64) {
    if cm.runtime.ragdoll.phase != inf_anim::RagdollPhase::GettingUp {
        return;
    }
    cm.runtime.ragdoll.time_in_phase_s += dt;
    if cm.runtime.ragdoll.time_in_phase_s >= GET_UP_BLEND_S {
        cm.runtime.ragdoll.phase = inf_anim::RagdollPhase::Inactive;
        cm.runtime.ragdoll.time_in_phase_s = 0.0;
        inf_ecs::anim_bridge::set_pose_match_entry(world, guid, false);
    }
}

/// **The blend weight**, as the renderer and the pose step would read it: `1` is
/// the physics pose, `0` is the machine's.
///
/// A one-line wrapper over [`inf_anim::ragdoll::blend_weight`] so a caller reads
/// the character's own runtime rather than assembling the arguments itself — and
/// so the purity claim has exactly one call site to be true of.
pub fn blend_weight(cm: &CharacterMovement) -> f32 {
    inf_anim::ragdoll_blend_weight(
        cm.runtime.ragdoll.phase,
        cm.runtime.ragdoll.time_in_phase_s,
        GET_UP_BLEND_S,
    )
}
