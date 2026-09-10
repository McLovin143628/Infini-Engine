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

/// **How far above its own ballistic path a ragdoll's centre of mass may be
/// while nothing is holding it up**, metres (wave WPN2d audit).
///
/// The third bound in this family, and the one the first two could not see.
/// [`MAX_LIMB_SPEED_MPS`] bounds a *speed* and
/// [`inf_anim::ragdoll::GRAVITY_CUTOFF_MPS`] bounds an *acceleration*; neither
/// notices an assembly whose limbs are all slow and all drifting the same way.
/// Measured on the island: a corpse left ragdolling for 295.6 s rose
/// **83.2 m at a near-constant 0.281 m/s**, with the finite differences a
/// ±1 m/s random walk about that mean — a body in free flight cannot do that,
/// because free flight is `y'' = -g` and nothing else.
///
/// The slack is what a solver's own noise is allowed to be. It is deliberately
/// larger than a step's worth of drift and far smaller than the ground probe's
/// reach, so a corpse that creeps upward off the floor is pulled back the
/// moment the probe under its pelvis loses the surface.
pub const FREE_FLIGHT_SLACK_M: f64 = 0.25;

/// **The ballistic reference a ragdoll's centre of mass is held to** while the
/// probe under its pelvis finds nothing to stand on (wave WPN2d audit).
///
/// It is re-seeded from the assembly itself on every step that *does* find
/// ground, so a contact may lift a body as hard as it likes; it is integrated
/// as `y'' = -g` on every step that does not. The bound is on the CENTRE OF
/// MASS, and the correction is subtracted uniformly from every body, so the
/// relative motion of the limbs — the flail — is left exactly as the solver
/// computed it. Only the drift of the whole is removed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FreeFlight {
    /// Where the reference centre of mass is, metres.
    pub y: f64,
    /// How fast it is rising, m/s.
    pub vy: f64,
}

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
    /// **The ballistic path the centre of mass is held to while it is airborne**
    /// (wave WPN2d audit) — `None` until the first step that has to bound it.
    /// See [`FreeFlight`] and [`FREE_FLIGHT_SLACK_M`].
    pub free_flight: Option<FreeFlight>,
    /// **Which joint each body IS, and how to read its rotation as that joint's**
    /// (wave CHAR1b.2), parallel to [`bodies`](Self::bodies).
    ///
    /// A capsule is built from the bone's head→tail segment, so its spawn
    /// orientation has no twist about that axis and is *not* the joint's
    /// rotation. The difference is a constant rigid offset — the two frames are
    /// glued to the same bone — so it is taken once, at spawn, and every later
    /// step recovers the joint's world rotation as `body_rotation · offset`.
    ///
    /// The head is recovered the same way: the bone's head in the body's own
    /// local frame, taken at spawn, is where the joint sits on that body for
    /// ever.
    pub joints_of_body: Vec<RagdollJointRef>,
}

/// **What one ragdoll body is, in the skeleton's terms** (wave CHAR1b.2).
///
/// The pose blend's whole content: the physics side owns bodies and the
/// animation side owns joints, and this is the constant that turns one into the
/// other for as long as the ragdoll lives.
#[derive(Clone, Debug, PartialEq)]
pub struct RagdollJointRef {
    /// The joint's name, as [`inf_ecs::anim_bridge::RigBone::name`] carried it.
    pub name: String,
    /// `body_rotation⁻¹ · joint_rotation`, taken at spawn — constant, because
    /// both frames are glued to the same rigid bone.
    pub offset: DQuat,
    /// The bone's head in the body's own local frame, taken at spawn — constant
    /// for the same reason.
    pub head_local: DVec3,
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
        .map(|b| {
            // The role and the parent ride across (SK1a) — `build_ragdoll` picks
            // its own path from whether any bone carries one, so a rig with no
            // table reaches exactly the classifier it always reached.
            RagdollBone::new(b.name.clone(), b.head.to_dvec3(), b.tail.to_dvec3())
                .with_role(b.parent, b.role)
        })
        .collect();
    let parts = build_ragdoll(&bones, RagdollConfig::default());
    if parts.is_empty() {
        return None;
    }
    // The rig by name, so each part can find the joint it was built from — the
    // one place the two vocabularies are matched up, and the reason
    // `RagdollJointRef` is a constant rather than a per-step search.
    let rig_by_name: std::collections::BTreeMap<&str, &inf_ecs::anim_bridge::RigBone> =
        rig.iter().map(|b| (b.name.as_str(), b)).collect();
    let mut out = SpawnedRagdoll::default();
    let w = bridge.world_mut();
    for part in &parts {
        let body = w.add_body(part.body.kind, part.position, part.rotation);
        if let Some(c) = w.add_collider(body, part.collider.clone()) {
            out.colliders.push(c);
        }
        // **The capsule→joint offset, taken once.** See `RagdollJointRef`.
        if let Some(b) = rig_by_name.get(part.name.as_str()) {
            let inv = part.rotation.inverse();
            out.joints_of_body.push(RagdollJointRef {
                name: part.name.clone(),
                offset: (inv * b.rot).normalize(),
                head_local: inv * (b.head.to_dvec3() - part.position),
            });
        } else {
            // A part whose bone is not in the rig by name cannot be read back as
            // a joint. An identity offset would silently draw the capsule's own
            // frame onto some joint; a refusal draws the machine's pose there,
            // which is what every joint without a body already gets.
            out.joints_of_body.push(RagdollJointRef {
                name: String::new(),
                offset: DQuat::IDENTITY,
                head_local: DVec3::ZERO,
            });
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
    overlays: &model::OverlayRegistry,
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
                return finish(world, bridge, guid, cm, false, overlays);
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
            return finish(world, bridge, guid, cm, grounded, overlays);
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
    if let Some(mut spawned) = bridge.ragdoll_of(guid).cloned() {
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
        // **...AND A CEILING ON THE ALTITUDE THE WHOLE ASSEMBLY MAY GAIN** (wave
        // WPN2d audit).
        //
        // The two bounds above are both bounds on ONE BODY: an acceleration
        // (`gravity_enabled`) and a speed (`MAX_LIMB_SPEED_MPS`). Neither can
        // see fifteen slow limbs all drifting the same way, and that is what
        // the island produced. Wave WPN2d's own session 3 photographed seven
        // frames captioned "the mesh in the hero's hands" of a corpse **33 to
        // 48 m above the street**: the hero blew itself up with its own
        // launcher at 2.2 m, entered `Ragdoll`, and rose **83.2 m over 295.6 s
        // at a near-constant 0.281 m/s** -- a straight line, not a parabola,
        // with a metre a second of jitter about it. Every frame the wave took
        // after that was of the sky.
        //
        // The source is upstream and is carried by name since island wave I5:
        // an articulated body whose joints are seeded violating their limits
        // has energy fed into it every step, and `MAX_LIMB_SPEED_MPS` -- which
        // rescales a limb's velocity vector -- destroys momentum
        // asymmetrically when it bites, so the residual is a systematic drift
        // rather than a random walk. **This is a bound, not a cure**, exactly
        // as that one is.
        //
        // What it bounds is the one thing that is not a matter of taste: a
        // body with nothing under it is in FREE FLIGHT, and free flight is
        // `y'' = -g`. The reference path is re-seeded from the assembly itself
        // on every step the pelvis probe finds ground -- so a contact, an
        // explosion or a car may throw a corpse as hard as it likes -- and
        // integrated as a parabola on every step it does not. The excess is
        // taken off the CENTRE OF MASS and subtracted uniformly, so the limbs'
        // motion relative to one another, which is the flail, is left exactly
        // as the solver computed it.
        {
            let mut mass = 0.0f64;
            let mut mom_y = 0.0f64;
            let mut mass_y = 0.0f64;
            for b in &spawned.bodies {
                let m = bridge.world().body_mass(*b).unwrap_or(0.0);
                if !(m.is_finite() && m > 0.0) {
                    continue;
                }
                let v = bridge.world().body_linvel(*b).unwrap_or(DVec3::ZERO);
                let t = bridge.world().body_translation(*b).unwrap_or(DVec3::ZERO);
                if !(v.y.is_finite() && t.y.is_finite()) {
                    continue;
                }
                mass += m;
                mom_y += m * v.y;
                mass_y += m * t.y;
            }
            if mass > 0.0 {
                let com_y = mass_y / mass;
                let com_vy = mom_y / mass;
                // The probe under the pelvis has not run yet this step, so this
                // is the last answer it gave -- which is the honest one: what
                // the assembly was standing on when it was last asked.
                let grounded = cm.runtime.ragdoll.on_ground;
                match spawned.free_flight {
                    Some(mut r) if !grounded => {
                        let g = bridge.world().gravity().y;
                        // 1. The velocity: free flight allows exactly `g*dt` of
                        //    change and no more. A centre of mass rising faster
                        //    than that is being pushed by nothing.
                        let want_vy = r.vy + g * dt;
                        let mut com_vy = com_vy;
                        if com_vy > want_vy {
                            let excess = com_vy - want_vy;
                            for b in &spawned.bodies {
                                let Some(v) = bridge.world().body_linvel(*b) else {
                                    continue;
                                };
                                bridge
                                    .world_mut()
                                    .set_body_linvel(*b, DVec3::new(v.x, v.y - excess, v.z));
                            }
                            com_vy = want_vy;
                        }
                        // A centre of mass falling FASTER than the reference is
                        // fine -- it hit something, or the speed clamp took a
                        // bite -- and the reference follows it down, so the
                        // difference is never banked into a later rise.
                        r.vy = com_vy;
                        // 2. The position, because a solver that corrects
                        //    penetration moves bodies without moving their
                        //    velocities: a purely positional drift would pass
                        //    the check above and still leave the level.
                        let want_y = r.y + r.vy * dt;
                        if com_y > want_y + FREE_FLIGHT_SLACK_M {
                            let excess = com_y - (want_y + FREE_FLIGHT_SLACK_M);
                            for b in &spawned.bodies {
                                let Some(t) = bridge.world().body_translation(*b) else {
                                    continue;
                                };
                                bridge
                                    .world_mut()
                                    .set_body_translation(*b, DVec3::new(t.x, t.y - excess, t.z));
                            }
                            r.y = want_y + FREE_FLIGHT_SLACK_M;
                        } else {
                            r.y = com_y.min(want_y + FREE_FLIGHT_SLACK_M);
                        }
                        spawned.free_flight = Some(r);
                    }
                    // On the ground, or on the first step this has ever run for
                    // this ragdoll: the reference IS the assembly. Whatever a
                    // contact just did to it is legitimate by definition.
                    _ => {
                        spawned.free_flight = Some(FreeFlight {
                            y: com_y,
                            vy: com_vy,
                        });
                    }
                }
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
        if root_vel.length() < SETTLE_SPEED_MPS && on_ground {
            spawned.settled_s += dt;
        } else {
            spawned.settled_s = 0.0;
        }
        let settled = spawned.settled_s >= SETTLE_TIME_S;
        cm.runtime.ragdoll.settled_hint = settled;
        // **The pose crosses back** (wave CHAR1b.2). Every step the bodies own
        // the pose, they say so in the skeleton's own vocabulary, and the pose
        // step blends toward it at `blend_weight`'s number — which until this
        // wave had no consumer at all, so a ragdolling character drew a flail
        // clip while its bodies fell somewhere else entirely.
        publish_pose(world, bridge, guid, &cm, &spawned);
        bridge.set_ragdoll(guid, spawned);
        cm.runtime.ragdoll.on_ground = on_ground;
        cm.runtime.body_yaw_deg = cm.runtime.ragdoll.pelvis_yaw_deg;
        cm.runtime.target_yaw_deg = cm.runtime.body_yaw_deg;

        // ── 3. Get up, when it has settled or when the player asks — **unless
        //    the character is dead** (island wave I6).
        //
        //    A corpse that settles gets up, and a damage system that hands a
        //    dead body to the ragdoll hands it over again on the next step. That
        //    is not a report artefact: it is a body twitching upright and
        //    flopping down, for ever, at the settle interval. Measured on the
        //    first weapons fixture that killed something: **two handoffs in
        //    thirty steps** where there should be one.
        //
        //    `Health` is the only thing in this engine that means "this body has
        //    stopped working", and it is a runtime component, so a character
        //    with no health component is unaffected — which is every character
        //    committed before this wave.
        let dead = world
            .world()
            .get::<inf_ecs::weapon::Health>(entity)
            .is_some_and(|h| h.dead);
        if !dead && (settled || cm.runtime.press_jump) {
            cm.runtime.press_jump = false;
            return finish(world, bridge, guid, cm, on_ground, overlays);
        }
    }

    write_back(world, bridge, guid, entity, &cm, position, half, overlays);
    let mode = cm.mode;
    Some(MoveOutcome {
        guid,
        mode,
        refusal: MovementRefusal::None,
        grounded: on_ground,
        landed: LandingKind::None,
        cover_sweeps: 0,
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
    overlays: &model::OverlayRegistry,
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
    // **Did it actually go down?** (wave CHAR1b.2). The pelvis's world height
    // above the character's own feet, against half a standing capsule: a body
    // lying on the ground carries its pelvis about a foot up, a standing one
    // about a metre. `RagdollRuntime::pelvis` has recorded the position since
    // P29.4 and nothing read it.
    let feet_y = position.y - half - super::movement::FALLBACK_RADIUS_M;
    cm.runtime.ragdoll.upright = cm.runtime.ragdoll.pelvis.y - feet_y > half * 0.5;
    let upright = cm.runtime.ragdoll.upright;
    let mode = cm.mode;
    write_back(world, bridge, guid, entity, &cm, position, half, overlays);
    // **The pose the bodies left.** The bones published on the last simulating
    // step stay exactly as they are and only the weight moves from here: `1` at
    // the start of a get-up, `0` for a ragdoll that ended in the air — and a
    // zero removes the entry, so a character that resumes its fall draws the
    // machine's pose the very next step.
    inf_ecs::anim_bridge::set_ragdoll_pose(world, guid, blend_weight(&cm), Vec::new());
    // The two doors the `anim.*` kit uses, and nothing else: a parameter that
    // says which way up, and a trigger that says now.
    if on_ground {
        let kind = if face_up {
            inf_anim::GetUp::Supine
        } else {
            inf_anim::GetUp::Prone
        };
        inf_ecs::anim_bridge::set_anim_param(world, guid, PARAM_FACE_UP, kind.param_value());
        // …and the THREE-valued one the get-up edges compare (wave CHAR1b.2).
        // `face_up` stays exactly as it was; this is a second fact, not a
        // replacement, because "on its back" and "never went down" are different
        // questions and the second has an authored clip now.
        inf_ecs::anim_bridge::set_anim_param(
            world,
            guid,
            inf_anim::als::GETUP_VAR,
            if upright {
                inf_anim::als::GETUP_STANDING
            } else if face_up {
                inf_anim::als::GETUP_BACK
            } else {
                inf_anim::als::GETUP_FRONT
            },
        );
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
        cover_sweeps: 0,
    })
}

/// Write the character's transform, capsule and component back.
///
/// `#[allow(clippy::too_many_arguments)]` for this file's own reason: every one
/// of the eight is a distinct authority this function needs and none can be
/// derived from another — the two worlds, the identity in each, the state being
/// written, and the two numbers the placement is made of. Bundling them into a
/// struct would name a type whose only purpose is to satisfy a count.
#[allow(clippy::too_many_arguments)]
fn write_back(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    guid: uuid::Uuid,
    entity: inf_ecs::Entity,
    cm: &CharacterMovement,
    position: DVec3,
    half: f64,
    overlays: &model::OverlayRegistry,
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
    // ── **THE MODE REACHES THE MACHINE** (wave CHAR1b.2) ─────────────────────
    //
    // Every other movement path publishes the character's state into its
    // machine's parameters at its write-back — the standing step, the mantle,
    // the seat, flight — and this one never did. `mode` therefore stayed frozen
    // at whatever the last GROUNDED step had published for the whole life of a
    // ragdoll, and the consequences were both invisible and total: measured on
    // the island, a hero put into `MovementMode::Ragdoll` sat in the `idle`
    // state for all 88 steps of its ragdoll, so `ALS_Flail` never played, the
    // `ragdoll → getup_*` edges could not fire because the machine was never in
    // `ragdoll` to leave it, and the get-up blended out into a standing idle
    // instead of a get-up clip.
    //
    // The three get-up states, the flail and the whole `getup` parameter this
    // wave added were unreachable in the running game for that one reason.
    let overlay = overlays.id_of(&cm.overlay);
    inf_ecs::anim_bridge::publish_character_params(world, guid, cm, overlay);
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
    release_get_up_hold(world, guid);
    if cm.runtime.ragdoll.phase != inf_anim::RagdollPhase::GettingUp {
        return;
    }
    cm.runtime.ragdoll.time_in_phase_s += dt;
    if cm.runtime.ragdoll.time_in_phase_s >= GET_UP_BLEND_S {
        cm.runtime.ragdoll.phase = inf_anim::RagdollPhase::Inactive;
        cm.runtime.ragdoll.time_in_phase_s = 0.0;
        inf_ecs::anim_bridge::set_pose_match_entry(world, guid, false);
    }
    // **The blend out** (wave CHAR1b.2). The bodies are gone by now, so the
    // bones stay exactly as the last simulating step left them and only the
    // weight moves — the get-up eases out of the heap the ragdoll really made.
    // A weight of zero removes the entry, which is how a level stops paying for
    // a ragdoll that has finished.
    inf_ecs::anim_bridge::set_ragdoll_pose(world, guid, blend_weight(cm), Vec::new());
}

/// **Put `getup` back to `GETUP_NONE` once the get-up has played out** (wave
/// CHAR1b.2).
///
/// The parameter does two jobs: it says *which* of the three get-ups to enter,
/// and — since this wave — it is what stops the `Any → idle` edge cutting that
/// get-up short one step after it starts. The second job means the parameter has
/// to be *released*, or a character that has ragdolled once has the `Any → idle`
/// safety net wedged shut for the rest of the level.
///
/// The release rule is the machine's own answer rather than a second clock: hold
/// while the machine is in `ragdoll` or in one of the three
/// [`inf_anim::als::GETUP_STATES`], release the step it is anywhere else. So the
/// hold lasts exactly as long as the clip does — a 0.9 exit time on
/// `getup_* → idle` ends it — and a machine with no get-up states at all
/// releases on the first step, which is the pre-CHAR1b.2 behaviour.
///
/// Called from the ordinary movement step every step, before the phase clock,
/// because the phase is `Inactive` again long before the get-up *clip* is over
/// (`GET_UP_BLEND_S` is 0.35 s and the get-ups are seconds long).
fn release_get_up_hold(world: &mut EcsWorld, guid: uuid::Uuid) {
    let Some(v) = inf_ecs::anim_bridge::anim_param(world, guid, inf_anim::als::GETUP_VAR) else {
        return;
    };
    if v < inf_anim::als::GETUP_FRONT {
        return;
    }
    let holding = inf_ecs::anim_bridge::anim_state(world, guid).is_some_and(|s| {
        s.name == "ragdoll" || inf_anim::als::GETUP_STATES.contains(&s.name.as_str())
    });
    if !holding {
        inf_ecs::anim_bridge::set_anim_param(
            world,
            guid,
            inf_anim::als::GETUP_VAR,
            inf_anim::als::GETUP_NONE,
        );
    }
}

/// **Publish the articulated bodies as a pose**, in the skeleton's vocabulary
/// (wave CHAR1b.2, clause 6).
///
/// Each body's world rotation is turned back into its joint's world rotation
/// through the constant offset taken at spawn ([`RagdollJointRef`]), and its
/// head is the same constant carried through the body's transform. The weight is
/// [`blend_weight`]'s, so the number the pose step draws with is a pure function
/// of `(phase, clock)` and the doctrine's sentence has exactly one call site.
///
/// A body whose joint could not be named at spawn is skipped rather than
/// published under an identity offset: those joints keep the machine's pose,
/// which is what every joint with no body already gets.
fn publish_pose(
    world: &mut EcsWorld,
    bridge: &PhysicsBridge3D,
    guid: uuid::Uuid,
    cm: &CharacterMovement,
    spawned: &SpawnedRagdoll,
) {
    let mut bones = Vec::with_capacity(spawned.bodies.len());
    for (body, jref) in spawned.bodies.iter().zip(spawned.joints_of_body.iter()) {
        if jref.name.is_empty() {
            continue;
        }
        let (Some(t), Some(r)) = (
            bridge.world().body_translation(*body),
            bridge.world().body_rotation(*body),
        ) else {
            continue;
        };
        let head = t + r * jref.head_local;
        bones.push(inf_ecs::anim_bridge::RagdollBonePose {
            name: jref.name.clone(),
            rot: (r * jref.offset).normalize(),
            head: inf_ecs::Vec3d::new(head.x, head.y, head.z),
        });
    }
    if bones.is_empty() {
        return;
    }
    inf_ecs::anim_bridge::set_ragdoll_pose(world, guid, blend_weight(cm), bones);
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
