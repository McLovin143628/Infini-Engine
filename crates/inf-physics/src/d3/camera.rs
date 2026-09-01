//! **The locomotion camera's fixed step** (P29.6) — the one door both hosts
//! call, and `cast_shape`'s third consumer.
//!
//! The pure half is [`inf_ecs::camera`]; this is the half that needs a world.
//! Two things happen here and nowhere else:
//!
//! 1. The **subject is read**. A camera follows a character, and everything it
//!    follows — the pivot, the aim, the gait, the mode — is *read* off that
//!    character's [`CharacterMovement`] runtime. Nothing is written back, which
//!    is Ruling 4 kept literally: `ViewMode` never crosses the sim wire, and
//!    there is no camera → sim path at all. The aim the camera chases is the one
//!    the movement step integrated from the look axes, through the one movement
//!    door.
//! 2. The **sweep**. A sphere is cast from the pivot to the desired camera
//!    position; on a blocking hit the camera comes in to the contact. This is
//!    Ruling 3's third consumer of `cast_shape` (after the crouch clearance probe
//!    and the mantle's ledge probes), and it is what stops a third-person camera
//!    ending up inside a wall.
//!
//! # Why the pivot is derived and not authored
//!
//! ALS's pivot is the midpoint of the rig's `head` and `root` sockets. Ours is
//! the character's own capsule — feet plus
//! [`CameraTuning::pivot_height_ratio`](inf_ecs::camera::CameraTuning::pivot_height_ratio)
//! of its standing height — for two reasons. It needs no rig, so a character
//! whose skeleton has not resolved still has a camera; and it scales, so a 1.2 m
//! character gets a proportionate camera without a second table. The feet come
//! from the same arithmetic the movement step uses
//! ([`inf_ecs::movement::feet_offset_m`]), so the camera and the foot bridge
//! cannot disagree about where the character's own origin is.
//!
//! # This is not a system, and it is outside everything the trace folds
//!
//! `step_locomotion_camera` is called by each host as the **last statement of
//! its fixed step** — after the movement step, after the solver, after the
//! write-back, after the pose and after audio — once per stepped frame, with the
//! same `dt`. It is not inside `step_character_movement` and it is not a
//! `SimSchedule` system, because the camera is not sim state: folding it in
//! would put a camera's smoothing inside the function whose output is compared
//! byte for byte between PIE and shipping.
//!
//! **Stated precisely, because the first cut of this paragraph said "not in the
//! fixed step" and the code says otherwise** (P29.6 audit). What the ViewMode
//! ruling requires is that no camera value reach `state_bytes`, and what keeps
//! that true is position (last) plus direction (this door takes `&EcsWorld`, not
//! `&mut`, and writes nothing into any component).
//!
//! # The one `&mut` this door does hold, and why it is safe
//!
//! `bridge: &mut PhysicsBridge3D`, because `cast_shape_where` has to
//! `ensure_query_pipeline()` and that brings a BVH in line with the colliders.
//! The tree it produces is a pure function of the colliders and the integration
//! parameters, and every collider or body mutator marks what it touched (island
//! wave I4b replaced the one `query_dirty` flag with the
//! `query_rebuild` / `query_moved` / `query_moved_bodies` marks — same rule, at
//! the granularity of what actually changed), so bringing it up to date early
//! yields the identical structure and the next step's own queries cannot tell.
//! It is the only shared
//! state the camera can touch, which is why
//! `stepping_a_camera_changes_nothing_about_the_simulation` samples the bridge's
//! bodies as well as the world's components — a mutation through this `&mut`
//! survived the arm before it did.

use glam::{DQuat, DVec3};
use uuid::Uuid;

use inf_ecs::camera::{CameraInput, CameraPose, LocomotionCamera};
use inf_ecs::components::{CharacterMovement, Collider3D, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;

use super::{CastTargets, ColliderId3D, ColliderShape3D, PhysicsBridge3D};

/// **Advance the camera one step against `world`.**
///
/// `subject` is the character the camera follows. Answers `None` — and leaves the
/// camera exactly as it was — when that character is not in the world or carries
/// no [`CharacterMovement`]: a camera with nothing to follow is a **value**, not
/// a failure, and holding the last pose is the right answer for the frame a
/// character despawns on.
pub fn step_locomotion_camera(
    world: &EcsWorld,
    bridge: &mut PhysicsBridge3D,
    cam: &mut LocomotionCamera,
    subject: Uuid,
    dt: f64,
) -> Option<CameraPose> {
    let entity = world.entity_of(subject)?;
    let (cm, centre, collider) = {
        let w = world.world();
        let cm = w.get::<CharacterMovement>(entity)?.clone();
        let t = w.get::<Transform>(entity)?;
        (
            cm,
            t.translation.to_dvec3(),
            w.get::<Collider3D>(entity).copied(),
        )
    };

    // The character's own origin — the SAME arithmetic the pose publisher uses,
    // so the camera's pivot and the character's feet cannot disagree.
    let drop = inf_ecs::movement::feet_offset_m(&cm, collider.as_ref());
    let feet = centre - DVec3::Y * drop;
    // Standing height, not the worn one: a crouch must lower the camera by the
    // crouch *offset block*, not by re-deriving the pivot every step, or the view
    // would drop and rise with a slide the way a head-mounted camera does.
    let standing = cm.stand_half_height_m + collider.map(|c| c.radius).unwrap_or(0.0);
    let pivot_target = feet + DVec3::Y * (2.0 * standing * cam.tuning.pivot_height_ratio);

    // **The drive camera's view of the car** (island wave VEH2a). Two of its
    // three numbers are already on the movement runtime, written by
    // `step_driving` itself: `body_yaw_deg` IS the chassis's heading while
    // seated, and `velocity` IS the chassis's own linear velocity. Only the
    // half-length has to be looked up, and it comes off the chassis collider
    // rather than out of a new field — the same "derive it from what the level
    // already carries" rule the whole vehicle rig is built on. Nothing here is
    // new simulation state, which is why the drive camera costs no schema, no
    // trace bytes and no mirror.
    let driving = (cm.mode == inf_ecs::components::MovementMode::Driving
        && cm.runtime.seat.is_seated())
    .then(|| {
        let half_length_m = world
            .entity_of(cm.runtime.seat.vehicle)
            .and_then(|e| world.world().get::<Collider3D>(e))
            .map(|c| c.half_extents.z.abs())
            .unwrap_or(0.0);
        inf_ecs::camera::DrivingView {
            chassis_yaw_deg: cm.runtime.body_yaw_deg,
            velocity: cm.runtime.velocity,
            half_length_m,
        }
    });

    cam.advance(
        &CameraInput {
            pivot_target: Vec3d::from_dvec3(pivot_target),
            aim_yaw_deg: cm.runtime.aim_yaw_deg,
            aim_pitch_deg: cm.runtime.aim_pitch_deg,
            rotation_mode: cm.rotation_mode,
            gait: cm.runtime.actual_gait,
            mode: cm.mode,
            driving,
        },
        dt,
    );

    // ── the sweep ──
    //
    // From the pivot to where the camera wants to be. The character's own
    // collider is excluded — a camera that stopped at the shoulder it is looking
    // over would never leave the character — and `CastTargets::All` is right
    // here where the mantle wants `Fixed`: a crate you can shove is not a ledge,
    // but it is very much something to not see through.
    //
    // `All` includes **sensors**, so this sweep pulls in on a trigger volume
    // (VEH1a audit). That is the surviving half of P29.7's carried bound: the
    // wheel ray took `CastTargets::AllSolid` and this one deliberately did not,
    // because "is a checkpoint something to not see through" is a behaviour
    // question with its own arms and no measurement behind it yet. See
    // `CastTargets::All`'s own doc for the enumeration.
    let mut exclude: std::collections::BTreeSet<ColliderId3D> = std::collections::BTreeSet::new();
    if let Some(c) = bridge.collider_of(subject) {
        exclude.insert(c);
    }
    // **…and the subject's ragdoll limbs** (P29.6 audit, A4). The capsule above
    // is the character's *one* mirrored collider, and a ragdoll DISABLES it —
    // `ragdoll_bridge` turns it off the step the ragdoll starts — while spawning
    // a dozen live limb colliders in its place. Without this the death cam sweeps
    // into the body it is filming: the first limb in the path is a blocking hit
    // at nearly zero distance, so the camera snaps to `min_arm_fraction` and
    // looks out of the character's own chest, for the whole of the ragdoll.
    //
    // The rule and the list are the ground probe's, two hundred lines away in
    // `ragdoll_bridge`: `SpawnedRagdoll::colliders` exists precisely so a query
    // can be blind to the ragdoll's own limbs, and a camera is such a query.
    if let Some(r) = bridge.ragdoll_of(subject) {
        for c in &r.colliders {
            exclude.insert(*c);
        }
    }
    // **…and the vehicle the subject is driving** (P29.7) — the same defect
    // shape a third time. A seated character's own collider is parked, and the
    // thing filling the space around it is a four-metre chassis; without this
    // the drive camera sweeps into the car's own bodywork on the first frame and
    // sits at `min_arm_fraction` for the whole segment. The seat is on the
    // movement runtime, which is where the link between a character and the
    // chassis it is riding lives.
    if let Some(cm) = world
        .entity_of(subject)
        .and_then(|e| world.world().get::<CharacterMovement>(e))
    {
        if cm.runtime.seat.is_seated() {
            if let Some(c) = bridge.collider_of(cm.runtime.seat.vehicle) {
                exclude.insert(c);
            }
        }
    }
    let origin = cam.sweep_origin().to_dvec3();
    let desired = cam.desired.to_dvec3();
    let delta = desired - origin;
    let reach = delta.length();
    let resolved = if reach > 1e-6 {
        let dir = delta / reach;
        let radius = cam.tuning.collision_radius_m.max(1e-3);
        match bridge.world_mut().cast_shape_where(
            &ColliderShape3D::Sphere { radius },
            origin,
            DQuat::IDENTITY,
            dir,
            reach,
            &exclude,
            CastTargets::All,
        ) {
            // A sweep that STARTS inside something has no usable contact — the
            // pivot is in a wall, which happens when a character stands with its
            // head through a low ceiling. Sit at the floor rather than at the
            // hit: the alternative is a camera that snaps to the pivot and looks
            // out of the character's own skull.
            Some(hit) => {
                let floor = reach * cam.tuning.min_arm_fraction.clamp(0.0, 1.0);
                let toi = if hit.started_penetrating {
                    floor
                } else {
                    hit.toi.max(floor)
                };
                origin + dir * toi.min(reach)
            }
            None => desired,
        }
    } else {
        desired
    };
    cam.resolve(Vec3d::from_dvec3(resolved));
    Some(cam.pose)
}
