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
    // **The subject's own rig, if it has one** (wave CHAR1c, clause 2).
    //
    // An OVERRIDE and not a replacement: a character with no `CameraRig` — which
    // is every character in every level committed before this wave — leaves the
    // host's table exactly where it was, so the behaviour of an unrigged subject
    // is byte-for-byte what P29.6 shipped. A rigged one brings its whole table,
    // its shoulder and its seat, which is what makes possessing an NPC a camera
    // change rather than a nothing.
    if let Some(rig) = world.world().get::<inf_ecs::camera::CameraRig>(entity) {
        cam.tuning = rig.tuning;
        cam.right_shoulder = rig.right_shoulder;
        cam.view_mode = if rig.first_person {
            inf_ecs::camera::ViewMode::FirstPerson
        } else {
            inf_ecs::camera::ViewMode::ThirdPerson
        };
    }
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
    // **…and every OTHER character's BODY** (wave CHAR1c), when the rig's
    // collision policy says so — which the shipped one does.
    //
    // The subject's own capsule has been excluded since P29.6 for the obvious
    // reason; every other character was not, and this engine has crowds now. A
    // pedestrian walking behind the hero is a blocking hit two metres up the
    // boom, so the camera snapped onto the hero's neck and back out again once
    // per passer-by — which is carried 89's shape with a moving cause. A
    // ragdolling character's limbs go with its capsule, because a body on the
    // floor is the same body.
    //
    // **Their VEHICLES do not**, and the first cut of this block had them.
    // Excluding "the car every character is sitting in" reads as the same rule
    // as excluding the subject's own — and it is not: on the island every
    // traffic car has a driver, so the whole moving fleet became invisible to
    // the camera. Measured, by this wave's own island arm: 11 frames of 1 800
    // with the camera inside a **Traffic Car** and the boom reporting no clip at
    // all, because the sweep had been told not to look. A pedestrian is
    // something a camera may pass through; two tonnes of bodywork is not. The
    // SUBJECT's own car stays excluded, below, for P29.7's reason — a seated
    // character's capsule is parked and the chassis is what fills the space
    // around it.
    //
    // The cost is one `O(characters)` query per step over a world whose
    // characters are already being iterated four times by the movement step, and
    // it is measured in the wave's cost row rather than assumed small.
    if cam.tuning.collision.ignore_characters {
        let w = world.world();
        if let Some(mut q) =
            w.try_query_filtered::<(&inf_ecs::components::Guid, &CharacterMovement), ()>()
        {
            let guids: Vec<uuid::Uuid> = q.iter(w).map(|(g, _)| g.0).collect();
            for g in guids {
                if let Some(c) = bridge.collider_of(g) {
                    exclude.insert(c);
                }
                if let Some(r) = bridge.ragdoll_of(g) {
                    for c in &r.colliders {
                        exclude.insert(*c);
                    }
                }
            }
        }
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
    let radius = cam.tuning.collision_radius_m.max(1e-3);
    let floor = reach * cam.tuning.min_arm_fraction.clamp(0.0, 1.0);
    // One closure, so the main sweep and every whisker ask the world the same
    // question with the same exclusions and the same "started inside something"
    // rule. Answers how far along `dir` the sphere may travel.
    let mut sweep = |dir: DVec3, len: f64| -> f64 {
        if len <= 1e-6 {
            return len;
        }
        match bridge.world_mut().cast_shape_where(
            &ColliderShape3D::Sphere { radius },
            origin,
            DQuat::IDENTITY,
            dir,
            len,
            &exclude,
            CastTargets::All,
        ) {
            // A sweep that STARTS inside something has no usable contact — the
            // pivot is in a wall, which happens when a character stands with its
            // head through a low ceiling. Sit at the floor rather than at the
            // hit: the alternative is a camera that snaps to the pivot and looks
            // out of the character's own skull.
            //
            // **THE FLOOR MAY NOT OVERRIDE A CLEAN CONTACT** (wave CHAR1c's
            // audit). This branch used to read `hit.toi.max(floor)`, and that
            // `max` is the user's own reported defect: when a wall is nearer to
            // the pivot than `reach * min_arm_fraction` (0.152 m on a 3.035 m
            // boom) the floor pushes the camera PAST the contact the sweep just
            // found, straight into the surface. Measured on Harbour City's own
            // façades by `the_camera_never_ends_inside_the_islands_geometry_on_a_
            // hostile_route`: **104 of 3480 frames** with the optical centre
            // inside geometry, **every one of them at a boom of 0.1517 m**,
            // which is that floor exactly. Dropping the `max` takes it to
            // **8**, and the eight left are frames where the CHARACTER is inside
            // the geometry (five with the pivot itself buried) and no camera
            // position is legal.
            //
            // The floor's own argument does not survive this wave, which is why
            // it can go: "looks out of the character's own skull" was written in
            // P29.6, before the near fade existed. A boom under
            // `near_fade_end_m` (0.35 m) now draws no subject at all, so a camera
            // that comes all the way to a contact shows the player the room and
            // not the inside of their own character. The floor stays in the
            // PENETRATING branch, where there is no contact to respect.
            Some(hit) => {
                if hit.started_penetrating {
                    floor
                } else {
                    hit.toi.min(len)
                }
            }
            None => len,
        }
    };

    let free = if reach > 1e-6 {
        sweep(delta / reach, reach)
    } else {
        reach
    };

    // ── the whisker fan (wave CHAR1c, clause 1) ──
    //
    // A short fan of casts either side of the boom, so the camera answers a wall
    // it is ABOUT to swing into rather than one it is already in. Two things come
    // out of it: a **steer**, which is what makes the camera "simply adjust
    // position" instead of jumping in, and a **predictive bound** on the boom —
    // a whisker blocked at `d` along a direction `cos θ` off the boom means there
    // is geometry at boom-depth `d·cos θ`, so the boom is bounded by the smallest
    // such depth over the fan.
    //
    // The whiskers share the boom's own sphere radius on purpose: a whisker cast
    // with a different radius is answering a question about a different camera.
    let mut steer_deg = 0.0f64;
    let arm_full = cam.arm_full();
    if reach > 1e-6 {
        for a in cam.tuning.collision.whisker_angles_deg() {
            let target = cam
                .camera_at(arm_full, cam.whisker_steer_deg + a)
                .to_dvec3();
            let d = target - origin;
            let r = d.length();
            if r <= 1e-6 {
                continue;
            }
            let dir = d / r;
            let toi = sweep(dir, r);
            steer_deg += cam
                .tuning
                .collision
                .whisker_steer_deg(a, (toi / r).clamp(0.0, 1.0));
            // **A whisker STEERS and does not shorten** — measured, and the
            // measurement is why the line that shortened is not here.
            //
            // The obvious second use of a blocked whisker is to bound the boom
            // at the obstacle's own depth along the boom axis (`toi · cos θ`).
            // It is wrong twice. A whisker that reached its full length still
            // ends 92 % of the reach up that axis at 22.5°, so bounding on every
            // whisker takes 8 % off the boom in an EMPTY field —
            // `camera_3d`'s own control arm caught that at 0.231 m. And bounding
            // only on a whisker that HIT is still too eager: a wall 45 cm to the
            // side of the corridor took a metre off a 3.04 m boom (measured by
            // this wave's own fan arm), which is a camera that bobs in and out
            // once per alley for geometry it was never going to touch.
            //
            // The steer is the prediction. If the boom really is going into
            // something the MAIN sweep is what says so, and it is the one thing
            // that shortens the arm.
        }
    }
    cam.resolve_swept(free, steer_deg, dt);

    // ── the ragdoll follow (wave CHAR1c, clause 3) ──
    //
    // A character that has gone down is no longer where its capsule says it is —
    // `ragdoll_bridge` disables that capsule and seventeen limbs fall down the
    // hill in its place. The gameplay rig keeps framing the capsule, so the death
    // the player just had happens off screen. This is the director's `Override`
    // layer's whole reason to exist: a request, per step, that frames the PELVIS
    // the bodies actually ended at, and stops being pushed the step the ragdoll
    // is cleaned up.
    //
    // It is a request rather than a branch inside the rig because a photo mode, a
    // takedown camera and a cutscene all want the same seat and only one of them
    // can have it — which is a priority question, and a priority question wants a
    // stack.
    if let Some(r) = bridge.ragdoll_of(subject) {
        if let Some(body) = r.pelvis.or(r.root) {
            if let Some(at) = bridge.world_mut().body_translation(body) {
                let pose = ragdoll_follow_pose(cam, at);
                cam.director
                    .request(inf_ecs::camera::CameraRequest::blended(
                        inf_ecs::camera::CameraLayer::Override,
                        inf_ecs::camera::CAMERA_TAG_RAGDOLL,
                        pose,
                        RAGDOLL_FOLLOW_BLEND_S,
                    ));
            }
        }
    }

    // ── the cover camera (wave COV1, clause 4) ──
    //
    // A claim on the director's `Override` layer, pushed every step the subject
    // is in cover and simply not pushed the step it leaves — the same shape the
    // ragdoll follow above has, and for the same reason: a photo mode, a
    // takedown camera and a cutscene all want this seat and only one of them can
    // have it, which is a priority question and a priority question wants a
    // stack. There is no second camera anywhere in this wave.
    if cm.mode.is_cover() && cm.runtime.cover.active {
        match cover_camera_pose(cam, &cm, feet, bridge, &exclude) {
            Some(pose) => {
                // **THE FADE FOLLOWS THE CLAIM'S OWN BOOM** (wave COV1).
                //
                // `subject_fade` is a pure function of `arm_m`, and `arm_m` is
                // the GAMEPLAY rig's boom — so a claim that brings the camera in
                // closer than the rig would draws a body the fade rule has not
                // been told about. Measured in the demo loop's peek frame: the
                // camera a hand's width from a character's back, `subject_fade`
                // 1.0000, and the inside of its own mesh filling the window.
                //
                // The fix is one call to the SAME rule (CHAR1c's law: the near
                // fade is one function and both projectors read it), on the
                // distance this claim is actually asking for, taking whichever
                // of the two is thinner.
                let d = (pose.position.to_dvec3() - cam.pivot.to_dvec3()).length();
                cam.subject_fade = cam.subject_fade.min(cam.tuning.collision.near_fade(d));
                cam.cover_hold = Some(pose);
                cam.director
                    .request(inf_ecs::camera::CameraRequest::blended(
                        inf_ecs::camera::CameraLayer::Override,
                        inf_ecs::camera::CAMERA_TAG_COVER,
                        pose,
                        COVER_CAMERA_BLEND_S,
                    ));
            }
            // **The refusal** (carried 161). The pivot is inside geometry and
            // there is no legal camera position at all: hold the last one there
            // was, and push nothing at all if there has never been one.
            None => {
                if let Some(hold) = cam.cover_hold {
                    cam.director
                        .request(inf_ecs::camera::CameraRequest::blended(
                            inf_ecs::camera::CameraLayer::Override,
                            inf_ecs::camera::CAMERA_TAG_COVER,
                            hold,
                            COVER_CAMERA_BLEND_S,
                        ));
                }
            }
        }
    } else {
        cam.cover_hold = None;
    }

    Some(cam.direct(dt))
}

/// How long the cover camera takes to arrive, and to leave, seconds.
///
/// Shorter than the death cam's: taking cover is a control the player made and
/// a camera that took two-thirds of a second to answer it would feel late.
pub const COVER_CAMERA_BLEND_S: f64 = 0.35;

/// **How far the cover camera slides toward the open side**, metres.
///
/// The whole point of the state: a camera behind a character whose back is
/// against a wall is a camera looking at a wall. Sliding it toward the side the
/// character can shoot from is what lets the player see what they are about to
/// lean into — GTA's own cover framing, and Gears'.
pub const COVER_CAMERA_SHIFT_M: f64 = 0.55;

/// **How much shorter the boom is in cover**, as a fraction of the rig's own.
///
/// A tenth off. The subject is not going anywhere and the interesting part of
/// the frame is what is past the corner, not the character — and the number is
/// small because the rig's own boom is already the one the world allowed, so
/// taking much off it is asking for a camera inside the subject. The first cut
/// took a quarter and the demo loop photographed the inside of a character's
/// own mesh; the near-fade line above is the other half of that repair.
pub const COVER_CAMERA_ARM_SCALE: f64 = 0.90;

/// **How much room the swept cover boom leaves at its contact**, metres.
///
/// Two centimetres. A shape cast answers the distance at which the sphere
/// TOUCHES, and a camera parked exactly there reads as penetrating to the next
/// frame's query — which is how a "clear" boom becomes a black frame.
pub const COVER_CAMERA_SKIN_M: f64 = 0.02;

/// **The shortest cover boom that is still a camera**, metres.
///
/// Below this there is no room behind the shifted pivot at all and the claim
/// refuses rather than sitting on the subject's own head. It is under the near
/// fade's end (`CameraCollision::near_fade_end_m`), so a boom this short draws
/// no subject and shows the player the room.
pub const COVER_CAMERA_MIN_ARM_M: f64 = 0.10;

/// **Where the camera goes while its subject is in cover** (wave COV1).
///
/// The rig's own yaw and pitch, around a pivot shifted toward the **open side**
/// — the corner the character can lean around, or the top of a low cover — with
/// a shorter boom. `None` when the shifted pivot is inside geometry, which is
/// the refusal `LocomotionCamera::cover_hold` answers.
///
/// The shift is along the cover's own tangent, which is why the surface's normal
/// is what this reads rather than the camera's yaw: a player who has swung the
/// camera round to look down the wall must still see past the corner the
/// character is standing at.
fn cover_camera_pose(
    cam: &LocomotionCamera,
    cm: &CharacterMovement,
    feet: DVec3,
    bridge: &mut PhysicsBridge3D,
    exclude: &std::collections::BTreeSet<ColliderId3D>,
) -> Option<CameraPose> {
    use inf_ecs::cover::CoverSide;
    let c = cm.runtime.cover;
    let left = inf_ecs::cover::tangent_left(c.normal).to_dvec3();
    if left == DVec3::ZERO {
        return None;
    }
    // Which way is "open". A corner peek's open side is the corner it leans
    // around; a low cover has none, so the camera stays where the rig put it and
    // only the boom shortens — the character rises into frame rather than
    // stepping out of it.
    let shift = match c.side {
        CoverSide::Left => COVER_CAMERA_SHIFT_M,
        CoverSide::Right => -COVER_CAMERA_SHIFT_M,
        // Not leaning yet: bias toward whichever corner is nearer, so the
        // player is already looking at the way out before they press aim.
        CoverSide::Behind | CoverSide::Over => {
            if c.left_m <= c.right_m {
                COVER_CAMERA_SHIFT_M * 0.5
            } else {
                -COVER_CAMERA_SHIFT_M * 0.5
            }
        }
    };
    let pivot = cam.pivot.to_dvec3() + left * shift;
    // **The pivot must be clear.** A sphere the boom's own radius at the shifted
    // pivot: if it starts penetrating there is no legal camera position on any
    // boom out of it, and the caller holds its last one.
    let r = cam.tuning.collision_radius_m.max(1e-3);
    if bridge
        .world_mut()
        .cast_shape_where(
            &ColliderShape3D::Sphere { radius: r },
            pivot,
            DQuat::IDENTITY,
            DVec3::Y,
            1e-3,
            exclude,
            CastTargets::All,
        )
        .is_some_and(|h| h.started_penetrating)
    {
        return None;
    }
    let (_, _, forward) = inf_ecs::camera::basis(cam.pose.yaw_deg, cam.pose.pitch_deg);
    let arm = cam.arm_m.max(0.05) * COVER_CAMERA_ARM_SCALE;
    // ── **AND THE BOOM IS SWEPT** (the COV1 audit) ──
    //
    // `cam.arm_m` is the distance the world allowed out of the GAMEPLAY rig's
    // pivot, and this claim moves the pivot sideways by up to
    // `COVER_CAMERA_SHIFT_M` before hanging the boom off it. A boom that is
    // legal at one pivot is not legal at another: shifted half a metre along a
    // Harbour City shop front, the same 3 m arm reaches into the next shop.
    //
    // Measured by `cov1_gate::the_cover_camera_is_outside_the_world_or_it_is_
    // not_there` over eight stations on the island: at (-1782.0, 2074.0) the
    // claim was pushed with its optical centre INSIDE the world. That is the
    // very failure CHAR1c's whole camera wave is named after, re-introduced by
    // a claim that did not ask the question the rig asks.
    //
    // So it asks it, with the same sphere, the same exclusions and the same
    // rule the rig's own `sweep` closure uses — CHAR1c's law that a camera
    // question has one implementation. A sweep that starts penetrating is the
    // pivot case above and has already refused; anything else stops at the
    // contact, less a skin so the next frame's query does not call the touch a
    // penetration.
    let free = match bridge.world_mut().cast_shape_where(
        &ColliderShape3D::Sphere { radius: r },
        pivot,
        DQuat::IDENTITY,
        -forward,
        arm,
        exclude,
        CastTargets::All,
    ) {
        Some(hit) if hit.started_penetrating => return None,
        Some(hit) => (hit.toi.min(arm) - COVER_CAMERA_SKIN_M).max(0.0),
        None => arm,
    };
    // Nowhere to stand: the shifted pivot is clear but everything behind it is
    // solid within a sphere's width. That is the refusal `cover_hold` answers,
    // and it is better than a camera sitting on the pivot looking out of the
    // character's own skull.
    if free <= COVER_CAMERA_MIN_ARM_M {
        return None;
    }
    // The feet are read so a low cover's camera drops with the crouch rather
    // than staying at a standing character's eye line — the pivot already
    // carries the stance, and this is the assertion that it does.
    debug_assert!(feet.y <= cam.pivot.y + 1.0e-6);
    Some(CameraPose {
        position: Vec3d::from_dvec3(pivot - forward * free),
        yaw_deg: cam.pose.yaw_deg,
        pitch_deg: cam.pose.pitch_deg,
        fov_deg: cam.pose.fov_deg,
    })
}

/// **Advance the camera, and drain the world's camera claims into it first**
/// (wave CHAR1c).
///
/// The door both hosts call. `step_locomotion_camera` is left as the door that
/// takes an immutable world — every arm written against it since P29.6 still
/// compiles and still means the same thing — and this is the one that also
/// empties [`inf_ecs::camera::CameraDirectorRes`], which is where a Blueprint's
/// `camera.shot` and the editor sequencer's camera track put their claims.
///
/// The drain is FIRST, so a claim raised by a Blueprint on this step's Tick is
/// answered on this step's camera rather than one behind.
pub fn step_camera_with_requests(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    cam: &mut LocomotionCamera,
    subject: Uuid,
    dt: f64,
) -> Option<CameraPose> {
    for r in inf_ecs::camera::take_camera_requests(world) {
        cam.director.request(r);
    }
    step_locomotion_camera(world, bridge, cam, subject, dt)
}

/// How long the death cam takes to arrive, and to leave, seconds.
pub const RAGDOLL_FOLLOW_BLEND_S: f64 = 0.6;

/// How far the death cam pulls back beyond the rig's own boom, as a multiplier.
pub const RAGDOLL_FOLLOW_ARM_SCALE: f64 = 1.35;

/// How far down the death cam looks, degrees, on top of the player's own pitch.
pub const RAGDOLL_FOLLOW_PITCH_DEG: f64 = -12.0;

/// **Where the camera goes while its subject is a ragdoll** (wave CHAR1c).
///
/// The rig's own yaw, pitch tilted down, and the boom lengthened — around the
/// body's actual position rather than the parked capsule's. Split out as a pure
/// function of `(camera, world point)` so the rule can be read, and asserted,
/// without a physics world.
pub fn ragdoll_follow_pose(cam: &LocomotionCamera, at: DVec3) -> CameraPose {
    let pitch = (cam.pose.pitch_deg + RAGDOLL_FOLLOW_PITCH_DEG).clamp(
        cam.tuning.collision.pitch_min_deg,
        cam.tuning.collision.pitch_max_deg,
    );
    let (_, _, forward) = inf_ecs::camera::basis(cam.pose.yaw_deg, pitch);
    let arm = (cam.arm_m.max(cam.settings.arm_length_m * 0.5)) * RAGDOLL_FOLLOW_ARM_SCALE;
    CameraPose {
        position: Vec3d::from_dvec3(at - forward * arm),
        yaw_deg: cam.pose.yaw_deg,
        pitch_deg: pitch,
        fov_deg: cam.pose.fov_deg,
    }
}
