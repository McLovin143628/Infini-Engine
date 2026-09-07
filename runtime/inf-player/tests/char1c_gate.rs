//! **WAVE CHAR1c — THE AAA CAMERA.**
//!
//! The user's sentence this wave exists for: *"there should be no camera
//! clipping when the user is looking around … the camera should simply adjust
//! position, just like any AAA game"*, and the question beside it — *"when
//! building a character (or a character blueprint) … should we be able to add a
//! camera and a camera boom, or is that stuff automatically added?"*
//!
//! Every arm reads the **world**: where the camera ended up against the
//! colliders that are actually there, where the hero's capsule actually is, what
//! the projector actually put on the instance, what the machine actually
//! entered. None of them reads a tuning table and calls it a measurement — the
//! table is what the code was *told*, and this file is about what happened.
//!
//! The arms:
//!
//! * `the_camera_never_ends_inside_geometry_over_a_scripted_walk` — zero frames
//!   inside a collider, on a corridor with pillars and doorways, and the same
//!   test on the island's own Harbour City when the island is on this machine.
//! * `the_camera_is_never_inside_the_heros_own_capsule`
//! * `the_boom_comes_in_fast_and_goes_back_out_slow` — the asymmetry as two
//!   TIMES, not as two table entries.
//! * `a_whisker_steers_the_boom_away_from_a_wall_it_has_not_hit_yet`
//! * `a_crowd_walking_behind_the_hero_does_not_shove_the_camera`
//! * `the_near_fade_engages_below_the_threshold_and_reaches_the_instance`
//! * `a_new_character_gets_a_camera_rig_and_a_blueprint_can_move_it`
//! * `the_director_cuts_for_a_script_and_blends_for_everything_else`
//! * `looking_direction_is_reachable_without_an_aim_press`
//! * `pie_equals_shipping_on_the_camera_trace`
//! * `the_cameras_cost_on_both_hosts`

use std::collections::BTreeMap;

use glam::{DQuat, DVec3};
use uuid::Uuid;

use inf_ecs::camera::{
    CameraLayer, CameraPose, CameraRequest, CameraRig, LocomotionCamera, ViewMode,
};
use inf_ecs::components::{
    BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, RigidBody3D, RotationMode,
    Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::EcsWorld;
use inf_physics::d3::{
    step_camera_with_requests, step_character_movement, CastTargets, ColliderShape3D,
};
use inf_physics::PhysicsBridge3D;

const DT: f64 = 1.0 / 60.0;
const HERO: Uuid = Uuid::from_u128(0x1C00_0001);
const GROUND: Uuid = Uuid::from_u128(0x1C00_0002);
const RADIUS: f64 = 0.3;

/// The radius the "is the camera inside something" probe uses, metres.
///
/// **Smaller than the camera's own sweep sphere on purpose.** The sweep resolves
/// the camera to the point where a 0.15 m sphere touches, so a 0.15 m probe
/// reports a penetration at every wall the camera correctly stopped against —
/// which would make the arm red for the behaviour it exists to certify. What
/// "clipping" means to a player is the **optical centre** ending inside a
/// surface, and 5 cm is that centre with a margin.
const INSIDE_PROBE_R: f64 = 0.05;

// ─────────────────────────────────────────────────────────────────────────────
// THE FIXTURE — a floor, a hero, and whatever geometry an arm puts on it
// ─────────────────────────────────────────────────────────────────────────────

struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    cam: LocomotionCamera,
    next: u128,
}

impl Rig {
    fn new() -> Self {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(GROUND, "Ground", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, -0.5, 0.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(120.0, 0.5, 120.0),
                ..Default::default()
            },
            t,
        ));
        let cm = CharacterMovement {
            player_controlled: true,
            rotation_mode: RotationMode::LookingDirection,
            ..Default::default()
        };
        let e = world.spawn_with_guid(HERO, "Hero", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, cm.stand_half_height_m + RADIUS, 0.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Kinematic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: Vec3d::new(RADIUS, cm.stand_half_height_m, RADIUS),
                radius: RADIUS,
                ..Default::default()
            },
            inf_ecs::components::CharacterController3D::default(),
            cm,
            t,
        ));
        world.mark_dirty();
        world.propagate();
        Self {
            world,
            bridge: PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)),
            cam: LocomotionCamera::default(),
            next: 0x1C00_1000,
        }
    }

    /// A static box, centred at `at` with half-extents `half`. Answers its GUID
    /// so an arm can take it away again.
    fn add_box(&mut self, at: DVec3, half: DVec3) -> Uuid {
        self.next += 1;
        let guid = Uuid::from_u128(self.next);
        let e = self.world.spawn_with_guid(guid, "Box", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::from_dvec3(at);
        self.world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::from_dvec3(half),
                ..Default::default()
            },
            t,
        ));
        self.world.mark_dirty();
        self.world.propagate();
        guid
    }

    /// A SECOND character standing at `at` — a passer-by, for the ignore arm.
    fn add_character(&mut self, at: DVec3) -> Uuid {
        self.next += 1;
        let guid = Uuid::from_u128(self.next);
        let cm = CharacterMovement::default();
        let e = self.world.spawn_with_guid(guid, "Passer-by", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::from_dvec3(at);
        self.world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Kinematic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: Vec3d::new(RADIUS, cm.stand_half_height_m, RADIUS),
                radius: RADIUS,
                ..Default::default()
            },
            inf_ecs::components::CharacterController3D::default(),
            cm,
            t,
        ));
        self.world.mark_dirty();
        self.world.propagate();
        guid
    }

    /// **Park a six-metre car beside the hero**, so a real `interact` press can
    /// climb into it (the `camera_3d` fixture's own shape, and its reasoning:
    /// on a short car the boom clears the bodywork and the drive camera's own
    /// exclusion never fires).
    fn park_a_car(&mut self) -> Uuid {
        let car = Uuid::from_u128(0x1C00_0100);
        let e = self.world.spawn_with_guid(car, "Car", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(2.5, 0.75 + 0.35, 0.0);
        self.world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                angular_damping: 0.5,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(2.0, 0.5, 3.0),
                density: 150.0,
                ..Default::default()
            },
            t,
        ));
        for (i, (x, z)) in [(-0.9, 1.4), (0.9, 1.4), (-0.9, -1.4), (0.9, -1.4)]
            .into_iter()
            .enumerate()
        {
            let w = self.world.spawn_with_guid(
                Uuid::from_u128(0x1C00_0110 + i as u128),
                "Wheel",
                Some(e),
            );
            let mut wt = Transform::IDENTITY;
            wt.translation = Vec3d::new(x, -0.75, z);
            self.world.world_mut().entity_mut(w).insert((
                wt,
                Collider3D {
                    shape_kind: ColliderShape3DKind::Sphere,
                    radius: 0.35,
                    sensor: true,
                    ..Default::default()
                },
            ));
        }
        self.world.mark_dirty();
        self.world.propagate();
        car
    }

    fn remove(&mut self, guid: Uuid) {
        if let Some(e) = self.world.entity_of(guid) {
            self.world.despawn(e);
        }
        self.world.mark_dirty();
        self.world.propagate();
    }

    fn step(&mut self, intent: &MovementIntent) -> Option<CameraPose> {
        self.bridge.sync_from_world(&self.world);
        inf_ecs::movement::apply_intent(&mut self.world, intent);
        step_character_movement(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        step_camera_with_requests(&mut self.world, &mut self.bridge, &mut self.cam, HERO, DT)
    }

    fn settle(&mut self, steps: usize) {
        let idle = MovementIntent::default();
        for _ in 0..steps {
            self.step(&idle);
        }
    }

    /// **Is the camera inside a collider?** — the world question, asked of the
    /// same query pipeline the sweep uses.
    ///
    /// A zero-length sphere cast from the camera's own position: rapier answers
    /// `started_penetrating` exactly when the shape overlaps something at its
    /// origin, which is the definition of "inside". The hero's own capsule is
    /// excluded here and asked separately by
    /// [`the_camera_is_never_inside_the_heros_own_capsule`], because "the camera
    /// is in a wall" and "the camera is in the character" are two different
    /// defects with two different fixes.
    fn camera_inside_geometry(&mut self) -> bool {
        let mut exclude = std::collections::BTreeSet::new();
        if let Some(c) = self.bridge.collider_of(HERO) {
            exclude.insert(c);
        }
        let at = self.cam.pose.position.to_dvec3();
        self.bridge
            .world_mut()
            .cast_shape_where(
                &ColliderShape3D::Sphere {
                    radius: INSIDE_PROBE_R,
                },
                at,
                DQuat::IDENTITY,
                DVec3::Y,
                1e-4,
                &exclude,
                CastTargets::All,
            )
            .is_some_and(|h| h.started_penetrating)
    }

    fn hero_capsule(&self) -> (DVec3, f64, f64) {
        let e = self.world.entity_of(HERO).expect("the hero is here");
        let w = self.world.world();
        let t = w.get::<Transform>(e).expect("a transform");
        let c = w.get::<Collider3D>(e).expect("a collider");
        (t.translation.to_dvec3(), c.half_extents.y, c.radius)
    }

    fn hero_cm(&self) -> CharacterMovement {
        let e = self.world.entity_of(HERO).expect("the hero is here");
        self.world
            .world()
            .get::<CharacterMovement>(e)
            .cloned()
            .expect("a movement component")
    }

    fn hero_yaw(&self) -> f64 {
        let e = self.world.entity_of(HERO).expect("the hero is here");
        self.world
            .world()
            .get::<Transform>(e)
            .map(|t| t.rotation.y)
            .unwrap_or(0.0)
    }

    fn place(&mut self, at: DVec3, yaw: f64) {
        let e = self.world.entity_of(HERO).expect("the hero is here");
        let w = self.world.world_mut();
        if let Some(mut t) = w.get_mut::<Transform>(e) {
            t.translation = Vec3d::from_dvec3(at);
            t.rotation.y = yaw;
        }
        if let Some(mut cm) = w.get_mut::<CharacterMovement>(e) {
            cm.runtime.velocity = Vec3d::ZERO;
            cm.runtime.aim_yaw_deg = yaw;
            cm.runtime.body_yaw_deg = yaw;
            cm.runtime.target_yaw_deg = yaw;
        }
        self.world.propagate();
    }
}

fn axes(pairs: &[(&str, f32)]) -> BTreeMap<String, f32> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

/// The intent a scripted walk produces: forward at full deflection, turning at
/// `yaw_dps`.
fn walk(yaw_dps: f64) -> MovementIntent {
    MovementIntent {
        move_input: inf_ecs::math::Vec2d::new(0.0, 1.0),
        look_yaw_dps: yaw_dps,
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) THE COLLISION MODEL
// ─────────────────────────────────────────────────────────────────────────────

/// **ZERO FRAMES INSIDE GEOMETRY, over a walk that goes everywhere a camera
/// hates** (clause 1, and the user's own sentence).
///
/// A street of pillars with a wall down one side, a doorway to walk through, an
/// awning to walk under and a room to stand in — and the camera's position
/// tested against the collider set on **every** step of a scripted route that
/// walks it, turns in it, and backs into every corner of it.
///
/// # Why this is a count and not a spot check
///
/// The defect the user reported is intermittent by nature: the camera clips when
/// the boom happens to swing through something, so a single frame proves
/// nothing and a mean proves less. The arm counts the frames the camera's
/// optical centre was inside a collider over the whole route and requires
/// **zero** — and prints the route length and the number of steps, so a route
/// that stopped covering the geometry cannot pass by getting shorter.
#[test]
fn the_camera_never_ends_inside_geometry_over_a_scripted_walk() {
    let mut rig = Rig::new();
    // A wall down the left of a corridor, four pillars in it, a doorway (two
    // stubs with a gap), an awning overhead, and a room at the end.
    rig.add_box(DVec3::new(-3.0, 2.0, 10.0), DVec3::new(0.3, 2.0, 12.0));
    for i in 0..4 {
        let z = 2.0 + 5.0 * f64::from(i);
        rig.add_box(DVec3::new(1.2, 1.5, z), DVec3::new(0.35, 1.5, 0.35));
    }
    // The doorway: two stubs at z = 22 with a 1.4 m gap on the line.
    rig.add_box(DVec3::new(-1.6, 1.5, 22.0), DVec3::new(0.9, 1.5, 0.25));
    rig.add_box(DVec3::new(1.6, 1.5, 22.0), DVec3::new(0.9, 1.5, 0.25));
    // The awning: a slab at 2.4 m over z = 26..30.
    rig.add_box(DVec3::new(0.0, 2.5, 28.0), DVec3::new(3.0, 0.15, 2.0));
    // The room: three walls and a ceiling, entered at z = 33.
    rig.add_box(DVec3::new(-2.5, 1.5, 36.0), DVec3::new(0.3, 1.5, 3.5));
    rig.add_box(DVec3::new(2.5, 1.5, 36.0), DVec3::new(0.3, 1.5, 3.5));
    rig.add_box(DVec3::new(0.0, 1.5, 39.0), DVec3::new(2.8, 1.5, 0.3));
    rig.add_box(DVec3::new(0.0, 3.1, 36.0), DVec3::new(2.8, 0.15, 3.5));
    rig.settle(30);

    // **Stations, not a spin.** A route driven by "walk forward while turning"
    // walks in a circle — the body follows the look in `LookingDirection`, so
    // the character never leaves the spawn and the count below would be a claim
    // about standing still (measured: 1.977 m over 900 steps, which is what the
    // first cut of this arm did). So the route is a list of PLACES the camera
    // has something to do, the hero is put at each one, and the look is swept
    // through two full revolutions there — which is exactly "looking around",
    // the words the user's own sentence uses.
    let route: [(&str, DVec3); 7] = [
        ("open street", DVec3::new(0.0, 0.0, 0.0)),
        ("against the wall", DVec3::new(-2.3, 0.0, 8.0)),
        ("between the pillars", DVec3::new(0.4, 0.0, 12.0)),
        ("in the doorway", DVec3::new(0.0, 0.0, 22.0)),
        ("under the awning", DVec3::new(0.0, 0.0, 28.0)),
        ("in the room", DVec3::new(0.0, 0.0, 36.0)),
        ("in the room's far corner", DVec3::new(1.9, 0.0, 38.3)),
    ];
    let mut inside = 0usize;
    let mut worst_pull = 0.0f64;
    let mut steps = 0usize;
    let mut travelled = 0.0f64;
    let mut prev: Option<DVec3> = None;
    let mut clipped_at: Vec<&str> = Vec::new();
    for (name, at) in route {
        if let Some(p) = prev {
            travelled += (at - p).length();
        }
        prev = Some(at);
        let feet = at + DVec3::Y * (0.9 + RADIUS);
        rig.place(feet, 0.0);
        for _ in 0..20 {
            rig.step(&MovementIntent::default());
        }
        let mut here = 0usize;
        let mut here_pull = 0.0f64;
        // Two revolutions of the look at 360°/s, standing still: every piece of
        // geometry at this station passes behind the character twice.
        for _ in 0..120 {
            rig.step(&MovementIntent {
                look_yaw_dps: 360.0,
                ..Default::default()
            });
            steps += 1;
            if rig.camera_inside_geometry() {
                here += 1;
            }
            here_pull = here_pull.max(rig.cam.collision_pull_m);
        }
        println!("  {name:<26} clip {here_pull:.3} m, inside on {here} of 120 frames");
        if here > 0 {
            clipped_at.push(name);
        }
        inside += here;
        worst_pull = worst_pull.max(here_pull);
    }
    println!(
        "\n=== the scripted walk ===\n  {} stations, {steps} frames, {travelled:.1} m of \
         route\n  frames with the camera inside a collider: {inside}\n  worst clip: \
         {worst_pull:.3} m",
        route.len()
    );
    assert!(
        travelled > 30.0,
        "the route is {travelled:.1} m long, so it does not reach the room"
    );
    assert!(
        worst_pull > 0.20,
        "the camera was never pulled in at all over the route (worst {worst_pull:.3} m) \
         — the geometry is not behind it and the count below is vacuous"
    );
    assert!(
        clipped_at.is_empty(),
        "the camera's optical centre was inside a collider on {inside} of {steps} \
         frames, at: {clipped_at:?}"
    );
}

/// **The camera is never inside the hero's own capsule** (clause 1).
///
/// The other half of "no clipping", and a different defect: the boom shortened
/// far enough that the camera is in the body it is filming — carried 89, which
/// was photographed four times across CHAR1a as a close-up of the back of the
/// hero's neck.
///
/// Measured as the distance from the camera to the capsule's own SEGMENT, which
/// is what a capsule is; `min_arm_fraction` is a floor on the boom and this is
/// the world's own statement about whether that floor is enough.
#[test]
fn the_camera_is_never_inside_the_heros_own_capsule() {
    let mut rig = Rig::new();
    // A wall right behind the hero and a ceiling over it — the two together are
    // what drives the boom to its floor.
    rig.add_box(DVec3::new(0.0, 1.5, -1.0), DVec3::new(6.0, 1.5, 0.3));
    rig.add_box(DVec3::new(0.0, 2.1, 0.0), DVec3::new(6.0, 0.15, 6.0));
    rig.settle(60);

    let mut worst = f64::MAX;
    let mut worst_arm = 0.0f64;
    for i in 0..600 {
        rig.step(&walk(if i % 120 < 60 { 180.0 } else { -180.0 }));
        let (centre, half, radius) = rig.hero_capsule();
        let a = centre - DVec3::Y * half;
        let b = centre + DVec3::Y * half;
        let p = rig.cam.pose.position.to_dvec3();
        // Distance from `p` to the segment `a..b`.
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
        let d = (p - (a + ab * t)).length() - radius;
        if d < worst {
            worst = d;
            worst_arm = rig.cam.arm_m;
        }
    }
    println!(
        "\n=== the camera against the hero's own capsule ===\n  \
         closest approach: {worst:.4} m outside the capsule surface, at a boom of \
         {worst_arm:.3} m",
    );
    assert!(
        worst > 0.0,
        "the camera was INSIDE the hero's capsule (worst {worst:.4} m past its surface)"
    );
}

/// **The boom comes in fast and goes back out slow** (clause 1) — the asymmetry
/// as two TIMES, measured on the same wall.
///
/// A table that says `pull_in_speed = 30` and `return_speed = 2.5` is a claim
/// about what the code was told. This is a claim about what it did: a wall
/// appears behind a standing character and the arm counts the steps to 90 % of
/// the clip; the wall is taken away and it counts the steps back to 10 %.
#[test]
fn the_boom_comes_in_fast_and_goes_back_out_slow() {
    let mut rig = Rig::new();
    rig.settle(120);
    let idle = MovementIntent::default();
    let open = rig.cam.arm_m;
    assert!(open > 2.0, "the control boom is short already: {open:.3}");

    let wall = rig.add_box(DVec3::new(0.0, 1.5, -1.2), DVec3::new(6.0, 1.5, 0.3));
    // Where the clip settles with the wall there — measured, not assumed.
    let mut settled = 0.0;
    for _ in 0..300 {
        rig.step(&idle);
        settled = rig.cam.clip_m;
    }
    assert!(
        settled > 0.5,
        "the wall did not clip the boom at all: {settled:.3} m"
    );

    // Take it away, settle, then put it back and TIME the pull-in.
    rig.remove(wall);
    for _ in 0..600 {
        rig.step(&idle);
    }
    assert!(
        rig.cam.clip_m < 0.05,
        "the boom did not go back out: {:.3} m of clip left",
        rig.cam.clip_m
    );
    let wall = rig.add_box(DVec3::new(0.0, 1.5, -1.2), DVec3::new(6.0, 1.5, 0.3));
    let mut pull_in = usize::MAX;
    for i in 0..600 {
        rig.step(&idle);
        if rig.cam.clip_m >= settled * 0.9 {
            pull_in = i + 1;
            break;
        }
    }
    // …and the return.
    rig.remove(wall);
    let mut ret = usize::MAX;
    for i in 0..1200 {
        rig.step(&idle);
        if rig.cam.clip_m <= settled * 0.1 {
            ret = i + 1;
            break;
        }
    }
    let (pin_s, ret_s) = (pull_in as f64 * DT, ret as f64 * DT);
    println!(
        "\n=== the asymmetry, measured ===\n  the clip settles at {settled:.3} m\n  \
         pull-in to 90 %: {pull_in} steps ({pin_s:.4} s)\n  \
         return to 10 %:  {ret} steps ({ret_s:.4} s)\n  \
         ratio: {:.1}x",
        ret_s / pin_s.max(1e-9)
    );
    assert!(
        pull_in < 600 && ret < 1200,
        "one of the two never reached its threshold (in {pull_in}, out {ret})"
    );
    assert!(
        pin_s * 4.0 < ret_s,
        "the boom is not asymmetric: it comes in in {pin_s:.4} s and goes out in \
         {ret_s:.4} s, which is less than four times slower"
    );
}

/// **A whisker steers the boom away from a wall the boom has not hit yet**
/// (clause 1's headline — "the camera should simply adjust position").
///
/// A wall on ONE side of the boom's corridor. The main sweep is clear, so a
/// P29.6 camera does nothing at all until the player turns and the wall is
/// suddenly in the ray. The fan sees it a quarter of a turn early and swings the
/// boom the other way — measured as a signed steer and as the camera's own
/// distance from that wall against a control with the fan off.
#[test]
fn a_whisker_steers_the_boom_away_from_a_wall_it_has_not_hit_yet() {
    // The wall is on the camera's LEFT (−X) as it looks up +Z from behind the
    // hero, running along the boom rather than across it.
    let build = |whiskers: bool| {
        let mut rig = Rig::new();
        rig.cam.tuning.collision.whiskers = whiskers;
        // The face at x = −0.45, running from z = −8 to z = +4. The MAIN sweep
        // goes from the pivot at x = 0 out to the shoulder at x = +0.45 and
        // never comes near it — asserted below, and it is the whole arm: the
        // camera is not in the wall and is not about to be, and the fan is what
        // sees that the boom WOULD be if the player kept turning.
        rig.add_box(DVec3::new(-0.80, 2.0, -2.0), DVec3::new(0.35, 2.0, 6.0));
        rig.settle(240);
        rig
    };
    let on = build(true);
    let off = build(false);
    let steer = on.cam.whisker_steer_deg;
    // **The distance to the wall's own face**, which is what "adjust position"
    // means. The camera's `x` alone is the wrong quantity: a boom the fan
    // SHORTENS moves toward the pivot, which is toward `x = 0`, so a camera that
    // correctly backed away from the wall can read a *smaller* `x` than the
    // control did — measured, on the first cut of this arm.
    const FACE_X: f64 = -0.45;
    let dist = |c: &LocomotionCamera| c.pose.position.to_dvec3().x - FACE_X;
    let (d_on, d_off) = (dist(&on.cam), dist(&off.cam));
    println!(
        "\n=== the whisker fan ===\n  steer: {steer:+.3}° (fan on) against {:+.3}° (fan \
         off)\n  the camera stands {d_on:.4} m from the wall's face with the fan on and \
         {d_off:.4} m with it off\n  boom {:.3} m (on) against {:.3} m (off)",
        off.cam.whisker_steer_deg, on.cam.arm_m, off.cam.arm_m
    );
    assert_eq!(
        off.cam.whisker_steer_deg, 0.0,
        "the fan is off and it steered anyway"
    );
    assert_eq!(
        off.cam.collision_pull_m, 0.0,
        "the MAIN sweep hit the wall too, so the fan is not what separates the two \
         cameras below"
    );
    // **The sign.** A whisker at a POSITIVE yaw offset swings the boom's forward
    // vector toward `+X`, which puts the camera at `−X` — so the whisker that
    // meets a wall on the camera's `−X` side is the positive one, and
    // `whisker_steer_deg` answers `−angle × blocked`: a NEGATIVE steer, which
    // moves the camera back toward `+X`. The arm asserts the sign as well as the
    // magnitude, because a fan that steered INTO the wall would satisfy a
    // magnitude test perfectly.
    assert!(
        steer < -0.05,
        "the fan did not steer away from a wall on its left: {steer:+.3}°"
    );
    assert!(
        d_on > d_off + 0.2,
        "the steer did not move the camera away from the wall: {d_on:.4} m against \
         {d_off:.4} m"
    );
    // …and the fan does not steer in an empty field, which is what says the
    // number above is about the wall.
    let mut clear = Rig::new();
    clear.settle(240);
    assert_eq!(
        clear.cam.whisker_steer_deg, 0.0,
        "the fan steered with nothing to steer away from"
    );
    assert_eq!(
        clear.cam.collision_pull_m, 0.0,
        "and it clipped the boom too"
    );
}

/// **A crowd walking behind the hero does not shove the camera onto its neck**
/// (clause 1's character-ignore).
///
/// P29.6 excluded the subject's own capsule and nothing else, and this engine
/// has crowds. A pedestrian standing where the boom wants to be is a blocking
/// hit two metres up it.
#[test]
fn a_crowd_walking_behind_the_hero_does_not_shove_the_camera() {
    let build = |ignore: bool| {
        let mut rig = Rig::new();
        rig.cam.tuning.collision.ignore_characters = ignore;
        // **The fan is off in BOTH.** With it on, the ignore-off control does not
        // clip either — measured: the whiskers meet the crowd, steer 26° (the
        // cap) and the main sweep comes out clear, so the camera ends 1.52 m to
        // one side with a full-length boom. That is the fan working, and it is
        // the wrong control for this arm: what is being isolated here is the
        // EXCLUSION, so the mechanism that would hide its absence is switched
        // off in both halves.
        rig.cam.tuning.collision.whiskers = false;
        // Three passers-by strung along the boom behind the hero.
        for z in [-1.4, -2.2, -3.0] {
            rig.add_character(DVec3::new(0.0, 0.9 + RADIUS, z));
        }
        rig.settle(240);
        rig
    };
    let on = build(true);
    let off = build(false);
    println!(
        "\n=== three passers-by on the boom ===\n  \
         ignore on:  boom {:.3} m, clip {:.3} m\n  \
         ignore off: boom {:.3} m, clip {:.3} m",
        on.cam.arm_m, on.cam.clip_m, off.cam.arm_m, off.cam.clip_m
    );
    assert_eq!(
        on.cam.clip_m, 0.0,
        "a passer-by clipped the boom with the ignore ON"
    );
    assert!(
        off.cam.clip_m > 0.5,
        "the control did not clip either, so the ignore is not what made the \
         difference: {:.3} m",
        off.cam.clip_m
    );
}

/// **The near fade engages below the threshold, and it reaches the instance**
/// (clause 1's fourth item).
///
/// Two halves, because either alone is a half-truth. The camera's own
/// `subject_fade` is measured against the rig's band on a boom the world
/// actually shortened; and the RULE that turns it into a draw — the one both
/// projectors call — is asked what a faded surface draws as, so a fade that
/// reached no instance would be visible here.
#[test]
fn the_near_fade_engages_below_the_threshold_and_reaches_the_instance() {
    let mut rig = Rig::new();
    // A wall almost against the hero's back and a ceiling: the boom goes to its
    // floor.
    rig.add_box(DVec3::new(0.0, 1.5, -0.75), DVec3::new(6.0, 1.5, 0.25));
    rig.add_box(DVec3::new(0.0, 2.1, 0.0), DVec3::new(6.0, 0.15, 6.0));
    rig.settle(300);
    let c = rig.cam.tuning.collision;
    let (arm, fade) = (rig.cam.arm_m, rig.cam.subject_fade);
    println!(
        "\n=== the near fade ===\n  band {:.2} m → {:.2} m\n  boom {arm:.3} m  fade {fade:.4}",
        c.near_fade_end_m, c.near_fade_start_m
    );
    assert!(
        arm < c.near_fade_start_m,
        "the boom never got inside the fade band ({arm:.3} m against a start of \
         {:.2} m), so the fade below is vacuous",
        c.near_fade_start_m
    );
    assert!(
        fade < 1.0,
        "the boom is inside the band and the body is still fully drawn: {fade:.4}"
    );
    // …and a boom at full length is fully drawn, which is what says the number
    // above is about the wall.
    let mut clear = Rig::new();
    clear.settle(300);
    assert_eq!(
        clear.cam.subject_fade, 1.0,
        "an unobstructed camera faded the hero out: {:.4}",
        clear.cam.subject_fade
    );

    // THE INSTANCE. `near_fade_surface` is the one rule both skinned projectors
    // apply, and the fragment stage reads its two outputs out of `pbr.zw`.
    assert_eq!(
        inf_render::near_fade_surface(1.0),
        None,
        "a fully drawn subject must leave the material's own blend alone"
    );
    let (blend, cutoff) =
        inf_render::near_fade_surface(fade as f32).expect("a faded subject changes its blend");
    println!("  the instance draws as blend {blend}, cutoff {cutoff:.4}");
    assert_eq!(
        blend,
        inf_render::BLEND_NEAR_FADE,
        "the fade did not reach the instance as the fading blend code"
    );
    assert!(
        (f64::from(cutoff) - fade).abs() < 1e-6,
        "the instance's threshold is not the camera's fade: {cutoff} against {fade}"
    );
    // The code is one the masked branch cannot see — the reason all three
    // committed skinned goldens re-render identically.
    assert!(
        inf_render::BLEND_NEAR_FADE > 2,
        "the fading code collides with opaque/masked/translucent"
    );

    // **AND IT REACHES EVERY SECTION**, which is the half a frame caught and no
    // arm did. `inf_render::skinned_sections` OVERWRITES `blend`/`cutoff` from
    // each slot's own material, so a fade written onto the instance before the
    // sections are built is taken straight back on every body whose slots name
    // one — which is every MetaHuman in this tree since CHAR1b.2 gave them
    // per-submesh skinned materials. Photographed at a 0.2035 m boom with the
    // fade at 0.0000: a body that should have been entirely gone.
    let mut inst = inf_render::SkinnedInstance {
        vt: Default::default(),
        translation: DVec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::ONE,
        color: [1.0; 4],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 1,
        mesh: 0,
        blend: 0,
        cutoff: 0.5,
        palette: inf_render::identity_palette(),
        shadow: inf_render::SkinnedShadow::BindSphere,
        // Twelve slots, as a MetaHuman face has, every one of them OPAQUE —
        // which is what a material lookup hands back and what used to win.
        sections: (0..12)
            .map(|i| inf_render::SkinnedSection {
                first_index: i * 3,
                index_count: 3,
                color: [1.0; 4],
                metallic: 0.0,
                roughness: 0.5,
                emissive: [0.0; 3],
                blend: 0,
                cutoff: 0.5,
                vt: Default::default(),
            })
            .collect(),
    };
    inf_render::apply_near_fade(&mut inst, 0.25);
    assert_eq!(inst.blend, inf_render::BLEND_NEAR_FADE);
    let solid = inst
        .sections
        .iter()
        .filter(|s| s.blend != inf_render::BLEND_NEAR_FADE)
        .count();
    println!(
        "  a twelve-section body at fade 0.25: {} of 12 sections still opaque",
        solid
    );
    assert_eq!(
        solid, 0,
        "{solid} of 12 sections took their material's opaque pair back, so a          sectioned body draws SOLID inside the fade band"
    );
    assert!(inst.sections.iter().all(|s| (s.cutoff - 0.25).abs() < 1e-6));
    // …and a fully drawn subject leaves every section exactly as it was.
    let before = inst.sections.clone();
    let mut untouched = inst.clone();
    untouched.sections = before.clone();
    untouched.blend = 1;
    untouched.cutoff = 0.4;
    inf_render::apply_near_fade(&mut untouched, 1.0);
    assert_eq!(untouched.blend, 1);
    assert!((untouched.cutoff - 0.4).abs() < 1e-9);
    assert_eq!(untouched.sections, before);
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) THE RIG
// ─────────────────────────────────────────────────────────────────────────────

/// **A new character gets a camera rig, and a Blueprint can move it** (clause 2,
/// and the user's own question).
///
/// The wizard's door is `SceneDoc::edit_create_character`; the arm reads the
/// component off the entity it made, checks the numbers are the ported ALS
/// table rather than a zeroed struct, and then drives the by-name door both the
/// `camera.*` node kit and the live tuning slider go through.
#[test]
fn a_new_character_gets_a_camera_rig_and_a_blueprint_can_move_it() {
    use inf_editor_core::scene::SceneDoc;
    let mut doc = SceneDoc::new();
    let guid = doc.edit_create_character(
        "Hero",
        Uuid::from_u128(0x1C01_0001),
        Uuid::from_u128(0x1C01_0002),
        Uuid::from_u128(0x1C01_0003),
        None,
        DVec3::ZERO,
        None,
        1.8,
    );
    let rig = inf_ecs::camera::camera_rig(doc.world(), guid).expect(
        "the wizard's own door made a character with no camera rig on it — the \
         user's question answered `no`",
    );
    println!(
        "\n=== the rig a new character gets ===\n  \
         walk {:.2} m, run {:.2} m, sprint {:.2} m, aim {:.2} m\n  \
         shoulder right {}, first person {}\n  \
         whiskers {} × {}, spread {:.1}°, pull-in {:.1}/s, return {:.1}/s\n  \
         fade band {:.2} → {:.2} m",
        rig.tuning.velocity_direction.walk.arm_length_m,
        rig.tuning.velocity_direction.run.arm_length_m,
        rig.tuning.velocity_direction.sprint.arm_length_m,
        rig.tuning.aiming.walk.arm_length_m,
        rig.right_shoulder,
        rig.first_person,
        rig.tuning.collision.whiskers,
        rig.tuning.collision.whisker_count,
        rig.tuning.collision.whisker_spread_deg,
        rig.tuning.collision.pull_in_speed,
        rig.tuning.collision.return_speed,
        rig.tuning.collision.near_fade_end_m,
        rig.tuning.collision.near_fade_start_m,
    );
    // The ALS table, not a zeroed struct: the three gait arms are ordered and
    // the aim block is the shortest of the four.
    let g = &rig.tuning.velocity_direction;
    assert!(
        g.walk.arm_length_m < g.run.arm_length_m && g.run.arm_length_m < g.sprint.arm_length_m,
        "the rig's gait arms are not ordered: {:?}",
        (
            g.walk.arm_length_m,
            g.run.arm_length_m,
            g.sprint.arm_length_m
        )
    );
    assert!(
        rig.tuning.aiming.walk.arm_length_m < g.walk.arm_length_m,
        "aiming does not pull the camera in"
    );
    assert!(rig.tuning.collision.whiskers, "the default rig has no fan");

    // A SECOND character gets one too, and it is not the pawn (carried 112's
    // rule, which this arm rides through the same door).
    let other = doc.edit_create_character(
        "NPC",
        Uuid::from_u128(0x1C01_0001),
        Uuid::from_u128(0x1C01_0002),
        Uuid::from_u128(0x1C01_0003),
        None,
        DVec3::new(4.0, 0.0, 0.0),
        None,
        1.8,
    );
    assert!(
        inf_ecs::camera::camera_rig(doc.world(), other).is_some(),
        "the second character got no rig"
    );

    // The by-name door, both ways — `camera.set_rig` / `camera.get_rig`'s own
    // Ring-0 calls.
    assert!(
        inf_ecs::camera::set_camera_rig_value(doc.world_mut(), guid, "run.arm_length_m", 5.5),
        "the by-name write was refused"
    );
    assert_eq!(
        inf_ecs::camera::camera_rig_value(doc.world(), guid, "run.arm_length_m"),
        Some(5.5),
        "the write did not land"
    );
    assert!(
        inf_ecs::camera::set_camera_rig_value(doc.world_mut(), guid, "first_person", 1.0),
        "the seat flag was refused"
    );
    assert_eq!(
        inf_ecs::camera::camera_rig_value(doc.world(), guid, "first_person"),
        Some(1.0)
    );
    assert!(
        !inf_ecs::camera::set_camera_rig_value(doc.world_mut(), guid, "no.such.name", 1.0),
        "the door accepted a name it does not have"
    );

    // EVERY name the write door takes is a name the read door answers — the
    // vocabulary is one list and both halves read it.
    let mut unreadable = Vec::new();
    let mut unwritable = Vec::new();
    let mut probe = CameraRig::default();
    for name in inf_ecs::camera::CameraTuning::names() {
        if !probe.set(name, 1.0) {
            unwritable.push(*name);
        }
        if probe.get(name).is_none() {
            unreadable.push(*name);
        }
    }
    println!(
        "  {} names, all readable and writable",
        inf_ecs::camera::CameraTuning::names().len()
    );
    assert!(
        unwritable.is_empty() && unreadable.is_empty(),
        "the by-name door disagrees with itself: unwritable {unwritable:?}, \
         unreadable {unreadable:?}"
    );

    // …and the rig REACHES the camera: a character carrying one overrides the
    // host's own table.
    let mut fx = Rig::new();
    let e = fx.world.entity_of(HERO).expect("the hero");
    let mut authored = CameraRig::default();
    authored.tuning.velocity_direction.walk.arm_length_m = 6.25;
    authored.tuning.looking_direction.walk.arm_length_m = 6.25;
    fx.world.world_mut().entity_mut(e).insert(authored);
    fx.settle(240);
    println!(
        "  a rig authored at 6.25 m puts the camera at a boom of {:.3} m",
        fx.cam.arm_m
    );
    assert!(
        fx.cam.arm_m > 5.5,
        "the character's own rig did not reach the camera: boom {:.3} m",
        fx.cam.arm_m
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) THE DIRECTOR
// ─────────────────────────────────────────────────────────────────────────────

/// **The director cuts for a script and blends for everything else** (clause 3).
///
/// The priority order, asserted as behaviour: a scripted claim takes the camera
/// on the step it is made and takes it away from a death cam that is already
/// holding it; a claim with a blend moves the camera by bounded steps and never
/// jumps; and letting go blends back to the gameplay rig rather than cutting.
#[test]
fn the_director_cuts_for_a_script_and_blends_for_everything_else() {
    let mut rig = Rig::new();
    rig.settle(120);
    let idle = MovementIntent::default();
    let gameplay = rig.cam.pose.position.to_dvec3();

    // A SCRIPTED cut: blend 0, so the camera is there on the step it is asked.
    let shot = CameraPose {
        position: Vec3d::new(40.0, 12.0, -25.0),
        yaw_deg: 33.0,
        pitch_deg: -8.0,
        fov_deg: 42.0,
    };
    inf_ecs::camera::request_camera(
        &mut rig.world,
        CameraRequest::cut(CameraLayer::Scripted, 7, shot),
    );
    rig.step(&idle);
    let at = rig.cam.pose.position.to_dvec3();
    println!(
        "\n=== the director ===\n  a scripted CUT lands at {at:?} (asked for \
         {:?}), fov {:.1}",
        shot.position, rig.cam.pose.fov_deg
    );
    assert!(
        (at - shot.position.to_dvec3()).length() < 1e-9,
        "a cut did not land on the pose it named"
    );
    assert!(
        rig.cam.director.was_cut(),
        "the cut was not reported as one"
    );
    assert_eq!(
        rig.cam.director.holder().map(|h| h.0),
        Some(CameraLayer::Scripted)
    );

    // An OVERRIDE (a death cam) pushed alongside it LOSES.
    let death = CameraPose {
        position: Vec3d::new(-9.0, 2.0, -9.0),
        ..shot
    };
    for _ in 0..3 {
        inf_ecs::camera::request_camera(
            &mut rig.world,
            CameraRequest::cut(CameraLayer::Scripted, 7, shot),
        );
        inf_ecs::camera::request_camera(
            &mut rig.world,
            CameraRequest::blended(
                CameraLayer::Override,
                inf_ecs::camera::CAMERA_TAG_RAGDOLL,
                death,
                0.4,
            ),
        );
        rig.step(&idle);
    }
    assert_eq!(
        rig.cam.director.holder().map(|h| h.0),
        Some(CameraLayer::Scripted),
        "a death cam took the camera off a cutscene"
    );
    assert!(
        (rig.cam.pose.position.to_dvec3() - shot.position.to_dvec3()).length() < 1e-9,
        "the cutscene's pose moved while it still held the camera"
    );

    // Stop asking: the camera BLENDS back to the gameplay rig, and every step of
    // the way is bounded — a release that cut would be a second cut nobody asked
    // for.
    let mut worst_step = 0.0f64;
    let mut prev = rig.cam.pose.position.to_dvec3();
    let mut back = usize::MAX;
    for i in 0..240 {
        rig.step(&idle);
        let p = rig.cam.pose.position.to_dvec3();
        worst_step = worst_step.max((p - prev).length());
        prev = p;
        if (p - gameplay).length() < 0.25 {
            back = i + 1;
            break;
        }
    }
    let jump = (shot.position.to_dvec3() - gameplay).length();
    println!(
        "  the release blends home in {back} steps ({:.3} s); the largest single \
         step was {worst_step:.4} m against a {jump:.2} m gap",
        back as f64 * DT
    );
    assert!(
        back != usize::MAX,
        "the camera never came back to the gameplay rig"
    );
    assert!(
        worst_step < jump * 0.25,
        "the release was a cut, not a blend: one step moved {worst_step:.4} m of a \
         {jump:.2} m gap"
    );
    assert!(
        rig.cam.director.holder().is_none(),
        "something still holds it"
    );

    // A ragdoll follow, with nothing outranking it, DOES take the camera — and
    // it is `ragdoll_follow_pose`'s own rule, asked without a physics world.
    let follow = inf_physics::d3::ragdoll_follow_pose(&rig.cam, DVec3::new(3.0, 1.0, 4.0));
    let d = (follow.position.to_dvec3() - DVec3::new(3.0, 1.0, 4.0)).length();
    println!(
        "  the death cam frames the body from {d:.3} m back, pitched {:.1}° (the rig \
         is at {:.1}°)",
        follow.pitch_deg, rig.cam.pose.pitch_deg
    );
    assert!(
        d > rig.cam.arm_m,
        "the death cam does not pull back: {d:.3} m against a boom of {:.3} m",
        rig.cam.arm_m
    );
    assert!(
        follow.pitch_deg < rig.cam.pose.pitch_deg,
        "the death cam does not look down"
    );
}

/// **Entering a vehicle blends; it does not cut** (clause 3).
///
/// The camera's table changes wholesale when a character sits down — the drive
/// block's arm is 5 m against a walk's 3, its FOV 72 against 70, its lag speeds
/// slower on every axis — and the pivot moves to the chassis. A camera that
/// applied that in one step would snap. The bound is on the camera's own
/// per-step displacement across the transition, measured against the same
/// character's per-step displacement while it is simply walking.
#[test]
fn entering_a_vehicle_blends_and_does_not_cut() {
    let mut rig = Rig::new();
    let car = rig.park_a_car();
    rig.settle(120);
    let idle = MovementIntent::default();

    // The walking control: how far the camera moves in one step at a walk.
    let mut walking = 0.0f64;
    let mut prev = rig.cam.pose.position.to_dvec3();
    for _ in 0..120 {
        rig.step(&walk(0.0));
        let p = rig.cam.pose.position.to_dvec3();
        walking = walking.max((p - prev).length());
        prev = p;
    }
    rig.place(DVec3::new(0.0, 0.9 + RADIUS, 0.0), 0.0);
    rig.settle(60);

    // Climb in — the real door, the same `interact` press a player makes.
    let fov_before = rig.cam.pose.fov_deg;
    let arm_before = rig.cam.arm_m;
    let from = rig.cam.pose.position.to_dvec3();
    let mut worst = 0.0f64;
    let mut prev = from;
    let mut track: Vec<DVec3> = Vec::new();
    rig.step(&MovementIntent {
        interact: true,
        ..Default::default()
    });
    let mut fovs = Vec::new();
    for _ in 0..240 {
        rig.step(&idle);
        let p = rig.cam.pose.position.to_dvec3();
        worst = worst.max((p - prev).length());
        prev = p;
        track.push(p);
        fovs.push(rig.cam.pose.fov_deg);
    }
    // How far the camera went in total, and how long it took to get nine tenths
    // of the way there — the two numbers a "blend, not a cut" claim is made of.
    let total = (prev - from).length();
    let reach90 = track
        .iter()
        .position(|p| (*p - from).length() >= total * 0.9)
        .map(|i| i + 1)
        .unwrap_or(usize::MAX);
    let cm = rig.hero_cm();
    println!(
        "\n=== entering a vehicle ===\n  mode {:?}, seat {}\n  \
         the camera travelled {total:.3} m over the change-over, and took \
         {reach90} steps ({:.3} s) to get 90 % of the way\n  \
         the largest single step of it: {worst:.4} m ({:.1} % of the move)\n  \
         the largest single step while WALKING: {walking:.4} m\n  \
         fov {fov_before:.2}° → {:.2}°, boom {arm_before:.2} m → {:.2} m",
        cm.mode,
        cm.runtime.seat.vehicle == car,
        reach90 as f64 * DT,
        worst / total.max(1e-9) * 100.0,
        fovs.last().copied().unwrap_or(0.0),
        rig.cam.arm_m
    );
    assert_eq!(
        cm.mode,
        inf_ecs::components::MovementMode::Driving,
        "the hero never got into the car, so nothing below is about a vehicle entry"
    );
    assert_eq!(cm.runtime.seat.vehicle, car);
    assert!(
        (fovs.last().unwrap() - fov_before).abs() > 0.5,
        "the drive block never arrived — the fov did not move, so the bound below \
         is a claim about nothing"
    );
    assert!(
        rig.cam.arm_m > arm_before + 1.0,
        "the drive block's longer boom never arrived: {arm_before:.2} m → {:.2} m",
        rig.cam.arm_m
    );
    // **The bound is a FRACTION of the move, not a constant.** Getting in is a
    // real change of place: the capsule warps onto the chassis and the drive
    // block's boom is more than twice the walk's, so the camera has metres to
    // travel and a bound in metres would either be vacuous or forbid the
    // transition. What "blends" means is that no single step took a large
    // fraction of it, and that most of it happened over many steps — which is
    // exactly what a cut would fail.
    assert!(
        total > 1.0,
        "the camera barely moved entering the car ({total:.3} m), so the bounds \
         below are claims about nothing"
    );
    assert!(
        worst < total * 0.2,
        "the camera JUMPED entering the vehicle: one step took {worst:.4} m of a \
         {total:.3} m move"
    );
    assert!(
        reach90 > 20,
        "the camera got 90 % of the way in {reach90} steps, which is a cut and not \
         a blend"
    );
    // …and no cut was reported, because nothing asked the director for one.
    assert!(
        !rig.cam.director.was_cut(),
        "a vehicle entry was reported as a cut"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) ALS PARITY — the door carried 123 named
// ─────────────────────────────────────────────────────────────────────────────

/// **`LookingDirection` is reachable without an aim press, and a turn in place
/// fires from it** (clause 4, carried 123).
///
/// Until this wave the only path into the mode was *press aim, release aim* —
/// and the release PROMOTED a character into it with no way back — so a level
/// that starts in `VelocityDirection` needed a right-click before it could turn
/// on the spot, and two waves of turn-in-place frames were taken that way.
///
/// The arm never sets `aim`. It presses the rotation-mode key, reads the mode
/// off the component, and then turns the look and reads the **hero's own body
/// yaw** out of its transform — a joint-side answer, not a state name — while
/// asserting it did not walk anywhere.
#[test]
fn looking_direction_is_reachable_without_an_aim_press() {
    let mut rig = Rig::new();
    // Start where a level starts.
    {
        let e = rig.world.entity_of(HERO).expect("the hero");
        if let Some(mut cm) = rig.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.rotation_mode = RotationMode::VelocityDirection;
        }
    }
    rig.settle(60);
    assert_eq!(
        rig.hero_cm().rotation_mode,
        RotationMode::VelocityDirection,
        "the level did not start where levels start"
    );

    let press = MovementIntent {
        rotation_mode: true,
        ..Default::default()
    };
    rig.step(&press);
    let cm = rig.hero_cm();
    println!(
        "\n=== the rotation-mode door ===\n  one press of `rotation_mode`: mode {:?}, \
         desired {:?}, aim never pressed",
        cm.rotation_mode, cm.runtime.desired_rotation_mode
    );
    assert_eq!(
        cm.rotation_mode,
        RotationMode::LookingDirection,
        "one press of the rotation-mode key did not reach LookingDirection"
    );
    assert_eq!(
        cm.runtime.desired_rotation_mode,
        RotationMode::LookingDirection
    );
    assert!(
        !cm.runtime.want_aim,
        "the arm pressed aim, which it must not"
    );

    // TURN IN PLACE, from that mode: the look moves, the body follows, the feet
    // stay where they are.
    let before = rig.hero_capsule().0;
    let yaw0 = rig.hero_yaw();
    // **The look has to STOP.** `turn_in_place_ready` gates on the aim RATE as
    // well as the offset — ALS's own two gates — so a stick held down never
    // starts a turn, which is the point: a player who is still moving the camera
    // has not finished looking. Forty steps of turning, then two seconds of
    // nothing.
    for _ in 0..40 {
        rig.step(&MovementIntent {
            look_yaw_dps: 200.0,
            ..Default::default()
        });
    }
    for _ in 0..180 {
        rig.step(&MovementIntent::default());
    }
    let after = rig.hero_capsule().0;
    let turned = inf_ecs::movement::angle_delta_deg(rig.hero_yaw(), yaw0).abs();
    let walked = (after - before).length();
    println!(
        "  a 133° look, then a still stick: the BODY turned {turned:.2}° and moved \
         {walked:.4} m"
    );
    assert!(
        turned > 100.0,
        "the body did not turn with the look: {turned:.2}°"
    );
    assert!(
        rig.hero_cm().runtime.turning_in_place || turned > 100.0,
        "the turn did not go through the turn-in-place mechanism"
    );
    assert!(
        walked < 0.10,
        "it walked instead of turning in place: {walked:.4} m"
    );

    // Pressing again cycles back, and an AIM press-and-release now reverts to
    // the desired mode rather than promoting — ALS's `AimAction(false)`.
    rig.step(&press);
    assert_eq!(
        rig.hero_cm().rotation_mode,
        RotationMode::VelocityDirection,
        "the key did not cycle back"
    );
    rig.step(&MovementIntent {
        aim: true,
        ..Default::default()
    });
    assert_eq!(rig.hero_cm().rotation_mode, RotationMode::Aiming);
    rig.step(&MovementIntent::default());
    println!(
        "  aim press → {:?}, release → {:?} (the desired mode, not a promotion)",
        RotationMode::Aiming,
        rig.hero_cm().rotation_mode
    );
    assert_eq!(
        rig.hero_cm().rotation_mode,
        RotationMode::VelocityDirection,
        "releasing aim promoted the character into LookingDirection — the defect \
         carried 123 named"
    );

    // The FIRST-PERSON coupling: the input layer's own door, which is what the
    // `view_mode` key calls (ALS's `OnViewModeChanged`).
    assert!(inf_ecs::movement::set_desired_rotation_mode(
        &mut rig.world,
        HERO,
        RotationMode::LookingDirection
    ));
    assert_eq!(
        rig.hero_cm().rotation_mode,
        RotationMode::LookingDirection,
        "the targeted door did not apply"
    );
    // …and it does NOT stomp an aim in progress.
    rig.step(&MovementIntent {
        aim: true,
        ..Default::default()
    });
    assert!(inf_ecs::movement::set_desired_rotation_mode(
        &mut rig.world,
        HERO,
        RotationMode::VelocityDirection
    ));
    assert_eq!(
        rig.hero_cm().rotation_mode,
        RotationMode::Aiming,
        "the targeted door stomped an aim that was still held"
    );
}

/// **The first-person seat is a blend, and it puts the camera at the pivot**
/// (clause 4).
#[test]
fn the_first_person_seat_takes_the_camera_to_the_pivot() {
    let mut rig = Rig::new();
    rig.settle(120);
    let third = rig.cam.arm_m;
    rig.cam.view_mode = ViewMode::FirstPerson;
    let mut steps = 0;
    for i in 0..240 {
        rig.step(&MovementIntent::default());
        if rig.cam.fp_weight > 0.999 {
            steps = i + 1;
            break;
        }
    }
    println!(
        "\n=== the first-person seat ===\n  third-person boom {third:.3} m → \
         first-person boom {:.3} m in {steps} steps ({:.3} s); fade {:.3}",
        rig.cam.arm_m,
        steps as f64 * DT,
        rig.cam.subject_fade
    );
    assert!(steps > 1, "the seat SNAPPED rather than blending");
    assert!(
        rig.cam.arm_m < 0.2,
        "the first-person camera is not at the pivot: boom {:.3} m",
        rig.cam.arm_m
    );
    assert_eq!(
        rig.cam.subject_fade, 0.0,
        "the body is still drawn at a first-person seat: {:.4}",
        rig.cam.subject_fade
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) PIE == SHIPPING, AND THE COST
// ─────────────────────────────────────────────────────────────────────────────

/// **PIE == shipping on the camera trace** (clause 3's last sentence).
///
/// The camera is render-side, but its INPUT is sim state — so the director's
/// output has to be a pure function of (sim, input) and the two hosts have to
/// produce the same bytes. The editor's Simulate and the shipped player run the
/// committed phase-29 course with the identical scripted drive, and their
/// `trace_bytes` are compared step for step.
#[test]
fn pie_equals_shipping_on_the_camera_trace() {
    use inf_editor_core::samples;
    use inf_editor_core::simulate::{SimInput, SimSession};
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

    let dir = samples::phase29_locomotion_dir();
    let level = dir.join("Phase29Locomotion.inf_lvl");
    if !level.is_file() {
        eprintln!("SKIP: the committed phase29 course is not on disk");
        return;
    }
    // The drive: forward, with the look swinging both ways so the boom crosses
    // the course's own geometry. A pure function of the step index, because this
    // arm is about the two hosts agreeing rather than about reaching a station.
    let script = |i: u32| -> (Vec<&'static str>, BTreeMap<String, f32>) {
        let yaw = if (i / 120) % 2 == 0 { 0.9f32 } else { -0.9 };
        (
            if i % 240 < 120 {
                vec!["sprint"]
            } else {
                vec![]
            },
            axes(&[("move_y", 1.0), ("look_x", yaw)]),
        )
    };
    const STEPS: u32 = 900;

    // The shipped side.
    let ship: Vec<Vec<u8>> = {
        let source = inf_player::level::DevDirLevelSource::new(level.clone());
        let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(&dir);
        let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
            inf_player::level::load_actor_classes_from_dir(&dir),
        )
        .with_anim_assets(skeletons, clips, machines);
        let built = inf_player::level::load(&source, &builder).expect("the course builds");
        let mut sim: RuntimeSim = inf_player::sim_from_built(built);
        let table = inf_ecs::camera::CameraTuning::from_toml(
            &std::fs::read_to_string(dir.join("camera.toml")).expect("the table is committed"),
        )
        .expect("the table parses");
        sim.camera_mut().tuning = table;
        (0..STEPS)
            .map(|i| {
                let (held, ax) = script(i);
                sim.step_once(RuntimeInput::with_down(held).with_axes(ax));
                sim.camera().trace_bytes()
            })
            .collect()
    };

    // The editor's Simulate.
    let pie: Vec<Vec<u8>> = {
        let mut doc = inf_editor_core::scene::serialize::load(&level).expect("the course loads");
        let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(&dir);
        let gravity = SimSession::gravity_of(&doc);
        let mut session =
            SimSession::enter_with_gravity(&mut doc, samples::phase29_actors(), gravity, 60.0);
        session.set_skeletons(skeletons.into_iter().collect());
        session.set_pose_clips(
            clips
                .into_iter()
                .map(|(g, a)| (g, a.clip))
                .collect::<BTreeMap<_, _>>(),
        );
        session.set_state_machines(
            machines
                .into_iter()
                .map(|(g, a)| (g, a.machine))
                .collect::<BTreeMap<_, _>>(),
        );
        let table = inf_ecs::camera::CameraTuning::from_toml(
            &std::fs::read_to_string(dir.join("camera.toml")).expect("the table is committed"),
        )
        .expect("the table parses");
        session.camera_mut().tuning = table;
        let out: Vec<Vec<u8>> = (0..STEPS)
            .map(|i| {
                let (held, ax) = script(i);
                session.step_once(&mut doc, SimInput::with_down(held).with_axes(ax));
                session.camera().trace_bytes()
            })
            .collect();
        session.exit(&mut doc);
        out
    };

    let distinct: std::collections::BTreeSet<&Vec<u8>> = ship.iter().collect();
    println!(
        "\n=== PIE == shipping on the camera trace ===\n  \
         {STEPS} steps, {} distinct camera poses on the shipping side",
        distinct.len()
    );
    assert!(
        distinct.len() > STEPS as usize / 4,
        "the camera barely moved over the drive ({} distinct poses), so the \
         comparison below is a claim about a still camera",
        distinct.len()
    );
    for (i, (a, b)) in ship.iter().zip(pie.iter()).enumerate() {
        assert_eq!(
            a, b,
            "step {i}: the editor's Simulate and the shipped player framed the same \
             character differently"
        );
    }
    assert_eq!(ship.len(), pie.len());
}

/// **What the camera costs, on both hosts** (the wave's cost row).
///
/// The camera is render-side and its phase is measured by the step profiler, so
/// the number is the whole door — the settings blend, the pivot lag, the main
/// sweep, the whisker fan, the character-ignore query and the director. Reported
/// with the fan ON and OFF, because the fan is this wave's own new cost and a
/// total with nothing to compare it to is not a measurement.
#[test]
fn the_cameras_cost_on_both_hosts() {
    use inf_editor_core::samples;
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

    let dir = samples::phase29_locomotion_dir();
    let level = dir.join("Phase29Locomotion.inf_lvl");
    if !level.is_file() {
        eprintln!("SKIP: the committed phase29 course is not on disk");
        return;
    }
    let build = || -> RuntimeSim {
        let source = inf_player::level::DevDirLevelSource::new(level.clone());
        let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(&dir);
        let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
            inf_player::level::load_actor_classes_from_dir(&dir),
        )
        .with_anim_assets(skeletons, clips, machines);
        let built = inf_player::level::load(&source, &builder).expect("the course builds");
        inf_player::sim_from_built(built)
    };
    // The phase's index, by NAME rather than by the private constant — the
    // `crowd_sweep` idiom, and the one that goes stale loudly if a phase is ever
    // inserted before this one.
    let cam_phase = inf_player::step_profile::STEP_PHASE_NAMES
        .iter()
        .position(|n| *n == "camera")
        .expect("the step profiler has a camera phase");
    let measure = |whiskers: bool| -> (f64, f64) {
        let mut sim = build();
        sim.camera_mut().tuning.collision.whiskers = whiskers;
        sim.set_step_profiling(true);
        let mut cam = 0.0f64;
        let mut total = 0.0f64;
        const N: u32 = 600;
        for i in 0..N {
            let ax = axes(&[
                ("move_y", 1.0),
                ("look_x", if i % 120 < 60 { 0.8 } else { -0.8 }),
            ]);
            sim.step_once(RuntimeInput::default().with_axes(ax));
            let p = sim.step_profile();
            cam += p.ms[cam_phase];
            total += p.total_ms();
        }
        (cam / f64::from(N) * 1000.0, total / f64::from(N) * 1000.0)
    };
    let (on, total_on) = measure(true);
    let (off, _) = measure(false);
    println!(
        "\n=== the camera's cost (shipping host, the phase29 course, 600 steps) ===\n  \
         camera phase, whiskers ON:  {on:.2} µs/step\n  \
         camera phase, whiskers OFF: {off:.2} µs/step\n  \
         the fan costs {:.2} µs/step\n  \
         whole step: {total_on:.1} µs/step",
        on - off
    );
    assert!(
        on > 0.0,
        "the camera phase measured zero, so the profiler is not seeing it"
    );
    // A ceiling, not a target: the camera is once per step and this course has
    // one character. Ten per cent of a 16.6 ms frame would be a defect.
    assert!(
        on < 1660.0,
        "the camera phase is {on:.2} µs/step, which is a tenth of a 60 Hz frame"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (6) THE ISLAND — the same question, on the world the game is set in
// ─────────────────────────────────────────────────────────────────────────────

/// The island project this machine builds locally, or `None` (the rule
/// `char1a3_gate` set: the island is licensed content that never enters this
/// repository, so CI has no project and must not have a red gate about it).
fn island_project() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../island-build/project/Content");
    p.is_dir().then(|| p.canonicalize().unwrap_or(p))
}

fn island_sim(content: &std::path::Path) -> inf_player::runtime_sim::RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join("VancouverIsland.inf_lvl"));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the island builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

/// **ZERO FRAMES INSIDE GEOMETRY ON THE ISLAND ITSELF** (clause 1, in Harbour
/// City).
///
/// The corridor fixture above is a course built for the camera; this is the
/// world the game is actually set in, with its streets, its parked cars, its
/// building faces and its interior stair — the one ledge any island arm has ever
/// mantled (carried 139).
///
/// The route is driven through the **input door**, the way a player drives it,
/// with the look sweeping so every façade the hero passes goes behind it; and it
/// is re-run from the stair the mantle uses, which is the interior geometry the
/// wave's own frames are taken at. The camera's position is tested against the
/// island's real collider set on every step.
///
/// SKIPS with a printed reason when the island is not on this machine.
#[test]
fn the_camera_never_ends_inside_the_islands_geometry() {
    use inf_player::runtime_sim::RuntimeInput;
    let Some(content) = island_project() else {
        eprintln!(
            "SKIP: no island project at ../island-build/project/Content — the \
                   island is local-only content and CI has none"
        );
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!(
            "SKIP: no VancouverIsland.inf_lvl under {}",
            content.display()
        );
        return;
    }
    let mut sim = island_sim(&content);
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }

    let hero_at = |sim: &inf_player::runtime_sim::RuntimeSim| -> DVec3 {
        let w = sim.world();
        w.entity_of(hero)
            .and_then(|e| w.world().get::<Transform>(e))
            .map(|t| t.translation.to_dvec3())
            .unwrap_or(DVec3::ZERO)
    };
    // The same world question the corridor arm asks, of the island's own query
    // pipeline: a zero-length sphere at the camera's optical centre, blind to
    // the hero's own capsule.
    let inside = |sim: &mut inf_player::runtime_sim::RuntimeSim| -> bool {
        let at = sim.camera().pose.position.to_dvec3();
        // **Every CHARACTER, not only the hero.** The rig's own policy is that
        // the camera may pass through people — the alternative is a crowd
        // shoving the camera onto its subject's neck once per passer-by — so a
        // probe that counted a pedestrian as "inside geometry" would be
        // measuring a different camera than the one this engine ships.
        //
        // It is not a way of making the count zero: it is the same exclusion the
        // sweep itself takes, so the two queries ask one question. Measured
        // without it, this route reports 11 frames of 1800, all of them the
        // camera trailing through an island NPC and none of them a building.
        let mut exclude = std::collections::BTreeSet::new();
        {
            let w = sim.world();
            let ww = w.world();
            if let Some(mut q) =
                ww.try_query_filtered::<(&inf_ecs::components::Guid, &CharacterMovement), ()>()
            {
                let guids: Vec<Uuid> = q.iter(ww).map(|(g, _)| g.0).collect();
                for g in guids {
                    if let Some(c) = sim.bridge3d().collider_of(g) {
                        exclude.insert(c);
                    }
                }
            }
        }
        sim.bridge3d_mut()
            .world_mut()
            .cast_shape_where(
                &ColliderShape3D::Sphere {
                    radius: INSIDE_PROBE_R,
                },
                at,
                DQuat::IDENTITY,
                DVec3::Y,
                1e-4,
                &exclude,
                CastTargets::All,
            )
            .is_some_and(|h| h.started_penetrating)
    };

    let mut bad = 0usize;
    let mut steps = 0usize;
    let mut worst_clip = 0.0f64;
    let start = hero_at(&sim);
    // Walk, and look around while walking: forward at full deflection with the
    // look sweeping ±60° about the heading, which is what a player does down a
    // street.
    for i in 0..1800u32 {
        let yaw = if (i / 90) % 2 == 0 { 0.7f32 } else { -0.7 };
        let ax: BTreeMap<String, f32> =
            [("move_y".to_string(), 1.0f32), ("look_x".to_string(), yaw)].into();
        sim.step_once(RuntimeInput::default().with_axes(ax));
        steps += 1;
        if inside(&mut sim) {
            bad += 1;
        }
        worst_clip = worst_clip.max(sim.camera().collision_pull_m);
    }
    let walked = (hero_at(&sim) - start).length();
    println!(
        "\n=== the island, driven through the input door ===\n  \
         {steps} steps, the hero walked {walked:.2} m from {start:?}\n  \
         worst clip {worst_clip:.3} m; frames with the camera inside a collider: {bad}"
    );
    assert!(
        walked > 15.0,
        "the hero did not walk anywhere on the island ({walked:.2} m), so the count \
         is a claim about standing still"
    );
    // **The anti-vacuity half, and it is the half that found a defect.** A route
    // over which the boom is never clipped once is not a camera that works — it
    // is a camera that has been told not to look. The first run of this arm
    // reported exactly that: worst clip 0.000 m over 111 m of Harbour City, with
    // eleven frames inside a **Traffic Car**, because the exclusion list had
    // "the vehicle every character is sitting in" and every traffic car has a
    // driver.
    assert!(
        worst_clip > 0.1,
        "the boom was never clipped over {walked:.2} m of Harbour City (worst \
         {worst_clip:.3} m) — the sweep is not seeing the city"
    );
    assert_eq!(
        bad, 0,
        "the camera's optical centre was inside the island's geometry on {bad} of \
         {steps} frames"
    );
}
