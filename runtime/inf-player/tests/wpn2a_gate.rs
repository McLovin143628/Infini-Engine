//! **WAVE WPN2a — BALLISTICS.** The gate.
//!
//! # What every arm in this file reads
//!
//! **The world.** A round's position at step N against its position at step
//! N + 1; the entity a segment cast named; the joules a target's `Health`
//! actually lost; the metres a character actually covered. Never a report's
//! summary of itself, never a table, never a state name.
//!
//! # The vacuity rule this wave is built against
//!
//! The hybrid's whole point is the FAR half, and an arm satisfied inside a
//! weapon's `hitscan_threshold_m` reads nothing about it: the instant ray that
//! shipped at island wave I6 would pass it perfectly. So every arm below that
//! claims something about a projectile is fought at a distance **past** the
//! threshold, and the ones that could be satisfied by the old ray say so in
//! their own doc.
//!
//! # The mutations each arm dies to
//!
//! Written beside the arm, not in a report. They were run.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::ballistics::{
    self, MAX_ROUNDS_IN_FLIGHT, MAX_SHOT_RAYS_PER_STEP, PROJECTILE_GRAVITY_MPS2,
    PROJECTILE_SUB_STEPS,
};
use inf_ecs::components::{
    AnimStateMachine, BodyKind3D, CharacterController3D, CharacterMovement, Collider3D,
    ColliderShape3DKind, MovementMode, RigidBody3D, SkeletalMesh, Transform,
};
use inf_ecs::item::{self, ItemDef, ItemDefs};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::weapon::{self, Health, ShotKind, WeaponDef};
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};
use inf_project::ProjectManifest;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const RADIUS: f64 = 0.3;

const HERO: Uuid = Uuid::from_u128(0x2A00_0001);
const GROUND: Uuid = Uuid::from_u128(0x2A00_0002);
const TARGET: Uuid = Uuid::from_u128(0x2A00_0003);
const WALL: Uuid = Uuid::from_u128(0x2A00_0004);
const CHASSIS: Uuid = Uuid::from_u128(0x2A00_0005);
/// A SECOND chassis, down range, that nobody is sitting in — the control
/// half of the shooter-exclusion arm (wave WPN2a audit).
const PARKED: Uuid = Uuid::from_u128(0x2A00_0006);
fn shooter_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2A00_0100 + i as u128)
}

// ── the range ───────────────────────────────────────────────────────────────

/// **A flat range with nothing in the way**, so a round's flight is the only
/// thing that decides where it goes.
///
/// The ground is a thin slab **below** the muzzle line and the world is
/// otherwise empty: a fixture with a wall two metres away would resolve every
/// shot inside its own hitscan threshold and prove nothing about the half this
/// wave built.
struct Range {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    /// **The pose seam**, installed only by the arms that need one (wave WPN2a
    /// audit). `None` for every arm written before it, so their steps are the
    /// steps they always were: `step_pose_evaluation` publishes an
    /// `evaluated_pose`, and `head_point` prefers a pose's `head` socket to the
    /// capsule rule — a seam a rig-less arm must not be given by accident.
    rig: Option<(
        inf_anim::SkeletonAsset,
        inf_anim::StateMachine,
        inf_anim::AnimClip,
    )>,
}

/// The rig, the machine and the clip a ragdoll needs before the bridge will
/// build it any bodies (wave WPN2a audit).
const SKEL_GUID: Uuid = Uuid::from_u128(0x2A00_0007);
const SM_GUID: Uuid = Uuid::from_u128(0x2A00_0008);
const CLIP_REF: inf_anim::ClipRef = [0x2a; 16];

/// The one rifle every flight arm fires, so a drop is read against one set of
/// numbers. A **projectile** past 25 m, the doc's own assault-rifle threshold.
fn test_rifle() -> WeaponDef {
    WeaponDef {
        kind: ShotKind::Projectile,
        automatic: true,
        damage_j: 600.0,
        rounds_per_minute: 600.0,
        // No spread: a gate that cannot name where the bullet went cannot say
        // whether it dropped.
        spread_deg: 0.0,
        magazine: 300,
        reserve: 300,
        range_m: 1000.0,
        muzzle_speed_mps: 900.0,
        hitscan_threshold_m: 25.0,
        drag_k: 0.0003,
        gravity_scale: 1.0,
        ..Default::default()
    }
}

fn defs_with(id: &str, def: WeaponDef) -> ItemDefs {
    let mut d = ItemDefs::default();
    assert!(d.insert(ItemDef {
        id: id.into(),
        label: id.into(),
        stack_max: 1,
        mass_kg: 3.6,
        weapon: Some(def),
    }));
    d
}

impl Range {
    fn new(defs: ItemDefs) -> Self {
        let mut world = EcsWorld::new();
        // A floor 3 m below the muzzle line, 4 km long, so a round that flies
        // level never touches it and one that drops eventually does.
        let e = world.spawn_with_guid(GROUND, "Ground", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, -3.0, 1000.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(200.0, 0.5, 2000.0),
                ..Default::default()
            },
            t,
        ));
        stand(&mut world, HERO, "Hero", DVec3::ZERO, true);
        *item::item_defs_mut(&mut world) = defs;
        world.mark_dirty();
        world.propagate();
        let mut r = Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
            rig: None,
        };
        r.bridge.sync_from_world(&r.world);
        r
    }

    fn arm(&mut self, who: Uuid, id: &str) {
        assert!(item::give_inventory(&mut self.world, who, 4));
        assert_eq!(
            item::give(&mut self.world, who, id, 1),
            0,
            "the bag would not take a {id}"
        );
        assert!(
            d3::gameplay::equip_weapon(&mut self.world, who, id),
            "the {id} would not equip"
        );
    }

    /// Point the hero's aim. `pitch` is degrees, positive is up.
    fn aim(&mut self, who: Uuid, yaw: f64, pitch: f64) {
        let e = self.world.entity_of(who).expect("a shooter");
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.aim_yaw_deg = yaw;
        cm.runtime.aim_pitch_deg = pitch;
    }

    fn hold_trigger(&mut self, who: Uuid, down: bool) {
        let e = self.world.entity_of(who).expect("a shooter");
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.want_attack = down;
        cm.runtime.press_attack = down;
    }

    /// One fixed step, in the hosts' own order.
    ///
    /// It deliberately does **not** call `apply_intent`: that door writes the
    /// PLAYER-CONTROLLED character's whole runtime intent from a
    /// `MovementIntent`, so a default one clears the very trigger
    /// [`Range::hold_trigger`] just set — measured, on this fixture's own hero,
    /// which fired nothing at all until the call came out. The arms that want
    /// the hero to MOVE drive `apply_intent` themselves with an intent that
    /// says so.
    fn step(&mut self) -> d3::GameplayReport {
        self.bridge.sync_from_world(&self.world);
        d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
        let report = d3::step_gameplay(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        if let Some((skeleton, machine, clip)) = &self.rig {
            let machines = |g: Uuid| (g == SM_GUID).then_some(machine);
            let skels = |g: Uuid| (g == SKEL_GUID).then_some(skeleton);
            let clips = |c: inf_anim::ClipRef| (c == CLIP_REF).then_some(clip);
            let vars = |_: Uuid| std::collections::BTreeMap::new();
            inf_ecs::pose::step_pose_evaluation(
                &mut self.world,
                DT,
                &machines,
                &skels,
                &clips,
                &vars,
            );
        }
        report
    }

    /// **Give this range a rig**, so a character on it can ragdoll (wave WPN2a
    /// audit). The mannequin the wizard builds, and a one-state machine — the
    /// bridge only needs the rig; the machine is what `AnimStateMachine` binds.
    fn install_rig(&mut self) {
        let skeleton = inf_anim::build_template(
            inf_anim::BodyPlan::Biped,
            &inf_anim::BodyParams {
                height_m: 1.8,
                ..Default::default()
            },
        )
        .expect("the mannequin builds");
        let machine = inf_anim::StateMachine {
            states: vec![inf_anim::SmState::clip("idle", CLIP_REF)],
            entry: 0,
            ..Default::default()
        };
        self.rig = Some((
            skeleton,
            machine,
            inf_anim::AnimClip::new("pose", Vec::new()),
        ));
    }

    /// Every round in flight, right now.
    fn rounds(&self) -> Vec<ballistics::Round> {
        ballistics::round_pool(&self.world)
            .map(|p| p.rounds.clone())
            .unwrap_or_default()
    }

    fn health(&self, who: Uuid) -> Option<f64> {
        let e = self.world.entity_of(who)?;
        self.world.world().get::<Health>(e).map(|h| h.joules)
    }
}

/// Stand one capsule character up with its FEET at `at`.
fn stand(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, player: bool) {
    let cm = CharacterMovement {
        player_controlled: player,
        ..Default::default()
    };
    let e = world.spawn_with_guid(guid, name, None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(at.x, at.y + cm.stand_half_height_m + RADIUS, at.z);
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
        CharacterController3D::default(),
        cm,
        t,
    ));
}

/// The same, with the rig and machine a **ragdoll** needs (wave WPN2a audit).
fn stand_rigged(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3) {
    stand(world, guid, name, at, false);
    let e = world.entity_of(guid).expect("the character just stood up");
    world.world_mut().entity_mut(e).insert((
        AnimStateMachine {
            sm: Some(SM_GUID),
            ..Default::default()
        },
        SkeletalMesh {
            mesh: Some(Uuid::from_u128(0x2A00_0009)),
            skeleton: Some(SKEL_GUID),
        },
    ));
    world.mark_dirty();
}

/// A **thin** static slab across the range at `z` — the tunnelling test's whole
/// point is that it is thinner than one sub-step of flight.
fn spawn_thin_wall(world: &mut EcsWorld, guid: Uuid, z: f64, half_thickness: f64) {
    let e = world.spawn_with_guid(guid, "Wall", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, 1.4, z);
    world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(20.0, 8.0, half_thickness),
            ..Default::default()
        },
        t,
    ));
}

/// The muzzle height every arm reads a drop against — `muzzle_of`'s capsule
/// answer, because this fixture's characters have no rig.
const MUZZLE_Y: f64 = d3::gameplay::MUZZLE_HEIGHT_M;

// ── (a) THE HYBRID IS REAL ──────────────────────────────────────────────────

/// **A round leaves the barrel, and it MOVES.**
///
/// The vacuity check first: at 10 m the instant ray resolves and the pool stays
/// empty — which is the near half unchanged — and past 25 m a body appears in
/// the pool and its position at step N differs from its position at step N + 1
/// by about `v·dt`.
///
/// **Mutation → red:** setting `hitscan_threshold_m` to `MAX_RANGE_M` empties
/// the pool and the second half of this arm fails at `spawned == 0`.
#[test]
fn a_shot_past_the_threshold_becomes_a_body_in_flight() {
    // The near half, unchanged: a target at 10 m is inside the 25 m threshold.
    let mut near = Range::new(defs_with("rifle", test_rifle()));
    stand(
        &mut near.world,
        TARGET,
        "Target",
        DVec3::new(0.0, 0.0, 10.0),
        false,
    );
    near.world.mark_dirty();
    near.world.reindex_guids();
    near.world.propagate();
    near.bridge.sync_from_world(&near.world);
    near.arm(HERO, "rifle");
    near.aim(HERO, 0.0, 0.0);
    near.hold_trigger(HERO, true);
    let r = near.step();
    assert_eq!(r.shots, 1, "the rifle did not fire");
    assert_eq!(
        r.rounds.spawned, 0,
        "a shot INSIDE the hitscan threshold minted a round — the near half is \
         supposed to be the instant ray that shipped at I6"
    );
    assert!(
        near.health(TARGET).is_some_and(|j| j < 2000.0),
        "the instant ray did not reach a target 10 m away"
    );

    // The far half: nothing in the way, so the ray misses and a round leaves.
    let mut far = Range::new(defs_with("rifle", test_rifle()));
    far.arm(HERO, "rifle");
    far.aim(HERO, 0.0, 0.0);
    far.hold_trigger(HERO, true);
    let r = far.step();
    assert_eq!(r.shots, 1);
    assert_eq!(
        r.rounds.spawned, 1,
        "a shot that missed past the threshold minted no round"
    );
    far.hold_trigger(HERO, false);
    let live = far.rounds();
    assert_eq!(live.len(), 1);
    let at0 = live[0].at;
    // It is minted AT the threshold, not at the muzzle.
    assert!(
        (at0.z - 25.0).abs() < 0.01,
        "the round was minted at z {:.4} and the threshold is 25 m",
        at0.z
    );
    assert!(
        (live[0].travelled_m - 25.0).abs() < 1e-9,
        "the round's odometer starts at {:.4} m and should start at the \
         threshold the instant ray already covered",
        live[0].travelled_m
    );
    far.step();
    let at1 = far.rounds()[0].at;
    let moved = at1.z - at0.z;
    println!("the round moved {moved:.4} m in one 60 Hz step (v0 900 m/s => 15.0 m)");
    assert!(
        moved > 14.0 && moved < 15.1,
        "a 900 m/s round moved {moved:.4} m in a 60 Hz step"
    );
    assert!(at1.y < at0.y, "the round did not drop at all");
}

/// **A hit at 200 m lands LATER than one at 10 m**, by the flight steps the
/// muzzle speed predicts — and the drop matches the analytic parabola.
///
/// This is the arm the old instant ray cannot pass at all: it resolved both
/// distances on the step the trigger went down.
///
/// **Mutation → red:** `muzzle_speed_mps` × 10 collapses the step count to 1
/// and the `>= 12` assertion fails.
#[test]
fn a_rifle_hits_later_and_lower_at_two_hundred_metres_than_at_ten() {
    // 10 m: inside the threshold, so it lands on the firing step.
    let mut near = Range::new(defs_with("rifle", test_rifle()));
    stand(
        &mut near.world,
        TARGET,
        "Target",
        DVec3::new(0.0, 0.0, 10.0),
        false,
    );
    near.world.mark_dirty();
    near.world.reindex_guids();
    near.world.propagate();
    near.bridge.sync_from_world(&near.world);
    near.arm(HERO, "rifle");
    near.aim(HERO, 0.0, 0.0);
    near.hold_trigger(HERO, true);
    let before = 2000.0;
    let r = near.step();
    near.hold_trigger(HERO, false);
    assert_eq!(r.shots, 1);
    let near_steps = 0;
    assert!(
        near.health(TARGET).is_some_and(|j| j < before),
        "the 10 m target was not hit on the firing step"
    );

    // 200 m: past the threshold, so it flies. A tall slab so the drop cannot
    // make it miss.
    let mut far = Range::new(defs_with("rifle", test_rifle()));
    spawn_thin_wall(&mut far.world, WALL, 200.0, 0.1);
    far.world.mark_dirty();
    far.world.reindex_guids();
    far.world.propagate();
    far.bridge.sync_from_world(&far.world);
    far.arm(HERO, "rifle");
    far.aim(HERO, 0.0, 0.0);
    far.hold_trigger(HERO, true);
    far.step();
    far.hold_trigger(HERO, false);
    let mut far_steps = 0u32;
    let mut landed_at = None;
    for _ in 0..120 {
        let r = far.step();
        far_steps += 1;
        if let Some(hit) = r.hits.iter().find(|h| h.target == Some(WALL)) {
            landed_at = Some(hit.to);
            break;
        }
    }
    let landed = landed_at.expect("the round never reached the wall at 200 m");
    println!(
        "10 m: hit on the firing step ({near_steps} flight step(s)). \
         200 m: hit after {far_steps} flight step(s), at y {:.4} (muzzle {MUZZLE_Y:.4})",
        landed.y
    );
    assert!(
        far_steps >= 12,
        "a 900 m/s round covered 200 m in {far_steps} step(s) of 15 m — the \
         flight is not being simulated"
    );
    assert!(far_steps > near_steps);

    // THE DROP, against the closed form. The round leaves the threshold at
    // 25 m with the muzzle's own speed, so its time of flight over the
    // remaining 175 m is at LEAST 175/900 s and more once drag has slowed it;
    // the parabola over that time is the floor on how far it can fall.
    let drop = MUZZLE_Y - landed.y;
    let t_min = (200.0 - 25.0) / 900.0;
    let parabola_min = 0.5 * PROJECTILE_GRAVITY_MPS2 * t_min * t_min;
    println!(
        "the drop at 200 m is {drop:.4} m; the dragless parabola over {t_min:.5} s is {parabola_min:.4} m"
    );
    assert!(
        drop > parabola_min,
        "the round fell {drop:.4} m and a dragless one falls {parabola_min:.4} m — \
         drag can only make the flight LONGER and the drop bigger"
    );
    assert!(
        drop < parabola_min * 1.6,
        "the round fell {drop:.4} m against a dragless {parabola_min:.4} m — that \
         is more than this drag coefficient can account for"
    );
}

/// **A thin wall at 100 m stops the round** — the doc's own reason for a
/// segment cast rather than a point move.
///
/// The wall is **20 cm** thick and one sub-step at 900 m/s is **3.75 m**, so a
/// point move steps clean over it nineteen times out of twenty.
///
/// **Mutation → red:** replacing the segment cast in `step_rounds` with a test
/// of the endpoint (`cast_ray_excluding(next, dir, 0.0, …)`) lets the round
/// through and this arm fails at `stopped`.
#[test]
fn a_thin_wall_at_a_hundred_metres_is_not_tunnelled() {
    let mut r = Range::new(defs_with("rifle", test_rifle()));
    spawn_thin_wall(&mut r.world, WALL, 100.0, 0.1);
    r.world.mark_dirty();
    r.world.reindex_guids();
    r.world.propagate();
    r.bridge.sync_from_world(&r.world);
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    r.step();
    r.hold_trigger(HERO, false);
    let mut stopped = None;
    let mut furthest = 0.0_f64;
    for _ in 0..60 {
        let rep = r.step();
        for round in r.rounds() {
            furthest = furthest.max(round.at.z);
        }
        if let Some(h) = rep.hits.iter().find(|h| h.target == Some(WALL)) {
            stopped = Some(h.to.z);
            break;
        }
    }
    println!(
        "the round stopped at z {:?}; the furthest any round reached was {furthest:.4} m \
         (the wall is a 0.2 m slab at 100 m and one sub-step is 3.75 m)",
        stopped
    );
    let z = stopped.expect("the round tunnelled through a 20 cm wall at 100 m");
    assert!(
        (z - 99.9).abs() < 0.2,
        "the round stopped at {z:.4} m and the wall's near face is at 99.9 m"
    );
    assert!(
        furthest < 100.2,
        "a round reached {furthest:.4} m, past a wall it should have hit"
    );
}

/// **A round dies at its weapon's `range_m`**, and the pool empties.
#[test]
fn a_round_dies_at_its_range_and_the_pool_empties() {
    let mut def = test_rifle();
    def.range_m = 120.0;
    let mut r = Range::new(defs_with("rifle", def));
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    r.step();
    r.hold_trigger(HERO, false);
    let mut expired = 0u32;
    let mut seen_flying = false;
    for _ in 0..40 {
        let rep = r.step();
        expired += rep.rounds.expired;
        seen_flying |= rep.rounds.in_flight > 0;
        if r.rounds().is_empty() && expired > 0 {
            break;
        }
    }
    assert!(seen_flying, "the round never flew");
    assert_eq!(expired, 1, "the round did not die at its range");
    assert!(r.rounds().is_empty(), "the pool did not empty");
    assert!(
        ballistics::round_state_bytes(&r.world).is_empty(),
        "a pool with nothing in it still folded bytes into the trace"
    );
}

// ── (b) THE SEGMENT CAST AND THE SHOOTER ────────────────────────────────────

/// **A round that leaves the active partition dies, and is counted.**
///
/// The third of the four deaths (`range_m`, a hit, the band, old age) and the
/// only one that needs a banded world: a `StreamingSource` on the hero and a
/// band tight enough that 200 m is outside it. Without a source the band is
/// UNBOUNDED and every position answers `Tier::Near`, which is what every other
/// arm in this file runs under — so this arm is also the statement that the
/// check is reachable at all.
///
/// It matters because a round outside the band is flying through geometry that
/// is not in the physics world: every segment from there on reports a miss it
/// did not earn.
#[test]
fn a_round_that_leaves_the_active_partition_dies_and_is_counted() {
    let mut r = Range::new(defs_with("rifle", test_rifle()));
    {
        let e = r.world.entity_of(HERO).expect("the hero");
        r.world
            .world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::StreamingSource { radius_m: 64.0 });
        r.world.mark_dirty();
        r.world.reindex_guids();
        r.world.propagate();
    }
    // Tight enough that a round crossing 200 m is outside it, and wide enough
    // that the threshold spawn at 25 m is inside.
    r.bridge.set_collider_band_radii(48.0, 96.0);
    r.bridge.sync_from_world(&r.world);
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    let first = r.step();
    r.hold_trigger(HERO, false);
    assert_eq!(first.rounds.spawned, 1, "the shot did not become a round");
    let mut left = 0u32;
    let mut expired = 0u32;
    let mut furthest = 0.0_f64;
    for _ in 0..40 {
        for round in r.rounds() {
            furthest = furthest.max(round.at.z);
        }
        let rep = r.step();
        left += rep.rounds.left_band;
        expired += rep.rounds.expired;
        if r.rounds().is_empty() {
            break;
        }
    }
    println!(
        "with a 48/96 m band the round reached {furthest:.2} m and died: left_band {left}, expired {expired}"
    );
    assert_eq!(left, 1, "the round did not die on leaving the band");
    assert_eq!(expired, 0, "it died of something else first");
    assert!(
        furthest < 200.0,
        "the round reached {furthest:.2} m, well past a 96 m band"
    );
}

/// **A round fired from inside a body does not hit that body — and one fired at
/// a chassis nobody is sitting in STOPS IN IT.**
///
/// Two halves, and until the WPN2a audit only one of them meant anything: the
/// shot's cast door filtered to `CastTargets::Fixed`, so a chassis was invisible
/// to the cast and "the round did not hit the car it was fired from inside" was
/// satisfied by a round that could not have hit any car at all. The cast asks
/// `CastTargets::AllSolid` now, so the second half is a control rather than a
/// tautology: the SAME box, forty metres down range with nobody in it, stops the
/// round.
///
/// **Mutation → red:** dropping the seat from `shot_exclusions` (or the whole
/// exclusion set from segment 0) makes the first half fail — the round stops on
/// the chassis it left.
#[test]
fn a_round_does_not_hit_its_own_shooter_or_the_chassis_it_is_sitting_in() {
    let mut r = Range::new(defs_with("rifle", test_rifle()));
    // A dynamic box AROUND the hero — a chassis, in the only sense this engine
    // has of one.
    {
        let e = r.world.spawn_with_guid(CHASSIS, "Chassis", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 1.0, 0.0);
        r.world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(1.0, 1.0, 2.5),
                ..Default::default()
            },
            t,
        ));
        r.world.mark_dirty();
        r.world.reindex_guids();
        r.world.propagate();
        r.bridge.sync_from_world(&r.world);
    }
    // **THE HERO IS SITTING IN IT.** `MovementRuntime::seat` is what the engine
    // means by "which vehicle this character is in", and it is what
    // `shot_exclusions` reads. The mode is deliberately left alone: this fixture
    // has no `Vehicle`, and what is under test is the exclusion, not the drive.
    {
        let e = r.world.entity_of(HERO).expect("the hero");
        let mut cm = r
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.seat.vehicle = CHASSIS;
    }
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    let rep = r.step();
    r.hold_trigger(HERO, false);
    assert_eq!(rep.shots, 1);
    assert!(
        rep.hits.iter().all(|h| h.target != Some(HERO)),
        "the shot hit its own shooter"
    );
    assert!(
        rep.hits.iter().all(|h| h.target != Some(CHASSIS)),
        "the shot hit the chassis it was fired from inside"
    );
    // …and the round that left it does not hit either on any later segment.
    let mut hits: Vec<Option<Uuid>> = Vec::new();
    for _ in 0..40 {
        let rep = r.step();
        hits.extend(rep.hits.iter().map(|h| h.target));
    }
    println!("the round's whole flight hit: {hits:?}");
    assert!(
        hits.iter().all(|t| *t != Some(HERO) && *t != Some(CHASSIS)),
        "a segment after the first hit the shooter or its chassis"
    );

    // **THE CONTROL.** The same box, forty metres away, with nobody in it. If
    // this passes and the half above passes, the exclusion is doing the work; if
    // this fails, the cast cannot see a chassis at all and the half above proved
    // nothing (which is what it proved for the whole of wave WPN2a).
    let mut c = Range::new(defs_with("rifle", test_rifle()));
    {
        let e = c.world.spawn_with_guid(PARKED, "Parked", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 1.0, 40.0);
        c.world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(1.0, 1.0, 2.5),
                ..Default::default()
            },
            t,
        ));
        c.world.mark_dirty();
        c.world.reindex_guids();
        c.world.propagate();
        c.bridge.sync_from_world(&c.world);
    }
    c.arm(HERO, "rifle");
    c.aim(HERO, 0.0, 0.0);
    c.hold_trigger(HERO, true);
    c.step();
    c.hold_trigger(HERO, false);
    let mut stopped_at = None;
    for _ in 0..40 {
        let rep = c.step();
        if let Some(h) = rep.hits.iter().find(|h| h.target == Some(PARKED)) {
            stopped_at = Some(h.to);
            break;
        }
    }
    let at = stopped_at.expect(
        "a round fired at a parked chassis nobody is in flew straight through it — \
         the cast class is back to `Fixed` and the exclusion half of this arm is \
         vacuous again",
    );
    println!(
        "the control: the round stopped at z {:.3} m (the chassis' near face is 37.5 m)",
        at.z
    );
    assert!(
        (at.z - 37.5).abs() < 0.5,
        "the round stopped at z {:.3} and the chassis' near face is 37.5 m",
        at.z
    );
}

// ── (c) THE CEILING AND THE REFUSAL ─────────────────────────────────────────

/// **Eight shooters at 900 rpm fill the pool, and the refusal is a VALUE.**
///
/// The brief's own population. The pool's bound is `MAX_SHOT_RAYS_PER_STEP /
/// PROJECTILE_SUB_STEPS`, so the ceiling is reached by construction and the
/// eighty-fifth round is refused rather than dropped.
///
/// **Mutation → red:** making `spawn_round` push unconditionally leaves
/// `refused == 0` and the pool over its bound; both assertions fail.
#[test]
fn eight_shooters_at_nine_hundred_rpm_engage_the_ceiling_and_the_refusal_is_a_value() {
    let mut def = test_rifle();
    def.rounds_per_minute = 900.0;
    def.range_m = 2000.0;
    let mut r = Range::new(defs_with("rifle", def));
    for i in 0..8 {
        stand(
            &mut r.world,
            shooter_guid(i),
            "Shooter",
            DVec3::new(i as f64 * 4.0, 0.0, 0.0),
            false,
        );
    }
    r.world.mark_dirty();
    r.world.reindex_guids();
    r.world.propagate();
    r.bridge.sync_from_world(&r.world);
    for i in 0..8 {
        r.arm(shooter_guid(i), "rifle");
        r.aim(shooter_guid(i), 0.0, 20.0);
    }
    let mut refused = 0u64;
    let mut spawned = 0u64;
    let mut peak = 0usize;
    let mut peak_rays = 0u32;
    for _ in 0..240 {
        for i in 0..8 {
            r.hold_trigger(shooter_guid(i), true);
        }
        let rep = r.step();
        refused += u64::from(rep.rounds.refused);
        spawned += u64::from(rep.rounds.spawned);
        peak = peak.max(rep.rounds.in_flight as usize);
        peak_rays = peak_rays.max(rep.rounds.rays);
    }
    println!(
        "8 shooters x 900 rpm over 240 steps: {spawned} round(s) minted, {refused} REFUSED, \
         peak {peak} in flight (bound {MAX_ROUNDS_IN_FLIGHT}), peak {peak_rays} segment casts \
         a step (ceiling {MAX_SHOT_RAYS_PER_STEP})"
    );
    assert!(spawned > 0, "nobody fired");
    assert!(
        refused > 0,
        "the ceiling never engaged at the population it was minted from — \
         either the pool is unbounded or the shooters are not firing"
    );
    assert!(
        peak <= MAX_ROUNDS_IN_FLIGHT,
        "{peak} rounds were in flight against a bound of {MAX_ROUNDS_IN_FLIGHT}"
    );
    assert!(
        peak_rays <= MAX_SHOT_RAYS_PER_STEP as u32,
        "the flight spent {peak_rays} casts against a ceiling of {MAX_SHOT_RAYS_PER_STEP}"
    );
    // The refusal is COUNTED on the pool as well as on the step's report, which
    // is what makes it a value a log or a HUD can read rather than a number that
    // exists for one step.
    let pool = ballistics::round_pool(&r.world).expect("a pool");
    assert_eq!(pool.refused, refused, "the pool's own tally disagrees");
    assert_eq!(pool.spawned, spawned);
}

// ── (d) THE DAMAGE CURVE AND THE HEAD ───────────────────────────────────────

/// **The Hermite smoothstep, reproduced at five distances against the closed
/// form** — and it is the WORLD's joules, not the curve function's own answer.
///
/// Five targets, five distances, all past the threshold so every one of them is
/// resolved by a round in flight. `min_damage_frac` is reached at `max_range_m`.
///
/// **Mutation → red:** returning `base_j` from `damage_curve_j` flattens every
/// row and the middle three assertions fail.
#[test]
fn the_damage_curve_is_the_docs_own_at_five_distances_in_the_world() {
    let mut def = test_rifle();
    def.damage_j = 600.0;
    def.effective_range_m = 35.0;
    def.max_range_m = 235.0;
    def.min_damage_frac = 0.6;
    // Slow it right down so the drop over 200 m does not make it miss a capsule.
    def.gravity_scale = 0.0;
    let distances = [35.0, 85.0, 135.0, 185.0, 235.0];
    let mut rows = Vec::new();
    for d in distances {
        let mut r = Range::new(defs_with("rifle", def));
        stand(
            &mut r.world,
            TARGET,
            "Target",
            DVec3::new(0.0, 0.0, d),
            false,
        );
        r.world.mark_dirty();
        r.world.reindex_guids();
        r.world.propagate();
        r.bridge.sync_from_world(&r.world);
        r.arm(HERO, "rifle");
        r.aim(HERO, 0.0, 0.0);
        r.hold_trigger(HERO, true);
        r.step();
        r.hold_trigger(HERO, false);
        for _ in 0..80 {
            r.step();
            if r.health(TARGET).is_some() {
                break;
            }
        }
        let left = r.health(TARGET).unwrap_or(weapon::DEFAULT_VITALITY_J);
        let spent = weapon::DEFAULT_VITALITY_J - left;
        // The CLOSED FORM, written out here rather than called, so the arm and
        // the implementation are two independent statements of the doc §5.
        let t = ((d - 35.0) / (235.0 - 35.0)).clamp(0.0, 1.0);
        let want = 600.0 + (t * t * (3.0 - 2.0 * t)) * (600.0 * 0.6 - 600.0);
        rows.push((d, spent, want));
    }
    for (d, spent, want) in &rows {
        println!("at {d:>6.1} m the target lost {spent:8.3} J; the closed form says {want:8.3} J");
    }
    for (d, spent, want) in rows {
        assert!(
            (spent - want).abs() < 1.0,
            "at {d} m the world spent {spent} J and the doc's curve says {want} J"
        );
    }
}

/// **A shot at the head multiplies; a shot at the pelvis does not.**
///
/// One target, two aims, one weapon. The head sphere is 12 cm about the rig's
/// own `head` socket, or — for this fixture's rig-less capsules — the capsule's
/// top-sphere centre, which is the SAME door (`head_point`) with its second
/// answer.
///
/// **Mutation → red:** `HEAD_RADIUS_M = 0.0` makes the head shot spend exactly
/// what the pelvis one did and the multiplier assertion fails.
#[test]
fn a_shot_at_the_head_multiplies_and_one_at_the_pelvis_does_not() {
    let mut def = test_rifle();
    def.damage_j = 300.0;
    def.headshot_mult = 2.0;
    // Inside the threshold on purpose for the first pair: the head zone is a
    // property of the SHOT, not of the pool, and it must work for both halves.
    let cm = CharacterMovement::default();
    // `head_point`'s own capsule rule: a capsule stands with its centre at
    // `feet + h + r` and its top sphere `h` above that.
    let head_y = 2.0 * cm.stand_half_height_m + RADIUS;
    let pelvis_y = cm.stand_half_height_m;
    let at = 12.0;
    let spend = |aim_y: f64| -> (f64, bool) {
        let mut r = Range::new(defs_with("rifle", def));
        stand(
            &mut r.world,
            TARGET,
            "Target",
            DVec3::new(0.0, 0.0, at),
            false,
        );
        r.world.mark_dirty();
        r.world.reindex_guids();
        r.world.propagate();
        r.bridge.sync_from_world(&r.world);
        r.arm(HERO, "rifle");
        // The muzzle is at MUZZLE_Y; aim so the ray arrives at `aim_y`.
        let pitch = ((aim_y - MUZZLE_Y) / at).atan().to_degrees();
        r.aim(HERO, 0.0, pitch);
        r.hold_trigger(HERO, true);
        let rep = r.step();
        let hit = rep
            .hits
            .iter()
            .find(|h| h.target == Some(TARGET))
            .copied()
            .expect("the shot missed the target entirely");
        (
            weapon::DEFAULT_VITALITY_J - r.health(TARGET).expect("a body"),
            hit.headshot,
        )
    };
    let (head, head_flag) = spend(head_y);
    let (pelvis, pelvis_flag) = spend(pelvis_y);
    println!(
        "a shot at the head ({head_y:.3} m) spent {head} J (headshot {head_flag}); \
         one at the pelvis ({pelvis_y:.3} m) spent {pelvis} J (headshot {pelvis_flag})"
    );
    assert!(head_flag, "a shot at the head joint was not a headshot");
    assert!(
        !pelvis_flag,
        "a shot at the pelvis was recorded as a headshot"
    );
    assert!(
        (head - pelvis * 2.0).abs() < 1e-6,
        "the head shot spent {head} J and the pelvis shot {pelvis} J — the \
         multiplier is 2.0"
    );
}

/// …and it works for the FAR half too, which is the half that could be vacuous.
#[test]
fn a_headshot_past_the_threshold_multiplies_as_well() {
    let mut def = test_rifle();
    def.damage_j = 300.0;
    def.headshot_mult = 2.0;
    def.gravity_scale = 0.0;
    let cm = CharacterMovement::default();
    let head_y = 2.0 * cm.stand_half_height_m + RADIUS;
    let at = 120.0;
    let mut r = Range::new(defs_with("rifle", def));
    stand(
        &mut r.world,
        TARGET,
        "Target",
        DVec3::new(0.0, 0.0, at),
        false,
    );
    r.world.mark_dirty();
    r.world.reindex_guids();
    r.world.propagate();
    r.bridge.sync_from_world(&r.world);
    r.arm(HERO, "rifle");
    let pitch = ((head_y - MUZZLE_Y) / at).atan().to_degrees();
    r.aim(HERO, 0.0, pitch);
    r.hold_trigger(HERO, true);
    let first = r.step();
    r.hold_trigger(HERO, false);
    assert_eq!(first.rounds.spawned, 1, "the shot did not become a round");
    let mut headshots = 0u32;
    let mut spent = 0.0;
    for _ in 0..40 {
        let rep = r.step();
        headshots += rep.rounds.headshots;
        if r.health(TARGET).is_some() {
            spent = weapon::DEFAULT_VITALITY_J - r.health(TARGET).expect("a body");
            break;
        }
    }
    println!("a round that flew {at} m and hit the head spent {spent} J ({headshots} headshot(s))");
    assert_eq!(headshots, 1, "the flying round's head test never fired");
    assert!(
        (spent - 600.0).abs() < 1e-6,
        "a 300 J round with a 2.0 head multiplier spent {spent} J at the head"
    );
}

/// **A SHORTER body's head band is LOWER** — the WPN2a audit's arm, and the one
/// that decides whether the band is derived or chosen.
///
/// The wave's finding is that a headshot is a BAND and not a sphere, because a
/// ray stops on a capsule's surface 0.30 m from the joint it is tested against.
/// Both of the wave's own head arms fire at the SAME default capsule, so a
/// `head_point` that answered a constant 1.50 m would pass them both — and the
/// claim that the band comes off the character's own body rather than off a
/// number would be untested. That is exactly the shape of the thing the wave
/// itself caught one door along (an unreachable sphere), so it is worth an arm.
///
/// `head_point`'s capsule answer is `feet + 2·half_height_for(mode) + radius`,
/// which is a property of the TARGET. This arm stands a **1.40 m** body — half
/// height 0.40 against the default's 0.60 — and fires at its own head point
/// (1.10 m) and at its chest (0.50 m). A constant 1.50 m band is 0.40 m above
/// the first, so the headshot would not fire at all.
///
/// **Mutation → red:** any constant in `head_point`'s second answer makes the
/// head shot spend exactly what the chest shot did.
#[test]
fn a_shorter_bodys_head_band_is_lower() {
    let mut def = test_rifle();
    def.damage_j = 300.0;
    def.headshot_mult = 2.0;
    const SHORT_HALF: f64 = 0.4;
    let tall_head = 2.0 * CharacterMovement::default().stand_half_height_m + RADIUS;
    let short_head = 2.0 * SHORT_HALF + RADIUS;
    assert!(
        tall_head - short_head > 2.0 * ballistics::HEAD_RADIUS_M,
        "the default body's head point is {tall_head:.3} m and this one's is \
         {short_head:.3} m, and the band is {:.3} m tall — a constant would be \
         indistinguishable",
        2.0 * ballistics::HEAD_RADIUS_M
    );
    let at = 12.0;
    let spend = |aim_y: f64| -> (f64, bool) {
        let mut r = Range::new(defs_with("rifle", def));
        // The short body, stood up by hand: `stand`'s own bundle with one
        // number changed, because the height IS the subject.
        {
            let cm = CharacterMovement {
                stand_half_height_m: SHORT_HALF,
                ..Default::default()
            };
            let e = r.world.spawn_with_guid(TARGET, "Short", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(0.0, SHORT_HALF + RADIUS, at);
            r.world.world_mut().entity_mut(e).insert((
                RigidBody3D {
                    kind: BodyKind3D::Kinematic,
                    ..Default::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Capsule,
                    half_extents: Vec3d::new(RADIUS, SHORT_HALF, RADIUS),
                    radius: RADIUS,
                    ..Default::default()
                },
                CharacterController3D::default(),
                cm,
                t,
            ));
        }
        r.world.mark_dirty();
        r.world.reindex_guids();
        r.world.propagate();
        r.bridge.sync_from_world(&r.world);
        r.arm(HERO, "rifle");
        let pitch = ((aim_y - MUZZLE_Y) / at).atan().to_degrees();
        r.aim(HERO, 0.0, pitch);
        r.hold_trigger(HERO, true);
        let rep = r.step();
        let hit = rep
            .hits
            .iter()
            .find(|h| h.target == Some(TARGET))
            .copied()
            .expect("the shot missed the short body entirely");
        (
            weapon::DEFAULT_VITALITY_J - r.health(TARGET).expect("a body"),
            hit.headshot,
        )
    };
    let (head, head_flag) = spend(short_head);
    let (chest, chest_flag) = spend(SHORT_HALF + RADIUS - 0.2);
    println!(
        "a 1.40 m body (the default is 1.80): a shot at its OWN head point \
         ({short_head:.3} m, not the default's {tall_head:.3}) spent {head} J \
         (headshot {head_flag}); one at its chest spent {chest} J (headshot \
         {chest_flag})"
    );
    assert!(
        head_flag,
        "a shot at a short body's own head point was not a headshot — the band \
         is not derived from the body it is about"
    );
    assert!(
        !chest_flag,
        "a shot at the chest was recorded as a headshot"
    );
    assert!((head - chest * 2.0).abs() < 1e-6);
}

/// **The registry's CLASS census, read off the rows** — the WPN2a audit's arm.
///
/// `the_registry_is_the_docs_own_eighty_five_rows` sums a hard-coded
/// `[("pistols", 10), …]` and asserts the sum is 85, which is a constant the
/// compiler could fold: a registry of eighty-five pistols passes it as long as
/// seven named exemplars exist. This one partitions the PARSED rows by the one
/// field that is unique per class in `weapons.toml`'s own class-rule table —
/// `range_m` — and counts them.
///
/// **Mutation → red:** deleting any row, or moving one between classes, moves a
/// count.
#[test]
fn the_registrys_class_census_is_read_off_the_rows() {
    let mut defs = ItemDefs::default();
    assert_eq!(
        defs.merge_toml(weapon::WEAPON_REGISTRY_TOML)
            .expect("the registry parses"),
        85
    );
    // The class rule's own `range_m` column, which is unique per class — the
    // header comment of `weapons.toml` is where these seven numbers are stated.
    let class_of = |range_m: f64| -> &'static str {
        match range_m as i64 {
            150 => "pistols",
            180 => "smgs",
            500 => "assault rifles",
            800 => "dmrs",
            1800 => "snipers",
            80 => "shotguns",
            900 => "launchers",
            _ => "UNCLASSED",
        }
    };
    let mut census: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for (id, item) in defs.0.iter() {
        let w = item
            .weapon
            .unwrap_or_else(|| panic!("{id} is in the weapon registry and is not a weapon"));
        *census.entry(class_of(w.range_m)).or_default() += 1;
    }
    println!("the registry's census, off the rows themselves: {census:?}");
    for (class, want) in [
        ("pistols", 10usize),
        ("smgs", 20),
        ("assault rifles", 20),
        ("dmrs", 10),
        ("snipers", 10),
        ("shotguns", 10),
        ("launchers", 5),
    ] {
        assert_eq!(
            census.get(class).copied().unwrap_or(0),
            want,
            "the doc gives {want} {class} and the registry holds {}",
            census.get(class).copied().unwrap_or(0)
        );
    }
    assert_eq!(
        census.get("UNCLASSED").copied().unwrap_or(0),
        0,
        "a row carries a range no class rule names"
    );
    assert_eq!(census.values().sum::<usize>(), 85);
}

// ── (e) THE MOVE-SPEED CONSUMER ─────────────────────────────────────────────

/// **A sniper really is slower than a bare pair of hands** — measured in metres
/// covered, not in a field read back.
///
/// **Mutation → red:** dropping `equip_scale` out of `settings_for`'s product
/// makes the two distances equal.
#[test]
fn the_registrys_move_speed_multiplier_is_metres_on_the_ground() {
    let defs = {
        let mut d = ItemDefs::default();
        assert_eq!(
            d.merge_toml(weapon::WEAPON_REGISTRY_TOML)
                .expect("the registry parses"),
            85
        );
        d
    };
    let sniper = defs
        .get("barrett_m82")
        .and_then(|i| i.weapon)
        .expect("the registry has a Barrett");
    let run = |arm_with: Option<&str>| -> f64 {
        let mut r = Range::new(defs.clone());
        if let Some(id) = arm_with {
            r.arm(HERO, id);
        }
        let start = {
            let e = r.world.entity_of(HERO).expect("hero");
            r.world
                .world()
                .get::<Transform>(e)
                .expect("a transform")
                .translation
                .to_dvec3()
        };
        for _ in 0..180 {
            r.bridge.sync_from_world(&r.world);
            inf_ecs::movement::apply_intent(
                &mut r.world,
                &MovementIntent {
                    move_input: inf_ecs::math::Vec2d::new(0.0, 1.0),
                    ..Default::default()
                },
            );
            d3::step_character_movement(&mut r.world, &mut r.bridge, DT);
            d3::step_gameplay(&mut r.world, &mut r.bridge, DT);
            r.bridge.step(DT);
            r.bridge.write_back_into(&mut r.world);
            r.world.propagate();
        }
        let end = {
            let e = r.world.entity_of(HERO).expect("hero");
            r.world
                .world()
                .get::<Transform>(e)
                .expect("a transform")
                .translation
                .to_dvec3()
        };
        (end - start).length()
    };
    let bare = run(None);
    let with_sniper = run(Some("barrett_m82"));
    println!(
        "3 s of running: unarmed {bare:.4} m, with a Barrett M82 {with_sniper:.4} m \
         (the registry's multiplier is {:.2})",
        sniper.move_speed_mult
    );
    assert!(bare > 1.0, "the hero did not move at all unarmed");
    assert!(
        with_sniper < bare,
        "a 0.76 move multiplier covered {with_sniper:.4} m against an unarmed \
         {bare:.4} m — the multiplier reached no speed"
    );
    // Not a magic number: the ratio is the registry's own field, within the
    // acceleration ramp's slack over three seconds.
    let ratio = with_sniper / bare;
    assert!(
        (ratio - sniper.move_speed_mult).abs() < 0.05,
        "the distance ratio is {ratio:.4} and the registry says {:.4}",
        sniper.move_speed_mult
    );
}

// ── (f) THE REGISTRY ────────────────────────────────────────────────────────

/// The doc's own class census, as ids, so the count is a claim about the
/// dataset's seven sections rather than about a number 85.
const CLASS_CENSUS: [(&str, usize, &str); 7] = [
    ("pistols", 10, "glock_17"),
    ("smgs", 20, "mp5"),
    ("assault rifles", 20, "m4a1"),
    ("dmrs", 10, "svd_dragunov"),
    ("snipers", 10, "barrett_m82"),
    ("shotguns", 10, "remington_870"),
    ("launchers", 5, "rpg_7"),
];

/// **Eighty-five rows, each one parsed, counted per class and round-tripped
/// through the by-name door.**
///
/// The census is by CLASS and by a named exemplar of each, because "85 rows
/// parsed" is satisfied by eighty-five pistols.
#[test]
fn the_registry_is_the_docs_own_eighty_five_rows() {
    let mut defs = ItemDefs::default();
    let taken = defs
        .merge_toml(weapon::WEAPON_REGISTRY_TOML)
        .expect("the registry parses");
    assert_eq!(taken, 85, "the registry defines {taken} items, not 85");
    assert!(
        taken <= item::MAX_ITEM_DEFS,
        "the registry is over the catalogue cap"
    );
    // Every row is a weapon, and every row is a PROJECTILE — which is what
    // makes the hybrid the registry's behaviour rather than an option in it.
    let mut classes: Vec<(&str, usize)> = Vec::new();
    for (name, want, exemplar) in CLASS_CENSUS {
        let def = defs
            .get(exemplar)
            .unwrap_or_else(|| panic!("the registry has no `{exemplar}`"))
            .weapon
            .unwrap_or_else(|| panic!("`{exemplar}` is not a weapon"));
        assert_eq!(
            def.kind,
            ShotKind::Projectile,
            "`{exemplar}` is not a projectile — the hybrid would never engage for it"
        );
        classes.push((name, want));
    }
    let total: usize = classes.iter().map(|(_, n)| n).sum();
    println!("the registry's census: {classes:?} = {total}");
    assert_eq!(total, 85);

    // The BY-NAME DOOR: every settable name takes a value and refuses a NaN,
    // for every row, which is what `set`/`names()` being one door means.
    let names = WeaponDef::names();
    assert_eq!(names.len(), 24, "the door has {} names", names.len());
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, names, "`names()` is not sorted");
    for (id, item) in defs.0.iter() {
        let mut def = item
            .weapon
            .unwrap_or_else(|| panic!("{id} is not a weapon"));
        for n in names {
            assert!(def.set(n, 1.0), "{id}: the door does not know {n}");
            assert!(!def.set(n, f64::NAN), "{id}: {n} took a NaN");
        }
        assert!(
            !def.set("no_such_key", 1.0),
            "{id}: an unknown key was taken"
        );
    }

    // The doc's own eight metrics, re-derived for four named rows rather than
    // read back from the file that authored them.
    for (id, torso_hp, head_hp, rpm, v0, mag, move_mult) in [
        ("glock_17", 30.0, 48.0, 450.0, 375.0, 17u32, 0.98),
        ("m4a1", 30.0, 45.0, 800.0, 910.0, 30, 0.90),
        ("barrett_m82", 110.0, 250.0, 150.0, 853.0, 10, 0.76),
        ("svd_dragunov", 65.0, 115.0, 250.0, 830.0, 10, 0.84),
    ] {
        let d = defs.get(id).and_then(|i| i.weapon).expect(id);
        assert_eq!(
            d.damage_j,
            torso_hp * weapon::JOULES_PER_HIT_POINT,
            "{id}: the doc's {torso_hp} HP torso is {} J at 20 J/HP",
            torso_hp * weapon::JOULES_PER_HIT_POINT
        );
        assert!(
            (d.headshot_mult - head_hp / torso_hp).abs() < 1e-4,
            "{id}: the head multiplier is {} and the doc's ratio is {}",
            d.headshot_mult,
            head_hp / torso_hp
        );
        assert_eq!(d.rounds_per_minute, rpm, "{id}: rpm");
        assert_eq!(d.muzzle_speed_mps, v0, "{id}: muzzle velocity");
        assert_eq!(d.magazine, mag, "{id}: magazine");
        assert!(
            (d.move_speed_mult - move_mult).abs() < 1e-9,
            "{id}: move speed"
        );
        assert!(d.ads_time_ms > 0.0, "{id}: no ADS time for wave WPN2b");
        assert!(
            d.recoil_intensity > 0.0,
            "{id}: no recoil stat for wave WPN2b"
        );
        assert!(d.report_max_m > 0.0, "{id}: no report range");
    }
}

/// **An unknown sub-table is refused BY NAME, not skipped.**
///
/// The whole reason the sub-tables are read rather than ignored: a `[ballisitcs]`
/// typo that silently dropped a muzzle velocity would fire the default at a
/// designer who had authored a number.
#[test]
fn a_misspelled_sub_table_is_refused_by_name() {
    let mut defs = ItemDefs::default();
    let e = defs
        .merge_toml(concat!(
            "[x]\n",
            "[x.weapon]\n",
            "[x.weapon.ballisitcs]\n",
            "muzzle_speed_mps = 900.0\n",
        ))
        .expect_err("a misspelled sub-table was accepted");
    println!("the reader said: {e}");
    assert!(
        e.contains("ballisitcs"),
        "the refusal does not name the table: {e}"
    );
    assert!(
        e.contains("ballistics"),
        "the refusal does not say what the known tables are: {e}"
    );
    // …and an unknown KEY inside a known table is refused too.
    let mut defs = ItemDefs::default();
    let e = defs
        .merge_toml(concat!(
            "[x]\n",
            "[x.weapon]\n",
            "[x.weapon.ballistics]\n",
            "bullet_velocity = 900.0\n",
        ))
        .expect_err("an unknown key inside a known sub-table was accepted");
    assert!(e.contains("bullet_velocity"), "{e}");
    // The four the doc names are all known.
    for t in weapon::WEAPON_SUB_TABLES {
        let mut defs = ItemDefs::default();
        let src = format!("[x]\n[x.weapon]\n[x.weapon.{t}]\ndamage_j = 1.0\n");
        assert_eq!(
            defs.merge_toml(&src).expect("a known sub-table"),
            1,
            "[{t}] is not a known sub-table"
        );
    }
}

// ── (g) DETERMINISM ─────────────────────────────────────────────────────────

/// **The pool's bytes are at the TAIL and are EMPTY on a quiet level.**
///
/// The strongest form available without a committed hash to compare against:
/// inserting an EMPTY pool must not change `state_bytes` by one byte, and
/// inserting a round must. If the section were folded anywhere but the tail, or
/// folded a length prefix, or folded the counters, the first half would fail.
///
/// **Mutation → red:** folding `pool.spawned` into `round_state_bytes` makes an
/// empty pool non-empty and the first assertion fails — which is the exact
/// reason the counters are outside the fold.
#[test]
fn an_empty_pool_changes_no_trace_and_a_round_changes_every_one() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pack = cook_fixture(tmp.path());
    let mut sim = pack_sim(&pack);
    for _ in 0..60 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let quiet = sim.state_bytes();
    assert!(
        ballistics::round_pool(sim.world()).is_none(),
        "a level nobody has fired on has a pool"
    );
    // An EMPTY pool, inserted by hand: the section must still be empty.
    sim.world_mut()
        .world_mut()
        .insert_resource(ballistics::RoundPool::default());
    let with_empty_pool = sim.state_bytes();
    assert_eq!(
        quiet, with_empty_pool,
        "an empty pool moved the trace — every level committed before this wave \
         would have a different hash"
    );
    assert!(ballistics::round_state_bytes(sim.world()).is_empty());
    // One round, and the trace moves.
    let r = ballistics::Round {
        shooter: Uuid::from_u128(1),
        at: DVec3::new(1.0, 2.0, 3.0),
        velocity: DVec3::Z * 900.0,
        travelled_m: 25.0,
        age_s: 0.0,
        first_segment: true,
        def: WeaponDef::default(),
    };
    assert!(ballistics::spawn_round(sim.world_mut(), r, 0));
    let with_round = sim.state_bytes();
    assert_ne!(
        quiet, with_round,
        "a round in the air is invisible to the trace — two hosts that \
         integrated it differently would agree for eight seconds and then one \
         of them would kill somebody"
    );
    println!(
        "quiet {} bytes; +empty pool {} bytes; +one round {} bytes",
        quiet.len(),
        with_empty_pool.len(),
        with_round.len()
    );
    // …at the TAIL: everything the quiet trace held is still a prefix of it.
    assert!(
        with_round.starts_with(&quiet),
        "the round's bytes were not appended at the tail — a section inserted \
         before the thirteen frozen ones moves every committed hash in the tree"
    );
}

/// **PIE == shipping, and two independent cooks, over a PROJECTILE course.**
///
/// The fixture's own hero, armed from the registry, firing into the sky so the
/// rounds fly for the whole comparison rather than landing on the first step.
/// The trace compared is `state_bytes`, which now carries the pool.
#[test]
fn pie_equals_shipping_and_two_cooks_agree_over_a_projectile_course() {
    let a = tempfile::tempdir().expect("tempdir a");
    let b = tempfile::tempdir().expect("tempdir b");
    let pack_a = cook_fixture(a.path());
    let pack_b = cook_fixture(b.path());
    let ta = projectile_course(pack_sim(&pack_a));
    let tb = projectile_course(pack_sim(&pack_b));
    let tp = projectile_course(pie_sim());
    assert_eq!(ta.0.len(), tb.0.len());
    assert_eq!(ta.0.len(), tp.0.len());
    println!(
        "the projectile course: {} steps, {} round(s) minted, peak {} in flight",
        ta.0.len(),
        ta.1,
        ta.2
    );
    assert!(ta.1 > 0, "the course minted no rounds — it proves nothing");
    assert!(
        ta.2 > 0,
        "no round was ever in flight during the comparison"
    );
    for (i, (x, y)) in ta.0.iter().zip(tb.0.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: two independent cooks diverged");
    }
    for (i, (x, y)) in ta.0.iter().zip(tp.0.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: PIE and shipping diverged");
    }
    assert_eq!((ta.1, ta.2), (tb.1, tb.2));
    assert_eq!((ta.1, ta.2), (tp.1, tp.2));
}

/// Drive one sim through: settle, arm an M4A1 from the level's own registry,
/// aim up, hold the trigger, let the rounds fly. Answers
/// `(trace, rounds minted, peak in flight)`.
fn projectile_course(mut sim: inf_player::runtime_sim::RuntimeSim) -> (Vec<Vec<u8>>, u32, u32) {
    use inf_ecs::components::CharacterMovement as CM;
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    let mut trace = Vec::new();
    let mut minted = 0u32;
    let mut peak = 0u32;
    let push = |sim: &mut inf_player::runtime_sim::RuntimeSim,
                trace: &mut Vec<Vec<u8>>,
                minted: &mut u32,
                peak: &mut u32| {
        trace.push(sim.state_bytes());
        let r = sim.gameplay();
        *minted += r.rounds.spawned;
        *peak = (*peak).max(r.rounds.in_flight);
    };
    for _ in 0..40 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        push(&mut sim, &mut trace, &mut minted, &mut peak);
    }
    assert_eq!(
        inf_ecs::item::give(sim.world_mut(), hero, "m4a1", 1),
        0,
        "the fixture's catalogue has no m4a1 — the registry did not reach the level"
    );
    assert!(inf_physics::d3::gameplay::equip_weapon(
        sim.world_mut(),
        hero,
        "m4a1"
    ));
    {
        let e = sim.world().entity_of(hero).expect("the hero");
        if let Some(mut cm) = sim.world_mut().world_mut().get_mut::<CM>(e) {
            cm.runtime.aim_pitch_deg = 30.0;
        }
    }
    // The trigger goes through the SHIPPED input path — the left mouse button
    // resolved by the default map — rather than by writing `want_attack` on the
    // component: a course that set the intent by hand would prove the pool
    // works and nothing about the key a player presses.
    let mut state = inf_input::InputState::new(inf_input::default_map());
    for i in 0..200 {
        let events = [inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: i < 60,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
        push(&mut sim, &mut trace, &mut minted, &mut peak);
    }
    (trace, minted, peak)
}

// ── (h) THE WORLD AROUND THE HIT ────────────────────────────────────────────

/// **A projectile hit raises a witnessed act exactly as a hitscan hit does** —
/// the same `report.hits` door, and the same EMS3 log.
///
/// A hitscan hit on flesh at 10 m and a projectile hit on flesh at 120 m, each
/// with one bystander watching, and the act count is the same.
///
/// **Mutation → red:** not pushing the impact into `report.hits` in
/// `step_rounds` leaves the far half with zero acts.
#[test]
fn a_projectile_hit_is_witnessed_exactly_as_a_hitscan_hit_is() {
    let acts = |distance: f64, threshold: f64| -> usize {
        let mut def = test_rifle();
        def.hitscan_threshold_m = threshold;
        def.gravity_scale = 0.0;
        def.damage_j = 100.0;
        let mut r = Range::new(defs_with("rifle", def));
        stand(
            &mut r.world,
            TARGET,
            "Target",
            DVec3::new(0.0, 0.0, distance),
            false,
        );
        // **Somebody who can SEE it, and it has to be a CROWD agent**:
        // `witness::candidates_near` walks `CrowdPopulationRes` and nothing
        // else, so a standing capsule character is not a witness to anything.
        // Measured on this arm's own first draft, which put a character there
        // and read zero acts for both halves of the hybrid.
        {
            let a = inf_ecs::crowd::CrowdArchetype::humanoid(None, None, None);
            let mut records = std::collections::BTreeMap::new();
            records.insert(
                shooter_guid(9),
                inf_ecs::crowd::CrowdRecord::standing(a, DVec3::new(4.0, 0.0, distance * 0.5)),
            );
            assert_eq!(
                inf_ecs::crowd::add_agents(&mut r.world, records),
                0,
                "the witness was refused (the count is the REFUSALS)"
            );
        }
        r.world.mark_dirty();
        r.world.reindex_guids();
        r.world.propagate();
        r.bridge.sync_from_world(&r.world);
        r.arm(HERO, "rifle");
        r.aim(HERO, 0.0, 0.0);
        r.hold_trigger(HERO, true);
        // The FIRING step's report counts: a hitscan's whole act is raised on
        // it, and an arm that started counting on the next one would read zero
        // for the near half and call it a difference between the two halves.
        let mut witnessed = r.step().witnessed;
        r.hold_trigger(HERO, false);
        for _ in 0..60 {
            let rep = r.step();
            witnessed += rep.witnessed;
        }
        assert!(
            r.health(TARGET)
                .is_some_and(|j| j < weapon::DEFAULT_VITALITY_J),
            "the target at {distance} m was never hit"
        );
        witnessed as usize
    };
    let hitscan = acts(10.0, 25.0);
    let projectile = acts(120.0, 25.0);
    println!("a hitscan hit raised {hitscan} act(s); a projectile hit raised {projectile}");
    assert!(hitscan > 0, "a hitscan hit raised no act at all");
    assert!(
        projectile > 0,
        "a projectile hit raised no witnessed act — the far half is invisible to \
         EMS3, so a murder at 120 m would go on nobody's file"
    );
    // **EXACTLY as**, and this is the half `WeaponHit::arrived` exists for: a
    // projectile is TWO records, and without the flag its arrival lands in the
    // quiet-crime bucket and files a second act — an ASSAULT — against a shooter
    // whose `Shot` is already on the record.
    assert_eq!(
        hitscan, projectile,
        "the two halves of the hybrid file different numbers of acts for one \
         person being shot"
    );
}

/// **A PROJECTILE KILL IS FILED AGAINST THE SHOOTER** — and this is the arm the
/// one above could not be.
///
/// The count-of-acts comparison is **vacuous for the far half**, measured: a
/// projectile's `Shot` act is raised by the MUZZLE record on the firing step, so
/// deleting the impact from `report.hits` entirely leaves the count at one and
/// the arm green. What only the impact can produce is the `Killed` act's
/// **actor**: `step_witness` finds the killer with `hits.iter().find(|h|
/// h.target == Some(victim))`, so a round that killed somebody without
/// appearing in `report.hits` files the murder against `Uuid::nil()` — a crime
/// scene on nobody's file, which is exactly the defect the EMS3 audit fixed for
/// hitscans.
///
/// **Mutation → red:** not pushing the impact into `report.hits` in
/// `step_rounds` (which the arm above survives) makes the killer nil here.
#[test]
fn a_kill_at_range_names_the_shooter_and_not_nobody() {
    let mut def = test_rifle();
    def.gravity_scale = 0.0;
    // Enough to end a 2 000 J body in one round.
    def.damage_j = 2500.0;
    let mut r = Range::new(defs_with("rifle", def));
    let at = 120.0;
    stand(
        &mut r.world,
        TARGET,
        "Target",
        DVec3::new(0.0, 0.0, at),
        false,
    );
    {
        let a = inf_ecs::crowd::CrowdArchetype::humanoid(None, None, None);
        let mut records = std::collections::BTreeMap::new();
        records.insert(
            shooter_guid(9),
            inf_ecs::crowd::CrowdRecord::standing(a, DVec3::new(4.0, 0.0, at * 0.5)),
        );
        assert_eq!(inf_ecs::crowd::add_agents(&mut r.world, records), 0);
    }
    r.world.mark_dirty();
    r.world.reindex_guids();
    r.world.propagate();
    r.bridge.sync_from_world(&r.world);
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    let first = r.step();
    r.hold_trigger(HERO, false);
    assert_eq!(first.rounds.spawned, 1, "the shot did not become a round");
    let mut kills = 0u32;
    for _ in 0..60 {
        kills += r.step().kills;
    }
    assert_eq!(kills, 1, "the round at {at} m did not stop the body");
    let acts = inf_ecs::witness::witnessed(&r.world);
    let killed: Vec<&inf_ecs::witness::WitnessedAct> = acts
        .iter()
        .filter(|a| a.kind == inf_ecs::witness::ActKind::Killed)
        .collect();
    println!(
        "the log holds {} act(s); the killing(s): {:?}",
        acts.len(),
        killed
            .iter()
            .map(|a| (a.actor, a.observers.len()))
            .collect::<Vec<_>>()
    );
    assert_eq!(killed.len(), 1, "the killing was not recorded");
    assert_eq!(
        killed[0].actor, HERO,
        "a murder at {at} m was filed against {} instead of the shooter — the \
         projectile's impact never reached `report.hits`, so nothing in this \
         step could say who did it",
        killed[0].actor
    );
}

/// **A round STOPS IN a parked car, and the car spends nothing** — the WPN2a
/// audit's closure of carried 200, and VEH3c's half said out loud beside it.
///
/// Until this audit the shot's cast door filtered to `CastTargets::Fixed`, so
/// neither half of the hybrid could SEE a dynamic chassis: a bullet flew through
/// a car, a crate, a fractured chunk and a body on the floor, and this arm
/// asserted that it did. In a GTA-parity game a round must stop at a car. The
/// damage to the car is still VEH3c's — a chassis has no `Health` and no
/// `Destructible`, so the honest outcome is a round that ENDS there and spends
/// nothing, which is what is measured.
///
/// **Mutation → red:** putting the cast back to `cast_ray_excluding` (`Fixed`)
/// makes the round fly through and `stopped` is `None`.
#[test]
fn a_round_stops_in_a_parked_car_and_the_car_spends_nothing() {
    let mut r = Range::new(defs_with("rifle", test_rifle()));
    {
        let e = r.world.spawn_with_guid(CHASSIS, "Car", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 1.0, 80.0);
        r.world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(1.0, 1.0, 2.5),
                ..Default::default()
            },
            t,
        ));
        r.world.mark_dirty();
        r.world.reindex_guids();
        r.world.propagate();
        r.bridge.sync_from_world(&r.world);
    }
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    r.step();
    r.hold_trigger(HERO, false);
    let mut stopped = None;
    let mut owed = 0usize;
    let mut furthest = 0.0_f64;
    for _ in 0..60 {
        for round in r.rounds() {
            furthest = furthest.max(round.at.z);
        }
        let rep = r.step();
        owed += rep.destruct.len();
        if let Some(h) = rep.hits.iter().find(|h| h.target == Some(CHASSIS)) {
            stopped = Some(*h);
        }
    }
    let hit = stopped.expect(
        "a round fired at a parked car flew straight through it -- the cast door \
         is back to `CastTargets::Fixed` and carried 200 is open again",
    );
    println!(
        "a round flown at a parked car: stopped at z {:.3} (the near face is 77.5 m), \
         on_flesh {}, {} J of energy carried, {owed} entr(ies) owed at the P22 door, \
         health after: {:?} -- a chassis has no `Health` and no `Destructible`, so it \
         spends NOTHING, which is VEH3c's",
        hit.to.z,
        hit.on_flesh,
        hit.energy_j,
        weapon::health_of(&r.world, CHASSIS).map(|h| h.joules),
    );
    assert!(
        (hit.to.z - 77.5).abs() < 0.5,
        "the round stopped at z {:.3} and the car's near face is 77.5 m",
        hit.to.z
    );
    assert!(!hit.on_flesh, "a car is not flesh");
    assert!(
        furthest < 80.0,
        "a round reached {furthest:.2} m, which is past the car it should have \
         stopped in"
    );
    // …and NOTHING is spent on it. Both doors, because they are the two ways a
    // hit can cost something: the health door and the P22 destructible door.
    assert_eq!(owed, 0, "a car owed joules at the P22 door");
    assert!(
        weapon::health_of(&r.world, CHASSIS).is_none(),
        "a car grew a `Health` from being shot -- that is VEH3c's decision, not \
         this wave's"
    );
    assert!(r.rounds().is_empty(), "the pool did not empty");
}

/// **A round hits a body on the FLOOR** — the second half of carried 200, and
/// the one that decides whether a ragdolled person is bulletproof.
///
/// A ragdolling character's own capsule is DISABLED and its limbs are dynamic
/// articulated bodies the bridge attached itself. Two things had to be true for
/// a round to hurt one and neither was: the cast had to be able to see a dynamic
/// body (it filtered to `Fixed`), and the collider it hit had to name the
/// character (ragdoll limbs are in no document entity's row, so
/// `guid_of_collider` answered `None`, `is_flesh` answered `false`, and the
/// joules went nowhere).
///
/// **Mutation → red:** dropping `guid_of_ragdoll_collider` out of `hit_owner`
/// leaves the target at full health with the round stopped in mid-air on a limb
/// that belongs to nobody.
#[test]
fn a_round_into_a_body_on_the_floor_spends_its_joules() {
    let mut def = test_rifle();
    def.damage_j = 400.0;
    // Inside the threshold: the ragdoll is 8 m away and what is under test is
    // WHO the cast named, which is the same door for both halves.
    let mut r = Range::new(defs_with("rifle", def));
    r.install_rig();
    stand_rigged(&mut r.world, TARGET, "Target", DVec3::new(0.0, 0.0, 8.0));
    r.world.mark_dirty();
    r.world.reindex_guids();
    r.world.propagate();
    r.bridge.sync_from_world(&r.world);
    // Two steps for the pose to publish a rig and the bridge to build bodies.
    r.step();
    assert!(
        d3::ragdoll_bridge::start_ragdoll(&mut r.world, TARGET),
        "the target would not ragdoll"
    );
    let mut spawned = false;
    for _ in 0..6 {
        r.step();
        if r.bridge.ragdoll_count() > 0 {
            spawned = true;
            break;
        }
    }
    assert!(
        spawned,
        "no ragdoll bodies were built -- this arm is vacuous"
    );
    let limbs = r
        .bridge
        .ragdoll_of(TARGET)
        .map(|s| s.colliders.len())
        .unwrap_or(0);
    assert!(limbs >= 7, "a ragdoll of {limbs} colliders is not a body");
    assert_eq!(
        r.world
            .entity_of(TARGET)
            .and_then(|e| r.world.world().get::<CharacterMovement>(e))
            .map(|cm| cm.mode),
        Some(MovementMode::Ragdoll),
        "the target is not ragdolling"
    );
    // Fire at it, level, and keep firing while it settles: a body on the floor
    // is a moving target and the arm is about whether it can be hit at all.
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, -6.0);
    let mut spent = 0.0;
    let mut named = 0usize;
    for _ in 0..90 {
        r.hold_trigger(HERO, true);
        let rep = r.step();
        named += rep.hits.iter().filter(|h| h.target == Some(TARGET)).count();
        if let Some(h) = weapon::health_of(&r.world, TARGET) {
            spent = weapon::DEFAULT_VITALITY_J - h.joules;
        }
        if spent > 0.0 {
            break;
        }
    }
    println!(
        "a ragdolled body of {limbs} limbs was named by {named} hit(s) and lost \
         {spent} J"
    );
    assert!(
        named > 0,
        "every round went through a body lying on the floor -- carried 200's \
         ragdoll half is open"
    );
    assert!(
        spent > 0.0,
        "a ragdolled body was hit {named} time(s) and lost no joules at all"
    );
}

// ── (i) COST ────────────────────────────────────────────────────────────────

/// **What a full pool costs the `gameplay` phase.**
///
/// Reported everywhere, asserted under `--release` off CI — `CITY_STEP_BUDGET_MS`'s
/// conditioning, for its reasons. The DELTA is what makes the number
/// attributable: the same world, the same step, with and without rounds in the
/// air.
#[test]
fn a_full_pool_costs_what_it_costs() {
    let mut def = test_rifle();
    def.rounds_per_minute = 900.0;
    def.range_m = 4000.0;
    let mut r = Range::new(defs_with("rifle", def));
    for i in 0..8 {
        stand(
            &mut r.world,
            shooter_guid(i),
            "Shooter",
            DVec3::new(i as f64 * 4.0, 0.0, 0.0),
            false,
        );
    }
    r.world.mark_dirty();
    r.world.reindex_guids();
    r.world.propagate();
    r.bridge.sync_from_world(&r.world);
    // The quiet baseline FIRST, on the same world, before anybody is armed.
    let quiet = time_steps(&mut r, 60);
    for i in 0..8 {
        r.arm(shooter_guid(i), "rifle");
        r.aim(shooter_guid(i), 0.0, 30.0);
        r.hold_trigger(shooter_guid(i), true);
    }
    // Fill the pool.
    let mut peak = 0u32;
    for _ in 0..90 {
        peak = peak.max(r.step().rounds.in_flight);
    }
    assert!(
        peak as usize >= MAX_ROUNDS_IN_FLIGHT / 2,
        "the pool only reached {peak} rounds — this is not a measurement of a \
         full pool"
    );
    let busy = time_steps(&mut r, 60);
    let live = r.rounds().len();
    let per_round_us = if live > 0 {
        (busy - quiet) * 1000.0 / live as f64
    } else {
        0.0
    };
    println!(
        "the gameplay phase on this range ({} build): {quiet:.4} ms quiet, {busy:.4} ms with \
         {live} round(s) in flight ({} segment casts a step) — the pool's own share is \
         {:.4} ms, {per_round_us:.2} us a round, against a {} ms ceiling at {} rounds",
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        },
        live * PROJECTILE_SUB_STEPS as usize,
        busy - quiet,
        inf_player::budget::WEAPON_STEP_BUDGET_MS,
        inf_player::budget::WEAPON_BUDGET_ROUNDS
    );
    assert!(live > 0, "the budget was measured with an empty pool");
    if cfg!(debug_assertions) {
        eprintln!("dev build: the weapon budget is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the weapon budget is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        busy <= inf_player::budget::WEAPON_STEP_BUDGET_MS,
        "the gameplay phase cost {busy:.4} ms with {live} rounds in flight against a \
         {} ms ceiling {}",
        inf_player::budget::WEAPON_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

/// The mean millisecond of `n` gameplay steps, minus the rest of the fixed step.
fn time_steps(r: &mut Range, n: u32) -> f64 {
    // MIN of rounds is the discipline everywhere else in this tree; a phase is
    // one call, so the mean over `n` of them is what stands in for it.
    let mut total = std::time::Duration::ZERO;
    for _ in 0..n {
        r.bridge.sync_from_world(&r.world);
        inf_ecs::movement::apply_intent(&mut r.world, &MovementIntent::default());
        d3::step_character_movement(&mut r.world, &mut r.bridge, DT);
        let t0 = std::time::Instant::now();
        d3::step_gameplay(&mut r.world, &mut r.bridge, DT);
        total += t0.elapsed();
        r.bridge.step(DT);
        r.bridge.write_back_into(&mut r.world);
        r.world.propagate();
    }
    total.as_secs_f64() * 1000.0 / f64::from(n)
}

// ── the fixture plumbing (the phase30 gate's own, so both read one level) ────

fn sample_files() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(inf_editor_core::samples::gameplay_dir())
        .expect("the committed fixture is there")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .collect();
    v.sort();
    v
}

fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Island Gameplay", "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("a content root");
    for f in sample_files() {
        std::fs::copy(
            inf_editor_core::samples::gameplay_dir().join(&f),
            content.join(&f),
        )
        .expect("copy");
    }
    proj
}

fn cook_fixture(tmp: &Path) -> PathBuf {
    let proj = scaffold(tmp);
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the fixture cooks");
    out
}

fn pack_sim(pack: &Path) -> inf_player::runtime_sim::RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let mut sim = inf_player::sim_from_built(built);
    let reader = std::sync::Arc::new(
        inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
            .expect("the pack reader opens"),
    );
    inf_player::fracture::attach_fractures(
        &mut sim,
        &inf_player::fracture::FractureRegistry::from_pack(reader),
    );
    sim
}

fn read_asset(dir: &Path, guid: Uuid) -> Option<Vec<u8>> {
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|e| e == "toml") {
            continue;
        }
        if let Ok(side) = inf_asset::AssetSidecar::load(&path) {
            if side.guid.0 == guid {
                return std::fs::read(&path).ok();
            }
        }
    }
    None
}

/// **The PIE side**, through `sim_from_payload` — the ONE PIE boot seam the real
/// `--pie` subprocess takes. `phase30_gameplay_gate`'s own, so both gates boot
/// one level by one route.
fn pie_sim() -> inf_player::runtime_sim::RuntimeSim {
    let dir = inf_editor_core::samples::gameplay_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Gameplay.inf_lvl"))
        .expect("the level loads");
    let class = inf_editor_core::samples::gameplay_controller();
    let pcg = read_asset(&dir, inf_editor_core::samples::GAMEPLAY_PCG_GUID)
        .expect("the house graph is on disk");
    let mesh = read_asset(&dir, inf_editor_core::samples::GAMEPLAY_TARGET_MESH_GUID)
        .expect("the target mesh is on disk");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_ACTOR_GUID).then(|| class.clone()),
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_PCG_GUID).then(|| pcg.clone()),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_TARGET_MESH_GUID).then(|| mesh.clone()),
        |_| None,
        |_| None,
        60,
        false,
    )
    .expect("the payload builds");
    assert_eq!(
        payload.classes.len(),
        1,
        "the author class must ride the wire"
    );
    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
}

// ── (j) THE ISLAND'S OWN WEAPON DOOR (the WPN2a audit's closure of 204) ──────

/// **THE ISLAND ARMS ITSELF, WITH NO ENV VAR** — carried 204, closed and
/// measured on the committed island level.
///
/// Wave WPN2a's registry reached a LEVEL through the `phase30-gameplay`
/// fixture's `item.define` node, and reached the ISLAND through
/// `INF_PIE_ARM_HERO` — a dev-only preview env var read by the demo loop. A
/// player who booted the showcase had no weapon and no way to get one. The
/// island now carries a level Blueprint of its own
/// (`inf_editor_core::island::island_author_class`) which defines the registry
/// and puts one sidearm on the kerb the hero starts beside.
///
/// **Everything below goes through a SHIPPED door.** The level is the committed
/// `IslandFixture.inf_lvl`, loaded by the player's own loose loader; the pickup
/// is spawned by the level's own `BeginPlay`; the E key is
/// `RuntimeInput::with_down(["interact"])`, which is the same
/// `MovementIntent::from_actions` the window host feeds; the trigger is
/// `["attack"]`. No `INF_PIE_*` is read and no component is written by hand
/// except the aim, which is the mouse.
///
/// **Mutation → red:** dropping the `item.spawn_pickup` call from the class
/// leaves `on_the_kerb` `None`; dropping the `item.define` call in front of it
/// makes `spawn_pickup` refuse the id and does the same.
#[test]
fn the_island_puts_a_registry_weapon_on_the_kerb_and_the_hero_picks_it_up() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let content = island_fixture_project(tmp.path());
    assert!(
        content.join("IslandFixtureAuthor.inf_act").is_file(),
        "the recipe's `[content]` list does not carry the island's own author \
         class, so a BUILT project has no weapon catalogue"
    );
    let mut sim = island_fixture_sim(&content);
    // Settle: the class's `BeginPlay` runs on the entity's first tick.
    for _ in 0..30 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a hero");

    // ── the kerb ────────────────────────────────────────────────────────────
    let on_the_kerb = pickups(sim.world());
    println!("the island's pickups after 30 steps: {on_the_kerb:?}");
    let (pickup_guid, pickup_id, pickup_at) = on_the_kerb
        .into_iter()
        .next()
        .expect("the island spawned no pickup at all -- carried 204 is open");
    assert_eq!(
        pickup_id,
        inf_editor_core::island::ISLAND_SIDEARM_ID,
        "the island put a `{pickup_id}` on the kerb"
    );
    // It is a REGISTRY row, with the registry's own numbers — not a definition
    // the island invented.
    let def = item::item_defs(sim.world())
        .and_then(|d| d.get(&pickup_id))
        .and_then(|d| d.weapon)
        .expect("the sidearm on the kerb is not a weapon in the level's catalogue");
    assert_eq!(def.kind, ShotKind::Projectile);
    assert_eq!(def.damage_j, 30.0 * weapon::JOULES_PER_HIT_POINT);
    assert_eq!(def.magazine, 17, "the doc's Glock 17 carries seventeen");
    // …and it is in REACH of where the hero actually stands.
    let feet = hero_feet(sim.world(), hero).expect("the hero has feet");
    let reach = (pickup_at - feet).length();
    println!(
        "the sidearm `{pickup_id}` ({pickup_guid}) lies {reach:.2} m from the hero's feet \
         (the E key's own reach is {:.1} m)",
        inf_ecs::interact::DEFAULT_REACH_M
    );
    assert!(
        reach < inf_ecs::interact::DEFAULT_REACH_M,
        "the sidearm is {reach:.2} m away and the E key reaches {:.1} m",
        inf_ecs::interact::DEFAULT_REACH_M
    );

    // ── the E key ───────────────────────────────────────────────────────────
    assert!(
        weapon::equipped_def(sim.world(), hero).is_none(),
        "the hero starts the level already holding something"
    );
    let mut in_the_bag = 0u32;
    for i in 0..40 {
        let input = if i % 4 == 0 {
            inf_player::runtime_sim::RuntimeInput::with_down(["interact"])
        } else {
            inf_player::runtime_sim::RuntimeInput::default()
        };
        sim.step_once(input);
        in_the_bag = item::inventory_of(sim.world(), hero)
            .map(|inv| inv.count_of(&pickup_id))
            .unwrap_or(0);
        if in_the_bag > 0 {
            break;
        }
    }
    println!("after the E key the hero's bag holds {in_the_bag} x `{pickup_id}`");
    assert_eq!(
        in_the_bag, 1,
        "the hero pressed the shipped interact key beside a pickup for forty steps and \
         the bag is still empty"
    );
    assert!(
        sim.world().entity_of(pickup_guid).is_none(),
        "the pickup is still lying there after it was taken"
    );

    // -- the wheel --------------------------------------------------------
    //
    // Picking a weapon up puts it in the BAG; equipping it is the shipped
    // `weapon_switch` verb, which is the scroll wheel and reaches the intent as
    // an axis. Deliberately not a component write: the whole claim is that every
    // step of this is a control a player has.
    let mut equipped = None;
    for _ in 0..40 {
        sim.step_once(
            inf_player::runtime_sim::RuntimeInput::default().axis_at("weapon_switch", 1.0),
        );
        if let Some(d) = weapon::equipped_def(sim.world(), hero) {
            equipped = Some(d);
            break;
        }
    }
    let held = equipped.expect(
        "the hero turned the shipped weapon wheel for forty steps with a pistol in the \
         bag and is holding nothing",
    );
    assert_eq!(
        held.1.magazine, def.magazine,
        "a different weapon was equipped"
    );

    // ── the trigger ─────────────────────────────────────────────────────────
    //
    // Aimed level and a touch up, so the shot leaves the muzzle rather than
    // into the kerb the hero is standing on.
    {
        let e = sim.world().entity_of(hero).expect("the hero");
        let mut cm = sim
            .world_mut()
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.aim_pitch_deg = 6.0;
    }
    let mut shots = 0u32;
    for _ in 0..40 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::with_down(["attack"]));
        shots += sim.gameplay().shots;
        if shots > 0 {
            break;
        }
    }
    println!(
        "the hero fired {shots} shot(s) of a `{pickup_id}` on the island through the \
         window's own input door, with no environment variable set"
    );
    assert!(
        shots > 0,
        "the hero held the shipped attack key for forty steps with a registry weapon \
         equipped and fired nothing"
    );
}

/// Every `ItemPickup` in the world, as `(guid, id, world position)`.
fn pickups(world: &EcsWorld) -> Vec<(Uuid, String, DVec3)> {
    let mut out = Vec::new();
    let w = world.world();
    for (guid, pickup, t) in w
        .iter_entities()
        .filter_map(|e| {
            Some((
                e.get::<inf_ecs::components::Guid>()?,
                e.get::<inf_ecs::item::ItemPickup>()?,
                e.get::<Transform>()?,
            ))
        })
        .collect::<Vec<_>>()
    {
        out.push((guid.0, pickup.id.clone(), t.translation.to_dvec3()));
    }
    out.sort_by_key(|(g, _, _)| *g);
    out
}

/// Where a character's feet are, from its transform and its own capsule.
fn hero_feet(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    let e = world.entity_of(guid)?;
    let w = world.world();
    let cm = w.get::<CharacterMovement>(e)?;
    let t = w.get::<Transform>(e)?;
    let radius = w.get::<Collider3D>(e).map(|c| c.radius).unwrap_or(RADIUS);
    Some(t.translation.to_dvec3() - DVec3::Y * (cm.half_height_for(cm.mode) + radius))
}

/// **Build the CI-scale island the way `inf island build` does** and answer its
/// `Content` — `island_gate::build_project`'s own door, so this arm also proves
/// the recipe's `[content]` list carries the author class into a real project.
///
/// It is a build rather than a read of `samples/island-fixture` for one reason
/// the first draft of this arm found the hard way: the committed folder ships no
/// `.inf_terrain` (it is derived, and gitignored), so a hero loaded off it FALLS
/// FOR EVER — and a character in `FallFree` never takes the interact edge, which
/// `step_one` only reads on a grounded step. The arm read "the bag is empty" and
/// the cause was that there was no ground.
fn island_fixture_project(tmp: &Path) -> PathBuf {
    let recipe =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture/island.toml");
    let recipe = inf_island::IslandRecipe::load(&recipe).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let proj = tmp.join("island");
    ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    inf_island::write_content(&build, &content).expect("the island's content writes");
    content
}

/// The built island fixture, in the SHIPPING host's own loose loader —
/// `island_gate::loose_sim`'s shape, cut to what this arm reads.
fn island_fixture_sim(content: &Path) -> inf_player::runtime_sim::RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join("IslandFixture.inf_lvl"));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    // **The persisted binding map**, which is how an `ActorClass` on an entity
    // finds its class: `with_defaults` alone is a FALLBACK list, and the island's
    // author entity names its class by asset GUID.
    .with_bindings(inf_player::level::load_actor_classes_by_guid_from_dir(
        content,
    ))
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the island fixture loads");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

/// A tiny guard so the unused-import lint cannot fire on a set this file grows.
#[test]
fn the_gate_names_the_constants_it_is_about() {
    let _ = BTreeSet::<u8>::new();
    println!(
        "sub-steps {PROJECTILE_SUB_STEPS}, ray ceiling {MAX_SHOT_RAYS_PER_STEP}, pool bound \
         {MAX_ROUNDS_IN_FLIGHT}, gravity {PROJECTILE_GRAVITY_MPS2} m/s^2"
    );
    assert_eq!(
        MAX_ROUNDS_IN_FLIGHT * PROJECTILE_SUB_STEPS as usize,
        MAX_SHOT_RAYS_PER_STEP,
        "the pool bound is no longer derived from the ray ceiling"
    );
}
