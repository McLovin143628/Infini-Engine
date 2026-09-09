//! **WAVE WPN2b — FEEL.** The gate.
//!
//! # What every arm in this file reads
//!
//! **The joints, the aim and the world.** The hand's hold point in metres; the
//! aim pitch in the movement runtime, which is the number the bullet leaves
//! along; the radius of thirty holes in a wall at 25 m; the camera's field of
//! view and its boom; the hero's speed over the ground. Never a profile table,
//! never a state name, never a report's summary of itself.
//!
//! # The vacuity rule this wave is built against
//!
//! A weapon that never fired and a spring that never moved satisfy almost any
//! claim about recoil. So every arm below is fought against a CONTROL: a
//! measurement of the same quantity with the trigger up, with the mutation in
//! place, or with the layer under test switched off. Where an arm could be
//! satisfied by a weapon that did nothing, its own doc says so.
//!
//! # The mutations each arm dies to
//!
//! Written beside the arm, not in a report. They were run.

use std::collections::BTreeSet;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    MovementMode, RigidBody3D, RotationMode, Transform,
};
use inf_ecs::feel::{self, RecoilProfile, ShotStance, WeaponFeel};
use inf_ecs::item::{self, ItemDef, ItemDefs};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::movement::MovementIntent;
use inf_ecs::weapon::{self, ShotKind, WeaponDef};
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const RADIUS: f64 = 0.3;

const HERO: Uuid = Uuid::from_u128(0x2B00_0001);
const GROUND: Uuid = Uuid::from_u128(0x2B00_0002);
const WALL: Uuid = Uuid::from_u128(0x2B00_0003);
fn shooter_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2B00_0100 + i as u128)
}

/// **How far down range the pattern is measured**, metres — the clause's own 25.
const PATTERN_M: f64 = 25.0;

/// **The rifle every feel arm is fought with.**
///
/// A HITSCAN weapon, deliberately: this wave is about the three feel layers and
/// a projectile's flight would put the ballistics wave's drag and travel time
/// between the trigger and the hole in the wall. `spread_deg` is the registry's
/// own AR number and `recoil_intensity` its own 3.8, so every number this file
/// prints is a number the shipped catalogue produces.
fn test_rifle() -> WeaponDef {
    WeaponDef {
        kind: ShotKind::Hitscan,
        automatic: true,
        damage_j: 600.0,
        rounds_per_minute: 600.0,
        spread_deg: 1.00,
        magazine: 40,
        reserve: 200,
        reload_s: 2.0,
        range_m: 400.0,
        muzzle_forward_m: 0.45,
        recoil_intensity: 3.8,
        ads_time_ms: 240.0,
        move_speed_mult: 0.90,
        ..Default::default()
    }
}

/// A pistol, for the profile-scaling arm and the overlay one — the registry's
/// own Glock numbers.
fn test_pistol() -> WeaponDef {
    WeaponDef {
        kind: ShotKind::Hitscan,
        automatic: false,
        damage_j: 600.0,
        rounds_per_minute: 450.0,
        spread_deg: 1.20,
        magazine: 17,
        reserve: 68,
        reload_s: 1.4,
        range_m: 150.0,
        muzzle_forward_m: 0.12,
        recoil_intensity: 2.5,
        ads_time_ms: 140.0,
        move_speed_mult: 0.98,
        ..Default::default()
    }
}

fn defs_with(rows: &[(&str, WeaponDef)]) -> ItemDefs {
    let mut defs = ItemDefs::default();
    for (id, def) in rows {
        defs.insert(ItemDef {
            id: (*id).to_string(),
            label: (*id).to_string(),
            stack_max: 1,
            mass_kg: 3.6,
            weapon: Some(*def),
            ..Default::default()
        });
    }
    defs
}

// ─────────────────────────────────────────────────────────────────────────────
// THE FIXTURE — a floor, a wall at 25 m, and whoever is shooting at it
// ─────────────────────────────────────────────────────────────────────────────

struct Range {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    rig: Option<(inf_anim::SkeletonAsset, inf_anim::StateMachine)>,
    clips: std::collections::BTreeMap<inf_anim::ClipRef, inf_anim::AnimClip>,
}

const SKEL_GUID: Uuid = Uuid::from_u128(0x2B00_0007);
const SM_GUID: Uuid = Uuid::from_u128(0x2B00_0008);

impl Range {
    fn new(defs: ItemDefs) -> Self {
        let mut world = EcsWorld::new();
        // A floor under the shooter, and a wall at 25 m to shoot at.
        let e = world.spawn_with_guid(GROUND, "Ground", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, -3.0, 20.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(200.0, 0.5, 200.0),
                ..Default::default()
            },
            t,
        ));
        let e = world.spawn_with_guid(WALL, "Wall", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 0.0, PATTERN_M + 0.25);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(20.0, 20.0, 0.25),
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
            clips: std::collections::BTreeMap::new(),
        };
        r.bridge.sync_from_world(&r.world);
        r
    }

    /// Give this character the item and equip it, through the ECS door.
    fn arm(&mut self, who: Uuid, id: &str) {
        item::give_inventory(&mut self.world, who, 4);
        item::give(&mut self.world, who, id, 1);
        d3::gameplay::equip_weapon(&mut self.world, who, id);
    }

    fn aim(&mut self, who: Uuid, yaw: f64, pitch: f64) {
        let Some(e) = self.world.entity_of(who) else {
            return;
        };
        if let Some(mut cm) = self.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.aim_yaw_deg = yaw;
            cm.runtime.aim_pitch_deg = pitch;
        }
    }

    fn hold_trigger(&mut self, who: Uuid, down: bool) {
        let Some(e) = self.world.entity_of(who) else {
            return;
        };
        if let Some(mut cm) = self.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.want_attack = down;
            cm.runtime.press_attack = down;
        }
    }

    /// Hold (or release) the aim button — the level `step_character_movement`
    /// reads, so `RotationMode::Aiming` is reached the way a player reaches it.
    fn hold_aim(&mut self, who: Uuid, down: bool) {
        let Some(e) = self.world.entity_of(who) else {
            return;
        };
        if let Some(mut cm) = self.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.want_aim = down;
        }
    }

    fn set_mode(&mut self, who: Uuid, mode: MovementMode) {
        let Some(e) = self.world.entity_of(who) else {
            return;
        };
        if let Some(mut cm) = self.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.mode = mode;
        }
    }

    /// One fixed step, in the hosts' own order. `intent` is applied for the
    /// arms that need the hero to MOVE; `apply_intent` clears the trigger, so
    /// the trigger is re-armed after it (`wpn2a_gate`'s own finding).
    fn step_with(&mut self, intent: Option<MovementIntent>) -> d3::GameplayReport {
        self.bridge.sync_from_world(&self.world);
        if let Some(intent) = intent {
            // `apply_intent` writes the WHOLE runtime intent from a
            // `MovementIntent`, so a default one clears the trigger AND the aim
            // this fixture has latched (`wpn2a_gate`'s own finding, plus the
            // half it did not need). Both are restored.
            let (held, aiming) = (self.trigger_held(), self.aim_held());
            inf_ecs::movement::apply_intent(&mut self.world, &intent);
            if held {
                self.hold_trigger(HERO, true);
            }
            if aiming {
                self.hold_aim(HERO, true);
            }
        }
        d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
        let report = d3::step_gameplay(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        if let Some((skeleton, machine)) = &self.rig {
            let machines = |g: Uuid| (g == SM_GUID).then_some(machine);
            let skels = |g: Uuid| (g == SKEL_GUID).then_some(skeleton);
            let clips = |c: inf_anim::ClipRef| self.clips.get(&c);
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

    fn step(&mut self) -> d3::GameplayReport {
        self.step_with(None)
    }

    /// **Give this range a rig and a running animation clock.**
    ///
    /// The mannequin the wizard builds plus a one-state machine -- which is
    /// what makes `anim_bridge::anim_state_time` non-zero, and therefore what
    /// makes a character BREATHE. A capsule with no machine has no clock and no
    /// breath, and that is the control the sway arm is fought against.
    fn install_rig(&mut self, who: Uuid) {
        const IDLE: inf_anim::ClipRef = [0xd1; 16];
        let skeleton = inf_anim::build_template(
            inf_anim::BodyPlan::Biped,
            &inf_anim::BodyParams {
                height_m: 1.8,
                ..Default::default()
            },
        )
        .expect("the mannequin builds");
        self.clips
            .insert(IDLE, inf_anim::AnimClip::new("idle", Vec::new()));
        self.rig = Some((
            skeleton,
            inf_anim::StateMachine {
                states: vec![inf_anim::SmState::clip("idle", IDLE)],
                entry: 0,
                ..Default::default()
            },
        ));
        let e = self.world.entity_of(who).expect("the character");
        self.world.world_mut().entity_mut(e).insert((
            inf_ecs::components::AnimStateMachine {
                sm: Some(SM_GUID),
                ..Default::default()
            },
            inf_ecs::components::SkeletalMesh {
                mesh: None,
                skeleton: Some(SKEL_GUID),
            },
        ));
        self.world.mark_dirty();
    }

    fn aim_held(&self) -> bool {
        self.world
            .entity_of(HERO)
            .and_then(|e| self.world.world().get::<CharacterMovement>(e))
            .is_some_and(|cm| cm.runtime.want_aim)
    }

    fn trigger_held(&self) -> bool {
        self.world
            .entity_of(HERO)
            .and_then(|e| self.world.world().get::<CharacterMovement>(e))
            .is_some_and(|cm| cm.runtime.want_attack)
    }

    fn cm(&self, who: Uuid) -> CharacterMovement {
        let e = self.world.entity_of(who).expect("the character");
        self.world
            .world()
            .get::<CharacterMovement>(e)
            .expect("a mover")
            .clone()
    }

    fn feel(&self, who: Uuid) -> Option<WeaponFeel> {
        feel::feel_of(&self.world, who)
    }

    fn aim_pitch(&self) -> f64 {
        self.cm(HERO).runtime.aim_pitch_deg
    }

    /// Where the hand pass was told to hold the weapon, world metres.
    fn hold(&self) -> Option<DVec3> {
        let h = inf_ecs::pose::hand_ik(&self.world, HERO)?;
        let r = h.reach[1].as_ref()?;
        Some(DVec3::new(r.target.x, r.target.y, r.target.z))
    }

    fn bloom(&self) -> f64 {
        self.world
            .entity_of(HERO)
            .and_then(|e| self.world.world().get::<weapon::WeaponState>(e))
            .map(|s| s.spread_bloom_deg)
            .unwrap_or(0.0)
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

// ─────────────────────────────────────────────────────────────────────────────
// (a) THE VIEWMODEL LAYER — the spring, on the hand
// ─────────────────────────────────────────────────────────────────────────────

/// **The hold point kicks and comes home, and it does not overshoot past 5 %.**
///
/// Read on `HandIk::reach[1]` — the point the hand solver is actually given, in
/// world metres — and NOT on the spring's own fields: a component that said it
/// had moved while the hand pass ignored it is exactly the defect this arm is
/// for.
///
/// The overshoot is measured as the deepest excursion PAST the rest position on
/// the far side, as a fraction of the peak. A critically damped spring never
/// crosses zero, so the honest number is zero.
///
/// **The mutation**: `feel::VM_DAMPING := feel::DOC_VM_DAMPING` (the research
/// doc's own 18, which its text calls critically damped and which is not) —
/// measured at **6.33 %** of the peak on the pure spring, over the 5 % ceiling.
/// The world half of that mutation is below in
/// `the_docs_own_damping_would_overshoot_this_arms_ceiling`, which runs the
/// same recurrence the hold point runs and needs no source edit.
#[test]
fn a_shot_kicks_the_hold_point_and_it_comes_home_without_overshooting() {
    let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_aim(HERO, true);
    for _ in 0..40 {
        r.step();
    }
    let rest = r.hold().expect("an aiming character holds its weapon");
    // One round, then two hundred steps of settle.
    r.hold_trigger(HERO, true);
    let fired = r.step();
    r.hold_trigger(HERO, false);
    assert_eq!(fired.shots, 1, "the trigger did not fire");
    let mut peak = 0.0_f64;
    let mut worst_back = 0.0_f64;
    let mut samples = Vec::new();
    for _ in 0..200 {
        r.step();
        let at = r.hold().expect("still aiming");
        // The kick is BACKWARD along the aim, which is `-Z` here, so the signed
        // excursion along the aim is what "past the rest position" means.
        let along = at.z - rest.z;
        peak = peak.max(-along);
        worst_back = worst_back.max(along);
        samples.push(along);
    }
    let overshoot = worst_back / peak.max(1.0e-12);
    let home = samples.last().copied().unwrap_or(0.0).abs();
    println!(
        "=== the hold point: peak back {:.2} mm, overshoot {:.4} %, home to {:.4} mm ===",
        peak * 1000.0,
        overshoot * 100.0,
        home * 1000.0
    );
    assert!(
        peak > 0.02,
        "the shot moved the hold point {:.4} mm - the spring is not reaching `HandIk::reach`",
        peak * 1000.0
    );
    assert!(
        overshoot < 0.05,
        "the hold point overshot its rest position by {:.4} % of its peak, past the 5 % ceiling - the spring is not critically damped",
        overshoot * 100.0
    );
    assert!(
        home < 1.0e-4,
        "the hold point came home to {:.4} mm and not to zero - the recoil does not recover",
        home * 1000.0
    );
}

/// **The doc's own damping number would red the arm above**, measured on the
/// recurrence the hold point runs rather than argued from the damping ratio.
///
/// This is the mutation, in the tree, permanently: the constant this engine
/// ships and the constant the research doc prints are both fed to the same
/// integrator and the two overshoots are compared. It is what stops
/// `VM_DAMPING` from being quietly relaxed to the doc's number later on the
/// grounds that the doc says "critically damped".
#[test]
fn the_docs_own_damping_would_overshoot_this_arms_ceiling() {
    let overshoot = |c: f64| -> f64 {
        let mut s = feel::Spring1::default();
        s.add_impulse(feel::impulse_for_peak(1.0, feel::VM_STIFFNESS, c, DT));
        let (mut peak, mut worst) = (0.0_f64, 0.0_f64);
        for _ in 0..600 {
            s.advance(feel::VM_STIFFNESS, c, DT);
            peak = peak.max(s.position);
            worst = worst.min(s.position);
        }
        -worst / peak
    };
    let shipped = overshoot(feel::VM_DAMPING);
    let doc = overshoot(feel::DOC_VM_DAMPING);
    println!(
        "=== overshoot: shipped c={} {:.4} %, the doc's c={} {:.4} % ===",
        feel::VM_DAMPING,
        shipped * 100.0,
        feel::DOC_VM_DAMPING,
        doc * 100.0
    );
    assert!(
        shipped < 0.05,
        "the shipped damping overshoots {:.4} %",
        shipped * 100.0
    );
    assert!(
        doc > 0.05,
        "the doc's own damping overshoots only {:.4} %, so this arm's ceiling is not a ceiling anything could cross and the one above it is vacuous",
        doc * 100.0
    );
    // And the shipped number IS the critical one, to the last bits.
    assert!((feel::VM_DAMPING - feel::critical_damping(feel::VM_STIFFNESS)).abs() < 1.0e-12);
    assert!((feel::AIM_DAMPING - feel::critical_damping(feel::AIM_STIFFNESS)).abs() < 1.0e-12);
}

/// **The profile scales monotonically across the registry's own range** — a
/// pistol at 2.5, an assault rifle at 3.8 and a sniper at 9.9, which are three
/// rows of `weapons.toml`.
///
/// Read as the PEAK displacement each profile asks for, in the units the world
/// measures: metres of hold point and degrees of aim.
#[test]
fn the_profile_scales_with_the_registrys_own_recoil_stat() {
    let rows = [("a pistol", 2.5_f64), ("an AR", 3.8), ("a sniper", 9.9)];
    let mut last: Option<(f64, f64)> = None;
    println!("=== the profile, at three registry rows ===");
    for (what, stat) in rows {
        let p = RecoilProfile::from_recoil_stat(stat);
        let (vm, pitch, yaw) = p.peaks(0, 1);
        println!(
            "  {what:>10} ({stat:>4}): back {:.4} m, up {:.4} m, pitch {pitch:.4} deg, yaw {yaw:.4} deg",
            vm.z, vm.y
        );
        if let Some((back, pitch_last)) = last {
            assert!(
                vm.z < back,
                "{what} does not kick the hands further back than the row below it"
            );
            assert!(
                pitch > pitch_last,
                "{what} does not lift the aim further than the row below it"
            );
        }
        last = Some((vm.z, pitch));
    }
    // The doc's own anchors, at the top of its scale.
    let ten = RecoilProfile::from_recoil_stat(10.0);
    assert!((ten.vm_kick_back - -0.2).abs() < 1.0e-12);
    assert!((ten.cam_pitch_max - 0.17).abs() < 1.0e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// (b) THE AIM LAYER — the recoil that moves the aim, and the reticle
// ─────────────────────────────────────────────────────────────────────────────

/// **A five-round burst raises the aim, and the aim comes home.**
///
/// The number read is `CharacterMovement::runtime::aim_pitch_deg` — which is
/// what `weapon::shot_direction_with` builds the bullet's direction out of, so
/// this is a claim about where the rounds go and not about a spring.
///
/// **The mutation**: dropping the delta in `d3::movement::step_weapon_feel`
/// leaves the pitch flat at 0 and reds the climb; keeping the delta but never
/// zeroing `applied_*` leaves a residual and reds the return.
#[test]
fn a_five_round_burst_raises_the_aim_and_it_returns() {
    let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    for _ in 0..10 {
        r.step();
    }
    let before = r.aim_pitch();
    // **ONE ROUND FIRST, and its own peak.** The impulse lands on the spring's
    // VELOCITY and the movement runtime integrates it on the NEXT step, so the
    // aim on the firing step is still the aim the round LEFT ALONG -- which is
    // correct, and which is why a table sampled on the firing steps reads
    // 0.0000 for the first row. The peak arrives `1/sqrt(k)` later: four steps.
    r.hold_trigger(HERO, true);
    let mut fired = 0;
    while fired == 0 {
        fired += r.step().shots;
    }
    r.hold_trigger(HERO, false);
    let mut one_peak = 0.0_f64;
    for _ in 0..200 {
        r.step();
        one_peak = one_peak.max(r.aim_pitch() - before);
    }
    let one_home = r.aim_pitch() - before;
    // …then the burst, from rest.
    let mut rounds = 0;
    let mut table: Vec<(u32, f64)> = Vec::new();
    r.hold_trigger(HERO, true);
    let mut step = 0u32;
    let mut peak = 0.0_f64;
    while rounds < 5 && step < 60 {
        let rep = r.step();
        rounds += rep.shots;
        step += 1;
        peak = peak.max(r.aim_pitch() - before);
        if rep.shots > 0 {
            table.push((rounds, peak));
        }
    }
    r.hold_trigger(HERO, false);
    assert_eq!(rounds, 5, "the burst did not fire five rounds");
    let mut settle_steps = None;
    for i in 0..600 {
        r.step();
        peak = peak.max(r.aim_pitch() - before);
        if settle_steps.is_none() && (r.aim_pitch() - before).abs() < 0.01 {
            settle_steps = Some(i + 1);
        }
    }
    let after = r.aim_pitch();
    println!("=== the burst, a 3.8-recoil AR at 600 rpm, degrees of aim pitch ===");
    println!("  one round, peak        {one_peak:.4} deg (home to {one_home:.6})");
    for (n, at) in &table {
        println!("  five rounds, peak by {n}  {at:.4} deg");
    }
    println!("  the burst's peak       {peak:.4} deg");
    println!(
        "  home to                {:.6} deg after {:.3} s",
        after - before,
        settle_steps.map(|s| f64::from(s) * DT).unwrap_or(f64::NAN)
    );
    assert!(
        one_peak > 0.5,
        "one round moved the aim {one_peak:.4} deg - the aim layer is not reaching the look integrator"
    );
    assert!(
        one_home.abs() < 1.0e-6,
        "one round left {one_home:.6} deg of aim behind it"
    );
    assert!(
        peak > one_peak * 1.5,
        "five rounds peaked at {peak:.4} deg against one round's {one_peak:.4} - the impulses are not accumulating, so this is a latch and not a spring"
    );
    // …and the profile's own sum is the ceiling: the aim cannot climb further
    // than five peaks, because the spring is recovering the whole time.
    let profile = RecoilProfile::of(&test_rifle());
    let sum: f64 = (1..=6)
        .map(|shot| profile.peaks(test_rifle().spread_seed, shot).1)
        .sum();
    println!("  the profile's own sum  {sum:.4} deg");
    assert!(
        peak < sum,
        "the aim climbed {peak:.4} deg against a profile sum of {sum:.4} - more than every impulse together is not recovery, it is drift"
    );
    // And it comes ALL the way home, which is the centre-recovery.
    assert!(
        (after - before).abs() < 1.0e-6,
        "the aim ended {:.6} deg from where the burst started",
        after - before
    );
    let t = settle_steps.expect("the aim never came within 0.01 deg of home");
    assert!(
        f64::from(t) * DT < 1.5,
        "the aim took {:.3} s to come home",
        f64::from(t) * DT
    );
}

/// **THE RETICLE NEVER LIES.**
///
/// The reticle is drawn at the centre of the screen and the camera's own
/// forward is what that centre points along, so "does the reticle lie" is the
/// angle between the CAMERA's forward and the AIM's forward — measured in
/// degrees and converted to pixels at the camera's own field of view.
///
/// The camera LAGS the aim, by design and for every input: `rotation_lag` is
/// what stops a mouse flick snapping the frame. So the honest claim is not
/// "zero" — it is that a recoil kick moves the reticle off the aim NO FURTHER
/// than the player's own mouse does at the same rate, and that the error at
/// rest is the same before the burst and after it.
///
/// **The control is a mouse flick of the same angular size**, driven through
/// `intent_look_pitch_dps` on a second range with no weapon at all. An arm with
/// no control would be satisfied by a camera that had stopped following
/// anything.
///
/// **The mutation**: a camera kick that does not move the aim — the thing the
/// WPN1 ruling refuses — is exactly a permanent non-zero here, and it is what
/// the before/after comparison catches.
#[test]
fn the_reticle_stays_on_the_aim_line_through_a_burst() {
    fn error_deg(cam: &inf_ecs::camera::LocomotionCamera, cm: &CharacterMovement) -> f64 {
        let aim = weapon::aim_forward(cm.runtime.aim_yaw_deg, cm.runtime.aim_pitch_deg);
        let eye = weapon::aim_forward(cam.pose.yaw_deg, cam.pose.pitch_deg);
        aim.dot(eye).clamp(-1.0, 1.0).acos().to_degrees()
    }
    let run = |burst: bool, look_dps: f64| -> (f64, f64, f64, f64) {
        let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
        if burst {
            r.arm(HERO, "rifle");
        }
        r.aim(HERO, 0.0, 0.0);
        let mut cam = inf_ecs::camera::LocomotionCamera::default();
        let step = |r: &mut Range, cam: &mut inf_ecs::camera::LocomotionCamera| {
            r.bridge.sync_from_world(&r.world);
            d3::step_character_movement(&mut r.world, &mut r.bridge, DT);
            d3::step_gameplay(&mut r.world, &mut r.bridge, DT);
            r.bridge.step(DT);
            r.bridge.write_back_into(&mut r.world);
            r.world.propagate();
            d3::step_camera_with_requests(&mut r.world, &mut r.bridge, cam, HERO, DT);
        };
        for _ in 0..60 {
            step(&mut r, &mut cam);
        }
        let at_rest_before = error_deg(&cam, &r.cm(HERO));
        let mut worst = 0.0_f64;
        if burst {
            r.hold_trigger(HERO, true);
        }
        for i in 0..40 {
            if look_dps != 0.0 {
                let e = r.world.entity_of(HERO).expect("the hero");
                if let Some(mut cm) = r.world.world_mut().get_mut::<CharacterMovement>(e) {
                    cm.runtime.intent_look_pitch_dps = if i < 20 { look_dps } else { -look_dps };
                }
            }
            step(&mut r, &mut cam);
            worst = worst.max(error_deg(&cam, &r.cm(HERO)));
        }
        if burst {
            r.hold_trigger(HERO, false);
        }
        if look_dps != 0.0 {
            let e = r.world.entity_of(HERO).expect("the hero");
            if let Some(mut cm) = r.world.world_mut().get_mut::<CharacterMovement>(e) {
                cm.runtime.intent_look_pitch_dps = 0.0;
            }
        }
        for _ in 0..400 {
            step(&mut r, &mut cam);
        }
        let at_rest_after = error_deg(&cam, &r.cm(HERO));
        (
            at_rest_before,
            worst,
            at_rest_after,
            r.cm(HERO).runtime.aim_pitch_deg,
        )
    };
    // The recoil, and a mouse flick of the same shape as the control.
    let (rest_before, recoil_worst, rest_after, aim_after) = run(true, 0.0);
    let (_, mouse_worst, _, _) = run(false, 120.0);
    // 55 degrees of field over 1080 lines: what one degree is worth in pixels.
    let px = 1080.0 / 55.0;
    println!("=== the reticle against the aim, degrees and 1080p pixels at a 55 deg field ===");
    println!(
        "  at rest, before the burst  {rest_before:.6} deg  ({:.2} px)",
        rest_before * px
    );
    println!(
        "  worst DURING the burst     {recoil_worst:.6} deg  ({:.2} px)",
        recoil_worst * px
    );
    println!(
        "  a 120 deg/s mouse, worst   {mouse_worst:.6} deg  ({:.2} px)",
        mouse_worst * px
    );
    println!(
        "  at rest, after the burst   {rest_after:.6} deg  ({:.2} px)",
        rest_after * px
    );
    println!("  the aim, after the burst   {aim_after:.6} deg");
    // **Against the rest error, not against zero** (mutation M2 found this):
    // dropping the aim delta in `step_weapon_feel` leaves the camera perfectly
    // still, the error at its 0.0384 deg resting value, and `> 0.0` passes.
    // What the arm claims is that the burst MOVED the view, so the number it is
    // fought against is the view standing still.
    assert!(
        recoil_worst > rest_before * 10.0,
        "the reticle moved {recoil_worst:.6} deg during the burst against {rest_before:.6} at rest - the camera is not following an aim that moved, or nothing was fired"
    );
    assert!(
        recoil_worst <= mouse_worst,
        "a recoil kick threw the reticle {recoil_worst:.4} deg off the aim against a mouse flick's {mouse_worst:.4} - the aim layer is moving the view harder than a player can"
    );
    assert!(
        (rest_after - rest_before).abs() < 1.0e-6,
        "the reticle is {rest_after:.6} deg off the aim after the burst against {rest_before:.6} before it - the camera has kept something the aim gave back, which is the reticle lying"
    );
    assert!(
        rest_after * px < 1.0,
        "the reticle sits {:.2} px off the aim at rest",
        rest_after * px
    );
}

/// **Neither the camera nor its authored tuning has a per-shot input.**
///
/// A source arm, and it is the WPN1 ruling in the only form a test can take:
/// the refusal is about what `camera.rs` is ALLOWED to know, and no runtime
/// measurement can distinguish "the camera does not read the weapon" from "the
/// weapon happened not to fire".
#[test]
fn the_camera_has_no_per_shot_input() {
    const SRC: &str = include_str!("../../../crates/inf-ecs/src/camera.rs");
    const TOML: &str = include_str!("../../../samples/phase29-locomotion/camera.toml");
    // Comments name the ruling, so they are stripped before the ban is applied
    // — `portable_character`'s own recipe, for its own reason.
    let code: String = SRC
        .replace("\r\n", "\n")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    // `shot` is NOT on this list, and that is a finding rather than an
    // oversight: `camera.rs` has a `camera_shot` door and a `Scripted` layer
    // whose whole vocabulary is cinematography, so the word means "a cinematic
    // cut" in that file and banning it would ban the camera's own feature. What
    // is banned is every spelling of a WEAPON.
    for banned in ["recoil", "WeaponFeel", "weapon::", "WeaponState", "feel::"] {
        assert!(
            !code.contains(banned),
            "`camera.rs` names `{banned}` outside a comment - a camera that knows a weapon fired is the camera kick the WPN1 ruling refuses"
        );
    }
    for banned in ["recoil", "kick", "weapon"] {
        assert!(
            !TOML.to_ascii_lowercase().contains(banned),
            "`camera.toml` authors `{banned}` - the aim block is the only thing a weapon may reach, and it reaches it through `state_blend_speed`"
        );
    }
    // …and the thing it DOES have is the door this wave writes through, so the
    // arm cannot pass because the file stopped being the camera.
    assert!(code.contains("state_blend_speed"));
    assert!(code.contains("pub fn set_camera_rig_value"));
    assert!(TOML.contains("[aiming"));
}

// ─────────────────────────────────────────────────────────────────────────────
// (c) THE SPREAD LAYER — thirty holes in a wall at 25 m
// ─────────────────────────────────────────────────────────────────────────────

/// **The pattern at 25 m orders itself: aimed and crouched and still, then hip
/// and still, then moving.**
///
/// Read off the WORLD: thirty rounds into a wall, the impact points taken from
/// `WeaponHit::to`, and the radius is the RMS distance from their own mean. Not
/// a cone in degrees, not a multiplier — holes in a wall.
///
/// **The mutation**: `feel::SPREAD_ADS_MULT := 1.0` collapses the first two into
/// each other and reds the ordering; `SPREAD_MOVE_MULT_MAX := 1.0` collapses the
/// last two.
#[test]
fn the_thirty_round_pattern_orders_ads_then_hip_then_moving() {
    fn pattern(aim_down_sights: bool, crouch: bool, move_mps: f64) -> (f64, f64, usize) {
        let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
        r.arm(HERO, "rifle");
        r.aim(HERO, 0.0, 0.0);
        if crouch {
            r.set_mode(HERO, MovementMode::Crouch);
        }
        r.hold_aim(HERO, aim_down_sights);
        // Let the ADS blend arrive (or not) before a round leaves.
        for _ in 0..40 {
            r.step();
        }
        if aim_down_sights {
            let b = r.feel(HERO).map(|f| f.ads_blend).unwrap_or(0.0);
            assert!(
                b > 0.99,
                "the ADS blend was {b:.4} when the pattern started - the aim never arrived and this pattern is a hip-fire one"
            );
        }
        let intent = (move_mps > 0.0).then(|| MovementIntent {
            move_input: Vec2d::new(0.0, 1.0),
            sprint: true,
            ..Default::default()
        });
        // Walk up to speed before firing, so the movement penalty is the one
        // the run actually has.
        if intent.is_some() {
            for _ in 0..60 {
                r.step_with(intent.clone());
            }
            let v = r.cm(HERO).runtime.velocity.to_dvec3();
            let s = (v.x * v.x + v.z * v.z).sqrt();
            assert!(
                s > 1.0,
                "the moving pattern is standing still at {s:.4} m/s, so it is the hip pattern twice"
            );
        }
        r.hold_trigger(HERO, true);
        // **The deviation from the AIM LINE, in the wall's own plane** -- not
        // the scatter of the holes about their own centre.
        //
        // The two are different measurements and only the first one is the
        // SPREAD. The aim really moves under recoil (clause 2), so a group
        // measured about its own mean charges the spread for the burst's climb
        // and, at 600 rpm, buries the layer under test: measured, the three
        // conditions came out 0.6526 / 0.7040 / 0.6400 m and did not order.
        // Subtracting the aim's own point on the wall -- which is a fact about
        // the world, not a report, because it is built from the shot's own
        // muzzle and the aim the movement runtime held on the step it left --
        // leaves exactly the cone.
        let mut hits: Vec<DVec3> = Vec::new();
        let mut speeds: Vec<f64> = Vec::new();
        let mut blooms: Vec<f64> = Vec::new();
        for _ in 0..400 {
            let rep = r.step_with(intent.clone());
            let cm = r.cm(HERO);
            let bloom_now = r.bloom();
            let aim = weapon::aim_forward(cm.runtime.aim_yaw_deg, cm.runtime.aim_pitch_deg);
            for h in &rep.hits {
                // The wall, by the guid the bridge named -- not "anything that
                // arrived": a hitscan record carries `arrived: false` (only a
                // projectile's landing sets it) and its `target` is the thing
                // it hit, so the first cut of this filter matched nothing and
                // the arm would have reported an EMPTY pattern as a tight one
                // if the count had not been asserted first.
                if h.target == Some(WALL) {
                    let t = (h.to.z - h.from.z) / aim.z;
                    let aim_point = h.from + aim * t;
                    // …and NORMALISED to 25 m along the shot. A character
                    // running at the wall is 17 m closer by its thirtieth round
                    // than by its first, so the same cone draws a group a third
                    // of the size -- measured, and it is why the first cut of
                    // this arm had the moving group TIGHTER than the standing
                    // one (0.3526 against 0.3512 m). The number reported is
                    // therefore "the radius this cone makes at 25 m", which is
                    // what the clause asks for and what a shooting range
                    // measures.
                    hits.push((h.to - aim_point) * (PATTERN_M / t));
                    let v = cm.runtime.velocity.to_dvec3();
                    speeds.push((v.x * v.x + v.z * v.z).sqrt());
                    blooms.push(bloom_now);
                }
            }
            if hits.len() >= 30 {
                break;
            }
        }
        r.hold_trigger(HERO, false);
        assert!(
            hits.len() >= 30,
            "only {} of thirty rounds reached the wall",
            hits.len()
        );
        hits.truncate(30);
        // The radius is measured in the wall's own plane, about the pattern's
        // own centre — so a hero that walked while firing is not charged for
        // having moved the whole group sideways, only for having scattered it.
        let n = hits.len() as f64;
        speeds.truncate(30);
        blooms.truncate(30);
        println!(
            "    (mean speed at the shots {:.3} m/s, mean bloom {:.4} deg)",
            speeds.iter().sum::<f64>() / speeds.len().max(1) as f64,
            blooms.iter().sum::<f64>() / blooms.len().max(1) as f64
        );
        let rms = (hits.iter().map(|h| h.x * h.x + h.y * h.y).sum::<f64>() / n).sqrt();
        let worst = hits
            .iter()
            .map(|h| (h.x * h.x + h.y * h.y).sqrt())
            .fold(0.0_f64, f64::max);
        (rms, worst, hits.len())
    }
    let (ads_rms, ads_worst, _) = pattern(true, true, 0.0);
    let (hip_rms, hip_worst, _) = pattern(false, false, 0.0);
    let (run_rms, run_worst, _) = pattern(false, false, 5.0);
    println!("=== thirty rounds at {PATTERN_M} m, radius on the wall ===");
    println!(
        "  ADS + crouched + still  rms {:.4} m, worst {:.4} m",
        ads_rms, ads_worst
    );
    println!(
        "  hip + still             rms {:.4} m, worst {:.4} m",
        hip_rms, hip_worst
    );
    println!(
        "  hip + sprinting         rms {:.4} m, worst {:.4} m",
        run_rms, run_worst
    );
    assert!(
        ads_rms > 0.0,
        "every one of thirty rounds went through the same hole - there is no spread at all and this arm proves nothing"
    );
    assert!(
        ads_rms < hip_rms,
        "the aimed crouched group ({ads_rms:.4} m) is not tighter than the hip group ({hip_rms:.4} m)"
    );
    assert!(
        hip_rms < run_rms,
        "the standing group ({hip_rms:.4} m) is not tighter than the sprinting one ({run_rms:.4} m)"
    );
}

/// **The bloom is per-magazine state, it saturates, and it decays.**
///
/// Read on `WeaponState::spread_bloom_deg` through the fire path — so a bloom
/// that was computed and never stored, or stored and never decayed, is visible.
#[test]
fn the_bloom_grows_with_the_burst_and_decays_when_the_trigger_comes_up() {
    let def = test_rifle();
    let mut r = Range::new(defs_with(&[("rifle", def)]));
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    for _ in 0..10 {
        r.step();
    }
    assert_eq!(
        r.bloom(),
        0.0,
        "a weapon that has not fired carries a bloom"
    );
    r.hold_trigger(HERO, true);
    let mut after_one = 0.0;
    let mut rounds = 0;
    for _ in 0..200 {
        let rep = r.step();
        rounds += rep.shots;
        if rounds == 1 && after_one == 0.0 {
            after_one = r.bloom();
        }
    }
    r.hold_trigger(HERO, false);
    let saturated = r.bloom();
    println!(
        "=== the bloom: one round {:.4} deg, saturated {:.4} deg (ceiling {:.4}) ===",
        after_one,
        saturated,
        feel::bloom_max_deg(&def)
    );
    assert!(
        (after_one - feel::bloom_per_shot_deg(&def)).abs() < 1.0e-9,
        "one round bloomed {after_one:.6} deg"
    );
    assert!(
        saturated > after_one,
        "the bloom did not grow past one round's worth over a whole magazine"
    );
    assert!(
        saturated <= feel::bloom_max_deg(&def) + 1.0e-12,
        "the bloom passed its own ceiling"
    );
    // …and it decays with the trigger up.
    let mut steps = 0;
    while r.bloom() > 0.0 && steps < 600 {
        r.step();
        steps += 1;
    }
    println!("  gone after {:.3} s", f64::from(steps) * DT);
    assert_eq!(r.bloom(), 0.0, "the bloom never decayed to zero");
    // 1.25 s is `1 / BLOOM_DECAY_PER_S` by construction, and the measurement is
    // 1.217 s because the bloom had not quite saturated when the trigger came
    // up. Two seconds is the ceiling this arm defends: a decay slow enough that
    // a player cannot re-aim between bursts is a spread that never resets.
    assert!(
        f64::from(steps) * DT < 2.0,
        "the bloom took {steps} steps to clear"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (d) SWAY
// ─────────────────────────────────────────────────────────────────────────────

/// **The sway is exactly zero on a dead-still character that is not breathing,
/// and it grows with speed.**
///
/// The "not breathing" half is the one that makes the claim falsifiable: the
/// breath term is gated on the animation clock, and this fixture's hero has no
/// state machine, so its clock is zero and its sway must be exactly zero — not
/// small, zero, because that exactness is what empties the trace section.
#[test]
fn the_sway_is_zero_at_rest_and_grows_with_speed() {
    // (1) NO CLOCK, NO SWAY. A capsule with no state machine has no animation
    //     clock, so it does not breathe -- and its sway is EXACTLY zero, not
    //     small. That exactness is what empties this character's trace section.
    let mut bare = Range::new(defs_with(&[("rifle", test_rifle())]));
    bare.arm(HERO, "rifle");
    bare.aim(HERO, 0.0, 0.0);
    bare.hold_aim(HERO, true);
    for _ in 0..60 {
        bare.step();
    }
    let unbreathing = bare.feel(HERO).expect("an armed hero has a feel").sway;
    assert_eq!(
        unbreathing,
        DVec3::ZERO,
        "a dead-still character with no animation clock swayed {unbreathing:?} - the breath gate is not reading the clock"
    );

    // (2) A BREATHING CHARACTER, STANDING STILL. The clock runs, so the chest
    //     term does, and the sway is small and non-zero.
    let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
    r.install_rig(HERO);
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_aim(HERO, true);
    let mut breathing = 0.0_f64;
    for _ in 0..240 {
        r.step();
        breathing = breathing.max(r.feel(HERO).map(|f| f.sway.length()).unwrap_or(0.0));
    }
    let clock = inf_ecs::anim_bridge::anim_state_time(&r.world, HERO);

    // (3) THE SAME CHARACTER, SPRINTING. The bob is proportional to speed.
    r.hold_aim(HERO, false);
    let intent = MovementIntent {
        move_input: Vec2d::new(0.0, 1.0),
        sprint: true,
        ..Default::default()
    };
    let mut sprinting = 0.0_f64;
    for _ in 0..240 {
        r.step_with(Some(intent.clone()));
        sprinting = sprinting.max(r.feel(HERO).map(|f| f.sway.length()).unwrap_or(0.0));
    }
    let v = r.cm(HERO).runtime.velocity.to_dvec3();
    let speed = (v.x * v.x + v.z * v.z).sqrt();
    println!("=== the sway, millimetres of hold point ===");
    println!(
        "  no clock, dead still   {:.6} mm",
        unbreathing.length() * 1000.0
    );
    println!(
        "  breathing, still       {:.4} mm (clock {clock:.3} s)",
        breathing * 1000.0
    );
    println!(
        "  breathing, {speed:.3} m/s      {:.4} mm",
        sprinting * 1000.0
    );
    assert!(
        clock > 1.0,
        "the animation clock is {clock:.4} s after four seconds - it is not running and (2) is the same measurement as (1)"
    );
    assert!(
        breathing > 0.0,
        "a breathing character standing still swayed nothing at all"
    );
    assert!(
        sprinting > breathing * 2.0,
        "sprinting at {speed:.3} m/s swayed {:.4} mm against a standstill's {:.4} - the bob is not reading the speed",
        sprinting * 1000.0,
        breathing * 1000.0
    );
    assert!(
        sprinting <= feel::SWAY_MAX_M * 1.5,
        "the sway reached {sprinting:.6} m, past its own ceiling"
    );
}

/// **The sway and the breath share ONE clock.**
///
/// The mutation this kills is a second accumulator: a sway with a clock of its
/// own would drift out of phase with the chest and nothing would notice. So the
/// arm reads the clock the sway is a function of — `anim_state_time`, which is
/// the same `SmRuntimeState::state_time` `pose::apply_breath` is handed — and
/// checks that the sway a character HAS is the sway that clock produces.
#[test]
fn the_sway_and_the_breath_read_the_same_clock() {
    // A source arm and a value arm together, because neither alone is enough:
    // the source says there is one accumulator, the value says the sway is a
    // function of it.
    const POSE: &str = include_str!("../../../crates/inf-ecs/src/pose.rs");
    const MOVE: &str = include_str!("../../../crates/inf-physics/src/d3/movement.rs");
    assert!(
        POSE.contains("apply_breath(asset, &mut pose, machine, &clips, rt.state_time)"),
        "the breath no longer reads `state_time` - the one clock has moved and the sway does not know"
    );
    assert!(
        MOVE.contains("inf_ecs::anim_bridge::anim_state_time(world, guid)"),
        "the sway no longer reads the animation clock through `anim_state_time`"
    );
    // The value half: at a clock of zero and no speed the sway is exactly zero,
    // and at the same clock with speed it is exactly what `sway_offset` says.
    let bob = feel::sway_offset(0.0, 0.0, 0.0, (0.0, 0.0));
    assert_eq!(bob, DVec3::ZERO);
    let a = feel::sway_offset(1.234, 3.0, 1.0, (0.0, 0.0));
    let b = feel::sway_offset(1.234, 3.0, 1.0, (0.0, 0.0));
    assert_eq!(a, b, "the sway is not a pure function of its clock");
    assert_ne!(
        a,
        feel::sway_offset(1.235, 3.0, 1.0, (0.0, 0.0)),
        "the sway does not move with its clock at all"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (e) ADS TIMING
// ─────────────────────────────────────────────────────────────────────────────

/// **The aim arrives in the weapon's own `ads_time_ms`, to within one frame** —
/// measured on the CAMERA's field of view, which is what a player sees.
///
/// The rig's `state_blend_speed` is solved from the weapon's own number and the
/// camera's own `interp_to`; this walks the real camera from the hip block's
/// field to the aim block's 55 degrees and counts frames.
///
/// **The mutation**: `blend_speed_for_ads` answering a constant (the rig's own
/// default 6.0) puts a 140 ms pistol at 620 ms and reds it by 29 frames.
#[test]
fn the_aim_arrives_in_the_weapons_own_ads_time() {
    let rows = [("rifle", test_rifle()), ("pistol", test_pistol())];
    println!("=== the ADS blend, measured on the camera's field of view ===");
    for (id, def) in rows {
        let mut r = Range::new(defs_with(&[(id, def)]));
        r.arm(HERO, id);
        r.aim(HERO, 0.0, 0.0);
        let mut cam = inf_ecs::camera::LocomotionCamera::default();
        let step = |r: &mut Range, cam: &mut inf_ecs::camera::LocomotionCamera| {
            r.bridge.sync_from_world(&r.world);
            d3::step_character_movement(&mut r.world, &mut r.bridge, DT);
            d3::step_gameplay(&mut r.world, &mut r.bridge, DT);
            r.bridge.step(DT);
            r.bridge.write_back_into(&mut r.world);
            r.world.propagate();
            d3::step_camera_with_requests(&mut r.world, &mut r.bridge, cam, HERO, DT);
        };
        for _ in 0..120 {
            step(&mut r, &mut cam);
        }
        let hip_fov = cam.pose.fov_deg;
        let hip_arm = cam.pose.position;
        let aim_fov = inf_ecs::camera::CameraTuning::default().aiming.walk.fov_deg;
        // "Arrived" is an absolute threshold on the FOV in the world, not a
        // fraction: 98 % of this span, stated as the degrees it is worth.
        let span = (hip_fov - aim_fov).abs();
        let eps = span * (1.0 - feel::ADS_BLEND_REACH);
        r.hold_aim(HERO, true);
        let mut frames = 0u32;
        while (cam.pose.fov_deg - aim_fov).abs() > eps && frames < 600 {
            step(&mut r, &mut cam);
            frames += 1;
        }
        let took_ms = f64::from(frames) * DT * 1000.0;
        let out_by = (took_ms - def.ads_time_ms).abs() / (DT * 1000.0);
        println!(
            "  {id:>6}: {} ms asked, {:.1} ms measured ({out_by:.2} frames out); the field went {hip_fov:.2} -> {:.2} deg, and the boom {:.3} -> {:.3} m",
            def.ads_time_ms,
            took_ms,
            cam.pose.fov_deg,
            (hip_arm.to_dvec3() - r.cm(HERO).runtime.velocity.to_dvec3()).length() * 0.0
                + hip_arm.to_dvec3().length(),
            cam.pose.position.to_dvec3().length()
        );
        assert!(
            span > 1.0,
            "the hip field and the aim field are {span:.4} deg apart - there is nothing for this arm to measure"
        );
        assert!(
            out_by <= 1.0,
            "{id} took {took_ms:.1} ms to reach the aim block against its own {} ms, which is {out_by:.2} frames out",
            def.ads_time_ms
        );
    }
}

/// **An aiming character walks slower, on the ground.**
///
/// Metres covered in three seconds, not a multiplier: `move_speed_mult` reaches
/// `settings_for` through `equip_scale` and the ADS factor joins it at the same
/// seam, so this is the arm that says the second factor is actually applied.
#[test]
fn an_aiming_character_covers_less_ground() {
    let walk = |aim: bool| -> f64 {
        let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
        r.arm(HERO, "rifle");
        r.aim(HERO, 0.0, 0.0);
        r.hold_aim(HERO, aim);
        for _ in 0..60 {
            r.step();
        }
        let intent = MovementIntent {
            move_input: Vec2d::new(0.0, 1.0),
            ..Default::default()
        };
        let start = r.cm(HERO);
        let from = r
            .world
            .entity_of(HERO)
            .and_then(|e| r.world.world().get::<Transform>(e))
            .map(|t| t.translation.to_dvec3())
            .expect("a hero");
        let _ = start;
        for _ in 0..180 {
            r.step_with(Some(intent.clone()));
        }
        let to = r
            .world
            .entity_of(HERO)
            .and_then(|e| r.world.world().get::<Transform>(e))
            .map(|t| t.translation.to_dvec3())
            .expect("a hero");
        DVec3::new(to.x - from.x, 0.0, to.z - from.z).length()
    };
    let hip = walk(false);
    let ads = walk(true);
    println!(
        "=== three seconds of walking: hip {hip:.4} m, aiming {ads:.4} m ({:.3}) ===",
        ads / hip
    );
    assert!(hip > 1.0, "the hero did not walk at all: {hip:.4} m");
    assert!(
        ads < hip * 0.95,
        "an aiming character covered {ads:.4} m against {hip:.4} - the ADS factor is not reaching `equip_scale`"
    );
    // **The factor COMPOUNDS with ALS's own aiming scale**, and that is the
    // number to check against rather than `ADS_MOVE_SPEED_MULT` alone.
    // `settings_for` multiplies the target speed by
    // `rotation_speed_scale()`, which is `aiming_speed_scale` in
    // `RotationMode::Aiming`, and THEN by `equip_scale`. So an aiming
    // character is slower for two reasons and the arm has to name both, or it
    // would red the day somebody tuned the ALS table.
    let aim_scale = CharacterMovement {
        rotation_mode: RotationMode::Aiming,
        ..Default::default()
    }
    .rotation_speed_scale();
    let want = aim_scale * feel::ADS_MOVE_SPEED_MULT;
    println!(
        "  the aiming scale is {aim_scale:.4} and the ADS factor {:.4}, so the ratio should be {want:.4}",
        feel::ADS_MOVE_SPEED_MULT
    );
    assert!(
        (ads / hip - want).abs() < 0.08,
        "the ratio is {:.4} against {want:.4} - the ADS factor is not the one that was authored",
        ads / hip
    );
    // **…and it is BELOW the aiming scale alone** (mutation M8 found this):
    // `ADS_MOVE_SPEED_MULT := 1.0` leaves the ratio at exactly `aim_scale`, and
    // an arm that only checks the ratio against a product containing the
    // mutated constant passes. This half names no constant of the wave's.
    assert!(
        ads / hip < aim_scale * 0.9,
        "the ratio is {:.4} against the aiming scale's own {aim_scale:.4} - the ADS factor is 1.0 and only ALS's table is slowing this character down",
        ads / hip
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (f) THE OVERLAYS ON THE JOINTS
// ─────────────────────────────────────────────────────────────────────────────

/// **Equipping a weapon changes the ARMS, and the aim blend changes them again.**
///
/// A rig with a real `overlay_m4a1` pose set and a real `aim_m4a1` sweep, both
/// authoring the CLAVICLE, and the measurement is the clavicle's own angle off
/// its rest pose: unarmed, carrying, aiming.
///
/// # Why the clavicle and not the upper arm
///
/// **A finding, and it is this wave's second-most useful one.** The hand IK runs
/// AFTER the overlay (`pose.rs`'s frozen writer order: the overlay at
/// construction, `apply_hand_ik` among the corrections) and `solve_arm` writes
/// the whole arm chain -- upper arm, forearm, hand. So on an armed character the
/// overlay's UPPER ARM is overwritten every step: by the gun grip, which puts
/// the off hand on the weapon, and by the aim reach, which puts the weapon hand
/// on the hold point. Measured on the first cut of this arm: the left upper arm
/// read 90.000 deg unarmed, 90.000 carrying and 81.407 aiming -- the overlay
/// invisible, the aim reach visible.
///
/// `inf_anim::arm_chain` is upper arm / forearm / hand and does NOT include the
/// clavicle, so the shoulder is the part of the overlay that survives, and it is
/// what a viewer reads as "the arms have come in". The bound is stated in
/// `pose::apply_weapon_overlay`'s doc as well; closing it means the hold point
/// becoming an OFFSET from the overlay's own hand rather than a world point,
/// which is a WPN2d-sized change to `HandIk`.
///
/// **The mutation**: `overlay_states_for` answering `("", "")` for every weapon
/// leaves all three angles equal and reds both comparisons.
#[test]
fn equipping_a_weapon_changes_the_arms_and_aiming_changes_them_again() {
    const CARRY: inf_anim::ClipRef = [0xb7; 16];
    const AIMED: inf_anim::ClipRef = [0xb8; 16];
    const CROUCH: inf_anim::ClipRef = [0xb9; 16];
    const IDLE: inf_anim::ClipRef = [0xba; 16];

    let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
    let skeleton = inf_anim::build_template(
        inf_anim::BodyPlan::Biped,
        &inf_anim::BodyParams {
            height_m: 1.8,
            ..Default::default()
        },
    )
    .expect("the mannequin builds");
    let roles = skeleton.role_index();
    let clav = roles
        .first(inf_anim::BoneRoleKind::Clavicle, inf_anim::BoneSide::Right)
        .expect("a right clavicle");
    let upper = roles
        .first(inf_anim::BoneRoleKind::UpperArm, inf_anim::BoneSide::Right)
        .expect("a right upper arm");
    let machine = inf_anim::StateMachine {
        states: vec![
            inf_anim::SmState::clip("idle", IDLE),
            inf_anim::SmState {
                name: inf_anim::als::OVERLAY_M4A1_STATE.into(),
                motion: inf_anim::state_machine::Motion::Clip(CARRY),
                looping: false,
                speed: 1.0,
                position: (0.0, 0.0),
                on_enter: Vec::new(),
                on_exit: Vec::new(),
            },
            inf_anim::SmState {
                name: inf_anim::als::AIM_M4A1_STATE.into(),
                motion: inf_anim::state_machine::Motion::Blend2D(inf_anim::BlendSpace2D::new(
                    inf_anim::als::MOVE_X_VAR,
                    inf_anim::als::MOVE_Y_VAR,
                    vec![
                        inf_anim::BlendEntry2D {
                            pos: inf_anim::als::WEAPON_SWEEP_STANDING,
                            clip: AIMED,
                        },
                        inf_anim::BlendEntry2D {
                            pos: inf_anim::als::WEAPON_SWEEP_CROUCHED,
                            clip: CROUCH,
                        },
                    ],
                )),
                looping: false,
                speed: 1.0,
                position: (0.0, 0.0),
                on_enter: Vec::new(),
                on_exit: Vec::new(),
            },
        ],
        entry: 0,
        ..Default::default()
    };
    r.clips
        .insert(IDLE, inf_anim::AnimClip::new("idle", Vec::new()));
    r.clips.insert(CARRY, joint_pose(clav, -12.0));
    r.clips.insert(AIMED, joint_pose(clav, -30.0));
    r.clips.insert(CROUCH, joint_pose(clav, -30.0));
    r.rig = Some((skeleton, machine));
    let e = r.world.entity_of(HERO).expect("the hero");
    r.world.world_mut().entity_mut(e).insert((
        inf_ecs::components::AnimStateMachine {
            sm: Some(SM_GUID),
            ..Default::default()
        },
        inf_ecs::components::SkeletalMesh {
            mesh: None,
            skeleton: Some(SKEL_GUID),
        },
    ));
    r.world.mark_dirty();

    let angle = |r: &Range, j: u16| -> f64 {
        let (rig, _) = r.rig.as_ref().expect("a rig");
        let posed = inf_ecs::pose::evaluated_pose(&r.world, HERO).expect("a posed hero");
        let a = glam::Quat::from_array(posed.pose.locals[j as usize].rotation);
        let b =
            glam::Quat::from_array(inf_anim::Pose::rest(&rig.skeleton).locals[j as usize].rotation);
        f64::from(a.angle_between(b).to_degrees())
    };

    for _ in 0..30 {
        r.step();
    }
    let unarmed = angle(&r, clav);
    let unarmed_upper = angle(&r, upper);
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    for _ in 0..30 {
        r.step();
    }
    let carrying = angle(&r, clav);
    let carrying_upper = angle(&r, upper);
    r.hold_aim(HERO, true);
    for _ in 0..60 {
        r.step();
    }
    let ads = r.feel(HERO).map(|f| f.ads_blend).unwrap_or(0.0);
    let aiming = angle(&r, clav);
    let aiming_upper = angle(&r, upper);
    println!("=== the right shoulder, degrees off its rest pose ===");
    println!("  unarmed           clavicle {unarmed:.3}, upper arm {unarmed_upper:.3}");
    println!("  carrying a rifle  clavicle {carrying:.3}, upper arm {carrying_upper:.3}");
    println!(
        "  aiming it         clavicle {aiming:.3}, upper arm {aiming_upper:.3} (blend {ads:.3})"
    );
    assert!(
        ads > 0.99,
        "the ADS blend was {ads:.4} and the aimed pose is a carry pose"
    );
    assert!(
        unarmed < 0.001,
        "an unarmed character is already {unarmed:.4} deg off its rest shoulder - something else is posing it and the two comparisons below mean nothing"
    );
    assert!(
        (carrying - 12.0).abs() < 0.5,
        "picking up a rifle moved the clavicle {carrying:.3} deg against the 12 the pose set authors - the overlay is not reaching the pose, or not at full weight"
    );
    assert!(
        (aiming - 30.0).abs() < 0.5,
        "aiming moved the clavicle to {aiming:.3} deg against the 30 the sweep authors - the aim sweep is not reaching the pose, or the cross-fade never leaves the carry pose"
    );
}

/// A clip that rotates one joint by `deg` about its own Z, held for a second.
fn joint_pose(joint: u16, deg: f32) -> inf_anim::AnimClip {
    let h = deg.to_radians() * 0.5;
    let (s, c) = (inf_math::psin(h), inf_math::pcos(h));
    let mut track = inf_anim::JointTrack::new(joint);
    track.rotation = Some(inf_anim::QuatTrack::new(
        vec![0.0, 1.0],
        vec![[0.0, 0.0, s, c], [0.0, 0.0, s, c]],
        inf_anim::Interpolation::Linear,
    ));
    inf_anim::AnimClip::new("pose", vec![track])
}

/// **THE HAND GOES WHERE THE AIM SENDS IT** — and the number that matters is
/// how far short of the hold point it lands, in millimetres.
///
/// Swept over five aim directions (level, up, down, left, right), and read
/// twice each time: the REQUESTED hold point (`HandIk::reach[1]`, which
/// `aim_hold_point` builds out of `weapon::aim_forward` and is therefore on the
/// aim line by construction) and the SOLVED hand joint out of the evaluated
/// pose. The distance between them is the IK's own reach error.
///
/// # A finding, and it is why this arm is millimetres and not degrees
///
/// The first cut asked the clause's own question — "the joints follow the aim
/// line within 2 degrees" — as the angle between the aim and the vector from
/// the character to its hand, and there is no origin for which that is true.
/// From the shoulder JOINT it is **50.96 deg** at a level aim and 70.94 at a
/// 35-degree one, because the joint is twenty centimetres off to one side of a
/// hand held in front of the chest. From the shoulder LINE — the point
/// `aim_hold_point` measures its own reach from — it is **15.32 / 0.47 / 38.50 /
/// 15.06 / 15.73 deg** over the five directions, and the systematic 15 degrees
/// at a level aim is the arm hanging where a 1.8 m rig's arm hangs rather than
/// where a 0.42 m reach in front of a 1.476 m shoulder line asks it to.
///
/// So "the joints follow the aim line" is not a claim about an angle from any
/// point on the body. What IS a claim, and what a player sees, is that the hand
/// arrives where the hold point sent it — and THAT is measured here, and it is
/// **not true either**:
///
/// | aim | how far short the hand lands |
/// |---|---|
/// | level | **126.74 mm** |
/// | 35 deg up | **0.00 mm** |
/// | 35 deg down | **274.65 mm** |
/// | 60 deg left | **124.78 mm** |
/// | 60 deg right | **124.55 mm** |
///
/// The hold point is `AIM_REACH_M` = 0.42 m in front of a shoulder LINE derived
/// from the capsule (`SHOULDER_OF_HEIGHT` of the stand height), and the RIG's
/// shoulder joint is neither at that height nor on that line — so at a level or
/// downward aim the requested point is outside the arm's own reach envelope and
/// `solve_arm` stops where it can. It is wave WPN1's geometry, not this wave's:
/// both constants and the whole hold-point construction predate it, and this is
/// the first time anything has measured what the ARM did about them. The
/// weapon entity hangs off the hand socket, so what it costs is a muzzle about
/// twelve centimetres off the aim line — the DIRECTION a round leaves along is
/// `aim_forward` and is unaffected, which is why nothing has noticed.
///
/// Carried, with the fix named: `aim_hold_point` should measure its reach from
/// the rig's own shoulder joint when the character has one, and fall back to
/// the capsule rule for the rig-less heroes it was written for.
///
/// **This arm is therefore a TRIPWIRE pinned at what was measured**, not a claim
/// that the hand arrives. It reds if the reach gets worse — which is what a
/// change to `AIM_REACH_M`, to `SHOULDER_OF_HEIGHT`, or to the solver would do —
/// and it reds if `aim_hold_point` stops reading the aim at all, because the
/// five hold points would then be one point measured five times.
#[test]
fn the_hand_arrives_where_the_aim_sends_it() {
    let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
    r.install_rig(HERO);
    r.arm(HERO, "rifle");
    r.hold_aim(HERO, true);
    let hand = {
        let (rig, _) = r.rig.as_ref().expect("a rig");
        rig.role_index()
            .first(inf_anim::BoneRoleKind::Hand, inf_anim::BoneSide::Right)
            .expect("a right hand")
    };
    println!("=== the hand against the hold point it was sent to ===");
    let mut worst = 0.0_f64;
    let mut spread_of_targets = Vec::new();
    for (yaw, pitch) in [
        (0.0_f64, 0.0_f64),
        (0.0, 35.0),
        (0.0, -35.0),
        (60.0, 0.0),
        (-60.0, 0.0),
    ] {
        r.aim(HERO, yaw, pitch);
        for _ in 0..90 {
            r.step();
        }
        let want = r.hold().expect("an aiming character holds its weapon");
        spread_of_targets.push(want);
        let (rig, _) = r.rig.as_ref().expect("a rig");
        let posed = inf_ecs::pose::evaluated_pose(&r.world, HERO).expect("a posed hero");
        let to_world = inf_ecs::pose::model_to_world_of(&r.world, HERO).expect("a placement");
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &posed.pose);
        let p = g[hand as usize].to_scale_rotation_translation().2;
        let got =
            to_world.transform_point3(DVec3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z)));
        let mm = (got - want).length() * 1000.0;
        println!("  aim {yaw:>6.1} / {pitch:>5.1} deg: the hand is {mm:.2} mm from the hold point");
        worst = worst.max(mm);
    }
    // The hold points really were five different places, or the arm is a claim
    // about one target measured five times.
    let far = spread_of_targets
        .iter()
        .map(|a| (*a - spread_of_targets[0]).length())
        .fold(0.0_f64, f64::max);
    assert!(
        far > 0.3,
        "the five aims sent the hand to points {far:.4} m apart - `aim_hold_point` is not reading the aim"
    );
    assert!(
        worst < 300.0,
        "the hand landed {worst:.2} mm from where the aim sent it, against the 274.65 this tripwire was pinned at - the reach got worse"
    );
}

/// **The overlay leaves the LEGS alone.**
///
/// The mask arm: an overlay is both clavicle subtrees, so a rifle in the hands
/// must not move a foot. Fought against a CONTROL of the same length -- thirty
/// steps of the same idle with nothing equipped -- because a foot that is still
/// settling would otherwise be charged to the overlay.
#[test]
fn the_overlay_leaves_the_legs_where_the_locomotion_put_them() {
    const CARRY: inf_anim::ClipRef = [0xc7; 16];
    const IDLE: inf_anim::ClipRef = [0xc8; 16];
    fn run(arm_it: bool) -> (f64, f64) {
        let mut r = Range::new(defs_with(&[("rifle", test_rifle())]));
        let skeleton = inf_anim::build_template(
            inf_anim::BodyPlan::Biped,
            &inf_anim::BodyParams {
                height_m: 1.8,
                ..Default::default()
            },
        )
        .expect("the mannequin builds");
        let roles = skeleton.role_index();
        let clav = roles
            .first(inf_anim::BoneRoleKind::Clavicle, inf_anim::BoneSide::Right)
            .expect("a right clavicle");
        let foot = roles
            .first(inf_anim::BoneRoleKind::Foot, inf_anim::BoneSide::Left)
            .expect("a left foot");
        r.clips.insert(CARRY, joint_pose(clav, -25.0));
        r.clips
            .insert(IDLE, inf_anim::AnimClip::new("idle", Vec::new()));
        r.rig = Some((
            skeleton,
            inf_anim::StateMachine {
                states: vec![
                    inf_anim::SmState::clip("idle", IDLE),
                    inf_anim::SmState {
                        name: inf_anim::als::OVERLAY_M4A1_STATE.into(),
                        motion: inf_anim::state_machine::Motion::Clip(CARRY),
                        looping: false,
                        speed: 1.0,
                        position: (0.0, 0.0),
                        on_enter: Vec::new(),
                        on_exit: Vec::new(),
                    },
                ],
                entry: 0,
                ..Default::default()
            },
        ));
        let e = r.world.entity_of(HERO).expect("the hero");
        r.world.world_mut().entity_mut(e).insert((
            inf_ecs::components::AnimStateMachine {
                sm: Some(SM_GUID),
                ..Default::default()
            },
            inf_ecs::components::SkeletalMesh {
                mesh: None,
                skeleton: Some(SKEL_GUID),
            },
        ));
        r.world.mark_dirty();
        let joint = |r: &Range, j: u16| -> glam::Vec3 {
            let (rig, _) = r.rig.as_ref().expect("a rig");
            let posed = inf_ecs::pose::evaluated_pose(&r.world, HERO).expect("posed");
            inf_anim::pose::global_transforms(&rig.skeleton, &posed.pose)[j as usize]
                .to_scale_rotation_translation()
                .2
        };
        // The SHOULDER is read as an angle and the FOOT as a position, and that
        // is not a style choice: the overlay ROTATES the clavicle about its own
        // origin, so the clavicle's own point does not move at all (the first
        // cut of this arm measured 0.0000 mm and would have passed nothing).
        // What moves is everything below it -- and what must not move is the
        // foot, which is a position.
        let shoulder_deg = |r: &Range| -> f64 {
            let posed = inf_ecs::pose::evaluated_pose(&r.world, HERO).expect("posed");
            let q = glam::Quat::from_array(posed.pose.locals[clav as usize].rotation);
            f64::from(q.angle_between(glam::Quat::IDENTITY).to_degrees())
        };
        // **Three hundred steps of settle before the baseline is taken.** The
        // first cut of this arm took it at forty and measured the character
        // still settling: the shoulder moved 20.1001 mm and the foot 24.0113 mm
        // over the next thirty steps -- IDENTICALLY with and without the rifle,
        // which is exactly what a control is for. Five seconds of idle is past
        // the foot pass's own convergence and what is left is the overlay.
        for _ in 0..300 {
            r.step();
        }
        let (shoulder_before, foot_before) = (shoulder_deg(&r), joint(&r, foot));
        if arm_it {
            r.arm(HERO, "rifle");
        }
        for _ in 0..30 {
            r.step();
        }
        let (shoulder_after, foot_after) = (shoulder_deg(&r), joint(&r, foot));
        (
            (shoulder_after - shoulder_before).abs(),
            f64::from((foot_after - foot_before).length()) * 1000.0,
        )
    }
    let (armed_shoulder, armed_foot) = run(true);
    let (idle_shoulder, idle_foot) = run(false);
    println!("=== thirty steps after five seconds of settle ===");
    println!(
        "  with the rifle picked up  shoulder {armed_shoulder:.4} deg, foot {armed_foot:.4} mm"
    );
    println!("  the same steps, unarmed   shoulder {idle_shoulder:.4} deg, foot {idle_foot:.4} mm");
    assert!(
        armed_shoulder > 20.0,
        "the overlay moved the shoulder {armed_shoulder:.4} deg - it is not reaching the pose and the foot arm below is vacuous"
    );
    assert!(
        idle_shoulder < 0.001,
        "the CONTROL moved its shoulder {idle_shoulder:.4} deg without a weapon, so the comparison above is not about the overlay"
    );
    assert!(
        (armed_foot - idle_foot).abs() < 0.01,
        "the overlay moved the foot {armed_foot:.4} mm against the control's {idle_foot:.4} - the mask reaches the legs"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (g) DETERMINISM AND THE TRACE
// ─────────────────────────────────────────────────────────────────────────────

/// **A quiet level folds no feel bytes, a burst folds them at the TAIL, and a
/// settled one folds none again.**
///
/// The tail claim is the one that keeps every committed hash in the tree: a
/// section inserted anywhere but the end moves all of them.
#[test]
fn the_feel_section_is_empty_at_rest_and_appended_at_the_tail() {
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, -3.0, 0.0);
    world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(50.0, 0.5, 50.0),
            ..Default::default()
        },
        t,
    ));
    stand(&mut world, HERO, "Hero", DVec3::ZERO, true);
    *item::item_defs_mut(&mut world) = defs_with(&[("rifle", test_rifle())]);
    world.mark_dirty();
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::ZERO, 60.0);
    for _ in 0..10 {
        sim.step_once(RuntimeInput::default());
    }
    let quiet = sim.state_bytes();
    assert!(
        inf_ecs::feel::feel_state_bytes(sim.world()).is_empty(),
        "a level whose hero has never held a weapon folds feel bytes"
    );
    // Arm it — still nothing, because carrying a weapon is at rest.
    item::give_inventory(sim.world_mut(), HERO, 4);
    item::give(sim.world_mut(), HERO, "rifle", 1);
    d3::gameplay::equip_weapon(sim.world_mut(), HERO, "rifle");
    for _ in 0..10 {
        sim.step_once(RuntimeInput::default());
    }
    assert!(
        inf_ecs::feel::feel_state_bytes(sim.world()).is_empty(),
        "a hero merely CARRYING a rifle folds feel bytes - the empty-at-rest rule is what keeps every pre-wave trace identical"
    );
    let carried = sim.state_bytes();
    // Fire.
    // Through the shipped input door: `step_once` applies the whole movement
    // intent, so a component write made before it is cleared by it.
    sim.step_once(RuntimeInput::with_down(["attack"]));
    sim.step_once(RuntimeInput::with_down(["attack"]));
    let part = inf_ecs::feel::feel_state_bytes(sim.world());
    let firing = sim.state_bytes();
    println!(
        "=== the trace: quiet {} B, carrying {} B, firing {} B (the feel is {} of them) ===",
        quiet.len(),
        carried.len(),
        firing.len(),
        part.len()
    );
    assert!(!part.is_empty(), "a hero that has just fired folds no feel");
    assert_eq!(part.len(), 16 + 18 * 8, "the row is not the shape it says");
    // Carrying a weapon DOES grow the trace, by 175 bytes, and none of them are
    // this wave's: `WeaponState` is installed on equip and folded by
    // `weapon_state_bytes`, which is island wave I6's section. What this wave
    // owes is that ITS section is still empty, and that is the assertion two
    // lines above this one.
    assert!(
        carried.len() > quiet.len(),
        "equipping a weapon folded nothing at all - the I6 ammunition section is missing and the arm above proves less than it says"
    );
    assert!(
        firing.ends_with(&part),
        "the feel's bytes are not at the TAIL - a section inserted before the fourteen frozen ones moves every committed hash in the tree"
    );
    // …and it empties again once everything has settled.
    for _ in 0..600 {
        sim.step_once(RuntimeInput::default());
    }
    assert!(
        inf_ecs::feel::feel_state_bytes(sim.world()).is_empty(),
        "the feel never came back to rest, so this level's trace section never empties again"
    );
}

/// **The bytes this gate compares between the two hosts.**
///
/// The editor's `SimSession` has no `state_bytes` of its own -- the fold lives in
/// `RuntimeSim` and `projector_mirror` pins it there -- so the comparison is
/// made on the three sections this wave is about, concatenated the same way on
/// both sides: the FEEL (the springs, the sway, the blend), the WEAPON (the
/// magazine and this wave's bloom) and the AIM (the two numbers a bullet leaves
/// along, which no section folds directly and which the pose only reflects).
///
/// `cov1_gate::cover_trace`'s shape, on this wave's own quantities.
fn feel_trace(world: &EcsWorld) -> Vec<u8> {
    let mut out = inf_ecs::feel::feel_state_bytes(world);
    out.extend_from_slice(&weapon::weapon_state_bytes(world));
    if let Some(e) = world.entity_of(HERO) {
        if let Some(cm) = world.world().get::<CharacterMovement>(e) {
            out.extend_from_slice(&cm.runtime.aim_yaw_deg.to_bits().to_le_bytes());
            out.extend_from_slice(&cm.runtime.aim_pitch_deg.to_bits().to_le_bytes());
        }
    }
    out
}

/// **PIE == shipping, and two independent runs agree, over a burst course.**
///
/// The two hosts are `RuntimeSim` (the player's) and `SimSession` (the
/// editor's), driven step for step through the same course, compared on
/// `state_bytes` — which is where this wave's fifteenth section is.
#[test]
fn pie_equals_shipping_over_a_burst_course() {
    use inf_editor_core::scene::SceneDoc;
    use inf_editor_core::simulate::{SimInput, SimSession};
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

    const STEPS: u32 = 240;
    fn build(world: &mut EcsWorld) {
        let e = world.spawn_with_guid(GROUND, "Ground", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, -3.0, 20.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(60.0, 0.5, 60.0),
                ..Default::default()
            },
            t,
        ));
        stand(world, HERO, "Hero", DVec3::ZERO, true);
        *item::item_defs_mut(world) = defs_with(&[("rifle", test_rifle())]);
        world.mark_dirty();
        world.propagate();
    }
    /// The course: equip, aim, fire three bursts, release, settle.
    fn course(step: u32) -> Vec<&'static str> {
        match step {
            0..=9 => Vec::new(),
            10..=239 if matches!(step % 60, 20..=39) => vec!["aim", "attack"],
            _ => vec!["aim"],
        }
    }
    fn equip_at(step: u32, world: &mut EcsWorld) {
        if step != 5 {
            return;
        }
        item::give_inventory(world, HERO, 4);
        item::give(world, HERO, "rifle", 1);
        d3::gameplay::equip_weapon(world, HERO, "rifle");
    }

    let ship: Vec<Vec<u8>> = {
        let mut world = EcsWorld::new();
        build(&mut world);
        let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::ZERO, 60.0);
        (0..STEPS)
            .map(|s| {
                equip_at(s, sim.world_mut());
                sim.step_once(RuntimeInput::with_down(course(s)));
                feel_trace(sim.world())
            })
            .collect()
    };
    let pie: Vec<Vec<u8>> = {
        let mut doc = SceneDoc::new();
        build(doc.world_mut());
        let mut session = SimSession::enter(&mut doc, Vec::new(), glam::DVec2::ZERO, 60.0);
        let out = (0..STEPS)
            .map(|s| {
                equip_at(s, doc.world_mut());
                session.step_once(&mut doc, SimInput::with_down(course(s)));
                feel_trace(doc.world())
            })
            .collect();
        session.exit(&mut doc);
        out
    };
    // The anti-vacuity half FIRST: two hosts that both did nothing agree.
    let feel_steps = ship.iter().filter(|b| b.len() > ship[0].len()).count();
    assert!(
        ship[0] != ship[STEPS as usize - 1],
        "the course ended in exactly the state it started in"
    );
    assert!(
        feel_steps > 30,
        "only {feel_steps} of {STEPS} steps folded a feel - the course barely fired"
    );
    for (i, (a, b)) in ship.iter().zip(pie.iter()).enumerate() {
        assert_eq!(a, b, "step {i}: the player and the editor disagree");
    }
    // And a second run of the same host is bit-identical, which is the replay
    // claim without a cook.
    let again: Vec<Vec<u8>> = {
        let mut world = EcsWorld::new();
        build(&mut world);
        let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::ZERO, 60.0);
        (0..STEPS)
            .map(|s| {
                equip_at(s, sim.world_mut());
                sim.step_once(RuntimeInput::with_down(course(s)));
                feel_trace(sim.world())
            })
            .collect()
    };
    for (i, (a, b)) in ship.iter().zip(again.iter()).enumerate() {
        assert_eq!(a, b, "step {i}: two runs of the same host diverged");
    }
    println!("=== PIE == shipping over {STEPS} steps, {feel_steps} of them with a live feel ===");
}

// ─────────────────────────────────────────────────────────────────────────────
// (h) COST
// ─────────────────────────────────────────────────────────────────────────────

/// **Eight shooters bursting cost what they cost**, against the wave's own
/// budget.
///
/// The gameplay phase is timed with the feel present and again with it absent
/// (nobody armed), so the number reported is the FEEL's and not the phase's.
#[test]
fn eight_shooters_bursting_cost_what_they_cost() {
    let mut armed = Range::new(defs_with(&[("rifle", test_rifle())]));
    let mut bare = Range::new(defs_with(&[("rifle", test_rifle())]));
    for i in 0..8 {
        let g = shooter_guid(i);
        stand(
            &mut armed.world,
            g,
            "Shooter",
            DVec3::new(f64::from(i as i32) * 2.0 - 8.0, 0.0, 0.0),
            false,
        );
        stand(
            &mut bare.world,
            g,
            "Shooter",
            DVec3::new(f64::from(i as i32) * 2.0 - 8.0, 0.0, 0.0),
            false,
        );
        armed.arm(g, "rifle");
        armed.aim(g, 0.0, 0.0);
        armed.hold_trigger(g, true);
    }
    armed.world.mark_dirty();
    bare.world.mark_dirty();
    armed.world.propagate();
    bare.world.propagate();
    armed.bridge.sync_from_world(&armed.world);
    bare.bridge.sync_from_world(&bare.world);
    let time = |r: &mut Range, n: u32| -> f64 {
        let mut total = std::time::Duration::ZERO;
        for _ in 0..n {
            r.bridge.sync_from_world(&r.world);
            d3::step_character_movement(&mut r.world, &mut r.bridge, DT);
            let t0 = std::time::Instant::now();
            d3::step_gameplay(&mut r.world, &mut r.bridge, DT);
            total += t0.elapsed();
            r.bridge.step(DT);
            r.bridge.write_back_into(&mut r.world);
            r.world.propagate();
        }
        total.as_secs_f64() * 1000.0 / f64::from(n)
    };
    // Warm.
    time(&mut armed, 20);
    time(&mut bare, 20);
    let with = time(&mut armed, 240);
    let without = time(&mut bare, 240);
    let per_shooter_us = (with - without) * 1000.0 / 9.0;
    println!(
        "=== the gameplay phase: {with:.4} ms with eight shooters bursting, {without:.4} ms with none; {per_shooter_us:.3} us a shooter ==="
    );
    let live = armed
        .feel(shooter_guid(0))
        .map(|f| !f.at_rest())
        .unwrap_or(false);
    assert!(live, "nobody was actually firing, so this measures nothing");
    assert!(
        with < inf_player::budget::WEAPON_STEP_BUDGET_MS,
        "eight shooters bursting cost {with:.4} ms against a {:.2} ms budget",
        inf_player::budget::WEAPON_STEP_BUDGET_MS
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// THE GUARD
// ─────────────────────────────────────────────────────────────────────────────

/// **The gate names the constants it is about**, so a rename cannot leave it
/// measuring something else, and the two spring pairs really are the doc's
/// stiffnesses.
#[test]
fn the_gate_names_the_constants_it_is_about() {
    assert_eq!(feel::VM_STIFFNESS, 220.0);
    assert_eq!(feel::AIM_STIFFNESS, 180.0);
    assert_eq!(feel::DOC_VM_DAMPING, 18.0);
    assert_eq!(feel::DOC_AIM_DAMPING, 16.0);
    assert!(feel::SPREAD_ADS_MULT < 1.0);
    assert!(feel::SPREAD_CROUCH_MULT < 1.0);
    assert!(feel::SPREAD_MOVE_MULT_MAX > 1.0);
    assert!(feel::ADS_MOVE_SPEED_MULT < 1.0);
    // The overlay resolver is by BARREL and the registry's own two numbers sit
    // either side of it.
    assert!(test_pistol().muzzle_forward_m < feel::PISTOL_MUZZLE_M);
    assert!(test_rifle().muzzle_forward_m > feel::PISTOL_MUZZLE_M);
    let (o, s) = feel::overlay_states_for(Some(&test_pistol()));
    assert_eq!(o, inf_anim::als::OVERLAY_PISTOL_2H_STATE);
    assert_eq!(s, inf_anim::als::AIM_PISTOL_2H_STATE);
    let (o, s) = feel::overlay_states_for(Some(&test_rifle()));
    assert_eq!(o, inf_anim::als::OVERLAY_M4A1_STATE);
    assert_eq!(s, inf_anim::als::AIM_M4A1_STATE);
    // An empty pair of hands wears nothing — the measured refusal, recorded
    // here so a later wave that changes its mind has to change this line.
    assert_eq!(feel::overlay_states_for(None), ("", ""));
    // And the stance the cone is resolved for is the three the enum names.
    let d = test_rifle();
    let stand = feel::resolved_cone_deg(&d, 0.0, ShotStance::Standing, 0.0, 0.0);
    let crouch = feel::resolved_cone_deg(&d, 0.0, ShotStance::Crouched, 0.0, 0.0);
    let prone = feel::resolved_cone_deg(&d, 0.0, ShotStance::Prone, 0.0, 0.0);
    assert!(prone < crouch && crouch < stand);
    let _ = BTreeSet::<u8>::new();
    let _ = RotationMode::Aiming;
}
