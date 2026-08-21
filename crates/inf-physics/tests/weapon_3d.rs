//! **I6: weapons, inventories and health, against a real world.**
//!
//! `inf_ecs::weapon`'s and `inf_ecs::item`'s own tests pin the *rules* — the
//! fire clock, the stacking, the joules — as functions of numbers. These pin
//! what happens when those rules meet a world: whether a shot really casts a
//! ray, whether a hit really reaches the thing it hit, whether a body that runs
//! out of joules really goes limp.
//!
//! **Every arm asserts the WORLD**: a target's remaining joules, a character's
//! movement mode, a magazine read back out of the ECS — never a report's own
//! summary of itself.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    Destructible, MovementMode, RigidBody3D, RotationMode, Transform,
};
use inf_ecs::item::{self, ItemDef, ItemDefs};
use inf_ecs::math::Vec3d;
use inf_ecs::movement::MovementIntent;
use inf_ecs::weapon::{self, Health, ShotKind, WeaponDef, WeaponState};
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);

const HERO: Uuid = Uuid::from_u128(0x1601_0001);
const GROUND: Uuid = Uuid::from_u128(0x1601_0002);
const TARGET: Uuid = Uuid::from_u128(0x1601_0003);
const WALL: Uuid = Uuid::from_u128(0x1601_0004);
const LOOT: Uuid = Uuid::from_u128(0x1601_0005);

const RADIUS: f64 = 0.3;
/// The rifle's own damage, so every claim below reads against one number.
const RIFLE_J: f64 = 1700.0;

fn defs() -> ItemDefs {
    let mut d = ItemDefs::default();
    assert!(d.insert(ItemDef {
        id: "rifle".into(),
        label: "Rifle".into(),
        stack_max: 1,
        mass_kg: 3.6,
        weapon: Some(WeaponDef {
            // No spread, so a gate can name where the bullet went.
            spread_deg: 0.0,
            damage_j: RIFLE_J,
            ..Default::default()
        }),
    }));
    assert!(d.insert(ItemDef {
        id: "pistol".into(),
        label: "Pistol".into(),
        stack_max: 1,
        mass_kg: 0.9,
        weapon: Some(WeaponDef {
            spread_deg: 0.0,
            damage_j: 500.0,
            automatic: false,
            magazine: 12,
            kind: ShotKind::Projectile,
            spread_seed: 7,
            ..Default::default()
        }),
    }));
    assert!(d.insert(ItemDef {
        id: "bandage".into(),
        label: "Bandage".into(),
        stack_max: 5,
        mass_kg: 0.1,
        weapon: None,
    }));
    d
}

fn spawn_ground(w: &mut EcsWorld) {
    let e = w.spawn_with_guid(GROUND, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, -0.5, 0.0);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(80.0, 0.5, 80.0),
            ..Default::default()
        },
        t,
    ));
}

fn spawn_hero(w: &mut EcsWorld) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let e = w.spawn_with_guid(HERO, "Hero", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, cm.stand_half_height_m + RADIUS, 0.0);
    w.world_mut().entity_mut(e).insert((
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

/// A standing body at `z`, with joules to give — the thing a bullet is aimed at.
fn spawn_target(w: &mut EcsWorld, z: f64, capacity_j: f64) {
    let cm = CharacterMovement::default();
    let e = w.spawn_with_guid(TARGET, "Target", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, cm.stand_half_height_m + RADIUS, z);
    w.world_mut().entity_mut(e).insert((
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
        Health::new(capacity_j),
        t,
    ));
}

/// A destructible wall at `z` — what a shot that misses flesh hits.
fn spawn_wall(w: &mut EcsWorld, z: f64) {
    let e = w.spawn_with_guid(WALL, "Wall", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, 1.4, z);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(3.0, 1.5, 0.2),
            ..Default::default()
        },
        Destructible::default(),
        t,
    ));
}

struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Rig {
    fn new() -> Self {
        let mut world = EcsWorld::new();
        spawn_ground(&mut world);
        spawn_hero(&mut world);
        *item::item_defs_mut(&mut world) = defs();
        world.mark_dirty();
        world.propagate();
        let mut rig = Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
        };
        rig.bridge.sync_from_world(&rig.world);
        rig
    }

    fn arm(&mut self, id: &str) {
        assert!(item::give_inventory(&mut self.world, HERO, 6));
        let defs = defs();
        let e = self.world.entity_of(HERO).expect("the hero");
        {
            let mut inv = self
                .world
                .world_mut()
                .get_mut::<item::Inventory>(e)
                .expect("a bag");
            assert_eq!(inv.add(&defs, id, 1), 0, "the bag would not take a {id}");
        }
        assert!(
            d3::gameplay::equip_weapon(&mut self.world, HERO, id),
            "the {id} would not equip"
        );
    }

    fn step(&mut self, intent: &MovementIntent) -> d3::GameplayReport {
        self.bridge.sync_from_world(&self.world);
        inf_ecs::movement::apply_intent(&mut self.world, intent);
        d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
        let report = d3::step_gameplay(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        report
    }

    fn steps(&mut self, intent: &MovementIntent, n: u32) -> Vec<d3::GameplayReport> {
        (0..n).map(|_| self.step(intent)).collect()
    }

    fn magazine(&self) -> u32 {
        let e = self.world.entity_of(HERO).expect("the hero");
        self.world
            .world()
            .get::<WeaponState>(e)
            .expect("an ammunition clock")
            .magazine
    }

    fn reserve(&self) -> u32 {
        let e = self.world.entity_of(HERO).expect("the hero");
        self.world
            .world()
            .get::<WeaponState>(e)
            .expect("an ammunition clock")
            .reserve
    }

    fn equipped(&self) -> Option<String> {
        item::inventory_of(&self.world, HERO)?
            .equipped_id()
            .map(|s| s.to_string())
    }

    fn rotation_mode(&self) -> RotationMode {
        let e = self.world.entity_of(HERO).expect("the hero");
        self.world
            .world()
            .get::<CharacterMovement>(e)
            .expect("a movement component")
            .rotation_mode
    }

    fn target_health(&self) -> Health {
        weapon::health_of(&self.world, TARGET).expect("the target has health")
    }

    fn target_mode(&self) -> MovementMode {
        let e = self.world.entity_of(TARGET).expect("the target");
        self.world
            .world()
            .get::<CharacterMovement>(e)
            .expect("a movement component")
            .mode
    }
}

fn idle() -> MovementIntent {
    MovementIntent::default()
}

fn hold_trigger() -> MovementIntent {
    MovementIntent {
        attack: true,
        attack_pressed: true,
        ..Default::default()
    }
}

fn hold_trigger_no_edge() -> MovementIntent {
    MovementIntent {
        attack: true,
        ..Default::default()
    }
}

fn aim() -> MovementIntent {
    MovementIntent {
        aim: true,
        ..Default::default()
    }
}

fn press_reload() -> MovementIntent {
    MovementIntent {
        reload: true,
        ..Default::default()
    }
}

fn scroll(dir: i32) -> MovementIntent {
    MovementIntent {
        weapon_switch: dir,
        ..Default::default()
    }
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **THE HEADLINE: a shot is a ray, and what it hits loses joules.**
#[test]
fn a_round_reaches_the_body_it_is_aimed_at_and_takes_its_joules() {
    let mut rig = Rig::new();
    spawn_target(&mut rig.world, 8.0, 5000.0);
    rig.world.mark_dirty();
    rig.world.propagate();
    rig.arm("rifle");
    rig.step(&idle());
    let before = rig.target_health().joules;
    let r = rig.step(&hold_trigger());
    println!(
        "one rifle round: {} shot(s), {} hit(s), the target went {before} -> {} J",
        r.shots,
        r.hits.len(),
        rig.target_health().joules
    );
    assert_eq!(r.shots, 1, "the trigger did not fire");
    assert_eq!(r.hits.len(), 1);
    let hit = r.hits[0];
    assert_eq!(hit.target, Some(TARGET), "the shot hit {:?}", hit.target);
    assert!(
        hit.on_flesh,
        "the target has health and the shot says otherwise"
    );
    // **Assert the WORLD**: the target's own joules, read back out of the ECS.
    assert!(
        (before - rig.target_health().joules - RIFLE_J).abs() < 1e-9,
        "the target lost {} J and the round carries {RIFLE_J}",
        before - rig.target_health().joules
    );
    // …and the magazine really spent a round.
    assert_eq!(rig.magazine(), WeaponDef::default().magazine - 1);
    // The shot ended AT the target rather than at the end of its range.
    let travelled = (hit.to - hit.from).length();
    println!("the round travelled {travelled} m to a target 8 m away");
    assert!(travelled < 8.5, "{travelled}");
    // A control: with nothing in the way the same shot runs to its range.
    let mut empty = Rig::new();
    empty.arm("rifle");
    empty.step(&idle());
    let r = empty.step(&hold_trigger());
    let miss = r.hits[0];
    assert_eq!(miss.target, None);
    println!(
        "with nothing in front of it the same round reached {} m",
        (miss.to - miss.from).length()
    );
    assert!((miss.to - miss.from).length() > 100.0);
}

/// **Two rounds stop a 2 000 J body, and the body goes limp.**
///
/// The whole health→ragdoll handoff, measured as the target's `MovementMode`.
#[test]
fn a_body_that_runs_out_of_joules_is_handed_to_the_ragdoll() {
    let mut rig = Rig::new();
    spawn_target(&mut rig.world, 8.0, weapon::DEFAULT_VITALITY_J);
    rig.world.mark_dirty();
    rig.world.propagate();
    rig.arm("rifle");
    rig.step(&idle());
    assert_eq!(rig.target_mode(), MovementMode::Grounded);
    // Hold the trigger: an automatic rifle at 600 rpm fires every tenth of a
    // second, so two rounds take about twelve steps.
    let mut shots = 0u32;
    let mut kills = 0u32;
    let mut modes: Vec<MovementMode> = Vec::new();
    for _ in 0..30 {
        let r = rig.step(&hold_trigger_no_edge());
        shots += r.shots;
        kills += r.kills;
        modes.push(rig.target_mode());
    }
    let h = rig.target_health();
    println!(
        "{shots} rounds of {RIFLE_J} J against a {} J body: {} J left, dead = {}, {kills} handoff(s); the modes it passed through were {:?}",
        weapon::DEFAULT_VITALITY_J,
        h.joules,
        h.dead,
        {
            let mut seen: Vec<MovementMode> = Vec::new();
            for m in &modes {
                if seen.last() != Some(m) {
                    seen.push(*m);
                }
            }
            seen
        }
    );
    assert!(shots >= 2, "only {shots} rounds went off");
    assert!(h.dead, "{h:?}");
    assert_eq!(h.joules, 0.0);
    assert_eq!(kills, 1, "the handoff fired {kills} times");
    // **The WORLD**: the body really entered the ragdoll.
    assert!(
        modes.contains(&MovementMode::Ragdoll),
        "the body never went limp: {modes:?}"
    );
    // …and the latch is down, which is what makes the handoff once.
    assert!(weapon::is_downed(&rig.world, TARGET));
    // …so however many more rounds arrive, nothing is handed over again.
    let more: u32 = rig
        .steps(&hold_trigger_no_edge(), 30)
        .iter()
        .map(|r| r.kills)
        .sum();
    assert_eq!(more, 0, "the handoff fired again");
    // **THE BOUND, measured rather than hidden.** This fixture's target has no
    // skeleton, so no rig ever answers the ragdoll bridge's request and P29.4's
    // own "no rig is coming" branch hands the character straight back after
    // `RIG_WAIT_S`. A corpse with no rig therefore ends up standing: death is
    // visible as `Health::dead` plus `Downed`, and the limp body is what a
    // rigged character gets. The arm says so out loud rather than asserting a
    // mode that would be a lie about this fixture.
    println!(
        "with no skeleton to build a ragdoll from, the body ended in {:?} - the rigless bound",
        rig.target_mode()
    );
}

/// **A miss on flesh is energy owed to the P22 door**, and the gameplay step
/// does not spend it — the host does, through its own wrapper.
#[test]
fn a_round_into_a_wall_comes_back_as_joules_for_the_hosts_damage_wrapper() {
    let mut rig = Rig::new();
    spawn_wall(&mut rig.world, 6.0);
    rig.world.mark_dirty();
    rig.world.propagate();
    rig.arm("rifle");
    rig.step(&idle());
    let r = rig.step(&hold_trigger());
    assert_eq!(r.shots, 1);
    assert_eq!(r.hits.len(), 1);
    assert!(!r.hits[0].on_flesh, "a wall is not flesh");
    assert_eq!(r.hits[0].target, Some(WALL));
    println!("a round into a wall owes {:?} to the P22 door", r.destruct);
    assert_eq!(r.destruct, vec![(WALL, RIFLE_J)]);
    // A burst on one wall is ONE blow, coalesced — because damage is not
    // banked, and letting three small blows arrive separately would make the
    // rate of fire a hidden multiplier on damage.
    let mut burst = Rig::new();
    spawn_wall(&mut burst.world, 6.0);
    burst.world.mark_dirty();
    burst.world.propagate();
    burst.arm("rifle");
    burst.step(&idle());
    // Fire twice on one step by shortening the cycle to nothing.
    {
        let e = burst.world.entity_of(HERO).expect("the hero");
        let mut st = burst
            .world
            .world_mut()
            .get_mut::<WeaponState>(e)
            .expect("a clock");
        st.cooldown_s = 0.0;
    }
    let r = burst.step(&hold_trigger());
    assert_eq!(r.destruct.len(), 1, "one wall, one entry: {:?}", r.destruct);
}

/// **The reload is gated on the animation's notify**, with a fuse for a
/// character nothing is animating.
#[test]
fn a_reload_waits_for_its_notify_and_falls_back_to_its_clock() {
    let mut rig = Rig::new();
    rig.arm("rifle");
    rig.step(&idle());
    // Spend some rounds.
    rig.steps(&hold_trigger_no_edge(), 60);
    let spent = rig.magazine();
    assert!(spent < WeaponDef::default().magazine, "nothing was fired");
    let reserve_before = rig.reserve();
    rig.step(&press_reload());
    // It has NOT finished on the press.
    assert_eq!(rig.magazine(), spent, "the reload finished instantly");
    // The notify lands it early.
    {
        let mut map = std::collections::BTreeMap::new();
        map.insert(HERO, vec![weapon::RELOAD_NOTIFY.to_string()]);
        rig.world
            .world_mut()
            .insert_resource(inf_ecs::pose::AnimEventsRes(map));
    }
    let r = rig.step(&idle());
    println!(
        "the notify landed the reload: magazine {spent} -> {}, reserve {reserve_before} -> {}",
        rig.magazine(),
        rig.reserve()
    );
    assert_eq!(r.reloads, 1, "the notify did not land the reload");
    assert_eq!(rig.magazine(), WeaponDef::default().magazine);
    assert!(rig.reserve() < reserve_before, "the reserve paid nothing");

    // The CLOCK path: the same reload with no animation at all.
    let mut clock = Rig::new();
    clock.arm("rifle");
    clock.step(&idle());
    clock.steps(&hold_trigger_no_edge(), 60);
    let spent = clock.magazine();
    clock.step(&press_reload());
    let mut n = 0;
    let mut done = 0;
    while n < 300 && done == 0 {
        done += clock.step(&idle()).reloads;
        n += 1;
    }
    println!("with nothing animating it, the reload finished on its clock after {n} steps");
    assert_eq!(done, 1);
    assert!(clock.magazine() > spent);
    assert!(
        (n as f64 * DT - WeaponDef::default().reload_s).abs() < 0.05,
        "{n} steps is {} s",
        n as f64 * DT
    );
}

/// **The wheel changes the weapon, and the ammunition clock goes with it.**
#[test]
fn the_wheel_cycles_the_equipped_weapon_and_replaces_its_magazine() {
    let mut rig = Rig::new();
    rig.arm("rifle");
    // A second weapon and a non-weapon in the same bag.
    {
        let defs = defs();
        let e = rig.world.entity_of(HERO).expect("the hero");
        let mut inv = rig
            .world
            .world_mut()
            .get_mut::<item::Inventory>(e)
            .expect("a bag");
        assert_eq!(inv.add(&defs, "pistol", 1), 0);
        assert_eq!(inv.add(&defs, "bandage", 3), 0);
    }
    rig.step(&idle());
    assert_eq!(rig.equipped().as_deref(), Some("rifle"));
    assert_eq!(rig.magazine(), 30);
    rig.step(&scroll(1));
    println!("one notch forward equipped {:?}", rig.equipped());
    assert_eq!(
        rig.equipped().as_deref(),
        Some("pistol"),
        "the wheel did not reach the pistol"
    );
    // The magazine is the PISTOL's, not the rifle's — two weapons must not
    // share one clock.
    assert_eq!(
        rig.magazine(),
        12,
        "the pistol inherited the rifle's magazine"
    );
    // …and it skips the bandage: the filter is "is it a weapon".
    rig.step(&scroll(1));
    assert_eq!(
        rig.equipped().as_deref(),
        Some("rifle"),
        "the wheel equipped a bandage"
    );
    assert_eq!(rig.magazine(), 30);
    // Backwards works too.
    rig.step(&scroll(-1));
    assert_eq!(rig.equipped().as_deref(), Some("pistol"));
    // …and the wheel's sign is CONSUMED, so a frame that produced one notch
    // does not keep cycling.
    let was = rig.equipped();
    rig.steps(&idle(), 5);
    assert_eq!(
        rig.equipped(),
        was,
        "the wheel kept turning after the notch"
    );
}

/// **RMB is the aim**, and it reaches the rotation mode the camera reads.
#[test]
fn holding_aim_puts_the_character_in_the_aiming_rotation_mode() {
    let mut rig = Rig::new();
    rig.arm("rifle");
    rig.step(&idle());
    assert_ne!(rig.rotation_mode(), RotationMode::Aiming);
    rig.step(&aim());
    println!("holding aim put the character in {:?}", rig.rotation_mode());
    assert_eq!(rig.rotation_mode(), RotationMode::Aiming);
    // Letting go leaves it looking rather than back where it started, which is
    // P29.6's own rule and not this wave's to change.
    rig.step(&idle());
    assert_eq!(rig.rotation_mode(), RotationMode::LookingDirection);
}

/// **A semi-automatic weapon fires once per press even with the button held.**
#[test]
fn a_semi_automatic_weapon_needs_the_trigger_released() {
    let mut rig = Rig::new();
    rig.arm("pistol");
    rig.step(&idle());
    // Held for a second: at 600 rpm an automatic would fire ten times.
    let held: u32 = rig
        .steps(&hold_trigger_no_edge(), 60)
        .iter()
        .map(|r| r.shots)
        .sum();
    println!("a semi-automatic pistol held for one second fired {held} round(s)");
    assert_eq!(
        held, 1,
        "a semi-automatic weapon fired {held} times on one press"
    );
    // Release and press again.
    rig.steps(&idle(), 10);
    let again: u32 = rig
        .steps(&hold_trigger_no_edge(), 10)
        .iter()
        .map(|r| r.shots)
        .sum();
    assert_eq!(again, 1, "the second press did not fire");
}

/// **THE PICK-UP, through I5's one interaction site.**
///
/// Not through `item::pick_up` directly: the claim is that the E key reaches
/// it, which is a claim about the movement step's verb dispatch.
#[test]
fn the_e_key_picks_an_item_up_off_the_floor() {
    let mut rig = Rig::new();
    assert!(item::give_inventory(&mut rig.world, HERO, 6));
    assert_eq!(
        item::spawn_pickup(&mut rig.world, LOOT, "rifle", 1, Vec3d::new(0.0, 0.4, 1.2)),
        Some(LOOT)
    );
    rig.world.mark_dirty();
    rig.world.propagate();
    rig.step(&idle());
    assert_eq!(
        item::inventory_of(&rig.world, HERO)
            .expect("a bag")
            .count_of("rifle"),
        0
    );
    rig.step(&MovementIntent {
        interact: true,
        ..Default::default()
    });
    println!(
        "after one E press the bag holds {} rifle(s) and the floor holds {}",
        item::inventory_of(&rig.world, HERO)
            .expect("a bag")
            .count_of("rifle"),
        u32::from(rig.world.entity_of(LOOT).is_some())
    );
    // **Assert the WORLD** on both sides: the bag has it and the floor does not.
    assert_eq!(
        item::inventory_of(&rig.world, HERO)
            .expect("a bag")
            .count_of("rifle"),
        1,
        "the E key did not reach the pick-up verb"
    );
    assert!(rig.world.entity_of(LOOT).is_none(), "the floor kept it too");
    // …and dropping it puts a new entity back in the world, in front of the
    // character, which the E key can take again.
    let dropped = item::drop_slot(&mut rig.world, HERO, 0, 1).expect("it dropped");
    rig.world.mark_dirty();
    rig.world.propagate();
    assert!(rig.world.entity_of(dropped).is_some());
    assert_eq!(
        item::inventory_of(&rig.world, HERO)
            .expect("a bag")
            .count_of("rifle"),
        0
    );
    rig.step(&MovementIntent {
        interact: true,
        ..Default::default()
    });
    assert_eq!(
        item::inventory_of(&rig.world, HERO)
            .expect("a bag")
            .count_of("rifle"),
        1,
        "the dropped rifle could not be picked up again"
    );
}

/// **An unarmed character costs the step nothing and carries no clock.**
#[test]
fn a_character_with_no_weapon_has_no_ammunition_clock() {
    let mut rig = Rig::new();
    rig.step(&idle());
    let e = rig.world.entity_of(HERO).expect("the hero");
    assert!(rig.world.world().get::<WeaponState>(e).is_none());
    let r = rig.step(&hold_trigger());
    assert_eq!(r.shots, 0, "an unarmed character fired something");
    assert!(r.hits.is_empty());
    assert!(inf_ecs::weapon::weapon_state_bytes(&rig.world).is_empty());
    // Arm it, then take the weapon away: the clock goes with it.
    rig.arm("rifle");
    rig.step(&idle());
    assert!(rig.world.world().get::<WeaponState>(e).is_some());
    {
        let mut inv = rig
            .world
            .world_mut()
            .get_mut::<item::Inventory>(e)
            .expect("a bag");
        inv.unequip();
    }
    rig.step(&idle());
    assert!(
        rig.world.world().get::<WeaponState>(e).is_none(),
        "a holstered weapon left its magazine behind"
    );
}
