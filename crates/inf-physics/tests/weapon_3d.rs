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

/// **A standing body at `z` with no [`Health`] at all** (wave WPN1) — the shape
/// every character in this engine has until something hurts it.
///
/// The one difference from [`spawn_target`] is the component that is missing,
/// and that is the whole point: before this wave a round into one of these was
/// `on_flesh == false`, went to the destructible branch, and was owed to an
/// entity with no `Destructible`.
fn spawn_bare_body(w: &mut EcsWorld, z: f64) {
    let cm = CharacterMovement::default();
    let e = w.spawn_with_guid(TARGET, "Bystander", None);
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

/// The guids the **rigged** hero's registries answer to (SK1b audit).
const SM: Uuid = Uuid::from_u128(0x1601_0010);
const SKEL: Uuid = Uuid::from_u128(0x1601_0011);

/// The thing with a handle on it (SK1c audit), and where it is — in front of
/// the hero, at about the height a door handle sits.
const HANDLE: Uuid = Uuid::from_u128(0x1601_0012);
const HANDLE_AT: DVec3 = DVec3::new(0.0, 1.1, 0.9);

struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    /// The hero's skeleton, when this fixture gave it one. `None` is the bare
    /// capsule every arm in this file but two runs on.
    skeleton: Option<inf_anim::SkeletonAsset>,
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
            skeleton: None,
        };
        rig.bridge.sync_from_world(&rig.world);
        rig
    }

    /// **Give the hero a rig**, so its weapon hangs off a hand instead of a
    /// height (SK1b audit). `socket` is what the skeleton publishes for its right
    /// hand — pass something the rig does *not* have to exercise the fallback.
    fn rig_the_hero(&mut self, socket: &str) {
        let mut asset =
            inf_anim::build_manny(&inf_anim::BodyParams::default()).expect("a mannequin");
        // Republish the hand socket under the name this fixture wants, and drop
        // every other spelling of it, so "the rig authors `hand_r`" is a property
        // of this fixture rather than of the generator's socket list.
        let hand = asset
            .skeleton
            .index_of("hand_r")
            .expect("the mannequin has a right hand");
        asset
            .sockets
            .retain(|s| s.name != d3::gameplay::WEAPON_SOCKET && s.name != socket);
        asset.sockets.push(inf_anim::Socket::new(socket, hand));
        let e = self.world.entity_of(HERO).expect("the hero");
        self.world.world_mut().entity_mut(e).insert((
            inf_ecs::components::AnimStateMachine {
                sm: Some(SM),
                ..Default::default()
            },
            inf_ecs::components::SkeletalMesh {
                mesh: None,
                skeleton: Some(SKEL),
            },
        ));
        self.world.reindex_guids();
        self.world.mark_dirty();
        self.skeleton = Some(asset);
    }

    /// The pose + attachment half of a host's fixed step, in the hosts' own
    /// order: gameplay, then the pose, then the attachments. Inert with no rig.
    fn step_pose_and_attachments(&mut self) {
        let Some(rig) = self.skeleton.clone() else {
            return;
        };
        let machine = inf_anim::StateMachine {
            states: vec![inf_anim::SmState::clip("idle", [0u8; 16])],
            transitions: Vec::new(),
            entry: 0,
            ..Default::default()
        };
        let machines = |g: Uuid| (g == SM).then_some(&machine);
        let skeletons = |g: Uuid| (g == SKEL).then_some(&rig);
        let clips = |_: inf_anim::ClipRef| None;
        let vars = |_: Uuid| std::collections::BTreeMap::new();
        inf_ecs::pose::step_pose_evaluation(
            &mut self.world,
            DT,
            &machines,
            &skeletons,
            &clips,
            &vars,
        );
        self.world.propagate();
        inf_ecs::update_attachments(&mut self.world);
        self.world.propagate();
    }

    fn arm(&mut self, id: &str) {
        assert!(item::give_inventory(&mut self.world, HERO, 6));
        // **The world's own catalogue, not this file's** (wave WPN1). They are
        // the same table for every arm that predates this one — `Rig::new` seeds
        // the world from `defs()` — and they stop being the same the moment a
        // test authors an item of its own, which the melee arm does. Reading the
        // world is also what an equip really does.
        let defs = item::item_defs(&self.world).cloned().unwrap_or_default();
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
        self.step_pose_and_attachments();
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

    /// **Put a thing with a handle on it in front of the hero** (SK1c audit).
    ///
    /// `InteractVerb::Grab` deliberately: it is the one verb `step_one` does not
    /// consume, so what this fixture measures is the *hand*, with no door
    /// swinging and no item leaving the floor to confuse it.
    fn handle(&mut self) {
        let e = self.world.spawn_with_guid(HANDLE, "Handle", None);
        self.world.world_mut().entity_mut(e).insert((
            Transform::from_translation(HANDLE_AT),
            inf_ecs::interact::Interactable {
                verb: inf_ecs::interact::InteractVerb::Grab,
                label: "handle".into(),
                grip: Some(inf_anim::GRIP_HANDLE.to_string()),
                ..Default::default()
            },
        ));
        self.world.reindex_guids();
        self.world.mark_dirty();
        self.world.propagate();
    }

    /// Fingertip-to-wrist span for `[left, right]`, metres — the aperture a
    /// closed hand shortens, measured off the pose the fixed step published.
    fn spans(&self) -> [f64; 2] {
        let asset = self.skeleton.as_ref().expect("a rigged hero");
        let ep = inf_ecs::pose::evaluated_pose(&self.world, HERO).expect("a pose");
        let g = inf_anim::global_transforms(&asset.skeleton, &ep.pose);
        let at = |n: &str| {
            let i = asset.skeleton.index_of(n).expect(n) as usize;
            let p = g[i].to_scale_rotation_translation().2;
            DVec3::new(p.x as f64, p.y as f64, p.z as f64)
        };
        [
            (at("middle_03_l") - at("hand_l")).length(),
            (at("middle_03_r") - at("hand_r")).length(),
        ]
    }

    /// Where `[left, right]` wrist is, world metres.
    fn wrists(&self) -> [DVec3; 2] {
        let asset = self.skeleton.as_ref().expect("a rigged hero");
        let ep = inf_ecs::pose::evaluated_pose(&self.world, HERO).expect("a pose");
        let g = inf_anim::global_transforms(&asset.skeleton, &ep.pose);
        let at = |n: &str| {
            let i = asset.skeleton.index_of(n).expect(n) as usize;
            let p = g[i].to_scale_rotation_translation().2;
            DVec3::new(p.x as f64, p.y as f64, p.z as f64)
        };
        [at("hand_l"), at("hand_r")]
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

fn press_interact() -> MovementIntent {
    MovementIntent {
        interact: true,
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

/// **An unarmed character costs the step nothing and carries no clock** — until
/// it throws a punch.
///
/// The second half changed shape at wave WPN1 and the change is the finding: an
/// empty hand is now the third consumer of the attack edge, so *pressing the
/// button* installs a fist clock and swings. What the arm still holds — and what
/// keeps every trace committed before this wave byte-identical — is that an
/// unarmed character which has **not** pressed it carries nothing at all, and
/// that a fist is not a *shot*: it costs the ray cast nothing and resolves as a
/// swing.
#[test]
fn a_character_with_no_weapon_has_no_ammunition_clock() {
    let mut rig = Rig::new();
    let r = rig.step(&idle());
    let e = rig.world.entity_of(HERO).expect("the hero");
    assert!(rig.world.world().get::<WeaponState>(e).is_none());
    assert_eq!(r.shots, 0);
    assert!(inf_ecs::weapon::weapon_state_bytes(&rig.world).is_empty());
    // …and it stays empty over a whole second of doing nothing, which is the
    // half a lazy install that fired on every step would break.
    for r in rig.steps(&idle(), 60) {
        assert_eq!(r.shots, 0, "an idle unarmed character did something");
    }
    assert!(inf_ecs::weapon::weapon_state_bytes(&rig.world).is_empty());
    // The BUTTON is what changes it, and what it makes is a swing at nobody.
    let r = rig.step(&hold_trigger());
    assert_eq!(
        r.shots, 1,
        "the attack button did nothing with an empty hand"
    );
    assert_eq!(r.swings, 1, "the punch resolved as a shot");
    assert_eq!(r.hits.len(), 1);
    assert_eq!(
        r.hits[0].target, None,
        "there is nobody in the fixture to hit"
    );
    assert!(
        !inf_ecs::weapon::weapon_state_bytes(&rig.world).is_empty(),
        "the punch left no ammunition clock, so a second one could arrive on \
         the next step"
    );
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

/// **THE CONTROL for SK1b's muzzle change: the old number and the new path agree
/// on a capsule hero, exactly.**
///
/// The wave replaced "a shot leaves the character 1.4 m above its feet" with
/// "a shot leaves the weapon's own muzzle", and every level committed before it
/// — every fixture in this file, the whole `phase30-gameplay` course — has a hero
/// with no rig. `muzzle_of` sends those to the capsule rule, and this is the arm
/// that says the capsule rule is still the same rule. Asserted on `hit.from`,
/// which nothing in this file had ever looked at.
///
/// Mutation: change `MUZZLE_HEIGHT_M`, or drop the `evaluated_pose` guard in
/// `weapon_muzzle` so an unposed character reads its weapon entity's origin
/// (which is its capsule CENTRE, 0.2 m low), and this goes red.
#[test]
fn the_new_muzzle_agrees_with_the_old_one_on_a_capsule_hero() {
    let mut rig = Rig::new();
    rig.arm("rifle");
    rig.step(&idle());
    let r = rig.step(&hold_trigger());
    assert_eq!(r.shots, 1);
    let from = r.hits[0].from;
    let feet = d3::gameplay::feet_of(&rig.world, HERO).expect("the hero stands somewhere");
    let old = feet + glam::DVec3::Y * d3::gameplay::MUZZLE_HEIGHT_M;
    println!("the shot left {from:?}; the pre-SK1b rule says {old:?}");
    assert!(
        (from - old).length() < 1e-12,
        "the muzzle moved: {from:?} against {old:?}"
    );
    // …and the hero really has no pose to read a socket off, which is the
    // premise this control rests on.
    assert!(
        inf_ecs::pose::evaluated_pose(&rig.world, HERO).is_none(),
        "this fixture is supposed to be a bare capsule"
    );
}

/// **An equipped weapon is a real entity, attached to the hand** (SK1b), and it
/// leaves the world when nothing is equipped.
#[test]
fn the_equipped_weapon_is_an_entity_attached_to_the_hand_socket() {
    use inf_ecs::components::{AttachedTo, MeshRef};
    let mut rig = Rig::new();
    let weapon = d3::gameplay::equipped_weapon_guid(HERO);
    rig.step(&idle());
    assert!(
        rig.world.entity_of(weapon).is_none(),
        "an unarmed character carries a weapon entity"
    );

    rig.arm("rifle");
    rig.step(&idle());
    let e = rig
        .world
        .entity_of(weapon)
        .expect("an equipped weapon is an entity");
    let a = rig
        .world
        .world()
        .get::<AttachedTo>(e)
        .expect("it is attached");
    assert_eq!(a.target, HERO);
    assert_eq!(a.socket, d3::gameplay::WEAPON_SOCKET);
    assert!(
        rig.world.world().get::<MeshRef>(e).is_some(),
        "nothing would draw it"
    );
    // **…at the size of the weapon, not at the size of the unit cube** (SK1b
    // audit). This scale used to be written here and overwritten by
    // `inf_ecs::update_attachments` one pass later — that pass composed the
    // target's scale onto its follower — so the placeholder drew as a **1 m
    // cube** in the character's hand on every armed character in the tree, and
    // nothing looked. The attachment pass leaves a follower's size alone now
    // (`an_attachment_places_a_follower_without_resizing_it`); this is the half
    // that says the size it is left with is the barrel's.
    let t = *rig
        .world
        .world()
        .get::<inf_ecs::components::Transform>(e)
        .expect("a weapon has a transform");
    let barrel = defs()
        .get("rifle")
        .and_then(|d| d.weapon)
        .expect("the fixture's rifle")
        .muzzle_forward_m;
    assert!(
        (t.scale.z - barrel).abs() < 1e-9,
        "the placeholder is {:?} long against a {barrel} m barrel",
        t.scale.z
    );
    assert!(
        t.scale.x < 0.1 && t.scale.y < 0.1,
        "the placeholder is {:?} across, which is a crate and not a rifle",
        t.scale
    );
    // The guid is a pure function of the owner, so two hosts spawn the same
    // entity — the P22 content-derived rule, which is what makes this legal
    // inside a fixed step at all.
    assert_eq!(weapon, d3::gameplay::equipped_weapon_guid(HERO));
    assert_ne!(weapon, d3::gameplay::equipped_weapon_guid(TARGET));

    // Unequip and it goes: a holstered character is a character with no weapon
    // entity, not one with an invisible weapon still in the trace.
    {
        let h = rig.world.entity_of(HERO).expect("the hero");
        let mut inv = rig
            .world
            .world_mut()
            .get_mut::<item::Inventory>(h)
            .expect("a bag");
        inv.unequip();
    }
    rig.step(&idle());
    assert!(
        rig.world.entity_of(weapon).is_none(),
        "the weapon outlived its equip"
    );
}

/// **THE TRIPWIRE ON THE MUZZLE'S SILENT HALF** (SK1b audit).
///
/// `muzzle_of` has two answers and only one of them is a measurement. A rig-less
/// hero gets [`d3::gameplay::MUZZLE_HEIGHT_M`], which is right and is what
/// `the_new_muzzle_agrees_with_the_old_one_on_a_capsule_hero` pins. A **rigged**
/// hero whose skeleton stops publishing `hand_r` gets the same 1.4 m, silently —
/// and that is not right, it is a regression wearing the fallback's clothes.
/// Nothing in the wave could tell the two apart: the only muzzle arm in the tree
/// ran on a capsule.
///
/// Both branches, on the same rig, differing only in what the skeleton publishes:
///
/// * with `hand_r`, the shot leaves the weapon's own muzzle — **not** 1.4 m — and
///   `muzzles_without_a_socket` is zero;
/// * without it, the shot is back at 1.4 m and the counter names it.
#[test]
fn a_rigged_hero_shoots_from_its_weapon_and_says_so_when_it_cannot() {
    for (socket, from_weapon) in [
        (d3::gameplay::WEAPON_SOCKET, true),
        ("hand_of_glory", false),
    ] {
        let mut rig = Rig::new();
        rig.rig_the_hero(socket);
        rig.arm("rifle");
        // Two steps: `step_gameplay` runs BEFORE the pose and the attachments in
        // both hosts, so the transform a muzzle is read off is the one the
        // previous step settled. That one-step latency is the wave's own stated
        // bound; here it is the reason the shot is taken on the second step.
        rig.step(&idle());
        let r = rig.step(&hold_trigger());
        assert_eq!(r.shots, 1, "{socket}: the rifle did not fire");
        let feet = d3::gameplay::feet_of(&rig.world, HERO).expect("the hero stands somewhere");
        let capsule = feet + glam::DVec3::Y * d3::gameplay::MUZZLE_HEIGHT_M;
        let off = (r.hits[0].from - capsule).length();
        println!("socket `{socket}`: the shot left {off:.4} m from the capsule rule");
        assert!(
            inf_ecs::pose::evaluated_pose(&rig.world, HERO).is_some(),
            "{socket}: this arm is about a RIGGED hero and this one has no pose"
        );
        if from_weapon {
            assert!(
                off > 0.2,
                "the rigged hero still shot from the capsule rule ({off} m away)"
            );
            assert_eq!(
                r.muzzles_without_a_socket, 0,
                "a rig that publishes `{socket}` was counted as missing it"
            );
        } else {
            assert!(
                off < 1e-12,
                "a rig with no weapon socket did not fall back ({off} m away)"
            );
            assert_eq!(
                r.muzzles_without_a_socket, 1,
                "a rigged hero fell back to 1.4 m and nothing counted it"
            );
        }
    }
}

/// **AN ARMED CHARACTER GRABS WITH ITS FREE HAND — and that hand is the one
/// whose fingers close** (SK1c audit, H1).
///
/// # The sentence this arm exists to make true
///
/// SK1c's hand pass writes down one precedence rule — *the weapon owns the hand
/// it is in and a grab takes the other one*, so "a character reaching for a door
/// handle with a rifle in its right hand reaches with its left". Neither half of
/// that happened, and no arm in the tree ran the two together: `weapon_hands_gate`
/// unequips ten steps before it presses E, so its grab is always an unarmed one.
///
/// Two independent defects, both measured on this fixture before the fix:
///
/// * `apply_hand_ik` resolved a grip to the hand the **affordance** names, and
///   `grip_catalogue` authors `handle` on the RIGHT hand — so the left slot's
///   `handle` closed the right hand, which the `rifle` grip in slot 1 then
///   overwrote. The reaching hand's fingers never moved (fingertip-to-wrist
///   **0.1839 m**, a fully open hand) and the off hand sprang open out of its
///   fore-grip (**0.0957 → 0.1839 m**);
/// * the `GunGrip` off-hand solve runs *after* the reaches inside
///   `apply_hand_ik`, and ran unconditionally — so it overwrote the grab's reach
///   every step and the left wrist stayed exactly where the fore-grip put it,
///   to the bit.
///
/// So the only observable effect of pressing E while armed was the support hand
/// letting go of the weapon, while `GameplayReport::hands.1` counted a grab.
///
/// # What is asserted
///
/// The armed grab is measured against the **unarmed** one, because "the left
/// hand closed" says nothing on its own: the claim is that the free hand does
/// what the unarmed hand does, and that the holding hand is undisturbed.
#[test]
fn an_armed_character_grabs_with_its_free_hand() {
    // -- the control: unarmed, so the grab is in the RIGHT hand --
    let mut bare = Rig::new();
    bare.rig_the_hero(d3::gameplay::WEAPON_SOCKET);
    bare.handle();
    bare.steps(&idle(), 2);
    let r = bare.step(&press_interact());
    assert_eq!(r.hands, (0, 1), "an unarmed press asked for no grab");
    bare.steps(&idle(), 8);
    let unarmed = bare.spans();
    assert!(
        unarmed[1] < unarmed[0] * 0.9,
        "the unarmed hero's right hand did not close on the handle: {unarmed:?}"
    );

    // -- armed: the rifle stays in the right hand and the LEFT one grabs --
    let mut rig = Rig::new();
    rig.rig_the_hero(d3::gameplay::WEAPON_SOCKET);
    rig.handle();
    rig.arm("rifle");
    rig.steps(&idle(), 2);
    let held = rig.spans();
    let held_wrist = rig.wrists();
    assert!(
        held[0] < 0.9 * unarmed[0] && held[1] < 0.9 * unarmed[1],
        "an armed hero should hold the weapon with BOTH hands: {held:?}"
    );

    let r = rig.step(&press_interact());
    assert_eq!(
        r.hands,
        (1, 1),
        "an armed press should be counted as a hold AND a grab"
    );
    rig.steps(&idle(), 8);
    let grabbing = rig.spans();
    let grabbing_wrist = rig.wrists();
    println!(
        "ARMED GRAB: spans held {held:?} -> grabbing {grabbing:?} (unarmed {unarmed:?}); \
         left wrist {held_wrist:?} -> {grabbing_wrist:?}, handle at {HANDLE_AT:?}"
    );

    // 1. THE LEFT ARM MOVED, and it moved TOWARDS the handle. Measured against
    //    where the fore-grip had it, so a pass that simply did nothing fails.
    let was = (held_wrist[0] - HANDLE_AT).length();
    let now = (grabbing_wrist[0] - HANDLE_AT).length();
    assert!(
        now < was - 0.1,
        "the left wrist did not reach for the handle: {was:.4} m away, then \
         {now:.4} m — the gun solve is overwriting the grab's reach"
    );

    // 2. THE LEFT HAND CLOSED ON THE HANDLE, to the same aperture the unarmed
    //    hero's hand closes to. A right-handed affordance in the left slot used
    //    to leave this hand fully open.
    assert!(
        (grabbing[0] - unarmed[1]).abs() < 1.0e-6,
        "the free hand did not take the handle the way an unarmed one does: \
         {:.4} against {:.4}",
        grabbing[0],
        unarmed[1]
    );

    // 3. THE RIGHT HAND IS UNDISTURBED — still on the rifle, to the bit. The
    //    grab must not reach into the hand the weapon owns.
    assert!(
        (grabbing[1] - held[1]).abs() < 1.0e-12,
        "the grab moved the hand holding the weapon: {:.6} against {:.6}",
        grabbing[1],
        held[1]
    );
    assert!(
        (grabbing_wrist[1] - held_wrist[1]).length() < 1.0e-12,
        "the grab moved the weapon hand's wrist"
    );

    // 4. …and it LETS GO: once the grab has aged out, the off hand is back on
    //    the fore-grip, byte-identical to where it was before the press. The
    //    claim `apply_grip`'s "a curl is a pose, not a delta" rests on, met at
    //    the composition level rather than at the solver's.
    rig.steps(&idle(), 70);
    assert!(
        inf_ecs::interact::hand_grab(&rig.world, HERO).is_none(),
        "the grab never aged out"
    );
    let after = rig.spans();
    assert!(
        (after[0] - held[0]).abs() < 1.0e-12 && (after[1] - held[1]).abs() < 1.0e-12,
        "the hands did not go back on the weapon after the grab: {after:?} \
         against {held:?}"
    );
}

/// **A ROUND INTO A PERSON HURTS THE PERSON** — the silent shot, closed (wave
/// WPN1).
///
/// Until this wave `on_flesh` asked whether the target carried a `Health`
/// component, and I6 gave one to the hero and to nothing else. So a round into
/// any other character answered `false`, went to the destructible branch, and
/// came back in `GameplayReport::destruct` — where the host logs a
/// `NoDestructible` refusal, once per round, ten times a second on a held
/// trigger. The person was unhurt, the log was full, and the flood was the only
/// symptom.
///
/// Three claims, and the second is the one a "lazy health" that never fires
/// would fail:
///
/// 1. the body has **no** `Health` before the shot — or this arm is
///    `a_round_reaches_the_body_it_is_aimed_at` wearing a different name;
/// 2. it has one **after**, with the round's joules already out of it;
/// 3. `destruct` is **empty**, so nothing was owed to a door that does not
///    exist.
#[test]
fn a_round_into_a_character_with_no_health_gives_it_one_and_owes_nothing() {
    let mut rig = Rig::new();
    spawn_bare_body(&mut rig.world, 8.0);
    rig.world.mark_dirty();
    rig.world.propagate();
    rig.arm("rifle");
    rig.step(&idle());
    assert!(
        weapon::health_of(&rig.world, TARGET).is_none(),
        "the fixture gave the bystander a body, so this arm proves nothing"
    );
    let r = rig.step(&hold_trigger());
    assert_eq!(r.shots, 1, "the trigger did not fire");
    assert_eq!(r.hits.len(), 1);
    assert_eq!(r.hits[0].target, Some(TARGET));
    assert!(
        r.hits[0].on_flesh,
        "a round into a person was not on flesh, so it went to the P22 door"
    );
    let h = weapon::health_of(&rig.world, TARGET).expect("the round gave it a body");
    println!(
        "one rifle round on a bystander with no authored health: {} J of {} left; \
         destruct owes {:?}; {} stagger(s), {} knockdown(s)",
        h.joules, h.capacity_j, r.destruct, r.staggers, r.knockdowns
    );
    assert!(
        (h.capacity_j - weapon::DEFAULT_VITALITY_J).abs() < 1e-9,
        "the lazy body is not the default vitality: {}",
        h.capacity_j
    );
    assert!(
        (h.joules - (weapon::DEFAULT_VITALITY_J - RIFLE_J)).abs() < 1e-9,
        "the round's joules did not come out of the body it just made: {}",
        h.joules
    );
    assert!(
        r.destruct.is_empty(),
        "a round into a person owed {:?} to the P22 damage door — this is the \
         log flood, and the host answers every one of these with a \
         `NoDestructible` refusal",
        r.destruct
    );

    // …and the DESTRUCTIBLE branch still works, which is what says the guard
    // above discriminates rather than simply refusing everything.
    let mut wall = Rig::new();
    spawn_wall(&mut wall.world, 8.0);
    wall.world.mark_dirty();
    wall.world.propagate();
    wall.arm("rifle");
    wall.step(&idle());
    let r = wall.step(&hold_trigger());
    assert_eq!(
        r.destruct.len(),
        1,
        "a round into a destructible owes it nothing: {:?}",
        r.destruct
    );
    assert!((r.destruct[0].1 - RIFLE_J).abs() < 1e-9);
}

/// **A ROUND INTO A NON-DESTRUCTIBLE PROP OWES NOTHING AT ALL** — the other half
/// of the flood (wave WPN1).
///
/// The ground is the case a level cannot avoid: it is a static collider, it has
/// no `Destructible` and it is what most missed rounds end at. Before this wave
/// every one of those was a `NoDestructible` line in the log.
#[test]
fn a_round_into_the_ground_owes_the_p22_door_nothing() {
    let mut rig = Rig::new();
    rig.arm("rifle");
    rig.step(&idle());
    // Straight down: the ground slab is the only thing under the hero.
    {
        let e = rig.world.entity_of(HERO).expect("the hero");
        let mut cm = rig
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a mover");
        cm.runtime.aim_pitch_deg = -89.0;
    }
    let r = rig.step(&hold_trigger_no_edge());
    println!(
        "{} round(s) into the ground: {} hit(s), {:?} owed",
        r.shots,
        r.hits.len(),
        r.destruct
    );
    assert_eq!(r.shots, 1, "the trigger did not fire");
    assert_eq!(r.hits.len(), 1);
    assert_eq!(
        r.hits[0].target,
        Some(GROUND),
        "the shot did not reach the ground: {:?}",
        r.hits[0].target
    );
    assert!(!r.hits[0].on_flesh, "the ground is not flesh");
    assert!(
        r.destruct.is_empty(),
        "a round into the ground owed {:?} to a door the ground does not have",
        r.destruct
    );
}

/// **A HIT SHOWS, AND A HEAVY ONE PUTS A BODY ON THE FLOOR** (wave WPN1).
///
/// The two halves are asserted apart, because a `stagger` that always knocked
/// down and a `knockdown` that never fired both satisfy a single counter:
///
/// * a **rifle round** — 1 700 J against a 2 000 J body, 0.85 of what it had —
///   arms the reaction AND takes the mode;
/// * a **pistol round** — 500 J against a 5 000 J body, a tenth — arms the
///   reaction and leaves the body standing.
#[test]
fn a_non_fatal_hit_arms_a_reaction_and_a_heavy_one_takes_the_mode() {
    // The heavy blow.
    let mut rig = Rig::new();
    spawn_target(&mut rig.world, 8.0, weapon::DEFAULT_VITALITY_J);
    rig.world.mark_dirty();
    rig.world.propagate();
    rig.arm("rifle");
    rig.step(&idle());
    let mode_before = rig.target_mode();
    let r = rig.step(&hold_trigger());
    println!(
        "a {RIFLE_J} J round on a {} J body: {} stagger(s), {} knockdown(s), mode {:?} -> {:?}",
        weapon::DEFAULT_VITALITY_J,
        r.staggers,
        r.knockdowns,
        mode_before,
        rig.target_mode()
    );
    assert_eq!(
        r.staggers, 1,
        "a round that hurt somebody armed no reaction"
    );
    assert_eq!(
        r.knockdowns, 1,
        "a round worth 85 % of a body left it standing"
    );
    assert_eq!(
        rig.target_mode(),
        inf_ecs::components::MovementMode::FallControlled,
        "the mode table did not let go"
    );
    // The reaction really reached the animation seam, and it is a ONE-SHOT: the
    // trigger is consumed by whoever asks first, so a second read finds nothing.
    assert!(
        inf_ecs::anim_bridge::consume_anim_notify(&mut rig.world, TARGET, weapon::STAGGER_TRIGGER)
            || inf_ecs::anim_bridge::bridge(&rig.world).is_none(),
        "the hit reaction never reached the animation bridge"
    );

    // The light blow, on a body with plenty left.
    let mut soft = Rig::new();
    spawn_target(&mut soft.world, 8.0, 5000.0);
    soft.world.mark_dirty();
    soft.world.propagate();
    soft.arm("pistol");
    soft.step(&idle());
    let r = soft.step(&hold_trigger());
    println!(
        "a pistol round on a 5000 J body: {} stagger(s), {} knockdown(s), mode {:?}",
        r.staggers,
        r.knockdowns,
        soft.target_mode()
    );
    assert_eq!(r.staggers, 1, "a pistol round armed no reaction");
    assert_eq!(
        r.knockdowns, 0,
        "a pistol round worth an eighth of a body knocked it down — the \
         threshold is not discriminating and every hit is a knockdown"
    );
    assert_ne!(
        soft.target_mode(),
        inf_ecs::components::MovementMode::FallControlled
    );
}

/// **AN EMPTY HAND IS THE THIRD CONSUMER OF THE ATTACK EDGE** (wave WPN1).
///
/// The claims, and each one kills a different mutation:
///
/// 1. an unarmed character that has **never** pressed the button carries no
///    `WeaponState` at all — the lazy install, which is what keeps every trace
///    committed before this wave byte-identical;
/// 2. pressing it throws a punch that reaches a body **1 m** away and takes
///    `FIST_DAMAGE_J` out of it, through the same `Health` door a bullet uses;
/// 3. the punch is a **swing** and not a shot: `swings` counts it, and the arc
///    resolution is what found the target rather than a cast;
/// 4. a body **behind** the swinger is not hit, so the cone is doing work.
#[test]
fn an_unarmed_character_punches_the_body_in_front_of_it() {
    let mut rig = Rig::new();
    spawn_bare_body(&mut rig.world, 1.0);
    rig.world.mark_dirty();
    rig.world.propagate();
    // NO `arm()`: the hero has nothing in its hands and nothing in its bag.
    rig.step(&idle());
    let e = rig.world.entity_of(HERO).expect("the hero");
    assert!(
        rig.world.world().get::<WeaponState>(e).is_none(),
        "an unarmed character that has never punched carries an ammunition clock"
    );

    let r = rig.step(&hold_trigger());
    println!(
        "one punch: {} shot(s), {} swing(s), {} hit(s), {} stagger(s); the body has {:?} J",
        r.shots,
        r.swings,
        r.hits.len(),
        r.staggers,
        weapon::health_of(&rig.world, TARGET).map(|h| h.joules)
    );
    assert_eq!(
        r.shots, 1,
        "the attack button did nothing with an empty hand"
    );
    assert_eq!(
        r.swings, 1,
        "the punch resolved as a SHOT rather than a swing"
    );
    assert_eq!(r.hits.len(), 1);
    assert_eq!(r.hits[0].target, Some(TARGET), "the punch found nobody");
    assert!(r.hits[0].on_flesh);
    // **The world**: the body it landed on, read back out of the ECS.
    let h = weapon::health_of(&rig.world, TARGET).expect("the punch gave it a body");
    assert!(
        (h.joules - (weapon::DEFAULT_VITALITY_J - weapon::FIST_DAMAGE_J)).abs() < 1e-9,
        "the punch took {} J and a fist carries {}",
        weapon::DEFAULT_VITALITY_J - h.joules,
        weapon::FIST_DAMAGE_J
    );
    assert_eq!(r.staggers, 1, "the punch armed no hit reaction");
    assert_eq!(
        r.knockdowns,
        0,
        "a punch worth {} J of a {} J body knocked it down",
        weapon::FIST_DAMAGE_J,
        weapon::DEFAULT_VITALITY_J
    );
    // The clock is installed and it is the FIST's, so equipping a rifle later
    // gets a fresh magazine rather than this.
    let s = rig
        .world
        .world()
        .get::<WeaponState>(e)
        .expect("the punch installed a clock")
        .clone();
    assert_eq!(s.item_id, weapon::FIST_ITEM);

    // …and it is SEMI-AUTOMATIC: the button held throws one punch, not sixty.
    let before = weapon::health_of(&rig.world, TARGET)
        .expect("a body")
        .joules;
    let rs = rig.steps(&hold_trigger_no_edge(), 30);
    let more: u32 = rs.iter().map(|r| r.swings).sum();
    println!(
        "holding the button for another 30 steps threw {more} more punch(es); \
         the body went {before} -> {} J",
        weapon::health_of(&rig.world, TARGET)
            .expect("a body")
            .joules
    );
    assert_eq!(
        more, 0,
        "a held button threw {more} punches — the fist is automatic"
    );

    // **THE CONE**: the same press with the body BEHIND the hero lands nothing.
    let mut behind = Rig::new();
    spawn_bare_body(&mut behind.world, -1.0);
    behind.world.mark_dirty();
    behind.world.propagate();
    behind.step(&idle());
    let r = behind.step(&hold_trigger());
    println!(
        "a punch at a body 1 m BEHIND the hero: {} swing(s), target {:?}",
        r.swings, r.hits[0].target
    );
    assert_eq!(r.swings, 1, "the swing did not happen at all");
    assert_eq!(
        r.hits[0].target, None,
        "the swing reached a body behind the swinger — the arc is not being applied"
    );
    assert!(
        weapon::health_of(&behind.world, TARGET).is_none(),
        "a body behind the swinger was hurt"
    );
}

/// **A MELEE `WeaponDef` IS A REACH AND AN ARC, NOT A RAY** (wave WPN1) — a bat
/// out of the catalogue, through the same equip door a rifle takes.
///
/// The arm a fist alone cannot make: the fist's numbers are engine constants, so
/// a `WeaponDef` whose kind is melee but which never reached the arc resolution
/// would still let the punch through. This one is authored **by name**, through
/// `WeaponDef::set` — the live-tuning door a designer's slider and the item
/// TOML's own reader both go through. (The TOML spelling `kind = "melee"` is
/// pinned in `inf_ecs::weapon`'s own tests; `inf-physics` has no `toml`
/// dependency and this arm is about the resolution, not about the parser.)
#[test]
fn an_authored_melee_weapon_swings_instead_of_casting() {
    let mut rig = Rig::new();
    spawn_bare_body(&mut rig.world, 1.6);
    rig.world.mark_dirty();
    rig.world.propagate();
    let mut def = WeaponDef::default();
    for (name, value) in [
        ("melee", 1.0),
        ("damage_j", 900.0),
        ("range_m", 2.0),
        ("melee_arc_deg", 120.0),
        ("rounds_per_minute", 60.0),
    ] {
        assert!(def.set(name, value), "the tuning door does not know {name}");
    }
    assert!(def.is_melee(), "`melee` did not reach the def");
    assert!((def.reach_m() - 2.0).abs() < 1e-12);
    {
        let defs = item::item_defs_mut(&mut rig.world);
        assert!(defs.insert(ItemDef {
            id: "bat".into(),
            label: "Bat".into(),
            stack_max: 1,
            mass_kg: 1.1,
            weapon: Some(def),
        }));
    }
    rig.arm("bat");
    rig.step(&idle());
    let r = rig.step(&hold_trigger());
    println!(
        "a 2 m bat at a body 1.6 m away: {} swing(s), target {:?}, {} J left",
        r.swings,
        r.hits[0].target,
        weapon::health_of(&rig.world, TARGET)
            .map(|h| h.joules)
            .unwrap_or(-1.0)
    );
    assert_eq!(r.swings, 1);
    assert_eq!(r.hits[0].target, Some(TARGET));
    let h = weapon::health_of(&rig.world, TARGET).expect("a body");
    assert!((weapon::DEFAULT_VITALITY_J - h.joules - 900.0).abs() < 1e-9);
    // …and the reach is the CLAIM: the same bat cannot touch a body at 3 m,
    // where a hitscan of the same `range_m` would have reached easily.
    let mut far = Rig::new();
    spawn_bare_body(&mut far.world, 3.0);
    far.world.mark_dirty();
    far.world.propagate();
    {
        let defs = item::item_defs_mut(&mut far.world);
        assert!(defs.insert(ItemDef {
            id: "bat".into(),
            label: "Bat".into(),
            stack_max: 1,
            mass_kg: 1.1,
            weapon: Some(def),
        }));
    }
    far.arm("bat");
    far.step(&idle());
    let r = far.step(&hold_trigger());
    assert_eq!(r.swings, 1, "the swing did not happen");
    assert_eq!(
        r.hits[0].target, None,
        "a 2 m bat reached a body 3 m away — the reach is not being applied"
    );
}
