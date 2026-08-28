//! **I6: doors, kicks and crash-throughs, against a real world.**
//!
//! `inf_ecs::door`'s own tests pin the *rules* — the swing, the lock's price,
//! the momentum arithmetic — as functions of numbers. These pin what happens
//! when those rules meet a world: whether the leaf really becomes a collider,
//! whether a body really cannot walk through a shut door, whether a kick really
//! opens one, and whether a sprint really carries its momentum out the other
//! side.
//!
//! **Every arm here asserts the WORLD.** A door's state and the character's
//! position, never a function's own report — the P21 law, met here because a
//! door system that reported "opened" and left the leaf across the doorway would
//! pass the easy version of all of these.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    MovementMode, RigidBody3D, Transform,
};
use inf_ecs::door::{self, Door, DoorSide, DoorSpec, DoorState};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::movement::MovementIntent;
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);

const HERO: Uuid = Uuid::from_u128(0x1600_0001);
const GROUND: Uuid = Uuid::from_u128(0x1600_0002);
const DOOR: Uuid = Uuid::from_u128(0x1600_0003);
const WEDGE: Uuid = Uuid::from_u128(0x1600_0004);

const RADIUS: f64 = 0.3;

/// The doorway is at `z = 4`, hinged on its left post at `x = -0.45`, with the
/// shut leaf lying along `+x` and the **inside** face pointing `+z`.
///
/// So a character walking `+z` from the origin is **outside**, and the leaf
/// swings **inward** (a negative limit) — away from anyone standing outside it,
/// which is what a door being kicked in does and what the wedge arm relies on.
/// The first draft of this fixture swung the leaf outward and the door
/// correctly refused to open because the character kicking it was standing in
/// its arc; the system was right and the fixture was wrong.
const HINGE: DVec3 = DVec3::new(-0.45, 1.05, 4.0);

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
            half_extents: Vec3d::new(60.0, 0.5, 60.0),
            ..Default::default()
        },
        t,
    ));
}

/// A door hinged at [`HINGE`], swinging `+95` degrees, inside face toward `+x`.
///
/// The entity's transform IS the hinge and is never written by the door step —
/// see `inf_physics::d3::door`'s header for why the leaf is a synthetic body.
fn spawn_door(w: &mut EcsWorld, locked: bool) {
    let e = w.spawn_with_guid(DOOR, "Front Door", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::from_dvec3(HINGE);
    w.world_mut().entity_mut(e).insert((
        Door {
            spec: DoorSpec {
                // The leaf runs from the hinge toward +x when shut, and swings
                // toward +z (inside) — a NEGATIVE limit.
                closed_yaw_deg: 90.0,
                open_limit_deg: -inf_ecs::door::DEFAULT_OPEN_LIMIT_DEG,
                inside_yaw_deg: 0.0,
                lock_side: DoorSide::Inside,
                locked_at_spawn: locked,
                ..Default::default()
            },
            label: "front door".into(),
        },
        t,
    ));
    w.mark_dirty();
    w.propagate();
}

fn spawn_hero(w: &mut EcsWorld, z: f64) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let e = w.spawn_with_guid(HERO, "Hero", None);
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
    w.mark_dirty();
    w.propagate();
}

struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Rig {
    fn new(locked: bool, hero_z: f64) -> Self {
        let mut world = EcsWorld::new();
        spawn_ground(&mut world);
        spawn_door(&mut world, locked);
        spawn_hero(&mut world, hero_z);
        world.mark_dirty();
        world.propagate();
        let mut rig = Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
        };
        rig.bridge.sync_from_world(&rig.world);
        rig
    }

    /// The same rig with **no door at all** - the control every momentum claim
    /// is measured against, so one step of the movement step's own friction
    /// cancels out of both sides.
    fn doorless(hero_z: f64) -> Self {
        let mut world = EcsWorld::new();
        spawn_ground(&mut world);
        spawn_hero(&mut world, hero_z);
        world.mark_dirty();
        world.propagate();
        let mut rig = Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
        };
        rig.bridge.sync_from_world(&rig.world);
        rig
    }

    /// One fixed step, in the order **both hosts run it**: sync, movement,
    /// gameplay, solver, write-back, propagate. A test that stepped in a
    /// different order would be testing a sequence nobody ships.
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

    fn steps(&mut self, intent: &MovementIntent, n: u32) -> d3::GameplayReport {
        let mut last = d3::GameplayReport::default();
        for _ in 0..n {
            last = self.step(intent);
        }
        last
    }

    fn hero_pos(&self) -> DVec3 {
        let e = self.world.entity_of(HERO).expect("the hero");
        self.world
            .world()
            .get::<Transform>(e)
            .expect("a transform")
            .translation
            .to_dvec3()
    }

    fn hero_feet(&self) -> DVec3 {
        let e = self.world.entity_of(HERO).expect("the hero");
        let cm = self
            .world
            .world()
            .get::<CharacterMovement>(e)
            .expect("a movement component");
        self.hero_pos() - DVec3::Y * (cm.half_height_for(cm.mode) + RADIUS)
    }

    fn hero_mode(&self) -> MovementMode {
        let e = self.world.entity_of(HERO).expect("the hero");
        self.world
            .world()
            .get::<CharacterMovement>(e)
            .expect("a movement component")
            .mode
    }

    fn hero_speed(&self) -> f64 {
        let e = self.world.entity_of(HERO).expect("the hero");
        let v = self
            .world
            .world()
            .get::<CharacterMovement>(e)
            .expect("a movement component")
            .runtime
            .velocity
            .to_dvec3();
        DVec3::new(v.x, 0.0, v.z).length()
    }

    fn set_hero_velocity(&mut self, v: DVec3) {
        let e = self.world.entity_of(HERO).expect("the hero");
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a movement component");
        cm.runtime.velocity = Vec3d::from_dvec3(v);
        cm.runtime.seeded = true;
    }

    /// Point the hero (and its aim) at a compass yaw.
    fn face(&mut self, yaw_deg: f64) {
        let e = self.world.entity_of(HERO).expect("the hero");
        {
            let mut t = self
                .world
                .world_mut()
                .get_mut::<Transform>(e)
                .expect("a transform");
            t.rotation.y = yaw_deg;
        }
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a movement component");
        cm.runtime.aim_yaw_deg = yaw_deg;
        cm.runtime.body_yaw_deg = yaw_deg;
        cm.runtime.target_yaw_deg = yaw_deg;
        cm.runtime.seeded = true;
    }

    fn set_hero_mode(&mut self, mode: MovementMode) {
        let e = self.world.entity_of(HERO).expect("the hero");
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a movement component");
        cm.mode = mode;
    }

    fn door_state(&self) -> DoorState {
        let spec = self.door_spec();
        door::door_field(&self.world)
            .map(|f| f.get(DOOR, &spec))
            .unwrap_or_else(|| DoorState::fresh(&spec))
    }

    fn door_spec(&self) -> DoorSpec {
        let e = self.world.entity_of(DOOR).expect("the door");
        self.world
            .world()
            .get::<Door>(e)
            .expect("a door component")
            .spec
    }

    /// Where the leaf's collider actually is, read out of the **physics world**
    /// rather than out of the door's own state.
    fn leaf_body_centre(&mut self) -> Option<DVec3> {
        let leaf = d3::door_leaf_guid(DOOR);
        let body = self.bridge.body_of(leaf)?;
        self.bridge.world_mut().body_translation(body)
    }

    fn prompt(&self) -> Option<String> {
        let feet = self.hero_feet();
        let e = self.world.entity_of(HERO).expect("the hero");
        let yaw = self
            .world
            .world()
            .get::<CharacterMovement>(e)
            .expect("a movement component")
            .runtime
            .aim_yaw_deg;
        let exclude = std::collections::BTreeSet::from([HERO]);
        d3::interact::resolve(&self.world, &self.bridge, feet, yaw, &exclude).map(|h| h.label)
    }
}

fn idle() -> MovementIntent {
    MovementIntent::default()
}

fn press_e() -> MovementIntent {
    MovementIntent {
        interact: true,
        ..Default::default()
    }
}

/// **The lock control** (island wave I8b) — its own key, because E always opens.
fn press_lock() -> MovementIntent {
    MovementIntent {
        lock: true,
        ..Default::default()
    }
}

fn press_attack() -> MovementIntent {
    MovementIntent {
        attack: true,
        attack_pressed: true,
        ..Default::default()
    }
}

fn walk() -> MovementIntent {
    MovementIntent {
        move_input: Vec2d::new(0.0, 1.0),
        ..Default::default()
    }
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **THE HEADLINE: the leaf is a real collider, and it moves.**
///
/// A door that opened only in its own state would satisfy every rule test in
/// `inf_ecs::door` and would let a character walk through a shut one. So this
/// reads the leaf's body out of the **physics world**, before and after.
#[test]
fn a_doors_leaf_is_a_body_in_the_world_and_the_swing_moves_it() {
    let mut rig = Rig::new(false, 2.4);
    rig.step(&idle());
    let shut = rig.leaf_body_centre().expect("the leaf has a body");
    // Shut, the leaf's centre is half a width along +x from the hinge.
    let want = HINGE + DVec3::X * (DoorSpec::default().width_m * 0.5);
    println!("the shut leaf's body is at {shut:?}; the geometry says {want:?}");
    assert!((shut - want).length() < 1e-5, "{shut:?}");
    // Open it and step until it settles.
    rig.step(&press_e());
    assert!(
        rig.door_state().powered || rig.door_state().open_deg != 0.0,
        "the E press did not power the door"
    );
    rig.steps(&idle(), 60);
    let open = rig.leaf_body_centre().expect("the leaf still has a body");
    println!("the open leaf's body is at {open:?}");
    assert!(
        (open - shut).length() > 0.5,
        "the leaf's COLLIDER did not move: {shut:?} to {open:?}"
    );
    // …and it is where the door's own state says it is, through the one pose
    // function. Two answers to "where is this leaf" is the defect this checks.
    let p = inf_ecs::door::DoorPlacement {
        guid: DOOR,
        hinge: HINGE,
        spec: rig.door_spec(),
        label: "front door".into(),
    };
    let (centre, _, _) = inf_ecs::door::leaf_pose(&p, rig.door_state().open_deg);
    assert!(
        (open - centre).length() < 1e-9,
        "the collider and the state disagree: {open:?} vs {centre:?}"
    );
    assert!(rig.door_state().is_open(&rig.door_spec()));
}

/// **A shut door stops a body; the same door open does not.**
///
/// The claim the whole system exists for, measured as a position in the world.
#[test]
fn a_body_cannot_walk_through_a_shut_door_and_can_walk_through_an_open_one() {
    // Shut: walk into it for two seconds and stop short.
    let mut shut = Rig::new(false, 2.4);
    shut.steps(&walk(), 120);
    let blocked_z = shut.hero_pos().z;
    println!("walking into a shut door stopped at z = {blocked_z}");
    assert!(
        blocked_z < 4.0,
        "the character walked through a shut door to z = {blocked_z}"
    );
    // Open: the same walk goes past it.
    let mut open = Rig::new(false, 2.4);
    open.step(&press_e());
    open.steps(&idle(), 60);
    assert!(open.door_state().is_open(&open.door_spec()));
    open.steps(&walk(), 120);
    let through_z = open.hero_pos().z;
    println!("walking through the open door reached z = {through_z}");
    assert!(
        through_z > 4.3,
        "the character did not get through an OPEN door: z = {through_z}"
    );
    assert!(
        through_z - blocked_z > 0.5,
        "opening the door made no difference: {blocked_z} vs {through_z}"
    );
}

/// **A locked door says "Locked" and does not open** — and the prompt and the
/// press agree, because they are one call.
#[test]
fn a_locked_door_says_so_from_outside_and_refuses_the_key() {
    let mut rig = Rig::new(true, 3.0);
    rig.step(&idle());
    let prompt = rig.prompt().expect("a door prompt");
    println!("standing outside a locked door, the prompt reads: {prompt}");
    assert!(prompt.contains("Locked"), "{prompt}");
    // The press does what the prompt says: nothing.
    rig.step(&press_e());
    rig.steps(&idle(), 60);
    assert!(rig.door_state().locked, "the lock gave to a key press");
    assert_eq!(rig.door_state().open_deg, 0.0, "a locked door swung");
    assert!(!rig.door_state().is_open(&rig.door_spec()));
    // …and the body still cannot get past it.
    rig.steps(&walk(), 120);
    assert!(rig.hero_pos().z < 4.0, "{}", rig.hero_pos().z);
}

/// **The lock verb is on ONE face**, and the face is where the character is.
#[test]
fn the_lock_verb_appears_on_the_inside_and_not_on_the_outside() {
    // Inside is +z of the hinge (`inside_yaw_deg` is 0, so the inside normal is
    // +Z). Put the hero there, facing back at the door.
    let mut inside = Rig::new(false, 5.0);
    inside.face(180.0);
    inside.step(&idle());
    let feet = inside.hero_feet();
    assert_eq!(
        inside.door_spec().side_of(HINGE, feet),
        DoorSide::Inside,
        "the fixture does not put the hero inside"
    );
    let prompt = inside.prompt().expect("a door prompt");
    println!("standing inside, the prompt reads: {prompt}");
    assert!(prompt.contains("lock"), "{prompt}");
    // **BOTH VERBS, from the same place** (island wave I8b). The prompt names
    // the pair, so both halves are asserted here: the LOCK control throws the
    // bolt without moving the leaf...
    inside.step(&press_lock());
    assert!(
        inside.door_state().locked,
        "the lock control did not lock it"
    );
    assert_eq!(inside.door_state().open_deg, 0.0);
    // ...pressing it again draws the bolt rather than double-locking...
    inside.step(&press_lock());
    assert!(
        !inside.door_state().locked,
        "the second lock press did not unlock"
    );
    // ...and E, from this same face, OPENS. That is the whole of the I8a audit's
    // MED-5: before this the shut leaf took the lock verb whatever its bolt was
    // doing, so a character who closed the door behind them could never open it
    // again.
    inside.step(&press_e());
    assert!(
        inside.door_state().open_deg != 0.0,
        "E from the lock side did not open the door: {:?}",
        inside.door_state()
    );
    assert!(
        !inside.door_state().locked,
        "opening threw the bolt as a side effect"
    );

    // And from the outside the same door offers no lock verb at all.
    let mut outside = Rig::new(false, 2.4);
    outside.step(&idle());
    let prompt = outside.prompt().expect("a door prompt");
    println!("standing outside, the prompt reads: {prompt}");
    assert!(!prompt.contains("lock"), "{prompt}");
    // ...and the same press opens it, which is the control that says the two
    // faces really do different things.
    outside.step(&press_e());
    outside.steps(&idle(), 60);
    assert!(outside.door_state().is_open(&outside.door_spec()));
    assert!(!outside.door_state().locked);
}

/// **THE KICK.** The attack button on a locked door in reach breaks the lock,
/// and the energies are the P22 rule's own.
///
/// The kick lands on its **fuse** here, because a test rig has no rig to notify
/// it — which is the path this fixture exists to arm. The notify path is armed
/// by the arm below it.
#[test]
fn a_kick_breaks_a_locked_door_and_the_energies_are_the_p22_rules() {
    let mut rig = Rig::new(true, 3.0);
    rig.step(&idle());
    assert!(rig.door_state().locked);
    // Close enough to kick: the doorway's centre is at z = 4, the hero at 3.
    let feet = rig.hero_feet();
    let reach = (inf_ecs::door::prompt_position(&inf_ecs::door::DoorPlacement {
        guid: DOOR,
        hinge: HINGE,
        spec: rig.door_spec(),
        label: String::new(),
    }) - feet)
        .length();
    println!(
        "the hero's feet are {reach} m from the doorway; the kick reaches {} m",
        inf_ecs::door::KICK_REACH_M
    );
    assert!(
        reach <= inf_ecs::door::KICK_REACH_M,
        "the fixture is out of reach"
    );
    // The press arms the kick and does NOT break the lock — the impulse is on
    // the notify (or the fuse), never on the button.
    let armed = rig.step(&press_attack());
    assert_eq!(armed.kicks, 0, "the kick landed on the button press");
    assert!(
        rig.door_state().locked,
        "the lock broke before the leg moved"
    );
    assert!(
        inf_ecs::door::pending_kick(&rig.world, HERO).is_some(),
        "no kick was armed"
    );
    // …and it lands within the fuse.
    let mut landed = 0;
    let mut broke = 0;
    for _ in 0..60 {
        let r = rig.step(&idle());
        landed += r.kicks;
        broke += r.locks_broken;
    }
    println!(
        "a {} J kick against a {} J lock: {landed} kick(s), {broke} lock(s) broken",
        inf_ecs::door::kick_energy_j(),
        rig.door_spec().lock_energy_j()
    );
    assert_eq!(landed, 1, "the kick landed {landed} times");
    assert_eq!(broke, 1);
    let state = rig.door_state();
    assert!(state.lock_broken && !state.locked, "{state:?}");
    // **Assert the WORLD**: the leaf swung open and the body can now get past.
    rig.steps(&idle(), 90);
    assert!(
        rig.door_state().is_open(&rig.door_spec()),
        "{:?}",
        rig.door_state()
    );
    rig.steps(&walk(), 150);
    println!(
        "after kicking it in, the hero reached z = {}",
        rig.hero_pos().z
    );
    assert!(rig.hero_pos().z > 4.3, "{}", rig.hero_pos().z);
}

/// **The kick lands on its NOTIFY when there is one**, and the fuse does not
/// double it.
#[test]
fn a_kick_lands_on_the_animation_notify_when_one_arrives() {
    let mut rig = Rig::new(true, 3.0);
    rig.step(&press_attack());
    assert!(inf_ecs::door::pending_kick(&rig.world, HERO).is_some());
    // The notify arrives on the very next step — well inside the fuse.
    let before = inf_ecs::door::pending_kick(&rig.world, HERO)
        .expect("a kick")
        .fuse_s;
    assert!(before > 0.0);
    inf_ecs::anim_bridge::set_anim_trigger(&mut rig.world, HERO, "unused");
    {
        // Publish the notify the way the pose step does.
        let mut map = std::collections::BTreeMap::new();
        map.insert(HERO, vec![inf_ecs::weapon::KICK_NOTIFY.to_string()]);
        rig.world
            .world_mut()
            .insert_resource(inf_ecs::pose::AnimEventsRes(map));
    }
    let r = rig.step(&idle());
    println!("the notify landed the kick on step 2, {before} s inside the fuse");
    assert_eq!(r.kicks, 1, "the notify did not land the kick");
    assert_eq!(r.locks_broken, 1);
    assert!(inf_ecs::door::pending_kick(&rig.world, HERO).is_none());
    // …and the fuse does not fire a second one.
    let mut later = 0;
    for _ in 0..60 {
        later += rig.step(&idle()).kicks;
    }
    assert_eq!(
        later, 0,
        "the fuse fired a kick the notify had already landed"
    );
}

/// **THE CRASH-THROUGH.** A sprint into a shut door goes through it and keeps
/// most of its speed; the same door at a jog does not.
///
/// **Three worlds, one step, three speeds**, and the claim is their ORDER:
/// a locked door costs more than an unlocked one, an unlocked one costs more
/// than no door at all, and both gaps are printed. Three is what it takes,
/// because the movement step integrates friction after the breach and friction
/// is not additive - a single run's number is the breach's cost tangled with a
/// deceleration that depends on the speed the breach left. The exact arithmetic
/// is pinned where it is arithmetic, in `inf_ecs::door`'s own arms.
///
/// The ordering dies under every mutation that matters: a lock that stops
/// costing joules collapses the first gap, and a restitution of 1.0 collapses
/// the second.
#[test]
fn a_sprint_breaches_a_locked_door_and_a_jog_does_not() {
    // Too slow: the run speed is below the breach gate.
    let mut slow = Rig::new(true, 3.4);
    slow.set_hero_velocity(DVec3::new(0.0, 0.0, 3.75));
    let r = slow.step(&idle());
    assert_eq!(r.doors.doors, 1);
    assert!(
        slow.door_state().locked,
        "a 3.75 m/s jog breached a locked door"
    );
    assert!(!slow.door_state().lock_broken);

    let mut locked = Rig::new(true, 3.4);
    locked.set_hero_velocity(DVec3::new(0.0, 0.0, 6.5));
    locked.step(&idle());
    let locked_out = locked.hero_speed();

    let mut unlocked = Rig::new(false, 3.4);
    unlocked.set_hero_velocity(DVec3::new(0.0, 0.0, 6.5));
    unlocked.step(&idle());
    let unlocked_out = unlocked.hero_speed();

    let mut bare = Rig::doorless(3.4);
    bare.set_hero_velocity(DVec3::new(0.0, 0.0, 6.5));
    bare.step(&idle());
    let bare_out = bare.hero_speed();

    println!(
        "a 6.5 m/s sprint one step later: {locked_out} m/s through a {} J lock, {unlocked_out} m/s through a shut door, {bare_out} m/s through nothing",
        locked.door_spec().lock_energy_j()
    );
    println!(
        "the lock costs {} m/s and the leaf itself costs {} m/s",
        unlocked_out - locked_out,
        bare_out - unlocked_out
    );
    assert!(locked.door_state().lock_broken, "the sprint did not breach");
    assert!(!locked.door_state().locked);
    // The control went through too, and the WORLD says so rather than a flag:
    // its leaf is moving. It is deliberately **not** `lock_broken` — nothing was
    // holding a shut door, so nothing about it broke, and a door somebody
    // barged through is still a door they can lock behind them (I6 audit).
    assert!(
        unlocked.door_state().open_deg != 0.0,
        "the control did not go through"
    );
    assert!(
        !unlocked.door_state().lock_broken,
        "barging through a door that was not locked broke its lock"
    );
    assert!(
        locked_out < unlocked_out,
        "the lock cost nothing: {locked_out} vs {unlocked_out}"
    );
    assert!(
        unlocked_out < bare_out,
        "hitting a door cost nothing: {unlocked_out} vs {bare_out}"
    );
    // **Momentum mostly kept**: over 75 % of the entry speed survives the lock
    // AND the step's own friction.
    assert!(locked_out > 0.75 * 6.5, "{locked_out} of 6.5");
    // ...and the WORLD: the leaf swings out of the way.
    locked.steps(&idle(), 90);
    assert!(locked.door_state().is_open(&locked.door_spec()));
}

/// **The mode CONTINUES through a breach** - a slide stays a slide.
///
/// The control is a world with **no door in it at all**, stepped identically.
/// The breach costs the leaf's own restitution and nothing else, so the two
/// speeds differ by exactly that and the MODE does not differ at all - which is
/// the half of the claim the owner's mandate is about.
#[test]
fn a_sliding_body_is_still_sliding_on_the_other_side() {
    let mut rig = Rig::new(false, 3.4);
    rig.set_hero_mode(MovementMode::Slide);
    rig.set_hero_velocity(DVec3::new(0.0, 0.0, 6.5));
    assert_eq!(rig.hero_mode(), MovementMode::Slide);
    rig.step(&idle());

    // The control: the same slide, with nothing to breach.
    let mut bare = Rig::doorless(3.4);
    bare.set_hero_mode(MovementMode::Slide);
    bare.set_hero_velocity(DVec3::new(0.0, 0.0, 6.5));
    bare.step(&idle());

    println!(
        "a slide at 6.5 m/s through a shut door is in mode {:?} at {} m/s; with no door at all, mode {:?} at {} m/s",
        rig.hero_mode(),
        rig.hero_speed(),
        bare.hero_mode(),
        bare.hero_speed()
    );
    assert_eq!(
        rig.hero_mode(),
        MovementMode::Slide,
        "the breach ended the slide"
    );
    assert_eq!(
        bare.hero_mode(),
        MovementMode::Slide,
        "the control is not a slide"
    );
    assert!(
        rig.door_state().open_deg != 0.0,
        "the slide did not go through"
    );
    // It cost the leaf's restitution and no more: a slide keeps most of what it
    // had, and what it lost is the door.
    assert!(
        rig.hero_speed() > 0.8 * bare.hero_speed(),
        "the door took {} of {} m/s",
        bare.hero_speed() - rig.hero_speed(),
        bare.hero_speed()
    );
    assert!(
        rig.hero_speed() < bare.hero_speed(),
        "the door cost nothing at all"
    );
}

/// **A blocked leaf stops rather than pushing**, which is what makes a door an
/// obstacle rather than a piston.
#[test]
fn a_wedge_in_the_swing_stops_the_leaf_where_it_is() {
    let mut rig = Rig::new(false, 2.4);
    // A block in the leaf's arc, on the inside face's side of the doorway.
    let e = rig.world.spawn_with_guid(WEDGE, "Wedge", None);
    let mut t = Transform::IDENTITY;
    // Placed where the leaf's FREE EDGE arrives at about forty degrees, and not
    // where the shut leaf already is: a wedge touching the shut door blocks the
    // very first step and measures "a door that never opened" rather than "a
    // door that was stopped". The first draft sat at z = 4.4 and did exactly
    // that, which is why the arm below asserts the press reached the door before
    // it asserts anything about the wedge.
    t.translation = Vec3d::new(0.3, 1.05, 4.8);
    rig.world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(0.3, 1.05, 0.3),
            ..Default::default()
        },
        t,
    ));
    rig.world.mark_dirty();
    rig.world.propagate();
    // One settling step before the press: the interact verb is only honoured on
    // a grounded step, and a character spawned this frame has not landed yet.
    rig.step(&idle());
    rig.step(&press_e());
    assert!(
        rig.door_state().powered,
        "the E press did not reach the door, so this arm would measure nothing"
    );
    let mut blocked_steps = 0;
    for _ in 0..90 {
        blocked_steps += rig.step(&idle()).doors.blocked;
    }
    let state = rig.door_state();
    println!(
        "a wedge stopped the leaf at {} degrees after {blocked_steps} blocked steps",
        state.open_deg
    );
    assert!(blocked_steps > 0, "nothing was ever reported as blocked");
    assert!(state.open_deg != 0.0, "the leaf never started moving");
    assert!(state.is_at_rest(), "a blocked leaf is still under power");
    // The control: the same door with no wedge reaches its stop. Without it the
    // arm would pass on a door that never opened at all.
    let mut clear = Rig::new(false, 2.4);
    clear.step(&idle());
    clear.step(&press_e());
    clear.steps(&idle(), 90);
    let limit = clear.door_spec().open_limit_deg;
    println!(
        "with nothing in the way the same leaf reached {} degrees",
        clear.door_state().open_deg
    );
    assert!(
        (clear.door_state().open_deg - limit).abs() < 1e-9,
        "the control door did not reach its stop of {limit}: {}",
        clear.door_state().open_deg
    );
    assert!(
        state.open_deg.abs() < limit.abs() - 1.0,
        "the wedged leaf reached the same angle as the clear one"
    );
}

/// **A level with no door costs the trace nothing and the sync nothing.**
#[test]
fn a_world_without_doors_produces_no_leaf_and_no_trace_bytes() {
    let mut world = EcsWorld::new();
    spawn_ground(&mut world);
    spawn_hero(&mut world, 0.0);
    world.mark_dirty();
    world.propagate();
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    bridge.sync_from_world(&world);
    let report = d3::step_gameplay(&mut world, &mut bridge, DT);
    assert_eq!(report.doors.doors, 0);
    assert_eq!(report.doors.leaves, 0);
    assert!(inf_ecs::door::door_state_bytes(&world).is_empty());
    assert!(bridge.body_of(d3::door_leaf_guid(DOOR)).is_none());
    // …and a door that nobody has touched still produces no trace bytes, which
    // is what keeps every pre-I6 committed trace byte-identical.
    let mut rig = Rig::new(false, 2.0);
    rig.steps(&idle(), 10);
    assert!(
        inf_ecs::door::door_state_bytes(&rig.world).is_empty(),
        "an untouched door wrote trace bytes"
    );
    rig.step(&press_e());
    assert!(!inf_ecs::door::door_state_bytes(&rig.world).is_empty());
}

// ── the grammar's own doorways ──────────────────────────────────────────────

/// Build one `House` and put its population — solids **and doorways** — on a
/// `PcgVolume`, exactly as both hosts' `population_of` does.
fn spawn_house(w: &mut EcsWorld, guid: Uuid, at: DVec3) -> usize {
    use inf_pcg::building::{ArchetypeId, BuildingParams, Rect2};
    let params = BuildingParams {
        archetype: ArchetypeId::House,
        footprint: Rect2::new(
            glam::DVec2::new(at.x - 6.0, at.z - 5.0),
            glam::DVec2::new(at.x + 6.0, at.z + 5.0),
        ),
        base_y: at.y,
        seed: 11,
        floors: 1,
    };
    let out = inf_pcg::building::build(&params, 11, false);
    let doorways = inf_pcg::building::doorways_of(&out.plan);
    let solids: Vec<inf_ecs::ScatteredSolid> = out
        .colliders
        .iter()
        .map(|c| inf_ecs::ScatteredSolid {
            center: c.center,
            half_extents: c.half_extents,
            rotation: c.rotation,
        })
        .collect();
    let slots: Vec<inf_ecs::DoorwaySlot> = doorways
        .iter()
        .map(|d| inf_ecs::DoorwaySlot {
            hinge: d.hinge,
            closed_yaw_deg: d.closed_yaw_deg,
            width_m: d.width_m,
            height_m: d.height_m,
            thickness_m: d.thickness_m,
            inside_yaw_deg: d.inside_yaw_deg,
            exterior: d.exterior,
            floor: d.floor,
        })
        .collect();
    let n = slots.len();
    let e = w.spawn_with_guid(guid, "House", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::from_dvec3(at);
    let mut vol = inf_ecs::PcgVolume::default();
    vol.set_population(
        Vec::new(),
        solids,
        Vec::new(),
        slots,
        Vec::new(),
        Default::default(),
    );
    w.world_mut().entity_mut(e).insert((t, vol));
    w.mark_dirty();
    w.propagate();
    n
}

/// **THE GRAMMAR EMITS DOORS**, and each one becomes a real leaf in the world.
///
/// The claim that makes clause 1 of the mandate true: a building the PCG
/// grammar planned has a door in every doorway it planned, hinged where the plan
/// says, with a collider a body can be stopped by — and it arrives through the
/// SAME `DoorPlacement` list an authored `Door` entity does, so one set of rules
/// swings both.
#[test]
fn a_grammar_built_house_hangs_a_leaf_in_every_doorway_it_planned() {
    let mut world = EcsWorld::new();
    spawn_ground(&mut world);
    spawn_hero(&mut world, -20.0);
    let volume = Uuid::from_u128(0x1600_00AA);
    let planned = spawn_house(&mut world, volume, DVec3::new(0.0, 0.0, 0.0));
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    bridge.sync_from_world(&world);
    let report = d3::step_gameplay(&mut world, &mut bridge, DT);
    println!(
        "a one-storey House planned {planned} doorways and the world holds {} doors",
        report.doors.doors
    );
    assert!(planned > 0, "the House plans no doors at all");
    assert_eq!(
        report.doors.doors as usize, planned,
        "the world holds a different number of doors than the plan wanted"
    );
    // **Assert the WORLD**: every one of them is a body with a collider.
    bridge.sync_from_world(&world);
    let mut leaves = 0;
    for i in 0..planned {
        let leaf = d3::door_leaf_guid(d3::door::pcg_doorway_guid(volume, i));
        if bridge.body_of(leaf).is_some() {
            assert!(
                bridge.collider_of(leaf).is_some(),
                "leaf {i} has no collider"
            );
            leaves += 1;
        }
    }
    println!("{leaves} of {planned} doorways have a live leaf body in the band");
    assert_eq!(leaves, planned, "a planned doorway got no leaf");
    // …and the identities are their own: no leaf aliases a structure collider.
    for i in 0..planned {
        let door = d3::door::pcg_doorway_guid(volume, i);
        let leaf = d3::door_leaf_guid(door);
        assert_ne!(leaf, door);
        assert_ne!(leaf, inf_physics::d3::pcg_structure_guid(volume, i));
        assert_ne!(door, inf_physics::d3::pcg_structure_guid(volume, i));
        assert_ne!(door, inf_physics::d3::pcg_shell_guid(volume, i));
    }
    // …and none of them starts locked, because nothing authored a lock: a city
    // whose every interior door was bolted is a city nobody can walk through.
    let field_before = inf_ecs::door::door_state_bytes(&world);
    assert!(
        field_before.is_empty(),
        "a fresh house touched the door field"
    );
    for p in d3::door::placements(&world) {
        assert!(!inf_ecs::door::DoorState::fresh(&p.spec).locked, "{p:?}");
    }
}

/// **The band decides how many doors are SOLID**, exactly as it decides how many
/// walls are — so a city of twenty thousand doorways costs the fixed step the
/// ones in reach.
#[test]
fn a_doorway_outside_the_collider_band_gets_no_leaf() {
    let mut world = EcsWorld::new();
    spawn_ground(&mut world);
    // A streaming source at the origin is what the band is anchored on.
    let src = world.spawn_with_guid(Uuid::from_u128(0x1600_00BB), "Source", None);
    world.world_mut().entity_mut(src).insert((
        Transform::IDENTITY,
        inf_ecs::components::StreamingSource { radius_m: 256.0 },
    ));
    let near = Uuid::from_u128(0x1600_00CC);
    let far = Uuid::from_u128(0x1600_00DD);
    let n_near = spawn_house(&mut world, near, DVec3::new(0.0, 0.0, 10.0));
    let n_far = spawn_house(&mut world, far, DVec3::new(0.0, 0.0, 4000.0));
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    bridge.sync_from_world(&world);
    let mut near_leaves = 0;
    let mut far_leaves = 0;
    for i in 0..n_near {
        if bridge
            .body_of(d3::door_leaf_guid(d3::door::pcg_doorway_guid(near, i)))
            .is_some()
        {
            near_leaves += 1;
        }
    }
    for i in 0..n_far {
        if bridge
            .body_of(d3::door_leaf_guid(d3::door::pcg_doorway_guid(far, i)))
            .is_some()
        {
            far_leaves += 1;
        }
    }
    println!(
        "with a source at the origin: {near_leaves} of {n_near} leaves 10 m away are solid, {far_leaves} of {n_far} four kilometres away are"
    );
    assert_eq!(near_leaves, n_near, "a door in reach got no collider");
    assert_eq!(far_leaves, 0, "a door four kilometres away is solid");
    // The control: with NO source the band fails open and everything is solid —
    // which is IB-2a's own rule, and the direction that keeps a body on the
    // floor rather than dropping it through the world.
    let mut open = EcsWorld::new();
    spawn_ground(&mut open);
    let n = spawn_house(&mut open, far, DVec3::new(0.0, 0.0, 4000.0));
    let mut b2 = PhysicsBridge3D::new(GRAVITY);
    b2.sync_from_world(&open);
    let mut solid = 0;
    for i in 0..n {
        if b2
            .body_of(d3::door_leaf_guid(d3::door::pcg_doorway_guid(far, i)))
            .is_some()
        {
            solid += 1;
        }
    }
    println!(
        "with no streaming source at all the band fails open: {solid} of {n} leaves are solid"
    );
    assert_eq!(solid, n, "the band did not fail open");
}

/// **AN EDGE A STEP COULD NOT USE MUST NOT SURVIVE IT** — the P29.7 A1 class,
/// at the attack button (island wave I6 audit).
///
/// The law is the wave's own: `step_weapons` takes `press_attack` from the
/// runtime **before** it decides whether anything can use it, so a press made
/// where there is nothing to kick is spent there and not somewhere else. It had
/// no arm — measured, deleting the line that clears the edge left `door_3d`,
/// `weapon_3d` and `phase30_gameplay_gate` all green — and an unconsumed edge
/// is latched by `apply_intent`'s `|=`, so it would survive for the rest of the
/// session and kick the next door that happened to be locked.
///
/// The press here lands on an **unlocked** door in reach, which is the case
/// that arms nothing. The door is then locked the way the lock verb would lock
/// it, and the arm asserts that sixty idle steps kick nothing.
#[test]
fn an_attack_spent_at_an_unlocked_door_does_not_kick_it_when_it_is_locked_later() {
    let mut rig = Rig::new(false, 3.0);
    rig.step(&idle());
    let r = rig.step(&press_attack());
    assert_eq!(r.kicks, 0, "an unlocked door was kicked");
    assert!(
        door::pending_kick(&rig.world, HERO).is_none(),
        "an unlocked door armed a kick"
    );
    // Lock it — the state the press could not have known about when it was made.
    {
        let spec = rig.door_spec();
        door::door_field_mut(&mut rig.world)
            .entry(DOOR, &spec)
            .locked = true;
    }
    assert!(rig.door_state().locked, "the fixture did not lock");
    let mut later = 0;
    for _ in 0..60 {
        later += rig.step(&idle()).kicks;
    }
    println!("sixty idle steps after a spent press landed {later} kick(s)");
    assert_eq!(
        later, 0,
        "the press outlived the step that could not use it and kicked a door \
         that was locked afterwards - `press_attack` is not being consumed"
    );
    assert!(
        !rig.door_state().lock_broken,
        "the lock broke with nobody pressing anything"
    );
    // …and the control: a press made NOW does kick it, so the arm above is
    // about the edge's lifetime rather than about a rig that cannot kick.
    let mut kicks = 0;
    for _ in 0..40 {
        kicks += rig.steps(&press_attack(), 1).kicks;
    }
    assert_eq!(kicks, 1, "a fresh press did not kick the locked door");
    assert!(rig.door_state().lock_broken);
}

/// **`door.is_open` answers about the door that is THERE** — the Blueprint kit's
/// one read, which had no arm at all (I6 audit).
///
/// Both kinds under one question, which is the point of the flattened
/// `DoorPlacement` list: an authored `Door` entity and a doorway the grammar
/// planned are the same subject. And a point with no door near it answers
/// `false`, which is the honest value rather than a refusal.
#[test]
fn the_is_open_probe_reads_authored_doors_grammar_doorways_and_empty_air() {
    let mut world = EcsWorld::new();
    spawn_ground(&mut world);
    spawn_door(&mut world, false);
    spawn_hero(&mut world, 3.0);
    let volume = Uuid::from_u128(0x1600_00AB);
    let planned = spawn_house(&mut world, volume, DVec3::new(40.0, 0.0, 0.0));
    assert!(planned > 0);
    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    bridge.sync_from_world(&world);

    // The authored door: shut, so the probe says so from its own doorway.
    let spec = {
        let e = world.entity_of(DOOR).expect("the door");
        world.world().get::<Door>(e).expect("a door").spec
    };
    let at = HINGE + DVec3::new(spec.width_m * 0.5, 0.0, 0.0);
    assert!(
        !d3::door::is_open_near(&world, at),
        "a shut door read as open"
    );
    // Empty air is `false` and not a panic — there is no door out there.
    assert!(!d3::door::is_open_near(&world, DVec3::new(0.0, 0.0, 900.0)));

    // Open it through the one resolution site, and the same probe flips.
    let feet = DVec3::new(0.0, 0.0, 3.0);
    assert_eq!(
        d3::door::use_door(&mut world, DOOR, feet),
        inf_ecs::door::DoorVerdict::Opening
    );
    for _ in 0..90 {
        d3::step_gameplay(&mut world, &mut bridge, DT);
    }
    assert!(
        d3::door::is_open_near(&world, at),
        "an open door read as shut"
    );

    // A GRAMMAR doorway, forty metres away: shut, then open, through the same
    // probe and the same field.
    let house: Vec<_> = d3::door::placements(&world)
        .into_iter()
        .filter(|p| p.guid != DOOR)
        .collect();
    assert_eq!(house.len(), planned, "the house's doorways are not listed");
    let target = &house[0];
    let where_it_is = inf_ecs::door::prompt_position(target);
    assert!(
        !d3::door::is_open_near(&world, where_it_is),
        "a fresh grammar doorway read as open"
    );
    let spec = target.spec;
    let guid = target.guid;
    {
        let field = inf_ecs::door::door_field_mut(&mut world);
        let s = field.entry(guid, &spec);
        s.open_deg = spec.open_limit_deg;
    }
    assert!(
        d3::door::is_open_near(&world, where_it_is),
        "an open grammar doorway read as shut"
    );
    // …and the authored door forty metres away is not what answered.
    assert!(!d3::door::is_open_near(
        &world,
        where_it_is + DVec3::new(0.0, 0.0, 100.0)
    ));
}

/// **A DOOR YOU BARGED THROUGH IS STILL A DOOR YOU CAN LOCK** (I6 audit).
///
/// `try_break` answers `broke` for a shut-but-unlocked door too — nothing was
/// holding it — and `apply_break` used to mark that as a broken lock. A broken
/// lock never re-engages, so one sprint through a house's own front door
/// retired its lock for the session; **every** door the building grammar emits
/// starts unlocked, so on the shipped city that was every door in it.
///
/// The control is the locked half: a lock that *was* holding still breaks and
/// still refuses to come back, which is what makes this arm about the
/// distinction rather than about the flag.
#[test]
fn a_sprint_through_an_unlocked_door_leaves_a_lock_that_still_works() {
    let mut rig = Rig::new(false, 3.4);
    rig.set_hero_velocity(DVec3::new(0.0, 0.0, 6.5));
    rig.step(&idle());
    let after = rig.door_state();
    println!("after barging through a shut door: {after:?}");
    assert!(after.open_deg != 0.0, "the sprint did not go through");
    assert!(
        !after.lock_broken,
        "a door that was not locked had its lock broken by a shoulder"
    );
    // …so the lock verb is still on offer, from the inside, and it works.
    //
    // **Two presses, on two controls** (island wave I8b): E shuts the leaf and
    // the lock control throws the bolt. The bolt still refuses an OPEN door —
    // a door standing open with its bolt thrown would be a lock nobody could
    // see — which is why the shut has to come first. That is the gate's own
    // lock station, met here from the other side of a breach.
    rig.steps(&idle(), 90);
    let spec = rig.door_spec();
    let inside = HINGE + DVec3::new(0.0, 0.0, 1.0);
    assert_eq!(spec.side_of(HINGE, inside), inf_ecs::door::DoorSide::Inside);
    let shut = d3::door::use_door(&mut rig.world, DOOR, inside);
    assert_eq!(shut, inf_ecs::door::DoorVerdict::Closing, "{shut:?}");
    rig.steps(&idle(), 90);
    // Not "exactly zero": the hero is standing where it landed, which is in the
    // leaf's own arc, so the closing leaf stops against its capsule a few
    // degrees out. That is the system working (the gate's lock station found the
    // same thing from the other direction) and it is still shut for the lock's
    // purposes, which is the question here.
    assert!(
        !rig.door_state().is_open(&spec),
        "the leaf did not shut: {:?}",
        rig.door_state()
    );
    let v = d3::door::lock_door(&mut rig.world, DOOR, inside);
    println!("shutting with E then locking from the inside face: {shut:?}, then {v:?}");
    assert!(
        rig.door_state().locked,
        "the door could not be locked after being barged through: {v:?}"
    );

    // The control: a lock that WAS holding breaks, and stays broken.
    let mut locked = Rig::new(true, 3.4);
    locked.set_hero_velocity(DVec3::new(0.0, 0.0, 6.5));
    locked.step(&idle());
    assert!(locked.door_state().lock_broken, "the locked control held");
    assert!(!locked.door_state().locked);
    let spec = locked.door_spec();
    assert_eq!(
        inf_ecs::door::set_locked(
            &spec,
            &mut locked.door_state(),
            inf_ecs::door::DoorSide::Inside,
            true
        ),
        inf_ecs::door::DoorVerdict::WrongSide,
        "a broken lock re-engaged"
    );
}

// -- NPC1c: the crowd's own door verb -----------------------------------------

/// **AN NPC OPENS THE DOOR IN ITS WAY** (island wave NPC1c, clause 3).
///
/// The verb is `d3::door::use_door` -- the same function the interact button and
/// the `door.use` node dispatch to, so a crowd agent cannot open a door the
/// player could not. What is new is the TRIGGER: a crowd agent whose body has
/// fallen `BLOCKED_LAG_M` behind its own route clock, which is what standing
/// against a shut leaf makes it.
///
/// Three claims and a control, because "the pass ran" and "a door opened" are
/// different facts:
///
/// * an agent that is not yet blocked presses **nothing** -- the pass is not a
///   per-step door-mash;
/// * once blocked, it presses once and the **world** says the leaf moved (the
///   door's own `DoorField` state, never the pass's report -- this file's own
///   header rule);
/// * a **locked** door is pressed and refuses, exactly as it refuses a player,
///   and the counters say `pressed` without `opened` -- which is the number a
///   designer wants when a district stops working.
fn crowd_at_the_door(locked: bool) -> (EcsWorld, PhysicsBridge3D) {
    use inf_ecs::crowd::{CrowdArchetype, CrowdRecord, CrowdRoute};
    use std::collections::BTreeMap;

    let mut world = EcsWorld::new();
    spawn_ground(&mut world);
    spawn_door(&mut world, locked);
    // The streaming source stands ON the doorway, so the agent beside it is
    // `Full` -- a tier with a body, a controller and a movement model, which is
    // what `feet_of` and the blocked verdict both need.
    let src = world.spawn_with_guid(Uuid::from_u128(0xF1), "Player", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, 0.0, 4.0);
    world
        .world_mut()
        .entity_mut(src)
        .insert((inf_ecs::components::StreamingSource { radius_m: 0.0 }, t));
    world.mark_dirty();
    world.propagate();

    // One agent standing a stride outside the doorway, on a route that walks
    // straight through it. Nothing in this test runs `step_character_movement`,
    // so the body stands where it is placed while the clock runs on -- which is
    // exactly the state a body wedged against a shut leaf is in.
    let mut records = BTreeMap::new();
    records.insert(
        Uuid::from_u128(0xC0DE_1C01),
        CrowdRecord::walking(
            CrowdArchetype::humanoid(None, None, None),
            CrowdRoute::between(DVec3::new(0.0, 0.0, 3.0), DVec3::new(0.0, 0.0, 8.0), 1.4),
        ),
    );
    inf_ecs::crowd::set_population(&mut world, records);

    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    inf_ecs::crowd::step_crowd(&mut world, DT);
    world.propagate();
    bridge.sync_from_world(&world);
    (world, bridge)
}

fn door_is_open(world: &EcsWorld) -> bool {
    let p = d3::door::placement_of(world, DOOR).expect("the fixture door");
    inf_ecs::door::door_field(world)
        .map(|f| f.get(DOOR, &p.spec))
        .unwrap_or_else(|| DoorState::fresh(&p.spec))
        .is_open(&p.spec)
}

#[test]
fn a_blocked_crowd_agent_opens_the_door_it_is_standing_against() {
    let (mut world, mut bridge) = crowd_at_the_door(false);

    // The control: on the step it materializes, the agent is not behind its own
    // clock and the pass does nothing at all.
    let first = d3::step_gameplay(&mut world, &mut bridge, DT).crowd_doors;
    assert_eq!(
        (first.considered, first.pressed),
        (0, 0),
        "an unblocked agent pressed a door -- the trigger is not the lag"
    );
    assert!(!door_is_open(&world), "the fixture starts with a shut door");

    // Now let the clock run on while the body stands. `BLOCKED_LAG_M` is 2 m and
    // the agent walks 1.4 m/s, so this is about 1.2 s of leaning on the door.
    let mut opened_at = None;
    let mut pressed_total = 0usize;
    for step in 0..200u32 {
        inf_ecs::crowd::step_crowd(&mut world, DT);
        world.propagate();
        // Through the WHOLE gameplay step, not the pass alone: the leaf has to
        // swing, and `step_doors` is what swings it. That ordering -- press
        // first, swing second -- is the reason this pass runs where it does.
        let r = d3::step_gameplay(&mut world, &mut bridge, DT).crowd_doors;
        pressed_total += r.pressed;
        if r.opened > 0 && opened_at.is_none() {
            opened_at = Some(step);
        }
        if door_is_open(&world) {
            break;
        }
    }
    let at = opened_at.expect(
        "a blocked crowd agent standing at a shut door never opened it over 200 \
         steps -- the door verb has no caller",
    );
    assert!(
        door_is_open(&world),
        "the report moved and the WORLD did not"
    );
    assert!(
        pressed_total <= 2,
        "the pass pressed {pressed_total} times to open one door"
    );
    println!(
        "NPC1c door verb: a blocked agent opened its door at step {at} ({:.2} s)",
        f64::from(at) * DT
    );
}

#[test]
fn a_locked_door_refuses_a_crowd_agent_exactly_as_it_refuses_a_player() {
    let (mut world, mut bridge) = crowd_at_the_door(true);
    let mut pressed = 0usize;
    let mut opened = 0usize;
    for _ in 0..200 {
        inf_ecs::crowd::step_crowd(&mut world, DT);
        world.propagate();
        let r = d3::step_gameplay(&mut world, &mut bridge, DT).crowd_doors;
        pressed += r.pressed;
        opened += r.opened;
    }
    assert!(
        pressed > 0,
        "the agent never reached the locked door, so this arm is not posing the \
         problem"
    );
    assert_eq!(opened, 0, "a locked door let a crowd agent through");
    assert!(!door_is_open(&world));
    println!("NPC1c door verb: {pressed} press(es) at a locked door, 0 opened");
}

/// **WHAT A CITY'S DOORWAY GATHER COSTS, AND WHY THE PHASE PAYS IT ONCE**
/// (NPC1c audit — the wave's own carried item 2, priced and closed).
///
/// `placements_near` visits **every** `DoorwaySlot` a level plans and keeps the
/// handful the band admits. That visit is not free: the shipped city plans
/// 19 790 doorways, and NPC1c added a second per-step caller
/// (`gameplay::step_crowd_doors`) beside `step_doors`. Its own ledger carried
/// the arithmetic:
///
/// > *"`step_gameplay` gathers the band's doors TWICE a step … measured as part
/// > of the `gameplay +0.84 ms` at N = 1 000 with 32 blocked agents. One shared
/// > gather is the fix; it is a signature change to two functions and it is not
/// > landed here."*
///
/// It is landed now, and the guarantee is **structural rather than a clock**:
/// `step_crowd_doors` no longer takes a `PhysicsBridge3D`, so it has nothing to
/// derive a band from and cannot gather at all — the list is a parameter, taken
/// once by `step_gameplay` and handed to both passes. A future edit that gave it
/// a bridge back would have to say so in its signature.
///
/// What this arm adds is the **price of one gather** at city scale, printed, so
/// the saving has a number beside it rather than a ratio. The clock is printed
/// and the *shape* is asserted, per this crate's standing rule.
#[test]
fn a_citys_doorway_gather_is_paid_once_a_gameplay_step() {
    use std::time::Instant;

    // The shipped city's own order of magnitude — 19 790 doorway slots — as one
    // volume, so the arm measures the VISIT rather than the entity walk.
    const SLOTS: usize = 19_790;
    let mut w = EcsWorld::new();
    let src = w.spawn_with_guid(Uuid::from_u128(0x1600_00F1), "Player", None);
    w.world_mut().entity_mut(src).insert((
        Transform::IDENTITY,
        inf_ecs::components::StreamingSource { radius_m: 256.0 },
    ));
    let e = w.spawn_with_guid(Uuid::from_u128(0x1600_00F2), "City", None);
    // Spread over 1.26 km, the shipped city's own span, so the band admits a
    // fraction rather than all or nothing.
    let slots: Vec<inf_ecs::DoorwaySlot> = (0..SLOTS)
        .map(|i| {
            let a = i as f64 * 0.7;
            inf_ecs::DoorwaySlot {
                hinge: DVec3::new(a % 1260.0 - 630.0, 0.0, (a * 1.7) % 1260.0 - 630.0),
                closed_yaw_deg: 0.0,
                width_m: 0.9,
                height_m: 2.1,
                thickness_m: 0.05,
                inside_yaw_deg: 90.0,
                exterior: i % 40 == 0,
                floor: 0,
            }
        })
        .collect();
    let mut vol = inf_ecs::PcgVolume::default();
    vol.set_population(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        slots,
        Vec::new(),
        Default::default(),
    );
    w.world_mut()
        .entity_mut(e)
        .insert((Transform::IDENTITY, vol));
    w.mark_dirty();
    w.propagate();

    let mut bridge = PhysicsBridge3D::new(GRAVITY);
    bridge.sync_from_world(&w);
    let band = bridge.sim_band(&w);

    // MIN of five rounds of twenty gathers, the `kinematic_pairing_cost` shape.
    let mut near = 0usize;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..20 {
            near = d3::door::placements_near(&w, &band).len();
        }
        best = best.min(t.elapsed().as_secs_f64() * 1000.0 / 20.0);
    }
    println!(
        "NPC1c audit: one `placements_near` over {SLOTS} doorway slots is \
         {best:.4} ms and keeps {near} ({:.2} % of them); `step_gameplay` paid \
         it TWICE a step with a blocked crowd and now pays it once",
        100.0 * near as f64 / SLOTS as f64
    );
    // ANTI-VACUITY: a gather that admitted nothing would be the cheapest of all,
    // and a gather that admitted everything would not be a band.
    assert!(best > 0.0, "the gather took no measurable time");
    assert!(near > 0, "the band admitted no doorway at all");
    assert!(
        near < SLOTS / 4,
        "the band admitted {near} of {SLOTS} slots, which is not a band"
    );
}
