//! **RECOGNITION, AGAINST A WORLD** (wave EMS3) — an officer, a suspect, and a
//! wall.
//!
//! The channel arithmetic, the freshness curve, the severity ladder and the two
//! `last_seen` writers are unit-tested in `inf_ecs::crime`, where they need no
//! rapier. This file exists for the four claims that are only true of a *world*:
//!
//! * an officer with a clear line to somebody who matches the description
//!   **recognises them**, and the ledger moves to where the officer saw them;
//! * a **wall** between them stops it, and the pass says so rather than
//!   silently scoring zero;
//! * a suspect who **changed their clothes and left the car** walks past in full
//!   view and is not recognised — the mandate's evasion route, as a measurement;
//! * and **THE POLICE DO NOT CHEAT**: with the wall in place the suspect walks a
//!   hundred metres and `last_seen` does not move a millimetre, which is the
//!   falsifiable form of "the search converges on last-seen and never on the
//!   player's true transform".
//!
//! `step_recognition` is called directly rather than through `step_dispatch`,
//! and the fleet and the runs are installed by hand: getting a cruiser out of
//! its bay needs a carriageway, a route and several thousand steps of driving,
//! which is `ems3_crime_gate`'s job. What is under test here is the *look*.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, StreamingSource, TimeOfDay, Transform,
};
use inf_ecs::crime;
use inf_ecs::crowd::{set_appearance, Appearance};
use inf_ecs::dispatch::{self, FleetRes, FleetUnit, UnitKind, UnitRun, UnitState};
use inf_ecs::math::{Color, Vec3d};
use inf_ecs::witness::{ActKind, WitnessedAct};
use inf_ecs::EcsWorld;
use inf_physics::d3::PhysicsBridge3D;

const GROUND: Uuid = Uuid::from_u128(0x0E53_0001);
const CRUISER: Uuid = Uuid::from_u128(0x0E53_0002);
const SUSPECT: Uuid = Uuid::from_u128(0x0E53_0003);
const WITNESS: Uuid = Uuid::from_u128(0x0E53_0004);
const WALL: Uuid = Uuid::from_u128(0x0E53_0005);
const GETAWAY: Uuid = Uuid::from_u128(0x0E53_0006);
/// An innocent man in the same coat as the criminal.
const TWIN: Uuid = Uuid::from_u128(0x0E53_0007);

/// The outfit the crime is committed in, and the one it is changed into.
const OUTFIT_A: u8 = 2;
const OUTFIT_B: u8 = 6;

struct Beat {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Beat {
    /// An officer at the origin, a suspect `d` metres down `+X`, and a getaway
    /// car parked beside the suspect.
    fn new(d: f64, hour: f64) -> Self {
        let mut world = EcsWorld::new();
        // Ground, so the ray has a world to be cast in at all.
        let e = world.spawn_with_guid(GROUND, "Ground", None);
        world.world_mut().entity_mut(e).insert((
            Transform::from_translation(DVec3::new(0.0, -0.5, 0.0)),
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(400.0, 0.5, 400.0),
                ..Default::default()
            },
        ));
        let sky = world.spawn_with_guid(Uuid::from_u128(0x0E53_00FF), "Sky", None);
        world.world_mut().entity_mut(sky).insert(TimeOfDay {
            seconds: hour * 3600.0,
            rate: 0.0,
            ..Default::default()
        });
        // The officer's body. A crew member is `crew_guid(chassis)`, which is
        // the guid `step_recognition` derives — spelled through the same door so
        // the fixture cannot disagree with the engine about who the officer is.
        let officer = dispatch::crew_guid(CRUISER);
        let e = world.spawn_with_guid(officer, "Officer", None);
        world.world_mut().entity_mut(e).insert((
            Transform::from_translation(DVec3::new(0.0, 1.0, 0.0)),
            StreamingSource { radius_m: 1024.0 },
        ));
        // The suspect.
        let e = world.spawn_with_guid(SUSPECT, "Suspect", None);
        world
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(DVec3::new(d, 1.0, 0.0)));
        // The getaway car, beside them — a real chassis, so `vehicle_digest`
        // reads a collider and a paint rather than a fixture's opinion.
        let e = world.spawn_with_guid(GETAWAY, "Getaway", None);
        world.world_mut().entity_mut(e).insert((
            Transform::from_translation(DVec3::new(d, 0.6, 3.0)),
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(0.92, 0.6, 2.2),
                ..Default::default()
            },
            inf_ecs::components::Material {
                base_color: Color::new(0.85, 0.1, 0.1, 1.0),
                ..Default::default()
            },
        ));
        set_appearance(&mut world, SUSPECT, Appearance { outfit: OUTFIT_A });
        // ONE police unit, out of its bay. Installed by hand — see the header.
        let mut units = std::collections::BTreeMap::new();
        units.insert(
            CRUISER,
            FleetUnit {
                kind: UnitKind::Police,
                station: Uuid::nil(),
                home: DVec3::ZERO,
                home_yaw_deg: 0.0,
            },
        );
        world.world_mut().insert_resource(FleetRes {
            units,
            stamp: 1,
            derivations: 1,
        });
        let mut runs = std::collections::BTreeMap::new();
        runs.insert(
            CRUISER,
            UnitRun {
                state: UnitState::EnRoute,
                ..Default::default()
            },
        );
        world
            .world_mut()
            .insert_resource(inf_ecs::dispatch::DispatchRes {
                runs,
                ..Default::default()
            });
        world.mark_dirty();
        world.propagate();
        let mut beat = Self {
            world,
            bridge: PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)),
        };
        beat.sync();
        beat
    }

    fn sync(&mut self) {
        self.bridge
            .sync_from_world_sim(&self.world, &Default::default(), &Default::default());
    }

    /// A solid wall across the line of sight, half way.
    fn wall(&mut self, x: f64) {
        let e = self.world.spawn_with_guid(WALL, "Wall", None);
        self.world.world_mut().entity_mut(e).insert((
            Transform::from_translation(DVec3::new(x, 2.0, 0.0)),
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(0.2, 4.0, 20.0),
                ..Default::default()
            },
        ));
        self.world.mark_dirty();
        self.world.propagate();
        self.sync();
    }

    /// Commit a carjack in front of a witness, at `step`, and file it.
    fn commit(&mut self, step: u64) {
        let look = inf_ecs::witness::look_digest(&self.world, SUSPECT);
        inf_ecs::witness::record_act(
            &mut self.world,
            WitnessedAct {
                kind: ActKind::Carjack,
                actor: SUSPECT,
                at: DVec3::new(200.0, 1.0, 200.0),
                step,
                observers: vec![WITNESS],
                actor_look: look,
                actor_vehicle: Some(GETAWAY),
            },
        );
        let filed = crime::file_new_acts(&mut self.world);
        assert_eq!(filed, 1, "the crime was not filed");
    }

    /// Put somebody with no criminal record at `at`, wearing `outfit`.
    fn bystander(&mut self, guid: Uuid, at: DVec3, outfit: u8) {
        let e = self.world.spawn_with_guid(guid, "Bystander", None);
        self.world
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(at));
        set_appearance(&mut self.world, guid, Appearance { outfit });
        self.world.mark_dirty();
        self.world.propagate();
        self.sync();
    }

    fn place(&mut self, guid: Uuid, at: DVec3) {
        let e = self.world.entity_of(guid).expect("in the world");
        if let Some(mut t) = self.world.world_mut().get_mut::<Transform>(e) {
            t.translation = Vec3d::from_dvec3(at);
        }
        self.world.propagate();
    }

    fn look(&mut self, step: u64) -> inf_physics::d3::RecognitionStats {
        inf_physics::d3::crime::step_recognition(&mut self.world, &mut self.bridge, step)
    }

    fn last_seen(&self) -> Option<DVec3> {
        crime::profile_of(&self.world, SUSPECT).map(|p| p.last_seen())
    }
}

/// **A CLEAR VIEW OF SOMEBODY WHO MATCHES IS A RECOGNITION**, and the file moves
/// to where the officer saw them.
#[test]
fn an_officer_with_a_clear_line_recognises_the_description() {
    let mut beat = Beat::new(8.0, 12.0);
    beat.commit(10);
    // Filed at the crime scene, 280 m away — the only place a witness put them.
    assert_eq!(beat.last_seen(), Some(DVec3::new(200.0, 1.0, 200.0)));
    let s = beat.look(11);
    println!(
        "officers {} files {} in-range {} rays {} blocked {} unrecognised {} recognised {}",
        s.officers, s.files, s.in_range, s.rays, s.blocked, s.unrecognised, s.recognised
    );
    assert_eq!(s.officers, 1, "the crew was not on the beat");
    assert_eq!(s.files, 1);
    assert_eq!(s.in_range, 1);
    assert_eq!(s.rays, 1, "the ray budget was not spent on the one pair");
    assert_eq!(s.blocked, 0);
    assert_eq!(
        s.recognised, 1,
        "a matching description in plain view was missed"
    );
    // …and the ledger now holds the place the OFFICER saw them, not the scene.
    assert_eq!(beat.last_seen(), Some(DVec3::new(8.0, 1.0, 0.0)));
}

/// **A WALL IS NOT A DISCOUNT** — and the pass says which of the two it was.
#[test]
fn a_wall_between_them_stops_the_recognition_and_is_counted() {
    let mut beat = Beat::new(8.0, 12.0);
    beat.wall(4.0);
    beat.commit(10);
    let before = beat.last_seen().expect("a file");
    let s = beat.look(11);
    println!(
        "with a wall: rays {} blocked {} recognised {}",
        s.rays, s.blocked, s.recognised
    );
    assert_eq!(s.rays, 1, "the pair was not even considered");
    assert_eq!(s.blocked, 1, "the ray went through a 8 m wall");
    assert_eq!(s.recognised, 0);
    assert_eq!(
        beat.last_seen(),
        Some(before),
        "a blocked view moved the file"
    );
}

/// **THE MANDATE'S OWN EVASION ROUTE, MEASURED**: change the coat, leave the
/// car, walk past in full view.
///
/// The three states are asserted in order against **one** officer at **one**
/// distance in **one** light, so the only thing that changed between them is
/// what the suspect is wearing and driving.
#[test]
fn changing_the_clothes_and_leaving_the_car_defeats_the_description() {
    let mut beat = Beat::new(8.0, 12.0);
    beat.commit(10);
    // (a) unchanged: recognised.
    assert_eq!(beat.look(11).recognised, 1);
    let seen_at = beat.last_seen().expect("a file");

    // (b) the CAR is gone from the description the moment they are not in it —
    //     which is already true here, because a suspect on foot answers `None`
    //     for the vehicle channel. So the coat alone is what is being read, and
    //     at 8 m in daylight it is still enough.
    let outfit_only = crime::describe(&beat.world, SUSPECT);
    assert_eq!(outfit_only.vehicle, None);

    // (c) …and the coat goes at a wardrobe. Same officer, same 8 m, same noon.
    set_appearance(&mut beat.world, SUSPECT, Appearance { outfit: OUTFIT_B });
    let s = beat.look(12);
    println!(
        "after the swap: in-range {} rays {} unrecognised {} recognised {}",
        s.in_range, s.rays, s.unrecognised, s.recognised
    );
    assert_eq!(s.recognised, 0, "a swapped outfit was still recognised");
    assert_eq!(
        s.rays, 0,
        "a ray was spent on somebody who scores zero on every channel"
    );
    assert_eq!(s.unrecognised, 1, "the pass did not even consider the pair");
    assert_eq!(
        beat.last_seen(),
        Some(seen_at),
        "an unrecognised suspect moved the file"
    );
}

/// **AN INNOCENT MAN IN THE CRIMINAL'S COAT IS NEVER LOOKED AT** (wave EMS3
/// audit) — the bound the wave's prose did not have.
///
/// # What three doc comments claimed
///
/// `crowd::Appearance`, `witness::actor_look` and the wave's ledger all said
/// that two people dressed alike collide *"on purpose"* and that this is *"what
/// a description costs somebody innocent"*; the carried list priced it at a
/// one-in-eight false-positive rate. The channel does collide — that half is
/// measured in `witness`' own arm — but the recognition pass walks
/// `officers x wanted` and scores each suspect **against their own file**, so a
/// person with no file is never in a scored pair at all. The cost of a
/// description to an innocent man is, today, **zero**.
///
/// That is a legitimate cost bound (the honest walk is
/// `officers x candidates_near x files`) and it is not a weakening of the
/// police-don't-cheat law — nothing here can write a `last_seen` without a ray.
/// It is written down as an arm so the day somebody implements the wrong man
/// being stopped, a test fails instead of a paragraph being believed.
#[test]
fn an_innocent_in_the_same_coat_is_never_looked_at() {
    let mut beat = Beat::new(8.0, 12.0);
    beat.commit(10);
    // The twin: no file, the criminal's exact outfit, standing two metres from
    // the officer — nearer than the suspect and in the same clear daylight.
    beat.bystander(TWIN, DVec3::new(2.0, 1.0, 0.0), OUTFIT_A);
    assert_eq!(
        inf_ecs::witness::look_digest(&beat.world, TWIN),
        inf_ecs::witness::look_digest(&beat.world, SUSPECT),
        "the fixture did not actually dress the two men alike"
    );
    let s = beat.look(11);
    println!(
        "one criminal at 8 m and one innocent twin at 2 m: {} file(s), \
         {} pair(s) in range, {} ray(s), {} recognition(s)",
        s.files, s.in_range, s.rays, s.recognised
    );
    // ONE pair, and it is the officer with the file's own subject. A pass that
    // compared what it could see against what it was carrying would read two.
    assert_eq!(s.files, 1);
    assert_eq!(
        s.in_range, 1,
        "the pass formed {} pair(s) — it is scoring somebody who has no file",
        s.in_range
    );
    assert_eq!(s.recognised, 1, "the criminal was missed");
    // …and the ledger holds the CRIMINAL's place and not the twin's, which is
    // what a false positive would have written.
    assert_eq!(beat.last_seen(), Some(DVec3::new(8.0, 1.0, 0.0)));

    // Now take the criminal out of range entirely and leave only the twin in
    // front of the officer. Nobody is scored, nothing is looked at, and a man
    // in a wanted coat walks past a policeman who has his description.
    beat.place(SUSPECT, DVec3::new(300.0, 1.0, 0.0));
    beat.sync();
    let s = beat.look(12);
    println!(
        "  the criminal 300 m away, the twin still at 2 m: {} pair(s) in range, \
         {} ray(s), {} recognition(s)",
        s.in_range, s.rays, s.recognised
    );
    assert_eq!(s.in_range, 0, "somebody without a file was measured");
    assert_eq!(s.rays, 0);
    assert_eq!(s.recognised, 0);
}

/// **THE POLICE DO NOT CHEAT** — the wave's signature arm, at unit scale.
///
/// With a wall in the way the suspect walks a hundred metres, one step at a
/// time, and the ledger's `last_seen` does not move. Every step calls the same
/// pass the engine calls; the only thing withheld is a line of sight.
///
/// The falsifier is built in: the suspect's *true* position is printed beside
/// the ledger's, and they end a hundred metres apart. A pass that read the
/// transform would have them equal.
#[test]
fn the_search_never_learns_a_position_nobody_saw() {
    let mut beat = Beat::new(8.0, 12.0);
    beat.wall(4.0);
    beat.commit(10);
    let filed_at = beat.last_seen().expect("a file");
    let mut rays = 0usize;
    let mut blocked = 0usize;
    for step in 11..111u64 {
        // A metre a step, straight down `+X`, in the SAME coat and past the same
        // officer — so nothing about the description has changed and only the
        // wall is doing the work.
        beat.place(SUSPECT, DVec3::new(8.0 + (step - 11) as f64, 1.0, 0.0));
        beat.sync();
        let s = beat.look(step);
        rays += s.rays;
        blocked += s.blocked;
        assert_eq!(s.recognised, 0, "recognised through a wall on step {step}");
    }
    let truth = beat
        .world
        .entity_of(SUSPECT)
        .and_then(|e| beat.world.world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .expect("the suspect is in the world");
    println!(
        "after 100 steps: the suspect is at {truth:?}, the police believe {:?} \
         ({rays} rays cast, {blocked} blocked)",
        beat.last_seen()
    );
    assert!(
        rays > 0,
        "no ray was ever cast — this arm certified a pass that did nothing"
    );
    // The 32 is the RANGE GATE and not `MAX_RECOGNITION_RAYS`, which happens to
    // be the same number: the suspect starts 8 m out and walks a metre a step,
    // so the pair leaves `RECOGNITION_RANGE_M` (40 m) after 32 of them and the
    // remaining 68 steps cost nothing at all. Said out loud because two 32s in
    // one printout is exactly the coincidence a later reader mis-diagnoses.
    assert_eq!(
        rays,
        (crime::RECOGNITION_RANGE_M as usize) - 8,
        "the range gate is not what bounded this run"
    );
    assert_eq!(blocked, rays, "some ray got through the wall");
    assert_eq!(
        beat.last_seen(),
        Some(filed_at),
        "the ledger learned a position no witness ever gave it"
    );
    assert!(
        (truth - filed_at).length() > 100.0,
        "the fixture did not actually separate the two"
    );
}
