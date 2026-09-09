//! **WAVE WPN2c — SOUND + BRASS.** The gate.
//!
//! # What every arm in this file reads
//!
//! **The command stream and the world.** A gunshot has no representation in the
//! world at all — it is thirty milliseconds of air — so the only thing there is
//! to assert about it is the sequence of [`inf_audio::AudioCommand`]s two hosts
//! push, and this file reads them command for command: how many, in what order,
//! on which keys, naming which clips, at which volumes and cutoffs. The brass
//! and the round DO exist in the world, and those arms read the pool.
//!
//! Never a report's summary of itself, and never a claim in a doc.
//!
//! # `dropped == 0` comes first
//!
//! `RuntimeSim::audio_command_log` is a `BoundedLog`. If it has evicted, the
//! slice is a TAIL and every arm that counts from the beginning is reasoning
//! about a window it did not choose. So the count arms assert
//! `dropped_audio_commands() == 0` before they compare anything, and one arm
//! exists to prove the ceiling holds at the population this wave created.
//!
//! # The vacuity rule this wave is built against
//!
//! A weapon that fired with wave WPN1's single-layer report would pass an arm
//! that only counted `Play`s naming `WEAPON_REPORT_CLIP` — that clip still
//! exists and is still played, because it is the assault rifle's BODY layer. So
//! every arm below that claims something about the four layers counts all four
//! and names the three that did not exist before, and the ones that could have
//! been satisfied by the old report say so in their own doc.
//!
//! # The mutations each arm dies to
//!
//! Written beside the arm, not in a report. They were run.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use glam::DVec3;
use uuid::Uuid;

use inf_audio::AudioCommand;
use inf_ecs::ballistics::{self, CRACK_RADIUS_M, SPEED_OF_SOUND_MPS};
use inf_ecs::casing::{self, CasingPool, MAX_CASINGS_LIVE};
use inf_ecs::components::{
    BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, RigidBody3D, Transform,
};
use inf_ecs::item::{self, ItemDef, ItemDefs};
use inf_ecs::math::Vec3d;
use inf_ecs::weapon::{self, ReportClip, ReportLayerKind, ShotKind, WeaponClass, WeaponDef};
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};
use inf_project::ProjectManifest;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const RADIUS: f64 = 0.3;

const HERO: Uuid = Uuid::from_u128(0x2C00_0001);
const GROUND: Uuid = Uuid::from_u128(0x2C00_0002);
const EAR: Uuid = Uuid::from_u128(0x2C00_0003);

fn wall_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2C00_0100 + i as u128)
}

fn shooter_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2C00_0200 + i as u128)
}

// ── the range ───────────────────────────────────────────────────────────────

/// **A place to fire from**, with a floor and — optionally — a room around it.
///
/// The two shapes are the whole of what the enclosure probe distinguishes, so
/// the fixture builds them and nothing else: a STREET is a floor, and a ROOM is
/// a floor, a ceiling and four walls inside the probe's own eight metres.
struct Range {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Range {
    fn new(defs: ItemDefs, indoors: bool) -> Self {
        let mut world = EcsWorld::new();
        slab(
            &mut world,
            GROUND,
            "Ground",
            DVec3::new(0.0, -0.5, 0.0),
            Vec3d::new(60.0, 0.5, 60.0),
        );
        if indoors {
            // A room 6 m across and 3 m high: four walls, a ceiling, and the
            // floor that is already there. Every one of the six probe rays from
            // a muzzle at its centre finds something inside eight metres.
            slab(
                &mut world,
                wall_guid(0),
                "Ceiling",
                DVec3::new(0.0, 3.0, 0.0),
                Vec3d::new(6.0, 0.2, 6.0),
            );
            for (i, (x, z)) in [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)]
                .into_iter()
                .enumerate()
            {
                let half = if x == 0.0 {
                    Vec3d::new(6.0, 3.0, 0.2)
                } else {
                    Vec3d::new(0.2, 3.0, 6.0)
                };
                slab(
                    &mut world,
                    wall_guid(i + 1),
                    "Wall",
                    DVec3::new(x, 1.5, z),
                    half,
                );
            }
        }
        stand(&mut world, HERO, "Hero", DVec3::ZERO, true);
        *item::item_defs_mut(&mut world) = defs;
        world.mark_dirty();
        world.propagate();
        let mut r = Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
        };
        r.bridge.sync_from_world(&r.world);
        r
    }

    /// Put an ear somewhere. Without one the distant layer is silent and no
    /// round can crack, which is the honest answer to "nobody is listening" and
    /// is exactly what half these arms need to be able to turn off.
    fn listen(&mut self, at: DVec3) {
        let e = self.world.spawn_with_guid(EAR, "Ear", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(at.x, at.y, at.z);
        self.world
            .world_mut()
            .entity_mut(e)
            .insert((t, inf_ecs::components::AudioListener { active: true }));
        self.world.propagate();
    }

    fn arm(&mut self, who: Uuid, id: &str) {
        assert!(item::give_inventory(&mut self.world, who, 4));
        assert_eq!(item::give(&mut self.world, who, id, 1), 0);
        assert!(d3::gameplay::equip_weapon(&mut self.world, who, id));
    }

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

    /// One fixed step, in the hosts' own order (`wpn2a_gate::Range::step`'s
    /// shape, without the rig: nothing here needs a pose).
    fn step(&mut self) -> d3::GameplayReport {
        self.bridge.sync_from_world(&self.world);
        d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
        let report = d3::step_gameplay(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        report
    }

    fn casings(&self) -> Vec<casing::Casing> {
        casing::casing_pool(&self.world)
            .map(|p| p.casings.clone())
            .unwrap_or_default()
    }
}

fn slab(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, half: Vec3d) {
    let e = world.spawn_with_guid(guid, name, None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(at.x, at.y, at.z);
    world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: half,
            ..Default::default()
        },
        t,
    ));
}

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
        inf_ecs::components::CharacterController3D::default(),
        cm,
        t,
    ));
}

/// The rifle every arm here fires unless it says otherwise: an assault rifle,
/// no spread (a gate that cannot name where the bullet went cannot say what it
/// passed), and a real ejection port.
fn test_rifle() -> WeaponDef {
    WeaponDef {
        kind: ShotKind::Projectile,
        class: Some(WeaponClass::Ar),
        automatic: true,
        rounds_per_minute: 600.0,
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

// ── (a) FOUR LAYERS PER SHOT ────────────────────────────────────────────────

/// **A GUNSHOT IS FOUR COMMANDS, ON FOUR KEYS, IN ONE ORDER.**
///
/// Read off the shipped host's own command log after a burst through the
/// fixture's real fire path. Wave WPN1's report was ONE command on the bare
/// shooter key; this arm counts four, checks that none of the four keys is the
/// shooter's own, and pins the order.
///
/// **Mutation → red:** dropping any layer from `report_layers` (3 per shot);
/// giving two layers the same salt (2 distinct keys); reordering `ALL` (the
/// order assertion).
#[test]
fn every_loud_shot_is_four_plays_on_four_salted_keys_in_a_pinned_order() {
    let mut sim = pie_sim();
    for _ in 0..40 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    assert_eq!(item::give(sim.world_mut(), hero, "m4a1", 1), 0);
    assert!(d3::gameplay::equip_weapon(sim.world_mut(), hero, "m4a1"));
    let before = sim.audio_command_log().len();
    let mut fired = 0u32;
    // Thirty rounds at the M4A1's own rate is three seconds of trigger, not
    // two: the first draft held it for 120 steps and read twenty.
    let mut state = inf_input::InputState::new(inf_input::default_map());
    for i in 0..400 {
        let events = [inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: i < 380,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
        fired += sim.gameplay().shots;
        if fired >= 30 {
            break;
        }
    }
    assert_eq!(
        sim.dropped_audio_commands(),
        0,
        "the log evicted, so everything below is a tail rather than the stream"
    );
    assert!(fired >= 30, "the course fired {fired} rounds, not thirty");
    let plays: Vec<&inf_audio::PlayCommand> = sim.audio_command_log()[before..]
        .iter()
        .filter_map(|c| match c {
            AudioCommand::Play(p) => Some(p),
            _ => None,
        })
        .collect();
    let key = hero.as_u128() as u64;
    let want: Vec<u64> = ReportLayerKind::ALL
        .into_iter()
        .map(|k| weapon::layer_source_key(key, k))
        .collect();
    // **THE FOUR KEYS ARE FOUR**, asserted before anything is counted against
    // them. Measured on this arm's own first draft: a mutation giving two
    // layers the same salt left it GREEN, because `contains` matched every
    // command on the shared key and the order check compared `want[3]` against
    // itself. A set that is not a set makes every count below it a lie.
    let uniq: BTreeSet<u64> = want.iter().copied().collect();
    assert_eq!(
        uniq.len(),
        4,
        "two of the four layers share a voice, so one of them is inaudible"
    );
    let layered: Vec<&&inf_audio::PlayCommand> =
        plays.iter().filter(|p| want.contains(&p.source)).collect();
    println!(
        "{fired} rounds -> {} Play commands, {} of them the shooter's layers",
        plays.len(),
        layered.len()
    );
    assert_eq!(
        layered.len(),
        fired as usize * 4,
        "four layers a shot; got {} for {fired} rounds",
        layered.len()
    );
    // **The order is the contract.** The first four are the first shot's, in
    // the enum's own order, and a queue a gate never asserted an ordering on is
    // a queue two hosts can reorder.
    for (i, k) in want.iter().enumerate() {
        assert_eq!(
            layered[i].source, *k,
            "layer {i} of the first shot is on the wrong key"
        );
    }
    // **The carried collision is closed**: none of the four is the shooter's
    // own emitter key, which is what a character with an autoplay `AudioSource`
    // would have lost its voice to.
    assert!(
        !plays.iter().any(|p| p.source == key),
        "a layer is still keyed on the shooter's own emitter namespace"
    );
    // …and the four name four DIFFERENT clips, three of which did not exist
    // before this wave.
    let clips: BTreeSet<Uuid> = layered.iter().take(4).map(|p| p.clip).collect();
    assert_eq!(clips.len(), 4, "two layers are playing the same clip");
    assert!(
        clips.contains(&weapon::WEAPON_REPORT_CLIP),
        "the rifle's BODY is still wave WPN1's committed clip"
    );
}

/// **THE LOG HOLDS TWO MINUTES OF EIGHT SHOOTERS.**
///
/// The arm the re-priced ceiling owes. Eight characters at 600 rpm for 120 s is
/// 9 600 shots, 38 400 layer commands, plus a casing each and a listener
/// command a step: `dropped == 0` or every arm above it is reading a window.
///
/// # It reads the SHIPPED log, not a model of it (wave WPN2c's audit)
///
/// The first spelling built its own `BoundedLog` beside a `Range` and pushed
/// `report.shots * 4 + bounces + cracks + 1` placeholders into it per step. That
/// is the host's arithmetic re-typed, and it can only ever answer the question
/// it already assumed: a host that queued a fifth command per shot, or a
/// `SetOcclusion` per step per audible loop -- which the island's venues DO --
/// would evict inside the window and this arm would still have read zero,
/// because the thing it was counting was not the thing that fills up.
///
/// So it drives `RuntimeSim` -- the shipped host, the shipped fence, the
/// shipped `AudioCommand` log -- and reads `dropped_audio_commands()` off it.
/// The count below is what the stream ACTUALLY carried.
///
/// **Mutation → red:** `AUDIO_LOG_CAPACITY` back to `inf_core`'s 8 192 evicts
/// after about 17.8 s of this course.
#[test]
fn the_audio_log_holds_a_hundred_and_twenty_seconds_of_eight_shooters() {
    // **TWO MINUTES OF AMMUNITION.** `test_rifle` carries 300 + 300, which is a
    // minute of fire and then eight silent shooters — measured on this arm's own
    // first draft, which read 2 400 rounds where the arithmetic says 9 600. The
    // magazine is the bound the CEILING has to be priced against, so it is
    // raised here rather than the expectation lowered.
    let deep = WeaponDef {
        magazine: weapon::MAX_MAGAZINE,
        reserve: weapon::MAX_MAGAZINE,
        ..test_rifle()
    };
    let mut sim = pie_sim();
    for _ in 0..40 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    // The deep rifle joins the level's own registry rather than replacing it,
    // so the hero's `m4a1` and everything else the fixture defines still exist.
    item::item_defs_mut(sim.world_mut())
        .insert(ItemDef {
            id: "deep_rifle".into(),
            label: "deep_rifle".into(),
            stack_max: 1,
            mass_kg: 3.6,
            weapon: Some(deep),
        })
        .then_some(())
        .expect("the deep rifle is a new id");
    // Eight shooters, on the ground the fixture's hero is standing on, spread
    // along x so no two are inside each other.
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    let at = {
        let e = sim.world().entity_of(hero).expect("the fixture's hero");
        sim.world()
            .world()
            .get::<inf_ecs::components::GlobalTransform>(e)
            .map(|g| g.translation())
            .expect("a placed hero")
    };
    for i in 0..8 {
        stand(
            sim.world_mut(),
            shooter_guid(i),
            "Shooter",
            DVec3::new(at.x + i as f64 * 4.0 - 14.0, at.y - 0.9, at.z + 6.0),
            false,
        );
    }
    sim.world_mut().mark_dirty();
    sim.world_mut().propagate();
    for i in 0..8 {
        let g = shooter_guid(i);
        assert!(item::give_inventory(sim.world_mut(), g, 4));
        assert_eq!(item::give(sim.world_mut(), g, "deep_rifle", 1), 0);
        assert!(d3::gameplay::equip_weapon(sim.world_mut(), g, "deep_rifle"));
        let e = sim.world().entity_of(g).expect("a shooter");
        let mut cm = sim
            .world_mut()
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.aim_pitch_deg = 8.0;
        cm.runtime.want_attack = true;
        cm.runtime.press_attack = true;
    }
    // The per-phase clock is OFF on every shipped run and answers all zeroes
    // until it is armed -- measured on this arm's own first draft, which printed
    // `0.0000 ms` and would have held any budget ever written.
    sim.set_step_profiling(true);
    let before = sim.audio_command_log().len();
    let steps = (120.0 / DT) as usize;
    let mut shots = 0u64;
    // **THE AUDIO PHASE AT THIS WAVE'S OWN POPULATION** (the audit's (i)).
    // `AUDIO_STEP_BUDGET_MS` is 1.0 and the only arm that holds it stands in a
    // VENUE, where nobody is shooting -- so the budget had never been read at
    // the population this wave created: four `Play`s a shot and a casing, at
    // eight shooters, for two minutes.
    let mut audio_ms = 0.0f64;
    for _ in 0..steps {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        shots += u64::from(sim.gameplay().shots);
        for (name, ms) in sim.step_profile().rows() {
            if name == "audio" {
                audio_ms += ms;
            }
        }
        // The trigger is re-pressed because the host clears the edge every step;
        // `want_attack` alone is what an automatic weapon needs, and both are set
        // for the shooter that is not one.
        for i in 0..8 {
            let g = shooter_guid(i);
            if let Some(e) = sim.world().entity_of(g) {
                if let Some(mut cm) = sim.world_mut().world_mut().get_mut::<CharacterMovement>(e) {
                    cm.runtime.want_attack = true;
                }
            }
        }
    }
    let commands = sim.audio_command_log().len() - before;
    let audio_mean = audio_ms / steps as f64;
    println!(
        "120 s at eight shooters, on the shipped host: {shots} rounds, \
         {commands} commands, {} dropped (ceiling {}); the audio phase costs \
         {audio_mean:.4} ms a step (budget {})",
        sim.dropped_audio_commands(),
        inf_audio::AUDIO_LOG_CAPACITY,
        inf_player::budget::AUDIO_STEP_BUDGET_MS
    );
    // Timed only off CI and out of a debug build, on `island_gate`'s own terms:
    // a wall clock under a debug build's instrumentation is not a budget.
    assert!(
        audio_mean > 0.0,
        "the phase clock read zero, so this measurement is of nothing"
    );
    let timed = !cfg!(debug_assertions) && std::env::var_os("CI").is_none();
    assert!(
        !timed || audio_mean <= inf_player::budget::AUDIO_STEP_BUDGET_MS,
        "the audio phase costs {audio_mean:.4} ms a step at eight shooters against \
         a {} ms budget",
        inf_player::budget::AUDIO_STEP_BUDGET_MS
    );
    assert!(
        shots > 8_000,
        "only {shots} rounds in two minutes — eight shooters at 600 rpm is 9 600"
    );
    // **The stream is at least the four layers a shot**, so a host that had
    // quietly stopped queueing three of them could not pass this by evicting
    // nothing.
    assert!(
        commands as u64 >= shots * 4,
        "{commands} commands for {shots} rounds is under four layers a shot"
    );
    assert_eq!(
        sim.dropped_audio_commands(),
        0,
        "the shipped log evicted over the arm its ceiling was re-priced for"
    );
}

// ── (b) THE ENCLOSURE PROBE ─────────────────────────────────────────────────

/// **A ROOM SOUNDS LIKE A ROOM AND A STREET DOES NOT.**
///
/// The same weapon, the same shot, two places: the probe answers six hits in a
/// room and one on open ground, and the tail clip follows.
///
/// **Mutation → red:** `ENCLOSURE_INDOOR_HITS` to 1 makes the street indoors;
/// to 7 makes the room outdoors; casting the six rays at 0.5 m instead of eight
/// makes the room outdoors.
#[test]
fn the_probe_calls_a_room_a_room_and_a_street_a_street() {
    let outside = fire_one(false);
    let inside = fire_one(true);
    println!(
        "street: {} hits, indoors={}; room: {} hits, indoors={}",
        outside.0, outside.1, inside.0, inside.1
    );
    // Open ground: the floor, and nothing else.
    assert_eq!(outside.0, 1, "an open field is not a room");
    assert!(!outside.1);
    // A closed room: all six.
    assert_eq!(inside.0, 6, "a six-sided room should answer six");
    assert!(inside.1);
    // …and the TAIL follows, which is the whole point of the probe.
    let street = weapon::report_layers(WeaponClass::Ar, outside.1, 320.0, 10.0, 0);
    let room = weapon::report_layers(WeaponClass::Ar, inside.1, 320.0, 10.0, 0);
    assert_eq!(
        street[2].source.clip,
        Some(weapon::report_clip(
            WeaponClass::Ar,
            ReportClip::OutdoorTail
        ))
    );
    assert_eq!(
        room[2].source.clip,
        Some(weapon::report_clip(WeaponClass::Ar, ReportClip::IndoorTail))
    );
    assert_ne!(street[2].source.clip, room[2].source.clip);
    assert_eq!(room[2].lowpass_hz, Some(weapon::INDOOR_TAIL_LOWPASS_HZ));
    assert_eq!(street[2].lowpass_hz, None);
}

/// The probe's own hit count at the hero's muzzle, and the verdict the shot
/// carried.
fn fire_one(indoors: bool) -> (u8, bool) {
    let mut r = Range::new(defs_with("rifle", test_rifle()), indoors);
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    let mut verdict = None;
    for _ in 0..8 {
        let report = r.step();
        if let Some(h) = report.hits.iter().find(|h| h.loud) {
            verdict = Some(h.indoors);
            break;
        }
    }
    // **THE SHOOTER HAS TO BE EXCLUDED**, and this arm's own first draft did
    // not: a probe cast from inside the hero's own capsule answers SIX hits on
    // an open field, because a ray that starts inside a solid collider hits it
    // immediately in every direction. The shipped path never had the defect
    // (`resolve_shot` passes `shot_exclusions`), and the verdict this arm reads
    // beside the count is the shipped one — which is exactly why the count had
    // to be measured too, rather than trusted.
    let muzzle = DVec3::new(0.0, 1.4, 0.0);
    let mut exclude = BTreeSet::new();
    if let Some(c) = r.bridge.collider_of(HERO) {
        exclude.insert(c);
    }
    let probe = d3::audio::enclosure_at(r.bridge.world_mut(), muzzle, &exclude);
    (probe.hits, verdict.expect("the hero fired"))
}

/// **THE PROBE'S RAYS ARE ON THE BILL.**
///
/// Six casts a loud shot, counted, and the pool's spawn refusal is priced
/// against seven a shot rather than one. Before this wave a shot cost one ray
/// and the ceiling was a claim about a sixth of the real number.
///
/// **Mutation → red:** dropping `report.rounds.probe_rays +=` (zero); pricing
/// `rays_already` at `report.shots` alone (the refusal threshold moves and the
/// printed ratio is 1 rather than 7).
#[test]
fn the_enclosure_probe_pays_for_its_own_rays() {
    let mut r = Range::new(defs_with("rifle", test_rifle()), true);
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    let (mut shots, mut probe_rays, mut indoor) = (0u32, 0u32, 0u32);
    for _ in 0..60 {
        let report = r.step();
        shots += report.shots;
        probe_rays += report.rounds.probe_rays;
        indoor += report.rounds.indoor_shots;
    }
    println!("{shots} shots cost {probe_rays} probe rays and {indoor} were indoors");
    assert!(shots > 0, "nothing fired");
    assert_eq!(
        probe_rays,
        shots * d3::audio::ENCLOSURE_PROBE_RAYS as u32,
        "the probe is not being counted"
    );
    assert_eq!(indoor, shots, "every shot in a closed room is indoors");
    // …and the ceiling knows: one shot's whole bill is seven, not one.
    assert_eq!(d3::audio::ENCLOSURE_PROBE_RAYS, 6);
    assert!(
        ballistics::MAX_SHOT_RAYS_PER_STEP
            > d3::audio::ENCLOSURE_PROBE_RAYS * ballistics::PROJECTILE_SUB_STEPS as usize,
        "the ceiling cannot afford one shot"
    );
}

// ── (c) THE CLIPS ───────────────────────────────────────────────────────────

/// **THE COMMITTED CLIPS ARE THE ONES THE ENGINE NAMES.**
///
/// Thirty-six GUIDs, thirty-six files, and every one of them decodes at the
/// rate and the length the generator says. A `Play` naming a clip nothing
/// resolves is silence with no error, which is exactly the failure this arm
/// exists to make loud.
///
/// **Mutation → red:** dropping a class from `weapon_audio_clips` (the file is
/// missing); changing `REPORT_CLIP_BASE` (every GUID misses).
#[test]
fn every_clip_the_engine_names_is_a_file_that_decodes() {
    let dir = inf_editor_core::samples::weapon_audio_dir();
    if !dir
        .join(inf_editor_core::weapon_audio::CASING_FILE)
        .exists()
    {
        eprintln!("SKIP: the gunshot library has not been blessed yet");
        return;
    }
    // Every `.inf_audio` under `samples/`, by the GUID its sidecar carries.
    let mut by_guid: BTreeMap<Uuid, PathBuf> = BTreeMap::new();
    for root in [dir.clone(), inf_editor_core::samples::gameplay_dir()] {
        for entry in std::fs::read_dir(&root).expect("the folder is there") {
            let path = entry.expect("an entry").path();
            if path.extension().is_some_and(|e| e == "inf_audio") {
                if let Ok(side) = inf_asset::AssetSidecar::load(&path) {
                    by_guid.insert(side.guid.0, path);
                }
            }
        }
    }
    println!(
        "{:8} {:12} {:>8} {:>6} {:>8}",
        "class", "clip", "bytes", "rate", "seconds"
    );
    let mut n = 0usize;
    for class in WeaponClass::ALL {
        for clip in ReportClip::ALL {
            let guid = weapon::report_clip(class, clip);
            let path = by_guid
                .get(&guid)
                .unwrap_or_else(|| panic!("{} {} names {guid}, and no committed file has that id \u{2014} every Play of it is silent", class.name(), clip.file_stem()));
            let raw = std::fs::read(path).expect("readable");
            let asset: inf_audio::AudioAsset = inf_asset::decode(&raw).expect("it decodes");
            let sound = asset.decode().expect("kira decodes it");
            println!(
                "{:8} {:12} {:8} {:6} {:8.4}",
                class.name(),
                clip.file_stem(),
                asset.bytes.len(),
                sound.sample_rate(),
                sound.duration_secs()
            );
            assert_eq!(sound.sample_rate(), inf_audio::synth::SYNTH_RATE);
            // The brief's own envelopes, read off the committed file rather
            // than off the generator.
            let secs = sound.duration_secs();
            match clip {
                ReportClip::Transient => assert!(secs <= 0.0055, "a transient of {secs} s"),
                ReportClip::IndoorTail => {
                    assert!((secs - inf_audio::synth::INDOOR_TAIL_S).abs() < 0.01)
                }
                ReportClip::OutdoorTail => {
                    assert!((secs - inf_audio::synth::OUTDOOR_TAIL_S).abs() < 0.01)
                }
                ReportClip::Crack => assert!((secs - inf_audio::synth::CRACK_S).abs() < 0.01),
                ReportClip::Body => assert!((0.19..=0.56).contains(&secs), "a body of {secs} s"),
            }
            n += 1;
        }
    }
    assert_eq!(n, 35);
    assert!(by_guid.contains_key(&weapon::CASING_CLIP), "no brass clip");
}

/// **THE TWO CRATES AGREE ABOUT WHAT A CLASS IS.**
///
/// `inf-audio` sits beneath `inf-ecs` and cannot name a [`WeaponClass`], so the
/// contract between the generator and the engine is an INDEX. This is the arm
/// that stops the two lists drifting — a reordered `WeaponClass::ALL` would
/// give every gun somebody else's voice and nothing would fail to compile.
#[test]
fn the_generator_and_the_engine_number_the_classes_the_same_way() {
    for class in WeaponClass::ALL {
        assert_eq!(
            inf_audio::synth::CLASS_NAMES[class.index() as usize],
            class.name(),
            "class {} is index {} to the engine and {:?} to the generator",
            class.name(),
            class.index(),
            inf_audio::synth::CLASS_NAMES[class.index() as usize]
        );
    }
    for clip in ReportClip::ALL {
        assert_eq!(
            inf_audio::synth::CLIP_NAMES[clip.index() as usize],
            clip.file_stem()
        );
    }
    assert_eq!(inf_audio::synth::CLASS_NAMES.len(), WeaponClass::ALL.len());
    assert_eq!(inf_audio::synth::CLIP_NAMES.len(), ReportClip::ALL.len());
}

// ── (d) THE LOW-PASS IS AUDIBLE ─────────────────────────────────────────────

/// **THE CUTOFFS REACH THE COMMAND STREAM.**
///
/// CI cannot hear a filter, so the contract is the command: a distant layer and
/// an indoor tail carry a `lowpass_hz` and the other layers do not. The filter
/// ITSELF is proven by its impulse response in `inf_audio::filter`'s own tests,
/// which measure a DFT of a real impulse pushed through a real filter.
///
/// **Mutation → red:** dropping `cmd.lowpass_hz = layer.lowpass_hz` from the
/// fence (both cutoffs vanish from the stream).
#[test]
fn the_report_carries_its_cutoffs_all_the_way_to_the_command() {
    // The Ring-0 proof, restated here so the gate names the number: a one-pole
    // at its own cutoff is down 3.01 dB.
    let f = inf_audio::OnePole::new(weapon::DISTANT_LOWPASS_HZ, inf_audio::synth::SYNTH_RATE);
    let at = f.response_at(weapon::DISTANT_LOWPASS_HZ, inf_audio::synth::SYNTH_RATE);
    let db = 20.0 * at.log10();
    println!(
        "{} Hz one-pole: {db:.3} dB at its own cutoff",
        weapon::DISTANT_LOWPASS_HZ
    );
    assert!((db + 3.0103).abs() < 0.01);
    // …and eight octaves up it is gone.
    let up = f.response_at(
        weapon::DISTANT_LOWPASS_HZ * 8.0,
        inf_audio::synth::SYNTH_RATE,
    );
    assert!(up < at / 4.0, "the filter is not filtering");

    let inside = weapon::report_layers(WeaponClass::Ar, true, 320.0, 400.0, 0);
    let cutoffs: Vec<Option<f64>> = inside.iter().map(|l| l.lowpass_hz).collect();
    println!("indoors, far away: {cutoffs:?}");
    assert_eq!(
        cutoffs,
        vec![
            None,
            None,
            Some(weapon::INDOOR_TAIL_LOWPASS_HZ),
            Some(weapon::DISTANT_LOWPASS_HZ)
        ]
    );
    let outside = weapon::report_layers(WeaponClass::Ar, false, 320.0, 10.0, 0);
    assert_eq!(outside[2].lowpass_hz, None, "a street does not muffle");
    assert_eq!(
        outside[3].lowpass_hz,
        Some(weapon::DISTANT_LOWPASS_HZ),
        "air is a low-pass whether or not you are inside"
    );
}

/// **THE DISTANT LAYER IS AUDIBLE WHERE IT IS LOUDEST** — read off the shipped
/// command's own `Attenuation`, not off the layer that produced it.
///
/// # What this caught (wave WPN2c's audit)
///
/// The distant layer shipped with the BODY's `max_distance` — the weapon's own
/// `report_max_m`, 320 m for an M4A1 — and `Attenuation::gain` is **zero at and
/// past `max_distance`**. So the layer that exists to be the thing you hear a
/// long way off reached full command volume at `DISTANT_FULL_M` (300 m) and was
/// culled at 320; it was silent at the four hundred metres its own constant's
/// doc names as the case it is for; and `the_distant_layer_is_a_function_of_
/// where_the_listener_is` asserted its `volume` at 600 m on a 600 m sniper —
/// a command the engine plays at gain zero. Every one of those assertions is
/// about the COMMAND. This one is about the sound.
///
/// It reads the `Attenuation` **out of the queued `PlayCommand`**, so it owes
/// nothing to a re-derivation of how a host maps an `AudioSource` onto one.
///
/// **Mutation → red:** the distant layer back to `max` (its gain at 400 m is
/// 0.0); `DISTANT_REACH_MULT` to 1.0 (the same).
#[test]
fn the_distant_layer_is_audible_at_the_range_it_is_written_for() {
    let mut sim = pie_sim();
    for _ in 0..40 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    assert_eq!(item::give(sim.world_mut(), hero, "m4a1", 1), 0);
    assert!(d3::gameplay::equip_weapon(sim.world_mut(), hero, "m4a1"));
    let before = sim.audio_command_log().len();
    let mut state = inf_input::InputState::new(inf_input::default_map());
    for i in 0..60 {
        let events = [inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: i < 3,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
    }
    let key = |k| weapon::layer_source_key(hero.as_u128() as u64, k);
    let plays: Vec<&inf_audio::PlayCommand> = sim.audio_command_log()[before..]
        .iter()
        .filter_map(|c| match c {
            AudioCommand::Play(p) => Some(p),
            _ => None,
        })
        .collect();
    let of = |k| {
        *plays
            .iter()
            .find(|p| p.source == key(k))
            .unwrap_or_else(|| panic!("no {k:?} layer in the stream"))
    };
    let (body, distant) = (of(ReportLayerKind::Body), of(ReportLayerKind::Distant));
    println!(
        "the m4a1's body reaches {:.0} m, its distant layer {:.0} m; \
         spatial gain at 300 / 400 / 1000 m: body {:.4} / {:.4} / {:.4}, \
         distant {:.4} / {:.4} / {:.4}",
        body.attenuation.max_distance,
        distant.attenuation.max_distance,
        body.attenuation.gain(300.0),
        body.attenuation.gain(400.0),
        body.attenuation.gain(1000.0),
        distant.attenuation.gain(300.0),
        distant.attenuation.gain(400.0),
        distant.attenuation.gain(1000.0),
    );
    // The layer's ramp reaches full at 300 m, so it must still be audible there
    // and past it -- and at the 400 m its own cutoff constant is written for.
    assert!(
        distant.attenuation.gain(weapon::DISTANT_FULL_M) > 0.0,
        "the distant layer is culled at the range its own ramp reaches full"
    );
    assert!(
        distant.attenuation.gain(400.0) > 0.0,
        "a rifle at four hundred metres is silent, which is the one case this \
         layer exists for"
    );
    // …and it out-reaches the gun's own report, which the BODY does not.
    assert!(distant.attenuation.max_distance > body.attenuation.max_distance);
    assert_eq!(
        body.attenuation.gain(400.0),
        0.0,
        "the body layer is supposed to be gone by 400 m -- if it is not, the \
         distant layer is not carrying anything the body was not"
    );
}

// ── (e) THE SUPERSONIC CRACK ────────────────────────────────────────────────

/// **A ROUND CRACKS PAST YOUR HEAD AND NOT PAST THE NEXT STREET.**
///
/// The doc's own rule, on the SEGMENT: two metres cracks, ten does not, and a
/// subsonic round never does however close it passes.
///
/// **Mutation → red:** `SPEED_OF_SOUND_MPS` to 100 (the AS VAL cracks);
/// `CRACK_RADIUS_M` to 40 (the ten-metre pass cracks); testing the round's
/// POINT rather than its segment (nothing cracks at all — at 900 m/s a round
/// jumps 3.75 m a sub-step and never lands inside four metres of the ear).
#[test]
fn a_supersonic_round_cracks_within_four_metres_and_a_subsonic_one_never_does() {
    // **BOTH MUZZLE SPEEDS ARE THE REGISTRY'S OWN** (the wave's audit).
    // `fc103339` made this point about the AS VAL -- "an arm that names a weapon
    // and then invents its muzzle speed proves something about a number nobody
    // ships" -- and left an invented `900.0` for the M4A1 in the two rows above
    // it; `weapons.toml` says **910**. The ten metres a second change none of
    // the verdicts here, which is exactly why nothing caught it, and that is the
    // argument for reading the file rather than agreeing with it.
    let mut registry = ItemDefs::default();
    registry
        .merge_toml(weapon::WEAPON_REGISTRY_TOML)
        .expect("the shipped weapon registry parses");
    let v0_of = |id: &str| -> f64 {
        registry
            .get(id)
            .and_then(|d| d.weapon.as_ref())
            .map(|w| w.muzzle_speed_mps)
            .unwrap_or_else(|| panic!("{id} is not in the shipped registry"))
    };
    let (m4a1, as_val) = (v0_of("m4a1"), v0_of("as_val"));
    // The registry has weapons on both sides of the speed of sound ON PURPOSE,
    // and that is the premise this arm rests on rather than an accident of two
    // rows: an AS VAL authored 1 m/s faster would make the third case vacuous.
    assert!(
        m4a1 > SPEED_OF_SOUND_MPS && as_val < SPEED_OF_SOUND_MPS,
        "the registry's M4A1 ({m4a1} m/s) and AS VAL ({as_val} m/s) are no longer \
         on opposite sides of {SPEED_OF_SOUND_MPS} m/s"
    );
    println!("{:14} {:>6} {:>6} {:>9}", "weapon", "v0", "miss", "cracks");
    for (name, v0, miss, want) in [
        ("m4a1", m4a1, 2.0, true),
        ("m4a1", m4a1, 10.0, false),
        // The AS VAL is deliberately subsonic; it is silent by physics.
        // 330 m/s is the REGISTRY's own number for it
        // (`weapons.toml`, `[as_val.weapon] muzzle_speed_mps`), not a
        // convenient one. It is 13 m/s under the speed of sound, which is the
        // margin the row was authored with and the margin this arm is entitled
        // to assert.
        ("as_val", as_val, 2.0, false),
    ] {
        let mut def = test_rifle();
        def.muzzle_speed_mps = v0;
        def.gravity_scale = 0.0;
        def.hitscan_threshold_m = 5.0;
        let mut r = Range::new(defs_with("rifle", def), false);
        // The ear is `miss` metres off the flight line, 40 m down range — past
        // the threshold, so a ROUND is what goes by rather than a ray.
        r.listen(DVec3::new(miss, 1.4, 40.0));
        r.arm(HERO, "rifle");
        r.aim(HERO, 0.0, 0.0);
        r.hold_trigger(HERO, true);
        let mut cracks: Vec<ballistics::Crack> = Vec::new();
        for i in 0..30 {
            // **ONE ROUND.** The trigger is released after the first step so
            // the "one crack" assertion below is about a round rather than
            // about a burst — measured on this arm's own first draft, which held
            // the trigger, fired four rounds and read four cracks.
            if i == 1 {
                r.hold_trigger(HERO, false);
            }
            let report = r.step();
            cracks.extend(report.cracks.iter().copied());
        }
        println!(
            "{name:14} {v0:6.0} {miss:6.1} {:9}",
            if cracks.is_empty() {
                "no".to_string()
            } else {
                format!("yes ({:.2} m)", cracks[0].miss_m)
            }
        );
        assert_eq!(
            !cracks.is_empty(),
            want,
            "{name} at {v0} m/s passing {miss} m away"
        );
        if want {
            let c = cracks[0];
            assert!(c.miss_m <= CRACK_RADIUS_M);
            assert!(c.speed_mps > SPEED_OF_SOUND_MPS);
            // …at the CLOSEST POINT, which is beside the ear rather than at the
            // muzzle it left forty metres back.
            assert!(
                (c.at.z - 40.0).abs() < 4.0,
                "the crack was placed at {:?}, not beside the ear",
                c.at
            );
            // ONE crack per round, however many sub-steps it spends near.
            assert_eq!(cracks.len(), 1, "a round cracked more than once");
        }
    }
}

/// The point-to-segment door itself, which is what makes the arm above
/// possible: a round that jumps fifteen metres a step is only ever tested as a
/// line.
#[test]
fn the_crack_measures_a_segment_and_not_a_point() {
    let a = DVec3::new(0.0, 0.0, 0.0);
    let b = DVec3::new(0.0, 0.0, 30.0);
    let ear = DVec3::new(2.0, 0.0, 15.0);
    let (d, at) = ballistics::point_to_segment_m(ear, a, b);
    assert!((d - 2.0).abs() < 1e-12);
    assert!((at.z - 15.0).abs() < 1e-12);
    // Past either end it clamps rather than extrapolating.
    let (d, at) = ballistics::point_to_segment_m(DVec3::new(0.0, 0.0, 50.0), a, b);
    assert!((d - 20.0).abs() < 1e-12);
    assert!((at.z - 30.0).abs() < 1e-12);
    // A degenerate segment answers the distance to where it is.
    let (d, _) = ballistics::point_to_segment_m(DVec3::new(3.0, 0.0, 0.0), a, a);
    assert!((d - 3.0).abs() < 1e-12);
    // …and the POINT test the doc warns about would have missed: the round's
    // own endpoints are both eight metres from the ear.
    assert!((ear - a).length() > CRACK_RADIUS_M);
    assert!((ear - b).length() > CRACK_RADIUS_M);
}

// ── (f) THE CASING POOL ─────────────────────────────────────────────────────

/// **BRASS FALLS, BOUNCES ONCE, MAKES ONE NOISE, AND STOPS.**
///
/// **Mutation → red:** emitting a bounce on every contact (two sounds a
/// casing); `CASING_SETTLE_CONTACTS` to 99 (nothing ever settles); dropping the
/// eject from the fire path (nothing exists).
#[test]
fn a_casing_is_ejected_bounces_once_and_settles() {
    let mut r = Range::new(defs_with("rifle", test_rifle()), false);
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    // One round only: a single case is what a settle can be watched on.
    let mut ejected = 0u32;
    let mut bounces: Vec<d3::gameplay::CasingBounce> = Vec::new();
    for i in 0..180 {
        if i == 2 {
            r.hold_trigger(HERO, false);
        }
        let report = r.step();
        ejected += report.casings.ejected;
        bounces.extend(report.casings.bounces.iter().copied());
    }
    let live = r.casings();
    println!(
        "{ejected} ejected, {} bounce sounds, {} live, contacts {:?}",
        bounces.len(),
        live.len(),
        live.iter().map(|c| c.contacts).collect::<Vec<_>>()
    );
    assert!(ejected > 0, "the fire path ejected nothing");
    assert_eq!(
        bounces.len(),
        ejected as usize,
        "one landing sound per casing, on its FIRST contact"
    );
    assert!(!live.is_empty(), "the brass vanished before it settled");
    for c in &live {
        assert!(c.settled(), "a casing is still moving after three seconds");
        assert_eq!(c.velocity, DVec3::ZERO);
        // …and it is ON the floor rather than through it.
        assert!(c.at.y > -0.2 && c.at.y < 2.0, "a casing at {:?}", c.at);
    }
    // Every landing carries its own note, and they are not the same note.
    let pitches: BTreeSet<u64> = bounces.iter().map(|b| b.pitch.to_bits()).collect();
    assert_eq!(pitches.len(), bounces.len(), "two casings share a pitch");
    for b in &bounces {
        assert!((b.pitch - 1.0).abs() <= casing::CASING_PITCH_SPREAD + 1e-12);
    }
}

/// **THE RING IS A RING**, and the pool's bytes are empty when it is.
///
/// **Mutation → red:** an unbounded `Vec` (the count runs away); folding the
/// counters into `casing_state_bytes` (the quiet trace moves).
#[test]
fn the_casing_pool_is_bounded_and_folds_nothing_when_it_is_empty() {
    let mut r = Range::new(defs_with("rifle", test_rifle()), false);
    assert!(casing::casing_state_bytes(&r.world).is_empty());
    // **THREE SHOOTERS, BECAUSE ONE CANNOT FILL THE RING.** Measured on this
    // arm's own first draft: one rifle at 600 rpm ejects 8.57 cases a second
    // and a case lives eight, so the steady state is **69** of a hundred and
    // twenty-eight and the ring never wraps however long the arm runs. The
    // ceiling is a two-shooter fact, so the arm that proves it is a firefight.
    for i in 0..3 {
        stand(
            &mut r.world,
            shooter_guid(i),
            "Shooter",
            DVec3::new(i as f64 * 4.0 - 6.0, 0.0, 0.0),
            false,
        );
    }
    r.world.propagate();
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    for i in 0..3 {
        let g = shooter_guid(i);
        r.arm(g, "rifle");
        r.aim(g, 0.0, 0.0);
        r.hold_trigger(g, true);
    }
    // Twenty seconds: four rifles eject 34 cases a second and a case lives
    // eight, so the ring fills in under four and wraps for the rest.
    let mut peak = 0usize;
    for _ in 0..1_200 {
        r.step();
        peak = peak.max(casing::casings_live(&r.world));
    }
    let pool = casing::casing_pool(&r.world).expect("a pool");
    println!(
        "{} ejected, {} recycled, peak {peak} live (ceiling {MAX_CASINGS_LIVE})",
        pool.spawned, pool.recycled
    );
    assert!(
        pool.spawned > MAX_CASINGS_LIVE as u64,
        "the ring never wrapped"
    );
    assert!(peak <= MAX_CASINGS_LIVE, "the ring held {peak}");
    assert!(
        pool.recycled > 0,
        "nothing was recycled, so nothing wrapped"
    );
    // The bytes are the live casings and nothing else.
    assert_eq!(
        casing::casing_state_bytes(&r.world).len(),
        casing::casings_live(&r.world) * casing::CASING_TRACE_BYTES
    );
    // …and a world that has fired and let everything age out folds nothing.
    casing::clear_casings(&mut r.world);
    assert!(casing::casing_state_bytes(&r.world).is_empty());
}

/// **THE BRASS IS DRAWN**, one entity per live casing, on a derived guid.
///
/// **Mutation → red:** dropping `step_casing_entities` (no entity ever exists);
/// never despawning (the count grows past the pool's).
#[test]
fn every_live_casing_is_an_entity_and_every_dead_one_is_not() {
    let mut r = Range::new(defs_with("rifle", test_rifle()), false);
    assert!(casing::drawn_casings(&r.world).is_empty());
    r.arm(HERO, "rifle");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    for _ in 0..30 {
        r.step();
    }
    r.hold_trigger(HERO, false);
    let live = r.casings();
    let drawn = casing::drawn_casings(&r.world);
    println!("{} casings, {} entities", live.len(), drawn.len());
    assert!(!live.is_empty());
    assert_eq!(drawn.len(), live.len(), "the draw and the pool disagree");
    for c in &live {
        let g = casing::casing_guid(c.shooter, c.seq);
        assert!(drawn.contains(&g), "casing {} has no entity", c.seq);
        let e = r.world.entity_of(g).expect("the entity");
        let t = r
            .world
            .world()
            .get::<Transform>(e)
            .expect("a transform")
            .translation;
        assert!(
            (t.to_dvec3() - c.at).length() < 1e-9,
            "the drawn casing is not where the pool says"
        );
        assert!(r
            .world
            .world()
            .get::<inf_ecs::components::MeshRef>(e)
            .is_some());
    }
    // Let them age out; the entities go with them.
    for _ in 0..(casing::CASING_LIFETIME_S / DT) as usize + 30 {
        r.step();
    }
    assert_eq!(
        casing::casings_live(&r.world),
        0,
        "the brass never aged out"
    );
    assert!(
        casing::drawn_casings(&r.world).is_empty(),
        "the entities outlived the pool"
    );
}

/// **THE ROOM THE LAST SHOT WAS FIRED IN IS REMEMBERED IN THE SIM**, which is
/// what `hero.csv`'s tail column reads.
///
/// The latch lives on the casing pool rather than in the host, and the reason
/// is measured: the first draft latched it in `HeroLog::tick` off
/// `GameplayReport::shots`, and the island's own eight-round burst wrote `-` in
/// every row — a frame may run more than one fixed step, and the report is
/// replaced by each of them.
///
/// **Mutation → red:** dropping the `note_shot_room` call (the latch stays
/// `None` through a burst).
#[test]
fn the_sim_remembers_which_room_the_last_shot_was_fired_in() {
    for (indoors, want) in [(false, false), (true, true)] {
        let mut r = Range::new(defs_with("rifle", test_rifle()), indoors);
        assert_eq!(casing::last_shot_indoors(&r.world), None);
        r.arm(HERO, "rifle");
        r.aim(HERO, 0.0, 0.0);
        r.hold_trigger(HERO, true);
        let mut shots = 0u32;
        for _ in 0..30 {
            shots += r.step().shots;
        }
        assert!(shots > 0, "nothing fired");
        assert_eq!(
            casing::last_shot_indoors(&r.world),
            Some(want),
            "{shots} shots {} and the pool remembers otherwise",
            if indoors { "in a room" } else { "in the open" }
        );
    }
    // A punch is not a loud shot, so it does not move the latch: a hero that
    // fired outdoors and then threw a punch indoors still reads `outdoor`.
    let mut r = Range::new(ItemDefs::default(), true);
    r.hold_trigger(HERO, true);
    for _ in 0..30 {
        r.step();
    }
    assert_eq!(
        casing::last_shot_indoors(&r.world),
        None,
        "a punch moved the tail latch"
    );
}

// ── (h) DETERMINISM ─────────────────────────────────────────────────────────

/// **PIE == SHIPPING, AND TWO COOKS AGREE, OVER A COURSE THAT MAKES EVERY
/// NOISE THIS WAVE ADDS.**
///
/// The trace compared is `state_bytes`, which now carries the brass at its
/// tail; the COMMAND STREAM is compared beside it, command for command, because
/// a gunshot is not in the world and the trace cannot see it.
#[test]
fn pie_equals_shipping_and_two_cooks_agree_over_a_sound_course() {
    let a = tempfile::tempdir().expect("tempdir a");
    let b = tempfile::tempdir().expect("tempdir b");
    let pack_a = cook_fixture(a.path());
    let pack_b = cook_fixture(b.path());
    let ta = sound_course(pack_sim(&pack_a));
    let tb = sound_course(pack_sim(&pack_b));
    let tp = sound_course(pie_sim());
    println!(
        "the sound course: {} steps, {} audio commands, {} casings at the end",
        ta.trace.len(),
        ta.audio.len(),
        ta.casings
    );
    assert!(ta.audio.len() > 100, "the course made almost no noise");
    assert!(ta.casings > 0, "the course ejected no brass");
    // **THE FENCES ARE NOT VACUOUS ON THIS COURSE.** Every one of the four
    // layers is in the stream, and so is a casing landing — otherwise the three
    // comparisons below are agreeing about a stream with nothing in it.
    for (clip, what) in [
        (
            weapon::report_clip(WeaponClass::Ar, ReportClip::Transient),
            "the transient",
        ),
        (weapon::WEAPON_REPORT_CLIP, "the body"),
        (
            weapon::report_clip(WeaponClass::Ar, ReportClip::Crack),
            "the distant crack",
        ),
        (weapon::CASING_CLIP, "a casing landing"),
    ] {
        let needle = format!("{clip}");
        assert!(
            ta.audio.iter().any(|c| c.contains(&needle)),
            "{what} is not in the stream, so the arms below compare a stream that does not contain it"
        );
    }
    // **THE FIXTURE IS A ROOM**, measured rather than assumed: the gameplay
    // level grows a PCG house around the spot the hero stands on, so the
    // enclosure probe calls every shot on it INDOORS and the tail in this
    // stream is the short one. That is the probe working on committed content
    // rather than on a fixture built to make it work, and the two tails are
    // named here so the day the house moves the arm says which way.
    let indoor = format!(
        "{}",
        weapon::report_clip(WeaponClass::Ar, ReportClip::IndoorTail)
    );
    let outdoor = format!(
        "{}",
        weapon::report_clip(WeaponClass::Ar, ReportClip::OutdoorTail)
    );
    let saw_indoor = ta.audio.iter().any(|c| c.contains(&indoor));
    let saw_outdoor = ta.audio.iter().any(|c| c.contains(&outdoor));
    println!("the fixture's tail: indoor={saw_indoor} outdoor={saw_outdoor}");
    assert!(
        saw_indoor != saw_outdoor,
        "the course played both tails or neither ({saw_indoor}/{saw_outdoor})"
    );
    assert!(
        saw_indoor,
        "the gameplay fixture's hero stands inside its own PCG house, so the probe should call it indoors"
    );
    for (i, (x, y)) in ta.trace.iter().zip(tb.trace.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: two independent cooks diverged");
    }
    for (i, (x, y)) in ta.trace.iter().zip(tp.trace.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: PIE and shipping diverged");
    }
    assert_eq!(ta.audio, tb.audio, "two cooks made different noises");
    assert_eq!(ta.audio, tp.audio, "PIE and shipping made different noises");
    assert_eq!(ta.casings, tp.casings);
}

/// **THE COOKED PACK CARRIES THE CLIPS NO ENTITY REFERENCES** -- island carried
/// 43's closure, read off the PACK.
///
/// # Why this arm exists (wave WPN2c's audit)
///
/// The wave closed carried 43 by teaching `cook::asset_deps` and
/// `pie::build_scene_payload` to read `inf_ecs::audio::engine_spawned_clips`,
/// and **nothing falsified it**. Measured: deleting the whole `deps.extend(...
/// engine_spawned_clips ...)` block from the cook left all seventeen arms of
/// this gate GREEN, because every other arm here reads the COMMAND STREAM --
/// and the command stream is identical whether or not the clip it names is in
/// the pack. That is carried 43's own defect exactly: *"a `Play` whose clip does
/// not resolve is silence with no error"*, which is why the venue's music was
/// silent in every shipped build for a whole island wave and nobody heard it.
///
/// So this one reads the pack's asset index instead of the sim's queue, and it
/// is the only arm in the tree that does.
///
/// # The fixture is scaffolded WITH the library
///
/// `scaffold` copies the `phase30-gameplay` folder, which holds exactly one
/// `.inf_audio` -- wave WPN1's rifle body. The other thirty-five live in
/// `samples/weapon-audio/`, which is engine content a project opts into, so the
/// arm copies them in: a closure that cannot be seen to pull thirty-six files
/// proves less than one that can.
///
/// **Mutation -> red:** the `engine_spawned_clips` extension deleted from
/// `asset_deps` (the pack then carries only what the document names).
#[test]
fn a_cooked_pack_carries_every_clip_the_engine_names() {
    let lib = inf_editor_core::samples::weapon_audio_dir();
    if !lib
        .join(inf_editor_core::weapon_audio::CASING_FILE)
        .exists()
    {
        eprintln!("SKIP: the gunshot library has not been blessed yet");
        return;
    }
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = scaffold(tmp.path());
    let content = proj.join("Content");
    for entry in std::fs::read_dir(&lib).expect("the library folder") {
        let path = entry.expect("an entry").path();
        let Some(name) = path.file_name() else {
            continue;
        };
        std::fs::copy(&path, content.join(name)).expect("copy");
    }
    let out = tmp.path().join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default()).expect("it cooks");

    // THE PACK's own index, not the source tree's.
    let source = inf_player::level::PackLevelSource::open(&out).expect("the pack opens");
    let carried = source.audio_assets().expect("the pack's audio index reads");

    // Every clip the ENGINE names, and whether the pack has it.
    let mut want: Vec<(String, Uuid)> = Vec::new();
    for class in WeaponClass::ALL {
        for clip in ReportClip::ALL {
            want.push((
                format!("{} {}", class.name(), clip.file_stem()),
                weapon::report_clip(class, clip),
            ));
        }
    }
    want.push(("the casing".into(), weapon::CASING_CLIP));
    println!(
        "the cooked pack carries {} .inf_audio entries",
        carried.len()
    );
    let mut missing: Vec<String> = Vec::new();
    for (what, guid) in &want {
        match carried.get(guid) {
            Some(a) => println!("  {what:22} {guid} {:6} bytes", a.bytes.len()),
            None => missing.push(format!("{what} ({guid})")),
        }
    }
    assert!(
        missing.is_empty(),
        "the cook closed over a level that plays these and the pack does not carry them: {missing:?}"
    );
    assert_eq!(
        want.len(),
        36,
        "thirty-five report clips and the brass; the list moved"
    );
    // …and the bytes in the pack are the bytes that decode, so "carried" is not
    // "carried as something the player cannot play".
    for (what, guid) in &want {
        let a = carried.get(guid).expect("checked above");
        let sound = a.decode().unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(sound.sample_rate(), inf_audio::synth::SYNTH_RATE);
    }
    // **THE VENUE LOOP IS THE ONE THE ENGINE NAMES AND THIS PROJECT DOES NOT
    // HAVE**, and its absence is the arm's own control: the closure pulls what
    // the project HOLDS and invents nothing. It rides the island's pack, where
    // its file is (`samples/settlement/Venue_Music.inf_audio`, named in both
    // island recipes), and `island_gate` is where that is asserted.
    assert!(
        !carried.contains_key(&inf_ecs::venue::VENUE_MUSIC_CLIP),
        "the fixture project has no venue music file, so a pack carrying one \
         means the closure is inventing assets"
    );
}

/// **THE BRASS REACHES THE RENDERER**, read off the shipped projector's own
/// output rather than off the ECS.
///
/// # Why the entity arm is not this arm (wave WPN2c's audit)
///
/// `every_live_casing_is_an_entity_and_every_dead_one_is_not` proves the pool
/// and the WORLD agree: one entity per casing, at the pool's own position, with
/// a `MeshRef` on it. It cannot prove the entity is DRAWN — the shipped player
/// projects with `render::project_scene`, and between an entity and an instance
/// there is a `ComputedVisibility` read, a `GlobalTransform` read that falls
/// back to the ORIGIN when it is missing, and a branch per component kind. The
/// audit went looking for brass in the pixels of an island session with nine
/// casings live and could not find any, which is a question the ECS cannot
/// answer either way.
///
/// So this drives the same `project_scene` a dozen gates drive, and matches the
/// projected instances against the pool by POSITION. It is the arm that would
/// have caught a projector that dropped them or drew them at the world origin.
///
/// **Mutation → red:** `step_casing_entities` returning early (no entity, so no
/// instance); a `MeshRef` the projector has no branch for.
#[test]
fn every_live_casing_is_an_instance_the_renderer_draws() {
    let mut sim = pie_sim();
    for _ in 0..40 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    assert_eq!(item::give(sim.world_mut(), hero, "m4a1", 1), 0);
    assert!(d3::gameplay::equip_weapon(sim.world_mut(), hero, "m4a1"));
    let mut state = inf_input::InputState::new(inf_input::default_map());
    for i in 0..120 {
        let events = [inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: i < 60,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
    }
    let live = casing::casing_pool(sim.world())
        .map(|p| p.casings.clone())
        .unwrap_or_default();
    assert!(!live.is_empty(), "the burst left no brass to look for");

    let mut scene = inf_render::RenderScene::default();
    inf_player::render::project_scene(
        &mut scene,
        &sim,
        0.0,
        &inf_player::vmesh::VmeshRegistry::new(),
    );
    // A casing's instance is the one within a millimetre of where the pool says
    // it is. Position rather than scale or colour, because position is the thing
    // a projector gets wrong silently: a missing `GlobalTransform` draws at the
    // world ORIGIN and every other field still looks right.
    let mut missed: Vec<String> = Vec::new();
    let mut at_origin = 0usize;
    for c in &live {
        let hit = scene
            .instances
            .iter()
            .any(|i| (i.translation - c.at).length() < 1e-3);
        if !hit {
            let near_origin = scene
                .instances
                .iter()
                .any(|i| i.translation.length() < 1e-6 && i.scale.z < 0.05);
            if near_origin {
                at_origin += 1;
            }
            missed.push(format!("{:?}", c.at));
        }
    }
    println!(
        "{} live casings, {} projected instances in the scene, {} casings with no \
         instance at their position ({at_origin} of them with a candidate at the origin)",
        live.len(),
        scene.instances.len(),
        missed.len()
    );
    assert!(
        missed.is_empty(),
        "{} of {} live casings are not drawn where the pool says they are: {missed:?}",
        missed.len(),
        live.len()
    );
    // …and the instance really is casing-sized, so "drawn" is not "drawn as a
    // placeholder cube the size of a car".
    let c = &live[0];
    let inst = scene
        .instances
        .iter()
        .find(|i| (i.translation - c.at).length() < 1e-3)
        .expect("checked above");
    assert!(
        (f64::from(inst.scale.z) - casing::CASING_LENGTH_M).abs() < 1e-6
            && (f64::from(inst.scale.x) - casing::CASING_RADIUS_M * 2.0).abs() < 1e-6,
        "a casing is drawn at {:?}, not {} x {} m",
        inst.scale,
        casing::CASING_RADIUS_M * 2.0,
        casing::CASING_LENGTH_M
    );
}

struct Course {
    trace: Vec<Vec<u8>>,
    audio: Vec<String>,
    casings: usize,
}

/// Settle, arm, aim up, hold the trigger through the SHIPPED input path, and
/// let the brass fall.
fn sound_course(mut sim: inf_player::runtime_sim::RuntimeSim) -> Course {
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    let mut trace = Vec::new();
    for _ in 0..40 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        trace.push(sim.state_bytes());
    }
    assert_eq!(item::give(sim.world_mut(), hero, "m4a1", 1), 0);
    assert!(d3::gameplay::equip_weapon(sim.world_mut(), hero, "m4a1"));
    {
        let e = sim.world().entity_of(hero).expect("the hero");
        if let Some(mut cm) = sim.world_mut().world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.aim_pitch_deg = 20.0;
        }
    }
    let before = sim.audio_command_log().len();
    // Four seconds of trigger and four more of falling brass: forty steps was
    // this course's own first draft and it made forty commands, most of them
    // the listener's.
    let mut state = inf_input::InputState::new(inf_input::default_map());
    for i in 0..500 {
        let events = [inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: i < 240,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
        trace.push(sim.state_bytes());
    }
    assert_eq!(sim.dropped_audio_commands(), 0);
    // The stream as text: a command's Debug is a complete description of it,
    // which is what makes a mismatch readable rather than a byte offset.
    let audio: Vec<String> = sim.audio_command_log()[before..]
        .iter()
        .map(|c| format!("{c:?}"))
        .collect();
    Course {
        trace,
        audio,
        casings: casing::casings_live(sim.world()),
    }
}

/// **A QUIET LEVEL FOLDS NOTHING** — the half that keeps every trace committed
/// before this wave byte-identical.
#[test]
fn the_brass_costs_a_level_that_never_fires_nothing() {
    let mut sim = pie_sim();
    for _ in 0..20 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let quiet = sim.state_bytes();
    // An empty pool is not a pool.
    sim.world_mut()
        .world_mut()
        .insert_resource(CasingPool::default());
    assert_eq!(
        quiet,
        sim.state_bytes(),
        "an empty casing pool moved the trace"
    );
    assert!(casing::casing_state_bytes(sim.world()).is_empty());
    // …and one casing moves it, at the TAIL.
    casing::eject_casing(
        sim.world_mut(),
        Uuid::from_u128(1),
        WeaponClass::Ar,
        DVec3::new(1.0, 2.0, 3.0),
        DVec3::new(2.0, 1.0, 0.0),
    )
    .expect("a casing");
    let with = sim.state_bytes();
    assert_ne!(quiet, with);
    assert!(
        with.starts_with(&quiet),
        "the brass was not appended at the tail \u{2014} a section inserted before \
         the fifteen frozen ones moves every committed hash in the tree"
    );
    println!(
        "quiet {} bytes; +one casing {} bytes (+{})",
        quiet.len(),
        with.len(),
        with.len() - quiet.len()
    );
}

// ── (i) COST ────────────────────────────────────────────────────────────────

/// **WHAT A FIREFIGHT COSTS**, printed rather than claimed.
///
/// Commands a step at eight shooters, the casing pool's own microseconds, and
/// the ray bill against its ceiling. No threshold is asserted on the timing —
/// this machine is not a budget — but the ENGAGEMENT is, because a measurement
/// of a system that did nothing is not a measurement.
#[test]
fn the_cost_of_eight_shooters_is_measured_and_printed() {
    let mut r = Range::new(defs_with("rifle", test_rifle()), false);
    r.listen(DVec3::new(0.0, 1.5, -6.0));
    for i in 0..8 {
        stand(
            &mut r.world,
            shooter_guid(i),
            "Shooter",
            DVec3::new(i as f64 * 4.0 - 14.0, 0.0, 0.0),
            false,
        );
    }
    r.world.propagate();
    for i in 0..8 {
        let g = shooter_guid(i);
        r.arm(g, "rifle");
        r.aim(g, 0.0, 6.0);
        r.hold_trigger(g, true);
    }
    // Fill the pools first, so the measured window is the steady state.
    for _ in 0..300 {
        r.step();
    }
    let t0 = std::time::Instant::now();
    let (mut shots, mut commands, mut casing_rays, mut shot_rays, mut peak) =
        (0u32, 0usize, 0u32, 0u32, 0usize);
    const N: usize = 300;
    for _ in 0..N {
        let report = r.step();
        shots += report.shots;
        commands +=
            report.shots as usize * 4 + report.casings.bounces.len() + report.cracks.len() + 1;
        casing_rays += report.casings.rays;
        shot_rays += report.rounds.rays + report.rounds.probe_rays + report.shots;
        peak = peak.max(casing::casings_live(&r.world));
    }
    let per_step_us = t0.elapsed().as_secs_f64() * 1e6 / N as f64;
    println!(
        "eight shooters, {N} steps: {shots} shots, {commands} audio commands \
         ({:.1}/step), {shot_rays} shot rays ({:.1}/step, ceiling {}), \
         {casing_rays} casing rays ({:.1}/step), peak {peak} casings, \
         {per_step_us:.1} us/step (whole gameplay phase)",
        commands as f64 / N as f64,
        f64::from(shot_rays) / N as f64,
        ballistics::MAX_SHOT_RAYS_PER_STEP,
        f64::from(casing_rays) / N as f64,
    );
    assert!(shots > 0 && commands > 0, "nothing was measured");
    assert!(peak > 0, "no brass was in the world during the measurement");
    assert!(
        casing_rays > 0,
        "the casing pass never cast a ray, so its cost is not what was measured"
    );
    // The size of the thing the log holds, so the ceiling's memory claim is a
    // measurement rather than an estimate.
    println!(
        "size_of::<AudioCommand>() = {} bytes; the log's ceiling is {} entries",
        std::mem::size_of::<AudioCommand>(),
        inf_audio::AUDIO_LOG_CAPACITY
    );
}

// ── (j) THE TEN-SECOND COMMAND LOG ──────────────────────────────────────────

/// **TEN SECONDS OF A FIREFIGHT, AS TEXT** — and the same weapon on the same
/// level chooses a different TAIL depending on where it is standing.
///
/// The wave's own deliverable and its sharpest arm at once. It runs the shipped
/// host on the committed gameplay level twice: once where the hero spawns,
/// which is inside the PCG house the level grows, and once sixty metres away
/// from it in the open. The command streams are printed in full and the tail
/// clip is compared.
///
/// **Mutation → red:** `ENCLOSURE_INDOOR_HITS` to 1 or to 7 makes both places
/// answer the same thing and the two streams carry the same tail.
#[test]
fn ten_seconds_of_a_firefight_indoors_and_out_choose_different_tails() {
    let indoor_clip = weapon::report_clip(WeaponClass::Ar, ReportClip::IndoorTail);
    let outdoor_clip = weapon::report_clip(WeaponClass::Ar, ReportClip::OutdoorTail);
    let mut tails = Vec::new();
    for (place, away) in [("INSIDE THE HOUSE", 0.0), ("OUT ON THE OPEN GROUND", 60.0)] {
        let (log, shots) = firefight_log(away);
        println!("\n=== TEN SECONDS, {place} ({shots} rounds) ===");
        for (i, line) in log.iter().enumerate() {
            println!("{i:5} {line}");
        }
        let saw_indoor = log.iter().any(|c| c.contains(&format!("{indoor_clip}")));
        let saw_outdoor = log.iter().any(|c| c.contains(&format!("{outdoor_clip}")));
        println!("--- {place}: indoor tail {saw_indoor}, outdoor tail {saw_outdoor}");
        assert!(shots > 0, "{place}: nothing fired");
        assert!(
            saw_indoor != saw_outdoor,
            "{place} played both tails or neither"
        );
        tails.push(saw_indoor);
    }
    assert_eq!(
        tails,
        vec![true, false],
        "the same weapon on the same level chose the same tail inside a house \
         and sixty metres away from it \u{2014} the probe is not reading the room"
    );
}

/// Ten seconds of the fixture's hero holding the trigger, `away` metres from
/// where it spawns. Answers the audio command stream as text and the rounds it
/// took to make it.
fn firefight_log(away: f64) -> (Vec<String>, u32) {
    let mut sim = pie_sim();
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    for _ in 0..40 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    if away != 0.0 {
        let e = sim.world().entity_of(hero).expect("the hero");
        let mut t = *sim
            .world()
            .world()
            .get::<Transform>(e)
            .expect("a transform");
        t.translation.x += away;
        sim.world_mut().world_mut().entity_mut(e).insert(t);
        sim.world_mut().mark_dirty();
        for _ in 0..20 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
    }
    assert_eq!(item::give(sim.world_mut(), hero, "m4a1", 1), 0);
    assert!(d3::gameplay::equip_weapon(sim.world_mut(), hero, "m4a1"));
    let before = sim.audio_command_log().len();
    let mut shots = 0u32;
    let mut state = inf_input::InputState::new(inf_input::default_map());
    for i in 0..600 {
        let events = [inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: i < 300,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
        shots += sim.gameplay().shots;
    }
    assert_eq!(sim.dropped_audio_commands(), 0);
    let log = sim.audio_command_log()[before..]
        .iter()
        .map(|c| format!("{c:?}"))
        .collect();
    (log, shots)
}

// ── the fixture (wpn2a_gate's own helpers, one gate over) ───────────────────

fn sample_files() -> Vec<String> {
    std::fs::read_dir(inf_editor_core::samples::gameplay_dir())
        .expect("the fixture folder")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
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
    inf_player::sim_from_built(built)
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
    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
}
