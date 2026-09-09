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
        self.world.world_mut().entity_mut(e).insert((
            t,
            inf_ecs::components::AudioListener { active: true },
        ));
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
    let mut r = Range::new(defs_with("rifle", deep), false);
    r.listen(DVec3::new(0.0, 1.5, -6.0));
    let mut log: inf_core::BoundedLog<AudioCommand> =
        inf_core::BoundedLog::new(inf_audio::AUDIO_LOG_CAPACITY);
    for i in 0..8 {
        let g = shooter_guid(i);
        stand(
            &mut r.world,
            g,
            "Shooter",
            DVec3::new(i as f64 * 4.0 - 14.0, 0.0, 0.0),
            false,
        );
    }
    r.world.propagate();
    for i in 0..8 {
        let g = shooter_guid(i);
        r.arm(g, "rifle");
        r.aim(g, 0.0, 8.0);
        r.hold_trigger(g, true);
    }
    let steps = (120.0 / DT) as usize;
    let mut shots = 0u64;
    for _ in 0..steps {
        let report = r.step();
        shots += u64::from(report.shots);
        // The host's own arithmetic: four layers a loud shot, one landing per
        // first contact, and one listener command a step.
        for _ in 0..report.shots * 4 + report.casings.bounces.len() as u32 + 1 {
            log.push(AudioCommand::Stop { source: 0 });
        }
        for _ in 0..report.cracks.len() {
            log.push(AudioCommand::Stop { source: 0 });
        }
    }
    println!(
        "120 s at eight shooters: {shots} rounds, {} commands, {} dropped (ceiling {})",
        log.len(),
        log.dropped(),
        inf_audio::AUDIO_LOG_CAPACITY
    );
    assert!(
        shots > 8_000,
        "only {shots} rounds in two minutes — eight shooters at 600 rpm is 9 600"
    );
    assert_eq!(
        log.dropped(),
        0,
        "the log evicted {} commands over the arm it was re-priced for",
        log.dropped()
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
        Some(weapon::report_clip(WeaponClass::Ar, ReportClip::OutdoorTail))
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
    if !dir.join(inf_editor_core::weapon_audio::CASING_FILE).exists() {
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
    println!("class    clip          bytes    rate   seconds");
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
    println!("{} Hz one-pole: {db:.3} dB at its own cutoff", weapon::DISTANT_LOWPASS_HZ);
    assert!((db + 3.0103).abs() < 0.01);
    // …and eight octaves up it is gone.
    let up = f.response_at(weapon::DISTANT_LOWPASS_HZ * 8.0, inf_audio::synth::SYNTH_RATE);
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
    println!("weapon          v0     miss    cracks");
    for (name, v0, miss, want) in [
        ("m4a1", 900.0, 2.0, true),
        ("m4a1", 900.0, 10.0, false),
        // The AS VAL is deliberately subsonic; it is silent by physics.
        ("as_val", 295.0, 2.0, false),
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
    assert!(pool.spawned > MAX_CASINGS_LIVE as u64, "the ring never wrapped");
    assert!(peak <= MAX_CASINGS_LIVE, "the ring held {peak}");
    assert!(pool.recycled > 0, "nothing was recycled, so nothing wrapped");
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
    assert_eq!(casing::casings_live(&r.world), 0, "the brass never aged out");
    assert!(
        casing::drawn_casings(&r.world).is_empty(),
        "the entities outlived the pool"
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
        (weapon::report_clip(WeaponClass::Ar, ReportClip::Transient), "the transient"),
        (weapon::WEAPON_REPORT_CLIP, "the body"),
        (weapon::report_clip(WeaponClass::Ar, ReportClip::Crack), "the distant crack"),
        (weapon::CASING_CLIP, "a casing landing"),
    ] {
        let needle = format!("{clip}");
        assert!(
            ta.audio.iter().any(|c| c.contains(&needle)),
            "{what} is not in the course's command stream, so the arms below              compare a stream that does not contain it"
        );
    }
    // **THE FIXTURE IS A ROOM**, measured rather than assumed: the gameplay
    // level grows a PCG house around the spot the hero stands on, so the
    // enclosure probe calls every shot on it INDOORS and the tail in this
    // stream is the short one. That is the probe working on committed content
    // rather than on a fixture built to make it work, and the two tails are
    // named here so the day the house moves the arm says which way.
    let indoor = format!("{}", weapon::report_clip(WeaponClass::Ar, ReportClip::IndoorTail));
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
        "the gameplay fixture's hero stands inside its own PCG house, so the          probe should call it indoors"
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
        commands += report.shots as usize * 4
            + report.casings.bounces.len()
            + report.cracks.len()
            + 1;
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
