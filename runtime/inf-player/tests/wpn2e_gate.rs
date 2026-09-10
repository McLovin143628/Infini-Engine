//! **WAVE WPN2e — NPC GUNPLAY.** The gate.
//!
//! # What every arm in this file reads
//!
//! **The world, and never a policy table's opinion of itself.** An aim is
//! `MovementRuntime::aim_yaw_deg` after the pass ran; a shot is a `WeaponHit`
//! with an origin, a target and joules; a line of sight is a ray and its hit
//! entity; a hold is a round that did **not** leave when everything else about
//! the step said it should; a wanted level is `CrimeRes`' own heat.
//!
//! # THE LAW THIS GATE EXISTS FOR
//!
//! **The police do not cheat.** An officer aims at somebody only when the file's
//! own [`inf_ecs::crime::Profile::last_seen`] is fresh and in range **and** a ray
//! cast this step came back clear. `a_unit_with_no_sight_and_no_trail_never_aims`
//! is that arm, and it is written to fail if either half is removed — the
//! mutation is named beside it and was run.
//!
//! # For every arm: what it reads, and whether a no-op would pass it
//!
//! | arm | reads | would a policy that did nothing pass? |
//! |---|---|---|
//! | `an_arriving_police_crew_is_issued_a_weapon…` | the crew's `WeaponState` and its item id | no — it asserts a weapon EXISTS on the officer |
//! | `the_ladder_is_three_behaviours…` | `aimed` / `triggers` / `shots` per rung | no — Swat must FIRE and Patrol must AIM |
//! | `a_unit_with_no_sight_and_no_trail_never_aims` | `aimed`, with a wall and with a cold trail | **yes** — and that is the point: it is the arm a cheating policy fails, so it is paired with a positive control in the same test |
//! | `a_unit_behind_another_unit_holds_its_fire` | `friendly_holds` and `shots` | no — the control fires |
//! | `a_civilian_in_the_line_stops_the_shot` | `civilian_holds`, `shots`, and the ray's own first hit | no — the control fires |
//! | `a_unit_that_is_not_engaged_runs_no_engage_rays` | `rays` | no — the control spends rays |
//! | `an_officer_in_cover_fires_on_the_peek…` | the cover mode, `peek`, and shots per peek | no — it asserts shots happened |
//! | `a_blind_round_leaves_along_the_covers_own_normal` | the hit's `from`/`to` against the surface plane | no — it asserts a round left and cleared the wall |
//! | `the_hero_fires_blind_from_cover_too` | `blind_shots` and the round's direction | no |
//! | `a_wounding_raises_its_own_act…` | the witness log and `CrimeRes` heat | no |
//! | `a_killing_is_filed_against_the_shooter` | the act's `actor` | no |
//! | `a_car_shot_at_spends_nothing` | the chassis' joules, and the round that stopped in it | **yes**, deliberately: it is VEH3c's boundary, recorded rather than fixed |
//! | `pie_equals_shipping…` | three `state_bytes` traces and three audio logs | no — it asserts the course fired |
//! | `a_quiet_level_folds_nothing…` | `state_bytes` on a level with no file | **yes** — it is the byte-identity half and is a control by construction |
//! | the two budget arms | a CONTROL step and a measured step in the same process | no — they assert engagement first |
//!
//! # The mutations each arm dies to
//!
//! Written beside the arm, not in a report. They were run.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    MovementMode, RigidBody3D, Transform,
};
use inf_ecs::crime::{self, Response};
use inf_ecs::dispatch::{self, FleetRes, FleetUnit, UnitKind, UnitRun, UnitState};
use inf_ecs::engage;
use inf_ecs::item::{self, ItemDefs};
use inf_ecs::math::Vec3d;
use inf_ecs::weapon::{self, WeaponDef};
use inf_ecs::witness::{ActKind, WitnessedAct};
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};
use inf_project::ProjectManifest;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const RADIUS: f64 = 0.3;

const HERO: Uuid = Uuid::from_u128(0x2E00_0001);
const GROUND: Uuid = Uuid::from_u128(0x2E00_0002);

fn chassis_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2E00_0100 + i as u128)
}

fn civilian_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2E00_0200 + i as u128)
}

fn wall_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2E00_0300 + i as u128)
}

// ── the beat ────────────────────────────────────────────────────────────────

/// **A street with a hero on it, a duty roster, and a clock.**
///
/// Three things are hand-installed and each is a real door rather than a poke:
///
/// * the **roster** is [`inf_ecs::dispatch::FleetRes`] plus
///   [`inf_ecs::dispatch::DispatchRes`]`::runs`, which is *exactly* what
///   `d3::crime::officers` reads — a fixture that invented its own officer list
///   would be testing a list nothing else in the engine looks at;
/// * the **clock** is a hand-installed [`inf_ecs::traffic::TrafficPopulationRes`]
///   (`hand_installed` exists for this: nothing derives over one). EMS3's own
///   carried item 3 is that **a level with no streets has no clock** and every
///   act is stamped zero, which would freeze the cadence and the trail age. So
///   the beat advances `steps` itself, once per fixed step, exactly as
///   `step_traffic` would;
/// * the **file** is [`inf_ecs::crime::report_act`], the same door the witness
///   pass files through, with a real observer list — it refuses an act nobody
///   saw.
struct Beat {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    officers: Vec<Uuid>,
}

impl Beat {
    fn new() -> Self {
        let mut world = EcsWorld::new();
        slab(
            &mut world,
            GROUND,
            "Ground",
            DVec3::new(0.0, -0.5, 0.0),
            Vec3d::new(400.0, 0.5, 400.0),
        );
        stand(&mut world, HERO, "Hero", DVec3::ZERO, true);
        *item::item_defs_mut(&mut world) = registry();
        // The clock. See the struct doc.
        world
            .world_mut()
            .insert_resource(inf_ecs::traffic::TrafficPopulationRes {
                hand_installed: true,
                ..Default::default()
            });
        world.mark_dirty();
        world.propagate();
        let mut b = Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
            officers: Vec::new(),
        };
        b.bridge.sync_from_world(&b.world);
        b
    }

    /// Put a police unit on the street: a chassis in the fleet, a run that is
    /// **on scene**, a crew body at `at`, and the responder flag the panic
    /// exemption and the cover pass both read.
    fn officer(&mut self, i: usize, at: DVec3) -> Uuid {
        let chassis = chassis_guid(i);
        let crew = dispatch::crew_guid(chassis);
        {
            let mut fleet = self
                .world
                .world_mut()
                .remove_resource::<FleetRes>()
                .unwrap_or_default();
            fleet.units.insert(
                chassis,
                FleetUnit {
                    kind: UnitKind::Police,
                    station: Uuid::nil(),
                    home: at,
                    home_yaw_deg: 0.0,
                },
            );
            self.world.world_mut().insert_resource(fleet);
            let mut res = self
                .world
                .world_mut()
                .remove_resource::<dispatch::DispatchRes>()
                .unwrap_or_default();
            res.runs.insert(
                chassis,
                UnitRun {
                    state: UnitState::OnScene,
                    incident: None,
                    since_step: 0,
                    path: None,
                },
            );
            self.world.world_mut().insert_resource(res);
        }
        stand(&mut self.world, crew, "Officer", at, false);
        self.resync();
        dispatch::set_responder(&mut self.world, crew, true);
        weapon::give_health(&mut self.world, crew, weapon::DEFAULT_VITALITY_J);
        self.officers.push(crew);
        crew
    }

    fn arm(&mut self, who: Uuid, id: &str) {
        assert_eq!(item::give(&mut self.world, who, id, 1), 0, "{id} not given");
        assert!(
            d3::gameplay::equip_weapon(&mut self.world, who, id),
            "{id} not equipped"
        );
    }

    /// A pedestrian: a body in the world **and** a record in the crowd
    /// population, because `witness::candidates_near` — the door the discipline
    /// rule's cone walk uses — reads the population and not the entities.
    fn civilian(&mut self, i: usize, at: DVec3) -> Uuid {
        let guid = civilian_guid(i);
        stand(&mut self.world, guid, "Pedestrian", at, false);
        self.resync();
        let mut recs = BTreeMap::new();
        recs.insert(
            guid,
            inf_ecs::crowd::CrowdRecord::standing(
                inf_ecs::crowd::CrowdArchetype::humanoid(None, None, None),
                at,
            ),
        );
        // `add_agents` refuses a guid the world already has an entity for, which
        // is exactly this case — so the record goes in directly, through the
        // resource the population itself is.
        let mut pop = self
            .world
            .world_mut()
            .remove_resource::<inf_ecs::crowd::CrowdPopulationRes>()
            .unwrap_or_default();
        pop.hand_installed = true;
        pop.records.extend(recs);
        self.world.world_mut().insert_resource(pop);
        weapon::give_health(&mut self.world, guid, weapon::DEFAULT_VITALITY_J);
        guid
    }

    /// **Open a file on the hero** at the rung `want` — through
    /// [`inf_ecs::crime::report_act`], the same door the witness pass files
    /// through, repeated until the heat reaches the rung.
    ///
    /// `at` is where the police last had them, which is the ONLY positional
    /// input the whole policy is allowed.
    fn file_on_hero(&mut self, want: Response, at: DVec3) -> u32 {
        let step = inf_ecs::traffic::steps(&self.world);
        let look = inf_ecs::witness::look_digest(&self.world, HERO);
        let mut heat = 0;
        for _ in 0..8 {
            if Response::for_heat(heat) >= want {
                break;
            }
            heat = crime::report_act(
                &mut self.world,
                &WitnessedAct {
                    kind: ActKind::Shot,
                    actor: HERO,
                    at,
                    step,
                    observers: vec![civilian_guid(90)],
                    actor_look: look,
                    actor_vehicle: None,
                    heard_by: 0,
                },
                None,
            )
            .expect("the file opens");
        }
        assert_eq!(
            Response::for_heat(heat),
            want,
            "the fixture wanted {} and the file is at {} ({heat} heat)",
            want.name(),
            Response::for_heat(heat).name()
        );
        heat
    }

    /// **Keep a magazine full**, so a fixture's suppression is a sustained fact
    /// rather than a four-second one.
    ///
    /// A rifle carries thirty rounds and a fixture that wants an officer to sit
    /// behind a wall for ten seconds needs six hundred. Reloading through the
    /// real door would work and would spend a third of every run in a reload
    /// animation nobody is measuring; this tops the clock up and says so.
    fn top_up(&mut self, who: Uuid) {
        let Some(e) = self.world.entity_of(who) else {
            return;
        };
        let Some((_, def)) = weapon::equipped_def(&self.world, who) else {
            return;
        };
        if let Some(mut st) = self.world.world_mut().get_mut::<weapon::WeaponState>(e) {
            st.magazine = def.magazine;
            st.reserve = def.magazine * 4;
        }
    }

    fn resync(&mut self) {
        self.world.mark_dirty();
        self.world.reindex_guids();
        self.world.propagate();
        self.bridge.sync_from_world(&self.world);
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

    fn cm(&self, who: Uuid) -> CharacterMovement {
        let e = self.world.entity_of(who).expect("a character");
        self.world
            .world()
            .get::<CharacterMovement>(e)
            .cloned()
            .expect("a character")
    }

    fn health(&self, guid: Uuid) -> f64 {
        weapon::health_of(&self.world, guid)
            .map(|h| h.joules)
            .unwrap_or(f64::NAN)
    }

    /// One fixed step, in the hosts' own order, with the clock advanced exactly
    /// as `step_traffic` advances it.
    fn step(&mut self) -> d3::GameplayReport {
        self.bridge.sync_from_world(&self.world);
        // **THE WANTED SYSTEM, WHERE THE HOSTS RUN IT** -- `step_dispatch` calls
        // `step_recognition` at the top of the `dispatch` phase, seven phases
        // before gameplay, and it is the ONLY thing that turns a witnessed act
        // into heat (`crime::file_new_acts`). A beat that skipped it would fire
        // rounds all day and never escalate anybody, and the escalation arm
        // would be measuring a frozen ledger.
        //
        // `step_recognition` and not the whole `step_dispatch`, because this
        // fixture hand-installs its roster: `sync_fleet` derives a fleet from the
        // level's BLOCKS, and this street has none.
        let step = inf_ecs::traffic::steps(&self.world);
        d3::crime::step_recognition(&mut self.world, &mut self.bridge, step);
        d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
        let report = d3::step_gameplay(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        if let Some(mut pop) = self
            .world
            .world_mut()
            .get_resource_mut::<inf_ecs::traffic::TrafficPopulationRes>()
        {
            pop.steps += 1;
        }
        report
    }

    /// Run `n` steps and add up everything the policy did.
    fn run(&mut self, n: usize) -> Tally {
        let mut t = Tally::default();
        for _ in 0..n {
            let r = self.step();
            t.add(&r);
        }
        t
    }
}

/// What a run of the beat did — the policy's counters, summed, plus what the
/// weapons actually produced.
#[derive(Default, Debug, Clone)]
struct Tally {
    rays: usize,
    aimed: usize,
    warned: usize,
    triggers: usize,
    friendly_holds: usize,
    civilian_holds: usize,
    posture_holds: usize,
    cadence_holds: usize,
    blocked: usize,
    cold_trail: usize,
    blind: usize,
    shots: u32,
    blind_shots: u32,
    /// Every hit whose shooter was NOT the hero — the officers' own rounds.
    npc_hits: Vec<d3::WeaponHit>,
    /// Steps on which at least one officer was in `MovementMode::Cover`.
    peeks: usize,
}

impl Tally {
    fn add(&mut self, r: &d3::GameplayReport) {
        self.rays += r.engage.rays;
        self.aimed += r.engage.aimed;
        self.warned += r.engage.warned;
        self.triggers += r.engage.triggers;
        self.friendly_holds += r.engage.friendly_holds;
        self.civilian_holds += r.engage.civilian_holds;
        self.posture_holds += r.engage.posture_holds;
        self.cadence_holds += r.engage.cadence_holds;
        self.blocked += r.engage.blocked;
        self.cold_trail += r.engage.cold_trail;
        self.blind += r.engage.blind;
        self.shots += r.shots;
        self.blind_shots += r.blind_shots;
        self.peeks += usize::from(r.npc_cover.peeking > 0);
        for h in &r.hits {
            if h.shooter != HERO {
                self.npc_hits.push(*h);
            }
        }
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
        CharacterController3D::default(),
        cm,
        t,
    ));
}

/// The shipped registry, so an arm that names a weapon reads the row the game
/// ships rather than inventing one (the WPN2c audit's finding).
fn registry() -> ItemDefs {
    let mut d = ItemDefs::default();
    d.merge_toml(weapon::WEAPON_REGISTRY_TOML)
        .expect("the shipped weapon registry parses");
    d
}

fn row(id: &str) -> WeaponDef {
    registry()
        .get(id)
        .and_then(|i| i.weapon)
        .unwrap_or_else(|| panic!("{id} is not in the shipped registry"))
}

// ═════════════════════════════════════════════════════════════════════════════
// (a) TARGET SELECTION AND THE LADDER
// ═════════════════════════════════════════════════════════════════════════════

/// **AN ARRIVING POLICE CREW IS ISSUED A WEAPON, AND A FIRE CREW IS NOT.**
///
/// `equip_weapon` had no dispatch caller for two waves. This reads the officer's
/// own `WeaponState` off the world after `d3::dispatch::arrive` ran, which is the
/// only place `arm_crew` is called from.
///
/// The rung decides the row: a `Patrol` gets the sidearm, a `Swat` response gets
/// the rifle, and a **`Cold`** town issues nothing at all — which is what keeps
/// every level committed before this wave equipping nothing.
///
/// **Mutation → red:** dropping the `arm_crew` call in `arrive` (no
/// `WeaponState`); removing the `kind != Police` refusal (the fire crew is
/// armed); returning `POLICE_SIDEARM_IDS` for every rung (SWAT carries a
/// pistol).
#[test]
fn an_arriving_police_crew_is_issued_a_weapon_and_a_cold_town_issues_none() {
    for (want, expect) in [
        (Response::Cold, None),
        (Response::Patrol, Some("glock_17")),
        (Response::MultiUnit, Some("glock_17")),
        (Response::Swat, Some("m4a1")),
    ] {
        let mut b = Beat::new();
        let crew = b.officer(0, DVec3::new(0.0, 0.0, 12.0));
        if want != Response::Cold {
            b.file_on_hero(want, DVec3::ZERO);
        }
        let issued = d3::engage::arm_crew(&mut b.world, crew, UnitKind::Police);
        let have = weapon::equipped_def(&b.world, crew).map(|(id, _)| id);
        println!(
            "  {:<11} -> issued {:?}, equipped {:?}",
            want.name(),
            issued,
            have
        );
        assert_eq!(
            issued,
            expect,
            "the {} rung issued the wrong row",
            want.name()
        );
        assert_eq!(
            have.as_deref(),
            expect,
            "the {} rung's officer is carrying the wrong thing",
            want.name()
        );
        // …and the LEDGER counted it, which is what tells "the door ran" from
        // "the officer already had one".
        let equips = engage::engage_of(&b.world).map(|r| r.equips).unwrap_or(0);
        assert_eq!(
            equips,
            u64::from(expect.is_some()),
            "the {} rung issued {issued:?} and the ledger counted {equips}",
            want.name()
        );
    }
    // …and a FIRE crew standing at the same scene is issued nothing, at the
    // hottest rung there is.
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 12.0));
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    assert_eq!(
        d3::engage::arm_crew(&mut b.world, crew, UnitKind::Fire),
        None,
        "a fire crew was issued a weapon"
    );
    assert!(
        weapon::equipped_def(&b.world, crew).is_none(),
        "a fire crew is carrying a weapon"
    );
    // And the door refuses to hand out a SECOND magazine to a crew that already
    // has the row — `equip_weapon` reinstalls a full one, so an officer that
    // arrived twice would have infinite ammunition.
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    assert!(d3::engage::arm_crew(&mut b.world, crew, UnitKind::Police).is_some());
    let e = b.world.entity_of(crew).expect("the officer");
    if let Some(mut st) = b.world.world_mut().get_mut::<weapon::WeaponState>(e) {
        st.magazine = 3;
    }
    assert!(d3::engage::arm_crew(&mut b.world, crew, UnitKind::Police).is_some());
    let mag = b
        .world
        .world()
        .get::<weapon::WeaponState>(e)
        .map(|s| s.magazine)
        .unwrap_or(0);
    assert_eq!(mag, 3, "arriving twice refilled the officer's magazine");
}

/// **THE LADDER IS THREE BEHAVIOURS, NOT THREE UNIT COUNTS** — the arm the
/// mandate's "response" line is actually about.
///
/// Three runs of the same street with the same officer at the same distance,
/// differing only in how much heat is on the hero's file:
///
/// * **Patrol** — the weapon comes out and points, and **no round leaves**;
/// * **MultiUnit** — nothing leaves until the hero fires, and then it does;
/// * **Swat** — rounds leave on sight.
///
/// **Mutation → red:** `posture_for` returning `FireOnSight` for every rung
/// (the patrol fires); returning `Warn` for every rung (Swat fires nothing);
/// `Posture::may_fire` ignoring `fired_upon` (the MultiUnit control fires
/// unprovoked).
#[test]
fn the_ladder_is_three_behaviours_on_one_street() {
    // A tidy row for the report, and the numbers the design rests on.
    println!("=== the response ladder, as behaviour ===");
    println!(
        "  {:<11}  {:<14}  {:>5}  {:>6}  {:>7}  {:>9}",
        "rung", "posture", "aimed", "warned", "trigger", "npc shots"
    );
    let mut rows: Vec<(Response, Tally, u64, u64)> = Vec::new();
    for rung in [Response::Patrol, Response::MultiUnit, Response::Swat] {
        let mut b = Beat::new();
        let crew = b.officer(0, DVec3::new(0.0, 0.0, 14.0));
        b.file_on_hero(rung, DVec3::ZERO);
        b.arm(crew, "glock_17");
        // Two engagement cycles, so every unit's cadence window opens.
        let t = b.run(2 * engage::ENGAGE_PERIOD_STEPS as usize);
        println!(
            "  {:<11}  {:<14}  {:>5}  {:>6}  {:>7}  {:>9}",
            rung.name(),
            engage::posture_for(rung).name(),
            t.aimed,
            t.warned,
            t.triggers,
            t.npc_hits.len()
        );
        let (shots, holds) = engage::engage_of(&b.world)
            .map(|r| (r.shots, r.holds))
            .unwrap_or((0, 0));
        rows.push((rung, t, shots, holds));
    }
    let patrol = &rows[0].1;
    let multi = &rows[1].1;
    let swat = &rows[2].1;
    // The LEDGER's own session counters, beside the report's per-step ones —
    // two places that must agree about whether anybody fired, and the only
    // readers `EngageRes::shots` and `EngageRes::holds` have.
    println!(
        "  the ledger over the SWAT run: {} trigger-steps, {} refused",
        rows[2].2, rows[2].3
    );
    assert_eq!(
        rows[2].2 as usize, swat.triggers,
        "the ledger counted {} trigger-steps and the report counted {}",
        rows[2].2, swat.triggers
    );
    assert_eq!(
        rows[0].2, 0,
        "the ledger counted {} trigger-steps for a PATROL",
        rows[0].2
    );
    assert!(
        rows[0].3 > 0,
        "a patrol refused nothing, so `EngageRes::holds` never moved"
    );
    // ARMED: the pass ran and pointed weapons in all three.
    for (rung, t, _, _) in &rows {
        assert!(
            t.aimed > 0,
            "the {} rung never aimed at anybody — the fixture is not engaged",
            rung.name()
        );
    }
    // A PATROL WARNS. Every aim it made had the trigger closed.
    assert_eq!(
        patrol.triggers, 0,
        "a patrol pulled the trigger {} times",
        patrol.triggers
    );
    assert_eq!(
        patrol.warned, patrol.aimed,
        "a patrol aimed {} times and warned {} — the two must be the same thing",
        patrol.aimed, patrol.warned
    );
    assert!(
        patrol.npc_hits.is_empty(),
        "a patrol fired {} rounds",
        patrol.npc_hits.len()
    );
    // A MULTI-UNIT RESPONSE HOLDS UNTIL IT IS FIRED UPON.
    assert_eq!(
        multi.triggers, 0,
        "a multi-unit response opened fire with nobody shooting at it"
    );
    assert!(
        multi.posture_holds > 0,
        "the multi-unit rung never recorded a posture hold, so the rule is vacuous"
    );
    // SWAT FIRES ON SIGHT.
    assert!(
        swat.triggers > 0,
        "a SWAT response never pulled a trigger over {} steps",
        2 * engage::ENGAGE_PERIOD_STEPS
    );
    assert!(
        !swat.npc_hits.is_empty(),
        "a SWAT response pulled the trigger {} times and no round left a barrel",
        swat.triggers
    );
    // …and the middle rung DOES fire once the hero shoots at it.
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 14.0));
    b.file_on_hero(Response::MultiUnit, DVec3::ZERO);
    b.arm(crew, "glock_17");
    b.arm(HERO, "glock_17");
    b.aim(HERO, 0.0, 0.0);
    b.hold_trigger(HERO, true);
    let provoked = b.run(2 * engage::ENGAGE_PERIOD_STEPS as usize);
    println!(
        "  multi-unit, once fired upon: {} triggers, {} rounds returned",
        provoked.triggers,
        provoked.npc_hits.iter().filter(|h| h.loud).count()
    );
    assert!(
        provoked.triggers > 0,
        "a multi-unit response was shot at and never returned fire"
    );
}

/// **THE LAW: A UNIT WITH NO LINE OF SIGHT AND NO TRAIL NEVER AIMS.**
///
/// Three streets, one control and two refusals, each isolating one half of
/// [`inf_ecs::engage::may_engage`]:
///
/// 1. **clear and fresh** — the officer aims (the positive control, without
///    which the two refusals below are satisfied by any policy that does
///    nothing);
/// 2. **a wall on the line** — the ray comes back stopped, `blocked` rises, and
///    NOTHING is aimed;
/// 3. **a cold trail** — the ray would be clear and nobody has seen the suspect
///    for longer than [`inf_ecs::engage::TRAIL_STALE_STEPS`], so no ray is even
///    spent.
///
/// **THE MUTATION, and it is the CRITICAL one:** make the applier pass the
/// suspect's real transform to `may_engage` in place of `Profile::last_seen`, or
/// hard-code the `line_of_sight` argument to `true`. Either turns case 2 or
/// case 3 into an aim and this arm goes red. Both were run.
#[test]
fn a_unit_with_no_sight_and_no_trail_never_aims() {
    // 1. THE CONTROL.
    let mut clear = Beat::new();
    let a = clear.officer(0, DVec3::new(0.0, 0.0, 14.0));
    clear.file_on_hero(Response::Swat, DVec3::ZERO);
    clear.arm(a, "glock_17");
    let t_clear = clear.run(30);
    println!(
        "  clear line: {} rays, {} blocked, {} aimed",
        t_clear.rays, t_clear.blocked, t_clear.aimed
    );
    assert!(
        t_clear.aimed > 0,
        "the control never aimed, so the two refusals below prove nothing"
    );

    // 2. A WALL ON THE LINE.
    let mut walled = Beat::new();
    let b = walled.officer(0, DVec3::new(0.0, 0.0, 14.0));
    walled.file_on_hero(Response::Swat, DVec3::ZERO);
    walled.arm(b, "glock_17");
    slab(
        &mut walled.world,
        wall_guid(0),
        "Wall",
        DVec3::new(0.0, 2.0, 7.0),
        Vec3d::new(6.0, 2.0, 0.5),
    );
    walled.resync();
    let t_walled = walled.run(30);
    println!(
        "  a wall between: {} rays, {} blocked, {} aimed",
        t_walled.rays, t_walled.blocked, t_walled.aimed
    );
    assert!(
        t_walled.rays > 0,
        "no ray was spent, so `blocked` proves nothing about the wall"
    );
    assert!(
        t_walled.blocked > 0,
        "the ray went through a 12 m wall — the LOS test is not casting"
    );
    assert_eq!(
        t_walled.aimed, 0,
        "an officer aimed at somebody it has no line of sight to — THE POLICE CHEATED"
    );

    // 3. A COLD TRAIL. The file is opened, and then the clock is run past
    //    `TRAIL_STALE_STEPS` with nobody looking, so `last_seen` ages out. The
    //    officer is spawned AFTER, so it never had a fresh trail at all.
    let mut cold = Beat::new();
    cold.file_on_hero(Response::Swat, DVec3::ZERO);
    if let Some(mut pop) = cold
        .world
        .world_mut()
        .get_resource_mut::<inf_ecs::traffic::TrafficPopulationRes>()
    {
        pop.steps = engage::TRAIL_STALE_STEPS + 60;
    }
    let c = cold.officer(0, DVec3::new(0.0, 0.0, 14.0));
    cold.arm(c, "glock_17");
    let t_cold = cold.run(30);
    println!(
        "  a trail {} steps cold: {} rays, {} refused for age, {} aimed",
        engage::TRAIL_STALE_STEPS + 60,
        t_cold.rays,
        t_cold.cold_trail,
        t_cold.aimed
    );
    assert!(
        t_cold.cold_trail > 0,
        "the trail-age rule refused nothing, so it is vacuous here"
    );
    assert_eq!(
        t_cold.aimed, 0,
        "an officer aimed at a trail three seconds cold — THE POLICE REMEMBERED WITHOUT LOOKING"
    );
    assert_eq!(
        t_cold.rays, 0,
        "a ray was spent on a pair the trail-age rule had already refused"
    );

    // 4. **THE MAN IS STANDING IN FRONT OF YOU AND THE FILE SAYS HE IS A
    //    HUNDRED METRES AWAY** (wave WPN2e audit).
    //
    //    # Why this street had to be added
    //
    //    Streets 1-3 cannot see the cheat this arm is named for. The wave's own
    //    report names the falsifier as *"pass the suspect's transform to
    //    `may_engage` in place of `last_seen`"* -- and this audit RAN that
    //    mutation and the arm stayed **GREEN**. It has to: street 2 is refused
    //    by a RAY and street 3 by the trail's AGE, and neither of those reads a
    //    position at all, so substituting one position for another changes
    //    nothing in either. The one thing `last_seen` decides on its own is the
    //    RANGE, and until now no street made the remembered place and the real
    //    body disagree about it.
    //
    //    Here they disagree by seventy metres. The file was opened at 100 m and
    //    is FRESH; the suspect is standing 30 m from the officer with a clear
    //    line and nothing in the way. Read `last_seen` and the pair is out of
    //    `ENGAGE_RANGE_M` before a ray is spent; read the transform and the
    //    officer points a weapon at a man it has no reason to know is there.
    //
    //    Thirty metres and not fourteen, deliberately: outfit-only recognition
    //    reaches **16.7 m** in daylight (`crime::Channel::weight`'s own table),
    //    so at fourteen the officer would legitimately RECOGNISE the hero, and
    //    `crime::sight` would move `last_seen` onto him -- which is the police
    //    doing their job and would dissolve the premise.
    //
    //    **The mutation**: replace `*last_seen` with `Vec3d::from_dvec3(*at)` at
    //    EITHER call site in `d3::engage::step_engage` -- `worth_a_ray`'s or
    //    `may_engage`'s -- and this street reds while the three above stay green.
    let mut lying = Beat::new();
    let d = lying.officer(0, DVec3::new(0.0, 0.0, 30.0));
    lying.file_on_hero(Response::Swat, DVec3::new(0.0, 0.0, 100.0));
    lying.arm(d, "glock_17");
    let t_lying = lying.run(30);
    let remembered = crime::profile_of(&lying.world, HERO)
        .map(|f| f.last_seen())
        .expect("the file");
    println!(
        "  the file says (0, 0, {:.0}) and the man is at (0, 0, 0), 30 m from the officer: {} rays, {} aimed",
        remembered.z, t_lying.rays, t_lying.aimed
    );
    assert!(
        (remembered.z - 100.0).abs() < 1.0,
        "the fixture's own premise moved: `last_seen` is at z {:.1}, so nobody is being remembered in the wrong place",
        remembered.z
    );
    assert_eq!(
        t_lying.rays, 0,
        "a ray was spent on a suspect the FILE puts 70 m outside `ENGAGE_RANGE_M` — the range gate is reading a position nobody looked at"
    );
    assert_eq!(
        t_lying.aimed, 0,
        "an officer aimed at a man whose file puts him a hundred metres away — THE POLICE READ THE TRANSFORM"
    );

    // …and the CONTROL for street 4, which is what makes its two zeros mean
    // something: the same street with the file pointing at where the man really
    // is. Same officer, same distance, same everything else.
    let mut honest = Beat::new();
    let e = honest.officer(0, DVec3::new(0.0, 0.0, 30.0));
    honest.file_on_hero(Response::Swat, DVec3::ZERO);
    honest.arm(e, "glock_17");
    let t_honest = honest.run(30);
    println!(
        "  the CONTROL, the same street with the file pointing at the man: {} rays, {} aimed",
        t_honest.rays, t_honest.aimed
    );
    assert!(
        t_honest.aimed > 0,
        "the control never aimed at 30 m, so street 4's zeros are about the range and not about the file"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (b) FIRE DISCIPLINE
// ═════════════════════════════════════════════════════════════════════════════

/// **A UNIT BEHIND ANOTHER UNIT HOLDS ITS FIRE.**
///
/// Two officers on the same bearing from the hero, one at 8 m and one at 16 m.
/// The far one's aim line passes through its colleague, so its trigger is held
/// while the near one's is not — and the count of holds is read off the pass.
///
/// **Mutation → red:** deleting the `friendly_in_cone` call (both fire, holds
/// zero); making `in_cone` ignore the "nearer than the target" clause (the NEAR
/// officer also holds, because its colleague behind it is now "in the way").
#[test]
fn a_unit_behind_another_unit_holds_its_fire() {
    let mut b = Beat::new();
    // Both on the +Z bearing from the hero at the origin.
    let near = b.officer(0, DVec3::new(0.0, 0.0, 8.0));
    let far = b.officer(1, DVec3::new(0.0, 0.0, 16.0));
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.arm(near, "glock_17");
    b.arm(far, "glock_17");
    let t = b.run(3 * engage::ENGAGE_PERIOD_STEPS as usize);
    let by: BTreeMap<Uuid, usize> = t.npc_hits.iter().fold(BTreeMap::new(), |mut m, h| {
        *m.entry(h.shooter).or_default() += 1;
        m
    });
    println!("=== a unit behind another unit ===");
    println!(
        "  near ({near}) fired {}, far ({far}) fired {}",
        by.get(&near).copied().unwrap_or(0),
        by.get(&far).copied().unwrap_or(0)
    );
    println!(
        "  friendly holds {}, civilian holds {}, cadence holds {}",
        t.friendly_holds, t.civilian_holds, t.cadence_holds
    );
    assert!(
        by.get(&near).copied().unwrap_or(0) > 0,
        "the NEAR officer, with a clear line, fired nothing — the fixture is not engaged"
    );
    assert!(
        t.friendly_holds > 0,
        "nobody held fire for a colleague standing in the line"
    );
    assert_eq!(
        by.get(&far).copied().unwrap_or(0),
        0,
        "the FAR officer fired through its own colleague"
    );
}

/// **A CIVILIAN IN THE LINE STOPS THE SHOT** — measured on a crowded street.
///
/// A pedestrian standing between the officer and the suspect. The engage ray's
/// first hit is that pedestrian, the shot is held, and the control — the same
/// street with the pedestrian four metres to one side — fires.
///
/// **Mutation → red:** deleting the `civilian_in_the_way` call (the officer
/// fires and the crowd walk counts nothing); treating `Sight::Person` as
/// `Sight::Wall` (the officer never aims at all, which is a different and
/// wrong behaviour — the aim is what makes an officer point a weapon at you
/// while a bystander is in the way).
#[test]
fn a_civilian_in_the_line_stops_the_shot() {
    let run = |offset_x: f64| -> (Tally, Uuid) {
        let mut b = Beat::new();
        let crew = b.officer(0, DVec3::new(0.0, 0.0, 14.0));
        b.file_on_hero(Response::Swat, DVec3::ZERO);
        b.arm(crew, "glock_17");
        let c = b.civilian(0, DVec3::new(offset_x, 0.0, 7.0));
        (b.run(3 * engage::ENGAGE_PERIOD_STEPS as usize), c)
    };
    let (blocked, who) = run(0.0);
    let (clear, _) = run(6.0);
    println!("=== a civilian in the line ===");
    println!(
        "  in the line   : {} aimed, {} civilian holds, {} rounds fired",
        blocked.aimed,
        blocked.civilian_holds,
        blocked.npc_hits.len()
    );
    println!(
        "  six m aside   : {} aimed, {} civilian holds, {} rounds fired",
        clear.aimed,
        clear.civilian_holds,
        clear.npc_hits.len()
    );
    assert!(
        !clear.npc_hits.is_empty(),
        "the control fired nothing, so the refusal below proves nothing"
    );
    assert!(
        blocked.aimed > 0,
        "the officer did not even AIM with somebody in the way — a body is not a wall"
    );
    assert!(
        blocked.civilian_holds > 0,
        "nobody held fire for a pedestrian standing in the line"
    );
    assert!(
        blocked.npc_hits.is_empty(),
        "{} round(s) were fired through a pedestrian",
        blocked.npc_hits.len()
    );
    // …and nobody shot the pedestrian, which is what the hold is FOR.
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 14.0));
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.arm(crew, "glock_17");
    b.civilian(0, DVec3::new(0.0, 0.0, 7.0));
    let t = b.run(3 * engage::ENGAGE_PERIOD_STEPS as usize);
    assert!(
        t.npc_hits.iter().all(|h| h.target != Some(who)),
        "the pedestrian was shot"
    );
}

/// **A UNIT THAT IS NOT ENGAGED RUNS ZERO ENGAGE RAYS**, and an engaged street
/// never runs more than its own budget.
///
/// `NPC_ENGAGE_RAYS_PER_STEP` is minted by this wave and is neither WPN1's
/// witness budget nor EMS3's recognition budget nor
/// `MAX_SHOT_RAYS_PER_STEP` — the arm below asserts all three are different
/// numbers or different things, so a later wave cannot quietly collapse them.
///
/// **Mutation → red:** dropping the `wanted.is_empty()` early return (rays on a
/// quiet street); dropping the `stats.rays >= NPC_ENGAGE_RAYS_PER_STEP` refusal
/// (the saturation case exceeds the ceiling).
#[test]
fn a_unit_that_is_not_engaged_runs_no_engage_rays() {
    // 1. A street with officers on it and nobody wanted.
    let mut quiet = Beat::new();
    for i in 0..4 {
        let c = quiet.officer(i, DVec3::new(i as f64 * 3.0, 0.0, 12.0));
        quiet.arm(c, "glock_17");
    }
    let t_quiet = quiet.run(120);
    println!(
        "  four armed officers, nobody wanted: {} engage rays over 120 steps",
        t_quiet.rays
    );
    assert_eq!(
        t_quiet.rays, 0,
        "the firing policy spent {} rays on a street where nobody is wanted",
        t_quiet.rays
    );
    assert_eq!(t_quiet.aimed, 0, "somebody was aimed at with no open file");

    // 2. The same street with a file open — the falsifier for the zero above.
    let mut hot = Beat::new();
    for i in 0..4 {
        let c = hot.officer(i, DVec3::new(i as f64 * 3.0, 0.0, 12.0));
        hot.arm(c, "glock_17");
    }
    hot.file_on_hero(Response::Swat, DVec3::ZERO);
    let mut worst = 0usize;
    for _ in 0..120 {
        worst = worst.max(hot.step().engage.rays);
    }
    println!(
        "  the same street, one file open: at most {worst} rays in a step (ceiling {})",
        engage::NPC_ENGAGE_RAYS_PER_STEP
    );
    assert!(worst > 0, "an engaged street spent no rays at all");
    assert!(
        worst <= engage::NPC_ENGAGE_RAYS_PER_STEP,
        "the policy spent {worst} rays in one step against its own ceiling of {}",
        engage::NPC_ENGAGE_RAYS_PER_STEP
    );

    // 3. Saturation: more pairs than the ceiling. Sixteen officers and one file
    //    is sixteen pairs, which is exactly the budget; twenty officers is over
    //    it and must still stop at the ceiling.
    let mut many = Beat::new();
    for i in 0..20 {
        let a = i as f64 * 0.31;
        let c = many.officer(i, DVec3::new(a.cos() * 15.0, 0.0, a.sin() * 15.0 + 16.0));
        many.arm(c, "glock_17");
    }
    many.file_on_hero(Response::Swat, DVec3::ZERO);
    let mut sat = 0usize;
    for _ in 0..60 {
        sat = sat.max(many.step().engage.rays);
    }
    println!(
        "  twenty officers, one file: at most {sat} rays in a step (ceiling {})",
        engage::NPC_ENGAGE_RAYS_PER_STEP
    );
    assert_eq!(
        sat,
        engage::NPC_ENGAGE_RAYS_PER_STEP,
        "twenty officers and one open file did not saturate the ray budget"
    );

    // …and the three budgets are three numbers, or three different things.
    println!(
        "  budgets: engage {} / recognition {} / witness acts {} x observers {} / shot rays {}",
        engage::NPC_ENGAGE_RAYS_PER_STEP,
        inf_physics::d3::crime::MAX_RECOGNITION_RAYS,
        inf_ecs::witness::MAX_WITNESSED_ACTS,
        inf_ecs::witness::MAX_OBSERVERS,
        inf_ecs::ballistics::MAX_SHOT_RAYS_PER_STEP,
    );
    assert_ne!(
        engage::NPC_ENGAGE_RAYS_PER_STEP,
        inf_physics::d3::crime::MAX_RECOGNITION_RAYS,
        "the engage budget has become the recognition budget — two different questions, one number"
    );
    assert_ne!(
        engage::NPC_ENGAGE_RAYS_PER_STEP,
        inf_ecs::ballistics::MAX_SHOT_RAYS_PER_STEP,
        "the engage budget has become the shot ceiling"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (c) COVER, SUPPRESSION AND BLIND FIRE
// ═════════════════════════════════════════════════════════════════════════════

/// **AN OFFICER UNDER FIRE TAKES COVER AND FIRES FROM IT** — the `step_npc_cover`
/// seam, closed.
///
/// COV1 left the officer standing behind a wall with its body aim on the threat
/// and its trigger untouched, and said so in its own doc. This drives the whole
/// chain on one street: the hero fires, the panic sources reach the cover pass,
/// the officer searches, walks, presses cover — and **the firing policy pulls
/// the trigger from behind it**.
///
/// The measurement the brief asks for is the CADENCE: how many rounds leave per
/// step the officer is in cover, and how much of the run its trigger was open.
///
/// **Mutation → red:** deleting the `step_engage` call in `step_gameplay` (the
/// officer takes cover and fires nothing, which is exactly COV1's state).
#[test]
fn an_officer_under_fire_takes_cover_and_fires_from_it() {
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 16.0));
    // A low wall a metre and a half in front of the officer, facing the hero.
    slab(
        &mut b.world,
        wall_guid(0),
        "Parapet",
        DVec3::new(0.0, 0.45, 14.0),
        Vec3d::new(4.0, 0.45, 0.35),
    );
    b.resync();
    // **A WITNESS, SO THE FILE STAYS WARM.** `TRAIL_STALE_STEPS` is 180 and the
    // whole policy stops at it -- correctly: a suspect nobody has seen for three
    // seconds is a search and not a target. In a real street the hero's own
    // gunfire keeps refreshing the file, because somebody sees it; this fixture
    // has to put that somebody in. Twelve metres off to one side, which is 70
    // degrees away from the officer's firing line and therefore outside the
    // discipline cone. Measured without it: the officer engaged for the first
    // three seconds of a nineteen-second run and then stopped.
    b.civilian(5, DVec3::new(12.0, 0.0, 4.0));
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.arm(crew, "m4a1");
    // **AN AUTOMATIC WEAPON IN THE HERO'S HANDS, DELIBERATELY.** `try_fire`'s
    // edge rule means a held trigger on a SEMI-automatic is exactly one round,
    // so a hero holding a Glock puts one panic source on one step and the
    // officer is under fire for a sixtieth of a second. Suppression is a
    // sustained fact and needs a sustained noise.
    b.arm(HERO, "m4a1");
    b.aim(HERO, 0.0, 0.0);
    b.hold_trigger(HERO, true);
    let mut in_cover = 0usize;
    let mut peeking = 0usize;
    let mut npc_rounds = 0usize;
    let mut blind = 0usize;
    let steps = 12 * engage::ENGAGE_PERIOD_STEPS as usize;
    for _ in 0..steps {
        b.top_up(HERO);
        let r = b.step();
        in_cover += r.npc_cover.in_cover;
        peeking += r.npc_cover.peeking;
        npc_rounds += r.hits.iter().filter(|h| h.shooter == crew).count();
        blind += r.engage.blind;
    }
    let mode = b.cm(crew).mode;
    println!("=== the cover cadence ===");
    println!("  {steps} steps: {in_cover} in cover, {peeking} of them leaned out");
    println!(
        "  the officer fired {npc_rounds} rounds; the policy made {blind} BLIND decisions (steps its trigger was open with the peek shut)"
    );
    println!(
        "  duty cycle: {:.1} % out (`NPC_PEEK_OUT_S` {} / `NPC_PEEK_IN_S` {})",
        100.0 * peeking as f64 / in_cover.max(1) as f64,
        inf_ecs::cover::NPC_PEEK_OUT_S,
        inf_ecs::cover::NPC_PEEK_IN_S
    );
    if in_cover > 0 {
        println!(
            "  shots per step in cover: {:.4}",
            npc_rounds as f64 / in_cover as f64
        );
    }
    assert_eq!(
        mode,
        MovementMode::Cover,
        "the officer never got into cover, so the cadence measures nothing"
    );
    assert!(
        in_cover > 0 && peeking > 0,
        "the officer was in cover for {in_cover} steps and leaned out on {peeking} — the duty cycle is not running"
    );
    assert!(
        npc_rounds > 0,
        "an officer stood behind a wall for {steps} steps with a weapon and never fired — the WPN2e seam is still open"
    );
}

/// **A BLIND ROUND LEAVES ALONG THE COVER'S OWN NORMAL, AND CLEARS THE WALL** —
/// the ray arm COV1 priced.
///
/// Read off the HIT, not off a flag: the round's `from` is on the far side of the
/// surface plane, its direction is inside the resolved cone of the cover's own
/// outward normal, and the first thing it hits is **not the cover**. That last
/// clause is the whole reason `blind_fire_shot` moves the origin: a round that
/// left from where the hand actually is stops in its own wall on segment zero.
///
/// **Mutation → red:** returning the muzzle unchanged as the origin (the round
/// hits the parapet); using `+normal` instead of `-normal` (the round goes
/// backwards, away from the threat); dropping `BLIND_FIRE_CONE_DEG` (the cone
/// assertion below still passes — so the cone is asserted as a NUMBER against
/// the def's own, which a dropped addend fails).
#[test]
fn a_blind_round_leaves_along_the_covers_own_normal_and_clears_the_wall() {
    let mut b = Beat::new();
    // The hero presses into a low parapet at +Z, so its outward normal is +Z.
    // **Inside the press's own reach.** `CoverSettings::reach_m` is 0.9 m from
    // the capsule, and the capsule's radius is 0.3, so a face at 1.25 m is out
    // of reach and the press finds nothing — measured: the hero stayed
    // `Grounded` for forty steps against a parapet 1.6 m away.
    slab(
        &mut b.world,
        wall_guid(0),
        "Parapet",
        DVec3::new(0.0, 0.45, 0.9),
        Vec3d::new(4.0, 0.45, 0.35),
    );
    b.resync();
    b.arm(HERO, "glock_17");
    // Press cover the way a player's key does.
    {
        let e = b.world.entity_of(HERO).expect("the hero");
        let mut cm = b
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.body_yaw_deg = 0.0;
        cm.runtime.aim_yaw_deg = 0.0;
        cm.runtime.target_yaw_deg = 0.0;
        cm.runtime.press_cover = true;
    }
    for _ in 0..40 {
        b.step();
    }
    let cm = b.cm(HERO);
    assert_eq!(
        cm.mode,
        MovementMode::Cover,
        "the hero never got into cover, so nothing below is about blind fire"
    );
    let cover = cm.runtime.cover;
    println!("=== blind fire ===");
    println!(
        "  cover: class {:?}, normal ({:.3}, {:.3}, {:.3}), anchor ({:.3}, {:.3}, {:.3}), top {:.3} m",
        cover.class,
        cover.normal.x,
        cover.normal.y,
        cover.normal.z,
        cover.anchor.x,
        cover.anchor.y,
        cover.anchor.z,
        cover.top_m
    );
    // Trigger down, aim NOT held — the blind-fire condition.
    b.hold_trigger(HERO, true);
    {
        let e = b.world.entity_of(HERO).expect("the hero");
        let mut c = b
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        c.runtime.want_aim = false;
    }
    let mut hits: Vec<d3::WeaponHit> = Vec::new();
    let mut blind = 0u32;
    for _ in 0..60 {
        let r = b.step();
        blind += r.blind_shots;
        hits.extend(r.hits.iter().filter(|h| h.shooter == HERO).cloned());
        {
            let e = b.world.entity_of(HERO).expect("the hero");
            let mut c = b
                .world
                .world_mut()
                .get_mut::<CharacterMovement>(e)
                .expect("a character");
            c.runtime.want_attack = true;
            c.runtime.want_aim = false;
        }
    }
    assert!(blind > 0, "the hero in cover fired {blind} blind rounds");
    assert!(!hits.is_empty(), "no round left at all");
    let out = -cover.normal.to_dvec3();
    let out = DVec3::new(out.x, 0.0, out.z).normalize();
    // The parapet's own numbers: its near face and its top, in world metres.
    let near_face_z = 0.9 - 0.35;
    let top_y = 0.9;
    let cone = row("glock_17").spread_deg + inf_ecs::cover::BLIND_FIRE_CONE_DEG;
    // A LOW cover is fired OVER, so the origin's proof is its HEIGHT: the
    // surface's own measured top plus `BLIND_CLEAR_M`, above the anchor. (A HIGH
    // cover is fired ROUND, and its origin's proof is the lateral offset —
    // `inf_ecs::cover`'s own unit tests hold that half, because a fixture cannot
    // make the island's grammar produce a high wall with an end in it.)
    assert!(
        cover.class.crouches(),
        "the parapet came out as {:?} and this arm is about the low case",
        cover.class
    );
    let want_y = cover.anchor.y + cover.top_m + inf_ecs::cover::BLIND_CLEAR_M;
    for (i, h) in hits.iter().enumerate().take(6) {
        let dir = (h.to - h.from).normalize();
        let deg = out.dot(dir).clamp(-1.0, 1.0).acos().to_degrees();
        println!(
            "  round {i}: from ({:.3}, {:.3}, {:.3}) -> ({:.3}, {:.3}, {:.3}), {deg:.2} deg off the normal, target {:?}",
            h.from.x, h.from.y, h.from.z, h.to.x, h.to.y, h.to.z, h.target
        );
        // **OVER THE TOP**: the round leaves above the parapet, which is what
        // lets it clear a wall the shooter is crouched behind.
        assert!(
            (h.from.y - want_y).abs() < 1.0e-6,
            "a blind round left from y = {:.4} and the rule says {want_y:.4} (the parapet's top {top_y:.2} m + {:.2} m of clearance)",
            h.from.y,
            inf_ecs::cover::BLIND_CLEAR_M
        );
        assert!(
            h.from.y > top_y,
            "a blind round left from y = {:.3}, which is INSIDE a parapet whose top is {top_y:.2} m",
            h.from.y
        );
        // …and it moved OUTWARD, toward the threat rather than away from it.
        assert!(
            h.from.z > cover.anchor.z && h.from.z > near_face_z - 0.05,
            "a blind round left from z = {:.3}, which is not out toward the wall from the anchor at {:.3}",
            h.from.z,
            cover.anchor.z
        );
        // …and it leaves ALONG the outward normal, inside the widened cone.
        assert!(
            deg <= cone + 1.0e-6,
            "a blind round left {deg:.2} deg off the cover's normal against a {cone:.2} deg cone"
        );
        // …and it did not hit the thing it was fired over.
        assert_ne!(
            h.target,
            Some(wall_guid(0)),
            "a blind round hit its own cover"
        );
    }
    // …and NONE of the rounds hit the parapet, over the whole run.
    assert!(
        hits.iter().all(|h| h.target != Some(wall_guid(0))),
        "{} of {} blind rounds stopped in their own cover",
        hits.iter()
            .filter(|h| h.target == Some(wall_guid(0)))
            .count(),
        hits.len()
    );
    // The cone really is WIDER than an aimed shot's — a number, not a flag.
    println!(
        "  the blind cone is {cone:.2} deg against the row's own {:.2} deg",
        row("glock_17").spread_deg
    );
    assert!(
        cone > row("glock_17").spread_deg,
        "a blind shot is no less accurate than an aimed one"
    );
}

/// **AN NPC FIRES BLIND TOO** — the same branch, the other author.
///
/// The hero's half is above; this is the officer's. It also proves the rule is a
/// BEHAVIOUR and not a key: nothing sets a "blind" flag on the officer — its peek
/// duty cycle simply happens to be in its shut half when the cadence opens.
#[test]
fn an_npc_fires_blind_from_cover_too() {
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 16.0));
    slab(
        &mut b.world,
        wall_guid(0),
        "Parapet",
        DVec3::new(0.0, 0.45, 14.0),
        Vec3d::new(4.0, 0.45, 0.35),
    );
    b.resync();
    // A witness, so the file stays warm -- see
    // `an_officer_under_fire_takes_cover_and_fires_from_it` for the measurement.
    b.civilian(5, DVec3::new(12.0, 0.0, 4.0));
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.arm(crew, "m4a1");
    // Automatic, for `an_officer_under_fire_takes_cover_and_fires_from_it`'s
    // measured reason: a held semi-automatic is one round and one panic source.
    b.arm(HERO, "m4a1");
    b.aim(HERO, 0.0, 0.0);
    b.hold_trigger(HERO, true);
    let mut blind_decisions = 0usize;
    let mut blind_rounds = 0u32;
    let mut clip = 0.0f64;
    for _ in 0..24 * engage::ENGAGE_PERIOD_STEPS as usize {
        b.top_up(HERO);
        b.top_up(crew);
        let r = b.step();
        blind_decisions += r.engage.blind;
        blind_rounds += r.blind_shots;
        clip = clip.max(b.cm(crew).runtime.blind_fire_s);
    }
    println!("=== an NPC firing blind ===");
    println!(
        "  {blind_decisions} blind decisions, {blind_rounds} blind rounds, the one-shot reached {clip:.3} s (clip {:.2} s)",
        inf_anim::BLIND_FIRE_S
    );
    assert!(
        blind_decisions > 0,
        "the officer never took a blind shot — its trigger never came up inside the peek's shut half"
    );
    assert!(
        blind_rounds > 0,
        "{blind_decisions} blind decisions produced no blind round"
    );
    assert!(
        clip > 0.0,
        "the authored one-shot's clock was never started, so nothing would draw the arm going over the wall"
    );
    assert!(
        clip <= inf_anim::BLIND_FIRE_S + 1.0e-9,
        "the one-shot clock reached {clip:.3} s against a {:.2} s clip",
        inf_anim::BLIND_FIRE_S
    );
}

/// **A HIT UNIT STAGGERS, A CIVILIAN FLEES, AND THE RESPONSE ESCALATES.**
///
/// One street, one burst, and three consequences read off three different
/// places: WPN1's stagger counter, the crowd's own flee latch, and `CrimeRes`'
/// heat before and after.
///
/// **Mutation → red:** dropping `ActKind::Wounded` from `step_witness` (the heat
/// does not move); dropping the panic pass's `flee_from` (nobody runs).
#[test]
fn a_hit_officer_staggers_a_civilian_flees_and_the_heat_rises() {
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 10.0));
    b.civilian(0, DVec3::new(5.0, 0.0, 6.0));
    b.file_on_hero(Response::Patrol, DVec3::ZERO);
    b.arm(crew, "glock_17");
    b.arm(HERO, "m4a1");
    b.aim(HERO, 0.0, 0.0);
    let before = crime::heat_of(&b.world, HERO);
    let stars_before = crime::wanted_readout(&b.world, HERO);
    b.hold_trigger(HERO, true);
    let mut staggers = 0u32;
    let mut fled = 0usize;
    for _ in 0..180 {
        let r = b.step();
        staggers += r.staggers;
        fled += r.panic.fled;
    }
    let after = crime::heat_of(&b.world, HERO);
    let stars_after = crime::wanted_readout(&b.world, HERO);
    println!("=== the consequences of shooting at the police ===");
    println!("  heat {before} -> {after}; stars {stars_before:?} -> {stars_after:?}");
    println!(
        "  {staggers} staggers, {fled} agents fled, the officer has {:.0} J left",
        b.health(crew)
    );
    assert!(
        staggers > 0,
        "nothing was hit hard enough to stagger, so the escalation below is not about gunfire"
    );
    assert!(
        after > before,
        "the hero shot at the police and the response did not escalate ({before} -> {after})"
    );
    assert!(
        Response::for_heat(after) > Response::for_heat(before),
        "the heat rose from {before} to {after} and the RUNG did not move ({} -> {})",
        Response::for_heat(before).name(),
        Response::for_heat(after).name()
    );
    assert!(fled > 0, "nobody on the street ran from a firefight");
    assert!(
        b.health(crew) < weapon::DEFAULT_VITALITY_J,
        "the officer took no joules"
    );
}

/// **THE HERO UNDER NPC FIRE TAKES THE LAZY HEALTH PATH, AND THE HUD REACTS.**
///
/// Before the officers fire the hero has **no `Health` component at all** —
/// WPN1's lazy rule, and the reason a level that has never been shot at folds no
/// health bytes. The first round that lands installs it. The stars are read
/// through `wanted_readout`, which is the same door the HUD draws.
#[test]
fn the_hero_under_fire_takes_the_lazy_health_path_and_the_hud_reacts() {
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 10.0));
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.arm(crew, "m4a1");
    let e = b.world.entity_of(HERO).expect("the hero");
    assert!(
        b.world.world().get::<weapon::Health>(e).is_none(),
        "the hero already carries a Health component before anybody shot at it"
    );
    let stars_before = crime::wanted_readout(&b.world, HERO);
    let mut landed = 0usize;
    for _ in 0..6 * engage::ENGAGE_PERIOD_STEPS as usize {
        let r = b.step();
        landed += r
            .hits
            .iter()
            .filter(|h| h.shooter == crew && h.target == Some(HERO))
            .count();
    }
    let health = b.health(HERO);
    let stars_after = crime::wanted_readout(&b.world, HERO);
    println!("=== the hero under police fire ===");
    println!(
        "  {landed} rounds landed on the hero; {health:.0} J left of {:.0}",
        weapon::DEFAULT_VITALITY_J
    );
    println!("  the wanted readout: {stars_before:?} -> {stars_after:?}");
    assert!(landed > 0, "the officers never hit the hero");
    assert!(
        b.world
            .world()
            .get::<weapon::Health>(b.world.entity_of(HERO).expect("the hero"))
            .is_some(),
        "a round landed on the hero and no Health was installed — the lazy path did not run"
    );
    assert!(
        health < weapon::DEFAULT_VITALITY_J,
        "the hero was hit {landed} times and lost nothing"
    );
    let (stars, max) = stars_after.expect("the hero is wanted");
    assert!(
        stars > 0,
        "the hero is wanted and the HUD reads {stars} stars"
    );
    assert_eq!(max, crime::WANTED_STARS.len() as u8);
}

// ═════════════════════════════════════════════════════════════════════════════
// (d) THE ACTS
// ═════════════════════════════════════════════════════════════════════════════

/// **A NON-FATAL ROUND RAISES AN ACT OF ITS OWN** — the WPN2a audit's carried
/// item 205, closed.
///
/// The same street twice: the hero fires at a wall, and the hero fires at a
/// person. Before this wave both recorded exactly one `Shot` and the town could
/// not tell them apart. Now the second also records a `Wounded` against the
/// shooter, and the heat differs by `ActKind::Wounded::heat()`.
///
/// **Mutation → red:** deleting the `Wounded` bucket in `step_witness` (the two
/// runs record the same acts and the same heat); dropping the `killed`/`downed`
/// refusals (a corpse is wounded sixty times a second and the heat runs away).
#[test]
fn a_non_fatal_round_raises_an_act_of_its_own() {
    let shoot_at = |flesh: bool| -> (Vec<ActKind>, u32) {
        let mut b = Beat::new();
        if flesh {
            b.civilian(0, DVec3::new(0.0, 0.0, 6.0));
        } else {
            slab(
                &mut b.world,
                wall_guid(0),
                "Backstop",
                DVec3::new(0.0, 1.4, 6.0),
                Vec3d::new(3.0, 1.4, 0.4),
            );
            b.resync();
        }
        // A witness who is not the target, so the acts have observers.
        b.civilian(1, DVec3::new(6.0, 0.0, 3.0));
        b.arm(HERO, "glock_17");
        b.aim(HERO, 0.0, 0.0);
        b.hold_trigger(HERO, true);
        // One pull only: the glock is semi-automatic through `try_fire`'s own
        // edge rule, so the trigger stays down and exactly one round leaves.
        for _ in 0..4 {
            b.step();
        }
        let acts: Vec<ActKind> = inf_ecs::witness::witnessed(&b.world)
            .iter()
            .map(|a| a.kind)
            .collect();
        for a in inf_ecs::witness::witnessed(&b.world).to_vec() {
            crime::report_act(&mut b.world, &a, None);
        }
        (acts, crime::heat_of(&b.world, HERO))
    };
    let (at_wall, heat_wall) = shoot_at(false);
    let (at_person, heat_person) = shoot_at(true);
    println!("=== a round that connects, against one that does not ===");
    println!("  at a wall  : {at_wall:?}, heat {heat_wall}");
    println!("  at a person: {at_person:?}, heat {heat_person}");
    assert!(
        at_wall.contains(&ActKind::Shot),
        "firing at a wall recorded no gunshot at all — the fixture is not firing"
    );
    assert!(
        !at_wall.contains(&ActKind::Wounded),
        "a round that hit a wall wounded somebody"
    );
    assert!(
        at_person.contains(&ActKind::Wounded),
        "a round landed on a person and raised no act of its own — carried 205 is still open"
    );
    assert_eq!(
        heat_person,
        heat_wall + ActKind::Wounded.heat(),
        "shooting a person costs the same heat as shooting a wall"
    );
    // …and the file names the SHOOTER.
    let mut b = Beat::new();
    b.civilian(0, DVec3::new(0.0, 0.0, 6.0));
    b.civilian(1, DVec3::new(6.0, 0.0, 3.0));
    b.arm(HERO, "glock_17");
    b.aim(HERO, 0.0, 0.0);
    b.hold_trigger(HERO, true);
    for _ in 0..4 {
        b.step();
    }
    let wounded: Vec<WitnessedAct> = inf_ecs::witness::witnessed(&b.world)
        .iter()
        .filter(|a| a.kind == ActKind::Wounded)
        .cloned()
        .collect();
    assert!(!wounded.is_empty());
    for a in &wounded {
        assert_eq!(a.actor, HERO, "a wounding was filed against {}", a.actor);
    }
}

/// **A KILLING IS STILL FILED AGAINST THE SHOOTER, AND A CORPSE IS NOT WOUNDED
/// AGAIN.**
///
/// The EMS3 audit's fix, re-asserted where this wave could have broken it: the
/// new `Wounded` bucket sits immediately after the death bucket and shares its
/// `killed` list, so a wave that got the refusal backwards would file a wounding
/// against the killer on the step its victim died — on top of the killing — and
/// would go on filing one for every round that reached the body on the floor.
///
/// The fixture is a **one-shot kill** (the victim carries 500 J and an `m4a1`
/// round carries 1 700), so the first round is a killing and not a wounding, and
/// then the hero **aims down at the corpse** and empties four magazines into it.
/// Rounds really do land — WPN2a's `guid_of_ragdoll_collider` is what makes a
/// body on the floor hittable at all — and not one of them raises a wounding.
///
/// **Mutation → red:** dropping `killed.contains(&target)` from the bucket's
/// refusal (the death step also files a wounding); dropping
/// `weapon::is_downed(world, target)` (the corpse is wounded once per round).
/// **A GUNSHOT DOES NOT HAVE TO BE SEEN** (wave WPN2e audit) -- the EARS
/// channel, and the closure of this arc's link 1.
///
/// # What the wave measured, and why it was not a level-design fact
///
/// `the_islands_own_chain_from_a_gunshot_to_an_engaged_officer` found seventeen
/// gunshots at the showcase spawn recorded as seventeen acts with **zero**
/// observers -- the nearest crowd agent is 117 m away with a city block in the
/// way -- so `crime::report_act` refused every one and the town never opened a
/// file. The wave carried that as *"the island lacks somebody to see"*.
///
/// It is not a level-design fact. **You do not have to see a gunshot.** What was
/// missing is a channel, and this arm is it.
///
/// # What it reads, all of it the world
///
/// The witness log's own acts (`observers`, `heard_by`), the ledger's own heat,
/// the ledger's own EVIDENCE table, `Profile::last_seen` before and after the
/// shooter moves, and `GameplayReport::rounds.shot_rays` -- the named ray budget
/// the ears channel must not touch.
///
/// # The five claims
///
/// 1. a listener **beyond the witness radius** and with a wall in the way hears
///    a shot that nobody could possibly have seen, and it opens a file;
/// 2. it is worth **less than sight**: `ActKind::heard_heat` is one against
///    `heat`'s two, so the same burst brings a lower rung;
/// 3. it files **no description at all** -- the evidence table stays empty, so
///    nothing on the recognition path can match on a file opened by ear. That is
///    *"a lower evidence weight"* as a mechanism rather than as an adjective;
/// 4. it never moves `Profile::last_seen`. The shooter walks twenty metres and
///    fires again, and the place the police are searching does not move --
///    EMS3's law, intact, with hearing on top of it;
/// 5. it spends **zero rays**. The whole run's `shot_rays` is identical with the
///    listener present and absent, because sound is not a line of sight.
///
/// **The mutations**: return `0.0` from `weapon::audible_radius_m` (claim 1 goes
/// red, and the control below stays green -- which is what tells the two apart);
/// make `heard_heat` answer `2` (claim 2); file the outfit in `report_heard`
/// (claim 3); write `last_seen` in `report_heard` (claim 4).
#[test]
fn a_gunshot_nobody_saw_is_still_heard_and_opens_a_file() {
    // A listener at 200 m: outside `WITNESS_RADIUS_M` (120 m) and inside a
    // `glock_17`'s own report range, with a wall between it and the muzzle.
    const FAR_M: f64 = 200.0;

    let run = |listener: Option<f64>| -> (usize, usize, u32, u32, u32, DVec3, DVec3) {
        let mut beat = Beat::new();
        slab(
            &mut beat.world,
            wall_guid(1),
            "Wall",
            DVec3::new(0.0, 1.5, 12.0),
            Vec3d::new(30.0, 1.5, 0.5),
        );
        if let Some(d) = listener {
            beat.civilian(1, DVec3::new(0.0, 0.0, d));
        }
        beat.resync();
        beat.arm(HERO, "glock_17");
        beat.aim(HERO, 0.0, 0.0);
        let mut rays = 0u32;
        for i in 0..240 {
            beat.top_up(HERO);
            beat.hold_trigger(HERO, i % 4 == 0);
            let r = beat.step();
            rays += r.rounds.shot_rays;
        }
        let first_seen = crime::profile_of(&beat.world, HERO)
            .map(|f| f.last_seen())
            .unwrap_or(DVec3::ZERO);
        // …and now the shooter WALKS, and fires again from somewhere else.
        if let Some(e) = beat.world.entity_of(HERO) {
            if let Some(mut t) = beat.world.world_mut().get_mut::<Transform>(e) {
                t.translation = Vec3d::new(20.0, 0.0, 0.0);
            }
        }
        beat.resync();
        for i in 0..240 {
            beat.top_up(HERO);
            beat.hold_trigger(HERO, i % 4 == 0);
            let r = beat.step();
            rays += r.rounds.shot_rays;
        }
        let acts = inf_ecs::witness::witnessed(&beat.world);
        let seen = acts.iter().filter(|a| !a.observers.is_empty()).count();
        let heard = acts.iter().filter(|a| a.heard_by > 0).count();
        let heat = crime::heat_of(&beat.world, HERO);
        let evidence = crime::profile_of(&beat.world, HERO)
            .map(|f| f.evidence.len())
            .unwrap_or(0) as u32;
        let moved_seen = crime::profile_of(&beat.world, HERO)
            .map(|f| f.last_seen())
            .unwrap_or(DVec3::ZERO);
        (seen, heard, heat, evidence, rays, first_seen, moved_seen)
    };

    let (seen, heard, heat, evidence, rays, first, after) = run(Some(FAR_M));
    let (c_seen, c_heard, c_heat, _, control_rays, _, _) = run(None);
    println!("\n=== A GUNSHOT NOBODY SAW ===");
    println!(
        "  a listener at {FAR_M:.0} m, behind a wall, outside the {:.0} m witness radius:",
        d3::gameplay::WITNESS_RADIUS_M
    );
    println!(
        "    acts SEEN {seen} / acts HEARD {heard}; heat {heat}, rung {}",
        Response::for_heat(heat).name()
    );
    println!("    the file's evidence channels: {evidence}");
    println!("    last_seen {first:?} -> {after:?} after the shooter walked 20 m");
    println!(
        "  the CONTROL, nobody within earshot: acts SEEN {c_seen} / HEARD {c_heard}, heat {c_heat}"
    );
    println!("  shot rays: {rays} with a listener, {control_rays} without");

    // (1) heard, and filed.
    assert_eq!(
        seen, 0,
        "the listener is inside the witness radius after all - the fixture is not testing hearing"
    );
    assert!(
        heard > 0,
        "a gunshot 200 m from a pedestrian was heard by nobody"
    );
    assert!(heat > 0, "{heard} heard gunshots opened no file");
    // …and the CONTROL: with nobody in earshot, nothing at all.
    assert_eq!(c_heard, 0, "an empty street heard {c_heard} gunshots");
    assert_eq!(c_heat, 0, "an empty street opened a file");

    // (2) worth LESS than sight. The same acts, filed by `report_act`, would be
    //     worth `ActKind::Shot::heat`; by ear they are worth `heard_heat`.
    assert!(
        ActKind::Shot.heard_heat() < ActKind::Shot.heat(),
        "hearing a shot is worth as much as watching one"
    );
    assert_eq!(ActKind::Shot.heard_heat(), 1);
    // …and nothing but a gunshot is heard at all.
    for kind in [
        ActKind::Killed,
        ActKind::Wounded,
        ActKind::Carjack,
        ActKind::Assault,
    ] {
        assert_eq!(kind.heard_heat(), 0, "{} makes a noise now", kind.name());
    }

    // (3) NO DESCRIPTION. You cannot describe somebody you only heard.
    assert_eq!(
        evidence, 0,
        "a file opened by ear carries {evidence} description channel(s) - a hearer described somebody"
    );

    // (4) `last_seen` NEVER MOVES. EMS3's law: only a witness or a recognition
    //     may write it, and hearing is neither.
    assert_eq!(
        first, after,
        "the shooter walked 20 m, fired, and the place the police are searching MOVED - hearing wrote `last_seen`"
    );

    // (5) ZERO RAYS. The named budget is `MAX_SHOT_RAYS_PER_STEP`, and the ears
    //     channel does not draw on it or on any other.
    assert_eq!(
        rays,
        control_rays,
        "the ears channel spent {} ray(s) - sound is not a line of sight",
        rays.saturating_sub(control_rays)
    );
}

/// **A SHOT FIRED INSIDE IS NOT HEARD ACROSS THE ISLAND** -- the enclosure rule,
/// in the world (wave WPN2e audit).
///
/// The ears channel's radius is `inf_ecs::weapon::audible_radius_m`, which is the
/// weapon's own report range QUARTERED when the enclosure probe said the muzzle
/// was inside ([`inf_ecs::weapon::INDOOR_TAIL_REACH_FRACTION`], the audio
/// system's own number for how much of a report a room keeps to itself). This
/// arm builds the room out of slabs and reads the verdict off the shot itself.
///
/// **The mutation**: ignore `indoors` in `audible_radius_m` and the far listener
/// hears the indoor shot, which is a gunshot in a basement bringing a police car
/// four streets away.
#[test]
fn a_shot_fired_indoors_is_heard_a_quarter_as_far() {
    let outdoors = inf_ecs::weapon::audible_radius_m(weapon::REPORT_MAX_M, false);
    let indoors = inf_ecs::weapon::audible_radius_m(weapon::REPORT_MAX_M, true);
    assert_eq!(outdoors, weapon::REPORT_MAX_M);
    assert!(
        (indoors - weapon::REPORT_MAX_M * inf_ecs::weapon::INDOOR_TAIL_REACH_FRACTION).abs() < 1e-9
    );

    // A ROOM: six slabs, so the probe's six axial rays all hit inside
    // `ENCLOSURE_PROBE_M`. `ENCLOSURE_INDOOR_HITS` is four of six.
    let mut beat = Beat::new();
    for (i, (at, half)) in [
        (DVec3::new(0.0, 1.5, 4.0), Vec3d::new(6.0, 3.0, 0.3)),
        (DVec3::new(0.0, 1.5, -4.0), Vec3d::new(6.0, 3.0, 0.3)),
        (DVec3::new(4.0, 1.5, 0.0), Vec3d::new(0.3, 3.0, 6.0)),
        (DVec3::new(-4.0, 1.5, 0.0), Vec3d::new(0.3, 3.0, 6.0)),
        (DVec3::new(0.0, 4.0, 0.0), Vec3d::new(6.0, 0.3, 6.0)),
    ]
    .into_iter()
    .enumerate()
    {
        slab(&mut beat.world, wall_guid(20 + i), "Room", at, half);
    }
    // One listener just outside the indoor radius, one well inside it.
    let near = beat.civilian(2, DVec3::new(0.0, 0.0, 40.0));
    let far = beat.civilian(3, DVec3::new(0.0, 0.0, 100.0));
    beat.resync();
    beat.arm(HERO, "glock_17");
    beat.aim(HERO, 0.0, 0.0);
    let mut indoors_seen = 0u32;
    for i in 0..120 {
        beat.top_up(HERO);
        beat.hold_trigger(HERO, i % 4 == 0);
        let r = beat.step();
        indoors_seen += r.hits.iter().filter(|h| h.loud && h.indoors).count() as u32;
    }
    let acts = inf_ecs::witness::witnessed(&beat.world).to_vec();
    let heard: Vec<u8> = acts.iter().map(|a| a.heard_by).collect();
    let any = heard.iter().copied().max().unwrap_or(0);
    println!("\n=== A SHOT FIRED INDOORS ===");
    println!("  audible radius: {outdoors:.1} m outdoors, {indoors:.1} m indoors");
    println!("  loud shots the enclosure probe called INDOORS: {indoors_seen}");
    println!(
        "  listeners: one at 40 m ({near}), one at 100 m ({far}); the most any act was heard by: {any}"
    );
    assert!(
        indoors_seen > 0,
        "the room is not a room - the enclosure probe called every shot outdoors, so this arm measures nothing"
    );
    assert_eq!(
        any, 1,
        "an indoor shot reached {any} listener(s); the one at 40 m is inside {indoors:.1} m and the one at 100 m is not"
    );
}

/// **AN OFFICER RELOADS** (wave WPN2e audit, closing carried 274).
///
/// # What the wave shipped
///
/// `weapon::try_reload` has existed since wave I6 with exactly one caller: a
/// PLAYER's `press_reload`. So a police officer's magazine emptied and the
/// officer stopped firing for the rest of the session — which reads on screen as
/// trigger discipline and is a man standing in a firefight holding an empty gun.
/// The wave's own report says so, and its fixtures top magazines up by hand and
/// say why.
///
/// # What this reads — the world, and nothing tops anything up
///
/// A `Swat` street, one officer at fourteen metres, a hero it may fire on sight,
/// and sixty seconds — which is far longer than a `glock_17`'s seventeen rounds
/// can cover. Then:
///
/// * the rounds that actually left the barrel (`GameplayReport::shots`) against
///   the magazine size — **more than a magazine means it reloaded**;
/// * `EngageStats::reloads`, the counter the policy writes when it presses;
/// * the officer's own `WeaponState::reserve`, which is where the rounds came
///   from — a magazine that refilled from nowhere would leave it untouched.
///
/// **The mutation**: delete the `needs_a_reload` branch in `step_engage`. The
/// officer fires exactly one magazine and goes silent for the rest of the run —
/// `reloads` 0, `shots` capped at the magazine, `reserve` unmoved.
#[test]
fn an_officer_whose_magazine_empties_reloads_and_goes_on_firing() {
    let mut beat = Beat::new();
    let officer = beat.officer(0, DVec3::new(0.0, 0.0, 14.0));
    // A witness twelve metres to one side, outside the discipline cone: without
    // one the file goes cold in three seconds (`TRAIL_STALE_STEPS`) and the
    // whole policy stops — `an_officer_under_fire_takes_cover_and_fires_from_it`
    // measured the same thing and says so.
    beat.civilian(50, DVec3::new(12.0, 0.0, 4.0));
    beat.resync();
    beat.file_on_hero(Response::Swat, DVec3::ZERO);
    beat.arm(officer, "glock_17");
    // …and the hero holds an AUTOMATIC weapon down, which is what keeps the file
    // being refreshed: the witness sees each burst. The officer's own magazine is
    // never touched — that is the whole measurement.
    beat.arm(HERO, "m4a1");
    beat.aim(HERO, 0.0, 0.0);
    beat.hold_trigger(HERO, true);
    let magazine = weapon::equipped_def(&beat.world, officer)
        .map(|(_, d)| d.magazine)
        .unwrap_or(0);
    let reserve_before = beat
        .world
        .entity_of(officer)
        .and_then(|e| beat.world.world().get::<weapon::WeaponState>(e))
        .map(|st| st.reserve)
        .unwrap_or(0);
    let mut rounds = 0u32;
    let mut reloads = 0u64;
    let mut ran_dry = 0u32;
    for _ in 0..3_600 {
        beat.top_up(HERO);
        let r = beat.step();
        rounds += r.hits.iter().filter(|h| h.shooter == officer).count() as u32;
        reloads += r.engage.reloads as u64;
        let mag = beat
            .world
            .entity_of(officer)
            .and_then(|e| beat.world.world().get::<weapon::WeaponState>(e))
            .map(|st| st.magazine)
            .unwrap_or(0);
        if mag == 0 {
            ran_dry += 1;
        }
    }
    let reserve_after = beat
        .world
        .entity_of(officer)
        .and_then(|e| beat.world.world().get::<weapon::WeaponState>(e))
        .map(|st| st.reserve)
        .unwrap_or(0);
    println!("\n=== AN OFFICER THAT RUNS DRY ===");
    println!("  a `glock_17` holds {magazine} and the officer fired {rounds} round(s) over 60 s");
    println!(
        "  the policy asked for {reloads} reload(s); the magazine read empty on {ran_dry} step(s)"
    );
    println!("  the reserve went {reserve_before} -> {reserve_after}");
    assert!(
        rounds > magazine,
        "the officer fired {rounds} rounds out of a {magazine}-round magazine, so it never reloaded"
    );
    assert!(
        reloads > 0,
        "{rounds} rounds left the barrel and the policy asked for no reload at all"
    );
    assert!(
        reserve_after < reserve_before,
        "the reserve did not move ({reserve_before} -> {reserve_after}), so the magazine refilled from nowhere"
    );
}

/// **A TOWN THAT COOLS DOWN PUTS THE WEAPONS AWAY** (wave WPN2e audit, closing
/// carried 276).
///
/// # What the carry said, and what was actually true
///
/// The wave carried *"`Posture::Warn` NEVER HOLSTERS … putting it away again
/// needs an `unequip` door `inf_ecs::item` does not have — there is
/// `Inventory::equip` and no inverse"*. **`Inventory::unequip` has existed since
/// wave I6.** What was missing is the `d3::gameplay` door beside
/// `equip_weapon`, and the caller.
///
/// The moment is not the `Warn` rung — a patrol with its weapon out and pointed
/// at you is the whole of what one star buys — it is the town going **cold**.
/// `arm_crew` hands a weapon out at arrival and nothing ever took one back, so
/// an officer that answered one call carried a rifle for the rest of the
/// session on a street where nobody was wanted any more.
///
/// # What it reads
///
/// `weapon::equipped_def` on the officer's own body, before and after — the
/// world, not the ledger — and `EngageStats::holstered` beside it.
///
/// **The mutation**: delete the `unequip_weapon` call in `step_engage`'s cold
/// branch and the officer is still carrying the Glock at the end.
#[test]
fn a_town_that_goes_cold_takes_the_weapons_back() {
    let mut beat = Beat::new();
    let officer = beat.officer(0, DVec3::new(0.0, 0.0, 14.0));
    beat.civilian(50, DVec3::new(14.0, 0.0, 7.0));
    beat.resync();
    beat.file_on_hero(Response::Swat, DVec3::ZERO);
    beat.arm(officer, "glock_17");
    for _ in 0..120 {
        beat.step();
    }
    let armed_while_hot = weapon::equipped_def(&beat.world, officer).map(|(id, _)| id.to_string());
    assert_eq!(
        armed_while_hot,
        Some("glock_17".to_string()),
        "the fixture's officer is not armed, so there is nothing to put away"
    );
    // **The town cools down.** `clear_crime` is the Simulate twin and the only
    // door that closes a file without waiting out `HEAT_DECAY_STEPS` — a heat of
    // 6 takes six minutes of fixed steps to bleed off, which is a gate nobody
    // runs. What the policy sees is identical: `crime::wanted` is empty.
    //
    // **The witness log goes with it**, and that is a measurement rather than
    // tidiness: `clear_crime` drops `seen_act_step` too, so the very next
    // `file_new_acts` re-reads the whole log from step zero — and with this
    // audit's EARS channel the officer's own earlier gunshots have a hearer, so
    // the town would re-open a file **on the officer** and never be cold at all.
    // Both resources are the pair `SimSession` clears together.
    crime::clear_crime(&mut beat.world);
    inf_ecs::witness::clear_witness(&mut beat.world);
    let mut holstered = 0u64;
    for _ in 0..120 {
        let r = beat.step();
        holstered += r.engage.holstered as u64;
    }
    let after = weapon::equipped_def(&beat.world, officer).map(|(id, _)| id.to_string());
    // …and the weapon ENTITY goes with it, which is `step_equipped_weapons`'
    // own rule and the reason this is one field rather than three.
    let weapon_entity = beat
        .world
        .entity_of(d3::gameplay::equipped_weapon_guid(officer))
        .is_some();
    println!("\n=== A TOWN THAT COOLS DOWN ===");
    println!("  while wanted: the officer carries {armed_while_hot:?}");
    println!(
        "  once cold   : it carries {after:?}; {holstered} holster(s); the weapon entity {}",
        if weapon_entity {
            "is STILL in the world"
        } else {
            "is gone"
        }
    );
    assert_eq!(
        after, None,
        "the town went cold and the officer is still holding a weapon"
    );
    assert!(holstered > 0, "the weapon went away and nothing counted it");
    assert!(
        !weapon_entity,
        "the weapon is unequipped and its entity is still in the world"
    );
    // …and the item is STILL IN THE INVENTORY: holstering is not dropping.
    assert!(
        beat.world
            .entity_of(officer)
            .and_then(|e| beat.world.world().get::<inf_ecs::item::Inventory>(e))
            .map(|inv| inv.count_of("glock_17"))
            .unwrap_or(0)
            > 0,
        "the officer's holstered sidearm left its inventory — that is dropping, not holstering"
    );
}

/// **THE TRIGGER COMES DOWN WHEN THE POLICY STOPS LOOKING** (wave WPN2e audit).
///
/// # Why this arm exists
///
/// This is wave WPN2e's own headline defect — *"`npc_aim_at` writes `want_attack`
/// as a LEVEL and nothing lowered it … **258 rounds** left an officer's weapon
/// over a run in which the policy decided to fire ZERO times"* — and its fix,
/// `gameplay::npc_set_trigger`, **had no arm that could see it removed**. This
/// audit ran the mutation: delete the release loop in `step_engage` and every
/// arm in the file stays green, including the one the wave's report names.
///
/// It has to. In that fixture the hero fires all the way through, so the file
/// never goes cold, the officer is visited on every step, and the release path
/// is never taken at all. A fix nothing exercises is a fix nothing pins.
///
/// # What this reads
///
/// The world, in two halves of one run. The officer engages a hero it can see
/// and fires; then the hero is **taken two hundred metres away**, out of
/// `ENGAGE_RANGE_M` and out of every ray — so the policy stops visiting that
/// officer entirely, which is the exact condition the defect needed. The arm
/// counts the rounds that left the officer's barrel in each half and reads
/// `EngageStats::released` beside them.
///
/// **The mutation**: `if false && super::gameplay::npc_set_trigger(world, unit,
/// false)` in `step_engage`'s release loop — the second half fills with rounds
/// nobody decided to fire.
#[test]
fn an_officer_the_policy_stops_visiting_stops_firing() {
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 14.0));
    // The witness that keeps the file warm while the hero is still here.
    b.civilian(5, DVec3::new(12.0, 0.0, 4.0));
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.arm(crew, "m4a1");
    let mut first = 0usize;
    // **Run until the trigger is DOWN**, and that is the whole fixture: the
    // cadence opens it for `ENGAGE_BURST_STEPS` out of every
    // `ENGAGE_PERIOD_STEPS`, and the defect is a `true` the policy leaves behind
    // — so a run that happens to stop in the shut half of the cycle would find
    // nothing to release and would pass for the wrong reason.
    let mut held = false;
    for _ in 0..(8 * engage::ENGAGE_PERIOD_STEPS as usize) {
        b.top_up(crew);
        let r = b.step();
        first += r.hits.iter().filter(|h| h.shooter == crew).count();
        if first > 0 && b.cm(crew).runtime.want_attack {
            held = true;
            break;
        }
    }
    assert!(
        held,
        "the fixture never caught the officer with its trigger down, so there is nothing to release"
    );
    // **AND NOW THE SUSPECT IS BEHIND A WALL.** A twelve-metre slab goes up on
    // the line, so the officer is still VISITED — the file is warm and
    // `last_seen` is in range, so a ray is spent — and the ray comes back
    // stopped. It is not in cover, so it cannot fire blind either. That is the
    // shape the defect needed: a unit the policy looks at and decides not to
    // engage, whose `want_attack` the last visit left `true`.
    slab(
        &mut b.world,
        wall_guid(0),
        "Wall",
        DVec3::new(0.0, 2.0, 7.0),
        Vec3d::new(6.0, 2.0, 0.5),
    );
    b.resync();
    let mut second = 0usize;
    let mut released = 0usize;
    for _ in 0..(4 * engage::ENGAGE_PERIOD_STEPS as usize) {
        b.top_up(crew);
        let r = b.step();
        second += r.hits.iter().filter(|h| h.shooter == crew).count();
        released += r.engage.released;
    }
    println!("\n=== A TRIGGER NOBODY LOWERS ===");
    println!("  with the suspect in front of it: {first} round(s), trigger held {held}");
    println!("  once a wall goes up between them: {second} round(s), {released} release(s)");
    assert!(
        first > 0,
        "the officer never fired in the first half, so the second half's zero proves nothing"
    );
    assert!(
        released > 0,
        "the policy stopped visiting the officer and never released its trigger"
    );
    assert_eq!(
        second, 0,
        "{second} round(s) left an officer's weapon over a run in which the policy decided to fire NONE — the trigger is a level nobody lowered"
    );
    assert!(
        !b.cm(crew).runtime.want_attack,
        "the officer is still holding its trigger down with nobody to shoot at"
    );
}

/// **AN OFFICER DOES NOT TAKE COVER FROM ITS OWN GUNSHOT** (wave WPN2e audit,
/// closing carried 279).
///
/// `inf_ecs::cover::under_fire`'s own doc has said *"a responder inside
/// `radius_m` of a place this step's gunfire came from, **that it did not fire
/// itself**"* since wave COV1, and `step_npc_cover` passed it `panic_sources` —
/// which coalesces every LOUD shot in the step, including the responder's own
/// muzzle. So an officer that pulled a trigger stood zero metres from a place
/// gunfire came from and dived behind a wall from itself.
///
/// Wave WPN2e found it, fixed its OWN reader (`d3::engage::note_incoming`) and
/// left the cover half alone because fixing it moves COV1's measured duty
/// cycle. It does, and this audit re-measured it: 242 → 205 steps in cover,
/// 133 → 96 leaned out, 55.0 % → 46.8 % duty. `cov1_gate` is 16/16 green.
///
/// # What it reads
///
/// `NpcCoverReport::under_fire` and the officer's own `MovementMode` on a street
/// where **the only gunfire in the world is the officer's own** — the hero never
/// pulls a trigger, so every source in `panic_sources` came from the muzzle of
/// the unit being asked about.
///
/// **The mutation**: `if false && shooters.contains(&unit)` in `step_npc_cover`
/// — the officer takes cover from itself and both assertions go red.
#[test]
fn an_officer_that_fires_is_not_fired_upon_by_its_own_weapon() {
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 14.0));
    // Something to take cover behind, a metre and a half in front of it — so a
    // failure is "it took cover" and not "it had nowhere to go".
    slab(
        &mut b.world,
        wall_guid(0),
        "Parapet",
        DVec3::new(0.0, 0.45, 12.5),
        Vec3d::new(4.0, 0.45, 0.35),
    );
    b.resync();
    b.civilian(5, DVec3::new(12.0, 0.0, 4.0));
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.arm(crew, "m4a1");
    // **THE HERO NEVER FIRES.** Its trigger is untouched for the whole run, so
    // every loud shot in this world leaves the officer's own barrel.
    let mut officer_rounds = 0usize;
    let mut under_fire = 0usize;
    let mut in_cover = 0usize;
    let mut hero_rounds = 0usize;
    for _ in 0..(6 * engage::ENGAGE_PERIOD_STEPS as usize) {
        b.top_up(crew);
        let r = b.step();
        officer_rounds += r.hits.iter().filter(|h| h.shooter == crew).count();
        hero_rounds += r.hits.iter().filter(|h| h.shooter == HERO).count();
        under_fire += r.npc_cover.under_fire;
        in_cover += r.npc_cover.in_cover;
    }
    println!("\n=== AN OFFICER FIRING AT A HERO THAT NEVER SHOOTS BACK ===");
    println!("  the officer fired {officer_rounds} round(s); the hero fired {hero_rounds}");
    println!(
        "  the cover pass called it under fire on {under_fire} step(s); in cover on {in_cover}"
    );
    assert_eq!(
        hero_rounds, 0,
        "the fixture's hero fired {hero_rounds} rounds, so the sources are not the officer's own"
    );
    assert!(
        officer_rounds > 0,
        "the officer never fired, so there was never a source to mistake for incoming"
    );
    assert_eq!(
        under_fire, 0,
        "an officer was 'under fire' on {under_fire} step(s) in a world where the only gunshots were its own"
    );
    assert_eq!(
        in_cover, 0,
        "an officer took cover from its own gunshot on {in_cover} step(s)"
    );
}

#[test]
fn a_killing_is_filed_against_the_shooter_and_a_corpse_is_not_also_wounded() {
    let mut b = Beat::new();
    let victim = b.civilian(0, DVec3::new(0.0, 0.0, 6.0));
    b.civilian(1, DVec3::new(6.0, 0.0, 3.0));
    // Five hundred joules against a 1 700 J round: the first one is fatal, so
    // the round that kills cannot also be a wounding by arithmetic.
    weapon::give_health(&mut b.world, victim, 500.0);
    b.arm(HERO, "m4a1");
    b.aim(HERO, 0.0, 0.0);
    b.hold_trigger(HERO, true);
    let mut kills = 0u32;
    for _ in 0..90 {
        b.top_up(HERO);
        kills += b.step().kills;
    }
    let acts = inf_ecs::witness::witnessed(&b.world).to_vec();
    let killed: Vec<&WitnessedAct> = acts.iter().filter(|a| a.kind == ActKind::Killed).collect();
    let wounded_before = acts.iter().filter(|a| a.kind == ActKind::Wounded).count();
    println!("=== a killing ===");
    println!(
        "  {kills} kill(s), {} death act(s), {wounded_before} wounding(s), the victim has {:.0} J",
        killed.len(),
        b.health(victim)
    );
    assert!(kills >= 1, "the fixture killed nobody");
    assert!(!killed.is_empty(), "no death act was recorded");
    for a in &killed {
        assert_eq!(
            a.actor, HERO,
            "the killing was filed against {} — the man who was shot",
            a.actor
        );
    }
    assert!(
        b.health(victim) <= 0.0,
        "the victim is not actually dead: {:.0} J",
        b.health(victim)
    );

    // …and now the corpse. Aim DOWN and keep firing.
    // **A SWEEP, not one angle.** The muzzle is at 1.4 m and the corpse's own
    // capsule is somewhere under 0.4 m six metres away, which is about 10 deg —
    // and a ragdoll settles, so a single pitch chosen in advance misses. Two
    // measured attempts at -22 deg put every round in the ground at 3.5 m.
    let mut on_corpse = 0usize;
    for i in 0..600 {
        b.aim(HERO, 0.0, -4.0 - (i % 20) as f64);
        b.top_up(HERO);
        let r = b.step();
        on_corpse += r
            .hits
            .iter()
            .filter(|h| h.target == Some(victim) && h.on_flesh)
            .count();
    }
    let after = inf_ecs::witness::witnessed(&b.world)
        .iter()
        .filter(|a| a.kind == ActKind::Wounded)
        .count();
    println!(
        "  {on_corpse} further rounds reached the body on the floor; woundings {wounded_before} -> {after}"
    );
    assert!(
        on_corpse > 0,
        "not one round reached the corpse, so the refusal below is vacuous — `guid_of_ragdoll_collider` is what makes a body on the floor hittable"
    );
    // The log is a ring of 256 and this run fired far more than that, so the
    // comparison is against ZERO woundings in the current window rather than
    // against the earlier count.
    assert_eq!(
        after, 0,
        "{after} wounding(s) were filed against a man who was already dead"
    );
}

/// **A CAR SHOT AT SPENDS NOTHING** — VEH3c's boundary, recorded rather than
/// fixed.
///
/// The round stops in the chassis (WPN2a made the cast `AllSolid`, so it can see
/// one at all), it names the chassis as its target, it carries joules — and the
/// car has no `Health` and no `Destructible`, so `apply_hit` spends them nowhere.
/// **This arm passes today and is designed to CHANGE when VEH3c gives a chassis
/// health**, which is what makes it a boundary and not a hole.
#[test]
fn a_car_shot_at_stops_the_round_and_spends_nothing() {
    let mut b = Beat::new();
    let chassis = Uuid::from_u128(0x2E00_0900);
    let e = b.world.spawn_with_guid(chassis, "Car", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, 0.7, 8.0);
    b.world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            angular_damping: 0.5,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(1.0, 0.7, 2.4),
            ..Default::default()
        },
    ));
    b.resync();
    b.arm(HERO, "m4a1");
    // Down five degrees: the muzzle is at `MUZZLE_HEIGHT_M` (1.4 m) and the
    // chassis' own roof is at 1.4 m, so a level shot grazes the top of the car
    // and hits the ground behind it. Measured: 0 rounds stopped in the chassis
    // at pitch 0.
    b.aim(HERO, 0.0, -5.0);
    b.hold_trigger(HERO, true);
    let mut on_car: Vec<d3::WeaponHit> = Vec::new();
    for _ in 0..60 {
        let r = b.step();
        on_car.extend(r.hits.iter().filter(|h| h.target == Some(chassis)).cloned());
    }
    println!("=== a car shot at ===");
    println!(
        "  {} rounds stopped in the chassis, the first carrying {:.0} J",
        on_car.len(),
        on_car.first().map(|h| h.energy_j).unwrap_or(0.0)
    );
    assert!(
        !on_car.is_empty(),
        "no round stopped in the car at all — the cast cannot see a dynamic body"
    );
    assert!(
        on_car[0].energy_j > 0.0,
        "the round that stopped in the car carried no energy"
    );
    assert!(!on_car[0].on_flesh, "a car is flesh");
    assert!(
        weapon::health_of(&b.world, chassis).is_none(),
        "the car has a Health component, so this arm is no longer VEH3c's boundary — rewrite it"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (e) DETERMINISM: PIE == SHIPPING, TWO COOKS
// ═════════════════════════════════════════════════════════════════════════════

/// **THE SHOOTOUT COURSE IS THE SAME ON BOTH HOSTS AND ACROSS TWO COOKS.**
///
/// The policy is a **pure function of sim state** and this is what says so: the
/// hero, two officers behind cover, and the same script driven on the editor's
/// preview world and on two independently cooked packs. Every `state_bytes` and
/// every audio command must match, step for step.
///
/// The officers are injected into all three sims by **one function**
/// ([`shootout`]), so a divergence cannot be a difference in the fixtures.
#[test]
fn pie_equals_shipping_and_two_cooks_agree_over_the_shootout_course() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let a = cook_fixture(&tmp.path().join("a"));
    let b = cook_fixture(&tmp.path().join("b"));
    let ta = shootout(pack_sim(&a));
    let tb = shootout(pack_sim(&b));
    let tp = shootout(pie_sim());
    println!("=== the shootout course ===");
    println!(
        "  {} steps traced; hero fired {}, officers fired {}, {} engage rays, {} holds",
        ta.trace.len(),
        ta.hero_shots,
        ta.npc_shots,
        ta.rays,
        ta.holds
    );
    assert!(ta.hero_shots > 0, "the hero fired nothing");
    assert!(
        ta.npc_shots > 0,
        "the police fired nothing, so this course does not test the policy"
    );
    assert!(
        ta.trace.len() > 300,
        "the course traced {} steps, which is not a course",
        ta.trace.len()
    );
    assert_eq!(
        (ta.trace.len(), ta.trace.len()),
        (tb.trace.len(), tp.trace.len()),
        "the three traces are not the same length, so `zip` would compare a prefix and call it agreement"
    );
    for (i, (x, y)) in ta.trace.iter().zip(tb.trace.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: two independent cooks diverged");
    }
    for (i, (x, y)) in ta.trace.iter().zip(tp.trace.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: PIE and shipping diverged");
    }
    assert_eq!(ta.audio, tb.audio, "two cooks made different noises");
    assert_eq!(ta.audio, tp.audio, "PIE and shipping made different noises");
    assert_eq!(
        (ta.hero_shots, ta.npc_shots, ta.rays, ta.holds),
        (tb.hero_shots, tb.npc_shots, tb.rays, tb.holds),
        "two cooks did not do the same things"
    );
    assert_eq!(
        (ta.hero_shots, ta.npc_shots, ta.rays, ta.holds),
        (tp.hero_shots, tp.npc_shots, tp.rays, tp.holds),
        "PIE and shipping did not do the same things"
    );
}

struct Course {
    trace: Vec<Vec<u8>>,
    audio: Vec<String>,
    hero_shots: u32,
    npc_shots: u32,
    rays: usize,
    holds: usize,
}

/// The course: put two officers on the fixture, open a file on the hero, and let
/// the hero hold its trigger while the police answer.
///
/// **One function for all three sims** — the arm's whole value is that the only
/// difference between the runs is which host is stepping.
fn shootout(mut sim: inf_player::runtime_sim::RuntimeSim) -> Course {
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    let mut trace = Vec::new();
    for _ in 0..20 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        trace.push(sim.state_bytes());
    }
    let before = sim.audio_command_log().len();
    // The hero's own position, so the officers stand a fixed distance from
    // wherever the fixture spawned it.
    let at = sim
        .world()
        .entity_of(hero)
        .and_then(|e| sim.world().world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .expect("the fixture has a hero");
    {
        let w = sim.world_mut();
        // The clock, hand-installed — the gameplay fixture has no streets.
        w.world_mut()
            .insert_resource(inf_ecs::traffic::TrafficPopulationRes {
                hand_installed: true,
                ..Default::default()
            });
        // **THE FLEET IS STAMPED WITH THE LEVEL'S OWN BLOCK STAMP**, or
        // `sync_fleet` rebuilds it from the level's blocks on the very next step
        // and this fixture has none. Measured without it: the fleet went 2 -> 0
        // on step 1, `park` took both crews off duty, and the course ran 400
        // steps with nobody on the roster.
        let mut fleet = FleetRes {
            stamp: inf_ecs::traffic::block_stamp(w),
            ..FleetRes::default()
        };
        let mut res = dispatch::DispatchRes::default();
        // …and the units are ON SCENE AT AN INCIDENT. `work_the_scene` answers
        // "finished" for a run with no incident at all, so an on-scene unit
        // without one is sent home on its first step. `SECURE_S` is 10 s and the
        // course is 7, so the two officers stand their post for the whole of it.
        let scene = Uuid::from_u128(0x2E00_0A00);
        res.incidents.insert(
            scene,
            dispatch::Incident {
                kind: dispatch::IncidentKind::Crime { severity: 2 },
                at,
                state: dispatch::IncidentState::OnScene,
                opened_step: 0,
                unit: None,
                resolved_step: None,
            },
        );
        for i in 0..2usize {
            let chassis = chassis_guid(i);
            fleet.units.insert(
                chassis,
                FleetUnit {
                    kind: UnitKind::Police,
                    station: Uuid::nil(),
                    home: at,
                    home_yaw_deg: 0.0,
                },
            );
            res.runs.insert(
                chassis,
                UnitRun {
                    state: UnitState::OnScene,
                    incident: Some(scene),
                    since_step: 0,
                    path: None,
                },
            );
            let crew = dispatch::crew_guid(chassis);
            // **BEHIND THE HERO, and that is a measurement rather than a
            // taste.** The `phase30-gameplay` fixture's own furniture stands in
            // front of it: two officers twelve metres along +Z had **712 of 712
            // engage rays blocked** over a four-hundred-step course and returned
            // nothing, which is the law working and a course testing nothing.
            // Ten metres along -Z is clear (0 blocked).
            stand(
                w,
                crew,
                "Officer",
                at + DVec3::new(i as f64 * 4.0 - 2.0, 0.0, -10.0),
                false,
            );
        }
        // **A WITNESS**, so the hero's own gunfire keeps its file warm.
        // `TRAIL_STALE_STEPS` is 180 and the course is 400: without somebody who
        // can see the hero shooting, the trail goes cold two thirds of the way
        // through and the officers stop engaging (measured: 180 engage rays and
        // zero rounds returned). Twelve metres to one side, which is well
        // outside the officers' own firing cone.
        let watcher = Uuid::from_u128(0x2E00_0B00);
        stand(
            w,
            watcher,
            "Bystander",
            at + DVec3::new(12.0, 0.0, 4.0),
            false,
        );
        let mut pop = w
            .world_mut()
            .remove_resource::<inf_ecs::crowd::CrowdPopulationRes>()
            .unwrap_or_default();
        pop.hand_installed = true;
        pop.records.insert(
            watcher,
            inf_ecs::crowd::CrowdRecord::standing(
                inf_ecs::crowd::CrowdArchetype::humanoid(None, None, None),
                at + DVec3::new(12.0, 0.0, 4.0),
            ),
        );
        w.world_mut().insert_resource(pop);
        w.world_mut().insert_resource(fleet);
        w.world_mut().insert_resource(res);
        w.mark_dirty();
        w.reindex_guids();
        w.propagate();
        for i in 0..2usize {
            let crew = dispatch::crew_guid(chassis_guid(i));
            dispatch::set_responder(w, crew, true);
            weapon::give_health(w, crew, weapon::DEFAULT_VITALITY_J);
            // The rifle, which is what `crew_weapon_ids` issues at the rung
            // this course sets.
            assert_eq!(item::give(w, crew, "m4a1", 1), 0);
            assert!(d3::gameplay::equip_weapon(w, crew, "m4a1"));
        }
        // The file, through `report_act` — the same door the witness pass uses.
        let mut heat = 0u32;
        while Response::for_heat(heat) < Response::Swat {
            heat = crime::report_act(
                w,
                &WitnessedAct {
                    kind: ActKind::Shot,
                    actor: hero,
                    at,
                    step: 0,
                    observers: vec![chassis_guid(9)],
                    actor_look: inf_ecs::witness::look_digest(w, hero),
                    actor_vehicle: None,
                    heard_by: 0,
                },
                None,
            )
            .expect("the file opens");
        }
        assert_eq!(item::give(w, hero, "m4a1", 1), 0);
        assert!(d3::gameplay::equip_weapon(w, hero, "m4a1"));
    }
    let (mut hero_shots, mut npc_shots, mut rays, mut holds) = (0u32, 0u32, 0usize, 0usize);
    let mut state = inf_input::InputState::new(inf_input::default_map());
    for i in 0..400 {
        let events: Vec<inf_input::InputEvent> = vec![inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: (i / 30) % 2 == 0,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
        let rep = sim.gameplay();
        for h in &rep.hits {
            if h.shooter == hero {
                hero_shots += u32::from(h.loud);
            } else {
                npc_shots += u32::from(h.loud);
            }
        }
        rays += rep.engage.rays;
        holds += rep.engage.friendly_holds + rep.engage.civilian_holds + rep.engage.cadence_holds;
        trace.push(sim.state_bytes());
        // The clock, advanced exactly as `step_traffic` would.
        if let Some(mut pop) = sim
            .world_mut()
            .world_mut()
            .get_resource_mut::<inf_ecs::traffic::TrafficPopulationRes>()
        {
            pop.steps += 1;
        }
    }
    assert_eq!(sim.dropped_audio_commands(), 0);
    let audio: Vec<String> = sim.audio_command_log()[before..]
        .iter()
        .map(|c| format!("{c:?}"))
        .collect();
    Course {
        trace,
        audio,
        hero_shots,
        npc_shots,
        rays,
        holds,
    }
}

/// **A QUIET LEVEL FOLDS NOTHING THIS WAVE ADDED** — the byte-identity half.
///
/// The policy's whole memory is a bevy resource ([`inf_ecs::engage::EngageRes`])
/// and the blind-fire clock is a `f64` that is zero until somebody fires without
/// looking, so a level where nobody is wanted steps exactly the bytes it stepped
/// before this wave.
#[test]
fn a_quiet_level_folds_nothing_this_wave_added() {
    let mut b = Beat::new();
    let crew = b.officer(0, DVec3::new(0.0, 0.0, 12.0));
    b.arm(crew, "glock_17");
    for _ in 0..40 {
        b.step();
    }
    assert!(
        engage::engage_of(&b.world).is_none(),
        "a level where nobody is wanted grew an engagement ledger"
    );
    assert_eq!(engage::engaged_units(&b.world), 0);
    assert!(
        inf_ecs::crime::profile_state_bytes(&b.world).is_empty(),
        "a level where nobody is wanted folds a profile"
    );
    assert_eq!(
        b.cm(crew).runtime.blind_fire_s,
        0.0,
        "an officer that has never fired blind carries a clock"
    );
    // …and the ledger is DROPPED again when the last file closes, so a session
    // that went quiet does not carry an engagement into the next thing that
    // happens.
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.run(30);
    assert!(
        engage::engage_of(&b.world).is_some(),
        "an engaged street grew no ledger"
    );
    inf_ecs::crime::clear_crime(&mut b.world);
    b.run(2);
    assert_eq!(
        engage::engaged_units(&b.world),
        0,
        "the ledger still names a target after the last file closed"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (f) COST
// ═════════════════════════════════════════════════════════════════════════════

/// **EIGHT SHOOTERS, TWO HUNDRED ROUNDS IN FLIGHT AND SIXTY-FOUR CASINGS ARE
/// INSIDE `WEAPON_STEP_BUDGET_MS`.**
///
/// A CONTROL step and the measured step in the **same process**, so the
/// difference is the firefight and not a cold cache — the WPN2d audit's own law,
/// which caught a budget arm timing a fixture's first step.
#[test]
fn a_firefight_is_inside_the_weapon_budget() {
    let mut b = Beat::new();
    // Eight officers in an arc, all engaged, all firing.
    for i in 0..8usize {
        let a = i as f64 * 0.6;
        let c = b.officer(i, DVec3::new(a.sin() * 18.0, 0.0, a.cos() * 18.0 + 6.0));
        b.arm(c, "m4a1");
    }
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    b.arm(HERO, "m4a1");
    b.aim(HERO, 0.0, 8.0);
    b.hold_trigger(HERO, true);
    // Fill the pool and the brass.
    let mut rounds = 0usize;
    let mut casings = 0usize;
    for _ in 0..240 {
        b.top_up(HERO);
        for c in b.officers.clone() {
            b.top_up(c);
        }
        b.step();
        rounds = rounds.max(
            inf_ecs::ballistics::round_pool(&b.world)
                .map(|p| p.rounds.len())
                .unwrap_or(0),
        );
        casings = casings.max(inf_ecs::casing::casings_live(&b.world));
    }
    // The control: the same world with every trigger released.
    b.hold_trigger(HERO, false);
    for c in b.officers.clone() {
        let e = b.world.entity_of(c).expect("an officer");
        if let Some(mut cm) = b.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.want_attack = false;
        }
    }
    // **THE CONTROL IS NOT A COLD FIRST STEP** (the WPN2d audit's law): the
    // world has stepped 240 times already, and the released-trigger step below
    // is run three times with only the last one timed.
    let mut control_us = 0.0;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let _ = b.step();
        control_us = t.elapsed().as_secs_f64() * 1e6;
    }
    b.hold_trigger(HERO, true);
    b.top_up(HERO);
    for c in b.officers.clone() {
        b.top_up(c);
        let e = b.world.entity_of(c).expect("an officer");
        if let Some(mut cm) = b.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.want_attack = true;
        }
    }
    let t1 = std::time::Instant::now();
    let rep = b.step();
    let step_us = t1.elapsed().as_secs_f64() * 1e6;
    let cost_us = (step_us - control_us).max(0.0);
    println!("=== the firefight's cost ===");
    println!(
        "  {} shooters, {rounds} rounds in flight at the peak, {casings} casings",
        b.officers.len() + 1
    );
    println!(
        "  {step_us:.1} us for the firing step, {control_us:.1} us for the same step with the triggers released, {cost_us:.1} us for the firefight ({} shot rays, {} engage rays; budget {:.1} ms)",
        rep.rounds.shot_rays,
        rep.engage.rays,
        inf_player::budget::WEAPON_STEP_BUDGET_MS
    );
    assert!(
        b.officers.len() >= 8,
        "the fixture has {} shooters",
        b.officers.len()
    );
    assert!(
        cost_us <= inf_player::budget::WEAPON_STEP_BUDGET_MS * 1000.0,
        "a firefight cost {cost_us:.1} us over the same step without it ({step_us:.1} vs {control_us:.1}) against a {:.1} ms budget",
        inf_player::budget::WEAPON_STEP_BUDGET_MS
    );
}

/// **THE POLICY IS INSIDE `NPC_STEP_BUDGET_MS` AT A THOUSAND AGENTS.**
///
/// A thousand crowd records, one open file, and the same
/// control-then-measure-in-one-process rule. The number that matters is the
/// DIFFERENCE the policy makes to a step that already has a thousand agents in
/// it, which is what a step budget is about.
#[test]
fn the_firing_policy_is_inside_the_npc_budget_at_a_thousand_agents() {
    let mut b = Beat::new();
    for i in 0..8usize {
        let a = i as f64 * 0.6;
        let c = b.officer(i, DVec3::new(a.sin() * 20.0, 0.0, a.cos() * 20.0 + 8.0));
        b.arm(c, "glock_17");
    }
    // A thousand pedestrians, as population records — the set the discipline
    // rule's cone walk visits.
    let mut pop = b
        .world
        .world_mut()
        .remove_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        .unwrap_or_default();
    pop.hand_installed = true;
    for i in 0..1000usize {
        let a = i as f64 * 0.0063;
        pop.records.insert(
            Uuid::from_u128(0x2E00_5000 + i as u128),
            inf_ecs::crowd::CrowdRecord::standing(
                inf_ecs::crowd::CrowdArchetype::humanoid(None, None, None),
                DVec3::new(
                    a.cos() * (10.0 + i as f64 * 0.05),
                    0.0,
                    a.sin() * (10.0 + i as f64 * 0.05),
                ),
            ),
        );
    }
    let agents = pop.records.len();
    b.world.world_mut().insert_resource(pop);
    b.resync();
    // **WARM FIRST** (the WPN2d audit's law): a fixture's first step builds
    // every broad-phase structure the next thousand reuse, and timing it as a
    // control makes the measured step look free. Ten steps, then the control is
    // the LAST of three.
    for _ in 0..10 {
        b.step();
    }
    // The CONTROL: the same thousand agents, nobody wanted, so the policy runs
    // its early return.
    let mut control_us = 0.0;
    let mut control = b.step();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        control = b.step();
        control_us = t0.elapsed().as_secs_f64() * 1e6;
    }
    assert_eq!(control.engage.rays, 0, "the control is already engaged");
    // …and now with a file open.
    b.file_on_hero(Response::Swat, DVec3::ZERO);
    // A few steps to settle the ledger, then the measurement — the LAST of
    // three, on the control's own rule.
    let mut hot = b.step();
    let mut hot_us = 0.0;
    for _ in 0..3 {
        let t1 = std::time::Instant::now();
        hot = b.step();
        hot_us = t1.elapsed().as_secs_f64() * 1e6;
    }
    let cost_us = (hot_us - control_us).max(0.0);
    println!("=== the policy at a thousand agents ===");
    println!(
        "  {agents} crowd records, {} officers: control {control_us:.1} us, engaged {hot_us:.1} us, the policy {cost_us:.1} us ({} rays, {} aimed; budget {:.1} ms)",
        b.officers.len(),
        hot.engage.rays,
        hot.engage.aimed,
        inf_player::budget::NPC_STEP_BUDGET_MS
    );
    assert_eq!(agents, 1000);
    assert!(
        hot.engage.rays > 0,
        "the measured step ran no engage rays, so it is a second control"
    );
    assert!(
        cost_us <= inf_player::budget::NPC_STEP_BUDGET_MS * 1000.0,
        "the firing policy cost {cost_us:.1} us over the same step without it ({hot_us:.1} vs {control_us:.1}) against a {:.1} ms budget",
        inf_player::budget::NPC_STEP_BUDGET_MS
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (g) THE PARITY MEMO
// ═════════════════════════════════════════════════════════════════════════════

/// **EVERY ROW OF THE PARITY MEMO CITES AN ARM THAT EXISTS.**
///
/// `docs/memos/gunplay-parity.md` claims one checklist row per feature of the
/// research doc, and each row names the file and the test function that proves
/// it. A row citing an arm that does not exist is a claim with nothing behind
/// it, and this is the arm that finds one.
///
/// It reads the memo's own table, extracts every `` `file::name` `` pair, and
/// looks each one up in the tree. **Mutation → red:** renaming any cited arm.
#[test]
fn every_row_of_the_parity_memo_cites_an_arm_that_exists() {
    let memo_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/memos/gunplay-parity.md");
    let memo = std::fs::read_to_string(&memo_path)
        .unwrap_or_else(|e| panic!("the parity memo is not on disk ({e})"));
    // Every `arm: <file>::<fn>` citation in the memo.
    let mut cites: Vec<(String, String)> = Vec::new();
    for line in memo.lines() {
        for tok in line.split('`') {
            let Some((file, name)) = tok.split_once("::") else {
                continue;
            };
            // **A CITATION IS A TEST FILE AND A TEST NAME**, and nothing else.
            // The memo is full of module paths (`inf_ecs::engage`,
            // `inf_anim::BLIND_FIRE_S`) and none of them is a claim about an arm:
            // a citation names a file ending in `_gate` or `_3d` and a function
            // whose name is a sentence, which in this repository means at least
            // four words.
            if !file.ends_with("_gate") && !file.ends_with("_3d") {
                continue;
            }
            if name.matches('_').count() < 3
                || !name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                continue;
            }
            cites.push((file.to_string(), name.to_string()));
        }
    }
    cites.sort();
    cites.dedup();
    assert!(
        cites.len() >= 8,
        "the memo cites only {} arms, which is not a checklist",
        cites.len()
    );
    let roots = [
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/inf-physics/tests"),
    ];
    let mut missing: Vec<String> = Vec::new();
    for (file, name) in &cites {
        let mut found = false;
        for root in &roots {
            let p = root.join(format!("{file}.rs"));
            if let Ok(text) = std::fs::read_to_string(&p) {
                if text.contains(&format!("fn {name}(")) {
                    found = true;
                    break;
                }
            }
        }
        if !found {
            missing.push(format!("{file}::{name}"));
        }
    }
    println!(
        "the parity memo cites {} arms; {} could not be found",
        cites.len(),
        missing.len()
    );
    assert!(
        missing.is_empty(),
        "the parity memo cites arms that do not exist: {missing:?}"
    );
    // …and it says, in so many words, what parity is NOT.
    for must in ["PAR2", "VEH3c", "parity is NOT yet"] {
        assert!(
            memo.contains(must),
            "the parity memo never mentions {must:?} — the honest half is missing"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// (h) THE ISLAND'S OWN CHAIN, LINK BY LINK
// ═════════════════════════════════════════════════════════════════════════════

/// **WHERE THE SHIPPED ISLAND'S RESPONSE CHAIN ACTUALLY STOPS** — the arm the
/// demo loop's own failure asked for.
///
/// The session this wave filmed pressed the trigger ten times in a street and
/// waited **150 s** for `engaged` to move. It never did, and `hero.csv` cannot
/// say which link broke: it carries what the POLICE are doing and nothing about
/// whether anybody saw the crime, whether a file was opened, or whether a unit
/// was ever sent. This arm boots the same island and prints every link:
///
/// 1. **witnessed** — `witness::witnessed()` after the shots;
/// 2. **filed** — `crime::wanted()` and the heat on the hero's file;
/// 3. **the fleet** — how many police units the level derived at all;
/// 4. **dispatched** — `DispatchRes::assigned` / `unanswered`;
/// 5. **arrived** — units `OnScene`;
/// 6. **armed** — crews carrying a weapon;
/// 7. **engaged** — `engage::engaged_units`.
///
/// It **asserts the two links this wave owns** — the player's trigger works on
/// the shipped island, and a witness two metres away opens a file — and PRINTS
/// the rest, deliberately: a fleet's distance from a spawn and a crowd's density
/// at it are level-design facts, and an arm that went red on them would be a gate
/// failing about where somebody put a building.
///
/// # What it found, and it is why the wave's own shootout frames do not exist
///
/// | link | the island | the control |
/// |---|---|---|
/// | the hero fires | **17 rounds** | 10 more |
/// | acts recorded | 17 | 10 |
/// | …with an **OBSERVER** | **0** | **10** |
/// | the nearest crowd agent | **117 m** | 2 m |
/// | files open | **0**, heat 0 | **1**, heat **20** (`swat`) |
/// | police in the fleet | 3 | 3 |
/// | assignments | 2 | 3 |
/// | units **on scene** | 0 | **0** after three minutes |
/// | the nearest responder got to | — | **93 m** |
/// | units engaged | 0 | **0** |
///
/// Two facts, both about the island and neither about the policy:
///
/// 1. **a gunshot on the showcase island is witnessed by nobody**, because the
///    nearest crowd agent is 117 m away and the witness ray does not cross that
///    much terrain. Put one pedestrian two metres away and the file opens at
///    heat 20 on the first burst;
/// 2. **a dispatched unit does not arrive.** With a `Swat`-grade file open, three
///    assignments were made and the nearest responder closed to **93 m** and
///    stopped — over three minutes, and `ON_SCENE_M` is 12. It is never on
///    scene, so it is never issued a weapon and never engages.
///
/// Local-only: the island is not in this repository, so CI skips it.
#[test]
fn the_islands_own_chain_from_a_gunshot_to_an_engaged_officer() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = island_sim(&content);
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    // Let the level stream and the society settle, exactly as the demo loop's
    // own 20 s does.
    for _ in 0..1200 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let at = sim
        .world()
        .entity_of(hero)
        .and_then(|e| sim.world().world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .expect("the hero has a place");
    // **THE CATALOGUE, MERGED HERE**, and the reason is a measurement this arm
    // made on its first run: a LOOSELY loaded island has **0 item definitions**
    // after 1 200 steps. The island's own `Quartermaster` Blueprint defines the
    // registry on `BeginPlay` and reaches a real preview (the demo loop's
    // SIDEARM leg picks the `glock_17` off the kerb with no environment
    // variable), but `inf_player::level::load` + `sim_from_built` is not that
    // path. Without this the hero fires its FISTS, which are quiet, raise no
    // act, and make every link below read zero for the wrong reason.
    //
    // `merge_toml` is the same door `item.define` dispatches to.
    {
        let defs = inf_ecs::item::item_defs_mut(sim.world_mut());
        if defs.is_empty() {
            defs.merge_toml(weapon::WEAPON_REGISTRY_TOML)
                .expect("the shipped registry parses");
        }
    }
    let rows = inf_ecs::item::item_defs(sim.world())
        .map(|d| d.len())
        .unwrap_or(0);
    let left = item::give(sim.world_mut(), hero, "glock_17", 1);
    let equipped = d3::gameplay::equip_weapon(sim.world_mut(), hero, "glock_17");
    println!(
        "  0. the level's catalogue     : {rows} rows; give left {left} over, equipped {equipped}"
    );
    // Fire, through the shipped input path, for ten seconds.
    let mut state = inf_input::InputState::new(inf_input::default_map());
    let mut shots = 0u32;
    for i in 0..600 {
        let events: Vec<inf_input::InputEvent> = vec![inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: (i / 15) % 2 == 0,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
        shots += sim.gameplay().shots;
    }
    let acts = inf_ecs::witness::witnessed(sim.world()).to_vec();
    let witnessed = acts.len();
    let with_observers = acts.iter().filter(|a| !a.observers.is_empty()).count();
    // **THE EARS CHANNEL** (wave WPN2e audit). The link that was missing: on this
    // island a gunshot is SEEN by nobody and HEARD by the same crowd that could
    // not see it.
    let with_hearers = acts.iter().filter(|a| a.heard_by > 0).count();
    let clock = inf_ecs::traffic::steps(sim.world());
    let candidates =
        inf_ecs::witness::candidates_near(sim.world(), at, d3::gameplay::WITNESS_RADIUS_M);
    let near = candidates.len();
    let mut ranges: Vec<String> = candidates
        .iter()
        .map(|(_, p)| format!("{:.0} m", (*p - at).length()))
        .collect();
    ranges.truncate(8);
    let kinds: std::collections::BTreeSet<&str> = acts.iter().map(|a| a.kind.name()).collect();
    // …and then wait, driving nothing, for the response.
    let mut peak_assigned = 0u64;
    let mut peak_on_scene = 0usize;
    let mut peak_engaged = 0usize;
    let mut peak_armed = 0usize;
    for _ in 0..3600 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        let (assigned, on_scene) = dispatch::dispatch_of(sim.world())
            .map(|r| {
                (
                    r.assigned,
                    r.runs
                        .values()
                        .filter(|u| u.state == UnitState::OnScene)
                        .count(),
                )
            })
            .unwrap_or((0, 0));
        peak_assigned = peak_assigned.max(assigned);
        peak_on_scene = peak_on_scene.max(on_scene);
        peak_engaged = peak_engaged.max(engage::engaged_units(sim.world()));
        let armed = dispatch::responders(sim.world())
            .into_iter()
            .filter(|g| weapon::equipped_def(sim.world(), *g).is_some())
            .count();
        peak_armed = peak_armed.max(armed);
    }
    let police = dispatch::fleet_of(sim.world())
        .map(|f| {
            f.units
                .values()
                .filter(|u| u.kind == UnitKind::Police)
                .count()
        })
        .unwrap_or(0);
    let unanswered = dispatch::dispatch_of(sim.world())
        .map(|r| r.unanswered)
        .unwrap_or(0);
    let heat = crime::heat_of(sim.world(), hero);
    println!(
        "\n=== THE ISLAND'S CHAIN, from ({:.0}, {:.0}) ===",
        at.x, at.z
    );
    println!("  {:<29}: {shots} rounds", "1. the hero fired");
    println!(
        "  {:<29}: {witnessed} ({kinds:?}); {with_observers} of them have an OBSERVER, {with_hearers} a HEARER",
        "2. acts recorded"
    );
    println!(
        "     the crowd within {:.0} m : {near} agents at {ranges:?}; the traffic clock reads {clock}",
        d3::gameplay::WITNESS_RADIUS_M
    );
    println!(
        "  {:<29}: {} (the hero's heat {heat}, rung {})",
        "3. files open",
        crime::wanted(sim.world()).len(),
        Response::for_heat(heat).name()
    );
    println!("  {:<29}: {police}", "4. police units in the fleet");
    println!(
        "  {:<29}: {peak_assigned} (unanswered {unanswered})",
        "5. assignments made"
    );
    println!("  {:<29}: {peak_on_scene}", "6. units on scene");
    println!("  {:<29}: {peak_armed}", "7. crews carrying a weapon");
    println!("  {:<29}: {peak_engaged}", "8. units ENGAGED");
    // ── THE CONTROL, and it is what tells a CONTENT fact from an ENGINE one.
    //
    // Put a pedestrian eight metres away — a distance the island's own crowd
    // simply does not happen to stand at — and fire again. If a file opens, the
    // chain works and what the island lacks is somebody close enough to see;
    // if it does not, the witness pass itself is broken on this level.
    let watcher = Uuid::from_u128(0x2E00_0C00);
    {
        let w = sim.world_mut();
        let mut pop = w
            .world_mut()
            .remove_resource::<inf_ecs::crowd::CrowdPopulationRes>()
            .unwrap_or_default();
        pop.hand_installed = true;
        pop.records.insert(
            watcher,
            inf_ecs::crowd::CrowdRecord::standing(
                inf_ecs::crowd::CrowdArchetype::humanoid(None, None, None),
                at + DVec3::new(2.0, -1.0, 0.0),
            ),
        );
        w.world_mut().insert_resource(pop);
    }
    // **AND A FULL MAGAZINE**, or the control is vacuous: a `glock_17` holds
    // seventeen rounds and the first burst fired exactly seventeen. Nothing
    // reloads an idle hero, so without this the control's own trigger produces
    // no gunshot at all and the zero below would mean "nobody fired" rather than
    // "nobody saw".
    let mut state2 = inf_input::InputState::new(inf_input::default_map());
    let mut control_shots = 0u32;
    for i in 0..300 {
        {
            let w = sim.world_mut();
            if let Some((_, def)) = weapon::equipped_def(w, hero) {
                if let Some(e) = w.entity_of(hero) {
                    if let Some(mut st) = w.world_mut().get_mut::<weapon::WeaponState>(e) {
                        st.magazine = def.magazine;
                    }
                }
            }
        }
        let events: Vec<inf_input::InputEvent> = vec![inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: (i / 15) % 2 == 0,
        }];
        state2.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state2, DT));
        control_shots += sim.gameplay().shots;
    }
    for _ in 0..240 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let seen = inf_ecs::witness::witnessed(sim.world())
        .iter()
        .filter(|a| !a.observers.is_empty())
        .count();
    let close = inf_ecs::witness::candidates_near(sim.world(), at, 20.0).len();
    println!(
        "  9. THE CONTROL, one pedestrian two metres away: {control_shots} more rounds, {close} candidate(s) inside 20 m, {seen} act(s) with an observer, {} file(s) open, heat {}",
        crime::wanted(sim.world()).len(),
        crime::heat_of(sim.world(), hero)
    );

    // ── AND THEN THE WHOLE CHAIN, from a file that really is open.
    //
    // Sixty seconds of the dispatcher with a `Swat`-grade file on the hero: does
    // a unit leave, arrive, get issued a weapon, and point it?
    let mut a2 = 0u64;
    let mut on2 = 0usize;
    let mut armed2 = 0usize;
    let mut eng2 = 0usize;
    let mut closest = f64::INFINITY;
    let mut engaged_at: Option<f64> = None;
    // **Three minutes**, because sixty seconds was measured and was not enough:
    // the nearest responder was still 93 m out when the first cut of this arm
    // stopped counting.
    for step in 0..10_800 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        if let Some(res) = dispatch::dispatch_of(sim.world()) {
            a2 = a2.max(res.assigned);
            on2 = on2.max(
                res.runs
                    .values()
                    .filter(|u| u.state == UnitState::OnScene)
                    .count(),
            );
        }
        for crew in dispatch::responders(sim.world()) {
            if weapon::equipped_def(sim.world(), crew).is_some() {
                armed2 = armed2.max(1);
            }
            if let Some(e) = sim.world().entity_of(crew) {
                if let Some(t) = sim.world().world().get::<Transform>(e) {
                    closest = closest.min((t.translation.to_dvec3() - at).length());
                }
            }
        }
        let now = engage::engaged_units(sim.world());
        if now > 0 && engaged_at.is_none() {
            engaged_at = Some(f64::from(step) / 60.0);
        }
        eng2 = eng2.max(now);
    }
    println!(
        " 10. three minutes with a WARM file: {a2} assignment(s), {on2} on scene, {armed2} armed crew, {eng2} ENGAGED; the nearest responder got within {closest:.0} m; first engaged at {}",
        engaged_at
            .map(|t| format!("{t:.1} s"))
            .unwrap_or_else(|| "never".to_string())
    );

    // **The one assertion**: the player's own trigger works on the shipped
    // island. Everything below it is printed, because a police station's
    // distance from a spawn is a level-design fact and a gate that failed on it
    // would be red about where somebody put a building.
    assert!(
        shots > 0,
        "the island's own sidearm fired nothing through the shipped input path"
    );
    // **LINK 1, CLOSED** (wave WPN2e audit). The wave measured seventeen gunshots
    // at the spawn with ZERO observers and zero files, and carried it as a
    // level-design fact. It is not: a gunshot does not need to be seen. With the
    // ears channel the same seventeen shots are HEARD by the same crowd that
    // could not see them, and the town opens a file on the player's own gunfire.
    //
    // **The mutation**: return 0.0 from `weapon::audible_radius_m` and both of
    // these go red, with the observer count still at zero beside them.
    assert!(
        with_hearers > 0,
        "{witnessed} gunshots at the island's own spawn were heard by nobody - the ears channel is not reaching the crowd"
    );
    assert!(
        heat > 0,
        "{with_hearers} HEARD gunshots opened no file on the shipped island"
    );
    // …and the CONTROL, which is the half that tells a content fact from an
    // engine one: with somebody standing close enough to see, the shipped
    // island's own chain gets as far as a warm file at the top rung.
    assert!(
        control_shots > 0,
        "the control fired nothing — a `glock_17` holds 17 rounds and the first burst used all of them"
    );
    assert!(
        seen > 0,
        "a pedestrian two metres from {control_shots} gunshots witnessed none of them — the witness pass is broken on this level, not the island's crowd density"
    );
    assert!(
        crime::heat_of(sim.world(), hero) > 0,
        "{seen} witnessed gunshots opened no file on the shipped island"
    );
}

/// **THE ISLAND'S HERO CARRIES ITS WEAPON IN ITS HAND** (wave WPN2e audit,
/// closing the WPN2d audit's carried 266 and this arc's carried 280).
///
/// # What it reads
///
/// The world, three ways, on the SHIPPED island with the SHIPPED rig:
///
/// 1. the hero's own evaluated pose publishes a `hand_r` socket
///    ([`inf_ecs::pose::EvaluatedPose::socket`]) -- which is what
///    `d3::gameplay`'s muzzle rule requires before it will read a weapon's
///    barrel at all;
/// 2. the equipped weapon ENTITY's own `GlobalTransform` is at that socket and
///    not at the character's origin -- measured as two distances, so a weapon
///    that had simply moved somewhere else would fail as loudly as one that had
///    not moved at all;
/// 3. `GameplayReport::muzzles_without_a_socket`, the tripwire wave SK1b minted
///    for exactly this and which nothing on the island had ever read, is ZERO
///    over the run.
///
/// # Why it is an AUDIT arm and not the wave's
///
/// The wave shipped with the island's hero drawing its Glock at its PELVIS, and
/// its report says so (carried 280) -- *"one re-import closes it"*. It does not:
/// a re-import fixes the rigs somebody re-imports and leaves every `.inf_skel`
/// already on disk exactly as broken. Measured on this machine before the fix:
/// **0 of the island's 104 `.inf_skel` files publish a socket table**, the
/// hero's `Starter.inf_skel` among them.
///
/// What closes it is `inf_anim::asset`'s own load-time door -- `migrate` derives
/// the table for a rig that authors none, inside `inf_asset::decode`, which is
/// the one door BOTH hosts read a `.inf_skel` through. This arm is that door
/// seen from the far end of the engine.
///
/// **The mutation**: delete the `derive_sockets` call in
/// `SkeletonAsset::migrate` and every assertion below goes red -- the socket is
/// `None`, the weapon sits at the character origin, and the muzzle tripwire
/// counts every shot.
#[test]
fn the_islands_own_hero_carries_its_weapon_in_its_hand() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    // -- FIRST, the rigs on disk, read through the shipped decode door.
    let mut rigs = 0usize;
    let mut with_hand = 0usize;
    let mut authored = 0usize;
    for entry in walk_skeletons(&content) {
        let Ok(bytes) = std::fs::read(&entry) else {
            continue;
        };
        let Ok(rig) = inf_asset::decode::<inf_anim::SkeletonAsset>(&bytes) else {
            continue;
        };
        rigs += 1;
        if inf_anim::sockets::find_socket(&rig.sockets, "hand_r").is_some() {
            with_hand += 1;
        }
        if rig.skeleton.joints().iter().any(|j| j.name == "hand_r") {
            authored += 1;
        }
    }
    println!("\n=== THE ISLAND'S RIGS, THROUGH `inf_asset::decode` ===");
    println!(
        "  {rigs} `.inf_skel` decoded; {authored} have a `hand_r` JOINT; {with_hand} publish a `hand_r` SOCKET"
    );
    assert!(rigs > 0, "the island has no rigs at all");
    assert_eq!(
        with_hand,
        authored,
        "{} of {authored} island rigs with a `hand_r` joint publish no `hand_r` socket - a weapon on one of them draws at the pelvis",
        authored - with_hand
    );

    // -- THEN the world: the hero, its pose, and the weapon entity.
    let mut sim = island_sim(&content);
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..600 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    {
        let defs = inf_ecs::item::item_defs_mut(sim.world_mut());
        if defs.is_empty() {
            defs.merge_toml(weapon::WEAPON_REGISTRY_TOML)
                .expect("the shipped registry parses");
        }
    }
    item::give(sim.world_mut(), hero, "glock_17", 1);
    assert!(
        d3::gameplay::equip_weapon(sim.world_mut(), hero, "glock_17"),
        "the island's own sidearm would not equip"
    );
    // Enough steps for the weapon entity to be spawned, posed and attached.
    let mut no_socket = 0u32;
    for _ in 0..120 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        no_socket += sim.gameplay().muzzles_without_a_socket;
    }
    let socket = inf_ecs::pose::evaluated_pose(sim.world(), hero)
        .and_then(|p| p.socket(d3::gameplay::WEAPON_SOCKET));
    let origin = sim
        .world()
        .entity_of(hero)
        .and_then(|e| sim.world().world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .expect("the hero has a place");
    let weapon_guid = d3::gameplay::equipped_weapon_guid(hero);
    let weapon_at = sim
        .world()
        .entity_of(weapon_guid)
        .and_then(|e| {
            sim.world()
                .world()
                .get::<inf_ecs::components::GlobalTransform>(e)
        })
        .map(|g| g.0.transform_point3(DVec3::ZERO))
        .expect("the equipped weapon is an entity");
    // **The same composition `inf_ecs::attach::update_attachments` makes** --
    // `pose::model_to_world` (the character-space lift, NOT the raw entity
    // transform) times the socket's own model-space matrix. Spelling it a second
    // way would measure a place nothing draws.
    let hero_pose =
        inf_ecs::pose::model_to_world_of(sim.world(), hero).expect("the hero has a transform");
    let hand_world = socket
        .map(|m| hero_pose * glam::DAffine3::from_mat4(m.as_dmat4()))
        .map(|a| a.translation);
    let from_origin = (weapon_at - origin).length();
    let from_hand = hand_world
        .map(|h| (weapon_at - h).length())
        .unwrap_or(f64::NAN);
    println!("=== THE ISLAND'S HERO, WITH THE GLOCK ===");
    println!(
        "  hand_r socket           : {}",
        if socket.is_some() {
            "published"
        } else {
            "MISSING"
        }
    );
    println!(
        "  the weapon entity is    : {from_origin:.3} m from the character origin, {from_hand:.3} m from the hand socket"
    );
    println!("  muzzles without a socket: {no_socket} over 120 steps");
    assert!(
        socket.is_some(),
        "the island's hero publishes no `{}` socket - its weapon draws at its pelvis",
        d3::gameplay::WEAPON_SOCKET
    );
    assert!(
        from_origin > 0.25,
        "the weapon is {from_origin:.3} m from the character's own origin - that IS the pelvis"
    );
    assert!(
        from_hand < 0.25,
        "the weapon is {from_hand:.3} m from the hand socket it is supposed to be attached to"
    );
    assert_eq!(
        no_socket, 0,
        "the muzzle fell back to the capsule rule on a rig that publishes a socket"
    );
}

/// Every `.inf_skel` under a content root, recursively.
fn walk_skeletons(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "inf_skel") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// **A DISPATCHED UNIT REACHES THE SCENE** (wave WPN2e audit) -- link 2 of this
/// arc's island chain, diagnosed and closed.
///
/// The wave measured a dispatched unit closing to **93 m and stopping**, over
/// three minutes, against an `ON_SCENE_M` of 12, and carried it as EMS1/EMS2
/// ground. This arm drives the same road with the instrument on: per unit, every
/// ten seconds, its state, how far it still is from the incident it is on, how
/// much of its own route is left, and how fast it is going -- so a stop can be
/// told apart from a route that ends in the wrong place.
///
/// **What it reads**: `DispatchRes::runs` (the state and the `NavPath`), the
/// incident's own `at`, and the chassis' `Transform` -- the world, not a report.
#[test]
fn a_dispatched_unit_reaches_a_warm_file_on_the_island() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project - local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = island_sim(&content);
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..1200 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    {
        let defs = inf_ecs::item::item_defs_mut(sim.world_mut());
        if defs.is_empty() {
            defs.merge_toml(weapon::WEAPON_REGISTRY_TOML)
                .expect("the shipped registry parses");
        }
    }
    item::give(sim.world_mut(), hero, "glock_17", 1);
    d3::gameplay::equip_weapon(sim.world_mut(), hero, "glock_17");
    // Fire, through the shipped input path, until the town has a file. With the
    // ears channel the island's own crowd HEARS it; nothing here injects one.
    let mut state = inf_input::InputState::new(inf_input::default_map());
    let mut shots = 0u32;
    for i in 0..900 {
        {
            let w = sim.world_mut();
            if let Some((_, def)) = weapon::equipped_def(w, hero) {
                if let Some(e) = w.entity_of(hero) {
                    if let Some(mut st) = w.world_mut().get_mut::<weapon::WeaponState>(e) {
                        st.magazine = def.magazine;
                    }
                }
            }
        }
        let events: Vec<inf_input::InputEvent> = vec![inf_input::InputEvent::MouseButton {
            button: inf_input::MouseButton::Left,
            pressed: (i / 15) % 2 == 0,
        }];
        state.apply_dt(&events, DT);
        sim.step_once(inf_player::input::held_actions(&state, DT));
        shots += sim.gameplay().shots;
    }
    let heat = crime::heat_of(sim.world(), hero);
    println!("\n=== THE DRIVE, ON THE ISLAND ===");
    println!(
        "  the hero fired {shots} rounds; the file is at heat {heat} ({})",
        Response::for_heat(heat).name()
    );
    assert!(
        heat > 0,
        "no file opened on the island, so there is nothing to drive to"
    );

    let mut best = f64::INFINITY;
    let mut arrived_at: Option<f64> = None;
    let mut escorted = 0usize;
    for step in 0..10_800u32 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        let Some(res) = dispatch::dispatch_of(sim.world()) else {
            continue;
        };
        let runs: Vec<(Uuid, UnitState, Option<Uuid>, Option<(f64, f64)>)> = res
            .runs
            .iter()
            .map(|(chassis, run)| {
                (
                    *chassis,
                    run.state,
                    run.incident,
                    run.path.as_ref().map(|p| (p.length_m(), 0.0)),
                )
            })
            .collect();
        let incidents: BTreeMap<Uuid, DVec3> =
            res.incidents.iter().map(|(g, i)| (*g, i.at)).collect();
        let mut line: Vec<String> = Vec::new();
        for (chassis, ustate, incident, path) in runs {
            let here = sim
                .world()
                .entity_of(chassis)
                .and_then(|e| sim.world().world().get::<Transform>(e))
                .map(|t| t.translation.to_dvec3());
            let (Some(here), Some(target)) = (here, incident.and_then(|g| incidents.get(&g)))
            else {
                continue;
            };
            let d = (here - *target).length();
            if ustate == UnitState::EnRoute || ustate == UnitState::OnScene {
                best = best.min(d);
                if d <= dispatch::ON_SCENE_M && arrived_at.is_none() {
                    arrived_at = Some(f64::from(step) / 60.0);
                }
            }
            if step % 900 == 0 {
                let left = dispatch::dispatch_of(sim.world())
                    .and_then(|r| r.runs.get(&chassis).cloned())
                    .and_then(|r| r.path.map(|p| p.length_m() - p.project(here).s_m))
                    .unwrap_or(f64::NAN);
                let _route = path.map(|(l, _)| l).unwrap_or(f64::NAN);
                // **WHY IT IS OR IS NOT MOVING**, off the world: does the chassis
                // have a rapier body at all, is its crew in its seat, and what
                // stick is the crew holding.
                let crew = dispatch::crew_guid(chassis);
                let body = sim.bridge3d().body_of(chassis).is_some();
                let ent = sim.world().entity_of(chassis).is_some();
                let is_vehicle = sim.bridge3d().vehicle_guids().contains(&chassis);
                let speed = sim
                    .bridge3d()
                    .body_of(chassis)
                    .and_then(|b| sim.bridge3d().world().body_linvel(b))
                    .map(|v| v.length())
                    .unwrap_or(f64::NAN);
                let wheels = inf_ecs::vehicle::rig_of(sim.world(), chassis)
                    .map(|r| r.wheels.len())
                    .unwrap_or(0);
                let out = sim
                    .vehicles()
                    .iter()
                    .find(|v| v.chassis == chassis)
                    .copied();
                let grounded = out.map(|o| o.wheels_grounded).unwrap_or(99);
                let load = out.map(|o| o.load_n).unwrap_or(f64::NAN);
                let revs = out.map(|o| o.revs).unwrap_or(f64::NAN);
                let hb = sim
                    .world()
                    .entity_of(crew)
                    .and_then(|e| sim.world().world().get::<CharacterMovement>(e))
                    .map(|cm| (cm.runtime.want_handbrake, cm.mode))
                    .unwrap_or((false, MovementMode::Grounded));
                let seated = sim
                    .world()
                    .entity_of(crew)
                    .and_then(|e| sim.world().world().get::<CharacterMovement>(e))
                    .map(|cm| cm.runtime.seat.vehicle == chassis)
                    .unwrap_or(false);
                let stick = sim
                    .world()
                    .entity_of(crew)
                    .and_then(|e| sim.world().world().get::<CharacterMovement>(e))
                    .map(|cm| cm.runtime.intent_move)
                    .unwrap_or_default();
                line.push(format!(
                    "{:?} d={d:.0} left={left:.0} y={:.1} ent={ent} body={body} seat={seated} mode={:?} hb={} stick=({:.2},{:.2}) veh={is_vehicle} wheels={wheels}/{grounded} load={load:.0}N revs={revs:.2} v={speed:.2}",
                    ustate, here.y, hb.1, hb.0, stick.x, stick.y
                ));
            }
        }
        escorted += sim.dispatch_stats().escorted;
        if step % 900 == 0 && !line.is_empty() {
            println!(
                "  t={:>5.0} s escorted={escorted} steered={} {}",
                f64::from(step) / 60.0,
                sim.dispatch_stats().steered,
                line.join("\n              ")
            );
        }
    }
    println!(
        "  the nearest responder got within {best:.0} m of the incident (`ON_SCENE_M` {:.0}); first on scene at {}",
        dispatch::ON_SCENE_M,
        arrived_at
            .map(|t| format!("{t:.1} s"))
            .unwrap_or_else(|| "never".to_string())
    );
    assert!(
        arrived_at.is_some(),
        "no dispatched unit reached within {:.0} m of a warm file in three minutes; the nearest got to {best:.0} m",
        dispatch::ON_SCENE_M
    );
}

// ── the island, loosely (cov1_gate's `loose_sim`, verbatim) ─────────────────

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The island project, or `None` on a machine (CI included) that has none: its
/// content is licensed and local-only.
fn island_project() -> Option<PathBuf> {
    let p = repo().join("../../../island-build/project/Content");
    p.is_dir().then(|| p.canonicalize().unwrap_or(p))
}

fn island_sim(content: &Path) -> inf_player::runtime_sim::RuntimeSim {
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
    .with_cloth_assets(inf_player::level::load_cloth_assets_from_dir(content))
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

// ── the fixture plumbing (wpn2d_gate's, verbatim) ───────────────────────────

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
