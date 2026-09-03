//! **THE INSTITUTIONS ARE STAFFED, AND THEY ARE STAFFED AT BOTH HOURS**
//! (wave EMS1).
//!
//! # What this gate is for
//!
//! The wave's mandate is four buildings full of people: *"police stations with
//! police cars and police officers, and SWAT teams … fire halls with fire
//! fighters and fire trucks … hospitals with nurses and doctors and ambulances
//! and paramedics … clinics with doctors … administrative workers at the front
//! desk and behind the scenes."* Every one of those nouns is derived rather
//! than authored — a room type implies an occupancy, an occupancy implies a
//! slot, a slot becomes an agent — so the way to certify it is to build the
//! four archetypes, settle a society over them, and **count who is inside at
//! ten in the morning and at ten at night**.
//!
//! Ten and twenty-two are not decorative. `crews_of` is this wave's one new
//! rule and the whole of it is that three rooms — a cell block, an apparatus
//! bay, a ward — are worked by **two** crews rather than one. A gate that
//! looked only at ten would be satisfied by a table where every institution
//! shut at six; a gate that looked only at twenty-two would be satisfied by one
//! where they all opened when the bars do. It takes both, and the falsifying
//! subject is the **clinic**, which is asserted to be EMPTY at twenty-two.
//!
//! # The four arms
//!
//! * **(a)** the derivation — the four institutions, a society, and the table
//!   of who is where at each hour;
//! * **(b)** the fleet — read out of the **committed** `.inf_lvl` rather than
//!   from the generator, because "the recipe would park an appliance" and "the
//!   document has one in it" are different facts and the second is the one that
//!   ships;
//! * **(c)** PIE == shipping, byte for byte, over a settled town of
//!   institutions;
//! * **(d)** the budget table, against the ceilings that already exist. No new
//!   budget constant is minted: an institution is a `PcgVolume` and its people
//!   are crowd agents, and both already have a row and a ceiling.
//!
//! # No cook, and why that is honest here
//!
//! What is being measured is a *society*, and a society is a pure function of
//! `PcgVolume::residents` — which this file builds through
//! `settlement::zone_payload`, the same generator
//! `committed_sample_matches_generators` byte-locks the shipped `.inf_pcg`
//! documents against. So the passes here are the shipped passes; what is
//! skipped is a file round trip that another arm already pins.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{SlotPosture, SlotRole};
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_pcg::ArchetypeId;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;

/// The working morning — when a clinic is open and a ward's day crew is on.
const DAY_HOUR: f64 = 10.0;
/// The late evening — when a clinic is shut and a cell block is still watched.
const NIGHT_HOUR: f64 = 22.0;

/// Half-extent of every block this gate places, metres.
///
/// Large enough that a `Hospital` (a 52 × 50 m lot) gets a lot at all, which is
/// what puts a `Ward` and a `Waiting` room in it — a block too small for the
/// anchors is a building with no institution in it, and the arm would then be
/// measuring a partition rather than a palette.
const EXTENT: f64 = 46.0;

/// How far apart the blocks stand. Wide enough that no two overlap and near
/// enough that a resident can walk between them.
const PITCH: f64 = 130.0;

/// Steps the two hosts are compared over.
///
/// A society settles in its first few syncs and then plans; ninety steps is
/// past that and short enough to run in a gate.
const STEPS: u32 = 90;

/// The four institutions, their guids and where they stand.
///
/// A named table rather than a loop over `ArchetypeId::ALL.filter(is_institution)`,
/// because the arms below assert **different things about each** — a clinic is
/// empty at night and a fire hall is not — and a sweep would have to restate the
/// distinction anyway.
fn stations() -> [(Uuid, ArchetypeId, DVec3); 4] {
    [
        (
            Uuid::from_u128(0x0E5A_0001),
            ArchetypeId::PoliceStation,
            DVec3::new(-PITCH, 0.0, -PITCH),
        ),
        (
            Uuid::from_u128(0x0E5A_0002),
            ArchetypeId::FireHall,
            DVec3::new(PITCH, 0.0, -PITCH),
        ),
        (
            Uuid::from_u128(0x0E5A_0003),
            ArchetypeId::Hospital,
            DVec3::new(-PITCH, 0.0, PITCH),
        ),
        (
            Uuid::from_u128(0x0E5A_0004),
            ArchetypeId::Clinic,
            DVec3::new(PITCH, 0.0, PITCH),
        ),
    ]
}

/// The residential blocks that make the institutions' staff *people* rather
/// than empty slots.
///
/// A `Work` slot is only ever taken by an agent who has a `Home`, so a town of
/// four institutions and no houses would settle to zero workers and every arm
/// below would pass on an empty population. Three apartment blocks are more
/// homes than the institutions have jobs, which is the direction that leaves the
/// jobs as the scarce thing.
fn homes() -> [(Uuid, ArchetypeId, DVec3); 3] {
    [
        (
            Uuid::from_u128(0x0E5A_1001),
            ArchetypeId::Apartment,
            DVec3::ZERO,
        ),
        (
            Uuid::from_u128(0x0E5A_1002),
            ArchetypeId::Apartment,
            DVec3::new(0.0, 0.0, -PITCH),
        ),
        (
            Uuid::from_u128(0x0E5A_1003),
            ArchetypeId::Apartment,
            DVec3::new(0.0, 0.0, PITCH),
        ),
    ]
}

/// The building passes one archetype's committed zone document lowers to.
///
/// Read from `settlement::zone_payload` and not from the file, and that is the
/// same passes: `committed_sample_matches_generators` asserts every
/// `Zone_<A>.inf_pcg` on disk is byte-identical to this function's output, so a
/// document that had drifted would fail there rather than silently here.
fn zone_passes(a: ArchetypeId) -> Vec<inf_pcg::BuildingPass> {
    let payload = inf_editor_core::settlement::zone_payload(a).expect("the zone document encodes");
    let graph = payload.graph().expect("the graph is the source of truth");
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    assert!(lowered.ok, "{}: {:?}", a.name(), lowered.issues);
    assert_eq!(
        lowered.buildings.len(),
        1,
        "{} lowers to one building pass",
        a.name()
    );
    lowered.buildings
}

/// Put one block in the world, populated the way the shipped player populates
/// one (`inf_player::level::population_of`).
fn place(world: &mut EcsWorld, guid: Uuid, a: ArchetypeId, centre: DVec3) {
    let passes = zone_passes(a);
    let extent = DVec2::new(EXTENT, EXTENT);
    let cx = inf_pcg::GrammarContext {
        entity: Some(guid),
        center: centre,
        extent,
        seed_offset: 0x0E_5A1,
    };
    let height = inf_pcg::FnHeight::new(move |_, _| Some(centre.y));
    let out = inf_pcg::evaluate_buildings(&passes, &inf_pcg::NoSplines, &height, &cx);
    let (baked, solid, groups, doorways, residents, interior, lights, emitters) =
        inf_player::level::population_of(inf_pcg::compose_volume(Vec::new(), out));
    world.spawn_with_guid(guid, a.name(), None);
    let e = world.entity_of(guid).expect("the block");
    let mut vol = inf_ecs::components::PcgVolume {
        extent: inf_ecs::math::Vec2d::new(extent.x, extent.y),
        ..Default::default()
    };
    vol.set_population(
        baked, solid, groups, doorways, residents, interior, lights, emitters,
    );
    world.world_mut().entity_mut(e).insert((
        inf_ecs::components::Transform {
            translation: inf_ecs::math::Vec3d::new(centre.x, centre.y, centre.z),
            ..Default::default()
        },
        inf_ecs::components::GlobalTransform(glam::DAffine3::from_translation(centre)),
        vol,
    ));
}

/// **The one fixture**, so the two hosts of arm (c) cannot be given different
/// towns. The order is the order the blocks are placed in, which is what a
/// document's bytes and a society's fold both depend on.
fn build(world: &mut EcsWorld) {
    for (guid, a, at) in stations() {
        place(world, guid, a, at);
    }
    for (guid, a, at) in homes() {
        place(world, guid, a, at);
    }
}

/// Settle the society: fold every volume, then plan until nothing is pending.
fn settle(world: &mut EcsWorld) -> inf_ecs::society::SocietyStats {
    let mut stats = inf_ecs::society::sync_society(world);
    for _ in 0..600 {
        stats = inf_ecs::society::sync_society(world);
        if stats.pending == 0 && stats.planned_now == 0 {
            break;
        }
    }
    stats
}

/// Who is standing inside one institution, at one hour.
#[derive(Default, Clone, Copy, PartialEq, Debug)]
struct Staffed {
    /// Agents whose active leg has ARRIVED at a `Work` place inside this
    /// block's own footprint.
    workers: usize,
    /// …of which, standing (which is every institution slot: no institution
    /// room offers a seat or a dance floor).
    standing: usize,
}

/// **Who is inside each institution at `hour`.**
///
/// Read off the arrival rather than off the hour, which is VEN1b's own
/// measured law: `HOME_H` and `NIGHT_WORK_START_H` are both eighteen hundred, so
/// a classifier that read a leg's `start_h` counted 155 night workers in a town
/// with 31 night jobs. What identifies the *place* is the leg's own path: its
/// last point is where the body ends up, and a point inside a block's footprint
/// is inside that block.
fn staffed_at(world: &EcsWorld, hour: f64) -> BTreeMap<&'static str, Staffed> {
    let clock = inf_ecs::crowd::CrowdClock::new(0.0, hour);
    let pop = world
        .world()
        .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        .expect("a population");
    let mut out: BTreeMap<&'static str, Staffed> = stations()
        .iter()
        .map(|(_, a, _)| (a.name(), Staffed::default()))
        .collect();
    for (guid, rec) in &pop.records {
        let leg = rec.leg_at(*guid, clock);
        let arrival = rec.arrival_on(leg);
        if arrival.role != Some(SlotRole::Work) {
            continue;
        }
        let Some(end) = rec.path_on(leg).points().last().copied() else {
            continue;
        };
        for (_, a, centre) in stations() {
            if (end.x - centre.x).abs() <= EXTENT && (end.z - centre.z).abs() <= EXTENT {
                let e = out.get_mut(a.name()).expect("a named station");
                e.workers += 1;
                if arrival.posture == SlotPosture::Stand {
                    e.standing += 1;
                }
            }
        }
    }
    out
}

// ── (a) the derivation ──────────────────────────────────────────────────────

/// **THE HEADLINE.** Four institutions, one society, and the table of who is
/// inside at ten and at twenty-two.
#[test]
fn the_institutions_are_staffed_at_ten_in_the_morning_and_at_ten_at_night() {
    let mut world = EcsWorld::new();
    build(&mut world);
    let stats = settle(&mut world);
    println!(
        "EMS1 society: {} agent(s), {} day job(s), {} night job(s) -> {} night \
         worker(s)",
        stats.agents, stats.works, stats.night_jobs, stats.night_workers
    );
    // ANTI-VACUITY, first: a town with no agents in it agrees with every claim
    // below by holding nobody at all.
    assert!(stats.agents > 0, "the town settled to nobody");
    assert!(
        stats.night_jobs > 0,
        "no night job anywhere — `crews_of`'s round-the-clock rooms are not \
         reaching a slot at all, and every arm about twenty-two hundred below \
         is then about an empty list"
    );

    let day = staffed_at(&world, DAY_HOUR);
    let night = staffed_at(&world, NIGHT_HOUR);
    println!(
        "\n{:<16} {:>10} {:>12}",
        "institution", "at 10:00", "at 22:00"
    );
    for (_, a, _) in stations() {
        println!(
            "{:<16} {:>10} {:>12}",
            inf_pcg::archetype(a).display,
            day[a.name()].workers,
            night[a.name()].workers
        );
    }

    // **The three that never close are staffed at BOTH hours.**
    for a in [
        ArchetypeId::PoliceStation,
        ArchetypeId::FireHall,
        ArchetypeId::Hospital,
    ] {
        assert!(
            day[a.name()].workers > 0,
            "{} is empty at {DAY_HOUR:.0}:00",
            a.name()
        );
        assert!(
            night[a.name()].workers > 0,
            "{} is empty at {NIGHT_HOUR:.0}:00 — a ward, a bay and a cell block \
             are the three rooms `crews_of` says are worked round the clock, \
             and nothing is in this one",
            a.name()
        );
    }
    // **And the clinic SHUTS**, which is the arm that makes the three above a
    // measurement rather than a property of the word "institution".
    assert!(
        day[ArchetypeId::Clinic.name()].workers > 0,
        "a clinic holds no clinician at ten in the morning"
    );
    assert_eq!(
        night[ArchetypeId::Clinic.name()].workers,
        0,
        "a clinic is staffed at {NIGHT_HOUR:.0}:00 — then nothing in this gate \
         distinguishes a round-the-clock room from a day one"
    );
    // Every institution body is on its feet: no institution room offers a seat
    // or a dance floor, so a `Sit` or a `Dance` here would mean a leisure slot
    // has leaked into a civic building.
    for (_, a, _) in stations() {
        for (label, t) in [("10:00", &day), ("22:00", &night)] {
            let s = t[a.name()];
            assert_eq!(
                s.workers,
                s.standing,
                "{} at {label}: {} of {} bodies are not standing — a civic \
                 building has grown a leisure slot",
                a.name(),
                s.workers - s.standing,
                s.workers
            );
        }
    }
}

/// **THE FRONT DESK AND THE BACK OFFICE** — the mandate's *"administrative
/// workers at the front desk and behind the scenes"*, as two counts.
///
/// A separate arm from the one above because it is a different claim about the
/// same town: that the staff are not all in the operational rooms. It reads the
/// PLAN rather than the society, because a desk is a station and a station is a
/// pure function of the plan and the palette.
#[test]
fn every_institution_has_a_front_desk_and_an_office_behind_it() {
    println!(
        "\n{:<16} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "institution", "rooms", "desks", "office", "day", "night"
    );
    for (_, a, _) in stations() {
        let arch = inf_pcg::archetype(a);
        let footprint = inf_pcg::Rect2::new(DVec2::new(-20.0, -16.0), DVec2::new(20.0, 16.0));
        let mut params = inf_pcg::BuildingParams::new(a, footprint, 0.0, 0x0E_5A1);
        params.floors = 3;
        let out = inf_pcg::building::build(&params, 0x0E_5A1, true);
        let slots = inf_pcg::building::society::slots_of(&out.plan, 0, 0);
        let stations = inf_pcg::building::society::station_slots(&out.plan, &out.stations, 0, 0);
        let desks = out
            .stations
            .iter()
            .filter(|s| s.use_kind == inf_pcg::StationUse::Tend)
            .count();
        let offices = out
            .plan
            .rooms
            .iter()
            .filter(|r| r.kind == inf_pcg::RoomType::Office)
            .count();
        let work = |sh: inf_pcg::SlotShift| {
            slots
                .iter()
                .chain(stations.iter())
                .filter(|s| s.role == inf_pcg::SlotRole::Work && s.shift == sh)
                .count()
        };
        println!(
            "{:<16} {:>7} {:>7} {:>7} {:>7} {:>7}",
            arch.display,
            out.plan.rooms.len(),
            desks,
            offices,
            work(inf_pcg::SlotShift::Day),
            work(inf_pcg::SlotShift::Night),
        );
        assert!(
            desks > 0,
            "{}: no counter anybody stands behind — the front desk is a `Run` \
             piece and a `Tend` station is derived where a run is PLACED, so a \
             zero here is a waiting room the placer refused",
            arch.display
        );
        assert!(
            work(inf_pcg::SlotShift::Day) > 0,
            "{}: nobody works it by day",
            arch.display
        );
    }
}

// ── (b) the fleet, out of the committed document ────────────────────────────

/// **THE FLEET IS IN THE COMMITTED LEVEL, WEARING ITS LIVERY.**
///
/// Read out of `samples/island-fixture/IslandFixture.inf_lvl` — the document
/// that ships — rather than out of the generator, because "the recipe would
/// park an appliance" and "the file has one in it" are different facts and only
/// the second one is what a player loads.
///
/// The fixture's four-block town places one fire hall and therefore one
/// appliance; the shipped island's seventeen vehicles are measured by
/// `island::tests::every_station_parks_its_fleet_in_its_own_livery`, which runs
/// the generator. Both are checked because the two answer different questions.
#[test]
fn the_committed_level_holds_a_liveried_appliance() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lvl = root.join("samples/island-fixture/IslandFixture.inf_lvl");
    if !lvl.exists() {
        println!("SKIP: no committed fixture level in this tree");
        return;
    }
    let doc = inf_editor_core::scene::serialize::load(&lvl).expect("the fixture level loads");
    let world = doc.world();

    // Every light bar in the document, by the name the livery gives it.
    let mut bars = 0usize;
    let mut bloomed = 0usize;
    let mut chassis = 0usize;
    for e in world.world().iter_entities() {
        let Some(name) = world.world().get::<inf_ecs::components::Name>(e.id()) else {
            continue;
        };
        if name.0 != "light_bar" {
            continue;
        }
        bars += 1;
        let m = world
            .world()
            .get::<inf_ecs::components::Material>(e.id())
            .expect("a light bar is drawn");
        let lin = m.emissive_linear();
        if lin[0].max(lin[1]).max(lin[2]) > 1.0 {
            bloomed += 1;
        }
        // …and it is ON a car: its parent is a chassis the rig recogniser finds.
        let parent = inf_ecs::hierarchy::parent_of(world.world(), e.id());
        if let Some(p) = parent {
            if let Some(g) = world.world().get::<inf_ecs::components::Guid>(p) {
                if inf_ecs::vehicle::rig_of(world, g.0).is_some() {
                    chassis += 1;
                }
            }
        }
    }
    println!("EMS1 committed fixture: {bars} light bar(s), {bloomed} bloom, {chassis} on a rig");
    assert!(
        bars > 0,
        "the committed fixture level holds no emergency vehicle at all"
    );
    assert_eq!(
        bloomed, bars,
        "a light bar that does not bloom is a grey box"
    );
    assert_eq!(chassis, bars, "a light bar is floating off a car");
}

// ── (c) PIE == shipping ─────────────────────────────────────────────────────

fn player_trace() -> Vec<Vec<u8>> {
    let mut world = EcsWorld::new();
    build(&mut world);
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            inf_ecs::crowd::crowd_state_bytes(sim.world())
        })
        .collect()
}

fn editor_trace() -> Vec<Vec<u8>> {
    let mut doc = SceneDoc::new();
    build(doc.world_mut());
    let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::ZERO, HZ);
    let out = (0..STEPS)
        .map(|_| {
            session.step_once(&mut doc, SimInput::default());
            inf_ecs::crowd::crowd_state_bytes(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

/// **PIE == SHIPPING, BYTE FOR BYTE, OVER A TOWN OF INSTITUTIONS.**
///
/// The trace is `state_bytes`, which folds the crowd — so what is being
/// compared is the whole population the four institutions and three apartment
/// blocks imply, agent by agent, step by step. Armed before it is compared: two
/// empty worlds agree perfectly, so the trace has to hold a population and it
/// has to CHANGE.
#[test]
fn pie_equals_shipping_over_a_town_of_institutions() {
    let ship = player_trace();
    let pie = editor_trace();
    assert_eq!(ship.len(), STEPS as usize);
    assert_eq!(pie.len(), ship.len(), "the two hosts ran different courses");
    // Armed at the FAR END, because the society is derived by the sim rather
    // than before it: `SimSession::enter` clears the crowd, the society and the
    // traffic on purpose — a Simulate session starts from the author's document
    // and not from a world somebody already simulated — so a pre-settled
    // fixture would have compared a populated player against an empty editor.
    // The trace therefore starts EMPTY and fills, and what is armed is the end.
    let last = ship.last().expect("a trace");
    assert!(
        last.len() > 64,
        "the trace ended at {} bytes — the town never settled a population over \
         {STEPS} steps, so there is nothing in this world to compare",
        last.len()
    );
    assert!(
        ship.windows(2).any(|w| w[0] != w[1]),
        "the town never changed state over {STEPS} steps, so this arm compares \
         two constants"
    );
    for (i, (a, b)) in ship.iter().zip(pie.iter()).enumerate() {
        assert_eq!(a, b, "PIE and shipping diverged at step {i}");
    }
}

// ── (d) the budget ──────────────────────────────────────────────────────────

/// **WHAT A TOWN OF INSTITUTIONS COSTS**, per phase, against the ceilings that
/// already exist.
///
/// No new budget constant is minted, and that is the point: an institution is a
/// `PcgVolume` and its people are crowd agents, so the phases it costs are the
/// `crowd` and `society` rows, whose ceilings `NPC_STEP_BUDGET_MS` and
/// `SOCIETY_STEP_BUDGET_MS` already are. A wave that added a row here would be
/// a wave that added a step phase, and this one did not.
#[test]
fn a_town_of_institutions_costs_what_it_costs() {
    let mut world = EcsWorld::new();
    build(&mut world);
    let stats = settle(&mut world);
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_step_profiling(true);
    // MIN of three rounds of sixty settled steps — the discipline everywhere
    // else in this tree, because a step profile is one step and a single one of
    // them is a fact about a scheduler.
    let (rounds, per_round) = (3u32, 60u32);
    let mut best: Option<inf_player::step_profile::StepProfile> = None;
    for _ in 0..rounds {
        let mut mean = inf_player::step_profile::StepProfile::default();
        for _ in 0..per_round {
            sim.step_once(RuntimeInput::default());
            mean.accumulate(&sim.step_profile());
        }
        mean.scale(1.0 / f64::from(per_round));
        if best.as_ref().is_none_or(|b| mean.total_ms() < b.total_ms()) {
            best = Some(mean);
        }
    }
    let mean = best.expect("three rounds");
    println!(
        "\nA TOWN OF FOUR INSTITUTIONS ({} build), {} agent(s), {:.4} ms total, \
         MIN of {rounds} rounds of {per_round}:",
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        },
        stats.agents,
        mean.total_ms()
    );
    for (name, ms) in mean.dearest_first() {
        if ms > 0.0005 {
            println!("  {name:>18}  {ms:.4} ms");
        }
    }
    let idx = |name: &str| {
        inf_player::step_profile::STEP_PHASE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("the `{name}` phase exists"))
    };
    let crowd = mean.ms[idx("crowd")];
    let society = mean.ms[idx("society")];
    println!(
        "  the `crowd` row: {crowd:.4} ms for {} agent(s) against a {:.1} ms \
         budget at {} agents",
        stats.agents,
        inf_player::budget::NPC_STEP_BUDGET_MS,
        inf_player::budget::NPC_BUDGET_AGENTS
    );
    println!(
        "  the `society` row: {society:.4} ms against a {:.1} ms budget",
        inf_player::budget::SOCIETY_STEP_BUDGET_MS
    );
    // A budget met by a door that returned early is a budget about nothing.
    assert!(
        stats.agents > 0,
        "the `crowd` phase was priced on a town with nobody in it"
    );
    if cfg!(debug_assertions) {
        eprintln!("dev build: the phase table is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the phase table is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        crowd <= inf_player::budget::NPC_STEP_BUDGET_MS,
        "the `crowd` phase cost {crowd:.4} ms against a {} ms ceiling {}",
        inf_player::budget::NPC_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
    assert!(
        society <= inf_player::budget::SOCIETY_STEP_BUDGET_MS,
        "the `society` phase cost {society:.4} ms against a {} ms ceiling {}",
        inf_player::budget::SOCIETY_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}
