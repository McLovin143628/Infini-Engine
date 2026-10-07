//! **WAVE PAR1a — BUILDING ILLUMINATION** — the gate.
//!
//! Every room of every archetype on every floor of the island hangs a real
//! fixture (a fitting you can see + a real `Light` from one vocabulary per
//! `RoomType`), exteriors carry porch / facade / sign / forecourt lights, and
//! the occupancy half of the night schedule decides which are lit. The arms:
//!
//! * the zero-unlit-rooms census over the WHOLE shipped island, by archetype
//!   and floor, read off the one volume door both hosts pass
//!   (`inf_pcg::compose_volume`) — and the same census WORLD-side, off the
//!   `ScatteredLight`s a loaded island actually holds;
//! * the 24 h schedule sweep: lit-room count by hour against the society's
//!   own day, PIE == shipping on the projected light list at every hour;
//! * the party wall: a lit room's fixture lights nothing in the room beside it
//!   (its box), with the unboxed control that does;
//! * (off CI) the luminance arms and the island's 21:00 light census.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use inf_pcg::building::fixtures::{self, CensusRow, FixtureRow};
use inf_pcg::building::FixtureTag;
use inf_pcg::ArchetypeId;

/// The shipped island's recipe.
fn shipped_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island/island.toml")
}

/// One evaluated settlement block, through the door the hosts use.
struct BlockEval {
    archetype: ArchetypeId,
    plans: Vec<inf_pcg::building::BuildingPlan>,
    lights: Vec<inf_pcg::building::PcgLight>,
}

/// Evaluate one block exactly as `inf_player::level` does: its zone graph's
/// building pass over its own extent, salted by its volume's guid, joined by
/// `compose_volume`. The ground is flat (a datum moves no room).
fn eval_block(block: &inf_editor_core::settlement::Block, guid: uuid::Uuid) -> BlockEval {
    let graph = inf_editor_core::settlement::zone_graph(block.archetype);
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    assert!(lowered.ok, "{:?}: {:?}", block.archetype, lowered.issues);
    let cx = inf_pcg::GrammarContext {
        entity: Some(guid),
        center: glam::DVec3::new(block.centre.x, 0.0, block.centre.y),
        extent: block.half,
        seed_offset: u64::from(block.seed),
    };
    let height = inf_pcg::FnHeight::new(|_, _| Some(0.0));
    let out = inf_pcg::evaluate_buildings(&lowered.buildings, &inf_pcg::NoSplines, &height, &cx);
    let plans = inf_pcg::building::plans_of(&lowered.buildings, &inf_pcg::NoSplines, &height, &cx);
    let vol = inf_pcg::compose_volume(Vec::new(), out);
    BlockEval {
        archetype: block.archetype,
        plans,
        lights: vol.lights,
    }
}

fn lit_rooms<'a>(tags: impl Iterator<Item = (u32, u32)> + 'a) -> BTreeSet<(u32, u32)> {
    tags.filter(|(_, r)| *r != FixtureTag::EXTERIOR).collect()
}

/// **THE ZERO-UNLIT-ROOMS CENSUS, OVER THE WHOLE SHIPPED ISLAND** (clause 2).
///
/// Reads: every block of every settlement of `samples/island`, evaluated
/// through `compose_volume`; for every room of every plan, whether a fixture
/// names it. The base tree (venue rigs only) fails it on every archetype but
/// the three venues' rig rooms.
#[test]
fn the_island_census_reads_zero_unlit_rooms_by_archetype_and_floor() {
    let recipe =
        inf_island::IslandRecipe::load(&shipped_recipe()).expect("the island recipe loads");
    let design = inf_island::read_design(&recipe).expect("the island design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let mut table: BTreeMap<(ArchetypeId, u32), CensusRow> = BTreeMap::new();
    let mut by_row: BTreeMap<FixtureRow, usize> = BTreeMap::new();
    let mut exterior_by_arch: BTreeMap<ArchetypeId, (usize, usize)> = BTreeMap::new();
    let (mut blocks, mut buildings, mut lights) = (0usize, 0usize, 0usize);
    for s in &plans {
        for b in &s.blocks {
            let guid = inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row);
            let e = eval_block(b, guid);
            blocks += 1;
            buildings += e.plans.len();
            lights += e.lights.len();
            for l in &e.lights {
                *by_row.entry(l.tag.row).or_default() += 1;
            }
            let lit = lit_rooms(e.lights.iter().map(|l| (l.tag.building, l.tag.room)));
            fixtures::census(&e.plans, &lit, &mut table);
            // A porch light on every building's street door.
            let porches: BTreeSet<u32> = e
                .lights
                .iter()
                .filter(|l| l.tag.row == FixtureRow::Porch)
                .map(|l| l.tag.building)
                .collect();
            let x = exterior_by_arch.entry(e.archetype).or_default();
            x.0 += e.plans.len();
            x.1 += porches.len();
        }
    }
    println!("PAR1a census: {blocks} blocks, {buildings} buildings, {lights} fixtures");
    println!(
        "{:<14} {:>5} {:>7} {:>6}",
        "archetype", "floor", "rooms", "unlit"
    );
    let mut unlit = 0u32;
    let mut rooms = 0u32;
    for ((a, f), r) in &table {
        println!("{:<14} {:>5} {:>7} {:>6}", a.name(), f, r.rooms, r.unlit);
        unlit += r.unlit;
        rooms += r.rooms;
    }
    for (row, n) in &by_row {
        println!("  row {:<18} {n}", row.name());
    }
    for (a, (b, p)) in &exterior_by_arch {
        println!("  {:<14} {b} buildings, {p} with a porch light", a.name());
        assert_eq!(b, p, "{a:?}: {} buildings have no porch light", b - p);
    }
    println!("PAR1a census: {rooms} rooms, {unlit} unlit");
    assert!(rooms > 1_000, "the census saw only {rooms} rooms");
    assert!(
        table.keys().filter(|(_, f)| *f >= 3).count() > 3,
        "the census saw no fourth floor anywhere"
    );
    assert_eq!(
        unlit, 0,
        "{unlit} of {rooms} rooms on the island hang no fixture"
    );
}
