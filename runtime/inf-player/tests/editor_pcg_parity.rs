//! **The editor's blocks are the player's blocks** (wave EDIT1, clause 1) — the
//! player's half.
//!
//! EDIT1 made the editor camera evaluate the PCG volumes it approaches, so an
//! author sees the city instead of the empty boxes the level carries until
//! something asks. The claim that makes that safe is that the editor's answer is
//! the *shipped player's* answer: not a preview, not an approximation, the same
//! buildings in the same places.
//!
//! # Why it is two test binaries and not one
//!
//! Neither host can link the other. `evaluate_pcg_volumes_in` lives in this
//! crate; the editor's `evaluate_volume_into` lives in the Tauri crate, which
//! nothing here may name. So the comparison is mediated by a **committed
//! digest**: this file builds the fixture city with the player's own world
//! types, evaluates it through the player's own door, and asserts the four
//! numbers below — and `inf_studio::commands::pcg_stream::tests::a_blocks_
//! population_does_not_depend_on_which_blocks_were_evaluated_first` computes the
//! identical digest through the editor's door and asserts the same four. If
//! either host drifts, its own test goes red.
//!
//! The zone documents are the REAL ones — `settlement::zone_payload`, the same
//! fourteen the island's 172 blocks name, reached here through the dev-only
//! `inf-editor-core` dependency this crate already carries.
//!
//! # What the digest is over
//!
//! Every scattered instance's position, rotation, scale, mesh GUID, kind and
//! extent, and every structural solid's centre, half-extents and rotation. The
//! solids are in it because `STRUCTURE_LOD_M` and `INTERIOR_LOD_M` band a
//! structure by its own size, so the LOD ladder's inputs are part of what the
//! two hosts must agree about. Not a count: a count is satisfied by any thousand
//! buildings.

use std::collections::HashMap;

use inf_ecs::components::{PcgVolume, Terrain};
use inf_ecs::{EcsWorld, TerrainData, Transform, Vec2d, Vec3d};
use inf_pcg::building::ArchetypeId;
use uuid::Uuid;

/// The four archetypes, in the order the digests below are listed.
const BLOCKS: [ArchetypeId; 4] = [
    ArchetypeId::Office,
    ArchetypeId::Shop,
    ArchetypeId::Apartment,
    ArchetypeId::House,
];

/// **The committed answer**, per block: `(digest, instances, solids)`.
///
/// MIRROR: the same four rows are asserted on the editor's side by
/// `inf-studio`'s `commands::pcg_stream::tests`. They are duplicated rather than
/// shared because the two hosts have no crate in common that should carry test
/// data — and duplication is what makes this a *pin*: a change to either host's
/// evaluation turns exactly one of the two tests red, and which one it is says
/// which host moved.
const EXPECTED: [(u64, usize, usize); 4] = [
    (0x6437_8b2a_54b3_6b5a, 5054, 4677), // Office
    (0xb254_2d28_93b0_ceff, 1950, 1801), // Shop
    (0xc21c_5456_d6c1_78dc, 6194, 5811), // Apartment
    (0x6e9c_c242_18ad_cca6, 2302, 2146), // House
];

/// MIRROR of the editor test's `digest`, field for field and in the same order.
fn digest(world: &EcsWorld, guid: Uuid) -> (u64, usize, usize) {
    let e = world.entity_of(guid).expect("volume entity");
    let vol = world.world().get::<PcgVolume>(e).expect("volume component");
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |x: f64| {
        h ^= x.to_bits();
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for i in &vol.evaluated {
        mix(i.position.x);
        mix(i.position.y);
        mix(i.position.z);
        mix(i.rotation.x);
        mix(i.rotation.y);
        mix(i.rotation.z);
        mix(i.scale);
        mix(i.mesh.map_or(0.0, |m| m.as_u128() as f64));
        mix(f64::from(i.kind));
        if let Some(e) = i.extent {
            for x in e {
                mix(f64::from(x));
            }
        }
    }
    for s in &vol.structures {
        mix(s.center.x);
        mix(s.center.y);
        mix(s.center.z);
        mix(s.half_extents.x);
        mix(s.half_extents.y);
        mix(s.half_extents.z);
        mix(s.rotation.x);
        mix(s.rotation.y);
        mix(s.rotation.z);
        mix(s.rotation.w);
    }
    (h, vol.evaluated.len(), vol.structures.len())
}

/// MIRROR of the editor test's `city`, in the player's own world types: the same
/// guids, the same nine flat terrain tiles, the same four blocks at the same
/// places with the same extents and the same seeds.
///
/// The terrain's guid sorts BELOW the blocks' on purpose — this host picks its
/// height source in Guid order, and a terrain that sorted after a volume would
/// be a different fixture, not a different bug.
fn city() -> (EcsWorld, Vec<Uuid>, HashMap<Uuid, inf_pcg::PcgAssetPayload>) {
    let mut world = EcsWorld::new();
    let ground = Uuid::from_u128(0x6100);
    let ge = world.spawn_with_guid(ground, "Ground", None);
    let mut data = TerrainData::new(64, 1.0);
    for ty in -1..=1 {
        for tx in -1..=1 {
            data.author_tile((tx, ty), |_, _| 0.0);
        }
    }
    world.world_mut().entity_mut(ge).insert(Terrain {
        data,
        ..Terrain::default()
    });

    let mut guids = Vec::new();
    let mut pcgs = HashMap::new();
    for (k, a) in BLOCKS.into_iter().enumerate() {
        let guid = Uuid::from_u128(0x7000 + k as u128);
        let e = world.spawn_with_guid(guid, a.name(), None);
        let graph = inf_editor_core::settlement::zone_guid(a);
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: Vec3d::new(60.0 * (k % 2) as f64, 0.0, 60.0 * (k / 2) as f64),
                rotation: Vec3d::ZERO,
                scale: Vec3d::ONE,
            },
            PcgVolume {
                graph: Some(graph),
                extent: Vec2d::new(28.0, 28.0),
                seed: 1_000 + k as u32,
                ..PcgVolume::default()
            },
        ));
        pcgs.insert(
            graph,
            inf_editor_core::settlement::zone_payload(a).expect("the committed zone document"),
        );
        guids.push(guid);
    }
    world.propagate();
    (world, guids, pcgs)
}

/// **THE ARM.** The shipped player's evaluation of four Harbour-City-shaped
/// blocks, digested, against the numbers the editor's own door produces.
#[test]
fn the_shipped_players_blocks_are_the_editors_blocks() {
    let (mut world, guids, pcgs) = city();
    inf_player::level::evaluate_pcg_volumes_in(&mut world, &pcgs, None);
    let got: Vec<(u64, usize, usize)> = guids.iter().map(|g| digest(&world, *g)).collect();
    println!("EDIT1 clause 1 — the PLAYER's blocks:");
    for ((h, n, s), a) in got.iter().zip(BLOCKS.iter()) {
        println!("  {:10} {n:5} inst / {s:5} solid / {h:016x}", a.name());
    }
    for (i, a) in BLOCKS.iter().enumerate() {
        assert_eq!(
            got[i],
            EXPECTED[i],
            "the shipped player's {} block is not the one the editor draws \
             (see this file's module doc: the editor's half asserts the same row)",
            a.name()
        );
    }
}

/// The arm above compares against a constant, and a constant can be wrong. This
/// one says the fixture is not vacuous: every block placed something, and the
/// four blocks are not four copies of one another.
#[test]
fn the_fixture_places_four_different_cities() {
    let (mut world, guids, pcgs) = city();
    inf_player::level::evaluate_pcg_volumes_in(&mut world, &pcgs, None);
    let got: Vec<(u64, usize, usize)> = guids.iter().map(|g| digest(&world, *g)).collect();
    assert!(
        got.iter().all(|(_, n, s)| *n > 0 && *s > 0),
        "a block placed nothing: {got:?}"
    );
    let mut seen: Vec<u64> = got.iter().map(|(h, _, _)| *h).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 4, "two blocks digested the same: {got:?}");
}
