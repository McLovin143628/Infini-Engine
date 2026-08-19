//! **PCG cannot see a streamed terrain, and this arm pins that it cannot.**
//!
//! Found by the AAA-readiness audit while walking the island starter map's
//! critical path. It is not a regression — it has been true since asset-backed
//! terrain landed at P16 — and it is the single largest blocker between this
//! engine and a 50 km² world with "full PCG" on it.
//!
//! # The mechanism, in three facts that compose
//!
//! 1. An **asset-backed terrain ships no inline tiles.** Its heights live in the
//!    `.inf_terrain` and are paged in by the streamer;
//!    `samples::streamed_terrain_scene` asserts this about itself
//!    (`debug_assert!(terrain.data.is_empty(), "a streamed terrain ships no tiles")`).
//! 2. `evaluate_pcg_volumes` picks its height source with
//!    **`if t.data.is_empty() { return None }`** — i.e. it looks for the first
//!    terrain that already has tiles in memory.
//! 3. Its `None` arm is **`FnHeight::new(|_, _| Some(0.0))`**.
//!
//! And the ordering makes 1 unavoidable: `evaluate_pcg_volumes` runs inside the
//! world builder, while `attach_terrain_streaming` runs *after* `RuntimeSim::new`.
//! There is no arrangement of a level in which an asset-backed terrain has pages
//! resident at the moment PCG asks.
//!
//! # Why this is worse than "everything is at y = 0"
//!
//! The measured counts below are **220 instances over a real hill and 929 over
//! the flat**. A scatter rule is not merely displaced by the flattening — slope
//! and height masks all evaluate against a plane, so the *rule set itself*
//! produces a different population. "Full PCG over the island" would not be a
//! world that needs nudging downward; it would be a different world.
//!
//! # This gate asserts the DEFECT
//!
//! Deliberately, and by house precedent (the P22 damage-locality gate did the
//! same): the day PCG is given a height source that can page — an evaluation
//! deferred until after streaming attaches, or a `TerrainSource`-backed
//! `HeightProvider` — **this test fails**, and whoever fixes it rewrites this
//! file and the ledger entry it names. A remainder that no test can see is a
//! remainder that gets rediscovered by an author.
//!
//! The editor mirror (`commands::pcg::pcg_evaluate`) has the identical
//! `is_empty()` fallback, so the two hosts *agree* — PIE == shipping still holds.
//! They agree on the wrong answer, which is why no existing gate caught this.

use std::collections::HashMap;

use inf_ecs::components::{Guid, PcgVolume, Terrain, Transform};
use inf_ecs::math::Vec2d;
use inf_ecs::world::EcsWorld;
use inf_pcg::asset::PcgAssetPayload;
use uuid::Uuid;

const TERRAIN_GUID: Uuid = Uuid::from_u128(0x0000_1111);
const VOLUME_GUID: Uuid = Uuid::from_u128(0x0000_2222);
const GRAPH_GUID: Uuid = Uuid::from_u128(0x0000_3333);

/// A world with one terrain and one scatter volume over it.
///
/// `inline_tiles` chooses the two cases: an **authored** terrain that carries its
/// heights in the level (what every committed sample does today), and an
/// **asset-backed** one that streams them (what a 50 km² island must do — the
/// heights are hundreds of megabytes and a `.inf_lvl` is the author's document).
fn build(inline_tiles: bool) -> (EcsWorld, HashMap<Uuid, PcgAssetPayload>) {
    let mut world = EcsWorld::default();

    let t = world.spawn_with_guid(TERRAIN_GUID, "Terrain", None);
    let mut terrain = Terrain::configured(64, 1.0);
    if inline_tiles {
        // A real hill, so "did the scatter follow the ground" has a falsifiable
        // answer rather than resting on a flat control.
        let span = terrain.data.tile_span();
        terrain
            .data
            .write_region(glam::DVec2::ZERO, glam::DVec2::splat(span), |x, z| {
                50.0 + 10.0 * inf_math::psin64(x * 0.05) * inf_math::pcos64(z * 0.05)
            });
        assert!(!terrain.data.is_empty(), "the inline case must have tiles");
    } else {
        terrain.asset = Some(Uuid::from_u128(0x0000_4444));
        assert!(
            terrain.data.is_empty(),
            "an asset-backed terrain ships no inline tiles — if this ever stops \
             being true, the premise of this whole file has changed"
        );
    }
    world.world_mut().entity_mut(t).insert(terrain);
    world.world_mut().entity_mut(t).insert(Transform::default());

    let v = world.spawn_with_guid(VOLUME_GUID, "Scatter", None);
    world.world_mut().entity_mut(v).insert(PcgVolume {
        graph: Some(GRAPH_GUID),
        extent: Vec2d::new(30.0, 30.0),
        ..PcgVolume::default()
    });
    world.world_mut().entity_mut(v).insert(Transform::default());
    world.propagate();

    let mut pcgs = HashMap::new();
    pcgs.insert(
        GRAPH_GUID,
        PcgAssetPayload {
            schema_version: PcgAssetPayload::CURRENT_VERSION,
            graph_json: None,
            document: inf_editor_core::samples::terrain_demo_pcg_document(),
        },
    );
    (world, pcgs)
}

/// Every scattered instance's world Y.
fn instance_ys(world: &EcsWorld) -> Vec<f64> {
    let w = world.world();
    w.iter_entities()
        .filter_map(|e| e.get::<Guid>().map(|g| (g.0, e.id())))
        .filter(|(g, _)| *g == VOLUME_GUID)
        .filter_map(|(_, e)| w.get::<PcgVolume>(e))
        .flat_map(|v| v.evaluated.iter().map(|i| i.position.y))
        .collect()
}

#[test]
fn pcg_over_an_asset_backed_terrain_scatters_at_sea_level() {
    // ── The control: an authored terrain, where this all works. ─────────────
    let (mut world, pcgs) = build(true);
    inf_player::level::evaluate_pcg_volumes(&mut world, &pcgs);
    let inline = instance_ys(&world);
    assert!(
        !inline.is_empty(),
        "the control scattered nothing — the fixture is not posing the problem"
    );
    let (lo, hi) = (
        inline.iter().cloned().fold(f64::INFINITY, f64::min),
        inline.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    assert!(
        lo > 40.0 && hi > lo + 1.0,
        "over an authored hill the scatter must FOLLOW it — got y in {lo}..{hi}"
    );

    // ── The defect: the same volume over an asset-backed terrain. ───────────
    let (mut world, pcgs) = build(false);
    inf_player::level::evaluate_pcg_volumes(&mut world, &pcgs);
    let streamed = instance_ys(&world);
    assert!(
        !streamed.is_empty(),
        "the streamed case scattered nothing at all, which is a different bug \
         from the one this file pins"
    );
    let off_zero = streamed.iter().filter(|y| y.abs() > 1e-9).count();
    assert_eq!(
        off_zero,
        0,
        "PINNED DEFECT: {} of {} instances are off y=0. If this is because PCG \
         can now page a streamed terrain's heights, DELETE THIS TEST and retire \
         the island-blocker entry in docs/memos/aaa-readiness-certification.md. \
         If it is because the scatter moved for another reason, something else \
         changed.",
        off_zero,
        streamed.len()
    );

    // …and the population itself differs, which is the half that makes this more
    // than a vertical offset: slope and height masks evaluate against a plane.
    assert_ne!(
        inline.len(),
        streamed.len(),
        "the two cases produced the same COUNT, so the flattening is only \
         vertical after all and this file overstates the defect"
    );
    eprintln!(
        "PINNED: authored terrain -> {} instances following the ground; \
         asset-backed terrain -> {} instances, all at y=0",
        inline.len(),
        streamed.len()
    );
}
