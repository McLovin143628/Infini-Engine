//! **The editor camera streams the world** (wave EDIT1, clause 1) — the Ring-2
//! half.
//!
//! The policy is [`inf_editor_core::pcg_stream`] (Ring 1, unit-tested on the
//! Linux CI leg); this file is the *fetch* and the *tick*, which is the same
//! split `commands::pcg` states for biome bindings and `commands::assets` states
//! for imports. What it adds to the editor is one sentence: **the buildings the
//! player will see are evaluated as the author's camera approaches them**,
//! instead of only when the author right-clicks each of a hundred and seventy
//! two blocks in turn, or presses Play.
//!
//! # Why a thread of its own
//!
//! Not `assets::spawn_tick`: that function is under a source gate
//! (`commands::dcc`'s `the_asset_tick_...` arms extract its body and refuse
//! sim-related tokens in it), and mixing a world-mutating evaluator into the
//! asset watcher would be the wrong concern besides. Not the viewport thread
//! either: a block of grammar buildings costs tens of milliseconds and cannot be
//! preempted half-built, so evaluating on the render thread would drop frames
//! for exactly as long as it took. A plain named thread that takes the document
//! lock for a budgeted slice is the shape that leaves the viewport free to draw
//! whatever is already there.
//!
//! # Why the graphs are read off disk
//!
//! [`super::pcg::PcgState`] is a *scratch* workspace: its documents are created
//! by `pcg_create` and keyed `"pcg:{n}"`, never by asset GUID, and nothing ever
//! loads a `.inf_pcg` into it — double-clicking one in the Content Drawer opens
//! the panel and the panel makes a new graph. So the island's fourteen zone
//! documents are not in it and never will be. The resolver below is the same one
//! `pcg_evaluate_biomes` uses (`AssetState::load_pcg_bytes` →
//! `PcgAssetPayload::decode` → the authored graph re-lowered when there is one,
//! else the stored lowered mirror), cached per GUID because fourteen documents
//! would otherwise be read and lowered on every tick of every second.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use glam::{DVec2, DVec3};
use inf_asset::AssetId;
use inf_ecs::components::{GlobalTransform, PcgVolume, Transform};
use inf_editor_core::ipc::PcgStreamStatusDto;
use inf_editor_core::pcg_stream::{EditorPcgStreams, VolumeCandidate};
use inf_pcg::graph::{lower_graph_with, pcg_registry, LoweredPcg};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

use super::assets::AssetState;
use super::pcg::{evaluate_volume_into, page_for_pcg};
use super::scene::{emit_world_delta, SceneState};

/// How often the streaming tick wakes. 250 ms, four times the asset tick's
/// interval: a tick that finds work holds the document lock for up to the
/// streamer's budget, and waking four times a second is already faster than an
/// author can fly out of a radius that is hundreds of metres wide.
const TICK: Duration = Duration::from_millis(250);

/// How long the `.inf_terrain` path index is reused before it is walked again.
///
/// The walk is a recursive directory scan of the content root and it is the one
/// thing here that is not cheap; it is also nearly static (a terrain import is a
/// rare event, and the asset tick already refreshes the viewport's own index
/// when one lands). Ten seconds is short enough that a freshly imported terrain
/// is streamed against within a breath, and long enough that the scan is not
/// this feature's cost.
const TERRAIN_INDEX_TTL: Duration = Duration::from_secs(10);

/// The editor's PCG streaming state: the Ring-1 policy, the camera it follows,
/// and the two caches that keep a tick cheap.
#[derive(Default)]
pub struct PcgStreamState {
    streams: Mutex<EditorPcgStreams>,
    /// The editor camera, from `ViewportEvent::EyeMoved`. `None` until the
    /// viewport has rendered a frame — and a `None` camera streams nothing,
    /// which is the right answer for a headless or not-yet-drawn session.
    eye: Mutex<Option<DVec3>>,
    /// Lowered zone graphs by ASSET guid. A miss that fails to resolve is cached
    /// as `None` too: a level bound to a document the project does not carry
    /// would otherwise re-read and re-fail for every volume of every tick.
    lowered: Mutex<HashMap<Uuid, Option<Arc<LoweredPcg>>>>,
    terrain_paths: Mutex<Option<(Instant, HashMap<Uuid, std::path::PathBuf>)>>,
    /// The document version the ground was last paged at.
    ///
    /// `page_for_pcg` pages the regions of EVERY `PcgVolume` and EVERY `Spline`
    /// in the level, because that is what the shipped player pages and a grammar
    /// pass reads heights along a spline that may leave its own volume's box —
    /// narrowing it to the volumes a tick is about to evaluate would be a
    /// different world, not a cheaper one. It is idempotent and the tiles stay
    /// resident, so the cost is a scan after the first call; this remembers the
    /// version so even that scan is not paid four times a second while a camera
    /// crosses a city.
    paged_version: Mutex<Option<u64>>,
    ticking: AtomicBool,
    /// The last status published, so a settled editor emits nothing.
    last_status: Mutex<Option<PcgStreamStatusDto>>,
}

impl PcgStreamState {
    /// Record where the editor camera is (`ViewportEvent::EyeMoved`).
    pub fn set_eye(&self, eye: DVec3) {
        if let Ok(mut g) = self.eye.lock() {
            *g = Some(eye);
        }
    }

    /// Forget everything about the level that was just replaced. The document is
    /// the caller's business; this only drops what the streamer remembers about
    /// it, so a new level does not inherit the last one's owned set.
    pub fn clear(&self) {
        if let Ok(mut s) = self.streams.lock() {
            s.clear();
        }
        if let Ok(mut l) = self.lowered.lock() {
            l.clear();
        }
        if let Ok(mut t) = self.terrain_paths.lock() {
            *t = None;
        }
        if let Ok(mut v) = self.paged_version.lock() {
            *v = None;
        }
        if let Ok(mut s) = self.last_status.lock() {
            *s = None;
        }
    }

    fn status(&self, loading: bool) -> PcgStreamStatusDto {
        let s = self.streams.lock().map(|g| g.stats()).unwrap_or_default();
        PcgStreamStatusDto {
            enabled: self.streams.lock().map(|g| g.is_enabled()).unwrap_or(false),
            loading,
            volumes: s.volumes as u32,
            in_radius: s.in_radius as u32,
            in_radius_populated: s.in_radius_populated as u32,
            populated: s.populated as u32,
            last_tick_evaluated: s.last_tick_evaluated as u32,
            last_tick_ms: s.last_tick_ms,
            budget_ms: s.budget_ms,
            evaluated_total: s.evaluated_total,
            released_total: s.released_total,
            activation_m: s.activation_m,
            prefetch_m: s.prefetch_m,
        }
    }
}

/// Apply the editor preferences (wave EDIT1, clause 1). Called from
/// `settings_apply`'s door, and once on boot.
pub fn apply_settings(app: &AppHandle, enabled: bool, radius_scale: f32) {
    let Some(state) = app.try_state::<PcgStreamState>() else {
        return;
    };
    let Ok(mut s) = state.streams.lock() else {
        return;
    };
    s.set_enabled(enabled);
    s.set_radius_scale(f64::from(radius_scale));
}

/// The candidates the policy ranks, read off the live document under a short
/// lock: every entity carrying a `PcgVolume` that names a graph.
///
/// A volume with `graph: None` is skipped rather than counted — it has nothing
/// to evaluate, so counting it would make the "Loading world…" indicator show a
/// denominator it can never reach.
fn candidates_of(doc: &inf_editor_core::scene::SceneDoc) -> Vec<(VolumeCandidate, Uuid)> {
    let mut out = Vec::new();
    for guid in doc.order().iter().copied() {
        let Some(e) = doc.entity_of(guid) else {
            continue;
        };
        let w = doc.world().world();
        let Some(vol) = w.get::<PcgVolume>(e) else {
            continue;
        };
        let Some(graph) = vol.graph else { continue };
        let centre = w
            .get::<GlobalTransform>(e)
            .map(|g| g.translation())
            .or_else(|| w.get::<Transform>(e).map(|t| t.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        out.push((
            VolumeCandidate {
                guid,
                centre,
                half_extent: DVec2::new(vol.extent.x, vol.extent.y),
                populated: !vol.evaluated.is_empty() || !vol.structures.is_empty(),
            },
            graph,
        ));
    }
    out
}

/// Resolve and lower one `.inf_pcg` asset, memoized.
///
/// MIRROR of `pcg_evaluate_biomes`' resolver, and deliberately: *"the stored
/// authored graph re-lowered when there is one (the graph is the source of
/// truth), else the stored lowered mirror"* is the rule both hosts read a zone
/// document by, and a second spelling of it here would be a second world.
fn lowered_for(
    state: &PcgStreamState,
    assets: &AssetState,
    registry: &inf_graph::NodeRegistry,
    graph: Uuid,
) -> Option<Arc<LoweredPcg>> {
    if let Ok(cache) = state.lowered.lock() {
        if let Some(hit) = cache.get(&graph) {
            return hit.clone();
        }
    }
    let resolved = (|| {
        let bytes = assets.load_pcg_bytes(AssetId(graph))?;
        let payload = inf_pcg::PcgAssetPayload::decode(&bytes).ok()?;
        let lowered = match payload.graph() {
            Some(g) => lower_graph_with(&g, registry, &inf_pcg::graph::NoMasks),
            None => LoweredPcg {
                document: payload.document.clone(),
                grammars: Vec::new(),
                buildings: Vec::new(),
                issues: Vec::new(),
                ok: true,
            },
        };
        lowered.ok.then(|| Arc::new(lowered))
    })();
    if resolved.is_none() {
        tracing::warn!("pcg stream: cannot lower zone document {graph}");
    }
    if let Ok(mut cache) = state.lowered.lock() {
        cache.insert(graph, resolved.clone());
    }
    resolved
}

fn terrain_paths(state: &PcgStreamState, assets: &AssetState) -> HashMap<Uuid, std::path::PathBuf> {
    if let Ok(g) = state.terrain_paths.lock() {
        if let Some((at, paths)) = g.as_ref() {
            if at.elapsed() < TERRAIN_INDEX_TTL {
                return paths.clone();
            }
        }
    }
    let paths = assets
        .content_root()
        .map(|root| inf_editor_core::terrain_stream::terrain_paths_by_guid(&root))
        .unwrap_or_default();
    if let Ok(mut g) = state.terrain_paths.lock() {
        *g = Some((Instant::now(), paths.clone()));
    }
    paths
}

/// Start the streaming tick. Idempotent — the `ticking` latch means a second
/// call is a no-op, on the `photogrammetry::spawn_tick` precedent.
pub fn init_pcg_stream_on_boot(app: &AppHandle) {
    let Some(state) = app.try_state::<PcgStreamState>() else {
        return;
    };
    if state.ticking.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    std::thread::Builder::new()
        .name("pcg-stream".into())
        .spawn(move || {
            let registry = pcg_registry();
            loop {
                std::thread::sleep(TICK);
                tick(&app, &registry);
            }
        })
        .expect("spawn pcg stream tick");
}

/// One tick: plan, evaluate what fits in the budget, release what fell out,
/// publish.
fn tick(app: &AppHandle, registry: &inf_graph::NodeRegistry) {
    let (Some(state), Some(scene), Some(assets)) = (
        app.try_state::<PcgStreamState>(),
        app.try_state::<SceneState>(),
        app.try_state::<AssetState>(),
    ) else {
        return;
    };
    let Some(eye) = state.eye.lock().ok().and_then(|g| *g) else {
        return;
    };

    // ── plan, under a SHORT lock ────────────────────────────────────────────
    //
    // Reading the candidates is a walk of the document's order; it must not be
    // holding the lock while the graphs are read off disk below, or a cold cache
    // would stall every viewport frame for the length of a directory walk.
    let (plan, graph_of, partitioned) = {
        let Ok(doc) = scene.doc.lock() else { return };
        let cands = candidates_of(&doc);
        let partition = doc.settings().partition;
        let just: Vec<VolumeCandidate> = cands.iter().map(|(c, _)| *c).collect();
        let graph_of: HashMap<Uuid, Uuid> = cands.iter().map(|(c, g)| (c.guid, *g)).collect();
        let Ok(mut s) = state.streams.lock() else {
            return;
        };
        (
            s.plan_tick(eye, &just, &partition),
            graph_of,
            !just.is_empty(),
        )
    };
    if !partitioned {
        publish(app, &state, false);
        return;
    }
    if plan.evaluate.is_empty() && plan.release.is_empty() {
        publish(app, &state, false);
        return;
    }

    // ── resolve the graphs the plan needs, OUTSIDE the lock ─────────────────
    let budget_ms = state.streams.lock().map(|s| s.budget_ms()).unwrap_or(8.0);
    let mut programs: Vec<(Uuid, Arc<LoweredPcg>)> = Vec::new();
    for guid in &plan.evaluate {
        let Some(graph) = graph_of.get(guid).copied() else {
            continue;
        };
        let Some(lowered) = lowered_for(&state, &assets, registry, graph) else {
            continue;
        };
        programs.push((*guid, lowered));
    }
    let paths = terrain_paths(&state, &assets);

    // ── evaluate, under the lock, to the budget ─────────────────────────────
    let started = Instant::now();
    let mut evaluated: Vec<Uuid> = Vec::new();
    let mut released: Vec<Uuid> = Vec::new();
    let mut changed = false;
    let mut paged_now = false;
    {
        let Ok(mut doc) = scene.doc.lock() else {
            return;
        };
        if !plan.release.is_empty() {
            for guid in &plan.release {
                let Some(e) = doc.entity_of(*guid) else {
                    continue;
                };
                let w = doc.world_mut().world_mut();
                if let Some(mut vol) = w.get_mut::<PcgVolume>(e) {
                    if !vol.evaluated.is_empty() || !vol.structures.is_empty() {
                        // The same door an evaluation writes through, with
                        // nothing in it: `set_population` is what stamps
                        // `structures_gen`, and the physics bridge and the
                        // sim→render fold both read that stamp to notice a
                        // volume's content went away.
                        vol.set_population(
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            inf_nav::NavGraph::default(),
                            Vec::new(),
                            Vec::new(),
                        );
                        released.push(*guid);
                        changed = true;
                    }
                }
            }
        }
        if !programs.is_empty() {
            doc.world_mut().propagate();
            // Once per document version — see `paged_version`. The stamp is
            // written AFTER this tick's own bump, at the bottom, or every tick
            // would invalidate the stamp it had just written.
            let want_page = state
                .paged_version
                .lock()
                .map(|v| *v != Some(doc.version()))
                .unwrap_or(true);
            if want_page {
                paged_now = true;
                let t = Instant::now();
                let tiles = page_for_pcg(&mut doc, &paths);
                if tiles > 0 {
                    tracing::info!(
                        "inf-studio: pcg stream paged {tiles} terrain tile(s) in {:.1} ms",
                        t.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            for (guid, lowered) in &programs {
                match evaluate_volume_into(&mut doc, *guid, lowered) {
                    Ok(_) => {
                        evaluated.push(*guid);
                        changed = true;
                    }
                    Err(e) => tracing::warn!("pcg stream: {guid} did not evaluate: {e}"),
                }
                // **The budget is a ceiling on STARTING work.** The check is
                // after the first volume, never before it, so a level whose
                // blocks all cost more than the budget still finishes — one a
                // tick — instead of making no progress for ever.
                if started.elapsed().as_secs_f64() * 1000.0 >= budget_ms {
                    break;
                }
            }
        }
        if changed {
            // ONE bump for the whole tick. The viewport's projection is
            // version-gated and rebuilds the entire scene, so bumping per volume
            // would pay for that rebuild once per block.
            doc.bump_version_for_runtime();
        }
        // Stamped only when this tick actually paged: a tick that did nothing
        // but RELEASE volumes has paged no ground, and stamping there would let
        // the next tick that does have work skip the paging it needs.
        if paged_now {
            if let Ok(mut v) = state.paged_version.lock() {
                *v = Some(doc.version());
            }
        }
    }
    let spent = started.elapsed().as_secs_f64() * 1000.0;
    if let Ok(mut s) = state.streams.lock() {
        s.note_tick(&evaluated, &released, spent);
    }
    if changed {
        emit_world_delta(app, &scene);
    }
    if !evaluated.is_empty() || !released.is_empty() {
        if let Ok(s) = state.streams.lock() {
            tracing::info!("inf-studio: {}", s.stats().summary());
        }
    }
    publish(app, &state, !plan.evaluate.is_empty());
}

/// Emit `pcg://stream` when the status changed. A settled editor is silent, on
/// the `viewport://tool-status` precedent: an event per tick would be sixty
/// thousand a session saying the same thing.
fn publish(app: &AppHandle, state: &PcgStreamState, loading: bool) {
    let status = state.status(loading);
    let mut last = match state.last_status.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if last.as_ref() == Some(&status) {
        return;
    }
    *last = Some(status.clone());
    drop(last);
    if let Err(e) = app.emit("pcg://stream", status) {
        tracing::warn!("pcg://stream emit failed: {e}");
    }
}

/// The Streaming overlay's read (wave EDIT1, clause 5), for a panel that mounts
/// after the last event was emitted.
#[tauri::command]
pub async fn pcg_stream_status(
    state: State<'_, PcgStreamState>,
) -> Result<PcgStreamStatusDto, String> {
    let loading = state
        .streams
        .lock()
        .map(|s| s.stats().in_radius_populated < s.stats().in_radius)
        .unwrap_or(false);
    Ok(state.status(loading))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::components::Terrain;
    use inf_ecs::{Vec2d, Vec3d};
    use inf_editor_core::ipc::SpawnKind;
    use inf_editor_core::scene::SceneDoc;
    use inf_editor_core::settlement::zone_payload;
    use inf_pcg::building::ArchetypeId;

    /// Four of the archetypes Harbour City's own blocks are built from.
    const BLOCKS: [ArchetypeId; 4] = [
        ArchetypeId::Office,
        ArchetypeId::Shop,
        ArchetypeId::Apartment,
        ArchetypeId::House,
    ];

    /// A digest of exactly what the clause-1 arm is about: the instances a
    /// volume was populated with, **and the inputs the draw-side LOD ladder
    /// reads off them**.
    ///
    /// Not a count. A count is satisfied by any thousand buildings; what has to
    /// match is which mesh stands where, facing which way, how big — and
    /// `ScatteredSolid`'s extent, because `STRUCTURE_LOD_M` and `INTERIOR_LOD_M`
    /// band a structure by its own size.
    fn digest(doc: &SceneDoc, guid: Uuid) -> (u64, usize, usize) {
        let e = doc.entity_of(guid).expect("volume entity");
        let vol = doc
            .world()
            .world()
            .get::<PcgVolume>(e)
            .expect("volume component");
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

    /// A document shaped like a corner of Harbour City: a flat terrain and one
    /// `PcgVolume` per block, each bound to a REAL committed zone document and
    /// carrying its own seed.
    fn city(order: &[usize]) -> (SceneDoc, Vec<Uuid>, Vec<Arc<LoweredPcg>>) {
        let mut doc = SceneDoc::new();
        // A flat inline terrain, so the height provider takes its terrain arm
        // rather than the flat-`Some(0.0)` fallback — the IB-1 distinction.
        let ground = Uuid::from_u128(0x6100);
        doc.create_with_guid(ground, SpawnKind::Empty, "Ground", None);
        {
            let e = doc.entity_of(ground).unwrap();
            let w = doc.world_mut().world_mut();
            // A flat, non-empty inline terrain: `evaluate_volume_into` picks
            // the first NON-EMPTY `Terrain`, and an empty one sends the height
            // provider down its flat-`Some(0.0)` fallback arm instead (IB-1).
            // Nine tiles, because the blocks below stand up to 120 m apart and
            // a volume off the edge of the terrain evaluates to NOTHING — the
            // first cut of this fixture authored one 64 m tile and measured
            // 660 instances in the first block and zero in the other three,
            // which is the height provider working exactly as IB-1 says.
            let mut data = inf_ecs::TerrainData::new(64, 1.0);
            for ty in -1..=1 {
                for tx in -1..=1 {
                    data.author_tile((tx, ty), |_, _| 0.0);
                }
            }
            w.entity_mut(e).insert(Terrain {
                data,
                ..Terrain::default()
            });
        }
        let registry = pcg_registry();
        let (mut guids, mut programs) = (Vec::new(), Vec::new());
        for &k in order {
            let a = BLOCKS[k];
            let guid = Uuid::from_u128(0x7000 + k as u128);
            doc.create_with_guid(guid, SpawnKind::Empty, a.name(), None);
            let e = doc.entity_of(guid).unwrap();
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(inf_ecs::Transform {
                translation: Vec3d::new(60.0 * (k % 2) as f64, 0.0, 60.0 * (k / 2) as f64),
                rotation: Vec3d::ZERO,
                scale: Vec3d::ONE,
            });
            w.entity_mut(e).insert(PcgVolume {
                extent: Vec2d::new(28.0, 28.0),
                seed: 1_000 + k as u32,
                ..PcgVolume::default()
            });
            let payload = zone_payload(a).expect("the committed zone document");
            let graph = payload.graph().expect("the authored graph");
            let lowered = lower_graph_with(&graph, &registry, &inf_pcg::graph::NoMasks);
            assert!(lowered.ok, "{} did not lower", a.name());
            guids.push(guid);
            programs.push(Arc::new(lowered));
        }
        doc.world_mut().propagate();
        (doc, guids, programs)
    }

    /// **THE CLAUSE-1 ARM.** A volume's population is a pure function of its own
    /// graph, seed, extent and ground — *not* of which volumes the camera
    /// reached first. That is the property that lets a camera drive the
    /// evaluation at all: the shipped player evaluates a block when its cell
    /// activates, the editor when the author flies near it, and the two orders
    /// have nothing to do with each other.
    #[test]
    fn a_blocks_population_does_not_depend_on_which_blocks_were_evaluated_first() {
        let (mut a, ga, pa) = city(&[0, 1, 2, 3]);
        // **The budget table** (wave EDIT1, clause 1's deliverable). Printed
        // rather than asserted: an absolute millisecond ceiling would be a
        // machine gate, and what `EDITOR_PCG_STEP_BUDGET_MS` promises is a bound
        // on what a TICK starts, which `plan`'s own arms cover. What this
        // records is the shape of the cost — how much of a block the editor buys
        // for eight milliseconds.
        let mut cost = Vec::new();
        for (g, p) in ga.iter().zip(pa.iter()) {
            let t = std::time::Instant::now();
            evaluate_volume_into(&mut a, *g, p).expect("evaluate");
            cost.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let forward: Vec<(u64, usize, usize)> = ga.iter().map(|g| digest(&a, *g)).collect();
        let total: f64 = cost.iter().sum();
        println!(
            "EDIT1 clause 1 — cost per block: {} | {:.1} ms for four, {:.1} blocks/s",
            cost.iter()
                .zip(BLOCKS.iter())
                .map(|(ms, a)| format!("{} {ms:.1} ms", a.name()))
                .collect::<Vec<_>>()
                .join(", "),
            total,
            4.0 / (total / 1000.0)
        );
        assert!(
            forward.iter().all(|(_, n, s)| *n > 0 || *s > 0),
            "the committed zone documents placed nothing: {forward:?}"
        );

        // The same four blocks, reached in the opposite order — a camera that
        // flew in from the other end of the street.
        let (mut b, gb, pb) = city(&[3, 2, 1, 0]);
        for (g, p) in gb.iter().zip(pb.iter()) {
            evaluate_volume_into(&mut b, *g, p).expect("evaluate");
        }
        for (i, guid) in ga.iter().enumerate() {
            let back = digest(&b, *guid);
            assert_eq!(
                forward[i],
                back,
                "block {i} ({}) differs by the order it was reached in",
                BLOCKS[i].name()
            );
        }
        println!(
            "EDIT1 clause 1 — order independence: {}",
            forward
                .iter()
                .zip(BLOCKS.iter())
                .map(|((h, n, s), a)| format!("{} {n} inst / {s} solid / {h:016x}", a.name()))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    /// **The seed fold must not accumulate**, and this is the arm that says so.
    ///
    /// `evaluate_volume_into` folds the volume's seed into every rule of the
    /// lowered document. The streaming tick evaluates many volumes from ONE
    /// cached lowering, so folding in place would give the second block a
    /// different city than a session that visited it first — and would make a
    /// block change every time the camera passed it again. Cloning the document
    /// per volume is the fix; re-evaluating the same volume is how it is caught.
    #[test]
    fn re_evaluating_a_block_from_the_same_cached_lowering_gives_the_same_block() {
        let (mut doc, guids, programs) = city(&[0, 1]);
        evaluate_volume_into(&mut doc, guids[0], &programs[0]).expect("first");
        let first = digest(&doc, guids[0]);
        // Somebody else's block in between, from its own program...
        evaluate_volume_into(&mut doc, guids[1], &programs[1]).expect("neighbour");
        // ...and then back, through the SAME `Arc<LoweredPcg>` the cache holds.
        evaluate_volume_into(&mut doc, guids[0], &programs[0]).expect("again");
        let again = digest(&doc, guids[0]);
        assert_eq!(
            first, again,
            "the cached lowering accumulated a seed fold: {first:?} then {again:?}"
        );
    }
}
