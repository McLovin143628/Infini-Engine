//! **Biome → PCG binding** (P19.3): painted biomes drive procedural population.
//!
//! P19.2 gave a terrain a [`BiomeSet`] and a per-sample id layer, and each
//! [`BiomeDef`] carries a `pcg_graph`. This module is the thing that finally
//! *uses* both: for a terrain with a biome set, every biome's graph evaluates
//! **over the region its own id is painted on**, and the results merge into one
//! deterministic population.
//!
//! ```text
//!   BiomeSet ──┬─ biome 1 · graph A ─┬─ mask(id=1, feather) × A's samplers ─┐
//!              ├─ biome 2 · graph B ─┼─ mask(id=2, feather) × B's samplers ─┼─▶ instances
//!              └─ biome 7 · (none)   ┘  (contributes nothing)               ┘
//!                    ascending id order
//! ```
//!
//! # A sibling of the volume path, not a replacement
//!
//! A [`PcgVolume`] scatters *one* graph over *a box the author placed*. A binding
//! scatters *many* graphs over *the regions the author painted*. Both exist, both
//! run at the same two moments (the editor's evaluate command and the player's
//! load-time pass), and neither reads the other's output. The volume stays the
//! right tool for "put this here"; the binding is "the world grows what belongs
//! on it".
//!
//! # Determinism
//!
//! Three rules, all machine-checked:
//!
//! 1. **Dispatch order is ascending biome id.** [`BiomeBinding::new`] sorts, so
//!    the merged list does not depend on the order the set happened to declare
//!    its biomes in, nor on a hash map's iteration.
//! 2. **The counter-hash is preserved, with the biome id folded in.** Each rule's
//!    scatter seed becomes `hash(seed, BIOME_SALT, id)` — see [`biome_seed`].
//!    Placement stays a pure function of an integer coordinate tuple (the
//!    doctrine); the tuple simply grew the biome.
//!
//!    **What actually prevents co-placement today is disjointness, not the
//!    seed.** [`TerrainFields::biome_id`]
//!    answers with exactly **one** id per world position, and
//!    [`BiomeMask`](crate::sampler::BiomeMask) scores `0` unless that id matches —
//!    so at any point at most one biome's mask is positive, and the feather only
//!    *thins* a biome inside its own region rather than extending it into its
//!    neighbour's. There is no both-masks-positive band in P19.3. The seed fold is
//!    a **forward guard**, not the present mechanism: the moment anything makes
//!    the masks overlap — a persisted per-level feather that blends across the
//!    border, a soft id field, a future "biome influence" sampler — two biomes
//!    sharing one graph would otherwise stamp the identical instance at the
//!    identical position, and nothing else in the pipeline would catch it. It is
//!    gated on its own terms (`the_biome_id_is_part_of_the_counter_hash` runs one
//!    document as two ids over the *same* fully-overlapping region), so it cannot
//!    quietly stop working before the day it is needed.
//! 3. **Masking is multiplicative, so it only ever removes.** The scatter
//!    kernel's acceptance draw does not depend on the density value, so wrapping
//!    a sampler in `Multiply(mask, …)` can never move an instance — it can only
//!    reject one. A biome's population is therefore a *subset* of what its graph
//!    would place unmasked, which is what makes the region property testable.
//!
//! # The border blend
//!
//! [`BiomeMask`](crate::sampler::BiomeMask) does the actual feathering (see its
//! docs for the distance metric). The binding's job is only to choose the width,
//! and it takes it as a parameter defaulting to [`DEFAULT_BIOME_FEATHER`].
//!
//! **Where the width is authored, honestly stated:** nowhere persistent, yet. A
//! per-level value would have to live on `Terrain` (a persisted component field —
//! bincode is positional, so schema v17 in both codec mirrors) or on `BiomeSet`
//! (a `.inf_biomes` schema bump with a frozen record), and P19.3 buys no schema
//! bump. What *is* authored today is per-graph: the `mask.biome` node carries its
//! own `feather` param, so a graph that wants a specific blend states it and
//! multiplies it in. Persisting a per-level default is a P19.4+ remainder.
//!
//! [`PcgVolume`]: the ECS component
//! [`BiomeSet`]: inf_terrain::BiomeSet
//! [`BiomeDef`]: inf_terrain::BiomeDef

use std::collections::BTreeMap;

use inf_terrain::{BiomeSet, TerrainData, TileKey, UNASSIGNED_BIOME};
use uuid::Uuid;

use crate::fields::{OffsetTerrain, TerrainFields};
use crate::hash::Hash64;
use crate::height::{FnHeight, HeightProvider, FN_HEIGHT_NORMAL_EPS};
use crate::rules::{evaluate_with_in, PcgDocument, SamplerDef};
use crate::sampler::MAX_FEATHER_SAMPLES;
use crate::scatter::{PcgInstance, Region};

/// The engine-wide default biome-border blend width, in **metres**.
///
/// Eight metres is a little over a tree crown: wide enough that a forest visibly
/// thins into a meadow rather than stopping at a line, narrow enough that the
/// [`BiomeMask`](crate::sampler::BiomeMask) feather search stays a handful of
/// lattice rings at the 1–2 m spacings terrain ships with.
pub const DEFAULT_BIOME_FEATHER: f64 = 8.0;

/// The salt that separates a biome dispatch from a bare graph evaluation in the
/// counter-hash stream. Arbitrary and fixed — changing it repositions every
/// biome-bound instance in every project, so it is a constant, never a knob.
const BIOME_SALT: u64 = 0xB10E;

/// The scatter seed a rule runs under when dispatched for `biome`.
///
/// Folding the id into the seed is what makes the biome part of the coordinate
/// tuple every draw is a pure function of. Two biomes sharing one graph therefore
/// produce two *different* populations rather than the same one twice.
///
/// **Today that is a forward guard, not a live fix** — biome masks are disjoint
/// by construction (one id per position), so nowhere in P19.3 are two masks
/// positive at once. It exists because the first thing that makes them overlap —
/// the deferred per-level feather blending across a border, a soft id field —
/// would otherwise produce co-located duplicates with nothing to catch them. See
/// the module docs.
pub fn biome_seed(seed: u64, biome: u8) -> u64 {
    Hash64::new(seed)
        .mix_u64(BIOME_SALT)
        .mix_u64(biome as u64)
        .finish()
}

/// One biome's bound graph: the id painted on the terrain and the document its
/// `.inf_pcg` lowers to.
#[derive(Debug, Clone, PartialEq)]
pub struct BiomeGraph {
    /// The painted id this graph populates.
    pub id: u8,
    /// The lowered runtime document (`lower_graph(…).document`, or a v1 payload's
    /// stored mirror).
    pub document: PcgDocument,
}

/// The set of biome→graph dispatches a terrain evaluates, plus the border blend
/// width they share.
#[derive(Debug, Clone, PartialEq)]
pub struct BiomeBinding {
    /// Dispatches, **sorted ascending by id** — the deterministic merge order.
    graphs: Vec<BiomeGraph>,
    /// Border blend width in metres (see [`DEFAULT_BIOME_FEATHER`]).
    pub feather: f64,
}

impl BiomeBinding {
    /// Bind `graphs`, sorting them ascending by id (the merge order) and dropping
    /// any that names [`UNASSIGNED_BIOME`] — id `0` means *no biome*, so it can
    /// never own a population.
    ///
    /// A duplicate id keeps **both** dispatches (a set cannot produce one —
    /// `BiomeSet::validate` rejects duplicates — but this type does not assume its
    /// caller went through a set, and silently dropping content is worse than
    /// running it twice). The sort is stable, so declaration order breaks the tie.
    pub fn new(graphs: impl IntoIterator<Item = BiomeGraph>, feather: f64) -> Self {
        let mut graphs: Vec<BiomeGraph> = graphs
            .into_iter()
            .filter(|g| g.id != UNASSIGNED_BIOME)
            .collect();
        graphs.sort_by_key(|g| g.id);
        Self { graphs, feather }
    }

    /// Build a binding from a [`BiomeSet`] by resolving each biome's `pcg_graph`
    /// asset GUID through `resolve`.
    ///
    /// **This is the parity seam.** The editor's evaluate command and the player's
    /// load-time pass differ only in how they *fetch* a `.inf_pcg` (a content-root
    /// file vs. a cooked pack entry); everything from the GUID onward — which
    /// biomes dispatch, in what order, under which feather — is this one function,
    /// so the two cannot drift. A biome with no `pcg_graph`, or one whose graph
    /// does not resolve, contributes nothing rather than failing the load: a
    /// dangling reference is the cook's deduplicated advisory (P19.2), not a
    /// runtime error.
    pub fn from_set(
        set: &BiomeSet,
        feather: f64,
        mut resolve: impl FnMut(Uuid) -> Option<PcgDocument>,
    ) -> Self {
        let graphs = set.biomes.iter().filter_map(|b| {
            let document = resolve(b.pcg_graph?)?;
            Some(BiomeGraph { id: b.id, document })
        });
        Self::new(graphs.collect::<Vec<_>>(), feather)
    }

    /// The bound dispatches, ascending by id.
    pub fn graphs(&self) -> &[BiomeGraph] {
        &self.graphs
    }

    /// `true` when nothing is bound — the common case for a terrain with no biome
    /// set, or one whose biomes name no graphs. Callers skip the whole pass.
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Evaluate every bound biome over `region` and concatenate the results in
    /// ascending biome-id order.
    ///
    /// `fields` supplies the painted ids (an [`OffsetTerrain`]
    /// at the terrain entity's world origin, in practice); `height` supplies the
    /// ground the instances land on. With no ids anywhere the masks all score `0`
    /// and the result is empty — a terrain nobody painted grows nothing.
    pub fn evaluate(
        &self,
        height: &dyn HeightProvider,
        fields: &dyn TerrainFields,
        region: Region,
    ) -> Vec<PcgInstance> {
        self.evaluate_in(inf_core::global(), height, fields, region)
    }

    /// [`evaluate`](Self::evaluate) on a **caller-supplied pool** — the seam the
    /// determinism guard drives, mirroring
    /// [`scatter_region_in`](crate::scatter::scatter_region_in) two layers down.
    ///
    /// The merged population is byte-identical for any thread count. Exposed so
    /// the pool-size property can be proved through the *real* path — every
    /// biome, every layer, every rule, the kind picks and the concatenation order
    /// — rather than through a rule lifted out of a document by hand, which is
    /// the version of the test that passes while the thing it claims to cover is
    /// broken.
    pub fn evaluate_in(
        &self,
        pool: &inf_core::JobPool,
        height: &dyn HeightProvider,
        fields: &dyn TerrainFields,
        region: Region,
    ) -> Vec<PcgInstance> {
        let mut out = Vec::new();
        for graph in &self.graphs {
            let bound = bind_document(&graph.document, graph.id, self.feather);
            out.extend(evaluate_with_in(pool, &bound, height, fields, region));
        }
        out
    }

    /// **The population of a terrain's RESIDENT ground** — one pass, tile by
    /// tile, memoized in `cache`, and a pure function of what is resident.
    ///
    /// This is the door both hosts evaluate a biome binding through, and it is
    /// the door the island phase's wave I7b opened. Before it, both hosts asked
    /// [`evaluate`](Self::evaluate) for one region taken from
    /// [`TerrainData::xz_bounds`], which is correct exactly once — at load, for a
    /// terrain whose tiles are all in memory. **A streamed terrain ships no
    /// tiles**, so the bounds were `None` and a 51 km² island grew nothing; and
    /// once the ground *does* page in, "the bounds" keeps moving, so the question
    /// is not *what region* but *which tiles*.
    ///
    /// Returns `true` when [`population`](BiomeScatterCache::population) may
    /// have changed, so a caller can leave a component alone on the
    /// overwhelming majority of steps where nothing paged. It is a *conservative*
    /// answer in one direction only: a residency change that happens to grow the
    /// identical forest (a tile arriving with nothing on it) reports `true` and
    /// costs a redundant copy, and no change is ever reported as no change.
    ///
    /// # Why per tile is EXACT and not an approximation
    ///
    /// [`scatter_region_in`](crate::scatter::scatter_region_in) walks a lattice
    /// derived from **world** coordinates (`floor(x / cell_size)`) and clips each
    /// candidate against the region **half-open** — its own comment says
    /// *"seamless tiling, no double placement"*. So the union of the populations
    /// of two abutting boxes is exactly the population of their union, instance
    /// for instance. Splitting the walk by tile therefore changes *nothing* about
    /// which instances exist; it changes only the order they are concatenated in
    /// (ascending tile coordinate, then ascending biome id) and what has to be
    /// recomputed when the ground moves.
    ///
    /// # Why the key carries the NEIGHBOURS
    ///
    /// A candidate near a tile's edge reads terrain *outside* that tile:
    /// [`BiomeMask`](crate::sampler::BiomeMask)'s feather search walks up to
    /// [`MAX_FEATHER_SAMPLES`] lattice rings, and the slope filter's numerical
    /// normal probes [`FN_HEIGHT_NORMAL_EPS`] either side. Evaluate a tile alone
    /// and those reads answer `None` — off-terrain — so the candidate is
    /// rejected; evaluate it with its neighbours resident and it is not. Keying
    /// only on the tile's own stamp would therefore memoize *the answer for the
    /// residency it first arrived under*, and a player driving east would grow a
    /// different forest from one driving west. That is P21's first-sight law, and
    /// the fix is the same one: the key names every tile the evaluation can read
    /// (a [`neighbour_rings`] neighbourhood, sized from the reach above), so a
    /// neighbour arriving re-keys its neighbours and they re-evaluate.
    ///
    /// A non-resident neighbour contributes stamp `0`
    /// ([`TerrainData::tile_version`]'s documented answer), so *absence* is part
    /// of the key rather than invisible to it.
    pub fn refresh_resident(
        &self,
        data: &TerrainData,
        origin: glam::DVec3,
        cache: &mut BiomeScatterCache,
    ) -> bool {
        self.refresh_resident_in(inf_core::global(), data, origin, cache)
    }

    /// [`refresh_resident`](Self::refresh_resident) on a **caller-supplied
    /// pool** — the seam the pool-invariance guard drives, mirroring
    /// [`evaluate_in`](Self::evaluate_in).
    pub fn refresh_resident_in(
        &self,
        pool: &inf_core::JobPool,
        data: &TerrainData,
        origin: glam::DVec3,
        cache: &mut BiomeScatterCache,
    ) -> bool {
        // Resident level-0 coordinates, ascending. Sorted rather than taken in
        // map order: the merge order must be a function of the resident set, not
        // of a hash seed.
        let mut resident: Vec<(i32, i32)> = data.tiles().map(|(c, _)| *c).collect();
        resident.sort_unstable();

        // The memo is bounded by residency: a tile that paged out takes its
        // slice with it. (A pin with no release is a leak with a deadline —
        // P21.4's law, met here.)
        cache.tiles.retain(|c, _| data.has_tile(*c));

        if self.graphs.is_empty() || resident.is_empty() {
            let changed = !cache.population.is_empty();
            cache.tiles.clear();
            cache.population.clear();
            cache.resident = resident;
            return changed;
        }

        let rings = neighbour_rings(data);
        let stale: Vec<((i32, i32), Vec<u64>)> = resident
            .iter()
            .filter_map(|&c| {
                let stamps = neighbourhood_stamps(data, c, rings);
                match cache.tiles.get(&c) {
                    Some(t) if t.stamps == stamps => None,
                    _ => Some((c, stamps)),
                }
            })
            .collect();
        if stale.is_empty() && cache.resident == resident {
            return false;
        }

        if !stale.is_empty() {
            let fields = OffsetTerrain::new(data, origin);
            let height = FnHeight::new(|x, z| fields.height_at(x, z));
            let span = data.tile_span();
            // One fan-out over the stale tiles; each tile's own biome walk uses
            // the same pool, which rayon runs inline when it is already on it.
            // `parallel_map_ref` is the deterministic in-order map, so the
            // result does not depend on the thread count.
            let fresh: Vec<Vec<PcgInstance>> = pool.parallel_map_ref(&stale, |(c, _)| {
                let o = data.tile_origin_xz(*c);
                let (x0, z0) = (o.x + origin.x, o.y + origin.z);
                self.evaluate_in(
                    pool,
                    &height,
                    &fields,
                    Region::from_xz(x0, z0, x0 + span, z0 + span),
                )
            });
            cache.evaluated += stale.len() as u64;
            for ((c, stamps), instances) in stale.into_iter().zip(fresh) {
                cache.tiles.insert(c, CachedTile { stamps, instances });
            }
        }

        cache.population.clear();
        for c in &resident {
            if let Some(t) = cache.tiles.get(c) {
                cache.population.extend_from_slice(&t.instances);
            }
        }
        cache.resident = resident;
        true
    }

    /// [`refresh_resident`](Self::refresh_resident) with **no memo** — the
    /// one-shot form, for a load-time pass and for an author pressing "evaluate".
    pub fn evaluate_resident(&self, data: &TerrainData, origin: glam::DVec3) -> Vec<PcgInstance> {
        let mut cache = BiomeScatterCache::default();
        self.refresh_resident(data, origin, &mut cache);
        std::mem::take(&mut cache.population)
    }
}

/// How far past a candidate's own position a biome dispatch may read the
/// terrain, in **metres**, for a terrain whose samples are `spacing` apart.
///
/// Two readers set it, and both are bounded rather than assumed:
///
/// * [`BiomeMask`](crate::sampler::BiomeMask)'s feather search walks at most
///   [`MAX_FEATHER_SAMPLES`] lattice rings **whatever feather an author writes**
///   — the cap is the mask's own, so this is an upper bound for every document
///   rather than for the one in front of us;
/// * the numerical normal a slope filter reads probes
///   [`FN_HEIGHT_NORMAL_EPS`] either side.
///
/// It is what makes [`BiomeBinding::refresh_resident`]'s per-tile memo exact.
pub fn scatter_reach_m(spacing: f64) -> f64 {
    let lattice = if spacing.is_finite() && spacing > 0.0 {
        MAX_FEATHER_SAMPLES as f64 * spacing
    } else {
        0.0
    };
    lattice + FN_HEIGHT_NORMAL_EPS
}

/// How many rings of neighbouring tiles a tile's evaluation can read, i.e. the
/// Chebyshev radius [`BiomeBinding::refresh_resident`] keys on.
///
/// `1` for every grid this engine ships (a 257²-sample tile at 1 m is 256 m
/// across against a 64.1 m reach); larger only for a terrain whose tiles are
/// smaller than the reach, which is a test fixture rather than content.
pub fn neighbour_rings(data: &TerrainData) -> i32 {
    let span = data.tile_span();
    // `!is_finite() || <= 0.0` rather than `!(span > 0.0)`: the two agree on
    // every value including NaN, and only one of them is a negated comparison on
    // a partially-ordered type.
    if !span.is_finite() || span <= 0.0 {
        return 1;
    }
    let rings = (scatter_reach_m(data.meters_per_sample()) / span).ceil();
    if rings.is_finite() {
        (rings as i32).max(1)
    } else {
        1
    }
}

/// The stamps of `coord` and every tile within `rings` of it, in a fixed
/// (dz-major, dx-minor) order — the content key of one tile's slice.
fn neighbourhood_stamps(data: &TerrainData, coord: (i32, i32), rings: i32) -> Vec<u64> {
    let side = (2 * rings + 1) as usize;
    let mut out = Vec::with_capacity(side * side);
    for dz in -rings..=rings {
        for dx in -rings..=rings {
            out.push(data.tile_version(TileKey::lod0((
                coord.0.saturating_add(dx),
                coord.1.saturating_add(dz),
            ))));
        }
    }
    out
}

/// One resident tile's memoized slice of the population.
#[derive(Debug, Clone, PartialEq)]
struct CachedTile {
    /// See [`neighbourhood_stamps`] — the tile's own stamp and its neighbours'.
    stamps: Vec<u64>,
    instances: Vec<PcgInstance>,
}

/// The per-tile memo [`BiomeBinding::refresh_resident`] keeps between calls.
///
/// Session state: a stamp is runtime cache identity and never content
/// ([`TerrainData::tile_version`] says so), so nothing here is serialized,
/// hashed or compared — what comes *out* of it is a pure function of the
/// resident tiles' contents, which is the property the whole design is for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BiomeScatterCache {
    tiles: BTreeMap<(i32, i32), CachedTile>,
    /// The resident coordinates the current [`population`](Self::population) was
    /// concatenated from, ascending.
    resident: Vec<(i32, i32)>,
    population: Vec<PcgInstance>,
    evaluated: u64,
}

impl BiomeScatterCache {
    /// The merged population of the resident ground, as of the last
    /// [`refresh_resident`](BiomeBinding::refresh_resident).
    pub fn population(&self) -> &[PcgInstance] {
        &self.population
    }

    /// How many resident tiles the memo holds a slice for.
    pub fn resident_tiles(&self) -> usize {
        self.tiles.len()
    }

    /// **The engagement counter**: how many tile evaluations have been run since
    /// this cache was created. A gate reads it to tell "the memo answered" from
    /// "nothing was asked" — the two readings a hit rate cannot distinguish.
    pub fn tiles_evaluated(&self) -> u64 {
        self.evaluated
    }
}

/// Rewrite `doc` for dispatch under `biome`: intersect every rule's sampler with
/// the biome's region mask, and fold the id into every rule's scatter seed.
///
/// Kept public so a caller can inspect exactly what a biome runs — and so the
/// two rewrites (mask, seed) live in one place instead of being re-derived at
/// each evaluation site.
///
/// The mask is the **first** operand of the `Multiply` so a reader of a lowered
/// document sees `biome × authored`, matching the dispatch's causality; the
/// product is commutative, so nothing depends on it.
pub fn bind_document(doc: &PcgDocument, biome: u8, feather: f64) -> PcgDocument {
    let mut bound = doc.clone();
    for layer in &mut bound.layers {
        for rule in &mut layer.rules {
            let mask = SamplerDef::Biome { id: biome, feather };
            let authored = std::mem::replace(&mut rule.sampler, SamplerDef::Constant(1.0));
            rule.sampler = SamplerDef::Multiply(Box::new(mask), Box::new(authored));
            rule.scatter.seed = biome_seed(rule.scatter.seed, biome);
        }
    }
    bound
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::OffsetTerrain;
    use crate::height::FnHeight;
    use crate::rules::{PcgKind, PcgLayer, PcgRule};
    use crate::scatter::{RotationMode, ScatterParams};
    use glam::DVec3;
    use inf_terrain::{BiomeDef, TerrainData};

    /// A 4 × 4-tile flat terrain (res 9, 1 m spacing → 32 m square) whose left
    /// half is painted biome 1 and right half biome 2.
    fn split_terrain() -> TerrainData {
        let mut t = TerrainData::new(9, 1.0);
        for tz in 0..4 {
            for tx in 0..4 {
                t.author_tile((tx, tz), |_, _| 0.0);
            }
        }
        let res = t.tile_resolution();
        let span = (res - 1) as i32;
        let coords: Vec<(i32, i32)> = t.tiles().map(|(&c, _)| c).collect();
        for coord in coords {
            let tile = t.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    let gx = coord.0 * span + i as i32;
                    tile.set_biome_sample(res, i, j, if gx < 16 { 1 } else { 2 });
                }
            }
        }
        t
    }

    fn flat() -> FnHeight<impl Fn(f64, f64) -> Option<f64> + Send + Sync> {
        FnHeight::new(|_, _| Some(0.0))
    }

    fn doc(seed: u64, mesh: u128) -> PcgDocument {
        PcgDocument::single_layer(
            "veg",
            vec![PcgRule {
                name: "r".into(),
                sampler: SamplerDef::Constant(1.0),
                scatter: ScatterParams {
                    seed,
                    cell_size: 8.0,
                    base_density: 0.25,
                    jitter: 1.0,
                    align_to_normal: false,
                    scale_range: (1.0, 1.0),
                    rotation: RotationMode::RandomYaw,
                    altitude_offset: 0.0,
                },
                kinds: vec![PcgKind::mesh(Uuid::from_u128(mesh))],
            }],
        )
    }

    fn region() -> Region {
        Region::from_xz(0.0, 0.0, 32.0, 32.0)
    }

    /// The x where `split_terrain` switches from biome 1 to biome 2.
    const BORDER_X: f64 = 16.0;

    /// One rule with explicit seed / kind / density / cell size.
    fn rule_with(seed: u64, mesh: u128, base_density: f64, cell_size: f64) -> PcgRule {
        PcgRule {
            name: format!("r{seed}"),
            sampler: SamplerDef::Constant(1.0),
            scatter: ScatterParams {
                seed,
                cell_size,
                base_density,
                jitter: 1.0,
                align_to_normal: false,
                scale_range: (1.0, 1.0),
                rotation: RotationMode::RandomYaw,
                altitude_offset: 0.0,
            },
            kinds: vec![PcgKind::mesh(Uuid::from_u128(mesh))],
        }
    }

    /// **The region property**: a painted biome populates its own half and
    /// nothing outside it.
    #[test]
    fn a_biome_populates_only_where_its_id_is_painted() {
        let t = split_terrain();
        let fields = OffsetTerrain::new(&t, DVec3::ZERO);
        let b = BiomeBinding::new(
            vec![BiomeGraph {
                id: 1,
                document: doc(11, 1),
            }],
            0.0,
        );
        let out = b.evaluate(&flat(), &fields, region());
        assert!(!out.is_empty(), "the painted half must populate");
        for i in &out {
            assert!(
                t.biome_at(glam::DVec2::new(i.pos.x, i.pos.z)) == Some(1),
                "instance at {:?} is outside biome 1",
                i.pos
            );
        }
        // …and the other biome's own dispatch lands entirely on the other side.
        let b2 = BiomeBinding::new(
            vec![BiomeGraph {
                id: 2,
                document: doc(11, 2),
            }],
            0.0,
        );
        let out2 = b2.evaluate(&flat(), &fields, region());
        assert!(!out2.is_empty());
        assert!(out2.iter().all(|i| i.pos.x >= 15.0));
        // Two disjoint hard-edged halves ⇒ no instance is in both lists.
        assert!(out.iter().all(|a| !out2.iter().any(|b| b.pos == a.pos)));
    }

    /// An unpainted terrain grows **nothing** — the masks all score `0`.
    #[test]
    fn an_unpainted_terrain_grows_nothing() {
        let mut t = TerrainData::new(9, 1.0);
        for tz in 0..4 {
            for tx in 0..4 {
                t.author_tile((tx, tz), |_, _| 0.0);
            }
        }
        assert!(t.biomes_are_default());
        let fields = OffsetTerrain::new(&t, DVec3::ZERO);
        let b = BiomeBinding::new(
            vec![BiomeGraph {
                id: 1,
                document: doc(3, 1),
            }],
            DEFAULT_BIOME_FEATHER,
        );
        assert!(b.evaluate(&flat(), &fields, region()).is_empty());
    }

    /// **The feather band blends monotonically.** Walking from the border into
    /// biome 1, the mask density must be non-decreasing and reach 1 by the
    /// feather width — so the population thins toward the border rather than
    /// stopping at a line.
    #[test]
    fn the_feather_band_blends_density_monotonically() {
        use crate::sampler::{BiomeMask, DensityField};
        let t = split_terrain();
        let fields = OffsetTerrain::new(&t, DVec3::ZERO);
        let mask = BiomeMask {
            fields: &fields,
            id: 1,
            feather: 6.0,
        };
        // x = 16 is the first biome-2 column, so the border sits at x ≈ 15.5.
        let mut prev = -1.0;
        for step in 0..=12 {
            let x = 15.0 - step as f64 * 0.5;
            let d = mask.density(x, 8.0);
            assert!(
                d >= prev - 1e-12,
                "density fell going deeper: x={x} {d} < {prev}"
            );
            prev = d;
        }
        assert!(
            mask.density(15.0, 8.0) < 0.2,
            "right at the border the blend must be near zero, got {}",
            mask.density(15.0, 8.0)
        );
        assert_eq!(mask.density(8.0, 8.0), 1.0, "deep inside is fully dense");
        assert_eq!(mask.density(20.0, 8.0), 0.0, "outside is zero");
        // A feathered dispatch places strictly fewer than a crisp one (the mask
        // only ever removes) and still places something.
        let crisp = BiomeBinding::new(
            vec![BiomeGraph {
                id: 1,
                document: doc(9, 1),
            }],
            0.0,
        )
        .evaluate(&flat(), &fields, region())
        .len();
        let soft = BiomeBinding::new(
            vec![BiomeGraph {
                id: 1,
                document: doc(9, 1),
            }],
            6.0,
        )
        .evaluate(&flat(), &fields, region())
        .len();
        assert!(soft > 0 && soft < crisp, "crisp={crisp} soft={soft}");
    }

    /// Dispatch order is **ascending id**, whatever order the graphs arrive in,
    /// and the merged list is byte-identical across runs.
    #[test]
    fn dispatch_order_is_ascending_id_and_the_merge_is_deterministic() {
        let t = split_terrain();
        let fields = OffsetTerrain::new(&t, DVec3::ZERO);
        let g1 = BiomeGraph {
            id: 1,
            document: doc(5, 1),
        };
        let g2 = BiomeGraph {
            id: 2,
            document: doc(6, 2),
        };
        let a = BiomeBinding::new(vec![g2.clone(), g1.clone()], 0.0);
        let b = BiomeBinding::new(vec![g1, g2], 0.0);
        assert_eq!(
            a.graphs().iter().map(|g| g.id).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(a, b, "declaration order must not survive the constructor");
        let ia = a.evaluate(&flat(), &fields, region());
        let ib = b.evaluate(&flat(), &fields, region());
        assert!(!ia.is_empty());
        assert_eq!(ia, ib);
        assert_eq!(ia, a.evaluate(&flat(), &fields, region()), "pure");
        // Biome 1 owns the left half and is dispatched first, so the merged list
        // starts on the left.
        assert!(ia[0].pos.x < 16.0);
    }

    /// A terrain that is **entirely one biome**, so a dispatch for that id covers
    /// the whole region at full density — the only way to make two biomes' masks
    /// overlap *completely*, which is what the seed fold has to survive.
    struct UniformBiome(u8);

    impl TerrainFields for UniformBiome {
        fn biome_id(&self, x: f64, z: f64) -> Option<u8> {
            ((0.0..32.0).contains(&x) && (0.0..32.0).contains(&z)).then_some(self.0)
        }
        fn sample_spacing(&self) -> f64 {
            1.0
        }
    }

    /// **The seed fold does something, and here is the case that proves it.**
    ///
    /// The *present* guarantee against two biomes co-placing is disjointness —
    /// one id per position, so at most one mask is positive anywhere (asserted by
    /// `a_biome_populates_only_where_its_id_is_painted`). That means a
    /// two-biome-on-a-split-terrain test would pass with the fold deleted, which
    /// makes it worthless as a gate for the fold.
    ///
    /// So this runs **one document over one region twice** — as id 1 against a
    /// terrain that is all id 1, then as id 2 against a terrain that is all id 2 —
    /// i.e. two dispatches whose masks are `1.0` over exactly the same ground.
    /// Everything except the biome id is identical, so any difference in the
    /// output is the fold and nothing else. Delete `biome_seed` from
    /// `bind_document` and this fails; the split-terrain tests do not.
    #[test]
    fn the_biome_id_is_part_of_the_counter_hash() {
        assert_ne!(biome_seed(7, 1), biome_seed(7, 2));
        assert_ne!(biome_seed(7, 1), 7, "a dispatch is not a bare evaluation");
        assert_eq!(biome_seed(7, 1), biome_seed(7, 1), "pure");

        let shared = doc(42, 1);
        let positions = |id: u8| -> Vec<(u64, u64)> {
            let fields = UniformBiome(id);
            let out = BiomeBinding::new(
                vec![BiomeGraph {
                    id,
                    document: shared.clone(),
                }],
                0.0,
            )
            .evaluate(&flat(), &fields, region());
            out.iter()
                .map(|i| (i.pos.x.to_bits(), i.pos.z.to_bits()))
                .collect()
        };
        let as_one = positions(1);
        let as_two = positions(2);
        assert!(as_one.len() > 50, "the fixture must place something");
        // Same document, same region, same fully-covering mask — only the id
        // differs, and the populations must not be the same points.
        assert_ne!(
            as_one, as_two,
            "one document dispatched as two biomes produced the IDENTICAL \
             placement — the biome id is not reaching the counter-hash"
        );
        let overlap = as_one.iter().filter(|p| as_two.contains(p)).count();
        assert!(
            overlap * 4 < as_one.len(),
            "the two dispatches co-placed {overlap} of {} instances — the fold is \
             barely decorrelating",
            as_one.len()
        );
        // …and each dispatch is still a pure function of its id.
        assert_eq!(as_one, positions(1));
    }

    // ────────────────────────── the resident walk (island wave I7b) ──────

    /// A 3 × 3-tile terrain (res 9, 1 m spacing → 8 m tiles, 24 m square) on a
    /// gentle slope, **every sample painted biome 1**.
    ///
    /// One biome everywhere on purpose: what these arms are about is which
    /// *tiles* are resident, so the only thing that may vary the population is
    /// residency. A slope is there because the island's own cover document is
    /// slope-limited, so the numerical normal — the second half of the reach —
    /// is exercised rather than assumed.
    fn paged_terrain() -> TerrainData {
        let mut t = TerrainData::new(9, 1.0);
        for tz in 0..3 {
            for tx in 0..3 {
                t.author_tile((tx, tz), |x, z| 0.05 * x + 0.02 * z);
            }
        }
        let res = t.tile_resolution();
        let coords: Vec<(i32, i32)> = t.tiles().map(|(&c, _)| c).collect();
        for coord in coords {
            let tile = t.get_tile_mut(coord).unwrap();
            for j in 0..res {
                for i in 0..res {
                    tile.set_biome_sample(res, i, j, 1);
                }
            }
        }
        t
    }

    /// The same terrain with only `coords` resident — what a streamed terrain's
    /// `Terrain.data` holds part-way through a drive.
    fn paged_subset(full: &TerrainData, coords: &[(i32, i32)]) -> TerrainData {
        let mut t = TerrainData::new(full.tile_resolution(), full.meters_per_sample());
        for &c in coords {
            page_in(&mut t, full, c);
        }
        t
    }

    /// Page one tile of `full` into `data` — **the streamer's own door**
    /// ([`TerrainData::insert_resident_tile`]), so an arrival stamps exactly
    /// the tile that arrived and leaves its neighbours' stamps alone.
    ///
    /// Building a *fresh* `TerrainData` per arrival instead would re-stamp every
    /// tile on every step, which makes every key move and hides precisely the
    /// defect these arms exist for. (The first draft did that, and the
    /// `rings = 0` mutation passed it.)
    fn page_in(data: &mut TerrainData, full: &TerrainData, coord: (i32, i32)) {
        let tile = full
            .get_tile(coord)
            .expect("the fixture has that tile")
            .clone();
        data.insert_resident_tile(TileKey::lod0(coord), tile);
    }

    /// A slope-limited cover document, the shape `island_cover_document` builds.
    fn cover(seed: u64) -> PcgDocument {
        PcgDocument::single_layer(
            "cover",
            vec![PcgRule {
                name: "cover".into(),
                sampler: SamplerDef::Slope {
                    min_deg: 0.0,
                    max_deg: 34.0,
                    feather_deg: 6.0,
                },
                scatter: ScatterParams {
                    seed,
                    cell_size: 8.0,
                    base_density: 1.0,
                    jitter: 1.0,
                    align_to_normal: false,
                    scale_range: (1.0, 1.0),
                    rotation: RotationMode::RandomYaw,
                    altitude_offset: 0.0,
                },
                kinds: vec![PcgKind::mesh(Uuid::from_u128(1))],
            }],
        )
    }

    fn cover_binding() -> BiomeBinding {
        BiomeBinding::new(
            vec![BiomeGraph {
                id: 1,
                document: cover(4242),
            }],
            DEFAULT_BIOME_FEATHER,
        )
    }

    /// Positions as bit patterns, **sorted** — so two readings are compared as
    /// sets of places rather than as lists in one order (the I1 law: positions,
    /// not counts).
    fn places(v: &[PcgInstance]) -> Vec<(u64, u64, u64)> {
        let mut p: Vec<(u64, u64, u64)> = v
            .iter()
            .map(|i| (i.pos.x.to_bits(), i.pos.y.to_bits(), i.pos.z.to_bits()))
            .collect();
        p.sort_unstable();
        p
    }

    /// **THE UNION OF THE TILES IS THE WHOLE.** Splitting the walk by tile
    /// changes *which* instances are placed in no way at all — the scatter
    /// lattice is world-anchored and its region clip is half-open, so two
    /// abutting boxes tile seamlessly.
    ///
    /// If this ever fails, `refresh_resident` is not a re-ordering of
    /// `evaluate` and every "the streamed island grows what the author
    /// previewed" sentence in this repository is wrong.
    #[test]
    fn a_per_tile_walk_places_exactly_what_one_region_places() {
        let t = paged_terrain();
        let b = cover_binding();
        let fields = OffsetTerrain::new(&t, DVec3::ZERO);
        let height = FnHeight::new(|x, z| fields.height_at(x, z));
        let (min, max) = t.xz_bounds().expect("the fixture is resident");
        let whole = b.evaluate(
            &height,
            &fields,
            Region::from_xz(min.x, min.y, max.x, max.y),
        );
        let per_tile = b.evaluate_resident(&t, DVec3::ZERO);
        assert!(
            whole.len() > 100,
            "the fixture placed only {} instances — too few to say anything",
            whole.len()
        );
        assert_eq!(
            places(&whole),
            places(&per_tile),
            "the per-tile walk placed {} instances where one region places {}",
            per_tile.len(),
            whole.len()
        );
    }

    /// **A TILE THAT ARRIVES SECOND GROWS WHAT A TILE THAT WAS ALWAYS THERE
    /// GROWS** — P21's first-sight law, armed.
    ///
    /// The three readings are: the ground paged **all at once**, the ground
    /// paged **one tile at a time in ascending order**, and the ground paged
    /// **in descending order**. All three must produce the same places, because
    /// the population is a function of what is resident and not of the order it
    /// arrived in.
    ///
    /// The mutation it is built to catch is dropping the neighbours from
    /// [`neighbourhood_stamps`]: a tile evaluated alone rejects candidates
    /// within the feather of its own edge (off-terrain reads as *unlike*), and a
    /// tile-only key would memoize that answer for ever. The `edge` assertion
    /// below measures that the effect is real rather than assuming it.
    #[test]
    fn an_arrival_order_cannot_change_what_grows() {
        let full = paged_terrain();
        let b = cover_binding();
        let all: Vec<(i32, i32)> = (0..3).flat_map(|z| (0..3).map(move |x| (x, z))).collect();

        // The effect the key exists for, measured: one tile alone grows strictly
        // less than the same tile inside a paged neighbourhood.
        let lone = b.evaluate_resident(&paged_subset(&full, &[(1, 1)]), DVec3::ZERO);
        let mut whole_cache = BiomeScatterCache::default();
        b.refresh_resident(&full, DVec3::ZERO, &mut whole_cache);
        let middle = whole_cache
            .population()
            .iter()
            .filter(|i| (8.0..16.0).contains(&i.pos.x) && (8.0..16.0).contains(&i.pos.z))
            .count();
        assert!(
            middle > lone.len(),
            "a lone tile grew {} and the same tile inside its neighbours grew \
             {middle} — the feather does not reach across a tile edge in this \
             fixture, so the arms below cannot fail",
            lone.len()
        );

        let at_once = {
            let mut c = BiomeScatterCache::default();
            b.refresh_resident(&full, DVec3::ZERO, &mut c);
            places(c.population())
        };
        for (label, order) in [
            ("ascending", all.clone()),
            ("descending", all.iter().rev().copied().collect::<Vec<_>>()),
        ] {
            let mut c = BiomeScatterCache::default();
            let mut data = TerrainData::new(full.tile_resolution(), full.meters_per_sample());
            for coord in order {
                page_in(&mut data, &full, coord);
                b.refresh_resident(&data, DVec3::ZERO, &mut c);
            }
            assert_eq!(
                at_once,
                places(c.population()),
                "{label} arrival grew a different forest from the same ground \
                 paged all at once"
            );
        }
    }

    /// **THE GROUND STREAMS OUT AS WELL AS IN.** A tile that pages away takes
    /// exactly its own instances with it, and paging it back restores them.
    #[test]
    fn a_tile_that_pages_out_takes_exactly_its_own_instances() {
        let full = paged_terrain();
        let b = cover_binding();
        let all: Vec<(i32, i32)> = (0..3).flat_map(|z| (0..3).map(move |x| (x, z))).collect();

        // One `TerrainData`, paged the way a streamer pages it.
        let mut data = TerrainData::new(full.tile_resolution(), full.meters_per_sample());
        for &coord in &all {
            page_in(&mut data, &full, coord);
        }
        let mut c = BiomeScatterCache::default();
        assert!(b.refresh_resident(&data, DVec3::ZERO, &mut c));
        let before = places(c.population());
        assert_eq!(c.resident_tiles(), 9);

        assert!(
            data.evict_tile(TileKey::lod0((2, 2))),
            "the tile was resident"
        );
        assert!(
            b.refresh_resident(&data, DVec3::ZERO, &mut c),
            "paging a tile out must change the population"
        );
        assert_eq!(c.resident_tiles(), 8, "the memo is bounded by residency");
        let after = places(c.population());
        assert!(
            after.len() < before.len(),
            "{} instances before, {} after — nothing left with the tile",
            before.len(),
            after.len()
        );
        // Everything that survived is where it was: paging out moved nothing.
        let gone = paged_subset(&full, &[(2, 2)]);
        assert!(after.iter().all(|p| before.contains(p)));
        assert!(
            !b.evaluate_resident(&gone, DVec3::ZERO).is_empty(),
            "the tile that left grew nothing, so its departure proves nothing"
        );

        // …and paging it back restores exactly the reading it had — under a
        // NEW stamp, so this is the re-evaluation agreeing rather than the memo
        // having kept the answer.
        page_in(&mut data, &full, (2, 2));
        assert!(b.refresh_resident(&data, DVec3::ZERO, &mut c));
        assert_eq!(before, places(c.population()));
    }

    /// **THE MEMO ANSWERS, AND THE COUNTER IS WHAT SAYS SO.** A second refresh
    /// over unchanged ground evaluates no tile, reports "nothing changed", and
    /// still holds the same population.
    #[test]
    fn an_unchanged_resident_set_evaluates_nothing_and_answers_the_same() {
        let full = paged_terrain();
        let b = cover_binding();
        let mut c = BiomeScatterCache::default();
        assert!(b.refresh_resident(&full, DVec3::ZERO, &mut c));
        assert_eq!(c.tiles_evaluated(), 9, "the first pass walks every tile");
        let first = places(c.population());

        assert!(
            !b.refresh_resident(&full, DVec3::ZERO, &mut c),
            "an unchanged resident set must report no change"
        );
        assert_eq!(
            c.tiles_evaluated(),
            9,
            "the second pass re-evaluated a tile whose stamps had not moved"
        );
        assert_eq!(first, places(c.population()));
    }

    /// The reach is a **bound**, not a guess, and the neighbourhood is sized
    /// from it: at the grids this engine ships, one ring.
    #[test]
    fn the_reach_bounds_the_neighbourhood_it_keys_on() {
        assert_eq!(scatter_reach_m(1.0), 64.0 + FN_HEIGHT_NORMAL_EPS);
        assert_eq!(scatter_reach_m(0.0), FN_HEIGHT_NORMAL_EPS, "no lattice");
        // A shipped island tile: 257 samples at 1 m = 256 m across.
        assert_eq!(neighbour_rings(&TerrainData::new(257, 1.0)), 1);
        assert_eq!(neighbour_rings(&TerrainData::new(129, 2.0)), 1);
        // A test-sized tile is smaller than the reach and says so.
        assert_eq!(neighbour_rings(&TerrainData::new(9, 1.0)), 9);
        // …and the key really is (2r+1)² stamps, absence included.
        let t = paged_subset(&paged_terrain(), &[(0, 0)]);
        assert_eq!(neighbourhood_stamps(&t, (0, 0), 1).len(), 9);
        assert_eq!(
            neighbourhood_stamps(&t, (0, 0), 1)
                .iter()
                .filter(|&&s| s == 0)
                .count(),
            8,
            "a non-resident neighbour must contribute a 0, not nothing"
        );
    }

    /// Pool-size invariance **through the resident walk**, which fans out over
    /// tiles on top of the fan-out over cells the older arm covers.
    #[test]
    fn the_resident_population_is_invariant_under_pool_size() {
        use inf_core::JobPool;
        let full = paged_terrain();
        let b = cover_binding();
        let runs: Vec<Vec<(u64, u64, u64)>> = [1usize, 2, 4, 8]
            .into_iter()
            .map(|n| {
                let mut c = BiomeScatterCache::default();
                b.refresh_resident_in(&JobPool::new(n), &full, DVec3::ZERO, &mut c);
                places(c.population())
            })
            .collect();
        assert!(
            runs[0].len() > 100,
            "the fixture must place enough to matter"
        );
        for (n, run) in [1usize, 2, 4, 8].into_iter().zip(&runs).skip(1) {
            assert_eq!(run, &runs[0], "the population differs on a {n}-worker pool");
        }
    }

    /// An empty binding, and a terrain with no resident tile, both answer with
    /// an empty population rather than with the last one they held.
    #[test]
    fn nothing_resident_and_nothing_bound_both_grow_nothing() {
        let full = paged_terrain();
        let b = cover_binding();
        let mut c = BiomeScatterCache::default();
        assert!(b.refresh_resident(&full, DVec3::ZERO, &mut c));
        assert!(!c.population().is_empty());

        let empty = TerrainData::new(9, 1.0);
        assert!(b.refresh_resident(&empty, DVec3::ZERO, &mut c));
        assert!(c.population().is_empty(), "no ground grows nothing");
        assert_eq!(c.resident_tiles(), 0);

        let unbound = BiomeBinding::new(Vec::new(), DEFAULT_BIOME_FEATHER);
        let mut c2 = BiomeScatterCache::default();
        assert!(!unbound.refresh_resident(&full, DVec3::ZERO, &mut c2));
        assert!(c2.population().is_empty());
    }

    /// The *present* mechanism, stated as its own test: biome masks are
    /// **disjoint**, so no point is inside two biomes at once — which is why the
    /// seed fold is a forward guard rather than today's fix.
    #[test]
    fn biome_masks_are_disjoint_by_construction() {
        use crate::sampler::{BiomeMask, DensityField};
        let t = split_terrain();
        let fields = OffsetTerrain::new(&t, DVec3::ZERO);
        let one = BiomeMask {
            fields: &fields,
            id: 1,
            feather: 8.0,
        };
        let two = BiomeMask {
            fields: &fields,
            id: 2,
            feather: 8.0,
        };
        // Sweep straight across the border on a fine step: the product is zero
        // everywhere, i.e. the two masks are never simultaneously positive.
        for step in 0..=320 {
            let x = step as f64 * 0.1;
            let (a, b) = (one.density(x, 8.0), two.density(x, 8.0));
            assert_eq!(
                a * b,
                0.0,
                "masks overlap at x={x}: {a} and {b} — the feather has started \
                 blending ACROSS the border, and `biome_seed` is now load-bearing"
            );
        }
    }

    /// Masking is multiplicative, so a bound rule's output is a **subset** of the
    /// unbound rule's — a mask can reject a candidate but never move one.
    #[test]
    fn binding_only_ever_removes_instances() {
        let t = split_terrain();
        let fields = OffsetTerrain::new(&t, DVec3::ZERO);
        let base = doc(77, 1);
        // The reference is the SAME rule at the SAME seed the bound dispatch will
        // run under — `biome_seed(seed, 1)`, applied here by hand — just without
        // the mask wrapped around its sampler. So the only difference between the
        // two lists is the mask, which is what makes "subset, never moved"
        // meaningful rather than a comparison of two unrelated scatters.
        let mut reference = base.clone();
        for layer in &mut reference.layers {
            for rule in &mut layer.rules {
                rule.scatter.seed = biome_seed(rule.scatter.seed, 1);
            }
        }
        let all = crate::rules::evaluate_with(&reference, &flat(), &fields, region());
        let bound = BiomeBinding::new(
            vec![BiomeGraph {
                id: 1,
                document: base,
            }],
            0.0,
        )
        .evaluate(&flat(), &fields, region());
        assert!(!bound.is_empty() && bound.len() < all.len());
        for i in &bound {
            assert!(
                all.iter()
                    .any(|a| a.pos == i.pos && a.rotation == i.rotation),
                "the mask MOVED an instance to {:?}",
                i.pos
            );
        }
    }

    /// `from_set` skips biomes with no graph and biomes whose graph does not
    /// resolve, and never binds the reserved id.
    #[test]
    fn from_set_binds_only_resolvable_graphs() {
        let mut set = BiomeSet::new("s");
        let bound_guid = Uuid::from_u128(0xB0);
        let missing = Uuid::from_u128(0xB1);
        set.biomes.push(BiomeDef {
            pcg_graph: Some(bound_guid),
            ..BiomeDef::new(3, "three")
        });
        set.biomes.push(BiomeDef {
            pcg_graph: Some(missing),
            ..BiomeDef::new(1, "one")
        });
        set.biomes.push(BiomeDef::new(2, "two")); // no graph at all
        set.validate().unwrap();

        let b = BiomeBinding::from_set(&set, DEFAULT_BIOME_FEATHER, |g| {
            (g == bound_guid).then(|| doc(1, 1))
        });
        assert_eq!(
            b.graphs().iter().map(|g| g.id).collect::<Vec<_>>(),
            vec![3],
            "only the resolvable graph binds"
        );
        assert!(!b.is_empty());
        assert_eq!(b.feather, DEFAULT_BIOME_FEATHER);

        // The reserved id can never own a population, even if handed one.
        let zeroed = BiomeBinding::new(
            vec![BiomeGraph {
                id: UNASSIGNED_BIOME,
                document: doc(1, 1),
            }],
            0.0,
        );
        assert!(zeroed.is_empty());
    }

    /// `bind_document` states the rewrite in one place: mask first, id-folded
    /// seed, every rule of every layer.
    #[test]
    fn bind_document_masks_every_rule_and_salts_every_seed() {
        let mut d = doc(4, 1);
        d.layers.push(crate::rules::PcgLayer {
            name: "second".into(),
            enabled: false,
            rules: d.layers[0].rules.clone(),
        });
        let bound = bind_document(&d, 5, 2.5);
        assert_eq!(bound.layers.len(), 2);
        assert!(!bound.layers[1].enabled, "layer flags survive the rewrite");
        for layer in &bound.layers {
            for rule in &layer.rules {
                assert_eq!(rule.scatter.seed, biome_seed(4, 5));
                match &rule.sampler {
                    SamplerDef::Multiply(a, b) => {
                        assert_eq!(
                            **a,
                            SamplerDef::Biome {
                                id: 5,
                                feather: 2.5
                            }
                        );
                        assert_eq!(**b, SamplerDef::Constant(1.0), "the authored tree survives");
                    }
                    other => panic!("expected Multiply(mask, authored), got {other:?}"),
                }
            }
        }
    }

    /// **Pool-size invariance, through the real path.**
    ///
    /// Not a rule lifted out of a document by hand — the whole
    /// [`BiomeBinding::evaluate_in`] pass over **two biomes × two layers × three
    /// rules**, including the per-biome seed fold, the mask wrapping, the
    /// weighted kind picks and the concatenation order. Those are exactly the
    /// steps a hand-extracted single-rule check cannot see, and they are where a
    /// future "optimization" that collects per-biome results out of order would
    /// land.
    #[test]
    fn the_population_is_invariant_under_pool_size() {
        use inf_core::JobPool;

        let t = split_terrain();
        let fields = OffsetTerrain::new(&t, DVec3::ZERO);

        // Two layers, three rules between them, distinct kinds — so layer order,
        // rule order and the kind pick all participate.
        let multi = |seed: u64, mesh: u128| PcgDocument {
            layers: vec![
                PcgLayer {
                    name: "ground".into(),
                    enabled: true,
                    rules: vec![
                        rule_with(seed, mesh, 0.25, 8.0),
                        rule_with(seed + 1, mesh + 1, 0.1, 16.0),
                    ],
                },
                PcgLayer {
                    name: "canopy".into(),
                    enabled: true,
                    rules: vec![rule_with(seed + 2, mesh + 2, 0.05, 32.0)],
                },
            ],
        };
        let b = BiomeBinding::new(
            vec![
                BiomeGraph {
                    id: 1,
                    document: multi(11, 10),
                },
                BiomeGraph {
                    id: 2,
                    document: multi(22, 20),
                },
            ],
            4.0,
        );

        let runs: Vec<Vec<PcgInstance>> = [1usize, 2, 4, 8]
            .into_iter()
            .map(|n| b.evaluate_in(&JobPool::new(n), &flat(), &fields, region()))
            .collect();
        assert!(
            runs[0].len() > 100,
            "the fixture must place enough to matter"
        );
        for (n, run) in [1usize, 2, 4, 8].into_iter().zip(&runs).skip(1) {
            assert_eq!(run, &runs[0], "the population differs on a {n}-worker pool");
        }
        // Both biomes and every kind really are represented, so the invariance
        // above is over the whole merged list rather than a lucky subset.
        assert!(runs[0].iter().any(|i| i.pos.x < BORDER_X));
        assert!(runs[0].iter().any(|i| i.pos.x >= BORDER_X));
        assert!(
            runs[0].iter().all(|i| i.kind_index == 0),
            "one kind per rule"
        );
        // …and the default entry point agrees with the explicit-pool one.
        assert_eq!(b.evaluate(&flat(), &fields, region()), runs[0]);
    }
}
