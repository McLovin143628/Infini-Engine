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
//!    seed.** [`TerrainFields::biome_id`](crate::fields::TerrainFields::biome_id)
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

use inf_terrain::{BiomeSet, UNASSIGNED_BIOME};
use uuid::Uuid;

use crate::fields::TerrainFields;
use crate::hash::Hash64;
use crate::height::HeightProvider;
use crate::rules::{evaluate_with_in, PcgDocument, SamplerDef};
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
    /// `fields` supplies the painted ids (an [`OffsetTerrain`](crate::fields::OffsetTerrain)
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
