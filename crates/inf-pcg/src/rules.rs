//! The serializable PCG rule model — the runtime shape of a `.inf_pcg` document.
//!
//! ## Why a typed struct pipeline, not an `inf-graph` node kit (yet)
//!
//! Blueprints and materials build their editors on the shared `inf-graph`
//! substrate, and PCG will too — but the **editor canvas kit** (the visual
//! node graph) lands with the P10.5b editor batch. The *runtime* model must be
//! stable and serialization-locked first, so v1 is a plain typed pipeline: a
//! [`SamplerDef`] tree (mirroring the [`sampler`](crate::sampler) combinators)
//! feeds a [`ScatterParams`] pass that emits weighted [`PcgKind`]s. When the
//! canvas kit arrives it will *lower* to exactly this model, the way the
//! blueprint lowerer targets `BlueprintFn` — keeping preview == runtime.
//!
//! Evaluation order is fully deterministic: layers in order, rules in order,
//! instances in scatter (cell) order, kinds resolved by a hashed weighted pick.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::fields::{NoFields, TerrainFields};
use crate::hash::Hash64;
use crate::height::HeightProvider;
use crate::noise::ValueNoise;
use crate::sampler::{
    AltitudeFilter, BiomeMask, Constant, DataMapMask, DensityField, Invert, MaskImage, Max, Min,
    Multiply, Noise, SlopeFilter,
};
use crate::scatter::{scatter_region_in, PcgInstance, Region, ScatterParams};

/// A serde tree mirroring the [`sampler`](crate::sampler) combinators. Built into
/// a boxed [`DensityField`] against a terrain provider by [`SamplerDef::build`].
///
/// Uses the *default* (externally-tagged) enum representation — bincode cannot
/// encode internally-tagged enums (the engine-wide constraint noted in
/// `inf-asset`).
///
/// # THE ORDERING LAW: new variants go at the **end**, always
///
/// bincode encodes an externally-tagged enum as its **declaration index**, so
/// inserting a variant in the middle silently renumbers everything after it and
/// every committed `.inf_pcg` decodes as the wrong sampler. P19.3 hit exactly
/// this — adding `DataMap`/`Biome` beside the other sources shifted
/// `Multiply/Max/Min/Invert` by two and moved a sample's bytes — and the fix is
/// the rule, not the patch: **append**, never insert, and let the reading order
/// be documentation's problem rather than the wire's. The P19.3 additions are
/// marked below.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplerDef {
    /// A uniform density (clamped to `[0, 1]`).
    Constant(f64),
    /// fBm value noise.
    Noise(ValueNoise),
    /// Terrain-slope band in degrees, feathered.
    Slope {
        min_deg: f64,
        max_deg: f64,
        feather_deg: f64,
    },
    /// Terrain-altitude band, feathered.
    Altitude { min: f64, max: f64, feather: f64 },
    /// A grayscale bitmap mask over a world rect (`[min_x, min_z, max_x, max_z]`).
    Mask {
        rect: [f64; 4],
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
    /// Product (intersection) of two fields.
    Multiply(Box<SamplerDef>, Box<SamplerDef>),
    /// Maximum (union) of two fields.
    Max(Box<SamplerDef>, Box<SamplerDef>),
    /// Minimum of two fields.
    Min(Box<SamplerDef>, Box<SamplerDef>),
    /// `1 − a`.
    Invert(Box<SamplerDef>),
    // ── APPENDED IN P19.3 — see the ordering law on the enum ────────────────
    /// A **normalized** read of one P19.1 erosion data map. The stored
    /// accumulators are raw (m³ / metres), so the window lives here — see
    /// [`DataMapMask`]. Reads the [`TerrainFields`] seam; scores `0` without one.
    DataMap {
        kind: inf_terrain::DataMapKind,
        min: f64,
        max: f64,
    },
    /// A **biome-id membership mask**: `1` inside `id`, `0` outside, blended over
    /// `feather` metres of border — see [`BiomeMask`]. Reads the
    /// [`TerrainFields`] seam; scores `0` without one.
    Biome { id: u8, feather: f64 },
}

impl SamplerDef {
    /// Build this definition into a boxed [`DensityField`] that borrows `height`
    /// for its slope/altitude filters. The returned field lives as long as the
    /// borrow.
    ///
    /// The terrain-**layer** masks (`DataMap`, `Biome`) have no source here and
    /// score `0` — use [`build_with`](Self::build_with) to supply one.
    pub fn build<'a>(&self, height: &'a dyn HeightProvider) -> Box<dyn DensityField + 'a> {
        self.build_with(height, &NO_FIELDS)
    }

    /// Build this definition against **both** terrain seams: `height` for the
    /// slope/altitude filters and `fields` for the P19.3 data-map and biome masks.
    /// The returned field borrows both.
    pub fn build_with<'a>(
        &self,
        height: &'a dyn HeightProvider,
        fields: &'a dyn TerrainFields,
    ) -> Box<dyn DensityField + 'a> {
        match self {
            SamplerDef::Constant(v) => Box::new(Constant(*v)),
            SamplerDef::Noise(n) => Box::new(Noise(*n)),
            SamplerDef::Slope {
                min_deg,
                max_deg,
                feather_deg,
            } => Box::new(SlopeFilter {
                height,
                min_deg: *min_deg,
                max_deg: *max_deg,
                feather_deg: *feather_deg,
            }),
            SamplerDef::Altitude { min, max, feather } => Box::new(AltitudeFilter {
                height,
                min: *min,
                max: *max,
                feather: *feather,
            }),
            SamplerDef::Mask {
                rect,
                width,
                height: h,
                data,
            } => Box::new(MaskImage {
                rect: *rect,
                width: *width,
                height: *h,
                data: data.clone(),
            }),
            SamplerDef::DataMap { kind, min, max } => Box::new(DataMapMask {
                fields,
                kind: *kind,
                min: *min,
                max: *max,
            }),
            SamplerDef::Biome { id, feather } => Box::new(BiomeMask {
                fields,
                id: *id,
                feather: *feather,
            }),
            SamplerDef::Multiply(a, b) => Box::new(Multiply(
                a.build_with(height, fields),
                b.build_with(height, fields),
            )),
            SamplerDef::Max(a, b) => Box::new(Max(
                a.build_with(height, fields),
                b.build_with(height, fields),
            )),
            SamplerDef::Min(a, b) => Box::new(Min(
                a.build_with(height, fields),
                b.build_with(height, fields),
            )),
            SamplerDef::Invert(a) => Box::new(Invert(a.build_with(height, fields))),
        }
    }
}

/// The `'static` empty layer source [`SamplerDef::build`] falls back to, so the
/// no-terrain path allocates nothing and the two build entry points share one
/// body.
static NO_FIELDS: NoFields = NoFields;

/// One spawnable variant of a rule: a mesh asset (by GUID) and a relative weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcgKind {
    /// The mesh asset to instance, or `None` for a bare transform (debug/marker).
    #[serde(default)]
    pub mesh: Option<Uuid>,
    /// Relative selection weight (non-positive weights are ignored).
    pub weight: f64,
}

impl PcgKind {
    /// A kind that instances `mesh` with unit weight.
    pub fn mesh(mesh: Uuid) -> Self {
        Self {
            mesh: Some(mesh),
            weight: 1.0,
        }
    }
}

/// A single scatter rule: sample a density, scatter it, pick among `kinds`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcgRule {
    pub name: String,
    pub sampler: SamplerDef,
    pub scatter: ScatterParams,
    /// Weighted spawn palette; `kind_index` on each instance indexes this list.
    pub kinds: Vec<PcgKind>,
}

/// A named, toggleable group of rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcgLayer {
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub rules: Vec<PcgRule>,
}

fn default_true() -> bool {
    true
}

/// A whole `.inf_pcg` document: ordered layers of rules.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PcgDocument {
    pub layers: Vec<PcgLayer>,
}

impl PcgDocument {
    /// A document with a single enabled layer holding `rules`.
    pub fn single_layer(name: impl Into<String>, rules: Vec<PcgRule>) -> Self {
        Self {
            layers: vec![PcgLayer {
                name: name.into(),
                enabled: true,
                rules,
            }],
        }
    }
}

/// Evaluate a whole document within `region`: every enabled layer's rules run in
/// order, their instances concatenated, each instance's `kind_index` resolved by
/// a hashed weighted pick over the rule's `kinds`.
pub fn evaluate(
    doc: &PcgDocument,
    height: &dyn HeightProvider,
    region: Region,
) -> Vec<PcgInstance> {
    evaluate_with(doc, height, &NoFields, region)
}

/// [`evaluate`] with a terrain **layer** source (P19.3) — the entry point a
/// document using `mask.flow` / `mask.biome` / … needs. Identical in every other
/// respect; `evaluate` is this with [`NoFields`].
pub fn evaluate_with(
    doc: &PcgDocument,
    height: &dyn HeightProvider,
    fields: &dyn TerrainFields,
    region: Region,
) -> Vec<PcgInstance> {
    evaluate_with_in(inf_core::global(), doc, height, fields, region)
}

/// [`evaluate_with`] on a **caller-supplied pool** — the seam the determinism
/// guard drives, mirroring
/// [`scatter_region_in`](crate::scatter::scatter_region_in) one layer up.
///
/// The result is independent of the pool's thread count, so a test can prove the
/// property through the *real* evaluation path (documents, layers, rules, kind
/// picks and all) rather than through a hand-extracted rule.
pub fn evaluate_with_in(
    pool: &inf_core::JobPool,
    doc: &PcgDocument,
    height: &dyn HeightProvider,
    fields: &dyn TerrainFields,
    region: Region,
) -> Vec<PcgInstance> {
    let mut out = Vec::new();
    for layer in &doc.layers {
        if !layer.enabled {
            continue;
        }
        for rule in &layer.rules {
            let field = rule.sampler.build_with(height, fields);
            let mut insts = scatter_region_in(pool, &rule.scatter, field.as_ref(), height, region);
            for inst in &mut insts {
                inst.kind_index = weighted_pick(&rule.kinds, rule.scatter.seed, inst.pos);
            }
            out.append(&mut insts);
        }
    }
    out
}

/// Deterministically pick a kind index for an instance from its position. Returns
/// 0 when there are no positively-weighted kinds.
fn weighted_pick(kinds: &[PcgKind], seed: u64, pos: glam::DVec3) -> u32 {
    let total: f64 = kinds.iter().map(|k| k.weight.max(0.0)).sum();
    if total <= 0.0 {
        return 0;
    }
    let u = Hash64::new(seed)
        .mix_u64(0x4b1d)
        .mix_f64(pos.x)
        .mix_f64(pos.z)
        .unit()
        * total;
    let mut acc = 0.0;
    for (idx, k) in kinds.iter().enumerate() {
        acc += k.weight.max(0.0);
        if u < acc {
            return idx as u32;
        }
    }
    (kinds.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::height::FnHeight;
    use crate::scatter::RotationMode;

    fn flat() -> FnHeight<impl Fn(f64, f64) -> Option<f64> + Send + Sync> {
        FnHeight::new(|_, _| Some(0.0))
    }

    fn demo_scatter(seed: u64) -> ScatterParams {
        ScatterParams {
            seed,
            cell_size: 16.0,
            base_density: 0.5,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (0.9, 1.1),
            rotation: RotationMode::RandomYaw,
            altitude_offset: 0.0,
        }
    }

    fn demo_doc() -> PcgDocument {
        // Two kinds, forests on gentle slopes intersected with noise.
        let sampler = SamplerDef::Multiply(
            Box::new(SamplerDef::Noise(ValueNoise {
                seed: 7,
                frequency: 0.02,
                octaves: 3,
                lacunarity: 2.0,
                gain: 0.5,
            })),
            Box::new(SamplerDef::Slope {
                min_deg: 0.0,
                max_deg: 30.0,
                feather_deg: 5.0,
            }),
        );
        let rule = PcgRule {
            name: "trees".into(),
            sampler,
            scatter: demo_scatter(2024),
            kinds: vec![
                PcgKind {
                    mesh: Some(Uuid::from_u128(1)),
                    weight: 3.0,
                },
                PcgKind {
                    mesh: Some(Uuid::from_u128(2)),
                    weight: 1.0,
                },
            ],
        };
        PcgDocument::single_layer("vegetation", vec![rule])
    }

    /// **The ordering tripwire.** bincode writes an externally-tagged enum as its
    /// declaration index, so this pins every `SamplerDef` variant to the index a
    /// committed `.inf_pcg` was written with. A variant inserted anywhere but the
    /// end fails here instead of silently decoding a `Multiply` as a `Min` in
    /// somebody's project.
    #[test]
    fn sampler_variant_discriminants_are_frozen() {
        let cfg = bincode::config::standard();
        let tag = |d: &SamplerDef| bincode::serde::encode_to_vec(d, cfg).unwrap()[0];
        let boxed = || Box::new(SamplerDef::Constant(0.0));
        assert_eq!(tag(&SamplerDef::Constant(0.0)), 0);
        assert_eq!(tag(&SamplerDef::Noise(ValueNoise::default())), 1);
        assert_eq!(
            tag(&SamplerDef::Slope {
                min_deg: 0.0,
                max_deg: 0.0,
                feather_deg: 0.0
            }),
            2
        );
        assert_eq!(
            tag(&SamplerDef::Altitude {
                min: 0.0,
                max: 0.0,
                feather: 0.0
            }),
            3
        );
        assert_eq!(
            tag(&SamplerDef::Mask {
                rect: [0.0; 4],
                width: 0,
                height: 0,
                data: Vec::new()
            }),
            4
        );
        assert_eq!(tag(&SamplerDef::Multiply(boxed(), boxed())), 5);
        assert_eq!(tag(&SamplerDef::Max(boxed(), boxed())), 6);
        assert_eq!(tag(&SamplerDef::Min(boxed(), boxed())), 7);
        assert_eq!(tag(&SamplerDef::Invert(boxed())), 8);
        // P19.3 appended these two, in this order.
        assert_eq!(
            tag(&SamplerDef::DataMap {
                kind: inf_terrain::DataMapKind::Flow,
                min: 0.0,
                max: 1.0
            }),
            9
        );
        assert_eq!(
            tag(&SamplerDef::Biome {
                id: 1,
                feather: 0.0
            }),
            10
        );

        // **The NESTED enum is on this wire too, and needs the same law.**
        // `DataMapKind` lives in `inf-terrain`, where nothing else about it is
        // persisted — so without this, appending a fourth *thermal* channel in
        // the middle (the P19.1 remainder contemplates exactly that) would
        // silently turn every committed `mask.wear` into a `mask.deposition`.
        // The second byte of an encoded `DataMap` is that kind's own index.
        let kind_tag = |kind| {
            bincode::serde::encode_to_vec(
                &SamplerDef::DataMap {
                    kind,
                    min: 0.0,
                    max: 0.0,
                },
                cfg,
            )
            .unwrap()[1]
        };
        assert_eq!(kind_tag(inf_terrain::DataMapKind::Flow), 0);
        assert_eq!(kind_tag(inf_terrain::DataMapKind::Deposition), 1);
        assert_eq!(kind_tag(inf_terrain::DataMapKind::Wear), 2);
        // Spelled out once as raw bytes, so the encoding itself is visible and
        // not merely re-derived by the helper above: `Wear` at zeroed bounds.
        assert_eq!(
            &bincode::serde::encode_to_vec(
                &SamplerDef::DataMap {
                    kind: inf_terrain::DataMapKind::Wear,
                    min: 0.0,
                    max: 0.0,
                },
                cfg,
            )
            .unwrap()[..3],
            &[9, 2, 0],
            "SamplerDef::DataMap(Wear) must encode as [variant 9, kind 2, …]"
        );
        // …and the kind's storage-channel order is a SEPARATE statement, so the
        // two can never drift into agreeing only by accident.
        for (n, kind) in inf_terrain::DataMapKind::ALL.iter().enumerate() {
            assert_eq!(kind.channel(), n, "{kind:?} moved channel");
        }
    }

    /// The two P19.3 variants round-trip through **both** codecs — the permanent
    /// serde discipline, applied to the new wire content.
    #[test]
    fn the_p19_3_variants_round_trip_in_both_codecs() {
        let cfg = bincode::config::standard();
        for def in [
            SamplerDef::DataMap {
                kind: inf_terrain::DataMapKind::Deposition,
                min: -1.5,
                max: 42.0,
            },
            SamplerDef::Biome {
                id: 200,
                feather: 12.5,
            },
            // …and nested, where a shifted discriminant would corrupt silently.
            SamplerDef::Multiply(
                Box::new(SamplerDef::Biome {
                    id: 3,
                    feather: 4.0,
                }),
                Box::new(SamplerDef::DataMap {
                    kind: inf_terrain::DataMapKind::Wear,
                    min: 0.0,
                    max: 2.0,
                }),
            ),
        ] {
            let json = serde_json::to_string(&def).unwrap();
            assert_eq!(serde_json::from_str::<SamplerDef>(&json).unwrap(), def);
            let bytes = bincode::serde::encode_to_vec(&def, cfg).unwrap();
            let (back, _): (SamplerDef, _) =
                bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
            assert_eq!(back, def);
            assert_eq!(bytes, bincode::serde::encode_to_vec(&back, cfg).unwrap());
        }
    }

    #[test]
    fn serde_round_trips() {
        let doc = demo_doc();
        let json = serde_json::to_string(&doc).unwrap();
        let back: PcgDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(doc, back);
    }

    #[test]
    fn evaluate_is_deterministic() {
        let doc = demo_doc();
        let region = Region::from_xz(0.0, 0.0, 200.0, 200.0);
        let a = evaluate(&doc, &flat(), region);
        let b = evaluate(&doc, &flat(), region);
        assert!(!a.is_empty());
        assert_eq!(a, b);
    }

    #[test]
    fn disabled_layer_produces_nothing() {
        let mut doc = demo_doc();
        doc.layers[0].enabled = false;
        let out = evaluate(&doc, &flat(), Region::from_xz(0.0, 0.0, 200.0, 200.0));
        assert!(out.is_empty());
    }

    #[test]
    fn kind_index_respects_weights() {
        let doc = demo_doc();
        let out = evaluate(&doc, &flat(), Region::from_xz(0.0, 0.0, 400.0, 400.0));
        assert!(out.len() > 100);
        // All indices valid.
        assert!(out.iter().all(|i| i.kind_index < 2));
        // The 3:1 weighting should make kind 0 clearly dominant.
        let k0 = out.iter().filter(|i| i.kind_index == 0).count();
        let k1 = out.iter().filter(|i| i.kind_index == 1).count();
        assert!(k0 > k1, "k0={k0} k1={k1}");
    }

    #[test]
    fn empty_kinds_default_to_zero() {
        assert_eq!(weighted_pick(&[], 1, glam::DVec3::ZERO), 0);
        let zero_weight = vec![PcgKind {
            mesh: None,
            weight: 0.0,
        }];
        assert_eq!(weighted_pick(&zero_weight, 1, glam::DVec3::ZERO), 0);
    }
}
