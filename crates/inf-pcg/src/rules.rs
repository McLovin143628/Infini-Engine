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

use crate::hash::Hash64;
use crate::height::HeightProvider;
use crate::noise::ValueNoise;
use crate::sampler::{
    AltitudeFilter, Constant, DensityField, Invert, MaskImage, Max, Min, Multiply, Noise,
    SlopeFilter,
};
use crate::scatter::{scatter_region, PcgInstance, Region, ScatterParams};

/// A serde tree mirroring the [`sampler`](crate::sampler) combinators. Built into
/// a boxed [`DensityField`] against a terrain provider by [`SamplerDef::build`].
///
/// Uses the *default* (externally-tagged) enum representation — bincode cannot
/// encode internally-tagged enums (the engine-wide constraint noted in
/// `inf-asset`).
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
}

impl SamplerDef {
    /// Build this definition into a boxed [`DensityField`] that borrows `height`
    /// for its slope/altitude filters. The returned field lives as long as the
    /// borrow.
    pub fn build<'a>(&self, height: &'a dyn HeightProvider) -> Box<dyn DensityField + 'a> {
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
            SamplerDef::Multiply(a, b) => Box::new(Multiply(a.build(height), b.build(height))),
            SamplerDef::Max(a, b) => Box::new(Max(a.build(height), b.build(height))),
            SamplerDef::Min(a, b) => Box::new(Min(a.build(height), b.build(height))),
            SamplerDef::Invert(a) => Box::new(Invert(a.build(height))),
        }
    }
}

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
    let mut out = Vec::new();
    for layer in &doc.layers {
        if !layer.enabled {
            continue;
        }
        for rule in &layer.rules {
            let field = rule.sampler.build(height);
            let mut insts = scatter_region(&rule.scatter, field.as_ref(), height, region);
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
