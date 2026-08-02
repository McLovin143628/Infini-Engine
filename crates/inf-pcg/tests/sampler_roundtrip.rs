//! The **permanent round-trip discipline** for the PCG runtime model, applied to
//! P19.3's grown [`SamplerDef`] tree.
//!
//! A `.inf_pcg` is a bincode payload and `SamplerDef` is a recursive
//! externally-tagged enum, which makes it exactly the shape where a hand-written
//! corpus misses things: the failure mode is *a variant nobody wrote a case for*
//! or *a discriminant that shifted*, and both hide until somebody's project
//! decodes a `Multiply` as a `Min`. P19.3 hit the second one for real (see the
//! ordering law on `SamplerDef`), so the property is stated here rather than
//! implied by the unit tests.
//!
//! Three properties, on arbitrary trees:
//!
//! 1. **bincode round-trips** — decode(encode(x)) == x.
//! 2. **the encoding is canonical** — encode(decode(encode(x))) is byte-identical,
//!    which is what makes a `.inf_pcg` content-hash stable and a re-save a no-op.
//! 3. **JSON round-trips** — the same tree survives the human-readable codec, the
//!    one an `.inf_pcg`'s embedded graph and every diagnostic path use.

use proptest::prelude::*;

use inf_pcg::rules::{PcgDocument, PcgKind, PcgLayer, PcgRule, SamplerDef};
use inf_pcg::scatter::{RotationMode, ScatterParams};
use inf_pcg::ValueNoise;

/// Finite, well-behaved doubles: `NaN` is not `PartialEq` with itself, so a tree
/// containing one could never satisfy a round-trip property in *any* codec — the
/// property would be testing `f64`, not the encoding.
fn scalar() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(1.0),
        Just(-1.0),
        (-1.0e6f64..1.0e6f64),
        // Sub-normal-ish and huge magnitudes, where a lossy codec would show.
        Just(f64::MIN_POSITIVE),
        Just(1.0e300),
    ]
}

fn data_map_kind() -> impl Strategy<Value = inf_terrain::DataMapKind> {
    prop_oneof![
        Just(inf_terrain::DataMapKind::Flow),
        Just(inf_terrain::DataMapKind::Deposition),
        Just(inf_terrain::DataMapKind::Wear),
    ]
}

/// Every leaf variant, including the two P19.3 appended.
fn leaf() -> impl Strategy<Value = SamplerDef> {
    prop_oneof![
        scalar().prop_map(SamplerDef::Constant),
        (any::<u64>(), scalar(), 1u32..8, scalar(), scalar()).prop_map(
            |(seed, frequency, octaves, lacunarity, gain)| SamplerDef::Noise(ValueNoise {
                seed,
                frequency,
                octaves,
                lacunarity,
                gain,
            })
        ),
        (scalar(), scalar(), scalar()).prop_map(|(min_deg, max_deg, feather_deg)| {
            SamplerDef::Slope {
                min_deg,
                max_deg,
                feather_deg,
            }
        }),
        (scalar(), scalar(), scalar()).prop_map(|(min, max, feather)| SamplerDef::Altitude {
            min,
            max,
            feather
        }),
        (
            [scalar(), scalar(), scalar(), scalar()],
            0u32..8,
            0u32..8,
            prop::collection::vec(any::<u8>(), 0..64),
        )
            .prop_map(|(rect, width, height, data)| SamplerDef::Mask {
                rect,
                width,
                height,
                data,
            }),
        (data_map_kind(), scalar(), scalar()).prop_map(|(kind, min, max)| SamplerDef::DataMap {
            kind,
            min,
            max
        }),
        (any::<u8>(), scalar()).prop_map(|(id, feather)| SamplerDef::Biome { id, feather }),
    ]
}

/// Arbitrary sampler trees, up to a bounded depth.
fn sampler() -> impl Strategy<Value = SamplerDef> {
    leaf().prop_recursive(4, 24, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| SamplerDef::Multiply(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| SamplerDef::Max(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| SamplerDef::Min(Box::new(a), Box::new(b))),
            inner.prop_map(|a| SamplerDef::Invert(Box::new(a))),
        ]
    })
}

fn scatter_params() -> impl Strategy<Value = ScatterParams> {
    (
        any::<u64>(),
        1.0f64..512.0,
        0.0f64..4.0,
        0.0f64..1.0,
        any::<bool>(),
        (0.1f64..10.0, 0.1f64..10.0),
        prop_oneof![
            Just(RotationMode::RandomYaw),
            Just(RotationMode::AlignNormal)
        ],
        scalar(),
    )
        .prop_map(
            |(
                seed,
                cell_size,
                base_density,
                jitter,
                align_to_normal,
                scale_range,
                rotation,
                altitude_offset,
            )| ScatterParams {
                seed,
                cell_size,
                base_density,
                jitter,
                align_to_normal,
                scale_range,
                rotation,
                altitude_offset,
            },
        )
}

/// Whole documents — layers × rules, the shape P19.3's lowering can now produce.
fn document() -> impl Strategy<Value = PcgDocument> {
    let kind = (prop::option::of(any::<u128>()), scalar()).prop_map(|(mesh, weight)| PcgKind {
        mesh: mesh.map(uuid::Uuid::from_u128),
        weight,
    });
    let rule = (
        ".{0,12}",
        sampler(),
        scatter_params(),
        prop::collection::vec(kind, 0..4),
    )
        .prop_map(|(name, sampler, scatter, kinds)| PcgRule {
            name,
            sampler,
            scatter,
            kinds,
        });
    let layer = (".{0,12}", any::<bool>(), prop::collection::vec(rule, 0..3)).prop_map(
        |(name, enabled, rules)| PcgLayer {
            name,
            enabled,
            rules,
        },
    );
    prop::collection::vec(layer, 0..3).prop_map(|layers| PcgDocument { layers })
}

fn bincode_round_trip<T>(value: &T) -> (T, Vec<u8>, Vec<u8>)
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let cfg = bincode::config::standard();
    let first = bincode::serde::encode_to_vec(value, cfg).expect("encode");
    let (back, _): (T, usize) = bincode::serde::decode_from_slice(&first, cfg).expect("decode");
    let second = bincode::serde::encode_to_vec(&back, cfg).expect("re-encode");
    (back, first, second)
}

/// **The parity hazard this file actually found.**
///
/// A `.inf_pcg` stores its authored graph as a **JSON string** (`graph_json` — the
/// `inf_graph` model's `skip_serializing_if` fields are not bincode-safe), and the
/// shipped/PIE player *re-lowers that graph* rather than trusting the stored
/// document mirror. So every authored `f64` param crosses JSON on the way to the
/// player, and `serde_json`'s **default** parser is a fast path that can land one
/// ULP off. A param that came back a bit light gives the player a different
/// `ScatterParams` from the editor's, which surfaces as *the shipped world's
/// foliage sits somewhere else* — and nothing else in the pipeline would notice.
///
/// The fix is the workspace's `serde_json = { features = ["float_roundtrip"] }`
/// pin; this is the test that fails if anyone drops it.
#[test]
fn an_authored_graph_re_lowers_bit_identically_through_the_payload() {
    use inf_graph::{apply_edits, GraphEdit, Link, NodeId, ParamMap, ParamValue};

    // Params with full 17-significant-digit mantissas — exactly what a slider
    // dragged to an arbitrary position produces, and what the fast parser moves.
    let awkward = [
        -114668.51350953568f64,
        0.30000000000000004,
        1.7976931348623157e300,
        f64::MIN_POSITIVE,
    ];

    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    let mut params = ParamMap::new();
    params.insert("frequency".into(), ParamValue::Float(awkward[1]));
    params.insert("lacunarity".into(), ParamValue::Float(awkward[0]));
    params.insert("gain".into(), ParamValue::Float(awkward[3]));
    let mut scatter_params = ParamMap::new();
    scatter_params.insert("cell_size".into(), ParamValue::Float(awkward[1] + 32.0));
    scatter_params.insert("base_density".into(), ParamValue::Float(awkward[1]));
    scatter_params.insert("altitude_offset".into(), ParamValue::Float(awkward[2]));
    apply_edits(
        &mut g,
        &reg,
        &[
            GraphEdit::AddNode {
                id: NodeId(1),
                type_id: "noise.fbm".into(),
                x: 0.0,
                y: 0.0,
                params,
            },
            GraphEdit::AddNode {
                id: NodeId(2),
                type_id: "scatter.scatter".into(),
                x: 0.0,
                y: 0.0,
                params: scatter_params,
            },
            GraphEdit::AddNode {
                id: NodeId(3),
                type_id: "output.pcg".into(),
                x: 0.0,
                y: 0.0,
                params: ParamMap::new(),
            },
            GraphEdit::Connect {
                link: Link {
                    from: NodeId(1),
                    from_port: "out".into(),
                    to: NodeId(2),
                    to_port: "density".into(),
                },
            },
            GraphEdit::Connect {
                link: Link {
                    from: NodeId(2),
                    from_port: "out".into(),
                    to: NodeId(3),
                    to_port: "scatter".into(),
                },
            },
        ],
    );

    // The editor's document: lowered from the live graph.
    let editor = inf_pcg::lower_graph(&g, &reg);
    assert!(editor.ok, "{:?}", editor.issues);

    // The player's document: lowered from the graph as it survives the payload.
    let payload = inf_pcg::PcgAssetPayload::from_graph(&g, editor.document.clone());
    let back = inf_pcg::PcgAssetPayload::decode(&payload.encode().unwrap()).unwrap();
    let stored = back.graph().expect("the payload carries its graph");
    assert_eq!(stored, g, "the graph itself moved through JSON");
    let player = inf_pcg::lower_graph(&stored, &reg);
    assert_eq!(
        player.document, editor.document,
        "the player re-lowers a DIFFERENT document from the editor's — a param \
         lost precision crossing `graph_json`"
    );

    // Bit-for-bit on the values that actually drive placement, so a future
    // `PartialEq` that got sloppy about floats cannot hide it.
    let e = &editor.document.layers[0].rules[0];
    let p = &player.document.layers[0].rules[0];
    assert_eq!(e.scatter.cell_size.to_bits(), p.scatter.cell_size.to_bits());
    assert_eq!(
        e.scatter.base_density.to_bits(),
        p.scatter.base_density.to_bits()
    );
    assert_eq!(
        e.scatter.altitude_offset.to_bits(),
        p.scatter.altitude_offset.to_bits()
    );
    match (&e.sampler, &p.sampler) {
        (SamplerDef::Noise(a), SamplerDef::Noise(b)) => {
            assert_eq!(a.frequency.to_bits(), b.frequency.to_bits());
            assert_eq!(a.lacunarity.to_bits(), b.lacunarity.to_bits());
            assert_eq!(a.gain.to_bits(), b.gain.to_bits());
        }
        other => panic!("expected a Noise sampler, got {other:?}"),
    }
}

proptest! {
    /// An arbitrary sampler tree survives bincode, canonically.
    #[test]
    fn any_sampler_round_trips_through_bincode(def in sampler()) {
        let (back, first, second) = bincode_round_trip(&def);
        prop_assert_eq!(&back, &def);
        prop_assert_eq!(first, second, "the encoding is not canonical");
    }

    /// …and through the human-readable codec the embedded graph and every
    /// diagnostic path use.
    #[test]
    fn any_sampler_round_trips_through_json(def in sampler()) {
        let json = serde_json::to_string(&def).expect("to json");
        let back: SamplerDef = serde_json::from_str(&json).expect("from json");
        prop_assert_eq!(back, def);
    }

    /// A whole layers × rules document — the shape P19.3's lowering produces —
    /// round-trips through the real `.inf_pcg` payload envelope, canonically.
    #[test]
    fn any_document_round_trips_through_the_payload(doc in document()) {
        let payload = inf_pcg::PcgAssetPayload::new(doc.clone());
        let bytes = payload.encode().expect("encode payload");
        let back = inf_pcg::PcgAssetPayload::decode(&bytes).expect("decode payload");
        prop_assert_eq!(&back.document, &doc);
        prop_assert_eq!(bytes, back.encode().expect("re-encode"));
    }
}
