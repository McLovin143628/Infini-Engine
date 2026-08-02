//! **Property tests for the P19.4 grammar** — the round-trip discipline and the
//! exact-fill identity, generated rather than enumerated.
//!
//! The hand-written corpus in `grammar::dsl` and `grammar::expand` covers the
//! cases somebody thought of. These cover the ones nobody did, which is exactly
//! where a text DSL and a numeric layout fail: a shape the printer emits and the
//! parser rejects, a weight that round-trips to a different float, a span length
//! that lands one slot past a boundary.
//!
//! Three properties, each stated as an identity rather than a tolerance:
//!
//! 1. **`parse(print(g))` reproduces `g`'s rules and palette.** Line numbers are
//!    excluded and always will be — a printer cannot know the line a statement
//!    sat on, and comparing them would make the property vacuous.
//! 2. **The parser never panics**, on any input at all. It is fed authored text
//!    from a text area; the only acceptable failure is a `GrammarError` with a
//!    1-based anchor.
//! 3. **A layout partitions its span exactly**: first boundary bit-equal to `0`,
//!    last bit-equal to the span length, monotone in between, and every slot
//!    within its authored `[min, max]` unless the documented overflow/underflow
//!    policies applied.

use glam::DVec3;
use proptest::prelude::*;
use uuid::Uuid;

use inf_pcg::grammar::dsl::{
    Alternative, Element, Grammar, ModuleDef, Primary, Repeat, Rule, SizeSpec, MAX_NESTING,
};
use inf_pcg::grammar::expand::{derive, layout};
use inf_pcg::hash::Hash64;

// ── generators ──────────────────────────────────────────────────────────────

/// Symbol names that never collide with the `module` statement keyword.
fn ident() -> impl Strategy<Value = String> {
    prop::sample::select(vec![
        "A", "B", "C", "Post", "Panel", "Window", "Gap", "Bay", "Wall", "Trim", "_x", "n7",
    ])
    .prop_map(String::from)
}

/// Values that survive `f64::Display` → `str::parse` exactly (Rust prints the
/// shortest round-tripping form, so this holds for *any* finite f64; the corpus
/// deliberately includes full-mantissa values to say so out loud).
fn number() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(1.0),
        Just(2.0),
        Just(0.5),
        Just(-3.25),
        Just(1_234.567_890_123_456_7),
        Just(1e-7),
        (0.0f64..1000.0),
    ]
}

fn size_spec() -> impl Strategy<Value = SizeSpec> {
    prop_oneof![
        (0.0f64..10.0).prop_map(SizeSpec::Fixed),
        (0.0f64..5.0, 0.0f64..5.0).prop_map(|(a, b)| SizeSpec::Flex {
            min: a.min(b),
            max: a.max(b)
        }),
    ]
}

fn repeat() -> impl Strategy<Value = Repeat> {
    prop_oneof![
        Just(Repeat::One),
        Just(Repeat::Fill),
        Just(Repeat::FillAtLeastOne),
        Just(Repeat::Optional),
        (0u32..8).prop_map(Repeat::Exactly),
    ]
}

/// Alternatives, nested up to two group levels deep.
fn alternatives() -> impl Strategy<Value = Vec<Alternative>> {
    let leaf = (ident(), prop::option::of(size_spec()))
        .prop_map(|(name, size)| Primary::Symbol { name, size });
    leaf.prop_recursive(2, 12, 3, |inner| {
        prop::collection::vec(
            (
                prop::collection::vec((inner.clone(), repeat()), 0..3),
                prop_oneof![Just(1.0f64), (0.1f64..8.0)],
            )
                .prop_map(|(els, weight)| Alternative {
                    weight,
                    elements: els
                        .into_iter()
                        .map(|(primary, repeat)| Element { primary, repeat })
                        .collect(),
                }),
            1..3,
        )
        .prop_map(Primary::Group)
    })
    .prop_map(|p| {
        vec![Alternative {
            weight: 1.0,
            elements: vec![Element {
                primary: p,
                repeat: Repeat::One,
            }],
        }]
    })
}

fn module_def() -> impl Strategy<Value = ModuleDef> {
    (
        ident(),
        prop::option::of(0u128..64),
        (number(), number(), number()),
        (number(), number(), number()),
        (0.01f64..10.0),
        prop::option::of(0.0f64..10.0),
        // P19.5: the optional collider. Strictly positive on every axis — the
        // parser rejects anything else, so generating a zero would only
        // exercise the error path property 2 already owns.
        prop::option::of((0.01f64..4.0, 0.01f64..4.0, 0.01f64..4.0)),
    )
        .prop_map(|(name, mesh, off, rot, scale, size, collider)| ModuleDef {
            name,
            mesh: mesh.map(Uuid::from_u128),
            offset: DVec3::new(off.0, off.1, off.2),
            rotation_deg: DVec3::new(rot.0, rot.1, rot.2),
            scale,
            size,
            collider: collider.map(|c| DVec3::new(c.0, c.1, c.2)),
        })
}

/// A grammar assembled from generated parts, then made **legal** by giving every
/// referenced terminal a size and dropping any rule reference that would form a
/// cycle. Generating illegal grammars would only exercise the error path, which
/// property 2 already covers with arbitrary text.
fn legal_grammar() -> impl Strategy<Value = Grammar> {
    (
        prop::collection::vec(module_def(), 0..4),
        prop::collection::vec((ident(), alternatives()), 1..4),
    )
        .prop_filter_map("assembles into a legal grammar", |(modules, rules)| {
            // Deduplicate the names the generator may repeat, then render and
            // re-parse: the parser is the definition of "legal", so anything it
            // rejects is simply not part of the sample.
            let mut seen_m = std::collections::BTreeSet::new();
            let mut text = String::new();
            for m in &modules {
                if !seen_m.insert(m.name.clone()) {
                    continue;
                }
                text.push_str(&print_module(m));
            }
            let mut seen_r = std::collections::BTreeSet::new();
            for (symbol, alts) in &rules {
                if !seen_r.insert(symbol.clone()) || seen_m.contains(symbol) {
                    continue;
                }
                text.push_str(&print_rule(&Rule {
                    symbol: symbol.clone(),
                    alternatives: alts.clone(),
                }));
            }
            Grammar::parse(&text).ok()
        })
}

/// The printer is `Grammar::to_text`; these two helpers exist only to build the
/// *candidate* text a generator filters through the parser.
fn print_module(m: &ModuleDef) -> String {
    let g = Grammar::default();
    let _ = g;
    let mut single = String::from("module ");
    single.push_str(&m.name);
    single.push_str(" =");
    if let Some(mesh) = m.mesh {
        single.push_str(&format!(" mesh {mesh}"));
    }
    single.push_str(&format!(
        " offset {},{},{} rot {},{},{} scale {}",
        m.offset.x,
        m.offset.y,
        m.offset.z,
        m.rotation_deg.x,
        m.rotation_deg.y,
        m.rotation_deg.z,
        m.scale
    ));
    if let Some(size) = m.size {
        single.push_str(&format!(" size {size}"));
    }
    single.push('\n');
    single
}

fn print_rule(r: &Rule) -> String {
    // Round-trip through a one-rule grammar's own printer so the candidate text
    // and the canonical form cannot drift.
    let mut s = String::new();
    s.push_str(&r.symbol);
    s.push_str(" ->");
    print_alts(&r.alternatives, &mut s);
    s.push('\n');
    s
}

fn print_alts(alts: &[Alternative], out: &mut String) {
    for (i, alt) in alts.iter().enumerate() {
        if i > 0 {
            out.push_str(" |");
        }
        for el in &alt.elements {
            out.push(' ');
            match &el.primary {
                Primary::Symbol { name, size } => {
                    out.push_str(name);
                    match size {
                        Some(SizeSpec::Fixed(v)) => out.push_str(&format!("[{v}]")),
                        Some(SizeSpec::Flex { min, max }) => {
                            out.push_str(&format!("[{min}..{max}]"))
                        }
                        None => {}
                    }
                }
                Primary::Group(inner) => {
                    out.push('(');
                    let mut s = String::new();
                    print_alts(inner, &mut s);
                    out.push_str(s.trim_start());
                    out.push(')');
                }
            }
            match el.repeat {
                Repeat::One => {}
                Repeat::Fill => out.push('*'),
                Repeat::FillAtLeastOne => out.push('+'),
                Repeat::Optional => out.push('?'),
                Repeat::Exactly(n) => out.push_str(&format!("{{{n}}}")),
            }
        }
        if alt.weight != 1.0 {
            out.push_str(&format!(" @{}", alt.weight));
        }
    }
}

// ── the properties ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// **Property 1: the round trip.** Whatever the parser accepts, the printer
    /// re-emits and the parser accepts identically.
    #[test]
    fn any_grammar_round_trips_through_its_own_text(g in legal_grammar()) {
        let text = g.to_text();
        let back = Grammar::parse(&text)
            .unwrap_or_else(|e| panic!("printed text did not re-parse ({e}):\n{text}"));
        prop_assert_eq!(g.rules(), back.rules(), "rules differ after:\n{}", text);
        prop_assert_eq!(g.modules(), back.modules(), "modules differ after:\n{}", text);
        // …and printing is idempotent, so a canonical form really is canonical.
        prop_assert_eq!(text.clone(), back.to_text());
    }

    /// **Property 2: no input panics the parser**, and every rejection is
    /// anchored with a 1-based line and column.
    #[test]
    fn arbitrary_text_never_panics_the_parser(text in ".{0,400}") {
        match Grammar::parse(&text) {
            Ok(g) => {
                // A grammar that parsed must also print and re-parse.
                prop_assert!(Grammar::parse(&g.to_text()).is_ok());
            }
            Err(e) => {
                prop_assert!(e.line >= 1, "line {} is not 1-based", e.line);
                prop_assert!(e.col >= 1, "col {} is not 1-based", e.col);
                prop_assert!(!e.message.is_empty());
            }
        }
    }

    /// The same, over text made of the DSL's own alphabet — far denser in
    /// *nearly* valid inputs than random unicode, which is where a hand-rolled
    /// lexer actually breaks.
    #[test]
    fn dsl_shaped_text_never_panics_the_parser(
        text in "[A-Za-z0-9_ \n;|@*+?\\[\\]{}()=,.\\-># ]{0,400}"
    ) {
        let _ = Grammar::parse(&text).map(|g| Grammar::parse(&g.to_text()).is_ok());
    }

    /// **The 400-char cap above cannot reach the failure that matters**, so this
    /// one deliberately generates text that is *large and deep*: the parser's
    /// `parse_alternatives` → `parse_element` → group cycle is native-stack
    /// recursion, and the input that abuses it is not exotic — a few thousand
    /// `(` characters, well inside what somebody can paste into a text area, and
    /// reaching the editor, the cook AND the shipped player through a stored
    /// `graph_json`. A stack overflow is an uncatchable process abort, so the
    /// property is not "does not panic" but "returns an ordinary anchored
    /// error".
    #[test]
    fn deeply_nested_text_errors_instead_of_aborting(
        depth in 1usize..6000,
        closed in any::<bool>(),
        prefix in prop::sample::select(vec!["A -> ", "A -> X[1] | ", "A -> (B[1]) "]),
    ) {
        let mut text = String::from(prefix);
        text.push_str(&"(".repeat(depth));
        text.push_str("B[1]");
        if closed {
            text.push_str(&")".repeat(depth));
        }
        match Grammar::parse(&text) {
            Ok(_) => prop_assert!(
                depth < MAX_NESTING as usize,
                "depth {} parsed but the cap is {}", depth, MAX_NESTING
            ),
            Err(e) => {
                prop_assert!(e.line >= 1 && e.col >= 1);
                if depth > MAX_NESTING as usize {
                    prop_assert!(
                        e.message.contains("nested deeper than"),
                        "depth {} gave {:?}", depth, e.message
                    );
                }
            }
        }
    }

    /// Length is not depth: a very long FLAT document must still parse. The cap
    /// bounds the parser's recursion, and a test that only proved "big input is
    /// rejected" would be indistinguishable from a broken parser.
    #[test]
    fn long_flat_text_still_parses(n in 1usize..4000) {
        let text = format!("module M = size 1\nA -> {}", "M ".repeat(n));
        let g = Grammar::parse(&text).expect("a flat sequence is not deep");
        prop_assert_eq!(g.rules()[0].alternatives[0].elements.len(), n);
    }

    /// **Property 3: the exact fill.** For any legal grammar and any positive
    /// span length, the layout partitions the span — stated on `to_bits()`, so a
    /// near-miss cannot pass as equality.
    #[test]
    fn any_layout_partitions_its_span_exactly(
        g in legal_grammar(),
        len in 0.001f64..100_000.0,
        seed in 0u64..1024,
    ) {
        let Some(axiom) = g.default_axiom().map(str::to_string) else { return Ok(()); };
        let base = Hash64::new(seed);
        let d = derive(&g, &axiom, len, base);
        let lay = layout(d.slots, len, base);

        prop_assert_eq!(lay.bounds.len(), lay.slots.len() + 1);
        prop_assert_eq!(lay.bounds[0].to_bits(), 0.0f64.to_bits());
        if lay.slots.is_empty() {
            // Nothing fit: the span is untouched and entirely tail.
            prop_assert_eq!(lay.bounds, vec![0.0]);
            prop_assert_eq!(lay.tail, len);
            return Ok(());
        }
        prop_assert_eq!(
            lay.bounds.last().unwrap().to_bits(),
            len.to_bits(),
            "the span is not filled exactly"
        );
        for w in 1..lay.bounds.len() {
            prop_assert!(
                lay.bounds[w] >= lay.bounds[w - 1],
                "boundaries decreased at {}", w
            );
            prop_assert!(lay.bounds[w].is_finite());
        }
        // Every slot but the last (which absorbs the underflow tail) sits inside
        // its authored range, to within the quantization the module docs price.
        let quantum = len / (1u64 << 40) as f64;
        for (i, s) in lay.slots.iter().enumerate().take(lay.slots.len() - 1) {
            let got = lay.bounds[i + 1] - lay.bounds[i];
            let slack = quantum + 1e-9 * len.max(1.0);
            prop_assert!(
                got >= s.size.min() - slack && got <= s.size.max() + slack,
                "slot {} ({}) got {}, range {}..{}",
                i, s.symbol, got, s.size.min(), s.size.max()
            );
        }
        // Derivation and layout are pure: run them again and get the same thing.
        let d2 = derive(&g, &axiom, len, base);
        prop_assert_eq!(&layout(d2.slots, len, base), &lay);
    }
}
