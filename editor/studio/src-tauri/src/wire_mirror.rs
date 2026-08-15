//! **The drift gate for the hand-written TypeScript wire mirrors** (L7.M7).
//!
//! Four frontend files — `blueprintTypes.ts`, `materialTypes.ts`,
//! `pcgTypes.ts`, `smTypes.ts` — restate Ring-0 serde shapes by hand, because
//! their source of truth is `inf-graph` / `inf-blueprint` / `inf-pcg` /
//! `inf-anim` rather than `inf_editor_core::ipc`, and only the latter is
//! ts-rs-generated and drift-checked in CI. So these four sat **outside** the
//! bindings gate: a Rust enum could grow a variant and the canvas would simply
//! stop knowing about it, with nothing red anywhere.
//!
//! # What is checked, and what is not
//!
//! Every **discriminant string** a Rust wire enum can put on the wire must
//! appear in the mirror that claims to describe it. That is the drift that
//! actually happens (a variant is added; the TS union is not extended) and it is
//! the drift with no other symptom: the frontend's exhaustiveness checking has
//! nothing to be exhaustive over.
//!
//! It is deliberately **not** a structural equality check. The mirrors are
//! hand-written *because* they differ in shape from the Rust — they inline
//! aliases, widen numbers to `number`, and describe tagged unions as TS unions.
//! A gate that demanded structural identity would demand ts-rs, which was
//! rejected for these four in P6.2 for a reason that still holds: the types are
//! Ring 0 and `inf-graph` must not grow a ts-rs dependency to serve the editor.
//!
//! The direction is Rust → TS on purpose. A stale name left in the TS is
//! harmless (nothing produces it); a *missing* one is a value the canvas cannot
//! render.
//!
//! # Round 2 (R2.F2): the first version of this gate could not see drift
//!
//! It was written as `assert_eq!(tags.len(), 7, "a variant was added and this
//! list was not")` over a **hand-written** sample array — two constants
//! compared with each other, which is true forever. Adding a `PortType` variant
//! in `inf-graph` left every arm here green and the canvas ignorant, i.e. the
//! gate protected the direction it explicitly says it does not care about (TS
//! regression) and not the one it exists for. The eighth vacuous gate of this
//! campaign, and this module's own.
//!
//! What replaces it has two halves, and the drift has to get past both:
//!
//! 1. **A compile-time census.** `*_variant_name` is an exhaustive `match` with
//!    no wildcard arm, so a new variant in `inf-graph` **fails to compile**
//!    here. That is the loudest enforcement available and it needs no list.
//! 2. **A source census.** [`variants_declared`] reads the enum's own `pub
//!    enum` block out of the `inf-graph` source and asserts the sample list
//!    covers every declared name. That is what stops the compile error from
//!    being answered with a one-line arm and nothing else: the new variant must
//!    be *constructed*, so its wire tag is computed, so the tag must appear in
//!    the mirror.
//!
//! Same shape as `inf_ecs::components`' freeze-table census (R2.D) and as the
//! LSP URI pin in `commands::lsp` — a claim about two languages agreeing is
//! only worth the mechanism that can see them disagree.

use std::path::PathBuf;

/// Where the frontend's hand-written mirrors live, from this crate.
fn mirror(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("src")
        .join("lib")
        .join(name)
}

fn read_mirror(name: &str) -> String {
    let path = mirror(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("the mirror {} must exist: {e}", path.display()))
}

/// Where `inf-graph`'s wire enums are declared, from this crate.
fn graph_source(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("crates")
        .join("inf-graph")
        .join("src")
        .join(file)
}

/// The variant names declared by `pub enum <name>` in an `inf-graph` source
/// file — the census half that no hand-written list can satisfy by itself.
///
/// Deliberately a text read rather than a macro: `inf-graph` is Ring 0 and must
/// not grow a reflection dependency to serve the editor's drift gate, which is
/// the same reason these four mirrors are hand-written in the first place.
fn variants_declared(file: &str, enum_name: &str) -> Vec<String> {
    let path = graph_source(file);
    // The P22 CRLF law: a `.rs` read by a test is normalized first.
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is readable: {e}", path.display()))
        .replace("\r\n", "\n");
    let header = format!("pub enum {enum_name} {{");
    let start = src
        .find(&header)
        .unwrap_or_else(|| panic!("{} declares no `{header}`", path.display()))
        + header.len();

    let mut names = Vec::new();
    let mut depth = 1usize;
    for line in src[start..].lines() {
        let trimmed = line.trim();
        // A variant is an identifier at the enum's own brace depth. Track depth
        // first so struct-variant bodies and their fields are skipped whole.
        if depth == 1 {
            if let Some(ident) = trimmed
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .filter(|s| !s.is_empty())
            {
                let first = ident.chars().next().unwrap_or(' ');
                if first.is_ascii_uppercase() {
                    names.push(ident.to_string());
                }
            }
        }
        depth += trimmed.matches('{').count();
        depth = depth.saturating_sub(trimmed.matches('}').count());
        if depth == 0 {
            break;
        }
    }
    assert!(
        !names.is_empty(),
        "the census read no variants out of `{enum_name}` in {} — a census that finds \
         nothing covers everything",
        path.display()
    );
    names
}

/// Assert a sample list names every variant the Rust source declares, in both
/// directions.
fn census_matches(file: &str, enum_name: &str, sampled: &[&str]) {
    let declared = variants_declared(file, enum_name);
    let sampled_set: std::collections::BTreeSet<&str> = sampled.iter().copied().collect();
    let missing: Vec<&String> = declared
        .iter()
        .filter(|d| !sampled_set.contains(d.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "`{enum_name}` declares {missing:?}, which this gate never constructs — so their wire \
         tags are never computed and the TypeScript mirror is never checked for them. Add a \
         sample (the exhaustive `match` beside it already refused to compile without one)."
    );
    let declared_set: std::collections::BTreeSet<&str> =
        declared.iter().map(String::as_str).collect();
    let stale: Vec<&&str> = sampled
        .iter()
        .filter(|s| !declared_set.contains(*s))
        .collect();
    assert!(
        stale.is_empty(),
        "this gate samples {stale:?}, which `{enum_name}` no longer declares — a check about \
         nothing, which the next variant to take that name inherits"
    );
}

/// Assert every `needle` appears, quoted, in `haystack`.
fn must_mention(file: &str, source: &str, kind: &str, needles: &[&str]) {
    let text = read_mirror(file);
    let missing: Vec<&str> = needles
        .iter()
        .copied()
        .filter(|n| !text.contains(&format!("\"{n}\"")))
        .collect();
    assert!(
        missing.is_empty(),
        "{file} does not mention {missing:?} — {kind} in {source} has grown a variant the \
         canvas cannot render, and nothing else in this repo would have said so"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_graph::{GraphEdit, NodeId, ParamValue, PortType, UiHint};
    use serde_json::Value;

    /// The tag a serialized value carries under `key`.
    fn tag_of<T: serde::Serialize>(v: &T, key: &str) -> String {
        let json: Value = serde_json::to_value(v).expect("a wire type serializes");
        json.get(key)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("no `{key}` tag in {json}"))
            .to_string()
    }

    /// A unit-variant enum serializes to a bare string.
    fn bare<T: serde::Serialize>(v: &T) -> String {
        serde_json::to_value(v)
            .expect("serializes")
            .as_str()
            .expect("a unit variant is a bare string")
            .to_string()
    }

    /// **The compile-time census for `PortType`.** No wildcard arm: a variant
    /// added in `inf-graph` stops this crate's tests from compiling, which is
    /// the enforcement a stale `assert_eq!(len, 7)` only claimed to be.
    fn port_type_variant_name(p: &PortType) -> &'static str {
        match p {
            PortType::Exec => "Exec",
            PortType::Bool => "Bool",
            PortType::Int => "Int",
            PortType::Float => "Float",
            PortType::Str => "Str",
            PortType::Named(_) => "Named",
            PortType::Wildcard => "Wildcard",
        }
    }

    /// The compile-time census for `ParamValue`.
    fn param_value_variant_name(v: &ParamValue) -> &'static str {
        match v {
            ParamValue::Bool(_) => "Bool",
            ParamValue::Int(_) => "Int",
            ParamValue::Float(_) => "Float",
            ParamValue::Text(_) => "Text",
            ParamValue::Enum(_) => "Enum",
        }
    }

    /// The compile-time census for `UiHint`.
    fn ui_hint_variant_name(h: &UiHint) -> &'static str {
        match h {
            UiHint::Number => "Number",
            UiHint::Text => "Text",
            UiHint::Toggle => "Toggle",
            UiHint::Choice => "Choice",
            UiHint::Multiline => "Multiline",
        }
    }

    /// The compile-time census for `GraphEdit` — the wire the canvas *writes*.
    fn graph_edit_variant_name(e: &GraphEdit) -> &'static str {
        match e {
            GraphEdit::AddNode { .. } => "AddNode",
            GraphEdit::RemoveNode { .. } => "RemoveNode",
            GraphEdit::RestoreNode { .. } => "RestoreNode",
            GraphEdit::Connect { .. } => "Connect",
            GraphEdit::Disconnect { .. } => "Disconnect",
            GraphEdit::SetParam { .. } => "SetParam",
            GraphEdit::ClearParam { .. } => "ClearParam",
            GraphEdit::SetDisabled { .. } => "SetDisabled",
            GraphEdit::MoveNode { .. } => "MoveNode",
            GraphEdit::ResizeNode { .. } => "ResizeNode",
            GraphEdit::SetTitle { .. } => "SetTitle",
        }
    }

    /// `PortType` decides what a pin *is*, and a wire the frontend cannot type
    /// is a wire it will not let the author draw.
    #[test]
    fn blueprint_mirror_knows_every_port_type() {
        let all = [
            PortType::Exec,
            PortType::Bool,
            PortType::Int,
            PortType::Float,
            PortType::Str,
            PortType::Wildcard,
            PortType::Named("Vec3".into()),
        ];
        let sampled: Vec<&str> = all.iter().map(port_type_variant_name).collect();
        census_matches("registry.rs", "PortType", &sampled);

        let tags: Vec<String> = all.iter().map(|p| tag_of(p, "kind")).collect();
        let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        must_mention(
            "blueprintTypes.ts",
            "inf_graph::PortType",
            "PortType",
            &refs,
        );
    }

    /// `ParamValue` is what an inspector control reads and writes.
    #[test]
    fn blueprint_mirror_knows_every_param_value() {
        let all = [
            ParamValue::Bool(true),
            ParamValue::Int(1),
            ParamValue::Float(1.0),
            ParamValue::Text("x".into()),
            ParamValue::Enum("x".into()),
        ];
        let sampled: Vec<&str> = all.iter().map(param_value_variant_name).collect();
        census_matches("model.rs", "ParamValue", &sampled);

        let tags: Vec<String> = all.iter().map(|p| tag_of(p, "type")).collect();
        let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        must_mention(
            "blueprintTypes.ts",
            "inf_graph::ParamValue",
            "ParamValue",
            &refs,
        );
    }

    /// `UiHint` chooses the control. A hint the mirror does not know falls back
    /// to a text box, silently.
    #[test]
    fn blueprint_mirror_knows_every_ui_hint() {
        let all = [
            UiHint::Number,
            UiHint::Text,
            UiHint::Toggle,
            UiHint::Choice,
            UiHint::Multiline,
        ];
        let sampled: Vec<&str> = all.iter().map(ui_hint_variant_name).collect();
        census_matches("registry.rs", "UiHint", &sampled);

        let tags: Vec<String> = all.iter().map(bare).collect();
        let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        must_mention("blueprintTypes.ts", "inf_graph::UiHint", "UiHint", &refs);
    }

    /// The census's own falsifier, in both directions: it has to be able to see
    /// a variant the sample list forgot, and a sample list that names a variant
    /// the enum no longer has. Without these two, "the census matched" is the
    /// same sentence a vacuous gate says.
    #[test]
    fn the_census_can_see_a_forgotten_variant() {
        let declared = variants_declared("registry.rs", "PortType");
        assert!(
            declared.contains(&"Wildcard".to_string()) && declared.len() >= 7,
            "the census is not reading `PortType`: {declared:?}"
        );

        let short: Vec<&str> = declared
            .iter()
            .map(String::as_str)
            .filter(|n| *n != "Wildcard")
            .collect();
        let forgot = std::panic::catch_unwind(|| census_matches("registry.rs", "PortType", &short));
        assert!(
            forgot.is_err(),
            "a sample list missing `Wildcard` must fail the census"
        );

        let mut invented: Vec<&str> = declared.iter().map(String::as_str).collect();
        invented.push("QuaternionOfHolding");
        let stale =
            std::panic::catch_unwind(|| census_matches("registry.rs", "PortType", &invented));
        assert!(
            stale.is_err(),
            "a sample list naming a variant that does not exist must fail the census"
        );
    }

    /// `GraphEdit` is the wire the canvas *writes*. The frontend hand-builds its
    /// kebab-case tagged JSON, so a Rust variant the mirror does not know is an
    /// edit the author cannot make — and one the mirror invents is an edit the
    /// backend silently refuses, which is C4-44's other half.
    #[test]
    fn blueprint_mirror_knows_every_graph_edit() {
        let n = NodeId(1);
        let link = inf_graph::Link {
            from: n,
            from_port: "a".into(),
            to: n,
            to_port: "b".into(),
        };
        let all = [
            GraphEdit::AddNode {
                id: n,
                type_id: "t".into(),
                x: 0.0,
                y: 0.0,
                params: Default::default(),
            },
            GraphEdit::RemoveNode { id: n },
            GraphEdit::RestoreNode {
                node: inf_graph::Node {
                    id: n,
                    type_id: "t".into(),
                    params: Default::default(),
                    ui: Default::default(),
                    disabled: false,
                },
                links: vec![link.clone()],
            },
            GraphEdit::Connect { link: link.clone() },
            GraphEdit::Disconnect { link },
            GraphEdit::SetParam {
                id: n,
                name: "k".into(),
                value: ParamValue::Int(1),
            },
            GraphEdit::ClearParam {
                id: n,
                name: "k".into(),
            },
            GraphEdit::SetDisabled {
                id: n,
                disabled: true,
            },
            GraphEdit::MoveNode {
                id: n,
                x: 0.0,
                y: 0.0,
            },
            GraphEdit::ResizeNode {
                id: n,
                w: 1.0,
                h: 1.0,
            },
            GraphEdit::SetTitle {
                id: n,
                title: "t".into(),
            },
        ];
        let sampled: Vec<&str> = all.iter().map(graph_edit_variant_name).collect();
        census_matches("edits.rs", "GraphEdit", &sampled);

        let tags: Vec<String> = all.iter().map(|e| tag_of(e, "kind")).collect();
        let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        must_mention(
            "blueprintTypes.ts",
            "inf_graph::GraphEdit",
            "GraphEdit",
            &refs,
        );
    }

    /// The three domain mirrors restate the SAME `inf-graph` substrate — the
    /// material, PCG and state-machine canvases are the blueprint canvas with a
    /// different registry — so each must know the shapes it re-declares. The
    /// check is per file rather than shared, because "it is in one of them" is
    /// not what a `import { PortType } from "./materialTypes"` gets.
    #[test]
    fn every_domain_mirror_knows_the_substrate_it_restates() {
        for file in ["materialTypes.ts", "pcgTypes.ts", "smTypes.ts"] {
            let text = read_mirror(file);
            // Each of these files either re-exports the blueprint mirror's
            // substrate types or restates them. Whichever it does, the name has
            // to be there — an import of a type that does not exist is a
            // typecheck failure, and a restatement that has drifted is not.
            assert!(
                text.contains("blueprintTypes") || text.contains("PortType"),
                "{file} neither imports the shared substrate nor restates it; the \
                 canvas it serves cannot be typed against the graph the backend sends"
            );
        }
    }

    /// The gate's own falsifier: `must_mention` has to be able to fail. A drift
    /// check that reads the wrong file passes forever, and this campaign has
    /// caught five vacuous gates already.
    #[test]
    #[should_panic(expected = "does not mention")]
    fn the_gate_can_see_a_missing_variant() {
        must_mention(
            "blueprintTypes.ts",
            "a variant that does not exist",
            "PortType",
            &["quaternion-of-holding"],
        );
    }
}
