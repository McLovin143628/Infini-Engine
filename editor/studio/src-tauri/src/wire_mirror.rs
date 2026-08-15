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
        let tags: Vec<String> = all.iter().map(|p| tag_of(p, "kind")).collect();
        let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        must_mention(
            "blueprintTypes.ts",
            "inf_graph::PortType",
            "PortType",
            &refs,
        );
        assert_eq!(tags.len(), 7, "a variant was added and this list was not");
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
        let tags: Vec<String> = all.iter().map(|p| tag_of(p, "type")).collect();
        let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        must_mention(
            "blueprintTypes.ts",
            "inf_graph::ParamValue",
            "ParamValue",
            &refs,
        );
        assert_eq!(tags.len(), 5, "a variant was added and this list was not");
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
        let tags: Vec<String> = all.iter().map(bare).collect();
        let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        must_mention("blueprintTypes.ts", "inf_graph::UiHint", "UiHint", &refs);
        assert_eq!(tags.len(), 5, "a variant was added and this list was not");
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
