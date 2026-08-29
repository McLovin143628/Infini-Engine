//! Name resolution for calls: `namespace.verb(…)` → an IR call path, checked
//! against the node kit **at parse time**.
//!
//! # The two seams, and why the resolver has to know both
//!
//! The SCRIPT0 audit's finding 9 is the one this module exists to honour: the
//! surface a script reaches the world through is **two** seams, not one.
//!
//! * `math.*` (except the operators) is dispatched **hostlessly**, by
//!   `inf_blueprint::interp::dispatch_math`, straight into
//!   `inf_blueprint::math_builtins` — the same functions the transpiled Rust
//!   calls, which is why the math palette passes interpreter-vs-compiled parity
//!   *by construction* rather than by two implementations matching. `math.sin`
//!   and `math.cos` route to `inf_math::portable::psin64`/`pcos64` there, not to
//!   `std`.
//! * Everything external — engine calls, member variables, spawning, physics,
//!   audio — crosses the `Host` trait.
//!
//! A `.infini` identifier must resolve into one of those two and **never** into
//! `std` or `libm`. It does so structurally: a call must name a **registered
//! node**, and the registry is the node kit. That is what makes "a script cannot
//! name a transcendental — only a verb" a property of the surface rather than of
//! a review.
//!
//! # What is refused, and why each refusal is a good one
//!
//! | shape | verdict |
//! |---|---|
//! | `math.add(a, b)`, `cmp.lt(a, b)`, `logic.not(a)`, `math.neg(a)` | refused — these are the `+`, `<`, `not`, `-` **operators** here. Two spellings for one IR node is a second grammar to keep in sync |
//! | `flow.branch(…)`, `flow.while(…)` | refused — control flow is *syntax* in a text face: `if`, `while`, `for`, `return` |
//! | `event.tick(…)` | refused — an event is a handler's *header*, not a call |
//! | `lit.float(…)` | refused — a literal is written `1.5` |
//! | `vars.get(…)`, `nodestate.set(…)` | refused — the lowerer's own internals. A member variable is written by its name; `var.get("…")` is the escape hatch for a name that is not an identifier |
//! | an unregistered `ns.verb` | refused, and the message says whether the *namespace* or the *verb* is the unknown half |
//!
//! Every one of those is a value with a line and a column, never a panic.

use inf_blueprint::lower::{host_call_path, role_of};
use inf_blueprint::nodekit::{blueprint_registry, NodeRole, EXEC_IN};
use inf_graph::NodeRegistry;

/// What a resolved call is, once the registry has spoken.
#[derive(Debug, Clone, PartialEq)]
pub struct Verb {
    /// The IR call path (`engine::set_rotation`, `physics2d::raycast::hit`).
    pub path: Vec<String>,
    /// The node `type_id` it came from — what the emitter prints back.
    pub type_id: String,
    /// The declared data inputs, in order, as `(name, required)`.
    pub inputs: Vec<(String, bool)>,
    /// True when the node carries an exec pin, i.e. it is an *action*: a
    /// statement in a graph and a statement here.
    pub is_action: bool,
    /// True when the call yields a value usable in an expression.
    pub produces_value: bool,
}

/// Why a call could not be resolved. Every variant is a sentence a person can
/// act on — the message names the remedy, not only the fault.
#[derive(Debug, Clone, PartialEq)]
pub enum VerbError {
    /// The path had fewer than two segments.
    NotNamespaced(String),
    /// The whole namespace is unknown.
    NoNamespace(String),
    /// The namespace is real, the verb is not.
    NoVerb { namespace: String, verb: String },
    /// A registered node that is deliberately not callable from text.
    NotCallable { type_id: String, why: String },
    /// A multi-output pure node needs its output naming, or a single-output one
    /// was given a third segment it has no use for.
    Output { type_id: String, why: String },
}

impl VerbError {
    /// The diagnostic text.
    pub fn message(&self) -> String {
        match self {
            VerbError::NotNamespaced(name) => format!(
                "`{name}` is not a verb — a call names a namespace and a verb, \
                 like `debug.print(\"hello\")`"
            ),
            VerbError::NoNamespace(ns) => {
                format!("there is no `{ns}` namespace in the InfiniScript verb surface")
            }
            VerbError::NoVerb { namespace, verb } => {
                format!("the `{namespace}` namespace has no verb `{verb}`")
            }
            VerbError::NotCallable { type_id, why } => format!("`{type_id}` {why}"),
            VerbError::Output { type_id, why } => format!("`{type_id}` {why}"),
        }
    }
}

/// The verb surface: the blueprint node kit, read as a call table.
pub struct Verbs {
    reg: NodeRegistry,
}

impl Default for Verbs {
    fn default() -> Self {
        Self::new()
    }
}

impl Verbs {
    pub fn new() -> Self {
        Self {
            reg: blueprint_registry(),
        }
    }

    /// The underlying registry (the emitter reads output ports from it).
    pub fn registry(&self) -> &NodeRegistry {
        &self.reg
    }

    /// How many verbs the surface has, and how many namespaces they fall into.
    /// Reported by the spec's own arm rather than written down twice.
    pub fn census(&self) -> (usize, usize) {
        let mut namespaces: Vec<&str> = self
            .reg
            .ordered()
            .filter_map(|d| d.type_id.split('.').next())
            .collect();
        namespaces.sort_unstable();
        namespaces.dedup();
        (self.reg.len(), namespaces.len())
    }

    /// Resolve `segments` (already split on `.`) into a call.
    pub fn resolve(&self, segments: &[String]) -> Result<Verb, VerbError> {
        if segments.len() < 2 {
            return Err(VerbError::NotNamespaced(segments.join(".")));
        }
        if segments.len() > 3 {
            return Err(VerbError::NoVerb {
                namespace: segments[0].clone(),
                verb: segments[1..].join("."),
            });
        }
        let type_id = format!("{}.{}", segments[0], segments[1]);
        let Some(def) = self.reg.get(&type_id) else {
            let ns = format!("{}.", segments[0]);
            return Err(if self.reg.ordered().any(|d| d.type_id.starts_with(&ns)) {
                VerbError::NoVerb {
                    namespace: segments[0].clone(),
                    verb: segments[1].clone(),
                }
            } else {
                VerbError::NoNamespace(segments[0].clone())
            });
        };

        let has_exec = def.input(EXEC_IN).is_some();
        let role = role_of(&type_id, has_exec);
        let not_callable = |why: &str| {
            Err(VerbError::NotCallable {
                type_id: type_id.clone(),
                why: why.to_string(),
            })
        };
        match role {
            NodeRole::BinaryOp => {
                return not_callable(&format!(
                    "is the `{}` operator in InfiniScript — write `a {} b`",
                    operator_spelling(&type_id),
                    operator_spelling(&type_id)
                ))
            }
            NodeRole::NotOp => return not_callable("is the `not` operator — write `not a`"),
            NodeRole::NegOp => return not_callable("is unary minus — write `-(a)`"),
            NodeRole::Event => {
                return not_callable(
                    "is an event, not a call — a handler is written `on <event>(…) … end`",
                )
            }
            NodeRole::Literal => {
                return not_callable("is a literal — write the value itself, like `1.5`")
            }
            NodeRole::VarGet | NodeRole::VarSet => {
                // Handled by the parser before it reaches here (bare names and
                // the `var.get("…")` escape hatch); reaching this arm means the
                // parser's own routing changed.
                return not_callable(
                    "is a member variable — write its name, or `var.get(\"…\")` when the \
                     name is not an identifier",
                );
            }
            NodeRole::Branch
            | NodeRole::Sequence
            | NodeRole::Return
            | NodeRole::WhileLoop
            | NodeRole::ForLoop
            | NodeRole::DoOnce
            | NodeRole::FlipFlop
            | NodeRole::Gate => {
                return not_callable(&format!(
                    "is control flow, which InfiniScript writes as syntax — {}",
                    flow_spelling(&type_id)
                ))
            }
            NodeRole::PureCall | NodeRole::Action => {}
        }

        let inputs: Vec<(String, bool)> = def
            .inputs
            .iter()
            .filter(|p| !p.ty.is_exec() && !p.param_pin)
            .map(|p| (p.name.clone(), p.required))
            .collect();
        let outputs: Vec<&str> = def
            .outputs
            .iter()
            .filter(|p| !p.ty.is_exec())
            .map(|p| p.name.as_str())
            .collect();

        let mut path = host_call_path(&type_id);
        // A pure node with several data outputs fans each pin to its own
        // `…::<field>` call, so one wire carries one scalar (the lowerer's rule).
        // In text that field is the third segment.
        let multi = !has_exec && outputs.len() > 1;
        match (segments.get(2), multi) {
            (Some(field), true) => {
                if !outputs.contains(&field.as_str()) {
                    return Err(VerbError::Output {
                        type_id: type_id.clone(),
                        why: format!(
                            "has no result `{field}` — its results are {}",
                            list(&outputs)
                        ),
                    });
                }
                path.push(field.clone());
            }
            (None, true) => {
                return Err(VerbError::Output {
                    type_id: type_id.clone(),
                    why: format!(
                        "returns several results, so a call must name one: {}",
                        list(&outputs)
                    ),
                })
            }
            (Some(field), false) => {
                return Err(VerbError::Output {
                    type_id: type_id.clone(),
                    why: format!("has no result `{field}` to name"),
                })
            }
            (None, false) => {}
        }

        Ok(Verb {
            path,
            type_id,
            inputs,
            is_action: has_exec,
            produces_value: !outputs.is_empty(),
        })
    }
}

/// `["hit", "point"]` → "`hit`, `point`".
fn list(items: &[&str]) -> String {
    items
        .iter()
        .map(|s| format!("`{s}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The infix spelling of an operator node, for its refusal message.
fn operator_spelling(type_id: &str) -> &'static str {
    match type_id {
        "math.add" => "+",
        "math.sub" => "-",
        "math.mul" => "*",
        "math.div" => "/",
        "math.rem" => "%",
        "cmp.eq" => "==",
        "cmp.ne" => "~=",
        "cmp.lt" => "<",
        "cmp.le" => "<=",
        "cmp.gt" => ">",
        "cmp.ge" => ">=",
        "logic.and" => "and",
        "logic.or" => "or",
        _ => "?",
    }
}

/// The syntax a flow node is written as.
fn flow_spelling(type_id: &str) -> &'static str {
    match type_id {
        "flow.branch" => "`if … then … end`",
        "flow.while" => "`while … do … end`",
        "flow.for" => "`for i = first, last do … end`",
        "flow.return" => "`return`",
        "flow.sequence" => "statements simply follow one another",
        "flow.do_once" => "guard it with a member variable (`if not fired then … end`)",
        "flow.flip_flop" => "guard it with a member variable",
        "flow.gate" => "guard it with a member variable",
        _ => "as syntax",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(s: &str) -> Vec<String> {
        s.split('.').map(str::to_string).collect()
    }

    #[test]
    fn an_action_resolves_to_its_host_path() {
        let v = Verbs::new();
        let r = v.resolve(&segs("engine.set_rotation")).unwrap();
        assert_eq!(r.path, vec!["engine".to_string(), "set_rotation".into()]);
        assert!(r.is_action);
        assert_eq!(r.inputs, vec![("angle".to_string(), true)]);
        assert!(!r.produces_value);
    }

    /// The **builtins seam**: `math.sqrt` is not a Host call at all.
    #[test]
    fn a_math_builtin_resolves_hostlessly_and_produces_a_value() {
        let v = Verbs::new();
        let r = v.resolve(&segs("math.sqrt")).unwrap();
        assert_eq!(r.path, vec!["math".to_string(), "sqrt".into()]);
        assert!(!r.is_action);
        assert!(r.produces_value);
    }

    /// The dispatcher rename is applied on the way in, so a script's
    /// `dispatch.call` reaches the sim's one dispatcher.
    #[test]
    fn the_dispatcher_verbs_carry_their_host_rename() {
        let v = Verbs::new();
        let r = v.resolve(&segs("dispatch.call")).unwrap();
        assert_eq!(r.path, vec!["event".to_string(), "dispatch".into()]);
        assert_eq!(r.type_id, "dispatch.call");
    }

    #[test]
    fn a_multi_result_query_must_name_its_result() {
        let v = Verbs::new();
        let e = v.resolve(&segs("physics2d.raycast")).unwrap_err();
        assert!(e.message().contains("several results"), "{}", e.message());
        let r = v.resolve(&segs("physics2d.raycast.hit")).unwrap();
        assert_eq!(
            r.path,
            vec!["physics2d".to_string(), "raycast".into(), "hit".into()]
        );
        let e = v.resolve(&segs("physics2d.raycast.nope")).unwrap_err();
        assert!(e.message().contains("no result `nope`"), "{}", e.message());
    }

    #[test]
    fn a_single_result_query_refuses_a_third_segment() {
        let v = Verbs::new();
        assert!(v.resolve(&segs("physics2d.is_grounded")).is_ok());
        let e = v.resolve(&segs("physics2d.is_grounded.value")).unwrap_err();
        assert!(e.message().contains("no result `value`"), "{}", e.message());
    }

    #[test]
    fn the_operators_are_refused_by_their_infix_spelling() {
        let v = Verbs::new();
        for (id, want) in [
            ("math.add", "write `a + b`"),
            ("cmp.lt", "write `a < b`"),
            ("logic.and", "write `a and b`"),
            ("logic.not", "write `not a`"),
            ("math.neg", "write `-(a)`"),
        ] {
            let e = v.resolve(&segs(id)).unwrap_err();
            assert!(e.message().contains(want), "{id}: {}", e.message());
        }
    }

    #[test]
    fn control_flow_and_events_and_literals_are_refused_with_their_syntax() {
        let v = Verbs::new();
        for (id, want) in [
            ("flow.branch", "if … then … end"),
            ("flow.while", "while … do … end"),
            ("flow.for", "for i = first, last do … end"),
            ("flow.sequence", "statements simply follow"),
            ("flow.do_once", "member variable"),
            ("event.tick", "on <event>"),
            ("lit.float", "write the value itself"),
            ("var.get", "write its name"),
        ] {
            let e = v.resolve(&segs(id)).unwrap_err();
            assert!(e.message().contains(want), "{id}: {}", e.message());
        }
    }

    #[test]
    fn an_unknown_name_says_which_half_it_did_not_know() {
        let v = Verbs::new();
        assert_eq!(
            v.resolve(&segs("math.sinn")).unwrap_err(),
            VerbError::NoVerb {
                namespace: "math".into(),
                verb: "sinn".into()
            }
        );
        assert_eq!(
            v.resolve(&segs("moth.sin")).unwrap_err(),
            VerbError::NoNamespace("moth".into())
        );
        assert!(matches!(
            v.resolve(&segs("print")).unwrap_err(),
            VerbError::NotNamespaced(_)
        ));
    }

    /// **The determinism claim, structurally.** Every namespace a script can
    /// reach is a namespace of the node kit; there is no path from `.infini` text
    /// to `std` or `libm` that does not go through this table.
    #[test]
    fn the_whole_surface_is_the_node_kit_and_nothing_else() {
        let v = Verbs::new();
        let (verbs, namespaces) = v.census();
        assert_eq!(
            (verbs, namespaces),
            (132, 26),
            "the verb surface moved; the spec's table, the memo and the \
             generated API manual all quote these two numbers"
        );
        // Everything that resolves came out of the registry, so its path is a
        // node's host path — never an arbitrary string.
        let mut resolvable = 0;
        for def in v.registry().ordered() {
            let mut segments: Vec<String> = def.type_id.split('.').map(str::to_string).collect();
            // Multi-result pure queries need their result naming.
            if let Some(out) = def
                .outputs
                .iter()
                .find(|p| !p.ty.is_exec())
                .filter(|_| def.outputs.iter().filter(|p| !p.ty.is_exec()).count() > 1)
            {
                segments.push(out.name.clone());
            }
            if let Ok(r) = v.resolve(&segments) {
                resolvable += 1;
                assert!(
                    v.registry().contains(&r.type_id),
                    "{} resolved to an unregistered type",
                    def.type_id
                );
            }
        }
        assert!(
            resolvable >= 87,
            "only {resolvable} of {verbs} verbs are callable from text — the \
             refusals are meant to be the operators, the flow palette, the \
             literals, the events and the two variable nodes, not most of the kit"
        );
    }
}
