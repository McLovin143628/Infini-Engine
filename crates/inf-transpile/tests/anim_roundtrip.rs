//! **P29.4: the `anim.*` kit survives the transpiler round trip, its identifiers
//! are frozen, and the interpreter agrees with the compiled Rust.**
//!
//! The `destruct_roundtrip` module's argument, two phases on: the lowerer and the
//! generator treat any `namespace.name` node generically, so a new kit needs no
//! transpiler code at all — and this file is what makes that a *claim* rather than
//! an assumption.
//!
//! The third arm is the one §13's pillar **S6** is about. "One IR" is only a
//! guarantee if something checks that the interpreted and the compiled halves
//! answer the same, so the same animation-driver program is run twice — once
//! through `inf_blueprint::interp` against a stand-in bridge, once as real
//! compiled Rust whose body mirrors `generate_fn`'s output — and the two traces
//! are compared step for step, with a string pin keeping the mirror honest.

use std::collections::HashMap;

use inf_blueprint::interp::{eval_fn, FnHost, Value};
use inf_blueprint::lower::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_blueprint::{Binding, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty};
use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};
use inf_transpile::{generate_fn, lift_file, FileEntry};

fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
    g.links.push(Link {
        from,
        from_port: fp.into(),
        to,
        to_port: tp.into(),
    });
}

fn int(g: &mut Graph, v: i64) -> NodeId {
    let n = g.insert("lit.int", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Int(v));
    n
}

fn float(g: &mut Graph, v: f64) -> NodeId {
    let n = g.insert("lit.float", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Float(v));
    n
}

fn text(g: &mut Graph, v: &str) -> NodeId {
    let n = g.insert("lit.str", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Text(v.into()));
    n
}

/// Generate → lift, returning the single lifted blueprint fn.
fn relift(f: &BlueprintFn) -> BlueprintFn {
    let src = generate_fn(f).expect("generate");
    let lifted = lift_file(&src).expect("lift");
    lifted
        .file
        .entries
        .into_iter()
        .find_map(|e| match e {
            FileEntry::Blueprint(bp) => Some(bp),
            FileEntry::Verbatim(_) => None,
        })
        .expect("one blueprint fn")
}

/// A `Tick` handler that drives a character's machine: set a parameter, arm a
/// trigger, take a footstep, and branch on what state the machine is in — one
/// graph exercising all four nodes, with the pure query consumed so it is not
/// dead code on the way through.
#[test]
fn the_anim_kit_graph_round_trips() {
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let ev = g.insert("event.tick", NodeUi::default());
    let who = int(&mut g, 7);
    let speed = float(&mut g, 3.5);
    let p_name = text(&mut g, "speed");
    let t_name = text(&mut g, "go");
    let n_name = text(&mut g, "footstep_l");
    let s_name = text(&mut g, "idle");

    let set_param = g.insert("anim.set_param", NodeUi::default());
    let notify = g.insert("anim.consume_notify", NodeUi::default());
    let query = g.insert("anim.query_state", NodeUi::default());
    let branch = g.insert("flow.branch", NodeUi::default());
    let trigger = g.insert("anim.set_trigger", NodeUi::default());

    wire(&mut g, ev, EXEC_THEN, set_param, "exec");
    wire(&mut g, who, "value", set_param, "entity");
    wire(&mut g, p_name, "value", set_param, "name");
    wire(&mut g, speed, "value", set_param, "value");

    wire(&mut g, set_param, EXEC_THEN, notify, "exec");
    wire(&mut g, who, "value", notify, "entity");
    wire(&mut g, n_name, "value", notify, "name");

    wire(&mut g, notify, EXEC_THEN, branch, "exec");
    wire(&mut g, who, "value", query, "entity");
    wire(&mut g, s_name, "value", query, "name");
    wire(&mut g, query, "active", branch, "condition");

    wire(&mut g, branch, "true", trigger, "exec");
    wire(&mut g, who, "value", trigger, "entity");
    wire(&mut g, t_name, "value", trigger, "name");

    let lowered = lower_graph(&g, &reg).expect("lower").pop().unwrap();
    let lifted = relift(&lowered);
    assert_eq!(&lifted, &lowered, "lowered anim IR must survive transpile");
    assert_eq!(
        generate_fn(&lowered).unwrap(),
        generate_fn(&lifted).unwrap(),
        "regeneration must be idempotent"
    );

    let src = generate_fn(&lowered).unwrap();
    for fragment in [
        "anim::set_param(",
        "anim::set_trigger(",
        "anim::query_state(",
        "anim::consume_notify(",
    ] {
        assert!(src.contains(fragment), "missing {fragment} in:\n{src}");
    }
}

/// Each call shape survives the IR round trip on its own — arities 3, 2, 2, 2.
#[test]
fn every_anim_call_ir_round_trips() {
    let e = || Expr::Lit(Lit::Int(7));
    let s = |v: &str| Expr::Lit(Lit::Str(v.into()));
    let call = |segs: &[&str], args: Vec<Expr>| Expr::Call {
        path: segs.iter().map(|x| x.to_string()).collect(),
        args,
    };
    let bind = |id: u32, value: Expr| Stmt::Let {
        id: LocalId(id),
        binding: Binding::Anon,
        ty: None,
        mutable: false,
        value,
    };
    let func = BlueprintFn {
        id: "tick".into(),
        name: "tick".into(),
        params: vec![Param {
            name: "dt".into(),
            ty: Ty::Float,
        }],
        ret: Ty::Unit,
        body: vec![
            bind(
                1,
                call(
                    &["anim", "set_param"],
                    vec![e(), s("speed"), Expr::Lit(Lit::Float(2.0))],
                ),
            ),
            bind(2, call(&["anim", "set_trigger"], vec![e(), s("go")])),
            bind(3, call(&["anim", "query_state"], vec![e(), s("idle")])),
            bind(
                4,
                call(&["anim", "consume_notify"], vec![e(), s("footstep_l")]),
            ),
        ],
    };
    assert_eq!(relift(&func), func, "every anim call shape must round-trip");
}

/// **THE FROZEN-ID PIN.** These strings are what a committed `.inf_act` stores,
/// what a generated Rust `fn` calls, and what both hosts' match arms read.
#[test]
fn the_anim_ids_are_frozen() {
    let reg = blueprint_registry();
    for id in [
        "anim.set_param",
        "anim.set_trigger",
        "anim.query_state",
        "anim.consume_notify",
    ] {
        assert!(
            reg.get(id).is_some(),
            "the frozen node id `{id}` is not registered"
        );
    }
}

// ── S6: interpreted == compiled, on the kit ─────────────────────────────────

/// The animation driver as lowering produces it, written directly so the
/// compiled mirror below can be read against it:
///
/// ```text
/// fn tick(dt) {
///     let n1 = anim::set_param(7, "speed", dt * 10);
///     let n2 = anim::consume_notify(7, "footstep_l");
///     if anim::query_state(7, "idle") { let n3 = anim::set_trigger(7, "go"); }
/// }
/// ```
fn driver() -> BlueprintFn {
    let e = || Expr::Lit(Lit::Int(7));
    let s = |v: &str| Expr::Lit(Lit::Str(v.into()));
    let call = |segs: &[&str], args: Vec<Expr>| Expr::Call {
        path: segs.iter().map(|x| x.to_string()).collect(),
        args,
    };
    let bind = |id: u32, value: Expr| Stmt::Let {
        id: LocalId(id),
        binding: Binding::Anon,
        ty: None,
        mutable: false,
        value,
    };
    BlueprintFn {
        id: "tick".into(),
        name: "tick".into(),
        params: vec![Param {
            name: "dt".into(),
            ty: Ty::Float,
        }],
        ret: Ty::Unit,
        body: vec![
            bind(
                1,
                call(
                    &["anim", "set_param"],
                    vec![
                        e(),
                        s("speed"),
                        Expr::Binary(
                            inf_blueprint::BinOp::Mul,
                            Box::new(Expr::Param("dt".into())),
                            Box::new(Expr::Lit(Lit::Float(10.0))),
                        ),
                    ],
                ),
            ),
            bind(
                2,
                call(&["anim", "consume_notify"], vec![e(), s("footstep_l")]),
            ),
            Stmt::If {
                cond: call(&["anim", "query_state"], vec![e(), s("idle")]),
                then_body: vec![bind(3, call(&["anim", "set_trigger"], vec![e(), s("go")]))],
                else_body: Vec::new(),
            },
        ],
    }
}

/// A stand-in for the bridge: the four doors' observable effects, and nothing
/// else. Both halves of the parity arm drive one of these, so "the same program"
/// really is the same program over the same state.
#[derive(Default, Clone, PartialEq, Debug)]
struct Bridge {
    params: Vec<(String, f64)>,
    triggers: Vec<String>,
    /// Notifies pending this step; `consume` takes the first match.
    pending: Vec<String>,
    /// The state the machine is in, which the driver branches on.
    state: String,
    taken: Vec<String>,
}

impl Bridge {
    fn set_param(&mut self, name: &str, value: f64) -> bool {
        self.params.push((name.to_string(), value));
        true
    }
    fn set_trigger(&mut self, name: &str) -> bool {
        self.triggers.push(name.to_string());
        true
    }
    fn query_state(&self, name: &str) -> bool {
        self.state == name
    }
    fn consume_notify(&mut self, name: &str) -> bool {
        match self.pending.iter().position(|n| n == name) {
            Some(i) => {
                self.pending.remove(i);
                self.taken.push(name.to_string());
                true
            }
            None => false,
        }
    }
}

/// The **compiled** side: a hand-written Rust translation of the generated body,
/// exercised by the normal test compile. This is what the transpiled+compiled
/// game runs; the string pin at the end of the arm is what stops it drifting.
mod compiled {
    use super::Bridge;
    // Body mirrors generate_fn(driver()) exactly.
    pub fn tick(b: &mut Bridge, dt: f64) {
        let _n1 = b.set_param("speed", dt * 10.0);
        let _n2 = b.consume_notify("footstep_l");
        if b.query_state("idle") {
            let _n3 = b.set_trigger("go");
        }
    }
}

#[test]
fn interpreted_and_compiled_drive_the_bridge_identically() {
    let f = driver();
    let steps: [(f64, &str, &[&str]); 5] = [
        (0.016, "idle", &["footstep_l"]),
        (0.5, "walk", &[]),
        (0.25, "idle", &[]),
        (1.0, "walk", &["footstep_l", "footstep_r"]),
        (0.016, "idle", &["footstep_r"]),
    ];

    // --- compiled side ---
    let mut c = Bridge::default();
    let mut compiled_trace = Vec::new();
    for (dt, state, pending) in steps {
        c.state = state.to_string();
        c.pending = pending.iter().map(|s| s.to_string()).collect();
        compiled::tick(&mut c, dt);
        compiled_trace.push(c.clone());
    }

    // --- interpreted side (the same IR the generator renders) ---
    let mut i = Bridge::default();
    let mut interp_trace = Vec::new();
    for (dt, state, pending) in steps {
        i.state = state.to_string();
        i.pending = pending.iter().map(|s| s.to_string()).collect();
        {
            let mut host = FnHost(|path: &[String], args: &[Value]| {
                let name = |k: usize| args[k].as_str().unwrap().to_string();
                Ok(match path {
                    p if p == ["anim", "set_param"] => {
                        Value::Bool(i.set_param(&name(1), args[2].as_float().unwrap()))
                    }
                    p if p == ["anim", "set_trigger"] => Value::Bool(i.set_trigger(&name(1))),
                    p if p == ["anim", "query_state"] => Value::Bool(i.query_state(&name(1))),
                    p if p == ["anim", "consume_notify"] => Value::Bool(i.consume_notify(&name(1))),
                    _ => Value::Unit,
                })
            });
            let a: HashMap<String, Value> = [("dt".into(), Value::Float(dt))].into();
            eval_fn(&f, &a, &mut host).unwrap();
        }
        interp_trace.push(i.clone());
    }

    assert_eq!(
        interp_trace, compiled_trace,
        "interpreter and compiled Rust must drive the bridge identically"
    );
    // Not vacuous: the program really did all four things.
    let last = interp_trace.last().unwrap();
    assert_eq!(last.params.len(), 5, "one parameter per step");
    assert_eq!(last.triggers.len(), 3, "armed on the three idle steps");
    assert_eq!(last.taken, ["footstep_l", "footstep_l"], "{last:?}");

    // --- keep the compiled reference honest: the generator must still emit the
    // body `compiled::tick` was written to mirror. ---
    let src = generate_fn(&f).unwrap();
    for fragment in [
        "anim::set_param(7, \"speed\", dt * 10.0);",
        "anim::consume_notify(7, \"footstep_l\");",
        "if anim::query_state(7, \"idle\")",
        "anim::set_trigger(7, \"go\");",
    ] {
        assert!(src.contains(fragment), "missing `{fragment}` in:\n{src}");
    }
}
