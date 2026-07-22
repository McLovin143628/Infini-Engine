//! Interpreter-vs-compiled parity for the math palette (ROADMAP B-P1).
//!
//! Each math family is authored as a tiny fixture graph
//! (`begin_play → return math.op(literals)`), lowered to IR, and **interpreted**;
//! the result is compared against the **compiled** path — real Rust that calls
//! the very same `crate::math_builtins` functions the transpiler emits
//! (`math::<name>` below re-exports them, exactly the module path a generated
//! blueprint file resolves against). Because both sides bottom out in one
//! implementation, parity holds by construction; the sweep (negative / zero /
//! large / NaN-producing inputs) guards the *coercion + dispatch* around it.
//!
//! A string pin keeps the generator honest: the emitted Rust must still be the
//! `math::<name>(..)` call the compiled mirror is written against.

use std::collections::HashMap;

use inf_blueprint::interp::{eval_fn, PureHost, Value};
use inf_blueprint::lower_graph;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};
use inf_transpile::generate_fn;

// The compiled mirror's `math` module: exactly what a generated blueprint file
// binds `math::*` to. Calling these IS the compiled path.
mod math {
    pub use inf_blueprint::math_builtins::*;
}

fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
    g.links.push(Link {
        from,
        from_port: fp.into(),
        to,
        to_port: tp.into(),
    });
}

fn lit_float(g: &mut Graph, v: f64) -> NodeId {
    let n = g.insert("lit.float", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Float(v));
    n
}

fn lit_int(g: &mut Graph, v: i64) -> NodeId {
    let n = g.insert("lit.int", NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Int(v));
    n
}

/// Build `begin_play → return op(<args>)`, lower, and interpret it (math is
/// hostless, so `PureHost` suffices). `args` are `(port, literal-node)`.
fn interp_math(g: &mut Graph, op: NodeId, args: &[(&str, NodeId)]) -> Value {
    let reg = blueprint_registry();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let ret = g.insert("flow.return", NodeUi::default());
    for (port, src) in args {
        wire(g, *src, "value", op, port);
    }
    wire(g, op, "out", ret, "value");
    wire(g, bp, EXEC_THEN, ret, "exec");
    let f = lower_graph(g, &reg).unwrap().pop().unwrap();
    eval_fn(&f, &HashMap::new(), &mut PureHost).unwrap()
}

/// Bit-exact float / exact int comparison (so NaN == NaN passes when both sides
/// are the same computation).
#[track_caller]
fn same(interp: Value, expected: Value) {
    match (&interp, &expected) {
        (Value::Float(a), Value::Float(b)) => {
            assert_eq!(a.to_bits(), b.to_bits(), "float {a} vs {b}");
        }
        _ => assert_eq!(interp, expected),
    }
}

/// The sweep of interesting inputs.
const SWEEP: &[f64] = &[
    -1.0e6, -100.0, -3.5, -1.0, -0.0, 0.0, 0.5, 1.0, 3.5, 100.0, 1.0e6,
];

#[test]
fn unary_families_match_compiled() {
    for &v in SWEEP {
        // neg is the IR unary `-a` (not a math:: call); mirror is plain negation.
        let mut g = Graph::empty();
        let n = g.insert("math.neg", NodeUi::default());
        let a = lit_float(&mut g, v);
        same(interp_math(&mut g, n, &[("a", a)]), Value::Float(-v));

        for (id, f) in [
            ("math.abs", math::abs as fn(f64) -> f64),
            ("math.floor", math::floor),
            ("math.ceil", math::ceil),
            ("math.round", math::round),
            ("math.sqrt", math::sqrt),
            ("math.sin", math::sin),
            ("math.cos", math::cos),
        ] {
            let mut g = Graph::empty();
            let n = g.insert(id, NodeUi::default());
            let a = lit_float(&mut g, v);
            same(interp_math(&mut g, n, &[("a", a)]), Value::Float(f(v)));
        }
    }
}

#[test]
fn binary_families_match_compiled() {
    for &a in SWEEP {
        for &b in SWEEP {
            for (id, f) in [
                ("math.min", math::min as fn(f64, f64) -> f64),
                ("math.max", math::max),
                ("math.pow", math::pow),
            ] {
                let mut g = Graph::empty();
                let n = g.insert(id, NodeUi::default());
                let an = lit_float(&mut g, a);
                let bn = lit_float(&mut g, b);
                same(
                    interp_math(&mut g, n, &[("a", an), ("b", bn)]),
                    Value::Float(f(a, b)),
                );
            }
        }
    }
}

#[test]
fn ternary_families_match_compiled() {
    for &x in SWEEP {
        // clamp(x, 0, 10) and an inverted-range clamp(x, 10, 0).
        for (lo, hi) in [(0.0, 10.0), (10.0, 0.0)] {
            let mut g = Graph::empty();
            let n = g.insert("math.clamp", NodeUi::default());
            let xn = lit_float(&mut g, x);
            let lon = lit_float(&mut g, lo);
            let hin = lit_float(&mut g, hi);
            same(
                interp_math(&mut g, n, &[("x", xn), ("min", lon), ("max", hin)]),
                Value::Float(math::clamp(x, lo, hi)),
            );
        }
        // lerp(-2, 8, x) — x doubles as an (unclamped) interpolation factor.
        let mut g = Graph::empty();
        let n = g.insert("math.lerp", NodeUi::default());
        let an = lit_float(&mut g, -2.0);
        let bn = lit_float(&mut g, 8.0);
        let tn = lit_float(&mut g, x);
        same(
            interp_math(&mut g, n, &[("a", an), ("b", bn), ("t", tn)]),
            Value::Float(math::lerp(-2.0, 8.0, x)),
        );
    }
}

#[test]
fn converters_match_compiled() {
    for &v in SWEEP {
        let mut g = Graph::empty();
        let n = g.insert("math.to_int", NodeUi::default());
        let a = lit_float(&mut g, v);
        same(
            interp_math(&mut g, n, &[("a", a)]),
            Value::Int(math::to_int(v)),
        );
    }
    // to_float takes an Int literal.
    for v in [-1_000_000_i64, -5, 0, 7, 1_000_000] {
        let mut g = Graph::empty();
        let n = g.insert("math.to_float", NodeUi::default());
        let a = lit_int(&mut g, v);
        same(
            interp_math(&mut g, n, &[("a", a)]),
            Value::Float(math::to_float(v)),
        );
    }
}

#[test]
fn to_int_saturation_and_nan_match_compiled() {
    // The pathological float→int inputs the mirror must also saturate on.
    for &v in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0e30, -1.0e30] {
        let mut g = Graph::empty();
        let n = g.insert("math.to_int", NodeUi::default());
        let a = lit_float(&mut g, v);
        // NB: a NaN/inf float literal cannot be *generated* as Rust source, but
        // it lowers + interprets fine; this asserts the runtime dispatch only.
        same(
            interp_math(&mut g, n, &[("a", a)]),
            Value::Int(math::to_int(v)),
        );
    }
}

#[test]
fn generated_source_is_the_math_call_the_mirror_targets() {
    // Pin the emitted Rust for a `sqrt` fixture: it must be the `math::sqrt(..)`
    // free call the compiled mirror above resolves through `math_builtins`.
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let sqrt = g.insert("math.sqrt", NodeUi::default());
    let a = lit_float(&mut g, 16.0);
    let ret = g.insert("flow.return", NodeUi::default());
    wire(&mut g, a, "value", sqrt, "a");
    wire(&mut g, sqrt, "out", ret, "value");
    wire(&mut g, bp, EXEC_THEN, ret, "exec");
    let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
    let src = generate_fn(&f).unwrap();
    assert!(src.contains("math::sqrt(16"), "src:\n{src}");

    // And a clamp fixture emits the three-arg call.
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let clamp = g.insert("math.clamp", NodeUi::default());
    let x = lit_float(&mut g, 5.0);
    let lo = lit_float(&mut g, 0.0);
    let hi = lit_float(&mut g, 10.0);
    let ret = g.insert("flow.return", NodeUi::default());
    wire(&mut g, x, "value", clamp, "x");
    wire(&mut g, lo, "value", clamp, "min");
    wire(&mut g, hi, "value", clamp, "max");
    wire(&mut g, clamp, "out", ret, "value");
    wire(&mut g, bp, EXEC_THEN, ret, "exec");
    let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
    let src = generate_fn(&f).unwrap();
    assert!(src.contains("math::clamp("), "src:\n{src}");
}
