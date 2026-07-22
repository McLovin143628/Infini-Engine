//! Interpreter-vs-compiled parity for the control-flow palette (ROADMAP B-P2).
//!
//! The loops (`flow.for`/`flow.while`) and the stateful nodes
//! (`flow.do_once`/`flow.flip_flop`/`flow.gate`) are authored as fixture graphs,
//! lowered, and run through the interpreter; a hand-written **compiled mirror**
//! reproduces the exact semantics (its `nodestate::*` is a `BTreeMap`, exactly
//! how a game-loop shim would back the reserved `__bp_<kind>_<NodeId>` keys).
//! The two must agree — including the runaway-guard trip, where a capped loop
//! must stop at [`LOOP_GUARD_MAX`] and emit the guard's `debug.print` on both
//! paths.
//!
//! Node-id stability note: every fixture graph is built with a fixed node
//! insertion order, so `Graph::insert` hands out deterministic `NodeId`s — a
//! stateful node therefore keys the *same* `nodestate` slot across the separate
//! per-entry graphs a scripted sequence lowers, letting one persistent host map
//! thread state through the whole run.

use std::collections::BTreeMap;

use inf_blueprint::interp::{eval_fn, Host, RunError, Value};
use inf_blueprint::lower::RUNAWAY_MSG;
use inf_blueprint::nodekit::{blueprint_registry, EXEC_THEN};
use inf_blueprint::{lower_graph, BlueprintFn, LOOP_GUARD_MAX};
use inf_graph::{Graph, Link, NodeId, NodeUi, ParamValue};
use std::collections::HashMap;

/// A host backing `vars::*`, `nodestate::*`, and `debug.print` over `BTreeMap`s —
/// the interpreter side of the parity check. State persists across `run` calls,
/// so a scripted invocation sequence threads state exactly like the game loop.
#[derive(Default)]
struct FlowHost {
    vars: BTreeMap<String, Value>,
    state: BTreeMap<String, Value>,
    logs: Vec<String>,
}

impl Host for FlowHost {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        match (
            path.first().map(String::as_str),
            path.get(1).map(String::as_str),
        ) {
            (Some("vars"), Some("get")) => Ok(self
                .vars
                .get(args[0].as_str().unwrap())
                .cloned()
                .unwrap_or(Value::Int(0))),
            (Some("vars"), Some("set")) => {
                self.vars
                    .insert(args[0].as_str().unwrap().to_string(), args[1].clone());
                Ok(Value::Unit)
            }
            (Some("nodestate"), Some("get_or")) => Ok(self
                .state
                .get(args[0].as_str().unwrap())
                .cloned()
                .unwrap_or_else(|| args[1].clone())),
            (Some("nodestate"), Some("set")) => {
                self.state
                    .insert(args[0].as_str().unwrap().to_string(), args[1].clone());
                Ok(Value::Unit)
            }
            (Some("debug"), Some("print")) => {
                self.logs.push(args[0].as_str().unwrap().to_string());
                Ok(Value::Unit)
            }
            _ => Ok(Value::Unit),
        }
    }
}

impl FlowHost {
    fn run(&mut self, f: &BlueprintFn) {
        eval_fn(f, &HashMap::new(), self).unwrap();
    }
    fn int(&self, name: &str) -> i64 {
        self.vars
            .get(name)
            .and_then(|v| v.as_int().ok())
            .unwrap_or(0)
    }
}

fn wire(g: &mut Graph, from: NodeId, fp: &str, to: NodeId, tp: &str) {
    g.links.push(Link {
        from,
        from_port: fp.into(),
        to,
        to_port: tp.into(),
    });
}

fn named(g: &mut Graph, ty: &str, key: &str, val: &str) -> NodeId {
    let n = g.insert(ty, NodeUi::default());
    g.node_mut(n)
        .unwrap()
        .params
        .insert(key.into(), ParamValue::Text(val.into()));
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

/// A `var.set("name", var.get("name") + 1)` incrementer, wired to fire from
/// `(src, port)`. Returns the created `var.set` node.
fn incr_var(g: &mut Graph, name: &str, src: NodeId, port: &str) {
    let get = named(g, "var.get", "name", name);
    let one = lit_int(g, 1);
    let add = g.insert("math.add", NodeUi::default());
    let set = named(g, "var.set", "name", name);
    wire(g, get, "value", add, "a");
    wire(g, one, "value", add, "b");
    wire(g, add, "out", set, "value");
    wire(g, src, port, set, "exec");
}

// ── loops ────────────────────────────────────────────────────────────────────

#[test]
fn for_loop_sum_matches_compiled() {
    for n in [0_i64, 1, 5, 20] {
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let first = lit_int(&mut g, 0);
        let last = lit_int(&mut g, n);
        let forn = g.insert("flow.for", NodeUi::default());
        let getsum = named(&mut g, "var.get", "name", "sum");
        let add = g.insert("math.add", NodeUi::default());
        let setsum = named(&mut g, "var.set", "name", "sum");
        wire(&mut g, first, "value", forn, "first");
        wire(&mut g, last, "value", forn, "last");
        wire(&mut g, getsum, "value", add, "a");
        wire(&mut g, forn, "index", add, "b");
        wire(&mut g, add, "out", setsum, "value");
        wire(&mut g, bp, EXEC_THEN, forn, "exec");
        wire(&mut g, forn, "loop_body", setsum, "exec");
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();

        let mut host = FlowHost::default();
        host.run(&f);

        // compiled mirror
        let mut sum = 0_i64;
        for i in 0..=n {
            sum += i;
        }
        assert_eq!(host.int("sum"), sum, "for-loop sum 0..={n}");
    }
}

#[test]
fn while_countdown_matches_compiled() {
    for start in [0_i64, 1, 7, 50] {
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let getn = named(&mut g, "var.get", "name", "n");
        let zero = lit_int(&mut g, 0);
        let gt = g.insert("cmp.gt", NodeUi::default());
        let wh = g.insert("flow.while", NodeUi::default());
        let getn2 = named(&mut g, "var.get", "name", "n");
        let one = lit_int(&mut g, 1);
        let sub = g.insert("math.sub", NodeUi::default());
        let setn = named(&mut g, "var.set", "name", "n");
        wire(&mut g, getn, "value", gt, "a");
        wire(&mut g, zero, "value", gt, "b");
        wire(&mut g, gt, "out", wh, "condition");
        wire(&mut g, getn2, "value", sub, "a");
        wire(&mut g, one, "value", sub, "b");
        wire(&mut g, sub, "out", setn, "value");
        wire(&mut g, bp, EXEC_THEN, wh, "exec");
        wire(&mut g, wh, "loop_body", setn, "exec");
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();

        let mut host = FlowHost::default();
        host.vars.insert("n".into(), Value::Int(start));
        host.run(&f);

        // compiled mirror
        let mut n = start;
        while n > 0 {
            n -= 1;
        }
        assert_eq!(host.int("n"), n);
        assert!(host.logs.is_empty(), "no runaway for a terminating loop");
    }
}

#[test]
fn while_runaway_trips_the_guard_on_both_paths() {
    // condition = literal true → the loop only ever stops on the guard.
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default());
    let t = g.insert("lit.bool", NodeUi::default());
    g.node_mut(t)
        .unwrap()
        .params
        .insert("value".into(), ParamValue::Bool(true));
    let wh = g.insert("flow.while", NodeUi::default());
    wire(&mut g, t, "value", wh, "condition");
    incr_var(&mut g, "count", wh, "loop_body");
    wire(&mut g, bp, EXEC_THEN, wh, "exec");
    let f = lower_graph(&g, &reg).unwrap().pop().unwrap();

    let mut host = FlowHost::default();
    host.vars.insert("count".into(), Value::Int(0));
    host.run(&f);

    // compiled mirror: the same cap, the same report. `user_cond` stands in for
    // the always-true literal condition (kept as a binding so the mirror mirrors
    // the lowered `user_cond && counter < CAP` guard).
    let mut count = 0_i64;
    let mut counter = 0_i64;
    let mut logs: Vec<String> = Vec::new();
    let user_cond = true;
    while user_cond && counter < LOOP_GUARD_MAX {
        count += 1;
        counter += 1;
    }
    if counter >= LOOP_GUARD_MAX {
        logs.push(RUNAWAY_MSG.to_string());
    }

    assert_eq!(host.int("count"), count);
    assert_eq!(host.int("count"), LOOP_GUARD_MAX);
    assert_eq!(host.logs, logs);
    assert_eq!(host.logs, vec![RUNAWAY_MSG.to_string()]);
}

// ── stateful nodes ────────────────────────────────────────────────────────────

/// Compiled-mirror nodestate over a `BTreeMap`, with the exact key format the
/// lowerer emits (`__bp_<kind>_<NodeId>`, `NodeId` = `N<n>`).
#[derive(Default)]
struct MirrorState {
    map: BTreeMap<String, bool>,
}
impl MirrorState {
    fn get_or(&self, key: &str, default: bool) -> bool {
        self.map.get(key).copied().unwrap_or(default)
    }
    fn set(&mut self, key: &str, v: bool) {
        self.map.insert(key.into(), v);
    }
}

#[test]
fn do_once_with_reset_matches_compiled() {
    // Two fixed-shape graphs so the do_once node keeps the SAME id (⇒ same
    // state key) across the scripted exec/reset invocations.
    fn do_once_graph(entry: &str) -> (BlueprintFn, NodeId) {
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default()); // N1
        let once = g.insert("flow.do_once", NodeUi::default()); // N2
        incr_var(&mut g, "hits", once, EXEC_THEN);
        wire(&mut g, bp, EXEC_THEN, once, entry);
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        (f, once)
    }
    let (exec_fn, node) = do_once_graph("exec");
    let (reset_fn, _) = do_once_graph("reset");
    let key = format!("__bp_once_{node}");

    // Script: fire, fire, fire, reset, fire, fire (10-ish invocations).
    let script = [
        "exec", "exec", "exec", "reset", "exec", "exec", "reset", "reset", "exec", "exec",
    ];
    let mut host = FlowHost::default();
    let mut m_state = MirrorState::default();
    let mut m_hits = 0_i64;
    for step in script {
        if step == "exec" {
            host.run(&exec_fn);
            // mirror
            if !m_state.get_or(&key, false) {
                m_state.set(&key, true);
                m_hits += 1;
            }
        } else {
            host.run(&reset_fn);
            m_state.set(&key, false);
        }
    }
    assert_eq!(host.int("hits"), m_hits);
    // fires on first exec, again after each reset re-arms it → 3 total.
    assert_eq!(host.int("hits"), 3);
}

#[test]
fn flip_flop_matches_compiled() {
    let reg = blueprint_registry();
    let mut g = Graph::empty();
    let bp = g.insert("event.begin_play", NodeUi::default()); // N1
    let ff = g.insert("flow.flip_flop", NodeUi::default()); // N2
    incr_var(&mut g, "a_count", ff, "a");
    incr_var(&mut g, "b_count", ff, "b");
    wire(&mut g, bp, EXEC_THEN, ff, "exec");
    let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
    let key = format!("__bp_flip_{ff}");

    let mut host = FlowHost::default();
    let mut m_state = MirrorState::default();
    let (mut m_a, mut m_b) = (0_i64, 0_i64);
    for _ in 0..10 {
        host.run(&f);
        // mirror: read is_a (default true), branch, then toggle.
        let is_a = m_state.get_or(&key, true);
        m_state.set(&key, !is_a);
        if is_a {
            m_a += 1;
        } else {
            m_b += 1;
        }
    }
    assert_eq!(host.int("a_count"), m_a);
    assert_eq!(host.int("b_count"), m_b);
    // starts on A, strictly alternates over 10 runs → 5/5.
    assert_eq!((host.int("a_count"), host.int("b_count")), (5, 5));
}

#[test]
fn gate_scripted_sequence_matches_compiled() {
    // Fixed-shape per-entry graphs → the gate keeps a stable id / state key.
    fn gate_graph(entry: &str) -> (BlueprintFn, NodeId) {
        let reg = blueprint_registry();
        let mut g = Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default()); // N1
        let gate = g.insert("flow.gate", NodeUi::default()); // N2
        incr_var(&mut g, "passes", gate, "exit");
        wire(&mut g, bp, EXEC_THEN, gate, entry);
        let f = lower_graph(&g, &reg).unwrap().pop().unwrap();
        (f, gate)
    }
    let node = gate_graph("enter").1;
    let key = format!("__bp_gate_{node}");
    let start_open = true; // the node's start_open param default.

    let script = [
        "enter", "enter", "close", "enter", "enter", "open", "enter", "toggle", "enter", "enter",
    ];
    let mut host = FlowHost::default();
    let mut m_state = MirrorState::default();
    let mut m_passes = 0_i64;
    for step in script {
        let (f, _) = gate_graph(step);
        host.run(&f);
        // mirror
        match step {
            "open" => m_state.set(&key, true),
            "close" => m_state.set(&key, false),
            "toggle" => {
                let v = m_state.get_or(&key, start_open);
                m_state.set(&key, !v);
            }
            _ => {
                if m_state.get_or(&key, start_open) {
                    m_passes += 1;
                }
            }
        }
    }
    assert_eq!(host.int("passes"), m_passes);
    // enter,enter (open) → 2; close; enter,enter (closed) → 0; open; enter → 1;
    // toggle (→closed); enter,enter (closed) → 0. Total = 3.
    assert_eq!(host.int("passes"), 3);
}
