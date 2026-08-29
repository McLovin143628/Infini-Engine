//! # THE CROWN GATE — transpile an InfiniScript, **compile it, run it**, and
//! diff the trace against the interpreter byte for byte.
//!
//! This is the gate the repository has claimed since P6 and never had.
//! `infiniscript-direction.md` §2 states the bound in the repository's own
//! words: *"**No test in this repository compiles the transpiler's output and
//! runs it.**"* `parity.rs`'s module doc calls itself *"the CI-cheap half of the
//! parity story (no runtime `cargo build`)"*, and the four parity families
//! (`parity` / `flow_parity` / `math_parity` / `coyote_parity`) each run the
//! interpreter against a **hand-written Rust mirror** of what `generate_fn`
//! emits, string-pinned so the two cannot drift.
//!
//! A hand-written mirror proves the two *programs* agree. It cannot prove the
//! **generator** is right, because a defect in `emit` moves the generator's
//! output and leaves the mirror where it was. That is the hole this closes.
//!
//! ## The mechanism, and why it is `rustc` and not `cargo`
//!
//! The generated program has **zero dependencies**: the host shims are emitted
//! beside it, so one `rustc` invocation compiles and links the whole thing.
//! Measured on this machine at **~1 s**, against the `inf-hotreload` fixture
//! path's `cargo build` — which needs a dedicated `CARGO_TARGET_DIR`, an
//! exclusive build-directory lock and a process-private artifact stash precisely
//! *because* concurrent test processes race over one. There is no lock to take
//! here, no workspace to resolve and no manifest to write. `rustc` is present by
//! construction: the test running this was compiled by it.
//!
//! ## What the shims are, and what they are not
//!
//! They are the **engine**. `parity.rs`'s compiled side hand-writes both the
//! engine *and the handler body*; here the body is the transpiler's real output
//! and only the engine is written down — the same engine, in the same shape, as
//! the interpreter's `Host`. Both sides' `math.*` is silent for the same reason
//! it is silent in the IR: `dispatch_math` is **hostless**, so a math builtin is
//! not a host call on either side.
//!
//! ## The bound this gate does NOT cover, named rather than discovered
//!
//! **Transcendentals.** `math.sin` / `math.cos` route to
//! `inf_math::portable::psin64` / `pcos64`, and a zero-dependency shim cannot
//! call them without becoming a *second implementation* of a bit-exact
//! polynomial — which is the one thing worse than no coverage. They are covered
//! where they already are: `portable_math_law.rs` proves the interpreter's
//! routing bit for bit, and `math_parity.rs` proves the two sides agree **by
//! construction**, because both bottom out in `inf_blueprint::math_builtins`.
//! The exact-IEEE builtins (`sqrt`, `abs`, `floor`, `min`, `max`) *are* here,
//! and each shim calls **`std`'s own function, the one `math_builtins` calls**
//! — never a hand-written equivalent.
//!
//! That last clause is the SCRIPT1b audit's correction, and it is the same law
//! the transcendental paragraph above states, met one level down. The wave read
//! "exact-IEEE" as "one machine instruction, so any spelling will do" and wrote
//! `min`/`max` as `if a < b { a } else { b }`. They are not one instruction and
//! any spelling will not do: `math_builtins::{min,max}` are documented
//! **NaN-absorbing** (they delegate to `f64::min`/`f64::max`, with
//! `min_max_nan_absorbing` asserting it), and the comparison form propagates
//! the NaN instead. `math.max(2.0, NaN)` was **`2.0` interpreted and `NaN`
//! compiled**, and no arm could see it because nothing in the fixture's trace
//! was ever not a number. The fixture now carries a NaN and both shims delegate.
//!
//! **Monomorphic `vars::get`.** See
//! [`a_bool_member_variable_does_not_compile`] — a measured refusal, not a
//! guess.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use inf_blueprint::interp::{Host, RunError, Value};
use inf_blueprint::{BlueprintClass, EventKind};
use inf_transpile::{generate_file, BlueprintFile, FileEntry};

/// **The fixture.** One script, chosen to reach every construct class the
/// generator has an opinion about: member variables read and written, a local, a
/// `while` with its counter guard, a `for` with its index, a three-way `if`, a
/// short-circuiting `and` **whose right operand is a host call**, unary minus, a
/// `math.*` builtin in value position, a comparison, a **parenthesised**
/// sub-expression whose precedence a re-associating generator would lose, and
/// two events with different parameter types.
///
/// It is a `const` rather than a file because the gate's whole subject is the
/// text — a reviewer reading this arm must see the program it compiles.
const SCRIPT: &str = r#"
actor "CrownGate"

var angle_deg: float = 0.0
var speed: float = 90.0
var hits: float = 0.0

on begin_play()
  debug.print("armed")
  angle_deg = 0.0
  -- A NaN through both NaN-ABSORBING builtins, and in the argument position
  -- that separates `f64::min`/`f64::max` from a comparison written by hand.
  -- Without this leg the gate certified a shim that answered NaN where the
  -- interpreter answers 2.0, because nothing in the trace was ever not a
  -- number (SCRIPT1b audit).
  local nan = 0.0 / 0.0
  engine.set_rotation(math.max(2.0, nan))
  engine.set_rotation(math.min(1.0, nan))
end

on tick(dt)
  -- PARENTHESISED on purpose: `Binary(Mul, Binary(Add, .., ..), ..)` is the
  -- shape a generator that re-associated by precedence would get wrong, which
  -- is the Spike B law (`emit` builds syn ASTs, never parsed token streams) and
  -- the SCRIPT1a emitter finding met from the transpiler's side.
  local step = (speed + 10.0) * dt
  angle_deg = angle_deg + step
  while angle_deg > 360.0 do
    angle_deg = angle_deg - 360.0
  end
  for i = 1, 3 do
    hits = hits + 1.0
  end
  if angle_deg > 180.0 then
    engine.set_rotation(math.sqrt(angle_deg))
  elseif angle_deg > 90.0 then
    engine.set_rotation(-angle_deg)
  else
    engine.set_rotation(math.floor(angle_deg) + math.abs(-1.0))
  end
end

on input "fire"(pressed)
  -- The right operand is a HOST CALL, so an interpreter that did not
  -- short-circuit would make one more `vars::get` than the compiled build.
  if pressed and speed > 1000.0 then
    debug.print("impossible")
  end
  if pressed then
    hits = hits + 10.0
    engine.set_rotation(math.min(hits, 99.0))
  end
end
"#;

/// An argument a driver step passes to a handler.
#[derive(Clone, Copy, Debug)]
enum Arg {
    F(f64),
    B(bool),
}

/// **The event sequence both sides run**, in order. One list, so the two hosts
/// cannot be driven differently — the failure mode a gate comparing two
/// separately-written drivers would have.
fn steps() -> Vec<(EventKind, Vec<Arg>)> {
    let mut v = vec![(EventKind::BeginPlay, vec![])];
    // Deliberately irregular `dt`s, and one big enough to wrap the `while`
    // several times — a fixed `dt` would leave the loop guard unexercised.
    for dt in [0.016_f64, 0.5, 2.0, 4.5, 0.016, 1.0] {
        v.push((EventKind::Tick, vec![Arg::F(dt)]));
    }
    v.push((EventKind::Input("fire".into()), vec![Arg::B(true)]));
    v.push((EventKind::Tick, vec![Arg::F(0.25)]));
    v.push((EventKind::Input("fire".into()), vec![Arg::B(false)]));
    v.push((EventKind::Tick, vec![Arg::F(0.25)]));
    v
}

// ── the trace format, spelled once ──────────────────────────────────────────
//
// Floats print as their **bit pattern**, never as decimal text: a formatter is
// a second thing that can agree, and the whole claim here is byte identity.

fn fmt_f(v: f64) -> String {
    format!("f:{:016x}", v.to_bits())
}

fn fmt_value(v: &Value) -> String {
    match v {
        Value::Float(x) => fmt_f(*x),
        Value::Int(i) => format!("i:{i}"),
        Value::Bool(b) => format!("b:{b}"),
        Value::Str(s) => format!("s:{s}"),
        Value::Unit => "u".to_string(),
    }
}

fn fmt_arg(a: Arg) -> String {
    match a {
        Arg::F(x) => fmt_f(x),
        Arg::B(b) => format!("b:{b}"),
    }
}

/// The Rust literal a driver step passes on the compiled side.
fn rust_arg(a: Arg) -> String {
    match a {
        // From the bit pattern, so the driver cannot introduce a decimal
        // rounding difference the trace would then blame on the transpiler.
        Arg::F(x) => format!("f64::from_bits({:#018x}u64)", x.to_bits()),
        Arg::B(b) => format!("{b}"),
    }
}

// ── the interpreted side ────────────────────────────────────────────────────

/// A recording engine: member variables, three verbs, and every call written
/// down in the shared format.
struct Recorder {
    vars: BTreeMap<String, Value>,
    log: String,
}

impl Host for Recorder {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        let key = path.join("::");
        let ret = match key.as_str() {
            "vars::get" => {
                let n = args[0]
                    .as_str()
                    .map_err(|_| RunError::Type("name".into()))?;
                self.vars
                    .get(n)
                    .cloned()
                    .ok_or_else(|| RunError::Host(key.clone(), format!("no var `{n}`")))?
            }
            "vars::set" => {
                let n = args[0]
                    .as_str()
                    .map_err(|_| RunError::Type("name".into()))?;
                self.vars.insert(n.to_string(), args[1].clone());
                Value::Unit
            }
            "engine::set_rotation" | "debug::print" => Value::Unit,
            other => return Err(RunError::NoSuchHostFn(other.to_string())),
        };
        let argv: Vec<String> = args.iter().map(fmt_value).collect();
        self.log.push_str(&format!(
            "{key}({}) = {}\n",
            argv.join(", "),
            fmt_value(&ret)
        ));
        Ok(ret)
    }
}

fn interpreted_trace(class: &BlueprintClass) -> String {
    let mut host = Recorder {
        vars: class
            .variables
            .iter()
            .map(|v| (v.name.clone(), v.default_value()))
            .collect(),
        log: String::new(),
    };
    let mut out = String::new();
    for (i, (event, args)) in steps().into_iter().enumerate() {
        let binding = class
            .handler(&event)
            .unwrap_or_else(|| panic!("the fixture must handle {:?}", event));
        let mut argmap: HashMap<String, Value> = HashMap::new();
        for (p, a) in binding.body.params.iter().zip(args.iter()) {
            argmap.insert(
                p.name.clone(),
                match a {
                    Arg::F(x) => Value::Float(*x),
                    Arg::B(b) => Value::Bool(*b),
                },
            );
        }
        let argtxt: Vec<String> = args.iter().copied().map(fmt_arg).collect();
        out.push_str(&format!(
            "-- {i} {}({})\n",
            binding.body.name,
            argtxt.join(", ")
        ));
        host.log.clear();
        let ret = inf_blueprint::eval_fn(&binding.body, &argmap, &mut host)
            .unwrap_or_else(|e| panic!("the interpreter refused step {i}: {e}"));
        out.push_str(&host.log);
        out.push_str(&format!("= {}\n", fmt_value(&ret)));
        for (k, v) in &host.vars {
            out.push_str(&format!("state {k} {}\n", fmt_value(v)));
        }
    }
    out
}

// ── the compiled side ───────────────────────────────────────────────────────

/// The host shims: the engine, written once, in the shape the interpreter's
/// `Recorder` has. Everything here that *logs* is a host call in the IR;
/// everything that does not (`math::*`) is hostless in the IR too.
const SHIMS: &str = r#"
use std::cell::RefCell;
use std::collections::BTreeMap;

#[derive(Clone, PartialEq)]
pub enum V { F(f64), I(i64), B(bool), S(String), U }

pub fn fmt(v: &V) -> String {
    match v {
        V::F(x) => format!("f:{:016x}", x.to_bits()),
        V::I(i) => format!("i:{i}"),
        V::B(b) => format!("b:{b}"),
        V::S(s) => format!("s:{s}"),
        V::U => "u".to_string(),
    }
}

thread_local! {
    pub static LOG: RefCell<String> = RefCell::new(String::new());
    pub static VARS: RefCell<BTreeMap<String, V>> = RefCell::new(BTreeMap::new());
}

pub fn record(path: &str, args: &[V], ret: &V) {
    let a: Vec<String> = args.iter().map(fmt).collect();
    LOG.with(|l| l.borrow_mut().push_str(&format!("{path}({}) = {}\n", a.join(", "), fmt(ret))));
}

pub mod vars {
    use super::{record, V, VARS};
    pub fn get(name: &str) -> f64 {
        let v = VARS.with(|m| m.borrow().get(name).cloned())
            .unwrap_or_else(|| panic!("no var `{name}`"));
        record("vars::get", &[V::S(name.to_string())], &v);
        match v { V::F(x) => x, _ => panic!("var `{name}` is not a float") }
    }
    pub fn set(name: &str, value: f64) {
        record("vars::set", &[V::S(name.to_string()), V::F(value)], &V::U);
        VARS.with(|m| m.borrow_mut().insert(name.to_string(), V::F(value)));
    }
}

pub mod engine {
    use super::{record, V};
    pub fn set_rotation(angle: f64) {
        record("engine::set_rotation", &[V::F(angle)], &V::U);
    }
}

pub mod debug {
    use super::{record, V};
    pub fn print(message: &str) {
        record("debug::print", &[V::S(message.to_string())], &V::U);
    }
}

// HOSTLESS, exactly as `dispatch_math` is hostless in the interpreter: these do
// not record, because a math builtin is not a host call on either side. Only
// the exact-IEEE operations are here — see this file's module doc for why
// `sin`/`cos` are deliberately out of the compiled leg.
//
// EVERY ONE OF THESE IS `std`'s OWN FUNCTION, spelled the way
// `inf_blueprint::math_builtins` spells it, and that is the SCRIPT1b audit's
// finding rather than a tidy-up. `min`/`max` used to read `if a < b { a } else
// { b }`, which is a SECOND IMPLEMENTATION — the exact thing this file's module
// doc refuses for `sin`/`cos` — and it disagrees with the engine on the input
// the engine has a contract and a test about: `math_builtins::{min,max}` are
// documented **NaN-absorbing** (`min_max_nan_absorbing`), while the comparison
// form propagates the NaN. Measured: `math.max(2.0, NaN)` is `2.0` in the
// interpreter and `NaN` under the old shim.
pub mod math {
    pub fn sqrt(x: f64) -> f64 { x.sqrt() }
    pub fn abs(x: f64) -> f64 { x.abs() }
    pub fn floor(x: f64) -> f64 { x.floor() }
    pub fn min(a: f64, b: f64) -> f64 { a.min(b) }
    pub fn max(a: f64, b: f64) -> f64 { a.max(b) }
}
"#;

/// Strip the `#[infinity::blueprint(id = "…")]` marker the transpiler puts on
/// every generated fn.
///
/// **`inf_packager::mods` does the same thing, and its comment says the marker's
/// proc-macro "ships with the engine runtime". It does not** — see
/// [`the_generated_marker_has_no_macro_behind_it`], which measures it. Stripping
/// is therefore not a mod-crate concession; it is the only way generated Rust
/// compiles anywhere.
fn strip_marker(rust: &str) -> String {
    rust.lines()
        .filter(|l| !l.trim_start().starts_with("#[infinity::blueprint"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The whole compiled program: the shims, the transpiler's **real output**, and
/// a driver over the same step list the interpreter runs.
fn compiled_program(class: &BlueprintClass, keep_marker: bool) -> String {
    let file = BlueprintFile {
        entries: class
            .events
            .iter()
            .map(|b| FileEntry::Blueprint(b.body.clone()))
            .collect(),
    };
    let generated = generate_file(&file).expect("the transpiler renders the fixture");
    let body = if keep_marker {
        generated
    } else {
        strip_marker(&generated)
    };

    let mut src = String::new();
    src.push_str("#![allow(unused, clippy::all)]\n");
    src.push_str(SHIMS);
    src.push_str("\n// ── the transpiler's OUTPUT, verbatim ──\n");
    src.push_str(&body);
    src.push_str("\n\nfn main() {\n");
    for v in &class.variables {
        let d = match v.default_value() {
            Value::Float(x) => format!("V::F(f64::from_bits({:#018x}u64))", x.to_bits()),
            Value::Int(i) => format!("V::I({i})"),
            Value::Bool(b) => format!("V::B({b})"),
            Value::Str(s) => format!("V::S({s:?}.to_string())"),
            Value::Unit => "V::U".to_string(),
        };
        src.push_str(&format!(
            "    VARS.with(|m| m.borrow_mut().insert({:?}.to_string(), {d}));\n",
            v.name
        ));
    }
    src.push_str("    let mut out = String::new();\n");
    for (i, (event, args)) in steps().into_iter().enumerate() {
        // A fixture that does not declare an event is simply not driven with it
        // — the refusal arms below carry one handler each, and requiring them to
        // carry three would make the thing they measure harder to read.
        let Some(binding) = class.handler(&event) else {
            continue;
        };
        let argtxt: Vec<String> = args.iter().copied().map(fmt_arg).collect();
        src.push_str(&format!(
            "    out.push_str(&format!(\"-- {i} {}({})\\n\"));\n",
            binding.body.name,
            argtxt.join(", ")
        ));
        src.push_str("    LOG.with(|l| l.borrow_mut().clear());\n");
        let call_args: Vec<String> = args.iter().copied().map(rust_arg).collect();
        src.push_str(&format!(
            "    let ret = {}({});\n",
            binding.body.name,
            call_args.join(", ")
        ));
        src.push_str("    out.push_str(&LOG.with(|l| l.borrow().clone()));\n");
        // **The `= …` line is a UNIT line, and that is asserted rather than
        // assumed** (SCRIPT1b audit). The interpreted side prints whatever
        // `eval_fn` returned; this side prints `V::U` as a literal, because a
        // generated `fn tick(dt: f64)` returns `()` and there is nothing to
        // read. Two sides where one of them is a constant cannot falsify, so the
        // constant is legal only while every driven handler is `Ty::Unit`, and
        // that is what the assertion below holds. The `let _: () = ret;` beneath
        // it is the same claim handed to `rustc`: if the generator ever emits a
        // handler returning a value, the compiled program stops building here
        // rather than silently comparing against `u`.
        assert_eq!(
            binding.body.ret,
            inf_blueprint::Ty::Unit,
            "handler `{}` returns {:?}: the compiled side prints its `=` line as \
             a literal Unit, so a non-Unit return would be compared against a \
             constant. Read `ret` here before driving such a handler.",
            binding.body.name,
            binding.body.ret
        );
        src.push_str("    out.push_str(&format!(\"= {}\\n\", fmt(&V::U)));\n");
        src.push_str("    let _: () = ret;\n");
        src.push_str(
            "    VARS.with(|m| for (k, v) in m.borrow().iter() { \
             out.push_str(&format!(\"state {k} {}\\n\", fmt(v))); });\n",
        );
    }
    src.push_str("    print!(\"{out}\");\n}\n");
    src
}

/// Where the gate builds. Under `target/` so `cargo clean` reaches it, and
/// per-process so two test binaries cannot write one another's files.
///
/// # …and it sweeps its own leavings (SCRIPT1b audit)
///
/// A directory per process id, never removed, is a leak with a slow fuse: the
/// audit found **thirteen** of them at 19 MB after one wave's testing, and this
/// repository has a law about `target/` growth paid for with three disk-full
/// incidents. A test cannot run teardown after itself (the three arms here share
/// one process and one directory), so the sweep runs at the *front*: any sibling
/// older than [`STALE`] belongs to a process that has long since exited.
///
/// The age bound rather than a liveness check on purpose — asking the operating
/// system whether a pid is alive is three platforms of code to delete a
/// megabyte, and a pid is reused. An hour is longer than any run of this gate
/// (measured in tenths of a second) by four orders of magnitude, so a
/// concurrently-running sibling is never in range.
fn build_dir() -> PathBuf {
    /// How old a sibling build directory must be before it is swept.
    const STALE: Duration = Duration::from_secs(3600);

    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("target")
        });
    let root = base.join("crown-parity");
    let mine = root.join(format!("{}", std::process::id()));

    // Best effort throughout: a sweep that failed is a directory left behind,
    // never a red gate about somebody else's file lock.
    if let Ok(entries) = std::fs::read_dir(&root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path == mine || !path.is_dir() {
                continue;
            }
            let old = entry
                .metadata()
                .and_then(|m| m.modified())
                .map(|t| t.elapsed().map(|e| e > STALE).unwrap_or(false))
                .unwrap_or(false);
            if old {
                let _ = std::fs::remove_dir_all(&path);
            }
        }
    }

    std::fs::create_dir_all(&mine).expect("create the crown-gate build dir");
    mine
}

struct Built {
    stdout: String,
    compile: Duration,
    run: Duration,
}

/// Compile `src` with `rustc` and run it. `Err` carries rustc's own stderr,
/// which is what makes a *refusal* measurable rather than an assertion.
fn rustc_and_run(name: &str, src: &str) -> Result<Built, String> {
    let dir = build_dir();
    let rs = dir.join(format!("{name}.rs"));
    std::fs::write(&rs, src).expect("write the generated program");
    let exe = dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());

    let t0 = Instant::now();
    let out = Command::new(&rustc)
        .arg("--edition=2021")
        .arg("-o")
        .arg(&exe)
        .arg(&rs)
        .output()
        .map_err(|e| format!("could not run `{rustc}`: {e}"))?;
    let compile = t0.elapsed();
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }

    let t1 = Instant::now();
    let run = Command::new(&exe)
        .output()
        .map_err(|e| format!("could not run the built program: {e}"))?;
    let run_t = t1.elapsed();
    assert!(
        run.status.success(),
        "the compiled program failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    Ok(Built {
        stdout: String::from_utf8_lossy(&run.stdout).to_string(),
        compile,
        run: run_t,
    })
}

/// **A LOAD-class budget**, never a frame one.
///
/// This is a one-shot cold `rustc` invocation, so it is measured once and held
/// against a load ceiling, exactly as `inf_player::budget::LOAD_BUDGET_MS`'s
/// doc requires of *"any arm that times a one-shot world build"*. The number
/// lives here rather than beside that one because the §8 rule is that a ratchet
/// belongs where the gate reading it lives, and this is the only gate that reads
/// it.
///
/// Measured on this machine at **~1 s** for compile+link of a zero-dependency
/// program. 60 s is sixty times that: a shared CI runner under a cold toolchain
/// cache is a different machine, and a budget nobody can meet is a budget
/// somebody disables. It may only ever go **down**.
const SCRIPT_COMPILE_BUDGET_MS: f64 = 60_000.0;

/// # THE CROWN GATE
///
/// Transpile a `.infini`, **compile it, run it**, and require the trace to be
/// byte-identical to the interpreter's over the same event sequence: the host
/// calls in order, every argument as a bit pattern, every handler's return, and
/// the member-variable state after each event.
#[test]
fn the_crown_gate_interpreted_equals_compiled() {
    let (class, warnings) =
        inf_script::compile(SCRIPT, "script:crown").expect("the fixture parses");
    assert!(warnings.is_empty(), "{warnings:?}");

    let interpreted = interpreted_trace(&class);
    let program = compiled_program(&class, false);
    let built = rustc_and_run("crown", &program).unwrap_or_else(|e| {
        panic!("the transpiler's output does not compile:\n{e}\n--- program ---\n{program}")
    });

    println!(
        "crown gate: rustc {:.0} ms (budget {SCRIPT_COMPILE_BUDGET_MS:.0} ms, LOAD class), run \
         {:.1} ms, program {} bytes, trace {} lines",
        built.compile.as_secs_f64() * 1000.0,
        built.run.as_secs_f64() * 1000.0,
        program.len(),
        interpreted.lines().count()
    );

    // **Anti-vacuity, before the comparison.** Two empty traces are equal.
    let calls = interpreted.lines().filter(|l| l.contains(") = ")).count();
    assert!(
        calls >= 60,
        "only {calls} host calls in the trace — the fixture stopped exercising the surface"
    );
    assert!(
        interpreted.lines().filter(|l| l.starts_with("-- ")).count() == steps().len(),
        "every step must appear"
    );

    if interpreted != built.stdout {
        // A divergence point is most of the diagnosis (the phase22 gate's rule),
        // so report the first differing line rather than two whole traces.
        for (i, (a, b)) in interpreted.lines().zip(built.stdout.lines()).enumerate() {
            assert_eq!(a, b, "traces diverge at line {i}");
        }
        assert_eq!(
            interpreted.lines().count(),
            built.stdout.lines().count(),
            "one trace is longer than the other"
        );
    }
    assert_eq!(
        interpreted, built.stdout,
        "interpreted != compiled over the same events"
    );

    assert!(
        built.compile.as_secs_f64() * 1000.0 < SCRIPT_COMPILE_BUDGET_MS,
        "compiling the transpiler's output took {:.0} ms, over the {SCRIPT_COMPILE_BUDGET_MS:.0} \
         ms LOAD budget (§8: investigate, never raise it)",
        built.compile.as_secs_f64() * 1000.0
    );
}

/// **The generated marker has no macro behind it** — measured, not asserted.
///
/// `inf_transpile::emit` writes `#[infinity::blueprint(id = "…")]` onto every
/// generated fn, and `inf_packager::mods` strips it with a comment saying the
/// proc-macro *"ships with the engine runtime, not a mod"*. There is no crate,
/// module or macro named `infinity` anywhere in this workspace. So the Code
/// tab's output — `<project>/src/blueprints/<Class>_<guid8>.rs`, which the
/// scaffolded `lib.rs` declares and `cargo build` compiles — **does not
/// compile**, and nothing noticed because nothing ever compiled it.
///
/// This arm compiles the same program *with* the marker and requires `rustc` to
/// refuse it, naming `infinity`. It is a **tripwire**: the day the marker gains
/// a macro, or the emitter stops writing it, this goes red and the carried
/// ledger item is retired.
#[test]
fn the_generated_marker_has_no_macro_behind_it() {
    let (class, _) = inf_script::compile(SCRIPT, "script:crown").expect("parses");
    let program = compiled_program(&class, true);
    assert!(
        program.contains("#[infinity::blueprint"),
        "the transpiler stopped emitting the marker — retire this arm and the ledger item"
    );
    match rustc_and_run("marker", &program) {
        Ok(_) => panic!(
            "the marker COMPILED — an `infinity` macro now exists (or the attribute became \
             inert). Retire the strip in `inf_packager::mods` and this arm together."
        ),
        Err(stderr) => {
            println!(
                "rustc on the marker: {}",
                stderr.lines().next().unwrap_or("")
            );
            assert!(
                stderr.contains("infinity"),
                "expected rustc to name the missing `infinity` path:\n{stderr}"
            );
        }
    }
}

/// **`vars::get` is monomorphic in generated Rust**, so a member variable that
/// is not a float cannot be transpiled — measured by compiling it and reading
/// `rustc`'s refusal.
///
/// The IR is untyped at the call: `vars::get("flag")` is the same `Expr::Call`
/// whether the variable holds a float or a bool, and the interpreter answers
/// with whatever `Value` is in the map. Rust cannot. One support module cannot
/// return both, so the shipped shape of the transpiled path is: **member
/// variables are floats**. Nothing in the tree said so, because nothing had
/// compiled the output.
///
/// A bound, carried by name — not a defect this wave fixes, because fixing it
/// means typed accessors in the emitter and a decision about the support module
/// that does not exist yet.
#[test]
fn a_bool_member_variable_does_not_compile() {
    let src = "\
var flag: bool = true

on tick(dt)
  if flag then
    debug.print(\"yes\")
  end
end
";
    let (class, _) = inf_script::compile(src, "script:flag").expect("it PARSES and it LOWERS");
    // The interpreter runs it perfectly — which is the whole point.
    let mut host = Recorder {
        vars: class
            .variables
            .iter()
            .map(|v| (v.name.clone(), v.default_value()))
            .collect(),
        log: String::new(),
    };
    let binding = class.handler(&EventKind::Tick).expect("a tick handler");
    let args: HashMap<String, Value> = [("dt".to_string(), Value::Float(0.5))].into();
    inf_blueprint::eval_fn(&binding.body, &args, &mut host).expect("the interpreter runs it");
    assert!(host.log.contains("debug::print"), "{}", host.log);

    let program = compiled_program(&class, false);
    match rustc_and_run("boolvar", &program) {
        Ok(_) => panic!(
            "a bool member variable compiled — `vars::get` gained a type, or the shim did. \
             Retire this bound from the ledger and from `crown_parity`'s module doc."
        ),
        Err(stderr) => {
            println!(
                "rustc on a bool member variable: {}",
                stderr.lines().next().unwrap_or("")
            );
            assert!(
                stderr.contains("mismatched types") || stderr.contains("expected `bool`"),
                "expected a type error from the monomorphic `vars::get`:\n{stderr}"
            );
        }
    }
}
