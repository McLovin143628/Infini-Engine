//! Interpreter-vs-compiled parity (ROADMAP P6.6): the guarantee that in-editor
//! preview equals shipped code. The rotate-on-tick blueprint is (a) run through
//! the `inf-blueprint` interpreter and (b) run as **real compiled Rust** whose
//! body is exactly what the transpiler emits, over the same tick sequence — the
//! two must agree value-for-value. A string check pins the compiled reference
//! to the generator's actual output so they cannot silently drift.
//!
//! This is the CI-cheap half of the parity story (no runtime `cargo build`);
//! the dylib hot-swap path is exercised separately by `inf-hotreload`.

use std::collections::HashMap;

use inf_blueprint::interp::{eval_fn, FnHost, Value};
use inf_blueprint::{BinOp, Binding, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty};
use inf_transpile::generate_fn;

/// The rotate-on-tick handler as lowering would produce it:
/// `fn tick(dt) { let n1 = vars::get("angle") + vars::get("speed") * dt;
///               vars::set("angle", n1); engine::set_rotation(n1); }`
fn rotate_tick() -> BlueprintFn {
    let get = |name: &str| Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.into()))],
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
            Stmt::Let {
                id: LocalId(1),
                binding: Binding::Anon,
                ty: None,
                mutable: false,
                value: Expr::Binary(
                    BinOp::Add,
                    Box::new(get("angle")),
                    Box::new(Expr::Binary(
                        BinOp::Mul,
                        Box::new(get("speed")),
                        Box::new(Expr::Param("dt".into())),
                    )),
                ),
            },
            Stmt::ExprStmt(Expr::Call {
                path: vec!["vars".into(), "set".into()],
                args: vec![Expr::Lit(Lit::Str("angle".into())), Expr::Local(LocalId(1))],
            }),
            Stmt::ExprStmt(Expr::Call {
                path: vec!["engine".into(), "set_rotation".into()],
                args: vec![Expr::Local(LocalId(1))],
            }),
        ],
    }
}

/// The **compiled** side: a hand-written Rust translation of the generated
/// body, exercised by the normal test compile. Actor state is passed by
/// reference to stand in for the `vars` namespace; `engine::set_rotation`
/// records the last angle. This is what the transpiled+compiled game runs.
mod compiled {
    pub struct Actor {
        pub angle: f64,
        pub speed: f64,
        pub last_rotation: Option<f64>,
    }

    // Body mirrors generate_fn(rotate_tick()) exactly:
    //   let n1 = angle + speed * dt; angle = n1; set_rotation(n1);
    pub fn tick(a: &mut Actor, dt: f64) {
        let n1 = a.angle + a.speed * dt;
        a.angle = n1;
        a.last_rotation = Some(n1);
    }
}

#[test]
fn interpreter_matches_compiled_over_a_tick_sequence() {
    let f = rotate_tick();

    // --- compiled side ---
    let mut actor = compiled::Actor {
        angle: 0.0,
        speed: 90.0,
        last_rotation: None,
    };
    let dts = [0.016_f64, 0.5, 0.25, 1.0, 0.016];
    let mut compiled_trace = Vec::new();
    for &dt in &dts {
        compiled::tick(&mut actor, dt);
        compiled_trace.push((actor.angle, actor.last_rotation.unwrap()));
    }

    // --- interpreted side (same IR the generator renders) ---
    let mut vars: HashMap<String, Value> = [
        ("angle".into(), Value::Float(0.0)),
        ("speed".into(), Value::Float(90.0)),
    ]
    .into();
    let mut interp_trace = Vec::new();
    for &dt in &dts {
        let mut last = 0.0;
        {
            let mut host = FnHost(|path: &[String], args: &[Value]| match path {
                p if p == ["vars", "get"] => {
                    Ok(vars.get(args[0].as_str().unwrap()).cloned().unwrap())
                }
                p if p == ["vars", "set"] => {
                    vars.insert(args[0].as_str().unwrap().to_string(), args[1].clone());
                    Ok(Value::Unit)
                }
                p if p == ["engine", "set_rotation"] => {
                    last = args[0].as_float().unwrap();
                    Ok(Value::Unit)
                }
                _ => Ok(Value::Unit),
            });
            let a: HashMap<String, Value> = [("dt".into(), Value::Float(dt))].into();
            eval_fn(&f, &a, &mut host).unwrap();
        }
        let angle = vars.get("angle").unwrap().as_float().unwrap();
        interp_trace.push((angle, last));
    }

    assert_eq!(
        interp_trace, compiled_trace,
        "interpreter and compiled Rust must agree tick-for-tick"
    );

    // --- keep the compiled reference honest: the generator must still emit the
    // body the `compiled::tick` fn was written to mirror. ---
    let src = generate_fn(&f).unwrap();
    assert!(
        src.contains("let n1 = vars::get(\"angle\") + vars::get(\"speed\") * dt;"),
        "src:\n{src}"
    );
    assert!(src.contains("vars::set(\"angle\", n1);"), "src:\n{src}");
    assert!(src.contains("engine::set_rotation(n1);"), "src:\n{src}");
}
