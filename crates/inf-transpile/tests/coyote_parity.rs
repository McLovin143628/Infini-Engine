//! Interpreter-vs-compiled **parity for the coyote-time jump** (ROADMAP P8.4 /
//! §8): the P8.4 extension of the P6.6 parity gate to the physics node kit.
//!
//! The coyote-time handler is (a) run through the `inf-blueprint` interpreter
//! with a physics **`Physics2dHost`** and an input host, and (b) run as real
//! compiled Rust whose body mirrors what the transpiler emits, over the same
//! scripted (dt, jump, floor) sequence — driving an **identical** mock physics on
//! both sides. The two must agree value-for-value (vy, coyote timer, height). A
//! codegen check pins the compiled reference to the generator's actual output.
//!
//! ## Why this test exists (the harness seam)
//!
//! The P6.6 `parity.rs` harness is a single hard-coded fixture with an inline
//! `FnHost` closure — it has **no seam to inject a physics host**, so it cannot
//! host a `physics2d.*` graph as-is. Rather than widen that fixture, this test
//! follows the same three-part pattern (IR ⇄ interp ⇄ hand-mirrored compiled)
//! but with a struct host that overrides `Host::physics()`. There is no
//! production `physics2d::*` runtime module yet (a documented P9 game-loop shim),
//! so the "compiled" side hand-writes the mock module it would bind to — exactly
//! as `parity.rs` hand-writes the `vars`/`engine` stand-ins.

use std::collections::HashMap;

use inf_blueprint::interp::{
    eval_fn, Host, MoveResult2d, Physics2dHost, RayHit2d, RunError, Value,
};
use inf_blueprint::{BinOp, Binding, BlueprintFn, Expr, Lit, LocalId, Param, Stmt, Ty};
use inf_transpile::generate_fn;

const GRAVITY: f64 = 30.0;
const JUMP_SPEED: f64 = 9.0;
const COYOTE_TIME: f64 = 0.12;

// ── The shared mock physics (identical on both sides) ────────────────────────
/// A 1-D vertical mock: the character sits at height `y`; `floor` is the current
/// ground height (lowered when it "walks off a ledge"). `move_and_slide` clamps
/// to the floor and reports grounded; `is_grounded` probes without moving.
#[derive(Clone, Copy)]
struct Phys {
    y: f64,
    floor: f64,
}

impl Phys {
    fn move_and_slide(&mut self, dy: f64) -> bool {
        self.y += dy;
        if self.y <= self.floor {
            self.y = self.floor;
            true
        } else {
            false
        }
    }
    fn is_grounded(&self) -> bool {
        self.y <= self.floor
    }
}

// ── IR: the coyote-time Tick handler (what lowering produces) ────────────────
fn s(v: &str) -> Expr {
    Expr::Lit(Lit::Str(v.into()))
}
fn f(v: f64) -> Expr {
    Expr::Lit(Lit::Float(v))
}
fn getv(name: &str) -> Expr {
    Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![s(name)],
    }
}
fn setv(name: &str, value: Expr) -> Stmt {
    Stmt::ExprStmt(Expr::Call {
        path: vec!["vars".into(), "set".into()],
        args: vec![s(name), value],
    })
}
fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Binary(op, Box::new(a), Box::new(b))
}
fn call(path: &[&str], args: Vec<Expr>) -> Expr {
    Expr::Call {
        path: path.iter().map(|p| p.to_string()).collect(),
        args,
    }
}

fn coyote_tick() -> BlueprintFn {
    let e = || Expr::Local(LocalId(1));
    let grounded = || Expr::Local(LocalId(2));
    let dt = || Expr::Param("dt".into());
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
                binding: Binding::Named("entity".into()),
                ty: None,
                mutable: false,
                value: getv("entity"),
            },
            Stmt::Let {
                id: LocalId(2),
                binding: Binding::Named("grounded".into()),
                ty: None,
                mutable: false,
                value: call(&["physics2d", "is_grounded"], vec![e()]),
            },
            Stmt::If {
                cond: grounded(),
                then_body: vec![setv("coyote", f(COYOTE_TIME))],
                else_body: vec![setv("coyote", bin(BinOp::Sub, getv("coyote"), dt()))],
            },
            setv(
                "vy",
                bin(BinOp::Sub, getv("vy"), bin(BinOp::Mul, f(GRAVITY), dt())),
            ),
            Stmt::If {
                cond: bin(
                    BinOp::And,
                    call(&["input", "just_pressed"], vec![s("jump")]),
                    bin(
                        BinOp::Or,
                        grounded(),
                        bin(BinOp::Gt, getv("coyote"), f(0.0)),
                    ),
                ),
                then_body: vec![setv("vy", f(JUMP_SPEED)), setv("coyote", f(0.0))],
                else_body: vec![],
            },
            Stmt::Let {
                id: LocalId(3),
                binding: Binding::Named("grounded_after".into()),
                ty: None,
                mutable: false,
                value: call(
                    &["physics2d", "move_and_slide"],
                    vec![e(), f(0.0), bin(BinOp::Mul, getv("vy"), dt())],
                ),
            },
            Stmt::If {
                cond: bin(
                    BinOp::And,
                    Expr::Local(LocalId(3)),
                    bin(BinOp::Lt, getv("vy"), f(0.0)),
                ),
                then_body: vec![setv("vy", f(0.0))],
                else_body: vec![],
            },
        ],
    }
}

// ── The compiled side: a hand-written mirror of generate_fn(coyote_tick()) ───
mod compiled {
    use super::Phys;
    pub struct Actor {
        pub vy: f64,
        pub coyote: f64,
    }
    // Mirrors the generated body: is_grounded → coyote timer → gravity → jump
    // gate → move_and_slide → landing reset. `jump` stands in for
    // input::just_pressed("jump"), `phys` for the physics2d::* free calls.
    pub fn tick(a: &mut Actor, dt: f64, jump: bool, phys: &mut Phys, entity: i64) {
        let _ = entity;
        let grounded = phys.is_grounded();
        if grounded {
            a.coyote = super::COYOTE_TIME;
        } else {
            a.coyote -= dt;
        }
        a.vy -= super::GRAVITY * dt;
        if jump && (grounded || a.coyote > 0.0) {
            a.vy = super::JUMP_SPEED;
            a.coyote = 0.0;
        }
        let grounded_after = phys.move_and_slide(a.vy * dt);
        if grounded_after && a.vy < 0.0 {
            a.vy = 0.0;
        }
    }
}

// ── The interpreter host: vars + input + a real Physics2dHost ────────────────
struct CoyoteHost<'a> {
    vars: &'a mut HashMap<String, Value>,
    jump: bool,
    phys: &'a mut Phys,
}

impl Host for CoyoteHost<'_> {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        match (
            path.first().map(String::as_str),
            path.get(1).map(String::as_str),
        ) {
            (Some("vars"), Some("get")) => Ok(self
                .vars
                .get(args[0].as_str().unwrap())
                .cloned()
                .unwrap_or(Value::Float(0.0))),
            (Some("vars"), Some("set")) => {
                self.vars
                    .insert(args[0].as_str().unwrap().to_string(), args[1].clone());
                Ok(Value::Unit)
            }
            (Some("input"), Some("just_pressed")) => Ok(Value::Bool(self.jump)),
            (Some("input"), Some("is_down")) => Ok(Value::Bool(false)),
            _ => Ok(Value::Unit),
        }
    }
    fn physics(&mut self) -> Option<&mut dyn Physics2dHost> {
        Some(self)
    }
}

impl Physics2dHost for CoyoteHost<'_> {
    fn move_and_slide(&mut self, _entity: i64, motion: [f64; 2]) -> Result<MoveResult2d, String> {
        let grounded = self.phys.move_and_slide(motion[1]);
        Ok(MoveResult2d {
            applied: motion,
            grounded,
        })
    }
    fn is_grounded(&mut self, _entity: i64) -> Result<bool, String> {
        Ok(self.phys.is_grounded())
    }
    fn raycast(&mut self, _o: [f64; 2], _d: [f64; 2], _m: f64) -> Result<Option<RayHit2d>, String> {
        Ok(None)
    }
    fn set_velocity(&mut self, _e: i64, _v: [f64; 2]) -> Result<(), String> {
        Ok(())
    }
    fn get_velocity(&mut self, _e: i64) -> Result<[f64; 2], String> {
        Ok([0.0, 0.0])
    }
    fn apply_impulse(&mut self, _e: i64, _v: [f64; 2]) -> Result<(), String> {
        Ok(())
    }
}

/// A scripted step: fixed dt, whether jump was pressed this step, and the current
/// ground height (dropped to walk off a ledge).
struct Step {
    dt: f64,
    jump: bool,
    floor: f64,
}

fn script() -> Vec<Step> {
    let dt = 1.0 / 60.0;
    let mut steps = Vec::new();
    // 3 steps grounded (floor 0).
    for _ in 0..3 {
        steps.push(Step {
            dt,
            jump: false,
            floor: 0.0,
        });
    }
    // Walk off the ledge: floor drops far below.
    for _ in 0..2 {
        steps.push(Step {
            dt,
            jump: false,
            floor: -100.0,
        });
    }
    // Coyote jump (still within the window).
    steps.push(Step {
        dt,
        jump: true,
        floor: -100.0,
    });
    // Rise then fall.
    for _ in 0..8 {
        steps.push(Step {
            dt,
            jump: false,
            floor: -100.0,
        });
    }
    // A late jump long after the window closed — must be ignored.
    steps.push(Step {
        dt,
        jump: true,
        floor: -100.0,
    });
    for _ in 0..4 {
        steps.push(Step {
            dt,
            jump: false,
            floor: -100.0,
        });
    }
    steps
}

#[test]
fn interpreter_matches_compiled_over_a_coyote_sequence() {
    let f = coyote_tick();
    let steps = script();

    // --- compiled side ---
    let mut c_actor = compiled::Actor {
        vy: 0.0,
        coyote: 0.0,
    };
    let mut c_phys = Phys { y: 0.0, floor: 0.0 };
    let mut compiled_trace = Vec::new();
    for s in &steps {
        c_phys.floor = s.floor;
        compiled::tick(&mut c_actor, s.dt, s.jump, &mut c_phys, 1);
        compiled_trace.push((round(c_actor.vy), round(c_actor.coyote), round(c_phys.y)));
    }

    // --- interpreted side (the same IR the generator renders) ---
    let mut vars: HashMap<String, Value> = [
        ("entity".into(), Value::Int(1)),
        ("vy".into(), Value::Float(0.0)),
        ("coyote".into(), Value::Float(0.0)),
    ]
    .into();
    let mut i_phys = Phys { y: 0.0, floor: 0.0 };
    let mut interp_trace = Vec::new();
    for s in &steps {
        i_phys.floor = s.floor;
        {
            let mut host = CoyoteHost {
                vars: &mut vars,
                jump: s.jump,
                phys: &mut i_phys,
            };
            let args: HashMap<String, Value> = [("dt".into(), Value::Float(s.dt))].into();
            eval_fn(&f, &args, &mut host).unwrap();
        }
        let vy = vars.get("vy").unwrap().as_float().unwrap();
        let coyote = vars.get("coyote").unwrap().as_float().unwrap();
        interp_trace.push((round(vy), round(coyote), round(i_phys.y)));
    }

    assert_eq!(
        interp_trace, compiled_trace,
        "interpreter and compiled Rust must agree tick-for-tick over the coyote sequence"
    );

    // The coyote jump actually lifted the character at some point (the scenario
    // is non-trivial), and the trajectory is not constant.
    let max_y = interp_trace.iter().map(|(_, _, y)| *y).max().unwrap();
    assert!(
        max_y > 0,
        "the coyote jump should have lifted the character"
    );

    // --- keep the compiled reference honest: the generator still emits the free
    // calls the mirror was written against. ---
    let src = generate_fn(&f).unwrap();
    assert!(src.contains("physics2d::is_grounded("), "src:\n{src}");
    assert!(src.contains("physics2d::move_and_slide("), "src:\n{src}");
    assert!(src.contains("input::just_pressed("), "src:\n{src}");
}

fn round(v: f64) -> i64 {
    (v * 1e6).round() as i64
}
