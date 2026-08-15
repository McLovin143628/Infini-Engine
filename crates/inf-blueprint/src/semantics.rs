//! Blueprint semantics + assets (ROADMAP P6.3): the event model, member
//! variables, components, and the two on-disk units — the `.inf_act` actor
//! *class* and the `.inf_fn` function *library* — that wrap [`BlueprintFn`]
//! bodies into something the engine can instantiate and tick.
//!
//! An actor class is: some component instances, some member variables with
//! defaults, a set of event handlers ([`EventKind`] → body), and user
//! functions. At runtime an [`ActorInstance`] holds the live variable values;
//! [`ActorHost`] bridges the pure interpreter to that state and to the engine,
//! so a Tick handler can read/write `vars::*` and call engine APIs through the
//! same [`Host`](crate::interp::Host) boundary the transpiled Rust uses.
//!
//! Enums here use serde's **default (externally-tagged) representation** — kept
//! that way because bincode rejects internally-tagged enums, and because it is
//! what makes a variant travel as its *name*.
//!
//! **The wire is pretty JSON, not bincode** (lens 5 F15, Hardening Wave G).
//! `BlueprintClass` carries `skip_serializing_if` fields, which a
//! non-self-describing stream cannot round-trip, so `.inf_act`/`.inf_fn` are
//! written by `inf_editor_core::samples::encode_actor` as JSON. This paragraph
//! said "bincode payloads" and had done since P6, which reads as the positional
//! rule — reorder-unsafe, rename-safe — and the truth is the exact opposite:
//! reordering these enums is wire-safe and **renaming a variant is not**. The
//! full contract, with its pin, is on [`EventKind`]'s water block below.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::interp::{
    eval_fn_traced, AudioHost, Debug, Host, Physics2dHost, Physics3dHost, RunError, Trace, Value,
};
use crate::{BlueprintFn, Lit, Param, Ty};

/// Schema version stamped into every `.inf_act`/`.inf_fn` payload.
pub const SCHEMA_VERSION: u32 = 1;

/// What fires a handler. `Input`/`Custom` carry a name; keep this
/// externally-tagged (no `#[serde(tag)]`) for bincode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    /// Once, when the actor enters play.
    BeginPlay,
    /// Every frame; the handler receives `dt: Float` (seconds).
    Tick,
    /// A named input action fired (e.g. `"jump"`); handler receives `pressed: Bool`.
    Input(String),
    /// A collision began; handler receives `other: Int` (the other entity id).
    Collision,
    /// A user-named event, invoked explicitly; handler receives no args.
    Custom(String),
    // ── water (P20.2) ────────────────────────────────────────────────────
    //
    // **What the wire contract actually is.** A `.inf_act` is **pretty JSON**,
    // not bincode (`BlueprintClass` carries `skip_serializing_if` fields a
    // non-self-describing stream cannot round-trip — see
    // `inf_editor_core::samples::encode_actor`), so an externally-tagged variant
    // is written as its **name**, not as its declaration index. Reordering this
    // enum is therefore wire-safe and **renaming a variant is not** — the exact
    // opposite of the positional rule `WaterKind` lives under. The identifiers
    // that must never move are the [`key`](Self::key) strings, which are also
    // what a generated Rust handler is named and what `raise`/`lower` match on;
    // `crates/inf-transpile/tests/water_roundtrip.rs::the_water_event_ids_are_frozen`
    // is the pin. Appending is still the habit — a variant added in the middle
    // costs nothing on the wire but churns every `match` in the tree.
    //
    // Three variants rather than one with a phase argument, because the *point*
    // of a splash is that a handler can subscribe to it alone: "play a sound when
    // something hits the water hard" should not have to run on every quiet entry
    // and then test a float.
    /// The entity's lowest point went under a water surface. Handler receives
    /// `water: Int` (the water body's entity id) and `speed: Float` (m/s along
    /// gravity's up-axis at the crossing).
    WaterEnter,
    /// The entity cleared a water surface. Same signature as
    /// [`WaterEnter`](Self::WaterEnter).
    WaterExit,
    /// A surface crossing — in **either** direction — fast enough to throw water.
    /// Fires *in addition to* the enter/exit it accompanies, never instead of it,
    /// so a handler that only cares about wet/dry never has to know it exists.
    WaterSplash,
    // ── destruction (P22.3) ──────────────────────────────────────────────
    //
    // Appended after `WaterSplash`, per the wire note above: a `.inf_act` is
    // pretty JSON, so an externally-tagged variant is written as its NAME.
    // Appending is free; renaming would not be.
    //
    // **One variant, not two.** A `ChunkDetached` event was considered and
    // dropped: a collapse detaches chunks by the dozen inside one fixed step, so
    // a per-chunk event is a handler fired dozens of times with a chunk index it
    // cannot name anything with (chunks are not entities — see the `destruct.*`
    // kit docs). "How far has this collapsed" is a *state*, and
    // `destruct.is_intact` polls it. "This actor is finished" is an **edge**, and
    // an edge is what an event is for. Fewer is better.
    /// Every one of a destructible actor's chunks has come off — the actor is
    /// finished. Fires **once**, on the first fixed step at which that is true.
    /// Handler receives `chunks: Int` (how many came off).
    Destroyed,
}

impl EventKind {
    /// The parameters an event handler for this kind receives.
    pub fn signature(&self) -> Vec<Param> {
        match self {
            EventKind::Tick => vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            EventKind::Input(_) => vec![Param {
                name: "pressed".into(),
                ty: Ty::Bool,
            }],
            EventKind::Collision => vec![Param {
                name: "other".into(),
                ty: Ty::Int,
            }],
            EventKind::WaterEnter | EventKind::WaterExit | EventKind::WaterSplash => vec![
                Param {
                    name: "water".into(),
                    ty: Ty::Int,
                },
                Param {
                    name: "speed".into(),
                    ty: Ty::Float,
                },
            ],
            EventKind::Destroyed => vec![Param {
                name: "chunks".into(),
                ty: Ty::Int,
            }],
            EventKind::BeginPlay | EventKind::Custom(_) => vec![],
        }
    }

    /// A stable key for grouping/looking up handlers.
    pub fn key(&self) -> String {
        match self {
            EventKind::BeginPlay => "begin_play".into(),
            EventKind::Tick => "tick".into(),
            EventKind::Input(a) => format!("input:{a}"),
            EventKind::Collision => "collision".into(),
            EventKind::WaterEnter => "water_enter".into(),
            EventKind::WaterExit => "water_exit".into(),
            EventKind::WaterSplash => "water_splash".into(),
            EventKind::Destroyed => "destroyed".into(),
            EventKind::Custom(n) => format!("custom:{n}"),
        }
    }
}

/// A member variable: named, typed, with a literal default and editor exposure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub ty: Ty,
    pub default: Lit,
    /// Editable in the Details panel (an "instance-editable" variable).
    #[serde(default)]
    pub exposed: bool,
}

impl Variable {
    /// This variable's default as a runtime [`Value`].
    pub fn default_value(&self) -> Value {
        Value::from(&self.default)
    }
}

/// A component instance attached to an actor class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Component {
    /// Instance name, unique within the class (e.g. `"mesh"`).
    pub name: String,
    /// Component type (e.g. `"MeshRenderer"`, `"RigidBody"`).
    pub type_name: String,
    /// Per-property default overrides.
    #[serde(default)]
    pub defaults: BTreeMap<String, Lit>,
}

/// An event handler: a kind plus the body that runs when it fires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventBinding {
    pub event: EventKind,
    pub body: BlueprintFn,
}

/// The `.inf_act` actor class: the unit the editor instantiates into a scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlueprintClass {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    /// Base class id, if this derives from another actor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(default)]
    pub components: Vec<Component>,
    #[serde(default)]
    pub variables: Vec<Variable>,
    #[serde(default)]
    pub events: Vec<EventBinding>,
    /// Member functions callable from this class's graphs.
    #[serde(default)]
    pub functions: Vec<BlueprintFn>,
}

impl BlueprintClass {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            id: id.into(),
            name: name.into(),
            parent: None,
            components: Vec::new(),
            variables: Vec::new(),
            events: Vec::new(),
            functions: Vec::new(),
        }
    }

    /// The handler for `event`, if the class defines one.
    pub fn handler(&self, event: &EventKind) -> Option<&EventBinding> {
        self.events.iter().find(|b| &b.event == event)
    }

    /// Migrate an older payload to the current schema. v1 is current
    /// (identity); future versions add arms here.
    pub fn migrate(&mut self) {
        // no migrations yet; stamp current.
        self.schema_version = SCHEMA_VERSION;
    }
}

/// The `.inf_fn` function library: reusable pure/impure functions shared across
/// blueprints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlueprintLibrary {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub functions: Vec<BlueprintFn>,
}

impl BlueprintLibrary {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            id: id.into(),
            name: name.into(),
            functions: Vec::new(),
        }
    }

    pub fn function(&self, name: &str) -> Option<&BlueprintFn> {
        self.functions.iter().find(|f| f.name == name)
    }

    pub fn migrate(&mut self) {
        self.schema_version = SCHEMA_VERSION;
    }
}

fn default_schema() -> u32 {
    SCHEMA_VERSION
}

/// Live per-instance state: the current value of every member variable.
#[derive(Debug, Clone, Default)]
pub struct ActorInstance {
    pub vars: HashMap<String, Value>,
}

impl ActorInstance {
    /// A fresh instance with variables seeded from the class defaults.
    pub fn new(class: &BlueprintClass) -> Self {
        let vars = class
            .variables
            .iter()
            .map(|v| (v.name.clone(), v.default_value()))
            .collect();
        Self { vars }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.vars.get(name)
    }
}

/// Bridges the pure interpreter to actor state + the engine. Calls to the
/// `vars` namespace (`vars::get(name)` / `vars::set(name, value)`) read and
/// write member variables; everything else delegates to `inner` (the engine
/// host: rotate, spawn, log, …).
pub struct ActorHost<'a> {
    pub actor: &'a mut ActorInstance,
    pub inner: &'a mut dyn Host,
}

impl Host for ActorHost<'_> {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        if path.len() == 2 && path[0] == "vars" {
            let name = args
                .first()
                .ok_or_else(|| RunError::Host(path.join("::"), "missing variable name".into()))?
                .as_str()?
                .to_string();
            match path[1].as_str() {
                "get" => {
                    return self.actor.vars.get(&name).cloned().ok_or_else(|| {
                        RunError::Host("vars::get".into(), format!("no var `{name}`"))
                    });
                }
                "set" => {
                    let value = args.get(1).cloned().ok_or_else(|| {
                        RunError::Host("vars::set".into(), "missing value".into())
                    })?;
                    self.actor.vars.insert(name, value);
                    return Ok(Value::Unit);
                }
                _ => {}
            }
        }
        // `nodestate::get_or(key, default)` / `nodestate::set(key, value)` back
        // the stateful flow nodes (do_once/flip_flop/gate). State lives in the
        // same `instance.vars` map under reserved `__bp_<kind>_<NodeId>` keys
        // (the lowerer's format), which never collide with user variable names.
        // `get_or` returns the stored value or the supplied default on a miss —
        // so a never-fired node reads its initial state cleanly.
        if path.len() == 2 && path[0] == "nodestate" {
            let key = args
                .first()
                .ok_or_else(|| RunError::Host(path.join("::"), "missing state key".into()))?
                .as_str()?
                .to_string();
            match path[1].as_str() {
                "get_or" => {
                    let default = args.get(1).cloned().unwrap_or(Value::Unit);
                    return Ok(self.actor.vars.get(&key).cloned().unwrap_or(default));
                }
                "set" => {
                    let value = args.get(1).cloned().ok_or_else(|| {
                        RunError::Host("nodestate::set".into(), "missing value".into())
                    })?;
                    self.actor.vars.insert(key, value);
                    return Ok(Value::Unit);
                }
                _ => {}
            }
        }
        self.inner.call(path, args)
    }

    /// Forward physics to the wrapped engine host, so a Tick handler over a
    /// physics-capable host reaches `physics2d.*` nodes through the actor layer.
    fn physics(&mut self) -> Option<&mut dyn Physics2dHost> {
        self.inner.physics()
    }

    /// Forward 3D physics likewise, so `physics3d.*` nodes reach the engine host
    /// through the actor layer (the `d3` mirror of [`physics`](Self::physics)).
    fn physics3d(&mut self) -> Option<&mut dyn Physics3dHost> {
        self.inner.physics3d()
    }

    /// Forward audio likewise, so `audio.*` nodes reach the engine host's command
    /// sink through the actor layer (P12.3).
    fn audio(&mut self) -> Option<&mut dyn AudioHost> {
        self.inner.audio()
    }
}

/// Fire an event on an actor instance, running its handler through the
/// interpreter with variable access + engine calls. `args` supplies the
/// event's signature params (e.g. `dt` for Tick). No handler → a no-op.
pub fn run_event(
    class: &BlueprintClass,
    actor: &mut ActorInstance,
    event: &EventKind,
    args: &HashMap<String, Value>,
    engine: &mut dyn Host,
    debug: &Debug,
) -> Result<Trace, RunError> {
    let Some(binding) = class.handler(event) else {
        return Ok(Trace::default());
    };
    let mut host = ActorHost {
        actor,
        inner: engine,
    };
    let (_v, trace) = eval_fn_traced(&binding.body, args, &mut host, debug)?;
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::FnHost;
    use crate::{BinOp, Binding, Expr, Lit, LocalId, Stmt};

    /// The Phase-6 gate handler: "rotate on tick".
    /// fn tick(dt) { let a = vars::get("angle") + dt * vars::get("speed");
    ///               vars::set("angle", a); engine::set_rotation(a); }
    fn rotate_on_tick() -> BlueprintClass {
        let body = BlueprintFn {
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
                    binding: Binding::Named("a".into()),
                    ty: None,
                    mutable: false,
                    value: Expr::Binary(
                        BinOp::Add,
                        Box::new(Expr::Call {
                            path: vec!["vars".into(), "get".into()],
                            args: vec![Expr::Lit(Lit::Str("angle".into()))],
                        }),
                        Box::new(Expr::Binary(
                            BinOp::Mul,
                            Box::new(Expr::Param("dt".into())),
                            Box::new(Expr::Call {
                                path: vec!["vars".into(), "get".into()],
                                args: vec![Expr::Lit(Lit::Str("speed".into()))],
                            }),
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
        };
        let mut class = BlueprintClass::new("act:spinner", "Spinner");
        class.variables = vec![
            Variable {
                name: "angle".into(),
                ty: Ty::Float,
                default: Lit::Float(0.0),
                exposed: true,
            },
            Variable {
                name: "speed".into(),
                ty: Ty::Float,
                default: Lit::Float(90.0),
                exposed: true,
            },
        ];
        class.events = vec![EventBinding {
            event: EventKind::Tick,
            body,
        }];
        class
    }

    #[test]
    fn tick_advances_angle_and_calls_engine() {
        let class = rotate_on_tick();
        let mut actor = ActorInstance::new(&class);
        assert_eq!(actor.get("angle"), Some(&Value::Float(0.0)));

        let mut last_rotation = None;
        {
            let mut engine = FnHost(|path: &[String], args: &[Value]| {
                if path == ["engine", "set_rotation"] {
                    last_rotation = Some(args[0].as_float().unwrap());
                }
                Ok(Value::Unit)
            });
            // Two 0.5s ticks at speed 90 → angle 45, then 90.
            let dt: HashMap<String, Value> = [("dt".to_string(), Value::Float(0.5))].into();
            run_event(
                &class,
                &mut actor,
                &EventKind::Tick,
                &dt,
                &mut engine,
                &Debug::default(),
            )
            .unwrap();
            run_event(
                &class,
                &mut actor,
                &EventKind::Tick,
                &dt,
                &mut engine,
                &Debug::default(),
            )
            .unwrap();
        }
        assert_eq!(actor.get("angle"), Some(&Value::Float(90.0)));
        assert_eq!(last_rotation, Some(90.0));
    }

    #[test]
    fn event_signatures() {
        assert_eq!(EventKind::Tick.signature()[0].name, "dt");
        assert!(EventKind::BeginPlay.signature().is_empty());
        assert_eq!(EventKind::Input("jump".into()).key(), "input:jump");
    }

    #[test]
    fn class_and_library_round_trip_json() {
        let class = rotate_on_tick();
        let json = serde_json::to_string(&class).unwrap();
        let back: BlueprintClass = serde_json::from_str(&json).unwrap();
        assert_eq!(class, back);

        let mut lib = BlueprintLibrary::new("fn:math", "Math");
        lib.functions.push(class.events[0].body.clone());
        let j2 = serde_json::to_string(&lib).unwrap();
        let back2: BlueprintLibrary = serde_json::from_str(&j2).unwrap();
        assert_eq!(lib, back2);
    }

    #[test]
    fn missing_handler_is_noop() {
        let class = BlueprintClass::new("act:empty", "Empty");
        let mut actor = ActorInstance::new(&class);
        let mut engine = crate::interp::PureHost;
        let trace = run_event(
            &class,
            &mut actor,
            &EventKind::BeginPlay,
            &HashMap::new(),
            &mut engine,
            &Debug::default(),
        )
        .unwrap();
        assert!(trace.wires.is_empty());
    }

    /// **`.inf_act` is JSON, and the player re-decodes it — so every `Lit::Float`
    /// must survive `serde_json` bit for bit.**
    ///
    /// This is the blueprint twin of `inf-pcg`'s
    /// `an_authored_graph_re_lowers_bit_identically_through_the_payload`, and it
    /// guards the *higher-traffic* half of the same hazard: a `.inf_act` is
    /// stored, cooked, packed and streamed as JSON, and the shipped/PIE player
    /// decodes it at load on every boot path (dev dir, pack, and the PIE
    /// `ScenePayload.classes` list). `serde_json`'s **default** float parser is a
    /// fast path that can land one ULP off; a literal that came back a bit light
    /// makes the shipped actor compute something imperceptibly different from the
    /// preview — a divergence that no schema check, no hash and no gate below the
    /// simulation would notice, because both sides are internally consistent.
    ///
    /// The workspace pins `serde_json = { features = ["float_roundtrip"] }` for
    /// exactly this; deleting that feature must fail here as well as in `inf-pcg`,
    /// so neither crate is the sole reason the pin exists.
    #[test]
    fn every_float_literal_survives_the_inf_act_json_round_trip_bit_for_bit() {
        // Full 17-significant-digit mantissas — what an author's slider, an
        // imported curve, or a computed default actually produces.
        let awkward: [f64; 6] = [
            -114668.51350953568,
            0.1 + 0.2, // 0.30000000000000004
            1.7976931348623157e300,
            f64::MIN_POSITIVE,
            // The largest *subnormal*, named by its bits rather than by a
            // decimal literal — the nastiest thing a float parser can be handed,
            // and immune to anyone "tidying" the digits.
            f64::from_bits(0x000F_FFFF_FFFF_FFFF),
            9.007199254740993e15, // just past 2^53, where doubles go sparse
        ];

        let mut class = BlueprintClass::new("bp.floats", "Floats");
        let body = BlueprintFn {
            id: "begin".into(),
            name: "begin".into(),
            params: Vec::new(),
            ret: Ty::Unit,
            body: awkward
                .iter()
                .enumerate()
                .map(|(i, &f)| Stmt::Let {
                    id: LocalId(i as u32),
                    binding: Binding::Named(format!("v{i}")),
                    ty: Some(Ty::Float),
                    mutable: false,
                    // Nested inside a binary op, so the literal is not merely a
                    // top-level number the parser might special-case.
                    value: Expr::Binary(
                        BinOp::Add,
                        Box::new(Expr::Lit(Lit::Float(f))),
                        Box::new(Expr::Lit(Lit::Float(0.0))),
                    ),
                })
                .collect(),
        };
        class.events.push(EventBinding {
            event: EventKind::BeginPlay,
            body,
        });
        // …and on a member variable's default, the other place a float persists.
        class.variables.push(Variable {
            name: "speed".into(),
            ty: Ty::Float,
            default: Lit::Float(awkward[0]),
            exposed: true,
        });

        let json = serde_json::to_string(&class).expect("to json");
        let back: BlueprintClass = serde_json::from_str(&json).expect("from json");
        assert_eq!(back, class, "the class did not round-trip through JSON");

        // `PartialEq` on `f64` would already have caught a moved bit, but state
        // it on the bits so a future `PartialEq` that got clever about floats
        // cannot quietly weaken this.
        let floats = |c: &BlueprintClass| -> Vec<u64> {
            let mut out: Vec<u64> = Vec::new();
            fn walk(e: &Expr, out: &mut Vec<u64>) {
                match e {
                    Expr::Lit(Lit::Float(f)) => out.push(f.to_bits()),
                    Expr::Binary(_, lhs, rhs) => {
                        walk(lhs, out);
                        walk(rhs, out);
                    }
                    _ => {}
                }
            }
            for ev in &c.events {
                for st in &ev.body.body {
                    if let Stmt::Let { value, .. } = st {
                        walk(value, &mut out);
                    }
                }
            }
            for v in &c.variables {
                if let Lit::Float(f) = &v.default {
                    out.push(f.to_bits());
                }
            }
            out
        };
        let (before, after) = (floats(&class), floats(&back));
        assert_eq!(
            before.len(),
            awkward.len() * 2 + 1,
            "the walk found them all"
        );
        assert_eq!(
            before, after,
            "a float literal moved a bit crossing `.inf_act` JSON — the \
             `serde_json` `float_roundtrip` feature is not enabled"
        );
        // Re-serializing is byte-identical, so a cook that rewrites a class does
        // not churn its content hash.
        assert_eq!(json, serde_json::to_string(&back).unwrap());
    }
}
