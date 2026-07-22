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
//! Enums here use serde's **default (externally-tagged) representation** so the
//! `.inf_act`/`.inf_fn` bincode payloads round-trip (bincode rejects
//! internally-tagged enums).

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
    use crate::{BinOp, Binding, Expr, LocalId, Stmt};

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
}
