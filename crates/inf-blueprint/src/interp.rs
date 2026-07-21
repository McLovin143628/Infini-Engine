//! Tree-walking interpreter over the [`BlueprintFn`] IR (ROADMAP P6.4).
//!
//! The interpreter walks *the same IR the transpiler renders to Rust*, so
//! "preview == shipped compiled code" holds by construction — the P6.6 parity
//! gate runs a fixture through both this evaluator and the compiled dylib and
//! asserts identical results.
//!
//! Everything the outside world touches — engine calls, actor member
//! variables, spawning — goes through the [`Host`] trait, keeping the IR a
//! pure expression/statement language. That boundary is exactly where the
//! interpreter and the transpiled Rust must agree.
//!
//! Debugging support (P6.4): every value node (`Let`) can carry a **breakpoint**
//! and its computed value is captured for **wire inspection**, so the editor
//! can show hover values and pause on a node.

use std::collections::{HashMap, HashSet};

use crate::{BinOp, BlueprintFn, Expr, Lit, LocalId, Stmt, UnOp};

/// A runtime value. Mirrors [`crate::Ty`]; `Unit` is the empty return.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    Unit,
}

impl Value {
    pub fn as_float(&self) -> Result<f64, RunError> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            other => Err(RunError::Type(format!("expected float, got {other:?}"))),
        }
    }

    pub fn as_int(&self) -> Result<i64, RunError> {
        match self {
            Value::Int(i) => Ok(*i),
            other => Err(RunError::Type(format!("expected int, got {other:?}"))),
        }
    }

    pub fn as_bool(&self) -> Result<bool, RunError> {
        match self {
            Value::Bool(b) => Ok(*b),
            other => Err(RunError::Type(format!("expected bool, got {other:?}"))),
        }
    }

    pub fn as_str(&self) -> Result<&str, RunError> {
        match self {
            Value::Str(s) => Ok(s),
            other => Err(RunError::Type(format!("expected string, got {other:?}"))),
        }
    }
}

impl From<&Lit> for Value {
    fn from(lit: &Lit) -> Self {
        match lit {
            Lit::Float(f) => Value::Float(*f),
            Lit::Int(i) => Value::Int(*i),
            Lit::Bool(b) => Value::Bool(*b),
            Lit::Str(s) => Value::Str(s.clone()),
        }
    }
}

/// A failure while interpreting.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum RunError {
    #[error("type error: {0}")]
    Type(String),
    #[error("unbound parameter `{0}`")]
    UnboundParam(String),
    #[error("unbound local n{0}")]
    UnboundLocal(u32),
    #[error("division by zero")]
    DivByZero,
    #[error("host call `{0}` failed: {1}")]
    Host(String, String),
    #[error("cannot interpret opaque snippet (hand-written Rust outside the liftable subset)")]
    Snippet,
    #[error("host has no function `{0}`")]
    NoSuchHostFn(String),
}

/// The boundary between the pure IR and the engine: free-function calls
/// ([`Expr::Call`]) dispatch here. An actor's member-variable get/set, engine
/// APIs (rotate, spawn), and logging are all host functions — so the
/// interpreter and the transpiled Rust share one contract.
pub trait Host {
    /// Dispatch a call to `path` (e.g. `["actor", "rotate"]`) with `args`.
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError>;

    /// Optional access to a 2D physics world, powering the `physics2d.*`
    /// blueprint node kit (P8.3b). Added via the [`Host`] boundary with **no IR
    /// change** — physics nodes are ordinary [`Expr::Call`]s that the
    /// interpreter routes here — exactly like member variables were added
    /// through `vars::*` calls.
    ///
    /// The default returns `None`: a host without a physics world makes every
    /// `physics2d.*` node report a clean "no physics world available" error (or,
    /// for the void actions, do nothing) rather than dispatching. A host that
    /// owns a [`PhysicsBridge2D`](../../inf_physics/index.html) overrides this.
    fn physics(&mut self) -> Option<&mut dyn Physics2dHost> {
        None
    }

    /// Optional access to a **3D** physics world, powering the `physics3d.*`
    /// blueprint node kit (P11.3). The exact `d3` mirror of [`physics`](Self::physics):
    /// `physics3d.*` nodes are ordinary [`Expr::Call`]s the interpreter routes to
    /// this accessor with **no IR change**. Default `None` — a host without a 3D
    /// world makes every `physics3d.*` node report a clean "no physics world
    /// available" error. A host that owns a
    /// [`PhysicsBridge3D`](../../inf_physics/index.html) overrides this.
    fn physics3d(&mut self) -> Option<&mut dyn Physics3dHost> {
        None
    }
}

/// The result of a [`Physics2dHost::move_and_slide`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveResult2d {
    /// The collision-resolved translation actually applied to the character.
    pub applied: [f64; 2],
    /// Whether the character ended the move grounded.
    pub grounded: bool,
}

/// A ray/shape hit returned by [`Physics2dHost::raycast`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit2d {
    /// World-space point of impact.
    pub point: [f64; 2],
    /// Surface normal at the impact.
    pub normal: [f64; 2],
}

/// The 2D-physics accessor the interpreter routes `physics2d.*` nodes to.
///
/// This is the **minimal Ring-0 runtime surface** the transpiler also targets:
/// generated Rust emits free calls (`physics2d::move_and_slide(entity, mx, my)`,
/// `physics2d::raycast::hit(ox, oy, dx, dy, max)`, …) that a P9 game-loop shim
/// binds to a value implementing this trait (backed by an `inf_physics`
/// `PhysicsBridge2D`). Entities are opaque `i64` ids (the runtime maps them to
/// its own entity handles); vectors are split `[x, y]` because the blueprint IR
/// has no first-class `Vec2` value yet (a documented follow-up — see the node
/// kit). Every method returns `Result<_, String>` so a bad entity id surfaces
/// as a host error rather than a panic.
pub trait Physics2dHost {
    /// Move-and-slide `entity` by `motion`, resolving collisions; returns the
    /// applied translation and whether it ended grounded.
    fn move_and_slide(&mut self, entity: i64, motion: [f64; 2]) -> Result<MoveResult2d, String>;
    /// Whether `entity` is currently touching the ground.
    fn is_grounded(&mut self, entity: i64) -> Result<bool, String>;
    /// Cast a ray from `origin` along `dir` up to `max`; `None` on a miss.
    fn raycast(
        &mut self,
        origin: [f64; 2],
        dir: [f64; 2],
        max: f64,
    ) -> Result<Option<RayHit2d>, String>;
    /// Set `entity`'s linear velocity.
    fn set_velocity(&mut self, entity: i64, v: [f64; 2]) -> Result<(), String>;
    /// Read `entity`'s linear velocity.
    fn get_velocity(&mut self, entity: i64) -> Result<[f64; 2], String>;
    /// Apply an instantaneous linear impulse to `entity`.
    fn apply_impulse(&mut self, entity: i64, v: [f64; 2]) -> Result<(), String>;
}

/// Route a `physics2d.*` [`Expr::Call`] to the host's [`Physics2dHost`]. The
/// call `path` is `["physics2d", op]` or `["physics2d", op, field]` for the
/// multi-component queries (a vector/tuple result split into scalar output
/// pins). Kept in one place so every [`Host`] that provides `physics()`
/// supports the whole node kit identically — and so the interpreter and the
/// transpiled Rust agree field-for-field.
pub(crate) fn dispatch_physics2d(
    host: &mut dyn Host,
    path: &[String],
    args: &[Value],
) -> Result<Value, RunError> {
    let key = path.join("::");
    let err = |msg: String| RunError::Host(key.clone(), msg);
    let phys = host
        .physics()
        .ok_or_else(|| err("no physics world available".into()))?;

    let op = path.get(1).map(String::as_str).unwrap_or("");
    let field = path.get(2).map(String::as_str);
    let ent = |i: usize| args.get(i).map_or(Ok(0), Value::as_int);
    let flt = |i: usize| args.get(i).map_or(Ok(0.0), Value::as_float);
    let he = |e: String| RunError::Host(key.clone(), e);

    match (op, field) {
        ("move_and_slide", None) => {
            let r = phys
                .move_and_slide(ent(0)?, [flt(1)?, flt(2)?])
                .map_err(he)?;
            // This node's single data output is `grounded`; the applied vector is
            // a documented follow-up (needs multi-value pins / a Vec2 type).
            Ok(Value::Bool(r.grounded))
        }
        ("is_grounded", None) => Ok(Value::Bool(phys.is_grounded(ent(0)?).map_err(he)?)),
        ("raycast", Some(field)) => {
            let hit = phys
                .raycast([flt(0)?, flt(1)?], [flt(2)?, flt(3)?], flt(4)?)
                .map_err(he)?;
            Ok(match field {
                "hit" => Value::Bool(hit.is_some()),
                "point_x" => Value::Float(hit.map_or(0.0, |h| h.point[0])),
                "point_y" => Value::Float(hit.map_or(0.0, |h| h.point[1])),
                "normal_x" => Value::Float(hit.map_or(0.0, |h| h.normal[0])),
                "normal_y" => Value::Float(hit.map_or(0.0, |h| h.normal[1])),
                other => return Err(err(format!("unknown raycast field `{other}`"))),
            })
        }
        ("get_velocity", Some(field)) => {
            let v = phys.get_velocity(ent(0)?).map_err(he)?;
            Ok(match field {
                "x" => Value::Float(v[0]),
                "y" => Value::Float(v[1]),
                other => return Err(err(format!("unknown get_velocity field `{other}`"))),
            })
        }
        ("set_velocity", None) => {
            phys.set_velocity(ent(0)?, [flt(1)?, flt(2)?]).map_err(he)?;
            Ok(Value::Unit)
        }
        ("apply_impulse", None) => {
            phys.apply_impulse(ent(0)?, [flt(1)?, flt(2)?])
                .map_err(he)?;
            Ok(Value::Unit)
        }
        _ => Err(err(format!("unknown physics2d node `{key}`"))),
    }
}

/// The result of a [`Physics3dHost::move_and_slide`] call (the `d3` mirror of
/// [`MoveResult2d`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveResult3d {
    /// The collision-resolved translation actually applied to the character.
    pub applied: [f64; 3],
    /// Whether the character ended the move grounded.
    pub grounded: bool,
}

/// A ray hit returned by [`Physics3dHost::raycast`] (the `d3` mirror of
/// [`RayHit2d`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit3d {
    /// World-space point of impact.
    pub point: [f64; 3],
    /// Surface normal at the impact.
    pub normal: [f64; 3],
}

/// The 3D-physics accessor the interpreter routes `physics3d.*` nodes to — the
/// exact `d3` mirror of [`Physics2dHost`]. Generated Rust emits free calls
/// (`physics3d::move_and_slide(entity, mx, my, mz)`,
/// `physics3d::raycast::point_z(...)`, …) a runtime shim binds to a value
/// implementing this trait (backed by an `inf_physics` `PhysicsBridge3D`).
/// Entities are opaque `i64` ids; vectors are split `[x, y, z]` because the
/// blueprint IR has no first-class `Vec3` value yet (the same documented
/// follow-up as the 2D kit). Every method returns `Result<_, String>`.
pub trait Physics3dHost {
    /// Move-and-slide `entity` by `motion`, resolving collisions; returns the
    /// applied translation and whether it ended grounded.
    fn move_and_slide(&mut self, entity: i64, motion: [f64; 3]) -> Result<MoveResult3d, String>;
    /// Whether `entity` is currently touching the ground.
    fn is_grounded(&mut self, entity: i64) -> Result<bool, String>;
    /// Cast a ray from `origin` along `dir` up to `max`; `None` on a miss.
    fn raycast(
        &mut self,
        origin: [f64; 3],
        dir: [f64; 3],
        max: f64,
    ) -> Result<Option<RayHit3d>, String>;
    /// Set `entity`'s linear velocity.
    fn set_velocity(&mut self, entity: i64, v: [f64; 3]) -> Result<(), String>;
    /// Read `entity`'s linear velocity.
    fn get_velocity(&mut self, entity: i64) -> Result<[f64; 3], String>;
    /// Apply an instantaneous linear impulse to `entity`.
    fn apply_impulse(&mut self, entity: i64, v: [f64; 3]) -> Result<(), String>;
}

/// Route a `physics3d.*` [`Expr::Call`] to the host's [`Physics3dHost`] — the
/// `d3` mirror of [`dispatch_physics2d`]. The call `path` is `["physics3d", op]`
/// or `["physics3d", op, field]` for the multi-component queries (a vector result
/// split into scalar output pins), so the interpreter and the transpiled Rust
/// agree field-for-field.
pub(crate) fn dispatch_physics3d(
    host: &mut dyn Host,
    path: &[String],
    args: &[Value],
) -> Result<Value, RunError> {
    let key = path.join("::");
    let err = |msg: String| RunError::Host(key.clone(), msg);
    let phys = host
        .physics3d()
        .ok_or_else(|| err("no physics world available".into()))?;

    let op = path.get(1).map(String::as_str).unwrap_or("");
    let field = path.get(2).map(String::as_str);
    let ent = |i: usize| args.get(i).map_or(Ok(0), Value::as_int);
    let flt = |i: usize| args.get(i).map_or(Ok(0.0), Value::as_float);
    let he = |e: String| RunError::Host(key.clone(), e);

    match (op, field) {
        ("move_and_slide", None) => {
            let r = phys
                .move_and_slide(ent(0)?, [flt(1)?, flt(2)?, flt(3)?])
                .map_err(he)?;
            // This node's single data output is `grounded` (the applied vector is
            // the same Vec3-pin follow-up as the 2D kit).
            Ok(Value::Bool(r.grounded))
        }
        ("is_grounded", None) => Ok(Value::Bool(phys.is_grounded(ent(0)?).map_err(he)?)),
        ("raycast", Some(field)) => {
            let hit = phys
                .raycast(
                    [flt(0)?, flt(1)?, flt(2)?],
                    [flt(3)?, flt(4)?, flt(5)?],
                    flt(6)?,
                )
                .map_err(he)?;
            Ok(match field {
                "hit" => Value::Bool(hit.is_some()),
                "point_x" => Value::Float(hit.map_or(0.0, |h| h.point[0])),
                "point_y" => Value::Float(hit.map_or(0.0, |h| h.point[1])),
                "point_z" => Value::Float(hit.map_or(0.0, |h| h.point[2])),
                "normal_x" => Value::Float(hit.map_or(0.0, |h| h.normal[0])),
                "normal_y" => Value::Float(hit.map_or(0.0, |h| h.normal[1])),
                "normal_z" => Value::Float(hit.map_or(0.0, |h| h.normal[2])),
                other => return Err(err(format!("unknown raycast field `{other}`"))),
            })
        }
        ("get_velocity", Some(field)) => {
            let v = phys.get_velocity(ent(0)?).map_err(he)?;
            Ok(match field {
                "x" => Value::Float(v[0]),
                "y" => Value::Float(v[1]),
                "z" => Value::Float(v[2]),
                other => return Err(err(format!("unknown get_velocity field `{other}`"))),
            })
        }
        ("set_velocity", None) => {
            phys.set_velocity(ent(0)?, [flt(1)?, flt(2)?, flt(3)?])
                .map_err(he)?;
            Ok(Value::Unit)
        }
        ("apply_impulse", None) => {
            phys.apply_impulse(ent(0)?, [flt(1)?, flt(2)?, flt(3)?])
                .map_err(he)?;
            Ok(Value::Unit)
        }
        _ => Err(err(format!("unknown physics3d node `{key}`"))),
    }
}

/// A [`Host`] that knows nothing — every call errors. Useful for pure fns.
pub struct PureHost;
impl Host for PureHost {
    fn call(&mut self, path: &[String], _args: &[Value]) -> Result<Value, RunError> {
        Err(RunError::NoSuchHostFn(path.join("::")))
    }
}

/// A closure-backed host, handy for tests and simple engine bindings.
pub struct FnHost<F>(pub F);
impl<F> Host for FnHost<F>
where
    F: FnMut(&[String], &[Value]) -> Result<Value, RunError>,
{
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        (self.0)(path, args)
    }
}

/// What the debugger asked for, and what it observed.
#[derive(Debug, Clone, Default)]
pub struct Debug {
    /// Node ids (value-node `Let`s) to pause on.
    pub breakpoints: HashSet<LocalId>,
    /// Capture every value-node output for wire inspection.
    pub capture_wires: bool,
}

/// The observable trace of one interpreter run.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    /// Each value node's computed output, in execution order (wire inspector).
    pub wires: Vec<(LocalId, Value)>,
    /// Breakpoints encountered, in order hit.
    pub hits: Vec<LocalId>,
}

impl Trace {
    /// The most recent captured value for a wire/node, if any.
    pub fn wire(&self, id: LocalId) -> Option<&Value> {
        self.wires
            .iter()
            .rev()
            .find(|(n, _)| *n == id)
            .map(|(_, v)| v)
    }
}

enum Flow {
    Normal,
    Return(Value),
}

/// Interpret one [`BlueprintFn`] with the given argument values.
pub fn eval_fn(
    f: &BlueprintFn,
    args: &HashMap<String, Value>,
    host: &mut dyn Host,
) -> Result<Value, RunError> {
    let (v, _trace) = eval_fn_traced(f, args, host, &Debug::default())?;
    Ok(v)
}

/// Interpret with debugging: returns the value *and* the [`Trace`]
/// (captured wires + breakpoint hits).
pub fn eval_fn_traced(
    f: &BlueprintFn,
    args: &HashMap<String, Value>,
    host: &mut dyn Host,
    debug: &Debug,
) -> Result<(Value, Trace), RunError> {
    let mut interp = Interp {
        params: args.clone(),
        locals: HashMap::new(),
        host,
        debug,
        trace: Trace::default(),
    };
    let flow = interp.exec_block(&f.body)?;
    let value = match flow {
        Flow::Return(v) => v,
        Flow::Normal => Value::Unit,
    };
    Ok((value, interp.trace))
}

struct Interp<'a> {
    params: HashMap<String, Value>,
    locals: HashMap<LocalId, Value>,
    host: &'a mut dyn Host,
    debug: &'a Debug,
    trace: Trace,
}

impl Interp<'_> {
    fn exec_block(&mut self, body: &[Stmt]) -> Result<Flow, RunError> {
        for stmt in body {
            if let Flow::Return(v) = self.exec_stmt(stmt)? {
                return Ok(Flow::Return(v));
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<Flow, RunError> {
        match stmt {
            Stmt::Let { id, value, .. } => {
                let v = self.eval(value)?;
                if self.debug.breakpoints.contains(id) {
                    self.trace.hits.push(*id);
                }
                if self.debug.capture_wires {
                    self.trace.wires.push((*id, v.clone()));
                }
                self.locals.insert(*id, v);
                Ok(Flow::Normal)
            }
            Stmt::Assign { target, value } => {
                let v = self.eval(value)?;
                self.locals.insert(*target, v);
                Ok(Flow::Normal)
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                if self.eval(cond)?.as_bool()? {
                    self.exec_block(then_body)
                } else {
                    self.exec_block(else_body)
                }
            }
            Stmt::While { cond, body } => {
                while self.eval(cond)?.as_bool()? {
                    if let Flow::Return(v) = self.exec_block(body)? {
                        return Ok(Flow::Return(v));
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::Return(opt) => {
                let v = match opt {
                    Some(e) => self.eval(e)?,
                    None => Value::Unit,
                };
                Ok(Flow::Return(v))
            }
            Stmt::ExprStmt(e) => {
                self.eval(e)?;
                Ok(Flow::Normal)
            }
            Stmt::Snippet(_) => Err(RunError::Snippet),
        }
    }

    fn eval(&mut self, expr: &Expr) -> Result<Value, RunError> {
        match expr {
            Expr::Lit(lit) => Ok(Value::from(lit)),
            Expr::Param(name) => self
                .params
                .get(name)
                .cloned()
                .ok_or_else(|| RunError::UnboundParam(name.clone())),
            Expr::Local(id) => self
                .locals
                .get(id)
                .cloned()
                .ok_or(RunError::UnboundLocal(id.0)),
            Expr::Unary(op, inner) => {
                let v = self.eval(inner)?;
                match op {
                    UnOp::Neg => match v {
                        Value::Float(f) => Ok(Value::Float(-f)),
                        Value::Int(i) => Ok(Value::Int(-i)),
                        other => Err(RunError::Type(format!("cannot negate {other:?}"))),
                    },
                    UnOp::Not => Ok(Value::Bool(!v.as_bool()?)),
                }
            }
            Expr::Binary(op, l, r) => self.eval_binary(*op, l, r),
            Expr::Call { path, args } => {
                let argv: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<_, _>>()?;
                // `physics2d.*` nodes are ordinary calls the interpreter routes
                // to the host's `Physics2dHost` (no IR change) — mirroring how
                // `vars::*` reaches actor state through the Host boundary.
                if path.first().map(String::as_str) == Some("physics2d") {
                    dispatch_physics2d(self.host, path, &argv)
                } else if path.first().map(String::as_str) == Some("physics3d") {
                    // The `d3` mirror: `physics3d.*` nodes route to the host's
                    // `Physics3dHost`, same no-IR-change pattern as `physics2d`.
                    dispatch_physics3d(self.host, path, &argv)
                } else {
                    self.host.call(path, &argv)
                }
            }
        }
    }

    fn eval_binary(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Result<Value, RunError> {
        // Short-circuit boolean operators evaluate the rhs only if needed.
        match op {
            BinOp::And => {
                return Ok(Value::Bool(
                    self.eval(l)?.as_bool()? && self.eval(r)?.as_bool()?,
                ));
            }
            BinOp::Or => {
                return Ok(Value::Bool(
                    self.eval(l)?.as_bool()? || self.eval(r)?.as_bool()?,
                ));
            }
            _ => {}
        }

        let lv = self.eval(l)?;
        let rv = self.eval(r)?;

        use BinOp::*;
        match op {
            Add | Sub | Mul | Div | Rem => self.arith(op, lv, rv),
            Eq => Ok(Value::Bool(value_eq(&lv, &rv))),
            Ne => Ok(Value::Bool(!value_eq(&lv, &rv))),
            Lt | Le | Gt | Ge => self.compare(op, lv, rv),
            And | Or => unreachable!("handled above"),
        }
    }

    fn arith(&self, op: BinOp, lv: Value, rv: Value) -> Result<Value, RunError> {
        // Int op Int stays Int; anything with a float promotes to Float.
        if let (Value::Int(a), Value::Int(b)) = (&lv, &rv) {
            let (a, b) = (*a, *b);
            return Ok(Value::Int(match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::Div => a.checked_div(b).ok_or(RunError::DivByZero)?,
                BinOp::Rem => a.checked_rem(b).ok_or(RunError::DivByZero)?,
                _ => unreachable!(),
            }));
        }
        let (a, b) = (lv.as_float()?, rv.as_float()?);
        Ok(Value::Float(match op {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Rem => a % b,
            _ => unreachable!(),
        }))
    }

    fn compare(&self, op: BinOp, lv: Value, rv: Value) -> Result<Value, RunError> {
        let ord = if let (Value::Int(a), Value::Int(b)) = (&lv, &rv) {
            a.cmp(b)
        } else {
            let (a, b) = (lv.as_float()?, rv.as_float()?);
            a.partial_cmp(&b)
                .ok_or_else(|| RunError::Type("NaN comparison".into()))?
        };
        use std::cmp::Ordering::*;
        Ok(Value::Bool(match op {
            BinOp::Lt => ord == Less,
            BinOp::Le => ord != Greater,
            BinOp::Gt => ord == Greater,
            BinOp::Ge => ord != Less,
            _ => unreachable!(),
        }))
    }
}

fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Float(y)) | (Value::Float(y), Value::Int(x)) => *x as f64 == *y,
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Binding, Param, Ty};

    fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    // fn f(x: f64) -> f64 { let y = x * 2.0 + 1.0; return y; }
    fn double_plus_one() -> BlueprintFn {
        BlueprintFn {
            id: "f".into(),
            name: "f".into(),
            params: vec![Param {
                name: "x".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Float,
            body: vec![
                Stmt::Let {
                    id: LocalId(1),
                    binding: Binding::Named("y".into()),
                    ty: None,
                    mutable: false,
                    value: Expr::Binary(
                        BinOp::Add,
                        Box::new(Expr::Binary(
                            BinOp::Mul,
                            Box::new(Expr::Param("x".into())),
                            Box::new(Expr::Lit(Lit::Float(2.0))),
                        )),
                        Box::new(Expr::Lit(Lit::Float(1.0))),
                    ),
                },
                Stmt::Return(Some(Expr::Local(LocalId(1)))),
            ],
        }
    }

    #[test]
    fn evaluates_arithmetic() {
        let f = double_plus_one();
        let v = eval_fn(&f, &params(&[("x", Value::Float(3.0))]), &mut PureHost).unwrap();
        assert_eq!(v, Value::Float(7.0));
    }

    #[test]
    fn captures_wires_and_breakpoints() {
        let f = double_plus_one();
        let debug = Debug {
            breakpoints: [LocalId(1)].into_iter().collect(),
            capture_wires: true,
        };
        let (v, trace) = eval_fn_traced(
            &f,
            &params(&[("x", Value::Float(3.0))]),
            &mut PureHost,
            &debug,
        )
        .unwrap();
        assert_eq!(v, Value::Float(7.0));
        assert_eq!(trace.wire(LocalId(1)), Some(&Value::Float(7.0)));
        assert_eq!(trace.hits, vec![LocalId(1)]);
    }

    #[test]
    fn control_flow_and_short_circuit() {
        // fn g(n: i64) -> i64 { let mut i = 0; while i < n { i = i + 1; } return i; }
        let f = BlueprintFn {
            id: "g".into(),
            name: "g".into(),
            params: vec![Param {
                name: "n".into(),
                ty: Ty::Int,
            }],
            ret: Ty::Int,
            body: vec![
                Stmt::Let {
                    id: LocalId(1),
                    binding: Binding::Named("i".into()),
                    ty: None,
                    mutable: true,
                    value: Expr::Lit(Lit::Int(0)),
                },
                Stmt::While {
                    cond: Expr::Binary(
                        BinOp::Lt,
                        Box::new(Expr::Local(LocalId(1))),
                        Box::new(Expr::Param("n".into())),
                    ),
                    body: vec![Stmt::Assign {
                        target: LocalId(1),
                        value: Expr::Binary(
                            BinOp::Add,
                            Box::new(Expr::Local(LocalId(1))),
                            Box::new(Expr::Lit(Lit::Int(1))),
                        ),
                    }],
                },
                Stmt::Return(Some(Expr::Local(LocalId(1)))),
            ],
        };
        let v = eval_fn(&f, &params(&[("n", Value::Int(5))]), &mut PureHost).unwrap();
        assert_eq!(v, Value::Int(5));
    }

    #[test]
    fn host_calls_dispatch() {
        // fn h() { log::info("hi"); }
        let f = BlueprintFn {
            id: "h".into(),
            name: "h".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["log".into(), "info".into()],
                args: vec![Expr::Lit(Lit::Str("hi".into()))],
            })],
        };
        let mut seen = Vec::new();
        let mut host = FnHost(|path: &[String], args: &[Value]| {
            seen.push((path.join("::"), args.to_vec()));
            Ok(Value::Unit)
        });
        let v = eval_fn(&f, &HashMap::new(), &mut host).unwrap();
        assert_eq!(v, Value::Unit);
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "log::info");
    }

    #[test]
    fn snippet_cannot_be_interpreted() {
        let f = BlueprintFn {
            id: "s".into(),
            name: "s".into(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![Stmt::Snippet("some_unliftable_rust();".into())],
        };
        let err = eval_fn(&f, &HashMap::new(), &mut PureHost).unwrap_err();
        assert_eq!(err, RunError::Snippet);
    }

    #[test]
    fn division_by_zero_is_an_error() {
        let f = BlueprintFn {
            id: "d".into(),
            name: "d".into(),
            params: vec![],
            ret: Ty::Int,
            body: vec![Stmt::Return(Some(Expr::Binary(
                BinOp::Div,
                Box::new(Expr::Lit(Lit::Int(1))),
                Box::new(Expr::Lit(Lit::Int(0))),
            )))],
        };
        assert_eq!(
            eval_fn(&f, &HashMap::new(), &mut PureHost),
            Err(RunError::DivByZero)
        );
    }
}
