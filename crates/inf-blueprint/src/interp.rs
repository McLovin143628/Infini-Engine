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
                self.host.call(path, &argv)
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
