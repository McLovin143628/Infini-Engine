//! Ring-2 command surface for the Blueprint graph editor (ROADMAP P6.2).
//!
//! Holds the in-memory blueprint graph documents, the node registry (the
//! palette), and per-graph undo journals. The frontend drives everything
//! through these `graph_*` commands: fetch the registry, CRUD documents, apply
//! edits (optimistic mirror + authoritative re-validate), run a graph through
//! the interpreter, and read the generated Rust ("Open generated Rust").
//!
//! Types cross the wire as the `inf-graph`/`inf-blueprint` serde shapes
//! (camelCase), kept self-consistent within the blueprint feature. Edits
//! arrive as `inf_graph::GraphEdit` (the frontend hand-builds the kebab-case
//! tagged JSON); Tauri deserializes them directly.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use inf_blueprint::interp::{eval_fn, Host, RunError, Value};
use inf_blueprint::{blueprint_registry, lower_graph};
use inf_graph::{
    apply_edits, compile::validate, GraphDoc, GraphEdit, GraphIssue, GraphJournal, NodeDef,
    NodeRegistry,
};
use inf_transpile::{generate_file, BlueprintFile, FileEntry};
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

/// In-memory blueprint workspace.
struct GraphStore {
    registry: NodeRegistry,
    docs: BTreeMap<String, GraphDoc>,
    journals: BTreeMap<String, GraphJournal>,
    counter: u32,
}

pub struct GraphState {
    inner: Mutex<GraphStore>,
}

impl Default for GraphState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(GraphStore {
                registry: blueprint_registry(),
                docs: BTreeMap::new(),
                journals: BTreeMap::new(),
                counter: 0,
            }),
        }
    }
}

/// Result of applying an edit batch: structural issues + undo availability.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphApplyResult {
    pub issues: Vec<GraphIssue>,
    pub can_undo: bool,
    pub can_redo: bool,
}

/// Result of a preview run through the interpreter.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRunResult {
    /// Log lines from `debug.print` and engine calls, in execution order.
    pub logs: Vec<String>,
    /// Final member-variable values after the run.
    pub vars: BTreeMap<String, f64>,
    /// Handlers that were lowered and executed (by event key).
    pub handlers: Vec<String>,
    /// A fatal error, if the graph could not be lowered or run.
    pub error: Option<String>,
}

impl GraphState {
    fn with<R>(&self, f: impl FnOnce(&mut GraphStore) -> Result<R, String>) -> Result<R, String> {
        let mut store = self.inner.lock().map_err(|e| e.to_string())?;
        f(&mut store)
    }

    /// Drop a document + its undo journal from the workspace. Idempotent
    /// (closing an unknown id is a no-op).
    fn close(&self, id: &str) -> Result<(), String> {
        self.with(|s| {
            s.docs.remove(id);
            s.journals.remove(id);
            Ok(())
        })
    }
}

/// The node palette (registry) in insertion order.
#[tauri::command]
pub async fn graph_registry(state: State<'_, GraphState>) -> Result<Vec<NodeDef>, String> {
    state.with(|s| Ok(s.registry.dtos()))
}

/// All open graph documents.
#[tauri::command]
pub async fn graph_list(state: State<'_, GraphState>) -> Result<Vec<GraphDoc>, String> {
    state.with(|s| Ok(s.docs.values().cloned().collect()))
}

/// Create a new, empty blueprint graph.
#[tauri::command]
pub async fn graph_create(
    app: AppHandle,
    name: String,
    state: State<'_, GraphState>,
) -> Result<GraphDoc, String> {
    let doc = state.with(|s| {
        s.counter += 1;
        let id = format!("bp:{}", s.counter);
        let doc = GraphDoc {
            id: id.clone(),
            name: if name.is_empty() {
                format!("Blueprint {}", s.counter)
            } else {
                name
            },
            graph: inf_graph::Graph::empty(),
            viewport: None,
            modified_ms: 0,
        };
        s.docs.insert(id.clone(), doc.clone());
        s.journals.insert(id, GraphJournal::new(64));
        Ok(doc)
    })?;
    let _ = app.emit("graph://sync", doc.id.clone());
    Ok(doc)
}

/// Fetch one document.
#[tauri::command]
pub async fn graph_get(id: String, state: State<'_, GraphState>) -> Result<GraphDoc, String> {
    state.with(|s| {
        s.docs
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("no graph `{id}`"))
    })
}

/// Close a document: free its graph + undo journal (the whole memory an open
/// blueprint holds). Called when the editing surface is discarded so documents
/// don't accumulate for the life of the session.
#[tauri::command]
pub async fn graph_close(id: String, state: State<'_, GraphState>) -> Result<(), String> {
    state.close(&id)
}

/// Apply an edit batch (optimistic frontend already mirrored it). Records one
/// undo entry, re-validates, and returns issues + undo availability.
#[tauri::command]
pub async fn graph_apply(
    app: AppHandle,
    id: String,
    edits: Vec<GraphEdit>,
    label: String,
    state: State<'_, GraphState>,
) -> Result<GraphApplyResult, String> {
    let result = state.with(|s| {
        let registry = &s.registry;
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no graph `{id}`"))?;
        let journal = s.journals.get_mut(&id).ok_or("missing journal")?;
        journal.record(label, doc.graph.clone());
        apply_edits(&mut doc.graph, registry, &edits);
        let issues = validate(&doc.graph, registry);
        Ok(GraphApplyResult {
            issues,
            can_undo: journal.can_undo(),
            can_redo: journal.can_redo(),
        })
    })?;
    let _ = app.emit("graph://sync", id);
    Ok(result)
}

/// Undo the last edit group; returns the restored document.
#[tauri::command]
pub async fn graph_undo(
    app: AppHandle,
    id: String,
    state: State<'_, GraphState>,
) -> Result<Option<GraphDoc>, String> {
    let doc = state.with(|s| {
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no graph `{id}`"))?;
        let journal = s.journals.get_mut(&id).ok_or("missing journal")?;
        match journal.undo(doc.graph.clone()) {
            Some(prev) => {
                doc.graph = prev;
                Ok(Some(doc.clone()))
            }
            None => Ok(None),
        }
    })?;
    if doc.is_some() {
        let _ = app.emit("graph://sync", id);
    }
    Ok(doc)
}

/// Redo the last undone edit group.
#[tauri::command]
pub async fn graph_redo(
    app: AppHandle,
    id: String,
    state: State<'_, GraphState>,
) -> Result<Option<GraphDoc>, String> {
    let doc = state.with(|s| {
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no graph `{id}`"))?;
        let journal = s.journals.get_mut(&id).ok_or("missing journal")?;
        match journal.redo(doc.graph.clone()) {
            Some(next) => {
                doc.graph = next;
                Ok(Some(doc.clone()))
            }
            None => Ok(None),
        }
    })?;
    if doc.is_some() {
        let _ = app.emit("graph://sync", id);
    }
    Ok(doc)
}

/// Run the graph through the interpreter: lower each event handler, run
/// BeginPlay once then Tick three times, collecting logs + final variables.
#[tauri::command]
pub async fn graph_run(id: String, state: State<'_, GraphState>) -> Result<GraphRunResult, String> {
    state.with(|s| {
        let doc = s.docs.get(&id).ok_or_else(|| format!("no graph `{id}`"))?;
        let fns = match lower_graph(&doc.graph, &s.registry) {
            Ok(f) => f,
            Err(e) => {
                return Ok(GraphRunResult {
                    logs: Vec::new(),
                    vars: BTreeMap::new(),
                    handlers: Vec::new(),
                    error: Some(e.to_string()),
                })
            }
        };
        let mut host = RunHost::default();
        let mut handlers = Vec::new();
        let mut error = None;

        // BeginPlay once.
        for f in fns.iter().filter(|f| f.id == "begin_play") {
            handlers.push(f.id.clone());
            if let Err(e) = eval_fn(f, &HashMap::new(), &mut host) {
                error = Some(format!("begin_play: {e}"));
            }
        }
        // Tick three frames at 1/60 s.
        for f in fns.iter().filter(|f| f.id == "tick") {
            handlers.push(f.id.clone());
            for _ in 0..3 {
                let args: HashMap<String, Value> =
                    [("dt".to_string(), Value::Float(1.0 / 60.0))].into();
                if let Err(e) = eval_fn(f, &args, &mut host) {
                    error = Some(format!("tick: {e}"));
                    break;
                }
            }
        }

        let vars = host
            .vars
            .iter()
            .filter_map(|(k, v)| value_as_f64(v).map(|f| (k.clone(), f)))
            .collect();
        Ok(GraphRunResult {
            logs: host.logs,
            vars,
            handlers,
            error,
        })
    })
}

/// The Rust `inf-transpile` generates for this graph ("Open generated Rust").
#[tauri::command]
pub async fn graph_generate(id: String, state: State<'_, GraphState>) -> Result<String, String> {
    state.with(|s| {
        let doc = s.docs.get(&id).ok_or_else(|| format!("no graph `{id}`"))?;
        let fns = lower_graph(&doc.graph, &s.registry).map_err(|e| e.to_string())?;
        let file = BlueprintFile {
            entries: fns.into_iter().map(FileEntry::Blueprint).collect(),
        };
        generate_file(&file).map_err(|e| e.to_string())
    })
}

/// The interpreter host for a preview run: member variables live in a map
/// (unset reads default to 0.0), and engine/debug calls append to the log.
#[derive(Default)]
struct RunHost {
    vars: HashMap<String, Value>,
    logs: Vec<String>,
}

impl Host for RunHost {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        match (
            path.first().map(String::as_str),
            path.get(1).map(String::as_str),
        ) {
            (Some("vars"), Some("get")) => {
                let name = args.first().and_then(as_str).unwrap_or_default();
                Ok(self.vars.get(&name).cloned().unwrap_or(Value::Float(0.0)))
            }
            (Some("vars"), Some("set")) => {
                let name = args.first().and_then(as_str).unwrap_or_default();
                if let Some(v) = args.get(1) {
                    self.vars.insert(name, v.clone());
                }
                Ok(Value::Unit)
            }
            // `nodestate::*` backs the stateful flow nodes (do_once/flip_flop/
            // gate) with the same reserved `__bp_<kind>_<NodeId>` keys the
            // lowerer emits, stored in this run's var map. `get_or` returns the
            // stored value or the supplied default on a miss.
            (Some("nodestate"), Some("get_or")) => {
                let key = args.first().and_then(as_str).unwrap_or_default();
                let default = args.get(1).cloned().unwrap_or(Value::Unit);
                Ok(self.vars.get(&key).cloned().unwrap_or(default))
            }
            (Some("nodestate"), Some("set")) => {
                let key = args.first().and_then(as_str).unwrap_or_default();
                if let Some(v) = args.get(1) {
                    self.vars.insert(key, v.clone());
                }
                Ok(Value::Unit)
            }
            (Some("debug"), Some("print")) => {
                let msg = args.first().and_then(as_str).unwrap_or_default();
                self.logs.push(msg);
                Ok(Value::Unit)
            }
            _ => {
                self.logs
                    .push(format!("{}({})", path.join("::"), fmt_args(args)));
                Ok(Value::Unit)
            }
        }
    }
}

fn as_str(v: &Value) -> Option<String> {
    match v {
        Value::Str(s) => Some(s.clone()),
        _ => None,
    }
}

fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn fmt_args(args: &[Value]) -> String {
    args.iter()
        .map(|v| match v {
            Value::Float(f) => format!("{f}"),
            Value::Int(i) => format!("{i}"),
            Value::Bool(b) => format!("{b}"),
            Value::Str(s) => format!("{s:?}"),
            Value::Unit => "()".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Insert a bare document + journal directly (mimics `graph_create`, which
    /// needs a Tauri `State`/`AppHandle` we can't build in a unit test).
    fn seed(state: &GraphState, id: &str) {
        state
            .with(|s| {
                s.docs.insert(
                    id.to_string(),
                    GraphDoc {
                        id: id.to_string(),
                        name: "T".into(),
                        graph: inf_graph::Graph::empty(),
                        viewport: None,
                        modified_ms: 0,
                    },
                );
                s.journals.insert(id.to_string(), GraphJournal::new(8));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn close_frees_the_doc_and_journal() {
        let state = GraphState::default();
        seed(&state, "bp:1");
        state
            .with(|s| {
                assert_eq!(s.docs.len(), 1);
                assert_eq!(s.journals.len(), 1);
                Ok(())
            })
            .unwrap();

        state.close("bp:1").unwrap();

        state
            .with(|s| {
                assert!(s.docs.is_empty(), "doc freed");
                assert!(s.journals.is_empty(), "journal freed");
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn close_unknown_id_is_a_noop() {
        let state = GraphState::default();
        seed(&state, "bp:1");
        state.close("bp:does-not-exist").unwrap();
        state
            .with(|s| {
                assert_eq!(s.docs.len(), 1, "unrelated doc untouched");
                Ok(())
            })
            .unwrap();
    }
}
