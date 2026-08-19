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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Mutex;

use inf_blueprint::interp::{eval_fn, eval_fn_traced, Host, RunError, Trace, Value};
use inf_blueprint::{blueprint_registry, lower_graph, lower_graph_debug, InterpDebug, LowerMap};
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
    /// **How many of the submitted edits the backend REFUSED** (C4-44).
    ///
    /// Non-zero means the canvas and the graph disagree: the canvas is
    /// fully controlled, so it must re-derive from the backend rather than
    /// keep drawing a wire that does not exist.
    pub rejected: u32,
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

/// One handler of an actor class, as offered to the frontend.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActorHandlerDto {
    /// The `EventKind::key()` (or function name) — stable, and what `handler`
    /// takes on the way back in.
    pub key: String,
    /// Human label for the switcher ("Begin Play", "Tick", "jump", …).
    pub label: String,
    /// Whether this handler can be shown as a graph at all.
    pub raisable: bool,
    /// Why not, when it cannot. Never empty when `raisable` is false.
    pub reason: Option<String>,
}

/// `graph_open_actor`'s reply: the opened document plus the class's whole
/// handler list, so the canvas can offer a switcher instead of pretending the
/// actor has one graph.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActorGraphDto {
    pub doc: GraphDoc,
    /// The `.inf_act` GUID this document was raised from.
    pub asset_id: String,
    /// The class's display name.
    pub class_name: String,
    /// Which handler `doc` is showing.
    pub handler: String,
    pub handlers: Vec<ActorHandlerDto>,
}

/// **Open the blueprint OF an actor** (Wave E, batch B).
///
/// # What was missing
///
/// Every piece of this existed and none of them were joined. An entity carries
/// `ActorClass(Uuid)`; the asset decodes to a `BlueprintClass`; the class's
/// handlers are `BlueprintFn` IR; `inf_blueprint::raise_fn` turns IR back into a
/// `Graph` and is proven to invert lowering (`lower(raise(f)) == f` on
/// lowering's image) — and **nothing in Ring 2 had ever called it**. There was
/// no door from "this object in my level" to "the blueprint that runs it".
///
/// # Identity
///
/// The document id is `bp:<asset>` — **idempotent by asset**, mirroring
/// `dcc_open`'s `dcc:{id}`, and idempotent in the sense `dcc_open` means it:
/// a second open of the same actor on the same handler returns the document
/// already in flight rather than discarding its edits and its undo journal.
/// (The first Wave-E version rebuilt unconditionally, which the audit caught —
/// "idempotent" was true of the ID and false of the DOCUMENT.) Asking for a
/// *different* handler does rebuild: that is a deliberate switch, not a re-open.
/// It also retires, for actor graphs, the process-scoped `bp:{counter}`
/// identity `blueprintStore` documents at length: the same actor gets the same
/// document id in every session.
///
/// # Refusals are values
///
/// `raise` only inverts lowering's IMAGE. A handler written (or hand-edited)
/// into a shape with no unambiguous node form cannot be drawn, and the honest
/// answer is to say which handler and why — never an empty canvas. Every
/// handler is reported with `raisable` + `reason`, and a class where *nothing*
/// raises is an `Err` naming the first cause rather than an empty document.
#[tauri::command]
pub async fn graph_open_actor(
    app: AppHandle,
    asset_id: String,
    handler: Option<String>,
    state: State<'_, GraphState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<ActorGraphDto, String> {
    let id: inf_asset::AssetId = asset_id.parse().map_err(|e| format!("bad asset id: {e}"))?;
    let class = assets.load_blueprint_class_result(id)?;

    let candidates = actor_candidates(&class);
    if candidates.is_empty() {
        return Err(format!(
            "“{}” has no event handlers or functions to show as a graph.",
            class.name
        ));
    }
    let handlers = actor_handlers(&candidates);
    let chosen = choose_handler(&class.name, &candidates, &handlers, handler.as_deref())?;

    let doc_id = format!("bp:{id}");
    let doc_name = actor_doc_name(&class.name, &chosen.1);

    // Idempotent by asset AND handler: a re-open of what is already showing
    // hands back the live document. Rebuilding would silently throw away the
    // author's in-flight edits and their undo stack — `dcc_open` says so at its
    // own signature and this is its twin.
    if let Some(existing) = state.with(|s| Ok(reusable_actor_doc(s, &doc_id, &doc_name)))? {
        return Ok(ActorGraphDto {
            doc: existing,
            asset_id: id.to_string(),
            class_name: class.name.clone(),
            handler: chosen.0.clone(),
            handlers,
        });
    }

    let graph = inf_blueprint::raise_fn(chosen.2).map_err(|e| {
        format!(
            "“{}” ▸ {} cannot be drawn as a graph ({e}); open the generated code instead.",
            class.name, chosen.1
        )
    })?;

    let doc = state.with(|s| {
        let doc = GraphDoc {
            id: doc_id.clone(),
            name: doc_name,
            graph,
            viewport: None,
            modified_ms: 0,
        };
        s.docs.insert(doc_id.clone(), doc.clone());
        // A fresh journal: the document's content just changed wholesale, and an
        // undo stack from the previous handler would restore that handler's
        // nodes into this one.
        s.journals.insert(doc_id.clone(), GraphJournal::new(64));
        Ok(doc)
    })?;
    let _ = app.emit("graph://sync", doc.id.clone());

    Ok(ActorGraphDto {
        doc,
        asset_id: id.to_string(),
        class_name: class.name.clone(),
        handler: chosen.0.clone(),
        handlers,
    })
}

/// One openable handler: `(stable key, display label, body)`.
///
/// The key is what the frontend's switcher sends back, so it must not be the
/// label — two `Custom` events could share a label after a rename, never a key.
type ActorCandidate<'a> = (String, String, &'a inf_blueprint::BlueprintFn);

/// The document title for a raised handler. One spelling, because it is also
/// the equality test the idempotent re-open above uses.
fn actor_doc_name(class_name: &str, handler_label: &str) -> String {
    format!("{class_name} ▸ {handler_label}")
}

/// The live document for `doc_id`, but **only when it is already showing
/// `doc_name`** — i.e. this actor, on this handler.
///
/// Pulled out of the command so the idempotency claim has an arm (Wave E audit,
/// A3): re-opening must hand back the document in flight, edits and undo
/// journal intact, exactly as `dcc_open` does. Switching handler is a different
/// title and therefore a genuine rebuild, which is what the switcher wants.
fn reusable_actor_doc(s: &GraphStore, doc_id: &str, doc_name: &str) -> Option<GraphDoc> {
    s.docs.get(doc_id).filter(|d| d.name == doc_name).cloned()
}

/// Every handler a class offers, **events first then member functions**, in the
/// order the class declares them — so the switcher is stable across sessions.
///
/// Pulled out of the command (Wave E audit, A3) so the routing rules are
/// testable: the command itself needs a Tauri `State` and an `AppHandle` and
/// therefore had no arm at all.
fn actor_candidates(class: &inf_blueprint::BlueprintClass) -> Vec<ActorCandidate<'_>> {
    let mut out: Vec<ActorCandidate<'_>> = Vec::new();
    for ev in &class.events {
        out.push((ev.event.key().to_string(), event_label(&ev.event), &ev.body));
    }
    for f in &class.functions {
        out.push((f.id.clone(), f.name.clone(), f));
    }
    out
}

/// Report EVERY handler with `raisable`/`reason`.
///
/// A handler that cannot be drawn is reported, not dropped: hiding it would
/// make the actor look like it has fewer handlers than it has, and the user
/// would never learn that the one they are looking for is hand-edited past the
/// node kit's image.
fn actor_handlers(candidates: &[ActorCandidate<'_>]) -> Vec<ActorHandlerDto> {
    candidates
        .iter()
        .map(|(key, label, body)| match inf_blueprint::raise_fn(body) {
            Ok(_) => ActorHandlerDto {
                key: key.clone(),
                label: label.clone(),
                raisable: true,
                reason: None,
            },
            Err(e) => ActorHandlerDto {
                key: key.clone(),
                label: label.clone(),
                raisable: false,
                reason: Some(format!(
                    "{e} — this handler has no unambiguous node form; open the generated code \
                     instead."
                )),
            },
        })
        .collect()
}

/// Pick the handler to draw: the one asked for, else the first that **can** be
/// drawn — so a class where two of three handlers raise opens on one of the two
/// rather than failing whole.
///
/// `Err` only when the request names a handler the class does not have, or when
/// nothing at all can be drawn; the latter names the first cause instead of
/// handing back an empty canvas.
fn choose_handler<'c>(
    class_name: &str,
    candidates: &'c [ActorCandidate<'c>],
    handlers: &[ActorHandlerDto],
    requested: Option<&str>,
) -> Result<&'c ActorCandidate<'c>, String> {
    match requested {
        Some(k) => candidates
            .iter()
            .find(|(key, _, _)| key == k)
            .ok_or_else(|| format!("“{class_name}” has no handler `{k}`.")),
        None => candidates
            .iter()
            .find(|(key, _, _)| handlers.iter().any(|h| &h.key == key && h.raisable))
            .ok_or_else(|| {
                let first = handlers
                    .iter()
                    .find_map(|h| h.reason.clone())
                    .unwrap_or_else(|| "no raisable handler".to_string());
                format!("No handler of “{class_name}” can be drawn as a graph: {first}")
            }),
    }
}

/// The switcher label for an event kind (its `key` is the identity).
fn event_label(kind: &inf_blueprint::semantics::EventKind) -> String {
    use inf_blueprint::semantics::EventKind as E;
    match kind {
        E::BeginPlay => "Begin Play".to_string(),
        E::Tick => "Tick".to_string(),
        E::Input(action) => format!("Input: {action}"),
        E::Collision => "Collision".to_string(),
        E::Custom(name) => format!("Custom: {name}"),
        other => other.key().to_string(),
    }
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
        // C4-44: `apply_edits` returns how many edits took effect, and all three
        // canvases discarded it. `apply_edit` answers `false` for a Connect that
        // would cycle, a missing node or an unknown param — so a fully-controlled
        // canvas kept drawing a wire the backend had rejected, `validate()`
        // reported nothing (the wire is not in the graph to be invalid), and the
        // command returned `Ok`.
        //
        // The snapshot is also taken BEFORE and recorded only when something
        // landed: journalling first left a dead undo step for a batch that
        // changed nothing.
        let before = doc.graph.clone();
        let applied = apply_edits(&mut doc.graph, registry, &edits);
        let rejected = edits.len() - applied;
        if applied > 0 {
            journal.record(label, before);
        }
        let issues = validate(&doc.graph, registry);
        Ok(GraphApplyResult {
            issues,
            can_undo: journal.can_undo(),
            can_redo: journal.can_redo(),
            rejected: rejected as u32,
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

/// One inspected wire in a debug run: the source `node`/`port` and its most
/// recent value, stringified for display (the frontend shows it as a pin chip).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DebugWire {
    /// The producer node id (`NodeId.0`).
    pub node: u32,
    /// The producer's output port.
    pub port: String,
    /// The captured value, stringified.
    pub value: String,
}

/// Result of a **debug** preview run (B-P4 tier A): which nodes a breakpoint
/// paused on, every captured wire value, plus the same logs/vars a normal run
/// returns.
///
/// **Trace semantics** — a preview run is milliseconds, so there is no live
/// "pause": the graph runs to completion (BeginPlay + 3× Tick) under the
/// interpreter's trace, and the debugger *displays* the collected hits + wire
/// values post-hoc. Live pause-on-hit is the Simulate seam (tier A′), not this.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DebugRunResult {
    /// Node ids (canvas `NodeId`s) a breakpoint was hit on, ascending & unique.
    pub hits: Vec<u32>,
    /// Every captured wire's latest value (sorted by node then port).
    pub wires: Vec<DebugWire>,
    /// Log lines from `debug.print` and engine calls, in execution order.
    pub logs: Vec<String>,
    /// Final member-variable values after the run.
    pub vars: BTreeMap<String, f64>,
    /// Handlers that were lowered and executed (by event key).
    pub handlers: Vec<String>,
    /// A fatal error, if the graph could not be lowered or run.
    pub error: Option<String>,
}

/// Stringify a runtime [`Value`] for wire-inspector display.
fn value_display(v: &Value) -> String {
    match v {
        Value::Float(f) => format!("{f}"),
        Value::Int(i) => format!("{i}"),
        Value::Bool(b) => format!("{b}"),
        Value::Str(s) => format!("{s:?}"),
        Value::Unit => "()".to_string(),
    }
}

/// The per-handler [`InterpDebug`] for a debug run: arm every `LocalId` whose
/// originating `NodeId` is a requested breakpoint, and capture wires on request.
fn per_fn_debug(map: &LowerMap, breakpoints: &HashSet<u32>, capture: bool) -> InterpDebug {
    InterpDebug {
        breakpoints: map
            .locals
            .iter()
            .filter(|(_, (node, _))| breakpoints.contains(&node.0))
            .map(|(lid, _)| *lid)
            .collect(),
        capture_wires: capture,
    }
}

/// Project one handler's [`Trace`] back onto the canvas through its [`LowerMap`]:
/// breakpoint hits → node ids, captured wires → latest `(node, port) → value`.
fn project_trace(
    map: &LowerMap,
    trace: &Trace,
    hits: &mut BTreeSet<u32>,
    wires: &mut BTreeMap<(u32, String), String>,
) {
    for lid in &trace.hits {
        if let Some((node, _)) = map.locals.get(lid) {
            hits.insert(node.0);
        }
    }
    for (lid, val) in &trace.wires {
        if let Some((node, port)) = map.locals.get(lid) {
            wires.insert((node.0, port.clone()), value_display(val));
        }
    }
}

/// The debug-run core (sync, testable without a Tauri `State`): lower the graph
/// under **debug lowering** + provenance, run BeginPlay once then Tick 3×, and
/// collect breakpoint hits + wire values translated back to canvas node ids.
fn debug_run_store(
    store: &GraphStore,
    id: &str,
    breakpoints: &[u32],
    capture: bool,
) -> Result<DebugRunResult, String> {
    let doc = store
        .docs
        .get(id)
        .ok_or_else(|| format!("no graph `{id}`"))?;
    let lowered = match lower_graph_debug(&doc.graph, &store.registry) {
        Ok(v) => v,
        Err(e) => {
            return Ok(DebugRunResult {
                error: Some(e.to_string()),
                ..Default::default()
            })
        }
    };
    let bp: HashSet<u32> = breakpoints.iter().copied().collect();
    let mut host = RunHost::default();
    let mut handlers = Vec::new();
    let mut error = None;
    let mut hit_nodes: BTreeSet<u32> = BTreeSet::new();
    let mut wire_latest: BTreeMap<(u32, String), String> = BTreeMap::new();

    // BeginPlay once.
    for (f, map) in lowered.iter().filter(|(f, _)| f.id == "begin_play") {
        handlers.push(f.id.clone());
        let dbg = per_fn_debug(map, &bp, capture);
        match eval_fn_traced(f, &HashMap::new(), &mut host, &dbg) {
            Ok((_v, trace)) => project_trace(map, &trace, &mut hit_nodes, &mut wire_latest),
            Err(e) => error = Some(format!("begin_play: {e}")),
        }
    }
    // Tick three frames at 1/60 s.
    for (f, map) in lowered.iter().filter(|(f, _)| f.id == "tick") {
        handlers.push(f.id.clone());
        let dbg = per_fn_debug(map, &bp, capture);
        for _ in 0..3 {
            let args: HashMap<String, Value> =
                [("dt".to_string(), Value::Float(1.0 / 60.0))].into();
            match eval_fn_traced(f, &args, &mut host, &dbg) {
                Ok((_v, trace)) => project_trace(map, &trace, &mut hit_nodes, &mut wire_latest),
                Err(e) => {
                    error = Some(format!("tick: {e}"));
                    break;
                }
            }
        }
    }

    let vars = host
        .vars
        .iter()
        .filter_map(|(k, v)| value_as_f64(v).map(|f| (k.clone(), f)))
        .collect();
    let wires = wire_latest
        .into_iter()
        .map(|((node, port), value)| DebugWire { node, port, value })
        .collect();
    Ok(DebugRunResult {
        hits: hit_nodes.into_iter().collect(),
        wires,
        logs: host.logs,
        vars,
        handlers,
        error,
    })
}

/// Run the graph under **debug lowering** with the given canvas-node breakpoints
/// (B-P4 tier A). `breakpoints` are `NodeId`s; `capture` enables wire inspection.
/// Returns the hits (as node ids), captured wire values, logs, and final vars.
#[tauri::command]
pub async fn graph_debug_run(
    id: String,
    breakpoints: Vec<u32>,
    capture: bool,
    state: State<'_, GraphState>,
) -> Result<DebugRunResult, String> {
    state.with(|s| debug_run_store(s, &id, &breakpoints, capture))
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
            // Wave 3 event dispatchers (`dispatch.*` → `event::*`): the graph
            // preview run has no live actor world or dispatch queue, so they are
            // log-only here (the real firing happens in the sim's `drain_dispatch`).
            (Some("event"), Some("dispatch"))
            | (Some("event"), Some("bind"))
            | (Some("event"), Some("unbind")) => {
                self.logs
                    .push(format!("{}({})", path.join("::"), fmt_args(args)));
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

    // ── graph_open_actor's routing rules (Wave E audit, A3) ──────────────────
    //
    // The command needs a Tauri `State` and an `AppHandle` and so had no arm at
    // all; these cover the three claims that are actually load-bearing — the
    // handler ORDER, the partial-raise subset, and the all-refused message.

    use inf_blueprint::semantics::{EventBinding, EventKind};
    use inf_blueprint::{BlueprintClass, BlueprintFn, Ty};

    /// A handler whose body raises cleanly (an empty body is in lowering's
    /// image: the event node alone).
    fn raisable_fn(id: &str) -> BlueprintFn {
        BlueprintFn {
            id: id.to_string(),
            name: id.to_string(),
            params: vec![],
            ret: Ty::Unit,
            body: vec![],
        }
    }

    /// A handler that is NOT in lowering's image: a hand-written Rust snippet
    /// has no unambiguous node form, which is exactly the case the DTO's
    /// `raisable`/`reason` pair exists for.
    fn unraisable_fn(id: &str) -> BlueprintFn {
        BlueprintFn {
            body: vec![inf_blueprint::Stmt::Snippet("self.x += 1;".into())],
            ..raisable_fn(id)
        }
    }

    fn class_with(events: Vec<EventBinding>, functions: Vec<BlueprintFn>) -> BlueprintClass {
        let mut c = BlueprintClass::new("cls", "Turret");
        c.events = events;
        c.functions = functions;
        c
    }

    /// Events before functions, declaration order preserved, keys stable.
    #[test]
    fn actor_candidates_lists_events_then_functions_in_declaration_order() {
        let class = class_with(
            vec![
                EventBinding {
                    event: EventKind::Tick,
                    body: raisable_fn("tick"),
                },
                EventBinding {
                    event: EventKind::BeginPlay,
                    body: raisable_fn("begin_play"),
                },
            ],
            vec![raisable_fn("fire")],
        );
        let c = actor_candidates(&class);
        assert_eq!(
            c.iter().map(|(k, _, _)| k.as_str()).collect::<Vec<_>>(),
            ["tick", "begin_play", "fire"]
        );
        assert_eq!(c[0].1, "Tick");
        assert_eq!(c[1].1, "Begin Play");
    }

    /// **Two of three raise: the doc opens on one of the two, and the third is
    /// reported with a reason — not hidden, and not a whole-class failure.**
    #[test]
    fn a_partly_raisable_class_opens_on_a_raisable_handler_and_names_the_rest() {
        let class = class_with(
            vec![
                EventBinding {
                    event: EventKind::BeginPlay,
                    body: unraisable_fn("begin_play"),
                },
                EventBinding {
                    event: EventKind::Tick,
                    body: raisable_fn("tick"),
                },
            ],
            vec![raisable_fn("fire")],
        );
        let candidates = actor_candidates(&class);
        let handlers = actor_handlers(&candidates);

        // All three are OFFERED — a hidden handler teaches the user nothing.
        assert_eq!(handlers.len(), 3);
        assert!(!handlers[0].raisable);
        let reason = handlers[0].reason.as_deref().expect("a refusal is a value");
        assert!(!reason.trim().is_empty());
        assert!(reason.contains("no unambiguous node form"), "{reason}");
        assert!(handlers[1].raisable && handlers[1].reason.is_none());
        assert!(handlers[2].raisable);

        // …and the default open lands on the first one that can be DRAWN, not
        // the first one declared.
        let chosen = choose_handler("Turret", &candidates, &handlers, None).unwrap();
        assert_eq!(chosen.0, "tick");
        assert!(inf_blueprint::raise_fn(chosen.2).is_ok());
    }

    /// A class where nothing raises refuses **by name**, carrying the first
    /// cause — never an empty canvas.
    #[test]
    fn a_class_where_nothing_raises_refuses_with_the_first_cause() {
        let class = class_with(
            vec![EventBinding {
                event: EventKind::Tick,
                body: unraisable_fn("tick"),
            }],
            vec![],
        );
        let candidates = actor_candidates(&class);
        let handlers = actor_handlers(&candidates);
        let err = choose_handler("Turret", &candidates, &handlers, None)
            .expect_err("nothing raisable must refuse");
        assert!(err.contains("No handler of “Turret”"), "{err}");
        assert!(err.contains("can be drawn as a graph"), "{err}");
        // The first CAUSE travels with the refusal — "nothing raised" alone
        // sends the user looking for a bug in the editor.
        assert!(err.contains("no unambiguous node form"), "{err}");
    }

    /// An explicit request for a handler the class does not have is refused by
    /// name rather than silently falling back to another one.
    #[test]
    fn an_unknown_handler_request_is_refused_by_name() {
        let class = class_with(
            vec![EventBinding {
                event: EventKind::Tick,
                body: raisable_fn("tick"),
            }],
            vec![],
        );
        let candidates = actor_candidates(&class);
        let handlers = actor_handlers(&candidates);
        let err = choose_handler("Turret", &candidates, &handlers, Some("custom:nope"))
            .expect_err("an unknown handler must refuse");
        assert!(err.contains("custom:nope"), "{err}");
        // …and the request that DOES name a handler resolves to that one, even
        // when it is not the first raisable — that is what the switcher needs.
        let chosen = choose_handler("Turret", &candidates, &handlers, Some("tick")).unwrap();
        assert_eq!(chosen.0, "tick");
    }

    /// **Re-opening the same actor on the same handler keeps the live document**
    /// (Wave E audit, A3) — edits and undo journal intact, as `dcc_open` says at
    /// its own signature. The first Wave-E version rebuilt unconditionally, so
    /// "idempotent by asset" was true of the ID and false of the DOCUMENT.
    #[test]
    fn re_opening_the_same_handler_reuses_the_live_document() {
        let state = GraphState::default();
        let doc_id = "bp:11111111-1111-1111-1111-111111111111";
        let name = actor_doc_name("Turret", "Tick");
        seed(&state, doc_id);
        // Give the seeded doc the title a re-open would compute, plus a node the
        // author "added" — the thing a rebuild would throw away.
        state
            .with(|s| {
                let d = s.docs.get_mut(doc_id).unwrap();
                d.name = name.clone();
                d.graph.insert("event.tick", inf_graph::NodeUi::default());
                Ok(())
            })
            .unwrap();

        let reused = state
            .with(|s| Ok(reusable_actor_doc(s, doc_id, &name)))
            .unwrap()
            .expect("the same actor on the same handler must reuse");
        assert_eq!(reused.graph.nodes.len(), 1, "the author's work survives");

        // A DIFFERENT handler is a different title, so it rebuilds — which is
        // what the switcher asks for, and the reason the test is on the title
        // rather than on the id alone.
        assert!(
            state
                .with(|s| Ok(reusable_actor_doc(
                    s,
                    doc_id,
                    &actor_doc_name("Turret", "Begin Play")
                )))
                .unwrap()
                .is_none(),
            "switching handler must rebuild, not hand back the other handler's nodes"
        );
        // …and an actor with no document open yet has nothing to reuse.
        assert!(state
            .with(|s| Ok(reusable_actor_doc(s, "bp:nope", &name)))
            .unwrap()
            .is_none());
    }

    /// The re-open equality test is the document NAME, so it must be built by
    /// one function — a second `format!` at the comparison site is how an
    /// "idempotent" open silently stops being one.
    #[test]
    fn the_document_name_has_exactly_one_spelling() {
        assert_eq!(actor_doc_name("Turret", "Tick"), "Turret ▸ Tick");
        let src = include_str!("graph.rs");
        assert_eq!(
            src.matches("format!(\"{class_name} ▸ {handler_label}\")")
                .count(),
            1,
            "the class ▸ handler title must be minted only by `actor_doc_name`"
        );
    }

    fn wire(
        g: &mut inf_graph::Graph,
        from: inf_graph::NodeId,
        fp: &str,
        to: inf_graph::NodeId,
        tp: &str,
    ) {
        g.links.push(inf_graph::Link {
            from,
            from_port: fp.into(),
            to,
            to_port: tp.into(),
        });
    }

    /// A debug run over `begin_play → var.set("out", get("x") + 1)`: breakpoint on
    /// the `+` node, capture on. Its `LocalId` hit + wire value must translate back
    /// to the `+` node id, and the run's vars/logs must still be produced.
    #[test]
    fn debug_run_translates_breakpoints_and_wires_to_node_ids() {
        use inf_graph::{NodeUi, ParamValue};
        let mut g = inf_graph::Graph::empty();
        let bp = g.insert("event.begin_play", NodeUi::default());
        let getx = g.insert("var.get", NodeUi::default());
        g.node_mut(getx)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("x".into()));
        let one = g.insert("lit.int", NodeUi::default());
        g.node_mut(one)
            .unwrap()
            .params
            .insert("value".into(), ParamValue::Int(1));
        let add = g.insert("math.add", NodeUi::default());
        let setv = g.insert("var.set", NodeUi::default());
        g.node_mut(setv)
            .unwrap()
            .params
            .insert("name".into(), ParamValue::Text("out".into()));
        wire(&mut g, getx, "value", add, "a");
        wire(&mut g, one, "value", add, "b");
        wire(&mut g, add, "out", setv, "value");
        wire(&mut g, bp, "then", setv, "exec");

        let state = GraphState::default();
        state
            .with(|s| {
                s.docs.insert(
                    "bp:1".into(),
                    GraphDoc {
                        id: "bp:1".into(),
                        name: "T".into(),
                        graph: g.clone(),
                        viewport: None,
                        modified_ms: 0,
                    },
                );
                Ok(())
            })
            .unwrap();

        let res = state
            .with(|s| debug_run_store(s, "bp:1", &[add.0], true))
            .unwrap();
        assert!(res.error.is_none(), "run error: {:?}", res.error);
        assert!(
            res.hits.contains(&add.0),
            "breakpoint on `+` should hit: {:?}",
            res.hits
        );
        let add_wire = res
            .wires
            .iter()
            .find(|w| w.node == add.0 && w.port == "out");
        assert!(add_wire.is_some(), "captured `+` wire: {:?}", res.wires);
        assert_eq!(add_wire.unwrap().value, "1");
        assert_eq!(res.vars.get("out"), Some(&1.0));
    }
}
