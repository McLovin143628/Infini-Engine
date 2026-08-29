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
use inf_editor_core::ipc::GeneratedSourceDto;
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

/// What separates a class from its handler in a document title. **One
/// spelling**, and it is a constant because it is read from two directions:
/// `actor_doc_name` joins with it and `class_of_doc_name` splits on it.
///
/// It was two spellings. The join used this character and the split used an
/// em dash and an ASCII hyphen, so nothing ever split: the CODE tab wrote
/// `<Class><Handler>_<guid8>.rs`, one file per HANDLER rather than the one
/// file per class the convention is, and a class with a hyphen in its name
/// (`Fire-Door`) was truncated at the hyphen for good measure.
const ACTOR_DOC_SEPARATOR: &str = " ▸ ";

/// The document title for a raised handler. One spelling, because it is also
/// the equality test the idempotent re-open above uses.
fn actor_doc_name(class_name: &str, handler_label: &str) -> String {
    format!("{class_name}{ACTOR_DOC_SEPARATOR}{handler_label}")
}

/// The class half of a document title — the inverse of [`actor_doc_name`],
/// and asserted to be.
///
/// `rsplit_once`, so a class name that itself contains the separator keeps
/// all of itself and the handler half — always the last segment — is what
/// is dropped.
fn class_of_doc_name(name: &str) -> &str {
    name.rsplit_once(ACTOR_DOC_SEPARATOR)
        .map_or(name, |(class, _)| class)
        .trim()
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

/// **The unit-local functions a document's class declares** (wave SCRIPT2b).
///
/// An actor document's id is `bp:<asset>` — `graph_open_actor` says so at its
/// own signature and the identity is deliberate — so the class behind a preview
/// is recoverable from the id alone, with no second map to keep in step.
/// Anything else (a scratch `bp:{counter}` document from the palette) has no
/// class and therefore no local functions, which is `Vec::new()` rather than an
/// error: a preview of a scratch graph is not broken.
///
/// `load_blueprint_class` (the `Option` twin) rather than the `Result` one: a
/// class that will not load already logs its reason, and a PREVIEW that refused
/// to run because the asset moved would be a worse answer than a preview whose
/// local calls fall through the way they did before this existed.
fn actor_local_fns(
    doc_id: &str,
    assets: &super::assets::AssetState,
) -> Vec<inf_blueprint::BlueprintFn> {
    let Some(rest) = doc_id.strip_prefix("bp:") else {
        return Vec::new();
    };
    let Ok(asset) = rest.parse::<inf_asset::AssetId>() else {
        return Vec::new();
    };
    assets
        .load_blueprint_class(asset)
        .map(|c| c.functions)
        .unwrap_or_default()
}

/// Run lowered handlers the way the engine runs them: **`LocalFns` outermost**,
/// over the host that owns member variables.
///
/// This is `inf_blueprint::semantics::run_event`'s composition, and it is here
/// for the reason that function states at its own body: a local call recurses
/// through the OUTERMOST host, so whatever answers `vars::*` has to be inside
/// the thing that answers a one-segment call. The preview had neither layer
/// stacked — it called the interpreter directly — so a class holding a call
/// form previewed with its own function falling through to `RunHost`'s
/// unknown-call arm, which logs the path and answers `Unit`. The canvas cannot
/// draw such a class today (`raise` refuses the form by name), and "cannot be
/// reached today" is not the same claim as "cannot be reached".
fn run_handlers(
    fns: &[inf_blueprint::BlueprintFn],
    functions: &[inf_blueprint::BlueprintFn],
) -> GraphRunResult {
    let mut inner = RunHost::default();
    let mut handlers = Vec::new();
    let mut error = None;
    {
        let mut host = inf_blueprint::interp::LocalFns::new(&mut inner, functions);

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
    }

    let vars = inner
        .vars
        .iter()
        .filter_map(|(k, v)| value_as_f64(v).map(|f| (k.clone(), f)))
        .collect();
    GraphRunResult {
        logs: inner.logs,
        vars,
        handlers,
        error,
    }
}

/// Run the graph through the interpreter: lower each event handler, run
/// BeginPlay once then Tick three times, collecting logs + final variables.
#[tauri::command]
pub async fn graph_run(
    id: String,
    state: State<'_, GraphState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<GraphRunResult, String> {
    let functions = actor_local_fns(&id, &assets);
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
        Ok(run_handlers(&fns, &functions))
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
    functions: &[inf_blueprint::BlueprintFn],
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
    let mut inner = RunHost::default();
    let mut handlers = Vec::new();
    let mut error = None;
    let mut hit_nodes: BTreeSet<u32> = BTreeSet::new();
    let mut wire_latest: BTreeMap<(u32, String), String> = BTreeMap::new();
    {
        // The same stack `run_handlers` builds, for the same reason.
        let mut host = inf_blueprint::interp::LocalFns::new(&mut inner, functions);

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
    }

    let vars = inner
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
        logs: inner.logs,
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
    assets: State<'_, super::assets::AssetState>,
) -> Result<DebugRunResult, String> {
    let functions = actor_local_fns(&id, &assets);
    state.with(|s| debug_run_store(s, &id, &breakpoints, capture, &functions))
}

/// The Rust `inf-transpile` generates for this graph ("Open generated Rust").
#[tauri::command]
pub async fn graph_generate(id: String, state: State<'_, GraphState>) -> Result<String, String> {
    state.with(|s| Ok(generated_source(s, &id)?.1))
}

/// Lower a document and render it, plus the graph's **content hash**.
///
/// One function for `graph_generate` and `graph_write_source`, so the string in
/// the drawer and the bytes on disk cannot be two different renderings of one
/// graph — which is exactly what two call sites of `generate_file` would become.
fn generated_source(s: &GraphStore, id: &str) -> Result<(u64, String), String> {
    let doc = s.docs.get(id).ok_or_else(|| format!("no graph `{id}`"))?;
    let fns = lower_graph(&doc.graph, &s.registry).map_err(|e| e.to_string())?;
    let file = BlueprintFile {
        entries: fns.into_iter().map(FileEntry::Blueprint).collect(),
    };
    let rust = generate_file(&file).map_err(|e| e.to_string())?;
    // The GRAPH's hash, not the rendered text's: the stamp answers "does this
    // file describe this graph", and hashing the output would answer "does this
    // file describe this output", which is true of a file the generator itself
    // changed under.
    let bytes = serde_json::to_vec(&doc.graph).map_err(|e| e.to_string())?;
    Ok((inf_graph::content_hash(&bytes), rust))
}

/// **Write a blueprint class's generated Rust to disk, and say where.**
///
/// The CODE TAB's backend. Wave E shipped four object-editor tabs and no code
/// one, and said why: *"`graph_generate`'s output has never been written to a
/// file — inventing a path for it is a transpiler-workflow decision, not a UX
/// one."* The decision is `inf_editor_core::blueprint_source`, which owns the
/// convention, the banner, the staleness stamp and the write; this is the
/// wiring.
///
/// Idempotent and cheap on the hit path: a graph whose hash already matches the
/// file's banner is not rewritten, so opening the tab twice is one read.
#[tauri::command]
pub async fn graph_write_source(
    id: String,
    state: State<'_, GraphState>,
    project: State<'_, super::project::ProjectState>,
) -> Result<GeneratedSourceDto, String> {
    use inf_editor_core::blueprint_source as src;

    let root = project
        .current_root()
        .ok_or_else(|| "open a project before generating blueprint code".to_string())?;
    // The class name and the asset id come from the DOCUMENT id (`bp:<asset>`),
    // which `graph_open_actor` is documented as keying idempotently — so the
    // file and the tab agree about identity by construction rather than by two
    // lookups that could disagree.
    let (kind, asset_id) = id.split_once(':').unwrap_or(("", id.as_str()));
    if kind != "bp" {
        return Err(format!(
            "`{id}` is not an actor blueprint document, so it has no class to \
             generate code for"
        ));
    }
    let (hash, rust, name) = state.with(|s| {
        let (hash, rust) = generated_source(s, &id)?;
        let name = s
            .docs
            .get(&id)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| asset_id.to_string());
        Ok((hash, rust, name))
    })?;
    // The document name is "<Class> ▸ <handler>"; the FILE is per
    // class, so the handler half is dropped rather than sanitized into the
    // path.
    let class = class_of_doc_name(&name).to_string();
    let (path, what) =
        src::write_if_stale(&root, &class, asset_id, hash, &rust).map_err(|e| e.to_string())?;
    Ok(GeneratedSourceDto {
        path: path.to_string_lossy().into_owned(),
        wrote: !matches!(what, src::SourceWrite::Current),
        created: matches!(what, src::SourceWrite::Created),
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

    /// **The scaffolded index and the write door agree about the marker**
    /// (SCRIPT0 audit — the carried item, armed).
    ///
    /// `// @generated by Infini Engine` is written twice by two crates that
    /// cannot see each other. `inf_project::template::scaffold` (Ring 0) writes
    /// the *first* `src/blueprints/mod.rs` a new project ever has, as a literal;
    /// `inf_editor_core::blueprint_source` (Ring 1) owns `GENERATED_MARKER` and
    /// is what later decides whether that file is ours. Ring 0 cannot depend on
    /// Ring 1, so the two agreed **by hand** — and the SCRIPT0 rebrand moved
    /// both only because one pair of eyes happened to catch the second.
    ///
    /// Had it moved only one, nothing would have crashed: the Code tab would
    /// have refused, once, to update the index it needs — `SourceError::NotOurs`,
    /// in a message accusing the author of hand-writing a file the scaffolder
    /// wrote. This module is the one place in the workspace that can see both
    /// crates, so the agreement is asserted here, against the **world** (a real
    /// scaffold on disk) rather than against two constants.
    ///
    /// Asserted with `starts_with(GENERATED_MARKER)` rather than
    /// `is_generated`, deliberately: `is_generated` also accepts
    /// `LEGACY_GENERATED_MARKER`, so it would keep saying yes to a Ring-0
    /// literal that had been left behind at the old spelling — which is the
    /// exact drift this arm exists to catch.
    #[test]
    fn a_scaffolded_project_writes_an_index_the_code_tab_owns() {
        use inf_editor_core::blueprint_source as src;

        let parent = tempfile::tempdir().expect("a temp dir");
        let made = inf_project::template::scaffold(
            parent.path(),
            "MarkerAgreement",
            inf_project::template::ProjectTemplate::Blank3d,
        )
        .expect("the scaffolder writes a project");

        let index = made.root.join(src::BLUEPRINT_SRC_DIR).join("mod.rs");
        let text = std::fs::read_to_string(&index)
            .expect("a scaffolded project has a blueprint module index");
        assert!(
            text.starts_with(src::GENERATED_MARKER),
            "the scaffolder's banner and `GENERATED_MARKER` have drifted apart — \
             the Code tab will refuse to update the index a new project ships \
             with, and blame the author for it.\nRing 0 wrote: {:?}\nRing 1 \
             expects: {:?}",
            text.lines().next().unwrap_or_default(),
            src::GENERATED_MARKER
        );
        // …and the door itself agrees, which is the consequence that matters.
        assert!(src::is_generated(&text));
    }

    /// **The document title round-trips through its own separator** (audit fix).
    ///
    /// `graph_write_source` derives the CODE tab's file name from the class half
    /// of the title, and for a while it split on characters `actor_doc_name` has
    /// never joined with — so the "one file per class" convention was in fact
    /// one file per *handler*, and every class name containing an ASCII hyphen
    /// lost everything after it. The two directions are asserted against each
    /// other rather than against a literal, which is the only form that cannot
    /// drift.
    #[test]
    fn a_document_title_round_trips_to_its_class() {
        for (class, handler) in [
            ("Door", "Tick"),
            ("Fire-Door", "Begin Play"),
            ("Turret", "Custom: Fire"),
            ("A", "B"),
            ("Class With Spaces", "Tick"),
            // a class that contains the separator itself keeps all of itself
            ("Odd \u{25b8} Name", "Tick"),
        ] {
            let name = actor_doc_name(class, handler);
            assert_eq!(
                class_of_doc_name(&name),
                class,
                "{name:?} did not split back into its class"
            );
        }
        // …and a title with no separator at all is entirely the class, which is
        // the fallback `graph_write_source` relies on for a bare asset id.
        assert_eq!(class_of_doc_name("Lonely"), "Lonely");
    }

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
        // The separator is a CONSTANT now, because the title is read from
        // two directions and the reader used to spell it differently from
        // the writer. So the pin moved with it: one declaration, and one
        // site that mints a title.
        assert_eq!(
            src.matches("const ACTOR_DOC_SEPARATOR: &str = \" ▸ \";")
                .count(),
            1,
            "the separator must be declared exactly once"
        );
        assert_eq!(
            src.matches("format!(\"{class_name}").count(),
            1,
            "the class ▸ handler title must be minted only by `actor_doc_name`"
        );
        // …and exactly one site splits it back, which is the half that was
        // missing when the two spellings diverged.
        assert_eq!(
            src.matches(concat!("rsplit_once(ACTOR_DOC", "_SEPARATOR)"))
                .count(),
            1,
            "the title must be split back only by `class_of_doc_name`"
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
            .with(|s| debug_run_store(s, "bp:1", &[add.0], true, &[]))
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
    /// **The preview stacks `LocalFns`, so a call form calls the function**
    /// (wave SCRIPT2b, the SCRIPT2a audit's LOW 3).
    ///
    /// `graph_run` and the debug run both called the interpreter directly, with
    /// no host layered over `RunHost`. A class holding a unit-local call
    /// therefore previewed with that call landing on `RunHost`'s unknown-call
    /// arm — logged as `helper()` and answered `Unit` — while the SAME program
    /// in Simulate, in PIE and in the shipped player runs the function, because
    /// `semantics::run_event` stacks `LocalFns` outermost.
    ///
    /// The arm runs `run_handlers` — the exact function both previews now use —
    /// with and without the class's functions, and requires the two to differ:
    /// with them the variable holds what the function computed, and without
    /// them the call fell through and was logged. Asserting BOTH sides is what
    /// makes this a measurement rather than a hope; the second half is the
    /// behaviour that used to be the only one.
    #[test]
    fn the_preview_runs_a_units_own_functions() {
        // A RAW string with real newlines, deliberately: this fixture's first
        // draft used `\n`-continuations, a scripted edit ate them, and the arm
        // passed anyway — legal Rust, identical output, wrong source (the ninth
        // catch of that law, and this wave's own). A raw literal has no
        // backslash for anything to eat.
        let source = r#"actor "Preview"

function double(x: float) -> float
  return x * 2.0
end

on begin_play()
  out = double(21.0)
end
"#;
        let (class, _w) = inf_script::compile(source, "preview").expect("the fixture compiles");
        assert_eq!(
            class.functions.len(),
            1,
            "the fixture declares one function"
        );
        let handlers: Vec<_> = class.events.iter().map(|b| b.body.clone()).collect();

        let with = run_handlers(&handlers, &class.functions);
        assert_eq!(with.error, None, "the preview errored: {:?}", with.error);
        assert_eq!(
            with.vars.get("out"),
            Some(&42.0),
            "the call form did not reach the function: logs {:?}",
            with.logs
        );

        // …and the shape this replaces, so the arm cannot pass vacuously.
        let without = run_handlers(&handlers, &[]);
        assert_ne!(without.vars.get("out"), Some(&42.0));
        assert!(
            without.logs.iter().any(|l| l.starts_with("double(")),
            "without the functions the call must fall through to the logger: {:?}",
            without.logs
        );
    }

    /// A document id names its class, or it names nothing — and "nothing" is a
    /// scratch graph rather than an error.
    #[test]
    fn only_an_actor_document_can_have_local_functions() {
        // `graph_open_actor` mints `bp:<asset>`; the palette mints `bp:<counter>`.
        let id = inf_asset::AssetId(uuid::Uuid::from_u128(
            0x1234_5678_9abc_def0_1234_5678_9abc_def0,
        ));
        assert_eq!(
            format!("bp:{id}")
                .strip_prefix("bp:")
                .and_then(|r| r.parse::<inf_asset::AssetId>().ok()),
            Some(id),
        );
        for scratch in ["bp:1", "bp:", "mat:1", "bp:not-a-guid"] {
            assert!(
                scratch
                    .strip_prefix("bp:")
                    .and_then(|r| r.parse::<inf_asset::AssetId>().ok())
                    .is_none(),
                "{scratch} must not resolve to an asset"
            );
        }
    }
}
