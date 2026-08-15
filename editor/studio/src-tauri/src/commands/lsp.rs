//! Language intelligence (P5.2): a minimal hand-rolled LSP client over stdio,
//! targeting rust-analyzer.
//!
//! Ported in spirit from the CodeR reference (which hand-rolls JSON-RPC rather
//! than pulling tower-lsp). One server process per language; a dedicated reader
//! thread demultiplexes replies (to pending requests) from notifications
//! (`textDocument/publishDiagnostics` → the `lsp://diagnostics` event). The
//! frontend pulls completion/hover/definition via `lsp_request` and receives
//! diagnostics as a push event.
//!
//! rust-analyzer is resolved from PATH, then a per-user cache dir. The
//! self-contained auto-downloader (fetch the pinned GitHub release over HTTPS +
//! gunzip, with SHA-256 verification for servers that publish sidecars) is the
//! documented follow-up — it needs an HTTP client whose TLS root store clears
//! the license gate, which the MVP declines to take on. Until then the editor
//! surfaces a clear "install rust-analyzer" message when it isn't found (it
//! ships with every rustup toolchain via `rustup component add rust-analyzer`).
//! Also deferred: multi-server registry, code-lens, inlay hints,
//! rename/references, signature help.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, State};

/// `lsp://diagnostics` payload (typed on the frontend; not a ts-rs DTO because
/// the diagnostics blob is opaque LSP JSON).
#[derive(Clone, Serialize)]
struct DiagnosticsEvent {
    uri: String,
    diagnostics: Value,
}

struct LspServer {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    next_id: Arc<AtomicI64>,
    pending: Arc<Mutex<HashMap<i64, Sender<Value>>>>,
}

/// Managed state: one server per language key ("rust").
#[derive(Default)]
pub struct LspState {
    servers: Mutex<HashMap<String, LspServer>>,
    /// Per-document `textDocument/didChange` version counters (L7.L3).
    ///
    /// The version was the literal `2` on every change. LSP requires it to
    /// **increase** per document, and a server is entitled to discard an edit
    /// that does not — so a session's second keystroke onward was, by the letter
    /// of the protocol, droppable.
    doc_versions: Mutex<HashMap<String, i64>>,
}

impl LspState {
    /// The next `didChange` version for `uri` — 2, 3, 4, ... after `didOpen`'s 1.
    fn next_version(&self, uri: &str) -> i64 {
        let Ok(mut map) = self.doc_versions.lock() else {
            // A poisoned counter is not a reason to stop editing; 2 is what the
            // whole session used before this existed.
            return 2;
        };
        let v = map.entry(uri.to_string()).or_insert(1);
        *v += 1;
        *v
    }

    /// Reset a document's counter when it is opened (its `didOpen` is version 1).
    fn open_version(&self, uri: &str) {
        if let Ok(mut map) = self.doc_versions.lock() {
            map.insert(uri.to_string(), 1);
        }
    }

    /// Forget a closed document's counter — the map is otherwise unbounded over
    /// a session that opens many files.
    fn close_version(&self, uri: &str) {
        if let Ok(mut map) = self.doc_versions.lock() {
            map.remove(uri);
        }
    }
}

// ── framing ────────────────────────────────────────────────────────────────

fn write_message(stdin: &Arc<Mutex<ChildStdin>>, value: &Value) -> Result<(), String> {
    let body = serde_json::to_string(value).map_err(|e| e.to_string())?;
    let mut w = stdin.lock().map_err(|e| e.to_string())?;
    write!(w, "Content-Length: {}\r\n\r\n{}", body.len(), body).map_err(|e| e.to_string())?;
    w.flush().map_err(|e| e.to_string())
}

fn send_notify(stdin: &Arc<Mutex<ChildStdin>>, method: &str, params: Value) -> Result<(), String> {
    write_message(
        stdin,
        &json!({ "jsonrpc": "2.0", "method": method, "params": params }),
    )
}

/// Send a request and block (up to `timeout`) on the reader thread's reply.
fn send_request(
    stdin: &Arc<Mutex<ChildStdin>>,
    next_id: &Arc<AtomicI64>,
    pending: &Arc<Mutex<HashMap<i64, Sender<Value>>>>,
    method: &str,
    params: Value,
    timeout: Duration,
) -> Result<Value, String> {
    let id = next_id.fetch_add(1, Ordering::Relaxed);
    let (tx, rx) = channel::<Value>();
    pending.lock().map_err(|e| e.to_string())?.insert(id, tx);
    // **The entry is removed on the write failure too** (Hardening D). The
    // timeout arm below already does this; the `?` here used to leave the
    // `Sender` in the map for ever, so once the server's stdin is a broken pipe
    // — which it is for the rest of the session, since nothing respawns it —
    // every subsequent request leaked one more entry, and the map grew at the
    // rate the editor sends requests (one per keystroke on the completion path).
    if let Err(e) = write_message(
        stdin,
        &json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }),
    ) {
        pending.lock().ok().and_then(|mut p| p.remove(&id));
        return Err(e);
    }
    match rx.recv_timeout(timeout) {
        Ok(v) => Ok(v),
        Err(_) => {
            pending.lock().ok().and_then(|mut p| p.remove(&id));
            Err(format!("lsp request '{method}' timed out"))
        }
    }
}

/// [`send_request`], **off the async workers** (Hardening Wave E).
///
/// `send_request` ends in `Receiver::recv_timeout`, a real thread block for up
/// to `timeout`. Both callers were `async fn`s that invoked it directly, so a
/// rust-analyzer that was busy — or gone — parked a Tokio worker for five
/// seconds per keystroke on the completion path, and for **thirty** on
/// `lsp_start`'s `initialize`. Every one of the editor's 239 commands shares
/// that runtime.
///
/// `lsp_start` already did this for the PATH probe and the `Command::spawn`
/// beside it, with the rationale written down (*"keep them off the async
/// workers (mirrors git.rs / package.rs)"*) — and then blocked for thirty
/// seconds on the next statement. The handles are all `Arc`s, so the move is a
/// clone of three pointers.
async fn send_request_async(
    stdin: Arc<Mutex<ChildStdin>>,
    next_id: Arc<AtomicI64>,
    pending: Arc<Mutex<HashMap<i64, Sender<Value>>>>,
    method: String,
    params: Value,
    timeout: Duration,
) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        send_request(&stdin, &next_id, &pending, &method, params, timeout)
    })
    .await
    .map_err(|e| format!("lsp request task failed to run: {e}"))?
}

/// The largest LSP message body this client will allocate for (round-2
/// finding **B5**).
///
/// `Content-Length` is a number in a header written by a subprocess, parsed
/// into a `usize` and handed straight to `vec![0u8; len]` — the allocation
/// happens *before* a single body byte is read, so a server that emits
/// `Content-Length: 18446744073709551615` (or is simply confused after a
/// framing slip) aborts the editor on an allocation failure rather than
/// failing one request. Wave F hardened the framing-*error* path in this same
/// function and did not add the ceiling.
///
/// 64 MiB is far above anything real: rust-analyzer's largest ordinary message
/// is a `workspace/symbol` or a full-file semantic-token response, kilobytes to
/// low megabytes. A body past this is a broken stream, and the reader says so
/// and stops — which is what a framing error already does, because after one
/// the stream position is guesswork.
const MAX_LSP_BODY: usize = 64 * 1024 * 1024;

/// Read one framed message body from the server stream.
///
/// `Ok(None)` is EOF — the server exited and the reader thread stops. Every
/// other failure is `Err` (C4-45): a malformed frame used to return the same
/// `None` as EOF, so ONE unparseable message killed the reader thread and with
/// it every diagnostic for the rest of the session — silently, and
/// indistinguishably from "rust-analyzer closed".
fn read_message(reader: &mut impl BufRead) -> Result<Option<Value>, String> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("reading a header line: {e}"))?;
        if n == 0 {
            return Ok(None); // EOF
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length =
                Some(rest.trim().parse().map_err(|e| {
                    format!("Content-Length `{}` is not a length: {e}", rest.trim())
                })?);
        }
    }
    let Some(len) = content_length else {
        return Err("a message arrived with no Content-Length header".into());
    };
    if len > MAX_LSP_BODY {
        return Err(format!(
            "a message declared a {len}-byte body, past the {MAX_LSP_BODY}-byte ceiling; \
             the stream is not framed LSP"
        ));
    }
    let mut body = vec![0u8; len];
    reader
        .read_exact(&mut body)
        .map_err(|e| format!("reading a {len}-byte body: {e}"))?;
    serde_json::from_slice(&body).map(Some).map_err(|e| {
        format!(
            "a {len}-byte body is not JSON: {e}; it began {:?}",
            String::from_utf8_lossy(&body[..body.len().min(80)])
        )
    })
}

/// The reader thread: routes replies to pending requests, and
/// publishDiagnostics notifications to the `lsp://diagnostics` event.
fn reader_loop(
    app: AppHandle,
    stdout: std::process::ChildStdout,
    pending: Arc<Mutex<HashMap<i64, Sender<Value>>>>,
) {
    let mut reader = BufReader::new(stdout);
    loop {
        let msg = match read_message(&mut reader) {
            Ok(Some(msg)) => msg,
            Ok(None) => break, // the server exited
            Err(e) => {
                // A framing error leaves the stream position unknown, so
                // resynchronizing would be guesswork — but the reader stopping
                // in silence is what made this invisible. Say it, then stop.
                tracing::error!("lsp reader: {e}; diagnostics stop for this session");
                break;
            }
        };
        // **A reply is a message with an `id` and NO `method`** (round-2
        // finding B6). Discriminating on `id` alone is wrong in JSON-RPC:
        // server→client *requests* carry one too, and rust-analyzer sends
        // several (`client/registerCapability`, `workspace/configuration`,
        // `window/workDoneProgress/create`). Each used to take the reply
        // branch and `pending.remove(&id)`; the server's ids and ours both
        // start low and are both dense, so a collision **resolved a real
        // completion or hover with `Null`** — indistinguishable, at every
        // caller, from "the server had no results". The comment three lines
        // down described behaviour the code did not have.
        if let Some(id) = reply_id(&msg) {
            // A reply to one of our requests (result or error).
            let payload = msg
                .get("result")
                .cloned()
                .unwrap_or_else(|| msg.get("error").cloned().unwrap_or(Value::Null));
            if let Some(tx) = pending.lock().ok().and_then(|mut p| p.remove(&id)) {
                let _ = tx.send(payload);
            }
        } else if msg.get("method").and_then(Value::as_str)
            == Some("textDocument/publishDiagnostics")
        {
            if let Some(params) = msg.get("params") {
                let uri = params
                    .get("uri")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let diagnostics = params
                    .get("diagnostics")
                    .cloned()
                    .unwrap_or(Value::Array(vec![]));
                let _ = app.emit("lsp://diagnostics", DiagnosticsEvent { uri, diagnostics });
            }
        }
        // Server-to-client requests (e.g. `workspace/configuration`) are
        // ignored in this MVP; rust-analyzer tolerates the missing reply. They
        // reach here — rather than the reply branch above — because that
        // branch requires the message to carry no `method`.
    }
    // **The reader owns the answer, so it must own the end of it** (round-2
    // MED, the lsp pending-sender cluster).
    //
    // Every `Sender` lives in this map, not on the reader's stack, so dropping
    // the reader thread did not close a single channel: `send_request`'s
    // `recv_timeout` had no `Disconnected` to see and burned its FULL timeout
    // against a server that had already exited — five seconds per keystroke on
    // the completion path, thirty on `initialize`. Harmless until Wave E moved
    // `send_request` onto `spawn_blocking`; after that it is ~25 parked
    // blocking threads in the steady state, and once that pool queues, every
    // other `spawn_blocking` in the editor stalls behind it.
    //
    // Clearing the map drops the senders, so every waiter returns immediately
    // with the error it should have had.
    if let Ok(mut p) = pending.lock() {
        let stranded = p.len();
        p.clear();
        if stranded > 0 {
            tracing::warn!(stranded, "lsp reader stopped; in-flight requests released");
        }
    }
}

/// **Round-2 findings B5 and B6**, as two predicates a test can hold.
///
/// Both defects lived in code that needs a live subprocess to reach, which is
/// why neither had an arm: `read_message` takes a `BufRead` (so it is
/// drivable), but `reader_loop` takes a `ChildStdout` and an `AppHandle` (so
/// it is not). The routing rule is therefore lifted into a function *the loop
/// itself calls*, per the campaign's rule that a claim about behaviour needs
/// a door a test can open.
///
/// Returns the id to resolve, or `None` when the message is not a reply to one
/// of our requests.
fn reply_id(msg: &Value) -> Option<i64> {
    if msg.get("method").is_some() {
        return None;
    }
    msg.get("id").and_then(Value::as_i64)
}

// ── rust-analyzer resolution ────────────────────────────────────────────────

fn ra_exe_name() -> &'static str {
    if cfg!(windows) {
        "rust-analyzer.exe"
    } else {
        "rust-analyzer"
    }
}

/// Resolve a rust-analyzer executable: PATH → per-user cache. Returns a clear
/// install hint when neither is present (the auto-downloader is a follow-up —
/// see the module docs). Blocking; runs on the async command worker.
fn resolve_rust_analyzer(app: &AppHandle) -> Result<PathBuf, String> {
    // 1. On PATH? (ships with rustup: `rustup component add rust-analyzer`.)
    if Command::new("rust-analyzer")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Ok(PathBuf::from("rust-analyzer"));
    }

    // 2. A copy the user dropped in the app data cache dir.
    if let Ok(dir) = app.path().app_data_dir() {
        let cached = dir.join("language-servers").join(ra_exe_name());
        if cached.is_file() {
            return Ok(cached);
        }
    }

    Err("rust-analyzer not found on PATH. Install it with \
         `rustup component add rust-analyzer` (or place the binary in the app's \
         language-servers cache). Auto-download is a follow-up."
        .to_string())
}

/// `file://` URI from an absolute path, **fully percent-encoded** (L7.L3).
///
/// This escaped the space and nothing else, so a path containing `#`, `%`,
/// `?` or any non-ASCII character produced a URI the server read as a
/// *different* document — and `publishDiagnostics` then came back under a URI
/// the editor never asked about, so **every diagnostic for that file was
/// silently dropped**. `#` truncates the path at the fragment, a literal `%`
/// corrupts any later escape, and a non-ASCII byte is not legal in a URI at
/// all.
///
/// Windows drive paths still come out as `file:///C:/…`.
///
/// # The TypeScript twin (round-2 findings R2.F3 / B4)
///
/// `lspBridge.ts` restates this encoder by hand, under a comment reading
/// *"Match the backend `path_to_uri`"*, because the frontend keys its
/// diagnostics store by URI and compares the two spellings for equality. When
/// this function grew full RFC-3986 escaping the twin was left at spaces-only,
/// so the two disagreed on `#`, `%`, `?`, `(`, `)`, `&`, `+` and every
/// non-ASCII byte — i.e. on `C:\Users\Müller\…` or `Game (v2)` — and inline
/// diagnostics were never painted at all. Wave C's own law: *two copies of one
/// expression across a language boundary are a contract nobody is measuring.*
///
/// `URI_FIXTURE_PATHS` and the committed `lspUriFixtures.json` are the
/// measurement. This side generates the file; the vitest arm beside it asserts
/// the TypeScript encoder reproduces every pair. Neither language can move
/// alone.
fn path_to_uri(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let mut p = String::with_capacity(normalized.len());
    for b in normalized.bytes() {
        // RFC 3986 unreserved, plus the two structural characters a file URI
        // needs verbatim: `/` is the separator and `:` is the Windows drive
        // letter's, which `file:///C:/...` requires.
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~' | b'/' | b':') {
            p.push(b as char);
        } else {
            // UTF-8 is escaped byte by byte, which is what the RFC requires.
            p.push('%');
            p.push_str(&format!("{b:02X}"));
        }
    }
    if p.starts_with('/') {
        format!("file://{p}")
    } else {
        format!("file:///{p}")
    }
}

/// **The cross-language corpus** for [`path_to_uri`] and its TypeScript twin.
///
/// Every case here is a path a real project produces on some platform, and each
/// one is a character the old spaces-only encoder let through untouched. The
/// list lives on this side because Rust is the authority: the server is told
/// this spelling, and `publishDiagnostics` comes back under it.
///
/// `#[cfg(test)]`: it is the fixture generator's input and nothing ships it.
#[cfg(test)]
const URI_FIXTURE_PATHS: &[&str] = &[
    // The two shapes of an absolute path.
    "C:/proj/src/main.rs",
    "/home/dev/src/main.rs",
    // A backslash path, as Windows hands it over.
    r"C:\proj\src\main.rs",
    // The one the old encoder handled.
    "C:/my proj/a.rs",
    // The ones it did not.
    "C:/a#b/c.rs",
    "C:/a%20b/c.rs",
    "C:/a?b/c.rs",
    "C:/Game (v2)/src/lib.rs",
    "C:/Bob's Game/src/lib.rs",
    "C:/a&b+c,d/e.rs",
    "C:/a[b]{c}/d.rs",
    "C:/a=b;d/e.rs",
    "C:/a@b!d/e.rs",
    // Non-ASCII, escaped byte by byte as the RFC requires.
    "C:/café/a.rs",
    "C:/Users/Müller/proj/src/main.rs",
    "C:/プロジェクト/src/main.rs",
    "/home/dev/Ω/main.rs",
    // The unreserved set, which must survive verbatim.
    "C:/a-b_c.d~e/f.rs",
];

// ── commands ────────────────────────────────────────────────────────────────

/// Start (if not running) the language server for `language` rooted at
/// `workspace`. rust-analyzer only, for now.
#[tauri::command]
pub async fn lsp_start(
    app: AppHandle,
    state: State<'_, LspState>,
    language: String,
    workspace: String,
) -> Result<(), String> {
    if language != "rust" {
        return Err(format!("no language server for '{language}'"));
    }
    if state
        .servers
        .lock()
        .map_err(|e| e.to_string())?
        .contains_key(&language)
    {
        return Ok(()); // already running
    }

    // The PATH probe (`rust-analyzer --version`) and the child spawn both block
    // on process creation — keep them off the async workers (mirrors git.rs /
    // package.rs). Everything they need is `Send + 'static` (AppHandle clone +
    // workspace string); `Child` is `Send`, so it comes back out.
    let app_probe = app.clone();
    let workspace_spawn = workspace.clone();
    let mut child = tauri::async_runtime::spawn_blocking(move || {
        let bin = resolve_rust_analyzer(&app_probe)?;
        Command::new(bin)
            .current_dir(&workspace_spawn)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn rust-analyzer: {e}"))
    })
    .await
    .map_err(|e| format!("lsp_start task failed to run: {e}"))??;

    let stdin = Arc::new(Mutex::new(child.stdin.take().ok_or("no stdin")?));
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let stderr = child.stderr.take().ok_or("no stderr")?;
    let pending: Arc<Mutex<HashMap<i64, Sender<Value>>>> = Arc::new(Mutex::new(HashMap::new()));
    let next_id = Arc::new(AtomicI64::new(1));

    // Reader thread (replies + diagnostics) and a stderr drain (else the pipe
    // fills and the server stalls).
    let app_reader = app.clone();
    let pending_reader = pending.clone();
    std::thread::Builder::new()
        .name("lsp-reader".into())
        .spawn(move || reader_loop(app_reader, stdout, pending_reader))
        .map_err(|e| e.to_string())?;
    std::thread::Builder::new()
        .name("lsp-stderr".into())
        .spawn(move || {
            let mut sink = Vec::new();
            let _ = BufReader::new(stderr).read_to_end(&mut sink);
        })
        .map_err(|e| e.to_string())?;

    // initialize / initialized handshake.
    let root_uri = path_to_uri(&workspace);
    let init_params = json!({
        "processId": std::process::id(),
        "rootUri": root_uri,
        "capabilities": {
            "textDocument": {
                "synchronization": { "dynamicRegistration": false, "didSave": false },
                "completion": { "completionItem": { "snippetSupport": false } },
                "hover": { "contentFormat": ["markdown", "plaintext"] },
                "definition": {},
                "publishDiagnostics": {}
            }
        },
        "workspaceFolders": [ { "uri": root_uri, "name": "workspace" } ]
    });
    send_request_async(
        stdin.clone(),
        next_id.clone(),
        pending.clone(),
        "initialize".to_string(),
        init_params,
        Duration::from_secs(30),
    )
    .await?;
    send_notify(&stdin, "initialized", json!({}))?;

    state.servers.lock().map_err(|e| e.to_string())?.insert(
        language,
        LspServer {
            child,
            stdin,
            next_id,
            pending,
        },
    );
    let _ = app.emit("lsp://started", json!({ "language": "rust" }));
    Ok(())
}

/// Stop the language server for `language`.
#[tauri::command]
pub async fn lsp_stop(
    app: AppHandle,
    state: State<'_, LspState>,
    language: String,
) -> Result<(), String> {
    if let Some(mut server) = state
        .servers
        .lock()
        .map_err(|e| e.to_string())?
        .remove(&language)
    {
        let _ = send_notify(&server.stdin, "exit", Value::Null);
        let _ = server.child.kill();
    }
    let _ = app.emit("lsp://stopped", json!({ "language": language }));
    Ok(())
}

fn notify_doc(
    state: &State<'_, LspState>,
    language: &str,
    build: impl FnOnce() -> (&'static str, Value),
) -> Result<(), String> {
    let servers = state.servers.lock().map_err(|e| e.to_string())?;
    if let Some(server) = servers.get(language) {
        let (method, params) = build();
        send_notify(&server.stdin, method, params)?;
    }
    Ok(())
}

#[tauri::command]
pub async fn lsp_did_open(
    state: State<'_, LspState>,
    language: String,
    path: String,
    text: String,
) -> Result<(), String> {
    let uri = path_to_uri(&path);
    state.open_version(&uri);
    notify_doc(&state, &language, || {
        (
            "textDocument/didOpen",
            json!({ "textDocument": {
                "uri": uri, "languageId": "rust", "version": 1, "text": text
            }}),
        )
    })
}

#[tauri::command]
pub async fn lsp_did_change(
    state: State<'_, LspState>,
    language: String,
    path: String,
    text: String,
) -> Result<(), String> {
    let uri = path_to_uri(&path);
    let version = state.next_version(&uri);
    notify_doc(&state, &language, || {
        (
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": version },
                "contentChanges": [ { "text": text } ]
            }),
        )
    })
}

#[tauri::command]
pub async fn lsp_did_close(
    state: State<'_, LspState>,
    language: String,
    path: String,
) -> Result<(), String> {
    let uri = path_to_uri(&path);
    state.close_version(&uri);
    notify_doc(&state, &language, || {
        (
            "textDocument/didClose",
            json!({ "textDocument": { "uri": uri } }),
        )
    })
}

/// Forward a request (completion/hover/definition/…) and return the result JSON.
#[tauri::command]
pub async fn lsp_request(
    state: State<'_, LspState>,
    language: String,
    method: String,
    params: Value,
) -> Result<Value, String> {
    // Clone the channel handles, then DROP the servers lock before blocking.
    let (stdin, next_id, pending) = {
        let servers = state.servers.lock().map_err(|e| e.to_string())?;
        let server = servers
            .get(&language)
            .ok_or_else(|| format!("no server for '{language}'"))?;
        (
            server.stdin.clone(),
            server.next_id.clone(),
            server.pending.clone(),
        )
    };
    send_request_async(
        stdin,
        next_id,
        pending,
        method,
        params,
        Duration::from_secs(5),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L7.L3: the URI is the document's identity on this wire. Escaping only the
    /// space meant a path with `#`, `%`, `?` or any non-ASCII character named a
    /// *different* document to the server, and `publishDiagnostics` then came
    /// back under a URI the editor never asked about — every diagnostic for that
    /// file dropped, silently.
    #[test]
    fn a_path_with_awkward_characters_still_names_the_same_document() {
        // The separator and the drive colon survive: `file:///C:/…` requires both.
        assert_eq!(
            path_to_uri("C:/proj/src/main.rs"),
            "file:///C:/proj/src/main.rs"
        );
        assert_eq!(
            path_to_uri("/home/dev/src/main.rs"),
            "file:///home/dev/src/main.rs"
        );
        // A space, as before.
        assert_eq!(path_to_uri("C:/my proj/a.rs"), "file:///C:/my%20proj/a.rs");
        // The characters the old escaper let through untouched.
        assert_eq!(path_to_uri("C:/a#b/c.rs"), "file:///C:/a%23b/c.rs");
        assert_eq!(path_to_uri("C:/a%20b/c.rs"), "file:///C:/a%2520b/c.rs");
        assert_eq!(path_to_uri("C:/a?b/c.rs"), "file:///C:/a%3Fb/c.rs");
        // Non-ASCII is escaped byte by byte, as RFC 3986 requires.
        assert_eq!(path_to_uri("C:/café/a.rs"), "file:///C:/caf%C3%A9/a.rs");
        // No output may carry a character that is not legal in a URI path.
        for raw in ["C:/a#b/c.rs", "C:/café/a.rs", "C:/a b/c.rs"] {
            let uri = path_to_uri(raw);
            assert!(
                uri.is_ascii() && !uri.contains(' ') && !uri.contains('#'),
                "{uri} is not a URI"
            );
        }
    }

    /// **The cross-language pin** (round-2 findings R2.F3 / B4).
    ///
    /// `lspBridge.ts` hand-restates this encoder and compares its output to the
    /// URIs this side produces, so the two agreeing is a *contract*; before this
    /// arm nothing measured it, and Wave F moved one side and not the other.
    /// The mechanism is the ts-rs bindings gate's, in miniature: Rust generates
    /// a committed fixture, the frontend's own test consumes it, and CI fails on
    /// either side moving alone.
    ///
    /// A DRIFT CHECK, not a generator: the committed file is compared, never
    /// silently rewritten, because a test that regenerates its own expectation
    /// is the vacuous shape this campaign has caught eight times. Re-bless with
    /// `INF_BLESS_LSP_URI=1` after deciding the encoder should move.
    #[test]
    fn the_typescript_mirror_is_pinned_to_this_encoder() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../src/lib/editor/__tests__/lspUriFixtures.json");

        let mut json = String::from(
            "{\n  \"note\": \"GENERATED by inf-studio's `the_typescript_mirror_is_pinned_to_this_encoder`. Do not hand-edit; re-bless with INF_BLESS_LSP_URI=1.\",\n  \"cases\": [\n",
        );
        for (i, raw) in URI_FIXTURE_PATHS.iter().enumerate() {
            let comma = if i + 1 == URI_FIXTURE_PATHS.len() {
                ""
            } else {
                ","
            };
            json.push_str(&format!(
                "    {{ \"path\": {}, \"uri\": {} }}{comma}\n",
                serde_json::to_string(raw).expect("a path serializes"),
                serde_json::to_string(&path_to_uri(raw)).expect("a uri serializes"),
            ));
        }
        json.push_str("  ]\n}\n");

        if std::env::var("INF_BLESS_LSP_URI").is_ok() {
            std::fs::write(&path, json.as_bytes()).expect("the fixture is writable");
            return;
        }

        let committed = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| {
                panic!(
                    "{} must exist — it is what the frontend's encoder is held to: {e}",
                    path.display()
                )
            })
            // The P22 CRLF law: a committed text file read by a test is
            // normalized first.
            .replace("\r\n", "\n");
        assert_eq!(
            committed, json,
            "the committed LSP URI fixture no longer matches this encoder. The \
             TypeScript twin in `lspBridge.ts` is held to that file, so moving one \
             side alone is exactly the drift that dropped every diagnostic on a path \
             with `#`, `%` or a non-ASCII character. Re-bless with \
             INF_BLESS_LSP_URI=1 and check the vitest arm goes with it."
        );
    }

    /// L7.L3: `didChange` sent the literal version `2` forever. LSP requires the
    /// version to increase per document, and a server may discard an edit that
    /// does not — so every keystroke after the first was, by the letter of the
    /// protocol, droppable.
    #[test]
    fn each_change_carries_a_higher_version_than_the_last() {
        let state = LspState::default();
        let a = "file:///C:/a.rs";
        let b = "file:///C:/b.rs";
        state.open_version(a);
        let versions: Vec<i64> = (0..4).map(|_| state.next_version(a)).collect();
        assert_eq!(
            versions,
            vec![2, 3, 4, 5],
            "didOpen is 1; changes follow it"
        );

        // Per document, not global: a second file starts its own sequence.
        state.open_version(b);
        assert_eq!(state.next_version(b), 2);
        assert_eq!(state.next_version(a), 6, "the first file kept its place");

        // Closing forgets the counter (the map is otherwise unbounded), and a
        // re-open restarts at the version its `didOpen` announced.
        state.close_version(a);
        state.open_version(a);
        assert_eq!(state.next_version(a), 2);
    }

    /// C4-45: EOF and a malformed frame were the same `None`, so ONE bad message
    /// killed the reader thread and every diagnostic with it — indistinguishable
    /// from "rust-analyzer exited".
    #[test]
    fn a_malformed_frame_is_not_end_of_stream() {
        let mut eof: &[u8] = b"";
        assert!(matches!(read_message(&mut eof), Ok(None)), "empty is EOF");

        let body = br#"{"jsonrpc":"2.0"}"#;
        let framed = [
            format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes(),
            body.to_vec(),
        ]
        .concat();
        let mut good: &[u8] = &framed;
        assert!(
            matches!(read_message(&mut good), Ok(Some(_))),
            "a well-formed frame reads"
        );

        for (why, bytes) in [
            ("no Content-Length", b"X-Other: 1\r\n\r\n{}".to_vec()),
            (
                "unparseable length",
                b"Content-Length: banana\r\n\r\n{}".to_vec(),
            ),
            (
                "body is not JSON",
                b"Content-Length: 5\r\n\r\nhello".to_vec(),
            ),
            ("body is short", b"Content-Length: 500\r\n\r\n{}".to_vec()),
        ] {
            let mut r: &[u8] = &bytes;
            assert!(
                read_message(&mut r).is_err(),
                "{why} must be an error, not end-of-stream"
            );
        }
    }

    /// **Round-2 finding B5**: `Content-Length` is a number a subprocess
    /// writes, and `vec![0u8; len]` allocates it before a body byte is read.
    #[test]
    fn a_body_past_the_ceiling_is_refused_before_it_is_allocated() {
        // The declared length is the whole finding — nothing follows it, so a
        // reader that got as far as `read_exact` would fail differently (and,
        // at these sizes, only after trying to allocate).
        for len in [MAX_LSP_BODY + 1, usize::MAX] {
            let bytes = format!(
                "Content-Length: {len}

"
            )
            .into_bytes();
            let mut r: &[u8] = &bytes;
            let e = read_message(&mut r).expect_err("a {len}-byte body was accepted");
            assert!(
                e.contains("ceiling"),
                "the refusal must name the ceiling rather than fail later: {e}"
            );
        }
        // The ceiling is not so tight that a real message trips it: a body at
        // exactly the ceiling is allowed through to the (short) read.
        let bytes = format!(
            "Content-Length: {MAX_LSP_BODY}

"
        )
        .into_bytes();
        let mut r: &[u8] = &bytes;
        let e = read_message(&mut r).expect_err("a short body must still fail");
        assert!(
            !e.contains("ceiling"),
            "a body AT the ceiling was refused by it: {e}"
        );
    }

    /// The reader owning the end of the answer: when the pending map is
    /// cleared, a waiting `recv_timeout` returns `Disconnected` at once
    /// instead of burning its full timeout.
    ///
    /// This drives `std::sync::mpsc` directly rather than `reader_loop`, which
    /// needs an `AppHandle` and a real `ChildStdout` — the arm is over the
    /// mechanism the loop's last statement relies on, and it is the mechanism
    /// that was missing, not the call.
    #[test]
    fn clearing_the_pending_map_releases_a_waiting_request() {
        let pending: Arc<Mutex<HashMap<i64, Sender<Value>>>> = Arc::default();
        let (tx, rx) = channel::<Value>();
        pending.lock().unwrap().insert(7, tx);

        // While the sender is held, the wait really does block to its timeout.
        let t0 = std::time::Instant::now();
        assert!(rx.recv_timeout(Duration::from_millis(60)).is_err());
        assert!(
            t0.elapsed() >= Duration::from_millis(50),
            "the control did not actually wait, so the arm below is vacuous"
        );

        // Dropping it — which is what the reader's exit does — is immediate.
        pending.lock().unwrap().clear();
        let t0 = std::time::Instant::now();
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(5)),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
        ));
        assert!(
            t0.elapsed() < Duration::from_secs(1),
            "a released request still waited out its timeout"
        );
    }

    /// **Round-2 finding B6**: a server→client *request* carries an `id` too,
    /// and routing on `id` alone resolved one of our pending requests with
    /// `Null` — indistinguishable from "the server had no results".
    #[test]
    fn a_server_request_is_not_mistaken_for_a_reply() {
        // Ours: an id and no method.
        assert_eq!(
            reply_id(&json!({"jsonrpc":"2.0","id":3,"result":{}})),
            Some(3)
        );
        assert_eq!(
            reply_id(&json!({"jsonrpc":"2.0","id":3,"error":{"code":-32601}})),
            Some(3)
        );

        // Theirs: an id AND a method. These are the three rust-analyzer
        // actually sends, and each one used to resolve pending request `id`.
        for method in [
            "client/registerCapability",
            "workspace/configuration",
            "window/workDoneProgress/create",
        ] {
            assert_eq!(
                reply_id(&json!({"jsonrpc":"2.0","id":3,"method":method,"params":{}})),
                None,
                "{method} was routed as a reply to our request 3"
            );
        }

        // Notifications have a method and no id, and were never replies.
        assert_eq!(
            reply_id(&json!({
                "jsonrpc":"2.0",
                "method":"textDocument/publishDiagnostics",
                "params":{}
            })),
            None
        );
    }
}
