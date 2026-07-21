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
    write_message(
        stdin,
        &json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }),
    )?;
    match rx.recv_timeout(timeout) {
        Ok(v) => Ok(v),
        Err(_) => {
            pending.lock().ok().and_then(|mut p| p.remove(&id));
            Err(format!("lsp request '{method}' timed out"))
        }
    }
}

/// Read one framed message body from the server stream. Returns None on EOF.
fn read_message(reader: &mut impl BufRead) -> Option<Value> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).ok()?;
        if n == 0 {
            return None; // EOF
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse().ok();
        }
    }
    let len = content_length?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

/// The reader thread: routes replies to pending requests, and
/// publishDiagnostics notifications to the `lsp://diagnostics` event.
fn reader_loop(
    app: AppHandle,
    stdout: std::process::ChildStdout,
    pending: Arc<Mutex<HashMap<i64, Sender<Value>>>>,
) {
    let mut reader = BufReader::new(stdout);
    while let Some(msg) = read_message(&mut reader) {
        if let Some(id) = msg.get("id").and_then(Value::as_i64) {
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
        // Server-to-client requests (e.g. workspace/configuration) are ignored
        // in this MVP; rust-analyzer tolerates the missing reply.
    }
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

/// `file://` URI from an absolute path (Windows drive paths → `file:///C:/…`).
fn path_to_uri(path: &str) -> String {
    let p = path.replace('\\', "/").replace(' ', "%20");
    if p.starts_with('/') {
        format!("file://{p}")
    } else {
        format!("file:///{p}")
    }
}

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

    let bin = resolve_rust_analyzer(&app)?;
    let mut child = Command::new(bin)
        .current_dir(&workspace)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn rust-analyzer: {e}"))?;

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
    send_request(
        &stdin,
        &next_id,
        &pending,
        "initialize",
        init_params,
        Duration::from_secs(30),
    )?;
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
    notify_doc(&state, &language, || {
        (
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": 2 },
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
    send_request(
        &stdin,
        &next_id,
        &pending,
        &method,
        params,
        Duration::from_secs(5),
    )
}
