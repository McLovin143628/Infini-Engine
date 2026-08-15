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
}
