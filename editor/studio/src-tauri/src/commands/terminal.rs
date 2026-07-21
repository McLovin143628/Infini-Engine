//! Embedded terminal (P5.3): a `portable-pty` pseudo-terminal per session.
//!
//! Ported from the CodeR reference. Each session spawns a shell in a real PTY
//! (ConPTY on Windows), and a background thread streams its output — base64'd,
//! chunked — to the webview on `pty://output/{id}`; `pty://exit/{id}` fires when
//! the shell ends. The frontend `usePty` hook decodes and feeds xterm.js.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Mutex;

use portable_pty::{native_pty_system, Child, CommandBuilder, MasterPty, PtySize};
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

/// `pty://output/{id}` payload.
#[derive(Clone, Serialize)]
struct PtyOutput {
    id: String,
    /// base64 of the raw shell bytes.
    data: String,
}

/// `pty://exit/{id}` payload.
#[derive(Clone, Serialize)]
struct PtyExit {
    id: String,
    code: Option<i32>,
}

struct PtySession {
    master: Box<dyn MasterPty + Send>,
    writer: Box<dyn Write + Send>,
    child: Box<dyn Child + Send + Sync>,
}

/// Managed state: the live sessions by id.
#[derive(Default)]
pub struct PtyState {
    sessions: Mutex<HashMap<String, PtySession>>,
}

fn default_shell() -> String {
    if cfg!(windows) {
        "powershell.exe".to_string()
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string())
    }
}

/// Open a new terminal session; returns its id. Output streams on
/// `pty://output/{id}`.
#[tauri::command]
pub async fn pty_create(
    app: AppHandle,
    state: State<'_, PtyState>,
    cwd: Option<String>,
    cols: u16,
    rows: u16,
) -> Result<String, String> {
    let size = PtySize {
        rows: rows.max(1),
        cols: cols.max(1),
        pixel_width: 0,
        pixel_height: 0,
    };
    let pair = native_pty_system()
        .openpty(size)
        .map_err(|e| format!("openpty: {e}"))?;

    let mut cmd = CommandBuilder::new(default_shell());
    if let Some(dir) = cwd.filter(|d| std::path::Path::new(d).is_dir()) {
        cmd.cwd(dir);
    }
    let child = pair
        .slave
        .spawn_command(cmd)
        .map_err(|e| format!("spawn shell: {e}"))?;
    // Close our handle to the slave so EOF is seen when the shell exits.
    drop(pair.slave);

    let reader = pair
        .master
        .try_clone_reader()
        .map_err(|e| format!("clone reader: {e}"))?;
    let writer = pair
        .master
        .take_writer()
        .map_err(|e| format!("take writer: {e}"))?;

    let id = Uuid::new_v4().to_string();

    // Stream output off-thread; never holds the sessions lock.
    let app_out = app.clone();
    let id_out = id.clone();
    std::thread::Builder::new()
        .name(format!("pty-{id}"))
        .spawn(move || read_loop(app_out, id_out, reader))
        .map_err(|e| format!("spawn read loop: {e}"))?;

    state.sessions.lock().map_err(|e| e.to_string())?.insert(
        id.clone(),
        PtySession {
            master: pair.master,
            writer,
            child,
        },
    );
    Ok(id)
}

fn read_loop(app: AppHandle, id: String, mut reader: Box<dyn Read + Send>) {
    let mut buf = [0u8; 4096];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let _ = app.emit(
                    &format!("pty://output/{id}"),
                    PtyOutput {
                        id: id.clone(),
                        data: base64(&buf[..n]),
                    },
                );
            }
            Err(_) => break,
        }
    }
    let _ = app.emit(&format!("pty://exit/{id}"), PtyExit { id, code: None });
}

/// Write user input (UTF-8 text) to a session's shell.
#[tauri::command]
pub async fn pty_write(state: State<'_, PtyState>, id: String, data: String) -> Result<(), String> {
    let mut sessions = state.sessions.lock().map_err(|e| e.to_string())?;
    let session = sessions.get_mut(&id).ok_or("no such pty session")?;
    session
        .writer
        .write_all(data.as_bytes())
        .map_err(|e| format!("pty write: {e}"))?;
    session
        .writer
        .flush()
        .map_err(|e| format!("pty flush: {e}"))
}

/// Resize a session (on the xterm `FitAddon` fit).
#[tauri::command]
pub async fn pty_resize(
    state: State<'_, PtyState>,
    id: String,
    cols: u16,
    rows: u16,
) -> Result<(), String> {
    let sessions = state.sessions.lock().map_err(|e| e.to_string())?;
    if let Some(session) = sessions.get(&id) {
        session
            .master
            .resize(PtySize {
                rows: rows.max(1),
                cols: cols.max(1),
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| format!("pty resize: {e}"))?;
    }
    Ok(())
}

/// Close a session and kill its shell.
#[tauri::command]
pub async fn pty_close(state: State<'_, PtyState>, id: String) -> Result<(), String> {
    if let Some(mut session) = state
        .sessions
        .lock()
        .map_err(|e| e.to_string())?
        .remove(&id)
    {
        let _ = session.child.kill();
    }
    Ok(())
}

/// Standard base64 (no line breaks) — matches the encoder in `commands::assets`,
/// kept local so the terminal module has no cross-module dependency.
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
