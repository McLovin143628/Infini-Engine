//! Tracing → Output Log bridge (ROADMAP P1.4.4).
//!
//! A `tracing_subscriber` layer that converts every event into an
//! [`LogLine`] and emits it on the `log://line` channel. Events fired
//! before the webview exists land in a bounded ring buffer and flush the
//! moment the app handle is installed, so startup logs aren't lost.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use inf_editor_core::ipc::{LogLevel, LogLine};
use tauri::Emitter;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// Bounded pre-webview buffer (drops oldest beyond this).
const BUFFER_CAP: usize = 512;
/// Recent-line ring for the crash reporter (P15.2) — always on, independent of
/// the webview, so a crash file can include the last log lines.
const CRASH_RING_CAP: usize = 128;

static APP: OnceLock<tauri::AppHandle> = OnceLock::new();
static BUFFER: Mutex<VecDeque<LogLine>> = Mutex::new(VecDeque::new());
static CRASH_RING: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
static SEQ: AtomicU64 = AtomicU64::new(0);

/// The most recent formatted log lines (oldest first), for a crash report.
///
/// Called from the panic hook (`commands::diagnostics::install_crash_hook`).
/// **Never blocks**: if a panic fires *inside* the log critical section on this
/// thread, the `CRASH_RING` guard from `on_event` is still held when the hook
/// runs (the hook executes before the panicking frame unwinds), so a blocking
/// `lock()` here would deadlock the hook and *no crash file would be written*.
/// `try_lock` returns whatever is available, or empty on contention/poison.
pub fn recent_lines() -> Vec<String> {
    match CRASH_RING.try_lock() {
        Ok(ring) => ring.iter().cloned().collect(),
        Err(_) => Vec::new(), // WouldBlock (reentrant) or poisoned → best-effort empty
    }
}

/// Install the app handle and flush everything buffered so far.
pub fn attach_app(app: tauri::AppHandle) {
    if APP.set(app.clone()).is_err() {
        return;
    }
    let drained: Vec<LogLine> = {
        // Recover from poison so a panicked logger can't wedge the startup flush.
        let mut buf = BUFFER.lock().unwrap_or_else(|p| p.into_inner());
        buf.drain(..).collect()
    };
    for line in drained {
        let _ = app.emit("log://line", &line);
    }
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as f64)
        .unwrap_or(0.0)
}

fn to_level(level: &Level) -> LogLevel {
    match *level {
        Level::TRACE => LogLevel::Trace,
        Level::DEBUG => LogLevel::Debug,
        Level::INFO => LogLevel::Info,
        Level::WARN => LogLevel::Warn,
        Level::ERROR => LogLevel::Error,
    }
}

/// Collects the `message` field verbatim and appends the rest as ` k=v`.
#[derive(Default)]
struct MessageVisitor {
    message: String,
    rest: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            let _ = write!(self.rest, " {}={value:?}", field.name());
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            let _ = write!(self.rest, " {}={value}", field.name());
        }
    }
}

/// The bridge layer. Filtering rides the subscriber's global `EnvFilter`;
/// anything that reaches `on_event` is forwarded.
pub struct LogBridgeLayer;

impl<S: Subscriber> Layer<S> for LogBridgeLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let mut message = visitor.message;
        message.push_str(&visitor.rest);

        let line = LogLine {
            seq: SEQ.fetch_add(1, Ordering::Relaxed),
            level: to_level(event.metadata().level()),
            target: event.metadata().target().to_string(),
            message,
            timestamp_ms: now_ms(),
        };

        // Feed the crash-report ring (bounded, always on). Recover from a
        // poisoned lock (a prior panic mid-push) via `into_inner` so one panicked
        // logger never disables the crash ring for the rest of the session.
        {
            let mut ring = CRASH_RING.lock().unwrap_or_else(|p| p.into_inner());
            if ring.len() >= CRASH_RING_CAP {
                ring.pop_front();
            }
            ring.push_back(format!(
                "{:>5?} {}: {}",
                line.level, line.target, line.message
            ));
        }

        if let Some(app) = APP.get() {
            let _ = app.emit("log://line", &line);
        } else {
            // Same poison recovery for the pre-webview buffer.
            let mut buf = BUFFER.lock().unwrap_or_else(|p| p.into_inner());
            if buf.len() >= BUFFER_CAP {
                buf.pop_front();
            }
            buf.push_back(line);
        }
    }
}
