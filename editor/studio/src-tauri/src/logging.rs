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

static APP: OnceLock<tauri::AppHandle> = OnceLock::new();
static BUFFER: Mutex<VecDeque<LogLine>> = Mutex::new(VecDeque::new());
static SEQ: AtomicU64 = AtomicU64::new(0);

/// Install the app handle and flush everything buffered so far.
pub fn attach_app(app: tauri::AppHandle) {
    if APP.set(app.clone()).is_err() {
        return;
    }
    let drained: Vec<LogLine> = {
        let mut buf = BUFFER.lock().expect("log buffer poisoned");
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

        if let Some(app) = APP.get() {
            let _ = app.emit("log://line", &line);
        } else if let Ok(mut buf) = BUFFER.lock() {
            if buf.len() >= BUFFER_CAP {
                buf.pop_front();
            }
            buf.push_back(line);
        }
    }
}
