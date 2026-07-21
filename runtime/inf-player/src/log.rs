//! Logging + crash capture (P9.3 item 3 & 5).
//!
//! [`init`] installs a `tracing` subscriber that tees every formatted log line to
//! stdout, an optional `--log-file`, **and** a bounded in-memory ring buffer. A
//! panic hook then writes a crash report — the panic message, its location, and
//! the last [`RING_CAP`] log lines — to stderr and to the `--crash-file` (default
//! `crash.txt` in the working directory), so a shipped player leaves a diagnosable
//! trail when a gameplay script panics.
//!
//! The ring is a process-global `OnceLock` shared by the subscriber and the panic
//! hook, so the two always agree regardless of how many times [`init`] runs (the
//! subscriber itself installs once via `try_init`; in-process tests may call
//! [`init`] repeatedly).

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

/// How many recent log lines the crash report includes.
pub const RING_CAP: usize = 256;

#[derive(Clone)]
struct Ring(Arc<Mutex<VecDeque<String>>>);

impl Ring {
    fn push_line(&self, line: &str) {
        let line = line.trim_end();
        if line.is_empty() {
            return;
        }
        if let Ok(mut g) = self.0.lock() {
            while g.len() >= RING_CAP {
                g.pop_front();
            }
            g.push_back(line.to_string());
        }
    }

    fn snapshot(&self) -> Vec<String> {
        self.0
            .lock()
            .map(|g| g.iter().cloned().collect())
            .unwrap_or_default()
    }
}

static RING: OnceLock<Ring> = OnceLock::new();

fn ring() -> Ring {
    RING.get_or_init(|| Ring(Arc::new(Mutex::new(VecDeque::with_capacity(RING_CAP)))))
        .clone()
}

struct CrashState {
    ring: Ring,
    crash_file: PathBuf,
}

static CRASH: OnceLock<Mutex<CrashState>> = OnceLock::new();

/// A `MakeWriter` factory: each event gets a [`TeeWriter`] that fans the
/// formatted bytes out to stdout, the optional log file, and the ring buffer.
#[derive(Clone)]
struct TeeFactory {
    ring: Ring,
    file: Option<Arc<Mutex<std::fs::File>>>,
}

struct TeeWriter {
    ring: Ring,
    file: Option<Arc<Mutex<std::fs::File>>>,
    buf: String,
}

impl Write for TeeWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let _ = io::stdout().write_all(data);
        if let Some(f) = &self.file {
            if let Ok(mut f) = f.lock() {
                let _ = f.write_all(data);
            }
        }
        self.buf.push_str(&String::from_utf8_lossy(data));
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let _ = io::stdout().flush();
        if let Some(f) = &self.file {
            if let Ok(mut f) = f.lock() {
                let _ = f.flush();
            }
        }
        Ok(())
    }
}

impl Drop for TeeWriter {
    fn drop(&mut self) {
        // The fmt layer writes one event as one line (+ newline); record it.
        for line in std::mem::take(&mut self.buf).split('\n') {
            self.ring.push_line(line);
        }
    }
}

impl<'a> MakeWriter<'a> for TeeFactory {
    type Writer = TeeWriter;
    fn make_writer(&'a self) -> Self::Writer {
        TeeWriter {
            ring: self.ring.clone(),
            file: self.file.clone(),
            buf: String::new(),
        }
    }
}

/// Install the tracing subscriber and the crash-capturing panic hook. Safe to
/// call more than once (the subscriber installs on the first call; later calls
/// only update the crash-file target).
pub fn init(log_file: Option<PathBuf>, crash_file: PathBuf) {
    let ring = ring();

    let file = log_file
        .and_then(|p| {
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&p)
                .ok()
        })
        .map(|f| Arc::new(Mutex::new(f)));

    let factory = TeeFactory {
        ring: ring.clone(),
        file,
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // `try_init` is a no-op (Err) if a subscriber is already set — fine.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .with_target(false)
        .with_writer(factory)
        .try_init();

    let cell = CRASH.get_or_init(|| {
        Mutex::new(CrashState {
            ring: ring.clone(),
            crash_file: crash_file.clone(),
        })
    });
    if let Ok(mut g) = cell.lock() {
        g.ring = ring;
        g.crash_file = crash_file;
    }

    install_panic_hook();
}

fn install_panic_hook() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return; // hook already installed
    }
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(cell) = CRASH.get() {
            if let Ok(state) = cell.lock() {
                let report = build_report(info, &state.ring.snapshot());
                eprintln!("{report}");
                let _ = std::fs::write(&state.crash_file, report.as_bytes());
            }
        }
        default(info);
    }));
}

/// Format a crash report from a panic and the recent log lines.
fn build_report(info: &std::panic::PanicHookInfo<'_>, log_lines: &[String]) -> String {
    let msg = payload_str(info.payload());
    let loc = info
        .location()
        .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
        .unwrap_or_else(|| "<unknown>".to_string());

    let mut out = String::new();
    out.push_str("inf-player CRASH\n");
    out.push_str(&format!("message : {msg}\n"));
    out.push_str(&format!("location: {loc}\n"));
    out.push_str(&format!(
        "--- last {} log line(s) ---\n",
        log_lines.len().min(RING_CAP)
    ));
    for line in log_lines {
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn payload_str(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_bounded_and_ordered() {
        let r = Ring(Arc::new(Mutex::new(VecDeque::new())));
        for i in 0..(RING_CAP + 10) {
            r.push_line(&format!("line {i}"));
        }
        let snap = r.snapshot();
        assert_eq!(snap.len(), RING_CAP);
        assert_eq!(snap.last().unwrap(), &format!("line {}", RING_CAP + 9));
    }
}
