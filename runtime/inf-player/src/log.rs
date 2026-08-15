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
use std::time::{SystemTime, UNIX_EPOCH};

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

/// How many recent log lines the crash report includes.
pub const RING_CAP: usize = 256;

/// The active GPU adapter description (`wgpu::AdapterInfo` debug), recorded at
/// render init so a crash report can name the GPU as a first-class field. `None`
/// on a headless/no-adapter run.
static ADAPTER: OnceLock<String> = OnceLock::new();

/// Record the active GPU adapter for the crash report (called once from render
/// init). First call wins.
pub fn set_adapter_info(info: String) {
    let _ = ADAPTER.set(info);
}

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
    // Layered registry form (equivalent to the old `fmt()` builder) so the Tracy
    // profiler layer can be appended behind the `tracy` feature (P15.1). `try_init`
    // is a no-op (Err) if a subscriber is already set — fine.
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_target(false)
        .with_writer(factory);
    let subscriber = tracing_subscriber::registry().with(filter).with(fmt_layer);
    #[cfg(feature = "tracy")]
    let subscriber = subscriber.with(tracing_tracy::TracyLayer::default());
    let _ = subscriber.try_init();

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
                // Also drop a timestamped copy into a `crashes/` dir beside the
                // primary crash file, so repeated crashes each leave a record
                // (P15.2). Best-effort — the primary write above is the contract.
                write_timestamped_copy(&state.crash_file, report.as_bytes());
            }
        }
        default(info);
    }));
}

/// Write a timestamped duplicate of the crash report into a `crashes/` directory
/// alongside `primary`. Best-effort (ignored on any error).
fn write_timestamped_copy(primary: &std::path::Path, bytes: &[u8]) {
    let dir = primary
        .parent()
        .map(|p| p.join("crashes"))
        .unwrap_or_else(|| PathBuf::from("crashes"));
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let _ = std::fs::write(dir.join(format!("crash-{millis}.txt")), bytes);
}

/// Format a crash report from a panic and the recent log lines. Fields are
/// engine version, OS/arch, GPU adapter (if known), panic location + message,
/// and a tail of recent log lines.
fn build_report(info: &std::panic::PanicHookInfo<'_>, log_lines: &[String]) -> String {
    let msg = payload_str(info.payload());
    let loc = info
        .location()
        .map(|l| {
            format!(
                "{}:{}:{}",
                sanitize_location(l.file()),
                l.line(),
                l.column()
            )
        })
        .unwrap_or_else(|| "<unknown>".to_string());
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    let mut out = String::new();
    out.push_str("inf-player CRASH\n");
    out.push_str(&format!("engine  : {}\n", env!("CARGO_PKG_VERSION")));
    out.push_str(&format!(
        "os      : {} ({})\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    if let Some(adapter) = ADAPTER.get() {
        out.push_str(&format!("adapter : {adapter}\n"));
    }
    out.push_str(&format!("time    : {millis} ms since epoch\n"));
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

/// Strip a build machine's identity out of a panic location (L7.L2).
///
/// `std::panic::Location::file` is `file!()`. For this workspace's own crates
/// cargo already passes relative paths, but a panic inside a dependency reports
/// its registry path — on Windows `C:\Users\<name>\.cargo\registry\src\…\
/// wgpu-30\src\lib.rs` — so a crash report a customer mails back carries the
/// **builder's OS username**. The compile-time remedy is
/// `[profile.release] trim-paths` (see the root `Cargo.toml`: not stabilized in
/// this toolchain), which is why the runtime one exists and does not depend on
/// how the binary was built.
///
/// Everything the reader needs is kept: the crate directory and the path inside
/// it. Only the machine prefix goes.
pub(crate) fn sanitize_location(file: &str) -> String {
    // A registry / git-checkout path: keep from `<crate>-<version>` onwards.
    for marker in [
        "registry/src/",
        "registry\\src\\",
        "git/checkouts/",
        "git\\checkouts\\",
    ] {
        if let Some(idx) = file.find(marker) {
            let rest = &file[idx + marker.len()..];
            // Drop the index/repo hash directory that follows the marker.
            let rest = rest
                .split_once(['/', '\\'])
                .map(|(_, tail)| tail)
                .unwrap_or(rest);
            return format!("<dep>/{}", rest.replace('\\', "/"));
        }
    }
    // The standard library, remapped by rustc itself but not on every path.
    if let Some(idx) = file.find("/rustc/") {
        return file[idx..].replace('\\', "/");
    }
    // Anything still absolute is a build directory: keep only the tail below the
    // workspace-looking segment, else the file name.
    let looks_absolute = file.starts_with('/')
        || file
            .as_bytes()
            .get(1)
            .is_some_and(|&c| c == b':' && file.len() > 2);
    if looks_absolute {
        let normalized = file.replace('\\', "/");
        for anchor in ["/crates/", "/editor/", "/runtime/", "/tools/"] {
            if let Some(idx) = normalized.find(anchor) {
                return normalized[idx + 1..].to_string();
            }
        }
        return normalized
            .rsplit_once('/')
            .map(|(_, name)| format!("<build>/{name}"))
            .unwrap_or_else(|| "<build>".to_string());
    }
    file.replace('\\', "/")
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

    /// L7.L2: a crash report is a file a customer mails back. It must not carry
    /// the machine that built the binary — on Windows that string is literally
    /// the builder's OS username.
    #[test]
    fn a_panic_location_never_carries_the_build_machine() {
        let cases = [
            (
                "C:\\Users\\someone\\.cargo\\registry\\src\\index.crates.io-1949cf8/wgpu-30.0.1/src/lib.rs",
                "<dep>/wgpu-30.0.1/src/lib.rs",
            ),
            (
                "/home/someone/.cargo/registry/src/index.crates.io-1949cf8/rapier3d-f64-0.34.0/src/pipeline.rs",
                "<dep>/rapier3d-f64-0.34.0/src/pipeline.rs",
            ),
            (
                "C:\\Users\\someone\\Desktop\\Infinity_Engine\\infinity_engine\\runtime\\inf-player\\src\\lib.rs",
                "runtime/inf-player/src/lib.rs",
            ),
            (
                "C:\\Users\\someone\\build\\weird\\thing.rs",
                "<build>/thing.rs",
            ),
            // Already relative: cargo's ordinary shape for a workspace member.
            ("runtime/inf-player/src/log.rs", "runtime/inf-player/src/log.rs"),
        ];
        for (raw, want) in cases {
            let got = sanitize_location(raw);
            assert_eq!(got, want, "sanitizing {raw}");
            assert!(
                !got.to_ascii_lowercase().contains("users")
                    && !got.contains("someone")
                    && !got.contains("home/"),
                "the sanitized location still names a machine: {got}"
            );
        }
    }

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
