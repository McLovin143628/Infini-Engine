//! Editor-side PIE session management (Spike D).
//!
//! Spawns `inf-player --pie` as a subprocess, hands it the cooked snapshot
//! over its stdin, and consumes its event stream. The subprocess boundary
//! is the crash-isolation guarantee: a panicking script kills the *player*,
//! and the editor observes it as [`SessionHealth::Exited`] with the panic
//! text captured from stderr — unsaved editor state is never at risk.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::Receiver;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use inf_runtime::pie::{read_msg, write_msg, EditorToPlayer, PlayerToEditor, PIE_PROTOCOL_VERSION};
use inf_runtime::CookedSnapshot;

#[derive(Debug, thiserror::Error)]
pub enum PieError {
    #[error("io error talking to the player: {0}")]
    Io(#[from] std::io::Error),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("player exited during handshake ({0})")]
    HandshakeExit(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionHealth {
    Running,
    /// The player process is gone. `code` is `None` if the OS reported no
    /// exit code (e.g. killed by signal).
    Exited {
        code: Option<i32>,
    },
}

/// A live play-in-editor session (one player subprocess).
pub struct PieSession {
    child: Child,
    stdin: ChildStdin,
    events: Receiver<PlayerToEditor>,
    stderr_lines: Arc<Mutex<Vec<String>>>,
}

impl PieSession {
    /// Spawn `player_bin --pie --tick-hz N`, complete the `Ready` handshake,
    /// send the snapshot, and wait for `Loaded`.
    pub fn spawn(
        player_bin: &Path,
        snapshot: &CookedSnapshot,
        tick_hz: u32,
    ) -> Result<Self, PieError> {
        let mut child = Command::new(player_bin)
            .args(["--pie", "--tick-hz", &tick_hz.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        // Protocol frames: reader thread → channel; ends on EOF (player
        // exit) by dropping the sender.
        let (tx, events) = mpsc::channel();
        std::thread::spawn(move || {
            let mut stdout = BufReader::new(stdout);
            while let Ok(msg) = read_msg::<PlayerToEditor>(&mut stdout) {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });

        // Player logs (and panic messages) line-buffered off stderr.
        let stderr_lines = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&stderr_lines);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                sink.lock().expect("stderr sink poisoned").push(line);
            }
        });

        let mut session = PieSession {
            child,
            stdin,
            events,
            stderr_lines,
        };

        match session.next_event(Duration::from_secs(10)) {
            Some(PlayerToEditor::Ready { protocol }) if protocol == PIE_PROTOCOL_VERSION => {}
            Some(PlayerToEditor::Ready { protocol }) => {
                return Err(PieError::Protocol(format!(
                    "player speaks protocol {protocol}, editor speaks {PIE_PROTOCOL_VERSION}"
                )));
            }
            Some(other) => {
                return Err(PieError::Protocol(format!("expected Ready, got {other:?}")));
            }
            None => return Err(PieError::HandshakeExit(session.describe_exit())),
        }

        session.send(&EditorToPlayer::Load(snapshot.clone()))?;
        match session.next_event(Duration::from_secs(10)) {
            Some(PlayerToEditor::Loaded { .. }) => Ok(session),
            Some(other) => Err(PieError::Protocol(format!(
                "expected Loaded, got {other:?}"
            ))),
            None => Err(PieError::HandshakeExit(session.describe_exit())),
        }
    }

    pub fn send(&mut self, msg: &EditorToPlayer) -> Result<(), PieError> {
        write_msg(&mut self.stdin, msg)?;
        Ok(())
    }

    /// Next player event, or `None` if the stream ended / nothing arrived
    /// within `timeout`.
    pub fn next_event(&self, timeout: Duration) -> Option<PlayerToEditor> {
        self.events.recv_timeout(timeout).ok()
    }

    /// Drain events until `matches` returns true; `None` on timeout or
    /// stream end. Non-matching events are consumed.
    pub fn wait_for(
        &self,
        timeout: Duration,
        mut matches: impl FnMut(&PlayerToEditor) -> bool,
    ) -> Option<PlayerToEditor> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.checked_duration_since(Instant::now())?;
            let event = self.next_event(remaining)?;
            if matches(&event) {
                return Some(event);
            }
        }
    }

    pub fn health(&mut self) -> SessionHealth {
        match self.child.try_wait() {
            Ok(Some(status)) => SessionHealth::Exited {
                code: status.code(),
            },
            Ok(None) => SessionHealth::Running,
            Err(_) => SessionHealth::Exited { code: None },
        }
    }

    /// Poll-wait for the player to exit (used after Stop / a crash).
    pub fn wait_exit(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                return Some(status);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Everything the player wrote to stderr so far (its logs; after a
    /// crash, the panic message).
    pub fn stderr_lines(&self) -> Vec<String> {
        self.stderr_lines
            .lock()
            .expect("stderr sink poisoned")
            .clone()
    }

    fn describe_exit(&mut self) -> String {
        let health = self.health();
        format!("{health:?}; stderr: {:?}", self.stderr_lines())
    }

    /// Graceful shutdown: send Stop, wait for exit (kill on timeout).
    pub fn stop(mut self, timeout: Duration) -> Result<ExitStatus, PieError> {
        // The player may already be gone; a send failure is fine.
        let _ = self.send(&EditorToPlayer::Stop);
        match self.wait_exit(timeout) {
            Some(status) => Ok(status),
            None => {
                self.child.kill()?;
                Ok(self.child.wait()?)
            }
        }
    }
}

impl Drop for PieSession {
    fn drop(&mut self) {
        // Never leave an orphaned player: if it's still running when the
        // session handle dies, kill it.
        if matches!(self.health(), SessionHealth::Running) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
