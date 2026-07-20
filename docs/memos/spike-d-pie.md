# Spike D memo — play-in-editor process model

**Status:** GO on subprocess-first PIE. `inf-player --pie` runs the cooked
snapshot in its own process, streams frames back over a local channel, and a
deliberately injected script panic kills only the player — the editor
observes the crash, keeps its state, captures the panic text for the Output
Log, and restarts a fresh session immediately. The cross-process
window-embedding experiment (Windows) also passed, with one hard-won
caveat recorded below. Crates: `inf-runtime` (sim/snapshot/protocol),
`inf-player` (the subprocess), `inf-editor-core::pie` (the editor side).

## Decisions

1. **Subprocess-first** (per roadmap): PIE is `inf-player` spawned by the
   editor — the same binary a shipped game uses — fed the same cooked
   snapshot bytes the packager will produce (`CookedSnapshot`, bincode +
   schema version). Preview cannot diverge from shipping, and no script
   can take down unsaved editor state. In-process Simulate mode
   (physics/PCG only) remains a cheap later addition (P9.4).
2. **Local channel = stdin/stdout pipes.** In `--pie` mode stdout carries
   *only* protocol frames (`u32` LE length + bincode of
   `EditorToPlayer`/`PlayerToEditor`); human logs — including the panic
   message on a crash — go to stderr, which the editor tails into the
   Output Log. No sockets, no ports, works identically on all three OSes,
   and the pipe closing doubles as liveness detection in both directions.
3. **Protocol v1**: `Ready{protocol}` handshake (version-checked) →
   `Load(snapshot)` → `Loaded` → `Frame{frame, state_hash, actors}` stream;
   `Pause/Resume/Stop` acked; `InjectPanic` is the permanent QA hook for
   crash drills. Oversized frames are rejected before allocation.
4. **Determinism is a CI assertion, not a hope.** The fixed-step world
   (60 Hz, pure f64, ordered actors) hashes its full state (xxh3 of the
   bincode encoding); `--headless --run-frames N` prints it, and tests
   assert two subprocess runs and an in-process run of the same snapshot
   agree bit-for-bit. This is the seed of the replay/parity gates the
   roadmap requires later.
5. **Crash isolation**: the player does *not* catch script panics — the
   process dies (that is the design; in-process containment is Spike C's
   job for the hot-reload tier). The editor's `PieSession` sees the exit
   status, exposes captured stderr, and its `Drop` kills any
   still-running player so sessions can never leak.
6. **Window embedding (the experiment):** a probe mode creates a bare
   Win32 window and a test in the editor's role reparents it cross-process
   (`SetWindowLongPtr` → `WS_CHILD`, then `SetParent`) into its own window
   tree — works, so P9's "PIE window inside the viewport hole" plan is
   viable on Windows with Spike A's rect-sync machinery on top.
   **Finding paid for in a deadlock:** while hosting a foreign child
   window, the embedding thread must *never block without pumping
   messages* — the child's `DestroyWindow` sends `WM_PARENTNOTIFY`
   synchronously to the parent, so a blocking `wait()` on the child
   process deadlocks both sides. The editor's embed path must keep its
   message loop alive during Stop/teardown (same family as the Spike A
   wnd_proc reentrancy rule).
   "Play in New Window" stays the v1 default.

## Platform honesty

- **macOS cannot reparent another process's window** (no cross-process
  NSView adoption). PIE on macOS is "Play in New Window" v1; a true
  embedded PIE would need shared-surface streaming (IOSurface) — a P9+
  investigation, not promised.
- Linux/X11 `XReparentWindow` is cross-process-capable (same family as
  the Windows result); Wayland has no reparenting — same streaming
  fallback story as the Spike A viewport.

## Verified (2026-07-19, Windows 11; suite runs in CI from this commit on)

- `--headless --run-frames 240` deterministic across runs and vs
  in-process stepping (`final-state-hash` identical).
- PIE session: handshake, snapshot handoff, advancing `Frame` stream with
  moving actors, `Pause` silences the stream / `Resume` restarts it,
  `Stop` → `Stopped` → exit 0.
- Crash drill: `InjectPanic` → nonzero exit observed via
  `SessionHealth::Exited`, panic text captured from stderr, fresh session
  spawns immediately afterward.
- Embed probe: foreign player window reparented into the test's window
  tree (`GetParent` == host, `WS_CHILD` applied), probe exits cleanly on
  stdin close — with the pump-while-waiting rule applied.
- Protocol/sim/snapshot unit tests: framing round trip + EOF + oversize
  rejection; bounce stays in bounds over 100k steps; snapshot
  encode/decode + schema-version rejection.

## Open items

- Pipe backpressure: at very high tick rates an editor that stops reading
  would eventually block the player on a full stdout pipe. Fine for now
  (the editor always drains); a frame-coalescing send policy is a P9
  refinement.
- Frame messages currently carry full actor lists; deltas/interest
  management arrive with the real renderer handoff (P9).
- The embedded-window path needs the Spike A rect-sync + DPI treatment
  when it becomes a real feature (P9.4 batch 3); the probe only proves
  the OS primitive.
