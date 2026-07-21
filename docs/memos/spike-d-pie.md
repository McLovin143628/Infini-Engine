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

## P9.4 — PIE productionized (2026-07-21)

The Spike D machinery is now the real Play-In-Editor feature. What landed:

1. **Real content handoff (PIE == shipping).** The wire protocol went to
   **v2**: a **versioned little-endian frame header** (`magic u32` +
   `frame_ver u16` + `len u32` + bincode) self-describes every frame, and a
   new `EditorToPlayer::LoadScene(ScenePayload)` streams the **live** editor
   scene — v3 `.inf_lvl` bytes of the *unsaved-included* `SceneDoc` + the
   bound blueprint classes as `(guid, json)`. The player builds its world
   through the **same `InfSceneWorldBuilder::with_bindings`** the cooked-pack
   boot uses. The gate: `pie_scene_trace_matches_shipping` streams the live
   platformer through a real subprocess and asserts its per-step xxh3 trace
   is **byte-identical** to the in-process pack-path build (`scene_trace`).
   Previewing cannot diverge from shipping.
2. **Control surface.** `Pause/Resume/Step{count}/Stop/Eject/SetViewport`
   control frames + `Window{handle}` / `State` / `Ejected` / `Error`
   reports on top of Spike D. Real headless PIE is **step-driven** (starts
   paused) so the determinism gate reads exactly N frames; the windowed path
   auto-runs. `Eject` is v1 = release input possession (a clean ack; true
   camera hand-back is deferred — **NOT faked**).
3. **Editor side.** `inf_editor_core::pie` (Ring 1 — already the subprocess
   owner since Spike D, so process IO here needs no new ring exception) gains
   `PieSession::spawn_scene`, the control methods, `build_scene_payload`
   (serializes the live doc + resolves bound classes, mirroring
   `samples::bound_actors`), and `find_player_bin` (sibling-of-exe +
   `INF_PLAYER_BIN` override). Ring-2 `commands/pie.rs` stays thin: spawn +
   a **crash-monitor thread** (waits on the child → crash toast + viewport
   restore + toolbar reset, editor intact) + `pie://state` events. Tests
   (in `inf-player/tests/pie.rs`, which dev-deps editor-core): PIE==shipping,
   step/pause/stop, real-content crash isolation + panic-text capture,
   zombie-free stop (graceful reap + `Drop` kill).
4. **Embedded window — HONEST status.** The **proven** `SetParent` sequence
   is wired into `inf-viewport` as `embed_foreign`/`release_foreign`, run on
   the **viewport render thread** (which pumps) with the parent set to the
   **Tauri main window** — so the child's teardown `WM_PARENTNOTIFY` lands on
   the always-pumping main thread, sidestepping the Spike D deadlock. The
   player reports its HWND; the monitor reparents it into the hole, hides the
   native viewport child (the refcounted `set_visible` discipline), and
   follows rect changes; stop/crash restores everything. **This live embed is
   Windows-only and human-verify-only** (GPU + window; not exercised in CI,
   like every GPU path). On non-Windows — or if the player never reports a
   usable HWND — PIE runs as **"Play in New Window"**, the roadmap-sanctioned
   fallback, which is the always-working path and the primary UX on
   macOS/Linux. The frontend split-button offers Embedded / New Window /
   Simulate explicitly; nothing about embed success is faked.

**Deferred (documented, not faked):** true camera possession + editor-camera
hand-back on Eject; input routing from the embedded window back through the
editor's focus-handoff channel (the player owns its own input today); a
flash-free embed (the reparent momentarily shows the player window top-level
before adoption); macOS/Wayland embedded PIE (needs shared-surface streaming).
