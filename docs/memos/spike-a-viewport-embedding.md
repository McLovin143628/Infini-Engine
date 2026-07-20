# Spike A memo — native wgpu viewport embedded in Tauri v2 (Windows)

**Status:** Windows path verified end-to-end (embedding, resize sync, flycam,
drag-drop handoff); macOS path implemented and compile-verified. Ongoing
through Phase 0/2; this memo records decisions and findings as they harden.

## Decisions

1. **Embedding model** — a `WS_CHILD | WS_CLIPSIBLINGS` window parented to the Tauri
   top-level HWND, pushed to `HWND_TOP` so it sits above the WebView2 sibling. The React
   layout reserves a "hole" (`ViewportPanel`), reports its rect via `ResizeObserver` →
   `viewport_set_rect` (physical px = CSS px × devicePixelRatio). Confirmed workable.
2. **Threading** — the child window and its wgpu surface live on a dedicated
   `inf-viewport` thread (Win32 ties a window to its creating thread). Commands arrive on
   an mpsc channel; rect updates are coalesced to the latest before applying. FIFO/vsync
   present paces the loop.
3. **Backend: Vulkan (temporary), not DX12.** wgpu-hal 30's DX12 path requires
   `windows` 0.62 while tauri 2.11 pins `windows` 0.61; cargo puts gpu-allocator
   (range `>=0.53,<=0.62`) in tauri's 0.61 bucket, and wgpu-hal's DX12 code then fails to
   compile against mismatched types. The `dx12` feature is compiled out
   (root `Cargo.toml`), backends = `VULKAN | METAL`. **Revisit when tauri moves to
   windows 0.62** (or wgpu-hal relaxes to 0.61) — DX12 remains the intended production
   backend on Windows for swapchain/flip-model quality.
4. **Idempotent attach** — `viewport_attach` ignores repeat calls (React StrictMode
   double-mounts in dev).
5. **Vulkan Win32 surfaces need `hinstance`.** `Win32WindowHandle::new(hwnd)` alone fails
   with "Vulkan requires raw-window-handle's Win32::hinstance to be set" — pass the module
   handle alongside the HWND. Cost us the first run; now baked into `SurfaceTarget`.
6. **Never call Win32 APIs while holding the input-state `RefCell` borrow.**
   `ReleaseCapture()` dispatches `WM_CAPTURECHANGED` *synchronously* back into the same
   wnd_proc; a nested `borrow_mut()` panics, and a panic in a non-unwinding
   `extern "system"` fn aborts the whole process (0xc0000409). Found the hard way (RMB
   release crashed the editor); all handlers now copy state out, drop the borrow, then
   call the API. This pattern is a standing rule for every future wnd_proc.
7. **Flycam input** — RMB press: `SetCapture` + `SetFocus` + hide cursor + raw
   `WM_INPUT` mouse deltas (`RegisterRawInputDevices`, usage 1/2); WASD/QE polled via
   `GetAsyncKeyState` per frame; wheel scales fly speed while captured; release restores
   the cursor to its press position. Injected input (SendInput-style) also arrives as
   WM_INPUT, which makes the path scriptable for tests.
8. **White-flash suppression** — `WM_ERASEBKGND → 1` (the swapchain covers every pixel).
   Splitter drags stay clean: mid-drag screenshots show the native window tracking the
   hole exactly, no background strip.
9. **Drag-drop handoff works via webview mouse capture.** During an HTML pointer drag the
   Chromium side holds OS mouse capture, so `pointermove`/`pointerup` keep firing even
   while the cursor is over the native child window (which normally swallows input). The
   HTML ghost is invisible over the hole (airspace rule) — expected; the drop point
   crosses via `viewport_drop(x, y, payload)` in hole-local physical px. Verified:
   engine log shows the drop with correct coordinates.
10. **DPI changes** — `ResizeObserver` + window `resize` + a re-arming
    `matchMedia("(resolution: …dppx)")` listener (cross-monitor drags can change
    devicePixelRatio without a resize event).
11. **macOS embedding = CAMetalLayer sublayer, not a child NSView.** `inf-viewport`'s
    macOS path adds a `CAMetalLayer` to the layer-backed contentView (main-thread setup,
    dispatched by the Ring-2 command via `run_on_main_thread`), renders from a thread via
    `SurfaceTargetUnsafe::CoreAnimationLayer`, and applies rect updates (px → points,
    bottom-left flip vs. superlayer bounds) inside `CATransaction` with actions disabled.
    Compile-verified against `aarch64-apple-darwin` from Windows; **runtime pass needs Mac
    hardware** — retina scale, coordinate flip, off-main-thread layer geometry, and scale
    changes across monitors are the open questions. Flycam input is not wired on macOS.

## Verified (2026-07-19, Windows 11, RTX 4070 Ti, Vulkan)

- [x] Workspace + editor compile with the embedded-viewport path (clippy/fmt clean).
- [x] 3D scene (infinite 1 m/10 m grid, axis lines, spinning triangle) renders in the
      native child window inside the editor shell, correctly framed by the React chrome.
- [x] Splitter resize without white flash (WM_ERASEBKGND + coalesced rect sync;
      mid-drag capture shows exact tracking).
- [x] RMB capture + raw-input flycam (yaw/pitch from WM_INPUT deltas, WASD fly,
      wheel speed, cursor restore; verified with injected input + screenshots).
- [x] Drag-drop coordinate handoff (Outliner chip → hole → `viewport_drop` → engine
      log with hole-local px).
- [x] DPI-change plumbing (dpr listener); 100 % scale verified.
- [ ] DPI matrix: 150 % / 200 %, cross-monitor drag (needs a manual pass — change
      Windows display scale and drag between mixed-DPI monitors).
- [x] macOS NSView/CAMetalLayer port — implemented, `cargo check
      --target aarch64-apple-darwin` clean; runtime verification needs Mac hardware.

## Known consequences (airspace rule)

HTML can never draw over the viewport hole. Gizmos/overlays must be engine-rendered;
menus near the viewport edge must flip inward; drag ghosts die over the hole (IPC handoff
required — now proven). These are Phase 2 work items, not spike blockers.
