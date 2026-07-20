# Spike A memo — native wgpu viewport embedded in Tauri v2 (Windows)

**Status:** first vertical slice landed (child HWND + wgpu triangle + React rect sync).
Ongoing through Phase 0/2; this memo records decisions and findings as they harden.

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
   (root `Cargo.toml`), backends = `VULKAN`. **Revisit when tauri moves to windows 0.62**
   (or wgpu-hal relaxes to 0.61) — DX12 remains the intended production backend on
   Windows for swapchain/flip-model quality.
4. **Idempotent attach** — `viewport_attach` ignores repeat calls (React StrictMode
   double-mounts in dev).
5. **Vulkan Win32 surfaces need `hinstance`.** `Win32WindowHandle::new(hwnd)` alone fails
   with "Vulkan requires raw-window-handle's Win32::hinstance to be set" — pass the module
   handle alongside the HWND. Cost us the first run; now baked into `Renderer::new`.

## Verified (2026-07-19, Windows 11, RTX 4070 Ti, Vulkan)

- [x] Workspace + editor compile with the embedded-viewport path (clippy/fmt clean).
- [x] Spinning triangle renders in the native child window inside the editor shell,
      correctly framed by the React chrome (screenshots in session log; rotation confirmed
      across two captures).
- [ ] 60 fps splitter resize without white flash.
- [ ] DPI matrix: 100% / 150% / 200%, cross-monitor drag.
- [ ] RMB capture + raw-input flycam prototype.
- [ ] Drag-drop coordinate handoff stub.
- [ ] macOS NSView/CAMetalLayer port.

## Known consequences (airspace rule)

HTML can never draw over the viewport hole. Gizmos/overlays must be engine-rendered;
menus near the viewport edge must flip inward; drag ghosts die over the hole (IPC handoff
required). These are Phase 2 work items, not spike blockers.
