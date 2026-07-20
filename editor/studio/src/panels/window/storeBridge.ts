import { emit, emitTo, listen } from "@tauri-apps/api/event";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import type { StoreApi, UseBoundStore } from "zustand";

/**
 * Cross-window store mirroring (ported from GeoCanvas, ROADMAP P1.2.4).
 * The MAIN window is authoritative for all app state; detached panel
 * windows run read-only MIRRORS of a whitelist of zustand stores and
 * forward every mutation back as an event.
 *
 * Protocol (all payloads JSON, `panel://` namespaced per §2.4):
 *  - `panel://sync`         main → all    { store, state } — one store's
 *    data slice (functions stripped), emitted on change, microtask-coalesced.
 *  - `panel://sync/request` panel → main  { label } — a booting panel
 *    window asks for full snapshots; main answers per-store directly to
 *    that label.
 *  - `panel://action`       panel → main  { store, action, args } — main
 *    executes `store.getState()[action](...args)` (fire-and-forget).
 *
 * Mechanics: the mirror applies data slices with `setState` (functions
 * survive — zustand merges), and REPLACES every action function with a
 * forwarder at bootstrap, so panel components call the exact same hooks
 * and actions they do in the main window, unchanged. Whole-snapshot sync
 * is deliberately simple + self-healing (a lost event heals on the next
 * change / focus resync).
 *
 * Stores opt in via `registerBridgedStore` (avoids import cycles and keeps
 * the bridge ignorant of domain stores). Snapshots only flow while ≥1
 * panel window is open (`setPanelWindowCount`).
 */

type AnyStore = UseBoundStore<StoreApi<Record<string, unknown>>>;

const BRIDGED = new Map<string, AnyStore>();

/** Whitelist a store for cross-window mirroring. Call at module scope of
 *  the store's own file (both windows must register the same set). */
export function registerBridgedStore(name: string, store: unknown): void {
  BRIDGED.set(name, store as AnyStore);
}

/** The serializable slice of a store's state (drop the actions). */
function dataSlice(state: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(state)) {
    if (typeof v !== "function") out[k] = v;
  }
  return out;
}

// =============================================================================
// Main-window side
// =============================================================================

let panelWindowCount = 0;
/** The window manager reports how many panel windows are live — snapshot
 *  emission is gated on > 0 so the main window pays nothing otherwise. */
export function setPanelWindowCount(n: number): void {
  const wasZero = panelWindowCount === 0;
  panelWindowCount = n;
  // First window just opened → push a full snapshot so it can't miss
  // changes that landed before its own sync request resolves.
  if (wasZero && n > 0) emitAllSnapshots();
}

function emitAllSnapshots(): void {
  for (const [name, store] of BRIDGED) {
    void emit("panel://sync", { store: name, state: dataSlice(store.getState()) });
  }
}

/** Install the host side (main window only). Returns the uninstaller. */
export function startStoreBridgeHost(): () => void {
  const unsubs: Array<() => void> = [];
  const pending = new Set<string>();
  let flushQueued = false;

  const flush = () => {
    flushQueued = false;
    for (const name of pending) {
      const store = BRIDGED.get(name);
      if (!store) continue;
      void emit("panel://sync", { store: name, state: dataSlice(store.getState()) });
    }
    pending.clear();
  };

  for (const [name, store] of BRIDGED) {
    unsubs.push(
      store.subscribe(() => {
        if (panelWindowCount === 0) return;
        pending.add(name);
        if (!flushQueued) {
          flushQueued = true;
          queueMicrotask(flush);
        }
      }),
    );
  }

  // `listen` resolves ASYNC while the uninstaller runs sync — if cleanup
  // fires before the promise resolves (React StrictMode's dev double-mount,
  // or any DockWorkspace remount), a naive `.then(u => arr.push(u))` pushes
  // into an abandoned array and the listener LEAKS: two live
  // `panel://action` handlers apply every forwarded action TWICE.
  // `disposed` makes the race safe: a late-resolving unlisten self-executes.
  let disposed = false;
  const unlistens: Array<() => void> = [];
  const track = (p: Promise<() => void>) => {
    void p.then((u) => {
      if (disposed) u();
      else unlistens.push(u);
    });
  };

  track(
    listen<{ store: string; action: string; args: unknown[] }>("panel://action", (e) => {
      const store = BRIDGED.get(e.payload.store);
      const fn = store?.getState()[e.payload.action];
      if (typeof fn === "function") {
        try {
          (fn as (...a: unknown[]) => void)(...(e.payload.args ?? []));
        } catch (err) {
          console.error(
            `[storeBridge] forwarded action ${e.payload.store}.${e.payload.action} threw:`,
            err,
          );
        }
      }
    }),
  );

  track(
    listen<{ label: string }>("panel://sync/request", (e) => {
      for (const [name, store] of BRIDGED) {
        void emitTo(e.payload.label, "panel://sync", {
          store: name,
          state: dataSlice(store.getState()),
        });
      }
    }),
  );

  return () => {
    disposed = true;
    for (const u of unsubs) u();
    for (const u of unlistens) u();
    unlistens.length = 0;
  };
}

// =============================================================================
// Panel-window side
// =============================================================================

/** Install the mirror side (panel windows only): wrap actions into
 *  forwarders, subscribe to sync events, request the initial snapshot. */
export async function startStoreBridgeMirror(): Promise<() => void> {
  // Replace every action with a forwarder BEFORE any UI renders, so no
  // stray local mutation can run against the mirror.
  for (const [name, store] of BRIDGED) {
    const wrapped: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(store.getState())) {
      if (typeof v !== "function") continue;
      wrapped[k] = (...args: unknown[]) => {
        void emitTo("main", "panel://action", { store: name, action: k, args });
      };
    }
    store.setState(wrapped as never);
  }

  const unlisten = await listen<{ store: string; state: Record<string, unknown> }>(
    "panel://sync",
    (e) => {
      BRIDGED.get(e.payload.store)?.setState(e.payload.state as never);
    },
  );

  const label = getCurrentWebviewWindow().label;
  void emit("panel://sync/request", { label });
  // Belt-and-braces resync whenever the window regains focus.
  const onFocus = () => void emit("panel://sync/request", { label });
  window.addEventListener("focus", onFocus);

  return () => {
    unlisten();
    window.removeEventListener("focus", onFocus);
  };
}
