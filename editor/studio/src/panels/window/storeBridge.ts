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
 *  - `panel://sync`         main → all/one  ONE store's update, carrying a
 *    per-store `seq`. Two shapes on the SAME channel so their order is the
 *    channel's order:
 *      · `{ store, seq, state }` — a full data slice (functions stripped).
 *      · `{ store, seq, patch }` — only what changed since `seq - 1`.
 *  - `panel://sync/request` panel → main    { label } — a booting (or
 *    out-of-step) panel window asks for full snapshots; main answers per-store
 *    directly to that label.
 *  - `panel://action`       panel → main    { store, action, args } — main
 *    executes `store.getState()[action](...args)` (fire-and-forget).
 *
 * Mechanics: the mirror applies data slices with `setState` (functions
 * survive — zustand merges), and REPLACES every action function with a
 * forwarder at bootstrap, so panel components call the exact same hooks
 * and actions they do in the main window, unchanged.
 *
 * Stores opt in via `registerBridgedStore` (avoids import cycles and keeps
 * the bridge ignorant of domain stores). Snapshots only flow while ≥1
 * panel window is open (`setPanelWindowCount`).
 *
 * ## Why there are patches (F-lens L7.H4)
 *
 * This bridge used to ship every bridged store's WHOLE data slice on every
 * mutation, and carried a note calling that "cheap for today's small stores".
 * The note had gone stale on both of its own examples:
 *
 *  - `logStore.lines` is a **5000-entry** ring, and the `log://line` batcher
 *    flushes roughly every 60 ms — so a busy build re-serialized the entire
 *    scrollback about **16 times a second**, per open panel window.
 *  - `sceneStore.nodes` is the **full scene node map**, rebuilt by `applyDelta`
 *    on every `world://delta` — so moving one actor re-shipped every actor.
 *
 * Both are fixed generically rather than per-store, because the next hot store
 * to be bridged should not have to rediscover this. `diffSlice` compares the
 * slice against the last one sent and emits only what moved, with two shapes
 * beyond a plain value:
 *
 *  - **arrays** are matched as `append + head-trim` — exactly what a capped
 *    ring buffer does — so a flush ships the handful of new lines and a count,
 *    not 5000 objects;
 *  - **plain objects** get a shallow key diff, so a delta that touched three
 *    nodes ships three nodes and a removal list.
 *
 * Anything that does not fit either shape falls back to sending the value
 * whole, and a patch that would not be smaller than the value is not sent as a
 * patch at all. Self-healing survives: `seq` is per store and strictly
 * increasing, a mirror that sees a gap asks for a full resync, and every full
 * snapshot re-bases the sequence.
 */

type AnyStore = UseBoundStore<StoreApi<Record<string, unknown>>>;

const BRIDGED = new Map<string, AnyStore>();

/** One key's change. `set` is "take this value"; the other two are deltas. */
export type KeyPatch =
  | { t: "set"; v: unknown }
  /** `next = [...cur.slice(trim), ...append]` — the capped-ring shape. */
  | { t: "arr"; trim: number; append: unknown[] }
  /** `next = {...cur} minus `del`, then `set` overlaid` — a shallow key diff. */
  | { t: "rec"; set: Record<string, unknown>; del: string[] };

export type SlicePatch = Record<string, KeyPatch>;

/** A `panel://sync` message: a full slice, or a patch onto the previous one. */
type SyncMessage = {
  store: string;
  seq: number;
  state?: Record<string, unknown>;
  patch?: SlicePatch;
};

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * An array as `append + head-trim`, or `null` when it is not that shape.
 *
 * The overlap is found by locating the previous array's LAST element in the
 * next one (reference identity — these are the same objects, not copies) and
 * then verifying the whole shared run element by element. That verification is
 * O(overlap) reference compares — a few thousand at the very worst, a few times
 * a second — and it is what makes the fast path safe to trust: without it, a
 * coincidental match would silently desync a mirror window's log.
 */
function arrayPatch(prev: unknown[], next: unknown[]): KeyPatch | null {
  if (prev.length === 0 || next.length === 0) return null;
  const j = next.lastIndexOf(prev[prev.length - 1]);
  if (j < 0) return null;
  const trim = prev.length - 1 - j;
  if (trim < 0) return null;
  for (let i = 0; i <= j; i++) {
    if (prev[trim + i] !== next[i]) return null;
  }
  const append = next.slice(j + 1);
  // A patch that carries nearly the whole array is just a slower full send.
  //
  // The bound is `j`, not a length comparison: `append` is `next.slice(j + 1)`
  // and `j >= 0` here, so `append.length >= next.length` is unreachable — the
  // guard it replaces could never fire (a round-2 LOW). What it MEANT is this,
  // and now that it can fire, the fast path is bounded rather than merely
  // believed to be.
  if (append.length * 2 >= next.length && next.length > 8) return null;
  return { t: "arr", trim, append };
}

/** A shallow key diff of two plain objects, or `null` when it saves nothing. */
function recordPatch(
  prev: Record<string, unknown>,
  next: Record<string, unknown>,
): KeyPatch | null {
  const set: Record<string, unknown> = {};
  const del: string[] = [];
  for (const k of Object.keys(next)) {
    if (!Object.is(prev[k], next[k])) set[k] = next[k];
  }
  for (const k of Object.keys(prev)) {
    if (!(k in next)) del.push(k);
  }
  if (Object.keys(set).length + del.length >= Object.keys(next).length) return null;
  return { t: "rec", set, del };
}

/**
 * What changed between two data slices, or `null` when nothing did.
 *
 * Only keys present in `next` are considered — matching `setState`'s merge, in
 * which a key the host dropped never disappeared from the mirror anyway.
 */
export function diffSlice(
  prev: Record<string, unknown>,
  next: Record<string, unknown>,
): SlicePatch | null {
  const patch: SlicePatch = {};
  let changed = 0;
  for (const k of Object.keys(next)) {
    const a = prev[k];
    const b = next[k];
    if (Object.is(a, b)) continue;
    changed += 1;
    const delta =
      Array.isArray(a) && Array.isArray(b)
        ? arrayPatch(a, b)
        : isPlainObject(a) && isPlainObject(b)
          ? recordPatch(a, b)
          : null;
    patch[k] = delta ?? { t: "set", v: b };
  }
  return changed === 0 ? null : patch;
}

/** Apply a patch onto a mirror's current state, returning the keys to merge. */
export function applySlicePatch(
  cur: Record<string, unknown>,
  patch: SlicePatch,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, p] of Object.entries(patch)) {
    if (p.t === "set") {
      out[k] = p.v;
    } else if (p.t === "arr") {
      const base = Array.isArray(cur[k]) ? (cur[k] as unknown[]) : [];
      out[k] = [...base.slice(p.trim), ...p.append];
    } else {
      const base = isPlainObject(cur[k]) ? cur[k] : {};
      const merged: Record<string, unknown> = { ...base };
      for (const d of p.del) delete merged[d];
      Object.assign(merged, p.set);
      out[k] = merged;
    }
  }
  return out;
}

/** Whitelist a store for cross-window mirroring. Call at module scope of
 *  the store's own file (both windows must register the same set). */
export function registerBridgedStore(name: string, store: unknown): void {
  BRIDGED.set(name, store as AnyStore);
}

/**
 * The serializable slice of a store's state (drop the actions).
 *
 * COST NOTE (accurate as of F-lens L7.H4): a *change* now ships only what moved
 * (see `diffSlice` and the module docs). A full slice is still built here on
 * every flush — to diff against — and is still SENT whole in three cases: the
 * first update after a panel window opens, an explicit `panel://sync/request`,
 * and any key whose new value does not admit a smaller patch. Emission of any
 * kind is gated on `panelWindowCount > 0` (see `setPanelWindowCount` and the
 * `subscribe` guard below), so with no detached window open the main window
 * pays nothing at all regardless of store size.
 *
 * Bridged state must stay JSON-plain: this walks one level and copies values by
 * reference, and the payload crosses a Tauri event, so a `Map`/`Set` in a
 * bridged store would arrive at the mirror as `{}`. (Today's three — shell,
 * scene, log — are plain; `assetStore`'s `thumbnails` Map is deliberately not
 * bridged.)
 */
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

/**
 * Per-store update counter and the last slice every up-to-date mirror holds.
 *
 * `lastSent` is what patches are computed against, so it must be re-based
 * whenever a full snapshot goes out to everyone. While `panelWindowCount === 0`
 * nothing is emitted and `lastSent` goes stale on purpose — `setPanelWindowCount`
 * re-bases it the moment the first window opens.
 */
const seqOf = new Map<string, number>();
const lastSent = new Map<string, Record<string, unknown>>();

/** The window manager reports how many panel windows are live — snapshot
 *  emission is gated on > 0 so the main window pays nothing otherwise. */
export function setPanelWindowCount(n: number): void {
  const wasZero = panelWindowCount === 0;
  panelWindowCount = n;
  // First window just opened → push a full snapshot so it can't miss
  // changes that landed before its own sync request resolves.
  if (wasZero && n > 0) emitAllSnapshots();
  // …and the last one just closed (round-2 finding R2-4). `lastSent` holds one
  // whole DATA SLICE per bridged store — the 5 000-line log ring, the full
  // scene node map — as the baseline patches are computed against. With no
  // window open nothing is emitted and the baseline is stale on purpose, so it
  // is not merely useless: it is a copy of the largest objects in the editor,
  // pinned for the rest of the session by a feature nobody is using.
  //
  // Dropping it is free. `emitAllSnapshots` above re-bases from live state the
  // moment a window opens again, and `flush` treats a missing baseline as a
  // full send, which is the correct answer for a mirror that holds nothing.
  if (n === 0) lastSent.clear();
}

function emitAllSnapshots(): void {
  for (const [name, store] of BRIDGED) {
    const state = dataSlice(store.getState());
    // Broadcast, so every mirror is now at this slice: it becomes the patch
    // baseline. The sequence is NOT advanced — nothing changed, this is a
    // re-statement of where things already stand.
    lastSent.set(name, state);
    void emit("panel://sync", { store: name, seq: seqOf.get(name) ?? 0, state } satisfies SyncMessage);
  }
}

/**
 * **How many baseline slices the host is holding** — Wave B's law, applied: a
 * fix whose whole effect is that something is no longer retained needs an
 * accessor, or a mutation that restores the retention passes every test.
 *
 * Test-only; nothing in the app reads it.
 */
export function __bridgeBaselineCount(): number {
  return lastSent.size;
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
      const next = dataSlice(store.getState());
      const prev = lastSent.get(name);
      const patch = prev ? diffSlice(prev, next) : null;
      // A subscriber fired but nothing in the DATA slice moved (an action
      // identity changed, or a `set` that wrote equal references): the mirrors
      // are already correct, so nothing is sent and the sequence stands.
      if (prev && !patch) continue;
      const seq = (seqOf.get(name) ?? 0) + 1;
      seqOf.set(name, seq);
      lastSent.set(name, next);
      void emit(
        "panel://sync",
        patch ? { store: name, seq, patch } : { store: name, seq, state: next },
      );
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
        // **The baseline, not the live state.** This reply is targeted, so
        // `lastSent` cannot be re-based — the other windows are still in step
        // with it, and the next patch will be computed against it. Sending the
        // live state instead would put this one window ahead of the sequence,
        // and the next patch would then be applied to the wrong base. On the
        // first request of a session there is no baseline yet, so the live
        // slice becomes it.
        let state = lastSent.get(name);
        if (!state) {
          state = dataSlice(store.getState());
          lastSent.set(name, state);
        }
        void emitTo(e.payload.label, "panel://sync", {
          store: name,
          seq: seqOf.get(name) ?? 0,
          state,
        } satisfies SyncMessage);
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
  //
  // WARNING (latent, documented): the forwarder is fire-and-forget — it emits
  // the action to the main window and returns `undefined`, DROPPING whatever
  // the real action returns (a value or a Promise). In a detached panel window
  // any action that normally returns something (e.g. `createEntity`'s new guid,
  // or any `async` action you'd `await`) yields `undefined`/an un-awaitable
  // no-op. Callers in bridged panels MUST NOT rely on a bridged action's return
  // value or awaiting it. Round-tripping results would need a request/response
  // correlation id over `panel://action` (out of scope; note only).
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

  const label = getCurrentWebviewWindow().label;
  const requestResync = () => void emit("panel://sync/request", { label });

  /**
   * The last `seq` applied per store, and whether a resync is already in
   * flight for it.
   *
   * A patch is only meaningful on top of `seq - 1`, so a gap — a dropped event,
   * a window that booted mid-stream, a host reinstall — must not be papered
   * over: the mirror asks for a full snapshot and ignores everything until one
   * arrives. `resyncing` keeps a burst of stale patches from emitting a burst
   * of requests.
   */
  const seen = new Map<string, number>();
  const resyncing = new Set<string>();

  const unlisten = await listen<SyncMessage>("panel://sync", (e) => {
    const { store: name, seq, state, patch } = e.payload;
    const store = BRIDGED.get(name);
    if (!store) return;
    if (state) {
      // A full snapshot re-bases the sequence, whatever it was.
      seen.set(name, seq);
      resyncing.delete(name);
      store.setState(state as never);
      return;
    }
    if (!patch) return;
    const last = seen.get(name);
    if (last === undefined || seq !== last + 1) {
      if (!resyncing.has(name)) {
        resyncing.add(name);
        requestResync();
      }
      return;
    }
    seen.set(name, seq);
    store.setState(applySlicePatch(store.getState(), patch) as never);
  });

  void emit("panel://sync/request", { label });
  // Belt-and-braces resync whenever the window regains focus.
  const onFocus = () => requestResync();
  window.addEventListener("focus", onFocus);

  return () => {
    unlisten();
    window.removeEventListener("focus", onFocus);
  };
}
