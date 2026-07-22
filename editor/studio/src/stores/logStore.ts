/**
 * Output Log state (P1.4.4): a bounded ring of structured tracing lines
 * plus the panel's severity/search filters. The MAIN window subscribes to
 * `log://line` (installed once from App); detached Output Log windows get
 * the lines through the store bridge instead of double-subscribing.
 */
import { create } from "zustand";
import type { LogLine } from "../bindings/LogLine";
import type { LogLevel } from "../bindings/LogLevel";
import { listenTo } from "../lib/events";
import { registerBridgedStore } from "../panels/window/storeBridge";

export const LOG_CAP = 5000;
export const LOG_LEVELS: LogLevel[] = ["trace", "debug", "info", "warn", "error"];

interface LogState {
  lines: LogLine[];
  /** Severities currently shown. */
  enabled: Record<LogLevel, boolean>;
  search: string;
  /** Pause appending (the stream keeps flowing; new lines drop). */
  paused: boolean;
  append: (line: LogLine) => void;
  clear: () => void;
  setSearch: (s: string) => void;
  toggleLevel: (level: LogLevel) => void;
  setPaused: (paused: boolean) => void;
  /** Append a whole batch in ONE store update (the batched stream path). */
  appendMany: (lines: LogLine[]) => void;
}

export const useLogStore = create<LogState>((set) => ({
  lines: [],
  enabled: { trace: false, debug: true, info: true, warn: true, error: true },
  search: "",
  paused: false,
  append: (line) =>
    set((s) => {
      if (s.paused) return s;
      const lines = s.lines.length >= LOG_CAP ? [...s.lines.slice(1), line] : [...s.lines, line];
      return { lines };
    }),
  clear: () => set({ lines: [] }),
  setSearch: (search) => set({ search }),
  toggleLevel: (level) =>
    set((s) => ({ enabled: { ...s.enabled, [level]: !s.enabled[level] } })),
  setPaused: (paused) => set({ paused }),
  appendMany: (incoming) =>
    set((s) => {
      if (s.paused || incoming.length === 0) return s;
      const combined = s.lines.length ? [...s.lines, ...incoming] : incoming.slice();
      // Cap-trim once, after the merge — O(batch) per flush instead of the old
      // O(N) whole-array copy PER line.
      const lines =
        combined.length > LOG_CAP ? combined.slice(combined.length - LOG_CAP) : combined;
      return { lines };
    }),
}));

/**
 * Coalesces a high-rate `log://line` stream into periodic batch flushes, so
 * the store (and the cross-window bridge that mirrors it) update at most once
 * per `intervalMs` instead of once per line. Pure + timer-injectable → unit-
 * testable with fake timers.
 */
export function createLogBatcher(flush: (lines: LogLine[]) => void, intervalMs = 60) {
  let buffer: LogLine[] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  const fire = () => {
    timer = null;
    if (buffer.length === 0) return;
    const batch = buffer;
    buffer = [];
    flush(batch);
  };
  return {
    push(line: LogLine) {
      buffer.push(line);
      if (timer === null) timer = setTimeout(fire, intervalMs);
    },
    /** Cancel the pending flush and drop any buffered lines. */
    dispose() {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      buffer = [];
    },
  };
}

registerBridgedStore("log", useLogStore);

/** Main window only: subscribe the store to the `log://line` stream. Lines are
 *  batched (~60 ms) into one store update to keep append O(batch), not O(N)
 *  per line. */
export function startLogListener(): () => void {
  let disposed = false;
  let unlisten: (() => void) | undefined;
  const batcher = createLogBatcher((lines) => useLogStore.getState().appendMany(lines));
  void listenTo("log://line", (line) => batcher.push(line)).then((u) => {
    if (disposed) u();
    else unlisten = u;
  });
  return () => {
    disposed = true;
    batcher.dispose();
    unlisten?.();
  };
}
