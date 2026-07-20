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
}));

registerBridgedStore("log", useLogStore);

/** Main window only: subscribe the store to the `log://line` stream. */
export function startLogListener(): () => void {
  let disposed = false;
  let unlisten: (() => void) | undefined;
  void listenTo("log://line", (line) => {
    useLogStore.getState().append(line);
  }).then((u) => {
    if (disposed) u();
    else unlisten = u;
  });
  return () => {
    disposed = true;
    unlisten?.();
  };
}
