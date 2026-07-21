/**
 * `usePty` (P5.3): binds one xterm.js terminal to a backend PTY session.
 *
 * On mount it builds the terminal (+ fit / web-links addons), opens it into the
 * given container, creates a backend PTY sized to the fitted grid, and wires
 * both directions — keystrokes → `pty.write`, `pty://output/{id}` → the
 * terminal. A ResizeObserver keeps the PTY grid in sync. StrictMode-safe: a
 * `disposed` guard closes a session whose async create resolves after unmount.
 */
import { useCallback, useEffect, useRef, type RefObject } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";
import "@xterm/xterm/css/xterm.css";

import { pty } from "./ipc";
import { onPtyExit, onPtyOutput, type UnlistenFn } from "./events";

/** A dark theme roughly matching the studio's `--ink-*` palette. */
const XTERM_THEME = {
  background: "#0e0f13",
  foreground: "#d6d9e0",
  cursor: "#6ea8fe",
  cursorAccent: "#0e0f13",
  selectionBackground: "#2a3550",
  black: "#20232b",
  red: "#e06c75",
  green: "#98c379",
  yellow: "#e5c07b",
  blue: "#61afef",
  magenta: "#c678dd",
  cyan: "#56b6c2",
  white: "#d6d9e0",
  brightBlack: "#5c6370",
  brightWhite: "#ffffff",
} as const;

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export interface PtyHandle {
  /** Send a command line (appends CR so the shell runs it). */
  runCommand: (cmd: string) => void;
  /** Focus the terminal. */
  focus: () => void;
}

export function usePty(container: RefObject<HTMLDivElement | null>, cwd: string | null): PtyHandle {
  const termRef = useRef<Terminal | null>(null);
  const idRef = useRef<string | null>(null);

  useEffect(() => {
    const el = container.current;
    if (!el) return;

    const term = new Terminal({
      fontSize: 13,
      fontFamily: 'ui-monospace, "Cascadia Code", "JetBrains Mono", Consolas, monospace',
      theme: XTERM_THEME,
      cursorBlink: true,
      scrollback: 5000,
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.loadAddon(new WebLinksAddon());
    term.open(el);
    try {
      fit.fit();
    } catch {
      /* container not laid out yet */
    }
    termRef.current = term;

    let disposed = false;
    let sessionId: string | null = null;
    const unlisten: UnlistenFn[] = [];

    void (async () => {
      try {
        const id = await pty.create(cwd, term.cols || 80, term.rows || 24);
        if (disposed) {
          void pty.close(id);
          return;
        }
        sessionId = id;
        idRef.current = id;
        unlisten.push(await onPtyOutput(id, (p) => term.write(base64ToBytes(p.data))));
        unlisten.push(
          await onPtyExit(id, () => term.writeln("\r\n\x1b[90m[process exited]\x1b[0m")),
        );
      } catch (e) {
        term.writeln(`\r\n\x1b[31mfailed to start terminal: ${String(e)}\x1b[0m`);
      }
    })();

    const dataSub = term.onData((d) => {
      if (sessionId) void pty.write(sessionId, d);
    });
    const ro = new ResizeObserver(() => {
      try {
        fit.fit();
      } catch {
        /* ignore */
      }
      if (sessionId) void pty.resize(sessionId, term.cols, term.rows);
    });
    ro.observe(el);

    return () => {
      disposed = true;
      ro.disconnect();
      dataSub.dispose();
      unlisten.forEach((u) => u());
      if (sessionId) void pty.close(sessionId);
      term.dispose();
      termRef.current = null;
      idRef.current = null;
    };
  }, [container, cwd]);

  const runCommand = useCallback((cmd: string) => {
    const id = idRef.current;
    if (id) void pty.write(id, `${cmd}\r`);
  }, []);
  const focus = useCallback(() => termRef.current?.focus(), []);

  return { runCommand, focus };
}
