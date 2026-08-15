/**
 * PTY session registry (P5.3 follow-up): decouples a backend PTY session's
 * lifetime from the React component that displays it.
 *
 * The bug this fixes: the dock keeps inactive tabs mounted, but *moving* a
 * panel (drag to a float card / re-dock / detached window) re-parents its
 * subtree in React = unmount + remount. If `usePty` created and closed the
 * PTY on mount/unmount, that move would kill a running `cargo build`.
 *
 * The fix: sessions live here, keyed by panel identity, NOT in the component.
 *  - The xterm terminal is opened ONCE into a detached "host" `<div>` this
 *    module owns; `attach` physically moves that div into the component's
 *    mount node and `detach` moves it back out. The terminal instance — and
 *    thus its scrollback buffer — is never disposed on a move, so re-parenting
 *    is lossless (no `term.open()` re-call, which xterm doesn't support).
 *  - A live backend PTY + its `pty://output|exit` listeners are created on
 *    first `acquire` and survive every detach; keystroke → `pty.write` and
 *    the ResizeObserver live in the component (per mount node) but target the
 *    session's stable backend id.
 *  - The PTY is closed only on an EXPLICIT panel close (wired to
 *    `onPanelClosed`) or app teardown (`beforeunload`) — never on a move.
 *
 * Testability: xterm + the backend IPC are injected (`__setPtyDepsForTest`),
 * so the adopt/replace/close *bookkeeping* is unit-tested without a DOM or a
 * real PTY. The attach/detach DOM plumbing is manual-verify (see report).
 *
 * KNOWN LIMITATION: a panel torn off into its own detached OS *window* runs in
 * a separate webview with a separate JS context, so this in-process registry
 * cannot hand the session across — that path still re-creates the PTY. Sharing
 * it would need a backend-side session handoff keyed by panel id (a Ring-2 +
 * ipc.ts change, out of scope here). Same-webview moves (tabs, float cards,
 * re-docks) are fully covered.
 */
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

/** The minimal xterm surface the registry uses (widened for test fakes). */
export interface TerminalLike {
  open(node: HTMLElement): void;
  onData(cb: (d: string) => void): { dispose(): void };
  focus(): void;
  write(data: Uint8Array | string): void;
  writeln(data: string): void;
  dispose(): void;
  readonly cols: number;
  readonly rows: number;
}

/** The backend PTY IPC surface (mirrors `ipc.pty`; injectable for tests). */
export interface PtyBackend {
  create(cwd: string | null, cols: number, rows: number): Promise<string>;
  write(id: string, data: string): Promise<void>;
  resize(id: string, cols: number, rows: number): Promise<void>;
  close(id: string): Promise<void>;
}

/** Everything the registry needs from the outside world — swapped in tests. */
export interface PtyDeps {
  backend: PtyBackend;
  /** Build a terminal + its fit-refit hook + the detached host node. */
  createTerminal(): { term: TerminalLike; fit: () => void; host: HTMLElement };
  listenOutput(id: string, term: TerminalLike): Promise<UnlistenFn>;
  listenExit(id: string, term: TerminalLike): Promise<UnlistenFn>;
}

const defaultDeps: PtyDeps = {
  backend: pty,
  createTerminal() {
    const host = document.createElement("div");
    host.style.width = "100%";
    host.style.height = "100%";
    const term = new Terminal({
      fontSize: 13,
      fontFamily: 'ui-monospace, "Cascadia Code", "JetBrains Mono", Consolas, monospace',
      theme: XTERM_THEME,
      cursorBlink: true,
      scrollback: 5000,
    });
    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.loadAddon(new WebLinksAddon());
    term.open(host);
    return {
      term: term as unknown as TerminalLike,
      fit: () => {
        try {
          fitAddon.fit();
        } catch {
          /* container not laid out yet */
        }
      },
      host,
    };
  },
  listenOutput: (id, term) => onPtyOutput(id, (p) => term.write(base64ToBytes(p.data))),
  listenExit: (id, term) =>
    onPtyExit(id, () => term.writeln("\r\n\x1b[90m[process exited]\x1b[0m")),
};

let deps: PtyDeps = defaultDeps;

export interface PtySession {
  key: string;
  cwd: string | null;
  term: TerminalLike;
  /** Re-fit the terminal to its current host size. */
  fit: () => void;
  /** The detached DOM node the terminal was opened into (moved on attach). */
  host: HTMLElement;
  /** Backend PTY id, null until `create` resolves. */
  id: string | null;
  /** Resolves once the backend session + listeners are wired (test hook). */
  ready: Promise<void>;
  /** True while the host div is mounted into a live component node. */
  attached: boolean;
  disposed: boolean;
  unlisten: UnlistenFn[];
  dataSub: { dispose(): void } | null;
}

const sessions = new Map<string, PtySession>();

/** Build a fresh, fully-wired session (terminal + backend + listeners). */
function createSession(key: string, cwd: string | null): PtySession {
  const { term, fit, host } = deps.createTerminal();
  const session: PtySession = {
    key,
    cwd,
    term,
    fit,
    host,
    id: null,
    ready: Promise.resolve(),
    attached: false,
    disposed: false,
    unlisten: [],
    dataSub: null,
  };
  // Keystrokes → backend. Bound to the stable terminal (survives detach).
  session.dataSub = term.onData((d) => {
    if (session.id) void deps.backend.write(session.id, d);
  });
  // `disposed` is re-checked after EVERY await, not only the first (F-lens
  // L7.M9). The original only guarded the `create`, leaving a two-await gap:
  // `closePtySession` runs synchronously, drains `session.unlisten` and calls
  // `term.dispose()`, and the two `listen` handles that resolved afterwards were
  // pushed into the now-drained array — permanently subscribed, writing PTY
  // bytes into a disposed xterm on every `pty://output` for the life of the
  // process. Each handle is released the moment it is seen to be late, and the
  // failure path stops writing into a terminal that no longer exists.
  session.ready = (async () => {
    try {
      const id = await deps.backend.create(cwd, term.cols || 80, term.rows || 24);
      if (session.disposed) {
        void deps.backend.close(id);
        return;
      }
      session.id = id;

      // Collected locally and published only once BOTH listens have resolved.
      // Pushing them one at a time would put the first handle into an array
      // `closePtySession` may drain before the second resolves — and a drained
      // array cannot release what lands in it afterwards. From here on
      // `session.id` is set, so the teardown closes the backend PTY itself;
      // only the listeners need releasing here.
      const handles: UnlistenFn[] = [];
      const releaseLate = () => {
        for (const u of handles) {
          try {
            u();
          } catch {
            /* ignore */
          }
        }
      };

      handles.push(await deps.listenOutput(id, term));
      if (session.disposed) return releaseLate();
      handles.push(await deps.listenExit(id, term));
      if (session.disposed) return releaseLate();
      session.unlisten.push(...handles);
    } catch (e) {
      if (session.disposed) return;
      term.writeln(`\r\n\x1b[31mfailed to start terminal: ${String(e)}\x1b[0m`);
    }
  })();
  sessions.set(key, session);
  return session;
}

/**
 * Return the live session for `key`, adopting an existing one when its `cwd`
 * still matches (the common re-parent case → lossless), or replacing a stale
 * one (project root changed) with a fresh session.
 */
export function acquirePtySession(key: string, cwd: string | null): PtySession {
  const existing = sessions.get(key);
  if (existing && !existing.disposed) {
    if (existing.cwd === cwd) return existing;
    // cwd changed out from under us (e.g. project switch) → replace.
    closePtySession(key);
  }
  return createSession(key, cwd);
}

/** Move a session's terminal into `mountEl` and fit it to that node. */
export function attachPtySession(session: PtySession, mountEl: HTMLElement): void {
  if (session.disposed) return;
  if (session.host.parentElement !== mountEl) mountEl.appendChild(session.host);
  session.attached = true;
  session.fit();
}

/**
 * Detach a session from its mount node WITHOUT closing it: the terminal + its
 * scrollback + the backend PTY all stay alive so a remount re-adopts them.
 */
export function detachPtySession(session: PtySession): void {
  if (session.disposed) return;
  session.host.remove();
  session.attached = false;
}

/** Push a resize through to the backend (component ResizeObserver calls this). */
export function resizePtySession(session: PtySession, cols: number, rows: number): void {
  if (session.id) void deps.backend.resize(session.id, cols, rows);
}

/** Send a command line (appends CR so the shell runs it). */
export function writePtyLine(session: PtySession, cmd: string): void {
  if (session.id) void deps.backend.write(session.id, `${cmd}\r`);
}

/** Fully tear down a session: an explicit panel close or app teardown only. */
export function closePtySession(key: string): void {
  const session = sessions.get(key);
  if (!session) return;
  session.disposed = true;
  sessions.delete(key);
  for (const u of session.unlisten) {
    try {
      u();
    } catch {
      /* ignore */
    }
  }
  session.unlisten = [];
  session.dataSub?.dispose();
  session.dataSub = null;
  session.host.remove();
  try {
    session.term.dispose();
  } catch {
    /* ignore */
  }
  if (session.id) void deps.backend.close(session.id);
}

/** Close every live session (app teardown). */
export function closeAllPtySessions(): void {
  for (const key of [...sessions.keys()]) closePtySession(key);
}

// ── Wiring: an explicit panel close frees its terminal ──────────────────────
// The session key is the panel id; the dock store fires `notifyPanelClosed`
// only on a real close/hide, never a location move.
import { onPanelClosed } from "../panels/dock/panelLifecycle";

onPanelClosed((panelId) => closePtySession(panelId));

if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", closeAllPtySessions);
}

// ── Test seams ──────────────────────────────────────────────────────────────

/** Test-only: swap the xterm/backend dependencies. */
export function __setPtyDepsForTest(next: Partial<PtyDeps>): void {
  deps = { ...deps, ...next };
}

/** Test-only: wipe the registry and restore real dependencies. */
export function __resetPtyRegistryForTest(): void {
  for (const session of sessions.values()) {
    session.disposed = true;
    session.dataSub?.dispose();
  }
  sessions.clear();
  deps = defaultDeps;
}

/** Test-only: peek at a session (or undefined). */
export function __peekPtySession(key: string): PtySession | undefined {
  return sessions.get(key);
}

/** Test-only: how many sessions are live. */
export function __ptySessionCount(): number {
  return sessions.size;
}
