// @vitest-environment jsdom
//
// The PTY session registry's adopt / replace / close bookkeeping — the pure
// module logic that keeps a running terminal alive across a dock/float
// re-parent. xterm + the backend PTY are injected fakes; the DOM attach/detach
// plumbing is manual-verify (see the fix report).
import { beforeEach, describe, expect, it, vi } from "vitest";

// Keep the real xterm packages out of the test runtime (browser-only + CSS).
vi.mock("@xterm/xterm", () => ({ Terminal: class {} }));
vi.mock("@xterm/addon-fit", () => ({ FitAddon: class {} }));
vi.mock("@xterm/addon-web-links", () => ({ WebLinksAddon: class {} }));
vi.mock("@xterm/xterm/css/xterm.css", () => ({}));

import {
  acquirePtySession,
  attachPtySession,
  closeAllPtySessions,
  closePtySession,
  detachPtySession,
  writePtyLine,
  __peekPtySession,
  __ptySessionCount,
  __resetPtyRegistryForTest,
  __setPtyDepsForTest,
  type PtyDeps,
} from "../ptyRegistry";
import { notifyPanelClosed } from "../../panels/dock/panelLifecycle";

function makeFakes() {
  let n = 0;
  const backend = {
    create: vi.fn(async () => `pty-${n++}`),
    write: vi.fn(async () => {}),
    resize: vi.fn(async () => {}),
    close: vi.fn(async () => {}),
  };
  const createTerminal: PtyDeps["createTerminal"] = () => {
    const host = document.createElement("div");
    const term = {
      cols: 80,
      rows: 24,
      open: vi.fn(),
      onData: vi.fn(() => ({ dispose: vi.fn() })),
      focus: vi.fn(),
      write: vi.fn(),
      writeln: vi.fn(),
      dispose: vi.fn(),
    };
    return { term, fit: vi.fn(), host };
  };
  return {
    backend,
    createTerminal,
    listenOutput: vi.fn(async () => vi.fn()),
    listenExit: vi.fn(async () => vi.fn()),
  } satisfies PtyDeps;
}

let fakes: PtyDeps;

beforeEach(() => {
  __resetPtyRegistryForTest();
  fakes = makeFakes();
  __setPtyDepsForTest(fakes);
});

describe("ptyRegistry", () => {
  it("creates exactly one backend session on first acquire", async () => {
    const s = acquirePtySession("terminal", "/proj");
    await s.ready;
    expect(fakes.backend.create).toHaveBeenCalledTimes(1);
    expect(s.id).toBe("pty-0");
    expect(__ptySessionCount()).toBe(1);
  });

  it("adopts the SAME live session on re-acquire with matching cwd", async () => {
    const first = acquirePtySession("terminal", "/proj");
    await first.ready;
    const second = acquirePtySession("terminal", "/proj");
    // Same object → the terminal + scrollback + PTY are preserved across the
    // re-parent; no second backend session is spawned.
    expect(second).toBe(first);
    expect(fakes.backend.create).toHaveBeenCalledTimes(1);
  });

  it("replaces the session when cwd changes (project switch)", async () => {
    const first = acquirePtySession("terminal", "/a");
    await first.ready;
    const second = acquirePtySession("terminal", "/b");
    await second.ready;
    expect(second).not.toBe(first);
    // Old backend session torn down, new one created.
    expect(fakes.backend.close).toHaveBeenCalledWith("pty-0");
    expect(fakes.backend.create).toHaveBeenCalledTimes(2);
    expect(second.cwd).toBe("/b");
    expect(__ptySessionCount()).toBe(1);
  });

  it("detach keeps the session alive (a move, not a close)", async () => {
    const s = acquirePtySession("terminal", "/proj");
    await s.ready;
    const mount = document.createElement("div");
    attachPtySession(s, mount);
    expect(mount.contains(s.host)).toBe(true);

    detachPtySession(s);
    expect(mount.contains(s.host)).toBe(false);
    expect(s.disposed).toBe(false);
    // Still registered → a remount re-adopts it.
    expect(__peekPtySession("terminal")).toBe(s);
    expect(fakes.backend.close).not.toHaveBeenCalled();
  });

  it("re-attaches after detach without re-creating the PTY", async () => {
    const s = acquirePtySession("terminal", "/proj");
    await s.ready;
    const m1 = document.createElement("div");
    attachPtySession(s, m1);
    detachPtySession(s);
    const m2 = document.createElement("div");
    const adopted = acquirePtySession("terminal", "/proj");
    attachPtySession(adopted, m2);
    expect(adopted).toBe(s);
    expect(m2.contains(s.host)).toBe(true);
    expect(fakes.backend.create).toHaveBeenCalledTimes(1);
  });

  it("closePtySession tears down backend + removes from registry (idempotent)", async () => {
    const s = acquirePtySession("terminal", "/proj");
    await s.ready;
    closePtySession("terminal");
    expect(fakes.backend.close).toHaveBeenCalledWith("pty-0");
    expect(s.disposed).toBe(true);
    expect(__peekPtySession("terminal")).toBeUndefined();
    // Second close is a no-op.
    closePtySession("terminal");
    expect(fakes.backend.close).toHaveBeenCalledTimes(1);
  });

  it("an explicit panel close (onPanelClosed) closes its session", async () => {
    const s = acquirePtySession("terminal", "/proj");
    await s.ready;
    notifyPanelClosed("terminal");
    expect(s.disposed).toBe(true);
    expect(__peekPtySession("terminal")).toBeUndefined();

    // An unrelated panel id doesn't touch it.
    const other = acquirePtySession("terminal", "/proj");
    await other.ready;
    notifyPanelClosed("someOtherPanel");
    expect(other.disposed).toBe(false);
  });

  it("closeAll tears down every session", async () => {
    await acquirePtySession("term:a", "/a").ready;
    await acquirePtySession("term:b", "/b").ready;
    expect(__ptySessionCount()).toBe(2);
    closeAllPtySessions();
    expect(__ptySessionCount()).toBe(0);
    expect(fakes.backend.close).toHaveBeenCalledTimes(2);
  });

  it("writePtyLine sends the line with a trailing CR to the backend", async () => {
    const s = acquirePtySession("terminal", "/proj");
    await s.ready;
    writePtyLine(s, "cargo build");
    expect(fakes.backend.write).toHaveBeenCalledWith("pty-0", "cargo build\r");
  });
});

/**
 * **`disposed` after EVERY await** (F-lens L7.M9).
 *
 * Wiring a session is three awaits — `create`, then the output and exit
 * listeners — and only the first was guarded. `closePtySession` runs
 * synchronously: it sets `disposed`, DRAINS `session.unlisten`, and disposes the
 * terminal. A listener whose `listen` resolved after that drain was pushed into
 * an array nobody would ever read again — permanently subscribed, writing PTY
 * bytes into a disposed xterm on every `pty://output` for the rest of the
 * process, per closed terminal panel.
 */
describe("the close race while a session is still wiring up (L7.M9)", () => {
  /** Deps whose two `listen` calls can be resolved on demand. */
  function gatedListeners() {
    const outUnlisten = vi.fn();
    const exitUnlisten = vi.fn();
    let releaseOut!: () => void;
    let releaseExit!: () => void;
    const outGate = new Promise<void>((r) => (releaseOut = r));
    const exitGate = new Promise<void>((r) => (releaseExit = r));
    return {
      outUnlisten,
      exitUnlisten,
      releaseOut: () => releaseOut(),
      releaseExit: () => releaseExit(),
      deps: {
        listenOutput: vi.fn(async () => {
          await outGate;
          return outUnlisten;
        }),
        listenExit: vi.fn(async () => {
          await exitGate;
          return exitUnlisten;
        }),
      },
    };
  }

  it("releases the OUTPUT listener that resolves after the close", async () => {
    const g = gatedListeners();
    __setPtyDepsForTest(g.deps);

    const s = acquirePtySession("terminal", "/proj");
    // Let `create` land so the session has an id, then close mid-wiring.
    await Promise.resolve();
    await Promise.resolve();
    closePtySession("terminal");
    expect(s.disposed).toBe(true);

    g.releaseOut();
    g.releaseExit();
    await s.ready;

    // Both handles released, and neither was parked in the drained array.
    expect(g.outUnlisten).toHaveBeenCalledTimes(1);
    expect(s.unlisten).toHaveLength(0);
  });

  it("releases BOTH listeners when the close lands between them", async () => {
    const g = gatedListeners();
    __setPtyDepsForTest(g.deps);

    const s = acquirePtySession("terminal", "/proj");
    await Promise.resolve();
    await Promise.resolve();
    // The output listener is up; the exit listener is not.
    g.releaseOut();
    await Promise.resolve();
    await Promise.resolve();

    closePtySession("terminal");
    g.releaseExit();
    await s.ready;

    expect(g.outUnlisten).toHaveBeenCalledTimes(1);
    expect(g.exitUnlisten).toHaveBeenCalledTimes(1);
    // Released exactly once each — the drain must not double-call them either.
    expect(s.unlisten).toHaveLength(0);
  });

  it("keeps both handles when nothing closed", async () => {
    // The other half: the guard must not release a listener the session still
    // wants. Without this, "always release" would pass the two cases above.
    const g = gatedListeners();
    __setPtyDepsForTest(g.deps);

    const s = acquirePtySession("terminal", "/proj");
    g.releaseOut();
    g.releaseExit();
    await s.ready;

    expect(g.outUnlisten).not.toHaveBeenCalled();
    expect(g.exitUnlisten).not.toHaveBeenCalled();
    expect(s.unlisten).toHaveLength(2);
  });

  it("does not write a failure banner into a terminal that was disposed", async () => {
    __setPtyDepsForTest({
      backend: {
        create: vi.fn(() => Promise.reject(new Error("no pty available"))),
        write: vi.fn(async () => {}),
        resize: vi.fn(async () => {}),
        close: vi.fn(async () => {}),
      },
    });
    const s = acquirePtySession("terminal", "/proj");
    closePtySession("terminal");
    await s.ready;
    expect(s.term.writeln).not.toHaveBeenCalled();
  });
});
