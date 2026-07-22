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
