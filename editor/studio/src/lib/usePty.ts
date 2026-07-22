/**
 * `usePty` (P5.3): binds one xterm.js terminal to a backend PTY session.
 *
 * The session itself lives in `lib/ptyRegistry`, keyed by panel identity, so
 * its lifetime is DECOUPLED from this component: dragging the terminal panel to
 * a float card / re-docking / switching tabs re-parents the React subtree
 * (unmount + remount) WITHOUT killing a running `cargo build`. On mount we
 * `acquire` (adopt an existing live session, re-attaching its terminal + full
 * scrollback, or spawn a fresh one) and on unmount we `detach` — the PTY is
 * closed only on an explicit panel close (`onPanelClosed`) or app teardown.
 *
 * This hook owns only the per-mount-node concerns: moving the session's
 * terminal into the container and a ResizeObserver that refits + resizes the
 * backend grid.
 */
import { useCallback, useEffect, useRef, type RefObject } from "react";

import {
  acquirePtySession,
  attachPtySession,
  detachPtySession,
  resizePtySession,
  writePtyLine,
  type PtySession,
} from "./ptyRegistry";

export interface PtyHandle {
  /** Send a command line (appends CR so the shell runs it). */
  runCommand: (cmd: string) => void;
  /** Focus the terminal. */
  focus: () => void;
}

/**
 * @param sessionKey stable identity for the session (the panel id) — the
 *   registry keys the live PTY on it so remounts adopt the same session.
 */
export function usePty(
  container: RefObject<HTMLDivElement | null>,
  cwd: string | null,
  sessionKey: string,
): PtyHandle {
  const sessionRef = useRef<PtySession | null>(null);

  useEffect(() => {
    const el = container.current;
    if (!el) return;

    const session = acquirePtySession(sessionKey, cwd);
    sessionRef.current = session;
    attachPtySession(session, el);

    const ro = new ResizeObserver(() => {
      session.fit();
      resizePtySession(session, session.term.cols, session.term.rows);
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      detachPtySession(session);
      sessionRef.current = null;
    };
  }, [container, cwd, sessionKey]);

  const runCommand = useCallback((cmd: string) => {
    const session = sessionRef.current;
    if (session) writePtyLine(session, cmd);
  }, []);
  const focus = useCallback(() => sessionRef.current?.term.focus(), []);

  return { runCommand, focus };
}
