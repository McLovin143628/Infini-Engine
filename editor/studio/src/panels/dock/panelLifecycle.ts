/**
 * Panel lifecycle notifier (P5.3 follow-up).
 *
 * The dock store moves panels between locations constantly (dock ⇄ float ⇄
 * detached window), and every move unmounts+remounts the panel's React
 * subtree. Most panels don't care, but some own a resource whose lifetime must
 * outlive a mere re-parent yet still be torn down on a *real* close — the
 * embedded terminal's PTY being the motivating case (see `lib/ptyRegistry`).
 *
 * This is a tiny pub/sub the dock store fires ONLY when a panel is genuinely
 * closed/hidden (`closePanel`/`hidePanel`), never on a location move
 * (`applyDrop`). Resource owners subscribe and free on the matching id.
 */

type Listener = (panelId: string) => void;

const listeners = new Set<Listener>();

/** Subscribe to explicit panel-close events. Returns an unsubscribe fn. */
export function onPanelClosed(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Fire the close notification for `panelId` (dock store only). */
export function notifyPanelClosed(panelId: string): void {
  for (const l of [...listeners]) {
    try {
      l(panelId);
    } catch (e) {
      console.error("[panelLifecycle] close listener threw:", e);
    }
  }
}

/** Test-only: drop all subscribers between cases. */
export function __resetPanelLifecycleForTest(): void {
  listeners.clear();
}
