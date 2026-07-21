/**
 * The `infinity:open-file` app event (P5.4 → P5.1 bridge).
 *
 * The File Explorer and Search panels request opening a source file by absolute
 * path; the Code Editor panel (P5.1) listens and opens a tab. Kept as a window
 * CustomEvent so the two panels stay decoupled (either may be closed).
 */

const CHANNEL = "infinity:open-file";

/** Ask the editor to open `absPath`. */
export function requestOpenFile(absPath: string): void {
  window.dispatchEvent(new CustomEvent(CHANNEL, { detail: { path: absPath } }));
}

/** Subscribe to open-file requests; returns an unsubscribe fn. */
export function onOpenFile(handler: (absPath: string) => void): () => void {
  const listener = (e: Event) => {
    const path = (e as CustomEvent<{ path: string }>).detail?.path;
    if (path) handler(path);
  };
  window.addEventListener(CHANNEL, listener);
  return () => window.removeEventListener(CHANNEL, listener);
}
