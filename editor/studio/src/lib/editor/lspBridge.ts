/**
 * LSP ⇆ editor bridge (P5.2). Owns the `lsp://` event subscriptions and drives
 * the editor: when a Rust file becomes active it ensures rust-analyzer is
 * running (rooted at the open project), sends didOpen, and installs the
 * per-file LSP extension into the editor's `extraCompartment` seam; incoming
 * diagnostics update the Problems store and, for the active file, paint
 * squiggles. Kept out of both stores to avoid an import cycle.
 */
import {
  onLspDiagnostics,
  onLspStarted,
  onLspStopped,
  refCountedInit,
  type UnlistenFn,
} from "../events";
import { lsp } from "../ipc";
import { useProjectStore } from "../../stores/projectStore";
import {
  activeTabPath,
  getActiveView,
  useEditorStore,
} from "../../stores/editorStore";
import { useLspStore } from "../../stores/lspStore";
import { pathToUri } from "./fileUri";
import { setExtraExtensions } from "./setup";
import { lspExtensionFor, pushDiagnostics } from "./lspExtension";

const LANG = "rust";

/** Re-exported: the encoder moved to `fileUri.ts` in SCRIPT2b (see there). */
export { pathToUri };

function isRust(path: string | null): path is string {
  return !!path && path.toLowerCase().endsWith(".rs");
}

let started = false;
const openedUris = new Set<string>();
let installedForPath: string | null = null;

/** Ensure the server is up and the active Rust file is wired to the editor. */
async function activateForActive(): Promise<void> {
  const path = activeTabPath();
  const view = getActiveView();
  if (!isRust(path) || !view) return;

  const root = useProjectStore.getState().current?.root;
  if (!root) return; // no project → nothing to root the server at

  const uri = pathToUri(path);
  const lspApi = useLspStore.getState();

  // Start rust-analyzer once, rooted at the project.
  if (!started) {
    started = true;
    lspApi.setStatus(LANG, "starting");
    try {
      await lsp.start(LANG, root);
      lspApi.setStatus(LANG, "running");
    } catch (e) {
      started = false;
      lspApi.setStatus(LANG, "error");
      lspApi.setError(String(e));
      return;
    }
  }

  // didOpen once per file.
  if (!openedUris.has(uri)) {
    openedUris.add(uri);
    const state = view.state;
    void lsp.didOpen(LANG, path, state.doc.toString());
  }

  // Install the per-file LSP extension into the live view (idempotent per path).
  if (installedForPath !== path) {
    installedForPath = path;
    setExtraExtensions(view, lspExtensionFor({ path, uri }));
  }

  // Paint any diagnostics we already have for this file.
  const known = useLspStore.getState().diagnostics[uri];
  if (known) pushDiagnostics(view, known);
}

/**
 * Start the bridge: subscribe to events + active-tab changes.
 *
 * **Refcounted, not flag-guarded** (F-lens L7.M2 — this was the LSP-is-dead
 * bug). The previous shape was `if (inited) return () => {}; inited = true;`,
 * which looks StrictMode-safe because the guard is set before the first
 * `await`. It is not, because of what the *second* caller is handed. React runs
 * mount → cleanup → mount synchronously in one commit:
 *
 *  1. mount #1 sets `inited` and starts subscribing;
 *  2. cleanup #1 marks itself disposed and arms `.then((fn) => fn())`;
 *  3. mount #2 sees `inited` still true (nothing has resolved) and receives a
 *     **no-op disposer** — it holds nothing;
 *  4. the subscriptions resolve, cleanup #1's armed disposer fires and tears
 *     down every `lsp://` listener plus the editor-store subscription, and
 *     resets `inited = false`.
 *
 * Net result in `tauri dev`: no listeners, no re-init, **no LSP at all** — no
 * diagnostics, no squiggles, and `activateForActive` never runs again, so
 * rust-analyzer is never even started. Counting holders instead of flipping a
 * flag makes step 3 a real second holder, so step 4 decrements to 1 rather than
 * to 0; and when the last holder does release, `refCountedInit` clears its
 * memo so a genuine remount re-subscribes.
 *
 * `started`/`openedUris`/`installedForPath` above are deliberately NOT reset by
 * the teardown: they describe the *backend server's* state, which outlives this
 * bridge's listeners.
 */
export const initLsp = refCountedInit(async (sink) => {
  // **Three subscribes at once, and every survivor accounted for** (round-2
  // LOW + R2-7). This awaited them one at a time, which is the ptyRegistry
  // lesson Wave F wrote down and did not apply to its sibling in the same wave:
  // a rejection on the second left the first subscribed with nothing holding a
  // reference to release it.
  //
  // `allSettled` rather than `all`, deliberately: `all` rejects on the first
  // failure while the others are still resolving, and those handles then arrive
  // with no owner at all. Settling every one first means each is either in the
  // sink or never existed.
  const settled = await Promise.allSettled([
    onLspStarted((p) => useLspStore.getState().setStatus(p.language, "running")),
    onLspStopped((p) => useLspStore.getState().setStatus(p.language, "stopped")),
    onLspDiagnostics(({ uri, diagnostics }) => {
      useLspStore.getState().setDiagnostics(uri, diagnostics);
      const path = activeTabPath();
      const view = getActiveView();
      if (view && path && pathToUri(path) === uri) pushDiagnostics(view, diagnostics);
    }),
  ]);
  const disposers: UnlistenFn[] = [];
  for (const r of settled) {
    if (r.status === "fulfilled") {
      disposers.push(r.value);
      sink(r.value);
    }
  }
  const failed = settled.find((r) => r.status === "rejected");
  if (failed && failed.status === "rejected") {
    // `refCountedInit` drains the sink on the way out, so the handles that DID
    // resolve are released rather than left subscribed for the process.
    throw failed.reason instanceof Error ? failed.reason : new Error(String(failed.reason));
  }

  // React to the active tab changing.
  let lastActive: string | null = null;
  const unsubscribeStore = useEditorStore.subscribe((s) => {
    if (s.activeId !== lastActive) {
      lastActive = s.activeId;
      void activateForActive();
    }
  });
  // Handle a file already open at init.
  void activateForActive();

  return () => {
    disposers.forEach((d) => d());
    disposers.length = 0;
    unsubscribeStore();
  };
});
