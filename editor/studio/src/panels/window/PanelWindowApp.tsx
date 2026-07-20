import { useEffect, useMemo, useState } from "react";
import { emit, emitTo } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { Minus, Square, X } from "lucide-react";
// Panel types must be registered before the id below resolves.
import "../registerPanels";
import { panelDefFor, panelTitle } from "../panelRegistry";
import { PanelChromeContext } from "../dock/panelChromeContext";
import { startStoreBridgeMirror } from "./storeBridge";
import { beginDetachedWindowDrag } from "./detachedWindowDrag";

/**
 * Root component for a detached panel window (label `panel:<panelId>`,
 * loaded at `#/panel/<panelId>`). A LEAN bootstrap — deliberately skips
 * everything one-window-assuming in the main App: shell chrome, dock
 * workspace, viewport. State arrives through the store-bridge mirror;
 * mutations forward back to main; Tauri invokes work directly.
 */

export function panelIdFromHash(): string | null {
  const m = /#\/panel\/(.+)$/.exec(window.location.hash);
  return m?.[1] ? decodeURIComponent(m[1]) : null;
}

export default function PanelWindowApp() {
  const panelId = useMemo(() => panelIdFromHash(), []);
  const [ready, setReady] = useState(false);

  // Store-bridge mirror + ready handshake + show the (spawned-hidden) window.
  useEffect(() => {
    let mounted = true;
    let stop: (() => void) | undefined;
    void (async () => {
      stop = await startStoreBridgeMirror();
      // Async-setup vs sync-cleanup race (StrictMode double-mount): if this
      // effect was already cleaned up, dispose the just-resolved mirror
      // instead of leaking its listeners.
      if (!mounted) {
        stop();
        stop = undefined;
        return;
      }
      setReady(true);
      const win = getCurrentWindow();
      await win.show().catch(() => {});
      await win.setFocus().catch(() => {});
      // Activation belt: a window spawned by a TEAR-OUT comes up the
      // instant a captured pointer drag releases outside the main window,
      // and WebView2 can bring it up without input focus — leaving the
      // custom (JS-driven) titlebar inert. A second focus a tick later
      // wins the activation race. Harmless for the fresh-open path.
      setTimeout(() => {
        if (mounted) void win.setFocus().catch(() => {});
      }, 80);
      void emit("panel://ready", { panelId });
    })();
    return () => {
      mounted = false;
      stop?.();
    };
  }, [panelId]);

  // Close → tell main (panel returns to hidden) and let the window die
  // (the manager reacts to the layout change and destroys us).
  useEffect(() => {
    const win = getCurrentWindow();
    let unlisten: (() => void) | undefined;
    void win
      .onCloseRequested(() => {
        void emitTo("main", "panel://closed", { panelId });
      })
      .then((u) => (unlisten = u));
    return () => unlisten?.();
  }, [panelId]);

  // Geometry reports (debounced) → main persists windowRect.
  useEffect(() => {
    const win = getCurrentWindow();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const report = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        void (async () => {
          try {
            const [pos, size] = await Promise.all([win.outerPosition(), win.outerSize()]);
            void emitTo("main", "panel://geometry", {
              panelId,
              x: pos.x,
              y: pos.y,
              w: size.width,
              h: size.height,
            });
          } catch {
            /* window mid-teardown */
          }
        })();
      }, 400);
    };
    const unlistens: Array<() => void> = [];
    void win.onMoved(report).then((u) => unlistens.push(u));
    void win.onResized(report).then((u) => unlistens.push(u));
    return () => {
      if (timer) clearTimeout(timer);
      for (const u of unlistens) u();
    };
  }, [panelId]);

  // Browser context-menu suppression, same as the main App.
  useEffect(() => {
    const onContextMenu = (e: MouseEvent) => e.preventDefault();
    document.addEventListener("contextmenu", onContextMenu);
    return () => document.removeEventListener("contextmenu", onContextMenu);
  }, []);

  const def = panelId ? panelDefFor(panelId) : undefined;
  const params = panelId?.includes(":") ? panelId.slice(panelId.indexOf(":") + 1) : null;
  const title = panelId ? panelTitle(panelId, params) : "Panel";

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-(--ink-bg-1) text-(--ink-text)">
      <PanelWindowChrome panelId={panelId ?? ""} title={title} />
      <main className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        {def && panelId && ready ? (
          <PanelChromeContext.Provider value="window">
            <div className="flex h-full min-h-0 flex-col overflow-hidden">
              <def.component panelId={panelId} params={params} />
            </div>
          </PanelChromeContext.Provider>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-(--ink-text-dim)">
            {def ? "Connecting…" : `Unknown panel: ${panelId ?? "?"}`}
          </div>
        )}
      </main>
    </div>
  );
}

/**
 * The detached window's titlebar: icon + title + min/max/close. Dragging it
 * goes through OUR controller (not `data-tauri-drag-region`) so the main
 * window can render dock-back previews while the window moves — the trade
 * (documented) is that panel windows forfeit Windows Snap.
 */
function PanelWindowChrome({ panelId, title }: { panelId: string; title: string }) {
  const def = panelDefFor(panelId);
  const Icon = def?.icon;

  const winButton =
    "inline-flex h-8 w-10 items-center justify-center text-(--ink-text-dim) transition-colors hover:bg-(--ink-bg-3) hover:text-(--ink-text)";

  return (
    <header
      className="flex h-9 shrink-0 cursor-grab select-none items-center border-b border-(--ink-border) bg-(--ink-bg-2) active:cursor-grabbing"
      onPointerDown={(e) => {
        if ((e.target as HTMLElement).closest("button")) return;
        beginDetachedWindowDrag(panelId, e);
      }}
    >
      <div className="flex min-w-0 flex-1 items-center gap-2 px-3">
        {Icon && <Icon size={15} className="shrink-0 text-(--ink-accent)" />}
        <span className="truncate text-xs font-semibold">{title}</span>
      </div>
      <button
        type="button"
        aria-label="Minimize"
        className={winButton}
        onClick={() => void getCurrentWindow().minimize()}
      >
        <Minus size={14} />
      </button>
      <button
        type="button"
        aria-label="Maximize or restore"
        className={winButton}
        onClick={() => void getCurrentWindow().toggleMaximize()}
      >
        <Square size={12} />
      </button>
      <button
        type="button"
        aria-label="Close"
        className={`${winButton} hover:bg-(--ink-error) hover:text-white`}
        onClick={() => {
          // Route through the close-requested path so main gets notified.
          void getCurrentWebviewWindow().close();
        }}
      >
        <X size={14} />
      </button>
    </header>
  );
}
