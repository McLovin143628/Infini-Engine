/**
 * Custom window chrome (P1.1 batch 1). Native decorations are off
 * (tauri.conf.json `decorations: false`); this strip provides the drag
 * region, the menu bar, and min/max/close controls. Double-click on the
 * drag region toggles maximize (handled natively by `data-tauri-drag-region`).
 */
import { useEffect, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { Copy, Minus, Square, X } from "lucide-react";
import MenuBar from "./MenuBar";

export default function TitleBar() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    const win = getCurrentWindow();
    let unlisten: (() => void) | undefined;
    let disposed = false;
    win
      .isMaximized()
      .then((m) => {
        if (!disposed) setMaximized(m);
      })
      .catch(() => {});
    win
      .onResized(() => {
        win
          .isMaximized()
          .then((m) => {
            if (!disposed) setMaximized(m);
          })
          .catch(() => {});
      })
      .then((fn) => {
        if (disposed) fn();
        else unlisten = fn;
      })
      .catch(() => {});
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  const win = () => getCurrentWindow();

  return (
    <div className="flex h-[34px] shrink-0 items-stretch border-b border-(--ink-border) bg-(--ink-bg-2)">
      <div className="flex items-center px-3 text-base font-semibold text-(--ink-accent)">∞</div>
      <MenuBar />
      {/* Drag region fills the middle; window title lives inside it. */}
      <div
        data-tauri-drag-region
        className="flex min-w-8 flex-1 items-center justify-center text-(--ink-text-dim)"
      >
        <span data-tauri-drag-region className="truncate px-4">
          Infini Engine
        </span>
      </div>
      <div className="flex items-stretch">
        <button
          aria-label="Minimize"
          className="w-11 hover:bg-(--ink-bg-3)"
          onClick={() => void win().minimize()}
        >
          <Minus size={14} className="mx-auto" />
        </button>
        <button
          aria-label={maximized ? "Restore" : "Maximize"}
          className="w-11 hover:bg-(--ink-bg-3)"
          onClick={() => void win().toggleMaximize()}
        >
          {maximized ? (
            <Copy size={12} className="mx-auto -scale-x-100" />
          ) : (
            <Square size={12} className="mx-auto" />
          )}
        </button>
        <button
          aria-label="Close"
          className="w-11 hover:bg-(--ink-error) hover:text-white"
          onClick={() => void win().close()}
        >
          <X size={15} className="mx-auto" />
        </button>
      </div>
    </div>
  );
}
