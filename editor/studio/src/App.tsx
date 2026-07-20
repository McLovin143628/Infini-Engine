import { useEffect, useState } from "react";
import { app } from "./lib/ipc";
import ViewportPanel from "./viewport/ViewportPanel";

/**
 * Placeholder shell proving the stack end-to-end (React ⇄ typed IPC ⇄ Rust).
 * Phase 1 replaces this with the real docking workspace, menu system, and
 * theme engine — see docs/ROADMAP.md §6 P1.
 */
export default function App() {
  const [version, setVersion] = useState<string>("…");

  useEffect(() => {
    app.version().then(setVersion).catch(() => setVersion("dev"));
  }, []);

  return (
    <div className="flex h-full flex-col">
      {/* Menu bar */}
      <div className="flex h-8 items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
        <span className="mr-2 font-semibold text-(--ink-accent)">∞</span>
        {["File", "Edit", "Window", "Tools", "Build", "Platforms", "Select", "Actor", "Help"].map(
          (m) => (
            <button
              key={m}
              className="rounded px-2 py-0.5 hover:bg-(--ink-bg-3)"
            >
              {m}
            </button>
          ),
        )}
        <span className="ml-auto text-(--ink-text-dim)">Infinity Engine</span>
      </div>

      {/* Main area */}
      <div className="flex min-h-0 flex-1">
        {/* The native wgpu child window mirrors this element's rectangle. */}
        <div className="m-1 flex flex-1">
          <ViewportPanel />
        </div>

        {/* Right stack: Outliner over Details */}
        <div className="m-1 ml-0 flex w-72 flex-col gap-1">
          <div className="flex-1 rounded border border-(--ink-border) bg-(--ink-bg-1)">
            <div className="border-b border-(--ink-border) bg-(--ink-bg-2) px-2 py-1">Outliner</div>
            <div className="p-2 text-(--ink-text-dim)">No world loaded</div>
          </div>
          <div className="flex-1 rounded border border-(--ink-border) bg-(--ink-bg-1)">
            <div className="border-b border-(--ink-border) bg-(--ink-bg-2) px-2 py-1">Details</div>
            <div className="p-2 text-(--ink-text-dim)">Select an object to view details</div>
          </div>
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-7 items-center gap-4 border-t border-(--ink-border) bg-(--ink-bg-2) px-3 text-(--ink-text-dim)">
        <button className="hover:text-(--ink-text)">Content Drawer</button>
        <button className="hover:text-(--ink-text)">Output Log</button>
        <span className="ml-auto">Infinity Engine v{version}</span>
      </div>
    </div>
  );
}
