/**
 * Project Settings (Wave E, batch A) — the dialog behind Edit ▸ Project
 * Settings…, which dispatched a handler-less command until this wave.
 *
 * Scope is deliberately "the per-project `.infinity/*` family that already has
 * a backend and no home": pixels-per-unit (`project_settings_*`) and the
 * **collision-layer names** (`collision_layers_*`), whose IPC has existed since
 * P12.1 with no UI at all. Sorting layers keep their own dialog (Window ▸
 * Sorting Layers…) and are linked from here rather than duplicated.
 *
 * Unlike Editor Preferences these are project files with explicit Save doors,
 * so the dialog keeps the SortingLayersDialog shape: load on open, edit locally,
 * write on Save. Airspace guard as always.
 */
import { useEffect, useState } from "react";
import { X } from "lucide-react";

import type { CollisionLayerDto } from "../bindings/CollisionLayerDto";
import { executeCommand } from "../lib/commands";
import { collisionLayers as collisionIpc, projectSettings as projectIpc } from "../lib/ipc";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useShellStore } from "../stores/shellStore";
import { useViewportStore } from "../stores/viewportStore";

export default function ProjectSettingsDialog() {
  const open = useShellStore((s) => s.projectSettingsOpen);
  const setOpen = useShellStore((s) => s.setProjectSettingsOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);
  const setPixelsPerUnit = useViewportStore((s) => s.setPixelsPerUnit);

  const [ppu, setPpu] = useState(100);
  const [layers, setLayers] = useState<CollisionLayerDto[]>([]);
  const [error, setError] = useState<string | null>(null);

  useViewportOverlay(open);

  useEffect(() => {
    if (!open) return;
    // The error is cleared in the resolve handlers, not here: a synchronous
    // setState in an effect body is a cascading render (eslint
    // react-hooks/set-state-in-effect is an error in this repo).
    projectIpc
      .get()
      .then((s) => {
        setError(null);
        setPpu(s.pixels_per_unit);
      })
      .catch((e) => setError(String(e)));
    collisionIpc
      .get()
      .then((rows) => {
        setError(null);
        setLayers(rows);
      })
      .catch((e) => setError(String(e)));
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  if (!open) return null;

  const save = async () => {
    try {
      // Pixels-per-unit goes through the viewport store's setter, not the raw
      // IPC: that setter is the one door that also re-pushes the 2D snap config
      // to the native viewport, so the setting APPLIES rather than merely
      // persisting.
      setPixelsPerUnit(ppu);
      const saved = await collisionIpc.set(layers);
      setLayers(saved);
      pushStatus(`Project settings saved (${saved.length} collision layers).`);
      setOpen(false);
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-20"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
    >
      <div
        className="flex max-h-[74vh] w-[520px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center border-b border-(--ink-border) px-3 py-2">
          <span className="flex-1 font-semibold">Project Settings</span>
          <button
            aria-label="Close dialog"
            className="rounded p-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
            onClick={() => setOpen(false)}
          >
            <X size={14} />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-3">
          {error && <div className="mb-2 text-xs text-(--ink-error)">{error}</div>}

          <div className="mb-2 text-xs text-(--ink-text-dim)">2D</div>
          <label className="mb-3 flex items-center gap-3 text-xs">
            <span className="w-40 shrink-0">
              Pixels per unit <span className="text-(--ink-text-faint)">(px / m)</span>
            </span>
            <input
              type="number"
              value={ppu}
              min={1}
              max={100000}
              step={1}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (!Number.isFinite(n) || n <= 0) return;
                setPpu(n);
              }}
              className="w-28 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 text-right outline-none focus:border-(--ink-accent)"
            />
          </label>

          <div className="mb-2 text-xs text-(--ink-text-dim)">
            Collision layers <span className="text-(--ink-text-faint)">(bit → name)</span>
          </div>
          <div className="mb-3 max-h-64 overflow-auto rounded border border-(--ink-border) p-2">
            {layers.length === 0 && (
              <div className="text-xs text-(--ink-text-faint)">No named layers yet.</div>
            )}
            {layers.map((row, i) => (
              <div key={row.bit} className="mb-1 flex items-center gap-2 text-xs">
                <span className="w-10 text-right text-(--ink-text-faint)">{row.bit}</span>
                <input
                  value={row.name}
                  onChange={(e) =>
                    setLayers((rs) =>
                      rs.map((r, j) => (j === i ? { ...r, name: e.target.value } : r)),
                    )
                  }
                  className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 outline-none focus:border-(--ink-accent)"
                />
              </div>
            ))}
          </div>

          <button
            onClick={() => {
              setOpen(false);
              executeCommand("window.sortingLayers");
            }}
            className="text-xs text-(--ink-accent) hover:underline"
          >
            Sorting layers (2D draw order) →
          </button>
        </div>

        <div className="flex justify-end gap-2 border-t border-(--ink-border) px-3 py-2">
          <button
            onClick={() => setOpen(false)}
            className="rounded px-3 py-1 text-xs text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
          >
            Cancel
          </button>
          <button
            onClick={() => void save()}
            className="rounded bg-(--ink-accent) px-3 py-1 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
