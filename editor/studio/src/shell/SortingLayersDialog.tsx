/**
 * Sorting Layers manager (P8.2a). A small project-level dialog (Window ▸ Sorting
 * Layers…) editing the named-layer registry: rows of `name ↔ i32 id` with
 * add/rename/remove. The Details panel keeps showing the raw `i32` on
 * `Sprite.sorting_layer` (upgrading that widget to a dropdown is a follow-up).
 *
 * Mirrors LayoutDialog: driven by `shellStore.sortingLayersOpen`, hides the
 * native viewport while open (airspace rule), persists via `layers_set`.
 */
import { useEffect, useState } from "react";
import { Plus, Trash2, X } from "lucide-react";

import type { SortingLayerDto } from "../bindings/SortingLayerDto";
import { layers as layersIpc } from "../lib/ipc";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useShellStore } from "../stores/shellStore";

export default function SortingLayersDialog() {
  const open = useShellStore((s) => s.sortingLayersOpen);
  const setOpen = useShellStore((s) => s.setSortingLayersOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);

  const [rows, setRows] = useState<SortingLayerDto[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Modal dialogs overlay the native viewport hole — hide it while open.
  useViewportOverlay(open);

  useEffect(() => {
    if (!open) return;
    layersIpc
      .get()
      .then((r) => {
        setError(null);
        setRows(r);
      })
      .catch((e) => setError(String(e)));
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  if (!open) return null;

  const update = (i: number, patch: Partial<SortingLayerDto>) =>
    setRows((r) => r.map((row, j) => (j === i ? { ...row, ...patch } : row)));
  const remove = (i: number) => setRows((r) => r.filter((_, j) => j !== i));
  const add = () => {
    const nextId = rows.reduce((m, r) => Math.max(m, r.id), 0) + 1;
    setRows((r) => [...r, { id: nextId, name: `Layer ${nextId}` }]);
  };

  const save = async () => {
    try {
      const saved = await layersIpc.set(rows);
      setRows(saved);
      pushStatus(`Sorting layers saved (${saved.length}).`);
      setOpen(false);
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-24"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
    >
      <div
        className="flex max-h-[70vh] w-[460px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center border-b border-(--ink-border) px-3 py-2">
          <span className="flex-1 font-semibold">Sorting Layers</span>
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
          <div className="mb-1 flex gap-2 px-1 text-xs text-(--ink-text-faint)">
            <span className="w-16">ID</span>
            <span className="flex-1">Name</span>
            <span className="w-6" />
          </div>
          {rows.map((row, i) => (
            <div key={i} className="mb-1 flex items-center gap-2">
              <input
                type="number"
                value={row.id}
                onChange={(e) => update(i, { id: Math.floor(Number(e.target.value) || 0) })}
                className="w-16 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1 text-right text-xs outline-none focus:border-(--ink-accent)"
              />
              <input
                value={row.name}
                onChange={(e) => update(i, { name: e.target.value })}
                className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 text-xs outline-none focus:border-(--ink-accent)"
              />
              <button
                aria-label={`Remove layer ${row.name}`}
                onClick={() => remove(i)}
                className="rounded p-1 text-(--ink-text-faint) hover:bg-(--ink-error)/15 hover:text-(--ink-error)"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
          <button
            onClick={add}
            className="mt-1 flex h-6 items-center gap-1 rounded border border-(--ink-border) px-2 text-xs hover:border-(--ink-accent)"
          >
            <Plus size={12} /> Add Layer
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
