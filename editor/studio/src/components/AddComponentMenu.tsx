/**
 * "+ Add Component" menu (E-P1). A button that opens a fuzzy-filtered popover of
 * the addable components (from `scene_list_addable_components`); picking one adds
 * a Default instance to the current selection via the scene store.
 */
import { useEffect, useMemo, useState } from "react";
import { Plus } from "lucide-react";
import type { AddableComponentDto } from "../bindings/AddableComponentDto";
import { scene as sceneIpc } from "../lib/ipc";
import { fuzzyMatch } from "../lib/fuzzy";
import { useSceneStore } from "../stores/sceneStore";

export function AddComponentMenu() {
  const addComponent = useSceneStore((s) => s.addComponent);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const [items, setItems] = useState<AddableComponentDto[]>([]);

  // Load the addable list lazily the first time the menu opens.
  useEffect(() => {
    if (open && items.length === 0) {
      sceneIpc
        .listAddableComponents()
        .then(setItems)
        .catch((e) => console.error("listAddableComponents failed", e));
    }
  }, [open, items.length]);

  const matches = useMemo(() => {
    const q = filter.trim();
    return q ? items.filter((c) => fuzzyMatch(q, c.display) !== null) : items;
  }, [items, filter]);

  const pick = (typePath: string) => {
    void addComponent(typePath);
    setOpen(false);
    setFilter("");
  };

  return (
    <div className="relative">
      <button
        className="flex h-6 w-full items-center justify-center gap-1 rounded border border-dashed border-(--ink-border) text-xs text-(--ink-text-dim) hover:border-(--ink-accent) hover:text-(--ink-text)"
        onClick={() => setOpen((o) => !o)}
      >
        <Plus size={12} /> Add Component
      </button>
      {open && (
        <div className="absolute top-7 left-0 z-50 flex max-h-64 w-full flex-col rounded border border-(--ink-border) bg-(--ink-bg-1) shadow-lg">
          <input
            autoFocus
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
            placeholder="Search components…"
            className="m-1 h-6 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 text-xs outline-none focus:border-(--ink-accent)"
          />
          <div className="min-h-0 flex-1 overflow-auto pb-1">
            {matches.length === 0 && (
              <div className="px-2 py-1 text-[11px] text-(--ink-text-faint)">No components</div>
            )}
            {matches.map((c) => (
              <button
                key={c.type_path}
                className="flex w-full items-center px-2 py-0.5 text-left text-xs hover:bg-(--ink-bg-3)"
                onClick={() => pick(c.type_path)}
              >
                {c.display}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
