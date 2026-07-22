/**
 * Entity-reference picker (E-P1). A component field of kind `entity_ref` (e.g. a
 * physics joint's `other` body) renders as a button showing the referenced
 * entity's name; clicking opens a popover that lists the scene tree with a fuzzy
 * filter. Picking an entity writes its GUID; a clear button unbinds it.
 */
import { useMemo, useRef, useState } from "react";
import { Link2, Link2Off, X } from "lucide-react";
import { fuzzyMatch } from "../lib/fuzzy";
import { useSceneStore } from "../stores/sceneStore";

export function EntityRefField({
  value,
  onChange,
}: {
  /** The referenced entity GUID, or `null` when unbound. */
  value: string | null;
  onChange: (guid: string | null) => void;
}) {
  const nodes = useSceneStore((s) => s.nodes);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);

  const name = value ? (nodes[value]?.name ?? "(missing)") : "None";

  const matches = useMemo(() => {
    const q = filter.trim();
    const all = Object.values(nodes);
    const rows = q ? all.filter((n) => fuzzyMatch(q, n.name) !== null) : all;
    return rows.sort((a, b) => a.name.localeCompare(b.name)).slice(0, 200);
  }, [nodes, filter]);

  const pick = (guid: string | null) => {
    onChange(guid);
    setOpen(false);
    setFilter("");
  };

  return (
    <div ref={rootRef} className="relative flex min-w-0 flex-1 items-center gap-1">
      <button
        className="flex h-6 min-w-0 flex-1 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 text-xs hover:border-(--ink-accent)"
        onClick={() => setOpen((o) => !o)}
        title={value ?? "Unbound"}
      >
        {value ? (
          <Link2 size={11} className="shrink-0 text-(--ink-text-faint)" />
        ) : (
          <Link2Off size={11} className="shrink-0 text-(--ink-text-faint)" />
        )}
        <span className="min-w-0 flex-1 truncate text-left">{name}</span>
      </button>
      {value && (
        <button
          aria-label="Clear entity reference"
          className="flex size-5 shrink-0 items-center justify-center rounded-sm text-(--ink-text-faint) hover:text-(--ink-text)"
          onClick={() => pick(null)}
        >
          <X size={12} />
        </button>
      )}
      {open && (
        <div className="absolute top-7 left-0 z-50 flex max-h-56 w-56 flex-col rounded border border-(--ink-border) bg-(--ink-bg-1) shadow-lg">
          <input
            autoFocus
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
            placeholder="Search entities…"
            className="m-1 h-6 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 text-xs outline-none focus:border-(--ink-accent)"
          />
          <div className="min-h-0 flex-1 overflow-auto pb-1">
            {matches.length === 0 && (
              <div className="px-2 py-1 text-[11px] text-(--ink-text-faint)">No entities</div>
            )}
            {matches.map((n) => (
              <button
                key={n.guid}
                className="flex w-full items-center gap-2 px-2 py-0.5 text-left text-xs hover:bg-(--ink-bg-3)"
                onClick={() => pick(n.guid)}
              >
                <span className="min-w-0 flex-1 truncate">{n.name}</span>
                <span className="shrink-0 text-[10px] text-(--ink-text-faint)">{n.kind}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
