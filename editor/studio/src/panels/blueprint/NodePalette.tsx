/** Right-click / double-click add-node menu: searchable, grouped by category. */
import { useMemo, useRef, useState, useEffect } from "react";

import type { NodeDef } from "../../lib/blueprintTypes";
import { FLAG_HIDDEN } from "../../lib/blueprintTypes";
import { categoryColor } from "./pinTheme";

export interface PaletteAnchor {
  x: number;
  y: number;
}

export function NodePalette({
  registry,
  anchor,
  onPick,
  onClose,
}: {
  registry: NodeDef[];
  anchor: PaletteAnchor;
  onPick: (typeId: string) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const visible = registry.filter(
      (d) =>
        (d.flags & FLAG_HIDDEN) === 0 &&
        (q === "" ||
          d.display.toLowerCase().includes(q) ||
          d.typeId.toLowerCase().includes(q) ||
          d.category.toLowerCase().includes(q)),
    );
    const byCat = new Map<string, NodeDef[]>();
    for (const d of visible) {
      const list = byCat.get(d.category) ?? [];
      list.push(d);
      byCat.set(d.category, list);
    }
    return [...byCat.entries()];
  }, [registry, query]);

  return (
    <>
      <div className="bp-palette__scrim" onClick={onClose} onContextMenu={(e) => e.preventDefault()} />
      <div
        className="bp-palette"
        style={{ left: Math.max(4, anchor.x), top: Math.max(4, anchor.y) }}
        onContextMenu={(e) => e.preventDefault()}
      >
        <input
          ref={inputRef}
          className="bp-palette__search"
          placeholder="Search nodes…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") onClose();
          }}
        />
        <div className="bp-palette__list">
          {groups.length === 0 && <div className="bp-palette__empty">No matches</div>}
          {groups.map(([cat, defs]) => (
            <div key={cat} className="bp-palette__group">
              <div className="bp-palette__cat">
                <span className="bp-palette__dot" style={{ background: categoryColor(cat) }} />
                {cat}
              </div>
              {defs.map((d) => (
                <button key={d.typeId} className="bp-palette__item" onClick={() => onPick(d.typeId)}>
                  {d.display}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
