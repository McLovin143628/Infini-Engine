/**
 * Outliner (P1.4.1): searchable actor tree with type column + visibility
 * eyes, UE layout parity. Mock data until the ECS world binding (P3.2).
 */
import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Eye, EyeOff, Search } from "lucide-react";
import { cn } from "../lib/utils";
import { fuzzyMatch } from "../lib/fuzzy";
import { flattenTree, useSceneStore, type MockActor } from "../stores/sceneStore";

export default function OutlinerPanel() {
  const roots = useSceneStore((s) => s.roots);
  const collapsed = useSceneStore((s) => s.collapsed);
  const selectedIds = useSceneStore((s) => s.selectedIds);
  const select = useSceneStore((s) => s.select);
  const toggleVisible = useSceneStore((s) => s.toggleVisible);
  const toggleCollapsed = useSceneStore((s) => s.toggleCollapsed);
  const [search, setSearch] = useState("");

  const rows = useMemo(() => {
    const all = flattenTree(roots, search ? [] : collapsed);
    if (!search.trim()) return all;
    return all.filter(({ actor }) => fuzzyMatch(search.trim(), actor.name) !== null);
  }, [roots, collapsed, search]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Search row */}
      <div className="flex items-center gap-1 border-b border-(--ink-border) p-1.5">
        <div className="flex h-6 flex-1 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 focus-within:border-(--ink-accent)">
          <Search size={12} className="shrink-0 text-(--ink-text-faint)" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search…"
            className="w-full bg-transparent text-xs outline-none placeholder:text-(--ink-text-faint)"
          />
        </div>
      </div>

      {/* Header */}
      <div className="flex items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2 py-0.5 text-[11px] text-(--ink-text-dim)">
        <span className="w-5" />
        <span className="flex-1">Item Label</span>
        <span className="w-28 truncate border-l border-(--ink-border) pl-2">Type</span>
      </div>

      {/* Tree rows */}
      <div className="min-h-0 flex-1 overflow-auto py-0.5" role="tree">
        {rows.map(({ actor, depth }) => {
          const selected = selectedIds.includes(actor.id);
          const hasChildren = actor.children.length > 0;
          const isCollapsed = collapsed.includes(actor.id);
          const parentHidden = !isEffectivelyVisible(useSceneStore.getState().roots, actor.id);
          return (
            <div
              key={actor.id}
              role="treeitem"
              aria-selected={selected}
              aria-expanded={hasChildren ? !isCollapsed : undefined}
              className={cn(
                "group flex h-6 cursor-default items-center gap-1 px-2 text-xs",
                selected
                  ? "bg-(--ink-selection) text-(--ink-text)"
                  : "hover:bg-(--ink-bg-3)/70",
              )}
              style={{ paddingLeft: 8 + depth * 14 }}
              onClick={(e) => select([actor.id], e.ctrlKey)}
            >
              <button
                aria-label={isCollapsed ? "Expand" : "Collapse"}
                className={cn(
                  "flex size-4 shrink-0 items-center justify-center text-(--ink-text-faint) hover:text-(--ink-text)",
                  !hasChildren && "invisible",
                )}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleCollapsed(actor.id);
                }}
              >
                {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
              </button>
              <button
                aria-label={actor.visible ? `Hide ${actor.name}` : `Show ${actor.name}`}
                className={cn(
                  "flex size-4 shrink-0 items-center justify-center",
                  actor.visible && !parentHidden
                    ? "text-(--ink-text-dim) opacity-0 group-hover:opacity-100"
                    : "text-(--ink-text-faint)",
                )}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleVisible(actor.id);
                }}
              >
                {actor.visible ? <Eye size={12} /> : <EyeOff size={12} />}
              </button>
              <span
                className={cn(
                  "flex-1 truncate",
                  (!actor.visible || parentHidden) && "text-(--ink-text-faint)",
                )}
              >
                {actor.name}
              </span>
              <span className="w-28 truncate text-(--ink-text-faint)">{actor.type}</span>
            </div>
          );
        })}
      </div>

      {/* Footer count */}
      <div className="border-t border-(--ink-border) px-2 py-0.5 text-[11px] text-(--ink-text-faint)">
        {rows.length} {rows.length === 1 ? "actor" : "actors"}
        {search && " (filtered)"}
      </div>
    </div>
  );
}

/** An actor renders dimmed when any ancestor is hidden. */
function isEffectivelyVisible(roots: MockActor[], id: string): boolean {
  const path: MockActor[] = [];
  const walk = (list: MockActor[], trail: MockActor[]): boolean => {
    for (const a of list) {
      const next = [...trail, a];
      if (a.id === id) {
        path.push(...next);
        return true;
      }
      if (walk(a.children, next)) return true;
    }
    return false;
  };
  walk(roots, []);
  return path.every((a) => a.visible);
}
