/**
 * Outliner (P1.4.1 shell, P3.2 live): the world tree, bound to the backend ECS
 * document. Supports create (Add menu), rename (double-click), reparent (drag),
 * delete (Del / context), multi-select (Ctrl-click), and visibility toggles —
 * all routed through scene commands; the tree re-syncs from `world://delta`.
 */
import { useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Plus,
  Search,
} from "lucide-react";
import type { SpawnKind } from "../bindings/SpawnKind";
import { cn } from "../lib/utils";
import { fuzzyMatch } from "../lib/fuzzy";
import { SPAWNABLE_SECTIONS } from "../lib/spawnables";
import { useDockLayout } from "./dock/dockLayoutStore";
import { outlinerRows, useSceneStore } from "../stores/sceneStore";

export default function OutlinerPanel() {
  const nodes = useSceneStore((s) => s.nodes);
  const roots = useSceneStore((s) => s.roots);
  const collapsed = useSceneStore((s) => s.collapsed);
  const selection = useSceneStore((s) => s.selection);
  const select = useSceneStore((s) => s.select);
  const toggleVisible = useSceneStore((s) => s.toggleVisible);
  const toggleCollapsed = useSceneStore((s) => s.toggleCollapsed);
  const createEntity = useSceneStore((s) => s.createEntity);
  const deleteSelected = useSceneStore((s) => s.deleteSelected);
  const rename = useSceneStore((s) => s.rename);
  const reparent = useSceneStore((s) => s.reparent);

  const [search, setSearch] = useState("");
  const [addOpen, setAddOpen] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<string | null>(null);
  const dragGuid = useRef<string | null>(null);

  const rows = useMemo(() => {
    const all = outlinerRows(nodes, roots, search ? [] : collapsed);
    if (!search.trim()) return all;
    return all.filter(({ node }) => fuzzyMatch(search.trim(), node.name) !== null);
  }, [nodes, roots, collapsed, search]);

  const onAdd = (kind: SpawnKind) => {
    setAddOpen(false);
    const parent = selection.length === 1 ? selection[0] : null;
    void createEntity(kind, parent);
    // Authoring a tilemap opens its paint panel (which then binds to the new,
    // auto-selected entity via the selection-awareness in TilemapPanel).
    if (kind === "tilemap") useDockLayout.getState().openPanel("tilemap");
  };

  return (
    <div
      className="flex min-h-0 flex-1 flex-col"
      onKeyDown={(e) => {
        if ((e.key === "Delete" || e.key === "Backspace") && renaming === null) {
          e.preventDefault();
          deleteSelected();
        }
      }}
      tabIndex={0}
    >
      {/* Search + Add */}
      <div className="flex items-center gap-1 border-b border-(--ink-border) p-1.5">
        <div className="relative">
          <button
            aria-label="Add actor"
            className="flex h-6 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 text-xs hover:border-(--ink-accent)"
            onClick={() => setAddOpen((o) => !o)}
          >
            <Plus size={12} /> Add
          </button>
          {addOpen && (
            <div className="absolute left-0 top-7 z-10 max-h-96 w-44 overflow-auto rounded border border-(--ink-border) bg-(--ink-bg-1) py-1 shadow-lg">
              {SPAWNABLE_SECTIONS.map((section, si) => (
                <div key={si} className={si > 0 ? "mt-1 border-t border-(--ink-border) pt-1" : ""}>
                  {section.heading && (
                    <div className="px-3 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-(--ink-text-faint)">
                      {section.heading}
                    </div>
                  )}
                  {section.items.map((item) => (
                    <button
                      key={item.kind}
                      className="block w-full px-3 py-1 text-left text-xs hover:bg-(--ink-bg-3)"
                      onClick={() => onAdd(item.kind)}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
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

      {/* Tree */}
      <div
        className="min-h-0 flex-1 overflow-auto py-0.5"
        role="tree"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (dragGuid.current) reparent(dragGuid.current, null);
          dragGuid.current = null;
          setDragOver(null);
        }}
      >
        {rows.map(({ node, depth }) => {
          const selected = selection.includes(node.guid);
          const hasChildren = node.children.length > 0;
          const isCollapsed = collapsed.includes(node.guid);
          const dimmed = !node.effective_visible;
          return (
            <div
              key={node.guid}
              role="treeitem"
              aria-selected={selected}
              aria-expanded={hasChildren ? !isCollapsed : undefined}
              draggable={renaming !== node.guid}
              onDragStart={() => (dragGuid.current = node.guid)}
              onDragOver={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setDragOver(node.guid);
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                if (dragGuid.current && dragGuid.current !== node.guid) {
                  reparent(dragGuid.current, node.guid);
                }
                dragGuid.current = null;
                setDragOver(null);
              }}
              className={cn(
                "group flex h-6 cursor-default items-center gap-1 px-2 text-xs",
                selected
                  ? "bg-(--ink-selection) text-(--ink-text)"
                  : "hover:bg-(--ink-bg-3)/70",
                dragOver === node.guid && "outline outline-1 outline-(--ink-accent)",
              )}
              style={{ paddingLeft: 8 + depth * 14 }}
              onClick={(e) => select([node.guid], e.ctrlKey || e.metaKey)}
              onDoubleClick={() => setRenaming(node.guid)}
            >
              <button
                aria-label={isCollapsed ? "Expand" : "Collapse"}
                className={cn(
                  "flex size-4 shrink-0 items-center justify-center text-(--ink-text-faint) hover:text-(--ink-text)",
                  !hasChildren && "invisible",
                )}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleCollapsed(node.guid);
                }}
              >
                {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
              </button>
              <button
                aria-label={node.visible ? `Hide ${node.name}` : `Show ${node.name}`}
                className={cn(
                  "flex size-4 shrink-0 items-center justify-center",
                  node.visible && !dimmed
                    ? "text-(--ink-text-dim) opacity-0 group-hover:opacity-100"
                    : "text-(--ink-text-faint)",
                )}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleVisible(node.guid);
                }}
              >
                {node.visible ? <Eye size={12} /> : <EyeOff size={12} />}
              </button>
              {renaming === node.guid ? (
                <input
                  autoFocus
                  defaultValue={node.name}
                  className="flex-1 rounded border border-(--ink-accent) bg-(--ink-bg-1) px-1 text-xs outline-none"
                  onClick={(e) => e.stopPropagation()}
                  onBlur={(e) => {
                    const v = e.target.value.trim();
                    if (v && v !== node.name) rename(node.guid, v);
                    setRenaming(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                    if (e.key === "Escape") setRenaming(null);
                  }}
                />
              ) : (
                <span className={cn("flex-1 truncate", dimmed && "text-(--ink-text-faint)")}>
                  {node.name}
                </span>
              )}
              <span className="w-28 truncate text-(--ink-text-faint)">{node.kind}</span>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="border-t border-(--ink-border) px-2 py-0.5 text-[11px] text-(--ink-text-faint)">
        {rows.length} {rows.length === 1 ? "actor" : "actors"}
        {search && " (filtered)"}
      </div>
    </div>
  );
}
