/**
 * Place Actors palette: a searchable catalog of spawnable actor types. Clicking
 * a type creates it at the world origin via `scene_create` (the store's
 * `createEntity`). The catalog is the shared `lib/spawnables` list — the same
 * set the Outliner's "+ Add" menu offers — so the two stay in sync.
 *
 * Dragging a type onto the viewport hole spawns it at the cursor's world point
 * instead (Wave 2): the same pointer-capture → `[data-viewport-hole]` → IPC
 * pathway the Content Drawer uses, with a `"spawn:<kind>"` drop payload the
 * native viewport parses (`viewport_drop` → `EngineHost::spawn_drop`). A plain
 * click still spawns at the origin.
 */
import { useMemo, useRef, useState } from "react";
import {
  Box,
  Camera,
  Circle,
  Cone,
  Cylinder,
  Folder,
  Frame,
  Grid2x2,
  Image,
  Lightbulb,
  Mountain,
  Search,
  Sparkles,
  Spotlight,
  Square,
  Sun,
  Type,
} from "lucide-react";

import type { SpawnKind } from "../../bindings/SpawnKind";
import { fuzzyMatch } from "../../lib/fuzzy";
import { viewport } from "../../lib/ipc";
import { SPAWNABLE_SECTIONS, type SpawnableItem } from "../../lib/spawnables";
import { useSceneStore } from "../../stores/sceneStore";

/** Drag threshold (px) before a pointer-down becomes a viewport drag-spawn. */
const DRAG_THRESHOLD = 5;

/** Static per-kind glyph (a switch, not a dynamic component binding, to satisfy
 *  the react-hooks static-components rule — same idiom as the Content Drawer). */
function SpawnGlyph({ kind, size }: { kind: SpawnKind; size: number }) {
  const p = { size, className: "shrink-0 text-(--ink-accent)" };
  switch (kind) {
    case "cube":
      return <Box {...p} />;
    case "sphere":
      return <Circle {...p} />;
    case "plane":
      return <Square {...p} />;
    case "cylinder":
      return <Cylinder {...p} />;
    case "cone":
      return <Cone {...p} />;
    case "terrain":
      return <Mountain {...p} />;
    case "point_light":
      return <Lightbulb {...p} />;
    case "directional_light":
      return <Sun {...p} />;
    case "spot_light":
      return <Spotlight {...p} />;
    case "camera":
      return <Camera {...p} />;
    case "sprite":
      return <Image {...p} />;
    case "tilemap":
      return <Grid2x2 {...p} />;
    case "text2d":
      return <Type {...p} />;
    case "nine_slice":
      return <Frame {...p} />;
    case "light2d":
      return <Sparkles {...p} />;
    default:
      return <Folder {...p} />;
  }
}

/** One catalog tile: click spawns at the origin; dragging onto the viewport
 *  hole drops a `"spawn:<kind>"` payload at the cursor's world point (Wave 2). */
function SpawnTile({
  item,
  ready,
  onClickSpawn,
}: {
  item: SpawnableItem;
  ready: boolean;
  onClickSpawn: (item: SpawnableItem) => void;
}) {
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const dragging = useRef(false);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || !ready) return;
    dragStart.current = { x: e.clientX, y: e.clientY };
    dragging.current = false;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragStart.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    if (!dragging.current && Math.hypot(dx, dy) > DRAG_THRESHOLD) dragging.current = true;
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const wasDragging = dragging.current;
    dragStart.current = null;
    dragging.current = false;
    if (!wasDragging) {
      onClickSpawn(item); // a plain click → spawn at the origin
      return;
    }
    const hole = document.querySelector("[data-viewport-hole]");
    if (!hole) return;
    const r = hole.getBoundingClientRect();
    const inside =
      e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    if (!inside) return;
    // Physical pixels relative to the hole's top-left corner (the viewport's
    // native-window contract).
    const dpr = window.devicePixelRatio || 1;
    void viewport
      .drop({
        x: (e.clientX - r.left) * dpr,
        y: (e.clientY - r.top) * dpr,
        payload: `spawn:${item.kind}`,
      })
      .catch((err) => console.error("drag-spawn failed", err));
  };

  return (
    <button
      disabled={!ready}
      title={`Add ${item.label} — click for origin, drag onto the viewport to place`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      className="flex touch-none select-none items-center gap-2 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1.5 text-left text-xs hover:border-(--ink-accent) hover:bg-(--ink-bg-3) disabled:opacity-40"
    >
      <SpawnGlyph kind={item.kind} size={15} />
      <span className="truncate">{item.label}</span>
    </button>
  );
}

export default function PlaceActorsPanel() {
  const createEntity = useSceneStore((s) => s.createEntity);
  const ready = useSceneStore((s) => s.ready);
  const [search, setSearch] = useState("");

  const sections = useMemo(() => {
    const q = search.trim();
    if (!q) return SPAWNABLE_SECTIONS;
    return SPAWNABLE_SECTIONS.map((sec) => ({
      ...sec,
      items: sec.items.filter((it) => fuzzyMatch(q, it.label) !== null),
    })).filter((sec) => sec.items.length > 0);
  }, [search]);

  const spawn = (item: SpawnableItem) => {
    // Unparented → the backend places it at the world origin.
    void createEntity(item.kind, null);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Search */}
      <div className="flex items-center gap-1 border-b border-(--ink-border) p-1.5">
        <div className="flex h-6 flex-1 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 focus-within:border-(--ink-accent)">
          <Search size={12} className="shrink-0 text-(--ink-text-faint)" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search actor types…"
            className="w-full bg-transparent text-xs outline-none placeholder:text-(--ink-text-faint)"
          />
        </div>
      </div>

      {/* Catalog */}
      <div className="min-h-0 flex-1 overflow-auto p-2">
        {sections.length === 0 ? (
          <div className="p-2 text-xs text-(--ink-text-faint)">No actor types match “{search}”.</div>
        ) : (
          sections.map((section, si) => (
            <div key={section.heading ?? "primitives"} className={si > 0 ? "mt-3" : ""}>
              {section.heading && (
                <div className="mb-1 px-1 text-[10px] font-semibold uppercase tracking-wide text-(--ink-text-faint)">
                  {section.heading}
                </div>
              )}
              <div className="grid grid-cols-2 gap-1.5">
                {section.items.map((item) => (
                  <SpawnTile key={item.kind} item={item} ready={ready} onClickSpawn={spawn} />
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-(--ink-border) px-2 py-1 text-[11px] text-(--ink-text-faint)">
        Click to add at the origin, or drag onto the viewport to place at the cursor.
      </div>
    </div>
  );
}
