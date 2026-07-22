/**
 * Place Actors palette: a searchable catalog of spawnable actor types. Clicking
 * a type creates it at the world origin via `scene_create` (the store's
 * `createEntity`). The catalog is the shared `lib/spawnables` list — the same
 * set the Outliner's "+ Add" menu offers — so the two stay in sync.
 *
 * Scope note: this is click-to-spawn. Drag-to-viewport-at-cursor (the Content
 * Drawer's pointer-capture → `[data-viewport-hole]` → IPC pathway) can't be
 * reused as-is: that path calls `scene_spawn_asset(assetId)`, and there is no
 * command to drop a bare `SpawnKind` at a pick point (`scene_create` always
 * places at the origin). Wiring it would need a backend command — see report.
 */
import { useMemo, useState } from "react";
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
import { SPAWNABLE_SECTIONS, type SpawnableItem } from "../../lib/spawnables";
import { useSceneStore } from "../../stores/sceneStore";

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
                  <button
                    key={item.kind}
                    disabled={!ready}
                    title={`Add ${item.label} at origin`}
                    onClick={() => spawn(item)}
                    className="flex items-center gap-2 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1.5 text-left text-xs hover:border-(--ink-accent) hover:bg-(--ink-bg-3) disabled:opacity-40"
                  >
                    <SpawnGlyph kind={item.kind} size={15} />
                    <span className="truncate">{item.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-(--ink-border) px-2 py-1 text-[11px] text-(--ink-text-faint)">
        Click a type to add it at the origin.
      </div>
    </div>
  );
}
