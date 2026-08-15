/**
 * Tilemap paint store (P8.2b).
 *
 * Binds to a selected Tilemap entity, mirrors its painted cells (via
 * `tilemap_get`), and commits paint strokes back with `tilemap_paint` — one
 * backend call, hence one undo step, per stroke. The authoritative chunk map
 * lives in the Rust `SceneDoc`; this store holds only the painted cells the
 * canvas needs plus the current tool/active-tile UI state.
 *
 * The panel accumulates a stroke locally for instant feedback and commits the
 * whole batch on pointer-up; after any commit (and after an external
 * undo/redo, which the panel re-fetches on the scene version bump) the store
 * re-reads the authoritative cells.
 */
import { create } from "zustand";

import type { TilemapCellDto } from "../bindings/TilemapCellDto";
import type { TilemapDto } from "../bindings/TilemapDto";
import { tilemap as tilemapIpc } from "../lib/ipc";

/** The paint tools. */
export type TileTool = "paint" | "erase" | "fill" | "pick";

/** Max cells a single bounded flood-fill will touch (guards the unbounded grid). */
export const FLOOD_FILL_LIMIT = 4096;
/** How far past the painted bounds a flood-fill of empty space may reach. */
const FLOOD_FILL_MARGIN = 24;

const key = (x: number, y: number) => `${x},${y}`;

/** Build the fast `"x,y" → tile` lookup from a DTO's cell list. */
function cellMapOf(dto: TilemapDto | null): Map<string, number> {
  const m = new Map<string, number>();
  if (dto) for (const c of dto.cells) m.set(key(c.x, c.y), c.tile);
  return m;
}

/**
 * Compute a bounded, 4-connected flood-fill (pure — unit-tested). Fills the
 * contiguous region of `seedTile === map[seed]` reachable from `(sx, sy)` with
 * `fill`, clamped to the painted bounding box expanded by [`FLOOD_FILL_MARGIN`]
 * and capped at [`FLOOD_FILL_LIMIT`] cells so filling empty space over the
 * unbounded grid can never run away. Returns the changed cells (empty when the
 * seed already holds `fill`).
 */
export function computeFloodFill(
  map: Map<string, number>,
  cells: Array<{ x: number; y: number }>,
  sx: number,
  sy: number,
  fill: number,
): TilemapCellDto[] {
  const target = map.get(key(sx, sy)) ?? 0;
  if (target === fill) return [];

  let minX = sx;
  let minY = sy;
  let maxX = sx;
  let maxY = sy;
  for (const c of cells) {
    minX = Math.min(minX, c.x);
    minY = Math.min(minY, c.y);
    maxX = Math.max(maxX, c.x);
    maxY = Math.max(maxY, c.y);
  }
  minX -= FLOOD_FILL_MARGIN;
  minY -= FLOOD_FILL_MARGIN;
  maxX += FLOOD_FILL_MARGIN;
  maxY += FLOOD_FILL_MARGIN;

  const out: TilemapCellDto[] = [];
  const seen = new Set<string>();
  const stack: Array<[number, number]> = [[sx, sy]];
  while (stack.length && out.length < FLOOD_FILL_LIMIT) {
    const [cx, cy] = stack.pop()!;
    if (cx < minX || cx > maxX || cy < minY || cy > maxY) continue;
    const k = key(cx, cy);
    if (seen.has(k)) continue;
    seen.add(k);
    if ((map.get(k) ?? 0) !== target) continue;
    out.push({ x: cx, y: cy, tile: fill });
    stack.push([cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]);
  }
  return out;
}

interface TilemapState {
  /** Bound entity GUID, or null when nothing paintable is selected. */
  entity: string | null;
  dto: TilemapDto | null;
  /** Authoritative painted cells: `"x,y" → 1-based tile index`. */
  cellMap: Map<string, number>;
  tool: TileTool;
  /** 1-based atlas index the paint/fill tools write. */
  activeTile: number;
  busy: boolean;
  error: string | null;

  bind: (entity: string | null) => Promise<void>;
  refresh: () => Promise<void>;
  setTool: (t: TileTool) => void;
  setActiveTile: (n: number) => void;
  tileAt: (x: number, y: number) => number;
  /** Commit one stroke (mouse-down→up batch) as a single undo step. */
  paintCells: (cells: TilemapCellDto[]) => Promise<void>;
  /** Bounded flood-fill from `(x, y)` with the active tile (or erase). */
  floodFill: (x: number, y: number) => Promise<TilemapCellDto[]>;
}

/**
 * The bind generation (round-2 finding R2.F12).
 *
 * Module-level rather than store state because nothing renders because of it —
 * it exists only so a reply can ask "am I still the one being waited for?".
 * The same shape as `blueprintStore`'s `openGen`.
 */
let bindGen = 0;

/** Test-only: forget any in-flight bind. */
export function __resetTilemapBindForTest(): void {
  bindGen += 1;
}

export const useTilemapStore = create<TilemapState>((set, get) => ({
  entity: null,
  dto: null,
  cellMap: new Map(),
  tool: "paint",
  activeTile: 1,
  busy: false,
  error: null,

  bind: async (entity) => {
    const gen = ++bindGen;
    if (entity === null) {
      set({ entity: null, dto: null, cellMap: new Map(), error: null });
      return;
    }
    set({ busy: true, error: null, entity });
    try {
      const dto = await tilemapIpc.get(entity);
      // **The generation check** (round-2 finding R2.F12). `entity` was set
      // synchronously and the reply applied without re-reading it, so selecting
      // tilemap A then B quickly — or an undo/redo, or StrictMode's double
      // mount — could leave the store holding `entity: B` with A's `cellMap`,
      // and `PaintCanvas.commit` then sent A's cells to B. A stale reply is
      // dropped whole rather than merged.
      if (gen !== bindGen) return;
      set((s) => ({
        dto,
        cellMap: cellMapOf(dto),
        busy: false,
        // Keep the active tile in range of the (possibly new) palette.
        activeTile: dto ? Math.min(s.activeTile, dto.palette_cols * dto.palette_rows || 1) : 1,
      }));
    } catch (e) {
      if (gen !== bindGen) return;
      set({ error: String(e), busy: false });
    }
  },

  refresh: async () => {
    const entity = get().entity;
    if (!entity) return;
    const gen = bindGen;
    try {
      const dto = await tilemapIpc.get(entity);
      // Same race, one door along: a refresh in flight when the selection moves
      // would write the OLD entity's cells over the new one's.
      if (gen !== bindGen || get().entity !== entity) return;
      set({ dto, cellMap: cellMapOf(dto) });
    } catch (e) {
      if (gen !== bindGen) return;
      set({ error: String(e) });
    }
  },

  setTool: (tool) => set({ tool }),
  setActiveTile: (activeTile) => set({ activeTile: Math.max(0, Math.floor(activeTile)) }),

  tileAt: (x, y) => get().cellMap.get(key(x, y)) ?? 0,

  paintCells: async (cells) => {
    const { entity } = get();
    if (!entity || cells.length === 0) return;
    try {
      await tilemapIpc.paint(entity, cells);
      await get().refresh();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  floodFill: async (x, y) => {
    const { cellMap, activeTile, dto } = get();
    const cells = computeFloodFill(cellMap, dto?.cells ?? [], x, y, activeTile);
    await get().paintCells(cells);
    return cells;
  },
}));
