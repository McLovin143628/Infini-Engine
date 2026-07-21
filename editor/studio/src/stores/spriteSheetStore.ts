/**
 * Sprite Sheet editor store (P8.2a).
 *
 * Loads one texture asset's slice model (grid + named manual rects) via
 * `texture_get_slices`, keeps a live local copy the panel edits, saves it back
 * with `texture_set_slices`, and applies a chosen slice to the selection with
 * `scene_apply_sprite_slice`. The grid overlay is computed in JS
 * ([`resolveSlices`], a mirror of the Rust `SpriteSheetSlices::resolve`) so the
 * preview updates instantly as the user drags the controls — no round-trip.
 */
import { create } from "zustand";

import type { SpriteGridDto } from "../bindings/SpriteGridDto";
import type { SpriteRectDto } from "../bindings/SpriteRectDto";
import type { SpriteSheetDto } from "../bindings/SpriteSheetDto";
import { assets as assetsIpc, scene as sceneIpc, texture as textureIpc } from "../lib/ipc";

/** A resolved slice: pixel rect + normalized UV corners + display name. */
export interface ResolvedSlice {
  name: string;
  /** Pixel rect [x, y, w, h]. */
  px: [number, number, number, number];
  uvMin: [number, number];
  uvMax: [number, number];
}

/** Default grid inserted when the user enables grid slicing on a fresh sheet. */
export function defaultGrid(): SpriteGridDto {
  return { columns: 4, rows: 4, margin_x: 0, margin_y: 0, padding_x: 0, padding_y: 0 };
}

/**
 * Resolve a slice model against its texture pixel size — a faithful port of the
 * Rust `SpriteSheetSlices::resolve` (top-left origin, grid cells first and
 * index-named, then manual rects in author order).
 */
export function resolveSlices(dto: SpriteSheetDto): ResolvedSlice[] {
  const tw = Math.max(1, dto.tex_width);
  const th = Math.max(1, dto.tex_height);
  const out: ResolvedSlice[] = [];

  const g = dto.grid;
  if (g) {
    const cols = Math.max(1, g.columns);
    const rows = Math.max(1, g.rows);
    const usableW = Math.max(0, tw - g.margin_x - (cols - 1) * g.padding_x);
    const usableH = Math.max(0, th - g.margin_y - (rows - 1) * g.padding_y);
    const cellW = usableW / cols;
    const cellH = usableH / rows;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x0 = g.margin_x + c * (cellW + g.padding_x);
        const y0 = g.margin_y + r * (cellH + g.padding_y);
        const x1 = x0 + cellW;
        const y1 = y0 + cellH;
        out.push({
          name: String(r * cols + c),
          px: [Math.round(x0), Math.round(y0), Math.round(cellW), Math.round(cellH)],
          uvMin: [x0 / tw, y0 / th],
          uvMax: [x1 / tw, y1 / th],
        });
      }
    }
  }
  for (const m of dto.manual) {
    out.push({
      name: m.name,
      px: [m.x, m.y, m.width, m.height],
      uvMin: [m.x / tw, m.y / th],
      uvMax: [(m.x + m.width) / tw, (m.y + m.height) / th],
    });
  }
  return out;
}

interface SpriteSheetState {
  /** The active texture asset id, or null when the panel has nothing open. */
  assetId: string | null;
  /** Live, editable slice model + texture dims. */
  doc: SpriteSheetDto | null;
  /** Preview thumbnail data-URL (the texture image). */
  image: string | null;
  dirty: boolean;
  busy: boolean;
  error: string | null;

  open: (assetId: string) => Promise<void>;
  setGrid: (patch: Partial<SpriteGridDto>) => void;
  toggleGrid: (enabled: boolean) => void;
  addManual: () => void;
  updateManual: (index: number, patch: Partial<SpriteRectDto>) => void;
  removeManual: (index: number) => void;
  save: () => Promise<void>;
  applySlice: (name: string, resize: boolean) => Promise<number>;
}

export const useSpriteSheetStore = create<SpriteSheetState>((set, get) => ({
  assetId: null,
  doc: null,
  image: null,
  dirty: false,
  busy: false,
  error: null,

  open: async (assetId) => {
    set({ busy: true, error: null, assetId });
    try {
      const [doc, image] = await Promise.all([
        textureIpc.getSlices(assetId),
        assetsIpc.thumbnail(assetId).catch(() => null),
      ]);
      set({ doc, image: image ?? null, dirty: false, busy: false });
    } catch (e) {
      set({ error: String(e), busy: false });
    }
  },

  setGrid: (patch) =>
    set((s) => {
      if (!s.doc?.grid) return s;
      return { doc: { ...s.doc, grid: { ...s.doc.grid, ...patch } }, dirty: true };
    }),

  toggleGrid: (enabled) =>
    set((s) => {
      if (!s.doc) return s;
      return { doc: { ...s.doc, grid: enabled ? (s.doc.grid ?? defaultGrid()) : null }, dirty: true };
    }),

  addManual: () =>
    set((s) => {
      if (!s.doc) return s;
      const n = s.doc.manual.length + 1;
      const rect: SpriteRectDto = {
        name: `sprite_${n}`,
        x: 0,
        y: 0,
        width: Math.min(32, s.doc.tex_width),
        height: Math.min(32, s.doc.tex_height),
      };
      return { doc: { ...s.doc, manual: [...s.doc.manual, rect] }, dirty: true };
    }),

  updateManual: (index, patch) =>
    set((s) => {
      if (!s.doc) return s;
      const manual = s.doc.manual.map((m, i) => (i === index ? { ...m, ...patch } : m));
      return { doc: { ...s.doc, manual }, dirty: true };
    }),

  removeManual: (index) =>
    set((s) => {
      if (!s.doc) return s;
      return { doc: { ...s.doc, manual: s.doc.manual.filter((_, i) => i !== index) }, dirty: true };
    }),

  save: async () => {
    const doc = get().doc;
    if (!doc) return;
    set({ busy: true, error: null });
    try {
      await textureIpc.setSlices(doc);
      set({ dirty: false, busy: false });
    } catch (e) {
      set({ error: String(e), busy: false });
    }
  },

  applySlice: async (name, resize) => {
    const { assetId, dirty } = get();
    if (!assetId) return 0;
    // Apply resolves against the persisted model, so flush pending edits first.
    if (dirty) await get().save();
    try {
      return await sceneIpc.applySpriteSlice(assetId, name, resize);
    } catch (e) {
      set({ error: String(e) });
      return 0;
    }
  },
}));
