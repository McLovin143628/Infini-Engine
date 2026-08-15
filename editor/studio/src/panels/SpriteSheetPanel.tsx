/**
 * Sprite Sheet panel (P8.2a): slice a texture atlas into addressable sub-sprites.
 *
 * Left: the texture preview on an HTML canvas with a live slice-grid overlay
 * (this is a normal DOM panel, not the native viewport — no airspace concern).
 * Right: grid controls (cols/rows/margin/padding) that update the overlay
 * instantly, a named manual-rect list, Save (persists to the sidecar), and
 * "Apply to Selection" for the highlighted slice.
 *
 * Opened by double-clicking a `.inf_tex` asset in the Content Drawer, mirroring
 * how a material opens the Material panel.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Plus, Save, Trash2 } from "lucide-react";

import { useShellStore } from "../stores/shellStore";
import {
  MAX_GRID_CELLS,
  resolveSlices,
  useSpriteSheetStore,
  type ResolvedSlice,
} from "../stores/spriteSheetStore";

/**
 * A labelled integer field bound to a value + setter, clamped at **both** ends
 * (round-2 finding R2.F6).
 *
 * This clamped only the minimum. `<input type=number>` accepts `1e999`, so
 * `Number(...)` was `Infinity`, `Math.max(1, Infinity)` was `Infinity`, and
 * `resolveSlices` looped `for (c = 0; c < Infinity; c++)` pushing objects until
 * the tab died. `100000` was barely better: 10^10 cells, each a heap object and
 * a `strokeRect`, on the UI thread, per keystroke.
 *
 * `MAX_GRID_CELLS` is the real bound and it lives in the store beside the
 * resolver (mirroring Rust's `MAX_GRID_CELLS`); this is the input's own guard,
 * so a value that would be truncated later is never stored in the first place.
 */
function NumField({
  label,
  value,
  min = 0,
  max = MAX_GRID_CELLS,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 text-xs">
      <span className="text-(--ink-text-dim)">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          const raw = Number(e.target.value);
          // A non-finite entry is not a number the author meant; it falls back
          // to the floor rather than through the clamps (`Math.max`/`Math.min`
          // propagate NaN, which is the C4-35 mechanism in JavaScript).
          const n = Number.isFinite(raw) ? Math.floor(raw) : min;
          onChange(Math.min(max, Math.max(min, n)));
        }}
        className="w-16 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 text-right outline-none focus:border-(--ink-accent)"
      />
    </label>
  );
}

/** The texture preview + slice overlay, drawn on a canvas fit to the panel. */
function Preview({
  image,
  slices,
  texW,
  texH,
  selected,
  onPick,
}: {
  image: string | null;
  slices: ResolvedSlice[];
  texW: number;
  texH: number;
  selected: string | null;
  onPick: (name: string) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);

  // Decode the preview data-URL into an <img> once per source. (No preview →
  // the canvas isn't rendered at all, so a stale bitmap is never shown.)
  useEffect(() => {
    if (!image) return;
    const el = new Image();
    el.onload = () => setImg(el);
    el.src = image;
    return () => {
      el.onload = null;
    };
  }, [image]);

  // The layout box the atlas is drawn into (fit, preserving aspect).
  const box = useMemo(() => {
    const maxW = 460;
    const maxH = 460;
    const scale = Math.min(maxW / Math.max(1, texW), maxH / Math.max(1, texH), 4);
    return { w: Math.max(1, texW * scale), h: Math.max(1, texH * scale) };
  }, [texW, texH]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(box.w * dpr);
    canvas.height = Math.round(box.h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, box.w, box.h);

    // Checkerboard so transparent atlases stay legible.
    const cell = 8;
    for (let y = 0; y < box.h; y += cell) {
      for (let x = 0; x < box.w; x += cell) {
        ctx.fillStyle = ((x / cell + y / cell) & 1) === 0 ? "#2a2a2f" : "#232327";
        ctx.fillRect(x, y, cell, cell);
      }
    }
    if (img) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, box.w, box.h);
    }

    // Slice overlay (UV → box pixels).
    for (const s of slices) {
      const x = s.uvMin[0] * box.w;
      const y = s.uvMin[1] * box.h;
      const w = (s.uvMax[0] - s.uvMin[0]) * box.w;
      const h = (s.uvMax[1] - s.uvMin[1]) * box.h;
      const on = s.name === selected;
      ctx.lineWidth = on ? 2 : 1;
      ctx.strokeStyle = on ? "#ffb454" : "rgba(120,170,255,0.85)";
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      if (on) {
        ctx.fillStyle = "rgba(255,180,84,0.18)";
        ctx.fillRect(x, y, w, h);
      }
    }
  }, [img, slices, box, selected]);

  const pick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const u = (e.clientX - rect.left) / rect.width;
    const v = (e.clientY - rect.top) / rect.height;
    // Topmost slice under the cursor (manual rects sit above grid cells).
    for (let i = slices.length - 1; i >= 0; i--) {
      const s = slices[i];
      if (u >= s.uvMin[0] && u <= s.uvMax[0] && v >= s.uvMin[1] && v <= s.uvMax[1]) {
        onPick(s.name);
        return;
      }
    }
  };

  return (
    <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto bg-(--ink-bg-0) p-3">
      {image ? (
        <canvas
          ref={canvasRef}
          onClick={pick}
          style={{ width: box.w, height: box.h }}
          className="cursor-crosshair rounded border border-(--ink-border)"
        />
      ) : (
        <div className="text-sm text-(--ink-text-dim)">
          No preview available for this texture.
        </div>
      )}
    </div>
  );
}

export default function SpriteSheetPanel() {
  const doc = useSpriteSheetStore((s) => s.doc);
  const image = useSpriteSheetStore((s) => s.image);
  const dirty = useSpriteSheetStore((s) => s.dirty);
  const busy = useSpriteSheetStore((s) => s.busy);
  const error = useSpriteSheetStore((s) => s.error);
  const setGrid = useSpriteSheetStore((s) => s.setGrid);
  const toggleGrid = useSpriteSheetStore((s) => s.toggleGrid);
  const addManual = useSpriteSheetStore((s) => s.addManual);
  const updateManual = useSpriteSheetStore((s) => s.updateManual);
  const removeManual = useSpriteSheetStore((s) => s.removeManual);
  const save = useSpriteSheetStore((s) => s.save);
  const applySlice = useSpriteSheetStore((s) => s.applySlice);
  const pushStatus = useShellStore((s) => s.pushStatus);

  const [selected, setSelected] = useState<string | null>(null);
  const [resize, setResize] = useState(false);

  const slices = useMemo(() => (doc ? resolveSlices(doc) : []), [doc]);

  if (!doc) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center p-4 text-sm text-(--ink-text-dim)">
        Double-click a texture in the Content Drawer to slice it.
      </div>
    );
  }

  const grid = doc.grid;
  const apply = async () => {
    if (!selected) return;
    const n = await applySlice(selected, resize);
    pushStatus(
      n > 0
        ? `Applied slice “${selected}” to ${n} ${n === 1 ? "entity" : "entities"}.`
        : "Select an actor first, then apply the slice.",
    );
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Toolbar */}
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-(--ink-border) bg-(--ink-bg-2) px-2 text-xs">
        <span className="font-semibold">Sprite Sheet</span>
        <span className="text-(--ink-text-faint)">
          {doc.tex_width}×{doc.tex_height}px · {slices.length} slices
        </span>
        <div className="flex-1" />
        <button
          onClick={() => void save()}
          disabled={busy || !dirty}
          className="flex h-6 items-center gap-1 rounded bg-(--ink-accent) px-2 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
        >
          <Save size={12} /> Save
        </button>
      </div>

      {error && (
        <div className="shrink-0 border-b border-(--ink-border) bg-(--ink-error)/10 px-2 py-1 text-xs text-(--ink-error)">
          {error}
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <Preview
          image={image}
          slices={slices}
          texW={doc.tex_width}
          texH={doc.tex_height}
          selected={selected}
          onPick={setSelected}
        />

        {/* Controls column */}
        <div className="flex w-72 shrink-0 flex-col gap-3 overflow-auto border-l border-(--ink-border) bg-(--ink-bg-1) p-3">
          {/* Grid section */}
          <section className="flex flex-col gap-2">
            <label className="flex items-center gap-2 text-xs font-semibold">
              <input
                type="checkbox"
                checked={!!grid}
                onChange={(e) => toggleGrid(e.target.checked)}
              />
              Grid Slicing
            </label>
            {grid && (
              <div className="flex flex-col gap-1.5 pl-1">
                <NumField label="Columns" min={1} value={grid.columns} onChange={(v) => setGrid({ columns: v })} />
                <NumField label="Rows" min={1} value={grid.rows} onChange={(v) => setGrid({ rows: v })} />
                <NumField label="Margin X" value={grid.margin_x} onChange={(v) => setGrid({ margin_x: v })} />
                <NumField label="Margin Y" value={grid.margin_y} onChange={(v) => setGrid({ margin_y: v })} />
                <NumField label="Padding X" value={grid.padding_x} onChange={(v) => setGrid({ padding_x: v })} />
                <NumField label="Padding Y" value={grid.padding_y} onChange={(v) => setGrid({ padding_y: v })} />
              </div>
            )}
          </section>

          {/* Manual rects */}
          <section className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold">Manual Slices</span>
              <button
                onClick={addManual}
                title="Add a manual slice"
                className="flex h-5 items-center gap-0.5 rounded border border-(--ink-border) px-1 text-xs hover:border-(--ink-accent)"
              >
                <Plus size={11} /> Add
              </button>
            </div>
            {doc.manual.map((m, i) => (
              <div
                key={i}
                className={`flex flex-col gap-1 rounded border p-1.5 ${
                  m.name === selected ? "border-(--ink-accent)" : "border-(--ink-border)"
                }`}
                onClick={() => setSelected(m.name)}
              >
                <div className="flex items-center gap-1">
                  <input
                    value={m.name}
                    onChange={(e) => updateManual(i, { name: e.target.value })}
                    className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 py-0.5 text-xs outline-none focus:border-(--ink-accent)"
                  />
                  <button
                    aria-label="Remove slice"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeManual(i);
                    }}
                    className="rounded p-0.5 text-(--ink-text-faint) hover:bg-(--ink-error)/15 hover:text-(--ink-error)"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-1">
                  <NumField label="X" value={m.x} onChange={(v) => updateManual(i, { x: v })} />
                  <NumField label="Y" value={m.y} onChange={(v) => updateManual(i, { y: v })} />
                  <NumField label="W" min={1} value={m.width} onChange={(v) => updateManual(i, { width: v })} />
                  <NumField label="H" min={1} value={m.height} onChange={(v) => updateManual(i, { height: v })} />
                </div>
              </div>
            ))}
            {doc.manual.length === 0 && (
              <div className="text-xs text-(--ink-text-faint)">
                No manual slices — use the grid, or add one.
              </div>
            )}
          </section>

          {/* Apply */}
          <section className="mt-auto flex flex-col gap-1.5 border-t border-(--ink-border) pt-2">
            <div className="text-xs text-(--ink-text-dim)">
              Selected slice: <span className="text-(--ink-text)">{selected ?? "—"}</span>
            </div>
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={resize} onChange={(e) => setResize(e.target.checked)} />
              Resize sprite to slice aspect
            </label>
            <button
              onClick={() => void apply()}
              disabled={!selected}
              className="rounded bg-(--ink-accent) px-2 py-1 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
            >
              Apply to Selection
            </button>
          </section>
        </div>
      </div>
    </div>
  );
}
