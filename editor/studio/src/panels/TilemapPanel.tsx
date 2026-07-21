/**
 * Tilemap panel (P8.2b): paint the selected Tilemap entity on a pannable,
 * zoomable canvas grid.
 *
 * Selection-aware like Details: it binds to whatever single Tilemap entity is
 * selected (kind "Tilemap") and re-reads its cells after any scene edit
 * (undo/redo, other views) via the scene version bump. Tiles render as colored
 * cells keyed by their atlas index — an atlas-image preview is a follow-up
 * (noted below). Tools: paint / erase / bounded flood-fill / pick; drag-paints,
 * and the whole mouse-down→up stroke commits as one undo step.
 *
 * This is a normal DOM panel (not the native viewport) — no airspace concern.
 */
import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Eraser, Paintbrush, PaintBucket, Pipette } from "lucide-react";

import type { TilemapCellDto } from "../bindings/TilemapCellDto";
import { cn } from "../lib/utils";
import { useSceneStore } from "../stores/sceneStore";
import { type TileTool, useTilemapStore } from "../stores/tilemapStore";

const key = (x: number, y: number) => `${x},${y}`;

/** Deterministic, legible color per 1-based tile index (index→color preview). */
function tileColor(idx: number): string {
  const hue = (idx * 47) % 360;
  const light = 46 + ((idx * 13) % 22);
  return `hsl(${hue} 58% ${light}%)`;
}

const TOOLS: Array<{ id: TileTool; label: string; icon: typeof Paintbrush }> = [
  { id: "paint", label: "Paint (drag to fill)", icon: Paintbrush },
  { id: "erase", label: "Erase", icon: Eraser },
  { id: "fill", label: "Flood Fill (bounded)", icon: PaintBucket },
  { id: "pick", label: "Pick tile", icon: Pipette },
];

export default function TilemapPanel() {
  const selection = useSceneStore((s) => s.selection);
  const nodes = useSceneStore((s) => s.nodes);
  const version = useSceneStore((s) => s.version);

  const dto = useTilemapStore((s) => s.dto);
  const cellMap = useTilemapStore((s) => s.cellMap);
  const tool = useTilemapStore((s) => s.tool);
  const activeTile = useTilemapStore((s) => s.activeTile);
  const boundEntity = useTilemapStore((s) => s.entity);
  const bind = useTilemapStore((s) => s.bind);
  const refresh = useTilemapStore((s) => s.refresh);
  const setTool = useTilemapStore((s) => s.setTool);
  const setActiveTile = useTilemapStore((s) => s.setActiveTile);
  const paintCells = useTilemapStore((s) => s.paintCells);
  const floodFill = useTilemapStore((s) => s.floodFill);

  // The single selected Tilemap entity, if any.
  const target = useMemo(() => {
    if (selection.length !== 1) return null;
    const n = nodes[selection[0]];
    return n && n.kind === "Tilemap" ? n.guid : null;
  }, [selection, nodes]);

  useEffect(() => {
    void bind(target);
  }, [target, bind]);

  // Re-sync painted cells after external edits (undo/redo, gizmo, other panels).
  useEffect(() => {
    if (boundEntity) void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version]);

  if (!target || !dto) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center p-4 text-center text-xs text-(--ink-text-dim)">
        Select a Tilemap entity to paint.
        <br />
        (Outliner → Add → 2D → Tilemap)
      </div>
    );
  }

  const name = nodes[target]?.name ?? "Tilemap";
  const paletteCount = Math.max(1, dto.palette_cols * dto.palette_rows);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Toolbar */}
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-(--ink-border) bg-(--ink-bg-2) px-2 text-xs">
        <span className="font-semibold">Tilemap</span>
        <span className="max-w-40 truncate text-(--ink-text-faint)">{name}</span>
        <div className="mx-1 h-4 w-px bg-(--ink-border)" />
        {TOOLS.map((t) => {
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              title={t.label}
              onClick={() => setTool(t.id)}
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded border",
                tool === t.id
                  ? "border-(--ink-accent) bg-(--ink-accent)/15 text-(--ink-accent)"
                  : "border-(--ink-border) hover:border-(--ink-accent)",
              )}
            >
              <Icon size={13} />
            </button>
          );
        })}
        <div className="flex-1" />
        <span className="text-(--ink-text-faint)">{dto.cells.length} tiles</span>
      </div>

      <div className="flex min-h-0 flex-1">
        <PaintCanvas
          cellMap={cellMap}
          tool={tool}
          activeTile={activeTile}
          onPaint={paintCells}
          onFill={(x, y) => void floodFill(x, y)}
          onPick={(idx) => setActiveTile(idx)}
        />

        {/* Palette column */}
        <div className="flex w-56 shrink-0 flex-col gap-2 overflow-auto border-l border-(--ink-border) bg-(--ink-bg-1) p-2">
          <div className="text-xs font-semibold">Palette</div>
          <div className="text-[11px] text-(--ink-text-faint)">
            {dto.palette_cols}×{dto.palette_rows} atlas · click to select
          </div>
          <div
            className="grid gap-1"
            style={{ gridTemplateColumns: `repeat(${Math.min(dto.palette_cols, 8)}, minmax(0, 1fr))` }}
          >
            {Array.from({ length: paletteCount }, (_, i) => {
              const idx = i + 1;
              const on = idx === activeTile;
              return (
                <button
                  key={idx}
                  title={`Tile ${idx}`}
                  onClick={() => setActiveTile(idx)}
                  className={cn(
                    "flex aspect-square items-center justify-center rounded text-[10px] font-medium text-black/70 outline-none",
                    on ? "ring-2 ring-(--ink-accent)" : "ring-1 ring-(--ink-border)",
                  )}
                  style={{ background: tileColor(idx) }}
                >
                  {idx}
                </button>
              );
            })}
          </div>
          <div className="mt-1 text-[11px] text-(--ink-text-faint)">
            Active tile: <span className="text-(--ink-text)">{activeTile}</span>
          </div>
          <div className="mt-auto border-t border-(--ink-border) pt-2 text-[11px] leading-relaxed text-(--ink-text-faint)">
            Wheel to zoom · middle-drag to pan. Tiles show as colored cells by
            index; atlas-image preview is a follow-up.
          </div>
        </div>
      </div>
    </div>
  );
}

/** The pannable/zoomable paint surface. */
function PaintCanvas({
  cellMap,
  tool,
  activeTile,
  onPaint,
  onFill,
  onPick,
}: {
  cellMap: Map<string, number>;
  tool: TileTool;
  activeTile: number;
  onPaint: (cells: TilemapCellDto[]) => void;
  onFill: (x: number, y: number) => void;
  onPick: (idx: number) => void;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState({ w: 400, h: 300 });
  const [view, setView] = useState({ panX: 200, panY: 150, zoom: 26 });
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);

  // The in-progress stroke (committed as one undo step on pointer-up).
  const pending = useRef<Map<string, number>>(new Map());
  const drag = useRef<null | "stroke" | "pan">(null);
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  // A monotonic tick so mid-stroke stamps (which mutate the `pending` ref, not
  // state) force the draw effect to re-run for instant feedback.
  const [tick, redraw] = useReducer((n: number) => n + 1, 0);

  // Fit the canvas to its container.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const tileAtScreen = (clientX: number, clientY: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    return {
      x: Math.floor((sx - view.panX) / view.zoom),
      y: Math.floor((sy - view.panY) / view.zoom),
    };
  };

  const merged = (x: number, y: number): number => {
    const p = pending.current.get(key(x, y));
    return p !== undefined ? p : (cellMap.get(key(x, y)) ?? 0);
  };

  // Draw.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(size.w * dpr));
    canvas.height = Math.max(1, Math.round(size.h * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#1b1b1f";
    ctx.fillRect(0, 0, size.w, size.h);

    const { panX, panY, zoom } = view;
    const x0 = Math.floor(-panX / zoom) - 1;
    const y0 = Math.floor(-panY / zoom) - 1;
    const x1 = Math.ceil((size.w - panX) / zoom) + 1;
    const y1 = Math.ceil((size.h - panY) / zoom) + 1;

    // Filled cells.
    for (let ty = y0; ty <= y1; ty++) {
      for (let tx = x0; tx <= x1; tx++) {
        const idx = merged(tx, ty);
        if (idx > 0) {
          ctx.fillStyle = tileColor(idx);
          ctx.fillRect(panX + tx * zoom, panY + ty * zoom, zoom, zoom);
        }
      }
    }

    // Grid lines.
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let tx = x0; tx <= x1; tx++) {
      const sx = Math.round(panX + tx * zoom) + 0.5;
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, size.h);
    }
    for (let ty = y0; ty <= y1; ty++) {
      const sy = Math.round(panY + ty * zoom) + 0.5;
      ctx.moveTo(0, sy);
      ctx.lineTo(size.w, sy);
    }
    ctx.stroke();

    // Origin axes.
    ctx.strokeStyle = "rgba(120,170,255,0.5)";
    ctx.beginPath();
    ctx.moveTo(0, Math.round(panY) + 0.5);
    ctx.lineTo(size.w, Math.round(panY) + 0.5);
    ctx.moveTo(Math.round(panX) + 0.5, 0);
    ctx.lineTo(Math.round(panX) + 0.5, size.h);
    ctx.stroke();

    // Hover cell.
    if (hover) {
      ctx.strokeStyle = "#ffb454";
      ctx.lineWidth = 2;
      ctx.strokeRect(panX + hover.x * zoom + 1, panY + hover.y * zoom + 1, zoom - 2, zoom - 2);
    }
  }, [size, view, hover, cellMap, activeTile, tick]); // eslint-disable-line react-hooks/exhaustive-deps

  const stampAt = (x: number, y: number) => {
    if (tool === "erase") pending.current.set(key(x, y), 0);
    else pending.current.set(key(x, y), activeTile);
    redraw();
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    // Middle mouse → pan.
    if (e.button === 1) {
      drag.current = "pan";
      panStart.current = { x: e.clientX, y: e.clientY, panX: view.panX, panY: view.panY };
      e.preventDefault();
      return;
    }
    if (e.button !== 0) return;
    const { x, y } = tileAtScreen(e.clientX, e.clientY);
    if (tool === "pick") {
      const idx = cellMap.get(key(x, y)) ?? 0;
      if (idx > 0) onPick(idx);
      return;
    }
    if (tool === "fill") {
      onFill(x, y);
      return;
    }
    drag.current = "stroke";
    pending.current = new Map();
    stampAt(x, y);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const { x, y } = tileAtScreen(e.clientX, e.clientY);
    setHover((h) => (h && h.x === x && h.y === y ? h : { x, y }));
    if (drag.current === "pan") {
      setView((v) => ({
        ...v,
        panX: panStart.current.panX + (e.clientX - panStart.current.x),
        panY: panStart.current.panY + (e.clientY - panStart.current.y),
      }));
    } else if (drag.current === "stroke") {
      stampAt(x, y);
    }
  };

  const commit = () => {
    if (drag.current === "stroke" && pending.current.size > 0) {
      const cells: TilemapCellDto[] = [];
      for (const [k, tile] of pending.current) {
        const [x, y] = k.split(",").map(Number);
        cells.push({ x, y, tile });
      }
      onPaint(cells);
    }
    pending.current = new Map();
    drag.current = null;
    redraw();
  };

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    setView((v) => {
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      const zoom = Math.max(6, Math.min(96, v.zoom * factor));
      // Keep the tile under the cursor stationary.
      const tx = (sx - v.panX) / v.zoom;
      const ty = (sy - v.panY) / v.zoom;
      return { zoom, panX: sx - tx * zoom, panY: sy - ty * zoom };
    });
  };

  return (
    <div ref={wrapRef} className="relative min-h-0 min-w-0 flex-1 overflow-hidden bg-(--ink-bg-0)">
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h, touchAction: "none" }}
        className={cn(
          tool === "pick" ? "cursor-copy" : tool === "fill" ? "cursor-cell" : "cursor-crosshair",
        )}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={commit}
        onPointerCancel={commit}
        onPointerLeave={() => setHover(null)}
        onWheel={onWheel}
      />
      {hover && (
        <div className="pointer-events-none absolute bottom-1 left-1 rounded bg-(--ink-bg-2)/90 px-1.5 py-0.5 text-[10px] text-(--ink-text-dim)">
          {hover.x}, {hover.y}
        </div>
      )}
    </div>
  );
}
