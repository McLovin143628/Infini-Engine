/**
 * **The Model Editor** (P23.4) — the embedded DCC's first visible surface.
 *
 * Left: the offscreen-rendered preview as a `<img>`, orbited with the mouse and
 * clicked to select. This is a normal DOM panel, not the native viewport, so
 * there is no airspace concern — and per the P23.1 memo the 3D view is the
 * `PreviewSession` offscreen path (0.34 ms warm at 512², measured) rather than a
 * second `EngineHost`.
 *
 * Right: the mode switch, the tools with their parameters, the selection
 * commands, and the readouts — counts, the reader's verdict on how the asset
 * arrived, and the writer's verdict on the last save.
 *
 * # The panel is a thin client, on purpose
 *
 * It holds no mesh, no selection and no camera. All three live in the backend
 * document, because all three are answers the *kernel's generation stamp* has to
 * arbitrate: ids are arena slots, a structural op renumbers them, and only the
 * side that can compare stamps may decide whether a selection survived. Every
 * call here returns the document and the store replaces its state with it.
 *
 * # Picking is CPU, deliberately
 *
 * The viewport's ID buffer has no sub-object path and the P23.1 memo rules it out
 * of this one. A pointer position goes to `dcc_pick`, which projects with the
 * *same* `Projector` the overlay is drawn with — so what lights up is what was
 * clicked, by construction rather than by agreement.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Box,
  ChevronsLeftRight,
  CircleDot,
  Crosshair,
  Expand,
  Frame,
  Grid2x2,
  Layers,
  Merge,
  Redo2,
  Save,
  Scissors,
  Shrink,
  Slice,
  Spline,
  Square,
  Triangle,
  Undo2,
} from "lucide-react";

import { DCC_PREVIEW_SIZE } from "../../lib/ipc";
import { useAssetStore } from "../../stores/assetStore";
import { useDccEntry, useDccStore } from "../../stores/dccStore";
import type { DccModeDto } from "../../bindings/DccModeDto";
import { cn } from "../../lib/utils";

/** A small labelled number input. */
function Num({
  label,
  value,
  step = 0.05,
  onChange,
}: {
  label: string;
  value: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 text-[11px]">
      <span className="text-(--ink-text-dim)">{label}</span>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value) || 0)}
        className="w-16 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 text-right outline-none focus:border-(--ink-accent)"
      />
    </label>
  );
}

function ToolButton({
  label,
  icon,
  onClick,
  disabled,
  title,
}: {
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      className="flex w-full items-center gap-2 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 text-left text-[11px] hover:bg-(--ink-bg-3) disabled:opacity-40"
      onClick={onClick}
      disabled={disabled}
      title={title ?? label}
    >
      {icon}
      {label}
    </button>
  );
}

/** A verdict chip: green when the counter is zero, amber when it is not. */
function Verdict({ label, value, good }: { label: string; value: number | string; good: boolean }) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-2 rounded px-1.5 py-0.5 text-[11px]",
        good ? "text-(--ink-text-dim)" : "bg-(--ink-warn-bg,#3a2a12) text-(--ink-warn,#ffb454)",
      )}
    >
      <span>{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}

export default function ModelEditor({ params }: { panelId: string; params: string | null }) {
  // **Every read and every write is keyed by this panel's own asset id.** The
  // dock keeps inactive tabs MOUNTED, so two Model Editors render at the same
  // time; a store read that did not name the asset gave both of them whichever
  // document was opened last, and every tool press went to the wrong mesh.
  const assetId = params;
  const { doc, image, previewError, refusal, lastSave, status, busy } = useDccEntry(assetId);
  const open = useDccStore((s) => s.open);
  const close = useDccStore((s) => s.close);
  const apply = useDccStore((s) => s.apply);
  const select = useDccStore((s) => s.select);
  const setMode = useDccStore((s) => s.setMode);
  const pick = useDccStore((s) => s.pick);
  const orbit = useDccStore((s) => s.orbit);
  const frame = useDccStore((s) => s.frame);
  const save = useDccStore((s) => s.save);
  const mergeAsset = useDccStore((s) => s.mergeAsset);
  const assetsById = useAssetStore((s) => s.assets);

  // Tool parameters. Local, because they are the popover's state and nothing
  // outside this panel has an opinion about them.
  const [distance, setDistance] = useState(0.25);
  const [inset, setInset] = useState(0.1);
  const [bevel, setBevel] = useState(0.05);
  const [cuts, setCuts] = useState(1);
  const [mirrorAxis, setMirrorAxis] = useState("x");
  const [individual, setIndividual] = useState(false);
  const [dropTarget, setDropTarget] = useState(false);

  // The panel instance is `model:<assetId>` — a singleton per asset, like the
  // material editor's per-graph tab.
  useEffect(() => {
    if (assetId) void open(assetId);
  }, [assetId, open]);
  // Closing frees THIS document's backend session and journal. Reading the id
  // from the closure rather than from the store is what stops a panel unmount
  // closing its neighbour.
  useEffect(() => {
    if (!assetId) return;
    return () => void close(assetId);
  }, [assetId, close]);

  // ── the preview surface ──────────────────────────────────────────────────
  const imgRef = useRef<HTMLImageElement | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  /** Pointer position in the PREVIEW's own pixel space (what `dcc_pick` wants). */
  const toPreviewPx = useCallback((e: React.PointerEvent): [number, number] => {
    const el = imgRef.current;
    if (!el) return [0, 0];
    const r = el.getBoundingClientRect();
    const sx = DCC_PREVIEW_SIZE / Math.max(1, r.width);
    const sy = DCC_PREVIEW_SIZE / Math.max(1, r.height);
    return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.abs(dx) + Math.abs(dy) < 3) return;
    d.moved = true;
    d.x = e.clientX;
    d.y = e.clientY;
    if (assetId) void orbit(assetId, -dx * 0.4, dy * 0.4, 0);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    // A drag orbits; a click selects. Separated by the same 3 px threshold, so a
    // slightly shaky click still selects rather than nudging the camera.
    if (d && !d.moved) {
      const [x, y] = toPreviewPx(e);
      if (assetId) void pick(assetId, x, y, e.shiftKey || e.ctrlKey);
    }
  };
  const onWheel = (e: React.WheelEvent) => {
    if (assetId) void orbit(assetId, 0, 0, e.deltaY > 0 ? 0.12 : -0.12);
  };

  // ── drag-and-drop: a mesh asset dropped on the panel merges in ───────────
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDropTarget(false);
    const id = e.dataTransfer.getData("application/x-inf-asset") || e.dataTransfer.getData("text/plain");
    if (!id) return;
    if (assetsById[id]?.kind !== "mesh") return;
    if (assetId) void mergeAsset(assetId, id);
  };

  if (!doc) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-xs text-(--ink-text-dim)">
        {status ?? (assetId ? "Opening…" : "Open a mesh asset from the Content Drawer to edit it.")}
      </div>
    );
  }

  const mode = doc.mode;
  const modeButton = (m: DccModeDto, icon: React.ReactNode, label: string) => (
    <button
      key={m}
      className={cn(
        "flex flex-1 items-center justify-center gap-1 rounded px-2 py-1 text-[11px]",
        mode === m
          ? "bg-(--ink-accent) text-(--ink-text-onaccent)"
          : "bg-(--ink-bg-2) hover:bg-(--ink-bg-3)",
      )}
      onClick={() => assetId && void setMode(assetId, m)}
      title={`${label} mode`}
    >
      {icon}
      {label}
    </button>
  );

  const nothing = doc.selected === 0;
  const imp = doc.import;

  return (
    <div className="flex h-full min-h-0 text-xs">
      {/* ── preview ───────────────────────────────────────────────────── */}
      <div className="flex min-w-0 flex-1 flex-col border-r border-(--ink-border)">
        <div className="flex h-8 shrink-0 items-center gap-2 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
          <span className="truncate font-medium">{doc.name}</span>
          {doc.dirty && <span className="text-(--ink-accent)">●</span>}
          <div className="ml-auto flex items-center gap-1">
            <button
              className="rounded p-1 hover:bg-(--ink-bg-3) disabled:opacity-40"
              onClick={() => assetId && void useDccStore.getState().undo(assetId)}
              disabled={!doc.canUndo}
              title="Undo (Ctrl+Z)"
            >
              <Undo2 size={13} />
            </button>
            <button
              className="rounded p-1 hover:bg-(--ink-bg-3) disabled:opacity-40"
              onClick={() => assetId && void useDccStore.getState().redo(assetId)}
              disabled={!doc.canRedo}
              title="Redo (Ctrl+Y)"
            >
              <Redo2 size={13} />
            </button>
            <button
              className="rounded p-1 hover:bg-(--ink-bg-3)"
              onClick={() => assetId && void frame(assetId)}
              title="Frame the whole mesh"
            >
              <Frame size={13} />
            </button>
            <button
              className="flex items-center gap-1 rounded bg-(--ink-accent) px-2 py-0.5 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              onClick={() => assetId && void save(assetId)}
              disabled={busy}
              title="Save the mesh asset (rewrites the payload and rebuilds its vmesh)"
            >
              <Save size={12} /> Save
            </button>
          </div>
        </div>
        <div
          className={cn(
            "flex min-h-0 flex-1 items-center justify-center bg-(--ink-bg-0) p-3",
            dropTarget && "outline outline-2 -outline-offset-4 outline-(--ink-accent)",
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setDropTarget(true);
          }}
          onDragLeave={() => setDropTarget(false)}
          onDrop={onDrop}
        >
          {image ? (
            <img
              ref={imgRef}
              src={image}
              alt="mesh preview"
              draggable={false}
              className="max-h-full max-w-full cursor-crosshair select-none rounded"
              style={{ imageRendering: "pixelated" }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onWheel={onWheel}
            />
          ) : (
            <div className="text-(--ink-text-dim)">{previewError ?? "Rendering…"}</div>
          )}
        </div>
        <div className="flex h-6 shrink-0 items-center gap-3 border-t border-(--ink-border) bg-(--ink-bg-2) px-2 text-[11px] text-(--ink-text-dim)">
          <span>{doc.verts} v</span>
          <span>{doc.edges} e</span>
          <span>{doc.faces} f</span>
          <span className="text-(--ink-text)">{doc.selected} selected</span>
          {doc.knifePoints > 1 && <span>knife: {doc.knifePoints} points</span>}
          {refusal && <span className="ml-auto truncate text-(--ink-warn,#ffb454)">{refusal}</span>}
        </div>
      </div>

      {/* ── tools ─────────────────────────────────────────────────────── */}
      <div className="flex w-64 shrink-0 flex-col gap-2 overflow-y-auto p-2">
        <div className="flex gap-1">
          {modeButton("vert", <CircleDot size={12} />, "Vert")}
          {modeButton("edge", <ChevronsLeftRight size={12} />, "Edge")}
          {modeButton("face", <Square size={12} />, "Face")}
        </div>

        <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">SELECT</div>
        <div className="grid grid-cols-2 gap-1">
          <ToolButton label="All" icon={<Layers size={12} />} onClick={() => assetId && void select(assetId, { action: "all" })} />
          <ToolButton label="None" icon={<Crosshair size={12} />} onClick={() => assetId && void select(assetId, { action: "none" })} />
          <ToolButton label="Invert" icon={<Expand size={12} />} onClick={() => assetId && void select(assetId, { action: "invert" })} />
          <ToolButton label="Linked" icon={<Merge size={12} />} onClick={() => assetId && void select(assetId, { action: "linked" })} />
          <ToolButton label="Grow" icon={<Expand size={12} />} onClick={() => assetId && void select(assetId, { action: "grow" })} />
          <ToolButton label="Shrink" icon={<Shrink size={12} />} onClick={() => assetId && void select(assetId, { action: "shrink" })} />
          <ToolButton
            label="Loop"
            icon={<Spline size={12} />}
            onClick={() => assetId && void select(assetId, { action: "loop" })}
            title="Edge loop through the last edge you clicked"
          />
          <ToolButton
            label="Ring"
            icon={<Spline size={12} />}
            onClick={() => assetId && void select(assetId, { action: "ring" })}
            title="Edge ring through the last edge you clicked"
          />
        </div>

        <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">MODEL</div>
        <Num label="Distance (m)" value={distance} onChange={setDistance} />
        <ToolButton
          label="Extrude"
          icon={<Box size={12} />}
          disabled={nothing || mode !== "face"}
          onClick={() => assetId && void apply(assetId, { tool: "extrude", distance })}
          title="Extrude the selected faces along the region normal. A multi-face selection moves as ONE block: only its border gets walls."
        />
        <Num label="Inset (m)" value={inset} onChange={setInset} />
        <label className="flex items-center gap-2 text-[11px] text-(--ink-text-dim)">
          <input type="checkbox" checked={individual} onChange={(e) => setIndividual(e.target.checked)} />
          Individual faces
        </label>
        <ToolButton
          label="Inset"
          icon={<Square size={12} />}
          disabled={nothing || mode !== "face"}
          onClick={() => assetId && void apply(assetId, { tool: "inset", amount: inset, individual })}
        />
        <Num label="Bevel (m)" value={bevel} onChange={setBevel} />
        <ToolButton
          label="Bevel"
          icon={<Slice size={12} />}
          disabled={nothing || mode !== "edge"}
          onClick={() => assetId && void apply(assetId, { tool: "bevel", amount: bevel })}
          title="One flat chamfer strip per edge (v1 is a single segment)"
        />
        <Num label="Cuts" value={cuts} step={1} onChange={(v) => setCuts(Math.max(1, Math.round(v)))} />
        <ToolButton
          label="Loop Cut"
          icon={<Grid2x2 size={12} />}
          disabled={mode !== "edge"}
          onClick={() => assetId && void apply(assetId, { tool: "loopCut", cuts })}
          title="Cut the quad strip through the last edge you clicked. Refuses on a non-quad."
        />
        <ToolButton
          label="Knife"
          icon={<Scissors size={12} />}
          disabled={mode !== "vert" || doc.knifePoints < 2}
          onClick={() => assetId && void apply(assetId, { tool: "knife" })}
          title="Cut along the vertices in the order you clicked them"
        />
        <ToolButton
          label="Subdivide"
          icon={<Grid2x2 size={12} />}
          disabled={nothing || mode !== "face"}
          onClick={() => assetId && void apply(assetId, { tool: "subdivide" })}
          title="Simple midpoint — density changes, the shape does not"
        />
        <div className="grid grid-cols-2 gap-1">
          <ToolButton
            label="Merge ctr"
            icon={<Merge size={12} />}
            disabled={nothing}
            onClick={() => assetId && void apply(assetId, { tool: "merge", center: true })}
          />
          <ToolButton
            label="Merge last"
            icon={<Merge size={12} />}
            disabled={nothing}
            onClick={() => assetId && void apply(assetId, { tool: "merge", center: false })}
          />
        </div>
        <div className="flex items-center gap-1">
          <select
            value={mirrorAxis}
            onChange={(e) => setMirrorAxis(e.target.value)}
            className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 py-0.5 text-[11px]"
          >
            <option value="x">X</option>
            <option value="y">Y</option>
            <option value="z">Z</option>
          </select>
          <div className="flex-1">
            <ToolButton
              label="Mirror at 0"
              icon={<Triangle size={12} />}
              onClick={() => assetId && void apply(assetId, { tool: "mirror", axis: mirrorAxis, coord: 0 })}
              title="Reflect across the axis plane, welding anything exactly on it"
            />
          </div>
        </div>
        <ToolButton
          label="Delete selection"
          icon={<Crosshair size={12} />}
          disabled={nothing}
          onClick={() => assetId && void apply(assetId, { tool: "delete" })}
        />

        {/* ── readouts ──────────────────────────────────────────────── */}
        <div className="mt-1 text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
          HOW IT OPENED
        </div>
        <Verdict
          label="Boundary edges"
          value={imp.boundaryEdges}
          good={imp.boundaryEdges === 0}
        />
        {imp.boundaryEdges > 0 && (
          <p className="text-[10px] leading-snug text-(--ink-text-dim)">
            This mesh is open. If you expected a closed solid, its seam positions differ
            — the reader welds by exact equality and told you rather than guessing an
            epsilon.
          </p>
        )}
        <Verdict label="Fan splits" value={imp.fanSplits} good={imp.fanSplits === 0} />
        <Verdict
          label="Degenerate tris"
          value={imp.degenerateTrianglesSkipped}
          good={imp.degenerateTrianglesSkipped === 0}
        />
        <Verdict
          label="Non-finite"
          value={imp.nonFiniteValues}
          good={imp.nonFiniteValues === 0}
        />

        {lastSave && (
          <>
            <div className="mt-1 text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
              LAST SAVE
            </div>
            <Verdict label="Triangles" value={lastSave.export.triangles} good />
            <Verdict label="Vertices" value={lastSave.export.vertices} good />
            <Verdict label="vmesh" value={lastSave.vmesh} good={lastSave.vmesh !== "skipped"} />
            {lastSave.advisories.map((a) => (
              <p
                key={a}
                className="rounded bg-(--ink-warn-bg,#3a2a12) p-1 text-[10px] leading-snug text-(--ink-warn,#ffb454)"
              >
                {a}
              </p>
            ))}
          </>
        )}
        {status && <p className="text-[10px] text-(--ink-text-dim)">{status}</p>}
        <p className="mt-auto pt-2 text-[10px] leading-snug text-(--ink-text-dim)">
          Drag a mesh asset onto the preview to merge it in as a second part.
        </p>
      </div>
    </div>
  );
}
