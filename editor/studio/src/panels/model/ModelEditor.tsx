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
import type { DccDragDto } from "../../bindings/DccDragDto";
import type { DccGizmoModeDto } from "../../bindings/DccGizmoModeDto";
import type { DccModeDto } from "../../bindings/DccModeDto";
import type { DccSculptModeDto } from "../../bindings/DccSculptModeDto";
import type { SculptFalloffDto } from "../../bindings/SculptFalloffDto";
import { cn } from "../../lib/utils";

/**
 * Which gesture the left pointer button makes on the preview.
 *
 * `select` is the P23.4 behaviour (click picks, drag orbits) and stays the
 * default: a modeller whose click did something unexpected is a modeller who
 * stops trusting the panel. `sculpt` and `gizmo` are the P23.5 additions, and
 * both of them **fall back to orbit when the pointer misses** — a stroke that
 * starts off the silhouette is a camera move, so neither tool costs the author
 * their navigation.
 */
type PointerTool = "select" | "sculpt" | "gizmo";

/**
 * The smallest brush radius the backend accepts, metres — mirrored from
 * `inf_dcc::MIN_BRUSH_RADIUS_M`.
 *
 * A mirror rather than a fetched value because it is a *constant of the tool*,
 * and the backend refuses below it regardless: this only stops the number box
 * offering an author a radius that will be turned down.
 */
const MIN_BRUSH_RADIUS_M = 1e-3;

/** What the pointer is doing between down and up. */
interface DragState {
  x: number;
  y: number;
  /** The preview-space pixel the gesture started on, for a click that picks. */
  downX: number;
  downY: number;
  moved: boolean;
  /**
   * `pending` while `dragBegin` is in flight — the backend has not said yet
   * whether the pointer grabbed anything, and moves during that window are
   * dropped rather than guessed at.
   */
  kind: "pending" | "drag" | "orbit";
  /** The pointer came up before `dragBegin` resolved. */
  released: boolean;
  shift: boolean;
}

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
  const { doc, image, previewError, refusal, lastSave, lastUnwrap, uvImage, status, busy } =
    useDccEntry(assetId);
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
  const setGizmo = useDccStore((s) => s.setGizmo);
  const dragBegin = useDccStore((s) => s.dragBegin);
  const dragMove = useDccStore((s) => s.dragMove);
  const dragEnd = useDccStore((s) => s.dragEnd);
  const dragCancel = useDccStore((s) => s.dragCancel);
  const unwrap = useDccStore((s) => s.unwrap);
  const uvRefresh = useDccStore((s) => s.uvRefresh);
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

  // ── P23.5 tool state ─────────────────────────────────────────────────────
  const [tool, setTool] = useState<PointerTool>("select");
  const [sculptMode, setSculptMode] = useState<DccSculptModeDto>("draw");
  const [brushRadius, setBrushRadius] = useState(0.3);
  const [brushStrength, setBrushStrength] = useState(0.05);
  const [falloff, setFalloff] = useState<SculptFalloffDto>("Smooth");
  const [gizmoMode, setGizmoMode] = useState<DccGizmoModeDto>("translate");
  const [snap, setSnap] = useState(0);
  const [softRadius, setSoftRadius] = useState(0);
  /** The 2D UV view, shown beside the 3D one rather than instead of it. */
  const [showUv, setShowUv] = useState(false);

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
  // **The handles are drawn backend-side**, so arming the gizmo is a call, not a
  // local flag. Sent whenever the tool or its mode changes — including on the
  // way out of the tool, which is what makes the handles disappear rather than
  // linger over a select-mode click.
  useEffect(() => {
    if (!assetId || !doc) return;
    const want = tool === "gizmo" ? gizmoMode : null;
    if (doc.gizmo !== want) void setGizmo(assetId, want);
  }, [assetId, doc, tool, gizmoMode, setGizmo]);

  // **The UV frame follows the journal.** Keyed on the generation stamp, which
  // is the one thing that moves when the mesh does — a `dirty` flag would miss an
  // undo, and refreshing on every render would re-render on every mouse move.
  // Keyed on the journal stamp AND the selection revision: a pick does not move
  // the journal, and `selected` is a count that reads `1` for two different
  // one-face selections. The store re-renders on both too — this effect is what
  // fills the pane the first time it is opened.
  const generation = doc?.generation;
  const selectionRev = doc?.selectionRev;
  useEffect(() => {
    if (assetId && showUv && generation !== undefined) void uvRefresh(assetId);
  }, [assetId, showUv, generation, selectionRev, uvRefresh]);

  // ── the preview surface ──────────────────────────────────────────────────
  const imgRef = useRef<HTMLImageElement | null>(null);
  const drag = useRef<DragState | null>(null);

  /** Pointer position in the PREVIEW's own pixel space (what `dcc_pick` wants). */
  const toPreviewPx = useCallback((e: React.PointerEvent): [number, number] => {
    const el = imgRef.current;
    if (!el) return [0, 0];
    const r = el.getBoundingClientRect();
    const sx = DCC_PREVIEW_SIZE / Math.max(1, r.width);
    const sy = DCC_PREVIEW_SIZE / Math.max(1, r.height);
    return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
  }, []);

  /**
   * The drag this pointer-down would start, or `null` for the select tool.
   *
   * Built here rather than in the store because the parameters are the panel's
   * popover state and nothing outside this component has an opinion about them —
   * the same rule the P23.4 tool parameters already follow.
   */
  const dragRequest = (): DccDragDto | null => {
    if (tool === "sculpt") {
      return {
        kind: "sculpt",
        mode: sculptMode,
        radius: brushRadius,
        strength: brushStrength,
        falloff,
      };
    }
    if (tool === "gizmo") {
      return { kind: "gizmo", mode: gizmoMode, snap, softRadius, falloff };
    }
    return null;
  };

  const onPointerDown = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const [px, py] = toPreviewPx(e);
    const d: DragState = {
      x: e.clientX,
      y: e.clientY,
      downX: px,
      downY: py,
      moved: false,
      kind: "orbit",
      released: false,
      shift: e.shiftKey || e.ctrlKey,
    };
    drag.current = d;
    const request = assetId ? dragRequest() : null;
    if (!assetId || !request) return;

    // The backend decides whether the pointer grabbed anything, because only it
    // knows where the surface and the handles are. Until it answers, the gesture
    // is `pending` and its moves are dropped — a few milliseconds of path, which
    // is cheaper than guessing wrong and orbiting the camera through a stroke.
    d.kind = "pending";
    void dragBegin(assetId, request, px, py).then((grabbed) => {
      const settle = () => {
        if (grabbed) void dragEnd(assetId);
        else if (!d.moved) void pick(assetId, d.downX, d.downY, d.shift);
      };
      if (drag.current !== d) {
        // The gesture is already over (or another has started): finish this one
        // so the backend is never left holding a drag nobody will end. The
        // backend settles orphans anyway — this is the panel doing its half.
        settle();
        return;
      }
      d.kind = grabbed ? "drag" : "orbit";
      if (d.released) {
        drag.current = null;
        settle();
      }
    });
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
    if (!assetId) return;
    if (d.kind === "drag") {
      const [px, py] = toPreviewPx(e);
      void dragMove(assetId, px, py);
    } else if (d.kind === "orbit") {
      void orbit(assetId, -dx * 0.4, dy * 0.4, 0);
    }
    // `pending`: the backend has not answered yet, so there is nothing to move.
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === "pending") {
      // Let the `dragBegin` resolver finish the gesture — it is the only place
      // that knows whether there is a drag to end.
      d.released = true;
      return;
    }
    drag.current = null;
    if (!assetId) return;
    if (d.kind === "drag") {
      void dragEnd(assetId);
      return;
    }
    // A drag orbits; a click selects. Separated by the same 3 px threshold, so a
    // slightly shaky click still selects rather than nudging the camera.
    if (!d.moved) {
      const [x, y] = toPreviewPx(e);
      void pick(assetId, x, y, e.shiftKey || e.ctrlKey);
    }
  };
  const onWheel = (e: React.WheelEvent) => {
    if (assetId) void orbit(assetId, 0, 0, e.deltaY > 0 ? 0.12 : -0.12);
  };
  /** Escape throws the drag away — the author's explicit "no". */
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape" && assetId && doc?.dragging) {
      e.preventDefault();
      drag.current = null;
      void dragCancel(assetId);
    }
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
              className={cn(
                "rounded p-1 hover:bg-(--ink-bg-3)",
                showUv && "bg-(--ink-accent) text-(--ink-text-onaccent)",
              )}
              onClick={() => setShowUv((v) => !v)}
              title="Show the 2D UV layout (the selection is shared with the 3D view)"
            >
              <Grid2x2 size={13} />
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
              className={cn(
                "max-h-full max-w-full select-none rounded outline-none",
                tool === "sculpt" ? "cursor-cell" : "cursor-crosshair",
              )}
              style={{ imageRendering: "pixelated" }}
              tabIndex={0}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onWheel={onWheel}
              onKeyDown={onKeyDown}
            />
          ) : (
            <div className="text-(--ink-text-dim)">{previewError ?? "Rendering…"}</div>
          )}
        </div>
        {showUv && (
          <div className="flex h-48 shrink-0 items-center justify-center border-t border-(--ink-border) bg-(--ink-bg-0) p-2">
            {uvImage ? (
              <img
                src={uvImage}
                alt="uv layout"
                draggable={false}
                className="max-h-full max-w-full select-none rounded"
                style={{ imageRendering: "pixelated" }}
              />
            ) : (
              <span className="text-[11px] text-(--ink-text-dim)">
                Mark seams in Edge mode, then Unwrap.
              </span>
            )}
          </div>
        )}
        <div className="flex h-6 shrink-0 items-center gap-3 border-t border-(--ink-border) bg-(--ink-bg-2) px-2 text-[11px] text-(--ink-text-dim)">
          <span>{doc.verts} v</span>
          <span>{doc.edges} e</span>
          <span>{doc.faces} f</span>
          <span className="text-(--ink-text)">{doc.selected} selected</span>
          {doc.knifePoints > 1 && <span>knife: {doc.knifePoints} points</span>}
          {doc.dragging && (
            <span className="text-(--ink-accent)">
              {doc.dragPoints > 0 ? `stroke: ${doc.dragPoints} points` : "dragging"} · Esc cancels
            </span>
          )}
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

        <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
          POINTER
        </div>
        <div className="flex gap-1">
          {(["select", "sculpt", "gizmo"] as PointerTool[]).map((t) => (
            <button
              key={t}
              className={cn(
                "flex flex-1 items-center justify-center gap-1 rounded px-2 py-1 text-[11px] capitalize",
                tool === t
                  ? "bg-(--ink-accent) text-(--ink-text-onaccent)"
                  : "bg-(--ink-bg-2) hover:bg-(--ink-bg-3)",
              )}
              onClick={() => setTool(t)}
              title={
                t === "select"
                  ? "Click picks, drag orbits"
                  : t === "sculpt"
                    ? "Drag on the surface to paint. One stroke = one undo step."
                    : "Drag a handle on the selection. Off the handles, the drag orbits."
              }
            >
              {t}
            </button>
          ))}
        </div>

        {tool === "sculpt" && (
          <>
            <div className="flex gap-1">
              {(["draw", "smooth", "flatten", "grab"] as DccSculptModeDto[]).map((m) => (
                <button
                  key={m}
                  className={cn(
                    "flex-1 rounded px-1 py-1 text-[10px] capitalize",
                    sculptMode === m
                      ? "bg-(--ink-accent) text-(--ink-text-onaccent)"
                      : "bg-(--ink-bg-2) hover:bg-(--ink-bg-3)",
                  )}
                  onClick={() => setSculptMode(m)}
                >
                  {m}
                </button>
              ))}
            </div>
            {/* Clamped at the input to the backend's own floor, so the common
                case is a number box that will not go below it rather than a
                refusal after the fact. The backend still refuses — this is the
                affordance, not the guard. */}
            <Num
              label="Radius (m)"
              value={brushRadius}
              onChange={(v) => setBrushRadius(Math.max(MIN_BRUSH_RADIUS_M, v))}
            />
            <Num label="Strength" value={brushStrength} step={0.01} onChange={setBrushStrength} />
            <p className="text-[10px] leading-snug text-(--ink-text-dim)">
              Radius is <b>geodesic</b> metres — measured across the surface, so the brush
              does not reach through a thin wall. Strength is metres at full weight for
              draw, a blend fraction for smooth and flatten, and a multiplier on the drag
              for grab.
            </p>
          </>
        )}

        {tool === "gizmo" && (
          <>
            <div className="flex gap-1">
              {(["translate", "rotate", "scale"] as DccGizmoModeDto[]).map((m) => (
                <button
                  key={m}
                  className={cn(
                    "flex-1 rounded px-1 py-1 text-[10px] capitalize",
                    gizmoMode === m
                      ? "bg-(--ink-accent) text-(--ink-text-onaccent)"
                      : "bg-(--ink-bg-2) hover:bg-(--ink-bg-3)",
                  )}
                  onClick={() => setGizmoMode(m)}
                >
                  {m}
                </button>
              ))}
            </div>
            <Num label="Snap" value={snap} step={0.05} onChange={(v) => setSnap(Math.max(0, v))} />
            <Num
              label="Soft radius (m)"
              value={softRadius}
              onChange={(v) => setSoftRadius(Math.max(0, v))}
            />
            <p className="text-[10px] leading-snug text-(--ink-text-dim)">
              The handles sit on the selection&rsquo;s centroid. Snap 0 is off; a soft radius
              above 0 blends the move into the geodesic neighbourhood. The Translate box
              below journals the identical op.
            </p>
          </>
        )}

        {(tool === "sculpt" || tool === "gizmo") && (
          <label className="flex items-center justify-between gap-2 text-[11px]">
            <span className="text-(--ink-text-dim)">Falloff</span>
            <select
              value={falloff}
              onChange={(e) => setFalloff(e.target.value as SculptFalloffDto)}
              className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 py-0.5 text-[11px]"
            >
              <option value="Smooth">Smooth</option>
              <option value="Sphere">Sphere</option>
              <option value="Linear">Linear</option>
              <option value="Sharp">Sharp</option>
            </select>
          </label>
        )}

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

        <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">UV</div>
        <div className="grid grid-cols-2 gap-1">
          <ToolButton
            label="Mark seam"
            icon={<Scissors size={12} />}
            disabled={nothing || mode !== "edge"}
            onClick={() => assetId && void apply(assetId, { tool: "seam", seam: true })}
            title="Cut the selected edges. Charts are the components the cuts leave behind."
          />
          <ToolButton
            label="Clear seam"
            icon={<Scissors size={12} />}
            disabled={nothing || mode !== "edge"}
            onClick={() => assetId && void apply(assetId, { tool: "seam", seam: false })}
          />
        </div>
        <ToolButton
          label="Unwrap"
          icon={<Grid2x2 size={12} />}
          disabled={busy}
          onClick={() => {
            if (!assetId) return;
            setShowUv(true);
            void unwrap(assetId);
          }}
          title="Least-squares conformal unwrap, one chart per seam-cut component, packed into 0..1."
        />
        <div className="flex gap-2 text-[11px] text-(--ink-text-dim)">
          <span>{doc.seams} seams</span>
          <span>{doc.charts} charts</span>
        </div>
        {lastUnwrap && (
          <>
            <Verdict
              label="Charts"
              value={lastUnwrap.charts}
              good={lastUnwrap.ok}
            />
            {/* Three numbers, three questions. Distortion is a property of the
                shape, convergence of the solve, and folds of neither — a chart
                can converge perfectly and still overlap itself. One number gave
                the same advice for all three. */}
            <Verdict
              label="Stretch"
              value={lastUnwrap.worstResidual.toExponential(1)}
              good={lastUnwrap.worstResidual < 1e-4}
            />
            <Verdict
              label="Converged"
              value={lastUnwrap.worstConvergence.toExponential(1)}
              good={lastUnwrap.worstConvergence < 1e-6}
            />
            <Verdict
              label="Folded tris"
              value={`${lastUnwrap.flipped} / ${lastUnwrap.triangles}`}
              good={lastUnwrap.flipped === 0}
            />
            {lastUnwrap.worstConvergence >= 1e-6 && (
              <p className="rounded bg-(--ink-warn-bg,#3a2a12) p-1 text-[10px] leading-snug text-(--ink-warn,#ffb454)">
                The solver did not finish on this mesh. The layout below is not the
                answer it was converging to — treat it as provisional and report the
                mesh.
              </p>
            )}
            {lastUnwrap.flipped > 0 && (
              <p className="rounded bg-(--ink-warn-bg,#3a2a12) p-1 text-[10px] leading-snug text-(--ink-warn,#ffb454)">
                {lastUnwrap.flipped} triangles are folded over their neighbours: part
                of this model <b>cannot be flattened</b> without a cut. A tube or a
                closed shell has no flat form at all — mark a seam along it and unwrap
                again.
              </p>
            )}
            {lastUnwrap.worstResidual >= 1e-4 && lastUnwrap.flipped === 0 && (
              <p className="text-[10px] leading-snug text-(--ink-text-dim)">
                The layout is stretched but not folded: the shape is curved, so a
                conformal map has to distort it. Another seam through the stretched
                area reduces it.
              </p>
            )}
            {lastUnwrap.refusal && (
              <p className="rounded bg-(--ink-warn-bg,#3a2a12) p-1 text-[10px] leading-snug text-(--ink-warn,#ffb454)">
                {lastUnwrap.refusal}
              </p>
            )}
          </>
        )}

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
