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
  Circle,
  CircleDot,
  Crosshair,
  Expand,
  Frame,
  Grid2x2,
  Layers,
  Merge,
  Move,
  Redo2,
  Save,
  Scissors,
  Shirt,
  Shrink,
  Slice,
  Sparkles,
  Spline,
  Square,
  Triangle,
  Undo2,
} from "lucide-react";

import {
  ASSET_DROP_ATTR,
  onAssetDrop,
  type AssetDropDetail,
} from "../../lib/assetDrop";
import { DCC_PREVIEW_SIZE, MAX_BEVEL_SEGMENTS } from "../../lib/ipc";
import { useDockLayout } from "../dock/dockLayoutStore";
import { useAssetStore } from "../../stores/assetStore";
import { useDccEntry, useDccStore } from "../../stores/dccStore";
import type { DccDragDto } from "../../bindings/DccDragDto";
import type { DccGizmoModeDto } from "../../bindings/DccGizmoModeDto";
import type { DccModeDto } from "../../bindings/DccModeDto";
import type { DccOrientDto } from "../../bindings/DccOrientDto";
import type { DccPaintModeDto } from "../../bindings/DccPaintModeDto";
import type { DccPivotDto } from "../../bindings/DccPivotDto";
import type { DccPrimitiveDto } from "../../bindings/DccPrimitiveDto";
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
type PointerTool = "select" | "box" | "sculpt" | "weights" | "gizmo";

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
  kind: "pending" | "drag" | "orbit" | "box";
  /** The pointer came up before `dragBegin` resolved. */
  released: boolean;
  shift: boolean;
}

/** The rubber band, in preview-space pixels, while a marquee is being dragged. */
interface Marquee {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
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
  const { doc, image, previewError, refusal, lastSave, lastGroom, lastUnwrap, uvImage, status, busy, history } =
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
  const makeGarment = useDccStore((s) => s.makeGarment);
  const growHair = useDccStore((s) => s.growHair);
  const setGizmo = useDccStore((s) => s.setGizmo);
  const dragBegin = useDccStore((s) => s.dragBegin);
  const dragMove = useDccStore((s) => s.dragMove);
  const dragEnd = useDccStore((s) => s.dragEnd);
  const dragCancel = useDccStore((s) => s.dragCancel);
  const unwrap = useDccStore((s) => s.unwrap);
  const uvRefresh = useDccStore((s) => s.uvRefresh);
  const boxSelect = useDccStore((s) => s.boxSelect);
  const setViewOpts = useDccStore((s) => s.setViewOpts);
  const newMesh = useDccStore((s) => s.newMesh);
  const historyRefresh = useDccStore((s) => s.historyRefresh);
  const amend = useDccStore((s) => s.amend);
  const setHover = useDccStore((s) => s.setHover);
  const uvPick = useDccStore((s) => s.uvPick);
  const uvMove = useDccStore((s) => s.uvMove);
  const assetsById = useAssetStore((s) => s.assets);
  // What the drawer is dragging right now, so the zone lights up for a drop it
  // can actually accept rather than for any native drag that crosses it.
  const dragAsset = useAssetStore((s) => s.dragAsset);

  // Tool parameters. Local, because they are the popover's state and nothing
  // outside this panel has an opinion about them.
  const [distance, setDistance] = useState(0.25);
  const [inset, setInset] = useState(0.1);
  const [bevel, setBevel] = useState(0.05);
  const [bevelSegments, setBevelSegments] = useState(1);
  const [cuts, setCuts] = useState(1);
  const [mirrorAxis, setMirrorAxis] = useState("x");
  const [individual, setIndividual] = useState(false);
  const [dropTarget, setDropTarget] = useState(false);
  // ── Wave D tool parameters ───────────────────────────────────────────────
  const [slide, setSlide] = useState(0.25);
  const [smoothAngle, setSmoothAngle] = useState(30);
  const [seamAngle, setSeamAngle] = useState(40);
  const [showHistory, setShowHistory] = useState(false);
  const [mergeTolerance, setMergeTolerance] = useState(0.001);
  /** The rubber band, while a marquee is being dragged. */
  const [marquee, setMarquee] = useState<Marquee | null>(null);
  /** The numeric transform box — the caller `dcc::transform_ops` never had. */
  const [nudge, setNudge] = useState<[number, number, number]>([0, 0, 0]);
  const [spinAxis, setSpinAxis] = useState<"x" | "y" | "z">("y");
  const [spinDeg, setSpinDeg] = useState(15);
  const [scaleBy, setScaleBy] = useState<[number, number, number]>([1, 1, 1]);
  /** The New Mesh dialog. */
  const [newOpen, setNewOpen] = useState(false);
  const [newPrimitive, setNewPrimitive] = useState<DccPrimitiveDto>("cube");
  const [newSize, setNewSize] = useState(1);
  const [newSegments, setNewSegments] = useState(16);
  const [newRings, setNewRings] = useState(8);
  const [newName, setNewName] = useState("");
  /** The material-slot name being typed. */
  const [slotName, setSlotName] = useState("");

  // ── P23.5 tool state ─────────────────────────────────────────────────────
  const [tool, setTool] = useState<PointerTool>("select");
  const [sculptMode, setSculptMode] = useState<DccSculptModeDto>("draw");
  const [brushRadius, setBrushRadius] = useState(0.3);
  const [brushStrength, setBrushStrength] = useState(0.05);
  const [falloff, setFalloff] = useState<SculptFalloffDto>("Smooth");
  // ── P24.2 weight paint ───────────────────────────────────────────────────
  //
  // The influence is a NUMBER, not a name, and that is the honest surface: a
  // `.inf_mesh` records no skeleton (the mesh-to-rig pairing lives in the
  // scene's `SkeletalMesh`), so the kernel knows how many joints its indices
  // address and not what any of them is called. `skinJoints` is the bound, and
  // names arrive with P24.3's skeleton binding UI.
  const [paintJoint, setPaintJoint] = useState(0);
  const [paintMode, setPaintMode] = useState<DccPaintModeDto>("add");
  const [paintStrength, setPaintStrength] = useState(0.25);
  const [gizmoMode, setGizmoMode] = useState<DccGizmoModeDto>("translate");
  const [snap, setSnap] = useState(0);
  const [softRadius, setSoftRadius] = useState(0);
  /** The 2D UV view, shown beside the 3D one rather than instead of it. */
  const [showUv, setShowUv] = useState(false);

  // ── P24.4 cloth & hair knobs ─────────────────────────────────────────────
  //
  // Local, like every other tool parameter here: they are the section's state,
  // and nothing outside this panel has an opinion about them. The OPERANDS are
  // not here at all — the vertex selection is the garment's pin list and the face
  // selection is the hairstyle's scalp, both resolved backend-side, so there is
  // no second copy of the selection to drift.
  //
  // The defaults mirror `inf_anim`'s shipped materials and
  // `inf_editor_core::groom`'s spec defaults; the units are on the labels
  // because a compliance is m/N and a damping is 1/s, and a number box with no
  // unit is a number box that gets guessed at.
  const [bendCompliance, setBendCompliance] = useState(0.001);
  const [clothDamping, setClothDamping] = useState(0.5);
  const [clothThickness, setClothThickness] = useState(0.005);
  const [clothSubsteps, setClothSubsteps] = useState(8);
  const [bodyRadius, setBodyRadius] = useState(0.08);
  const [rigId, setRigId] = useState("");
  const [hairLength, setHairLength] = useState(0.25);
  const [hairSegments, setHairSegments] = useState(6);
  const [ribbonWidth, setRibbonWidth] = useState(0.004);
  const [clumpStrength, setClumpStrength] = useState(0.4);
  const [clumpSpacing, setClumpSpacing] = useState(0.02);
  const [curlRadius, setCurlRadius] = useState(0);
  const [curlTurns, setCurlTurns] = useState(2);
  const [hairJoint, setHairJoint] = useState(0);

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

  // …and the HISTORY rows the same way (audit fix). They were fetched on
  // opening the disclosure and after an amendment, and nowhere else — so with
  // the list open, an extrude did not appear in it and an undo did not grey a
  // row. The rows describe the journal, and the stamp is what says the
  // journal moved; the UV pane next door already had exactly this shape.
  useEffect(() => {
    if (assetId && showHistory && generation !== undefined) void historyRefresh(assetId);
  }, [assetId, showHistory, generation, historyRefresh]);

  // ── the preview surface ──────────────────────────────────────────────────
  const imgRef = useRef<HTMLImageElement | null>(null);
  const drag = useRef<DragState | null>(null);

  /**
 * One history row's draggable number.
 *
 * **Controlled, and the journal is the authority** (audit fix). This was an
 * uncontrolled `<input defaultValue>` under a stable `key`, so React kept the
 * DOM node: when the kernel REFUSED an amendment — which it does inertly, that
 * being the whole contract — the field went on showing the rejected number
 * while the journal held the old one, and the panel displayed a value the mesh
 * did not have.
 *
 * The sync is a render-phase adjustment rather than an effect: React documents
 * it for exactly this ("reset state when a prop changes"), it re-renders before
 * anything is painted, and `set-state-in-effect` is banned repo-wide.
 *
 * `onBlur` and Enter, not `onChange`: an amendment replays the whole tail, and
 * doing that per keystroke would replay it four times for "0.65".
 */
function AmendField({
  value,
  disabled,
  onCommit,
}: {
  value: number;
  disabled: boolean;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  const [seen, setSeen] = useState(value);
  if (seen !== value) {
    setSeen(value);
    setText(String(value));
  }
  return (
    <input
      type="number"
      step={0.01}
      value={text}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        const v = Number(text);
        if (Number.isFinite(v) && v !== value) onCommit(v);
        else setText(String(value));
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
      }}
      className="w-16 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1 py-0 text-right outline-none focus:border-(--ink-accent) disabled:opacity-40"
    />
  );
}

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
    if (tool === "weights") {
      return {
        kind: "weightPaint",
        joint: paintJoint,
        mode: paintMode,
        radius: brushRadius,
        strength: paintStrength,
        falloff,
      };
    }
    if (tool === "gizmo") {
      return { kind: "gizmo", mode: gizmoMode, snap, softRadius, falloff };
    }
    return null;
  };

  // ── the UV pane's pointer (Wave D) ───────────────────────────────────────
  //
  // A separate, much smaller state machine than the 3D one: there is no camera
  // to orbit and no backend drag to own, so a gesture is "down, accumulate,
  // commit on up" and the whole move is ONE journal entry. The accumulation is a
  // ref rather than state — it changes per pointer-move, and re-rendering the
  // panel to hold a number the backend has not seen yet would be re-rendering
  // for nothing.
  const uvRef = useRef<HTMLImageElement | null>(null);
  const uvDrag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  /** Pointer position in the UV pane's own pixel space. */
  const toUvPx = (e: React.PointerEvent): [number, number] => {
    const el = uvRef.current;
    if (!el) return [0, 0];
    const r = el.getBoundingClientRect();
    const s = DCC_PREVIEW_SIZE / Math.max(1, r.width);
    const t = DCC_PREVIEW_SIZE / Math.max(1, r.height);
    return [(e.clientX - r.left) * s, (e.clientY - r.top) * t];
  };

  const onUvPointerDown = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const [x, y] = toUvPx(e);
    uvDrag.current = { x, y, moved: false };
  };

  const onUvPointerMove = (e: React.PointerEvent) => {
    const d = uvDrag.current;
    if (!d || !assetId) return;
    const [x, y] = toUvPx(e);
    const dx = x - d.x;
    const dy = y - d.y;
    // The same 3 px threshold the 3D view uses, so a shaky click still picks.
    if (!d.moved && Math.abs(dx) + Math.abs(dy) < 3) return;
    d.moved = true;
    d.x = x;
    d.y = y;
    // Committed per move rather than accumulated to pointer-up: the pane has to
    // show the UVs moving, and `Op::MoveUvs` is id-preserving, so N of them in a
    // drag is N undo steps of the same kind — which is the shape the seam tool
    // already has and the one an author can unwind.
    void uvMove(assetId, dx, dy);
  };

  const onUvPointerUp = (e: React.PointerEvent) => {
    const d = uvDrag.current;
    uvDrag.current = null;
    if (!d || !assetId) return;
    if (!d.moved) {
      const [x, y] = toUvPx(e);
      void uvPick(assetId, x, y, e.shiftKey || e.ctrlKey);
    }
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
    // The marquee never asks the backend whether it grabbed something — it IS
    // the gesture, and it is resolved on pointer-up against a rectangle. Set
    // here rather than in `dragRequest`, which answers "what backend drag does
    // this start" and the answer for a marquee is "none".
    if (tool === "box") {
      d.kind = "box";
      setMarquee({ x0: px, y0: py, x1: px, y1: py });
      return;
    }
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

  /**
   * The hover brush ring (Wave D, closing the P23 remainder). Sent only while a
   * brush tool is armed — `setHover` collapses a repeat, and the store's preview
   * gate collapses the storm, so this is one queued frame per move at worst.
   */
  const trackHover = (e: React.PointerEvent | null) => {
    if (!assetId) return;
    const brushing = tool === "sculpt" || tool === "weights";
    if (!brushing || !e) {
      setHover(assetId, null);
      return;
    }
    const [px, py] = toPreviewPx(e);
    setHover(assetId, [px, py, brushRadius]);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) {
      // No drag: this is a hover, and the ring is what it is for.
      trackHover(e);
      return;
    }
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
    } else if (d.kind === "box") {
      const [px, py] = toPreviewPx(e);
      setMarquee((m) => (m ? { ...m, x1: px, y1: py } : m));
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
    if (d.kind === "box") {
      const [px, py] = toPreviewPx(e);
      setMarquee(null);
      // A marquee that never moved is a click, and a click in box mode should
      // still pick — otherwise switching to the marquee tool costs the author
      // ordinary selection.
      if (assetId) {
        if (d.moved) void boxSelect(assetId, d.downX, d.downY, px, py, d.shift);
        else void pick(assetId, d.downX, d.downY, d.shift);
      }
      return;
    }
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
  //
  // **Round-2 finding R2.F9.** This read `e.dataTransfer` and nothing in the
  // tree ever called `setData`: the Content Drawer drags with pointer events
  // (it has to — its other target is the native viewport hole, which is not a
  // DOM node), so the zone highlighted for any native drag and silently
  // discarded it. `dcc_merge_asset` had no reachable caller at all, which is
  // also why R2.F5 and the two merge defects beside it were latent.
  const dropRef = useRef<HTMLDivElement | null>(null);
  const onAsset = useCallback(
    (detail: AssetDropDetail) => {
      setDropTarget(false);
      // The kind comes with the drop; `assetsById` is the cross-check, so a
      // stale drawer snapshot cannot make a non-mesh reach the merge door.
      if (detail.kind !== "mesh" || assetsById[detail.id]?.kind !== "mesh") return;
      if (assetId) void mergeAsset(assetId, detail.id);
    },
    [assetId, assetsById, mergeAsset],
  );
  useEffect(() => {
    const el = dropRef.current;
    if (!el) return;
    return onAssetDrop(el, onAsset);
  }, [onAsset]);
  // Light up while a mesh is in flight over this panel. The listener is on
  // `window` rather than on the zone because the drawer cell holds pointer
  // capture for the whole drag, so `pointerenter`/`pointerleave` never fire on
  // anything else — the same reason the drawer hit-tests coordinates.
  useEffect(() => {
    if (dragAsset?.kind !== "mesh") return;
    const el = dropRef.current;
    if (!el) return;
    const over = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      setDropTarget(
        e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom,
      );
    };
    window.addEventListener("pointermove", over);
    // The un-light is in the CLEANUP, which is also what runs when the drag
    // ends — so a drag that leaves the window, is cancelled, or is dropped
    // elsewhere cannot leave this zone outlined.
    return () => {
      window.removeEventListener("pointermove", over);
      setDropTarget(false);
    };
  }, [dragAsset]);

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
          ref={dropRef}
          {...{ [ASSET_DROP_ATTR]: "mesh" }}
          className={cn(
            "flex min-h-0 flex-1 items-center justify-center bg-(--ink-bg-0) p-3",
            dropTarget && "outline outline-2 -outline-offset-4 outline-(--ink-accent)",
          )}
        >
          {image ? (
            // `relative` so the marquee can be positioned against the IMAGE's own
            // box rather than the panel's — the rectangle is in preview pixels,
            // and the image is letterboxed inside its container.
            <div className="relative">
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
              {marquee && (
                <div
                  className="pointer-events-none absolute border border-dashed border-(--ink-accent) bg-(--ink-accent)/10"
                  style={{
                    left: `${(Math.min(marquee.x0, marquee.x1) / DCC_PREVIEW_SIZE) * 100}%`,
                    top: `${(Math.min(marquee.y0, marquee.y1) / DCC_PREVIEW_SIZE) * 100}%`,
                    width: `${(Math.abs(marquee.x1 - marquee.x0) / DCC_PREVIEW_SIZE) * 100}%`,
                    height: `${(Math.abs(marquee.y1 - marquee.y0) / DCC_PREVIEW_SIZE) * 100}%`,
                  }}
                />
              )}
            </div>
          ) : (
            <div className="text-(--ink-text-dim)">{previewError ?? "Rendering…"}</div>
          )}
        </div>
        {showUv && (
          <div className="flex h-48 shrink-0 items-center justify-center border-t border-(--ink-border) bg-(--ink-bg-0) p-2">
            {uvImage ? (
              // **The UV pane answers a pointer** (Wave D). Click picks — into
              // the SHARED selection, so the 3D view lights up too — and a drag
              // moves the selection's corners as ONE journal entry.
              <img
                ref={uvRef}
                src={uvImage}
                alt="uv layout"
                draggable={false}
                className="max-h-full max-w-full cursor-crosshair select-none rounded"
                style={{ imageRendering: "pixelated" }}
                onPointerDown={onUvPointerDown}
                onPointerMove={onUvPointerMove}
                onPointerUp={onUvPointerUp}
                onPointerCancel={onUvPointerUp}
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
          {/* The live delta / angle / factor — the P23 carried remainder, for
              all three gizmo modes rather than only the rotate one it named. */}
          {doc.readout && (
            <span className="font-mono text-(--ink-accent)">{doc.readout}</span>
          )}
          {refusal && <span className="ml-auto truncate text-(--ink-warn,#ffb454)">{refusal}</span>}
        </div>
      </div>

      {/* ── tools ─────────────────────────────────────────────────────── */}
      <div className="flex w-64 shrink-0 flex-col gap-2 overflow-y-auto p-2">
        {/* ── starting a model at all (Wave D) ────────────────────────────
            The kernel has had four primitives since P23.3 and Ring 2 had no
            door for any of them, so the Model Editor could open an imported
            mesh and nothing else. */}
        <ToolButton
          label={newOpen ? "New mesh ▾" : "New mesh ▸"}
          icon={<Box size={12} />}
          onClick={() => setNewOpen((v) => !v)}
          title="Mint a .inf_mesh from a primitive and open it. It lands in Content/Meshes."
        />
        {newOpen && (
          <div className="flex flex-col gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) p-2">
            <select
              value={newPrimitive}
              onChange={(e) => setNewPrimitive(e.target.value as DccPrimitiveDto)}
              className="rounded border border-(--ink-border) bg-(--ink-bg-1) px-1 py-0.5 text-[11px]"
            >
              <option value="cube">Cube</option>
              <option value="plane">Plane</option>
              <option value="cylinder">Cylinder</option>
              <option value="torus">Torus</option>
            </select>
            <label className="flex items-center justify-between gap-2 text-[11px]">
              <span className="text-(--ink-text-dim)">Name</span>
              <input
                value={newName}
                placeholder={newPrimitive}
                onChange={(e) => setNewName(e.target.value)}
                className="w-28 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1.5 py-0.5 outline-none focus:border-(--ink-accent)"
              />
            </label>
            <Num label="Size (m)" value={newSize} onChange={setNewSize} />
            {(newPrimitive === "cylinder" || newPrimitive === "torus") && (
              <Num
                label="Segments"
                value={newSegments}
                step={1}
                onChange={(v) => setNewSegments(Math.round(v))}
              />
            )}
            {newPrimitive === "torus" && (
              <Num
                label="Rings"
                value={newRings}
                step={1}
                onChange={(v) => setNewRings(Math.round(v))}
              />
            )}
            <ToolButton
              label="Create"
              icon={<Sparkles size={12} />}
              onClick={() => {
                void newMesh({
                  primitive: newPrimitive,
                  name: newName.trim() || null,
                  sizeM: newSize,
                  segments: newSegments,
                  rings: newRings,
                }).then((id) => {
                  if (!id) return;
                  setNewOpen(false);
                  // Open a Model Editor ON the mesh just made, so "Create" ends
                  // with the author editing it rather than hunting the drawer.
                  // `openPanel` is keyed by `type:params`, so pressing Create
                  // twice for the same asset re-focuses one panel.
                  useDockLayout.getState().openPanel("model", id);
                });
              }}
            />
            <p className="text-[10px] leading-snug text-(--ink-text-dim)">
              Size is the bounding box: a cube&rsquo;s edge, a cylinder&rsquo;s diameter and
              height, a torus&rsquo;s outer diameter.
            </p>
          </div>
        )}

        <div className="flex gap-1">
          {modeButton("vert", <CircleDot size={12} />, "Vert")}
          {modeButton("edge", <ChevronsLeftRight size={12} />, "Edge")}
          {modeButton("face", <Square size={12} />, "Face")}
        </div>

        <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
          POINTER
        </div>
        <div className="flex gap-1">
          {(["select", "box", "sculpt", "weights", "gizmo"] as PointerTool[]).map((t) => (
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
                    : t === "weights"
                      ? "Paint one skinning influence. Needs a mesh bound to a skeleton."
                      : "Drag a handle on the selection. Off the handles, the drag orbits."
              }
            >
              {t}
            </button>
          ))}
        </div>

        {tool === "weights" && (
          <>
            {/* The bound skeleton's joint count is the picker's range, and its
                absence is the whole story: a rigid mesh has no influence to
                paint, so the panel says so rather than offering a control that
                only ever refuses. */}
            {doc?.skinJoints == null ? (
              <p className="text-[10px] leading-snug text-(--ink-text-dim)">
                This mesh carries <b>no skin</b>, so there is no influence to
                paint. Bind it to a skeleton first — the weight solve and the
                binding UI arrive in P24.3.
              </p>
            ) : (
              <>
                <div className="flex gap-1">
                  {(["add", "subtract", "replace", "smooth"] as DccPaintModeDto[]).map(
                    (m) => (
                      <button
                        key={m}
                        className={cn(
                          "flex-1 rounded px-1 py-1 text-[10px] capitalize",
                          paintMode === m
                            ? "bg-(--ink-accent) text-(--ink-text-onaccent)"
                            : "bg-(--ink-bg-2) hover:bg-(--ink-bg-3)",
                        )}
                        onClick={() => setPaintMode(m)}
                      >
                        {m}
                      </button>
                    ),
                  )}
                </div>
                <label className="flex items-center justify-between gap-2 text-[11px]">
                  <span className="text-(--ink-text-dim)">
                    Influence (0–{doc.skinJoints - 1})
                  </span>
                  <input
                    type="number"
                    min={0}
                    max={doc.skinJoints - 1}
                    step={1}
                    value={paintJoint}
                    onChange={(e) =>
                      setPaintJoint(
                        Math.min(
                          (doc?.skinJoints ?? 1) - 1,
                          Math.max(0, Math.round(Number(e.target.value) || 0)),
                        ),
                      )
                    }
                    className="w-20 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 py-0.5 text-right text-[11px]"
                  />
                </label>
                <Num
                  label="Radius (m)"
                  value={brushRadius}
                  onChange={(v) => setBrushRadius(Math.max(MIN_BRUSH_RADIUS_M, v))}
                />
                <Num
                  label="Strength"
                  value={paintStrength}
                  step={0.05}
                  onChange={(v) => setPaintStrength(Math.min(1, Math.max(0, v)))}
                />
                <p className="text-[10px] leading-snug text-(--ink-text-dim)">
                  Strength is a <b>weight delta</b> at full coverage, and coverage
                  is the largest influence any dab of the stroke gave a vertex —
                  so a slow drag paints the same as a fast one over the same
                  ground. One stroke = one undo step.
                </p>
              </>
            )}
          </>
        )}

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
              The handles sit on the pivot chosen under TRANSFORM. Snap 0 is off; a soft
              radius above 0 blends the move into the geodesic neighbourhood and journals
              the whole drag as ONE entry. The Translate / Rotate / Scale boxes under
              TRANSFORM journal the identical ops &mdash; one function, not two paths.
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

        {/* ── the numeric transform box (Wave D) ────────────────────────────
            The panel's own help text has claimed since P23.4 that "the Translate
            box below journals the identical op", beside no box at all. These
            three go through `dcc::transform_ops`, which is the same function the
            dragged gizmo commits through — so the claim is now true of the
            product and not only of the Rust. */}
        <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
          TRANSFORM
        </div>
        <div className="flex items-center gap-1">
          <select
            value={doc.pivot}
            onChange={(e) =>
              assetId && void setViewOpts(assetId, { pivot: e.target.value as DccPivotDto })
            }
            className="flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 py-0.5 text-[11px]"
            title="Where the gizmo sits. Individual origins is not built — it means one op per element, which this tool does not produce."
          >
            <option value="median">Pivot: median</option>
            <option value="boundingBox">Pivot: bbox</option>
            <option value="worldOrigin">Pivot: origin</option>
            <option value="activeElement">Pivot: active</option>
          </select>
          <select
            value={doc.orient}
            onChange={(e) =>
              assetId && void setViewOpts(assetId, { orient: e.target.value as DccOrientDto })
            }
            className="flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 py-0.5 text-[11px]"
            title="Which way the gizmo's axes point"
          >
            <option value="global">Axes: global</option>
            <option value="normal">Axes: normal</option>
            <option value="view">Axes: view</option>
          </select>
        </div>
        <div className="grid grid-cols-3 gap-1">
          <Num label="X" value={nudge[0]} onChange={(v) => setNudge([v, nudge[1], nudge[2]])} />
          <Num label="Y" value={nudge[1]} onChange={(v) => setNudge([nudge[0], v, nudge[2]])} />
          <Num label="Z" value={nudge[2]} onChange={(v) => setNudge([nudge[0], nudge[1], v])} />
        </div>
        <ToolButton
          label="Translate (m)"
          icon={<Move size={12} />}
          disabled={nothing}
          onClick={() => assetId && void apply(assetId, { tool: "translate", delta: nudge })}
          title="Move the selection by an exact delta. The identical op the gizmo journals — one function, not two paths kept in step."
        />
        <div className="flex items-center gap-1">
          <select
            value={spinAxis}
            onChange={(e) => setSpinAxis(e.target.value as "x" | "y" | "z")}
            className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 py-0.5 text-[11px]"
          >
            <option value="x">X</option>
            <option value="y">Y</option>
            <option value="z">Z</option>
          </select>
          <div className="flex-1">
            <Num label="Degrees" value={spinDeg} step={1} onChange={setSpinDeg} />
          </div>
        </div>
        <ToolButton
          label="Rotate (°)"
          icon={<Circle size={12} />}
          disabled={nothing}
          onClick={() =>
            assetId &&
            void apply(assetId, {
              tool: "rotate",
              axis: spinAxis === "x" ? [1, 0, 0] : spinAxis === "y" ? [0, 1, 0] : [0, 0, 1],
              degrees: spinDeg,
            })
          }
          title="Rotate about the pivot chosen above. Degrees here, radians in the op."
        />
        <div className="grid grid-cols-3 gap-1">
          <Num
            label="X"
            value={scaleBy[0]}
            onChange={(v) => setScaleBy([v, scaleBy[1], scaleBy[2]])}
          />
          <Num
            label="Y"
            value={scaleBy[1]}
            onChange={(v) => setScaleBy([scaleBy[0], v, scaleBy[2]])}
          />
          <Num
            label="Z"
            value={scaleBy[2]}
            onChange={(v) => setScaleBy([scaleBy[0], scaleBy[1], v])}
          />
        </div>
        <ToolButton
          label="Scale (x)"
          icon={<Expand size={12} />}
          disabled={nothing}
          onClick={() => assetId && void apply(assetId, { tool: "scale", factor: scaleBy })}
          title="Scale about the pivot. 1 is unchanged; negative mirrors, which is what dragging a handle past the pivot does."
        />
        <Num label="Extrude edges (m)" value={distance} onChange={setDistance} />
        <ToolButton
          label="Extrude edges"
          icon={<Move size={12} />}
          disabled={nothing || mode !== "edge"}
          onClick={() =>
            assetId &&
            void apply(assetId, {
              tool: "extrudeEdges",
              delta: nudge.every((v) => v === 0) ? [0, distance, 0] : nudge,
            })
          }
          title="Grow faces out of the selected BOUNDARY edges by the XYZ delta above (or straight up, if it is all zeroes). An edge has no canonical direction, so you supply one."
        />
        <label className="flex items-center gap-2 text-[11px] text-(--ink-text-dim)">
          <input
            type="checkbox"
            checked={softRadius > 0}
            onChange={(e) => setSoftRadius(e.target.checked ? 0.5 : 0)}
          />
          Proportional (soft) — one undo step
        </label>
        {softRadius > 0 && (
          <ToolButton
            label="Soft translate"
            icon={<Move size={12} />}
            disabled={nothing}
            onClick={() =>
              assetId &&
              void apply(assetId, {
                tool: "softTranslate",
                delta: nudge,
                radius: softRadius,
                falloff,
              })
            }
            title="The whole neighbourhood moves, weighted by geodesic distance in metres — and the whole drag is ONE journal entry."
          />
        )}

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
        <Num
          label="Bevel segments"
          value={bevelSegments}
          step={1}
          onChange={(v) => setBevelSegments(Math.min(MAX_BEVEL_SEGMENTS, Math.max(1, Math.round(v))))}
        />
        <ToolButton
          label="Bevel"
          icon={<Slice size={12} />}
          disabled={nothing || mode !== "edge"}
          onClick={() => assetId && void apply(assetId, { tool: "bevel", amount: bevel, segments: bevelSegments })}
          title="1 segment is a flat chamfer; above it the profile rounds on a circular arc. Edges that MEET at a right angle still refuse — a corner join is not built."
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

        {/* ── the history (Wave D) ──────────────────────────────────────────
            `edit.undoHistory` has been in the Edit menu since Phase 1 with no
            panel behind it. This is the panel — and it does the thing Blender's
            redo panel cannot: re-parameterize an edit that is NOT the last one,
            and re-derive everything after it. */}
        <ToolButton
          label={showHistory ? "History ▾" : "History ▸"}
          icon={<Layers size={12} />}
          onClick={() => setShowHistory(!showHistory)}
          title="Every edit in this session. An edit with a single number can be changed where it is, and the rest of the model re-derives."
        />
        {showHistory && (
          <div className="flex max-h-64 flex-col gap-0.5 overflow-y-auto rounded border border-(--ink-border) bg-(--ink-bg-2) p-1">
            {history.length === 0 && (
              <span className="px-1 py-0.5 text-[11px] text-(--ink-text-dim)">
                No edits yet.
              </span>
            )}
            {history.map((h) => (
              <div
                key={h.index}
                className={cn(
                  "flex items-center gap-1 rounded px-1 py-0.5 text-[11px]",
                  !h.applied && "opacity-40",
                )}
                title={h.reason ?? "Change this edit; everything after it re-derives."}
              >
                <span className="w-6 shrink-0 text-right font-mono text-[10px] text-(--ink-text-dim)">
                  {h.index}
                </span>
                <span className="min-w-0 flex-1 truncate">{h.kind}</span>
                {h.value !== null && (
                  <AmendField
                    value={h.value}
                    disabled={!h.amendable}
                    onCommit={(v) => {
                      if (assetId) void amend(assetId, h.index, v);
                    }}
                  />
                )}
                <span className="w-4 shrink-0 text-[10px] text-(--ink-text-dim)">{h.unit}</span>
              </div>
            ))}
          </div>
        )}

        <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">TOPOLOGY</div>
        <div className="grid grid-cols-2 gap-1">
          <ToolButton
            label="Dissolve"
            icon={<Slice size={12} />}
            disabled={nothing || mode !== "edge"}
            onClick={() => assetId && void apply(assetId, { tool: "dissolve" })}
            title="Merge the two faces across each selected edge into one n-gon. The vertices stay — that is the difference from a collapse."
          />
          <ToolButton
            label="Bridge"
            icon={<Merge size={12} />}
            disabled={nothing || mode !== "edge"}
            onClick={() => assetId && void apply(assetId, { tool: "bridge" })}
            title="Stitch two open borders together. Needs exactly two loops with the same edge count."
          />
        </div>
        <ToolButton
          label="Flip / recalc normals"
          icon={<Triangle size={12} />}
          onClick={() => assetId && void apply(assetId, { tool: "flip" })}
          title="Reverse the winding of the selected faces, or of the whole mesh when nothing is selected."
        />
        <div className="grid grid-cols-2 gap-1">
          <ToolButton
            label="Shade smooth"
            icon={<Circle size={12} />}
            onClick={() => assetId && void apply(assetId, { tool: "shade", smooth: true, angleDeg: null })}
            title="Clear the crease on every edge of the selection (or of the whole mesh)."
          />
          <ToolButton
            label="Shade flat"
            icon={<Square size={12} />}
            onClick={() => assetId && void apply(assetId, { tool: "shade", smooth: false, angleDeg: null })}
          />
        </div>
        <Num
          label="Auto-smooth (°)"
          value={smoothAngle}
          step={1}
          onChange={(v) => setSmoothAngle(Math.min(180, Math.max(0, v)))}
        />
        <ToolButton
          label="Auto-smooth"
          icon={<Circle size={12} />}
          onClick={() =>
            assetId && void apply(assetId, { tool: "shade", smooth: true, angleDeg: smoothAngle })
          }
          title="Crease every edge whose two faces disagree by more than the angle; smooth the rest."
        />
        <Num label="Slide (−1…1)" value={slide} onChange={(v) => setSlide(Math.min(1, Math.max(-1, v)))} />
        <ToolButton
          label="Slide"
          icon={<Move size={12} />}
          disabled={nothing}
          onClick={() => assetId && void apply(assetId, { tool: "slide", t: slide })}
          title="Slide the selection along its own ring edges — select an edge loop and this is edge slide. One undo step."
        />
        <Num label="Merge dist (m)" value={mergeTolerance} onChange={setMergeTolerance} />
        <ToolButton
          label="Merge by distance"
          icon={<Merge size={12} />}
          disabled={nothing}
          onClick={() => assetId && void apply(assetId, { tool: "mergeByDistance", tolerance: mergeTolerance })}
          title="Fuse selected vertices closer than the tolerance. One undo step per cluster — the reader itself never welds by epsilon."
        />

        {/* ── material slots (Wave D) ───────────────────────────────────────
            `Op::AddMaterialSlots` and `Op::SetFaceSlot` have existed since P24.3
            with no UI caller at all — the only consumer was the drop-merge — so
            a multi-material prop could not be authored here. */}
        <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
          MATERIAL SLOTS
        </div>
        <div className="flex flex-col gap-0.5">
          <button
            className="flex items-center justify-between rounded px-1.5 py-0.5 text-left text-[11px] hover:bg-(--ink-bg-3) disabled:opacity-40"
            disabled={nothing || mode !== "face"}
            onClick={() => assetId && void apply(assetId, { tool: "assignSlot", slot: null })}
            title="Put the selected faces back on the default material"
          >
            <span className="text-(--ink-text-dim)">(default)</span>
            <span className="font-mono text-[10px]">assign</span>
          </button>
          {doc.materialSlots.map((name, i) => (
            <button
              key={`${i}-${name}`}
              className="flex items-center justify-between rounded px-1.5 py-0.5 text-left text-[11px] hover:bg-(--ink-bg-3) disabled:opacity-40"
              disabled={nothing || mode !== "face"}
              onClick={() => assetId && void apply(assetId, { tool: "assignSlot", slot: i })}
              title={`Assign the selected faces to slot ${i}`}
            >
              <span className="truncate">
                <span className="text-(--ink-text-dim)">{i}. </span>
                {name}
              </span>
              <span className="font-mono text-[10px]">assign</span>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <input
            value={slotName}
            placeholder="new slot name"
            onChange={(e) => setSlotName(e.target.value)}
            className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 text-[11px] outline-none focus:border-(--ink-accent)"
          />
          <button
            className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-0.5 text-[11px] hover:bg-(--ink-bg-3) disabled:opacity-40"
            disabled={!slotName.trim()}
            onClick={() => {
              if (!assetId) return;
              void apply(assetId, { tool: "addSlots", names: [slotName.trim()] });
              setSlotName("");
            }}
            title="Append a slot. Append-only: a face records its slot as an INDEX, so inserting one would silently repaint every face after it."
          >
            Add
          </button>
        </div>

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
        <Num
          label="Auto-seam (°)"
          value={seamAngle}
          step={1}
          onChange={(v) => setSeamAngle(Math.min(180, Math.max(0, v)))}
        />
        <div className="grid grid-cols-2 gap-1">
          <ToolButton
            label="Auto-seam"
            icon={<Scissors size={12} />}
            onClick={() =>
              assetId &&
              void apply(assetId, { tool: "autoSeam", angleDeg: seamAngle, replace: true })
            }
            title="Cut every edge whose faces disagree by more than the angle, plus every border. One undo step, unlike marking by hand."
          />
          <ToolButton
            label="Add seams"
            icon={<Scissors size={12} />}
            onClick={() =>
              assetId &&
              void apply(assetId, { tool: "autoSeam", angleDeg: seamAngle, replace: false })
            }
            title="The same cuts, added to the seams you already have"
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


        {/* ── P24.4 cloth & hair authoring ──────────────────────────────
            The two buttons that give `ClothAsset::from_garment` and
            `HairAsset::grow` a caller outside a test. Each writes a NEW asset
            (Content/Cloth, Content/Hair) rather than editing this one, so
            neither is a save and neither touches the journal. */}
        <div className="mt-1 text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
          CLOTH &amp; HAIR
        </div>
        <label className="flex items-center justify-between gap-2 text-[11px]">
          <span className="text-(--ink-text-dim)">Rig (.inf_skel)</span>
          <input
            type="text"
            value={rigId}
            placeholder="asset GUID"
            onChange={(e) => setRigId(e.target.value)}
            className="w-28 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 text-right font-mono text-[10px] outline-none focus:border-(--ink-accent)"
          />
        </label>
        <p className="text-[10px] leading-snug text-(--ink-text-dim)">
          The rig&apos;s bones become the collision capsules a garment and a
          hairstyle are held outside — one radius for all of them, which is the
          v1 bound. Leave it empty and the cloth hangs through its wearer.
        </p>
        <Num label="Body radius (m)" value={bodyRadius} step={0.01} onChange={setBodyRadius} />

        <Num label="Bend compliance (m/N)" value={bendCompliance} step={0.0005} onChange={setBendCompliance} />
        <Num label="Cloth damping (1/s)" value={clothDamping} step={0.05} onChange={setClothDamping} />
        <Num label="Thickness (m)" value={clothThickness} step={0.001} onChange={setClothThickness} />
        <Num label="Substeps" value={clothSubsteps} step={1} onChange={(v) => setClothSubsteps(Math.round(v))} />
        <ToolButton
          label="Make garment"
          icon={<Shirt size={12} />}
          disabled={busy}
          onClick={() =>
            assetId &&
            void makeGarment(assetId, {
              stretchCompliance: 0,
              bendCompliance,
              damping: clothDamping,
              thicknessM: clothThickness,
              substeps: clothSubsteps,
              iterations: 1,
              bodyRadiusM: bodyRadius,
              skeleton: rigId.trim() ? rigId.trim() : null,
              name: null,
            })
          }
          title="Write this mesh as a .inf_cloth. The SELECTED VERTICES become the pins."
        />
        <p className="text-[10px] leading-snug text-(--ink-text-dim)">
          Select the vertices that should stay put — a collar, a waistband — and
          they are pinned. Nothing selected is a garment that falls.
        </p>

        <Num label="Hair length (m)" value={hairLength} step={0.05} onChange={setHairLength} />
        <Num label="Segments" value={hairSegments} step={1} onChange={(v) => setHairSegments(Math.round(v))} />
        <Num label="Ribbon width (m)" value={ribbonWidth} step={0.001} onChange={setRibbonWidth} />
        <Num label="Clump strength" value={clumpStrength} step={0.1} onChange={setClumpStrength} />
        <Num label="Clump cell (m)" value={clumpSpacing} step={0.005} onChange={setClumpSpacing} />
        <Num label="Curl radius (m)" value={curlRadius} step={0.005} onChange={setCurlRadius} />
        <Num label="Curl turns" value={curlTurns} step={0.5} onChange={setCurlTurns} />
        <Num label="Root joint" value={hairJoint} step={1} onChange={(v) => setHairJoint(Math.max(0, Math.round(v)))} />
        <ToolButton
          label="Grow guides"
          icon={<Sparkles size={12} />}
          disabled={busy || mode !== "face"}
          onClick={() =>
            assetId &&
            void growHair(assetId, {
              lengthM: hairLength,
              segments: hairSegments,
              segmentCompliance: 0,
              damping: 2,
              thicknessM: 0.004,
              substeps: 8,
              ribbonWidthM: ribbonWidth,
              clumpStrength,
              clumpSpacingM: clumpSpacing,
              curlRadiusM: curlRadius,
              curlTurns,
              fallbackJoint: hairJoint,
              bodyRadiusM: bodyRadius,
              skeleton: rigId.trim() ? rigId.trim() : null,
              name: null,
            })
          }
          title="Grow one guide out of each SELECTED FACE, along its normal."
        />
        <p className="text-[10px] leading-snug text-(--ink-text-dim)">
          Switch to face mode and select the scalp. One guide grows out of each
          face, from its centre, along its normal. &quot;Root joint&quot; is only
          used where the mesh carries no skin weights of its own.
        </p>
        {lastGroom && (
          <>
            {lastGroom.stats.map((s) => (
              <Verdict key={s.label} label={s.label} value={s.value} good={s.value > 0} />
            ))}
            {lastGroom.path && (
              <p className="text-[10px] leading-snug break-all text-(--ink-text-dim)">
                {lastGroom.path}
              </p>
            )}
            {lastGroom.refusal && (
              <p className="rounded bg-(--ink-warn-bg,#3a2a12) p-1 text-[10px] leading-snug text-(--ink-warn,#ffb454)">
                {lastGroom.refusal}
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
        {/* P24.2: the skin weld's advisory. Non-zero means two source vertices
            at one position disagreed about their influences — first occurrence
            won, and re-exporting will not reproduce the loser's weights. */}
        <Verdict
          label="Skin conflicts"
          value={imp.skinConflicts}
          good={imp.skinConflicts === 0}
        />
        {imp.skinConflicts > 0 && (
          <p className="text-[10px] leading-snug text-(--ink-text-dim)">
            Two source vertices at the same position carried <b>different</b>{" "}
            skinning weights. The first one won; re-exporting will not reproduce
            the other. Nothing was averaged.
          </p>
        )}
        {/* Wave D: the three import totals the `NOT_SHOWN` freeze recorded as
            "an author reads these off the mesh stats instead". They do not —
            `sourceVertices` is what the FILE had, and the stats are what the
            KERNEL has, which is exactly the pair that says how much welding
            happened. */}
        <Verdict label="Source vertices" value={imp.sourceVertices} good />
        <Verdict label="Welded positions" value={imp.weldedPositions} good />
        <Verdict label="Sharp edges" value={imp.sharpEdges} good />
        {/* Wave D: the non-manifold repair. Through P24 an asset with any of
            these was REFUSED outright and there was no repair door anywhere in
            the product — so a large fraction of real game art could not be
            opened. Now it opens, and the counts are the contract. */}
        <Verdict
          label="Duplicate faces dropped"
          value={imp.duplicateFacesDropped}
          good={imp.duplicateFacesDropped === 0}
        />
        <Verdict
          label="Faces reoriented"
          value={imp.facesReoriented}
          good={imp.facesReoriented === 0}
        />
        <Verdict
          label="Non-manifold detached"
          value={imp.nonManifoldSplits}
          good={imp.nonManifoldSplits === 0}
        />
        {imp.nonManifoldSplits > 0 && (
          <p className="text-[10px] leading-snug text-(--ink-text-dim)">
            Three or more faces met at one edge — an interior partition, or a
            double-sided sheet. That is not a surface, so the extras came in as{" "}
            <b>separate shells</b> at the same coordinates. Nothing was thrown away and
            nothing was moved; they are simply no longer joined.
          </p>
        )}
        {imp.facesReoriented > 0 && (
          <p className="text-[10px] leading-snug text-(--ink-text-dim)">
            Some faces were wound the other way round from their neighbours and were
            flipped to agree. The surface is identical &mdash; the authored normals were
            kept verbatim, so re-exporting looks exactly as it arrived.
          </p>
        )}

        {lastSave && (
          <>
            <div className="mt-1 text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
              LAST SAVE
            </div>
            <Verdict label="Triangles" value={lastSave.export.triangles} good />
            {/* P24.2 re-audit minor 1: `optimized` reached the DTO and the TS
                binding and stopped there — which by this gate's own headline law
                ("a report field that never reaches the author is a field that
                does not exist") meant it still did not exist. It is a SETTING,
                not a count, so it is shown as the state it is. */}
            <Verdict
              label="meshopt"
              value={lastSave.export.optimized ? "ran" : "off"}
              good={!lastSave.export.optimized}
            />
            {/* P24.2: `optimize` cannot permute a parallel skin stream, so it is
                skipped on a skinned submesh — and the author is told, because the
                flag's documented effect is "smaller and faster". */}
            {lastSave.export.optimizeSkippedSkinned > 0 && (
              <Verdict
                label="Optimize skipped (skinned)"
                value={lastSave.export.optimizeSkippedSkinned}
                good={false}
              />
            )}
            <Verdict label="Vertices" value={lastSave.export.vertices} good />
            <Verdict label="vmesh" value={lastSave.vmesh} good={lastSave.vmesh !== "skipped"} />
            {/* Wave D: the seven export counters the `NOT_SHOWN` freeze recorded
                as having "never had rows". Five of them also produce a sentence
                in `advisories` below; the row is what makes the ZERO visible,
                which is the half a sentence cannot say. */}
            <Verdict label="Submeshes" value={lastSave.export.submeshes} good />
            <Verdict
              label="Coincident vertices"
              value={lastSave.export.coincidentVertices}
              good={lastSave.export.coincidentVertices === 0}
            />
            <Verdict
              label="Reused diagonals"
              value={lastSave.export.reusedDiagonals}
              good={lastSave.export.reusedDiagonals === 0}
            />
            <Verdict
              label="Fan fallbacks"
              value={lastSave.export.fanFallbacks}
              good={lastSave.export.fanFallbacks === 0}
            />
            <Verdict
              label="Fallback tangents"
              value={lastSave.export.fallbackTangents}
              good={lastSave.export.fallbackTangents === 0}
            />
            <Verdict
              label="Non-finite written"
              value={lastSave.export.nonFiniteWritten}
              good={lastSave.export.nonFiniteWritten === 0}
            />
            <Verdict
              label="Non-unit normals"
              value={lastSave.export.nonUnitNormalsWritten}
              good={lastSave.export.nonUnitNormalsWritten === 0}
            />
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
