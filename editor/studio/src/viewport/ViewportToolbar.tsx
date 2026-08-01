/**
 * Viewport toolbar (P8.2c): the Perspective / 2D projection toggle and, in 2D
 * mode, the grid- and pixel-snapping controls. Rendered as a strip ABOVE the
 * native viewport hole (never over it — the native child window would occlude
 * HTML there; airspace rule). All state lives in `viewportStore`, which mirrors
 * each change to the native viewport over typed IPC.
 */
import { useViewportStore } from "../stores/viewportStore";
import { useShellStore } from "../stores/shellStore";
import type { SculptFalloffDto } from "../bindings/SculptFalloffDto";
import type { SculptOpDto } from "../bindings/SculptOpDto";
import type { ToolModeDto } from "../bindings/ToolModeDto";
import type { ViewModeDto } from "../bindings/ViewModeDto";
import type { ViewportModeDto } from "../bindings/ViewportModeDto";

const MODES: [ViewportModeDto, string][] = [
  ["Perspective", "Perspective"],
  ["TwoD", "2D"],
];

/** Shading view modes (R-P2): [id, label, tooltip]. Wireframe is optimistically
 * enabled — the renderer clamps it to Unlit on GPUs without line-polygon raster
 * (POLYGON_MODE_LINE), so we don't gate the button on a caps query. */
const VIEW_MODES: [ViewModeDto, string, string][] = [
  ["Lit", "Lit", "Full lighting (default)"],
  ["Unlit", "Unlit", "Flat albedo + emissive (no lighting)"],
  ["Wireframe", "Wireframe", "Edge wireframe (falls back to Unlit if the GPU can't raster lines)"],
];

const TOOLS: [ToolModeDto, string, string][] = [
  ["Select", "Select", "Pick + transform (Q/W/E/R gizmos)"],
  ["Sculpt", "Sculpt", "Terrain height brush (perspective only)"],
  ["Foliage", "Foliage", "Scatter foliage onto the terrain (perspective only)"],
];

const SCULPT_OPS: [SculptOpDto, string][] = [
  ["Raise", "Raise"],
  ["Lower", "Lower"],
  ["Smooth", "Smooth"],
  ["Flatten", "Flatten"],
  ["Noise", "Noise"],
  ["Paint", "Paint"],
];

const SCULPT_FALLOFFS: SculptFalloffDto[] = ["Smooth", "Linear", "Sphere", "Sharp"];

/** Swatch colour + name for each splat layer (sRGB approximations of the default
 * `inf_ecs` layer palette: grass / rock / dirt / snow). Per-terrain layer-colour
 * editing in Details is the documented follow-up. */
const LAYER_SWATCHES: [string, string][] = [
  ["#7a9c66", "Grass"],
  ["#99928b", "Rock"],
  ["#aa9273", "Dirt"],
  ["#eef1f8", "Snow"],
];

/** Why the brush tools are off while a streamed terrain is projected. */
const STREAMED_TERRAIN_HINT =
  "This terrain streams from a .inf_terrain asset — sculpt and paint cannot edit it yet.";

export default function ViewportToolbar() {
  const mode = useViewportStore((s) => s.mode);
  const setMode = useViewportStore((s) => s.setMode);
  const gridSnapEnabled = useViewportStore((s) => s.gridSnapEnabled);
  const gridSnapSize = useViewportStore((s) => s.gridSnapSize);
  const pixelSnapEnabled = useViewportStore((s) => s.pixelSnapEnabled);
  const pixelsPerUnit = useViewportStore((s) => s.pixelsPerUnit);
  const setGridSnapEnabled = useViewportStore((s) => s.setGridSnapEnabled);
  const setGridSnapSize = useViewportStore((s) => s.setGridSnapSize);
  const setPixelSnapEnabled = useViewportStore((s) => s.setPixelSnapEnabled);
  const setPixelsPerUnit = useViewportStore((s) => s.setPixelsPerUnit);
  const toolMode = useViewportStore((s) => s.toolMode);
  const setToolMode = useViewportStore((s) => s.setToolMode);
  const terrainStreamed = useViewportStore((s) => s.terrainStreamed);
  const viewMode = useViewportStore((s) => s.viewMode);
  const setViewMode = useViewportStore((s) => s.setViewMode);

  return (
    <div className="flex h-8 shrink-0 items-center gap-3 rounded border border-(--ink-border) bg-(--ink-bg-1) px-2 text-xs">
      {/* Projection toggle. */}
      <div className="flex items-center rounded bg-(--ink-bg-0) p-0.5" role="group" aria-label="Viewport projection">
        {MODES.map(([id, label]) => (
          <button
            key={id}
            type="button"
            aria-pressed={mode === id}
            title={id === "TwoD" ? "Orthographic 2D editing (XY plane)" : "Perspective 3D"}
            className={`flex h-6 items-center rounded px-2 ${
              mode === id
                ? "bg-(--ink-bg-3) text-(--ink-text)"
                : "text-(--ink-text-dim) hover:text-(--ink-text)"
            }`}
            onClick={() => setMode(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* 2D snapping controls (only meaningful in ortho mode). */}
      {mode === "TwoD" && (
        <>
          <span className="text-(--ink-text-dim)">Snap</span>
          <SnapToggle
            label="Grid"
            title="Snap translate to the grid increment (world units)"
            enabled={gridSnapEnabled}
            onToggle={() => setGridSnapEnabled(!gridSnapEnabled)}
          >
            <NumberField
              value={gridSnapSize}
              min={0}
              step={0.05}
              onChange={setGridSnapSize}
              title="Grid snap increment (world units)"
            />
          </SnapToggle>
          <SnapToggle
            label="Pixel"
            title="Snap translate to whole pixels (1/PPU world units)"
            enabled={pixelSnapEnabled}
            onToggle={() => setPixelSnapEnabled(!pixelSnapEnabled)}
          >
            <NumberField
              value={pixelsPerUnit}
              min={1}
              step={1}
              onChange={setPixelsPerUnit}
              title="Pixels per unit (per-project)"
              suffix="PPU"
            />
          </SnapToggle>
        </>
      )}

      {/* Tool toggle + sculpt brush controls (perspective only; sculpting is a
          3D-terrain tool). */}
      {mode === "Perspective" && (
        <>
          {/* Shading view mode (Lit / Unlit / Wireframe), perspective only (R-P2). */}
          <div className="h-4 w-px bg-(--ink-border)" />
          <div
            className="flex items-center rounded bg-(--ink-bg-0) p-0.5"
            role="group"
            aria-label="Viewport shading"
          >
            {VIEW_MODES.map(([id, label, title]) => (
              <button
                key={id}
                type="button"
                aria-pressed={viewMode === id}
                title={title}
                className={`flex h-6 items-center rounded px-2 ${
                  viewMode === id
                    ? "bg-(--ink-bg-3) text-(--ink-text)"
                    : "text-(--ink-text-dim) hover:text-(--ink-text)"
                }`}
                onClick={() => setViewMode(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="h-4 w-px bg-(--ink-border)" />
          <div
            className="flex items-center rounded bg-(--ink-bg-0) p-0.5"
            role="group"
            aria-label="Viewport tool"
          >
            {TOOLS.map(([id, label, title]) => {
              // A streamed (.inf_terrain) terrain has no editable working set in
              // the document, so the brush tools are refused by the viewport
              // host. Say so up front rather than letting the stroke bounce
              // (P16.4a; write-back is P16.4b).
              const blocked = terrainStreamed && id === "Sculpt";
              return (
                <button
                  key={id}
                  type="button"
                  aria-pressed={toolMode === id}
                  disabled={blocked}
                  title={blocked ? STREAMED_TERRAIN_HINT : title}
                  className={`flex h-6 items-center rounded px-2 ${
                    toolMode === id
                      ? "bg-(--ink-bg-3) text-(--ink-text)"
                      : "text-(--ink-text-dim) hover:text-(--ink-text)"
                  } disabled:opacity-40`}
                  onClick={() => setToolMode(id)}
                >
                  {label}
                </button>
              );
            })}
          </div>
          {terrainStreamed && (
            <span
              className="rounded bg-(--ink-bg-0) px-2 py-0.5 text-(--ink-text-faint)"
              title={STREAMED_TERRAIN_HINT}
            >
              Streamed terrain
            </span>
          )}
          {toolMode === "Sculpt" && <SculptControls />}
          {toolMode === "Foliage" && <FoliageControls />}
        </>
      )}
    </div>
  );
}

/** Radius / density / kind / jitter / seed + erase toggle for the Foliage tool
 * (E-P6). Painting scatters the selected `Foliage` entity, or auto-creates one. */
function FoliageControls() {
  const radius = useViewportStore((s) => s.foliageRadius);
  const density = useViewportStore((s) => s.foliageDensity);
  const erase = useViewportStore((s) => s.foliageErase);
  const kind = useViewportStore((s) => s.foliageKind);
  const scaleJitter = useViewportStore((s) => s.foliageScaleJitter);
  const seed = useViewportStore((s) => s.foliageSeed);
  const setRadius = useViewportStore((s) => s.setFoliageRadius);
  const setDensity = useViewportStore((s) => s.setFoliageDensity);
  const setErase = useViewportStore((s) => s.setFoliageErase);
  const setKind = useViewportStore((s) => s.setFoliageKind);
  const setScaleJitter = useViewportStore((s) => s.setFoliageScaleJitter);
  const setSeed = useViewportStore((s) => s.setFoliageSeed);

  return (
    <>
      <button
        type="button"
        aria-pressed={erase}
        title="Erase: an LMB-drag removes foliage within the brush instead of placing it"
        className={`flex h-6 items-center rounded px-2 ${
          erase
            ? "bg-(--ink-accent) text-(--ink-bg-0)"
            : "bg-(--ink-bg-0) text-(--ink-text-dim) hover:text-(--ink-text)"
        }`}
        onClick={() => setErase(!erase)}
      >
        Erase
      </button>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Radius
        <NumberField
          value={Math.round(radius * 100) / 100}
          min={0.1}
          step={0.5}
          onChange={setRadius}
          title="Brush radius (world metres)"
          suffix="m"
        />
      </label>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Density
        <NumberField
          value={density}
          min={0}
          step={0.1}
          onChange={setDensity}
          title="Target instances per m² of brush area (before min-spacing rejection)"
        />
      </label>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Kind
        <NumberField
          value={kind}
          min={0}
          step={1}
          onChange={setKind}
          title="Palette slot new instances draw from (edit the palette in Details)"
        />
      </label>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Jitter
        <NumberField
          value={scaleJitter}
          min={0}
          step={0.05}
          onChange={setScaleJitter}
          title="± fractional scale spread per instance"
        />
      </label>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Seed
        <NumberField
          value={seed}
          min={0}
          step={1}
          onChange={setSeed}
          title="Deterministic scatter seed (same stroke reproduces identical instances)"
        />
      </label>
    </>
  );
}

/** Brush op / radius / strength / falloff for the Sculpt tool (P10.2b). */
function SculptControls() {
  const sculptOp = useViewportStore((s) => s.sculptOp);
  const sculptRadius = useViewportStore((s) => s.sculptRadius);
  const sculptStrength = useViewportStore((s) => s.sculptStrength);
  const sculptFalloff = useViewportStore((s) => s.sculptFalloff);
  const sculptPaintLayer = useViewportStore((s) => s.sculptPaintLayer);
  const setSculptOp = useViewportStore((s) => s.setSculptOp);
  const setSculptRadius = useViewportStore((s) => s.setSculptRadius);
  const setSculptStrength = useViewportStore((s) => s.setSculptStrength);
  const setSculptFalloff = useViewportStore((s) => s.setSculptFalloff);
  const setSculptPaintLayer = useViewportStore((s) => s.setSculptPaintLayer);
  const openErode = useShellStore((s) => s.setErodeOpen);

  return (
    <>
      <div className="flex items-center rounded bg-(--ink-bg-0) p-0.5" role="group" aria-label="Sculpt op">
        {SCULPT_OPS.map(([id, label]) => (
          <button
            key={id}
            type="button"
            aria-pressed={sculptOp === id}
            title={id === "Paint" ? "Paint splat layer weights (P10.4)" : `${label} brush`}
            className={`flex h-6 items-center rounded px-2 ${
              sculptOp === id
                ? "bg-(--ink-accent) text-(--ink-bg-0)"
                : "text-(--ink-text-dim) hover:text-(--ink-text)"
            }`}
            onClick={() => setSculptOp(id)}
          >
            {label}
          </button>
        ))}
      </div>
      {/* Splat layer picker (Paint op only): four swatches showing the layer
          colours; the selected layer is what the brush paints toward. */}
      {sculptOp === "Paint" && (
        <div
          className="flex items-center gap-1 rounded bg-(--ink-bg-0) p-0.5"
          role="group"
          aria-label="Splat layer"
        >
          {LAYER_SWATCHES.map(([color, name], i) => (
            <button
              key={name}
              type="button"
              aria-pressed={sculptPaintLayer === i}
              title={`Paint layer ${i + 1} (${name})`}
              onClick={() => setSculptPaintLayer(i)}
              className={`flex h-6 w-6 items-center justify-center rounded text-[10px] font-medium ${
                sculptPaintLayer === i
                  ? "ring-2 ring-(--ink-accent)"
                  : "ring-1 ring-(--ink-border) hover:ring-(--ink-text-dim)"
              }`}
              style={{ backgroundColor: color, color: "#111" }}
            >
              {i + 1}
            </button>
          ))}
        </div>
      )}
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Radius
        <NumberField
          value={Math.round(sculptRadius * 100) / 100}
          min={0.5}
          step={0.5}
          onChange={setSculptRadius}
          title="Brush radius (world metres) — also [ / ]"
          suffix="m"
        />
      </label>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Strength
        <NumberField
          value={sculptStrength}
          min={0}
          step={0.05}
          onChange={setSculptStrength}
          title="Per-dab strength (metres; blend fraction for Smooth/Flatten; flow rate for Paint)"
        />
      </label>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Falloff
        <select
          value={sculptFalloff}
          title="Brush falloff curve"
          onChange={(e) => setSculptFalloff(e.target.value as SculptFalloffDto)}
          className="h-6 rounded border border-(--ink-border) bg-(--ink-bg-0) px-1 text-(--ink-text)"
        >
          {SCULPT_FALLOFFS.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
      </label>
      <div className="h-4 w-px bg-(--ink-border)" />
      <button
        type="button"
        title="Bake hydraulic + thermal erosion onto the terrain (GPU compute; one undo step)"
        onClick={() => openErode(true)}
        className="flex h-6 items-center rounded border border-(--ink-border) px-2 text-(--ink-text-dim) hover:border-(--ink-accent) hover:text-(--ink-text)"
      >
        Erode…
      </button>
    </>
  );
}

function SnapToggle(props: {
  label: string;
  title: string;
  enabled: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        aria-pressed={props.enabled}
        title={props.title}
        className={`flex h-6 items-center rounded px-2 ${
          props.enabled
            ? "bg-(--ink-accent) text-(--ink-bg-0)"
            : "bg-(--ink-bg-0) text-(--ink-text-dim) hover:text-(--ink-text)"
        }`}
        onClick={props.onToggle}
      >
        {props.label}
      </button>
      {props.children}
    </div>
  );
}

function NumberField(props: {
  value: number;
  min: number;
  step: number;
  onChange: (v: number) => void;
  title: string;
  suffix?: string;
}) {
  return (
    <div className="flex items-center gap-0.5">
      <input
        type="number"
        min={props.min}
        step={props.step}
        value={props.value}
        title={props.title}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) props.onChange(v);
        }}
        className="h-6 w-14 rounded border border-(--ink-border) bg-(--ink-bg-0) px-1 text-(--ink-text) tabular-nums"
      />
      {props.suffix && <span className="text-(--ink-text-dim)">{props.suffix}</span>}
    </div>
  );
}
