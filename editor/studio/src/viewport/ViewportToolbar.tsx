/**
 * Viewport toolbar (P8.2c): the Perspective / 2D projection toggle and, in 2D
 * mode, the grid- and pixel-snapping controls. Rendered as a strip ABOVE the
 * native viewport hole (never over it — the native child window would occlude
 * HTML there; airspace rule). All state lives in `viewportStore`, which mirrors
 * each change to the native viewport over typed IPC.
 */
import { linearToCssHex } from "../lib/biomeColor";
import { terrain } from "../lib/ipc";
import { useViewportStore } from "../stores/viewportStore";
import { useShellStore } from "../stores/shellStore";
import type { SculptFalloffDto } from "../bindings/SculptFalloffDto";
import type { SculptOpDto } from "../bindings/SculptOpDto";
import type { ToolModeDto } from "../bindings/ToolModeDto";
import type { RiverReportDto } from "../bindings/RiverReportDto";
import type { WaterToolKindDto } from "../bindings/WaterToolKindDto";
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
  ["Biomes", "Biomes", "Tint terrain by painted biome; other geometry unlit"],
];

const TOOLS: [ToolModeDto, string, string][] = [
  ["Select", "Select", "Pick + transform (Q/W/E/R gizmos)"],
  ["Sculpt", "Sculpt", "Terrain height brush (perspective only)"],
  ["Foliage", "Foliage", "Scatter foliage onto the terrain (perspective only)"],
  ["Biome", "Biome", "Paint named biomes onto the terrain (perspective only)"],
  ["Water", "Water", "Place rivers and lakes against the terrain (perspective only)"],
];

/** River vs Lake, and what each gesture is. */
const WATER_KINDS: [WaterToolKindDto, string, string][] = [
  ["River", "River", "Click the ground to add a control point; the first click starts a river"],
  ["Lake", "Lake", "Drag a rectangle; the waterline preview shows where it lands"],
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

/** The *unassigned* biome's swatch (id 0, the eraser). Mirrors
 *  `inf_terrain::UNASSIGNED_BIOME_COLOR` — a neutral mid grey, stated here
 *  because id 0 is reserved and can never be a `BiomeDef` the backend sends. */
const UNASSIGNED_SWATCH = linearToCssHex([0.22, 0.22, 0.24, 1.0]);

/** Why the brush tools are off on a streamed terrain whose asset is read-only. */
const READONLY_TERRAIN_HINT =
  "This terrain's .inf_terrain asset is read-only, so sculpt and paint have nowhere to save to.";

/** The chip's tooltip while a streamed terrain is projected and clean. */
const STREAMED_TERRAIN_HINT =
  "This terrain streams from a .inf_terrain asset. Sculpt and paint page tiles in as you brush; " +
  "Ctrl+S writes them back into the asset.";

/** …and once it carries edits that only an explicit save will persist. */
const UNSAVED_TERRAIN_HINT =
  "Unsaved terrain edits. Terrain is written into its .inf_terrain asset by Ctrl+S only — " +
  "autosave does not save terrain, so a crash would lose these.";

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
  const terrainEditable = useViewportStore((s) => s.terrainEditable);
  const terrainUnsavedEdits = useViewportStore((s) => s.terrainUnsavedEdits);
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
              // A streamed terrain IS editable (P16.4b) — tiles page into the
              // document as you brush and Ctrl+S writes them back. The only
              // refusal left is an asset the editor cannot write, which the
              // viewport host also rejects; say so up front rather than letting
              // the stroke bounce.
              const blocked =
                terrainStreamed && !terrainEditable && (id === "Sculpt" || id === "Biome");
              return (
                <button
                  key={id}
                  type="button"
                  aria-pressed={toolMode === id}
                  disabled={blocked}
                  title={blocked ? READONLY_TERRAIN_HINT : title}
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
              className={`rounded px-2 py-0.5 ${
                terrainUnsavedEdits
                  ? "bg-(--ink-bg-0) text-(--ink-warning)"
                  : "bg-(--ink-bg-0) text-(--ink-text-faint)"
              }`}
              title={
                terrainUnsavedEdits
                  ? UNSAVED_TERRAIN_HINT
                  : terrainEditable
                    ? STREAMED_TERRAIN_HINT
                    : READONLY_TERRAIN_HINT
              }
            >
              {terrainUnsavedEdits
                ? "Unsaved terrain edits"
                : terrainEditable
                  ? "Streamed terrain"
                  : "Streamed terrain (read-only)"}
            </span>
          )}
          {toolMode === "Sculpt" && <SculptControls />}
          {toolMode === "Foliage" && <FoliageControls />}
          {toolMode === "Biome" && <BiomeControls />}
          {toolMode === "Water" && <WaterControls />}
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

/**
 * Biome picker / set picker / radius / strength / falloff for the Biome tool
 * (P19.2).
 *
 * The swatches are the biome definitions of whichever `.inf_biomes` the terrain
 * binds, so with nothing bound there is nothing to paint — that case renders a
 * hint plus the set picker rather than an empty control group.
 */
function BiomeControls() {
  const radius = useViewportStore((s) => s.biomeRadius);
  const strength = useViewportStore((s) => s.biomeStrength);
  const falloff = useViewportStore((s) => s.biomeFalloff);
  const biomeId = useViewportStore((s) => s.biomeId);
  const biomes = useViewportStore((s) => s.terrainBiomes);
  const setRadius = useViewportStore((s) => s.setBiomeRadius);
  const setStrength = useViewportStore((s) => s.setBiomeStrength);
  const setFalloff = useViewportStore((s) => s.setBiomeFalloff);
  const setBiomeId = useViewportStore((s) => s.setBiomeId);
  const refreshBiomes = useViewportStore((s) => s.refreshBiomes);

  const rebind = async (asset: string) => {
    if (!biomes) return;
    try {
      await terrain.setBiomeSet(biomes.entity, asset === "" ? null : asset);
    } catch (e) {
      console.error("terrain.setBiomeSet failed", e);
    }
    // Re-read either way: a failed bind still leaves the picker showing a value
    // the document never took.
    await refreshBiomes();
  };

  return (
    <>
      {/* Set picker. Present even with nothing bound — it is the way to bind. */}
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Set
        <select
          value={biomes?.biome_set ?? ""}
          disabled={!biomes}
          title="The .inf_biomes asset this terrain's painted biome ids name"
          onChange={(e) => void rebind(e.target.value)}
          className="h-6 max-w-40 rounded border border-(--ink-border) bg-(--ink-bg-0) px-1 text-(--ink-text) disabled:opacity-40"
        >
          <option value="">(none)</option>
          {(biomes?.available ?? []).map(([id, name]) => (
            <option key={id} value={id}>
              {name}
            </option>
          ))}
        </select>
      </label>

      {biomes && biomes.biomes.length > 0 ? (
        <div
          className="flex items-center gap-1 rounded bg-(--ink-bg-0) p-0.5"
          role="group"
          aria-label="Biome"
        >
          {/* The eraser is id 0 — reserved, never a definition, so it is stated
              here rather than coming from the set. */}
          <BiomeSwatch
            color={UNASSIGNED_SWATCH}
            label="—"
            title="Unassigned (erase): paints the reserved id 0 back over the brush"
            selected={biomeId === 0}
            onSelect={() => setBiomeId(0)}
          />
          {biomes.biomes.map((b) => (
            <BiomeSwatch
              key={b.id}
              color={linearToCssHex(b.color)}
              label={String(b.id)}
              title={`${b.name} (id ${b.id})`}
              selected={biomeId === b.id}
              onSelect={() => setBiomeId(b.id)}
            />
          ))}
        </div>
      ) : (
        <span className="text-(--ink-text-faint)">
          {biomes ? "Bind a Biome Set to paint" : "No terrain in this level"}
        </span>
      )}

      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Radius
        <NumberField
          value={Math.round(radius * 100) / 100}
          min={0.5}
          step={0.5}
          onChange={setRadius}
          title="Brush radius (world metres)"
          suffix="m"
        />
      </label>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Strength
        <NumberField
          value={strength}
          min={0}
          step={0.05}
          onChange={setStrength}
          title="Where the hard biome boundary falls, 0..1 — 1 stamps the whole disk"
        />
      </label>
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Falloff
        <select
          value={falloff}
          title="The curve the boundary contour is taken from"
          onChange={(e) => setFalloff(e.target.value as SculptFalloffDto)}
          className="h-6 rounded border border-(--ink-border) bg-(--ink-bg-0) px-1 text-(--ink-text)"
        >
          {SCULPT_FALLOFFS.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
      </label>
    </>
  );
}

/**
 * River / Lake picker plus the dimensions a NEW body takes (P20.4).
 *
 * There is deliberately no *level* field: the level comes from where you click —
 * the painted biome's `water_hint` when it has one, otherwise the ground — and
 * `Level +` shifts it. A level typed here would be a number the author has to
 * keep re-deriving as they move around the terrain.
 */
function WaterControls() {
  const kind = useViewportStore((s) => s.waterKind);
  const width = useViewportStore((s) => s.waterWidth);
  const depth = useViewportStore((s) => s.waterDepth);
  const flow = useViewportStore((s) => s.waterFlow);
  const offset = useViewportStore((s) => s.waterLevelOffset);
  const setKind = useViewportStore((s) => s.setWaterKind);
  const setWidth = useViewportStore((s) => s.setWaterWidth);
  const setDepth = useViewportStore((s) => s.setWaterDepth);
  const setFlow = useViewportStore((s) => s.setWaterFlow);
  const setOffset = useViewportStore((s) => s.setWaterLevelOffset);

  return (
    <>
      <div className="flex items-center rounded bg-(--ink-bg-0) p-0.5" role="group" aria-label="Water body">
        {WATER_KINDS.map(([id, label, title]) => (
          <button
            key={id}
            type="button"
            aria-pressed={kind === id}
            title={title}
            className={`flex h-6 items-center rounded px-2 ${
              kind === id
                ? "bg-(--ink-accent) text-(--ink-bg-0)"
                : "text-(--ink-text-dim) hover:text-(--ink-text)"
            }`}
            onClick={() => setKind(id)}
          >
            {label}
          </button>
        ))}
      </div>
      {kind === "River" && (
        <>
          <label className="flex items-center gap-1 text-(--ink-text-dim)">
            Width
            <NumberField
              value={width}
              min={0}
              step={0.5}
              onChange={setWidth}
              title="Full width of a new river (world metres)"
              suffix="m"
            />
          </label>
          <label className="flex items-center gap-1 text-(--ink-text-dim)">
            Depth
            <NumberField
              value={depth}
              min={0}
              step={0.25}
              onChange={setDepth}
              title="Depth to the bed of a new river (world metres) — drives the tint and the shallow foam band"
              suffix="m"
            />
          </label>
          <label className="flex items-center gap-1 text-(--ink-text-dim)">
            Flow
            <NumberField
              value={flow}
              min={-100}
              step={0.5}
              onChange={setFlow}
              title="Surface flow speed (m/s). NEGATIVE reverses the river without re-authoring its points."
              suffix="m/s"
            />
          </label>
        </>
      )}
      <label className="flex items-center gap-1 text-(--ink-text-dim)">
        Level +
        <NumberField
          value={offset}
          min={-1000}
          step={0.5}
          onChange={setOffset}
          title="Added to the level the click suggests (the biome water hint, else the ground). Negative sinks it."
          suffix="m"
        />
      </label>
      {kind === "River" && <RiverVerdict />}
    </>
  );
}

/**
 * The selected river's verdict (P20.4): the two COOK advisories re-run — so the
 * toolbar says what the build will say — plus the terrain-aware bed conflicts
 * only the editor can make (a cook has no heightfield).
 *
 * Silent when nothing river-shaped is selected: "no river selected" and "this
 * river is fine" must not look the same.
 */
function RiverVerdict() {
  const report = useViewportStore((s) => s.waterRiverReport);
  const refresh = useViewportStore((s) => s.refreshRiverReport);

  if (!report) {
    return (
      <span className="text-(--ink-text-faint)" title="Select a river to see its verdict">
        No river selected
      </span>
    );
  }
  const issues = riverIssues(report);
  const partial =
    report.sampled_frames < report.total_frames
      ? ` — terrain answered for ${report.sampled_frames}/${report.total_frames} frames`
      : "";
  // One tooltip line per fact, joined — a template literal with embedded
  // newlines would put the file's own indentation inside the tooltip.
  const title = [
    `${report.length_m.toFixed(0)} m, falls ${report.fall_m.toFixed(1)} m${partial}`,
    `${report.points} control points`,
    ...(issues.length ? issues : ["No hydrology issues."]),
    "Click to re-check.",
  ].join("\n");
  return (
    <button
      type="button"
      onClick={() => void refresh()}
      title={title}
      className={`flex h-6 items-center rounded px-2 ${
        issues.length
          ? "bg-(--ink-bg-0) text-(--ink-warning)"
          : "bg-(--ink-bg-0) text-(--ink-text-faint)"
      }`}
    >
      {issues.length
        ? `⚠ ${issues.length} issue${issues.length > 1 ? "s" : ""}: ${issues[0]}`
        : `✓ ${report.length_m.toFixed(0)} m, falls ${report.fall_m.toFixed(1)} m`}
    </button>
  );
}

/**
 * One human line per hydrology problem, in the order an author would fix them:
 * the ground first (the river is invisible or floating), then the surface, then
 * the authored bed.
 *
 * Exported for the store tests, which pin the wording of the two cook advisories
 * against the messages the cook itself emits.
 */
export function riverIssues(r: RiverReportDto): string[] {
  const out: string[] = [];
  const buried = r.bed_conflicts.filter((c) => c.issue === "buried");
  const perched = r.bed_conflicts.filter((c) => c.issue === "perched");
  const worst = (
    xs: { worst_m: number; from_s: number; to_s: number }[],
  ): { worst_m: number; from_s: number; to_s: number } =>
    xs.reduce((a, b) => (b.worst_m > a.worst_m ? b : a));
  if (buried.length) {
    const w = worst(buried);
    out.push(
      `buried in the ground for ${(w.to_s - w.from_s).toFixed(0)} m (worst ${w.worst_m.toFixed(1)} m)`,
    );
  }
  if (perched.length) {
    const w = worst(perched);
    out.push(
      `perched over a drop for ${(w.to_s - w.from_s).toFixed(0)} m (worst ${w.worst_m.toFixed(1)} m)`,
    );
  }
  if (r.surface_climbs.length) {
    const rise = r.surface_climbs.reduce((a, c) => a + c.rise_m, 0);
    out.push(`flows UPHILL — climbs ${rise.toFixed(1)} m (the cook will say so)`);
  }
  if (r.bed_climbs.length) {
    const rise = r.bed_climbs.reduce((a, c) => a + c.rise_m, 0);
    out.push(`its BED climbs ${rise.toFixed(1)} m — a basin (the cook will say so)`);
  }
  return out;
}

/** One biome swatch in the picker (also used for the id-0 eraser). */
function BiomeSwatch(props: {
  color: string;
  label: string;
  title: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={props.selected}
      title={props.title}
      onClick={props.onSelect}
      className={`flex h-6 min-w-6 items-center justify-center rounded px-1 text-[10px] font-medium ${
        props.selected
          ? "ring-2 ring-(--ink-accent)"
          : "ring-1 ring-(--ink-border) hover:ring-(--ink-text-dim)"
      }`}
      style={{ backgroundColor: props.color, color: "#111" }}
    >
      {props.label}
    </button>
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
