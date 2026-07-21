/**
 * Terrain Erosion bake dialog (P10.3b). Opened from the Sculpt toolbar's
 * "Erode…" button (terrain is the sculpt context). Edits an `ErosionParamsDto`
 * (defaults mirror `inf_terrain::ErosionParams::default`) + a step count, then
 * runs `terrain_erode` on the selected (or first) Terrain entity. The bake runs
 * the GPU compute pipeline (CPU reference fallback with no adapter) and commits
 * ONE undoable height delta — so Ctrl+Z reverts the whole erosion.
 *
 * DETERMINISM NOTE: eroded terrain is DATA (the height delta is stored in the
 * level), so a saved level reloads byte-exact on any machine. Only the bake
 * ACTION varies by GPU adapter (GPU f32 vs the partly-f64 CPU reference); the
 * result line flags whether the GPU ran.
 *
 * Mirrors SortingLayersDialog: driven by `shellStore.erodeOpen`, hides the
 * native viewport while open (airspace rule).
 */
import { useEffect, useMemo, useState } from "react";
import { Waves, X } from "lucide-react";

import type { ErosionParamsDto } from "../bindings/ErosionParamsDto";
import type { ErosionReportDto } from "../bindings/ErosionReportDto";
import { terrain as terrainIpc } from "../lib/ipc";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useSceneStore } from "../stores/sceneStore";
import { useShellStore } from "../stores/shellStore";

/** Defaults mirroring `inf_terrain::ErosionParams::default`. */
const DEFAULTS: ErosionParamsDto = {
  dt: 0.02,
  rain_rate: 0.02,
  rain_variation: 0.0,
  rain_seed: 0,
  rain_frequency: 0.02,
  rain_octaves: 3,
  evaporation: 0.015,
  gravity: 9.81,
  pipe_area: 1.0,
  sediment_capacity: 1.0,
  dissolving: 0.5,
  deposition: 1.0,
  min_tilt: 0.01,
  max_erode_depth: 0.5,
  thermal_talus_deg: 33.0,
  thermal_rate: 0.5,
  max_thermal_depth: 0.5,
  thermal_every: 1,
};

type Field = { key: keyof ErosionParamsDto; label: string; step: number };

const WATER_FIELDS: Field[] = [
  { key: "rain_rate", label: "Rain rate", step: 0.005 },
  { key: "rain_variation", label: "Rain variation", step: 0.05 },
  { key: "evaporation", label: "Evaporation", step: 0.005 },
  { key: "sediment_capacity", label: "Sediment capacity", step: 0.1 },
  { key: "dissolving", label: "Dissolving (Ks)", step: 0.05 },
  { key: "deposition", label: "Deposition (Kd)", step: 0.05 },
  { key: "pipe_area", label: "Pipe area", step: 0.5 },
  { key: "max_erode_depth", label: "Max erode / step", step: 0.05 },
];

const THERMAL_FIELDS: Field[] = [
  { key: "thermal_talus_deg", label: "Talus angle (deg)", step: 1 },
  { key: "thermal_rate", label: "Thermal rate", step: 0.05 },
  { key: "max_thermal_depth", label: "Max thermal / step", step: 0.05 },
  { key: "thermal_every", label: "Thermal every N", step: 1 },
];

export default function ErodeDialog() {
  const open = useShellStore((s) => s.erodeOpen);
  const setOpen = useShellStore((s) => s.setErodeOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);

  const nodes = useSceneStore((s) => s.nodes);
  const selection = useSceneStore((s) => s.selection);

  const [params, setParams] = useState<ErosionParamsDto>(DEFAULTS);
  const [steps, setSteps] = useState(50);
  const [baking, setBaking] = useState(false);
  const [result, setResult] = useState<ErosionReportDto | null>(null);
  const [error, setError] = useState<string | null>(null);

  useViewportOverlay(open);

  // The terrain to erode: a selected Terrain, else the first Terrain in scene.
  const target = useMemo(() => {
    const sel = selection.find((g) => nodes[g]?.kind === "Terrain");
    if (sel) return { guid: sel, name: nodes[sel]?.name ?? "Terrain" };
    const first = Object.values(nodes).find((n) => n.kind === "Terrain");
    return first ? { guid: first.guid, name: first.name } : null;
  }, [nodes, selection]);

  // Clear transient state on close (not in an open-effect — the repo bans
  // synchronous setState in effects), so a re-open starts fresh.
  const close = () => {
    setResult(null);
    setError(null);
    setOpen(false);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !baking) close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, baking]);

  if (!open) return null;

  const setField = (key: keyof ErosionParamsDto, v: number) =>
    setParams((p) => ({ ...p, [key]: v }));

  const bake = async () => {
    if (!target) return;
    setBaking(true);
    setError(null);
    setResult(null);
    pushStatus(`Eroding ${target.name} (${steps} steps)…`, 60000);
    try {
      const report = await terrainIpc.erode(target.guid, params, steps);
      setResult(report);
      pushStatus(
        `Erosion baked: ${report.cells_changed.toLocaleString()} cells changed` +
          (report.used_gpu ? " (GPU)" : " (CPU)"),
      );
    } catch (e) {
      setError(String(e));
      pushStatus(`Erosion failed: ${String(e)}`);
    } finally {
      setBaking(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-16"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget && !baking) close();
      }}
    >
      <div
        className="flex max-h-[80vh] w-[520px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center gap-2 border-b border-(--ink-border) px-3 py-2">
          <Waves size={15} className="text-(--ink-accent)" />
          <span className="flex-1 font-semibold">Erode Terrain</span>
          <button
            aria-label="Close dialog"
            disabled={baking}
            className="rounded p-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text) disabled:opacity-40"
            onClick={close}
          >
            <X size={14} />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-3">
          <div className="mb-2 text-xs text-(--ink-text-dim)">
            {target ? (
              <>
                Target: <span className="text-(--ink-text)">{target.name}</span>
              </>
            ) : (
              <span className="text-(--ink-error)">
                No terrain in the scene — spawn or select a Terrain first.
              </span>
            )}
          </div>

          {/* Steps */}
          <div className="mb-3 flex items-center gap-2">
            <label className="w-40 text-xs text-(--ink-text-dim)">Simulation steps</label>
            <input
              type="number"
              min={1}
              step={5}
              value={steps}
              onChange={(e) => setSteps(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
              className="w-24 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 text-right text-xs tabular-nums outline-none focus:border-(--ink-accent)"
            />
          </div>

          <ParamGroup title="Hydraulic (water)" fields={WATER_FIELDS} params={params} onSet={setField} />
          <ParamGroup title="Thermal (talus)" fields={THERMAL_FIELDS} params={params} onSet={setField} />

          {error && <div className="mt-2 text-xs text-(--ink-error)">{error}</div>}
          {result && (
            <div className="mt-3 rounded border border-(--ink-border) bg-(--ink-bg-0) p-2 text-xs">
              <div className="mb-1 font-semibold text-(--ink-text)">Result</div>
              <div className="text-(--ink-text-dim)">
                Cells changed: <span className="text-(--ink-text)">{result.cells_changed.toLocaleString()}</span>
              </div>
              <div className="text-(--ink-text-dim)">
                Mass delta:{" "}
                <span className="text-(--ink-text)">{result.mass_delta.toFixed(2)} m³</span>
              </div>
              <div className="text-(--ink-text-dim)">
                Sediment moved:{" "}
                <span className="text-(--ink-text)">
                  {result.sediment_moved != null ? `${result.sediment_moved.toFixed(2)} m³` : "—"}
                </span>
              </div>
              <div className="text-(--ink-text-dim)">
                Path: <span className="text-(--ink-text)">{result.used_gpu ? "GPU compute" : "CPU reference"}</span>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-(--ink-border) px-3 py-2">
          <button
            onClick={() => setParams(DEFAULTS)}
            disabled={baking}
            className="rounded px-2 py-1 text-xs text-(--ink-text-dim) hover:bg-(--ink-bg-3) disabled:opacity-40"
          >
            Reset defaults
          </button>
          <div className="flex gap-2">
            <button
              onClick={close}
              disabled={baking}
              className="rounded px-3 py-1 text-xs text-(--ink-text-dim) hover:bg-(--ink-bg-3) disabled:opacity-40"
            >
              Close
            </button>
            <button
              onClick={() => void bake()}
              disabled={baking || !target}
              className="rounded bg-(--ink-accent) px-3 py-1 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
            >
              {baking ? "Eroding…" : "Erode"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ParamGroup(props: {
  title: string;
  fields: Field[];
  params: ErosionParamsDto;
  onSet: (key: keyof ErosionParamsDto, v: number) => void;
}) {
  return (
    <div className="mb-3">
      <div className="mb-1 text-xs font-semibold text-(--ink-text-faint)">{props.title}</div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1">
        {props.fields.map((f) => (
          <label key={f.key} className="flex items-center justify-between gap-2 text-xs text-(--ink-text-dim)">
            <span className="truncate">{f.label}</span>
            <input
              type="number"
              step={f.step}
              value={props.params[f.key]}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v)) props.onSet(f.key, v);
              }}
              className="w-20 shrink-0 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1 text-right tabular-nums outline-none focus:border-(--ink-accent)"
            />
          </label>
        ))}
      </div>
    </div>
  );
}
