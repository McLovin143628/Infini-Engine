/**
 * GIS Import wizard (IB-3) — "point at a Shapefile, get a city".
 *
 * Three steps, one dialog:
 *   1. PICK       a `.shp` or `.geojson` (native file picker).
 *   2. CONFIGURE  what the layer becomes — the target kind, the CRS the `.prj`
 *                 resolved to (or the one the author states when it could not),
 *                 the **entity cap** with a live note of what it will drop, and
 *                 the per-target extras: a road surface, a land-cover paint, a
 *                 footprint bake.
 *   3. DONE       exactly what was created and what was not, with the plan's
 *                 digest — the same number `inf gis plan` prints.
 *
 * The wizard makes no import decision: every one of them is
 * `inf_gis::import` in Ring 0, which is also what the CLI calls. What lives
 * here is the author's answers and the report.
 *
 * UNITS: every length that crosses IPC is SI metres (the units doctrine).
 *
 * AIRSPACE: the native viewport draws over the webview, so the dialog holds a
 * `useViewportOverlay` acquisition for its whole lifetime — the guard every
 * other shell overlay uses.
 */
import { useEffect, useState } from "react";
import { AlertTriangle, Globe2, X } from "lucide-react";

import { useViewportOverlay } from "../lib/viewportOverlay";
import { useShellStore } from "../stores/shellStore";
import {
  capNote,
  describeCrs,
  gisSettingsIssue,
  GIS_ISLAND_MAX_ENTITIES,
  kindsFor,
  useGisImportStore,
} from "../stores/gisImportStore";

/** Vector containers the readers accept. */
const VECTOR_EXTENSIONS = ["shp", "geojson", "json"];

export default function GisImportDialog() {
  const open = useShellStore((s) => s.gisImportOpen);
  const setOpen = useShellStore((s) => s.setGisImportOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);

  const step = useGisImportStore((s) => s.step);
  const probe = useGisImportStore((s) => s.probe);
  const settings = useGisImportStore((s) => s.settings);
  const result = useGisImportStore((s) => s.result);
  const error = useGisImportStore((s) => s.error);
  const busy = useGisImportStore((s) => s.busy);
  const [crsDraft, setCrsDraft] = useState("");

  useViewportOverlay(open);

  const close = () => {
    useGisImportStore.getState().reset();
    setOpen(false);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, busy]);

  if (!open) return null;

  const pick = async () => {
    const { open: openDialog } = await import("@tauri-apps/plugin-dialog");
    const picked = await openDialog({
      multiple: false,
      filters: [{ name: "Vector data", extensions: VECTOR_EXTENSIONS }],
    });
    if (!picked || Array.isArray(picked)) return;
    await useGisImportStore.getState().pick(picked);
  };

  const start = async () => {
    await useGisImportStore.getState().start();
    const r = useGisImportStore.getState().result;
    if (r) pushStatus(r.summary, 12000);
  };

  const issue = gisSettingsIssue(probe, settings);
  const cap = capNote(probe, settings);
  const patch = useGisImportStore.getState().patchSettings;

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-16"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget && !busy) close();
      }}
    >
      <div
        className="flex max-h-[80vh] w-[600px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center gap-2 border-b border-(--ink-border) px-3 py-2">
          <Globe2 size={15} className="text-(--ink-accent)" />
          <span className="flex-1 font-semibold">Import GIS Data</span>
          <button
            aria-label="Close dialog"
            disabled={busy}
            className="rounded p-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text) disabled:opacity-40"
            onClick={close}
          >
            <X size={14} />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-3 text-xs">
          {/* ── 1. pick ─────────────────────────────────────────────────── */}
          {step === "pick" && (
            <div className="flex flex-col items-center gap-3 py-8 text-center">
              <Globe2 size={40} className="text-(--ink-text-faint)" />
              <div className="text-(--ink-text-dim)">
                Choose a Shapefile (<code>.shp</code>, with its <code>.dbf</code> and{" "}
                <code>.prj</code> beside it) or a GeoJSON.
                <br />
                Roads, watercourses, land cover, building footprints and parcels all
                come in through this one door.
              </div>
              <button
                onClick={() => void pick()}
                disabled={busy}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              >
                {busy ? "Reading…" : "Choose Vector File…"}
              </button>
              {error && <CrsRetry error={error} draft={crsDraft} setDraft={setCrsDraft} />}
            </div>
          )}

          {/* ── 2. configure ────────────────────────────────────────────── */}
          {step === "configure" && probe && settings && (
            <>
              <Section title="Source">
                <Row label="Layer" value={probe.layer_name} />
                <Row
                  label="Features"
                  value={`${probe.features} (${probe.points} points, ${probe.polylines} lines, ${probe.polygons} areas)`}
                />
                <Row label="Coordinate system" value={describeCrs(probe)} />
                {probe.crs.vertical_unit_m !== 1 && (
                  <Row
                    label="Vertical unit"
                    value={`${probe.crs.vertical_unit_m} m per unit`}
                  />
                )}
                {probe.centre_lat !== null && probe.centre_lon !== null && (
                  <Row
                    label="Centre"
                    value={`${probe.centre_lat.toFixed(5)}, ${probe.centre_lon.toFixed(5)}`}
                  />
                )}
                <Row
                  label="Level anchor"
                  value={probe.level_anchor_crs ?? "none — set one in World Settings"}
                />
                {probe.suggested_anchor_epsg !== null &&
                  probe.level_anchor_crs !== `EPSG:${probe.suggested_anchor_epsg}` && (
                    <Row
                      label="Suggested anchor"
                      value={`EPSG:${probe.suggested_anchor_epsg}`}
                    />
                  )}
              </Section>

              <Section title="Fields">
                <div className="max-h-28 overflow-auto">
                  {probe.fields.length === 0 && (
                    <div className="text-(--ink-text-faint)">
                      this layer carries no attribute table
                    </div>
                  )}
                  {probe.fields.map((f) => (
                    <Row
                      key={f.name}
                      label={f.name}
                      value={`${f.present} set${f.numeric > 0 ? `, ${f.numeric} numeric` : ""}${
                        f.sample ? ` — ${f.sample}` : ""
                      }`}
                    />
                  ))}
                </div>
              </Section>

              <Section title="Target">
                <SelectRow
                  label="Import as"
                  value={settings.kind}
                  options={kindsFor(probe.dominant_kind).map((k) => [k, k])}
                  onChange={(v) => patch({ kind: v, road_surface: v === "roads" })}
                />
                <NumberRow
                  label="Entity cap"
                  value={settings.max_entities}
                  step={1024}
                  min={1}
                  integer
                  onChange={(v) =>
                    patch({ max_entities: Math.min(v, GIS_ISLAND_MAX_ENTITIES) })
                  }
                />
                <NumberRow
                  label="Skip features shorter than (m)"
                  value={settings.min_length_m}
                  step={1}
                  min={0}
                  onChange={(v) => patch({ min_length_m: v })}
                />
                {settings.kind === "streams" && (
                  <CheckRow
                    label="Reverse flow direction"
                    checked={settings.reverse_flow}
                    onChange={(v) => patch({ reverse_flow: v })}
                  />
                )}
                {cap && (
                  <div className="mt-1 flex items-start gap-1 text-(--ink-warn)">
                    <AlertTriangle size={12} className="mt-0.5 shrink-0" />
                    <span>{cap}</span>
                  </div>
                )}
              </Section>

              {settings.kind === "roads" && (
                <Section title="Road surface">
                  <CheckRow
                    label="Build a drivable surface (.inf_mesh)"
                    checked={settings.road_surface}
                    onChange={(v) => patch({ road_surface: v })}
                  />
                  <NumberRow
                    label="Lift above the ground (m)"
                    value={settings.road_lift_m}
                    step={0.01}
                    min={0}
                    onChange={(v) => patch({ road_lift_m: v })}
                  />
                  <NumberRow
                    label="Ground sample step (m)"
                    value={settings.road_ground_step_m}
                    step={0.5}
                    min={0.1}
                    onChange={(v) => patch({ road_ground_step_m: v })}
                  />
                </Section>
              )}

              {settings.kind === "biomes" && (
                <Section title="Land cover">
                  <label className="flex items-center justify-between gap-2 py-0.5 text-(--ink-text-dim)">
                    <span className="truncate">Paint into terrain (entity id)</span>
                    <input
                      value={settings.biome_terrain}
                      placeholder="paste the terrain's id"
                      onChange={(e) => patch({ biome_terrain: e.target.value })}
                      className="w-56 shrink-0 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1 outline-none focus:border-(--ink-accent)"
                    />
                  </label>
                  <SelectRow
                    label="Class attribute"
                    value={settings.biome_attribute}
                    options={[
                      ["", "(probe the known spellings)"],
                      ...probe.fields.map((f) => [f.name, f.name] as [string, string]),
                    ]}
                    onChange={(v) => patch({ biome_attribute: v })}
                  />
                  <NumberRow
                    label="Classes (numeric layers)"
                    value={settings.biome_classes}
                    step={1}
                    min={1}
                    integer
                    onChange={(v) => patch({ biome_classes: v })}
                  />
                </Section>
              )}

              {(settings.kind === "buildings" || settings.kind === "parcels") && (
                <Section title="Buildings">
                  <CheckRow
                    label="Build each footprint into geometry"
                    checked={settings.buildings}
                    onChange={(v) => patch({ buildings: v })}
                  />
                  <NumberRow
                    label="Building cap"
                    value={settings.max_buildings}
                    step={64}
                    min={1}
                    integer
                    onChange={(v) => patch({ max_buildings: v })}
                  />
                  <CheckRow
                    label="Furnish interiors"
                    checked={settings.furnish}
                    onChange={(v) => patch({ furnish: v })}
                  />
                </Section>
              )}

              {probe.advisories.length > 0 && (
                <Section title="Advisories">
                  {probe.advisories.map((a, i) => (
                    <div key={i} className="flex items-start gap-1 py-0.5 text-(--ink-warn)">
                      <AlertTriangle size={12} className="mt-0.5 shrink-0" />
                      <span>{a}</span>
                    </div>
                  ))}
                </Section>
              )}
            </>
          )}

          {/* ── 3. done ─────────────────────────────────────────────────── */}
          {step === "done" && result && (
            <>
              <Section title="Imported">
                <Row label="Layer" value={result.layer_name} />
                <Row label="Coordinate system" value={result.crs} />
                <Row label="Entities" value={String(result.spawned)} />
                {result.too_short > 0 && (
                  <Row label="Too short" value={String(result.too_short)} />
                )}
                {result.unusable > 0 && (
                  <Row label="Unusable" value={String(result.unusable)} />
                )}
                {result.truncated > 0 && (
                  <Row
                    label="NOT imported"
                    value={`${result.truncated} (cap ${result.cap})`}
                  />
                )}
                <Row label="Plan digest" value={result.digest} />
              </Section>
              {result.road_summary && (
                <Section title="Road surface">
                  <div className="text-(--ink-text-dim)">{result.road_summary}</div>
                </Section>
              )}
              {result.biome_summary && (
                <Section title="Land cover">
                  <div className="text-(--ink-text-dim)">{result.biome_summary}</div>
                </Section>
              )}
              {result.building_summary && (
                <Section title="Buildings">
                  <div className="text-(--ink-text-dim)">{result.building_summary}</div>
                </Section>
              )}
              {result.advisories.length > 0 && (
                <Section title="Advisories">
                  {result.advisories.map((a, i) => (
                    <div key={i} className="flex items-start gap-1 py-0.5 text-(--ink-warn)">
                      <AlertTriangle size={12} className="mt-0.5 shrink-0" />
                      <span>{a}</span>
                    </div>
                  ))}
                </Section>
              )}
            </>
          )}

          {error && step !== "pick" && (
            <div className="mt-2 flex items-start gap-1 text-(--ink-error)">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 border-t border-(--ink-border) px-3 py-2">
          {step === "configure" && issue && (
            <span className="flex-1 truncate text-(--ink-warn)" title={issue}>
              {issue}
            </span>
          )}
          {!(step === "configure" && issue) && <span className="flex-1" />}
          {step === "configure" && (
            <>
              <button
                className="rounded px-3 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                onClick={() => useGisImportStore.getState().reset()}
              >
                Choose another file
              </button>
              <button
                disabled={!!issue || busy}
                onClick={() => void start()}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              >
                {busy ? "Importing…" : "Import"}
              </button>
            </>
          )}
          {step === "done" && (
            <>
              <button
                className="rounded px-3 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                onClick={() => useGisImportStore.getState().back()}
              >
                Back
              </button>
              <button
                onClick={close}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)"
              >
                Done
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * The escape hatch for a source whose `.prj` this engine could not resolve —
 * the refusal names the remedy, and this is where the author applies it.
 */
function CrsRetry(props: {
  error: string;
  draft: string;
  setDraft: (v: string) => void;
}) {
  return (
    <div className="w-full rounded border border-(--ink-border) bg-(--ink-bg-0) p-2 text-left">
      <div className="flex items-start gap-1 text-(--ink-error)">
        <AlertTriangle size={12} className="mt-0.5 shrink-0" />
        <span>{props.error}</span>
      </div>
      <label className="mt-2 flex items-center gap-2 text-(--ink-text-dim)">
        <span className="shrink-0">Source CRS</span>
        <input
          value={props.draft}
          placeholder="EPSG:26910, or a proj4 string"
          onChange={(e) => props.setDraft(e.target.value)}
          className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1 outline-none focus:border-(--ink-accent)"
        />
        <button
          disabled={props.draft.trim() === ""}
          onClick={() => void useGisImportStore.getState().restateCrs(props.draft.trim())}
          className="shrink-0 rounded bg-(--ink-accent) px-2 py-1 text-(--ink-text-onaccent) disabled:opacity-40"
        >
          Retry
        </button>
      </label>
    </div>
  );
}

// ── the local row kit (mirrors TerrainImportDialog's) ───────────────────────

function Section(props: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <div className="mb-1 font-semibold text-(--ink-text-faint)">{props.title}</div>
      <div className="rounded border border-(--ink-border) bg-(--ink-bg-0) p-2">
        {props.children}
      </div>
    </div>
  );
}

function Row(props: { label: string; value: string; title?: string }) {
  return (
    <div className="flex items-center justify-between gap-2 py-0.5 text-(--ink-text-dim)">
      <span className="shrink-0">{props.label}</span>
      <span className="truncate text-(--ink-text)" title={props.title ?? props.value}>
        {props.value}
      </span>
    </div>
  );
}

function NumberRow(props: {
  label: string;
  value: number;
  step: number;
  min?: number;
  integer?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 py-0.5 text-(--ink-text-dim)">
      <span className="truncate">{props.label}</span>
      <input
        type="number"
        step={props.step}
        min={props.min}
        value={props.value}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (!Number.isFinite(v)) return;
          props.onChange(props.integer ? Math.round(v) : v);
        }}
        className="w-28 shrink-0 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1 text-right tabular-nums outline-none focus:border-(--ink-accent)"
      />
    </label>
  );
}

function CheckRow(props: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 py-0.5 text-(--ink-text-dim)">
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(e) => props.onChange(e.target.checked)}
      />
      <span className="truncate">{props.label}</span>
    </label>
  );
}

function SelectRow(props: {
  label: string;
  value: string;
  options: [string, string][];
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 py-0.5 text-(--ink-text-dim)">
      <span className="truncate">{props.label}</span>
      <select
        className="rounded bg-(--ink-bg-2) px-1 py-0.5 text-(--ink-text)"
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
      >
        {props.options.map(([v, label]) => (
          <option key={v} value={v}>
            {label}
          </option>
        ))}
      </select>
    </label>
  );
}
