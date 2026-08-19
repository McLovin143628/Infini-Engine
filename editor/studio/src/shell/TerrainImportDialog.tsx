/**
 * Terrain Import wizard (P16.4a) — "drop a heightmap, get developing".
 *
 * Four steps, one dialog:
 *   1. PICK       a PNG16 / EXR heightmap (native file picker).
 *   2. CONFIGURE  metres-per-sample, the height range (or float-metres for a
 *                 float EXR), and where the lattice sits — with a live readback
 *                 of the REAL-WORLD extent in km and the tile count.
 *   3. IMPORTING  a progress bar fed by `assets://import`, cancellable.
 *   4. DONE       the produced `.inf_terrain`'s numbers + "Add to Scene", which
 *                 spawns a streamed Terrain entity the viewport starts paging.
 *
 * UNITS: every number that crosses IPC is SI metres (units doctrine). The km
 * readback is a display division; nothing is ever stored scaled.
 *
 * AIRSPACE: the native viewport draws over the webview, so the dialog holds a
 * `useViewportOverlay` acquisition for its whole lifetime (the same guard every
 * other shell overlay uses).
 */
import { useEffect } from "react";
import { AlertTriangle, Mountain, X } from "lucide-react";

import { useViewportOverlay } from "../lib/viewportOverlay";
import { useShellStore } from "../stores/shellStore";
import {
  formatBytes,
  formatKm,
  progressPercent,
  settingsIssue,
  useTerrainImportStore,
} from "../stores/terrainImportStore";

/** Heightmap containers the backend decoder accepts. */
// Wave G adds TIFF/GeoTIFF — the format real elevation data ships in. The
// picker's filter is what an author actually meets first, so a format the
// importer handles and the dialog hides is a format nobody finds.
const HEIGHTMAP_EXTENSIONS = ["png", "exr", "tif", "tiff"];

export default function TerrainImportDialog() {
  const open = useShellStore((s) => s.terrainImportOpen);
  const setOpen = useShellStore((s) => s.setTerrainImportOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);

  const step = useTerrainImportStore((s) => s.step);
  const probe = useTerrainImportStore((s) => s.probe);
  const settings = useTerrainImportStore((s) => s.settings);
  const plan = useTerrainImportStore((s) => s.plan);
  const done = useTerrainImportStore((s) => s.done);
  const total = useTerrainImportStore((s) => s.total);
  const stage = useTerrainImportStore((s) => s.stage);
  const result = useTerrainImportStore((s) => s.result);
  const advisories = useTerrainImportStore((s) => s.advisories);
  const error = useTerrainImportStore((s) => s.error);
  const busy = useTerrainImportStore((s) => s.busy);

  useViewportOverlay(open);

  const importing = step === "importing";

  const close = () => {
    useTerrainImportStore.getState().reset();
    setOpen(false);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !importing) close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, importing]);

  if (!open) return null;

  const pick = async () => {
    const { open: openDialog } = await import("@tauri-apps/plugin-dialog");
    const picked = await openDialog({
      multiple: false,
      filters: [{ name: "Heightmaps", extensions: HEIGHTMAP_EXTENSIONS }],
    });
    if (!picked || Array.isArray(picked)) return;
    await useTerrainImportStore.getState().pick(picked);
  };

  const start = async () => {
    await useTerrainImportStore.getState().start();
    pushStatus("Importing terrain…", 60000);
  };

  const addToScene = async () => {
    const guid = await useTerrainImportStore.getState().addToScene();
    if (guid) {
      pushStatus(`Added ${result?.name ?? "terrain"} to the scene`);
      close();
    }
  };

  const issue = settings ? settingsIssue(settings, probe) : null;
  const pct = progressPercent(done, total);

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-16"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget && !importing) close();
      }}
    >
      <div
        className="flex max-h-[80vh] w-[560px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center gap-2 border-b border-(--ink-border) px-3 py-2">
          <Mountain size={15} className="text-(--ink-accent)" />
          <span className="flex-1 font-semibold">Import Terrain</span>
          <button
            aria-label="Close dialog"
            disabled={importing}
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
              <Mountain size={40} className="text-(--ink-text-faint)" />
              <div className="text-(--ink-text-dim)">
                Choose a 16-bit grayscale PNG or an EXR heightmap.
                <br />
                It is decoded in row bands, so a 16 384 × 16 384 source never has to fit
                in memory.
              </div>
              <button
                onClick={() => void pick()}
                disabled={busy}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              >
                {busy ? "Reading header…" : "Choose Heightmap…"}
              </button>
            </div>
          )}

          {/* ── 2. configure ────────────────────────────────────────────── */}
          {step === "configure" && probe && settings && (
            <>
              <Section title="Source">
                <Row label="File" value={fileNameOf(probe.path)} title={probe.path} />
                <Row
                  label="Size"
                  value={`${probe.width} × ${probe.height} samples`}
                />
                <Row
                  label="Format"
                  value={`${probe.format} · ${probe.bit_depth}-bit ${probe.float_samples ? "float" : "integer"} · channel ${probe.channel}`}
                />
                {probe.geo && (
                  <>
                    <Row
                      label="Coordinate system"
                      value={
                        probe.geo.crs_name ??
                        (probe.geo.epsg ? `EPSG:${probe.geo.epsg}` : "not declared")
                      }
                    />
                    {probe.geo.pixel_size != null && (
                      <Row
                        label="Source resolution"
                        // A geographic CRS's pixel size is in DEGREES. Labelling
                        // 0.000278 as metres would read as a quarter of a
                        // millimetre and send the author looking for a bug.
                        value={
                          probe.geo.crs_is_geographic
                            ? `${probe.geo.pixel_size}° per pixel (geographic — not metres)`
                            : `${probe.geo.pixel_size} m per pixel`
                        }
                      />
                    )}
                    {probe.geo.vertical_units !== "metres" && (
                      <Row
                        label="Elevation units"
                        value={`${probe.geo.vertical_units} (converted to metres on import)`}
                      />
                    )}
                    {probe.geo.nodata != null && (
                      <Row label="Declared no-data" value={`${probe.geo.nodata}`} />
                    )}
                  </>
                )}
              </Section>

              {probe.geo?.is_georeferenced && (
                <Section title="Placement">
                  <CheckRow
                    label="Place using this level's geo-anchor"
                    checked={settings.use_georeference}
                    onChange={(use_georeference) =>
                      useTerrainImportStore.getState().patchSettings({ use_georeference })
                    }
                  />
                  <Row
                    label=""
                    value="Off places the terrain at the world origin. On places it where the survey says it is, so roads and rivers from the same source line up with it."
                  />
                </Section>
              )}

              <Section title="No data">
                <SelectRow
                  label="Policy"
                  value={nodataKind(settings.nodata_policy)}
                  options={[
                    ["refuse", "Refuse the import"],
                    ["clamp", "Clamp to an elevation"],
                    ["fill-row", "Fill small gaps"],
                  ]}
                  onChange={(kind) =>
                    useTerrainImportStore
                      .getState()
                      .patchSettings({ nodata_policy: defaultPolicyFor(kind) })
                  }
                />
                {nodataKind(settings.nodata_policy) === "clamp" && (
                  <NumberRow
                    label="Clamp elevation (m)"
                    value={nodataArg(settings.nodata_policy)}
                    step={1}
                    onChange={(v) =>
                      useTerrainImportStore
                        .getState()
                        .patchSettings({ nodata_policy: `clamp:${v}` })
                    }
                  />
                )}
                {nodataKind(settings.nodata_policy) === "fill-row" && (
                  <NumberRow
                    label="Widest gap to fill (samples)"
                    value={nodataArg(settings.nodata_policy)}
                    step={1}
                    min={1}
                    integer
                    onChange={(v) =>
                      useTerrainImportStore
                        .getState()
                        .patchSettings({ nodata_policy: `fill-row:${Math.max(1, Math.round(v))}` })
                    }
                  />
                )}
                {probe.geo?.nodata == null && (
                  <NumberRow
                    label="No-data value (this source declares none)"
                    value={settings.nodata_sentinel ?? 0}
                    step={1}
                    onChange={(v) =>
                      useTerrainImportStore
                        .getState()
                        .patchSettings({ nodata_sentinel: v })
                    }
                  />
                )}
              </Section>

              <Section title="Sampling">
                <NumberRow
                  label="Metres per sample"
                  value={settings.meters_per_sample}
                  step={1}
                  min={0.001}
                  onChange={(v) =>
                    useTerrainImportStore.getState().patchSettings({ meters_per_sample: v })
                  }
                />
                <NumberRow
                  label="Tile resolution"
                  value={settings.tile_resolution}
                  step={1}
                  min={2}
                  integer
                  onChange={(v) =>
                    useTerrainImportStore.getState().patchSettings({ tile_resolution: v })
                  }
                />
                <CheckRow
                  label="Centre on world origin"
                  checked={settings.center}
                  onChange={(center) =>
                    useTerrainImportStore.getState().patchSettings({ center })
                  }
                />
              </Section>

              <Section title="Elevation">
                {probe.absolute_samples && (
                  <CheckRow
                    label="Float metres (use the source's values as absolute heights)"
                    checked={settings.float_meters}
                    onChange={(float_meters) =>
                      useTerrainImportStore.getState().patchSettings({ float_meters })
                    }
                  />
                )}
                {!settings.float_meters && (
                  <>
                    <NumberRow
                      label="Min height (m)"
                      value={settings.min_height}
                      step={10}
                      onChange={(v) =>
                        useTerrainImportStore.getState().patchSettings({ min_height: v })
                      }
                    />
                    <NumberRow
                      label="Max height (m)"
                      value={settings.max_height}
                      step={10}
                      onChange={(v) =>
                        useTerrainImportStore.getState().patchSettings({ max_height: v })
                      }
                    />
                  </>
                )}
              </Section>

              <Section title="World">
                <Row
                  label="Extent"
                  value={
                    plan
                      ? `${formatKm(plan.extent_x_m)} × ${formatKm(plan.extent_z_m)}`
                      : "—"
                  }
                />
                <Row
                  label="Tiles"
                  value={
                    plan ? `${plan.tiles_x} × ${plan.tiles_z} (${plan.tiles})` : "—"
                  }
                />
              </Section>

              {issue && <div className="mt-2 text-(--ink-error)">{issue}</div>}
            </>
          )}

          {/* ── 3. importing ────────────────────────────────────────────── */}
          {step === "importing" && (
            <div className="py-6">
              <div className="mb-2 text-(--ink-text-dim)">
                Importing {probe ? fileNameOf(probe.path) : "heightmap"}…
              </div>
              <div
                className="h-2 w-full overflow-hidden rounded bg-(--ink-bg-3)"
                role="progressbar"
                aria-valuenow={pct}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="h-full bg-(--ink-accent) transition-[width] duration-150"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="mt-1 tabular-nums text-(--ink-text-faint)">
                {total > 0
                  ? `${done.toLocaleString()} / ${total.toLocaleString()} tiles (${pct}%)${stage ? ` · ${stage}` : ""}`
                  : "Preparing…"}
              </div>
            </div>
          )}

          {/* ── 4. done ─────────────────────────────────────────────────── */}
          {step === "done" && (
            <div className="py-2">
              {result ? (
                <Section title={result.name}>
                  <Row
                    label="Extent"
                    value={`${formatKm(result.extent_x_m)} × ${formatKm(result.extent_z_m)}`}
                  />
                  <Row label="Tiles" value={`${result.tiles_x} × ${result.tiles_z}`} />
                  <Row
                    label="Pages"
                    value={`${result.tiles} across ${result.lod_levels} LOD level${result.lod_levels === 1 ? "" : "s"}`}
                  />
                  <Row label="Payload" value={formatBytes(Number(result.bytes))} />
                </Section>
              ) : (
                <div className="text-(--ink-text-dim)">Reading the imported asset…</div>
              )}
              {/* **R2.F7.** The one importer whose source normally lives
                  outside the project, so L7.H7's "no source path is recorded"
                  note is the COMMON case here — and it had no surface at all:
                  the author found out later, from `reimport` refusing with "no
                  import source". The Content Drawer's badge carries it too;
                  this says it while the author is still looking at the import. */}
              {advisories.length > 0 && (
                <div className="mt-2 rounded border border-(--ink-warning)/50 bg-(--ink-warning)/10 p-2">
                  <div className="mb-1 flex items-center gap-1 font-semibold text-(--ink-warning)">
                    <AlertTriangle size={12} /> {advisories.length} advisory
                    {advisories.length === 1 ? "" : "s"}
                  </div>
                  <ul className="list-disc pl-4 text-(--ink-text-dim)">
                    {advisories.map((a, i) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                </div>
              )}
              <div className="mt-2 text-(--ink-text-faint)">
                The terrain streams from its `.inf_terrain` — the level stays small and the
                viewport pages tiles as the camera moves.
              </div>
            </div>
          )}

          {error && <div className="mt-2 text-(--ink-error)">{error}</div>}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-(--ink-border) px-3 py-2">
          <button
            onClick={() => useTerrainImportStore.getState().reset()}
            disabled={importing || step === "pick"}
            className="rounded px-2 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) disabled:opacity-40"
          >
            Choose another file
          </button>
          <div className="flex gap-2">
            {importing ? (
              <button
                onClick={() => void useTerrainImportStore.getState().cancel()}
                className="rounded px-3 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
              >
                Cancel
              </button>
            ) : (
              <button
                onClick={close}
                className="rounded px-3 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
              >
                Close
              </button>
            )}
            {step === "configure" && (
              <button
                onClick={() => void start()}
                disabled={busy || issue !== null}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              >
                Import
              </button>
            )}
            {step === "done" && (
              <button
                onClick={() => void addToScene()}
                disabled={busy || !result}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              >
                Add to Scene
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function fileNameOf(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

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

function CheckRow(props: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
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
        className="bg-(--ink-bg-2) text-(--ink-text) rounded px-1 py-0.5"
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

// ── the no-data policy label (Wave G) ──────────────────────────────────────
//
// The policy crosses the IPC as its stable label ("refuse", "clamp:0",
// "fill-row:32") rather than as a tagged union, because that same string is
// what lands in the asset's sidecar — where it has to stay readable and
// diffable. These three helpers are the only place the frontend takes it apart.

/** The policy kind, ignoring its argument. */
export function nodataKind(policy: string): string {
  const i = policy.indexOf(":");
  return i < 0 ? policy : policy.slice(0, i);
}

/** The policy's numeric argument, or 0 when it has none. */
export function nodataArg(policy: string): number {
  const i = policy.indexOf(":");
  if (i < 0) return 0;
  const n = Number(policy.slice(i + 1));
  return Number.isFinite(n) ? n : 0;
}

/** A fresh policy label of the given kind, with a sensible argument. */
export function defaultPolicyFor(kind: string): string {
  switch (kind) {
    // Sea level: the usual answer for a coastal DEM whose no-data is ocean.
    case "clamp":
      return "clamp:0";
    case "fill-row":
      return "fill-row:32";
    default:
      return "refuse";
  }
}
