/**
 * Capture wizard (P25.4) — "drop photos, get an asset".
 *
 * Four steps, one dialog:
 *   1. PICK       choose photographs (native file picker). The pre-flight runs
 *                 immediately: unreadable files by NAME, too few, mixed sizes.
 *   2. CONFIGURE  the assumed lens, the SCALE STEP, the triangle budget and the
 *                 atlas — with the blocking findings shown before Reconstruct
 *                 is enabled at all.
 *   3. RUNNING    a stage bar fed by `photogrammetry://progress`, cancellable
 *                 BETWEEN stages (the dialog says so, because a Cancel that
 *                 looks instant and is not is worse than one that admits it).
 *   4. REVIEW     the offscreen preview + the baked atlas, the numbers, the
 *                 COVERAGE OVERLAY (what each camera saw), every finding with
 *                 its remedy, the known-size scale helper, and Import.
 *
 * UNITS: `metresPerUnit` is metres per reconstruction unit. Structure from
 * motion is scale-ambiguous, so `1.0` is the reconstruction's own baseline
 * units and is honest rather than metric.
 *
 * AIRSPACE: the native viewport draws over the webview, so the dialog holds a
 * `useViewportOverlay` acquisition for its whole lifetime (the same guard every
 * other shell overlay uses).
 */
import { useEffect, useState } from "react";
import { Camera, X } from "lucide-react";

import type { CaptureIssueDto } from "../bindings/CaptureIssueDto";
import type { CaptureResultDto } from "../bindings/CaptureResultDto";
import { useViewportOverlay } from "../lib/viewportOverlay";
import {
  blockingIssues,
  fileNameOf,
  formatDuration,
  overallPercent,
  scaleForLongestSide,
  useCaptureStore,
} from "../stores/captureWizardStore";
import { useShellStore } from "../stores/shellStore";
import { useAssetStore } from "../stores/assetStore";

/** Image containers the photograph decoder accepts. */
const PHOTO_EXTENSIONS = ["png", "jpg", "jpeg", "tga", "bmp", "hdr", "exr"];

/** The stage names, in the order the backend reports them. */
const STAGES = ["load", "sfm", "dense", "finish", "write"] as const;
const STAGE_LABEL: Record<string, string> = {
  load: "Reading photographs",
  sfm: "Solving camera poses",
  dense: "Building depth and fusing",
  finish: "Retopology, unwrap and bakes",
  write: "Writing assets",
};

export default function CaptureWizardDialog() {
  const open = useShellStore((s) => s.captureWizardOpen);
  const setOpen = useShellStore((s) => s.setCaptureWizardOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);

  const step = useCaptureStore((s) => s.step);
  const status = useCaptureStore((s) => s.status);
  const progress = useCaptureStore((s) => s.progress);
  const preview = useCaptureStore((s) => s.preview);
  const result = useCaptureStore((s) => s.result);
  const yaw = useCaptureStore((s) => s.yaw);
  const pitch = useCaptureStore((s) => s.pitch);
  const error = useCaptureStore((s) => s.error);
  const busy = useCaptureStore((s) => s.busy);

  useViewportOverlay(open);

  const running = step === "running";

  const close = () => {
    void useCaptureStore.getState().reset();
    setOpen(false);
  };

  useEffect(() => {
    if (!open) return;
    void useCaptureStore.getState().refresh();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !running) close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, running]);

  if (!open) return null;

  const settings = status?.settings ?? null;
  const issues = status?.issues ?? [];
  const blocking = blockingIssues(issues);
  const numbers = status?.result ?? null;
  const pct = overallPercent(progress);

  const pick = async () => {
    const { open: openDialog } = await import("@tauri-apps/plugin-dialog");
    const picked = await openDialog({
      multiple: true,
      filters: [{ name: "Photographs", extensions: PHOTO_EXTENSIONS }],
    });
    if (!picked) return;
    const paths = Array.isArray(picked) ? picked : [picked];
    await useCaptureStore.getState().loadPhotos(paths);
  };

  const start = async () => {
    await useCaptureStore.getState().start();
    pushStatus("Reconstructing…", 120000);
  };

  const orbit = async (dy: number, dp: number) => {
    useCaptureStore.getState().orbit(yaw + dy, pitch + dp);
    await useCaptureStore.getState().refreshPreview();
  };

  const doImport = async () => {
    const name = defaultName(status?.photos?.[0]?.name);
    const written = await useCaptureStore.getState().importScan(name);
    if (!written) return;
    pushStatus(`Imported ${written.name} into Content/${written.folder}`);
    // REVEAL it, unlike the P24.5 wizard: the drawer has the machinery and a
    // scan the user cannot find is a scan they will make again.
    const assets = useAssetStore.getState();
    await assets.refresh();
    assets.setFolder(written.folder);
    assets.setSelected(written.mesh);
    useShellStore.getState().setDrawerOpen(true);
  };

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-12"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget && !running) close();
      }}
    >
      <div
        className="flex max-h-[86vh] w-[640px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center gap-2 border-b border-(--ink-border) px-3 py-2">
          <Camera size={15} className="text-(--ink-accent)" />
          <span className="flex-1 font-semibold">Capture from Photographs</span>
          <button
            aria-label="Close dialog"
            disabled={running}
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
              <Camera size={40} className="text-(--ink-text-faint)" />
              <div className="text-(--ink-text-dim)">
                Choose at least three photographs of one object, taken from positions
                that see the same surfaces from different angles.
                <br />
                They are reconstructed in-engine: camera poses, depth, a fused surface,
                then a retopologized mesh with baked colour, normals and occlusion.
              </div>
              <button
                onClick={() => void pick()}
                disabled={busy}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              >
                {busy ? "Reading…" : "Choose Photographs…"}
              </button>
            </div>
          )}

          {/* ── 2. configure ────────────────────────────────────────────── */}
          {step === "configure" && settings && (
            <>
              <Section title={`Photographs (${status?.photos.length ?? 0})`}>
                <div className="max-h-40 overflow-auto">
                  {status?.photos.map((p) => (
                    <div
                      key={p.path}
                      className="flex items-center justify-between gap-2 py-0.5"
                    >
                      <span className="truncate text-(--ink-text-dim)" title={p.path}>
                        {p.name}
                      </span>
                      <span
                        className={`shrink-0 tabular-nums ${p.error ? "text-(--ink-error)" : "text-(--ink-text-faint)"}`}
                      >
                        {p.error ? "unreadable" : `${p.width} × ${p.height}`}
                      </span>
                    </div>
                  ))}
                </div>
              </Section>

              <Section title="Assumed lens">
                <div className="mb-1 text-(--ink-text-faint)">
                  Camera intrinsics are an INPUT — the solve refines poses and never
                  the lens — so these are a guess unless you know the camera.
                </div>
                <NumberRow
                  label="Focal length ÷ longest side"
                  value={settings.camera.focalRatio}
                  step={0.05}
                  min={0.1}
                  onChange={(v) =>
                    void useCaptureStore.getState().patchCamera({ focalRatio: v })
                  }
                />
                <NumberRow
                  label="Radial k1"
                  value={settings.camera.k1}
                  step={0.01}
                  onChange={(v) => void useCaptureStore.getState().patchCamera({ k1: v })}
                />
                <NumberRow
                  label="Radial k2"
                  value={settings.camera.k2}
                  step={0.01}
                  onChange={(v) => void useCaptureStore.getState().patchCamera({ k2: v })}
                />
              </Section>

              <Section title="Scale">
                <div className="mb-1 text-(--ink-text-faint)">
                  A reconstruction has no size of its own. 1.0 leaves it in its own
                  baseline units; after the first run you can type the real length of
                  its longest side and re-finish.
                </div>
                <NumberRow
                  label="Metres per unit"
                  value={settings.metresPerUnit}
                  step={0.05}
                  onChange={(v) =>
                    void useCaptureStore.getState().patchSettings({ metresPerUnit: v })
                  }
                />
              </Section>

              <Section title="Output">
                <NumberRow
                  label="Triangle budget"
                  value={settings.targetTriangles}
                  step={1000}
                  min={100}
                  integer
                  onChange={(v) =>
                    void useCaptureStore.getState().patchSettings({ targetTriangles: v })
                  }
                />
                <NumberRow
                  label="Atlas size (texels)"
                  value={settings.atlasSize}
                  step={256}
                  min={64}
                  integer
                  onChange={(v) =>
                    void useCaptureStore.getState().patchSettings({ atlasSize: v })
                  }
                />
                <NumberRow
                  label="Occlusion rays per texel"
                  value={settings.aoRays}
                  step={8}
                  min={1}
                  integer
                  onChange={(v) =>
                    void useCaptureStore.getState().patchSettings({ aoRays: v })
                  }
                />
                <CheckRow
                  label="Remove geometry no camera photographed"
                  checked={settings.trimUnseen}
                  onChange={(trimUnseen) =>
                    void useCaptureStore.getState().patchSettings({ trimUnseen })
                  }
                />
                <CheckRow
                  label="Attempt de-lighting (declines when the fit is not believable)"
                  checked={settings.delight}
                  onChange={(delight) =>
                    void useCaptureStore.getState().patchSettings({ delight })
                  }
                />
              </Section>

              <Diagnostics issues={issues} />
            </>
          )}

          {/* ── 3. running ──────────────────────────────────────────────── */}
          {step === "running" && (
            <div className="py-4">
              <div className="mb-2 text-(--ink-text-dim)">
                {progress
                  ? (STAGE_LABEL[progress.stage] ?? progress.stage)
                  : "Starting…"}
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
                {progress?.detail || "Preparing…"} · {pct}%
              </div>
              <ol className="mt-3">
                {STAGES.map((stage, i) => {
                  const at = progress ? progress.stageIndex : -1;
                  const state = i < at ? "done" : i === at ? "now" : "todo";
                  return (
                    <li
                      key={stage}
                      className={`py-0.5 ${
                        state === "now"
                          ? "text-(--ink-text)"
                          : state === "done"
                            ? "text-(--ink-text-dim)"
                            : "text-(--ink-text-faint)"
                      }`}
                    >
                      {state === "done" ? "✓" : state === "now" ? "▸" : "·"}{" "}
                      {STAGE_LABEL[stage]}
                    </li>
                  );
                })}
              </ol>
              <div className="mt-3 text-(--ink-text-faint)">
                Cancel stops the run at the end of the current stage — a stage is one
                solve and cannot be interrupted part-way. If that stage is the last one
                (the bakes) there is nothing left to skip and the run finishes. Nothing
                is written until you press Import either way, so a cancelled run leaves
                no assets behind.
              </div>
            </div>
          )}

          {/* ── 4. review ───────────────────────────────────────────────── */}
          {step === "review" && numbers && settings && (
            <>
              <div className="mb-3 flex gap-2">
                <PreviewPane
                  title="Geometry"
                  image={preview?.geometry ?? null}
                  fallback={preview?.error ?? "Rendering…"}
                />
                <PreviewPane
                  title="Base colour"
                  image={preview?.albedo ?? null}
                  fallback="Baking…"
                />
              </div>
              <div className="mb-3 flex gap-1">
                <button
                  className="rounded border border-(--ink-border) px-2 py-0.5 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                  onClick={() => void orbit(-30, 0)}
                >
                  ⟲
                </button>
                <button
                  className="rounded border border-(--ink-border) px-2 py-0.5 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                  onClick={() => void orbit(30, 0)}
                >
                  ⟳
                </button>
                <button
                  className="rounded border border-(--ink-border) px-2 py-0.5 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                  onClick={() => void orbit(0, 15)}
                >
                  ▲
                </button>
                <button
                  className="rounded border border-(--ink-border) px-2 py-0.5 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                  onClick={() => void orbit(0, -15)}
                >
                  ▼
                </button>
                <button
                  className="ml-auto rounded border border-(--ink-border) px-2 py-0.5 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                  onClick={() => void useCaptureStore.getState().refreshPreview()}
                >
                  Refresh preview
                </button>
              </div>

              <Section title="Result">
                <Row
                  label="Cameras"
                  value={`${numbers.registered} of ${numbers.views} registered · RMS ${numbers.reprojectionRmsPx.toFixed(3)} px`}
                />
                <Row
                  label="Mesh"
                  value={`${numbers.triangles.toLocaleString()} triangles, ${numbers.vertices.toLocaleString()} vertices`}
                />
                <Row
                  label="Atlas"
                  value={`${numbers.charts} charts covering ${(numbers.atlasCoverage * 100).toFixed(1)}% of ${settings.atlasSize}²`}
                />
                <Row
                  label="Size"
                  value={`${numbers.extentUnits.toFixed(3)} units across · ${numbers.extentMetres.toFixed(3)} m at ${settings.metresPerUnit} m/unit`}
                />
                <Row
                  label="Time"
                  value={numbers.elapsedMs
                    .slice(0, 4)
                    .map((ms, i) => `${STAGES[i]} ${formatDuration(Number(ms))}`)
                    .join(" · ")}
                />
              </Section>

              <Section title="Scale">
                <div className="mb-1 text-(--ink-text-faint)">
                  Type the real length of the longest side and re-finish. Only the
                  finish stage runs again — the poses and the fused surface are
                  already solved.
                </div>
                <KnownSizeRow extentUnits={numbers.extentUnits} />
              </Section>

              <Coverage numbers={numbers} />
              <Diagnostics issues={issues} />

              {result && (
                <Section title={`Imported as ${result.name}`}>
                  <Row label="Folder" value={`Content/${result.folder}`} />
                  <Row label="Mesh" value={result.mesh} />
                  <Row label="Material" value={result.material} />
                  {result.notes.map((n) => (
                    <div key={n} className="mt-1 text-(--ink-text-faint)">
                      {n}
                    </div>
                  ))}
                </Section>
              )}
            </>
          )}

          {error && <div className="mt-2 text-(--ink-error)">{error}</div>}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-(--ink-border) px-3 py-2">
          <button
            onClick={() => void useCaptureStore.getState().reset()}
            disabled={running || step === "pick"}
            className="rounded px-2 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) disabled:opacity-40"
          >
            Choose other photographs
          </button>
          <div className="flex gap-2">
            {running ? (
              <button
                onClick={() => void useCaptureStore.getState().cancel()}
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
                disabled={busy || blocking.length > 0}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
                title={blocking[0]?.message}
              >
                Reconstruct
              </button>
            )}
            {step === "review" && (
              <button
                onClick={() => void doImport()}
                disabled={busy}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              >
                {result ? "Import again" : "Import"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** A scan's default name, from its first photograph. */
function defaultName(first: string | undefined): string {
  if (!first) return "Scan";
  const base = fileNameOf(first).replace(/\.[^.]+$/, "");
  const cleaned = base.replace(/[^A-Za-z0-9_-]/g, "");
  return cleaned.length > 0 ? cleaned : "Scan";
}

/** The coverage overlay: what each camera saw, and how much rests on one view. */
function Coverage(props: { numbers: CaptureResultDto }) {
  const c = props.numbers.coverage;
  return (
    <div className="mb-3">
      <div className="mb-1 font-semibold text-(--ink-text-faint)">Coverage</div>
      <div className="rounded border border-(--ink-border) bg-(--ink-bg-0) p-2">
        <Row
          label="Photographed"
          value={`${(c.coveredFraction * 100).toFixed(1)}% of the surface · ${(c.overlapFraction * 100).toFixed(1)}% by two cameras or more`}
        />
        <Row
          label="Invented texels"
          value={`${c.unseenTexels.toLocaleString()} of ${c.coveredTexels.toLocaleString()} filled from neighbours`}
        />
        <div className="mt-1">
          {c.views.map((v) => (
            <div key={v.view} className="flex items-center gap-2 py-0.5">
              <span
                className="w-32 shrink-0 truncate text-(--ink-text-dim)"
                title={v.photo}
              >
                {v.photo}
              </span>
              <div className="h-1.5 flex-1 overflow-hidden rounded bg-(--ink-bg-3)">
                <div
                  className={`h-full ${v.registered ? "bg-(--ink-accent)" : "bg-(--ink-error)"}`}
                  style={{ width: `${Math.round(v.fraction * 100)}%` }}
                />
              </div>
              <span className="w-24 shrink-0 text-right tabular-nums text-(--ink-text-faint)">
                {v.registered ? `${(v.fraction * 100).toFixed(0)}% seen` : "no pose"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Every finding, grouped by severity, with its own remedy in the sentence. */
function Diagnostics(props: { issues: CaptureIssueDto[] }) {
  if (props.issues.length === 0) return null;
  const colour: Record<string, string> = {
    blocking: "text-(--ink-error)",
    warning: "text-(--ink-warning)",
    note: "text-(--ink-text-faint)",
  };
  return (
    <div className="mb-3">
      <div className="mb-1 font-semibold text-(--ink-text-faint)">
        Diagnostics ({props.issues.length})
      </div>
      <div className="max-h-48 overflow-auto rounded border border-(--ink-border) bg-(--ink-bg-0) p-2">
        {props.issues.map((issue, i) => (
          <div
            key={`${issue.stage}-${i}`}
            className={`py-0.5 ${colour[issue.severity] ?? "text-(--ink-text-dim)"}`}
          >
            <span className="mr-1 uppercase opacity-60">{issue.stage}</span>
            {issue.message}
          </div>
        ))}
      </div>
    </div>
  );
}

/** The known-size scale helper: a length in, a multiplier and a re-finish out. */
function KnownSizeRow(props: { extentUnits: number }) {
  const [metres, setMetres] = useState(1);
  const scale = scaleForLongestSide(props.extentUnits, metres);
  return (
    <div className="flex items-center gap-2 py-0.5">
      <span className="text-(--ink-text-dim)">Longest side is</span>
      <input
        type="number"
        step={0.1}
        min={0.001}
        value={metres}
        onChange={(e) => setMetres(Number(e.target.value))}
        className="w-24 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1 text-right tabular-nums outline-none focus:border-(--ink-accent)"
      />
      <span className="text-(--ink-text-dim)">m</span>
      <span className="flex-1 truncate text-(--ink-text-faint)">
        {scale === null ? "—" : `= ${scale.toPrecision(5)} m/unit`}
      </span>
      <button
        disabled={scale === null}
        onClick={() => {
          if (scale === null) return;
          void (async () => {
            await useCaptureStore.getState().patchSettings({ metresPerUnit: scale });
            await useCaptureStore.getState().refinish();
          })();
        }}
        className="rounded border border-(--ink-border) px-2 py-0.5 text-(--ink-text-dim) hover:bg-(--ink-bg-3) disabled:opacity-40"
      >
        Apply &amp; re-finish
      </button>
    </div>
  );
}

function PreviewPane(props: { title: string; image: string | null; fallback: string }) {
  return (
    <div className="flex-1">
      <div className="mb-1 font-semibold text-(--ink-text-faint)">{props.title}</div>
      <div className="flex aspect-square items-center justify-center overflow-hidden rounded border border-(--ink-border) bg-(--ink-bg-0)">
        {props.image ? (
          <img src={props.image} alt={props.title} className="h-full w-full object-contain" />
        ) : (
          <span className="px-2 text-center text-(--ink-text-faint)">{props.fallback}</span>
        )}
      </div>
    </div>
  );
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
