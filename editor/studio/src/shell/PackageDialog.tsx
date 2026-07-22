/**
 * Package / Cook dialog (Build ▸ Package Project… / Cook Content, P9.2 item 3).
 *
 * Runs `inf_packager::cook` against the open project via the `project_package`
 * command (a blocking backend task) and renders the report: per-kind asset
 * counts, total pack size, output paths, and warnings. A cook failure shows a
 * structured error panel — blueprint failures are anchored to their class +
 * handler. Mirrors SortingLayersDialog: driven by `shellStore.packageDialogOpen`
 * and hides the native viewport while open (airspace rule). Output paths carry
 * Reveal (open in the OS file browser) + Copy affordances.
 *
 * Honest scope: per-stage progress is not surfaced — the cook API has no progress
 * callback, so the state area shows a start→finish spinner only.
 */
import { useEffect, useState } from "react";
import { AlertTriangle, Copy, FolderOpen, FolderSearch, Loader2, Package, X } from "lucide-react";

import type { PackageErrorDto } from "../bindings/PackageErrorDto";
import type { PackageResultDto } from "../bindings/PackageResultDto";
import { packaging, shell } from "../lib/ipc";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useProjectStore } from "../stores/projectStore";
import { useShellStore } from "../stores/shellStore";

/** Format a byte count as a compact human-readable string. */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}

/** Title-case an asset-kind slug ("point_light" → "Point Light"). */
function prettyKind(slug: string): string {
  return slug
    .split(/[_\s]+/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

/** Coerce a rejected `project_package` value into a structured error. */
function toPackageError(e: unknown): PackageErrorDto {
  if (e && typeof e === "object" && "message" in e && "class" in e) {
    return e as PackageErrorDto;
  }
  return { class: "internal", message: String(e), blueprint_class: null, handler: null, guid: null };
}

async function pickDirectory(title: string): Promise<string | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const picked = await open({ directory: true, multiple: false, title });
  return typeof picked === "string" ? picked : null;
}

export default function PackageDialog() {
  const open = useShellStore((s) => s.packageDialogOpen);
  const setOpen = useShellStore((s) => s.setPackageDialogOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);
  const project = useProjectStore((s) => s.current);

  // `outDir` is an override; null means "use the derived default" so a project
  // change (or first open) recomputes it without a setState-in-effect.
  const [outDirOverride, setOutDirOverride] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PackageResultDto | null>(null);
  const [error, setError] = useState<PackageErrorDto | null>(null);

  const defaultOutDir = project ? `${project.root}/Build` : "Build";
  const outDir = outDirOverride ?? defaultOutDir;
  const noProject = !project;

  // Modal dialogs overlay the native viewport hole — hide it while open.
  useViewportOverlay(open);

  // Reset transient state and close (also clears the override so the next open
  // recomputes the default output dir).
  const close = () => {
    setResult(null);
    setError(null);
    setOutDirOverride(null);
    setOpen(false);
  };

  // Escape closes when idle (setState lives in the callback, not the effect body).
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

  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await packaging.cook(outDir.trim() || null, null);
      setResult(r);
      pushStatus(`Packaged “${r.project_name}” — ${r.asset_count} assets, ${formatBytes(r.pack_bytes)}.`);
    } catch (e) {
      setError(toPackageError(e));
    } finally {
      setBusy(false);
    }
  };

  const browse = async () => {
    const dir = await pickDirectory("Choose the package output folder");
    if (dir) setOutDirOverride(dir);
  };

  const copy = (text: string) => {
    void navigator.clipboard?.writeText(text);
    pushStatus("Path copied to clipboard.");
  };

  const reveal = (text: string) => {
    void shell.reveal(text).catch((e) => pushStatus(`Reveal failed: ${String(e)}`));
  };

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-20"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget && !busy) close();
      }}
    >
      <div
        className="flex max-h-[80vh] w-[560px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center gap-2 border-b border-(--ink-border) px-3 py-2">
          <Package size={15} className="text-(--ink-accent)" />
          <span className="flex-1 font-semibold">Package Project</span>
          <button
            aria-label="Close dialog"
            disabled={busy}
            className="rounded p-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text) disabled:opacity-40"
            onClick={close}
          >
            <X size={14} />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-3">
          {noProject ? (
            <div className="text-xs text-(--ink-text-dim)">
              No project is open. Open a project before packaging.
            </div>
          ) : (
            <>
              {/* Output directory */}
              <label className="mb-1 block text-xs text-(--ink-text-faint)">Output Directory</label>
              <div className="mb-3 flex gap-2">
                <input
                  value={outDir}
                  onChange={(e) => setOutDirOverride(e.target.value)}
                  disabled={busy}
                  spellCheck={false}
                  className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 font-mono text-xs outline-none focus:border-(--ink-accent) disabled:opacity-50"
                />
                <button
                  onClick={() => void browse()}
                  disabled={busy}
                  className="flex items-center gap-1 rounded border border-(--ink-border) px-2 text-xs hover:border-(--ink-accent) disabled:opacity-40"
                >
                  <FolderOpen size={13} /> Browse…
                </button>
              </div>
              <p className="mb-3 text-[11px] text-(--ink-text-faint)">
                Cooks the project into a content-addressed <code>content.inf_pack</code> plus a
                deterministic <code>manifest.toml</code>. All levels and gameplay scripts are packed
                by default, pulling in their referenced assets.
              </p>

              {/* Live state area */}
              {busy && (
                <div className="mb-3 flex items-center gap-2 rounded border border-(--ink-border) bg-(--ink-bg-2) px-3 py-2 text-xs text-(--ink-text-dim)">
                  <Loader2 size={14} className="animate-spin text-(--ink-accent)" />
                  Cooking… (scanning, validating blueprints, writing pack)
                </div>
              )}

              {/* Error panel */}
              {error && !busy && (
                <div className="mb-3 rounded border border-(--ink-error)/50 bg-(--ink-error)/10 p-3 text-xs">
                  <div className="mb-1 flex items-center gap-1.5 font-semibold text-(--ink-error)">
                    <AlertTriangle size={13} />
                    Cook failed
                    <span className="ml-1 rounded bg-(--ink-error)/20 px-1.5 py-0.5 text-[10px] font-normal uppercase tracking-wide">
                      {error.class}
                    </span>
                  </div>
                  {(error.blueprint_class || error.handler) && (
                    <div className="mb-1 font-mono text-(--ink-text-dim)">
                      {error.blueprint_class ?? "?"}
                      {error.handler ? ` / ${error.handler}` : ""}
                    </div>
                  )}
                  <div className="whitespace-pre-wrap break-words text-(--ink-text)">
                    {error.message}
                  </div>
                  {error.guid && (
                    <div className="mt-1 font-mono text-[10px] text-(--ink-text-faint)">
                      asset {error.guid}
                    </div>
                  )}
                </div>
              )}

              {/* Report */}
              {result && !busy && (
                <div className="mb-1 rounded border border-(--ink-border) bg-(--ink-bg-2) p-3 text-xs">
                  <div className="mb-2 flex items-baseline justify-between">
                    <span className="font-semibold text-(--ink-text)">
                      {result.project_name}
                    </span>
                    <span className="text-(--ink-text-faint)">engine {result.engine_version}</span>
                  </div>

                  <div className="mb-2 grid grid-cols-3 gap-2">
                    <Stat label="Assets" value={String(result.asset_count)} />
                    <Stat label="Pack Size" value={formatBytes(result.pack_bytes)} />
                    <Stat label="Levels" value={String(result.levels.length)} />
                    <Stat label="Blueprints" value={String(result.blueprints_validated)} />
                    <Stat label="Levels Rewritten" value={String(result.levels_rewritten)} />
                    <Stat
                      label="Boot Level"
                      value={result.root_level ? "set" : "none"}
                    />
                  </div>

                  {result.kinds.length > 0 && (
                    <div className="mb-2">
                      <div className="mb-1 text-(--ink-text-faint)">By kind</div>
                      <div className="rounded border border-(--ink-border)">
                        {result.kinds.map((k, i) => (
                          <div
                            key={k.kind}
                            className={`flex justify-between px-2 py-1 ${
                              i > 0 ? "border-t border-(--ink-border)" : ""
                            }`}
                          >
                            <span className="text-(--ink-text-dim)">{prettyKind(k.kind)}</span>
                            <span className="font-mono">{k.count}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <PathRow label="Pack" path={result.pack_path} onCopy={copy} onReveal={reveal} />
                  <PathRow
                    label="Manifest"
                    path={result.manifest_path}
                    onCopy={copy}
                    onReveal={reveal}
                  />

                  {result.warnings.length > 0 && (
                    <div className="mt-2">
                      <div className="mb-1 flex items-center gap-1 text-(--ink-warning)">
                        <AlertTriangle size={12} /> {result.warnings.length} warning
                        {result.warnings.length > 1 ? "s" : ""}
                      </div>
                      <ul className="list-disc pl-4 text-(--ink-text-dim)">
                        {result.warnings.map((w, i) => (
                          <li key={i}>{w}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-(--ink-border) px-3 py-2">
          <button
            onClick={close}
            disabled={busy}
            className="rounded px-3 py-1 text-xs text-(--ink-text-dim) hover:bg-(--ink-bg-3) disabled:opacity-40"
          >
            Close
          </button>
          <button
            onClick={() => void run()}
            disabled={busy || noProject}
            className="flex items-center gap-1.5 rounded bg-(--ink-accent) px-3 py-1 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
          >
            {busy ? <Loader2 size={13} className="animate-spin" /> : <Package size={13} />}
            {result ? "Cook Again" : "Cook"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-(--ink-border) bg-(--ink-bg-1) px-2 py-1">
      <div className="text-[10px] uppercase tracking-wide text-(--ink-text-faint)">{label}</div>
      <div className="font-mono text-(--ink-text)">{value}</div>
    </div>
  );
}

function PathRow({
  label,
  path,
  onCopy,
  onReveal,
}: {
  label: string;
  path: string;
  onCopy: (text: string) => void;
  onReveal: (text: string) => void;
}) {
  return (
    <div className="mt-1 flex items-center gap-2">
      <span className="w-16 shrink-0 text-(--ink-text-faint)">{label}</span>
      <input
        readOnly
        value={path}
        spellCheck={false}
        onFocus={(e) => e.currentTarget.select()}
        className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1.5 py-0.5 font-mono text-[11px] outline-none"
      />
      <button
        aria-label={`Reveal ${label} in file browser`}
        onClick={() => onReveal(path)}
        className="rounded p-1 text-(--ink-text-faint) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
      >
        <FolderSearch size={12} />
      </button>
      <button
        aria-label={`Copy ${label} path`}
        onClick={() => onCopy(path)}
        className="rounded p-1 text-(--ink-text-faint) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
      >
        <Copy size={12} />
      </button>
    </div>
  );
}
