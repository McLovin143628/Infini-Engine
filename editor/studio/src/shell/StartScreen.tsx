/**
 * Start screen (Phase 5.5): the New / Open / Recent project surface.
 *
 * Shown full-surface when no project is open (or File → New/Open forces it).
 * Airspace rule: it can cross the native viewport hole, so it holds a
 * viewport-overlay acquisition (hides the native child) for its lifetime, like
 * the menus / palette / dialogs.
 */
import { useMemo, useState } from "react";
import { Box, Clock, FolderOpen, FolderPlus, Rocket, X } from "lucide-react";

import { cn } from "../lib/utils";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useProjectStore } from "../stores/projectStore";

export default function StartScreen() {
  const ready = useProjectStore((s) => s.ready);
  const current = useProjectStore((s) => s.current);
  const forced = useProjectStore((s) => s.showStartScreen);

  const visible = ready && (current === null || forced);
  useViewportOverlay(visible);
  if (!visible) return null;

  return <StartScreenBody canCancel={current !== null} />;
}

function StartScreenBody({ canCancel }: { canCancel: boolean }) {
  const templates = useProjectStore((s) => s.templates);
  const recent = useProjectStore((s) => s.recent);
  const error = useProjectStore((s) => s.error);
  const newViaDialog = useProjectStore((s) => s.newViaDialog);
  const openViaDialog = useProjectStore((s) => s.openViaDialog);
  const openProject = useProjectStore((s) => s.openProject);
  const setShowStartScreen = useProjectStore((s) => s.setShowStartScreen);

  const [name, setName] = useState("MyGame");
  const [template, setTemplate] = useState(() => templates[0]?.slug ?? "blank-3d");

  // Keep the selected template valid once templates load.
  const selected = useMemo(
    () => (templates.some((t) => t.slug === template) ? template : (templates[0]?.slug ?? template)),
    [templates, template],
  );

  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center bg-(--ink-bg-0)/95 p-6">
      <div className="flex h-full max-h-[560px] w-full max-w-4xl overflow-hidden rounded-lg border border-(--ink-border) bg-(--ink-bg-1) shadow-2xl">
        {/* New project */}
        <div className="flex min-w-0 flex-1 flex-col border-r border-(--ink-border) p-6">
          <div className="mb-4 flex items-center gap-2">
            <Rocket size={20} className="text-(--ink-accent)" />
            <h1 className="text-lg font-semibold">Infinity Engine</h1>
            {canCancel && (
              <button
                className="ml-auto flex size-6 items-center justify-center rounded text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
                onClick={() => setShowStartScreen(false)}
                aria-label="Close"
              >
                <X size={15} />
              </button>
            )}
          </div>

          <label className="mb-1 text-[11px] font-semibold text-(--ink-text-dim)">
            Project Name
          </label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mb-4 h-8 rounded border border-(--ink-border) bg-(--ink-bg-0) px-2 text-sm outline-none focus:border-(--ink-accent)"
            placeholder="MyGame"
          />

          <div className="mb-1 text-[11px] font-semibold text-(--ink-text-dim)">Template</div>
          <div className="grid min-h-0 flex-1 grid-cols-1 gap-2 overflow-auto pr-1">
            {templates.map((t) => (
              <button
                key={t.slug}
                className={cn(
                  "flex items-start gap-2 rounded border p-2 text-left",
                  selected === t.slug
                    ? "border-(--ink-accent) bg-(--ink-accent-muted)"
                    : "border-(--ink-border) bg-(--ink-bg-2) hover:border-(--ink-border-strong)",
                )}
                onClick={() => setTemplate(t.slug)}
              >
                <Box size={16} className="mt-0.5 shrink-0 text-(--ink-info)" />
                <span className="min-w-0">
                  <span className="block text-sm">{t.label}</span>
                  <span className="block truncate text-[11px] text-(--ink-text-faint)">
                    {t.description}
                  </span>
                </span>
              </button>
            ))}
          </div>

          {error && <div className="mt-2 text-xs text-(--ink-danger)">{error}</div>}

          <div className="mt-4 flex gap-2">
            <button
              className="flex h-8 flex-1 items-center justify-center gap-1.5 rounded bg-(--ink-accent) text-sm text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)"
              onClick={() => void newViaDialog(name, selected)}
            >
              <FolderPlus size={15} /> Create Project…
            </button>
            <button
              className="flex h-8 items-center justify-center gap-1.5 rounded border border-(--ink-border) px-3 text-sm hover:bg-(--ink-bg-3)"
              onClick={() => void openViaDialog()}
            >
              <FolderOpen size={15} /> Open…
            </button>
          </div>
        </div>

        {/* Recent */}
        <div className="flex w-72 shrink-0 flex-col p-4">
          <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold text-(--ink-text-dim)">
            <Clock size={12} /> Recent Projects
          </div>
          {recent.length === 0 ? (
            <div className="text-[11px] text-(--ink-text-faint)">No recent projects.</div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-auto">
              {recent.map((r) => (
                <button
                  key={r.path}
                  className="flex flex-col rounded px-2 py-1 text-left hover:bg-(--ink-bg-3)"
                  onClick={() => void openProject(r.path)}
                  title={r.path}
                >
                  <span className="truncate text-sm">{r.name}</span>
                  <span className="truncate text-[10px] text-(--ink-text-faint)">{r.path}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
