/**
 * Save/Load layout preset dialogs (ROADMAP P1.2.5), driven by the
 * `layoutDialog` field on the shell store (opened from the Window menu).
 */
import { useEffect, useState } from "react";
import { Trash2, X } from "lucide-react";
import type { LayoutSummary } from "../lib/ipc";
import { layouts } from "../lib/ipc";
import { ACTIVE_LAYOUT_NAME, loadLayoutPreset, saveLayoutPreset } from "../panels/dock/dockLayoutStore";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useShellStore } from "../stores/shellStore";

export default function LayoutDialog() {
  const dialog = useShellStore((s) => s.layoutDialog);
  const close = useShellStore((s) => s.closeLayoutDialog);
  const pushStatus = useShellStore((s) => s.pushStatus);

  // Modal dialogs overlay the native viewport hole — hide it while open.
  useViewportOverlay(dialog !== null);

  useEffect(() => {
    if (!dialog) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dialog, close]);

  if (!dialog) return null;

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-32"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div
        className="w-[420px] rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center border-b border-(--ink-border) px-3 py-2">
          <span className="flex-1 font-semibold">
            {dialog === "save" ? "Save Layout As" : "Load Layout"}
          </span>
          <button
            aria-label="Close dialog"
            className="rounded p-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
            onClick={close}
          >
            <X size={14} />
          </button>
        </div>
        {dialog === "save" ? (
          <SaveBody
            onDone={(name) => {
              close();
              pushStatus(`Layout saved as “${name}”.`);
            }}
          />
        ) : (
          <LoadBody
            onDone={(name) => {
              close();
              pushStatus(`Layout “${name}” loaded.`);
            }}
          />
        )}
      </div>
    </div>
  );
}

function SaveBody({ onDone }: { onDone: (name: string) => void }) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const valid = /^[A-Za-z0-9 _-]{1,64}$/.test(name) && !name.startsWith(" ") && !name.endsWith(" ");

  const save = async () => {
    if (!valid) return;
    try {
      await saveLayoutPreset(name.trim());
      onDone(name.trim());
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div className="p-3">
      <input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") void save();
        }}
        placeholder="Layout name (letters, digits, space, - _)"
        className="w-full rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1.5 outline-none focus:border-(--ink-accent)"
      />
      {error && <div className="mt-2 text-xs text-(--ink-error)">{error}</div>}
      <div className="mt-3 flex justify-end gap-2">
        <button
          disabled={!valid}
          onClick={() => void save()}
          className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
        >
          Save
        </button>
      </div>
    </div>
  );
}

function LoadBody({ onDone }: { onDone: (name: string) => void }) {
  const [presets, setPresets] = useState<LayoutSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    layouts
      .list()
      .then((all) => setPresets(all.filter((p) => p.name !== ACTIVE_LAYOUT_NAME)))
      .catch((e) => setError(String(e)));
  };
  useEffect(refresh, []);

  return (
    <div className="max-h-80 overflow-auto p-2">
      {error && <div className="p-2 text-xs text-(--ink-error)">{error}</div>}
      {presets && presets.length === 0 && (
        <div className="p-3 text-(--ink-text-dim)">
          No saved layouts yet — use Window ▸ Save Layout As…
        </div>
      )}
      {presets?.map((p) => (
        <div key={p.name} className="group flex items-center gap-1 rounded hover:bg-(--ink-bg-3)">
          <button
            className="flex-1 truncate px-2 py-1.5 text-left"
            onClick={() => {
              void loadLayoutPreset(p.name).then((ok) => {
                if (ok) onDone(p.name);
                else setError(`Could not load “${p.name}”.`);
              });
            }}
          >
            {p.name}
          </button>
          <button
            aria-label={`Delete layout ${p.name}`}
            className="mr-1 hidden rounded p-1 text-(--ink-text-faint) group-hover:block hover:bg-(--ink-error)/15 hover:text-(--ink-error)"
            onClick={() => {
              void layouts.delete(p.name).then(refresh);
            }}
          >
            <Trash2 size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}
