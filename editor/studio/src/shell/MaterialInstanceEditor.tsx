/**
 * Material-instance override editor (E-P2): edit the sparse PBR overrides a
 * `.inf_mati` layers over its parent material.
 *
 * Rendered INSIDE the Content Drawer (airspace-safe, like DataAssetEditor) when
 * a material_instance asset is opened. Four rows — base color, metallic,
 * roughness, emissive — each an override checkbox + a typed editor. An unchecked
 * row inherits the parent's resolved value (shown grayed + disabled). Save writes
 * the overrides; the content-hash change refreshes the thumbnail automatically
 * via `assets://changed`.
 *
 * Placed actors bake a COPY of the material at apply time, so editing the
 * instance does NOT live-update entities already in the scene — the honest update
 * path is "Apply to Selection" (re-applies the resolved material). The caption
 * states this.
 */
import { useEffect, useState } from "react";
import { ArrowLeft, Paintbrush, Save } from "lucide-react";

import { cn } from "../lib/utils";
import { assets as assetsIpc, scene as sceneIpc } from "../lib/ipc";
import type { MaterialInstanceDto, MatOverridesDto } from "../lib/ipc";
import { useShellStore } from "../stores/shellStore";

/** Float [0,1] RGB → `#rrggbb`. */
function toHex(rgb: readonly number[]): string {
  const c = (v: number) =>
    Math.round(Math.max(0, Math.min(1, v)) * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${c(rgb[0] ?? 0)}${c(rgb[1] ?? 0)}${c(rgb[2] ?? 0)}`;
}
/** `#rrggbb` → float [0,1] RGB triple. */
function fromHex(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16) || 0;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

const rowCls = "flex items-center gap-2 border-b border-(--ink-border) px-3 py-2";
const numCls =
  "h-6 w-16 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1.5 text-xs outline-none focus:border-(--ink-accent)";
const btnCls = "flex h-6 items-center gap-1 rounded px-2 text-xs hover:bg-(--ink-bg-3)";

/** One override row: a checkbox that toggles inherit ↔ override, plus the editor
 *  (disabled + grayed while inheriting). */
function OverrideRow({
  label,
  overridden,
  onToggle,
  children,
}: {
  label: string;
  overridden: boolean;
  onToggle: (on: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <div className={rowCls}>
      <label className="flex w-32 shrink-0 items-center gap-1.5 text-xs" title="Override this parameter">
        <input
          type="checkbox"
          checked={overridden}
          onChange={(e) => onToggle(e.target.checked)}
        />
        <span className={cn(overridden ? "text-(--ink-text)" : "text-(--ink-text-dim)")}>
          {label}
        </span>
      </label>
      <div className={cn("flex items-center gap-2", !overridden && "opacity-40")}>{children}</div>
      {!overridden && <span className="text-[10px] text-(--ink-text-faint)">inherited</span>}
    </div>
  );
}

export default function MaterialInstanceEditor({
  id,
  onClose,
}: {
  id: string;
  onClose: () => void;
}) {
  const pushStatus = useShellStore((s) => s.pushStatus);
  const [dto, setDto] = useState<MaterialInstanceDto | null>(null);
  const [overrides, setOverrides] = useState<MatOverridesDto | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const d = await assetsIpc.getMaterialInstance(id);
        if (cancelled) return;
        setDto(d);
        setOverrides(d.overrides);
        setDirty(false);
      } catch (e) {
        console.error("asset.getMaterialInstance failed", e);
        if (!cancelled) setDto(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id]);

  const patch = (p: Partial<MatOverridesDto>) => {
    setOverrides((o) => (o ? { ...o, ...p } : o));
    setDirty(true);
  };

  const save = async () => {
    if (!overrides) return;
    setSaving(true);
    try {
      await assetsIpc.saveMaterialInstance(id, overrides);
      setDirty(false);
      pushStatus("Saved material instance overrides.");
    } catch (e) {
      pushStatus(`Save failed: ${String(e)}`);
    } finally {
      setSaving(false);
    }
  };

  const applyToSelection = async () => {
    try {
      const n = await sceneIpc.applyMaterial(id);
      pushStatus(
        n > 0
          ? `Re-applied to ${n} actor(s).`
          : "Select an actor with a material first, then apply.",
      );
    } catch (e) {
      pushStatus(`Apply failed: ${String(e)}`);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
        <button className={btnCls} onClick={onClose} title="Back to content">
          <ArrowLeft size={13} /> Back
        </button>
        <div className="flex items-center gap-1.5 text-xs">
          <Paintbrush size={13} className="text-(--ink-accent)" />
          <span className="font-semibold">Material Instance</span>
          {dto && (
            <span className="text-(--ink-text-dim)">
              — inherits <span className="text-(--ink-text)">{dto.parent_name}</span>
            </span>
          )}
        </div>
        <div className="flex-1" />
        <button
          className={cn(btnCls, dirty && "bg-(--ink-accent) text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)")}
          onClick={() => void save()}
          disabled={!dirty || saving}
          title="Save overrides"
        >
          <Save size={13} /> {saving ? "Saving…" : "Save"}
        </button>
      </div>

      {!dto || !overrides ? (
        <div className="p-4 text-xs text-(--ink-text-faint)">Loading instance…</div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto">
          {/* Base color (RGBA) */}
          <OverrideRow
            label="Base Color"
            overridden={overrides.base_color !== null}
            onToggle={(on) =>
              patch({ base_color: on ? [...dto.resolved.base_color] : null })
            }
          >
            {(() => {
              const v = overrides.base_color ?? dto.resolved.base_color;
              const disabled = overrides.base_color === null;
              return (
                <>
                  <input
                    type="color"
                    className="h-6 w-9 rounded border border-(--ink-border) bg-transparent"
                    value={toHex(v)}
                    disabled={disabled}
                    onChange={(e) => {
                      const [r, g, b] = fromHex(e.target.value);
                      patch({ base_color: [r, g, b, v[3] ?? 1] });
                    }}
                  />
                  <label className="flex items-center gap-1 text-[10px] text-(--ink-text-dim)">
                    A
                    <input
                      type="number"
                      className={numCls}
                      min={0}
                      max={1}
                      step={0.05}
                      value={v[3] ?? 1}
                      disabled={disabled}
                      onChange={(e) =>
                        patch({ base_color: [v[0], v[1], v[2], Number(e.target.value)] })
                      }
                    />
                  </label>
                </>
              );
            })()}
          </OverrideRow>

          {/* Metallic */}
          <OverrideRow
            label="Metallic"
            overridden={overrides.metallic !== null}
            onToggle={(on) => patch({ metallic: on ? dto.resolved.metallic : null })}
          >
            {(() => {
              const v = overrides.metallic ?? dto.resolved.metallic;
              const disabled = overrides.metallic === null;
              return (
                <>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={v}
                    disabled={disabled}
                    onChange={(e) => patch({ metallic: Number(e.target.value) })}
                  />
                  <input
                    type="number"
                    className={numCls}
                    min={0}
                    max={1}
                    step={0.05}
                    value={v}
                    disabled={disabled}
                    onChange={(e) => patch({ metallic: Number(e.target.value) })}
                  />
                </>
              );
            })()}
          </OverrideRow>

          {/* Roughness */}
          <OverrideRow
            label="Roughness"
            overridden={overrides.roughness !== null}
            onToggle={(on) => patch({ roughness: on ? dto.resolved.roughness : null })}
          >
            {(() => {
              const v = overrides.roughness ?? dto.resolved.roughness;
              const disabled = overrides.roughness === null;
              return (
                <>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={v}
                    disabled={disabled}
                    onChange={(e) => patch({ roughness: Number(e.target.value) })}
                  />
                  <input
                    type="number"
                    className={numCls}
                    min={0}
                    max={1}
                    step={0.05}
                    value={v}
                    disabled={disabled}
                    onChange={(e) => patch({ roughness: Number(e.target.value) })}
                  />
                </>
              );
            })()}
          </OverrideRow>

          {/* Emissive (RGB) */}
          <OverrideRow
            label="Emissive"
            overridden={overrides.emissive !== null}
            onToggle={(on) => patch({ emissive: on ? [...dto.resolved.emissive] : null })}
          >
            {(() => {
              const v = overrides.emissive ?? dto.resolved.emissive;
              const disabled = overrides.emissive === null;
              return (
                <input
                  type="color"
                  className="h-6 w-9 rounded border border-(--ink-border) bg-transparent"
                  value={toHex(v)}
                  disabled={disabled}
                  onChange={(e) => patch({ emissive: fromHex(e.target.value) })}
                />
              );
            })()}
          </OverrideRow>

          {/* Apply footer + caption */}
          <div className="flex items-center gap-2 px-3 py-3">
            <button
              className={cn(btnCls, "border border-(--ink-border)")}
              onClick={() => void applyToSelection()}
            >
              <Paintbrush size={13} /> Apply to Selection
            </button>
            <span className="text-[11px] text-(--ink-text-faint)">
              Placed actors keep their applied copy — re-apply to update.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
