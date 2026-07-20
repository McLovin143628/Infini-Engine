/**
 * Details panel (P1.4.2 shell, P3.3 live): reflection-driven property grid for
 * the current selection. Component sections + field rows come from the backend
 * `DetailsDto` (walked from `bevy_reflect`); edits post through `scene_set_property`
 * and apply to every selected object (multi-edit). No hard-coded property list.
 */
import { useState } from "react";
import { Search } from "lucide-react";
import {
  CheckboxField,
  ColorField,
  EnumField,
  NumberField,
  PropertyRow,
  PropertySection,
  TextField,
  Vec3Field,
} from "../components/propertyRows";
import type { PropValueDto } from "../bindings/PropValueDto";
import { fuzzyMatch } from "../lib/fuzzy";
import { useSceneStore } from "../stores/sceneStore";

function rgbaToHex(c: number[]): string {
  const to = (v: number) =>
    Math.round(Math.max(0, Math.min(1, v)) * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${to(c[0] ?? 0)}${to(c[1] ?? 0)}${to(c[2] ?? 0)}`;
}

function hexToRgba(hex: string, a = 1): number[] {
  const n = hex.replace("#", "");
  const p = (i: number) => parseInt(n.slice(i, i + 2), 16) / 255;
  return [p(0), p(2), p(4), a];
}

export default function DetailsPanel() {
  const details = useSceneStore((s) => s.details);
  const setProperty = useSceneStore((s) => s.setProperty);
  const resetProperty = useSceneStore((s) => s.resetProperty);
  const [filter, setFilter] = useState("");

  if (!details || details.selection.length === 0) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center p-4 text-xs text-(--ink-text-dim)">
        Select an object to view details
      </div>
    );
  }

  const show = (label: string) => !filter.trim() || fuzzyMatch(filter.trim(), label) !== null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Header + filter */}
      <div className="border-b border-(--ink-border) p-1.5">
        <div className="flex items-center gap-2 px-0.5 pb-1.5">
          <span className="truncate text-xs font-semibold">{details.name}</span>
          <span className="truncate text-[11px] text-(--ink-text-faint)">{details.kind}</span>
        </div>
        <div className="flex h-6 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 focus-within:border-(--ink-accent)">
          <Search size={12} className="shrink-0 text-(--ink-text-faint)" />
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter properties…"
            className="w-full bg-transparent text-xs outline-none placeholder:text-(--ink-text-faint)"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {details.components.map((comp) => {
          const fields = comp.fields.filter((f) => show(f.label));
          if (fields.length === 0) return null;
          return (
            <PropertySection key={comp.type_path} title={comp.display}>
              {fields.map((field) => {
                const set = (value: PropValueDto) =>
                  void setProperty(comp.type_path, field.name, value);
                const label = field.same ? field.label : `${field.label} *`;
                return (
                  <PropertyRow
                    key={field.name}
                    label={label}
                    changed
                    onReset={() => void resetProperty(comp.type_path, field.name)}
                  >
                    {renderControl(field.value, set)}
                  </PropertyRow>
                );
              })}
            </PropertySection>
          );
        })}
      </div>
    </div>
  );
}

function renderControl(value: PropValueDto, set: (v: PropValueDto) => void) {
  switch (value.kind) {
    case "bool":
      return <CheckboxField value={value.value} onChange={(v) => set({ kind: "bool", value: v })} />;
    case "number":
      return (
        <NumberField value={value.value} onChange={(v) => set({ kind: "number", value: v })} />
      );
    case "text":
      return <TextField value={value.value} onChange={(v) => set({ kind: "text", value: v })} />;
    case "vec3": {
      const v = value.value;
      const tuple: [number, number, number] = [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];
      return (
        <Vec3Field value={tuple} onChange={(next) => set({ kind: "vec3", value: [...next] })} />
      );
    }
    case "color":
      return (
        <ColorField
          value={rgbaToHex(value.value)}
          onChange={(hex) => set({ kind: "color", value: hexToRgba(hex, value.value[3] ?? 1) })}
        />
      );
    case "enum":
      return (
        <EnumField
          value={value.value}
          options={value.options}
          onChange={(v) => set({ kind: "enum", value: v, options: value.options })}
        />
      );
    default:
      return null;
  }
}
