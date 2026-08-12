/**
 * Details panel (P1.4.2 shell, P3.3 live, E-P1 deep editing): reflection-driven
 * property grid for the current selection. Component sections + field rows come
 * from the backend `DetailsDto` (walked from `bevy_reflect`); edits post through
 * `scene_set_property` and apply to every selected object (multi-edit).
 *
 * E-P1 adds arrays (ListField), nested structs (StructField), entity-reference
 * pickers (EntityRefField), and add/remove component. Complex rows (list / struct
 * / entity-ref) are read-only under a multi-selection (rendered "— mixed").
 */
import { useState } from "react";
import { MoreVertical, Search, Trash2 } from "lucide-react";
import {
  CheckboxField,
  ColorField,
  EnumField,
  ListField,
  NumberField,
  PropertyRow,
  PropertySection,
  StructField,
  TextField,
  Vec3Field,
} from "../components/propertyRows";
import { AddComponentMenu } from "../components/AddComponentMenu";
import { EntityRefField } from "../components/EntityRefField";
import { AssetRefField } from "../components/AssetRefField";
import type { PropValueDto } from "../bindings/PropValueDto";
import { scene as sceneIpc } from "../lib/ipc";
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

/** Kinds that carry nested/opaque data — read-only under a multi-selection. */
function isComplex(kind: PropValueDto["kind"]): boolean {
  return kind === "list" || kind === "struct" || kind === "entity_ref" || kind === "asset_ref";
}

export default function DetailsPanel() {
  const details = useSceneStore((s) => s.details);
  const setProperty = useSceneStore((s) => s.setProperty);
  const resetProperty = useSceneStore((s) => s.resetProperty);
  const removeComponent = useSceneStore((s) => s.removeComponent);
  const [filter, setFilter] = useState("");

  if (!details || details.selection.length === 0) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center p-4 text-xs text-(--ink-text-dim)">
        Select an object to view details
      </div>
    );
  }

  const multi = details.multi;
  const primaryGuid = details.selection[0];
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
            <PropertySection
              key={comp.type_path}
              title={comp.display}
              action={
                <ComponentOverflow
                  onRemove={() => void removeComponent(primaryGuid, comp.type_path)}
                />
              }
            >
              {fields.map((field) => {
                const set = (value: PropValueDto) =>
                  void setProperty(comp.type_path, field.name, value);
                const label = field.same ? field.label : `${field.label} *`;
                // Complex rows can't be meaningfully multi-edited: show read-only.
                if (multi && isComplex(field.value.kind)) {
                  return (
                    <PropertyRow key={field.name} label={label}>
                      <span className="text-xs text-(--ink-text-faint)">— mixed / complex</span>
                    </PropertyRow>
                  );
                }
                return (
                  <PropertyRow
                    key={field.name}
                    label={label}
                    changed
                    onReset={() => void resetProperty(comp.type_path, field.name)}
                  >
                    {renderControl(field.value, set, comp.type_path, field.name)}
                  </PropertyRow>
                );
              })}
            </PropertySection>
          );
        })}

        {/* "+ Add Component" (E-P1) — adds to the whole selection. */}
        <div className="p-2">
          <AddComponentMenu />
        </div>
      </div>
    </div>
  );
}

/** Per-section overflow menu with "Remove Component" (E-P1). */
function ComponentOverflow({ onRemove }: { onRemove: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        aria-label="Component options"
        className="flex size-5 items-center justify-center rounded-sm text-(--ink-text-faint) hover:text-(--ink-text)"
        onClick={() => setOpen((o) => !o)}
      >
        <MoreVertical size={13} />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute top-6 right-0 z-50 w-40 rounded border border-(--ink-border) bg-(--ink-bg-1) py-1 shadow-lg">
            <button
              className="flex w-full items-center gap-2 px-2 py-1 text-left text-xs text-(--ink-error) hover:bg-(--ink-bg-3)"
              onClick={() => {
                setOpen(false);
                onRemove();
              }}
            >
              <Trash2 size={12} /> Remove Component
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * Render the widget for a property value. `typePath`/`path` locate the value for
 * list-element defaults (`scene_list_default`). `set` receives the WHOLE updated
 * value (lists/structs replace as a whole; the backend rebuilds the collection).
 */
export function renderControl(
  value: PropValueDto,
  set: (v: PropValueDto) => void,
  typePath: string,
  path: string,
): React.ReactNode {
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
    case "entity_ref":
      return (
        <EntityRefField
          value={value.value}
          onChange={(guid) => set({ kind: "entity_ref", value: guid })}
        />
      );
    // P26.3b: read-only. `set` is deliberately not called — the binding is
    // written by dragging a `.inf_mat` onto the entity (`scene_apply_material`),
    // which is the one call site that knows the parameters and the asset
    // together. See `AssetRefField` for why a picker here would be a second
    // write path for one fact.
    case "asset_ref":
      return <AssetRefField value={value.value} assetKind={value.asset_kind} />;
    case "list":
      return (
        <ListField
          value={value.value}
          onChange={(next) => set({ kind: "list", value: next })}
          onAdd={() => {
            void sceneIpc
              .listDefault(typePath, path)
              .then((elem) => set({ kind: "list", value: [...value.value, elem] }))
              .catch((e) => console.error("listDefault failed", e));
          }}
          renderElement={(elem, index, setElem) =>
            // Element paths are informational; list writes replace the whole list.
            renderControl(elem, setElem, typePath, `${path}[${index}]`)
          }
        />
      );
    case "struct":
      return (
        <StructField
          fields={value.fields}
          onChange={(next) => set({ kind: "struct", fields: next })}
          renderChild={(child, setChild) => renderControl(child, setChild, typePath, path)}
        />
      );
    default:
      return null;
  }
}
