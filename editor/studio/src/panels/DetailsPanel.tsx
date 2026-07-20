/**
 * Details panel (P1.4.2): property rows for the current selection. Mock
 * per-actor property state until reflection-driven population (P3.3);
 * the row primitives in components/propertyRows.tsx are the real ones.
 */
import { useState } from "react";
import { Search } from "lucide-react";
import {
  CheckboxField,
  ColorField,
  EnumField,
  PropertyRow,
  PropertySection,
  TextField,
  Vec3Field,
} from "../components/propertyRows";
import { fuzzyMatch } from "../lib/fuzzy";
import { findActor, useSceneStore } from "../stores/sceneStore";

interface MockProps {
  transform: {
    location: [number, number, number];
    rotation: [number, number, number];
    scale: [number, number, number];
  };
  mobility: string;
  castShadows: boolean;
  tint: string;
}

const DEFAULTS: MockProps = {
  transform: { location: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] },
  mobility: "Static",
  castShadows: true,
  tint: "#ffffff",
};

export default function DetailsPanel() {
  const selectedIds = useSceneStore((s) => s.selectedIds);
  const roots = useSceneStore((s) => s.roots);
  const [filter, setFilter] = useState("");
  // Per-actor mock property state, keyed by actor id. Dropped when the
  // real reflection pipeline lands (P3.3).
  const [propsById, setPropsById] = useState<Record<string, MockProps>>({});

  const id = selectedIds[0] ?? null;
  const actor = id ? findActor(roots, id) : null;

  if (!actor) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center p-4 text-xs text-(--ink-text-dim)">
        {selectedIds.length > 1 ? "Multi-object editing lands with P3.3" : "Select an object to view details"}
      </div>
    );
  }

  const props = propsById[actor.id] ?? DEFAULTS;
  const update = (patch: Partial<MockProps>) =>
    setPropsById((all) => ({ ...all, [actor.id]: { ...props, ...patch } }));
  const show = (label: string) => !filter.trim() || fuzzyMatch(filter.trim(), label) !== null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Selection header + property filter */}
      <div className="border-b border-(--ink-border) p-1.5">
        <div className="flex items-center gap-2 px-0.5 pb-1.5">
          <span className="truncate text-xs font-semibold">{actor.name}</span>
          <span className="truncate text-[11px] text-(--ink-text-faint)">{actor.type}</span>
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
        <PropertySection title="Transform">
          {show("Location") && (
            <PropertyRow
              label="Location"
              changed={props.transform.location.some((v) => v !== 0)}
              onReset={() =>
                update({ transform: { ...props.transform, location: [0, 0, 0] } })
              }
            >
              <Vec3Field
                value={props.transform.location}
                onChange={(location) => update({ transform: { ...props.transform, location } })}
              />
            </PropertyRow>
          )}
          {show("Rotation") && (
            <PropertyRow
              label="Rotation"
              changed={props.transform.rotation.some((v) => v !== 0)}
              onReset={() =>
                update({ transform: { ...props.transform, rotation: [0, 0, 0] } })
              }
            >
              <Vec3Field
                value={props.transform.rotation}
                onChange={(rotation) => update({ transform: { ...props.transform, rotation } })}
              />
            </PropertyRow>
          )}
          {show("Scale") && (
            <PropertyRow
              label="Scale"
              changed={props.transform.scale.some((v) => v !== 1)}
              onReset={() => update({ transform: { ...props.transform, scale: [1, 1, 1] } })}
            >
              <Vec3Field
                value={props.transform.scale}
                onChange={(scale) => update({ transform: { ...props.transform, scale } })}
              />
            </PropertyRow>
          )}
        </PropertySection>

        <PropertySection title="General">
          {show("Name") && (
            <PropertyRow label="Name">
              <TextField value={actor.name} onChange={() => {}} placeholder={actor.name} />
            </PropertyRow>
          )}
          {show("Mobility") && (
            <PropertyRow
              label="Mobility"
              changed={props.mobility !== DEFAULTS.mobility}
              onReset={() => update({ mobility: DEFAULTS.mobility })}
            >
              <EnumField
                value={props.mobility}
                options={["Static", "Stationary", "Movable"]}
                onChange={(mobility) => update({ mobility })}
              />
            </PropertyRow>
          )}
        </PropertySection>

        <PropertySection title="Rendering">
          {show("Cast Shadows") && (
            <PropertyRow
              label="Cast Shadows"
              changed={props.castShadows !== DEFAULTS.castShadows}
              onReset={() => update({ castShadows: DEFAULTS.castShadows })}
            >
              <CheckboxField
                value={props.castShadows}
                onChange={(castShadows) => update({ castShadows })}
              />
            </PropertyRow>
          )}
          {show("Tint") && (
            <PropertyRow
              label="Tint"
              changed={props.tint !== DEFAULTS.tint}
              onReset={() => update({ tint: DEFAULTS.tint })}
            >
              <ColorField value={props.tint} onChange={(tint) => update({ tint })} />
            </PropertyRow>
          )}
        </PropertySection>
      </div>
    </div>
  );
}
