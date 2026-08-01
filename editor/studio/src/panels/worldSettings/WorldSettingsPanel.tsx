/**
 * World Settings: honest, per-scene information plus the render/viewport
 * settings that are genuinely IPC-backed today.
 *
 * - Scene facts (title, unsaved state, entity counts, revision) read from the
 *   live `sceneStore` — the frontend mirror of the backend ECS world.
 * - Project roots read from `projectStore` (read-only info rows).
 * - The projection mode and the 2D grid/pixel snap settings drive the native
 *   viewport through `viewportStore` (the only render settings the webview can
 *   currently set: `viewport_set_mode` / `viewport_set_snap2d`).
 *
 * Rows with no backend setter are read-only by design — we do not fabricate
 * editable settings that nothing is listening to. See the task report for
 * world-level settings that deserve backend support later.
 */
import type { ReactNode } from "react";

import type { LevelSettingsDto } from "../../bindings/LevelSettingsDto";
import type { RenderSettingsRecordDto } from "../../bindings/RenderSettingsRecordDto";
import {
  CheckboxField,
  NumberField,
  PropertyRow,
  PropertySection,
  Vec3Field,
} from "../../components/propertyRows";
import { useProjectStore } from "../../stores/projectStore";
import { useSceneStore } from "../../stores/sceneStore";
import { useViewportStore } from "../../stores/viewportStore";

/** A read-only value cell for info rows that have no setter. */
function ReadOnly({ children }: { children: ReactNode }) {
  return <span className="min-w-0 flex-1 truncate text-xs text-(--ink-text)">{children}</span>;
}

export default function WorldSettingsPanel() {
  const ready = useSceneStore((s) => s.ready);
  const title = useSceneStore((s) => s.title);
  const dirty = useSceneStore((s) => s.dirty);
  const version = useSceneStore((s) => s.version);
  const entityCount = useSceneStore((s) => Object.keys(s.nodes).length);
  const rootCount = useSceneStore((s) => s.roots.length);
  const selCount = useSceneStore((s) => s.selection.length);

  const project = useProjectStore((s) => s.current);

  const settings = useSceneStore((s) => s.levelSettings);
  const setLevelSettings = useSceneStore((s) => s.setLevelSettings);
  const patchWorld = (patch: Partial<LevelSettingsDto>) =>
    settings && setLevelSettings({ ...settings, ...patch });
  const patchRender = (patch: Partial<RenderSettingsRecordDto>) =>
    settings && setLevelSettings({ ...settings, render: { ...settings.render, ...patch } });

  const mode = useViewportStore((s) => s.mode);
  const setMode = useViewportStore((s) => s.setMode);
  const gridSnapEnabled = useViewportStore((s) => s.gridSnapEnabled);
  const setGridSnapEnabled = useViewportStore((s) => s.setGridSnapEnabled);
  const gridSnapSize = useViewportStore((s) => s.gridSnapSize);
  const setGridSnapSize = useViewportStore((s) => s.setGridSnapSize);
  const pixelSnapEnabled = useViewportStore((s) => s.pixelSnapEnabled);
  const setPixelSnapEnabled = useViewportStore((s) => s.setPixelSnapEnabled);
  const pixelsPerUnit = useViewportStore((s) => s.pixelsPerUnit);
  const setPixelsPerUnit = useViewportStore((s) => s.setPixelsPerUnit);
  // Terrain edits live in the `.inf_terrain` asset and are written ONLY by an
  // explicit save (autosave never touches assets — P16.4b), so the level's
  // "Unsaved" row above does not cover them. Say so here, next to it.
  const terrainStreamed = useViewportStore((s) => s.terrainStreamed);
  const terrainUnsavedEdits = useViewportStore((s) => s.terrainUnsavedEdits);

  return (
    <div className="min-h-0 flex-1 overflow-auto">
      <PropertySection title="Scene">
        <PropertyRow label="Title">
          <ReadOnly>{ready ? title : "Loading…"}</ReadOnly>
        </PropertyRow>
        <PropertyRow label="Unsaved">
          <ReadOnly>
            {dirty ? (
              <span className="text-(--ink-warning)">Yes — unsaved changes</span>
            ) : (
              "No"
            )}
          </ReadOnly>
        </PropertyRow>
        <PropertyRow label="Terrain">
          <ReadOnly>
            {terrainUnsavedEdits ? (
              <span className="text-(--ink-warning)">
                Unsaved edits — Ctrl+S writes them into the .inf_terrain asset
              </span>
            ) : terrainStreamed ? (
              "Streamed (.inf_terrain)"
            ) : (
              "Inline"
            )}
          </ReadOnly>
        </PropertyRow>
        <PropertyRow label="Actors">
          <ReadOnly>{entityCount}</ReadOnly>
        </PropertyRow>
        <PropertyRow label="Root Actors">
          <ReadOnly>{rootCount}</ReadOnly>
        </PropertyRow>
        <PropertyRow label="Selected">
          <ReadOnly>{selCount}</ReadOnly>
        </PropertyRow>
        <PropertyRow label="Revision">
          <ReadOnly>{version}</ReadOnly>
        </PropertyRow>
      </PropertySection>

      {settings && (
        <>
          <PropertySection title="World">
            <PropertyRow label="Gravity 2D">
              <div className="flex min-w-0 flex-1 gap-1">
                <NumberField
                  value={settings.gravity_2d[0]}
                  step={0.1}
                  onChange={(v) => patchWorld({ gravity_2d: [v, settings.gravity_2d[1]] })}
                />
                <NumberField
                  value={settings.gravity_2d[1]}
                  step={0.1}
                  onChange={(v) => patchWorld({ gravity_2d: [settings.gravity_2d[0], v] })}
                />
              </div>
            </PropertyRow>
            <PropertyRow label="Gravity 3D">
              <Vec3Field
                value={settings.gravity_3d}
                onChange={(v) => patchWorld({ gravity_3d: v })}
              />
            </PropertyRow>
            <PropertyRow label="Sim Rate (Hz)">
              <NumberField
                value={settings.sim_hz}
                step={1}
                onChange={(v) => patchWorld({ sim_hz: v })}
              />
            </PropertyRow>
          </PropertySection>

          <PropertySection title="Rendering">
            <PropertyRow label="Exposure">
              <NumberField
                value={settings.render.exposure}
                step={0.1}
                onChange={(v) => patchRender({ exposure: v })}
              />
            </PropertyRow>
            <PropertyRow label="Dither">
              <CheckboxField
                value={settings.render.dither}
                onChange={(v) => patchRender({ dither: v })}
              />
            </PropertyRow>
            <PropertyRow label="Temporal AA">
              <CheckboxField
                value={settings.render.taa}
                onChange={(v) => patchRender({ taa: v })}
              />
            </PropertyRow>
          </PropertySection>

          <PropertySection title="Bloom">
            <PropertyRow label="Enabled">
              <CheckboxField
                value={settings.render.bloom_enabled}
                onChange={(v) => patchRender({ bloom_enabled: v })}
              />
            </PropertyRow>
            <PropertyRow label="Threshold">
              <NumberField
                value={settings.render.bloom_threshold}
                step={0.05}
                onChange={(v) => patchRender({ bloom_threshold: v })}
              />
            </PropertyRow>
            <PropertyRow label="Knee">
              <NumberField
                value={settings.render.bloom_knee}
                step={0.05}
                onChange={(v) => patchRender({ bloom_knee: v })}
              />
            </PropertyRow>
            <PropertyRow label="Intensity">
              <NumberField
                value={settings.render.bloom_intensity}
                step={0.01}
                onChange={(v) => patchRender({ bloom_intensity: v })}
              />
            </PropertyRow>
          </PropertySection>

          <PropertySection title="SSAO">
            <PropertyRow label="Enabled">
              <CheckboxField
                value={settings.render.ssao_enabled}
                onChange={(v) => patchRender({ ssao_enabled: v })}
              />
            </PropertyRow>
            <PropertyRow label="Radius">
              <NumberField
                value={settings.render.ssao_radius}
                step={0.05}
                onChange={(v) => patchRender({ ssao_radius: v })}
              />
            </PropertyRow>
            <PropertyRow label="Intensity">
              <NumberField
                value={settings.render.ssao_intensity}
                step={0.05}
                onChange={(v) => patchRender({ ssao_intensity: v })}
              />
            </PropertyRow>
            <PropertyRow label="Bias">
              <NumberField
                value={settings.render.ssao_bias}
                step={0.005}
                onChange={(v) => patchRender({ ssao_bias: v })}
              />
            </PropertyRow>
          </PropertySection>

          <PropertySection title="Shadows">
            <PropertyRow label="Enabled">
              <CheckboxField
                value={settings.render.shadows_enabled}
                onChange={(v) => patchRender({ shadows_enabled: v })}
              />
            </PropertyRow>
            <PropertyRow label="Max Distance">
              <NumberField
                value={settings.render.shadows_max_distance}
                step={5}
                onChange={(v) => patchRender({ shadows_max_distance: v })}
              />
            </PropertyRow>
          </PropertySection>

          <PropertySection title="Global Illumination">
            <PropertyRow label="Enabled">
              <CheckboxField
                value={settings.render.gi_enabled}
                onChange={(v) => patchRender({ gi_enabled: v })}
              />
            </PropertyRow>
            <PropertyRow label="Intensity">
              <NumberField
                value={settings.render.gi_intensity}
                step={0.05}
                onChange={(v) => patchRender({ gi_intensity: v })}
              />
            </PropertyRow>
          </PropertySection>
        </>
      )}

      <PropertySection title="Project">
        {project ? (
          <>
            <PropertyRow label="Name">
              <ReadOnly>{project.name}</ReadOnly>
            </PropertyRow>
            <PropertyRow label="Template">
              <ReadOnly>{project.template}</ReadOnly>
            </PropertyRow>
            <PropertyRow label="Root">
              <ReadOnly>
                <span className="font-mono text-[11px] text-(--ink-text-dim)">{project.root}</span>
              </ReadOnly>
            </PropertyRow>
            <PropertyRow label="Content">
              <ReadOnly>
                <span className="font-mono text-[11px] text-(--ink-text-dim)">
                  {project.content_dir}
                </span>
              </ReadOnly>
            </PropertyRow>
            <PropertyRow label="Levels">
              <ReadOnly>
                <span className="font-mono text-[11px] text-(--ink-text-dim)">
                  {project.levels_dir}
                </span>
              </ReadOnly>
            </PropertyRow>
          </>
        ) : (
          <div className="px-2 py-1 text-xs text-(--ink-text-faint)">No project is open.</div>
        )}
      </PropertySection>

      <PropertySection title="Viewport">
        <PropertyRow label="Projection">
          <div className="flex min-w-0 flex-1 items-center rounded bg-(--ink-bg-1) p-0.5">
            {(
              [
                ["Perspective", "Perspective"],
                ["TwoD", "2D"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                onClick={() => setMode(value)}
                className={`flex h-5 flex-1 items-center justify-center rounded px-2 text-xs ${
                  mode === value
                    ? "bg-(--ink-bg-3) text-(--ink-text)"
                    : "text-(--ink-text-dim) hover:text-(--ink-text)"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </PropertyRow>
      </PropertySection>

      <PropertySection title="2D Snapping">
        <PropertyRow label="Grid Snap">
          <CheckboxField value={gridSnapEnabled} onChange={setGridSnapEnabled} />
        </PropertyRow>
        <PropertyRow label="Grid Size">
          <NumberField value={gridSnapSize} step={0.25} onChange={setGridSnapSize} />
        </PropertyRow>
        <PropertyRow label="Pixel Snap">
          <CheckboxField value={pixelSnapEnabled} onChange={setPixelSnapEnabled} />
        </PropertyRow>
        <PropertyRow label="Pixels / Unit">
          <NumberField value={pixelsPerUnit} step={1} onChange={setPixelsPerUnit} />
        </PropertyRow>
      </PropertySection>
    </div>
  );
}
