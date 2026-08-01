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
import type { PartitionSettingsDto } from "../../bindings/PartitionSettingsDto";
import type { RenderSettingsRecordDto } from "../../bindings/RenderSettingsRecordDto";
import type { SkyAtmosphereDto } from "../../bindings/SkyAtmosphereDto";
import type { TimeOfDayDto } from "../../bindings/TimeOfDayDto";
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
import {
  clampDayOfYear,
  clampLatitude,
  clampLongitude,
  compassPoint,
  formatClock,
  SECONDS_PER_DAY,
  fogVisibility,
  sunLabel,
  wrapSeconds,
} from "./todModel";

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
  const patchPartition = (patch: Partial<PartitionSettingsDto>) =>
    settings && setLevelSettings({ ...settings, partition: { ...settings.partition, ...patch } });
  // Every time-of-day write sets `present: true` — that is the signal the backend
  // reads as "create the sky authority if this level has no clock yet", which is
  // how a level opts into a dynamic sun from this panel.
  const patchTod = (patch: Partial<TimeOfDayDto>) =>
    settings &&
    setLevelSettings({
      ...settings,
      time_of_day: { ...settings.time_of_day, present: true, ...patch },
    });
  // Same story for the atmosphere: it lives on the same authority entity, so a
  // write here also carries `present: true` and opts a clockless level in.
  const patchAtmos = (patch: Partial<SkyAtmosphereDto>) =>
    settings &&
    setLevelSettings({
      ...settings,
      atmosphere: { ...settings.atmosphere, present: true, ...patch },
    });

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

          <PropertySection title="Time of Day">
            <p className="px-2 pb-1 text-[11px] leading-snug text-(--ink-text-dim)">
              Drives the sun and moon from a real solar model. Editing any row gives this
              level a <span className="text-(--ink-text)">Sky</span> actor carrying the Time
              of Day and Sky Atmosphere components &mdash; tune the rest in Details, key the
              time in the Sequencer, or drive it from Blueprints.
              {!settings.time_of_day.present && " This level has no clock yet."}
            </p>
            <PropertyRow label="Time">
              <div className="flex min-w-0 flex-1 items-center gap-2">
                <input
                  type="range"
                  min={0}
                  max={SECONDS_PER_DAY - 1}
                  step={60}
                  value={wrapSeconds(settings.time_of_day.seconds)}
                  onChange={(e) => patchTod({ seconds: wrapSeconds(Number(e.target.value)) })}
                  className="min-w-0 flex-1 accent-(--ink-accent)"
                  aria-label="Time of day"
                />
                <span className="w-12 shrink-0 text-right font-mono text-xs text-(--ink-text)">
                  {formatClock(settings.time_of_day.seconds)}
                </span>
              </div>
            </PropertyRow>
            <PropertyRow label="Day of Year">
              <NumberField
                value={settings.time_of_day.day_of_year}
                step={1}
                onChange={(v) => patchTod({ day_of_year: clampDayOfYear(v) })}
              />
            </PropertyRow>
            <PropertyRow label="Latitude (°N)">
              <NumberField
                value={settings.time_of_day.latitude_deg}
                step={1}
                onChange={(v) => patchTod({ latitude_deg: clampLatitude(v) })}
              />
            </PropertyRow>
            <PropertyRow label="Longitude (°E)">
              <NumberField
                value={settings.time_of_day.longitude_deg}
                step={1}
                onChange={(v) => patchTod({ longitude_deg: clampLongitude(v) })}
              />
            </PropertyRow>
            <PropertyRow label="Rate (×)">
              <NumberField
                value={settings.time_of_day.rate}
                step={1}
                onChange={(v) => patchTod({ rate: v })}
              />
            </PropertyRow>
            <PropertyRow label="Sun">
              <ReadOnly>
                {sunLabel(settings.time_of_day.sun_elevation_deg)}
                {" \u00b7 "}
                {compassPoint(settings.time_of_day.sun_azimuth_deg)}
              </ReadOnly>
            </PropertyRow>
          </PropertySection>

          <PropertySection title="Atmosphere">
            <p className="px-2 pb-1 text-[11px] leading-snug text-(--ink-text-dim)">
              A physically-based sky &mdash; Rayleigh, Mie and ozone scattering baked into
              lookup tables each time the sun moves &mdash; plus sun and moon discs, stars, and
              distance scattering on everything the camera can see. Colours live in Details on
              the <span className="text-(--ink-text)">Sky</span> actor.
              {!settings.atmosphere.present && " This level has no sky yet."}
            </p>
            <PropertyRow label="Physical Sky">
              <CheckboxField
                value={settings.atmosphere.physical}
                onChange={(v) => patchAtmos({ physical: v })}
              />
            </PropertyRow>
            <PropertyRow label="Sun Lights Scene">
              <CheckboxField
                value={settings.atmosphere.enabled}
                onChange={(v) => patchAtmos({ enabled: v })}
              />
            </PropertyRow>
            <PropertyRow label="Sky Intensity">
              <NumberField
                value={settings.atmosphere.sky_intensity}
                step={0.1}
                onChange={(v) => patchAtmos({ sky_intensity: v })}
              />
            </PropertyRow>
            <PropertyRow label="Turbidity (haze)">
              <NumberField
                value={settings.atmosphere.turbidity}
                step={0.1}
                onChange={(v) => patchAtmos({ turbidity: v })}
              />
            </PropertyRow>
            <PropertyRow label="Haze Anisotropy">
              <NumberField
                value={settings.atmosphere.mie_anisotropy}
                step={0.05}
                onChange={(v) => patchAtmos({ mie_anisotropy: v })}
              />
            </PropertyRow>
            <PropertyRow label="Sun Size (°)">
              <NumberField
                value={settings.atmosphere.sun_disc_deg}
                step={0.05}
                onChange={(v) => patchAtmos({ sun_disc_deg: v })}
              />
            </PropertyRow>
            <PropertyRow label="Moon Size (°)">
              <NumberField
                value={settings.atmosphere.moon_disc_deg}
                step={0.05}
                onChange={(v) => patchAtmos({ moon_disc_deg: v })}
              />
            </PropertyRow>
            <PropertyRow label="Stars">
              <NumberField
                value={settings.atmosphere.star_intensity}
                step={0.1}
                onChange={(v) => patchAtmos({ star_intensity: v })}
              />
            </PropertyRow>
            <PropertyRow label="Gradient Tint">
              <NumberField
                value={settings.atmosphere.tint_strength}
                step={0.05}
                onChange={(v) => patchAtmos({ tint_strength: v })}
              />
            </PropertyRow>
            <PropertyRow label="Aerial Perspective">
              <NumberField
                value={settings.atmosphere.aerial_perspective}
                step={0.1}
                onChange={(v) => patchAtmos({ aerial_perspective: v })}
              />
            </PropertyRow>
            <PropertyRow label="Fog Density (1/m)">
              <NumberField
                value={settings.atmosphere.fog_density}
                step={0.0001}
                onChange={(v) => patchAtmos({ fog_density: v })}
              />
            </PropertyRow>
            <PropertyRow label="Fog Falloff (1/m)">
              <NumberField
                value={settings.atmosphere.fog_falloff}
                step={0.0005}
                onChange={(v) => patchAtmos({ fog_falloff: v })}
              />
            </PropertyRow>
            <PropertyRow label="Fog Height (m)">
              <NumberField
                value={settings.atmosphere.fog_height}
                step={1}
                onChange={(v) => patchAtmos({ fog_height: v })}
              />
            </PropertyRow>
            <PropertyRow label="Visibility">
              <ReadOnly>{fogVisibility(settings.atmosphere.fog_density)}</ReadOnly>
            </PropertyRow>
          </PropertySection>

          <PropertySection title="World Partition">
            <p className="px-2 pb-1 text-[11px] leading-snug text-(--ink-text-dim)">
              Splits this level into streamed cells at cook time. The editor stays a single
              document &mdash; nothing streams while you author. Play-in-editor and a shipped
              build both stream it.
            </p>
            <PropertyRow label="Enabled">
              <CheckboxField
                value={settings.partition.enabled}
                onChange={(v) => patchPartition({ enabled: v })}
              />
            </PropertyRow>
            <PropertyRow label="Cell Size (m)">
              <NumberField
                value={settings.partition.cell_size_m}
                step={16}
                onChange={(v) => patchPartition({ cell_size_m: v })}
              />
            </PropertyRow>
            <PropertyRow label="Activation Radius (m)">
              <NumberField
                value={settings.partition.activation_radius_m}
                step={16}
                onChange={(v) => patchPartition({ activation_radius_m: v })}
              />
            </PropertyRow>
            <PropertyRow label="Prefetch Margin (m)">
              <NumberField
                value={settings.partition.prefetch_margin_m}
                step={16}
                onChange={(v) => patchPartition({ prefetch_margin_m: v })}
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
