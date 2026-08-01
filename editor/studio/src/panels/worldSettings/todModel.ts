/**
 * Pure helpers for the World Settings time-of-day rows (P17.1).
 *
 * Kept out of the component (mirroring `panels/audio/mixerModel.ts`) so the
 * clock formatting and the range clamps are unit-testable without a DOM. The
 * *astronomy* is deliberately NOT here — sun elevation/azimuth arrive already
 * computed on `TimeOfDayDto`, because a second implementation in TypeScript is
 * exactly the kind of mirror that drifts.
 */

import type { WeatherPresetDto } from "../../bindings/WeatherPresetDto";

/** Seconds in one solar day — the modulus the clock lives in. */
export const SECONDS_PER_DAY = 86_400;

/** Days in the engine's fixed, leap-free year. */
export const DAYS_PER_YEAR = 365;

/** Wrap an arbitrary seconds value into `[0, 86400)`. */
export function wrapSeconds(seconds: number): number {
  if (!Number.isFinite(seconds)) return 0;
  const s = seconds % SECONDS_PER_DAY;
  return s < 0 ? s + SECONDS_PER_DAY : s;
}

/** `HH:MM` for the slider readback. */
export function formatClock(seconds: number): string {
  const total = Math.floor(wrapSeconds(seconds));
  const hh = Math.floor(total / 3600);
  const mm = Math.floor(total / 60) % 60;
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}

/** `HH:MM:SS`, for a tooltip / precise readout. */
export function formatClockSeconds(seconds: number): string {
  const total = Math.floor(wrapSeconds(seconds));
  return `${formatClock(seconds)}:${String(total % 60).padStart(2, "0")}`;
}

/** Clamp a day-of-year into `1..=365`, rounding a slider's float. */
export function clampDayOfYear(day: number): number {
  if (!Number.isFinite(day)) return 1;
  return Math.min(DAYS_PER_YEAR, Math.max(1, Math.round(day)));
}

/** Clamp a latitude into `[-90, 90]`. */
export function clampLatitude(deg: number): number {
  if (!Number.isFinite(deg)) return 0;
  return Math.min(90, Math.max(-90, deg));
}

/** Clamp a longitude into `[-180, 180]`. */
export function clampLongitude(deg: number): number {
  if (!Number.isFinite(deg)) return 0;
  return Math.min(180, Math.max(-180, deg));
}

/**
 * A human label for the sun's altitude — what the panel shows beside the
 * numbers so an author can tell noon from midnight at a glance.
 */
export function sunLabel(elevationDeg: number): string {
  if (!Number.isFinite(elevationDeg)) return "—";
  const rounded = Math.round(elevationDeg);
  if (elevationDeg > 0) return `${rounded}° above horizon`;
  // Civil twilight runs to −6°, nautical to −12°, astronomical to −18°.
  if (elevationDeg > -6) return `${rounded}° — twilight`;
  return `${rounded}° — night`;
}

/** Compass point for an azimuth in degrees clockwise from north. */
export function compassPoint(azimuthDeg: number): string {
  if (!Number.isFinite(azimuthDeg)) return "—";
  const points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  const idx = Math.round((((azimuthDeg % 360) + 360) % 360) / 45) % 8;
  return points[idx];
}

/**
 * Meteorological visibility implied by a height-fog density, as a label.
 *
 * Koschmieder's law: a black object vanishes into the haze at the distance where
 * contrast drops to ~2 %, i.e. `V = 3 / sigma`. The panel shows this because
 * "0.0004 per metre" means nothing to anyone, whereas "visibility ~7.5 km" is a
 * number an author can picture and can check against their level's scale.
 *
 * `density` is in **m^-1** (SI, matching `SkyAtmosphere.fog_density`); 0 or
 * non-finite reads as clear air.
 */
export function fogVisibility(density: number): string {
  if (!Number.isFinite(density) || density <= 0) return "clear (no fog)";
  const metres = 3 / density;
  if (metres >= 1000) return `~${(metres / 1000).toFixed(metres < 10000 ? 1 : 0)} km`;
  if (metres >= 10) return `~${Math.round(metres)} m`;
  return "< 10 m";
}

/**
 * Plain-language sky-cover label for a cloud `coverage` value (P17.3).
 *
 * The slider is a **bias on a procedural weather field**, not a literal area
 * fraction — 0.35 realises about 60 % sky cover, which nobody would guess from
 * the number. Rather than pretend otherwise, the panel names the result in the
 * terms a meteorologist (and every flight sim) uses, so an author picks the sky
 * they want instead of hunting for it: clear, few, scattered, broken, overcast.
 *
 * The thresholds come from the renderer's measured coverage response (see
 * `inf_render::clouds::weather`), not from the octa scale directly — the two
 * agree on the words, which is the point.
 */
export function cloudCoverLabel(coverage: number): string {
  if (!Number.isFinite(coverage) || coverage <= 0.2) return "clear";
  if (coverage < 0.3) return "few";
  if (coverage < 0.42) return "scattered";
  if (coverage < 0.55) return "broken";
  return "overcast";
}

/**
 * Cloud-layer thickness label for the World Settings readback (P17.3).
 *
 * Both altitudes are **metres** (SI, matching `SkyAtmosphere.cloud_bottom` /
 * `cloud_top`). A slab whose top is at or below its bottom draws nothing at all,
 * which is a silent no-op the author would otherwise have to discover by
 * squinting at an empty sky — so it is named here instead.
 */
export function cloudLayerLabel(bottom: number, top: number): string {
  if (!Number.isFinite(bottom) || !Number.isFinite(top)) return "—";
  const thickness = top - bottom;
  if (thickness <= 0) return "empty (top is not above bottom)";
  if (thickness >= 1000) return `${(thickness / 1000).toFixed(1)} km thick`;
  return `${Math.round(thickness)} m thick`;
}

// ── weather (P17.4) ─────────────────────────────────────────────────────────

/**
 * The five weather presets, in menu order — the frontend mirror of
 * `inf_ecs::components::WeatherPreset::ALL`.
 *
 * Typed as the generated `WeatherPresetDto` union, so adding a preset in Rust
 * and forgetting it here is a **type error** rather than a missing button.
 */
export const WEATHER_PRESETS = [
  "clear",
  "overcast",
  "storm",
  "fog",
  "snow",
] as const satisfies readonly WeatherPresetDto[];

/**
 * The numbers a preset button writes — the TypeScript mirror of
 * `WeatherPreset::params()`.
 *
 * Duplicated across the ring boundary on purpose: the button must produce its
 * values *optimistically*, before the IPC round-trip, or clicking "Storm" would
 * leave the sliders showing the old sky for a debounce interval. The Rust side
 * remains authoritative — `WeatherDto::snapped_to` recomputes the same state
 * server-side from `WeatherPreset::params()`, so a drift here is corrected on
 * the next snapshot rather than persisted. `weather_preset_table_matches_rust`
 * pins the two together.
 */
export function weatherPreset(preset: WeatherPresetDto): {
  coverage: number;
  cloud_type: number;
  wind_x: number;
  wind_z: number;
  fog_density: number;
  precipitation: number;
  snowiness: number;
} {
  switch (preset) {
    case "overcast":
      return {
        coverage: 0.85,
        cloud_type: 0.2,
        wind_x: 9.0,
        wind_z: 3.5,
        fog_density: 7.5e-5,
        precipitation: 0.0,
        snowiness: 0.0,
      };
    case "storm":
      return {
        coverage: 1.0,
        cloud_type: 0.35,
        wind_x: 22.0,
        wind_z: 9.0,
        fog_density: 6.0e-4,
        precipitation: 1.0,
        snowiness: 0.0,
      };
    case "fog":
      return {
        coverage: 0.5,
        cloud_type: 0.1,
        wind_x: 1.5,
        wind_z: 0.5,
        fog_density: 6.0e-3,
        precipitation: 0.0,
        snowiness: 0.0,
      };
    case "snow":
      return {
        coverage: 0.9,
        cloud_type: 0.3,
        wind_x: 5.0,
        wind_z: 2.0,
        fog_density: 1.2e-3,
        precipitation: 0.7,
        snowiness: 1.0,
      };
    case "clear":
    default:
      return {
        coverage: 0.08,
        cloud_type: 0.75,
        wind_x: 4.0,
        wind_z: 1.5,
        fog_density: 0.0,
        precipitation: 0.0,
        snowiness: 0.0,
      };
  }
}

/**
 * Plain-language readback for the precipitation pair, so an author can see what
 * "0.35 / 0.8" is going to look like without rendering it.
 *
 * `snowiness` is a blend rather than a switch, so the words follow it: below a
 * third it is rain, above two thirds it is snow, and the band between is sleet —
 * which is what a continuous rain-to-snow transition physically passes through.
 */
export function precipLabel(intensity: number, snowiness: number): string {
  if (!Number.isFinite(intensity) || intensity <= 0) return "dry";
  const s = Number.isFinite(snowiness) ? Math.min(Math.max(snowiness, 0), 1) : 0;
  const kind = s < 1 / 3 ? "rain" : s > 2 / 3 ? "snow" : "sleet";
  const i = Math.min(Math.max(intensity, 0), 1);
  const strength = i < 0.25 ? "light" : i < 0.6 ? "steady" : i < 0.9 ? "heavy" : "torrential";
  return `${strength} ${kind}`;
}
