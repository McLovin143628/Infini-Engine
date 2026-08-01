/**
 * Pure helpers for the World Settings time-of-day rows (P17.1).
 *
 * Kept out of the component (mirroring `panels/audio/mixerModel.ts`) so the
 * clock formatting and the range clamps are unit-testable without a DOM. The
 * *astronomy* is deliberately NOT here — sun elevation/azimuth arrive already
 * computed on `TimeOfDayDto`, because a second implementation in TypeScript is
 * exactly the kind of mirror that drifts.
 */

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
