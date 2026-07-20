/**
 * Theme system — JSON themes → CSS variables (ROADMAP §4, P1.1 batch 4).
 *
 * A theme is a plain JSON document (one file per theme under `src/themes/`)
 * mapping the token set below onto `--ink-*` CSS variables on `<html>`.
 * Every component styles itself exclusively through those variables, so a
 * theme swap repaints the whole shell without re-rendering React.
 *
 * Ported pattern from CodeR's theme engine, reduced to Infinity's token set:
 * themes must be *total* (every key present) so `applyTheme` never writes
 * `undefined`; `validateTheme` backfills missing keys from the default dark
 * theme to keep user-supplied JSON forgiving.
 */

import infinityDark from "../themes/infinity-dark.json";
import infinityLight from "../themes/infinity-light.json";
import graphite from "../themes/graphite.json";
import midnight from "../themes/midnight.json";

/** Token keys (CSS variable = `--ink-<key>`). Every theme MUST set all. */
export const THEME_COLOR_KEYS = [
  // Surfaces, deepest → most raised
  "bg-0", // window gutter, viewport surround, drawers
  "bg-1", // panel background
  "bg-2", // raised: headers, toolbars, title bar
  "bg-3", // hover / active surfaces, inputs
  // Lines
  "border",
  "border-strong",
  // Text
  "text",
  "text-dim",
  "text-faint",
  "text-onaccent",
  // The single accent hue (ROADMAP §4) + its states
  "accent",
  "accent-hover",
  "accent-muted", // translucent accent tint for selections/hovers
  // Interaction
  "selection", // selected row / tab background
  "focus", // focus ring
  // Semantics (log severities, badges, pin/wire colors seed)
  "success",
  "warning",
  "error",
  "info",
  // Chrome details
  "shadow", // rgba shadow color for popups / floating panels
  "scrollbar",
  "scrollbar-hover",
] as const;

export type ThemeColorKey = (typeof THEME_COLOR_KEYS)[number];
export type ThemeColors = Record<ThemeColorKey, string>;

export interface StudioTheme {
  id: string;
  name: string;
  /** Sets `data-theme` on `<html>` for the rare dark/light-conditional CSS. */
  type: "dark" | "light";
  colors: ThemeColors;
}

const STORAGE_KEY = "infinity:theme";
export const DEFAULT_THEME_ID = "infinity-dark";

/** Built-in themes, dark default first. User themes come later (per-project
 *  settings storage arrives with P5.5). */
export const BUILTIN_THEMES: StudioTheme[] = [
  infinityDark,
  infinityLight,
  graphite,
  midnight,
].map((raw) => validateTheme(raw));

/** Validate unknown JSON as a StudioTheme. Missing/blank colors backfill
 *  from the default dark theme so the shell never renders `undefined`. */
export function validateTheme(value: unknown): StudioTheme {
  if (typeof value !== "object" || value === null) {
    throw new Error("theme must be a JSON object");
  }
  const v = value as Record<string, unknown>;
  if (typeof v.id !== "string" || v.id.length === 0) throw new Error("theme.id is required");
  if (typeof v.name !== "string" || v.name.length === 0) throw new Error("theme.name is required");
  if (v.type !== "dark" && v.type !== "light") {
    throw new Error(`theme.type must be "dark" or "light" (got ${JSON.stringify(v.type)})`);
  }
  const colorsIn = (typeof v.colors === "object" && v.colors !== null ? v.colors : {}) as Record<
    string,
    unknown
  >;
  const fallback = (infinityDark as { colors: Record<string, string> }).colors;
  const colors = {} as ThemeColors;
  for (const key of THEME_COLOR_KEYS) {
    const raw = colorsIn[key];
    colors[key] = typeof raw === "string" && raw.length > 0 ? raw : fallback[key];
  }
  return { id: v.id, name: v.name, type: v.type, colors };
}

/** Write a theme's tokens to `:root`. Idempotent; total by construction. */
export function applyTheme(theme: StudioTheme): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.setAttribute("data-theme", theme.type);
  for (const key of THEME_COLOR_KEYS) {
    root.style.setProperty(`--ink-${key}`, theme.colors[key]);
  }
}

export function getThemeById(id: string): StudioTheme | undefined {
  return BUILTIN_THEMES.find((t) => t.id === id);
}

export function currentThemeId(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? DEFAULT_THEME_ID;
  } catch {
    return DEFAULT_THEME_ID;
  }
}

/** Apply + persist a theme choice. Unknown ids fall back to the default. */
export function setTheme(id: string): StudioTheme {
  const theme = getThemeById(id) ?? getThemeById(DEFAULT_THEME_ID)!;
  applyTheme(theme);
  try {
    localStorage.setItem(STORAGE_KEY, theme.id);
  } catch {
    // Storage unavailable (headless tests) — the in-memory apply still holds.
  }
  return theme;
}

/** Startup hook: apply the persisted choice (or dark default). */
export function initTheme(): StudioTheme {
  return setTheme(currentThemeId());
}
