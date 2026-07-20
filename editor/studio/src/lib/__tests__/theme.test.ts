// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";
import {
  BUILTIN_THEMES,
  DEFAULT_THEME_ID,
  THEME_COLOR_KEYS,
  applyTheme,
  currentThemeId,
  getThemeById,
  initTheme,
  setTheme,
  validateTheme,
} from "../theme";

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("style");
  document.documentElement.removeAttribute("data-theme");
});

describe("built-in themes", () => {
  it("include the dark default first", () => {
    expect(BUILTIN_THEMES[0].id).toBe(DEFAULT_THEME_ID);
    expect(BUILTIN_THEMES[0].type).toBe("dark");
  });

  it("are total (every token key present, non-empty)", () => {
    for (const theme of BUILTIN_THEMES) {
      for (const key of THEME_COLOR_KEYS) {
        expect(theme.colors[key], `${theme.id} missing ${key}`).toBeTruthy();
      }
    }
  });

  it("have unique ids", () => {
    const ids = BUILTIN_THEMES.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("validateTheme", () => {
  it("rejects non-objects and missing required fields", () => {
    expect(() => validateTheme(null)).toThrow();
    expect(() => validateTheme("nope")).toThrow();
    expect(() => validateTheme({ id: "x", name: "X", type: "dusk" })).toThrow(/dark.*light/);
    expect(() => validateTheme({ name: "X", type: "dark" })).toThrow(/id/);
  });

  it("backfills missing colors from the default dark theme", () => {
    const theme = validateTheme({ id: "sparse", name: "Sparse", type: "dark", colors: { accent: "#ff0000" } });
    expect(theme.colors.accent).toBe("#ff0000");
    expect(theme.colors["bg-0"]).toBe(getThemeById(DEFAULT_THEME_ID)!.colors["bg-0"]);
  });
});

describe("applyTheme / setTheme / initTheme", () => {
  it("writes every token as --ink-* on :root and sets data-theme", () => {
    const theme = getThemeById(DEFAULT_THEME_ID)!;
    applyTheme(theme);
    const root = document.documentElement;
    expect(root.getAttribute("data-theme")).toBe("dark");
    for (const key of THEME_COLOR_KEYS) {
      expect(root.style.getPropertyValue(`--ink-${key}`)).toBe(theme.colors[key]);
    }
  });

  it("setTheme persists and falls back to default on unknown ids", () => {
    const applied = setTheme("infinity-light");
    expect(applied.id).toBe("infinity-light");
    expect(currentThemeId()).toBe("infinity-light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");

    const fallback = setTheme("does-not-exist");
    expect(fallback.id).toBe(DEFAULT_THEME_ID);
  });

  it("initTheme applies the persisted choice", () => {
    setTheme("midnight");
    document.documentElement.removeAttribute("data-theme");
    const applied = initTheme();
    expect(applied.id).toBe("midnight");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });
});
