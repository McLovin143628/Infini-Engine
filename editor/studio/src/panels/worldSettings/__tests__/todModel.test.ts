import { describe, expect, it } from "vitest";

import {
  clampDayOfYear,
  clampLatitude,
  clampLongitude,
  compassPoint,
  DAYS_PER_YEAR,
  formatClock,
  fogVisibility,
  formatClockSeconds,
  SECONDS_PER_DAY,
  sunLabel,
  wrapSeconds,
} from "../todModel";

describe("World Settings time-of-day model (P17.1)", () => {
  it("wraps an arbitrary seconds value into one solar day", () => {
    expect(wrapSeconds(0)).toBe(0);
    expect(wrapSeconds(36_000)).toBe(36_000);
    expect(wrapSeconds(SECONDS_PER_DAY)).toBe(0);
    expect(wrapSeconds(SECONDS_PER_DAY + 1)).toBe(1);
    // Dragging a slider backwards past midnight must land on the previous
    // evening, not on a negative clock the backend would then clamp to zero.
    expect(wrapSeconds(-1)).toBe(SECONDS_PER_DAY - 1);
    expect(wrapSeconds(NaN)).toBe(0);
  });

  it("formats the clock readback", () => {
    expect(formatClock(0)).toBe("00:00");
    expect(formatClock(36_000)).toBe("10:00");
    expect(formatClock(45_296)).toBe("12:34");
    expect(formatClock(86_399)).toBe("23:59");
    expect(formatClock(-60)).toBe("23:59");
    expect(formatClockSeconds(45_296)).toBe("12:34:56");
  });

  it("clamps every authored range", () => {
    expect(clampDayOfYear(0)).toBe(1);
    expect(clampDayOfYear(1.4)).toBe(1);
    expect(clampDayOfYear(400)).toBe(DAYS_PER_YEAR);
    expect(clampDayOfYear(172.6)).toBe(173);
    expect(clampDayOfYear(NaN)).toBe(1);

    expect(clampLatitude(-120)).toBe(-90);
    expect(clampLatitude(120)).toBe(90);
    expect(clampLatitude(48.9)).toBe(48.9);
    expect(clampLatitude(NaN)).toBe(0);

    expect(clampLongitude(-400)).toBe(-180);
    expect(clampLongitude(400)).toBe(180);
    expect(clampLongitude(151.2)).toBe(151.2);
  });

  it("labels the sun's altitude by day / twilight / night", () => {
    expect(sunLabel(34.2)).toBe("34° above horizon");
    expect(sunLabel(-3)).toBe("-3° — twilight");
    expect(sunLabel(-40)).toBe("-40° — night");
    expect(sunLabel(NaN)).toBe("—");
  });

  it("maps an azimuth to a compass point", () => {
    expect(compassPoint(0)).toBe("N");
    expect(compassPoint(90)).toBe("E");
    expect(compassPoint(180)).toBe("S");
    expect(compassPoint(270)).toBe("W");
    expect(compassPoint(359)).toBe("N");
    expect(compassPoint(-90)).toBe("W");
    expect(compassPoint(NaN)).toBe("—");
  });
});

describe("height-fog visibility label (P17.2)", () => {
  it("reports clear air for zero or nonsense density", () => {
    expect(fogVisibility(0)).toBe("clear (no fog)");
    expect(fogVisibility(-1)).toBe("clear (no fog)");
    expect(fogVisibility(Number.NaN)).toBe("clear (no fog)");
  });

  it("converts m^-1 extinction to Koschmieder visibility", () => {
    // V = 3 / sigma.
    expect(fogVisibility(4e-4)).toBe("~7.5 km");
    expect(fogVisibility(1.5e-3)).toBe("~2.0 km");
    expect(fogVisibility(3e-4)).toBe("~10 km");
    expect(fogVisibility(0.01)).toBe("~300 m");
    expect(fogVisibility(1)).toBe("< 10 m");
  });

  it("reports a shorter visibility as density rises", () => {
    // Parse the label back out rather than re-deriving 3/sigma locally: a test
    // that recomputes the formula it is testing proves only that arithmetic works.
    const metres = (label: string): number => {
      if (label === "< 10 m") return 5;
      const m = /^~([\d.]+) (km|m)$/.exec(label);
      expect(m, `unparsable visibility label: ${label}`).not.toBeNull();
      const [, value, unit] = m!;
      return Number(value) * (unit === "km" ? 1000 : 1);
    };
    let previous = Number.POSITIVE_INFINITY;
    let saturated = false;
    for (let d = 1e-5; d < 1; d *= 2) {
      const label = fogVisibility(d);
      const current = metres(label);
      if (label === "< 10 m") {
        // The label saturates rather than reporting centimetres; once there it
        // stays there.
        saturated = true;
        expect(current).toBeLessThanOrEqual(previous);
      } else {
        expect(saturated, "visibility came back after saturating").toBe(false);
        expect(current).toBeLessThan(previous);
      }
      previous = current;
    }
    expect(saturated, "the sweep never reached the saturated label").toBe(true);
  });
});
