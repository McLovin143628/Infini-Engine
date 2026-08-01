import { describe, expect, it } from "vitest";

import {
  clampDayOfYear,
  clampLatitude,
  clampLongitude,
  compassPoint,
  DAYS_PER_YEAR,
  formatClock,
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
