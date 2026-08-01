import { describe, expect, it } from "vitest";

import {
  clampDayOfYear,
  clampLatitude,
  clampLongitude,
  cloudCoverLabel,
  cloudLayerLabel,
  compassPoint,
  DAYS_PER_YEAR,
  formatClock,
  fogVisibility,
  formatClockSeconds,
  SECONDS_PER_DAY,
  precipLabel,
  sunLabel,
  weatherPreset,
  WEATHER_PRESETS,
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

describe("cloud sky-cover label (P17.3)", () => {
  it("names the sky the way an author would", () => {
    // The slider is a bias on a procedural field, not an area fraction — the
    // label is the only place the panel says what the number will actually look
    // like, so these are the words the rows are worth having.
    expect(cloudCoverLabel(0)).toBe("clear");
    expect(cloudCoverLabel(0.2)).toBe("clear");
    expect(cloudCoverLabel(0.25)).toBe("few");
    // The component default: broken cumulus with real gaps, which is what the
    // `clouds_scattered` golden pictures.
    expect(cloudCoverLabel(0.35)).toBe("scattered");
    expect(cloudCoverLabel(0.5)).toBe("broken");
    expect(cloudCoverLabel(1)).toBe("overcast");
  });

  it("is monotone across the slider and survives nonsense", () => {
    const order = ["clear", "few", "scattered", "broken", "overcast"];
    let seen = 0;
    for (let i = 0; i <= 100; i++) {
      const rank = order.indexOf(cloudCoverLabel(i / 100));
      expect(rank).toBeGreaterThanOrEqual(0);
      expect(rank).toBeGreaterThanOrEqual(seen);
      seen = rank;
    }
    // ...and it reaches both ends rather than parking in the middle.
    expect(seen).toBe(order.length - 1);
    expect(cloudCoverLabel(NaN)).toBe("clear");
    expect(cloudCoverLabel(-5)).toBe("clear");
  });
});

describe("cloud layer label (P17.3)", () => {
  it("reports thickness in the unit that reads", () => {
    // The component default layer, 1500 m to 4000 m.
    expect(cloudLayerLabel(1500, 4000)).toBe("2.5 km thick");
    expect(cloudLayerLabel(900, 2200)).toBe("1.3 km thick");
    expect(cloudLayerLabel(1000, 1400)).toBe("400 m thick");
  });

  it("names the silent no-op instead of leaving it to be discovered", () => {
    // A top at or below the base draws nothing at all. Without this the author
    // sees an empty sky and no explanation.
    expect(cloudLayerLabel(2000, 2000)).toBe("empty (top is not above bottom)");
    expect(cloudLayerLabel(4000, 1500)).toBe("empty (top is not above bottom)");
    expect(cloudLayerLabel(NaN, 4000)).toBe("—");
    expect(cloudLayerLabel(1500, Infinity)).toBe("—");
  });
});

// ── weather (P17.4) ─────────────────────────────────────────────────────────

describe("World Settings weather model (P17.4)", () => {
  it("lists every preset exactly once, in menu order", () => {
    expect(WEATHER_PRESETS).toEqual(["clear", "overcast", "storm", "fog", "snow"]);
    expect(new Set(WEATHER_PRESETS).size).toBe(WEATHER_PRESETS.length);
  });

  /**
   * The preset table is duplicated across the ring boundary (the button must
   * write its values optimistically, before the IPC round-trip). These literals
   * are the SAME numbers `WeatherPreset::params()` returns in
   * `crates/inf-ecs/src/components.rs`; if that table is re-tuned, this fails and
   * says so, which is the whole reason the duplication is affordable.
   */
  it("mirrors the Rust preset table field for field", () => {
    expect(weatherPreset("clear")).toEqual({
      coverage: 0.08,
      cloud_type: 0.75,
      wind_x: 4.0,
      wind_z: 1.5,
      fog_density: 0.0,
      precipitation: 0.0,
      snowiness: 0.0,
    });
    expect(weatherPreset("overcast")).toEqual({
      coverage: 0.85,
      cloud_type: 0.2,
      wind_x: 9.0,
      wind_z: 3.5,
      fog_density: 7.5e-5,
      precipitation: 0.0,
      snowiness: 0.0,
    });
    expect(weatherPreset("storm")).toEqual({
      coverage: 1.0,
      cloud_type: 0.35,
      wind_x: 22.0,
      wind_z: 9.0,
      fog_density: 6.0e-4,
      precipitation: 1.0,
      snowiness: 0.0,
    });
    expect(weatherPreset("fog")).toEqual({
      coverage: 0.5,
      cloud_type: 0.1,
      wind_x: 1.5,
      wind_z: 0.5,
      fog_density: 6.0e-3,
      precipitation: 0.0,
      snowiness: 0.0,
    });
    expect(weatherPreset("snow")).toEqual({
      coverage: 0.9,
      cloud_type: 0.3,
      wind_x: 5.0,
      wind_z: 2.0,
      fog_density: 1.2e-3,
      precipitation: 0.7,
      snowiness: 1.0,
    });
  });

  it("gives every preset a distinct sky", () => {
    const seen = WEATHER_PRESETS.map((p) => JSON.stringify(weatherPreset(p)));
    expect(new Set(seen).size).toBe(WEATHER_PRESETS.length);
  });

  it("only Snow accumulates — the P22 hook's frontend tell", () => {
    for (const p of WEATHER_PRESETS) {
      const w = weatherPreset(p);
      if (p === "snow") expect(w.snowiness).toBe(1);
      else expect(w.snowiness).toBe(0);
    }
  });

  it("labels precipitation by strength and phase", () => {
    expect(precipLabel(0, 0)).toBe("dry");
    expect(precipLabel(-1, 1)).toBe("dry");
    expect(precipLabel(NaN, 0)).toBe("dry");
    expect(precipLabel(0.1, 0)).toBe("light rain");
    expect(precipLabel(0.4, 0)).toBe("steady rain");
    expect(precipLabel(0.7, 0)).toBe("heavy rain");
    expect(precipLabel(1, 0)).toBe("torrential rain");
    // Snowiness is a blend, so the words pass through sleet on the way.
    expect(precipLabel(0.7, 0.5)).toBe("heavy sleet");
    expect(precipLabel(0.7, 1)).toBe("heavy snow");
    // Hostile input clamps rather than producing "undefined".
    expect(precipLabel(9, 9)).toBe("torrential snow");
    expect(precipLabel(0.5, NaN)).toBe("steady rain");
  });
});
