import { describe, expect, it } from "vitest";

import { cssHexToLinear, linearToCssHex } from "../biomeColor";

describe("linear ↔ CSS colour (P19.2)", () => {
  it("encodes with the sRGB transfer function, not a bare ×255", () => {
    // Mid-grey linear 0.5 encodes to ~0.7354 sRGB → 0xbc. A naive ×255 would
    // give 0x80, which is what makes a swatch look wrong beside the viewport.
    expect(linearToCssHex([0.5, 0.5, 0.5, 1])).toBe("#bcbcbc");
    expect(linearToCssHex([0, 0, 0, 1])).toBe("#000000");
    expect(linearToCssHex([1, 1, 1, 1])).toBe("#ffffff");
  });

  it("round-trips a picked colour to display precision", () => {
    for (const hex of ["#000000", "#ffffff", "#3b7a19", "#dbe3f0", "#010203"]) {
      expect(linearToCssHex(cssHexToLinear(hex))).toBe(hex);
    }
  });

  it("clamps out-of-gamut and non-finite channels instead of emitting NaN", () => {
    expect(linearToCssHex([2, -1, Number.NaN, 1])).toBe("#ff0000");
    expect(linearToCssHex([])).toBe("#000000");
  });

  it("reads a malformed hex as opaque black rather than throwing", () => {
    // This sits on an input event — a half-typed value must not break a render.
    expect(cssHexToLinear("nonsense")).toEqual([0, 0, 0, 1]);
    expect(cssHexToLinear("#abc")).toEqual([0, 0, 0, 1]);
  });

  it("carries alpha through and clamps it", () => {
    expect(cssHexToLinear("#000000", 0.25)[3]).toBe(0.25);
    expect(cssHexToLinear("#000000", 5)[3]).toBe(1);
    expect(cssHexToLinear("#000000", Number.NaN)[3]).toBe(0);
  });

  it("accepts a hex with or without the leading #", () => {
    expect(cssHexToLinear("ffffff")).toEqual(cssHexToLinear("#ffffff"));
  });
});
