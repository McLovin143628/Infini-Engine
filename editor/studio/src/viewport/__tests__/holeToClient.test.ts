// @vitest-environment jsdom
//
// **The coordinate contract, both directions** (Wave E).
//
// A drop into the viewport is sent as `(clientX - holeRect.left) * dpr`
// (`ContentDrawer`), so a right-click coming BACK out must be
// `holeRect.left + x / dpr`. Getting this wrong does not throw: the menu simply
// opens in the wrong place — usually near the window origin on a HiDPI display,
// which reads as "the menu is broken" rather than "the transform is inverted".
import { describe, expect, it } from "vitest";

import { holeToClient } from "../ViewportContextMenu";

/** A stand-in for the hole element's `getBoundingClientRect()`. */
const rect = (left: number, top: number): DOMRect =>
  ({ left, top, x: left, y: top, width: 800, height: 600, right: left + 800, bottom: top + 600 }) as DOMRect;

describe("holeToClient", () => {
  it("inverts the drop transform exactly, at dpr 1", () => {
    expect(holeToClient(120, 80, 1, rect(240, 64))).toEqual({ x: 360, y: 144 });
  });

  it("divides by the device pixel ratio (the HiDPI case)", () => {
    // The native side reports PHYSICAL pixels; the DOM positions in CSS pixels.
    expect(holeToClient(200, 100, 2, rect(100, 50))).toEqual({ x: 200, y: 100 });
  });

  it("round-trips the ContentDrawer's forward transform", () => {
    const holeRect = rect(313, 97);
    const dpr = 1.5;
    for (const [clientX, clientY] of [
      [400, 200],
      [313, 97],
      [1000, 640],
    ]) {
      // Forward: what the drawer sends over `viewport_drop`.
      const px = (clientX - holeRect.left) * dpr;
      const py = (clientY - holeRect.top) * dpr;
      const back = holeToClient(px, py, dpr, holeRect);
      expect(back.x).toBeCloseTo(clientX, 6);
      expect(back.y).toBeCloseTo(clientY, 6);
    }
  });

  it("degrades to the raw point when the hole element is missing", () => {
    // A viewport panel that is closed or not yet mounted: better a menu at the
    // wrong place than no menu and no explanation.
    expect(holeToClient(11, 22, 2, null)).toEqual({ x: 11, y: 22 });
  });

  it("treats a non-finite or zero dpr as 1 rather than dividing by it", () => {
    expect(holeToClient(10, 10, 0, rect(0, 0))).toEqual({ x: 10, y: 10 });
    expect(holeToClient(10, 10, Number.NaN, rect(0, 0))).toEqual({ x: 10, y: 10 });
  });
});
