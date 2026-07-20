import { expect, test } from "vitest";

import { toPhysicalRect } from "../viewportRect";

test("dpr 1.0 is the identity", () => {
  const rect = { x: 12.5, y: 40, width: 800, height: 600 };
  expect(toPhysicalRect(rect, 1)).toEqual(rect);
});

test("150% DPI (the classic Windows laptop default) scales every component", () => {
  expect(toPhysicalRect({ x: 10, y: 20, width: 640, height: 480 }, 1.5)).toEqual({
    x: 15,
    y: 30,
    width: 960,
    height: 720,
  });
});

test("fractional CSS coordinates are passed through unrounded — rounding to device pixels is the backend's job (single-rounding rule)", () => {
  const physical = toPhysicalRect({ x: 10.4, y: 0, width: 100.3, height: 50 }, 2);
  expect(physical.x).toBeCloseTo(20.8);
  expect(physical.width).toBeCloseTo(200.6);
});
