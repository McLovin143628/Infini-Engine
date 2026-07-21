import { describe, expect, it } from "vitest";

import { computeFloodFill, FLOOD_FILL_LIMIT } from "../tilemapStore";

const key = (x: number, y: number) => `${x},${y}`;

function mapOf(cells: Array<{ x: number; y: number; tile: number }>): {
  map: Map<string, number>;
  cells: Array<{ x: number; y: number }>;
} {
  const map = new Map<string, number>();
  for (const c of cells) map.set(key(c.x, c.y), c.tile);
  return { map, cells: cells.map((c) => ({ x: c.x, y: c.y })) };
}

describe("computeFloodFill", () => {
  it("fills a contiguous region of the seed's tile", () => {
    // A 2×2 block of tile 1; fill from inside with 2.
    const { map, cells } = mapOf([
      { x: 0, y: 0, tile: 1 },
      { x: 1, y: 0, tile: 1 },
      { x: 0, y: 1, tile: 1 },
      { x: 1, y: 1, tile: 1 },
    ]);
    const out = computeFloodFill(map, cells, 0, 0, 2);
    expect(out).toHaveLength(4);
    expect(out.every((c) => c.tile === 2)).toBe(true);
  });

  it("does not cross a different-tile boundary", () => {
    // Row of 1s split by a 9 in the middle: fill from the left only reaches left.
    const { map, cells } = mapOf([
      { x: 0, y: 0, tile: 1 },
      { x: 1, y: 0, tile: 9 },
      { x: 2, y: 0, tile: 1 },
    ]);
    const out = computeFloodFill(map, cells, 0, 0, 5);
    expect(out).toEqual([{ x: 0, y: 0, tile: 5 }]);
  });

  it("returns nothing when the seed already holds the fill tile", () => {
    const { map, cells } = mapOf([{ x: 0, y: 0, tile: 3 }]);
    expect(computeFloodFill(map, cells, 0, 0, 3)).toHaveLength(0);
  });

  it("bounds a fill of empty space to the painted box + margin (never runs away)", () => {
    // One painted tile far away; fill empty space from the origin. The region is
    // clamped, so the result is finite and within the cap.
    const { map, cells } = mapOf([{ x: 3, y: 3, tile: 1 }]);
    const out = computeFloodFill(map, cells, 0, 0, 2);
    expect(out.length).toBeGreaterThan(0);
    expect(out.length).toBeLessThanOrEqual(FLOOD_FILL_LIMIT);
    // Every filled cell replaced an empty (target 0) cell — the tile-1 island is
    // untouched.
    expect(out.every((c) => !(c.x === 3 && c.y === 3))).toBe(true);
  });
});
