import { describe, expect, it } from "vitest";

import type { SpriteSheetDto } from "../../bindings/SpriteSheetDto";
import { resolveSlices } from "../spriteSheetStore";

/**
 * The JS resolver must match the Rust `SpriteSheetSlices::resolve` exactly (it
 * draws the overlay the backend applies). These mirror the Rust unit tests.
 */
describe("resolveSlices", () => {
  const sheet = (partial: Partial<SpriteSheetDto>): SpriteSheetDto => ({
    texture_id: "t",
    grid: null,
    manual: [],
    tex_width: 64,
    tex_height: 32,
    ...partial,
  });

  it("resolves a plain grid with no margin/padding", () => {
    const s = resolveSlices(
      sheet({ grid: { columns: 4, rows: 2, margin_x: 0, margin_y: 0, padding_x: 0, padding_y: 0 } }),
    );
    expect(s).toHaveLength(8);
    expect(s[0].name).toBe("0");
    expect(s[0].px).toEqual([0, 0, 16, 16]);
    expect(s[0].uvMax).toEqual([0.25, 0.5]);
    // Tile 5 = row 1, col 1.
    expect(s[5].uvMin).toEqual([0.25, 0.5]);
    expect(s[5].uvMax).toEqual([0.5, 1.0]);
  });

  it("honors margin and padding", () => {
    // 2×2 over 40×40, 4px margin, 2px padding → 17px cells.
    const s = resolveSlices(
      sheet({
        tex_width: 40,
        tex_height: 40,
        grid: { columns: 2, rows: 2, margin_x: 4, margin_y: 4, padding_x: 2, padding_y: 2 },
      }),
    );
    expect(s[0].px).toEqual([4, 4, 17, 17]);
    expect(s[1].px).toEqual([23, 4, 17, 17]);
  });

  it("keeps UV precision for non-square cells", () => {
    const s = resolveSlices(
      sheet({
        tex_width: 100,
        tex_height: 10,
        grid: { columns: 3, rows: 1, margin_x: 0, margin_y: 0, padding_x: 0, padding_y: 0 },
      }),
    );
    expect(s[1].uvMin[0]).toBeCloseTo(1 / 3, 9);
    expect(s[2].uvMax[0]).toBeCloseTo(1.0, 9);
  });

  it("appends manual rects after grid cells", () => {
    const s = resolveSlices(
      sheet({
        tex_width: 100,
        tex_height: 100,
        grid: { columns: 2, rows: 1, margin_x: 0, margin_y: 0, padding_x: 0, padding_y: 0 },
        manual: [{ name: "hero", x: 10, y: 20, width: 30, height: 40 }],
      }),
    );
    expect(s.map((x) => x.name)).toEqual(["0", "1", "hero"]);
    expect(s[2].uvMin).toEqual([0.1, 0.2]);
    expect(s[2].uvMax).toEqual([0.4, 0.6]);
  });
});
