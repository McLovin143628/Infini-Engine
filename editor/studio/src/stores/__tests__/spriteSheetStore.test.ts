import { describe, expect, it } from "vitest";

import type { SpriteSheetDto } from "../../bindings/SpriteSheetDto";
import { MAX_GRID_CELLS, resolveSlices } from "../spriteSheetStore";

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

/**
 * **The bound the port did not have** (round-2 finding R2.F6).
 *
 * C4-15 bounded the Rust resolver by `MAX_GRID_CELLS` because `columns`/`rows`
 * come out of an editable TOML sidecar. This mirror had no bound at all, and it
 * is the copy that runs on every keystroke: `<input type=number>` accepts
 * `1e999`, so `Number(...)` was `Infinity`, `Math.max(1, Infinity)` was
 * `Infinity`, and the loop pushed slice objects until the tab died. Even
 * `100000` was 10^10 cells on the UI thread, each a heap object and a
 * `strokeRect`.
 */
describe("resolveSlices is bounded (R2.F6)", () => {
  const sheet = (grid: {
    columns: number;
    rows: number;
  }): SpriteSheetDto => ({
    texture_id: "t",
    grid: { ...grid, margin_x: 0, margin_y: 0, padding_x: 0, padding_y: 0 },
    manual: [],
    tex_width: 4096,
    tex_height: 4096,
  });

  it("refuses to grow past MAX_GRID_CELLS", () => {
    // The hand-editable value the Rust bound exists for. Without the budget
    // this call does not return.
    const s = resolveSlices(sheet({ columns: 100_000, rows: 100_000 }));
    expect(s).toHaveLength(MAX_GRID_CELLS);
  });

  it("survives a non-finite grid", () => {
    // THE defect: one keystroke of `1e999` into Columns.
    expect(resolveSlices(sheet({ columns: Infinity, rows: 4 }))).toHaveLength(4);
    expect(resolveSlices(sheet({ columns: 4, rows: Infinity }))).toHaveLength(4);
    expect(resolveSlices(sheet({ columns: NaN, rows: NaN }))).toHaveLength(1);
  });

  it("truncates exactly where the Rust does", () => {
    // `GridSlicing::count` is `min(columns * rows, MAX_GRID_CELLS)` — a cell
    // BUDGET, not a per-axis clamp — so both sides drop the same cells.
    const s = resolveSlices(sheet({ columns: 512, rows: 512 }));
    expect(512 * 512).toBeGreaterThan(MAX_GRID_CELLS);
    expect(s).toHaveLength(MAX_GRID_CELLS);
    // Row-major: what survives is the first N in reading order.
    expect(s[0].name).toBe("0");
    expect(s[MAX_GRID_CELLS - 1].name).toBe(String(MAX_GRID_CELLS - 1));
  });

  it("still resolves an ordinary grid whole", () => {
    // The other direction — a bound that truncated real atlases would be a
    // worse defect than the one it fixes.
    expect(resolveSlices(sheet({ columns: 16, rows: 16 }))).toHaveLength(256);
  });
});
