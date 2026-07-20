import { describe, expect, it } from "vitest";
import { fuzzyFilter, fuzzyMatch } from "../fuzzy";

describe("fuzzyMatch", () => {
  it("matches subsequences case-insensitively", () => {
    expect(fuzzyMatch("svl", "Save Current Level")).not.toBeNull();
    expect(fuzzyMatch("xyz", "Save Current Level")).toBeNull();
  });

  it("empty needle matches everything with zero score", () => {
    expect(fuzzyMatch("", "anything")).toEqual({ score: 0, indices: [] });
  });

  it("prefers word-boundary and consecutive matches", () => {
    const boundary = fuzzyMatch("save", "File: Save All")!;
    const scattered = fuzzyMatch("save", "s-a-v-e-x")!;
    expect(boundary.score).toBeGreaterThan(scattered.score);
  });

  it("returns matched indices in order", () => {
    const m = fuzzyMatch("cl", "Current Level")!;
    expect(m.indices).toHaveLength(2);
    expect(m.indices[0]).toBeLessThan(m.indices[1]);
  });
});

describe("fuzzyFilter", () => {
  const items = ["Save All", "Save Current Level", "Load Layout", "Open Level"];

  it("returns all items for a blank needle", () => {
    expect(fuzzyFilter("", items, (s) => s)).toEqual(items);
  });

  it("filters and ranks", () => {
    const out = fuzzyFilter("level", items, (s) => s);
    expect(out).toContain("Open Level");
    expect(out).toContain("Save Current Level");
    expect(out).not.toContain("Save All");
  });
});
