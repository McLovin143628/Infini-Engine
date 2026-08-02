/** The PCG wire/category colour map (P19.4 added the grammar wires).
 *
 *  This is a tiny map, but it is the one place a *new backend wire type* can be
 *  forgotten: the registry is served from Rust, so a `grammar.expand` node
 *  appears in the palette the moment it is registered — and its pins render a
 *  silent, indistinguishable grey until this file learns the name. A test that
 *  pins every wire the Rust registry declares is what turns that into a
 *  failure. */
import { describe, expect, it } from "vitest";

import { pcgCategoryColor, pcgPinColor } from "../pcgPinTheme";

/** Every `PortType::Named` key `inf_pcg::graph` declares, in registry order. */
const WIRES = ["density", "scatter", "layer", "span", "rules"] as const;

/** Every `NodeDef::category` the PCG registry uses, in registry order. */
const CATEGORIES = [
  "sources",
  "filters",
  "masks",
  "combine",
  "scatter",
  "grammar",
  "layer",
  "output",
] as const;

const FALLBACK = "#9aa0a6";
const CATEGORY_FALLBACK = "#4a5058";

describe("pcgPinColor", () => {
  it("gives every declared wire its own colour", () => {
    const seen = new Map<string, string>();
    for (const name of WIRES) {
      const color = pcgPinColor({ kind: "named", name });
      expect(color, `${name} fell through to the unknown-wire grey`).not.toBe(FALLBACK);
      expect(color).toMatch(/^#[0-9a-f]{6}$/i);
      const clash = [...seen.entries()].find(([, c]) => c === color);
      expect(clash, `${name} shares a colour with ${clash?.[0]}`).toBeUndefined();
      seen.set(name, color);
    }
  });

  it("falls back for an unknown wire rather than throwing", () => {
    expect(pcgPinColor({ kind: "named", name: "not-a-wire" })).toBe(FALLBACK);
    expect(pcgPinColor({ kind: "wildcard" })).toBe(FALLBACK);
    expect(pcgPinColor({ kind: "exec" })).toBe(FALLBACK);
  });
});

describe("pcgCategoryColor", () => {
  it("gives every declared category its own header tint", () => {
    for (const category of CATEGORIES) {
      expect(
        pcgCategoryColor(category),
        `${category} fell through to the unknown-category tint`,
      ).not.toBe(CATEGORY_FALLBACK);
    }
    // Grammar is visually distinct from scatter — they are sibling generators
    // and a canvas mixing both should not read as one block.
    expect(pcgCategoryColor("grammar")).not.toBe(pcgCategoryColor("scatter"));
  });

  it("falls back for an unknown category", () => {
    expect(pcgCategoryColor("nope")).toBe(CATEGORY_FALLBACK);
  });
});
