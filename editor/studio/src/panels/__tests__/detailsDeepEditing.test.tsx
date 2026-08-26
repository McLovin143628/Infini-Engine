// @vitest-environment jsdom
//
// Details-panel deep editing (E-P1): renderControl produces the right widget for
// the new value kinds (entity_ref / list / struct) and the callbacks send the
// WHOLE updated value back through `set`; plus the pure list-mutation helpers.
import { isValidElement } from "react";
import { describe, expect, it, vi } from "vitest";

import { renderControl } from "../DetailsPanel";
import {
  EnumField,
  ListField,
  NumberField,
  RangedNumberField,
  StructField,
  listMove,
  listRemoveAt,
  listSetAt,
} from "../../components/propertyRows";
import { EntityRefField } from "../../components/EntityRefField";
import type { PropValueDto } from "../../bindings/PropValueDto";

// renderControl reaches into ipc for list-default; stub it so imports resolve.
vi.mock("../../lib/ipc", () => ({ scene: { listDefault: vi.fn(() => Promise.resolve(null)) } }));

const num = (n: number): PropValueDto => ({ kind: "number", value: n });
const vec = (x: number, y: number, z: number): PropValueDto => ({ kind: "vec3", value: [x, y, z] });

describe("renderControl new kinds", () => {
  it("renders an EntityRefField and clears/sets the guid", () => {
    const set = vi.fn();
    const el = renderControl({ kind: "entity_ref", value: "abc" }, set, "Joint3D", "other");
    expect(isValidElement(el)).toBe(true);
    expect((el as React.ReactElement).type).toBe(EntityRefField);
    // The picker's onChange forwards the whole entity_ref value.
    (el as React.ReactElement<{ onChange: (g: string | null) => void }>).props.onChange(null);
    expect(set).toHaveBeenCalledWith({ kind: "entity_ref", value: null });
  });

  it("renders a ListField whose onChange replaces the whole list", () => {
    const set = vi.fn();
    const list: PropValueDto = { kind: "list", value: [vec(1, 2, 3), vec(4, 5, 6)] };
    const el = renderControl(list, set, "Spline", "points");
    expect((el as React.ReactElement).type).toBe(ListField);
    const next = [vec(9, 9, 9)];
    (el as React.ReactElement<{ onChange: (v: PropValueDto[]) => void }>).props.onChange(next);
    expect(set).toHaveBeenCalledWith({ kind: "list", value: next });
  });

  // Wave VIS1b: a numeric row draws a slider only when Ring 0 handed it a range.
  // The range is a HINT -- the panel must not invent one, and must not lose one.
  it("renders a slider only for a field the engine gave a range", () => {
    const set = vi.fn();
    const plain = renderControl(num(0.5), set, "Material", "metallic");
    expect((plain as React.ReactElement).type).toBe(NumberField);

    const ranged = renderControl(num(0.5), set, "Material", "metallic", [0, 1, 0.01]);
    expect((ranged as React.ReactElement).type).toBe(RangedNumberField);
    expect((ranged as React.ReactElement<{ range: number[] }>).props.range).toEqual([0, 1, 0.01]);
    (ranged as React.ReactElement<{ onChange: (v: number) => void }>).props.onChange(0.75);
    expect(set).toHaveBeenCalledWith({ kind: "number", value: 0.75 });

    // A null range is the absent case, not a range of nulls.
    const nulled = renderControl(num(0.5), set, "Material", "metallic", null);
    expect((nulled as React.ReactElement).type).toBe(NumberField);
  });

  it("renders a StructField whose onChange replaces the whole struct", () => {
    const set = vi.fn();
    const struct: PropValueDto = {
      kind: "struct",
      fields: [{ name: "metallic", label: "Metallic", value: num(0.5), same: true, range: null }],
    };
    const el = renderControl(struct, set, "Material", "props");
    expect((el as React.ReactElement).type).toBe(StructField);
    const next = [{ name: "metallic", label: "Metallic", value: num(0.9), same: true, range: null }];
    (el as React.ReactElement<{ onChange: (f: unknown) => void }>).props.onChange(next);
    expect(set).toHaveBeenCalledWith({ kind: "struct", fields: next });
  });

  it("still renders the scalar kinds (enum)", () => {
    const el = renderControl({ kind: "enum", value: "Point", options: ["Point", "Spot"] }, vi.fn(), "Light", "kind");
    expect((el as React.ReactElement).type).toBe(EnumField);
  });
});

describe("list mutation helpers", () => {
  const a = num(1);
  const b = num(2);
  const c = num(3);

  it("setAt replaces one element immutably", () => {
    const src = [a, b, c];
    const out = listSetAt(src, 1, num(9));
    expect(out).toEqual([a, num(9), c]);
    expect(src).toEqual([a, b, c]); // original untouched
  });

  it("removeAt drops the element", () => {
    expect(listRemoveAt([a, b, c], 0)).toEqual([b, c]);
    expect(listRemoveAt([a, b, c], 2)).toEqual([a, b]);
  });

  it("move swaps within bounds and no-ops at the edges", () => {
    expect(listMove([a, b, c], 0, 1)).toEqual([b, a, c]);
    expect(listMove([a, b, c], 2, 1)).toEqual([a, b, c]); // past end → unchanged
    expect(listMove([a, b, c], 0, -1)).toEqual([a, b, c]); // before start → unchanged
  });
});
