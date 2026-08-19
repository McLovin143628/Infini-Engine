// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EntityEditorsDto } from "../../bindings/EntityEditorsDto";
import type { ContextMenuItem } from "../../components/ContextMenu";
import { __resetCommandsForTest, registerCommands } from "../commands";
import { buildObjectMenuItems, type ObjectMenuActions } from "../objectMenu";

const row = (over: Partial<EntityEditorsDto>): EntityEditorsDto => ({
  entity: "e1",
  name: "Thing",
  kind: "Static Mesh",
  mesh: null,
  skeletal_mesh: null,
  skeleton: null,
  material: null,
  actor_class: null,
  primitive: null,
  no_editor_reason: null,
  ...over,
});

const actions = (): ObjectMenuActions => ({
  open: vi.fn(),
  openRoute: vi.fn(),
  rename: vi.fn(),
  toggleVisible: vi.fn(),
});

const labels = (items: ReturnType<typeof buildObjectMenuItems>) =>
  items.filter((i): i is ContextMenuItem => i !== "separator").map((i) => i.label);

describe("buildObjectMenuItems", () => {
  beforeEach(() => {
    __resetCommandsForTest();
    registerCommands([
      { id: "edit.duplicate", title: "Duplicate", category: "Edit", run: () => {} },
      { id: "edit.delete", title: "Delete", category: "Edit", run: () => {} },
    ]);
  });

  it("offers the object's routes, plus open/rename on a single selection", () => {
    const items = buildObjectMenuItems(["e1"], [row({ mesh: "m" })], actions(), true);
    expect(labels(items)).toEqual([
      "Open in Editor",
      "Edit Mesh",
      "Rename",
      "Hide",
      "Duplicate",
      "Delete",
    ]);
  });

  it("shows a DISABLED row carrying the reason when nothing can be edited", () => {
    const items = buildObjectMenuItems(
      ["e1"],
      [row({ primitive: "Cube", no_editor_reason: "no mesh asset, drop one on it" })],
      actions(),
      true,
    );
    const refusal = items.find(
      (i): i is ContextMenuItem => i !== "separator" && i.disabled === true,
    );
    expect(refusal).toBeDefined();
    expect(refusal!.hint).toContain("drop one on it");
    // …and no edit route is offered at all.
    expect(labels(items)).not.toContain("Edit Mesh");
    expect(labels(items)).not.toContain("Open in Editor");
  });

  it("a multi-selection shows the intersection and pluralized destructive labels", () => {
    const items = buildObjectMenuItems(
      ["a", "b"],
      [row({ entity: "a", mesh: "m1", actor_class: "c" }), row({ entity: "b", mesh: "m2" })],
      actions(),
    );
    const l = labels(items);
    expect(l).toContain("Edit Mesh");
    expect(l).not.toContain("Open Blueprint"); // only one of the two has a class
    expect(l).not.toContain("Rename"); // single-selection only
    expect(l).toContain("Duplicate 2 Objects");
    expect(l).toContain("Delete 2 Objects");
  });

  it("omits rows whose command is not registered — never a dead item", () => {
    __resetCommandsForTest();
    const items = buildObjectMenuItems(["e1"], [row({ mesh: "m" })], actions(), null);
    const l = labels(items);
    expect(l).not.toContain("Duplicate");
    expect(l).not.toContain("Delete");
    // `view.focusSelection` does not exist until the viewport registers it.
    expect(l).not.toContain("Focus");
  });

  it("never emits a leading, trailing or doubled separator", () => {
    __resetCommandsForTest();
    const items = buildObjectMenuItems(["e1"], [row({ mesh: "m" })], actions(), null);
    expect(items[0]).not.toBe("separator");
    expect(items[items.length - 1]).not.toBe("separator");
    for (let i = 1; i < items.length; i++) {
      expect(items[i] === "separator" && items[i - 1] === "separator").toBe(false);
    }
  });

  it("wires Rename and the route action to the host panel's callbacks", () => {
    const a = actions();
    const items = buildObjectMenuItems(["e1"], [row({ mesh: "m" })], a, true);
    const byLabel = (label: string) =>
      items.find((i): i is ContextMenuItem => i !== "separator" && i.label === label)!;
    byLabel("Rename").onSelect();
    expect(a.rename).toHaveBeenCalledWith("e1");
    byLabel("Edit Mesh").onSelect();
    expect(a.openRoute).toHaveBeenCalledWith("mesh", ["e1"]);
    byLabel("Open in Editor").onSelect();
    expect(a.open).toHaveBeenCalledWith("e1");
    byLabel("Hide").onSelect();
    expect(a.toggleVisible).toHaveBeenCalledWith("e1");
  });
});
