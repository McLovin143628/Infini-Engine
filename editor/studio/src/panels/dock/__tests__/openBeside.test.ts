// @vitest-environment jsdom
//
// `openBesidePanel` (Wave E): the API behind "open this object's mesh, blueprint
// and code as tabs of ONE group". The docking substrate could always express a
// tab group — nothing could ask for it.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import "../../registerPanels";
import { defaultDockLayout, useDockLayout } from "../dockLayoutStore";

const container = { w: 1600, h: 900 };
const dock = () => useDockLayout.getState();

beforeEach(() => {
  mockIPC(() => undefined);
  useDockLayout.setState({ layout: defaultDockLayout(), hydrated: true, container });
});
afterEach(() => clearMocks());

/** The group a docked panel belongs to, or null. */
function groupOf(id: string): { side: string; tabs: string[] } | null {
  const { layout } = dock();
  const loc = layout.panels[id]?.location;
  if (!loc || loc.kind !== "docked") return null;
  const group = layout.docks[loc.side].groups.find((g) => g.id === loc.groupId);
  return group ? { side: loc.side, tabs: group.tabs } : null;
}

describe("openBesidePanel", () => {
  it("docks the new panel as a TAB of the anchor's group, and activates it", () => {
    const anchor = dock().openPanel("model", "mesh-a");
    expect(groupOf(anchor)).not.toBeNull();

    const beside = dock().openBesidePanel(anchor, "skeleton", "rig-a");
    const group = groupOf(beside);
    expect(group).not.toBeNull();
    expect(group!.tabs).toEqual(groupOf(anchor)!.tabs);
    expect(group!.tabs).toContain(anchor);
    expect(group!.tabs).toContain(beside);
    // Appended after the anchor, and active.
    expect(group!.tabs.indexOf(beside)).toBeGreaterThan(group!.tabs.indexOf(anchor));
    const loc = dock().layout.panels[beside].location;
    if (loc.kind !== "docked") throw new Error("expected docked");
    expect(dock().layout.docks[loc.side].groups.find((g) => g.id === loc.groupId)!.activeTab).toBe(
      beside,
    );
  });

  it("is idempotent: opening the same panel beside the same anchor twice keeps one tab", () => {
    const anchor = dock().openPanel("model", "mesh-a");
    const a = dock().openBesidePanel(anchor, "skeleton", "rig-a");
    const b = dock().openBesidePanel(anchor, "skeleton", "rig-a");
    expect(a).toBe(b);
    expect(groupOf(a)!.tabs.filter((t) => t === a)).toHaveLength(1);
  });

  it("falls back to a plain open when the anchor is floating — never throws", () => {
    const anchor = dock().openPanel("model", "mesh-a");
    dock().applyDrop(anchor, { kind: "float", rect: { x: 10, y: 10, w: 300, h: 200 } });
    const beside = dock().openBesidePanel(anchor, "skeleton", "rig-a");
    expect(dock().layout.panels[beside]).toBeDefined();
  });

  it("falls back when the anchor does not exist at all", () => {
    const beside = dock().openBesidePanel("no-such-panel", "model", "mesh-a");
    expect(dock().layout.panels[beside]).toBeDefined();
  });

  it("an anchor equal to the panel being opened is a no-op open", () => {
    const anchor = dock().openPanel("model", "mesh-a");
    expect(dock().openBesidePanel(anchor, "model", "mesh-a")).toBe(anchor);
  });
});

describe("openPanel on a SINGLETON with new params", () => {
  /**
   * The blueprint panel is a singleton, so its instance id is its type and a
   * second `openPanel("blueprint", …)` used to be a pure focus — the canvas kept
   * showing the FIRST actor for ever. Wave E's routing depends on this.
   */
  it("adopts the new params", () => {
    dock().openPanel("blueprint", "actor:a-1");
    expect(dock().layout.panels.blueprint.params).toBe("actor:a-1");
    dock().openPanel("blueprint", "actor:a-2");
    expect(dock().layout.panels.blueprint.params).toBe("actor:a-2");
  });

  it("a params-less toggle does NOT clear what the panel was showing", () => {
    dock().openPanel("blueprint", "actor:a-1");
    dock().openPanel("blueprint");
    expect(dock().layout.panels.blueprint.params).toBe("actor:a-1");
  });
});
