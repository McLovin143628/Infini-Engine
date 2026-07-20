// @vitest-environment jsdom
//
// Store transitions + structural invariants for the dock layout. Persist
// calls ride the layouts IPC domain — mockIPC swallows them.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
// Side-effect: registers the built-in panel types (repairLayout drops
// unregistered ones).
import "../../registerPanels";
import {
  defaultDockLayout,
  repairLayout,
  useDockLayout,
} from "../dockLayoutStore";
import type { DockLayout } from "../dockTypes";

const container = { w: 1600, h: 900 };

beforeEach(() => {
  mockIPC(() => undefined);
  useDockLayout.setState({
    layout: defaultDockLayout(),
    hydrated: true,
    container,
  });
});

afterEach(() => {
  clearMocks();
});

describe("defaultDockLayout", () => {
  it("is UE-parity: outliner over details on the right, rest hidden", () => {
    const l = defaultDockLayout();
    expect(l.docks.right.groups).toHaveLength(2);
    expect(l.docks.right.groups[0].tabs).toEqual(["outliner"]);
    expect(l.docks.right.groups[1].tabs).toEqual(["details"]);
    expect(l.panels.outputLog.location.kind).toBe("hidden");
    expect(l.panels.outputLog.lastDock).toBe("bottom");
    expect(repairLayout(l)).toEqual(l); // default is structurally sound
  });
});

describe("location transitions", () => {
  it("hidePanel remembers the dock side; openPanel restores it", () => {
    const store = useDockLayout.getState();
    store.hidePanel("outliner");
    let l = useDockLayout.getState().layout;
    expect(l.panels.outliner.location.kind).toBe("hidden");
    expect(l.panels.outliner.lastDock).toBe("right");
    expect(l.docks.right.groups).toHaveLength(1); // empty group pruned

    useDockLayout.getState().openPanel("outliner");
    l = useDockLayout.getState().layout;
    expect(l.panels.outliner.location.kind).toBe("docked");
    if (l.panels.outliner.location.kind === "docked") {
      expect(l.panels.outliner.location.side).toBe("right");
    }
  });

  it("showPanel restores a hidden bottom panel to the bottom", () => {
    useDockLayout.getState().showPanel("outputLog");
    const l = useDockLayout.getState().layout;
    expect(l.panels.outputLog.location.kind).toBe("docked");
    expect(l.docks.bottom.groups[0].tabs).toContain("outputLog");
  });

  it("applyDrop float → window → back to dock keeps identity", () => {
    const store = useDockLayout.getState();
    store.applyDrop("details", { kind: "float", rect: { x: 40, y: 40, w: 300, h: 300 } });
    expect(useDockLayout.getState().layout.panels.details.location.kind).toBe("floating");

    useDockLayout.getState().applyDrop("details", { kind: "window" });
    const p = useDockLayout.getState().layout.panels.details;
    expect(p.location.kind).toBe("window");
    expect(p.windowRect).toBeNull(); // fresh detach spawns at the cursor

    const g = useDockLayout.getState().layout.docks.right.groups[0];
    useDockLayout
      .getState()
      .applyDrop("details", { kind: "dock", side: "right", groupId: g.id, index: 1 });
    const l = useDockLayout.getState().layout;
    expect(l.docks.right.groups[0].tabs).toEqual(["outliner", "details"]);
    expect(l.docks.right.groups[0].activeTab).toBe("details");
  });

  it("tab insert into an existing group prunes the source group", () => {
    const l0 = useDockLayout.getState().layout;
    const target = l0.docks.right.groups[0]; // outliner's group
    useDockLayout
      .getState()
      .applyDrop("details", { kind: "dock", side: "right", groupId: target.id, index: 0 });
    const l = useDockLayout.getState().layout;
    expect(l.docks.right.groups).toHaveLength(1);
    expect(l.docks.right.groups[0].tabs).toEqual(["details", "outliner"]);
  });

  it("floating panels stack by z; bringToFront raises", () => {
    const store = useDockLayout.getState();
    store.applyDrop("outliner", { kind: "float", rect: { x: 10, y: 10, w: 300, h: 300 } });
    useDockLayout
      .getState()
      .applyDrop("details", { kind: "float", rect: { x: 60, y: 60, w: 300, h: 300 } });
    let l = useDockLayout.getState().layout;
    expect(l.panels.details.floatZ).toBeGreaterThan(l.panels.outliner.floatZ);
    useDockLayout.getState().bringToFront("outliner");
    l = useDockLayout.getState().layout;
    expect(l.panels.outliner.floatZ).toBeGreaterThan(l.panels.details.floatZ);
  });
});

describe("repairLayout", () => {
  it("drops unregistered panel types and their tabs", () => {
    const l = defaultDockLayout();
    const broken: DockLayout = {
      ...l,
      panels: {
        ...l.panels,
        ghost: { ...l.panels.outliner, type: "ghost" },
      },
      docks: {
        ...l.docks,
        left: {
          size: 300,
          groups: [{ id: "gx", tabs: ["ghost"], activeTab: "ghost", weight: 1 }],
        },
      },
    };
    const repaired = repairLayout(broken);
    expect(repaired.panels.ghost).toBeUndefined();
    expect(repaired.docks.left.groups).toHaveLength(0);
  });

  it("re-floats a docked panel whose group is gone", () => {
    const l = defaultDockLayout();
    const broken: DockLayout = {
      ...l,
      docks: { ...l.docks, right: { size: 320, groups: [] } },
    };
    const repaired = repairLayout(broken);
    expect(repaired.panels.outliner.location.kind).toBe("floating");
    expect(repaired.panels.details.location.kind).toBe("floating");
  });
});
