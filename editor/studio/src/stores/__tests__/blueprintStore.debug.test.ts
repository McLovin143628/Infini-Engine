// @vitest-environment jsdom
//
// Blueprint debugger slice (B-P4): breakpoint toggling + per-graph localStorage
// persistence, and the debug run folding captured wires into nodeId → port →
// value + hit highlights. IPC is mocked.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  graph: {
    debugRun: vi.fn(),
    registry: vi.fn(() => Promise.resolve([])),
    list: vi.fn(() => Promise.resolve([])),
    create: vi.fn(),
    close: vi.fn(() => Promise.resolve()),
  },
}));

import { graph } from "../../lib/ipc";
import { useBlueprintStore } from "../blueprintStore";
import type { BpDoc, DebugRunResult } from "../../lib/blueprintTypes";

const fakeDoc: BpDoc = {
  id: "bp:test",
  name: "T",
  graph: { nodes: {}, links: [], nextId: 1 },
  viewport: null,
  modifiedMs: 0,
};

function reset(): void {
  useBlueprintStore.setState({
    doc: fakeDoc,
    debugBreakpoints: new Set(),
    debugHits: new Set(),
    debugWireValues: {},
    running: false,
    runResult: null,
  });
}

beforeEach(() => {
  localStorage.clear();
  reset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("breakpoint slice", () => {
  it("toggleBreakpoint adds then removes a node id", () => {
    useBlueprintStore.getState().toggleBreakpoint(5);
    expect(useBlueprintStore.getState().debugBreakpoints.has(5)).toBe(true);
    useBlueprintStore.getState().toggleBreakpoint(5);
    expect(useBlueprintStore.getState().debugBreakpoints.has(5)).toBe(false);
  });

  it("persists breakpoints per graph in localStorage", () => {
    useBlueprintStore.getState().toggleBreakpoint(7);
    useBlueprintStore.getState().toggleBreakpoint(9);
    const raw = localStorage.getItem("bp:debug:bp:test");
    expect(raw).not.toBeNull();
    expect(JSON.parse(raw as string).sort()).toEqual([7, 9]);

    // Removing one updates the persisted set.
    useBlueprintStore.getState().toggleBreakpoint(7);
    expect(JSON.parse(localStorage.getItem("bp:debug:bp:test") as string)).toEqual([9]);
  });
});

describe("debugRun", () => {
  it("sends the current breakpoints and folds wires + hits", async () => {
    useBlueprintStore.getState().toggleBreakpoint(5);
    const result: DebugRunResult = {
      hits: [5],
      wires: [
        { node: 5, port: "out", value: "3" },
        { node: 2, port: "value", value: "true" },
      ],
      logs: ["hello"],
      vars: { x: 1 },
      handlers: ["begin_play"],
      error: null,
    };
    vi.mocked(graph.debugRun).mockResolvedValue(result);

    await useBlueprintStore.getState().debugRun();

    expect(vi.mocked(graph.debugRun)).toHaveBeenCalledWith("bp:test", [5], true);
    const s = useBlueprintStore.getState();
    expect(s.debugHits.has(5)).toBe(true);
    expect(s.debugWireValues[5].out).toBe("3");
    expect(s.debugWireValues[2].value).toBe("true");
    // The run output drawer is reused for logs / vars.
    expect(s.runResult?.logs).toEqual(["hello"]);
    expect(s.running).toBe(false);
  });

  it("clearDebugValues drops hits + wire values", () => {
    useBlueprintStore.setState({
      debugHits: new Set([1]),
      debugWireValues: { 1: { out: "9" } },
    });
    useBlueprintStore.getState().clearDebugValues();
    expect(useBlueprintStore.getState().debugHits.size).toBe(0);
    expect(useBlueprintStore.getState().debugWireValues).toEqual({});
  });
});
