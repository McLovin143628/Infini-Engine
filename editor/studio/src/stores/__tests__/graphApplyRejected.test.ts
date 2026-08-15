// @vitest-environment jsdom
//
// **`GraphApplyResult.rejected`, at the blueprint canvas** (round-2 finding
// R2.F1).
//
// All three graph canvases are *fully controlled* and apply optimistically: an
// edit is mirrored into a local clone the instant the author makes it, and
// `*_apply` is the authority that runs afterwards. `apply_edit` answers `false`
// for a `connect` that would cycle, a node that is not there, or an unknown
// param — and when it does, the canvas is drawing a graph the backend does not
// have.
//
// `issues` cannot report that. It is computed over the backend's graph, where
// the refused wire does not exist to be invalid; the command still returns
// `Ok`. `rejected` is the only signal, it reached the wire in Wave F, and it
// was declared in no TypeScript type and read by nobody — so the desync it was
// added to end was fully intact.
//
// The sibling arms live in `materialStore.test.ts` and `pcgStore.test.ts`; this
// file exists because `blueprintStore` had no `apply` coverage at all.
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  graph: {
    registry: vi.fn(),
    list: vi.fn(),
    create: vi.fn(),
    get: vi.fn(),
    close: vi.fn(),
    apply: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
  },
}));

import { graph } from "../../lib/ipc";
import type { BpDoc } from "../../lib/blueprintTypes";
import { __resetBlueprintInitForTest, useBlueprintStore } from "../blueprintStore";

function makeDoc(): BpDoc {
  return {
    id: "bp:1",
    name: "Main",
    graph: {
      nodes: {
        "1": {
          id: 1,
          typeId: "event.begin_play",
          params: {},
          disabled: false,
          ui: { x: 0, y: 0, title: "", w: 0, h: 0 },
        },
      },
      links: [],
      nextId: 2,
    },
    viewport: null,
    modifiedMs: 0,
  };
}

const ADD = {
  kind: "add-node",
  id: 2,
  typeId: "debug.print",
  x: 10,
  y: 20,
  params: {},
} as const;

beforeEach(() => {
  __resetBlueprintInitForTest();
  useBlueprintStore.setState({
    registry: [],
    registryById: {},
    doc: null,
    issues: [],
    canUndo: false,
    canRedo: false,
    ready: false,
    running: false,
    runResult: null,
    generated: null,
  });
  vi.mocked(graph.registry).mockResolvedValue([]);
  vi.mocked(graph.list).mockResolvedValue([]);
  vi.mocked(graph.create).mockResolvedValue(makeDoc());
  vi.mocked(graph.get).mockResolvedValue(makeDoc());
  vi.mocked(graph.apply).mockResolvedValue({
    issues: [],
    canUndo: true,
    canRedo: false,
    rejected: 0,
  });
});

describe("blueprintStore.apply and a refused edit", () => {
  it("re-derives from the backend when an edit was refused", async () => {
    await useBlueprintStore.getState().init();
    vi.mocked(graph.apply).mockResolvedValue({
      issues: [],
      canUndo: true,
      canRedo: false,
      rejected: 1,
    });

    await useBlueprintStore.getState().apply([ADD], "Add node");

    expect(vi.mocked(graph.get)).toHaveBeenCalledWith("bp:1");
    expect(
      useBlueprintStore.getState().doc?.graph.nodes["2"],
      "the canvas kept drawing a node the backend refused",
    ).toBeUndefined();
  });

  it("does NOT re-derive when every edit landed", async () => {
    // The conditional half. A round trip per edit would retire the optimistic
    // path this canvas is built on, so the re-derive has to be the exception.
    await useBlueprintStore.getState().init();
    vi.mocked(graph.get).mockClear();

    await useBlueprintStore.getState().apply([ADD], "Add node");

    expect(vi.mocked(graph.get)).not.toHaveBeenCalled();
    expect(useBlueprintStore.getState().doc?.graph.nodes["2"]).toBeDefined();
  });

  it("still reports issues and undo availability on the refused path", async () => {
    // The re-derive must not swallow the rest of the answer.
    await useBlueprintStore.getState().init();
    vi.mocked(graph.apply).mockResolvedValue({
      issues: [{ kind: "noSink" }],
      canUndo: true,
      canRedo: true,
      rejected: 3,
    });

    await useBlueprintStore.getState().apply([ADD], "Add node");

    const s = useBlueprintStore.getState();
    expect(s.issues).toEqual([{ kind: "noSink" }]);
    expect(s.canUndo).toBe(true);
    expect(s.canRedo).toBe(true);
  });
});
