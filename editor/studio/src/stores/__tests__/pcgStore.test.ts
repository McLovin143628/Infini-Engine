// @vitest-environment jsdom
//
// PCG editor store (P10.5b): mirrors the material store. Verifies init seeds the
// registry + a document and compiles; apply is optimistic + authoritative +
// recompiles (except for position-only moves); evaluate scatters into the scene.
// IPC is mocked; the reused `applyEditsLocal` reducer runs for real.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  pcg: {
    registry: vi.fn(),
    list: vi.fn(),
    create: vi.fn(),
    get: vi.fn(),
    close: vi.fn(),
    apply: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    compile: vi.fn(),
    save: vi.fn(),
    evaluate: vi.fn(),
  },
}));

import { pcg } from "../../lib/ipc";
import type { BpDoc, NodeDef } from "../../lib/blueprintTypes";
import { usePcgStore } from "../pcgStore";

const scatterDef: NodeDef = {
  typeId: "scatter.scatter",
  display: "Scatter",
  category: "scatter",
  description: "",
  inputs: [{ name: "density", label: "Density", ty: { kind: "named", name: "density" }, required: false, paramPin: false }],
  outputs: [{ name: "out", label: "Out", ty: { kind: "named", name: "scatter" }, required: false, paramPin: false }],
  params: [],
  flags: 0,
};

function makeDoc(): BpDoc {
  return {
    id: "pcg:1",
    name: "Main",
    graph: {
      nodes: {
        "1": { id: 1, typeId: "scatter.scatter", params: {}, disabled: false, ui: { x: 0, y: 0, title: "", w: 0, h: 0 } },
      },
      links: [],
      nextId: 2,
    },
    viewport: null,
    modifiedMs: 0,
  };
}

const compileResult = { ok: true, issues: [], ruleCount: 1, summary: "1 rule(s)" };

beforeEach(() => {
  usePcgStore.setState({
    registry: [],
    registryById: {},
    doc: null,
    issues: [],
    canUndo: false,
    canRedo: false,
    ready: false,
    compiling: false,
    compileResult: null,
    evaluating: false,
    lastEval: null,
  });
  vi.mocked(pcg.registry).mockResolvedValue([scatterDef]);
  vi.mocked(pcg.list).mockResolvedValue([]);
  vi.mocked(pcg.create).mockResolvedValue(makeDoc());
  vi.mocked(pcg.close).mockResolvedValue(undefined);
  vi.mocked(pcg.apply).mockResolvedValue({ issues: [], canUndo: true, canRedo: false });
  vi.mocked(pcg.compile).mockResolvedValue(compileResult);
  vi.mocked(pcg.evaluate).mockResolvedValue({ entity: "abc", placed: 42, issues: [], ok: true });
  vi.mocked(pcg.save).mockResolvedValue("PCG_Graph.inf_pcg");
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("init", () => {
  it("creates a document when none exist and indexes the registry", async () => {
    await usePcgStore.getState().init();
    expect(vi.mocked(pcg.create)).toHaveBeenCalledWith("Main");
    const s = usePcgStore.getState();
    expect(s.ready).toBe(true);
    expect(s.doc?.id).toBe("pcg:1");
    expect(s.registryById["scatter.scatter"]).toEqual(scatterDef);
    expect(vi.mocked(pcg.compile)).toHaveBeenCalledWith("pcg:1");
  });

  it("adopts the first existing document instead of creating one", async () => {
    vi.mocked(pcg.list).mockResolvedValue([makeDoc()]);
    await usePcgStore.getState().init();
    expect(vi.mocked(pcg.create)).not.toHaveBeenCalled();
    expect(usePcgStore.getState().doc?.id).toBe("pcg:1");
  });
});

describe("apply", () => {
  it("optimistically adds a node, calls pcg_apply, and recompiles", async () => {
    await usePcgStore.getState().init();
    vi.mocked(pcg.compile).mockClear();
    await usePcgStore.getState().apply(
      [{ kind: "add-node", id: 2, typeId: "const.density", x: 10, y: 20, params: {} }],
      "Add node",
    );
    // Optimistic local reducer applied the node immediately.
    expect(usePcgStore.getState().doc?.graph.nodes["2"]?.typeId).toBe("const.density");
    expect(vi.mocked(pcg.apply)).toHaveBeenCalledWith(
      "pcg:1",
      [{ kind: "add-node", id: 2, typeId: "const.density", x: 10, y: 20, params: {} }],
      "Add node",
    );
    // A structural edit triggers a recompile.
    expect(vi.mocked(pcg.compile)).toHaveBeenCalledWith("pcg:1");
  });

  it("skips the recompile for position-only moves", async () => {
    await usePcgStore.getState().init();
    vi.mocked(pcg.compile).mockClear();
    await usePcgStore.getState().apply([{ kind: "move-node", id: 1, x: 99, y: 99 }], "Move");
    expect(vi.mocked(pcg.apply)).toHaveBeenCalled();
    expect(vi.mocked(pcg.compile)).not.toHaveBeenCalled();
  });
});

describe("evaluate", () => {
  it("scatters into the scene and stores the result", async () => {
    await usePcgStore.getState().init();
    await usePcgStore.getState().evaluate();
    expect(vi.mocked(pcg.evaluate)).toHaveBeenCalledWith("pcg:1", null);
    expect(usePcgStore.getState().lastEval?.placed).toBe(42);
    expect(usePcgStore.getState().evaluating).toBe(false);
  });
});

describe("save", () => {
  it("persists the graph and returns the file name", async () => {
    await usePcgStore.getState().init();
    const file = await usePcgStore.getState().save("PCG Graph");
    expect(vi.mocked(pcg.save)).toHaveBeenCalledWith("pcg:1", "PCG Graph");
    expect(file).toBe("PCG_Graph.inf_pcg");
  });
});

describe("close", () => {
  it("frees the backend document and resets to an un-inited state", async () => {
    await usePcgStore.getState().init();
    expect(usePcgStore.getState().doc?.id).toBe("pcg:1");
    await usePcgStore.getState().close();
    expect(vi.mocked(pcg.close)).toHaveBeenCalledWith("pcg:1");
    const s = usePcgStore.getState();
    expect(s.doc).toBeNull();
    expect(s.ready).toBe(false);
    expect(s.compileResult).toBeNull();
  });

  it("is a no-op with no open document (never calls the backend)", async () => {
    await usePcgStore.getState().close();
    expect(vi.mocked(pcg.close)).not.toHaveBeenCalled();
  });
});
