// @vitest-environment jsdom
//
// Material editor store (P7.2 + P13.2). Mirrors the PCG/blueprint stores: init
// seeds the registry + a document and compiles; apply is optimistic +
// authoritative + recompiles (except position-only moves); the compile result
// carries the P13.2 complexity report. IPC is mocked; the reused
// `applyEditsLocal` reducer runs for real.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  material: {
    registry: vi.fn(),
    list: vi.fn(),
    create: vi.fn(),
    get: vi.fn(),
    close: vi.fn(),
    apply: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    compile: vi.fn(),
    bake: vi.fn(),
  },
}));

import { material } from "../../lib/ipc";
import type { BpDoc, NodeDef } from "../../lib/blueprintTypes";
import type { ComplexityReport, MaterialCompileResult } from "../../lib/materialTypes";
import { useMaterialStore } from "../materialStore";

const slabDef: NodeDef = {
  typeId: "slab.surface",
  display: "Surface Slab",
  category: "slab",
  description: "",
  inputs: [{ name: "base_color", label: "Base Color", ty: { kind: "named", name: "vec3f" }, required: false, paramPin: false }],
  outputs: [{ name: "out", label: "Slab", ty: { kind: "named", name: "slab" }, required: false, paramPin: false }],
  params: [],
  flags: 0,
};

function makeDoc(): BpDoc {
  return {
    id: "mat:1",
    name: "Main",
    graph: {
      nodes: {
        "1": { id: 1, typeId: "output.surface", params: {}, disabled: false, ui: { x: 0, y: 0, title: "", w: 0, h: 0 } },
      },
      links: [],
      nextId: 2,
    },
    viewport: null,
    modifiedMs: 0,
  };
}

const complexity: ComplexityReport = {
  nodes: 1,
  slabs: 0,
  textures: 0,
  estAluOps: 0,
  maxNodes: 64,
  maxSlabs: 8,
  maxTextures: 16,
  maxAluOps: 512,
  nodesOver: false,
  slabsOver: false,
  texturesOver: false,
  aluOver: false,
  overBudget: false,
};

const compileResult: MaterialCompileResult = {
  wgsl: "// wgsl",
  ok: true,
  issues: [],
  textureCount: 0,
  image: null,
  complexity,
};

beforeEach(() => {
  useMaterialStore.setState({
    registry: [],
    registryById: {},
    doc: null,
    issues: [],
    canUndo: false,
    canRedo: false,
    ready: false,
    compiling: false,
    compileResult: null,
    showWgsl: false,
  });
  vi.mocked(material.registry).mockResolvedValue([slabDef]);
  vi.mocked(material.list).mockResolvedValue([]);
  vi.mocked(material.create).mockResolvedValue(makeDoc());
  vi.mocked(material.close).mockResolvedValue(undefined);
  vi.mocked(material.apply).mockResolvedValue({
    issues: [],
    canUndo: true,
    canRedo: false,
    rejected: 0,
  });
  vi.mocked(material.compile).mockResolvedValue(compileResult);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("init", () => {
  it("creates a document when none exist, indexes the registry, and compiles", async () => {
    await useMaterialStore.getState().init();
    expect(vi.mocked(material.create)).toHaveBeenCalledWith("Main");
    const s = useMaterialStore.getState();
    expect(s.ready).toBe(true);
    expect(s.doc?.id).toBe("mat:1");
    expect(s.registryById["slab.surface"]).toEqual(slabDef);
    expect(vi.mocked(material.compile)).toHaveBeenCalledWith("mat:1");
  });

  it("adopts the first existing document instead of creating one", async () => {
    vi.mocked(material.list).mockResolvedValue([makeDoc()]);
    await useMaterialStore.getState().init();
    expect(vi.mocked(material.create)).not.toHaveBeenCalled();
    expect(useMaterialStore.getState().doc?.id).toBe("mat:1");
  });
});

describe("compile", () => {
  it("stores the compile result including the P13.2 complexity report", async () => {
    await useMaterialStore.getState().init();
    const s = useMaterialStore.getState();
    expect(s.compileResult?.complexity.maxSlabs).toBe(8);
    expect(s.compileResult?.complexity.overBudget).toBe(false);
  });

  it("surfaces over-budget flags from the backend", async () => {
    vi.mocked(material.compile).mockResolvedValue({
      ...compileResult,
      complexity: { ...complexity, slabs: 12, slabsOver: true, overBudget: true },
    });
    await useMaterialStore.getState().init();
    const c = useMaterialStore.getState().compileResult?.complexity;
    expect(c?.slabsOver).toBe(true);
    expect(c?.overBudget).toBe(true);
  });
});

describe("apply", () => {
  it("optimistically adds a slab node, calls material_apply, and recompiles", async () => {
    await useMaterialStore.getState().init();
    vi.mocked(material.compile).mockClear();
    await useMaterialStore.getState().apply(
      [{ kind: "add-node", id: 2, typeId: "slab.surface", x: 10, y: 20, params: {} }],
      "Add node",
    );
    expect(useMaterialStore.getState().doc?.graph.nodes["2"]?.typeId).toBe("slab.surface");
    expect(vi.mocked(material.apply)).toHaveBeenCalledWith(
      "mat:1",
      [{ kind: "add-node", id: 2, typeId: "slab.surface", x: 10, y: 20, params: {} }],
      "Add node",
    );
    expect(vi.mocked(material.compile)).toHaveBeenCalledWith("mat:1");
  });

  it("skips the recompile for position-only moves", async () => {
    await useMaterialStore.getState().init();
    vi.mocked(material.compile).mockClear();
    await useMaterialStore.getState().apply([{ kind: "move-node", id: 1, x: 99, y: 99 }], "Move");
    expect(vi.mocked(material.apply)).toHaveBeenCalled();
    expect(vi.mocked(material.compile)).not.toHaveBeenCalled();
  });

  // **R2.F1.** The optimistic clone is a bet that the backend agrees. When it
  // does not — a connect that would cycle, an unknown param — `rejected` is the
  // ONLY signal: `issues` is computed over the backend's graph, where the
  // refused wire does not exist to be invalid.
  it("re-derives from the backend when an edit was refused", async () => {
    await useMaterialStore.getState().init();
    vi.mocked(material.apply).mockResolvedValue({
      issues: [],
      canUndo: true,
      canRedo: false,
      rejected: 1,
    });
    // What the backend actually holds: the node was never added.
    vi.mocked(material.get).mockResolvedValue(makeDoc());

    await useMaterialStore.getState().apply(
      [{ kind: "add-node", id: 2, typeId: "slab.surface", x: 10, y: 20, params: {} }],
      "Add node",
    );

    expect(vi.mocked(material.get)).toHaveBeenCalledWith("mat:1");
    expect(
      useMaterialStore.getState().doc?.graph.nodes["2"],
      "the canvas kept drawing a node the backend refused",
    ).toBeUndefined();
  });

  it("does NOT re-derive when every edit landed", async () => {
    // The other direction: a round trip per edit would undo the whole point of
    // the optimistic path, so the re-derive has to be conditional.
    await useMaterialStore.getState().init();
    vi.mocked(material.get).mockClear();
    await useMaterialStore.getState().apply(
      [{ kind: "add-node", id: 2, typeId: "slab.surface", x: 10, y: 20, params: {} }],
      "Add node",
    );
    expect(vi.mocked(material.get)).not.toHaveBeenCalled();
    expect(useMaterialStore.getState().doc?.graph.nodes["2"]).toBeDefined();
  });
});

describe("close", () => {
  it("frees the backend document and resets to an un-inited state", async () => {
    await useMaterialStore.getState().init();
    expect(useMaterialStore.getState().doc?.id).toBe("mat:1");
    await useMaterialStore.getState().close();
    expect(vi.mocked(material.close)).toHaveBeenCalledWith("mat:1");
    const s = useMaterialStore.getState();
    expect(s.doc).toBeNull();
    expect(s.ready).toBe(false);
    expect(s.compileResult).toBeNull();
  });

  it("is a no-op with no open document (never calls the backend)", async () => {
    await useMaterialStore.getState().close();
    expect(vi.mocked(material.close)).not.toHaveBeenCalled();
  });
});
