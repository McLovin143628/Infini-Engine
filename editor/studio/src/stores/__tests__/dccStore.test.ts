// @vitest-environment jsdom
//
// Model Editor store (P23.4). Two things are worth testing here and the rest is
// plumbing: the store **replaces** its document from every reply (because the
// backend owns the selection and a structural op may have dropped it), and the
// preview is a **serialized queue of one** (because an orbit fires faster than a
// render + PNG + base64 round trip and a pile-up makes the panel lag the mouse).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  dcc: {
    open: vi.fn(),
    close: vi.fn(),
    list: vi.fn(),
    apply: vi.fn(),
    select: vi.fn(),
    pick: vi.fn(),
    orbit: vi.fn(),
    frame: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    preview: vi.fn(),
    save: vi.fn(),
    mergeAsset: vi.fn(),
  },
  DCC_PREVIEW_SIZE: 256,
}));

import { dcc } from "../../lib/ipc";
import type { DccDocDto } from "../../bindings/DccDocDto";
import { __resetDccPreviewQueueForTest, useDccStore } from "../dccStore";

const mocked = vi.mocked(dcc);

function docOf(over: Partial<DccDocDto> = {}): DccDocDto {
  return {
    id: "dcc:abc",
    assetId: "abc",
    name: "Prop",
    mode: "face",
    verts: 8,
    edges: 12,
    faces: 6,
    selected: 0,
    canUndo: false,
    canRedo: false,
    dirty: false,
    generation: 1,
    knifePoints: 0,
    import: {
      sourceVertices: 24,
      weldedPositions: 8,
      fanSplits: 0,
      degenerateTrianglesSkipped: 0,
      sharpEdges: 12,
      boundaryEdges: 0,
      nonFiniteValues: 0,
    },
    ...over,
  };
}

/** A deferred promise, so a test can hold a call open and watch what queues. */
function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

beforeEach(() => {
  vi.clearAllMocks();
  __resetDccPreviewQueueForTest();
  useDccStore.setState({
    doc: null,
    assetId: null,
    image: null,
    previewError: null,
    refusal: null,
    lastSave: null,
    busy: false,
    status: null,
  });
  mocked.preview.mockResolvedValue({ image: "data:image/png;base64,AA", error: null, size: 256 });
});

afterEach(() => {
  __resetDccPreviewQueueForTest();
});

describe("open/close", () => {
  it("opens an asset and renders the first frame", async () => {
    mocked.open.mockResolvedValue(docOf());
    await useDccStore.getState().open("abc");
    expect(mocked.open).toHaveBeenCalledWith("abc");
    expect(useDccStore.getState().doc?.id).toBe("dcc:abc");
    expect(useDccStore.getState().image).toContain("data:image/png");
  });

  it("re-opening the same asset does not re-fetch", async () => {
    mocked.open.mockResolvedValue(docOf());
    await useDccStore.getState().open("abc");
    await useDccStore.getState().open("abc");
    expect(mocked.open).toHaveBeenCalledTimes(1);
  });

  it("surfaces an open failure as a status rather than throwing", async () => {
    mocked.open.mockRejectedValue("not a mesh asset");
    await useDccStore.getState().open("abc");
    expect(useDccStore.getState().doc).toBeNull();
    expect(useDccStore.getState().status).toContain("not a mesh asset");
  });

  it("closes the backend document and clears the panel", async () => {
    mocked.open.mockResolvedValue(docOf());
    mocked.close.mockResolvedValue(undefined);
    await useDccStore.getState().open("abc");
    await useDccStore.getState().close();
    expect(mocked.close).toHaveBeenCalledWith("dcc:abc");
    expect(useDccStore.getState().doc).toBeNull();
    expect(useDccStore.getState().image).toBeNull();
  });
});

describe("the document is replaced, never patched", () => {
  beforeEach(async () => {
    mocked.open.mockResolvedValue(docOf());
    await useDccStore.getState().open("abc");
  });

  it("adopts the whole document a tool returns", async () => {
    // The load-bearing case: an extrude renumbers, so the backend rebuilt the
    // selection from the op's outcome. A store that merged fields would keep the
    // OLD `selected` and show a count for geometry that no longer exists.
    mocked.apply.mockResolvedValue({
      ok: true,
      refusal: null,
      doc: docOf({ faces: 10, verts: 12, selected: 5, generation: 2, dirty: true }),
    });
    await useDccStore.getState().apply({ tool: "extrude", distance: 0.5 });
    const doc = useDccStore.getState().doc!;
    expect(doc.faces).toBe(10);
    expect(doc.selected).toBe(5);
    expect(doc.generation).toBe(2);
    expect(doc.dirty).toBe(true);
  });

  it("shows a refusal as a value and keeps the document", async () => {
    mocked.apply.mockResolvedValue({
      ok: false,
      refusal: "the region has no border, so there is nothing to inset",
      doc: docOf(),
    });
    await useDccStore.getState().apply({ tool: "inset", amount: 0.1, individual: false });
    expect(useDccStore.getState().refusal).toContain("no border");
    expect(useDccStore.getState().doc).not.toBeNull();
  });

  it("clears the refusal on the next successful tool", async () => {
    mocked.apply.mockResolvedValueOnce({ ok: false, refusal: "nope", doc: docOf() });
    await useDccStore.getState().apply({ tool: "subdivide" });
    expect(useDccStore.getState().refusal).toBe("nope");
    mocked.apply.mockResolvedValueOnce({ ok: true, refusal: null, doc: docOf() });
    await useDccStore.getState().apply({ tool: "subdivide" });
    expect(useDccStore.getState().refusal).toBeNull();
  });

  it("sends the pick in preview pixels at the shared size", async () => {
    mocked.pick.mockResolvedValue(docOf({ selected: 1 }));
    await useDccStore.getState().pick(12, 34, true);
    // The size is the ONE constant `dcc_preview` was given: a pick computed
    // against a different projection lands somewhere else.
    expect(mocked.pick).toHaveBeenCalledWith("dcc:abc", 12, 34, 256, true);
    expect(useDccStore.getState().doc?.selected).toBe(1);
  });

  it("routes a mode switch through the select command", async () => {
    mocked.select.mockResolvedValue(docOf({ mode: "edge" }));
    await useDccStore.getState().setMode("edge");
    expect(mocked.select).toHaveBeenCalledWith("dcc:abc", { action: "mode", mode: "edge" });
    expect(useDccStore.getState().doc?.mode).toBe("edge");
  });
});

describe("the preview queue", () => {
  beforeEach(async () => {
    mocked.open.mockResolvedValue(docOf());
    await useDccStore.getState().open("abc");
    vi.clearAllMocks();
  });

  it("collapses a burst of orbits into one in-flight render and one follow-up", async () => {
    // The claim: an orbit that fires ten times while a render is running must
    // produce ONE more render, not ten. Without the gate the panel trails the
    // mouse by however many frames are queued, and the last one to arrive is not
    // necessarily the last one asked for.
    const first = deferred<{ image: string | null; error: string | null; size: number }>();
    mocked.preview.mockReturnValueOnce(first.promise);
    mocked.preview.mockResolvedValue({ image: "data:image/png;base64,BB", error: null, size: 256 });
    mocked.orbit.mockResolvedValue(undefined);

    const bursts = Array.from({ length: 10 }, () => useDccStore.getState().orbit(5, 0, 0));
    // All ten orbit calls reach the backend — the CAMERA is cheap and must not
    // be dropped, or the model would stop following the mouse.
    await Promise.resolve();
    expect(mocked.orbit).toHaveBeenCalledTimes(10);
    expect(mocked.preview).toHaveBeenCalledTimes(1);

    first.resolve({ image: "data:image/png;base64,AA", error: null, size: 256 });
    await Promise.all(bursts);
    expect(mocked.preview).toHaveBeenCalledTimes(2);
    expect(useDccStore.getState().image).toContain("BB");
  });

  it("reports a preview failure without clearing the document", async () => {
    mocked.preview.mockResolvedValue({
      image: null,
      error: "no GPU adapter on this machine",
      size: 256,
    });
    useDccStore.getState().refresh();
    await vi.waitFor(() =>
      expect(useDccStore.getState().previewError).toContain("no GPU adapter"),
    );
    expect(useDccStore.getState().doc).not.toBeNull();
  });
});

describe("save", () => {
  beforeEach(async () => {
    mocked.open.mockResolvedValue(docOf({ dirty: true }));
    await useDccStore.getState().open("abc");
  });

  it("keeps the verdict and re-reads the document's dirty flag", async () => {
    mocked.save.mockResolvedValue({
      ok: true,
      vmesh: "built",
      advisories: [],
      export: {
        submeshes: 1,
        vertices: 24,
        triangles: 12,
        fanFallbacks: 0,
        fallbackTangents: 0,
        coincidentVertices: 0,
        reusedDiagonals: 0,
        nonFiniteWritten: 0,
        nonUnitNormalsWritten: 0,
      },
    });
    // `dirty` is derived from the generation stamp on the BACKEND, so the only
    // honest way to learn it after a save is to ask.
    mocked.list.mockResolvedValue([docOf({ dirty: false })]);
    await useDccStore.getState().save();
    expect(useDccStore.getState().lastSave?.vmesh).toBe("built");
    expect(useDccStore.getState().doc?.dirty).toBe(false);
    expect(useDccStore.getState().status).toContain("12 triangles");
  });

  it("keeps the writer's advisories so the panel can show them", async () => {
    mocked.save.mockResolvedValue({
      ok: true,
      vmesh: "built",
      advisories: ["2 vertices share a position with another."],
      export: {
        submeshes: 1,
        vertices: 24,
        triangles: 12,
        fanFallbacks: 0,
        fallbackTangents: 0,
        coincidentVertices: 2,
        reusedDiagonals: 0,
        nonFiniteWritten: 0,
        nonUnitNormalsWritten: 0,
      },
    });
    mocked.list.mockResolvedValue([docOf()]);
    await useDccStore.getState().save();
    expect(useDccStore.getState().lastSave?.advisories).toHaveLength(1);
    expect(useDccStore.getState().lastSave?.export.coincidentVertices).toBe(2);
  });

  it("surfaces a save failure as a status", async () => {
    mocked.save.mockRejectedValue("disk full");
    await useDccStore.getState().save();
    expect(useDccStore.getState().status).toContain("disk full");
  });
});
