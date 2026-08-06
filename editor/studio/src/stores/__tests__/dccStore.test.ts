// @vitest-environment jsdom
//
// Model Editor store (P23.4). Three things are worth testing here and the rest is
// plumbing:
//
// 1. **Two documents are two documents.** The panel is `singleton: false` and the
//    dock keeps inactive tabs mounted, so a store holding one `doc` gave both
//    panels the same mesh — every tool press in A edited B, and closing A closed
//    B. That is the blocker this file exists to hold shut.
// 2. The store **replaces** its document from every reply, because the backend
//    owns the selection and a structural op may have dropped it.
// 3. The preview is a **serialized queue of one, per document** — an orbit fires
//    faster than a render + PNG + base64 round trip completes.
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
    setGizmo: vi.fn(),
    dragBegin: vi.fn(),
    dragMove: vi.fn(),
    dragEnd: vi.fn(),
    dragCancel: vi.fn(),
    unwrap: vi.fn(),
    uvPreview: vi.fn(),
  },
  DCC_PREVIEW_SIZE: 256,
}));

import { dcc } from "../../lib/ipc";
import type { DccDocDto } from "../../bindings/DccDocDto";
import { __resetDccPreviewQueueForTest, useDccStore } from "../dccStore";

const mocked = vi.mocked(dcc);

function docOf(assetId: string, over: Partial<DccDocDto> = {}): DccDocDto {
  return {
    id: `dcc:${assetId}`,
    assetId,
    name: `Prop-${assetId}`,
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
    dragging: false,
    dragPoints: 0,
    gizmo: null,
    selectionRev: 1,
    seams: 0,
    charts: 1,
    import: {
      sourceVertices: 24,
      weldedPositions: 8,
      fanSplits: 0,
      degenerateTrianglesSkipped: 0,
      sharpEdges: 12,
      boundaryEdges: 0,
      nonFiniteValues: 0,
    },
    // P24.2: a fixture mesh is rigid unless a case says otherwise.
    skinJoints: null,
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

const entry = (assetId: string) => useDccStore.getState().docs[assetId];

beforeEach(() => {
  vi.clearAllMocks();
  __resetDccPreviewQueueForTest();
  useDccStore.setState({ docs: {} });
  mocked.preview.mockResolvedValue({ image: "data:image/png;base64,AA", error: null, size: 256 });
  mocked.open.mockImplementation((assetId: string) => Promise.resolve(docOf(assetId)));
});

afterEach(() => {
  __resetDccPreviewQueueForTest();
});

describe("two documents are two documents (the P23.4 blocker)", () => {
  beforeEach(async () => {
    await useDccStore.getState().open("aaa");
    await useDccStore.getState().open("bbb");
  });

  it("keeps both open at once, each with its own document", () => {
    expect(entry("aaa").doc?.id).toBe("dcc:aaa");
    expect(entry("bbb").doc?.id).toBe("dcc:bbb");
    expect(entry("aaa").doc?.name).toBe("Prop-aaa");
  });

  it("sends a tool press to the document it was pressed in", async () => {
    mocked.apply.mockResolvedValue({
      ok: true,
      refusal: null,
      doc: docOf("aaa", { faces: 10, dirty: true }),
    });
    await useDccStore.getState().apply("aaa", { tool: "extrude", distance: 0.5 });
    expect(mocked.apply).toHaveBeenCalledWith("dcc:aaa", { tool: "extrude", distance: 0.5 });
    expect(entry("aaa").doc?.faces).toBe(10);
    expect(entry("bbb").doc?.faces).toBe(6);
    expect(entry("bbb").doc?.dirty).toBe(false);
  });

  it("keeps a refusal on the document that refused", async () => {
    mocked.apply.mockResolvedValue({ ok: false, refusal: "no border", doc: docOf("bbb") });
    await useDccStore.getState().apply("bbb", { tool: "inset", amount: 0.1, individual: false });
    expect(entry("bbb").refusal).toBe("no border");
    expect(entry("aaa").refusal).toBeNull();
  });

  it("closes only the document it was asked to close", async () => {
    mocked.close.mockResolvedValue(undefined);
    await useDccStore.getState().close("aaa");
    expect(mocked.close).toHaveBeenCalledTimes(1);
    expect(mocked.close).toHaveBeenCalledWith("aaa");
    expect(entry("aaa")).toBeUndefined();
    expect(entry("bbb")?.doc?.id).toBe("dcc:bbb");
  });

  it("routes undo to the document it names", async () => {
    mocked.undo.mockResolvedValue(docOf("bbb", { canRedo: true }));
    await useDccStore.getState().undo("bbb");
    expect(mocked.undo).toHaveBeenCalledWith("dcc:bbb");
    expect(entry("bbb").doc?.canRedo).toBe(true);
    expect(entry("aaa").doc?.canRedo).toBe(false);
  });

  it("gives each document its own preview queue", async () => {
    // A global gate would have made two open panels take turns: a render in
    // flight for A would queue B's instead of starting it.
    vi.clearAllMocks();
    const a = deferred<{ image: string | null; error: string | null; size: number }>();
    mocked.preview.mockReturnValueOnce(a.promise);
    mocked.preview.mockResolvedValue({ image: "data:image/png;base64,BB", error: null, size: 256 });
    mocked.orbit.mockResolvedValue(undefined);

    const pa = useDccStore.getState().orbit("aaa", 5, 0, 0);
    const pb = useDccStore.getState().orbit("bbb", 5, 0, 0);
    await Promise.resolve();
    expect(mocked.preview).toHaveBeenCalledTimes(2);
    a.resolve({ image: "data:image/png;base64,AA", error: null, size: 256 });
    await Promise.all([pa, pb]);
  });
});

describe("open/close", () => {
  it("re-opening the same asset does not re-fetch", async () => {
    await useDccStore.getState().open("aaa");
    await useDccStore.getState().open("aaa");
    expect(mocked.open).toHaveBeenCalledTimes(1);
  });

  it("surfaces an open failure as a status rather than throwing", async () => {
    mocked.open.mockRejectedValue("not a mesh asset");
    await useDccStore.getState().open("aaa");
    expect(entry("aaa").doc).toBeNull();
    expect(entry("aaa").status).toContain("not a mesh asset");
  });
});

describe("the document is replaced, never patched", () => {
  beforeEach(async () => {
    await useDccStore.getState().open("aaa");
  });

  it("adopts the whole document a tool returns", async () => {
    // The load-bearing case: an extrude renumbers, so the backend rebuilt the
    // selection from the op's outcome. A store that merged fields would keep the
    // OLD `selected` and show a count for geometry that no longer exists.
    mocked.apply.mockResolvedValue({
      ok: true,
      refusal: null,
      doc: docOf("aaa", { faces: 10, verts: 12, selected: 1, generation: 2, dirty: true }),
    });
    await useDccStore.getState().apply("aaa", { tool: "extrude", distance: 0.5 });
    const doc = entry("aaa").doc!;
    expect(doc.faces).toBe(10);
    expect(doc.selected).toBe(1);
    expect(doc.generation).toBe(2);
    expect(doc.dirty).toBe(true);
  });

  it("clears the refusal on the next successful tool", async () => {
    mocked.apply.mockResolvedValueOnce({ ok: false, refusal: "nope", doc: docOf("aaa") });
    await useDccStore.getState().apply("aaa", { tool: "subdivide" });
    expect(entry("aaa").refusal).toBe("nope");
    mocked.apply.mockResolvedValueOnce({ ok: true, refusal: null, doc: docOf("aaa") });
    await useDccStore.getState().apply("aaa", { tool: "subdivide" });
    expect(entry("aaa").refusal).toBeNull();
  });

  it("sends the pick in preview pixels at the shared size", async () => {
    mocked.pick.mockResolvedValue(docOf("aaa", { selected: 1 }));
    await useDccStore.getState().pick("aaa", 12, 34, true);
    // The size is the ONE constant `dcc_preview` was given: a pick computed
    // against a different projection lands somewhere else.
    expect(mocked.pick).toHaveBeenCalledWith("dcc:aaa", 12, 34, 256, true);
    expect(entry("aaa").doc?.selected).toBe(1);
  });

  it("routes a mode switch through the select command", async () => {
    mocked.select.mockResolvedValue(docOf("aaa", { mode: "edge" }));
    await useDccStore.getState().setMode("aaa", "edge");
    expect(mocked.select).toHaveBeenCalledWith("dcc:aaa", { action: "mode", mode: "edge" });
    expect(entry("aaa").doc?.mode).toBe("edge");
  });
});

describe("the preview queue", () => {
  beforeEach(async () => {
    await useDccStore.getState().open("aaa");
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

    const bursts = Array.from({ length: 10 }, () => useDccStore.getState().orbit("aaa", 5, 0, 0));
    // All ten orbit calls reach the backend — the CAMERA is cheap and must not
    // be dropped, or the model would stop following the mouse.
    await Promise.resolve();
    expect(mocked.orbit).toHaveBeenCalledTimes(10);
    expect(mocked.preview).toHaveBeenCalledTimes(1);

    first.resolve({ image: "data:image/png;base64,AA", error: null, size: 256 });
    await Promise.all(bursts);
    expect(mocked.preview).toHaveBeenCalledTimes(2);
    expect(entry("aaa").image).toContain("BB");
  });

  it("reports a preview failure without clearing the document", async () => {
    mocked.preview.mockResolvedValue({
      image: null,
      error: "no GPU adapter on this machine",
      size: 256,
    });
    useDccStore.getState().refresh("aaa");
    await vi.waitFor(() => expect(entry("aaa").previewError).toContain("no GPU adapter"));
    expect(entry("aaa").doc).not.toBeNull();
  });
});

describe("save", () => {
  const report = (coincident: number) => ({
    submeshes: 1,
    vertices: 24,
    triangles: 12,
    fanFallbacks: 0,
    fallbackTangents: 0,
    coincidentVertices: coincident,
    reusedDiagonals: 0,
    nonFiniteWritten: 0,
    nonUnitNormalsWritten: 0,
  });

  beforeEach(async () => {
    mocked.open.mockImplementation((assetId: string) =>
      Promise.resolve(docOf(assetId, { dirty: true })),
    );
    await useDccStore.getState().open("aaa");
  });

  it("keeps the verdict and re-reads the document's dirty flag", async () => {
    mocked.save.mockResolvedValue({ ok: true, vmesh: "built", advisories: [], export: report(0) });
    // `dirty` is derived from the generation stamp on the BACKEND, so the only
    // honest way to learn it after a save is to ask.
    mocked.list.mockResolvedValue([docOf("aaa", { dirty: false })]);
    await useDccStore.getState().save("aaa");
    expect(entry("aaa").lastSave?.vmesh).toBe("built");
    expect(entry("aaa").doc?.dirty).toBe(false);
    expect(entry("aaa").status).toContain("12 triangles");
  });

  it("keeps the coincidence advisory until the NEXT SAVE", async () => {
    // It used to be cleared by the next action. That warning is the one sentence
    // in the product saying "what is on disk is not what is in this session", and
    // an author who pressed Save and then moved the mouse never read it.
    mocked.save.mockResolvedValue({
      ok: true,
      vmesh: "built",
      advisories: ["2 vertices share a position with another."],
      export: report(2),
    });
    mocked.list.mockResolvedValue([docOf("aaa")]);
    await useDccStore.getState().save("aaa");
    expect(entry("aaa").lastSave?.advisories).toHaveLength(1);

    mocked.apply.mockResolvedValue({ ok: true, refusal: null, doc: docOf("aaa") });
    await useDccStore.getState().apply("aaa", { tool: "subdivide" });
    await useDccStore.getState().select("aaa", { action: "all" }).catch(() => {});
    expect(entry("aaa").lastSave?.export.coincidentVertices).toBe(2);

    mocked.save.mockResolvedValue({ ok: true, vmesh: "built", advisories: [], export: report(0) });
    await useDccStore.getState().save("aaa");
    expect(entry("aaa").lastSave?.advisories).toHaveLength(0);
  });

  it("surfaces a save failure as a status", async () => {
    mocked.save.mockRejectedValue("disk full");
    await useDccStore.getState().save("aaa");
    expect(entry("aaa").status).toContain("disk full");
  });
});

describe("the close race (P23.4 re-audit — M-2/M-3)", () => {
  it("closes a document whose open has not resolved yet", async () => {
    // The leak the tombstone exists for. `close` used to read `entry().doc`,
    // find null because the open was still in flight, and send NOTHING — so the
    // backend `DccDoc`, its `MeshSession` and its journal lived for the process.
    // React StrictMode walks this path on every dev mount.
    const open = deferred<ReturnType<typeof docOf>>();
    mocked.open.mockReturnValueOnce(open.promise);
    mocked.close.mockResolvedValue(undefined);

    const opening = useDccStore.getState().open("aaa");
    const closing = useDccStore.getState().close("aaa");
    await closing;
    // Sent by ASSET id, without waiting for a document that does not exist yet.
    expect(mocked.close).toHaveBeenCalledWith("aaa");

    open.resolve(docOf("aaa"));
    await opening;

    // The late reply neither resurrects the entry…
    expect(entry("aaa")).toBeUndefined();
    expect(Object.keys(useDccStore.getState().docs)).toHaveLength(0);
    // …nor leaves the session it created behind: the first close found nothing,
    // so the open's own reply sends another.
    expect(mocked.close).toHaveBeenCalledTimes(2);
  });

  it("drops a late tool reply for a closed document instead of resurrecting it", async () => {
    // `patch` used to create an entry unconditionally, so ANY late-resolving
    // open/apply/save/mergeAsset put a deleted document straight back and `docs`
    // never shrank.
    await useDccStore.getState().open("aaa");
    const apply = deferred<{ ok: boolean; refusal: string | null; doc: DccDocDto }>();
    mocked.apply.mockReturnValueOnce(apply.promise);
    mocked.close.mockResolvedValue(undefined);

    const applying = useDccStore.getState().apply("aaa", { tool: "subdivide" });
    await useDccStore.getState().close("aaa");
    expect(entry("aaa")).toBeUndefined();

    apply.resolve({ ok: true, refusal: null, doc: docOf("aaa", { faces: 24 }) });
    await applying;
    expect(entry("aaa")).toBeUndefined();
    expect(Object.keys(useDccStore.getState().docs)).toHaveLength(0);
  });

  it("re-opening after a close starts fresh rather than being tombstoned forever", async () => {
    mocked.close.mockResolvedValue(undefined);
    await useDccStore.getState().open("aaa");
    await useDccStore.getState().close("aaa");
    await useDccStore.getState().open("aaa");
    expect(entry("aaa").doc?.id).toBe("dcc:aaa");
  });
});

describe("pointer drags (P23.5)", () => {
  beforeEach(async () => {
    await useDccStore.getState().open("aaa");
    await useDccStore.getState().open("bbb");
    vi.clearAllMocks();
    mocked.preview.mockResolvedValue({ image: "data:image/png;base64,AA", error: null, size: 256 });
  });

  it("reports a MISS as a miss rather than as a refusal", () => {
    // The distinction the whole tool rests on: a pointer that found neither the
    // surface nor a handle is the panel's cue to ORBIT, and putting "missed" in
    // the status bar as a refusal would train an author to ignore that line.
    mocked.dragBegin.mockResolvedValue({
      grabbed: false,
      handle: null,
      refusal: null,
      doc: docOf("aaa"),
    });
    return useDccStore
      .getState()
      .dragBegin("aaa", { kind: "sculpt", mode: "draw", radius: 0.3, strength: 0.05, falloff: "Smooth" }, 10, 20)
      .then((grabbed) => {
        expect(grabbed).toBe(false);
        expect(entry("aaa").refusal).toBeNull();
      });
  });

  it("passes the preview size both calls share, and adopts the returned document", async () => {
    // The P23.4 rule, still true for drags: the projection is built from `size`,
    // so a drag begun at one size against an image rendered at another lands
    // somewhere else.
    mocked.dragBegin.mockResolvedValue({
      grabbed: true,
      handle: "X",
      refusal: null,
      doc: docOf("aaa", { dragging: true, dragPoints: 1 }),
    });
    const grabbed = await useDccStore
      .getState()
      .dragBegin("aaa", { kind: "gizmo", mode: "translate", snap: 0, softRadius: 0, falloff: "Smooth" }, 10, 20);
    expect(grabbed).toBe(true);
    expect(mocked.dragBegin).toHaveBeenCalledWith(
      "dcc:aaa",
      { kind: "gizmo", mode: "translate", snap: 0, softRadius: 0, falloff: "Smooth" },
      10,
      20,
      256,
    );
    expect(entry("aaa").doc?.dragging).toBe(true);
    expect(entry("bbb").doc?.dragging).toBe(false);
  });

  it("renders a frame per move without asking for a document", async () => {
    // A move is the highest-frequency call in the panel. It returns nothing on
    // purpose — the preview frame IS the feedback — so a drag costs one round
    // trip per pointer-move rather than two.
    mocked.dragMove.mockResolvedValue(undefined);
    await useDccStore.getState().dragMove("aaa", 11, 22);
    expect(mocked.dragMove).toHaveBeenCalledWith("dcc:aaa", 11, 22, 256);
    expect(mocked.preview).toHaveBeenCalled();
  });

  it("surfaces a refusal from the settle, on the document that refused", async () => {
    mocked.dragEnd.mockResolvedValue({
      ok: false,
      refusal: "a sculpted vertex position is not finite",
      doc: docOf("aaa", { dragging: false }),
    });
    await useDccStore.getState().dragEnd("aaa");
    expect(entry("aaa").refusal).toContain("not finite");
    expect(entry("bbb").refusal).toBeNull();
  });

  it("cancelling clears the drag and arming the gizmo goes to the backend", async () => {
    mocked.dragCancel.mockResolvedValue(docOf("aaa", { dragging: false }));
    await useDccStore.getState().dragCancel("aaa");
    expect(entry("aaa").doc?.dragging).toBe(false);

    mocked.setGizmo.mockResolvedValue(docOf("aaa", { gizmo: "rotate" }));
    await useDccStore.getState().setGizmo("aaa", "rotate");
    expect(mocked.setGizmo).toHaveBeenCalledWith("dcc:aaa", "rotate");
    expect(entry("aaa").doc?.gizmo).toBe("rotate");
  });
});

describe("UV (P23.5)", () => {
  beforeEach(async () => {
    await useDccStore.getState().open("aaa");
    await useDccStore.getState().open("bbb");
    vi.clearAllMocks();
    mocked.preview.mockResolvedValue({ image: "data:image/png;base64,AA", error: null, size: 256 });
    mocked.uvPreview.mockResolvedValue({
      image: "data:image/png;base64,UV",
      error: null,
      size: 256,
    });
  });

  it("keeps the unwrap verdict — including its residual — on the document", async () => {
    // The residual is the one honest number a fixed-iteration solver owes its
    // caller, so it has to survive the round trip and stay put: an author who
    // unwraps and then clicks anything else must still be able to read it.
    mocked.unwrap.mockResolvedValue({
      ok: true,
      refusal: null,
      charts: 6,
      corners: 24,
      seams: 12,
      worstResidual: 3.5e-12,
      worstConvergence: 1.2e-15,
      flipped: 0,
      triangles: 24,
      doc: docOf("aaa", { seams: 12, charts: 6 }),
    });
    await useDccStore.getState().unwrap("aaa");
    expect(entry("aaa").lastUnwrap?.charts).toBe(6);
    expect(entry("aaa").lastUnwrap?.worstResidual).toBeCloseTo(3.5e-12);
    expect(entry("aaa").doc?.charts).toBe(6);
    expect(entry("aaa").uvImage).toContain("UV");
    // …and it went to the document that asked, not to its neighbour.
    expect(entry("bbb").lastUnwrap).toBeNull();
    expect(entry("bbb").uvImage).toBeNull();
  });

  it("shows a refusal instead of pretending the unwrap worked", async () => {
    mocked.unwrap.mockResolvedValue({
      ok: false,
      refusal: "chart 2 cannot be unwrapped: every triangle in it has zero area",
      charts: 0,
      corners: 0,
      seams: 4,
      worstResidual: 0,
      worstConvergence: 0,
      flipped: 0,
      triangles: 0,
      doc: docOf("aaa"),
    });
    await useDccStore.getState().unwrap("aaa");
    expect(entry("aaa").refusal).toContain("zero area");
    expect(entry("aaa").lastUnwrap?.ok).toBe(false);
  });

  it("renders a UV frame on demand, per document", async () => {
    await useDccStore.getState().uvRefresh("aaa");
    expect(mocked.uvPreview).toHaveBeenCalledWith("dcc:aaa", 256);
    expect(entry("aaa").uvImage).toContain("UV");
    expect(entry("bbb").uvImage).toBeNull();
  });
});

describe("the UV pane follows the selection, not only the journal (P23.5 audit M3)", () => {
  beforeEach(async () => {
    await useDccStore.getState().open("aaa");
    vi.clearAllMocks();
    mocked.preview.mockResolvedValue({ image: "data:image/png;base64,AA", error: null, size: 256 });
    mocked.uvPreview.mockResolvedValue({
      image: "data:image/png;base64,UV1",
      error: null,
      size: 256,
    });
    // The pane has been opened once, which is what makes it something to keep
    // current. A document that never opened it does not pay for the refresh.
    await useDccStore.getState().uvRefresh("aaa");
    vi.clearAllMocks();
  });

  it("re-renders the UV frame when a pick changes the SELECTION", async () => {
    // The defect: the pane keyed on `doc.generation`, the JOURNAL stamp, which a
    // selection change does not move — so picking a different face left the UV
    // view showing the previous one. `doc.selected` could not have fixed it
    // either: it is a count, and face A and face B both read 1.
    mocked.uvPreview.mockResolvedValue({
      image: "data:image/png;base64,UV2",
      error: null,
      size: 256,
    });
    mocked.pick.mockResolvedValue(
      docOf("aaa", { generation: 1, selected: 1, selectionRev: 999 }),
    );
    await useDccStore.getState().pick("aaa", 10, 10, false);
    expect(mocked.uvPreview).toHaveBeenCalledTimes(1);
    expect(entry("aaa").uvImage).toContain("UV2");
  });

  it("does not re-render when nothing about the document moved", async () => {
    // The other half: a call that returns the same selection at the same
    // generation must not cost a UV render. Without this the fix is "refresh
    // always", which is a different defect.
    mocked.select.mockResolvedValue(docOf("aaa", { generation: 1, selectionRev: 1 }));
    await useDccStore.getState().select("aaa", { action: "grow" });
    expect(mocked.uvPreview).not.toHaveBeenCalled();
  });

  it("re-renders when the journal moves, too", async () => {
    mocked.apply.mockResolvedValue({
      ok: true,
      refusal: null,
      doc: docOf("aaa", { generation: 42, selectionRev: 1 }),
    });
    await useDccStore.getState().apply("aaa", { tool: "subdivide" });
    expect(mocked.uvPreview).toHaveBeenCalledTimes(1);
  });

  it("surfaces a drag REFUSAL while leaving a plain miss silent", async () => {
    // A miss becomes a camera orbit and must not put anything in the status bar;
    // a refusal — the brush radius below its floor — is a sentence, or the author
    // is left wondering whether they missed the model.
    mocked.dragBegin.mockResolvedValue({
      grabbed: false,
      handle: null,
      refusal: null,
      doc: docOf("aaa"),
    });
    await useDccStore
      .getState()
      .dragBegin("aaa", { kind: "sculpt", mode: "draw", radius: 0.3, strength: 0.05, falloff: "Smooth" }, 1, 1);
    expect(entry("aaa").refusal).toBeNull();

    mocked.dragBegin.mockResolvedValue({
      grabbed: false,
      handle: null,
      refusal: "a brush radius of 0.000000000001 m is below the 0.001 m floor",
      doc: docOf("aaa"),
    });
    await useDccStore
      .getState()
      .dragBegin("aaa", { kind: "sculpt", mode: "draw", radius: 1e-12, strength: 0.05, falloff: "Smooth" }, 1, 1);
    expect(entry("aaa").refusal).toContain("below the 0.001 m floor");
  });
});
