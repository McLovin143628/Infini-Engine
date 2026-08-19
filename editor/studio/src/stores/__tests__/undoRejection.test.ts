// @vitest-environment jsdom
//
// **Undo/redo must not reject** (F-lens L7.M4).
//
// Every caller of these actions drops the promise: the canvas toolbars do
// `onClick={() => void undo()}`, and `dispatchUndo` does `void scope.undo(...)`.
// A `void`-ed promise still rejects — it just rejects with nobody listening — so
// a `graph_undo` that fails (a closed document, a backend panic, an empty
// history) used to surface as an unhandled promise rejection and NOTHING at all
// in the product. Six of these paths were bare `await`s while every other
// backend call in the same stores already had a `try`/`catch`.
//
// Each arm asserts the failure is CONTAINED — the action resolves rather than
// rejecting — and that the store is left usable. `unhandledrejection` is watched
// globally so an arm cannot pass by moving the rejection somewhere quieter.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  graph: { undo: vi.fn(), redo: vi.fn() },
  material: { undo: vi.fn(), redo: vi.fn(), compile: vi.fn() },
  pcg: { undo: vi.fn(), redo: vi.fn(), compile: vi.fn() },
  dcc: { undo: vi.fn(), redo: vi.fn(), preview: vi.fn(), open: vi.fn(), close: vi.fn() },
  scene: { undo: vi.fn(), redo: vi.fn() },
  sm: { list: vi.fn(), create: vi.fn(), close: vi.fn(), listClips: vi.fn() },
  DCC_PREVIEW_SIZE: 256,
}));

import { dcc, graph, material, pcg, scene } from "../../lib/ipc";
import type { BpDoc } from "../../lib/blueprintTypes";
import { useBlueprintStore } from "../blueprintStore";
import { useMaterialStore } from "../materialStore";
import { usePcgStore } from "../pcgStore";
import { useSceneStore } from "../sceneStore";
import { useDccStore } from "../dccStore";
import { useShellStore } from "../shellStore";

function bpDoc(): BpDoc {
  return {
    id: "doc:1",
    name: "Main",
    graph: { nodes: {}, links: [], nextId: 1 },
    viewport: null,
    modifiedMs: 0,
  };
}

/** Fail the test if anything reaches the global unhandled-rejection hook. */
let unhandled: unknown[] = [];
const onUnhandled = (e: PromiseRejectionEvent) => {
  unhandled.push(e.reason);
  e.preventDefault();
};

beforeEach(() => {
  vi.clearAllMocks();
  unhandled = [];
  window.addEventListener("unhandledrejection", onUnhandled);
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(async () => {
  // Rejections are reported a turn late; give them one.
  await new Promise((r) => setTimeout(r, 0));
  window.removeEventListener("unhandledrejection", onUnhandled);
  expect(unhandled).toEqual([]);
  vi.restoreAllMocks();
});

describe.each([
  {
    name: "blueprintStore",
    arrange: () => useBlueprintStore.setState({ doc: bpDoc(), ready: true }),
    ipc: () => vi.mocked(graph),
    undo: () => useBlueprintStore.getState().undo(),
    redo: () => useBlueprintStore.getState().redo(),
    stillUsable: () => expect(useBlueprintStore.getState().doc?.id).toBe("doc:1"),
  },
  {
    name: "materialStore",
    arrange: () => {
      vi.mocked(material.compile).mockResolvedValue(undefined as never);
      useMaterialStore.setState({ doc: bpDoc(), ready: true });
    },
    ipc: () => vi.mocked(material),
    undo: () => useMaterialStore.getState().undo(),
    redo: () => useMaterialStore.getState().redo(),
    stillUsable: () => expect(useMaterialStore.getState().doc?.id).toBe("doc:1"),
  },
  {
    name: "pcgStore",
    arrange: () => {
      vi.mocked(pcg.compile).mockResolvedValue(undefined as never);
      usePcgStore.setState({ doc: bpDoc(), ready: true });
    },
    ipc: () => vi.mocked(pcg),
    undo: () => usePcgStore.getState().undo(),
    redo: () => usePcgStore.getState().redo(),
    stillUsable: () => expect(usePcgStore.getState().doc?.id).toBe("doc:1"),
  },
])("$name undo/redo contain a backend rejection", (s) => {
  it("undo resolves instead of rejecting", async () => {
    s.arrange();
    s.ipc().undo.mockRejectedValue(new Error("document is closed"));
    await expect(s.undo()).resolves.toBeUndefined();
    s.stillUsable();
  });

  it("redo resolves instead of rejecting", async () => {
    s.arrange();
    s.ipc().redo.mockRejectedValue(new Error("document is closed"));
    await expect(s.redo()).resolves.toBeUndefined();
    s.stillUsable();
  });
});

describe("sceneStore (L7.M4)", () => {
  it("reports a refused undo in the status bar instead of dropping it", async () => {
    // `void sceneIpc.undo()` dropped the rejection along with the value. An
    // empty history is the common case, and the author saw nothing at all.
    vi.mocked(scene.undo).mockRejectedValue("nothing to undo");
    useSceneStore.getState().undo();
    await vi.waitFor(() =>
      expect(useShellStore.getState().statusMessage).toContain("Undo failed"),
    );
  });

  it("reports a refused redo too", async () => {
    vi.mocked(scene.redo).mockRejectedValue("nothing to redo");
    useSceneStore.getState().redo();
    await vi.waitFor(() =>
      expect(useShellStore.getState().statusMessage).toContain("Redo failed"),
    );
  });
});

describe("dccStore (L7.M4)", () => {
  beforeEach(async () => {
    vi.mocked(dcc.preview).mockResolvedValue({ image: null, error: null, size: 256 } as never);
    useDccStore.setState({
      docs: {
        aaa: {
          doc: { id: "dcc:aaa" } as never,
          image: null,
          previewError: null,
          refusal: null,
          lastSave: null,
          busy: false,
          status: null,
          uvImage: null,
          lastUnwrap: null,
          lastGroom: null,
          history: [],
        },
      },
    });
  });

  it("surfaces an undo rejection as a refusal on the document", async () => {
    vi.mocked(dcc.undo).mockRejectedValue("journal cursor is at the base");
    await expect(useDccStore.getState().undo("aaa")).resolves.toBeUndefined();
    expect(useDccStore.getState().docs.aaa?.refusal).toContain("journal cursor");
  });

  it("surfaces a redo rejection the same way", async () => {
    vi.mocked(dcc.redo).mockRejectedValue("nothing to redo");
    await expect(useDccStore.getState().redo("aaa")).resolves.toBeUndefined();
    expect(useDccStore.getState().docs.aaa?.refusal).toContain("nothing to redo");
  });
});
