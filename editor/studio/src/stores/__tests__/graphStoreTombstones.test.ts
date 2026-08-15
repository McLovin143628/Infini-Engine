// @vitest-environment jsdom
//
// **The tombstone, in the four single-document graph stores** (F-lens L7.M3).
//
// `blueprintStore`, `materialStore`, `pcgStore` and `smStore` all open exactly
// one backend document by calling `list()` and, if it comes back empty,
// `create("Main")`. Two defects lived in that shape, and `dccStore`/`skelStore`
// had already paid to learn both:
//
//  1. **The double create.** `init` guarded on `get().ready`, which is set
//     several awaits later. React StrictMode's mount → cleanup → mount runs both
//     mounts before the first `list()` returns, so both saw `ready === false`,
//     both found nothing, and both called `create` — two backend documents, each
//     with its own journal, one adopted and one orphaned for the process. Unlike
//     `dcc_open`, `*_create` is not keyed by an asset, so the backend cannot
//     dedupe it: the second `init` has to JOIN the first.
//  2. **The close that had nothing to name.** `close` reads `get().doc`, which
//     is null while the create is in flight, so it sent no `*_close` at all and
//     the document the reply was about to hand over leaked.
//
// One table drives all four, because they are the same store four times over —
// and a defect repaired in one of them is worth catching in the others.
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  graph: {
    registry: vi.fn(),
    list: vi.fn(),
    create: vi.fn(),
    close: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
  },
  material: {
    registry: vi.fn(),
    list: vi.fn(),
    create: vi.fn(),
    close: vi.fn(),
    compile: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
  },
  pcg: {
    registry: vi.fn(),
    list: vi.fn(),
    create: vi.fn(),
    close: vi.fn(),
    compile: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
  },
  sm: {
    list: vi.fn(),
    create: vi.fn(),
    close: vi.fn(),
    listClips: vi.fn(),
  },
}));

import { graph, material, pcg, sm } from "../../lib/ipc";
import type { BpDoc } from "../../lib/blueprintTypes";
import { useBlueprintStore, __resetBlueprintInitForTest } from "../blueprintStore";
import { useMaterialStore, __resetMaterialInitForTest } from "../materialStore";
import { usePcgStore, __resetPcgInitForTest } from "../pcgStore";
import { useSmStore, __resetSmInitForTest } from "../smStore";

/** A promise a test can hold open, so "still in flight" is reproducible. */
function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

function bpDoc(id: string): BpDoc {
  return {
    id,
    name: "Main",
    graph: { nodes: {}, links: [], nextId: 1 },
    viewport: null,
    modifiedMs: 0,
  };
}

interface Subject {
  name: string;
  ipc: {
    list: ReturnType<typeof vi.fn>;
    create: ReturnType<typeof vi.fn>;
    close: ReturnType<typeof vi.fn>;
  };
  reset: () => void;
  init: () => Promise<void>;
  close: () => Promise<void>;
  docId: () => string | undefined;
  ready: () => boolean;
  /** Make every call this store's `init` performs resolve normally. */
  arrange: () => void;
}

const SUBJECTS: Subject[] = [
  {
    name: "blueprintStore",
    ipc: vi.mocked(graph),
    reset: () => {
      __resetBlueprintInitForTest();
      useBlueprintStore.setState({ doc: null, ready: false, registry: [], registryById: {} });
    },
    init: () => useBlueprintStore.getState().init(),
    close: () => useBlueprintStore.getState().close(),
    docId: () => useBlueprintStore.getState().doc?.id,
    ready: () => useBlueprintStore.getState().ready,
    arrange: () => {
      vi.mocked(graph.registry).mockResolvedValue([]);
      vi.mocked(graph.close).mockResolvedValue(undefined);
    },
  },
  {
    name: "materialStore",
    ipc: vi.mocked(material),
    reset: () => {
      __resetMaterialInitForTest();
      useMaterialStore.setState({ doc: null, ready: false, registry: [], registryById: {} });
    },
    init: () => useMaterialStore.getState().init(),
    close: () => useMaterialStore.getState().close(),
    docId: () => useMaterialStore.getState().doc?.id,
    ready: () => useMaterialStore.getState().ready,
    arrange: () => {
      vi.mocked(material.registry).mockResolvedValue([]);
      vi.mocked(material.close).mockResolvedValue(undefined);
      vi.mocked(material.compile).mockRejectedValue("no compiler in this test");
    },
  },
  {
    name: "pcgStore",
    ipc: vi.mocked(pcg),
    reset: () => {
      __resetPcgInitForTest();
      usePcgStore.setState({ doc: null, ready: false, registry: [], registryById: {} });
    },
    init: () => usePcgStore.getState().init(),
    close: () => usePcgStore.getState().close(),
    docId: () => usePcgStore.getState().doc?.id,
    ready: () => usePcgStore.getState().ready,
    arrange: () => {
      vi.mocked(pcg.registry).mockResolvedValue([]);
      vi.mocked(pcg.close).mockResolvedValue(undefined);
      vi.mocked(pcg.compile).mockRejectedValue("no compiler in this test");
    },
  },
  {
    name: "smStore",
    ipc: vi.mocked(sm),
    reset: () => {
      __resetSmInitForTest();
      useSmStore.setState({ doc: null, ready: false, clips: [] });
    },
    init: () => useSmStore.getState().init(),
    close: () => useSmStore.getState().close(),
    docId: () => useSmStore.getState().doc?.id,
    ready: () => useSmStore.getState().ready,
    arrange: () => {
      vi.mocked(sm.listClips).mockResolvedValue([]);
      vi.mocked(sm.close).mockResolvedValue(undefined);
    },
  },
];

describe.each(SUBJECTS)("$name — the tombstone (L7.M3)", (s) => {
  beforeEach(() => {
    vi.clearAllMocks();
    s.reset();
    s.arrange();
  });

  it("creates ONE document across a StrictMode double init", async () => {
    // The headline. `list()` is held open so both mounts are genuinely in
    // flight at the same time, which is exactly what StrictMode produces.
    const list = deferred<BpDoc[]>();
    s.ipc.list.mockReturnValueOnce(list.promise);
    s.ipc.list.mockResolvedValue([]);
    s.ipc.create.mockImplementation(() => Promise.resolve(bpDoc("doc:1")));

    const a = s.init();
    const b = s.init();
    list.resolve([]);
    await Promise.all([a, b]);

    expect(s.ipc.create).toHaveBeenCalledTimes(1);
    expect(s.ipc.list).toHaveBeenCalledTimes(1);
    expect(s.docId()).toBe("doc:1");
    expect(s.ready()).toBe(true);
    // Nothing was orphaned, so nothing had to be cleaned up.
    expect(s.ipc.close).not.toHaveBeenCalled();
  });

  it("adopts the existing document rather than minting a second", async () => {
    s.ipc.list.mockResolvedValue([bpDoc("doc:existing")]);
    await s.init();
    expect(s.ipc.create).not.toHaveBeenCalled();
    expect(s.docId()).toBe("doc:existing");
  });

  it("closes a document whose init had not resolved when the panel closed", async () => {
    // The leak the tombstone exists for. `close` used to read `get().doc`, find
    // null because the create was still in flight, and send NOTHING — so the
    // backend document and its journal lived for the process.
    const list = deferred<BpDoc[]>();
    s.ipc.list.mockReturnValueOnce(list.promise);
    s.ipc.create.mockResolvedValue(bpDoc("doc:orphan"));

    const initing = s.init();
    await s.close();
    // The close itself had no id to name…
    expect(s.ipc.close).not.toHaveBeenCalled();

    list.resolve([]);
    await initing;

    // …so the init's own reply sent it, and adopted nothing.
    expect(s.ipc.close).toHaveBeenCalledTimes(1);
    expect(s.ipc.close).toHaveBeenCalledWith("doc:orphan");
    expect(s.docId()).toBeUndefined();
    expect(s.ready()).toBe(false);
  });

  it("lets a StrictMode remount ADOPT rather than tear down", async () => {
    // init → close → init, all before the first `list()` returns: the shape
    // React actually produces. A re-init clears the tombstone (exactly as
    // `dccStore.open` deletes it), so the in-flight document is adopted, one
    // document exists, and nothing is closed.
    const list = deferred<BpDoc[]>();
    s.ipc.list.mockReturnValueOnce(list.promise);
    s.ipc.create.mockResolvedValue(bpDoc("doc:1"));

    const first = s.init();
    await s.close();
    const second = s.init();
    list.resolve([]);
    await Promise.all([first, second]);

    expect(s.ipc.create).toHaveBeenCalledTimes(1);
    expect(s.ipc.close).not.toHaveBeenCalled();
    expect(s.docId()).toBe("doc:1");
    expect(s.ready()).toBe(true);
  });

  it("re-opening after a settled close starts fresh, not tombstoned forever", async () => {
    s.ipc.list.mockResolvedValue([]);
    s.ipc.create.mockResolvedValueOnce(bpDoc("doc:1")).mockResolvedValueOnce(bpDoc("doc:2"));

    await s.init();
    expect(s.docId()).toBe("doc:1");
    await s.close();
    expect(s.ipc.close).toHaveBeenCalledWith("doc:1");

    await s.init();
    expect(s.docId()).toBe("doc:2");
    expect(s.ready()).toBe(true);
  });

  it("a failed init does not wedge the memo — the next one retries", async () => {
    s.ipc.list.mockRejectedValueOnce("backend not up");
    await s.init();
    expect(s.ready()).toBe(false);

    s.ipc.list.mockResolvedValue([]);
    s.ipc.create.mockResolvedValue(bpDoc("doc:retry"));
    await s.init();
    expect(s.docId()).toBe("doc:retry");
  });
});
