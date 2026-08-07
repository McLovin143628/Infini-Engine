// @vitest-environment jsdom
//
// Skeleton Editor store (P24.3). The three things worth testing, and the rest is
// plumbing:
//
//  * the store is KEYED, so two open rigs do not share a document (the defect
//    `dccStore` shipped with and had to be re-audited for);
//  * the close race — a panel that unmounts before its `open` resolves must not
//    leak a backend `SkelSession`, and its late reply must not resurrect the
//    entry (the tombstone rule);
//  * the selection is CLAMPED across an edit that shrinks the rig, and the
//    warning/refusal split behaves — a refusal keeps the previous warning
//    readable, a clean success clears it.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  skel: {
    open: vi.fn(),
    close: vi.fn(),
    list: vi.fn(),
    renameJoint: vi.fn(),
    setJointTransform: vi.fn(),
    setLimit: vi.fn(),
    addSocket: vi.fn(),
    removeSocket: vi.fn(),
    placeSocket: vi.fn(),
    instantiateTemplate: vi.fn(),
    mergePart: vi.fn(),
    undo: vi.fn(),
    redo: vi.fn(),
    save: vi.fn(),
    fitToMesh: vi.fn(),
  },
}));

import { skel } from "../../lib/ipc";
import type { SkelDocDto } from "../../bindings/SkelDocDto";
import type { SkelJointDto } from "../../bindings/SkelJointDto";
import { __resetSkelTombstonesForTest, useSkelStore } from "../skelStore";

const mocked = vi.mocked(skel);

function jointOf(
  index: number,
  over: Partial<SkelJointDto> = {},
): SkelJointDto {
  return {
    index,
    name: `j${index}`,
    parent: index === 0 ? null : index - 1,
    translation: [0, index, 0],
    rotation: [0, 0, 0, 1],
    scale: [1, 1, 1],
    canonical: false,
    mirror: null,
    sidedWithoutTwin: false,
    limitMinDeg: null,
    limitMaxDeg: null,
    ...over,
  };
}

function docOf(
  assetId: string,
  joints = 4,
  over: Partial<SkelDocDto> = {},
): SkelDocDto {
  return {
    id: assetId,
    name: `Rig-${assetId}`,
    joints: Array.from({ length: joints }, (_, i) => jointOf(i)),
    sockets: [],
    dirty: false,
    canUndo: false,
    canRedo: false,
    unmatchedSided: [],
    ...over,
  };
}

const ok = (doc: SkelDocDto, warning: string | null = null) => ({
  ok: true,
  refusal: null,
  warning,
  doc,
});
const refused = (doc: SkelDocDto, refusal: string) => ({
  ok: false,
  refusal,
  warning: null,
  doc,
});

/** A deferred promise, so a test can hold a call open and watch what queues. */
function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const entry = (assetId: string) => useSkelStore.getState().docs[assetId];

beforeEach(() => {
  vi.clearAllMocks();
  __resetSkelTombstonesForTest();
  useSkelStore.setState({ docs: {} });
  mocked.open.mockImplementation((assetId: string) =>
    Promise.resolve(docOf(assetId)),
  );
  mocked.close.mockResolvedValue(undefined);
});

afterEach(() => __resetSkelTombstonesForTest());

describe("two open rigs are two documents", () => {
  beforeEach(async () => {
    await useSkelStore.getState().open("aaa");
    await useSkelStore.getState().open("bbb");
  });

  it("keeps a document per asset", () => {
    expect(entry("aaa").doc?.name).toBe("Rig-aaa");
    expect(entry("bbb").doc?.name).toBe("Rig-bbb");
  });

  it("sends an edit to the rig it was made in", async () => {
    mocked.renameJoint.mockResolvedValue(
      ok(
        docOf("aaa", 4, {
          dirty: true,
          joints: [jointOf(0, { name: "pelvis" })],
        }),
      ),
    );
    await useSkelStore.getState().renameJoint("aaa", 0, "pelvis");
    expect(mocked.renameJoint).toHaveBeenCalledWith("aaa", 0, "pelvis");
    expect(entry("aaa").doc?.joints[0].name).toBe("pelvis");
    expect(entry("aaa").doc?.dirty).toBe(true);
    // …and the neighbour is untouched.
    expect(entry("bbb").doc?.joints[0].name).toBe("j0");
    expect(entry("bbb").doc?.dirty).toBe(false);
  });

  it("closes only the document it was asked to close", async () => {
    await useSkelStore.getState().close("aaa");
    expect(mocked.close).toHaveBeenCalledTimes(1);
    expect(mocked.close).toHaveBeenCalledWith("aaa");
    expect(entry("aaa")).toBeUndefined();
    expect(entry("bbb")?.doc?.id).toBe("bbb");
  });

  it("selects per document", () => {
    useSkelStore.getState().select("aaa", 2);
    expect(entry("aaa").selected).toBe(2);
    expect(entry("bbb").selected).toBeNull();
  });
});

describe("the close race (the tombstone rule)", () => {
  it("closes a document whose open has not resolved yet", async () => {
    const open = deferred<SkelDocDto>();
    mocked.open.mockReturnValueOnce(open.promise);

    const opening = useSkelStore.getState().open("aaa");
    await useSkelStore.getState().close("aaa");
    // Sent by ASSET id, without waiting for a document that does not exist yet.
    expect(mocked.close).toHaveBeenCalledWith("aaa");

    open.resolve(docOf("aaa"));
    await opening;

    // The late reply neither resurrects the entry…
    expect(entry("aaa")).toBeUndefined();
    expect(Object.keys(useSkelStore.getState().docs)).toHaveLength(0);
    // …nor leaves the session it created behind: the first close found nothing,
    // so the open's own reply sends another.
    expect(mocked.close).toHaveBeenCalledTimes(2);
  });

  it("drops a late edit reply for a closed document instead of resurrecting it", async () => {
    await useSkelStore.getState().open("aaa");
    const edit = deferred<ReturnType<typeof ok>>();
    mocked.renameJoint.mockReturnValueOnce(edit.promise);

    const editing = useSkelStore.getState().renameJoint("aaa", 0, "spine");
    await useSkelStore.getState().close("aaa");
    expect(entry("aaa")).toBeUndefined();

    edit.resolve(ok(docOf("aaa", 9)));
    await editing;
    expect(entry("aaa")).toBeUndefined();
    expect(Object.keys(useSkelStore.getState().docs)).toHaveLength(0);
  });

  it("re-opening after a close starts fresh rather than being tombstoned forever", async () => {
    await useSkelStore.getState().open("aaa");
    await useSkelStore.getState().close("aaa");
    await useSkelStore.getState().open("aaa");
    expect(entry("aaa").doc?.id).toBe("aaa");
  });
});

describe("adopting a reply", () => {
  beforeEach(async () => {
    await useSkelStore.getState().open("aaa");
    useSkelStore.getState().select("aaa", 3);
  });

  it("clamps a selection the edit put out of range", async () => {
    // A template swap or an undo can shrink the rig under the selected index.
    mocked.instantiateTemplate.mockResolvedValue(ok(docOf("aaa", 2)));
    await useSkelStore.getState().instantiateTemplate("aaa", "quadruped");
    expect(entry("aaa").doc?.joints).toHaveLength(2);
    expect(entry("aaa").selected).toBeNull();
  });

  it("keeps a selection the edit left in range", async () => {
    mocked.setLimit.mockResolvedValue(ok(docOf("aaa", 4, { dirty: true })));
    await useSkelStore.getState().setLimit("aaa", 3, [0, 0, 0], [90, 0, 0]);
    expect(entry("aaa").selected).toBe(3);
  });

  it("shows a rename's warning and clears it on the next clean edit", async () => {
    mocked.renameJoint.mockResolvedValue(
      ok(docOf("aaa"), "`hand_l` lost its twin"),
    );
    await useSkelStore.getState().renameJoint("aaa", 1, "grabber");
    expect(entry("aaa").warning).toBe("`hand_l` lost its twin");
    expect(entry("aaa").refusal).toBeNull();

    mocked.addSocket.mockResolvedValue(ok(docOf("aaa")));
    await useSkelStore.getState().addSocket("aaa", "grip", 1);
    expect(entry("aaa").warning).toBeNull();
  });

  it("keeps the previous warning readable through a refusal", async () => {
    mocked.renameJoint.mockResolvedValue(
      ok(docOf("aaa"), "left the canonical set"),
    );
    await useSkelStore.getState().renameJoint("aaa", 1, "grabber");

    mocked.addSocket.mockResolvedValue(
      refused(docOf("aaa"), 'a socket named "grip" already exists'),
    );
    await useSkelStore.getState().addSocket("aaa", "grip", 1);
    expect(entry("aaa").refusal).toBe('a socket named "grip" already exists');
    expect(entry("aaa").warning).toBe(
      "left the canonical set",
      // The refusal is what to read NOW; the rename's cost is still true and an
      // author who has not read it yet must still be able to.
    );
  });

  it("clears a stale refusal on the next success", async () => {
    mocked.addSocket.mockResolvedValueOnce(
      refused(docOf("aaa"), "no such joint"),
    );
    await useSkelStore.getState().addSocket("aaa", "grip", 99);
    expect(entry("aaa").refusal).toBe("no such joint");
    mocked.addSocket.mockResolvedValueOnce(
      ok(docOf("aaa", 4, { dirty: true })),
    );
    await useSkelStore.getState().addSocket("aaa", "grip", 1);
    expect(entry("aaa").refusal).toBeNull();
  });
});

describe("the doors reach the backend by document id", () => {
  beforeEach(async () => {
    await useSkelStore.getState().open("aaa");
  });

  it("saves, undoes and redoes the document it was asked about", async () => {
    mocked.save.mockResolvedValue(ok(docOf("aaa")));
    mocked.undo.mockResolvedValue(ok(docOf("aaa", 4, { canRedo: true })));
    mocked.redo.mockResolvedValue(ok(docOf("aaa", 4, { canUndo: true })));
    await useSkelStore.getState().save("aaa");
    await useSkelStore.getState().undo("aaa");
    expect(entry("aaa").doc?.canRedo).toBe(true);
    await useSkelStore.getState().redo("aaa");
    expect(entry("aaa").doc?.canUndo).toBe(true);
    for (const fn of [mocked.save, mocked.undo, mocked.redo]) {
      expect(fn).toHaveBeenCalledWith("aaa");
    }
  });

  it("places a socket with an identity orientation and the given offset", async () => {
    mocked.placeSocket.mockResolvedValue(ok(docOf("aaa")));
    await useSkelStore.getState().placeSocket("aaa", "grip", 2, [0.1, 0, 0]);
    expect(mocked.placeSocket).toHaveBeenCalledWith(
      "aaa",
      "grip",
      2,
      [0.1, 0, 0],
      [0, 0, 0, 1],
      [1, 1, 1],
    );
  });

  it("carries a merge's joint offset back as a warning", async () => {
    mocked.mergePart.mockResolvedValue(
      ok(docOf("aaa", 6), "the part's joints start at index 4"),
    );
    await useSkelStore.getState().mergePart("aaa", "part-asset", 1);
    expect(mocked.mergePart).toHaveBeenCalledWith("aaa", "part-asset", 1);
    expect(entry("aaa").warning).toContain("index 4");
  });

  it("fits into a mesh and adopts the fitted rig", async () => {
    mocked.fitToMesh.mockResolvedValue(
      ok(docOf("aaa", 19), "fitted to a 1.80 m mesh"),
    );
    await useSkelStore.getState().fitToMesh("aaa", "mesh-asset", "biped");
    expect(mocked.fitToMesh).toHaveBeenCalledWith("aaa", "mesh-asset", "biped");
    expect(entry("aaa").doc?.joints).toHaveLength(19);
    expect(entry("aaa").warning).toContain("1.80 m");
    expect(entry("aaa").busy).toBe(false);
  });
});
