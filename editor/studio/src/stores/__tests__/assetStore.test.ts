// @vitest-environment jsdom
//
// Asset store — named collections (E-P8). Covers the pure `visibleAssets`
// collection-scoping filter and the store's collection state/actions. IPC is
// mocked; the collection mutations set state from the (mocked) command return.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  assets: {
    snapshot: vi.fn(),
    thumbnail: vi.fn(),
  },
  collections: {
    list: vi.fn(),
    create: vi.fn(),
    rename: vi.fn(),
    delete: vi.fn(),
    add: vi.fn(),
    remove: vi.fn(),
  },
}));

import type { AssetDto } from "../../bindings/AssetDto";
import type { CollectionDto } from "../../bindings/CollectionDto";
import type { ImportEventDto } from "../../bindings/ImportEventDto";
import { assets as assetsIpc, collections as collectionsIpc } from "../../lib/ipc";
import { THUMBNAIL_CAP, useAssetStore, visibleAssets } from "../assetStore";

function asset(id: string, name: string, kind = "mesh", folder = ""): AssetDto {
  return {
    id,
    name,
    kind,
    kind_label: kind,
    folder,
    path: `${folder}/${name}`.replace(/^\//, ""),
    content_hash: "0",
    tags: [],
    source: null,
    dep_count: 0,
    ref_count: 0,
    previewable: false,
  };
}

const A = asset("a", "Apple", "mesh", "props");
const B = asset("b", "Boat", "texture", "env");
const C = asset("c", "Car", "mesh", "props");
const MAP: Record<string, AssetDto> = { a: A, b: B, c: C };

describe("visibleAssets collection scoping", () => {
  it("scopes to the collection id set across folders when active", () => {
    const out = visibleAssets(MAP, "", null, "", ["a", "b"]);
    expect(out.map((x) => x.id).sort()).toEqual(["a", "b"]);
  });

  it("ignores the folder filter in collection mode but keeps kind + search", () => {
    // folder "env" would exclude A/C, but the collection scope wins; kind=mesh
    // then narrows to A within the {a,b} set.
    const out = visibleAssets(MAP, "env", "mesh", "", ["a", "b"]);
    expect(out.map((x) => x.id)).toEqual(["a"]);
  });

  it("falls back to folder scoping when no collection ids are given", () => {
    const out = visibleAssets(MAP, "props", null, "", null);
    expect(out.map((x) => x.id).sort()).toEqual(["a", "c"]);
  });
});

describe("collection state + actions", () => {
  beforeEach(() => {
    useAssetStore.setState({ collections: [], activeCollection: null });
    vi.clearAllMocks();
  });
  afterEach(() => vi.clearAllMocks());

  it("fetchCollections populates the list", async () => {
    const list: CollectionDto[] = [{ name: "Props", ids: ["a"] }];
    vi.mocked(collectionsIpc.list).mockResolvedValue(list);
    await useAssetStore.getState().fetchCollections();
    expect(useAssetStore.getState().collections).toEqual(list);
  });

  it("clears the active collection when it disappears from a refetch", async () => {
    useAssetStore.setState({
      collections: [{ name: "Gone", ids: [] }],
      activeCollection: "Gone",
    });
    vi.mocked(collectionsIpc.list).mockResolvedValue([{ name: "Other", ids: [] }]);
    await useAssetStore.getState().fetchCollections();
    expect(useAssetStore.getState().activeCollection).toBeNull();
  });

  it("setActiveCollection toggles off when re-selecting the same name", () => {
    useAssetStore.getState().setActiveCollection("Props");
    expect(useAssetStore.getState().activeCollection).toBe("Props");
    useAssetStore.getState().setActiveCollection("Props");
    expect(useAssetStore.getState().activeCollection).toBeNull();
  });

  it("addToCollection sets state from the command's returned list", async () => {
    const updated: CollectionDto[] = [{ name: "Props", ids: ["a", "c"] }];
    vi.mocked(collectionsIpc.add).mockResolvedValue(updated);
    await useAssetStore.getState().addToCollection("Props", "c");
    expect(vi.mocked(collectionsIpc.add)).toHaveBeenCalledWith("Props", "c");
    expect(useAssetStore.getState().collections).toEqual(updated);
  });

  it("renameCollection retargets the active filter", async () => {
    useAssetStore.setState({
      collections: [{ name: "Old", ids: [] }],
      activeCollection: "Old",
    });
    vi.mocked(collectionsIpc.rename).mockResolvedValue([{ name: "New", ids: [] }]);
    await useAssetStore.getState().renameCollection("Old", "New");
    expect(useAssetStore.getState().activeCollection).toBe("New");
  });
});

/**
 * **The thumbnail cache** (F-lens L7.H5) — three defects in ten lines of code:
 *
 *  1. **Keyed by asset id.** A re-import keeps the GUID and changes the bytes,
 *     so the "already requested" guard answered yes and the drawer showed the
 *     PREVIOUS picture for the rest of the session. The `content_hash` is the
 *     backend's own thumbnail key (`inf_editor_core::thumbnail::ThumbnailCache`)
 *     and it is what this now keys on too.
 *  2. **Spread-copied on every insert.** `{ ...s.thumbnails, [id]: url }` copies
 *     the whole cache per thumbnail, so filling a grid of N assets was O(N²).
 *  3. **Unbounded.** Base64 PNG data-URLs, one per previewable asset ever
 *     scrolled past, held for the life of the session.
 */
describe("thumbnail cache (L7.H5)", () => {
  // The cap is read from the store, not restated: it MOVED in round 2 (a wide
  // drawer mounts ~300 virtualized cells against a cap of 256, so the later
  // cells evicted the earlier ones while both were on screen and the earlier
  // ones stayed blank for as long as the drawer was open). A test carrying its
  // own copy of a bound is a test that stops describing the bound.
  const CAP = THUMBNAIL_CAP;

  beforeEach(() => {
    useAssetStore.setState({ thumbnails: new Map() });
    vi.clearAllMocks();
    vi.mocked(assetsIpc.thumbnail).mockImplementation((id: string) =>
      Promise.resolve(`data:image/png;base64,${id}`),
    );
  });
  afterEach(() => vi.clearAllMocks());

  const cache = () => useAssetStore.getState().thumbnails;

  it("keys on content_hash, so a re-import re-renders", async () => {
    // THE defect. Same asset id, new bytes → a new key, so the request is made
    // again rather than short-circuited by a stale entry.
    useAssetStore.getState().loadThumbnail("asset-1", "hash-before");
    await vi.waitFor(() => expect(cache().get("hash-before")).toContain("asset-1"));
    expect(vi.mocked(assetsIpc.thumbnail)).toHaveBeenCalledTimes(1);

    useAssetStore.getState().loadThumbnail("asset-1", "hash-after");
    expect(vi.mocked(assetsIpc.thumbnail)).toHaveBeenCalledTimes(2);
    await vi.waitFor(() => expect(cache().get("hash-after")).toContain("asset-1"));
  });

  it("asks once per content, however many assets share it", async () => {
    // The other side of hashing: two identical files are one thumbnail. The
    // second asset is answered from the cache without a round trip.
    useAssetStore.getState().loadThumbnail("asset-1", "same-hash");
    await vi.waitFor(() => expect(cache().get("same-hash")).toBeTruthy());
    useAssetStore.getState().loadThumbnail("asset-2", "same-hash");
    expect(vi.mocked(assetsIpc.thumbnail)).toHaveBeenCalledTimes(1);
  });

  it("does not re-request while the first request is still in flight", async () => {
    // The placeholder "" entry is what holds that shut; it must count as
    // "asked for", not as "absent".
    useAssetStore.getState().loadThumbnail("asset-1", "h");
    useAssetStore.getState().loadThumbnail("asset-1", "h");
    expect(vi.mocked(assetsIpc.thumbnail)).toHaveBeenCalledTimes(1);
    expect(cache().get("h")).toBe("");
  });

  it("is a Map, and is not copied on insert", () => {
    // The O(N²) half: the container is mutated in place and the STATE object is
    // what changes, so subscribers still re-run their selectors.
    const before = cache();
    useAssetStore.getState().loadThumbnail("asset-1", "h1");
    expect(cache()).toBeInstanceOf(Map);
    expect(cache()).toBe(before);
  });

  it("evicts least-recently-used entries past the cap", async () => {
    for (let i = 0; i < CAP + 10; i++) {
      useAssetStore.getState().loadThumbnail(`asset-${i}`, `hash-${i}`);
    }
    await vi.waitFor(() => expect(cache().get(`hash-${CAP + 9}`)).toBeTruthy());

    expect(cache().size).toBe(CAP);
    // The oldest ten are gone; the newest are held.
    expect(cache().has("hash-0")).toBe(false);
    expect(cache().has("hash-9")).toBe(false);
    expect(cache().has("hash-10")).toBe(true);
    expect(cache().has(`hash-${CAP + 9}`)).toBe(true);
  });

  /**
   * **The test-integrity finding**: the load-bearing `set({ thumbnails })` had
   * no arm.
   *
   * `putThumbnail` mutates the `Map` IN PLACE — that is the O(N^2) fix — so
   * every assertion above reads the mutation through `getState()` and passes
   * whether or not zustand was ever told. Delete both `set` calls and all six
   * arms stay green while the Content Drawer's cells never re-render: the
   * thumbnails land in the cache and no subscriber hears about it.
   *
   * The subscriber is the missing half, and it is the whole reason the cache
   * builds a fresh STATE object around the same container.
   */
  it("notifies subscribers, so a mounted cell re-renders when its picture lands", async () => {
    let notifications = 0;
    const unsub = useAssetStore.subscribe(() => {
      notifications += 1;
    });
    try {
      useAssetStore.getState().loadThumbnail("asset-1", "h");
      expect(
        notifications,
        "the pending sentinel went into the map without telling zustand",
      ).toBeGreaterThanOrEqual(1);

      const afterRequest = notifications;
      await vi.waitFor(() => expect(cache().get("h")).toContain("asset-1"));
      expect(
        notifications,
        "the picture arrived and no subscriber was told, so the cell stays blank",
      ).toBeGreaterThan(afterRequest);
    } finally {
      unsub();
    }
  });

  /**
   * **R2.F13's other half**: a refusal must not become the most durable entry
   * in the cache.
   *
   * A `null` answer left the `""` "requested" sentinel in place, and
   * `touchThumbnail` renewed it as most-recently-used on every re-read — so the
   * failed entry outlived real pictures, was never retried, and could never be
   * evicted while the grid kept asking for it.
   */
  it("records a refusal as an answer, not as a permanent pending", async () => {
    vi.mocked(assetsIpc.thumbnail).mockResolvedValue(null);
    useAssetStore.getState().loadThumbnail("asset-1", "h");
    await vi.waitFor(() => expect(cache().get("h")).toBeNull());

    // It is an ANSWER: asking again costs no round trip...
    useAssetStore.getState().loadThumbnail("asset-1", "h");
    expect(vi.mocked(assetsIpc.thumbnail)).toHaveBeenCalledTimes(1);
    // ...and it is a normal LRU entry, so it ages out like any other.
    for (let i = 0; i < CAP; i++) {
      useAssetStore.getState().loadThumbnail(`asset-${i}`, `hash-${i}`);
    }
    expect(cache().has("h")).toBe(false);
  });

  it("lets a thrown request be retried", async () => {
    // An error is not an answer. Leaving the sentinel behind made a transient
    // failure permanent for the session.
    vi.mocked(assetsIpc.thumbnail).mockRejectedValueOnce(new Error("busy"));
    useAssetStore.getState().loadThumbnail("asset-1", "h");
    await vi.waitFor(() => expect(cache().has("h")).toBe(false));

    vi.mocked(assetsIpc.thumbnail).mockResolvedValue("data:image/png;base64,ok");
    useAssetStore.getState().loadThumbnail("asset-1", "h");
    await vi.waitFor(() => expect(cache().get("h")).toContain("ok"));
    expect(vi.mocked(assetsIpc.thumbnail)).toHaveBeenCalledTimes(2);
  });

  it("a re-read renews an entry, so it survives the next eviction", async () => {
    // What makes it an LRU rather than a FIFO: the entry still being asked for
    // is the one that must not be dropped.
    useAssetStore.getState().loadThumbnail("asset-0", "hash-0");
    await vi.waitFor(() => expect(cache().get("hash-0")).toBeTruthy());
    for (let i = 1; i < CAP; i++) {
      useAssetStore.getState().loadThumbnail(`asset-${i}`, `hash-${i}`);
    }
    // Touch the oldest entry — the grid scrolling back over it.
    useAssetStore.getState().loadThumbnail("asset-0", "hash-0");
    // …then push one more in, which evicts whatever is now oldest.
    useAssetStore.getState().loadThumbnail("asset-new", "hash-new");

    expect(cache().size).toBe(CAP);
    expect(cache().has("hash-0")).toBe(true);
    expect(cache().has("hash-1")).toBe(false);
  });
});

/**
 * **The import-job ring** (F-lens L7.M11). `importAdvisories` next door was
 * already capped at 20; `imports` kept every job the backend ever reported, for
 * the life of the session, in a record that is spread-copied on every event.
 */
describe("import job pruning (L7.M11)", () => {
  const IMPORT_CAP = 50;

  function event(job: number, phase: string): ImportEventDto {
    return {
      job,
      source: `file-${job}.png`,
      phase,
      produced: [],
      primary: null,
      cached: false,
      error: null,
      done: null,
      total: null,
      stage: null,
      advisories: [],
    };
  }

  beforeEach(() => {
    useAssetStore.setState({ imports: {}, importAdvisories: [] });
    vi.clearAllMocks();
    vi.mocked(assetsIpc.snapshot).mockResolvedValue({
      assets: [],
      folders: [],
      root: "",
      version: 0n as unknown as bigint,
    } as never);
  });
  afterEach(() => vi.clearAllMocks());

  const jobs = () => useAssetStore.getState().imports;

  it("keeps at most the newest cap of finished jobs", () => {
    for (let i = 0; i < IMPORT_CAP + 25; i++) {
      useAssetStore.getState().applyImportEvent(event(i, "finished"));
    }
    expect(Object.keys(jobs())).toHaveLength(IMPORT_CAP);
    expect(jobs()[0]).toBeUndefined();
    expect(jobs()[IMPORT_CAP + 24]).toBeDefined();
  });

  it("never evicts a job that is still running", () => {
    // The progress strip is the only place a long terrain import reports
    // itself; pruning it mid-import would make the bar vanish.
    useAssetStore.getState().applyImportEvent(event(0, "progress"));
    for (let i = 1; i < IMPORT_CAP + 25; i++) {
      useAssetStore.getState().applyImportEvent(event(i, "finished"));
    }
    expect(jobs()[0]?.phase).toBe("progress");
    expect(Object.keys(jobs())).toHaveLength(IMPORT_CAP + 1);
  });

  it("leaves a small set completely alone", () => {
    for (let i = 0; i < 5; i++) {
      useAssetStore.getState().applyImportEvent(event(i, "finished"));
    }
    expect(Object.keys(jobs())).toHaveLength(5);
  });
});
