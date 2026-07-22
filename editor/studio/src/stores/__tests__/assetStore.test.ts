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
import { collections as collectionsIpc } from "../../lib/ipc";
import { useAssetStore, visibleAssets } from "../assetStore";

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
