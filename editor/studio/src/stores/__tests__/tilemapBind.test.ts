// @vitest-environment jsdom
//
// **The tilemap bind race** (round-2 finding R2.F12).
//
// `bind` set `entity` synchronously, awaited `tilemap_get`, and applied the
// reply without re-reading `get().entity`. Select tilemap A then B quickly — or
// undo/redo, or StrictMode's double mount — and if A's reply lands last the
// store holds `entity: B` with **A's `cellMap`**, and `PaintCanvas.commit` then
// sends those cells to B. One entity's painting written into another's.
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  tilemap: {
    get: vi.fn(),
    paint: vi.fn(),
  },
}));

import { tilemap } from "../../lib/ipc";
import type { TilemapDto } from "../../bindings/TilemapDto";
import { __resetTilemapBindForTest, useTilemapStore } from "../tilemapStore";

function dto(entity: string, tile: number): TilemapDto {
  return {
    entity,
    tile_size: [1, 1],
    palette_cols: 4,
    palette_rows: 4,
    palette_texture: null,
    sorting_layer: 0,
    order_in_layer: 0,
    cells: [{ x: 0, y: 0, tile }],
  } as unknown as TilemapDto;
}

/** A promise plus the trigger that settles it — so replies can be reordered. */
function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

beforeEach(() => {
  vi.clearAllMocks();
  __resetTilemapBindForTest();
  useTilemapStore.setState({
    entity: null,
    dto: null,
    cellMap: new Map(),
    busy: false,
    error: null,
  });
});

describe("bind", () => {
  it("drops a stale reply rather than merging it into the new entity", async () => {
    // A's fetch is slow; B's is instant. The author clicked A then B.
    const slowA = deferred<TilemapDto>();
    vi.mocked(tilemap.get)
      .mockReturnValueOnce(slowA.promise)
      .mockResolvedValueOnce(dto("B", 7));

    const bindA = useTilemapStore.getState().bind("A");
    const bindB = useTilemapStore.getState().bind("B");
    await bindB;

    expect(useTilemapStore.getState().entity).toBe("B");
    expect(useTilemapStore.getState().cellMap.get("0,0")).toBe(7);

    // …and now A's reply arrives.
    slowA.resolve(dto("A", 3));
    await bindA;

    expect(useTilemapStore.getState().entity).toBe("B");
    expect(
      useTilemapStore.getState().cellMap.get("0,0"),
      "A's cells were written under B's entity — PaintCanvas.commit would send them to B",
    ).toBe(7);
  });

  it("drops a stale FAILURE too", async () => {
    // Otherwise the new selection inherits an error about the old one and the
    // panel refuses to paint a tilemap that is perfectly fine.
    const slowA = deferred<TilemapDto>();
    vi.mocked(tilemap.get).mockReturnValueOnce(
      slowA.promise.then(() => {
        throw new Error("A is gone");
      }),
    );
    vi.mocked(tilemap.get).mockResolvedValueOnce(dto("B", 7));

    const bindA = useTilemapStore.getState().bind("A");
    await useTilemapStore.getState().bind("B");
    slowA.resolve(dto("A", 3));
    await bindA;

    expect(useTilemapStore.getState().error).toBeNull();
    expect(useTilemapStore.getState().busy).toBe(false);
  });

  it("still applies the reply it is actually waiting for", async () => {
    // The other direction: a guard that dropped everything would be a
    // correct-looking store that never paints.
    vi.mocked(tilemap.get).mockResolvedValue(dto("A", 5));
    await useTilemapStore.getState().bind("A");
    expect(useTilemapStore.getState().entity).toBe("A");
    expect(useTilemapStore.getState().cellMap.get("0,0")).toBe(5);
    expect(useTilemapStore.getState().busy).toBe(false);
  });
});

describe("refresh", () => {
  it("does not write an old entity's cells after the selection moved", async () => {
    vi.mocked(tilemap.get).mockResolvedValue(dto("A", 5));
    await useTilemapStore.getState().bind("A");

    const slow = deferred<TilemapDto>();
    vi.mocked(tilemap.get).mockReturnValueOnce(slow.promise);
    const refreshing = useTilemapStore.getState().refresh();

    vi.mocked(tilemap.get).mockResolvedValueOnce(dto("B", 9));
    await useTilemapStore.getState().bind("B");

    slow.resolve(dto("A", 3));
    await refreshing;

    expect(useTilemapStore.getState().entity).toBe("B");
    expect(useTilemapStore.getState().cellMap.get("0,0")).toBe(9);
  });
});
