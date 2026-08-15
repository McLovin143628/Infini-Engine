// @vitest-environment jsdom
//
// **The store bridge's diff-sync** (F-lens L7.H4).
//
// The bridge used to ship every bridged store's WHOLE data slice on every
// mutation. Its own comment called that "cheap for today's small stores" and had
// gone stale on both examples it named: `logStore.lines` is a 5000-entry ring
// flushed ~16×/s, and `sceneStore.nodes` is the full scene node map rebuilt on
// every `world://delta`.
//
// The property that has to hold for a patch protocol to be safe is a round trip:
// applying `diffSlice(prev, next)` to `prev` must reconstruct `next` exactly,
// for every shape either of them can take — including the ones that fall back to
// a whole-value send. Every case below asserts that, and then asserts the SIZE
// claim separately, because a diff that is correct but ships everything anyway
// fixes nothing.
import { describe, expect, it, vi } from "vitest";

// The bridge emits a Tauri event when a window opens; there is no Tauri here.
vi.mock("@tauri-apps/api/event", () => ({
  emit: vi.fn(async () => {}),
  listen: vi.fn(async () => () => {}),
}));

import {
  applySlicePatch,
  diffSlice,
  registerBridgedStore,
  setPanelWindowCount,
  __bridgeBaselineCount,
} from "../window/storeBridge";

/** Apply a patch the way the mirror does — `setState`'s shallow merge. */
function roundTrip(
  prev: Record<string, unknown>,
  next: Record<string, unknown>,
): Record<string, unknown> {
  const patch = diffSlice(prev, next);
  if (!patch) return prev;
  return { ...prev, ...applySlicePatch(prev, patch) };
}

/** Stand-ins for `LogLine`s: identity is what the array delta matches on. */
const line = (seq: number) => ({ seq, message: `m${seq}` });

describe("array slices — the capped log ring", () => {
  it("ships only the appended lines", () => {
    const a = [line(1), line(2), line(3)];
    const next = [...a, line(4), line(5)];
    const patch = diffSlice({ lines: a }, { lines: next })!;
    expect(patch.lines).toEqual({ t: "arr", trim: 0, append: [next[3], next[4]] });
    expect(roundTrip({ lines: a }, { lines: next })).toEqual({ lines: next });
  });

  it("ships an append + head-trim as a count and a tail", () => {
    // What `appendMany` does once the ring is full: merge, then trim the front
    // back to LOG_CAP. Both ends move at once.
    const a = [line(1), line(2), line(3), line(4)];
    const next = [a[2]!, a[3]!, line(5), line(6)];
    const patch = diffSlice({ lines: a }, { lines: next })!;
    expect(patch.lines).toEqual({ t: "arr", trim: 2, append: [next[2], next[3]] });
    expect(roundTrip({ lines: a }, { lines: next })).toEqual({ lines: next });
  });

  it("is a handful of entries, not five thousand", () => {
    // The headline claim, as an assertion rather than a comment.
    const full = Array.from({ length: 5000 }, (_, i) => line(i));
    const next = [...full.slice(3), line(5000), line(5001), line(5002)];
    const patch = diffSlice({ lines: full }, { lines: next })!;
    const arr = patch.lines as { t: "arr"; trim: number; append: unknown[] };
    expect(arr.t).toBe("arr");
    expect(arr.append).toHaveLength(3);
    expect(arr.trim).toBe(3);
    expect(roundTrip({ lines: full }, { lines: next })).toEqual({ lines: next });
  });

  it("falls back to the whole array when it is not an append (a clear)", () => {
    const a = [line(1), line(2)];
    expect(diffSlice({ lines: a }, { lines: [] })!.lines).toEqual({ t: "set", v: [] });
    expect(roundTrip({ lines: a }, { lines: [] })).toEqual({ lines: [] });
  });

  it("falls back when the array was replaced wholesale", () => {
    // A re-seeded list shares no references, so there is no run to find. It
    // MUST come through as a full value rather than a bad overlap.
    const a = [line(1), line(2), line(3)];
    const next = [line(1), line(2), line(3)]; // equal by value, different objects
    const patch = diffSlice({ lines: a }, { lines: next })!;
    expect(patch.lines).toEqual({ t: "set", v: next });
    expect(roundTrip({ lines: a }, { lines: next })).toEqual({ lines: next });
  });

  it("rejects an overlap whose middle does not line up", () => {
    // The reason the whole run is verified element by element rather than at
    // its ends: the last element lines up at the right offset here, and the run
    // still is not a run. Trusting the endpoint alone would emit a patch that
    // silently reconstructs the WRONG array in the mirror window.
    const x = line(1);
    const y = line(2);
    const z = line(3);
    const w = line(4);
    const a = [x, y, z];
    const next = [y, w, z]; // trim = 0 and the tails match; a[0] !== next[0]
    const patch = diffSlice({ lines: a }, { lines: next })!;
    expect(patch.lines).toEqual({ t: "set", v: next });
    expect(roundTrip({ lines: a }, { lines: next })).toEqual({ lines: next });
  });

  it("rejects an overlap that would need a negative trim", () => {
    const x = line(1);
    const y = line(2);
    const z = line(3);
    const a = [x, y, z];
    const next = [x, z, y, z]; // the last element reappears further along
    expect(diffSlice({ lines: a }, { lines: next })!.lines).toEqual({ t: "set", v: next });
    expect(roundTrip({ lines: a }, { lines: next })).toEqual({ lines: next });
  });
});

describe("record slices — the scene node map", () => {
  const node = (guid: string, x: number) => ({ guid, x });

  it("ships only the nodes a delta touched", () => {
    const prev = { a: node("a", 0), b: node("b", 0), c: node("c", 0), d: node("d", 0) };
    const next = { ...prev, b: node("b", 5) };
    const patch = diffSlice({ nodes: prev }, { nodes: next })!;
    expect(patch.nodes).toEqual({ t: "rec", set: { b: next.b }, del: [] });
    expect(roundTrip({ nodes: prev }, { nodes: next })).toEqual({ nodes: next });
  });

  it("carries removals, so a deleted actor really leaves the mirror", () => {
    const prev = { a: node("a", 0), b: node("b", 0), c: node("c", 0), d: node("d", 0) };
    const next = { a: prev.a, b: prev.b, c: prev.c };
    const patch = diffSlice({ nodes: prev }, { nodes: next })!;
    expect(patch.nodes).toEqual({ t: "rec", set: {}, del: ["d"] });
    expect(roundTrip({ nodes: prev }, { nodes: next })).toEqual({ nodes: next });
  });

  it("sends the map whole when nearly all of it changed", () => {
    // A patch bigger than the value is a slower full send. Loading a level is
    // this case, and it must not pay twice.
    const prev = { a: node("a", 0), b: node("b", 0) };
    const next = { a: node("a", 1), b: node("b", 1), c: node("c", 1) };
    expect(diffSlice({ nodes: prev }, { nodes: next })!.nodes).toEqual({ t: "set", v: next });
    expect(roundTrip({ nodes: prev }, { nodes: next })).toEqual({ nodes: next });
  });
});

describe("the slice as a whole", () => {
  it("says nothing changed when nothing changed", () => {
    // A subscriber can fire without the DATA moving; the bridge then sends
    // nothing at all and the sequence stands still.
    const lines = [line(1)];
    const slice = { lines, paused: false, search: "" };
    expect(diffSlice(slice, { ...slice })).toBeNull();
  });

  it("leaves untouched keys out of the patch entirely", () => {
    // The other half of the log finding: a flush used to re-ship `enabled`,
    // `search` and `paused` alongside the 5000 lines.
    const a = [line(1)];
    const enabled = { info: true };
    const prev = { lines: a, enabled, search: "", paused: false };
    const next = { lines: [...a, line(2)], enabled, search: "", paused: false };
    const patch = diffSlice(prev, next)!;
    expect(Object.keys(patch)).toEqual(["lines"]);
    expect(roundTrip(prev, next)).toEqual(next);
  });

  it("handles scalars, nulls and type changes", () => {
    const prev = { a: 1, b: "x", c: null as unknown, d: [1, 2] as unknown };
    const next = { a: 2, b: "y", c: { deep: true } as unknown, d: null as unknown };
    const patch = diffSlice(prev, next)!;
    expect(patch.a).toEqual({ t: "set", v: 2 });
    expect(patch.c).toEqual({ t: "set", v: next.c });
    expect(patch.d).toEqual({ t: "set", v: null });
    expect(roundTrip(prev, next)).toEqual(next);
  });

  it("survives a JSON round trip, because the patch crosses an event", () => {
    // The patch is serialized by Tauri. A shape that only works in-process
    // would pass every test above and fail in the product.
    const a = [line(1), line(2)];
    const prev = { lines: a, nodes: { x: { v: 1 }, y: { v: 2 } } };
    const next = { lines: [a[1]!, line(3)], nodes: { x: { v: 1 }, y: { v: 9 } } };
    const wire = JSON.parse(JSON.stringify(diffSlice(prev, next))) as ReturnType<
      typeof diffSlice
    >;
    expect({ ...prev, ...applySlicePatch(prev, wire!) }).toEqual(next);
  });
});

/**
 * **The baseline the host keeps holding after the last window closes**
 * (round-2 finding R2-4).
 *
 * `lastSent` is one whole DATA SLICE per bridged store — the 5 000-line log
 * ring, the full scene node map — kept as the baseline patches are computed
 * against. With no panel window open nothing is emitted and the baseline is
 * stale on purpose, so it is not merely useless: it is a pinned copy of the
 * largest objects in the editor, held for the rest of the session by a feature
 * nobody is using.
 *
 * The clear is safe because `setPanelWindowCount(n > 0)` re-bases from live
 * state, and `flush` treats a missing baseline as a full send — which is the
 * correct answer for a mirror that holds nothing.
 */
describe("the host's patch baseline", () => {
  /** A zustand-shaped store holding one big-ish slice. */
  function fakeStore(lines: string[]) {
    const state = { lines, act: () => {} };
    return {
      getState: () => state,
      subscribe: () => () => {},
    };
  }

  it("is dropped when the last panel window closes", () => {
    registerBridgedStore("r2-4-log", fakeStore(["a", "b", "c"]));
    setPanelWindowCount(1); // a window opens -> every store is re-based
    expect(__bridgeBaselineCount()).toBeGreaterThan(0);

    setPanelWindowCount(0);
    expect(
      __bridgeBaselineCount(),
      "the host is still pinning a full slice per store with nothing to send it to",
    ).toBe(0);
  });

  it("is re-based the moment a window opens again", () => {
    // The other direction: dropping it must not leave the next window syncing
    // against nothing. `emitAllSnapshots` re-bases from LIVE state, which is
    // also why the stale copy was worth nothing.
    registerBridgedStore("r2-4-log", fakeStore(["a"]));
    setPanelWindowCount(0);
    expect(__bridgeBaselineCount()).toBe(0);
    setPanelWindowCount(1);
    expect(__bridgeBaselineCount()).toBeGreaterThan(0);
    setPanelWindowCount(0);
  });
});
