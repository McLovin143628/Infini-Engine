import { describe, expect, it } from "vitest";

import type { SeqKeyDto } from "../../bindings/SeqKeyDto";
import type { SeqTrackDto } from "../../bindings/SeqTrackDto";
import { advancePlayhead, retimeKeys, sampleTrackAt, snapTime } from "../sequencerStore";

const key = (t: number, value: number, interp: "step" | "linear" = "linear"): SeqKeyDto => ({
  t,
  value,
  interp,
});

const track = (keys: SeqKeyDto[]): SeqTrackDto => ({
  target: "00000000-0000-0000-0000-000000000000",
  path: "Transform.translation.x",
  keys,
});

describe("sampleTrackAt (mirror of the Rust sampler)", () => {
  it("holds at the endpoints", () => {
    const tr = track([key(1, 10), key(3, 30)]);
    expect(sampleTrackAt(tr, 0)).toBe(10); // before first
    expect(sampleTrackAt(tr, 1)).toBe(10);
    expect(sampleTrackAt(tr, 3)).toBe(30);
    expect(sampleTrackAt(tr, 5)).toBe(30); // after last
  });

  it("linearly interpolates within a segment", () => {
    const tr = track([key(0, 0), key(2, 20)]);
    expect(sampleTrackAt(tr, 1)).toBe(10);
    expect(sampleTrackAt(tr, 0.5)).toBe(5);
  });

  it("step holds the left value until the next key", () => {
    const tr = track([key(0, 0, "step"), key(2, 20, "step")]);
    expect(sampleTrackAt(tr, 1.9)).toBe(0);
    expect(sampleTrackAt(tr, 2)).toBe(20);
  });

  it("returns null for an empty track", () => {
    expect(sampleTrackAt(track([]), 1)).toBeNull();
  });
});

describe("snapTime (retime rounding)", () => {
  it("rounds to the nearest 1/fps frame", () => {
    expect(snapTime(30, 0.05)).toBeCloseTo(2 / 30, 9); // 0.05 → frame 2 (0.0667)
    expect(snapTime(30, 0.0)).toBe(0);
    expect(snapTime(10, 0.44)).toBeCloseTo(0.4, 9);
  });

  it("no grid when fps <= 0", () => {
    expect(snapTime(0, 0.123)).toBe(0.123);
  });
});

describe("retimeKeys", () => {
  it("moves a key to the snapped time and re-sorts", () => {
    const keys = [key(0, 0), key(1, 1), key(2, 2)];
    // Move key 0 (t=0, value 0) to 1.48 → snaps to frame 44 @ 30fps = 44/30.
    const out = retimeKeys(keys, 0, 1.48, 30);
    const times = out.map((k) => k.t);
    expect(times).toEqual([...times].sort((a, b) => a - b)); // stays sorted
    expect(out.some((k) => Math.abs(k.t - 44 / 30) < 1e-9 && k.value === 0)).toBe(true);
  });

  it("drops a key that collides with the moved key's snapped time", () => {
    const keys = [key(0, 0), key(1, 1)];
    // Move key 0 onto t=1 (already occupied) → the old t=1 key is dropped.
    const out = retimeKeys(keys, 0, 1.0, 30);
    expect(out).toHaveLength(1);
    expect(out[0].t).toBe(1);
    expect(out[0].value).toBe(0); // the moved key survives
  });

  it("clamps negative times to 0", () => {
    const out = retimeKeys([key(1, 5)], 0, -3, 30);
    expect(out[0].t).toBe(0);
  });
});

describe("advancePlayhead (Play state machine)", () => {
  it("advances by dt while inside the duration", () => {
    expect(advancePlayhead(0, 0.1, 5)).toEqual({ t: 0.1, done: false });
  });

  it("clamps and signals done at the end", () => {
    expect(advancePlayhead(4.95, 0.1, 5)).toEqual({ t: 5, done: true });
    expect(advancePlayhead(5, 0.1, 5)).toEqual({ t: 5, done: true });
  });
});
