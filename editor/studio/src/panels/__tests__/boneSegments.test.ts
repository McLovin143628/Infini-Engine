// @vitest-environment jsdom
//
// **A malformed rig must not take the Skeleton Editor down** (round-2 finding
// R2.F-med-extra).
//
// The bone renderer read `placed[p.joint.parent].x` with no guard, while
// `projectRig` defends the identical lookup 100 lines above under a comment
// saying a malformed rig "must not crash the panel". A `.inf_skel` naming a
// parent past its own joint list therefore threw on `undefined.x` and unmounted
// the whole editor.
//
// The file reaches here unvalidated, which is the load-bearing part: Wave H
// found the Rust half of exactly this — serde's derive reconstructs a
// `Skeleton` straight from the wire, so `Skeleton::new`'s validation never runs
// on the decode path, and a parent past the joint list panicked
// `pose::global_transforms` in the editor, in PIE and in the shipped player.
import { describe, expect, it } from "vitest";

import { boneSegments, type Placed } from "../skeleton/SkeletonEditor";

function placed(index: number, parent: number | null, x: number, y: number): Placed {
  return {
    joint: {
      index,
      parent,
      name: `j${index}`,
      translation: [0, 0, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
      mirror: null,
    } as unknown as Placed["joint"],
    world: [0, 0, 0],
    x,
    y,
  };
}

describe("boneSegments", () => {
  it("draws one segment per parented joint", () => {
    const rig = [placed(0, null, 10, 10), placed(1, 0, 20, 30), placed(2, 1, 40, 50)];
    const bones = boneSegments(rig);
    expect(bones).toHaveLength(2);
    expect(bones[0]).toEqual({ key: "b1", x1: 10, y1: 10, x2: 20, y2: 30 });
    expect(bones[1]).toEqual({ key: "b2", x1: 20, y1: 30, x2: 40, y2: 50 });
  });

  it("draws nothing for a root", () => {
    expect(boneSegments([placed(0, null, 1, 2)])).toEqual([]);
  });

  it("skips a parent index outside the rig instead of throwing", () => {
    // THE defect. Before the guard this threw on `undefined.x` and the panel
    // unmounted; a rig with one bad joint must still draw its good ones.
    const rig = [placed(0, null, 10, 10), placed(1, 99, 20, 30), placed(2, 0, 40, 50)];
    expect(() => boneSegments(rig)).not.toThrow();
    const bones = boneSegments(rig);
    expect(bones).toHaveLength(1);
    expect(bones[0].key).toBe("b2");
  });

  it("skips a negative parent index too", () => {
    // `-1` is the other spelling of "no parent" in a lot of rig formats, and it
    // is not the one this DTO uses — so it must be refused, not indexed with.
    const rig = [placed(0, null, 10, 10), placed(1, -1, 20, 30)];
    expect(boneSegments(rig)).toEqual([]);
  });
});
