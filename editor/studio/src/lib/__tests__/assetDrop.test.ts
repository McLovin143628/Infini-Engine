// @vitest-environment jsdom
//
// **The Content Drawer → panel drop door** (round-2 finding R2.F9).
//
// The Model Editor's merge was reachable only from an HTML5 `dataTransfer`
// payload that **nothing in the tree ever set**: the zone highlighted for any
// native drag and then silently discarded it, so `dcc_merge_asset` had no
// caller and "drag-and-drop modular model building" — the DCC vision's
// headline — was a dead door.
//
// It cannot be fixed by making the drawer's cells `draggable`: starting an
// HTML5 drag cancels the pointer stream, and the pointer stream is the only way
// to reach the *native viewport child window*, which is not a DOM node at all.
// So the drawer hit-tests DOM zones on pointer-up exactly as it already
// hit-tests the viewport hole, and both ends of that contract live in
// `assetDrop.ts` — which is what makes it checkable here.
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  ASSET_DROP_ATTR,
  deliverAssetDrop,
  onAssetDrop,
  type AssetDropDetail,
} from "../assetDrop";

const MESH: AssetDropDetail = { id: "guid-1", kind: "mesh", name: "Bolt" };

// jsdom does not implement `elementFromPoint` (it has no layout), so the hit
// test is driven directly: what is under the cursor is whatever this returns.
let under: Element | null = null;
beforeEach(() => {
  Object.defineProperty(document, "elementFromPoint", {
    configurable: true,
    writable: true,
    value: () => under,
  });
  under = null;
});

/** A zone at a known point, with `elementFromPoint` pointed at `inner`. */
function zoneAt(accept: string): { zone: HTMLElement; inner: HTMLElement } {
  const zone = document.createElement("div");
  zone.setAttribute(ASSET_DROP_ATTR, accept);
  // Drops land on whatever is painted inside the zone (the preview image), so
  // the hit test has to walk UP to the zone — `closest`, not the target itself.
  const inner = document.createElement("img");
  zone.appendChild(inner);
  document.body.appendChild(zone);
  under = inner;
  return { zone, inner };
}

afterEach(() => {
  document.body.innerHTML = "";
  under = null;
});

describe("deliverAssetDrop", () => {
  it("delivers to the zone under the pointer, through the painted child", () => {
    const { zone } = zoneAt("mesh");
    const seen: AssetDropDetail[] = [];
    const off = onAssetDrop(zone, (d) => seen.push(d));

    expect(deliverAssetDrop(10, 10, MESH)).toBe(true);
    expect(seen).toEqual([MESH]);
    off();
  });

  it("answers false when nothing under the pointer is a zone", () => {
    // The load-bearing half: `false` is what makes the drawer fall through to
    // the native viewport hole, which is the drag that already worked.
    const loose = document.createElement("div");
    document.body.appendChild(loose);
    under = loose;

    expect(deliverAssetDrop(10, 10, MESH)).toBe(false);
  });

  it("answers false over empty space", () => {
    under = null;
    expect(deliverAssetDrop(10, 10, MESH)).toBe(false);
  });

  it("stops delivering once the subscriber disposes", () => {
    // A panel that unmounts must not keep receiving drops — the same class as
    // every listener leak this campaign closed.
    const { zone } = zoneAt("mesh");
    const seen: AssetDropDetail[] = [];
    onAssetDrop(zone, (d) => seen.push(d))();

    expect(deliverAssetDrop(10, 10, MESH)).toBe(true); // the zone is still there
    expect(seen, "a disposed panel received a drop").toEqual([]);
  });

  it("carries the kind, so a zone can refuse what it cannot take", () => {
    const { zone } = zoneAt("mesh");
    const taken: string[] = [];
    onAssetDrop(zone, (d) => {
      if (d.kind === "mesh") taken.push(d.id);
    });

    deliverAssetDrop(10, 10, { id: "guid-2", kind: "texture", name: "Rust" });
    deliverAssetDrop(10, 10, MESH);
    expect(taken).toEqual(["guid-1"]);
  });
});
