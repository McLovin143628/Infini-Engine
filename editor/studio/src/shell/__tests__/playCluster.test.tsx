// @vitest-environment jsdom
//
// **The Play split-button dropdown crosses the viewport hole** (UX2 audit).
//
// Wave UX2 found that this dropdown has no airspace guard at all and recorded it
// as out of scope, because "it only exists while PIE is running" — and while a
// foreign player window occupies the hole, hiding our own child would not
// uncover anything. The reasoning is sound; the premise is false. The chevron is
// rendered in the `!running` branch as well, where the hole belongs to our own
// child window and it draws OVER the dropdown: a menu whose lower half is
// invisible and whose clicks land in the 3D view. That is the failure the whole
// wave exists to remove, one component away from the ones it fixed.
//
// The arms below assert at the same seam the rest of UX2 does — which command
// crossed, carrying what — and the first one is written so that BOTH regressions
// fail it: no guard at all pushes nothing, and the pre-UX2 full-hide guard
// pushes `hide`.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import { act } from "react-dom/test-utils";

import { PlayCluster } from "../MainToolbar";
import {
  CUTOUT_ATTR,
  __resetViewportOverlayForTest,
  __setCutoutSupportedForTest,
} from "../../lib/viewportOverlay";
import { usePieStore } from "../../stores/pieStore";
import { useSimStore } from "../../stores/simStore";

let container: HTMLDivElement;
let root: Root;
let log: string[];

beforeEach(() => {
  __resetViewportOverlayForTest();
  usePieStore.setState({ running: false, paused: false });
  useSimStore.setState({ running: false, paused: false });
  log = [];
  mockIPC((cmd, args) => {
    if (cmd === "viewport_set_visible") {
      log.push((args as { visible: boolean }).visible ? "show" : "hide");
    }
    if (cmd === "viewport_set_region") {
      const rects = (
        args as { rects: Array<{ x: number; y: number; width: number; height: number }> }
      ).rects;
      log.push(
        rects.length === 0
          ? "region:none"
          : `region:${rects.map((r) => `${r.x},${r.y},${r.width},${r.height}`).join(";")}`,
      );
    }
  });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  clearMocks();
  __resetViewportOverlayForTest();
});

/** `setVisible`/`setRegion` are fire-and-forget; let their promises settle. */
const settle = () => new Promise((r) => setTimeout(r, 0));

/** jsdom lays nothing out, so the marked panel needs a size to be measurable. */
function sizeTheDropdown(): () => void {
  const original = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function (this: Element) {
    if (!this.hasAttribute(CUTOUT_ATTR)) return original.call(this);
    return {
      x: 900,
      y: 36,
      left: 900,
      top: 36,
      width: 176,
      height: 104,
      right: 1076,
      bottom: 140,
      toJSON: () => ({}),
    } as DOMRect;
  };
  return () => {
    Element.prototype.getBoundingClientRect = original;
  };
}

const click = (el: Element) =>
  act(() => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

describe("the Play options dropdown", () => {
  it("exists with no session running — the premise UX2 got backwards", () => {
    act(() => root.render(<PlayCluster />));
    expect(usePieStore.getState().running).toBe(false);
    // If this ever becomes PIE-only, the wave's carried sentence becomes true
    // and this file — and the ledger entry it belongs to — must say so.
    expect(container.querySelector('[aria-label="Play options"]')).not.toBeNull();
  });

  it("carves itself out of the viewport instead of being drawn under it", async () => {
    const restore = sizeTheDropdown();
    try {
      __setCutoutSupportedForTest(true);
      await settle();
      act(() => root.render(<PlayCluster />));
      await settle();
      log.length = 0;

      const chevron = container.querySelector('[aria-label="Play options"]')!;
      click(chevron);
      await settle();
      // Nothing at all is what "no guard" looks like; `hide` is what the
      // pre-UX2 guard looks like. Neither passes.
      expect(log).toEqual(["region:900,36,176,104"]);

      click(chevron);
      await settle();
      expect(log).toEqual(["region:900,36,176,104", "region:none"]);
    } finally {
      restore();
    }
  });

  it("falls back to the full hide where there is no cutout backend", async () => {
    // macOS and Linux: the dropdown must still get the viewport out of its way,
    // by the only means those platforms have.
    const restore = sizeTheDropdown();
    try {
      __setCutoutSupportedForTest(false);
      await settle();
      act(() => root.render(<PlayCluster />));
      await settle();
      log.length = 0;

      click(container.querySelector('[aria-label="Play options"]')!);
      await settle();
      expect(log).toEqual(["hide"]);
    } finally {
      restore();
    }
  });
});
