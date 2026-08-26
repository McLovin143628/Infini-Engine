// @vitest-environment jsdom
//
// **The fly-out submenu, which nothing else notices** (UX2 audit).
//
// A menu-bar submenu is positioned OUTSIDE its dropdown's box, so its rectangle
// is not contained in the dropdown's and has to be measured separately — and it
// opens on a CHILD component's state (`MenuList`'s `openSub`), so `MenuBar`
// never re-renders and the hook's per-render re-measure never runs. The
// `MutationObserver` under the root is the only thing that sees it. The wave
// claimed that in three places and pinned it nowhere; a submenu whose rectangle
// never crossed would hang half-visible over the 3D view, which is the failure
// the whole wave is about.
//
// Asserted at the same seam as the rest of UX2: which command crossed, carrying
// what.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import { act } from "react-dom/test-utils";

import MenuBar from "../MenuBar";
import {
  CUTOUT_ATTR,
  __resetViewportOverlayForTest,
  __setCutoutSupportedForTest,
} from "../../lib/viewportOverlay";

let container: HTMLDivElement;
let root: Root;
let log: string[];

beforeEach(() => {
  __resetViewportOverlayForTest();
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

const settle = () => new Promise((r) => setTimeout(r, 0));

/**
 * jsdom lays nothing out. Give every marked element a size, and a DIFFERENT one
 * for a marked element inside another — which is exactly what a fly-out is, and
 * the only thing that makes "both rectangles crossed" a real assertion.
 */
function sizeTheMenus(): () => void {
  const original = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function (this: Element) {
    if (!this.hasAttribute(CUTOUT_ATTR)) return original.call(this);
    const nested = this.parentElement?.closest(`[${CUTOUT_ATTR}]`) != null;
    const [x, y, width, height] = nested ? [320, 60, 200, 150] : [100, 24, 220, 300];
    return {
      x,
      y,
      left: x,
      top: y,
      width,
      height,
      right: x + width,
      bottom: y + height,
      toJSON: () => ({}),
    } as DOMRect;
  };
  return () => {
    Element.prototype.getBoundingClientRect = original;
  };
}

const fire = (el: Element, type: string) =>
  act(() => {
    el.dispatchEvent(new MouseEvent(type, { bubbles: true }));
  });

describe("MenuBar cutouts", () => {
  it("carves the open dropdown, and adds the fly-out when it appears", async () => {
    const restore = sizeTheMenus();
    try {
      __setCutoutSupportedForTest(true);
      await settle();
      act(() => root.render(<MenuBar />));
      await settle();
      log.length = 0;

      // Open the File menu (pointerdown, not click — native-menu feel).
      const file = [...container.querySelectorAll("button")].find(
        (b) => b.textContent === "File",
      )!;
      fire(file, "pointerdown");
      await settle();
      expect(log).toEqual(["region:100,24,220,300"]);

      // Hover "Recent Projects". Its panel is a sibling-positioned box OUTSIDE
      // the dropdown's rectangle, opened by `MenuList`'s own state — `MenuBar`
      // does not re-render, so only the MutationObserver can see it.
      const row = [...container.querySelectorAll("span")].find(
        (s) => s.textContent === "Recent Projects",
      )!;
      fire(row, "pointerover");
      await settle();
      expect(log).toEqual([
        "region:100,24,220,300",
        "region:100,24,220,300;320,60,200,150",
      ]);

      // Closing takes both away and never hides.
      fire(file, "pointerdown");
      await settle();
      expect(log.at(-1)).toBe("region:none");
      expect(log.filter((l) => l === "hide")).toEqual([]);
    } finally {
      restore();
    }
  });
});
