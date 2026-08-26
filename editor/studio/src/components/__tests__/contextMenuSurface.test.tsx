// @vitest-environment jsdom
//
// **The airspace guard on the controlled context-menu surface** (Wave E).
//
// The viewport context menu is the FIRST overlay whose anchor is inside the
// native viewport hole, so if it fails to hide the native child window the menu
// is drawn UNDER the 3D view: invisible, and clicking where an item appears to
// be hits the viewport instead. That failure is silent — nothing throws, nothing
// logs — which is why it is pinned here rather than left to a human pass.
//
// The claim is "for exactly its OPEN lifetime": a surface that acquired on mount
// (rather than on open) would blank the 3D view permanently, since the surface
// is mounted for the whole session next to `ViewportPanel`.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import { act } from "react-dom/test-utils";

import { ContextMenuSurface } from "../ContextMenu";
import {
  __overlayAllCountForTest,
  __resetViewportOverlayForTest,
  __setCutoutSupportedForTest,
} from "../../lib/viewportOverlay";

let container: HTMLDivElement;
let root: Root;
/** Every viewport push this render made, in order (UX2). */
let log: string[];

beforeEach(() => {
  __resetViewportOverlayForTest();
  log = [];
  mockIPC((cmd, args) => {
    if (cmd === "viewport_set_visible") {
      log.push((args as { visible: boolean }).visible ? "show" : "hide");
    }
    if (cmd === "viewport_set_region") {
      const rects = (args as {
        rects: Array<{ x: number; y: number; width: number; height: number }>;
      }).rects;
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

/**
 * jsdom lays nothing out, so every rect is 0×0 and a real menu would measure as
 * unmeasurable. Give the menu panel a size for the arms that are about what
 * crosses the seam — and read its POSITION back off its inline style, which is
 * where the surface writes the clamp. A fixed position here would make the
 * clamp-before-measure arm vacuous.
 */
function sizeTheMenu(width: number, height: number): () => void {
  const original = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function (this: Element) {
    if (this.getAttribute("role") !== "menu") return original.call(this);
    const style = (this as HTMLElement).style;
    const x = parseFloat(style.left) || 0;
    const y = parseFloat(style.top) || 0;
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

describe("ContextMenuSurface", () => {
  it("holds the viewport overlay guard for exactly its open lifetime", () => {
    // Mounted CLOSED: no guard, so the 3D view keeps drawing.
    act(() => {
      root.render(<ContextMenuSurface at={null} items={[]} onClose={() => {}} />);
    });
    expect(__overlayAllCountForTest()).toBe(0);

    // Opened: exactly one acquisition, not two.
    act(() => {
      root.render(
        <ContextMenuSurface
          at={{ x: 10, y: 20 }}
          items={[{ label: "Edit Mesh", onSelect: () => {} }]}
          onClose={() => {}}
        />,
      );
    });
    expect(__overlayAllCountForTest()).toBe(1);

    // Moved to a new point WITHOUT closing (a second right-click): still one.
    act(() => {
      root.render(
        <ContextMenuSurface
          at={{ x: 90, y: 90 }}
          items={[{ label: "Edit Mesh", onSelect: () => {} }]}
          onClose={() => {}}
        />,
      );
    });
    expect(__overlayAllCountForTest()).toBe(1);

    // Closed: released.
    act(() => {
      root.render(<ContextMenuSurface at={null} items={[]} onClose={() => {}} />);
    });
    expect(__overlayAllCountForTest()).toBe(0);
  });

  it("releases the guard when unmounted while OPEN", () => {
    act(() => {
      root.render(
        <ContextMenuSurface at={{ x: 1, y: 1 }} items={[]} onClose={() => {}} />,
      );
    });
    expect(__overlayAllCountForTest()).toBe(1);
    act(() => root.unmount());
    // A leak here means the 3D view stays blank for the rest of the session.
    expect(__overlayAllCountForTest()).toBe(0);
    root = createRoot(container); // afterEach unmounts again
  });

  it("renders its items into a portal, and a separator is not a button", () => {
    act(() => {
      root.render(
        <ContextMenuSurface
          at={{ x: 5, y: 5 }}
          items={[
            { label: "Open in Editor", onSelect: () => {} },
            "separator",
            { label: "Delete", danger: true, onSelect: () => {} },
          ]}
          onClose={() => {}}
        />,
      );
    });
    const menu = document.querySelector('[role="menu"]');
    expect(menu).not.toBeNull();
    const items = menu!.querySelectorAll('[role="menuitem"]');
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toContain("Open in Editor");
  });

  // ── UX2: the cutout ───────────────────────────────────────────────────────

  it("carves its own rectangle out of the viewport instead of blacking it out", async () => {
    // The wave's claim, end to end from a DOM rect to the IPC seam: opening the
    // menu sends the menu's rectangle and NOT a hide. Before UX2 the only push
    // was the hide, and the 3D view went black behind every right-click.
    const restore = sizeTheMenu(220, 300);
    try {
      __setCutoutSupportedForTest(true);
      await settle();
      log.length = 0;

      act(() => {
        root.render(
          <ContextMenuSurface
            at={{ x: 400, y: 260 }}
            items={[{ label: "Focus", onSelect: () => {} }]}
            onClose={() => {}}
          />,
        );
      });
      await settle();
      expect(log).toEqual(["region:400,260,220,300"]);

      act(() => {
        root.render(<ContextMenuSurface at={null} items={[]} onClose={() => {}} />);
      });
      await settle();
      // Released on close — a region nobody releases is a permanent hole.
      expect(log).toEqual(["region:400,260,220,300", "region:none"]);
    } finally {
      restore();
    }
  });

  it("clamps the menu into the window BEFORE the hole is punched", async () => {
    // The rect handed to the native side has to be where the menu ends up. The
    // clamp used to be a passive effect and ran after the measurement, so a
    // menu opened near the right edge punched its hole at the unclamped point
    // — a visible black bar beside a menu that had already moved.
    const restore = sizeTheMenu(200, 120);
    try {
      __setCutoutSupportedForTest(true);
      await settle();
      log.length = 0;
      // jsdom's window is 1024×768, so a 200×120 menu asked for at (1000, 700)
      // is moved to (820, 644).
      act(() => {
        root.render(
          <ContextMenuSurface
            at={{ x: 1000, y: 700 }}
            items={[{ label: "Focus", onSelect: () => {} }]}
            onClose={() => {}}
          />,
        );
      });
      await settle();
      const el = document.querySelector('[role="menu"]') as HTMLElement;
      expect(el.style.left).toBe("820px");
      expect(el.style.top).toBe("644px");
      // **The FIRST push already carries the clamped rectangle.** With the
      // clamp as a passive effect the first push is the unclamped one and a
      // second corrects it — a hole beside a menu that had already moved.
      expect(log[0]).toBe("region:820,644,200,120");
      expect(log).toHaveLength(1);
    } finally {
      restore();
    }
  });

  it("falls back to hiding the viewport when there is nothing to measure", async () => {
    // No layout (a portal that has not been positioned, a zero-size surface):
    // the guard must take the pre-UX2 path rather than leave the viewport
    // visible over a menu it could not place.
    __setCutoutSupportedForTest(true);
    await settle();
    log.length = 0;

    act(() => {
      root.render(
        <ContextMenuSurface
          at={{ x: 10, y: 20 }}
          items={[{ label: "Focus", onSelect: () => {} }]}
          onClose={() => {}}
        />,
      );
    });
    await settle();
    expect(log).toEqual(["hide"]);
  });

  it("a disabled item does not fire its action", () => {
    let fired = 0;
    let closed = 0;
    act(() => {
      root.render(
        <ContextMenuSurface
          at={{ x: 5, y: 5 }}
          items={[
            {
              label: "No editor for this selection",
              disabled: true,
              hint: "This is a built-in primitive",
              onSelect: () => {
                fired += 1;
              },
            },
          ]}
          onClose={() => {
            closed += 1;
          }}
        />,
      );
    });
    const item = document.querySelector('[role="menuitem"]') as HTMLButtonElement;
    expect(item.disabled).toBe(true);
    expect(item.title).toContain("built-in primitive");
    act(() => {
      item.click();
    });
    expect(fired).toBe(0);
    expect(closed).toBe(0); // a disabled row is inert, not a dismiss
  });
});
