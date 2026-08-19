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
} from "../../lib/viewportOverlay";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  __resetViewportOverlayForTest();
  mockIPC(() => undefined);
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
