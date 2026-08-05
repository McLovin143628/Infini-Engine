// @vitest-environment jsdom
//
// The airspace refcount, now KEYED PER VIEWPORT (P23.2a).
//
// The rule it enforces is the oldest one in the shell: the native viewport
// draws over the webview, so an HTML surface that can cross the hole must hide
// it for its lifetime. What changed is that the count is per viewport id
// instead of one module-level number — and the whole point of this file is that
// the *existing* flows (palette, menus, the Content Drawer, nested overlays)
// behave exactly as they did, because every caller defaults to the scene
// viewport.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, beforeEach, expect, test } from "vitest";

import {
  __overlayCountForTest,
  acquireViewportOverlay,
} from "../viewportOverlay";
import { PRIMARY_VIEWPORT } from "../viewportIds";

let calls: Array<{ visible: boolean; viewport: string | undefined }>;

beforeEach(() => {
  calls = [];
  mockIPC((cmd, args) => {
    if (cmd === "viewport_set_visible") {
      const a = args as { visible: boolean; viewport?: string };
      calls.push({ visible: a.visible, viewport: a.viewport });
    }
  });
});

afterEach(() => {
  clearMocks();
});

/** `setVisible` is fire-and-forget; let its promise settle. */
const settle = () => new Promise((r) => setTimeout(r, 0));

test("an overlay hides the scene viewport and releasing shows it again", async () => {
  const release = acquireViewportOverlay();
  await settle();
  expect(__overlayCountForTest()).toBe(1);
  expect(calls).toEqual([{ visible: false, viewport: PRIMARY_VIEWPORT }]);

  release();
  await settle();
  expect(__overlayCountForTest()).toBe(0);
  expect(calls).toEqual([
    { visible: false, viewport: PRIMARY_VIEWPORT },
    { visible: true, viewport: PRIMARY_VIEWPORT },
  ]);
});

test("nested overlays hide once and show once — the menu-over-palette flow", async () => {
  // A menu opens, then a dialog over it, then the drag ghost: three acquires,
  // one hide. Releasing in any order must not un-hide early.
  const a = acquireViewportOverlay();
  const b = acquireViewportOverlay();
  const c = acquireViewportOverlay();
  await settle();
  expect(__overlayCountForTest()).toBe(3);
  expect(calls.filter((c) => !c.visible)).toHaveLength(1);

  b();
  c();
  await settle();
  expect(calls.filter((c) => c.visible)).toHaveLength(0);

  a();
  await settle();
  expect(__overlayCountForTest()).toBe(0);
  expect(calls.filter((c) => c.visible)).toHaveLength(1);
});

test("a release fires once however many times it is called", async () => {
  const release = acquireViewportOverlay();
  const second = acquireViewportOverlay();
  await settle();

  release();
  release();
  release();
  await settle();
  // Still held by `second` — a double release must not have decremented twice
  // and un-hidden the viewport under an open overlay.
  expect(__overlayCountForTest()).toBe(1);
  expect(calls.filter((c) => c.visible)).toHaveLength(0);

  second();
  await settle();
  expect(__overlayCountForTest()).toBe(0);
});

test("counts are per viewport: one overlay never un-hides another's window", async () => {
  // The reason the counter was keyed. With a single module-level number, the
  // model viewport's release below would have driven the count to zero and
  // shown the SCENE viewport while its palette was still open.
  const scene = acquireViewportOverlay();
  const model = acquireViewportOverlay("model");
  await settle();
  expect(__overlayCountForTest(PRIMARY_VIEWPORT)).toBe(1);
  expect(__overlayCountForTest("model")).toBe(1);
  expect(calls).toEqual([
    { visible: false, viewport: PRIMARY_VIEWPORT },
    { visible: false, viewport: "model" },
  ]);

  model();
  await settle();
  // The scene viewport is still covered by its own overlay.
  expect(__overlayCountForTest(PRIMARY_VIEWPORT)).toBe(1);
  expect(calls.at(-1)).toEqual({ visible: true, viewport: "model" });

  scene();
  await settle();
  expect(calls.at(-1)).toEqual({ visible: true, viewport: PRIMARY_VIEWPORT });
});

test("an unknown viewport's count starts at zero", () => {
  expect(__overlayCountForTest("never-acquired")).toBe(0);
});
