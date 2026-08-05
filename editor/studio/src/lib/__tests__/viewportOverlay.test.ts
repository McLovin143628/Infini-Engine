// @vitest-environment jsdom
//
// The airspace refcount (P23.2a).
//
// The rule it enforces is the oldest one in the shell: the native viewport
// draws over the webview, so an HTML surface that can cross the hole must hide
// it for its lifetime. What P23.2a changed is that the default acquisition is
// **window-wide** — every attached viewport — because every overlay the shell
// has (menus, palette, dialogs, drag ghost) is drawn over the whole workspace.
//
// The first three cases are the *existing* flows, asserted unchanged: with one
// viewport, window-wide and per-viewport are the same thing, and they had
// better still hide once and show once. The last four are what the change buys.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, beforeEach, expect, test } from "vitest";

import {
  __overlayAllCountForTest,
  __overlayCountForTest,
  __resetViewportOverlayForTest,
  acquireViewportOverlay,
  acquireViewportOverlayFor,
  registerViewport,
} from "../viewportOverlay";
import { PRIMARY_VIEWPORT } from "../viewportIds";

let calls: Array<{ visible: boolean; viewport: string | undefined }>;

beforeEach(() => {
  __resetViewportOverlayForTest();
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
  expect(__overlayAllCountForTest()).toBe(1);
  expect(calls).toEqual([{ visible: false, viewport: PRIMARY_VIEWPORT }]);

  release();
  await settle();
  expect(__overlayAllCountForTest()).toBe(0);
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
  expect(__overlayAllCountForTest()).toBe(3);
  expect(calls.filter((c) => !c.visible)).toHaveLength(1);

  b();
  c();
  await settle();
  expect(calls.filter((c) => c.visible)).toHaveLength(0);

  a();
  await settle();
  expect(__overlayAllCountForTest()).toBe(0);
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
  expect(__overlayAllCountForTest()).toBe(1);
  expect(calls.filter((c) => c.visible)).toHaveLength(0);

  second();
  await settle();
  expect(__overlayAllCountForTest()).toBe(0);
});

test("a window-wide overlay hides EVERY attached viewport", async () => {
  // The bug this closes: `Target::All` existed in Rust while the frontend
  // primitive could only name one viewport, so with a second viewport attached
  // every menu and dialog in the shell would have been painted over by it.
  const disposeModel = registerViewport("model");
  await settle();
  calls.length = 0;

  const release = acquireViewportOverlay();
  await settle();
  expect(calls).toEqual(
    expect.arrayContaining([
      { visible: false, viewport: PRIMARY_VIEWPORT },
      { visible: false, viewport: "model" },
    ]),
  );
  expect(calls).toHaveLength(2);

  release();
  await settle();
  expect(calls.filter((c) => c.visible)).toHaveLength(2);
  disposeModel();
});

test("a viewport that attaches while an overlay is open comes up hidden", async () => {
  const release = acquireViewportOverlay();
  await settle();
  calls.length = 0;

  // Opening a second viewport with the palette up must not punch through it.
  const dispose = registerViewport("model");
  await settle();
  expect(calls).toEqual([{ visible: false, viewport: "model" }]);

  release();
  await settle();
  expect(calls.filter((c) => c.visible && c.viewport === "model")).toHaveLength(1);
  dispose();
});

test("a scoped overlay hides only its own viewport", async () => {
  const dispose = registerViewport("model");
  await settle();
  calls.length = 0;

  const release = acquireViewportOverlayFor("model");
  await settle();
  expect(__overlayCountForTest("model")).toBe(1);
  expect(__overlayCountForTest(PRIMARY_VIEWPORT)).toBe(0);
  expect(calls).toEqual([{ visible: false, viewport: "model" }]);

  release();
  await settle();
  expect(calls.at(-1)).toEqual({ visible: true, viewport: "model" });
  expect(__overlayCountForTest("model")).toBe(0);
  dispose();
});

test("a scoped hold survives a window-wide release", async () => {
  // Both mechanisms at once: the window-wide overlay closes, but the panel-local
  // one is still up, so its viewport must stay hidden.
  const scoped = acquireViewportOverlayFor(PRIMARY_VIEWPORT);
  const wide = acquireViewportOverlay();
  await settle();
  calls.length = 0;

  wide();
  await settle();
  expect(calls).toEqual([]);

  scoped();
  await settle();
  expect(calls).toEqual([{ visible: true, viewport: PRIMARY_VIEWPORT }]);
});
