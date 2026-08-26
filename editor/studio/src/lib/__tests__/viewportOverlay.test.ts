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
  __setCutoutSupportedForTest,
  acquireViewportOverlay,
  acquireViewportOverlayFor,
  registerViewport,
} from "../viewportOverlay";
import { PRIMARY_VIEWPORT } from "../viewportIds";
import { viewport } from "../ipc";

let calls: Array<{ visible: boolean; viewport: string | undefined }>;
/** `viewport_set_region` pushes, in order (UX2). */
let regions: Array<{
  rects: Array<{ x: number; y: number; width: number; height: number }>;
  viewport: string | undefined;
}>;
/** Both channels in ONE ordered list — the cutout claims are about order. */
let log: string[];
/** `viewport_attach` ids, in order — so a test can assert attach-then-register. */
let attachCalls: string[];

beforeEach(() => {
  __resetViewportOverlayForTest();
  calls = [];
  regions = [];
  log = [];
  attachCalls = [];
  mockIPC((cmd, args) => {
    if (cmd === "viewport_set_visible") {
      const a = args as { visible: boolean; viewport?: string };
      calls.push({ visible: a.visible, viewport: a.viewport });
      log.push(a.visible ? "show" : "hide");
    }
    if (cmd === "viewport_set_region") {
      const a = args as {
        rects: Array<{ x: number; y: number; width: number; height: number }>;
        viewport?: string;
      };
      regions.push({ rects: a.rects, viewport: a.viewport });
      log.push(
        a.rects.length === 0
          ? "region:none"
          : `region:${a.rects.map((r) => `${r.x},${r.y},${r.width},${r.height}`).join(";")}`,
      );
    }
    if (cmd === "viewport_attach") {
      attachCalls.push((args as { viewport?: string }).viewport ?? PRIMARY_VIEWPORT);
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
  // **The real sequence** (P23.2a audit). `ViewportPanel` registers only once
  // `viewport.attach` has RESOLVED — an earlier version registered from its own
  // effect, which ran first, so the hide reached a backend whose viewport map
  // was still empty, was dropped, and the native child then attached and drew
  // straight over the open overlay. The test has to run the order the panel
  // does or it certifies a fix that is not there.
  const release = acquireViewportOverlay();
  await settle();
  calls.length = 0;

  // 1. The panel mounts and asks the backend to attach…
  await viewport.attach("model");
  // 2. …and only then announces itself. This is the hide that must happen.
  const dispose = registerViewport("model");
  await settle();

  const attachIndex = attachCalls.indexOf("model");
  expect(attachIndex, "attach must precede registration").toBeGreaterThanOrEqual(0);
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

// ── UX2: the cutout ─────────────────────────────────────────────────────────
//
// The claim under test is the wave's whole point: an overlay that can say where
// it is no longer blacks out the 3D view. The frontend cannot photograph a
// native child window, so every arm below asserts at the SEAM — which command
// crossed, carrying what, in what order — and the arm that matters most is the
// negative one: `viewport_set_visible(false)` must not be sent at all.

/** Put the module in the state a running editor is in: the platform can carve,
 *  the viewport is up and visible, and nothing has been pushed since. */
async function readyToCarve(): Promise<void> {
  __setCutoutSupportedForTest(true);
  await settle();
  calls.length = 0;
  regions.length = 0;
  log.length = 0;
}

const rect = (x: number, y: number, width: number, height: number) => ({
  x,
  y,
  width,
  height,
});

test("a measured overlay carves the viewport instead of hiding it", async () => {
  await readyToCarve();

  const release = acquireViewportOverlay([rect(400, 260, 220, 300)]);
  await settle();
  expect(regions).toEqual([
    { rects: [rect(400, 260, 220, 300)], viewport: PRIMARY_VIEWPORT },
  ]);
  // **The negative arm.** A hide here is the bug the wave removes; it would also
  // pass a test that only checked the region was sent.
  expect(calls).toEqual([]);
  expect(log).toEqual(["region:400,260,220,300"]);

  release();
  await settle();
  // Released, not left behind: a stale region is a permanent hole in the view.
  expect(regions.at(-1)).toEqual({ rects: [], viewport: PRIMARY_VIEWPORT });
  expect(calls).toEqual([]);
});

test("a menu that moves re-carves without a hide in between", async () => {
  await readyToCarve();
  const hold = acquireViewportOverlay([rect(10, 10, 100, 80)]);
  await settle();

  hold.setRects([rect(500, 300, 100, 80)]);
  await settle();
  expect(log).toEqual(["region:10,10,100,80", "region:500,300,100,80"]);

  // An unchanged measurement (every render re-measures) costs no IPC at all.
  hold.setRects([rect(500, 300, 100, 80)]);
  await settle();
  expect(log).toHaveLength(2);
  hold();
});

test("one unmeasured overlay hides everything, however well the others measure", async () => {
  // The pessimism rule: the palette opening over an open menu must black the
  // view out, because the palette cannot say where it is.
  await readyToCarve();
  const menu = acquireViewportOverlay([rect(400, 260, 220, 300)]);
  await settle();
  expect(calls).toEqual([]);

  const palette = acquireViewportOverlay();
  await settle();
  expect(log).toEqual(["region:400,260,220,300", "hide", "region:none"]);

  // Closing the palette gives the menu its cutout back.
  palette();
  await settle();
  expect(log.slice(3)).toEqual(["region:400,260,220,300", "show"]);
  menu();
});

test("the pushes are ordered so the child is never whole under an open menu", async () => {
  // Coming back into view the region goes first; going away the visibility
  // does. Either way round the wrong way is a one-frame flash of exactly the
  // artefact the mechanism exists to remove.
  await readyToCarve();
  const menu = acquireViewportOverlay([rect(1, 2, 3, 4)]);
  const palette = acquireViewportOverlay();
  await settle();
  // Each acquisition syncs as it happens, so the menu's cutout lands before the
  // palette takes it away — and the hide is pushed BEFORE the region is
  // released, never after.
  expect(log).toEqual(["region:1,2,3,4", "hide", "region:none"]);

  log.length = 0;
  palette();
  await settle();
  // …and coming back, the region is restored BEFORE the child is shown.
  expect(log).toEqual(["region:1,2,3,4", "show"]);
  menu();
});

test("cutouts cross in PHYSICAL pixels, like every other viewport rect", async () => {
  const dpr = window.devicePixelRatio;
  Object.defineProperty(window, "devicePixelRatio", { value: 2, configurable: true });
  try {
    await readyToCarve();
    const hold = acquireViewportOverlay([rect(400, 260, 220, 300)]);
    await settle();
    expect(regions[0].rects).toEqual([rect(800, 520, 440, 600)]);
    hold();
  } finally {
    Object.defineProperty(window, "devicePixelRatio", { value: dpr, configurable: true });
  }
});

test("with no cutout backend a measured overlay still hides — macOS and Linux", async () => {
  // The one-platform law at the seam. `viewport_cutout_supported` answers
  // `cfg!(windows)`; everywhere else the menu would be drawn UNDER a viewport
  // that stayed visible, which is worse than the blackout.
  __setCutoutSupportedForTest(false);
  await settle();
  calls.length = 0;
  regions.length = 0;
  log.length = 0;

  const hold = acquireViewportOverlay([rect(400, 260, 220, 300)]);
  await settle();
  expect(log).toEqual(["hide"]);
  expect(regions).toEqual([]);

  hold();
  await settle();
  expect(log).toEqual(["hide", "show"]);
});

test("a panel-local hold has no rect to offer, so it hides", async () => {
  // `acquireViewportOverlayFor` is the popover case that does not exist yet; it
  // must not silently borrow a measured overlay's cutout.
  await readyToCarve();
  const scoped = acquireViewportOverlayFor(PRIMARY_VIEWPORT);
  await settle();
  expect(log).toEqual(["hide"]);
  scoped();
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
