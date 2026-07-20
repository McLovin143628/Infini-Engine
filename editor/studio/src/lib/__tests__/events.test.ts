// @vitest-environment jsdom
//
// The event helpers wrap Tauri's `listen`, which rides the same IPC bridge
// as `invoke` (`plugin:event|listen`) — mockIPC intercepts it, and we push
// events through the registered handler via the window event emitter.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, expect, test, vi } from "vitest";

import { listenTo, listenToDynamic } from "../events";

afterEach(() => {
  clearMocks();
});

test("listenTo registers on the exact namespaced channel", async () => {
  const registered: string[] = [];
  mockIPC((cmd, args) => {
    if (cmd === "plugin:event|listen") {
      registered.push((args as { event: string }).event);
      return 1;
    }
  });
  const handler = vi.fn();
  await listenTo("log://line", handler);
  expect(registered).toEqual(["log://line"]);
});

test("listenToDynamic accepts parameterized channels", async () => {
  const registered: string[] = [];
  mockIPC((cmd, args) => {
    if (cmd === "plugin:event|listen") {
      registered.push((args as { event: string }).event);
      return 2;
    }
  });
  await listenToDynamic<{ id: string }>("assets://changed/xyz", () => {});
  expect(registered).toEqual(["assets://changed/xyz"]);
});
