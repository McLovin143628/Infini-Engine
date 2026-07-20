// @vitest-environment jsdom
//
// Locks the typed IPC wrappers to the backend command names and payload
// shapes they were written against (Tauri's mockIPC intercepts `invoke`).
// The payload shapes themselves are the ts-rs generated bindings, so a
// backend type change shows up here as a compile error before it ships.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, expect, test } from "vitest";

import { app, layouts, viewport } from "../ipc";

afterEach(() => {
  clearMocks();
});

test("every wrapper invokes its registered command with the typed payload", async () => {
  const calls: Array<[string, unknown]> = [];
  mockIPC((cmd, args) => {
    calls.push([cmd, args]);
    if (cmd === "app_version") return "0.1.0";
  });

  await expect(app.version()).resolves.toBe("0.1.0");
  await viewport.attach();
  await viewport.setRect({ x: 1, y: 2, width: 3, height: 4 });
  await viewport.drop({ x: 5, y: 6, payload: "TestActor" });

  expect(calls.map(([cmd]) => cmd)).toEqual([
    "app_version",
    "viewport_attach",
    "viewport_set_rect",
    "viewport_drop",
  ]);
  expect(calls[2][1]).toEqual({ rect: { x: 1, y: 2, width: 3, height: 4 } });
  expect(calls[3][1]).toEqual({ drop: { x: 5, y: 6, payload: "TestActor" } });
});

test("layout wrappers invoke the layout_* commands with typed payloads", async () => {
  const calls: Array<[string, unknown]> = [];
  mockIPC((cmd, args) => {
    calls.push([cmd, args]);
    if (cmd === "layout_load") return '{"v":1}';
    if (cmd === "layout_list") return [{ name: "Default", modified_ms: 0 }];
    if (cmd === "layout_delete") return true;
  });

  await layouts.save("Default", '{"v":1}');
  await expect(layouts.load("Default")).resolves.toBe('{"v":1}');
  await expect(layouts.list()).resolves.toEqual([{ name: "Default", modified_ms: 0 }]);
  await expect(layouts.delete("Default")).resolves.toBe(true);

  expect(calls.map(([cmd]) => cmd)).toEqual([
    "layout_save",
    "layout_load",
    "layout_list",
    "layout_delete",
  ]);
  expect(calls[0][1]).toEqual({ name: "Default", json: '{"v":1}' });
});
