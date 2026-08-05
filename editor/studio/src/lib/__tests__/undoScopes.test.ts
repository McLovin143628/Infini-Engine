// @vitest-environment jsdom
//
// **Panel-focus undo routing** (P23.2a) — the fix for a real shipping bug.
//
// `edit.undo` was wired straight to the scene store, so Ctrl+Z pressed while the
// Material editor had focus undid THE SCENE: an actor moved somewhere the author
// could not see, in a panel they were not looking at, while the graph in front
// of them did nothing. The material graph has had a server-side journal since
// P7.2; it just had no route from the keyboard.
//
// The three cases below are the whole contract: a focused panel that claims a
// scope gets it, a focused panel that claims none falls through to the scene,
// and no focus at all is the scene — the last two being exactly the pre-P23.2a
// behaviour, which is why they are asserted rather than assumed.
import { beforeEach, expect, test, vi } from "vitest";

import { useDockLayout } from "../../panels/dock/dockLayoutStore";
import { registerPanelType } from "../../panels/panelRegistry";
import {
  __resetUndoScopesForTest,
  dispatchRedo,
  dispatchUndo,
  registerUndoScope,
  undoScopeFor,
} from "../undoScopes";

/** A minimal registered type so `panelTypeOf`/`panelDefFor` resolve. */
function stubPanel(type: string) {
  registerPanelType({
    type,
    title: () => type,
    icon: (() => null) as never,
    component: (() => null) as never,
    singleton: true,
    defaultLocation: "float",
    defaultSize: { w: 100, h: 100 },
  });
}

beforeEach(() => {
  __resetUndoScopesForTest();
  useDockLayout.getState().setFocusedPanel(null);
  stubPanel("material");
  stubPanel("outliner");
});

test("focus the material panel → Ctrl+Z routes to material.undo", () => {
  const undo = vi.fn();
  const redo = vi.fn();
  registerUndoScope("material", { undo, redo });

  useDockLayout.getState().setFocusedPanel("material");

  expect(dispatchUndo()).toBe(true);
  expect(undo).toHaveBeenCalledTimes(1);
  expect(redo).not.toHaveBeenCalled();

  expect(dispatchRedo()).toBe(true);
  expect(redo).toHaveBeenCalledTimes(1);
});

test("focus the outliner → nothing claims it, so undo falls through to the scene", () => {
  const undo = vi.fn();
  registerUndoScope("material", { undo, redo: vi.fn() });

  useDockLayout.getState().setFocusedPanel("outliner");

  // `false` is the caller's signal to run the scene's undo (see `sceneStore`'s
  // `wire("edit.undo", …)`). The material scope must NOT have fired.
  expect(dispatchUndo()).toBe(false);
  expect(dispatchRedo()).toBe(false);
  expect(undo).not.toHaveBeenCalled();
});

test("no focused panel → the scene, which is the pre-P23.2a behaviour", () => {
  registerUndoScope("material", { undo: vi.fn(), redo: vi.fn() });
  expect(useDockLayout.getState().focusedPanel).toBeNull();
  expect(dispatchUndo()).toBe(false);
  expect(dispatchRedo()).toBe(false);
});

test("a dynamic instance id routes by its TYPE", () => {
  // Non-singleton panels mint `"<type>:<params>"` ids (`panelRegistry`), and a
  // per-asset Material tab is exactly that shape. Routing on the raw id would
  // find nothing and silently undo the scene — the original bug, back.
  const undo = vi.fn();
  registerUndoScope("material", { undo, redo: vi.fn() });
  useDockLayout.getState().setFocusedPanel("material:abc-123");
  expect(dispatchUndo()).toBe(true);
  expect(undo).toHaveBeenCalledTimes(1);
});

test("closing the focused panel clears the focus, so undo goes back to the scene", () => {
  const undo = vi.fn();
  registerUndoScope("material", { undo, redo: vi.fn() });
  const dock = useDockLayout.getState();
  dock.openPanel("material");
  dock.setFocusedPanel("material");
  expect(dispatchUndo()).toBe(true);

  useDockLayout.getState().hidePanel("material");
  expect(useDockLayout.getState().focusedPanel).toBeNull();
  expect(dispatchUndo()).toBe(false);
  expect(undo).toHaveBeenCalledTimes(1);
});

test("an unregistered type has no scope", () => {
  expect(undoScopeFor("nobody")).toBeUndefined();
  expect(undoScopeFor(null)).toBeUndefined();
});
