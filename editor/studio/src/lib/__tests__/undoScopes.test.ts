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
import { __resetCommandsForTest, registerCommands, setCommandHandler } from "../commands";
import { __resetKeybindingsForTest, bindKey } from "../keybindings";
import {
  __resetUndoScopesForTest,
  dispatchRedo,
  dispatchUndo,
  registerUndoScope,
  undoScopeFor,
} from "../undoScopes";
import { focusViewport, handleViewportChord, VIEWPORT_PANEL_ID } from "../viewportFocus";

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
  __resetCommandsForTest();
  __resetKeybindingsForTest();
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

// ── B1: the viewport takes focus back (P23.2a audit) ───────────────────────
//
// **The exact repro.** Click the Material editor, click into the viewport,
// press Ctrl+Z. Before the fix the graph rewound and the actor did not move,
// repeatably — `focusedPanel` still said "material", because a click on the 3D
// view is not a DOM event (the native child window swallows it) and no
// `setFocusedPanel` call site could ever name the viewport.
//
// "Click into the viewport" is exercised through the path that actually carries
// it: win32 forwards every Ctrl chord it declines on `viewport://key`, and the
// arrival of one is proof the native viewport holds OS focus.

test("B1: focus material, then Ctrl+Z over the viewport → the SCENE undoes", () => {
  stubPanel("viewport");
  const materialUndo = vi.fn();
  const sceneUndo = vi.fn();
  registerUndoScope("material", { undo: materialUndo, redo: vi.fn() });

  // Mirror `registerSceneCommands`' wiring exactly: undo routes to the focused
  // panel's scope, and falls through to the scene when nothing claims it.
  registerCommands([{ id: "edit.undo", title: "Undo", category: "Edit" }]);
  setCommandHandler("edit.undo", () => {
    if (!dispatchUndo()) sceneUndo();
  });
  bindKey({ chord: "Ctrl+Z", command: "edit.undo" });

  // 1. The author clicks the Material panel.
  useDockLayout.getState().setFocusedPanel("material");
  expect(dispatchUndo()).toBe(true);
  expect(materialUndo).toHaveBeenCalledTimes(1);

  // 2. …clicks into the viewport and presses Ctrl+Z. The chord arrives over
  //    `viewport://key`, which is the whole of "the viewport has focus".
  expect(handleViewportChord("Ctrl+Z")).toBe(true);

  // 3. The SCENE undid. The material graph did not move.
  expect(sceneUndo).toHaveBeenCalledTimes(1);
  expect(materialUndo).toHaveBeenCalledTimes(1);
  expect(useDockLayout.getState().focusedPanel).toBe(VIEWPORT_PANEL_ID);
});

test("B1: a pointer-down on the viewport's DOM margin focuses it too", () => {
  // The toolbar strip and the padding around the hole ARE real DOM surface, so
  // `App` hangs `focusViewport` on them. The hole itself cannot report.
  stubPanel("viewport");
  registerUndoScope("material", { undo: vi.fn(), redo: vi.fn() });
  useDockLayout.getState().setFocusedPanel("material");

  focusViewport();

  expect(useDockLayout.getState().focusedPanel).toBe(VIEWPORT_PANEL_ID);
  expect(dispatchUndo()).toBe(false); // ⇒ the caller runs the scene's undo
});

test("B1: the viewport registers no undo scope, so it rides the scene default", () => {
  // If a future edit gave "viewport" a scope, the two tests above would still
  // pass while Ctrl+Z over the 3D view stopped undoing the level.
  stubPanel("viewport");
  expect(undoScopeFor(VIEWPORT_PANEL_ID)).toBeUndefined();
});
