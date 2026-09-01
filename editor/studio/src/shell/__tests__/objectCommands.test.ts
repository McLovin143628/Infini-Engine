// @vitest-environment jsdom
/**
 * Wave E batch B's wiring table: **every `object.*` id the Actor menu offers has
 * a handler**, and every route the resolver can produce has a command.
 *
 * The failure mode this catches is the one the whole wave is about: a menu entry
 * that looks live, dispatches, reaches the unhandled hook and toasts "is not
 * implemented yet" — which is exactly what `edit.editorPreferences` and
 * `actor.rename` did for fifteen phases.
 */
import { describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  scene: { entityEditors: vi.fn().mockResolvedValue([]) },
  app: { buildInfo: vi.fn().mockResolvedValue("") },
}));

import { getCommand } from "../../lib/commands";
import { MENU_BAR, type MenuAction, type MenuNode } from "../../lib/menus";
import { bootstrapShellCommands } from "../shellCommands";
import {
  OBJECT_EDIT_COMMANDS,
  commandIdFor,
  openObject,
  registerObjectEditorCommands,
} from "../../stores/objectEditorCommands";
import { useShellStore } from "../../stores/shellStore";

function actionsOf(nodes: MenuNode[]): MenuAction[] {
  const out: MenuAction[] = [];
  for (const n of nodes) {
    if (n.kind === "action") out.push(n);
    else if (n.kind === "submenu") out.push(...actionsOf(n.items));
  }
  return out;
}

describe("object editor command wiring", () => {
  bootstrapShellCommands();
  registerObjectEditorCommands();

  const actorMenu = MENU_BAR.find((m) => m.id === "actor")!;
  const ids = actionsOf(actorMenu.items).map((a) => a.command);

  it("the Actor menu offers the object routing", () => {
    expect(ids).toContain("object.open");
    for (const c of OBJECT_EDIT_COMMANDS) expect(ids).toContain(c.id);
  });

  it("every object.* menu id has a HANDLER, not just a definition", () => {
    for (const id of ids.filter((i) => i.startsWith("object."))) {
      const cmd = getCommand(id);
      expect(cmd, `${id} is not registered`).toBeDefined();
      expect(typeof cmd!.run, `${id} is enumerated with no handler`).toBe("function");
    }
  });

  /**
   * **Place Actor ▸ Starter Character has a HANDLER** (wave GTA1 audit).
   *
   * Its own three arms in `shellCommands.test.ts` check that the row is in the
   * menu, that it is not in the `SpawnKind` table and that it has no stub hint —
   * and a row whose `setCommandHandler` call was deleted passes all three, then
   * dispatches into the unhandled hook and toasts "is not implemented yet",
   * which is the exact failure that file's own header says it exists to catch.
   * This is the assertion those three cannot make, and it belongs here because
   * this is the suite that calls `bootstrapShellCommands`.
   */
  it("actor.place.starterCharacter is wired, not just enumerated", () => {
    const cmd = getCommand("actor.place.starterCharacter");
    expect(cmd, "the Place Actor row is not registered").toBeDefined();
    expect(typeof cmd!.run, "the Place Actor row is enumerated with no handler").toBe("function");
  });

  it("actor.rename — advertised with F2 since Phase 1 — is finally wired", () => {
    const rename = getCommand("actor.rename");
    expect(rename).toBeDefined();
    expect(typeof rename!.run).toBe("function");
    // And the menu still advertises F2, which the keybinding table now honours.
    const entry = actionsOf(actorMenu.items).find((a) => a.command === "actor.rename");
    expect(entry?.shortcut).toBe("F2");
  });

  it("commandIdFor agrees with the table (one naming rule, not two)", () => {
    for (const c of OBJECT_EDIT_COMMANDS) {
      expect(commandIdFor(c.route)).toBe(c.id);
    }
  });

  /**
   * **A stale target is a refusal, not silence** (Wave E audit, A5).
   *
   * `entity_editors_many` deliberately SKIPS guids the document no longer has —
   * a context menu left open across a delete is a normal race, not a failure. So
   * `openObject` gets an empty list back, and returning quietly makes a
   * double-click on a just-deleted row look like a broken feature. The wave's
   * own doctrine is that a refusal is a value; this is where it applies.
   */
  it("opening an object that no longer exists says so", async () => {
    useShellStore.setState({ statusMessage: null });
    await openObject("00000000-0000-0000-0000-0000000dead0");
    expect(useShellStore.getState().statusMessage).toContain("no longer exists");
  });

  /**
   * The `object.edit.*` handlers must RETURN their promise: `executeCommand`
   * surfaces a rejected handler in the status bar, and a `void`-ed promise turns
   * a backend refusal into an unhandled rejection nobody sees.
   */
  it("the edit handlers return a promise so executeCommand can surface a failure", () => {
    for (const c of OBJECT_EDIT_COMMANDS) {
      const result = getCommand(c.id)!.run!();
      expect(typeof (result as Promise<void>)?.then).toBe("function");
      void (result as Promise<void>).catch(() => {});
    }
  });
});
