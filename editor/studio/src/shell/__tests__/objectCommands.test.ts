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
  registerObjectEditorCommands,
} from "../../stores/objectEditorCommands";

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
});
