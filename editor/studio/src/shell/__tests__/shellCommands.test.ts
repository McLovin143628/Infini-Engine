import { describe, expect, it } from "vitest";
import { MENU_BAR, type MenuNode } from "../../lib/menus";
import { SPAWNABLES } from "../../lib/spawnables";
import { ACTOR_PLACE_KINDS, stubHint } from "../shellCommands";

describe("actor.place wiring table", () => {
  const validKinds = new Set(SPAWNABLES.map((s) => s.kind));

  it("keys are all actor.place.* command ids", () => {
    for (const id of Object.keys(ACTOR_PLACE_KINDS)) {
      expect(id.startsWith("actor.place.")).toBe(true);
    }
  });

  it("maps every entry to a real spawnable kind", () => {
    for (const kind of Object.values(ACTOR_PLACE_KINDS)) {
      expect(validKinds.has(kind), `unknown kind ${kind}`).toBe(true);
    }
  });

  it("includes the toolbar Add default (actor.place.empty → empty)", () => {
    expect(ACTOR_PLACE_KINDS["actor.place.empty"]).toBe("empty");
  });
});

describe("Place Actor ▸ Starter Character (wave GTA1)", () => {
  /** Every command id anywhere in the menu tree, submenus included. */
  function ids(items: readonly MenuNode[]): string[] {
    const out: string[] = [];
    for (const item of items) {
      if (item.kind === "submenu") out.push(...ids(item.items));
      else if (item.kind === "action") out.push(item.command);
    }
    return out;
  }

  it("is a row in the menu", () => {
    const all = MENU_BAR.flatMap((m) => ids(m.items));
    expect(all).toContain("actor.place.starterCharacter");
  });

  it("is NOT a primitive spawn", () => {
    // The row places the COMMITTED character through the character door. Wiring
    // it into this table would spawn a `SpawnKind` instead — a bare actor with
    // no rig, no capsule and no movement component, i.e. not a pawn at all.
    expect(ACTOR_PLACE_KINDS["actor.place.starterCharacter"]).toBeUndefined();
  });

  it("is not a stub", () => {
    expect(stubHint("actor.place.starterCharacter")).toBeUndefined();
  });
});

describe("stubHint — honest, phase-free messages", () => {
  it("has no hint for commands that are actually wired", () => {
    // Cut/copy/paste + duplicate are now live (scene clipboard, editor seams),
    // so they no longer surface a stub hint.
    for (const id of [
      "file.saveLevel",
      "file.saveLevelAs",
      "edit.undo",
      "edit.cut",
      "edit.copy",
      "edit.paste",
      "edit.duplicate",
      "actor.duplicate",
      "actor.place.cube",
      "select.all",
    ]) {
      expect(stubHint(id)).toBeUndefined();
    }
  });
});
