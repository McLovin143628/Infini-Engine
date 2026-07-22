import { describe, expect, it } from "vitest";
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

describe("stubHint — honest, phase-free messages", () => {
  it("returns a clipboard message for cut/copy/paste", () => {
    for (const id of ["edit.cut", "edit.copy", "edit.paste"]) {
      expect(stubHint(id)).toMatch(/clipboard/i);
      expect(stubHint(id)).not.toMatch(/phase/i);
    }
  });

  it("returns a duplicate message for edit/actor duplicate", () => {
    expect(stubHint("edit.duplicate")).toMatch(/duplicat/i);
    expect(stubHint("actor.duplicate")).toMatch(/duplicat/i);
  });

  it("has no hint for commands that are actually wired", () => {
    for (const id of ["file.saveLevel", "edit.undo", "actor.place.cube", "select.all"]) {
      expect(stubHint(id)).toBeUndefined();
    }
  });
});
