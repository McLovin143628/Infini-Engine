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
