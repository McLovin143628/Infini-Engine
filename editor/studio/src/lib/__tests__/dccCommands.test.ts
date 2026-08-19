/**
 * The Model Editor's keyboard: the two halves must agree, and the chords must
 * not collide with the shell's.
 *
 * The value of these is entirely about the NEXT chord somebody adds — a
 * `dcc.*` binding pointing at a command that does not exist is a key that does
 * nothing with no error anywhere, which is the failure shape a registry is
 * supposed to make impossible.
 */
import { describe, expect, it } from "vitest";

import { DCC_COMMANDS, focusedMeshAsset } from "../dccCommands";
import { DEFAULT_KEYBINDINGS } from "../keybindings";
import { useDockLayout } from "../../panels/dock/dockLayoutStore";

const dccChords = DEFAULT_KEYBINDINGS.filter((b) => b.command.startsWith("dcc."));

describe("the DCC command table", () => {
  it("binds only commands that exist", () => {
    const ids = new Set(DCC_COMMANDS.map((c) => c.id));
    for (const b of dccChords) {
      expect(ids, `${b.chord} is bound to a command nobody defines`).toContain(b.command);
    }
    // …and it is not vacuous: there really are chords.
    expect(dccChords.length).toBeGreaterThan(10);
  });

  it("has no chord twice, DCC or otherwise", () => {
    const seen = new Map<string, string>();
    for (const b of DEFAULT_KEYBINDINGS) {
      const prev = seen.get(b.chord);
      expect(prev, `${b.chord} is bound to both ${prev} and ${b.command}`).toBeUndefined();
      seen.set(b.chord, b.command);
    }
  });

  it("does not take Ctrl+S for the mesh", () => {
    // The level's save. Quietly changing what the most-pressed chord in the
    // editor means inside one panel is how work gets lost, so the mesh has its
    // own — and this arm is what stops a later tidy-up from "unifying" them.
    const ctrlS = DEFAULT_KEYBINDINGS.find((b) => b.chord === "Ctrl+S");
    expect(ctrlS?.command).toBe("file.saveLevel");
    expect(DEFAULT_KEYBINDINGS.find((b) => b.command === "dcc.save")?.chord).toBe("Ctrl+Shift+M");
  });

  it("every command carries the chord it is actually bound to", () => {
    // `shortcut` is display-only, so nothing enforces it — which is exactly why
    // it drifts, and a palette showing the wrong key is worse than none.
    //
    // **The `if (!bound) continue` is gone** (audit fix): it skipped exactly
    // the case this arm names. `dcc.select.grow` advertised `Ctrl+NumpadAdd`,
    // was bound to nothing, and the one test written to catch a phantom chord
    // exempted every phantom chord by construction.
    for (const c of DCC_COMMANDS) {
      const bound = DEFAULT_KEYBINDINGS.find((b) => b.command === c.id);
      if (!c.shortcut) {
        expect(bound, `${c.id} is bound to ${bound?.chord} and advertises nothing`).toBe(
          undefined,
        );
        continue;
      }
      expect(bound, `${c.id} advertises ${c.shortcut} and is bound to nothing`).toBeDefined();
      expect(c.shortcut, `${c.id} advertises ${c.shortcut} and is bound to ${bound?.chord}`).toBe(
        bound?.chord,
      );
    }
  });
});

describe("focusedMeshAsset", () => {
  it("is null when no panel is focused, and null for a panel that is not a mesh", () => {
    // The default layout focuses nothing, so a chord pressed at boot must do
    // nothing rather than reach whichever document the store last held.
    expect(focusedMeshAsset()).toBeNull();

    const id = useDockLayout.getState().openPanel("outliner");
    useDockLayout.getState().setFocusedPanel(id);
    expect(focusedMeshAsset()).toBeNull();
  });
});
