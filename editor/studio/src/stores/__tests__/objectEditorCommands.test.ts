// @vitest-environment jsdom
//
// **The object → editor doors, and the one route that is not a panel** (audit,
// Wave D).
//
// Wave D added a fifth route, `code`, whose `panelType` is a MARKER: the opener
// is meant to call the backend and hand the resulting PATH to the shell's
// Editor panel. What shipped instead docked it like any other route — and no
// panel of type `"code"` is registered, so double-clicking any actor with a
// blueprint class opened an empty, untitled tab, while `openGeneratedCode` sat
// in this file with zero callers anywhere in the app.
//
// Both halves are asserted here: the backend IS called, and the dock is NOT.
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  scene: { entityEditors: vi.fn() },
  graph: { writeSource: vi.fn() },
}));
vi.mock("../../lib/openFile", () => ({ requestOpenFile: vi.fn() }));

const openPanel = vi.fn(() => "blueprint:actor:a-1");
const openBesidePanel = vi.fn(() => "beside");
const activateTab = vi.fn();
vi.mock("../../panels/dock/dockLayoutStore", () => ({
  useDockLayout: { getState: () => ({ openPanel, openBesidePanel, activateTab }) },
}));

import type { EntityEditorsDto } from "../../bindings/EntityEditorsDto";
import { graph as graphIpc, scene as sceneIpc } from "../../lib/ipc";
import { requestOpenFile } from "../../lib/openFile";
import { OBJECT_EDIT_COMMANDS, openObject, openObjectEditor } from "../objectEditorCommands";

const row = (over: Partial<EntityEditorsDto>): EntityEditorsDto => ({
  entity: "e1",
  name: "Turret",
  kind: "Static Mesh",
  mesh: null,
  skeletal_mesh: null,
  skeleton: null,
  material: null,
  actor_class: null,
  primitive: null,
  no_editor_reason: null,
  ...over,
});

describe("the code route is a marker, not a panel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(graphIpc.writeSource).mockResolvedValue({
      path: "C:/p/src/blueprints/Turret_a1b2c3d4.rs",
      wrote: true,
      created: true,
    });
  });

  it("double-clicking an actor generates its Rust instead of docking a dead tab", async () => {
    vi.mocked(sceneIpc.entityEditors).mockResolvedValue([
      row({ mesh: "m-1", actor_class: "a-1" }),
    ]);
    await openObject("e1");
    expect(graphIpc.writeSource).toHaveBeenCalledWith("bp:a-1");
    expect(requestOpenFile).toHaveBeenCalledWith("C:/p/src/blueprints/Turret_a1b2c3d4.rs");
    // …and nothing was docked as a `code` panel, which is the shape that shipped.
    for (const call of [...openPanel.mock.calls, ...openBesidePanel.mock.calls]) {
      expect(call).not.toContain("code");
    }
  });

  it("the command family reaches it, so the palette and the menus do", async () => {
    expect(OBJECT_EDIT_COMMANDS.map((c) => c.route)).toContain("code");
    vi.mocked(sceneIpc.entityEditors).mockResolvedValue([row({ actor_class: "a-1" })]);
    await openObjectEditor("code", ["e1"]);
    expect(graphIpc.writeSource).toHaveBeenCalledWith("bp:a-1");
    expect(openPanel).not.toHaveBeenCalled();
  });

  it("an actor with no class has no code route to open", async () => {
    vi.mocked(sceneIpc.entityEditors).mockResolvedValue([row({ mesh: "m-1" })]);
    await openObjectEditor("code", ["e1"]);
    expect(graphIpc.writeSource).not.toHaveBeenCalled();
  });
});
