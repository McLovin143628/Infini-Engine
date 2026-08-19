/**
 * **The object → editor doors** (Wave E, batch B).
 *
 * One command family, `object.edit.*`, plus the "open everything this object
 * has" action a double-click and the context menus fire. Registered the way
 * `registerViewportCommands` registers `view.*`/`tool.*` — the family is not in
 * the menu tree's static data, so it is `registerCommands` + `setCommandHandler`
 * rather than an attach to an enumerated id. `actor.rename` IS in the menu
 * (advertising F2 with no handler since Phase 1), so that one attaches.
 *
 * # Why the commands and not the components do the work
 *
 * The house rule is one command registry: menus, the palette, keybindings and
 * every context menu dispatch the SAME commands. So the Outliner's double-click,
 * the Details buttons, the Outliner context menu and (batch C) the viewport
 * context menu all route here, and a keyboard user reaches every one of them
 * through Ctrl+Shift+P.
 */
import type { EntityEditorsDto } from "../bindings/EntityEditorsDto";
import { registerCommands, setCommandHandler, getCommand } from "../lib/commands";
import { scene as sceneIpc } from "../lib/ipc";
import {
  noEditorReason,
  primaryRoute,
  resolveObjectEditors,
  type ObjectEditorRoute,
} from "../lib/objectEditors";
import { useDockLayout } from "../panels/dock/dockLayoutStore";
import { useSceneStore } from "./sceneStore";
import { useShellStore } from "./shellStore";

/** The command id for a route, e.g. `mesh` → `object.edit.mesh`. */
export function commandIdFor(routeId: ObjectEditorRoute["id"]): string {
  return `object.edit.${routeId}`;
}

/** The enumerated family — id, title and the route each one opens. */
export const OBJECT_EDIT_COMMANDS: {
  id: string;
  title: string;
  route: ObjectEditorRoute["id"];
}[] = [
  { id: "object.edit.mesh", title: "Actor: Edit Mesh", route: "mesh" },
  { id: "object.edit.rig", title: "Actor: Edit Skeleton", route: "rig" },
  { id: "object.edit.blueprint", title: "Actor: Open Blueprint", route: "blueprint" },
  { id: "object.edit.material", title: "Actor: Edit Material", route: "material" },
];

/** Report a refusal in the status bar — never silently do nothing. */
function refuse(message: string): void {
  useShellStore.getState().pushStatus(message, 6000);
}

/** The guids a command acts on: the argument, else the live selection. */
function targets(guids?: string[]): string[] {
  if (guids && guids.length > 0) return guids;
  return useSceneStore.getState().selection;
}

/**
 * Open one editor for each target that has that route.
 *
 * Each target is resolved separately, so a multi-selection of three props opens
 * three Model Editors on three different meshes rather than three copies of the
 * first one. Duplicate asset ids collapse naturally — the panels are keyed by
 * `type:params`, so two entities sharing a mesh open one panel.
 */
export async function openObjectEditor(
  routeId: ObjectEditorRoute["id"],
  guids?: string[],
): Promise<void> {
  const list = targets(guids);
  if (list.length === 0) {
    refuse("Select an object first.");
    return;
  }
  const dtos = await sceneIpc.entityEditors(list);
  const dock = useDockLayout.getState();
  let opened = 0;
  let anchor: string | null = null;
  for (const dto of dtos) {
    const route = resolveObjectEditors(dto).find((r) => r.id === routeId);
    if (!route) continue;
    const id: string = anchor
      ? dock.openBesidePanel(anchor, route.panelType, route.params)
      : dock.openPanel(route.panelType, route.params);
    anchor ??= id;
    opened += 1;
  }
  if (opened === 0) {
    refuse(noEditorReason(dtos) ?? "Nothing to open for this selection.");
  }
}

/**
 * **What a double-click means**: open this object's primary editor, and dock
 * every other editor it has as a TAB of the same group.
 *
 * That is the mandate's shape — the DCC opens with the object loaded and the
 * object's blueprint sits beside it. The grouping is why `openBesidePanel`
 * exists; without it three routes would open three panels in three places.
 *
 * **A CODE tab is not among them, and that is honest rather than an oversight**:
 * a blueprint class has no on-disk Rust to open. `.inf_act` stores the lowered
 * IR, and `graph_generate`'s output has never been written to a file — inventing
 * a path for it is a transpiler-workflow decision, not a UX one. Carried as a
 * named remainder in the Wave E ledger.
 */
export async function openObject(guid: string): Promise<void> {
  const [dto] = await sceneIpc.entityEditors([guid]);
  if (!dto) return;
  const primary = primaryRoute(dto);
  if (!primary) {
    refuse(noEditorReason([dto]) ?? "This object has no editable asset.");
    return;
  }
  const dock = useDockLayout.getState();
  const anchor = dock.openPanel(primary.panelType, primary.params);
  for (const route of resolveObjectEditors(dto)) {
    if (route.id === primary.id) continue;
    dock.openBesidePanel(anchor, route.panelType, route.params);
  }
  // The primary stays the active tab: it is what the user asked for.
  dock.activateTab(anchor);
  useShellStore.getState().pushStatus(`Opened ${dto.name} — ${primary.label}.`);
}

/** Fetch the editor rows for a selection (context menus, Details buttons). */
export async function fetchEntityEditors(guids: string[]): Promise<EntityEditorsDto[]> {
  if (guids.length === 0) return [];
  return sceneIpc.entityEditors(guids);
}

/**
 * Register the family. Called once at shell bootstrap, after
 * `bootstrapShellCommands` (which enumerates the menu tree).
 */
export function registerObjectEditorCommands(): void {
  registerCommands(
    OBJECT_EDIT_COMMANDS.map((c) => ({
      id: c.id,
      title: c.title,
      category: "Actor",
      run: () => void openObjectEditor(c.route),
    })),
  );
  registerCommands([
    {
      id: "object.open",
      title: "Actor: Open in Editor",
      category: "Actor",
      run: () => {
        const sel = useSceneStore.getState().selection;
        if (sel.length === 0) {
          refuse("Select an object first.");
          return;
        }
        void openObject(sel[0]);
      },
    },
  ]);

  // `actor.rename` has advertised F2 in the Actor menu since Phase 1 and had no
  // handler; the rename UI lives in the Outliner, which listens for this.
  if (getCommand("actor.rename")) {
    setCommandHandler("actor.rename", () => {
      const sel = useSceneStore.getState().selection;
      if (sel.length !== 1) {
        refuse(
          sel.length === 0 ? "Select an object to rename." : "Select a single object to rename.",
        );
        return;
      }
      requestRename(sel[0]);
    });
  }
}

// ── rename, as an event ──────────────────────────────────────────────────────
//
// The rename UI is an inline input inside the Outliner row, so the command
// cannot perform the rename itself — it asks. A window CustomEvent, exactly like
// `infinity:open-file`, so the command and the panel stay decoupled and the
// panel may be closed (in which case nothing happens, which is correct).

const RENAME_CHANNEL = "infinity:rename-object";

/** Ask whichever panel owns `guid`'s row to start inline renaming. */
export function requestRename(guid: string): void {
  window.dispatchEvent(new CustomEvent(RENAME_CHANNEL, { detail: { guid } }));
}

/** Subscribe to rename requests; returns the disposer. */
export function onRenameRequest(handler: (guid: string) => void): () => void {
  const listener = (e: Event) => {
    const guid = (e as CustomEvent<{ guid: string }>).detail?.guid;
    if (guid) handler(guid);
  };
  window.addEventListener(RENAME_CHANNEL, listener);
  return () => window.removeEventListener(RENAME_CHANNEL, listener);
}
