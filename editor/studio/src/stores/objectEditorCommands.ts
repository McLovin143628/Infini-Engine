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
import { graph as graphIpc, scene as sceneIpc } from "../lib/ipc";
import { requestOpenFile } from "../lib/openFile";
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
  // Wave D shipped the CODE tab's whole write door and never registered a
  // command for it, so `openGeneratedCode` had no caller anywhere in the app.
  { id: "object.edit.code", title: "Actor: Open Generated Rust", route: "code" },
];

/**
 * Route ids whose `panelType` is a **marker**, not a registered dock panel.
 *
 * `code` is one: the opener calls the backend and hands the resulting PATH to
 * the shell's existing Editor panel, because a second code editor would be a
 * second answer to a question Phase 5 already answered. Anything listed here
 * MUST be dispatched by `openRoute` rather than docked — `openPanel("code")`
 * opens an untitled, empty tab, which is what shipped.
 */
const MARKER_ROUTES: ReadonlySet<ObjectEditorRoute["id"]> = new Set(["code"]);

/**
 * Open one route: through the dock for a real panel, through its named opener
 * for a marker.
 *
 * **The one door.** Both `openObjectEditor` and `openObject` used to call
 * `dock.openPanel`/`openBesidePanel` directly, so a route the dock cannot open
 * became an empty tab in two places rather than a refusal in none.
 */
async function openRoute(
  route: ObjectEditorRoute,
  dock: ReturnType<typeof useDockLayout.getState>,
  anchor: string | null,
): Promise<string | null> {
  if (MARKER_ROUTES.has(route.id)) {
    await openGeneratedCode(route.assetId);
    return anchor;
  }
  return anchor
    ? dock.openBesidePanel(anchor, route.panelType, route.params)
    : dock.openPanel(route.panelType, route.params);
}

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
    anchor = await openRoute(route, dock, anchor);
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
 * **A CODE tab IS among them since Wave D.** Wave E's reason for its absence —
 * *"a blueprint class has no on-disk Rust to open"* — stopped being true when
 * `graph_write_source` started rendering the class into
 * `<project>/src/blueprints/<Class>_<guid8>.rs`. It is not a dock panel, though:
 * its route is a marker, `openRoute` calls the backend and the PATH goes to the
 * shell's Editor panel. Docking it as a panel type (which is what shipped)
 * opens an empty untitled tab, because no panel of that type is registered.
 */
export async function openObject(guid: string): Promise<void> {
  const [dto] = await sceneIpc.entityEditors([guid]);
  if (!dto) {
    // The object went away between the gesture and the answer — a context menu
    // left open across a delete, or a double-click on a row the outliner has
    // not re-synced yet. `entity_editors_many` SKIPS unknown guids, so this is
    // a value the command has to name: silence here reads as "the feature is
    // broken", which is the exact failure this wave exists to remove.
    refuse("That object no longer exists.");
    return;
  }
  const primary = primaryRoute(dto);
  if (!primary) {
    refuse(noEditorReason([dto]) ?? "This object has no editable asset.");
    return;
  }
  const dock = useDockLayout.getState();
  const anchor = await openRoute(primary, dock, null);
  for (const route of resolveObjectEditors(dto)) {
    if (route.id === primary.id) continue;
    await openRoute(route, dock, anchor);
  }
  // The primary stays the active tab: it is what the user asked for.
  if (anchor) dock.activateTab(anchor);
  useShellStore.getState().pushStatus(`Opened ${dto.name} — ${primary.label}.`);
}

/**
 * Generate (or find current) a blueprint class's Rust and open it in the Editor
 * panel.
 *
 * The path is the backend's answer, never built here: the convention lives in
 * `inf_editor_core::blueprint_source` and a frontend copy of it would be a
 * second convention that drifts the first time the backend's changes.
 */
export async function openGeneratedCode(assetId: string): Promise<void> {
  try {
    const out = await graphIpc.writeSource(`bp:${assetId}`);
    requestOpenFile(out.path);
    useShellStore
      .getState()
      .pushStatus(
        out.created
          ? `Generated ${out.path}`
          : out.wrote
            ? `Regenerated ${out.path}`
            : `Opened ${out.path} (already current)`,
      );
  } catch (e) {
    refuse(String(e));
  }
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
  // The handlers RETURN their promise rather than `void`-ing it: `executeCommand`
  // surfaces a rejected handler in the status bar with the command's title
  // (`lib/commands.ts`), and a discarded promise makes a backend refusal an
  // unhandled rejection in the console instead.
  registerCommands(
    OBJECT_EDIT_COMMANDS.map((c) => ({
      id: c.id,
      title: c.title,
      category: "Actor",
      run: () => openObjectEditor(c.route),
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
        return openObject(sel[0]);
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
