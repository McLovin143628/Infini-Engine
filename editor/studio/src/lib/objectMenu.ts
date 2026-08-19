/**
 * **The object context menu, built once** (Wave E).
 *
 * The Outliner's menu and the native viewport's menu (batch C) are the same
 * menu over the same selection, so they are the same function — the
 * `lib/assetDrop.ts` doctrine again. Two hand-rolled copies would have drifted
 * the first time an item was added to one of them.
 *
 * # Rules encoded here
 *
 * * **Everything dispatches a registered command.** No second command system:
 *   the menu, the menu bar, the palette and the keybindings all reach the same
 *   handlers, so anything here is also reachable by keyboard and by search.
 * * **The intersection, on a multi-selection.** An action offered over five
 *   objects must apply to five objects; `commonRoutes` decides.
 * * **A refusal is an item, not an absence.** When nothing can be edited, the
 *   menu shows one DISABLED row carrying the reason — so the user learns *why*
 *   there is no Model Editor for a built-in cube, rather than concluding the
 *   feature is missing.
 * * **Items whose command is not registered are omitted**, never shown dead.
 */
import { Copy, Eye, EyeOff, PencilLine, SquarePen, Trash2, Crosshair } from "lucide-react";

import type { EntityEditorsDto } from "../bindings/EntityEditorsDto";
import type { ContextMenuEntry } from "../components/ContextMenu";
import { executeCommand, getCommand } from "./commands";
import { commonRoutes, noEditorReason } from "./objectEditors";

/** Callbacks the host panel supplies for the things it owns. */
export interface ObjectMenuActions {
  /** Open the object's editors as a tab group (the double-click action). */
  open: (guid: string) => void;
  /** Open one specific editor route for the selection. */
  openRoute: (routeId: string, guids: string[]) => void;
  /** Start inline rename on `guid`. */
  rename: (guid: string) => void;
  /** Toggle visibility of `guid`. */
  toggleVisible: (guid: string) => void;
}

/**
 * Build the menu for `guids` (already the effective selection) given the
 * backend's answer about what they carry.
 *
 * `visible` is the current visibility of the primary object, used only to pick
 * the Show/Hide label; `null` when the caller does not know (the viewport menu
 * in batch C, which has no tree to read).
 */
export function buildObjectMenuItems(
  guids: string[],
  dtos: EntityEditorsDto[],
  actions: ObjectMenuActions,
  visible: boolean | null = null,
): ContextMenuEntry[] {
  const items: ContextMenuEntry[] = [];
  const single = guids.length === 1;
  const primary = guids[0];

  const routes = commonRoutes(dtos);
  if (routes.length > 0) {
    if (single) {
      items.push({
        label: "Open in Editor",
        icon: SquarePen,
        hint: "Double-click",
        onSelect: () => actions.open(primary),
      });
      items.push("separator");
    }
    for (const route of routes) {
      items.push({
        label: route.label,
        icon: PencilLine,
        onSelect: () => actions.openRoute(route.id, guids),
      });
    }
  } else {
    const reason = noEditorReason(dtos);
    if (reason) {
      items.push({
        label: "No editor for this selection",
        disabled: true,
        hint: reason,
        onSelect: () => {},
      });
    }
  }

  items.push("separator");

  // Focus is a viewport capability; the command is registered by the viewport
  // store only where it exists, so this row appears only when it can work.
  if (getCommand("view.focusSelection")) {
    items.push({
      label: "Focus",
      icon: Crosshair,
      hint: "F",
      onSelect: () => executeCommand("view.focusSelection"),
    });
  }

  if (single) {
    items.push({
      label: "Rename",
      icon: PencilLine,
      hint: "F2",
      onSelect: () => actions.rename(primary),
    });
    if (visible !== null) {
      items.push({
        label: visible ? "Hide" : "Show",
        icon: visible ? EyeOff : Eye,
        onSelect: () => actions.toggleVisible(primary),
      });
    }
  }

  if (getCommand("edit.duplicate")) {
    items.push({
      label: guids.length > 1 ? `Duplicate ${guids.length} Objects` : "Duplicate",
      icon: Copy,
      hint: "Ctrl+D",
      onSelect: () => executeCommand("edit.duplicate"),
    });
  }

  if (getCommand("edit.delete")) {
    items.push({
      label: guids.length > 1 ? `Delete ${guids.length} Objects` : "Delete",
      icon: Trash2,
      hint: "Del",
      danger: true,
      onSelect: () => executeCommand("edit.delete"),
    });
  }

  // Never end on a separator, and never emit two in a row: the builder is
  // conditional enough that both are reachable.
  return tidy(items);
}

/** Drop leading/trailing/duplicate separators. */
function tidy(items: ContextMenuEntry[]): ContextMenuEntry[] {
  const out: ContextMenuEntry[] = [];
  for (const item of items) {
    if (item === "separator" && (out.length === 0 || out[out.length - 1] === "separator")) continue;
    out.push(item);
  }
  while (out.length > 0 && out[out.length - 1] === "separator") out.pop();
  return out;
}
