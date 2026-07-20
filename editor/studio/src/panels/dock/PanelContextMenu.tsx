import type { ReactNode } from "react";
import {
  AppWindow,
  ArrowDownToLine,
  ArrowLeftToLine,
  ArrowRightToLine,
  ArrowUpToLine,
  EyeOff,
  PictureInPicture2,
} from "lucide-react";
import { ContextMenu, type ContextMenuEntry } from "../../components/ContextMenu";
import { panelDefFor } from "../panelRegistry";
import { clampPanelRect } from "../panelRect";
import { useDockLayout } from "./dockLayoutStore";
import type { DockSide } from "./dockTypes";

const SIDES: Array<{ side: DockSide; label: string; icon: typeof ArrowLeftToLine }> = [
  { side: "left", label: "Dock left", icon: ArrowLeftToLine },
  { side: "right", label: "Dock right", icon: ArrowRightToLine },
  { side: "top", label: "Dock top", icon: ArrowUpToLine },
  { side: "bottom", label: "Dock bottom", icon: ArrowDownToLine },
];

/** Move a panel to a side's dock (join its first group / open a new one) —
 *  the keyboard/menu twin of the drag gesture. */
export function dockPanelToSide(panelId: string, side: DockSide): void {
  const store = useDockLayout.getState();
  const first = store.layout.docks[side].groups[0];
  store.applyDrop(
    panelId,
    first
      ? { kind: "dock", side, groupId: first.id, index: first.tabs.length }
      : { kind: "new-group", side, index: 0 },
  );
}

export function floatPanel(panelId: string): void {
  const store = useDockLayout.getState();
  const p = store.layout.panels[panelId];
  if (!p) return;
  const rect = store.container ? clampPanelRect(p.floatRect, store.container) : p.floatRect;
  store.applyDrop(panelId, { kind: "float", rect });
}

/**
 * Right-click menu shared by dock tabs and floating panel headers: every
 * drag outcome has a menu equivalent (accessibility parity for the unified
 * drag system).
 */
export function PanelContextMenu({
  panelId,
  children,
}: {
  panelId: string;
  children: ReactNode;
}) {
  const items = (): ContextMenuEntry[] => {
    const store = useDockLayout.getState();
    const location = store.layout.panels[panelId]?.location.kind;
    const entries: ContextMenuEntry[] = [
      {
        label: "Float panel",
        icon: PictureInPicture2,
        disabled: location === "floating",
        onSelect: () => floatPanel(panelId),
      },
    ];
    if (panelDefFor(panelId)?.canDetach !== false) {
      entries.push({
        label: "Move to new window",
        icon: AppWindow,
        disabled: location === "window",
        onSelect: () => store.applyDrop(panelId, { kind: "window" }),
      });
    }
    entries.push("separator");
    for (const { side, label, icon } of SIDES) {
      entries.push({ label, icon, onSelect: () => dockPanelToSide(panelId, side) });
    }
    entries.push("separator");
    entries.push({
      label: "Hide panel",
      icon: EyeOff,
      onSelect: () => store.hidePanel(panelId),
    });
    return entries;
  };

  return <ContextMenu items={items}>{children}</ContextMenu>;
}
