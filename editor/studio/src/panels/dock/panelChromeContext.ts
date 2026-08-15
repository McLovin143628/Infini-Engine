import { createContext, useContext } from "react";

/**
 * Which host a panel body is currently rendered in. Panel bodies can read
 * this to adapt (e.g. hide a redundant title when the dock tab strip
 * already shows it):
 *
 *  - `"floating"` — full card with header (drag handle, minimize, hide).
 *  - `"docked"`  — body only; the dock group's tab strip IS the header.
 *  - `"window"`  — body only; the detached window's titlebar IS the header.
 */
export type PanelChromeVariant = "floating" | "docked" | "window";

export const PanelChromeContext = createContext<PanelChromeVariant>("floating");

export function usePanelChromeVariant(): PanelChromeVariant {
  return useContext(PanelChromeContext);
}

/**
 * Whether this panel body is currently **on screen** (Hardening Wave E).
 *
 * A docked group keeps every tab's body MOUNTED and merely `hidden` — that is
 * deliberate (scroll positions, form state and editor sessions survive a tab
 * switch), and it means a background tab's effects keep running for ever. A
 * panel that polls has to know the difference: `GitPanel` shelled out to
 * `git status` every four seconds for the life of the session from the moment
 * it was first opened, whether or not anybody could see it.
 *
 * `true` by default, which is right for every host that has no tabs: a floating
 * card and a detached window are visible by construction, and a panel that
 * never reads this behaves exactly as it did.
 *
 * It is a *visibility* signal, not a focus one — `focusedPanel` in the dock
 * store answers a different question (which panel Ctrl+Z belongs to) and a
 * visible panel is routinely unfocused.
 */
export const PanelVisibleContext = createContext<boolean>(true);

export function usePanelVisible(): boolean {
  return useContext(PanelVisibleContext);
}
