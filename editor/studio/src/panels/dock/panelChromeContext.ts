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
