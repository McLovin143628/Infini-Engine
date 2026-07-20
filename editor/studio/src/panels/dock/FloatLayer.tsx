import { FloatingPanelFrame } from "../FloatingPanelFrame";
import { panelDefFor } from "../panelRegistry";
import { floatingPanelIds, useDockLayout } from "./dockLayoutStore";
import { PanelChromeContext } from "./panelChromeContext";

/**
 * Hosts the FLOATING panels: a `pointer-events-none` overlay across the
 * whole workspace area (docks + center — floats can hover anywhere), each
 * panel opting pointer events back in via its `FloatingPanelFrame`.
 */
export function FloatLayer() {
  const layout = useDockLayout((s) => s.layout);
  const hydrated = useDockLayout((s) => s.hydrated);

  if (!hydrated) return null;

  return (
    <div className="pointer-events-none absolute inset-0 z-20">
      <PanelChromeContext.Provider value="floating">
        {floatingPanelIds(layout).map((id) => {
          const def = panelDefFor(id);
          const panel = layout.panels[id];
          if (!def || !panel) return null;
          const Body = def.component;
          return (
            <FloatingPanelFrame key={id} id={id}>
              <Body panelId={id} params={panel.params} />
            </FloatingPanelFrame>
          );
        })}
      </PanelChromeContext.Provider>
    </div>
  );
}
