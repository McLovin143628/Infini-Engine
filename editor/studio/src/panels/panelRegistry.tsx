import type { ComponentType } from "react";
import type { LucideIcon } from "lucide-react";
import type { DockSide } from "./dock/dockTypes";

/**
 * Panel type registry (ported from GeoCanvas, ROADMAP P1.2). Every dockable
 * surface — Outliner, Details, Output Log, and every future editor panel —
 * is one `registerPanelType` call. The dock layout store and the
 * region/float/window hosts render panels exclusively through this
 * registry, so adding a panel is a registry entry, not surgery.
 *
 * Instance ids: singletons use `id === type` ("outliner", "details", …);
 * dynamic types mint `"<type>:<params>"` (e.g. a per-asset graph editor
 * tab in Phase 6).
 */

export interface PanelTypeDef {
  /** Registry key + singleton instance id. */
  type: string;
  /** Instance title (tab label / floating header / window title). */
  title: (params: string | null) => string;
  icon: LucideIcon;
  /** The panel body. Rendered inside the dock group / floating card /
   *  detached window chrome — it must NOT render its own window chrome. */
  component: ComponentType<{ panelId: string; params: string | null }>;
  /** One instance max, id === type. */
  singleton: boolean;
  /** Placement for a brand-new instance opened via `openPanel`. First-run
   *  defaults for the core singletons live in `defaultDockLayout`. */
  defaultLocation: DockSide | "float" | "window" | "hidden";
  /** Initial floating size. */
  defaultSize: { w: number; h: number };
  /** False = never detachable to an OS window (default true). */
  canDetach?: boolean;
  /** True = session-only: instances are dropped at hydrate instead of
   *  restored across app restarts. */
  transient?: boolean;
}

const registry = new Map<string, PanelTypeDef>();

export function registerPanelType(def: PanelTypeDef): void {
  registry.set(def.type, def);
}

export function panelTypeDef(type: string): PanelTypeDef | undefined {
  return registry.get(type);
}

export function panelTypeOf(panelId: string): string {
  const i = panelId.indexOf(":");
  return i === -1 ? panelId : panelId.slice(0, i);
}

/** Registry lookup for an instance id (`"outliner"`, `"graph:xyz"`). */
export function panelDefFor(panelId: string): PanelTypeDef | undefined {
  return registry.get(panelTypeOf(panelId));
}

export function panelTitle(panelId: string, params: string | null): string {
  const def = panelDefFor(panelId);
  return def ? def.title(params) : panelId;
}

export function registeredPanelTypes(): PanelTypeDef[] {
  return [...registry.values()];
}
