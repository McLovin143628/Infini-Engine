/**
 * **The one resolver: which editors can open this object** (Wave E, batch B).
 *
 * The frontend half of `inf_editor_core::scene::editors`. Ring 1 answers *what
 * assets does this entity carry, and if none, why*; this module turns that into
 * *which panel opens, with which params, under which label*.
 *
 * One module, on the `lib/assetDrop.ts` doctrine ("one module so the two sides
 * cannot drift"): the Details buttons, the Outliner context menu, the viewport
 * context menu, the command palette and the double-click handler all call
 * `resolveObjectEditors`. Before this wave the Model Editor had exactly **two**
 * doors, both inside the Content Drawer, and neither of them started from an
 * object in the level.
 *
 * # Priority, and why
 *
 * `primaryRoute` is what a double-click activates, so the order is "what did the
 * user most likely mean by double-clicking this thing":
 *
 * 1. **Model Editor** on the mesh — the DCC is the feature this wave exists to
 *    surface, and a prop's mesh is what a double-click means everywhere else in
 *    this editor (the Content Drawer has meant *open* on double-click since P4).
 * 2. **Skeleton Editor** on the rig — a character with a skeleton but no mesh.
 * 3. **Blueprint** on the actor class — behaviour, when there is no geometry.
 * 4. **Material** — last, because a material is a property of the surface rather
 *    than the object's identity.
 *
 * A route is never fabricated: when the DTO carries no asset at all there are
 * ZERO routes and `no_editor_reason` is what the caller shows.
 */
import type { EntityEditorsDto } from "../bindings/EntityEditorsDto";

/** Panel types this resolver can route to. */
export type ObjectEditorPanel = "model" | "skeleton" | "blueprint" | "material";

/** One openable editor for an object. */
export interface ObjectEditorRoute {
  /** Stable id — also the `object.edit.*` command suffix. */
  id: "mesh" | "rig" | "blueprint" | "material";
  /** Menu / button label. */
  label: string;
  /** The panel registry type to open. */
  panelType: ObjectEditorPanel;
  /**
   * The panel's `params` — the asset GUID for the asset-scoped editors, and
   * `actor:<guid>` for the blueprint panel (which is keyed by document, not
   * asset, so the prefix is what tells `BlueprintCanvas` to raise rather than
   * to create).
   */
  params: string;
  /** The asset GUID this route opens (for status messages and tests). */
  assetId: string;
}

/**
 * Every editor that can open `dto`, in priority order.
 *
 * `SkeletalMesh.mesh` routes to the **Model Editor** exactly as `MeshRef.asset`
 * does: both are `.inf_mesh` assets and `dcc_open` accepts either. The rig is a
 * separate route because it is a separate asset (`.inf_skel`).
 */
export function resolveObjectEditors(dto: EntityEditorsDto): ObjectEditorRoute[] {
  const routes: ObjectEditorRoute[] = [];
  const mesh = dto.mesh ?? dto.skeletal_mesh;
  if (mesh) {
    routes.push({
      id: "mesh",
      label: "Edit Mesh",
      panelType: "model",
      params: mesh,
      assetId: mesh,
    });
  }
  if (dto.skeleton) {
    routes.push({
      id: "rig",
      label: "Edit Skeleton",
      panelType: "skeleton",
      params: dto.skeleton,
      assetId: dto.skeleton,
    });
  }
  if (dto.actor_class) {
    routes.push({
      id: "blueprint",
      label: "Open Blueprint",
      panelType: "blueprint",
      params: `actor:${dto.actor_class}`,
      assetId: dto.actor_class,
    });
  }
  if (dto.material) {
    routes.push({
      id: "material",
      label: "Edit Material",
      panelType: "material",
      params: dto.material,
      assetId: dto.material,
    });
  }
  return routes;
}

/** What a double-click activates; `null` when nothing can be opened. */
export function primaryRoute(dto: EntityEditorsDto): ObjectEditorRoute | null {
  return resolveObjectEditors(dto)[0] ?? null;
}

/**
 * The routes shared by EVERY object in a multi-selection, by route id.
 *
 * A menu over a mixed selection shows the **intersection**: an action offered
 * over five objects must be applicable to five objects. The route objects
 * returned are the first object's — the caller re-resolves per target when it
 * acts, so nothing here can open the wrong asset.
 */
export function commonRoutes(dtos: EntityEditorsDto[]): ObjectEditorRoute[] {
  if (dtos.length === 0) return [];
  const first = resolveObjectEditors(dtos[0]);
  return first.filter((r) =>
    dtos.every((d) => resolveObjectEditors(d).some((o) => o.id === r.id)),
  );
}

/**
 * The message to show when nothing can be opened for `dtos`.
 *
 * `null` when at least one object HAS a route (the menu shows the intersection
 * instead). For a mixed selection where no route is shared, the reason names the
 * mismatch rather than borrowing one object's excuse — quoting "this is a
 * built-in primitive" while a rigged character is also selected would be a lie.
 */
export function noEditorReason(dtos: EntityEditorsDto[]): string | null {
  if (dtos.length === 0) return null;
  const withRoutes = dtos.filter((d) => resolveObjectEditors(d).length > 0);
  if (withRoutes.length === dtos.length && commonRoutes(dtos).length > 0) return null;
  if (withRoutes.length === 0) {
    // Every object refuses. One object → its own reason; many → say how many.
    if (dtos.length === 1) {
      return dtos[0].no_editor_reason ?? "This object has no editable asset.";
    }
    const reasons = new Set(dtos.map((d) => d.no_editor_reason ?? ""));
    if (reasons.size === 1) return [...reasons][0] || "These objects have no editable asset.";
    return `None of the ${dtos.length} selected objects has an editable asset.`;
  }
  return `The ${dtos.length} selected objects have no editor in common — select one at a time to edit them.`;
}
