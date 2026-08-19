import { describe, expect, it } from "vitest";

import type { EntityEditorsDto } from "../../bindings/EntityEditorsDto";
import {
  commonRoutes,
  noEditorReason,
  primaryRoute,
  resolveObjectEditors,
} from "../objectEditors";

const row = (over: Partial<EntityEditorsDto>): EntityEditorsDto => ({
  entity: "e1",
  name: "Thing",
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

const PRIMITIVE_REASON =
  "This is a built-in primitive — it has no mesh asset to edit. Import or create a mesh in the Content Drawer and drop it onto this actor.";

describe("resolveObjectEditors", () => {
  it("routes a mesh-bound entity to the Model Editor on its asset", () => {
    const routes = resolveObjectEditors(row({ mesh: "m-1" }));
    expect(routes).toHaveLength(1);
    expect(routes[0]).toMatchObject({ id: "mesh", panelType: "model", params: "m-1" });
    expect(primaryRoute(row({ mesh: "m-1" }))?.id).toBe("mesh");
  });

  it("routes a character to BOTH the Model Editor and the Skeleton Editor", () => {
    const routes = resolveObjectEditors(
      row({ kind: "Skeletal Mesh", skeletal_mesh: "sm-1", skeleton: "sk-1" }),
    );
    expect(routes.map((r) => r.id)).toEqual(["mesh", "rig"]);
    // A skeletal mesh IS a `.inf_mesh`; `dcc_open` takes either.
    expect(routes[0].params).toBe("sm-1");
    expect(routes[1].panelType).toBe("skeleton");
  });

  it("prefixes the blueprint route's params so the canvas raises rather than creates", () => {
    const [route] = resolveObjectEditors(row({ actor_class: "a-1" }));
    expect(route).toMatchObject({ id: "blueprint", panelType: "blueprint", params: "actor:a-1" });
  });

  it("orders mesh → rig → blueprint → code → material", () => {
    const routes = resolveObjectEditors(
      row({ mesh: "m", skeleton: "s", actor_class: "a", material: "mat" }),
    );
    // Wave D put `code` after `blueprint` and before `material`: the generated
    // Rust IS the blueprint, seen the other way round, and a material is a
    // property of the surface rather than of the object's identity.
    expect(routes.map((r) => r.id)).toEqual(["mesh", "rig", "blueprint", "code", "material"]);
  });

  it("gives an actor class a code route, and nothing else one", () => {
    // The route exists exactly when there is a class to generate from — a mesh
    // has no Rust and must not offer to open some.
    const withClass = resolveObjectEditors(row({ actor_class: "a-1" }));
    expect(withClass.find((r) => r.id === "code")).toMatchObject({
      panelType: "code",
      assetId: "a-1",
    });
    const meshOnly = resolveObjectEditors(row({ mesh: "m-1" }));
    expect(meshOnly.find((r) => r.id === "code")).toBeUndefined();
  });

  /**
   * The vacuous-check law: asserting only `routes.length === 0` would pass for a
   * resolver that returns nothing for everything. The REASON is the claim.
   */
  it("a bare primitive gets zero routes and exactly one non-empty reason", () => {
    const dto = row({ primitive: "Cube", no_editor_reason: PRIMITIVE_REASON });
    expect(resolveObjectEditors(dto)).toHaveLength(0);
    expect(primaryRoute(dto)).toBeNull();
    const reason = noEditorReason([dto]);
    expect(reason).toBe(PRIMITIVE_REASON);
    expect(reason).toContain("built-in primitive");
    expect(reason).toContain("Content Drawer");
  });
});

describe("multi-selection", () => {
  it("offers only the INTERSECTION", () => {
    const a = row({ entity: "a", mesh: "m1", actor_class: "c1" });
    const b = row({ entity: "b", mesh: "m2" });
    expect(commonRoutes([a, b]).map((r) => r.id)).toEqual(["mesh"]);
    expect(noEditorReason([a, b])).toBeNull();
  });

  it("names the MISMATCH rather than borrowing one object's excuse", () => {
    const rigged = row({ entity: "a", kind: "Skeletal Mesh", skeleton: "sk" });
    const cube = row({ entity: "b", primitive: "Cube", no_editor_reason: PRIMITIVE_REASON });
    expect(commonRoutes([rigged, cube])).toHaveLength(0);
    const reason = noEditorReason([rigged, cube]);
    // Quoting "this is a built-in primitive" while a rigged character is also
    // selected would be a lie about the selection.
    expect(reason).not.toContain("built-in primitive");
    expect(reason).toContain("no editor in common");
  });

  it("collapses identical reasons across a uniform selection", () => {
    const a = row({ entity: "a", primitive: "Cube", no_editor_reason: PRIMITIVE_REASON });
    const b = row({ entity: "b", primitive: "Sphere", no_editor_reason: PRIMITIVE_REASON });
    expect(noEditorReason([a, b])).toBe(PRIMITIVE_REASON);
  });

  it("counts, when several objects refuse for different reasons", () => {
    const a = row({ entity: "a", no_editor_reason: "one" });
    const b = row({ entity: "b", no_editor_reason: "two" });
    expect(noEditorReason([a, b])).toContain("None of the 2 selected objects");
  });

  it("an empty selection is not a refusal", () => {
    expect(noEditorReason([])).toBeNull();
    expect(commonRoutes([])).toHaveLength(0);
  });
});
