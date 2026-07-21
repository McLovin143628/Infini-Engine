/** Material wire + category colors. Vectors are `Named("vec2f"|"vec3f"|"vec4f")`
 *  on the wire; scalars are `float`. */
import type { PortType } from "../../lib/blueprintTypes";

export function matPinColor(ty: PortType): string {
  if (ty.kind === "float") return "#7ee06e";
  if (ty.kind === "wildcard") return "#9aa0a6";
  if (ty.kind === "named") {
    switch (ty.name) {
      case "vec2f":
        return "#1abc9c";
      case "vec3f":
        return "#5b9bd5";
      case "vec4f":
        return "#b57ce0";
      case "slab":
        return "#e08a4a";
      default:
        return "#5b9bd5";
    }
  }
  return "#9aa0a6";
}

export function matCategoryColor(category: string): string {
  switch (category) {
    case "input":
      return "#3a6ea5";
    case "constants":
      return "#5a5f6a";
    case "math":
      return "#7a5230";
    case "vector":
      return "#2f6f5a";
    case "procedural":
      return "#6a4a7a";
    case "texture":
      return "#8a5a2a";
    case "slab":
      return "#8a5a3a";
    case "output":
      return "#8a3a3a";
    default:
      return "#4a5058";
  }
}
