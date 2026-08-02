/** PCG wire + category colors. Wires are `Named("density"|"scatter"|"layer")`. */
import type { PortType } from "../../lib/blueprintTypes";

export function pcgPinColor(ty: PortType): string {
  if (ty.kind === "named") {
    switch (ty.name) {
      case "density":
        return "#6ec1e0"; // density field (blue)
      case "scatter":
        return "#7ee06e"; // scatter pass (green)
      case "layer":
        return "#e0b36e"; // layer group (gold) — P19.3
      default:
        return "#9aa0a6";
    }
  }
  if (ty.kind === "float") return "#7ee06e";
  return "#9aa0a6";
}

export function pcgCategoryColor(category: string): string {
  switch (category) {
    case "sources":
      return "#3a6ea5";
    case "filters":
      return "#2f6f5a";
    case "masks":
      return "#3a7a7a";
    case "combine":
      return "#6a4a7a";
    case "scatter":
      return "#7a5230";
    case "layer":
      return "#8a7a2a";
    case "output":
      return "#8a3a3a";
    default:
      return "#4a5058";
  }
}
