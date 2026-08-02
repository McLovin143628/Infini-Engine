/** PCG wire + category colors. Wires are
 *  `Named("density"|"scatter"|"layer"|"span"|"rules"|"building")` — every one of
 *  them a `Named`, because the graph substrate is deliberately domain-free. */
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
      case "span":
        return "#e07ec1"; // grammar span (magenta) — P19.4
      case "rules":
        return "#c9a0ff"; // grammar rule table (violet) — P19.4
      case "building":
        return "#e0a06e"; // building archetype (terracotta) — P19.5
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
    case "grammar":
      return "#6a2f6f";
    case "building":
      return "#7a3550";
    case "layer":
      return "#8a7a2a";
    case "output":
      return "#8a3a3a";
    default:
      return "#4a5058";
  }
}
