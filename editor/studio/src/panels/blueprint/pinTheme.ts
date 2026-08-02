/** Pin/wire colors by port type. Exec pins are the neutral white flow pins;
 *  data pins take a hue per type (Unreal-ish). */
import type { PortType } from "../../lib/blueprintTypes";

export function pinColor(ty: PortType): string {
  switch (ty.kind) {
    case "exec":
      return "#e8e8ec";
    case "bool":
      return "#c0392b";
    case "int":
      return "#1abc9c";
    case "float":
      return "#7ee06e";
    case "str":
      return "#d17ce0";
    case "wildcard":
      return "#9aa0a6";
    case "named":
      return "#5b9bd5";
    default:
      return "#9aa0a6";
  }
}

export function categoryColor(category: string): string {
  switch (category) {
    case "events":
      return "#c0392b";
    case "flow":
      return "#7f8c8d";
    case "math":
    case "compare":
    case "logic":
      return "#2e7d5b";
    case "variables":
      return "#2c6fbb";
    case "literals":
      return "#8e7cc3";
    case "engine":
      return "#b8860b";
    case "debug":
      return "#555b66";
    // PCG categories reach this map through the shared NodePalette's group dot.
    case "grammar":
      return "#6a2f6f";
    default:
      return "#4a5058";
  }
}
