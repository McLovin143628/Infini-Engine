/**
 * Local, optimistic twin of the backend edit door (`inf_graph::edits`). Applied
 * to a cloned graph for instant canvas feedback; the backend re-validates
 * authoritatively and the `graph://sync` event reconciles. Structural only —
 * mirrors `apply_edit` for the edit kinds the canvas produces.
 */
import type { BpEdit, BpGraph, BpNode } from "../../lib/blueprintTypes";

/** Would connecting `from → to` create a cycle? Walk upstream from `from`. */
export function wouldCycle(g: BpGraph, from: number, to: number): boolean {
  if (from === to) return true;
  const stack = [from];
  const seen = new Set<number>();
  while (stack.length) {
    const n = stack.pop() as number;
    if (n === to) return true;
    if (seen.has(n)) continue;
    seen.add(n);
    for (const l of g.links) if (l.to === n) stack.push(l.from);
  }
  return false;
}

/** Apply one edit to `g` in place, returning whether it took effect. */
export function applyEditLocal(g: BpGraph, edit: BpEdit): boolean {
  switch (edit.kind) {
    case "add-node": {
      const key = String(edit.id);
      if (g.nodes[key]) return false;
      const node: BpNode = {
        id: edit.id,
        typeId: edit.typeId,
        params: edit.params ?? {},
        disabled: false,
        ui: { x: edit.x, y: edit.y, title: "", w: 0, h: 0 },
      };
      g.nodes[key] = node;
      g.nextId = Math.max(g.nextId, edit.id + 1);
      return true;
    }
    case "remove-node": {
      const key = String(edit.id);
      if (!g.nodes[key]) return false;
      delete g.nodes[key];
      g.links = g.links.filter((l) => l.from !== edit.id && l.to !== edit.id);
      return true;
    }
    case "connect": {
      const { link } = edit;
      if (!g.nodes[String(link.from)] || !g.nodes[String(link.to)]) return false;
      if (link.from === link.to || wouldCycle(g, link.from, link.to)) return false;
      // Replace-on-connect: one link per input port.
      g.links = g.links.filter((l) => !(l.to === link.to && l.toPort === link.toPort));
      g.links.push(link);
      return true;
    }
    case "disconnect": {
      const before = g.links.length;
      g.links = g.links.filter(
        (l) =>
          !(
            l.from === edit.link.from &&
            l.fromPort === edit.link.fromPort &&
            l.to === edit.link.to &&
            l.toPort === edit.link.toPort
          ),
      );
      return g.links.length !== before;
    }
    case "set-param": {
      const node = g.nodes[String(edit.id)];
      if (!node) return false;
      node.params[edit.name] = edit.value;
      return true;
    }
    case "move-node": {
      const node = g.nodes[String(edit.id)];
      if (!node) return false;
      node.ui.x = edit.x;
      node.ui.y = edit.y;
      return true;
    }
    case "set-title": {
      const node = g.nodes[String(edit.id)];
      if (!node) return false;
      node.ui.title = edit.title;
      return true;
    }
    default:
      return false;
  }
}

export function applyEditsLocal(g: BpGraph, edits: BpEdit[]): void {
  for (const e of edits) applyEditLocal(g, e);
}
