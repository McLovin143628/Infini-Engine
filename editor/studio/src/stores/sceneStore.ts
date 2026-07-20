/**
 * Placeholder scene state for the Phase 1 Outliner/Details panels: a mock
 * actor tree with selection + visibility. Phase 3 replaces this with the
 * live ECS world binding (diff-based snapshot IPC, selection service) —
 * the component-facing shape here anticipates that so the panels don't
 * need rewriting, only the data source.
 */
import { create } from "zustand";
import { registerBridgedStore } from "../panels/window/storeBridge";

export interface MockActor {
  id: string;
  name: string;
  /** UE-style type column text ("Static Mesh Actor", "Point Light", …). */
  type: string;
  visible: boolean;
  children: MockActor[];
}

function actor(id: string, name: string, type: string, children: MockActor[] = []): MockActor {
  return { id, name, type, visible: true, children };
}

const MOCK_WORLD: MockActor[] = [
  actor("world", "Untitled Level", "World", [
    actor("floor", "Floor", "Static Mesh Actor"),
    actor("player-start", "PlayerStart", "Player Start"),
    actor("lighting", "Lighting", "Folder", [
      actor("sun", "DirectionalLight", "Directional Light"),
      actor("sky", "SkyLight", "Sky Light"),
      actor("fill", "PointLight_Fill", "Point Light"),
    ]),
    actor("props", "Props", "Folder", [
      actor("cube-1", "Cube", "Static Mesh Actor"),
      actor("cube-2", "Cube2", "Static Mesh Actor"),
      actor("sphere", "Sphere", "Static Mesh Actor"),
    ]),
    actor("camera", "CineCamera", "Cine Camera Actor"),
  ]),
];

interface SceneState {
  roots: MockActor[];
  selectedIds: string[];
  /** Collapsed outliner rows (ids). Everything starts expanded. */
  collapsed: string[];
  select: (ids: string[], additive?: boolean) => void;
  toggleVisible: (id: string) => void;
  toggleCollapsed: (id: string) => void;
}

function mapTree(list: MockActor[], fn: (a: MockActor) => MockActor): MockActor[] {
  return list.map((a) => fn({ ...a, children: mapTree(a.children, fn) }));
}

export const useSceneStore = create<SceneState>((set) => ({
  roots: MOCK_WORLD,
  selectedIds: [],
  collapsed: [],
  select: (ids, additive = false) =>
    set((s) => ({
      selectedIds: additive
        ? [...new Set([...s.selectedIds, ...ids])]
        : ids,
    })),
  toggleVisible: (id) =>
    set((s) => ({
      roots: mapTree(s.roots, (a) => (a.id === id ? { ...a, visible: !a.visible } : a)),
    })),
  toggleCollapsed: (id) =>
    set((s) => ({
      collapsed: s.collapsed.includes(id)
        ? s.collapsed.filter((c) => c !== id)
        : [...s.collapsed, id],
    })),
}));

registerBridgedStore("scene", useSceneStore);

/** Depth-first flatten honoring collapse state; used by the Outliner. */
export function flattenTree(
  roots: MockActor[],
  collapsed: string[],
): Array<{ actor: MockActor; depth: number }> {
  const out: Array<{ actor: MockActor; depth: number }> = [];
  const walk = (list: MockActor[], depth: number) => {
    for (const a of list) {
      out.push({ actor: a, depth });
      if (!collapsed.includes(a.id)) walk(a.children, depth + 1);
    }
  };
  walk(roots, 0);
  return out;
}

/** Find an actor anywhere in the tree. */
export function findActor(roots: MockActor[], id: string): MockActor | null {
  for (const a of roots) {
    if (a.id === id) return a;
    const hit = findActor(a.children, id);
    if (hit) return hit;
  }
  return null;
}
