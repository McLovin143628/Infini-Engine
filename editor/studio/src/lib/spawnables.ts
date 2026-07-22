/**
 * The shared spawnable-actor catalog: every entity type the editor can create
 * through the `scene_create` command (`SpawnKind`). Both the Outliner's
 * "+ Add" menu and the Place Actors palette read this list so the two never
 * drift. Pure data (kind + label) — no React/lucide imports — so it stays
 * trivially unit-testable and the panels own their own iconography.
 */
import type { SpawnKind } from "../bindings/SpawnKind";

export interface SpawnableItem {
  kind: SpawnKind;
  label: string;
}

export interface SpawnableSection {
  /** Optional group heading; the first (primitives) section is unheaded. */
  heading?: string;
  items: SpawnableItem[];
}

export const SPAWNABLE_SECTIONS: SpawnableSection[] = [
  {
    items: [
      { kind: "cube", label: "Cube" },
      { kind: "sphere", label: "Sphere" },
      { kind: "plane", label: "Plane" },
      { kind: "cylinder", label: "Cylinder" },
      { kind: "cone", label: "Cone" },
      { kind: "terrain", label: "Terrain" },
    ],
  },
  {
    heading: "Lights & Camera",
    items: [
      { kind: "point_light", label: "Point Light" },
      { kind: "directional_light", label: "Directional Light" },
      { kind: "spot_light", label: "Spot Light" },
      { kind: "camera", label: "Camera" },
    ],
  },
  {
    heading: "2D",
    items: [
      { kind: "sprite", label: "Sprite" },
      { kind: "tilemap", label: "Tilemap" },
      { kind: "text2d", label: "Text" },
      { kind: "nine_slice", label: "Nine-Slice" },
      { kind: "light2d", label: "2D Light" },
    ],
  },
  { heading: "Other", items: [{ kind: "empty", label: "Empty (Folder)" }] },
];

/** Flat list of every spawnable, in section order. */
export const SPAWNABLES: SpawnableItem[] = SPAWNABLE_SECTIONS.flatMap((s) => s.items);

/** Look up a spawnable by kind. */
export function spawnableByKind(kind: SpawnKind): SpawnableItem | undefined {
  return SPAWNABLES.find((s) => s.kind === kind);
}
