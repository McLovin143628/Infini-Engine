import { describe, expect, it } from "vitest";
import type { SpawnKind } from "../../bindings/SpawnKind";
import { SPAWNABLES, SPAWNABLE_SECTIONS, spawnableByKind } from "../spawnables";

/** The full `SpawnKind` union the backend accepts (mirror of the ts-rs binding).
 *  If the backend adds a kind, this list — and the catalog — must grow with it. */
const ALL_KINDS: SpawnKind[] = [
  "empty",
  "cube",
  "sphere",
  "plane",
  "cylinder",
  "cone",
  "directional_light",
  "point_light",
  "spot_light",
  "camera",
  "sprite",
  "tilemap",
  "text2d",
  "nine_slice",
  "light2d",
  "terrain",
  "trigger_volume",
  "blocking_volume",
];

describe("spawnables catalog", () => {
  it("covers every SpawnKind exactly once", () => {
    const kinds = SPAWNABLES.map((s) => s.kind);
    expect(new Set(kinds).size).toBe(kinds.length); // no duplicates
    expect(new Set(kinds)).toEqual(new Set(ALL_KINDS)); // full coverage
  });

  it("gives every entry a non-empty label", () => {
    for (const s of SPAWNABLES) expect(s.label.trim()).not.toBe("");
  });

  it("flattens the sections in order", () => {
    const flat = SPAWNABLE_SECTIONS.flatMap((sec) => sec.items);
    expect(flat).toEqual(SPAWNABLES);
  });

  it("leaves the first (primitives) section unheaded and heads the rest", () => {
    expect(SPAWNABLE_SECTIONS[0].heading).toBeUndefined();
    for (const sec of SPAWNABLE_SECTIONS.slice(1)) expect(sec.heading).toBeTruthy();
  });

  it("looks up by kind", () => {
    expect(spawnableByKind("cube")?.label).toBe("Cube");
    expect(spawnableByKind("point_light")?.label).toBe("Point Light");
  });

  it("exposes the trigger/blocking volumes in a Volumes section (E-P4)", () => {
    const volumes = SPAWNABLE_SECTIONS.find((s) => s.heading === "Volumes");
    expect(volumes).toBeDefined();
    expect(volumes?.items.map((i) => i.kind)).toEqual(["trigger_volume", "blocking_volume"]);
    expect(spawnableByKind("trigger_volume")?.label).toBe("Trigger Volume");
    expect(spawnableByKind("blocking_volume")?.label).toBe("Blocking Volume");
  });
});
