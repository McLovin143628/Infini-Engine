/**
 * The GIS Import wizard's judgement, unit-tested without a backend (IB-3).
 *
 * Everything the wizard decides for itself is in the pure functions here; the
 * store is a thin shell over them. What it must NOT decide is anything about
 * the import — that is `inf_gis::import`, and the arms below check the wizard
 * refuses the requests the door would refuse rather than duplicating the door.
 */
import { describe, expect, it } from "vitest";

import type { GisImportSettingsDto } from "../../bindings/GisImportSettingsDto";
import type { GisProbeDto } from "../../bindings/GisProbeDto";
import {
  capNote,
  describeCrs,
  gisSettingsIssue,
  GIS_ISLAND_MAX_ENTITIES,
  initialGisMachine,
  kindsFor,
} from "../gisImportStore";

function probe(over: Partial<GisProbeDto> = {}): GisProbeDto {
  return {
    path: "C:/data/roads.shp",
    format: "shapefile",
    layer_name: "roads",
    features: 4200,
    points: 0,
    polylines: 4200,
    polygons: 0,
    dominant_kind: "polyline",
    fields: [],
    crs: {
      spec: "EPSG:26910",
      origin: "prj",
      name: "NAD83 / UTM zone 10N",
      vertical_unit_m: 1,
      prj_wkt: null,
    },
    centre_lat: 49.28,
    centre_lon: -123.12,
    suggested_anchor_epsg: 32610,
    level_anchor_crs: "EPSG:32610",
    skipped: [],
    advisories: [],
    ...over,
  };
}

function settings(over: Partial<GisImportSettingsDto> = {}): GisImportSettingsDto {
  return {
    kind: "roads",
    source_crs: "",
    vertical_unit_m: 0,
    max_entities: 4096,
    min_length_m: 8,
    reverse_flow: false,
    name_prefix: "",
    road_surface: true,
    road_lift_m: 0.02,
    road_ground_step_m: 1,
    biome_terrain: "",
    biome_attribute: "",
    biome_classes: 8,
    buildings: false,
    max_buildings: 512,
    furnish: false,
    ...over,
  };
}

describe("gisSettingsIssue", () => {
  it("accepts a well-formed request", () => {
    expect(gisSettingsIssue(probe(), settings())).toBeNull();
  });

  it("refuses a level with no geo-anchor, and says why", () => {
    const issue = gisSettingsIssue(probe({ level_anchor_crs: null }), settings());
    expect(issue).toMatch(/geo-anchor/);
    expect(issue).toMatch(/World Settings/);
  });

  it("refuses a cap the door would refuse", () => {
    expect(gisSettingsIssue(probe(), settings({ max_entities: 0 }))).toMatch(/at least 1/);
    expect(
      gisSettingsIssue(probe(), settings({ max_entities: GIS_ISLAND_MAX_ENTITIES + 1 })),
    ).toMatch(/tops out/);
    expect(gisSettingsIssue(probe(), settings({ min_length_m: -1 }))).toMatch(/zero or more/);
  });

  it("refuses an extra that does not belong to its target kind", () => {
    expect(
      gisSettingsIssue(probe(), settings({ kind: "lakes", road_surface: true })),
    ).toMatch(/road surface/);
    expect(
      gisSettingsIssue(probe(), settings({ kind: "roads", buildings: true })),
    ).toMatch(/footprint or parcel/);
    expect(
      gisSettingsIssue(probe(), settings({ kind: "roads", biome_terrain: "abc" })),
    ).toMatch(/land cover/);
  });

  it("refuses a file with nothing in it", () => {
    expect(gisSettingsIssue(probe({ features: 0 }), settings())).toMatch(/no features/);
    expect(gisSettingsIssue(null, null)).toMatch(/pick a vector file/);
  });
});

describe("capNote — IB-14's missing half", () => {
  it("says what the cap will drop, BEFORE the import, and names the number to raise it to", () => {
    const note = capNote(probe({ features: 10000 }), settings({ max_entities: 4096 }));
    expect(note).toMatch(/5904 of 10000/);
    expect(note).toMatch(/NOT be imported/);
    expect(note).toMatch(/Raise it to 10000/);
  });

  it("is silent when the whole layer fits", () => {
    expect(capNote(probe({ features: 100 }), settings({ max_entities: 4096 }))).toBeNull();
    expect(capNote(probe({ features: 4096 }), settings({ max_entities: 4096 }))).toBeNull();
  });
});

describe("describeCrs", () => {
  it("says where a CRS came from, and calls a name match a guess", () => {
    expect(describeCrs(probe())).toBe("EPSG:26910 (from its .prj — NAD83 / UTM zone 10N)");
    const guessed = probe({
      crs: { ...probe().crs, origin: "prj-name" },
    });
    expect(describeCrs(guessed)).toMatch(/GUESSED/);
    const stated = probe({
      crs: { ...probe().crs, origin: "stated", name: null },
    });
    expect(describeCrs(stated)).toBe("EPSG:26910 (as stated)");
  });
});

describe("kindsFor", () => {
  it("offers the kinds a layer's geometry can actually be", () => {
    expect(kindsFor("polyline")[0]).toBe("roads");
    expect(kindsFor("polygon")[0]).toBe("buildings");
    expect(kindsFor("polyline")).not.toContain("buildings");
    expect(kindsFor("polygon")).not.toContain("roads");
    expect(kindsFor("point")).toEqual(["generic"]);
  });
});

describe("initialGisMachine", () => {
  it("starts at the file picker with nothing in it", () => {
    const m = initialGisMachine();
    expect(m.step).toBe("pick");
    expect(m.probe).toBeNull();
    expect(m.settings).toBeNull();
    expect(m.result).toBeNull();
    expect(m.busy).toBe(false);
  });
});
