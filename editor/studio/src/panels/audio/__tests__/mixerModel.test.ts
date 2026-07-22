import { describe, expect, it } from "vitest";

import type { MixerBusDto } from "../../../bindings/MixerBusDto";
import type { MixerConfigDto } from "../../../bindings/MixerConfigDto";
import {
  MASTER,
  addBus,
  deleteBus,
  descendantsOf,
  gainDb,
  lowpassCutoffs,
  renameBus,
  setGainDb,
  uniqueBusName,
  validParents,
  validateMixer,
} from "../mixerModel";

const bus = (name: string, parent: string | null = MASTER, volume = 1): MixerBusDto => ({
  name,
  parent,
  volume,
  effects: [],
});

const cfg = (buses: MixerBusDto[]): MixerConfigDto => ({ buses });

const DEFAULT = cfg([bus("master", null), bus("sfx"), bus("music")]);

describe("validateMixer", () => {
  it("accepts the default hierarchy", () => {
    expect(validateMixer(DEFAULT)).toBeNull();
  });

  it("rejects an empty mixer", () => {
    expect(validateMixer(cfg([]))).toMatch(/at least one/);
  });

  it("rejects an empty bus name", () => {
    expect(validateMixer(cfg([bus("master", null), bus("  ")]))).toMatch(/empty/);
  });

  it("rejects duplicate names", () => {
    expect(validateMixer(cfg([bus("master", null), bus("sfx"), bus("sfx")]))).toMatch(/Duplicate/);
  });

  it("requires a master bus", () => {
    expect(validateMixer(cfg([bus("sfx", null)]))).toMatch(/master/);
  });

  it("requires master to be a root", () => {
    expect(validateMixer(cfg([bus("root", null), bus("master", "root")]))).toMatch(/root/);
  });

  it("rejects an unknown parent", () => {
    expect(validateMixer(cfg([bus("master", null), bus("sfx", "ghost")]))).toMatch(/unknown parent/);
  });

  it("rejects a routing cycle", () => {
    expect(validateMixer(cfg([bus("master", null), bus("a", "b"), bus("b", "a")]))).toMatch(/cycle/);
  });
});

describe("hierarchy helpers", () => {
  it("computes descendants including the root itself", () => {
    const c = cfg([bus("master", null), bus("group"), bus("sub", "group")]);
    expect(descendantsOf(c.buses, "group")).toEqual(new Set(["group", "sub"]));
  });

  it("excludes self + descendants from valid parents (no cycles)", () => {
    const c = cfg([bus("master", null), bus("group"), bus("sub", "group")]);
    const opts = validParents(c.buses, "group");
    expect(opts).toContain("master");
    expect(opts).not.toContain("group"); // self
    expect(opts).not.toContain("sub"); // descendant
  });

  it("offers no parent choices for master (fixed root)", () => {
    expect(validParents(DEFAULT.buses, "master")).toEqual([]);
  });
});

describe("mutations", () => {
  it("adds a uniquely-named bus parented to master", () => {
    const next = addBus(DEFAULT);
    expect(next.buses).toHaveLength(4);
    const added = next.buses[3];
    expect(added.parent).toBe("master");
    expect(DEFAULT.buses.map((b) => b.name)).not.toContain(added.name);
    expect(validateMixer(next)).toBeNull();
  });

  it("derives unique names", () => {
    expect(uniqueBusName(DEFAULT.buses)).toBe("bus");
    expect(uniqueBusName(cfg([bus("master", null), bus("bus")]).buses)).toBe("bus 2");
  });

  it("deletes a bus and reparents its children to master", () => {
    const c = cfg([bus("master", null), bus("group"), bus("sub", "group")]);
    const next = deleteBus(c, "group");
    expect(next.buses.map((b) => b.name)).toEqual(["master", "sub"]);
    expect(next.buses.find((b) => b.name === "sub")?.parent).toBe("master");
    expect(validateMixer(next)).toBeNull();
  });

  it("never deletes master", () => {
    expect(deleteBus(DEFAULT, "master")).toBe(DEFAULT);
  });

  it("renames a bus and repoints its children", () => {
    const c = cfg([bus("master", null), bus("group"), bus("sub", "group")]);
    const next = renameBus(c, 1, "renamed");
    expect(next.buses[1].name).toBe("renamed");
    expect(next.buses.find((b) => b.name === "sub")?.parent).toBe("renamed");
    expect(validateMixer(next)).toBeNull();
  });
});

describe("effects", () => {
  it("adds, reads, edits, and removes a single Gain preserving Lowpass", () => {
    const withLp: MixerBusDto = {
      name: "sfx",
      parent: MASTER,
      volume: 1,
      effects: [{ kind: "lowpass", cutoff_hz: 800 }],
    };
    expect(gainDb(withLp)).toBeNull();

    const added = setGainDb(withLp, -6);
    expect(gainDb(added)).toBe(-6);
    expect(lowpassCutoffs(added)).toEqual([800]); // lowpass preserved

    const edited = setGainDb(added, -3);
    expect(gainDb(edited)).toBe(-3);
    expect(edited.effects.filter((e) => e.kind === "gain")).toHaveLength(1); // no dup

    const removed = setGainDb(edited, null);
    expect(gainDb(removed)).toBeNull();
    expect(lowpassCutoffs(removed)).toEqual([800]); // lowpass still there
  });
});
