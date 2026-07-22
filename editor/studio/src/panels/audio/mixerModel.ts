/**
 * Pure editing logic for the Audio Mixer panel (E-P9) — no React, no IPC, so it
 * is unit-tested directly (see `__tests__/mixerModel.test.ts`).
 *
 * Mirrors the backend's `inf_editor_core::ipc::validate_mixer` rules so the
 * panel can disable Save (and surface the reason) before the round-trip. The
 * authoritative validation still runs backend-side in `mixer_save`.
 */
import type { MixerBusDto } from "../../bindings/MixerBusDto";
import type { MixerConfigDto } from "../../bindings/MixerConfigDto";
import type { MixerEffectDto } from "../../bindings/MixerEffectDto";

export type { MixerBusDto, MixerConfigDto, MixerEffectDto };

/** The undeletable, always-present root bus every voice folds through. */
export const MASTER = "master";

/**
 * First validation error, or `null` when the config is savable. Mirrors the
 * backend rules: ≥1 bus, non-empty unique names, a present + rootless `master`,
 * existing parents, no cycles.
 */
export function validateMixer(cfg: MixerConfigDto): string | null {
  const buses = cfg.buses;
  if (buses.length === 0) return "The mixer must have at least one bus.";

  const seen = new Set<string>();
  for (const b of buses) {
    if (b.name.trim() === "") return "Bus names must not be empty.";
    if (seen.has(b.name)) return `Duplicate bus name: ${b.name}`;
    seen.add(b.name);
  }

  const master = buses.find((b) => b.name === MASTER);
  if (!master) return "The master bus must exist and cannot be deleted.";
  if (master.parent != null) return "The master bus must be a root (no parent).";

  const names = new Set(buses.map((b) => b.name));
  for (const b of buses) {
    if (b.parent != null && !names.has(b.parent)) {
      return `Bus ${b.name} has an unknown parent: ${b.parent}`;
    }
  }

  const cycle = findCycle(buses);
  if (cycle) return `Routing cycle through bus: ${cycle}`;

  return null;
}

/** The bus whose parent chain loops, or `null` when the graph is acyclic. */
function findCycle(buses: MixerBusDto[]): string | null {
  const parentOf = new Map(buses.map((b) => [b.name, b.parent] as const));
  for (const start of buses) {
    const visited = new Set<string>();
    let cur: string | null | undefined = start.name;
    while (cur != null) {
      if (visited.has(cur)) return cur;
      visited.add(cur);
      cur = parentOf.get(cur);
    }
  }
  return null;
}

/** Names of every bus reachable *downward* from `root` (its descendants + itself). */
export function descendantsOf(buses: MixerBusDto[], root: string): Set<string> {
  const out = new Set<string>([root]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const b of buses) {
      if (b.parent != null && out.has(b.parent) && !out.has(b.name)) {
        out.add(b.name);
        grew = true;
      }
    }
  }
  return out;
}

/**
 * Valid parent choices for `busName`: every other bus that is not one of its own
 * descendants (which would form a cycle). `master` has no choices — it is a
 * fixed root — so this returns `[]` for it.
 */
export function validParents(buses: MixerBusDto[], busName: string): string[] {
  if (busName === MASTER) return [];
  const blocked = descendantsOf(buses, busName);
  return buses.filter((b) => !blocked.has(b.name)).map((b) => b.name);
}

/** A bus name not already used, based on `base` ("bus", "bus 2", …). */
export function uniqueBusName(buses: MixerBusDto[], base = "bus"): string {
  const names = new Set(buses.map((b) => b.name));
  if (!names.has(base)) return base;
  for (let i = 2; ; i++) {
    const candidate = `${base} ${i}`;
    if (!names.has(candidate)) return candidate;
  }
}

/** Append a new unity-volume bus parented to `master`. */
export function addBus(cfg: MixerConfigDto): MixerConfigDto {
  const name = uniqueBusName(cfg.buses);
  return {
    buses: [...cfg.buses, { name, parent: MASTER, volume: 1, effects: [] }],
  };
}

/**
 * Delete `name` (never `master`). Its direct children reparent to `master` so the
 * hierarchy stays connected and valid — the simplest Ring-0-compatible behavior
 * (chosen over rejecting delete-with-children, which dead-ends the UX).
 */
export function deleteBus(cfg: MixerConfigDto, name: string): MixerConfigDto {
  if (name === MASTER) return cfg;
  return {
    buses: cfg.buses
      .filter((b) => b.name !== name)
      .map((b) => (b.parent === name ? { ...b, parent: MASTER } : b)),
  };
}

/** Replace the bus at `index` with `next` (returns a new config). */
export function updateBus(
  cfg: MixerConfigDto,
  index: number,
  next: MixerBusDto,
): MixerConfigDto {
  const buses = cfg.buses.slice();
  buses[index] = next;
  return { buses };
}

/**
 * Rename the bus at `index`, repointing every child's `parent`. A no-op when the
 * new name is unchanged; the caller still validates (dup/empty names surface as a
 * Save-blocking error rather than being silently rejected here).
 */
export function renameBus(cfg: MixerConfigDto, index: number, newName: string): MixerConfigDto {
  const old = cfg.buses[index]?.name;
  if (old === undefined || old === newName) return cfg;
  return {
    buses: cfg.buses.map((b, i) => {
      const name = i === index ? newName : b.name;
      const parent = b.parent === old ? newName : b.parent;
      return { ...b, name, parent };
    }),
  };
}

/** The first `Gain` effect's dB, or `null` when the bus has none. */
export function gainDb(bus: MixerBusDto): number | null {
  for (const e of bus.effects) if (e.kind === "gain") return e.db;
  return null;
}

/**
 * Set (or, when `db` is null, remove) the bus's single editable Gain effect,
 * preserving any Lowpass (and other) effects in place. A new Gain is appended.
 */
export function setGainDb(bus: MixerBusDto, db: number | null): MixerBusDto {
  const hasGain = bus.effects.some((e) => e.kind === "gain");
  let effects: MixerEffectDto[];
  if (db === null) {
    effects = bus.effects.filter((e) => e.kind !== "gain");
  } else if (hasGain) {
    let replaced = false;
    effects = bus.effects.map((e) => {
      if (e.kind === "gain" && !replaced) {
        replaced = true;
        return { kind: "gain", db };
      }
      return e;
    });
  } else {
    effects = [...bus.effects, { kind: "gain", db }];
  }
  return { ...bus, effects };
}

/** The bus's Lowpass cutoffs (read-only chips), in chain order. */
export function lowpassCutoffs(bus: MixerBusDto): number[] {
  return bus.effects.flatMap((e) => (e.kind === "lowpass" ? [e.cutoff_hz] : []));
}
