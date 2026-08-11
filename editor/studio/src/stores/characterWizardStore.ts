/**
 * New Character from Template wizard state (P24.5).
 *
 * Three steps — pick a plan, shape it, done — and the "shape it" step is a live
 * loop: every edit re-runs `character.preview`, which generates the rig AND all
 * three clips in the backend and hands back the numbers read off them. Nothing
 * in this file predicts what the backend will produce; it only shows what the
 * backend already produced.
 *
 * The pure functions below (`characterWizardReducer`, `defaultSpec`,
 * `projectRig`, `specIssue`) own the logic and are unit-tested without a
 * backend, the `terrainImportStore` pattern.
 *
 * UNITS: every length that crosses IPC is metres and every angle is degrees at
 * the authoring boundary (units doctrine). Nothing here scales.
 */
import { create } from "zustand";

import type { BodyPlanName } from "../lib/ipc";
import type { CharacterCreateDto } from "../bindings/CharacterCreateDto";
import type { CharacterJointDto } from "../bindings/CharacterJointDto";
import type { CharacterRigDto } from "../bindings/CharacterRigDto";
import type { CharacterSpecDto } from "../bindings/CharacterSpecDto";
import { character as characterIpc } from "../lib/ipc";

/** Where the wizard is. */
export type CharacterWizardStep = "plan" | "shape" | "done";

/** The part of the wizard the reducer owns (everything a test needs). */
export interface CharacterWizardMachine {
  step: CharacterWizardStep;
  spec: CharacterSpecDto;
  /** The last successful preview, or `null` before the first one lands. */
  rig: CharacterRigDto | null;
  /** The generator's own message when the current spec is not buildable. */
  refusal: string | null;
  /** Add the actor to the open level when the character is created. */
  addToScene: boolean;
  result: CharacterCreateDto | null;
  /** A command failed outright (not a spec refusal — those live in `refusal`). */
  error: string | null;
  busy: boolean;
}

/**
 * The spec the wizard opens with — the backend generator's own defaults,
 * duplicated here because a wizard that had to round-trip before it could draw
 * its first frame would flash an empty dialog.
 *
 * `the_wizard_defaults_match_the_backends` (in the Rust command tests) is not a
 * thing that can exist across a language boundary, so this is instead pinned the
 * only way it can be: the first `preview` call replaces `rig` with whatever the
 * backend actually made from these numbers, and every readout comes from that.
 * A drifted default here therefore shows the user a different-but-honest
 * character, never a wrong readout.
 */
export function defaultSpec(): CharacterSpecDto {
  return {
    name: "Character",
    plan: "biped",
    legs: null,
    params: {
      heightM: 1.75,
      hipHeightRatio: 0.53,
      shoulderHeightRatio: 0.82,
      headHeightRatio: 0.93,
      spineSegments: 2,
      neckSegments: 1,
      shoulderWidthM: 0.4,
      hipWidthM: 0.22,
      upperLimbRatio: 0.52,
      armLengthRatio: 0.42,
      bodyLengthM: 0.9,
      headForwardM: 0.25,
    },
    gait: {
      walkCadenceHz: 0.9,
      runCadenceHz: 1.6,
      hipSwingDeg: 22,
      kneeFlexDeg: 45,
      armSwingDeg: 18,
      runStrideScale: 1.6,
      bobRatio: 0.02,
      idlePeriodS: 4,
      idleBobRatio: 0.006,
      idlePitchDeg: 2,
      keysPerCycle: 12,
    },
    meshAsset: null,
  };
}

/** The wizard's initial state. */
export function initialMachine(): CharacterWizardMachine {
  return {
    step: "plan",
    spec: defaultSpec(),
    rig: null,
    refusal: null,
    addToScene: true,
    result: null,
    error: null,
    busy: false,
  };
}

/**
 * What is locally wrong with a spec, before the backend is asked.
 *
 * Deliberately thin: the generator owns every real rule and says so by name, and
 * a second copy of "hips must be below shoulders" here would be a rule with two
 * spellings (the P22 law). What this catches is only what the backend cannot
 * *see* — an empty name, and an `npedal` with no leg count, which would be a
 * command error rather than a spec refusal.
 */
export function specIssue(spec: CharacterSpecDto): string | null {
  if (!spec.name.trim()) return "A character needs a name.";
  if (spec.plan === "npedal") {
    const legs = spec.legs ?? 0;
    if (legs < 2 || legs > 64) return "An N-pedal body needs 2–64 legs.";
    if (legs % 2 !== 0) return "A body plan needs an even leg count.";
  }
  return null;
}

/** A projected joint of the rig diagram: a point in a `[0,1]²` box. */
export interface ProjectedJoint {
  name: string;
  x: number;
  y: number;
  parent: number | null;
}

/**
 * Project a previewed rig front-on into a unit box, for the SVG diagram.
 *
 * The **same simplification the Skeleton Editor's `projectRig` makes, and the
 * same bound**: rest-pose globals are the running sum of `translation` down the
 * `parent` chain, so joint rotations and scales are NOT composed. A rig whose
 * bind pose carries rotations draws slightly wrong. That is deliberate here for
 * the reason it is deliberate there — the cost is a bone at the wrong angle in a
 * diagram, not a deformed character — and it is doubly safe for the wizard,
 * whose rigs come from `build_template`, which emits identity rotations by
 * construction.
 */
export function projectRig(joints: CharacterJointDto[]): ProjectedJoint[] {
  const world: { x: number; y: number }[] = [];
  for (const j of joints) {
    const base = j.parent == null ? { x: 0, y: 0 } : world[j.parent];
    world.push({
      x: base.x + j.translation[0],
      y: base.y + j.translation[1],
    });
  }
  if (world.length === 0) return [];
  const xs = world.map((p) => p.x);
  const ys = world.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  // One scale for both axes so the rig is not stretched into its box; the
  // narrow axis is centred.
  const span = Math.max(maxX - minX, maxY - minY, 1e-6);
  const offX = (span - (maxX - minX)) / 2;
  const offY = (span - (maxY - minY)) / 2;
  return joints.map((j, i) => ({
    name: j.name,
    parent: j.parent,
    x: (world[i].x - minX + offX) / span,
    // SVG y grows downward and the world's grows up.
    y: 1 - (world[i].y - minY + offY) / span,
  }));
}

/** Fold a preview answer into the machine. */
export function characterWizardReducer(
  state: CharacterWizardMachine,
  preview: { refusal: string | null; rig: CharacterRigDto | null },
): CharacterWizardMachine {
  return {
    ...state,
    // A refusal keeps the LAST good rig on screen rather than blanking the
    // diagram: a slider passing through an invalid value would otherwise make
    // the whole preview flicker away and back.
    rig: preview.rig ?? state.rig,
    refusal: preview.refusal,
  };
}

interface CharacterWizardState extends CharacterWizardMachine {
  reset: () => void;
  choosePlan: (plan: BodyPlanName, legs?: number) => Promise<void>;
  setName: (name: string) => void;
  setParam: (key: keyof CharacterSpecDto["params"], value: number) => void;
  setGait: (key: keyof CharacterSpecDto["gait"], value: number) => void;
  setMesh: (assetId: string | null) => void;
  setAddToScene: (on: boolean) => void;
  refresh: () => Promise<void>;
  createCharacter: () => Promise<CharacterCreateDto | null>;
}

export const useCharacterWizardStore = create<CharacterWizardState>((set, get) => ({
  ...initialMachine(),

  reset: () => set(initialMachine()),

  choosePlan: async (plan, legs) => {
    set((s) => ({
      step: "shape",
      spec: { ...s.spec, plan, legs: plan === "npedal" ? (legs ?? 6) : null },
    }));
    await get().refresh();
  },

  setName: (name) => set((s) => ({ spec: { ...s.spec, name } })),

  setParam: (key, value) =>
    set((s) => ({ spec: { ...s.spec, params: { ...s.spec.params, [key]: value } } })),

  setGait: (key, value) =>
    set((s) => ({ spec: { ...s.spec, gait: { ...s.spec.gait, [key]: value } } })),

  setMesh: (assetId) => set((s) => ({ spec: { ...s.spec, meshAsset: assetId } })),

  setAddToScene: (addToScene) => set({ addToScene }),

  refresh: async () => {
    const spec = get().spec;
    const local = specIssue(spec);
    if (local) {
      set({ refusal: local });
      return;
    }
    try {
      const preview = await characterIpc.preview(spec);
      set((s) => characterWizardReducer(s, preview));
    } catch (e) {
      set({ error: String(e) });
    }
  },

  createCharacter: async () => {
    const { spec, addToScene } = get();
    const local = specIssue(spec);
    if (local) {
      set({ refusal: local });
      return null;
    }
    set({ busy: true, error: null });
    try {
      const result = await characterIpc.create(spec, { addToScene });
      set({ busy: false, result, step: "done" });
      return result;
    } catch (e) {
      set({ busy: false, error: String(e) });
      return null;
    }
  },
}));
