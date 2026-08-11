// @vitest-environment jsdom
//
// New Character wizard (P24.5): the pure projection + the local pre-flight check
// + the store actions that drive them. IPC is mocked — nothing here generates a
// rig or a clip; that is Ring 0's (`inf_anim::locomotion`) and Ring 1's
// (`inf_editor_core::character`), both of which have their own batteries.
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  character: {
    preview: vi.fn(),
    create: vi.fn(),
    folder: vi.fn(),
  },
}));

import type { CharacterRigDto } from "../../bindings/CharacterRigDto";
import { character as characterIpc } from "../../lib/ipc";
import {
  characterWizardReducer,
  defaultSpec,
  initialMachine,
  projectRig,
  specIssue,
  useCharacterWizardStore,
} from "../characterWizardStore";

const RIG: CharacterRigDto = {
  joints: [
    { name: "hips", parent: null, translation: [0, 0.9, 0] },
    { name: "chest", parent: 0, translation: [0, 0.5, 0] },
    { name: "upper_leg_l", parent: 0, translation: [-0.1, 0, 0] },
  ],
  sockets: ["foot_l", "head"],
  limits: 4,
  heightM: 1.75,
  legs: [{ name: "upper_leg_l", lengthM: 0.9, phase: 0 }],
  durationsS: [4, 1.11, 0.62],
  walkSpeedMS: 0.6,
  runSpeedMS: 1.6,
  walkThresholdMS: 0.15,
  runThresholdMS: 1.1,
  bodyVertices: 480,
  bodyTriangles: 240,
};

beforeEach(() => {
  vi.clearAllMocks();
  useCharacterWizardStore.setState(initialMachine());
});

describe("specIssue", () => {
  it("passes the default spec", () => {
    expect(specIssue(defaultSpec())).toBeNull();
  });

  it("catches only what the backend cannot see", () => {
    expect(specIssue({ ...defaultSpec(), name: "  " })).toMatch(/name/i);
    // An N-pedal with no leg count would be a COMMAND error, not a spec
    // refusal, so it is caught here.
    expect(specIssue({ ...defaultSpec(), plan: "npedal", legs: null })).toMatch(/2–64/);
    expect(specIssue({ ...defaultSpec(), plan: "npedal", legs: 5 })).toMatch(/even/);
    expect(specIssue({ ...defaultSpec(), plan: "npedal", legs: 6 })).toBeNull();
    // …and a proportion the GENERATOR owns is NOT second-guessed here: hips
    // above shoulders is `TemplateError::InvertedBody`, and a second spelling of
    // that rule in the frontend is exactly the drift the P22 law is about.
    expect(
      specIssue({
        ...defaultSpec(),
        params: { ...defaultSpec().params, hipHeightRatio: 0.99 },
      }),
    ).toBeNull();
  });
});

describe("projectRig", () => {
  it("sums translations down the parent chain and flips Y for SVG", () => {
    const p = projectRig(RIG.joints);
    expect(p).toHaveLength(3);
    // The chest is above the hips in the world, so it is ABOVE them in SVG too
    // (smaller y).
    const hips = p.find((j) => j.name === "hips")!;
    const chest = p.find((j) => j.name === "chest")!;
    expect(chest.y).toBeLessThan(hips.y);
    // The left leg is at −x, so it lands left of the hips.
    expect(p.find((j) => j.name === "upper_leg_l")!.x).toBeLessThan(hips.x);
    // Everything is inside the unit box.
    for (const j of p) {
      expect(j.x).toBeGreaterThanOrEqual(0);
      expect(j.x).toBeLessThanOrEqual(1);
      expect(j.y).toBeGreaterThanOrEqual(0);
      expect(j.y).toBeLessThanOrEqual(1);
    }
  });

  it("uses ONE scale for both axes, so a tall rig is not squashed into a square", () => {
    const p = projectRig(RIG.joints);
    const xs = p.map((j) => j.x);
    // The rig is 1.4 m tall and 0.1 m wide, so its projected width must be a
    // small fraction of the box — a per-axis normalization would stretch it to
    // the full width and make every rig look identical.
    expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(0.2);
  });

  it("survives an empty rig", () => {
    expect(projectRig([])).toEqual([]);
  });
});

describe("characterWizardReducer", () => {
  it("adopts a good rig", () => {
    const next = characterWizardReducer(initialMachine(), { refusal: null, rig: RIG });
    expect(next.rig).toBe(RIG);
    expect(next.refusal).toBeNull();
  });

  it("keeps the LAST good rig on screen when a spec is refused", () => {
    const good = characterWizardReducer(initialMachine(), { refusal: null, rig: RIG });
    const bad = characterWizardReducer(good, {
      refusal: "the body's heights must ascend",
      rig: null,
    });
    // The diagram must not blank out while a slider passes through an invalid
    // value — it would flicker away and back on every drag.
    expect(bad.rig).toBe(RIG);
    expect(bad.refusal).toMatch(/ascend/);
  });
});

describe("the store", () => {
  it("choosing a plan advances the step and previews it", async () => {
    vi.mocked(characterIpc.preview).mockResolvedValue({ refusal: null, rig: RIG });
    await useCharacterWizardStore.getState().choosePlan("quadruped");
    const s = useCharacterWizardStore.getState();
    expect(s.step).toBe("shape");
    expect(s.spec.plan).toBe("quadruped");
    expect(s.spec.legs).toBeNull();
    expect(s.rig).toBe(RIG);
    expect(characterIpc.preview).toHaveBeenCalledTimes(1);
  });

  it("an N-pedal plan carries a leg count", async () => {
    vi.mocked(characterIpc.preview).mockResolvedValue({ refusal: null, rig: RIG });
    await useCharacterWizardStore.getState().choosePlan("npedal", 8);
    expect(useCharacterWizardStore.getState().spec.legs).toBe(8);
  });

  it("a locally-refused spec never reaches the backend", async () => {
    useCharacterWizardStore.setState({ spec: { ...defaultSpec(), name: "" } });
    await useCharacterWizardStore.getState().refresh();
    expect(characterIpc.preview).not.toHaveBeenCalled();
    expect(useCharacterWizardStore.getState().refusal).toMatch(/name/i);
    // …and neither does a create.
    expect(await useCharacterWizardStore.getState().createCharacter()).toBeNull();
    expect(characterIpc.create).not.toHaveBeenCalled();
  });

  it("creating sends the whole spec and the add-to-scene choice", async () => {
    const made = {
      skeleton: "s",
      mesh: "m",
      idle: "i",
      walk: "w",
      run: "r",
      machine: "sm",
      actor: "a",
      mannequin: true,
      fit: null,
      weights: null,
      warnings: [],
    };
    vi.mocked(characterIpc.create).mockResolvedValue(made);
    useCharacterWizardStore.setState({ addToScene: false });
    const out = await useCharacterWizardStore.getState().createCharacter();
    expect(out).toBe(made);
    expect(characterIpc.create).toHaveBeenCalledWith(defaultSpec(), { addToScene: false });
    const s = useCharacterWizardStore.getState();
    expect(s.step).toBe("done");
    expect(s.busy).toBe(false);
  });

  it("a failed create leaves an error and clears busy", async () => {
    vi.mocked(characterIpc.create).mockRejectedValue("the disk went away");
    expect(await useCharacterWizardStore.getState().createCharacter()).toBeNull();
    const s = useCharacterWizardStore.getState();
    expect(s.busy).toBe(false);
    expect(s.error).toMatch(/disk/);
    expect(s.step).not.toBe("done");
  });
});
