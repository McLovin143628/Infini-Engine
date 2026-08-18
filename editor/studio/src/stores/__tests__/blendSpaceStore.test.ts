// @vitest-environment jsdom
//
// The blend-space authoring store (P29.2 clause 8). The panel owns its sample
// list and asks the ENGINE for every piece of geometry, so what there is to test
// here is the seam rather than the mathematics: that a gesture refreshes the
// preview, that the request carries the samples as placed, and that the push/pull
// pair round-trips through `smStore`'s document without inventing a second way to
// author a `.inf_sm`.
//
// The triangulation itself is asserted in Rust (`inf_anim::delaunay`), and
// deliberately not mirrored here — see `blendSpaceTypes.ts` on why a second
// implementation of a deterministic algorithm is the defect this design avoids.
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  blendSpace: { preview: vi.fn() },
  anim: { clipInfo: vi.fn(), rederive: vi.fn() },
  sm: {
    list: vi.fn(),
    create: vi.fn(),
    get: vi.fn(),
    close: vi.fn(),
    save: vi.fn(),
    listClips: vi.fn(),
    propose: vi.fn(),
  },
}));

import { anim, blendSpace, sm } from "../../lib/ipc";
import type { BlendPreviewDto } from "../../lib/blendSpaceTypes";
import type { SmDoc } from "../../lib/smTypes";
import { useBlendSpaceStore } from "../blendSpaceStore";
import { __resetSmInitForTest, useSmStore } from "../smStore";

const mockBs = blendSpace as unknown as { preview: ReturnType<typeof vi.fn> };
const mockAnim = anim as unknown as {
  clipInfo: ReturnType<typeof vi.fn>;
  rederive: ReturnType<typeof vi.fn>;
};
const mockSm = sm as unknown as {
  list: ReturnType<typeof vi.fn>;
  create: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  listClips: ReturnType<typeof vi.fn>;
};

function preview(partial: Partial<BlendPreviewDto> = {}): BlendPreviewDto {
  return {
    triangles: [[0, 1, 2]],
    hull: [
      [0, 1],
      [0, 2],
      [1, 2],
    ],
    degenerate: false,
    weights: [
      { index: 0, weight: 0.5 },
      { index: 1, weight: 0.5 },
    ],
    blendedDurationS: 0.8,
    joints: [],
    skeleton: null,
    refusal: null,
    ...partial,
  };
}

function makeDoc(): SmDoc {
  return {
    id: "sm1",
    name: "Main",
    machine: {
      states: [
        {
          name: "Locomotion",
          motion: {
            kind: "blend2d",
            paramX: "speedX",
            paramY: "speedY",
            entries: [
              { x: -1, y: 0, clip: "c1" },
              { x: 1, y: 0, clip: "c2" },
            ],
          },
          looping: true,
          speed: 1,
          x: 0,
          y: 0,
          onEnter: [],
          onExit: [],
        },
        {
          name: "Idle",
          motion: { kind: "clip", clip: null },
          looping: true,
          speed: 1,
          x: 0,
          y: 0,
          onEnter: [],
          onExit: [],
        },
      ],
      transitions: [],
      entry: 0,
      params: [],
      profiles: [],
    },
  };
}

beforeEach(async () => {
  vi.clearAllMocks();
  __resetSmInitForTest();
  useSmStore.setState({ doc: null, clips: [], ready: false, selectedTransition: null });
  mockSm.list.mockResolvedValue([makeDoc()]);
  mockSm.listClips.mockResolvedValue([
    { id: "c1", name: "Left" },
    { id: "c2", name: "Right" },
  ]);
  mockBs.preview.mockResolvedValue(preview());
  useBlendSpaceStore.setState({
    samples: [
      { x: 0, y: 0, clip: null },
      { x: 1, y: 0, clip: null },
      { x: 0, y: 1, clip: null },
    ],
    cursorX: 0.2,
    cursorY: 0.2,
    timeS: 0,
    preview: null,
    paramX: "vx",
    paramY: "vy",
    status: null,
  });
});

describe("blendSpaceStore", () => {
  it("asks the engine for the geometry, with the samples as placed", async () => {
    await useBlendSpaceStore.getState().refresh();
    expect(mockBs.preview).toHaveBeenCalledTimes(1);
    const [samples, x, y, t] = mockBs.preview.mock.calls[0];
    expect(samples).toHaveLength(3);
    expect(samples[1]).toEqual({ x: 1, y: 0, clip: null });
    expect([x, y, t]).toEqual([0.2, 0.2, 0]);
    expect(useBlendSpaceStore.getState().preview?.weights).toHaveLength(2);
  });

  it("refreshes on every gesture that can change the answer", async () => {
    const s = () => useBlendSpaceStore.getState();
    await s().addSample(0.5, 0.5);
    await s().moveSample(0, -0.2, 0.1);
    await s().setSampleClip(1, "c2");
    await s().setCursor(0.4, -0.1);
    await s().setTime(0.35);
    await s().removeSample(3);
    expect(mockBs.preview).toHaveBeenCalledTimes(6);
    // The edits really landed on the sample list.
    expect(s().samples).toHaveLength(3);
    expect(s().samples[0]).toEqual({ x: -0.2, y: 0.1, clip: null });
    expect(s().samples[1].clip).toBe("c2");
    expect(s().timeS).toBe(0.35);
  });

  it("survives a failing preview without losing the samples", async () => {
    mockBs.preview.mockRejectedValueOnce(new Error("no project"));
    await useBlendSpaceStore.getState().refresh();
    expect(useBlendSpaceStore.getState().preview).toBeNull();
    expect(useBlendSpaceStore.getState().samples).toHaveLength(3);
    expect(useBlendSpaceStore.getState().busy).toBe(false);
  });

  it("pulls a 2D blend space out of a state, and refuses anything else", async () => {
    await useSmStore.getState().init();
    const s = () => useBlendSpaceStore.getState();
    await s().pullFromState(0);
    expect(s().paramX).toBe("speedX");
    expect(s().paramY).toBe("speedY");
    expect(s().samples).toEqual([
      { x: -1, y: 0, clip: "c1" },
      { x: 1, y: 0, clip: "c2" },
    ]);
    // State 1 plays a Clip, not a blend space — a refusal, not a silent overwrite.
    const before = s().samples;
    await s().pullFromState(1);
    expect(s().samples).toBe(before);
    expect(s().status).toMatch(/does not play a 2D blend space/);
  });

  it("pushes into the machine document rather than saving a second asset", async () => {
    await useSmStore.getState().init();
    useBlendSpaceStore.setState({
      paramX: "vx",
      paramY: "vy",
      samples: [
        { x: 0, y: 0, clip: "c1" },
        { x: 1, y: 1, clip: "c2" },
      ],
    });
    useBlendSpaceStore.getState().pushToState(1);
    const motion = useSmStore.getState().doc!.machine.states[1].motion;
    expect(motion.kind).toBe("blend2d");
    if (motion.kind !== "blend2d") throw new Error("unreachable");
    expect(motion.paramX).toBe("vx");
    expect(motion.entries).toEqual([
      { x: 0, y: 0, clip: "c1" },
      { x: 1, y: 1, clip: "c2" },
    ]);
    // Nothing was written to disk by this gesture — `sm_save` is still the one
    // door, and the status says so.
    expect(useBlendSpaceStore.getState().status).toMatch(/save the machine/);
  });

  it("says so when there is no machine to push into", () => {
    useBlendSpaceStore.getState().pushToState(0);
    expect(useBlendSpaceStore.getState().status).toMatch(/no state machine/);
  });
});

// ── the derived-data pane (P29.5, pillar S2) ─────────────────────────────────
//
// What the panel shows about a clip is what the IMPORT measured on it, read back
// through one command. The seam is what is tested here — that a selection loads
// it, that clearing clears it, and that a re-derive RELOADS rather than trusting
// the report it just got, because the report describes what was written and the
// pane has to describe what is on disk.
describe("derived data", () => {
  const info = (over: Record<string, unknown> = {}) => ({
    id: "c1",
    name: "Walk",
    durationS: 1.0,
    curves: [
      { name: "MoveData_Speed", keys: 31, min: 0, max: 1.7, samples: [0, 1.7], derived: true },
      { name: "Mask_AimOffset", keys: 1, min: 1, max: 1, samples: [1, 1], derived: false },
    ],
    markers: [
      { timeS: 0.1, name: "plant_upper_leg_l", group: "foot" },
      { timeS: 0.1, name: "footstep_upper_leg_l", group: "" },
    ],
    rootMotion: { translation: [0, 0, 1.6], yawDeg: 0, distanceM: 1.6, keys: 31 },
    skeleton: "sk1",
    refusal: null,
    ...over,
  });

  it("loads a clip's derived data and clears it for an unassigned sample", async () => {
    mockAnim.clipInfo.mockResolvedValue(info());
    await useBlendSpaceStore.getState().loadDerived("c1");
    const s = useBlendSpaceStore.getState();
    expect(mockAnim.clipInfo).toHaveBeenCalledWith("c1");
    expect(s.derivedFor).toBe("c1");
    expect(s.derived?.markers).toHaveLength(2);
    expect(s.derived?.curves.filter((c) => c.derived)).toHaveLength(1);

    await useBlendSpaceStore.getState().loadDerived(null);
    expect(useBlendSpaceStore.getState().derived).toBeNull();
    expect(useBlendSpaceStore.getState().derivedFor).toBeNull();
  });

  it("a re-derive re-reads the clip rather than trusting its own report", async () => {
    mockAnim.clipInfo.mockResolvedValue(info());
    await useBlendSpaceStore.getState().loadDerived("c1");
    mockAnim.clipInfo.mockClear();

    mockAnim.rederive.mockResolvedValue({
      distanceM: 1.6,
      avgSpeedMps: 1.6,
      strideM: 0.8,
      gait: 0.97,
      riseM: 0,
      plants: 2,
      markers: 4,
      curves: ["MoveData_Speed", "W_Gait"],
      advisories: [],
      refusal: null,
    });
    mockAnim.clipInfo.mockResolvedValue(info({ durationS: 1.25 }));
    await useBlendSpaceStore.getState().rederive(true);

    expect(mockAnim.rederive).toHaveBeenCalledWith("c1", true);
    expect(mockAnim.clipInfo).toHaveBeenCalledWith("c1");
    const s = useBlendSpaceStore.getState();
    expect(s.lastDerive?.plants).toBe(2);
    expect(s.derived?.durationS).toBe(1.25);
  });

  it("a re-derive with nothing selected does nothing at all", async () => {
    await useBlendSpaceStore.getState().loadDerived(null);
    await useBlendSpaceStore.getState().rederive(false);
    expect(mockAnim.rederive).not.toHaveBeenCalled();
  });

  it("a refusal is a value the pane can show", async () => {
    mockAnim.clipInfo.mockResolvedValue(
      info({ refusal: "Walk is not an animation clip", curves: [], markers: [], rootMotion: null }),
    );
    await useBlendSpaceStore.getState().loadDerived("c1");
    expect(useBlendSpaceStore.getState().derived?.refusal).toMatch(/not an animation clip/);
  });
});
