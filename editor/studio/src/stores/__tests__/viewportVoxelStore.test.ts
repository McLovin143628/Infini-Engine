/**
 * P21.2: the voxel carve tool's store slice and its verdict readout.
 *
 * Four claims, and the last two are the ones that matter:
 *
 * 1. every setter **pushes the whole DTO** — a setter that updated state and
 *    forgot the push would leave the toolbar and the tool describing two
 *    different cuts, which for a tool that commits geometry means digging
 *    something other than what the author configured;
 * 2. a half-typed number input (`NaN`) and a negative length never reach the
 *    viewport;
 * 3. entering the tool **arms** both halves — the settings push and the verdict
 *    read — because a carve is refusable and an author must see *why* before the
 *    gesture, not after (the P20.4 arming-arm lesson, one tool over);
 * 4. `voxelIssues` distinguishes the three states that must never look alike:
 *    nothing to carve, a cut that would be refused, and a document that is
 *    ALREADY carrying cave mouths it cannot save.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { VoxelStatusDto } from "../../bindings/VoxelStatusDto";

vi.mock("../../lib/ipc", () => ({
  water: {
    riverReport: vi.fn(),
  },
  viewport: {
    setMode: vi.fn().mockResolvedValue(undefined),
    setSnap2d: vi.fn().mockResolvedValue(undefined),
    setToolMode: vi.fn().mockResolvedValue(undefined),
    setSculpt: vi.fn().mockResolvedValue(undefined),
    setFoliage: vi.fn().mockResolvedValue(undefined),
    setBiome: vi.fn().mockResolvedValue(undefined),
    setWater: vi.fn().mockResolvedValue(undefined),
    setVoxel: vi.fn().mockResolvedValue(undefined),
    voxelStatus: vi.fn(),
  },
  terrain: {
    biomes: vi.fn().mockResolvedValue(null),
  },
  projectSettings: {
    get: vi.fn().mockResolvedValue({ pixels_per_unit: 100 }),
    set: vi.fn().mockResolvedValue({ pixels_per_unit: 100 }),
  },
}));

import { viewport } from "../../lib/ipc";
import { voxelIssues, voxelVerdictTitle } from "../../viewport/ViewportToolbar";
import { useViewportStore } from "../viewportStore";

/**
 * `inf_editor_core::scene::undo::INLINE_TERRAIN_CARVE_REFUSAL`, character for
 * character. Copied rather than imported because it lives in Rust — and pinned
 * here so the tooltip test below is a real claim about the backend's sentence and
 * not about a fixture the toolbar invented.
 */
const REFUSAL =
  "Carve refused: this cut breaks through an INLINE terrain, and a terrain stored in the " +
  "level cannot save cave mouths (the level schema pins its tiles at a layout with no hole " +
  "mask). Nothing was carved. Convert the terrain to asset-backed — import or export it as " +
  "a .inf_terrain — and the same cut will work.";

/** A level that can be carved: one bound volume, one asset-backed terrain. */
function status(over: Partial<VoxelStatusDto> = {}): VoxelStatusDto {
  return {
    volumes: 1,
    bound_volumes: 1,
    asset_backed_terrains: 1,
    inline_terrains: [],
    unsaved_chunks: 0,
    refusal: null,
    advisories: [],
    ...over,
  };
}

/** The most recent DTO pushed to the viewport. */
function lastPush() {
  const calls = vi.mocked(viewport.setVoxel).mock.calls;
  expect(calls.length).toBeGreaterThan(0);
  return calls[calls.length - 1][0];
}

beforeEach(() => {
  useViewportStore.setState({
    voxelKind: "Brush",
    voxelRadius: 2,
    voxelDepth: 0,
    voxelMode: "Carve",
    voxelMaterial: 0,
    voxelStatus: null,
    toolMode: "Select",
  });
  vi.clearAllMocks();
  vi.mocked(viewport.voxelStatus).mockResolvedValue(status());
});

describe("voxel tool store", () => {
  it("defaults match the VoxelSettings the backend starts with", () => {
    const s = useViewportStore.getState();
    expect(s.voxelKind).toBe("Brush");
    expect(s.voxelRadius).toBe(2);
    expect(s.voxelDepth).toBe(0);
    expect(s.voxelMode).toBe("Carve");
    expect(s.voxelMaterial).toBe(0);
    // Nothing is pushed until something is set.
    expect(viewport.setVoxel).not.toHaveBeenCalled();
  });

  it("every setter pushes the WHOLE dto, not just its own field", () => {
    useViewportStore.getState().setVoxelKind("Tunnel");
    expect(lastPush()).toEqual({
      kind: "Tunnel",
      radius_m: 2,
      depth_m: 0,
      mode: "Carve",
      material: 0,
    });

    useViewportStore.getState().setVoxelRadius(5.5);
    expect(lastPush().radius_m).toBe(5.5);
    expect(lastPush().kind).toBe("Tunnel");

    useViewportStore.getState().setVoxelDepth(3);
    expect(lastPush().depth_m).toBe(3);
    useViewportStore.getState().setVoxelMode("Fill");
    expect(lastPush().mode).toBe("Fill");
    useViewportStore.getState().setVoxelMaterial(2);
    expect(lastPush().material).toBe(2);
    // Five setters, five pushes — none of them silent.
    expect(vi.mocked(viewport.setVoxel).mock.calls.length).toBe(5);
  });

  it("rejects NaN and negative lengths, and rounds the material into a u8", () => {
    const s = () => useViewportStore.getState();
    s().setVoxelRadius(-4);
    expect(s().voxelRadius).toBe(2); // kept the previous value
    s().setVoxelDepth(-1);
    expect(s().voxelDepth).toBe(0);

    s().setVoxelRadius(Number.NaN);
    s().setVoxelDepth(Number.NaN);
    s().setVoxelMaterial(Number.NaN);
    expect(s().voxelRadius).toBe(2);
    expect(s().voxelDepth).toBe(0);
    expect(s().voxelMaterial).toBe(0);

    s().setVoxelMaterial(3.7);
    expect(s().voxelMaterial).toBe(4);
    s().setVoxelMaterial(9999);
    expect(s().voxelMaterial).toBe(255);
    s().setVoxelMaterial(-3);
    expect(s().voxelMaterial).toBe(0);

    // …and nothing NaN or negative was ever pushed.
    for (const [dto] of vi.mocked(viewport.setVoxel).mock.calls) {
      expect(Number.isFinite(dto.radius_m)).toBe(true);
      expect(Number.isFinite(dto.depth_m)).toBe(true);
      expect(dto.radius_m).toBeGreaterThanOrEqual(0);
      expect(dto.depth_m).toBeGreaterThanOrEqual(0);
      expect(dto.material).toBeGreaterThanOrEqual(0);
      expect(dto.material).toBeLessThanOrEqual(255);
    }
  });

  it("allows a zero radius and a zero depth — 0 depth IS the cave-mouth cut", () => {
    useViewportStore.getState().setVoxelRadius(0);
    expect(useViewportStore.getState().voxelRadius).toBe(0);
    useViewportStore.getState().setVoxelDepth(0);
    expect(useViewportStore.getState().voxelDepth).toBe(0);
  });
});

describe("the voxel verdict", () => {
  it("arms the settings AND the verdict on entering the tool", async () => {
    useViewportStore.getState().setToolMode("Voxel");
    expect(viewport.setToolMode).toHaveBeenCalledWith("Voxel");
    expect(viewport.setVoxel).toHaveBeenCalled();
    expect(viewport.voxelStatus).toHaveBeenCalled();
    // The read is async; the store must actually take the answer.
    await useViewportStore.getState().refreshVoxelStatus();
    expect(useViewportStore.getState().voxelStatus?.bound_volumes).toBe(1);
  });

  it("survives a failing command without wedging on a stale verdict", async () => {
    await useViewportStore.getState().refreshVoxelStatus();
    expect(useViewportStore.getState().voxelStatus).not.toBeNull();
    vi.mocked(viewport.voxelStatus).mockRejectedValue(new Error("no viewport"));
    await useViewportStore.getState().refreshVoxelStatus();
    expect(useViewportStore.getState().voxelStatus).toBeNull();
  });

  it("says nothing about a carveable level", () => {
    expect(voxelIssues(status())).toEqual([]);
    expect(voxelIssues(status({ unsaved_chunks: 12 }))).toEqual([]);
  });

  /**
   * The refusal is the backend's own sentence, not a paraphrase — and it names
   * the terrain, because in a multi-terrain world "some terrain is inline" is not
   * an instruction anyone can act on.
   */
  it("names the terrain a surface-crossing cut would be refused over", () => {
    const issues = voxelIssues(
      status({
        asset_backed_terrains: 0,
        inline_terrains: ["Ground", "Island"],
        refusal: REFUSAL,
      }),
    );
    expect(issues).toHaveLength(1);
    expect(issues[0]).toContain("Ground, Island");
    expect(issues[0]).toContain("refused");
  });

  /**
   * THE THREE STATES. A level with no volume, an inline terrain and mouths
   * already on it has three separate problems, and collapsing any pair of them
   * would leave the author fixing the wrong one. The advisory comes FIRST because
   * it reports a loss that has already happened.
   */
  it("keeps damage-done, would-be-refused and nothing-to-carve apart", () => {
    const issues = voxelIssues(
      status({
        volumes: 0,
        bound_volumes: 0,
        asset_backed_terrains: 0,
        inline_terrains: ["Ground"],
        refusal: REFUSAL,
        advisories: ["Ground: This terrain has cave mouths on 2 tile(s) but is stored INLINE"],
      }),
    );
    expect(issues).toHaveLength(3);
    expect(issues[0]).toContain("cave mouths on 2 tile(s)");
    expect(issues[1]).toContain("refused");
    expect(issues[2]).toContain("no voxel volume");
  });

  /**
   * THE POINT OF THE WHOLE READOUT. The chip can only carry the shortest issue,
   * so the tooltip is where a refused author finds out what to change — and the
   * line that tells them is the backend's own sentence, which is the only place
   * the fix is written down.
   */
  it("quotes the backend refusal VERBATIM in the tooltip, fix and all", () => {
    const title = voxelVerdictTitle(
      status({ asset_backed_terrains: 0, inline_terrains: ["Ground"], refusal: REFUSAL }),
    );
    expect(title).toContain(REFUSAL);
    const joined = title.join("\n");
    expect(joined).toContain("Convert the terrain to asset-backed");
    expect(joined).toContain(".inf_terrain");
    // …and a level with nothing wrong says so without quoting a refusal at all.
    const clean = voxelVerdictTitle(status()).join("\n");
    expect(clean).toContain("Nothing in the way of a carve.");
    expect(clean).not.toContain("Carve refused");
  });

  it("puts the unsaved-chunk reminder in the tooltip, naming Ctrl+S", () => {
    const joined = voxelVerdictTitle(status({ unsaved_chunks: 7 })).join("\n");
    expect(joined).toContain("7 carved chunk(s)");
    expect(joined).toContain("Ctrl+S");
    // A clean level must not nag.
    expect(voxelVerdictTitle(status()).join("\n")).not.toContain("Ctrl+S");
  });

  it("distinguishes 'no volume at all' from 'a volume that resolved to nothing'", () => {
    expect(voxelIssues(status({ volumes: 0, bound_volumes: 0 }))[0]).toContain("no voxel volume");
    expect(voxelIssues(status({ volumes: 2, bound_volumes: 0 }))[0]).toContain(
      "no volume resolved to a .inf_voxel",
    );
  });
});
