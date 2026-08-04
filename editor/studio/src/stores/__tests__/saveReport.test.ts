/**
 * What a save actually tells the author (P16.4b + P21.2).
 *
 * A level save is not all-or-nothing: the `.inf_lvl`, the `.inf_terrain` assets
 * its streamed terrains reference and the `.inf_voxel` assets its volumes
 * reference are separate files, and any of them can fail while the others land.
 * The status line is the only place that difference is visible, so these pin:
 *
 * 1. a clean save says so, without inventing numbers;
 * 2. terrain and voxel counts are reported **together** — an author who sculpted
 *    a hillside and carved a cave under it in one session did one save, not two;
 * 3. every failure line survives to the message, terrain and voxel alike;
 * 4. **the inline-terrain hole advisory reaches the author.** It rides
 *    `voxel_warnings`, and it is the one warning that reports a loss the save has
 *    ALREADY committed — a level whose cave mouths were just sealed looks
 *    completely normal until it is reloaded.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SaveResultDto } from "../../bindings/SaveResultDto";

vi.mock("../../lib/ipc", () => ({
  scene: {},
  assets: {},
  viewport: {},
}));

import { reportSaveResult } from "../sceneStore";
import { useShellStore } from "../shellStore";

function result(over: Partial<SaveResultDto> = {}): SaveResultDto {
  return {
    path: "C:/Projects/Demo/Levels/World.inf_lvl",
    terrain_assets_written: 0,
    terrain_tiles_written: 0,
    terrain_failures: [],
    voxel_assets_written: 0,
    voxel_chunks_written: 0,
    voxel_warnings: [],
    ...over,
  };
}

/** The message the last `pushStatus` carried. */
function lastStatus(): string {
  const calls = vi.mocked(useShellStore.getState().pushStatus).mock.calls;
  expect(calls.length).toBeGreaterThan(0);
  return calls[calls.length - 1][0] as string;
}

beforeEach(() => {
  useShellStore.setState({ pushStatus: vi.fn() });
  vi.clearAllMocks();
});

describe("reportSaveResult", () => {
  it("reports a clean save plainly", () => {
    reportSaveResult(result());
    expect(lastStatus()).toBe("Level saved.");
  });

  it("reports terrain and voxel write-backs in ONE message", () => {
    reportSaveResult(
      result({
        terrain_assets_written: 1,
        terrain_tiles_written: 12,
        voxel_assets_written: 2,
        voxel_chunks_written: 30,
      }),
    );
    const msg = lastStatus();
    expect(msg).toContain("12 terrain tile(s) to 1 asset(s)");
    expect(msg).toContain("30 voxel chunk(s) to 2 asset(s)");
  });

  it("mentions only the half that actually wrote", () => {
    reportSaveResult(result({ voxel_assets_written: 1, voxel_chunks_written: 4 }));
    expect(lastStatus()).toContain("4 voxel chunk(s)");
    expect(lastStatus()).not.toContain("terrain tile(s)");
  });

  it("surfaces terrain AND voxel failures together, and wins over the counts", () => {
    reportSaveResult(
      result({
        terrain_assets_written: 1,
        terrain_tiles_written: 3,
        terrain_failures: ["terrain abc: its .inf_terrain is not under C:/Content"],
        voxel_warnings: ["volume def: write C:/Content/Cave.inf_voxel: access denied"],
      }),
    );
    const msg = lastStatus();
    expect(msg).toContain("NOT fully written");
    expect(msg).toContain("its .inf_terrain is not under");
    expect(msg).toContain("access denied");
    // A partially-failed save must not read as a success with a tile count.
    expect(msg).not.toContain("3 terrain tile(s) to 1 asset(s)");
  });

  /**
   * THE ADVISORY'S LAST MILE. `voxel_warnings` also carries the inline-terrain
   * hole advisory, which is not a write failure at all: it says the save just
   * sealed cave mouths this level could not store. If it did not reach the status
   * line, the only symptom would be caves that vanish on the next reload.
   */
  it("carries the inline-terrain hole advisory to the author, fix included", () => {
    reportSaveResult(
      result({
        voxel_warnings: [
          "Ground: This terrain has cave mouths on 3 tile(s) but is stored INLINE in the level, " +
            "and an inline terrain cannot persist its hole mask — saving and reloading will seal " +
            "them. Convert the terrain to asset-backed (import or export it as a .inf_terrain) " +
            "to keep them.",
        ],
      }),
    );
    const msg = lastStatus();
    expect(msg).toContain("Ground:");
    expect(msg).toContain("cave mouths on 3 tile(s)");
    expect(msg).toContain(".inf_terrain");
  });
});
