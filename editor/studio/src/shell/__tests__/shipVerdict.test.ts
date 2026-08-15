// @vitest-environment jsdom
//
// **The Package dialog's ship / no-ship verdict** (round-2 finding B10 ==
// R2.F8).
//
// C4-40 gave `CookReport` a `blocking` list and made `inf cook` exit non-zero
// on it, printing "this build must not ship". The editor's `report_to_dto`
// dropped the field and `PackageResultDto` had no slot for it — so the dialog,
// which is **the door an author actually uses**, rendered those same strings
// inside a yellow "N warnings" bullet list under a full success panel, with
// `Boot Level: none` beside it as a plain stat. The CLI refused to ship exactly
// the build the editor called done.
import { describe, expect, it } from "vitest";

import type { PackageResultDto } from "../../bindings/PackageResultDto";
import { shipVerdict } from "../PackageDialog";

function report(warnings: string[], blocking: string[]): PackageResultDto {
  return {
    project_name: "Demo",
    engine_version: "0.1.0",
    out_dir: "C:/proj/Build",
    pack_path: "C:/proj/Build/demo.inf_pack",
    manifest_path: "C:/proj/Build/manifest.toml",
    asset_count: 12,
    kinds: [],
    pack_bytes: 4096,
    levels: [],
    root_level: null,
    blueprints_validated: 0,
    levels_rewritten: 0,
    warnings,
    blocking,
  };
}

const NO_BOOT = "no boot level: the runtime has nothing to load";
const DANGLING = "a material binding is dangling";

describe("shipVerdict", () => {
  it("says a cook with blocking advisories must not ship", () => {
    const v = shipVerdict(report([NO_BOOT, DANGLING], [NO_BOOT]));
    expect(v.blocked).toBe(true);
    expect(v.blocking).toEqual([NO_BOOT]);
  });

  it("does not count a blocking advisory twice", () => {
    // `CookReport`'s own contract: every blocking entry ALSO appears in
    // `warnings`. Rendering both lists whole would show the sentence that
    // stops the ship as a bullet in the "N warnings" list beside it — which is
    // precisely the presentation this finding is about.
    const v = shipVerdict(report([NO_BOOT, DANGLING], [NO_BOOT]));
    expect(v.advisories).toEqual([DANGLING]);
  });

  it("says an ordinary warning does not stop a ship", () => {
    // The other direction. A verdict that blocked on any warning would make
    // the dialog cry wolf, and an author who learns to ignore it is back where
    // this finding started.
    const v = shipVerdict(report([DANGLING], []));
    expect(v.blocked).toBe(false);
    expect(v.blocking).toEqual([]);
    expect(v.advisories).toEqual([DANGLING]);
  });

  it("says a clean cook is clean", () => {
    const v = shipVerdict(report([], []));
    expect(v.blocked).toBe(false);
    expect(v.advisories).toEqual([]);
  });

  it("reads the decision rather than the message text", () => {
    // The class this closes: which advisories block shipping is a DECISION the
    // cook makes and carries as its own list, not a spelling to be recovered
    // by matching on words. A warning that merely sounds alarming must not
    // block, and a blocking entry that sounds mild must.
    const alarming = "WARNING: this looks fatal but the cook did not say so";
    const mild = "the pack has no levels";
    const v = shipVerdict(report([alarming, mild], [mild]));
    expect(v.blocked).toBe(true);
    expect(v.blocking).toEqual([mild]);
    expect(v.advisories).toEqual([alarming]);
  });
});
