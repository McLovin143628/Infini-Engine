// @vitest-environment jsdom
//
// **The Output Log's incremental filter** (F-lens L7.M12).
//
// The panel re-filtered all 5000 lines on every batcher flush (~16×/s) and ran
// the `stripAnsi` regex per line inside that filter, then again for every
// rendered row. `appendedLines` is what turns a flush into "test the three that
// arrived": `logStore.appendMany` merges a batch and trims the front back to
// `LOG_CAP`, so consecutive `lines` arrays overlap in a run, and recognising
// that run is the whole optimisation.
//
// A mis-detected overlap would silently drop lines from the panel, so the
// recognition is what these arms hold shut — in both directions, because
// "always return null" is a correct-but-useless implementation and "always
// return the tail" is a wrong one.
import { describe, expect, it } from "vitest";

import type { LogLevel } from "../../bindings/LogLevel";
import type { LogLine } from "../../bindings/LogLine";
import { appendedLines, filterStep, type FilterMemo } from "../OutputLogPanel";

const line = (seq: number): LogLine => ({
  seq,
  level: "info",
  target: "inf::test",
  message: `m${seq}`,
  timestamp_ms: seq,
});

describe("appendedLines", () => {
  it("returns just the appended tail when nothing was trimmed", () => {
    const a = [line(1), line(2)];
    const c = line(3);
    const d = line(4);
    expect(appendedLines(a, [...a, c, d])).toEqual([c, d]);
  });

  it("returns the tail when the ring also trimmed its head", () => {
    // Both ends move at once once the ring is full — the common steady state.
    const a = [line(1), line(2), line(3)];
    const d = line(4);
    expect(appendedLines(a, [a[1]!, a[2]!, d])).toEqual([d]);
  });

  it("returns an empty tail when the array did not move", () => {
    const a = [line(1), line(2)];
    expect(appendedLines(a, [...a])).toEqual([]);
  });

  it("treats an empty previous array as all-new", () => {
    const next = [line(1), line(2)];
    expect(appendedLines([], next)).toBe(next);
  });

  it("refuses a CLEAR", () => {
    expect(appendedLines([line(1), line(2)], [])).toBeNull();
  });

  it("refuses a wholesale replacement that only looks equal", () => {
    // Equal by value, different objects — there is no shared run, and treating
    // one as a shared run would silently drop the real lines.
    const a = [line(1), line(2)];
    expect(appendedLines(a, [line(1), line(2), line(3)])).toBeNull();
  });

  it("refuses an overlap whose middle does not line up", () => {
    // Why the whole run is verified rather than just the last element: the
    // endpoint matches at the right offset here and the run still is not one.
    const a = [line(1), line(2), line(3)];
    expect(appendedLines(a, [a[1]!, line(9), a[2]!])).toBeNull();
  });
});

/**
 * **The panel's incremental step, called rather than reproduced.**
 *
 * The previous version of this block RE-IMPLEMENTED the composition — the
 * `appendedLines` call, the head-drop on `seq`, the concatenation — and then
 * asserted its own reimplementation against a full re-filter. So it verified
 * that two pieces of test code agreed: deleting the head-drop from
 * `OutputLogPanel` (the line that bounds the visible list against the ring's
 * trimming, i.e. the unbounded-accumulation defect) left every arm here green.
 * That is the test-integrity finding, and the repair is to call the real thing:
 * `filterStep` is now the panel's own `useMemo` body, lifted out whole, and the
 * panel calls exactly this.
 */
describe("incremental filtering equals a full re-filter", () => {
  const CAP = 64;
  const ALL: Record<LogLevel, boolean> = {
    trace: true,
    debug: true,
    info: true,
    warn: true,
    error: true,
  };

  function fullFilter(lines: LogLine[], keep: (l: LogLine) => boolean) {
    return lines.filter(keep);
  }

  it("agrees with a full pass over a hundred capped flushes", () => {
    // Only INFO lines pass, so the head-drop has to remove entries that are not
    // simply the first few of `visible`. (The level is what the panel filters
    // on, so the arm drives the panel's own predicate rather than a stand-in.)
    const keep = (l: LogLine) => l.level === "info";
    const enabled: Record<LogLevel, boolean> = { ...ALL, warn: false };
    let lines: LogLine[] = [];
    let memo: FilterMemo | null = null;
    let seq = 0;

    for (let flush = 0; flush < 100; flush++) {
      const batch = Array.from({ length: (flush % 7) + 1 }, () => {
        const l = line(seq++);
        // Alternate the level so half the lines are filtered out.
        return { ...l, level: (l.seq % 2 === 0 ? "info" : "warn") as LogLevel };
      });
      const merged = [...lines, ...batch];
      const next = merged.length > CAP ? merged.slice(merged.length - CAP) : merged;

      memo = filterStep(memo, next, enabled, "");
      expect(memo.visible).toEqual(fullFilter(next, keep));

      lines = next;
    }
    // …and the run really did exercise the trimming half.
    expect(lines).toHaveLength(CAP);
    expect(memo!.visible.length).toBeLessThan(CAP);
  });

  it("stays bounded by the ring, which is the defect it exists for", () => {
    // The head-drop, on its own. Without it `visible` only ever GROWS: the
    // panel accumulates every line the ring has already thrown away, which is
    // unbounded memory behind a bounded store.
    const enabled = ALL;
    let lines: LogLine[] = [];
    let memo: FilterMemo | null = null;
    for (let seq = 0; seq < 500; seq++) {
      const merged = [...lines, line(seq)];
      lines = merged.length > CAP ? merged.slice(merged.length - CAP) : merged;
      memo = filterStep(memo, lines, enabled, "");
    }
    expect(memo!.visible).toHaveLength(CAP);
    expect(memo!.visible[0]!.seq).toBe(500 - CAP);
  });

  it("re-filters from scratch when the filter itself changes", () => {
    // A changed predicate invalidates every previous verdict, so the memo must
    // NOT be extended — extending it would keep lines the new filter rejects.
    const lines = [0, 1, 2, 3].map((n) => ({
      ...line(n),
      level: (n % 2 === 0 ? "info" : "warn") as LogLevel,
    }));
    const all = filterStep(null, lines, ALL, "");
    expect(all.visible).toHaveLength(4);

    const infoOnly = filterStep(all, lines, { ...ALL, warn: false }, "");
    expect(infoOnly.visible.map((l) => l.seq)).toEqual([0, 2]);
  });

  it("honours the search needle", () => {
    const lines = [line(1), line(2), line(3)];
    const step = filterStep(null, lines, ALL, "  M2  ");
    expect(step.visible.map((l) => l.seq)).toEqual([2]);
    // The needle is normalised, and the memo records the normalised form —
    // otherwise a whitespace-only change re-filters the whole ring.
    expect(step.needle).toBe("m2");
  });
});
