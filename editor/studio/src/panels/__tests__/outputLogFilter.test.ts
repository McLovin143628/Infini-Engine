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

import type { LogLine } from "../../bindings/LogLine";
import { appendedLines } from "../OutputLogPanel";

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
 * The panel's incremental step, reproduced outside React: it is the composition
 * of `appendedLines` with a head-drop on `seq`, and the composition is what has
 * to equal a full re-filter. If these two ever disagree the panel shows a log
 * that is not the log.
 */
describe("incremental filtering equals a full re-filter", () => {
  const CAP = 64;

  function fullFilter(lines: LogLine[], keep: (l: LogLine) => boolean) {
    return lines.filter(keep);
  }

  function incremental(
    prevLines: LogLine[],
    prevVisible: LogLine[],
    lines: LogLine[],
    keep: (l: LogLine) => boolean,
  ): LogLine[] | null {
    const appended = appendedLines(prevLines, lines);
    if (!appended) return null;
    const oldest = lines.length > 0 ? lines[0]!.seq : Number.POSITIVE_INFINITY;
    let head = 0;
    while (head < prevVisible.length && prevVisible[head]!.seq < oldest) head += 1;
    return [...prevVisible.slice(head), ...appended.filter(keep)];
  }

  it("agrees with a full pass over a hundred capped flushes", () => {
    // Only even-numbered lines pass, so the head-drop has to remove entries
    // that are NOT simply the first few of `visible`.
    const keep = (l: LogLine) => l.seq % 2 === 0;
    let lines: LogLine[] = [];
    let visible: LogLine[] = [];
    let seq = 0;

    for (let flush = 0; flush < 100; flush++) {
      const batch = Array.from({ length: (flush % 7) + 1 }, () => line(seq++));
      const merged = [...lines, ...batch];
      const next = merged.length > CAP ? merged.slice(merged.length - CAP) : merged;

      const step = incremental(lines, visible, next, keep);
      expect(step).not.toBeNull();
      expect(step).toEqual(fullFilter(next, keep));

      lines = next;
      visible = step!;
    }
    // …and the run really did exercise the trimming half.
    expect(lines).toHaveLength(CAP);
  });
});
