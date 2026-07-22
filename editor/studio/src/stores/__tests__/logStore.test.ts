import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LogLine } from "../../bindings/LogLine";
import { createLogBatcher, LOG_CAP, useLogStore } from "../logStore";

const line = (seq: number, level: LogLine["level"] = "info"): LogLine => ({
  seq,
  level,
  target: "test",
  message: `line ${seq}`,
  timestamp_ms: seq,
});

beforeEach(() => {
  useLogStore.setState({
    lines: [],
    enabled: { trace: false, debug: true, info: true, warn: true, error: true },
    search: "",
    paused: false,
  });
});

describe("logStore", () => {
  it("appends in order", () => {
    const { append } = useLogStore.getState();
    append(line(0));
    append(line(1));
    expect(useLogStore.getState().lines.map((l) => l.seq)).toEqual([0, 1]);
  });

  it("caps the ring buffer, dropping the oldest", () => {
    const { append } = useLogStore.getState();
    useLogStore.setState({
      lines: Array.from({ length: LOG_CAP }, (_, i) => line(i)),
    });
    append(line(LOG_CAP));
    const lines = useLogStore.getState().lines;
    expect(lines).toHaveLength(LOG_CAP);
    expect(lines[0].seq).toBe(1);
    expect(lines[lines.length - 1].seq).toBe(LOG_CAP);
  });

  it("drops lines while paused", () => {
    useLogStore.getState().setPaused(true);
    useLogStore.getState().append(line(0));
    expect(useLogStore.getState().lines).toHaveLength(0);
  });

  it("toggles severity filters", () => {
    useLogStore.getState().toggleLevel("info");
    expect(useLogStore.getState().enabled.info).toBe(false);
    useLogStore.getState().toggleLevel("info");
    expect(useLogStore.getState().enabled.info).toBe(true);
  });
});

describe("appendMany", () => {
  it("appends a batch in one update", () => {
    useLogStore.getState().appendMany([line(0), line(1), line(2)]);
    expect(useLogStore.getState().lines.map((l) => l.seq)).toEqual([0, 1, 2]);
  });

  it("cap-trims after the merge, keeping the newest", () => {
    useLogStore.setState({ lines: Array.from({ length: LOG_CAP }, (_, i) => line(i)) });
    useLogStore.getState().appendMany([line(LOG_CAP), line(LOG_CAP + 1)]);
    const lines = useLogStore.getState().lines;
    expect(lines).toHaveLength(LOG_CAP);
    expect(lines[0].seq).toBe(2);
    expect(lines[lines.length - 1].seq).toBe(LOG_CAP + 1);
  });

  it("drops a batch while paused", () => {
    useLogStore.getState().setPaused(true);
    useLogStore.getState().appendMany([line(0), line(1)]);
    expect(useLogStore.getState().lines).toHaveLength(0);
  });
});

describe("createLogBatcher", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("coalesces pushes into one flush per interval", () => {
    const flush = vi.fn();
    const batcher = createLogBatcher(flush, 60);
    batcher.push(line(0));
    batcher.push(line(1));
    batcher.push(line(2));
    expect(flush).not.toHaveBeenCalled(); // nothing before the timer fires
    vi.advanceTimersByTime(60);
    expect(flush).toHaveBeenCalledTimes(1);
    expect(flush.mock.calls[0][0].map((l: LogLine) => l.seq)).toEqual([0, 1, 2]);
  });

  it("starts a fresh window for pushes after a flush", () => {
    const flush = vi.fn();
    const batcher = createLogBatcher(flush, 60);
    batcher.push(line(0));
    vi.advanceTimersByTime(60);
    batcher.push(line(1));
    vi.advanceTimersByTime(60);
    expect(flush).toHaveBeenCalledTimes(2);
    expect(flush.mock.calls[1][0].map((l: LogLine) => l.seq)).toEqual([1]);
  });

  it("dispose cancels the pending flush and drops buffered lines", () => {
    const flush = vi.fn();
    const batcher = createLogBatcher(flush, 60);
    batcher.push(line(0));
    batcher.dispose();
    vi.advanceTimersByTime(120);
    expect(flush).not.toHaveBeenCalled();
  });
});
