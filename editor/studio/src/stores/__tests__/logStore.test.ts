import { beforeEach, describe, expect, it } from "vitest";
import type { LogLine } from "../../bindings/LogLine";
import { LOG_CAP, useLogStore } from "../logStore";

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
