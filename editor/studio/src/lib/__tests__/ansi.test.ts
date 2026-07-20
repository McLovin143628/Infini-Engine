import { describe, expect, it } from "vitest";
import { stripAnsi } from "../ansi";

describe("stripAnsi", () => {
  it("removes SGR color sequences", () => {
    expect(stripAnsi("\x1b[31merror\x1b[0m: boom")).toBe("error: boom");
    expect(stripAnsi("\x1b[1;32m✓\x1b[39;49m done")).toBe("✓ done");
  });

  it("removes OSC title sequences", () => {
    expect(stripAnsi("\x1b]0;title\x07text")).toBe("text");
  });

  it("leaves plain text (including brackets) alone", () => {
    expect(stripAnsi("array[3] = { x: 1 }")).toBe("array[3] = { x: 1 }");
  });
});
