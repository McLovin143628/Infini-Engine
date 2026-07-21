import { describe, expect, it } from "vitest";

import { languageExtensionFor } from "../languages";

describe("languageExtensionFor", () => {
  it("maps known source extensions to a language extension", () => {
    for (const p of ["main.rs", "a/b/App.tsx", "x.ts", "s.js", "data.json", "s.css", "i.html", "readme.md", "s.py"]) {
      expect(languageExtensionFor(p), p).not.toBeNull();
    }
  });

  it("returns null for unknown / extensionless files", () => {
    expect(languageExtensionFor("notes.txt")).toBeNull();
    expect(languageExtensionFor("Makefile")).toBeNull();
    expect(languageExtensionFor("data.bin")).toBeNull();
  });

  it("ignores directory dots when reading the extension", () => {
    // A dotted folder must not be mistaken for the file extension.
    expect(languageExtensionFor("my.app/src/main.rs")).not.toBeNull();
  });
});
