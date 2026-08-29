/**
 * The `.infini` tokenizer (wave SCRIPT2b).
 *
 * Two of these arms deliberately read **Ring 0's own source** rather than a
 * literal copied out of it: the reserved words and the symbol table. Two copies
 * of one expression across a language boundary are a contract nobody is
 * measuring (the Wave-C law `lspBridge.ts` already pays for), and a keyword the
 * language grows would otherwise be coloured as a variable for a whole wave
 * with nothing anywhere saying so.
 */
import { ensureSyntaxTree, StringStream } from "@codemirror/language";
import { EditorState } from "@codemirror/state";
import { describe, expect, it } from "vitest";

// The two Ring-0 files this suite pins itself against, read as text through
// Vite's `?raw` (see `src/raw.d.ts` for why not `node:fs`).
import LEX_RS from "../../../../../../crates/inf-script/src/lex.rs?raw";
import EXAMPLE from "../../../../../../templates/scripts/Example.infini?raw";

import {
  infini,
  INFINI_KEYWORDS,
  INFINI_TYPES,
  infiniStreamParser,
  type InfiniState,
} from "../infiniLanguage";
import { languageExtensionFor } from "../languages";

interface Token {
  text: string;
  style: string | null;
}

/** Run the stream parser over a whole document, as CodeMirror does: line by
 *  line, with one state that survives the line breaks. */
function tokenize(src: string): Token[] {
  const state = infiniStreamParser.startState?.(2) as InfiniState;
  const out: Token[] = [];
  for (const line of src.split("\n")) {
    const stream = new StringStream(line, 4, 2);
    while (!stream.eol()) {
      stream.start = stream.pos;
      const style = infiniStreamParser.token(stream, state);
      // The rule CodeMirror enforces with an exception: a token must consume.
      expect(stream.pos, `the tokenizer stalled at ${JSON.stringify(line)}`).toBeGreaterThan(
        stream.start,
      );
      out.push({ text: line.slice(stream.start, stream.pos), style });
    }
  }
  return out;
}

/** The style the tokenizer gives the first occurrence of `text`. */
function styleOf(tokens: Token[], text: string): string | null | undefined {
  return tokens.find((t) => t.text === text)?.style;
}

/** Pull a `["a", "b", …]` Rust array literal out of `lex.rs` by its name. */
function rustStringArray(source: string, name: string): string[] {
  const at = source.indexOf(`${name}:`);
  expect(at, `${name} is no longer declared in lex.rs`).toBeGreaterThan(-1);
  const open = source.indexOf("[", source.indexOf("=", at));
  const close = source.indexOf("];", open);
  return [...source.slice(open, close).matchAll(/"((?:[^"\\]|\\.)*)"/g)].map((m) =>
    m[1].replace(/\\(.)/g, "$1"),
  );
}

describe("the .infini language mode", () => {
  it("is what `languages.ts` returns for a .infini path", () => {
    expect(languageExtensionFor("Content/Scripts/Door.infini")).not.toBeNull();
    expect(languageExtensionFor("Content/Scripts/Door.INFINI")).not.toBeNull();
    // …and the seam is still a switch on the extension, not on the folder.
    expect(languageExtensionFor("Scripts/notes.txt")).toBeNull();
  });

  it("knows exactly the reserved words Ring 0 lexes", () => {
    // `KEYWORDS` in crates/inf-script/src/lex.rs is the language's own list.
    expect([...INFINI_KEYWORDS].sort()).toEqual(rustStringArray(LEX_RS, "KEYWORDS").sort());
  });

  it("gives every symbol Ring 0 lexes a colour, and invents none", () => {
    const ring0 = new Set([
      ...rustStringArray(LEX_RS, "SYMBOLS"),
      ...rustStringArray(LEX_RS, "SYMBOLS2"),
    ]);
    const mine = new Set<string>();
    for (const sym of ring0) {
      const tokens = tokenize(`a ${sym} b`);
      const t = tokens.find((x) => x.text === sym);
      expect(t, `\`${sym}\` did not lex as one token`).toBeDefined();
      expect(t?.style, `\`${sym}\` has no colour`).not.toBe("invalid");
      mine.add(sym);
    }
    expect(mine.size).toBe(ring0.size);
  });

  it("tokenizes the script every new project ships without one invalid token", () => {
    const tokens = tokenize(EXAMPLE);
    const bad = tokens.filter((t) => t.style === "invalid");
    expect(bad, `templates/scripts/Example.infini has unlexable text: ${JSON.stringify(bad)}`)
      .toEqual([]);
    // …and it really did read the file rather than an empty string.
    expect(tokens.length).toBeGreaterThan(50);
    expect(styleOf(tokens, "actor")).toBe("keyword");
    expect(styleOf(tokens, "45.0")).toBe("number");
    expect(styleOf(tokens, '"Example"')).toBe("string");
    expect(styleOf(tokens, "debug")).toBe("namespace");
    expect(styleOf(tokens, "print")).toBe("variableName.function");
    expect(styleOf(tokens, "angle_deg")).toBe("variableName");
    expect(styleOf(tokens, "float")).toBe("typeName");
  });

  it("reads a `--` comment to end of line, and does not mistake `-` or `->` for one", () => {
    const c = tokenize("local x = 1 -- a comment with -- inside it");
    expect(styleOf(c, "-- a comment with -- inside it")).toBe("comment");
    const minus = tokenize("local x = a - b");
    expect(styleOf(minus, "-")).toBe("operator");
    const arrow = tokenize("function f(a: float) -> float");
    expect(styleOf(arrow, "->")).toBe("operator");
    expect(styleOf(arrow, "float")).toBe("typeName");
  });

  it("keeps a long bracket open across lines, and closes it at its own level", () => {
    const src = ['rust [==[', '  let v = vec![[1, 2][0]];', ']==]', 'local after = 1'].join("\n");
    const tokens = tokenize(src);
    // Every token of the body is a string, including the `]]` that would have
    // closed a level-0 bracket — which is exactly why the emitter picks a level
    // the content does not contain.
    expect(styleOf(tokens, "  let v = vec![[1, 2][0]];")).toBe("string");
    // …and the code after the close is code again.
    expect(styleOf(tokens, "local")).toBe("keyword");
    expect(styleOf(tokens, "after")).toBe("variableName");
  });

  it("closes a level-0 long bracket at the first `]]`, as Ring 0 does", () => {
    const tokens = tokenize("rust [[ let x = 1; ]] local y = 2");
    expect(styleOf(tokens, "local")).toBe("keyword");
  });

  it("does not let a string span a line", () => {
    const tokens = tokenize('debug.print("unterminated\nlocal x = 1');
    // The opener runs to the end of ITS line and no further; Ring 0 refuses the
    // file, and the editor's second line is still code.
    expect(styleOf(tokens, "local")).toBe("keyword");
    expect(tokens.some((t) => t.style === "string" && t.text.includes("unterminated"))).toBe(true);
  });

  it("reads `var` as a name, because Ring 0 does", () => {
    const tokens = tokenize('local n = var.get("hit count")');
    expect(styleOf(tokens, "var")).toBe("namespace");
    expect(styleOf(tokens, "get")).toBe("variableName.function");
    expect(styleOf(tokens, "local")).toBe("keyword");
  });

  it("colours a unit-local call, a namespaced verb and a three-segment query", () => {
    const local = tokenize("local d = damage(10)");
    expect(styleOf(local, "damage")).toBe("variableName.function");
    const q = tokenize("local h = physics2d.raycast.hit(0, 0, 1, 0)");
    expect(styleOf(q, "physics2d")).toBe("namespace");
    expect(styleOf(q, "raycast")).toBe("namespace");
    expect(styleOf(q, "hit")).toBe("variableName.function");
  });

  it("colours `true`/`false` as booleans and only names a type after a `:`", () => {
    const t = tokenize("local ok: bool = true");
    expect(styleOf(t, "true")).toBe("bool");
    expect(styleOf(t, "bool")).toBe("typeName");
    for (const ty of INFINI_TYPES) {
      // The four type names are ordinary identifiers everywhere else — they are
      // not reserved words, so a script may still call a variable `int`.
      expect(styleOf(tokenize(`${ty} = 1`), ty)).toBe("variableName");
    }
  });

  it("lexes numbers the way Ring 0 does", () => {
    expect(styleOf(tokenize("local a = 1"), "1")).toBe("number");
    expect(styleOf(tokenize("local a = 1.5"), "1.5")).toBe("number");
    expect(styleOf(tokenize("local a = 1e-3"), "1e-3")).toBe("number");
    // A trailing `.` is not part of the number (Ring 0 needs a digit after it).
    const dotted = tokenize("local a = 1.x");
    expect(styleOf(dotted, "1")).toBe("number");
    expect(styleOf(dotted, ".")).toBe("punctuation");
  });

  it("survives CodeMirror's own driver over the shipped script", () => {
    // The helper above is my loop; this is CM's. `readToken` throws
    // "Stream parser failed to advance stream" if a token consumes nothing, and
    // the places that could — a blank line, end of line, an unterminated long
    // bracket running to the end of the file — are exactly the ones a
    // hand-written loop is most likely to drive differently.
    const doc = `${EXAMPLE}\n\nrust [==[\nlet unterminated = 1;\n`;
    const state = EditorState.create({ doc, extensions: [infini()] });
    const tree = ensureSyntaxTree(state, state.doc.length, 5000);
    expect(tree, "the language did not finish parsing the document").not.toBeNull();
    expect(tree!.length).toBe(state.doc.length);
  });

  it("marks a character the language has no token for as invalid", () => {
    expect(styleOf(tokenize("local a = 1 @ 2"), "@")).toBe("invalid");
    expect(styleOf(tokenize("local a = #b"), "#")).toBe("invalid");
  });
});
