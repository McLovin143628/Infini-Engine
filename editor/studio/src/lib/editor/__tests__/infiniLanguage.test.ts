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
  INFINI_SYMBOLS,
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
      // A bare `throw` rather than an `expect` per token, because the fuzz arms
      // below drive this helper over hundred-thousand-character lines and an
      // assertion object per character is the slowest thing in the suite.
      if (stream.pos <= stream.start) {
        throw new Error(
          `the tokenizer stalled at column ${stream.start} of ${JSON.stringify(line)}`,
        );
      }
      out.push({ text: line.slice(stream.start, stream.pos), style });
    }
  }
  return out;
}

/** The style the tokenizer gives the first occurrence of `text`. */
function styleOf(tokens: Token[], text: string): string | null | undefined {
  return tokens.find((t) => t.text === text)?.style;
}

/**
 * Pull a `["a", "b", …]` Rust array literal out of `lex.rs` by its name.
 *
 * **Checked against the array's own declared length**, which is the audit's
 * fix: this is a regex reading Rust, and a reflow it could not follow used to
 * fail *quietly* — a `close` it could not find makes `slice(open, -1)` the rest
 * of the file, and an extraction that found no literals at all returns `[]`.
 * The keyword arm would have gone red on either (it compares the whole list),
 * but the symbol arm iterated whatever came back, so an empty `SYMBOLS` would
 * have left it green over `SYMBOLS2`'s three. `[&str; N]` is Ring 0's own count
 * of its own table, so requiring the extraction to match it is still not a
 * number this test picked.
 */
function rustStringArray(source: string, name: string): string[] {
  const decl = new RegExp(`${name}:\\s*\\[&str;\\s*(\\d+)\\]\\s*=\\s*\\[`).exec(source);
  expect(
    decl,
    `${name} is no longer declared in lex.rs as a fixed-size \`[&str; N]\` array — ` +
      "this arm reads Ring 0's source and cannot follow that shape",
  ).not.toBeNull();
  const open = decl!.index + decl![0].length - 1;
  const close = source.indexOf("];", open);
  expect(close, `${name}'s array literal is not closed by \`];\``).toBeGreaterThan(open);
  const found = [...source.slice(open, close).matchAll(/"((?:[^"\\]|\\.)*)"/g)].map((m) =>
    m[1].replace(/\\(.)/g, "$1"),
  );
  expect(
    found.length,
    `read ${found.length} entries out of ${name}, which declares ${decl![1]}`,
  ).toBe(Number(decl![1]));
  return found;
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
    const ring0 = [
      ...rustStringArray(LEX_RS, "SYMBOLS"),
      ...rustStringArray(LEX_RS, "SYMBOLS2"),
    ];
    for (const sym of ring0) {
      const tokens = tokenize(`a ${sym} b`);
      const t = tokens.find((x) => x.text === sym);
      expect(t, `\`${sym}\` did not lex as one token`).toBeDefined();
      expect(t?.style, `\`${sym}\` has no colour`).not.toBe("invalid");
    }
    // BOTH directions, which is what "and invents none" in this arm's name
    // claims. The first draft compared the size of a set it had just filled
    // from `ring0` with `ring0`'s own size, so it was equal by construction and
    // said nothing (the audit's finding). A symbol spelled in the mode and NOT
    // in Ring 0 is the failure that matters: it would be coloured as an
    // operator inside a file the compiler refuses.
    expect([...INFINI_SYMBOLS].sort()).toEqual([...ring0].sort());
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
    const arrow = tokenize("function f(a: float) -> int");
    expect(styleOf(arrow, "->")).toBe("operator");
    // BOTH type positions the grammar has: after a `:` and after a `->`.
    expect(styleOf(arrow, "float")).toBe("typeName");
    expect(styleOf(arrow, "int")).toBe("typeName");
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

  /**
   * **A tokenizer is on the keystroke path, so it must not stall, throw or
   * run away — on any text at all** (added by the SCRIPT2b audit).
   *
   * The arms above are about what the mode *says*; these two are about whether
   * it answers. `readToken` throws "Stream parser failed to advance stream"
   * after ten calls that consume nothing, and every branch here that could is
   * a branch about the END of something — a line, a string, a long bracket, the
   * file. The corpus is those ends, one per row, plus the sizes at which a
   * quadratic would show. No wall-clock assertion: the ceiling is generous
   * enough that only a hang or a blow-up can reach it.
   */
  const HOSTILE: [string, string][] = [
    ["an escaped quote", 'debug.print("a \\" b")\nlocal x = 1'],
    ["an escaped backslash before the close", 'debug.print("a \\\\")\nlocal x = 1'],
    ["a quote at end of line", 'local s = "'],
    ["a backslash at end of line inside a string", 'local s = "abc\\'],
    ["a nested long bracket", "rust [==[\n  [[ inner ]]\n]==]\nlocal after = 1"],
    ["a long bracket that never closes", "rust [==[\nlet x = 1;"],
    ["a long opener that is the whole file", "rust [==["],
    ["a comment at EOF with no newline", "local x = 1 -- trailing"],
    ["a bare `--` at EOF", "local x = 1 --"],
    ["a lone `-` at EOF", "local x = 1 -"],
    ["a `[` at EOF", "local x = ["],
    ["a `[==` at EOF", "local x = [=="],
    ["an empty document", ""],
    ["nothing but newlines", "\n\n\n"],
    ["nothing but whitespace", "    \n\t\t\n"],
    ["a `:` at EOF, with the type position armed", "local x:"],
    ["a `->` at EOF, with the type position armed", "function f() ->"],
    ["a 10 000-character identifier", `local ${"a".repeat(10000)} = 1`],
    ["a 10 000-digit number", `local a = ${"9".repeat(10000)}`],
    ["a 10 000-character string", `local a = "${"x".repeat(10000)}"`],
    ["a 10 000-character comment", `-- ${"z".repeat(10000)}`],
    ["a 10 000-character long-bracket body", `rust [[${"q".repeat(10000)}]]`],
    ["10 000 backslashes inside a string", `local a = "${"\\".repeat(10000)}"`],
    ["50 000 one-character tokens on one line", "a ".repeat(50000)],
    ["100 000 dots on one line", ".".repeat(100000)],
    ["a long bracket open across 20 000 lines", `rust [==[\n${"x\n".repeat(20000)}`],
    ["20 000 bare quotes", '"'.repeat(20000)],
    ["5 000 `=` between the brackets", `rust [${"=".repeat(5000)}[`],
    ["a close with no open", "]==]"],
    ["an astral identifier", "local \u{1d54f} = 1"],
    ["an emoji where a value goes", "local a = \u{1f600}"],
    ["a lone surrogate", "local a = \ud83d"],
    ["a NUL byte", "local a = \0 1"],
    ["a combining mark", "local áb = 1"],
    ["a number with a trailing dot", "local a = 1."],
    ["an exponent with no digits", "local a = 1e"],
  ];

  it("never stalls, throws or runs away, on any of the ends it could", () => {
    for (const [name, src] of HOSTILE) {
      // `tokenize` asserts the consume rule itself; this is CodeMirror's own
      // driver over the same text, which is the one that would throw in a real
      // editor.
      expect(() => tokenize(src), `my loop, on ${name}`).not.toThrow();
      const state = EditorState.create({ doc: src, extensions: [infini()] });
      const tree = ensureSyntaxTree(state, state.doc.length, 20000);
      expect(tree, `CodeMirror's driver did not finish: ${name}`).not.toBeNull();
      expect(tree!.length, `the tree does not cover the document: ${name}`).toBe(
        state.doc.length,
      );
    }
  });

  it("survives four thousand pseudorandom documents through both drivers", () => {
    // The alphabet is the language's own pieces plus the characters it has no
    // token for, so the generator spends its time on token boundaries rather
    // than on prose. xorshift32, so a failure names a reproducible seed.
    const alphabet = [
      ...'abcXYZ_019 \t.,:;()+-*/%<>=~[]"\'#@!$&|^{}?\\',
      "--",
      "->",
      "==",
      "~=",
      "[[",
      "]]",
      "[=[",
      "]=]",
      "\n",
      "actor",
      "function",
      "end",
      "rust",
      "local",
      "\u{1d54f}",
      "\u{1f600}",
    ];
    let s = 0x1234_5678;
    const rnd = () => {
      s ^= s << 13;
      s >>>= 0;
      s ^= s >>> 17;
      s ^= s << 5;
      s >>>= 0;
      return s / 0x1_0000_0000;
    };
    for (let i = 0; i < 4000; i++) {
      const n = 1 + Math.floor(rnd() * 900);
      let doc = "";
      for (let j = 0; j < n; j++) doc += alphabet[Math.floor(rnd() * alphabet.length)];
      try {
        tokenize(doc);
        const state = EditorState.create({ doc, extensions: [infini()] });
        expect(ensureSyntaxTree(state, state.doc.length, 20000)).not.toBeNull();
      } catch (e) {
        throw new Error(`document ${i}: ${JSON.stringify(doc)}`, { cause: e });
      }
    }
  });
});
