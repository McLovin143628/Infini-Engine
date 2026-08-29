/**
 * The `.infini` CodeMirror language mode (wave SCRIPT2b).
 *
 * # It is a TOKENIZER, and that is the whole design
 *
 * InfiniScript has exactly one parser, and it is `crates/inf-script`'s — a
 * `.infini` file's meaning is `inf_blueprint::BlueprintFn` IR, and the arc's
 * founding rule is that nothing sits between the text and that IR. A second
 * grammar living in the frontend would be a second opinion about what a script
 * *is*, and the first time the two disagreed the editor would be confidently
 * wrong about a file the engine reads perfectly well.
 *
 * So this file colours tokens and nothing more. It has no notion of a block, a
 * declaration, a scope or a type. **Every semantic claim the editor makes comes
 * back over the wire from Ring 0** — see `scriptBridge.ts`, which is the same
 * `inf_script::compile` the watcher, the cook and the PIE payload builder use.
 *
 * The tokens mirror appendix A.1 of `docs/memos/infiniscript-direction.md` (the
 * lexical rules) construct for construct: `--` line comments, `"…"` strings
 * that may not span lines, Lua long brackets at any `=` level, decimal numbers
 * with an optional fraction and exponent, the twenty reserved words, and
 * identifiers that admit alphabetic Unicode.
 *
 * # The two places it is a heuristic rather than a mirror, stated
 *
 * 1. **A call's namespace is recognised by the character after the identifier**
 *    (`.` → a namespace segment, `(` → the verb or the unit-local function).
 *    `debug . print(…)`, with spaces around the dot, lexes identically in Ring 0
 *    and colours differently here. Colour, not meaning.
 * 2. **A type name is recognised in the two positions the grammar has one** —
 *    after a `:` (a declaration, a local, a parameter) and after a `->` (a
 *    return type). `float`/`int`/`bool`/`string` are *not* reserved words (the
 *    keyword list is twenty long on purpose), so everywhere else they are
 *    ordinary identifiers and are coloured as such.
 *
 * # The theme bridge
 *
 * Token names are resolved by `@codemirror/language` straight against
 * `@lezer/highlight`'s `tags`, so returning `"keyword"` or
 * `"variableName.function"` lands on the very tags `cmTheme.ts`'s
 * `HighlightStyle` already paints from the `--ink-*` palette (the P5.1 pattern).
 * There is no `.infini` colour table: a theme swap repaints this language with
 * everything else, because it never had colours of its own.
 */
import { StreamLanguage, type StreamParser, type StringStream } from "@codemirror/language";
import type { Extension } from "@codemirror/state";

/**
 * The reserved words, in `crates/inf-script/src/lex.rs`'s order.
 *
 * `var` is deliberately absent — it is contextual in Ring 0 (a declaration at
 * the top level, a name everywhere else), and a tokenizer that painted it as a
 * keyword would lie about `var.get("hit count")`.
 */
export const INFINI_KEYWORDS = [
  "actor",
  "and",
  "do",
  "else",
  "elseif",
  "end",
  "exposed",
  "false",
  "for",
  "function",
  "if",
  "local",
  "not",
  "on",
  "or",
  "return",
  "rust",
  "then",
  "true",
  "while",
] as const;

/** The four type names, which are identifiers everywhere except after a `:`. */
export const INFINI_TYPES = ["float", "int", "bool", "string"] as const;

const KEYWORDS = new Set<string>(INFINI_KEYWORDS);
const TYPES = new Set<string>(INFINI_TYPES);

/** The two-character symbols, longest-first so `<=` never lexes as `<` then `=`. */
const LONG_SYMBOLS = ["==", "~=", "<=", ">=", "->"];
const OPERATORS = new Set(["+", "-", "*", "/", "%", "<", ">", "="]);
const PUNCTUATION = new Set(["(", ")", ",", ".", ":", ";"]);

const IDENT = /^[\p{Alphabetic}_][\p{Alphabetic}\p{N}_]*/u;
const NUMBER = /^\d+(\.\d+)?([eE][+-]?\d+)?/;
const LONG_OPEN = /^\[(=*)\[/;

/** The tokenizer's whole memory. */
export interface InfiniState {
  /**
   * The `=` level of an open long bracket, or `null` outside one. This is the
   * only state that crosses a line, and it is what makes `rust [[ … ]]` blocks
   * survive a scroll.
   */
  long: number | null;
  /** The previous token was a `:`, so a type name is expected next. */
  expectType: boolean;
}

/**
 * Consume the body of an open long bracket up to its matching close, which may
 * be on this line or a later one.
 */
function eatLongBody(stream: StringStream, state: InfiniState): string {
  const close = `]${"=".repeat(state.long ?? 0)}]`;
  const at = stream.string.indexOf(close, stream.pos);
  if (at < 0) {
    stream.skipToEnd();
  } else {
    stream.pos = at + close.length;
    state.long = null;
  }
  return "string";
}

/** A `"…"` string. Ring 0 refuses one that spans a line; so does this. */
function eatQuoted(stream: StringStream): string {
  stream.next(); // the opening quote
  let escaped = false;
  for (;;) {
    if (stream.eol()) return "string"; // unterminated — Ring 0 says so
    const c = stream.next();
    if (escaped) {
      escaped = false;
    } else if (c === "\\") {
      escaped = true;
    } else if (c === '"') {
      return "string";
    }
  }
}

/** Read one token, advancing `stream`. `null` is whitespace. */
export function infiniToken(stream: StringStream, state: InfiniState): string | null {
  if (state.long !== null) return eatLongBody(stream, state);
  if (stream.eatSpace()) return null;

  const c = stream.peek();
  if (c === undefined || c === "") return null;

  // `--` to end of line. Checked before the operators, exactly as the lexer's
  // `skip_trivia` runs before `symbol`.
  if (c === "-" && stream.string.startsWith("--", stream.pos)) {
    stream.skipToEnd();
    state.expectType = false;
    return "comment";
  }

  if (c === '"') {
    state.expectType = false;
    return eatQuoted(stream);
  }

  if (c === "[") {
    const open = stream.match(LONG_OPEN, true) as RegExpMatchArray | null;
    if (open) {
      state.expectType = false;
      state.long = open[1].length;
      // Lua's rule, and Ring 0's: one newline straight after the opener is not
      // content. Nothing to do here — the opener token ends at the line's end.
      return "string";
    }
    // A bare `[` is not a token in this language at all.
    stream.next();
    state.expectType = false;
    return "invalid";
  }

  if (c >= "0" && c <= "9") {
    stream.match(NUMBER, true);
    state.expectType = false;
    return "number";
  }

  const word = stream.match(IDENT, true) as RegExpMatchArray | null;
  if (word) {
    const w = word[0];
    const expectingType = state.expectType;
    state.expectType = false;
    if (KEYWORDS.has(w)) return w === "true" || w === "false" ? "bool" : "keyword";
    if (expectingType && TYPES.has(w)) return "typeName";
    const after = stream.peek();
    if (after === ".") return "namespace";
    if (after === "(") return "variableName.function";
    return "variableName";
  }

  for (const sym of LONG_SYMBOLS) {
    if (stream.string.startsWith(sym, stream.pos)) {
      stream.pos += sym.length;
      // `->` opens a return type, the grammar's other type position
      // (`function IDENT params ('->' TYPE)? block`).
      state.expectType = sym === "->";
      return "operator";
    }
  }

  stream.next();
  if (OPERATORS.has(c)) {
    state.expectType = false;
    return "operator";
  }
  if (PUNCTUATION.has(c)) {
    // Only a `:` arms the type position — see the heuristic note at the top.
    state.expectType = c === ":";
    return "punctuation";
  }
  state.expectType = false;
  return "invalid";
}

/** The stream parser, exported so a test can drive it without an editor. */
export const infiniStreamParser: StreamParser<InfiniState> = {
  name: "infini",
  startState: () => ({ long: null, expectType: false }),
  token: (stream, state) => infiniToken(stream, state),
  languageData: {
    // `Mod-/` toggles a comment through `@codemirror/commands`, which reads it
    // from here rather than from a table this file would have to register in.
    commentTokens: { line: "--" },
    closeBrackets: { brackets: ["(", '"'] },
  },
};

/** The `.infini` language extension, for `languages.ts`'s one seam. */
export function infini(): Extension {
  return StreamLanguage.define(infiniStreamParser);
}
