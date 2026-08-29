/**
 * Vite's `?raw` import suffix, typed (wave SCRIPT2b).
 *
 * Used only by tests that pin a frontend constant against **Ring 0's own
 * source** — the `.infini` tokenizer's reserved words and symbol table against
 * `crates/inf-script/src/lex.rs`. Reading the Rust file is the point: a literal
 * copied out of it is a contract nobody is measuring.
 *
 * `?raw` rather than `node:fs` deliberately — this project has no
 * `@types/node`, and adding one to read two files would be a dependency bought
 * for a test.
 */
declare module "*?raw" {
  const contents: string;
  export default contents;
}
