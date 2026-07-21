/**
 * File-extension → CodeMirror language extension mapping (P5.1). Returns `null`
 * for unknown types (plain text, no highlighting).
 */
import { css } from "@codemirror/lang-css";
import { html } from "@codemirror/lang-html";
import { javascript } from "@codemirror/lang-javascript";
import { json } from "@codemirror/lang-json";
import { markdown } from "@codemirror/lang-markdown";
import { python } from "@codemirror/lang-python";
import { rust } from "@codemirror/lang-rust";
import type { Extension } from "@codemirror/state";

/** The language extension for a path, or `null` if unrecognized. */
export function languageExtensionFor(path: string): Extension | null {
  const ext = path.split(/[\\/]/).pop()?.split(".").pop()?.toLowerCase() ?? "";
  switch (ext) {
    case "rs":
      return rust();
    case "ts":
    case "tsx":
      return javascript({ typescript: true, jsx: ext === "tsx" });
    case "js":
    case "jsx":
    case "mjs":
    case "cjs":
      return javascript({ jsx: ext === "jsx" });
    case "json":
      return json();
    case "css":
      return css();
    case "html":
    case "htm":
      return html();
    case "md":
    case "markdown":
      return markdown();
    case "py":
      return python();
    default:
      return null;
  }
}
