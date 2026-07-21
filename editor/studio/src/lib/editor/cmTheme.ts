/**
 * CodeMirror 6 theme + syntax highlighting bridged to the studio `--ink-*`
 * palette (P5.1). Colors are `var(--ink-*)` references, so a theme swap
 * (lib/theme.ts writes the CSS variables on `:root`) repaints the editor with
 * no rebuild.
 */
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { EditorView } from "@codemirror/view";
import type { Extension } from "@codemirror/state";
import { tags as t } from "@lezer/highlight";

const MONO =
  'ui-monospace, "Cascadia Code", "JetBrains Mono", "Fira Code", Consolas, "SF Mono", Menlo, monospace';

/** The editor chrome theme (surfaces, gutters, selection, tooltips). */
export function editorTheme(): Extension {
  return EditorView.theme(
    {
      "&": {
        color: "var(--ink-text)",
        backgroundColor: "var(--ink-bg-1)",
        height: "100%",
      },
      ".cm-content": {
        caretColor: "var(--ink-accent)",
        fontFamily: MONO,
        fontSize: "13px",
      },
      ".cm-scroller": { fontFamily: MONO, lineHeight: "1.5" },
      ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--ink-accent)" },
      "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection":
        { backgroundColor: "var(--ink-accent-muted)" },
      ".cm-activeLine": { backgroundColor: "rgba(255,255,255,0.035)" },
      ".cm-gutters": {
        backgroundColor: "var(--ink-bg-1)",
        color: "var(--ink-text-faint)",
        border: "none",
      },
      ".cm-activeLineGutter": {
        backgroundColor: "rgba(255,255,255,0.035)",
        color: "var(--ink-text-dim)",
      },
      ".cm-foldPlaceholder": {
        backgroundColor: "var(--ink-bg-3)",
        border: "1px solid var(--ink-border)",
        color: "var(--ink-text-dim)",
      },
      ".cm-panels": { backgroundColor: "var(--ink-bg-2)", color: "var(--ink-text)" },
      ".cm-panels.cm-panels-top": { borderBottom: "1px solid var(--ink-border)" },
      ".cm-panels.cm-panels-bottom": { borderTop: "1px solid var(--ink-border)" },
      ".cm-searchMatch": {
        backgroundColor: "var(--ink-accent-muted)",
        outline: "1px solid var(--ink-accent)",
      },
      ".cm-searchMatch.cm-searchMatch-selected": { backgroundColor: "var(--ink-selection)" },
      ".cm-selectionMatch": { backgroundColor: "rgba(255,255,255,0.06)" },
      ".cm-matchingBracket, .cm-nonmatchingBracket": {
        backgroundColor: "var(--ink-bg-3)",
        outline: "1px solid var(--ink-border-strong)",
      },
      ".cm-tooltip": {
        backgroundColor: "var(--ink-bg-2)",
        border: "1px solid var(--ink-border)",
        color: "var(--ink-text)",
      },
      ".cm-tooltip-autocomplete > ul > li[aria-selected]": {
        backgroundColor: "var(--ink-selection)",
        color: "var(--ink-text)",
      },
      ".cm-tooltip-autocomplete > ul > li": { fontFamily: MONO },
    },
    { dark: true },
  );
}

/** Lezer highlight tag → color mapping, all theme-token driven. */
export function editorHighlighting(): Extension {
  const style = HighlightStyle.define([
    { tag: [t.keyword, t.modifier, t.controlKeyword, t.operatorKeyword], color: "var(--ink-accent)" },
    { tag: [t.string, t.special(t.string), t.regexp], color: "var(--ink-success)" },
    { tag: [t.number, t.bool, t.atom, t.null], color: "var(--ink-warning)" },
    { tag: [t.comment, t.lineComment, t.blockComment, t.docComment], color: "var(--ink-text-faint)", fontStyle: "italic" },
    { tag: [t.typeName, t.className, t.namespace, t.definition(t.typeName)], color: "var(--ink-info)" },
    { tag: [t.function(t.variableName), t.function(t.propertyName), t.macroName], color: "var(--ink-text)" },
    { tag: [t.variableName, t.propertyName, t.attributeName], color: "var(--ink-text)" },
    { tag: [t.operator, t.punctuation, t.separator, t.bracket], color: "var(--ink-text-dim)" },
    { tag: [t.meta, t.annotation, t.processingInstruction], color: "var(--ink-info)" },
    { tag: [t.escape, t.character], color: "var(--ink-warning)" },
    { tag: [t.heading], color: "var(--ink-accent)", fontWeight: "bold" },
    { tag: [t.link, t.url], color: "var(--ink-info)", textDecoration: "underline" },
    { tag: [t.strong], fontWeight: "bold" },
    { tag: [t.emphasis], fontStyle: "italic" },
    { tag: [t.invalid], color: "var(--ink-error)" },
  ]);
  return syntaxHighlighting(style);
}
