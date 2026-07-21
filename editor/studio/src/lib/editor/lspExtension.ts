/**
 * CodeMirror ⇆ LSP glue (P5.2). Built per active file and installed into the
 * editor's `extraCompartment` seam (setup.ts) by the LSP bridge. Provides an
 * autocomplete source, a hover tooltip, and a debounced didChange notifier;
 * diagnostics are pushed separately via `diagnosticsToCM`.
 *
 * All server I/O goes through the typed `lsp` IPC domain. LSP positions are
 * (line, utf-16 character); CM uses document offsets — the helpers below bridge
 * them (BMP-correct; astral-plane columns are a rare edge left for later).
 */
import { autocompletion, type CompletionSource } from "@codemirror/autocomplete";
import { setDiagnostics, type Diagnostic } from "@codemirror/lint";
import type { Text } from "@codemirror/state";
import { EditorView, hoverTooltip } from "@codemirror/view";

import { lsp } from "../ipc";
import type { LspDiagnostic } from "../events";

const LANG = "rust";

interface LspContext {
  path: string;
  uri: string;
}

// ── position conversion ─────────────────────────────────────────────────────

function offsetToPos(doc: Text, offset: number): { line: number; character: number } {
  const line = doc.lineAt(offset);
  return { line: line.number - 1, character: offset - line.from };
}
function posToOffset(doc: Text, pos: { line: number; character: number }): number {
  if (pos.line >= doc.lines) return doc.length;
  const line = doc.line(pos.line + 1);
  return Math.min(line.from + pos.character, line.to);
}

// ── completion ──────────────────────────────────────────────────────────────

/** Map an LSP CompletionItemKind to a CM completion `type` (rough). */
function kindToType(kind?: number): string {
  switch (kind) {
    case 2:
    case 3:
    case 4:
      return "function";
    case 5:
      return "property";
    case 6:
    case 7:
      return "variable";
    case 8:
    case 9:
      return "interface";
    case 10:
      return "enum";
    case 14:
      return "keyword";
    case 21:
      return "constant";
    default:
      return "text";
  }
}

function completionSource(ctx: LspContext): CompletionSource {
  return async (cctx) => {
    const doc = cctx.state.doc;
    const pos = offsetToPos(doc, cctx.pos);
    // Find the token start for the replacement range.
    const word = cctx.matchBefore(/[\w:]+/);
    const from = word ? word.from : cctx.pos;
    let result: unknown;
    try {
      result = await lsp.request(LANG, "textDocument/completion", {
        textDocument: { uri: ctx.uri },
        position: pos,
      });
    } catch {
      return null;
    }
    const rawItems = Array.isArray(result)
      ? result
      : ((result as { items?: unknown[] } | null)?.items ?? []);
    const options = (rawItems as Array<Record<string, unknown>>).map((item) => ({
      label: String(item.label ?? ""),
      type: kindToType(item.kind as number | undefined),
      detail: typeof item.detail === "string" ? item.detail : undefined,
    }));
    if (options.length === 0) return null;
    return { from, options, validFor: /[\w:]*/ };
  };
}

// ── hover ───────────────────────────────────────────────────────────────────

function hoverContentsToText(contents: unknown): string {
  if (typeof contents === "string") return contents;
  if (Array.isArray(contents)) return contents.map(hoverContentsToText).join("\n\n");
  if (contents && typeof contents === "object") {
    const v = (contents as { value?: unknown }).value;
    if (typeof v === "string") return v;
  }
  return "";
}

function lspHover(ctx: LspContext) {
  return hoverTooltip(async (view, pos) => {
    const doc = view.state.doc;
    const position = offsetToPos(doc, pos);
    let result: unknown;
    try {
      result = await lsp.request(LANG, "textDocument/hover", {
        textDocument: { uri: ctx.uri },
        position,
      });
    } catch {
      return null;
    }
    const text = hoverContentsToText((result as { contents?: unknown } | null)?.contents);
    if (!text.trim()) return null;
    return {
      pos,
      create() {
        const dom = document.createElement("div");
        dom.className = "cm-lsp-hover";
        dom.textContent = text;
        return { dom };
      },
    };
  });
}

// ── didChange ───────────────────────────────────────────────────────────────

function changeNotifier(ctx: LspContext) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  return EditorView.updateListener.of((u) => {
    if (!u.docChanged || u.transactions.length === 0) return;
    if (timer) clearTimeout(timer);
    const text = u.state.doc.toString();
    timer = setTimeout(() => {
      void lsp.didChange(LANG, ctx.path, text);
    }, 300);
  });
}

// ── public API ──────────────────────────────────────────────────────────────

/** The extension bundle for a file, for the editor's `extraCompartment`. */
export function lspExtensionFor(ctx: LspContext) {
  return [
    autocompletion({ override: [completionSource(ctx)] }),
    lspHover(ctx),
    changeNotifier(ctx),
    EditorView.theme({
      ".cm-lsp-hover": {
        padding: "4px 8px",
        maxWidth: "480px",
        whiteSpace: "pre-wrap",
        fontSize: "12px",
      },
    }),
  ];
}

/** Convert LSP diagnostics to CM lint diagnostics and push them onto `view`. */
export function pushDiagnostics(view: EditorView, diags: LspDiagnostic[]): void {
  const doc = view.state.doc;
  const cm: Diagnostic[] = diags.map((d) => {
    const from = posToOffset(doc, d.range.start);
    const to = Math.max(from, posToOffset(doc, d.range.end));
    const severity =
      d.severity === 1
        ? "error"
        : d.severity === 2
          ? "warning"
          : d.severity === 3
            ? "info"
            : "hint";
    return { from, to, severity, message: d.message, source: d.source };
  });
  view.dispatch(setDiagnostics(view.state, cm));
}
