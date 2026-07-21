/**
 * LSP state (P5.2): per-language server status + the diagnostics map that feeds
 * the Problems panel. Written by the LSP bridge (lib/editor/lspBridge.ts), which
 * owns the event subscriptions + editor wiring; this store is the read model.
 */
import { create } from "zustand";

import type { LspDiagnostic } from "../lib/events";

export type LspStatus = "idle" | "starting" | "running" | "stopped" | "error";

interface LspState {
  status: Record<string, LspStatus>;
  error: string | null;
  /** Diagnostics by file URI (each publish replaces the file's set). */
  diagnostics: Record<string, LspDiagnostic[]>;

  setStatus: (language: string, status: LspStatus) => void;
  setError: (error: string | null) => void;
  setDiagnostics: (uri: string, diags: LspDiagnostic[]) => void;
  reset: () => void;
}

export const useLspStore = create<LspState>((set) => ({
  status: {},
  error: null,
  diagnostics: {},

  setStatus: (language, status) =>
    set((s) => ({ status: { ...s.status, [language]: status } })),
  setError: (error) => set({ error }),
  setDiagnostics: (uri, diags) =>
    set((s) => {
      const next = { ...s.diagnostics };
      if (diags.length === 0) delete next[uri];
      else next[uri] = diags;
      return { diagnostics: next };
    }),
  reset: () => set({ status: {}, error: null, diagnostics: {} }),
}));

/** A flat, sorted problem list for the Problems panel. */
export interface Problem {
  uri: string;
  line: number;
  severity: number;
  message: string;
  source?: string;
}

export function problemList(diagnostics: Record<string, LspDiagnostic[]>): Problem[] {
  const out: Problem[] = [];
  for (const [uri, diags] of Object.entries(diagnostics)) {
    for (const d of diags) {
      out.push({
        uri,
        line: d.range.start.line,
        severity: d.severity ?? 1,
        message: d.message,
        source: d.source,
      });
    }
  }
  return out.sort((a, b) => a.severity - b.severity || a.uri.localeCompare(b.uri) || a.line - b.line);
}

export function problemCount(diagnostics: Record<string, LspDiagnostic[]>): number {
  return Object.values(diagnostics).reduce((n, d) => n + d.length, 0);
}
