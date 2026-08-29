/**
 * Problems panel (P5.2): diagnostics grouped by file, most-severe first.
 * Clicking a row opens the file in the editor.
 *
 * **Two producers since wave SCRIPT2b**, and the panel names them rather than
 * implying one: rust-analyzer publishes over `lsp://diagnostics`, and the
 * InfiniScript linter writes the same store from `inf_script::compile_bytes`.
 * Every row therefore carries its `source` — a header reading "rust-analyzer:
 * idle" above three InfiniScript refusals was the honest version of a lie.
 */
import { useMemo } from "react";
import { AlertCircle, AlertTriangle, Info, Lightbulb } from "lucide-react";

import { uriToPath } from "../lib/editor/fileUri";
import { requestOpenFile } from "../lib/openFile";
import { problemList, useLspStore, type Problem } from "../stores/lspStore";

function baseName(p: string): string {
  return p.split(/[\\/]/).pop() ?? p;
}

/**
 * **`--ink-danger` is not a theme token.** `THEME_COLOR_KEYS` in `lib/theme.ts`
 * defines `--ink-error`, and nothing has ever written `--ink-danger` — so
 * `color: var(--ink-danger)` is an invalid declaration the browser drops, and
 * this icon has been inheriting the row's colour rather than going red. Fixed
 * here, on the panel this wave wires InfiniScript's refusals into.
 *
 * Carried, because it is wider than this file: **23 more occurrences across 9
 * files** spell it `--ink-danger` (GitPanel's diff colouring, the Explorer's
 * status letters, the drawer's delete affordances, PreferencesDialog's error
 * text, `sm.css` — which alone supplies a fallback and so is the only one that
 * renders). One line in `applyTheme` aliasing `--ink-danger` to `error` would
 * make all of them real at once, and that is an app-wide visual change a human
 * should look at rather than a scripting wave should smuggle.
 */
function SeverityIcon({ severity }: { severity: number }) {
  const size = 13;
  if (severity === 1) return <AlertCircle size={size} className="text-(--ink-error)" />;
  if (severity === 2) return <AlertTriangle size={size} className="text-(--ink-warning)" />;
  if (severity === 3) return <Info size={size} className="text-(--ink-info)" />;
  return <Lightbulb size={size} className="text-(--ink-text-dim)" />;
}

export default function ProblemsPanel() {
  const diagnostics = useLspStore((s) => s.diagnostics);
  const status = useLspStore((s) => s.status.rust ?? "idle");
  const error = useLspStore((s) => s.error);

  const problems = useMemo(() => problemList(diagnostics), [diagnostics]);
  const byFile = useMemo(() => {
    const m = new Map<string, Problem[]>();
    for (const p of problems) {
      const arr = m.get(p.uri) ?? [];
      arr.push(p);
      m.set(p.uri, arr);
    }
    return [...m.entries()];
  }, [problems]);

  return (
    <div className="flex min-h-0 flex-1 flex-col text-xs">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-(--ink-border) bg-(--ink-bg-2) px-2 text-(--ink-text-dim)">
        <span>
          Problems{problems.length > 0 ? ` (${problems.length})` : ""}
        </span>
        <span className="text-[10px] uppercase tracking-wide text-(--ink-text-faint)">
          rust-analyzer: {status}
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {error && status === "error" && (
          <div className="m-2 rounded border border-(--ink-error) bg-(--ink-error)/10 p-2 text-(--ink-error)">
            {error}
          </div>
        )}
        {problems.length === 0 ? (
          <div className="p-3 text-(--ink-text-faint)">No problems detected.</div>
        ) : (
          byFile.map(([uri, items]) => {
            const path = uriToPath(uri);
            return (
              <div key={uri}>
                <div className="sticky top-0 bg-(--ink-bg-1) px-2 py-1 font-medium text-(--ink-text-dim)">
                  {baseName(path)}{" "}
                  <span className="text-(--ink-text-faint)">({items.length})</span>
                </div>
                {items.map((p, i) => (
                  <button
                    key={`${uri}:${i}`}
                    className="flex w-full items-start gap-1.5 px-3 py-1 text-left hover:bg-(--ink-bg-3)"
                    onClick={() => requestOpenFile(path)}
                    title={p.source ? `${p.source}: ${p.message}` : p.message}
                  >
                    <span className="mt-0.5 shrink-0">
                      <SeverityIcon severity={p.severity} />
                    </span>
                    <span className="min-w-0 flex-1 truncate">{p.message}</span>
                    {p.source && (
                      <span className="shrink-0 text-[10px] text-(--ink-text-faint)">
                        {p.source}
                      </span>
                    )}
                    <span className="shrink-0 text-(--ink-text-faint)">{p.line + 1}</span>
                  </button>
                ))}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
