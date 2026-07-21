/**
 * Problems panel (P5.2): the diagnostics rust-analyzer publishes, grouped by
 * file, most-severe first. Clicking a row opens the file in the editor.
 */
import { useMemo } from "react";
import { AlertCircle, AlertTriangle, Info, Lightbulb } from "lucide-react";

import { requestOpenFile } from "../lib/openFile";
import { problemList, useLspStore, type Problem } from "../stores/lspStore";

function uriToPath(uri: string): string {
  let p = uri.replace(/^file:\/\//, "").replace(/%20/g, " ");
  // file:///C:/… → C:/… on Windows.
  if (/^\/[A-Za-z]:/.test(p)) p = p.slice(1);
  return p;
}
function baseName(p: string): string {
  return p.split(/[\\/]/).pop() ?? p;
}

function SeverityIcon({ severity }: { severity: number }) {
  const size = 13;
  if (severity === 1) return <AlertCircle size={size} className="text-(--ink-danger)" />;
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
          <div className="m-2 rounded border border-(--ink-danger) bg-(--ink-danger)/10 p-2 text-(--ink-danger)">
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
