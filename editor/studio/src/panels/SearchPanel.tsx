/**
 * Search panel (P5.4): global content search (literal/regex, gitignore-aware)
 * plus a fuzzy "Go to File" quick-open. Opening a result/file dispatches the
 * `infinity:open-file` event the editor (P5.1) consumes.
 */
import { useEffect, useMemo, useState } from "react";
import { CaseSensitive, FileSearch, Regex, Search as SearchIcon } from "lucide-react";

import type { FileEntryDto } from "../bindings/FileEntryDto";
import type { SearchHitDto } from "../bindings/SearchHitDto";
import { files, search } from "../lib/ipc";
import { fuzzyFilter } from "../lib/fuzzy";
import { requestOpenFile } from "../lib/openFile";
import { cn } from "../lib/utils";
import { useProjectStore } from "../stores/projectStore";

type Grouped = Record<string, SearchHitDto[]>;

export default function SearchPanel() {
  const project = useProjectStore((s) => s.current);
  const [mode, setMode] = useState<"content" | "file">("content");

  if (!project) {
    return <div className="p-4 text-xs text-(--ink-text-faint)">Open a project to search.</div>;
  }
  return (
    <div className="flex h-full min-h-0 flex-col text-xs">
      <div className="flex h-8 shrink-0 items-center gap-1 border-b border-(--ink-border) px-2">
        <Tab active={mode === "content"} onClick={() => setMode("content")} icon={<SearchIcon size={13} />}>
          Search
        </Tab>
        <Tab active={mode === "file"} onClick={() => setMode("file")} icon={<FileSearch size={13} />}>
          Go to File
        </Tab>
      </div>
      {mode === "content" ? (
        <ContentSearch root={project.root} />
      ) : (
        <FileSearchMode root={project.root} />
      )}
    </div>
  );
}

function ContentSearch({ root }: { root: string }) {
  const [query, setQuery] = useState("");
  const [regex, setRegex] = useState(false);
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [hits, setHits] = useState<SearchHitDto[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    const id = window.setTimeout(() => {
      if (query.trim().length < 2) {
        if (alive) setHits([]);
        return;
      }
      setBusy(true);
      search
        .workspace(root, query, { regex, case_sensitive: caseSensitive })
        .then((h) => alive && setHits(h))
        .catch(() => alive && setHits([]))
        .finally(() => alive && setBusy(false));
    }, 200);
    return () => {
      alive = false;
      window.clearTimeout(id);
    };
  }, [root, query, regex, caseSensitive]);

  const grouped = useMemo(() => {
    const g: Grouped = {};
    for (const h of hits) (g[h.path] ??= []).push(h);
    return g;
  }, [hits]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-1 border-b border-(--ink-border) p-1.5">
        <input
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search in project…"
          className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-1) px-2 py-1 outline-none focus:border-(--ink-accent)"
        />
        <Toggle on={regex} onClick={() => setRegex((v) => !v)} title="Regular expression">
          <Regex size={13} />
        </Toggle>
        <Toggle on={caseSensitive} onClick={() => setCaseSensitive((v) => !v)} title="Match case">
          <CaseSensitive size={14} />
        </Toggle>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {busy && <div className="p-2 text-(--ink-text-faint)">Searching…</div>}
        {!busy && query.trim().length >= 2 && hits.length === 0 && (
          <div className="p-2 text-(--ink-text-faint)">No matches.</div>
        )}
        {Object.entries(grouped).map(([path, list]) => (
          <div key={path}>
            <div className="sticky top-0 bg-(--ink-bg-1) px-2 py-0.5 text-[11px] font-semibold text-(--ink-text-dim)">
              {path} <span className="text-(--ink-text-faint)">({list.length})</span>
            </div>
            {list.map((h, i) => (
              <button
                key={i}
                className="flex w-full items-baseline gap-2 px-2 py-0.5 text-left hover:bg-(--ink-bg-3)"
                onClick={() => requestOpenFile(`${root}/${path}`)}
                title={`${path}:${h.line}:${h.column}`}
              >
                <span className="shrink-0 font-mono text-[10px] text-(--ink-text-faint)">
                  {h.line}
                </span>
                <span className="min-w-0 flex-1 truncate font-mono">{h.text}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function FileSearchMode({ root }: { root: string }) {
  const [all, setAll] = useState<FileEntryDto[]>([]);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let alive = true;
    files
      .list(root)
      .then((f) => alive && setAll(f.filter((e) => !e.is_dir)))
      .catch(() => alive && setAll([]));
    return () => {
      alive = false;
    };
  }, [root]);

  const results = useMemo(
    () => fuzzyFilter(query, all, (e) => e.path).slice(0, 200),
    [query, all],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="shrink-0 border-b border-(--ink-border) p-1.5">
        <input
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Go to file…"
          className="w-full rounded border border-(--ink-border) bg-(--ink-bg-1) px-2 py-1 outline-none focus:border-(--ink-accent)"
        />
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {results.map((e) => (
          <button
            key={e.path}
            className="flex w-full items-baseline gap-2 px-2 py-0.5 text-left hover:bg-(--ink-bg-3)"
            onClick={() => requestOpenFile(`${root}/${e.path}`)}
          >
            <span className="min-w-0 truncate">{e.name}</span>
            <span className="min-w-0 flex-1 truncate text-[10px] text-(--ink-text-faint)">
              {e.path}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Tab({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-1 rounded px-2 py-0.5",
        active ? "bg-(--ink-selection) text-(--ink-text)" : "text-(--ink-text-dim) hover:bg-(--ink-bg-3)",
      )}
    >
      {icon} {children}
    </button>
  );
}

function Toggle({
  on,
  onClick,
  title,
  children,
}: {
  on: boolean;
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        "flex size-6 shrink-0 items-center justify-center rounded",
        on ? "bg-(--ink-accent) text-(--ink-text-onaccent)" : "text-(--ink-text-dim) hover:bg-(--ink-bg-3)",
      )}
    >
      {children}
    </button>
  );
}
