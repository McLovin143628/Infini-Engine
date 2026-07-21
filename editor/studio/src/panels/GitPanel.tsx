/**
 * Git panel (P5.4): branch + staged/unstaged changes with per-file
 * stage/unstage/discard, a commit box, and a colorized unified diff for the
 * selected file. Shells out to `git` in the backend.
 */
import { useEffect, useMemo, useState } from "react";
import {
  Check,
  GitBranch,
  GitCommitHorizontal,
  Minus,
  Plus,
  RotateCcw,
} from "lucide-react";

import type { GitFileDto } from "../bindings/GitFileDto";
import { git } from "../lib/ipc";
import { requestOpenFile } from "../lib/openFile";
import { cn } from "../lib/utils";
import { useGitStore } from "../stores/gitStore";
import { useProjectStore } from "../stores/projectStore";

export default function GitPanel() {
  const project = useProjectStore((s) => s.current);
  const status = useGitStore((s) => s.status);
  const refresh = useGitStore((s) => s.refresh);
  const stage = useGitStore((s) => s.stage);
  const unstage = useGitStore((s) => s.unstage);
  const discard = useGitStore((s) => s.discard);
  const commit = useGitStore((s) => s.commit);
  const init = useGitStore((s) => s.init);

  const [message, setMessage] = useState("");
  const [selected, setSelected] = useState<{ path: string; staged: boolean } | null>(null);
  const [diff, setDiff] = useState<string>("");

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(id);
  }, [refresh, project?.root]);

  useEffect(() => {
    if (!selected || !project) return; // DiffView is hidden without a selection
    let alive = true;
    git
      .fileDiff(project.root, selected.path, selected.staged)
      .then((d) => alive && setDiff(d))
      .catch(() => alive && setDiff(""));
    return () => {
      alive = false;
    };
  }, [selected, project]);

  const { staged, unstaged } = useMemo(() => {
    const s: GitFileDto[] = [];
    const u: GitFileDto[] = [];
    for (const f of status?.files ?? []) (f.staged ? s : u).push(f);
    return { staged: s, unstaged: u };
  }, [status]);

  if (!project) {
    return <Empty text="Open a project to use source control." />;
  }
  if (status && !status.is_repo) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-4 text-center">
        <div className="text-xs text-(--ink-text-dim)">This project isn't a git repository.</div>
        <button
          className="rounded bg-(--ink-accent) px-3 py-1 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)"
          onClick={() => void init()}
        >
          Initialize Repository
        </button>
      </div>
    );
  }

  const abs = (rel: string) => `${project.root}/${rel}`;

  return (
    <div className="flex h-full min-h-0 flex-col text-xs">
      {/* Branch header */}
      <div className="flex h-8 shrink-0 items-center gap-1.5 border-b border-(--ink-border) px-2 text-(--ink-text-dim)">
        <GitBranch size={13} />
        <span className="truncate text-(--ink-text)">{status?.branch ?? "…"}</span>
        {!!status?.ahead && <span title="ahead">↑{status.ahead}</span>}
        {!!status?.behind && <span title="behind">↓{status.behind}</span>}
      </div>

      {/* Commit box */}
      <div className="flex shrink-0 flex-col gap-1 border-b border-(--ink-border) p-2">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Commit message"
          rows={2}
          className="resize-none rounded border border-(--ink-border) bg-(--ink-bg-1) px-2 py-1 text-xs outline-none focus:border-(--ink-accent)"
        />
        <button
          className="flex items-center justify-center gap-1 rounded bg-(--ink-accent) py-1 text-xs text-(--ink-text-onaccent) enabled:hover:bg-(--ink-accent-hover) disabled:opacity-50"
          disabled={!message.trim() || staged.length === 0}
          onClick={() => {
            void commit(message);
            setMessage("");
          }}
        >
          <GitCommitHorizontal size={13} /> Commit {staged.length > 0 && `(${staged.length})`}
        </button>
      </div>

      {/* Changes */}
      <div className="min-h-0 flex-1 overflow-auto">
        <Section
          title="Staged Changes"
          files={staged}
          selected={selected}
          onSelect={(path) => setSelected({ path, staged: true })}
          onOpen={(rel) => requestOpenFile(abs(rel))}
          actions={(f) => (
            <IconBtn title="Unstage" onClick={() => void unstage([f.path])}>
              <Minus size={12} />
            </IconBtn>
          )}
        />
        <Section
          title="Changes"
          files={unstaged}
          selected={selected}
          onSelect={(path) => setSelected({ path, staged: false })}
          onOpen={(rel) => requestOpenFile(abs(rel))}
          actions={(f) => (
            <>
              <IconBtn title="Discard" onClick={() => void discard([f.path])}>
                <RotateCcw size={12} />
              </IconBtn>
              <IconBtn title="Stage" onClick={() => void stage([f.path])}>
                <Plus size={12} />
              </IconBtn>
            </>
          )}
        />
        {staged.length === 0 && unstaged.length === 0 && (
          <div className="flex items-center gap-1 p-3 text-(--ink-text-faint)">
            <Check size={13} /> No changes.
          </div>
        )}
      </div>

      {/* Diff */}
      {selected && <DiffView text={diff} path={selected.path} />}
    </div>
  );
}

function Section({
  title,
  files,
  selected,
  onSelect,
  onOpen,
  actions,
}: {
  title: string;
  files: GitFileDto[];
  selected: { path: string; staged: boolean } | null;
  onSelect: (path: string) => void;
  onOpen: (path: string) => void;
  actions: (f: GitFileDto) => React.ReactNode;
}) {
  if (files.length === 0) return null;
  return (
    <div>
      <div className="sticky top-0 bg-(--ink-bg-1) px-2 py-1 text-[11px] font-semibold text-(--ink-text-dim)">
        {title} ({files.length})
      </div>
      {files.map((f) => (
        <div
          key={`${title}:${f.path}`}
          className={cn(
            "group flex cursor-pointer items-center gap-1.5 px-2 py-0.5 hover:bg-(--ink-bg-3)",
            selected?.path === f.path && "bg-(--ink-selection)",
          )}
          onClick={() => onSelect(f.path)}
          onDoubleClick={() => onOpen(f.path)}
          title={f.path}
        >
          <span className={cn("w-3 text-center font-mono", statusColor(f.status))}>{f.status}</span>
          <span className="min-w-0 flex-1 truncate">{f.path.split("/").pop()}</span>
          <span className="truncate text-[10px] text-(--ink-text-faint)">
            {f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : ""}
          </span>
          <span className="flex shrink-0 items-center gap-0.5 opacity-0 group-hover:opacity-100">
            {actions(f)}
          </span>
        </div>
      ))}
    </div>
  );
}

function DiffView({ text, path }: { text: string; path: string }) {
  const lines = useMemo(() => text.split("\n"), [text]);
  return (
    <div className="h-40 shrink-0 overflow-auto border-t border-(--ink-border) bg-(--ink-bg-0) font-mono text-[11px]">
      <div className="sticky top-0 bg-(--ink-bg-1) px-2 py-1 text-[11px] text-(--ink-text-dim)">
        {path}
      </div>
      {text.trim() === "" ? (
        <div className="p-2 text-(--ink-text-faint)">No diff (untracked or binary).</div>
      ) : (
        lines.map((l, i) => (
          <div
            key={i}
            className={cn(
              "whitespace-pre px-2",
              l.startsWith("+") && !l.startsWith("+++") && "bg-(--ink-success)/10 text-(--ink-success)",
              l.startsWith("-") && !l.startsWith("---") && "bg-(--ink-danger)/10 text-(--ink-danger)",
              l.startsWith("@@") && "text-(--ink-info)",
            )}
          >
            {l || " "}
          </div>
        ))
      )}
    </div>
  );
}

function IconBtn({
  title,
  onClick,
  children,
}: {
  title: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      title={title}
      className="flex size-5 items-center justify-center rounded text-(--ink-text-dim) hover:bg-(--ink-bg-2) hover:text-(--ink-text)"
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
    >
      {children}
    </button>
  );
}

function Empty({ text }: { text: string }) {
  return <div className="p-4 text-xs text-(--ink-text-faint)">{text}</div>;
}

function statusColor(s: string): string {
  switch (s) {
    case "M":
      return "text-(--ink-warning)";
    case "A":
    case "?":
      return "text-(--ink-success)";
    case "D":
      return "text-(--ink-danger)";
    case "U":
      return "text-(--ink-danger)";
    default:
      return "text-(--ink-text-dim)";
  }
}
