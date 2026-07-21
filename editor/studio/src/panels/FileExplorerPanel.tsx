/**
 * File Explorer (P5.4): the open project's source tree (gitignore-aware), with
 * git status letters per file. Clicking a file opens it in the editor via the
 * `infinity:open-file` event (P5.1).
 */
import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight, File, Folder, FolderOpen, RefreshCw } from "lucide-react";

import type { FileEntryDto } from "../bindings/FileEntryDto";
import { files } from "../lib/ipc";
import { requestOpenFile } from "../lib/openFile";
import { cn } from "../lib/utils";
import { useGitStore } from "../stores/gitStore";
import { useProjectStore } from "../stores/projectStore";

interface TreeNode {
  name: string;
  path: string; // relative to root
  isDir: boolean;
  children: TreeNode[];
}

function buildTree(entries: FileEntryDto[]): TreeNode[] {
  const root: TreeNode = { name: "", path: "", isDir: true, children: [] };
  const dirs = new Map<string, TreeNode>([["", root]]);
  // Ensure a directory node exists for `path`, creating ancestors.
  const ensureDir = (path: string): TreeNode => {
    const existing = dirs.get(path);
    if (existing) return existing;
    const slash = path.lastIndexOf("/");
    const parent = ensureDir(slash === -1 ? "" : path.slice(0, slash));
    const node: TreeNode = {
      name: path.slice(slash + 1),
      path,
      isDir: true,
      children: [],
    };
    parent.children.push(node);
    dirs.set(path, node);
    return node;
  };
  for (const e of entries) {
    if (e.is_dir) {
      ensureDir(e.path);
    } else {
      const slash = e.path.lastIndexOf("/");
      const parent = ensureDir(slash === -1 ? "" : e.path.slice(0, slash));
      parent.children.push({ name: e.name, path: e.path, isDir: false, children: [] });
    }
  }
  const sort = (n: TreeNode) => {
    n.children.sort((a, b) =>
      a.isDir === b.isDir ? a.name.localeCompare(b.name) : a.isDir ? -1 : 1,
    );
    n.children.forEach(sort);
  };
  sort(root);
  return root.children;
}

export default function FileExplorerPanel() {
  const project = useProjectStore((s) => s.current);
  const gitStatus = useGitStore((s) => s.status);
  const [entries, setEntries] = useState<FileEntryDto[]>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const reload = useMemo(
    () => () => {
      if (!project) return;
      files
        .list(project.root)
        .then(setEntries)
        .catch(() => setEntries([]));
    },
    [project],
  );

  useEffect(reload, [reload]);

  const tree = useMemo(() => buildTree(entries), [entries]);
  const gitByPath = useMemo(() => {
    const m = new Map<string, string>();
    for (const f of gitStatus?.files ?? []) if (!m.has(f.path)) m.set(f.path, f.status);
    return m;
  }, [gitStatus]);

  if (!project) {
    return <div className="p-4 text-xs text-(--ink-text-faint)">No project open.</div>;
  }

  const toggle = (path: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const rows: React.ReactNode[] = [];
  const walk = (nodes: TreeNode[], depth: number) => {
    for (const n of nodes) {
      const isOpen = expanded.has(n.path);
      const gs = gitByPath.get(n.path);
      rows.push(
        <button
          key={n.path}
          className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-xs hover:bg-(--ink-bg-3)"
          style={{ paddingLeft: 6 + depth * 12 }}
          onClick={() => (n.isDir ? toggle(n.path) : requestOpenFile(`${project.root}/${n.path}`))}
          title={n.path}
        >
          {n.isDir ? (
            <>
              {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {isOpen ? (
                <FolderOpen size={13} className="text-(--ink-warning)" />
              ) : (
                <Folder size={13} className="text-(--ink-warning)" />
              )}
            </>
          ) : (
            <>
              <span className="w-3" />
              <File size={13} className="text-(--ink-text-dim)" />
            </>
          )}
          <span className={cn("min-w-0 flex-1 truncate", gs && gitColor(gs))}>{n.name}</span>
          {gs && <span className={cn("shrink-0 font-mono text-[10px]", gitColor(gs))}>{gs}</span>}
        </button>,
      );
      if (n.isDir && isOpen) walk(n.children, depth + 1);
    }
  };
  walk(tree, 0);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-(--ink-border) px-2 text-[11px] text-(--ink-text-dim)">
        <span className="truncate font-semibold">{project.name}</span>
        <button
          title="Refresh"
          className="flex size-5 items-center justify-center rounded hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
          onClick={reload}
        >
          <RefreshCw size={12} />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto py-1">
        {rows.length === 0 ? (
          <div className="p-2 text-xs text-(--ink-text-faint)">Empty project.</div>
        ) : (
          rows
        )}
      </div>
    </div>
  );
}

function gitColor(status: string): string {
  switch (status) {
    case "M":
      return "text-(--ink-warning)";
    case "A":
    case "?":
      return "text-(--ink-success)";
    case "D":
    case "U":
      return "text-(--ink-danger)";
    default:
      return "text-(--ink-text-dim)";
  }
}
