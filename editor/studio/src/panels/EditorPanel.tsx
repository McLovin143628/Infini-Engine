/**
 * Code Editor panel (P5.1): a tab bar over one live CodeMirror 6 view that
 * swaps per-tab EditorState (undo/cursor preserved). Files open via the
 * `infinity:open-file` event; Ctrl/Cmd+S saves the active tab.
 */
import { useEffect, useRef } from "react";
import { FileCode2, X } from "lucide-react";

import { EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";

import { cn } from "../lib/utils";
import { setTabState, tabStateFor, useEditorStore } from "../stores/editorStore";

export default function EditorPanel() {
  const tabs = useEditorStore((s) => s.tabs);
  const activeId = useEditorStore((s) => s.activeId);
  const activate = useEditorStore((s) => s.activate);
  const closeTab = useEditorStore((s) => s.closeTab);

  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const prevIdRef = useRef<string | null>(null);

  // One live view for the panel's lifetime.
  useEffect(() => {
    if (!hostRef.current) return;
    const view = new EditorView({ parent: hostRef.current });
    viewRef.current = view;
    return () => {
      view.destroy();
      viewRef.current = null;
    };
  }, []);

  // Swap the view's state when the active tab changes.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const prev = prevIdRef.current;
    if (prev && prev !== activeId && tabStateFor(prev)) {
      setTabState(prev, view.state); // persist the outgoing tab's live state
    }
    prevIdRef.current = activeId;
    const next = activeId ? tabStateFor(activeId) : undefined;
    view.setState(next ?? EditorState.create({ doc: "" }));
  }, [activeId]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Tab bar */}
      <div className="flex h-8 shrink-0 items-center overflow-x-auto border-b border-(--ink-border) bg-(--ink-bg-2)">
        {tabs.length === 0 && (
          <span className="px-3 text-xs text-(--ink-text-faint)">No open files</span>
        )}
        {tabs.map((t) => (
          <div
            key={t.id}
            className={cn(
              "group flex h-8 shrink-0 cursor-pointer items-center gap-1.5 border-r border-(--ink-border) px-2.5 text-xs",
              t.id === activeId
                ? "bg-(--ink-bg-1) text-(--ink-text)"
                : "text-(--ink-text-dim) hover:bg-(--ink-bg-3)",
            )}
            onClick={() => activate(t.id)}
          >
            <FileCode2 size={12} className="text-(--ink-text-faint)" />
            <span className="max-w-40 truncate">{t.name}</span>
            {t.dirty && <span className="size-1.5 rounded-full bg-(--ink-accent)" />}
            <button
              aria-label={`Close ${t.name}`}
              className="ml-0.5 flex size-4 items-center justify-center rounded opacity-0 hover:bg-(--ink-bg-3) group-hover:opacity-100"
              onClick={(e) => {
                e.stopPropagation();
                closeTab(t.id);
              }}
            >
              <X size={11} />
            </button>
          </div>
        ))}
      </div>

      {/* Editor host */}
      <div className="relative min-h-0 flex-1">
        {tabs.length === 0 && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center text-sm text-(--ink-text-faint)">
            Open a file from the Explorer or Search.
          </div>
        )}
        <div ref={hostRef} className="h-full w-full overflow-hidden" />
      </div>
    </div>
  );
}
