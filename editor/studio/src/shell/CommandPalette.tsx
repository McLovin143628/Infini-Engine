/**
 * Global command palette (P1.4.5, Ctrl+Shift+P): fuzzy search over every
 * registered command — the entire menu tree is reachable from the keyboard
 * (ROADMAP §4 "global command palette covering every menu action").
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Search } from "lucide-react";
import { allCommands, executeCommand, onCommandsChanged, type CommandDef } from "../lib/commands";
import { fuzzyFilter } from "../lib/fuzzy";
import { cn } from "../lib/utils";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useShellStore } from "../stores/shellStore";

const MAX_RESULTS = 60;

export default function CommandPalette() {
  const open = useShellStore((s) => s.paletteOpen);
  // Mounting the inner panel only while open gives every invocation fresh
  // query/cursor state without reset effects.
  if (!open) return null;
  return <PaletteInner />;
}

function PaletteInner() {
  const setOpen = useShellStore((s) => s.setPaletteOpen);
  // The palette overlays the native viewport hole — hide it while open.
  useViewportOverlay(true);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const [commands, setCommands] = useState(allCommands);
  const listRef = useRef<HTMLDivElement | null>(null);

  // Re-list when commands register (panels/phases add commands lazily).
  useEffect(() => onCommandsChanged(() => setCommands(allCommands())), []);

  const results = useMemo(
    () =>
      fuzzyFilter(query, commands, (c) => `${c.category}: ${c.title}`).slice(0, MAX_RESULTS),
    [query, commands],
  );

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>('[data-cursor="true"]');
    el?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  const run = (cmd: CommandDef) => {
    setOpen(false);
    executeCommand(cmd.id);
  };

  return (
    <div
      className="fixed inset-0 z-[86] flex items-start justify-center bg-black/30 pt-24"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
    >
      <div
        className="w-[560px] overflow-hidden rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center gap-2 border-b border-(--ink-border) px-3 py-2">
          <Search size={15} className="shrink-0 text-(--ink-text-faint)" />
          <input
            autoFocus
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setCursor(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") setOpen(false);
              else if (e.key === "ArrowDown") {
                e.preventDefault();
                setCursor((c) => Math.min(results.length - 1, c + 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setCursor((c) => Math.max(0, c - 1));
              } else if (e.key === "Enter" && results[cursor]) {
                run(results[cursor]);
              }
            }}
            placeholder="Type a command…"
            className="w-full bg-transparent outline-none placeholder:text-(--ink-text-faint)"
          />
        </div>
        <div ref={listRef} className="max-h-96 overflow-auto py-1">
          {results.length === 0 && (
            <div className="px-3 py-4 text-xs text-(--ink-text-dim)">No matching commands.</div>
          )}
          {results.map((cmd, i) => (
            <button
              key={cmd.id}
              data-cursor={i === cursor || undefined}
              className={cn(
                "flex w-full items-center gap-2 px-3 py-1.5 text-left",
                i === cursor ? "bg-(--ink-selection)" : "hover:bg-(--ink-bg-3)",
              )}
              onPointerMove={() => setCursor(i)}
              onClick={() => run(cmd)}
            >
              <span className="w-20 shrink-0 truncate text-[11px] text-(--ink-text-faint)">
                {cmd.category}
              </span>
              <span className="min-w-0 flex-1 truncate text-xs">{cmd.title}</span>
              {cmd.shortcut && (
                <span className="shrink-0 text-[11px] text-(--ink-text-faint)">{cmd.shortcut}</span>
              )}
              {!cmd.run && (
                <span className="shrink-0 rounded bg-(--ink-bg-3) px-1 text-[10px] text-(--ink-text-faint)">
                  soon
                </span>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
