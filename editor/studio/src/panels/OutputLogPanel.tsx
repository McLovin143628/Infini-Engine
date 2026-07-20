/**
 * Output Log (P1.4.4): the backend tracing stream with severity filter
 * chips, instant search, pause, and clear. ANSI escapes are stripped for
 * display (tool output piped through the log later carries SGR codes).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownToLine, Ban, Pause, Play, Search } from "lucide-react";
import type { LogLevel } from "../bindings/LogLevel";
import { stripAnsi } from "../lib/ansi";
import { cn } from "../lib/utils";
import { LOG_LEVELS, useLogStore } from "../stores/logStore";

const LEVEL_STYLE: Record<LogLevel, string> = {
  trace: "text-(--ink-text-faint)",
  debug: "text-(--ink-text-dim)",
  info: "text-(--ink-info)",
  warn: "text-(--ink-warning)",
  error: "text-(--ink-error)",
};

export default function OutputLogPanel() {
  const lines = useLogStore((s) => s.lines);
  const enabled = useLogStore((s) => s.enabled);
  const search = useLogStore((s) => s.search);
  const paused = useLogStore((s) => s.paused);
  const { setSearch, toggleLevel, setPaused, clear } = useLogStore.getState();

  const [follow, setFollow] = useState(true);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return lines.filter((l) => {
      if (!enabled[l.level]) return false;
      if (!needle) return true;
      return (
        stripAnsi(l.message).toLowerCase().includes(needle) ||
        l.target.toLowerCase().includes(needle)
      );
    });
  }, [lines, enabled, search]);

  useEffect(() => {
    if (!follow) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [visible, follow]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-1 border-b border-(--ink-border) p-1.5">
        <div className="flex h-6 min-w-32 flex-1 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 focus-within:border-(--ink-accent)">
          <Search size={12} className="shrink-0 text-(--ink-text-faint)" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search log…"
            className="w-full bg-transparent text-xs outline-none placeholder:text-(--ink-text-faint)"
          />
        </div>
        {LOG_LEVELS.map((level) => (
          <button
            key={level}
            className={cn(
              "h-6 rounded px-1.5 text-[11px] capitalize",
              enabled[level]
                ? cn("bg-(--ink-bg-3)", LEVEL_STYLE[level])
                : "text-(--ink-text-faint) line-through",
            )}
            onClick={() => toggleLevel(level)}
          >
            {level}
          </button>
        ))}
        <button
          title={paused ? "Resume" : "Pause"}
          aria-label={paused ? "Resume log" : "Pause log"}
          className={cn(
            "flex size-6 items-center justify-center rounded hover:bg-(--ink-bg-3)",
            paused ? "text-(--ink-warning)" : "text-(--ink-text-dim)",
          )}
          onClick={() => setPaused(!paused)}
        >
          {paused ? <Play size={13} /> : <Pause size={13} />}
        </button>
        <button
          title="Scroll to end"
          aria-label="Scroll to end"
          className={cn(
            "flex size-6 items-center justify-center rounded hover:bg-(--ink-bg-3)",
            follow ? "text-(--ink-accent)" : "text-(--ink-text-dim)",
          )}
          onClick={() => setFollow((f) => !f)}
        >
          <ArrowDownToLine size={13} />
        </button>
        <button
          title="Clear"
          aria-label="Clear log"
          className="flex size-6 items-center justify-center rounded text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-error)"
          onClick={clear}
        >
          <Ban size={13} />
        </button>
      </div>

      {/* Lines */}
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 select-text overflow-auto px-2 py-1 font-mono text-[11px] leading-4"
        onWheel={() => setFollow(false)}
      >
        {visible.length === 0 && (
          <div className="p-2 font-sans text-(--ink-text-faint)">
            {lines.length === 0 ? "Waiting for log output…" : "No lines match the filters."}
          </div>
        )}
        {visible.map((l) => (
          <div key={l.seq} className="flex gap-2 whitespace-pre-wrap break-all">
            <span className="shrink-0 text-(--ink-text-faint)">
              {new Date(l.timestamp_ms).toLocaleTimeString(undefined, { hour12: false })}
            </span>
            <span className={cn("w-10 shrink-0 uppercase", LEVEL_STYLE[l.level])}>{l.level}</span>
            <span className="shrink-0 text-(--ink-text-faint)">{l.target}</span>
            <span>{stripAnsi(l.message)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
