/**
 * Output Log (P1.4.4): the backend tracing stream with severity filter
 * chips, instant search, pause, and clear. ANSI escapes are stripped for
 * display (tool output piped through the log later carries SGR codes).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownToLine, Ban, Pause, Play, Search } from "lucide-react";
import { useVirtualizer } from "@tanstack/react-virtual";
import type { LogLevel } from "../bindings/LogLevel";
import type { LogLine } from "../bindings/LogLine";
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

/** How long typing settles before the 5000-line filter re-runs (F-lens L7.M12). */
const SEARCH_DEBOUNCE_MS = 150;

/**
 * Per-line derived text, computed once (F-lens L7.M12).
 *
 * `stripAnsi` is a global regex over the whole message. The filter ran it on
 * **every line, on every flush** — the batcher flushes ~16×/s and the ring holds
 * 5000 — and the row renderer ran it again for each visible row. A `LogLine` is
 * immutable and reaches the store by reference, so its stripped form can only
 * be computed once; a `WeakMap` means the entry dies with the line when the ring
 * trims it, with no cap to tune and nothing to evict.
 *
 * `needle` folds the message AND the target into one lower-cased string so the
 * search test is a single `includes` rather than two `toLowerCase()` calls per
 * line per keystroke.
 */
interface LineText {
  message: string;
  needle: string;
}
const LINE_TEXT = new WeakMap<LogLine, LineText>();

function lineText(l: LogLine): LineText {
  let t = LINE_TEXT.get(l);
  if (!t) {
    const message = stripAnsi(l.message);
    t = { message, needle: `${message}\n${l.target}`.toLowerCase() };
    LINE_TEXT.set(l, t);
  }
  return t;
}

/**
 * The lines `next` gained over `prev`, or `null` when `next` is not `prev`
 * plus a head-trim and a tail-append (F-lens L7.M12).
 *
 * `logStore.appendMany` merges a batch and then trims the front to `LOG_CAP` —
 * a capped ring — so consecutive `lines` arrays overlap in a run. Recognising
 * that run turns a flush from "re-test all 5000" into "test the three that
 * arrived". The whole overlap is verified element by element (reference
 * identity — the store keeps the very objects the events delivered), because a
 * mis-detected overlap would silently drop lines from the panel.
 *
 * Exported for the unit test that holds the recognition honest.
 */
export function appendedLines(prev: LogLine[], next: LogLine[]): LogLine[] | null {
  if (prev.length === 0) return next;
  const j = next.lastIndexOf(prev[prev.length - 1]!);
  if (j < 0) return null;
  const trim = prev.length - 1 - j;
  if (trim < 0) return null;
  for (let i = 0; i <= j; i++) {
    if (prev[trim + i] !== next[i]) return null;
  }
  return next.slice(j + 1);
}

export default function OutputLogPanel() {
  const lines = useLogStore((s) => s.lines);
  const enabled = useLogStore((s) => s.enabled);
  const search = useLogStore((s) => s.search);
  const paused = useLogStore((s) => s.paused);
  const { setSearch, toggleLevel, setPaused, clear } = useLogStore.getState();

  const [follow, setFollow] = useState(true);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // The search box holds its own text and pushes it to the store on a timer
  // (F-lens L7.M12). Typing "error" used to re-filter the whole ring five
  // times — and, because `search` is bridged, ship five store patches. The
  // draft is deliberately not re-synced from the store: a detached window's
  // `setSearch` forwards to main and comes back, and snapping the caret onto a
  // late echo is worse than the two staying briefly apart.
  const [draft, setDraft] = useState(search);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onSearchChange = (v: string) => {
    setDraft(v);
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => setSearch(v), SEARCH_DEBOUNCE_MS);
  };
  useEffect(
    () => () => {
      if (debounce.current) clearTimeout(debounce.current);
    },
    [],
  );

  // The previous filter result, so a flush extends it instead of redoing it.
  // A cache, not state: nothing renders because of it, it is read and written
  // only inside the `useMemo` below, and a miss is always safe (full re-filter).
  const filtered = useRef<{
    lines: LogLine[];
    enabled: Record<LogLevel, boolean>;
    needle: string;
    visible: LogLine[];
  } | null>(null);

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    // `lineText` is memoized per line, so a test is a level check plus at most
    // one `includes` over a string that was lower-cased once in its life — no
    // regex and no allocation.
    const keep = (l: LogLine) =>
      enabled[l.level] && (!needle || lineText(l).needle.includes(needle));

    const prev = filtered.current;
    // Only the LINES may have moved: a filter change invalidates every verdict,
    // so those fall through to the full pass.
    const appended =
      prev && prev.enabled === enabled && prev.needle === needle
        ? appendedLines(prev.lines, lines)
        : null;

    let out: LogLine[];
    if (prev && appended) {
      // Drop whatever the ring trimmed off the front. `seq` is a per-session
      // monotonic counter, so "older than the oldest line still held" is exactly
      // the set that left.
      const oldest = lines.length > 0 ? lines[0]!.seq : Number.POSITIVE_INFINITY;
      let head = 0;
      while (head < prev.visible.length && prev.visible[head]!.seq < oldest) head += 1;
      const tail = appended.filter(keep);
      out =
        head === 0 && tail.length === 0
          ? prev.visible
          : [...prev.visible.slice(head), ...tail];
    } else {
      out = lines.filter(keep);
    }

    filtered.current = { lines, enabled, needle, visible: out };
    return out;
  }, [lines, enabled, search]);

  // Row virtualization: only the visible window of rows is in the DOM, so the
  // panel's render cost is bounded by viewport size, not the 5k-line ring —
  // even while it's a hidden tab still subscribed to the store.
  // TanStack Virtual returns non-memoizable functions; none are passed to
  // memoized children, so the React-Compiler warning is moot.
  // eslint-disable-next-line react-hooks/incompatible-library
  const rowVirt = useVirtualizer({
    count: visible.length,
    getScrollElement: () => scrollRef.current,
    // Lines wrap (break-all), so heights vary — estimate one row, then let
    // `measureElement` correct each mounted row.
    estimateSize: () => 16,
    overscan: 16,
  });

  // Follow-tail: keep pinned to the newest line unless the user scrolled up.
  useEffect(() => {
    if (!follow || visible.length === 0) return;
    rowVirt.scrollToIndex(visible.length - 1, { align: "end" });
  }, [visible.length, follow, rowVirt]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-1 border-b border-(--ink-border) p-1.5">
        <div className="flex h-6 min-w-32 flex-1 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 focus-within:border-(--ink-accent)">
          <Search size={12} className="shrink-0 text-(--ink-text-faint)" />
          <input
            value={draft}
            onChange={(e) => onSearchChange(e.target.value)}
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
        // Scrolling UP unpins the tail; scrolling down (or the "scroll to end"
        // button) leaves follow intact. Preserves the old pause-on-scroll intent.
        onWheel={(e) => {
          if (e.deltaY < 0) setFollow(false);
        }}
      >
        {visible.length === 0 ? (
          <div className="p-2 font-sans text-(--ink-text-faint)">
            {lines.length === 0 ? "Waiting for log output…" : "No lines match the filters."}
          </div>
        ) : (
          <div style={{ height: rowVirt.getTotalSize(), position: "relative", width: "100%" }}>
            {rowVirt.getVirtualItems().map((vr) => {
              const l = visible[vr.index]!;
              return (
                <div
                  key={l.seq}
                  data-index={vr.index}
                  ref={rowVirt.measureElement}
                  className="absolute left-0 flex w-full gap-2 whitespace-pre-wrap break-all"
                  style={{ top: 0, transform: `translateY(${vr.start}px)` }}
                >
                  <span className="shrink-0 text-(--ink-text-faint)">
                    {new Date(l.timestamp_ms).toLocaleTimeString(undefined, { hour12: false })}
                  </span>
                  <span className={cn("w-10 shrink-0 uppercase", LEVEL_STYLE[l.level])}>
                    {l.level}
                  </span>
                  <span className="shrink-0 text-(--ink-text-faint)">{l.target}</span>
                  <span>{lineText(l).message}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
