/**
 * Terminal panel (P5.3): an embedded shell in a real PTY, driven by `usePty`.
 *
 * Runs in the open project's root (so `cargo check` builds the gameplay crate),
 * with task-preset buttons for the common Rust commands — the Phase 5 gate's
 * "cargo check runs in the embedded terminal".
 */
import { useRef } from "react";

import { usePty } from "../lib/usePty";
import { useProjectStore } from "../stores/projectStore";

const TASKS: { label: string; cmd: string }[] = [
  { label: "cargo check", cmd: "cargo check" },
  { label: "cargo test", cmd: "cargo test" },
  { label: "cargo build", cmd: "cargo build" },
];

export default function TerminalPanel({ panelId }: { panelId: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cwd = useProjectStore((s) => s.current?.root ?? null);
  // Key the PTY session on the panel id so a dock/float re-parent adopts the
  // same live session (keeping a running build alive) instead of respawning.
  const { runCommand, focus } = usePty(containerRef, cwd, panelId);

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-[#0e0f13]">
      <div className="flex h-7 shrink-0 items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
        {TASKS.map((t) => (
          <button
            key={t.cmd}
            className="rounded px-2 py-0.5 text-[11px] text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
            onClick={() => {
              runCommand(t.cmd);
              focus();
            }}
            title={`Run: ${t.cmd}`}
          >
            {t.label}
          </button>
        ))}
        <div className="flex-1" />
        <span className="text-[10px] text-(--ink-text-faint)">
          {cwd ? "project root" : "no project"}
        </span>
      </div>
      <div
        ref={containerRef}
        className="min-h-0 flex-1 overflow-hidden p-1"
        onClick={focus}
      />
    </div>
  );
}
