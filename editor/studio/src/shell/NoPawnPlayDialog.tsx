/**
 * **"This level has no player-controlled character"** (wave GTA1).
 *
 * Play on a level with no pawn is not a failure and must not be a refusal: the
 * player process starts, keeps its own overhead orthographic camera, and runs a
 * world nothing in it responds to input in. That is a legitimate thing to want
 * (a cinematic, a level being blocked out) and a bewildering thing to get by
 * accident, which is what it was — `camera_subject` returned `None`, nothing
 * said so anywhere, and the author was left looking at their furniture from
 * above wondering which key moves.
 *
 * So it is a question with two real answers:
 *
 *  * **Place Starter Character & Play** performs the REAL level edit —
 *    `character_place_starter`, one undoable document change through the same
 *    door the New Character wizard's spawn half takes — and then plays. The
 *    level that comes back is the level that ships; there is no PIE-only
 *    auto-spawn here, and there must never be, because a preview that spawns a
 *    player the build does not is the exact divergence PIE == shipping exists
 *    to forbid.
 *  * **Play Overhead** starts the session as asked.
 *
 * Airspace: `useViewportOverlay` hides the native viewport while this is up
 * (`inset-0` dim, so a cutout would punch an undimmed hole through it).
 */
import { useEffect, useState } from "react";
import { PersonStanding, X } from "lucide-react";

import { character as characterIpc } from "../lib/ipc";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { usePieStore } from "../stores/pieStore";
import { useShellStore } from "../stores/shellStore";

export default function NoPawnPlayDialog() {
  const mode = useShellStore((s) => s.noPawnPlay);
  const setMode = useShellStore((s) => s.setNoPawnPlay);
  const pushStatus = useShellStore((s) => s.pushStatus);
  const startAnyway = usePieStore((s) => s.startAnyway);

  const [busy, setBusy] = useState(false);

  useViewportOverlay(mode !== null);

  useEffect(() => {
    if (!mode) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) setMode(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, busy, setMode]);

  if (!mode) return null;

  const playOverhead = () => {
    setMode(null);
    void startAnyway(mode);
  };

  const placeAndPlay = async () => {
    setBusy(true);
    try {
      await characterIpc.placeStarter();
      pushStatus("Placed the starter character — Ctrl+Z undoes it.", 8000);
      setMode(null);
      await startAnyway(mode);
    } catch (e) {
      pushStatus(`Could not place the starter character: ${String(e)}`, 12000);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[86] flex items-start justify-center bg-black/40 pt-24"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget && !busy) setMode(null);
      }}
    >
      <div
        role="dialog"
        aria-label="This level has no player-controlled character"
        className="flex w-[460px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center gap-2 border-b border-(--ink-border) px-3 py-2">
          <PersonStanding size={15} className="text-(--ink-accent)" />
          <span className="flex-1 font-semibold">No player in this level</span>
          <button
            aria-label="Close dialog"
            disabled={busy}
            className="rounded p-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text) disabled:opacity-40"
            onClick={() => setMode(null)}
          >
            <X size={14} />
          </button>
        </div>

        <div className="p-3 text-xs leading-relaxed text-(--ink-text-dim)">
          This level has no <span className="text-(--ink-text)">player-controlled character</span>,
          so Play will run it from a fixed overhead camera and nothing in it will respond to
          input.
          <div className="mt-2">
            Placing the starter character edits the level — it is one undo step, and the build
            will contain it too.
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-(--ink-border) px-3 py-2">
          <button
            disabled={busy}
            className="rounded px-2 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text) disabled:opacity-40"
            onClick={playOverhead}
          >
            Play Overhead
          </button>
          <button
            disabled={busy}
            className="rounded bg-(--ink-accent) px-2 py-1 font-semibold text-(--ink-bg-0) hover:brightness-110 disabled:opacity-40"
            onClick={() => void placeAndPlay()}
          >
            {busy ? "Placing…" : "Place Starter Character & Play"}
          </button>
        </div>
      </div>
    </div>
  );
}
