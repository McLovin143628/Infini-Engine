/**
 * First-run tour overlay (P15.3). Renders the `tourStore` state machine as a
 * dim backdrop + a highlight ring around the current step's anchor element +
 * a CSS-positioned callout card with Back / Skip / Next.
 *
 * Airspace rule: the callouts are HTML, so while the tour is up we hold a
 * viewport-overlay acquisition (hides the native child window) exactly like the
 * StartScreen / palette / dialogs.
 */
import { useLayoutEffect, useState } from "react";
import { X } from "lucide-react";

import { useViewportOverlay } from "../lib/viewportOverlay";
import { TOUR_STEPS, useTourStore } from "../stores/tourStore";

const CARD_W = 340;
const CARD_MARGIN = 14;

interface Placed {
  /** Anchor rect (viewport px), or null when the step is centered / missing. */
  ring: { x: number; y: number; w: number; h: number } | null;
  card: { left: number; top: number };
}

/** Resolve the anchor element's rect and place the callout beside it. */
function placeStep(stepIndex: number): Placed {
  const step = TOUR_STEPS[stepIndex];
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const centerCard = { left: (vw - CARD_W) / 2, top: Math.max(24, vh / 2 - 120) };

  if (!step || !step.anchor || step.placement === "center") {
    return { ring: null, card: centerCard };
  }
  const el = document.querySelector(step.anchor);
  if (!el) return { ring: null, card: centerCard };

  const r = el.getBoundingClientRect();
  const ring = { x: r.left, y: r.top, w: r.width, h: r.height };

  // Estimate the card height for clamping; the real card grows to fit.
  const cardH = 190;
  let left: number;
  let top: number;
  switch (step.placement) {
    case "left":
      left = r.left - CARD_W - CARD_MARGIN;
      top = r.top;
      break;
    case "right":
      left = r.right + CARD_MARGIN;
      top = r.top;
      break;
    case "top":
      left = r.left + r.width / 2 - CARD_W / 2;
      top = r.top - cardH - CARD_MARGIN;
      break;
    case "bottom":
    default:
      left = r.left + r.width / 2 - CARD_W / 2;
      top = r.bottom + CARD_MARGIN;
      break;
  }
  // Clamp into the viewport.
  left = Math.max(CARD_MARGIN, Math.min(left, vw - CARD_W - CARD_MARGIN));
  top = Math.max(CARD_MARGIN, Math.min(top, vh - cardH - CARD_MARGIN));
  return { ring, card: { left, top } };
}

export default function FirstRunTour() {
  const active = useTourStore((s) => s.active);
  const step = useTourStore((s) => s.step);
  const next = useTourStore((s) => s.next);
  const prev = useTourStore((s) => s.prev);
  const skip = useTourStore((s) => s.skip);

  useViewportOverlay(active);
  const [placed, setPlaced] = useState<Placed>({ ring: null, card: { left: 0, top: 0 } });

  useLayoutEffect(() => {
    if (!active) return;
    const recompute = () => setPlaced(placeStep(step));
    recompute();
    window.addEventListener("resize", recompute);
    return () => window.removeEventListener("resize", recompute);
  }, [active, step]);

  if (!active) return null;
  const current = TOUR_STEPS[step];
  if (!current) return null;

  const isLast = step === TOUR_STEPS.length - 1;
  const isFirst = step === 0;

  return (
    <div className="fixed inset-0 z-[90]" role="dialog" aria-modal="true" aria-label="First-run tour">
      {/* Dim backdrop (click-through disabled — the tour is modal). */}
      <div className="absolute inset-0 bg-(--ink-bg-0)/60" />

      {/* Highlight ring around the anchor. */}
      {placed.ring && (
        <div
          className="pointer-events-none absolute rounded-md ring-2 ring-(--ink-accent)"
          style={{
            left: placed.ring.x - 4,
            top: placed.ring.y - 4,
            width: placed.ring.w + 8,
            height: placed.ring.h + 8,
            boxShadow: "0 0 0 9999px rgba(0,0,0,0.35)",
          }}
        />
      )}

      {/* Callout card. */}
      <div
        className="absolute flex flex-col gap-2 rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1) p-4 shadow-2xl"
        style={{ left: placed.card.left, top: placed.card.top, width: CARD_W }}
      >
        <div className="flex items-start gap-2">
          <h2 className="flex-1 text-sm font-semibold text-(--ink-text)">{current.title}</h2>
          <button
            className="flex size-5 items-center justify-center rounded text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
            onClick={() => skip()}
            aria-label="Close tour"
          >
            <X size={14} />
          </button>
        </div>
        <p className="text-xs leading-relaxed text-(--ink-text-dim)">{current.body}</p>
        <div className="mt-1 flex items-center gap-2">
          <span className="text-[11px] text-(--ink-text-faint)">
            {step + 1} / {TOUR_STEPS.length}
          </span>
          <div className="flex-1" />
          <button
            className="rounded px-2 py-1 text-xs text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
            onClick={() => skip()}
          >
            Skip
          </button>
          {!isFirst && (
            <button
              className="rounded border border-(--ink-border) px-2 py-1 text-xs hover:bg-(--ink-bg-3)"
              onClick={() => prev()}
            >
              Back
            </button>
          )}
          <button
            className="rounded bg-(--ink-accent) px-3 py-1 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)"
            onClick={() => next()}
          >
            {isLast ? "Finish" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
}
