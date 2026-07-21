/**
 * Template gallery previews (P15.3): a small hand-drawn inline-SVG vignette per
 * project template, plus the template→discipline mapping that drives the
 * first-run layout preset. Everything here is keyed by the backend template
 * *slug* (`ProjectTemplateDto.slug`), so it needs no changes to the ts-rs
 * bindings — new backend templates simply fall back to a generic vignette and
 * the `blank-3d`/3D discipline until an entry is added here.
 *
 * The SVGs are self-contained (no binary images) and theme-aware: strokes use
 * `currentColor` (the card's text color) and accents read the live `--ink-*`
 * theme variables.
 */
import type { ReactNode } from "react";

import type { Discipline } from "../lib/disciplines";

export interface TemplateVisual {
  /** The natural discipline for this template (default layout on create). */
  discipline: Discipline;
  preview: ReactNode;
}

const VIEWBOX = "0 0 120 72";

/** Blank 3D — a perspective ground grid with a wireframe cube and a sun. */
function Blank3dPreview(): ReactNode {
  return (
    <svg viewBox={VIEWBOX} className="h-full w-full" role="img" aria-label="Blank 3D preview">
      <rect width="120" height="72" fill="var(--ink-bg-0)" />
      {/* horizon + converging ground grid */}
      <line x1="0" y1="42" x2="120" y2="42" stroke="var(--ink-border)" strokeWidth="0.75" />
      {[10, 30, 50, 70, 90, 110].map((x) => (
        <line key={x} x1={x} y1="42" x2={(x - 60) * 2.4 + 60} y2="72" stroke="var(--ink-border)" strokeWidth="0.5" />
      ))}
      {[52, 62, 72].map((y, i) => (
        <line key={y} x1={-i * 10} y1={y} x2={120 + i * 10} y2={y} stroke="var(--ink-border)" strokeWidth="0.5" />
      ))}
      {/* sun */}
      <circle cx="96" cy="18" r="6" fill="var(--ink-warning)" opacity="0.85" />
      {/* wireframe cube */}
      <g stroke="var(--ink-accent)" strokeWidth="1.1" fill="none">
        <rect x="40" y="26" width="20" height="20" />
        <rect x="48" y="20" width="20" height="20" />
        <line x1="40" y1="26" x2="48" y2="20" />
        <line x1="60" y1="26" x2="68" y2="20" />
        <line x1="40" y1="46" x2="48" y2="40" />
        <line x1="60" y1="46" x2="68" y2="40" />
      </g>
    </svg>
  );
}

/** 2D Platformer — side-view platforms, a character, and a coin. */
function Platformer2dPreview(): ReactNode {
  return (
    <svg viewBox={VIEWBOX} className="h-full w-full" role="img" aria-label="2D Platformer preview">
      <rect width="120" height="72" fill="var(--ink-bg-0)" />
      {/* platforms */}
      <rect x="0" y="58" width="52" height="14" fill="var(--ink-bg-3)" />
      <rect x="66" y="46" width="34" height="8" rx="1" fill="var(--ink-bg-3)" />
      <rect x="104" y="30" width="16" height="6" rx="1" fill="var(--ink-bg-3)" />
      {/* character */}
      <rect x="18" y="46" width="10" height="12" rx="1.5" fill="var(--ink-accent)" />
      <circle cx="23" cy="43" r="3.5" fill="var(--ink-accent)" />
      {/* coin */}
      <circle cx="82" cy="38" r="4" fill="var(--ink-warning)" />
      <circle cx="82" cy="38" r="1.6" fill="var(--ink-bg-0)" />
      {/* jump arc */}
      <path d="M30 46 Q46 22 64 42" stroke="var(--ink-accent-hover)" strokeWidth="0.75" strokeDasharray="2 2" fill="none" />
    </svg>
  );
}

/** First Person — a horizon, a crosshair, and stylised viewmodel hands. */
function FirstPersonPreview(): ReactNode {
  return (
    <svg viewBox={VIEWBOX} className="h-full w-full" role="img" aria-label="First Person preview">
      <rect width="120" height="72" fill="var(--ink-bg-0)" />
      {/* sky / ground split */}
      <rect x="0" y="0" width="120" height="40" fill="var(--ink-bg-1)" />
      <line x1="0" y1="40" x2="120" y2="40" stroke="var(--ink-border)" strokeWidth="0.75" />
      {/* distant blocks */}
      <rect x="14" y="24" width="14" height="16" fill="var(--ink-bg-3)" />
      <rect x="90" y="20" width="18" height="20" fill="var(--ink-bg-3)" />
      {/* viewmodel (a simple gun barrel from the corner) */}
      <path d="M78 72 L92 56 L110 62 L104 72 Z" fill="var(--ink-bg-3)" stroke="var(--ink-border-strong)" strokeWidth="0.75" />
      {/* crosshair */}
      <g stroke="var(--ink-accent)" strokeWidth="1.1">
        <line x1="54" y1="34" x2="62" y2="34" />
        <line x1="58" y1="30" x2="58" y2="38" />
      </g>
    </svg>
  );
}

/** Hybrid 2.5D — a 3D ground plane with two billboarded 2D cards. */
function Hybrid25dPreview(): ReactNode {
  return (
    <svg viewBox={VIEWBOX} className="h-full w-full" role="img" aria-label="Hybrid 2.5D preview">
      <rect width="120" height="72" fill="var(--ink-bg-0)" />
      {/* ground plane (parallelogram) */}
      <path d="M8 66 L44 44 L112 44 L88 66 Z" fill="var(--ink-bg-2)" stroke="var(--ink-border)" strokeWidth="0.6" />
      {/* billboard cards standing on the plane */}
      <rect x="36" y="24" width="16" height="22" rx="1" fill="var(--ink-accent)" opacity="0.9" />
      <rect x="66" y="20" width="14" height="26" rx="1" fill="var(--ink-info)" opacity="0.9" />
      {/* card shadows */}
      <ellipse cx="44" cy="47" rx="8" ry="1.6" fill="var(--ink-shadow)" opacity="0.5" />
      <ellipse cx="73" cy="47" rx="7" ry="1.6" fill="var(--ink-shadow)" opacity="0.5" />
      {/* sun */}
      <circle cx="100" cy="16" r="5" fill="var(--ink-warning)" opacity="0.85" />
    </svg>
  );
}

/** Generic fallback for unknown/new template slugs. */
function GenericPreview(): ReactNode {
  return (
    <svg viewBox={VIEWBOX} className="h-full w-full" role="img" aria-label="Project preview">
      <rect width="120" height="72" fill="var(--ink-bg-0)" />
      <rect x="30" y="20" width="60" height="32" rx="3" fill="none" stroke="var(--ink-border-strong)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1="45" y1="36" x2="75" y2="36" stroke="var(--ink-accent)" strokeWidth="1.2" />
      <line x1="60" y1="28" x2="60" y2="44" stroke="var(--ink-accent)" strokeWidth="1.2" />
    </svg>
  );
}

/** Per-slug visuals + discipline. Unknown slugs fall back to `genericVisual`. */
const VISUALS: Record<string, TemplateVisual> = {
  "blank-3d": { discipline: "3d", preview: <Blank3dPreview /> },
  "2d-platformer": { discipline: "2d", preview: <Platformer2dPreview /> },
  "first-person": { discipline: "3d", preview: <FirstPersonPreview /> },
  "hybrid-2.5d": { discipline: "3d", preview: <Hybrid25dPreview /> },
};

const GENERIC: TemplateVisual = { discipline: "3d", preview: <GenericPreview /> };

/** Look up a template's gallery visual + discipline by slug. */
export function templateVisual(slug: string): TemplateVisual {
  return VISUALS[slug] ?? GENERIC;
}
