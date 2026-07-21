/**
 * Disciplines (P15.3): the three first-run layout flavors offered in the New
 * Project gallery and the Window ▸ Layout menu — 3D world-building, 2D, and a
 * code-forward Scripting layout. A leaf module (no React/dock imports) so both
 * the template gallery and the dock-layout store can share the type without a
 * cycle.
 */
export type Discipline = "3d" | "2d" | "scripting";

/** Human labels for the discipline chips / layout preset names. */
export const DISCIPLINE_LABEL: Record<Discipline, string> = {
  "3d": "3D",
  "2d": "2D",
  scripting: "Scripting",
};

/** One-line descriptions surfaced under each discipline chip. */
export const DISCIPLINE_HINT: Record<Discipline, string> = {
  "3d": "Place Actors, Outliner, Details, Output Log — spatial world-building.",
  "2d": "Outliner, Details, Output Log — a viewport-forward 2D layout.",
  scripting: "Explorer, Code Editor, Terminal, Problems — an IDE-forward layout.",
};

export const DISCIPLINES: Discipline[] = ["3d", "2d", "scripting"];
