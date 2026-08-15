/** Hand-written TS mirrors of the animation State Machine DTOs (camelCase to
 *  match the Ring-2 `commands/sm.rs` serde). A state machine is a plain typed
 *  model (states + transitions), NOT an `inf-graph` dataflow document — so it does
 *  not reuse `blueprintTypes.ts`.
 *
 *  ## `.inf_sm` v2 (P29.1)
 *
 *  The model grew typed parameters, condition **trees**, transition priority, an
 *  interruption policy, blend curves and per-joint blend profiles, any-state
 *  sources, nested sub-machines and state notifies. The v2 authoring UI for most
 *  of that is P29.5's; what this file has to do **now** is carry every field, so
 *  that opening a v2 machine in today's inspector and pressing save does not
 *  delete what the inspector could not draw.
 *
 *  That is the whole reason `conditions` is nullable: it is the flat AND-of-float
 *  view, present only when the tree really is one. When it is `null` the
 *  inspector shows a read-only summary and `condition` — the full tree — passes
 *  through untouched. See `dto_to_transition` in `commands/sm.rs` for the rule on
 *  the other side. */

/** Comparison operators for a transition condition (the wire string form). */
export type SmOp = ">" | "<" | ">=" | "<=" | "==" | "!=";

export const SM_OPS: SmOp[] = [">", "<", ">=", "<=", "==", "!="];

/** A declared parameter's kind. `float` is what an *undeclared* name reads as,
 *  which is v1's rule and why it is the fallback everywhere. */
export type SmParamKind = "bool" | "int" | "float" | "trigger";

export const SM_PARAM_KINDS: SmParamKind[] = ["bool", "int", "float", "trigger"];

/** How a cross-fade's linear progress is shaped. Every one is a polynomial in
 *  the engine (the pose path bans `std` transcendentals). */
export type SmBlendCurve = "linear" | "easeIn" | "easeOut" | "easeInOut" | "step";

export const SM_BLEND_CURVES: SmBlendCurve[] = [
  "linear",
  "easeIn",
  "easeOut",
  "easeInOut",
  "step",
];

/** Which in-progress cross-fade a transition may cut into. */
export type SmInterruptSource = "none" | "destination" | "sourceOrDestination";

export const SM_INTERRUPT_SOURCES: SmInterruptSource[] = [
  "none",
  "destination",
  "sourceOrDestination",
];

/** What the outgoing pose is when a transition interrupts a running fade. */
export type SmInterruptBlend = "snap" | "carry";

export const SM_INTERRUPT_BLENDS: SmInterruptBlend[] = ["snap", "carry"];

/** A state's motion, tagged by `kind`. The UI edits `clip`; blend spaces and
 *  sub-machines round-trip faithfully so a data-authored machine is never lossy
 *  on save.
 *
 *  `paramX` / `paramY` are camelCase on the wire only because `commands/sm.rs`
 *  spells them with an explicit `#[serde(rename)]`. `rename_all = "camelCase"`
 *  on a **tagged enum** renames the variants and not their fields, so every
 *  multi-word field inside `SmMotionDto` and `SmCondDto` needs its own rename —
 *  and until the P29.1 audit these two did not have one, so a 2D blend space's
 *  axis names read `undefined` here. The Rust round-trip arm now asserts every
 *  key of both enums by name, in both directions. */
export type SmMotion =
  | { kind: "clip"; clip: string | null }
  | { kind: "blend1d"; param: string; entries: { pos: number; clip: string | null }[] }
  | {
      kind: "blend2d";
      paramX: string;
      paramY: string;
      entries: { x: number; y: number; clip: string | null }[];
    }
  | { kind: "subMachine"; machine: SmMachineDto };

export interface SmStateDto {
  name: string;
  motion: SmMotion;
  looping: boolean;
  speed: number;
  x: number;
  y: number;
  /** Notify names emitted when this state is entered / exited (v2). */
  onEnter: string[];
  onExit: string[];
}

/** One row of the **flat** condition view: `var op value`. */
export interface SmConditionDto {
  var: string;
  op: SmOp;
  value: number;
}

/** A condition tree node — what P29.5's rule builder edits. */
export type SmCondDto =
  | { kind: "always" }
  | { kind: "compare"; param: string; op: SmOp; value: number; valueKind: "bool" | "int" | "float" }
  | { kind: "trigger"; param: string }
  | { kind: "and"; terms: SmCondDto[] }
  | { kind: "or"; terms: SmCondDto[] }
  | { kind: "not"; term: SmCondDto };

export interface SmTransitionDto {
  /** Source state index, or `null` for an **any-state** transition. */
  from: number | null;
  /** Any-state only: whether it may re-enter the state it targets. */
  excludeSelf: boolean;
  to: number;
  duration: number;
  /** The flat AND-of-float view, or `null` when the tree is not one. A `null`
   *  here is the signal that this transition must be edited through `condition`
   *  (or left alone) — flattening it would destroy data. */
  conditions: SmConditionDto[] | null;
  /** The full tree, always present. */
  condition: SmCondDto;
  exitTime: number | null;
  priority: number;
  interruptSource: SmInterruptSource;
  interruptBlend: SmInterruptBlend;
  curve: SmBlendCurve;
  profile: number | null;
}

export interface SmParamDto {
  name: string;
  kind: SmParamKind;
  /** The default, as a number (`bool` reads at `> 0.5`, matching the engine). */
  default: number;
}

export interface SmJointWeightDto {
  joint: number;
  weight: number;
}

export interface SmProfileDto {
  name: string;
  weights: SmJointWeightDto[];
}

export interface SmMachineDto {
  states: SmStateDto[];
  transitions: SmTransitionDto[];
  entry: number;
  params: SmParamDto[];
  profiles: SmProfileDto[];
}

export interface SmDoc {
  id: string;
  name: string;
  machine: SmMachineDto;
}

/** One imported `.inf_anim` clip, for the Clip-motion picker. */
export interface SmClipDto {
  id: string;
  name: string;
}

/** A short human summary of a state's motion for the node body. */
export function motionSummary(m: SmMotion, clipName: (id: string | null) => string): string {
  switch (m.kind) {
    case "clip":
      return m.clip ? clipName(m.clip) : "(no clip)";
    case "blend1d":
      return `Blend1D(${m.param}, ${m.entries.length})`;
    case "blend2d":
      return `Blend2D(${m.paramX},${m.paramY}, ${m.entries.length})`;
    case "subMachine":
      return `Sub-machine (${m.machine.states.length} states)`;
  }
}

/** A one-line, read-only rendering of a condition tree — what the inspector
 *  shows for a transition whose `conditions` came back `null`.
 *
 *  Read-only on purpose: a summary an author can *edit* is a summary that has to
 *  round-trip, and round-tripping a tree through a string is how the flattening
 *  this whole design avoids gets re-invented. The editable rule builder is
 *  P29.5's. */
export function conditionSummary(c: SmCondDto): string {
  switch (c.kind) {
    case "always":
      return "always";
    case "compare":
      return `${c.param} ${c.op} ${c.value}`;
    case "trigger":
      return `${c.param} (trigger)`;
    case "and":
      return c.terms.length === 0 ? "always" : c.terms.map(wrap).join(" and ");
    case "or":
      return c.terms.length === 0 ? "never" : c.terms.map(wrap).join(" or ");
    case "not":
      return `not ${wrap(c.term)}`;
  }
}

function wrap(c: SmCondDto): string {
  const s = conditionSummary(c);
  return c.kind === "and" || c.kind === "or" ? `(${s})` : s;
}

/** The transition an "add edge" gesture creates: v1's defaults, spelled in v2. */
export function newTransition(from: number, to: number): SmTransitionDto {
  return {
    from,
    excludeSelf: false,
    to,
    duration: 0.2,
    conditions: [],
    condition: { kind: "always" },
    exitTime: null,
    priority: 0,
    interruptSource: "destination",
    interruptBlend: "carry",
    curve: "linear",
    profile: null,
  };
}
