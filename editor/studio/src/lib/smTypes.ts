/** Hand-written TS mirrors of the animation State Machine DTOs (camelCase to
 *  match the Ring-2 `commands/sm.rs` serde). A state machine is a plain typed
 *  model (states + transitions), NOT an `inf-graph` dataflow document — so it does
 *  not reuse `blueprintTypes.ts`. */

/** Comparison operators for a transition condition (the wire string form). */
export type SmOp = ">" | "<" | ">=" | "<=" | "==" | "!=";

export const SM_OPS: SmOp[] = [">", "<", ">=", "<=", "==", "!="];

/** A state's motion, tagged by `kind`. v1 UI edits only `clip`; blend spaces
 *  round-trip faithfully so a data-authored machine is never lossy on save. */
export type SmMotion =
  | { kind: "clip"; clip: string | null }
  | { kind: "blend1d"; param: string; entries: { pos: number; clip: string | null }[] }
  | {
      kind: "blend2d";
      paramX: string;
      paramY: string;
      entries: { x: number; y: number; clip: string | null }[];
    };

export interface SmStateDto {
  name: string;
  motion: SmMotion;
  looping: boolean;
  speed: number;
  x: number;
  y: number;
}

export interface SmConditionDto {
  var: string;
  op: SmOp;
  value: number;
}

export interface SmTransitionDto {
  from: number;
  to: number;
  duration: number;
  conditions: SmConditionDto[];
  exitTime: number | null;
}

export interface SmMachineDto {
  states: SmStateDto[];
  transitions: SmTransitionDto[];
  entry: number;
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
  }
}
