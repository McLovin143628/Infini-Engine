/**
 * TypeScript mirrors of the `inf-graph` / `inf-blueprint` serde shapes that the
 * Blueprint editor exchanges with the backend. These types are hand-written
 * (not ts-rs generated) because their source of truth is the Ring-0 graph
 * crate, not `inf_editor_core::ipc`. Field casing matches the Rust `serde`
 * derives exactly: node/param definitions and the graph doc are camelCase;
 * tagged enums use their discriminant strings.
 */

/** A wire/pin type. Exec is the white execution-flow pin. */
export type PortType =
  | { kind: "exec" }
  | { kind: "bool" }
  | { kind: "int" }
  | { kind: "float" }
  | { kind: "str" }
  | { kind: "wildcard" }
  | { kind: "named"; name: string };

/** A stored scalar parameter value (tagged `{type, value}`). */
export type BpValue =
  | { type: "bool"; value: boolean }
  | { type: "int"; value: number }
  | { type: "float"; value: number }
  | { type: "text"; value: string }
  | { type: "enum"; value: string };

/** Which inspector control renders a param. `multiline` (P19.4) stores the same
 *  `text` value; only the control differs — see `UiHint` in `inf-graph`. */
export type UiHint = "number" | "text" | "toggle" | "choice" | "multiline";

export interface PortDef {
  name: string;
  label: string;
  ty: PortType;
  required: boolean;
  paramPin: boolean;
}

export interface ParamDef {
  name: string;
  label: string;
  description?: string | null;
  default: BpValue;
  min?: number | null;
  max?: number | null;
  options: string[];
  ui: UiHint;
}

export interface NodeDef {
  typeId: string;
  display: string;
  category: string;
  description: string;
  inputs: PortDef[];
  outputs: PortDef[];
  params: ParamDef[];
  flags: number;
}

/** Node behavior flag bits (mirror inf_graph registry consts). */
export const FLAG_SINK = 1;
export const FLAG_HIDDEN = 2;
export const FLAG_COMMENT = 4;
export const FLAG_PASSTHROUGH = 8;

export interface NodeUi {
  x: number;
  y: number;
  title: string;
  w: number;
  h: number;
  color?: string | null;
}

export interface BpNode {
  id: number;
  typeId: string;
  params: Record<string, BpValue>;
  disabled: boolean;
  ui: NodeUi;
}

export interface BpLink {
  from: number;
  fromPort: string;
  to: number;
  toPort: string;
}

export interface BpGraph {
  nodes: Record<string, BpNode>;
  links: BpLink[];
  nextId: number;
}

export interface GraphViewport {
  x: number;
  y: number;
  zoom: number;
}

export interface BpDoc {
  id: string;
  name: string;
  graph: BpGraph;
  viewport?: GraphViewport | null;
  modifiedMs: number;
}

/**
 * One handler of an actor class (`graph_open_actor`, Wave E).
 *
 * `raisable` is false when the handler's IR is outside lowering's image —
 * `raise` only inverts what `lower` produces — and `reason` then says so.
 * A non-raisable handler is offered and DISABLED, never hidden: hiding it would
 * make the actor look like it has fewer handlers than it has.
 */
export interface ActorHandlerDto {
  key: string;
  label: string;
  raisable: boolean;
  reason?: string | null;
}

/** `graph_open_actor`'s reply: the opened graph plus the class's handler list. */
export interface ActorGraphDto {
  doc: BpDoc;
  assetId: string;
  className: string;
  handler: string;
  handlers: ActorHandlerDto[];
}

/** A structural edit (tagged `{kind, …}`, kebab-case) sent to `graph_apply`.
 *
 *  **Every variant of `inf_graph::GraphEdit`, and the Rust side is the source of
 *  truth.** Four of them — `restore-node`, `clear-param`, `set-disabled`,
 *  `resize-node` — were missing here until the L7.M7 drift gate
 *  (`src-tauri/src/wire_mirror.rs`) first ran: the backend has applied them
 *  since P6.1 and no canvas could ask for them, with nothing red anywhere to
 *  say so. Adding a variant in Rust fails that gate until it is added here. */
export type BpEdit =
  | { kind: "add-node"; id: number; typeId: string; x: number; y: number; params?: Record<string, BpValue> }
  | { kind: "remove-node"; id: number }
  | { kind: "restore-node"; node: BpNode; links: BpLink[] }
  | { kind: "connect"; link: BpLink }
  | { kind: "disconnect"; link: BpLink }
  | { kind: "set-param"; id: number; name: string; value: BpValue }
  | { kind: "clear-param"; id: number; name: string }
  | { kind: "set-disabled"; id: number; disabled: boolean }
  | { kind: "move-node"; id: number; x: number; y: number }
  | { kind: "resize-node"; id: number; w: number; h: number }
  | { kind: "set-title"; id: number; title: string };

/** A non-fatal structural issue (tagged `{kind, …}`, camelCase). */
export type BpIssue =
  | { kind: "missingInput"; node: number; port: string }
  | { kind: "unknownType"; node: number; typeId: string }
  | { kind: "orphan"; node: number }
  | { kind: "typeMismatch"; link: number }
  | { kind: "noSink" };

export interface GraphApplyResult {
  issues: BpIssue[];
  canUndo: boolean;
  canRedo: boolean;
  /**
   * **How many of the submitted edits the backend REFUSED** (C4-44 / R2.F1).
   *
   * `apply_edit` answers `false` for a `connect` that would cycle, a missing
   * node or an unknown param. These canvases are fully controlled and apply
   * optimistically, so a non-zero count means the picture on screen and the
   * graph in the backend disagree — and `issues` cannot say so, because a wire
   * that is not in the graph has nothing to be invalid about.
   *
   * The field reached the wire in Wave F and stopped at this type: it was
   * declared nowhere and read by nobody, so the desync it exists to end was
   * fully intact. Every store now re-derives from the backend when it is
   * non-zero.
   */
  rejected: number;
}

export interface GraphRunResult {
  logs: string[];
  vars: Record<string, number>;
  handlers: string[];
  error?: string | null;
}

/** One inspected wire in a debug run (B-P4): source node/port + its value. */
export interface DebugWire {
  node: number;
  port: string;
  value: string;
}

/**
 * Result of a debug preview run (`graph_debug_run`). A preview run is
 * milliseconds, so there is no live pause: the graph runs to completion under
 * the interpreter's trace and the debugger *displays* the collected hits + wire
 * values. `hits` are canvas node ids a breakpoint paused on.
 */
export interface DebugRunResult {
  hits: number[];
  wires: DebugWire[];
  logs: string[];
  vars: Record<string, number>;
  handlers: string[];
  error?: string | null;
}

/** A pin's stable string key (for theming). */
export function portKey(ty: PortType): string {
  return ty.kind === "named" ? ty.name : ty.kind;
}

export function isExec(ty: PortType): boolean {
  return ty.kind === "exec";
}
