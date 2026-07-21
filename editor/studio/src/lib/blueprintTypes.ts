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

export type UiHint = "number" | "text" | "toggle" | "choice";

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

/** A structural edit (tagged `{kind, …}`, kebab-case) sent to `graph_apply`. */
export type BpEdit =
  | { kind: "add-node"; id: number; typeId: string; x: number; y: number; params?: Record<string, BpValue> }
  | { kind: "remove-node"; id: number }
  | { kind: "connect"; link: BpLink }
  | { kind: "disconnect"; link: BpLink }
  | { kind: "set-param"; id: number; name: string; value: BpValue }
  | { kind: "move-node"; id: number; x: number; y: number }
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
}

export interface GraphRunResult {
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
