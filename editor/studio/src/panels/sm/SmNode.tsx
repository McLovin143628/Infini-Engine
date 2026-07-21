/**
 * A State-Machine **state** node (P11.2). Unlike the dataflow node components
 * (typed exec/data pins), a state has just two connection points — a target
 * (incoming transitions, left) and a source (outgoing transitions, right) — since
 * transitions are free edges, not typed wires. The body shows the state name, a
 * one-line motion summary, and an "entry" badge on the start state.
 */
import { Handle, Position, type NodeProps } from "@xyflow/react";

export interface SmNodeData {
  name: string;
  summary: string;
  isEntry: boolean;
  [key: string]: unknown;
}

export function SmNode({ data, selected }: NodeProps) {
  const d = data as SmNodeData;
  return (
    <div className={`sm-node${selected ? " sm-node--selected" : ""}`}>
      <Handle type="target" position={Position.Left} className="sm-handle" />
      <div className="sm-node__title">
        {d.name}
        {d.isEntry && <span className="sm-node__entry">entry</span>}
      </div>
      <div className="sm-node__motion">{d.summary}</div>
      <Handle type="source" position={Position.Right} className="sm-handle" />
    </div>
  );
}
