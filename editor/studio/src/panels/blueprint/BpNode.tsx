/** A blueprint node: category-tinted header, exec/data pins as @xyflow
 *  Handles, and inline editing for literal nodes. */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { memo } from "react";

import type { BpValue, ParamDef, PortDef } from "../../lib/blueprintTypes";
import { isExec } from "../../lib/blueprintTypes";
import { useBlueprintStore } from "../../stores/blueprintStore";
import { categoryColor, pinColor } from "./pinTheme";

interface BpNodeData {
  typeId: string;
  [key: string]: unknown;
}

function PinRow({
  port,
  side,
  value,
}: {
  port: PortDef;
  side: "target" | "source";
  /** Captured wire value (source data pins only) → shown as a hover chip. */
  value?: string;
}) {
  const color = pinColor(port.ty);
  const exec = isExec(port.ty);
  return (
    <div
      style={{
        display: "flex",
        flexDirection: side === "target" ? "row" : "row-reverse",
        alignItems: "center",
        gap: 6,
        position: "relative",
        minHeight: 18,
      }}
    >
      <Handle
        type={side}
        position={side === "target" ? Position.Left : Position.Right}
        id={port.name}
        style={{
          width: 10,
          height: 10,
          background: exec ? "transparent" : color,
          border: `2px solid ${color}`,
          borderRadius: exec ? 2 : 6,
        }}
      />
      <span style={{ fontSize: 11, color: "#cfd3da", opacity: port.required ? 1 : 0.8 }}>
        {port.label}
        {port.required && !port.paramPin ? " *" : ""}
      </span>
      {value !== undefined && (
        <span className="bp-pin__chip" title={`${port.label} = ${value}`}>
          {value}
        </span>
      )}
    </div>
  );
}

function LiteralField({ nodeId, typeId, value }: { nodeId: number; typeId: string; value?: BpValue }) {
  const apply = useBlueprintStore((s) => s.apply);
  const commit = (v: BpValue) =>
    void apply([{ kind: "set-param", id: nodeId, name: "value", value: v }], "Edit value");

  if (typeId === "lit.bool") {
    const b = value?.type === "bool" ? value.value : false;
    return (
      <input
        type="checkbox"
        checked={b}
        onChange={(e) => commit({ type: "bool", value: e.target.checked })}
      />
    );
  }
  if (typeId === "lit.str") {
    const s = value?.type === "text" ? value.value : "";
    return (
      <input
        className="bp-input"
        defaultValue={s}
        onBlur={(e) => commit({ type: "text", value: e.target.value })}
      />
    );
  }
  const isInt = typeId === "lit.int";
  const n = value?.type === "int" || value?.type === "float" ? value.value : 0;
  return (
    <input
      className="bp-input"
      type="number"
      defaultValue={n}
      onBlur={(e) =>
        commit(
          isInt
            ? { type: "int", value: Math.trunc(Number(e.target.value) || 0) }
            : { type: "float", value: Number(e.target.value) || 0 },
        )
      }
    />
  );
}

/** A generic inline editor for a text-kind node param (Wave 3): covers
 *  `event.input`'s `action` and `event.custom`'s `name`. Value derives from the
 *  store; the edit commits through the same `set-param` action `LiteralField`
 *  uses (no local React state, so no set-state-in-effect). */
function ParamTextField({
  nodeId,
  param,
  value,
}: {
  nodeId: number;
  param: ParamDef;
  value?: BpValue;
}) {
  const apply = useBlueprintStore((s) => s.apply);
  const current = value?.type === "text" || value?.type === "enum" ? value.value : "";
  return (
    <label className="bp-node__param">
      <span className="bp-node__param-label">{param.label}</span>
      <input
        className="bp-input"
        // Keyed by the store value so an external change re-syncs the field
        // without a controlled-input effect.
        key={current}
        defaultValue={current}
        onBlur={(e) =>
          void apply(
            [{ kind: "set-param", id: nodeId, name: param.name, value: { type: "text", value: e.target.value } }],
            `Edit ${param.label}`,
          )
        }
      />
    </label>
  );
}

function BpNodeInner({ id, data, selected }: NodeProps) {
  const typeId = (data as BpNodeData).typeId;
  const def = useBlueprintStore((s) => s.registryById[typeId]);
  const node = useBlueprintStore((s) => s.doc?.graph.nodes[id]);
  const nodeId = Number(id);
  // ── B-P4 debugger state for this node ──
  const hasBreakpoint = useBlueprintStore((s) => s.debugBreakpoints.has(nodeId));
  const isHit = useBlueprintStore((s) => s.debugHits.has(nodeId));
  const wireValues = useBlueprintStore((s) => s.debugWireValues[nodeId]);
  const toggleBreakpoint = useBlueprintStore((s) => s.toggleBreakpoint);
  if (!def) {
    return <div className="bp-node bp-node--unknown">Unknown node: {typeId}</div>;
  }
  const isLiteral = typeId.startsWith("lit.");

  return (
    <div
      className={`bp-node${selected ? " bp-node--selected" : ""}${isHit ? " bp-node--hit" : ""}`}
    >
      <div
        className="bp-node__header"
        style={{ background: categoryColor(def.category) }}
        // Alt-click the header toggles a breakpoint (stop propagation so the
        // node isn't selected/dragged by the same gesture).
        onClickCapture={(e) => {
          if (e.altKey) {
            e.stopPropagation();
            e.preventDefault();
            toggleBreakpoint(nodeId);
          }
        }}
        title="Alt-click to toggle a breakpoint"
      >
        {hasBreakpoint && <span className="bp-node__bp" aria-label="breakpoint" />}
        <span className="bp-node__title">{node?.ui.title || def.display}</span>
        <span className="bp-node__cat">{def.category}</span>
      </div>
      <div className="bp-node__body">
        <div className="bp-node__col">
          {def.inputs.map((p) => (
            <PinRow key={`in:${p.name}`} port={p} side="target" />
          ))}
        </div>
        <div className="bp-node__col bp-node__col--out">
          {def.outputs.map((p) => (
            <PinRow
              key={`out:${p.name}`}
              port={p}
              side="source"
              value={wireValues?.[p.name]}
            />
          ))}
        </div>
      </div>
      {isLiteral && (
        <div className="bp-node__field">
          <LiteralField nodeId={nodeId} typeId={typeId} value={node?.params.value} />
        </div>
      )}
      {/* Wave 3: generic inline editors for any text-kind param on a non-literal
          node (event.input `action`, event.custom `name`). */}
      {!isLiteral &&
        def.params
          .filter((p) => p.ui === "text")
          .map((p) => (
            <div className="bp-node__field" key={`param:${p.name}`}>
              <ParamTextField nodeId={nodeId} param={p} value={node?.params[p.name]} />
            </div>
          ))}
    </div>
  );
}

export const BpNode = memo(BpNodeInner);
