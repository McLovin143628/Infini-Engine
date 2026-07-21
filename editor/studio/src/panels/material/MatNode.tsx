/** A material node: category-tinted header, typed data pins as @xyflow Handles,
 *  and inline editors for its stored parameters (const/proc/tex values). */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { memo } from "react";

import type { BpValue, ParamDef, PortDef } from "../../lib/blueprintTypes";
import { useMaterialStore } from "../../stores/materialStore";
import { matCategoryColor, matPinColor } from "./materialPinTheme";

interface MatNodeData {
  typeId: string;
  [key: string]: unknown;
}

function PinRow({ port, side }: { port: PortDef; side: "target" | "source" }) {
  const color = matPinColor(port.ty);
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
          background: color,
          border: `2px solid ${color}`,
          borderRadius: 6,
        }}
      />
      <span style={{ fontSize: 11, color: "#cfd3da", opacity: port.required ? 1 : 0.8 }}>
        {port.label}
      </span>
    </div>
  );
}

function paramValue(v: BpValue | undefined, def: ParamDef): BpValue {
  return v ?? def.default;
}

function ParamField({ nodeId, def, value }: { nodeId: number; def: ParamDef; value?: BpValue }) {
  const apply = useMaterialStore((s) => s.apply);
  const cur = paramValue(value, def);
  const commit = (v: BpValue) =>
    void apply([{ kind: "set-param", id: nodeId, name: def.name, value: v }], `Edit ${def.label}`);

  if (def.ui === "text") {
    const s = cur.type === "text" ? cur.value : "";
    return (
      <label className="mat-field">
        <span className="mat-field__label">{def.label}</span>
        <input
          className="bp-input"
          defaultValue={s}
          onBlur={(e) => commit({ type: "text", value: e.target.value })}
        />
      </label>
    );
  }
  if (def.ui === "toggle") {
    const b = cur.type === "bool" ? cur.value : false;
    return (
      <label className="mat-field mat-field--row">
        <span className="mat-field__label">{def.label}</span>
        <input type="checkbox" checked={b} onChange={(e) => commit({ type: "bool", value: e.target.checked })} />
      </label>
    );
  }
  // Numeric (the common case: const/proc scalars, color channels).
  const n = cur.type === "int" || cur.type === "float" ? cur.value : 0;
  return (
    <label className="mat-field mat-field--row">
      <span className="mat-field__label">{def.label}</span>
      <input
        className="bp-input"
        type="number"
        step={def.min != null && def.max != null && def.max - def.min <= 2 ? 0.05 : 0.1}
        min={def.min ?? undefined}
        max={def.max ?? undefined}
        defaultValue={n}
        onBlur={(e) => commit({ type: "float", value: Number(e.target.value) || 0 })}
      />
    </label>
  );
}

function MatNodeInner({ id, data, selected }: NodeProps) {
  const typeId = (data as MatNodeData).typeId;
  const def = useMaterialStore((s) => s.registryById[typeId]);
  const node = useMaterialStore((s) => s.doc?.graph.nodes[id]);
  if (!def) {
    return <div className="bp-node bp-node--unknown">Unknown node: {typeId}</div>;
  }
  const nodeId = Number(id);

  return (
    <div className={`bp-node${selected ? " bp-node--selected" : ""}`}>
      <div className="bp-node__header" style={{ background: matCategoryColor(def.category) }}>
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
            <PinRow key={`out:${p.name}`} port={p} side="source" />
          ))}
        </div>
      </div>
      {def.params.length > 0 && (
        <div className="bp-node__field">
          {def.params.map((p) => (
            <ParamField key={p.name} nodeId={nodeId} def={p} value={node?.params[p.name]} />
          ))}
        </div>
      )}
    </div>
  );
}

export const MatNode = memo(MatNodeInner);
