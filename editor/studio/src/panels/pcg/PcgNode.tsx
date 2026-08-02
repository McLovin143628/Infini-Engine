/** A PCG node: category-tinted header, typed density/scatter pins as @xyflow
 *  Handles, and inline editors for its stored parameters. Mirrors `MatNode`,
 *  adding a `choice` dropdown (the scatter rotation mode). */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { memo } from "react";

import type { BpValue, ParamDef, PortDef } from "../../lib/blueprintTypes";
import { usePcgStore } from "../../stores/pcgStore";
import { pcgCategoryColor, pcgPinColor } from "./pcgPinTheme";

interface PcgNodeData {
  typeId: string;
  [key: string]: unknown;
}

function PinRow({ port, side }: { port: PortDef; side: "target" | "source" }) {
  const color = pcgPinColor(port.ty);
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
  const apply = usePcgStore((s) => s.apply);
  const cur = paramValue(value, def);
  const commit = (v: BpValue) =>
    void apply([{ kind: "set-param", id: nodeId, name: def.name, value: v }], `Edit ${def.label}`);

  if (def.ui === "choice") {
    const s = cur.type === "enum" ? cur.value : def.options[0] ?? "";
    return (
      <label className="pcg-field pcg-field--row">
        <span className="pcg-field__label">{def.label}</span>
        <select
          className="bp-input"
          value={s}
          onChange={(e) => commit({ type: "enum", value: e.target.value })}
        >
          {def.options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (def.ui === "multiline") {
    // P19.4: an authored document (the grammar rule text) lives on its node, so
    // the node is its editor — there is deliberately no rule-text panel.
    // Uncontrolled + commit-on-blur, exactly like the single-line `text` branch,
    // so typing never round-trips through the store mid-edit.
    const s = cur.type === "text" ? cur.value : "";
    return (
      <label className="pcg-field pcg-field--multiline">
        <span className="pcg-field__label">{def.label}</span>
        <textarea
          className="bp-input nodrag nowheel"
          spellCheck={false}
          defaultValue={s}
          title={def.description ?? undefined}
          onBlur={(e) => commit({ type: "text", value: e.target.value })}
        />
      </label>
    );
  }
  if (def.ui === "text") {
    const s = cur.type === "text" ? cur.value : "";
    return (
      <label className="pcg-field">
        <span className="pcg-field__label">{def.label}</span>
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
      <label className="pcg-field pcg-field--row">
        <span className="pcg-field__label">{def.label}</span>
        <input type="checkbox" checked={b} onChange={(e) => commit({ type: "bool", value: e.target.checked })} />
      </label>
    );
  }
  // Numeric (the common case).
  const n = cur.type === "int" || cur.type === "float" ? cur.value : 0;
  return (
    <label className="pcg-field pcg-field--row">
      <span className="pcg-field__label">{def.label}</span>
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

function PcgNodeInner({ id, data, selected }: NodeProps) {
  const typeId = (data as PcgNodeData).typeId;
  const def = usePcgStore((s) => s.registryById[typeId]);
  const node = usePcgStore((s) => s.doc?.graph.nodes[id]);
  if (!def) {
    return <div className="bp-node bp-node--unknown">Unknown node: {typeId}</div>;
  }
  const nodeId = Number(id);

  return (
    <div className={`bp-node${selected ? " bp-node--selected" : ""}`}>
      <div className="bp-node__header" style={{ background: pcgCategoryColor(def.category) }}>
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

export const PcgNode = memo(PcgNodeInner);
