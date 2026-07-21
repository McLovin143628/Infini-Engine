/**
 * The Blueprint visual editor canvas (@xyflow/react). Renders the active graph
 * document as custom nodes + typed wires, translates canvas gestures into
 * `graph_apply` edits, and drives run / generate / undo / redo. (ROADMAP P6.2)
 */
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type IsValidConnection,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { BpDoc, BpEdit, BpLink, PortType } from "../../lib/blueprintTypes";
import { isExec } from "../../lib/blueprintTypes";
import { useBlueprintStore } from "../../stores/blueprintStore";
import { BpNode } from "./BpNode";
import { NodePalette, type PaletteAnchor } from "./NodePalette";
import { wouldCycle } from "./reducer";
import { pinColor } from "./pinTheme";
import "./blueprint.css";

const nodeTypes = { bp: BpNode };

function edgeId(l: BpLink): string {
  return `${l.from}.${l.fromPort}->${l.to}.${l.toPort}`;
}

function deriveNodes(doc: BpDoc): Node[] {
  return Object.values(doc.graph.nodes).map((n) => ({
    id: String(n.id),
    type: "bp",
    position: { x: n.ui.x, y: n.ui.y },
    data: { typeId: n.typeId },
  }));
}

function outputType(doc: BpDoc, byId: Record<string, { outputs: { name: string; ty: PortType }[] }>, l: BpLink): PortType {
  const def = byId[doc.graph.nodes[String(l.from)]?.typeId ?? ""];
  const port = def?.outputs.find((p) => p.name === l.fromPort);
  return port?.ty ?? { kind: "wildcard" };
}

function CanvasInner() {
  const doc = useBlueprintStore((s) => s.doc);
  const registryById = useBlueprintStore((s) => s.registryById);
  const registry = useBlueprintStore((s) => s.registry);
  const apply = useBlueprintStore((s) => s.apply);
  const run = useBlueprintStore((s) => s.run);
  const generate = useBlueprintStore((s) => s.generate);
  const undo = useBlueprintStore((s) => s.undo);
  const redo = useBlueprintStore((s) => s.redo);
  const running = useBlueprintStore((s) => s.running);
  const runResult = useBlueprintStore((s) => s.runResult);
  const generated = useBlueprintStore((s) => s.generated);
  const issues = useBlueprintStore((s) => s.issues);

  const { screenToFlowPosition } = useReactFlow();
  const previewMove = useBlueprintStore((s) => s.previewMove);
  const [palette, setPalette] = useState<{ anchor: PaletteAnchor; flow: { x: number; y: number } } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Fully controlled: nodes + edges derive from the store document, so there is
  // no local React state to keep in sync (satisfies react-hooks lint too).
  const nodes = useMemo<Node[]>(() => (doc ? deriveNodes(doc) : []), [doc]);
  const edges = useMemo<Edge[]>(
    () =>
      doc
        ? doc.graph.links.map((l) => ({
            id: edgeId(l),
            source: String(l.from),
            sourceHandle: l.fromPort,
            target: String(l.to),
            targetHandle: l.toPort,
            style: { stroke: pinColor(outputType(doc, registryById, l)), strokeWidth: 2 },
          }))
        : [],
    [doc, registryById],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const edits: BpEdit[] = [];
      for (const c of changes) {
        if (c.type === "position" && c.position) {
          // Reflect the drag locally every frame; persist on drag end.
          previewMove(Number(c.id), c.position.x, c.position.y);
          if (c.dragging === false) {
            edits.push({ kind: "move-node", id: Number(c.id), x: c.position.x, y: c.position.y });
          }
        } else if (c.type === "remove") {
          edits.push({ kind: "remove-node", id: Number(c.id) });
        }
      }
      if (edits.length) void apply(edits, "Edit nodes");
    },
    [apply, previewMove],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (!doc) return;
      const edits: BpEdit[] = [];
      for (const c of changes) {
        if (c.type === "remove") {
          const link = doc.graph.links.find((l) => edgeId(l) === c.id);
          if (link) edits.push({ kind: "disconnect", link });
        }
      }
      if (edits.length) void apply(edits, "Disconnect");
    },
    [apply, doc],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return;
      const link: BpLink = {
        from: Number(c.source),
        fromPort: c.sourceHandle,
        to: Number(c.target),
        toPort: c.targetHandle,
      };
      void apply([{ kind: "connect", link }], "Connect");
    },
    [apply],
  );

  const isValidConnection = useCallback<IsValidConnection>(
    (c) => {
      if (!doc) return false;
      const source = c.source != null ? c.source : "";
      const target = c.target != null ? c.target : "";
      if (source === target || source === "" || target === "") return false;
      const srcDef = registryById[doc.graph.nodes[source]?.typeId ?? ""];
      const dstDef = registryById[doc.graph.nodes[target]?.typeId ?? ""];
      const outTy = srcDef?.outputs.find((p) => p.name === c.sourceHandle)?.ty;
      const inTy = dstDef?.inputs.find((p) => p.name === c.targetHandle)?.ty;
      if (!outTy || !inTy) return false;
      if (!compatible(outTy, inTy)) return false;
      return !wouldCycle(doc.graph, Number(source), Number(target));
    },
    [doc, registryById],
  );

  const openPalette = useCallback(
    (clientX: number, clientY: number) => {
      const rect = wrapRef.current?.getBoundingClientRect();
      const anchor = { x: clientX - (rect?.left ?? 0), y: clientY - (rect?.top ?? 0) };
      const flow = screenToFlowPosition({ x: clientX, y: clientY });
      setPalette({ anchor, flow });
    },
    [screenToFlowPosition],
  );

  const pickNode = useCallback(
    (typeId: string) => {
      if (!doc || !palette) return;
      const id = doc.graph.nextId;
      void apply(
        [{ kind: "add-node", id, typeId, x: palette.flow.x, y: palette.flow.y, params: {} }],
        "Add node",
      );
      setPalette(null);
    },
    [apply, doc, palette],
  );

  const problemCount = issues.length;

  return (
    <div className="bp-canvas" ref={wrapRef}>
      <div className="bp-toolbar">
        <button className="bp-btn bp-btn--run" onClick={() => void run()} disabled={running}>
          {running ? "Running…" : "▶ Run"}
        </button>
        <button className="bp-btn" onClick={() => void generate()}>
          {"</> Generated Rust"}
        </button>
        <span className="bp-toolbar__sep" />
        <button className="bp-btn" onClick={() => void undo()} title="Undo">
          Undo
        </button>
        <button className="bp-btn" onClick={() => void redo()} title="Redo">
          Redo
        </button>
        <span className="bp-toolbar__spacer" />
        <button className="bp-btn" onClick={(e) => openPalette(e.clientX, e.clientY)}>
          + Add node
        </button>
        {problemCount > 0 && <span className="bp-toolbar__issues">{problemCount} issue(s)</span>}
      </div>

      <div className="bp-flow">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onPaneContextMenu={(e) => {
            e.preventDefault();
            openPalette(e.clientX, e.clientY);
          }}
          onDoubleClick={(e) => openPalette(e.clientX, e.clientY)}
          onPaneClick={() => setPalette(null)}
          deleteKeyCode={["Delete", "Backspace"]}
          minZoom={0.2}
          maxZoom={2.5}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={22} size={1.25} />
          <Controls showInteractive={false} />
          <MiniMap pannable zoomable />
        </ReactFlow>
        {palette && (
          <NodePalette registry={registry} anchor={palette.anchor} onPick={pickNode} onClose={() => setPalette(null)} />
        )}
      </div>

      {(runResult || generated) && (
        <div className="bp-output">
          {generated != null && (
            <pre className="bp-output__code">{generated}</pre>
          )}
          {runResult && (
            <div className="bp-output__run">
              {runResult.error && <div className="bp-output__err">⚠ {runResult.error}</div>}
              {runResult.logs.map((line, i) => (
                <div key={i} className="bp-output__log">
                  {line}
                </div>
              ))}
              <div className="bp-output__vars">
                {Object.entries(runResult.vars).map(([k, v]) => (
                  <span key={k} className="bp-output__var">
                    {k} = {v}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function compatible(a: PortType, b: PortType): boolean {
  if (isExec(a) || isExec(b)) return isExec(a) && isExec(b);
  if (a.kind === "wildcard" || b.kind === "wildcard") return true;
  if (a.kind === "named" && b.kind === "named") return a.name === b.name;
  return a.kind === b.kind;
}

export function BlueprintCanvas() {
  const init = useBlueprintStore((s) => s.init);
  const ready = useBlueprintStore((s) => s.ready);
  useEffect(() => {
    void init();
  }, [init]);
  if (!ready) return <div className="bp-canvas bp-canvas--loading">Loading blueprint…</div>;
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
