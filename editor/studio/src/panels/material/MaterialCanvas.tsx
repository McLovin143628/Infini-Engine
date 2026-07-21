/**
 * The Material graph editor canvas (@xyflow/react, Phase 7.2). Mirrors the
 * Blueprint canvas: the active material document renders as custom nodes + typed
 * wires; gestures become `material_apply` edits. A left column shows the live
 * preview sphere (recompiled + re-rendered after every edit) plus naga
 * diagnostics; a drawer shows the generated WGSL.
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
import { useMaterialStore } from "../../stores/materialStore";
import { NodePalette, type PaletteAnchor } from "../blueprint/NodePalette";
import { wouldCycle } from "../blueprint/reducer";
import { MatNode } from "./MatNode";
import { matPinColor } from "./materialPinTheme";
import "../blueprint/blueprint.css";
import "./material.css";

const nodeTypes = { mat: MatNode };

function edgeId(l: BpLink): string {
  return `${l.from}.${l.fromPort}->${l.to}.${l.toPort}`;
}

function deriveNodes(doc: BpDoc): Node[] {
  return Object.values(doc.graph.nodes).map((n) => ({
    id: String(n.id),
    type: "mat",
    position: { x: n.ui.x, y: n.ui.y },
    data: { typeId: n.typeId },
  }));
}

function outputType(
  doc: BpDoc,
  byId: Record<string, { outputs: { name: string; ty: PortType }[] }>,
  l: BpLink,
): PortType {
  const def = byId[doc.graph.nodes[String(l.from)]?.typeId ?? ""];
  const port = def?.outputs.find((p) => p.name === l.fromPort);
  return port?.ty ?? { kind: "wildcard" };
}

function CanvasInner() {
  const doc = useMaterialStore((s) => s.doc);
  const registryById = useMaterialStore((s) => s.registryById);
  const registry = useMaterialStore((s) => s.registry);
  const apply = useMaterialStore((s) => s.apply);
  const undo = useMaterialStore((s) => s.undo);
  const redo = useMaterialStore((s) => s.redo);
  const compiling = useMaterialStore((s) => s.compiling);
  const compileResult = useMaterialStore((s) => s.compileResult);
  const showWgsl = useMaterialStore((s) => s.showWgsl);
  const toggleWgsl = useMaterialStore((s) => s.toggleWgsl);

  const { screenToFlowPosition } = useReactFlow();
  const previewMove = useMaterialStore((s) => s.previewMove);
  const [palette, setPalette] = useState<{ anchor: PaletteAnchor; flow: { x: number; y: number } } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

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
            style: { stroke: matPinColor(outputType(doc, registryById, l)), strokeWidth: 2 },
          }))
        : [],
    [doc, registryById],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const edits: BpEdit[] = [];
      for (const c of changes) {
        if (c.type === "position" && c.position) {
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

  const errorCount = compileResult?.issues.filter((i) => i.severity === "error").length ?? 0;

  return (
    <div className="bp-canvas" ref={wrapRef}>
      <div className="bp-toolbar">
        <span className="mat-toolbar__title">Material</span>
        <span className="bp-toolbar__sep" />
        <button className="bp-btn" onClick={() => void undo()} title="Undo">
          Undo
        </button>
        <button className="bp-btn" onClick={() => void redo()} title="Redo">
          Redo
        </button>
        <button className="bp-btn" onClick={toggleWgsl}>
          {"</> WGSL"}
        </button>
        <span className="bp-toolbar__spacer" />
        <span className="mat-toolbar__status">
          {compiling ? "Compiling…" : compileResult?.ok ? "✓ compiled" : `${errorCount} error(s)`}
        </span>
        <button className="bp-btn" onClick={(e) => openPalette(e.clientX, e.clientY)}>
          + Add node
        </button>
      </div>

      <div className="mat-body">
        <div className="mat-preview">
          <div className="mat-preview__sphere">
            {compileResult?.image ? (
              <img src={compileResult.image} alt="material preview" />
            ) : (
              <div className="mat-preview__placeholder">
                {compiling ? "Rendering…" : "No preview"}
              </div>
            )}
          </div>
          <div className="mat-preview__issues">
            {compileResult?.issues.length ? (
              compileResult.issues.map((iss, i) => (
                <div key={i} className={`mat-issue mat-issue--${iss.severity}`}>
                  {iss.node != null ? `node ${iss.node}: ` : ""}
                  {iss.message}
                </div>
              ))
            ) : (
              <div className="mat-issue mat-issue--ok">No issues.</div>
            )}
          </div>
        </div>

        <div className="bp-flow mat-flow">
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
            <NodePalette
              registry={registry}
              anchor={palette.anchor}
              onPick={pickNode}
              onClose={() => setPalette(null)}
            />
          )}
        </div>
      </div>

      {showWgsl && compileResult && <pre className="bp-output__code mat-wgsl">{compileResult.wgsl}</pre>}
    </div>
  );
}

function compatible(a: PortType, b: PortType): boolean {
  if (isExec(a) || isExec(b)) return isExec(a) && isExec(b);
  if (a.kind === "wildcard" || b.kind === "wildcard") return true;
  if (a.kind === "named" && b.kind === "named") return a.name === b.name;
  return a.kind === b.kind;
}

export function MaterialCanvas() {
  const init = useMaterialStore((s) => s.init);
  const ready = useMaterialStore((s) => s.ready);
  useEffect(() => {
    void init();
  }, [init]);
  if (!ready) return <div className="bp-canvas bp-canvas--loading">Loading material…</div>;
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
