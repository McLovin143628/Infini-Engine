/**
 * The PCG graph editor canvas (@xyflow/react, Phase 10.5b). Mirrors the Material
 * canvas: the active PCG document renders as custom nodes + typed wires; gestures
 * become `pcg_apply` edits. A left column shows the lowered-document summary +
 * node-anchored diagnostics and an **Evaluate** button that scatters the graph
 * over the scene terrain into the selected `PcgVolume` (refreshing the viewport).
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
import { usePcgStore } from "../../stores/pcgStore";
import { NodePalette, type PaletteAnchor } from "../blueprint/NodePalette";
import { wouldCycle } from "../blueprint/reducer";
import { PcgNode } from "./PcgNode";
import { pcgPinColor } from "./pcgPinTheme";
import "../blueprint/blueprint.css";
import "./pcg.css";

const nodeTypes = { pcg: PcgNode };

function edgeId(l: BpLink): string {
  return `${l.from}.${l.fromPort}->${l.to}.${l.toPort}`;
}

function deriveNodes(doc: BpDoc): Node[] {
  return Object.values(doc.graph.nodes).map((n) => ({
    id: String(n.id),
    type: "pcg",
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
  const doc = usePcgStore((s) => s.doc);
  const registryById = usePcgStore((s) => s.registryById);
  const registry = usePcgStore((s) => s.registry);
  const apply = usePcgStore((s) => s.apply);
  const undo = usePcgStore((s) => s.undo);
  const redo = usePcgStore((s) => s.redo);
  const compiling = usePcgStore((s) => s.compiling);
  const compileResult = usePcgStore((s) => s.compileResult);
  const evaluate = usePcgStore((s) => s.evaluate);
  const evaluating = usePcgStore((s) => s.evaluating);
  const lastEval = usePcgStore((s) => s.lastEval);
  const save = usePcgStore((s) => s.save);

  const onSave = useCallback(async () => {
    const name = window.prompt("Save PCG graph as (asset name):", "PCG Graph");
    if (name == null) return;
    const file = await save(name);
    if (file) console.info(`saved PCG asset ${file}`);
  }, [save]);

  const { screenToFlowPosition } = useReactFlow();
  const previewMove = usePcgStore((s) => s.previewMove);
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
            style: { stroke: pcgPinColor(outputType(doc, registryById, l)), strokeWidth: 2 },
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
        <span className="pcg-toolbar__title">PCG</span>
        <span className="bp-toolbar__sep" />
        <button className="bp-btn" onClick={() => void undo()} title="Undo">
          Undo
        </button>
        <button className="bp-btn" onClick={() => void redo()} title="Redo">
          Redo
        </button>
        <button
          className="bp-btn"
          onClick={() => void onSave()}
          title="Save this graph to a .inf_pcg asset"
        >
          Save
        </button>
        <span className="bp-toolbar__spacer" />
        <span className="pcg-toolbar__status">
          {compiling ? "Compiling…" : compileResult?.ok ? "✓ compiled" : `${errorCount} error(s)`}
        </span>
        <button className="bp-btn" onClick={(e) => openPalette(e.clientX, e.clientY)}>
          + Add node
        </button>
      </div>

      <div className="pcg-body">
        <div className="pcg-side">
          <div className="pcg-side__summary">{compileResult?.summary ?? "…"}</div>
          <div className="pcg-side__eval">
            <button
              className="bp-btn"
              onClick={() => void evaluate()}
              disabled={!compileResult?.ok || evaluating}
              title="Scatter this graph over the scene terrain into the selected PCG Volume"
            >
              {evaluating ? "Evaluating…" : "⚡ Evaluate"}
            </button>
            {lastEval && (
              <div className="pcg-side__eval-result">
                {lastEval.ok
                  ? `Placed ${lastEval.placed} instance(s).`
                  : "Evaluate failed — see diagnostics."}
              </div>
            )}
          </div>
          <div className="pcg-issues">
            {compileResult?.issues.length ? (
              compileResult.issues.map((iss, i) => (
                <div key={i} className={`pcg-issue pcg-issue--${iss.severity}`}>
                  {iss.node != null ? `node ${iss.node}: ` : ""}
                  {iss.message}
                </div>
              ))
            ) : (
              <div className="pcg-issue pcg-issue--ok">No issues.</div>
            )}
          </div>
        </div>

        <div className="bp-flow pcg-flow">
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
    </div>
  );
}

function compatible(a: PortType, b: PortType): boolean {
  if (isExec(a) || isExec(b)) return isExec(a) && isExec(b);
  if (a.kind === "wildcard" || b.kind === "wildcard") return true;
  if (a.kind === "named" && b.kind === "named") return a.name === b.name;
  return a.kind === b.kind;
}

export function PcgCanvas() {
  const init = usePcgStore((s) => s.init);
  const close = usePcgStore((s) => s.close);
  const ready = usePcgStore((s) => s.ready);
  // Init on mount; free the backend document on unmount (panel close) so open
  // graphs don't accumulate for the session.
  useEffect(() => {
    void init();
    return () => {
      void close();
    };
  }, [init, close]);
  if (!ready) return <div className="bp-canvas bp-canvas--loading">Loading PCG…</div>;
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
