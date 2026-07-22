/**
 * The animation State Machine editor canvas (@xyflow/react, P11.2).
 *
 * A state machine is a **plain typed model**, not a dataflow graph: nodes are
 * states, edges are transitions (free edges — any node to any other). The canvas
 * renders `smStore`'s document directly and turns gestures into local model
 * edits; a right-hand inspector edits the selected state (name / entry / clip /
 * speed / looping) or the selected transition (duration / exit time / conditions).
 *
 * v1 scope (documented in the ROADMAP): a state's **Clip** motion is authorable
 * here via the imported-clip dropdown; **blend-space** motions round-trip
 * faithfully but are authored in code/data (their in-panel editor is v2).
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
  type Node,
  type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useSmStore } from "../../stores/smStore";
import { SM_OPS, motionSummary, type SmDoc } from "../../lib/smTypes";
import { SmNode } from "./SmNode";
import "../blueprint/blueprint.css";
import "./sm.css";

const nodeTypes = { smState: SmNode };

function deriveNodes(doc: SmDoc, clipName: (id: string | null) => string): Node[] {
  return doc.machine.states.map((s, i) => ({
    id: String(i),
    type: "smState",
    position: { x: s.x, y: s.y },
    data: {
      name: s.name,
      summary: motionSummary(s.motion, clipName),
      isEntry: doc.machine.entry === i,
    },
  }));
}

function deriveEdges(doc: SmDoc, selected: number | null): Edge[] {
  return doc.machine.transitions.map((t, i) => {
    const conds = t.conditions.length;
    const label =
      (conds > 0 ? `${conds} cond` : "") +
      (t.exitTime != null ? `${conds > 0 ? " · " : ""}exit ${t.exitTime}` : "");
    return {
      id: String(i),
      source: String(t.from),
      target: String(t.to),
      label: label || undefined,
      animated: i === selected,
      style: {
        stroke: i === selected ? "var(--ink-accent)" : "var(--ink-text-dim)",
        strokeWidth: i === selected ? 2.5 : 1.5,
      },
    };
  });
}

function CanvasInner() {
  const doc = useSmStore((s) => s.doc);
  const clips = useSmStore((s) => s.clips);
  const clipName = useSmStore((s) => s.clipName);
  const selectedTransition = useSmStore((s) => s.selectedTransition);
  const saving = useSmStore((s) => s.saving);

  const addState = useSmStore((s) => s.addState);
  const moveState = useSmStore((s) => s.moveState);
  const deleteState = useSmStore((s) => s.deleteState);
  const renameState = useSmStore((s) => s.renameState);
  const setEntry = useSmStore((s) => s.setEntry);
  const setStateClip = useSmStore((s) => s.setStateClip);
  const setStateSpeed = useSmStore((s) => s.setStateSpeed);
  const setStateLooping = useSmStore((s) => s.setStateLooping);

  const addTransition = useSmStore((s) => s.addTransition);
  const deleteTransition = useSmStore((s) => s.deleteTransition);
  const selectTransition = useSmStore((s) => s.selectTransition);
  const setTransitionDuration = useSmStore((s) => s.setTransitionDuration);
  const setTransitionExitTime = useSmStore((s) => s.setTransitionExitTime);
  const addCondition = useSmStore((s) => s.addCondition);
  const updateCondition = useSmStore((s) => s.updateCondition);
  const removeCondition = useSmStore((s) => s.removeCondition);
  const save = useSmStore((s) => s.save);

  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const { screenToFlowPosition } = useReactFlow();
  const wrapRef = useRef<HTMLDivElement>(null);

  const nodes = useMemo<Node[]>(() => (doc ? deriveNodes(doc, clipName) : []), [doc, clipName]);
  const edges = useMemo<Edge[]>(
    () => (doc ? deriveEdges(doc, selectedTransition) : []),
    [doc, selectedTransition],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      for (const c of changes) {
        if (c.type === "position" && c.position && c.dragging === false) {
          moveState(Number(c.id), c.position.x, c.position.y);
        } else if (c.type === "remove") {
          // Handled by onNodesDelete (which reindexes correctly).
        }
      }
    },
    [moveState],
  );

  const onNodesDelete = useCallback(
    (deleted: Node[]) => {
      // Delete highest index first so earlier indices stay valid.
      const idx = deleted.map((n) => Number(n.id)).sort((a, b) => b - a);
      for (const i of idx) deleteState(i);
      setSelectedNode(null);
    },
    [deleteState],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      if (c.source == null || c.target == null) return;
      addTransition(Number(c.source), Number(c.target));
    },
    [addTransition],
  );

  const onAddState = useCallback(() => {
    const rect = wrapRef.current?.getBoundingClientRect();
    const center = screenToFlowPosition({
      x: (rect?.left ?? 0) + (rect?.width ?? 400) / 2,
      y: (rect?.top ?? 0) + (rect?.height ?? 300) / 2,
    });
    addState(center.x, center.y);
  }, [addState, screenToFlowPosition]);

  const onSave = useCallback(async () => {
    const name = window.prompt("Save state machine as (asset name):", doc?.name ?? "StateMachine");
    if (name == null) return;
    const file = await save(name);
    if (file) console.info(`saved state machine ${file}`);
  }, [doc, save]);

  const st = selectedNode != null ? doc?.machine.states[selectedNode] : undefined;
  const tr = selectedTransition != null ? doc?.machine.transitions[selectedTransition] : undefined;

  return (
    <div className="bp-canvas" ref={wrapRef}>
      <div className="bp-toolbar">
        <span className="pcg-toolbar__title">State Machine</span>
        <span className="bp-toolbar__sep" />
        <button className="bp-btn" onClick={onAddState}>
          + Add State
        </button>
        <span className="bp-toolbar__spacer" />
        <button className="bp-btn" onClick={() => void onSave()} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </button>
      </div>

      <div className="pcg-body">
        <div className="bp-flow sm-flow">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onNodesDelete={onNodesDelete}
            onConnect={onConnect}
            onNodeClick={(_, n) => {
              setSelectedNode(Number(n.id));
              selectTransition(null);
            }}
            onNodeDoubleClick={(_, n) => {
              const i = Number(n.id);
              const name = window.prompt("Rename state:", doc?.machine.states[i]?.name ?? "");
              if (name) renameState(i, name);
            }}
            onEdgeClick={(_, e) => {
              selectTransition(Number(e.id));
              setSelectedNode(null);
            }}
            onEdgesDelete={(del) => del.forEach((e) => deleteTransition(Number(e.id)))}
            onPaneClick={() => {
              setSelectedNode(null);
              selectTransition(null);
            }}
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
        </div>

        <div className="sm-inspector">
          {st && selectedNode != null ? (
            <div className="sm-insp__body">
              <div className="sm-insp__title">State: {st.name}</div>
              <label className="sm-insp__row">
                <span>Entry</span>
                <input
                  type="checkbox"
                  checked={doc?.machine.entry === selectedNode}
                  onChange={() => setEntry(selectedNode)}
                />
              </label>
              <label className="sm-insp__row">
                <span>Clip</span>
                {st.motion.kind === "clip" ? (
                  <select
                    value={st.motion.clip ?? ""}
                    onChange={(e) => setStateClip(selectedNode, e.target.value || null)}
                  >
                    <option value="">(none)</option>
                    {clips.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="sm-insp__note">{st.motion.kind} (edit in data — v1)</span>
                )}
              </label>
              <label className="sm-insp__row">
                <span>Speed</span>
                <input
                  type="number"
                  step={0.1}
                  value={st.speed}
                  onChange={(e) => setStateSpeed(selectedNode, Number(e.target.value))}
                />
              </label>
              <label className="sm-insp__row">
                <span>Looping</span>
                <input
                  type="checkbox"
                  checked={st.looping}
                  onChange={(e) => setStateLooping(selectedNode, e.target.checked)}
                />
              </label>
            </div>
          ) : tr && selectedTransition != null ? (
            <div className="sm-insp__body">
              <div className="sm-insp__title">
                Transition: {doc?.machine.states[tr.from]?.name} → {doc?.machine.states[tr.to]?.name}
              </div>
              <label className="sm-insp__row">
                <span>Duration (s)</span>
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  value={tr.duration}
                  onChange={(e) => setTransitionDuration(selectedTransition, Number(e.target.value))}
                />
              </label>
              <label className="sm-insp__row">
                <span>Exit time</span>
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  placeholder="none"
                  value={tr.exitTime ?? ""}
                  onChange={(e) =>
                    setTransitionExitTime(
                      selectedTransition,
                      e.target.value === "" ? null : Number(e.target.value),
                    )
                  }
                />
              </label>
              <div className="sm-insp__subtitle">
                Conditions (AND)
                <button className="bp-btn bp-btn--sm" onClick={() => addCondition(selectedTransition)}>
                  +
                </button>
              </div>
              {tr.conditions.map((c, ci) => (
                <div key={ci} className="sm-cond">
                  <input
                    className="sm-cond__var"
                    value={c.var}
                    onChange={(e) =>
                      updateCondition(selectedTransition, ci, { var: e.target.value })
                    }
                  />
                  <select
                    value={c.op}
                    onChange={(e) =>
                      updateCondition(selectedTransition, ci, {
                        op: e.target.value as (typeof SM_OPS)[number],
                      })
                    }
                  >
                    {SM_OPS.map((op) => (
                      <option key={op} value={op}>
                        {op}
                      </option>
                    ))}
                  </select>
                  <input
                    className="sm-cond__val"
                    type="number"
                    step={0.1}
                    value={c.value}
                    onChange={(e) =>
                      updateCondition(selectedTransition, ci, { value: Number(e.target.value) })
                    }
                  />
                  <button
                    className="bp-btn bp-btn--sm"
                    onClick={() => removeCondition(selectedTransition, ci)}
                    title="Remove condition"
                  >
                    ×
                  </button>
                </div>
              ))}
              <button
                className="bp-btn bp-btn--sm sm-insp__delete"
                onClick={() => deleteTransition(selectedTransition)}
              >
                Delete transition
              </button>
            </div>
          ) : (
            <div className="sm-insp__empty">
              Select a state or transition. Drag from a state's right edge to another to add a
              transition.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function StateMachineCanvas() {
  const init = useSmStore((s) => s.init);
  const close = useSmStore((s) => s.close);
  const ready = useSmStore((s) => s.ready);
  // Init on mount; free the backend document on unmount (panel close) so open
  // state machines don't accumulate for the session.
  useEffect(() => {
    void init();
    return () => {
      void close();
    };
  }, [init, close]);
  if (!ready) return <div className="bp-canvas bp-canvas--loading">Loading State Machine…</div>;
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
