/**
 * The animation State Machine editor canvas (@xyflow/react, P11.2; the v2
 * authoring surface, P29.5).
 *
 * A state machine is a **plain typed model**, not a dataflow graph: nodes are
 * states, edges are transitions (free edges — any node to any other). The canvas
 * renders `smStore`'s document directly and turns gestures into local model
 * edits; the right-hand inspector edits whatever is selected, and two tabs
 * beside it edit the machine's declared **parameters** and its per-joint blend
 * **profiles**.
 *
 * ## What is authorable here (P29.5)
 *
 * Everything `.inf_sm` v2 decodes, which is the whole point: typed parameters
 * including `Trigger`, condition **trees** (the `RuleBuilder`), transition
 * priority and interruption, blend curves, blend profiles, `exit_time`,
 * **any-state** transitions, one level of nested **sub-machines** (drilled into
 * through the breadcrumb), and state enter/exit events. P29.1 shipped a model
 * this canvas could carry and not edit; nothing is read-only now except a 1D
 * blend space, which the Blend Space panel owns.
 *
 * ## The validator is the door, and its refusal is shown
 *
 * `sm_save` calls `StateMachine::validate` and refuses a machine its own reader
 * would reject (P29.2 A1). The refusal comes back as a string and is rendered
 * **inline** above the canvas — not a toast, because a toast that has faded is a
 * save the author believes happened.
 *
 * ## Fully controlled
 *
 * `nodes`/`edges` are `useMemo` over the store document; there is no `useState`
 * mirror and no effect that syncs one (`react-hooks/set-state-in-effect`). Only
 * ephemeral UI — which node is selected, which tab is open — is local state.
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
import {
  SM_BLEND_CURVES,
  SM_INTERRUPT_BLENDS,
  SM_INTERRUPT_SOURCES,
  SM_OPS,
  SM_PARAM_KINDS,
  conditionSummary,
  motionSummary,
  type SmCondDto,
  type SmMachineDto,
  type SmParamKind,
} from "../../lib/smTypes";
import { RuleBuilder } from "./RuleBuilder";
import { SmNode } from "./SmNode";
import { LiveTuning } from "./LiveTuning";
import "../blueprint/blueprint.css";
import "./sm.css";

const nodeTypes = { smState: SmNode };

/** Which inspector tab is open — the one piece of ephemeral state this panel
 *  keeps, beside the selected node. */
type Tab = "selection" | "params" | "profiles";

export function deriveNodes(m: SmMachineDto, clipName: (id: string | null) => string): Node[] {
  return m.states.map((s, i) => ({
    id: String(i),
    type: "smState",
    position: { x: s.x, y: s.y },
    data: {
      name: s.name,
      summary: motionSummary(s.motion, clipName),
      isEntry: m.entry === i,
    },
  }));
}

export function deriveEdges(m: SmMachineDto, selected: number | null): Edge[] {
  return m.transitions.map((t, i) => {
    const conds = t.conditions?.length ?? null;
    const rule =
      conds === null ? conditionSummary(t.condition) : conds > 0 ? `${conds} cond` : "";
    const bits = [rule];
    if (t.exitTime != null) bits.push(`exit ${t.exitTime}`);
    if (t.priority !== 0) bits.push(`p${t.priority}`);
    const label = bits.filter(Boolean).join(" · ");
    return {
      id: String(i),
      // An any-state transition has no source node, so it is drawn as a self
      // edge on its own target and labelled "any" — the honest picture, because
      // there is no node to leave from.
      source: String(t.from ?? t.to),
      target: String(t.to),
      label: (t.from === null ? `any${label ? " · " : ""}` : "") + label || undefined,
      animated: i === selected,
      style: {
        stroke: i === selected ? "var(--ink-accent)" : "var(--ink-text-dim)",
        strokeWidth: i === selected ? 2.5 : 1.5,
        strokeDasharray: t.from === null ? "6 3" : undefined,
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
  const proposing = useSmStore((s) => s.proposing);
  const proposalNotes = useSmStore((s) => s.proposalNotes);
  const refusal = useSmStore((s) => s.refusal);
  const path = useSmStore((s) => s.path);
  const activeMachine = useSmStore((s) => s.activeMachine);
  const setPath = useSmStore((s) => s.setPath);

  const addState = useSmStore((s) => s.addState);
  const moveState = useSmStore((s) => s.moveState);
  const deleteState = useSmStore((s) => s.deleteState);
  const renameState = useSmStore((s) => s.renameState);
  const setEntry = useSmStore((s) => s.setEntry);
  const setStateClip = useSmStore((s) => s.setStateClip);
  const setStateSpeed = useSmStore((s) => s.setStateSpeed);
  const setStateLooping = useSmStore((s) => s.setStateLooping);
  const setStateEvents = useSmStore((s) => s.setStateEvents);
  const makeSubMachine = useSmStore((s) => s.makeSubMachine);

  const addTransition = useSmStore((s) => s.addTransition);
  const addAnyTransition = useSmStore((s) => s.addAnyTransition);
  const deleteTransition = useSmStore((s) => s.deleteTransition);
  const selectTransition = useSmStore((s) => s.selectTransition);
  const setTransition = useSmStore((s) => s.setTransition);
  const setTransitionDuration = useSmStore((s) => s.setTransitionDuration);
  const setTransitionExitTime = useSmStore((s) => s.setTransitionExitTime);
  const setCondition = useSmStore((s) => s.setCondition);
  const addCondition = useSmStore((s) => s.addCondition);
  const updateCondition = useSmStore((s) => s.updateCondition);
  const removeCondition = useSmStore((s) => s.removeCondition);

  const addParam = useSmStore((s) => s.addParam);
  const updateParam = useSmStore((s) => s.updateParam);
  const removeParam = useSmStore((s) => s.removeParam);
  const addProfile = useSmStore((s) => s.addProfile);
  const updateProfile = useSmStore((s) => s.updateProfile);
  const removeProfile = useSmStore((s) => s.removeProfile);
  const addProfileWeight = useSmStore((s) => s.addProfileWeight);
  const setProfileWeight = useSmStore((s) => s.setProfileWeight);
  const removeProfileWeight = useSmStore((s) => s.removeProfileWeight);

  const propose = useSmStore((s) => s.propose);
  const dismissNotes = useSmStore((s) => s.dismissNotes);
  const save = useSmStore((s) => s.save);

  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>("selection");
  const { screenToFlowPosition } = useReactFlow();
  const wrapRef = useRef<HTMLDivElement>(null);

  const machine = doc ? activeMachine() : null;
  const root = doc?.machine ?? null;

  const nodes = useMemo<Node[]>(
    () => (machine ? deriveNodes(machine, clipName) : []),
    [machine, clipName],
  );
  const edges = useMemo<Edge[]>(
    () => (machine ? deriveEdges(machine, selectedTransition) : []),
    [machine, selectedTransition],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      for (const c of changes) {
        if (c.type === "position" && c.position && c.dragging === false) {
          moveState(Number(c.id), c.position.x, c.position.y);
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

  const onPropose = useCallback(async () => {
    const ids = clips.map((c) => c.id);
    if (ids.length === 0) {
      window.alert("Import some .inf_anim clips first — a proposal reads what they measured.");
      return;
    }
    const ok = window.confirm(
      `Propose a machine from all ${ids.length} imported clip(s)? This replaces the open document.`,
    );
    if (!ok) return;
    await propose(ids);
    setSelectedNode(null);
  }, [clips, propose]);

  const st = selectedNode != null ? machine?.states[selectedNode] : undefined;
  const tr = selectedTransition != null ? machine?.transitions[selectedTransition] : undefined;
  const inSub = path.length > 0;

  return (
    <div className="bp-canvas" ref={wrapRef}>
      <div className="bp-toolbar">
        <span className="pcg-toolbar__title">State Machine</span>
        {inSub && root && (
          <>
            <span className="bp-toolbar__sep" />
            <button className="bp-btn bp-btn--sm" onClick={() => setPath([])} title="Back to the root machine">
              ← {root.states[path[0]]?.name ?? "root"}
            </button>
            <span className="sm-insp__note">sub-machine</span>
          </>
        )}
        <span className="bp-toolbar__sep" />
        <button className="bp-btn" onClick={onAddState}>
          + Add State
        </button>
        <button
          className="bp-btn"
          onClick={() => selectedNode != null && addAnyTransition(selectedNode)}
          disabled={selectedNode == null}
          title="A transition into the selected state from ANY state (v2)"
        >
          + Any-state →
        </button>
        <span className="bp-toolbar__spacer" />
        <button
          className="bp-btn"
          onClick={() => void onPropose()}
          disabled={proposing || inSub}
          title="Propose a machine from the imported clips' derived speed and gait (P29.5)"
        >
          {proposing ? "Proposing…" : "Propose from clips"}
        </button>
        <button className="bp-btn" onClick={() => void onSave()} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </button>
      </div>

      {refusal && (
        <div className="sm-refusal" role="alert">
          <strong>Not saved.</strong> {refusal}
        </div>
      )}

      {proposalNotes.length > 0 && (
        <div className="sm-notes">
          <div className="sm-notes__head">
            <strong>Why this machine</strong>
            <button className="bp-btn bp-btn--sm" onClick={dismissNotes}>
              dismiss
            </button>
          </div>
          <ul>
            {proposalNotes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      )}

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
              setTab("selection");
            }}
            onNodeDoubleClick={(_, n) => {
              const i = Number(n.id);
              const state = machine?.states[i];
              // A double-click on a sub-machine drills IN; on anything else it
              // renames. Two gestures on one event, but they are the two things
              // a double-click means on a node that contains something.
              if (state?.motion.kind === "subMachine" && !inSub) {
                setPath([i]);
                setSelectedNode(null);
                return;
              }
              const name = window.prompt("Rename state:", state?.name ?? "");
              if (name) renameState(i, name);
            }}
            onEdgeClick={(_, e) => {
              selectTransition(Number(e.id));
              setSelectedNode(null);
              setTab("selection");
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
          <div className="sm-tabs">
            {(["selection", "params", "profiles"] as Tab[]).map((t) => (
              <button
                key={t}
                className={`sm-tab${tab === t ? " sm-tab--on" : ""}`}
                onClick={() => setTab(t)}
              >
                {t === "selection" ? "Selection" : t === "params" ? "Parameters" : "Profiles"}
              </button>
            ))}
          </div>

          {tab === "params" && root && (
            <div className="sm-insp__body">
              <div className="sm-insp__title">Parameters</div>
              <div className="sm-insp__note">
                Declared on the root machine; a sub-machine shares this table (the engine refuses
                one that declares its own). An <em>undeclared</em> name still reads as a float
                defaulting to 0 — declaring it is what buys typing.
              </div>
              {root.params.map((p, i) => (
                <div key={i} className="sm-param">
                  <input
                    className="sm-param__name"
                    value={p.name}
                    onChange={(e) => updateParam(i, { name: e.target.value })}
                  />
                  <select
                    value={p.kind}
                    onChange={(e) => updateParam(i, { kind: e.target.value as SmParamKind })}
                  >
                    {SM_PARAM_KINDS.map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </select>
                  {p.kind === "bool" || p.kind === "trigger" ? (
                    <input
                      type="checkbox"
                      checked={p.default > 0.5}
                      disabled={p.kind === "trigger"}
                      title={
                        p.kind === "trigger"
                          ? "A trigger is an EVENT — it is armed, never defaulted"
                          : "Default"
                      }
                      onChange={(e) => updateParam(i, { default: e.target.checked ? 1 : 0 })}
                    />
                  ) : (
                    <input
                      className="sm-param__val"
                      type="number"
                      step={p.kind === "int" ? 1 : 0.1}
                      value={p.default}
                      onChange={(e) => updateParam(i, { default: Number(e.target.value) })}
                    />
                  )}
                  <button className="bp-btn bp-btn--sm" onClick={() => removeParam(i)}>
                    ×
                  </button>
                </div>
              ))}
              <div className="sm-insp__row">
                {SM_PARAM_KINDS.map((k) => (
                  <button key={k} className="bp-btn bp-btn--sm" onClick={() => addParam(k)}>
                    + {k}
                  </button>
                ))}
              </div>
              <div className="sm-insp__subtitle">Live tuning (S4)</div>
              <LiveTuning params={root.params} />
            </div>
          )}

          {tab === "profiles" && root && (
            <div className="sm-insp__body">
              <div className="sm-insp__title">Blend profiles</div>
              <div className="sm-insp__note">
                Per-joint blend masks a transition may point at. A joint that is not listed blends
                at full weight.
              </div>
              {root.profiles.map((p, i) => (
                <div key={i} className="sm-profile">
                  <div className="sm-insp__row">
                    <input
                      value={p.name}
                      onChange={(e) => updateProfile(i, { name: e.target.value })}
                    />
                    <button className="bp-btn bp-btn--sm" onClick={() => addProfileWeight(i)}>
                      + joint
                    </button>
                    <button className="bp-btn bp-btn--sm" onClick={() => removeProfile(i)}>
                      ×
                    </button>
                  </div>
                  {p.weights.map((w, wi) => (
                    <div key={wi} className="sm-cond">
                      <input
                        className="sm-cond__val"
                        type="number"
                        min={0}
                        step={1}
                        value={w.joint}
                        onChange={(e) => setProfileWeight(i, wi, Number(e.target.value), w.weight)}
                        title="Joint index"
                      />
                      <input
                        className="sm-cond__val"
                        type="number"
                        min={0}
                        max={1}
                        step={0.05}
                        value={w.weight}
                        onChange={(e) => setProfileWeight(i, wi, w.joint, Number(e.target.value))}
                        title="Weight [0,1]"
                      />
                      <button
                        className="bp-btn bp-btn--sm"
                        onClick={() => removeProfileWeight(i, wi)}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              ))}
              <button className="bp-btn bp-btn--sm" onClick={addProfile}>
                + profile
              </button>
            </div>
          )}

          {tab === "selection" && st && selectedNode != null && (
            <div className="sm-insp__body">
              <div className="sm-insp__title">State: {st.name}</div>
              <label className="sm-insp__row">
                <span>Entry</span>
                <input
                  type="checkbox"
                  checked={machine?.entry === selectedNode}
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
                ) : st.motion.kind === "subMachine" ? (
                  <button className="bp-btn bp-btn--sm" onClick={() => setPath([selectedNode])}>
                    Edit sub-machine ({st.motion.machine.states.length} states)
                  </button>
                ) : (
                  <span className="sm-insp__note">
                    {st.motion.kind} — authored in the Blend Space panel
                  </span>
                )}
              </label>
              {st.motion.kind === "clip" && !inSub && (
                <button className="bp-btn bp-btn--sm" onClick={() => makeSubMachine(selectedNode)}>
                  Convert to sub-machine
                </button>
              )}
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
              <div className="sm-insp__subtitle">Events</div>
              <label className="sm-insp__row">
                <span>On enter</span>
                <input
                  value={st.onEnter.join(", ")}
                  placeholder="notify names, comma separated"
                  onChange={(e) =>
                    setStateEvents(
                      selectedNode,
                      e.target.value.split(",").map((s) => s.trim()),
                      st.onExit,
                    )
                  }
                />
              </label>
              <label className="sm-insp__row">
                <span>On exit</span>
                <input
                  value={st.onExit.join(", ")}
                  placeholder="notify names, comma separated"
                  onChange={(e) =>
                    setStateEvents(
                      selectedNode,
                      st.onEnter,
                      e.target.value.split(",").map((s) => s.trim()),
                    )
                  }
                />
              </label>
            </div>
          )}

          {tab === "selection" && tr && selectedTransition != null && (
            <div className="sm-insp__body">
              <div className="sm-insp__title">
                Transition: {tr.from === null ? "Any state" : machine?.states[tr.from]?.name} →{" "}
                {machine?.states[tr.to]?.name}
              </div>
              <label className="sm-insp__row">
                <span>From any state</span>
                <input
                  type="checkbox"
                  checked={tr.from === null}
                  title="Fire from any state (v2). Turning it off gives the edge state 0 as its source — an any-state transition never had one to restore."
                  onChange={(e) =>
                    setTransition(selectedTransition, { from: e.target.checked ? null : 0 })
                  }
                />
              </label>
              {tr.from === null && (
                <label className="sm-insp__row">
                  <span>Exclude self</span>
                  <input
                    type="checkbox"
                    checked={tr.excludeSelf}
                    title="Do not re-enter the state the machine is already in"
                    onChange={(e) =>
                      setTransition(selectedTransition, { excludeSelf: e.target.checked })
                    }
                  />
                </label>
              )}
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
              <label className="sm-insp__row">
                <span>Priority</span>
                <input
                  type="number"
                  step={1}
                  value={tr.priority}
                  title="Higher fires first; ties break by declaration order"
                  onChange={(e) =>
                    setTransition(selectedTransition, { priority: Math.round(Number(e.target.value)) })
                  }
                />
              </label>
              <label className="sm-insp__row">
                <span>Blend curve</span>
                <select
                  value={tr.curve}
                  onChange={(e) =>
                    setTransition(selectedTransition, {
                      curve: e.target.value as (typeof SM_BLEND_CURVES)[number],
                    })
                  }
                >
                  {SM_BLEND_CURVES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
              <label className="sm-insp__row">
                <span>Blend profile</span>
                <select
                  value={tr.profile ?? ""}
                  onChange={(e) =>
                    setTransition(selectedTransition, {
                      profile: e.target.value === "" ? null : Number(e.target.value),
                    })
                  }
                >
                  <option value="">(all joints)</option>
                  {(root?.profiles ?? []).map((p, i) => (
                    <option key={i} value={i}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="sm-insp__row">
                <span>May interrupt</span>
                <select
                  value={tr.interruptSource}
                  title="Which in-progress fade this transition may cut into"
                  onChange={(e) =>
                    setTransition(selectedTransition, {
                      interruptSource: e.target.value as (typeof SM_INTERRUPT_SOURCES)[number],
                    })
                  }
                >
                  {SM_INTERRUPT_SOURCES.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="sm-insp__row">
                <span>Outgoing pose</span>
                <select
                  value={tr.interruptBlend}
                  title="What the outgoing pose is when this transition interrupts a fade"
                  onChange={(e) =>
                    setTransition(selectedTransition, {
                      interruptBlend: e.target.value as (typeof SM_INTERRUPT_BLENDS)[number],
                    })
                  }
                >
                  {SM_INTERRUPT_BLENDS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>

              <div className="sm-insp__subtitle">
                Rule
                {tr.conditions !== null && (
                  <>
                    <button
                      className="bp-btn bp-btn--sm"
                      onClick={() => addCondition(selectedTransition)}
                    >
                      +
                    </button>
                    <button
                      className="bp-btn bp-btn--sm"
                      title="Edit this rule as a tree (an Or, a Not, a trigger, a typed compare)"
                      onClick={() => setCondition(selectedTransition, treeOf(tr.condition))}
                    >
                      Edit as tree
                    </button>
                  </>
                )}
              </div>
              {tr.conditions === null ? (
                <RuleBuilder
                  cond={tr.condition}
                  params={root?.params ?? []}
                  onChange={(next: SmCondDto) => setCondition(selectedTransition, next)}
                />
              ) : (
                (tr.conditions ?? []).map((c, ci) => (
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
                ))
              )}
              <button
                className="bp-btn bp-btn--sm sm-insp__delete"
                onClick={() => deleteTransition(selectedTransition)}
              >
                Delete transition
              </button>
            </div>
          )}

          {tab === "selection" && !st && !tr && (
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

/** The tree a flat-view transition already has.
 *
 * `condition` is **always** the whole tree on the wire — the flat list is a
 * projection of it, not an alternative to it — so switching to the builder is
 * nothing more than deciding to edit the field that was already there. Kept as a
 * named function so the "Edit as tree" button reads as the one-way door it is
 * rather than as a conversion that might lose something. */
export function treeOf(cond: SmCondDto): SmCondDto {
  return cond;
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
