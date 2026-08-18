/**
 * **Live tuning during Simulate** — pillar S4's surface (P29.5).
 *
 * While a Simulate session is running and an actor is selected, this drives
 * `sim_tune`: the machine's declared parameters, its triggers, and the movement
 * tunables an author reaches for most. Each edit lands at the top of the **next**
 * fixed step — see `inf_editor_core::tuning` for why that is a contract and not
 * an implementation detail.
 *
 * # Why it lives here and not in Details
 *
 * The Details panel already writes reflected component fields into the live
 * world, and that has worked since P29.3 (P23.6's edit-during-Simulate ruling).
 * What it cannot do is the half this panel is about: a machine's **parameters**
 * are not components — they are a gameplay overlay on the bridge — and a
 * **trigger** is an event that has to be armed rather than assigned, so there is
 * no field for either to appear beside. An author watching a graph needs to ask
 * it "what do you do at speed 4?" without writing a program to ask.
 *
 * # Two scopes, and the default is the modest one
 *
 * `Session` reverts at Stop — UE's PIE semantics, and right for a question.
 * `Keep past Stop` writes the value onto the document afterwards as one ordinary
 * undoable edit, however many times the slider moved.
 *
 * # It is a live surface over a world that is changing
 *
 * Every call returns whether a session was running, and a `false` is a **value**:
 * the panel says "not running" and moves on. A toast per slider tick after Stop
 * would be the whole behaviour of this component.
 */
import { useCallback, useState } from "react";

import { sim as simIpc } from "../../lib/ipc";
import { useSceneStore } from "../../stores/sceneStore";
import { useSimStore } from "../../stores/simStore";
import type { SmParamDto } from "../../lib/smTypes";

/** The movement tunables offered by name, with the unit each is in.
 *
 * A short list on purpose. Every field of `CharacterMovement` is already
 * reachable through Details; these are the ones an author reaches for *while
 * watching*, and a panel that mirrored the whole component would be a worse
 * Details. */
const MOVEMENT_TUNABLES: { field: string; label: string; step: number }[] = [
  { field: "walk_speed_mps", label: "Walk (m/s)", step: 0.05 },
  { field: "run_speed_mps", label: "Run (m/s)", step: 0.05 },
  { field: "sprint_speed_mps", label: "Sprint (m/s)", step: 0.1 },
  { field: "jump_speed_mps", label: "Jump (m/s)", step: 0.1 },
  { field: "step_height_m", label: "Step height (m)", step: 0.01 },
  { field: "slope_limit_deg", label: "Slope limit (deg)", step: 1 },
  { field: "air_control", label: "Air control", step: 0.05 },
];

export function LiveTuning({ params }: { params: SmParamDto[] }) {
  const running = useSimStore((s) => s.running);
  const selection = useSceneStore((s) => s.selection);
  const [keep, setKeep] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const guid = selection[0] ?? null;

  const push = useCallback(
    async (kind: "field" | "param" | "trigger", name: string, value: number) => {
      if (!guid) return;
      try {
        const applied = await simIpc.tune(kind, guid, name, value, keep);
        setStatus(applied ? `${name} = ${value} (next step)` : "no Simulate session is running");
      } catch (e) {
        setStatus(String(e));
      }
    },
    [guid, keep],
  );

  if (!running) {
    return (
      <div className="sm-insp__note">
        Live tuning is available while Simulate is running. Edits land on the next fixed step.
      </div>
    );
  }
  if (!guid) {
    return <div className="sm-insp__note">Select the actor to tune in the viewport.</div>;
  }

  return (
    <div className="sm-live">
      <label className="sm-insp__row">
        <span>Keep past Stop</span>
        <input
          type="checkbox"
          checked={keep}
          title="Write the value onto the document when the session ends, as one undoable edit"
          onChange={(e) => setKeep(e.target.checked)}
        />
      </label>

      <div className="sm-insp__subtitle">Machine parameters</div>
      {params.length === 0 && (
        <div className="sm-insp__note">This machine declares none. Add one above.</div>
      )}
      {params.map((p) => (
        <div key={p.name} className="sm-param">
          <span className="sm-param__name">{p.name}</span>
          {p.kind === "trigger" ? (
            <button className="bp-btn bp-btn--sm" onClick={() => void push("trigger", p.name, 1)}>
              Arm
            </button>
          ) : p.kind === "bool" ? (
            <>
              <button className="bp-btn bp-btn--sm" onClick={() => void push("param", p.name, 1)}>
                true
              </button>
              <button className="bp-btn bp-btn--sm" onClick={() => void push("param", p.name, 0)}>
                false
              </button>
            </>
          ) : (
            <input
              className="sm-param__val"
              type="number"
              step={p.kind === "int" ? 1 : 0.1}
              defaultValue={p.default}
              onChange={(e) => void push("param", p.name, Number(e.target.value))}
            />
          )}
        </div>
      ))}

      <div className="sm-insp__subtitle">Movement</div>
      {MOVEMENT_TUNABLES.map((t) => (
        <label className="sm-insp__row" key={t.field}>
          <span>{t.label}</span>
          <input
            type="number"
            step={t.step}
            placeholder="set"
            onChange={(e) =>
              e.target.value !== "" && void push("field", t.field, Number(e.target.value))
            }
          />
        </label>
      ))}
      {status && <div className="sm-insp__note">{status}</div>}
    </div>
  );
}
