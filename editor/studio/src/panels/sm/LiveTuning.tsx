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
const MOVEMENT_TUNABLES: { mfield: string; label: string; step: number }[] = [
  { mfield: "walk_speed_mps", label: "Walk (m/s)", step: 0.05 },
  { mfield: "run_speed_mps", label: "Run (m/s)", step: 0.05 },
  { mfield: "sprint_speed_mps", label: "Sprint (m/s)", step: 0.1 },
  { mfield: "jump_speed_mps", label: "Jump (m/s)", step: 0.1 },
  { mfield: "step_height_m", label: "Step height (m)", step: 0.01 },
  { mfield: "slope_limit_deg", label: "Slope limit (deg)", step: 1 },
  { mfield: "air_control", label: "Air control", step: 0.05 },
];

/** The **vehicle** tunables offered by name (P29.7).
 *
 * A vehicle is not a component — its rig is derived from the scene and its
 * tunables live on the running vehicle, like a ragdoll's bodies — so these cross
 * as `kind: "vehicle"` and are matched against
 * `inf_ecs::vehicle::VehicleTuning::names()` rather than against a reflected
 * field. Select the CAR to tune it: the guid is the chassis'.
 *
 * Session-scoped whatever the checkbox says, because there is no document field
 * for a kept value to land on. */
const VEHICLE_TUNABLES: { vfield: string; label: string; step: number }[] = [
  { vfield: "stiffness_n_per_m", label: "Spring (N/m)", step: 1000 },
  { vfield: "damping_ns_per_m", label: "Damper (Ns/m)", step: 100 },
  { vfield: "max_engine_force_n", label: "Engine (N)", step: 500 },
  { vfield: "brake_force_n", label: "Brake (N)", step: 500 },
  { vfield: "max_steer_deg", label: "Steer lock (deg)", step: 1 },
  { vfield: "lateral_grip", label: "Grip (mu)", step: 0.05 },
];

/** The **camera** tunables offered by name (P29.7).
 *
 * The locomotion camera is a host-owned field on the session — never a component
 * and never a resource (Ruling 4) — so these cross as `kind: "camera"` and carry
 * no entity at all. The names are `CameraTuning::set`'s vocabulary; a gait block
 * edits both rotation modes at once, which is that type's rule and not this
 * panel's. */
const CAMERA_TUNABLES: { cfield: string; label: string; step: number }[] = [
  { cfield: "run.arm_length_m", label: "Arm (m)", step: 0.1 },
  { cfield: "run.lag_x", label: "Lag X", step: 0.5 },
  { cfield: "run.fov_deg", label: "FOV (deg)", step: 1 },
  { cfield: "collision_radius_m", label: "Sweep radius (m)", step: 0.05 },
  { cfield: "pivot_height_ratio", label: "Pivot height", step: 0.05 },
];

/// The weapon numbers the panel offers, by `WeaponDef::set`'s own names (I6).
///
/// Keyed by `wfield` rather than `cfield` so the source gate that pins the
/// camera's list cannot accidentally read these as camera names — two lists in
/// one file need two markers, or a gate reading the first would pass the second
/// by walking past it.
const WEAPON_TUNABLES: { wfield: string; label: string; step: number }[] = [
  { wfield: "damage_j", label: "Damage (J)", step: 50 },
  { wfield: "rounds_per_minute", label: "Rate (rpm)", step: 25 },
  { wfield: "spread_deg", label: "Spread (deg)", step: 0.1 },
  { wfield: "magazine", label: "Magazine", step: 1 },
  { wfield: "reload_s", label: "Reload (s)", step: 0.1 },
  { wfield: "range_m", label: "Range (m)", step: 10 },
];

/// Which item id the weapon rows tune. A weapon is a DEFINITION, so tuning it
/// means every one in the level — which is what a designer dragging a fire-rate
/// slider means.
const WEAPON_ITEM_ID = "rifle";

export function LiveTuning({ params }: { params: SmParamDto[] }) {
  const running = useSimStore((s) => s.running);
  const selection = useSceneStore((s) => s.selection);
  const [keep, setKeep] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const guid = selection[0] ?? null;

  const push = useCallback(
    async (
      kind: "field" | "param" | "trigger" | "vehicle" | "camera" | "weapon",
      name: string,
      value: number,
      // I6: a weapon is tuned by ITEM ID rather than by entity, so the identity
      // argument carries a name on that path. The command's own `guid` field is
      // reused rather than a second one added; see `sim_tune`.
      subject?: string,
    ) => {
      // The camera belongs to the session and a weapon to the item catalogue, so
      // those two are the kinds that do not need a selection.
      if (!guid && kind !== "camera" && kind !== "weapon") return;
      try {
        const applied = await simIpc.tune(kind, subject ?? guid ?? "", name, value, keep);
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
        <label className="sm-insp__row" key={t.mfield}>
          <span>{t.label}</span>
          <input
            type="number"
            step={t.step}
            placeholder="set"
            onChange={(e) =>
              e.target.value !== "" && void push("field", t.mfield, Number(e.target.value))
            }
          />
        </label>
      ))}
      <div className="sm-insp__subtitle">Vehicle (select the car)</div>
      {VEHICLE_TUNABLES.map((t) => (
        <label className="sm-insp__row" key={t.vfield}>
          <span>{t.label}</span>
          <input
            type="number"
            step={t.step}
            placeholder="set"
            onChange={(e) =>
              e.target.value !== "" && void push("vehicle", t.vfield, Number(e.target.value))
            }
          />
        </label>
      ))}

      <div className="sm-insp__subtitle">Camera</div>
      {CAMERA_TUNABLES.map((t) => (
        <label className="sm-insp__row" key={t.cfield}>
          <span>{t.label}</span>
          <input
            type="number"
            step={t.step}
            placeholder="set"
            onChange={(e) =>
              e.target.value !== "" && void push("camera", t.cfield, Number(e.target.value))
            }
          />
        </label>
      ))}

      <div className="sm-insp__subtitle">Weapon ({WEAPON_ITEM_ID})</div>
      {WEAPON_TUNABLES.map((t) => (
        <label className="sm-insp__row" key={t.wfield}>
          <span>{t.label}</span>
          <input
            type="number"
            step={t.step}
            placeholder="set"
            onChange={(e) =>
              e.target.value !== "" &&
              void push("weapon", t.wfield, Number(e.target.value), WEAPON_ITEM_ID)
            }
          />
        </label>
      ))}
      {status && <div className="sm-insp__note">{status}</div>}
    </div>
  );
}
