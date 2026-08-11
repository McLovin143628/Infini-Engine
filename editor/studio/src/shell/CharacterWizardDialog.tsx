/**
 * New Character from Template (P24.5) — "pick a plan, shape it, get a character
 * that walks".
 *
 * Three steps, one dialog:
 *   1. PLAN   pick biped / quadruped / hexapod / N-pedal.
 *   2. SHAPE  drag the proportions with a LIVE rig diagram and a live readout of
 *             the generated locomotion set — the joint count, the leg lengths and
 *             gait phases, the three cycle durations, and the two speeds the
 *             state machine's thresholds are derived from. Every number comes
 *             back from the backend having actually generated the rig and the
 *             clips; nothing here predicts anything.
 *   3. DONE   the six assets, what the fit and the weight solve did, and the
 *             actor that was added to the level.
 *
 * UNITS: metres and seconds on the wire, degrees for angles at this authoring
 * boundary (units doctrine). The dialog displays what it is sent.
 *
 * AIRSPACE: the native viewport draws over the webview, so the dialog holds a
 * `useViewportOverlay` acquisition for its whole lifetime — the guard every
 * other shell overlay uses.
 */
import { useEffect } from "react";
import { PersonStanding, X } from "lucide-react";

import type { BodyPlanName } from "../lib/ipc";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useShellStore } from "../stores/shellStore";
import { projectRig, useCharacterWizardStore } from "../stores/characterWizardStore";

/** The four plans, with the one line each that says what it is for. */
const PLANS: { id: BodyPlanName; label: string; blurb: string }[] = [
  { id: "biped", label: "Biped", blurb: "Two legs, two arms, the canonical humanoid names." },
  { id: "quadruped", label: "Quadruped", blurb: "Four legs on two girdles, lateral-sequence walk." },
  { id: "hexapod", label: "Hexapod", blurb: "Six legs on three girdles, metachronal wave." },
  { id: "npedal", label: "N-pedal", blurb: "Any even number of legs, up to 64." },
];

export default function CharacterWizardDialog() {
  const open = useShellStore((s) => s.characterWizardOpen);
  const setOpen = useShellStore((s) => s.setCharacterWizardOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);

  const step = useCharacterWizardStore((s) => s.step);
  const spec = useCharacterWizardStore((s) => s.spec);
  const rig = useCharacterWizardStore((s) => s.rig);
  const refusal = useCharacterWizardStore((s) => s.refusal);
  const addToScene = useCharacterWizardStore((s) => s.addToScene);
  const result = useCharacterWizardStore((s) => s.result);
  const error = useCharacterWizardStore((s) => s.error);
  const busy = useCharacterWizardStore((s) => s.busy);

  useViewportOverlay(open);

  const close = () => {
    useCharacterWizardStore.getState().reset();
    setOpen(false);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, busy]);

  if (!open) return null;

  const create = async () => {
    const made = await useCharacterWizardStore.getState().createCharacter();
    if (made) {
      pushStatus(
        made.actor
          ? `${spec.name} created and added to the level`
          : `${spec.name} created under Content`,
      );
    }
  };

  const bump = (key: keyof typeof spec.params, value: number) => {
    useCharacterWizardStore.getState().setParam(key, value);
    void useCharacterWizardStore.getState().refresh();
  };
  const bumpGait = (key: keyof typeof spec.gait, value: number) => {
    useCharacterWizardStore.getState().setGait(key, value);
    void useCharacterWizardStore.getState().refresh();
  };

  const multiGirdle = spec.plan !== "biped";

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-12"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget && !busy) close();
      }}
    >
      <div
        className="flex max-h-[86vh] w-[720px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center gap-2 border-b border-(--ink-border) px-3 py-2">
          <PersonStanding size={15} className="text-(--ink-accent)" />
          <span className="flex-1 font-semibold">New Character from Template</span>
          <button
            aria-label="Close dialog"
            disabled={busy}
            className="rounded p-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text) disabled:opacity-40"
            onClick={close}
          >
            <X size={14} />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-3 text-xs">
          {/* ── 1. plan ─────────────────────────────────────────────────── */}
          {step === "plan" && (
            <div className="grid grid-cols-2 gap-2">
              {PLANS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => void useCharacterWizardStore.getState().choosePlan(p.id)}
                  className="rounded border border-(--ink-border) bg-(--ink-bg-0) p-3 text-left hover:border-(--ink-accent)"
                >
                  <div className="font-semibold text-(--ink-text)">{p.label}</div>
                  <div className="mt-1 text-(--ink-text-dim)">{p.blurb}</div>
                </button>
              ))}
            </div>
          )}

          {/* ── 2. shape ────────────────────────────────────────────────── */}
          {step === "shape" && (
            <div className="flex gap-3">
              <div className="w-[240px] shrink-0">
                <Section title="Rig">
                  <RigDiagram />
                  <Row label="Joints" value={rig ? String(rig.joints.length) : "—"} />
                  <Row
                    label="Height"
                    value={rig ? `${rig.heightM.toFixed(2)} m` : "—"}
                  />
                  <Row label="Sockets" value={rig ? String(rig.sockets.length) : "—"} />
                  <Row label="Limits" value={rig ? String(rig.limits) : "—"} />
                  <Row
                    label="Body"
                    value={
                      rig?.bodyVertices != null
                        ? `${rig.bodyVertices} verts / ${rig.bodyTriangles} tris (mannequin)`
                        : "from the chosen mesh"
                    }
                  />
                </Section>

                <Section title="Locomotion">
                  <Row
                    label="Idle / walk / run"
                    value={
                      rig
                        ? rig.durationsS.map((d) => `${d.toFixed(2)} s`).join(" · ")
                        : "—"
                    }
                  />
                  <Row
                    label="Walk speed"
                    value={rig ? `${rig.walkSpeedMS.toFixed(2)} m/s` : "—"}
                  />
                  <Row
                    label="Run speed"
                    value={rig ? `${rig.runSpeedMS.toFixed(2)} m/s` : "—"}
                  />
                  <Row
                    label="Thresholds"
                    value={
                      rig
                        ? `walk > ${rig.walkThresholdMS.toFixed(2)}, run > ${rig.runThresholdMS.toFixed(2)} m/s`
                        : "—"
                    }
                  />
                  <div className="mt-1 text-(--ink-text-faint)">
                    The generated machine reads one <code>speed</code> variable. The clips
                    animate in place — forward movement is your character controller&apos;s.
                  </div>
                </Section>

                <Section title="Legs">
                  {rig?.legs.map((l) => (
                    <Row
                      key={l.name}
                      label={l.name}
                      value={`${l.lengthM.toFixed(2)} m · phase ${l.phase.toFixed(2)}`}
                    />
                  )) ?? <div className="text-(--ink-text-faint)">—</div>}
                </Section>
              </div>

              <div className="min-w-0 flex-1">
                <Section title="Character">
                  <TextRow
                    label="Name"
                    value={spec.name}
                    onChange={(v) => useCharacterWizardStore.getState().setName(v)}
                  />
                  <Row label="Plan" value={spec.plan} />
                  {spec.plan === "npedal" && (
                    <NumberRow
                      label="Legs"
                      value={spec.legs ?? 6}
                      step={2}
                      min={2}
                      integer
                      onChange={(v) => {
                        useCharacterWizardStore.setState((s) => ({
                          spec: { ...s.spec, legs: Math.max(2, Math.round(v)) },
                        }));
                        void useCharacterWizardStore.getState().refresh();
                      }}
                    />
                  )}
                  <TextRow
                    label="Fit to mesh (GUID)"
                    value={spec.meshAsset ?? ""}
                    placeholder="empty → blocky mannequin"
                    onChange={(v) => {
                      useCharacterWizardStore.getState().setMesh(v.trim() || null);
                      void useCharacterWizardStore.getState().refresh();
                    }}
                  />
                </Section>

                <Section title="Proportions">
                  <NumberRow
                    label="Height (m)"
                    value={spec.params.heightM}
                    step={0.05}
                    min={0.05}
                    onChange={(v) => bump("heightM", v)}
                  />
                  <NumberRow
                    label="Hip height (fraction)"
                    value={spec.params.hipHeightRatio}
                    step={0.01}
                    onChange={(v) => bump("hipHeightRatio", v)}
                  />
                  <NumberRow
                    label="Shoulder height (fraction)"
                    value={spec.params.shoulderHeightRatio}
                    step={0.01}
                    onChange={(v) => bump("shoulderHeightRatio", v)}
                  />
                  <NumberRow
                    label="Head height (fraction)"
                    value={spec.params.headHeightRatio}
                    step={0.01}
                    onChange={(v) => bump("headHeightRatio", v)}
                  />
                  <NumberRow
                    label="Shoulder width (m)"
                    value={spec.params.shoulderWidthM}
                    step={0.02}
                    min={0.01}
                    onChange={(v) => bump("shoulderWidthM", v)}
                  />
                  <NumberRow
                    label="Hip width (m)"
                    value={spec.params.hipWidthM}
                    step={0.02}
                    min={0.01}
                    onChange={(v) => bump("hipWidthM", v)}
                  />
                  <NumberRow
                    label="Upper-limb fraction"
                    value={spec.params.upperLimbRatio}
                    step={0.01}
                    onChange={(v) => bump("upperLimbRatio", v)}
                  />
                  <NumberRow
                    label="Arm length (fraction)"
                    value={spec.params.armLengthRatio}
                    step={0.01}
                    onChange={(v) => bump("armLengthRatio", v)}
                  />
                  <NumberRow
                    label="Spine segments"
                    value={spec.params.spineSegments}
                    step={1}
                    min={2}
                    integer
                    onChange={(v) => bump("spineSegments", v)}
                  />
                  <NumberRow
                    label="Neck segments"
                    value={spec.params.neckSegments}
                    step={1}
                    min={1}
                    integer
                    onChange={(v) => bump("neckSegments", v)}
                  />
                  {multiGirdle && (
                    <>
                      <NumberRow
                        label="Body length (m)"
                        value={spec.params.bodyLengthM}
                        step={0.05}
                        min={0.01}
                        onChange={(v) => bump("bodyLengthM", v)}
                      />
                      <NumberRow
                        label="Head forward (m)"
                        value={spec.params.headForwardM}
                        step={0.05}
                        min={0}
                        onChange={(v) => bump("headForwardM", v)}
                      />
                    </>
                  )}
                </Section>

                <Section title="Gait">
                  <NumberRow
                    label="Walk cadence (Hz)"
                    value={spec.gait.walkCadenceHz}
                    step={0.1}
                    min={0.01}
                    onChange={(v) => bumpGait("walkCadenceHz", v)}
                  />
                  <NumberRow
                    label="Run cadence (Hz)"
                    value={spec.gait.runCadenceHz}
                    step={0.1}
                    min={0.01}
                    onChange={(v) => bumpGait("runCadenceHz", v)}
                  />
                  <NumberRow
                    label="Hip swing (°)"
                    value={spec.gait.hipSwingDeg}
                    step={1}
                    min={0.1}
                    onChange={(v) => bumpGait("hipSwingDeg", v)}
                  />
                  <NumberRow
                    label="Knee flex (°)"
                    value={spec.gait.kneeFlexDeg}
                    step={1}
                    min={0}
                    onChange={(v) => bumpGait("kneeFlexDeg", v)}
                  />
                  <NumberRow
                    label="Arm swing (°)"
                    value={spec.gait.armSwingDeg}
                    step={1}
                    min={0}
                    onChange={(v) => bumpGait("armSwingDeg", v)}
                  />
                </Section>

                <CheckRow
                  label="Add the character to the open level"
                  checked={addToScene}
                  onChange={(v) => useCharacterWizardStore.getState().setAddToScene(v)}
                />
                {refusal && <div className="mt-2 text-(--ink-error)">{refusal}</div>}
              </div>
            </div>
          )}

          {/* ── 3. done ─────────────────────────────────────────────────── */}
          {step === "done" && result && (
            <div className="py-2">
              <Section title="Written to Content/Characters">
                <Row label="Skeleton" value={result.skeleton} />
                <Row
                  label="Body"
                  value={`${result.mesh}${result.mannequin ? " (mannequin)" : " (skinned copy)"}`}
                />
                <Row label="Idle" value={result.idle} />
                <Row label="Walk" value={result.walk} />
                <Row label="Run" value={result.run} />
                <Row label="State machine" value={result.machine} />
              </Section>
              {(result.fit || result.weights) && (
                <Section title="Auto-rig">
                  {result.fit && <Row label="Fit" value={result.fit} />}
                  {result.weights && <Row label="Weights" value={result.weights} />}
                </Section>
              )}
              {result.warnings.length > 0 && (
                <Section title="Warnings">
                  {result.warnings.map((w) => (
                    <div key={w} className="py-0.5 text-(--ink-warn)">
                      {w}
                    </div>
                  ))}
                </Section>
              )}
              <div className="mt-2 text-(--ink-text-faint)">
                {result.actor
                  ? "The actor is in the level with its rig and machine assigned. Drive its `speed` variable and it will walk."
                  : "Drag the skeleton and the state machine onto an actor to use them."}
              </div>
            </div>
          )}

          {error && <div className="mt-2 text-(--ink-error)">{error}</div>}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-(--ink-border) px-3 py-2">
          <button
            onClick={() => useCharacterWizardStore.getState().reset()}
            disabled={busy || step === "plan"}
            className="rounded px-2 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) disabled:opacity-40"
          >
            Choose another plan
          </button>
          <div className="flex gap-2">
            <button
              onClick={close}
              disabled={busy}
              className="rounded px-3 py-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) disabled:opacity-40"
            >
              Close
            </button>
            {step === "shape" && (
              <button
                onClick={() => void create()}
                disabled={busy || refusal !== null}
                className="rounded bg-(--ink-accent) px-3 py-1 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              >
                {busy ? "Generating…" : "Create Character"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * The rig, front-on. An SVG diagram and not a render, for the Skeleton Editor's
 * reason and with its bound: rest-pose globals are summed translations, so joint
 * rotations are not composed. Template rigs carry identity rotations, so for the
 * rigs this wizard makes the diagram is exact.
 */
function RigDiagram() {
  const rig = useCharacterWizardStore((s) => s.rig);
  const points = rig ? projectRig(rig.joints) : [];
  const S = 200;
  const pad = 8;
  const at = (p: { x: number; y: number }) => ({
    x: pad + p.x * (S - 2 * pad),
    y: pad + p.y * (S - 2 * pad),
  });
  return (
    <svg
      role="img"
      aria-label="Rig preview"
      viewBox={`0 0 ${S} ${S}`}
      className="mb-2 w-full rounded border border-(--ink-border) bg-(--ink-bg-0)"
    >
      {points.map((p, i) =>
        p.parent == null ? null : (
          <line
            key={`b${i}`}
            x1={at(points[p.parent]).x}
            y1={at(points[p.parent]).y}
            x2={at(p).x}
            y2={at(p).y}
            stroke="var(--ink-text-faint)"
            strokeWidth={1.5}
          />
        ),
      )}
      {points.map((p, i) => (
        <circle key={`j${i}`} cx={at(p).x} cy={at(p).y} r={2.2} fill="var(--ink-accent)">
          <title>{p.name}</title>
        </circle>
      ))}
    </svg>
  );
}

function Section(props: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <div className="mb-1 font-semibold text-(--ink-text-faint)">{props.title}</div>
      <div className="rounded border border-(--ink-border) bg-(--ink-bg-0) p-2">
        {props.children}
      </div>
    </div>
  );
}

function Row(props: { label: string; value: string; title?: string }) {
  return (
    <div className="flex items-center justify-between gap-2 py-0.5 text-(--ink-text-dim)">
      <span className="shrink-0">{props.label}</span>
      <span className="truncate text-(--ink-text)" title={props.title ?? props.value}>
        {props.value}
      </span>
    </div>
  );
}

function TextRow(props: {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 py-0.5 text-(--ink-text-dim)">
      <span className="shrink-0">{props.label}</span>
      <input
        type="text"
        value={props.value}
        placeholder={props.placeholder}
        onChange={(e) => props.onChange(e.target.value)}
        className="w-56 shrink-0 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1 outline-none focus:border-(--ink-accent)"
      />
    </label>
  );
}

function NumberRow(props: {
  label: string;
  value: number;
  step: number;
  min?: number;
  integer?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 py-0.5 text-(--ink-text-dim)">
      <span className="truncate">{props.label}</span>
      <input
        type="number"
        step={props.step}
        min={props.min}
        value={props.value}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (!Number.isFinite(v)) return;
          props.onChange(props.integer ? Math.round(v) : v);
        }}
        className="w-28 shrink-0 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1 text-right tabular-nums outline-none focus:border-(--ink-accent)"
      />
    </label>
  );
}

function CheckRow(props: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 py-0.5 text-(--ink-text-dim)">
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(e) => props.onChange(e.target.checked)}
      />
      <span className="truncate">{props.label}</span>
    </label>
  );
}
