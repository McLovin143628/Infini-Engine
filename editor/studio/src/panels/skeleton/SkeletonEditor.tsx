/**
 * **The Skeleton Editor** (P24.3) — one panel instance per `.inf_skel` asset.
 *
 * The UI half of the rig pipeline: a joint tree, a transform/limit editor, the
 * socket list, template instantiation, auto-fit, and a live wireframe of the rig
 * being edited.
 *
 * # The preview is drawn HERE, from the joint list
 *
 * The Model Editor's preview is an offscreen wgpu render round-tripped as a
 * base64 PNG, because a mesh is a hundred thousand triangles and the GPU is the
 * only thing that can draw it. A skeleton is *tens of joints*, and its picture is
 * a line per bone — so `projectRig` composes the rest-pose globals from the
 * `parent` + `translation` chain the DTO already carries and draws SVG. No render
 * command, no preview queue, no adapter to be missing, and picking a bone is a
 * click on an element rather than a colour-id readback.
 *
 * This is the same judgement `skelStore` makes about the flight gate: copy the
 * Model Editor's patterns where the problem is the same, and say so where it is
 * not.
 *
 * # Rotations are shown and not edited
 *
 * `translation` and `scale` are `Vec3Field`s. `rotation` is a **quaternion**, and
 * four number boxes that must stay normalized is a control that produces invalid
 * rigs by design — a rotate gizmo is the right instrument and it is the Model
 * Editor's, not this panel's. The value is displayed (an author needs to see that
 * a joint carries one) and written back unchanged by every edit, so nothing is
 * lost. Ledgered in ROADMAP §12's P24 block.
 */
import { useEffect, useMemo, useState } from "react";
import {
  Bone,
  Crosshair,
  Plus,
  Redo2,
  Save,
  Scan,
  Trash2,
  Undo2,
} from "lucide-react";

import { Vec3Field } from "../../components/propertyRows";
import { cn } from "../../lib/utils";
import { useAssetStore } from "../../stores/assetStore";
import { useSkelEntry, useSkelStore } from "../../stores/skelStore";
import type { BodyPlanName } from "../../lib/ipc";
import type { SkelJointDto } from "../../bindings/SkelJointDto";

/** Size of the square wireframe viewport, in SVG user units. */
const VIEW = 320;

/** A joint's rest-pose position in model space, and where it lands on screen. */
interface Placed {
  joint: SkelJointDto;
  /** Model-space rest position (metres). */
  world: [number, number, number];
  /** SVG coordinates. */
  x: number;
  y: number;
}

/**
 * **The rig, composed and projected** — the panel's whole renderer.
 *
 * Rest-pose globals are the running sum of `translation` down the `parent` chain.
 * Rotation and scale are deliberately *not* composed: a rest pose whose joints
 * carry rotations would need full matrix composition to draw exactly, and the
 * error that introduces is a bone drawn slightly wrong in a diagram — whereas
 * getting it wrong in the *engine* is what `recompute_inverse_binds` exists to
 * prevent, and that runs in Rust. The picture is a diagram of the hierarchy, and
 * it says so.
 *
 * The projection is orthographic front-on (X right, Y **up** — SVG's Y grows
 * downward, hence the negation), auto-framed to the rig's own bounds so a 0.3 m
 * creature and a 2 m biped both fill the box.
 */
function projectRig(joints: SkelJointDto[]): Placed[] {
  const world: [number, number, number][] = [];
  for (const j of joints) {
    const p =
      j.parent === null
        ? ([0, 0, 0] as [number, number, number])
        : world[j.parent];
    // A forward reference cannot happen — the joint list is topological — but a
    // malformed one must not crash the panel.
    const base = p ?? [0, 0, 0];
    world.push([
      base[0] + j.translation[0],
      base[1] + j.translation[1],
      base[2] + j.translation[2],
    ]);
  }
  if (world.length === 0) return [];
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of world) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-6);
  const scale = (VIEW * 0.82) / span;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return joints.map((joint, i) => ({
    joint,
    world: world[i],
    x: VIEW / 2 + (world[i][0] - cx) * scale,
    y: VIEW / 2 - (world[i][1] - cy) * scale,
  }));
}

/** A small labelled number input, the Model Editor's local `Num`. */
function Num({
  label,
  value,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-2 text-[11px]">
      <span className="text-(--ink-text-dim)">{label}</span>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value) || 0)}
        className="w-16 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 text-right outline-none focus:border-(--ink-accent)"
      />
    </label>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[10px] font-semibold tracking-wide text-(--ink-text-dim)">
        {title}
      </div>
      {children}
    </div>
  );
}

export default function SkeletonEditor({
  params,
}: {
  panelId: string;
  params: string | null;
}) {
  // Every read and every write is keyed by this panel's own asset id — the dock
  // keeps inactive tabs MOUNTED, so two Skeleton Editors render at once.
  const assetId = params;
  const { doc, selected, refusal, warning, busy } = useSkelEntry(assetId);
  const open = useSkelStore((s) => s.open);
  const close = useSkelStore((s) => s.close);
  const select = useSkelStore((s) => s.select);
  const store = useSkelStore;
  const assetsById = useAssetStore((s) => s.assets);

  const [socketName, setSocketName] = useState("");
  const [plan, setPlan] = useState<BodyPlanName>("biped");
  const [heightM, setHeightM] = useState(1.8);
  const [fitMesh, setFitMesh] = useState("");

  useEffect(() => {
    if (assetId) void open(assetId);
  }, [assetId, open]);
  // Closing frees THIS document's backend session. Reading the id from the
  // closure rather than from the store is what stops a panel unmount closing its
  // neighbour.
  useEffect(() => {
    if (!assetId) return;
    return () => void close(assetId);
  }, [assetId, close]);

  const joints = useMemo(() => doc?.joints ?? [], [doc]);
  const placed = useMemo(() => projectRig(joints), [joints]);
  const sel = selected !== null ? joints[selected] : undefined;

  const meshAssets = useMemo(
    () => Object.values(assetsById).filter((a) => a.kind === "mesh"),
    [assetsById],
  );

  if (!doc || !assetId) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-xs text-(--ink-text-dim)">
        {refusal ??
          (assetId
            ? "Opening…"
            : "Open a skeleton asset from the Content Drawer to edit it.")}
      </div>
    );
  }

  const limited = sel?.limitMinDeg !== null && sel?.limitMinDeg !== undefined;

  return (
    <div className="flex h-full min-h-0 text-xs">
      {/* ── joint tree ──────────────────────────────────────────────────── */}
      <div className="flex w-56 shrink-0 flex-col border-r border-(--ink-border)">
        <div className="flex h-8 shrink-0 items-center gap-2 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
          <Bone size={13} />
          <span className="truncate font-medium" title={doc.id}>
            {doc.name}
          </span>
          {doc.dirty && <span className="text-(--ink-accent)">●</span>}
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto py-1">
          {joints.map((j) => (
            <button
              key={j.index}
              className={cn(
                "flex w-full items-center gap-1 px-2 py-0.5 text-left text-[11px] hover:bg-(--ink-bg-3)",
                selected === j.index &&
                  "bg-(--ink-accent) text-(--ink-text-onaccent)",
              )}
              style={{ paddingLeft: `${8 + depthOf(joints, j.index) * 10}px` }}
              onClick={() => select(assetId, j.index)}
              title={
                j.canonical
                  ? "A canonical humanoid joint — renaming it costs a retarget pairing"
                  : "Not one of the canonical humanoid joints"
              }
            >
              <span className="truncate">{j.name}</span>
              {j.canonical && (
                <span className="text-(--ink-text-faint)">◆</span>
              )}
              {j.sidedWithoutTwin && (
                <span
                  className="ml-auto text-(--ink-warn,#ffb454)"
                  title="This joint names a side but the rig has no twin for it — mirroring would weight the copy to the wrong side"
                >
                  ⚠
                </span>
              )}
            </button>
          ))}
        </div>
        {doc.unmatchedSided.length > 0 && (
          <div className="shrink-0 border-t border-(--ink-border) px-2 py-1 text-[10px] text-(--ink-warn,#ffb454)">
            unpaired: {doc.unmatchedSided.join(", ")}
          </div>
        )}
      </div>

      {/* ── the rig ─────────────────────────────────────────────────────── */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-8 shrink-0 items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
          <span className="text-(--ink-text-dim)">
            {joints.length} joints · {doc.sockets.length} sockets
          </span>
          <div className="ml-auto flex items-center gap-1">
            <button
              className="rounded p-1 hover:bg-(--ink-bg-3) disabled:opacity-40"
              onClick={() => void store.getState().undo(assetId)}
              disabled={!doc.canUndo}
              title="Undo (Ctrl+Z)"
            >
              <Undo2 size={13} />
            </button>
            <button
              className="rounded p-1 hover:bg-(--ink-bg-3) disabled:opacity-40"
              onClick={() => void store.getState().redo(assetId)}
              disabled={!doc.canRedo}
              title="Redo (Ctrl+Y)"
            >
              <Redo2 size={13} />
            </button>
            <button
              className="flex items-center gap-1 rounded bg-(--ink-accent) px-2 py-0.5 text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-40"
              onClick={() => void store.getState().save(assetId)}
              disabled={busy || !doc.dirty}
              title="Write the rig back to its .inf_skel"
            >
              <Save size={12} /> Save
            </button>
          </div>
        </div>
        <div className="flex min-h-0 flex-1 items-center justify-center bg-(--ink-bg-0) p-3">
          <svg
            viewBox={`0 0 ${VIEW} ${VIEW}`}
            className="max-h-full max-w-full"
            style={{ width: VIEW, height: VIEW }}
            role="img"
            aria-label="skeleton rest pose"
          >
            {/* Bones: a segment from each joint to its parent. */}
            {placed.map((p) =>
              p.joint.parent === null ? null : (
                <line
                  key={`b${p.joint.index}`}
                  x1={placed[p.joint.parent].x}
                  y1={placed[p.joint.parent].y}
                  x2={p.x}
                  y2={p.y}
                  stroke="var(--ink-border-strong)"
                  strokeWidth={1.5}
                />
              ),
            )}
            {/* The selected joint's mirror twin, so a pairing is visible. */}
            {sel?.mirror !== null &&
              sel?.mirror !== undefined &&
              placed[sel.mirror] && (
                <line
                  x1={placed[sel.index].x}
                  y1={placed[sel.index].y}
                  x2={placed[sel.mirror].x}
                  y2={placed[sel.mirror].y}
                  stroke="var(--ink-info)"
                  strokeWidth={0.75}
                  strokeDasharray="3 3"
                />
              )}
            {placed.map((p) => (
              <circle
                key={`j${p.joint.index}`}
                cx={p.x}
                cy={p.y}
                r={selected === p.joint.index ? 4.5 : 2.5}
                fill={
                  selected === p.joint.index
                    ? "var(--ink-accent)"
                    : p.joint.sidedWithoutTwin
                      ? "var(--ink-warning)"
                      : "var(--ink-text-dim)"
                }
                className="cursor-pointer"
                onClick={() => select(assetId, p.joint.index)}
              >
                <title>{`${p.joint.name} — ${p.world.map((v) => v.toFixed(2)).join(", ")} m`}</title>
              </circle>
            ))}
            {/* Sockets ride a joint; drawn as a ring on it. */}
            {doc.sockets.map((s) =>
              placed[s.joint] ? (
                <circle
                  key={`s${s.name}`}
                  cx={placed[s.joint].x + s.translation[0] * 6}
                  cy={placed[s.joint].y - s.translation[1] * 6}
                  r={6}
                  fill="none"
                  stroke="var(--ink-success)"
                  strokeWidth={1}
                >
                  <title>{`socket ${s.name} on ${joints[s.joint]?.name}`}</title>
                </circle>
              ) : null,
            )}
          </svg>
        </div>
        <div className="flex h-6 shrink-0 items-center gap-3 border-t border-(--ink-border) bg-(--ink-bg-2) px-2 text-[11px]">
          {warning && (
            <span className="truncate text-(--ink-warning)">{warning}</span>
          )}
          {refusal && (
            <span className="ml-auto truncate text-(--ink-warn,#ffb454)">
              {refusal}
            </span>
          )}
        </div>
      </div>

      {/* ── inspector ───────────────────────────────────────────────────── */}
      <div className="flex w-64 shrink-0 flex-col gap-3 overflow-y-auto border-l border-(--ink-border) p-2">
        {sel ? (
          <>
            <Section title="JOINT">
              <div className="flex gap-1">
                {/* **Uncontrolled, keyed by the joint** — not React state
                    synced by an effect. The draft belongs to the joint, so
                    selecting a different one must reset it; `key` does that by
                    remounting, which is React's own answer and the one the
                    repo's `set-state-in-effect` ban leaves standing (the P6
                    canvas law: an effect that calls setState is a cascading
                    render). */}
                <input
                  key={`${sel.index}:${sel.name}`}
                  defaultValue={sel.name}
                  onBlur={(e) =>
                    e.target.value !== sel.name &&
                    void store
                      .getState()
                      .renameJoint(assetId, sel.index, e.target.value)
                  }
                  className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 outline-none focus:border-(--ink-accent)"
                />
              </div>
              <div className="text-[10px] text-(--ink-text-dim)">
                #{sel.index}
                {sel.parent !== null &&
                  ` · child of ${joints[sel.parent]?.name}`}
                {sel.mirror !== null &&
                  ` · mirrors ${joints[sel.mirror]?.name}`}
              </div>
            </Section>

            <Section title="REST TRANSFORM">
              <Vec3Field
                value={sel.translation as [number, number, number]}
                onChange={(t) =>
                  void store
                    .getState()
                    .setJointTransform(
                      assetId,
                      sel.index,
                      t,
                      sel.rotation,
                      sel.scale,
                    )
                }
              />
              <Vec3Field
                value={sel.scale as [number, number, number]}
                onChange={(sc) =>
                  void store
                    .getState()
                    .setJointTransform(
                      assetId,
                      sel.index,
                      sel.translation,
                      sel.rotation,
                      sc,
                    )
                }
              />
              {/* Shown, not edited — see the module docs. */}
              <div className="font-mono text-[10px] text-(--ink-text-faint)">
                q {sel.rotation.map((v) => v.toFixed(3)).join(" ")}
              </div>
            </Section>

            <Section title="HINGE LIMIT">
              <label className="flex items-center gap-2 text-[11px]">
                <input
                  type="checkbox"
                  checked={limited}
                  onChange={(e) =>
                    void store
                      .getState()
                      .setLimit(
                        assetId,
                        sel.index,
                        e.target.checked ? [0, 0, 0] : null,
                        e.target.checked ? [90, 0, 0] : null,
                      )
                  }
                />
                <span className="text-(--ink-text-dim)">
                  limited (absent = free)
                </span>
              </label>
              {limited && sel.limitMinDeg && sel.limitMaxDeg && (
                <>
                  <Num
                    label="min °(X)"
                    value={sel.limitMinDeg[0]}
                    onChange={(v) =>
                      void store
                        .getState()
                        .setLimit(
                          assetId,
                          sel.index,
                          [v, 0, 0],
                          sel.limitMaxDeg as [number, number, number],
                        )
                    }
                  />
                  <Num
                    label="max °(X)"
                    value={sel.limitMaxDeg[0]}
                    onChange={(v) =>
                      void store
                        .getState()
                        .setLimit(
                          assetId,
                          sel.index,
                          sel.limitMinDeg as [number, number, number],
                          [v, 0, 0],
                        )
                    }
                  />
                </>
              )}
            </Section>

            <Section title="SOCKETS">
              <div className="flex gap-1">
                <input
                  value={socketName}
                  onChange={(e) => setSocketName(e.target.value)}
                  placeholder="name"
                  className="min-w-0 flex-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 outline-none focus:border-(--ink-accent)"
                />
                <button
                  className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 hover:bg-(--ink-bg-3)"
                  onClick={() => {
                    void store
                      .getState()
                      .addSocket(assetId, socketName, sel.index);
                    setSocketName("");
                  }}
                  title={`Add a socket on ${sel.name}`}
                >
                  <Plus size={12} />
                </button>
              </div>
              {doc.sockets.map((s) => (
                <div
                  key={s.name}
                  className="flex items-center gap-1 text-[11px]"
                >
                  <span
                    className="truncate"
                    title={`q ${s.rotation.join(" ")} · s ${s.scale.join(" ")}`}
                  >
                    {s.name}
                  </span>
                  <span className="text-(--ink-text-faint)">
                    @{joints[s.joint]?.name}
                  </span>
                  <button
                    className="ml-auto rounded p-0.5 hover:bg-(--ink-bg-3)"
                    onClick={() =>
                      void store
                        .getState()
                        .placeSocket(assetId, s.name, sel.index, [0, 0, 0])
                    }
                    title={`Move ${s.name} onto ${sel.name}`}
                  >
                    <Crosshair size={11} />
                  </button>
                  <button
                    className="rounded p-0.5 hover:bg-(--ink-bg-3)"
                    onClick={() =>
                      void store.getState().removeSocket(assetId, s.name)
                    }
                    title={`Remove ${s.name}`}
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              ))}
            </Section>
          </>
        ) : (
          <div className="text-(--ink-text-dim)">Select a joint.</div>
        )}

        <Section title="TEMPLATE">
          <select
            value={plan}
            onChange={(e) => setPlan(e.target.value as BodyPlanName)}
            className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 outline-none"
          >
            <option value="biped">Biped</option>
            <option value="quadruped">Quadruped</option>
            <option value="hexapod">Hexapod</option>
          </select>
          <Num
            label="height m"
            value={heightM}
            step={0.05}
            onChange={setHeightM}
          />
          <button
            className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 text-left hover:bg-(--ink-bg-3)"
            onClick={() =>
              void store
                .getState()
                .instantiateTemplate(assetId, plan, { heightM })
            }
            title="Replace this rig with a generated template (undoable)"
          >
            Regenerate from template
          </button>
        </Section>

        <Section title="AUTO-FIT">
          <select
            value={fitMesh}
            onChange={(e) => setFitMesh(e.target.value)}
            className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5 outline-none"
          >
            <option value="">pick a mesh…</option>
            {meshAssets.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
          <button
            className="flex items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 hover:bg-(--ink-bg-3) disabled:opacity-40"
            disabled={!fitMesh || busy}
            onClick={() =>
              void store.getState().fitToMesh(assetId, fitMesh, plan)
            }
            title="Fit a template into that mesh and replace this rig (undoable)"
          >
            <Scan size={12} /> Fit into mesh
          </button>
        </Section>
      </div>
    </div>
  );
}

/** How deep a joint sits, for the tree's indentation. */
function depthOf(joints: SkelJointDto[], index: number): number {
  let d = 0;
  let cur: number | null = index;
  // Bounded by the joint count: a malformed cyclic parent chain must indent
  // wrongly rather than hang the panel.
  while (cur !== null && d < joints.length) {
    cur = joints[cur]?.parent ?? null;
    if (cur !== null) d += 1;
  }
  return d;
}
