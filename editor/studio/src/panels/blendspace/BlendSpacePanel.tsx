/**
 * The 2D blend-space authoring panel (P29.2 clause 8).
 *
 * Place clips on a parameter plane, watch the **triangulation** the engine
 * actually uses appear under them, drag the query point and see which samples
 * carry it and what the blended pose looks like.
 *
 * # Fully controlled, like every other canvas here
 *
 * The SVG derives everything from `blendSpaceStore`; a drag writes through a store
 * action and the geometry comes back from the engine. There is no local mirror of
 * the sample list and no `setState` inside an effect (the eslint rule this repo
 * enforces) — the one effect calls `init`, exactly as `StateMachineCanvas` does.
 *
 * # What the preview is, and is not
 *
 * It is the **real blend**: `blendspace_preview` runs
 * `inf_anim::sample_blend_space_2d` over the placed clips and returns the posed
 * skeleton's joints in model space, which the right-hand pane draws as a stick
 * figure. It is not a shaded render — a lit preview needs the offscreen character
 * preview path, which is P29.5's `preview_character` work and is named in the
 * ROADMAP's remainders rather than faked here.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useBlendSpaceStore } from "../../stores/blendSpaceStore";
import { useSmStore } from "../../stores/smStore";
import "../blueprint/blueprint.css";
import "./blendspace.css";

/** Half-width of the drawn parameter plane, in parameter units. */
const EXTENT = 1.6;
/** The SVG's viewBox is a square this many units across. */
const VIEW = 200;

/** Parameter units → SVG coordinates. Y is flipped so +Y is up, which is what a
 *  locomotion author means by "forward". */
function toSvg(x: number, y: number): [number, number] {
  const half = VIEW / 2;
  return [half + (x / EXTENT) * half, half - (y / EXTENT) * half];
}

/** SVG coordinates → parameter units (the inverse of `toSvg`). */
function toParam(px: number, py: number): [number, number] {
  const half = VIEW / 2;
  return [((px - half) / half) * EXTENT, ((half - py) / half) * EXTENT];
}

export default function BlendSpacePanel() {
  const init = useBlendSpaceStore((s) => s.init);
  const samples = useBlendSpaceStore((s) => s.samples);
  const preview = useBlendSpaceStore((s) => s.preview);
  const cursorX = useBlendSpaceStore((s) => s.cursorX);
  const cursorY = useBlendSpaceStore((s) => s.cursorY);
  const timeS = useBlendSpaceStore((s) => s.timeS);
  const paramX = useBlendSpaceStore((s) => s.paramX);
  const paramY = useBlendSpaceStore((s) => s.paramY);
  const busy = useBlendSpaceStore((s) => s.busy);
  const status = useBlendSpaceStore((s) => s.status);

  const addSample = useBlendSpaceStore((s) => s.addSample);
  const moveSample = useBlendSpaceStore((s) => s.moveSample);
  const removeSample = useBlendSpaceStore((s) => s.removeSample);
  const setSampleClip = useBlendSpaceStore((s) => s.setSampleClip);
  const setCursor = useBlendSpaceStore((s) => s.setCursor);
  const setTime = useBlendSpaceStore((s) => s.setTime);
  const setParams = useBlendSpaceStore((s) => s.setParams);
  const reset = useBlendSpaceStore((s) => s.reset);
  const pullFromState = useBlendSpaceStore((s) => s.pullFromState);
  const pushToState = useBlendSpaceStore((s) => s.pushToState);

  const clips = useSmStore((s) => s.clips);
  const clipName = useSmStore((s) => s.clipName);
  const smDoc = useSmStore((s) => s.doc);

  const [selected, setSelected] = useState<number | null>(null);
  const [stateIndex, setStateIndex] = useState(0);
  /** `null` = not dragging; a number = the sample being dragged; `-1` = cursor. */
  const dragRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    void init();
  }, [init]);

  const weightOf = useMemo(() => {
    const m = new Map<number, number>();
    for (const w of preview?.weights ?? []) m.set(w.index, w.weight);
    return m;
  }, [preview]);

  const pointAt = useCallback((e: React.PointerEvent): [number, number] => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return [0, 0];
    const px = ((e.clientX - rect.left) / rect.width) * VIEW;
    const py = ((e.clientY - rect.top) / rect.height) * VIEW;
    return toParam(px, py);
  }, []);

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      const which = dragRef.current;
      if (which === null) return;
      const [x, y] = pointAt(e);
      if (which === -1) void setCursor(x, y);
      else void moveSample(which, x, y);
    },
    [moveSample, pointAt, setCursor],
  );

  const endDrag = useCallback(() => {
    dragRef.current = null;
  }, []);

  const onPaneDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const px = ((e.clientX - rect.left) / rect.width) * VIEW;
      const py = ((e.clientY - rect.top) / rect.height) * VIEW;
      const [x, y] = toParam(px, py);
      void addSample(x, y);
    },
    [addSample],
  );

  const [cx, cy] = toSvg(cursorX, cursorY);
  const sel = selected != null ? samples[selected] : undefined;

  // The preview skeleton, projected on its own XY plane and normalised into the
  // pane. A stick figure, drawn from the engine's joint positions.
  const figure = useMemo(() => {
    const joints = preview?.joints ?? [];
    if (joints.length === 0) return null;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const j of joints) {
      minX = Math.min(minX, j.position[0]);
      maxX = Math.max(maxX, j.position[0]);
      minY = Math.min(minY, j.position[1]);
      maxY = Math.max(maxY, j.position[1]);
    }
    const span = Math.max(maxX - minX, maxY - minY, 1e-3);
    const project = (p: [number, number, number]): [number, number] => [
      VIEW / 2 + ((p[0] - (minX + maxX) / 2) / span) * VIEW * 0.8,
      VIEW - 10 - ((p[1] - minY) / span) * VIEW * 0.8,
    ];
    return {
      points: joints.map((j) => project(j.position)),
      bones: joints
        .map((j, i) => (j.parent == null ? null : ([j.parent, i] as [number, number])))
        .filter((b): b is [number, number] => b !== null),
    };
  }, [preview]);

  return (
    <div className="bp-canvas bs-panel">
      <div className="bp-toolbar">
        <span className="pcg-toolbar__title">Blend Space</span>
        <span className="bp-toolbar__sep" />
        <label className="bs-field">
          X
          <input
            className="bs-input bs-input--name"
            value={paramX}
            onChange={(e) => setParams(e.target.value, paramY)}
          />
        </label>
        <label className="bs-field">
          Y
          <input
            className="bs-input bs-input--name"
            value={paramY}
            onChange={(e) => setParams(paramX, e.target.value)}
          />
        </label>
        <span className="bp-toolbar__sep" />
        <select
          className="bs-input"
          value={stateIndex}
          onChange={(e) => setStateIndex(Number(e.target.value))}
        >
          {(smDoc?.machine.states ?? []).map((s, i) => (
            <option key={i} value={i}>
              {s.name}
            </option>
          ))}
          {!smDoc && <option value={0}>(no state machine)</option>}
        </select>
        <button className="bp-btn" disabled={!smDoc} onClick={() => void pullFromState(stateIndex)}>
          Pull
        </button>
        <button className="bp-btn" disabled={!smDoc} onClick={() => pushToState(stateIndex)}>
          Push
        </button>
        <span className="bp-toolbar__spacer" />
        <button className="bp-btn" onClick={() => void reset()}>
          Reset
        </button>
      </div>

      <div className="pcg-body bs-body">
        <div className="bs-plane">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${VIEW} ${VIEW}`}
            className="bs-svg"
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerLeave={endDrag}
            onDoubleClick={onPaneDoubleClick}
          >
            {/* axes */}
            <line x1={0} y1={VIEW / 2} x2={VIEW} y2={VIEW / 2} className="bs-axis" />
            <line x1={VIEW / 2} y1={0} x2={VIEW / 2} y2={VIEW} className="bs-axis" />

            {/* the engine's triangulation */}
            {(preview?.triangles ?? []).map((t, i) => {
              const pts = t
                .map((k) => samples[k])
                .filter(Boolean)
                .map((s) => toSvg(s.x, s.y).join(","))
                .join(" ");
              return <polygon key={`t${i}`} points={pts} className="bs-tri" />;
            })}
            {/* the hull, or the chain when the samples have no area */}
            {(preview?.hull ?? []).map((e, i) => {
              const a = samples[e[0]];
              const b = samples[e[1]];
              if (!a || !b) return null;
              const [ax, ay] = toSvg(a.x, a.y);
              const [bx, by] = toSvg(b.x, b.y);
              return <line key={`h${i}`} x1={ax} y1={ay} x2={bx} y2={by} className="bs-hull" />;
            })}

            {/* the samples */}
            {samples.map((s, i) => {
              const [x, y] = toSvg(s.x, s.y);
              const w = weightOf.get(i) ?? 0;
              return (
                <g key={`s${i}`}>
                  <circle
                    cx={x}
                    cy={y}
                    r={4 + 6 * w}
                    className={`bs-sample${selected === i ? " bs-sample--sel" : ""}${
                      w > 0 ? " bs-sample--live" : ""
                    }`}
                    onPointerDown={(e) => {
                      e.stopPropagation();
                      dragRef.current = i;
                      setSelected(i);
                    }}
                  />
                  <text x={x + 7} y={y - 6} className="bs-label">
                    {s.clip ? clipName(s.clip) : "(no clip)"}
                    {w > 0 ? ` ${(w * 100).toFixed(0)}%` : ""}
                  </text>
                </g>
              );
            })}

            {/* the query point */}
            <g
              onPointerDown={(e) => {
                e.stopPropagation();
                dragRef.current = -1;
              }}
            >
              <circle cx={cx} cy={cy} r={5} className="bs-cursor" />
              <line x1={cx - 9} y1={cy} x2={cx + 9} y2={cy} className="bs-cursor-cross" />
              <line x1={cx} y1={cy - 9} x2={cx} y2={cy + 9} className="bs-cursor-cross" />
            </g>
          </svg>
          <div className="bs-hint">
            Double-click to place a clip · drag a sample to move it · drag the cross to blend
          </div>
        </div>

        <div className="bs-side">
          <div className="bs-section">
            <div className="bs-section__title">
              Blend at ({cursorX.toFixed(2)}, {cursorY.toFixed(2)}){busy ? " · …" : ""}
            </div>
            {preview?.degenerate && (
              <div className="bs-note">
                These samples have no area — the engine interpolates along the row.
              </div>
            )}
            {(preview?.weights ?? []).map((w) => (
              <div className="bs-weight" key={w.index}>
                <span className="bs-weight__name">
                  {samples[w.index]?.clip ? clipName(samples[w.index].clip) : `#${w.index}`}
                </span>
                <span className="bs-weight__bar">
                  <span style={{ width: `${(w.weight * 100).toFixed(1)}%` }} />
                </span>
                <span className="bs-weight__pct">{(w.weight * 100).toFixed(0)}%</span>
              </div>
            ))}
            <div className="bs-stat">
              cycle {preview ? preview.blendedDurationS.toFixed(3) : "—"} s
            </div>
          </div>

          <div className="bs-section">
            <div className="bs-section__title">Preview</div>
            <input
              type="range"
              min={0}
              max={2}
              step={0.01}
              value={timeS}
              onChange={(e) => void setTime(Number(e.target.value))}
              className="bs-slider"
            />
            <div className="bs-stat">t = {timeS.toFixed(2)} s</div>
            {preview?.refusal && <div className="bs-note">{preview.refusal}</div>}
            {figure && (
              <svg viewBox={`0 0 ${VIEW} ${VIEW}`} className="bs-figure">
                {figure.bones.map(([a, b], i) => (
                  <line
                    key={`b${i}`}
                    x1={figure.points[a][0]}
                    y1={figure.points[a][1]}
                    x2={figure.points[b][0]}
                    y2={figure.points[b][1]}
                    className="bs-bone"
                  />
                ))}
                {figure.points.map(([x, y], i) => (
                  <circle key={`j${i}`} cx={x} cy={y} r={2} className="bs-joint" />
                ))}
              </svg>
            )}
          </div>

          <div className="bs-section">
            <div className="bs-section__title">Sample</div>
            {sel ? (
              <>
                <label className="bs-field bs-field--row">
                  clip
                  <select
                    className="bs-input"
                    value={sel.clip ?? ""}
                    onChange={(e) =>
                      void setSampleClip(selected!, e.target.value === "" ? null : e.target.value)
                    }
                  >
                    <option value="">(none)</option>
                    {clips.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="bs-stat">
                  at ({sel.x.toFixed(2)}, {sel.y.toFixed(2)})
                </div>
                <button
                  className="bp-btn"
                  onClick={() => {
                    const i = selected!;
                    setSelected(null);
                    void removeSample(i);
                  }}
                >
                  Remove
                </button>
              </>
            ) : (
              <div className="bs-note">Select a sample to assign its clip.</div>
            )}
          </div>

          {status && <div className="bs-note bs-note--status">{status}</div>}
        </div>
      </div>
    </div>
  );
}
