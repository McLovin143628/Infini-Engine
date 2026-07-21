/**
 * Viewport toolbar (P8.2c): the Perspective / 2D projection toggle and, in 2D
 * mode, the grid- and pixel-snapping controls. Rendered as a strip ABOVE the
 * native viewport hole (never over it — the native child window would occlude
 * HTML there; airspace rule). All state lives in `viewportStore`, which mirrors
 * each change to the native viewport over typed IPC.
 */
import { useViewportStore } from "../stores/viewportStore";
import type { ViewportModeDto } from "../bindings/ViewportModeDto";

const MODES: [ViewportModeDto, string][] = [
  ["Perspective", "Perspective"],
  ["TwoD", "2D"],
];

export default function ViewportToolbar() {
  const mode = useViewportStore((s) => s.mode);
  const setMode = useViewportStore((s) => s.setMode);
  const gridSnapEnabled = useViewportStore((s) => s.gridSnapEnabled);
  const gridSnapSize = useViewportStore((s) => s.gridSnapSize);
  const pixelSnapEnabled = useViewportStore((s) => s.pixelSnapEnabled);
  const pixelsPerUnit = useViewportStore((s) => s.pixelsPerUnit);
  const setGridSnapEnabled = useViewportStore((s) => s.setGridSnapEnabled);
  const setGridSnapSize = useViewportStore((s) => s.setGridSnapSize);
  const setPixelSnapEnabled = useViewportStore((s) => s.setPixelSnapEnabled);
  const setPixelsPerUnit = useViewportStore((s) => s.setPixelsPerUnit);

  return (
    <div className="flex h-8 shrink-0 items-center gap-3 rounded border border-(--ink-border) bg-(--ink-bg-1) px-2 text-xs">
      {/* Projection toggle. */}
      <div className="flex items-center rounded bg-(--ink-bg-0) p-0.5" role="group" aria-label="Viewport projection">
        {MODES.map(([id, label]) => (
          <button
            key={id}
            type="button"
            aria-pressed={mode === id}
            title={id === "TwoD" ? "Orthographic 2D editing (XY plane)" : "Perspective 3D"}
            className={`flex h-6 items-center rounded px-2 ${
              mode === id
                ? "bg-(--ink-bg-3) text-(--ink-text)"
                : "text-(--ink-text-dim) hover:text-(--ink-text)"
            }`}
            onClick={() => setMode(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* 2D snapping controls (only meaningful in ortho mode). */}
      {mode === "TwoD" && (
        <>
          <span className="text-(--ink-text-dim)">Snap</span>
          <SnapToggle
            label="Grid"
            title="Snap translate to the grid increment (world units)"
            enabled={gridSnapEnabled}
            onToggle={() => setGridSnapEnabled(!gridSnapEnabled)}
          >
            <NumberField
              value={gridSnapSize}
              min={0}
              step={0.05}
              onChange={setGridSnapSize}
              title="Grid snap increment (world units)"
            />
          </SnapToggle>
          <SnapToggle
            label="Pixel"
            title="Snap translate to whole pixels (1/PPU world units)"
            enabled={pixelSnapEnabled}
            onToggle={() => setPixelSnapEnabled(!pixelSnapEnabled)}
          >
            <NumberField
              value={pixelsPerUnit}
              min={1}
              step={1}
              onChange={setPixelsPerUnit}
              title="Pixels per unit (per-project)"
              suffix="PPU"
            />
          </SnapToggle>
        </>
      )}
    </div>
  );
}

function SnapToggle(props: {
  label: string;
  title: string;
  enabled: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        aria-pressed={props.enabled}
        title={props.title}
        className={`flex h-6 items-center rounded px-2 ${
          props.enabled
            ? "bg-(--ink-accent) text-(--ink-bg-0)"
            : "bg-(--ink-bg-0) text-(--ink-text-dim) hover:text-(--ink-text)"
        }`}
        onClick={props.onToggle}
      >
        {props.label}
      </button>
      {props.children}
    </div>
  );
}

function NumberField(props: {
  value: number;
  min: number;
  step: number;
  onChange: (v: number) => void;
  title: string;
  suffix?: string;
}) {
  return (
    <div className="flex items-center gap-0.5">
      <input
        type="number"
        min={props.min}
        step={props.step}
        value={props.value}
        title={props.title}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) props.onChange(v);
        }}
        className="h-6 w-14 rounded border border-(--ink-border) bg-(--ink-bg-0) px-1 text-(--ink-text) tabular-nums"
      />
      {props.suffix && <span className="text-(--ink-text-dim)">{props.suffix}</span>}
    </div>
  );
}
