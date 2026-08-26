/**
 * Property-row primitives for the Details panel (P1.4.2): label + control
 * rows with per-property reset affordance. These are the building blocks
 * the reflection-driven Details walker (P3.3) will instantiate per type —
 * Phase 1 uses them with mock values.
 */
import { useId, useState, type ReactNode } from "react";
import { ChevronDown, ChevronUp, Plus, RotateCcw, X } from "lucide-react";
import { cn } from "../lib/utils";
import type { PropValueDto } from "../bindings/PropValueDto";
import type { PropFieldDto } from "../bindings/PropFieldDto";

export function PropertySection({
  title,
  action,
  children,
}: {
  title: string;
  /** Optional control rendered at the right of the header (e.g. E-P1 overflow). */
  action?: ReactNode;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border-b border-(--ink-border)">
      <div className="flex items-center bg-(--ink-bg-2) hover:bg-(--ink-bg-3)">
        <button
          className="flex min-w-0 flex-1 items-center gap-1 px-2 py-1 text-left text-xs font-semibold"
          onClick={() => setOpen((o) => !o)}
        >
          <span
            className={cn("inline-block transition-transform", open ? "rotate-90" : "rotate-0")}
          >
            ▸
          </span>
          <span className="truncate">{title}</span>
        </button>
        {action && <div className="shrink-0 pr-1">{action}</div>}
      </div>
      {open && <div className="py-0.5">{children}</div>}
    </div>
  );
}

export function PropertyRow({
  label,
  changed,
  onReset,
  children,
}: {
  label: string;
  /** Shows the per-property reset arrow (P3.4 wires real defaults). */
  changed?: boolean;
  onReset?: () => void;
  children: ReactNode;
}) {
  const id = useId();
  return (
    <div className="group flex min-h-6 items-center gap-2 px-2 py-0.5 hover:bg-(--ink-bg-2)/60">
      <label htmlFor={id} className="w-28 shrink-0 truncate text-xs text-(--ink-text-dim)">
        {label}
      </label>
      <div className="flex min-w-0 flex-1 items-center gap-1" data-prop-control-id={id}>
        {children}
      </div>
      <button
        aria-label={`Reset ${label}`}
        onClick={onReset}
        className={cn(
          "size-4 shrink-0 items-center justify-center rounded-sm text-(--ink-text-faint) hover:text-(--ink-text)",
          changed ? "flex" : "invisible flex",
        )}
      >
        <RotateCcw size={11} />
      </button>
    </div>
  );
}

const numberInput =
  "h-6 w-full min-w-0 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 text-xs outline-none focus:border-(--ink-accent)";

/**
 * A number row, guarded on the **write** as well as the display (round-2
 * finding R2.F11).
 *
 * The display guard was here; the write was not. Typing `1e999` into any of the
 * ~60 World Settings rows stored `Infinity` optimistically, rendered it as `0`
 * (so the field looked reset rather than wrong), and serialized to JSON `null`
 * — which `scene_set_settings` refuses. And because every `patch*` action
 * spreads the whole settings object, EVERY later edit re-sent the same
 * `Infinity` and failed too: one keystroke and World Settings stopped saving
 * for the session, silently.
 *
 * The sibling `ViewportToolbar.NumberField` already guarded with
 * `Number.isFinite`; this is that guard, at the door that writes.
 */
export function NumberField({
  value,
  onChange,
  step = 1,
  className,
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  className?: string;
}) {
  return (
    <input
      type="number"
      value={Number.isFinite(value) ? +value.toFixed(4) : 0}
      step={step}
      onChange={(e) => {
        const n = Number(e.target.value);
        // A non-finite entry is not a number the author meant. Dropping it
        // leaves the previous value, which is what an unparseable keystroke has
        // always done in this field ("" is `NaN` and was equally unguarded).
        if (Number.isFinite(n)) onChange(n);
      }}
      className={cn(numberInput, className)}
    />
  );
}

/**
 * A number row with a **slider** beside it (wave VIS1b), for the fields the
 * engine knows the units of.
 *
 * The range comes from `inf_ecs::props::prop_range` through `PropFieldDto.range`
 * — it is not typed here, and that is the point: the same `0 / 1 / 0.01` triple
 * was already retyped by hand in four panels, and a number restated in four
 * places is four chances to disagree.
 *
 * The **slider** clamps to the range because a slider is a range; the **number
 * box does not**, because a range is a display hint and a value an author
 * deliberately typed is not the widget's business. (An emissive intensity of 200
 * is a legal thing to want; the slider simply cannot reach it.) The write door
 * that does refuse is `inf_ecs::props::apply_value`, and it refuses exactly one
 * thing: a value that is not finite.
 */
export function RangedNumberField({
  value,
  onChange,
  range,
  className,
}: {
  value: number;
  onChange: (v: number) => void;
  range: readonly [number, number, number];
  className?: string;
}) {
  const [min, max, step] = range;
  const shown = Number.isFinite(value) ? value : min;
  return (
    <div className={cn("flex w-full min-w-0 items-center gap-1.5", className)}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={Math.min(Math.max(shown, min), max)}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(n);
        }}
        className="h-1 min-w-0 flex-1 accent-(--ink-accent)"
      />
      <NumberField value={value} onChange={onChange} step={step} className="w-14 shrink-0" />
    </div>
  );
}

const AXIS_TINTS = ["text-(--ink-error)", "text-(--ink-success)", "text-(--ink-info)"] as const;

export function Vec3Field({
  value,
  onChange,
}: {
  value: [number, number, number];
  onChange: (v: [number, number, number]) => void;
}) {
  return (
    <div className="flex min-w-0 flex-1 gap-1">
      {(["X", "Y", "Z"] as const).map((axis, i) => (
        <div key={axis} className="flex min-w-0 flex-1 items-center gap-1">
          <span className={cn("text-[10px] font-bold", AXIS_TINTS[i])}>{axis}</span>
          <NumberField
            value={value[i]}
            onChange={(v) => {
              const next: [number, number, number] = [...value];
              next[i] = v;
              onChange(next);
            }}
          />
        </div>
      ))}
    </div>
  );
}

export function TextField({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className={numberInput}
    />
  );
}

export function CheckboxField({
  value,
  onChange,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <input
      type="checkbox"
      checked={value}
      onChange={(e) => onChange(e.target.checked)}
      className="size-3.5 accent-(--ink-accent)"
    />
  );
}

export function EnumField({
  value,
  options,
  onChange,
}: {
  value: string;
  options: readonly string[];
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-6 w-full rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 text-xs outline-none focus:border-(--ink-accent)"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

export function ColorField({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="color"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-6 w-10 cursor-pointer rounded border border-(--ink-border) bg-(--ink-bg-2)"
      />
      <span className="text-xs text-(--ink-text-dim)">{value}</span>
    </div>
  );
}

// ── E-P1 deep editing ──────────────────────────────────────────────────────

const iconBtn =
  "flex size-5 shrink-0 items-center justify-center rounded-sm text-(--ink-text-faint) hover:bg-(--ink-bg-3) hover:text-(--ink-text) disabled:cursor-not-allowed disabled:opacity-30";

// Pure list-mutation helpers (E-P1) — exported so the ListField wiring is unit
// testable without a DOM. Each returns a NEW array (the whole list is sent back).
export function listSetAt(list: PropValueDto[], i: number, v: PropValueDto): PropValueDto[] {
  const next = list.slice();
  next[i] = v;
  return next;
}
export function listRemoveAt(list: PropValueDto[], i: number): PropValueDto[] {
  return list.filter((_, j) => j !== i);
}
export function listMove(list: PropValueDto[], i: number, dir: -1 | 1): PropValueDto[] {
  const j = i + dir;
  if (j < 0 || j >= list.length) return list;
  const next = list.slice();
  [next[i], next[j]] = [next[j], next[i]];
  return next;
}

/**
 * A homogeneous list (E-P1). Each row renders one element via the injected
 * `renderElement`; per-row remove / move-up / move-down and an "add" button
 * mutate the list and send the WHOLE list back through `onChange` (the backend
 * List write arm rebuilds the reflected collection). `onAdd` appends a default
 * element (the caller fetches it via `scene_list_default`).
 */
export function ListField({
  value,
  onChange,
  onAdd,
  renderElement,
}: {
  value: PropValueDto[];
  onChange: (next: PropValueDto[]) => void;
  onAdd: () => void;
  renderElement: (element: PropValueDto, index: number, set: (v: PropValueDto) => void) => ReactNode;
}) {
  const setAt = (i: number, e: PropValueDto) => onChange(listSetAt(value, i, e));
  const remove = (i: number) => onChange(listRemoveAt(value, i));
  const move = (i: number, dir: -1 | 1) => onChange(listMove(value, i, dir));
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-1">
      {value.length === 0 && (
        <span className="text-[11px] text-(--ink-text-faint)">(empty)</span>
      )}
      {value.map((elem, i) => (
        <div key={i} className="flex min-w-0 items-center gap-1">
          <span className="w-4 shrink-0 text-right text-[10px] text-(--ink-text-faint)">{i}</span>
          <div className="flex min-w-0 flex-1 items-center gap-1">
            {renderElement(elem, i, (e) => setAt(i, e))}
          </div>
          <button
            className={iconBtn}
            aria-label={`Move element ${i} up`}
            disabled={i === 0}
            onClick={() => move(i, -1)}
          >
            <ChevronUp size={12} />
          </button>
          <button
            className={iconBtn}
            aria-label={`Move element ${i} down`}
            disabled={i === value.length - 1}
            onClick={() => move(i, 1)}
          >
            <ChevronDown size={12} />
          </button>
          <button className={iconBtn} aria-label={`Remove element ${i}`} onClick={() => remove(i)}>
            <X size={12} />
          </button>
        </div>
      ))}
      <button
        className="flex h-5 items-center gap-1 self-start rounded-sm px-1 text-[11px] text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
        onClick={onAdd}
      >
        <Plus size={11} /> Add
      </button>
    </div>
  );
}

/**
 * A nested struct (E-P1): indented child rows. Editing a child updates that
 * child and sends the WHOLE struct back through `onChange`; the child control is
 * rendered by the injected `renderChild`.
 */
export function StructField({
  fields,
  onChange,
  renderChild,
}: {
  fields: PropFieldDto[];
  onChange: (next: PropFieldDto[]) => void;
  renderChild: (value: PropValueDto, set: (v: PropValueDto) => void) => ReactNode;
}) {
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-0.5 border-l border-(--ink-border) pl-1.5">
      {fields.map((f, i) => (
        <div key={f.name} className="flex min-w-0 items-center gap-2">
          <span className="w-20 shrink-0 truncate text-[11px] text-(--ink-text-dim)">
            {f.label}
          </span>
          <div className="flex min-w-0 flex-1 items-center gap-1">
            {renderChild(f.value, (v) => {
              const next = fields.slice();
              next[i] = { ...f, value: v };
              onChange(next);
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
