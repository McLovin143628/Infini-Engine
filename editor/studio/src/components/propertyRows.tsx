/**
 * Property-row primitives for the Details panel (P1.4.2): label + control
 * rows with per-property reset affordance. These are the building blocks
 * the reflection-driven Details walker (P3.3) will instantiate per type —
 * Phase 1 uses them with mock values.
 */
import { useId, useState, type ReactNode } from "react";
import { RotateCcw } from "lucide-react";
import { cn } from "../lib/utils";

export function PropertySection({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border-b border-(--ink-border)">
      <button
        className="flex w-full items-center gap-1 bg-(--ink-bg-2) px-2 py-1 text-left text-xs font-semibold hover:bg-(--ink-bg-3)"
        onClick={() => setOpen((o) => !o)}
      >
        <span
          className={cn("inline-block transition-transform", open ? "rotate-90" : "rotate-0")}
        >
          ▸
        </span>
        {title}
      </button>
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
      onChange={(e) => onChange(Number(e.target.value))}
      className={cn(numberInput, className)}
    />
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
