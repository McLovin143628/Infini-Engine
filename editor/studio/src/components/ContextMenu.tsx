import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { LucideIcon } from "lucide-react";
import { cn } from "../lib/utils";

/**
 * Minimal right-click context menu (no external UI dependency). Wrap a
 * trigger in `<ContextMenu items={…}>`; the wrapper uses `display:contents`
 * so it never disturbs layout or the dock system's DOM measurements.
 */

export interface ContextMenuItem {
  label: string;
  icon?: LucideIcon;
  disabled?: boolean;
  danger?: boolean;
  onSelect: () => void;
}

export type ContextMenuEntry = ContextMenuItem | "separator";

export function ContextMenu({
  items,
  children,
}: {
  items: ContextMenuEntry[] | (() => ContextMenuEntry[]);
  children: ReactNode;
}) {
  const [open, setOpen] = useState<{ x: number; y: number } | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(null);
    const onPointerDown = (e: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("keydown", onKey);
    window.addEventListener("blur", close);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("blur", close);
    };
  }, [open]);

  // Keep the menu inside the viewport (measure after first paint).
  useEffect(() => {
    if (!open || !menuRef.current) return;
    const r = menuRef.current.getBoundingClientRect();
    const nx = Math.min(open.x, window.innerWidth - r.width - 4);
    const ny = Math.min(open.y, window.innerHeight - r.height - 4);
    if (nx !== open.x || ny !== open.y) setOpen({ x: Math.max(0, nx), y: Math.max(0, ny) });
  }, [open]);

  const resolved = open ? (typeof items === "function" ? items() : items) : [];

  return (
    <span
      style={{ display: "contents" }}
      onContextMenu={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setOpen({ x: e.clientX, y: e.clientY });
      }}
    >
      {children}
      {open &&
        createPortal(
          <div
            ref={menuRef}
            className="fixed z-[90] min-w-48 rounded-md border border-(--ink-border) bg-(--ink-bg-2) py-1 text-(--ink-text)"
            style={{ left: open.x, top: open.y, boxShadow: `0 8px 24px var(--ink-shadow)` }}
            role="menu"
          >
            {resolved.map((entry, i) =>
              entry === "separator" ? (
                <div key={i} className="mx-2 my-1 h-px bg-(--ink-border)" />
              ) : (
                <button
                  key={i}
                  role="menuitem"
                  disabled={entry.disabled}
                  className={cn(
                    "mx-1 flex w-[calc(100%-0.5rem)] items-center gap-2 rounded px-2 py-1 text-left",
                    entry.disabled
                      ? "cursor-default text-(--ink-text-faint)"
                      : entry.danger
                        ? "hover:bg-(--ink-error)/15 hover:text-(--ink-error)"
                        : "hover:bg-(--ink-bg-3)",
                  )}
                  onClick={() => {
                    setOpen(null);
                    if (!entry.disabled) entry.onSelect();
                  }}
                >
                  {entry.icon && <entry.icon size={14} className="shrink-0" />}
                  <span className="flex-1">{entry.label}</span>
                </button>
              ),
            )}
          </div>,
          document.body,
        )}
    </span>
  );
}
