import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { LucideIcon } from "lucide-react";
import { cn } from "../lib/utils";
import { useViewportOverlay } from "../lib/viewportOverlay";

/**
 * Minimal right-click context menu (no external UI dependency).
 *
 * Two faces of one implementation:
 *
 * * [`ContextMenu`] wraps a trigger — `<ContextMenu items={…}>…</ContextMenu>` —
 *   using `display:contents` so it never disturbs layout or the dock system's
 *   DOM measurements. This is the original Phase-1 shape.
 * * [`ContextMenuSurface`] is the same menu, **controlled**: open it at an
 *   arbitrary `(x, y)` with items you computed however you liked. Wave E added
 *   it because two callers cannot use a trigger wrapper — a menu whose position
 *   arrives over an IPC event from the native viewport (there is no DOM element
 *   to wrap; the hole swallows pointer events), and a menu whose items are the
 *   result of an `await` (the Outliner asks the backend which editors an object
 *   has before deciding what to offer).
 *
 * **Airspace**: both faces hold `useViewportOverlay` for the menu's open
 * lifetime, so the native viewport child window is hidden while a menu that
 * could cross it is up. That is the shell's standing behaviour for every
 * overlay, and it is why the surface — not each caller — owns the guard.
 */

export interface ContextMenuItem {
  label: string;
  icon?: LucideIcon;
  disabled?: boolean;
  danger?: boolean;
  /** Shown right-aligned, dim — an accelerator or an explanation. */
  hint?: string;
  onSelect: () => void;
}

export type ContextMenuEntry = ContextMenuItem | "separator";

/** Where a controlled menu opens, in CSS pixels relative to the window. */
export interface MenuAnchor {
  x: number;
  y: number;
}

/**
 * The controlled menu surface. Renders nothing when `at` is null.
 *
 * Dismissal (outside pointer-down, Escape, window blur) calls `onClose`; the
 * caller owns the state, so a menu can be re-opened at a new point without a
 * close/open flicker.
 */
export function ContextMenuSurface({
  at,
  items,
  onClose,
}: {
  at: MenuAnchor | null;
  items: ContextMenuEntry[];
  onClose: () => void;
}) {
  const menuRef = useRef<HTMLDivElement | null>(null);
  const open = at !== null;

  // The menu can open over the native viewport hole — hide it while open.
  useViewportOverlay(open);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("keydown", onKey);
    window.addEventListener("blur", onClose);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("blur", onClose);
    };
  }, [open, onClose]);

  // Keep the menu inside the window (measure after first paint, imperative
  // style write — no state round-trip).
  useEffect(() => {
    const el = menuRef.current;
    if (!at || !el) return;
    const r = el.getBoundingClientRect();
    const nx = Math.max(0, Math.min(at.x, window.innerWidth - r.width - 4));
    const ny = Math.max(0, Math.min(at.y, window.innerHeight - r.height - 4));
    el.style.left = `${nx}px`;
    el.style.top = `${ny}px`;
  }, [at]);

  if (!at) return null;

  return createPortal(
    <div
      ref={menuRef}
      className="fixed z-[90] min-w-48 rounded-md border border-(--ink-border) bg-(--ink-bg-2) py-1 text-(--ink-text)"
      style={{ left: at.x, top: at.y, boxShadow: `0 8px 24px var(--ink-shadow)` }}
      role="menu"
    >
      {items.map((entry, i) =>
        entry === "separator" ? (
          <div key={i} className="mx-2 my-1 h-px bg-(--ink-border)" />
        ) : (
          <button
            key={i}
            role="menuitem"
            disabled={entry.disabled}
            title={entry.hint}
            className={cn(
              "mx-1 flex w-[calc(100%-0.5rem)] items-center gap-2 rounded px-2 py-1 text-left",
              entry.disabled
                ? "cursor-default text-(--ink-text-faint)"
                : entry.danger
                  ? "hover:bg-(--ink-error)/15 hover:text-(--ink-error)"
                  : "hover:bg-(--ink-bg-3)",
            )}
            onClick={() => {
              onClose();
              if (!entry.disabled) entry.onSelect();
            }}
          >
            {entry.icon && <entry.icon size={14} className="shrink-0" />}
            <span className="flex-1">{entry.label}</span>
            {entry.hint && (
              <span className="shrink-0 text-(--ink-text-faint)">{entry.hint}</span>
            )}
          </button>
        ),
      )}
    </div>,
    document.body,
  );
}

/** The trigger wrapper: right-click `children` to open the menu at the cursor. */
export function ContextMenu({
  items,
  children,
}: {
  items: ContextMenuEntry[] | (() => ContextMenuEntry[]);
  children: ReactNode;
}) {
  const [at, setAt] = useState<MenuAnchor | null>(null);
  const resolved = at ? (typeof items === "function" ? items() : items) : [];

  return (
    <span
      style={{ display: "contents" }}
      onContextMenu={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setAt({ x: e.clientX, y: e.clientY });
      }}
    >
      {children}
      <ContextMenuSurface at={at} items={resolved} onClose={() => setAt(null)} />
    </span>
  );
}
