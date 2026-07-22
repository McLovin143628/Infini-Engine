/**
 * Menu bar — renders the UE-parity tree from `lib/menus.ts`.
 * Click opens; hover switches while any menu is open; Escape / click-away
 * closes; submenus fly out on hover. Every action dispatches through the
 * command registry.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronRight } from "lucide-react";
import { executeCommand } from "../lib/commands";
import type { MenuNode, TopMenu } from "../lib/menus";
import { buildMenuBar } from "../lib/menus";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useProjectStore } from "../stores/projectStore";

export default function MenuBar() {
  const [openId, setOpenId] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // The File ▸ Recent Projects submenu is filled from the live recent list; every
  // other menu is static. Recompute only when the recent list changes.
  const recent = useProjectStore((s) => s.recent);
  const menuBar = useMemo(() => buildMenuBar(recent), [recent]);

  // Dropdowns extend over the native viewport hole — hide it while open.
  useViewportOverlay(openId !== null);

  const close = useCallback(() => setOpenId(null), []);

  useEffect(() => {
    if (openId === null) return;
    const onPointerDown = (e: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) close();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [openId, close]);

  return (
    <div ref={rootRef} className="flex h-full items-stretch">
      {menuBar.map((menu) => (
        <MenuButton
          key={menu.id}
          menu={menu}
          open={openId === menu.id}
          anyOpen={openId !== null}
          onOpen={() => setOpenId(menu.id)}
          onToggle={() => setOpenId((cur) => (cur === menu.id ? null : menu.id))}
          onClose={close}
        />
      ))}
    </div>
  );
}

function MenuButton(props: {
  menu: TopMenu;
  open: boolean;
  anyOpen: boolean;
  onOpen: () => void;
  onToggle: () => void;
  onClose: () => void;
}) {
  const { menu, open, anyOpen, onOpen, onToggle, onClose } = props;
  return (
    <div className="relative flex items-stretch">
      <button
        className={`px-2.5 text-(--ink-text) outline-none ${
          open ? "bg-(--ink-bg-3)" : "hover:bg-(--ink-bg-3)"
        }`}
        onPointerDown={(e) => {
          // pointerdown (not click) so opening feels instant, like native menus
          e.preventDefault();
          onToggle();
        }}
        onPointerEnter={() => {
          if (anyOpen && !open) onOpen();
        }}
      >
        {menu.label}
      </button>
      {open && (
        <div className="absolute top-full left-0 z-50">
          <MenuList items={menu.items} onAction={onClose} />
        </div>
      )}
    </div>
  );
}

function MenuList(props: { items: MenuNode[]; onAction: () => void }) {
  const { items, onAction } = props;
  const [openSub, setOpenSub] = useState<number | null>(null);
  return (
    <div
      className="min-w-56 rounded-md border border-(--ink-border) bg-(--ink-bg-2) py-1 shadow-lg"
      style={{ boxShadow: `0 8px 24px var(--ink-shadow)` }}
    >
      {items.map((node, i) => {
        if (node.kind === "separator") {
          return <div key={i} className="mx-2 my-1 h-px bg-(--ink-border)" />;
        }
        if (node.kind === "submenu") {
          return (
            <div
              key={i}
              className="relative"
              onPointerEnter={() => setOpenSub(i)}
              onPointerLeave={() => setOpenSub((cur) => (cur === i ? null : cur))}
            >
              <div className="mx-1 flex cursor-default items-center rounded px-2 py-1 hover:bg-(--ink-bg-3)">
                <span className="flex-1">{node.label}</span>
                <ChevronRight size={14} className="text-(--ink-text-dim)" />
              </div>
              {openSub === i && (
                <div className="absolute top-0 left-full z-50 -mt-1">
                  <MenuList items={node.items} onAction={onAction} />
                </div>
              )}
            </div>
          );
        }
        if (node.disabled) {
          return (
            <div
              key={i}
              title={node.detail}
              className="mx-1 flex items-center gap-6 rounded px-2 py-1 text-(--ink-text-faint)"
            >
              <span className="flex-1">{node.label}</span>
            </div>
          );
        }
        return (
          <button
            key={i}
            title={node.detail}
            className="mx-1 flex w-[calc(100%-0.5rem)] items-center gap-6 rounded px-2 py-1 text-left hover:bg-(--ink-bg-3)"
            onClick={() => {
              onAction();
              executeCommand(node.command);
            }}
          >
            <span className="flex-1 truncate">{node.label}</span>
            {node.shortcut && (
              <span className="text-xs text-(--ink-text-faint)">{node.shortcut}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
