/**
 * **The native viewport's right-click menu** (Wave E, batch C).
 *
 * The 3D view is a native child window: it swallows pointer events, so a
 * right-click there is not a DOM event and no `onContextMenu` will ever see it.
 * The viewport thread discriminates the gesture from a flycam drag, resolves the
 * selection, and sends `viewport://context-menu` with the point in PHYSICAL
 * pixels relative to the hole's top-left. This component converts that back into
 * window CSS pixels and opens the shell's own menu there.
 *
 * # Airspace — read this before changing anything here
 *
 * The menu is the FIRST overlay whose anchor is inside the hole, and the shell's
 * standing rule is that HTML can never draw over the native child. Until UX2 the
 * consequence was that **the 3D view went blank the instant the menu opened** —
 * accepted behaviour since Phase 1, and the thing a right-click in a game engine
 * must not do. `ContextMenuSurface` now holds `useViewportCutout`, which measures
 * the menu and has the native child carve that rectangle out of itself
 * (`SetWindowRgn`), so the scene keeps rendering around the menu.
 *
 * The blackout is still the fallback, and deliberately: an overlay that cannot
 * measure itself (the command palette, the dialogs, the panel-drag ghost) and a
 * platform with no cutout backend (macOS, Linux) both take it.
 *
 * # Coordinates
 *
 * `ViewportDrop` sends `(clientX - holeRect.left) * dpr` in the other direction
 * (`ContentDrawer`), so this is that transform inverted:
 * `cssX = holeRect.left + x / dpr`. Read from the live element rather than a
 * cached rect — the hole moves with every splitter drag.
 */
import { useEffect, useState } from "react";

import type { EntityEditorsDto } from "../bindings/EntityEditorsDto";
import { ContextMenuSurface, type ContextMenuEntry, type MenuAnchor } from "../components/ContextMenu";
import { listenTo } from "../lib/events";
import { buildObjectMenuItems } from "../lib/objectMenu";
import { isPrimaryViewport } from "../lib/viewportIds";
import { focusViewport } from "../lib/viewportFocus";
import {
  fetchEntityEditors,
  openObject,
  openObjectEditor,
  requestRename,
} from "../stores/objectEditorCommands";
import { useSceneStore } from "../stores/sceneStore";

/** Hole-relative physical pixels → window CSS pixels. */
export function holeToClient(x: number, y: number, dpr: number, rect: DOMRect | null): MenuAnchor {
  if (!rect) return { x, y };
  const scale = dpr > 0 && Number.isFinite(dpr) ? dpr : 1;
  return { x: rect.left + x / scale, y: rect.top + y / scale };
}

export default function ViewportContextMenu() {
  const [menu, setMenu] = useState<{ at: MenuAnchor; items: ContextMenuEntry[] } | null>(null);

  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;

    const open = async (payload: {
      x: number;
      y: number;
      guid: string | null;
      selection: string[];
      viewport: string;
    }) => {
      if (!isPrimaryViewport(payload.viewport)) return;
      // The same reason the chord path does it (the B1 fix): a click on the 3D
      // view is not a DOM event, so the arrival of this message is the only
      // signal that the user is working in the viewport now.
      focusViewport();

      const targets = payload.selection.length > 0 ? payload.selection : [];
      let dtos: EntityEditorsDto[] = [];
      if (targets.length > 0) {
        try {
          dtos = await fetchEntityEditors(targets);
        } catch {
          // The backend refused — the menu still opens with its non-editor
          // actions rather than the click doing nothing at all.
        }
      }
      const hole = document.querySelector("[data-viewport-hole]");
      const at = holeToClient(
        payload.x,
        payload.y,
        window.devicePixelRatio || 1,
        hole ? hole.getBoundingClientRect() : null,
      );

      // A right-click on empty space, with nothing selected, still offers the
      // shell's own actions rather than nothing — but it has no object rows.
      const node = targets.length === 1 ? useSceneStore.getState().nodes[targets[0]] : undefined;
      setMenu({
        at,
        items: buildObjectMenuItems(
          targets,
          dtos,
          {
            open: (g) => void openObject(g),
            openRoute: (routeId, guids) => void openObjectEditor(routeId, guids),
            rename: (g) => requestRename(g),
            toggleVisible: (g) => useSceneStore.getState().toggleVisible(g),
          },
          node?.visible ?? null,
        ),
      });
    };

    listenTo("viewport://context-menu", (payload) => void open(payload)).then((fn) => {
      if (disposed) fn();
      else unlisten = fn;
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  return (
    <ContextMenuSurface
      at={menu?.at ?? null}
      items={menu?.items ?? []}
      onClose={() => setMenu(null)}
    />
  );
}
