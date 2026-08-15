/**
 * **Dropping a Content Drawer asset onto a DOM panel** (round-2 finding R2.F9).
 *
 * The drawer drags with *pointer* events, not HTML5 drag-and-drop, and it has
 * to: its original target is the native viewport child window, a hole in the
 * DOM that no `dragover` will ever reach, so the cells are `draggable={false}`
 * and the drop crosses over IPC. That left every ordinary DOM drop zone
 * unreachable — the Model Editor's `dcc_merge_asset` had `onDrop` reading
 * `e.dataTransfer`, **nothing in the tree ever called `setData`**, and the zone
 * highlighted for any native drag and then silently discarded it. The DCC
 * vision's headline ("drag-and-drop modular model building married into the
 * engine") was a dead door.
 *
 * Making the cells `draggable` is not the fix: starting an HTML5 drag cancels
 * the pointer stream, and the pointer stream is the only way to reach the
 * viewport hole. So the drawer hit-tests DOM zones on pointer-up exactly as it
 * already hit-tests the hole, and delivers this event to the zone it found.
 *
 * One module so the two sides cannot drift: the attribute a zone marks itself
 * with, the event name, and the payload shape all live here.
 */

/** What a drop carries: enough to decide, and to name what arrived. */
export interface AssetDropDetail {
  id: string;
  /** The `AssetDto.kind` slug — a zone accepts or refuses on this. */
  kind: string;
  name: string;
}

/** The event a drop zone receives. */
export const ASSET_DROP_EVENT = "infinity:asset-drop";

/**
 * The attribute that makes an element a drop zone. Its VALUE is a
 * space-separated list of accepted kinds, for documentation and for a future
 * zone that wants to filter without JavaScript; the handler still decides.
 */
export const ASSET_DROP_ATTR = "data-asset-drop";

/**
 * Deliver a drop to whatever zone is under `(x, y)`; `true` if one took it.
 *
 * The drawer's half of the contract, here rather than inline in the component
 * so the two ends are one door — and so this is testable, which the inline
 * version was not. The drag ghost is `pointer-events-none`, so
 * `elementFromPoint` sees the zone rather than the cursor decoration.
 */
export function deliverAssetDrop(x: number, y: number, detail: AssetDropDetail): boolean {
  const zone = document.elementFromPoint(x, y)?.closest(`[${ASSET_DROP_ATTR}]`) ?? null;
  if (!zone) return false;
  zone.dispatchEvent(new CustomEvent<AssetDropDetail>(ASSET_DROP_EVENT, { detail, bubbles: true }));
  return true;
}

/**
 * Subscribe `el` to asset drops. Returns the disposer.
 *
 * The element must carry [`ASSET_DROP_ATTR`] or the drawer will never find it —
 * the attribute is what `elementFromPoint(...).closest(...)` looks for.
 */
export function onAssetDrop(
  el: HTMLElement,
  handler: (detail: AssetDropDetail) => void,
): () => void {
  const listener = (e: Event) => {
    const detail = (e as CustomEvent<AssetDropDetail>).detail;
    if (detail) handler(detail);
  };
  el.addEventListener(ASSET_DROP_EVENT, listener);
  return () => el.removeEventListener(ASSET_DROP_EVENT, listener);
}
