import type { ViewportRect } from "../bindings/ViewportRect";

/**
 * CSS-pixel rectangle → PHYSICAL-pixel [`ViewportRect`] for the native
 * viewport (Spike A contract: the frontend multiplies by devicePixelRatio;
 * the backend rounds to device pixels). Pure so the DPI math is unit-tested.
 */
export function toPhysicalRect(
  rect: { x: number; y: number; width: number; height: number },
  devicePixelRatio: number,
): ViewportRect {
  return {
    x: rect.x * devicePixelRatio,
    y: rect.y * devicePixelRatio,
    width: rect.width * devicePixelRatio,
    height: rect.height * devicePixelRatio,
  };
}
