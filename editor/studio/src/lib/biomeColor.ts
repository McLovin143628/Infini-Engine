/**
 * Linear RGBA ↔ CSS colour (P19.2).
 *
 * A `BiomeDefDto.color` is **linear** RGBA — it is uploaded straight into the
 * viewport's overlay palette, which writes into a linear HDR target. A CSS colour
 * (and `<input type="color">`) is **sRGB**. Converting with the real sRGB transfer
 * function, not a bare ×255, is what makes a toolbar swatch and the biome the
 * viewport tints actually the same colour.
 *
 * Both directions are pure and clamp their inputs, so a hand-edited or
 * out-of-gamut payload renders as the nearest displayable colour rather than as
 * `#NaNNaNNaN`.
 */

/** Clamp to `[0, 1]`; a non-finite input reads as `0`. */
function unit(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.min(Math.max(v, 0), 1);
}

/** The sRGB OETF (linear → encoded), per IEC 61966-2-1. */
function encodeSrgb(linear: number): number {
  const c = unit(linear);
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** The inverse (encoded → linear). */
function decodeSrgb(encoded: number): number {
  const c = unit(encoded);
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/**
 * Linear RGBA → `#rrggbb`. Alpha is dropped: `<input type="color">` cannot show
 * it, and a biome swatch is an identity marker rather than a blend.
 */
export function linearToCssHex(color: readonly number[]): string {
  const channel = (v: number) =>
    Math.round(encodeSrgb(v ?? 0) * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${channel(color[0] ?? 0)}${channel(color[1] ?? 0)}${channel(color[2] ?? 0)}`;
}

/**
 * `#rrggbb` → linear RGBA. `alpha` defaults to 1 (opaque) so a colour picked in
 * the editor round-trips through {@link linearToCssHex} unchanged to display
 * precision. A malformed string reads as opaque black rather than throwing —
 * this sits on an input event, and a half-typed value must not blow up a render.
 */
export function cssHexToLinear(hex: string, alpha = 1): [number, number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  const n = m ? parseInt(m[1], 16) : 0;
  return [
    decodeSrgb(((n >> 16) & 255) / 255),
    decodeSrgb(((n >> 8) & 255) / 255),
    decodeSrgb((n & 255) / 255),
    unit(alpha),
  ];
}
