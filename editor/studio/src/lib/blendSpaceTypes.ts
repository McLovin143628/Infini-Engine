/** Hand-written TS mirrors of the blend-space authoring DTOs (P29.2 clause 8),
 *  camelCase to match `commands/blendspace.rs`'s serde.
 *
 *  ## Why the geometry crosses the wire instead of being computed here
 *
 *  A 2D blend space's weights are **Delaunay barycentric** since P29.2
 *  (`inf_anim::delaunay`), and those weights are sim state: they choose the pose,
 *  and the pose is folded into the trace the PIE-==-shipping gate compares. A
 *  second triangulator written in TypeScript would be a second implementation of
 *  a deterministic algorithm, free to disagree with the engine's about a
 *  co-circular square — and the panel would then draw a blend the game does not
 *  play. So the canvas asks the engine for the triangles, the weights and the
 *  posed skeleton, and draws exactly what it is told. */

/** One placed sample: a clip pinned at a point of the parameter plane. */
export interface BlendSampleDto {
  x: number;
  y: number;
  /** Asset GUID of the `.inf_anim` clip, or `null` while unassigned. */
  clip: string | null;
}

/** One sample's share of the blend at the query point. */
export interface BlendWeightDto {
  index: number;
  weight: number;
}

/** One joint of the previewed pose, in the skeleton's model space (metres). */
export interface BlendJointDto {
  name: string;
  parent: number | null;
  position: [number, number, number];
}

/** Everything the canvas draws, in one round trip. */
export interface BlendPreviewDto {
  /** Triangles as index triples into the sample list. Empty when the samples
   *  have no area (fewer than three, all coincident, or a collinear row) — the
   *  engine then interpolates along the row, and `degenerate` says so. */
  triangles: [number, number, number][];
  /** The boundary edges of the triangulation, for the hull outline. */
  hull: [number, number][];
  /** True when there is no triangulation: the panel draws the chain instead. */
  degenerate: boolean;
  /** The resolved weights at the query point, summing to 1. */
  weights: BlendWeightDto[];
  /** The weight-blended cycle length in seconds — what the play-head is
   *  normalised against, and what an `exitTime` is a fraction of. */
  blendedDurationS: number;
  /** The blended pose, or empty when no clip resolves to a skeleton. */
  joints: BlendJointDto[];
  /** Asset GUID of the skeleton the preview was posed against, if one resolved. */
  skeleton: string | null;
  /** A human-readable reason the preview is empty, or `null` when it is not. */
  refusal: string | null;
}

/** A fresh sample at a point, with no clip yet. */
export function newSample(x: number, y: number): BlendSampleDto {
  return { x, y, clip: null };
}

/** The four-direction locomotion diamond plus an idle at the centre — the shape
 *  a wizard-generated 2D space has, and a useful starting point.
 *
 *  The centre sample is not decoration: the four outer points alone are exactly
 *  **co-circular**, which is the one configuration where a triangulation must
 *  span the middle with one of two arbitrary diagonals. With a centre the fan is
 *  well-formed and a query in one quadrant blends only that quadrant. */
export function locomotionDiamond(): BlendSampleDto[] {
  return [
    newSample(0, 0),
    newSample(0, 1),
    newSample(1, 0),
    newSample(0, -1),
    newSample(-1, 0),
  ];
}
