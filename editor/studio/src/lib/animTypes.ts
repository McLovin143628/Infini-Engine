/** Hand-written TS mirrors of the animation-derivation DTOs (camelCase, matching
 *  the Ring-2 `commands/anim.rs` serde) — P29.5, pillar S2.
 *
 *  These describe what the **import derived** from a clip: the curve channels it
 *  wrote, the foot-plant and footstep markers it found, and the root-motion track
 *  it baked. The panel shows them so an author can see the measurement rather
 *  than take the pipeline's word for it, and can re-run it when the answer looks
 *  wrong.
 *
 *  Like `smTypes.ts` and `blendSpaceTypes.ts` these are hand-written rather than
 *  ts-rs-generated, because the DTOs live in Ring 2 (`src-tauri`) and not in
 *  `inf_editor_core::ipc`. The Rust side carries the round-trip test that keeps
 *  the two spellings honest. */

/** One curve channel on a clip. */
export interface AnimCurveDto {
  name: string;
  /** How many authored keys it has. */
  keys: number;
  min: number;
  max: number;
  /** Uniform samples over the clip, for the sparkline. */
  samples: number[];
  /** Whether the engine derived this channel (and would replace it on a
   *  re-derive) rather than an author having written it. */
  derived: boolean;
}

/** One timed marker. A non-empty `group` makes it a sync marker; an empty one
 *  makes it an event notify. A marker can be read as both. */
export interface AnimMarkerDto {
  timeS: number;
  name: string;
  group: string;
}

/** The baked root motion, summarised. */
export interface AnimRootMotionDto {
  /** Total translation over the clip, metres, clip space (Y included). */
  translation: [number, number, number];
  /** Total turn over the clip, degrees. */
  yawDeg: number;
  /** Total ground distance, metres. */
  distanceM: number;
  keys: number;
}

/** What a clip carries. `refusal` is a **value**: a panel binds to the arrays
 *  and shows the reason beside them. */
export interface AnimClipInfoDto {
  id: string;
  name: string;
  durationS: number;
  curves: AnimCurveDto[];
  markers: AnimMarkerDto[];
  rootMotion: AnimRootMotionDto | null;
  skeleton: string | null;
  refusal: string | null;
}

/** What a re-derivation found. */
export interface AnimDeriveDto {
  distanceM: number;
  /** The speed the clip depicts, m/s -- the greater of the two below. */
  avgSpeedMps: number;
  /** What the ROOT travels. Zero for an in-place cycle, which is most authored
   *  locomotion and every clip the character wizard generates. */
  travelSpeedMps: number;
  /** `stride x cadence` -- what the FEET say. The number an in-place cycle
   *  answers with; a large gap between this and `travelSpeedMps` on a
   *  root-motion clip is foot slide the animator authored. */
  strideSpeedMps: number;
  /** How far a foot travels along the ground over one cycle, metres. */
  strideM: number;
  /** The 0-3 `W_Gait` scale: 0 stopped, 1 walk, 2 run, 3 sprint. */
  gait: number;
  /** Net rise over the clip, metres. */
  riseM: number;
  plants: number;
  markers: number;
  curves: string[];
  advisories: string[];
  refusal: string | null;
}

/** The gait tier a derived `W_Gait` value names, for a readout. */
export function gaitLabel(gait: number): string {
  if (!Number.isFinite(gait) || gait < 0.5) return "idle";
  if (gait < 1.5) return "walk";
  if (gait < 2.5) return "run";
  return "sprint";
}

/** Whether a marker is a footstep notify — the prefix
 *  `inf_anim::derive::FOOTSTEP_PREFIX` and `inf_ecs::FOOTSTEP_PREFIX` share. */
export function isFootstep(m: AnimMarkerDto): boolean {
  return m.group === "" && m.name.startsWith("footstep");
}
