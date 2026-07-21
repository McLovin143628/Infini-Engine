/**
 * Sequencer store (P11.4).
 *
 * Mirrors the project-local sequence files (`.infinity/sequences/*.toml`) the
 * backend owns, and drives the timeline panel: pick/create/save/delete a
 * sequence, move the playhead (scrub), capture/retime/delete keys, and Play (a
 * client-side rAF advancing the playhead). Scrubbing is a **non-undoable live
 * preview** on the backend — it never dirties the scene and is refused while
 * Simulate runs (this store also guards locally via `useSimStore`).
 *
 * The pure helpers (`sampleTrackAt`, `snapTime`, `retimeKeys`, `advancePlayhead`)
 * mirror the Rust sampling/retime semantics and are unit-tested; the canvas uses
 * them to draw the interpolated preview and to round key drags.
 */
import { create } from "zustand";

import type { SeqInterpDto } from "../bindings/SeqInterpDto";
import type { SeqKeyDto } from "../bindings/SeqKeyDto";
import type { SeqTrackDto } from "../bindings/SeqTrackDto";
import type { SequenceDto } from "../bindings/SequenceDto";
import { sequencer as seqIpc } from "../lib/ipc";
import { useSimStore } from "./simStore";

// ── pure helpers (unit-tested; also used by the canvas) ─────────────────────

/**
 * Sample one track at time `t`, holding at the endpoints — the exact mirror of
 * `inf_editor_core::sequencer::PropertyTrack::sample`. `null` when the track has
 * no keys. The LEFT key's `interp` governs its segment.
 */
export function sampleTrackAt(track: SeqTrackDto, t: number): number | null {
  const keys = track.keys;
  if (keys.length === 0) return null;
  const first = keys[0];
  const last = keys[keys.length - 1];
  if (t <= first.t) return first.value;
  if (t >= last.t) return last.value;
  for (let i = 0; i < keys.length - 1; i++) {
    const k0 = keys[i];
    const k1 = keys[i + 1];
    if (t >= k0.t && t < k1.t) {
      if (k0.interp === "step") return k0.value;
      const span = k1.t - k0.t;
      if (span <= 0) return k0.value;
      const f = (t - k0.t) / span;
      return k0.value + f * (k1.value - k0.value);
    }
  }
  return last.value;
}

/** Round `t` to the nearest frame on the `1 / fps` grid (`fps <= 0` ⇒ no grid). */
export function snapTime(fps: number, t: number): number {
  if (fps <= 0) return t;
  const step = 1 / fps;
  return Math.round(t / step) * step;
}

/**
 * Retime the key at `index` to `newT` (snapped to the fps grid, clamped ≥ 0),
 * re-sorting and dropping any other key that lands on the same snapped time.
 * Pure — returns a new key array.
 */
export function retimeKeys(
  keys: SeqKeyDto[],
  index: number,
  newT: number,
  fps: number,
): SeqKeyDto[] {
  if (index < 0 || index >= keys.length) return keys;
  const snapped = Math.max(0, snapTime(fps, newT));
  const moved: SeqKeyDto = { ...keys[index], t: snapped };
  const rest = keys.filter((k, i) => i !== index && k.t !== snapped);
  return [...rest, moved].sort((a, b) => a.t - b.t);
}

/**
 * Advance the playhead by `dt` for Play, clamping at `duration`. `done` signals
 * the end was reached (Play stops; no looping in v1).
 */
export function advancePlayhead(
  t: number,
  dt: number,
  duration: number,
): { t: number; done: boolean } {
  const next = t + dt;
  if (next >= duration) return { t: duration, done: true };
  return { t: next, done: false };
}

/** Throttle scrub IPC to ~30 Hz so a fast drag / rAF doesn't flood the backend. */
const SCRUB_MIN_INTERVAL_MS = 1000 / 30;
let lastScrubAt = 0;

// ── store ────────────────────────────────────────────────────────────────────

interface SequencerState {
  /** Known sequence names (sorted). */
  sequences: string[];
  /** The open sequence's name, or null. */
  current: string | null;
  /** The open sequence document, or null. */
  doc: SequenceDto | null;
  /** Playhead time in seconds. */
  playhead: number;
  /** A scrub gesture (drag or Play) is live — the backend holds a restore snapshot. */
  scrubbing: boolean;
  /** Play (client rAF) is advancing the playhead. */
  playing: boolean;
  /** Selected key `[trackIndex, keyIndex]`, or null. */
  selectedKey: [number, number] | null;
  error: string | null;

  refresh: () => Promise<void>;
  open: (name: string) => Promise<void>;
  create: (name: string) => Promise<void>;
  remove: (name: string) => Promise<void>;
  setDuration: (duration: number) => Promise<void>;

  /** Move the playhead and preview it (throttled, guarded against Simulate). */
  scrubTo: (t: number, opts?: { force?: boolean }) => Promise<void>;
  /** End a scrub/Play gesture: restore the pre-scrub scene. */
  endScrub: () => Promise<void>;
  setPlaying: (playing: boolean) => void;

  selectKey: (sel: [number, number] | null) => void;
  captureKey: (target: string, path: string) => Promise<void>;
  /** Change an existing key's interpolation (re-sets it at its own time/value). */
  setKeyInterp: (trackIndex: number, keyIndex: number, interp: SeqInterpDto) => Promise<void>;
  deleteKey: (trackIndex: number, keyIndex: number) => Promise<void>;
  retimeKey: (trackIndex: number, keyIndex: number, newT: number) => Promise<void>;
}

/** Guard: scrubbing must never fight the Simulate loop over the shared world. */
function simBlocks(): boolean {
  return useSimStore.getState().running;
}

export const useSequencerStore = create<SequencerState>((set, get) => ({
  sequences: [],
  current: null,
  doc: null,
  playhead: 0,
  scrubbing: false,
  playing: false,
  selectedKey: null,
  error: null,

  refresh: async () => {
    try {
      const sequences = await seqIpc.list();
      set({ sequences });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  open: async (name) => {
    try {
      const doc = await seqIpc.get(name);
      set({ current: name, doc, playhead: 0, selectedKey: null, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  create: async (name) => {
    const trimmed = name.trim();
    if (!trimmed) return;
    try {
      const doc = await seqIpc.save({ name: trimmed, duration: 5, fps_hint: 30, tracks: [] });
      await get().refresh();
      set({ current: doc.name, doc, playhead: 0, selectedKey: null, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  remove: async (name) => {
    try {
      await seqIpc.delete(name);
      await get().refresh();
      if (get().current === name) set({ current: null, doc: null, selectedKey: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  setDuration: async (duration) => {
    const doc = get().doc;
    if (!doc) return;
    const next = { ...doc, duration: Math.max(0.1, duration) };
    set({ doc: next });
    try {
      const saved = await seqIpc.save(next);
      set({ doc: saved });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  scrubTo: async (t, opts) => {
    const { current, doc } = get();
    const clamped = Math.max(0, Math.min(t, doc?.duration ?? t));
    set({ playhead: clamped });
    if (!current || simBlocks()) return;
    const now = Date.now();
    if (!opts?.force && now - lastScrubAt < SCRUB_MIN_INTERVAL_MS) return;
    lastScrubAt = now;
    try {
      set({ scrubbing: true });
      await seqIpc.scrub(current, clamped);
    } catch (e) {
      // A refusal (e.g. Simulate started mid-drag) is non-fatal; stop previewing.
      set({ error: String(e), scrubbing: false });
    }
  },

  endScrub: async () => {
    if (!get().scrubbing && !get().playing) return;
    set({ scrubbing: false, playing: false });
    try {
      await seqIpc.scrubStop();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  setPlaying: (playing) => set({ playing }),

  selectKey: (selectedKey) => set({ selectedKey }),

  captureKey: async (target, path) => {
    const { current, playhead } = get();
    if (!current) return;
    try {
      const doc = await seqIpc.captureKey(current, target, path, playhead);
      set({ doc });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  setKeyInterp: async (trackIndex, keyIndex, interp) => {
    const { current, doc } = get();
    if (!current || !doc) return;
    const track = doc.tracks[trackIndex];
    const k = track?.keys[keyIndex];
    if (!track || !k) return;
    try {
      const next = await seqIpc.keySet(current, track.target, track.path, k.t, k.value, interp);
      set({ doc: next });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  deleteKey: async (trackIndex, keyIndex) => {
    const doc = get().doc;
    if (!doc) return;
    const track = doc.tracks[trackIndex];
    if (!track) return;
    const keys = track.keys.filter((_, i) => i !== keyIndex);
    const tracks = doc.tracks
      .map((tr, i) => (i === trackIndex ? { ...tr, keys } : tr))
      // A track with no keys is dropped (matches backend normalize).
      .filter((tr) => tr.keys.length > 0);
    const next = { ...doc, tracks };
    set({ doc: next, selectedKey: null });
    try {
      const saved = await seqIpc.save(next);
      set({ doc: saved });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  retimeKey: async (trackIndex, keyIndex, newT) => {
    const doc = get().doc;
    if (!doc) return;
    const track = doc.tracks[trackIndex];
    if (!track) return;
    const keys = retimeKeys(track.keys, keyIndex, newT, doc.fps_hint);
    const tracks = doc.tracks.map((tr, i) => (i === trackIndex ? { ...tr, keys } : tr));
    const next = { ...doc, tracks };
    set({ doc: next });
    try {
      const saved = await seqIpc.save(next);
      set({ doc: saved });
    } catch (e) {
      set({ error: String(e) });
    }
  },
}));
