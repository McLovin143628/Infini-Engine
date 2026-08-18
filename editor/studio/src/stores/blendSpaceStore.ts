/**
 * Blend-space authoring store (P29.2 clause 8).
 *
 * The panel owns a **2D blend space** — a set of clips placed on a parameter
 * plane — and can pull one out of, or push one into, a state of the `.inf_sm`
 * document `smStore` holds. That is the whole persistence story and it is
 * deliberate: a `.inf_sm` is where a blend space lives, `sm_save` is the one door
 * that writes one, and a second asset path would be a second way to author the
 * same bytes.
 *
 * Everything geometric — the triangulation, the weights, the posed preview —
 * comes from the engine (`blendspace_preview`), never from a mirror written here.
 * See `blendSpaceTypes.ts` for why that is a rule rather than convenience.
 */
import { create } from "zustand";

import { blendSpace as bsIpc } from "../lib/ipc";
import { locomotionDiamond, newSample } from "../lib/blendSpaceTypes";
import type { BlendPreviewDto, BlendSampleDto } from "../lib/blendSpaceTypes";
import { useSmStore } from "./smStore";

interface BlendSpaceStore {
  paramX: string;
  paramY: string;
  samples: BlendSampleDto[];
  /** The query point the preview is taken at, in parameter units. */
  cursorX: number;
  cursorY: number;
  /** The play-head, seconds — the phase the preview pose is sampled at. */
  timeS: number;
  preview: BlendPreviewDto | null;
  busy: boolean;
  /** The last push/pull message, for the toolbar readout. */
  status: string | null;

  init: () => Promise<void>;
  refresh: () => Promise<void>;
  setParams: (x: string, y: string) => void;
  addSample: (x: number, y: number) => Promise<void>;
  moveSample: (index: number, x: number, y: number) => Promise<void>;
  removeSample: (index: number) => Promise<void>;
  setSampleClip: (index: number, clip: string | null) => Promise<void>;
  setCursor: (x: number, y: number) => Promise<void>;
  setTime: (t: number) => Promise<void>;
  reset: () => Promise<void>;
  pullFromState: (stateIndex: number) => Promise<void>;
  pushToState: (stateIndex: number) => void;
}

export const useBlendSpaceStore = create<BlendSpaceStore>((set, get) => ({
  paramX: "vx",
  paramY: "vy",
  samples: locomotionDiamond(),
  cursorX: 0.35,
  cursorY: 0.35,
  timeS: 0,
  preview: null,
  busy: false,
  status: null,

  init: async () => {
    // The clip list is `smStore`'s — one asset query, one owner.
    await useSmStore.getState().init();
    await get().refresh();
  },

  refresh: async () => {
    const { samples, cursorX, cursorY, timeS } = get();
    set({ busy: true });
    try {
      const preview = await bsIpc.preview(samples, cursorX, cursorY, timeS);
      set({ preview });
    } catch (e) {
      console.error("blendSpace.preview failed", e);
      set({ preview: null });
    } finally {
      set({ busy: false });
    }
  },

  setParams: (x, y) => set({ paramX: x, paramY: y }),

  addSample: async (x, y) => {
    set({ samples: [...get().samples, newSample(x, y)] });
    await get().refresh();
  },

  moveSample: async (index, x, y) => {
    const samples = get().samples.map((s, i) => (i === index ? { ...s, x, y } : s));
    set({ samples });
    await get().refresh();
  },

  removeSample: async (index) => {
    set({ samples: get().samples.filter((_, i) => i !== index) });
    await get().refresh();
  },

  setSampleClip: async (index, clip) => {
    const samples = get().samples.map((s, i) => (i === index ? { ...s, clip } : s));
    set({ samples });
    await get().refresh();
  },

  setCursor: async (x, y) => {
    set({ cursorX: x, cursorY: y });
    await get().refresh();
  },

  setTime: async (t) => {
    set({ timeS: t });
    await get().refresh();
  },

  reset: async () => {
    set({ samples: locomotionDiamond(), status: null });
    await get().refresh();
  },

  // ── the `.inf_sm` seam ──

  pullFromState: async (stateIndex) => {
    const doc = useSmStore.getState().doc;
    const motion = doc?.machine.states[stateIndex]?.motion;
    if (!motion || motion.kind !== "blend2d") {
      set({ status: "that state does not play a 2D blend space" });
      return;
    }
    set({
      paramX: motion.paramX,
      paramY: motion.paramY,
      samples: motion.entries.map((e) => ({ x: e.x, y: e.y, clip: e.clip })),
      status: `pulled from “${doc?.machine.states[stateIndex]?.name ?? stateIndex}”`,
    });
    await get().refresh();
  },

  pushToState: (stateIndex) => {
    const { paramX, paramY, samples } = get();
    const ok = useSmStore.getState().setStateBlend2d(stateIndex, paramX, paramY, samples);
    set({
      status: ok
        ? `pushed into “${useSmStore.getState().doc?.machine.states[stateIndex]?.name ?? stateIndex}” — save the machine to persist it`
        : "no state machine is open",
    });
  },
}));
