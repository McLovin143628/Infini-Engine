/**
 * **Model Editor store** (P23.4) — one open `.inf_mesh` document, its preview
 * frame, and the verdicts the panel reads out.
 *
 * # The state is REPLACED, never patched
 *
 * Every mutating call returns a `DccDocDto` and this store overwrites `doc` with
 * it. That is not laziness: a modelling op moves the kernel's journal generation,
 * and ids are arena slots with a LIFO free list, so a structural op hands the
 * *same numbers* back naming different geometry. The backend owns the selection
 * for exactly that reason (it can compare stamps; the frontend cannot), and a
 * frontend that merged fields would be maintaining a second, wrong copy of it.
 *
 * # The preview is a serialized queue of ONE
 *
 * An orbit fires pointer-moves faster than a render + PNG + base64 round trip
 * completes. Without a gate they pile up, the panel lags behind the mouse by
 * however many are in flight, and the *last* frame to arrive may not be the last
 * one asked for. So: one request in flight, at most one queued behind it, and the
 * queued one always carries the latest camera — which the backend owns, so
 * "latest" is simply "whatever it is when the request runs".
 */
import { create } from "zustand";

import { registerUndoScope } from "../lib/undoScopes";

import { dcc as dccIpc, DCC_PREVIEW_SIZE } from "../lib/ipc";
import type { DccApplyDto } from "../bindings/DccApplyDto";
import type { DccDocDto } from "../bindings/DccDocDto";
import type { DccModeDto } from "../bindings/DccModeDto";
import type { DccSaveDto } from "../bindings/DccSaveDto";
import type { DccSelectDto } from "../bindings/DccSelectDto";
import type { DccToolDto } from "../bindings/DccToolDto";

interface DccState {
  doc: DccDocDto | null;
  /** The open asset id, so a re-mount of the panel re-attaches to it. */
  assetId: string | null;
  image: string | null;
  /** Why there is no image (no GPU adapter, or a shader that did not validate). */
  previewError: string | null;
  /** The last tool refusal, shown until the next action. A value, not a throw. */
  refusal: string | null;
  /** The last save's verdict — counters and advisories (the P20.4 readout). */
  lastSave: DccSaveDto | null;
  busy: boolean;
  status: string | null;

  open: (assetId: string) => Promise<void>;
  close: () => Promise<void>;
  apply: (tool: DccToolDto) => Promise<void>;
  select: (action: DccSelectDto) => Promise<void>;
  setMode: (mode: DccModeDto) => Promise<void>;
  pick: (x: number, y: number, additive: boolean) => Promise<void>;
  orbit: (yawDeg: number, pitchDeg: number, dolly: number) => Promise<void>;
  frame: () => Promise<void>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  save: () => Promise<void>;
  mergeAsset: (assetId: string) => Promise<void>;
  refresh: () => void;
}

/**
 * Preview scheduling state, deliberately outside the store: it is transport
 * bookkeeping, and putting it in zustand would re-render every subscriber on a
 * flag that describes the network rather than the document.
 */
let inFlight = false;
let queued = false;

export const useDccStore = create<DccState>((set, get) => {
  /** Render one frame, then service whatever arrived while it was running. */
  const pump = async (): Promise<void> => {
    const doc = get().doc;
    if (!doc) return;
    if (inFlight) {
      queued = true;
      return;
    }
    inFlight = true;
    try {
      const frame = await dccIpc.preview(doc.id, DCC_PREVIEW_SIZE);
      set({ image: frame.image, previewError: frame.error });
    } catch (e) {
      set({ previewError: String(e) });
    } finally {
      inFlight = false;
    }
    if (queued) {
      queued = false;
      await pump();
    }
  };

  /** Adopt a returned document and re-render. */
  const adopt = (doc: DccDocDto): void => {
    set({ doc });
    void pump();
  };

  const applyResult = (res: DccApplyDto): void => {
    set({ doc: res.doc, refusal: res.ok ? null : res.refusal });
    void pump();
  };

  return {
    doc: null,
    assetId: null,
    image: null,
    previewError: null,
    refusal: null,
    lastSave: null,
    busy: false,
    status: null,

    open: async (assetId) => {
      if (get().assetId === assetId && get().doc) return;
      set({ busy: true, refusal: null, lastSave: null, status: null });
      try {
        const doc = await dccIpc.open(assetId);
        set({ doc, assetId, image: null, previewError: null });
        await pump();
      } catch (e) {
        set({ doc: null, assetId: null, status: `Could not open: ${String(e)}` });
      } finally {
        set({ busy: false });
      }
    },

    close: async () => {
      const doc = get().doc;
      set({
        doc: null,
        assetId: null,
        image: null,
        previewError: null,
        refusal: null,
        lastSave: null,
        status: null,
      });
      queued = false;
      if (doc) {
        try {
          await dccIpc.close(doc.id);
        } catch (e) {
          console.error("dcc.close failed", e);
        }
      }
    },

    apply: async (tool) => {
      const doc = get().doc;
      if (!doc) return;
      set({ busy: true });
      try {
        applyResult(await dccIpc.apply(doc.id, tool));
      } catch (e) {
        set({ refusal: String(e) });
      } finally {
        set({ busy: false });
      }
    },

    select: async (action) => {
      const doc = get().doc;
      if (!doc) return;
      try {
        adopt(await dccIpc.select(doc.id, action));
      } catch (e) {
        set({ refusal: String(e) });
      }
    },

    setMode: async (mode) => {
      await get().select({ action: "mode", mode });
    },

    pick: async (x, y, additive) => {
      const doc = get().doc;
      if (!doc) return;
      try {
        adopt(await dccIpc.pick(doc.id, x, y, DCC_PREVIEW_SIZE, additive));
      } catch (e) {
        set({ refusal: String(e) });
      }
    },

    orbit: async (yawDeg, pitchDeg, dolly) => {
      const doc = get().doc;
      if (!doc) return;
      try {
        await dccIpc.orbit(doc.id, yawDeg, pitchDeg, dolly);
        await pump();
      } catch (e) {
        console.error("dcc.orbit failed", e);
      }
    },

    frame: async () => {
      const doc = get().doc;
      if (!doc) return;
      await dccIpc.frame(doc.id);
      await pump();
    },

    undo: async () => {
      const doc = get().doc;
      if (!doc) return;
      adopt(await dccIpc.undo(doc.id));
    },

    redo: async () => {
      const doc = get().doc;
      if (!doc) return;
      adopt(await dccIpc.redo(doc.id));
    },

    save: async () => {
      const doc = get().doc;
      if (!doc) return;
      set({ busy: true, status: null });
      try {
        const lastSave = await dccIpc.save(doc.id);
        // Re-read the document: `dirty` is derived from the generation stamp on
        // the backend, so the only honest way to learn it is to ask.
        const docs = await dccIpc.list();
        set({
          lastSave,
          doc: docs.find((d) => d.id === doc.id) ?? get().doc,
          status: `Saved ${doc.name}: ${lastSave.export.triangles} triangles, vmesh ${lastSave.vmesh}.`,
        });
      } catch (e) {
        set({ status: `Save failed: ${String(e)}` });
      } finally {
        set({ busy: false });
      }
    },

    mergeAsset: async (assetId) => {
      const doc = get().doc;
      if (!doc) return;
      set({ busy: true });
      try {
        applyResult(await dccIpc.mergeAsset(doc.id, assetId));
      } catch (e) {
        set({ refusal: String(e) });
      } finally {
        set({ busy: false });
      }
    },

    refresh: () => void pump(),
  };
});

/** Test-only: clear the module-scoped preview queue between cases. */
export function __resetDccPreviewQueueForTest(): void {
  inFlight = false;
  queued = false;
}

// **Ctrl+Z inside the Model Editor undoes the MESH** (P23.2a's routing registry).
//
// The kernel's op journal is the undo stack — `undo` truncates the cursor and
// replays from the nearest checkpoint — so this claims the scope and the shell's
// chord reaches the right document. Without it, Ctrl+Z here would undo the
// SCENE: an actor the author cannot see, moving in a panel they are not looking
// at, while the mesh in front of them does nothing.
registerUndoScope("model", {
  undo: () => useDccStore.getState().undo(),
  redo: () => useDccStore.getState().redo(),
});
