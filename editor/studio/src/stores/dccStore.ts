/**
 * **Model Editor store** (P23.4) — one entry per open `.inf_mesh` asset.
 *
 * # Why this is keyed, and what it was before
 *
 * The Model Editor panel is registered `singleton: false`: opening two meshes
 * mints `model:<assetA>` and `model:<assetB>`, and the dock keeps **both tabs
 * mounted** (switching a tab changes visibility, not mounting). This store used
 * to hold a single `doc`, so the two panels shared it: both tabs rendered
 * whichever mesh was opened last, every tool press in A edited B, A's backend
 * session leaked for the life of the process, and closing A read the shared
 * `doc` and closed **B** — blanking a document with unsaved work in it.
 *
 * The backend was multi-document from the first commit (`docs: BTreeMap<Id, Doc>`
 * with a journal per id). This was purely the frontend collapsing it, which is
 * why the fix is a `Record` and not a redesign.
 *
 * # The state is REPLACED, never patched
 *
 * Every mutating call returns a `DccDocDto` and the entry is overwritten with it.
 * A modelling op moves the kernel's journal generation, and ids are arena slots
 * with a LIFO free list, so a structural op hands the *same numbers* back naming
 * different geometry. The backend owns the selection for exactly that reason (it
 * can compare stamps; the frontend cannot), and a store that merged fields would
 * be maintaining a second, wrong copy of it.
 *
 * # The preview is a serialized queue of one, PER DOCUMENT
 *
 * An orbit fires pointer-moves faster than a render + PNG + base64 round trip
 * completes. Without a gate they pile up, the panel lags the mouse by however
 * many are in flight, and the last frame to arrive may not be the last one asked
 * for. One request in flight per document, at most one queued behind it — *per
 * document*, because a single global gate would make two open panels take turns.
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

/** One open document's view state. */
export interface DccEntry {
  doc: DccDocDto | null;
  image: string | null;
  /** Why there is no image (no GPU adapter, or a shader that did not validate). */
  previewError: string | null;
  /** The last tool refusal, cleared by the next successful action. A value, not a throw. */
  refusal: string | null;
  /**
   * The last save's verdict — counters and advisories (the P20.4 readout).
   *
   * **Kept until the NEXT SAVE**, not until the next action. Its coincidence
   * warning is the one sentence in the product that says "what is on disk is not
   * what is in this session", and clearing it on the next click meant an author
   * who pressed Save and then moved the mouse never read it.
   */
  lastSave: DccSaveDto | null;
  busy: boolean;
  status: string | null;
}

interface DccState {
  /** Keyed by **asset id** — the same key the panel instance is parameterized by. */
  docs: Record<string, DccEntry>;

  open: (assetId: string) => Promise<void>;
  close: (assetId: string) => Promise<void>;
  apply: (assetId: string, tool: DccToolDto) => Promise<void>;
  select: (assetId: string, action: DccSelectDto) => Promise<void>;
  setMode: (assetId: string, mode: DccModeDto) => Promise<void>;
  pick: (assetId: string, x: number, y: number, additive: boolean) => Promise<void>;
  orbit: (assetId: string, yawDeg: number, pitchDeg: number, dolly: number) => Promise<void>;
  frame: (assetId: string) => Promise<void>;
  undo: (assetId: string) => Promise<void>;
  redo: (assetId: string) => Promise<void>;
  save: (assetId: string) => Promise<void>;
  mergeAsset: (assetId: string, dropped: string) => Promise<void>;
  refresh: (assetId: string) => void;
}

const EMPTY: DccEntry = {
  doc: null,
  image: null,
  previewError: null,
  refusal: null,
  lastSave: null,
  busy: false,
  status: null,
};

/**
 * Per-document preview scheduling, deliberately outside the store: it is
 * transport bookkeeping, and putting it in zustand would re-render every
 * subscriber on a flag that describes the network rather than the document.
 */
const flight = new Map<string, { inFlight: boolean; queued: boolean }>();

function gate(assetId: string): { inFlight: boolean; queued: boolean } {
  let g = flight.get(assetId);
  if (!g) {
    g = { inFlight: false, queued: false };
    flight.set(assetId, g);
  }
  return g;
}

export const useDccStore = create<DccState>((set, get) => {
  const entry = (assetId: string): DccEntry => get().docs[assetId] ?? EMPTY;

  const patch = (assetId: string, next: Partial<DccEntry>): void =>
    set((s) => ({ docs: { ...s.docs, [assetId]: { ...(s.docs[assetId] ?? EMPTY), ...next } } }));

  /** Render one frame for `assetId`, then service whatever arrived meanwhile. */
  const pump = async (assetId: string): Promise<void> => {
    const doc = entry(assetId).doc;
    if (!doc) return;
    const g = gate(assetId);
    if (g.inFlight) {
      g.queued = true;
      return;
    }
    g.inFlight = true;
    try {
      const frame = await dccIpc.preview(doc.id, DCC_PREVIEW_SIZE);
      patch(assetId, { image: frame.image, previewError: frame.error });
    } catch (e) {
      patch(assetId, { previewError: String(e) });
    } finally {
      g.inFlight = false;
    }
    if (g.queued) {
      g.queued = false;
      await pump(assetId);
    }
  };

  const adopt = (assetId: string, doc: DccDocDto): void => {
    patch(assetId, { doc });
    void pump(assetId);
  };

  const applyResult = (assetId: string, res: DccApplyDto): void => {
    patch(assetId, { doc: res.doc, refusal: res.ok ? null : res.refusal });
    void pump(assetId);
  };

  return {
    docs: {},

    open: async (assetId) => {
      if (entry(assetId).doc) return;
      patch(assetId, { busy: true, refusal: null, status: null });
      try {
        const doc = await dccIpc.open(assetId);
        patch(assetId, { doc, image: null, previewError: null });
        await pump(assetId);
      } catch (e) {
        patch(assetId, { doc: null, status: `Could not open: ${String(e)}` });
      } finally {
        patch(assetId, { busy: false });
      }
    },

    close: async (assetId) => {
      const doc = entry(assetId).doc;
      flight.delete(assetId);
      set((s) => {
        const docs = { ...s.docs };
        delete docs[assetId];
        return { docs };
      });
      if (doc) {
        try {
          await dccIpc.close(doc.id);
        } catch (e) {
          console.error("dcc.close failed", e);
        }
      }
    },

    apply: async (assetId, tool) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      patch(assetId, { busy: true });
      try {
        applyResult(assetId, await dccIpc.apply(doc.id, tool));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      } finally {
        patch(assetId, { busy: false });
      }
    },

    select: async (assetId, action) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.select(doc.id, action));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    setMode: async (assetId, mode) => {
      await get().select(assetId, { action: "mode", mode });
    },

    pick: async (assetId, x, y, additive) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.pick(doc.id, x, y, DCC_PREVIEW_SIZE, additive));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    orbit: async (assetId, yawDeg, pitchDeg, dolly) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        await dccIpc.orbit(doc.id, yawDeg, pitchDeg, dolly);
        await pump(assetId);
      } catch (e) {
        console.error("dcc.orbit failed", e);
      }
    },

    frame: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      await dccIpc.frame(doc.id);
      await pump(assetId);
    },

    undo: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      adopt(assetId, await dccIpc.undo(doc.id));
    },

    redo: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      adopt(assetId, await dccIpc.redo(doc.id));
    },

    save: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      patch(assetId, { busy: true, status: null });
      try {
        const lastSave = await dccIpc.save(doc.id);
        // Re-read the document: `dirty` is derived from the generation stamp on
        // the backend, so the only honest way to learn it is to ask.
        const docs = await dccIpc.list();
        patch(assetId, {
          lastSave,
          doc: docs.find((d) => d.id === doc.id) ?? entry(assetId).doc,
          status: `Saved ${doc.name}: ${lastSave.export.triangles} triangles, vmesh ${lastSave.vmesh}.`,
        });
      } catch (e) {
        patch(assetId, { status: `Save failed: ${String(e)}` });
      } finally {
        patch(assetId, { busy: false });
      }
    },

    mergeAsset: async (assetId, dropped) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      patch(assetId, { busy: true });
      try {
        applyResult(assetId, await dccIpc.mergeAsset(doc.id, dropped));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      } finally {
        patch(assetId, { busy: false });
      }
    },

    refresh: (assetId) => void pump(assetId),
  };
});

/** One document's view state, for a panel that knows its own asset id. */
export function useDccEntry(assetId: string | null): DccEntry {
  return useDccStore((s) => (assetId ? (s.docs[assetId] ?? EMPTY) : EMPTY));
}

/** Test-only: clear the per-document preview queues between cases. */
export function __resetDccPreviewQueueForTest(): void {
  flight.clear();
}

// **Ctrl+Z inside the Model Editor undoes the MESH, in the panel you are looking
// at** (P23.2a's routing registry; the aim is P23.4's fix).
//
// The scope is handed the focused panel's `params`, which for `model:<assetId>`
// is the asset id — so with two Model Editors open the chord reaches the document
// under the cursor rather than whichever one happened to be opened last. The
// kernel's op journal IS the undo stack (`undo` truncates the cursor and replays
// from the nearest checkpoint); without this registration Ctrl+Z here would undo
// the SCENE: an actor the author cannot see, moving in a panel they are not
// looking at, while the mesh in front of them does nothing.
registerUndoScope("model", {
  undo: (params) => {
    if (params) void useDccStore.getState().undo(params);
  },
  redo: (params) => {
    if (params) void useDccStore.getState().redo(params);
  },
});
