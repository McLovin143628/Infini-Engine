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
import type { DccDragDto } from "../bindings/DccDragDto";
import type { DccGarmentDto } from "../bindings/DccGarmentDto";
import type { DccGizmoModeDto } from "../bindings/DccGizmoModeDto";
import type { DccGroomDto } from "../bindings/DccGroomDto";
import type { DccGroomResultDto } from "../bindings/DccGroomResultDto";
import type { DccHistoryEntryDto } from "../bindings/DccHistoryEntryDto";
import type { DccModeDto } from "../bindings/DccModeDto";
import type { DccNewDto } from "../bindings/DccNewDto";
import type { DccOrientDto } from "../bindings/DccOrientDto";
import type { DccPivotDto } from "../bindings/DccPivotDto";
import type { DccSaveDto } from "../bindings/DccSaveDto";
import type { DccSelectDto } from "../bindings/DccSelectDto";
import type { DccToolDto } from "../bindings/DccToolDto";
import type { DccUnwrapDto } from "../bindings/DccUnwrapDto";

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
  /** The 2D UV view's last frame, or `null` before it was ever asked for. */
  uvImage: string | null;
  /** The last unwrap's verdict, kept until the NEXT unwrap — like `lastSave`. */
  lastUnwrap: DccUnwrapDto | null;
  /**
   * The last cloth/hair authoring verdict (P24.4), kept until the NEXT one —
   * `lastSave`'s rule, for `lastSave`'s reason: it names a file that was written
   * outside this document, and an author who pressed the button and moved the
   * mouse would otherwise never learn where it went.
   */
  lastGroom: DccGroomResultDto | null;
  /**
   * **The journal, as rows** (Wave D) — refreshed on demand rather than with
   * every document, because a forty-op history is forty rows the panel does not
   * draw until an author opens the list.
   */
  history: DccHistoryEntryDto[];
}

interface DccState {
  /** Keyed by **asset id** — the same key the panel instance is parameterized by. */
  docs: Record<string, DccEntry>;

  open: (assetId: string) => Promise<void>;
  /**
   * **Start a new model** (Wave D). Mints the asset, opens a session on it and
   * returns the new asset's GUID so the caller can route the panel to it.
   */
  newMesh: (spec: DccNewDto) => Promise<string | null>;
  close: (assetId: string) => Promise<void>;
  apply: (assetId: string, tool: DccToolDto) => Promise<void>;
  select: (assetId: string, action: DccSelectDto) => Promise<void>;
  setMode: (assetId: string, mode: DccModeDto) => Promise<void>;
  pick: (assetId: string, x: number, y: number, additive: boolean) => Promise<void>;
  /**
   * **Marquee select** (Wave D). Pixel coordinates in the preview's own space,
   * the same space `pick` takes.
   */
  boxSelect: (
    assetId: string,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    additive: boolean,
  ) => Promise<void>;
  /** Set the gizmo pivot / orientation / x-ray. Only what changed is sent. */
  setViewOpts: (
    assetId: string,
    opts: { pivot?: DccPivotDto; orient?: DccOrientDto; xray?: boolean },
  ) => Promise<void>;
  orbit: (assetId: string, yawDeg: number, pitchDeg: number, dolly: number) => Promise<void>;
  frame: (assetId: string) => Promise<void>;
  undo: (assetId: string) => Promise<void>;
  /** Refresh the history rows for a document. */
  historyRefresh: (assetId: string) => Promise<void>;
  /**
   * **Re-parameterize an edit that is already in the past.** A refusal lands in
   * `refusal`, exactly like a tool press.
   */
  amend: (assetId: string, index: number, value: number) => Promise<void>;
  redo: (assetId: string) => Promise<void>;
  save: (assetId: string) => Promise<void>;
  mergeAsset: (assetId: string, dropped: string) => Promise<void>;
  /** P24.4: mint a `.inf_cloth` from this mesh; the vertex selection is pinned. */
  makeGarment: (assetId: string, spec: DccGarmentDto) => Promise<void>;
  /** P24.4: grow guides out of the face selection into a `.inf_hair`. */
  growHair: (assetId: string, spec: DccGroomDto) => Promise<void>;
  refresh: (assetId: string) => void;
  /**
   * **Move the hover brush ring** (Wave D). `null` when the pointer leaves the
   * image or the tool is not a brush; a re-render follows only when the value
   * really changed, so a hover over a non-brush tool costs nothing.
   */
  setHover: (assetId: string, hover: [number, number, number] | null) => void;

  // ── pointer drags (P23.5) ───────────────────────────────────────────────
  setGizmo: (assetId: string, mode: DccGizmoModeDto | null) => Promise<void>;
  /** Returns whether the pointer actually grabbed something. */
  dragBegin: (assetId: string, drag: DccDragDto, x: number, y: number) => Promise<boolean>;
  dragMove: (assetId: string, x: number, y: number) => Promise<void>;
  dragEnd: (assetId: string) => Promise<void>;
  dragCancel: (assetId: string) => Promise<void>;

  // ── UV (P23.5) ──────────────────────────────────────────────────────────
  unwrap: (assetId: string) => Promise<void>;
  /** Render one UV frame. Called when the view is open, after every edit. */
  uvRefresh: (assetId: string) => Promise<void>;
  /** **Pick in the UV pane** (Wave D) — changes the shared selection. */
  uvPick: (assetId: string, x: number, y: number, additive: boolean) => Promise<void>;
  /** **Drag the selection in UV space** — one journal entry per gesture. */
  uvMove: (assetId: string, dx: number, dy: number) => Promise<void>;
}

const EMPTY: DccEntry = {
  doc: null,
  image: null,
  previewError: null,
  refusal: null,
  lastSave: null,
  busy: false,
  status: null,
  uvImage: null,
  lastUnwrap: null,
  lastGroom: null,
  history: [],
};

/**
 * Per-document preview scheduling, deliberately outside the store: it is
 * transport bookkeeping, and putting it in zustand would re-render every
 * subscriber on a flag that describes the network rather than the document.
 */
const flight = new Map<string, { inFlight: boolean; queued: boolean }>();

/**
 * Assets whose panel has closed — **the tombstones**.
 *
 * Every backend call is async, so between `open` being called and its reply
 * arriving the panel can unmount (React StrictMode does exactly this on every
 * dev mount). Without a tombstone that window leaked twice over: `close` read
 * `entry(assetId).doc`, found `null` because the open had not resolved, and
 * never sent `dcc_close` at all — so the backend `DccDoc`, its `MeshSession` and
 * its journal lived for the process — and then the open resolved and `patch` put
 * the entry straight back, because it created one unconditionally. `docs` never
 * shrank, and a remount could adopt a session an in-flight close was about to
 * destroy.
 *
 * The rule: a tombstoned asset accepts no state, and its `open` reply is answered
 * with another `dcc_close` — because the session it just created is exactly the
 * one the first close could not name.
 */
const tombstones = new Set<string>();

/**
 * **Where a brush is hovering, per document** (Wave D) — outside the store for
 * the reason `flight` is: it changes on every pointer-move, and putting that in
 * zustand would re-render the whole panel per move to draw a circle the backend
 * composites into the frame anyway.
 */
const hovers = new Map<string, [number, number, number] | null>();

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

  /**
   * Update an existing entry. **Never creates one** — a late reply for a
   * document that has been closed is dropped, not resurrected. Only `open`
   * creates, and only after clearing the tombstone.
   */
  const patch = (assetId: string, next: Partial<DccEntry>): void =>
    set((s) => {
      const cur = s.docs[assetId];
      if (!cur) return s;
      return { docs: { ...s.docs, [assetId]: { ...cur, ...next } } };
    });

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
      const frame = await dccIpc.preview(doc.id, DCC_PREVIEW_SIZE, hovers.get(assetId) ?? null);
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

  /**
   * Adopt a returned document and re-render whatever is on screen.
   *
   * **Both pictures, not just the 3D one.** The UV pane used to refresh only from
   * a `useEffect` keyed on `doc.generation` — the JOURNAL stamp — and a selection
   * change does not move it, so picking a different face left the UV view showing
   * the previous selection. `doc.selected` could not have fixed it either: it is a
   * COUNT, so face A and face B both read `1`. The coupling belongs here, where
   * one place decides "the picture is stale", and it is keyed on `selectionRev`,
   * a content revision of the set itself.
   *
   * The UV frame is only asked for once one has been asked for before: a document
   * whose UV pane was never opened does not pay for it.
   */
  const adopt = (assetId: string, doc: DccDocDto): void => {
    const before = entry(assetId);
    patch(assetId, { doc });
    void pump(assetId);
    const moved =
      before.doc?.selectionRev !== doc.selectionRev || before.doc?.generation !== doc.generation;
    if (before.uvImage && moved) void get().uvRefresh(assetId);
  };

  const applyResult = (assetId: string, res: DccApplyDto): void => {
    patch(assetId, { refusal: res.ok ? null : res.refusal });
    adopt(assetId, res.doc);
  };

  return {
    docs: {},

    open: async (assetId) => {
      if (entry(assetId).doc) return;
      tombstones.delete(assetId);
      // The ONE place an entry is created.
      set((s) => ({
        docs: { ...s.docs, [assetId]: { ...EMPTY, busy: true } },
      }));
      try {
        const doc = await dccIpc.open(assetId);
        if (tombstones.has(assetId)) {
          // Closed while this was in flight. The backend session exists NOW, and
          // the `dcc_close` the close already sent found nothing to free — so it
          // is sent again, and no state is adopted.
          await dccIpc.close(assetId).catch(() => {});
          return;
        }
        patch(assetId, { doc, image: null, previewError: null });
        await pump(assetId);
      } catch (e) {
        patch(assetId, { doc: null, status: `Could not open: ${String(e)}` });
      } finally {
        patch(assetId, { busy: false });
      }
    },

    newMesh: async (spec) => {
      // The backend mints the asset AND opens the session, so this creates the
      // store entry from the document it gets back rather than calling `open`
      // afterwards — a second `dcc_open` would be idempotent but would also
      // mean two round trips for one press, and a window in which the panel has
      // an asset id and no document.
      try {
        const doc = await dccIpc.new(spec);
        set((s) => ({
          docs: { ...s.docs, [doc.assetId]: { ...EMPTY, doc } },
        }));
        tombstones.delete(doc.assetId);
        await pump(doc.assetId);
        return doc.assetId;
      } catch (e) {
        console.error("dcc.new failed", e);
        return null;
      }
    },

    close: async (assetId) => {
      tombstones.add(assetId);
      flight.delete(assetId);
      // The hover belongs to the panel that is closing — an entry left behind
      // would be read by the next document to take this asset id.
      hovers.delete(assetId);
      set((s) => {
        const docs = { ...s.docs };
        delete docs[assetId];
        return { docs };
      });
      // **Sent unconditionally**, not "if we have a document". A panel that
      // unmounts before its open resolves has no document id, and the session it
      // is about to be handed is exactly the one that would leak. `dcc_close`
      // takes the ASSET id and is idempotent for that reason.
      try {
        await dccIpc.close(assetId);
      } catch (e) {
        console.error("dcc.close failed", e);
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

    boxSelect: async (assetId, x0, y0, x1, y1, additive) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.boxSelect(doc.id, x0, y0, x1, y1, DCC_PREVIEW_SIZE, additive));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    setViewOpts: async (assetId, opts) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.setViewOpts(doc.id, opts));
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

    // Found alongside L7.M4: the last bare `await` in this store, and its only
    // caller is `void frame(assetId)` on the Model Editor toolbar — so a refused
    // `dcc_frame` was an unhandled promise rejection with nothing shown. Framing
    // is a camera move, so like `orbit` it reports to the console rather than
    // occupying the refusal slot an author reads for tool feedback.
    frame: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        await dccIpc.frame(doc.id);
        await pump(assetId);
      } catch (e) {
        console.error("dcc.frame failed", e);
      }
    },

    // Undo/redo surface a rejection as a `refusal`, like every other action in
    // this store (F-lens L7.M4). They were the two bare `await`s left — and
    // their callers are `void undo(assetId)` (the toolbar) and
    // `void useDccStore.getState().undo(params)` (the `"model"` undo scope), so
    // a rejected `dcc_undo` was an unhandled promise rejection and the panel
    // showed nothing at all. `skelStore.run` already does exactly this, for the
    // reason written above it.
    setHover: (assetId, hover) => {
      const prev = hovers.get(assetId) ?? null;
      const same =
        prev === hover ||
        (prev !== null &&
          hover !== null &&
          prev[0] === hover[0] &&
          prev[1] === hover[1] &&
          prev[2] === hover[2]);
      if (same) return;
      hovers.set(assetId, hover);
      // The gate collapses the storm: a pointer-move during a frame queues one
      // and no more, which is the same rule an orbit already lives under.
      void pump(assetId);
    },

    uvPick: async (assetId, x, y, additive) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.uvPick(doc.id, x, y, DCC_PREVIEW_SIZE, additive));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    uvMove: async (assetId, dx, dy) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        applyResult(assetId, await dccIpc.uvMove(doc.id, dx, dy, DCC_PREVIEW_SIZE));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    historyRefresh: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        patch(assetId, { history: await dccIpc.history(doc.id) });
      } catch (e) {
        console.error("dcc.history failed", e);
      }
    },

    amend: async (assetId, index, value) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        applyResult(assetId, await dccIpc.amend(doc.id, index, value));
        // The rows describe the journal, and the journal just changed — but
        // so does every other door, and only this one refreshed them. The
        // panel now follows the generation stamp (audit fix), which is the
        // one thing that moves when the mesh does, so a refresh here would
        // be a second fetch of the same list.
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    undo: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.undo(doc.id));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    redo: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.redo(doc.id));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
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

    makeGarment: async (assetId, spec) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      patch(assetId, { busy: true, status: null });
      try {
        const lastGroom = await dccIpc.makeGarment(doc.id, spec);
        patch(assetId, {
          lastGroom,
          status: lastGroom.ok
            ? `Wrote ${lastGroom.path ?? "a garment"}.`
            : `Garment refused: ${lastGroom.refusal ?? "unknown"}`,
        });
      } catch (e) {
        patch(assetId, { status: `Garment failed: ${String(e)}` });
      } finally {
        patch(assetId, { busy: false });
      }
    },

    growHair: async (assetId, spec) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      patch(assetId, { busy: true, status: null });
      try {
        const lastGroom = await dccIpc.growHair(doc.id, spec);
        patch(assetId, {
          lastGroom,
          status: lastGroom.ok
            ? `Wrote ${lastGroom.path ?? "a hairstyle"}.`
            : `Groom refused: ${lastGroom.refusal ?? "unknown"}`,
        });
      } catch (e) {
        patch(assetId, { status: `Groom failed: ${String(e)}` });
      } finally {
        patch(assetId, { busy: false });
      }
    },

    refresh: (assetId) => void pump(assetId),

    // ── pointer drags ─────────────────────────────────────────────────────
    //
    // `dragMove` deliberately does NOT await a document: it appends a point and
    // the next preview frame is what shows it. One round trip per pointer-move
    // instead of two, on the path that fires the most.

    setGizmo: async (assetId, mode) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.setGizmo(doc.id, mode));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    dragBegin: async (assetId, drag, x, y) => {
      const doc = entry(assetId).doc;
      if (!doc) return false;
      try {
        const res = await dccIpc.dragBegin(doc.id, drag, x, y, DCC_PREVIEW_SIZE);
        adopt(assetId, res.doc);
        // A **miss** is silent — it becomes a camera orbit — but a **refusal** is
        // a sentence. An author whose brush did nothing needs to know it was the
        // radius, not wonder whether they missed the model.
        patch(assetId, { refusal: res.refusal });
        return res.grabbed;
      } catch (e) {
        patch(assetId, { refusal: String(e) });
        return false;
      }
    },

    dragMove: async (assetId, x, y) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        await dccIpc.dragMove(doc.id, x, y, DCC_PREVIEW_SIZE);
        await pump(assetId);
      } catch (e) {
        console.error("dcc.dragMove failed", e);
      }
    },

    dragEnd: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        applyResult(assetId, await dccIpc.dragEnd(doc.id));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    dragCancel: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        adopt(assetId, await dccIpc.dragCancel(doc.id));
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      }
    },

    // ── UV ────────────────────────────────────────────────────────────────

    unwrap: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      patch(assetId, { busy: true });
      try {
        const res = await dccIpc.unwrap(doc.id);
        patch(assetId, {
          doc: res.doc,
          lastUnwrap: res,
          refusal: res.ok ? null : res.refusal,
        });
        void pump(assetId);
        await get().uvRefresh(assetId);
      } catch (e) {
        patch(assetId, { refusal: String(e) });
      } finally {
        patch(assetId, { busy: false });
      }
    },

    uvRefresh: async (assetId) => {
      const doc = entry(assetId).doc;
      if (!doc) return;
      try {
        const frame = await dccIpc.uvPreview(doc.id, DCC_PREVIEW_SIZE);
        patch(assetId, { uvImage: frame.image });
      } catch (e) {
        console.error("dcc.uvPreview failed", e);
      }
    },
  };
});

/** One document's view state, for a panel that knows its own asset id. */
export function useDccEntry(assetId: string | null): DccEntry {
  return useDccStore((s) => (assetId ? (s.docs[assetId] ?? EMPTY) : EMPTY));
}

/** Test-only: clear the per-document preview queues and tombstones. */
export function __resetDccPreviewQueueForTest(): void {
  flight.clear();
  tombstones.clear();
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
