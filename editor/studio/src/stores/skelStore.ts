/**
 * **Skeleton Editor store** (P24.3) — one entry per open `.inf_skel` asset.
 *
 * A deliberate near-copy of `dccStore`'s shape, because the two panels have the
 * same lifecycle problem and the Model Editor already paid to learn it. What is
 * copied is copied *with its reasons*; what is not is called out below.
 *
 * # Keyed, and REPLACED rather than patched
 *
 * The panel is registered `singleton: false`, so two open rigs are two mounted
 * tabs. Every mutating call returns a whole `SkelDocDto` and the entry is
 * overwritten with it — a rename can change a joint's name and its twin's
 * `mirror`, an `instantiateTemplate` replaces every joint at once, and a store
 * that merged fields would be maintaining a second, wrong copy.
 *
 * # Tombstones, for the reason `dccStore` documents
 *
 * Every backend call is async, so between `open` being called and its reply
 * arriving the panel can unmount (React StrictMode does exactly this on every dev
 * mount). Without a tombstone that leaks the backend `SkelSession` for the life
 * of the process, and a remount can adopt a session an in-flight close was about
 * to destroy. A tombstoned asset accepts no state, and its `open` reply is
 * answered with another `skel.close`.
 *
 * # What is NOT copied: there is no preview queue
 *
 * `dccStore` serializes an offscreen render + PNG + base64 round trip per
 * document. A skeleton has no such round trip — the panel draws the rig from the
 * joint list it already holds (see `SkeletonEditor`'s `projectRig`), so there is
 * no in-flight gate here and nothing to schedule. Stating it because an absent
 * `flight` map in a file otherwise shaped like `dccStore` reads like an omission.
 */
import { create } from "zustand";

import { registerUndoScope } from "../lib/undoScopes";

import { skel as skelIpc, type BodyPlanName } from "../lib/ipc";
import type { SkelApplyDto } from "../bindings/SkelApplyDto";
import type { SkelDocDto } from "../bindings/SkelDocDto";

/** One open rig's view state. */
export interface SkelEntry {
  doc: SkelDocDto | null;
  /** The selected joint's index, or `null`. Frontend-only: the backend has no
   *  selection concept for a skeleton, because no edit is "act on the selection". */
  selected: number | null;
  /** The last refusal, cleared by the next successful action. A value, not a throw. */
  refusal: string | null;
  /**
   * The last **warning** — a rename that left the canonical vocabulary or broke
   * a mirror pair, or a merge's joint offset.
   *
   * Kept until the next *warning-producing or refusing* action rather than until
   * the next action of any kind: the `lastSave` lesson from the Model Editor, one
   * panel over. An author who renames a joint and then clicks anything at all
   * must still be able to read what the rename cost.
   */
  warning: string | null;
  busy: boolean;
}

interface SkelState {
  /** Keyed by **asset id** — the same key the panel instance is parameterized by. */
  docs: Record<string, SkelEntry>;

  open: (assetId: string) => Promise<void>;
  close: (assetId: string) => Promise<void>;
  select: (assetId: string, joint: number | null) => void;

  renameJoint: (assetId: string, joint: number, name: string) => Promise<void>;
  setJointTransform: (
    assetId: string,
    joint: number,
    translation: [number, number, number],
    rotation: [number, number, number, number],
    scale: [number, number, number],
  ) => Promise<void>;
  setLimit: (
    assetId: string,
    joint: number,
    minDeg: [number, number, number] | null,
    maxDeg: [number, number, number] | null,
  ) => Promise<void>;

  addSocket: (assetId: string, name: string, joint: number) => Promise<void>;
  removeSocket: (assetId: string, name: string) => Promise<void>;
  placeSocket: (
    assetId: string,
    name: string,
    joint: number,
    translation: [number, number, number],
  ) => Promise<void>;

  instantiateTemplate: (
    assetId: string,
    plan: BodyPlanName,
    opts?: { legs?: number; heightM?: number },
  ) => Promise<void>;
  mergePart: (
    assetId: string,
    partAsset: string,
    attach: number,
  ) => Promise<void>;
  fitToMesh: (
    assetId: string,
    meshAsset: string,
    plan: BodyPlanName,
  ) => Promise<void>;

  undo: (assetId: string) => Promise<void>;
  redo: (assetId: string) => Promise<void>;
  save: (assetId: string) => Promise<void>;
}

const EMPTY: SkelEntry = {
  doc: null,
  selected: null,
  refusal: null,
  warning: null,
  busy: false,
};

/** Assets whose panel has closed — the tombstones (see the module docs). */
const tombstones = new Set<string>();

export const useSkelStore = create<SkelState>((set, get) => {
  const entry = (assetId: string): SkelEntry => get().docs[assetId] ?? EMPTY;

  /**
   * Update an existing entry. **Never creates one** — a late reply for a
   * document that has been closed is dropped, not resurrected. Only `open`
   * creates, and only after clearing the tombstone.
   */
  const patch = (assetId: string, next: Partial<SkelEntry>): void =>
    set((s) => {
      const cur = s.docs[assetId];
      if (!cur) return s;
      return { docs: { ...s.docs, [assetId]: { ...cur, ...next } } };
    });

  /**
   * Adopt an `apply` result.
   *
   * The selection is **clamped, not cleared**: an undo or a template swap can
   * shrink the rig under a selected index, and a panel rendering joint 31 of a
   * 19-joint list would read `undefined` on every field. Clearing it outright
   * would instead lose the author's place on every ordinary edit.
   */
  const applyResult = (assetId: string, res: SkelApplyDto): void => {
    const joints = res.doc.joints.length;
    const cur = entry(assetId).selected;
    patch(assetId, {
      doc: res.doc,
      refusal: res.ok ? null : res.refusal,
      // A successful edit with nothing to say CLEARS a stale warning; a refusal
      // leaves the previous one alone, because the refusal is what to read.
      warning: res.ok ? res.warning : entry(assetId).warning,
      selected: cur !== null && cur < joints ? cur : null,
      busy: false,
    });
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
        const doc = await skelIpc.open(assetId);
        if (tombstones.has(assetId)) {
          // The panel closed while this was in flight. The session it just
          // created is exactly the one that close could not name.
          await skelIpc.close(assetId);
          return;
        }
        patch(assetId, { doc, busy: false });
      } catch (e) {
        patch(assetId, { refusal: String(e), busy: false });
      }
    },

    close: async (assetId) => {
      tombstones.add(assetId);
      set((s) => {
        const docs = { ...s.docs };
        delete docs[assetId];
        return { docs };
      });
      try {
        await skelIpc.close(assetId);
      } catch (e) {
        console.error("skel.close failed", e);
      }
    },

    select: (assetId, joint) => patch(assetId, { selected: joint }),

    renameJoint: async (assetId, joint, name) =>
      applyResult(assetId, await skelIpc.renameJoint(assetId, joint, name)),

    setJointTransform: async (assetId, joint, translation, rotation, scale) =>
      applyResult(
        assetId,
        await skelIpc.setJointTransform(
          assetId,
          joint,
          translation,
          rotation,
          scale,
        ),
      ),

    setLimit: async (assetId, joint, minDeg, maxDeg) =>
      applyResult(
        assetId,
        await skelIpc.setLimit(assetId, joint, minDeg, maxDeg),
      ),

    addSocket: async (assetId, name, joint) =>
      applyResult(assetId, await skelIpc.addSocket(assetId, name, joint)),

    removeSocket: async (assetId, name) =>
      applyResult(assetId, await skelIpc.removeSocket(assetId, name)),

    placeSocket: async (assetId, name, joint, translation) =>
      applyResult(
        assetId,
        await skelIpc.placeSocket(
          assetId,
          name,
          joint,
          translation,
          [0, 0, 0, 1],
          [1, 1, 1],
        ),
      ),

    instantiateTemplate: async (assetId, plan, opts) =>
      applyResult(
        assetId,
        await skelIpc.instantiateTemplate(assetId, plan, opts),
      ),

    mergePart: async (assetId, partAsset, attach) =>
      applyResult(assetId, await skelIpc.mergePart(assetId, partAsset, attach)),

    fitToMesh: async (assetId, meshAsset, plan) => {
      patch(assetId, { busy: true });
      const res = await skelIpc.fitToMesh(assetId, meshAsset, plan);
      applyResult(assetId, res);
    },

    undo: async (assetId) => applyResult(assetId, await skelIpc.undo(assetId)),
    redo: async (assetId) => applyResult(assetId, await skelIpc.redo(assetId)),
    save: async (assetId) => applyResult(assetId, await skelIpc.save(assetId)),
  };
});

/** One rig's view state, for a panel that knows its own asset id. */
export function useSkelEntry(assetId: string | null): SkelEntry {
  return useSkelStore((s) => (assetId ? (s.docs[assetId] ?? EMPTY) : EMPTY));
}

/** Test-only: clear the tombstones between cases. */
export function __resetSkelTombstonesForTest(): void {
  tombstones.clear();
}

// **Ctrl+Z inside the Skeleton Editor undoes the RIG, in the panel you are
// looking at** (the P23.2a routing registry, the P23.4 aim).
//
// The scope is handed the focused panel's `params`, which for `skeleton:<assetId>`
// is the asset id — so with two rigs open the chord reaches the one under the
// cursor. Without this registration Ctrl+Z here would undo the SCENE: an actor
// the author cannot see, moving in a panel they are not looking at, while the
// skeleton in front of them does nothing.
registerUndoScope("skeleton", {
  undo: (params) => {
    if (params) void useSkelStore.getState().undo(params);
  },
  redo: (params) => {
    if (params) void useSkelStore.getState().redo(params);
  },
});
