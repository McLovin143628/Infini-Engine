/**
 * Live asset store (Phase 4): the frontend mirror of the backend asset DB.
 *
 * The authoritative asset database lives in Rust
 * (`inf-editor-core::assets::AssetProject`). This store is a projection: it
 * loads a full `AssetSnapshot`, then re-fetches on the `assets://changed`
 * event (content changes are user-paced, so a re-fetch is cheaper than a delta
 * protocol). Import progress streams over `assets://import`. Every mutation is
 * a command call.
 */
import { create } from "zustand";

import type { AssetDto } from "../bindings/AssetDto";
import type { AssetFolderDto } from "../bindings/AssetFolderDto";
import type { AssetSnapshot } from "../bindings/AssetSnapshot";
import type { CollectionDto } from "../bindings/CollectionDto";
import type { DeleteResult } from "../bindings/DeleteResult";
import type { ImportEventDto } from "../bindings/ImportEventDto";
import { getCommand, setCommandHandler } from "../lib/commands";
import { listenTo, refCountedInit } from "../lib/events";
import { fuzzyMatch } from "../lib/fuzzy";
import { assets as assetsIpc, collections as collectionsIpc, skel as skelIpc } from "../lib/ipc";

export type { AssetDto, AssetFolderDto, CollectionDto };

/** A live/finished import job, for the progress strip. */
export interface ImportJob {
  job: number;
  source: string;
  phase: "started" | "progress" | "finished" | "failed";
  error?: string | null;
  /** Units done / total, on a job that reports progress (terrain, P16.4a). */
  done?: number | null;
  total?: number | null;
}

/**
 * What an import wanted the author to know (P26.5) — the P26.1 dimension pair
 * plus the tail-cost badge.
 *
 * These reached `tracing` from the day they existed, so they landed in the
 * Output Log and nowhere the person who had just dropped a file was looking.
 * That is what the P26.1 ledger deferred as "P26.5's badging": an author
 * importing forty 128 decals is told, once, while they can still say no.
 * Nothing here is an error — the import succeeded — so it is dismissible and it
 * never blocks.
 */
export interface ImportAdvisory {
  source: string;
  messages: string[];
}

/**
 * What a delete attempt did: the backend's own answer, plus the failure the
 * store caught (round-2 finding R2.F13).
 *
 * `DeleteResult` distinguishes "deleted" from "blocked by referrers"; it has no
 * shape for "the command threw", which is what a locked or read-only payload
 * produces — and that case used to be an unhandled rejection with the asset
 * still on screen.
 */
export interface DeleteOutcome extends DeleteResult {
  /** The error, when the command itself failed. */
  error?: string;
}

interface AssetState {
  assets: Record<string, AssetDto>;
  folders: Record<string, AssetFolderDto>;
  rootPath: string;
  version: number;
  ready: boolean;

  /** UI state (local, not from the backend). */
  folder: string; // "" = All
  search: string;
  kindFilter: string | null;
  selected: string | null;
  /** Id of the data asset open in the inline editor, or null. */
  editing: string | null;
  /** Id of the material instance open in the override editor, or null (E-P2). */
  editingInstance: string | null;
  /** Id of the `.inf_biomes` open in the biome-set editor, or null (P19.2). */
  editingBiomeSet: string | null;
  favorites: string[]; // favorited folder paths
  /** Named content collections (persisted backend state) (E-P8). */
  collections: CollectionDto[];
  /** Name of the collection whose members filter the grid, or null (E-P8). */
  activeCollection: string | null;
  /**
   * Thumbnail data-URLs keyed by **`content_hash`**.
   *
   * Three states, and the distinction is load-bearing (round-2 finding):
   * `""` = requested and still in flight, `null` = the backend answered that
   * this asset has no thumbnail, absent = not asked (or evicted, and therefore
   * askable again). A refusal used to leave the `""` sentinel, which
   * `touchThumbnail` then renewed as most-recently-used on every re-read — so
   * the failed entry became MORE durable than a real picture, was never
   * retried, and could not be evicted while the grid kept asking.
   *
   * See `THUMBNAIL_CAP` / `loadThumbnail` for why it is a hash, a `Map`, and
   * bounded.
   */
  thumbnails: Map<string, string | null>;
  /** Active/recent import jobs by job id. */
  imports: Record<number, ImportJob>;
  /** Undismissed advisories from finished imports (P26.5). */
  importAdvisories: ImportAdvisory[];
  /**
   * **The asset being pointer-dragged out of the Content Drawer**, or null
   * (round-2 finding R2.F9).
   *
   * The drawer drags with pointer events rather than HTML5 drag-and-drop,
   * because its original target is the **native viewport child window** — a
   * hole in the DOM that no `dragover` can reach. That is why the cells are
   * `draggable={false}`, and it is why every DOM drop zone in the editor had to
   * be reached some other way. This slot is that way: a panel subscribes to it
   * to light up while a drag it can accept is in flight.
   */
  dragAsset: { id: string; kind: string } | null;

  /** A drawer cell's pointer drag crossed its threshold. */
  beginAssetDrag: (id: string, kind: string) => void;
  /** …and ended, however it ended. Always paired, including on cancel. */
  endAssetDrag: () => void;

  applySnapshot: (s: AssetSnapshot) => void;
  refresh: () => Promise<void>;
  applyImportEvent: (e: ImportEventDto) => void;
  /** Advisories from finished imports, newest last; dismissed as a batch. */
  dismissImportAdvisories: () => void;

  setFolder: (path: string) => void;
  setSearch: (q: string) => void;
  setKindFilter: (kind: string | null) => void;
  setSelected: (id: string | null) => void;
  openEditor: (id: string) => void;
  closeEditor: () => void;
  openInstanceEditor: (id: string) => void;
  closeInstanceEditor: () => void;
  openBiomeSetEditor: (id: string) => void;
  closeBiomeSetEditor: () => void;
  toggleFavorite: (path: string) => void;

  // ── collections (E-P8) ─────────────────────────────────────────────────
  fetchCollections: () => Promise<void>;
  setActiveCollection: (name: string | null) => void;
  createCollection: (name: string) => Promise<void>;
  renameCollection: (oldName: string, newName: string) => Promise<void>;
  deleteCollection: (name: string) => Promise<void>;
  /** `true` only if the collection really changed (R2.F13). */
  addToCollection: (name: string, id: string) => Promise<boolean>;
  removeFromCollection: (name: string, id: string) => Promise<boolean>;

  /**
   * Request the thumbnail for `id`'s current content, keyed by `contentHash`.
   * A no-op if that exact content has already been asked for.
   */
  loadThumbnail: (id: string, contentHash: string) => void;
  importFiles: (sources: string[], dest?: string | null) => Promise<void>;
  createMaterial: () => Promise<void>;
  /**
   * Create a new asset of `kind`. The `skel:*` kinds (P24.1) generate a template
   * body plan rather than an empty document, so they go through their own IPC
   * door; they open no editor because the skeleton panel is P24.3's.
   */
  createAsset: (
    kind:
      | "mat"
      | "struct"
      | "enum"
      | "table"
      | "biomeset"
      | "skel:biped"
      | "skel:biped-canonical"
      | "skel:quadruped"
      | "skel:hexapod",
  ) => Promise<void>;
  deleteAsset: (id: string, force?: boolean) => Promise<DeleteOutcome>;
  /** The failure message, or `null` when it worked (R2.F13). */
  rename: (id: string, name: string) => Promise<string | null>;
  duplicate: (id: string) => Promise<string | null>;
}

/**
 * How many finished import jobs the progress strip keeps (F-lens L7.M11).
 *
 * `importAdvisories` next door was already capped at 20 for exactly this
 * reason; `imports` was not, and every job the backend has ever reported stayed
 * in the store for the life of the session. Only `started`/`progress` rows are
 * ever *rendered* (`ContentDrawer`'s `activeImports`), so the finished ones are
 * pure ballast — but they are ballast in a bridged-adjacent store that is
 * spread-copied on every event, and importing a folder of a few thousand files
 * makes each subsequent event O(jobs).
 */
const IMPORT_CAP = 50;

/**
 * How many rendered thumbnails stay in memory (F-lens L7.H5).
 *
 * A thumbnail is a base64 PNG data-URL — order 10–100 KB each — and the cache
 * had **no bound at all**: scrolling a content root of a few thousand
 * previewable assets accumulated every one of them for the life of the session,
 * as strings, in a zustand store.
 *
 * It must be larger than the number of cells that can be MOUNTED AT ONCE
 * (round-2 finding: the cap was 256 and a wide drawer mounts ~300 virtualized
 * cells). Under that, the later cells evict the earlier ones while both are on
 * screen, and the earlier ones stay blank for as long as the drawer is open —
 * a cache smaller than its own working set is not a cache, it is a treadmill.
 *
 * Beyond the working set the answer is cheap to re-fetch (the backend keeps its
 * own content-hash PNG cache on disk —
 * `inf_editor_core::thumbnail::ThumbnailCache`), so evicting costs an IPC round
 * trip, not a re-render.
 */
export const THUMBNAIL_CAP = 1024;

/**
 * Look a thumbnail up and mark it most-recently-used; `true` if present.
 *
 * `Map` iterates in insertion order, so "delete then re-set" IS the LRU touch —
 * it moves the key to the end, and `putThumbnail` evicts from the front.
 *
 * `undefined` (absent) is the only miss. `""` means a request is in flight and
 * `null` means the backend answered "this asset has no thumbnail" — both are
 * held, because both are things we already know.
 */
function touchThumbnail(cache: Map<string, string | null>, key: string): boolean {
  const hit = cache.get(key);
  if (hit === undefined) return false;
  cache.delete(key);
  cache.set(key, hit);
  return true;
}

/**
 * Insert (or refresh) a thumbnail, evicting the least-recently-used entries.
 *
 * **Mutates in place.** The old shape was `{ ...s.thumbnails, [id]: url }`,
 * which copied the entire cache — every string reference in it — on every single
 * insert, so filling a grid of N assets was O(N²) copies. The store is still
 * notified (`set({ thumbnails: cache })` builds a fresh *state* object, which is
 * what zustand compares), so subscribers re-run their selectors and see the new
 * value; what is skipped is duplicating the container itself.
 */
function putThumbnail(cache: Map<string, string | null>, key: string, url: string | null): void {
  cache.delete(key);
  cache.set(key, url);
  while (cache.size > THUMBNAIL_CAP) {
    const oldest = cache.keys().next();
    if (oldest.done) break;
    cache.delete(oldest.value);
  }
}

/**
 * Keep the newest `IMPORT_CAP` jobs, never dropping one that is still running.
 *
 * Jobs are numbered by the backend in issue order, so "newest" is the highest
 * job number. An active job is retained regardless of its age: the strip is the
 * only place a long-running terrain import reports itself, and evicting it would
 * make the progress bar vanish mid-import.
 */
function capImports(jobs: Record<number, ImportJob>): Record<number, ImportJob> {
  const keys = Object.keys(jobs);
  if (keys.length <= IMPORT_CAP) return jobs;
  const sorted = keys.map(Number).sort((a, b) => b - a);
  const keep = new Set(sorted.slice(0, IMPORT_CAP));
  const out: Record<number, ImportJob> = {};
  for (const k of sorted) {
    const job = jobs[k]!;
    if (keep.has(k) || job.phase === "started" || job.phase === "progress") out[k] = job;
  }
  return out;
}

function assetMap(list: AssetDto[]): Record<string, AssetDto> {
  const m: Record<string, AssetDto> = {};
  for (const a of list) m[a.id] = a;
  return m;
}
function folderMap(list: AssetFolderDto[]): Record<string, AssetFolderDto> {
  const m: Record<string, AssetFolderDto> = {};
  for (const f of list) m[f.path] = f;
  return m;
}

export const useAssetStore = create<AssetState>((set, get) => ({
  assets: {},
  folders: {},
  rootPath: "",
  version: 0,
  ready: false,
  folder: "",
  search: "",
  kindFilter: null,
  selected: null,
  editing: null,
  editingInstance: null,
  editingBiomeSet: null,
  favorites: [],
  collections: [],
  activeCollection: null,
  thumbnails: new Map(),
  imports: {},
  importAdvisories: [],
  dragAsset: null,

  beginAssetDrag: (id, kind) => set({ dragAsset: { id, kind } }),
  endAssetDrag: () => set({ dragAsset: null }),

  applySnapshot: (s) =>
    set({
      assets: assetMap(s.assets),
      folders: folderMap(s.folders),
      rootPath: s.root,
      version: Number(s.version),
      ready: true,
    }),

  refresh: async () => {
    try {
      get().applySnapshot(await assetsIpc.snapshot());
    } catch (e) {
      console.error("assets.snapshot failed", e);
    }
  },

  applyImportEvent: (e) => {
    set((state) => ({
      imports: capImports({
        ...state.imports,
        [e.job]: {
          job: Number(e.job),
          source: e.source,
          phase: e.phase as ImportJob["phase"],
          error: e.error,
          done: e.done == null ? null : Number(e.done),
          total: e.total == null ? null : Number(e.total),
        },
      }),
    }));
    // On finish, re-fetch and reveal the primary asset.
    if (e.phase === "finished") {
      void get().refresh();
      if (e.primary) set({ selected: e.primary });
      // …and surface whatever it had to advise about. Appended rather than
      // replaced: dropping ten files at once produces ten finished events, and
      // an author who imported a folder of small decals wants the count.
      if (e.advisories.length > 0) {
        set((state) => ({
          importAdvisories: [
            ...state.importAdvisories,
            { source: e.source, messages: e.advisories },
          ].slice(-20),
        }));
      }
    }
  },

  dismissImportAdvisories: () => set({ importAdvisories: [] }),

  // Selecting a folder leaves any active collection view (they're alternate
  // scopes for the grid).
  setFolder: (path) => set({ folder: path, activeCollection: null }),
  setSearch: (q) => set({ search: q }),
  setKindFilter: (kind) => set((s) => ({ kindFilter: s.kindFilter === kind ? null : kind })),
  setSelected: (id) => set({ selected: id }),
  // The three inline editors share the drawer's content area, so opening one
  // closes the others (the drawer renders whichever slot is set).
  openEditor: (id) =>
    set({ editing: id, editingInstance: null, editingBiomeSet: null, selected: id }),
  closeEditor: () => set({ editing: null }),
  openInstanceEditor: (id) =>
    set({ editingInstance: id, editing: null, editingBiomeSet: null, selected: id }),
  closeInstanceEditor: () => set({ editingInstance: null }),
  openBiomeSetEditor: (id) =>
    set({ editingBiomeSet: id, editing: null, editingInstance: null, selected: id }),
  closeBiomeSetEditor: () => set({ editingBiomeSet: null }),

  toggleFavorite: (path) =>
    set((s) => ({
      favorites: s.favorites.includes(path)
        ? s.favorites.filter((f) => f !== path)
        : [...s.favorites, path],
    })),

  fetchCollections: async () => {
    try {
      const list = await collectionsIpc.list();
      set((s) => ({
        collections: list,
        // Clear the active filter if its collection vanished (deleted/renamed).
        activeCollection:
          s.activeCollection && list.some((c) => c.name === s.activeCollection)
            ? s.activeCollection
            : null,
      }));
    } catch (e) {
      console.error("collections.list failed", e);
    }
  },
  // Toggle: clicking the active collection clears it. Entering a collection
  // leaves the folder-scoped view (folder stays, but the collection wins).
  setActiveCollection: (name) =>
    set((s) => ({ activeCollection: s.activeCollection === name ? null : name })),
  createCollection: async (name) => {
    try {
      set({ collections: await collectionsIpc.create(name) });
    } catch (e) {
      console.error("collections.create failed", e);
      throw e;
    }
  },
  renameCollection: async (oldName, newName) => {
    try {
      const list = await collectionsIpc.rename(oldName, newName);
      set((s) => ({
        collections: list,
        activeCollection: s.activeCollection === oldName ? newName : s.activeCollection,
      }));
    } catch (e) {
      console.error("collections.rename failed", e);
      throw e;
    }
  },
  deleteCollection: async (name) => {
    try {
      const list = await collectionsIpc.delete(name);
      set((s) => ({
        collections: list,
        activeCollection: s.activeCollection === name ? null : s.activeCollection,
      }));
    } catch (e) {
      console.error("collections.delete failed", e);
    }
  },
  // **These answer whether it happened** (round-2 finding R2.F13). The drawer
  // called them as `void addTo(...)` and then said "Added …" unconditionally,
  // because the store caught the failure internally — so the status bar stated
  // an asset was in a collection it was not in. A caught error that the caller
  // cannot see is worse than an uncaught one: it looks handled.
  addToCollection: async (name, id) => {
    try {
      set({ collections: await collectionsIpc.add(name, id) });
      return true;
    } catch (e) {
      console.error("collections.add failed", e);
      return false;
    }
  },
  removeFromCollection: async (name, id) => {
    try {
      set({ collections: await collectionsIpc.remove(name, id) });
      return true;
    } catch (e) {
      console.error("collections.remove failed", e);
      return false;
    }
  },

  loadThumbnail: (id, contentHash) => {
    const cache = get().thumbnails;
    if (touchThumbnail(cache, contentHash)) return; // this content already asked for
    putThumbnail(cache, contentHash, "");
    set({ thumbnails: cache });
    assetsIpc
      .thumbnail(id)
      .then((url) => {
        const live = get().thumbnails;
        // `null` is an ANSWER — no adapter, or a kind with no preview — and it
        // is recorded as one, so it is not re-requested and is not immortal
        // either. Leaving the `""` sentinel made a refusal the most durable
        // entry in the cache.
        putThumbnail(live, contentHash, url ?? null);
        set({ thumbnails: live });
      })
      .catch((e) => {
        // An error is NOT an answer: the sentinel is dropped so a transient
        // failure can be retried rather than blanking the cell for the session.
        console.error("asset.thumbnail failed", e);
        const live = get().thumbnails;
        if (live.get(contentHash) === "") {
          live.delete(contentHash);
          set({ thumbnails: live });
        }
      });
  },

  importFiles: async (sources, dest = null) => {
    if (!sources.length) return;
    try {
      await assetsIpc.import(sources, dest);
    } catch (e) {
      console.error("asset.import failed", e);
    }
  },

  createMaterial: async () => {
    try {
      const id = await assetsIpc.create("mat", "Materials", "New Material");
      set({ selected: id });
    } catch (e) {
      console.error("asset.create failed", e);
    }
  },

  createAsset: async (kind) => {
    try {
      if (kind.startsWith("skel:")) {
        // A template body plan (P24.1): generated by the backend, selected in the
        // drawer, and NOT opened — there is no skeleton editor until P24.3, and
        // opening the data editor on a rig would be worse than opening nothing.
        const plan = kind.slice(5) as
          | "biped"
          | "biped-canonical"
          | "quadruped"
          | "hexapod";
        set({ selected: await skelIpc.createTemplate(plan) });
        return;
      }
      const id = await assetsIpc.create(kind);
      // Materials have no inline editor (they open the graph panel); data kinds
      // and biome sets each open their own.
      if (kind === "mat") set({ selected: id });
      else if (kind === "biomeset") get().openBiomeSetEditor(id);
      else get().openEditor(id);
    } catch (e) {
      console.error("asset.create failed", e);
    }
  },

  // **The three context-menu mutations, caught** (round-2 finding R2.F13).
  // Eleven siblings in this file already had a `try`; these did not, and every
  // call site is `void store.x(...)`. A DELETE that failed — the payload locked
  // by another process, a read-only file — was an unhandled rejection with the
  // asset still on screen and nothing said.
  //
  // Each answers what happened rather than throwing, because the drawer's
  // caller is a menu item that has to decide what to tell the author.
  deleteAsset: async (id, force = false) => {
    try {
      const result = await assetsIpc.delete(id, force);
      if (result.deleted && get().selected === id) set({ selected: null });
      return result;
    } catch (e) {
      console.error("asset.delete failed", e);
      // Shaped like a refusal with no blockers, so the caller's existing
      // "not deleted" path runs and `error` carries the reason.
      return { deleted: false, blockers: [], error: String(e) };
    }
  },

  rename: async (id, name) => {
    try {
      await assetsIpc.rename(id, name);
      return null;
    } catch (e) {
      console.error("asset.rename failed", e);
      return String(e);
    }
  },
  duplicate: async (id) => {
    try {
      const newId = await assetsIpc.duplicate(id);
      set({ selected: newId });
      return null;
    } catch (e) {
      console.error("asset.duplicate failed", e);
      return String(e);
    }
  },
}));

/**
 * Assets visible under the current filters. A pure function of its inputs (call
 * inside `useMemo`, not as a store selector — it returns a fresh array each
 * call).
 *
 * When `collectionIds` is provided (a named collection is active) the grid is
 * scoped to that id set instead of the folder tree — collections are a
 * cross-folder view. The kind + search filters still apply in both modes.
 */
export function visibleAssets(
  assets: Record<string, AssetDto>,
  folder: string,
  kindFilter: string | null,
  search: string,
  collectionIds: string[] | null = null,
): AssetDto[] {
  const q = search.trim();
  const idSet = collectionIds ? new Set(collectionIds) : null;
  return Object.values(assets)
    .filter((a) => {
      const inScope = idSet
        ? idSet.has(a.id)
        : folder === "" || a.folder === folder || a.folder.startsWith(folder + "/");
      const matchesKind = kindFilter === null || a.kind === kindFilter;
      const matchesSearch = !q || fuzzyMatch(q, a.name) !== null;
      return inScope && matchesKind && matchesSearch;
    })
    .sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
}

/**
 * The absolute path of an asset's payload (wave SCRIPT2b).
 *
 * `AssetDto.path` is relative to the content root and forward-slashed, and
 * `AssetSnapshot.root` is the absolute root **already forward-slashed by
 * `snapshot.rs`** — so joining them needs no separator translation, only a
 * guard against doubling the slash. The one caller today is the drawer, which
 * hands the result to `infinity:open-file`; anything else that needs a real
 * file (rather than a GUID) should come through here rather than rebuild it.
 *
 * An empty root answers `""` rather than an absolute-looking path built from
 * nothing: "the content root is not known yet" must not become a path that
 * confidently points at the filesystem's root.
 */
export function contentAbsPath(root: string, relative: string): string {
  if (!root || !relative) return "";
  return `${root.replace(/[/\\]+$/, "")}/${relative.replace(/^[/\\]+/, "")}`;
}

/** Immediate child folders of `path`, sorted by name (pure; use in `useMemo`). */
export function childFolders(
  folders: Record<string, AssetFolderDto>,
  path: string,
): AssetFolderDto[] {
  const parent = folders[path];
  if (!parent) return [];
  return parent.children
    .map((p) => folders[p])
    .filter((f): f is AssetFolderDto => !!f)
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** Attach asset handlers to the enumerated File-menu commands. */
export function registerAssetCommands(): void {
  const wire = (id: string, run: () => void | Promise<void>) => {
    if (getCommand(id)) setCommandHandler(id, run);
  };
  wire("file.importIntoLevel", () => importViaDialog());
  wire("asset.import", () => importViaDialog());
}

/** Open the native file picker and import the chosen files. */
export async function importViaDialog(): Promise<void> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const picked = await open({
    multiple: true,
    filters: [
      {
        name: "Importable Assets",
        extensions: ["gltf", "glb", "png", "jpg", "jpeg", "tga", "bmp", "hdr", "exr"],
      },
    ],
  });
  if (!picked) return;
  const files = Array.isArray(picked) ? picked : [picked];
  await useAssetStore.getState().importFiles(files);
}

/**
 * Load the initial snapshot + collections and subscribe to `assets://changed`,
 * `assets://import`, `collections://changed`, and `project://changed` (the last
 * re-roots content → re-fetch collections). Returns a disposer.
 *
 * **Refcounted** (`refCountedInit`, F-lens L7.M1): the previous shape set its
 * `unlistenChanged` guard *after* the first `await`, so React StrictMode's
 * mount → cleanup → mount subscribed all four channels twice and leaked the
 * first set of handles. See `refCountedInit` for why a plain synchronous flag
 * is not enough either.
 */
export const initAssetSync = refCountedInit(async (sink) => {
  // Each handle goes into the sink AS IT IS TAKEN (round-2 finding R2-7). The
  // `Promise.all` below can reject, and before the sink that left all four
  // subscriptions live with nothing holding a reference to release them.
  const unlistenChanged = await listenTo("assets://changed", () =>
    void useAssetStore.getState().refresh(),
  );
  sink(unlistenChanged);
  const unlistenImport = await listenTo("assets://import", (e) =>
    useAssetStore.getState().applyImportEvent(e),
  );
  sink(unlistenImport);
  const unlistenCollections = await listenTo("collections://changed", () =>
    void useAssetStore.getState().fetchCollections(),
  );
  sink(unlistenCollections);
  const unlistenProject = await listenTo("project://changed", () =>
    void useAssetStore.getState().fetchCollections(),
  );
  sink(unlistenProject);
  await Promise.all([
    useAssetStore.getState().refresh(),
    useAssetStore.getState().fetchCollections(),
  ]);
  return () => {
    unlistenChanged();
    unlistenImport();
    unlistenCollections();
    unlistenProject();
  };
});
