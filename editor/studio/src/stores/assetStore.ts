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
import { listenTo, type UnlistenFn } from "../lib/events";
import { fuzzyMatch } from "../lib/fuzzy";
import { assets as assetsIpc, collections as collectionsIpc } from "../lib/ipc";

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
  favorites: string[]; // favorited folder paths
  /** Named content collections (persisted backend state) (E-P8). */
  collections: CollectionDto[];
  /** Name of the collection whose members filter the grid, or null (E-P8). */
  activeCollection: string | null;
  /** Thumbnail data-URLs by asset id ("" = requested/none). */
  thumbnails: Record<string, string>;
  /** Active/recent import jobs by job id. */
  imports: Record<number, ImportJob>;

  applySnapshot: (s: AssetSnapshot) => void;
  refresh: () => Promise<void>;
  applyImportEvent: (e: ImportEventDto) => void;

  setFolder: (path: string) => void;
  setSearch: (q: string) => void;
  setKindFilter: (kind: string | null) => void;
  setSelected: (id: string | null) => void;
  openEditor: (id: string) => void;
  closeEditor: () => void;
  openInstanceEditor: (id: string) => void;
  closeInstanceEditor: () => void;
  toggleFavorite: (path: string) => void;

  // ── collections (E-P8) ─────────────────────────────────────────────────
  fetchCollections: () => Promise<void>;
  setActiveCollection: (name: string | null) => void;
  createCollection: (name: string) => Promise<void>;
  renameCollection: (oldName: string, newName: string) => Promise<void>;
  deleteCollection: (name: string) => Promise<void>;
  addToCollection: (name: string, id: string) => Promise<void>;
  removeFromCollection: (name: string, id: string) => Promise<void>;

  loadThumbnail: (id: string) => void;
  importFiles: (sources: string[], dest?: string | null) => Promise<void>;
  createMaterial: () => Promise<void>;
  createAsset: (kind: "mat" | "struct" | "enum" | "table") => Promise<void>;
  deleteAsset: (id: string, force?: boolean) => Promise<DeleteResult>;
  rename: (id: string, name: string) => Promise<void>;
  duplicate: (id: string) => Promise<void>;
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
  favorites: [],
  collections: [],
  activeCollection: null,
  thumbnails: {},
  imports: {},

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
      imports: {
        ...state.imports,
        [e.job]: {
          job: Number(e.job),
          source: e.source,
          phase: e.phase as ImportJob["phase"],
          error: e.error,
          done: e.done == null ? null : Number(e.done),
          total: e.total == null ? null : Number(e.total),
        },
      },
    }));
    // On finish, re-fetch and reveal the primary asset.
    if (e.phase === "finished") {
      void get().refresh();
      if (e.primary) set({ selected: e.primary });
    }
  },

  // Selecting a folder leaves any active collection view (they're alternate
  // scopes for the grid).
  setFolder: (path) => set({ folder: path, activeCollection: null }),
  setSearch: (q) => set({ search: q }),
  setKindFilter: (kind) => set((s) => ({ kindFilter: s.kindFilter === kind ? null : kind })),
  setSelected: (id) => set({ selected: id }),
  openEditor: (id) => set({ editing: id, editingInstance: null, selected: id }),
  closeEditor: () => set({ editing: null }),
  openInstanceEditor: (id) => set({ editingInstance: id, editing: null, selected: id }),
  closeInstanceEditor: () => set({ editingInstance: null }),

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
  addToCollection: async (name, id) => {
    try {
      set({ collections: await collectionsIpc.add(name, id) });
    } catch (e) {
      console.error("collections.add failed", e);
    }
  },
  removeFromCollection: async (name, id) => {
    try {
      set({ collections: await collectionsIpc.remove(name, id) });
    } catch (e) {
      console.error("collections.remove failed", e);
    }
  },

  loadThumbnail: (id) => {
    if (get().thumbnails[id] !== undefined) return; // already requested
    set((s) => ({ thumbnails: { ...s.thumbnails, [id]: "" } }));
    assetsIpc
      .thumbnail(id)
      .then((url) => {
        if (url) set((s) => ({ thumbnails: { ...s.thumbnails, [id]: url } }));
      })
      .catch((e) => console.error("asset.thumbnail failed", e));
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
      const id = await assetsIpc.create(kind);
      // Materials have no editor; data kinds open the inline editor.
      set(kind === "mat" ? { selected: id } : { selected: id, editing: id });
    } catch (e) {
      console.error("asset.create failed", e);
    }
  },

  deleteAsset: async (id, force = false) => {
    const result = await assetsIpc.delete(id, force);
    if (result.deleted && get().selected === id) set({ selected: null });
    return result;
  },

  rename: async (id, name) => {
    await assetsIpc.rename(id, name);
  },
  duplicate: async (id) => {
    const newId = await assetsIpc.duplicate(id);
    set({ selected: newId });
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

let unlistenChanged: UnlistenFn | null = null;
let unlistenImport: UnlistenFn | null = null;
let unlistenCollections: UnlistenFn | null = null;
let unlistenProject: UnlistenFn | null = null;

/**
 * Load the initial snapshot + collections and subscribe to `assets://changed`,
 * `assets://import`, `collections://changed`, and `project://changed` (the last
 * re-roots content → re-fetch collections). Idempotent (StrictMode
 * double-mounts); returns a disposer.
 */
export async function initAssetSync(): Promise<() => void> {
  if (unlistenChanged) return () => {};
  unlistenChanged = await listenTo("assets://changed", () =>
    void useAssetStore.getState().refresh(),
  );
  unlistenImport = await listenTo("assets://import", (e) =>
    useAssetStore.getState().applyImportEvent(e),
  );
  unlistenCollections = await listenTo("collections://changed", () =>
    void useAssetStore.getState().fetchCollections(),
  );
  unlistenProject = await listenTo("project://changed", () =>
    void useAssetStore.getState().fetchCollections(),
  );
  await Promise.all([
    useAssetStore.getState().refresh(),
    useAssetStore.getState().fetchCollections(),
  ]);
  return () => {
    unlistenChanged?.();
    unlistenImport?.();
    unlistenCollections?.();
    unlistenProject?.();
    unlistenChanged = null;
    unlistenImport = null;
    unlistenCollections = null;
    unlistenProject = null;
  };
}
