/**
 * Content Drawer (Phase 4): the live asset browser.
 *
 * A slide-up surface over the real asset database (`assetStore`): Favorites +
 * folder tree | filter chips | breadcrumbs + a virtualized thumbnail grid, with
 * import, drag-to-viewport, context menus, and the delete-with-references
 * warning (the Phase 4 gate).
 *
 * Layout note (airspace rule): the drawer inserts BELOW the workspace and
 * pushes it up — HTML can never draw over the native viewport hole, so the
 * viewport rect-sync shrinks the engine window instead. Drag-to-viewport uses
 * pointer capture (not HTML5 DnD): the ghost dies over the native child, so the
 * drop crosses via IPC.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bone,
  Box,
  ChevronRight,
  Circle,
  Copy,
  FileCode2,
  FileText,
  Folder,
  FolderOpen,
  FolderPlus,
  FolderSearch,
  Image,
  Import,
  Layers,
  Library,
  Link2,
  Minus,
  Music2,
  Paintbrush,
  Palette,
  Pencil,
  Plus,
  Save,
  Search,
  Star,
  Trash2,
  X,
} from "lucide-react";
import { useVirtualizer } from "@tanstack/react-virtual";

import { cn } from "../lib/utils";
import { deliverAssetDrop } from "../lib/assetDrop";
import { executeCommand } from "../lib/commands";
import { scene as sceneIpc, shell, viewport } from "../lib/ipc";
import { CUTOUT_ATTR, useViewportCutout } from "../lib/viewportOverlay";
import { useDockLayout } from "../panels/dock/dockLayoutStore";
import { useShellStore } from "../stores/shellStore";
import { useSpriteSheetStore } from "../stores/spriteSheetStore";
import {
  childFolders,
  importViaDialog,
  useAssetStore,
  visibleAssets,
  type AssetDto,
} from "../stores/assetStore";
import BiomeSetEditor from "./BiomeSetEditor";
import DataAssetEditor from "./DataAssetEditor";
import MaterialInstanceEditor from "./MaterialInstanceEditor";

const DATA_KINDS = new Set(["struct", "enum", "table"]);

const KIND_TINT: Record<string, string> = {
  mesh: "text-(--ink-info)",
  texture: "text-(--ink-success)",
  material: "text-(--ink-accent)",
  material_instance: "text-(--ink-accent)",
  blueprint: "text-(--ink-accent)",
  audio: "text-(--ink-warning)",
  struct: "text-(--ink-info)",
  enum: "text-(--ink-info)",
  table: "text-(--ink-text-dim)",
  biome_set: "text-(--ink-success)",
};
function kindTint(kind: string): string {
  return KIND_TINT[kind] ?? "text-(--ink-text-dim)";
}

/** Static per-kind glyph — a switch (not a dynamic component binding) so the
 *  react-hooks static-components rule stays satisfied. */
function KindGlyph({
  kind,
  size,
  className,
}: {
  kind: string;
  size: number;
  className?: string;
}) {
  const p = { size, className };
  switch (kind) {
    case "mesh":
      return <Box {...p} />;
    case "texture":
      return <Image {...p} />;
    case "material":
      return <Circle {...p} />;
    case "material_instance":
      return <Paintbrush {...p} />;
    case "blueprint":
      return <FileCode2 {...p} />;
    case "audio":
      return <Music2 {...p} />;
    case "biome_set":
      return <Palette {...p} />;
    case "skeleton":
      return <Bone {...p} />;
    case "struct":
    case "enum":
      return <Layers {...p} />;
    default:
      return <FileText {...p} />;
  }
}

const CELL_W = 104;
const CELL_H = 124;
const GAP = 8;

/** A newline, spelled once — a tooltip is plain text and JSX escapes are not. */
const NL = String.fromCharCode(10);

export default function ContentDrawer() {
  const open = useShellStore((s) => s.drawerOpen);
  const setOpen = useShellStore((s) => s.setDrawerOpen);

  const ready = useAssetStore((s) => s.ready);
  const assetsById = useAssetStore((s) => s.assets);
  const folder = useAssetStore((s) => s.folder);
  const search = useAssetStore((s) => s.search);
  const kindFilter = useAssetStore((s) => s.kindFilter);
  const selected = useAssetStore((s) => s.selected);
  const favorites = useAssetStore((s) => s.favorites);
  const imports = useAssetStore((s) => s.imports);
  // P26.5's import badging: what the last imports wanted the author to know.
  const importAdvisories = useAssetStore((s) => s.importAdvisories);
  const dismissImportAdvisories = useAssetStore((s) => s.dismissImportAdvisories);

  const editing = useAssetStore((s) => s.editing);
  const editingInstance = useAssetStore((s) => s.editingInstance);
  const editingBiomeSet = useAssetStore((s) => s.editingBiomeSet);
  const collections = useAssetStore((s) => s.collections);
  const activeCollection = useAssetStore((s) => s.activeCollection);

  const setFolder = useAssetStore((s) => s.setFolder);
  const setSearch = useAssetStore((s) => s.setSearch);
  const setKindFilter = useAssetStore((s) => s.setKindFilter);
  const setSelected = useAssetStore((s) => s.setSelected);
  const toggleFavorite = useAssetStore((s) => s.toggleFavorite);
  const openEditor = useAssetStore((s) => s.openEditor);
  const closeEditor = useAssetStore((s) => s.closeEditor);
  const openInstanceEditor = useAssetStore((s) => s.openInstanceEditor);
  const closeInstanceEditor = useAssetStore((s) => s.closeInstanceEditor);
  const openBiomeSetEditor = useAssetStore((s) => s.openBiomeSetEditor);
  const closeBiomeSetEditor = useAssetStore((s) => s.closeBiomeSetEditor);
  const createAsset = useAssetStore((s) => s.createAsset);

  // The active collection's member id set (cross-folder grid scope).
  const activeIds = useMemo(
    () => collections.find((c) => c.name === activeCollection)?.ids ?? null,
    [collections, activeCollection],
  );
  const items = useMemo(
    () =>
      visibleAssets(
        assetsById,
        folder,
        kindFilter,
        search,
        activeCollection ? activeIds : null,
      ),
    [assetsById, folder, kindFilter, search, activeCollection, activeIds],
  );

  const [menu, setMenu] = useState<{
    x: number;
    y: number;
    asset: AssetDto;
  } | null>(null);
  const [addOpen, setAddOpen] = useState(false);

  const onCellOpen = (asset: AssetDto) => {
    if (DATA_KINDS.has(asset.kind)) openEditor(asset.id);
    else if (asset.kind === "material_instance")
      openInstanceEditor(asset.id); // E-P2
    else if (asset.kind === "biome_set")
      openBiomeSetEditor(asset.id); // P19.2
    else if (asset.kind === "material")
      useDockLayout.getState().openPanel("material"); // P7.2 material editor
    // P23.4: a mesh opens the Model Editor, one panel instance per asset.
    else if (asset.kind === "mesh")
      useDockLayout.getState().openPanel("model", asset.id);
    // P24.3: a rig opens the Skeleton Editor, one panel instance per asset.
    else if (asset.kind === "skeleton")
      useDockLayout.getState().openPanel("skeleton", asset.id);
    else if (asset.kind === "pcg")
      useDockLayout.getState().openPanel("pcg"); // P10.5b PCG editor
    else if (asset.kind === "state_machine")
      useDockLayout.getState().openPanel("stateMachine"); // P11.2 state machine editor
    else if (asset.kind === "texture") {
      // P8.2a: open the Sprite Sheet slicer bound to this texture.
      void useSpriteSheetStore.getState().open(asset.id);
      useDockLayout.getState().openPanel("spriteSheet");
    }
  };

  // Kinds present, for the filter column.
  const kinds = useMemo(() => {
    const set = new Set<string>();
    for (const a of Object.values(assetsById)) set.add(a.kind);
    return [...set].sort();
  }, [assetsById]);

  const activeImports = Object.values(imports).filter(
    (i) => i.phase === "started" || i.phase === "progress",
  );

  return (
    <div
      className="flex shrink-0 flex-col overflow-hidden border-t border-(--ink-border) bg-(--ink-bg-1) transition-[height] duration-150 ease-out"
      style={{ height: open ? 320 : 0 }}
      aria-hidden={!open}
      onClick={() => {
        setMenu(null);
        setAddOpen(false);
      }}
    >
      {/* Top bar */}
      <div className="flex h-9 shrink-0 items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
        <div className="relative">
          <button
            className="flex h-6 items-center gap-1 rounded bg-(--ink-accent) px-2 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)"
            onClick={(e) => {
              e.stopPropagation();
              setAddOpen((v) => !v);
            }}
            title="Create a new asset"
          >
            <Plus size={13} /> Add
          </button>
          {addOpen && (
            <div
              className="absolute left-0 top-7 z-50 min-w-36 overflow-hidden rounded border border-(--ink-border) bg-(--ink-bg-2) py-1 shadow-lg"
              onClick={(e) => e.stopPropagation()}
            >
              {(
                [
                  ["mat", "Material"],
                  ["struct", "Struct"],
                  ["enum", "Enum"],
                  ["table", "Data Table"],
                  ["biomeset", "Biome Set"],
                  // P24.1 template body plans: a generated, socketed,
                  // limit-carrying rig rather than an empty document. Since SK1a
                  // `skel:biped` is the 161-bone UE5 mannequin (the backend calls
                  // it "Mannequin Rig"), and the twenty-joint canonical rig every
                  // committed clip in this repository is index-bound to has its
                  // own entry — without it there was no door to it at all.
                  ["skel:biped", "Skeleton — Mannequin (161 bones)"],
                  ["skel:biped-canonical", "Skeleton — Biped (20 bones)"],
                  ["skel:quadruped", "Skeleton — Quadruped"],
                  ["skel:hexapod", "Skeleton — Hexapod"],
                ] as const
              ).map(([kind, label]) => (
                <button
                  key={kind}
                  className="block w-full px-2.5 py-1 text-left text-xs hover:bg-(--ink-bg-3)"
                  onClick={() => {
                    setAddOpen(false);
                    void createAsset(kind);
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          )}
        </div>
        <button
          className="flex h-6 items-center gap-1 rounded px-2 text-xs hover:bg-(--ink-bg-3)"
          onClick={() => void importViaDialog()}
        >
          <Import size={13} /> Import
        </button>
        <button
          className="flex h-6 items-center gap-1 rounded px-2 text-xs hover:bg-(--ink-bg-3)"
          onClick={() => executeCommand("file.saveAll")}
        >
          <Save size={13} /> Save All
        </button>
        <div className="mx-1 h-5 w-px bg-(--ink-border)" />
        {/* Breadcrumbs */}
        <Breadcrumbs folder={folder} onNavigate={setFolder} />
        {activeImports.length > 0 && (
          <span className="ml-2 animate-pulse text-[11px] text-(--ink-accent)">
            Importing {activeImports.length}…
          </span>
        )}
        {/*
          The import BADGE (P26.5). Not an error and never blocking — the import
          succeeded — so it is one dismissible chip carrying the count, with
          every sentence in its tooltip. It sits beside the progress line
          because that is where the author is already looking when a drop
          finishes.
        */}
        {importAdvisories.length > 0 && (
          <button
            className="ml-2 flex h-5 items-center gap-1 rounded border border-(--ink-warn,--ink-border) px-1.5 text-[11px] text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
            title={importAdvisories
              .map((a) => [a.source, ...a.messages.map((m) => "  - " + m)].join(NL))
              .join(NL + NL)}
            onClick={(e) => {
              e.stopPropagation();
              dismissImportAdvisories();
            }}
          >
            ⚠ {importAdvisories.reduce((n, a) => n + a.messages.length, 0)} import
            {importAdvisories.reduce((n, a) => n + a.messages.length, 0) === 1 ? "" : "s"} to review
          </button>
        )}
        <div className="flex-1" />
        <div className="flex h-6 w-56 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1.5 focus-within:border-(--ink-accent)">
          <Search size={12} className="shrink-0 text-(--ink-text-faint)" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search assets…"
            className="w-full bg-transparent text-xs outline-none placeholder:text-(--ink-text-faint)"
          />
        </div>
        <button
          aria-label="Close Content Drawer"
          className="flex size-6 items-center justify-center rounded text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
          onClick={() => setOpen(false)}
        >
          <X size={14} />
        </button>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Favorites + folder tree */}
        <div className="w-48 shrink-0 overflow-auto border-r border-(--ink-border) p-1.5">
          <div className="flex items-center gap-1 px-1 pb-1 text-[11px] font-semibold text-(--ink-text-dim)">
            <Star size={11} /> Favorites
          </div>
          {favorites.length === 0 ? (
            <div className="px-1 pb-2 text-[11px] text-(--ink-text-faint)">
              None yet
            </div>
          ) : (
            favorites.map((f) => (
              <FolderRow
                key={`fav-${f}`}
                path={f}
                label={f === "" ? "Content" : (f.split("/").pop() ?? f)}
                active={folder === f}
                onSelect={setFolder}
              />
            ))
          )}
          <CollectionsSection />
          <div className="px-1 pt-1 pb-1 text-[11px] font-semibold text-(--ink-text-dim)">
            Content
          </div>
          <FolderTree
            path=""
            activePath={folder}
            onSelect={setFolder}
            onFavorite={toggleFavorite}
          />
        </div>

        {/* Filter column */}
        <div className="w-28 shrink-0 overflow-auto border-r border-(--ink-border) p-1.5">
          <div className="px-1 pb-1 text-[11px] font-semibold text-(--ink-text-dim)">
            Filters
          </div>
          {kinds.length === 0 && (
            <div className="px-1 text-[11px] text-(--ink-text-faint)">—</div>
          )}
          {kinds.map((kind) => (
            <button
              key={kind}
              className={cn(
                "flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-xs capitalize",
                kindFilter === kind
                  ? "bg-(--ink-selection)"
                  : "hover:bg-(--ink-bg-3)",
              )}
              onClick={() => setKindFilter(kind)}
            >
              {kind}
            </button>
          ))}
        </div>

        {/* Inline data-asset editor, or the virtualized thumbnail grid */}
        <div className="min-h-0 flex-1">
          {editing ? (
            <DataAssetEditor key={editing} id={editing} onClose={closeEditor} />
          ) : editingInstance ? (
            <MaterialInstanceEditor
              key={editingInstance}
              id={editingInstance}
              onClose={closeInstanceEditor}
            />
          ) : editingBiomeSet ? (
            <BiomeSetEditor
              key={editingBiomeSet}
              id={editingBiomeSet}
              onClose={closeBiomeSetEditor}
            />
          ) : !ready ? (
            <div className="p-4 text-xs text-(--ink-text-faint)">
              Loading content…
            </div>
          ) : items.length === 0 ? (
            <div className="p-4 text-xs text-(--ink-text-faint)">
              No assets match. Use Import (or drag files in) to add content.
            </div>
          ) : (
            <AssetGrid
              items={items}
              selected={selected}
              onSelect={setSelected}
              onOpen={onCellOpen}
              onContext={(x, y, asset) => setMenu({ x, y, asset })}
            />
          )}
        </div>
      </div>

      {menu && (
        <AssetContextMenu
          x={menu.x}
          y={menu.y}
          asset={menu.asset}
          onClose={() => setMenu(null)}
        />
      )}
    </div>
  );
}

function Breadcrumbs({
  folder,
  onNavigate,
}: {
  folder: string;
  onNavigate: (p: string) => void;
}) {
  const parts = folder === "" ? [] : folder.split("/");
  return (
    <div className="flex min-w-0 items-center gap-0.5 text-xs text-(--ink-text-dim)">
      <button
        className="hover:text-(--ink-text)"
        onClick={() => onNavigate("")}
      >
        Content
      </button>
      {parts.map((part, i) => {
        const path = parts.slice(0, i + 1).join("/");
        return (
          <span key={path} className="flex items-center gap-0.5">
            <ChevronRight size={12} />
            <button
              className={cn(
                i === parts.length - 1
                  ? "text-(--ink-text)"
                  : "hover:text-(--ink-text)",
              )}
              onClick={() => onNavigate(path)}
            >
              {part}
            </button>
          </span>
        );
      })}
    </div>
  );
}

function FolderRow({
  path,
  label,
  active,
  onSelect,
}: {
  path: string;
  label: string;
  active: boolean;
  onSelect: (p: string) => void;
}) {
  return (
    <button
      className={cn(
        "flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-xs",
        active ? "bg-(--ink-selection)" : "hover:bg-(--ink-bg-3)",
      )}
      onClick={() => onSelect(path)}
    >
      {active ? (
        <FolderOpen size={13} className="text-(--ink-warning)" />
      ) : (
        <Folder size={13} className="text-(--ink-warning)" />
      )}
      <span className="truncate">{label}</span>
    </button>
  );
}

/** The persisted named-collections list (E-P8): click a collection to scope the
 *  grid to its members; "+" creates one; hover-X deletes; double-click renames. */
function CollectionsSection() {
  const collections = useAssetStore((s) => s.collections);
  const active = useAssetStore((s) => s.activeCollection);
  const setActive = useAssetStore((s) => s.setActiveCollection);
  const create = useAssetStore((s) => s.createCollection);
  const rename = useAssetStore((s) => s.renameCollection);
  const del = useAssetStore((s) => s.deleteCollection);

  const onNew = async () => {
    const name = window.prompt("New collection name");
    if (!name || !name.trim()) return;
    try {
      await create(name.trim());
    } catch (e) {
      window.alert(String(e));
    }
  };
  const onRename = async (old: string) => {
    const name = window.prompt("Rename collection", old);
    if (!name || !name.trim() || name.trim() === old) return;
    try {
      await rename(old, name.trim());
    } catch (e) {
      window.alert(String(e));
    }
  };
  const onDelete = async (name: string) => {
    if (
      window.confirm(
        `Delete collection "${name}"? (The assets themselves are not deleted.)`,
      )
    )
      await del(name);
  };

  return (
    <>
      <div className="flex items-center justify-between px-1 pt-2 pb-1 text-[11px] font-semibold text-(--ink-text-dim)">
        <span className="flex items-center gap-1">
          <Library size={11} /> Collections
        </span>
        <button
          className="rounded p-0.5 hover:bg-(--ink-bg-3)"
          onClick={() => void onNew()}
          title="New collection"
        >
          <Plus size={12} />
        </button>
      </div>
      {collections.length === 0 ? (
        <div className="px-1 pb-1 text-[11px] text-(--ink-text-faint)">
          None yet
        </div>
      ) : (
        collections.map((c) => (
          <div key={c.name} className="group flex items-center">
            <button
              className={cn(
                "flex min-w-0 flex-1 items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-xs",
                active === c.name
                  ? "bg-(--ink-selection)"
                  : "hover:bg-(--ink-bg-3)",
              )}
              onClick={() => setActive(c.name)}
              onDoubleClick={() => void onRename(c.name)}
              title="Click to filter the grid; double-click to rename"
            >
              <Library size={13} className="shrink-0 text-(--ink-accent)" />
              <span className="truncate">{c.name}</span>
              <span className="ml-auto shrink-0 text-[10px] text-(--ink-text-faint)">
                {c.ids.length}
              </span>
            </button>
            <button
              className="shrink-0 rounded p-0.5 text-(--ink-text-faint) opacity-0 hover:bg-(--ink-bg-3) hover:text-(--ink-danger) group-hover:opacity-100"
              onClick={() => void onDelete(c.name)}
              title="Delete collection"
            >
              <Minus size={12} />
            </button>
          </div>
        ))
      )}
    </>
  );
}

function FolderTree({
  path,
  activePath,
  onSelect,
  onFavorite,
  depth = 0,
}: {
  path: string;
  activePath: string;
  onSelect: (p: string) => void;
  onFavorite: (p: string) => void;
  depth?: number;
}) {
  const folders = useAssetStore((s) => s.folders);
  const children = useMemo(() => childFolders(folders, path), [folders, path]);
  const folderDto = folders[path];
  const label =
    path === ""
      ? "Content"
      : (folderDto?.name ?? path.split("/").pop() ?? path);
  return (
    <div style={{ paddingLeft: depth === 0 ? 0 : 10 }}>
      {depth > 0 && (
        <button
          className={cn(
            "flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-xs",
            activePath === path
              ? "bg-(--ink-selection)"
              : "hover:bg-(--ink-bg-3)",
          )}
          onClick={() => onSelect(path)}
          onDoubleClick={() => onFavorite(path)}
          title="Double-click to favorite"
        >
          {activePath === path ? (
            <FolderOpen size={13} className="text-(--ink-warning)" />
          ) : (
            <Folder size={13} className="text-(--ink-warning)" />
          )}
          <span className="truncate">{label}</span>
        </button>
      )}
      {children.map((c) => (
        <FolderTree
          key={c.path}
          path={c.path}
          activePath={activePath}
          onSelect={onSelect}
          onFavorite={onFavorite}
          depth={depth + 1}
        />
      ))}
    </div>
  );
}

function AssetGrid({
  items,
  selected,
  onSelect,
  onOpen,
  onContext,
}: {
  items: AssetDto[];
  selected: string | null;
  onSelect: (id: string) => void;
  onOpen: (asset: AssetDto) => void;
  onContext: (x: number, y: number, asset: AssetDto) => void;
}) {
  const parentRef = useRef<HTMLDivElement>(null);
  const [cols, setCols] = useState(6);

  useEffect(() => {
    const el = parentRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth - 8;
      setCols(Math.max(1, Math.floor((w + GAP) / (CELL_W + GAP))));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const rowCount = Math.ceil(items.length / cols);
  // TanStack Virtual returns non-memoizable functions; we don't pass them into
  // memoized children, so the React-Compiler incompatibility warning is moot.
  // eslint-disable-next-line react-hooks/incompatible-library
  const rowVirt = useVirtualizer({
    count: rowCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => CELL_H + GAP,
    overscan: 3,
  });

  return (
    <div ref={parentRef} className="h-full overflow-auto p-1">
      <div
        style={{
          height: rowVirt.getTotalSize(),
          position: "relative",
          width: "100%",
        }}
      >
        {rowVirt.getVirtualItems().map((vr) => {
          const start = vr.index * cols;
          const rowItems = items.slice(start, start + cols);
          return (
            <div
              key={vr.key}
              className="absolute left-0 flex w-full gap-2 px-1"
              style={{
                top: 0,
                transform: `translateY(${vr.start}px)`,
                height: CELL_H,
              }}
            >
              {rowItems.map((a) => (
                <AssetCell
                  key={a.id}
                  asset={a}
                  selected={selected === a.id}
                  onSelect={onSelect}
                  onOpen={onOpen}
                  onContext={onContext}
                />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AssetCell({
  asset,
  selected,
  onSelect,
  onOpen,
  onContext,
}: {
  asset: AssetDto;
  selected: boolean;
  onSelect: (id: string) => void;
  onOpen: (asset: AssetDto) => void;
  onContext: (x: number, y: number, asset: AssetDto) => void;
}) {
  // Keyed by CONTENT HASH, not asset id (F-lens L7.H5): a re-import keeps the
  // GUID and changes the bytes, and an id-keyed cache answered "already have
  // it" and left the old picture on screen for the rest of the session.
  const thumb = useAssetStore((s) => s.thumbnails.get(asset.content_hash));
  const loadThumbnail = useAssetStore((s) => s.loadThumbnail);
  const beginAssetDrag = useAssetStore((s) => s.beginAssetDrag);
  const endAssetDrag = useAssetStore((s) => s.endAssetDrag);
  const [ghost, setGhost] = useState<{ x: number; y: number } | null>(null);
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const dragging = useRef(false);

  // `thumb` is in the deps on purpose (round-2 finding): the request used to
  // fire once per mount, so a cell whose entry was EVICTED while it was still
  // on screen stayed blank until the drawer was closed and reopened.
  // `loadThumbnail` is idempotent per content hash — a pending or answered
  // entry short-circuits — so this asks exactly once per genuine miss.
  useEffect(() => {
    if (asset.previewable && thumb === undefined) {
      loadThumbnail(asset.id, asset.content_hash);
    }
  }, [asset.id, asset.previewable, asset.content_hash, thumb, loadThumbnail]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    onSelect(asset.id);
    dragStart.current = { x: e.clientX, y: e.clientY };
    dragging.current = false;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragStart.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    if (!dragging.current && Math.hypot(dx, dy) > 5) {
      dragging.current = true;
      beginAssetDrag(asset.id, asset.kind);
    }
    if (dragging.current) setGhost({ x: e.clientX, y: e.clientY });
  };
  // A drag that never ends leaves every drop zone lit and the store lying about
  // what is in flight. `pointercancel` (a touch turning into a scroll, the OS
  // taking the pointer) and `lostpointercapture` (an element removed mid-drag)
  // are the two ways it happens; `TilemapPanel` already handles the first and
  // this one had neither.
  const onPointerCancel = () => {
    setGhost(null);
    dragStart.current = null;
    dragging.current = false;
    endAssetDrag();
  };
  const onPointerUp = (e: React.PointerEvent) => {
    setGhost(null);
    const wasDragging = dragging.current;
    dragStart.current = null;
    dragging.current = false;
    endAssetDrag();
    if (!wasDragging) return; // a plain click already selected

    // **DOM drop zones first** (round-2 finding R2.F9). The Model Editor's
    // merge door was reachable only from an HTML5 `dataTransfer` payload that
    // nothing in the tree ever set: the drop zone highlighted for any native
    // drag and silently discarded it, so "drag-and-drop modular model building"
    // — the DCC vision's headline — had a dead door.
    //
    // It cannot be fixed by making these cells `draggable`: an HTML5 drag
    // cancels the pointer stream, and the pointer stream is the ONLY way to
    // reach the native viewport hole, which is not a DOM node. So the drawer
    // keeps its pointer drag and hit-tests DOM zones the same way it already
    // hit-tests the hole. The ghost is `pointer-events-none`, so
    // `elementFromPoint` sees what is under the cursor.
    const taken = deliverAssetDrop(e.clientX, e.clientY, {
      id: asset.id,
      kind: asset.kind,
      name: asset.name,
    });
    if (taken) return;

    const hole = document.querySelector("[data-viewport-hole]");
    if (!hole) return;
    const r = hole.getBoundingClientRect();
    const inside =
      e.clientX >= r.left &&
      e.clientX <= r.right &&
      e.clientY >= r.top &&
      e.clientY <= r.bottom;
    if (inside) {
      if (asset.kind === "material" || asset.kind === "material_instance") {
        // Apply-by-drag: a material/instance dropped over the viewport assigns
        // to the current selection (the drop carries no pick point) (P7.1/P7.4).
        sceneIpc
          .applyMaterial(asset.id)
          .catch((err) => console.error("apply material failed", err));
      } else if (asset.kind === "blueprint") {
        // Bind-by-drag: a `.inf_act` dropped over the viewport binds the current
        // selection to that blueprint class (its ActorClass link) (P9.5).
        sceneIpc
          .applyActor(asset.id)
          .catch((err) => console.error("bind actor failed", err));
      } else {
        // Place-by-drag (C): route through the wave-2 viewport drop so the
        // asset spawns at the cursor's world point (not the origin, as the old
        // direct `scene_spawn_asset` did). Coordinates are physical pixels
        // relative to the hole's top-left (the native-window contract).
        //
        // The payload is `asset:<kind>:<id>:<name>` (Wave E). The KIND is what
        // lets the viewport thread — which has no asset database — bind a
        // dropped mesh to `MeshRef.asset` instead of spawning yet another
        // anonymous placeholder cube. Format + parser:
        // `inf_editor_core::viewport_drop`.
        const dpr = window.devicePixelRatio || 1;
        viewport
          .drop({
            x: (e.clientX - r.left) * dpr,
            y: (e.clientY - r.top) * dpr,
            payload: `asset:${asset.kind}:${asset.id}:${asset.name}`,
          })
          .catch((err) => console.error("asset drop failed", err));
      }
    }
  };

  return (
    <>
      <button
        className={cn(
          "group flex touch-none select-none flex-col items-stretch rounded border p-1 text-left",
          selected
            ? "border-(--ink-accent) bg-(--ink-accent-muted)"
            : "border-(--ink-border) bg-(--ink-bg-2) hover:border-(--ink-border-strong)",
        )}
        style={{ width: CELL_W }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerCancel}
        onLostPointerCapture={onPointerCancel}
        onDoubleClick={() => onOpen(asset)}
        onContextMenu={(e) => {
          e.preventDefault();
          onSelect(asset.id);
          onContext(e.clientX, e.clientY, asset);
        }}
      >
        <div className="relative flex aspect-square items-center justify-center overflow-hidden rounded-sm bg-(--ink-bg-0)">
          {thumb ? (
            <img
              src={thumb}
              alt={asset.name}
              className="size-full object-cover"
              draggable={false}
            />
          ) : (
            <KindGlyph
              kind={asset.kind}
              size={30}
              className={kindTint(asset.kind)}
            />
          )}
          {asset.ref_count > 0 && (
            <span
              className="absolute right-0.5 top-0.5 rounded bg-(--ink-bg-2)/80 px-1 text-[9px] text-(--ink-text-dim)"
              title={`${asset.ref_count} reference(s)`}
            >
              ↗{asset.ref_count}
            </span>
          )}
        </div>
        <span className="mt-1 truncate text-[11px]" title={asset.name}>
          {asset.name}
        </span>
        <span className="truncate text-[10px] text-(--ink-text-faint)">
          {asset.kind_label}
        </span>
      </button>
      {ghost && (
        <div
          className="pointer-events-none fixed z-50 flex items-center gap-1 rounded border border-(--ink-accent) bg-(--ink-bg-2) px-2 py-0.5 text-xs opacity-80"
          style={{ left: ghost.x + 10, top: ghost.y + 6 }}
        >
          <KindGlyph
            kind={asset.kind}
            size={12}
            className={kindTint(asset.kind)}
          />{" "}
          {asset.name}
        </div>
      )}
    </>
  );
}

function AssetContextMenu({
  x,
  y,
  asset,
  onClose,
}: {
  x: number;
  y: number;
  asset: AssetDto;
  onClose: () => void;
}) {
  const pushStatus = useShellStore((s) => s.pushStatus);
  const rootPath = useAssetStore((s) => s.rootPath);
  const openInstanceEditor = useAssetStore((s) => s.openInstanceEditor);
  const collections = useAssetStore((s) => s.collections);
  const activeCollection = useAssetStore((s) => s.activeCollection);
  const addToCollection = useAssetStore((s) => s.addToCollection);
  const removeFromCollection = useAssetStore((s) => s.removeFromCollection);
  const createCollection = useAssetStore((s) => s.createCollection);
  const [subOpen, setSubOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  /**
   * **The airspace guard this menu never had** (Wave E, E2-4), now a CUTOUT
   * (UX2).
   *
   * It is a `position: fixed` popup at the cursor: with the drawer open near the
   * bottom of the window it opens UPWARD, across the viewport hole, and the
   * native child window draws over it — so the menu was invisible under the 3D
   * view, and clicking where an item appeared to be hit the viewport instead.
   * Every other overlay in the shell has held this guard since Phase 1; this one
   * was hand-rolled and skipped it.
   *
   * Held for the component's whole lifetime, which is exactly the menu's: it is
   * mounted only while open (`{menu && <AssetContextMenu …>}`).
   *
   * Still not migrated onto `ContextMenuSurface` (the shared surface Wave E
   * split out) because this menu has an inline expanding submenu — "Add to
   * Collection ▸" — that the surface does not model; that remainder stands.
   * Rect-supply needed no migration: the submenu expands INSIDE this element,
   * so one measured rectangle covers the menu at every size, and the hook's
   * `ResizeObserver` follows it as it grows.
   */
  useViewportCutout(true, menuRef);
  const store = {
    rename: useAssetStore((s) => s.rename),
    duplicate: useAssetStore((s) => s.duplicate),
    deleteAsset: useAssetStore((s) => s.deleteAsset),
  };
  const item = (
    label: string,
    glyph: React.ReactNode,
    run: () => void,
    danger = false,
  ) => (
    <button
      className={cn(
        "flex w-full items-center gap-2 px-2.5 py-1 text-left text-xs hover:bg-(--ink-bg-3)",
        danger && "text-(--ink-danger)",
      )}
      onClick={() => {
        run();
        onClose();
      }}
    >
      {glyph} {label}
    </button>
  );

  // **Seven context actions said nothing when they failed** (round-2 finding
  // R2.F13), and a silently failing DELETE was the worst of them: the asset
  // stayed on screen, the rejection went unhandled, and the author's only clue
  // was that nothing happened. Every one of them now ends in a status line.
  const doRename = () => {
    const name = window.prompt("Rename asset", asset.name);
    if (!name || name === asset.name) return;
    void store.rename(asset.id, name).then((err) => {
      if (err) pushStatus(`Rename failed: ${err}`, 8000);
    });
  };
  const doDuplicate = () => {
    void store.duplicate(asset.id).then((err) => {
      if (err) pushStatus(`Duplicate failed: ${err}`, 8000);
    });
  };
  const doDelete = async () => {
    const result = await store.deleteAsset(asset.id, false);
    if (result.error) {
      pushStatus(`Delete failed: ${result.error}`, 8000);
      return;
    }
    if (!result.deleted) {
      const names = result.blockers
        .map((b) => `• ${b.name} (${b.kind})`)
        .join("\n");
      const ok = window.confirm(
        `"${asset.name}" is still referenced by ${result.blockers.length} asset(s):\n\n${names}\n\nDelete anyway? References will break.`,
      );
      if (!ok) return;
      const forced = await store.deleteAsset(asset.id, true);
      if (forced.error) pushStatus(`Delete failed: ${forced.error}`, 8000);
      else if (!forced.deleted) pushStatus(`"${asset.name}" was not deleted.`, 8000);
    }
  };
  const showRefs = async () => {
    try {
      const refs = await import("../lib/ipc").then((m) =>
        m.assets.references(asset.id),
      );
      if (refs.length === 0)
        pushStatus(`"${asset.name}" is not referenced by anything.`);
      else pushStatus(`${asset.name} ← ${refs.map((r) => r.name).join(", ")}`);
    } catch (e) {
      pushStatus(`Could not read references: ${String(e)}`, 8000);
    }
  };
  // The asset's absolute payload path = content root + its content-relative path.
  const doReveal = async () => {
    try {
      await shell.reveal(`${rootPath}/${asset.path}`);
    } catch (e) {
      pushStatus(`Reveal failed: ${String(e)}`);
    }
  };
  const applyToSelection = async () => {
    try {
      const n = await sceneIpc.applyMaterial(asset.id);
      pushStatus(
        n > 0
          ? `Applied "${asset.name}" to ${n} actor(s).`
          : "Select an actor with a material first, then apply.",
      );
    } catch (e) {
      pushStatus(`Apply failed: ${String(e)}`, 8000);
    }
  };
  const createInstance = async () => {
    try {
      const { assets } = await import("../lib/ipc");
      await assets.createMaterialInstance(asset.id);
      pushStatus(`Created an instance of "${asset.name}".`);
    } catch (e) {
      pushStatus(`Could not create an instance: ${String(e)}`, 8000);
    }
  };
  const isMaterialLike =
    asset.kind === "material" || asset.kind === "material_instance";
  const bindToSelection = async () => {
    try {
      const n = await sceneIpc.applyActor(asset.id);
      pushStatus(
        n > 0
          ? `Bound "${asset.name}" to ${n} entity(ies).`
          : "Select an entity first, then bind the blueprint.",
      );
    } catch (e) {
      pushStatus(`Bind failed: ${String(e)}`, 8000);
    }
  };
  // ── collections (E-P8) ────────────────────────────────────────────────────
  // The status has to follow the ANSWER: the store catches internally, so an
  // unconditional "Added ..." stated a membership that did not exist.
  const addTo = async (name: string) => {
    const ok = await addToCollection(name, asset.id);
    pushStatus(
      ok
        ? `Added "${asset.name}" to "${name}".`
        : `Could not add "${asset.name}" to "${name}".`,
      ok ? undefined : 8000,
    );
    onClose();
  };
  const newCollectionAndAdd = async () => {
    const name = window.prompt("New collection name");
    if (!name || !name.trim()) return;
    try {
      await createCollection(name.trim());
      await addTo(name.trim());
    } catch (e) {
      window.alert(String(e));
    }
  };
  const removeFrom = async () => {
    if (!activeCollection) return;
    const ok = await removeFromCollection(activeCollection, asset.id);
    pushStatus(
      ok
        ? `Removed "${asset.name}" from "${activeCollection}".`
        : `Could not remove "${asset.name}" from "${activeCollection}".`,
      ok ? undefined : 8000,
    );
  };

  return (
    <div
      ref={menuRef}
      {...{ [CUTOUT_ATTR]: "" }}
      className="fixed z-50 min-w-44 overflow-hidden rounded border border-(--ink-border) bg-(--ink-bg-2) py-1 shadow-lg"
      style={{ left: x, top: y }}
      onClick={(e) => e.stopPropagation()}
    >
      {isMaterialLike &&
        item(
          "Apply to Selection",
          <Paintbrush size={13} />,
          () => void applyToSelection(),
        )}
      {asset.kind === "material_instance" &&
        item("Edit Instance", <Pencil size={13} />, () =>
          openInstanceEditor(asset.id),
        )}
      {asset.kind === "blueprint" &&
        item(
          "Bind to Selection",
          <Link2 size={13} />,
          () => void bindToSelection(),
        )}
      {asset.kind === "material" &&
        item(
          "Create Instance",
          <Copy size={13} />,
          () => void createInstance(),
        )}
      {asset.kind === "mesh" &&
        item("Edit Mesh", <Pencil size={13} />, () =>
          useDockLayout.getState().openPanel("model", asset.id),
        )}
      {asset.kind === "skeleton" &&
        item("Edit Skeleton", <Bone size={13} />, () =>
          useDockLayout.getState().openPanel("skeleton", asset.id),
        )}
      {item("Rename", <Pencil size={13} />, doRename)}
      {item("Duplicate", <Copy size={13} />, doDuplicate)}
      {item("Show References", <Link2 size={13} />, () => void showRefs())}
      {item(
        "Reveal in Explorer",
        <FolderSearch size={13} />,
        () => void doReveal(),
      )}
      <div className="my-1 h-px bg-(--ink-border)" />
      {/* Add to Collection ▸ (E-P8): a click-to-expand inline submenu. */}
      <button
        className="flex w-full items-center gap-2 px-2.5 py-1 text-left text-xs hover:bg-(--ink-bg-3)"
        onClick={(e) => {
          e.stopPropagation();
          setSubOpen((v) => !v);
        }}
      >
        <FolderPlus size={13} /> Add to Collection
        <ChevronRight
          size={12}
          className={cn("ml-auto transition-transform", subOpen && "rotate-90")}
        />
      </button>
      {subOpen && (
        <div className="border-y border-(--ink-border) bg-(--ink-bg-1) py-0.5">
          {collections.map((c) => (
            <button
              key={c.name}
              className="flex w-full items-center gap-2 px-2.5 py-1 pl-6 text-left text-xs hover:bg-(--ink-bg-3)"
              onClick={() => void addTo(c.name)}
            >
              <Library size={12} className="text-(--ink-accent)" /> {c.name}
            </button>
          ))}
          <button
            className="flex w-full items-center gap-2 px-2.5 py-1 pl-6 text-left text-xs text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
            onClick={() => void newCollectionAndAdd()}
          >
            <Plus size={12} /> New Collection…
          </button>
        </div>
      )}
      {activeCollection &&
        item(
          "Remove from Collection",
          <Minus size={13} />,
          () => void removeFrom(),
        )}
      <div className="my-1 h-px bg-(--ink-border)" />
      {item("Delete", <Trash2 size={13} />, () => void doDelete(), true)}
    </div>
  );
}
