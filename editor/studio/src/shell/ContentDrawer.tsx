/**
 * Content Drawer shell (P1.4.3): slide-up surface with Favorites + folder
 * tree | filter chips | breadcrumbs + thumbnail grid, mock assets. The
 * real asset backbone arrives in Phase 4 (virtualized grid, drag-drop to
 * viewport, collections).
 *
 * Layout note (airspace rule): the drawer does NOT overlay the workspace —
 * it inserts BELOW it and pushes it up, because HTML can never draw over
 * the native viewport hole. The viewport rect-sync shrinks the engine
 * child window instead, which reads identically to UE's overlay drawer.
 */
import { useMemo, useState } from "react";
import {
  Box,
  ChevronRight,
  FileCode2,
  Folder,
  FolderOpen,
  Image,
  Import,
  Music2,
  Plus,
  Save,
  Search,
  Star,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { cn } from "../lib/utils";
import { fuzzyMatch } from "../lib/fuzzy";
import { executeCommand } from "../lib/commands";
import { useShellStore } from "../stores/shellStore";

interface MockAsset {
  name: string;
  kind: "mesh" | "texture" | "audio" | "blueprint";
  folder: string;
}

const KIND_META: Record<MockAsset["kind"], { icon: LucideIcon; tint: string; label: string }> = {
  mesh: { icon: Box, tint: "text-(--ink-info)", label: "Static Mesh" },
  texture: { icon: Image, tint: "text-(--ink-success)", label: "Texture" },
  audio: { icon: Music2, tint: "text-(--ink-warning)", label: "Sound" },
  blueprint: { icon: FileCode2, tint: "text-(--ink-accent)", label: "Blueprint" },
};

const FOLDERS = ["All", "Meshes", "Textures", "Audio", "Blueprints"] as const;

const MOCK_ASSETS: MockAsset[] = [
  { name: "SM_Cube", kind: "mesh", folder: "Meshes" },
  { name: "SM_Sphere", kind: "mesh", folder: "Meshes" },
  { name: "SM_Rock_01", kind: "mesh", folder: "Meshes" },
  { name: "SM_Tree_Pine", kind: "mesh", folder: "Meshes" },
  { name: "T_Grass_D", kind: "texture", folder: "Textures" },
  { name: "T_Rock_N", kind: "texture", folder: "Textures" },
  { name: "T_Sky_HDR", kind: "texture", folder: "Textures" },
  { name: "S_Footstep_01", kind: "audio", folder: "Audio" },
  { name: "S_Ambience_Wind", kind: "audio", folder: "Audio" },
  { name: "BP_Door", kind: "blueprint", folder: "Blueprints" },
  { name: "BP_PickupCoin", kind: "blueprint", folder: "Blueprints" },
  { name: "BP_PlayerCharacter", kind: "blueprint", folder: "Blueprints" },
];

export default function ContentDrawer() {
  const open = useShellStore((s) => s.drawerOpen);
  const setOpen = useShellStore((s) => s.setDrawerOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);
  const [folder, setFolder] = useState<(typeof FOLDERS)[number]>("All");
  const [kindFilter, setKindFilter] = useState<MockAsset["kind"] | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const assets = useMemo(
    () =>
      MOCK_ASSETS.filter(
        (a) =>
          (folder === "All" || a.folder === folder) &&
          (kindFilter === null || a.kind === kindFilter) &&
          (!search.trim() || fuzzyMatch(search.trim(), a.name) !== null),
      ),
    [folder, kindFilter, search],
  );

  return (
    <div
      className={cn(
        "flex shrink-0 flex-col overflow-hidden border-t border-(--ink-border) bg-(--ink-bg-1) transition-[height] duration-150 ease-out",
      )}
      style={{ height: open ? 300 : 0 }}
      aria-hidden={!open}
    >
      {/* Top bar: actions + breadcrumbs + search */}
      <div className="flex h-9 shrink-0 items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
        <button
          className="flex h-6 items-center gap-1 rounded bg-(--ink-accent) px-2 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)"
          onClick={() => pushStatus("Asset creation arrives with Phase 4 (asset system).")}
        >
          <Plus size={13} /> Add
        </button>
        <button
          className="flex h-6 items-center gap-1 rounded px-2 text-xs hover:bg-(--ink-bg-3)"
          onClick={() => executeCommand("file.importIntoLevel")}
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
        <div className="flex min-w-0 items-center gap-0.5 text-xs text-(--ink-text-dim)">
          <span className="hover:text-(--ink-text)">Content</span>
          {folder !== "All" && (
            <>
              <ChevronRight size={12} />
              <span className="text-(--ink-text)">{folder}</span>
            </>
          )}
        </div>
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
          className="flex h-6 items-center gap-1 rounded px-2 text-xs text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
          onClick={() => pushStatus("Dock-in-layout for the Content Browser arrives with Phase 4.")}
        >
          Dock in Layout
        </button>
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
        <div className="w-44 shrink-0 overflow-auto border-r border-(--ink-border) p-1.5">
          <div className="flex items-center gap-1 px-1 pb-1 text-[11px] font-semibold text-(--ink-text-dim)">
            <Star size={11} /> Favorites
          </div>
          <div className="px-1 pb-2 text-[11px] text-(--ink-text-faint)">None yet</div>
          <div className="px-1 pb-1 text-[11px] font-semibold text-(--ink-text-dim)">Content</div>
          {FOLDERS.map((f) => (
            <button
              key={f}
              className={cn(
                "flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-xs",
                folder === f ? "bg-(--ink-selection)" : "hover:bg-(--ink-bg-3)",
              )}
              onClick={() => setFolder(f)}
            >
              {folder === f ? (
                <FolderOpen size={13} className="text-(--ink-warning)" />
              ) : (
                <Folder size={13} className="text-(--ink-warning)" />
              )}
              {f}
            </button>
          ))}
        </div>

        {/* Filter column */}
        <div className="w-28 shrink-0 overflow-auto border-r border-(--ink-border) p-1.5">
          <div className="px-1 pb-1 text-[11px] font-semibold text-(--ink-text-dim)">Filters</div>
          {(Object.keys(KIND_META) as MockAsset["kind"][]).map((kind) => (
            <button
              key={kind}
              className={cn(
                "flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-xs capitalize",
                kindFilter === kind ? "bg-(--ink-selection)" : "hover:bg-(--ink-bg-3)",
              )}
              onClick={() => setKindFilter((k) => (k === kind ? null : kind))}
            >
              {kind}
            </button>
          ))}
        </div>

        {/* Thumbnail grid (virtualization arrives with the 100k-asset DB, P4.4) */}
        <div className="min-h-0 flex-1 overflow-auto p-2">
          {assets.length === 0 ? (
            <div className="p-4 text-xs text-(--ink-text-faint)">No assets match.</div>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(92px,1fr))] gap-2">
              {assets.map((a) => {
                const meta = KIND_META[a.kind];
                const Icon = meta.icon;
                return (
                  <button
                    key={a.name}
                    className={cn(
                      "group flex flex-col items-stretch rounded border p-1 text-left",
                      selected === a.name
                        ? "border-(--ink-accent) bg-(--ink-accent-muted)"
                        : "border-(--ink-border) bg-(--ink-bg-2) hover:border-(--ink-border-strong)",
                    )}
                    onClick={() => setSelected(a.name)}
                    onDoubleClick={() =>
                      useShellStore
                        .getState()
                        .pushStatus(`Asset editors arrive with their phase (${meta.label}).`)
                    }
                  >
                    <div className="flex aspect-square items-center justify-center rounded-sm bg-(--ink-bg-0)">
                      <Icon size={30} className={meta.tint} />
                    </div>
                    <span className="mt-1 truncate text-[11px]">{a.name}</span>
                    <span className="truncate text-[10px] text-(--ink-text-faint)">
                      {meta.label}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
