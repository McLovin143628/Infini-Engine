/**
 * Biome-set editor (P19.2): the inline `.inf_biomes` editor.
 *
 * Rendered INSIDE the Content Drawer's content area, like `DataAssetEditor` and
 * `MaterialInstanceEditor` — the airspace rule means HTML over the native
 * viewport hole is occluded, and the drawer is safe ground below it.
 *
 * One row per biome: the id (read-only — it is the value stored per terrain
 * sample, so renumbering repaints the world), a name, the overlay colour, the
 * splat layer it shades as, the `.inf_pcg` graph it scatters with (the P19.3
 * hook) and the two advisory hints. Edits mutate a local draft; nothing reaches
 * disk until Save.
 *
 * **Saving validates on the backend** (`BiomeSet::validate`), and the message it
 * refuses with is surfaced verbatim in the status bar rather than swallowed —
 * a rejected save leaves the file untouched, so the draft is still the only copy
 * of the author's work.
 */
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Palette, Plus, Save, Trash2 } from "lucide-react";

import { cssHexToLinear, linearToCssHex } from "../lib/biomeColor";
import { cn } from "../lib/utils";
import { assets as assetsIpc } from "../lib/ipc";
import type { BiomeDefDto, BiomeSetDto } from "../lib/ipc";
import { useAssetStore } from "../stores/assetStore";
import { useShellStore } from "../stores/shellStore";

/** Number of splat layers a terrain has (mirrors `inf_terrain::SPLAT_LAYERS`). */
const SPLAT_LAYERS = 4;

const inputCls =
  "h-6 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1.5 text-xs outline-none focus:border-(--ink-accent)";
const selectCls =
  "h-6 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1 text-xs outline-none focus:border-(--ink-accent)";
const btnCls = "flex h-6 items-center gap-1 rounded px-2 text-xs hover:bg-(--ink-bg-3)";
const headCls = "px-1 pb-1 text-left text-[10px] font-semibold uppercase text-(--ink-text-faint)";

/** The lowest unused id in `1..=255`, or `null` when the set is full. The client
 *  mirror of `BiomeSet::next_free_id` — ids are never recycled by accident. */
function nextFreeId(used: readonly number[]): number | null {
  const taken = new Set(used);
  for (let id = 1; id <= 255; id++) if (!taken.has(id)) return id;
  return null;
}

/**
 * Why this draft cannot be saved, or `null`. The client mirror of the rules
 * `BiomeSet::validate` enforces — the backend is still the authority (it refuses
 * the write), this only stops the round-trip and names the reason on the button.
 */
function draftError(biomes: readonly BiomeDefDto[]): string | null {
  const seen = new Set<number>();
  for (const b of biomes) {
    if (b.id === 0) return "Biome id 0 is reserved for “unassigned” and cannot be defined.";
    if (seen.has(b.id)) return `Duplicate biome id ${b.id} — ids are the per-sample identity.`;
    seen.add(b.id);
    if (!b.name.trim()) return `Biome ${b.id} has an empty name.`;
  }
  return null;
}

export default function BiomeSetEditor({ id, onClose }: { id: string; onClose: () => void }) {
  const pushStatus = useShellStore((s) => s.pushStatus);
  const [dto, setDto] = useState<BiomeSetDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  // The `let cancelled` + `void (async () => …)()` shape is the mandatory dodge
  // for react-hooks/set-state-in-effect: state is only ever touched in an async
  // continuation, and a unmount before the await resolves drops the result.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const d = await assetsIpc.getBiomeSet(id);
        if (cancelled) return;
        setDto(d);
        setDirty(false);
      } catch (e) {
        console.error("assets.getBiomeSet failed", e);
        if (!cancelled) setDto(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id]);

  // Both are O(rows) over a handful of entries, so they run inline rather than
  // through a `useMemo` whose dependency would be a fresh array every render.
  const biomes = dto?.biomes ?? [];
  const invalid = draftError(biomes);
  const free = nextFreeId(biomes.map((b) => b.id));

  const setBiomes = (next: BiomeDefDto[]) => {
    setDto((d) => (d ? { ...d, biomes: next } : d));
    setDirty(true);
  };
  const update = (i: number, patch: Partial<BiomeDefDto>) =>
    setBiomes(biomes.map((b, n) => (n === i ? { ...b, ...patch } : b)));

  const addBiome = () => {
    if (free === null) return;
    setBiomes([
      ...biomes,
      {
        id: free,
        name: `Biome ${free}`,
        color: [0.5, 0.5, 0.5, 1],
        splat_layer: null,
        pcg_graph: null,
        water_hint: null,
        structure_hint: null,
      },
    ]);
  };

  const save = async () => {
    if (!dto || invalid) return;
    setSaving(true);
    try {
      await assetsIpc.saveBiomeSet(id, dto);
      setDirty(false);
      pushStatus(`Saved “${dto.name}”.`);
    } catch (e) {
      // The backend's validation message is the useful part; keep the draft so
      // nothing the author typed is lost to a refused write.
      pushStatus(`Save failed: ${String(e)}`, 8000);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <div className="p-4 text-xs text-(--ink-text-faint)">Loading biome set…</div>;
  }
  if (!dto) {
    return (
      <div className="flex flex-col gap-2 p-4 text-xs text-(--ink-text-faint)">
        <span>Not a biome set (or it failed to decode).</span>
        <button className={cn(btnCls, "self-start")} onClick={onClose}>
          <ArrowLeft size={13} /> Back
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex h-9 shrink-0 items-center gap-2 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
        <button className={btnCls} onClick={onClose} title="Back to content">
          <ArrowLeft size={13} /> Back
        </button>
        <div className="flex items-center gap-1.5 text-xs">
          <Palette size={13} className="text-(--ink-success)" />
          <span className="font-semibold">Biome Set</span>
          <span className="text-(--ink-text-dim)" title="Rename this asset from the content grid">
            — {dto.name}
          </span>
        </div>
        <div className="flex-1" />
        {invalid && <span className="truncate text-[11px] text-(--ink-danger)">{invalid}</span>}
        <button
          className={cn(
            btnCls,
            dirty &&
              !invalid &&
              "bg-(--ink-accent) text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)",
          )}
          onClick={() => void save()}
          disabled={!dirty || saving || invalid !== null}
          title={invalid ?? "Save this biome set"}
        >
          <Save size={13} /> {saving ? "Saving…" : "Save"}
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-2">
        <table className="border-separate border-spacing-x-1 border-spacing-y-0.5 text-xs">
          <thead>
            <tr>
              <th className={headCls}>Id</th>
              <th className={headCls}>Name</th>
              <th className={headCls}>Colour</th>
              <th className={headCls}>Splat</th>
              <th className={headCls}>PCG Graph</th>
              <th className={headCls}>Water (m)</th>
              <th className={headCls}>Structure</th>
              <th className={headCls} />
            </tr>
          </thead>
          <tbody>
            {biomes.map((b, i) => (
              <tr key={b.id}>
                <td>
                  {/* Read-only: the id IS the value stored per terrain sample, so
                      renumbering would repaint the world rather than rename it. */}
                  <span
                    className="inline-flex h-6 min-w-7 items-center justify-center rounded bg-(--ink-bg-3) px-1.5 tabular-nums text-(--ink-text-dim)"
                    title="Stable id stored per terrain sample — renaming is free, renumbering is not"
                  >
                    {b.id}
                  </span>
                </td>
                <td>
                  <input
                    className={cn(inputCls, "w-40")}
                    value={b.name}
                    onChange={(e) => update(i, { name: e.target.value })}
                    spellCheck={false}
                    aria-label={`Biome ${b.id} name`}
                  />
                </td>
                <td>
                  <input
                    type="color"
                    className="h-6 w-9 rounded border border-(--ink-border) bg-transparent"
                    value={linearToCssHex(b.color)}
                    onChange={(e) =>
                      update(i, { color: cssHexToLinear(e.target.value, b.color[3] ?? 1) })
                    }
                    title="Overlay swatch — what the Biomes view mode tints this biome"
                    aria-label={`Biome ${b.id} colour`}
                  />
                </td>
                <td>
                  <select
                    className={selectCls}
                    value={b.splat_layer ?? ""}
                    onChange={(e) =>
                      update(i, {
                        splat_layer: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                    title="Which terrain splat layer this biome shades as (advisory — painting a biome does not paint weights)"
                    aria-label={`Biome ${b.id} splat layer`}
                  >
                    <option value="">(none)</option>
                    {Array.from({ length: SPLAT_LAYERS }, (_, n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </td>
                <td>
                  <PcgGraphPicker
                    value={b.pcg_graph}
                    onChange={(pcg) => update(i, { pcg_graph: pcg })}
                    label={`Biome ${b.id} PCG graph`}
                  />
                </td>
                <td>
                  <input
                    type="number"
                    step={0.5}
                    className={cn(inputCls, "w-20 tabular-nums")}
                    value={b.water_hint ?? ""}
                    placeholder="—"
                    onChange={(e) =>
                      update(i, {
                        water_hint: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                    title="Still-water level in metres of absolute world height (advisory; P20 reads it)"
                    aria-label={`Biome ${b.id} water hint`}
                  />
                </td>
                <td>
                  <input
                    className={cn(inputCls, "w-32")}
                    value={b.structure_hint ?? ""}
                    placeholder="—"
                    onChange={(e) =>
                      update(i, { structure_hint: e.target.value === "" ? null : e.target.value })
                    }
                    spellCheck={false}
                    title="Building-palette name (advisory; P19.5 reads it)"
                    aria-label={`Biome ${b.id} structure hint`}
                  />
                </td>
                <td>
                  <button
                    className="flex size-6 items-center justify-center rounded text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-danger)"
                    onClick={() => setBiomes(biomes.filter((_, n) => n !== i))}
                    aria-label={`Remove biome ${b.id}`}
                    title="Remove this biome — terrain already painted with its id reads as unassigned"
                  >
                    <Trash2 size={13} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {biomes.length === 0 && (
          <div className="p-2 text-xs text-(--ink-text-faint)">
            No biomes yet. Add one, then bind this set to a terrain from the viewport toolbar.
          </div>
        )}

        <button
          className={cn(btnCls, "mt-2")}
          onClick={addBiome}
          disabled={free === null}
          title={
            free === null
              ? "This set already holds all 255 ids."
              : `Add a biome with the next free id (${free})`
          }
        >
          <Plus size={13} /> Add Biome
        </button>
      </div>
    </div>
  );
}

/** Asset-ref picker scoped to `.inf_pcg` graphs — the P19.3 scatter binding. */
function PcgGraphPicker({
  value,
  onChange,
  label,
}: {
  value: string | null;
  onChange: (id: string | null) => void;
  label: string;
}) {
  const assets = useAssetStore((s) => s.assets);
  const options = useMemo(
    () =>
      Object.values(assets)
        .filter((a) => a.kind === "pcg")
        .sort((a, b) => a.name.localeCompare(b.name)),
    [assets],
  );
  return (
    <select
      className={selectCls}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
      title="The .inf_pcg graph this biome scatters with (a real dependency edge)"
      aria-label={label}
    >
      <option value="">(none)</option>
      {/* A graph the project no longer has still shows, so a stale binding is
          visible rather than silently reset to (none) on the next save. */}
      {value !== null && !options.some((a) => a.id === value) && (
        <option value={value}>(missing asset)</option>
      )}
      {options.map((a) => (
        <option key={a.id} value={a.id}>
          {a.name}
        </option>
      ))}
    </select>
  );
}
