/**
 * Data-asset editor (P4.5): the inline struct / enum / table editor.
 *
 * Rendered INSIDE the Content Drawer's content area (not a screen modal) — the
 * airspace rule means HTML over the native viewport hole is occluded, and the
 * drawer area is safe HTML below it. Edits mutate a local draft; nothing is
 * persisted until Save (except CSV/JSON table import, which writes server-side
 * then reloads).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, Code, Import, Plus, Save, Trash2, X } from "lucide-react";

import { cn } from "../lib/utils";
import type { DataAssetDto, DataFieldDto } from "../lib/ipc";
import { assets as assetsIpc } from "../lib/ipc";
import { useAssetStore } from "../stores/assetStore";

const FIELD_TYPES: { value: string; label: string }[] = [
  { value: "bool", label: "Bool" },
  { value: "int", label: "Int" },
  { value: "float", label: "Float" },
  { value: "text", label: "Text" },
  { value: "vec3", label: "Vec3" },
  { value: "color", label: "Color" },
  { value: "asset_ref", label: "Asset Ref" },
  { value: "enum", label: "Enum" },
];

function newField(): DataFieldDto {
  return { name: "field", type: "text", enum_id: null, enum_name: null };
}

const inputCls =
  "h-6 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1.5 text-xs outline-none focus:border-(--ink-accent)";
const selectCls =
  "h-6 rounded border border-(--ink-border) bg-(--ink-bg-1) px-1 text-xs outline-none focus:border-(--ink-accent)";
const btnCls = "flex h-6 items-center gap-1 rounded px-2 text-xs hover:bg-(--ink-bg-3)";

export default function DataAssetEditor({ id, onClose }: { id: string; onClose: () => void }) {
  const [draft, setDraft] = useState<DataAssetDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [rust, setRust] = useState<string | null>(null);

  // Re-fetch the payload (used on mount and after a table import). Only touches
  // state in async continuations, so it never sets state synchronously.
  const load = useCallback(async () => {
    try {
      setDraft(await assetsIpc.data(id));
    } catch (e) {
      console.error("asset.data failed", e);
      setDraft(null);
    }
  }, [id]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      await load();
      if (!cancelled) setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [load]);

  const save = async () => {
    if (!draft) return;
    setSaving(true);
    try {
      await assetsIpc.dataSave(draft);
      onClose();
    } catch (e) {
      console.error("asset.dataSave failed", e);
    } finally {
      setSaving(false);
    }
  };

  const toggleRust = async () => {
    if (rust !== null) {
      setRust(null);
      return;
    }
    try {
      setRust(await assetsIpc.rustSource(id));
    } catch (e) {
      console.error("asset.rustSource failed", e);
    }
  };

  if (loading) {
    return <div className="p-4 text-xs text-(--ink-text-faint)">Loading…</div>;
  }
  if (!draft) {
    return (
      <div className="flex flex-col gap-2 p-4 text-xs text-(--ink-text-faint)">
        <span>Not a data asset.</span>
        <button className={cn(btnCls, "self-start")} onClick={onClose}>
          <X size={13} /> Close
        </button>
      </div>
    );
  }

  const isData = draft.kind === "struct" || draft.kind === "enum";

  return (
    <div className="flex h-full flex-col">
      {/* Editor header */}
      <div className="flex h-8 shrink-0 items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
        <button className={btnCls} onClick={onClose} title="Back to grid">
          <ArrowLeft size={13} />
        </button>
        <span className="text-[11px] uppercase tracking-wide text-(--ink-text-faint)">
          {draft.kind}
        </span>
        <input
          className={cn(inputCls, "w-56")}
          value={draft.name}
          onChange={(e) => setDraft({ ...draft, name: e.target.value })}
          spellCheck={false}
        />
        <div className="flex-1" />
        {isData && (
          <button
            className={cn(btnCls, rust !== null && "bg-(--ink-bg-3)")}
            onClick={() => void toggleRust()}
          >
            <Code size={13} /> View Rust
          </button>
        )}
        <button
          className="flex h-6 items-center gap-1 rounded bg-(--ink-accent) px-2 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover) disabled:opacity-50"
          onClick={() => void save()}
          disabled={saving}
        >
          <Save size={13} /> Save
        </button>
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="min-h-0 flex-1 overflow-auto p-2">
          {draft.kind === "struct" && <StructEditor draft={draft} setDraft={setDraft} />}
          {draft.kind === "enum" && <EnumEditor draft={draft} setDraft={setDraft} />}
          {draft.kind === "table" && (
            <TableEditor draft={draft} setDraft={setDraft} id={id} reload={load} />
          )}
        </div>
        {rust !== null && (
          <div className="flex w-2/5 min-w-64 flex-col border-l border-(--ink-border)">
            <div className="flex h-6 items-center border-b border-(--ink-border) bg-(--ink-bg-2) px-2 text-[11px] text-(--ink-text-dim)">
              Generated Rust
            </div>
            <pre className="min-h-0 flex-1 overflow-auto bg-(--ink-bg-0) p-2 font-mono text-[11px] leading-relaxed text-(--ink-text)">
              {rust}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ── struct ─────────────────────────────────────────────────────────────────

function StructEditor({
  draft,
  setDraft,
}: {
  draft: DataAssetDto;
  setDraft: (d: DataAssetDto) => void;
}) {
  const setFields = (fields: DataFieldDto[]) => setDraft({ ...draft, fields });
  const update = (i: number, patch: Partial<DataFieldDto>) =>
    setFields(draft.fields.map((f, n) => (n === i ? { ...f, ...patch } : f)));

  return (
    <div className="flex flex-col gap-1">
      {draft.fields.length === 0 && (
        <div className="pb-1 text-xs text-(--ink-text-faint)">No fields yet.</div>
      )}
      {draft.fields.map((f, i) => (
        <div key={i} className="flex items-center gap-1.5">
          <input
            className={cn(inputCls, "w-48")}
            value={f.name}
            onChange={(e) => update(i, { name: e.target.value })}
            spellCheck={false}
          />
          <select
            className={selectCls}
            value={f.type}
            onChange={(e) =>
              update(i, {
                type: e.target.value,
                enum_id: e.target.value === "enum" ? f.enum_id : null,
                enum_name: e.target.value === "enum" ? f.enum_name : null,
              })
            }
          >
            {FIELD_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
          {f.type === "enum" && (
            <AssetRefPicker
              kindFilter="enum"
              value={f.enum_id}
              onChange={(id, name) => update(i, { enum_id: id, enum_name: name })}
            />
          )}
          <button
            className="flex size-6 items-center justify-center rounded text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-danger)"
            onClick={() => setFields(draft.fields.filter((_, n) => n !== i))}
            aria-label="Remove field"
          >
            <Trash2 size={13} />
          </button>
        </div>
      ))}
      <button
        className={cn(btnCls, "mt-1 self-start")}
        onClick={() => setFields([...draft.fields, newField()])}
      >
        <Plus size={13} /> Add Field
      </button>
    </div>
  );
}

// ── enum ───────────────────────────────────────────────────────────────────

function EnumEditor({
  draft,
  setDraft,
}: {
  draft: DataAssetDto;
  setDraft: (d: DataAssetDto) => void;
}) {
  const setVariants = (variants: string[]) => setDraft({ ...draft, variants });
  return (
    <div className="flex flex-col gap-1">
      {draft.variants.length === 0 && (
        <div className="pb-1 text-xs text-(--ink-text-faint)">No variants yet.</div>
      )}
      {draft.variants.map((v, i) => (
        <div key={i} className="flex items-center gap-1.5">
          <input
            className={cn(inputCls, "w-56")}
            value={v}
            onChange={(e) => setVariants(draft.variants.map((x, n) => (n === i ? e.target.value : x)))}
            spellCheck={false}
          />
          <button
            className="flex size-6 items-center justify-center rounded text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-danger)"
            onClick={() => setVariants(draft.variants.filter((_, n) => n !== i))}
            aria-label="Remove variant"
          >
            <Trash2 size={13} />
          </button>
        </div>
      ))}
      <button
        className={cn(btnCls, "mt-1 self-start")}
        onClick={() => setVariants([...draft.variants, `Variant${draft.variants.length + 1}`])}
      >
        <Plus size={13} /> Add Variant
      </button>
    </div>
  );
}

// ── table ──────────────────────────────────────────────────────────────────

function TableEditor({
  draft,
  setDraft,
  id,
  reload,
}: {
  draft: DataAssetDto;
  setDraft: (d: DataAssetDto) => void;
  id: string;
  reload: () => Promise<void>;
}) {
  const setCols = (fields: DataFieldDto[]) => setDraft({ ...draft, fields });
  const setRows = (rows: string[][]) => setDraft({ ...draft, rows });

  const updateCol = (i: number, patch: Partial<DataFieldDto>) =>
    setCols(draft.fields.map((c, n) => (n === i ? { ...c, ...patch } : c)));

  const addColumn = () =>
    setDraft({
      ...draft,
      fields: [...draft.fields, { name: `Column${draft.fields.length + 1}`, type: "text", enum_id: null, enum_name: null }],
      rows: draft.rows.map((r) => [...r, ""]),
    });

  const removeColumn = (i: number) =>
    setDraft({
      ...draft,
      fields: draft.fields.filter((_, n) => n !== i),
      rows: draft.rows.map((r) => r.filter((_, n) => n !== i)),
    });

  const addRow = () => setRows([...draft.rows, draft.fields.map(() => "")]);
  const removeRow = (r: number) => setRows(draft.rows.filter((_, n) => n !== r));
  const setCell = (r: number, c: number, value: string) =>
    setRows(draft.rows.map((row, n) => (n === r ? row.map((v, m) => (m === c ? value : v)) : row)));

  const importTable = async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({
      multiple: false,
      filters: [{ name: "Table Data", extensions: ["csv", "json"] }],
    });
    if (!picked || Array.isArray(picked)) return;
    try {
      await assetsIpc.tableImport(id, picked);
      await reload();
    } catch (e) {
      console.error("asset.tableImport failed", e);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1">
        <button className={btnCls} onClick={addRow}>
          <Plus size={13} /> Row
        </button>
        <button className={btnCls} onClick={addColumn}>
          <Plus size={13} /> Column
        </button>
        <button className={btnCls} onClick={() => void importTable()}>
          <Import size={13} /> Import CSV/JSON…
        </button>
        <span className="ml-1 text-[11px] text-(--ink-text-faint)">
          {draft.rows.length} rows × {draft.fields.length} cols
        </span>
      </div>

      <div className="overflow-auto">
        <table className="border-collapse text-xs">
          <thead>
            <tr>
              <th className="w-6" />
              {draft.fields.map((c, i) => (
                <th key={i} className="p-0.5">
                  <div className="flex items-center gap-1">
                    <input
                      className={cn(inputCls, "w-28")}
                      value={c.name}
                      onChange={(e) => updateCol(i, { name: e.target.value })}
                      spellCheck={false}
                    />
                    <select
                      className={selectCls}
                      value={c.type}
                      onChange={(e) => updateCol(i, { type: e.target.value })}
                    >
                      {FIELD_TYPES.map((t) => (
                        <option key={t.value} value={t.value}>
                          {t.label}
                        </option>
                      ))}
                    </select>
                    <button
                      className="flex size-5 items-center justify-center rounded text-(--ink-text-dim) hover:text-(--ink-danger)"
                      onClick={() => removeColumn(i)}
                      aria-label="Remove column"
                    >
                      <X size={12} />
                    </button>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {draft.rows.map((row, r) => (
              <tr key={r}>
                <td className="align-middle">
                  <button
                    className="flex size-5 items-center justify-center rounded text-(--ink-text-dim) hover:text-(--ink-danger)"
                    onClick={() => removeRow(r)}
                    aria-label="Remove row"
                  >
                    <Trash2 size={12} />
                  </button>
                </td>
                {draft.fields.map((_, c) => (
                  <td key={c} className="p-0.5">
                    <input
                      className={cn(inputCls, "w-28")}
                      value={row[c] ?? ""}
                      onChange={(e) => setCell(r, c, e.target.value)}
                      spellCheck={false}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {draft.rows.length === 0 && (
          <div className="p-2 text-xs text-(--ink-text-faint)">
            No rows. Add a row or import CSV/JSON.
          </div>
        )}
      </div>
    </div>
  );
}

// ── asset-ref picker ─────────────────────────────────────────────────────────

function AssetRefPicker({
  value,
  kindFilter,
  onChange,
}: {
  value: string | null;
  kindFilter?: string;
  onChange: (id: string | null, name: string | null) => void;
}) {
  const assets = useAssetStore((s) => s.assets);
  const options = useMemo(
    () =>
      Object.values(assets)
        .filter((a) => (kindFilter ? a.kind === kindFilter : true))
        .sort((a, b) => a.name.localeCompare(b.name)),
    [assets, kindFilter],
  );
  return (
    <select
      className={selectCls}
      value={value ?? ""}
      onChange={(e) => {
        const pick = options.find((a) => a.id === e.target.value);
        onChange(pick?.id ?? null, pick?.name ?? null);
      }}
    >
      <option value="">(none)</option>
      {options.map((a) => (
        <option key={a.id} value={a.id}>
          {a.name}
        </option>
      ))}
    </select>
  );
}
