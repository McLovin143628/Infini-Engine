/**
 * Asset-reference row (P26.3b), **read-only**.
 *
 * A component field of kind `asset_ref` — today only `Material.asset`, the
 * `.inf_mat` binding scene v22 persists — renders as the referenced asset's
 * name, or "None" when the surface is scalars-only (the permanent no-texture
 * path). Hovering shows the GUID; an unresolvable GUID reads "(missing)", which
 * is a real state: a level can name a material the project no longer has, and a
 * row that hid that would hide exactly the case the cook raises an advisory for.
 *
 * **There is no picker yet, and that is deliberate rather than forgotten.**
 * Binding is done by dragging a `.inf_mat` from the Content Drawer onto the
 * entity (or "Apply to Selection"), which is `scene_apply_material` — the one
 * call site that knows both the parameters and the asset they came from. A
 * picker here would need a second write path for the same fact. It is the
 * standing gap named on `PropValueDto::AssetRef`, alongside `skel_merge_part`'s
 * and the Model Editor rig field's.
 */
import { FileImage, FileX2 } from "lucide-react";
import { useAssetStore } from "../stores/assetStore";

export function AssetRefField({
  value,
  assetKind,
}: {
  /** The referenced asset GUID, or `null` when unbound. */
  value: string | null;
  /** `AssetKind::slug` of what this row accepts (`"material"`). */
  assetKind: string;
}) {
  const entry = useAssetStore((s) => (value ? s.assets[value] : undefined));

  if (!value) {
    return (
      <span className="flex h-6 min-w-0 flex-1 items-center px-1.5 text-xs text-(--ink-text-faint)">
        None
      </span>
    );
  }

  const missing = !entry;
  const label = entry?.name ?? "(missing)";

  return (
    <span
      className="flex h-6 min-w-0 flex-1 items-center gap-1 px-1.5 text-xs"
      title={`${value} (${assetKind})`}
    >
      {missing ? (
        <FileX2 size={11} className="shrink-0 text-(--ink-warning)" />
      ) : (
        <FileImage size={11} className="shrink-0 text-(--ink-text-faint)" />
      )}
      <span
        className={`min-w-0 flex-1 truncate ${missing ? "text-(--ink-warning)" : "text-(--ink-text-dim)"}`}
      >
        {label}
      </span>
    </span>
  );
}
