//! Material-instance override editing glue (E-P2).
//!
//! A `.inf_mati` ([`MaterialInstance`]) inherits a parent [`MaterialAsset`] and
//! overrides a sparse subset of its PBR parameters. The override editor needs
//! two things the raw payload does not hand it directly: the parent's **resolved**
//! values (to show grayed as the inherited baseline) and the parent's display
//! name. [`get_material_instance`] assembles both; [`save_material_instance`]
//! writes edited overrides back through the standard payload-rewrite path (so the
//! content hash + `assets://changed` invalidation — and thus the thumbnail — come
//! for free).

use inf_asset::{AssetError, AssetId, AssetKind, Result};
use inf_material::{MatOverrides, MaterialAsset, MaterialInstance};

use crate::assets::AssetProject;

/// The override editor's view of a material instance.
pub struct MaterialInstanceView {
    /// The parent material/instance GUID.
    pub parent: AssetId,
    /// The parent's display name.
    pub parent_name: String,
    /// The parent chain resolved to concrete PBR values — the inherited baseline
    /// each unset override falls back to.
    pub resolved_parent: MaterialAsset,
    /// This instance's sparse overrides (`None` = inherit).
    pub overrides: MatOverrides,
}

/// Resolve a material or material-instance asset to concrete PBR parameters,
/// following the instance→parent chain (depth-guarded against cycles). Mirrors
/// the Ring-2 apply-by-drag resolver so the editor's inherited baseline matches
/// what actually gets applied. `None` if the asset is missing or not a material.
pub fn resolve_material(project: &AssetProject, id: AssetId, depth: u32) -> Option<MaterialAsset> {
    if depth > 16 {
        return None; // pathological instance chain
    }
    match project.db().get(id)?.kind() {
        AssetKind::Material => project.load_payload::<MaterialAsset>(id).ok(),
        AssetKind::MaterialInstance => {
            let inst = project.load_payload::<MaterialInstance>(id).ok()?;
            let parent = resolve_material(project, inst.parent, depth + 1)?;
            Some(inst.resolve(&parent))
        }
        _ => None,
    }
}

/// The `.inf_mat` a **scene binding** (`Material.asset`, scene v22) must name,
/// given the asset an author actually applied — following the instance→parent
/// chain to its root material (P26.3b audit).
///
/// # Why a binding cannot name a `.inf_mati`
///
/// Apply-by-drag accepts a material **instance**: the Content Drawer routes
/// `kind === "material_instance"` to `scene_apply_material`, and the Material
/// Instance Editor has its own "Apply to Selection". Both resolve the instance
/// chain to concrete PBR values *before* applying, which is right — but P26.3b
/// then persisted the applied asset's GUID as the binding, and **nothing on any
/// path resolves an instance GUID**:
///
/// * the cook derives a `.inf_matd` only for `AssetKind::Material`, and closes
///   the `.inf_mat` → `.inf_tex` edge only for `AssetKind::Material`, so the
///   record a runtime looks up under `derived_material_id(instance)` does not
///   exist;
/// * the PIE payload builder's loader is kind-checked to the same four kinds and
///   `.inf_mati` is not one of them, so the binding resolves to `None`;
/// * neither raises an advisory, because the asset *is* in the project — so a
///   surface silently loses its maps on both wires at once, which is precisely
///   the state this batch exists to abolish.
///
/// Naming the ROOT is not a workaround, it is the correct answer:
/// [`MatOverrides`] has **no texture slots**, so an instance's maps *are* its
/// parent's. The scalars keep coming from the instance (apply-material flattens
/// the resolved chain onto the component, as it has since P7.4) and the textures
/// come from the material those scalars were resolved against — one surface, one
/// story.
///
/// `None` when the asset is missing, is not a material or an instance, or its
/// parent chain is broken/cyclic — the same depth guard [`resolve_material`]
/// uses, and the same outcome an author sees: no binding is written at all,
/// rather than one that resolves nowhere.
pub fn material_binding_root(project: &AssetProject, id: AssetId, depth: u32) -> Option<AssetId> {
    if depth > 16 {
        return None; // pathological instance chain
    }
    match project.db().get(id)?.kind() {
        AssetKind::Material => Some(id),
        AssetKind::MaterialInstance => {
            let inst = project.load_payload::<MaterialInstance>(id).ok()?;
            material_binding_root(project, inst.parent, depth + 1)
        }
        _ => None,
    }
}

/// Build the override-editor view for a material instance. Errors if `id` is
/// missing or is not a material instance.
pub fn get_material_instance(project: &AssetProject, id: AssetId) -> Result<MaterialInstanceView> {
    let kind = project
        .db()
        .get(id)
        .ok_or(AssetError::UnknownAsset(id))?
        .kind();
    if kind != AssetKind::MaterialInstance {
        return Err(AssetError::Import(format!(
            "asset {id} is not a material instance"
        )));
    }
    let inst: MaterialInstance = project.load_payload(id)?;
    let parent_name = project
        .db()
        .get(inst.parent)
        .map(|e| e.name.clone())
        .unwrap_or_else(|| "Material".to_string());
    // A missing / broken parent chain still yields an editable instance — fall
    // back to default PBR values as the inherited baseline.
    let resolved_parent = resolve_material(project, inst.parent, 0).unwrap_or_default();
    Ok(MaterialInstanceView {
        parent: inst.parent,
        parent_name,
        resolved_parent,
        overrides: inst.overrides,
    })
}

/// Persist edited overrides onto a material instance, re-encoding the payload +
/// sidecar through the standard rewrite path (updates the content hash, keeps the
/// parent dependency edge). Errors if `id` is not a material instance.
pub fn save_material_instance(
    project: &mut AssetProject,
    id: AssetId,
    overrides: MatOverrides,
) -> Result<()> {
    let kind = project
        .db()
        .get(id)
        .ok_or(AssetError::UnknownAsset(id))?
        .kind();
    if kind != AssetKind::MaterialInstance {
        return Err(AssetError::Import(format!(
            "asset {id} is not a material instance"
        )));
    }
    let mut inst: MaterialInstance = project.load_payload(id)?;
    inst.overrides = overrides;
    let deps = inst.dependencies();
    project.rewrite_payload(id, &inst, deps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_and_save_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let mats = proj.content_dir("materials").unwrap();

        // A parent material with distinctive PBR values.
        let parent = MaterialAsset {
            base_color: [0.2, 0.4, 0.6, 1.0],
            metallic: 1.0,
            roughness: 0.25,
            emissive: [0.1, 0.0, 0.0],
            ..Default::default()
        };
        let parent_id = proj
            .write_asset(&mats, "Base", &parent, None, vec![], None)
            .unwrap();
        let inst_id = proj
            .write_asset(
                &mats,
                "Inst",
                &MaterialInstance::new(parent_id),
                None,
                vec![parent_id],
                None,
            )
            .unwrap();

        // Fresh instance inherits everything; the resolved baseline == parent.
        let view = get_material_instance(&proj, inst_id).unwrap();
        assert_eq!(view.parent, parent_id);
        assert_eq!(view.parent_name, "Base");
        assert_eq!(view.resolved_parent.roughness, 0.25);
        assert!(view.overrides.is_empty());

        // Override roughness + base color, save, and read back.
        let overrides = MatOverrides {
            roughness: Some(0.9),
            base_color: Some([1.0, 1.0, 1.0, 1.0]),
            ..Default::default()
        };
        save_material_instance(&mut proj, inst_id, overrides.clone()).unwrap();
        let back = get_material_instance(&proj, inst_id).unwrap();
        assert_eq!(back.overrides, overrides);
        // The parent dependency edge survives the rewrite.
        assert_eq!(
            proj.db().get(inst_id).unwrap().sidecar.dependencies,
            vec![parent_id]
        );
    }

    /// **A binding follows the instance chain to its root `.inf_mat`** (P26.3b
    /// audit), because nothing downstream resolves a `.inf_mati` GUID: the cook
    /// derives a `.inf_matd` only for `AssetKind::Material` and the PIE loader is
    /// kind-checked to the same, so an instance binding is a reference that
    /// resolves nowhere on BOTH wires — and raises no advisory, because the asset
    /// is in the project.
    ///
    /// Three rungs deep, so a walk that stopped after one would fail here; and
    /// the non-material control returns `None` rather than a plausible id,
    /// because "no binding" is the honest answer and a wrong one would be written
    /// into a level.
    #[test]
    fn a_binding_resolves_through_the_instance_chain_to_the_root_material() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let mats = proj.content_dir("materials").unwrap();

        let root = proj
            .write_asset(&mats, "Root", &MaterialAsset::default(), None, vec![], None)
            .unwrap();
        let mid = proj
            .write_asset(
                &mats,
                "Mid",
                &MaterialInstance::new(root),
                None,
                vec![root],
                None,
            )
            .unwrap();
        let leaf = proj
            .write_asset(
                &mats,
                "Leaf",
                &MaterialInstance::new(mid),
                None,
                vec![mid],
                None,
            )
            .unwrap();

        // Anti-vacuity: the three ids really are three, so "the walk arrived"
        // cannot be satisfied by the identity function.
        assert!(root != mid && mid != leaf && root != leaf);
        assert_eq!(material_binding_root(&proj, leaf, 0), Some(root));
        assert_eq!(material_binding_root(&proj, mid, 0), Some(root));
        // A material binds itself.
        assert_eq!(material_binding_root(&proj, root, 0), Some(root));
        // …and the resolved SCALARS still come from the instance, which is what
        // makes naming the root correct rather than lossy: `MatOverrides` has no
        // texture slots, so an instance's maps are its parent's.
        let mut tweaked = MaterialInstance::new(root);
        tweaked.overrides.roughness = Some(0.125);
        let tweaked_id = proj
            .write_asset(&mats, "Tweaked", &tweaked, None, vec![root], None)
            .unwrap();
        assert_eq!(material_binding_root(&proj, tweaked_id, 0), Some(root));
        assert_eq!(
            resolve_material(&proj, tweaked_id, 0).unwrap().roughness,
            0.125
        );

        // A non-material asset yields no binding at all.
        let tables = proj.content_dir("data").unwrap();
        let not_a_material = proj
            .write_asset(
                &tables,
                "Table",
                &inf_asset::data::TableAsset::new("Rows"),
                None,
                vec![],
                None,
            )
            .unwrap();
        assert_eq!(material_binding_root(&proj, not_a_material, 0), None);
        assert_eq!(material_binding_root(&proj, AssetId::new(), 0), None);
    }

    /// A **cyclic** instance chain terminates with no binding rather than
    /// recursing — the depth guard [`resolve_material`] already carried, applied
    /// to the id walk so the two doors refuse the same shapes.
    #[test]
    fn a_cyclic_instance_chain_yields_no_binding() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let mats = proj.content_dir("materials").unwrap();
        // An instance whose parent is itself: written once, then rewritten to
        // point at its own id (which only exists after the first write).
        let a = proj
            .write_asset(
                &mats,
                "Loop",
                &MaterialInstance::new(AssetId::new()),
                None,
                vec![],
                None,
            )
            .unwrap();
        proj.rewrite_payload(a, &MaterialInstance::new(a), vec![])
            .unwrap();
        assert_eq!(material_binding_root(&proj, a, 0), None);
        assert!(resolve_material(&proj, a, 0).is_none());
    }

    #[test]
    fn rejects_non_instance() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let mats = proj.content_dir("materials").unwrap();
        let mat_id = proj
            .write_asset(
                &mats,
                "Plain",
                &MaterialAsset::default(),
                None,
                vec![],
                None,
            )
            .unwrap();
        assert!(get_material_instance(&proj, mat_id).is_err());
        assert!(save_material_instance(&mut proj, mat_id, MatOverrides::default()).is_err());
    }
}
