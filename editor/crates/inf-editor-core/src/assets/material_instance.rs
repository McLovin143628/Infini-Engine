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
