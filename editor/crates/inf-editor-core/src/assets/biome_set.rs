//! `BiomeSet` (`.inf_biomes`) authoring glue (P19.2).
//!
//! The set is the vocabulary a terrain's per-sample biome ids name, so the editor
//! needs three things from it: create one with something usable in it, read it
//! for the inline editor and the toolbar's swatches, and write it back **through
//! the standard payload-rewrite path** so the content hash, the sidecar's
//! dependency list and the `assets://changed` invalidation all come for free —
//! the [`material_instance`](crate::assets::material_instance) precedent.
//!
//! The one thing this layer adds over a plain rewrite is that **saving
//! validates**. A set with duplicate ids, or one that claims the reserved id `0`,
//! makes every per-sample lookup ambiguous, and by the time it matters the ids are
//! already baked into terrain tiles. Refusing at the save is the only cheap place
//! to catch it — the cook refuses too (`inf_packager`'s `CookError::BiomeSet`),
//! but by then the author has painted with it.

use std::path::Path;

use inf_asset::{AssetError, AssetId, AssetKind, Result};
use inf_terrain::BiomeSet;

use crate::assets::AssetProject;

/// The content sub-folder a newly created biome set lands in.
pub const BIOME_SET_FOLDER: &str = "Biomes";

/// Create a `.inf_biomes` in `dir` seeded with [`BiomeSet::starter`] — three
/// biomes, so the paint tool and the overlay are usable the moment the asset
/// exists rather than after the author has invented a vocabulary.
pub fn create_default(project: &mut AssetProject, dir: &Path, name: &str) -> Result<AssetId> {
    let set = BiomeSet::new(name);
    let set = BiomeSet {
        biomes: BiomeSet::starter().biomes,
        ..set
    };
    let deps = set.dependencies();
    project.write_asset(dir, name, &set, None, deps, None)
}

/// Load a biome set. Errors if `id` is missing or is not a `.inf_biomes`.
///
/// The decode itself runs [`BiomeSet::validate`] (through the payload's
/// `migrate`), so a corrupt or hand-edited file surfaces here rather than as an
/// ambiguous lookup later.
pub fn get(project: &AssetProject, id: AssetId) -> Result<BiomeSet> {
    require_biome_set(project, id)?;
    project.load_payload::<BiomeSet>(id)
}

/// Persist an edited biome set, **validating first**.
///
/// The `pcg_graph` references become the sidecar's dependency edges (P19.3's
/// hook), so a set that names a graph keeps that asset alive through a
/// delete-with-references check and pulls it into the cook closure.
pub fn save(project: &mut AssetProject, id: AssetId, mut set: BiomeSet) -> Result<()> {
    require_biome_set(project, id)?;
    set.schema_version = inf_terrain::BIOME_SET_SCHEMA_VERSION;
    set.validate()
        .map_err(|e| AssetError::Import(format!("invalid biome set: {e}")))?;
    let deps = set.dependencies();
    project.rewrite_payload(id, &set, deps)
}

/// Every `.inf_biomes` in the project, as `(id, display name)`, sorted by name
/// then id — the picker's list, deterministic so the UI order never flickers.
pub fn list(project: &AssetProject) -> Vec<(AssetId, String)> {
    let mut out: Vec<(AssetId, String)> = project
        .db()
        .by_kind(AssetKind::BiomeSet)
        .map(|e| (e.id(), e.name.clone()))
        .collect();
    out.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    out
}

fn require_biome_set(project: &AssetProject, id: AssetId) -> Result<()> {
    let kind = project
        .db()
        .get(id)
        .ok_or(AssetError::UnknownAsset(id))?
        .kind();
    if kind != AssetKind::BiomeSet {
        return Err(AssetError::Import(format!("asset {id} is not a biome set")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_terrain::BiomeDef;

    fn project() -> (tempfile::TempDir, AssetProject) {
        let dir = tempfile::tempdir().unwrap();
        let proj = AssetProject::open(dir.path()).unwrap();
        (dir, proj)
    }

    #[test]
    fn create_get_save_round_trips_and_lists() {
        let (_dir, mut proj) = project();
        let folder = proj.content_dir(BIOME_SET_FOLDER).unwrap();
        let id = create_default(&mut proj, &folder, "WorldBiomes").unwrap();

        let set = get(&proj, id).unwrap();
        assert_eq!(set.biomes.len(), 3, "the starter set seeds three biomes");
        assert_eq!(set.name, "WorldBiomes");
        assert_eq!(list(&proj), vec![(id, "WorldBiomes".to_string())]);

        // Edit + save + reload.
        let mut edited = set.clone();
        edited.biomes.push(BiomeDef::new(9, "Marsh"));
        save(&mut proj, id, edited.clone()).unwrap();
        let back = get(&proj, id).unwrap();
        assert_eq!(back.biomes.len(), 4);
        assert_eq!(back.get(9).map(|b| b.name.as_str()), Some("Marsh"));
    }

    /// **Saving validates.** An ambiguous vocabulary must never reach disk — by
    /// the time it matters the ids are baked into terrain tiles.
    #[test]
    fn saving_an_invalid_set_is_refused_and_leaves_the_file_alone() {
        let (_dir, mut proj) = project();
        let folder = proj.content_dir(BIOME_SET_FOLDER).unwrap();
        let id = create_default(&mut proj, &folder, "Biomes").unwrap();
        let before = get(&proj, id).unwrap();

        let mut bad = before.clone();
        bad.biomes.push(BiomeDef::new(1, "Duplicate"));
        let err = save(&mut proj, id, bad).unwrap_err().to_string();
        assert!(err.contains("duplicate biome id 1"), "unhelpful: {err}");

        let mut reserved = before.clone();
        reserved.biomes.push(BiomeDef::new(0, "Unassigned"));
        let err = save(&mut proj, id, reserved).unwrap_err().to_string();
        assert!(err.contains("reserved"), "unhelpful: {err}");

        assert_eq!(
            get(&proj, id).unwrap(),
            before,
            "the file must be untouched"
        );
    }

    /// A biome's `pcg_graph` is a real dependency edge — the P19.3 hook — so it
    /// lands in the sidecar and keeps the graph alive through a
    /// delete-with-references check.
    #[test]
    fn pcg_graph_references_become_dependency_edges() {
        let (_dir, mut proj) = project();
        let folder = proj.content_dir(BIOME_SET_FOLDER).unwrap();
        let graph = proj
            .write_asset(
                &folder,
                "Scatter",
                &inf_pcg::PcgAssetPayload::new(Default::default()),
                None,
                vec![],
                None,
            )
            .unwrap();
        let id = create_default(&mut proj, &folder, "Biomes").unwrap();

        let mut set = get(&proj, id).unwrap();
        set.biomes[0].pcg_graph = Some(graph.0);
        save(&mut proj, id, set).unwrap();

        assert_eq!(proj.db().references_of(id), Some(&[graph][..]));
        assert!(
            proj.db().has_referrers(graph),
            "the graph must now be protected by the delete-with-references check"
        );
    }

    #[test]
    fn the_wrong_kind_is_refused() {
        let (_dir, mut proj) = project();
        let folder = proj.content_dir("Materials").unwrap();
        let mat = proj
            .write_asset(
                &folder,
                "M",
                &inf_material::MaterialAsset::default(),
                None,
                vec![],
                None,
            )
            .unwrap();
        assert!(get(&proj, mat).is_err());
        assert!(save(&mut proj, mat, BiomeSet::starter()).is_err());
    }
}
