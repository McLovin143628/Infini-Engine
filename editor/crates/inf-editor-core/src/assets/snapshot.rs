//! Projecting the [`AssetProject`] into the Content Drawer's IPC DTOs.
//!
//! The frontend never sees the [`AssetDb`](inf_asset::AssetDb) directly — it
//! gets a flat, GUID-keyed [`AssetSnapshot`] (assets + the folder tree derived
//! from their paths) and re-fetches it on the `assets://changed` event. This is
//! the asset analogue of the scene snapshot.

use std::collections::{BTreeMap, BTreeSet};

use inf_asset::{AssetEntry, AssetKind};

use crate::assets::AssetProject;
use crate::ipc::{AssetDto, AssetFolderDto, AssetRefDto, AssetSnapshot};

/// True for kinds the thumbnailer can render a real preview for.
pub fn is_previewable(kind: AssetKind) -> bool {
    matches!(
        kind,
        AssetKind::Mesh | AssetKind::Texture | AssetKind::Material
    )
}

/// Project one entry into its DTO. `ref_count` is the reverse-dependency count.
pub fn asset_dto(project: &AssetProject, entry: &AssetEntry) -> AssetDto {
    let id = entry.id();
    let rel = rel_path(project, entry);
    let folder = rel
        .rsplit_once('/')
        .map(|(dir, _)| dir.to_string())
        .unwrap_or_default();
    AssetDto {
        id: id.to_string(),
        name: entry.name.clone(),
        kind: entry.kind().slug().to_string(),
        kind_label: entry.kind().label().to_string(),
        folder,
        path: rel,
        content_hash: entry.content_hash().to_hex(),
        tags: entry.sidecar.tags.clone(),
        source: entry.sidecar.source.clone(),
        dep_count: entry.sidecar.dependencies.len() as u32,
        ref_count: project.db().referenced_by(id).len() as u32,
        previewable: is_previewable(entry.kind()),
    }
}

/// A referrer reference (for the delete-with-references warning).
pub fn ref_dto(project: &AssetProject, id: inf_asset::AssetId) -> AssetRefDto {
    match project.db().get(id) {
        Some(e) => AssetRefDto {
            id: id.to_string(),
            name: e.name.clone(),
            kind: e.kind().label().to_string(),
        },
        None => AssetRefDto {
            id: id.to_string(),
            name: "(missing)".into(),
            kind: "File".into(),
        },
    }
}

/// Build the full content snapshot.
pub fn build(project: &AssetProject) -> AssetSnapshot {
    let mut assets: Vec<AssetDto> = project.db().iter().map(|e| asset_dto(project, e)).collect();
    // **The GUID breaks the tie, and the source is why.** `AssetDb::iter` is
    // documented "all entries, unordered" — it walks a `HashMap` — and
    // lower-cased display names are *not* unique: `Rock.inf_mesh` and
    // `rock.inf_mesh` in two folders, or the same name in `props/` and `env/`,
    // collide. A sort on a non-total key over an unordered source leaves the
    // colliding pair in whatever order the hash walk produced, so the Content
    // Drawer's grid could reorder between two snapshots of an unchanged project
    // — a row moving under a click. `sort_by_key` is stable, which preserves the
    // input order for ties and therefore preserves exactly the thing that has no
    // order. The GUID is the identity the drawer already keys its rows on, so it
    // is the tie-break that costs nothing.
    assets.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.id.cmp(&b.id))
    });

    // Derive the folder tree from asset folders (+ all ancestors, + root).
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut all: BTreeSet<String> = BTreeSet::new();
    all.insert(String::new());
    for a in &assets {
        *counts.entry(a.folder.clone()).or_default() += 1;
        // Insert this folder and every ancestor.
        let mut cur = a.folder.clone();
        loop {
            all.insert(cur.clone());
            match cur.rsplit_once('/') {
                Some((parent, _)) => cur = parent.to_string(),
                None => break,
            }
        }
    }

    let folders = all
        .iter()
        .map(|path| {
            let name = path.rsplit_once('/').map(|(_, n)| n).unwrap_or(path);
            let children: Vec<String> = all
                .iter()
                .filter(|c| is_direct_child(path, c))
                .cloned()
                .collect();
            AssetFolderDto {
                path: path.clone(),
                name: name.to_string(),
                children,
                asset_count: counts.get(path).copied().unwrap_or(0),
            }
        })
        .collect();

    AssetSnapshot {
        version: project.version(),
        root: project.root().to_string_lossy().replace('\\', "/"),
        assets,
        folders,
    }
}

/// Payload path relative to the (canonicalized) content root, forward-slashed.
fn rel_path(project: &AssetProject, entry: &AssetEntry) -> String {
    let root =
        std::fs::canonicalize(project.root()).unwrap_or_else(|_| project.root().to_path_buf());
    entry
        .path
        .strip_prefix(&root)
        .unwrap_or(&entry.path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// True if `child` is a direct subfolder of `parent` (one path segment deeper).
fn is_direct_child(parent: &str, child: &str) -> bool {
    if child.is_empty() || parent == child {
        return false;
    }
    match child.rsplit_once('/') {
        Some((dir, _)) => dir == parent,
        None => parent.is_empty(), // top-level folder's parent is the root ("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_material::MaterialAsset;

    #[test]
    fn snapshot_lists_assets_and_derives_folder_tree() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let meshes = proj.content_dir("props/env").unwrap();
        let mats = proj.content_dir("materials").unwrap();
        proj.write_asset(&mats, "Wall", &MaterialAsset::default(), None, vec![], None)
            .unwrap();
        proj.write_asset(
            &meshes,
            "Rock",
            &MaterialAsset::default(),
            None,
            vec![],
            None,
        )
        .unwrap();

        let snap = build(&proj);
        assert_eq!(snap.assets.len(), 2);
        // Folder set: "", "materials", "props", "props/env".
        let paths: BTreeSet<&str> = snap.folders.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(""));
        assert!(paths.contains("materials"));
        assert!(paths.contains("props"));
        assert!(paths.contains("props/env"));

        let root = snap.folders.iter().find(|f| f.path.is_empty()).unwrap();
        let mut kids = root.children.clone();
        kids.sort();
        assert_eq!(kids, vec!["materials".to_string(), "props".to_string()]);

        let props = snap.folders.iter().find(|f| f.path == "props").unwrap();
        assert_eq!(props.children, vec!["props/env".to_string()]);
        assert_eq!(props.asset_count, 0, "props itself holds no assets");
        let env = snap.folders.iter().find(|f| f.path == "props/env").unwrap();
        assert_eq!(env.asset_count, 1);
    }
}
