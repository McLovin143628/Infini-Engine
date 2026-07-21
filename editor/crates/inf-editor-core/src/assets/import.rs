//! The import orchestrator: external file → one or more `.inf_*` assets.
//!
//! A glTF import fans out into textures + materials + meshes, wired together by
//! GUID dependency edges (mesh → materials → textures). A bare image imports to
//! a single texture. Imports are content-cached: re-importing an unchanged
//! source whose product still exists in the database is a hash lookup that
//! returns the existing asset instead of decoding again.

use std::collections::BTreeSet;
use std::path::Path;

use inf_asset::{AssetError, AssetId, ImportKey, Result};
use inf_material::{MaterialAsset, TextureImportSettings};
use inf_mesh::GltfImport;

use super::AssetProject;

/// What an import produced.
#[derive(Debug, Clone, Default)]
pub struct ImportOutcome {
    /// Every asset created (or, on a cache hit, the reused primary).
    pub produced: Vec<AssetId>,
    /// The headline asset (the mesh for glTF, the texture for an image) —
    /// what the UI selects/reveals after import.
    pub primary: Option<AssetId>,
    /// True if this was served from the import cache (nothing re-decoded).
    pub cached: bool,
}

/// Import one source file into `dest_dir`, routing by extension.
pub fn import_file(
    project: &mut AssetProject,
    source: &Path,
    dest_dir: &Path,
) -> Result<ImportOutcome> {
    let ext = source
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // Content-cache check: same source bytes + same importer → reuse if the
    // produced asset still exists.
    let bytes = std::fs::read(source)?;
    let key = ImportKey::new(&bytes, ext.as_bytes());
    if let Some(existing) = project.cache_mut().get(key) {
        if project.db().contains(existing) {
            return Ok(ImportOutcome {
                produced: vec![existing],
                primary: Some(existing),
                cached: true,
            });
        }
    }

    let outcome = match ext.as_str() {
        "gltf" | "glb" => import_gltf(project, source, dest_dir)?,
        "png" | "jpg" | "jpeg" | "tga" | "bmp" | "hdr" | "exr" => {
            import_image(project, source, &bytes, dest_dir)?
        }
        other => {
            return Err(AssetError::Import(format!(
                "no importer for .{other} (audio/table import arrive in later phases)"
            )))
        }
    };

    if let Some(primary) = outcome.primary {
        let _ = project.cache_mut().put(key, primary, &bytes);
    }
    Ok(outcome)
}

/// A single image → one texture asset (sRGB base-color defaults).
fn import_image(
    project: &mut AssetProject,
    source: &Path,
    bytes: &[u8],
    dest_dir: &Path,
) -> Result<ImportOutcome> {
    let settings = TextureImportSettings::default();
    let tex = inf_material::import_texture_bytes(bytes, settings)
        .map_err(|e| AssetError::Import(e.to_string()))?;
    let name = file_stem(source);
    let import_tbl = settings_table(&settings);
    let id = project.write_asset(
        dest_dir,
        &name,
        &tex,
        Some(rel_source(project, source)),
        vec![],
        import_tbl,
    )?;
    Ok(ImportOutcome {
        produced: vec![id],
        primary: Some(id),
        cached: false,
    })
}

/// glTF → textures + materials + meshes, wired by dependency edges.
fn import_gltf(
    project: &mut AssetProject,
    source: &Path,
    dest_dir: &Path,
) -> Result<ImportOutcome> {
    let g: GltfImport =
        inf_mesh::import_gltf(source).map_err(|e| AssetError::Import(e.to_string()))?;
    let source_rel = rel_source(project, source);
    let mut produced = Vec::new();

    // 1. Decide each image's usage (sRGB base color vs linear data) from the
    //    materials that reference it, then import.
    let mut srgb = vec![false; g.images.len()];
    for m in &g.materials {
        if let Some(i) = m.base_color_image {
            srgb[i] = true;
        }
    }
    let mut image_ids: Vec<Option<AssetId>> = vec![None; g.images.len()];
    // Decode + build mips + BC-compress every image *in parallel* — the CPU-bound,
    // side-effect-free stage — then commit to the asset DB serially in input
    // order below (so GUIDs and sidecar output stay deterministic regardless of
    // pool size). This is the P7.0 job system's first real consumer (§2.5).
    let settings: Vec<TextureImportSettings> = (0..g.images.len())
        .map(|i| {
            if srgb[i] {
                TextureImportSettings::default()
            } else {
                TextureImportSettings::data()
            }
        })
        .collect();
    let decoded: Vec<std::result::Result<inf_material::TextureAsset, String>> =
        inf_core::parallel_map((0..g.images.len()).collect(), |i| {
            let img = &g.images[i];
            inf_material::texture_from_rgba8(img.rgba8.clone(), img.width, img.height, settings[i])
                .map_err(|e| e.to_string())
        });
    for (i, tex) in decoded.into_iter().enumerate() {
        let tex = tex.map_err(AssetError::Import)?;
        let name = format!("{}_{}", file_stem(source), g.images[i].name);
        let id = project.write_asset(
            dest_dir,
            &name,
            &tex,
            Some(source_rel.clone()),
            vec![],
            settings_table(&settings[i]),
        )?;
        image_ids[i] = Some(id);
        produced.push(id);
    }

    // 2. Materials referencing their texture GUIDs.
    let mut material_ids: Vec<AssetId> = Vec::with_capacity(g.materials.len());
    for m in &g.materials {
        let mat = MaterialAsset {
            schema_version: MaterialAsset::CURRENT_VERSION,
            base_color: m.base_color,
            metallic: m.metallic,
            roughness: m.roughness,
            emissive: m.emissive,
            base_color_texture: m.base_color_image.and_then(|i| image_ids[i]),
            normal_texture: m.normal_image.and_then(|i| image_ids[i]),
            metallic_roughness_texture: m.metallic_roughness_image.and_then(|i| image_ids[i]),
        };
        let deps = mat.texture_dependencies();
        let name = format!("{}_{}", file_stem(source), m.name);
        let id =
            project.write_asset(dest_dir, &name, &mat, Some(source_rel.clone()), deps, None)?;
        material_ids.push(id);
        produced.push(id);
    }

    // 3. Meshes; each depends on the materials its submeshes use.
    let mut primary = None;
    for im in &g.meshes {
        let used: BTreeSet<u32> = im
            .mesh
            .submeshes
            .iter()
            .filter_map(|s| s.material_slot)
            .collect();
        let deps: Vec<AssetId> = used
            .into_iter()
            .filter_map(|slot| material_ids.get(slot as usize).copied())
            .collect();
        let id = project.write_asset(
            dest_dir,
            &im.name,
            &im.mesh,
            Some(source_rel.clone()),
            deps,
            None,
        )?;
        produced.push(id);
        primary.get_or_insert(id);
    }

    // Prefer the first mesh as the headline; fall back to whatever came out.
    let primary = primary.or_else(|| produced.first().copied());
    Ok(ImportOutcome {
        produced,
        primary,
        cached: false,
    })
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Imported")
        .to_string()
}

/// The source path relative to the project root, if it lives under it; else the
/// absolute path (imports can pull from anywhere on disk).
fn rel_source(project: &AssetProject, source: &Path) -> String {
    source
        .strip_prefix(project.root())
        .unwrap_or(source)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Serialize texture import settings into the sidecar `import` table.
fn settings_table(settings: &TextureImportSettings) -> Option<toml::Table> {
    toml::Value::try_from(settings)
        .ok()
        .and_then(|v| v.as_table().cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::AssetKind;

    /// Write a minimal single-triangle glTF with an external buffer.
    fn write_triangle_gltf(dir: &Path) -> std::path::PathBuf {
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut buf = Vec::new();
        for p in positions {
            for f in p {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        let idx_off = buf.len();
        for i in [0u16, 1, 2] {
            buf.extend_from_slice(&i.to_le_bytes());
        }
        std::fs::write(dir.join("tri.bin"), &buf).unwrap();
        let gltf = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"name":"Tri","primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],"materials":[{{"name":"Red","pbrMetallicRoughness":{{"baseColorFactor":[1,0,0,1]}}}}],"buffers":[{{"uri":"tri.bin","byteLength":{total}}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{pl},"target":34962}},{{"buffer":0,"byteOffset":{io},"byteLength":6,"target":34963}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},{{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}]}}"#,
            total = buf.len(),
            pl = idx_off,
            io = idx_off,
        );
        let path = dir.join("tri.gltf");
        std::fs::write(&path, gltf).unwrap();
        path
    }

    #[test]
    fn gltf_import_wires_mesh_to_material_dependency() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let gltf = write_triangle_gltf(src.path());

        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();
        let out = proj.import_file(&gltf, &dest).unwrap();

        // One material + one mesh (no textures on this material).
        let mesh_id = out.primary.unwrap();
        assert_eq!(proj.db().get(mesh_id).unwrap().kind(), AssetKind::Mesh);
        // The mesh depends on exactly one material.
        let deps = proj.db().references_of(mesh_id).unwrap();
        assert_eq!(deps.len(), 1);
        let mat_id = deps[0];
        assert_eq!(proj.db().get(mat_id).unwrap().kind(), AssetKind::Material);
        // Reverse edge: deleting the material now warns (mesh references it).
        assert_eq!(proj.referenced_by(mat_id), vec![mesh_id]);
    }

    #[test]
    fn reimport_hits_the_cache() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let gltf = write_triangle_gltf(src.path());
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();

        let first = proj.import_file(&gltf, &dest).unwrap();
        assert!(!first.cached);
        let count_after_first = proj.db().len();

        let second = proj.import_file(&gltf, &dest).unwrap();
        assert!(second.cached, "second import served from cache");
        assert_eq!(second.primary, first.primary);
        assert_eq!(proj.db().len(), count_after_first, "no duplicate assets");
    }
}
