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
use inf_audio::{AudioAsset, AudioFormat, AudioImportSettings};
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
        "obj" => import_obj(project, source, dest_dir)?,
        "png" | "jpg" | "jpeg" | "tga" | "bmp" | "hdr" | "exr" => {
            import_image(project, source, &bytes, dest_dir)?
        }
        "wav" | "ogg" | "mp3" | "flac" => import_audio(project, source, &bytes, &ext, dest_dir)?,
        other => {
            return Err(AssetError::Import(format!(
                "no importer for .{other} (table import arrives in a later phase)"
            )))
        }
    };

    if let Some(primary) = outcome.primary {
        let _ = project.cache_mut().put(key, primary, &bytes);
    }
    Ok(outcome)
}

/// A single audio file (WAV/OGG/FLAC/MP3) → one `.inf_audio` asset (P12.3).
///
/// The payload stores the **original compressed bytes** + a format tag + duration
/// metadata decoded once here (see `inf_audio::AudioAsset` for why not PCM). The
/// sidecar `import` table carries the default playback settings (loop flag /
/// default bus / volume trim). No thumbnail is produced — the Content Drawer falls
/// back to the audio glyph; a waveform thumbnail + an in-Details Play-button
/// preview are documented follow-ups (a device in the editor is needed to hear it).
fn import_audio(
    project: &mut AssetProject,
    source: &Path,
    bytes: &[u8],
    ext: &str,
    dest_dir: &Path,
) -> Result<ImportOutcome> {
    let settings = AudioImportSettings::default();
    let format = AudioFormat::from_extension(ext);
    let audio = AudioAsset::from_encoded(bytes.to_vec(), format)
        .map_err(|e| AssetError::Import(e.to_string()))?;
    let name = file_stem(source);
    let import_tbl = audio_settings_table(&settings);
    let id = project.write_asset(
        dest_dir,
        &name,
        &audio,
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

/// Serialize audio import settings into the sidecar's `import` TOML table
/// (mirrors [`settings_table`] for textures).
fn audio_settings_table(settings: &AudioImportSettings) -> Option<toml::Table> {
    toml::Value::try_from(settings)
        .ok()
        .and_then(|v| v.as_table().cloned())
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
    import_mesh_container(project, source, dest_dir, g)
}

/// Wavefront OBJ → the **same** container the glTF path yields (skeletons/clips
/// empty), so it reuses the identical textures → materials → meshes orchestration.
fn import_obj(project: &mut AssetProject, source: &Path, dest_dir: &Path) -> Result<ImportOutcome> {
    let g: GltfImport =
        inf_mesh::import_obj(source).map_err(|e| AssetError::Import(e.to_string()))?;
    import_mesh_container(project, source, dest_dir, g)
}

/// Fan a decoded mesh-import container (glTF or OBJ) out into textures +
/// materials + skeletons + clips + meshes, wired by GUID dependency edges.
fn import_mesh_container(
    project: &mut AssetProject,
    source: &Path,
    dest_dir: &Path,
    g: GltfImport,
) -> Result<ImportOutcome> {
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
    let decoded: Vec<std::result::Result<inf_material::TextureAsset, String>> = {
        let _span = tracing::info_span!("import_textures", images = g.images.len()).entered();
        inf_core::parallel_map((0..g.images.len()).collect(), |i| {
            let img = &g.images[i];
            inf_material::texture_from_rgba8(img.rgba8.clone(), img.width, img.height, settings[i])
                .map_err(|e| e.to_string())
        })
    };
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
            // R-P5 additive fields (blend/alpha_cutoff): imported glTF materials are
            // opaque by default.
            ..Default::default()
        };
        let deps = mat.texture_dependencies();
        let name = format!("{}_{}", file_stem(source), m.name);
        let id =
            project.write_asset(dest_dir, &name, &mat, Some(source_rel.clone()), deps, None)?;
        material_ids.push(id);
        produced.push(id);
    }

    // 3. Skeletons (P11.1): one `.inf_skel` per glTF skin. Meshes depend on their
    //    skeleton; clips depend on it too (skeleton ← clips, mesh → skeleton).
    let mut skeleton_ids: Vec<AssetId> = Vec::with_capacity(g.skeletons.len());
    for sk in &g.skeletons {
        let asset = inf_anim::SkeletonAsset::new(sk.skeleton.clone());
        let name = format!("{}_{}", file_stem(source), sk.name);
        let id = project.write_asset(
            dest_dir,
            &name,
            &asset,
            Some(source_rel.clone()),
            vec![],
            None,
        )?;
        skeleton_ids.push(id);
        produced.push(id);
    }

    // 4. Animation clips (P11.1): one `.inf_anim` per glTF animation, each with a
    //    dependency edge onto the skeleton its tracks target.
    for clip in &g.clips {
        let skel_id = clip.skeleton.and_then(|i| skeleton_ids.get(i).copied());
        let skel_bytes = skel_id.map(|id| *id.uuid().as_bytes());
        let asset = inf_anim::AnimClipAsset::new(clip.clip.clone(), skel_bytes);
        let deps: Vec<AssetId> = skel_id.into_iter().collect();
        let name = format!("{}_{}", file_stem(source), clip.name);
        let id = project.write_asset(
            dest_dir,
            &name,
            &asset,
            Some(source_rel.clone()),
            deps,
            None,
        )?;
        produced.push(id);
    }

    // 5. Meshes; each depends on the materials its submeshes use, plus its
    //    skeleton (when skinned).
    let mut primary = None;
    for im in &g.meshes {
        let used: BTreeSet<u32> = im
            .mesh
            .submeshes
            .iter()
            .filter_map(|s| s.material_slot)
            .collect();
        let mut deps: Vec<AssetId> = used
            .into_iter()
            .filter_map(|slot| material_ids.get(slot as usize).copied())
            .collect();
        if let Some(sid) = im.skin.and_then(|i| skeleton_ids.get(i).copied()) {
            deps.push(sid);
        }
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

    /// A programmatically built skinned glTF (a 2-joint chain, a skinned
    /// triangle, one rotation clip) with an external buffer. Mirrors the
    /// inf-mesh importer fixture; used to prove the editor's asset wiring.
    fn write_skinned_gltf(dir: &Path) -> std::path::PathBuf {
        let mut buf: Vec<u8> = Vec::new();
        let push_f32s = |buf: &mut Vec<u8>, vals: &[f32]| -> usize {
            let off = buf.len();
            for &v in vals {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            off
        };
        let pos_off = push_f32s(&mut buf, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        let wgt_off = push_f32s(
            &mut buf,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let ibm_off = buf.len();
        for _ in 0..2 {
            for v in identity {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        let time_off = push_f32s(&mut buf, &[0.0, 1.0]);
        let rot_off = push_f32s(
            &mut buf,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.707_106_77, 0.707_106_77],
        );
        while !buf.len().is_multiple_of(4) {
            buf.push(0);
        }
        let idx_off = buf.len();
        for i in [0u16, 1, 2] {
            buf.extend_from_slice(&i.to_le_bytes());
        }
        while !buf.len().is_multiple_of(4) {
            buf.push(0);
        }
        let joint_off = buf.len();
        for j in [[0u16, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]] {
            for c in j {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        }
        std::fs::write(dir.join("skin.bin"), &buf).unwrap();
        let gltf = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0, 2] }}],
  "nodes": [
    {{ "name": "j0", "translation": [0,0,0], "children": [1] }},
    {{ "name": "j1", "translation": [0,1,0] }},
    {{ "name": "skinNode", "mesh": 0, "skin": 0 }}
  ],
  "meshes": [{{ "name": "SkinMesh", "primitives": [{{ "attributes": {{ "POSITION": 0, "JOINTS_0": 6, "WEIGHTS_0": 1 }}, "indices": 5 }}] }}],
  "skins": [{{ "name": "Skel", "joints": [0,1], "inverseBindMatrices": 2 }}],
  "animations": [{{ "name": "Wave", "channels": [{{ "sampler": 0, "target": {{ "node": 1, "path": "rotation" }} }}], "samplers": [{{ "input": 3, "output": 4, "interpolation": "LINEAR" }}] }}],
  "buffers": [{{ "uri": "skin.bin", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {pos_off}, "byteLength": 36, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {wgt_off}, "byteLength": 48, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {ibm_off}, "byteLength": 128 }},
    {{ "buffer": 0, "byteOffset": {time_off}, "byteLength": 8 }},
    {{ "buffer": 0, "byteOffset": {rot_off}, "byteLength": 32 }},
    {{ "buffer": 0, "byteOffset": {idx_off}, "byteLength": 6, "target": 34963 }},
    {{ "buffer": 0, "byteOffset": {joint_off}, "byteLength": 24, "target": 34962 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,2,0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4" }},
    {{ "bufferView": 2, "componentType": 5126, "count": 2, "type": "MAT4" }},
    {{ "bufferView": 3, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0] }},
    {{ "bufferView": 4, "componentType": 5126, "count": 2, "type": "VEC4" }},
    {{ "bufferView": 5, "componentType": 5123, "count": 3, "type": "SCALAR" }},
    {{ "bufferView": 6, "componentType": 5123, "count": 3, "type": "VEC4" }}
  ]
}}"#,
            total = buf.len(),
            pos_off = pos_off,
            wgt_off = wgt_off,
            ibm_off = ibm_off,
            time_off = time_off,
            rot_off = rot_off,
            idx_off = idx_off,
            joint_off = joint_off,
        );
        let path = dir.join("skin.gltf");
        std::fs::write(&path, gltf).unwrap();
        path
    }

    #[test]
    fn skinned_gltf_import_creates_skeleton_and_clip_assets_with_edges() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let gltf = write_skinned_gltf(src.path());

        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();
        let out = proj.import_file(&gltf, &dest).unwrap();

        // Exactly one skeleton + one clip asset were produced.
        let kinds: Vec<AssetKind> = out
            .produced
            .iter()
            .map(|id| proj.db().get(*id).unwrap().kind())
            .collect();
        assert_eq!(
            kinds.iter().filter(|k| **k == AssetKind::Skeleton).count(),
            1
        );
        assert_eq!(
            kinds.iter().filter(|k| **k == AssetKind::AnimClip).count(),
            1
        );

        let skel_id = out
            .produced
            .iter()
            .copied()
            .find(|id| proj.db().get(*id).unwrap().kind() == AssetKind::Skeleton)
            .unwrap();
        let clip_id = out
            .produced
            .iter()
            .copied()
            .find(|id| proj.db().get(*id).unwrap().kind() == AssetKind::AnimClip)
            .unwrap();
        let mesh_id = out.primary.unwrap();

        // mesh → skeleton and clip → skeleton dependency edges both exist.
        assert!(proj.db().references_of(mesh_id).unwrap().contains(&skel_id));
        assert_eq!(proj.db().references_of(clip_id).unwrap(), vec![skel_id]);
        // Reverse edge: the skeleton is referenced by both the mesh and the clip.
        let mut referrers = proj.referenced_by(skel_id);
        referrers.sort();
        let mut expected = vec![mesh_id, clip_id];
        expected.sort();
        assert_eq!(referrers, expected);
    }

    /// A small OBJ + MTL written to a temp dir, imported end-to-end through the
    /// shared orchestrator. Proves an `.inf_mesh` lands as the primary asset,
    /// carries a `usemtl`-derived material dependency edge, and — since the
    /// source had no `vn` — that normals were generated (the meshopt weld also
    /// runs; a flat quad's four corners survive). (`map_Kd` → texture wiring is
    /// unit-tested in `inf-mesh`; `inf-editor-core` has no `image` dep to author
    /// a PNG here.)
    #[test]
    fn obj_import_lands_mesh_with_material_edge_and_normals() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();

        std::fs::write(
            src.path().join("quad.mtl"),
            "newmtl painted\nKd 0.2 0.4 0.8\n",
        )
        .unwrap();
        // A quad (fan → 2 tris) with no normals — exercises generation + optimize.
        std::fs::write(
            src.path().join("quad.obj"),
            "mtllib quad.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nusemtl painted\nf 1 2 3 4\n",
        )
        .unwrap();

        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();
        let out = proj
            .import_file(&src.path().join("quad.obj"), &dest)
            .unwrap();

        let mesh_id = out.primary.unwrap();
        assert_eq!(proj.db().get(mesh_id).unwrap().kind(), AssetKind::Mesh);

        // The mesh → material edge exists (from usemtl).
        let mat_deps = proj.db().references_of(mesh_id).unwrap();
        assert_eq!(mat_deps.len(), 1);
        assert_eq!(
            proj.db().get(mat_deps[0]).unwrap().kind(),
            AssetKind::Material
        );

        // Re-decode the container directly to confirm generated normals survived.
        let g = inf_mesh::import_obj(&src.path().join("quad.obj")).unwrap();
        let m = &g.meshes[0].mesh;
        assert_eq!(m.triangle_count(), 2);
        assert_eq!(m.vertex_count(), 4, "weld keeps 4 unique corners");
        for v in &m.submeshes[0].vertices {
            assert!(v.normal[2] > 0.9, "generated flat normal faces +Z");
        }
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
