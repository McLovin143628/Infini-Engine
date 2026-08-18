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
    /// **Non-fatal advisories** raised while importing (P26.1) — the P16
    /// cook-advisory shape: the import succeeded, and these say what it had to
    /// do that the author cannot see in the result.
    ///
    /// Every one is also `tracing::warn!`ed at the moment it is raised, so it
    /// reaches the Output Log panel without any frontend work. Carrying them on
    /// the `assets://import` event (a toast, a per-asset badge) is the P26.5
    /// editor batch's job; this is the Ring-1 record it will read.
    pub advisories: Vec<String>,
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
                // A cache hit re-decodes nothing, so it re-raises nothing: the
                // advisories were said when the bytes were first imported.
                advisories: Vec::new(),
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
        // C4-45: a failed cache write used to be invisible, so every re-import
        // silently re-decoded the same bytes and wrote a DUPLICATE asset. The
        // import itself succeeded, so this is an advisory, not a failure.
        if let Err(e) = project.cache_mut().put(key, primary, &bytes) {
            tracing::warn!(
                "import cache could not record {}: {e}; re-importing this file will \
                 create a duplicate asset instead of reusing it",
                source.display()
            );
        }
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
        super::sidecar_source(project, source),
        vec![],
        import_tbl,
    )?;
    Ok(ImportOutcome {
        produced: vec![id],
        primary: Some(id),
        cached: false,
        advisories: outside_root_advisories(project, source),
    })
}

/// Serialize audio import settings into the sidecar's `import` TOML table
/// (mirrors [`settings_table`] for textures).
fn audio_settings_table(settings: &AudioImportSettings) -> Option<toml::Table> {
    match toml::Value::try_from(settings) {
        Ok(toml::Value::Table(t)) => Some(t),
        Ok(other) => {
            tracing::warn!(
                "audio import settings serialized as {} rather than a table; the sidecar \
                 will record none and re-import will refuse",
                other.type_str()
            );
            None
        }
        Err(e) => {
            tracing::warn!(
                "audio import settings will not serialize ({e}); the sidecar will record \
                 none and re-import will refuse"
            );
            None
        }
    }
}

/// A single image → one **v2 tiled** texture asset (sRGB base-color defaults).
///
/// P26.1: the payload written here is the tiled container, not the bincode
/// record — every texture imported from now on is byte-addressable by tile. v1
/// payloads already on disk keep loading through `TextureAsset::from_payload`.
fn import_image(
    project: &mut AssetProject,
    source: &Path,
    bytes: &[u8],
    dest_dir: &Path,
) -> Result<ImportOutcome> {
    let settings = TextureImportSettings::default();
    let (rgba, w, h) =
        inf_material::decode_image_rgba8(bytes).map_err(|e| AssetError::Import(e.to_string()))?;
    let mut advisories = raise(source, inf_material::texture_import_advisories(w, h));
    advisories.extend(outside_root_advisories(project, source));
    let image = inf_material::build_tiled_texture(rgba, w, h, settings)
        .map_err(|e| AssetError::Import(e.to_string()))?;
    let name = file_stem(source);
    let import_tbl = settings_table(&settings);
    let id = project.write_tiled_texture(
        dest_dir,
        &name,
        &image,
        super::sidecar_source(project, source),
        import_tbl,
    )?;
    Ok(ImportOutcome {
        produced: vec![id],
        primary: Some(id),
        cached: false,
        advisories,
    })
}

/// Log each advisory where it is raised and hand the list back. One place, so
/// an advisory can never be recorded without being said (or said twice).
fn raise(source: &Path, advisories: Vec<String>) -> Vec<String> {
    for a in &advisories {
        tracing::warn!("import advisory ({}): {a}", source.display());
    }
    advisories
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
    let source_rel = super::sidecar_source(project, source);
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
    let decoded: Vec<std::result::Result<inf_material::TiledTextureImage, String>> = {
        let _span = tracing::info_span!("import_textures", images = g.images.len()).entered();
        inf_core::parallel_map((0..g.images.len()).collect(), |i| {
            let img = &g.images[i];
            inf_material::build_tiled_texture(img.rgba8.clone(), img.width, img.height, settings[i])
                .map_err(|e| e.to_string())
        })
    };
    let mut advisories = outside_root_advisories(project, source);
    for (i, tex) in decoded.into_iter().enumerate() {
        let tex = tex.map_err(AssetError::Import)?;
        advisories.extend(raise(
            source,
            inf_material::texture_import_advisories(g.images[i].width, g.images[i].height),
        ));
        let name = format!("{}_{}", file_stem(source), g.images[i].name);
        let id = project.write_tiled_texture(
            dest_dir,
            &name,
            &tex,
            source_rel.clone(),
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
        let id = project.write_asset(dest_dir, &name, &mat, source_rel.clone(), deps, None)?;
        material_ids.push(id);
        produced.push(id);
    }

    // 3. Skeletons (P11.1): one `.inf_skel` per glTF skin. Meshes depend on their
    //    skeleton; clips depend on it too (skeleton ← clips, mesh → skeleton).
    let mut skeleton_ids: Vec<AssetId> = Vec::with_capacity(g.skeletons.len());
    for sk in &g.skeletons {
        let asset = inf_anim::SkeletonAsset::new(sk.skeleton.clone());
        let name = format!("{}_{}", file_stem(source), sk.name);
        let id = project.write_asset(dest_dir, &name, &asset, source_rel.clone(), vec![], None)?;
        skeleton_ids.push(id);
        produced.push(id);
    }

    // 4. Animation clips (P11.1): one `.inf_anim` per glTF animation, each with a
    //    dependency edge onto the skeleton its tracks target.
    for clip in &g.clips {
        let skel_id = clip.skeleton.and_then(|i| skeleton_ids.get(i).copied());
        let skel_bytes = skel_id.map(|id| *id.uuid().as_bytes());
        // **Derived before it is written** (P29.5, pillar S2). What Epic ships as
        // seven AnimModifiers a human runs by hand over 891 clips happens here,
        // once, on the way in — so there is no window in which a `.inf_anim` in
        // this project has not been measured, and no clip that was imported on a
        // day somebody forgot.
        //
        // The rig comes out of the import in memory rather than off the asset
        // just written: same skeleton, one fewer decode.
        let mut payload = clip.clip.clone();
        let rig = clip.skeleton.and_then(|i| g.skeletons.get(i));
        let derived = super::anim_derive::derive_in_place(
            &clip.name,
            &mut payload,
            rig.map(|sk| inf_anim::SkeletonAsset::new(sk.skeleton.clone()))
                .as_ref(),
            &inf_anim::DeriveOptions::default(),
        );
        advisories.extend(raise(
            source,
            super::anim_derive::advisories(&clip.name, &derived),
        ));
        let asset = inf_anim::AnimClipAsset::new(payload, skel_bytes);
        let deps: Vec<AssetId> = skel_id.into_iter().collect();
        let name = format!("{}_{}", file_stem(source), clip.name);
        // **The identity tie** (round-2 finding R2.C): the clip's coupling to
        // its rig is POSITIONAL — `track.joint` is an index into the pose,
        // which is index-aligned to the skeleton's joint list — while the
        // binding it records is a GUID. The skeleton's content hash goes in the
        // sidecar so a re-imported rig can be NOTICED. Sidecar-only, so neither
        // bincode payload moves.
        let import = super::skeleton_binding::import_table(project, skel_id);
        let id = project.write_asset(dest_dir, &name, &asset, source_rel.clone(), deps, import)?;
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
        let id =
            project.write_asset(dest_dir, &im.name, &im.mesh, source_rel.clone(), deps, None)?;
        produced.push(id);
        primary.get_or_insert(id);
    }

    // 6. Derive a `.inf_vmesh` meshlet DAG per imported mesh (P18.3), so the
    //    interactive viewport can draw the real geometry the moment the asset is
    //    dropped into a scene. This is the editor's half of what the cook has done
    //    since P13.1; it runs here rather than lazily on the render thread because
    //    a multi-second `build_vgeom` must never land inside a frame.
    //
    //    The derived assets are appended to `produced` (they *are* products of this
    //    import — the Content Drawer showing them is honest) but `primary` is left
    //    alone: the headline of a mesh import is the mesh.
    let mesh_ids: Vec<AssetId> = produced
        .iter()
        .copied()
        .filter(|id| project.db().get(*id).map(|e| e.kind()) == Some(inf_asset::AssetKind::Mesh))
        .collect();
    let mut derived = Vec::new();
    for mesh_id in mesh_ids {
        match super::vmesh::ensure_vmesh(project, mesh_id) {
            Ok(d) => derived.extend(d.asset()),
            // A DAG that fails to build costs the entity its real geometry (it
            // falls back to the primitive placeholder) — it must not fail the
            // import that produced a perfectly good `.inf_mesh`.
            Err(e) => tracing::warn!("inf-editor-core: vmesh derivation for {mesh_id}: {e}"),
        }
    }
    produced.extend(derived);

    // Prefer the first mesh as the headline; fall back to whatever came out.
    let primary = primary.or_else(|| produced.first().copied());
    Ok(ImportOutcome {
        produced,
        primary,
        cached: false,
        advisories,
    })
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Imported")
        .to_string()
}

/// Zero or one advisory: the L7.H7 note, raised exactly where the sidecar
/// declines to record an outside-the-project path.
///
/// `pub(crate)` because the **terrain** wizard needs it too (round-2 finding
/// R2.F7) — and that is the door where an outside-the-project source is the
/// *norm*, since a heightmap comes off the author's disk rather than out of the
/// project.
pub(crate) fn outside_root_advisories(project: &AssetProject, source: &Path) -> Vec<String> {
    if super::sidecar_source(project, source).is_some() {
        return Vec::new();
    }
    raise(source, vec![super::outside_root_advisory(source)])
}

/// Serialize texture import settings into the sidecar `import` table.
///
/// C4-45: `None` here means the sidecar records **no import settings**, and a
/// later re-import then fails with "no recorded import settings" — the author
/// loses every wizard choice with no explanation of why. It cannot happen for
/// these types (they are plain serde structs), which is exactly why a silent
/// `.ok()` was cheap and a log line is cheaper.
fn settings_table(settings: &TextureImportSettings) -> Option<toml::Table> {
    match toml::Value::try_from(settings) {
        Ok(toml::Value::Table(t)) => Some(t),
        Ok(other) => {
            tracing::warn!(
                "texture import settings serialized as {} rather than a table; the sidecar \
                 will record none and re-import will refuse",
                other.type_str()
            );
            None
        }
        Err(e) => {
            tracing::warn!(
                "texture import settings will not serialize ({e}); the sidecar will record \
                 none and re-import will refuse"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::AssetKind;

    /// An asymmetric RGBA8 source — four distinct corner quadrants — encoded as a
    /// PNG on disk. Asymmetric on purpose: a symmetric image cannot fail a test
    /// about tile coordinates.
    fn write_corner_png(dir: &Path, name: &str, w: u32, h: u32) -> std::path::PathBuf {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let c: [u8; 3] = match (x * 2 < w, y * 2 < h) {
                    (true, true) => [220, 20, 20],
                    (false, true) => [20, 200, 20],
                    (true, false) => [20, 20, 210],
                    (false, false) => [230, 210, 30],
                };
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        let path = dir.join(format!("{name}.png"));
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&rgba).unwrap();
        path
    }

    /// P26.1: an imported image lands as the **v2 tiled container**, and the v1
    /// view of it is byte-identical to the record the v1 importer would have
    /// written from the same pixels.
    #[test]
    fn an_imported_image_is_a_tiled_container_that_still_reads_as_v1() {
        let proj_dir = tempfile::tempdir().unwrap();
        // **Inside the project** (L7.H7). A source outside it records no path
        // and raises an advisory saying so, which is a different test — see
        // `a_source_outside_the_project_records_no_path`.
        let src = proj_dir.path().join("Source");
        std::fs::create_dir_all(&src).unwrap();
        let png = write_corner_png(&src, "Corners", 260, 132);

        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();
        let out = proj.import_file(&png, &dest).unwrap();
        // 260×132 is a multiple of 4 and nowhere near 8:1 — but it IS small
        // enough to pay for the uniform page (P26.5), so the tail-cost badge
        // fires and it is the ONLY one. Asserted rather than counted away: the
        // point of that advisory is that it fires here, on ordinary-looking
        // content, which is the state the P26.1 audit measured at 6.8×.
        assert_eq!(out.advisories.len(), 1, "{:?}", out.advisories);
        assert!(
            out.advisories[0].contains("tiled .inf_tex"),
            "{:?}",
            out.advisories
        );

        let id = out.primary.unwrap();
        assert_eq!(proj.db().get(id).unwrap().kind(), AssetKind::Texture);
        let bytes = std::fs::read(&proj.db().get(id).unwrap().path).unwrap();
        assert!(
            inf_material::tiles::is_v2(&bytes),
            "the payload is not a tiled image"
        );
        // The v1 view equals what the v1 writer produces from the same source.
        let rgba = inf_material::decode_image_rgba8(&std::fs::read(&png).unwrap())
            .unwrap()
            .0;
        let expected =
            inf_material::texture_from_rgba8(rgba, 260, 132, TextureImportSettings::default())
                .unwrap();
        assert_eq!(proj.load_texture(id).unwrap(), expected);

        // The tile door works on the file as written (what P26.2 pages).
        let r = inf_material::TiledTextureReader::new(bytes.as_slice()).unwrap();
        assert_eq!((r.mips()[0].tiles_x, r.mips()[0].tiles_y), (3, 2));
        assert!(r.tile(0, 2, 1).is_some());
    }

    /// **L7.H7 — a source outside the project records no path.**
    ///
    /// A sidecar is committed TOML. An absolute path in it carries the importing
    /// machine's directory layout — and, on Windows, its OS username — into
    /// every other checkout of the project. It had never fired in this repo only
    /// because every committed sample is built programmatically rather than
    /// imported.
    ///
    /// The capability given up is real and is pinned here rather than
    /// discovered: re-import needs the recorded path, so a file imported from
    /// outside the project cannot be re-imported until a copy lives under it.
    /// The advisory says exactly that at import time.
    #[test]
    fn a_source_outside_the_project_records_no_path() {
        let outside = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let png = write_corner_png(outside.path(), "Elsewhere", 260, 132);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();

        let out = proj.import_file(&png, &dest).unwrap();
        let id = out.primary.unwrap();
        let sidecar = &proj.db().get(id).unwrap().sidecar;
        assert_eq!(
            sidecar.source, None,
            "the sidecar records a path from outside the project: {:?}",
            sidecar.source
        );

        // …and it is SAID, so the lost re-import is not a silent absence.
        let note = out
            .advisories
            .iter()
            .find(|a| a.contains("outside the project"))
            .unwrap_or_else(|| panic!("no H7 advisory in {:?}", out.advisories));
        assert!(note.contains("Elsewhere"), "names the file: {note}");
        assert!(note.contains("re-import"), "names what is lost: {note}");

        // The control: an image from INSIDE the project records its path,
        // relative and forward-slashed, and raises no such note.
        //
        // **Different dimensions on purpose.** `write_corner_png` is a pure
        // function of `(w, h)`, so a same-sized image has the same bytes, the
        // same `ImportKey`, and comes straight back out of the import cache as
        // the asset above — with that asset's `source: None`. The first draft of
        // this control failed for exactly that reason and looked like the fix
        // was broken.
        let inside = proj_dir.path().join("Source");
        std::fs::create_dir_all(&inside).unwrap();
        let here = write_corner_png(&inside, "Local", 264, 132);
        let ok = proj.import_file(&here, &dest).unwrap();
        let rec = proj
            .db()
            .get(ok.primary.unwrap())
            .unwrap()
            .sidecar
            .source
            .clone();
        assert_eq!(rec.as_deref(), Some("Source/Local.png"));
        assert!(
            !ok.advisories
                .iter()
                .any(|a| a.contains("outside the project")),
            "{:?}",
            ok.advisories
        );
    }

    /// The advisories reach the outcome, all three of them, and a large
    /// well-shaped texture raises none.
    #[test]
    fn import_reports_the_dimension_and_tail_cost_advisories() {
        let proj_dir = tempfile::tempdir().unwrap();
        let src = proj_dir.path().join("Source");
        std::fs::create_dir_all(&src).unwrap();
        let awkward = write_corner_png(&src, "Strip", 126, 15);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();

        let out = proj.import_file(&awkward, &dest).unwrap();
        assert_eq!(out.advisories.len(), 3, "{:?}", out.advisories);
        assert!(out.advisories[0].contains("multiple of 4"));
        assert!(out.advisories[1].contains("8:1"));
        // P26.5's badge, and it names its own measured factor rather than a
        // remembered one.
        assert!(
            out.advisories[2].contains("tiled .inf_tex"),
            "{:?}",
            out.advisories
        );
        assert!(
            out.advisories[2].contains("its own pixels"),
            "{:?}",
            out.advisories
        );
        // The import still SUCCEEDED — an advisory is not a refusal.
        assert!(proj.db().contains(out.primary.unwrap()));

        // …and the CONTROL: a 1024² material raises nothing at all, so the three
        // above are measuring their triggers rather than an importer that warns
        // about everything.
        let fine = write_corner_png(&src, "Fine", 1024, 1024);
        let ok = proj.import_file(&fine, &dest).unwrap();
        assert!(ok.advisories.is_empty(), "{:?}", ok.advisories);
    }

    /// **Two images imported into one project are two assets** (P26.1 audit).
    ///
    /// The batch's import arms all import ONE image, or import one image into
    /// two projects. Both are satisfied by a writer that hands every asset the
    /// same GUID — which is what `write_tiled_texture`'s door did, because
    /// `AssetId::default()` is the NIL sentinel. The second import replaced the
    /// first in the database and orphaned its payload on disk, and reported
    /// success. Asserted on the database, not on the return value alone: an id
    /// can be distinct and still not be *stored*.
    #[test]
    fn two_imported_images_are_two_assets() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let a = write_corner_png(src.path(), "Alpha", 68, 68);
        let b = write_corner_png(src.path(), "Beta", 132, 68);

        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();
        let ia = proj.import_file(&a, &dest).unwrap().primary.unwrap();
        let ib = proj.import_file(&b, &dest).unwrap().primary.unwrap();

        assert!(
            !ia.is_nil() && !ib.is_nil(),
            "an import minted the NIL guid"
        );
        assert_ne!(ia, ib, "two imported images share one guid");
        assert!(proj.db().contains(ia), "the first import was evicted");
        assert!(proj.db().contains(ib));
        assert_eq!(
            proj.db()
                .iter()
                .filter(|e| e.kind() == AssetKind::Texture)
                .count(),
            2
        );
        // Both payloads are on disk AND owned — an orphan is a file the database
        // does not know about, which is exactly what the eviction produced.
        for id in [ia, ib] {
            assert!(proj.db().get(id).unwrap().path.exists());
        }
        // …and they really are different textures, so this cannot pass by both
        // ids pointing at one file.
        assert_ne!(
            proj.load_texture(ia).unwrap().width,
            proj.load_texture(ib).unwrap().width
        );
    }

    /// A re-import of the same source produces the same bytes — the encode is a
    /// pure function of its input, so the content hash is stable and the import
    /// cache is telling the truth.
    #[test]
    fn re_importing_one_source_is_byte_identical() {
        let src = tempfile::tempdir().unwrap();
        let png = write_corner_png(src.path(), "Corners", 132, 132);

        let mut bytes: Vec<Vec<u8>> = Vec::new();
        for _ in 0..2 {
            let proj_dir = tempfile::tempdir().unwrap();
            let mut proj = AssetProject::open(proj_dir.path()).unwrap();
            let dest = proj.content_dir("imported").unwrap();
            let out = proj.import_file(&png, &dest).unwrap();
            let entry = proj.db().get(out.primary.unwrap()).unwrap();
            bytes.push(std::fs::read(&entry.path).unwrap());
        }
        assert_eq!(bytes[0], bytes[1], "two imports of one source differ");
    }

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

    /// **The import door really derives** (P29.5 audit, A6).
    ///
    /// §13's S2 promise is that there is no window in which a `.inf_anim` in a
    /// project has not been measured — and until this arm, nothing asserted that
    /// the derivation was wired into the importer at all. Deriving onto a
    /// throwaway clone passed the whole `inf-editor-core` battery: the rule was
    /// unit-tested (`anim_derive`), the wiring was not.
    ///
    /// So this reads the written asset back off disk. A `.inf_anim` decoded from
    /// a glTF carries neither a root-motion track nor a `W_Gait` channel — the
    /// format has the fields but the importer has nothing to put in them — so
    /// both are proof the derivation ran on the way in.
    #[test]
    fn an_imported_clip_is_derived_before_it_is_written() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let gltf = write_skinned_gltf(src.path());

        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let dest = proj.content_dir("imported").unwrap();
        let out = proj.import_file(&gltf, &dest).unwrap();

        let clip_id = out
            .produced
            .iter()
            .copied()
            .find(|id| proj.db().get(*id).unwrap().kind() == AssetKind::AnimClip)
            .expect("the fixture imports a clip");
        let asset = proj
            .load_payload::<inf_anim::AnimClipAsset>(clip_id)
            .expect(
            "the written clip decodes — a derivation must not bake bytes its own reader refuses",
        );

        assert!(
            asset.clip.root_motion.is_some(),
            "the imported clip carries no root-motion track, so nothing derived it"
        );
        assert!(
            asset.clip.distance.is_some(),
            "…and no distance track either"
        );
        assert!(
            asset.clip.curve(inf_anim::channels::als::W_GAIT).is_some(),
            "the derived gait channel is missing: {:?}",
            asset
                .clip
                .curves
                .iter()
                .map(|c| &c.name)
                .collect::<Vec<_>>()
        );

        // …and the derivation is the one the panel's button would run: a second
        // pass over what was written moves nothing.
        let rig = crate::assets::anim_derive::skeleton_for(&proj, clip_id, &asset)
            .expect("the bound rig");
        let (again, _) = inf_anim::derive_clip(
            &asset.clip,
            &rig.skeleton,
            &inf_anim::DeriveOptions::default(),
        )
        .expect("a re-derive");
        assert_eq!(
            again, asset.clip,
            "re-deriving what the importer wrote changed it"
        );
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
