//! **THE UNREAL BRIDGE GATE** (wave ASSET0) — the door proven on content this
//! repository owns.
//!
//! # Why the fixture is synthetic, and why that is not a weaker test
//!
//! The bridge's real input is 2.9 GB of Megascans and Marketplace content whose
//! licence for use outside Unreal is unestablished (see
//! `tools/ue-export/export.py`'s `PACKS`). None of it may enter this repository,
//! so none of it can be in CI.
//!
//! What CI can hold is a manifest and a map set written HERE, by this file, in
//! the exact shape `export.py` emits — a glTF quad, four PNGs at three
//! different extents, a material naming them by role, a mesh with four LOD
//! rungs, and a Blueprint fixture with a light on it. Every decision the bridge
//! makes is a function of that shape rather than of the pixels: which map is
//! sRGB, that a roughness and an occlusion map with no metallic pack into an
//! ORM with 0 in blue, that an 8 192 source clamps by halving through the mip
//! filter, that a `--bind` lands a material at a GUID a committed level names.
//!
//! So this gate is written to FALSIFY: each arm below names the mutation that
//! breaks it, and the numbers are asserted, not printed.

use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind};
use inf_editor_core::assets::ue_import::{
    engine_checkout_above, import_manifest, ue_cm_to_world_m, ue_intensity_to_candela,
    UeImportOptions,
};
use inf_editor_core::assets::AssetProject;
use inf_material::ground::GroundKind;

// ── the fixture ──────────────────────────────────────────────────────────────

/// A solid-colour RGBA8 PNG at `extent` square, written to `path`.
///
/// Not a flat fill: a two-tone checker, so a downscale that returned its input
/// and a downscale that averaged correctly do not produce the same bytes, and
/// so the BC encoder has something to encode.
fn png(path: &Path, extent: u32, a: [u8; 4], b: [u8; 4]) {
    let mut px = vec![0u8; (extent * extent * 4) as usize];
    for y in 0..extent {
        for x in 0..extent {
            let c = if ((x / 4) + (y / 4)) % 2 == 0 { a } else { b };
            let o = ((y * extent + x) * 4) as usize;
            px[o..o + 4].copy_from_slice(&c);
        }
    }
    let img: image::RgbaImage = image::ImageBuffer::from_raw(extent, extent, px).unwrap();
    img.save(path).unwrap();
}

/// A one-quad glTF (two triangles, four vertices) plus its `.bin`.
///
/// `scale` shrinks it, so the four "LOD" files a manifest names are four
/// different files rather than one copied four times — which is what makes the
/// "LOD 0 is the asset" arm below able to tell which one was imported.
fn quad_gltf(dir: &Path, stem: &str, scale: f32) -> PathBuf {
    let positions: [f32; 12] = [
        0.0, 0.0, 0.0, scale, 0.0, 0.0, scale, scale, 0.0, 0.0, scale, 0.0,
    ];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let mut buf: Vec<u8> = Vec::new();
    for v in positions {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let pos_len = positions.len() * 4;
    let idx_off = buf.len();
    for v in indices {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
    let bin_name = format!("{stem}.bin");
    let json = format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "name": "{stem}", "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "indices": 1 }}] }}],
  "buffers": [{{ "uri": "{bin_name}", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": {pos_len} }},
    {{ "buffer": 0, "byteOffset": {idx_off}, "byteLength": 12 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
       "min": [0,0,0], "max": [{scale},{scale},0] }},
    {{ "bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR" }}
  ]
}}"#,
        total = buf.len(),
    );
    std::fs::write(dir.join(&bin_name), &buf).unwrap();
    let path = dir.join(format!("{stem}.gltf"));
    std::fs::write(&path, json).unwrap();
    path
}

/// The staging directory `export.py` would have written, and the manifest that
/// describes it.
///
/// The map set is deliberately UNEVEN, because the real ones are: the albedo is
/// 8 192 (a Megascans surface), the normal 4 096, the roughness and occlusion
/// 2 048, and there is no metallic map at all. Every one of those is a case the
/// importer has to get right, and a fixture where they all matched would test
/// none of them.
fn staging(dir: &Path) -> PathBuf {
    let tex = dir.join("textures");
    let meshes = dir.join("meshes");
    std::fs::create_dir_all(&tex).unwrap();
    std::fs::create_dir_all(&meshes).unwrap();

    png(
        &tex.join("Surf_Albedo.png"),
        512,
        [190, 60, 40, 255],
        [120, 40, 30, 255],
    );
    png(
        &tex.join("Surf_Normal.png"),
        256,
        [128, 128, 255, 255],
        [140, 118, 250, 255],
    );
    png(
        &tex.join("Surf_Roughness.png"),
        128,
        [200, 200, 200, 255],
        [90, 90, 90, 255],
    );
    png(
        &tex.join("Surf_AO.png"),
        128,
        [255, 255, 255, 255],
        [180, 180, 180, 255],
    );
    // **Two roles this engine has nowhere to put** (ASSET0 audit), at 64 square
    // so the extent alone says which file reached a slot. `emissive` used to be
    // mapped onto `MapKind::Albedo`, and `planes` is keyed by kind over a
    // `BTreeMap` of role names, so it REPLACED the albedo it sorts after.
    png(
        &tex.join("Surf_Emissive.png"),
        64,
        [255, 80, 0, 255],
        [200, 60, 0, 255],
    );
    png(
        &tex.join("Surf_Opacity.png"),
        64,
        [255, 255, 255, 255],
        [0, 0, 0, 255],
    );

    for (lod, scale) in [(0, 1.0f32), (1, 0.9), (2, 0.8), (3, 0.7)] {
        quad_gltf(&meshes, &format!("Prop_LOD{lod}"), scale);
    }

    let manifest = serde_json::json!({
        "schema_version": 1,
        "generator": "the ASSET0 gate",
        "packs": [{ "name": "Fixture", "license": "this repository's own" }],
        "textures": [
            {"key": "t_albedo", "file": "textures/Surf_Albedo.png", "map": "albedo",
             "width": 512, "height": 512, "srgb": true},
            {"key": "t_normal", "file": "textures/Surf_Normal.png", "map": "normal",
             "width": 256, "height": 256, "srgb": false},
            {"key": "t_rough", "file": "textures/Surf_Roughness.png", "map": "roughness",
             "width": 128, "height": 128, "srgb": false},
            {"key": "t_ao", "file": "textures/Surf_AO.png", "map": "ao",
             "width": 128, "height": 128, "srgb": false},
            {"key": "t_emissive", "file": "textures/Surf_Emissive.png", "map": "emissive",
             "width": 64, "height": 64, "srgb": true},
            {"key": "t_opacity", "file": "textures/Surf_Opacity.png", "map": "opacity",
             "width": 64, "height": 64, "srgb": false}
        ],
        "materials": [{
            "key": "Fixture_Surface", "pack": "Fixture", "surface": true,
            "base_color": [1.0, 1.0, 1.0, 1.0], "metallic": 0.0, "roughness": 1.0,
            "emissive": [0.0, 0.0, 0.0], "opacity": 1.0, "blend": "opaque",
            "maps": {"albedo": "t_albedo", "normal": "t_normal",
                     "roughness": "t_rough", "ao": "t_ao",
                     "displacement": "t_rough",
                     "emissive": "t_emissive", "opacity": "t_opacity"}
        }],
        "meshes": [{
            "key": "Fixture_Prop", "pack": "Fixture", "nanite": false,
            "material_slots": ["Fixture_Surface"],
            "lods": [
                {"level": 0, "file": "meshes/Prop_LOD0.gltf", "screen_size": 1.0},
                {"level": 1, "file": "meshes/Prop_LOD1.gltf", "screen_size": 0.5},
                {"level": 2, "file": "meshes/Prop_LOD2.gltf", "screen_size": 0.25},
                {"level": 3, "file": "meshes/Prop_LOD3.gltf", "screen_size": 0.1}
            ]
        }],
        "fixtures": [{
            "key": "Fixture_Lamp", "pack": "Fixture",
            "meshes": [{"mesh": "Fixture_Prop", "component": "SM", "location_cm": [0.0, 0.0, 0.0]}],
            "lights": [{
                "name": "PointLight", "kind": "point",
                "location_cm": [100.0, 200.0, 550.0],
                "rotation_deg": [0.0, 0.0, 0.0],
                "color_srgb8": [255, 224, 160],
                "intensity": 7500.0, "radius_cm": 1500.0
            }]
        }],
        "errors": []
    });
    let path = dir.join("manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    path
}

// ── the arms ─────────────────────────────────────────────────────────────────

/// **The whole round trip.** A manifest in, a textured material and a real mesh
/// out, with the ORM PACKED out of maps no single file carries.
///
/// Un-fix mutations that break it, one per assertion block:
///
/// * dropping the ORM pack → `metallic_roughness_texture` is `None`;
/// * packing roughness into R instead of G → the blue-is-zero and
///   green-is-the-roughness arms;
/// * importing a rung other than LOD 0 → the vertex positions;
/// * classifying displacement as a texture → the texture count.
#[test]
fn a_manifest_becomes_a_textured_material_and_a_mesh() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = staging(dir.path());
    let content = dir.path().join("project").join("Content");
    let mut project = AssetProject::open(&content).expect("a project opens");

    let report = import_manifest(&mut project, &manifest, &UeImportOptions::default())
        .expect("the manifest imports");

    // ── the material ─────────────────────────────────────────────────────────
    assert_eq!(report.materials.len(), 1, "one material in, one out");
    let (key, mat_id) = report.materials[0].clone();
    assert_eq!(key, "Fixture_Surface");
    let mat: inf_material::MaterialAsset = project.load_payload(mat_id).expect("it decodes");
    assert!(mat.base_color_texture.is_some(), "no albedo bound");
    assert!(mat.normal_texture.is_some(), "no normal bound");
    assert!(
        mat.metallic_roughness_texture.is_some(),
        "NO ORM WAS PACKED -- the pack ships a roughness and an occlusion map and no \
         ORM, and packing one is the whole PBR remap"
    );

    // **THREE textures, not seven.** The manifest names seven map roles and
    // three of them (`displacement`, `emissive`, `opacity`) have no slot in this
    // engine's `.inf_mat`, while roughness and occlusion are consumed INTO the
    // ORM rather than written beside it. A larger count would mean two megabytes
    // an asset of channels no shader samples.
    assert_eq!(
        report.textures.len(),
        3,
        "expected albedo + normal + packed ORM, got {} textures",
        report.textures.len()
    );

    // **AND THE ALBEDO IS THE ALBEDO** (ASSET0 audit). `emissive` was mapped
    // onto `MapKind::Albedo`, and since `planes` is keyed by kind while
    // `maps` is a `BTreeMap` of role names, "emissive" sorted after "albedo"
    // and REPLACED it: the material would have shipped its glow map as its base
    // colour and every count above would still read three. The fixture's
    // emissive is 64 square and its albedo 512, so the extent is the witness.
    let albedo = project
        .load_texture(mat.base_color_texture.unwrap())
        .expect("the albedo decodes");
    assert_eq!(
        (albedo.width, albedo.height),
        (512, 512),
        "the base colour is {}x{} -- a 64-square map reached the albedo slot, so \
         a role with no slot of its own clobbered it",
        albedo.width,
        albedo.height
    );

    // …and the roles with nowhere to go are REPORTED, because "it imported" and
    // "it imported with half its maps" read identically without this.
    let unplaced = report
        .advisories
        .iter()
        .find(|a| a.contains("has no slot"))
        .unwrap_or_else(|| panic!("no advisory named the unplaced maps: {:?}", report.advisories));
    for role in ["displacement", "emissive", "opacity"] {
        assert!(
            unplaced.contains(role),
            "the advisory does not name {role}: {unplaced}"
        );
    }
    for id in &report.textures {
        assert_eq!(
            project.db().get(*id).map(|e| e.kind()),
            Some(AssetKind::Texture)
        );
    }

    // ── the ORM's channels really are O/R/M ──────────────────────────────────
    //
    // Read back through the texture door rather than trusted: this is the one
    // place a channel order could be wrong and everything still "work".
    let orm = project
        .load_texture(mat.metallic_roughness_texture.unwrap())
        .expect("the ORM decodes");
    assert_eq!(
        (orm.width, orm.height),
        (128, 128),
        "the ORM is the SMALLEST of its inputs -- packing a 128 roughness into a \
         512 grid reads past the end of it"
    );
    assert!(!orm.srgb, "an ORM is data, not a colour");

    // …**and the channels are read, not assumed** (ASSET0 audit). Until this
    // block the section titled "the ORM's channels really are O/R/M" asserted
    // an EXTENT and a colour-space flag and never looked at a texel, so the one
    // thing its own comment calls "the one place a channel order could be wrong
    // and everything still work" was the one thing it did not check: swapping
    // `pack_orm`'s first two arguments passed every assertion in this file.
    //
    // The fixture makes the three channels tell each other apart: occlusion is
    // a {255, 180} checker (mean 217.5), roughness a {200, 90} one (mean 145),
    // and there is no metallic map at all. The tolerance is BC1's — the checker
    // is on 4-px blocks, so every block is a solid colour and 5:6:5 is the whole
    // of the error.
    let px = orm.level_rgba8(0).expect("the ORM's level 0 decodes");
    let mean = |c: usize| -> f64 {
        px.chunks_exact(4).map(|p| f64::from(p[c])).sum::<f64>() / (px.len() / 4) as f64
    };
    let (r, g, b) = (mean(0), mean(1), mean(2));
    assert!(
        (r - 217.5).abs() < 12.0,
        "ORM red is {r:.1}, not the occlusion map's 217.5 -- O/R/M is not the \
         packed order (roughness would read 145)"
    );
    assert!(
        (g - 145.0).abs() < 12.0,
        "ORM green is {g:.1}, not the roughness map's 145.0 -- glTF puts \
         roughness in GREEN and every lit shader in this engine reads it there"
    );
    assert!(
        b < 1.0,
        "ORM blue is {b:.1}, not 0 -- the pack ships no metallic map, and a \
         non-zero blue is every surface here turning into metal"
    );
    assert!(
        px.chunks_exact(4).all(|p| p[2] == 0),
        "some texel carries a metallic value the pack never shipped"
    );

    // ── the mesh ─────────────────────────────────────────────────────────────
    assert_eq!(report.meshes.len(), 1);
    let (mkey, mesh_id, rungs, tris) = report.meshes[0].clone();
    assert_eq!(mkey, "Fixture_Prop");
    assert_eq!(
        rungs, 4,
        "the census must record every rung the pack shipped"
    );
    assert_eq!(tris, 2, "a quad is two triangles");
    let mesh: inf_mesh::MeshAsset = project.load_payload(mesh_id).expect("the mesh decodes");
    // **LOD 0 and not another rung.** The four fixture rungs are 1.0, 0.9, 0.8
    // and 0.7 units across, so the bound says which file was read.
    let w = mesh.bounds.max[0] - mesh.bounds.min[0];
    assert!(
        (w - 1.0).abs() < 1e-5,
        "the imported rung is {w} units across, so it is not LOD 0"
    );

    // ── the rung census reached the sidecar ──────────────────────────────────
    let path = project.db().get(mesh_id).expect("registered").path.clone();
    let side = inf_asset::AssetSidecar::load(&path).expect("a sidecar");
    let import = side.import.expect("an import table");
    assert_eq!(
        import.get("ue_lod_rungs").and_then(toml::Value::as_integer),
        Some(4),
        "the sidecar does not record what the pack shipped"
    );
    assert_eq!(
        import.get("ue_source").and_then(toml::Value::as_str),
        Some("Fixture_Prop")
    );

    // ── the fixture ──────────────────────────────────────────────────────────
    assert_eq!(report.fixtures.len(), 1);
    let f = &report.fixtures[0];
    // UE (100, 200, 550) cm -> (1, 5.5, -2) m. A lamp head five and a half
    // metres up, which is a lamp post; the same numbers read as (1, 2, 5.5) if
    // the axes are swapped the other way, which is a lamp lying in the road.
    assert!((f.offset_m[0] - 1.0).abs() < 1e-9);
    assert!((f.offset_m[1] - 5.5).abs() < 1e-9);
    assert!((f.offset_m[2] + 2.0).abs() < 1e-9);
    assert!(
        (f.intensity - 7500.0 / (4.0 * std::f32::consts::PI)).abs() < 1e-3,
        "7 500 lumens is 597 candela; taking the lumens as candela is a 4pi-times \
         floodlight"
    );
    assert!((f.range_m - 15.0).abs() < 1e-6);
    assert_eq!(f.mesh.as_deref(), Some("Fixture_Prop"));
    // The colour crosses UNCONVERTED, as 8-bit sRGB. See
    // `UeFixture::color_srgb8` for why: the transfer function is a `powf`
    // and this crate's portable-math gate refuses one, so the bridge carries
    // the source value and the conversion happens where a `Light` is
    // authored from it.
    assert_eq!(f.color_srgb8, [255, 224, 160]);

    println!(
        "ASSET0 GATE: {} materials, {} textures, {} meshes ({} tris), {} fixtures, {} bytes",
        report.materials.len(),
        report.textures.len(),
        report.meshes.len(),
        tris,
        report.fixtures.len(),
        report.bytes
    );
}

/// **A SECOND IMPORT OF THE SAME MANIFEST WRITES THE SAME ASSETS.**
///
/// An importer whose output is a pure function of its input has to overwrite its
/// own output, and this one did not: `write_asset` and `write_tiled_texture` both
/// call `unique_asset_path`, which is right for an author importing a file twice
/// on purpose and wrong for a tool that re-runs. Measured on the real island
/// project before the fix — **106 duplicate assets**, `X_1.inf_tex` beside
/// `X.inf_tex` under fresh GUIDs, doubling the texture bytes and leaving the
/// first copy referenced by nothing.
///
/// The assertion is on the IDS and on the file count, not on "it did not error",
/// because the duplicating version did not error either. Un-fix mutation:
/// swapping either writer back to its `unique_asset_path` door fails both.
#[test]
fn importing_one_manifest_twice_writes_one_set_of_assets() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = staging(dir.path());
    let content = dir.path().join("project").join("Content");

    let count = |c: &Path| -> usize {
        walk(c)
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("inf_tex") | Some("inf_mat") | Some("inf_mesh")
                )
            })
            .count()
    };

    let mut project = AssetProject::open(&content).expect("a project opens");
    let first = import_manifest(&mut project, &manifest, &UeImportOptions::default())
        .expect("the first import");
    let after_first = count(&content);
    drop(project);

    // A FRESH project over the same directory, which is what a second
    // `inf-import` run is: the database is rebuilt by scanning what is there.
    let mut project = AssetProject::open(&content).expect("it re-opens");
    let second = import_manifest(&mut project, &manifest, &UeImportOptions::default())
        .expect("the second import");
    let after_second = count(&content);

    assert_eq!(
        after_first,
        after_second,
        "the second import wrote {} more assets than the first",
        after_second as i64 - after_first as i64
    );
    assert_eq!(
        first.materials, second.materials,
        "a material's GUID moved between two imports of one manifest"
    );
    assert_eq!(
        first.textures, second.textures,
        "a texture's GUID moved between two imports of one manifest"
    );
    assert_eq!(
        first.meshes.iter().map(|m| m.1).collect::<Vec<_>>(),
        second.meshes.iter().map(|m| m.1).collect::<Vec<_>>(),
        "a mesh's GUID moved -- the import cache did not serve the second run"
    );
    println!("ASSET0 GATE: two imports, {after_first} assets both times, same ids");
}

/// Every file under `dir`, recursively — the gate's own walk, because counting
/// what an importer wrote is the only way to see a duplicate it did not report.
fn walk(dir: &Path) -> impl Iterator<Item = PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    out.into_iter()
}

/// **The clamp halves through the mip chain's own filter.**
///
/// Not "it got smaller": the clamped import must be bit-identical to the
/// unclamped import's mip at that extent, which is what makes
/// `--max-texture` a choice about *which mip is level 0* rather than a second,
/// private resampler. A box filter written beside the caller would pass a
/// "got smaller" assertion and fail this one.
#[test]
fn the_texture_clamp_is_the_mip_chain_and_not_a_second_filter() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("t.png");
    png(&p, 512, [200, 40, 10, 255], [30, 90, 220, 255]);
    let bytes = std::fs::read(&p).unwrap();
    let (rgba, w, h) = inf_material::decode_image_rgba8(&bytes).unwrap();

    let (small, sw, sh) = inf_material::downscale_rgba8(rgba.clone(), w, h, 128).unwrap();
    assert_eq!((sw, sh), (128, 128));

    // The same image, imported unclamped, at the level whose extent is 128.
    let full = inf_material::texture_from_rgba8(
        rgba.clone(),
        w,
        h,
        inf_material::TextureImportSettings {
            compression: inf_material::TextureCompression::None,
            ..Default::default()
        },
    )
    .unwrap();
    let level = full
        .mips
        .iter()
        .find(|m| m.width == 128)
        .expect("a 128 mip exists");
    assert_eq!(
        level.data, small,
        "the clamp and the mip chain disagree -- there are two box filters in \
         this tree and they have already drifted"
    );

    // …and it is a no-op when the source already fits, byte for byte.
    let (same, aw, ah) = inf_material::downscale_rgba8(rgba.clone(), w, h, 4096).unwrap();
    assert_eq!((aw, ah, same), (w, h, rgba));
}

/// **A rebind writes an imported material at a COMMITTED GUID.**
///
/// This is the clause that lets the public repository ship a licence-free
/// synthesised road and this machine's build wear a photographed one, with the
/// committed `.inf_lvl` naming one id either way. The assertion is on the id,
/// because everything else about the arrangement is prose.
///
/// Un-fix mutation: minting a fresh id in `write_asset_at_with_id` (which is
/// what `write_asset_at` does when the path is empty, and what this run's path
/// is) fails the first assertion.
#[test]
fn a_rebind_lands_on_the_ground_librarys_own_guid() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = staging(dir.path());
    let content = dir.path().join("project").join("Content");
    let mut project = AssetProject::open(&content).expect("a project opens");

    let opts = UeImportOptions {
        rebinds: vec![("Road_Asphalt".into(), "Fixture_Surface".into())],
        meshes: false,
        ..Default::default()
    };
    let report = import_manifest(&mut project, &manifest, &opts).expect("it imports");

    let want = inf_editor_core::ground::ground_material_guid(GroundKind::Asphalt);
    assert_eq!(report.rebinds.len(), 1);
    assert_eq!(
        report.rebinds[0],
        ("Road_Asphalt".to_string(), AssetId(want)),
        "the rebind did not land on the committed asphalt GUID"
    );
    // …and it is really on disk under the name a `[content]` copy would use, so
    // the island build's own copy of the synthesised material is the file this
    // overwrote rather than a second asset beside it.
    let path = content.join("Road_Asphalt.inf_mat");
    assert!(path.is_file(), "{} was not written", path.display());
    let side = inf_asset::AssetSidecar::load(&path).expect("a sidecar");
    assert_eq!(side.guid, AssetId(want));
    let mat: inf_material::MaterialAsset =
        inf_asset::decode(&std::fs::read(&path).unwrap()).expect("it decodes");
    assert!(
        mat.base_color_texture.is_some() && mat.metallic_roughness_texture.is_some(),
        "the rebound material carries no imported texels, which makes the whole \
         arrangement pointless"
    );

    // A stem that is not in the library is REFUSED by name rather than silently
    // minting an id nothing reads.
    let bad = UeImportOptions {
        rebinds: vec![("Ground_Marzipan".into(), "Fixture_Surface".into())],
        meshes: false,
        ..Default::default()
    };
    let err = import_manifest(&mut project, &manifest, &bad)
        .expect_err("an unknown stem must be refused")
        .to_string();
    assert!(
        err.contains("Ground_Marzipan") && err.contains("Road_Asphalt"),
        "the refusal must name the stem and the alternatives: {err}"
    );
}

/// **The unit conversions are the ones the geometry uses.**
///
/// Small and separate on purpose: `ue_cm_to_world_m` is the single most
/// reversible mistake in the bridge and the fixture arm above exercises it at
/// one point. This one pins the axes independently, so a sign flip cannot hide
/// behind a symmetric test point.
#[test]
fn the_frame_conversion_is_stated_axis_by_axis() {
    assert_eq!(ue_cm_to_world_m([100.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    assert_eq!(ue_cm_to_world_m([0.0, 100.0, 0.0]), [0.0, 0.0, -1.0]);
    assert_eq!(ue_cm_to_world_m([0.0, 0.0, 100.0]), [0.0, 1.0, 0.0]);
    // UE is left handed and this engine is right handed, so exactly ONE axis
    // flips. Two would be a rotation and none would be a mirror.
    let flipped = [
        ue_cm_to_world_m([100.0, 0.0, 0.0])[0] < 0.0,
        ue_cm_to_world_m([0.0, 0.0, 100.0])[1] < 0.0,
        ue_cm_to_world_m([0.0, 100.0, 0.0])[2] < 0.0,
    ];
    assert_eq!(flipped.iter().filter(|f| **f).count(), 1);
    assert!((ue_intensity_to_candela(7500.0) - 596.83).abs() < 0.01);
}

/// **A manifest from a newer bridge is refused, not half-read.**
///
/// Every field on this side is `#[serde(default)]`, which is what makes a
/// manifest that grows a key survive — and would also make a manifest whose
/// SHAPE changed import as empty and report success. The version check is what
/// separates the two.
#[test]
fn a_newer_manifest_schema_is_refused_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = staging(dir.path());
    let text = std::fs::read_to_string(&manifest).unwrap();
    let bumped = text.replace(r#""schema_version": 1"#, r#""schema_version": 99"#);
    assert_ne!(bumped, text, "the fixture manifest states its version");
    std::fs::write(&manifest, bumped).unwrap();

    let content = dir.path().join("project").join("Content");
    let mut project = AssetProject::open(&content).expect("a project opens");
    let err = import_manifest(&mut project, &manifest, &UeImportOptions::default())
        .expect_err("a v99 manifest must be refused")
        .to_string();
    assert!(
        err.contains("v99"),
        "the refusal must name the version: {err}"
    );
    assert!(
        err.contains("export.py"),
        "the refusal must name the remedy: {err}"
    );
}

/// **THE LICENCE LAW IS A DOOR, NOT A SENTENCE** (ASSET0 audit).
///
/// The wave's own rule — *nothing this bridge writes may enter this
/// repository* — was three sentences of documentation and the author's care.
/// Measured at the audit: `export.py` believed `INF_UE_OUT` and
/// `import_manifest` believed its destination, so one mistyped path put
/// hundreds of megabytes of Megascans-derived `.inf_tex` in the working tree of
/// a PUBLIC repository, untracked and one `git add -A` from publication.
///
/// This arm builds a directory shaped like the engine's checkout — a `.git`
/// beside `tools/ue-export/export.py`, which is the very script that produces
/// the content the law is about — and asserts the import refuses it **before
/// decoding anything**, then imports the same manifest into a sibling directory
/// to prove the refusal discriminates rather than simply failing.
///
/// Un-fix mutation: delete the `engine_checkout_above` guard at the top of
/// `import_manifest` and the first block fails on the `expect_err`.
#[test]
fn the_bridge_refuses_to_write_into_the_engine_checkout() {
    let dir = tempfile::tempdir().unwrap();
    let staging_dir = dir.path().join("ue-staging");
    std::fs::create_dir_all(&staging_dir).unwrap();
    let manifest = staging(&staging_dir);

    let checkout = dir.path().join("infinity_engine");
    std::fs::create_dir_all(checkout.join("tools").join("ue-export")).unwrap();
    std::fs::create_dir_all(checkout.join(".git")).unwrap();
    std::fs::write(
        checkout.join("tools").join("ue-export").join("export.py"),
        b"# the marker this repository is recognised by\n",
    )
    .unwrap();

    // ── inside the checkout: refused, and nothing written ────────────────────
    let inside = checkout.join("samples").join("ground");
    let mut project = AssetProject::open(&inside).expect("a project opens");
    let err = import_manifest(&mut project, &manifest, &UeImportOptions::default())
        .expect_err("an import into the engine checkout must be refused")
        .to_string();
    assert!(
        err.contains("refusing to import into"),
        "the refusal must say what it refused: {err}"
    );
    assert!(
        err.contains("may enter this repository"),
        "the refusal must say WHY, because the why is a licence: {err}"
    );
    let written = walk(&inside)
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("inf_tex") | Some("inf_mat") | Some("inf_mesh")
            )
        })
        .count();
    assert_eq!(
        written, 0,
        "the refusal must come BEFORE the first decode -- {written} assets \
         reached a public repository's working tree"
    );

    // ── outside it: the same manifest imports ────────────────────────────────
    let outside = dir.path().join("island-build").join("project").join("Content");
    let mut project = AssetProject::open(&outside).expect("a project opens");
    let report = import_manifest(&mut project, &manifest, &UeImportOptions::default())
        .expect("outside the checkout the same manifest imports");
    assert_eq!(report.materials.len(), 1, "the control must really import");

    // …and the detector names the checkout rather than any directory with a
    // `.git` in it: a user's own game repository is not this one.
    let found = engine_checkout_above(&inside).expect("the checkout is found");
    assert!(
        found.ends_with("infinity_engine"),
        "found {} rather than the checkout",
        found.display()
    );
    assert!(
        engine_checkout_above(&outside).is_none(),
        "a sibling of the checkout is not inside it"
    );
    println!("ASSET0 GATE: the bridge refuses {}", inside.display());
}
