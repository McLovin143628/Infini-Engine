//! **The Infini side of the Unreal bridge** (wave ASSET0, clause 2): a
//! `manifest.json` written by `tools/ue-export/export.py` → `.inf_tex`,
//! `.inf_mat` and `.inf_mesh` assets in a project's `Content`.
//!
//! # One door, not a second importer
//!
//! Meshes go through [`super::import::import_file`] — the same call the Content
//! Drawer's drag-and-drop makes — because a bridge that decoded glTF itself
//! would be a second producer of `.inf_mesh` bytes and the two would agree only
//! until one was touched. What this module adds is everything glTF cannot
//! carry: which of five loose PNGs is a roughness map, that a Megascans surface
//! has no ORM and one has to be *packed*, which material is a tiling surface
//! rather than a mesh's skin, and where a light sits on a lamp post.
//!
//! # The PBR remap, and why it is not a pass-through
//!
//! Every Megascans instance in the reference project parents
//! `Standard_MasterMaterial` and names four slots: `albedo`, `normal`,
//! `roughness`, `displacement`. **There is no ORM anywhere** — and this engine's
//! `.inf_mat` has one `metallic_roughness_texture`, glTF-channel-ordered, which
//! is what `vt_sample.wgsl` reads. So the import PACKS one:
//! occlusion → R, roughness → G, metallic → B, through
//! [`inf_material::pack_orm`], with 255/255/0 standing in for a map the pack
//! does not ship. Downtown_West ships a real AO and a real metallic; the
//! Megascans surfaces ship neither, and both import correctly.
//!
//! [`inf_material::pack_orm`] has existed since Wave T with **no caller at
//! all**. This is its first.
//!
//! Its sibling [`inf_material::plan_map_set`] still has none, and deliberately:
//! it recovers a map's role from a FILENAME, which is the right door for a
//! folder of loose Megascans files dragged into the Content Drawer and the wrong
//! one here — the manifest states every role explicitly, so planning by name
//! would be guessing at something already known. Said rather than left as an
//! absence, because "the planner has no caller" is a fact somebody will check.
//!
//! # The clamp
//!
//! The bridge exports at source resolution, which for most of these surfaces is
//! 8 192 square — 268 MB of RGBA a map. [`UeImportOptions::max_texture`] halves
//! through the mip chain's own box filter before the tiler and the BC encode
//! run, which is where nearly all of an import's time and disk goes.
//!
//! # The REBIND, and why a material can be written at somebody else's GUID
//!
//! An imported surface is worth nothing to a level that does not name it, and
//! the levels this repository commits name the **ground library's** GUIDs — the
//! island's four `TerrainLayer::material`s and, since clause 0, its `Roads`
//! entity's `Material::asset`. Those levels are committed and their bytes are
//! locked, and the content they would have to name is licensed content that
//! must never enter this repository.
//!
//! So the bridge writes the imported material **at the committed GUID**, into
//! the local project only. `samples/ground/Road_Asphalt.inf_mat` (synthesised,
//! committed, licence-free) and `Content/Road_Asphalt.inf_mat` (Megascans, local,
//! never committed) are the same asset identity with different texels, and the
//! island level does not know or care which one it got. That is what makes the
//! public repository buildable by anyone and this machine's build photoreal.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use inf_asset::{AssetError, AssetId, Result};
use inf_material::{MapKind, MatBlend, MaterialAsset, TextureImportSettings};
use serde::Deserialize;

use super::AssetProject;

/// How to run one manifest import.
#[derive(Debug, Clone)]
pub struct UeImportOptions {
    /// Pack names to import. Empty imports every pack the manifest carries.
    pub packs: Vec<String>,
    /// Ceiling on a texture's longest side. `0` keeps the source resolution.
    pub max_texture: u32,
    /// Subfolder of the project's `Content` the assets land in.
    pub dest: String,
    /// `(asset stem, manifest material key)` — write the imported material at
    /// the GUID the committed library assigns that stem. See the module note.
    pub rebinds: Vec<(String, String)>,
    /// Import meshes as well as materials. Meshes are the slow half and a
    /// materials-only run is the common one.
    pub meshes: bool,
}

impl Default for UeImportOptions {
    fn default() -> Self {
        Self {
            packs: Vec::new(),
            // 2 048, and it is a measurement rather than a round number: the
            // Megascans surfaces here tile at 2-4 m, so 2 048 is a 1-2 mm texel
            // — the same class the committed ground library spends 1 024 to
            // reach at half the tile size, and finer than the 3.9 mm the
            // synthesised asphalt it replaces achieves.
            max_texture: 2048,
            dest: "UE".to_string(),
            rebinds: Vec::new(),
            meshes: true,
        }
    }
}

/// What one manifest import produced.
#[derive(Debug, Clone, Default)]
pub struct UeImportReport {
    /// `(manifest key, asset)` per `.inf_mat` written.
    pub materials: Vec<(String, AssetId)>,
    /// `(manifest key, asset, rungs the pack shipped, triangles imported)`.
    pub meshes: Vec<(String, AssetId, usize, usize)>,
    /// Every `.inf_tex` written.
    pub textures: Vec<AssetId>,
    /// `(stem, asset)` per rebind performed — the committed GUIDs now carrying
    /// imported texels in this project.
    pub rebinds: Vec<(String, AssetId)>,
    /// The light fixtures the manifest carried, converted to this engine's frame.
    pub fixtures: Vec<UeFixture>,
    /// Non-fatal notices, in the P16 cook-advisory shape.
    pub advisories: Vec<String>,
    /// Bytes of `.inf_tex` + `.inf_mat` + `.inf_mesh` written.
    pub bytes: u64,
}

/// A prop's light, in **this engine's units and frame**.
#[derive(Debug, Clone, PartialEq)]
pub struct UeFixture {
    /// The Blueprint it came from.
    pub name: String,
    /// The mesh key the light hangs off, when the Blueprint had one.
    pub mesh: Option<String>,
    /// Offset from the prop's origin, **metres**, in `+X east, +Y up, -Z north`.
    pub offset_m: [f64; 3],
    /// Linear colour.
    pub color: [f32; 3],
    /// Range in metres (UE's attenuation radius).
    pub range_m: f32,
    /// Candela. See [`ue_intensity_to_candela`].
    pub intensity: f32,
}

/// **UE centimetres, Z up, left handed → Infini metres, Y up, right handed.**
///
/// One function, because the conversion is the bridge's single most reversible
/// mistake: `(x, y, z)_ue → (x/100, z/100, -y/100)`. UE's own glTF exporter
/// applies exactly this to the geometry (0.01 scale, Y and Z swapped, one axis
/// negated for handedness), so a fixture converted here lands where the mesh
/// beside it landed — which is the property [`UeImportReport::fixtures`] is
/// checked against, rather than asserted.
pub fn ue_cm_to_world_m(cm: [f64; 3]) -> [f64; 3] {
    [cm[0] / 100.0, cm[2] / 100.0, -cm[1] / 100.0]
}

/// UE's point-light `Intensity` (lumens by default) → candela.
///
/// A point light radiates over 4π steradians, so `cd = lm / 4π`. UE's default
/// unit for a `PointLightComponent` is lumens and the reference lamp posts carry
/// 7 500 of them, which is 597 cd — a street lamp. Naming the conversion is the
/// difference between a lamp and a floodlight; the reference project's own
/// number is unusable without it.
pub fn ue_intensity_to_candela(lumens: f32) -> f32 {
    lumens / (4.0 * std::f32::consts::PI)
}

// ── the manifest, as this side reads it ──────────────────────────────────────
//
// Only the fields the import uses, every one `#[serde(default)]`: the manifest
// is written by a script in another repository's language and a field it grows
// must not fail an import that does not read it.

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Manifest {
    schema_version: u32,
    packs: Vec<Pack>,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Pack {
    name: String,
    license: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Mesh {
    key: String,
    pack: String,
    lods: Vec<Lod>,
    material_slots: Vec<Option<String>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Lod {
    level: u32,
    file: Option<String>,
    screen_size: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Material {
    key: String,
    pack: String,
    surface: bool,
    maps: BTreeMap<String, String>,
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    emissive: [f32; 3],
    opacity: f32,
    blend: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Texture {
    key: String,
    file: Option<String>,
    map: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Fixture {
    key: String,
    lights: Vec<Light>,
    meshes: Vec<FixtureMesh>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Light {
    location_cm: [f64; 3],
    color_srgb8: [u8; 3],
    intensity: f32,
    radius_cm: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct FixtureMesh {
    mesh: String,
}

/// The manifest schema this build reads. A newer one is refused by name rather
/// than half-read: every field here is `default`, so a bump would otherwise
/// import an empty manifest and report success.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// **Import one manifest.**
pub fn import_manifest(
    project: &mut AssetProject,
    manifest_path: &Path,
    opts: &UeImportOptions,
) -> Result<UeImportReport> {
    let raw = std::fs::read_to_string(manifest_path)?;
    let m: Manifest = serde_json::from_str(&raw)
        .map_err(|e| AssetError::Import(format!("{}: {e}", manifest_path.display())))?;
    if m.schema_version > MANIFEST_SCHEMA_VERSION {
        return Err(AssetError::Import(format!(
            "{} is manifest schema v{}, and this build reads v{MANIFEST_SCHEMA_VERSION} — \
             re-export it with this tree's tools/ue-export/export.py",
            manifest_path.display(),
            m.schema_version
        )));
    }
    let base = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let dest = project.root().join(&opts.dest);
    std::fs::create_dir_all(&dest)?;

    let wanted = |pack: &str| opts.packs.is_empty() || opts.packs.iter().any(|p| p == pack);
    let mut report = UeImportReport::default();
    for p in &m.packs {
        if wanted(&p.name) {
            report
                .advisories
                .push(format!("pack {}: licence {}", p.name, p.license));
        }
    }

    // Texture records by key, so a material can find the file behind a map name.
    let by_key: BTreeMap<&str, &Texture> = m.textures.iter().map(|t| (t.key.as_str(), t)).collect();

    // ── 1. materials, each with its own map set ──────────────────────────────
    let mut mat_ids: BTreeMap<String, AssetId> = BTreeMap::new();
    let rebind_of = |key: &str| -> Option<&str> {
        opts.rebinds
            .iter()
            .find(|(_, k)| k == key)
            .map(|(stem, _)| stem.as_str())
    };
    for mat in &m.materials {
        if !wanted(&mat.pack) {
            continue;
        }
        let stem = rebind_of(&mat.key);
        let id = import_material(project, &base, &dest, mat, &by_key, opts, stem, &mut report)?;
        mat_ids.insert(mat.key.clone(), id);
        report.materials.push((mat.key.clone(), id));
        if let Some(stem) = stem {
            report.rebinds.push((stem.to_string(), id));
        }
    }

    // ── 2. meshes, through the one importer door ─────────────────────────────
    if opts.meshes {
        for mesh in &m.meshes {
            if !wanted(&mesh.pack) {
                continue;
            }
            // LOD 0 is the asset. The coarser rungs are RECORDED and not stored:
            // see the wave ledger — every drawn `.inf_mesh` in this engine goes
            // through a derived `.inf_vmesh`, whose LOD is a continuous meshlet
            // cut, so a second authored discrete ladder would be bytes nothing
            // reads. The census is what a future wave that seeds the DAG from
            // the pack's own rungs will need, and it is in the sidecar.
            let Some(lod0) = mesh.lods.iter().find(|l| l.level == 0) else {
                report
                    .advisories
                    .push(format!("{}: no LOD 0 in the manifest", mesh.key));
                continue;
            };
            let Some(file) = lod0.file.as_ref() else {
                report
                    .advisories
                    .push(format!("{}: LOD 0 exported no file", mesh.key));
                continue;
            };
            let src = base.join(file);
            if !src.is_file() {
                report
                    .advisories
                    .push(format!("{}: {} is not on disk", mesh.key, src.display()));
                continue;
            }
            let out = super::import::import_file(project, &src, &dest)?;
            report.advisories.extend(out.advisories);
            let Some(id) = out.primary else {
                report
                    .advisories
                    .push(format!("{}: produced no mesh", mesh.key));
                continue;
            };
            let tris = project
                .load_payload::<inf_mesh::MeshAsset>(id)
                .map(|m| m.triangle_count())
                .unwrap_or(0);
            record_rungs(project, id, mesh, &mut report);
            report
                .meshes
                .push((mesh.key.clone(), id, mesh.lods.len(), tris));
        }
    }

    // ── 3. fixtures ──────────────────────────────────────────────────────────
    for f in &m.fixtures {
        for l in &f.lights {
            report.fixtures.push(UeFixture {
                name: f.key.clone(),
                mesh: f.meshes.first().map(|fm| fm.mesh.clone()),
                offset_m: ue_cm_to_world_m(l.location_cm),
                color: srgb8_to_linear(l.color_srgb8),
                range_m: l.radius_cm / 100.0,
                intensity: ue_intensity_to_candela(l.intensity),
            });
        }
    }

    report.bytes = written_bytes(project, &report);
    Ok(report)
}

/// sRGB8 → linear, the same transfer the engine uses everywhere else.
fn srgb8_to_linear(c: [u8; 3]) -> [f32; 3] {
    let f = |v: u8| {
        let s = f32::from(v) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    [f(c[0]), f(c[1]), f(c[2])]
}

/// The rung census, into the mesh's sidecar `import` table.
///
/// Sidecar-only: no payload moves, no schema window, and a human reading the
/// TOML can see what the pack shipped and what this import kept.
fn record_rungs(project: &mut AssetProject, id: AssetId, mesh: &Mesh, report: &mut UeImportReport) {
    let Some(entry) = project.db().get(id) else {
        return;
    };
    let path = entry.path.clone();
    let Ok(mut side) = inf_asset::AssetSidecar::load(&path) else {
        return;
    };
    let mut t = side.import.take().unwrap_or_default();
    t.insert("ue_source".into(), mesh.key.clone().into());
    t.insert("ue_pack".into(), mesh.pack.clone().into());
    t.insert("ue_lod_rungs".into(), (mesh.lods.len() as i64).into());
    t.insert(
        "ue_lod_screen_sizes".into(),
        toml::Value::Array(
            mesh.lods
                .iter()
                .map(|l| toml::Value::Float(f64::from(l.screen_size)))
                .collect(),
        ),
    );
    t.insert(
        "ue_material_slots".into(),
        toml::Value::Array(
            mesh.material_slots
                .iter()
                .map(|s| toml::Value::String(s.clone().unwrap_or_default()))
                .collect(),
        ),
    );
    side.import = Some(t);
    if let Err(e) = side.save(&path) {
        report
            .advisories
            .push(format!("{}: rung census not recorded ({e})", mesh.key));
    }
}

/// Total bytes on disk of everything this run produced.
fn written_bytes(project: &AssetProject, report: &UeImportReport) -> u64 {
    let mut n = 0;
    for id in report
        .textures
        .iter()
        .copied()
        .chain(report.materials.iter().map(|(_, id)| *id))
        .chain(report.meshes.iter().map(|(_, id, _, _)| *id))
    {
        if let Some(e) = project.db().get(id) {
            n += std::fs::metadata(&e.path).map(|m| m.len()).unwrap_or(0);
        }
    }
    n
}

/// One material and its whole map set.
#[allow(clippy::too_many_arguments)]
fn import_material(
    project: &mut AssetProject,
    base: &Path,
    dest: &Path,
    mat: &Material,
    by_key: &BTreeMap<&str, &Texture>,
    opts: &UeImportOptions,
    rebind: Option<&str>,
    report: &mut UeImportReport,
) -> Result<AssetId> {
    // Decode every map this material names, clamped. `BTreeMap` so the walk is
    // ordered by role name and two runs of one manifest write the same assets in
    // the same order — the GUID-stability property every import in this tree has.
    let mut planes: BTreeMap<MapKind, (Vec<u8>, u32, u32)> = BTreeMap::new();
    for (role, key) in &mat.maps {
        let Some(kind) = role_to_kind(role) else {
            continue;
        };
        let Some(tex) = by_key.get(key.as_str()) else {
            report
                .advisories
                .push(format!("{}: no texture record for {key}", mat.key));
            continue;
        };
        let Some(file) = tex.file.as_ref() else {
            continue;
        };
        let path = base.join(file);
        let Ok(bytes) = std::fs::read(&path) else {
            report
                .advisories
                .push(format!("{}: {} is not on disk", mat.key, path.display()));
            continue;
        };
        let (rgba, w, h) = inf_material::decode_image_rgba8(&bytes)
            .map_err(|e| AssetError::Import(format!("{}: {e}", path.display())))?;
        let (rgba, w, h) = inf_material::downscale_rgba8(rgba, w, h, opts.max_texture)
            .map_err(|e| AssetError::Import(format!("{}: {e}", path.display())))?;
        planes.insert(kind, (rgba, w, h));
    }

    let name = short_name(&mat.key);
    let write = |kind: MapKind,
                 slot: &str,
                 rgba: Vec<u8>,
                 w: u32,
                 h: u32,
                 project: &mut AssetProject,
                 report: &mut UeImportReport|
     -> Result<AssetId> {
        // The SLOT's settings, from the engine's own table — sRGB for exactly
        // one map, BC5 for a normal, BC1 for the rest. `source_is_float` is
        // false: everything the bridge exports is 8-bit PNG.
        let settings: TextureImportSettings = kind.settings(false);
        let image = inf_material::build_tiled_texture(rgba, w, h, settings)
            .map_err(|e| AssetError::Import(format!("{}_{slot}: {e}", mat.key)))?;
        let id = project.write_tiled_texture(
            dest,
            &format!("{name}_{slot}"),
            &image,
            Some(mat.source_note()),
            None,
        )?;
        report.textures.push(id);
        Ok(id)
    };

    let albedo = match planes.remove(&MapKind::Albedo) {
        Some((px, w, h)) => Some(write(MapKind::Albedo, "Albedo", px, w, h, project, report)?),
        None => None,
    };
    let normal = match planes.remove(&MapKind::Normal) {
        Some((px, w, h)) => Some(write(MapKind::Normal, "Normal", px, w, h, project, report)?),
        None => None,
    };

    // **The ORM, packed.** Occlusion → R, roughness → G, metallic → B, with
    // 255/255/0 where the pack ships nothing — which is every Megascans surface
    // in this project, none of which has an AO or a metallic map. The extent is
    // the SMALLEST of the three: packing a 2 048 roughness into a 4 096 grid
    // would read past the end of it, and `pack_orm` refuses that rather than
    // guessing (it returns `None`).
    let orm_planes = [MapKind::Occlusion, MapKind::Roughness, MapKind::Metallic];
    let orm = if orm_planes.iter().any(|k| planes.contains_key(k)) {
        let (ew, eh) = orm_planes
            .iter()
            .filter_map(|k| planes.get(k))
            .map(|(_, w, h)| (*w, *h))
            .fold((u32::MAX, u32::MAX), |(aw, ah), (w, h)| {
                (aw.min(w), ah.min(h))
            });
        let plane = |k: MapKind| -> Result<Option<Vec<u8>>> {
            let Some((px, w, h)) = planes.get(&k) else {
                return Ok(None);
            };
            if (*w, *h) == (ew, eh) {
                return Ok(Some(px.clone()));
            }
            let (px, _, _) = inf_material::downscale_rgba8(px.clone(), *w, *h, ew.max(eh))
                .map_err(|e| AssetError::Import(format!("{}: {e}", mat.key)))?;
            Ok(Some(px))
        };
        let o = plane(MapKind::Occlusion)?;
        let r = plane(MapKind::Roughness)?;
        let mt = plane(MapKind::Metallic)?;
        match inf_material::pack_orm(o.as_deref(), r.as_deref(), mt.as_deref(), ew, eh) {
            Some(px) => Some(write(
                MapKind::Roughness,
                "ORM",
                px,
                ew,
                eh,
                project,
                report,
            )?),
            None => {
                report.advisories.push(format!(
                    "{}: its occlusion/roughness/metallic maps are not one size, so no ORM \
                     was packed and the material falls back to its scalar roughness",
                    mat.key
                ));
                None
            }
        }
    } else {
        None
    };

    let asset = MaterialAsset {
        schema_version: MaterialAsset::CURRENT_VERSION,
        base_color: mat.base_color,
        metallic: mat.metallic,
        // A Megascans instance carries `roughness = 1.0` as the MULTIPLIER on
        // its roughness map, not as a roughness. Kept as-is when a map is bound
        // (the map is the signal); it is only the fallback that matters, and a
        // surface with no map that says 1.0 really is fully rough.
        roughness: mat.roughness,
        emissive: mat.emissive,
        base_color_texture: albedo,
        normal_texture: normal,
        metallic_roughness_texture: orm,
        blend: match mat.blend.as_str() {
            "masked" => MatBlend::Masked,
            "blend" => MatBlend::Translucent,
            _ => MatBlend::Opaque,
        },
        ..Default::default()
    };
    let deps = asset.texture_dependencies();
    match rebind {
        // **At the committed GUID**, and at the committed FILE NAME, so the
        // asset scan finds one asset rather than two claiming one id.
        Some(stem) => {
            let guid = crate::ground::ground_material_guid(stem_kind(stem).ok_or_else(|| {
                AssetError::Import(format!(
                    "{stem} is not a ground-library surface; the stems are {}",
                    inf_material::ground::GroundKind::ALL
                        .iter()
                        .map(|k| k.stem())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?);
            let path = project.root().join(format!("{stem}.inf_mat"));
            let id = project.write_asset_at_with_id(&path, &asset, AssetId(guid), deps, None)?;
            report.advisories.push(format!(
                "{stem} now carries {} in this project — the committed synthesised \
                 material of the same GUID is overwritten LOCALLY and stays unchanged in \
                 the repository",
                mat.key
            ));
            Ok(id)
        }
        None => project.write_asset(dest, &name, &asset, Some(mat.source_note()), deps, None),
    }
}

/// The ground-library kind a rebind stem names.
fn stem_kind(stem: &str) -> Option<inf_material::ground::GroundKind> {
    inf_material::ground::GroundKind::ALL
        .into_iter()
        .find(|k| k.stem() == stem)
}

/// A manifest role name → the engine's [`MapKind`].
///
/// The export script's own vocabulary, mapped once. `displacement` is
/// deliberately absent: this engine has no displacement slot on `.inf_mat`, and
/// importing a height map as a texture nothing samples would be 2 MB an asset
/// for a channel no shader reads.
fn role_to_kind(role: &str) -> Option<MapKind> {
    Some(match role {
        "albedo" => MapKind::Albedo,
        "normal" => MapKind::Normal,
        "roughness" => MapKind::Roughness,
        "metallic" => MapKind::Metallic,
        "ao" => MapKind::Occlusion,
        "opacity" => MapKind::Opacity,
        "emissive" => MapKind::Albedo,
        _ => return None,
    })
}

/// A readable asset name out of a manifest key.
///
/// The keys are full object paths with the separators flattened and are up to
/// 150 characters long; a Content Drawer full of those is unusable. The last two
/// path-ish segments are what a human recognises.
fn short_name(key: &str) -> String {
    let parts: Vec<&str> = key.split('_').collect();
    let tail = if parts.len() > 6 {
        parts[parts.len() - 6..].join("_")
    } else {
        key.to_string()
    };
    tail.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

impl Material {
    /// What the sidecar records as this asset's source. Not a path — the source
    /// is in another engine's content tree and re-importing it needs the whole
    /// bridge — so it is the UE object path, which is what a human would look up.
    fn source_note(&self) -> String {
        format!("ue:{}", self.key)
    }
}
