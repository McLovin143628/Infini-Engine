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
//!
//! # …and the law that arrangement rests on is enforced here
//!
//! [`import_manifest`] **refuses a destination inside the engine checkout**,
//! before it decodes anything — see [`engine_checkout_above`]. The audit found
//! the rule stated in three places and checked in none.

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
    /// How many LOD rungs of a **character** to store (wave CHAR1a).
    ///
    /// Three, and the number is a consequence rather than a taste: a skinned
    /// mesh never reaches the meshlet path (`MeshAsset::vgeom_streams` drops the
    /// skin stream), so unlike a rigid mesh a character's ladder has to be
    /// stored rung by rung, and every stored rung is a whole `.inf_mesh` in the
    /// pack. Three is what the crowd tiers can actually select between --
    /// `CrowdTier::{Full, Near, Far}` -- so a fourth would be bytes nothing asks
    /// for.
    pub character_lods: usize,
    /// The `/Game/...` object path of the skeletal mesh whose rig every imported
    /// **clip** is retargeted onto. `None` uses the first skeleton the run
    /// imports, which is right when the manifest carries one body.
    pub retarget_to: Option<String>,
    /// **`(committed stem, manifest mesh key)` — write an imported RIGID mesh
    /// at the identity that stem derives** (wave WPN2d).
    ///
    /// [`rebind_character`]'s arrangement for a prop rather than a body, and for
    /// its reason exactly: the engine's weapon catalogue names its art by a
    /// GUID derived from a NAME (`inf_ecs::weapon::weapon_mesh_guid`), that
    /// catalogue is committed, and the meshes are licensed content that may
    /// never enter the repository. So the importer writes the imported mesh at
    /// the committed identity, into the local project only, and a checkout
    /// without the art resolves the identity to nothing and draws the
    /// placeholder.
    ///
    /// The stem is the ART KEY (`SM_AR4`), not a file name: the file is written
    /// as `<stem>.inf_mesh` in the project ROOT beside the starter body, which
    /// is where a rebind's products live and where the asset scan finds one
    /// asset rather than two claiming one id.
    pub rebind_meshes: Vec<(String, String)>,
    /// The manifest key of the skeletal mesh to write **at the starter
    /// character's committed GUIDs** — the REBIND, for a body.
    ///
    /// # Why a body needs the same door a road surface needed
    ///
    /// The island's hero entity names three fixed GUIDs (`0x5C10_00A0 + 0/1/2`
    /// — the starter rig, its skin material and its body mesh), the level that
    /// names them is committed and byte-locked, and the body a demo wants to
    /// show is licensed content that may never enter this repository. Exactly
    /// the arrangement clause 0 of ASSET0 solved for the road: write the
    /// imported asset **at the committed identity**, into the local project
    /// only. `samples/starter-character/Starter_Body.inf_mesh` (our own,
    /// committed, licence-free) and `Content/UE/.../Starter_Body.inf_mesh`
    /// (Unreal's mannequin, local, never committed) become the same asset
    /// identity with different vertices, and the island level does not know or
    /// care which one it got.
    ///
    /// Three assets move together or none do, because a mesh whose joint
    /// indices address one rig cannot be posed by another: the LOD-0 mesh, the
    /// skeleton it was skinned to, and the material in its first slot.
    pub rebind_character: Option<String>,
    /// The manifest key of a SECOND skeletal mesh to write at the **female**
    /// starter character's committed GUIDs (wave CHAR1a.3).
    ///
    /// # Why a second target rather than a second run
    ///
    /// The island's crowd wears whatever `(mesh, skeleton, machine)` triples the
    /// LEVEL's own entities carry (`inf_ecs::society::level_archetypes`), and the
    /// second committed body — `samples/starter-character-f`, whose GUIDs are
    /// `0x5C10_00B0 + n` — is the one the demo loop places to give the crowd a
    /// plural wardrobe. Rebinding it in the same run as the hero's is what makes
    /// the two MetaHumans a MALE and a FEMALE default rather than one body
    /// twice: two keys, two identities, one import.
    pub rebind_character_f: Option<String>,
    /// **Import only the meshes whose manifest key contains one of these**
    /// (wave OUTFIT1). Empty imports everything, which is every caller before it.
    ///
    /// # Why a manifest needs a narrower door than a pack filter
    ///
    /// One `export.py` run writes what the PACK holds, and the MetaHumans pack
    /// holds the bodies, the faces, the combined full-bodies, the outfits and
    /// the groom cards. A wave that wants the clothes does not want to re-import
    /// four bodies and 450 MB of skin textures on top of the ones the island
    /// already has — and carried item 109 says what a re-import costs when a
    /// material NAME differs: 18 `.inf_mesh` became 36 and 224 sidecars became
    /// 362 in one measured run.
    pub only: Vec<String>,
    /// **A `.inf_cloth` to put on every level's pawn** (wave OUTFIT1, carried
    /// item 137). `None` touches no level, which is every caller before it.
    pub wear_cloth: Option<AssetId>,
    /// **Wearables to write at the committed characters' clothes GUIDs** (wave
    /// OUTFIT1) — `rebind_character`'s rule for an outfit and a head of hair.
    pub wearables: Vec<WearableRebind>,
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
            character_lods: 3,
            retarget_to: None,
            rebind_meshes: Vec::new(),
            rebind_character: None,
            rebind_character_f: None,
            only: Vec::new(),
            wear_cloth: None,
            wearables: Vec::new(),
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
    /// `(manifest key, mesh asset, skeleton asset, rungs, triangles, joints)`
    /// per skeletal mesh imported (wave CHAR1a). The skeleton is `None` when the
    /// glTF carried a mesh with no skin, which is a defect worth seeing rather
    /// than a shape to tolerate.
    pub skeletal: Vec<(String, AssetId, Option<AssetId>, usize, usize, usize)>,
    /// `(manifest key, clip asset, tracks after retarget)` per clip imported.
    pub clips: Vec<(String, AssetId, usize)>,
    /// **The per-pack licence positions this import relied on**, carried out of
    /// the manifest so a caller can write them into its own ledger rather than
    /// re-deriving them: `(pack, licence text, may-ship)`.
    pub licences: Vec<(String, String, bool)>,
    /// Bytes of `.inf_tex` + `.inf_mat` + `.inf_mesh` written.
    pub bytes: u64,
    /// **Which pack each written asset came from** (wave CHAR1a.3, carried 96),
    /// so the licence position can be stamped onto the asset ON DISK rather than
    /// only printed.
    ///
    /// Collected as the import goes because that is the only place the answer is
    /// known: a `.inf_tex` is written inside `import_material` and has no key of
    /// its own, and a run that imports two packs cannot recover which is which
    /// from the finished report.
    pub asset_packs: Vec<(AssetId, String)>,
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
    /// The lamp's colour, **as UE stores it: 8-bit sRGB**.
    ///
    /// Not converted. The transfer function is a `powf`, and this crate's
    /// `portable_math_law` gate refuses one — correctly, because the engine
    /// already has exactly one sRGB decode (`inf_material::ground`'s sqrt
    /// ladder, transcendental-free so committed texels are byte-identical on
    /// every platform) and a second approximation here would be a second
    /// answer to one question. A bridge carries the source value; the
    /// conversion belongs where a `Light` is authored from it, next to
    /// whatever door that authoring path already uses.
    pub color_srgb8: [u8; 3],
    /// Range in metres (UE's attenuation radius).
    pub range_m: f32,
    /// Candela. See [`ue_intensity_to_candela`].
    pub intensity: f32,
}

/// **The engine checkout `path` sits inside, if it sits inside one** (ASSET0
/// audit).
///
/// A directory is this engine's checkout when it holds **both** a `.git` and
/// `tools/ue-export/export.py` — the second marker on purpose, because it is
/// the very script that produces the bytes this law is about, and a user's own
/// game repository must not be mistaken for ours.
///
/// # Why a mechanism and not a sentence
///
/// The wave's licence law — *nothing derived from the reference project's
/// marketplace, Fab or Megascans content may enter this repository* — was
/// upheld by three sentences of documentation and the author's care.
/// Measured at the audit: `export.py` takes its output directory from
/// `INF_UE_OUT` and `inf-import` takes its destination from `--into`, and
/// **neither refused a path inside the checkout**. A single mistyped
/// destination puts 4.1 GB of Megascans PNGs, or 892 MB of `.inf_tex`
/// converted from them, in the working tree of a PUBLIC repository, untracked
/// and one `git add -A` from being published. A law with no door that can say
/// no is a preference.
pub fn engine_checkout_above(path: &Path) -> Option<PathBuf> {
    let mut p: PathBuf = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    loop {
        if p.join(".git").exists() && p.join("tools/ue-export/export.py").is_file() {
            return Some(p);
        }
        if !p.pop() {
            return None;
        }
    }
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
    /// v2 (wave CHAR1a): skeletal meshes, each with its own LOD ladder.
    skeletal_meshes: Vec<SkeletalMesh>,
    /// v2 (wave CHAR1a): animation clips, exported one glTF each.
    clips: Vec<Clip>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Pack {
    name: String,
    license: String,
    /// v2: whether this pack's licence permits SHIPPING the content, as opposed
    /// to using it as a local reference. Recorded per pack because the three
    /// character packs differ: ALS is MIT (ship), the mannequins are Epic's
    /// UE-Only content (reference), MetaHumans ship in a cooked pack and are
    /// never committed.
    ship: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct SkeletalMesh {
    key: String,
    pack: String,
    source: String,
    skeleton: Option<String>,
    bones: u32,
    lods: Vec<SkelLod>,
    material_slots: Vec<Option<String>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct SkelLod {
    level: u32,
    file: Option<String>,
    /// Joints in the exported skin — read off the written glTF, not asserted.
    joints: u32,
    joint_names: Vec<String>,
    /// How many `JOINTS_n` attribute sets the exporter wrote. UE writes **two**
    /// (eight influences a vertex) and `.inf_mesh`'s `VertexSkin` holds four.
    influence_sets: u32,
    /// Triangles in the written glTF, counted off its accessors by the exporter.
    /// The number a ladder is chosen by — see [`distinct_rungs`].
    triangles: u32,
    primitives: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct Clip {
    key: String,
    pack: String,
    name: String,
    source: String,
    file: Option<String>,
    skeleton: Option<String>,
    skeleton_bones: u32,
    seconds: f32,
    joints: u32,
    joint_names: Vec<String>,
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
/// **v2 (wave CHAR1a)**: `skeletal_meshes` and `clips` joined the manifest.
///
/// The bump is not cosmetic. Every field on every manifest record is
/// `#[serde(default)]` — the deliberate choice at ASSET0, so a manifest that
/// grows a field does not fail an import that never reads it — and the exact
/// cost of that choice is that a v1 reader handed a v2 manifest imports the
/// meshes, ignores the two new **sections** entirely, and reports success
/// having imported no character at all. So the version gates the container and
/// a reader that grows an arm keys it on the version, which is the same law
/// `.ipack`'s header carries.
pub const MANIFEST_SCHEMA_VERSION: u32 = 2;

/// **Import one manifest.**
pub fn import_manifest(
    project: &mut AssetProject,
    manifest_path: &Path,
    opts: &UeImportOptions,
) -> Result<UeImportReport> {
    // **THE LICENCE LAW, AS A DOOR.** Before a single byte is decoded: this
    // bridge's output is licensed content, and the one place it may never land
    // is the public engine repository. See [`engine_checkout_above`].
    if let Some(root) = engine_checkout_above(project.root()) {
        return Err(AssetError::Import(format!(
            "refusing to import into {} — it is inside the engine checkout at \
             {}, and NOTHING this bridge writes may enter this repository. The \
             reference project's packs are Marketplace/Fab/Megascans content \
             whose licence for use outside Unreal is unestablished (see the \
             ASSET0 licence table in docs/memos/island-progress.md). Point the \
             destination at a project outside the checkout — the island's is \
             ../island-build/project.",
            project.root().display(),
            root.display()
        )));
    }
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
    // **The narrower door** (wave OUTFIT1, `--only`): a manifest holds what its
    // PACK holds, and a wave that wants the clothes does not want the four
    // bodies beside them re-imported. Empty is every caller before this option.
    let only = |key: &str| opts.only.is_empty() || opts.only.iter().any(|k| key.contains(k));
    let mut report = UeImportReport::default();
    for p in &m.packs {
        if wanted(&p.name) {
            report.advisories.push(format!(
                "pack {}: licence {} [{}]",
                p.name,
                p.license,
                if p.ship {
                    "MAY SHIP"
                } else {
                    "LOCAL REFERENCE ONLY - never cook, never commit"
                }
            ));
            report
                .licences
                .push((p.name.clone(), p.license.clone(), p.ship));
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
        let before = report.textures.len();
        let id = import_material(project, &base, &dest, mat, &by_key, opts, stem, &mut report)?;
        mat_ids.insert(mat.key.clone(), id);
        report.materials.push((mat.key.clone(), id));
        // The material and every texture IT wrote belong to its pack — the only
        // place a `.inf_tex`'s provenance is known, since a texture record has no
        // pack of its own.
        report.asset_packs.push((id, mat.pack.clone()));
        let fresh: Vec<AssetId> = report.textures[before..].to_vec();
        for t in fresh {
            report.asset_packs.push((t, mat.pack.clone()));
        }
        if let Some(stem) = stem {
            report.rebinds.push((stem.to_string(), id));
        }
    }

    // ── 2. meshes, through the one importer door ─────────────────────────────
    if opts.meshes {
        for mesh in &m.meshes {
            if !wanted(&mesh.pack) || !only(&mesh.key) {
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
            // Everything the one importer door produced belongs to this pack —
            // the mesh, and the skeleton, materials and textures a glTF carries
            // inside it. Collected here because `import_file` is where the set is
            // known, and because a licence stamped onto the mesh and not onto the
            // rig it needs is a licence somebody will read half of.
            let produced = out.produced.clone();
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
            for a in dependency_closure(project, &[&[id][..], &produced].concat()) {
                report.asset_packs.push((a, mesh.pack.clone()));
            }
            report
                .meshes
                .push((mesh.key.clone(), id, mesh.lods.len(), tris));
            // **THE RIGID REBIND** (wave WPN2d) — write this mesh a second time
            // at the identity a committed name derives, so a level that names
            // that identity draws licensed art it never references by path.
            // `rebind_character`'s arrangement, for a prop.
            for (stem, key) in &opts.rebind_meshes {
                if key != &mesh.key {
                    continue;
                }
                let want = AssetId(inf_ecs::weapon::weapon_mesh_guid(stem));
                let payload: inf_mesh::MeshAsset = project.load_payload(id)?;
                // The materials the mesh's own slot table names — the same edge
                // set `rebind_character` builds, so a cook that packs the
                // committed identity packs its skins with it.
                let deps: Vec<AssetId> = {
                    let mut out: Vec<AssetId> = Vec::new();
                    for m in payload.material_slot_assets.iter().flatten() {
                        if !out.contains(m) {
                            out.push(*m);
                        }
                    }
                    out
                };
                let path = project.root().join(format!("{stem}.inf_mesh"));
                project.write_asset_at_with_id(&path, &payload, want, deps, None)?;
                report.rebinds.push((format!("{stem}.inf_mesh"), want));
                report.asset_packs.push((want, mesh.pack.clone()));
                report.advisories.push(format!(
                    "rebind: {} -> {stem}.inf_mesh at {}",
                    mesh.key,
                    want.0
                ));
            }
        }
    }

    // -- 2b. SKELETAL meshes, through the same one door (wave CHAR1a) --------
    //
    // Same `import_file` the rigid meshes use -- it has parsed glTF `skins`,
    // `inverseBindMatrices`, `JOINTS_0`/`WEIGHTS_0` and `animations` since
    // P11.1, and writes the `.inf_skel` plus the dependency edge from the mesh
    // onto it. What this loop adds is the LADDER: unlike a rigid mesh (whose
    // coarser rungs are RECORDED and thrown away, because every rigid draw goes
    // through a continuous meshlet cut) a **skinned** mesh never reaches the
    // vgeom path at all -- `MeshAsset::vgeom_streams` drops the skin stream --
    // so a character that wants a LOD ladder has to have one stored, rung by
    // rung.
    let mut skeletons_by_source: BTreeMap<String, (AssetId, inf_anim::SkeletonAsset)> =
        BTreeMap::new();
    if opts.meshes {
        for sk in &m.skeletal_meshes {
            if !wanted(&sk.pack) || !only(&sk.key) {
                continue;
            }
            let mut rungs: Vec<(u32, AssetId, usize)> = Vec::new();
            let mut skel_id: Option<AssetId> = None;
            for lod in distinct_rungs(&sk.lods, opts.character_lods) {
                let Some(file) = lod.file.as_ref() else {
                    report
                        .advisories
                        .push(format!("{}: LOD {} exported no file", sk.key, lod.level));
                    continue;
                };
                let src = base.join(file);
                if !src.is_file() {
                    report
                        .advisories
                        .push(format!("{}: {} is not on disk", sk.key, src.display()));
                    continue;
                }
                let out = super::import::import_file(project, &src, &dest)?;
                // …and the same for a body: its rig, its baked materials and
                // their textures all cross under the pack's licence.
                let produced = out.produced.clone();
                report.advisories.extend(out.advisories);
                let Some(id) = out.primary else {
                    report
                        .advisories
                        .push(format!("{}: LOD {} produced no mesh", sk.key, lod.level));
                    continue;
                };
                // The skeleton is whichever product decoded as one. Taken from
                // LOD 0 only: every rung of one character shares one rig, and
                // importing rung 1's copy would give the ladder two skeletons
                // whose joint ORDER agrees only by luck.
                if lod.level == 0 {
                    // The products of THIS call, plus the mesh's own dependency
                    // edges. The second half is not belt-and-braces: on a
                    // re-import the content-hash dedupe reuses the assets that
                    // are already there and `produced` can be empty, so a body
                    // imported twice would look to this loop like a body with no
                    // rig -- which `rebind_character` then refuses, correctly and
                    // uselessly. The dependency edge is written by `import_file`
                    // when the mesh is skinned and survives the dedupe.
                    let mut candidates: Vec<AssetId> = out.produced.clone();
                    candidates.extend(
                        project
                            .db()
                            .get(id)
                            .map(|e| e.sidecar.dependencies.clone())
                            .unwrap_or_default(),
                    );
                    for pid in &candidates {
                        if project.db().get(*pid).map(|e| e.kind())
                            == Some(inf_asset::AssetKind::Skeleton)
                        {
                            if let Ok(asset) = project.load_payload::<inf_anim::SkeletonAsset>(*pid)
                            {
                                skeletons_by_source.insert(sk.source.clone(), (*pid, asset));
                            }
                            skel_id = Some(*pid);
                            break;
                        }
                    }
                }
                let tris = project
                    .load_payload::<inf_mesh::MeshAsset>(id)
                    .map(|mesh| mesh.triangle_count())
                    .unwrap_or(0);
                for a in dependency_closure(project, &[&[id][..], &produced].concat()) {
                    report.asset_packs.push((a, sk.pack.clone()));
                }
                // **THE SLOT TABLE** (wave CHAR1a.3, `.inf_mesh` v3). The
                // manifest states this mesh's material slots as manifest KEYS and
                // section 1 has already imported each of them, so this is the one
                // place in the tree where "slot 3 is that asset" is known. Written
                // into the payload rather than the sidecar because neither a
                // cooked `.ipack` nor a PIE `ScenePayload` carries a sidecar, and
                // a face whose eye slots resolved in the editor and not in the
                // game is the divergence PIE == shipping exists to stop.
                split_udim_sections(project, id, sk, &mut report);
                bind_slots(project, id, sk, &mat_ids, &mut report);
                rungs.push((lod.level, id, tris));
            }
            let Some((_, lod0, tris0)) = rungs.first().copied() else {
                report
                    .advisories
                    .push(format!("{}: no rung imported", sk.key));
                continue;
            };
            // **The influence-set notice.** UE writes eight influences a vertex
            // (`JOINTS_0` + `JOINTS_1`); this engine's `VertexSkin` holds four.
            // Said once per mesh with the number, because a silently halved
            // weight set is a body whose shoulders crease and nobody knows why.
            if let Some(l0) = sk.lods.first() {
                if l0.influence_sets > 1 {
                    report.advisories.push(format!(
                        "{}: the export carries {} influence sets ({} influences a \
                         vertex); this engine's VertexSkin holds 4, so the import \
                         keeps the four heaviest per vertex and renormalizes",
                        sk.key,
                        l0.influence_sets,
                        l0.influence_sets * 4
                    ));
                }
            }
            record_character_ladder(project, lod0, sk, &rungs, &mut report);
            let joints = skel_id
                .and_then(|id| project.load_payload::<inf_anim::SkeletonAsset>(id).ok())
                .map(|s| s.skeleton.len())
                .unwrap_or(0);
            if opts.rebind_character.as_deref() == Some(sk.key.as_str()) {
                rebind_character(
                    project,
                    lod0,
                    skel_id,
                    sk,
                    &mat_ids,
                    &crate::samples::starter_character_ids(),
                    ("Starter", "Starter_Body", "Starter_Skin"),
                    ("Starter_Idle", "Starter_Walk", "Starter_Run"),
                    &mut report,
                )?;
            }
            if opts.rebind_character_f.as_deref() == Some(sk.key.as_str()) {
                rebind_character(
                    project,
                    lod0,
                    skel_id,
                    sk,
                    &mat_ids,
                    &crate::samples::starter_character_f_ids(),
                    ("Starter_F", "Starter_F_Body", "Starter_F_Skin"),
                    ("Starter_F_Idle", "Starter_F_Walk", "Starter_F_Run"),
                    &mut report,
                )?;
            }
            report
                .skeletal
                .push((sk.key.clone(), lod0, skel_id, rungs.len(), tris0, joints));
        }
        // ── THE WEARABLES (wave OUTFIT1) ───────────────────────────────────
        //
        // After both mesh loops and after `rebind_character`, because a garment
        // is re-pointed onto the rig at `CharacterIds::skeleton` and that rig is
        // whatever the BODY rebind in this same run just wrote there. Running
        // this first would re-point a MetaHuman's clothes onto the low-poly
        // starter rig the level had a moment ago.
        //
        // A garment is looked up in `report.skeletal` (it is a skeletal mesh
        // with its own rig) and a groom's cards in `report.meshes` (a static
        // mesh with none), which is exactly the difference `rebind_wearable`
        // then acts on.
        for spec in &opts.wearables {
            let ids = if spec.female {
                crate::samples::starter_character_f_ids()
            } else {
                crate::samples::starter_character_ids()
            };
            // **THE COMMITTED FILE NAMES, both of them** — a rebind writes at a
            // committed GUID *and* at that GUID's committed FILE NAME, or the
            // asset scan finds TWO files claiming one id. Measured: the first
            // cut wrote `Starter_Outfit.inf_mat` beside the committed
            // `Starter_Outfit_Top.inf_mat`, both carrying `…a9`.
            let prefix = if spec.female { "Starter_F" } else { "Starter" };
            let stem = if spec.hair {
                format!("{prefix}_Hair_Mesh")
            } else {
                format!("{prefix}_Outfit")
            };
            let mat_stem = if spec.hair {
                format!("{prefix}_Hair")
            } else {
                format!("{prefix}_Outfit_Top")
            };
            // The PACK the mesh came from, so the licence follows the bytes onto
            // the committed GUID the level references.
            let pack = m
                .skeletal_meshes
                .iter()
                .find(|s| s.key.contains(&spec.key))
                .map(|s| s.pack.clone())
                .or_else(|| {
                    m.meshes
                        .iter()
                        .find(|s| s.key.contains(&spec.key))
                        .map(|s| s.pack.clone())
                })
                .unwrap_or_default();
            let skinned = report
                .skeletal
                .iter()
                .find(|(k, ..)| k.contains(&spec.key))
                .map(|(k, mesh, skel, ..)| (k.clone(), *mesh, *skel));
            let rigid = report
                .meshes
                .iter()
                .find(|(k, ..)| k.contains(&spec.key))
                .map(|(k, mesh, ..)| (k.clone(), *mesh, None));
            // **THE MATERIAL THE MANIFEST NAMES FOR THIS MESH** (wave OUTFIT1
            // AUDIT, carried item 164). A groom's cards mesh carries
            // `WorldGridMaterial` — Unreal's checkerboard — in its own static
            // slot, because in Unreal the GROOM COMPONENT applies the hair
            // material; the exporter already substitutes the real one into the
            // MANIFEST's slot list, and nothing on this side was reading it for
            // a mesh with one slot. Measured on the island: the committed
            // `Starter_Hair.inf_mat` was the glTF's own WHITE OPAQUE
            // `WorldGridMaterial`, which is why both characters' hair drew pale
            // — not the melanin brown the manifest states, and with no coverage
            // and no masked blend either.
            let manifest_mat = m
                .meshes
                .iter()
                .find(|s| s.key.contains(&spec.key))
                .map(|s| s.material_slots.clone())
                .or_else(|| {
                    m.skeletal_meshes
                        .iter()
                        .find(|s| s.key.contains(&spec.key))
                        .map(|s| s.material_slots.clone())
                })
                .filter(|slots| slots.len() == 1)
                .and_then(|slots| slots.into_iter().next().flatten())
                .and_then(|k| mat_ids.get(&k).copied());
            let Some((key, mesh, skel)) = skinned.or(rigid) else {
                report.advisories.push(format!(
                    "--wearable {}: no imported mesh's key contains it, so \
                     nothing was worn (did `--only` exclude it?)",
                    spec.key
                ));
                continue;
            };
            rebind_wearable(
                project,
                mesh,
                skel,
                manifest_mat,
                &ids,
                spec,
                &stem,
                &mat_stem,
                &key,
                &pack,
                &mut report,
            )?;
        }
    }
    // …and the garment a LEVEL's pawn wears (carried item 137), because a level
    // edit is re-applied by the same command sequence that rebuilt the project.
    if let Some(cloth) = opts.wear_cloth {
        let n = wear_cloth_in_levels(project, cloth, &mut report);
        report
            .advisories
            .push(format!("{n} level(s) dressed with the garment {cloth}"));
    }

    // -- 2c. CLIPS, retargeted onto the rig they will be played on ------------
    //
    // NOT through `import_file`: a clip glTF carries its own copy of the source
    // skin, and one hundred and twenty-six ALS clips would have written one
    // hundred and twenty-six identical `.inf_skel` assets that nothing plays and
    // whose joint indices differ from the body's. So the glTF is decoded here,
    // its clip is retargeted BY NAME onto the skeleton the body imported, and
    // one `.inf_anim` is written with a dependency edge onto that skeleton.
    for c in &m.clips {
        if !wanted(&c.pack) || !only(&c.key) {
            continue;
        }
        let Some(file) = c.file.as_ref() else {
            report
                .advisories
                .push(format!("{}: clip exported no file", c.key));
            continue;
        };
        let src = base.join(file);
        if !src.is_file() {
            report
                .advisories
                .push(format!("{}: {} is not on disk", c.key, src.display()));
            continue;
        }
        let g = match inf_mesh::import_gltf(&src) {
            Ok(g) => g,
            Err(e) => {
                report.advisories.push(format!("{}: {e}", c.key));
                continue;
            }
        };
        let (Some(imported), Some(src_skel)) = (g.clips.first(), g.skeletons.first()) else {
            report.advisories.push(format!(
                "{}: the glTF carries {} clips and {} skins -- a clip needs one of each",
                c.key,
                g.clips.len(),
                g.skeletons.len()
            ));
            continue;
        };
        // The rig to play it on: the one this manifest imported for this pack's
        // body, else the first skeleton imported at all. Named rather than
        // guessed, so a manifest that exported clips and no body says so.
        let Some((target_id, target)) = opts
            .retarget_to
            .as_ref()
            .and_then(|s| skeletons_by_source.get(s))
            .or_else(|| skeletons_by_source.values().next())
            .cloned()
        else {
            report.advisories.push(format!(
                "{}: no skeleton was imported to retarget onto -- export a skeletal \
                 mesh in the same manifest, or name one with --retarget-to",
                c.key
            ));
            continue;
        };
        let map =
            inf_anim::retarget::RetargetMap::shared_names(&src_skel.skeleton, &target.skeleton);
        let (payload, rep) = inf_anim::retarget::retarget_clip(
            &imported.clip,
            &src_skel.skeleton,
            &target.skeleton,
            &map,
            true,
        );
        if rep.is_vacuous() {
            // The silent failure, named. A clip with no tracks plays as a
            // perfect bind pose, which on this rig is a T.
            report.advisories.push(format!(
                "{}: retarget produced NO tracks ({} source joints, none named on \
                 the target rig) -- the clip would play as a bind pose",
                c.key,
                src_skel.skeleton.len()
            ));
            continue;
        }
        // **DERIVED BEFORE IT IS WRITTEN** (wave CHAR1b.1), the same rule the
        // glTF clip stage has followed since P29.5 and the one door this one did
        // not go through. Measured before the change: of the 164 clips this
        // manifest imports, **zero** carried `MoveData_Speed`, `W_Gait`,
        // `FootSpeed_*`, `FootLock_*` or `Enable_FootIK_*` — so the island's
        // hero ran ALS's whole foot-IK and foot-lock mechanism on channels that
        // were not there, which reads at every consumer as "this clip wants no
        // foot IK" and is why the feet had never once been put on the ground.
        //
        // Derived against the TARGET rig, because the payload has just been
        // retargeted onto it: a foot index is an index into the pose, and a
        // derivation measured against the donor would name the wrong joints.
        // A refusal costs this clip its channels and nothing else.
        let mut payload = payload;
        let target_rig = inf_anim::SkeletonAsset::new(target.skeleton.clone());
        let derived = super::anim_derive::derive_in_place(
            &c.key,
            &mut payload,
            Some(&target_rig),
            &inf_anim::DeriveOptions::default(),
        );
        report
            .advisories
            .extend(super::anim_derive::advisories(&c.key, &derived));
        let asset = inf_anim::AnimClipAsset::new(payload, Some(*target_id.uuid().as_bytes()));
        let name = format!("{}_{}", c.pack, c.name);
        let import = super::skeleton_binding::import_table(project, Some(target_id));
        // **IDENTITY-IDEMPOTENT** (carried item 94). `write_asset` allocates a
        // fresh GUID and side-steps a name collision by writing `X_1.inf_anim`;
        // four import runs of one manifest therefore left 656 `.inf_anim` in the
        // island project where 164 belong. The id is now a pure function of the
        // manifest key and the path a pure function of the name, so a re-import
        // overwrites its own output — the same rule the texture writer has
        // followed since ASSET0, applied to the one kind that did not.
        let id = project.write_asset_at_with_id(
            &dest.join(format!("{name}.inf_anim")),
            &asset,
            clip_guid(&c.key),
            vec![target_id],
            import,
        )?;
        // The source note the allocating door used to write.
        if let Some(entry) = project.db().get(id) {
            let path = entry.path.clone();
            if let Ok(mut side) = inf_asset::AssetSidecar::load(&path) {
                side.source = Some(c.source.clone());
                let _ = side.save(&path);
            }
        }
        report.asset_packs.push((id, c.pack.clone()));
        if !rep.dropped.is_empty() {
            report
                .advisories
                .push(format!("{}: {}", c.key, rep.summary()));
        }
        report.clips.push((c.key.clone(), id, rep.tracks_out));
    }

    // -- 2d. the REBOUND character's clips --------------------------------
    //
    // **Found by looking at the picture.** Rebinding the body and its rig alone
    // produced a hero standing in the street with one arm over its head and its
    // legs splayed, which is what a valid clip played on the wrong rig looks
    // like: `manny.rs` generates a rig whose every bind rotation is the
    // IDENTITY (deliberately -- it is what lets the inverse bind be a
    // translation and keeps the P14 no-trig law), and the shipped mannequin's
    // bind carries a real rotation on 137 of its 162 nodes. The committed
    // `Starter_Idle/Walk/Run` write absolute local rotations computed against
    // the first, and `sample_clip` seeds every untouched joint from the
    // second's `local_bind`. The two disagree everywhere.
    //
    // So a body rebind is not three assets, it is SIX: mesh, rig, skin, and the
    // three clips the hero's state machine names. The clips come from the same
    // pack as the body, so they were authored on exactly the rig that is now
    // underneath it.
    if opts.rebind_character.is_some() && !report.clips.is_empty() {
        rebind_character_clips(
            project,
            &m.clips,
            &report.clips.clone(),
            &crate::samples::starter_character_ids(),
            ("Starter_Idle", "Starter_Walk", "Starter_Run"),
            false,
            &mut report,
        )?;
    }
    if opts.rebind_character_f.is_some() && !report.clips.is_empty() {
        rebind_character_clips(
            project,
            &m.clips,
            &report.clips.clone(),
            &crate::samples::starter_character_f_ids(),
            ("Starter_F_Idle", "Starter_F_Walk", "Starter_F_Run"),
            true,
            &mut report,
        )?;
    }

    // -- 2e. the REBOUND character's GRAPH (wave CHAR1b.1) -----------------
    //
    // Step 2d rebinds the three clips the committed three-state machine names.
    // That machine is `idle` / `walk` / `run` on one `speed` parameter, which is
    // what the P24 wizard generates for a rig with no content — and the island's
    // hero has 164 authored ALS sequences sitting beside it. Eleven of the
    // fourteen catalogue modes had no animation at all.
    //
    // So the graph is rebuilt from `inf_anim::als::LOCOMOTION_MAP` over whatever
    // this import brought in, and written at the SAME machine GUID the level
    // already references — so nothing in the `.inf_lvl` moves and the hero picks
    // up a twenty-state graph on its next load.
    //
    // Local-only, exactly like the body rebind: the map is a table of names (MIT
    // ALS is cited, never copied) and the clips it resolves are this machine's
    // own import.
    for (want, stem) in [
        (
            opts.rebind_character.is_some(),
            (
                crate::samples::starter_character_ids(),
                "Starter_Locomotion",
            ),
        ),
        (
            opts.rebind_character_f.is_some(),
            (
                crate::samples::starter_character_f_ids(),
                "Starter_F_Locomotion",
            ),
        ),
    ] {
        if want {
            rebind_locomotion_graph(
                project,
                &m.clips,
                &report.clips.clone(),
                &stem.0,
                stem.1,
                &mut report,
            );
        }
    }

    // ── 3. fixtures ──────────────────────────────────────────────────────────
    for f in &m.fixtures {
        for l in &f.lights {
            report.fixtures.push(UeFixture {
                name: f.key.clone(),
                mesh: f.meshes.first().map(|fm| fm.mesh.clone()),
                offset_m: ue_cm_to_world_m(l.location_cm),
                color_srgb8: l.color_srgb8,
                range_m: l.radius_cm / 100.0,
                intensity: ue_intensity_to_candela(l.intensity),
            });
        }
    }

    // **WHAT THE RUNG CENSUS ACTUALLY CAPTURED** (ASSET0 audit). The LOD ruling
    // — store no authored ladder, record the pack's rungs in the sidecar for the
    // wave that seeds the meshlet DAG from them — rests on that census being
    // worth inheriting. Measured on this project: `screen_size` is UE's
    // auto-compute sentinel `-1` for **18 of 18 rungs across nine packs**, so
    // what the sidecars hold is rung counts and material slots, and no
    // thresholds at all. Said once per import, with the count, rather than left
    // as a column of `-1.0` for whoever tries to use it.
    let (auto, laddered) = m
        .meshes
        .iter()
        .filter(|x| wanted(&x.pack) && x.lods.len() > 1)
        .fold((0usize, 0usize), |(a, n), x| {
            let all_auto = x.lods.iter().all(|l| l.screen_size < 0.0);
            (a + usize::from(all_auto), n + 1)
        });
    if auto > 0 {
        report.advisories.push(format!(
            "{auto} of {laddered} multi-rung meshes state no LOD screen sizes — \
             UE reports its auto-compute sentinel (-1), so the sidecar census is \
             rung counts and material slots with no thresholds in it"
        ));
    }

    // **THE ROLE TABLES, ON DISK** (audit CHAR1b.1, carried item 119).
    let rederived = sweep_roles(project, &mut report);
    if rederived > 0 {
        report.advisories.push(format!(
            "{rederived} rig(s) in this project carried no role table and one was inferred from their own bone names - see `sweep_roles`"
        ));
    }

    // **THE LICENCE, ON DISK** (carried 96). Last, because it stamps everything
    // the run produced and the run is now over.
    let mut stamped = stamp_licences(project, &mut report);
    let (swept, exact) = sweep_licences(project, &dest, &mut report);
    stamped += swept;
    if swept > 0 {
        report.advisories.push(format!(
            "{swept} asset(s) in {} were produced by the import and are the \
             dependency of nothing, so their licence is {} - see \
             `sweep_licences`",
            dest.display(),
            if exact {
                "this run's single pack"
            } else {
                "every pack this run imported, at the most conservative ship position"
            }
        ));
    }
    if stamped > 0 {
        report.advisories.push(format!(
            "licence position written into {stamped} asset sidecar(s) on disk"
        ));
    } else if !report.licences.is_empty() {
        report.advisories.push(
            "NO asset sidecar carries a licence position — the packs' licences \
             exist only in this report, which is carried item 96"
                .to_string(),
        );
    }
    report.bytes = written_bytes(project, &report);
    Ok(report)
}

/// **A deterministic asset GUID for one manifest key** (wave CHAR1a.3, carried
/// item 94).
///
/// # The defect
///
/// The clip importer wrote every `.inf_anim` through `write_asset`, which
/// ALLOCATES a fresh GUID and, on a name collision, writes `X_1.inf_anim` beside
/// the one already there. Measured at the CHAR1a audit: the island project held
/// **656** `.inf_anim` from four import runs, 656 distinct GUIDs, and 134 of the
/// 164 sources with two byte-different payloads on disk. The importer was
/// content-deterministic and not **identity**-idempotent, and the difference is
/// the difference between a re-import that updates a project and one that grows
/// it.
///
/// # The rule
///
/// The GUID is a pure function of the manifest key — the UE object path — so a
/// re-import of the same source overwrites its own output, exactly as the texture
/// writer's deterministic PATH already does. FNV-1a over the key, spread across
/// sixteen bytes, written out rather than pulled from a hasher crate for the same
/// reason `short_name`'s digest is: this is an IDENTITY, and an identity whose
/// bytes depend on a dependency's default hasher is an identity that can move
/// under a `cargo update`.
///
/// The high nibble is forced to a UUID v4 shape so the value round-trips through
/// every reader that pretty-prints one, and the salt keeps a clip's id away from
/// any other derived id of the same key.
/// **Rebuild a committed identity's locomotion graph from the clips a project
/// already holds** (wave CHAR1b.2) — the door the import path had only as a
/// side effect of `--rebind-character`.
///
/// `rebind_locomotion_graph` runs inside [`import_manifest`] and only when a
/// body rebind was asked for, which couples two decisions that are not the same
/// one: *which mesh and rig this identity wears* and *which clips its graph
/// names*. The island's hero wears a MetaHuman body rebound from one manifest
/// and plays ALS sequences imported from another, so rebuilding its graph after
/// the map grows a row meant re-running the body rebind — and re-running the
/// wrong manifest's body rebind swaps the character.
///
/// This is that half on its own: no import, no mesh, no textures. The resolver
/// falls back to the project's own `.inf_anim` set exactly as the import path's
/// does, so every clip the map names is found wherever it was written, and the
/// authored sets are derived onto the identity's own rig here.
///
/// `stems` is `(ids, stem)` per identity — normally
/// `samples::starter_character_ids()` with `"Starter_Locomotion"` and the female
/// pair. Returns the report so a caller can print the advisories.
pub fn rebind_graphs(
    project: &mut AssetProject,
    stems: &[(crate::character::CharacterIds, &str)],
) -> UeImportReport {
    let mut report = UeImportReport::default();
    let manifest: Vec<Clip> = Vec::new();
    let imported: Vec<(String, AssetId, usize)> = Vec::new();
    for (ids, stem) in stems {
        rebind_locomotion_graph(project, &manifest, &imported, ids, stem, &mut report);
    }
    report
}

pub fn clip_guid(key: &str) -> AssetId {
    let mut bytes = [0u8; 16];
    for (i, chunk) in bytes.chunks_mut(8).enumerate() {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // The salt, so `clip_guid("X")` and any future `something_guid("X")`
        // cannot collide on one key.
        for b in b"inf_anim:ue-clip:".iter().chain(key.as_bytes()) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        chunk.copy_from_slice(&h.to_le_bytes());
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    AssetId(uuid::Uuid::from_bytes(bytes))
}

/// **Every asset reachable from `roots`**, dependencies included, to a depth of
/// three.
///
/// The licence stamp needs the CLOSURE and not the products of one call: on a
/// re-import the content-hash dedupe reuses the assets that are already there and
/// `ImportOutput::produced` comes back EMPTY — the same fact `rebind_character`'s
/// skeleton search already had to work around — so a stamp keyed on `produced`
/// gets every asset on the first run and none on the second, which is the run
/// somebody checks.
///
/// Three is the depth this content actually has: mesh → skeleton, mesh →
/// material → texture. Bounded rather than transitive-until-fixpoint because a
/// dependency graph read off disk is somebody else's bytes.
fn dependency_closure(project: &AssetProject, roots: &[AssetId]) -> Vec<AssetId> {
    let mut seen: std::collections::BTreeSet<AssetId> = roots.iter().copied().collect();
    let mut frontier: Vec<AssetId> = roots.to_vec();
    for _ in 0..3 {
        let mut next: Vec<AssetId> = Vec::new();
        for id in frontier.drain(..) {
            let Some(e) = project.db().get(id) else {
                continue;
            };
            for d in &e.sidecar.dependencies {
                if seen.insert(*d) {
                    next.push(*d);
                }
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    seen.into_iter().collect()
}

/// The sidecar `import` keys the licence position is written to.
///
/// Named once, here, because a gate reads them back: **carried item 96** was
/// that the MetaHuman licence row existed in the export manifest and in the
/// printed import report and **nowhere on disk** — a grep for `licen` over all
/// 272 imported sidecars returned nothing. A licence that travels only in a
/// console line is a licence nobody can find six months later, which for content
/// that MAY SHIP and MAY NOT BE COMMITTED is the one fact that has to be
/// attached to the bytes.
pub const LICENCE_KEY: &str = "licence";
/// Whether the pack's licence permits SHIPPING this asset — see [`LICENCE_KEY`].
pub const LICENCE_SHIP_KEY: &str = "licence_may_ship";
/// Which pack the asset came from — see [`LICENCE_KEY`].
pub const LICENCE_PACK_KEY: &str = "licence_pack";

/// The three keys a licence row is written as, together — so a reader that has
/// to CARRY one (a derivation, a rebind) copies all three rather than the one it
/// happened to remember.
pub const LICENCE_KEYS: [&str; 3] = [LICENCE_KEY, LICENCE_SHIP_KEY, LICENCE_PACK_KEY];

/// **Write the pack's licence position into every asset this run produced.**
///
/// Sidecar-only: no payload moves, no schema window, and the text is where a
/// human looking at the asset will look. Returns how many sidecars were stamped,
/// which is the number the report prints and the gate asserts non-zero.
fn stamp_licences(project: &mut AssetProject, report: &mut UeImportReport) -> usize {
    let by_pack: BTreeMap<&str, (&str, bool)> = report
        .licences
        .iter()
        .map(|(p, l, s)| (p.as_str(), (l.as_str(), *s)))
        .collect();
    let mut stamped = 0usize;
    let pairs = report.asset_packs.clone();
    let mut failures: Vec<String> = Vec::new();
    for (id, pack) in pairs {
        let Some((licence, ship)) = by_pack.get(pack.as_str()).copied() else {
            continue;
        };
        let Some(entry) = project.db().get(id) else {
            continue;
        };
        let path = entry.path.clone();
        let Ok(mut side) = inf_asset::AssetSidecar::load(&path) else {
            continue;
        };
        let mut t = side.import.take().unwrap_or_default();
        t.insert(LICENCE_KEY.into(), licence.to_string().into());
        t.insert(LICENCE_SHIP_KEY.into(), ship.into());
        t.insert(LICENCE_PACK_KEY.into(), pack.clone().into());
        side.import = Some(t);
        match side.save(&path) {
            Ok(()) => stamped += 1,
            Err(e) => failures.push(format!("{}: {e}", path.display())),
        }
    }
    for f in failures {
        report
            .advisories
            .push(format!("licence not recorded on disk for {f}"));
    }
    stamped
}

/// **The sweep**: anything in the destination folder this run wrote to that the
/// per-asset pass did not reach.
///
/// # Why a sweep is needed at all
///
/// `import_file` produces more than it reports. A glTF carries its own materials,
/// and a body's four LOD rungs each carry a copy of them, so a `Content/UE/...`
/// folder ends up holding materials that are the product of an import and the
/// dependency of nothing — measured: 345 of 441 sidecars were reached by the
/// closure and **96** were not. An asset with no licence beside it is the state
/// carried item 96 is about, and "most of them have one" is not the claim.
///
/// A run that imported ONE pack attributes them exactly. A run that imported
/// several cannot — nothing on the asset says which — so the row names every
/// candidate and the ship position is the CONSERVATIVE one: local-only if any of
/// the packs is local-only, because the cost of getting that wrong in the shipping
/// direction is a licence breach and in the other direction is a missing texture.
/// **Give every rig in the project a role table** (audit CHAR1b.1, carried item
/// 119).
///
/// `SkeletonAsset::imported` infers one at the glTF stage, and the content-
/// addressed `ImportCache` reuses an unchanged source without re-running it — so
/// a project whose rigs were imported before that door existed keeps a rig with
/// `roles: []` for ever, and `RoleIndex` over an empty table answers `None` to
/// everything. Look-at, the aim-offset mask and the SK1b hand pass are all
/// written to do nothing in that case (deliberately, so a quadruped is left
/// alone), so the failure is silent.
///
/// Measured on the island before this existed: **24 of 26 rigs carried no
/// table** — every Manny/Quinn LOD rung and every MetaHuman body and face rung.
/// Only the two the rebind writes had one.
///
/// It is a VALUE, not a migration: a rig whose bones are called nothing the
/// convention knows infers an empty table and is written back unchanged, so the
/// sweep cannot invent a head. Idempotent — a second run re-derives the same
/// table and finds nothing to write.
fn sweep_roles(project: &mut AssetProject, report: &mut UeImportReport) -> usize {
    let rigs: Vec<AssetId> = project
        .db()
        .by_kind(inf_asset::AssetKind::Skeleton)
        .map(|e| e.sidecar.guid)
        .collect();
    let mut n = 0usize;
    for id in rigs {
        let Ok(asset) = project.load_payload::<inf_anim::SkeletonAsset>(id) else {
            continue;
        };
        if !asset.roles.is_empty() {
            continue;
        }
        let roles = inf_anim::roles::infer_roles(&asset.skeleton);
        if roles.is_empty() {
            continue;
        }
        let deps = project
            .db()
            .get(id)
            .map(|e| e.sidecar.dependencies.clone())
            .unwrap_or_default();
        let mut out = asset;
        out.roles = roles;
        match project.rewrite_payload(id, &out, deps) {
            Ok(()) => n += 1,
            Err(e) => report
                .advisories
                .push(format!("a rig's role table was not written ({e})")),
        }
    }
    n
}

fn sweep_licences(
    project: &AssetProject,
    dest: &Path,
    report: &mut UeImportReport,
) -> (usize, bool) {
    if report.licences.is_empty() {
        return (0, true);
    }
    let exact = report.licences.len() == 1;
    let pack = if exact {
        report.licences[0].0.clone()
    } else {
        format!(
            "(unattributed: {})",
            report
                .licences
                .iter()
                .map(|(p, _, _)| p.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let licence = report
        .licences
        .iter()
        .map(|(p, l, _)| {
            if exact {
                l.clone()
            } else {
                format!("{p}: {l}")
            }
        })
        .collect::<Vec<_>>()
        .join("  ||  ");
    let ship = report.licences.iter().all(|(_, _, s)| *s);
    let _ = project;
    let mut n = 0usize;
    let Ok(entries) = std::fs::read_dir(dest) else {
        return (0, exact);
    };
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().is_none_or(|x| x != "toml") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if text.contains(LICENCE_PACK_KEY) {
            continue;
        }
        let Ok(mut side) = inf_asset::AssetSidecar::load(&path.with_extension("")) else {
            continue;
        };
        let mut t = side.import.take().unwrap_or_default();
        t.insert(LICENCE_KEY.into(), licence.clone().into());
        t.insert(LICENCE_SHIP_KEY.into(), ship.into());
        t.insert(LICENCE_PACK_KEY.into(), pack.clone().into());
        side.import = Some(t);
        if side.save(&path.with_extension("")).is_ok() {
            n += 1;
        }
    }
    (n, exact)
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

/// **SPLIT A UDIM MESH INTO ONE SECTION PER TILE** (wave CHAR1a.3 audit).
///
/// # The defect this exists for
///
/// UE's `CreateCombinedFaceAndBodyMesh` welds a MetaHuman's face onto its body
/// and keeps **both halves' texture atlases**, packed as UDIM: the face's uv in
/// tile 1001 (`u` in `[0,1)`), the body's in tile 1002 (`u` in `[1,2)`). It then
/// hands the result ONE material slot, because a UE material has one atlas and
/// UE has nothing to say two with. Imported as it stood, the head sampled the
/// BODY atlas — a map of hands, feet and underwear with no face on it — and both
/// committed MetaHumans stood on the island with blank, featureless heads.
/// Measured on the imported geometry: 34 514 triangles in tile 1001 spanning
/// y 1.3962…1.7798 m, 60 816 in tile 1002 spanning −0.0016…1.4792 m, and **zero**
/// triangles straddling a tile.
///
/// # The rule
///
/// A tile is an atlas and this engine samples one atlas per material, so a tile
/// is a SECTION. The bridge declares the combined mesh's material slots in TILE
/// ORDER (`metahuman.py`'s `_write_udim_slots`: slot *k* is the material of tile
/// 1001 + *k*), and this measures the tiles off the geometry and splits the
/// submesh accordingly — so neither end guesses which half is which.
///
/// Each section's uv is rebased into `[0,1)` by subtracting its tile. The
/// virtual-texture path wraps (`uv - floor(uv)` in `vt_sample.wgsl`) and would
/// not need it; every other reader of a `.inf_mesh` uv — a thumbnail, a bake, a
/// future non-VT sampler — does, and a section whose uv means "this atlas" is
/// the honest payload.
///
/// # What it refuses
///
/// Silence, in every case that is not the one above: more than one submesh (the
/// glTF already told us its sections), a payload that already declares as many
/// slots as the manifest, a tile count that is not the declared slot count, or a
/// single triangle straddling two tiles. Each is REPORTED and the mesh is left
/// exactly as it was.
fn split_udim_sections(
    project: &mut AssetProject,
    mesh: AssetId,
    sk: &SkeletalMesh,
    report: &mut UeImportReport,
) {
    if sk.material_slots.len() < 2 {
        return;
    }
    let Ok(mut asset) = project.load_payload::<inf_mesh::MeshAsset>(mesh) else {
        return;
    };
    let before = asset.submeshes.len();
    let (tiles, submeshes) = match asset.split_uv_tiles() {
        inf_mesh::UvTileSplit::NotNeeded => return,
        inf_mesh::UvTileSplit::Straddling(n) => {
            report.advisories.push(format!(
                "{}: {n} triangles straddle a uv tile boundary — the mesh is left \
                 unsplit and its {} material slots draw as one",
                sk.key,
                sk.material_slots.len()
            ));
            return;
        }
        inf_mesh::UvTileSplit::Split { tiles, submeshes } => (tiles, submeshes),
    };
    if tiles.len() != sk.material_slots.len() {
        report.advisories.push(format!(
            "{}: the uv occupies {} tiles {tiles:?} and the manifest declares {} \
             material slots — the mesh is left unsplit rather than split into \
             sections nothing can name",
            sk.key,
            tiles.len(),
            sk.material_slots.len()
        ));
        return;
    }
    let census: Vec<String> = tiles
        .iter()
        .zip(submeshes.iter())
        .enumerate()
        .map(|(slot, (tile, sub))| {
            format!(
                "tile {} → slot {slot}: {} triangles, {} vertices",
                1001 + tile,
                sub.triangle_count(),
                sub.vertex_count()
            )
        })
        .collect();
    // The payload's slot NAMES become the manifest's own keys, so the table
    // `bind_slots` writes next is indexable by the index `SubMesh::material_slot`
    // carries — the two lists are one list now, rather than two orders zipped.
    asset.material_slots = sk
        .material_slots
        .iter()
        .enumerate()
        .map(|(i, k)| {
            k.clone()
                .unwrap_or_else(|| format!("udim_{}", 1001 + i as i32))
        })
        .collect();
    asset.submeshes = submeshes;
    let Some(entry) = project.db().get(mesh) else {
        return;
    };
    let path = entry.path.clone();
    let deps = entry.sidecar.dependencies.clone();
    let import = entry.sidecar.import.clone();
    if let Err(e) = project.write_asset_at_with_id(&path, &asset, mesh, deps, import) {
        report
            .advisories
            .push(format!("{}: uv-tile sections not written ({e})", sk.key));
        return;
    }
    report.advisories.push(format!(
        "{}: {before} submesh(es) spanning {} uv tiles are now one section per tile ({})",
        sk.key,
        census.len(),
        census.join("; ")
    ));
}

/// **Bind a skeletal mesh's material slots into its payload** (`.inf_mesh` v3).
///
/// The manifest's `material_slots` are manifest KEYS in slot order and
/// `mat_ids` maps each to the asset section 1 imported it as, so this is a
/// straight zip — and it is the only place both halves are in scope.
///
/// A slot naming a material the run did not import is left `None` and REPORTED:
/// the alternative is a face whose eyelash slot silently inherits the skin, which
/// is the state this whole feature exists to leave behind.
fn bind_slots(
    project: &mut AssetProject,
    mesh: AssetId,
    sk: &SkeletalMesh,
    mat_ids: &BTreeMap<String, AssetId>,
    report: &mut UeImportReport,
) {
    if sk.material_slots.len() < 2 {
        // One slot (or none) is what every body in this tree has, and an
        // unsectioned mesh is what every reader already draws. Writing a
        // one-entry table would cost a payload rewrite per rung for a decision
        // nobody makes.
        return;
    }
    let Ok(mut asset) = project.load_payload::<inf_mesh::MeshAsset>(mesh) else {
        return;
    };
    let mut bound = 0usize;
    let mut missing: Vec<String> = Vec::new();
    // **The index a `SubMesh::material_slot` carries is the glTF's, and the
    // index the MANIFEST's slot list carries is UE's** — and they are not the
    // same order (wave CHAR1a.3 audit). UE's exporter writes a mesh's materials
    // DEDUPLICATED, so a MetaHuman face's twelve UE slots come out as eleven
    // glTF materials and every slot after the repeat is off by one: measured on
    // `SKM_INF_Dominic_FaceMesh`, the payload's `MI_Face_Skin_Baked_LOD1` was
    // bound to `M_Hide` and its `MI_Face_EyelashesHiLODs` to the head skin.
    //
    // So the table is built by NAME where the two lists agree about one — a
    // manifest key ends with the material's own name — and falls back to the
    // positional zip only where a name is absent or ambiguous, saying so.
    let by_name = |name: &str| -> Option<&String> {
        let suffix = format!("_{name}");
        let mut hits = sk
            .material_slots
            .iter()
            .flatten()
            .filter(|k| k.as_str() == name || k.ends_with(&suffix));
        let first = hits.next()?;
        hits.next().is_none().then_some(first)
    };
    let mut positional = 0usize;
    let pairs: Vec<(u32, AssetId)> = (0..asset.material_slots.len().max(sk.material_slots.len()))
        .filter_map(|i| {
            let named = asset.material_slots.get(i).and_then(|n| by_name(n));
            let key = match named {
                Some(k) => k,
                None => {
                    positional += 1;
                    match sk.material_slots.get(i).and_then(|k| k.as_ref()) {
                        Some(k) => k,
                        None => {
                            missing.push(format!("slot {i} names no material"));
                            return None;
                        }
                    }
                }
            };
            match mat_ids.get(key) {
                Some(id) => {
                    bound += 1;
                    Some((i as u32, *id))
                }
                None => {
                    missing.push(format!("slot {i} ({key}) did not import"));
                    None
                }
            }
        })
        .collect();
    if positional > 0 {
        missing.push(format!(
            "{positional} slot(s) matched by POSITION because the payload's slot \
             name is absent from the manifest's keys or matches more than one"
        ));
    }
    // The payload's own slot NAMES have to exist for the table to be indexable
    // by the same index `SubMesh::material_slot` carries; a glTF import writes
    // one per primitive material, so a mesh whose slots the manifest knows and
    // whose payload has none is a mesh this table cannot address.
    if asset.material_slots.len() < sk.material_slots.len() {
        report.advisories.push(format!(
            "{}: the manifest states {} material slots and the imported mesh has \
             {} — the slot table is bound over the shorter list",
            sk.key,
            sk.material_slots.len(),
            asset.material_slots.len()
        ));
    }
    asset.bind_material_slots(pairs);
    let Some(entry) = project.db().get(mesh) else {
        return;
    };
    let path = entry.path.clone();
    // **A slot material is a DEPENDENCY of the mesh** (wave CHAR1a.3 audit).
    // The cook packs a level's dependency closure; a section material reachable
    // only through the payload's slot table would resolve in the editor, which
    // reads loose assets, and be absent from the `.ipack` — the exact PIE ==
    // shipping divergence the table was moved out of the sidecar to avoid.
    let mut deps = entry.sidecar.dependencies.clone();
    for id in asset.material_slot_assets.iter().flatten() {
        if !deps.contains(id) {
            deps.push(*id);
        }
    }
    let import = entry.sidecar.import.clone();
    if let Err(e) = project.write_asset_at_with_id(&path, &asset, mesh, deps, import) {
        report
            .advisories
            .push(format!("{}: slot table not written ({e})", sk.key));
        return;
    }
    report.advisories.push(format!(
        "{}: {bound} of {} material slots bound into the mesh{}",
        sk.key,
        sk.material_slots.len(),
        if missing.is_empty() {
            String::new()
        } else {
            format!(" ({})", missing.join(", "))
        }
    ));
}

/// The character LOD ladder, into the LOD-0 mesh's sidecar `import` table.
///
/// Unlike [`record_rungs`], which records a census of rungs that were **not**
/// stored, this records the rungs that **were** -- their asset ids and their
/// measured triangle counts -- plus the switch distances derived from them.
///
/// # Where the switch distances come from
///
/// Not from the pack: UE reports its auto-compute sentinel (-1) for every
/// `screen_size` in this project, measured 18 of 18 at the ASSET0 audit and
/// again here. So they are derived from the thing that is actually known, which
/// is each rung's own triangle count against the engine's crowd bands: a rung
/// takes over where a body's on-screen height makes its triangles cost about
/// what the next rung's cost at the band above. The bands themselves are
/// `inf_ecs::crowd`'s, so a character's geometry ladder and its simulation
/// ladder switch at the same three distances instead of at two unrelated sets
/// of numbers.
/// The `[import]` table on `id`'s sidecar, read **off disk** rather than out of
/// the DB (wave CHAR1b.1).
///
/// `record_character_ladder` writes through `AssetSidecar::load`/`save`, so the
/// entry the DB holds is stale the instant it returns: a reader that went to
/// `project.db().get(id).sidecar.import` would find `None` and silently carry
/// nothing, which is the same defect item 111 already is, one layer down.
fn read_import_table(project: &AssetProject, id: AssetId) -> Option<toml::Table> {
    let path = project.db().get(id)?.path.clone();
    inf_asset::AssetSidecar::load(&path).ok()?.import
}

fn record_character_ladder(
    project: &mut AssetProject,
    lod0: AssetId,
    sk: &SkeletalMesh,
    rungs: &[(u32, AssetId, usize)],
    report: &mut UeImportReport,
) {
    let Some(entry) = project.db().get(lod0) else {
        return;
    };
    let path = entry.path.clone();
    let Ok(mut side) = inf_asset::AssetSidecar::load(&path) else {
        return;
    };
    let mut t = side.import.take().unwrap_or_default();
    t.insert("ue_source".into(), sk.key.clone().into());
    t.insert("ue_pack".into(), sk.pack.clone().into());
    t.insert("ue_bones".into(), i64::from(sk.bones).into());
    t.insert(
        "ue_lod_rungs_exported".into(),
        (sk.lods.len() as i64).into(),
    );
    t.insert("ue_lod_rungs_stored".into(), (rungs.len() as i64).into());
    t.insert(
        "character_lod_assets".into(),
        toml::Value::Array(
            rungs
                .iter()
                .map(|(_, id, _)| toml::Value::String(id.uuid().to_string()))
                .collect(),
        ),
    );
    t.insert(
        "character_lod_triangles".into(),
        toml::Value::Array(
            rungs
                .iter()
                .map(|(_, _, tris)| toml::Value::Integer(*tris as i64))
                .collect(),
        ),
    );
    t.insert(
        "character_lod_switch_m".into(),
        toml::Value::Array(
            character_lod_switch_m(rungs.len())
                .iter()
                .map(|d| toml::Value::Float(*d))
                .collect(),
        ),
    );
    side.import = Some(t);
    if let Err(e) = side.save(&path) {
        report
            .advisories
            .push(format!("{}: ladder not recorded ({e})", sk.key));
    }
}

/// **Write an imported body at the starter character's committed GUIDs.**
///
/// See [`UeImportOptions::rebind_character`] for why. The three assets are
/// copied rather than moved: the originals keep their own ids and stay in the
/// drawer, so an author can see both, and the rebind is a second file with a
/// borrowed identity exactly as the material rebind is.
///
/// A body whose skeleton did not import is REFUSED rather than half-rebound: a
/// mesh at the hero's mesh GUID with the old rig still at the hero's skeleton
/// GUID is 92 000 triangles addressed to the wrong joints, which draws as an
/// explosion and is worse than the low-poly body it replaced.
// Nine, and every one of them is a distinct fact this function cannot derive: the
// body, its rig, the manifest record, the material map, WHICH committed identity
// it becomes, that identity's three asset stems and its three clip stems. A
// struct here would be a struct with one caller per field.
/// What [`split_eye_sections`] lifted out of a combined body.
struct EyeSplit {
    left_verts: usize,
    right_verts: usize,
    tris: usize,
    left_key: String,
    right_key: String,
}

/// The widest a closed island may be, metres, and still be an EYEBALL.
const EYE_ISLAND_MAX_M: f32 = 0.050;
/// The least of the uv square an eyeball's own island covers.
const EYE_ISLAND_UV_SPAN: f32 = 0.85;
/// How round a closed island must be to be an eyeball rather than the cornea
/// shell in front of one — its thinnest extent over its widest.
const EYE_ISLAND_ROUNDNESS: f32 = 0.5;

/// **Give a combined MetaHuman's eyeballs a section of their own** — the closure
/// of clause 3 (wave OUTFIT1 AUDIT, carried item 167).
///
/// # What wave OUTFIT1 measured, and the half of it that was a box
///
/// The wave refused clause 3 on three measurements, and the first of them was
/// taken with a 40 × 40 × 50 mm BOX at the right eye: **2 478 vertices**. A box
/// at an eye holds an eyeball *and the lids, the socket and the cheek around
/// it*. The eyeball itself is a **connected component of 289 vertices** — the
/// box over-counts it 8.6-fold — and once it is addressed as a component rather
/// than as a box it is trivially separable, because nothing else in a 95 330
/// triangle body looks like it:
///
/// * it is CLOSED and SMALL — 29 × 29 × 22 mm, under [`EYE_ISLAND_MAX_M`] on
///   every axis;
/// * and it is addressed with a WHOLE UV TILE — u 0.010…0.990, v 0.010…0.990,
///   which is what an eye texture is addressed with and what no piece of a UDIM
///   face atlas is. The head component spans the same u and v and is 389 mm
///   across; the teeth are 66 mm and span 0.18 of v. Both tests are needed and
///   neither alone is enough.
///
/// The wave's second and third measurements stand: the uv island survived the
/// combine, and the head albedo at those uvs is skin. This is what makes the
/// split worth doing — the eyes were being painted with whatever the face atlas
/// holds at the eye texture's own coordinates.
///
/// # Why this door and not the two the wave priced
///
/// (a) wearing the FACE mesh re-opens the neck seam the combine exists to close;
/// (b) a slot-keeping combine is a `metahuman.py` change plus a re-assembly of
/// both characters, which is a step a human presses. This is (d): an import-side
/// split, which needs neither — the geometry and the materials are both already
/// on this side of the bridge.
fn split_eye_sections(
    mesh: &mut inf_mesh::MeshAsset,
    left_is_positive_x: bool,
    mat_ids: &BTreeMap<String, AssetId>,
) -> std::result::Result<EyeSplit, String> {
    // The islands, per section, by union-find over the triangle graph.
    let mut found: Vec<(usize, Vec<u32>, f32)> = Vec::new(); // (section, verts, centre x)
    for (si, sub) in mesh.submeshes.iter().enumerate() {
        let nv = sub.vertices.len();
        let mut parent: Vec<u32> = (0..nv as u32).collect();
        fn root(p: &mut [u32], mut a: u32) -> u32 {
            while p[a as usize] != a {
                p[a as usize] = p[p[a as usize] as usize];
                a = p[a as usize];
            }
            a
        }
        for t in sub.indices.chunks_exact(3) {
            let (a, b, c) = (
                root(&mut parent, t[0]),
                root(&mut parent, t[1]),
                root(&mut parent, t[2]),
            );
            parent[b as usize] = a;
            parent[c as usize] = a;
        }
        let mut islands: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for i in 0..nv as u32 {
            let r = root(&mut parent, i);
            islands.entry(r).or_default().push(i);
        }
        for (_, vs) in islands {
            if vs.len() < 64 {
                continue;
            }
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            let mut ulo = [f32::MAX; 2];
            let mut uhi = [f32::MIN; 2];
            for &i in &vs {
                let v = &sub.vertices[i as usize];
                for a in 0..3 {
                    lo[a] = lo[a].min(v.position[a]);
                    hi[a] = hi[a].max(v.position[a]);
                }
                for a in 0..2 {
                    ulo[a] = ulo[a].min(v.uv[a]);
                    uhi[a] = uhi[a].max(v.uv[a]);
                }
            }
            let small = (0..3).all(|a| hi[a] - lo[a] < EYE_ISLAND_MAX_M);
            let whole_tile = (0..2).all(|a| uhi[a] - ulo[a] > EYE_ISLAND_UV_SPAN);
            // …and it is a BALL and not a disc. Measured: the two islands in
            // front of the eyeballs — the cornea shells — are 30 × 11 × 11 mm
            // and are also addressed with a whole tile, so `small` and
            // `whole_tile` alone match four islands and an eyeball pair is two.
            // An eyeball is 29 × 29 × 22: its thinnest axis is three quarters of
            // its widest, and a shell's is a third.
            let extent = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            let widest = extent.iter().cloned().fold(0.0f32, f32::max);
            let thinnest = extent.iter().cloned().fold(f32::MAX, f32::min);
            let closed = widest > 1e-6 && thinnest / widest > EYE_ISLAND_ROUNDNESS;
            if small && whole_tile && closed {
                found.push((si, vs, (hi[0] + lo[0]) / 2.0));
            }
        }
    }
    if found.len() != 2 {
        return Err(format!(
            "{} island(s) are closed, under {:.0} mm across and addressed with a \
             whole uv tile, and an eyeball pair is exactly two",
            found.len(),
            EYE_ISLAND_MAX_M * 1000.0
        ));
    }
    // The eye MATERIALS, named off the section's own binding rather than off the
    // character's name: the face-skin slot this island sits in is
    // `<...>_Face_Materials_MI_Face_Skin_Baked_LOD1_<...>`, and the eyes are the
    // two siblings of it that Unreal binds by slot on the face mesh.
    let si = found[0].0;
    let slot = mesh.submeshes[si].material_slot.unwrap_or(0) as usize;
    let face = mesh.material_slots.get(slot).cloned().unwrap_or_default();
    let Some(stem) = face.split("_Materials_").next().map(|s| s.to_string()) else {
        return Err(format!(
            "the section's slot `{face}` names no material folder"
        ));
    };
    let key_for = |side: &str| format!("{stem}_Materials_MI_Eye{side}_Baked_MI_Eye{side}_Baked");
    let (lk, rk) = (key_for("L"), key_for("R"));
    let (Some(lm), Some(rm)) = (mat_ids.get(&lk), mat_ids.get(&rk)) else {
        return Err(format!(
            "neither {} nor {} was imported in this run — the eye materials live \
             on the FACE mesh, so `--only` has to let it across",
            short_name(&lk),
            short_name(&rk)
        ));
    };
    // Slots first, so both new sections index a slot that exists.
    while mesh.material_slot_assets.len() < mesh.material_slots.len() {
        mesh.material_slot_assets.push(None);
    }
    let left_slot = mesh.material_slots.len() as u32;
    mesh.material_slots.push(lk.clone());
    mesh.material_slot_assets.push(Some(*lm));
    let right_slot = mesh.material_slots.len() as u32;
    mesh.material_slots.push(rk.clone());
    mesh.material_slot_assets.push(Some(*rm));

    let mut out = EyeSplit {
        left_verts: 0,
        right_verts: 0,
        tris: 0,
        left_key: lk,
        right_key: rk,
    };
    let mut keep: Vec<bool> = vec![true; mesh.submeshes[si].vertices.len()];
    let mut new_sections: Vec<inf_mesh::SubMesh> = Vec::new();
    for (_, vs, cx) in &found {
        let is_left = (*cx > 0.0) == left_is_positive_x;
        let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
        let src = &mesh.submeshes[si];
        let mut sub = inf_mesh::SubMesh {
            name: if is_left {
                "EyeL".into()
            } else {
                "EyeR".into()
            },
            vertices: Vec::with_capacity(vs.len()),
            indices: Vec::new(),
            material_slot: Some(if is_left { left_slot } else { right_slot }),
            skin: Vec::with_capacity(vs.len()),
        };
        for &i in vs {
            keep[i as usize] = false;
            remap.insert(i, sub.vertices.len() as u32);
            sub.vertices.push(src.vertices[i as usize]);
            if let Some(s) = src.skin.get(i as usize) {
                sub.skin.push(*s);
            }
        }
        for t in src.indices.chunks_exact(3) {
            if let (Some(a), Some(b), Some(c)) =
                (remap.get(&t[0]), remap.get(&t[1]), remap.get(&t[2]))
            {
                sub.indices.extend_from_slice(&[*a, *b, *c]);
            }
        }
        out.tris += sub.triangle_count();
        if is_left {
            out.left_verts = sub.vertices.len();
        } else {
            out.right_verts = sub.vertices.len();
        }
        new_sections.push(sub);
    }
    // …and the face-skin section loses exactly those vertices and their
    // triangles. Compacted rather than left as orphans: an unreferenced vertex
    // is a vertex the cook packs and the skinning pass transforms.
    {
        let src = &mut mesh.submeshes[si];
        let mut remap: Vec<Option<u32>> = vec![None; src.vertices.len()];
        let mut verts = Vec::with_capacity(src.vertices.len());
        let mut skin = Vec::with_capacity(src.skin.len());
        for (i, k) in keep.iter().enumerate() {
            if *k {
                remap[i] = Some(verts.len() as u32);
                verts.push(src.vertices[i]);
                if let Some(s) = src.skin.get(i) {
                    skin.push(*s);
                }
            }
        }
        let mut indices = Vec::with_capacity(src.indices.len());
        for t in src.indices.chunks_exact(3) {
            if let (Some(a), Some(b), Some(c)) = (
                remap[t[0] as usize],
                remap[t[1] as usize],
                remap[t[2] as usize],
            ) {
                indices.extend_from_slice(&[a, b, c]);
            }
        }
        src.vertices = verts;
        src.skin = skin;
        src.indices = indices;
    }
    mesh.submeshes.extend(new_sections);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn rebind_character(
    project: &mut AssetProject,
    mesh: AssetId,
    skeleton: Option<AssetId>,
    sk: &SkeletalMesh,
    mat_ids: &BTreeMap<String, AssetId>,
    // **Which committed character this body becomes** (wave CHAR1a.3) — the
    // male starter's identity set or the female's. The stems ride with it,
    // because a rebind writes at a committed GUID *and* at that GUID's committed
    // FILE NAME, so the asset scan finds one asset rather than two claiming one
    // id.
    ids: &crate::character::CharacterIds,
    stems: (&str, &str, &str),
    clip_stems: (&str, &str, &str),
    report: &mut UeImportReport,
) -> Result<()> {
    let (Some(want_mesh), Some(want_skel), Some(want_mat)) = (ids.mesh, ids.skeleton, ids.material)
    else {
        return Ok(());
    };
    let Some(skeleton) = skeleton else {
        return Err(AssetError::Import(format!(
            "{}: refusing to rebind a body whose skeleton did not import — the \
             mesh's joint indices would address the rig it replaced",
            sk.key
        )));
    };
    let mut skel: inf_anim::SkeletonAsset = project.load_payload(skeleton)?;
    // **A rebound rig gets a ROLE TABLE** (wave CHAR1b.1). The glTF stage infers
    // one for a rig it writes, and this path does not always reach it: the
    // importer's content-addressed cache reuses an asset whose source has not
    // changed, so a project imported before that door existed keeps its
    // table-less rig for ever. This is the door that always runs — it writes the
    // committed GUID the level references — and a rig that already has a table
    // keeps it.
    //
    // What it costs to skip: a `RoleIndex` over an empty table answers `None` to
    // everything, and look-at, the aim-offset mask and the SK1b hand pass are
    // each written to do nothing when it does. Measured in PIE on the island: the
    // hero's aim reached -165.14 degrees and its head drew 0.00.
    if skel.roles.is_empty() {
        skel.roles = inf_anim::roles::infer_roles(&skel.skeleton);
        report.advisories.push(format!(
            "{}.inf_skel: inferred {} bone roles from the rig's own names",
            stems.0,
            skel.roles.len()
        ));
    }
    let mut body: inf_mesh::MeshAsset = project.load_payload(mesh)?;
    // **THE EYES GET A SECTION OF THEIR OWN** (wave OUTFIT1 AUDIT, carried item
    // 167). Before the dependency list below, because the split adds two
    // material references the cook's closure has to see.
    let bind =
        inf_anim::pose::global_transforms(&skel.skeleton, &inf_anim::Pose::rest(&skel.skeleton));
    let hand = |n: &str| {
        skel.skeleton
            .index_of(n)
            .and_then(|i| bind.get(i as usize))
            .map(|m| m.w_axis.x)
    };
    // Which side of the rig is its LEFT, from the rig's own `_l`/`_r` pair
    // rather than from a convention this file would have to be told. `true`
    // when nothing answers, which is the MetaHuman/UE frame this bridge writes.
    let left_is_positive_x = match (hand("hand_l"), hand("hand_r")) {
        (Some(l), Some(r)) => l > r,
        _ => true,
    };
    match split_eye_sections(&mut body, left_is_positive_x, mat_ids) {
        Ok(split) => report.advisories.push(format!(
            "{}: the EYES have a section of their own — two islands of {} and {} \
             vertices, {} triangles between them, lifted out of the face-skin \
             section and bound to {} and {}. They drew with the head atlas at \
             their own uvs before this, which is skin.",
            sk.key,
            split.left_verts,
            split.right_verts,
            split.tris,
            short_name(&split.left_key),
            short_name(&split.right_key),
        )),
        Err(why) => report.advisories.push(format!(
            "{}: the eyes were NOT split out — {why}. They draw with whatever \
             the head atlas holds at their own uvs.",
            sk.key
        )),
    }
    let body = body;
    let root = project.root().to_path_buf();
    // **The rig this identity WORE**, read before it is replaced — see
    // `retarget_committed_clips`. `None` on a first rebind into a project that
    // has never had this character, which is also the case where there is
    // nothing to re-retarget.
    let previous: Option<inf_anim::Skeleton> = project
        .load_payload::<inf_anim::SkeletonAsset>(want_skel)
        .ok()
        .map(|a| a.skeleton);

    let skel_path = root.join(format!("{}.inf_skel", stems.0));
    project.write_asset_at_with_id(&skel_path, &skel, want_skel, vec![], None)?;
    let mesh_path = root.join(format!("{}.inf_mesh", stems.1));
    // The rig, AND every material the body's own slot table names (wave CHAR1a.3
    // audit): a rebound MetaHuman draws its head from the face atlas and its
    // body from the body atlas, and a cook that cannot see those edges packs
    // neither.
    let mut mesh_deps = vec![want_skel];
    for id in body.material_slot_assets.iter().flatten() {
        if !mesh_deps.contains(id) {
            mesh_deps.push(*id);
        }
    }
    // **AND THE LOD LADDER COMES WITH IT** (carried item 111).
    //
    // `record_character_ladder` writes `character_lod_assets`,
    // `character_lod_triangles` and `character_lod_switch_m` into the sidecar of
    // the asset the IMPORT just minted, a few lines before this call — and this
    // call wrote `None`, which on the fresh-file path of
    // `write_asset_at_with_id` does not merely fail to add the table, it WIPES
    // whatever was there. So the committed GUID the level references had no
    // ladder at all and the island's hero drew its LOD-0 mesh at every distance.
    //
    // The rung ids go onto the dependency list too, for the reason `bind_slots`
    // records one section over: a reference a sidecar does not carry is a
    // reference the cook's closure cannot see, and a ladder whose rungs are not
    // packed is a ladder that does not exist in a shipped game.
    //
    // **The second half of item 111 is still open and is not this**: nothing in
    // the engine READS `character_lod_assets` — the whole-repo search finds the
    // writer and no consumer, `SkeletalMesh` carries one mesh GUID, and
    // `resolve_skinned` takes no view. Carrying the ladder is necessary and not
    // sufficient; the selector is PERF1's, with the numbers this wave's ledger
    // prints.
    let ladder = read_import_table(project, mesh);
    if let Some(t) = &ladder {
        if let Some(toml::Value::Array(rungs)) = t.get("character_lod_assets") {
            for v in rungs {
                let Some(id) = v
                    .as_str()
                    .and_then(|s| uuid::Uuid::parse_str(s).ok())
                    .map(AssetId)
                else {
                    continue;
                };
                if id != want_mesh && !mesh_deps.contains(&id) {
                    mesh_deps.push(id);
                }
            }
        }
    }
    project.write_asset_at_with_id(&mesh_path, &body, want_mesh, mesh_deps, ladder)?;
    report
        .rebinds
        .push((format!("{}.inf_skel", stems.0), want_skel));
    report
        .rebinds
        .push((format!("{}.inf_mesh", stems.1), want_mesh));
    // **THE LICENCE FOLLOWS THE BYTES** (wave OUTFIT1). `sweep_licences` reads
    // the DESTINATION folder and `stamp_licences` reads `asset_packs`, and a
    // rebind writes neither: it writes at a committed GUID in the project ROOT.
    // So the only assets in the project that actually SHIP -- the ones the
    // level references -- were the ones with no licence row on disk. Measured on
    // the island: `Starter_Body.inf_mesh`, `Starter.inf_skel`, `Starter_Skin.inf_mat`
    // and every derived `.inf_vmesh` beside them carried nothing, while 125
    // sidecars in `Content/UE/...` that nothing references carried the row.
    for id in [want_skel, want_mesh, want_mat] {
        report.asset_packs.push((id, sk.pack.clone()));
    }

    // The skin an instance wears — the material a reader that draws no sections
    // puts over the whole body, so it is the mesh's DOMINANT slot rather than
    // slot 0. On both mannequins that is slot 0 anyway (measured: `M_torso` on
    // Manny, `Quinn_01` on Quinn), and on a UDIM MetaHuman slot 0 is the FACE
    // atlas — 34 514 triangles against the body's 60 816 — so taking slot 0
    // would put a head texture over a whole person the moment a section did not
    // resolve (wave CHAR1a.3 audit).
    let dominant = {
        let mut tris: BTreeMap<u32, usize> = BTreeMap::new();
        for sub in &body.submeshes {
            *tris.entry(sub.material_slot.unwrap_or(0)).or_default() += sub.triangle_count();
        }
        tris.into_iter()
            .max_by_key(|(slot, n)| (*n, std::cmp::Reverse(*slot)))
            .map(|(slot, _)| slot as usize)
            .unwrap_or(0)
    };
    if let Some(mat) = sk
        .material_slots
        .get(dominant)
        .and_then(|s| s.as_ref())
        .and_then(|k| mat_ids.get(k))
    {
        let payload: MaterialAsset = project.load_payload(*mat)?;
        let deps = payload.texture_dependencies();
        let path = root.join(format!("{}.inf_mat", stems.2));
        project.write_asset_at_with_id(&path, &payload, want_mat, deps, None)?;
        report
            .rebinds
            .push((format!("{}.inf_mat", stems.2), want_mat));
    } else {
        report.advisories.push(format!(
            "{}: rebound the body and its rig, but slot 0 named no imported \
             material — the hero keeps the starter's neutral skin",
            sk.key
        ));
    }
    report.advisories.push(format!(
        "{}: REBOUND at the starter character's GUIDs — this project's hero is \
         now that body ({} triangles, {} joints). Local only.",
        sk.key,
        body.triangle_count(),
        skel.skeleton.len()
    ));
    // **AND THE CLIPS COME WITH IT** — see `retarget_committed_clips`.
    retarget_committed_clips(
        project,
        ids,
        clip_stems,
        previous.as_ref(),
        &skel.skeleton,
        report,
    );
    Ok(())
}

/// **A wearable to write at a committed character's clothes GUID** (wave
/// OUTFIT1) — one entry of [`UeImportOptions::wearables`].
///
/// The three facts a rebind cannot derive: WHICH imported asset (a substring of
/// its manifest key), WHICH committed identity wears it (`female`), and WHICH
/// SLOT it goes in — because an outfit and a head of hair land at two different
/// GUIDs and one of them is bound rigidly to a joint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WearableRebind {
    /// A substring of the manifest key of the mesh to import as this wearable.
    pub key: String,
    /// `false` = the male starter's clothes GUIDs, `true` = the female's.
    pub female: bool,
    /// `true` = the HAIR slot (`CharacterIds::hair`), `false` = the OUTFIT slot.
    pub hair: bool,
    /// The joint a mesh with **no skin stream** is bound rigidly to.
    ///
    /// A groom's cards are a `StaticMesh` in Unreal and arrive here with
    /// positions and no influences at all. Binding every vertex to the head at
    /// weight 1 is not an approximation of what a card set does — it is exactly
    /// what a card set does, and it is what makes the mesh a WEARABLE
    /// (`inf_ecs::wearable` reads the skeleton GUID, and a mesh with no skin
    /// carries no joint indices to read).
    pub joint: String,
}

/// Parse one `--wearable` value: `<m|f>:<outfit|hair>:<key substring>[:<joint>]`.
///
/// A refusal is a message naming what was wrong, because this is a command-line
/// argument and the operator is the only reader.
pub fn parse_wearable(v: &str) -> std::result::Result<WearableRebind, String> {
    let parts: Vec<&str> = v.splitn(4, ':').collect();
    if parts.len() < 3 {
        return Err(format!(
            "--wearable wants <m|f>:<outfit|hair>:<key>[:<joint>], got {v:?}"
        ));
    }
    let female = match parts[0] {
        "m" | "male" => false,
        "f" | "female" => true,
        other => {
            return Err(format!(
                "--wearable's first field is `m` or `f`, got {other:?}"
            ))
        }
    };
    let hair = match parts[1] {
        "outfit" | "clothes" => false,
        "hair" | "groom" => true,
        other => {
            return Err(format!(
                "--wearable's second field is `outfit` or `hair`, got {other:?}"
            ))
        }
    };
    if parts[2].trim().is_empty() {
        return Err("--wearable's key is empty, which matches every asset".into());
    }
    Ok(WearableRebind {
        key: parts[2].to_string(),
        female,
        hair,
        joint: parts
            .get(3)
            .map(|s| s.to_string())
            .unwrap_or_else(|| DEFAULT_WEARABLE_JOINT.to_string()),
    })
}

/// **How far a rebound garment stands off the body it is fitted to**, metres.
///
/// # Measured, on the frame it exists for
///
/// A MetaHuman's default garment is modelled ON its body, so the two surfaces are
/// coincident to within a fraction of a millimetre and the depth test picks
/// whichever won the rounding: the wave's first portrait shows a white tee with
/// **brown patches of the character's own chest showing through it**. UE hides
/// the covered body triangles behind a per-garment mask; this engine's combined
/// body has no such mask and no section to hide it with.
///
/// So the garment is pushed out along its own normals, which is the same rule
/// `crate::groom::wearable_shell` uses to make a shell garment visible at all.
/// 4 mm is a shirt's own thickness — enough that no rounding can put the body in
/// front of it, small enough that the silhouette is the garment's own.
///
/// It applies to a SKINNED wearable only. A groom's cards already stand off the
/// scalp by their own geometry, and pushing a hair card out along a normal that
/// points along the card would shear the hairstyle.
pub const WEARABLE_LIFT_M: f32 = 0.004;

/// How far a garment vertex will look for the body vertex whose weights it
/// takes, metres (wave OUTFIT1 AUDIT, carried item 166).
const WEARABLE_WEIGHT_REACH_M: f32 = 0.060;

/// **A garment deforms by exactly what the body under it deforms by** — the
/// second half of carried item 166's closure.
///
/// # Why re-pointing the garment's own influences is not enough
///
/// A MetaHuman garment arrives with its OWN skin weights, authored for cloth,
/// and this bridge re-points their joint indices by name onto the wearer's rig.
/// That is correct arithmetic and it does not make the garment fit: the two
/// surfaces are weighted differently, so they diverge the moment the character
/// leaves its bind pose, and a shirt that encloses a body in bind pose has the
/// chest through it in an idle. Measured on the island's own portrait: the
/// bind-pose intrusion fell from 11.60 % to 3.57 % of the garment's vertices
/// under `fit_wearable_over_wearer` alone, and 4.35 % of the POSED shirt was
/// still the character's skin.
///
/// So a garment takes the weights of the body vertex nearest to it — which is
/// exactly the rule [`crate::groom::wearable_shell`] already uses for the
/// committed defaults, where the skin stream is kept UNCHANGED for the same
/// reason: no transfer, no proximity solve, no seam, and the garment cannot
/// diverge from the body because it is driven by the same numbers.
///
/// A vertex with no body vertex within [`WEARABLE_WEIGHT_REACH_M`] keeps the
/// re-pointed weights it arrived with — a cape's far corner, a hat's crown —
/// and both counts are returned so the log can say which happened.
fn wear_the_wearers_weights(
    mesh: &mut inf_mesh::MeshAsset,
    body: &inf_mesh::MeshAsset,
) -> (usize, usize) {
    let cell = WEARABLE_WEIGHT_REACH_M;
    let key = |p: [f32; 3]| {
        [
            (p[0] / cell).floor() as i32,
            (p[1] / cell).floor() as i32,
            (p[2] / cell).floor() as i32,
        ]
    };
    let mut grid: BTreeMap<[i32; 3], Vec<([f32; 3], inf_mesh::VertexSkin)>> = BTreeMap::new();
    for sub in &body.submeshes {
        for (i, v) in sub.vertices.iter().enumerate() {
            let Some(skin) = sub.skin.get(i) else {
                continue;
            };
            grid.entry(key(v.position))
                .or_default()
                .push((v.position, *skin));
        }
    }
    let (mut moved, mut total) = (0usize, 0usize);
    for sub in &mut mesh.submeshes {
        for (i, v) in sub.vertices.iter().enumerate() {
            let Some(slot) = sub.skin.get_mut(i) else {
                continue;
            };
            total += 1;
            let k = key(v.position);
            let mut best = WEARABLE_WEIGHT_REACH_M * WEARABLE_WEIGHT_REACH_M;
            let mut take: Option<inf_mesh::VertexSkin> = None;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(list) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) else {
                            continue;
                        };
                        for (p, skin) in list {
                            let d = [
                                p[0] - v.position[0],
                                p[1] - v.position[1],
                                p[2] - v.position[2],
                            ];
                            let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            if l2 < best {
                                best = l2;
                                take = Some(*skin);
                            }
                        }
                    }
                }
            }
            if let Some(skin) = take {
                *slot = skin;
                moved += 1;
            }
        }
    }
    (moved, total)
}

/// **How far a fitted garment clears the body under it**, metres (wave OUTFIT1
/// AUDIT, carried item 166).
pub const WEARABLE_FIT_MARGIN_M: f32 = 0.002;

/// **The ceiling on a fit**, metres — no garment vertex is pushed further than
/// this however deep the body is behind it.
///
/// A ceiling is needed because the search below is a MAXIMUM, and a maximum over
/// a body that folds under a garment (an armpit, the crook of an elbow) will
/// name a distance no shirt has. 30 mm is three times a shirt's thickness and
/// still under the 35 mm the deepest measured intrusion needed; a vertex that
/// hits it is COUNTED and reported rather than silently clamped.
pub const WEARABLE_FIT_MAX_M: f32 = 0.030;

/// How far off a garment vertex's own normal a body vertex may be and still
/// count as "under" it, metres.
///
/// This is the whole difference between a fit and a balloon. A garment ends: at
/// a sleeve, a hem and a neck the body CONTINUES, and a body vertex 30 mm to the
/// side of the hem is 30 mm "outside" along the hem's normal without being under
/// it at all. So the search is a narrow CYLINDER along the normal and not a
/// ball, and the hem stays where the garment put it.
const WEARABLE_FIT_TANGENT_M: f32 = 0.008;

/// How far along the normal, in both directions, the fit looks for the body.
const WEARABLE_FIT_REACH_M: f32 = 0.060;

/// What [`fit_wearable_over_wearer`] did.
struct WearableFit {
    lifted: usize,
    pushed: usize,
    capped: usize,
    mean_mm: f32,
    max_mm: f32,
}

/// **Push a rebound garment out until the body it is on is inside it** — the
/// closure of carried item 166 (wave OUTFIT1 AUDIT).
///
/// # What the flat lift could not do, measured
///
/// Wave OUTFIT1 pushed every garment vertex a flat 4 mm along its own normal,
/// which is a shirt's thickness and is the right answer for the defect it was
/// aimed at: two coincident surfaces and a depth test picking whichever won the
/// rounding. It is the wrong answer for the defect that was actually there. A
/// MetaHuman garment is modelled on ITS OWN body and this engine re-points its
/// influences onto the wearer's rig by name, and the two surfaces are not
/// coincident at all: measured on the island's hero, **11.60 %** of the
/// garment's 11 470 vertices had the body OUTSIDE them after the flat lift, by
/// a median of 9.5 mm and a maximum of 35.0 mm — so 4.54 % of the whole visible
/// shirt in the wave's own portrait was the character's chest, not a shirt.
///
/// # The rule
///
/// For each garment vertex, the deepest the body reaches along that vertex's own
/// normal, within a narrow cylinder around it — plus a margin, floored at the
/// flat lift and capped at [`WEARABLE_FIT_MAX_M`]. The offset field is smooth
/// because it is a maximum over a ball of body vertices rather than a
/// nearest-neighbour, so no smoothing pass is needed and none is done: a
/// smoothing pass would pull the field back down exactly where it is load
/// bearing.
///
/// This is a LIFT and not UE's per-garment body hide mask, deliberately. The
/// hide mask would have to remove the covered body triangles, and in this engine
/// the body a dressed character wears is the same asset its crowd archetype
/// wears at `Far` — where `set_tier_wearables` deliberately takes the clothes
/// OFF. A baked hide mask would put a hole in the chest of every undressed agent
/// on the island. The lift touches only the garment.
fn fit_wearable_over_wearer(
    mesh: &mut inf_mesh::MeshAsset,
    body: &inf_mesh::MeshAsset,
) -> WearableFit {
    let cell = WEARABLE_FIT_REACH_M;
    let key = |p: [f32; 3]| {
        [
            (p[0] / cell).floor() as i32,
            (p[1] / cell).floor() as i32,
            (p[2] / cell).floor() as i32,
        ]
    };
    let mut grid: BTreeMap<[i32; 3], Vec<[f32; 3]>> = BTreeMap::new();
    for sub in &body.submeshes {
        for v in &sub.vertices {
            grid.entry(key(v.position)).or_default().push(v.position);
        }
    }
    let mut fit = WearableFit {
        lifted: 0,
        pushed: 0,
        capped: 0,
        mean_mm: 0.0,
        max_mm: 0.0,
    };
    let mut total = 0.0f64;
    for sub in &mut mesh.submeshes {
        for v in &mut sub.vertices {
            let n = v.normal;
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len <= 1e-6 {
                continue;
            }
            let n = [n[0] / len, n[1] / len, n[2] / len];
            let k = key(v.position);
            let mut deepest = 0.0f32;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(list) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) else {
                            continue;
                        };
                        for b in list {
                            let d = [
                                b[0] - v.position[0],
                                b[1] - v.position[1],
                                b[2] - v.position[2],
                            ];
                            let along = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
                            if !(0.0..=WEARABLE_FIT_REACH_M).contains(&along) {
                                continue;
                            }
                            let t = [
                                d[0] - along * n[0],
                                d[1] - along * n[1],
                                d[2] - along * n[2],
                            ];
                            if t[0] * t[0] + t[1] * t[1] + t[2] * t[2]
                                > WEARABLE_FIT_TANGENT_M * WEARABLE_FIT_TANGENT_M
                            {
                                continue;
                            }
                            deepest = deepest.max(along);
                        }
                    }
                }
            }
            let want = if deepest > 0.0 {
                deepest + WEARABLE_FIT_MARGIN_M
            } else {
                0.0
            };
            if want > WEARABLE_LIFT_M {
                fit.pushed += 1;
            }
            if want > WEARABLE_FIT_MAX_M {
                fit.capped += 1;
            }
            let offset = want.clamp(WEARABLE_LIFT_M, WEARABLE_FIT_MAX_M);
            for (axis, c) in n.iter().enumerate() {
                v.position[axis] += c * offset;
            }
            fit.lifted += 1;
            total += offset as f64;
            fit.max_mm = fit.max_mm.max(offset * 1000.0);
        }
    }
    if fit.lifted > 0 {
        fit.mean_mm = (total / fit.lifted as f64) as f32 * 1000.0;
    }
    fit
}

/// The joint a skinless wearable is bound to when `--wearable` names none.
///
/// `head`, because the skinless wearables this bridge crosses are groom cards
/// and every one of them is hair on a head. UE's own bone name, which both rigs
/// in this tree use.
pub const DEFAULT_WEARABLE_JOINT: &str = "head";

/// **Write an imported mesh at a committed character's OUTFIT or HAIR GUID**
/// (wave OUTFIT1) — `rebind_character`'s rule, one asset kind over.
///
/// # Why a wearable needs a rebind and not just an import
///
/// The same reason a body did. The island's hero wears the clothes at
/// `CharacterIds::outfit` / `::hair`, the level that names those GUIDs is
/// committed and byte-locked, and a MetaHuman's clothes are licensed content that
/// may never enter this repository. So the imported garment is written **at the
/// committed identity**, into the local project only, and the level does not know
/// which one it got.
///
/// # And why it needs a JOINT REMAP that a body did not
///
/// A body rebind moves the mesh *and its rig* together, so the mesh's joint
/// indices stay addressed to the skeleton beside them. A garment cannot: it has
/// to be posed by the skeleton the BODY is already using, and two skeletal meshes
/// exported from one Unreal skeleton do not necessarily arrive with the same
/// joint ORDER — a glTF skin lists the joints that mesh uses. So every influence
/// is re-pointed **by name** onto the target rig, exactly as
/// `inf_anim::retarget::RetargetMap::shared_names` re-points a clip's tracks, and
/// an influence whose bone the target does not have is dropped and the rest
/// renormalized.
///
/// A mesh with **no skin stream at all** (a groom's cards, which are a
/// `StaticMesh` in Unreal) is bound rigidly to `joint` at weight 1. That is not
/// an approximation: a hair card set rides the head bone and nothing else.
///
/// # Refusals
///
/// * a target identity with no `outfit`/`hair` GUID — nothing to write at;
/// * a target rig that does not resolve — a garment addressed to a rig that is
///   not there would draw at the origin in its bind pose;
/// * a rigid bind whose named joint the target rig does not have;
/// * **a remap that reached no joint at all** — every influence dropped means the
///   two rigs share no bone names, and a garment silently collapsed onto joint 0
///   is a shirt in a heap at the character's feet.
#[allow(clippy::too_many_arguments)]
fn rebind_wearable(
    project: &mut AssetProject,
    source_mesh: AssetId,
    source_skeleton: Option<AssetId>,
    manifest_mat: Option<AssetId>,
    ids: &crate::character::CharacterIds,
    spec: &WearableRebind,
    stem: &str,
    mat_stem: &str,
    key: &str,
    pack: &str,
    report: &mut UeImportReport,
) -> Result<()> {
    let (want_mesh, want_mat) = if spec.hair {
        (ids.hair, ids.hair_material)
    } else {
        (ids.outfit, ids.outfit_top)
    };
    let (Some(want_mesh), Some(want_skel)) = (want_mesh, ids.skeleton) else {
        return Ok(());
    };
    let target: inf_anim::SkeletonAsset = project.load_payload(want_skel).map_err(|e| {
        AssetError::Import(format!(
            "{key}: refusing to rebind a wearable whose target rig does not \
             resolve ({e}) — it would draw at the origin in its bind pose"
        ))
    })?;
    let mut mesh: inf_mesh::MeshAsset = project.load_payload(source_mesh)?;
    let skinned = mesh.submeshes.iter().any(|s| s.is_skinned());
    let mut remapped = 0usize;
    let mut dropped = 0usize;
    let mut fallback = 0usize;
    // Where a vertex whose every influence was dropped goes. The spec's joint if
    // the target rig has it, else joint 0 — which is what a rig with no such
    // bone can offer, and is why the skinless branch below refuses on the same
    // lookup rather than guessing.
    let fallback_joint = target.skeleton.index_of(&spec.joint).unwrap_or(0);
    if skinned {
        let source: inf_anim::SkeletonAsset = match source_skeleton {
            Some(id) => project.load_payload(id)?,
            None => {
                return Err(AssetError::Import(format!(
                    "{key}: the mesh carries skin weights and no skeleton was \
                     imported with it, so its joint indices cannot be re-pointed"
                )))
            }
        };
        // source index → target index, BY NAME. `None` is a bone the target rig
        // does not have, which is what a garment exported with helper joints on
        // it produces.
        let map: Vec<Option<u16>> = source
            .skeleton
            .joints()
            .iter()
            .map(|j| target.skeleton.index_of(&j.name))
            .collect();
        for sub in &mut mesh.submeshes {
            for k in &mut sub.skin {
                let mut joints = [0u16; 4];
                let mut weights = [0.0f32; 4];
                for i in 0..4 {
                    match map.get(k.joints[i] as usize).copied().flatten() {
                        Some(t) => {
                            joints[i] = t;
                            weights[i] = k.weights[i];
                            remapped += 1;
                        }
                        None => dropped += 1,
                    }
                }
                let sum: f32 = weights.iter().sum();
                *k = if sum > 1e-6 {
                    inf_mesh::VertexSkin { joints, weights }.normalized()
                } else {
                    // Every influence dropped for THIS vertex: every bone that
                    // moved it is one the target rig does not have. It goes to
                    // the SPEC'S OWN JOINT and not to the root — a vertex pinned
                    // to the root of a character rig is a vertex at the
                    // character's FEET, which is a garment torn open and dragged
                    // to the floor. `head` is the default for the same reason a
                    // skinless wearable takes it: what a wearable loses on this
                    // bridge is FACIAL influences.
                    fallback += 1;
                    inf_mesh::VertexSkin {
                        joints: [fallback_joint, 0, 0, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    }
                };
            }
        }
        // …and it stands off the body it is fitted to, so the depth test cannot
        // put the chest in front of the shirt — by the FLAT lift where nothing
        // is known about the body, and by the FITTED one where the body is in
        // hand. See `WEARABLE_LIFT_M` and `fit_wearable_over_wearer`.
        let wearer: Option<inf_mesh::MeshAsset> =
            ids.mesh.and_then(|id| project.load_payload(id).ok());
        match &wearer {
            Some(body) => {
                let moved = wear_the_wearers_weights(&mut mesh, body);
                report.advisories.push(format!(
                    "{key}: {} of {} vertices took the WEARER's own skin weights, so the \
                     garment deforms by exactly what the body under it \
                     deforms by",
                    moved.0, moved.1
                ));
                let fit = fit_wearable_over_wearer(&mut mesh, body);
                report.advisories.push(format!(
                    "{key}: {} vertices FITTED over the wearer's own surface — \
                     {} of them needed more than the flat {:.0} mm (mean {:.1} mm, \
                     max {:.1} mm, {} at the {:.0} mm ceiling)",
                    fit.lifted,
                    fit.pushed,
                    WEARABLE_LIFT_M * 1000.0,
                    fit.mean_mm,
                    fit.max_mm,
                    fit.capped,
                    WEARABLE_FIT_MAX_M * 1000.0
                ));
            }
            None => {
                let mut lifted = 0usize;
                for sub in &mut mesh.submeshes {
                    for v in &mut sub.vertices {
                        let n = v.normal;
                        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                        if len > 1e-6 {
                            for (axis, c) in n.iter().enumerate() {
                                v.position[axis] += c / len * WEARABLE_LIFT_M;
                            }
                            lifted += 1;
                        }
                    }
                }
                report.advisories.push(format!(
                    "{key}: {lifted} vertices lifted a FLAT {:.0} mm along their \
                     normals — the wearer's mesh did not resolve, so there was \
                     nothing to fit the garment over",
                    WEARABLE_LIFT_M * 1000.0
                ));
            }
        }
        if remapped == 0 {
            return Err(AssetError::Import(format!(
                "{key}: not one of the garment's {} influences named a bone the \
                 target rig has — the two rigs share no names, and the garment \
                 would collapse onto the root",
                dropped
            )));
        }
    } else {
        let Some(joint) = target.skeleton.index_of(&spec.joint) else {
            return Err(AssetError::Import(format!(
                "{key}: the target rig has no joint `{}` to bind this wearable \
                 to (it has {} joints)",
                spec.joint,
                target.skeleton.len()
            )));
        };
        for sub in &mut mesh.submeshes {
            sub.skin = vec![
                inf_mesh::VertexSkin {
                    joints: [joint, 0, 0, 0],
                    weights: [1.0, 0.0, 0.0, 0.0],
                };
                sub.vertices.len()
            ];
            remapped += sub.vertices.len();
        }
    }

    let root = project.root().to_path_buf();
    // **A ONE-SLOT WEARABLE BINDS ITS SLOT** (wave OUTFIT1 AUDIT, carried 164).
    // `bind_slots` writes a slot table only for a mesh with two or more, and a
    // groom's cards mesh has one — so the hair had no bound slot, no SECTION,
    // and therefore drew the surface the entity's `Material` component copied
    // when the level was authored: the committed default's tint, opaque, with
    // no coverage. The slot table is the only address at which a `.inf_mat`
    // reaches a draw, so this is what makes the atlas above visible at all.
    if mesh.material_slots.len() == 1 && mesh.material_slot_assets.iter().flatten().count() == 0 {
        if let Some(id) = manifest_mat {
            mesh.material_slot_assets = vec![Some(id)];
            report.advisories.push(format!(
                "{key}: its ONE material slot names `{}` and had no binding, so it drew \
                 the entity's own tint — bound to the manifest's material, which \
                 gives it a section and its own surface",
                mesh.material_slots[0]
            ));
        }
    }
    let mut deps = vec![want_skel];
    for id in mesh.material_slot_assets.iter().flatten() {
        if !deps.contains(id) {
            deps.push(*id);
        }
    }
    let path = root.join(format!("{stem}.inf_mesh"));
    project.write_asset_at_with_id(&path, &mesh, want_mesh, deps, None)?;
    report.rebinds.push((format!("{stem}.inf_mesh"), want_mesh));
    // The licence follows the bytes here for `rebind_character`'s own reason:
    // this is the asset the level references and the one that ships.
    report.asset_packs.push((want_mesh, pack.to_string()));

    // The fallback tint an instance wears when a section's own slot material does
    // not resolve — the mesh's DOMINANT slot, on `rebind_character`'s own
    // reasoning: taking slot 0 would put a beard's material over a whole outfit
    // the moment a section went missing.
    // The slot table where there is one; otherwise the mesh's own material
    // DEPENDENCY, which is what a rigid glTF import records instead (a groom's
    // cards have no slot table at all, and without this the hair drew the
    // committed default's tint rather than the groom's own).
    // …the MANIFEST's own answer next (a groom's cards name Unreal's
    // checkerboard in their slot table and the real material only in the
    // manifest), and the glTF's embedded dependency last.
    let fallback_mat = dominant_slot_material(&mesh).or(manifest_mat).or_else(|| {
        project
            .db()
            .get(source_mesh)
            .map(|e| e.sidecar.dependencies.clone())
            .unwrap_or_default()
            .into_iter()
            .find(|d| {
                project.db().get(*d).map(|e| e.kind()) == Some(inf_asset::AssetKind::Material)
            })
    });
    if let (Some(want_mat), Some(mat)) = (want_mat, fallback_mat) {
        if let Ok(payload) = project.load_payload::<MaterialAsset>(mat) {
            let mdeps = payload.texture_dependencies();
            let mpath = root.join(format!("{mat_stem}.inf_mat"));
            project.write_asset_at_with_id(&mpath, &payload, want_mat, mdeps, None)?;
            report
                .rebinds
                .push((format!("{mat_stem}.inf_mat"), want_mat));
            report.asset_packs.push((want_mat, pack.to_string()));
        }
    }
    report.advisories.push(format!(
        "{key}: REBOUND as the {} of the {} starter character ({} triangles, {} \
         influences re-pointed by name, {dropped} dropped, {fallback} vertices \
         pinned to `{}`). Local only.",
        if spec.hair { "HAIR" } else { "OUTFIT" },
        if spec.female { "female" } else { "male" },
        mesh.triangle_count(),
        remapped,
        spec.joint
    ));
    Ok(())
}

/// The material slot that owns the most triangles of `mesh`, or `None` for a
/// mesh whose slots name nothing.
///
/// `rebind_character`'s own rule, lifted so both callers read the same sentence:
/// slot 0 is not the dominant slot on a UDIM MetaHuman, and a fallback tint taken
/// from the wrong one is a head texture over a whole person.
fn dominant_slot_material(mesh: &inf_mesh::MeshAsset) -> Option<AssetId> {
    let mut tris: BTreeMap<u32, usize> = BTreeMap::new();
    for sub in &mesh.submeshes {
        *tris.entry(sub.material_slot.unwrap_or(0)).or_default() += sub.triangle_count();
    }
    let slot = tris
        .into_iter()
        .max_by_key(|(slot, n)| (*n, std::cmp::Reverse(*slot)))
        .map(|(slot, _)| slot as usize)?;
    mesh.material_slot_assets.get(slot).copied().flatten()
}

/// **Put a garment on a LEVEL's pawn** (wave OUTFIT1, carried item 137) — the
/// door that makes a cape a property of the world rather than of an environment
/// variable.
///
/// # Why this is an import verb and not a level edit somebody makes
///
/// The garment wave CHAR1b.2 authored lives in the island project's Content at a
/// fixed GUID and was worn only in a gate and through the demo loop's
/// `-WearCloth`, because the island's `.inf_lvl` is regenerated from its recipe
/// on **every** build — so an edit made in the editor is overwritten by the next
/// `inf island build`, and the level is not this repository's to commit either.
/// The whole shape is `--rebind-character`'s: the project is local, the edit is
/// re-applied by the same command sequence that rebuilt it, and nothing enters
/// the checkout.
///
/// Every `.inf_lvl` in the content root whose document has a player-controlled
/// pawn gets a `ClothSim` naming `cloth` on that pawn. A level with no pawn is
/// skipped with an advisory rather than refused: a project holds levels that are
/// not the showcase.
///
/// Returns how many levels were dressed.
pub fn wear_cloth_in_levels(
    project: &AssetProject,
    cloth: AssetId,
    report: &mut UeImportReport,
) -> usize {
    let mut worn = 0usize;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(project.root())
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "inf_lvl"))
        .collect();
    paths.sort();
    for path in paths {
        let mut doc = match crate::scene::serialize::load(&path) {
            Ok(d) => d,
            Err(e) => {
                report.advisories.push(format!(
                    "{}: not a level this build reads ({e})",
                    path.display()
                ));
                continue;
            }
        };
        let Some(pawn) = inf_ecs::movement::camera_subject(doc.world()) else {
            report.advisories.push(format!(
                "{}: no player-controlled pawn, so no garment was put on",
                path.display()
            ));
            continue;
        };
        let Some(e) = doc.world().entity_of(pawn) else {
            continue;
        };
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::ClothSim {
                asset: Some(cloth.0),
                enabled: true,
                ..Default::default()
            });
        doc.world_mut().mark_dirty();
        doc.world_mut().propagate();
        // The level's own GUID, read off the sidecar beside it, so the level
        // keeps its identity: `save` mints a fresh one for `None`, and a level
        // that changed GUID would orphan every reference the project holds to
        // it (and give the next scan a content-derived id that churns).
        let side = PathBuf::from(format!("{}.toml", path.display()));
        let guid = std::fs::read_to_string(&side)
            .ok()
            .and_then(|t| t.parse::<toml::Table>().ok())
            .and_then(|t| t.get("guid").and_then(|v| v.as_str()).map(str::to_string))
            .and_then(|g| g.parse::<uuid::Uuid>().ok());
        if guid.is_none() {
            report.advisories.push(format!(
                "{}: no sidecar GUID, so the level would be re-identified — skipped",
                path.display()
            ));
            continue;
        }
        match crate::scene::serialize::save(&doc, &path, guid) {
            Ok(_) => {
                worn += 1;
                report.advisories.push(format!(
                    "{}: the pawn is wearing {cloth}. Local only.",
                    path.display()
                ));
            }
            Err(e) => report.advisories.push(format!(
                "{}: the garment was not written ({e})",
                path.display()
            )),
        }
    }
    worn
}

/// **Re-retarget the clips a rebound identity already owns onto its NEW rig**
/// (wave CHAR1a.3).
///
/// # Why a rebind is not finished without it
///
/// A clip's coupling to a rig is POSITIONAL: `QuatTrack::joint` is a `u16` index
/// into `Pose::locals`, index-aligned to `Skeleton::joints`. So writing a
/// different skeleton at a character's committed GUID re-points every track in
/// its clips at a different bone — in range, so nothing refuses, and the
/// character animates with an arm where a spine should be. Wave CHAR1a found
/// exactly that picture ("one arm over its head and its legs splayed") when it
/// rebound a body without its clips, and fixed it by taking the clips from the
/// same PACK as the body.
///
/// That fix does not reach a pack with no clips of its own, which is what the
/// MetaHumans are: the assembly writes skeletal meshes and no `AnimSequence` at
/// all. Their hero would take a 342-joint rig and keep three clips indexed
/// against a 161-joint one.
///
/// So the clips are re-retargeted BY NAME, from the rig the identity wore to the
/// rig it now wears — `RetargetMap::shared_names` plus `CHAIN_INFILL`, the same
/// door every other clip in this bridge crosses. A joint the new rig has and the
/// old one did not (a MetaHuman body publishes the mannequin's 161 names among
/// its 342: twist chains, correctives, the face's neck) is simply untouched and
/// plays at its bind, which is the honest answer for a bone no source clip ever
/// moved.
///
/// A no-op when the rig did not change, and when there was no previous rig at
/// all.
fn retarget_committed_clips(
    project: &mut AssetProject,
    ids: &crate::character::CharacterIds,
    stems: (&str, &str, &str),
    previous: Option<&inf_anim::Skeleton>,
    target: &inf_anim::Skeleton,
    report: &mut UeImportReport,
) {
    // **THE RIG HASH IS RE-STAMPED WHATEVER HAPPENS TO THE TRACKS** (wave
    // CHAR1b.1). The two questions are different and this function used to
    // answer only one: "do the clips need re-retargeting" is about the rig's
    // JOINT NAMES, and "what rig do these clips animate" is about its BYTES.
    //
    // Found by `char1a3_gate::the_rebound_clips_record_the_rig_they_animate`
    // going red on this wave's own re-import: inferring a role table changes the
    // `.inf_skel`'s content hash and changes no joint name, so the early return
    // below fired, no clip was rewritten, and three sidecars went on naming a
    // hash that no longer exists — which is carried item 95's shape returning
    // through the very door it was fixed in.
    if let Some(table) = super::skeleton_binding::import_table(project, ids.skeleton) {
        for (stem, want) in [(stems.0, ids.idle), (stems.1, ids.walk), (stems.2, ids.run)] {
            let Some(path) = want
                .and_then(|w| project.db().get(w))
                .map(|e| e.path.clone())
            else {
                continue;
            };
            let Ok(mut side) = inf_asset::AssetSidecar::load(&path) else {
                continue;
            };
            let mut t = side.import.take().unwrap_or_default();
            for (k, v) in table.clone() {
                t.insert(k, v);
            }
            side.import = Some(t);
            if let Err(e) = side.save(&path) {
                report
                    .advisories
                    .push(format!("{stem}.inf_anim: rig hash not re-stamped ({e})"));
            }
        }
    }
    let Some(previous) = previous else { return };
    if previous.len() == target.len()
        && previous
            .joints()
            .iter()
            .zip(target.joints())
            .all(|(a, b)| a.name == b.name)
    {
        return;
    }
    // The body the clips are settled against: the mesh that was just written at
    // this identity's own GUID, read back off disk so the reader goes where the
    // runtime goes.
    let ground: Option<Vec<inf_anim::retarget::GroundVertex>> = ids
        .mesh
        .and_then(|id| project.load_payload::<inf_mesh::MeshAsset>(id).ok())
        .map(|m| {
            m.submeshes
                .iter()
                .filter(|s| s.is_skinned())
                .flat_map(|s| {
                    s.vertices
                        .iter()
                        .zip(s.skin.iter())
                        .map(|(v, k)| (v.position, k.joints, k.weights))
                })
                .collect()
        });
    let map = inf_anim::retarget::RetargetMap::shared_names(previous, target);
    let slots = [
        (format!("{}.inf_anim", stems.0), ids.idle),
        (format!("{}.inf_anim", stems.1), ids.walk),
        (format!("{}.inf_anim", stems.2), ids.run),
    ];
    for (file, want) in slots {
        let Some(want) = want else { continue };
        let Ok(mut payload) = project.load_payload::<inf_anim::AnimClipAsset>(want) else {
            continue;
        };
        let (clip, rep) =
            inf_anim::retarget::retarget_clip(&payload.clip, previous, target, &map, true);
        if rep.is_vacuous() {
            report.advisories.push(format!(
                "{file}: retarget onto the rebound rig produced NO tracks ({} source \
                 joints, none named on the {} of the new rig) — the clip is left as \
                 it was and will animate the wrong bones",
                previous.len(),
                target.len()
            ));
            continue;
        }
        payload.clip = clip;
        payload.skeleton = ids.skeleton.map(|s| *s.uuid().as_bytes());
        if let (Some(mesh), Some(_)) = (ground.as_ref(), ids.mesh) {
            let d = inf_anim::retarget::settle_to_ground_with_skin(&mut payload.clip, target, mesh);
            if d.abs() >= 0.001 {
                report.advisories.push(format!(
                    "{file}: settled {:.1} mm onto the rebound body's ground plane",
                    d * 1000.0
                ));
            }
        }
        let path = project.root().join(&file);
        let deps: Vec<AssetId> = ids.skeleton.into_iter().collect();
        let import = super::skeleton_binding::import_table(project, ids.skeleton);
        match project.write_asset_at_with_id(&path, &payload, want, deps, import) {
            Ok(_) => report.advisories.push(format!(
                "{file}: re-retargeted onto the rebound rig — {} of {} tracks kept, \
                 {} dropped ({} → {} joints)",
                rep.tracks_out,
                rep.tracks_in,
                rep.dropped.len(),
                previous.len(),
                target.len()
            )),
            Err(e) => report
                .advisories
                .push(format!("{file}: re-retarget not written ({e})")),
        }
    }
}

/// **The locomotion GRAPH, rebuilt from the ALS map over what this import
/// brought in** (wave CHAR1b.1, clause 2).
///
/// `inf_anim::als::LOCOMOTION_MAP` says which donor sequence fills which slot;
/// this resolves those names against the clips the manifest just imported and
/// writes the machine at the identity's own committed GUID. A clip is looked up
/// by its **manifest name** (`ALS_N_Walk_F`), which is the donor's asset name
/// and therefore the one thing the map can honestly hold: the GUIDs are this
/// project's and the file paths carry a pack prefix.
///
/// Everything about it is a **value**: a project with no ALS content gets a
/// graph with no states and an advisory saying so, and the committed three-state
/// machine is left exactly where it is. That is the CI case, and it is also what
/// happens on a re-import of a manifest that only carries meshes.
///
/// # Why it overwrites rather than writing a sibling
///
/// The level references the machine by GUID. A sibling would need the `.inf_lvl`
/// edited, which is a byte-locked committed file and a schema question; writing
/// at the same id is the same move `rebind_character` makes for the mesh, the
/// rig and the skin, and it is local-only for the same reason.
fn rebind_locomotion_graph(
    project: &mut AssetProject,
    manifest: &[Clip],
    imported: &[(String, AssetId, usize)],
    ids: &crate::character::CharacterIds,
    stem: &str,
    report: &mut UeImportReport,
) {
    let Some(want) = ids.machine else {
        return;
    };
    // name -> the asset this import wrote for it. `manifest` carries the name and
    // the key; `imported` carries the key and the id.
    let by_key: BTreeMap<&str, AssetId> = imported
        .iter()
        .map(|(k, id, _)| (k.as_str(), *id))
        .collect();
    let by_name: BTreeMap<&str, AssetId> = manifest
        .iter()
        .filter_map(|c| by_key.get(c.key.as_str()).map(|id| (c.name.as_str(), *id)))
        .collect();
    // **…and the clips this project already holds.**
    //
    // The MetaHuman rebind and the ALS clips are in **two manifests** — the
    // bodies come from `MHForge` and the sequences from the mannequin packs —
    // and neither run carries the other's assets. A resolver that only saw this
    // import's output would answer nothing on the run that rebinds the island's
    // hero, which is the run that matters. So the project's own `.inf_anim` set
    // is the fallback, matched on the file stem the clip importer writes
    // (`{pack}_{name}`) with the pack prefix allowed to differ: `ALS_N_Walk_F`
    // finds `ALS_Community_ALS_N_Walk_F` and nothing else, because the separator
    // is part of the needle.
    let existing: Vec<(String, AssetId)> = project
        .db()
        .by_kind(inf_asset::AssetKind::AnimClip)
        .map(|e| (e.name.clone(), e.sidecar.guid))
        .collect();
    let find = |name: &str| -> Option<AssetId> {
        if let Some(id) = by_name.get(name) {
            return Some(*id);
        }
        let tail = format!("_{name}");
        let mut hit = None;
        for (stem, id) in &existing {
            if stem == name || stem.ends_with(&tail) {
                // A second match is ambiguous and answering either would be a
                // guess; refuse and let the advisory name the slot.
                if hit.is_some() {
                    return None;
                }
                hit = Some(*id);
            }
        }
        hit
    };
    // **…AND ONTO THE RIG THAT WILL PLAY IT** (audit CHAR1b.1).
    //
    // A clip's coupling to a skeleton is POSITIONAL — `inf_anim::pose::sample_clip`
    // addresses `JointTrack::joint` as an index into the target's joint list and
    // has nothing to refuse a clip authored on another rig with. The ALS
    // sequences are exported and retargeted in the MANNEQUIN manifest, onto that
    // manifest's own 161-joint rig; the hero is a 342-joint MetaHuman written by
    // a different manifest. Binding the donor's clips straight into this
    // identity's graph therefore pointed every track past the twelfth joint at a
    // different bone: measured on the island, `ALS_N_Pose` has 71 tracks and
    // **59 of them landed on a bone of another name** — `lowerarm_l`'s rotation
    // on `upperarm_correctiveRoot_l`, `hand_l`'s on `pinky_01_l`, `upperarm_r`'s
    // on `ring_01_side_out_l` — so the hero's arms hung at 13.59°/44.90° instead
    // of the donor's symmetric 13.59°, its legs never left the bind pose at any
    // gait, and its fingers twisted up to 172°.
    //
    // This is `retarget_committed_clips`' rule (immediately above) applied to the
    // clips this door binds, and it is the third appearance of carried item 95:
    // the first was a pack's own clips after a rebind, the second a stale rig
    // hash, this one a new door that resolves clips it did not import.
    //
    // The retargeted copy is written **per identity** at a GUID derived from the
    // identity's machine and the donor's name, so a re-import overwrites its own
    // output (carried 94's rule) and the male and female bodies do not fight over
    // one file. A clip already on this rig is bound as it is.
    let target_rig: Option<inf_anim::Skeleton> = ids
        .skeleton
        .and_then(|id| project.load_payload::<inf_anim::SkeletonAsset>(id).ok())
        .map(|a| a.skeleton);
    let want_rig = ids.skeleton.map(|s| *s.uuid().as_bytes());
    let mut retargeted: BTreeMap<String, AssetId> = BTreeMap::new();
    let mut src_rigs: BTreeMap<[u8; 16], Option<inf_anim::Skeleton>> = BTreeMap::new();
    let names: Vec<&'static str> = inf_anim::als::LOCOMOTION_MAP
        .iter()
        .flat_map(|s| s.clips.iter().map(|(n, _)| *n))
        .collect();
    for name in names {
        let Some(id) = find(name) else {
            continue;
        };
        let Some(target) = target_rig.as_ref() else {
            // No rig to retarget onto: bind what is there and say so once.
            retargeted.insert(name.to_string(), id);
            continue;
        };
        let Ok(payload) = project.load_payload::<inf_anim::AnimClipAsset>(id) else {
            continue;
        };
        if payload.skeleton == want_rig {
            retargeted.insert(name.to_string(), id);
            continue;
        }
        let Some(src_id) = payload.skeleton else {
            report.advisories.push(format!(
                "{stem}: `{name}` names no skeleton, so it cannot be retargeted onto \
                 this body -- it is bound as it is and will animate whatever bone \
                 each track's index happens to be"
            ));
            retargeted.insert(name.to_string(), id);
            continue;
        };
        let src = src_rigs.entry(src_id).or_insert_with(|| {
            project
                .load_payload::<inf_anim::SkeletonAsset>(AssetId(uuid::Uuid::from_bytes(src_id)))
                .ok()
                .map(|a| a.skeleton)
        });
        let Some(src) = src.as_ref() else {
            report.advisories.push(format!(
                "{stem}: `{name}` is authored on a rig this project does not hold, so \
                 it cannot be retargeted -- the slot is left unbound"
            ));
            continue;
        };
        let map = inf_anim::retarget::RetargetMap::shared_names(src, target);
        let (clip, rep) = inf_anim::retarget::retarget_clip(&payload.clip, src, target, &map, true);
        if rep.is_vacuous() {
            report.advisories.push(format!(
                "{stem}: retargeting `{name}` onto this body produced NO tracks ({} \
                 source joints, none named on the {} of this rig) -- the slot is left \
                 unbound rather than bound to a clip that would pose a bind pose",
                src.len(),
                target.len()
            ));
            continue;
        }
        let out = inf_anim::AnimClipAsset::new(clip, want_rig);
        // **The file's own name may not look like the donor's** (audit
        // CHAR1b.1, measured the hard way). `find` above resolves a donor name
        // against the project's `.inf_anim` stems and REFUSES an ambiguous
        // match; a copy written as `ALS_N_Pose.inf_anim` is a second stem that
        // matches `ALS_N_Pose`, so the male identity's output made every one of
        // the female's sixty-five lookups ambiguous and her machine came out with
        // **0 states**. `{donor}--{stem}` matches neither `stem == name` nor
        // `stem.ends_with("_{name}")`, because the character before the donor's
        // name is a hyphen.
        let path = project
            .root()
            .join(format!("{stem}-loco"))
            .join(format!("{name}--{stem}.inf_anim"));
        let id_out = clip_guid(&format!("{stem}:loco:{name}"));
        let deps: Vec<AssetId> = ids.skeleton.into_iter().collect();
        let import = super::skeleton_binding::import_table(project, ids.skeleton);
        match project.write_asset_at_with_id(&path, &out, id_out, deps, import) {
            Ok(id_out) => {
                retargeted.insert(name.to_string(), id_out);
            }
            Err(e) => report
                .advisories
                .push(format!("{stem}: `{name}` was not re-retargeted ({e})")),
        }
    }
    // ── THE AUTHORED SETS (wave CHAR1b.2) ────────────────────────────────────
    //
    // ALS ships no slide, no throw, no swim, no prone and no standing get-up —
    // a census by name over its 164 sequences — and this engine's catalogue has
    // a MODE for four of them. `inf_anim::authored` derives them from the rig
    // that will play them, which is the same rule `crate::locomotion`'s
    // generated cycles follow and the reason there is nothing to retarget: a
    // clip authored against THIS identity's joint indices is already on it.
    //
    // Written beside the retargeted donor clips, under the same
    // `{name}--{stem}` spelling, for that naming rule's own reason: a copy
    // called `INF_Slide.inf_anim` would be a second stem that matches
    // `INF_Slide` and would make the other identity's lookup ambiguous.
    //
    // Derived after authoring, through the same `derive_clip` door every other
    // clip in this file goes through, so the authored sets carry the six ALS
    // channels and a foot-IK gate exactly as the imported ones do.
    let rig_asset = ids
        .skeleton
        .and_then(|id| project.load_payload::<inf_anim::SkeletonAsset>(id).ok());
    if let Some(rig_asset) = rig_asset {
        match inf_anim::author_clips(&rig_asset) {
            Ok(set) => {
                for (name, clip) in set {
                    let clip = match inf_anim::derive_clip(
                        &clip,
                        &rig_asset,
                        &inf_anim::DeriveOptions::default(),
                    ) {
                        Ok((c, _)) => c,
                        Err(e) => {
                            report.advisories.push(format!(
                                "{stem}: `{name}` was authored but not derived ({e}) -- it is \
                                 bound as it is and carries no curve channels"
                            ));
                            clip
                        }
                    };
                    let out = inf_anim::AnimClipAsset::new(clip, want_rig);
                    let path = project
                        .root()
                        .join(format!("{stem}-loco"))
                        .join(format!("{name}--{stem}.inf_anim"));
                    let id_out = clip_guid(&format!("{stem}:loco:{name}"));
                    let deps: Vec<AssetId> = ids.skeleton.into_iter().collect();
                    let import = super::skeleton_binding::import_table(project, ids.skeleton);
                    match project.write_asset_at_with_id(&path, &out, id_out, deps, import) {
                        Ok(id_out) => {
                            retargeted.insert(name.clone(), id_out);
                        }
                        Err(e) => report
                            .advisories
                            .push(format!("{stem}: `{name}` was not written ({e})")),
                    }
                }
            }
            Err(e) => report.advisories.push(format!(
                "{stem}: the authored sets were not derived ({e}) -- slide, throwing, \
                 swimming, prone and the standing get-up have no clip on this body"
            )),
        }
    }

    let find = |name: &str| -> Option<AssetId> { retargeted.get(name).copied() };
    let (machine, bind) = inf_anim::als::build_locomotion_graph(&|name: &str| {
        find(name).map(|id| *id.uuid().as_bytes())
    });
    if machine.states.is_empty() {
        report.advisories.push(format!(
            "{stem}: no ALS locomotion clip resolved -- the committed machine is \
             left as it is ({})",
            bind.summary()
        ));
        return;
    }
    if let Err(e) = machine.validate() {
        report
            .advisories
            .push(format!("{stem}: the built graph does not validate ({e})"));
        return;
    }
    // The clips the graph names ARE its dependencies, so a cook's closure packs
    // them. `bind_slots`' lesson at `ue_import.rs`'s section 2b, one asset kind
    // over: a reference the sidecar does not carry is a reference the cook
    // cannot see.
    let mut deps: Vec<AssetId> = ids.skeleton.into_iter().collect();
    for (_, name) in &bind.bound {
        if let Some(id) = find(name) {
            if !deps.contains(&id) {
                deps.push(id);
            }
        }
    }
    let states = machine.states.len();
    let edges = machine.transitions.len();
    let asset =
        inf_anim::StateMachineAsset::new(machine, ids.skeleton.map(|s| *s.uuid().as_bytes()));
    let path = project.root().join(format!("{stem}.inf_sm"));
    let import = super::skeleton_binding::import_table(project, ids.skeleton);
    match project.write_asset_at_with_id(&path, &asset, want, deps, import) {
        Ok(_) => {
            report.rebinds.push((format!("{stem}.inf_sm"), want));
            report.advisories.push(format!(
                "{stem}: the ALS locomotion graph -- {states} states, {edges} \
                 transitions, {}",
                bind.summary()
            ));
            for (state, clip) in &bind.unbound {
                report.advisories.push(format!(
                    "{stem}: {state} wanted `{clip}`, which this manifest does not carry"
                ));
            }
        }
        Err(e) => report
            .advisories
            .push(format!("{stem}.inf_sm: the graph was not written ({e})")),
    }
}

/// **The three clips the hero's state machine names, from the rebound body's own
/// pack.**
///
/// The mapping is a stated table rather than a heuristic, because "which clip is
/// the idle" is a decision:
///
/// | starter slot | mannequin clip | why |
/// |---|---|---|
/// | idle | `MM_Idle` / `MF_Idle` | the only idle either mannequin ships |
/// | walk | `MM_Walk_Fwd` / `MF_Walk_Fwd` | forward, root-motion, not the in-place variant |
/// | run  | `MM_Run_Fwd` / `MF_Run_Fwd` | ditto |
///
/// `MM_Walk_InPlace` is deliberately NOT the walk: this engine's locomotion
/// machine drives the body from the movement component and reads the clip for
/// the pose, so an in-place walk would slide the feet at exactly the speed the
/// character travels.
///
/// A slot with no matching clip keeps the committed generated one and is named
/// in an advisory — which is the honest failure, because the alternative is a
/// hero whose walk is somebody else's idle.
fn rebind_character_clips(
    project: &mut AssetProject,
    manifest: &[Clip],
    imported: &[(String, AssetId, usize)],
    ids: &crate::character::CharacterIds,
    stems: (&str, &str, &str),
    // **Which mannequin's clips this identity prefers** (wave CHAR1a.3). The
    // table below names two candidates per slot and `find` takes the FIRST the
    // manifest carries, which is sorted by object path -- so Manny sorts before
    // Quinn and BOTH identities took `MM_Idle`. A male default and a female
    // default that walk identically are one default twice.
    prefer_female: bool,
    report: &mut UeImportReport,
) -> Result<()> {
    // The body and the rig the hero was just rebound to, for the ground settle
    // below. Read back off disk rather than threaded through from
    // `rebind_character`: they were written at fixed GUIDs a moment ago, and a
    // reader that goes to the same place the runtime will is a reader that
    // cannot disagree with it.
    let rig: Option<inf_anim::Skeleton> = ids
        .skeleton
        .and_then(|id| project.load_payload::<inf_anim::SkeletonAsset>(id).ok())
        .map(|a| a.skeleton);
    let ground: Option<Vec<inf_anim::retarget::GroundVertex>> = ids
        .mesh
        .and_then(|id| project.load_payload::<inf_mesh::MeshAsset>(id).ok())
        .map(|m| {
            m.submeshes
                .iter()
                .filter(|s| s.is_skinned())
                .flat_map(|s| {
                    s.vertices
                        .iter()
                        .zip(s.skin.iter())
                        .map(|(v, k)| (v.position, k.joints, k.weights))
                })
                .collect()
        });
    let order = |m: &'static str, f: &'static str| -> [&'static str; 2] {
        if prefer_female {
            [f, m]
        } else {
            [m, f]
        }
    };
    let slots: [(&str, Option<AssetId>, [&str; 2]); 3] = [
        (stems.0, ids.idle, order("MM_Idle", "MF_Idle")),
        (stems.1, ids.walk, order("MM_Walk_Fwd", "MF_Walk_Fwd")),
        (stems.2, ids.run, order("MM_Run_Fwd", "MF_Run_Fwd")),
    ];
    let by_key: BTreeMap<&str, AssetId> = imported
        .iter()
        .map(|(k, id, _)| (k.as_str(), *id))
        .collect();
    for (stem, want, names) in slots {
        let Some(want) = want else { continue };
        // The PREFERENCE is honoured before the manifest's order: the first
        // name that exists wins, not the first record that matches either name.
        let found = names
            .iter()
            .find_map(|want| manifest.iter().find(|c| c.name == *want))
            .and_then(|c| by_key.get(c.key.as_str()).copied());
        let Some(found) = found else {
            report.advisories.push(format!(
                "{stem}: the rebound body's pack ships none of {names:?}, so the \
                 hero keeps the generated clip -- which was authored against a \
                 rig whose bind pose is the identity and will look wrong on this \
                 body"
            ));
            continue;
        };
        let mut payload: inf_anim::AnimClipAsset = project.load_payload(found)?;
        // **SETTLED AGAINST THIS BODY** (wave CHAR1a.2). `retarget_clip` already
        // settled the clip against the target RIG, and that pins the lowest
        // ball/foot joint — but a foot rotated at toe-off lifts its sole while
        // its joint stays put, so the mannequin's run still dipped **17.8 mm**
        // into the road after it. Here the rebound body is on disk, so the
        // question can be asked of the mesh: the lowest skinned vertex over the
        // cycle, the same arithmetic the shader runs. Measured after: the hero's
        // idle plants within 1.6 mm and its run within 1.9 mm.
        if let (Some(sk), Some(mesh)) = (rig.as_ref(), ground.as_ref()) {
            let d = inf_anim::retarget::settle_to_ground_with_skin(&mut payload.clip, sk, mesh);
            if d.abs() >= 0.001 {
                report.advisories.push(format!(
                    "{stem}: settled {:.1} mm onto the rebound body's ground plane",
                    d * 1000.0
                ));
            }
        }
        let deps: Vec<AssetId> = project
            .db()
            .get(found)
            .map(|e| e.sidecar.dependencies.clone())
            .unwrap_or_default();
        let path = project.root().join(format!("{stem}.inf_anim"));
        // **AND THE SKELETON HASH MOVES WITH THE RIG** (carried item 95). The
        // rebind writes a clip authored against the PACK's rig at the starter
        // character's GUID, over a rig that has just been replaced by that same
        // pack's — so the clip and the rig ARE authored together, and the sidecar
        // said otherwise: the island's four rebound assets recorded `8d06c1ee…`
        // while `Starter.inf_skel` hashed `5c7c1647…`, and the editor's content
        // scan printed *"the character animates the wrong bones"* on every boot
        // for content that was in fact correct. A false positive that will hide a
        // true one — so the table is rebuilt here, against the rig on disk.
        let import = super::skeleton_binding::import_table(project, ids.skeleton);
        project.write_asset_at_with_id(&path, &payload, want, deps, import)?;
        report.rebinds.push((format!("{stem}.inf_anim"), want));
    }
    Ok(())
}

/// The rungs of `lods` worth storing: at most `keep`, each strictly coarser
/// than the one before.
///
/// # Why "the first `keep` rungs" is the wrong rule
///
/// **Measured on `SKM_Manny`**: its four exported rungs are 92 178, 92 178,
/// 26 998 and 12 998 triangles — LOD 1 is a *copy* of LOD 0. Taking the first
/// three would have stored 92 178 triangles twice, shipped 4.1 MB of duplicate
/// `.inf_mesh` per body, and given the ladder a switch at 32 m that changes
/// nothing on screen while costing a mesh swap. Selecting by the triangle count
/// the exporter measured gives 92 178 / 26 998 / 12 998 — a real 3.4:1 and
/// 2.1:1 ladder — and the rule is a property of the content rather than of the
/// pack's LOD-settings asset.
///
/// A rung with an unknown (zero) triangle count is kept if it is the first, and
/// otherwise skipped: an unmeasured rung cannot be shown to be coarser than the
/// one before it, and a ladder built on a guess is worse than a short one.
fn distinct_rungs(lods: &[SkelLod], keep: usize) -> Vec<&SkelLod> {
    let mut out: Vec<&SkelLod> = Vec::new();
    let mut last = u32::MAX;
    for lod in lods {
        if out.len() >= keep {
            break;
        }
        if out.is_empty() {
            out.push(lod);
            last = lod.triangles;
            continue;
        }
        if lod.triangles > 0 && lod.triangles < last {
            out.push(lod);
            last = lod.triangles;
        }
    }
    out
}

/// The distances, in metres, at which each stored character rung takes over.
///
/// One entry per rung; entry `i` is the distance at which rung `i` starts being
/// drawn, so entry 0 is always 0. They are `inf_ecs::crowd`'s own tier radii,
/// which is the point: the geometry a character draws and the simulation it
/// runs change at the same distance, so a body cannot be posed by the full
/// animation graph while drawing its cheapest mesh, or the reverse.
pub fn character_lod_switch_m(rungs: usize) -> Vec<f64> {
    let bands = [
        0.0,
        inf_ecs::crowd::DEFAULT_CROWD_FULL_M,
        inf_ecs::crowd::DEFAULT_CROWD_NEAR_M,
        inf_ecs::crowd::DEFAULT_CROWD_FAR_M,
    ];
    (0..rungs)
        .map(|i| {
            bands
                .get(i)
                .copied()
                .unwrap_or(inf_ecs::crowd::DEFAULT_CROWD_FAR_M)
        })
        .collect()
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
    // Roles the manifest states and this engine has nowhere to put. Collected
    // and REPORTED rather than skipped in silence (ASSET0 audit): "the material
    // imported" and "the material imported with half its maps" looked identical
    // in the report, and one of the silent roles was clobbering the albedo.
    let mut unplaced: Vec<&str> = Vec::new();
    for (role, key) in &mat.maps {
        // `hair_coverage` has no `MapKind` and is not unplaced: it is composited
        // into the albedo's ALPHA below, which is where this engine keeps a
        // masked material's cut-out (wave OUTFIT1 AUDIT).
        if role == "hair_coverage" {
            continue;
        }
        let targets = role_to_planes(role);
        if targets.is_empty() {
            unplaced.push(role.as_str());
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
        // One source texture may fill more than one slot (a packed UE mask), and
        // a slot a role already claimed is NOT overwritten — a dedicated
        // `roughness` map beats the roughness channel of a packed mask, and the
        // walk is `BTreeMap`-ordered so which one that is is a property of the
        // manifest rather than of an iteration.
        for (kind, channel) in targets {
            if planes.contains_key(kind) {
                continue;
            }
            let plane = match channel {
                Some(c) => broadcast_channel(&rgba, *c),
                None => rgba.clone(),
            };
            planes.insert(*kind, (plane, w, h));
        }
    }
    if !unplaced.is_empty() {
        report.advisories.push(format!(
            "{}: the manifest names {} and this engine's `.inf_mat` has no slot \
             for {} — not imported. The material keeps its scalar values for \
             those channels.",
            mat.key,
            unplaced.join(", "),
            if unplaced.len() == 1 { "it" } else { "them" }
        ));
    }

    // **THE CARD ALPHA** (wave OUTFIT1 AUDIT, carried item 164) and **THE
    // EYEBALL** (carried item 167) — two roles whose home is the albedo's own
    // channels rather than a slot of their own.
    let mut cutoff_override: Option<f32> = None;
    if let Some((cov, cw, ch)) = load_map(base, &mat.maps, "hair_coverage", by_key, opts, report)? {
        // The coverage is channel R: measured on both grooms' atlases, R is the
        // strand mask on a pure black ground (81.6 % / 85.4 % of the atlas is
        // exactly zero there) while G and B are the attribute fields, which are
        // filled everywhere — 99.4 % of G is non-zero on Dominic's, so a bridge
        // that took G would draw the same solid ribbons with an extra texture.
        let alpha: Vec<u8> = cov.chunks_exact(4).map(|p| p[0]).collect();
        match planes.get_mut(&MapKind::Albedo) {
            // A groom's cards material has NO albedo (its colour is the shader's
            // melanin, which the bridge already reads into `base_color`), so the
            // usual case is the second one: a WHITE plane carrying the coverage
            // in its alpha, which multiplies the melanin tint by one and the
            // cut-out by itself.
            Some((px, w, h)) if *w == cw && *h == ch => {
                for (i, p) in px.chunks_exact_mut(4).enumerate() {
                    p[3] = alpha[i];
                }
            }
            _ => {
                let mut px = vec![255u8; cov.len()];
                for (i, p) in px.chunks_exact_mut(4).enumerate() {
                    p[3] = alpha[i];
                }
                planes.insert(MapKind::Albedo, (px, cw, ch));
            }
        }
        cutoff_override = Some(HAIR_CARD_CUTOFF);
        let open = alpha
            .iter()
            .filter(|a| **a as f32 / 255.0 >= HAIR_CARD_CUTOFF)
            .count();
        report.advisories.push(format!(
            "{}: the groom's own cards atlas is its ALPHA — {:.1} % of the atlas \
             is a strand at a {:.3} cutoff, and the rest is a hole. Before this \
             the material named no coverage at all and every card drew as a \
             solid ribbon.",
            mat.key,
            100.0 * open as f32 / alpha.len().max(1) as f32,
            HAIR_CARD_CUTOFF
        ));
    }
    if let Some(iris_key) = mat.maps.get("albedo").filter(|k| k.contains("EyeIris")) {
        let sclera_key = iris_key.replace("EyeIris", "EyeSclera");
        match (
            planes.remove(&MapKind::Albedo),
            by_key.get(sclera_key.as_str()),
        ) {
            (Some((iris, w, h)), Some(tex)) => {
                let sclera = tex
                    .file
                    .as_ref()
                    .map(|f| base.join(f))
                    .and_then(|p| std::fs::read(p).ok())
                    .and_then(|b| inf_material::decode_image_rgba8(&b).ok());
                match sclera {
                    Some((sc, sw, sh)) => {
                        let (px, r) = composite_eyeball(&iris, w, h, &sc, sw, sh);
                        report.advisories.push(format!(
                            "{}: the eyeball's albedo is its SCLERA with its IRIS \
                             composited into the disc the sclera's own limbus \
                             marks (r = {r:.3} of the uv square, derived from the \
                             sclera's radial luminance). The manifest binds only \
                             the iris, which alone would draw a whole eyeball the \
                             colour of an iris.",
                            mat.key
                        ));
                        planes.insert(MapKind::Albedo, (px, w, h));
                    }
                    None => {
                        planes.insert(MapKind::Albedo, (iris, w, h));
                    }
                }
            }
            (Some(a), _) => {
                planes.insert(MapKind::Albedo, a);
            }
            _ => {}
        }
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
        //
        // **The packed ORM is the exception** (wave OUTFIT1, carried item 105).
        // It is written under `MapKind::Roughness`, whose table entry is the
        // uncompressed data preset — correct for a lone roughness map, and
        // 25 758 080 bytes for a 2048² triple of shading terms that a lighting
        // multiply reads. It takes `TextureImportSettings::orm()` instead, which
        // is the same linear, mipped import at BC7.
        let settings: TextureImportSettings = if slot == "ORM" {
            TextureImportSettings::orm()
        } else {
            kind.settings(false)
        };
        let image = inf_material::build_tiled_texture(rgba, w, h, settings)
            .map_err(|e| AssetError::Import(format!("{}_{slot}: {e}", mat.key)))?;
        // **A DETERMINISTIC PATH**, so a second run over an unchanged manifest
        // overwrites its own output instead of writing `X_1.inf_tex` beside it.
        // Measured before this line existed: the second import wrote 106
        // duplicate assets and doubled the project's texture bytes.
        let id = project.write_tiled_texture_at(
            &dest.join(format!("{name}_{slot}.inf_tex")),
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
        alpha_cutoff: cutoff_override.unwrap_or(0.5),
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
        // …and the material likewise, for the same reason and by the same door.
        None => project.write_asset_at(&dest.join(format!("{name}.inf_mat")), &asset, deps, None),
    }
}

/// The ground-library kind a rebind stem names.
fn stem_kind(stem: &str) -> Option<inf_material::ground::GroundKind> {
    inf_material::ground::GroundKind::ALL
        .into_iter()
        .find(|k| k.stem() == stem)
}

/// A manifest role name → the engine's [`MapKind`]s, with a channel each.
///
/// **Exactly the roles this engine has somewhere to put**, and nothing else:
/// albedo and normal get their own slot, and occlusion/roughness/metallic are
/// packed into the one ORM. A role absent from this table is reported by
/// `import_material` rather than dropped in silence.
///
/// `displacement` is deliberately absent: this engine has no displacement slot
/// on `.inf_mat`, and importing a height map as a texture nothing samples would
/// be 2 MB an asset for a channel no shader reads.
///
/// # Two that were here and should not have been (ASSET0 audit)
///
/// `"emissive" => MapKind::Albedo` was a **silent clobber**. `planes` is keyed
/// by `MapKind` and `mat.maps` is a `BTreeMap`, so on any material carrying both
/// maps `"emissive"` sorts after `"albedo"` and *replaced* it — the material
/// would have shipped its glow map as its base colour. Nothing in the reference
/// project's thirty materials names an emissive map, so the defect was
/// unreachable today and one export-list edit from being reached. There is no
/// emissive texture slot on `.inf_mat`; the scalar `emissive` the manifest
/// states is what crosses.
///
/// `"opacity" => MapKind::Opacity` was a **silent drop with a bill attached**:
/// the plane was decoded and clamped — an 8 K source is 268 MB of RGBA — and
/// then never read, because only Albedo, Normal and the ORM trio are consumed.
/// The engine carries alpha in the base colour's own channel, so an opacity map
/// would have to be composited into the albedo to mean anything; until it is,
/// the honest answer is to say so and not pay for the decode.
///
/// # A packed UE mask, unpacked into the slots this engine has (wave CHAR1a.2)
///
/// Some roles are not one map: `msr` is Unreal's `T_*_MSR_MSK`, which carries
/// **metallic in R, specular in G and roughness in B** — the name is the spec and
/// a channel census of `T_Manny_01_MSR_MSK` confirms it (R bimodal 0/255 = a
/// metal mask; G flat at 92–118 = the 0.5 specular constant; B 0/116/255 = the
/// roughness). This engine's ORM is occlusion/roughness/metallic, so it is a
/// SWIZZLE and not a rename, which is exactly what CHAR1a carried as item 76.
///
/// `aniso_ao_paint` is `T_*_AS?AO?MASK_MSK`: anisotropy in R, **ambient
/// occlusion in G**, a paint mask in B. The importer used to take plane R,
/// whose mean over the mannequin is **3.5 of 255** — so every character imported
/// through this bridge had its ambient term multiplied by 0.014. That is not a
/// subtle wrong: it is the body reading almost unlit in shade.
///
/// `Some(channel)` means "broadcast that channel of the decoded RGBA into a grey
/// plane"; `pack_orm` reads channel 0 of whatever it is handed, so the broadcast
/// is what makes one source texture fill two different ORM channels.
///
/// Returns an EMPTY slice for a role this engine has nowhere to put, which the
/// caller reports as unplaced — `tangent`, `normal_second`, `decal` and
/// `clearcoat` are all real maps with no home here, and saying so is the ASSET0
/// audit's rule. Wave CHAR1a.3 adds four more of them from the MetaHuman
/// materials: `scatter` (the subsurface amount — PAR5's, and this engine has no
/// SSS term to feed), `detail_mask` (a 32-pixel micro-tiling mask; the engine's
/// own detail slot is a whole texture with a tiling rate, not a mask), and
/// `animated_delta` (the facial rig's per-curve basecolor/normal deltas, which
/// need the 875-joint face rig FACE1 will build). Each is a real map this bridge
/// carries across and this engine cannot yet spend, and the import log says so
/// per material.
pub fn role_to_planes(role: &str) -> &'static [(MapKind, Option<usize>)] {
    match role {
        "albedo" => &[(MapKind::Albedo, None)],
        "normal" => &[(MapKind::Normal, None)],
        "roughness" => &[(MapKind::Roughness, None)],
        "metallic" => &[(MapKind::Metallic, None)],
        "ao" => &[(MapKind::Occlusion, None)],
        "msr" => &[(MapKind::Metallic, Some(0)), (MapKind::Roughness, Some(2))],
        "aniso_ao_paint" => &[(MapKind::Occlusion, Some(1))],
        // **The MetaHuman mask** (wave CHAR1a.3). `T_*_SRMF` is
        // Specular / Roughness / Metallic / Fuzz and `T_Teeth_SRM` the same
        // three without the fourth -- the name is the spec, and a channel census
        // of three of them agrees: `T_Body_SRMF` R median 150 (the specular
        // constant), G median **187** (the roughness), B **exactly 0 at every
        // percentile including the max** (skin is not a metal), A median 200.
        // So roughness is G and metallic is B, which is a different swizzle from
        // `msr`'s and the reason both are in this table by name rather than one
        // "packed mask" rule that would have to guess.
        //
        // The specular plane has nowhere to go: this engine's PBR is
        // metallic-roughness and reads no specular map. It is dropped in silence
        // here rather than reported, because unlike `tangent` or `clearcoat` it
        // is a CHANNEL of a texture that IS imported -- the advisory would say
        // "srmf is unplaced" about a map two thirds of which just landed.
        "srmf" => &[(MapKind::Roughness, Some(1)), (MapKind::Metallic, Some(2))],
        _ => &[],
    }
}

/// **The threshold a hair card's coverage is cut at** — UE's own default
/// `OpacityMaskClipValue` for a `BLEND_Masked` material (wave OUTFIT1 AUDIT).
///
/// The atlas is nearly binary — 81.6 % of Dominic's coverage channel is exactly
/// zero and 10.8 % is above 224 — so the number between them barely moves the
/// silhouette; it is UE's rather than invented so that a card cut here is the
/// card Unreal draws.
pub const HAIR_CARD_CUTOFF: f32 = 0.333;

/// Decode the texture a manifest material binds to `role`, or `None`.
///
/// The plane loop above cannot be reused for a role with no [`MapKind`]: these
/// are roles whose home is a CHANNEL of another map, and they are read here so
/// that the loop's "a role with no slot is reported unplaced" rule keeps meaning
/// what it says.
fn load_map(
    base: &Path,
    maps: &BTreeMap<String, String>,
    role: &str,
    by_key: &BTreeMap<&str, &Texture>,
    opts: &UeImportOptions,
    report: &mut UeImportReport,
) -> Result<Option<(Vec<u8>, u32, u32)>> {
    let Some(key) = maps.get(role) else {
        return Ok(None);
    };
    let Some(tex) = by_key.get(key.as_str()) else {
        report
            .advisories
            .push(format!("{role}: no texture record for {key}"));
        return Ok(None);
    };
    let Some(file) = tex.file.as_ref() else {
        return Ok(None);
    };
    let path = base.join(file);
    let Ok(bytes) = std::fs::read(&path) else {
        report
            .advisories
            .push(format!("{role}: {} is not on disk", path.display()));
        return Ok(None);
    };
    let (rgba, w, h) = inf_material::decode_image_rgba8(&bytes)
        .map_err(|e| AssetError::Import(format!("{}: {e}", path.display())))?;
    let (rgba, w, h) = inf_material::downscale_rgba8(rgba, w, h, opts.max_texture)
        .map_err(|e| AssetError::Import(format!("{}: {e}", path.display())))?;
    Ok(Some((rgba, w, h)))
}

/// **One eyeball albedo out of the two textures a MetaHuman eye is painted
/// with** (wave OUTFIT1 AUDIT, carried item 167).
///
/// UE's eyeball shader samples the SCLERA over the whole uv square and refracts
/// the IRIS into the middle of it. This bridge carries one base-colour slot, so
/// the two are composited once, at import, and the radius they meet at is
/// DERIVED from the sclera itself rather than typed in: the sclera's own limbus
/// is where its radial luminance leaves the flat plateau under the iris —
/// measured at r ≈ 0.30 of the uv square's half-width on both characters, with
/// the plateau at 98 and the ring above it at 108.
///
/// Returns the composited plane and the radius it used, so the caller can say
/// which number this eye was built with.
fn composite_eyeball(
    iris: &[u8],
    w: u32,
    h: u32,
    sclera: &[u8],
    sw: u32,
    sh: u32,
) -> (Vec<u8>, f32) {
    let sample = |px: &[u8], pw: u32, ph: u32, u: f32, v: f32| -> [u8; 4] {
        let x = ((u.clamp(0.0, 1.0) * (pw - 1) as f32) as usize).min(pw as usize - 1);
        let y = ((v.clamp(0.0, 1.0) * (ph - 1) as f32) as usize).min(ph as usize - 1);
        let i = (y * pw as usize + x) * 4;
        [px[i], px[i + 1], px[i + 2], px[i + 3]]
    };
    // The limbus, off the sclera's own radial luminance.
    let lum = |c: [u8; 4]| 0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32;
    let ring = |r: f32| -> f32 {
        let mut sum = 0.0f32;
        let n = 256;
        for k in 0..n {
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            // `inf_math`'s portable pair and not `f32::cos`/`f32::sin` (the P14
            // law): this ring decides `iris_r`, `iris_r` decides every texel of
            // the eyeball albedo, and that texture is an ASSET a cook packs. A
            // std transcendental is not bit-portable, so two machines importing
            // one manifest would write two different eyes.
            sum += lum(sample(
                sclera,
                sw,
                sh,
                0.5 + 0.5 * r * inf_math::pcos64(a as f64) as f32,
                0.5 + 0.5 * r * inf_math::psin64(a as f64) as f32,
            ));
        }
        sum / n as f32
    };
    let plateau = (ring(0.02) + ring(0.06) + ring(0.10)) / 3.0;
    let mut iris_r = 0.30f32;
    let mut r = 0.12f32;
    while r < 0.70 {
        if ring(r) > plateau * 1.05 {
            iris_r = r;
            break;
        }
        r += 0.01;
    }
    let feather = 0.02f32;
    let mut out = vec![255u8; (w as usize) * (h as usize) * 4];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let u = x as f32 / (w - 1).max(1) as f32;
            let v = y as f32 / (h - 1).max(1) as f32;
            let d = (((u - 0.5) * 2.0).powi(2) + ((v - 0.5) * 2.0).powi(2)).sqrt();
            let sc = sample(sclera, sw, sh, u, v);
            let ir = sample(
                iris,
                w,
                h,
                0.5 + (u - 0.5) / iris_r.max(1e-3),
                0.5 + (v - 0.5) / iris_r.max(1e-3),
            );
            let t = ((d - (iris_r - feather)) / feather).clamp(0.0, 1.0);
            let i = (y * w as usize + x) * 4;
            for c in 0..3 {
                out[i + c] = (ir[c] as f32 * (1.0 - t) + sc[c] as f32 * t).round() as u8;
            }
            out[i + 3] = 255;
        }
    }
    (out, iris_r)
}

/// One channel of an RGBA plane, broadcast into a fresh grey RGBA plane.
pub fn broadcast_channel(rgba: &[u8], channel: usize) -> Vec<u8> {
    let mut out = vec![255u8; rgba.len()];
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let v = px[channel];
        out[i * 4] = v;
        out[i * 4 + 1] = v;
        out[i * 4 + 2] = v;
    }
    out
}

/// A readable asset name out of a manifest key.
///
/// The keys are full object paths with the separators flattened and are up to
/// 150 characters long; a Content Drawer full of those is unusable. The last two
/// path-ish segments are what a human recognises.
///
/// # The digest, and the silent overwrite it closes (wave CHAR1a audit)
///
/// The tail alone is **not unique**, and the first content that proved it was
/// the pair of MetaHumans wave CHAR1a.2 imported. Their material keys are
///
/// ```text
/// INF_Built_INF_Dominic_Body_Materials_MI_Body_Baked_MI_Body_Baked
/// INF_Built_INF_Vivian_Body_Materials_MI_Body_Baked_MI_Body_Baked
/// ```
///
/// — identical in their last six segments, because the character's name sits at
/// index 3 and the tail always drops it. The texture writer's path is
/// deliberately DETERMINISTIC (`dest.join(format!("{name}_{slot}.inf_tex"))`, so
/// a re-import overwrites its own output instead of writing `X_1.inf_tex`), so
/// the second character's maps overwrote the first's. Measured on the wave's own
/// import: 32 textures were written and **16 files** exist on disk, every
/// surviving sidecar recording a Vivian source and not one recording Dominic's.
/// Nothing raised an advisory, because from each material's point of view the
/// write succeeded.
///
/// So the name carries a four-hex-digit digest of the WHOLE key. It is still
/// deterministic — the same key gives the same name for ever, which is what the
/// overwrite rule needs — and two keys that differ anywhere now differ here.
fn short_name(key: &str) -> String {
    let parts: Vec<&str> = key.split('_').collect();
    let tail = if parts.len() > 6 {
        parts[parts.len() - 6..].join("_")
    } else {
        key.to_string()
    };
    let clean: String = tail
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    // FNV-1a over the full key, 16 bits of it. Written out rather than pulled
    // from a hasher crate: this is a NAME, and a name whose bytes depend on a
    // dependency's default hasher is a name that can move under a `cargo update`.
    let mut h: u32 = 0x811c_9dc5;
    for b in key.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    format!("{clean}_{:04x}", (h ^ (h >> 16)) & 0xffff)
}

impl Material {
    /// What the sidecar records as this asset's source. Not a path — the source
    /// is in another engine's content tree and re-importing it needs the whole
    /// bridge — so it is the UE object path, which is what a human would look up.
    fn source_note(&self) -> String {
        format!("ue:{}", self.key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **TWO CHARACTERS' MATERIALS ARE TWO NAMES** (wave CHAR1a audit).
    ///
    /// The two MetaHuman body materials wave CHAR1a.2 imported differ only in
    /// their fourth underscore segment, which `short_name`'s six-segment tail
    /// always drops — so both resolved to `MI_Body_Baked_MI_Body_Baked`, and the
    /// texture writer's deterministic path (which exists so a re-import
    /// overwrites its own output) made the second character's maps overwrite the
    /// first's. Measured on the wave's own import: 32 textures written, **16
    /// files** on disk, every survivor recording a Vivian source.
    ///
    /// **The mutation**: drop the digest from `short_name`. `a == b` and the
    /// first assertion fails with both names printed. Verified.
    #[test]
    fn two_characters_materials_do_not_shorten_to_one_name() {
        let a = short_name("INF_Built_INF_Dominic_Body_Materials_MI_Body_Baked_MI_Body_Baked");
        let b = short_name("INF_Built_INF_Vivian_Body_Materials_MI_Body_Baked_MI_Body_Baked");
        assert_ne!(
            a, b,
            "two characters' body materials shorten to one asset name, so the \
             second import silently overwrites the first's textures"
        );
        // …and the readable part survives, because a Content Drawer full of
        // digests is the problem this function exists to avoid.
        assert!(a.starts_with("MI_Body_Baked_MI_Body_Baked_"), "{a}");
        assert!(b.starts_with("MI_Body_Baked_MI_Body_Baked_"), "{b}");
        // …and it is DETERMINISTIC, which is what the overwrite rule needs: the
        // same key names the same file on every run, on every machine.
        assert_eq!(
            a,
            short_name("INF_Built_INF_Dominic_Body_Materials_MI_Body_Baked_MI_Body_Baked")
        );
    }
}
