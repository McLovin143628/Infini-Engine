//! **The map-set importer** (Wave T, `docs/Rust_Game_Engine_Textures.md` items
//! T14 and T20): a Megascans/Fab texture *family* on disk becomes one material's
//! worth of `.inf_tex` imports, with the right format and the right colour space
//! chosen per map, and the three loose single-channel maps packed into one ORM.
//!
//! # What was missing, and why it is two functions rather than a pipeline
//!
//! `import_texture_bytes` imports **one image at a time**, and the caller
//! decides `srgb` and `compression`. That is exactly right for a glTF import,
//! where the container already says which slot each image fills. It is exactly
//! wrong for the shape the document's whole premise rests on — *"Source Assets:
//! 4K/8K Megascans PNG/EXR from Fab"* — where the slot is written in the
//! **filename** and nothing reads it:
//!
//! ```text
//! rock_cliff_2K_Albedo.jpg        rock_cliff_2K_Normal.jpg
//! rock_cliff_2K_Roughness.jpg     rock_cliff_2K_AO.jpg
//! rock_cliff_2K_Displacement.exr  rock_cliff_2K_Cavity.jpg
//! ```
//!
//! Imported one at a time with the defaults, that set produces six sRGB BC1
//! textures: a normal map with its X quantised to 32 levels, three grayscale
//! maps each paying for three channels it does not use, and a displacement map
//! whose entire float range was thrown away on the way in. Every one of those is
//! a *silent* wrong answer.
//!
//! So this module is deliberately **two pure functions and no I/O**:
//!
//! * [`classify_map`] — filename → [`MapKind`], the one place a suffix
//!   convention is written down;
//! * [`plan_map_set`] — a list of filenames → a [`MapSetPlan`]: which files
//!   become which `.inf_tex`, with which [`TextureImportSettings`], and which
//!   three get packed into one ORM by [`pack_orm`].
//!
//! The caller does the reading and the writing. That keeps the *decisions*
//! testable with no filesystem, no project and no GPU — which is the same reason
//! `texture_import_advisories` is a pure function of an extent.

use crate::texture::{TextureCompression, TextureImportSettings};

/// Which slot of a PBR material a source file fills.
///
/// **Freeze-pinned in spirit rather than on a wire**: nothing serializes this,
/// so a variant may be added freely — but the *suffix table* below is a contract
/// with content that already exists on disk, and a suffix is only ever added,
/// never repurposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapKind {
    /// Base colour. The **only** sRGB map in the set.
    Albedo,
    /// Tangent-space normal.
    Normal,
    /// Ambient occlusion — packs into ORM's **R**.
    Occlusion,
    /// Roughness — packs into ORM's **G**.
    Roughness,
    /// Metallic — packs into ORM's **B**.
    Metallic,
    /// Height / displacement. Float where the source is float.
    Displacement,
    /// Opacity / alpha mask.
    Opacity,
    /// Cavity — a second, finer occlusion term. Kept as its own map rather than
    /// multiplied into AO at import, because that multiply is a *shading*
    /// decision and baking it in is unrecoverable.
    Cavity,
    /// Translucency / subsurface.
    Translucency,
}

impl MapKind {
    /// Whether this map's values are sRGB-encoded. **Exactly one is** — a base
    /// colour is a colour and everything else in a PBR set is data, and getting
    /// this backwards is the single most common texture-import mistake there is.
    pub fn srgb(self) -> bool {
        matches!(self, MapKind::Albedo)
    }

    /// Whether this map is one of the three [`pack_orm`] combines.
    pub fn packs_into_orm(self) -> bool {
        matches!(
            self,
            MapKind::Occlusion | MapKind::Roughness | MapKind::Metallic
        )
    }

    /// The import settings this map wants, given whether its source is float.
    ///
    /// The whole point of the module, in one function: an albedo is sRGB and
    /// block-compressed, a **normal map is BC5** (two channels, Z rebuilt — a
    /// quarter of the bytes an RGBA8 normal costs and strictly better than the
    /// BC1 a naive import would reach for), and a **float source keeps its
    /// range** rather than being flattened to 8 bits.
    ///
    /// Displacement and cavity default to `hdr` when the source is float because
    /// that is where the range actually lives: a Megascans displacement map is
    /// authored in metres and clips to nothing useful in `[0, 1]`.
    pub fn settings(self, source_is_float: bool) -> TextureImportSettings {
        match self {
            MapKind::Albedo => TextureImportSettings {
                srgb: true,
                generate_mips: true,
                compression: TextureCompression::Auto,
                hdr: false,
            },
            MapKind::Normal => TextureImportSettings::normal_map(),
            // The three ORM inputs are imported as ORM, not individually — see
            // `plan_map_set`. If a caller imports one alone it gets the data
            // preset, which keeps all eight bits of the one channel that matters.
            MapKind::Occlusion | MapKind::Roughness | MapKind::Metallic | MapKind::Opacity => {
                TextureImportSettings::data()
            }
            MapKind::Displacement | MapKind::Cavity | MapKind::Translucency => {
                TextureImportSettings {
                    hdr: source_is_float,
                    ..TextureImportSettings::data()
                }
            }
        }
    }
}

/// **The suffix table.** Lower-cased, matched against the filename stem's tail
/// after any separator, longest first so `_basecolor` cannot be eaten by
/// `_color`.
///
/// Every entry is a spelling that ships in the wild: Megascans/Fab, Substance's
/// default export names, and the glTF-adjacent conventions the engine's own
/// importer already writes. A suffix is only ever added.
const SUFFIXES: &[(&str, MapKind)] = &[
    ("translucency", MapKind::Translucency),
    ("displacement", MapKind::Displacement),
    ("metallness", MapKind::Metallic),
    ("metalness", MapKind::Metallic),
    ("basecolour", MapKind::Albedo),
    ("basecolor", MapKind::Albedo),
    ("occlusion", MapKind::Occlusion),
    ("roughness", MapKind::Roughness),
    ("metallic", MapKind::Metallic),
    ("normalgl", MapKind::Normal),
    ("normaldx", MapKind::Normal),
    ("diffuse", MapKind::Albedo),
    ("opacity", MapKind::Opacity),
    ("albedo", MapKind::Albedo),
    ("cavity", MapKind::Cavity),
    ("height", MapKind::Displacement),
    ("normal", MapKind::Normal),
    ("rough", MapKind::Roughness),
    ("alpha", MapKind::Opacity),
    ("color", MapKind::Albedo),
    ("disp", MapKind::Displacement),
    ("nrm", MapKind::Normal),
    ("occ", MapKind::Occlusion),
    ("ao", MapKind::Occlusion),
];

/// The source extensions this planner will consider, mirroring
/// `inf_asset::importable_source_kind`'s texture row.
///
/// **TIFF is absent and that is a finding, not an omission**: Megascans ships
/// some 16-bit maps as `.tif`, the workspace's `image` pin does not enable the
/// `tiff` feature, and turning it on is a dependency decision rather than a code
/// one. Recorded in `docs/memos/wave-t-textures-disposition.md`.
const SOURCE_EXTS: &[&str] = &["png", "jpg", "jpeg", "tga", "bmp", "hdr", "exr"];

/// Split a file name into `(stem, lower-cased extension)`.
fn split_name(file: &str) -> (&str, String) {
    let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
    match name.rsplit_once('.') {
        Some((stem, ext)) => (stem, ext.to_ascii_lowercase()),
        None => (name, String::new()),
    }
}

/// **Which PBR slot a file fills, and what the set it belongs to is called.**
///
/// Returns `None` for a name that carries no recognised suffix (and for a file
/// whose extension this project cannot decode at all), rather than guessing —
/// a guess here silently mis-imports somebody's whole material.
///
/// The "base name" is the stem with the suffix and its separator removed, and it
/// is what groups a family: `rock_cliff_2K_Albedo.jpg` and
/// `rock_cliff_2K_Normal.jpg` both answer `rock_cliff_2K`. Trailing separators
/// are trimmed so `Rock-Cliff-Normal` and `Rock_Cliff_Normal` group with each
/// other's siblings.
pub fn classify_map(file: &str) -> Option<(String, MapKind)> {
    let (stem, ext) = split_name(file);
    if !SOURCE_EXTS.contains(&ext.as_str()) {
        return None;
    }
    let lower = stem.to_ascii_lowercase();
    for (suffix, kind) in SUFFIXES {
        if let Some(base) = lower.strip_suffix(suffix) {
            // A suffix must be preceded by a separator or be the whole stem;
            // otherwise `disp` matches `wisp` and `ao` matches `cameo`.
            let ok = base.is_empty() || base.ends_with(['_', '-', ' ', '.']);
            if !ok {
                continue;
            }
            let base = base.trim_end_matches(['_', '-', ' ', '.']);
            return Some((
                if base.is_empty() {
                    stem.to_string()
                } else {
                    stem[..base.len()].to_string()
                },
                *kind,
            ));
        }
    }
    None
}

/// One `.inf_tex` a [`MapSetPlan`] will produce.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedTexture {
    /// The slot this texture fills. For a packed ORM it is
    /// [`MapKind::Roughness`] — the glTF metallic-roughness slot the engine's
    /// `VtMaterialMaps::orm` and `vt_sample.wgsl` already speak.
    pub kind: MapKind,
    /// The source files that go into it, in the order [`pack_orm`] wants them
    /// (occlusion, roughness, metallic) — one entry for every kind but ORM.
    pub sources: Vec<String>,
    /// Whether this is the packed ORM triple rather than a single image.
    pub packed_orm: bool,
    /// The settings to import it with.
    pub settings: TextureImportSettings,
}

/// What [`plan_map_set`] decided.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MapSetPlan {
    /// The family's base name (the longest common one, when the files disagree).
    pub base_name: String,
    /// The `.inf_tex` imports, in a stable order: albedo, normal, ORM, then the
    /// rest by [`MapKind`]'s own ordering. Stable because an import order that
    /// depends on directory listing order is an import that produces different
    /// GUIDs on two machines.
    pub textures: Vec<PlannedTexture>,
    /// Files that carried no recognised suffix, in input order — reported rather
    /// than dropped, because a set with an unrecognised member is a set the
    /// author should look at.
    pub unrecognised: Vec<String>,
    /// Non-fatal notices, in the P16 cook-advisory shape.
    pub advisories: Vec<String>,
}

/// **Plan a whole texture family's import.**
///
/// `float_sources` answers "is this file floating-point" for each input — the
/// caller knows, because it has the bytes; this module has only names. Pass an
/// empty slice to plan as though nothing is float.
pub fn plan_map_set(files: &[String], float_sources: &[bool]) -> MapSetPlan {
    let mut plan = MapSetPlan::default();
    let mut found: Vec<(MapKind, String, bool)> = Vec::new();
    let mut bases: Vec<String> = Vec::new();
    for (i, f) in files.iter().enumerate() {
        let is_float = float_sources.get(i).copied().unwrap_or(false);
        match classify_map(f) {
            Some((base, kind)) => {
                found.push((kind, f.clone(), is_float));
                if !bases.contains(&base) {
                    bases.push(base);
                }
            }
            None => plan.unrecognised.push(f.clone()),
        }
    }
    plan.base_name = bases.first().cloned().unwrap_or_default();
    if bases.len() > 1 {
        plan.advisories.push(format!(
            "these files carry {} different base names ({}); they are being imported as one \
             material set, which is wrong if they are two",
            bases.len(),
            bases.join(", ")
        ));
    }
    // Sorted by kind, then by name, so the plan is a pure function of the SET and
    // not of the order a directory walk happened to yield.
    found.sort();

    let take = |k: MapKind| -> Option<&(MapKind, String, bool)> {
        found.iter().find(|(kind, _, _)| *kind == k)
    };

    let mut push = |plan: &mut MapSetPlan, k: MapKind| {
        if let Some((_, file, is_float)) = take(k) {
            plan.textures.push(PlannedTexture {
                kind: k,
                sources: vec![file.clone()],
                packed_orm: false,
                settings: k.settings(*is_float),
            });
        }
    };

    push(&mut plan, MapKind::Albedo);
    push(&mut plan, MapKind::Normal);

    // The ORM triple. Present if ANY of the three is — a set with roughness and
    // no metallic is the common case for a rock or a fabric, and the packer
    // fills the missing channel with its neutral value rather than refusing.
    let orm: Vec<String> = [MapKind::Occlusion, MapKind::Roughness, MapKind::Metallic]
        .into_iter()
        .map(|k| take(k).map(|(_, f, _)| f.clone()).unwrap_or_default())
        .collect();
    if orm.iter().any(|f| !f.is_empty()) {
        plan.textures.push(PlannedTexture {
            kind: MapKind::Roughness,
            sources: orm.clone(),
            packed_orm: true,
            settings: TextureImportSettings::data(),
        });
        let missing: Vec<&str> = [("occlusion", 0usize), ("roughness", 1), ("metallic", 2)]
            .into_iter()
            .filter(|(_, i)| orm[*i].is_empty())
            .map(|(n, _)| n)
            .collect();
        if !missing.is_empty() {
            plan.advisories.push(format!(
                "the ORM pack has no {} map; that channel is filled with its neutral value \
                 ({})",
                missing.join(" and "),
                ORM_NEUTRAL_NOTE
            ));
        }
    }

    for k in [
        MapKind::Displacement,
        MapKind::Opacity,
        MapKind::Cavity,
        MapKind::Translucency,
    ] {
        push(&mut plan, k);
    }
    plan
}

/// What the packer writes into a channel it was given nothing for.
const ORM_NEUTRAL_NOTE: &str = "occlusion 1.0, roughness 1.0, metallic 0.0";

/// **Pack three loose single-channel maps into one ORM image** (item T14).
///
/// `R = occlusion, G = roughness, B = metallic, A = 255` — the glTF
/// metallic-roughness convention `vt_sample.wgsl` already reads and the importer
/// already writes, so nothing downstream learns a second layout.
///
/// Each input is an RGBA8 image of `width × height`; the packer takes its **red**
/// channel, because a grayscale map decoded through `image` has its one value in
/// all three colour channels and R is the one that is always there. A `None`
/// input fills its channel with the neutral value — 255 for occlusion (fully
/// lit), 255 for roughness (fully rough, the glTF factor default), 0 for metallic
/// (dielectric) — which is what makes a set with no metallic map, the common case
/// for stone and fabric, import correctly instead of being refused.
///
/// Pure and integer-only: the bytes it writes are content-hashed into a
/// reproducible pack, exactly like [`crate::bc`]'s.
///
/// Returns `None` if an input is shorter than the extent it is declared at,
/// which is the one thing that would read past the end of a buffer.
pub fn pack_orm(
    occlusion: Option<&[u8]>,
    roughness: Option<&[u8]>,
    metallic: Option<&[u8]>,
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    let n = width as usize * height as usize;
    for src in [occlusion, roughness, metallic].into_iter().flatten() {
        if src.len() < n * 4 {
            return None;
        }
    }
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        out[i * 4] = occlusion.map_or(255, |s| s[i * 4]);
        out[i * 4 + 1] = roughness.map_or(255, |s| s[i * 4]);
        out[i * 4 + 2] = metallic.map_or(0, |s| s[i * 4]);
        out[i * 4 + 3] = 255;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// **The suffix table recognises what Fab actually ships**, and refuses what
    /// it does not recognise instead of guessing.
    #[test]
    fn a_megascans_family_is_classified_by_its_suffixes() {
        for (name, kind) in [
            ("rock_cliff_2K_Albedo.jpg", MapKind::Albedo),
            ("rock_cliff_2K_Normal.jpg", MapKind::Normal),
            ("rock_cliff_2K_NormalGL.png", MapKind::Normal),
            ("rock_cliff_2K_Roughness.jpg", MapKind::Roughness),
            ("rock_cliff_2K_AO.jpg", MapKind::Occlusion),
            ("rock_cliff_2K_Metalness.jpg", MapKind::Metallic),
            ("rock_cliff_2K_Displacement.exr", MapKind::Displacement),
            ("rock_cliff_2K_Cavity.jpg", MapKind::Cavity),
            ("rock_cliff_2K_Opacity.png", MapKind::Opacity),
            ("Rock-Cliff-BaseColor.png", MapKind::Albedo),
        ] {
            let (base, got) = classify_map(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(got, kind, "{name}");
            assert!(
                base.eq_ignore_ascii_case("rock_cliff_2K")
                    || base.eq_ignore_ascii_case("Rock-Cliff"),
                "{name} grouped under {base:?}"
            );
        }
        // The whole family groups under ONE base name — the property that makes
        // a set a set.
        let bases: std::collections::BTreeSet<String> = [
            "rock_cliff_2K_Albedo.jpg",
            "rock_cliff_2K_Normal.jpg",
            "rock_cliff_2K_Roughness.jpg",
        ]
        .into_iter()
        .map(|n| classify_map(n).unwrap().0)
        .collect();
        assert_eq!(bases.len(), 1, "{bases:?}");

        // Refusals. A suffix must be preceded by a separator, an unknown suffix
        // is not guessed at, and an extension this project cannot decode is not
        // planned for.
        assert_eq!(classify_map("wisp.png"), None, "`disp` inside a word");
        assert_eq!(classify_map("cameo.png"), None, "`ao` inside a word");
        assert_eq!(classify_map("rock_cliff_2K.jpg"), None, "no suffix");
        assert_eq!(
            classify_map("rock_Albedo.tif"),
            None,
            "tiff is not decodable"
        );
        assert_eq!(classify_map("readme.txt"), None);
        // …but a bare suffix IS the whole stem, which is how a per-material
        // folder ("Rock/Albedo.png") is laid out.
        assert_eq!(classify_map("Albedo.png").unwrap().1, MapKind::Albedo);
    }

    /// **Exactly one map in a PBR set is sRGB**, and the normal map is BC5 —
    /// the two decisions that were silently wrong before this module.
    #[test]
    fn every_map_gets_the_right_colour_space_and_format() {
        assert!(MapKind::Albedo.srgb());
        for k in [
            MapKind::Normal,
            MapKind::Occlusion,
            MapKind::Roughness,
            MapKind::Metallic,
            MapKind::Displacement,
            MapKind::Opacity,
            MapKind::Cavity,
            MapKind::Translucency,
        ] {
            assert!(!k.srgb(), "{k:?} must not be sRGB");
            assert!(!k.settings(false).srgb, "{k:?}");
        }
        assert!(MapKind::Albedo.settings(false).srgb);
        assert_eq!(
            MapKind::Normal.settings(false).compression,
            TextureCompression::Bc5,
            "a normal map imported as anything else is the artifact this wave exists to end"
        );
        // A float displacement keeps its range; a JPEG one does not pretend to.
        assert!(MapKind::Displacement.settings(true).hdr);
        assert!(!MapKind::Displacement.settings(false).hdr);
        // …and an albedo is never promoted to float, however the source arrived:
        // a base colour lives in [0, 1] by definition.
        assert!(!MapKind::Albedo.settings(true).hdr);
    }

    /// **The plan is a pure function of the SET**, not of the order the files
    /// arrived in, and the three loose maps become one ORM.
    #[test]
    fn a_family_plans_into_albedo_normal_and_one_packed_orm() {
        let files = f(&[
            "rock_2K_Roughness.jpg",
            "rock_2K_Albedo.jpg",
            "rock_2K_AO.jpg",
            "rock_2K_Normal.jpg",
            "rock_2K_Displacement.exr",
            "notes.txt",
        ]);
        let float = vec![false, false, false, false, true, false];
        let plan = plan_map_set(&files, &float);

        assert_eq!(plan.base_name, "rock_2K");
        assert_eq!(plan.unrecognised, vec!["notes.txt".to_string()]);
        let kinds: Vec<(MapKind, bool)> = plan
            .textures
            .iter()
            .map(|t| (t.kind, t.packed_orm))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (MapKind::Albedo, false),
                (MapKind::Normal, false),
                (MapKind::Roughness, true),
                (MapKind::Displacement, false),
            ]
        );
        // The ORM entry names its three sources in R, G, B order with the absent
        // metallic left empty — the order `pack_orm` reads them in.
        let orm = &plan.textures[2];
        assert_eq!(
            orm.sources,
            vec![
                "rock_2K_AO.jpg".to_string(),
                "rock_2K_Roughness.jpg".to_string(),
                String::new()
            ]
        );
        assert_eq!(plan.advisories.len(), 1, "{:?}", plan.advisories);
        assert!(
            plan.advisories[0].contains("metallic"),
            "{:?}",
            plan.advisories
        );
        // The EXR displacement keeps its range; the JPEG albedo does not become
        // float because a sibling was.
        assert!(plan.textures[3].settings.hdr);
        assert!(!plan.textures[0].settings.hdr);
        assert_eq!(
            plan.textures[1].settings.compression,
            TextureCompression::Bc5
        );

        // Purity: a shuffled input plans identically.
        let mut shuffled = files.clone();
        shuffled.reverse();
        let mut shuffled_float = float.clone();
        shuffled_float.reverse();
        let other = plan_map_set(&shuffled, &shuffled_float);
        assert_eq!(other.textures, plan.textures);
    }

    /// A single loose map still plans — the degenerate set is a set.
    #[test]
    fn one_map_alone_still_plans() {
        let plan = plan_map_set(&f(&["moss_Albedo.png"]), &[]);
        assert_eq!(plan.textures.len(), 1);
        assert!(plan.textures[0].settings.srgb);
        assert!(plan.advisories.is_empty());
        // Nothing recognised at all is an empty plan, not a panic.
        let empty = plan_map_set(&f(&["a.txt", "b.bin"]), &[]);
        assert!(empty.textures.is_empty());
        assert_eq!(empty.unrecognised.len(), 2);
    }

    /// **The ORM packer**: R = AO, G = roughness, B = metallic, A = 255, with
    /// the neutral fill for a channel the set does not have.
    #[test]
    fn pack_orm_writes_the_gltf_convention_and_fills_what_is_missing() {
        let n = 4usize;
        let gray = |v: u8| -> Vec<u8> { (0..n).flat_map(|_| [v, v, v, 255]).collect() };
        let ao = gray(40);
        let rough = gray(180);
        let metal = gray(220);

        let all = pack_orm(Some(&ao), Some(&rough), Some(&metal), 2, 2).unwrap();
        assert_eq!(all.len(), n * 4);
        for px in all.chunks_exact(4) {
            assert_eq!(px, &[40, 180, 220, 255]);
        }

        // The common case: stone with no metallic map.
        let none = pack_orm(Some(&ao), Some(&rough), None, 2, 2).unwrap();
        for px in none.chunks_exact(4) {
            assert_eq!(
                px,
                &[40, 180, 0, 255],
                "a missing metallic map must read as dielectric, not as 1.0"
            );
        }
        // And nothing at all is the neutral surface, not black.
        let bare = pack_orm(None, None, None, 2, 2).unwrap();
        for px in bare.chunks_exact(4) {
            assert_eq!(px, &[255, 255, 0, 255]);
        }

        // A truncated input is refused rather than read past.
        assert_eq!(pack_orm(Some(&ao[..7]), None, None, 2, 2), None);
        // Pure.
        assert_eq!(
            pack_orm(Some(&ao), Some(&rough), Some(&metal), 2, 2),
            pack_orm(Some(&ao), Some(&rough), Some(&metal), 2, 2)
        );
    }

    /// The advisories are sentences an author reads — no eaten line
    /// continuations (the chr(92) law, now on its eleventh catch elsewhere).
    #[test]
    fn no_plan_advisory_carries_an_eaten_continuation() {
        let mut seen = 0usize;
        for files in [
            f(&["a_Albedo.png", "a_Roughness.png"]),
            f(&["a_AO.png"]),
            f(&["one_Albedo.png", "two_Normal.png"]),
        ] {
            for a in plan_map_set(&files, &[]).advisories {
                seen += 1;
                assert!(!a.contains("  "), "run of spaces: {a:?}");
                assert!(!a.contains('\n'), "multi-line: {a:?}");
            }
        }
        assert!(seen >= 3, "the sweep produced only {seen} advisories");
    }
}
