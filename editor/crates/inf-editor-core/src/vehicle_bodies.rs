//! **THE HERO BODIES** (wave VEH3f) -- the first authored meshes ever drawn on a
//! vehicle in this engine: five sets of body panels built in the P23 DCC
//! kernel, committed as samples, licence-free because they are ours.
//!
//! # What a set IS
//!
//! Not one shell mesh over the car. A hero set is one `.inf_mesh` per PANEL the
//! family's silhouette is made of -- the lower body, the greenhouse, and the
//! bonnet and boot where there are those ([`inf_ecs::vehicle::hero_parts`]) --
//! each modelled in the SAME unit box the panel's primitive fills, so it hangs
//! on the part's own `RigNode` (`MeshRef::asset`) with the part's own transform.
//! That is the whole reason it works with everything VEH3c and VEH3d built: a
//! bonnet that pops on its hinge carries its DCC mesh with it, a panel that
//! dents moves its mesh inward exactly as it moved its box, and a livery paints
//! the mesh through the part's own `Material`. Doors, glass and bumpers stay
//! the VEH3c proxies they are.
//!
//! # How a panel is modelled
//!
//! A LOFT, in the kernel's own ops: a sixteen-sided `inf_dcc::cylinder`, laid on
//! its side with `Op::RotateVerts`, cut into stations along its length with
//! `Op::LoopCut`, and every vertex moved onto a cross-section with
//! `Op::TranslateVerts` -- a rounded rectangle (a superellipse of exponent four,
//! whose `|c|^(1/2)` is a square root and therefore portable) of a width, a
//! floor and a roof that are functions of the station, with a tumblehome
//! narrowing the top. So a greenhouse rakes its screen and its backlight, a
//! bonnet falls to the nose, a boot to the tail, and a lower body rounds its
//! corners in plan. The kernel refuses a bevel on a corner (the P23 finding),
//! and a loft needs none.
//!
//! # Determinism
//!
//! `pcos64`/`psin64`/`patan2_64`/`sqrt` and the kernel's own ops only; the
//! bytes are compared against the committed files by
//! `samples::tests::committed_sample_matches_generators` on all three CI
//! platforms.

use std::path::{Path, PathBuf};

use inf_dcc::ops::apply;
use inf_dcc::{cylinder, to_mesh_asset, ExportOptions, Mesh, NormalPolicy, Op};
use uuid::Uuid;

/// The committed folder, under `samples/`.
pub const VEHICLE_BODIES_FOLDER: &str = "vehicle-bodies";

/// The sixteen sides of a panel's cross-section.
const SEGMENTS: usize = 16;

/// The stations a panel is cut into along its length (plus its two ends).
const STATIONS: u32 = 14;

/// **The five sets**: the base `Guid` each row names as `body_mesh`, the file
/// prefix, the family, and the profile each panel is lofted from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeroSet {
    /// The island's saloon (`sedan`).
    Sedan,
    /// The island's coupe (`sports`).
    Coupe,
    /// The island's wagon (`suv`).
    Suv,
    /// The island's pickup (`truck`).
    Pickup,
    /// The island's police cruiser (`cruiser`) -- the saloon's panels with a
    /// deeper nose for the push bar.
    Cruiser,
}

impl HeroSet {
    /// Every set, in file order.
    pub const ALL: [HeroSet; 5] = [
        HeroSet::Sedan,
        HeroSet::Coupe,
        HeroSet::Suv,
        HeroSet::Pickup,
        HeroSet::Cruiser,
    ];

    /// The base guid a catalogue row's `body_mesh` names.
    pub fn base(self) -> Uuid {
        Uuid::from_u128(match self {
            HeroSet::Sedan => 0x5645_4833_4845_524f_8000_0000_0000_0001,
            HeroSet::Coupe => 0x5645_4833_4845_524f_8000_0000_0000_0002,
            HeroSet::Suv => 0x5645_4833_4845_524f_8000_0000_0000_0003,
            HeroSet::Pickup => 0x5645_4833_4845_524f_8000_0000_0000_0004,
            HeroSet::Cruiser => 0x5645_4833_4845_524f_8000_0000_0000_0005,
        })
    }

    /// The file prefix.
    pub fn name(self) -> &'static str {
        match self {
            HeroSet::Sedan => "Sedan",
            HeroSet::Coupe => "Coupe",
            HeroSet::Suv => "Suv",
            HeroSet::Pickup => "Pickup",
            HeroSet::Cruiser => "Cruiser",
        }
    }

    /// The family whose panels this set replaces.
    pub fn family(self) -> inf_ecs::vehicle::VehicleBody {
        use inf_ecs::vehicle::VehicleBody;
        match self {
            HeroSet::Sedan | HeroSet::Cruiser => VehicleBody::Sedan,
            HeroSet::Coupe => VehicleBody::Sports,
            HeroSet::Suv => VehicleBody::Suv,
            HeroSet::Pickup => VehicleBody::Truck,
        }
    }

    /// The island catalogue row that wears it.
    pub fn row(self) -> &'static str {
        match self {
            HeroSet::Sedan => "sedan",
            HeroSet::Coupe => "sports",
            HeroSet::Suv => "suv",
            HeroSet::Pickup => "truck",
            HeroSet::Cruiser => "cruiser",
        }
    }
}

/// **One station's cross-section**, unit-box units: the half-width at the
/// floor, the floor and the roof, and how much narrower the roof is.
#[derive(Clone, Copy, Debug)]
struct Section {
    half_w: f64,
    y_lo: f64,
    y_hi: f64,
    tumble: f64,
}

/// A smooth 0..1 ramp over `[a, b]` -- `3t^2 - 2t^3`.
fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The profile of one panel at station `z` in `[-0.5, 0.5]` (`+Z` forward).
fn section(set: HeroSet, part: &str, z: f64) -> Section {
    let ends = smooth(0.30, 0.5, z.abs());
    match part {
        // The lower body: full width, corners rounded in plan (the ends pull
        // in), the nose and tail tucked under at the overhangs.
        "lower" => {
            let nose = smooth(0.25, 0.5, z);
            let push_bar = if set == HeroSet::Cruiser { 0.06 } else { 0.0 };
            Section {
                half_w: 0.5 - 0.07 * ends,
                y_lo: -0.5 + 0.18 * ends,
                y_hi: 0.5 - (0.12 - push_bar) * nose - 0.05 * smooth(0.3, 0.5, -z),
                tumble: 0.04,
            }
        }
        // The greenhouse: a raked screen at the front, a backlight at the
        // rear (a fastback on the coupe), and a narrower roof than sill.
        "cabin" | "cab" => {
            let (front_from, rear_from, rake) = match (set, part) {
                (HeroSet::Coupe, _) => (0.05, -0.05, 0.9),
                (HeroSet::Suv, _) => (0.30, -0.42, 0.45),
                (HeroSet::Pickup, _) => (0.25, -0.45, 0.4),
                _ => (0.12, -0.18, 0.75),
            };
            let front = smooth(front_from, 0.5, z);
            let rear = smooth(-rear_from, 0.5, -z);
            let drop = (front + rear * if set == HeroSet::Coupe { 1.0 } else { 0.9 }) * rake;
            Section {
                half_w: 0.5 - 0.03 * ends,
                y_lo: -0.5,
                y_hi: (0.5 - drop).max(-0.38),
                tumble: if matches!(set, HeroSet::Suv | HeroSet::Pickup) {
                    0.1
                } else {
                    0.18
                },
            }
        }
        // The bonnet falls to the nose; the boot to the tail. Thin panels.
        "bonnet" => Section {
            half_w: 0.5 - 0.06 * smooth(0.2, 0.5, z),
            y_lo: -0.5,
            y_hi: 0.5 - 0.55 * smooth(-0.5, 0.5, z),
            tumble: 0.05,
        },
        "boot" => Section {
            half_w: 0.5 - 0.06 * smooth(0.2, 0.5, -z),
            y_lo: -0.5,
            y_hi: 0.5 - 0.45 * smooth(-0.3, 0.5, -z),
            tumble: 0.05,
        },
        // The pickup's bed: a tub, square in section, rounded at the tail.
        _ => Section {
            half_w: 0.5 - 0.03 * ends,
            y_lo: -0.5,
            y_hi: 0.5,
            tumble: 0.0,
        },
    }
}

/// `sign(c) * sqrt(|c|)` -- the exponent-four superellipse's coordinate.
fn super4(c: f64) -> f64 {
    c.signum() * c.abs().sqrt()
}

/// **Loft one panel** in the DCC kernel. Every vertex ends inside the unit box
/// the panel's primitive fills.
pub fn loft_panel(set: HeroSet, part: &str) -> Mesh {
    let mut m = cylinder(0.5, 1.0, SEGMENTS);
    // Lay it on its side: +Y (the cylinder's axis) onto +Z (the car's length).
    let all: Vec<_> = m.vert_ids().collect();
    apply(
        &mut m,
        &Op::RotateVerts {
            verts: all,
            pivot: [0.0, 0.0, 0.0],
            axis: [1.0, 0.0, 0.0],
            radians: std::f64::consts::FRAC_PI_2,
        },
    )
    .expect("a rotation of a closed cylinder applies");
    // A longitudinal edge: its two ends a whole length apart in z.
    let half = m
        .half_ids()
        .find(|h| {
            let (Some(a), Some(t)) = (m.origin(*h), m.twin(*h)) else {
                return false;
            };
            let Some(b) = m.origin(t) else {
                return false;
            };
            match (m.position(a), m.position(b)) {
                (Some(p), Some(q)) => (p.z - q.z).abs() > 0.99,
                _ => false,
            }
        })
        .expect("a cylinder has a side edge");
    apply(
        &mut m,
        &Op::LoopCut {
            half,
            cuts: STATIONS,
        },
    )
    .expect("the side strip loop-cuts");
    let verts: Vec<_> = m.vert_ids().collect();
    for v in verts {
        let p = m.position(v).expect("a live vertex");
        let s = section(set, part, p.z);
        let theta = inf_math::patan2_64(p.y, p.x);
        let (c, sn) = (inf_math::pcos64(theta), inf_math::psin64(theta));
        let mid = 0.5 * (s.y_lo + s.y_hi);
        let h = 0.5 * (s.y_hi - s.y_lo);
        let y = mid + h * super4(sn);
        let up = if h > 0.0 {
            (y - s.y_lo) / (2.0 * h)
        } else {
            0.0
        };
        let x = s.half_w * super4(c) * (1.0 - s.tumble * up);
        let target = [x, y, p.z];
        let delta = [target[0] - p.x, target[1] - p.y, target[2] - p.z];
        if delta.iter().any(|d| d.abs() > 0.0) {
            apply(
                &mut m,
                &Op::TranslateVerts {
                    verts: vec![v],
                    delta,
                },
            )
            .expect("a vertex moves");
        }
    }
    m
}

/// One committed file: its name, its guid and its asset.
pub struct HeroMesh {
    /// `<Set>_<part>.inf_mesh`.
    pub file: String,
    /// `inf_ecs::vehicle::hero_part_mesh_guid(set.base(), part)`.
    pub guid: Uuid,
    /// The baked asset.
    pub asset: inf_mesh::MeshAsset,
}

/// **Every hero mesh**, in file order.
pub fn hero_meshes() -> Vec<HeroMesh> {
    let mut out = Vec::new();
    for set in HeroSet::ALL {
        for part in inf_ecs::vehicle::hero_parts(set.family()) {
            let mesh = loft_panel(set, part);
            let (asset, _) = to_mesh_asset(
                &mesh,
                &ExportOptions {
                    normals: NormalPolicy::Recompute,
                    optimize: false,
                },
            );
            out.push(HeroMesh {
                file: format!("{}_{}.inf_mesh", set.name(), part),
                guid: inf_ecs::vehicle::hero_part_mesh_guid(set.base(), part),
                asset,
            });
        }
    }
    out
}

/// Every file the library writes, payloads and sidecars, sorted.
pub fn hero_files() -> Vec<String> {
    let mut v: Vec<String> = hero_meshes()
        .into_iter()
        .flat_map(|m| [m.file.clone(), format!("{}.toml", m.file)])
        .collect();
    v.sort();
    v
}

/// The committed folder.
pub fn vehicle_bodies_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../samples")
        .join(VEHICLE_BODIES_FOLDER)
}

/// **Write the library** -- every hero mesh and its sidecar.
pub fn write_vehicle_bodies(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    for m in hero_meshes() {
        let bytes = inf_asset::encode(&m.asset).map_err(|e| format!("encode {}: {e}", m.file))?;
        let path = dir.join(&m.file);
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(m.guid),
            inf_asset::AssetKind::Mesh,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&path)
        .map_err(|e| format!("write the sidecar for {}: {e}", m.file))?;
    }
    Ok(())
}

// ── the ART FALLBACK (wave VEH3f) ───────────────────────────────────────────

/// The committed folder the art fallbacks live in, under `samples/`.
pub const VEHICLE_ART_FOLDER: &str = "vehicle-art";

/// Transform a baked mesh's vertices by a scale and a translation (and, for a
/// tyre, a quarter turn about `Z` that lays the cylinder's axis on `X`).
fn placed(
    asset: &inf_mesh::MeshAsset,
    scale: [f64; 3],
    at: [f64; 3],
    axle_x: bool,
) -> Vec<inf_mesh::SubMesh> {
    asset
        .submeshes
        .iter()
        .map(|sm| {
            let mut sm = sm.clone();
            for v in sm.vertices.iter_mut() {
                let mut p = [
                    v.position[0] as f64,
                    v.position[1] as f64,
                    v.position[2] as f64,
                ];
                let mut n = [v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64];
                if axle_x {
                    // (x, y, z) -> (y, -x, z): the cylinder's +Y axis onto +X.
                    p = [p[1], -p[0], p[2]];
                    n = [n[1], -n[0], n[2]];
                }
                v.position = [
                    (p[0] * scale[0] + at[0]) as f32,
                    (p[1] * scale[1] + at[1]) as f32,
                    (p[2] * scale[2] + at[2]) as f32,
                ];
                v.normal = [n[0] as f32, n[1] as f32, n[2] as f32];
            }
            sm.material_slot = None;
            sm
        })
        .collect()
}

fn merged(
    template: &inf_mesh::MeshAsset,
    submeshes: Vec<inf_mesh::SubMesh>,
) -> inf_mesh::MeshAsset {
    // One submesh: a fallback draws in its part's one material.
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for sm in submeshes {
        let base = vertices.len() as u32;
        vertices.extend(sm.vertices);
        indices.extend(sm.indices.into_iter().map(|i| i + base));
    }
    let bounds = inf_mesh::Aabb::from_points(vertices.iter().map(|v| v.position));
    inf_mesh::MeshAsset {
        submeshes: vec![inf_mesh::SubMesh {
            name: "fallback".to_string(),
            vertices,
            indices,
            material_slot: None,
            skin: Vec::new(),
        }],
        bounds,
        material_slots: Vec::new(),
        material_slot_assets: Vec::new(),
        ..template.clone()
    }
}

/// **The committed fallback of every roster machine** (wave VEH3f): its body
/// as the FAMILY's own parts (every part but the seats) baked into one mesh in
/// chassis metres, and its four tyres as DCC cylinders on `X` -- at the GUIDs
/// the art rows name (`inf_ecs::roster::art_body_guid` / `art_wheel_guid`).
///
/// This is what a checkout without the Unreal art draws, and it is the
/// primitive silhouette the row would have drawn anyway: `inf-import
/// --vehicles` overwrites these identities in a LOCAL project with the
/// machine's real art, and nothing from Unreal ever reaches this folder.
pub fn art_fallback_meshes() -> Vec<HeroMesh> {
    let opts = ExportOptions {
        normals: NormalPolicy::Recompute,
        optimize: false,
    };
    let (unit, _) = to_mesh_asset(&inf_dcc::cube(1.0), &opts);
    let mut out = Vec::new();
    for key in inf_ecs::roster::ArtKey::ALL {
        let Some((_, def)) = inf_ecs::roster::roster()
            .0
            .iter()
            .find(|(_, d)| d.art == Some(key))
        else {
            continue;
        };
        let h = def.half_extents;
        let parts: Vec<inf_mesh::SubMesh> = def
            .body
            .parts()
            .iter()
            .filter(|p| p.kind != inf_ecs::vehicle::BodyPartKind::Seat)
            .flat_map(|p| {
                placed(
                    &unit,
                    [
                        2.0 * p.half.x * h.x,
                        2.0 * p.half.y * h.y,
                        2.0 * p.half.z * h.z,
                    ],
                    [p.centre.x * h.x, p.centre.y * h.y, p.centre.z * h.z],
                    false,
                )
            })
            .collect();
        out.push(HeroMesh {
            file: format!("{}_body.inf_mesh", key.name()),
            guid: inf_ecs::roster::art_body_guid(key),
            asset: merged(&unit, parts),
        });
        let r = def.wheel_radius_m;
        let (tyre, _) = to_mesh_asset(
            &inf_dcc::cylinder(r, 2.0 * r * inf_ecs::vehicle::TYRE_WIDTH_FRAC, 20),
            &opts,
        );
        for i in 0..4 {
            out.push(HeroMesh {
                file: format!("{}_wheel{i}.inf_mesh", key.name()),
                guid: inf_ecs::roster::art_wheel_guid(key, i),
                asset: merged(&tyre, placed(&tyre, [1.0, 1.0, 1.0], [0.0; 3], true)),
            });
        }
    }
    out
}

/// Every file the art fallback writes, sorted.
pub fn art_fallback_files() -> Vec<String> {
    let mut v: Vec<String> = art_fallback_meshes()
        .into_iter()
        .flat_map(|m| [m.file.clone(), format!("{}.toml", m.file)])
        .collect();
    v.sort();
    v
}

/// The committed art-fallback folder.
pub fn vehicle_art_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../samples")
        .join(VEHICLE_ART_FOLDER)
}

/// **Write the art fallback** -- every mesh and its sidecar.
pub fn write_vehicle_art_fallback(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    for m in art_fallback_meshes() {
        let bytes = inf_asset::encode(&m.asset).map_err(|e| format!("encode {}: {e}", m.file))?;
        let path = dir.join(&m.file);
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(m.guid),
            inf_asset::AssetKind::Mesh,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&path)
        .map_err(|e| format!("write the sidecar for {}: {e}", m.file))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every panel is inside its own unit box and is a real shape** -- not a
    /// cube: its bounds fill the box on every axis (so the part's scale maps it
    /// onto the part), and it has more vertices than a box has.
    #[test]
    fn every_hero_panel_fills_its_unit_box_and_is_not_a_box() {
        let mut n = 0usize;
        for set in HeroSet::ALL {
            for part in inf_ecs::vehicle::hero_parts(set.family()) {
                let m = loft_panel(set, part);
                let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
                for v in m.vert_ids() {
                    let p = m.position(v).unwrap();
                    for (i, c) in [p.x, p.y, p.z].into_iter().enumerate() {
                        lo[i] = lo[i].min(c);
                        hi[i] = hi[i].max(c);
                    }
                }
                for i in 0..3 {
                    assert!(
                        lo[i] >= -0.5 - 1e-9 && hi[i] <= 0.5 + 1e-9,
                        "{set:?} {part}: axis {i} spans {} .. {}",
                        lo[i],
                        hi[i]
                    );
                    assert!(
                        hi[i] - lo[i] > 0.6,
                        "{set:?} {part}: axis {i} is {} of the box",
                        hi[i] - lo[i]
                    );
                }
                assert!(
                    m.vert_count() > 100,
                    "{set:?} {part}: {} vertices",
                    m.vert_count()
                );
                inf_dcc::validate(&m).unwrap_or_else(|e| panic!("{set:?} {part}: {e:?}"));
                n += 1;
            }
        }
        assert_eq!(n, 18, "five sets, eighteen panels");
    }

    /// The guids are the Ring-0 rule's, distinct, and the rows that name the
    /// sets name them.
    #[test]
    fn every_hero_mesh_has_the_guid_the_rig_asks_for() {
        let meshes = hero_meshes();
        let mut guids: Vec<Uuid> = meshes.iter().map(|m| m.guid).collect();
        guids.sort();
        guids.dedup();
        assert_eq!(guids.len(), meshes.len());
        let defs = crate::vehicle::island_vehicles();
        for set in HeroSet::ALL {
            let def = defs.get(set.row()).expect("the row");
            assert_eq!(
                def.body_mesh,
                Some(set.base()),
                "{} names its set",
                set.row()
            );
            assert_eq!(def.body, set.family());
        }
    }
}
