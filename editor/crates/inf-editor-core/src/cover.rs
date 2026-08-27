//! **The engine's ground-cover meshes** (wave TER2a, clause 5) — the three
//! things that stand on the ground, generated and committed beside the ground
//! itself.
//!
//! # What this closes
//!
//! The island's `.inf_pcg` declares three scatter kinds and **all three carry
//! `mesh: None`**. A kind with no mesh is a bare transform: the scatter runs, the
//! biome binding evaluates, the residency pages, the instances are counted — and
//! nothing is drawn. Two thousand six hundred and eighty-one instances of
//! nothing, every frame, on an island whose whole point is that you can walk
//! about on it.
//!
//! # Why they are generated rather than modelled
//!
//! The same reason `samples/starter-character/` is generated (SK1b's
//! `block_body_mesh`, whose shape this follows): a committed asset that came out
//! of somebody's DCC session is an asset nobody can regenerate, and the byte lock
//! that keeps sample content honest needs a generator on the other side of it.
//! These are `MeshAsset`s assembled from vertices and indices, in `f64`, with an
//! integer hash for every random number and **no transcendental in the path** —
//! the same discipline `inf_material::ground` states at length and for the same
//! byte-lock reason.
//!
//! # What they are, and what they are not
//!
//! Three low-poly props at the scale of what they represent:
//!
//! | kind | height | vertices | triangles | what it is |
//! |---|---|---|---|---|
//! | grass tuft | 0.307 m | 64 | 32 | eight tapering blades, hash-fanned |
//! | shrub | 0.741 m | 32 | 20 | a four-sided stem and three leaf cards |
//! | rock | 0.310 m | 384 | 128 | a twice-subdivided octahedron, hash-displaced and flat-shaded |
//!
//! (The numbers are the generator's, printed by
//! `every_cover_mesh_is_the_size_of_the_thing_it_is` and asserted against the
//! bounds a thing of that name should have.)
//!
//! They are **not** an art pass. They have no alpha cut-out (the scatter path
//! draws opaque), no wind animation of their own beyond the bend the deformation
//! field already applies, and no LOD band. What they are is the difference
//! between an island with ground cover on it and an island with a number.

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_mesh::asset::{MeshAsset, MeshVertex, SubMesh};
use uuid::Uuid;

/// The three kinds the island's `.inf_pcg` scatters, in **its own palette
/// order** — `kind_index` on a scattered instance indexes this list, so the
/// order is a wire contract with the committed `.inf_pcg`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverKind {
    /// The commonest: a tuft of grass. Weight 4 in the island's palette.
    GrassTuft,
    /// Weight 2 — a low shrub.
    Shrub,
    /// Weight 1 — a loose stone.
    Rock,
}

impl CoverKind {
    /// Every kind, in the island palette's frozen order.
    pub const ALL: [CoverKind; 3] = [CoverKind::GrassTuft, CoverKind::Shrub, CoverKind::Rock];

    /// The asset stem.
    pub const fn stem(self) -> &'static str {
        match self {
            CoverKind::GrassTuft => "Cover_GrassTuft",
            CoverKind::Shrub => "Cover_Shrub",
            CoverKind::Rock => "Cover_Rock",
        }
    }

    /// A label for a report.
    pub const fn label(self) -> &'static str {
        match self {
            CoverKind::GrassTuft => "grass tuft",
            CoverKind::Shrub => "shrub",
            CoverKind::Rock => "rock",
        }
    }

    const fn seed(self) -> u64 {
        match self {
            CoverKind::GrassTuft => 0xC0FE_0001,
            CoverKind::Shrub => 0xC0FE_0002,
            CoverKind::Rock => 0xC0FE_0003,
        }
    }

    /// The ground material this kind's surface reads as, for the material slot
    /// name. Cover shares the ground library rather than carrying textures of
    /// its own — the whole storage argument virtual texturing rests on.
    pub const fn slot_name(self) -> &'static str {
        match self {
            CoverKind::GrassTuft => "Ground_Grass",
            CoverKind::Shrub => "Ground_ForestFloor",
            CoverKind::Rock => "Ground_Rock",
        }
    }
}

/// The base of the cover library's GUID block, one after the ground library's
/// five sets of six. **Frozen**: the committed `.inf_pcg` names these.
const COVER_GUID_BASE: u128 = 0x9E20_0000 + 5 * 6;

/// The `.inf_mesh` GUID for a cover kind.
pub fn cover_mesh_guid(kind: CoverKind) -> Uuid {
    let slot = CoverKind::ALL
        .iter()
        .position(|k| *k == kind)
        .expect("every kind is in ALL") as u128;
    Uuid::from_u128(COVER_GUID_BASE + slot)
}

// ── the generators ──────────────────────────────────────────────────────────

#[inline]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic value in `[0, 1)` from a seed and an index.
#[inline]
fn unit(seed: u64, i: u64) -> f64 {
    (mix64(seed ^ mix64(i)) >> 11) as f64 / (1u64 << 53) as f64
}

/// A deterministic unit direction in the XZ plane, **without trigonometry**.
///
/// Rejection-sample a point in the unit disc and normalise it — the same
/// arrangement `inf_math`'s `unit_dir` uses, and for the same reason: `sin` and
/// `cos` are not bit-portable and these bytes are committed.
fn xz_dir(seed: u64, mut i: u64) -> (f64, f64) {
    loop {
        let x = unit(seed, i) * 2.0 - 1.0;
        let z = unit(seed, i + 0x5EED) * 2.0 - 1.0;
        let d2 = x * x + z * z;
        if d2 > 0.02 && d2 <= 1.0 {
            let d = d2.sqrt();
            return (x / d, z / d);
        }
        i += 1;
    }
}

fn vertex(p: [f64; 3], n: [f64; 3], uv: [f64; 2]) -> MeshVertex {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
    MeshVertex {
        position: [p[0] as f32, p[1] as f32, p[2] as f32],
        normal: [
            (n[0] / len) as f32,
            (n[1] / len) as f32,
            (n[2] / len) as f32,
        ],
        uv: [uv[0] as f32, uv[1] as f32],
        ..Default::default()
    }
}

/// A tapering blade: a quad from `base` along `dir`, leaning and narrowing.
///
/// Double-sided, because a blade seen from behind is not a hole — the scatter
/// path draws opaque with back-face culling on, and a one-sided card disappears
/// from half the compass.
fn blade(
    v: &mut Vec<MeshVertex>,
    idx: &mut Vec<u32>,
    base: [f64; 3],
    dir: (f64, f64),
    height: f64,
    width: f64,
    lean: f64,
) {
    // The blade's own frame: `dir` across, the lean tipping the top downwind.
    let (dx, dz) = dir;
    let (px, pz) = (-dz, dx);
    let tip = [
        base[0] + dx * lean * height,
        base[1] + height,
        base[2] + dz * lean * height,
    ];
    let hw = width * 0.5;
    let n = [px, 0.35, pz];
    let b0 = v.len() as u32;
    v.push(vertex(
        [base[0] - px * hw, base[1], base[2] - pz * hw],
        n,
        [0.0, 0.0],
    ));
    v.push(vertex(
        [base[0] + px * hw, base[1], base[2] + pz * hw],
        n,
        [1.0, 0.0],
    ));
    v.push(vertex(
        [tip[0] + px * hw * 0.18, tip[1], tip[2] + pz * hw * 0.18],
        n,
        [0.6, 1.0],
    ));
    v.push(vertex(
        [tip[0] - px * hw * 0.18, tip[1], tip[2] - pz * hw * 0.18],
        n,
        [0.4, 1.0],
    ));
    idx.extend_from_slice(&[b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3]);
    // …and the same quad wound the other way, with the normal flipped.
    let b1 = v.len() as u32;
    let back = [-n[0], n[1], -n[2]];
    for k in 0..4 {
        let src = &v[(b0 + k) as usize];
        v.push(vertex(
            [
                f64::from(src.position[0]),
                f64::from(src.position[1]),
                f64::from(src.position[2]),
            ],
            back,
            [f64::from(src.uv[0]), f64::from(src.uv[1])],
        ));
    }
    idx.extend_from_slice(&[b1, b1 + 2, b1 + 1, b1, b1 + 3, b1 + 2]);
}

/// **Generate one cover mesh.** Pure: the output is a function of `kind` alone.
pub fn build(kind: CoverKind) -> MeshAsset {
    let s = kind.seed();
    let mut v: Vec<MeshVertex> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();
    match kind {
        CoverKind::GrassTuft => {
            // Eight blades from a common root, fanned by an integer hash so the
            // tuft is not radially symmetric — a symmetric tuft reads as a
            // manufactured object at any distance a player can see it from.
            for b in 0..8u64 {
                let d = xz_dir(s, b);
                let jitter = unit(s, b + 100);
                let root = [d.0 * 0.012 * jitter, 0.0, d.1 * 0.012 * jitter];
                let h = 0.20 + 0.12 * unit(s, b + 200);
                let w = 0.016 + 0.008 * unit(s, b + 300);
                let lean = 0.25 + 0.35 * unit(s, b + 400);
                blade(&mut v, &mut idx, root, d, h, w, lean);
            }
        }
        CoverKind::Shrub => {
            // A stem (a four-sided taper) and three crossed leaf cards above it.
            let stem_h = 0.22;
            let r = 0.022;
            let ring = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
            let b0 = v.len() as u32;
            for (dx, dz) in ring {
                v.push(vertex([dx * r, 0.0, dz * r], [dx, 0.2, dz], [0.0, 0.0]));
            }
            for (dx, dz) in ring {
                v.push(vertex(
                    [dx * r * 0.6, stem_h, dz * r * 0.6],
                    [dx, 0.2, dz],
                    [0.0, 1.0],
                ));
            }
            for k in 0..4u32 {
                let a = b0 + k;
                let b = b0 + (k + 1) % 4;
                let c = b0 + 4 + (k + 1) % 4;
                let d = b0 + 4 + k;
                idx.extend_from_slice(&[a, b, c, a, c, d]);
            }
            for b in 0..3u64 {
                let d = xz_dir(s, b);
                let h = 0.42 + 0.18 * unit(s, b + 500);
                let w = 0.34 + 0.14 * unit(s, b + 600);
                blade(
                    &mut v,
                    &mut idx,
                    [0.0, stem_h * 0.7, 0.0],
                    d,
                    h,
                    w,
                    0.18 + 0.2 * unit(s, b + 700),
                );
            }
        }
        CoverKind::Rock => {
            // An octahedron subdivided twice and pushed out by a per-vertex hash
            // — a blob with facets, which is what a loose stone is.
            let mut pts: Vec<[f64; 3]> = vec![
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ];
            let mut tris: Vec<[usize; 3]> = vec![
                [0, 2, 4],
                [2, 1, 4],
                [1, 3, 4],
                [3, 0, 4],
                [2, 0, 5],
                [1, 2, 5],
                [3, 1, 5],
                [0, 3, 5],
            ];
            for _ in 0..2 {
                let mut next: Vec<[usize; 3]> = Vec::with_capacity(tris.len() * 4);
                let mut mid: std::collections::BTreeMap<(usize, usize), usize> =
                    std::collections::BTreeMap::new();
                for t in &tris {
                    let mut m = [0usize; 3];
                    for e in 0..3 {
                        let (a, b) = (t[e], t[(e + 1) % 3]);
                        let key = (a.min(b), a.max(b));
                        m[e] = *mid.entry(key).or_insert_with(|| {
                            let p = [
                                (pts[a][0] + pts[b][0]) * 0.5,
                                (pts[a][1] + pts[b][1]) * 0.5,
                                (pts[a][2] + pts[b][2]) * 0.5,
                            ];
                            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-9);
                            pts.push([p[0] / l, p[1] / l, p[2] / l]);
                            pts.len() - 1
                        });
                    }
                    next.push([t[0], m[0], m[2]]);
                    next.push([m[0], t[1], m[1]]);
                    next.push([m[2], m[1], t[2]]);
                    next.push([m[0], m[1], m[2]]);
                }
                tris = next;
            }
            // Squash it (a stone sits, it does not float) and rough it up.
            for (i, p) in pts.iter_mut().enumerate() {
                let bump = 0.78 + 0.34 * unit(s, i as u64);
                p[0] *= 0.24 * bump;
                p[1] *= 0.15 * bump;
                p[2] *= 0.21 * bump;
                p[1] += 0.15;
            }
            // Flat-shaded: a facet is what makes a low-poly stone read as stone,
            // and a smooth normal makes it read as a potato.
            for t in &tris {
                let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [
                    u[1] * w[2] - u[2] * w[1],
                    u[2] * w[0] - u[0] * w[2],
                    u[0] * w[1] - u[1] * w[0],
                ];
                let base = v.len() as u32;
                for (k, p) in [a, b, c].into_iter().enumerate() {
                    v.push(vertex(p, n, [k as f64 * 0.5, 0.5]));
                }
                idx.extend_from_slice(&[base, base + 1, base + 2]);
            }
        }
    }
    MeshAsset::new(
        vec![SubMesh {
            name: kind.label().to_string(),
            vertices: v,
            indices: idx,
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec![kind.slot_name().to_string()],
    )
}

/// One cover asset's bytes and its sidecar.
pub struct CoverFile {
    pub name: String,
    pub payload: Vec<u8>,
    pub sidecar: AssetSidecar,
}

/// Every file this library writes, as basenames.
pub fn cover_files() -> Vec<String> {
    let mut out = Vec::new();
    for kind in CoverKind::ALL {
        out.push(format!("{}.inf_mesh", kind.stem()));
        out.push(format!("{}.inf_mesh.toml", kind.stem()));
    }
    out.sort();
    out
}

/// **Generate the whole cover library in memory**, in the palette's own order.
pub fn cover_library() -> Result<Vec<CoverFile>, String> {
    let mut out = Vec::new();
    for kind in CoverKind::ALL {
        let mesh = build(kind);
        let payload =
            inf_asset::encode(&mesh).map_err(|e| format!("{}.inf_mesh: {e}", kind.stem()))?;
        let mut sidecar = AssetSidecar::new(
            AssetId(cover_mesh_guid(kind)),
            AssetKind::Mesh,
            ContentHash::of(&payload),
        );
        // The ground material this cover shares. A mesh's material slot is a
        // NAME, and the sidecar edge is what makes the cook ship the material
        // beside the mesh.
        sidecar.dependencies = vec![AssetId(crate::ground::ground_material_guid(match kind {
            CoverKind::GrassTuft => inf_material::ground::GroundKind::Grass,
            CoverKind::Shrub => inf_material::ground::GroundKind::ForestFloor,
            CoverKind::Rock => inf_material::ground::GroundKind::Rock,
        }))];
        sidecar.tags = vec!["ground".into(), "cover".into()];
        out.push(CoverFile {
            name: format!("{}.inf_mesh", kind.stem()),
            payload,
            sidecar,
        });
    }
    Ok(out)
}

/// Write the cover library into `dir`.
pub fn write_cover_library(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    for f in cover_library()? {
        let path = dir.join(&f.name);
        inf_asset::write_atomically(&path, &f.payload)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        f.sidecar
            .save(&path)
            .map_err(|e| format!("sidecar {}: {e}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The generators are pure**, which is the premise of committing them.
    #[test]
    fn two_builds_of_one_kind_agree_byte_for_byte() {
        for kind in CoverKind::ALL {
            let a = inf_asset::encode(&build(kind)).expect("encodes");
            let b = inf_asset::encode(&build(kind)).expect("encodes");
            assert_eq!(a, b, "{} is not a pure function", kind.label());
        }
    }

    /// Every mesh is a **real, bounded, correctly-sized** mesh — a generator
    /// that emitted an empty submesh, a degenerate bound or a metre-tall blade
    /// of grass would pass a byte-identity arm perfectly.
    #[test]
    fn every_cover_mesh_is_the_size_of_the_thing_it_is() {
        // (kind, min height, max height, max footprint radius)
        let want = [
            (CoverKind::GrassTuft, 0.15, 0.40, 0.15),
            (CoverKind::Shrub, 0.50, 1.10, 0.45),
            (CoverKind::Rock, 0.15, 0.60, 0.40),
        ];
        for (kind, lo, hi, rad) in want {
            let m = build(kind);
            let sub = &m.submeshes[0];
            assert!(!sub.vertices.is_empty(), "{} has no vertices", kind.label());
            assert_eq!(
                sub.indices.len() % 3,
                0,
                "{}'s index stream is not triangles",
                kind.label()
            );
            assert!(
                sub.indices
                    .iter()
                    .all(|i| (*i as usize) < sub.vertices.len()),
                "{} indexes past its vertex stream",
                kind.label()
            );
            let h = f64::from(m.bounds.max[1] - m.bounds.min[1]);
            assert!(
                (lo..=hi).contains(&h),
                "{} is {h:.3} m tall, outside {lo}..{hi}",
                kind.label()
            );
            let r = f64::from(
                (m.bounds.max[0] - m.bounds.min[0]).max(m.bounds.max[2] - m.bounds.min[2]),
            ) * 0.5;
            assert!(r <= rad, "{} is {r:.3} m across, past {rad}", kind.label());
            // It stands ON the ground: the lowest vertex is at or just under
            // zero, so a scattered instance is not floating or buried.
            let base = f64::from(m.bounds.min[1]);
            assert!(
                (-0.02..=0.02).contains(&base),
                "{}'s base is at y = {base:.3}, so every instance floats or sinks",
                kind.label()
            );
            // …and every normal is a unit vector, or the lighting is a lie.
            for v in &sub.vertices {
                let n = v.normal;
                let len = (f64::from(n[0]) * f64::from(n[0])
                    + f64::from(n[1]) * f64::from(n[1])
                    + f64::from(n[2]) * f64::from(n[2]))
                .sqrt();
                assert!(
                    (len - 1.0).abs() < 1e-3,
                    "{} has a normal of length {len:.4}",
                    kind.label()
                );
            }
            println!(
                "COVER {:>11}: {} verts, {} tris, {h:.3} m tall, {r:.3} m across",
                kind.label(),
                sub.vertices.len(),
                sub.indices.len() / 3
            );
        }
    }

    /// The three kinds are three meshes, three GUIDs and three stems — a
    /// copy-paste that reused a seed would make two of them identical and every
    /// arm above would still pass.
    #[test]
    fn the_three_kinds_are_three_of_everything() {
        let mut guids: Vec<Uuid> = CoverKind::ALL.iter().map(|k| cover_mesh_guid(*k)).collect();
        let n = guids.len();
        guids.sort();
        guids.dedup();
        assert_eq!(guids.len(), n, "two cover kinds share a GUID");
        let bodies: Vec<Vec<u8>> = CoverKind::ALL
            .iter()
            .map(|k| inf_asset::encode(&build(*k)).expect("encodes"))
            .collect();
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                assert_ne!(bodies[i], bodies[j], "two cover kinds are one mesh");
            }
        }
        // …and none of them collides with the ground library's block, which they
        // are laid out immediately after.
        for kind in inf_material::ground::GroundKind::ALL {
            let g = crate::ground::ground_ids(kind);
            for c in &guids {
                assert!(
                    ![g.material, g.albedo, g.normal, g.orm]
                        .iter()
                        .chain(g.detail.iter())
                        .any(|x| x == c),
                    "a cover mesh collides with the ground library's GUID block"
                );
            }
        }
    }

    /// A cover mesh's sidecar names the ground material it shares, so the cook's
    /// closure ships the two together. A mesh whose material is absent from the
    /// pack draws off its scalar attributes, which for a grass blade is a grey
    /// card.
    #[test]
    fn every_cover_mesh_declares_the_ground_it_shares() {
        let files = cover_library().expect("the cover library builds");
        assert_eq!(files.len(), 3);
        for f in &files {
            assert_eq!(
                f.sidecar.dependencies.len(),
                1,
                "{} names {} materials, not one",
                f.name,
                f.sidecar.dependencies.len()
            );
            assert_eq!(f.sidecar.content_hash, ContentHash::of(&f.payload));
        }
    }
}
