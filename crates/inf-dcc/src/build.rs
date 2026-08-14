//! Constructors: [`from_mesh_asset`] (the reader) and the primitives.
//!
//! # The weld/seam decision — the hard one
//!
//! A [`inf_mesh::MeshAsset`] is a *rendering* form: it stores one vertex per
//! **corner-attribute combination**, so a flat-shaded cube is 24 vertices at 8
//! distinct positions, and a UV-seam splits a vertex the same way. A half-edge
//! kernel needs the opposite: one vertex per *place*, because that is what
//! topology is about. So import has to weld — and welding is exactly where a
//! naive reader destroys the author's data.
//!
//! The rule, stated once:
//!
//! * **Positions weld by exact bit equality** ([`WELD_TOLERANCE`] is `0.0`, and
//!   that is a decision, not a placeholder — see below).
//! * **Every attribute that was split stays split**, because UVs and normals
//!   live on *corners* (face-side half-edges), not on vertices. Nothing is
//!   averaged and nothing is dropped: the 24 corners of that cube are still 24
//!   corners, sitting on 8 vertices. Export re-splits from the same information.
//!
//! ## Why the tolerance is exactly zero
//!
//! An epsilon weld is a *modelling operation* wearing a reader's clothes. It is
//! not transitive (a–b within ε and b–c within ε does not make a–c), so its
//! result depends on iteration order; it silently changes the topology of an
//! asset the author has already shipped; and it cannot be undone, because the
//! information about which corners used to be distinct is gone. Two positions
//! that came from the same `f32` are bit-equal — that is what an exporter
//! writing a shared vertex *does* — so an exact weld reconstructs the topology
//! the source file actually described. Merging *nearly* coincident vertices is a
//! deliberate edit; it is [`crate::ops::Op::WeldVerts`], it is journalled, and it
//! is undoable.
//!
//! ## Bowties are split, not refused
//!
//! Position-welding can create a vertex whose one-ring is two separate fans (two
//! boxes touching at a corner). The kernel promises vertex-manifoldness, so the
//! reader partitions each welded vertex's corners into fans and gives each fan
//! its own vertex — which is precisely the topology the source described. A DCC
//! that refused to open such a file would be wrong; one that welded it into a
//! bowtie would be broken.
//!
//! ## What import *does* refuse
//!
//! * A **newer schema** than this build understands.
//! * A **skinned** submesh. The kernel has no skinning model, so importing and
//!   exporting one would silently drop every weight. Refusing is the honest
//!   answer until P24 gives the kernel somewhere to put them.
//! * A **non-manifold edge** — the same directed edge used by two faces (two
//!   coincident triangles, or inconsistent winding). Unlike a bowtie this cannot
//!   be repaired by splitting a vertex without inventing geometry the source did
//!   not describe.
//!
//! Degenerate triangles (a repeated index, or two corners that weld together)
//! are **skipped and counted** in [`ImportReport`] rather than refused: they
//! carry no surface, they are common in exported content, and a count is an
//! advisory the caller can surface (the P16 cook-advisory doctrine).

use std::collections::BTreeMap;

use inf_math::{pcos64, psin64};
use inf_mesh::{MeshAsset, MeshVertex};
use serde::{Deserialize, Serialize};

use crate::skin::{SkinBinding, VertWeights};
use crate::topo::{CornerData, HalfId, Mesh, VertId};

/// The position weld tolerance: **exactly zero**. See the module docs.
pub const WELD_TOLERANCE: f64 = 0.0;

/// Why a [`MeshAsset`] could not become a [`Mesh`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ImportError {
    #[error("mesh asset schema v{found} is newer than this build understands (v{current})")]
    UnsupportedSchema { found: u32, current: u32 },
    /// A skin stream whose length does not match its submesh's vertex count.
    ///
    /// `SubMesh::skin` is a **parallel stream** — its own doc says
    /// `skin.len() == vertices.len()` when it is non-empty — so a mismatch means
    /// the two halves of one submesh came from different places. Refused rather
    /// than zipped to the shorter of the two, which would weight some of the mesh
    /// and silently leave the rest riding joint 0.
    ///
    /// (This replaces `ImportError::Skinned`, the P23 placeholder that refused
    /// *every* skinned asset because the kernel had nowhere to put a weight.)
    #[error("submesh {submesh} has {vertices} vertices but {skin} skin records")]
    SkinLengthMismatch {
        submesh: usize,
        vertices: usize,
        skin: usize,
    },
    #[error("submesh {submesh} has {indices} indices, which is not a whole number of triangles")]
    IndexCountNotTriangles { submesh: usize, indices: usize },
    #[error("submesh {submesh} index {index} is out of range ({vertices} vertices)")]
    IndexOutOfRange {
        submesh: usize,
        index: u32,
        vertices: usize,
    },
    #[error("the directed edge {from}→{to} is used by two faces (non-manifold edge)")]
    NonManifoldEdge { from: usize, to: usize },
    #[error("the asset has no triangles")]
    NoGeometry,
}

/// What the reader had to do to the source — advisories, not failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ImportReport {
    /// Vertices in the source asset, summed over submeshes.
    pub source_vertices: usize,
    /// Distinct positions after the exact weld.
    pub welded_positions: usize,
    /// Extra vertices minted to break bowties into single fans.
    pub fan_splits: usize,
    /// Triangles dropped because they carry no surface.
    pub degenerate_triangles_skipped: usize,
    /// Edges marked sharp because the corner normals disagree across them.
    pub sharp_edges: usize,
    /// **The exact-weld advisory.** Undirected edges with a face on only one
    /// side after import.
    ///
    /// [`WELD_TOLERANCE`] is zero, and it stays zero — an epsilon weld is not
    /// transitive, so its result would depend on iteration order, and it would
    /// silently re-topologize an asset the author has already shipped. But that
    /// decision has a consequence worth *measuring* rather than arguing about: a
    /// source whose seam positions differ by one ULP does not weld, and the mesh
    /// arrives fragmented.
    ///
    /// This counter is what makes that visible without a tolerance. A closed
    /// solid has **zero** boundary edges; the same solid whose seam failed to
    /// weld arrives with one boundary edge per seam edge — 24 for a cube split
    /// down every face. An author who believes their mesh is closed and sees a
    /// non-zero count has been told exactly what happened, and no epsilon had to
    /// be chosen on their behalf. (A genuinely open mesh — a plane, a cloth
    /// panel — reports its real boundary here and that is not a fault.)
    pub boundary_edges: usize,
    /// Source vertices carrying a non-finite position, normal or UV.
    ///
    /// The reader deliberately does **not** refuse these, and does not repair
    /// them: preserving attribute bits verbatim is what makes the export round
    /// trip exact, and an author who opened a file with one bad vertex should
    /// not be locked out of their own mesh.
    ///
    /// Precisely what that costs, stated rather than glossed: `validate` treats
    /// a non-finite position as a violation, and
    /// [`crate::journal::MeshSession::restore`] validates, so such a mesh
    /// **cannot be restored from a save**. It *can* be handed to
    /// [`crate::journal::MeshSession::new`], which does not validate because its
    /// input comes from this process — so a session started on an unchecked
    /// import will edit happily and then fail to reopen. That is a loud failure
    /// at the right moment rather than a silent one, but it is a real edge, and
    /// this counter is how a caller sees it coming. Read-side twin of
    /// [`crate::export::ExportReport::non_finite_written`]; the *edit* path has
    /// no such latitude and refuses outright
    /// ([`crate::ops::OpError::NonFinite`]).
    pub non_finite_values: usize,
    /// **The skin-weld advisory** (P24.2): welded positions where two source
    /// vertices disagreed about their skinning influences.
    ///
    /// A `MeshAsset` splits one surface vertex into several wherever a UV or a
    /// normal seam runs through it, and a well-formed exporter gives every copy
    /// the same weights — so this is normally zero, and a non-zero reading means
    /// the source's own split copies disagree. First occurrence wins (in index
    /// order, which is deterministic); averaging would invent a weight nobody
    /// authored, and refusing would lock an author out of a file that is only
    /// *slightly* wrong.
    ///
    /// It is also the exact number that makes the export round trip inexact: at
    /// zero, re-exporting reproduces every skin record; at `n`, up to `n` of them
    /// come back as the winner's.
    pub skin_conflicts: usize,
}

/// A mesh plus what the reader had to do to produce it.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshImport {
    pub mesh: Mesh,
    pub report: ImportReport,
}

/// Read a [`MeshAsset`] into the kernel. See the module docs for the weld rule.
pub fn from_mesh_asset(asset: &MeshAsset) -> Result<MeshImport, ImportError> {
    if asset.schema_version > MeshAsset::CURRENT_VERSION {
        return Err(ImportError::UnsupportedSchema {
            found: asset.schema_version,
            current: MeshAsset::CURRENT_VERSION,
        });
    }

    let mut report = ImportReport::default();
    // Provisional vertices: position bits → index, in first-appearance order.
    let mut weld: BTreeMap<[u64; 3], usize> = BTreeMap::new();
    let mut positions: Vec<[f64; 3]> = Vec::new();
    // **The skin channel** (P24.2), index-aligned to `positions`. `None` marks a
    // welded position no skinned submesh has spoken for yet, so the FIRST
    // occurrence wins and later disagreements are counted rather than fought
    // over — see `ImportReport::skin_conflicts`.
    let mut skins: Vec<Option<VertWeights>> = Vec::new();
    let mut faces: Vec<RawFace> = Vec::new();
    let mut any_skin = false;

    for (si, sm) in asset.submeshes.iter().enumerate() {
        if !sm.skin.is_empty() {
            if sm.skin.len() != sm.vertices.len() {
                return Err(ImportError::SkinLengthMismatch {
                    submesh: si,
                    vertices: sm.vertices.len(),
                    skin: sm.skin.len(),
                });
            }
            any_skin = true;
        }
        if sm.indices.len() % 3 != 0 {
            return Err(ImportError::IndexCountNotTriangles {
                submesh: si,
                indices: sm.indices.len(),
            });
        }
        report.source_vertices += sm.vertices.len();
        for tri in sm.indices.chunks_exact(3) {
            let mut verts = [0usize; 3];
            let mut corners = [CornerData::default(); 3];
            for (k, &raw) in tri.iter().enumerate() {
                let v: &MeshVertex =
                    sm.vertices
                        .get(raw as usize)
                        .ok_or(ImportError::IndexOutOfRange {
                            submesh: si,
                            index: raw,
                            vertices: sm.vertices.len(),
                        })?;
                let p = [
                    v.position[0] as f64,
                    v.position[1] as f64,
                    v.position[2] as f64,
                ];
                if !p.iter().all(|c| c.is_finite())
                    || !v.normal.iter().all(|c| c.is_finite())
                    || !v.uv.iter().all(|c| c.is_finite())
                {
                    report.non_finite_values += 1;
                }
                let key = bits3(p);
                let idx = *weld.entry(key).or_insert_with(|| {
                    positions.push(p);
                    skins.push(None);
                    positions.len() - 1
                });
                // The skin channel rides the weld (P24.2). A `MeshAsset` splits
                // one surface vertex into several whenever a UV or a normal seam
                // runs through it, and every copy carries the SAME influences —
                // so first-occurrence-wins is exact in the overwhelmingly common
                // case, and the rare disagreement is *counted* rather than
                // silently averaged into a weight nobody authored.
                if !sm.skin.is_empty() {
                    let s = sm.skin[raw as usize];
                    let w = VertWeights {
                        joints: s.joints,
                        weights: s.weights,
                    };
                    match &skins[idx] {
                        None => skins[idx] = Some(w),
                        Some(prev) if *prev != w => report.skin_conflicts += 1,
                        Some(_) => {}
                    }
                }
                verts[k] = idx;
                corners[k] = CornerData {
                    uv: [v.uv[0] as f64, v.uv[1] as f64],
                    normal: Some([v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64]),
                };
            }
            if verts[0] == verts[1] || verts[1] == verts[2] || verts[2] == verts[0] {
                report.degenerate_triangles_skipped += 1;
                continue;
            }
            faces.push(RawFace {
                verts: verts.to_vec(),
                corners: corners.to_vec(),
                slot: sm.material_slot,
            });
        }
    }

    report.welded_positions = positions.len();
    if faces.is_empty() {
        return Err(ImportError::NoGeometry);
    }

    // Edge-manifoldness is a property of the *source* and cannot be repaired by
    // splitting, so it is checked before anything is built.
    let mut directed: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for face in &faces {
        let n = face.verts.len();
        for i in 0..n {
            let key = (face.verts[i], face.verts[(i + 1) % n]);
            let count = directed.entry(key).or_insert(0);
            *count += 1;
            if *count > 1 {
                return Err(ImportError::NonManifoldEdge {
                    from: key.0,
                    to: key.1,
                });
            }
        }
    }

    report.fan_splits = split_bowties(&mut faces, &mut positions, &mut skins);

    // ── build ──────────────────────────────────────────────────────────────
    let mut mesh = Mesh::new();
    let ids: Vec<VertId> = positions.iter().map(|&p| mesh.alloc_vert(p)).collect();
    // The channel, if the asset carried one. `joints` is the tightest count the
    // FILE supports (one past the highest index it names) because a bare weight
    // stream says nothing more — see `SkinBinding::joints`.
    if any_skin {
        let mut top = 0u32;
        for (v, w) in ids.iter().zip(&skins) {
            let w = w.unwrap_or(VertWeights::RIGID).normalized();
            if let Some(j) = w.max_joint() {
                top = top.max(j as u32 + 1);
            }
            mesh.set_vert_weights(*v, w);
        }
        mesh.set_skin_binding(Some(SkinBinding {
            skeleton: None,
            joints: top.max(1),
        }));
    }
    mesh.set_material_slots(asset.material_slots.clone());
    for sm in &asset.submeshes {
        mesh.set_slot_name(sm.material_slot, sm.name.clone());
    }
    let mut touched = std::collections::BTreeSet::new();
    for face in &faces {
        let loop_verts: Vec<VertId> = face.verts.iter().map(|&i| ids[i]).collect();
        touched.extend(loop_verts.iter().copied());
        mesh.add_face_raw(&loop_verts, &face.corners, face.slot)
            .expect("import checked distinctness and edge-manifoldness up front");
    }
    mesh.finish_patch(&touched)
        .expect("bowties were split before building");

    report.sharp_edges = mark_sharp_from_normals(&mut mesh);
    report.boundary_edges = mesh
        .half_ids()
        .filter(|&h| mesh.is_boundary(h) == Some(true))
        .count();
    Ok(MeshImport { mesh, report })
}

struct RawFace {
    verts: Vec<usize>,
    corners: Vec<CornerData>,
    slot: Option<u32>,
}

/// Partition each welded vertex's corners into fans and give every fan after the
/// first its own vertex. Returns how many vertices were minted.
fn split_bowties(
    faces: &mut [RawFace],
    positions: &mut Vec<[f64; 3]>,
    skins: &mut Vec<Option<VertWeights>>,
) -> usize {
    // corner = (face index, corner index within that face)
    let mut at_vert: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (fi, face) in faces.iter().enumerate() {
        for ci in 0..face.verts.len() {
            at_vert.entry(face.verts[ci]).or_default().push((fi, ci));
        }
    }

    let mut remap: Vec<((usize, usize), usize)> = Vec::new();
    let mut minted = 0usize;
    for (&v, corners) in &at_vert {
        if corners.len() < 2 {
            continue;
        }
        // Union corners that share an undirected edge at `v`.
        let mut uf: Vec<usize> = (0..corners.len()).collect();
        let mut by_edge: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (k, &(fi, ci)) in corners.iter().enumerate() {
            let face = &faces[fi];
            let n = face.verts.len();
            let prev = face.verts[(ci + n - 1) % n];
            let next = face.verts[(ci + 1) % n];
            for other in [prev, next] {
                let key = if v < other { (v, other) } else { (other, v) };
                by_edge.entry(key).or_default().push(k);
            }
        }
        for group in by_edge.values() {
            for w in group.windows(2) {
                union(&mut uf, w[0], w[1]);
            }
        }
        // Fans, in ascending order of their lowest corner — deterministic.
        let mut fans: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for k in 0..corners.len() {
            fans.entry(find(&mut uf, k)).or_default().push(k);
        }
        if fans.len() < 2 {
            continue;
        }
        for (fan_index, (_, members)) in fans.iter().enumerate() {
            if fan_index == 0 {
                continue; // the first fan keeps the original vertex
            }
            positions.push(positions[v]);
            // A fan split mints a vertex at the SAME position, so it is the same
            // surface point and carries the same influences (P24.2).
            skins.push(skins[v]);
            let fresh = positions.len() - 1;
            minted += 1;
            for &k in members {
                remap.push((corners[k], fresh));
            }
        }
    }
    // Applied after every fan has been computed, so the grouping never sees a
    // half-remapped neighbourhood.
    for ((fi, ci), fresh) in remap {
        faces[fi].verts[ci] = fresh;
    }
    minted
}

fn find(uf: &mut [usize], mut i: usize) -> usize {
    while uf[i] != i {
        uf[i] = uf[uf[i]];
        i = uf[i];
    }
    i
}

fn union(uf: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(uf, a), find(uf, b));
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        uf[hi] = lo;
    }
}

/// Mark an interior edge sharp when the authored corner normals disagree across
/// it. Imported corners always carry authored normals, so this recovers the
/// source's smoothing: every edge of a flat-shaded cube is sharp, no edge of a
/// smooth sphere is. The flag is what a *later* op's derived normals will be
/// computed against, which is the only reason the reader bothers.
fn mark_sharp_from_normals(mesh: &mut Mesh) -> usize {
    let mut sharp: Vec<HalfId> = Vec::new();
    for h in mesh.half_ids() {
        let t = mesh.twin(h).expect("live half-edge id");
        if t < h {
            continue; // once per undirected edge
        }
        if mesh.is_boundary(h) == Some(true) || mesh.is_boundary(t) == Some(true) {
            continue; // a boundary edge is implicitly sharp; the flag is for interior ones
        }
        let a1 = h;
        let a2 = mesh.next(t).expect("live half-edge id");
        let b1 = mesh.next(h).expect("live half-edge id");
        let b2 = t;
        let differs = |x: HalfId, y: HalfId| normal_bits(mesh, x) != normal_bits(mesh, y);
        if differs(a1, a2) || differs(b1, b2) {
            sharp.push(h);
        }
    }
    for h in &sharp {
        mesh.set_sharp_pair(*h, true);
    }
    sharp.len()
}

fn normal_bits(mesh: &Mesh, h: HalfId) -> Option<[u64; 3]> {
    mesh.corner_normal(h).expect("live half-edge id").map(bits3)
}

fn bits(x: f64) -> u64 {
    if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}

/// A position as the **reader's weld key**: exact bits, with `-0.0` folded onto
/// `0.0`.
///
/// `pub(crate)` and not private, because [`crate::model::bevel_edges`] refuses a
/// construction that would place two vertices at one position and "one position"
/// has to mean *what the reader will fuse*, not a second opinion about it. One
/// definition, read by both sides — the alternative is two rules that agree until
/// somebody changes one.
pub(crate) fn bits3(v: [f64; 3]) -> [u64; 3] {
    [bits(v[0]), bits(v[1]), bits(v[2])]
}

// ── primitives ─────────────────────────────────────────────────────────────
//
// These are the kernel's test fixtures *and* the default shapes the Model Editor
// will create. Every one of them:
//   * is built with `f64` positions from `inf_math`'s bit-portable trig, so the
//     geometry is identical on every target (the P14 law — a cylinder authored in
//     the editor is committed content);
//   * leaves corner normals `None` where the surface is smooth and marks the
//     edges that should stay hard as sharp, so the export-time smooth-fan rule is
//     what decides shading rather than a baked-in guess;
//   * winds every face so its geometric normal points OUT.

/// A helper that builds a mesh from a polygon soup with default corner data.
fn from_polygons(positions: &[[f64; 3]], polys: &[(Vec<usize>, Vec<[f64; 2]>)]) -> Mesh {
    let mut mesh = Mesh::new();
    let ids: Vec<VertId> = positions.iter().map(|&p| mesh.alloc_vert(p)).collect();
    let mut touched = std::collections::BTreeSet::new();
    for (loop_idx, uvs) in polys {
        let loop_verts: Vec<VertId> = loop_idx.iter().map(|&i| ids[i]).collect();
        let corners: Vec<CornerData> = uvs
            .iter()
            .map(|&uv| CornerData { uv, normal: None })
            .collect();
        touched.extend(loop_verts.iter().copied());
        mesh.add_face_raw(&loop_verts, &corners, None)
            .expect("primitive loops are well-formed by construction");
    }
    mesh.finish_patch(&touched)
        .expect("primitive topology is manifold by construction");
    mesh
}

/// A single quad of `size × size` metres in the XZ plane, facing +Y. One face,
/// one boundary loop — the kernel's open-mesh fixture.
pub fn plane(size: f64) -> Mesh {
    let s = size * 0.5;
    let positions = [[-s, 0.0, s], [s, 0.0, s], [s, 0.0, -s], [-s, 0.0, -s]];
    let uvs = vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    from_polygons(&positions, &[(vec![0, 1, 2, 3], uvs)])
}

/// An axis-aligned cube of edge `size` metres, centred on the origin: 8
/// vertices, 12 edges, 6 quad faces, every edge sharp (so export derives flat
/// per-face normals and splits back to 24 corners).
pub fn cube(size: f64) -> Mesh {
    let s = size * 0.5;
    let positions = [
        [-s, -s, -s], // 0
        [s, -s, -s],  // 1
        [s, -s, s],   // 2
        [-s, -s, s],  // 3
        [-s, s, -s],  // 4
        [s, s, -s],   // 5
        [s, s, s],    // 6
        [-s, s, s],   // 7
    ];
    let quad_uv = || vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    // Each loop is wound so its Newell normal points OUT — asserted by the
    // signed-volume gate in `export`'s tests, which is what caught these being
    // uniformly inside-out the first time round.
    let polys = vec![
        (vec![0, 1, 2, 3], quad_uv()), // −Y
        (vec![7, 6, 5, 4], quad_uv()), // +Y
        (vec![4, 5, 1, 0], quad_uv()), // −Z
        (vec![6, 7, 3, 2], quad_uv()), // +Z
        (vec![7, 4, 0, 3], quad_uv()), // −X
        (vec![5, 6, 2, 1], quad_uv()), // +X
    ];
    let mut mesh = from_polygons(&positions, &polys);
    let halfs: Vec<HalfId> = mesh.half_ids().collect();
    for h in halfs {
        mesh.set_sharp_pair(h, true);
    }
    mesh
}

/// A closed cylinder: `segments` side quads between two n-gon caps, radius and
/// height in metres. Side edges stay smooth; the two cap rims are sharp, so the
/// barrel shades round and the caps stay flat.
///
/// `segments` is clamped to at least 3 — fewer bounds no volume.
pub fn cylinder(radius: f64, height: f64, segments: usize) -> Mesh {
    let n = segments.max(3);
    let hy = height * 0.5;
    let mut positions = Vec::with_capacity(2 * n);
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64) / (n as f64);
        let (x, z) = (radius * pcos64(theta), radius * psin64(theta));
        positions.push([x, -hy, z]);
    }
    for i in 0..n {
        let theta = std::f64::consts::TAU * (i as f64) / (n as f64);
        let (x, z) = (radius * pcos64(theta), radius * psin64(theta));
        positions.push([x, hy, z]);
    }
    let b = |i: usize| i % n;
    let t = |i: usize| n + i % n;

    let mut polys: Vec<(Vec<usize>, Vec<[f64; 2]>)> = Vec::with_capacity(n + 2);
    for i in 0..n {
        let (u0, u1) = (i as f64 / n as f64, (i + 1) as f64 / n as f64);
        polys.push((
            vec![b(i), t(i), t(i + 1), b(i + 1)],
            vec![[u0, 0.0], [u0, 1.0], [u1, 1.0], [u1, 0.0]],
        ));
    }
    let cap_uv = |idx: &[usize]| -> Vec<[f64; 2]> {
        idx.iter()
            .map(|&i| {
                let p = positions[i];
                [0.5 + p[0] / (2.0 * radius), 0.5 + p[2] / (2.0 * radius)]
            })
            .collect()
    };
    let bottom: Vec<usize> = (0..n).map(b).collect();
    let top: Vec<usize> = (0..n).rev().map(t).collect();
    polys.push((bottom.clone(), cap_uv(&bottom)));
    polys.push((top.clone(), cap_uv(&top)));

    let mut mesh = from_polygons(&positions, &polys);
    // The rims: the edges shared by a side quad and a cap.
    let ids: Vec<VertId> = mesh.vert_ids().collect();
    for i in 0..n {
        for (a, c) in [(b(i), b(i + 1)), (t(i), t(i + 1))] {
            if let Some(h) = mesh.find_half(ids[a], ids[c]) {
                mesh.set_sharp_pair(h, true);
            }
        }
    }
    mesh
}

/// A torus of revolution: `major_segments` around the Y axis, `minor_segments`
/// around the tube. All quads, all smooth, **genus 1** — the fixture that makes
/// the Euler check say something (`V − E + F = 0`).
pub fn torus(
    major_radius: f64,
    minor_radius: f64,
    major_segments: usize,
    minor_segments: usize,
) -> Mesh {
    let m = major_segments.max(3);
    let n = minor_segments.max(3);
    let mut positions = Vec::with_capacity(m * n);
    for i in 0..m {
        let theta = std::f64::consts::TAU * (i as f64) / (m as f64);
        let (ct, st) = (pcos64(theta), psin64(theta));
        for j in 0..n {
            let phi = std::f64::consts::TAU * (j as f64) / (n as f64);
            let (cp, sp) = (pcos64(phi), psin64(phi));
            let r = major_radius + minor_radius * cp;
            positions.push([r * ct, minor_radius * sp, r * st]);
        }
    }
    let at = |i: usize, j: usize| (i % m) * n + (j % n);
    let mut polys = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let (u0, u1) = (i as f64 / m as f64, (i + 1) as f64 / m as f64);
            let (v0, v1) = (j as f64 / n as f64, (j + 1) as f64 / n as f64);
            polys.push((
                vec![at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)],
                vec![[u0, v0], [u0, v1], [u1, v1], [u1, v0]],
            ));
        }
    }
    from_polygons(&positions, &polys)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::validate::validate;
    use inf_mesh::SubMesh;

    /// A flat-shaded, UV-mapped cube exactly as an exporter writes one: 24
    /// vertices at 8 positions, split per face for the normal AND the UV.
    pub(crate) fn textured_cube_asset() -> MeshAsset {
        let s = 0.5f32;
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            (
                [0.0, -1.0, 0.0],
                [[-s, -s, -s], [-s, -s, s], [s, -s, s], [s, -s, -s]],
            ),
            (
                [0.0, 1.0, 0.0],
                [[-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s]],
            ),
            (
                [0.0, 0.0, -1.0],
                [[-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s]],
            ),
            (
                [0.0, 0.0, 1.0],
                [[s, -s, s], [-s, -s, s], [-s, s, s], [s, s, s]],
            ),
            (
                [-1.0, 0.0, 0.0],
                [[-s, -s, s], [-s, -s, -s], [-s, s, -s], [-s, s, s]],
            ),
            (
                [1.0, 0.0, 0.0],
                [[s, -s, -s], [s, -s, s], [s, s, s], [s, s, -s]],
            ),
        ];
        let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (normal, corners) in faces {
            let base = vertices.len() as u32;
            for (k, position) in corners.iter().enumerate() {
                vertices.push(MeshVertex {
                    position: *position,
                    normal,
                    uv: uvs[k],
                    tangent: inf_mesh::TANGENT_PLACEHOLDER,
                });
            }
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        MeshAsset::new(
            vec![SubMesh {
                name: "cube".into(),
                vertices,
                indices,
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["Default".into()],
        )
    }

    #[test]
    fn the_seam_reconstruction_welds_positions_and_keeps_corners() {
        let asset = textured_cube_asset();
        assert_eq!(asset.vertex_count(), 24, "the fixture is split per corner");
        let imported = from_mesh_asset(&asset).unwrap();
        assert_eq!(imported.report.source_vertices, 24);
        assert_eq!(imported.report.welded_positions, 8, "8 distinct places");
        assert_eq!(imported.mesh.vert_count(), 8);
        assert_eq!(imported.mesh.face_count(), 12, "12 triangles, not welded");
        assert_eq!(imported.mesh.edge_count(), 18, "V−E+F = 8−18+12 = 2");
        assert_eq!(validate(&imported.mesh), Ok(()));

        // Every corner kept its own attributes: 8 vertices carry 36 corners with
        // 24 distinct (position, uv, normal) combinations — the split survives.
        let mut combos = std::collections::BTreeSet::new();
        for h in imported.mesh.half_ids() {
            if imported.mesh.is_boundary(h) == Some(true) {
                continue;
            }
            let v = imported.mesh.origin(h).unwrap();
            let p = bits3(imported.mesh.position(v).unwrap().to_array());
            let uv = imported.mesh.corner_uv(h).unwrap();
            let n = imported.mesh.corner_normal(h).unwrap().map(bits3);
            combos.insert((p, bits(uv[0]), bits(uv[1]), n));
        }
        assert_eq!(
            combos.len(),
            24,
            "24 distinct corner attribute combinations"
        );
    }

    #[test]
    fn a_one_ulp_seam_does_not_weld_and_the_counter_says_so() {
        // The exact-weld ruling stands (an epsilon weld is not transitive and
        // would re-topologize shipped assets), so this test does NOT assert that
        // the mesh comes back closed. It asserts that when a seam misses by one
        // ULP the author is TOLD — which is the whole point of preferring a
        // counter to a tolerance.
        let closed = from_mesh_asset(&textured_cube_asset()).unwrap();
        assert_eq!(
            closed.report.boundary_edges, 0,
            "the friendly fixture is a closed solid"
        );

        // Nudge the +X FACE's own four corners by one ULP. The same four
        // positions are still written exactly by the ±Y/±Z faces that share
        // them, so the seam misses by one ULP — the realistic shape of the
        // hazard, and the one an epsilon weld is usually proposed for.
        let mut asset = textured_cube_asset();
        let plus_x = 20..24; // the sixth face's corners, in fixture order
        for i in plus_x {
            let p = &mut asset.submeshes[0].vertices[i].position;
            p[0] = f32::from_bits(p[0].to_bits() + 1);
        }
        let split = from_mesh_asset(&asset).unwrap();
        assert_eq!(
            split.report.welded_positions, 12,
            "the four +X corners no longer weld with their twins"
        );
        assert!(
            split.report.boundary_edges > 0,
            "a mesh the author believes is closed must not arrive silently fragmented"
        );
        assert_eq!(
            split.report.boundary_edges, 8,
            "the detached quad's four border edges, plus the four around the hole \
             it left — the author is told the solid is open, and where"
        );
        assert_eq!(validate(&split.mesh), Ok(()), "fragmented, but still valid");
    }

    #[test]
    fn a_flat_shaded_import_marks_every_shared_edge_sharp() {
        let imported = from_mesh_asset(&textured_cube_asset()).unwrap();
        // 18 edges; the 6 face diagonals are interior to a quad and share a
        // normal, the 12 cube edges do not.
        assert_eq!(imported.report.sharp_edges, 12);
    }

    #[test]
    fn a_bowtie_is_split_into_two_vertices_rather_than_refused() {
        // Two triangles meeting at exactly one position.
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            ..Default::default()
        };
        let asset = MeshAsset::new(
            vec![SubMesh {
                name: "bowtie".into(),
                vertices: vec![
                    v(0.0, 0.0),
                    v(1.0, 0.0),
                    v(1.0, 1.0),
                    v(0.0, 0.0),
                    v(-1.0, -1.0),
                    v(-1.0, 0.0),
                ],
                indices: vec![0, 1, 2, 3, 4, 5],
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        );
        let imported = from_mesh_asset(&asset).unwrap();
        assert_eq!(imported.report.welded_positions, 5, "the apex welds first");
        assert_eq!(imported.report.fan_splits, 1, "and is then split back");
        assert_eq!(imported.mesh.vert_count(), 6);
        assert_eq!(validate(&imported.mesh), Ok(()));
    }

    /// **A skinned submesh now reads** (P24.2) — the P23 refusal is closed.
    ///
    /// Three claims: the weights arrive, the binding's joint count is the
    /// tightest one the FILE supports (a bare weight stream says no more), and
    /// the imported mesh is valid — which is the check that would fire if an
    /// index or a normalization slipped through the weld.
    #[test]
    fn a_skinned_submesh_reads_its_weights_into_the_channel() {
        let mut asset = textured_cube_asset();
        let n = asset.submeshes[0].vertices.len();
        // Every vertex on joint 3, and the `+x` half sharing with joint 1 — so
        // the channel is not a constant and "the weights arrived" is falsifiable.
        //
        // Keyed on the POSITION, not on the index: a `MeshAsset` cube is 24
        // vertices for 8 corners (the UV split), and an index-keyed pattern gives
        // the split copies of one corner different weights. That is a real
        // conflict, `skin_conflicts` counts it correctly, and it is a defect in
        // the fixture rather than in the reader — measured, on the first run of
        // this test.
        asset.submeshes[0].skin = (0..n)
            .map(|i| inf_mesh::VertexSkin {
                joints: [3, 1, 0, 0],
                weights: if asset.submeshes[0].vertices[i].position[0] > 0.0 {
                    [0.5, 0.5, 0.0, 0.0]
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                },
            })
            .collect();
        let import = from_mesh_asset(&asset).expect("a skinned asset now reads");
        assert_eq!(
            import.mesh.skin_binding(),
            Some(SkinBinding {
                skeleton: None,
                joints: 4
            }),
            "the binding is as tight as the file, and names no skeleton"
        );
        assert_eq!(validate(&import.mesh), Ok(()));
        assert_eq!(import.report.skin_conflicts, 0);
        let weighted = import
            .mesh
            .vert_ids()
            .filter(|&v| import.mesh.vert_weights(v).unwrap().weight_of(1) > 0.0)
            .count();
        assert!(
            weighted > 0 && weighted < import.mesh.vert_count(),
            "{weighted} of {} vertices share joint 1 — a constant channel would \
             prove nothing",
            import.mesh.vert_count()
        );
    }

    /// A skin stream that is not index-aligned to its vertices is refused by
    /// name — the two halves of one submesh came from different places.
    #[test]
    fn a_ragged_skin_stream_is_refused_by_name() {
        let mut asset = textured_cube_asset();
        asset.submeshes[0].skin = vec![Default::default(); 3];
        let vertices = asset.submeshes[0].vertices.len();
        assert_eq!(
            from_mesh_asset(&asset),
            Err(ImportError::SkinLengthMismatch {
                submesh: 0,
                vertices,
                skin: 3
            })
        );
    }

    /// **The weld's skin advisory is real, and it is counted rather than
    /// averaged.**
    ///
    /// Two source vertices at one position with different influences: the first
    /// wins, the disagreement is reported, and the mesh is still valid. The
    /// control below is the same fixture with matching weights, which reports
    /// zero — so the counter is measuring the conflict and not the split.
    #[test]
    fn split_copies_that_disagree_about_their_weights_are_counted() {
        let base = textured_cube_asset();
        let n = base.submeshes[0].vertices.len();
        let make = |disagree: bool| {
            let mut asset = base.clone();
            asset.submeshes[0].skin = (0..n)
                .map(|i| inf_mesh::VertexSkin {
                    joints: [if disagree && i % 2 == 1 { 2 } else { 1 }, 0, 0, 0],
                    weights: [1.0, 0.0, 0.0, 0.0],
                })
                .collect();
            from_mesh_asset(&asset).expect("reads")
        };
        let clean = make(false);
        assert_eq!(clean.report.skin_conflicts, 0, "the control must be clean");
        let messy = make(true);
        assert!(
            messy.report.skin_conflicts > 0,
            "a disagreement at a welded position must be reported"
        );
        assert_eq!(validate(&messy.mesh), Ok(()));
    }

    #[test]
    fn a_newer_schema_is_refused() {
        let mut asset = textured_cube_asset();
        asset.schema_version = MeshAsset::CURRENT_VERSION + 1;
        assert!(matches!(
            from_mesh_asset(&asset),
            Err(ImportError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn a_duplicated_triangle_is_a_non_manifold_edge_refusal() {
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            ..Default::default()
        };
        let asset = MeshAsset::new(
            vec![SubMesh {
                name: "double".into(),
                vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0)],
                indices: vec![0, 1, 2, 0, 1, 2],
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        );
        assert!(matches!(
            from_mesh_asset(&asset),
            Err(ImportError::NonManifoldEdge { .. })
        ));
    }

    #[test]
    fn degenerate_triangles_are_counted_not_fatal() {
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            ..Default::default()
        };
        let asset = MeshAsset::new(
            vec![SubMesh {
                name: "sliver".into(),
                vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0)],
                // One real triangle plus one with a repeated corner.
                indices: vec![0, 1, 2, 0, 1, 1],
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        );
        let imported = from_mesh_asset(&asset).unwrap();
        assert_eq!(imported.report.degenerate_triangles_skipped, 1);
        assert_eq!(imported.mesh.face_count(), 1);
    }

    #[test]
    fn primitives_are_deterministic_across_two_runs() {
        assert_eq!(plane(2.0).encoded(), plane(2.0).encoded());
        assert_eq!(cube(1.0).encoded(), cube(1.0).encoded());
        assert_eq!(
            cylinder(0.5, 2.0, 16).encoded(),
            cylinder(0.5, 2.0, 16).encoded()
        );
        assert_eq!(
            torus(1.0, 0.25, 12, 8).encoded(),
            torus(1.0, 0.25, 12, 8).encoded()
        );
    }

    #[test]
    fn cylinder_and_torus_have_the_counts_their_construction_implies() {
        let c = cylinder(0.5, 2.0, 16);
        assert_eq!(c.vert_count(), 32);
        assert_eq!(c.face_count(), 18, "16 side quads + 2 caps");
        assert_eq!(c.edge_count(), 48);
        assert!(c.half_ids().all(|h| c.is_boundary(h) == Some(false)));

        let t = torus(1.0, 0.25, 12, 8);
        assert_eq!(t.vert_count(), 96);
        assert_eq!(t.face_count(), 96);
        assert_eq!(t.edge_count(), 192);
    }

    #[test]
    fn the_cylinder_rim_is_sharp_and_its_barrel_is_not() {
        let c = cylinder(0.5, 2.0, 8);
        let sharp = c
            .half_ids()
            .filter(|&h| c.is_sharp(h) == Some(true))
            .count();
        assert_eq!(sharp, 32, "2 rims × 8 edges × 2 halves");
    }
}
