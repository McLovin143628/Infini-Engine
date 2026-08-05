//! **The writer `inf-mesh` never had**: kernel mesh → [`inf_mesh::MeshAsset`].
//!
//! Export is the inverse of the import weld (see [`crate::build`]): it splits one
//! kernel vertex back into one rendering vertex per distinct corner-attribute
//! combination, triangulates the n-gons, and fills in the two derived streams a
//! `MeshVertex` carries — normals and tangents.
//!
//! # Normals: authored wins, derived is a smooth fan
//!
//! A corner's normal is `Some` (**authored** — imported, or set by
//! [`crate::ops::Op::SetCornerNormal`]) or `None` (**derived**). Under the default
//! [`NormalPolicy::PreserveAuthored`], authored normals are written out verbatim
//! and derived ones are computed as the **area-weighted average of the corner's
//! smooth fan**: the run of faces reachable around the vertex without crossing a
//! sharp edge or a boundary. Area weighting is free — the un-normalized Newell
//! normal of a polygon already has twice its area as its length — and the fan is
//! summed in **ascending face id** order so that every corner in one fan gets a
//! bit-identical result and they therefore collapse to one output vertex.
//!
//! [`NormalPolicy::Recompute`] ignores authored normals entirely. Two reasons it
//! is **not** the default, and the second one is measured rather than assumed:
//!
//! * It would make [`crate::ops::Op::SetCornerNormal`] an op whose effect can
//!   never be observed in the thing the kernel writes.
//! * It is **not a round-trip fixed point on curved geometry**. A derived normal
//!   sums the corner's smooth fan; on the first export that fan is made of
//!   n-gons and after the round trip the same surface is triangles, so the sums
//!   differ. Flat, all-sharp geometry is unaffected (the fan is one face either
//!   way), which is why `plane` and `cube` are stable under `Recompute` and
//!   `cylinder` and `torus` are not — pinned by
//!   `the_recompute_policy_is_a_fixed_point_only_on_flat_geometry`.
//!
//! It exists for the "bake the shading I can see" case P23.5 will want.
//!
//! # Tangents: MikkTSpace-class, written here, deterministic
//!
//! No dependency and no vendored C. Per triangle, the standard UV-derivative
//! solve gives a tangent and a bitangent; both are accumulated per **output**
//! vertex over the triangles in emission order; the tangent is then
//! Gram-Schmidt-orthogonalized against the corner normal and its handedness `w`
//! taken from `sign(dot(cross(N, T), B))` — the glTF convention `MeshVertex`
//! already documents.
//!
//! Determinism comes from three properties, all of them structural: the
//! accumulation order is fixed (triangles in emission order, corners in loop
//! order), every intermediate is `f64`, and the only transcendental involved is
//! `sqrt`, which IEEE-754 specifies exactly — unlike `sin` or `cbrt`, whose
//! libm implementations differ between targets (the P14 law). Two runs on two
//! machines produce the same bytes.
//!
//! A triangle whose UV triangle has zero area contributes nothing (it carries no
//! direction). A vertex that ends with no usable accumulation — a mesh with no
//! UVs at all, or a corner where every incident triangle is UV-degenerate — gets
//! the constant [`TANGENT_FALLBACK`], the same `[1, 0, 0, 1]` the importers write
//! when a source file has no tangents.
//!
//! # Triangulation: ear clipping, always
//!
//! The kernel holds n-gons; `MeshAsset` holds triangles. Ear clipping in the
//! face's own plane handles **non-convex** polygons, and on a convex one it
//! produces exactly the fan a fan would. It is deterministic — the *first* valid
//! ear in loop order is clipped, never the "best" one. A polygon so degenerate
//! that no ear exists (self-intersecting, or entirely collinear) falls back to a
//! fan and is **counted** in [`ExportReport::fan_fallbacks`] rather than silently
//! producing inverted triangles.
//!
//! # `meshopt` is opt-in, and the flag says why
//!
//! [`ExportOptions::optimize`] is **off by default**, and that is the whole
//! point: an edit session's export must be stable, so that opening a mesh,
//! saving it, and opening it again is a no-op. `meshopt` is not cross-platform
//! (the P18 law: identical input, different byte counts on `x86_64-msvc` versus
//! `aarch64-apple-darwin`), so turning it on gives up byte-reproducibility across
//! machines in exchange for vertex-cache locality. That is the right trade for a
//! *final* export and the wrong one for a round trip, so it is a decision the
//! caller makes explicitly.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use inf_mesh::{MeshAsset, MeshVertex, SubMesh};
use serde::{Deserialize, Serialize};

use crate::topo::{FaceId, HalfId, Mesh, VertId};

/// The tangent written when nothing better can be derived: `[1, 0, 0, 1]`.
pub const TANGENT_FALLBACK: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// Where a written corner normal comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NormalPolicy {
    /// Authored corner normals verbatim; derived ones from the smooth fan.
    #[default]
    PreserveAuthored,
    /// Every normal from the smooth fan, authored values ignored.
    Recompute,
}

/// How to write the asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExportOptions {
    pub normals: NormalPolicy,
    /// Run `meshopt`'s weld + vertex-cache + vertex-fetch pass over each
    /// submesh. **Off by default**, and non-deterministic across platforms — see
    /// the module docs.
    pub optimize: bool,
}

/// What the writer had to do — advisories, not failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExportReport {
    pub submeshes: usize,
    pub vertices: usize,
    pub triangles: usize,
    /// Faces that had no ear and fell back to a fan (self-intersecting or
    /// collinear). Non-zero means the source polygon was degenerate.
    pub fan_fallbacks: usize,
    /// Output vertices that took [`TANGENT_FALLBACK`] because no incident
    /// triangle carried a usable UV gradient.
    pub fallback_tangents: usize,
    /// Whether the non-deterministic `meshopt` pass ran.
    pub optimized: bool,
    /// **The round-trip hazard.** How many kernel vertices share a position with
    /// another kernel vertex.
    ///
    /// The kernel distinguishes two vertices at the same place (a split bowtie,
    /// a merge that left geometry coincident, a modelling state passing through
    /// itself). A `MeshAsset` cannot: it carries positions, and
    /// [`crate::build::from_mesh_asset`] welds them **exactly**. So a non-zero
    /// count means writing and reading back is not the identity — at best the two
    /// vertices fuse, at worst the fused topology has an edge used twice in one
    /// direction and the read is *refused* as non-manifold.
    ///
    /// This is reported rather than fixed because both fixes are worse: nudging
    /// the positions falsifies the model, and refusing the export would make a
    /// legal intermediate modelling state unsaveable. It is an advisory in the
    /// P16 sense — the caller (P23.6's save) can surface it, and the alternative
    /// is a save that silently changes the mesh on the next open.
    pub coincident_vertices: usize,
    /// Triangulation diagonals that had to repeat an edge the mesh already has.
    ///
    /// The other way an exported asset can come back unreadable: an n-gon's ear
    /// diagonal running between two vertices that are *already* joined elsewhere
    /// puts three or four faces on one edge in the flattened soup. The ear
    /// chooser prefers a diagonal that is new, so this is normally zero; when a
    /// polygon has no such ear it is counted here rather than left to be
    /// discovered on the next open.
    pub reused_diagonals: usize,
    /// Written vertices carrying a non-finite position, normal or UV.
    ///
    /// # Why this is a counter here and a REFUSAL in [`crate::ops`]
    ///
    /// The read path refuses non-finite data (it is a
    /// [`crate::validate::Violation`]), and symmetry does argue for the writer
    /// refusing too. The kernel therefore closes the door where closing it is
    /// free: `AddVertex`, `TranslateVerts`, `SetCornerUv` and `SetCornerNormal`
    /// **refuse** a non-finite value outright, so no *edit* can produce this.
    ///
    /// What remains is data the kernel did not author — a `MeshAsset` whose
    /// positions were already NaN when the author's glTF was written.
    /// [`crate::build::from_mesh_asset`] deliberately does not police attribute
    /// *values* (it preserves them bit-for-bit so the round trip is exact), so
    /// such a value can reach the writer. Refusing there would mean an author
    /// who opened a bad file cannot save their work at all, over a value they
    /// did not create and may not be able to find. So it is counted, loudly, for
    /// P23.6's save path to surface — the P16 advisory doctrine, and the same
    /// call as `coincident_vertices` above.
    ///
    /// A non-zero count on a mesh this crate built is a bug in this crate.
    pub non_finite_written: usize,
    /// Written vertices whose normal is not unit length (tolerance 1e-3).
    ///
    /// Same door, same reasoning: `SetCornerNormal` refuses a non-unit authored
    /// normal, derived normals are normalized by construction, so this can only
    /// be an imported value passing through — a `[0, 5, 0]` in someone's asset.
    pub non_unit_normals_written: usize,
}

/// Write a kernel mesh as a `.inf_mesh` payload.
///
/// The produced asset is schema v2 — **the existing format, unchanged**. This
/// batch adds a writer, not a version.
pub fn to_mesh_asset(mesh: &Mesh, opts: &ExportOptions) -> (MeshAsset, ExportReport) {
    let mut report = ExportReport {
        optimized: opts.optimize,
        ..Default::default()
    };
    // Filled as vertices are interned, so the coincidence advisory measures what
    // was WRITTEN, in f32 — see `count_coincident`.
    let mut written: BTreeSet<([u32; 3], VertId)> = BTreeSet::new();

    // One submesh per material slot, ascending (`None` first — `Option`'s own
    // ordering, so the grouping needs no special case for "unassigned").
    let mut by_slot: BTreeMap<Option<u32>, Vec<FaceId>> = BTreeMap::new();
    for f in mesh.face_ids() {
        by_slot
            .entry(mesh.face_slot(f).expect("live face id"))
            .or_default()
            .push(f);
    }

    // Directed edges written so far, across EVERY submesh: the reader welds the
    // whole asset at once, so a diagonal that repeats one written under a
    // different material slot is just as unreadable.
    let mut emitted: BTreeSet<(VertId, VertId)> = BTreeSet::new();
    let mut submeshes = Vec::with_capacity(by_slot.len());
    for (slot, faces) in by_slot {
        let (vertices, indices, fallbacks) =
            build_submesh(mesh, &faces, opts, &mut report, &mut emitted, &mut written);
        report.fan_fallbacks += fallbacks;
        if indices.is_empty() {
            continue;
        }
        let name = mesh
            .slot_name(slot)
            .map(str::to_owned)
            .unwrap_or_else(|| match slot {
                Some(i) => format!("slot_{i}"),
                None => "mesh".to_string(),
            });

        // The one non-deterministic step in the crate, and it is opt-in. On
        // wasm32 `inf_mesh::optimize` does not exist (the meshopt C build is
        // host-only), so the flag is simply inert there.
        #[cfg(not(target_arch = "wasm32"))]
        let (vertices, indices) = if opts.optimize {
            inf_mesh::optimize(vertices, indices)
        } else {
            (vertices, indices)
        };

        report.vertices += vertices.len();
        report.triangles += indices.len() / 3;
        submeshes.push(SubMesh {
            name,
            vertices,
            indices,
            material_slot: slot,
            // The kernel has no skinning model, and import refuses a skinned
            // asset rather than dropping weights — so an exported submesh is
            // rigid by construction, never by omission.
            skin: Vec::new(),
        });
    }
    report.submeshes = submeshes.len();
    report.coincident_vertices = count_coincident(&written);

    // `MeshAsset::new` recomputes the bounds from the written positions.
    (
        MeshAsset::new(submeshes, mesh.material_slots().to_vec()),
        report,
    )
}

/// Vertices that collide **in the domain the reader actually welds in** — see
/// [`ExportReport::coincident_vertices`].
///
/// Two things this gets right that the obvious version does not:
///
/// * **`f32`, not `f64`.** The weld compares the positions in the *asset*, and
///   an asset position is `f32`. Two vertices 1e-9 m apart are distinct in the
///   kernel and bit-equal once written, so an `f64` comparison reports zero
///   while the reader fuses them and refuses the result as non-manifold — the
///   advisory reading clean at exactly the moment it is needed.
/// * **Only what was written.** Isolated vertices, and vertices whose every
///   incident face was skipped, never reach the payload, so counting them
///   inflates a hazard that cannot occur.
///
/// Input is the `(written f32 position, kernel vertex)` pairs the interning pass
/// produced; the count is the number of distinct kernel vertices sharing a
/// written position with another.
fn count_coincident(written: &BTreeSet<([u32; 3], VertId)>) -> usize {
    let mut by_position: BTreeMap<[u32; 3], BTreeSet<VertId>> = BTreeMap::new();
    for (pos, v) in written {
        by_position.entry(*pos).or_default().insert(*v);
    }
    by_position
        .values()
        .filter(|verts| verts.len() > 1)
        .map(BTreeSet::len)
        .sum()
}

fn bits32(p: [f32; 3]) -> [u32; 3] {
    // `-0.0` folded onto `+0.0`: the reader's weld compares the f64 promotion of
    // these bits, where the two compare EQUAL, so treating them as distinct here
    // would under-report.
    let one = |x: f32| if x == 0.0 { 0 } else { x.to_bits() };
    [one(p[0]), one(p[1]), one(p[2])]
}

/// The corner key an output vertex is deduplicated by: the vertex it sits on,
/// the **f32** bits of the attributes that will actually be written (so the
/// split is exactly the split the payload needs and not one bit finer), and the
/// corner's **UV handedness**.
///
/// Handedness is in the key because a tangent frame has one, and it is exactly
/// what MikkTSpace splits on. Two faces meeting at a vertex with mirrored UV
/// islands can carry the *same* uv and the *same* normal while winding opposite
/// ways in UV space; without handedness in the key they intern to one output
/// vertex whose accumulated bitangent is the sum of two opposing contributions,
/// so the written `w` depends on which triangles happened to be emitted first.
/// That is a seam that flips its normal map, and it is triangulation-dependent —
/// the worst kind of bug to chase.
type CornerKey = (VertId, [u32; 2], [u32; 3], i8);

/// The sign of a **triangle's** signed area in UV space: `+1`, `-1`, or `0` when
/// it carries no UV area at all (an unwrapped or degenerate triangle, where
/// there is no handedness to disagree about and no split is wanted).
///
/// Per *triangle*, not per face, and computed from the **f32** UVs, and both of
/// those are load-bearing.
///
/// * *Per triangle* is what MikkTSpace splits on, and it is the only version
///   that survives the round trip, because the round trip replaces every n-gon
///   with its triangles. Keying on the parent face's orientation made
///   `export_is_a_fixed_point` fail the moment an n-gon's UV loop summed to zero
///   while its triangles did not: 16 written vertices became 18 on the reread.
/// * *From f32* because this sign goes into the corner key, so it decides how
///   many vertices are written — and on the reread it will be recomputed from
///   the f32 values in the payload. A sign taken from the f64 kernel UVs and a
///   sign taken from their f32 images disagree whenever a triangle's UV area is
///   smaller than the rounding, and the two passes then split differently. Same
///   lesson as `count_coincident`: measure in the domain the reader will use.
fn uv_orientation(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> i8 {
    let (a, b, c) = (
        [a[0] as f64, a[1] as f64],
        [b[0] as f64, b[1] as f64],
        [c[0] as f64, c[1] as f64],
    );
    let twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    match twice_area.partial_cmp(&0.0) {
        Some(std::cmp::Ordering::Greater) => 1,
        Some(std::cmp::Ordering::Less) => -1,
        _ => 0,
    }
}

/// A corner's UV as it will be WRITTEN.
fn corner_uv32(mesh: &Mesh, h: HalfId) -> [f32; 2] {
    let uv = mesh.corner_uv(h).expect("live half-edge id");
    [uv[0] as f32, uv[1] as f32]
}

fn build_submesh(
    mesh: &Mesh,
    faces: &[FaceId],
    opts: &ExportOptions,
    report: &mut ExportReport,
    emitted: &mut BTreeSet<(VertId, VertId)>,
    written: &mut BTreeSet<([u32; 3], VertId)>,
) -> (Vec<MeshVertex>, Vec<u32>, usize) {
    let mut key_to_index: BTreeMap<CornerKey, u32> = BTreeMap::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut fallbacks = 0usize;

    for f in faces {
        let halfs = mesh.face_loop(*f).expect("live face id");
        let loop_verts: Vec<VertId> = halfs
            .iter()
            .map(|&h| mesh.origin(h).expect("live half-edge id"))
            .collect();
        let loop_positions: Vec<DVec3> = loop_verts
            .iter()
            .map(|&v| mesh.position(v).expect("live vertex id"))
            .collect();
        let n = halfs.len();
        let seen_edges = &*emitted;
        let taken = |a: VertId, b: VertId| {
            mesh.find_half(a, b).is_some()
                || mesh.find_half(b, a).is_some()
                || seen_edges.contains(&(a, b))
                || seen_edges.contains(&(b, a))
        };
        let forbidden = |i: usize, j: usize| taken(loop_verts[i], loop_verts[j]);
        let t = triangulate_with(&loop_positions, &forbidden);
        let (tris, fell_back) = (t.tris, t.fan_fallback);
        if fell_back {
            fallbacks += 1;
        }
        // Count and record the diagonals this face introduced. Done here rather
        // than inside the clipper because the fan FALLBACK emits diagonals the
        // clipper never got to choose — and those are exactly the ones that made
        // an exported asset unreadable with both advisories reading zero.
        // A pair adjacent in the loop is the face's own edge, not a diagonal.
        let mut new_diagonals: Vec<(VertId, VertId)> = Vec::new();
        for tri in &tris {
            for k in 0..3 {
                let (i, j) = (tri[k], tri[(k + 1) % 3]);
                if (j + n - i) % n == 1 || (i + n - j) % n == 1 {
                    continue;
                }
                let (a, b) = (loop_verts[i], loop_verts[j]);
                if taken(a, b) {
                    report.reused_diagonals += 1;
                }
                new_diagonals.push((a, b));
            }
        }
        emitted.extend(new_diagonals);

        // Corners are interned in the order the INDEX BUFFER first names them,
        // not in face-loop order. That is what makes an exported asset a fixed
        // point of the round trip: `from_mesh_asset` welds in index order, so
        // re-exporting reproduces this vertex buffer exactly. Interning in loop
        // order instead put the ear-clipper's first triangle out of step with the
        // vertex order and cost a byte-identical `export∘import∘export`.
        for tri in tris {
            let hand = uv_orientation(
                corner_uv32(mesh, halfs[tri[0]]),
                corner_uv32(mesh, halfs[tri[1]]),
                corner_uv32(mesh, halfs[tri[2]]),
            );
            for local in tri {
                let h = halfs[local];
                let v = mesh.origin(h).expect("live half-edge id");
                let uv = mesh.corner_uv(h).expect("live half-edge id");
                let n = corner_normal(mesh, h, opts.normals);
                let uv32 = [uv[0] as f32, uv[1] as f32];
                let n32 = [n.x as f32, n.y as f32, n.z as f32];
                let key: CornerKey = (
                    v,
                    [uv32[0].to_bits(), uv32[1].to_bits()],
                    [n32[0].to_bits(), n32[1].to_bits(), n32[2].to_bits()],
                    hand,
                );
                let next = positions.len() as u32;
                let idx = *key_to_index.entry(key).or_insert(next);
                if idx == next {
                    let p = mesh.position(v).expect("live vertex id");
                    let p32 = [p.x as f32, p.y as f32, p.z as f32];
                    // M6: the write path counts what it cannot refuse. See the
                    // `ExportReport` field docs for why this is a counter here
                    // and a REFUSAL in `ops`.
                    if !p32.iter().all(|c| c.is_finite())
                        || !n32.iter().all(|c| c.is_finite())
                        || !uv32.iter().all(|c| c.is_finite())
                    {
                        report.non_finite_written += 1;
                    }
                    let len2: f32 = n32.iter().map(|c| c * c).sum();
                    // NaN-safe on purpose: a non-finite normal is also non-unit.
                    if len2.is_nan() || (len2 - 1.0).abs() > 1e-3 {
                        report.non_unit_normals_written += 1;
                    }
                    written.insert((bits32(p32), v));
                    positions.push(p32);
                    normals.push(n32);
                    uvs.push(uv32);
                }
                indices.push(idx);
            }
        }
    }

    let (tangents, fallback_count) = tangents(&positions, &normals, &uvs, &indices);
    report.fallback_tangents += fallback_count;

    let vertices: Vec<MeshVertex> = (0..positions.len())
        .map(|i| MeshVertex {
            position: positions[i],
            normal: normals[i],
            uv: uvs[i],
            tangent: tangents[i],
        })
        .collect();
    (vertices, indices, fallbacks)
}

/// The normal written for one corner, under a policy.
fn corner_normal(mesh: &Mesh, h: HalfId, policy: NormalPolicy) -> DVec3 {
    if policy == NormalPolicy::PreserveAuthored {
        if let Some(n) = mesh.corner_normal(h).expect("live half-edge id") {
            return DVec3::from_array(n);
        }
    }
    derived_normal(mesh, h)
}

/// The area-weighted average of the corner's **smooth fan** — the run of faces
/// reachable around the vertex without crossing a sharp edge or a boundary.
///
/// Summed in ascending face id so that every corner in one fan produces
/// bit-identical output and they collapse to a single written vertex.
fn derived_normal(mesh: &Mesh, h: HalfId) -> DVec3 {
    let own = match mesh.face_of(h).expect("live half-edge id") {
        Some(f) => f,
        None => return DVec3::Y,
    };
    let mut fan: BTreeSet<FaceId> = [own].into_iter().collect();
    let budget = mesh.half_count() + 1;

    // Forwards: cross the edge `cur` itself into `next(twin(cur))`.
    let mut cur = h;
    for _ in 0..budget {
        if mesh.is_sharp(cur) == Some(true) {
            break;
        }
        let t = mesh.twin(cur).expect("live half-edge id");
        if mesh.is_boundary(t) == Some(true) {
            break;
        }
        let step = mesh.next(t).expect("live half-edge id");
        if step == h {
            break;
        }
        if let Some(Some(f)) = mesh.face_of(step) {
            fan.insert(f);
        }
        cur = step;
    }
    // Backwards: cross the edge `prev(cur)` into its twin.
    let mut cur = h;
    for _ in 0..budget {
        let p = mesh.prev(cur).expect("live half-edge id");
        if mesh.is_sharp(p) == Some(true) {
            break;
        }
        let t = mesh.twin(p).expect("live half-edge id");
        if mesh.is_boundary(t) == Some(true) || t == h {
            break;
        }
        if let Some(Some(f)) = mesh.face_of(t) {
            fan.insert(f);
        }
        cur = t;
    }

    let mut sum = DVec3::ZERO;
    for f in fan {
        sum += newell(mesh, f);
    }
    let len = sum.length();
    if len > 1e-18 {
        return sum / len;
    }
    // A fan that cancels out (a degenerate neighbourhood): fall back to the
    // corner's own face, then to +Y so the written normal is always finite.
    let own_n = newell(mesh, own);
    let own_len = own_n.length();
    if own_len > 1e-18 {
        own_n / own_len
    } else {
        DVec3::Y
    }
}

/// The Newell normal of a face: robust for non-planar polygons, and its length
/// is twice the polygon's area — which is exactly the weight a smooth-fan
/// average wants.
fn newell(mesh: &Mesh, f: FaceId) -> DVec3 {
    let halfs = mesh.face_loop(f).expect("live face id");
    let pts: Vec<DVec3> = halfs
        .iter()
        .map(|&h| {
            mesh.position(mesh.origin(h).expect("live half-edge id"))
                .expect("live vertex id")
        })
        .collect();
    newell_of(&pts)
}

fn newell_of(pts: &[DVec3]) -> DVec3 {
    let n = pts.len();
    let mut acc = DVec3::ZERO;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        acc.x += (a.y - b.y) * (a.z + b.z);
        acc.y += (a.z - b.z) * (a.x + b.x);
        acc.z += (a.x - b.x) * (a.y + b.y);
    }
    acc
}

/// An orthonormal basis `(u, v)` of the plane with normal `n`, chosen so that
/// `u × v = n` — which makes the 2D shoelace area agree in sign with the 3D
/// winding, so a CCW polygon projects to a CCW polygon.
fn plane_basis(n: DVec3) -> (DVec3, DVec3) {
    let axis = if n.x.abs() <= n.y.abs() && n.x.abs() <= n.z.abs() {
        DVec3::X
    } else if n.y.abs() <= n.z.abs() {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let u = n.cross(axis);
    let u = if u.length() > 1e-18 {
        u.normalize()
    } else {
        DVec3::X
    };
    (u, n.cross(u))
}

/// The result of triangulating one face loop.
struct Triangulation {
    /// Triangles as *local* indices into the face loop.
    tris: Vec<[usize; 3]>,
    /// The polygon had no ear at all (self-intersecting or collinear) and was
    /// fanned instead.
    fan_fallback: bool,
}

/// Ear-clip a face loop into triangles of *local* indices.
///
/// `forbidden(i, j)` reports that the diagonal between local corners `i` and `j`
/// would duplicate an edge that already exists — either elsewhere in the mesh or
/// in a triangle this export has already emitted. A duplicated **directed** edge
/// is exactly what `from_mesh_asset` refuses as non-manifold, so a writer that
/// ignored this would produce assets its own reader cannot open. It is a
/// *preference*, not a hard constraint: when no unforbidden ear exists, the best
/// available ear is taken and counted.
fn triangulate_with(pts: &[DVec3], forbidden: &dyn Fn(usize, usize) -> bool) -> Triangulation {
    let n = pts.len();
    let none = Triangulation {
        tris: Vec::new(),
        fan_fallback: false,
    };
    if n < 3 {
        return none;
    }
    if n == 3 {
        return Triangulation {
            tris: vec![[0, 1, 2]],
            ..none
        };
    }
    let normal = newell_of(pts);
    let len = normal.length();
    if len.is_nan() || len <= 1e-18 {
        return Triangulation {
            tris: fan(n),
            fan_fallback: true,
        };
    }
    let (u, v) = plane_basis(normal / len);
    let p2: Vec<(f64, f64)> = pts.iter().map(|p| (p.dot(u), p.dot(v))).collect();

    let mut remaining: Vec<usize> = (0..n).collect();
    let mut tris = Vec::with_capacity(n - 2);
    let mut fell_back = false;
    while remaining.len() > 3 {
        let m = remaining.len();
        // Only a REFLEX vertex of the remaining polygon can invalidate an ear,
        // and it invalidates it by lying inside OR ON the candidate triangle.
        //
        // Both halves of that sentence are load-bearing, and the first version
        // of this clipper got both wrong. It tested every vertex for STRICT
        // containment, so a reflex vertex sitting exactly on a candidate
        // diagonal — the ordinary case for rectilinear geometry, where an L, a
        // T or a staircase puts three corners on one line — failed the `> 0.0`
        // test on one edge and did not block. The clipper then cut an "ear"
        // straight across the notch: on a 3 m² L-hexagon it emitted 4 m² of
        // triangles, one of them outside the polygon and one wound backwards,
        // with `fan_fallbacks` reporting zero. Nothing else noticed, because the
        // signed areas cancel to exactly the right answer — which is why the
        // winding gate (signed volume) is blind to this by construction and
        // `every_ngon_triangulation_tiles_its_polygon` measures UNSIGNED area.
        //
        // Testing convex vertices with `>=` instead would be the opposite
        // mistake: a convex vertex adjacent to the ear legitimately touches it.
        let reflex: Vec<bool> = (0..m)
            .map(|i| {
                cross2(
                    p2[remaining[(i + m - 1) % m]],
                    p2[remaining[i]],
                    p2[remaining[(i + 1) % m]],
                ) <= 0.0
            })
            .collect();
        let is_ear = |i: usize| -> Option<(usize, usize, usize)> {
            let (a, b, c) = (
                remaining[(i + m - 1) % m],
                remaining[i],
                remaining[(i + 1) % m],
            );
            if cross2(p2[a], p2[b], p2[c]) <= 0.0 {
                return None; // reflex or collinear — not an ear
            }
            let blocked = remaining.iter().enumerate().any(|(ki, &k)| {
                reflex[ki] && k != a && k != b && k != c && inside_or_on(p2[a], p2[b], p2[c], p2[k])
            });
            if blocked {
                None
            } else {
                Some((a, b, c))
            }
        };
        // First choice: an ear whose diagonal is new. Second: any ear at all.
        let clipped = (0..m)
            .find_map(|i| {
                is_ear(i)
                    .filter(|&(a, _, c)| !forbidden(a, c))
                    .map(|t| (i, t))
            })
            .or_else(|| (0..m).find_map(|i| is_ear(i).map(|t| (i, t))));
        match clipped {
            Some((i, (a, b, c))) => {
                tris.push([a, b, c]);
                remaining.remove(i);
            }
            None => {
                // No ear: the polygon is self-intersecting or degenerate. Fan the
                // remainder and say so rather than looping forever.
                fell_back = true;
                for k in 1..remaining.len() - 1 {
                    tris.push([remaining[0], remaining[k], remaining[k + 1]]);
                }
                remaining.truncate(3);
                break;
            }
        }
    }
    if remaining.len() == 3 && !fell_back {
        tris.push([remaining[0], remaining[1], remaining[2]]);
    }
    Triangulation {
        tris,
        fan_fallback: fell_back,
    }
}

/// [`triangulate_with`] against a polygon that shares no edges with anything —
/// the shape the geometry tests exercise.
#[cfg(test)]
fn triangulate(pts: &[DVec3]) -> (Vec<[usize; 3]>, bool) {
    let t = triangulate_with(pts, &|_, _| false);
    (t.tris, t.fan_fallback)
}

fn fan(n: usize) -> Vec<[usize; 3]> {
    (1..n - 1).map(|k| [0, k, k + 1]).collect()
}

fn cross2(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Inside the triangle **or on its boundary**. `>=`, not `>`: a reflex vertex
/// lying exactly on a candidate diagonal is the case that matters (see
/// [`triangulate_with`]), and strict containment waves it through.
fn inside_or_on(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> bool {
    cross2(a, b, p) >= 0.0 && cross2(b, c, p) >= 0.0 && cross2(c, a, p) >= 0.0
}

/// MikkTSpace-class tangent generation. See the module docs for the determinism
/// argument.
fn tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> (Vec<[f32; 4]>, usize) {
    let count = positions.len();
    let mut tan = vec![DVec3::ZERO; count];
    let mut bit = vec![DVec3::ZERO; count];
    let p = |i: usize| {
        DVec3::new(
            positions[i][0] as f64,
            positions[i][1] as f64,
            positions[i][2] as f64,
        )
    };
    let uv = |i: usize| (uvs[i][0] as f64, uvs[i][1] as f64);

    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (p0, p1, p2) = (p(i0), p(i1), p(i2));
        let (t0, t1, t2) = (uv(i0), uv(i1), uv(i2));
        let d1 = (t1.0 - t0.0, t1.1 - t0.1);
        let d2 = (t2.0 - t0.0, t2.1 - t0.1);
        let det = d1.0 * d2.1 - d2.0 * d1.1;
        if det.abs() < 1e-20 {
            continue; // the UV triangle has no area, so it carries no direction
        }
        let r = 1.0 / det;
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let t = (e1 * d2.1 - e2 * d1.1) * r;
        let b = (e2 * d1.0 - e1 * d2.0) * r;
        for i in [i0, i1, i2] {
            tan[i] += t;
            bit[i] += b;
        }
    }

    let mut out = Vec::with_capacity(count);
    let mut fallbacks = 0usize;
    for i in 0..count {
        let n = DVec3::new(
            normals[i][0] as f64,
            normals[i][1] as f64,
            normals[i][2] as f64,
        );
        let t = tan[i] - n * n.dot(tan[i]);
        let len = t.length();
        if len.is_nan() || len <= 1e-12 || !n.is_finite() {
            fallbacks += 1;
            out.push(TANGENT_FALLBACK);
            continue;
        }
        let t = t / len;
        let w = if n.cross(t).dot(bit[i]) < 0.0 {
            -1.0
        } else {
            1.0
        };
        out.push([t.x as f32, t.y as f32, t.z as f32, w]);
    }
    (out, fallbacks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, cylinder, from_mesh_asset, plane, torus};
    use crate::ops::{apply, Op};
    use crate::validate::validate;

    fn export(mesh: &Mesh) -> MeshAsset {
        to_mesh_asset(mesh, &ExportOptions::default()).0
    }

    #[test]
    fn a_cube_writes_24_vertices_and_12_triangles() {
        let asset = export(&cube(1.0));
        assert_eq!(asset.submeshes.len(), 1);
        assert_eq!(asset.vertex_count(), 24, "every corner splits — all sharp");
        assert_eq!(asset.triangle_count(), 12, "6 quads, ear-clipped");
        assert_eq!(asset.bounds.min, [-0.5; 3]);
        assert_eq!(asset.bounds.max, [0.5; 3]);
        // Flat shading: each face's four corners share one normal.
        let mut normals: Vec<[u32; 3]> = asset.submeshes[0]
            .vertices
            .iter()
            .map(|v| {
                [
                    v.normal[0].to_bits(),
                    v.normal[1].to_bits(),
                    v.normal[2].to_bits(),
                ]
            })
            .collect();
        normals.sort_unstable();
        normals.dedup();
        assert_eq!(normals.len(), 6, "one normal per cube face");
    }

    #[test]
    fn a_smooth_torus_shares_one_normal_per_vertex() {
        let t = torus(1.0, 0.25, 12, 8);
        let asset = export(&t);
        assert_eq!(
            asset.vertex_count(),
            t.vert_count() + 12 + 8 + 1,
            "no normal seams; the UV seam rings are the only splits"
        );
    }

    /// The divergence-theorem volume of a closed triangle soup. Positive exactly
    /// when the winding faces outward — the whole cube was inside-out until this
    /// existed, and nothing else noticed (every count, every invariant and every
    /// round trip is winding-agnostic).
    fn signed_volume(asset: &MeshAsset) -> f64 {
        let mut vol = 0.0;
        for sm in &asset.submeshes {
            for tri in sm.indices.chunks_exact(3) {
                let p = |k: usize| {
                    let v = sm.vertices[tri[k] as usize].position;
                    DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
                };
                vol += p(0).dot(p(1).cross(p(2))) / 6.0;
            }
        }
        vol
    }

    #[test]
    fn every_closed_primitive_is_wound_outward() {
        let cases: [(&str, Mesh, f64); 3] = [
            ("cube", cube(2.0), 8.0),
            // π r² h = π · 0.25 · 2 ≈ 1.57, under-estimated by a 64-gon.
            ("cylinder", cylinder(0.5, 2.0, 64), 1.56),
            // 2π² R r² = 2π² · 1 · 0.0625 ≈ 1.234, under-estimated by the grid.
            ("torus", torus(1.0, 0.25, 48, 24), 1.21),
        ];
        for (name, mesh, expect) in cases {
            let v = signed_volume(&export(&mesh));
            assert!(v > 0.0, "{name} is wound inside-out (signed volume {v})");
            assert!(
                (v - expect).abs() < 0.05 * expect.abs() + 0.01,
                "{name} volume {v}, expected ≈{expect}"
            );
        }
    }

    #[test]
    fn every_written_normal_and_tangent_is_unit_length() {
        for m in [cube(1.0), cylinder(0.5, 2.0, 10), torus(1.0, 0.25, 10, 6)] {
            let asset = export(&m);
            for sm in &asset.submeshes {
                for v in &sm.vertices {
                    let n = DVec3::new(v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64);
                    assert!((n.length() - 1.0).abs() < 1e-5, "normal {n:?}");
                    let t = DVec3::new(
                        v.tangent[0] as f64,
                        v.tangent[1] as f64,
                        v.tangent[2] as f64,
                    );
                    assert!((t.length() - 1.0).abs() < 1e-5, "tangent {t:?}");
                    assert!(v.tangent[3] == 1.0 || v.tangent[3] == -1.0);
                }
            }
        }
    }

    #[test]
    fn tangent_generation_has_a_known_answer() {
        // A unit quad in XY with UVs transposed (u = y, v = x): the tangent must
        // come out along +Y with negative handedness. Worked by hand in the
        // module docs' terms: T = (0,1,0), B = (1,0,0), N = (0,0,1),
        // cross(N,T) = (-1,0,0), dot with B < 0 ⇒ w = −1.
        let positions = [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let normals = [[0.0f32, 0.0, 1.0]; 4];
        let uvs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let indices = [0u32, 1, 2, 0, 2, 3];
        let (t, fallbacks) = tangents(&positions, &normals, &uvs, &indices);
        assert_eq!(fallbacks, 0);
        for got in t {
            assert_eq!(got, [0.0, 1.0, 0.0, -1.0]);
        }
    }

    #[test]
    fn a_mirrored_uv_island_splits_its_shared_seam_vertices() {
        // M5. Two quads sharing an edge, with MIRRORED uv islands: the shared
        // corners carry the same uv and the same normal, and the two faces wind
        // opposite ways in UV space. Without handedness in the corner key they
        // intern to one vertex whose bitangent is the sum of two opposing
        // contributions, so the written `w` depends on which triangle the ear
        // clipper emitted first — a seam that flips its normal map for reasons
        // no one can reproduce.
        let mut m = Mesh::new();
        let p = [
            [0.0, 0.0, 0.0],  // 0 ─┐ shared edge 0–1
            [1.0, 0.0, 0.0],  // 1 ─┘
            [1.0, 0.0, 1.0],  // 2  right quad
            [0.0, 0.0, 1.0],  // 3
            [1.0, 0.0, -1.0], // 4  left quad
            [0.0, 0.0, -1.0], // 5
        ];
        let ids: Vec<VertId> = p
            .iter()
            .map(|q| {
                apply(&mut m, &Op::AddVertex { position: *q })
                    .unwrap()
                    .verts[0]
            })
            .collect();
        let corner = |uv: [f64; 2]| crate::topo::CornerData { uv, normal: None };
        // Right quad: uv winds one way.
        apply(
            &mut m,
            &Op::AddFace {
                verts: vec![ids[0], ids[1], ids[2], ids[3]],
                corners: vec![
                    corner([0.0, 0.0]),
                    corner([1.0, 0.0]),
                    corner([1.0, 1.0]),
                    corner([0.0, 1.0]),
                ],
                slot: None,
            },
        )
        .unwrap();
        // Left quad: MIRRORED in u, so its uv loop winds the other way while the
        // shared corners keep the very same uv values.
        apply(
            &mut m,
            &Op::AddFace {
                verts: vec![ids[1], ids[0], ids[5], ids[4]],
                corners: vec![
                    corner([1.0, 0.0]),
                    corner([0.0, 0.0]),
                    corner([0.0, 1.0]),
                    corner([1.0, 1.0]),
                ],
                slot: None,
            },
        )
        .unwrap();
        assert_eq!(validate(&m), Ok(()));

        // The two faces really do disagree about handedness — otherwise this
        // fixture would be testing nothing.
        let hands: Vec<i8> = m
            .face_ids()
            .map(|f| {
                let hs = m.face_loop(f).unwrap();
                uv_orientation(
                    corner_uv32(&m, hs[0]),
                    corner_uv32(&m, hs[1]),
                    corner_uv32(&m, hs[2]),
                )
            })
            .collect();
        assert_eq!(hands.len(), 2);
        assert_eq!(
            hands[0], -hands[1],
            "the islands must be mirrored: {hands:?}"
        );

        let asset = export(&m);
        // Six kernel vertices; the two on the shared edge split in two, so eight.
        assert_eq!(
            asset.vertex_count(),
            8,
            "the two seam vertices must not be shared across a handedness flip"
        );
        // And each side's tangents are internally consistent: every written
        // vertex has a definite handedness, not a cancelled one.
        for sm in &asset.submeshes {
            for v in &sm.vertices {
                assert!(v.tangent[3] == 1.0 || v.tangent[3] == -1.0);
                let t = DVec3::new(
                    v.tangent[0] as f64,
                    v.tangent[1] as f64,
                    v.tangent[2] as f64,
                );
                assert!(
                    (t.length() - 1.0).abs() < 1e-5,
                    "a cancelled accumulation would leave this degenerate: {t:?}"
                );
            }
        }
        let ws: BTreeSet<u32> = asset.submeshes[0]
            .vertices
            .iter()
            .map(|v| v.tangent[3].to_bits())
            .collect();
        assert_eq!(ws.len(), 2, "both handednesses are present and distinct");
    }

    #[test]
    fn the_write_path_counts_what_it_cannot_refuse() {
        // M6. The kernel's own ops refuse non-finite and non-unit values, so
        // this state can only arrive from an imported asset — which the reader
        // deliberately does not police, because preserving attribute bits is
        // what makes the round trip exact. So the writer counts it.
        let mut asset = crate::build::tests::textured_cube_asset();
        asset.submeshes[0].vertices[0].position[0] = f32::NAN;
        asset.submeshes[0].vertices[1].normal = [0.0, 5.0, 0.0];
        let m = from_mesh_asset(&asset).unwrap().mesh;
        let (_, report) = to_mesh_asset(&m, &ExportOptions::default());
        assert!(
            report.non_finite_written >= 1,
            "a NaN reached the payload uncounted"
        );
        assert!(
            report.non_unit_normals_written >= 1,
            "a [0,5,0] normal reached the payload uncounted"
        );

        // And the clean fixture reports zero, so the counters are not just
        // always-on noise.
        let clean = from_mesh_asset(&crate::build::tests::textured_cube_asset())
            .unwrap()
            .mesh;
        let (_, ok) = to_mesh_asset(&clean, &ExportOptions::default());
        assert_eq!(ok.non_finite_written, 0);
        assert_eq!(ok.non_unit_normals_written, 0);
    }

    #[test]
    fn coincidence_is_measured_in_the_f32_domain_the_reader_welds_in() {
        // M6's sibling defect, M2: two vertices 1e-9 m apart are distinct in the
        // f64 kernel and BIT-EQUAL once written as f32. Measuring the advisory in
        // f64 reported zero at exactly the moment the reader was about to fuse
        // them and refuse the result.
        // The offset has to sit BELOW the f32 ULP at that coordinate. Near 0 an
        // f32 is fine down to 1e-38, so a nudge there stays distinct; near 1.0
        // the ULP is ~1.2e-7 and 1e-9 vanishes. Getting this wrong is how the
        // f64 version of the check looked like it worked.
        let mut m = Mesh::new();
        let p = [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [1.0 + 1e-9, 0.0, 0.0], // distinct in f64, identical in f32
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ];
        let ids: Vec<VertId> = p
            .iter()
            .map(|q| {
                apply(&mut m, &Op::AddVertex { position: *q })
                    .unwrap()
                    .verts[0]
            })
            .collect();
        for tri in [[0, 1, 2], [3, 4, 5]] {
            apply(
                &mut m,
                &Op::AddFace {
                    verts: tri.iter().map(|&i| ids[i]).collect(),
                    corners: vec![Default::default(); 3],
                    slot: None,
                },
            )
            .unwrap();
        }
        assert_ne!(
            m.position(ids[0]).unwrap(),
            m.position(ids[3]).unwrap(),
            "distinct in the kernel"
        );
        let (asset, report) = to_mesh_asset(&m, &ExportOptions::default());
        assert_ne!(
            m.position(ids[0]).unwrap().x as f32 as f64,
            1.0 + 1e-9,
            "the fixture only works because f32 cannot hold the difference"
        );
        assert_eq!(
            report.coincident_vertices, 2,
            "both halves of the pair, measured where the weld happens"
        );
        // And the advisory is telling the truth: writing and reading back is not
        // the identity — the reader really does fuse the pair.
        let back = from_mesh_asset(&asset).unwrap();
        assert_eq!(
            back.report.welded_positions, 5,
            "6 kernel vertices came back as 5"
        );
        assert_eq!(
            back.mesh.vert_count(),
            6,
            "and the fused vertex split into fans"
        );
    }

    #[test]
    fn isolated_vertices_are_not_counted_as_a_write_hazard() {
        // The other half of M2: the old f64 count included vertices export never
        // writes, inflating a hazard that cannot occur.
        let mut m = cube(1.0);
        for _ in 0..3 {
            apply(
                &mut m,
                &Op::AddVertex {
                    position: [7.0, 7.0, 7.0], // three coincident, all isolated
                },
            )
            .unwrap();
        }
        let (_, report) = to_mesh_asset(&m, &ExportOptions::default());
        assert_eq!(
            report.coincident_vertices, 0,
            "nothing that never reaches the payload can collide in it"
        );
    }

    #[test]
    fn tangent_generation_is_identical_across_two_runs() {
        let m = cylinder(0.5, 2.0, 17);
        let a = export(&m);
        let b = export(&m);
        assert_eq!(a, b);
        let bytes = |x: &MeshAsset| inf_asset::encode(x).unwrap();
        assert_eq!(bytes(&a), bytes(&b), "byte-identical, not merely equal");
    }

    #[test]
    fn a_uv_less_mesh_takes_the_constant_tangent_fallback() {
        // Every corner UV is (0,0), so no triangle has a UV gradient.
        let mut m = plane(2.0);
        for h in m.half_ids().collect::<Vec<_>>() {
            if m.is_boundary(h) == Some(false) {
                apply(
                    &mut m,
                    &Op::SetCornerUv {
                        half: h,
                        uv: [0.0, 0.0],
                    },
                )
                .unwrap();
            }
        }
        let (asset, report) = to_mesh_asset(&m, &ExportOptions::default());
        assert_eq!(report.fallback_tangents, asset.vertex_count());
        assert!(asset.submeshes[0]
            .vertices
            .iter()
            .all(|v| v.tangent == TANGENT_FALLBACK));
    }

    /// Six rectilinear polygons, every one of which puts three corners on a line
    /// — the shape real inset/bevel/loop-cut output has, and the shape the first
    /// clipper mis-triangulated.
    fn rectilinear_shapes() -> Vec<(&'static str, Vec<DVec3>, f64)> {
        let poly = |pts: &[(f64, f64)]| -> Vec<DVec3> {
            pts.iter().map(|&(x, z)| DVec3::new(x, 0.0, z)).collect()
        };
        vec![
            (
                "L-hexagon",
                poly(&[
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (2.0, 1.0),
                    (1.0, 1.0),
                    (1.0, 2.0),
                    (0.0, 2.0),
                ]),
                3.0,
            ),
            (
                "L-hexagon mirrored",
                poly(&[
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (2.0, 2.0),
                    (1.0, 2.0),
                    (1.0, 1.0),
                    (0.0, 1.0),
                ]),
                3.0,
            ),
            (
                "T-octagon",
                poly(&[
                    (0.0, 0.0),
                    (3.0, 0.0),
                    (3.0, 1.0),
                    (2.0, 1.0),
                    (2.0, 2.0),
                    (1.0, 2.0),
                    (1.0, 1.0),
                    (0.0, 1.0),
                ]),
                4.0,
            ),
            (
                "U-octagon",
                poly(&[
                    (0.0, 0.0),
                    (3.0, 0.0),
                    (3.0, 2.0),
                    (2.0, 2.0),
                    (2.0, 1.0),
                    (1.0, 1.0),
                    (1.0, 2.0),
                    (0.0, 2.0),
                ]),
                5.0,
            ),
            (
                "staircase",
                poly(&[
                    (0.0, 0.0),
                    (3.0, 0.0),
                    (3.0, 1.0),
                    (2.0, 1.0),
                    (2.0, 2.0),
                    (1.0, 2.0),
                    (1.0, 3.0),
                    (0.0, 3.0),
                ]),
                6.0,
            ),
            (
                "plus-dodecagon",
                poly(&[
                    (1.0, 0.0),
                    (2.0, 0.0),
                    (2.0, 1.0),
                    (3.0, 1.0),
                    (3.0, 2.0),
                    (2.0, 2.0),
                    (2.0, 3.0),
                    (1.0, 3.0),
                    (1.0, 2.0),
                    (0.0, 2.0),
                    (0.0, 1.0),
                    (1.0, 1.0),
                ]),
                5.0,
            ),
        ]
    }

    #[test]
    fn every_ngon_triangulation_tiles_its_polygon() {
        // THE gate for the class the winding gate cannot see. Signed areas of an
        // escaped triangle and an inverted one cancel exactly, so signed volume
        // reads correct while the mesh has geometry outside itself. UNSIGNED
        // area does not cancel: Σ|tri| == |polygon| iff the triangles tile it.
        for (name, pts, want_area) in rectilinear_shapes() {
            let normal = newell_of(&pts);
            let poly_area = normal.length() * 0.5;
            assert!(
                (poly_area - want_area).abs() < 1e-12,
                "{name}: fixture area {poly_area}, expected {want_area}"
            );
            let (tris, fell_back) = triangulate(&pts);
            assert!(!fell_back, "{name}: fell back to a fan");
            assert_eq!(tris.len(), pts.len() - 2, "{name}: wrong triangle count");
            let mut unsigned = 0.0;
            for [a, b, c] in &tris {
                let cr = (pts[*b] - pts[*a]).cross(pts[*c] - pts[*a]);
                assert!(
                    cr.dot(normal) > 0.0,
                    "{name}: triangle {a},{b},{c} is wound against the polygon"
                );
                unsigned += cr.length() * 0.5;
            }
            assert!(
                (unsigned - poly_area).abs() < 1e-12,
                "{name}: triangles cover {unsigned} of a {poly_area} polygon — \
                 geometry escaped the outline"
            );
        }
    }

    #[test]
    fn a_rectilinear_ngon_survives_the_asset_round_trip() {
        // The same shapes through the real writer, not just the clipper.
        for (name, pts, want_area) in rectilinear_shapes() {
            let mut m = Mesh::new();
            let ids: Vec<VertId> = pts
                .iter()
                .map(|p| {
                    apply(
                        &mut m,
                        &Op::AddVertex {
                            position: p.to_array(),
                        },
                    )
                    .unwrap()
                    .verts[0]
                })
                .collect();
            let n = ids.len();
            apply(
                &mut m,
                &Op::AddFace {
                    verts: ids,
                    corners: vec![Default::default(); n],
                    slot: None,
                },
            )
            .unwrap();
            let (asset, report) = to_mesh_asset(&m, &ExportOptions::default());
            assert_eq!(report.fan_fallbacks, 0, "{name}");
            assert_eq!(report.triangles, n - 2, "{name}");
            let sm = &asset.submeshes[0];
            let mut area = 0.0;
            for tri in sm.indices.chunks_exact(3) {
                let p = |k: usize| {
                    let v = sm.vertices[tri[k] as usize].position;
                    DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
                };
                area += (p(1) - p(0)).cross(p(2) - p(0)).length() * 0.5;
            }
            assert!(
                (area - want_area).abs() < 1e-6,
                "{name}: written triangles cover {area}, polygon is {want_area}"
            );
            let back = from_mesh_asset(&asset).expect("{name} must read back");
            assert_eq!(validate(&back.mesh), Ok(()), "{name}");
        }
    }

    #[test]
    fn a_non_convex_ngon_ear_clips_without_inverting_a_triangle() {
        // An arrowhead pentagon in the XZ plane: vertex 4 is reflex, so a naive
        // fan from vertex 0 would emit a triangle outside the polygon.
        let pts: Vec<DVec3> = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, -1.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.0, 3.0],
            [2.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| DVec3::from_array(*p))
        .collect();
        let (tris, fell_back) = triangulate(&pts);
        assert!(!fell_back);
        assert_eq!(tris.len(), 3);
        let normal = newell_of(&pts);
        let poly_area = normal.length() * 0.5;
        let mut sum = 0.0;
        for [a, b, c] in &tris {
            let n = (pts[*b] - pts[*a]).cross(pts[*c] - pts[*a]);
            assert!(
                n.dot(normal) > 0.0,
                "triangle {a},{b},{c} is wound against the polygon"
            );
            sum += n.length() * 0.5;
        }
        assert!(
            (sum - poly_area).abs() < 1e-9,
            "the triangles must tile the polygon exactly: {sum} vs {poly_area}"
        );
    }

    #[test]
    fn the_ngon_survives_a_round_trip_through_the_asset() {
        // The same arrowhead, as a kernel face, exported and read back.
        let mut m = Mesh::new();
        let ids: Vec<VertId> = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, -1.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.0, 3.0],
            [2.0, 0.0, 1.0],
        ]
        .iter()
        .map(|p| {
            apply(&mut m, &Op::AddVertex { position: *p })
                .unwrap()
                .verts[0]
        })
        .collect();
        apply(
            &mut m,
            &Op::AddFace {
                verts: ids,
                corners: vec![Default::default(); 5],
                slot: None,
            },
        )
        .unwrap();
        let asset = export(&m);
        assert_eq!(asset.triangle_count(), 3);
        let back = from_mesh_asset(&asset).unwrap();
        assert_eq!(validate(&back.mesh), Ok(()));
        assert_eq!(back.mesh.vert_count(), 5);
        assert_eq!(
            back.mesh.face_count(),
            3,
            "the n-gon comes back triangulated"
        );
    }

    #[test]
    fn material_slots_and_submesh_names_survive_the_round_trip() {
        let asset = crate::build::tests::textured_cube_asset();
        let imported = from_mesh_asset(&asset).unwrap();
        let out = export(&imported.mesh);
        assert_eq!(out.material_slots, vec!["Default".to_string()]);
        assert_eq!(out.submeshes.len(), 1);
        assert_eq!(out.submeshes[0].name, "cube");
        assert_eq!(out.submeshes[0].material_slot, Some(0));
    }

    #[test]
    fn export_of_an_imported_asset_reproduces_everything_but_the_tangents() {
        let a0 = crate::build::tests::textured_cube_asset();
        let a1 = export(&from_mesh_asset(&a0).unwrap().mesh);
        let (v0, v1) = (&a0.submeshes[0].vertices, &a1.submeshes[0].vertices);
        assert_eq!(a0.submeshes[0].indices, a1.submeshes[0].indices);
        assert_eq!(v0.len(), v1.len());
        for (x, y) in v0.iter().zip(v1) {
            assert_eq!(x.position, y.position);
            assert_eq!(x.normal, y.normal, "authored normals are preserved exactly");
            assert_eq!(x.uv, y.uv);
        }
        assert_eq!(a0.bounds, a1.bounds);
        // The one honest exception, asserted rather than hand-waved: the source's
        // placeholder tangents are replaced by generated ones.
        assert!(
            v0.iter().zip(v1).any(|(x, y)| x.tangent != y.tangent),
            "the writer must generate real tangents, not copy the source's"
        );
    }

    #[test]
    fn coincident_vertices_are_reported_because_the_reader_will_fuse_them() {
        // Two triangles sharing a position but not a vertex — legal in the
        // kernel, and the reader's exact weld will make them one.
        let mut m = Mesh::new();
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0], // the same place as vertex 0, a different vertex
            [-1.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
        ];
        let ids: Vec<VertId> = p
            .iter()
            .map(|q| {
                apply(&mut m, &Op::AddVertex { position: *q })
                    .unwrap()
                    .verts[0]
            })
            .collect();
        for tri in [[0, 1, 2], [3, 4, 5]] {
            apply(
                &mut m,
                &Op::AddFace {
                    verts: tri.iter().map(|&i| ids[i]).collect(),
                    corners: vec![Default::default(); 3],
                    slot: None,
                },
            )
            .unwrap();
        }
        assert_eq!(validate(&m), Ok(()));
        let (_, report) = to_mesh_asset(&m, &ExportOptions::default());
        assert_eq!(report.coincident_vertices, 2, "both halves of the pair");
    }

    #[test]
    fn the_ear_chooser_avoids_a_diagonal_that_already_exists() {
        // A closed envelope over four vertices: one quad `a b c d` on one side,
        // and the SAME quad triangulated across `a–c` on the other. So the a–c
        // diagonal is already a real edge with two faces on it. Ear-clipping the
        // quad across a–c would put four faces on that edge in the written soup,
        // and `from_mesh_asset` would refuse its own writer's output. The b–d
        // diagonal is free, and the chooser has to take it.
        let mut m = Mesh::new();
        let ids: Vec<VertId> = [
            [0.0, 0.0, 0.0], // a
            [1.0, 0.0, 0.0], // b
            [1.0, 0.0, 1.0], // c
            [0.0, 0.0, 1.0], // d
        ]
        .iter()
        .map(|p| {
            apply(&mut m, &Op::AddVertex { position: *p })
                .unwrap()
                .verts[0]
        })
        .collect();
        let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
        for loop_verts in [vec![a, b, c, d], vec![b, a, c], vec![a, d, c]] {
            let n = loop_verts.len();
            apply(
                &mut m,
                &Op::AddFace {
                    verts: loop_verts,
                    corners: vec![Default::default(); n],
                    slot: None,
                },
            )
            .unwrap();
        }
        assert_eq!(validate(&m), Ok(()));
        assert!(m.find_half(a, c).is_some(), "a–c really is an edge");

        let (asset, report) = to_mesh_asset(&m, &ExportOptions::default());
        assert_eq!(report.reused_diagonals, 0, "the b–d diagonal was available");
        assert_eq!(report.coincident_vertices, 0);
        // The proof that matters: the writer's own reader accepts it.
        let back = from_mesh_asset(&asset).expect("the exported asset must be readable");
        assert_eq!(validate(&back.mesh), Ok(()));
        assert_eq!(
            back.mesh.face_count(),
            4,
            "quad → 2 triangles, plus the two"
        );
    }

    #[test]
    fn the_recompute_policy_is_a_fixed_point_only_on_flat_geometry() {
        // The claim the module docs USED to make was that `Recompute` breaks the
        // round trip on the flat-shaded cube. Measured, that is false twice
        // over: it holds on the cube, and what it actually breaks is the CURVED
        // case, which nothing tested.
        //
        // The mechanism: a derived normal is the area-weighted sum over the
        // corner's smooth fan. On the first export that fan is made of n-gons;
        // after the round trip the same surface is triangles, so the fan sums
        // different polygons and lands on different normals. Where every edge is
        // sharp the fan is a single face either way and nothing moves.
        let recompute = ExportOptions {
            normals: NormalPolicy::Recompute,
            optimize: false,
        };
        let fixed_point = |m: &Mesh| {
            let a1 = to_mesh_asset(m, &recompute).0;
            let a2 = to_mesh_asset(&from_mesh_asset(&a1).unwrap().mesh, &recompute).0;
            inf_asset::encode(&a1).unwrap() == inf_asset::encode(&a2).unwrap()
        };
        for (name, m) in [("plane", plane(2.0)), ("cube", cube(1.0))] {
            assert!(fixed_point(&m), "{name}: flat geometry must be stable");
        }
        for (name, m) in [
            ("cylinder", cylinder(0.5, 2.0, 9)),
            ("torus", torus(1.0, 0.25, 9, 5)),
        ] {
            assert!(
                !fixed_point(&m),
                "{name}: if this now holds, the smooth-fan derivation became                  triangulation-independent and the docs above are stale"
            );
        }
        // The DEFAULT policy is a fixed point on all four — that is what
        // `one_round_trip_reaches_a_fixed_point` pins, and it is the reason
        // authored normals win by default.
    }

    #[test]
    fn one_round_trip_reaches_a_fixed_point() {
        // The stability claim that matters for an editor: open, save, open, save
        // is a no-op. Asserted on bytes, over every primitive.
        for m in [
            plane(2.0),
            cube(1.0),
            cylinder(0.5, 2.0, 9),
            torus(1.0, 0.25, 9, 5),
        ] {
            let a1 = export(&m);
            let a2 = export(&from_mesh_asset(&a1).unwrap().mesh);
            assert_eq!(
                inf_asset::encode(&a1).unwrap(),
                inf_asset::encode(&a2).unwrap(),
                "export∘import∘export must equal export"
            );
        }
    }

    #[test]
    fn the_recompute_policy_overrides_authored_normals() {
        // Author a deliberately wrong normal on one corner of a cube. Preserve
        // must write it out; Recompute must ignore it and use the surface.
        let mut m = cube(1.0);
        // A corner of the −Y face, so +Z is unmistakably not its surface normal.
        let corner = m.face_loop(m.face_ids().next().unwrap()).unwrap()[0];
        apply(
            &mut m,
            &Op::SetCornerNormal {
                half: corner,
                normal: Some([0.0, 0.0, 1.0]),
            },
        )
        .unwrap();

        let has_lone_z = |a: &MeshAsset| {
            a.submeshes
                .iter()
                .flat_map(|s| &s.vertices)
                .filter(|v| v.normal == [0.0, 0.0, 1.0])
                .count()
        };
        let preserved = to_mesh_asset(&m, &ExportOptions::default()).0;
        let recomputed = to_mesh_asset(
            &m,
            &ExportOptions {
                normals: NormalPolicy::Recompute,
                optimize: false,
            },
        )
        .0;
        // The +Z face contributes 4 such vertices either way; the authored corner
        // is a fifth under Preserve and vanishes under Recompute.
        assert_eq!(has_lone_z(&preserved), 5, "the authored normal is written");
        assert_eq!(has_lone_z(&recomputed), 4, "and is ignored on recompute");
        assert_eq!(preserved.vertex_count(), 24, "still one vertex per corner");
        assert_eq!(recomputed.vertex_count(), 24);
    }
}
