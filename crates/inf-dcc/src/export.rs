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
//! [`NormalPolicy::Recompute`] ignores authored normals entirely. It is **not**
//! the default, and the reason is a gate rather than a preference: recomputing
//! always would make `SetCornerNormal` an op whose effect can never be observed
//! in the thing the kernel writes, and it would break the bit-identical round
//! trip on the one fixture this batch names (a flat-shaded, UV-mapped cube). It
//! exists for the "bake the shading I can see" case P23.5 will want.
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
}

/// Write a kernel mesh as a `.inf_mesh` payload.
///
/// The produced asset is schema v2 — **the existing format, unchanged**. This
/// batch adds a writer, not a version.
pub fn to_mesh_asset(mesh: &Mesh, opts: &ExportOptions) -> (MeshAsset, ExportReport) {
    let mut report = ExportReport {
        optimized: opts.optimize,
        coincident_vertices: count_coincident(mesh),
        ..Default::default()
    };

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
            build_submesh(mesh, &faces, opts, &mut report, &mut emitted);
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

    // `MeshAsset::new` recomputes the bounds from the written positions.
    (
        MeshAsset::new(submeshes, mesh.material_slots().to_vec()),
        report,
    )
}

/// Vertices that share a position with another vertex — see
/// [`ExportReport::coincident_vertices`]. Bit-exact, and `-0.0` folded onto
/// `+0.0`, because that is exactly the comparison the reader's weld makes.
fn count_coincident(mesh: &Mesh) -> usize {
    let mut buckets: BTreeMap<[u64; 3], usize> = BTreeMap::new();
    for v in mesh.vert_ids() {
        let p = mesh.position(v).expect("live vertex id").to_array();
        let key = [bits(p[0]), bits(p[1]), bits(p[2])];
        *buckets.entry(key).or_default() += 1;
    }
    buckets.values().filter(|&&n| n > 1).sum()
}

fn bits(x: f64) -> u64 {
    if x == 0.0 {
        0
    } else {
        x.to_bits()
    }
}

/// The corner key an output vertex is deduplicated by: the vertex it sits on
/// plus the **f32** bits of the attributes that will actually be written, so the
/// split is exactly the split the payload needs and not one bit finer.
type CornerKey = (VertId, [u32; 2], [u32; 3]);

fn build_submesh(
    mesh: &Mesh,
    faces: &[FaceId],
    opts: &ExportOptions,
    report: &mut ExportReport,
    emitted: &mut BTreeSet<(VertId, VertId)>,
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
        let written = &*emitted;
        let taken = |a: VertId, b: VertId| {
            mesh.find_half(a, b).is_some()
                || mesh.find_half(b, a).is_some()
                || written.contains(&(a, b))
                || written.contains(&(b, a))
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
                );
                let next = positions.len() as u32;
                let idx = *key_to_index.entry(key).or_insert(next);
                if idx == next {
                    let p = mesh.position(v).expect("live vertex id");
                    positions.push([p.x as f32, p.y as f32, p.z as f32]);
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
        let is_ear = |i: usize| -> Option<(usize, usize, usize)> {
            let (a, b, c) = (
                remaining[(i + m - 1) % m],
                remaining[i],
                remaining[(i + 1) % m],
            );
            if cross2(p2[a], p2[b], p2[c]) <= 0.0 {
                return None; // reflex or collinear — not an ear
            }
            let blocked = remaining.iter().any(|&k| {
                k != a && k != b && k != c && strictly_inside(p2[a], p2[b], p2[c], p2[k])
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

fn strictly_inside(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> bool {
    cross2(a, b, p) > 0.0 && cross2(b, c, p) > 0.0 && cross2(c, a, p) > 0.0
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
