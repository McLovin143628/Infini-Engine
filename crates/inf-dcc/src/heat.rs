//! **Bone-heat weight solve** (P24.2): diffuse an influence over the surface and
//! read the weights off where it settles.
//!
//! Baran–Popović's construction, which is what "automatic weights" means in every
//! tool that has them. For each bone, solve
//!
//! ```text
//!     (L + H) w = H p
//! ```
//!
//! over the vertices, where `L` is the mesh Laplacian, `H` is a diagonal of
//! *heat sources* — how strongly this vertex is pulled toward this bone — and `p`
//! marks the vertices the bone is the nearest **visible** bone to. The solution is
//! a smooth field that is 1 near the bone, falls off across the surface rather
//! than through space, and does not leak into a limb the bone cannot see.
//!
//! # Visibility is the whole trick
//!
//! Without it, a hand pressed against a hip is closer to the hip bone than to its
//! own wrist, and the hip weights the hand. The heat source is switched off when
//! the straight line from the vertex to the bone leaves the model — one
//! [`crate::bvh`] ray per (vertex, bone) pair — so influence travels *along* the
//! surface, which is the only route deformation can take.
//!
//! `the_visibility_term_is_what_stops_a_limb_bleeding` measures it: severing the
//! term re-weights an arm pressed against a torso, by name.
//!
//! # The solver is P23.5's, generalized
//!
//! `crate::uv`'s LSCM runs conjugate gradient for [`crate::uv::cg_iterations`]
//! steps — a **fixed integer function of the problem's size**, never a
//! `while residual > tol` loop, because a tolerance test makes the answer depend
//! on rounding and that is the same class of hazard on committed content as a
//! non-portable `sin`. Same rule here, same function, and the same *two* readings
//! reported rather than one: `residual` says how far the field is from satisfying
//! its own equation, and that is the number an author can act on.
//!
//! The system is symmetric positive definite (`L` is PSD, `H` is a non-negative
//! diagonal with at least one positive entry), so plain CG applies directly — no
//! normal equations, unlike LSCM's rectangular system.
//!
//! # What the journal records
//!
//! The **result**, as [`crate::ops::Op::AssignWeights`] — never "run the solver".
//! The meshopt lesson applied before it can bite: a journal is replayed by a
//! different build than wrote it, so an op meaning "whatever the current solver
//! says" would let improving the solver silently rewrite every session ever
//! recorded. Identical reasoning to [`crate::ops::Op::Unwrap`], and it puts the
//! brush ([`crate::paint`]) and the solve on one door.
//!
//! # The Laplacian is uniform-with-inverse-lengths, and that is a choice
//!
//! `L[v][u] = -1/|e_vu|`, `L[v][v] = Σ 1/|e_vu|`. The cotangent Laplacian is the
//! better discretization on a *triangle* mesh and this kernel's faces are n-gons,
//! so using it would mean fixing a triangulation — and the fan a proximity query
//! can get away with ([`crate::bvh::Bvh::from_mesh`]) is not one an operator
//! should be built on. Inverse edge length is n-gon-agnostic, symmetric, and
//! positive — the three properties the solve needs. The cost is accuracy on very
//! anisotropic meshes, which is a smoother-than-ideal falloff, not a wrong one.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use inf_anim::skeleton::Skeleton;

use crate::bvh::Bvh;
use crate::ops::Op;
use crate::skin::{VertWeights, MAX_INFLUENCES};
use crate::topo::{Mesh, VertId};

/// The heat-source constant: `H[v] = HEAT_STRENGTH / d(v)²` for the nearest
/// visible bone.
///
/// Baran–Popović's own figure. It sets how sharply the field is pinned to its
/// bone against how far it diffuses: bigger makes weights tighter and the seams
/// between bones harder, smaller makes everything blend into everything.
pub const HEAT_STRENGTH: f64 = 0.22;

/// Below this distance a bone is treated as being exactly this far away.
///
/// `1/d²` is unbounded, and a vertex sitting *on* a bone would make its row of
/// the system infinitely stiff — which is not "very strongly weighted", it is a
/// NaN. In metres, and two orders of magnitude below any feature a character has.
pub const MIN_BONE_DISTANCE_M: f64 = 1.0e-4;

/// Influences below this are dropped before the top-four cut.
///
/// A diffusion never reaches exactly zero, so without a floor every vertex names
/// four joints — including a foot naming a wrist at `1e-9`, which costs a slot
/// that a real influence needed and makes a weight table impossible to read.
pub const MIN_WEIGHT: f64 = 1.0e-3;

/// One solved bone's verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneReport {
    /// The joint this bone's weights are assigned to.
    pub joint: u16,
    /// Vertices whose nearest **visible** bone is this one — the field's sources.
    pub sources: usize,
    /// `‖(L + H)w − Hp‖ / ‖Hp‖` after the fixed iteration count. Zero means the
    /// field solves its own equation; a large value means CG stopped early, which
    /// is a *reading*, not a failure — see [`HeatReport::worst_residual`].
    pub residual: f64,
}

/// What the solve did.
#[derive(Debug, Clone, PartialEq)]
pub struct HeatReport {
    pub bones: Vec<BoneReport>,
    /// Vertices no bone could see at all.
    ///
    /// They keep [`VertWeights::RIGID`] — all of joint 0 — because a vertex with
    /// no visible bone has no evidence about which one should move it, and
    /// inventing one silently attaches geometry to the root. Non-zero means the
    /// mesh has pockets the skeleton is not inside of, which is exactly what an
    /// author needs told.
    pub unreached: usize,
    /// The worst `residual` over the bones — the single number that says whether
    /// the solver finished.
    pub worst_residual: f64,
    /// Vertices whose weights the solve changed.
    pub assigned: usize,
}

/// Why a solve refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HeatError {
    /// The mesh is not bound, so there is nothing for a joint index to mean.
    #[error("this mesh is not bound to a skeleton; bind it before solving weights")]
    NotSkinned,
    /// The bound joint count and the skeleton disagree — the mesh is bound to a
    /// different rig than the one handed in.
    ///
    /// Refused rather than reconciled: solving against the wrong skeleton
    /// produces a full, plausible, completely wrong weight table.
    #[error("the mesh is bound to a {bound}-joint skeleton but the one given has {given}")]
    SkeletonMismatch { bound: u32, given: usize },
    /// A mesh with no vertices, or a skeleton with no joints.
    #[error("nothing to solve: {what} is empty")]
    Empty { what: &'static str },
}

/// Solve weights for `skeleton` over `mesh`, and return the **op that records the
/// result** plus the numbers behind it.
///
/// `bvh` must be built from the same mesh (it is the visibility oracle). The
/// caller journals the op; nothing here mutates.
///
/// A vertex no bone can see keeps its current weights, so a solve is additive
/// evidence rather than a reset — and `HeatReport::unreached` says how much of
/// the mesh that was.
pub fn solve_heat_weights(
    mesh: &Mesh,
    bvh: &Bvh,
    skeleton: &Skeleton,
) -> Result<(Option<Op>, HeatReport), HeatError> {
    let Some(binding) = mesh.skin_binding() else {
        return Err(HeatError::NotSkinned);
    };
    if binding.joints as usize != skeleton.len() {
        return Err(HeatError::SkeletonMismatch {
            bound: binding.joints,
            given: skeleton.len(),
        });
    }
    let verts: Vec<VertId> = mesh.vert_ids().collect();
    if verts.is_empty() {
        return Err(HeatError::Empty { what: "the mesh" });
    }
    if skeleton.is_empty() {
        return Err(HeatError::Empty {
            what: "the skeleton",
        });
    }
    let index: BTreeMap<VertId, usize> = verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let n = verts.len();
    let positions: Vec<DVec3> = verts
        .iter()
        .map(|&v| mesh.position(v).expect("live vertex id"))
        .collect();

    // ── the bones ──────────────────────────────────────────────────────────
    let bones = bone_segments(skeleton);

    // ── nearest VISIBLE bone per vertex ────────────────────────────────────
    let mut nearest: Vec<Option<(usize, f64)>> = vec![None; n];
    let mut unreached = 0usize;
    for (i, &p) in positions.iter().enumerate() {
        let mut best: Option<(usize, f64)> = None;
        for (b, bone) in bones.iter().enumerate() {
            let c = closest_on_segment(p, bone.a, bone.b);
            if !visible(bvh, p, c) {
                continue;
            }
            let d = (c - p).length().max(MIN_BONE_DISTANCE_M);
            // Ties by BONE INDEX, not by iteration luck: a vertex exactly between
            // a left and a right bone on a symmetric mesh has two equal answers,
            // and it must pick the same one on every run.
            if best.is_none_or(|(bi, bd)| d < bd || (d == bd && b < bi)) {
                best = Some((b, d));
            }
        }
        if best.is_none() {
            unreached += 1;
        }
        nearest[i] = best;
    }

    // ── the Laplacian ──────────────────────────────────────────────────────
    let laplacian = laplacian_rows(mesh, &verts, &index, &positions);

    // ── one CG solve per bone ──────────────────────────────────────────────
    let mut field: Vec<Vec<f64>> = Vec::with_capacity(bones.len());
    let mut reports: Vec<BoneReport> = Vec::with_capacity(bones.len());
    for (b, bone) in bones.iter().enumerate() {
        // H and p. A vertex whose nearest visible bone is THIS one is a source;
        // every other vertex has no heat term at all and is pure diffusion.
        let mut h = vec![0.0f64; n];
        let mut rhs = vec![0.0f64; n];
        let mut sources = 0usize;
        for (i, near) in nearest.iter().enumerate() {
            let Some((nb, d)) = *near else { continue };
            if nb != b {
                continue;
            }
            let hv = HEAT_STRENGTH / (d * d);
            h[i] = hv;
            rhs[i] = hv; // H · p, with p[i] = 1
            sources += 1;
        }
        if sources == 0 {
            field.push(vec![0.0; n]);
            reports.push(BoneReport {
                joint: bone.joint,
                sources: 0,
                residual: 0.0,
            });
            continue;
        }
        let (w, residual) = solve_spd(&laplacian, &h, &rhs);
        field.push(w);
        reports.push(BoneReport {
            joint: bone.joint,
            sources,
            residual,
        });
    }

    // ── gather: the top influences per vertex ──────────────────────────────
    let mut weights: Vec<(VertId, VertWeights)> = Vec::new();
    for (i, &v) in verts.iter().enumerate() {
        if nearest[i].is_none() {
            continue; // no bone can see it; leave its weights alone
        }
        let mut pairs: Vec<(u16, f64)> = Vec::new();
        for (b, bone) in bones.iter().enumerate() {
            let w = field[b][i];
            if w.is_finite() && w > MIN_WEIGHT {
                pairs.push((bone.joint, w));
            }
        }
        if pairs.is_empty() {
            continue;
        }
        // `from_pairs` sums duplicate joints (a joint with several child bones
        // contributes once per bone), takes the top `MAX_INFLUENCES` and
        // normalizes — all with the deterministic tie-breaks that door owns.
        let solved = VertWeights::from_pairs(&pairs);
        if solved != mesh.vert_weights(v).expect("live vertex id") {
            weights.push((v, solved));
        }
    }

    let worst_residual = reports.iter().map(|r| r.residual).fold(0.0f64, |a, b| {
        if b.total_cmp(&a).is_gt() {
            b
        } else {
            a
        }
    });
    let assigned = weights.len();
    let op = (!weights.is_empty()).then_some(Op::AssignWeights { weights });
    Ok((
        op,
        HeatReport {
            bones: reports,
            unreached,
            worst_residual,
            assigned,
        },
    ))
}

/// One bone: a segment, and the joint whose transform moves it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bone {
    joint: u16,
    a: DVec3,
    b: DVec3,
}

/// The skeleton's bones, in joint order.
///
/// **The flesh between a joint and its child belongs to the joint**, because that
/// is the joint whose rotation moves it — the glTF convention, and the one
/// `inf_anim::skinning_matrices` implements. A joint with several children gets
/// one segment each; a **leaf** (a hand, a foot, the head) gets a degenerate
/// segment at its own position, so the flesh past the last joint still has
/// something to attach to instead of being pulled back up the chain.
fn bone_segments(skeleton: &Skeleton) -> Vec<Bone> {
    let joints = skeleton.joints();
    let mut globals: Vec<DVec3> = Vec::with_capacity(joints.len());
    for j in joints {
        let l = DVec3::new(
            j.local_bind.translation[0] as f64,
            j.local_bind.translation[1] as f64,
            j.local_bind.translation[2] as f64,
        );
        let base = j.parent.map(|p| globals[p as usize]).unwrap_or(DVec3::ZERO);
        globals.push(base + l);
    }
    let mut has_child = vec![false; joints.len()];
    for j in joints {
        if let Some(p) = j.parent {
            has_child[p as usize] = true;
        }
    }
    let mut out = Vec::with_capacity(joints.len());
    for (i, j) in joints.iter().enumerate() {
        if let Some(p) = j.parent {
            out.push(Bone {
                joint: p,
                a: globals[p as usize],
                b: globals[i],
            });
        }
        if !has_child[i] {
            out.push(Bone {
                joint: i as u16,
                a: globals[i],
                b: globals[i],
            });
        }
    }
    out
}

/// Whether the straight line from `p` to `c` stays inside the model.
///
/// The ray is fired from the *vertex*, which is on the surface, so
/// `crate::bvh::RAY_EPSILON` is what stops it hitting the triangle it left. A hit
/// strictly before the bone means a wall in between.
fn visible(bvh: &Bvh, p: DVec3, c: DVec3) -> bool {
    if !visibility_enabled() {
        return true;
    }
    let to = c - p;
    let len = to.length();
    if len <= MIN_BONE_DISTANCE_M {
        return true; // the bone is at the vertex; nothing can be between them
    }
    match bvh.first_hit(p, to) {
        // `to` is the full offset, so `t` is a fraction of the distance; a hit at
        // `t < 1` is a surface between the vertex and the bone. The margin keeps a
        // grazing exit at the far end from reading as an occluder.
        Some(hit) => hit.t >= 0.999,
        None => true,
    }
}

/// Whether [] is doing anything — **always true outside tests**.
///
/// The severed control the P24.2 audit asked for. The alternative was a test that
/// *describes* what a solve without visibility would do, which is what the first
/// version of  did: it caught
/// its mutation and never ran a solve, so the weight table it talked about did
/// not exist. A door the test can close makes the control a **measurement**.
///
///  on the switch itself, so the shipped function is the same two
/// lines it always was and this cannot be flipped by anything but a test in this
/// crate.
#[cfg(test)]
fn visibility_enabled() -> bool {
    !tests::VISIBILITY_SEVERED.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn visibility_enabled() -> bool {
    true
}

/// [] with the visibility term severed — the control.
#[cfg(test)]
fn solve_heat_weights_unchecked(
    mesh: &Mesh,
    bvh: &Bvh,
    skeleton: &Skeleton,
) -> Result<(Option<Op>, HeatReport), HeatError> {
    tests::VISIBILITY_SEVERED.with(|c| c.set(true));
    let out = solve_heat_weights(mesh, bvh, skeleton);
    tests::VISIBILITY_SEVERED.with(|c| c.set(false));
    out
}

/// A sparse symmetric Laplacian row: `(neighbour index, −weight)` plus the
/// diagonal.
struct Row {
    diagonal: f64,
    off: Vec<(usize, f64)>,
}

/// `L[v][u] = −1/|e|`, `L[v][v] = Σ 1/|e|` — see the module docs for why not
/// cotangents.
fn laplacian_rows(
    mesh: &Mesh,
    verts: &[VertId],
    index: &BTreeMap<VertId, usize>,
    positions: &[DVec3],
) -> Vec<Row> {
    verts
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let mut off = Vec::new();
            let mut diagonal = 0.0;
            // `BTreeSet`: a vertex reached twice around a fan contributes once,
            // and the order is the neighbour's index rather than the fan walk's.
            let mut seen: BTreeSet<usize> = BTreeSet::new();
            for &h in mesh.vert_outgoing(v).unwrap_or(&[]) {
                let Some(other) = mesh.dest(h) else { continue };
                let Some(&j) = index.get(&other) else {
                    continue;
                };
                if j == i || !seen.insert(j) {
                    continue;
                }
                let len = (positions[j] - positions[i]).length();
                let w = if len > 1e-12 { 1.0 / len } else { 1.0 / 1e-12 };
                diagonal += w;
                off.push((j, -w));
            }
            Row { diagonal, off }
        })
        .collect()
}

/// Conjugate gradient on `(L + H) x = rhs`, for
/// [`crate::uv::cg_iterations`]`(n)` steps.
///
/// A **fixed** count, not a tolerance loop — the P23.5 ruling, restated here
/// because it is the property that makes this solve replayable: the iteration
/// count is an integer function of the problem's size, so two runs do the same
/// arithmetic in the same order.
///
/// Returns the field and `‖Ax − rhs‖ / ‖rhs‖`.
fn solve_spd(rows: &[Row], h: &[f64], rhs: &[f64]) -> (Vec<f64>, f64) {
    let n = rows.len();
    let mul = |x: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let mut acc = (rows[i].diagonal + h[i]) * x[i];
                for &(j, w) in &rows[i].off {
                    acc += w * x[j];
                }
                acc
            })
            .collect()
    };
    let mut x = vec![0.0f64; n];
    let mut r = rhs.to_vec();
    let mut p = r.clone();
    let mut rs = dot(&r, &r);
    for _ in 0..crate::uv::cg_iterations(n) {
        if !(rs.is_finite() && rs > 0.0) {
            break;
        }
        let ap = mul(&p);
        let pap = dot(&p, &ap);
        if !(pap.is_finite() && pap.abs() > 1e-300) {
            break;
        }
        let alpha = rs / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if !rs_new.is_finite() {
            break;
        }
        let beta = rs_new / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = rs_new;
    }
    if !x.iter().all(|v| v.is_finite()) {
        return (vec![0.0; n], f64::INFINITY);
    }
    let ax = mul(&x);
    let num = dot_diff_norm(&ax, rhs);
    let den = dot(rhs, rhs).sqrt();
    let residual = if den > 1e-300 { num / den } else { num };
    (x, residual)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dot_diff_norm(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// The point of segment `[a, b]` closest to `p`. A degenerate segment (a leaf
/// bone) is the point itself.
fn closest_on_segment(p: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 <= 1e-24 {
        return a;
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    a + ab * t
}

/// Const-assert that the gather cannot outrun the channel: `from_pairs` cuts to
/// [`MAX_INFLUENCES`], which is what the vertex can store.
const _: () = assert!(MAX_INFLUENCES == 4);

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// Set by [] only.
        pub(super) static VISIBILITY_SEVERED: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    use crate::bvh::tests::box_tris;
    use crate::ops::apply;
    use crate::validate::validate;
    use inf_anim::skeleton::{Joint, JointTransform};
    use inf_anim::template::{build_template, BodyParams, BodyPlan};

    /// A **tube man** mesh in the kernel: a torso column subdivided enough that
    /// a diffusion has somewhere to go.
    fn tube_mesh() -> Mesh {
        // A 0.4 × 1.8 × 0.4 box, subdivided twice — 384 faces, ~390 vertices,
        // which is a real solve rather than a toy.
        let mut m = crate::build::cube(1.0);
        for _ in 0..3 {
            let faces: Vec<_> = m.face_ids().collect();
            apply(&mut m, &Op::SubdivideFaces { faces }).unwrap();
        }
        // Stretch it into a body: ±0.2 in x/z, 0 .. 1.8 in y.
        let ids: Vec<VertId> = m.vert_ids().collect();
        for v in ids {
            let p = m.position(v).unwrap();
            let q = DVec3::new(p.x * 0.4, (p.y + 0.5) * 1.8, p.z * 0.4);
            apply(
                &mut m,
                &Op::TranslateVerts {
                    verts: vec![v],
                    delta: (q - p).to_array(),
                },
            )
            .unwrap();
        }
        m
    }

    /// A straight 4-joint chain up the tube's axis, so "which joint is nearest"
    /// is a statement about height and the falloff is legible.
    fn spine(joints: usize) -> Skeleton {
        let step = 1.8 / joints as f32;
        let mut out = Vec::new();
        for i in 0..joints {
            out.push(Joint {
                name: format!("j{i}"),
                parent: (i > 0).then(|| i as u16 - 1),
                inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::from_trs(
                    glam::Vec3::new(0.0, if i == 0 { 0.0 } else { step }, 0.0),
                    glam::Quat::IDENTITY,
                    glam::Vec3::ONE,
                ),
            });
        }
        Skeleton::new(out).unwrap()
    }

    fn bind(m: &mut Mesh, joints: u32) {
        apply(
            m,
            &Op::BindSkin {
                skeleton: None,
                joints,
            },
        )
        .unwrap();
    }

    /// **THE GATE.** The solve produces a legal, non-trivial weight table, and
    /// the journal records the RESULT.
    #[test]
    fn the_solve_journals_a_legal_weight_table() {
        let mut m = tube_mesh();
        bind(&mut m, 4);
        let bvh = Bvh::from_mesh(&m);
        let sk = spine(4);
        let (op, report) = solve_heat_weights(&m, &bvh, &sk).expect("solves");
        let op = op.expect("the solve changed something");
        let Op::AssignWeights { weights } = &op else {
            panic!("the solve journals its result: {op:?}");
        };
        assert!(
            weights.len() > m.vert_count() / 2,
            "only {} of {} vertices were weighted",
            weights.len(),
            m.vert_count()
        );
        apply(&mut m, &op).unwrap();
        assert_eq!(validate(&m), Ok(()), "the solved mesh is valid");
        assert_eq!(report.unreached, 0, "the skeleton is inside the tube");
        assert!(
            report.worst_residual < 1e-6,
            "the solve did not converge: {report:?}"
        );

        // ANTI-VACUITY: the table is not a constant. Different heights are
        // weighted to different joints.
        let mut by_joint: BTreeSet<u16> = BTreeSet::new();
        for v in m.vert_ids() {
            for (j, _) in m.vert_weights(v).unwrap().influences() {
                by_joint.insert(j);
            }
        }
        assert!(
            by_joint.len() >= 3,
            "every vertex is on the same joints ({by_joint:?}); a constant table \
             would satisfy every assertion above"
        );
    }

    /// **HEAT LOCALITY.** A vertex's weights are ~0 on joints that are not its
    /// own or its neighbours' — the property that separates a diffusion from an
    /// inverse-distance blend, which weights everything a little bit.
    #[test]
    fn a_vertex_is_not_weighted_to_a_distant_joint() {
        let mut m = tube_mesh();
        bind(&mut m, 4);
        let bvh = Bvh::from_mesh(&m);
        let sk = spine(4);
        let (op, _) = solve_heat_weights(&m, &bvh, &sk).unwrap();
        apply(&mut m, &op.unwrap()).unwrap();

        // The lowest vertices belong to joint 0 (and at most its neighbour 1).
        let mut worst_weight = 0.0f32;
        let mut sample = 0;
        for v in m.vert_ids() {
            let p = m.position(v).unwrap();
            if p.y > 0.2 {
                continue;
            }
            sample += 1;
            let w = m.vert_weights(v).unwrap();
            worst_weight = worst_weight.max(w.weight_of(3));
        }
        assert!(sample > 4, "the sample is too small to mean anything");
        assert!(
            worst_weight < 0.05,
            "a vertex at the tube's foot carries {worst_weight} of the TOP \
             joint's weight; the diffusion is not local"
        );
    }

    /// **SYMMETRY.** A mesh symmetric about `x = 0` gets weights symmetric about
    /// `x = 0` — the property a mirror-modelling workflow depends on, and one an
    /// order-dependent solve breaks quietly.
    #[test]
    fn a_symmetric_mesh_gets_symmetric_weights() {
        let mut m = tube_mesh();
        bind(&mut m, 4);
        let bvh = Bvh::from_mesh(&m);
        let (op, _) = solve_heat_weights(&m, &bvh, &spine(4)).unwrap();
        apply(&mut m, &op.unwrap()).unwrap();

        let mut pairs = 0;
        for v in m.vert_ids() {
            let p = m.position(v).unwrap();
            if p.x <= 1e-9 {
                continue;
            }
            // Its mirror.
            let mirror = m.vert_ids().find(|&u| {
                let q = m.position(u).unwrap();
                (q.x + p.x).abs() < 1e-9 && (q.y - p.y).abs() < 1e-9 && (q.z - p.z).abs() < 1e-9
            });
            let Some(u) = mirror else { continue };
            pairs += 1;
            let (a, b) = (m.vert_weights(v).unwrap(), m.vert_weights(u).unwrap());
            assert_eq!(
                a.influences(),
                b.influences(),
                "the mirror pair at {p:?} disagrees"
            );
        }
        assert!(
            pairs > 20,
            "only {pairs} mirror pairs — too few to mean much"
        );
    }

    /// **THE VISIBILITY TERM, MEASURED ON A REAL SOLVE** (P24.2 audit M-VIS).
    ///
    /// The first version of this test caught its mutation and never called
    /// `solve_heat_weights` — its docstring promised "a second solve putting arm
    /// vertices on the torso" and there was no second solve and no weight table
    /// anywhere in the body. It asserted the *inputs* to the rule and left the
    /// rule's effect unmeasured, which is the failure mode this codebase keeps
    /// paying for: a gate must aim at the thing it names.
    ///
    /// This one runs **two full solves** over the same mesh and skeleton — one
    /// with visibility, one with it severed — and asserts on the weight tables:
    ///
    /// * with visibility, an arm vertex belongs to the arm bone;
    /// * without it, the same vertex is captured by the torso bone, because in a
    ///   straight line the torso bone is nearer;
    /// * and the two tables **differ**, which is the severing's whole footprint.
    #[test]
    fn the_visibility_term_is_what_stops_a_limb_bleeding() {
        // A torso and an arm that touch but do not merge — the case the term
        // exists for. Both are kernel geometry, so a real solve can run.
        let mut m = crate::build::cube(1.0);
        for _ in 0..3 {
            let faces: Vec<_> = m.face_ids().collect();
            apply(&mut m, &Op::SubdivideFaces { faces }).unwrap();
        }
        // Stretch the cube into a torso: ±0.2 x, 0..1.6 y, ±0.12 z.
        let ids: Vec<VertId> = m.vert_ids().collect();
        for v in ids {
            let p = m.position(v).unwrap();
            let q = DVec3::new(p.x * 0.4, (p.y + 0.5) * 1.6, p.z * 0.24);
            apply(
                &mut m,
                &Op::TranslateVerts {
                    verts: vec![v],
                    delta: (q - p).to_array(),
                },
            )
            .unwrap();
        }
        apply(
            &mut m,
            &Op::BindSkin {
                skeleton: None,
                joints: 4,
            },
        )
        .unwrap();

        // The BVH sees the torso *and* an arm box beside it, with a 6 cm gap — so
        // the two walls between them are real geometry a ray can hit.
        let mut tris: Vec<crate::bvh::Tri> = Vec::new();
        for f in m.face_ids() {
            let ps: Vec<DVec3> = m
                .face_verts(f)
                .unwrap()
                .iter()
                .map(|&v| m.position(v).unwrap())
                .collect();
            for i in 1..ps.len() - 1 {
                tris.push(crate::bvh::Tri {
                    a: ps[0],
                    b: ps[i],
                    c: ps[i + 1],
                });
            }
        }
        tris.extend(box_tris(
            DVec3::new(0.26, 1.00, -0.08),
            DVec3::new(0.80, 1.20, 0.08),
        ));
        let bvh = Bvh::new(tris);

        // TWO ROOT CHAINS, and that is the load-bearing detail. A spine that
        // *parents* the arm gives the arm a bone running from inside the torso
        // outward — visible from the torso's own wall, so visibility could never
        // be the deciding term. Measured on the first cut of this fixture: joint
        // 1's segment won every wall vertex and the arm joint won none, and the
        // two solves came out identical. Separate roots put the arm's bone
        // entirely inside the arm box, behind the wall, which is what a hand
        // pressed against a hip actually looks like.
        let joint = |name: &str, parent: Option<u16>, t: [f32; 3]| Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::from_trs(
                glam::Vec3::from_array(t),
                glam::Quat::IDENTITY,
                glam::Vec3::ONE,
            ),
        };
        let sk = Skeleton::new(vec![
            joint("spine", None, [0.0, 0.2, 0.0]),
            joint("spine_tip", Some(0), [0.0, 1.2, 0.0]),
            joint("arm", None, [0.36, 1.1, 0.0]),
            joint("arm_tip", Some(2), [0.34, 0.0, 0.0]),
        ])
        .unwrap();

        // THE FIXTURE'S PREMISE, **derived from the skeleton itself** and
        // asserted before anything is solved (re-audit F2a).
        //
        // The first version of this block computed both bones from hard-coded
        // literals, with no reference to `sk` at all — so it could not have
        // failed on the broken fixture that preceded it, which was a *skeleton*
        // mistake. A premise that cannot see the thing it is a premise about is
        // decoration. `bone_segments` is the same function the solve uses, so the
        // fixture and its premise cannot drift apart silently.
        let bones = bone_segments(&sk);
        let probe = DVec3::new(0.2, 1.1, 0.0);
        let nearest_of = |want_arm: bool| -> (DVec3, f64) {
            let mut best = (DVec3::ZERO, f64::INFINITY);
            for b in &bones {
                // Joints 2 and 3 are the arm chain's; 0 and 1 are the spine's.
                if (b.joint >= 2) != want_arm {
                    continue;
                }
                let c = closest_on_segment(probe, b.a, b.b);
                let d = (c - probe).length();
                if d < best.1 {
                    best = (c, d);
                }
            }
            best
        };
        let (arm_bone, arm_d) = nearest_of(true);
        let (spine_bone, spine_d) = nearest_of(false);
        assert!(
            arm_d.is_finite() && spine_d.is_finite(),
            "the skeleton has no arm chain or no spine chain: {bones:?}"
        );
        assert!(
            arm_d < spine_d,
            "the fixture must make the ARM's bone the nearer one, or distance \
             alone decides this and visibility is untested ({arm_d} vs {spine_d})"
        );
        assert!(
            !visible(&bvh, probe, arm_bone),
            "the arm's own wall must occlude its bone from the torso's surface"
        );
        assert!(
            visible(&bvh, probe, spine_bone),
            "...and the torso's own bone must be visible, or the refusal above is \
             a broken ray rather than a wall"
        );

        // TWO FULL SOLVES over the same mesh and skeleton.
        let (with_op, with_report) = solve_heat_weights(&m, &bvh, &sk).expect("solves");
        let (without_op, _) = solve_heat_weights_unchecked(&m, &bvh, &sk).expect("solves");
        // `with_op` is legitimately `None` here: with visibility on, every torso
        // vertex's nearest visible bone is the spine's, the solve reproduces the
        // rigid default, and an op that changes nothing is not journalled (the
        // "no undo step that does nothing" rule). That is the *point* — the arm
        // gets in only when the term is severed.
        let without_op = without_op.expect(
            "the severed solve assigned nothing at all; the arm's bone is not \
             winning any vertex even with the wall ignored, so this fixture \
             cannot measure the term",
        );

        // (1) **The severing has a footprint**, stated as the two things that
        //     are actually true rather than as a comparison that cannot fail
        //     (re-audit F2b: `assert_ne!(with_op.as_ref(), Some(&without_op))`
        //     was a tautology — `with_op` is `None` by construction and
        //     `without_op` had just been `.expect()`ed into `Some`).
        assert!(
            with_op.is_none(),
            "with visibility ON, every torso vertex's nearest VISIBLE bone is the \
             spine's, so the solve reproduces the rigid default and journals \
             nothing. It journalled {with_op:?} instead — the fixture no longer \
             isolates the term, and the capture count below would be measuring \
             something else."
        );
        // …and with it OFF a table appears at all. Together: the term is the
        // whole difference between "no weights change" and "these do".
        assert_eq!(with_report.unreached, 0, "{with_report:?}");

        // (2) ...and it is the ARM's bone that captures the torso's far wall.
        let table = |op: &Op| -> BTreeMap<VertId, VertWeights> {
            match op {
                Op::AssignWeights { weights } => weights.iter().copied().collect(),
                other => panic!("expected AssignWeights, got {other:?}"),
            }
        };
        let a = with_op.as_ref().map(table).unwrap_or_default();
        let b = table(&without_op);
        let (mut wall, mut captured) = (0usize, 0usize);
        for (v, w_without) in &b {
            let p = m.position(*v).unwrap();
            if p.x < 0.19 || p.y < 0.95 || p.y > 1.25 {
                continue; // not the stretch of far wall the arm lies against
            }
            wall += 1;
            let w_with = a.get(v).copied().unwrap_or(VertWeights::RIGID);
            let arm_share = |w: &VertWeights| w.weight_of(2) + w.weight_of(3);
            if arm_share(w_without) > arm_share(&w_with) + 0.05 {
                captured += 1;
            }
        }
        assert!(
            wall >= 4,
            "only {wall} far-wall vertices - too few to mean anything"
        );
        assert!(
            captured > 0,
            "{captured} of {wall} far-wall vertices were captured by the \
             neighbouring limb's bone when visibility was severed; the term is \
             not what keeps them apart, and this test does not measure what its \
             name says"
        );
    }

    #[test]
    fn the_solve_is_deterministic() {
        let mut m = tube_mesh();
        bind(&mut m, 4);
        let bvh = Bvh::from_mesh(&m);
        let sk = spine(4);
        let a = solve_heat_weights(&m, &bvh, &sk).unwrap();
        let b = solve_heat_weights(&m, &bvh, &sk).unwrap();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn every_refusal_is_a_value() {
        let m = tube_mesh();
        let bvh = Bvh::from_mesh(&m);
        assert_eq!(
            solve_heat_weights(&m, &bvh, &spine(4)),
            Err(HeatError::NotSkinned)
        );
        let mut bound = tube_mesh();
        bind(&mut bound, 4);
        assert_eq!(
            solve_heat_weights(&bound, &bvh, &spine(7)),
            Err(HeatError::SkeletonMismatch { bound: 4, given: 7 })
        );
    }

    /// The whole flow, end to end: fit a template biped into a mesh, bind it,
    /// solve, and journal — with the joint count coming from the fitted rig.
    #[test]
    fn a_fitted_template_solves_over_its_own_mesh() {
        let mut m = tube_mesh();
        let bvh = Bvh::from_mesh(&m);
        let params = BodyParams {
            height_m: 1.8,
            ..Default::default()
        };
        let template = build_template(BodyPlan::Biped, &params).unwrap();
        let joints = template.skeleton.len() as u32;
        bind(&mut m, joints);
        let (op, report) = solve_heat_weights(&m, &bvh, &template.skeleton).expect("solves");
        let op = op.expect("a template rig weights a tube");
        apply(&mut m, &op).unwrap();
        assert_eq!(validate(&m), Ok(()));
        assert!(
            report.assigned > 0 && report.worst_residual.is_finite(),
            "{report:?}"
        );
        // A biped's arms stick out of a 0.4 m tube, so some of its bones are
        // outside — which is exactly the case `unreached` and the per-bone
        // `sources` count exist to report rather than hide.
        let idle: usize = report.bones.iter().filter(|b| b.sources == 0).count();
        assert!(
            idle > 0,
            "a biped inside a tube must have bones no vertex is nearest to; the \
             report is not measuring what it claims"
        );
    }

    /// **WHAT 161 BONES COST THE WEIGHT SOLVER** (SK1a) — measured before
    /// anything is done about it, because a mitigation chosen without a number is
    /// a guess, and this repository has paid for one of those already (the caster
    /// pack whose routed cache could not have touched a GPU pass).
    ///
    /// The shape of the cost, from the code rather than from a hunch:
    /// `bone_segments` turns J joints into B segments, and both the visibility
    /// pass (one BVH ray per vertex per bone) and the diffusion (one conjugate
    /// gradient solve per bone) are linear in B. So the prediction is
    /// **B(161)/B(20)**, and what this arm measures is whether the constant
    /// factors keep that promise.
    ///
    /// Run it with `--nocapture` to read the numbers. What is *asserted* is the
    /// ratio against its own prediction, not a wall-clock budget: a machine-time
    /// ceiling in a unit test is a flake with a date on it, while "the solve grew
    /// the way its own structure says it should" is a property.
    #[test]
    fn the_weight_solve_costs_what_its_bone_count_says_it_should() {
        use std::time::Instant;
        let params = BodyParams {
            height_m: 1.8,
            ..Default::default()
        };
        let run = |plan: BodyPlan| -> (usize, usize, f64) {
            let rig = build_template(plan, &params).unwrap();
            let segments = bone_segments(&rig.skeleton).len();
            let mut m = tube_mesh();
            let bvh = Bvh::from_mesh(&m);
            bind(&mut m, rig.skeleton.len() as u32);
            // Three rounds, and the MIN — the instrument's own convention: the
            // fastest run is the one least polluted by whatever else the machine
            // was doing.
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let (op, _) = solve_heat_weights(&m, &bvh, &rig.skeleton).expect("solves");
                assert!(op.is_some(), "{plan:?} weighted nothing");
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            (rig.skeleton.len(), segments, best)
        };
        let (j20, b20, ms20) = run(BodyPlan::BipedCanonical);
        let (j161, b161, ms161) = run(BodyPlan::Biped);
        let verts = tube_mesh().vert_count();
        let predicted = b161 as f64 / b20 as f64;
        let measured = ms161 / ms20;
        println!(
            "heat solve over {verts} vertices: {j20} joints / {b20} segments = \
             {ms20:.3} ms; {j161} joints / {b161} segments = {ms161:.3} ms; \
             {measured:.2}x measured against {predicted:.2}x predicted"
        );
        assert_eq!((j20, j161), (20, inf_anim::MANNY_JOINT_COUNT));
        // **The measurement, on this machine, in release**: 386 vertices,
        // 24 -> 249 segments (10.4x), and the solve went 32.4 -> 78.3 ms —
        // **2.4x**, not 10.4x. The per-bone work is real and is not what dominates
        // at this mesh size: the Laplacian assembly, the neighbourhood walk and
        // the per-vertex top-4 gather are bone-independent and are most of the
        // clock. That is a finding, not a licence — the balance tips the other way
        // on a mesh with tens of thousands of vertices, where `field` alone is
        // B x V x 8 bytes (161 bones over 50 000 vertices is ~64 MB resident).
        //
        // So the ASSERTION is the upper bound, which is the one that guards
        // something: growing no faster than the bone count means nothing here
        // scales with its square. The lower bound only says the arm is measuring a
        // solve at all.
        assert!(
            measured < predicted,
            "the solve grew {measured:.2}x for {predicted:.2}x the bones, which is \
             worse than linear: something scales with the SQUARE of the bone count"
        );
        assert!(
            measured > 1.0,
            "the solve did not grow at all ({measured:.2}x) — this arm is not \
             measuring per-bone work"
        );
    }
}
