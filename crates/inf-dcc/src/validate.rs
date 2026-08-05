//! Every invariant the kernel promises, checked, and reported **as data**.
//!
//! [`validate`] is an independent auditor, not a guard rail. Deliberately:
//!
//! * The ops in [`crate::ops`] do **not** call it. If applying an op meant
//!   "mutate, then run `validate`, then refuse if it complains", the property
//!   test "every op leaves the mesh valid" would be asserting that a call it
//!   already made returned `Ok` — a vacuous check, and vacuous checks hide real
//!   intrusions (the P19 law). Ops enforce *local* preconditions and a *local*
//!   manifold test; `validate` is the thing that can catch them being wrong.
//! * It returns a `Vec<Violation>`, not a bool and not a string. A violation
//!   names the elements involved, so a failing property test prints the defect
//!   rather than the fact that there was one.
//!
//! # What is checked
//!
//! | | |
//! | --- | --- |
//! | twin involution | `twin(twin(h)) == h`, `twin(h) != h`, twin alive, `origin(twin(h)) != origin(h)` |
//! | loops | `prev(next(h)) == h`, every loop closes, one `face` value per loop, `dest(h) == origin(next(h))` |
//! | vertex index | `out(v)` is exactly the live half-edges originating at `v`, sorted, no duplicates |
//! | faces | `face.half` is alive and belongs to the face; every loop half agrees; ≥ 3 corners |
//! | manifold edge | at most one half-edge from `a` to `b` (⇒ at most two faces per edge) |
//! | manifold vertex | the one-ring fan walk closes and covers every outgoing half-edge |
//! | no wire edges | no edge is face-less on both sides |
//! | sharpness | the flag agrees across a twin pair |
//! | numbers | positions and UVs are finite |
//! | Euler | per connected component *with faces*: `V − E + F = 2 − 2g − b` admits a non-negative integer genus |
//!
//! # What is deliberately NOT checked
//!
//! * **Normal length.** An authored corner normal is a shading attribute carried
//!   verbatim from the source asset; requiring unit length would force import to
//!   renormalize, and renormalizing an `f32` normal through `f64` and back is
//!   exactly the kind of silent one-ULP edit that breaks a bit-identical round
//!   trip. Finiteness is checked; length is the exporter's business.
//! * **Self-intersection / planarity.** Both are geometric, both are legal
//!   intermediate states while modelling, and neither is a property any traversal
//!   depends on.
//! * **Isolated vertices.** Legal: `Op::AddVertex` produces one, and a face is
//!   built out of them. A component with no faces is exempt from the Euler check
//!   for the same reason.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::topo::{FaceId, HalfId, Mesh, VertId};

/// One broken invariant, with the elements that break it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Violation {
    /// `twin(h)` names a dead slot.
    DeadTwin { half: HalfId, twin: HalfId },
    /// `twin(twin(h)) != h`.
    TwinNotInvolutive {
        half: HalfId,
        twin: HalfId,
        twin_twin: HalfId,
    },
    /// A half-edge is its own twin.
    TwinIsSelf { half: HalfId },
    /// An edge whose two ends are the same vertex.
    DegenerateEdge { half: HalfId, vert: VertId },
    /// `next` or `prev` names a dead slot.
    DeadLink { half: HalfId, link: HalfId },
    /// `prev(next(h)) != h`.
    NextPrevMismatch {
        half: HalfId,
        next: HalfId,
        next_prev: HalfId,
    },
    /// `dest(h) != origin(next(h))` — the loop does not follow the surface.
    LoopChainBroken {
        half: HalfId,
        dest: VertId,
        next: HalfId,
        next_origin: VertId,
    },
    /// A loop did not return to its start within the arena's capacity.
    LoopNotClosed { start: HalfId, visited: usize },
    /// Two half-edges in one loop disagree about which face they belong to.
    LoopFaceMismatch {
        half: HalfId,
        face: Option<FaceId>,
        other: HalfId,
        other_face: Option<FaceId>,
    },
    /// `origin(h)` names a dead vertex.
    DeadOrigin { half: HalfId, vert: VertId },
    /// `face(h)` names a dead face record.
    ///
    /// The gap this closes: removing a face record while leaving its half-edges
    /// naming it produced a mesh `validate` called `Ok(())`. Every property in
    /// this crate leans on `validate` as the independent auditor, so a blind spot
    /// here is a blind spot in all of them at once.
    DeadFace { half: HalfId, face: FaceId },
    /// A live half-edge originating at `v` that `out(v)` does not list.
    VertOutMissing { vert: VertId, half: HalfId },
    /// `out(v)` lists a half-edge that is dead or does not originate at `v`.
    VertOutStale { vert: VertId, half: HalfId },
    /// `out(v)` is not strictly ascending (unsorted, or a duplicate).
    VertOutNotSorted { vert: VertId },
    /// A face's representative half-edge is dead.
    FaceHalfDead { face: FaceId, half: HalfId },
    /// A face's representative half-edge belongs to a different face.
    FaceHalfNotOwned {
        face: FaceId,
        half: HalfId,
        owner: Option<FaceId>,
    },
    /// A face loop with fewer than three corners.
    FaceTooSmall { face: FaceId, corners: usize },
    /// An edge with no face on either side.
    WireEdge { half: HalfId },
    /// Two half-edges run from `from` to `to`.
    DuplicateDirectedEdge {
        from: VertId,
        to: VertId,
        halves: Vec<HalfId>,
    },
    /// The one-ring of `vert` is not a single closed fan (a bowtie).
    NonManifoldVertex {
        vert: VertId,
        fan: usize,
        outgoing: usize,
    },
    /// `sharp` differs across a twin pair — the flag describes one undirected
    /// edge and must not depend on which half you ask.
    SharpDisagrees { half: HalfId, twin: HalfId },
    /// A face names a material slot the mesh has no name for.
    UnknownSlot {
        face: FaceId,
        slot: u32,
        slots: usize,
    },
    /// A non-finite coordinate.
    NonFinitePosition { vert: VertId },
    /// A non-finite UV.
    NonFiniteUv { half: HalfId },
    /// A non-finite authored corner normal.
    NonFiniteNormal { half: HalfId },
    /// `V − E + F` is not consistent with any non-negative integer genus for a
    /// component with `boundary_loops` boundaries.
    EulerInconsistent {
        /// The lowest vertex id in the component — its stable name.
        component: VertId,
        verts: usize,
        edges: usize,
        faces: usize,
        boundary_loops: usize,
        /// `V − E + F`.
        chi: i64,
    },
}

/// Check every invariant. `Ok(())` or every violation found, in a deterministic
/// order (elements ascending, checks in the order documented above).
pub fn validate(mesh: &Mesh) -> Result<(), Vec<Violation>> {
    let mut v = Vec::new();
    check_twins(mesh, &mut v);
    check_links(mesh, &mut v);
    check_vert_index(mesh, &mut v);
    check_faces(mesh, &mut v);
    check_edges(mesh, &mut v);
    check_numbers(mesh, &mut v);
    // The structural checks above are what the loop/fan walks assume. Running a
    // fan walk over links that are already known-broken produces cascades of
    // noise, so the topology-dependent checks only run on a structurally sound
    // mesh.
    if v.is_empty() {
        check_loops(mesh, &mut v);
        check_vertex_manifold(mesh, &mut v);
    }
    if v.is_empty() {
        check_euler(mesh, &mut v);
    }
    if v.is_empty() {
        Ok(())
    } else {
        Err(v)
    }
}

fn check_twins(mesh: &Mesh, out: &mut Vec<Violation>) {
    for h in mesh.half_ids() {
        let t = mesh.twin(h).expect("live half-edge id");
        if t == h {
            out.push(Violation::TwinIsSelf { half: h });
            continue;
        }
        let Some(tt) = mesh.twin(t) else {
            out.push(Violation::DeadTwin { half: h, twin: t });
            continue;
        };
        if tt != h {
            out.push(Violation::TwinNotInvolutive {
                half: h,
                twin: t,
                twin_twin: tt,
            });
        }
        if mesh.origin(h) == mesh.origin(t) {
            out.push(Violation::DegenerateEdge {
                half: h,
                vert: mesh.origin(h).expect("live half-edge id"),
            });
        }
        if mesh.is_sharp(h) != mesh.is_sharp(t) {
            out.push(Violation::SharpDisagrees { half: h, twin: t });
        }
        if mesh.face_of(h) == Some(None) && mesh.face_of(t) == Some(None) {
            out.push(Violation::WireEdge { half: h });
        }
    }
}

fn check_links(mesh: &Mesh, out: &mut Vec<Violation>) {
    for h in mesh.half_ids() {
        // Liveness of EVERY id a half-edge holds, before anything that might
        // `continue`: `twin` is checked in `check_twins`, `origin` and the two
        // links here, and `face` — the one that was missing, and the reason a
        // mesh whose face record had been removed while its loop still named it
        // read as `Ok(())`.
        if let Some(Some(f)) = mesh.face_of(h) {
            if !mesh.has_face(f) {
                out.push(Violation::DeadFace { half: h, face: f });
            }
        }
        let origin = mesh.origin(h).expect("live half-edge id");
        if !mesh.has_vert(origin) {
            out.push(Violation::DeadOrigin {
                half: h,
                vert: origin,
            });
        }
        let n = mesh.next(h).expect("live half-edge id");
        let p = mesh.prev(h).expect("live half-edge id");
        for link in [n, p] {
            if !mesh.has_half(link) {
                out.push(Violation::DeadLink { half: h, link });
            }
        }
        if !mesh.has_half(n) {
            continue;
        }
        let np = mesh.prev(n).expect("checked live");
        if np != h {
            out.push(Violation::NextPrevMismatch {
                half: h,
                next: n,
                next_prev: np,
            });
        }
        if let (Some(d), Some(no)) = (mesh.dest(h), mesh.origin(n)) {
            if d != no {
                out.push(Violation::LoopChainBroken {
                    half: h,
                    dest: d,
                    next: n,
                    next_origin: no,
                });
            }
        }
    }
}

fn check_vert_index(mesh: &Mesh, out: &mut Vec<Violation>) {
    let mut expected: BTreeMap<VertId, BTreeSet<HalfId>> = BTreeMap::new();
    for h in mesh.half_ids() {
        let o = mesh.origin(h).expect("live half-edge id");
        expected.entry(o).or_default().insert(h);
    }
    for v in mesh.vert_ids() {
        let listed = mesh.vert_outgoing(v).expect("live vertex id");
        if listed.windows(2).any(|w| w[0] >= w[1]) {
            out.push(Violation::VertOutNotSorted { vert: v });
        }
        let empty = BTreeSet::new();
        let want = expected.get(&v).unwrap_or(&empty);
        for &h in listed {
            if !want.contains(&h) {
                out.push(Violation::VertOutStale { vert: v, half: h });
            }
        }
        for &h in want {
            if !listed.contains(&h) {
                out.push(Violation::VertOutMissing { vert: v, half: h });
            }
        }
    }
    // Half-edges whose origin is not a live vertex at all.
    for (&v, halves) in &expected {
        if !mesh.has_vert(v) {
            for &h in halves {
                out.push(Violation::DeadOrigin { half: h, vert: v });
            }
        }
    }
}

fn check_faces(mesh: &Mesh, out: &mut Vec<Violation>) {
    for f in mesh.face_ids() {
        let h = mesh.face_half(f).expect("live face id");
        if !mesh.has_half(h) {
            out.push(Violation::FaceHalfDead { face: f, half: h });
            continue;
        }
        let owner = mesh.face_of(h).expect("checked live");
        if owner != Some(f) {
            out.push(Violation::FaceHalfNotOwned {
                face: f,
                half: h,
                owner,
            });
        }
        if let Some(slot) = mesh.face_slot(f).expect("live face id") {
            let slots = mesh.material_slots().len();
            // An empty name list means "unnamed slots" (an asset may carry
            // per-submesh slot indices with no names at all), so the range check
            // only applies once the mesh has names to be out of range of.
            if slots > 0 && slot as usize >= slots {
                out.push(Violation::UnknownSlot {
                    face: f,
                    slot,
                    slots,
                });
            }
        }
    }
}

fn check_edges(mesh: &Mesh, out: &mut Vec<Violation>) {
    let mut directed: BTreeMap<(VertId, VertId), Vec<HalfId>> = BTreeMap::new();
    for h in mesh.half_ids() {
        let (Some(a), Some(b)) = (mesh.origin(h), mesh.dest(h)) else {
            continue;
        };
        directed.entry((a, b)).or_default().push(h);
    }
    for ((a, b), halves) in directed {
        if halves.len() > 1 {
            out.push(Violation::DuplicateDirectedEdge {
                from: a,
                to: b,
                halves,
            });
        }
    }
}

fn check_numbers(mesh: &Mesh, out: &mut Vec<Violation>) {
    for v in mesh.vert_ids() {
        let p = mesh.position(v).expect("live vertex id");
        if !p.is_finite() {
            out.push(Violation::NonFinitePosition { vert: v });
        }
    }
    for h in mesh.half_ids() {
        if mesh.is_boundary(h) == Some(true) {
            continue; // boundary corner attributes are never read
        }
        let uv = mesh.corner_uv(h).expect("live half-edge id");
        if !uv.iter().all(|x| x.is_finite()) {
            out.push(Violation::NonFiniteUv { half: h });
        }
        if let Some(n) = mesh.corner_normal(h).expect("live half-edge id") {
            if !n.iter().all(|x| x.is_finite()) {
                out.push(Violation::NonFiniteNormal { half: h });
            }
        }
    }
}

fn check_loops(mesh: &Mesh, out: &mut Vec<Violation>) {
    let mut seen: BTreeSet<HalfId> = BTreeSet::new();
    for h in mesh.half_ids() {
        if seen.contains(&h) {
            continue;
        }
        let (walked, closed) = mesh.walk_loop(h);
        for &w in &walked {
            seen.insert(w);
        }
        if !closed {
            out.push(Violation::LoopNotClosed {
                start: h,
                visited: walked.len(),
            });
            continue;
        }
        let face = mesh.face_of(h).expect("live half-edge id");
        if let Some(&other) = walked
            .iter()
            .find(|&&w| mesh.face_of(w).expect("live half-edge id") != face)
        {
            out.push(Violation::LoopFaceMismatch {
                half: h,
                face,
                other,
                other_face: mesh.face_of(other).expect("live half-edge id"),
            });
        }
        if let Some(f) = face {
            if walked.len() < 3 {
                out.push(Violation::FaceTooSmall {
                    face: f,
                    corners: walked.len(),
                });
            }
        }
    }
}

fn check_vertex_manifold(mesh: &Mesh, out: &mut Vec<Violation>) {
    for v in mesh.vert_ids() {
        let outgoing = mesh.vert_outgoing(v).expect("live vertex id");
        let Some(&first) = outgoing.first() else {
            continue;
        };
        let (fan, closed) = mesh.walk_fan(first);
        if !closed || fan.len() != outgoing.len() {
            out.push(Violation::NonManifoldVertex {
                vert: v,
                fan: fan.len(),
                outgoing: outgoing.len(),
            });
        }
    }
}

/// Per connected component with at least one face: `χ = V − E + F` must equal
/// `2 − 2g − b` for some integer genus `g ≥ 0`, where `b` is the number of
/// boundary loops. We cannot know `g` independently, so what is checked is that
/// `2 − χ − b` is even and non-negative — which is exactly the statement that
/// the four counts are consistent with *some* surface.
fn check_euler(mesh: &Mesh, out: &mut Vec<Violation>) {
    let mut uf = UnionFind::new(mesh);
    for h in mesh.half_ids() {
        if let (Some(a), Some(b)) = (mesh.origin(h), mesh.dest(h)) {
            uf.union(a, b);
        }
    }

    let mut verts: BTreeMap<VertId, usize> = BTreeMap::new();
    let mut halves: BTreeMap<VertId, usize> = BTreeMap::new();
    let mut faces: BTreeMap<VertId, usize> = BTreeMap::new();
    let mut boundaries: BTreeMap<VertId, usize> = BTreeMap::new();

    for v in mesh.vert_ids() {
        *verts.entry(uf.find(v)).or_default() += 1;
    }
    for h in mesh.half_ids() {
        let o = mesh.origin(h).expect("live half-edge id");
        *halves.entry(uf.find(o)).or_default() += 1;
    }
    for f in mesh.face_ids() {
        let h = mesh.face_half(f).expect("live face id");
        let o = mesh.origin(h).expect("live half-edge id");
        *faces.entry(uf.find(o)).or_default() += 1;
    }
    let mut seen: BTreeSet<HalfId> = BTreeSet::new();
    for h in mesh.half_ids() {
        if mesh.is_boundary(h) != Some(true) || seen.contains(&h) {
            continue;
        }
        let (walked, closed) = mesh.walk_loop(h);
        for &w in &walked {
            seen.insert(w);
        }
        if closed {
            let o = mesh.origin(h).expect("live half-edge id");
            *boundaries.entry(uf.find(o)).or_default() += 1;
        }
    }

    for (root, &f) in &faces {
        if f == 0 {
            continue;
        }
        let v = verts.get(root).copied().unwrap_or(0);
        let e = halves.get(root).copied().unwrap_or(0) / 2;
        let b = boundaries.get(root).copied().unwrap_or(0);
        let chi = v as i64 - e as i64 + f as i64;
        let twice_genus = 2 - chi - b as i64;
        if twice_genus < 0 || twice_genus % 2 != 0 {
            out.push(Violation::EulerInconsistent {
                component: *root,
                verts: v,
                edges: e,
                faces: f,
                boundary_loops: b,
                chi,
            });
        }
    }
}

/// Union-find over vertex ids, keyed by arena index (ids are dense enough that a
/// `Vec` is both the fastest and the most deterministic structure).
struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    fn new(mesh: &Mesh) -> Self {
        let n = mesh.vert_ids().map(|v| v.0 + 1).max().unwrap_or(0).max(1) as usize;
        Self {
            parent: (0..n as u32).collect(),
        }
    }
    fn find(&mut self, v: VertId) -> VertId {
        let mut i = v.0;
        while self.parent[i as usize] != i {
            let g = self.parent[self.parent[i as usize] as usize];
            self.parent[i as usize] = g;
            i = g;
        }
        VertId(i)
    }
    fn union(&mut self, a: VertId, b: VertId) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Lowest id wins, so a component's name is stable and deterministic.
            let (lo, hi) = if ra.0 < rb.0 { (ra, rb) } else { (rb, ra) };
            self.parent[hi.0 as usize] = lo.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{cube, cylinder, plane, torus};

    #[test]
    fn the_primitives_are_valid() {
        for (name, m) in [
            ("plane", plane(2.0)),
            ("cube", cube(1.0)),
            ("cylinder", cylinder(0.5, 2.0, 12)),
            ("torus", torus(1.0, 0.3, 12, 8)),
        ] {
            assert_eq!(validate(&m), Ok(()), "{name} must be valid");
        }
    }

    #[test]
    fn euler_characteristics_match_the_genus_of_each_primitive() {
        // Closed genus-0: χ = 2.
        let c = cube(1.0);
        assert_eq!(
            c.vert_count() as i64 - c.edge_count() as i64 + c.face_count() as i64,
            2
        );
        // A disc: one boundary loop, χ = 1.
        let p = plane(1.0);
        assert_eq!(
            p.vert_count() as i64 - p.edge_count() as i64 + p.face_count() as i64,
            1
        );
        // Genus 1: χ = 0.
        let t = torus(1.0, 0.3, 12, 8);
        assert_eq!(
            t.vert_count() as i64 - t.edge_count() as i64 + t.face_count() as i64,
            0
        );
        assert_eq!(validate(&t), Ok(()));
    }

    #[test]
    fn a_broken_next_link_is_reported_not_hidden() {
        let mut m = cube(1.0);
        let victim = m.half_ids().next().unwrap();
        let wrong = m.half_ids().nth(5).unwrap();
        m.halfs.get_mut(victim.0).expect("live half-edge id").next = wrong;
        let errs = validate(&m).expect_err("a mangled link must be caught");
        assert!(
            errs.iter()
                .any(|e| matches!(e, Violation::NextPrevMismatch { half, .. } if *half == victim)),
            "{errs:?}"
        );
    }

    #[test]
    fn a_stale_vertex_index_is_reported() {
        let mut m = cube(1.0);
        let v = m.vert_ids().next().unwrap();
        let h = m.vert_outgoing(v).unwrap()[0];
        m.verts
            .get_mut(v.0)
            .expect("live vertex id")
            .out
            .retain(|&x| x != h);
        let errs = validate(&m).expect_err("a dropped index entry must be caught");
        assert!(
            errs.iter()
                .any(|e| matches!(e, Violation::VertOutMissing { vert, .. } if *vert == v)),
            "{errs:?}"
        );
    }

    #[test]
    fn a_sharpness_flag_that_disagrees_with_its_twin_is_reported() {
        let mut m = cube(1.0);
        let h = m.half_ids().next().unwrap();
        let sharp = m.is_sharp(h).unwrap();
        m.halfs.get_mut(h.0).expect("live half-edge id").sharp = !sharp;
        let errs = validate(&m).expect_err("one-sided sharpness must be caught");
        assert!(errs
            .iter()
            .any(|e| matches!(e, Violation::SharpDisagrees { .. })));
    }

    #[test]
    fn euler_catches_an_impossible_component() {
        // Delete a face record without touching its half-edges: F drops by one,
        // nothing else moves, so χ becomes odd for a closed surface.
        let mut m = cube(1.0);
        let f = m.face_ids().next().unwrap();
        m.faces.remove(f.0);
        let errs = validate(&m).expect_err("an impossible count must be caught");
        assert!(!errs.is_empty(), "{errs:?}");
    }

    #[test]
    fn a_half_edge_naming_a_dead_face_is_reported() {
        // The audit's M3. `validate` is the independent auditor every property in
        // this crate leans on, so a blind spot here was a blind spot in all of
        // them at once.
        //
        // Two fixtures, because they fail differently and only one of them shows
        // the whole hole:
        //
        // * On the CUBE, removing a face record also disturbs the Euler count, so
        //   something was reported even before this check existed — but only
        //   `EulerInconsistent`, which names the component and not the dangling
        //   reference, and which does not run at all unless every structural
        //   check is already clean.
        // * On the PLANE the component has no faces left, and the Euler check
        //   exempts face-less components (an isolated vertex is a legal staging
        //   state). So `validate` returned a flat `Ok(())` on a mesh whose every
        //   half-edge pointed at a face that did not exist.
        let mut cube_mesh = cube(1.0);
        let f = cube_mesh.face_ids().next().unwrap();
        cube_mesh.faces.remove(f.0);
        let errs = validate(&cube_mesh).expect_err("a dangling face reference must be caught");
        assert!(
            errs.iter()
                .any(|e| matches!(e, Violation::DeadFace { face, .. } if *face == f)),
            "{errs:?}"
        );
        assert_eq!(
            errs.iter()
                .filter(|e| matches!(e, Violation::DeadFace { .. }))
                .count(),
            4,
            "one per corner of the removed quad"
        );

        let mut plane_mesh = plane(2.0);
        let pf = plane_mesh.face_ids().next().unwrap();
        plane_mesh.faces.remove(pf.0);
        let errs = validate(&plane_mesh).expect_err("this one used to read Ok(())");
        assert!(
            errs.iter().all(|e| matches!(e, Violation::DeadFace { .. })),
            "the ONLY thing wrong with this mesh is the dangling reference, \
             which is why nothing else caught it: {errs:?}"
        );
        assert_eq!(errs.len(), 4);
    }
}
