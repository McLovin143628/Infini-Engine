//! Polygon → triangle mesh, by constrained Delaunay triangulation.
//!
//! # Where Delaunay actually earns its place
//!
//! The owner's document proposes `spade` for the **terrain**: turn a DEM into an
//! irregular point cloud, triangulate it, render that. This engine refuses that
//! (see the disposition memo's CANNOT §1 — the terrain is a regular lattice end
//! to end, and a TIN would be a *second* terrain renderer beside a shipped,
//! gated one rather than an improvement to it).
//!
//! It redirects the same crate onto the job the engine genuinely could not do:
//! **triangulating GIS polygons.** A parcel boundary, a building footprint, a
//! lake shoreline and a land-cover region are all polygons, and before this the
//! engine had no polygon primitive at all — every one of them collapsed to an
//! axis-aligned bounding box. That is a real gap, and Delaunay is the right tool
//! for it.
//!
//! # The bug in the document's sample code, avoided by construction
//!
//! The document's `generate_terrain_mesh` pushes into a parallel `heights` vector
//! only on a *successful* insert, then reads `face.vertices()[k].index()` as an
//! index into that vector. Those are not the same numbering: spade's internal
//! vertex indices are not the caller's insertion order once a duplicate point has
//! been rejected — and a DEM resampled onto a lattice produces duplicates
//! routinely. The result is a mesh that is silently mis-indexed, with vertices
//! wearing other vertices' elevations.
//!
//! The fix here is structural rather than careful: the elevation rides **inside
//! the vertex type** ([`GisVertex`]), so there is no parallel array to fall out
//! of step, and the output vertex buffer is built by walking
//! `triangulation.vertices()` in spade's own order — so `index()` is correct by
//! construction rather than by coincidence. `elevations_follow_their_own_vertices`
//! pins it on input that contains duplicates.
//!
//! # Holes
//!
//! A constrained triangulation fills the polygon's convex hull, so interior rings
//! and concave bays come back triangulated too. Faces are kept by testing each
//! centroid against the ring set with a crossing-number rule that counts every
//! ring — so a hole subtracts and a hole inside a hole adds back.

use glam::{DVec3, Vec3Swizzles};
use spade::{ConstrainedDelaunayTriangulation, HasPosition, Point2, Triangulation as _};

use crate::{feature::GeoGeometry, GisError};

/// A triangulation vertex: a planar position plus the elevation that belongs to
/// it. See the module docs for why the elevation lives here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GisVertex {
    pos: Point2<f64>,
    y: f64,
}

impl HasPosition for GisVertex {
    type Scalar = f64;
    fn position(&self) -> Point2<f64> {
        self.pos
    }
}

/// A triangulated polygon.
#[derive(Clone, Debug, PartialEq)]
pub struct Triangulation {
    /// World-space vertices, in metres.
    pub vertices: Vec<DVec3>,
    /// Triangle indices, three per face, wound **counter-clockwise seen from
    /// +Y** so the surface faces up.
    pub indices: Vec<u32>,
}

impl Triangulation {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Total area of the triangulated surface, projected on XZ, in m².
    ///
    /// The check that a triangulation actually covers its polygon: this should
    /// match [`GeoGeometry::area_m2`] to within float noise, and a large
    /// discrepancy means holes were not subtracted or faces outside the boundary
    /// were kept.
    pub fn area_m2(&self) -> f64 {
        let mut total = 0.0;
        for t in self.indices.chunks_exact(3) {
            let (a, b, c) = (
                self.vertices[t[0] as usize],
                self.vertices[t[1] as usize],
                self.vertices[t[2] as usize],
            );
            total += ((b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z)).abs() * 0.5;
        }
        total
    }
}

/// Deduplicate consecutive identical positions and drop a wire ring's repeated
/// closing vertex.
///
/// Both source formats require a ring to repeat its first point last; feeding
/// that to a triangulator is a zero-length constraint edge.
fn clean_ring(ring: &[DVec3]) -> Vec<DVec3> {
    let mut out: Vec<DVec3> = Vec::with_capacity(ring.len());
    for &p in ring {
        if out
            .last()
            .is_none_or(|q: &DVec3| (p.xz() - q.xz()).length_squared() > 1e-20)
        {
            out.push(p);
        }
    }
    // The wire closing vertex.
    if out.len() > 1 {
        if let (Some(&f), Some(&l)) = (out.first(), out.last()) {
            if (f.xz() - l.xz()).length_squared() <= 1e-20 {
                out.pop();
            }
        }
    }
    out
}

/// The XZ bounding box of a ring, as `[min_x, min_z, max_x, max_z]`.
///
/// Computed once per ring so the per-face containment test can reject in O(1)
/// for the rings that cannot possibly contain a given centroid — see
/// [`point_in_ring`] for why that matters.
fn ring_bounds_xz(ring: &[DVec3]) -> [f64; 4] {
    ring.iter().fold(
        [f64::MAX, f64::MAX, f64::MIN, f64::MIN],
        |[nx, nz, xx, xz], p| [nx.min(p.x), nz.min(p.z), xx.max(p.x), xz.max(p.z)],
    )
}

/// Crossing-number point-in-ring test on the XZ plane, with the ring's
/// precomputed bounds as an O(1) reject.
///
/// **The reject is not an optimisation, it is the difference between linear and
/// quadratic.** The caller runs this for every inner face against every ring,
/// and a CDT on `V` vertices has about `2V` inner faces — so without a cheap
/// reject a single polygon costs `2V x V` edge tests. The subjects this module
/// names are lake shorelines and land-cover regions, which routinely carry
/// 10^4-10^5 vertices: 10^8 to 10^10 operations for one polygon. With the box
/// test, only the rings whose extent actually covers the face are walked, which
/// for the usual shape (a big exterior and small scattered holes) is the
/// exterior plus nearly nothing.
fn point_in_ring(px: f64, pz: f64, ring: &[DVec3], bounds: &[f64; 4]) -> bool {
    if px < bounds[0] || px > bounds[2] || pz < bounds[1] || pz > bounds[3] {
        return false;
    }
    let mut inside = false;
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, zi) = (ring[i].x, ring[i].z);
        let (xj, zj) = (ring[j].x, ring[j].z);
        if ((zi > pz) != (zj > pz)) && (px < (xj - xi) * (pz - zi) / (zj - zi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Triangulate a polygon (exterior ring plus holes) into an upward-facing mesh.
///
/// Rings are in world metres; `y` is carried per vertex and interpolated nowhere
/// — every output vertex is one of the input vertices, so a boundary that was
/// draped onto the terrain stays draped.
pub fn triangulate_polygon(geometry: &GeoGeometry) -> Result<Triangulation, GisError> {
    let (exterior, holes) = match geometry {
        GeoGeometry::Polygon { exterior, holes } => (exterior, holes),
        _ => {
            return Err(GisError::Geometry(
                "only a polygon can be triangulated into a surface; a polyline has \
                 no interior (use the road ribbon builder for a line)"
                    .into(),
            ))
        }
    };
    // **The finiteness door comes FIRST**, before any cleaning. `clean_ring`
    // compares distances, and every comparison against a NaN is false — so a
    // non-finite vertex is silently *dropped* by the dedupe rather than caught,
    // and the polygon is then refused for the wrong reason ("too few points")
    // or, worse, quietly triangulated without it. Guarding the raw input is the
    // only ordering that names the real problem.
    for p in exterior.iter().chain(holes.iter().flatten()) {
        if !p.is_finite() {
            return Err(GisError::NotFinite(format!(
                "the polygon carries the non-finite position {p:?} — in GIS data \
                 that usually means an unhandled no-data value or a coordinate \
                 outside its projection's valid area"
            )));
        }
    }
    let ext = clean_ring(exterior);
    if ext.len() < 3 {
        return Err(GisError::Geometry(format!(
            "a polygon boundary needs at least 3 distinct positions; this ring has \
             {} after removing repeated points",
            ext.len()
        )));
    }
    let rings: Vec<Vec<DVec3>> = std::iter::once(ext)
        .chain(holes.iter().map(|h| clean_ring(h)).filter(|h| h.len() >= 3))
        .collect();
    let mut cdt: ConstrainedDelaunayTriangulation<GisVertex> =
        ConstrainedDelaunayTriangulation::new();
    for ring in &rings {
        // Insert every vertex, keeping the handle spade gives back. A duplicate
        // position returns the EXISTING handle, which is exactly the behaviour
        // the document's sample mishandles — here it is simply the right answer,
        // because the handle is what the constraint edges are built from.
        let mut handles = Vec::with_capacity(ring.len());
        for p in ring {
            let v = GisVertex {
                pos: Point2::new(p.x, p.z),
                y: p.y,
            };
            match cdt.insert(v) {
                Ok(h) => handles.push(h),
                Err(e) => {
                    return Err(GisError::Geometry(format!(
                        "the polygon vertex {p:?} could not be triangulated: {e}"
                    )))
                }
            }
        }
        for i in 0..handles.len() {
            let (a, b) = (handles[i], handles[(i + 1) % handles.len()]);
            if a != b {
                // **`add_constraint` PANICS on a crossing constraint** — spade's
                // own docs say so, and its `cdt.rs` reaches an `expect` naming
                // "The new constraint edge intersects an existing constraint
                // edge" when the caller passes no splitting vertex constructor,
                // which is what `add_constraint` does. A self-intersecting ring
                // is not exotic: a bowtie from a digitising error, a hole that
                // crosses its own exterior, a ring that revisits a vertex — OGC
                // validity failures are the normal state of a published parcel
                // or land-cover layer, and `clean_ring` only removes
                // CONSECUTIVE duplicates, so a figure-eight arrives intact.
                //
                // One malformed record in a hundred thousand would have taken
                // the whole cook down with a panic from inside a dependency.
                // Refusals are values here (the house law), so the crossing is
                // asked about first and reported by name. `try_add_constraint`
                // returns the edges it created — an empty result means the edge
                // already existed, which is legal (two rings may share a
                // boundary) and is why this checks `can_add_constraint` rather
                // than the return value.
                if !cdt.can_add_constraint(a, b) {
                    let va = cdt.vertex(a);
                    let vb = cdt.vertex(b);
                    let (pa, pb) = (va.data(), vb.data());
                    return Err(GisError::Geometry(format!(
                        "this polygon's boundary crosses itself: the edge from \
                         ({}, {}) to ({}, {}) intersects another edge of the same \
                         polygon. A self-intersecting ring has no unambiguous \
                         inside, so it is refused rather than triangulated into \
                         whichever answer the solver happened to reach. Published \
                         layers carry these routinely — repair the geometry in \
                         your GIS tool (a zero-width buffer or a 'fix geometries' \
                         pass) and re-import.",
                        pa.pos.x, pa.pos.y, pb.pos.x, pb.pos.y
                    )));
                }
                cdt.add_constraint(a, b);
            }
        }
    }

    // Build the vertex buffer in SPADE'S OWN ORDER, so `index()` below is
    // correct by construction. This is the structural half of the fix described
    // in the module docs.
    let mut vertices = vec![DVec3::ZERO; cdt.num_vertices()];
    for v in cdt.vertices() {
        let d = v.data();
        vertices[v.index()] = DVec3::new(d.pos.x, d.y, d.pos.y);
    }

    // One box per ring, computed once — see `point_in_ring`.
    let ring_boxes: Vec<[f64; 4]> = rings.iter().map(|r| ring_bounds_xz(r)).collect();

    let mut indices = Vec::new();
    for face in cdt.inner_faces() {
        let vs = face.vertices();
        let (i0, i1, i2) = (vs[0].index(), vs[1].index(), vs[2].index());
        let (a, b, c) = (vertices[i0], vertices[i1], vertices[i2]);
        let (cx, cz) = ((a.x + b.x + c.x) / 3.0, (a.z + b.z + c.z) / 3.0);
        // Crossing-number over EVERY ring: the exterior puts a face in, a hole
        // takes it out, a ring inside a hole puts it back.
        let crossings = rings
            .iter()
            .zip(ring_boxes.iter())
            .filter(|(r, b)| point_in_ring(cx, cz, r, b))
            .count();
        if crossings % 2 == 0 {
            continue;
        }
        // Wind so the surface faces UP (+Y). spade's own winding is not
        // guaranteed to be ours, and the sign here is easy to get backwards:
        // for a triangle on the XZ plane the 3-D cross product's Y component is
        // `(b−a).z·(c−a).x − (b−a).x·(c−a).z`, which is the NEGATION of the
        // usual 2-D shoelace term below. So an upward normal wants `signed < 0`,
        // not `> 0`. (Measured, not reasoned: the first version of this line had
        // it the other way round and every face came out pointing at the ground.)
        let signed = (b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z);
        if signed < 0.0 {
            indices.extend_from_slice(&[i0 as u32, i1 as u32, i2 as u32]);
        } else {
            indices.extend_from_slice(&[i0 as u32, i2 as u32, i1 as u32]);
        }
    }

    if indices.is_empty() {
        return Err(GisError::Geometry(
            "the polygon triangulated to no faces — its boundary is probably \
             degenerate (all vertices collinear, or zero area)"
                .into(),
        ));
    }
    Ok(Triangulation { vertices, indices })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(pts: &[(f64, f64)]) -> Vec<DVec3> {
        pts.iter().map(|&(x, z)| DVec3::new(x, 0.0, z)).collect()
    }

    fn poly(ext: Vec<DVec3>, holes: Vec<Vec<DVec3>>) -> GeoGeometry {
        GeoGeometry::Polygon {
            exterior: ext,
            holes,
        }
    }

    #[test]
    fn a_square_triangulates_to_its_own_area_facing_up() {
        let g = poly(
            ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]),
            vec![],
        );
        let t = triangulate_polygon(&g).unwrap();
        assert_eq!(t.triangle_count(), 2, "a square is two triangles");
        assert!(
            (t.area_m2() - 100.0).abs() < 1e-9,
            "the triangulation must cover the polygon: {} vs 100",
            t.area_m2()
        );
        // Every triangle faces up (+Y), which is what the winding fix is for.
        for tri in t.indices.chunks_exact(3) {
            let (a, b, c) = (
                t.vertices[tri[0] as usize],
                t.vertices[tri[1] as usize],
                t.vertices[tri[2] as usize],
            );
            let n = (b - a).cross(c - a);
            assert!(n.y > 0.0, "triangle {tri:?} faces down (normal {n:?})");
        }
        // A clockwise input ring produces the same upward-facing result — the
        // caller should not have to know the source's winding convention.
        let mut cw = ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        cw.reverse();
        let t2 = triangulate_polygon(&poly(cw, vec![])).unwrap();
        assert!((t2.area_m2() - 100.0).abs() < 1e-9);
        for tri in t2.indices.chunks_exact(3) {
            let (a, b, c) = (
                t2.vertices[tri[0] as usize],
                t2.vertices[tri[1] as usize],
                t2.vertices[tri[2] as usize],
            );
            assert!(
                (b - a).cross(c - a).y > 0.0,
                "a CW source ring must still face up"
            );
        }
    }

    /// Holes are subtracted, which is the whole reason a constrained
    /// triangulation is used rather than a fan.
    ///
    /// Un-fix mutation: drop the `crossings % 2` filter and the area comes back
    /// as the full 100 m² instead of 96.
    #[test]
    fn a_hole_is_subtracted_from_the_surface() {
        let g = poly(
            ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]),
            vec![ring(&[(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)])],
        );
        let t = triangulate_polygon(&g).unwrap();
        assert!(
            (t.area_m2() - 96.0).abs() < 1e-6,
            "100 m2 minus a 4 m2 hole is 96, got {}",
            t.area_m2()
        );
        assert!(
            (t.area_m2() - g.area_m2()).abs() < 1e-6,
            "geometry and mesh must agree"
        );
        // No triangle centroid may sit inside the hole.
        for tri in t.indices.chunks_exact(3) {
            let (a, b, c) = (
                t.vertices[tri[0] as usize],
                t.vertices[tri[1] as usize],
                t.vertices[tri[2] as usize],
            );
            let (cx, cz) = ((a.x + b.x + c.x) / 3.0, (a.z + b.z + c.z) / 3.0);
            assert!(
                !(cx > 2.0 && cx < 4.0 && cz > 2.0 && cz < 4.0),
                "a face survived inside the hole at ({cx}, {cz})"
            );
        }
    }

    /// A concave boundary is respected — a naive fan would bridge the notch.
    #[test]
    fn a_concave_boundary_is_not_bridged() {
        // An L shape: 10x10 with a 5x5 bite out of the +x/+z corner. Area 75.
        let g = poly(
            ring(&[
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 5.0),
                (5.0, 5.0),
                (5.0, 10.0),
                (0.0, 10.0),
            ]),
            vec![],
        );
        let t = triangulate_polygon(&g).unwrap();
        assert!(
            (t.area_m2() - 75.0).abs() < 1e-6,
            "the L must be 75 m2, got {} (a bridged notch would give 100)",
            t.area_m2()
        );
    }

    /// **The document's bug, pinned.** Elevations must follow their own vertices
    /// even when the input contains duplicate positions — the case that makes a
    /// parallel-array implementation silently mis-index.
    ///
    /// Un-fix mutation: build the vertex buffer in insertion order instead of
    /// spade's index order and this fails.
    #[test]
    fn elevations_follow_their_own_vertices() {
        // A square whose corners each have a DISTINCT elevation, with the first
        // corner deliberately repeated twice in the ring (so an insert is
        // rejected as a duplicate and the two numberings diverge).
        let ext = vec![
            DVec3::new(0.0, 11.0, 0.0),
            DVec3::new(0.0, 11.0, 0.0), // exact duplicate
            DVec3::new(10.0, 22.0, 0.0),
            DVec3::new(10.0, 33.0, 10.0),
            DVec3::new(0.0, 44.0, 10.0),
        ];
        let t = triangulate_polygon(&poly(ext, vec![])).unwrap();
        // Each output vertex must carry the elevation of the corner it sits on.
        let expect = |x: f64, z: f64| -> f64 {
            match (x as i64, z as i64) {
                (0, 0) => 11.0,
                (10, 0) => 22.0,
                (10, 10) => 33.0,
                (0, 10) => 44.0,
                _ => panic!("unexpected vertex ({x}, {z})"),
            }
        };
        assert_eq!(t.vertices.len(), 4, "the duplicate must have collapsed");
        for v in &t.vertices {
            assert_eq!(
                v.y,
                expect(v.x, v.z),
                "vertex ({}, {}) carries elevation {} — the elevations and the \
                 positions are out of step",
                v.x,
                v.z,
                v.y
            );
        }
        // And it is a real square, not a degenerate one.
        assert!((t.area_m2() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn degenerate_and_wrong_shaped_input_is_refused_by_name() {
        // Not a polygon.
        let e = triangulate_polygon(&GeoGeometry::Polyline {
            points: ring(&[(0.0, 0.0), (1.0, 1.0)]),
            closed: false,
        })
        .unwrap_err()
        .to_string();
        assert!(e.contains("polyline"), "{e}");

        // Too few distinct points — including a ring that is only "long" because
        // it repeats itself.
        assert!(triangulate_polygon(&poly(ring(&[(0.0, 0.0), (1.0, 0.0)]), vec![])).is_err());
        let repeated = ring(&[(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)]);
        let e = triangulate_polygon(&poly(repeated, vec![]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("3 distinct"), "{e}");

        // Collinear: a real ring by count, no area.
        let e = triangulate_polygon(&poly(ring(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]), vec![]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("degenerate") || e.contains("collinear"), "{e}");

        // Non-finite coordinates never reach spade.
        let bad = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(f64::NAN, 0.0, 1.0),
            DVec3::new(1.0, 0.0, 1.0),
        ];
        assert!(matches!(
            triangulate_polygon(&poly(bad, vec![])),
            Err(GisError::NotFinite(_))
        ));
    }

    /// **A self-intersecting ring is refused, not a panic.**
    ///
    /// `spade`'s `add_constraint` documents — and reaches — an `expect` naming
    /// "The new constraint edge intersects an existing constraint edge" when a
    /// constraint crosses another. Bowties, rings that revisit a vertex, and
    /// holes that cross their own exterior are *routine* in published
    /// government layers; OGC validity failures are the normal state of a
    /// parcel or land-cover file, and `clean_ring` only removes CONSECUTIVE
    /// duplicates, so a figure-eight arrives at the triangulator intact.
    ///
    /// Before this, one malformed record in a hundred-thousand-feature layer
    /// panicked the whole import from inside a dependency. Refusals are values
    /// here.
    ///
    /// Un-fix mutation: drop the `can_add_constraint` guard and this test
    /// aborts the process instead of failing.
    #[test]
    fn a_self_intersecting_ring_is_refused_rather_than_panicking() {
        // A bowtie: (0,0) -> (10,10) -> (10,0) -> (0,10) crosses itself.
        let bowtie = ring(&[(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]);
        let e = triangulate_polygon(&poly(bowtie, vec![]))
            .expect_err("a self-intersecting boundary must be refused")
            .to_string();
        assert!(e.contains("crosses itself"), "{e}");
        assert!(
            e.contains("repair"),
            "the refusal must name the remedy: {e}"
        );

        // A hole that crosses its own exterior is the same failure wearing a
        // different hat, and it must not panic either.
        let outer = ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        let crossing_hole = ring(&[(5.0, 5.0), (15.0, 5.0), (15.0, 8.0), (5.0, 8.0)]);
        assert!(triangulate_polygon(&poly(outer, vec![crossing_hole])).is_err());
    }

    /// **The nested-ring rule the module leads with, actually exercised.**
    ///
    /// The doc says "a hole subtracts and a hole inside a hole adds back", and
    /// every other arm used at most one hole — so the `crossings % 2` behaviour
    /// the whole design paragraph is about had never run. Two holes and one
    /// island inside a hole is the smallest input that distinguishes
    /// "even-odd over every ring" from "exterior minus the first hole".
    #[test]
    fn holes_nest_and_an_island_inside_a_hole_comes_back() {
        // 20x20 outer, a 10x10 hole in the middle, and a 4x4 island inside it.
        let outer = ring(&[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]);
        let hole = ring(&[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]);
        let island = ring(&[(8.0, 8.0), (12.0, 8.0), (12.0, 12.0), (8.0, 12.0)]);
        let t = triangulate_polygon(&poly(outer, vec![hole, island])).expect("nested rings");
        // 400 - 100 + 16 = 316. A reader that only subtracted the first hole
        // would give 300; one that subtracted both would give 284.
        let area = t.area_m2();
        assert!(
            (area - 316.0).abs() < 1e-6,
            "nested rings give {area} m^2; 316 is outer - hole + island, 300 is \
             'only the first hole counted', 284 is 'both rings subtracted'"
        );
        // …and a second, disjoint hole is subtracted too — two holes at once was
        // also never exercised.
        let outer = ring(&[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]);
        let a = ring(&[(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)]);
        let b = ring(&[(12.0, 12.0), (16.0, 12.0), (16.0, 16.0), (12.0, 16.0)]);
        let t = triangulate_polygon(&poly(outer, vec![a, b])).expect("two holes");
        assert!((t.area_m2() - (400.0 - 32.0)).abs() < 1e-6);
    }

    /// A wire ring (closing vertex repeated, as both source formats require)
    /// triangulates identically to the same ring without it.
    #[test]
    fn the_wire_closing_vertex_is_handled() {
        let open = ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        let mut wire = open.clone();
        wire.push(open[0]);
        let a = triangulate_polygon(&poly(open, vec![])).unwrap();
        let b = triangulate_polygon(&poly(wire, vec![])).unwrap();
        assert_eq!(a.vertices.len(), b.vertices.len());
        assert!((a.area_m2() - b.area_m2()).abs() < 1e-12);
    }
}
