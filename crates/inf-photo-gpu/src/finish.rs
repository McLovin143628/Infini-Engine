//! **The finish pipeline's texel-space half** (P25.3): a UV'd mesh and a dense
//! reconstruction in, three texture channels out.
//!
//! # What is here and what is not
//!
//! P25.3's pipeline is retopologize → unwrap → bake → write assets. Two of
//! those four need doors this crate cannot see: the unwrapper is
//! `inf-dcc`'s half-edge LSCM (P23.5) and the asset writer is
//! `AssetProject` (Ring 1). The orchestrator therefore lives in
//! `inf_editor_core::photogrammetry`, and **this** module is the part that is
//! pure photogrammetry: given texel positions on a target surface, and given
//! the dense reconstruction those texels are being sampled *from*, produce the
//! three channels.
//!
//! The split is not filing. `inf-dcc` is a **dev**-dependency of `inf-player`
//! so the shipped player never links the modelling kernel, and a Ring-0 crate
//! that depended on it would put that arrangement one `Cargo.toml` edit away
//! from being wrong. What crosses the boundary instead is [`DenseSurface`] —
//! two queries, closest point and occlusion — which Ring 1 implements over
//! `inf_dcc::Bvh` and this module's own tests implement over analytic geometry.
//!
//! # Units, stated once (again)
//!
//! Everything here is in the reconstruction's **baseline units**, the same ones
//! [`crate::DenseReconstruction`] is in: texel positions, ray lengths,
//! occlusion tolerances. The metric scale is one multiplier applied to the
//! *mesh* at the very end of the pipeline, after every bake, because scaling
//! the geometry and the depth maps at different moments is the one way to make
//! an occlusion test compare a metre against a baseline unit and silently
//! reject every sample.
//!
//! # Determinism
//!
//! The crate's rules carry over: every parallel stage is
//! [`inf_core::job::JobPool::parallel_map`] over **rows**, which is an in-order
//! pure map; no float sum is accumulated across workers; the ambient-occlusion
//! direction set is built by a counter hash and `sqrt` with **no
//! transcendental in it at all** (see [`hemisphere_basis`]), so it is the same
//! table on every machine that reads this code.
//!
//! That last point is the P14 law applied where it actually bites. These bytes
//! *are* committed — that is the whole point of P25.3 — and the ruling on which
//! parts of the pipeline the portable family governs is written once, in
//! [`inf_photo`]'s crate docs, under "The line, as P25.3 settled it". The short
//! version: the import is one-shot and content-addressed like a glTF import, so
//! the solvers keep their exemption; a table the bake is a *function of* does
//! not, because a machine that computed it differently would be running a
//! different bake.
//!
//! # The three bakes, and the one thing each measures
//!
//! | channel | source | encoding |
//! |---|---|---|
//! | normal | the dense mesh's own normal at the closest point to the texel | **object space**, `n * 0.5 + 0.5`, linear |
//! | ambient occlusion | a fixed hemisphere of rays against the dense mesh | linear gray |
//! | albedo | every camera that can see the texel, occlusion-tested against its depth map | the photographs' own encoding, sRGB by convention |
//!
//! **Object space, not tangent space, and that is a decision.** A tangent-space
//! normal map needs the consumer and the baker to agree about the tangent
//! frame, down to the handedness convention and the exact interpolation. The
//! interactive renderer does not sample normal maps *at all* yet — that arrives
//! with Phase 26's virtual texturing — so there is no consumer to agree with,
//! and inventing a tangent-space convention now would be choosing one blind and
//! then discovering the mismatch a phase later. Object space has no convention
//! to get wrong: the texel stores the surface normal in the mesh's own frame,
//! which is what the mesh's vertex normals are already in. The material model
//! already carries a `normal_texture` GUID, so the reference is wired and the
//! sampler arrives later.

use glam::{DVec2, DVec3};
use inf_core::job::JobPool;
use inf_photo::rgb::RgbImage;
use inf_photo::sfm::RegisteredCamera;

use crate::sweep::DepthMap;
use crate::tsdf::SurfaceHints;

/// A point on the dense surface, as [`DenseSurface`] reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint {
    /// The closest point itself, in baseline units.
    pub point: DVec3,
    /// The dense surface's unit normal there.
    pub normal: DVec3,
    /// How far the query point was from it.
    pub distance: f64,
}

/// The two questions a bake asks of the dense mesh.
///
/// A trait rather than a concrete type because the acceleration structure this
/// workspace already owns (`inf_dcc::Bvh`, P23.4/P24.2, `f64` and
/// mutation-hardened) lives in a crate the shipped player must not link — see
/// the module docs. Ring 1 implements this over that BVH; the tests here
/// implement it over a plane and a box, where the right answer is arithmetic
/// rather than another traversal.
pub trait DenseSurface: Sync {
    /// The nearest surface point to `p`, or `None` if the surface is empty.
    fn closest(&self, p: DVec3) -> Option<SurfacePoint>;

    /// Whether anything blocks the segment from `origin` along unit
    /// `direction` for `max_distance` baseline units.
    fn occluded(&self, origin: DVec3, direction: DVec3, max_distance: f64) -> bool;
}

/// Which texels of an atlas carry surface, and what surface.
///
/// Positions and normals are only meaningful where [`AtlasSamples::covered`] is
/// set; elsewhere they are zero, which is a value rather than a sentinel
/// because a `NaN` here would be read by a bake and turn into a texel.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasSamples {
    /// Side of the square atlas, in texels.
    pub size: u32,
    /// Surface position per texel, in **baseline units**.
    pub position: Vec<DVec3>,
    /// Interpolated target-mesh normal per texel, unit where covered.
    pub normal: Vec<DVec3>,
    /// Whether a triangle covered this texel's centre.
    pub covered: Vec<bool>,
    /// Texels a second triangle also covered, and therefore did not get.
    ///
    /// A well-packed atlas has none. A chart that folded on itself has some,
    /// and this is how a caller learns that rather than by looking at the
    /// texture. First writer wins, in triangle index order — deterministic, and
    /// stated so nobody has to read the loop to find out.
    pub overlaps: usize,
    /// Triangles whose UVs enclosed no texel centre at all.
    ///
    /// Below roughly one texel of area a triangle falls between the sample
    /// points. Its texels come back from [`dilate`] like any other gutter, but
    /// the count is what says how much of the mesh was never directly sampled.
    pub unsampled_triangles: usize,
}

impl AtlasSamples {
    /// How many texels carry surface.
    pub fn covered_count(&self) -> usize {
        self.covered.iter().filter(|c| **c).count()
    }
}

/// Rasterize a UV'd triangle mesh into an atlas of surface samples.
///
/// `positions` are in baseline units, `uvs` are in `[0, 1]` (anything outside is
/// simply outside the atlas and contributes nothing), `normals` are per vertex,
/// and all three are parallel arrays indexed by `indices`.
///
/// # The sampling rule
///
/// A texel is covered when its **centre** — `(x + 0.5) / size` — falls inside a
/// triangle's UV image. That is the same pixel-centre convention the rest of
/// Phase 25 is on, and it is what makes the barycentric interpolation exact at
/// the point the texture will be sampled at. It also means the outermost half
/// texel of every chart is *not* covered, which is the gutter every atlas has
/// and which [`dilate`] fills.
///
/// # The fill rule, which is not decoration
///
/// Two triangles that share an edge share every texel centre that lands
/// *exactly on* that edge. Inclusive barycentrics (`w >= 0` on all three) give
/// both of them those texels, so a quad split into two triangles reports its
/// whole diagonal as an overlap — **measured**: 16 of 256 texels on a 16x16
/// atlas, on the first run of this function's own test — and `overlaps` stops
/// being a signal about folded charts and becomes a count of how many diagonals
/// the mesh has.
///
/// So the edges use the standard **top-left rule**: the triangle is first wound
/// to a positive area in texel space, then a texel exactly on an edge is
/// included only if that edge is the triangle's top or left one. Every interior
/// point belongs to exactly one triangle, shared edges included, and `overlaps`
/// means what it says.
pub fn rasterize_atlas(
    positions: &[DVec3],
    normals: &[DVec3],
    uvs: &[[f32; 2]],
    indices: &[u32],
    size: u32,
) -> AtlasSamples {
    let n = size as usize * size as usize;
    let mut out = AtlasSamples {
        size,
        position: vec![DVec3::ZERO; n],
        normal: vec![DVec3::ZERO; n],
        covered: vec![false; n],
        overlaps: 0,
        unsampled_triangles: 0,
    };
    if size == 0 {
        return out;
    }
    let s = size as f64;
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        // UV space scaled into texel space, still on the pixel-centre
        // convention: texel `x` is sampled at `x + 0.5`.
        let uv = |i: usize| DVec2::new(uvs[i][0] as f64 * s, uvs[i][1] as f64 * s);
        let raw_area = (uv(i1).x - uv(i0).x) * (uv(i2).y - uv(i0).y)
            - (uv(i2).x - uv(i0).x) * (uv(i1).y - uv(i0).y);
        if !(raw_area.abs() > 0.0) || !raw_area.is_finite() {
            // A degenerate UV triangle carries no direction and no interior.
            out.unsampled_triangles += 1;
            continue;
        }
        // Wound to a positive area so the top-left rule has one orientation to
        // reason about. Swapping two vertices swaps their barycentrics with
        // them, so the interpolation is untouched.
        let (i0, i1, i2) = if raw_area < 0.0 {
            (i0, i2, i1)
        } else {
            (i0, i1, i2)
        };
        let (a, b, c) = (uv(i0), uv(i1), uv(i2));
        let area = raw_area.abs();
        // Edge k is the one opposite vertex k, and `w_k == 0` puts the texel on
        // it. Interior lies along `(-e.y, e.x)`, so with y running down the
        // atlas a top edge is horizontal with `e.x > 0` and a left edge has
        // `e.y < 0`.
        let top_left = |e: DVec2| e.y < 0.0 || (e.y == 0.0 && e.x > 0.0);
        let keep = [top_left(c - b), top_left(a - c), top_left(b - a)];
        let lo_x = a.x.min(b.x).min(c.x).floor().max(0.0) as i64;
        let hi_x = (a.x.max(b.x).max(c.x).ceil() as i64).min(size as i64 - 1);
        let lo_y = a.y.min(b.y).min(c.y).floor().max(0.0) as i64;
        let hi_y = (a.y.max(b.y).max(c.y).ceil() as i64).min(size as i64 - 1);
        let mut wrote = 0usize;
        for ty in lo_y..=hi_y {
            for tx in lo_x..=hi_x {
                let p = DVec2::new(tx as f64 + 0.5, ty as f64 + 0.5);
                // Barycentrics from the same cross products the area came from,
                // so a point exactly on an edge is decided by one sign rule.
                let w0 = ((b.x - p.x) * (c.y - p.y) - (c.x - p.x) * (b.y - p.y)) / area;
                let w1 = ((c.x - p.x) * (a.y - p.y) - (a.x - p.x) * (c.y - p.y)) / area;
                let w2 = 1.0 - w0 - w1;
                let inside = [w0, w1, w2]
                    .iter()
                    .zip(keep)
                    .all(|(w, top_left)| *w > 0.0 || (*w == 0.0 && top_left));
                if !inside {
                    continue;
                }
                let index = ty as usize * size as usize + tx as usize;
                if out.covered[index] {
                    out.overlaps += 1;
                    continue;
                }
                out.position[index] = positions[i0] * w0 + positions[i1] * w1 + positions[i2] * w2;
                let nrm = normals[i0] * w0 + normals[i1] * w1 + normals[i2] * w2;
                let len = nrm.length();
                out.normal[index] = if len > 0.0 { nrm / len } else { DVec3::Y };
                out.covered[index] = true;
                wrote += 1;
            }
        }
        if wrote == 0 {
            out.unsampled_triangles += 1;
        }
    }
    out
}

/// How the three bakes are tuned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BakeConfig {
    /// Atlas side in texels. Square, and a power of two by convention rather
    /// than by requirement — BC block compression wants a multiple of four and
    /// the mip chain wants powers of two.
    pub size: u32,
    /// How far from a texel the dense surface may be, in **voxels of the
    /// fusion grid**, before the normal bake gives up on it and keeps the
    /// target mesh's own normal.
    ///
    /// In voxels rather than units because that is the scale at which the dense
    /// surface is actually resolved: a retopologized mesh sits inside its own
    /// decimation error of the dense one, and the bound that matters is "did we
    /// land on the surface or in the next room".
    pub normal_search_voxels: f64,
    /// Ambient-occlusion rays per texel.
    pub ao_rays: usize,
    /// How far an occlusion ray travels, in **voxels**. Short rays measure
    /// creases; long ones measure the room.
    pub ao_distance_voxels: f64,
    /// How far a ray starts from the surface, in **voxels**, so it does not
    /// immediately hit the triangle it came from.
    pub ao_bias_voxels: f64,
    /// How much closer a view's depth may be than the texel before the texel is
    /// judged **occluded** from that view, as a fraction of the depth.
    ///
    /// Relative rather than absolute because a stereo depth error grows with
    /// depth, and a fixed tolerance would be strict near the camera and useless
    /// far from it.
    pub occlusion_tolerance: f32,
    /// Minimum blended weight for a texel to count as **well seen** — the set
    /// the de-lighting solve fits on and the coverage advisory measures.
    pub well_seen_weight: f32,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            size: 1024,
            normal_search_voxels: 4.0,
            ao_rays: 32,
            ao_distance_voxels: 24.0,
            ao_bias_voxels: 0.75,
            occlusion_tolerance: 0.05,
            well_seen_weight: 1.0e-3,
        }
    }
}

/// A baked channel: three `f64` components per texel plus the mask that says
/// which of them mean anything.
///
/// `f64` rather than bytes because two stages run *after* a bake — dilation and
/// de-lighting — and quantizing between them would round twice. The narrowing
/// to `u8` happens once, in [`Channel::to_rgb8`], and that is the seam.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    /// Atlas side.
    pub size: u32,
    /// Three components per texel, row-major.
    pub data: Vec<[f64; 3]>,
    /// Whether this texel was written by a bake rather than left at its default.
    pub valid: Vec<bool>,
}

impl Channel {
    /// A channel of `fill` everywhere, nothing valid.
    pub fn new(size: u32, fill: [f64; 3]) -> Self {
        let n = size as usize * size as usize;
        Self {
            size,
            data: vec![fill; n],
            valid: vec![false; n],
        }
    }

    /// How many texels a bake actually wrote.
    pub fn valid_count(&self) -> usize {
        self.valid.iter().filter(|v| **v).count()
    }

    /// Narrow to interleaved 8-bit RGBA with an opaque alpha — the layout
    /// `inf_material::texture_from_rgba8` takes.
    ///
    /// **The named narrowing seam.** Everything upstream of this line is `f64`;
    /// everything downstream is a texture byte. Rounding is round-half-away-
    /// from-zero on a `[0, 1]` clamp, which is exact and has one answer.
    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 4);
        for c in &self.data {
            for v in c {
                out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            out.push(255);
        }
        out
    }
}

/// **Area-weighted vertex normals from the triangulation**, oriented to agree
/// with `hint`.
///
/// # Why the fused mesh's own normals are not used
///
/// [`crate::DenseMesh::normals`] come from the Surface-Nets mesher, which takes
/// them as the central-difference gradient of the fused field. That field is a
/// **projective** TSDF — each view writes the difference between a voxel's depth
/// and the depth it measured *along that view's ray* — so its gradient points
/// along the rays that wrote it rather than along the surface. Fusing several
/// views averages the bias down but does not remove it, and all six stations of
/// a normal capture look at the subject from roughly one side.
///
/// **MEASURED on the committed fixture**, against the analytic scene: the fused
/// mesh's own normals are **50.6° from the truth at the median** and 81.2° at
/// p90 — which is not a normal map, it is noise with a normal's type. The
/// vertex *positions* from the same mesher are good (P25.2 measured them at 1.59
/// voxels of the truth surface at the mean), so the triangulation knows the
/// shape even where the gradient does not, and this reads the normals off the
/// triangulation instead.
///
/// The cross product is left **unnormalized** before accumulation, which
/// area-weights each face's contribution — the standard construction, and the
/// one that keeps a sliver from counting as much as the quad beside it.
///
/// `smoothing` then runs that many rounds of neighbour averaging over the mesh's
/// own edges. A fused surface is **rough at the scale of a voxel** — P25.2's
/// depth error is 0.87 voxels at the median, so a triangle one voxel across can
/// be tilted tens of degrees by noise alone — and a normal read off it inherits
/// every bit of that. MEASURED on the fixture: **45.1° at the median raw, 36.4°
/// after two rounds, 33.2° after four, 31.8° after six.** It converges because
/// averaging removes noise and not shape; four rounds is where the return stops
/// paying for the detail it costs.
///
/// `hint` supplies only the **sign**: a fused scan is a slab with two sheets
/// whose normals genuinely oppose, so a global orientation would be wrong on one
/// of them. The gradient's *direction* is unreliable and its *sign* — inside
/// versus outside — is not, which is exactly the part this uses. The smoothing
/// carries the same care: a neighbour's normal is flipped onto the vertex's own
/// side before it is added, so averaging across the slab's rim does not cancel
/// two correct normals into nothing.
pub fn geometric_normals(
    positions: &[DVec3],
    indices: &[u32],
    hint: &[[f32; 3]],
    smoothing: usize,
) -> Vec<DVec3> {
    let mut acc = vec![DVec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let n = (positions[i1] - positions[i0]).cross(positions[i2] - positions[i0]);
        if !n.is_finite() {
            continue;
        }
        acc[i0] += n;
        acc[i1] += n;
        acc[i2] += n;
    }
    let mut out: Vec<DVec3> = acc
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let h = hint
                .get(i)
                .map(|v| DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64))
                .unwrap_or(DVec3::ZERO);
            let g = n.normalize_or_zero();
            if g == DVec3::ZERO {
                // An isolated vertex, or one whose faces cancelled exactly.
                return h.normalize_or(DVec3::Y);
            }
            if h.length_squared() > 0.0 && g.dot(h) < 0.0 {
                -g
            } else {
                g
            }
        })
        .collect();
    for _ in 0..smoothing {
        let previous = out.clone();
        let mut sum = previous.clone();
        for tri in indices.chunks_exact(3) {
            let idx = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            if idx.iter().any(|i| *i >= previous.len()) {
                continue;
            }
            for a in 0..3 {
                for b in 0..3 {
                    if a == b {
                        continue;
                    }
                    let (i, j) = (idx[a], idx[b]);
                    // Flipped onto the vertex's own side first — see the docs.
                    let n = if previous[j].dot(previous[i]) < 0.0 {
                        -previous[j]
                    } else {
                        previous[j]
                    };
                    sum[i] += n;
                }
            }
        }
        for (i, n) in sum.iter().enumerate() {
            out[i] = n.normalize_or(previous[i]);
        }
    }
    out
}

/// A fixed set of unit vectors on the sphere, built without a transcendental.
///
/// # Why not a spiral
///
/// The usual answers — a Fibonacci spiral, a concentric disc map, spherical
/// coordinates — all need `sin`/`cos`, and this table is part of the *rule* a
/// committed texture was baked by (see the module docs). So this is the shape
/// the workspace already uses for the same reason (P22's `unit_dir`): **reject
/// a cube sample that falls outside the unit ball, then normalize**. The only
/// non-arithmetic operation is `sqrt`, which is IEEE-754 correctly rounded and
/// identical everywhere.
///
/// Rejection makes the *count* of hash draws depend on the draws, which sounds
/// like an order dependency and is not one: the draws are a counter sequence
/// from a fixed seed, so the `k`-th accepted vector is a pure function of `k`.
pub fn hemisphere_basis(count: usize) -> Vec<DVec3> {
    let mut out = Vec::with_capacity(count);
    let base = inf_photo::hash::Hash64::new(0x50_25_03_A0_1E_5A_11_07);
    let mut draw = 0u64;
    // The cap is checked **before** the draw, not after the push. Written the
    // other way round a `continue` skips it, so the one regime it exists for —
    // a hash that rejects every sample — is the one regime it cannot stop.
    while out.len() < count && draw <= 1_000_000 {
        let h = base.mix_u64(draw);
        draw += 1;
        let x = h.unit() * 2.0 - 1.0;
        let y = h.mix_u64(1).unit() * 2.0 - 1.0;
        let z = h.mix_u64(2).unit() * 2.0 - 1.0;
        let v = DVec3::new(x, y, z);
        let l2 = v.length_squared();
        // Outside the ball, or so close to the centre that normalizing it is
        // numerical noise rather than a direction.
        if !(1.0e-6..=1.0).contains(&l2) {
            continue;
        }
        out.push(v / l2.sqrt());
    }
    out
}

/// Bake object-space normals from the dense surface onto the atlas.
///
/// Texels whose closest dense point is further than
/// [`BakeConfig::normal_search_voxels`] keep the **target mesh's own** normal
/// and are counted: a texel that found nothing is not a texel with no normal,
/// it is one whose detail could not be transferred, and writing a flat
/// `(0, 0, 1)` there would put a visible seam in the middle of a smooth surface.
pub fn bake_normals(
    atlas: &AtlasSamples,
    surface: &dyn DenseSurface,
    voxel_size: f64,
    cfg: &BakeConfig,
    pool: &JobPool,
) -> (Channel, NormalBakeReport) {
    let size = atlas.size;
    let limit = cfg.normal_search_voxels * voxel_size;
    let rows: Vec<u32> = (0..size).collect();
    type NormalRow = (Vec<[f64; 3]>, Vec<bool>, usize, f64);
    let baked: Vec<NormalRow> = pool.parallel_map_ref(&rows, |&y| {
        let mut data = vec![[0.5, 0.5, 1.0]; size as usize];
        let mut valid = vec![false; size as usize];
        let mut fallbacks = 0usize;
        let mut worst = 0.0f64;
        for x in 0..size as usize {
            let index = y as usize * size as usize + x;
            if !atlas.covered[index] {
                continue;
            }
            let target = atlas.normal[index];
            let n = match surface.closest(atlas.position[index]) {
                Some(hit) if hit.distance <= limit && hit.normal.length_squared() > 0.0 => {
                    worst = worst.max(hit.distance);
                    // The dense normal, flipped onto the side the target mesh
                    // faces. Surface Nets orients from the SDF gradient and a
                    // retopologized shell can sit on either side of it; a normal
                    // that disagrees with the surface it is written onto is an
                    // inside-out patch, not detail.
                    if hit.normal.dot(target) < 0.0 {
                        -hit.normal
                    } else {
                        hit.normal
                    }
                }
                _ => {
                    fallbacks += 1;
                    target
                }
            };
            data[x] = [n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, n.z * 0.5 + 0.5];
            valid[x] = true;
        }
        (data, valid, fallbacks, worst)
    });
    let mut channel = Channel::new(size, [0.5, 0.5, 1.0]);
    let mut report = NormalBakeReport::default();
    for (y, (data, valid, fallbacks, worst)) in baked.into_iter().enumerate() {
        let base = y * size as usize;
        channel.data[base..base + size as usize].copy_from_slice(&data);
        channel.valid[base..base + size as usize].copy_from_slice(&valid);
        report.fallbacks += fallbacks;
        report.worst_transfer_distance = report.worst_transfer_distance.max(worst);
    }
    report.texels = channel.valid_count();
    (channel, report)
}

/// What the normal bake found.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NormalBakeReport {
    /// Texels written.
    pub texels: usize,
    /// Texels that kept the target mesh's normal because the dense surface was
    /// too far away.
    pub fallbacks: usize,
    /// The furthest a transferred normal came from, in baseline units.
    pub worst_transfer_distance: f64,
}

/// Bake ambient occlusion by casting a fixed hemisphere of rays per texel.
///
/// The directions are `normalize(n + s)` for `s` in [`hemisphere_basis`] — the
/// standard cosine-weighted construction, which needs no trigonometry and puts
/// more rays where they matter. A sample whose `s` lands antipodal to `n` (the
/// sum is nearly zero) falls back to `n` itself rather than becoming a
/// direction chosen by rounding.
pub fn bake_ao(
    atlas: &AtlasSamples,
    surface: &dyn DenseSurface,
    voxel_size: f64,
    cfg: &BakeConfig,
    pool: &JobPool,
) -> Channel {
    let size = atlas.size;
    let basis = hemisphere_basis(cfg.ao_rays.max(1));
    let distance = cfg.ao_distance_voxels * voxel_size;
    let bias = cfg.ao_bias_voxels * voxel_size;
    let rows: Vec<u32> = (0..size).collect();
    let baked: Vec<(Vec<[f64; 3]>, Vec<bool>)> = pool.parallel_map_ref(&rows, |&y| {
        let mut data = vec![[1.0, 1.0, 1.0]; size as usize];
        let mut valid = vec![false; size as usize];
        for x in 0..size as usize {
            let index = y as usize * size as usize + x;
            if !atlas.covered[index] {
                continue;
            }
            let n = atlas.normal[index];
            let origin = atlas.position[index] + n * bias;
            let mut blocked = 0usize;
            for s in &basis {
                let d = n + *s;
                let l = d.length();
                let dir = if l > 1.0e-6 { d / l } else { n };
                if surface.occluded(origin, dir, distance) {
                    blocked += 1;
                }
            }
            let open = 1.0 - blocked as f64 / basis.len() as f64;
            data[x] = [open, open, open];
            valid[x] = true;
        }
        (data, valid)
    });
    let mut channel = Channel::new(size, [1.0, 1.0, 1.0]);
    for (y, (data, valid)) in baked.into_iter().enumerate() {
        let base = y * size as usize;
        channel.data[base..base + size as usize].copy_from_slice(&data);
        channel.valid[base..base + size as usize].copy_from_slice(&valid);
    }
    channel
}

/// One registered camera's photograph, its depth map and its per-pixel fusion
/// weights — everything the albedo bake needs from one view.
pub struct AlbedoView<'a> {
    /// The camera's pose and intrinsics, as the reconstruction registered them.
    pub camera: &'a RegisteredCamera,
    /// The original photograph, in colour.
    pub image: &'a RgbImage,
    /// The view's depth map, used **only** to answer "can this camera see that
    /// point" — the reason P25.2 hands depth maps back at all.
    pub depth: &'a DepthMap,
    /// The view's per-pixel `confidence * incidence`, used as the blend weight.
    pub hints: &'a SurfaceHints,
}

/// What the albedo bake did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AlbedoBakeReport {
    /// Covered texels at least one camera contributed colour to.
    pub seen: usize,
    /// Covered texels **no** camera contributed colour to. These are filled by
    /// [`dilate`], which means they end up looking plausible — so this is the
    /// number the orchestrator must raise as an advisory rather than let a
    /// smooth-looking texture imply coverage nobody had.
    pub unseen: usize,
    /// Texels whose blended weight cleared [`BakeConfig::well_seen_weight`].
    pub well_seen: usize,
    /// Total (texel, view) pairs that contributed.
    pub samples: usize,
    /// (texel, view) pairs rejected because the view's own depth put something
    /// closer along the same ray.
    pub occluded: usize,
    /// (texel, view) pairs where the view had no depth at that pixel, so it
    /// could offer no evidence either way.
    pub no_evidence: usize,
}

/// Project every covered texel into every camera, keep what is visible, and
/// blend.
///
/// # The occlusion rule, which is why depth maps are an output
///
/// A texel's world point is projected into a view; the view's own depth at that
/// pixel is read; if the depth is **closer** than the texel by more than
/// [`BakeConfig::occlusion_tolerance`] of itself, something stands between the
/// camera and the texel and the sample is dropped. A pixel with **no** depth
/// offers no evidence in either direction and is also dropped, counted
/// separately: accepting it would let a camera paint the far side of an object
/// it cannot see, and those are exactly the pixels a plane sweep fails on
/// (untextured, occluded, or grazing).
///
/// # The weight
///
/// `hints.weights` (`confidence * cos(incidence)`, P25.2's own blend weight)
/// times the incidence at the **texel's** normal. Both terms are needed: the
/// first knows how well the *stereo* did there, the second knows how squarely
/// this camera faces the *surface being baked*, and a retopologized surface is
/// not the same surface the depth map's normal was estimated on.
pub fn bake_albedo(
    atlas: &AtlasSamples,
    views: &[AlbedoView<'_>],
    cfg: &BakeConfig,
    pool: &JobPool,
) -> (Channel, Channel, AlbedoBakeReport) {
    let size = atlas.size;
    let rows: Vec<u32> = (0..size).collect();
    type Row = (Vec<[f64; 3]>, Vec<bool>, Vec<[f64; 3]>, AlbedoBakeReport);
    let baked: Vec<Row> = pool.parallel_map_ref(&rows, |&y| {
        let mut data = vec![[0.0f64; 3]; size as usize];
        let mut valid = vec![false; size as usize];
        let mut weights = vec![[0.0f64; 3]; size as usize];
        let mut report = AlbedoBakeReport::default();
        for x in 0..size as usize {
            let index = y as usize * size as usize + x;
            if !atlas.covered[index] {
                continue;
            }
            let p = atlas.position[index];
            let n = atlas.normal[index];
            let mut sum = [0.0f64; 3];
            let mut weight = 0.0f64;
            for view in views {
                let Some((colour, w)) = sample_view(view, p, n, cfg, &mut report) else {
                    continue;
                };
                sum[0] += colour[0] * w;
                sum[1] += colour[1] * w;
                sum[2] += colour[2] * w;
                weight += w;
                report.samples += 1;
            }
            if weight > 0.0 {
                data[x] = [sum[0] / weight, sum[1] / weight, sum[2] / weight];
                valid[x] = true;
                weights[x] = [weight; 3];
                report.seen += 1;
                if weight >= cfg.well_seen_weight as f64 {
                    report.well_seen += 1;
                }
            } else {
                report.unseen += 1;
            }
        }
        (data, valid, weights, report)
    });
    let mut channel = Channel::new(size, [0.0; 3]);
    let mut weight_channel = Channel::new(size, [0.0; 3]);
    let mut report = AlbedoBakeReport::default();
    for (y, (data, valid, weights, row)) in baked.into_iter().enumerate() {
        let base = y * size as usize;
        channel.data[base..base + size as usize].copy_from_slice(&data);
        channel.valid[base..base + size as usize].copy_from_slice(&valid);
        weight_channel.data[base..base + size as usize].copy_from_slice(&weights);
        weight_channel.valid[base..base + size as usize].copy_from_slice(&valid);
        report.seen += row.seen;
        report.unseen += row.unseen;
        report.well_seen += row.well_seen;
        report.samples += row.samples;
        report.occluded += row.occluded;
        report.no_evidence += row.no_evidence;
    }
    (channel, weight_channel, report)
}

/// One (texel, view) pair: `None` when the camera cannot be shown to see it.
fn sample_view(
    view: &AlbedoView<'_>,
    p: DVec3,
    n: DVec3,
    cfg: &BakeConfig,
    report: &mut AlbedoBakeReport,
) -> Option<([f64; 3], f64)> {
    let camera_space = view.camera.pose.transform(p);
    if !(camera_space.z > 0.0) {
        return None;
    }
    let px = view.camera.intrinsics.project(camera_space)?;
    if !view.camera.intrinsics.contains(px) {
        return None;
    }
    let w = view.depth.width as usize;
    let h = view.depth.height as usize;
    // The depth map is addressed at the **nearest** pixel: a depth map is not an
    // interpolable field. Two neighbouring depths across an occlusion boundary
    // average to a depth that belongs to neither surface, and this is a
    // visibility test, where being between two answers is the wrong answer.
    let ix = (px.x.round() as i64).clamp(0, w as i64 - 1) as usize;
    let iy = (px.y.round() as i64).clamp(0, h as i64 - 1) as usize;
    let i = iy * w + ix;
    let d = view.depth.depth[i];
    if !(d > 0.0) {
        report.no_evidence += 1;
        return None;
    }
    let z = camera_space.z as f32;
    if z - d > cfg.occlusion_tolerance * d {
        report.occluded += 1;
        return None;
    }
    // How squarely this camera faces the surface being baked. `to_camera` is
    // built from the pose rather than from the depth, so it is right even where
    // the depth map's own normal estimate was not taken.
    let to_camera = (view.camera.pose.center() - p).normalize_or_zero();
    let incidence = n.dot(to_camera);
    if !(incidence > 0.0) {
        return None;
    }
    let weight = view.hints.weights[i] as f64 * incidence;
    if !(weight > 0.0) {
        return None;
    }
    Some((bilinear_rgb(view.image, px), weight))
}

/// Bilinear colour lookup on the pixel-centre convention, clamp-to-edge.
///
/// Interpolated where the depth map was not, and for the opposite reason: this
/// is a *radiance* sample, which is a continuous field across a surface, and
/// nearest-neighbour here would quantize the albedo to the photograph's own
/// pixel grid at whatever magnification the atlas happens to have.
fn bilinear_rgb(image: &RgbImage, px: DVec2) -> [f64; 3] {
    let x0 = px.x.floor();
    let y0 = px.y.floor();
    let tx = px.x - x0;
    let ty = px.y - y0;
    let (x0, y0) = (x0 as i64, y0 as i64);
    let p00 = image.clamped(x0, y0);
    let p10 = image.clamped(x0 + 1, y0);
    let p01 = image.clamped(x0, y0 + 1);
    let p11 = image.clamped(x0 + 1, y0 + 1);
    let mut out = [0.0f64; 3];
    for c in 0..3 {
        let top = p00[c] as f64 + (p10[c] as f64 - p00[c] as f64) * tx;
        let bottom = p01[c] as f64 + (p11[c] as f64 - p01[c] as f64) * tx;
        out[c] = (top + (bottom - top) * ty) / 255.0;
    }
    out
}

/// Fill every invalid texel from its valid neighbours by **pull-push**.
///
/// Build a weighted pyramid down to 1x1 by averaging valid texels (the pull),
/// then walk back up filling invalid texels from the level above (the push).
/// The cost is that of a mip chain and it fills a hole of any size, which
/// iterative dilation does not — a chart the cameras never saw at all is one
/// hole the width of the chart, and a 3x3 pass would need as many iterations as
/// the hole is wide.
///
/// Everything comes back valid. A caller who needs to know *which* texels were
/// invented keeps the mask it passed in; the returned channel is a texture, and
/// a texture with holes in it is a texture with garbage in it.
pub fn dilate(channel: &Channel) -> Channel {
    let size = channel.size;
    if size == 0 {
        return channel.clone();
    }
    // ── pull ────────────────────────────────────────────────────────────────
    // Level 0 is the input, weighted 1 where valid and 0 where not. Each
    // coarser level is a 2x2 weighted sum, so a level's weight is "how many of
    // the texels under me had data".
    let mut levels: Vec<(u32, Vec<[f64; 3]>, Vec<f64>)> = Vec::new();
    let mut w = size;
    let mut data: Vec<[f64; 3]> = channel.data.clone();
    let mut weight: Vec<f64> = channel
        .valid
        .iter()
        .map(|v| if *v { 1.0 } else { 0.0 })
        .collect();
    for (i, d) in data.iter_mut().enumerate() {
        if weight[i] == 0.0 {
            *d = [0.0; 3];
        }
    }
    levels.push((w, data.clone(), weight.clone()));
    while w > 1 {
        let cw = w.div_ceil(2);
        let mut cd = vec![[0.0f64; 3]; cw as usize * cw as usize];
        let mut cwt = vec![0.0f64; cw as usize * cw as usize];
        for y in 0..cw {
            for x in 0..cw {
                let mut acc = [0.0f64; 3];
                let mut acw = 0.0f64;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let sx = x * 2 + dx;
                        let sy = y * 2 + dy;
                        if sx >= w || sy >= w {
                            continue;
                        }
                        let si = sy as usize * w as usize + sx as usize;
                        let sw = weight[si];
                        if sw <= 0.0 {
                            continue;
                        }
                        for c in 0..3 {
                            acc[c] += data[si][c] * sw;
                        }
                        acw += sw;
                    }
                }
                let ci = y as usize * cw as usize + x as usize;
                if acw > 0.0 {
                    for c in 0..3 {
                        cd[ci][c] = acc[c] / acw;
                    }
                    cwt[ci] = acw;
                }
            }
        }
        levels.push((cw, cd.clone(), cwt.clone()));
        w = cw;
        data = cd;
        weight = cwt;
    }
    // ── push ────────────────────────────────────────────────────────────────
    for level in (0..levels.len() - 1).rev() {
        let (cw, coarse, coarse_w) = {
            let (cw, cd, cwt) = &levels[level + 1];
            (*cw, cd.clone(), cwt.clone())
        };
        let (fw, fine, fine_w) = &mut levels[level];
        let fw = *fw;
        for y in 0..fw {
            for x in 0..fw {
                let fi = y as usize * fw as usize + x as usize;
                if fine_w[fi] > 0.0 {
                    continue;
                }
                let ci = (y / 2) as usize * cw as usize + (x / 2) as usize;
                if coarse_w[ci] <= 0.0 {
                    continue;
                }
                fine[fi] = coarse[ci];
                fine_w[fi] = coarse_w[ci];
            }
        }
    }
    let (_, filled, filled_w) = levels.swap_remove(0);
    Channel {
        size,
        data: filled,
        valid: filled_w.iter().map(|w| *w > 0.0).collect(),
    }
}

/// How the de-lighting solve is configured. **Off by default** — see
/// [`delight`] for why.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelightConfig {
    /// Whether to run at all.
    pub enabled: bool,
    /// Whether the photographs are sRGB-encoded and must be linearized before
    /// a division by irradiance means anything.
    ///
    /// True for real photographs. The synthetic fixture renders in linear light
    /// on purpose, so its measurements isolate the *estimator* rather than the
    /// transfer function.
    pub input_is_srgb: bool,
    /// Candidate light directions searched.
    pub directions: usize,
    /// The directional term must explain at least this fraction of the
    /// residual a constant model leaves, or the solve reports **no light found**
    /// and divides nothing out.
    ///
    /// This is the guard that stops de-lighting from "correcting" a scene whose
    /// brightness variation is mostly *albedo*, where the best fit is noise and
    /// dividing by it stamps that noise into the texture.
    ///
    /// **0.25, and the number is a measurement.** On the P25.3 fixture — a
    /// textured scene under a light the renderer applied exactly, so the answer
    /// is known — the best directional fit explains **6.4%** of the brightness
    /// variation and recovers a directional term **more than eight times too
    /// small** (0.070 against a true 0.600 in luma). The other 93.6% is the dot
    /// texture and the plane tints. Applied anyway it made the albedo error
    /// **worse**: p50 0.0680 to 0.0700 against the known albedo. A threshold
    /// under 0.064 would have accepted that.
    ///
    /// The move is small because the fit is small — a directional term of 0.070
    /// against an ambient of 0.260 makes the mean-preserving divide nearly the
    /// identity — so the case for the guard is that the number is *noise*, not
    /// that trusting it ruins a texture. See
    /// `docs/memos/p25-delighting-v1.md`, whose closing section records that the
    /// figures here were once 8.2% / 0.078 / 0.255 and why they were wrong.
    pub min_explained: f64,
}

impl Default for DelightConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            input_is_srgb: true,
            directions: 64,
            min_explained: 0.25,
        }
    }
}

/// What the de-lighting solve found.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DelightReport {
    /// Whether the division was actually applied.
    pub applied: bool,
    /// Texels the fit was made over.
    pub texels: usize,
    /// The ambient term, in the linear space the fit was made in.
    pub ambient: f64,
    /// The directional term at full incidence.
    pub directional: f64,
    /// The chosen light direction (from a surface toward the light).
    pub direction: [f64; 3],
    /// Fraction of the constant model's residual the directional term explains.
    /// Compared against [`DelightConfig::min_explained`].
    pub explained: f64,
}

/// **De-lighting v1**: estimate one ambient-plus-directional illumination model
/// from (baked normal, observed luma) and divide it out.
///
/// # What this can and cannot do, before anything else
///
/// It removes **shading**: the smooth brightening of surfaces that face the
/// light and darkening of those that do not. It removes **nothing else**. It
/// does not remove a cast shadow (a shadow is not a function of the normal), it
/// does not remove a specular highlight (that is a function of the *view*,
/// which differs per photograph and has already been averaged away by the
/// blend), it does not remove coloured light (the fit is on luma, so a warm key
/// and a cool fill come out as one grey scale factor), and it cannot recover
/// anything a clipped highlight threw away.
///
/// It also rests on an assumption it cannot check: that albedo and normal are
/// **uncorrelated** over the surface — grey world. A scene whose upward-facing
/// surfaces are genuinely a different colour from its side-facing ones will
/// have that difference read as illumination and flattened. That is why this is
/// off by default and why [`DelightConfig::min_explained`] exists.
///
/// # The solve
///
/// For each candidate direction `L` in [`hemisphere_basis`], least-squares
/// `luma ≈ a + b · max(0, n · L)` over the well-seen texels — a 2x2 normal
/// equation, solved in closed form. The direction with the smallest residual
/// wins, ties by index. The result is applied as a **mean-preserving** divide:
/// each texel is scaled by `mean(s) / s(n)`, so de-lighting changes the
/// *distribution* of brightness and not the overall exposure, which nothing
/// here has any information about.
pub fn delight(
    albedo: &Channel,
    normals: &Channel,
    mask: &[bool],
    cfg: &DelightConfig,
) -> (Channel, DelightReport) {
    let mut report = DelightReport::default();
    if !cfg.enabled {
        return (albedo.clone(), report);
    }
    // The fit set: well-seen texels with a baked normal.
    let mut samples: Vec<(DVec3, f64)> = Vec::new();
    for i in 0..albedo.data.len() {
        if !mask.get(i).copied().unwrap_or(false) || !normals.valid[i] {
            continue;
        }
        let n = DVec3::new(
            normals.data[i][0] * 2.0 - 1.0,
            normals.data[i][1] * 2.0 - 1.0,
            normals.data[i][2] * 2.0 - 1.0,
        );
        if n.length_squared() < 1.0e-6 {
            continue;
        }
        let c = albedo.data[i];
        let lin = |v: f64| {
            if cfg.input_is_srgb {
                srgb_to_linear(v)
            } else {
                v
            }
        };
        // Rec.709 luma over linear components — the same weighting the `image`
        // crate applies, kept here rather than routed through it because that
        // door takes bytes and this is mid-pipeline `f64`.
        let luma = 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2]);
        samples.push((n.normalize(), luma));
    }
    report.texels = samples.len();
    if samples.len() < 64 {
        return (albedo.clone(), report);
    }
    // The constant model's residual, which is what a directional term has to
    // beat to be believed.
    let mean: f64 = samples.iter().map(|(_, l)| *l).sum::<f64>() / samples.len() as f64;
    let constant_sse: f64 = samples.iter().map(|(_, l)| (l - mean) * (l - mean)).sum();
    if !(constant_sse > 0.0) {
        return (albedo.clone(), report);
    }

    let mut best: Option<(f64, DVec3, f64, f64)> = None;
    for dir in hemisphere_basis(cfg.directions.max(1)) {
        // Normal equations for `l ~ a + b * x`, x = max(0, n·L).
        let (mut sx, mut sxx, mut sl, mut sxl) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (n, l) in &samples {
            let x = n.dot(dir).max(0.0);
            sx += x;
            sxx += x * x;
            sl += *l;
            sxl += x * l;
        }
        let m = samples.len() as f64;
        let det = m * sxx - sx * sx;
        if det.abs() < 1.0e-12 {
            continue;
        }
        let a = (sxx * sl - sx * sxl) / det;
        let b = (m * sxl - sx * sl) / det;
        if !(b > 0.0) || !a.is_finite() || !b.is_finite() {
            // A negative directional term is a light behind the surface, which
            // this model does not have a meaning for.
            continue;
        }
        let sse: f64 = samples
            .iter()
            .map(|(n, l)| {
                let r = l - (a + b * n.dot(dir).max(0.0));
                r * r
            })
            .sum();
        if best.map(|(bs, _, _, _)| sse < bs).unwrap_or(true) {
            best = Some((sse, dir, a, b));
        }
    }
    let Some((sse, dir, a, b)) = best else {
        return (albedo.clone(), report);
    };
    report.ambient = a;
    report.directional = b;
    report.direction = dir.to_array();
    report.explained = (constant_sse - sse) / constant_sse;
    if report.explained < cfg.min_explained {
        return (albedo.clone(), report);
    }
    // Mean-preserving divide.
    let mean_s: f64 = samples
        .iter()
        .map(|(n, _)| a + b * n.dot(dir).max(0.0))
        .sum::<f64>()
        / samples.len() as f64;
    if !(mean_s > 0.0) {
        return (albedo.clone(), report);
    }
    let mut out = albedo.clone();
    for i in 0..out.data.len() {
        if !normals.valid[i] {
            continue;
        }
        let n = DVec3::new(
            normals.data[i][0] * 2.0 - 1.0,
            normals.data[i][1] * 2.0 - 1.0,
            normals.data[i][2] * 2.0 - 1.0,
        );
        if n.length_squared() < 1.0e-6 {
            continue;
        }
        let s = a + b * n.normalize().dot(dir).max(0.0);
        if !(s > 1.0e-6) {
            continue;
        }
        let scale = mean_s / s;
        for c in 0..3 {
            let v = albedo.data[i][c];
            out.data[i][c] = if cfg.input_is_srgb {
                linear_to_srgb((srgb_to_linear(v) * scale).clamp(0.0, 1.0))
            } else {
                (v * scale).clamp(0.0, 1.0)
            };
        }
    }
    report.applied = true;
    (out, report)
}

/// The sRGB electro-optical transfer function, IEC 61966-2-1.
///
/// `powf` is a `std` transcendental, and it is here deliberately: de-lighting
/// is the one stage that has to work in linear light, and it runs on the
/// author's machine as part of a one-shot import — the class the P25.3 ruling
/// exempts (see [`inf_photo`]'s crate docs). Nothing re-derives its output.
#[inline]
pub fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// The inverse of [`srgb_to_linear`].
#[inline]
pub fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_photo::camera::{Intrinsics, Pose};

    /// A floor at `y = 0` facing up, with a wall standing on it at `x = wall_x`
    /// and one unit tall — the smallest surface that has both an open face and a
    /// concave crease, which is what an ambient-occlusion order assertion needs.
    ///
    /// The wall is **finite in `y`** rather than an infinite plane so that a
    /// query far above the floor is far from everything. An infinite wall is
    /// close to every point at its own `x`, however high up, which turned the
    /// normal bake's "nothing within reach" arm into a measurement of the test
    /// surface instead of the bake (measured: 56 fallbacks of 64, not 64).
    struct PlaneAndWall {
        /// The wall stands at `x = wall_x`, from `y = 0` to `y = 1`.
        wall_x: f64,
    }

    const WALL_TOP: f64 = 1.0;

    impl DenseSurface for PlaneAndWall {
        fn closest(&self, p: DVec3) -> Option<SurfacePoint> {
            // Distance to the floor, and to the wall's face.
            let floor = SurfacePoint {
                point: DVec3::new(p.x, 0.0, p.z),
                normal: DVec3::Y,
                distance: p.y.abs(),
            };
            let on_wall = DVec3::new(self.wall_x, p.y.clamp(0.0, WALL_TOP), p.z);
            let wall = SurfacePoint {
                point: on_wall,
                normal: DVec3::NEG_X,
                distance: (p - on_wall).length(),
            };
            Some(if floor.distance <= wall.distance {
                floor
            } else {
                wall
            })
        }

        fn occluded(&self, origin: DVec3, direction: DVec3, max_distance: f64) -> bool {
            // The wall, `x = wall_x` for `0 <= y <= WALL_TOP`.
            if direction.x.abs() > 1.0e-9 {
                let t = (self.wall_x - origin.x) / direction.x;
                let y = origin.y + direction.y * t;
                if t > 1.0e-6 && t < max_distance && (0.0..=WALL_TOP).contains(&y) {
                    return true;
                }
            }
            // The floor plane `y = 0`.
            if direction.y.abs() > 1.0e-9 {
                let t = -origin.y / direction.y;
                if t > 1.0e-6 && t < max_distance {
                    return true;
                }
            }
            false
        }
    }

    fn pool() -> JobPool {
        JobPool::new(2)
    }

    /// One quad in the `xz` plane, mapped over the whole atlas.
    fn quad() -> (Vec<DVec3>, Vec<DVec3>, Vec<[f32; 2]>, Vec<u32>) {
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(0.0, 0.0, 1.0),
        ];
        let normals = vec![DVec3::Y; 4];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = vec![0, 1, 2, 0, 2, 3];
        (positions, normals, uvs, indices)
    }

    #[test]
    fn the_rasterizer_covers_the_interior_and_interpolates_exactly() {
        let (p, n, uv, i) = quad();
        let atlas = rasterize_atlas(&p, &n, &uv, &i, 16);
        // Every texel centre of a full-atlas quad is inside it.
        assert_eq!(
            atlas.covered_count(),
            256,
            "the quad did not fill the atlas"
        );
        assert_eq!(atlas.overlaps, 0, "two triangles claimed one texel");
        assert_eq!(atlas.unsampled_triangles, 0);
        // Texel (x, y) centre is at uv ((x+0.5)/16, (y+0.5)/16), which on this
        // quad is world (u, 0, v). Exact, so assert it exactly enough.
        for y in 0..16u32 {
            for x in 0..16u32 {
                let got = atlas.position[y as usize * 16 + x as usize];
                let want = DVec3::new((x as f64 + 0.5) / 16.0, 0.0, (y as f64 + 0.5) / 16.0);
                assert!(
                    (got - want).length() < 1.0e-12,
                    "texel ({x},{y}) interpolated to {got}, not {want}"
                );
                assert!((atlas.normal[y as usize * 16 + x as usize] - DVec3::Y).length() < 1e-12);
            }
        }
    }

    #[test]
    fn a_degenerate_uv_triangle_is_counted_rather_than_dividing_by_zero() {
        let (p, n, mut uv, i) = quad();
        // Collapse the second triangle's UVs onto a line.
        uv[2] = [0.0, 0.0];
        uv[3] = [0.0, 0.0];
        let atlas = rasterize_atlas(&p, &n, &uv, &i, 16);
        assert!(
            atlas.unsampled_triangles >= 1,
            "a collapsed UV triangle was not counted"
        );
        assert!(atlas.position.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn an_overlapping_chart_is_counted_and_the_first_writer_wins() {
        // Two triangles mapped onto the SAME uv square, at different positions.
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(0.0, 5.0, 0.0),
            DVec3::new(1.0, 5.0, 0.0),
            DVec3::new(1.0, 5.0, 1.0),
        ];
        let normals = vec![DVec3::Y; 6];
        let uvs = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        let atlas = rasterize_atlas(&positions, &normals, &uvs, &[0, 1, 2, 3, 4, 5], 8);
        assert!(atlas.overlaps > 0, "the double cover was not noticed");
        // First writer (the y = 0 triangle) kept every texel it wrote.
        for (i, covered) in atlas.covered.iter().enumerate() {
            if *covered {
                assert!(
                    atlas.position[i].y.abs() < 1e-9,
                    "the second triangle overwrote texel {i}"
                );
            }
        }
    }

    #[test]
    fn the_hemisphere_is_unit_deterministic_and_not_one_direction_repeated() {
        let a = hemisphere_basis(64);
        let b = hemisphere_basis(64);
        assert_eq!(a.len(), 64);
        assert_eq!(
            a, b,
            "the direction table is not a pure function of nothing"
        );
        for v in &a {
            assert!((v.length() - 1.0).abs() < 1e-12, "{v} is not unit");
        }
        // It spans the sphere: the mean of a uniform set is near zero, and a
        // clustered set's is not.
        let mean = a.iter().fold(DVec3::ZERO, |acc, v| acc + *v) / a.len() as f64;
        assert!(mean.length() < 0.30, "the directions cluster: mean {mean}");
        // And no direction is repeated.
        for i in 0..a.len() {
            for j in (i + 1)..a.len() {
                assert!(
                    (a[i] - a[j]).length() > 1e-9,
                    "directions {i} and {j} match"
                );
            }
        }
    }

    #[test]
    fn ambient_occlusion_is_darker_in_a_crease_than_on_an_open_face() {
        // The ORDER assertion, not a magic constant: a texel in the corner where
        // the wall meets the floor must be more occluded than one far from it.
        let surface = PlaneAndWall { wall_x: 1.0 };
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 1.0),
            DVec3::new(0.0, 0.0, 1.0),
        ];
        let normals = vec![DVec3::Y; 4];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let atlas = rasterize_atlas(&positions, &normals, &uvs, &[0, 1, 2, 0, 2, 3], 16);
        let cfg = BakeConfig {
            size: 16,
            ao_rays: 64,
            ao_distance_voxels: 4.0,
            ao_bias_voxels: 0.02,
            ..BakeConfig::default()
        };
        let ao = bake_ao(&atlas, &surface, 0.1, &cfg, &pool());
        // Texel column 15 is hard against the wall; column 0 is a metre away.
        let near = ao.data[8 * 16 + 15][0];
        let far = ao.data[8 * 16][0];
        assert!(
            near < far,
            "the crease ({near}) is not darker than the open face ({far})"
        );
        // Anti-vacuity: both must be real occlusion values, not 0 and 1.
        assert!(
            far > 0.05 && near < 0.99,
            "ao is saturated: near {near}, far {far}"
        );
    }

    #[test]
    fn the_ao_ray_length_is_read_rather_than_decorative() {
        // `ao_distance_voxels` is documented as the knob that chooses what the
        // bake measures — "short rays measure creases; long ones measure the
        // room" — and until this arm existed **nothing anywhere asserted that it
        // is read at all**. MEASURED: replacing the ray length with infinity
        // left the whole battery green, the P25.3 gate included, because a
        // uniformly darker ambient-occlusion map is still ordered and still
        // unsaturated. An order assertion cannot see a scale.
        //
        // So this is the scale assertion: one texel, far from the only occluder
        // there is, with a ray too short to reach it and a ray long enough.
        let surface = PlaneAndWall { wall_x: 1.0 };
        let (p, n, uv, i) = quad();
        let atlas = rasterize_atlas(&p, &n, &uv, &i, 16);
        let bake = |voxels: f64| {
            let cfg = BakeConfig {
                size: 16,
                ao_rays: 64,
                ao_distance_voxels: voxels,
                ao_bias_voxels: 0.02,
                ..BakeConfig::default()
            };
            bake_ao(&atlas, &surface, 0.1, &cfg, &pool())
        };
        // Column 0 sits 0.97 units from the wall, and the only other surface is
        // the floor the texel is *on* — which a cosine hemisphere above it never
        // fires below. So 0.4 units of ray reach nothing at all, and 4.0 units
        // reach the wall.
        let far = 8 * 16;
        let short = bake(4.0).data[far][0];
        let long = bake(40.0).data[far][0];
        assert_eq!(
            short, 1.0,
            "a 0.4-unit ray found geometry 0.97 units away ({short}) — the length is not bounding \
             the occlusion query"
        );
        assert!(
            long < 0.9,
            "a 4.0-unit ray did not reach a wall 0.97 units away: {long}"
        );
    }

    #[test]
    fn the_normal_bake_transfers_the_dense_normal_and_counts_what_it_could_not() {
        let surface = PlaneAndWall { wall_x: 1.0 };
        let (p, n, uv, i) = quad();
        // Lift the quad well off the floor so the dense surface is out of reach.
        let lifted: Vec<DVec3> = p.iter().map(|q| *q + DVec3::Y * 5.0).collect();
        let atlas = rasterize_atlas(&lifted, &n, &uv, &i, 8);
        let cfg = BakeConfig {
            size: 8,
            normal_search_voxels: 1.0,
            ..BakeConfig::default()
        };
        let (_, report) = bake_normals(&atlas, &surface, 0.1, &cfg, &pool());
        assert_eq!(report.texels, 64);
        assert_eq!(
            report.fallbacks, 64,
            "a surface five metres away was treated as a transfer"
        );
        // On the floor itself every texel transfers, and the encoding is the
        // documented one.
        let atlas = rasterize_atlas(&p, &n, &uv, &i, 8);
        let (channel, report) = bake_normals(&atlas, &surface, 0.1, &cfg, &pool());
        assert_eq!(report.fallbacks, 0, "a texel on the floor found nothing");
        for c in &channel.data {
            // +Y encodes as (0.5, 1.0, 0.5).
            assert!((c[0] - 0.5).abs() < 1e-9 && (c[1] - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn dilation_fills_every_hole_including_one_that_spans_the_atlas() {
        let mut ch = Channel::new(8, [0.0; 3]);
        // One valid texel in a corner; a 3x3 pass would need seven iterations.
        ch.data[0] = [0.25, 0.5, 0.75];
        ch.valid[0] = true;
        let filled = dilate(&ch);
        assert_eq!(filled.valid_count(), 64, "dilation left holes");
        for c in &filled.data {
            assert!(
                (c[0] - 0.25).abs() < 1e-9 && (c[2] - 0.75).abs() < 1e-9,
                "a filled texel is {c:?}, not the only colour there was"
            );
        }
        // An entirely empty channel comes back entirely empty rather than
        // inventing a colour out of nothing.
        let empty = dilate(&Channel::new(4, [0.0; 3]));
        assert_eq!(empty.valid_count(), 0);
    }

    #[test]
    fn dilation_does_not_move_a_texel_that_already_had_a_value() {
        let mut ch = Channel::new(8, [0.0; 3]);
        for i in 0..64 {
            ch.data[i] = [i as f64 / 64.0, 0.0, 0.0];
            ch.valid[i] = i % 2 == 0;
        }
        let filled = dilate(&ch);
        for i in (0..64).step_by(2) {
            assert!(
                (filled.data[i][0] - i as f64 / 64.0).abs() < 1e-12,
                "texel {i} was rewritten by the push"
            );
        }
    }

    /// One camera two units above the quad, looking straight down, with an
    /// **analytic** depth map of the `y = 0` plane and a flat colour.
    fn overhead_view(
        size: u32,
        colour: [u8; 3],
    ) -> (RegisteredCamera, RgbImage, DepthMap, SurfaceHints) {
        let intrinsics = Intrinsics::centred(size, size, size as f64);
        let pose = Pose::look_at(
            DVec3::new(0.5, 2.0, 0.5),
            DVec3::new(0.5, 0.0, 0.5),
            // Not exactly anti-parallel to the view, so `look_at` has a frame.
            DVec3::new(0.03, 0.0, 1.0),
        );
        let camera = RegisteredCamera {
            view: 0,
            pose,
            intrinsics,
            observations: 100,
        };
        let mut image = RgbImage::new(size, size).unwrap();
        let mut depth = DepthMap::empty(0, size, size);
        let mut weights = vec![0.0f32; (size * size) as usize];
        let inverse = pose.rotation.inverse();
        let eye = pose.center();
        for y in 0..size {
            for x in 0..size {
                image.set(x, y, colour);
                let n = intrinsics.to_normalized(DVec2::new(x as f64, y as f64));
                let ray = inverse * DVec3::new(n.x, n.y, 1.0).normalize();
                if ray.y.abs() < 1e-9 {
                    continue;
                }
                let t = -eye.y / ray.y;
                if t <= 0.0 {
                    continue;
                }
                let world = eye + ray * t;
                let z = pose.transform(world).z;
                let i = (y * size + x) as usize;
                depth.depth[i] = z as f32;
                depth.confidence[i] = 1.0;
                weights[i] = 1.0;
            }
        }
        let hints = SurfaceHints {
            normals: vec![[0.0, 0.0, -1.0]; (size * size) as usize],
            weights,
        };
        (camera, image, depth, hints)
    }

    #[test]
    fn the_albedo_bake_reads_the_colour_the_camera_saw() {
        let (p, n, uv, i) = quad();
        let atlas = rasterize_atlas(&p, &n, &uv, &i, 16);
        let (camera, image, depth, hints) = overhead_view(64, [200, 100, 50]);
        let views = vec![AlbedoView {
            camera: &camera,
            image: &image,
            depth: &depth,
            hints: &hints,
        }];
        let (albedo, weight, report) = bake_albedo(&atlas, &views, &BakeConfig::default(), &pool());
        assert_eq!(report.unseen, 0, "{} texels saw no camera", report.unseen);
        assert_eq!(report.seen, 256);
        assert_eq!(report.occluded, 0);
        for (i, c) in albedo.data.iter().enumerate() {
            assert!(
                (c[0] - 200.0 / 255.0).abs() < 1e-9
                    && (c[1] - 100.0 / 255.0).abs() < 1e-9
                    && (c[2] - 50.0 / 255.0).abs() < 1e-9,
                "texel {i} baked {c:?}"
            );
        }
        // The weight channel carries a real number, not a flag: a texel seen
        // face-on weighs more than one seen at an angle, and the atlas's
        // corners are further off-axis than its middle.
        let middle = weight.data[8 * 16 + 8][0];
        let corner = weight.data[0][0];
        assert!(
            middle > corner && corner > 0.0,
            "the incidence weight is flat: middle {middle}, corner {corner}"
        );
    }

    #[test]
    fn a_texel_the_camera_cannot_see_is_left_unseen_rather_than_painted() {
        // THE reason P25.2 hands depth maps back. Put something closer in the
        // depth map over half the frame and the texels behind it must come back
        // with no colour at all — not with the colour of whatever was in front.
        let (p, n, uv, i) = quad();
        let atlas = rasterize_atlas(&p, &n, &uv, &i, 16);
        let (camera, image, mut depth, hints) = overhead_view(64, [200, 100, 50]);
        for y in 0..64u32 {
            for x in 0..32u32 {
                let i = (y * 64 + x) as usize;
                if depth.depth[i] > 0.0 {
                    depth.depth[i] *= 0.4;
                }
            }
        }
        let views = vec![AlbedoView {
            camera: &camera,
            image: &image,
            depth: &depth,
            hints: &hints,
        }];
        let (_, _, report) = bake_albedo(&atlas, &views, &BakeConfig::default(), &pool());
        assert!(
            report.occluded > 100,
            "only {} samples were rejected as occluded",
            report.occluded
        );
        assert!(
            report.unseen > 100 && report.seen > 100,
            "the split is {} seen / {} unseen — the mask is not half the atlas",
            report.seen,
            report.unseen
        );
        assert_eq!(report.seen + report.unseen, 256);
    }

    #[test]
    fn a_view_with_no_depth_offers_no_evidence_and_is_counted_saying_so() {
        let (p, n, uv, i) = quad();
        let atlas = rasterize_atlas(&p, &n, &uv, &i, 8);
        let (camera, image, mut depth, hints) = overhead_view(64, [10, 20, 30]);
        for d in depth.depth.iter_mut() {
            *d = 0.0;
        }
        let views = vec![AlbedoView {
            camera: &camera,
            image: &image,
            depth: &depth,
            hints: &hints,
        }];
        let (albedo, _, report) = bake_albedo(&atlas, &views, &BakeConfig::default(), &pool());
        assert_eq!(report.seen, 0, "a depthless view painted the atlas anyway");
        assert_eq!(report.unseen, 64);
        assert_eq!(report.no_evidence, 64);
        assert_eq!(albedo.valid_count(), 0);
    }

    #[test]
    fn the_srgb_transfer_round_trips() {
        for i in 0..=255 {
            let v = i as f64 / 255.0;
            let back = linear_to_srgb(srgb_to_linear(v));
            assert!((back - v).abs() < 1e-9, "{v} round-tripped to {back}");
        }
    }

    #[test]
    fn delighting_removes_a_shading_gradient_it_was_given() {
        // A synthetic atlas: one constant albedo, normals sweeping through a
        // quarter turn, observed = albedo * (ambient + directional * n·L).
        let size = 32u32;
        let n_texels = size as usize * size as usize;
        let light = DVec3::new(0.2, 0.9, 0.3).normalize();
        let (ambient, directional) = (0.30, 0.70);
        let albedo_truth = [0.55f64, 0.40, 0.30];
        let mut albedo = Channel::new(size, [0.0; 3]);
        let mut normals = Channel::new(size, [0.5, 0.5, 1.0]);
        let mask = vec![true; n_texels];
        for i in 0..n_texels {
            // A deterministic normal sweep with no trigonometry: lerp between
            // two axes and normalize.
            let t = i as f64 / n_texels as f64;
            let n = (DVec3::Y * (1.0 - t) + DVec3::X * t + DVec3::Z * 0.15).normalize();
            normals.data[i] = [n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, n.z * 0.5 + 0.5];
            normals.valid[i] = true;
            let e = ambient + directional * n.dot(light).max(0.0);
            for (c, truth) in albedo_truth.iter().enumerate() {
                albedo.data[i][c] = truth * e;
            }
            albedo.valid[i] = true;
        }
        let cfg = DelightConfig {
            enabled: true,
            input_is_srgb: false,
            ..DelightConfig::default()
        };
        let (out, report) = delight(&albedo, &normals, &mask, &cfg);
        assert!(report.applied, "de-lighting refused a scene it could fit");
        assert!(
            report.explained > 0.9,
            "only {} of the variance was explained",
            report.explained
        );
        // The output must be FLATTER than the input: measure the spread of the
        // green channel before and after.
        let spread = |c: &Channel| {
            let vals: Vec<f64> = c.data.iter().map(|v| v[1]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            (vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64).sqrt()
        };
        let before = spread(&albedo);
        let after = spread(&out);
        assert!(
            after < before * 0.2,
            "de-lighting flattened the gradient only from {before} to {after}"
        );
    }

    #[test]
    fn delighting_refuses_a_flat_scene_rather_than_inventing_a_light() {
        let size = 32u32;
        let n_texels = size as usize * size as usize;
        let mut albedo = Channel::new(size, [0.0; 3]);
        let mut normals = Channel::new(size, [0.5, 0.5, 1.0]);
        let mask = vec![true; n_texels];
        for i in 0..n_texels {
            let t = i as f64 / n_texels as f64;
            let n = (DVec3::Y * (1.0 - t) + DVec3::X * t + DVec3::Z * 0.15).normalize();
            normals.data[i] = [n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, n.z * 0.5 + 0.5];
            normals.valid[i] = true;
            // Albedo that varies with the texel index and NOT with the normal's
            // relation to any single direction... except that it does vary with
            // t, which every direction sees. So the guard has to be the
            // explained fraction, not the presence of a fit.
            albedo.data[i] = [0.5, 0.5, 0.5];
            albedo.valid[i] = true;
        }
        let cfg = DelightConfig {
            enabled: true,
            input_is_srgb: false,
            ..DelightConfig::default()
        };
        let (out, report) = delight(&albedo, &normals, &mask, &cfg);
        assert!(!report.applied, "a flat scene was 'de-lit'");
        assert_eq!(out.data, albedo.data, "a refusal still changed the texture");
    }

    #[test]
    fn delighting_disabled_is_the_identity() {
        let ch = Channel::new(8, [0.3, 0.4, 0.5]);
        let (out, report) = delight(&ch, &ch, &[true; 64], &DelightConfig::default());
        assert!(!report.applied);
        assert_eq!(out, ch);
    }
}
