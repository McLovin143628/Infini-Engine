//! **The finish pipeline** (P25.3): a dense reconstruction in, a standard
//! textured asset out.
//!
//! # Where the four steps live, and why here
//!
//! ```text
//!   DenseReconstruction (inf-photo-gpu, baseline units)
//!         │
//!         ├─ 1. retopo ......... inf_mesh::{simplify, optimize}  ← meshopt
//!         ├─ 2. unwrap ......... inf_dcc::unwrap                 ← P23.5 LSCM
//!         ├─ 3. bake ........... inf_photo_gpu::finish           ← Ring 0
//!         └─ 4. write .......... AssetProject                    ← Ring 1
//! ```
//!
//! Three of those four are doors that already exist and are reused rather than
//! rebuilt. The orchestrator is **Ring 1** because two of them are only
//! reachable from here: `AssetProject` is the editor's, and `inf-dcc` is a crate
//! the shipped player must never link (it is a **dev**-dependency of
//! `inf-player`, deliberately). What could be Ring 0 *is*: the whole
//! texel-space half — atlas rasterization and the three bakes — lives in
//! `inf_photo_gpu::finish` and reaches the modelling kernel's BVH through the
//! two-method [`DenseSurface`] trait, implemented here by [`BvhSurface`].
//!
//! # The bakes are on the CPU, and that is a ruling rather than an omission
//!
//! The obvious reading of "on the existing headless bake machinery" is
//! [`crate::thumbnail::scene_render::bake_texture`], the P7.3 compute path. That
//! door takes **a WGSL string** and runs it over a storage texture; it is the
//! material-graph baker, and what it bakes is a function of a texel's own UV.
//! A dense-to-retopo transfer is not that: every texel needs a closest-point
//! query and thirty-two ray casts against a two-hundred-thousand-triangle mesh,
//! which means the BVH on the GPU, which is a Phase-26-sized piece of work.
//!
//! It would also make the P25.3 gate adapter-dependent, and P25.2's audit
//! settled what that costs: eight numbers measured on one adapter had been
//! gated on every one. **LAW, met again: do not put a measurement on the GPU
//! unless the GPU is what is being measured.** So the bakes run on
//! [`inf_core::job::JobPool`], which is the workspace's deterministic
//! in-order parallel map, and the gate runs everywhere with no skip.
//!
//! What *is* reused from the bake machinery is the part that was actually
//! shared: `inf_material::texture_from_rgba8` — mip chain, BC1/BC3 selection,
//! the sRGB-versus-linear flag — which is the same door the P7.3 baker writes
//! its `.inf_tex` through.
//!
//! # Units and the scale step
//!
//! Everything from the reconstruction is in **baseline units** (structure from
//! motion is scale-ambiguous; the gauge is the first camera pair). The bakes
//! run in those units, because the occlusion test compares a texel's depth
//! against a depth map that is in them. [`FinishConfig::metres_per_unit`] is
//! applied **once, at the end**, to the mesh's positions and bounds and nothing
//! else — a uniform scale leaves normals, tangents and UVs alone. Its default
//! is `1.0`, documented as "the reconstruction's own unit", and P25.4's wizard
//! is what will supply a real one from a known distance, a marker or EXIF.
//!
//! # Determinism, and the one thing that is not portable
//!
//! Every stage is a pure function of its input, and the parallel ones are
//! in-order maps. The exception is named rather than hidden: **meshopt is not
//! cross-platform** (P18's law). `inf_mesh::optimize`'s module docs carry the
//! full ruling; the short version is that this is a *one-shot import* — it runs
//! once, on the author's machine, over photographs the author supplies, and
//! writes content-addressed assets that nothing re-derives. That is the same
//! standing glTF import has had since Phase 4. A gate over this pipeline must
//! therefore compare **within one run or within one machine**, never a
//! committed number against a freshly-solved one, and
//! `tests/photogrammetry_gate.rs` is written to that rule.

use std::path::Path;

use glam::DVec3;
use inf_asset::AssetId;
use inf_dcc::{Bvh, Mesh, Op, Tri};
use inf_material::{
    texture_from_rgba8, MaterialAsset, TextureAsset, TextureCompression, TextureImportSettings,
};
use inf_mesh::asset::{MeshAsset, MeshVertex, SubMesh};
use inf_photo::rgb::RgbImage;
use inf_photo::sfm::Reconstruction;
use inf_photo_gpu::finish::{
    bake_albedo, bake_ao, bake_normals, delight, dilate, geometric_normals, rasterize_atlas,
    AlbedoBakeReport, AlbedoView, BakeConfig, Channel, DelightConfig, DelightReport, DenseSurface,
    NormalBakeReport, SurfacePoint,
};
use inf_photo_gpu::tsdf::DenseMesh;
use inf_photo_gpu::DenseReconstruction;

use inf_core::job::JobPool;

/// The material slot name the scan's one submesh carries.
pub const SCAN_SLOT: &str = "Scan";

/// The virtualized-geometry threshold a finished mesh has to clear to be
/// virtualized at cook — `VgeomCookOptions::min_triangles`.
///
/// Mirrored here as a **number to advise against**, not as a second source of
/// truth: `inf-packager` is Ring 2 and this is Ring 1, so the cook's own
/// constant cannot be read from here. A finished mesh below it draws as a
/// **placeholder cube** in a shipped build (P23's finding, met again), which is
/// exactly the kind of silent hazard the cook-advisory doctrine exists for.
pub const VGEOM_MIN_TRIANGLES: usize = 2048;

/// How many rounds of neighbour averaging the dense mesh's geometric normals get
/// before anything reads them.
///
/// A constant rather than a config knob because both readers — [`retopo`], which
/// writes them into the mesh, and [`BvhSurface`], which transfers them into the
/// normal map — **must use the same number**. Two different smoothings would
/// bake a normal map that disagrees with the mesh it is applied to, and the
/// disagreement would look exactly like a lighting bug. See
/// [`geometric_normals`] for the measurement behind the value.
pub const NORMAL_SMOOTHING_ROUNDS: usize = 4;

/// How the finish pipeline is tuned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinishConfig {
    /// Triangle budget for the retopologized mesh.
    ///
    /// The default clears [`VGEOM_MIN_TRIANGLES`] with room to spare, because a
    /// scan that decimates *below* it ships as a placeholder cube.
    pub target_triangles: usize,
    /// How many low-pass rounds smooth the normal field the seams are cut from.
    /// See [`seam_edges`].
    pub seam_smoothing_passes: usize,
    /// Charts smaller than this are folded into a neighbour, so an atlas is not
    /// mostly gutter. See [`merge_small_charts`](seam_edges).
    pub min_chart_faces: usize,
    /// Whether to drop geometry no camera photographed — the TSDF slab's far
    /// sheet. See [`trim_unseen`].
    pub trim_unseen: bool,
    /// Atlas size and the three bakes.
    pub bake: BakeConfig,
    /// De-lighting. Off by default.
    pub delight: DelightConfig,
    /// Metres per reconstruction unit — **the scale step**. `1.0` means "leave
    /// it in baseline units", which is honest rather than metric and is what a
    /// caller gets until P25.4's wizard measures something.
    ///
    /// Must be finite and strictly positive;
    /// [`finish_reconstruction`] refuses anything else by name
    /// ([`FinishError::BadScale`]) rather than scaling by it. A wizard reading a
    /// distance out of a text field is exactly where a `0` comes from.
    pub metres_per_unit: f64,
    /// Roughness written into the material and into the ORM texture's green
    /// channel. A photographed surface has no measured roughness; this is the
    /// engine's own default made explicit rather than inferred from pixels.
    pub roughness: f32,
    /// Metallic, likewise. Photogrammetry of a metal object is a known-hard
    /// case (a view-dependent surface breaks the constant-albedo premise), and
    /// guessing `1.0` from a bright highlight would be worse than saying zero.
    pub metallic: f32,
}

impl Default for FinishConfig {
    fn default() -> Self {
        Self {
            target_triangles: 20_000,
            seam_smoothing_passes: 3,
            min_chart_faces: 64,
            trim_unseen: true,
            bake: BakeConfig::default(),
            delight: DelightConfig::default(),
            metres_per_unit: 1.0,
            roughness: 0.8,
            metallic: 0.0,
        }
    }
}

/// Why a finish **refused**.
///
/// `PartialEq` without `Eq`, the same standing `inf_dcc::HeatError` has: one
/// variant carries the scale the caller asked for, and a scale is an `f64`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FinishError {
    /// The dense stage produced nothing to retopologize.
    #[error("the dense reconstruction has {vertices} vertices and {triangles} triangles — there is nothing to finish")]
    EmptyDense {
        /// Dense vertices.
        vertices: usize,
        /// Dense triangles.
        triangles: usize,
    },
    /// Retopology left no geometry at all.
    #[error(
        "retopology reduced {before} triangles to {after}; the manifold filter dropped {dropped} \
         of them, which usually means the fused surface is mostly self-intersecting"
    )]
    RetopoEmpty {
        /// Triangles in the dense mesh.
        before: usize,
        /// Triangles after decimation and filtering.
        after: usize,
        /// Triangles the manifold filter dropped.
        dropped: usize,
    },
    /// The reconstruction registered no cameras at all.
    ///
    /// Reachable because the sparse reconstruction and the dense one arrive as
    /// two arguments: a caller can hand over a reconstruction that registered
    /// nothing. Without this the pipeline runs, trims every triangle for want of
    /// a camera, keeps the untrimmed mesh, and writes an asset whose albedo is
    /// entirely dilation — a plausible-looking texture nobody photographed.
    #[error("the reconstruction registered no cameras — there is nothing to bake colour from")]
    NoCameras,
    /// The dense reconstruction was not built from the sparse one it was handed
    /// with.
    ///
    /// [`DenseReconstruction::depth_maps`] and `surface` are **parallel to the
    /// registered cameras, in ascending view order** — they are indexed by slot,
    /// not by view id. Two arguments that disagree about how many cameras there
    /// are is a caller pairing a reconstruction with somebody else's dense
    /// stage, and it used to be an index panic rather than a refusal.
    #[error(
        "the dense reconstruction has {depth_maps} depth maps and {surface} surface hints for a \
         reconstruction with {cameras} registered cameras — they were not solved together"
    )]
    ViewsMismatch {
        /// Registered cameras in the sparse reconstruction.
        cameras: usize,
        /// Depth maps in the dense reconstruction.
        depth_maps: usize,
        /// Surface-hint images in the dense reconstruction.
        surface: usize,
    },
    /// [`FinishConfig::metres_per_unit`] is not a scale.
    ///
    /// Zero collapses the mesh onto its own origin; a negative multiplier
    /// mirrors it, which reverses every triangle's winding while leaving the
    /// vertex normals and the baked normal map facing the way they were.
    #[error(
        "metres_per_unit is {metres_per_unit}, which is not a scale — zero collapses the mesh to a \
         point and a negative value mirrors it away from its own normals"
    )]
    BadScale {
        /// What the caller asked for.
        metres_per_unit: f64,
    },
    /// A photograph is missing for a registered camera.
    #[error("registered camera {view} has no photograph in the {given} supplied")]
    MissingPhoto {
        /// The camera's view index.
        view: u32,
        /// How many photographs the caller passed.
        given: usize,
    },
    /// A photograph does not match the intrinsics its camera declares.
    #[error("photograph {view} is {img_w}x{img_h} but its camera declares {int_w}x{int_h}")]
    PhotoMismatch {
        /// The offending view.
        view: u32,
        /// Photograph width.
        img_w: u32,
        /// Photograph height.
        img_h: u32,
        /// Declared width.
        int_w: u32,
        /// Declared height.
        int_h: u32,
    },
    /// The modelling kernel refused the retopologized mesh.
    ///
    /// The interesting case is `NonManifoldEdge`, which [`manifold_filter`]
    /// exists to prevent and which reaching here means it missed something.
    #[error("the retopologized mesh could not enter the modelling kernel: {message}")]
    Kernel {
        /// The kernel's own complaint.
        message: String,
    },
    /// The unwrapper refused.
    #[error("UV unwrap refused: {message}")]
    Unwrap {
        /// The unwrapper's own complaint.
        message: String,
    },
    /// A texture could not be encoded.
    #[error("encoding the {channel} texture failed: {message}")]
    Texture {
        /// Which channel.
        channel: &'static str,
        /// The encoder's complaint.
        message: String,
    },
    /// An asset write failed. The partial build has already been rolled back.
    #[error("writing the {what} asset failed: {message}")]
    Write {
        /// Which asset.
        what: &'static str,
        /// The writer's complaint.
        message: String,
    },
}

/// A non-fatal finding: something P25.4's diagnostics panel should show, that
/// did not stop the finish.
///
/// Ordered and canonical, the same doctrine as
/// [`inf_photo::Advisory`] and [`inf_photo_gpu::DenseAdvisory`]: an advisory
/// appearing or disappearing is a visible change rather than a log line nobody
/// reads.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FinishAdvisory {
    /// The fused surface had edges shared by more than two faces (or by two
    /// faces wound the same way), and those triangles were dropped so the mesh
    /// could enter the half-edge kernel at all.
    NonManifoldDropped {
        /// Triangles dropped.
        dropped: usize,
        /// Triangles examined.
        examined: usize,
    },
    /// Triangles **no camera photographed** were removed before unwrapping —
    /// the TSDF slab's far sheet, and anything the capture never faced.
    ///
    /// Not a fault: it is the honest boundary of what the photographs contain.
    /// It is an advisory because it is also the number that says how *closed*
    /// the result is — a scan trimmed of half its triangles is a shell, not a
    /// solid, and a caller placing it in a scene needs to know which.
    UnphotographedTrimmed {
        /// Triangles dropped.
        dropped: usize,
        /// Triangles the retopology produced.
        examined: usize,
    },
    /// Decimation could not reach the budget. meshopt stops when the topology
    /// stops it; this is that, said out loud.
    DecimationShortOfBudget {
        /// The budget.
        asked: usize,
        /// What came out.
        got: usize,
    },
    /// The finished mesh is below the cook's virtualized-geometry threshold, so
    /// a **shipped** build draws it as a placeholder cube.
    BelowVirtualizationThreshold {
        /// The mesh's triangle count.
        triangles: usize,
        /// [`VGEOM_MIN_TRIANGLES`].
        threshold: usize,
    },
    /// UV charts whose parameterization folded. Their texels are sampled from
    /// the wrong place.
    UvChartsFlipped {
        /// Folded triangles, summed over charts.
        flipped: usize,
        /// Corners the unwrap wrote.
        corners: usize,
    },
    /// Two triangles claimed the same texel — a chart overlapping itself or
    /// another in the atlas.
    AtlasOverlaps {
        /// Texels claimed twice.
        texels: usize,
    },
    /// **Texels no camera could see**, filled by dilation.
    ///
    /// The no-silent-caps doctrine applied to a texture: after dilation these
    /// texels look exactly like the ones that were photographed, so the number
    /// has to travel with the asset or the caller will believe coverage nobody
    /// had.
    UnseenTexels {
        /// Covered texels no camera contributed to.
        unseen: usize,
        /// Covered texels in total.
        covered: usize,
    },
    /// Texels whose normal could not be transferred because the dense surface
    /// was out of search range; they kept the retopologized mesh's own normal.
    NormalTransferFallbacks {
        /// Texels that fell back.
        fallbacks: usize,
        /// Texels baked.
        texels: usize,
    },
    /// De-lighting was asked for and **declined**: the directional term did not
    /// explain enough of the variance to be believed.
    DelightRefused {
        /// What it explained, as a percentage rounded to an integer so the
        /// advisory keeps a total order.
        explained_percent: u32,
        /// What it needed to explain.
        required_percent: u32,
    },
}

impl std::fmt::Display for FinishAdvisory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishAdvisory::NonManifoldDropped { dropped, examined } => write!(
                f,
                "{dropped} of {examined} fused triangles shared an edge with two others and were \
                 dropped so the mesh could be unwrapped; the surface has holes where they were — \
                 this is a property of the fused surface rather than of a setting, and what \
                 reduces it is more overlapping photographs of the thin or doubled regions"
            ),
            FinishAdvisory::UnphotographedTrimmed { dropped, examined } => write!(
                f,
                "{dropped} of {examined} retopologized triangles were seen by no camera and were \
                 removed; the result is the surface the photographs cover, which is a SHELL rather \
                 than a closed solid — photograph the missing side, or turn the unphotographed \
                 trim off to keep the geometry with invented colour on it"
            ),
            FinishAdvisory::DecimationShortOfBudget { asked, got } => write!(
                f,
                "decimation asked for {asked} triangles and stopped at {got} — the topology would \
                 not collapse further; nothing needs doing unless {got} is itself too many, and \
                 then the fix is a coarser fusion rather than a lower budget"
            ),
            FinishAdvisory::BelowVirtualizationThreshold {
                triangles,
                threshold,
            } => write!(
                f,
                "the finished mesh has {triangles} triangles, below the cook's virtualized-geometry \
                 threshold of {threshold}, so a SHIPPED build draws it as a placeholder cube; raise \
                 the triangle budget or lower [vgeom] min_triangles"
            ),
            FinishAdvisory::UvChartsFlipped { flipped, corners } => write!(
                f,
                "{flipped} triangles folded during UV unwrap over {corners} corners; those texels \
                 are sampled from the wrong side of the surface — raise the seam smoothing passes \
                 so a bent chart is cut instead of stretched, or lower the small-chart merge \
                 threshold so it is not folded into a neighbour"
            ),
            FinishAdvisory::AtlasOverlaps { texels } => write!(
                f,
                "{texels} atlas texels were claimed by two triangles; the first won and the second \
                 is textured from somewhere else — raise the atlas size, which widens the gutter \
                 every chart is packed with"
            ),
            FinishAdvisory::UnseenTexels { unseen, covered } => write!(
                f,
                "{unseen} of {covered} textured texels were seen by no camera and were filled from \
                 their neighbours; that part of the albedo is invented, not photographed — \
                 photograph the object from the directions the coverage panel shows empty, \
                 because a larger atlas cannot add colour nobody captured"
            ),
            FinishAdvisory::NormalTransferFallbacks { fallbacks, texels } => write!(
                f,
                "{fallbacks} of {texels} texels found no dense surface within the search radius and \
                 kept the retopologized normal; detail is missing there, not wrong — raise the \
                 normal search radius, or lower the triangle budget so the retopologized surface \
                 stays nearer the fused one"
            ),
            FinishAdvisory::DelightRefused {
                explained_percent,
                required_percent,
            } => write!(
                f,
                "de-lighting found a light explaining {explained_percent}% of the brightness \
                 variation, under the {required_percent}% it needs to be believed, so nothing was \
                 divided out — leave it off unless the capture really was lit by one dominant \
                 light, because dividing out a light that was not found makes the albedo worse"
            ),
        }
    }
}

/// The numbers a finish produced.
#[derive(Clone, Debug, PartialEq)]
pub struct FinishReport {
    /// Triangles in the fused dense mesh.
    pub dense_triangles: usize,
    /// Triangles the manifold filter dropped before decimation.
    pub non_manifold_dropped: usize,
    /// Triangles after decimation, before the kernel round trip.
    pub decimated_triangles: usize,
    /// Triangles removed because no camera photographed them.
    pub unphotographed_dropped: usize,
    /// Triangles in the finished asset.
    pub final_triangles: usize,
    /// Vertices in the finished asset (split at UV seams, so above the
    /// welded count).
    pub final_vertices: usize,
    /// Undirected edges marked as seams.
    pub seams: usize,
    /// UV charts.
    pub charts: usize,
    /// Folded triangles over all charts.
    pub flipped: usize,
    /// The worst chart's conformal residual.
    pub worst_residual: f64,
    /// The worst chart's solver convergence residual.
    pub worst_convergence: f64,
    /// Fraction of the atlas the charts cover, as `covered / size²`.
    pub atlas_coverage: f64,
    /// Extent of the packed charts along `u`, in `[0, 1]`.
    pub atlas_u_span: f64,
    /// Extent of the packed charts along `v`.
    pub atlas_v_span: f64,
    /// The normal bake.
    pub normals: NormalBakeReport,
    /// The albedo bake.
    pub albedo: AlbedoBakeReport,
    /// The de-lighting solve.
    pub delight: DelightReport,
    /// The fusion voxel size, in baseline units — the scale every "in voxels"
    /// knob in [`BakeConfig`] is measured against.
    pub voxel_size: f64,
}

/// A finished asset, in memory, before anything is written.
///
/// Everything that can refuse has already refused by the time this exists —
/// the P24.5 shape, and the reason [`write_finished`] is a separate call.
#[derive(Clone, Debug)]
pub struct FinishedAsset {
    /// The mesh, in **metres**, local-origin centred.
    pub mesh: MeshAsset,
    /// Where the local origin sat, in **baseline units**, so a caller can put
    /// the geometry back where the reconstruction had it.
    pub origin_units: DVec3,
    /// Base colour, sRGB.
    pub albedo: TextureAsset,
    /// Object-space normals, linear.
    pub normal: TextureAsset,
    /// Occlusion / roughness / metallic, linear — the glTF ORM packing, with
    /// ambient occlusion in **red**.
    pub orm: TextureAsset,
    /// The material, with its three texture slots left `None` until
    /// [`write_finished`] mints the GUIDs.
    pub material: MaterialAsset,
    /// Findings, in canonical order.
    pub advisories: Vec<FinishAdvisory>,
    /// The numbers.
    pub report: FinishReport,
}

/// The GUIDs a written finish minted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinishedIds {
    /// The `.inf_mesh`.
    pub mesh: AssetId,
    /// The base-colour `.inf_tex`.
    pub albedo: AssetId,
    /// The object-space normal `.inf_tex`.
    pub normal: AssetId,
    /// The ORM `.inf_tex`.
    pub orm: AssetId,
    /// The `.inf_mat`.
    pub material: AssetId,
}

// ── 0. the dense surface, through the kernel's BVH ──────────────────────────

/// [`DenseSurface`] over `inf_dcc::Bvh` — the P23.4/P24.2 hierarchy, reused
/// rather than rebuilt.
///
/// The BVH answers "which triangle and where"; this adds "and what is its
/// normal", by interpolating the dense mesh's own per-vertex normals at the
/// closest point. Per-vertex rather than per-face because a dense mesh is
/// finer than an atlas texel, so a flat-shaded transfer would bake the fused
/// mesh's own faceting into the normal map — the detail the map exists to
/// carry, replaced by the tessellation that carried it.
pub struct BvhSurface<'a> {
    bvh: Bvh,
    mesh: &'a DenseMesh,
    /// Per-vertex normals read off the **triangulation**, not off the mesher's
    /// SDF gradient — see [`geometric_normals`] for the measurement that made
    /// that necessary.
    normals: Vec<DVec3>,
}

impl<'a> BvhSurface<'a> {
    /// Build the hierarchy over a dense mesh.
    pub fn new(mesh: &'a DenseMesh) -> Self {
        let tris: Vec<Tri> = mesh
            .indices
            .chunks_exact(3)
            .map(|t| Tri {
                a: mesh.positions[t[0] as usize],
                b: mesh.positions[t[1] as usize],
                c: mesh.positions[t[2] as usize],
            })
            .collect();
        Self {
            bvh: Bvh::new(tris),
            normals: geometric_normals(
                &mesh.positions,
                &mesh.indices,
                &mesh.normals,
                NORMAL_SMOOTHING_ROUNDS,
            ),
            mesh,
        }
    }

    /// Whether there is any geometry to query.
    pub fn is_empty(&self) -> bool {
        self.bvh.is_empty()
    }
}

impl DenseSurface for BvhSurface<'_> {
    fn closest(&self, p: DVec3) -> Option<SurfacePoint> {
        let hit = self.bvh.closest_point(p)?;
        // `hit.triangle` is a LEAF index; `source_index` maps it back to the
        // order the triangles went in, which is the order the index buffer is
        // in. Skipping that step reads the normals of a different triangle and
        // is invisible on a smooth mesh.
        let source = self.bvh.source_index(hit.triangle)?;
        let base = source * 3;
        let (i0, i1, i2) = (
            *self.mesh.indices.get(base)? as usize,
            *self.mesh.indices.get(base + 1)? as usize,
            *self.mesh.indices.get(base + 2)? as usize,
        );
        let (a, b, c) = (
            self.mesh.positions[i0],
            self.mesh.positions[i1],
            self.mesh.positions[i2],
        );
        let n = |i: usize| self.normals[i];
        let (w0, w1, w2) = barycentric(hit.point, a, b, c);
        let normal = n(i0) * w0 + n(i1) * w1 + n(i2) * w2;
        let len = normal.length();
        let normal = if len > 1.0e-12 {
            normal / len
        } else {
            // Every interpolated normal cancelled (a fold in the dense mesh).
            // The face normal is still a direction, and a zero vector is not.
            (b - a).cross(c - a).normalize_or_zero()
        };
        Some(SurfacePoint {
            point: hit.point,
            normal,
            distance: hit.distance,
        })
    }

    fn occluded(&self, origin: DVec3, direction: DVec3, max_distance: f64) -> bool {
        match self.bvh.first_hit(origin, direction) {
            // `direction` is unit, so `t` is a distance.
            Some(hit) => hit.t > 0.0 && hit.t < max_distance,
            None => false,
        }
    }
}

/// Barycentric coordinates of `p` in triangle `(a, b, c)`, clamped to the
/// simplex. `p` comes from a closest-point query so it is already on the
/// triangle; the clamp is for the rounding, not for the geometry.
fn barycentric(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> (f64, f64, f64) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    // `is_normal()` rather than `!= 0.0`: a degenerate triangle gives zero, and
    // a triangle whose edges differ by twenty orders of magnitude gives a
    // subnormal that divides into an infinity. Both are "there is no useful
    // barycentric here", and both must take the same exit.
    if !denom.abs().is_normal() {
        return (1.0, 0.0, 0.0);
    }
    let v = ((d11 * d20 - d01 * d21) / denom).clamp(0.0, 1.0);
    let w = ((d00 * d21 - d01 * d20) / denom).clamp(0.0, 1.0);
    let u = (1.0 - v - w).clamp(0.0, 1.0);
    (u, v, w)
}

// ── 1. retopology ───────────────────────────────────────────────────────────

/// Drop every triangle whose insertion would give a **directed** edge two
/// users, in index order.
///
/// # Why this is not optional
///
/// `inf_dcc::from_mesh_asset` **refuses** a mesh with a directed edge used
/// twice — it cannot build a half-edge kernel over one, and repairing it is not
/// something a reader may do to an author's file. But a TSDF's Surface-Nets
/// output is not manifold: P25.2 measured **8 221 of 330 125 edges (2.5%)** on
/// the committed fixture. Without this filter the unwrap step refuses every
/// real scan, and photogrammetry has no UVs.
///
/// Greedy and in index order, so it is a pure function of the input and of
/// nothing else. What it costs is a hole where the offending triangle was, and
/// the count is raised as an advisory rather than absorbed.
pub fn manifold_filter(indices: &[u32]) -> (Vec<u32>, usize) {
    use std::collections::BTreeSet;
    let mut used: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut out = Vec::with_capacity(indices.len());
    let mut dropped = 0usize;
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        if a == b || b == c || c == a {
            dropped += 1;
            continue;
        }
        let edges = [(a, b), (b, c), (c, a)];
        if edges.iter().any(|e| used.contains(e)) {
            dropped += 1;
            continue;
        }
        for e in edges {
            used.insert(e);
        }
        out.extend_from_slice(&[a, b, c]);
    }
    (out, dropped)
}

/// What retopology did.
#[derive(Clone, Debug, PartialEq)]
pub struct RetopoOutput {
    /// The decimated mesh, in **baseline units**, local-origin centred, with no
    /// UVs yet.
    pub asset: MeshAsset,
    /// Where the local origin sat, in baseline units.
    pub origin: DVec3,
    /// Triangles dropped by [`manifold_filter`], summed over both passes.
    pub dropped: usize,
    /// Triangles after decimation.
    pub triangles: usize,
    /// meshopt's own error estimate for the decimation, relative to the mesh's
    /// extent.
    pub error: f32,
}

/// **Decimate the dense mesh to a triangle budget** and hand back a `MeshAsset`
/// the modelling kernel will accept.
///
/// The order is load-bearing and each step is here for a reason the next one
/// depends on:
///
/// 1. **Local origin, then narrow to `f32`.** The dense mesh is `f64` in
///    baseline units; a `MeshAsset` is `f32`. Subtracting the bounds' centre
///    first is the floating-origin doctrine — it costs nothing and it means the
///    narrowing loses precision relative to the object rather than to wherever
///    the reconstruction's gauge happened to put it.
/// 2. **Manifold filter**, so meshopt is given the manifold input its
///    topology-preserving collapses assume, and so the kernel will take the
///    result.
/// 3. **Decimate.**
/// 4. **Weld and re-order** (`inf_mesh::optimize`), which is also what compacts
///    the vertex buffer meshopt left alone.
/// 5. **Manifold filter again**, because the weld can fuse two vertices and
///    make an edge that was fine into one that is not.
pub fn retopo(dense: &DenseMesh, cfg: &FinishConfig) -> Result<RetopoOutput, FinishError> {
    if dense.positions.is_empty() || dense.indices.len() < 3 {
        return Err(FinishError::EmptyDense {
            vertices: dense.positions.len(),
            triangles: dense.triangle_count(),
        });
    }
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for p in &dense.positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let origin = (lo + hi) * 0.5;

    // ── the named narrowing seam: f64 baseline units -> f32, once ───────────
    let normals = geometric_normals(
        &dense.positions,
        &dense.indices,
        &dense.normals,
        NORMAL_SMOOTHING_ROUNDS,
    );
    let vertices: Vec<MeshVertex> = dense
        .positions
        .iter()
        .zip(&normals)
        .map(|(p, n)| {
            let local = *p - origin;
            MeshVertex {
                position: [local.x as f32, local.y as f32, local.z as f32],
                normal: [n.x as f32, n.y as f32, n.z as f32],
                uv: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            }
        })
        .collect();

    let (filtered, dropped_first) = manifold_filter(&dense.indices);
    let simplified = inf_mesh::simplify(&vertices, &filtered, cfg.target_triangles);
    let (vertices, indices) = inf_mesh::optimize(vertices, simplified.indices);
    let (indices, dropped_second) = manifold_filter(&indices);
    let triangles = indices.len() / 3;
    if triangles == 0 {
        return Err(FinishError::RetopoEmpty {
            before: dense.triangle_count(),
            after: 0,
            dropped: dropped_first + dropped_second,
        });
    }
    let asset = MeshAsset::new(
        vec![SubMesh {
            name: SCAN_SLOT.to_string(),
            vertices,
            indices,
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec![SCAN_SLOT.to_string()],
    );
    Ok(RetopoOutput {
        asset,
        origin,
        dropped: dropped_first + dropped_second,
        triangles,
        error: simplified.error,
    })
}

// ── 2. seams and the unwrap ─────────────────────────────────────────────────

/// Cut seams where the surface changes which way it mostly faces.
///
/// # Why a bucket rule, and not a dihedral angle
///
/// `inf-dcc` has **no auto-seam** — the ROADMAP names it as a P23 remainder —
/// and the Model Editor's seams are author-marked one edge at a time, which is
/// not available to a wizard. So P25.3 needs a rule, and the requirement is
/// harder than "find the sharp edges": LSCM needs each chart to be roughly a
/// *disc*, and a scanned surface has no sharp edges to speak of. A dihedral
/// threshold on a noisy scan either cuts nothing (leaving one chart wrapped
/// around a closed shell, which no conformal map can flatten) or cuts
/// everywhere.
///
/// This is the **box-projection** rule instead: every face is bucketed by which
/// of the six axis directions its normal is closest to, and an edge between two
/// buckets is a seam. Six buckets over a closed surface give charts that are
/// disc-like by construction, and the rule has no threshold to tune.
///
/// # The smoothing pass, which is what makes it usable
///
/// Raw bucketing on a scan speckles: a face whose normal sits near a bucket
/// boundary flips with the noise, and every speckle is a one-face chart. So the
/// **normal field** is low-passed first — `passes` rounds of
/// `normalize(own + Σ neighbours)` over edge neighbours — and the buckets are
/// taken from the smoothed normals.
///
/// Smoothing the *normals* rather than voting on the *buckets* is not a detail.
/// A triangle has three edge neighbours, so a majority vote over buckets can
/// only move a face when all three agree on some other bucket, which true noise
/// almost never does: **measured**, three rounds of majority vote took a
/// jittered grid from 238 seam edges to 226 — a 5% reduction where the noise was
/// most of the cut. Averaging the normals is a real low-pass and takes the same
/// grid well under half.
///
/// Both forms are Jacobi — every face reads the *previous* round — so the
/// result does not depend on the order faces are visited in.
///
/// # The bound the smoothing buys, stated
///
/// A low-pass has a **scale**, and a feature narrower than it is
/// indistinguishable from noise. Measured: a six-triangle spike on a grid is
/// removed entirely by three passes, so its charts merge into the surrounding
/// one and its texels are parameterized as though it were flat. A cube's faces
/// survive the same three passes, and for a reason worth knowing — the four
/// neighbours of a cube face are two *opposed* pairs and cancel exactly, so the
/// face keeps its own normal however many rounds run. The rule is therefore
/// "features wider than a few faces survive; pinpricks do not", and
/// `seam_smoothing_passes` is the knob that says how few.
pub fn seam_edges(mesh: &Mesh, passes: usize, min_chart_faces: usize) -> Vec<inf_dcc::HalfId> {
    use std::collections::BTreeMap;
    const AXES: [DVec3; 6] = [
        DVec3::X,
        DVec3::NEG_X,
        DVec3::Y,
        DVec3::NEG_Y,
        DVec3::Z,
        DVec3::NEG_Z,
    ];
    let faces: Vec<inf_dcc::FaceId> = mesh.face_ids().collect();
    // Edge neighbours, once.
    let mut neighbours: BTreeMap<inf_dcc::FaceId, Vec<inf_dcc::FaceId>> = BTreeMap::new();
    for &f in &faces {
        let Some(loop_halfs) = mesh.face_loop(f) else {
            continue;
        };
        let mut list = Vec::new();
        for h in loop_halfs {
            let Some(t) = mesh.twin(h) else { continue };
            if let Some(Some(n)) = mesh.face_of(t) {
                list.push(n);
            }
        }
        neighbours.insert(f, list);
    }
    let mut normal: BTreeMap<inf_dcc::FaceId, DVec3> = faces
        .iter()
        .map(|&f| (f, inf_dcc::face_normal(mesh, f).unwrap_or(DVec3::Y)))
        .collect();
    for _ in 0..passes {
        let previous = normal.clone();
        for &f in &faces {
            let mut acc = previous[&f];
            for n in &neighbours[&f] {
                acc += previous[n];
            }
            // A neighbourhood whose normals cancel exactly — the inside of a
            // one-face-thick fin — keeps what it had rather than becoming a
            // direction chosen by rounding.
            normal.insert(f, acc.normalize_or(previous[&f]));
        }
    }
    let mut bucket: BTreeMap<inf_dcc::FaceId, usize> = BTreeMap::new();
    for &f in &faces {
        let n = normal[&f];
        let mut best = (f64::NEG_INFINITY, 0usize);
        for (i, axis) in AXES.iter().enumerate() {
            let d = n.dot(*axis);
            // Strictly greater, so a normal exactly between two axes takes the
            // lower index and the rule stays a function.
            if d > best.0 {
                best = (d, i);
            }
        }
        bucket.insert(f, best.1);
    }
    merge_small_charts(&faces, &neighbours, &mut bucket, min_chart_faces);
    let mut out = Vec::new();
    for &f in &faces {
        let Some(loop_halfs) = mesh.face_loop(f) else {
            continue;
        };
        for h in loop_halfs {
            let Some(canonical) = inf_dcc::canonical_edge(mesh, h) else {
                continue;
            };
            if canonical != h {
                continue;
            }
            let Some(t) = mesh.twin(h) else { continue };
            let other = match mesh.face_of(t) {
                Some(Some(n)) => n,
                // A boundary edge is already a chart border; marking it a seam
                // as well would be a second name for one cut.
                _ => continue,
            };
            if bucket[&f] != bucket[&other] {
                out.push(h);
            }
        }
    }
    out
}

/// Fold every chart smaller than `min_faces` into whichever neighbouring chart
/// it shares the most edges with.
///
/// # Why this is not optional either
///
/// Bucketing plus smoothing still leaves a scan full of small islands — a
/// handful of faces that agreed on a bucket their surroundings did not. Each one
/// is a **chart**, each chart is packed with a margin around it, and the packer's
/// shelf is sized by the widest chart. **MEASURED on the committed fixture
/// before this step existed: 9 412 triangles became 2 081 charts — 4.5 triangles
/// each — packed into 19.9% of the atlas.** The texture was mostly gutter, and
/// every island's own gutter is a seam an artist would see.
///
/// The rule is a fixed point rather than one pass: absorbing an island can make
/// its host large enough to absorb the next, so rounds run until nothing moves
/// or `MAX_MERGE_ROUNDS` is spent. Ties go to the lowest bucket index, and
/// components are visited in face order, so the result is a pure function of the
/// mesh.
fn merge_small_charts(
    faces: &[inf_dcc::FaceId],
    neighbours: &std::collections::BTreeMap<inf_dcc::FaceId, Vec<inf_dcc::FaceId>>,
    bucket: &mut std::collections::BTreeMap<inf_dcc::FaceId, usize>,
    min_faces: usize,
) {
    use std::collections::{BTreeMap, BTreeSet};
    /// A cap, so a pathological mesh cannot spin here.
    const MAX_MERGE_ROUNDS: usize = 8;
    if min_faces < 2 {
        return;
    }
    for _ in 0..MAX_MERGE_ROUNDS {
        // Connected components of same-bucket faces, in face order.
        let mut component: BTreeMap<inf_dcc::FaceId, usize> = BTreeMap::new();
        let mut members: Vec<Vec<inf_dcc::FaceId>> = Vec::new();
        for &f in faces {
            if component.contains_key(&f) {
                continue;
            }
            let id = members.len();
            let mut group = Vec::new();
            let mut stack = vec![f];
            component.insert(f, id);
            while let Some(g) = stack.pop() {
                group.push(g);
                for n in &neighbours[&g] {
                    if bucket[n] == bucket[&g] && !component.contains_key(n) {
                        component.insert(*n, id);
                        stack.push(*n);
                    }
                }
            }
            group.sort();
            members.push(group);
        }
        let mut changed = false;
        for group in &members {
            if group.len() >= min_faces {
                continue;
            }
            let own = bucket[&group[0]];
            let inside: BTreeSet<inf_dcc::FaceId> = group.iter().copied().collect();
            let mut shared: BTreeMap<usize, usize> = BTreeMap::new();
            for f in group {
                for n in &neighbours[f] {
                    if inside.contains(n) {
                        continue;
                    }
                    *shared.entry(bucket[n]).or_insert(0) += 1;
                }
            }
            // `max_by_key` over a BTreeMap returns the LAST maximum; the fold is
            // explicit so the tie-break is the lowest bucket index and is
            // readable as such.
            let mut best: Option<(usize, usize)> = None;
            for (b, count) in &shared {
                if best.map(|(bc, _)| *count > bc).unwrap_or(true) {
                    best = Some((*count, *b));
                }
            }
            if let Some((_, target)) = best {
                if target != own {
                    for f in group {
                        bucket.insert(*f, target);
                    }
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Drop every triangle **no camera can see**, and say how many.
///
/// # Why a scan has geometry nobody photographed
///
/// A TSDF only carries a signed distance within its truncation band, so fusing
/// one-sided observations of a wall produces a **slab**: the surface that was
/// photographed, and a second sheet a few voxels behind it where the band runs
/// out. Surface Nets extracts both, because both are zero crossings.
///
/// That far sheet is real geometry, and it is geometry no photograph contains.
/// Left in, it takes half the atlas — **MEASURED on the fixture: 6 505 of
/// 13 076 covered texels seen by no camera** — and every one of those texels is
/// filled by dilation, which makes invented colour look exactly like
/// photographed colour.
///
/// So it goes, by the same test the albedo bake uses: project the triangle's
/// centroid into every view and ask whether any of them has a depth there that
/// agrees. A triangle is kept if **any** camera sees it.
pub fn trim_unseen(
    asset: &MeshAsset,
    origin: DVec3,
    views: &[AlbedoView<'_>],
    tolerance: f32,
) -> (MeshAsset, usize) {
    let mut out = asset.clone();
    let mut dropped = 0usize;
    for sm in &mut out.submeshes {
        let mut kept = Vec::with_capacity(sm.indices.len());
        for tri in sm.indices.chunks_exact(3) {
            let centre = tri
                .iter()
                .map(|&i| {
                    let p = sm.vertices[i as usize].position;
                    DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)
                })
                .fold(DVec3::ZERO, |a, b| a + b)
                / 3.0
                + origin;
            if views.iter().any(|v| view_sees(v, centre, tolerance)) {
                kept.extend_from_slice(tri);
            } else {
                dropped += 1;
            }
        }
        sm.indices = kept;
    }
    (out, dropped)
}

/// Whether one view's depth map agrees that `p` is its nearest surface along
/// that ray — the albedo bake's visibility test, without the colour.
///
/// Public because it is the one question a diagnostics panel and a gate both
/// need to ask, and because a second spelling of it would be a second answer to
/// "did a camera see this", which is the whole basis of the unseen-texel
/// advisory.
///
/// The `!(x > 0.0)` guards are the `inf-photo-gpu` rule, allowed here for the
/// same one reason and only here: a `NaN` compares false against everything, so
/// `!(x > 0.0)` **rejects** it and `x <= 0.0` **accepts** it. Rewriting these to
/// read more smoothly would make a `NaN` depth count as a visible surface.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn view_sees(view: &AlbedoView<'_>, p: DVec3, tolerance: f32) -> bool {
    let camera_space = view.camera.pose.transform(p);
    if !(camera_space.z > 0.0) {
        return false;
    }
    let Some(px) = view.camera.intrinsics.project(camera_space) else {
        return false;
    };
    if !view.camera.intrinsics.contains(px) {
        return false;
    }
    let w = view.depth.width as usize;
    let h = view.depth.height as usize;
    let ix = (px.x.round() as i64).clamp(0, w as i64 - 1) as usize;
    let iy = (px.y.round() as i64).clamp(0, h as i64 - 1) as usize;
    let d = view.depth.depth[iy * w + ix];
    if !(d > 0.0) {
        return false;
    }
    (camera_space.z as f32 - d) <= tolerance * d
}

/// Mark the seams, unwrap, and write the result back — the P23.5 door, used
/// exactly as the Model Editor uses it.
///
/// Returns the mesh asset the writer produced (UVs on the corners, tangents
/// derived from the UV gradients — that derivation is `inf-dcc`'s, and it is
/// the reason the finished asset has tangents at all) and the unwrap's report.
fn unwrap_in_place(
    asset: &MeshAsset,
    cfg: &FinishConfig,
) -> Result<(MeshAsset, inf_dcc::UnwrapReport), FinishError> {
    let import = inf_dcc::from_mesh_asset(asset).map_err(|e| FinishError::Kernel {
        message: e.to_string(),
    })?;
    let mut mesh = import.mesh;
    for h in seam_edges(&mesh, cfg.seam_smoothing_passes, cfg.min_chart_faces) {
        inf_dcc::ops::apply(
            &mut mesh,
            &Op::SetEdgeSeam {
                half: h,
                seam: true,
            },
        )
        .map_err(|e| FinishError::Unwrap {
            message: format!("marking a seam failed: {e}"),
        })?;
    }
    let unwrapped = inf_dcc::unwrap(&mesh).map_err(|e| FinishError::Unwrap {
        message: e.to_string(),
    })?;
    inf_dcc::ops::apply(&mut mesh, &unwrapped.op).map_err(|e| FinishError::Unwrap {
        message: format!("applying the unwrap failed: {e}"),
    })?;
    // `ExportOptions::default()` — `optimize` OFF. meshopt has already run, in
    // `retopo`, where its output is the *input* to everything downstream; a
    // second pass here would reorder the buffer after the UVs are on it for no
    // gain.
    let (out, _report) = inf_dcc::to_mesh_asset(&mesh, &inf_dcc::ExportOptions::default());
    Ok((out, unwrapped.report))
}

// ── 3 + 4. the whole pipeline ───────────────────────────────────────────────

/// A step of [`finish_reconstruction_with_progress`], reported as it **starts**.
///
/// The finish is the longest stage of a capture and it is the one stage whose
/// orchestrator is ours, so it is the one that can say where it has got to.
/// P25.4's wizard reports the two Ring-0 stages either side of it (structure
/// from motion, and the dense solve) as whole stages, because their entry points
/// are single blocking calls in crates this module does not own.
///
/// Ordered, and [`ALL`](FinishStep::ALL) is the order — a progress bar divides
/// by its length rather than by a number written down twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FinishStep {
    /// Decimating the fused mesh and filtering it into the half-edge kernel.
    Retopology,
    /// Dropping triangles no camera photographed.
    Trim,
    /// Seams and the LSCM unwrap.
    Unwrap,
    /// Rasterizing the atlas.
    Atlas,
    /// The normal transfer.
    NormalBake,
    /// The ambient-occlusion bake (the ray-cast one — usually the longest).
    OcclusionBake,
    /// The albedo bake, which re-reads every photograph.
    AlbedoBake,
    /// The de-lighting solve, the dilation, and encoding the three textures.
    Finalize,
}

impl FinishStep {
    /// Every step, in the order [`finish_reconstruction_with_progress`] reports
    /// them.
    pub const ALL: [FinishStep; 8] = [
        FinishStep::Retopology,
        FinishStep::Trim,
        FinishStep::Unwrap,
        FinishStep::Atlas,
        FinishStep::NormalBake,
        FinishStep::OcclusionBake,
        FinishStep::AlbedoBake,
        FinishStep::Finalize,
    ];

    /// A short label for a progress readout.
    pub fn label(self) -> &'static str {
        match self {
            FinishStep::Retopology => "retopology",
            FinishStep::Trim => "trimming unphotographed geometry",
            FinishStep::Unwrap => "UV unwrap",
            FinishStep::Atlas => "atlas",
            FinishStep::NormalBake => "normal bake",
            FinishStep::OcclusionBake => "occlusion bake",
            FinishStep::AlbedoBake => "albedo bake",
            FinishStep::Finalize => "de-lighting and encoding",
        }
    }

    /// Its position in [`ALL`](FinishStep::ALL).
    pub fn index(self) -> usize {
        FinishStep::ALL
            .iter()
            .position(|s| *s == self)
            .expect("every step is in ALL")
    }
}

/// **Run the finish pipeline.** Everything that can refuse refuses here;
/// nothing is written.
///
/// `photos` is indexed by **view index**, the same indexing
/// [`inf_photo_gpu::reconstruct_dense`] takes for its views, so a caller hands
/// over the same list it reconstructed from — re-decoded in colour.
pub fn finish_reconstruction(
    dense: &DenseReconstruction,
    reconstruction: &Reconstruction,
    photos: &[RgbImage],
    cfg: &FinishConfig,
    pool: &JobPool,
) -> Result<FinishedAsset, FinishError> {
    finish_reconstruction_with_progress(dense, reconstruction, photos, cfg, pool, &mut |_| {})
}

/// [`finish_reconstruction`], with `on_step` called **as each step begins**.
///
/// One function with a callback rather than two orchestrators: P25.4's wizard
/// needs a progress bar, and the way this codebase has paid for a second copy of
/// a pipeline is well enough documented that driving the public per-stage
/// functions from the session would have been a second answer to "what is a
/// finish". The callback runs on the calling thread, between steps, and does not
/// see anything the returned [`FinishedAsset`] does not.
///
/// It is **not** a cancellation seam. Every step is one blocking call into a
/// pool-parallel kernel, so the finest granularity anything here can stop at is
/// a whole step — and a caller that stopped between steps would still hold no
/// result. The wizard's Cancel is therefore between *stages*, and says so.
pub fn finish_reconstruction_with_progress(
    dense: &DenseReconstruction,
    reconstruction: &Reconstruction,
    photos: &[RgbImage],
    cfg: &FinishConfig,
    pool: &JobPool,
    on_step: &mut dyn FnMut(FinishStep),
) -> Result<FinishedAsset, FinishError> {
    // ── refusals first ──────────────────────────────────────────────────────
    //
    // The two reconstructions arrive as two arguments and nothing has ever
    // checked that they came from one solve. `depth_maps` and `surface` are
    // indexed by SLOT below, so a disagreement was an index panic — which is a
    // refusal nobody can read and a wizard cannot show.
    if reconstruction.cameras.is_empty() {
        return Err(FinishError::NoCameras);
    }
    if dense.depth_maps.len() != reconstruction.cameras.len()
        || dense.surface.len() != reconstruction.cameras.len()
    {
        return Err(FinishError::ViewsMismatch {
            cameras: reconstruction.cameras.len(),
            depth_maps: dense.depth_maps.len(),
            surface: dense.surface.len(),
        });
    }
    if !(cfg.metres_per_unit.is_finite() && cfg.metres_per_unit > 0.0) {
        return Err(FinishError::BadScale {
            metres_per_unit: cfg.metres_per_unit,
        });
    }
    for camera in reconstruction.cameras.values() {
        let Some(photo) = photos.get(camera.view as usize) else {
            return Err(FinishError::MissingPhoto {
                view: camera.view,
                given: photos.len(),
            });
        };
        if photo.width() != camera.intrinsics.width || photo.height() != camera.intrinsics.height {
            return Err(FinishError::PhotoMismatch {
                view: camera.view,
                img_w: photo.width(),
                img_h: photo.height(),
                int_w: camera.intrinsics.width,
                int_h: camera.intrinsics.height,
            });
        }
    }
    let mut advisories: Vec<FinishAdvisory> = Vec::new();

    // ── 1. retopo ───────────────────────────────────────────────────────────
    on_step(FinishStep::Retopology);
    let retopo_out = retopo(&dense.mesh, cfg)?;
    if retopo_out.dropped > 0 {
        advisories.push(FinishAdvisory::NonManifoldDropped {
            dropped: retopo_out.dropped,
            examined: dense.mesh.triangle_count(),
        });
    }
    if retopo_out.triangles > cfg.target_triangles {
        advisories.push(FinishAdvisory::DecimationShortOfBudget {
            asked: cfg.target_triangles,
            got: retopo_out.triangles,
        });
    }

    // ── the views, built once ───────────────────────────────────────────────
    //
    // Before the unwrap rather than after, because the trim runs on them and
    // trimming geometry after parameterizing it would leave charts with holes
    // the packer had already made room for.
    let views: Vec<AlbedoView<'_>> = reconstruction
        .cameras
        .values()
        .enumerate()
        .map(|(slot, camera)| AlbedoView {
            camera,
            image: &photos[camera.view as usize],
            // `depth_maps` and `surface` are in ASCENDING REGISTERED-VIEW order,
            // which is the order a `BTreeMap`'s values come out in — not indexed
            // by view id. Reading them with `camera.view` would silently pair a
            // camera with another view's depth on any reconstruction that failed
            // to register a view.
            depth: &dense.depth_maps[slot],
            hints: &dense.surface[slot],
        })
        .collect();

    // ── 1b. trim what nobody photographed ───────────────────────────────────
    on_step(FinishStep::Trim);
    let mut trimmed = retopo_out.asset.clone();
    let mut trimmed_triangles = 0usize;
    if cfg.trim_unseen {
        let (out, dropped) = trim_unseen(
            &retopo_out.asset,
            retopo_out.origin,
            &views,
            cfg.bake.occlusion_tolerance,
        );
        if out.triangle_count() > 0 {
            trimmed = out;
            trimmed_triangles = dropped;
        }
        // A trim that removes EVERYTHING is a trim that was wrong about the
        // cameras, not a scan nobody photographed. Keeping the untrimmed mesh
        // in that case turns a silent empty asset into a visible bad one, which
        // is the direction this codebase errs in.
    }
    if trimmed_triangles > 0 {
        advisories.push(FinishAdvisory::UnphotographedTrimmed {
            dropped: trimmed_triangles,
            examined: retopo_out.triangles,
        });
    }

    // ── 2. unwrap ───────────────────────────────────────────────────────────
    on_step(FinishStep::Unwrap);
    let (mut asset, unwrap_report) = unwrap_in_place(&trimmed, cfg)?;
    if unwrap_report.flipped > 0 {
        advisories.push(FinishAdvisory::UvChartsFlipped {
            flipped: unwrap_report.flipped,
            corners: unwrap_report.corners,
        });
    }

    // ── 3. bake ─────────────────────────────────────────────────────────────
    //
    // Back into the reconstruction's frame: the atlas positions have to be
    // where the depth maps and the BVH think they are, and the mesh is local.
    on_step(FinishStep::Atlas);
    let (positions, normals, uvs, indices) = streams(&asset, retopo_out.origin);
    let atlas = rasterize_atlas(&positions, &normals, &uvs, &indices, cfg.bake.size);
    if atlas.overlaps > 0 {
        advisories.push(FinishAdvisory::AtlasOverlaps {
            texels: atlas.overlaps,
        });
    }
    let surface = BvhSurface::new(&dense.mesh);
    let voxel_size = dense.report.voxel_size;
    on_step(FinishStep::NormalBake);
    let (normal_channel, normal_report) =
        bake_normals(&atlas, &surface, voxel_size, &cfg.bake, pool);
    if normal_report.fallbacks > 0 {
        advisories.push(FinishAdvisory::NormalTransferFallbacks {
            fallbacks: normal_report.fallbacks,
            texels: normal_report.texels,
        });
    }
    on_step(FinishStep::OcclusionBake);
    let ao_channel = bake_ao(&atlas, &surface, voxel_size, &cfg.bake, pool);
    on_step(FinishStep::AlbedoBake);
    let (albedo_channel, weight_channel, albedo_report) =
        bake_albedo(&atlas, &views, &cfg.bake, pool);
    if albedo_report.unseen > 0 {
        advisories.push(FinishAdvisory::UnseenTexels {
            unseen: albedo_report.unseen,
            covered: atlas.covered_count(),
        });
    }

    // De-lighting fits on the WELL-SEEN texels only: a texel one camera glanced
    // at contributes as much noise as signal to a global illumination estimate.
    on_step(FinishStep::Finalize);
    let well_seen: Vec<bool> = weight_channel
        .data
        .iter()
        .map(|w| w[0] >= cfg.bake.well_seen_weight as f64)
        .collect();
    let (albedo_channel, delight_report) =
        delight(&albedo_channel, &normal_channel, &well_seen, &cfg.delight);
    if cfg.delight.enabled && !delight_report.applied {
        advisories.push(FinishAdvisory::DelightRefused {
            explained_percent: (delight_report.explained.max(0.0) * 100.0).round() as u32,
            required_percent: (cfg.delight.min_explained * 100.0).round() as u32,
        });
    }

    let albedo_filled = dilate(&albedo_channel);
    let normal_filled = dilate(&normal_channel);
    let ao_filled = dilate(&ao_channel);

    // ── the ORM pack ────────────────────────────────────────────────────────
    let orm = orm_channel(&ao_filled, cfg);

    // ── textures ────────────────────────────────────────────────────────────
    let size = cfg.bake.size;
    let albedo = texture_from_rgba8(
        albedo_filled.to_rgba8(),
        size,
        size,
        TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression: TextureCompression::Auto,
        },
    )
    .map_err(|e| FinishError::Texture {
        channel: "base colour",
        message: e.to_string(),
    })?;
    // Normals and ORM are DATA, not colour: linear, and uncompressed. BC1 on a
    // normal map is the artifact `TextureImportSettings::data` was written to
    // avoid, and the workspace has no BC5.
    let normal = texture_from_rgba8(
        normal_filled.to_rgba8(),
        size,
        size,
        TextureImportSettings::data(),
    )
    .map_err(|e| FinishError::Texture {
        channel: "normal",
        message: e.to_string(),
    })?;
    let orm = texture_from_rgba8(orm.to_rgba8(), size, size, TextureImportSettings::data())
        .map_err(|e| FinishError::Texture {
            channel: "ORM",
            message: e.to_string(),
        })?;

    // ── the scale step, applied once, at the end ────────────────────────────
    scale_mesh(&mut asset, cfg.metres_per_unit);

    let final_triangles = asset.triangle_count();
    if final_triangles < VGEOM_MIN_TRIANGLES {
        advisories.push(FinishAdvisory::BelowVirtualizationThreshold {
            triangles: final_triangles,
            threshold: VGEOM_MIN_TRIANGLES,
        });
    }
    advisories.sort();
    advisories.dedup();

    let (u_span, v_span) = uv_span(&asset);
    let report = FinishReport {
        dense_triangles: dense.mesh.triangle_count(),
        non_manifold_dropped: retopo_out.dropped,
        decimated_triangles: retopo_out.triangles,
        unphotographed_dropped: trimmed_triangles,
        final_triangles,
        final_vertices: asset.vertex_count(),
        seams: unwrap_report.seams,
        charts: unwrap_report.charts.len(),
        flipped: unwrap_report.flipped,
        worst_residual: unwrap_report.worst_residual,
        worst_convergence: unwrap_report.worst_convergence,
        atlas_coverage: atlas.covered_count() as f64
            / (cfg.bake.size as f64 * cfg.bake.size as f64),
        atlas_u_span: u_span,
        atlas_v_span: v_span,
        normals: normal_report,
        albedo: albedo_report,
        delight: delight_report,
        voxel_size,
    };
    let material = MaterialAsset {
        metallic: cfg.metallic,
        roughness: cfg.roughness,
        ..MaterialAsset::default()
    };
    Ok(FinishedAsset {
        mesh: asset,
        origin_units: retopo_out.origin,
        albedo,
        normal,
        orm,
        material,
        advisories,
        report,
    })
}

/// The mesh's streams, back in the reconstruction's frame.
fn streams(asset: &MeshAsset, origin: DVec3) -> (Vec<DVec3>, Vec<DVec3>, Vec<[f32; 2]>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for sm in &asset.submeshes {
        let base = positions.len() as u32;
        for v in &sm.vertices {
            positions.push(
                DVec3::new(
                    v.position[0] as f64,
                    v.position[1] as f64,
                    v.position[2] as f64,
                ) + origin,
            );
            normals.push(
                DVec3::new(v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64)
                    .normalize_or(DVec3::Y),
            );
            uvs.push(v.uv);
        }
        indices.extend(sm.indices.iter().map(|i| i + base));
    }
    (positions, normals, uvs, indices)
}

/// Ambient occlusion in **red**, roughness in green, metallic in blue — the
/// glTF ORM packing.
///
/// The material model has three texture slots and none of them is "occlusion";
/// glTF's answer is that occlusion rides the red channel of the
/// metallic-roughness map, and following it means the AO texture has a
/// **referrer** rather than being an orphan the delete-with-references check
/// would happily remove. The engine's PBR pass does not read red yet — the same
/// "the consumer arrives later" story the object-space normal map is on.
fn orm_channel(ao: &Channel, cfg: &FinishConfig) -> Channel {
    let mut out = Channel::new(ao.size, [1.0, cfg.roughness as f64, cfg.metallic as f64]);
    for i in 0..out.data.len() {
        out.data[i] = [ao.data[i][0], cfg.roughness as f64, cfg.metallic as f64];
        out.valid[i] = true;
    }
    out
}

/// The scale step: metres per baseline unit, applied to positions and bounds
/// and to nothing else.
///
/// A uniform scale leaves unit normals unit, leaves tangent directions where
/// they were, and does not touch UVs — which is why the whole metric question
/// can be one multiplier at the end rather than a unit carried through every
/// stage.
fn scale_mesh(asset: &mut MeshAsset, metres_per_unit: f64) {
    if metres_per_unit == 1.0 {
        return;
    }
    let s = metres_per_unit as f32;
    for sm in &mut asset.submeshes {
        for v in &mut sm.vertices {
            v.position = [v.position[0] * s, v.position[1] * s, v.position[2] * s];
        }
    }
    asset.bounds = inf_mesh::asset::Aabb::from_points(
        asset
            .submeshes
            .iter()
            .flat_map(|s| s.vertices.iter().map(|v| v.position)),
    );
}

/// The extent of the written UVs along each axis — the atlas-usage measurement
/// P23.6 learned to take (its bake reached 0.13 of the `u` axis before the
/// bevel ruling and 0.77 after).
fn uv_span(asset: &MeshAsset) -> (f64, f64) {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for sm in &asset.submeshes {
        for v in &sm.vertices {
            for c in 0..2 {
                lo[c] = lo[c].min(v.uv[c] as f64);
                hi[c] = hi[c].max(v.uv[c] as f64);
            }
        }
    }
    if lo[0] > hi[0] {
        return (0.0, 0.0);
    }
    (hi[0] - lo[0], hi[1] - lo[1])
}

// ── the write ───────────────────────────────────────────────────────────────

/// Write a finished asset into a project — **all five, or none**.
///
/// # The torn-build rollback (P24.5's pattern, met again)
///
/// Five assets that reference each other are not five writes; they are one
/// build. A failure at the fourth leaves three files on disk whose names are
/// taken, whose GUIDs the caller never learned, and one of which references a
/// material that does not exist. So every successful write is remembered and a
/// failure deletes them **newest first, with `force = true`** — the edges point
/// *into* the set (the mesh names the material, the material names the
/// textures), so a reference-guarded delete would refuse the textures and leave
/// half a scan behind.
///
/// Write order is dependency order: textures, material, mesh.
///
/// # `dir` is taken verbatim, and so is the name
///
/// It goes straight to [`AssetProject::write_asset`], which does **not** join it
/// onto the project root — a relative path resolves against the process's
/// working directory. Callers pass an absolute path, or one from
/// `AssetProject::content_dir`.
///
/// # What this does NOT write
///
/// A **`.inf_vmesh`**. The editor's interactive viewport has exactly one door
/// for real geometry — a meshlet DAG — and derives one beside every mesh the
/// glTF importer writes (`assets::vmesh::ensure_vmesh`, P18.3) and every mesh a
/// modelling session saves (`dcc::save`). A finished scan therefore lands in the
/// content root as a correct, re-openable asset that the viewport draws as a
/// **placeholder cube**, whatever its triangle count — which is a stronger
/// version of the hazard [`FinishAdvisory::BelowVirtualizationThreshold`] names
/// for shipped builds, and one raising the triangle budget does not fix.
///
/// It is not done here because `ensure_vmesh` runs for seconds on a
/// fifteen-thousand-triangle mesh and would put a sixth asset outside this
/// function's all-or-none set. P25.4's wizard is the door that places a finish in
/// a scene and is where the derivation belongs, on the pattern the import
/// orchestrator already uses.
pub fn write_finished(
    project: &mut crate::assets::AssetProject,
    dir: &Path,
    name: &str,
    finished: &FinishedAsset,
) -> Result<FinishedIds, FinishError> {
    let mut written: Vec<AssetId> = Vec::new();
    let albedo = write_one(
        project,
        &mut written,
        dir,
        &format!("{name}_BaseColor"),
        &finished.albedo,
        Vec::new(),
        "base colour",
    )?;
    let normal = write_one(
        project,
        &mut written,
        dir,
        &format!("{name}_Normal"),
        &finished.normal,
        Vec::new(),
        "normal",
    )?;
    let orm = write_one(
        project,
        &mut written,
        dir,
        &format!("{name}_ORM"),
        &finished.orm,
        Vec::new(),
        "ORM",
    )?;
    let mut material = finished.material.clone();
    material.base_color_texture = Some(albedo);
    material.normal_texture = Some(normal);
    material.metallic_roughness_texture = Some(orm);
    let material_id = write_one(
        project,
        &mut written,
        dir,
        name,
        &material,
        vec![albedo, normal, orm],
        "material",
    )?;
    let mesh = write_one(
        project,
        &mut written,
        dir,
        name,
        &finished.mesh,
        vec![material_id],
        "mesh",
    )?;
    Ok(FinishedIds {
        mesh,
        albedo,
        normal,
        orm,
        material: material_id,
    })
}

/// One write, remembered, with the rollback on failure.
fn write_one<T: inf_asset::AssetPayload>(
    project: &mut crate::assets::AssetProject,
    written: &mut Vec<AssetId>,
    dir: &Path,
    name: &str,
    payload: &T,
    dependencies: Vec<AssetId>,
    what: &'static str,
) -> Result<AssetId, FinishError> {
    match project.write_asset(dir, name, payload, None, dependencies, None) {
        Ok(id) => {
            written.push(id);
            Ok(id)
        }
        Err(e) => {
            roll_back(project, written);
            Err(FinishError::Write {
                what,
                message: e.to_string(),
            })
        }
    }
}

/// Undo a partial build.
///
/// Newest first and forced, for the reason [`write_finished`] gives. Failures
/// are ignored on purpose: this runs on an already-failing path, and a second
/// I/O error here would replace the diagnosis with its own consequence.
fn roll_back(project: &mut crate::assets::AssetProject, written: &[AssetId]) {
    for id in written.iter().rev() {
        let _ = project.delete(*id, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifold_filter_keeps_a_closed_surface_and_drops_the_third_face_on_an_edge() {
        // Two triangles sharing an edge, wound consistently: nothing dropped.
        let (kept, dropped) = manifold_filter(&[0, 1, 2, 0, 2, 3]);
        assert_eq!(dropped, 0, "a legal quad lost a triangle");
        assert_eq!(kept.len(), 6);
        // A third face on the same directed edge: dropped.
        let (kept, dropped) = manifold_filter(&[0, 1, 2, 0, 2, 3, 0, 1, 4]);
        assert_eq!(dropped, 1, "the third user of edge 0->1 survived");
        assert_eq!(kept, vec![0, 1, 2, 0, 2, 3]);
        // Two faces wound the SAME way across a shared edge is also refused by
        // the kernel, so it must be refused here.
        let (_, dropped) = manifold_filter(&[0, 1, 2, 1, 0, 5, 0, 1, 6]);
        assert_eq!(dropped, 1, "a same-winding pair was not caught");
        // A degenerate triangle is dropped rather than passed to the kernel.
        let (kept, dropped) = manifold_filter(&[0, 1, 1]);
        assert_eq!((kept.len(), dropped), (0, 1));
    }

    #[test]
    fn the_manifold_filter_is_a_pure_function_of_its_input() {
        let indices: Vec<u32> = (0..600u32).collect();
        let a = manifold_filter(&indices);
        let b = manifold_filter(&indices);
        assert_eq!(a, b);
    }

    #[test]
    fn a_filtered_mesh_actually_enters_the_kernel() {
        // The whole reason the filter exists, measured against the door that
        // refuses: a cube plus a stray fin sharing one of its edges.
        let cube = inf_dcc::cube(1.0);
        let (asset, _) = inf_dcc::to_mesh_asset(&cube, &inf_dcc::ExportOptions::default());
        let sm = &asset.submeshes[0];
        let mut vertices = sm.vertices.clone();
        let mut indices = sm.indices.clone();
        // A fin on the cube's first edge, wound so it duplicates a directed
        // edge that is already used.
        let fin = vertices.len() as u32;
        vertices.push(MeshVertex {
            position: [4.0, 4.0, 4.0],
            ..MeshVertex::default()
        });
        indices.extend_from_slice(&[indices[0], indices[1], fin]);
        let broken = MeshAsset::new(
            vec![SubMesh {
                name: "t".into(),
                vertices: vertices.clone(),
                indices: indices.clone(),
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["t".into()],
        );
        assert!(
            inf_dcc::from_mesh_asset(&broken).is_err(),
            "the fin did not make the kernel refuse — this test proves nothing"
        );
        let (filtered, dropped) = manifold_filter(&indices);
        assert_eq!(dropped, 1);
        let fixed = MeshAsset::new(
            vec![SubMesh {
                name: "t".into(),
                vertices,
                indices: filtered,
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["t".into()],
        );
        assert!(
            inf_dcc::from_mesh_asset(&fixed).is_ok(),
            "the filtered mesh still will not open"
        );
    }

    #[test]
    fn the_seam_rule_cuts_a_cube_into_its_six_faces() {
        // The box-projection rule's simplest possible case, and the one that
        // says the rule is doing what it claims: a cube's six faces are six
        // buckets, so every one of its twelve edges is a seam.
        let cube = inf_dcc::cube(1.0);
        let seams = seam_edges(&cube, 0, 0);
        assert_eq!(seams.len(), 12, "a cube did not cut into six charts");
        // AND SMOOTHING MUST NOT UNDO IT. A cube is the adversarial case for
        // any smoother: every face disagrees with all four of its neighbours.
        // The first rule here — a majority vote over buckets — broke a
        // six-way tie by index, put every face in bucket 0, and cut a cube into
        // **zero** seams and one chart wrapped around a closed solid. Normal
        // averaging survives it for a better reason than a tie-break: the four
        // neighbours of a cube face are two opposed pairs and cancel exactly.
        assert_eq!(
            seam_edges(&cube, 3, 0).len(),
            12,
            "smoothing dissolved a cube's own faces"
        );
    }

    /// An `n`-by-`n` quad grid in the `xz` plane, as a `MeshAsset` — the DCC
    /// kernel's own `plane` is a single quad, which has no interior edges to
    /// bucket and no neighbours to vote.
    ///
    /// `bump` folds the grid into a tent of that slope along `x` (a real
    /// feature, many faces wide); `jitter` displaces **every** vertex by a hash
    /// of its coordinates (scan noise). The jitter is a pure function of
    /// `(i, j)`, so the grid is the same mesh on every run.
    fn grid_asset(n: u32, bump: Option<f32>, jitter: f32) -> MeshAsset {
        let mut vertices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let (x, z) = (i as f32 / n as f32 - 0.5, j as f32 / n as f32 - 0.5);
                let h = inf_photo::hash::Hash64::new(0x9e37)
                    .mix_usize(i as usize)
                    .mix_usize(j as usize)
                    .unit() as f32;
                // `bump` folds the grid into a tent along `x` — a feature many
                // faces wide, which is the kind the smoother must keep.
                let y = bump.map(|slope| slope * x.abs()).unwrap_or(0.0) + (h - 0.5) * 2.0 * jitter;
                vertices.push(MeshVertex {
                    position: [x, y, z],
                    normal: [0.0, 1.0, 0.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
            }
        }
        let idx = |i: u32, j: u32| j * (n + 1) + i;
        let mut indices = Vec::new();
        for j in 0..n {
            for i in 0..n {
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j + 1), idx(i + 1, j)]);
                indices.extend_from_slice(&[idx(i, j), idx(i, j + 1), idx(i + 1, j + 1)]);
            }
        }
        MeshAsset::new(
            vec![SubMesh {
                name: "p".into(),
                vertices,
                indices,
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["p".into()],
        )
    }

    #[test]
    fn the_seam_rule_leaves_a_flat_plane_uncut() {
        // The other end of the same claim: a surface that faces one way has no
        // seams at all, so the unwrapper gets one chart and not a hundred.
        let mesh = inf_dcc::from_mesh_asset(&grid_asset(6, None, 0.0))
            .expect("a grid is manifold")
            .mesh;
        assert!(
            seam_edges(&mesh, 3, 0).is_empty(),
            "a flat plane was cut into charts"
        );
    }

    #[test]
    fn smoothing_removes_the_speckle_scan_noise_creates() {
        // The case the smoothing pass exists for, and the one a real scan is:
        // a nominally flat surface whose vertices are displaced by noise of the
        // order of the cell size, so face normals wobble across the bucket
        // boundaries and the raw rule cuts a seam almost everywhere.
        let mesh = inf_dcc::from_mesh_asset(&grid_asset(12, None, 0.14))
            .expect("a jittered grid is still manifold")
            .mesh;
        let raw = seam_edges(&mesh, 0, 0).len();
        let smoothed = seam_edges(&mesh, 3, 0).len();
        assert!(
            raw > 40,
            "the noise did not speckle the buckets: {raw} seams"
        );
        assert!(
            (smoothed as f64) < raw as f64 * 0.6,
            "smoothing barely helped: {raw} raw, {smoothed} smoothed"
        );
    }

    #[test]
    fn smoothing_keeps_a_feature_that_is_not_a_speckle() {
        // The other half of the claim: a tent's two halves are planar and many
        // faces wide, so averaging inside each one changes nothing and the fold
        // survives. A smoother that dissolved it would flatten every real crease
        // on the model along with the noise, and the seam rule would stop
        // cutting where the surface actually turns.
        //
        // The bound the module docs state is the same experiment on a
        // SIX-triangle spike, which three passes remove entirely — a feature
        // narrower than the kernel is noise as far as any low-pass is concerned.
        let mesh = inf_dcc::from_mesh_asset(&grid_asset(8, Some(2.0), 0.0))
            .expect("a tent is still manifold")
            .mesh;
        let raw = seam_edges(&mesh, 0, 0).len();
        let smoothed = seam_edges(&mesh, 3, 0).len();
        assert!(raw > 0, "the fold did not create a bucket boundary at all");
        assert!(
            smoothed * 2 >= raw,
            "smoothing dissolved a real crease: {raw} raw, {smoothed} smoothed"
        );
    }

    #[test]
    fn the_scale_step_moves_positions_and_bounds_and_nothing_else() {
        let cube = inf_dcc::cube(1.0);
        let (asset, _) = inf_dcc::to_mesh_asset(&cube, &inf_dcc::ExportOptions::default());
        let mut scaled = asset.clone();
        scale_mesh(&mut scaled, 2.5);
        for (a, b) in asset.submeshes[0]
            .vertices
            .iter()
            .zip(&scaled.submeshes[0].vertices)
        {
            for c in 0..3 {
                assert!((b.position[c] - a.position[c] * 2.5).abs() < 1e-6);
            }
            assert_eq!(a.normal, b.normal, "a uniform scale moved a normal");
            assert_eq!(a.uv, b.uv, "a uniform scale moved a UV");
            assert_eq!(a.tangent, b.tangent, "a uniform scale moved a tangent");
        }
        assert!(
            (scaled.bounds.max[0] - asset.bounds.max[0] * 2.5).abs() < 1e-6,
            "the bounds did not follow the positions"
        );
        // The documented default is the identity, and it must be exactly that.
        let mut untouched = asset.clone();
        scale_mesh(&mut untouched, 1.0);
        assert_eq!(untouched, asset);
    }

    #[test]
    fn the_barycentric_helper_reproduces_the_point_it_was_given() {
        let (a, b, c) = (
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 3.0),
        );
        for (u, v, w) in [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.25, 0.5, 0.25),
        ] {
            let p = a * u + b * v + c * w;
            let (gu, gv, gw) = barycentric(p, a, b, c);
            assert!(
                (gu - u).abs() < 1e-9 && (gv - v).abs() < 1e-9 && (gw - w).abs() < 1e-9,
                "({u},{v},{w}) came back as ({gu},{gv},{gw})"
            );
        }
        // A degenerate triangle is a value, not a division by zero.
        let (gu, _, _) = barycentric(a, a, a, a);
        assert_eq!(gu, 1.0);
    }

    #[test]
    fn the_bvh_surface_maps_a_leaf_back_to_the_triangle_it_came_from() {
        // The one thing this wrapper adds over `inf_dcc::Bvh` (which is
        // mutation-hardened where it lives): the normal it reports. `Bvh`
        // reorders triangles into leaves, so `ClosestHit::triangle` is a LEAF
        // index and reading the index buffer with it lands on somebody else's
        // vertices. On a smooth mesh that is invisible; here every triangle
        // faces a different way, so it is not.
        //
        // Two things about the construction are load-bearing. The triangles
        // share **no vertices**, which keeps `geometric_normals`' smoothing out
        // of the measurement — an isolated triangle has no neighbour to average
        // with. And they are laid out in an order that is **not** their spatial
        // order, because a median-split hierarchy over triangles already sorted
        // along an axis leaves them where they were and `source_index` is then
        // the identity: measured, with the mapping severed and the triangles in
        // a row, all 24 probes still passed.
        //
        // (An earlier version used ONE triangle and asserted an interpolation
        // between three hand-picked vertex normals. Four rounds of smoothing
        // collapse a lone triangle's normals onto its face normal — the smoother
        // working — and the arm failed for a reason that had nothing to do with
        // the BVH.)
        const N: u32 = 24;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        let mut faces: Vec<(DVec3, DVec3)> = Vec::new();
        for i in 0..N {
            // A deterministic spread of orientations, no trig: the normal is a
            // normalized lattice vector.
            let n = DVec3::new(
                ((i % 5) as f64) - 2.0,
                ((i % 3) as f64) + 1.0,
                ((i % 7) as f64) - 3.0,
            )
            .normalize();
            let t = if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
            let u = n.cross(t).normalize();
            let v = n.cross(u);
            // Spatially scrambled against the index order.
            let centre = DVec3::new(
                ((i * 7) % N) as f64 * 4.0,
                ((i * 11) % 5) as f64 * 4.0,
                ((i * 13) % 7) as f64 * 4.0,
            );
            let base = positions.len() as u32;
            for corner in [u, v, -u - v] {
                positions.push(centre + corner);
                normals.push([n.x as f32, n.y as f32, n.z as f32]);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2]);
            faces.push((centre, n));
        }
        let mesh = DenseMesh {
            positions,
            normals,
            confidence: vec![1.0; (N * 3) as usize],
            indices,
        };
        let surface = BvhSurface::new(&mesh);
        for (i, (centre, want)) in faces.iter().enumerate() {
            let hit = surface
                .closest(*centre + *want * 0.35)
                .expect("a mesh with triangles has a closest point");
            assert!((hit.normal.length() - 1.0).abs() < 1e-9, "not unit");
            assert!(
                hit.normal.dot(*want) > 0.99,
                "triangle {i}: the hierarchy reported {} where {want} lives — the leaf index was \
                 not mapped back through `source_index`",
                hit.normal
            );
        }
        // And the occlusion query is a distance in the units it was given.
        let flat = DenseMesh {
            positions: vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 2.0),
            ],
            normals: vec![[0.0, 1.0, 0.0]; 3],
            confidence: vec![1.0; 3],
            indices: vec![0, 1, 2],
        };
        let flat = BvhSurface::new(&flat);
        assert!(flat.occluded(DVec3::new(1.5, 1.0, 0.5), DVec3::NEG_Y, 2.0));
        assert!(!flat.occluded(DVec3::new(1.5, 1.0, 0.5), DVec3::NEG_Y, 0.5));
        assert!(!flat.occluded(DVec3::new(1.5, 1.0, 0.5), DVec3::Y, 100.0));
    }

    #[test]
    fn an_empty_dense_mesh_refuses_by_name() {
        let empty = DenseMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            confidence: Vec::new(),
            indices: Vec::new(),
        };
        assert!(matches!(
            retopo(&empty, &FinishConfig::default()),
            Err(FinishError::EmptyDense { .. })
        ));
    }

    /// Every variant, once, so a text assertion below cannot miss one.
    fn every_advisory() -> Vec<FinishAdvisory> {
        vec![
            FinishAdvisory::NonManifoldDropped {
                dropped: 1,
                examined: 2,
            },
            FinishAdvisory::UnphotographedTrimmed {
                dropped: 3,
                examined: 4,
            },
            FinishAdvisory::DecimationShortOfBudget { asked: 5, got: 6 },
            FinishAdvisory::BelowVirtualizationThreshold {
                triangles: 7,
                threshold: VGEOM_MIN_TRIANGLES,
            },
            FinishAdvisory::UvChartsFlipped {
                flipped: 8,
                corners: 9,
            },
            FinishAdvisory::AtlasOverlaps { texels: 10 },
            FinishAdvisory::UnseenTexels {
                unseen: 11,
                covered: 12,
            },
            FinishAdvisory::NormalTransferFallbacks {
                fallbacks: 13,
                texels: 14,
            },
            FinishAdvisory::DelightRefused {
                explained_percent: 15,
                required_percent: 25,
            },
        ]
    }

    #[test]
    fn the_advisories_are_ordered_and_say_what_happened() {
        let mut list = vec![
            FinishAdvisory::UnseenTexels {
                unseen: 3,
                covered: 10,
            },
            FinishAdvisory::NonManifoldDropped {
                dropped: 1,
                examined: 2,
            },
        ];
        list.sort();
        assert!(matches!(list[0], FinishAdvisory::NonManifoldDropped { .. }));
        for a in &list {
            let text = a.to_string();
            assert!(text.len() > 30, "an advisory says almost nothing: {text}");
        }
        // The one that names the remedy names it.
        let below = FinishAdvisory::BelowVirtualizationThreshold {
            triangles: 100,
            threshold: VGEOM_MIN_TRIANGLES,
        }
        .to_string();
        assert!(below.contains("min_triangles"), "no remedy: {below}");
        assert!(below.contains("PLACEHOLDER") || below.contains("placeholder"));
    }

    #[test]
    fn no_advisory_carries_a_mangled_continuation() {
        // P22's law, met a fourth time: a scripted edit that eats the `\` before
        // a newline leaves the source indentation *inside* the literal, and the
        // reader gets eighteen spaces in the middle of a sentence. It compiles,
        // it formats, and nothing but a human reading the rendered string sees
        // it — which is why this is asserted rather than reviewed.
        //
        // These strings are also read by the P25.3 gate's determinism byte
        // image, so a run of spaces is a committed artefact and not only a
        // cosmetic one.
        for a in every_advisory() {
            let text = a.to_string();
            assert!(
                !text.contains("  "),
                "{a:?} renders a run of spaces — a line continuation was eaten: {text}"
            );
            assert!(text.len() > 30, "an advisory says almost nothing: {text}");
        }
    }
}
