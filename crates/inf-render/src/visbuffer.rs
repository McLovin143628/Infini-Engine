//! **The visibility buffer's packing contract** (P28.1).
//!
//! One `R32Uint` texel names the triangle a pixel sees:
//! `instance ⊕ meshlet ⊕ triangle`, packed into thirty-two bits. Everything the
//! material-resolve pass needs — the model matrix, the material, the three
//! vertices, the uv — is reachable from that one word plus the shared
//! virtualized-geometry pools, so a pixel's shading is a pure function of its id.
//!
//! # Why the packing is a contract and not an implementation detail
//!
//! Two independent programs read these bits: `visbuffer.wgsl` writes them,
//! `vis_resolve.wgsl` and `vis_feedback.wgsl` read them, and the arms below read
//! them a third time in Rust. Three readers of one layout is exactly the shape
//! that drifts silently — a shifted field is not a validation error, it is a
//! frame that shades the wrong triangle and keeps rendering. So the split is
//! *stated* here, mirrored by a `const` in each shader, and pinned by
//! `the_shaders_bit_split_is_the_rusts` reading the WGSL source.
//!
//! # The bit split, designed against the real ceilings
//!
//! | field | bits | range | the ceiling it was designed against |
//! |---|---|---|---|
//! | triangle | 7 | 0..=127 | `inf_vgeom::MeshletParams::max_triangles` defaults to **124** and every cooked `.inf_vmesh` in the tree carries that; `meshopt`'s hard cap is 512, so 128 is a *refusal* boundary rather than a format one |
//! | meshlet | 14 | 0..=16 383 | a **shared pool slot**, not an asset-local id. The flagship `vgeom-demo` DAG is ~500 slots for a 10 M-triangle scene, because meshlets are stored once per asset and instanced — 16 384 is 32× that |
//! | instance | 11 | 0..=2 046 | `RenderScene::vgeom_instances`; `vgeom-demo` places **324** (an 18 × 18 grid) |
//!
//! The instance field stores `instance + 1`, which is what makes [`VIS_EMPTY`]
//! (zero) unreachable by a real fragment and therefore usable as the cleared
//! value. Spending one instance rather than one *packed word* is the cheaper
//! side of that trade: the alternative — clearing to `u32::MAX` — costs the
//! all-ones combination, which is a legal triangle of a legal meshlet of a legal
//! instance and would have to be refused at pack time, inside the hot loop.
//!
//! # What happens when a scene exceeds them
//!
//! [`VisPacking::admit`] is a **typed refusal at registration** (the P27.1
//! grid-overflow law): a scene with more instances than the field addresses does
//! not pack a wrong id, it fails to enter the VisBuffer path at all and the
//! frame renders through the forward meshlet raster — which P28.1 keeps
//! precisely so that there is somewhere to fall back *to*. The refusal is a pure
//! function of the committed scene and the streamer's residency, so two runs of
//! one scripted path refuse on the same frames.
//!
//! # The alternative that was measured and not taken
//!
//! A **frame-derived** split — `meshlet_bits = ceil(log2(resident_slots))`, the
//! remaining 25 − that going to instances — is strictly more capacious: the
//! `vgeom-demo`'s 500 slots would leave 16 bits for instances (65 535 of them)
//! instead of 11. It is not taken here because it makes the layout a per-frame
//! value rather than a constant, which costs the shaders a uniform read per
//! unpack, costs the arms below their ability to be a *compile-time* mirror, and
//! moves the refusal from registration into the frame. Recorded in
//! `docs/memos/p28-1-visbuffer.md` §2 with its arithmetic, and routed to P28.3,
//! where one streamer decides pool capacities and therefore knows the answer
//! before a frame starts.

use glam::{Mat3, Vec2, Vec3, Vec4};

// ── the split ────────────────────────────────────────────────────────────────

/// Bits the **triangle index within its meshlet** occupies (LSB-most field).
pub const VIS_TRI_BITS: u32 = 7;
/// Bits the **shared-pool meshlet slot** occupies.
pub const VIS_MESHLET_BITS: u32 = 14;
/// Bits the **biased instance index** occupies (MSB-most field).
pub const VIS_INSTANCE_BITS: u32 = 11;

const _: () = assert!(VIS_TRI_BITS + VIS_MESHLET_BITS + VIS_INSTANCE_BITS == 32);

/// First bit of the meshlet field.
pub const VIS_MESHLET_SHIFT: u32 = VIS_TRI_BITS;
/// First bit of the instance field.
pub const VIS_INSTANCE_SHIFT: u32 = VIS_TRI_BITS + VIS_MESHLET_BITS;

/// The cleared value: **no geometry at this pixel**. Unreachable by
/// [`VisPacking::pack`] because the instance field is stored biased by one.
pub const VIS_EMPTY: u32 = 0;

/// Triangles a meshlet may hold and still be addressable — `1 << VIS_TRI_BITS`.
pub const VIS_MAX_TRIANGLES_PER_MESHLET: u32 = 1 << VIS_TRI_BITS;
/// Shared-pool meshlet slots the field addresses.
pub const VIS_MAX_MESHLET_SLOTS: u32 = 1 << VIS_MESHLET_BITS;
/// Virtualized-geometry instances a frame may hold. One short of the field's
/// range: the bias that makes [`VIS_EMPTY`] unreachable costs exactly this.
pub const VIS_MAX_INSTANCES: u32 = (1 << VIS_INSTANCE_BITS) - 1;

// ── the refusal ──────────────────────────────────────────────────────────────

/// Why a scene cannot enter the VisBuffer path.
///
/// Every variant names the measured value **and** the ceiling, because a refusal
/// that says only "too many" cannot be acted on — the host has to know whether it
/// is over by one instance or by a factor of ten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisPackError {
    /// More virtualized-geometry instances in the frame than the instance field
    /// addresses.
    Instances { count: u32, ceiling: u32 },
    /// The shared meshlet pool holds more slots than the meshlet field
    /// addresses. Measured off the pool's **capacity**, not its occupancy: a slot
    /// that exists can be named by a resident page, and residency moves inside a
    /// frame.
    MeshletSlots { count: u32, ceiling: u32 },
    /// An asset's meshlets hold more triangles than the triangle field
    /// addresses. Read off the `.inf_vmesh` header's whole-mesh maximum, which is
    /// the same value the indirect draw's vertex count is derived from.
    TrianglesPerMeshlet { max_tri: u32, ceiling: u32 },
}

impl std::fmt::Display for VisPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Instances { count, ceiling } => write!(
                f,
                "the visibility buffer addresses {ceiling} virtualized-geometry \
                 instances and this frame holds {count}"
            ),
            Self::MeshletSlots { count, ceiling } => write!(
                f,
                "the visibility buffer addresses {ceiling} meshlet pool slots and \
                 the pool holds {count}"
            ),
            Self::TrianglesPerMeshlet { max_tri, ceiling } => write!(
                f,
                "the visibility buffer addresses {ceiling} triangles a meshlet and \
                 an asset cooks {max_tri}"
            ),
        }
    }
}

impl std::error::Error for VisPackError {}

/// What one VisBuffer texel names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisFragment {
    /// Index into the frame's **flat** instance table (all vgeom assets
    /// concatenated in the deterministic asset order the pass walks).
    pub instance: u32,
    /// Slot in the **shared** meshlet pool — already through the per-asset
    /// remap, so the resolve needs no asset identity to find the geometry.
    pub meshlet: u32,
    /// Triangle within that meshlet, `0..meshlet.triangle_count`.
    pub tri: u32,
}

/// The packing itself — a unit struct rather than free functions so the contract
/// has one name to cite, and so `admit` reads as the door it is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VisPacking;

impl VisPacking {
    /// Refuse a frame whose vgeom set the packing cannot address.
    ///
    /// `meshlet_slots` is the shared pool's capacity in records and `max_tri` the
    /// largest whole-mesh triangle-per-meshlet maximum over the frame's assets.
    /// Checked in field order (triangle, meshlet, instance) so a scene that
    /// breaks two ceilings reports the same one on every run.
    pub fn admit(instances: u32, meshlet_slots: u32, max_tri: u32) -> Result<(), VisPackError> {
        if max_tri > VIS_MAX_TRIANGLES_PER_MESHLET {
            return Err(VisPackError::TrianglesPerMeshlet {
                max_tri,
                ceiling: VIS_MAX_TRIANGLES_PER_MESHLET,
            });
        }
        if meshlet_slots > VIS_MAX_MESHLET_SLOTS {
            return Err(VisPackError::MeshletSlots {
                count: meshlet_slots,
                ceiling: VIS_MAX_MESHLET_SLOTS,
            });
        }
        if instances > VIS_MAX_INSTANCES {
            return Err(VisPackError::Instances {
                count: instances,
                ceiling: VIS_MAX_INSTANCES,
            });
        }
        Ok(())
    }

    /// Pack one fragment id. `None` when any field is out of range — the caller
    /// is [`admit`](Self::admit)'s downstream, so this returning `None` in the
    /// renderer means the admission door was bypassed.
    pub fn pack(f: VisFragment) -> Option<u32> {
        if f.instance >= VIS_MAX_INSTANCES
            || f.meshlet >= VIS_MAX_MESHLET_SLOTS
            || f.tri >= VIS_MAX_TRIANGLES_PER_MESHLET
        {
            return None;
        }
        Some(((f.instance + 1) << VIS_INSTANCE_SHIFT) | (f.meshlet << VIS_MESHLET_SHIFT) | f.tri)
    }

    /// Unpack one texel. `None` for [`VIS_EMPTY`] — the cleared value, which the
    /// bias makes unreachable from [`pack`](Self::pack).
    pub fn unpack(id: u32) -> Option<VisFragment> {
        let biased = id >> VIS_INSTANCE_SHIFT;
        if biased == 0 {
            return None;
        }
        Some(VisFragment {
            instance: biased - 1,
            meshlet: (id >> VIS_MESHLET_SHIFT) & (VIS_MAX_MESHLET_SLOTS - 1),
            tri: id & (VIS_MAX_TRIANGLES_PER_MESHLET - 1),
        })
    }
}

// ── barycentrics + analytic gradients ────────────────────────────────────────

/// A pixel's position on its triangle, with the **exact** screen derivatives of
/// the perspective-correct barycentrics.
///
/// [`VisBary::of`] is the Rust twin of `vis_resolve.wgsl`'s `vis_bary`, and it is
/// here so the construction can be *stated* in a language that has tests: the
/// arms below assert the gradient is the **limit of a central difference** at a
/// second-order signature, with an inline control that drops the `2/width` pixel
/// step and must disagree by 400×.
///
/// **What it is NOT, stated because the first draft of this block claimed
/// otherwise** (P28.1 audit): it is not compared against the device. There is no
/// arm that feeds the twin the vertices and the pixel the GPU used and checks the
/// numbers the GPU produced — the visibility id names a slot in the *shared*
/// meshlet pool and the slot→asset remap lives on the GPU, so a host cannot
/// decode a readback id back to three vertices without rebuilding the streamer's
/// residency. So this is a **mirror, not a measurement** (the P24 two-hosts law),
/// and the device's gradients are witnessed only indirectly, through
/// `parity_textured_virtual_texture` — the one row whose shading a derivative can
/// reach at all. Closing that is P28.3's, where one streamer owns the remap and a
/// host can name a slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisBary {
    /// Perspective-correct barycentric weights, summing to 1.
    pub lambda: Vec3,
    /// ∂λ/∂x, per **pixel** across the screen.
    pub d_dx: Vec3,
    /// ∂λ/∂y, per **pixel** down the screen.
    pub d_dy: Vec3,
}

impl VisBary {
    /// Barycentrics and their screen gradients for the pixel whose NDC position
    /// is `ndc`, on the triangle whose **clip-space** vertices are `c0..c2`.
    ///
    /// # The construction, and why it is not a derivative
    ///
    /// A fragment shader gets `dpdx` by differencing against its neighbours in a
    /// 2 × 2 quad, which a resolve pass cannot use: the pixel next door is a
    /// *different triangle*, so the difference is a step across a silhouette and
    /// the mip it justifies is nonsense. So this is solved rather than sampled.
    ///
    /// Write `M = [c0.xyw | c1.xyw | c2.xyw]` (columns). A point on the triangle
    /// projects to `ndc` exactly when `M · μ ∝ (ndc.x, ndc.y, 1)` for some
    /// positive scale, so `μ = M⁻¹ · (ndc.x, ndc.y, 1)` and the perspective-
    /// correct weights are `λ = μ / Σμ`. Because `μ` is **linear** in
    /// `(ndc.x, ndc.y, 1)`, its NDC gradients are two *columns of `M⁻¹`* — no
    /// differencing anywhere — and `λ`'s follow by the quotient rule. The
    /// per-pixel step is `2/width` across and `-2/height` down, the reverse-Y of
    /// NDC against pixel coordinates.
    ///
    /// A degenerate triangle (`|det M|` at the floor) yields `λ = (1, 0, 0)` and
    /// zero gradients — the first vertex's attributes, flat. That is the same
    /// direction `vt_apply_normal` fails in (its own `abs(det) < 1e-12` returns
    /// the geometric normal): a degenerate case resolves to *something local*
    /// rather than to a NaN that would poison a whole tile of the atlas.
    ///
    /// # What the floor does and does not buy, measured (P28.1 audit)
    ///
    /// The guard carries an explicit `|| det.is_nan()` (`det != det` in the two
    /// shaders) because every comparison against NaN is **false**: `|det| < ε`
    /// alone lets a NaN vertex walk through the guard, through the solve and into
    /// the frame. Measured, before this clause: one NaN clip coordinate produced
    /// `λ = (NaN, NaN, NaN)` and non-finite gradients, from a function whose own
    /// doc said a degenerate case resolves "rather than to a NaN".
    ///
    /// The floor is **not** a conditioning bound and does not pretend to be.
    /// Measured on a sliver whose two edges are `ε` apart: at `ε = 1e-3` the
    /// gradient is already 6.2 per pixel, at `1e-5` it is 33.6 and `λ` is
    /// `(-0.40, 0.80, 0.60)` — finite, outside the triangle, meaningless. What
    /// the floor guarantees is only that nothing **non-finite** leaves here; a
    /// near-degenerate triangle leaves a garbage gradient, which `vt_mip`'s
    /// `min(…, mip_count - 1)` turns into the coarsest level of the pyramid. That
    /// is the safe direction (a blurry texel, not a NaN tile) and it is a
    /// property of the consumer rather than of this function.
    pub fn of(c0: Vec4, c1: Vec4, c2: Vec4, ndc: Vec2, width: f32, height: f32) -> Self {
        let m = Mat3::from_cols(
            Vec3::new(c0.x, c0.y, c0.w),
            Vec3::new(c1.x, c1.y, c1.w),
            Vec3::new(c2.x, c2.y, c2.w),
        );
        let det = m.determinant();
        if det.abs() < VIS_DET_EPS || det.is_nan() {
            return Self {
                lambda: Vec3::X,
                d_dx: Vec3::ZERO,
                d_dy: Vec3::ZERO,
            };
        }
        let inv = m.inverse();
        let mu = inv * Vec3::new(ndc.x, ndc.y, 1.0);
        // `inv * (1,0,0)` and `inv * (0,1,0)` — the first two columns, spelled as
        // the products they are so the mirror in WGSL reads the same way.
        let dmu_du = inv * Vec3::X;
        let dmu_dv = inv * Vec3::Y;
        let d = mu.x + mu.y + mu.z;
        if d.abs() < VIS_DET_EPS || d.is_nan() {
            return Self {
                lambda: Vec3::X,
                d_dx: Vec3::ZERO,
                d_dy: Vec3::ZERO,
            };
        }
        let inv_d = 1.0 / d;
        let lambda = mu * inv_d;
        let quotient = |dmu: Vec3| {
            let ds: f32 = dmu.x + dmu.y + dmu.z;
            (dmu - lambda * ds) * inv_d
        };
        // NDC per pixel: +2/w across, -2/h down (NDC y points up).
        Self {
            lambda,
            d_dx: quotient(dmu_du) * (2.0 / width.max(1.0)),
            d_dy: quotient(dmu_dv) * (-2.0 / height.max(1.0)),
        }
    }

    /// Interpolate a scalar attribute and its two screen gradients.
    pub fn attr(&self, a0: f32, a1: f32, a2: f32) -> (f32, f32, f32) {
        let a = Vec3::new(a0, a1, a2);
        (self.lambda.dot(a), self.d_dx.dot(a), self.d_dy.dot(a))
    }
}

/// The determinant floor below which a triangle is treated as degenerate. Shared
/// with `vis_resolve.wgsl`'s `VIS_DET_EPS` and pinned against it.
pub const VIS_DET_EPS: f32 = 1e-20;

#[cfg(test)]
mod tests {
    use super::*;

    // ── the packing round-trips ──────────────────────────────────────────────

    #[test]
    fn every_field_round_trips_at_both_ends_of_its_range() {
        let corners = [
            (0, 0, 0),
            (
                VIS_MAX_INSTANCES - 1,
                VIS_MAX_MESHLET_SLOTS - 1,
                VIS_MAX_TRIANGLES_PER_MESHLET - 1,
            ),
            (0, VIS_MAX_MESHLET_SLOTS - 1, 0),
            (VIS_MAX_INSTANCES - 1, 0, 0),
            (0, 0, VIS_MAX_TRIANGLES_PER_MESHLET - 1),
            (1, 1, 1),
            (7, 9_999, 123),
        ];
        for (instance, meshlet, tri) in corners {
            let f = VisFragment {
                instance,
                meshlet,
                tri,
            };
            let id = VisPacking::pack(f).unwrap_or_else(|| panic!("{f:?} must pack"));
            assert_ne!(id, VIS_EMPTY, "{f:?} packed to the cleared value");
            assert_eq!(VisPacking::unpack(id), Some(f), "{f:?} did not round-trip");
        }
    }

    /// The exhaustive half the corners cannot give: every id in a dense sweep of
    /// the two small fields, against every instance, unpacks to what packed it.
    /// 2 047 × 64 × 8 is 1 048 064 round trips and runs in well under a second.
    #[test]
    fn the_packing_is_injective_over_a_dense_sweep() {
        let mut seen_empty = 0u32;
        for instance in 0..VIS_MAX_INSTANCES {
            for meshlet in (0..VIS_MAX_MESHLET_SLOTS).step_by(256) {
                for tri in (0..VIS_MAX_TRIANGLES_PER_MESHLET).step_by(16) {
                    let f = VisFragment {
                        instance,
                        meshlet,
                        tri,
                    };
                    let id = VisPacking::pack(f).unwrap();
                    if id == VIS_EMPTY {
                        seen_empty += 1;
                    }
                    assert_eq!(VisPacking::unpack(id), Some(f));
                }
            }
        }
        assert_eq!(
            seen_empty, 0,
            "the cleared value is reachable from a real fragment, so an empty \
             pixel and a real one are the same word"
        );
    }

    #[test]
    fn the_cleared_value_unpacks_to_nothing_and_nothing_packs_to_it() {
        assert_eq!(VisPacking::unpack(VIS_EMPTY), None);
        // The whole instance field at zero is the only way to reach it, and the
        // bias forbids that — spelled as a sweep over the other two fields so
        // the claim is not just the one word.
        for meshlet in [0, 1, VIS_MAX_MESHLET_SLOTS - 1] {
            for tri in [0, 1, VIS_MAX_TRIANGLES_PER_MESHLET - 1] {
                let id = VisPacking::pack(VisFragment {
                    instance: 0,
                    meshlet,
                    tri,
                })
                .unwrap();
                assert_ne!(id, VIS_EMPTY);
                assert!(
                    id >> VIS_INSTANCE_SHIFT >= 1,
                    "instance 0 must store as 1, or the empty sentinel is a lie"
                );
            }
        }
    }

    #[test]
    fn a_field_past_its_ceiling_refuses_to_pack() {
        assert_eq!(
            VisPacking::pack(VisFragment {
                instance: VIS_MAX_INSTANCES,
                meshlet: 0,
                tri: 0
            }),
            None
        );
        assert_eq!(
            VisPacking::pack(VisFragment {
                instance: 0,
                meshlet: VIS_MAX_MESHLET_SLOTS,
                tri: 0
            }),
            None
        );
        assert_eq!(
            VisPacking::pack(VisFragment {
                instance: 0,
                meshlet: 0,
                tri: VIS_MAX_TRIANGLES_PER_MESHLET
            }),
            None
        );
    }

    // ── the refusal is typed, ordered, and names both numbers ────────────────

    #[test]
    fn admission_refuses_each_ceiling_by_name_and_admits_the_flagship() {
        // The `vgeom-demo`'s own shape: 324 instances (18 × 18), ~500 pool slots
        // for the whole DAG, 124 triangles a meshlet.
        assert_eq!(VisPacking::admit(324, 512, 124), Ok(()));
        // Exactly at every ceiling is still admitted.
        assert_eq!(
            VisPacking::admit(
                VIS_MAX_INSTANCES,
                VIS_MAX_MESHLET_SLOTS,
                VIS_MAX_TRIANGLES_PER_MESHLET
            ),
            Ok(())
        );
        assert_eq!(
            VisPacking::admit(VIS_MAX_INSTANCES + 1, 0, 0),
            Err(VisPackError::Instances {
                count: VIS_MAX_INSTANCES + 1,
                ceiling: VIS_MAX_INSTANCES
            })
        );
        assert_eq!(
            VisPacking::admit(0, VIS_MAX_MESHLET_SLOTS + 1, 0),
            Err(VisPackError::MeshletSlots {
                count: VIS_MAX_MESHLET_SLOTS + 1,
                ceiling: VIS_MAX_MESHLET_SLOTS
            })
        );
        // meshopt's own hard cap, which a cooked asset may legally carry.
        assert_eq!(
            VisPacking::admit(0, 0, 512),
            Err(VisPackError::TrianglesPerMeshlet {
                max_tri: 512,
                ceiling: VIS_MAX_TRIANGLES_PER_MESHLET
            })
        );
    }

    #[test]
    fn a_scene_over_two_ceilings_reports_the_same_one_every_run() {
        let over = VisPacking::admit(u32::MAX, u32::MAX, u32::MAX);
        for _ in 0..8 {
            assert_eq!(over, VisPacking::admit(u32::MAX, u32::MAX, u32::MAX));
        }
        assert!(matches!(
            over,
            Err(VisPackError::TrianglesPerMeshlet { .. })
        ));
    }

    #[test]
    fn every_refusal_prints_the_measurement_and_the_ceiling() {
        for e in [
            VisPackError::Instances {
                count: 3000,
                ceiling: VIS_MAX_INSTANCES,
            },
            VisPackError::MeshletSlots {
                count: 20_000,
                ceiling: VIS_MAX_MESHLET_SLOTS,
            },
            VisPackError::TrianglesPerMeshlet {
                max_tri: 512,
                ceiling: VIS_MAX_TRIANGLES_PER_MESHLET,
            },
        ] {
            let s = e.to_string();
            let (measured, ceiling) = match e {
                VisPackError::Instances { count, ceiling } => (count, ceiling),
                VisPackError::MeshletSlots { count, ceiling } => (count, ceiling),
                VisPackError::TrianglesPerMeshlet { max_tri, ceiling } => (max_tri, ceiling),
            };
            assert!(
                s.contains(&measured.to_string()) && s.contains(&ceiling.to_string()),
                "a refusal that does not print both numbers cannot be acted on: {s}"
            );
        }
    }

    // ── the shaders read the same split ──────────────────────────────────────

    /// The three-readers guard. A shifted field is not a validation error, so
    /// the only thing that can catch it is a comparison against the source.
    #[test]
    fn the_shaders_bit_split_is_the_rusts() {
        // `lines()` + `trim_end_matches('\r')`: every `.wgsl` in the working tree
        // is CRLF (the P27.1 observation), so a pattern with a newline in it
        // matches nothing.
        let want = [
            format!("const VIS_TRI_BITS: u32 = {VIS_TRI_BITS}u;"),
            format!("const VIS_MESHLET_BITS: u32 = {VIS_MESHLET_BITS}u;"),
            format!("const VIS_INSTANCE_BITS: u32 = {VIS_INSTANCE_BITS}u;"),
            format!("const VIS_EMPTY: u32 = {VIS_EMPTY}u;"),
        ];
        for (name, src) in [
            ("visbuffer.wgsl", include_str!("shaders/visbuffer.wgsl")),
            ("vis_resolve.wgsl", include_str!("shaders/vis_resolve.wgsl")),
            (
                "vis_feedback.wgsl",
                include_str!("shaders/vis_feedback.wgsl"),
            ),
        ] {
            let code: Vec<&str> = src
                .lines()
                .map(|l| l.trim_end_matches('\r').trim())
                .filter(|l| !l.starts_with("//"))
                .collect();
            for w in &want {
                assert!(
                    code.iter().any(|l| l.starts_with(w.as_str())),
                    "{name} does not declare `{w}` — three readers of one layout, \
                     and one of them has drifted"
                );
            }
        }
    }

    /// **The structural reason a voxel surface is still not shaded here**
    /// (P28.1's correction to P27.5's routing).
    ///
    /// P27.5 refused `voxel.wgsl` a shadow receiver and routed the fix to "the
    /// VisBuffer resolve, where meshlet and voxel surfaces are shaded through one
    /// material pass". Building the pass showed the sentence to be wrong: the
    /// meshlet field is a slot in the **shared meshlet pool**, and a voxel chunk
    /// — a Surface-Nets mesh in its own buffers, with no meshlet structure, no
    /// DAG and no page — has none. Naming a second geometry kind would have to be
    /// paid for out of one of the three fields, and all thirty-two bits are
    /// spent.
    ///
    /// This arm is the falsifier: it fails the day the id space grows, which is
    /// the day the voxel door genuinely opens, and it fails if any field is
    /// widened at another's expense without the routing being revisited.
    #[test]
    fn the_visbuffer_id_space_has_no_room_for_a_second_geometry_kind() {
        assert_eq!(
            VIS_TRI_BITS + VIS_MESHLET_BITS + VIS_INSTANCE_BITS,
            32,
            "the packing no longer spends exactly one R32Uint; voxel.wgsl's \
             refusal cites this arithmetic as the reason it has no door, and \
             that reason has just changed"
        );
        // …and the refusal's own text, so the two cannot drift apart. Read over
        // `lines()` because every `.wgsl` in the working tree is CRLF.
        let voxel: Vec<&str> = include_str!("shaders/voxel.wgsl")
            .lines()
            .map(|l| l.trim_end_matches('\r'))
            .collect();
        for want in [
            "P28.1: THE ROUTING WAS WRONG, AND HERE IS THE MEASUREMENT",
            "the_visbuffer_id_space_has_no_room_for_a_second_geometry_kind",
            "meshletize voxel chunks",
        ] {
            assert!(
                voxel.iter().any(|l| l.contains(want)),
                "voxel.wgsl's P28.1 correction lost `{want}` — a routing that \
                 summarises instead of naming its blocker is the P27.5 finding, \
                 one phase later"
            );
        }
    }

    /// **The parity oracle's independence, pinned** (P28.1 audit).
    ///
    /// `tests/visbuffer_parity.rs` rests entirely on the forward meshlet path
    /// being a SECOND derivation of the same frame — the P27.5 law that a gate
    /// cannot see an error the subsystems share. That property was stated in a
    /// comment and checked by nothing, which is the same shape as a cited gate
    /// that does not exist.
    ///
    /// So: every piece of shading arithmetic the two paths agree about must be
    /// **written twice**. The day someone de-duplicates `shade_light` into the
    /// shared lit prelude — an obviously good refactor, and the one that silently
    /// retires the whole nucleus — one of the two files stops declaring it (it
    /// must; a second declaration of a composed symbol does not compile) and this
    /// arm fails.
    ///
    /// # The boundary this arm does NOT extend past, measured
    ///
    /// Both files are `ShaderKind::Lit(2)`, so both are composed with the same
    /// prelude: `vt_sample`, `vsm_receive`, the environment lighting, the
    /// atmosphere, the wetness. Those functions ARE shared, deliberately (the
    /// resolve must reach the shadow atlas through the one declaration
    /// `vsm_receive` owns), and an error inside them is invisible to every parity
    /// arm. Mutation-measured: adding `+ 1.0` to `vt_sample.wgsl`'s `vt_mip` —
    /// a wrong mip for the whole engine — leaves all twelve arms green, because
    /// both paths shift a level together. The nucleus is a gate on the
    /// *reconstruction*, not on the shading library, and this arm marks exactly
    /// where the one stops and the other begins.
    #[test]
    fn the_resolve_derives_its_own_shading_rather_than_borrowing_the_forward_paths() {
        let resolve = include_str!("shaders/vis_resolve.wgsl");
        let forward = include_str!("shaders/vgeom_mesh.wgsl");
        // The BRDF and the vertex pull: the arithmetic a parity claim is about.
        for f in [
            "shade_light",
            "distribution_ggx",
            "geometry_smith",
            "fresnel_schlick",
            "point_attenuation",
            "read_tri_byte",
        ] {
            let decl = format!("fn {f}(");
            for (name, src) in [("vis_resolve.wgsl", resolve), ("vgeom_mesh.wgsl", forward)] {
                assert!(
                    src.lines()
                        .map(|l| l.trim_end_matches('\r'))
                        .any(|l| l.starts_with(&decl)),
                    "{name} no longer declares `{f}` itself. If it has moved into \
                     the shared lit prelude then the two paths are one program, \
                     and `tests/visbuffer_parity.rs` compares it with itself — \
                     which is the P27.5 law, and the ONLY thing standing between \
                     that suite and being a mirror."
                );
            }
        }
        // …and no CODE line of the resolve reaches for the forward path's file.
        // Comments may cite it (one does, about branch order) — a citation is
        // documentation; an `include` would be the coupling.
        assert!(
            !resolve
                .lines()
                .map(|l| l.trim_end_matches('\r').trim())
                .any(|l| !l.starts_with("//") && l.contains("vgeom_mesh")),
            "a code line of vis_resolve.wgsl names vgeom_mesh.wgsl"
        );
        // The reconstruction the forward path does NOT have, and must not gain:
        // it interpolates through the rasterizer, so a `vis_bary` in there would
        // mean the oracle solved the same system the subject does.
        assert!(
            !forward.lines().any(|l| l.starts_with("fn vis_bary(")),
            "vgeom_mesh.wgsl now solves for barycentrics too — the forward path's \
             independence IS that it gets them from the rasterizer's \
             interpolators"
        );
    }

    /// **The per-pixel mark asks for the level the sampler reads** (P28.1 audit).
    ///
    /// `vis_feedback.wgsl`'s header cited an arm of this name and the arm did not
    /// exist — the P20 law, three files over from where this batch caught it in
    /// P27.5's. This is it: the LOD rule is written twice, in `vt_sample.wgsl`
    /// (what a sample reads) and in `vis_feedback.wgsl` (what the streamer is
    /// asked for), and a feedback pass that chased a different level than the
    /// sampler reads is a streamer paging tiles nothing samples — which presents
    /// as a permanent backlog with every counter green.
    #[test]
    fn the_per_pixel_mark_uses_the_same_mip_rule() {
        // The three lines that ARE the rule: the footprint in texels, the larger
        // axis, and the level it justifies. Everything else in the two functions
        // is table addressing.
        let core = [
            "let dx = ddx * size;",
            "let dy = ddy * size;",
            "let rho = max(length(dx), length(dy));",
            "let lod = max(log2(max(rho, 1e-8)), 0.0);",
        ];
        for (name, src) in [
            ("vt_sample.wgsl", include_str!("shaders/vt_sample.wgsl")),
            (
                "vis_feedback.wgsl",
                include_str!("shaders/vis_feedback.wgsl"),
            ),
        ] {
            let code: Vec<&str> = src
                .lines()
                .map(|l| l.trim_end_matches('\r').trim())
                .collect();
            for want in core {
                assert!(
                    code.contains(&want),
                    "{name} no longer computes the mip as `{want}` — the producer \
                     and the sampler have to name one level or the streamer is \
                     paging tiles nothing reads"
                );
            }
            // Both clamp to the pyramid's top rather than trusting `lod`.
            assert!(
                code.iter().any(|l| l.starts_with("return min(u32(lod),")),
                "{name} no longer clamps the level to the pyramid"
            );
        }
    }

    /// The determinant floor is one number in two languages too.
    #[test]
    fn the_degenerate_floor_is_one_number() {
        let src = include_str!("shaders/vis_resolve.wgsl");
        assert!(
            src.lines()
                .map(|l| l.trim_end_matches('\r').trim())
                .any(|l| l.starts_with("const VIS_DET_EPS: f32 = 1e-20;")),
            "vis_resolve.wgsl's degenerate floor is not the Rust's {VIS_DET_EPS}"
        );
    }

    // ── the barycentric construction ─────────────────────────────────────────

    /// A clip-space triangle built from a real perspective projection, sampled at
    /// its own vertices: each must recover a unit weight on itself.
    #[test]
    fn a_vertexs_own_pixel_recovers_its_own_barycentric() {
        let (c0, c1, c2) = fixture_triangle();
        for (k, c) in [c0, c1, c2].iter().enumerate() {
            let ndc = Vec2::new(c.x / c.w, c.y / c.w);
            let b = VisBary::of(c0, c1, c2, ndc, 800.0, 600.0);
            let mut want = Vec3::ZERO;
            want[k] = 1.0;
            assert!(
                (b.lambda - want).abs().max_element() < 1e-4,
                "vertex {k} recovered {:?}, wanted {want:?}",
                b.lambda
            );
        }
    }

    #[test]
    fn the_weights_sum_to_one_across_the_triangle() {
        let (c0, c1, c2) = fixture_triangle();
        for i in 0..9 {
            for j in 0..9 {
                let ndc = Vec2::new(-0.4 + 0.1 * i as f32, -0.4 + 0.1 * j as f32);
                let b = VisBary::of(c0, c1, c2, ndc, 800.0, 600.0);
                assert!(
                    (b.lambda.element_sum() - 1.0).abs() < 1e-5,
                    "weights at {ndc:?} sum to {}",
                    b.lambda.element_sum()
                );
                // The gradient of a constant is zero: the three derivative
                // components must cancel, which is the differential form of the
                // assertion above and catches a quotient rule that dropped a term.
                assert!(
                    b.d_dx.element_sum().abs() < 1e-6 && b.d_dy.element_sum().abs() < 1e-6,
                    "the weights sum to 1 everywhere, so their gradients must sum \
                     to 0: {:?} / {:?}",
                    b.d_dx,
                    b.d_dy
                );
            }
        }
    }

    /// The analytic gradient against a **central finite difference** of the same
    /// construction — the check the fragment stage's `dpdx` would have been.
    ///
    /// It is asserted as a **convergence order**, not as a tolerance, and the
    /// difference matters. A perspective-correct barycentric is a rational
    /// function of the pixel, so a central difference over a step `h` is exact
    /// only in the limit and carries an `O(h²)` term everywhere else — at a full
    /// pixel that term is real (measured: 5.2 × 10⁻⁵ on this fixture). Any
    /// tolerance loose enough to admit it is loose enough to admit a *wrong*
    /// gradient of the same magnitude. So the arm halves the step and asserts the
    /// error falls by four: second order is a property only the true derivative
    /// has, and a formula off by a constant factor — the classic mistake, a
    /// missing `2/width` — converges to the wrong number and fails immediately.
    #[test]
    fn the_analytic_gradient_is_the_limit_of_the_finite_difference() {
        let (c0, c1, c2) = fixture_triangle();
        let (w, h) = (800.0f32, 600.0f32);
        let at = |px: f32, py: f32| {
            let ndc = Vec2::new(2.0 * px / w - 1.0, 1.0 - 2.0 * py / h);
            VisBary::of(c0, c1, c2, ndc, w, h).lambda
        };
        // **Sampled INSIDE the triangle, by construction.** The first draft of
        // this arm sampled a grid of pixels and failed at 5.2 × 10⁻⁵ with a
        // convergence ratio of 2.1, which reads as a defect and is not one: half
        // those pixels lie outside the triangle, where the weights are large and
        // `D = Σμ` approaches its zero line, so an f32 relative error of 10⁻⁶ on a
        // weight of magnitude 50 is an absolute error of 5 × 10⁻⁵. The same
        // arithmetic in f64 truncates at 2.8 × 10⁻¹¹ and converges at exactly 4×,
        // which is what says the formula was right and the *sampling* was wrong.
        // A resolve pass only ever evaluates a pixel its own id says is covered,
        // so the arm evaluates where the shader does: interior points, reached by
        // building a clip-space point from weights and projecting it back.
        let interior_px = |wts: [f32; 3]| {
            let cp = c0 * wts[0] + c1 * wts[1] + c2 * wts[2];
            let ndc = Vec2::new(cp.x / cp.w, cp.y / cp.w);
            ((ndc.x + 1.0) * w * 0.5, (1.0 - ndc.y) * h * 0.5)
        };
        let samples: Vec<(f32, f32)> = [
            [0.60, 0.25, 0.15],
            [0.34, 0.33, 0.33],
            [0.15, 0.70, 0.15],
            [0.20, 0.20, 0.60],
        ]
        .into_iter()
        .map(interior_px)
        .collect();
        // f64 accumulation: the quantity under test is the *difference* of two
        // f32 gradients.
        let worst_at = |step: f32| {
            let mut worst = 0.0f64;
            for &(px, py) in &samples {
                let b = VisBary::of(
                    c0,
                    c1,
                    c2,
                    Vec2::new(2.0 * px / w - 1.0, 1.0 - 2.0 * py / h),
                    w,
                    h,
                );
                assert!(
                    b.lambda.min_element() > -1e-3 && b.lambda.max_element() < 1.001,
                    "the sample at ({px}, {py}) is not inside the triangle: {:?}",
                    b.lambda
                );
                let fd_x = (at(px + step * 0.5, py) - at(px - step * 0.5, py)) / step;
                let fd_y = (at(px, py + step * 0.5) - at(px, py - step * 0.5)) / step;
                worst = worst
                    .max((b.d_dx - fd_x).abs().max_element() as f64)
                    .max((b.d_dy - fd_y).abs().max_element() as f64);
            }
            worst
        };
        let coarse = worst_at(1.0);
        let fine = worst_at(0.5);
        assert!(
            coarse > 0.0,
            "a central difference of a perspective-correct weight is second-order \
             accurate and NOT exact; a zero here means the test compared \
             something with itself"
        );
        // **The residual is f32 ROUNDING, not truncation, and the arm asserts
        // that shape.** The same construction in f64 truncates at 2.8 × 10⁻¹¹ and
        // falls by exactly 4× per halving; in f32 the truncation term is four
        // orders below the epsilon of the weights it is subtracted from, so what
        // is left is `eps / step` — an error that *grows* when the step shrinks.
        // Measured: 1.4 × 10⁻⁷ at a whole pixel, 2.8 × 10⁻⁷ at half. So the claim
        // "these two agree to f32" is asserted as a **1/step** signature, which a
        // wrong formula cannot produce (see the control below) and a right one
        // cannot avoid.
        assert!(
            coarse < 1e-6,
            "the whole-pixel disagreement is {coarse:.3e}; the f32 floor on this \
             fixture is ~1.4e-7, so this is a real difference rather than rounding"
        );
        let ratio = coarse / fine;
        assert!(
            (0.3..=0.8).contains(&ratio),
            "halving the step changed the disagreement by {ratio:.2}x. Rounding \
             gives ~0.5 (the error is eps/step) and a *wrong* gradient gives ~1.0 \
             (a constant bias the step cannot touch) — this is neither \
             (coarse {coarse:.3e}, fine {fine:.3e})"
        );
        // THE CONTROL, inline, because an agreement test with no disagreement
        // case is a test that cannot fail: drop the `2/width` pixel step — the
        // one mistake this construction invites — and watch both the magnitude
        // and the signature change.
        let wrong_at = |step: f32| {
            let mut worst = 0.0f64;
            for &(px, py) in &samples {
                let b = VisBary::of(
                    c0,
                    c1,
                    c2,
                    Vec2::new(2.0 * px / w - 1.0, 1.0 - 2.0 * py / h),
                    w,
                    h,
                );
                let wrong_dx = b.d_dx * (w * 0.5); // as if ndc == pixels
                let fd_x = (at(px + step * 0.5, py) - at(px - step * 0.5, py)) / step;
                worst = worst.max((wrong_dx - fd_x).abs().max_element() as f64);
            }
            worst
        };
        let wrong_coarse = wrong_at(1.0);
        let wrong_ratio = wrong_coarse / wrong_at(0.5);
        assert!(
            wrong_coarse > 1e-4 && (0.9..=1.1).contains(&wrong_ratio),
            "the viewport-step control did not behave like a wrong formula \
             ({wrong_coarse:.3e}, ratio {wrong_ratio:.2}) — if a gradient off by \
             400x passes this arm, the arm is measuring nothing"
        );
    }

    #[test]
    fn a_degenerate_triangle_resolves_flat_rather_than_to_a_nan() {
        let c = Vec4::new(0.1, 0.2, 0.5, 1.0);
        let b = VisBary::of(c, c, c, Vec2::ZERO, 800.0, 600.0);
        assert_eq!(b.lambda, Vec3::X);
        assert_eq!(b.d_dx, Vec3::ZERO);
        assert_eq!(b.d_dy, Vec3::ZERO);
        assert!(b.lambda.is_finite() && b.d_dx.is_finite());
    }

    /// **A NaN vertex takes the degenerate branch**, and it did not before
    /// (P28.1 audit).
    ///
    /// Every comparison against NaN is false, so the guard's natural spelling —
    /// `det.abs() < VIS_DET_EPS` — is NaN-*blind*: measured, one NaN clip
    /// coordinate produced `λ = (NaN, NaN, NaN)` and non-finite gradients out of
    /// a function whose own doc said a degenerate case resolves "rather than to a
    /// NaN". The explicit `|| det.is_nan()` is what makes that sentence true, and
    /// `det != det` is the same clause in the two shaders.
    #[test]
    fn a_nan_vertex_takes_the_degenerate_branch_rather_than_the_solve() {
        let good = fixture_triangle();
        for k in 0..3 {
            for field in 0..4 {
                let mut c = [good.0, good.1, good.2];
                c[k][field] = f32::NAN;
                let b = VisBary::of(c[0], c[1], c[2], Vec2::new(0.02, 0.03), 800.0, 600.0);
                assert!(
                    b.lambda.is_finite() && b.d_dx.is_finite() && b.d_dy.is_finite(),
                    "a NaN in vertex {k} component {field} reached the output: \
                     {b:?}"
                );
            }
        }
        // …and the clause is spelled in the two shaders too, so a tidy-up that
        // "simplified" it back to `abs(det) < eps` is a failing test.
        for (name, src) in [
            ("vis_resolve.wgsl", include_str!("shaders/vis_resolve.wgsl")),
            (
                "vis_feedback.wgsl",
                include_str!("shaders/vis_feedback.wgsl"),
            ),
        ] {
            let code: Vec<&str> = src
                .lines()
                .map(|l| l.trim_end_matches('\r').trim())
                .collect();
            assert!(
                code.contains(&"if (abs(det) < VIS_DET_EPS || det != det) {"),
                "{name}'s determinant guard has lost its NaN clause"
            );
            assert!(
                code.contains(&"if (abs(d) < VIS_DET_EPS || d != d) {"),
                "{name}'s weight-sum guard has lost its NaN clause"
            );
        }
    }

    /// **What the floor does NOT buy**, recorded as a measurement rather than
    /// left to be assumed (P28.1 audit).
    ///
    /// `VIS_DET_EPS` is a *finiteness* floor, not a conditioning bound. A sliver
    /// well above it produces a finite, enormous, meaningless gradient — and that
    /// is the safe direction only because `vt_mip` clamps to the top of the
    /// pyramid, which is a property of the consumer. The arm pins both halves: no
    /// non-finite output anywhere, and a gradient that really does blow up long
    /// before the floor is reached, so nobody reads the guard as conditioning.
    #[test]
    fn the_determinant_floor_bounds_finiteness_and_not_conditioning() {
        let mut worst_above_floor = 0.0f32;
        for eps in [1e-3f32, 1e-5, 1e-7, 1e-9, 1e-12, 1e-15, 1e-18, 1e-24] {
            let c0 = Vec4::new(0.10, 0.20, 0.5, 1.0);
            let c1 = Vec4::new(0.10 + eps, 0.20, 0.5, 1.0);
            let c2 = Vec4::new(0.10, 0.20 + eps, 0.5, 1.0);
            let b = VisBary::of(c0, c1, c2, Vec2::new(0.1, 0.2), 320.0, 180.0);
            assert!(
                b.lambda.is_finite() && b.d_dx.is_finite() && b.d_dy.is_finite(),
                "a sliver at eps {eps:e} left a non-finite result: {b:?}"
            );
            worst_above_floor = worst_above_floor.max(b.d_dx.abs().max_element());
        }
        assert!(
            worst_above_floor > 1.0,
            "no sliver above the floor produced a large gradient ({worst_above_floor}), \
             so this arm is not measuring the conditioning it is named for"
        );
    }

    /// The gradients scale with the **viewport**, not with the triangle: half the
    /// width is twice the gradient per pixel. A construction that forgot the
    /// `2/width` step would be viewport-independent, and the mip it justifies
    /// would be wrong by exactly the resolution ratio.
    #[test]
    fn halving_the_viewport_width_doubles_the_horizontal_gradient() {
        let (c0, c1, c2) = fixture_triangle();
        let ndc = Vec2::new(0.05, -0.05);
        let wide = VisBary::of(c0, c1, c2, ndc, 800.0, 600.0);
        let narrow = VisBary::of(c0, c1, c2, ndc, 400.0, 600.0);
        assert!(
            (narrow.d_dx - wide.d_dx * 2.0).abs().max_element() < 1e-9,
            "{:?} is not twice {:?}",
            narrow.d_dx,
            wide.d_dx
        );
        assert_eq!(narrow.d_dy, wide.d_dy, "height did not change");
    }

    #[test]
    fn an_attribute_interpolates_and_differentiates_through_the_same_weights() {
        let (c0, c1, c2) = fixture_triangle();
        let b = VisBary::of(c0, c1, c2, Vec2::new(0.02, 0.03), 800.0, 600.0);
        let (v, dx, dy) = b.attr(1.0, 1.0, 1.0);
        assert!(
            (v - 1.0).abs() < 1e-5,
            "a constant attribute must interpolate flat"
        );
        assert!(dx.abs() < 1e-6 && dy.abs() < 1e-6);
        let (v2, dx2, _) = b.attr(0.0, 1.0, 2.0);
        assert!((v2 - b.lambda.dot(Vec3::new(0.0, 1.0, 2.0))).abs() < 1e-6);
        assert!((dx2 - b.d_dx.dot(Vec3::new(0.0, 1.0, 2.0))).abs() < 1e-9);
    }

    /// A triangle in clip space through the renderer's own reverse-infinite-Z
    /// projection, tilted so the perspective correction is not a no-op.
    fn fixture_triangle() -> (Vec4, Vec4, Vec4) {
        let proj =
            glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 800.0 / 600.0, 0.1);
        let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let vp = proj * view;
        (
            vp * Vec4::new(-1.0, -1.0, -3.0, 1.0),
            vp * Vec4::new(1.5, -0.8, -9.0, 1.0),
            vp * Vec4::new(0.0, 1.2, -4.0, 1.0),
        )
    }
}
