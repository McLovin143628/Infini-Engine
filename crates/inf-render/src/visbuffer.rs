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
//! | field | word | bits | range | the ceiling it is designed against |
//! |---|---|---|---|---|
//! | triangle | 0 | 7 | 0..=127 | `inf_vgeom::BuildParams::max_triangles` defaults to **124** and every cooked `.inf_vmesh` in the tree carries that; `meshopt`'s hard cap is 512, so 128 is a *refusal* boundary rather than a format one, and it is deliberately kept reachable |
//! | meshlet | 0 | 25 | 0..=33 554 431 | a **shared pool slot**, not an asset-local id. `VisState::admit` reads the pool's **capacity**, so the ceiling is a fraction of the streaming budget — see below |
//! | instance | 1 | 24 | 0..=16 777 214 | `RenderScene::vgeom_instances` |
//! | *reserved* | 1 | 8 | must be zero | see "what the widening did NOT close" |
//!
//! The instance field stores `instance + 1`, which is what makes [`VIS_EMPTY`]
//! (zero) unreachable by a real fragment and therefore usable as the cleared
//! value. Spending one instance rather than one *packed word* is the cheaper
//! side of that trade: the alternative — clearing to all-ones — costs a
//! combination that is a legal triangle of a legal meshlet of a legal instance
//! and would have to be refused at pack time, inside the hot loop.
//!
//! # IB-8: why the id is SIXTY-FOUR bits
//!
//! It was thirty-two — triangle 7, meshlet 14, instance **11** — and the
//! certification's IB-8 is that `VIS_MAX_INSTANCES = 2047`, refused beyond, so
//! *"a city of thousands of Nanite-class building meshes cannot all enter the
//! visbuffer path in one frame"*.
//!
//! Thirty-two bits could not be re-cut to fix it, and the arithmetic says so.
//! The meshlet field addresses **pool capacity**, and P28.1 measured descriptors
//! at a stable **5.26 – 5.44 %** of a cooked asset's pool bytes, so:
//!
//! | meshlet bits | slots | descriptor bytes | resident pool it refuses past | of the 256 MiB default budget |
//! |---|---|---|---|---|
//! | 14 (as shipped) | 16 384 | 1 MiB | ~18.4 – 19.0 MiB | **7.4 %** |
//! | 12 | 4 096 | 256 KiB | ~4.6 – 4.8 MiB | 1.8 % |
//! | 11 | 2 048 | 128 KiB | ~2.3 – 2.4 MiB | **0.9 %** |
//!
//! So the meshlet field was *already* the binding one — P28.1's own words: "the
//! ceiling is nonetheless reachable by ordinary streamed content" — and buying
//! instances with meshlet bits would have made an already-firing refusal fire at
//! a ninth of the budget. The triangle field is worse still: shrinking it means
//! lowering `max_triangles`, which *increases* the meshlet count for the same
//! geometry and spends the bits it borrowed. Every re-cut of thirty-two bits
//! makes the frame refuse sooner somewhere.
//!
//! Widening is what a visibility buffer does elsewhere for the same reason, and
//! it costs **four bytes a pixel** — 3.7 MB at 1280 × 720, 33 MB at 4K — paid
//! only while `VgeomSettings::visbuffer` is on, which is off on every tier
//! (§5 of the memo) — eight bytes a pixel in total, 7.4 MB at 1280 × 720.
//! Against that: [`VIS_MAX_INSTANCES`] goes from 2 047 to **16 777 215**
//! (indices `0..=16 777 214`), which is 8 196× the brief's 16k target, and the
//! meshlet ceiling stops being a fraction of the budget at all.
//!
//! **WGSL has no 64-bit integer**, so the id is a `vec2<u32>` and no field may
//! straddle the word boundary. That is not a limitation worked around — it is
//! the reason the split is 7 + 25 in word 0 and 24 + 8 reserved in word 1, and
//! why the meshlet field got the whole remainder of word 0 rather than a
//! carefully-sized share.
//!
//! # What the widening did NOT close, precisely
//!
//! P28.1's `voxel.wgsl` header says the visibility path cannot name a second
//! geometry kind because *"all thirty-two bits are spent"*. Eight bits are now
//! free, and **the routing stands anyway**: the resolve shades by pulling three
//! vertices out of the shared meshlet pool, and a voxel chunk is a Surface-Nets
//! mesh in its own buffers with no meshlet structure, no DAG and no page. The
//! id was never the binding constraint; exhaustion was the reason *given*.
//! Widening it is what tells the two apart, and the shader's header now says so.
//! The reserved bits are asserted zero by [`VisPacking::pack`] and rejected by
//! [`VisPacking::unpack`], so the day a kind field is real it starts from a
//! stated invariant rather than from whatever happened to be there.
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
//! Two of the three refusals are now unreachable by real content, and that is
//! the point rather than a defect: the binding ceiling moves to the streaming
//! budget (`VgeomStreamBudget::budget_bytes`) and to
//! `MAX_CPU_SCATTER_INSTANCES`, which are *tunable* and *reported*, where a
//! packed-field ceiling was neither. The **triangle** refusal stays reachable —
//! it is `meshopt`'s own configurable cap, and
//! `a_scene_past_a_ceiling_falls_back_to_the_forward_path_and_says_which` cooks
//! a scene past it verbatim.
//!
//! # The alternative that was measured and not taken
//!
//! A **frame-derived** split — `meshlet_bits = ceil(log2(resident_slots))`, the
//! rest going to instances — is more capacious still within 32 bits. It is not
//! taken because it makes the layout a per-frame value rather than a constant,
//! which costs the shaders a uniform read per unpack, costs the arms below their
//! ability to be a *compile-time* mirror, and moves the refusal from
//! registration into the frame. Widening buys the same capacity and keeps all
//! three properties; it was routed to P28.3 and is hereby retired rather than
//! deferred again. Recorded in `docs/memos/p28-1-visbuffer.md` §1.

use glam::{Mat3, Vec2, Vec3, Vec4};

// ── the split ────────────────────────────────────────────────────────────────

/// Bits the **triangle index within its meshlet** occupies — word 0, LSB-most.
pub const VIS_TRI_BITS: u32 = 7;
/// Bits the **shared-pool meshlet slot** occupies — word 0, above the triangle.
pub const VIS_MESHLET_BITS: u32 = 25;
/// Bits the **biased instance index** occupies — word 1, LSB-most.
pub const VIS_INSTANCE_BITS: u32 = 24;

/// Word 0 is exactly full. **WGSL has no 64-bit integer**, so the id is a
/// `vec2<u32>` and a field that straddled the word boundary would need two
/// shifts, a mask and an or in every one of the three shaders that touch it —
/// which is a packing bug waiting to be written three times. Word 0 being
/// exactly 32 bits is what makes the triangle and meshlet unpacks single-word
/// expressions, unchanged from the 32-bit contract.
const _: () = assert!(VIS_TRI_BITS + VIS_MESHLET_BITS == 32);
/// …and word 1 has room to spare, which is stated rather than spent (see the
/// module docs on what the widening did *not* close).
const _: () = assert!(VIS_INSTANCE_BITS <= 32);
/// The whole id, both words.
pub const VIS_ID_BITS: u32 = 64;
const _: () = assert!(VIS_TRI_BITS + VIS_MESHLET_BITS + VIS_INSTANCE_BITS <= VIS_ID_BITS);

/// First bit of the meshlet field **within word 0**.
pub const VIS_MESHLET_SHIFT: u32 = VIS_TRI_BITS;
/// First bit of the instance field **within word 1**.
///
/// Zero, and named anyway: the shaders' three unpack expressions read the same
/// four constants the Rust side declares, and a shift that is implicitly zero is
/// a shift nobody notices when the layout moves.
pub const VIS_INSTANCE_SHIFT: u32 = 0;

/// Bits of word 1 above the instance field. **Must be zero.** Reserved, not
/// spare: see the module docs.
pub const VIS_RESERVED_BITS: u32 = 32 - VIS_INSTANCE_BITS;

/// The cleared value: **no geometry at this pixel**. Unreachable by
/// [`VisPacking::pack`] because the instance field is stored biased by one, so
/// word 1 of a real fragment is never zero.
pub const VIS_EMPTY: u64 = 0;

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
    pub fn pack(f: VisFragment) -> Option<u64> {
        if f.instance >= VIS_MAX_INSTANCES
            || f.meshlet >= VIS_MAX_MESHLET_SLOTS
            || f.tri >= VIS_MAX_TRIANGLES_PER_MESHLET
        {
            return None;
        }
        Some(Self::from_words([
            (f.meshlet << VIS_MESHLET_SHIFT) | f.tri,
            (f.instance + 1) << VIS_INSTANCE_SHIFT,
        ]))
    }

    /// Unpack one texel. `None` for [`VIS_EMPTY`] — the cleared value, which the
    /// bias makes unreachable from [`pack`](Self::pack) — and `None` for an id
    /// whose reserved bits are set, which no packer of this contract can produce.
    pub fn unpack(id: u64) -> Option<VisFragment> {
        let [w0, w1] = Self::words(id);
        if VIS_RESERVED_BITS > 0 && (w1 >> VIS_INSTANCE_BITS) != 0 {
            return None;
        }
        let biased = w1 >> VIS_INSTANCE_SHIFT;
        if biased == 0 {
            return None;
        }
        Some(VisFragment {
            instance: biased - 1,
            meshlet: (w0 >> VIS_MESHLET_SHIFT) & (VIS_MAX_MESHLET_SLOTS - 1),
            tri: w0 & (VIS_MAX_TRIANGLES_PER_MESHLET - 1),
        })
    }

    /// The id's two GPU words, `[word 0, word 1]` — the `vec2<u32>` the shaders
    /// write and read, in the channel order the `Rg32Uint` texel stores.
    #[inline]
    pub fn words(id: u64) -> [u32; 2] {
        [id as u32, (id >> 32) as u32]
    }

    /// The inverse of [`words`](Self::words).
    #[inline]
    pub fn from_words(w: [u32; 2]) -> u64 {
        u64::from(w[0]) | (u64::from(w[1]) << 32)
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

// ── the parity criterion ─────────────────────────────────────────────────────
//
// **The P28.1 audit recorded a criterion and P28.5 has to execute it.** It is
// stated here, once, so `tests/visbuffer_parity.rs` (the nucleus) and
// `runtime/inf-player/tests/phase28_gate.rs` (the phase gate) measure the same
// thing rather than two spellings of it that can drift apart. The ROADMAP's
// wording, verbatim:
//
// > **byte-exact on interior pixels to `PARITY_MAX_STEP = 1`, on every row whose
// > shading a derivative cannot reach; and on the textured row, bounded AND
// > classified — under 12 % of interior pixels differing, ≥ 80 % of them
// > bordering an agreeing pixel, ≤ 10 % of them centres of a solid 3 × 3
// > block.** Silhouettes and intersection curves are excluded by construction
// > (the interior test) and measured separately. No row may be exempted by a
// > fraction bound alone.
//
// The last sentence is why [`ParityVerdict`] carries the two class counts beside
// the population: a fraction is satisfied equally by a one-pixel mip boundary
// and by a smear over whole triangles, and only the class counts tell them
// apart. The GPU-free unit arms below hold a synthetic boundary and a synthetic
// smear that the fraction bound cannot separate and [`parity_ok`] must.

/// **The parity bound, and why it is one least-significant step rather than
/// zero.**
///
/// The forward path's barycentrics come from the rasterizer's interpolators; the
/// resolve solves for them. Those are the same numbers in exact arithmetic and
/// not always the same `f32`, and the frame is written to an **8-bit** target, so
/// a pixel whose shaded value sits within an f32 ulp of a `1/255` boundary can
/// round the other way. That is a quantization artefact of the comparison, not a
/// disagreement about shading — and the way to say so honestly is to bound it at
/// exactly one step and forbid two.
pub const PARITY_MAX_STEP: u8 = 1;

/// The fraction of interior pixels a **non-textured** row may leave on a
/// rounding boundary. A systematic offset looks exactly like a rounding boundary
/// once it is smaller than a step, so the population is bounded as well as the
/// magnitude.
pub const PARITY_UNTEXTURED_MAX_FRACTION: f64 = 0.02;

/// The fraction of interior pixels the **textured** row may leave differing.
///
/// The two paths are not expected to agree exactly there: the forward path's
/// `dpdx(uv)` is a first-order finite difference over a 2 × 2 quad and the
/// resolve's is the exact analytic derivative, so the two choose a different mip
/// where a footprint sits on a level boundary — a different texel, which is a
/// large delta rather than a rounding step.
pub const PARITY_TEXTURED_MAX_FRACTION: f64 = 0.12;

/// The fraction of the textured row's **differing** pixels that must border an
/// AGREEING interior pixel. A mip boundary is a curve; a wrong gradient is a
/// region, and a region's interior borders nothing that agrees.
pub const PARITY_TEXTURED_MIN_BORDERING: f64 = 0.80;

/// The fraction of the textured row's **differing** pixels that may sit at the
/// centre of a solid 3 × 3 block of disagreement. A boundary one or two pixels
/// thick has no such centres; a patch is made of them.
pub const PARITY_TEXTURED_MAX_SOLID_CENTRES: f64 = 0.10;

/// What one comparison of a forward frame against a resolved frame measured,
/// over the interior pixels it was handed.
///
/// The three class counts are the criterion's own vocabulary: `differing` is the
/// *population*, `bordering` and `solid_centres` are the *shape*, and a claim
/// made on the population alone is the one the recorded criterion forbids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParityVerdict {
    /// Interior pixels compared.
    pub interior: usize,
    /// …of which this many differ in at least one of the four channels.
    pub differing: usize,
    /// The worst single-channel delta, in 8-bit steps.
    pub worst_step: u8,
    /// Differing pixels with an orthogonal neighbour that is interior AND
    /// agreeing — the "this is a boundary" count.
    pub bordering: usize,
    /// Differing pixels at the centre of a solid 3 × 3 differing block — the
    /// "this is a patch" count.
    pub solid_centres: usize,
}

impl ParityVerdict {
    /// Differing pixels as a fraction of the interior population. Zero over an
    /// empty population, which [`parity_ok`] refuses separately rather than
    /// letting it read as agreement.
    pub fn differing_fraction(&self) -> f64 {
        if self.interior == 0 {
            return 0.0;
        }
        self.differing as f64 / self.interior as f64
    }

    /// Bordering pixels as a fraction of the **differing** ones. `1.0` when
    /// nothing differs — a row with no disagreement has no disagreement that is
    /// the wrong shape.
    pub fn bordering_fraction(&self) -> f64 {
        if self.differing == 0 {
            return 1.0;
        }
        self.bordering as f64 / self.differing as f64
    }

    /// Solid-block centres as a fraction of the **differing** pixels.
    pub fn solid_centre_fraction(&self) -> f64 {
        if self.differing == 0 {
            return 0.0;
        }
        self.solid_centres as f64 / self.differing as f64
    }

    /// One line, in the shape the parity suites already print.
    pub fn summary(&self) -> String {
        format!(
            "{} interior, {} differing ({:.2} %), worst step {}, {:.1} % bordering \
             an agreeing pixel, {} solid 3x3 centres ({:.1} %)",
            self.interior,
            self.differing,
            100.0 * self.differing_fraction(),
            self.worst_step,
            100.0 * self.bordering_fraction(),
            self.solid_centres,
            100.0 * self.solid_centre_fraction(),
        )
    }
}

/// Compare two RGBA8 frames over the interior pixel set, and classify the
/// disagreement.
///
/// `interior` is the pixel set the caller has already established a derivative
/// cannot reach — in the renderer that is [`crate::passes::visbuffer::VisImage::interior`],
/// a pixel whose four neighbours carry its own id. Silhouettes are excluded
/// *there*, by construction, because a 4× MSAA resolve and a single-sample one
/// are obliged to differ at partial coverage; this function measures only what
/// it is handed.
///
/// Both frames are indexed as `((y * width + x) * 4)`, so they must be the same
/// dimensions. Coordinates outside either buffer are skipped rather than
/// panicking, which keeps a truncated readback a *small* population (which
/// [`parity_ok`] refuses) rather than an abort.
pub fn parity_verdict(
    forward: &[u8],
    resolved: &[u8],
    width: u32,
    interior: &[(u32, u32)],
) -> ParityVerdict {
    use std::collections::BTreeSet;

    let set: BTreeSet<(u32, u32)> = interior.iter().copied().collect();
    let mut v = ParityVerdict::default();
    let mut differing: BTreeSet<(u32, u32)> = BTreeSet::new();

    for &(x, y) in &set {
        let i = ((y as usize * width as usize) + x as usize) * 4;
        if i + 4 > forward.len() || i + 4 > resolved.len() {
            continue;
        }
        v.interior += 1;
        let mut differs = false;
        for c in 0..4 {
            let d = forward[i + c].abs_diff(resolved[i + c]);
            if d > 0 {
                differs = true;
                v.worst_step = v.worst_step.max(d);
            }
        }
        if differs {
            differing.insert((x, y));
        }
    }
    v.differing = differing.len();

    for &(x, y) in &differing {
        // A boundary: some orthogonal neighbour is interior and agrees.
        let neighbours = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        if neighbours
            .iter()
            .any(|p| set.contains(p) && !differing.contains(p))
        {
            v.bordering += 1;
        }
        // A patch: the whole 3 × 3 around it differs.
        if x >= 1
            && y >= 1
            && (0..3).all(|dy| (0..3).all(|dx| differing.contains(&(x + dx - 1, y + dy - 1))))
        {
            v.solid_centres += 1;
        }
    }
    v
}

/// The recorded criterion, applied. `Err` carries the measurement AND the bound,
/// because a verdict that says only "too different" cannot be acted on.
///
/// `textured` selects which half of the criterion the row is held to:
///
/// * `false` — a row whose shading a screen derivative cannot reach. Every
///   interior pixel agrees to [`PARITY_MAX_STEP`], and no more than
///   [`PARITY_UNTEXTURED_MAX_FRACTION`] of them sit on a rounding boundary.
/// * `true` — the textured row. The magnitude is *not* bounded (a different mip
///   is a different texel); the population is, at
///   [`PARITY_TEXTURED_MAX_FRACTION`], **and so is the shape**, at
///   [`PARITY_TEXTURED_MIN_BORDERING`] and
///   [`PARITY_TEXTURED_MAX_SOLID_CENTRES`]. The recorded criterion's closing
///   sentence — *no row may be exempted by a fraction bound alone* — is those
///   two clauses.
///
/// An empty population is refused in both modes: a parity claim over no pixels
/// is the vacuous shape the P19 audit named, and it is the failure mode a
/// mis-wired fixture actually produces.
pub fn parity_ok(v: &ParityVerdict, textured: bool) -> Result<(), String> {
    if v.interior == 0 {
        return Err(
            "the comparison was handed no interior pixels; a parity claim over \
             nothing is not one"
                .to_string(),
        );
    }
    if !textured {
        if v.worst_step > PARITY_MAX_STEP {
            return Err(format!(
                "an interior pixel differs by {} of 255, past the {PARITY_MAX_STEP} \
                 step this bound allows. The two paths agree to within 8-bit \
                 rounding or they do not agree; anything past one step is a \
                 disagreement about shading, not about quantization. ({})",
                v.worst_step,
                v.summary()
            ));
        }
        if v.differing_fraction() > PARITY_UNTEXTURED_MAX_FRACTION {
            return Err(format!(
                "{:.2} % of interior pixels sit on a rounding boundary, past the \
                 {:.0} % this bound allows — a systematic offset looks exactly \
                 like this once it is smaller than a step. ({})",
                100.0 * v.differing_fraction(),
                100.0 * PARITY_UNTEXTURED_MAX_FRACTION,
                v.summary()
            ));
        }
        return Ok(());
    }
    if v.differing_fraction() >= PARITY_TEXTURED_MAX_FRACTION {
        return Err(format!(
            "{:.2} % of interior pixels disagree, past the {:.0} % this bound \
             allows for the analytic-vs-quad-derivative mip boundary — the \
             gradients disagree about more than a boundary. ({})",
            100.0 * v.differing_fraction(),
            100.0 * PARITY_TEXTURED_MAX_FRACTION,
            v.summary()
        ));
    }
    if v.bordering_fraction() < PARITY_TEXTURED_MIN_BORDERING {
        return Err(format!(
            "only {:.1} % of the differing pixels border an AGREEING interior \
             pixel, under the {:.0} % this bound requires — the disagreement is a \
             region, not a boundary, and 'the mip-transition set' is the wrong \
             name for it. ({})",
            100.0 * v.bordering_fraction(),
            100.0 * PARITY_TEXTURED_MIN_BORDERING,
            v.summary()
        ));
    }
    if v.solid_centre_fraction() > PARITY_TEXTURED_MAX_SOLID_CENTRES {
        return Err(format!(
            "{:.1} % of the differing pixels sit at the centre of a solid 3x3 \
             block of disagreement, past the {:.0} % this bound allows — a mip \
             boundary is one or two pixels thick and this is a patch, which is \
             what a wrong gradient over a whole triangle looks like. ({})",
            100.0 * v.solid_centre_fraction(),
            100.0 * PARITY_TEXTURED_MAX_SOLID_CENTRES,
            v.summary()
        ));
    }
    Ok(())
}

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
    /// the three fields unpacks to what packed it, and no two of them collide.
    ///
    /// **The strides are derived from the field widths** (IB-8). They used to be
    /// literals — `instance 0..2047`, `meshlet` by 256, `tri` by 16 — which was
    /// 1 048 064 round trips at 32 bits and would be **2.1 × 10⁹** at 64. A sweep
    /// whose size is a function of the constants it sweeps is the only kind that
    /// survives the constants moving, and this file exists to let them move.
    ///
    /// Injectivity is asserted directly rather than inferred from the round
    /// trip: `unpack(pack(f)) == f` for every `f` already implies `pack` is
    /// injective, but it implies it by an argument a reader has to make, and the
    /// property that matters — two different fragments are never the same texel —
    /// is one line to state.
    #[test]
    fn the_packing_is_injective_over_a_dense_sweep() {
        const SAMPLES: u32 = 64;
        let stride = |max: u32| (max / SAMPLES).max(1);
        let (si, sm, st) = (
            stride(VIS_MAX_INSTANCES),
            stride(VIS_MAX_MESHLET_SLOTS),
            stride(VIS_MAX_TRIANGLES_PER_MESHLET),
        );
        let mut seen: std::collections::BTreeMap<u64, VisFragment> =
            std::collections::BTreeMap::new();
        for instance in (0..VIS_MAX_INSTANCES).step_by(si as usize) {
            for meshlet in (0..VIS_MAX_MESHLET_SLOTS).step_by(sm as usize) {
                for tri in (0..VIS_MAX_TRIANGLES_PER_MESHLET).step_by(st as usize) {
                    let f = VisFragment {
                        instance,
                        meshlet,
                        tri,
                    };
                    let id = VisPacking::pack(f).unwrap();
                    assert_ne!(
                        id, VIS_EMPTY,
                        "{f:?} packs to the cleared value, so an empty pixel and \
                         a real one are the same word"
                    );
                    assert_eq!(VisPacking::unpack(id), Some(f));
                    if let Some(other) = seen.insert(id, f) {
                        panic!("{f:?} and {other:?} pack to the same id {id:#018x}");
                    }
                }
            }
        }
        // ANTI-VACUITY: a sweep that visited one cell of each field would satisfy
        // every line above.
        assert!(
            seen.len() > 200_000,
            "the sweep covered {} fragments — too few to be exhaustive of \
             anything",
            seen.len()
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
                let [_, w1] = VisPacking::words(id);
                assert!(
                    w1 >> VIS_INSTANCE_SHIFT >= 1,
                    "instance 0 must store as 1, or the empty sentinel is a lie"
                );
                assert_eq!(
                    w1 >> VIS_INSTANCE_BITS,
                    0,
                    "the reserved bits of word 1 must be zero"
                );
            }
        }
        // …and an id that DOES set them is refused, so the reserved bits are an
        // invariant rather than a note. Nothing this contract packs can reach
        // here; the day a geometry-kind field is real it starts from a stated
        // zero rather than from whatever a stale texel held.
        let good = VisPacking::pack(VisFragment {
            instance: 3,
            meshlet: 4,
            tri: 5,
        })
        .unwrap();
        assert!(VisPacking::unpack(good).is_some());
        let [w0, w1] = VisPacking::words(good);
        let poisoned = VisPacking::from_words([w0, w1 | (1 << VIS_INSTANCE_BITS)]);
        assert_eq!(
            VisPacking::unpack(poisoned),
            None,
            "an id with a reserved bit set must not decode as geometry"
        );
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

            // **And the DECLARATION is not the contract — the USE is** (I3
            // audit). Every shader above declared the widened constants and one
            // of them could still have masked with the old literal: replacing
            // `VIS_MESHLET_BITS` with `14u` inside `vis_feedback.wgsl`'s unpack
            // left all seven feedback arms green, because no fixture in the tree
            // resides past the old field and a truncating mask is a no-op until
            // one does. That is the I1 audit's law met in WGSL — a pin on a
            // needle that is a prefix of a declaration can never fail — so the
            // use sites are pinned as COUNTS.
            //
            // Per shader: `VIS_TRI_BITS` three times (its own declaration,
            // `VIS_MESHLET_SHIFT`'s definition, and the triangle mask);
            // `VIS_MESHLET_BITS` / `VIS_MESHLET_SHIFT` / `VIS_INSTANCE_SHIFT`
            // twice. `VIS_INSTANCE_BITS` is twice in the writer, which masks the
            // biased instance into its field, and ONCE in each reader, which
            // subtracts one from the whole word because the bias is what makes
            // word 1 unambiguous. If a use site is legitimately added or
            // removed, rewrite this pin rather than deleting it.
            let uses = |needle: &str| code.iter().filter(|l| l.contains(needle)).count();
            for (needle, want) in [
                ("VIS_TRI_BITS", 3),
                ("VIS_MESHLET_BITS", 2),
                ("VIS_MESHLET_SHIFT", 2),
                ("VIS_INSTANCE_SHIFT", 2),
                (
                    "VIS_INSTANCE_BITS",
                    if name == "visbuffer.wgsl" { 2 } else { 1 },
                ),
            ] {
                assert_eq!(
                    uses(needle),
                    want,
                    "{name} names `{needle}` on {} code lines and the layout \
                     expects {want} — a field that is declared and then masked \
                     with a literal is a truncation nothing in this tree renders \
                     large enough to see",
                    uses(needle)
                );
            }
        }
    }

    /// **The vertex record's stride is one number in five places** (P28.2).
    ///
    /// The meshlet vertex pool is bound as `array<f32>` and every shader that
    /// reads it multiplies a vertex index by the record's float count. A wrong
    /// stride does not error and does not fail validation: it reads a
    /// neighbouring vertex's bytes as this one's position, which is a mesh that
    /// renders and is wrong everywhere. So the four shaders declare the constant
    /// and this compares all four against `VERTEX_REC_LEN` — the Rust the
    /// container writes with — rather than against a literal.
    #[test]
    fn the_shaders_vertex_stride_is_the_rusts() {
        let floats = inf_vgeom::asset::VERTEX_REC_LEN / 4;
        assert_eq!(
            inf_vgeom::asset::VERTEX_REC_LEN % 4,
            0,
            "a vertex record that is not a whole number of floats cannot be read \
             out of an `array<f32>` at all"
        );
        let want = format!("const VGEOM_VSTRIDE: u32 = {floats}u;");
        for (name, src) in [
            ("vgeom_mesh.wgsl", include_str!("shaders/vgeom_mesh.wgsl")),
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
            assert!(
                code.iter().any(|l| l.starts_with(want.as_str())),
                "{name} does not declare `{want}` — the vertex record moved and \
                 one of its readers did not"
            );
            // And the constant is USED (P28.2 audit: banning one literal
            // spelling left `global_v * 10u` passing both arms — a ban
            // enumerates what you thought of, so require the constant instead).
            assert!(
                code.iter().any(|l| l.contains("global_v * VGEOM_VSTRIDE")),
                "{name} declares the stride and steps by something else"
            );
            assert!(
                !code
                    .iter()
                    .any(|l| l.contains("global_v * ") && !l.contains("VGEOM_VSTRIDE")),
                "{name} multiplies a vertex index by something that is not the \
                 declared stride"
            );
        }
    }

    /// **The tangent word's unpack is one function in two languages** (P28.2).
    ///
    /// `inf_vgeom::unpack_tangent` is the reference and `vt_sample.wgsl`'s
    /// `vgeom_unpack_tangent` is the mirror; there is no way to run the WGSL half
    /// on the host, so what is checked is that the mirror is built from the same
    /// four constants the Rust half is — the quantization scale, the handedness
    /// bit, and the two shift amounts that sign-extend the 11-bit fields. Each of
    /// them is a silent wrong answer if it drifts: a wrong scale tilts every
    /// tangent, a wrong bit flips every bitangent, a wrong shift decodes a
    /// different direction entirely.
    #[test]
    fn the_shaders_tangent_unpack_is_the_rusts() {
        let src = include_str!("shaders/vt_sample.wgsl");
        let code: Vec<String> = src
            .lines()
            .map(|l| l.trim_end_matches('\r').trim().to_string())
            .filter(|l| !l.starts_with("//"))
            .collect();
        let body = code.join("\n");
        for (what, needle) in [
            ("the quantization scale", "/ 1023.0"),
            ("the handedness bit", "(word & 0x400000u) != 0u"),
            ("the low field's sign extension", "i32(word << 21u) >> 21u"),
            ("the high field's sign extension", "i32(word << 10u) >> 21u"),
            ("the sentinel", "if (word == 0u)"),
            // **The octahedral fold, which had no pin at all** (P28.2 audit).
            // Every line below is a silently wrong hemisphere if it drifts, and
            // the five needles above are all satisfied while the fold is deleted
            // outright, has its two signs swapped, or drops the x/y swap.
            ("the fold's z", "1.0 - abs(ex) - abs(ey)"),
            ("the fold's guard", "if (v.z < 0.0)"),
            ("the fold's x sign", "let sx = select(1.0, -1.0, v.x < 0.0)"),
            ("the fold's y sign", "let sy = select(1.0, -1.0, v.y < 0.0)"),
            (
                "the fold's swap",
                "v = vec3<f32>((1.0 - abs(v.y)) * sx, (1.0 - abs(v.x)) * sy, v.z)",
            ),
            // And the handedness is read as a SIGN, never as a magnitude: the
            // forward path interpolates `w` as an ordinary varying, so a seam
            // delivers a fractional one and `cross(n, t) * w` would hand back a
            // short bitangent — a non-orthonormal frame, which is the exact
            // failure the Gram-Schmidt above exists to prevent.
            (
                "the handedness as a sign",
                "select(-1.0, 1.0, tan4.w > 0.0)",
            ),
        ] {
            assert!(
                body.contains(needle),
                "vt_sample.wgsl's tangent unpack no longer spells {what} \
                 (`{needle}`) — it has drifted from `inf_vgeom::unpack_tangent`"
            );
        }
        // The Rust half really does use those numbers, so the pin above is a
        // comparison and not two copies of a guess.
        //
        // The direction is chosen to EXERCISE them (P28.2 audit): this arm used
        // to round-trip `[0, 0, 1]`, whose two quantized fields are both **zero**,
        // so any pair of shift amounts and any fold at all decoded it correctly
        // and the assert saw nothing but the handedness bit. This one is in the
        // lower hemisphere (so the fold runs) with two different non-zero
        // magnitudes of opposite sign (so a swapped `sx`/`sy`, a dropped swap, or
        // a shift that reads the wrong field all move it).
        let d = {
            let v = [0.3f32, -0.7, -0.5];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let (t, w) = inf_vgeom::unpack_tangent(inf_vgeom::pack_tangent(d, -1.0)).expect("packed");
        assert_eq!(w, -1.0, "the handedness bit round-trips on the host");
        let cos = t[0] * d[0] + t[1] * d[1] + t[2] * d[2];
        assert!(
            1.0 - cos < 1e-5,
            "the host round-trip lost the direction: {t:?} against {d:?}"
        );
        assert!(inf_vgeom::unpack_tangent(inf_vgeom::NO_TANGENT).is_none());
    }

    /// **The voxel routing survives the id widening** — and the widening is what
    /// tells the two reasons apart (IB-8).
    ///
    /// This arm was `the_visbuffer_id_space_has_no_room_for_a_second_geometry_
    /// kind`, and it asserted `VIS_TRI_BITS + VIS_MESHLET_BITS +
    /// VIS_INSTANCE_BITS == 32`. Its own doc said it was *"the falsifier: it
    /// fails the day the id space grows, which is the day the voxel door
    /// genuinely opens"*. IB-8 grew the id space, the arm failed exactly as
    /// designed — and the prediction it encoded turned out to be **wrong**, which
    /// is the whole value of having written it down.
    ///
    /// The door did not open. P27.5 routed `voxel.wgsl`'s missing shadow receiver
    /// to "the VisBuffer resolve, where meshlet and voxel surfaces are shaded
    /// through one material pass", and P28.1 refused it citing bit exhaustion.
    /// There are now **eight reserved bits** in word 1, and the refusal stands
    /// anyway: the resolve shades by pulling three vertices out of the *shared
    /// meshlet pool*, and a voxel chunk is a Surface-Nets mesh in its own
    /// buffers, with no meshlet structure, no DAG and no page. A kind field would
    /// name a geometry the resolve still cannot read.
    ///
    /// So the arm keeps its job with its subject corrected: the id has room, the
    /// routing is structural, and both statements are checked. It fails the day
    /// somebody spends the reserved bits without revisiting the routing, and the
    /// day `voxel.wgsl` stops naming its real blocker.
    // `VIS_RESERVED_BITS > 0` const-folds, and that is the point: the subject of
    // this arm is a *constant*, and the arm's job is to fail — visibly, with its
    // message — the day somebody spends those bits. A `const _: () = assert!(…)`
    // would turn that into a compile error, which stops the build instead of
    // sending the reader to the routing.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn the_voxel_routing_survives_the_id_widening() {
        assert!(
            VIS_RESERVED_BITS > 0,
            "the id has no reserved bits left; if a kind field was added, this \
             arm's subject — 'there is room and it is still not enough' — has \
             changed and needs rewriting rather than deleting"
        );
        // The room is real: a kind would fit where nothing is packed today.
        let id = VisPacking::pack(VisFragment {
            instance: VIS_MAX_INSTANCES - 1,
            meshlet: VIS_MAX_MESHLET_SLOTS - 1,
            tri: VIS_MAX_TRIANGLES_PER_MESHLET - 1,
        })
        .expect("the widest legal fragment packs");
        assert_eq!(
            VisPacking::words(id)[1] >> VIS_INSTANCE_BITS,
            0,
            "even the widest fragment leaves the reserved bits clear"
        );
        // …and the refusal's own text, so the two cannot drift apart. Read over
        // `lines()` because every `.wgsl` in the working tree is CRLF.
        let voxel: Vec<&str> = include_str!("shaders/voxel.wgsl")
            .lines()
            .map(|l| l.trim_end_matches('\r'))
            .collect();
        for want in [
            "P28.1: THE ROUTING WAS WRONG, AND HERE IS THE MEASUREMENT",
            "the_voxel_routing_survives_the_id_widening",
            "meshletize voxel chunks",
            // The corrected blocker, by name. A routing whose stated reason has
            // been retired and not replaced is a routing nobody can act on.
            "no meshlet structure",
        ] {
            assert!(
                voxel.iter().any(|l| l.contains(want)),
                "voxel.wgsl's P28.1 correction lost `{want}` — a routing that \
                 summarises instead of naming its blocker is the P27.5 finding, \
                 one phase later"
            );
        }
        // The retired reason must be GONE, not merely joined by the real one:
        // leaving "all thirty-two bits are spent" in the file would send the next
        // reader to widen an id that is already wide.
        assert!(
            !voxel
                .iter()
                .any(|l| l.contains("thirty-two bits are spent")),
            "voxel.wgsl still gives bit exhaustion as its blocker, and the id is \
             sixty-four bits with eight reserved"
        );
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

    // ── the parity criterion, on synthetic frames ────────────────────────────

    /// A synthetic pair of RGBA8 frames and the interior population over them.
    ///
    /// The interior is the whole frame inset by one pixel, which is what the
    /// device's `VisImage::interior` produces for a fixture that fills the view:
    /// a pixel needs its four neighbours to be *addressable* for the boundary
    /// test to mean anything. `delta` is added to the red channel wherever
    /// `differs` says so, which is the one-channel case an 8-bit comparison
    /// actually sees.
    fn synthetic(
        w: u32,
        h: u32,
        delta: u8,
        differs: impl Fn(u32, u32) -> bool,
    ) -> (Vec<u8>, Vec<u8>, Vec<(u32, u32)>) {
        let mut fwd = vec![128u8; (w * h * 4) as usize];
        let mut res = fwd.clone();
        let mut interior = Vec::new();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                interior.push((x, y));
                if differs(x, y) {
                    let i = ((y * w + x) * 4) as usize;
                    res[i] = 128u8.wrapping_add(delta);
                }
            }
        }
        // The frames are otherwise identical, so anything the verdict reports is
        // the `differs` predicate and not the fixture.
        fwd[0] = 128;
        (fwd, res, interior)
    }

    const SYN: u32 = 64;

    /// **The classifier separates a boundary from a smear, and the fraction
    /// bound alone cannot.**
    ///
    /// This is the P28.1 audit's finding made executable. Both fixtures below
    /// differ on a population the textured bound admits — 1.6 % and 10.4 %, both
    /// under [`PARITY_TEXTURED_MAX_FRACTION`] — so a criterion made of the
    /// fraction alone passes both. One of them is a one-pixel curve (what a mip
    /// transition looks like) and the other is a patch (what a wrong gradient
    /// over whole triangles looks like), and the two class counts are the only
    /// thing that tells them apart.
    #[test]
    fn the_classifier_separates_a_mip_boundary_from_a_smear() {
        // A one-pixel diagonal: every differing pixel has four agreeing
        // orthogonal neighbours, and no 3 × 3 around it is solid.
        let (fwd, res, interior) = synthetic(SYN, SYN, 40, |x, y| x == y);
        let boundary = parity_verdict(&fwd, &res, SYN, &interior);
        assert_eq!(boundary.interior, ((SYN - 2) * (SYN - 2)) as usize);
        assert_eq!(boundary.differing, (SYN - 2) as usize);
        assert_eq!(boundary.worst_step, 40, "the fixture's delta is mip-scale");
        assert_eq!(
            boundary.bordering, boundary.differing,
            "every pixel of a one-pixel curve borders an agreeing pixel"
        );
        assert_eq!(
            boundary.solid_centres, 0,
            "a curve one pixel thick has no solid 3x3 centres"
        );
        assert_eq!(parity_ok(&boundary, true), Ok(()), "{}", boundary.summary());

        // A 20 × 20 patch: 400 of 3 844 interior pixels, which is 10.4 % — under
        // the same fraction bound the boundary passed.
        let (fwd, res, interior) = synthetic(SYN, SYN, 40, |x, y| {
            (10..30).contains(&x) && (10..30).contains(&y)
        });
        let smear = parity_verdict(&fwd, &res, SYN, &interior);
        assert_eq!(smear.differing, 400);
        assert_eq!(
            smear.bordering, 76,
            "only the patch's rim borders agreement"
        );
        assert_eq!(smear.solid_centres, 324, "18 x 18 of the patch is interior");

        // **THE POINT, asserted before the verdict**: the population bound does
        // not separate them. If this ever fails the two fixtures have stopped
        // being the pair this arm is about.
        assert!(
            smear.differing_fraction() < PARITY_TEXTURED_MAX_FRACTION,
            "the smear is {:.2} % of the interior, which the fraction bound \
             already rejects — this arm would then prove nothing about the class \
             counts",
            100.0 * smear.differing_fraction()
        );

        // …and the criterion rejects it anyway, by class.
        let err = parity_ok(&smear, true).expect_err("a smear must be refused");
        assert!(
            err.contains("region, not a boundary"),
            "the refusal does not name the class it refused on: {err}"
        );
        assert!(
            err.contains("19.0 %") && err.contains("80 %"),
            "a refusal must print the measurement AND the bound: {err}"
        );
    }

    /// **The solid-block clause is load-bearing ON ITS OWN**, and this is the
    /// fixture that says so.
    ///
    /// Five well-separated 3 × 3 patches. Eight of every nine pixels are rim, so
    /// **88.9 % border an agreeing pixel** and the bordering clause is satisfied;
    /// 1.2 % of the interior differs, so the population clause is satisfied too.
    /// The only thing that sees these is the solid-centre count — one centre per
    /// patch, **11.1 %**, past the 10 % bound. Drop that clause and this fixture
    /// passes a criterion it must not.
    #[test]
    fn a_field_of_small_patches_is_refused_by_the_solid_clause_alone() {
        const PATCHES: [(u32, u32); 5] = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)];
        let (fwd, res, interior) = synthetic(SYN, SYN, 40, |x, y| {
            PATCHES
                .iter()
                .any(|(bx, by)| x.abs_diff(*bx) <= 1 && y.abs_diff(*by) <= 1)
        });
        let v = parity_verdict(&fwd, &res, SYN, &interior);
        assert_eq!(v.differing, 45, "five 3x3 patches");
        assert_eq!(v.bordering, 40, "eight rim pixels a patch");
        assert_eq!(v.solid_centres, 5, "one centre a patch");
        // The other two clauses are SATISFIED — asserted first, so the refusal
        // below is attributable to the third and to nothing else.
        assert!(
            v.differing_fraction() < PARITY_TEXTURED_MAX_FRACTION,
            "{}",
            v.summary()
        );
        assert!(
            v.bordering_fraction() >= PARITY_TEXTURED_MIN_BORDERING,
            "{}",
            v.summary()
        );
        let err = parity_ok(&v, true).expect_err("a field of patches is a smear");
        assert!(err.contains("solid 3x3"), "{err}");
        assert!(
            err.contains("11.1 %") && err.contains("10 %"),
            "a refusal must print the measurement AND the bound: {err}"
        );
    }

    /// **The untextured half admits one step and forbids two** — the
    /// `PARITY_MAX_STEP` clause, in both directions, plus the population bound
    /// that catches a systematic offset hiding under a step.
    #[test]
    fn the_untextured_half_admits_one_step_and_forbids_two() {
        // Sparse and one step: rounding, which is what the bound is for.
        let sparse = |x: u32, y: u32| (x * 7 + y * 13).is_multiple_of(401);
        let (fwd, res, interior) = synthetic(SYN, SYN, 1, sparse);
        let v = parity_verdict(&fwd, &res, SYN, &interior);
        assert_eq!(v.worst_step, 1);
        assert!(v.differing > 0, "the fixture differs nowhere at all");
        assert!(v.differing_fraction() < PARITY_UNTEXTURED_MAX_FRACTION);
        assert_eq!(parity_ok(&v, false), Ok(()), "{}", v.summary());

        // The same population, two steps: a disagreement about shading.
        let (fwd, res, interior) = synthetic(SYN, SYN, 2, sparse);
        let v = parity_verdict(&fwd, &res, SYN, &interior);
        assert_eq!(v.worst_step, 2);
        let err = parity_ok(&v, false).expect_err("two steps is not rounding");
        assert!(err.contains("differs by 2 of 255"), "{err}");

        // One step everywhere: the systematic offset the population bound is
        // for. The magnitude clause is satisfied and the criterion still refuses.
        let (fwd, res, interior) = synthetic(SYN, SYN, 1, |_, _| true);
        let v = parity_verdict(&fwd, &res, SYN, &interior);
        assert_eq!(v.worst_step, PARITY_MAX_STEP);
        let err = parity_ok(&v, false).expect_err("a whole-frame offset is not rounding");
        assert!(err.contains("rounding boundary"), "{err}");
    }

    /// **A parity claim over nothing is refused**, in both modes — the vacuous
    /// shape a mis-wired fixture actually produces, and the one a fraction of
    /// `0 / 0` would otherwise report as perfect agreement.
    #[test]
    fn a_parity_claim_over_no_interior_pixels_is_refused() {
        let v = parity_verdict(&[0u8; 16], &[0u8; 16], 2, &[]);
        assert_eq!(v, ParityVerdict::default());
        for textured in [false, true] {
            let err = parity_ok(&v, textured).expect_err("an empty population is not agreement");
            assert!(err.contains("no interior pixels"), "{err}");
        }
        // …and a row that agrees everywhere over a real population is fine in
        // both modes, so the refusal above is about the population and not about
        // the absence of disagreement.
        let (fwd, res, interior) = synthetic(SYN, SYN, 0, |_, _| false);
        let v = parity_verdict(&fwd, &res, SYN, &interior);
        assert_eq!(v.differing, 0);
        assert_eq!(v.bordering_fraction(), 1.0);
        assert_eq!(v.solid_centre_fraction(), 0.0);
        assert_eq!(parity_ok(&v, false), Ok(()));
        assert_eq!(parity_ok(&v, true), Ok(()));
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
