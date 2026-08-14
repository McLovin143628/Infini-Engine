//! **The virtual-texture streaming loop** (P26.4): the analytic want floor, the
//! GPU coverage feedback that refines it, and the pop-in instruments.
//!
//! One call per frame, at the renderer's sync point, doing five things in a fixed
//! order:
//!
//! 1. gather this frame's **coverage** — every drawn surface that names a virtual
//!    texture, as a render-local bounding sphere;
//! 2. compute the **analytic floor** from `(camera, bounds, registry)`;
//! 3. read the feedback mask from **frame F − 2** (or nothing) and decode it into
//!    refinement wants in virtual-address order;
//! 4. apply `floor ∪ refinement` to the residency and upload the pages;
//! 5. record the next feedback pass and hand its buffer to the ring.
//!
//! # The floor is a floor
//!
//! Residency can never fall below the analytic floor because of anything the
//! feedback does or fails to do, and that is structural rather than a policy:
//!
//! * every floor want is [`VT_PRIORITY_FLOOR`](inf_vt::VT_PRIORITY_FLOOR) and
//!   every refinement is [`VT_PRIORITY_FEEDBACK`](inf_vt::VT_PRIORITY_FEEDBACK),
//!   and `VtResidency::apply_wants` sorts on priority first and never evicts a
//!   page it has already touched **this transaction** — so a refinement cannot
//!   take a floor tile's slot;
//! * a late, dropped or never-arriving mask contributes **zero** refinements, so
//!   the transaction is exactly the floor's. A gate can therefore assert that a
//!   frame with feedback disabled produces the same trace as a frame whose ring
//!   missed, which is what "degrades deterministically" has to mean.
//!
//! # What the floor is, and what it costs
//!
//! `VtTextures::want_floor`'s three coarsest levels are **camera-free** and
//! bounded at 21 pages per texture — and on a 4096² texture those three levels
//! are 4×4, 2×2 and 1×1 *texels*. That is "visibly textured, not sharp", exactly
//! as the P26.3 ledger says. The analytic floor adds, per **visible** surface,
//! the finest level whose tile count fits [`VT_FLOOR_MAX_TILES`] and that is no
//! finer than the surface's screen footprint justifies. So the floor's cost is
//! bounded by `visible surfaces × VT_FLOOR_MAX_TILES` and its quality tracks the
//! camera; the feedback then asks for the level the footprint *really* justifies,
//! uncapped, and the budget decides how much of it is served.
//!
//! Both halves compute the level from the same rule. That is deliberate: the
//! floor is the answer the CPU can always give, the feedback is the same question
//! asked where the whole draw set is already on the GPU and where P27/P28 extend
//! it. See `docs/memos/p26-4-feedback-mechanism.md`.

use std::collections::BTreeMap;

use glam::{Mat4, Vec3};
use inf_vt::{TileCoord, VtFeedbackLayout, VtTextureHandle, VtWant, VT_PRIORITY_FLOOR};

use crate::camera::RenderView;
use crate::readback::ReadbackRing;
use crate::scene::{RenderScene, VtTextureSet};
use crate::vt_library::VtTextures;

/// The most tiles the **analytic floor** claims for one visible surface texture.
///
/// Sixteen: a 4×4 tile grid is 512² payload texels, which is a legible surface at
/// any framing a floor has to survive, and sixteen BC1 pages is 148 KiB. The
/// number bounds the floor's whole cost — `visible surfaces × 16` — which is what
/// lets it be claimed unconditionally, before any feedback, on every frame.
///
/// It is **not** the sharpness ceiling: the GPU feedback marks the level the
/// footprint justifies with no such cap, and the budget decides how much of that
/// is served. This is the floor below which residency never falls.
pub const VT_FLOOR_MAX_TILES: u32 = 16;

/// The most tiles **one feedback request** may mark, handed to the shader.
///
/// A bound on the compute pass's inner loop rather than on quality: a level with
/// more tiles than this is refused in favour of the next coarser one, so a
/// single 8192² texture filling the screen asks for a level that can plausibly
/// be served instead of for four thousand pages that will all be deferred. 256
/// tiles is 4 Mtexels of unique detail for one surface.
pub const VT_FEEDBACK_MAX_TILES: u32 = 256;

/// One drawn surface that names a virtual texture (P26.4).
///
/// Deliberately a **bounding sphere and a texture set**, not a mesh: what the
/// floor and the feedback both need is "how much of the screen does this surface
/// cover", and every path — rigid primitive, meshlet asset, skinned character —
/// can answer that from its transform and its bounds without the projector
/// learning anything about virtual texturing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VtCoverage {
    /// Render-local centre (metres).
    pub centre: Vec3,
    /// Bounding-sphere radius (metres).
    pub radius: f32,
    /// The three slots this surface samples (`handle + 1`, 0 = none).
    pub set: VtTextureSet,
    /// Whether this surface is drawn by the **meshlet** path (P28.1).
    ///
    /// It exists so `feedback_requests` can hand the meshlet set to the
    /// per-FRAGMENT producer instead of marking it per surface — which is what
    /// makes `docs/memos/p26-4-feedback-mechanism.md`'s three losses actually
    /// *recovered* rather than merely also-covered. A union of a coarse mark and
    /// a precise one is the coarse mark. `analytic_floor` ignores this field
    /// entirely: the floor covers every surface whatever draws it, which is the
    /// property that lets a dropped mask degrade to exactly the analytic
    /// residency.
    pub vgeom: bool,
}

/// Pop-in instrumentation (P26.4, clause 5) — **per tile class**, because the two
/// classes fail differently and one number hides which.
///
/// A *floor* tile at fallback is a budget failure: the level's conservative,
/// bounded claim did not fit, so the surface is blurrier than the floor promises.
/// A *refinement* tile at fallback is the ordinary steady state on the way to
/// sharp — it is what pop-in **is** — and the interesting reading is how long it
/// lasts.
///
/// All counters are frame-summed `(frame, want)` pairs, so "how many frames did
/// this scene spend at fallback" is a ratio against [`wants`](Self::floor_wants).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VtPopIn {
    /// Floor wants offered, summed over frames.
    pub floor_wants: u64,
    /// …of which resolved to an ancestor rather than to themselves.
    pub floor_fallback: u64,
    /// Refinement wants offered, summed over frames.
    pub refine_wants: u64,
    /// …of which resolved to an ancestor rather than to themselves.
    pub refine_fallback: u64,
    /// Frames in which a feedback mask arrived from the ring.
    pub feedback_frames: u64,
    /// Frames in which it did not, and the floor stood alone.
    pub feedback_misses: u64,
    /// Wants the residency could not seat, summed over frames.
    pub deferred: u64,
    /// Pages admitted, summed over frames — the **anti-vacuity** number: a
    /// streaming loop that never admits anything also never reports a fallback.
    pub admits: u64,
    /// Frames this streamer has run.
    pub frames: u64,
    /// **Pages actually written into the atlas**, summed over frames (P28.3) —
    /// `VtApplyReport::pages`, which reached a caller per call and was summed
    /// nowhere.
    ///
    /// The P26.5 routing asked for *"re-upload regressions visible to
    /// counters"*, and this is the pair that makes them visible: `admits` is
    /// what residency decided and this is what the queue paid for. They agree
    /// on a healthy frame and diverge exactly when the mirror writes a page
    /// residency did not newly admit — a pool re-created, a slot re-staged, a
    /// growth's full re-upload — which is the regression class nothing in this
    /// tree could name before.
    pub page_uploads: u64,
    /// Bytes of page data written, summed over frames — the same signal in the
    /// unit a VRAM budget is spent in.
    pub page_upload_bytes: u64,
    /// **Frames in which at least one floor want was at fallback** (P28.4) —
    /// *the* A/B counter.
    ///
    /// A third reading of the same evidence, and the one the ROADMAP's clause
    /// asks for by name (*"fallback-frame counters strictly reduced"*).
    /// [`floor_fallback`](Self::floor_fallback) is summed over `(frame, want)`
    /// pairs, so it moves when a *big* surface pops as well as when a frame
    /// pops, and a predictor that halved the tiles of one bad frame would read
    /// like one that removed a frame. This counts frames, so it answers the
    /// question a player asks.
    pub floor_fallback_frames: u64,
    /// Speculative wants offered, summed over frames (P28.4) — the predictor's
    /// **anti-vacuity** number: a predictor wired to an empty history and one
    /// switched off produce the same residency, and only this tells them apart.
    pub predict_wants: u64,
    /// …of which resolved to an ancestor rather than to themselves.
    ///
    /// Not a failure: a speculative want is *expected* to be at fallback the
    /// frame it is first offered — that is what prefetching means. What it
    /// measures is how much of the speculation is still outstanding, and a ratio
    /// that never falls is a horizon the budget cannot serve.
    pub predict_fallback: u64,
    /// **Frames in which at least one refinement want was at fallback** (P28.4)
    /// — the *other* A/B counter, and the one a whip-pan actually moves.
    ///
    /// The floor's camera-driven level is subsumed by the pinned camera-free
    /// floor on a square pyramid ([`VT_PREDICT_MAX_TILES`]), so what a turning
    /// camera changes is which *refinements* are wanted — which is what
    /// `VtPopIn`'s own header calls "what pop-in **is**". A frame counter over
    /// it answers the question a player asks: how often did I look at something
    /// blurry.
    pub refine_fallback_frames: u64,
}

impl VtPopIn {
    /// A one-line human summary, in the shape `VtStats::summary` and
    /// `TerrainStreamStats::summary` already ship so three streamers read alike
    /// in one log.
    pub fn summary(&self) -> String {
        format!(
            "vt streaming: {} frames, {} admits ({} uploads, {:.2} MiB), {} deferred, \
             floor {}/{} at fallback over {} frames, refine {}/{} at fallback over \
             {} frames, predict {}/{} at fallback, feedback {} landed / {} missed",
            self.frames,
            self.admits,
            self.page_uploads,
            self.page_upload_bytes as f64 / (1024.0 * 1024.0),
            self.deferred,
            self.floor_fallback,
            self.floor_wants,
            self.floor_fallback_frames,
            self.refine_fallback,
            self.refine_wants,
            // P28.5: `refine_fallback_frames` reached this line in NEITHER of the
            // two batches that added it and its sibling. Its own doc calls it
            // "the *other* A/B counter, and the one a whip-pan actually moves",
            // and the summary printed the floor's frame count beside it and not
            // its own — so the counter a player's question maps onto was the one
            // number the host's line did not carry.
            self.refine_fallback_frames,
            self.predict_fallback,
            self.predict_wants,
            self.feedback_frames,
            self.feedback_misses,
        )
    }
}

/// **Which mip level a surface's screen footprint justifies** — the ONE rule, on
/// the CPU, mirrored by `vt_feedback.wgsl`.
///
/// `extent` is the texture's mip-0 long side in texels, `screen_px` the diameter
/// of the surface's projected bounding sphere in pixels. A level of `extent /
/// 2^L` texels covering `screen_px` pixels is right when its texels land about
/// one per pixel, so `L = ceil(log2(extent / screen_px))`, clamped into the
/// pyramid.
///
/// `ceil` and not `round`: erring coarse costs blur, erring fine costs pages
/// that will be deferred and a want set that never converges.
pub fn justified_mip(extent: u32, screen_px: f32, mip_count: u32) -> u32 {
    if mip_count == 0 {
        return 0;
    }
    let want = (extent.max(1) as f32 / screen_px.max(1.0)).log2().ceil();
    want.clamp(0.0, (mip_count - 1) as f32) as u32
}

/// The projected diameter, in pixels, of a bounding sphere at `centre` — the
/// quantity both the floor and the shader derive their level from.
///
/// `proj_scale` is pixels per world unit at one metre
/// (`0.5 · height / tan(fov_y / 2)`); an orthographic view has no perspective
/// divide, so it uses its own constant scale and the distance term drops out.
pub fn screen_diameter_px(centre: Vec3, radius: f32, eye: Vec3, proj_scale: f32) -> f32 {
    let dist = (centre - eye).length().max(1e-4);
    2.0 * radius * proj_scale / dist
}

/// Pixels per world unit at one metre for `view`.
pub fn projection_scale(view: &RenderView) -> f32 {
    match view.ortho {
        // Orthographic: the whole frame is `2 · half_height` metres tall, so the
        // scale is constant and the caller's distance term is inert (it divides
        // by `dist` and this multiplies by it — see `screen_diameter_px`, where
        // an ortho view is handled by making the product distance-free).
        Some(o) => 0.5 * view.height as f32 / o.half_height.max(1e-4),
        None => 0.5 * view.height as f32 / (view.fov_y * 0.5).tan().max(1e-4),
    }
}

/// **The NDC margin a bounding sphere earns** — the ONE expression [`on_screen`]
/// and `vt_feedback.wgsl`'s frustum test are both handed (P26.4 audit).
///
/// `screen_px` is the sphere's projected **diameter** (what
/// [`screen_diameter_px`] returns) and `half_height_px` is half the viewport
/// height, so the result is the sphere's screen extent in NDC units — generous
/// by a factor of two against a strict radius, deliberately, because a
/// conservative test wants to keep too much.
///
/// It is a named function rather than an inline division because the two sides
/// had *drifted*: the shader used `r_px / half_height`, half of this, so there
/// was a band at the edge of the frustum the floor claimed and the feedback
/// dropped — a surface at the screen edge paid for its floor pages and was never
/// refined, with no counter moving. The shader spells this
/// `2.0 * r_px / max(f32(params.counts.w), 1.0)`, where `r_px` is the RADIUS in
/// pixels, and `the_feedback_and_the_floor_agree_about_what_is_on_screen` sweeps
/// the two against each other on a real device.
#[inline]
pub fn ndc_margin(screen_px: f32, half_height_px: f32) -> f32 {
    screen_px / half_height_px.max(1.0)
}

/// **Is this surface on screen?** — the camera term of the analytic floor, and
/// the CPU twin of the shader's frustum test.
///
/// A conservative sphere-vs-frustum test in clip space: behind the eye is out,
/// **unless the sphere still straddles it**, and outside the NDC box by more than
/// [`ndc_margin`] is out. Conservative in the "keeps too much" direction, which is
/// the correct direction for a floor.
///
/// The straddle case is not a nicety and the shader used to lack it (P26.4
/// audit): a camera standing on a terrain-sized quad is *inside* that quad's
/// bounding sphere, so a bare `clip.w <= 0` return means the largest surface on
/// screen is the one that never gets refined. `vt_feedback.wgsl` mirrors both
/// branches now.
pub fn on_screen(view_proj: &Mat4, centre: Vec3, radius: f32, ndc_margin: f32) -> bool {
    let clip = *view_proj * centre.extend(1.0);
    if clip.w <= 0.0 {
        // Behind the eye — unless the sphere still straddles it.
        return radius > -clip.w;
    }
    let ndc = clip.truncate() / clip.w;
    ndc.x.abs() <= 1.0 + ndc_margin && ndc.y.abs() <= 1.0 + ndc_margin
}

/// **Gather every drawn surface that names a virtual texture** (P26.4).
///
/// Reads the render scene the frame is about to draw — rigid instances, meshlet
/// instances and skinned characters — so the coverage is a function of what is
/// actually being submitted rather than of what the document contains. A surface
/// whose set is [`VtTextureSet::NONE`] contributes nothing, so a textureless
/// scene produces an empty list and the whole loop below is skipped.
///
/// Bounds are the honest conservative ones each path can offer: a built-in
/// primitive is a unit shape, so its sphere is `√3/2 · max(scale)`; a meshlet
/// asset carries its own bounding sphere in its header (read without paging
/// anything); a skinned character has no cheap bound, so it takes the primitive
/// rule against its instance scale — which is why a character's floor is
/// conservative rather than tight.
pub fn scene_coverage(scene: &RenderScene, origin: &inf_math::FloatingOrigin) -> Vec<VtCoverage> {
    /// Half the diagonal of a unit cube — the bounding sphere of every built-in
    /// primitive at unit scale.
    const UNIT_RADIUS: f32 = 0.866_025_4;

    let mut out = Vec::new();
    for i in &scene.instances {
        if i.vt.is_none() {
            continue;
        }
        out.push(VtCoverage {
            centre: origin.to_render(i.translation),
            radius: UNIT_RADIUS * i.scale.abs().max_element().max(1e-4),
            set: i.vt,
            vgeom: false,
        });
    }
    let bounds: BTreeMap<u128, f32> = scene
        .vgeom_assets
        .iter()
        .map(|a| (a.id, a.bounds().1))
        .collect();
    for i in &scene.vgeom_instances {
        if i.vt.is_none() {
            continue;
        }
        let r = bounds.get(&i.asset).copied().unwrap_or(UNIT_RADIUS);
        out.push(VtCoverage {
            centre: origin.to_render(i.translation),
            radius: r * i.scale.abs().max_element().max(1e-4),
            set: i.vt,
            vgeom: true,
        });
    }
    for i in &scene.skinned {
        if i.vt.is_none() {
            continue;
        }
        out.push(VtCoverage {
            centre: origin.to_render(i.translation),
            radius: UNIT_RADIUS * i.scale.abs().max_element().max(1e-4),
            set: i.vt,
            vgeom: false,
        });
    }
    out
}

/// **The analytic want-set floor** (P26.4, clause 4): `(camera, bounds, registry)`
/// → a conservative tile set, computed every frame.
///
/// Two parts, and the first is the reason the second can be bounded:
///
/// * the **camera-free** part is `VtTextures::want_floor` — the three coarsest
///   levels of every registered texture, at most 21 pages each however large the
///   texture. It is what guarantees a sample is never a hole even for a surface
///   that is off screen, behind the camera, or not drawn this frame at all.
/// * the **camera-driven** part adds, for every *visible* surface, the finest
///   level of each of its three maps that fits [`VT_FLOOR_MAX_TILES`] and is no
///   finer than [`justified_mip`] allows. Bounded by construction, so the floor's
///   cost is `visible surfaces × 16` pages and cannot be made unbounded by a
///   scene.
///
/// Every want is [`VT_PRIORITY_FLOOR`], which is what makes it a floor: the
/// residency serves the whole class before a refinement is offered a slot.
///
/// **Deterministic in committed input.** Nothing here reads a clock, a frame
/// counter or a previous frame's state — two runs of one scripted camera path
/// over one scene produce one want sequence, which is what the phase gate pins.
pub fn analytic_floor(lib: &VtTextures, view: &RenderView, coverage: &[VtCoverage]) -> Vec<VtWant> {
    let mut out = lib.want_floor();
    out.extend(camera_wants(
        lib.residency(),
        view,
        coverage,
        VT_PRIORITY_FLOOR,
        VT_FLOOR_MAX_TILES,
    ));
    out
}

/// The most tiles **one speculative want** may claim for one surface (P28.4).
///
/// **Exactly [`VT_FEEDBACK_MAX_TILES`]**, and the equality is a ruling reached
/// by refuting the other two candidates with a measurement rather than by
/// preferring this one.
///
/// # A prefetch must speak the language of the class it prefetches for
///
/// A tile is an *address* — `(texture, mip, x, y)` — so a speculation at a
/// different cap settles on a different **mip** and shares not one tile with
/// what the future want will ask for. Whatever this lane prefetches for, it must
/// use that class's cap exactly. So the question is only *which class*.
///
/// # And the floor cannot be the answer, because it cannot be prefetched
///
/// `VtResidency::apply_wants` admits a miss the frame it is offered, out of the
/// same pool, with no per-frame admission throttle anywhere in the loop
/// (`VT_ADMITS_PER_FRAME_CEILING` is a *gate ceiling*, not a governor). So the
/// floor's fallback count is `max(0, demand − pool)`: under the pool nothing
/// misses, over it the shortfall is the arithmetic difference, and in **neither
/// regime does having asked earlier change the number**. Measured, not argued —
/// `whip_pan::a_saturated_floor_cannot_be_prefetched_and_the_arm_says_so` runs
/// the whole 360° path over an undersized pool and the two arms come out
/// byte-identical on every floor counter.
///
/// What *does* lag the camera is the **GPU refinement**: it is marked off a
/// depth buffer, so it can only ever ask for surfaces that are already visible,
/// and it arrives `READBACK_LATENCY_FRAMES` later than that. That gap is what
/// pop-in is (`VtPopIn`'s own header says so), it is the only thing in this
/// subsystem a prediction can close, and closing it needs the refinement's own
/// cap.
pub const VT_PREDICT_MAX_TILES: u32 = VT_FEEDBACK_MAX_TILES;

/// **What a camera justifies** — one footprint rule, at an arbitrary camera, at
/// an arbitrary lane, under an arbitrary per-surface cap (P28.4).
///
/// This is [`analytic_floor`]'s camera-driven half, lifted so the three want
/// classes stop being three copies of it. Every class is *this* rule with two
/// numbers changed, which is the whole design:
///
/// | class | camera | lane | cap |
/// |---|---|---|---|
/// | analytic floor | committed | `VT_PRIORITY_FLOOR` | [`VT_FLOOR_MAX_TILES`] (16) |
/// | GPU refinement | committed | `VT_PRIORITY_FEEDBACK` | [`VT_FEEDBACK_MAX_TILES`] (256) |
/// | speculation | **predicted** | `VT_PRIORITY_PREDICT` | [`VT_PREDICT_MAX_TILES`] (256, = the refinement's) |
///
/// The two existing classes already differed only by those two numbers and said
/// so in prose (*"Both halves compute the level from the same rule"*); the third
/// is what made writing it down worth doing. A predictor with a footprint rule
/// of its own is not prefetching, it is streaming something else, and the
/// failure is silent because both want sets look plausible alone.
///
/// The camera-free part ([`VtTextures::want_floor`]) is deliberately *not* here:
/// it is a property of the registry, so a predicted camera adds nothing to it
/// and re-emitting it would be a duplicate the dedup then has to eat.
///
/// **Deterministic in committed input.** Nothing here reads a clock, a frame
/// counter or a previous frame's state, so two runs of one scripted camera path
/// over one scene produce one want sequence — which holds for the predicted
/// camera exactly as it holds for the committed one, because
/// `inf_math::dead_reckon` is itself a pure function of the committed history.
pub fn camera_wants(
    res: &inf_vt::VtResidency,
    view: &RenderView,
    coverage: &[VtCoverage],
    lane: inf_vt::VtPriority,
    max_tiles: u32,
) -> Vec<VtWant> {
    let mut out = Vec::new();
    if coverage.is_empty() {
        return out;
    }
    let view_proj = view.view_proj();
    let proj_scale = projection_scale(view);
    let eye = view.eye_local();
    let half_h = (view.height as f32 * 0.5).max(1.0);
    for c in coverage {
        let px = screen_diameter_px(c.centre, c.radius, eye, proj_scale);
        if !on_screen(&view_proj, c.centre, c.radius, ndc_margin(px, half_h)) {
            continue;
        }
        for slot in c.set.slots() {
            if slot == 0 {
                continue;
            }
            let handle = VtTextureHandle(slot - 1);
            let Some(desc) = res.desc(handle) else {
                continue;
            };
            let extent = desc.mips[0].width.max(desc.mips[0].height);
            let mut lv = justified_mip(extent, px, desc.mip_count());
            // Coarser until the level fits this class's per-surface cap.
            while lv + 1 < desc.mip_count() && desc.mips[lv as usize].tile_count() > max_tiles {
                lv += 1;
            }
            let m = desc.mips[lv as usize];
            for y in 0..m.tiles_y {
                for x in 0..m.tiles_x {
                    out.push(VtWant::new(handle, TileCoord::new(lv, x, y)).with_priority(lane));
                }
            }
        }
    }
    out
}

/// **The view the predictor says is coming** — the committed view with its eye
/// and frame replaced, everything else kept (P28.4).
///
/// The floating **origin** is deliberately the committed one: `VtCoverage`
/// centres are render-local against it, so a predicted view with a predicted
/// origin would compare a predicted eye against surfaces expressed in another
/// frame. The cost is that the predicted eye is up to `horizon × speed` metres
/// further from the origin than the real one — 15 m at 50 m/s and a 300 ms
/// horizon, against `inf_math::REBASE_DISTANCE`'s 1 024 — so the f32 the
/// footprint rule works in is unaffected.
///
/// A degenerate predicted direction falls back to the committed one rather than
/// building a view matrix out of a zero vector.
pub fn predicted_view(view: &RenderView, p: &inf_math::Prediction) -> RenderView {
    let forward = p.forward.as_vec3().normalize_or_zero();
    let up = p.up.as_vec3().normalize_or_zero();
    RenderView {
        eye_world: p.eye,
        forward: if forward.length_squared() > 0.0 {
            forward
        } else {
            view.forward
        },
        up: if up.length_squared() > 0.0 {
            up
        } else {
            view.up
        },
        ..*view
    }
}

/// **The speculative want set** (P28.4, clause 2): the one footprint rule, asked
/// at the predicted camera, at [`VT_PRIORITY_PREDICT`](inf_vt::VT_PRIORITY_PREDICT),
/// under [`VT_PREDICT_MAX_TILES`].
///
/// It predicts the **refinement** an arriving surface will need, not the floor,
/// and that is a measured ruling rather than a preference: the floor is admitted
/// the frame it is asked and cannot be prefetched at all. See
/// [`VT_PREDICT_MAX_TILES`].
///
/// Every one of these wants is strictly below both the floor and the feedback,
/// so the whole set is free in the only sense that matters: it can take a slot
/// nothing better wants, and it can never keep a slot something better wants.
/// A prediction that is wrong therefore costs bytes that were idle and nothing
/// else — which is what makes a 200–500 ms horizon a measurement rather than a
/// risk.
pub fn speculative_wants(
    res: &inf_vt::VtResidency,
    view: &RenderView,
    coverage: &[VtCoverage],
    prediction: &inf_math::Prediction,
) -> Vec<VtWant> {
    camera_wants(
        res,
        &predicted_view(view, prediction),
        coverage,
        inf_vt::VT_PRIORITY_PREDICT,
        VT_PREDICT_MAX_TILES,
    )
}

/// One feedback request as the compute shader reads it — 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FeedbackRequest {
    /// xyz = render-local centre, w = radius (metres).
    pub centre: [f32; 4],
    /// x = word offset of the texture's block in the table, y = its first bit in
    /// the mask, zw = reserved.
    pub tex: [u32; 4],
}

/// The uniform the feedback pass reads — 96 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FeedbackParams {
    pub view_proj: [f32; 16],
    /// xyz = render-local eye, w = pixels per world unit at one metre.
    pub eye: [f32; 4],
    /// x = request count, y = mask words, z = max tiles per request, w = half
    /// the viewport height in pixels.
    pub counts: [u32; 4],
}

/// **Build the feedback request list** for one frame — the same coverage the
/// analytic floor used, expanded to one entry per (surface × bound slot) and
/// resolved into table offsets on the CPU.
///
/// The block offset and the base bit are computed here rather than in the shader
/// because both are already known exactly: `VtResidency::table_block` gives the
/// first, [`VtFeedbackLayout::texture_base`] the second, and a shader that
/// re-derived either would be a second copy of the layout.
///
/// Ordered by `(slot, index)` — a stable function of the coverage list, which is
/// itself a stable function of the scene. The mask does not care (OR is
/// order-independent), and a deterministic upload keeps the *command stream*
/// comparable between runs, which the golden harness does care about.
///
/// **`skip_vgeom` is P28.1's handover.** With the visibility buffer on, meshlet
/// surfaces are marked per FRAGMENT by `vis_feedback.wgsl` — with occlusion, with
/// the uv extent a pixel actually reached, and with a level per screen region
/// rather than one per surface. Leaving their per-surface requests in as well
/// would put the coarse marks back into the same mask, and a union with a coarse
/// mark is a coarse mark. The **floor** is untouched either way, so a frame that
/// refuses the visibility path, or a mask that never arrives, still lands on
/// exactly the analytic residency.
pub fn feedback_requests(
    lib: &VtTextures,
    layout: &VtFeedbackLayout,
    coverage: &[VtCoverage],
    skip_vgeom: bool,
) -> Vec<FeedbackRequest> {
    let mut out = Vec::new();
    for c in coverage {
        if skip_vgeom && c.vgeom {
            continue;
        }
        for slot in c.set.slots() {
            if slot == 0 {
                continue;
            }
            let handle = VtTextureHandle(slot - 1);
            let (Some((block, _)), Some(base)) = (
                lib.residency().table_block(handle),
                layout.texture_base(handle),
            ) else {
                continue;
            };
            out.push(FeedbackRequest {
                centre: [c.centre.x, c.centre.y, c.centre.z, c.radius],
                tex: [block as u32, base, 0, 0],
            });
        }
    }
    out
}

/// Split a want list into the floor and refinement classes — what the pop-in
/// counters are summed over.
pub(crate) fn count_fallbacks(lib: &VtTextures, wants: &[VtWant], stats: &mut VtPopIn) {
    let mut floor_missed = false;
    let mut refine_missed = false;
    for w in wants {
        let at_fallback = !lib.residency().is_resident(w.texture, w.tile);
        // **A match and not an `else`** (P28.4). Before the predictor existed
        // the two classes were exhaustive and the `else` was harmless; with a
        // third lane it would count every speculative want as a floor want, so
        // switching the predictor on would *raise* `floor_wants` and
        // `floor_fallback` — the two numbers the A/B arm reads to decide whether
        // it helped. The gate would have measured its own instrument.
        match w.priority {
            inf_vt::VT_PRIORITY_FEEDBACK => {
                stats.refine_wants += 1;
                stats.refine_fallback += u64::from(at_fallback);
                refine_missed |= at_fallback;
            }
            inf_vt::VT_PRIORITY_PREDICT => {
                stats.predict_wants += 1;
                stats.predict_fallback += u64::from(at_fallback);
            }
            _ => {
                stats.floor_wants += 1;
                stats.floor_fallback += u64::from(at_fallback);
                floor_missed |= at_fallback;
            }
        }
    }
    stats.floor_fallback_frames += u64::from(floor_missed);
    stats.refine_fallback_frames += u64::from(refine_missed);
}

/// The GPU half of the loop: the coverage bitmask, its compute pass, and the
/// readback ring that reads it at a pinned latency.
pub struct VtFeedback {
    layout: VtFeedbackLayout,
    mask: wgpu::Buffer,
    params: wgpu::Buffer,
    requests: wgpu::Buffer,
    request_cap: u32,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    bind: Option<(u64, wgpu::BindGroup)>,
    ring: ReadbackRing,
}

impl VtFeedback {
    /// A feedback pass sized for `layout`.
    pub fn new(device: &wgpu::Device, layout: VtFeedbackLayout, request_cap: u32) -> Self {
        let words = layout.words() as u64;
        let mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt-feedback-mask"),
            size: words * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt-feedback-params"),
            size: std::mem::size_of::<FeedbackParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let requests = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt-feedback-requests"),
            size: (request_cap.max(1) as u64) * std::mem::size_of::<FeedbackRequest>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vt-feedback"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vt_feedback.wgsl").into()),
        });
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vt-feedback"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(3, storage(false)),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vt-feedback"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vt-feedback"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("cs_feedback"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            mask,
            params,
            requests,
            request_cap: request_cap.max(1),
            pipeline,
            bgl,
            bind: None,
            ring: ReadbackRing::new(device, "vt-feedback", words * 4),
        }
    }

    /// The bitmask layout this pass writes.
    #[inline]
    pub fn layout(&self) -> &VtFeedbackLayout {
        &self.layout
    }

    /// The ring, for a host or a gate that wants its hit/miss counts.
    #[inline]
    pub fn ring(&self) -> &ReadbackRing {
        &self.ring
    }

    /// **Read frame `frame`'s feedback** — the mask recorded at `frame − 2`, or
    /// `None`. Never an adjacent frame: see [`crate::readback`].
    pub fn take_wants(
        &mut self,
        device: &wgpu::Device,
        lib: &VtTextures,
        frame: u64,
    ) -> Option<Vec<VtWant>> {
        self.ring.poll(device);
        let layout = self.layout.clone();
        let residency = lib.residency();
        self.ring.take(frame, |bytes| {
            let words: &[u32] = bytemuck::cast_slice(bytes);
            layout.wants(residency, words)
        })
    }

    /// **Record the next feedback pass** into `encoder` and hand its buffer to
    /// the ring.
    ///
    /// The mask is cleared first, in the same encoder, so a frame's coverage is
    /// this frame's and never an OR with the last one — which would make the
    /// signal depend on the frame history, the exact property the pinned ring
    /// exists to avoid.
    ///
    /// Returns the number of requests dispatched (0 = nothing recorded, and the
    /// ring gets no copy, so the read two frames later misses and the floor
    /// stands — the same degradation as a late mask).
    ///
    /// The frame index left with the ring copy in P28.1 — this function no
    /// longer hands anything to the ring, so a frame number here would be an
    /// argument nothing reads. See [`finish`](Self::finish).
    ///
    /// Eight arguments, and bundling them would hide the two that matter. The
    /// table and its **generation** travel together on purpose: a bind group
    /// cached across a re-creation of the indirection buffer marks bits against
    /// a table from before the new texture existed, which is the same hazard
    /// `passes::ResourceKey` carries its fourth component for (that key is a
    /// bare `(u64, u64, u64, u64)` and the component is fed by
    /// `VtPools::table_generation` — named precisely here because the P26.4
    /// audit found the field-style citation ungreppable). A
    /// struct is exactly where a field like that becomes easy to forget to fill
    /// (the P21.4 note on `run_pie`, one crate over).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        table: &wgpu::Buffer,
        table_generation: u64,
        view: &RenderView,
        requests: &[FeedbackRequest],
    ) -> u32 {
        let count = (requests.len() as u32).min(self.request_cap);
        // Cleared UNCONDITIONALLY (P28.1), and before the early return. A second
        // producer marks into this buffer from inside the render graph, so "no
        // per-surface requests this frame" must still mean "the mask starts from
        // zero" — otherwise a scene whose only textured surfaces are meshlets
        // would OR its per-fragment marks into whatever the last frame left.
        encoder.clear_buffer(&self.mask, 0, None);
        if count == 0 {
            return 0;
        }
        queue.write_buffer(
            &self.requests,
            0,
            bytemuck::cast_slice(&requests[..count as usize]),
        );
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&FeedbackParams {
                view_proj: view.view_proj().to_cols_array(),
                eye: {
                    let e = view.eye_local();
                    [e.x, e.y, e.z, projection_scale(view)]
                },
                counts: [
                    count,
                    self.layout.words() as u32,
                    VT_FEEDBACK_MAX_TILES,
                    (view.height / 2).max(1),
                ],
            }),
        );
        // The indirection buffer is RE-CREATED on a registration, so a bind group
        // cached across one keeps the old allocation alive and the pass would mark
        // bits against a table from before the new texture existed — the same
        // hazard `passes::ResourceKey`'s fourth component — fed by
        // `VtPools::table_generation` — exists for, one pass over.
        if self.bind.as_ref().map(|(g, _)| *g) != Some(table_generation) {
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vt-feedback"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.requests.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.mask.as_entire_binding(),
                    },
                ],
            });
            self.bind = Some((table_generation, bind));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vt-feedback"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind.as_ref().expect("just built").1, &[]);
            pass.dispatch_workgroups(count.div_ceil(64).max(1), 1, 1);
        }
        count
    }

    /// **Hand the mask to the ring**, after every producer has marked into it.
    ///
    /// Split out of [`record`](Self::record) in P28.1, and the split is the whole
    /// mechanism by which a SECOND producer became possible. `record` runs at the
    /// frame's sync point, before the render graph; the per-fragment marker
    /// (`vis_feedback.wgsl`) runs inside it, from a visibility buffer that does
    /// not exist yet when `record` is called. With the copy still inside `record`
    /// the per-fragment marks would land in the mask *after* it had been read,
    /// and would reach the streamer one frame late or not at all.
    ///
    /// So the sequence is now clear → per-surface dispatch → (graph, including
    /// the per-fragment dispatch) → copy, all in **one encoder and one submit**.
    /// The latency the ring pins is unchanged: frame `k`'s mask is still read at
    /// `k + 2` and never "whenever it resolves".
    ///
    /// `produced` is whether **any** producer marked into the mask this frame.
    ///
    /// It is deliberately not `record`'s return value alone, and the difference
    /// is a defect this batch's own gate caught: with the visibility path on, the
    /// meshlet set is handed to the per-FRAGMENT producer, so a scene whose only
    /// textured surfaces are meshlets dispatches **zero** per-surface requests —
    /// and gating the ring copy on that number means the mask the per-pixel pass
    /// just filled is never read, on exactly the scenes the per-pixel pass exists
    /// for. It presents as a streamer that silently sits on the analytic floor
    /// for ever, which is the quietest failure this subsystem has.
    ///
    /// `false` means nothing marked, the ring gets no copy, and the read two
    /// frames later misses — the same degradation as a late mask, and the
    /// behaviour `record` had when it owned this line.
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder, frame: u64, produced: bool) {
        if !produced {
            return;
        }
        self.ring.record(encoder, &self.mask, frame);
    }

    /// The mask buffer itself — what a second producer marks into.
    #[inline]
    pub fn mask(&self) -> &wgpu::Buffer {
        &self.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── P28.4: the speculative lane's producer ───────────────────────────────

    /// A view looking down `forward` from `eye`, 512 px tall.
    fn look(eye: glam::DVec3, forward: Vec3) -> RenderView {
        RenderView {
            origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
            eye_world: eye,
            forward: forward.normalize(),
            up: Vec3::Y,
            fov_y: 60_f32.to_radians(),
            near: 0.1,
            width: 512,
            height: 512,
            ortho: None,
        }
    }

    /// One 1 024² texture in an otherwise empty residency.
    fn one_texture() -> (inf_vt::VtResidency, VtTextureHandle) {
        let (mut res, _adv) = inf_vt::VtResidency::new(inf_vt::VtPoolConfig::default());
        let h = res
            .register_texture(inf_vt::full_pyramid(1024, 1024, 128, 4, true))
            .expect("the floor fits the default budget");
        (res, h)
    }

    fn cover(centre: Vec3, h: VtTextureHandle) -> VtCoverage {
        VtCoverage {
            centre,
            radius: 1.0,
            set: crate::scene::VtTextureSet {
                albedo: h.0 + 1,
                ..crate::scene::VtTextureSet::NONE
            },
            vgeom: false,
        }
    }

    /// **One rule, three classes** — the property that would break if the
    /// predictor grew a footprint rule of its own: at one camera and one cap,
    /// two lanes produce the same addresses in the same order and differ only in
    /// the rank.
    #[test]
    fn a_speculative_want_is_the_floors_own_rule_at_another_lane() {
        let (res, h) = one_texture();
        let cov = [cover(Vec3::new(0.0, 0.0, -6.0), h)];
        let view = look(glam::DVec3::ZERO, Vec3::NEG_Z);

        let floor = camera_wants(
            &res,
            &view,
            &cov,
            inf_vt::VT_PRIORITY_FLOOR,
            VT_FLOOR_MAX_TILES,
        );
        let spec = camera_wants(
            &res,
            &view,
            &cov,
            inf_vt::VT_PRIORITY_PREDICT,
            VT_FLOOR_MAX_TILES,
        );
        assert!(!floor.is_empty(), "the fixture wants nothing at all");
        assert_eq!(floor.len(), spec.len());
        for (f, s) in floor.iter().zip(&spec) {
            assert_eq!((f.texture, f.tile), (s.texture, s.tile));
            assert_eq!(f.priority, inf_vt::VT_PRIORITY_FLOOR);
            assert_eq!(s.priority, inf_vt::VT_PRIORITY_PREDICT);
        }
    }

    /// **THE MEASUREMENT THAT CHOSE THE CAP**, and it is a refutation of the
    /// obvious choice rather than a preference between two good ones.
    ///
    /// "A guess should claim less than a proof" argues for a speculative cap
    /// between the floor's 16 and the feedback's 256. It is wrong, and the
    /// reason is that a tile is an *address*: a different cap picks a different
    /// **mip**, and two mips share no tile. A speculation at any cap but the one
    /// belonging to the class it prefetches for fills the pool with pages that
    /// class never asks for — a prefetch that prefetches something else.
    ///
    /// Swept over five square pyramids. Where the caps bind at different levels
    /// the two address sets are **disjoint**, and at least one extent in the
    /// sweep must exhibit it or this arm is about a fixture too small to
    /// separate them. The shipped cap is the refinement's, so the shipped set
    /// is the refinement's set exactly — see [`VT_PREDICT_MAX_TILES`] for why
    /// the floor is not the class a prediction can serve.
    #[test]
    fn a_prediction_at_a_finer_cap_names_addresses_the_floor_will_never_ask_for() {
        let mut separated = 0usize;
        for extent in [512u32, 1024, 2048, 4096, 8192] {
            let (mut res, _adv) = inf_vt::VtResidency::new(inf_vt::VtPoolConfig::default());
            let h = res
                .register_texture(inf_vt::full_pyramid(extent, extent, 128, 4, true))
                .expect("the floor fits");
            // A LARGE surface, close: the cap only binds when the footprint
            // justifies a level finer than the cap allows, and a 1 m sphere at
            // three metres in a 512 px viewport justifies mip 5 of an 8 192²
            // texture — where both caps agree and the arm sees nothing. This is
            // a wall filling the frame.
            let cov = [VtCoverage {
                centre: Vec3::new(0.0, 0.0, -2.5),
                radius: 60.0,
                set: crate::scene::VtTextureSet {
                    albedo: h.0 + 1,
                    ..crate::scene::VtTextureSet::NONE
                },
                vgeom: false,
            }];
            let view = look(glam::DVec3::ZERO, Vec3::NEG_Z);
            let at = |cap| -> std::collections::BTreeSet<(u32, u32, u32, u32)> {
                camera_wants(&res, &view, &cov, inf_vt::VT_PRIORITY_FLOOR, cap)
                    .iter()
                    .map(|w| (w.texture.0, w.tile.mip, w.tile.x, w.tile.y))
                    .collect()
            };
            let floor = at(VT_FLOOR_MAX_TILES);
            let finer = at(VT_FEEDBACK_MAX_TILES);
            assert!(!floor.is_empty(), "{extent}: the surface is not visible");
            if floor != finer {
                separated += 1;
                assert!(
                    floor.is_disjoint(&finer),
                    "{extent}: two caps that chose different levels shared a tile"
                );
            }
            // …and the shipped cap is the refinement's, so the speculative set
            // is the refinement's set exactly and the floor's not at all.
            assert_eq!(at(VT_PREDICT_MAX_TILES), finer);
        }
        assert!(
            separated > 0,
            "no extent in the sweep made the two caps disagree — the arm cannot \
             see its own subject"
        );
    }

    /// **The prefetch, as a set difference.** A surface behind the camera is
    /// wanted by nothing; the same surface, at the camera dead reckoning says is
    /// coming, is wanted speculatively — which is the only thing the predictor
    /// is for.
    #[test]
    fn a_surface_the_committed_camera_cannot_see_is_wanted_by_the_predicted_one() {
        let (res, h) = one_texture();
        // Behind and to the left, well outside a 60° frustum looking down −Z.
        let cov = [cover(Vec3::new(-14.0, 0.0, 4.0), h)];
        let view = look(glam::DVec3::ZERO, Vec3::NEG_Z);
        assert!(
            camera_wants(
                &res,
                &view,
                &cov,
                inf_vt::VT_PRIORITY_FLOOR,
                VT_FLOOR_MAX_TILES
            )
            .is_empty(),
            "the fixture is visible already, so it cannot show a prefetch"
        );

        // A camera turning left at 0.06 rad/tick reaches it inside the horizon.
        let mut hist = inf_math::CameraHistory::new();
        for t in 0..6u64 {
            let a = 0.06 * t as f64;
            hist.commit(inf_math::CameraSample {
                tick: t,
                eye: glam::DVec3::ZERO,
                forward: glam::DVec3::new(-inf_math::psin64(a), 0.0, -inf_math::pcos64(a)),
                up: glam::DVec3::Y,
            });
        }
        let p = inf_math::dead_reckon(&hist, 18).expect("six samples");
        let spec = speculative_wants(&res, &view, &cov, &p);
        assert!(
            !spec.is_empty(),
            "the predicted camera reaches nothing the committed one missed"
        );
        assert!(spec
            .iter()
            .all(|w| w.priority == inf_vt::VT_PRIORITY_PREDICT));
    }

    /// **The third class is a class**, not an `else`.
    ///
    /// Before P28.4 the classifier's two arms were exhaustive and its `else` was
    /// harmless. With a third lane the `else` counts every speculative want as a
    /// floor want, so switching the predictor on would *raise* `floor_wants` and
    /// `floor_fallback` — the two numbers the A/B arm reads to decide whether it
    /// helped. The gate would have measured its own instrument.
    #[test]
    fn the_pop_in_counters_keep_the_three_lanes_apart() {
        let (mut lib, _adv) = crate::vt_library::VtTextures::new(inf_vt::VtPoolConfig::default());
        let h = lib
            .residency_mut()
            .register_texture(inf_vt::full_pyramid(1024, 1024, 128, 4, true))
            .expect("the floor fits");
        // Mip 0 is never pinned, so every one of these is at fallback.
        let t = |x| inf_vt::TileCoord::new(0, x, 0);
        let wants = [
            VtWant::new(h, t(0)),
            VtWant::refine(h, t(1)),
            VtWant::speculate(h, t(2)),
        ];
        let mut stats = VtPopIn::default();
        count_fallbacks(&lib, &wants, &mut stats);
        assert_eq!((stats.floor_wants, stats.floor_fallback), (1, 1));
        assert_eq!((stats.refine_wants, stats.refine_fallback), (1, 1));
        assert_eq!((stats.predict_wants, stats.predict_fallback), (1, 1));
        assert_eq!(
            (stats.floor_fallback_frames, stats.refine_fallback_frames),
            (1, 1)
        );

        // A frame whose floor is entirely resident does not count a fallback
        // frame, however much speculation is outstanding — which is what makes
        // the frame counter a statement about what the player sees.
        let root = lib
            .residency()
            .desc(h)
            .expect("registered")
            .mip_count()
            .saturating_sub(1);
        let mut stats = VtPopIn::default();
        count_fallbacks(
            &lib,
            &[
                VtWant::new(h, inf_vt::TileCoord::new(root, 0, 0)),
                VtWant::speculate(h, t(2)),
            ],
            &mut stats,
        );
        assert_eq!(
            stats.floor_fallback, 0,
            "the coarsest level is pinned at registration and must be resident"
        );
        assert_eq!(
            (stats.floor_fallback_frames, stats.refine_fallback_frames),
            (0, 0)
        );
        assert_eq!(stats.predict_fallback, 1);
    }

    /// The level rule: one texel per pixel, erring **coarse**, clamped into the
    /// pyramid at both ends.
    ///
    /// This is the function `vt_feedback.wgsl` mirrors, so it is pinned as a
    /// table rather than described — a drift between the two shows up as a floor
    /// and a feedback that disagree about what a frame needs, which reads as
    /// permanent pop-in rather than as a wrong number.
    #[test]
    fn the_justified_level_is_one_texel_per_pixel_rounded_coarse() {
        // A 1024-texel texture on a 1024-pixel surface: mip 0.
        assert_eq!(justified_mip(1024, 1024.0, 11), 0);
        // Half the screen size: one level coarser.
        assert_eq!(justified_mip(1024, 512.0, 11), 1);
        assert_eq!(justified_mip(1024, 256.0, 11), 2);
        // A sliver: clamped at the coarsest level rather than running off.
        assert_eq!(justified_mip(1024, 0.5, 11), 10);
        // Bigger on screen than it is in texels: mip 0, never negative.
        assert_eq!(justified_mip(1024, 4096.0, 11), 0);
        // Erring coarse: 700 pixels of a 1024 texture is level 1, not level 0.
        assert_eq!(justified_mip(1024, 700.0, 11), 1);
        // A one-level pyramid has one answer.
        assert_eq!(justified_mip(1024, 1.0, 1), 0);
        assert_eq!(justified_mip(1024, 1.0, 0), 0);
    }

    /// The projected diameter falls off as 1/distance and scales with the
    /// radius — the two properties the floor's level depends on.
    #[test]
    fn the_projected_diameter_tracks_distance_and_size() {
        let eye = Vec3::ZERO;
        let near = screen_diameter_px(Vec3::new(0.0, 0.0, -5.0), 1.0, eye, 500.0);
        let far = screen_diameter_px(Vec3::new(0.0, 0.0, -10.0), 1.0, eye, 500.0);
        assert!((near - 200.0).abs() < 1e-3, "{near}");
        assert!((far - near * 0.5).abs() < 1e-3, "{far} vs {near}");
        let big = screen_diameter_px(Vec3::new(0.0, 0.0, -5.0), 2.0, eye, 500.0);
        assert!((big - 2.0 * near).abs() < 1e-3);
        // A surface at the eye does not divide by zero.
        assert!(screen_diameter_px(eye, 1.0, eye, 500.0).is_finite());
    }

    /// **The pop-in line carries every counter the struct has** (P28.5).
    ///
    /// `refine_fallback_frames` landed in P28.4 beside `floor_fallback_frames`
    /// and did not reach this line — so the one counter whose own doc says it
    /// *"answers the question a player asks: how often did I look at something
    /// blurry"* was the one number the host's line did not carry.
    ///
    /// Asserted as a **field count** and not as a list of tokens, because a list
    /// enumerates what you thought of (the P22 law): every field is given a
    /// distinct value and every value has to appear, so a counter added later
    /// and forgotten fails here.
    #[test]
    fn the_pop_in_line_prints_every_counter_it_holds() {
        let p = VtPopIn {
            floor_wants: 11,
            floor_fallback: 12,
            refine_wants: 13,
            refine_fallback: 14,
            feedback_frames: 15,
            feedback_misses: 16,
            deferred: 17,
            admits: 18,
            frames: 19,
            page_uploads: 20,
            page_upload_bytes: 21 * 1024 * 1024,
            floor_fallback_frames: 22,
            predict_wants: 23,
            predict_fallback: 24,
            refine_fallback_frames: 25,
        };
        let s = p.summary();
        for (n, name) in [
            (11, "floor_wants"),
            (12, "floor_fallback"),
            (13, "refine_wants"),
            (14, "refine_fallback"),
            (15, "feedback_frames"),
            (16, "feedback_misses"),
            (17, "deferred"),
            (18, "admits"),
            (19, "frames"),
            (20, "page_uploads"),
            (22, "floor_fallback_frames"),
            (23, "predict_wants"),
            (24, "predict_fallback"),
            (25, "refine_fallback_frames"),
        ] {
            assert!(
                s.contains(&n.to_string()),
                "`{name}` ({n}) is not in the host's line: {s}"
            );
        }
        assert!(s.contains("21.00 MiB"), "{s}");
        // …and the values really are distinguishable, which is what makes the
        // check above a check: fifteen fields, fifteen distinct values.
        let vals = [
            11u64, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        ];
        assert_eq!(
            vals.iter().collect::<std::collections::BTreeSet<_>>().len(),
            15
        );
    }
}
