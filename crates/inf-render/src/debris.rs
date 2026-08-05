//! **Small-debris instancing and the per-tier debris budget (P22.4).**
//!
//! Two things live here, and they are the two halves of "destruction at scale":
//! how the *rubble* a break sheds reaches the GPU, and how much of it a machine
//! is asked to keep.
//!
//! # 1. Rubble is render-only dressing, derived from detach events
//!
//! P22.3 draws every detached **chunk** — a real solver-owned body with its own
//! vertex buffer. That is right for tens of chunks and wrong for the *visual*
//! impression of a building coming down, which is mostly fragments far below the
//! size anything wants a convex hull for. So each live chunk also sheds
//! [`DEBRIS_RUBBLE_PER_CHUNK`] sub-chunk fragments, and they go down the P18.5 GPU
//! scatter path as **one [`ScatterBatch`] per broken actor** rather than one draw
//! each.
//!
//! They are **dressing**, in the strict sense the deformation field's docs use
//! for the render window: derived from sim state, never fed back into it. Nothing
//! here is a body, nothing here is queried, and a host that dropped this module
//! entirely would run a bit-identical simulation.
//!
//! ## The keying problem, and why the sites are FROZEN
//!
//! [`ScatterData::key`] is a content hash over the packed instance bytes
//! (`scene.rs`, "content-addressed since P18.3"). That is exactly right for a
//! foliage stroke and exactly wrong for anything that moves: a batch whose
//! instance list changes every step re-uploads the whole buffer every step, which
//! is the cost the scatter path exists to avoid.
//!
//! So the rubble does not follow the tumbling chunk. Each fragment is placed
//! around its chunk's **rest centre** — `placement · center_of_mass` — which
//! `FractureState` freezes the instant the first chunk detaches and never moves
//! again. The batch's content is therefore a pure function of *which* chunks are
//! live, i.e. of the actor's fracture **generation**, and it re-uploads exactly
//! once per break. That is the same invariant [`crate::passes::fracture`]'s
//! geometry cache gets from an explicit stamp, obtained here structurally instead:
//! there is no stamp to get wrong because the bytes genuinely do not change.
//!
//! Physically it is also the honest picture. The fragments are what was left *at
//! the break* — the mortar dust and the spall around the hole — not a second
//! swarm of projectiles the solver never agreed to.
//!
//! ## Determinism
//!
//! Every fragment's offset, scale and orientation is a pure function of
//! `(entity, chunk index, detach order, fragment index)` through the SplitMix64
//! finalizer below. **No wall clock and no frame index**, which is the same rule
//! `inf_pcg::hash` states for scattering: a replay, a second host and a second
//! machine must lay the same rubble.
//!
//! # 2. The per-tier budget
//!
//! [`debris_budget_for`] is the roadmap's "per-tier debris budgets", and it lives
//! **here** rather than in `inf-physics` for the reason P22.3's ledger gives: a
//! fixed step must never be a function of the graphics settings, so physics owns
//! the budget as *data* (`inf_physics::d3::DebrisBudget`) and never learns the
//! word [`RenderTier`]. This function is the mapping, expressed in plain numbers
//! so that neither crate has to name the other; the **host** — which already knows
//! both — reads its tier and calls `set_debris_budget`.
//!
//! It only ever clamps **down**, exactly like [`RenderTier::apply`]: the High row
//! *is* the physics default, so a capable machine runs the engine's own numbers
//! and a weak one keeps less rubble. See the type's docs for the consequence that
//! buys, and for who must not apply it.

use glam::{DVec3, Quat};

use crate::caps::RenderTier;
use crate::scene::{ScatterBatch, ScatterData, ScatterInstance};
use crate::{PrimMesh, ID_NONE};

// ─────────────────────────────────────────────────────────────────────────────
// The mixer
// ─────────────────────────────────────────────────────────────────────────────

/// The golden-ratio odd constant used to decorrelate successive mixed words.
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 finalizer: a full-avalanche 64→64 bit mix.
///
/// A re-statement of the *specification* `inf_pcg::hash`, `inf_terrain::noise`
/// and `inf_mesh::fracture` all carry — the same three constants and the same
/// three shifts — rather than a dependency, because `inf-render` may not depend
/// on any of them and a renderer that pulled in the PCG runtime to lay gravel
/// would be the wrong shape entirely. [`the_mixer_matches_the_published_vector`]
/// pins it against a known output so a typo in a constant cannot pass as
/// "different but still random".
#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A decorrelated draw from a counter tuple — the `inf_pcg::Hash64` shape,
/// flattened because this module only ever folds four words.
#[inline]
fn draw(entity: u64, chunk: u32, order: u64, fragment: u32, salt: u64) -> u64 {
    let mut h = mix64(entity ^ GOLDEN);
    h = mix64(h ^ (chunk as u64).wrapping_mul(GOLDEN));
    h = mix64(h ^ order.wrapping_mul(GOLDEN));
    h = mix64(h ^ (fragment as u64).wrapping_mul(GOLDEN));
    mix64(h ^ salt.wrapping_mul(GOLDEN))
}

/// A uniform `f64` in `[0, 1)` from a draw — the top 53 bits, so the result is
/// exactly representable and independent of the low bits' quality.
#[inline]
fn unit(v: u64) -> f64 {
    (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// The sites
// ─────────────────────────────────────────────────────────────────────────────

/// Fragments shed per live chunk.
///
/// **Six.** The number is bounded from both ends by things that are already
/// fixed:
///
/// * From above by the budget it multiplies. A level at the default cap of 256
///   live chunks carries 1 536 fragments — two orders below the 100 k the P18.5
///   scatter path was built and measured for, and, because they arrive as one
///   batch per broken actor, a handful of draws rather than 1 536.
/// * From below by what it has to read as. One or two fragments per chunk reads
///   as litter rather than as rubble; the eye wants a small *pile*, and six
///   around a chunk's rest centre is the smallest count that does.
///
/// It is a **visual constant**, not a ratchet: raising it costs instances
/// linearly and changes no simulation anywhere. It is deliberately *not* per
/// tier — the tier already clamps this content twice over, through the debris
/// budget that decides how many chunks are live at all and through
/// `ScatterSettings`' own per-tier bands and impostor switch, and a third knob
/// would make the two hosts' projections differ by machine for no gain.
pub const DEBRIS_RUBBLE_PER_CHUNK: u32 = 6;

/// The smallest fragment, as a fraction of its parent chunk's characteristic
/// radius.
pub const DEBRIS_MIN_SCALE: f32 = 0.10;
/// The largest fragment, as a fraction of its parent chunk's characteristic
/// radius. Under a quarter of the chunk, so "sub-chunk debris" is true by
/// construction and a fragment can never be mistaken for a piece the solver owns.
pub const DEBRIS_MAX_SCALE: f32 = 0.22;

/// One broken chunk's contribution to the rubble batch.
///
/// `center` is the chunk's **rest** centre and not its live pose — see the module
/// docs; that is what makes the batch's content key stable between breaks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebrisSite {
    /// The owning actor's render id — folded into the seed so two walls that
    /// shatter identically still lay different rubble.
    pub entity: u64,
    /// The chunk's index in its `.inf_fracture`.
    pub chunk: u32,
    /// The chunk's detach order. Part of the seed so a chunk that came off, was
    /// reclaimed and (in some future) came off again is not laid twice the same.
    pub order: u64,
    /// The chunk's rest-pose world centre.
    pub center: DVec3,
    /// The chunk's characteristic radius, metres — the radius of the sphere with
    /// the chunk's volume. Fragments land inside it, so rubble never escapes the
    /// space the chunk itself occupied.
    pub radius_m: f64,
}

/// Lay `per_chunk` fragments around each site, in site order then fragment order.
///
/// A **pure function** of the sites — no clock, no camera, no floating origin —
/// so two hosts, two runs and two machines produce the same `Vec`, byte for byte.
///
/// Each fragment is placed inside its site's sphere by the **cube-root radius
/// rule** (`r = R · u^(1/3)`), which is what makes the placement uniform by
/// *volume* rather than piling everything against the surface; the direction
/// comes from the standard `z ∈ [-1, 1]` + azimuth construction. Azimuth uses
/// `f64` `sin`/`cos`, which is legal here and would not be in committed content:
/// the P14 LAW is about bytes an author *ships*, and nothing in this function is
/// ever serialized — it is re-derived identically on every host from the same
/// integers.
pub fn debris_instances(
    sites: &[DebrisSite],
    per_chunk: u32,
    color: [f32; 4],
) -> Vec<ScatterInstance> {
    let mut out = Vec::with_capacity(sites.len() * per_chunk as usize);
    for site in sites {
        if !(site.radius_m.is_finite() && site.radius_m > 0.0) || !site.center.is_finite() {
            continue;
        }
        for f in 0..per_chunk {
            let seed = |salt: u64| unit(draw(site.entity, site.chunk, site.order, f, salt));
            // Direction: uniform on the sphere.
            let z = 2.0 * seed(1) - 1.0;
            let phi = std::f64::consts::TAU * seed(2);
            let rxy = (1.0 - z * z).max(0.0).sqrt();
            let dir = DVec3::new(rxy * phi.cos(), z, rxy * phi.sin());
            // Radius: uniform by volume inside the chunk's own sphere.
            let r = site.radius_m * seed(3).cbrt();
            let position = site.center + dir * r;
            let t = seed(4) as f32;
            let scale = site.radius_m as f32
                * (DEBRIS_MIN_SCALE + t * (DEBRIS_MAX_SCALE - DEBRIS_MIN_SCALE));
            // Orientation: a seeded unit quaternion, so no two fragments read as
            // copies of one another.
            let (a, b, c) = (seed(5), seed(6), seed(7));
            let (s1, s2) = ((1.0 - a).max(0.0).sqrt(), a.sqrt());
            let (t1, t2) = (std::f64::consts::TAU * b, std::f64::consts::TAU * c);
            let rotation = Quat::from_xyzw(
                (s1 * t1.sin()) as f32,
                (s1 * t1.cos()) as f32,
                (s2 * t2.sin()) as f32,
                (s2 * t2.cos()) as f32,
            )
            .normalize();
            out.push(ScatterInstance {
                position,
                rotation,
                scale,
                // Tint is the caller's: rubble off a painted wall is that wall's
                // colour, which is what stops it reading as a second material.
                // It rides on the *instance* because `ScatterBatch` has no tint —
                // the batch carries only the PBR constants — and being inside the
                // packed bytes it is part of the content key, which is correct: a
                // repainted wall is different rubble.
                color,
            });
        }
    }
    out
}

/// Build one broken actor's rubble batch, or `None` when it sheds nothing.
///
/// The anchor is the batch's own first site rather than the actor's transform:
/// the transform is frozen at the break anyway, and anchoring on content keeps
/// the packed offsets small (P18.5's precision note) without a second input.
///
/// The pick id is [`ID_NONE`] on purpose, and this is the one field a reader
/// might expect to differ between hosts. It does not: a fragment is not an
/// entity, so there is nothing for a click to select — the same decision
/// `passes::fracture`'s `pack` records for the chunks themselves, and it makes
/// the two hosts' projections identical rather than merely equivalent.
pub fn debris_batch(
    sites: &[DebrisSite],
    per_chunk: u32,
    color: [f32; 4],
    roughness: f32,
) -> Option<ScatterBatch> {
    if sites.is_empty() || per_chunk == 0 {
        return None;
    }
    let instances = debris_instances(sites, per_chunk, color);
    if instances.is_empty() {
        return None;
    }
    let anchor = sites[0].center;
    Some(ScatterBatch {
        // A cube, tumbled: broken masonry has flat faces and sharp arrises, and
        // a sphere at this size reads as gravel on a driveway.
        data: std::sync::Arc::new(ScatterData::build(PrimMesh::Cube, anchor, instances)),
        anchor,
        metallic: 0.0,
        roughness,
        emissive: [0.0; 3],
        id: ID_NONE,
        // Unlimited: the renderer's own tier bands already decide how far rubble
        // draws, and content that asked for less would be asking on behalf of a
        // machine it cannot see.
        draw_distance: 0.0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// The per-tier budget
// ─────────────────────────────────────────────────────────────────────────────

/// A debris budget as **plain numbers** — the shape that lets this mapping exist
/// without `inf-render` naming `inf-physics` or `inf-physics` naming
/// [`RenderTier`].
///
/// The host converts it into an `inf_physics::d3::DebrisBudget`, which is a
/// two-field struct with these exact meanings, and calls `set_debris_budget`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebrisBudgetSpec {
    /// Level-wide cap on simultaneously-live detached chunks.
    pub max_live: u32,
    /// Seconds a detached chunk survives before it may be reclaimed.
    pub lifetime_s: f64,
}

/// The **High** row — pinned equal to `inf_physics::d3::{DEFAULT_DEBRIS_MAX_LIVE,
/// DEFAULT_DEBRIS_LIFETIME_S}` by `inf-player`'s
/// `the_high_tier_budget_is_the_physics_default` (the crate that can see both, on
/// the `destructible_defaults_match_the_mesh_crate` precedent).
///
/// Equality is not decoration. It is what makes a capable machine run the
/// engine's own numbers, so every headless gate, every replay and every
/// `--pie`-vs-shipping comparison in this repository is comparing the *unclamped*
/// behaviour — the clamp is a thing weak machines opt into, never a thing the
/// reference path silently picked up.
pub const DEBRIS_BUDGET_HIGH: DebrisBudgetSpec = DebrisBudgetSpec {
    max_live: 256,
    lifetime_s: 20.0,
};

/// Map a GPU capability tier onto a debris budget.
///
/// # The rows, argued
///
/// * **High** — the physics default, unchanged (see [`DEBRIS_BUDGET_HIGH`]).
/// * **Medium** — half the chunks, and they clear in twelve seconds rather than
///   twenty. Medium is the tier with no meshlet path and no GPU scatter cull
///   (`RenderTier::apply` turns `scatter.gpu` off), so every fragment this module
///   lays is CPU-culled on that machine; halving the *source* of them is worth
///   more there than any per-instance tuning.
/// * **Low** — 48 chunks for six seconds. Low already drops bloom, SSAO, TAA,
///   shadows, GI and scatter impostors; a machine that cannot afford a shadow map
///   cannot afford two hundred loose convex hulls and their contact pairs either.
///   48 is still enough for one building's collapse to read as a collapse, which
///   is the floor worth defending: the alternative — *no* debris on Low — turns a
///   scripted set piece into a wall that vanishes.
///
/// No number here is a ratchet and none of them is asserted against. The only
/// thing a gate asserts is the ordering (`Low ≤ Medium ≤ High`) and that High is
/// the physics default; the values themselves are a product policy, and a game
/// that wants different ones calls `set_debris_budget` with its own.
///
/// # It only ever clamps DOWN — and what that costs
///
/// [`RenderTier::apply`]'s contract, applied to a budget: High is the identity,
/// and nothing here can ask for *more* debris than the engine's default.
///
/// **The consequence, stated rather than buried.** The debris budget is read by
/// `step_fractures`, which is fixed-step simulation — so a host that sets it from
/// the adapter has made its simulation a function of the machine, and two players
/// on different tiers will not agree about where a building's rubble ended up.
/// That is a real cost and it is why:
///
/// * the **editor's Simulate does not apply this mapping**. Simulate is the
///   preview half of `PIE == shipping`, and a preview whose rubble count came
///   from the author's graphics card would fail the house gate on any other
///   machine. It runs the engine default, which is the High row.
/// * every headless path (`run_headless`, `scene_trace`, the `--pie` protocol
///   loop) likewise never builds a render host and therefore never sees a tier.
/// * a **replay or a networked session must pin the budget explicitly** rather
///   than inherit it. There is nothing in the engine that stops it being pinned —
///   `set_debris_budget` is the same seam either way — but nothing that does it
///   for you either, and P22.4's ledger says so.
///
/// The windowed player is the one caller, because it is the one place that has
/// both a tier and a session and no comparison to lose.
pub fn debris_budget_for(tier: RenderTier) -> DebrisBudgetSpec {
    match tier {
        RenderTier::High => DEBRIS_BUDGET_HIGH,
        RenderTier::Medium => DebrisBudgetSpec {
            max_live: 128,
            lifetime_s: 12.0,
        },
        RenderTier::Low => DebrisBudgetSpec {
            max_live: 48,
            lifetime_s: 6.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(chunk: u32, order: u64, y: f64) -> DebrisSite {
        DebrisSite {
            entity: 0x2204,
            chunk,
            order,
            center: DVec3::new(1.0, y, -2.0),
            radius_m: 0.5,
        }
    }

    /// The mixer really is SplitMix64's finalizer. A constant with a transposed
    /// digit still avalanches, so "the output looks random" proves nothing — this
    /// pins the published vector for `mix64(0)`… which is `0`, and for a
    /// non-degenerate input as well.
    #[test]
    fn the_mixer_matches_the_published_vector() {
        assert_eq!(mix64(0), 0);
        // SplitMix64's finalizer applied to the golden-ratio constant — i.e. the
        // reference generator's first output for seed 0.
        assert_eq!(mix64(GOLDEN), 0xe220_a839_7b1d_cdaf);
    }

    /// **Determinism.** The same sites lay the same rubble, twice.
    #[test]
    fn the_same_sites_lay_byte_identical_rubble() {
        let sites = [site(0, 1, 0.5), site(3, 2, 1.5)];
        let a = debris_instances(&sites, DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4]);
        let b = debris_instances(&sites, DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4]);
        assert_eq!(a.len(), 2 * DEBRIS_RUBBLE_PER_CHUNK as usize);
        assert_eq!(a, b);
    }

    /// …and two different chunks of the same actor lay *different* rubble, so
    /// "deterministic" is not "constant".
    #[test]
    fn two_chunks_do_not_lay_the_same_pile() {
        let a = debris_instances(&[site(0, 1, 0.5)], DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4]);
        let b = debris_instances(&[site(1, 2, 0.5)], DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4]);
        assert_ne!(a[0].rotation, b[0].rotation);
        let c = DVec3::new(1.0, 0.5, -2.0);
        assert_ne!(a[0].position - c, b[0].position - c);
    }

    /// Every fragment lands **inside** its chunk's own sphere and is genuinely
    /// sub-chunk in size — the two claims the name "sub-chunk debris" makes.
    #[test]
    fn rubble_stays_inside_the_chunk_it_came_from() {
        let s = site(7, 9, 4.0);
        for i in debris_instances(&[s], 64, [1.0; 4]) {
            assert!(
                (i.position - s.center).length() <= s.radius_m + 1e-9,
                "a fragment escaped its chunk's sphere"
            );
            assert!(i.scale > 0.0 && i.scale <= s.radius_m as f32 * DEBRIS_MAX_SCALE + 1e-6);
            assert!((i.rotation.length() - 1.0).abs() < 1e-5, "not a unit quat");
        }
    }

    /// **THE KEYING CLAIM.** A batch's content key is a function of the live chunk
    /// set alone, so a collapse re-uploads once per break rather than once per
    /// step. Sites are rest centres; nothing here changes as the chunks tumble.
    #[test]
    fn the_batch_key_is_stable_while_chunks_tumble() {
        let sites = [site(0, 1, 0.5), site(1, 2, 1.5)];
        let a = debris_batch(&sites, DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4], 0.8).unwrap();
        let b = debris_batch(&sites, DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4], 0.8).unwrap();
        assert_eq!(a.data.key(), b.data.key());
        // …and a break that takes another chunk off IS a different batch.
        let grown = [site(0, 1, 0.5), site(1, 2, 1.5), site(2, 3, 2.5)];
        let c = debris_batch(&grown, DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4], 0.8).unwrap();
        assert_ne!(a.data.key(), c.data.key());
    }

    /// Nothing broken ⇒ no batch, so an intact level's scatter list is unchanged
    /// and the pass sees exactly what it always did.
    #[test]
    fn an_unbroken_actor_sheds_no_batch() {
        assert!(debris_batch(&[], DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4], 0.8).is_none());
        assert!(debris_batch(&[site(0, 1, 0.5)], 0, [1.0; 4], 0.8).is_none());
    }

    /// A degenerate site is skipped rather than laid at NaN — the refusal-as-a-
    /// value rule, on the render side.
    #[test]
    fn a_degenerate_site_lays_nothing() {
        let bad = DebrisSite {
            radius_m: f64::NAN,
            ..site(0, 1, 0.0)
        };
        assert!(debris_instances(&[bad], DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4]).is_empty());
        let zero = DebrisSite {
            radius_m: 0.0,
            ..site(0, 1, 0.0)
        };
        assert!(debris_instances(&[zero], DEBRIS_RUBBLE_PER_CHUNK, [1.0; 4]).is_empty());
    }

    /// **The tier clamps DOWN and never up**, and High is the identity.
    #[test]
    fn the_tier_budget_only_ever_clamps_down() {
        let high = debris_budget_for(RenderTier::High);
        let medium = debris_budget_for(RenderTier::Medium);
        let low = debris_budget_for(RenderTier::Low);
        assert_eq!(high, DEBRIS_BUDGET_HIGH);
        assert!(medium.max_live <= high.max_live);
        assert!(low.max_live <= medium.max_live);
        assert!(medium.lifetime_s <= high.lifetime_s);
        assert!(low.lifetime_s <= medium.lifetime_s);
        // …and a weak machine still gets enough for a collapse to read as one.
        assert!(low.max_live >= 32, "Low drops debris to a vanishing wall");
        assert!(low.lifetime_s > 0.0);
    }
}
