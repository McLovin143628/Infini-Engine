//! **A round in flight** (wave WPN2a): the far half of the hybrid every AAA
//! shooter's gunplay is built on, and the damage curve both halves share.
//!
//! # The hybrid, and why it is a hybrid
//!
//! The user's research doc §1 states the rule this module implements: *"AAA
//! games use a hybrid distance-switch threshold. Raycasts (hitscan) handle
//! instant hit detection at close range to keep networking responsive, while
//! simulated projectile physics take over at longer distances."* Both halves are
//! needed and neither is enough:
//!
//! * an instant ray at 400 m is a rifle with no bullet drop, no travel time and
//!   no lead — which is what this engine shipped from island wave I6 to this
//!   wave, stated as an honest v1 in `resolve_shot`'s own doc;
//! * a body in flight at 3 m is a shot that arrives two frames after the trigger
//!   for no perceptible gain and four ray casts instead of one.
//!
//! So a shot casts ONE ray to [`crate::weapon::WeaponDef::hitscan_threshold_m`]
//! and, if that ray found nothing, a **round** is minted here at the threshold
//! point carrying the muzzle speed along the same spread direction. Everything
//! below that threshold is byte-for-byte the behaviour that shipped at I6.
//!
//! # It is SIM STATE, and that decides everything about its shape
//!
//! A round is not a particle and not a decoration: where it is decides whether
//! somebody is hit, so two hosts that disagree about it have diverged. That puts
//! this module under every determinism door in the repository at once:
//!
//! * the pool is a **bevy resource** on [`crate::deform::DeformFieldRes`]'s own
//!   doctrine — never serialized, so **no schema moves** (scene v27 and
//!   `ScenePayload` 13 are untouched by this wave);
//! * [`round_state_bytes`] appends at the **TAIL** of `RuntimeSim::state_bytes`
//!   and is **EMPTY when no round is in flight**, so every trace committed
//!   before this wave is byte-identical — the rule `runtime_sim.rs` states
//!   about the twelve sections in front of it;
//! * the integrator uses `+ - * /` and `sqrt` only. No `sin`, no `cos`, no
//!   `powf`, no `cbrt`: IEEE-754 specifies the four operators and the square
//!   root exactly, and the P14 law bans exactly the transcendentals it does not
//!   (`inf_math::libm_ban`). `inf-physics/tests/portable_character.rs` scans
//!   this file for them;
//! * a round carries its shooter's guid and no guid of its own, so a fixed step
//!   mints nothing random.
//!
//! # The sub-stepped segment cast, and the reason it is not a point move
//!
//! The doc again, and it is the one thing about a projectile that is not
//! negotiable: *"Never use simple point movement (`position += velocity * dt`),
//! as high-speed bullets will tunnel through thin geometry. Instead, raycast
//! between `position_previous` and `position_current` on every engine step."* A
//! 900 m/s round moves **15 metres** in one 60 Hz step; a wall is 20 cm thick.
//! So each fixed step is split into [`PROJECTILE_SUB_STEPS`] and each sub-step
//! is a **segment cast** from the previous position to the next one — never a
//! test of the endpoint. `inf_physics::d3::gameplay::step_rounds` is the caller
//! and `wpn2a_gate`'s wall arm is what keeps it honest: mutating the segment
//! into a point move puts the round through a wall at 100 m.
//!
//! # What is NOT here
//!
//! Pellets, blasts, lock-on, fuses and bounce are WPN2c/WPN2d's; this pool is
//! deliberately the substrate they will extend rather than five kinds today.
//! Recoil springs, sway and spread state are WPN2b's.
//!
//! A round that hits a **dynamic** body — a car, a crate, a fractured chunk, a
//! ragdoll — **stops there** since this wave's audit: both halves of the hybrid
//! cast `CastTargets::AllSolid` minus everything the shooter is
//! (`d3::gameplay::shot_exclusions`), and a ragdoll's limb names the character
//! it belongs to (`PhysicsBridge3D::guid_of_ragdoll_collider`), so a body on the
//! floor takes the joules. What a **car** does about being shot is still
//! VEH3c's: a chassis has no `Health` and no `Destructible`, so the round ends
//! there and spends nothing — measured in
//! `wpn2a_gate::a_round_stops_in_a_parked_car_and_the_car_spends_nothing`.

use bevy_ecs::prelude::Resource;
use glam::DVec3;
use uuid::Uuid;

use crate::components::{AudioSource, DistanceModel};
use crate::weapon::WeaponDef;
use crate::world::EcsWorld;

// ── the numbers ─────────────────────────────────────────────────────────────

/// **Gravity on a round**, m/s², before
/// [`WeaponDef::gravity_scale`](crate::weapon::WeaponDef::gravity_scale)
/// multiplies it.
///
/// 9.81, which is the research doc's own `gravity_scale = 9.81`. It is a
/// constant here rather than a read of the physics world's gravity vector for
/// one reason: this integrator is a **Ring-0 pure function** with unit tests
/// that compare it against the closed-form parabola, and a world argument would
/// make the drop table a fact about a fixture instead of about the model. A
/// level that wants moon gravity for bullets sets `gravity_scale` per weapon.
pub const PROJECTILE_GRAVITY_MPS2: f64 = 9.81;

/// **How many sub-steps one fixed step of flight is split into.**
///
/// Four — the doc's own `let sub_steps = 4;`. It is what turns a 15 m jump at
/// 900 m/s into four 3.75 m segments, and the segment is what a wall stops. It
/// is also the multiplier on the ray cost: this is the number
/// [`MAX_ROUNDS_IN_FLIGHT`] is derived from.
pub const PROJECTILE_SUB_STEPS: u32 = 4;

/// **How many rays the flight of every round in the world may cost one fixed
/// step.**
///
/// Two hundred and fifty-six. It is minted beside
/// `inf_physics::d3::gameplay::MAX_ACTS_PER_STEP` and for the same kind of
/// reason: a cost bound, so a firefight's cast count is a constant rather than a
/// function of how many people are shooting.
///
/// The arithmetic that sizes it, with its population named: **eight** shooters
/// (the figure `island-progress.md`'s audio eviction table already prices a
/// firefight at) with an assault rifle at **900 rpm** is 8 × 15 = 120 rounds a
/// second, or two a step; a 930 m/s round crossing a 500 m `range_m` against
/// this module's drag lives about **0.7 s**, so the steady-state population is
/// ≈ 84 rounds. At four sub-steps that is 336 rays — **over this ceiling**, and
/// deliberately: the ceiling is what the refusal is measured at, and a bound
/// nothing ever reaches is a bound nobody has tested. Sixty-four rounds is a
/// firefight; the eighty-fifth is refused with a value.
pub const MAX_SHOT_RAYS_PER_STEP: usize = 256;

/// **How many rounds may be in flight at once**, derived from the ray ceiling.
///
/// `MAX_SHOT_RAYS_PER_STEP / PROJECTILE_SUB_STEPS` = **64**, and it is derived
/// rather than chosen so the two can never disagree. Bounding the POOL rather
/// than throttling the flight is the deliberate half: a ray budget spent
/// mid-flight would have to advance some rounds and not others, which is time
/// dilation — the round that waited arrives late, at the wrong place, and its
/// drop is wrong for the rest of its life. Refusing the **spawn** costs one
/// round that was never fired and leaves every round that exists exact.
pub const MAX_ROUNDS_IN_FLIGHT: usize = MAX_SHOT_RAYS_PER_STEP / PROJECTILE_SUB_STEPS as usize;

/// **The longest a round may live**, seconds.
///
/// Eight. A round dies at its weapon's `range_m`, on a hit, or on leaving the
/// active partition — this is the fourth answer, for the round none of the three
/// reach: a shot fired straight up on an unbounded band at a weapon whose range
/// it never travels. At the slowest muzzle speed in the registry (an RPG at
/// 115 m/s) eight seconds is 920 m, past every `range_m` in it.
pub const MAX_ROUND_LIFETIME_S: f64 = 8.0;

/// **How tall the head's own band is**, metres — the half-height of the zone a
/// round has to arrive in to be a headshot.
///
/// Twelve centimetres either side of the head point, which is a 24 cm band: an
/// adult head is about that.
///
/// # It is a BAND, not a sphere, and the collider is why
///
/// A character in this engine is a **capsule** of radius ~0.30 m. A ray fired at
/// a standing body stops on that capsule's SURFACE, which is a capsule radius
/// away from the axis the head socket sits on — so a 12 cm sphere about the head
/// point is a target no shot can ever reach, and a headshot would be
/// unreachable on every character in the game. Measured, on this wave's own
/// first draft: a shot aimed exactly at the head socket of a target 12 m away
/// arrives 0.300 m from it and the sphere test answered `false`.
///
/// What a capsule CAN carry is the height a round arrived at, so that is what
/// [`is_headshot`] measures — the vertical distance to the head point, plus a
/// horizontal bound that is the body's OWN radius (the surface a ray can reach)
/// plus this band. Nothing is fudged and no constant is chosen twice: the day a
/// character grows a real head collider, the test reads the same numbers off a
/// smaller radius.
///
/// It is deliberately **not a new collider**: adding a body part per character
/// would put a shape in the physics world for every crowd agent, which is the
/// cost `apply_hit`'s lazy-health doc already refuses one field along.
pub const HEAD_RADIUS_M: f64 = 0.12;

// ── the round ───────────────────────────────────────────────────────────────

/// **One round in flight.**
///
/// It carries its weapon's whole definition rather than a reference to a
/// catalogue, for `WeaponHit::loud`'s reason verbatim: by the time this round
/// arrives the shooter may have scrolled, and what a bullet does is a property
/// of the shot and not of whatever is in the hand afterwards. [`WeaponDef`] is
/// `Copy` and carries no `Serialize`, so this costs a memcpy and no schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Round {
    /// Who fired it. Excluded from the FIRST segment's cast and from nothing
    /// afterwards — see [`Round::first_segment`].
    pub shooter: Uuid,
    /// Where it is now, world metres.
    pub at: DVec3,
    /// How fast it is going, world metres per second.
    pub velocity: DVec3,
    /// **How far it has flown from the MUZZLE**, metres — not from the spawn
    /// point. It starts at the weapon's `hitscan_threshold_m`, because that is
    /// the distance the instant ray already covered, and it is what the damage
    /// curve is evaluated at: a round that killed at 250 m must spend the joules
    /// the curve owes at 250 m and not at 225.
    pub travelled_m: f64,
    /// How long it has been alive, seconds.
    pub age_s: f64,
    /// **Whether the next segment is its first.** The shooter's own collider is
    /// excluded on segment 0 only, which is where the round is a hand's breadth
    /// from the body that fired it; a round that comes back at its shooter after
    /// a ricochet (which nothing in this engine produces yet) should hit them.
    pub first_segment: bool,
    /// **Whether this round has already cracked past the listener** (wave
    /// WPN2c) — a latch, so a round that spends four sub-steps inside
    /// [`CRACK_RADIUS_M`] makes ONE noise rather than four.
    ///
    /// It is FOLDED into [`round_state_bytes`], which is what makes the trace
    /// grow from 81 bytes a round to 82 and is stated as the cause when a
    /// committed trace moves. Folding it is not optional: it is a latch over
    /// history rather than a function of the positions beside it, so two hosts
    /// that disagreed about it would agree about every number the trace carried
    /// and one of them would crack twice.
    pub cracked: bool,
    /// The weapon that fired it.
    pub def: WeaponDef,
}

/// **Every round in flight** — the pool, a bevy resource.
///
/// [`crate::deform::DeformFieldRes`]'s doctrine: never serialized, absent on a
/// level that has never fired one, wiped by `clear_rounds` on the way into and
/// out of a Simulate session.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct RoundPool {
    /// The live rounds, in spawn order. **This is the only field
    /// [`round_state_bytes`] folds.**
    pub rounds: Vec<Round>,
    /// How many rounds have ever been minted in this session.
    pub spawned: u64,
    /// **How many spawns were REFUSED** because the pool was full or the ray
    /// ceiling was reached — the value the law asks for. A round that is
    /// silently dropped is a shot the player fired and nobody can account for.
    pub refused: u64,
    /// How many rounds ended on a hit.
    pub impacts: u64,
    /// How many died at their weapon's `range_m` or of old age.
    pub expired: u64,
    /// How many died leaving the active partition.
    pub left_band: u64,
    /// **How far the last round that hit something had flown**, metres — the
    /// number the demo loop's `hero.csv` carries so a frame of an impact has a
    /// distance beside it.
    ///
    /// It is **not folded** into [`round_state_bytes`], and that is deliberate
    /// rather than an omission: it is a pure function of the rounds that ARE
    /// folded (a round's own `travelled_m` at the step it hit), so two hosts
    /// that disagreed about it must first have disagreed about a round's
    /// position, which the trace sees first and sees earlier. That is
    /// `GameplayReport::muzzles_without_a_socket`'s own argument for a
    /// diagnostic beside the state rather than inside it.
    pub last_flight_m: f64,
}

/// The pool, if this world has ever fired a round.
pub fn round_pool(world: &EcsWorld) -> Option<&RoundPool> {
    world.world().get_resource::<RoundPool>()
}

/// **How many rounds are in flight right now.** `0` on a world with no pool.
pub fn rounds_in_flight(world: &EcsWorld) -> usize {
    round_pool(world).map(|p| p.rounds.len()).unwrap_or(0)
}

/// **Mint a round, or refuse with a value.**
///
/// `rays_already` is how many casts this fixed step has already spent — the
/// hitscan half's one-per-shot — so the ceiling covers the whole step's ray bill
/// and not only the flight's. Answers `false` when the pool is full or the
/// ceiling would be crossed, having counted the refusal; the caller's shot still
/// happened, still made a noise and still panicked the street, which is the
/// honest outcome for a round that could not be afforded.
pub fn spawn_round(world: &mut EcsWorld, round: Round, rays_already: usize) -> bool {
    if !round.at.is_finite() || !round.velocity.is_finite() {
        return false;
    }
    let w = world.world_mut();
    if w.get_resource::<RoundPool>().is_none() {
        w.insert_resource(RoundPool::default());
    }
    let mut pool = w.resource_mut::<RoundPool>();
    let after = rays_already + (pool.rounds.len() + 1) * PROJECTILE_SUB_STEPS as usize;
    if pool.rounds.len() >= MAX_ROUNDS_IN_FLIGHT || after > MAX_SHOT_RAYS_PER_STEP {
        pool.refused += 1;
        return false;
    }
    pool.rounds.push(round);
    pool.spawned += 1;
    true
}

/// **Forget every round** — the Simulate session's door, on
/// `crate::crowd::clear_crowd`'s terms: run 2 of a session must begin where run
/// 1 did, and a round still in the air when the author pressed stop is run 1's.
pub fn clear_rounds(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<RoundPool>();
}

/// **The rounds' trace bytes** — 82 a round (16 guid + six f64 + two f64 + two
/// flags), in flight order, and **empty when nothing is flying**.
///
/// It was 81 until wave WPN2c added [`Round::cracked`]; see that field for why
/// a latch has to be folded.
///
/// Empty is the load-bearing half: it is what keeps every trace committed before
/// this wave byte-identical, and it is why the counters above are not in here.
/// A level that has fired and landed everything folds nothing, exactly as a
/// level that has never fired does.
pub fn round_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(pool) = round_pool(world) else {
        return Vec::new();
    };
    if pool.rounds.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pool.rounds.len() * ROUND_TRACE_BYTES);
    for r in &pool.rounds {
        out.extend_from_slice(r.shooter.as_bytes());
        for v in [
            r.at.x,
            r.at.y,
            r.at.z,
            r.velocity.x,
            r.velocity.y,
            r.velocity.z,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&r.travelled_m.to_bits().to_le_bytes());
        out.extend_from_slice(&r.age_s.to_bits().to_le_bytes());
        out.push(u8::from(r.first_segment));
        out.push(u8::from(r.cracked));
    }
    out
}

/// **How many bytes one round folds into the trace** — 16 for the shooter's
/// guid, eight f64 for the two vectors and the two scalars, and two flags.
///
/// Named because the fold and the arm that measures it must not be able to
/// disagree about the arithmetic, which is a mistake this wave made once
/// already one module over.
pub const ROUND_TRACE_BYTES: usize = 16 + 8 * 8 + 2;

// ── the flight ──────────────────────────────────────────────────────────────

/// **One sub-step of flight**, as a pure function: `(next position, next
/// velocity)` from a current one.
///
/// The doc's own integration, semi-implicit Euler with the velocity updated
/// first so the position uses the velocity it will leave with:
///
/// ```text
/// a      = (0, -g·gravity_scale, 0)  −  v̂ · drag_k · |v|²
/// v(t+h) = v(t) + a·h
/// p(t+h) = p(t) + v(t+h)·h
/// ```
///
/// # `drag_k`, and its units, stated
///
/// The doc writes the drag as `−½ρ v² C_d A · v̂` and then collapses it in its
/// own Rust to one `drag_coefficient` multiplying `speed_sq`. So does this: the
/// whole of `½ρC_dA/m` is **one** [`WeaponDef::drag_k`] whose unit is **1/m** —
/// `drag_k · v²` has to come out in m/s², and `(1/m)·(m/s)² = m/s²`. The doc's
/// rifle figure is 0.0003–0.0004 /m, which at 900 m/s is 243–324 m/s² of
/// deceleration, or about 6 % of the muzzle speed lost over 200 m. Splitting it
/// back into four fields would be four numbers an author cannot check against
/// anything and one they can.
///
/// # No transcendentals
///
/// `|v|` is a `sqrt`, which IEEE-754 specifies exactly and
/// `inf_math::libm_ban`'s own header names as deliberately not banned. There is
/// no `powf` (the square is a multiply) and no trigonometry at all: the
/// direction comes in as a vector from `weapon::shot_direction`, which already
/// did its `psin64`/`pcos64` on the aim.
pub fn advance_round(at: DVec3, velocity: DVec3, def: &WeaponDef, sub_dt: f64) -> (DVec3, DVec3) {
    let speed = velocity.length();
    // A round at rest has no drag direction; `v/|v|` would be a NaN, and a NaN
    // position is a round that hits nothing for ever.
    let drag = if speed > 1e-9 {
        velocity * (-def.drag_k.max(0.0) * speed)
    } else {
        DVec3::ZERO
    };
    let gravity = DVec3::new(0.0, -PROJECTILE_GRAVITY_MPS2 * def.gravity_scale, 0.0);
    let v = velocity + (gravity + drag) * sub_dt;
    (at + v * sub_dt, v)
}

// ── the damage curve ────────────────────────────────────────────────────────

/// **The Hermite smoothstep**, `t²(3 − 2t)` — the doc §5's own
/// `t * t * (3.0 - 2.0 * t)`.
///
/// Named rather than inlined so a gate can compare the curve against the closed
/// form at five distances and mutate one place to red it.
pub fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// **What a shot is worth at a distance**, joules — the doc §5's
/// `DamageCurve::calculate_damage`, in this engine's unit.
///
/// * at or inside `effective_m`: the full `base_j`;
/// * at or beyond `max_m`: `base_j · min_frac`;
/// * between: the Hermite smoothstep between the two, so the fall-off has no
///   corner at either end — which is the whole reason the doc uses a smoothstep
///   rather than a lerp.
///
/// **A curve with no `max_m` is FLAT**, and that is what makes every level
/// committed before this wave byte-identical: [`WeaponDef::default`] leaves both
/// ranges at zero, `max_m > effective_m` is false, and the answer is `base_j`
/// at every distance — the I6 behaviour exactly.
pub fn damage_curve_j(base_j: f64, min_frac: f64, effective_m: f64, max_m: f64, at_m: f64) -> f64 {
    // The negation is on the BOOL and not on the comparison, which is what
    // keeps a NaN out of the curve: `max_m > effective_m` is false for a NaN
    // either way, and `is_finite` says so out loud rather than by accident.
    let bends =
        max_m.is_finite() && effective_m.is_finite() && at_m.is_finite() && max_m > effective_m;
    if !bends {
        return base_j;
    }
    if at_m <= effective_m {
        return base_j;
    }
    let min_j = base_j * min_frac.clamp(0.0, 1.0);
    if at_m >= max_m {
        return min_j;
    }
    let t = (at_m - effective_m) / (max_m - effective_m);
    base_j + smoothstep(t) * (min_j - base_j)
}

/// **Did this round arrive at the head?**
///
/// `head` is the world position of the rig's own `head` socket (or the capsule
/// rule's stand-in for it; see `inf_physics::d3::gameplay::head_point`, which is
/// the one door both answers come out of), and `body_radius_m` is the target's
/// own collider radius.
///
/// The test is a **band**, and [`HEAD_RADIUS_M`] says why: within that distance
/// of the head point vertically, and within the body's own radius plus the same
/// band horizontally. A sphere would be a target the capsule world can never
/// present.
pub fn is_headshot(point: DVec3, head: DVec3, body_radius_m: f64) -> bool {
    let d = point - head;
    if !d.is_finite() {
        return false;
    }
    let across = (d.x * d.x + d.z * d.z).sqrt();
    d.y.abs() <= HEAD_RADIUS_M && across <= body_radius_m.max(0.0) + HEAD_RADIUS_M
}

// ── the supersonic crack (wave WPN2c) ───────────────────────────────────────

/// **The speed of sound**, m/s — the research doc section 4's own 343.
///
/// A round slower than this makes no crack, and that is not a tuning knob: a
/// sonic boom is what a body faster than its own pressure wave leaves behind,
/// and a subsonic round leaves nothing. The registry has weapons on both sides
/// of it on purpose — the AS VAL is 295 m/s and is silent by physics rather
/// than by a flag.
pub const SPEED_OF_SOUND_MPS: f64 = 343.0;

/// **How close a supersonic round must pass to be heard cracking**, metres —
/// the doc's own four.
///
/// It is a distance from the LISTENER to the round's path, not to the round: a
/// crack is heard where the shock cone crosses an ear, and at 900 m/s a round
/// covers fifteen metres in a fixed step, so a point test would miss almost
/// every one of them. See [`point_to_segment_m`].
pub const CRACK_RADIUS_M: f64 = 4.0;

/// **The volume a crack is played at.** Full: it is the loudest thing a person
/// who is being shot at hears, and it is the whole point of the effect.
pub const CRACK_VOLUME: f64 = 1.0;

/// Metres inside which a crack is at full volume — one, because within four of
/// the path it is essentially at the ear.
pub const CRACK_MIN_M: f64 = 1.0;

/// **Metres past which a crack is silent.** Twenty: it is a local event by
/// construction ([`CRACK_RADIUS_M`] is four), and this exists so the spatial
/// model has a curve rather than a cliff.
pub const CRACK_MAX_M: f64 = 20.0;

/// **The salt a crack's audio source key carries**, so a round cracking past
/// somebody does not take its shooter's report layers' voices.
pub const CRACK_SALT: u64 = 0x5750_4e32_0000_0005;

/// **One round going past an ear** — what a host turns into a `Play`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crack {
    /// **What kind of gun fired it** — which class's N-wave plays.
    ///
    /// On the crack rather than looked up from the shooter, for
    /// `WeaponHit::class`'s reason exactly: the round has been in the air for up
    /// to eight seconds and the shooter may have scrolled twice.
    pub class: crate::weapon::WeaponClass,
    /// Who fired the round. The source key is derived from it and
    /// [`CRACK_SALT`], so one shooter's rounds are one crack voice.
    pub shooter: Uuid,
    /// **The closest point on the round's own segment to the listener** — where
    /// the crack is heard from, which is beside the ear rather than at the
    /// muzzle or at whatever the round eventually hits.
    pub at: DVec3,
    /// How fast the round was going, m/s. Above [`SPEED_OF_SOUND_MPS`] by
    /// construction; carried so a log can show the margin.
    pub speed_mps: f64,
    /// How far the path passed from the listener, metres. At most
    /// [`CRACK_RADIUS_M`].
    pub miss_m: f64,
}

/// **How close a segment passes to a point**, metres — the research doc's own
/// `check_supersonic_crack`, as a pure function.
///
/// Answers `(distance, closest point on the segment)`. A degenerate segment
/// answers the distance to its start, which is the right answer for a round
/// that did not move.
///
/// Portable: one dot product, one clamp and one length. No trigonometry, so it
/// is bit-identical on every target — which it has to be, because the crack it
/// decides is a command two hosts are compared on.
/// `a > b`, written once so the **NaN-rejecting** negation `!greater(a, b)`
/// reads as intent rather than as a negated comparison.
///
/// `inf_anim::greater`'s discipline exactly, and for its reason: clippy's
/// `neg_cmp_op_on_partial_ord` is right that `!(a > b)` reads badly and wrong
/// that `partial_cmp` is the fix, because `partial_cmp` answers `None` for a
/// NaN and both callers here want the NaN on the REFUSING side. A segment whose
/// length is not a number must not be divided by, and a round whose speed is
/// not a number must not crack.
#[inline]
fn greater(a: f64, b: f64) -> bool {
    a > b
}

pub fn point_to_segment_m(point: DVec3, a: DVec3, b: DVec3) -> (f64, DVec3) {
    let seg = b - a;
    let len2 = seg.length_squared();
    if !greater(len2, 0.0) || !point.is_finite() {
        return ((point - a).length(), a);
    }
    let t = ((point - a).dot(seg) / len2).clamp(0.0, 1.0);
    let closest = a + seg * t;
    ((point - closest).length(), closest)
}

/// **Does this segment of this round crack past this listener?**
///
/// The doc's rule exactly: faster than sound, and passing within
/// [`CRACK_RADIUS_M`] of the ear. `None` for everything else, including for a
/// level with no listener — a crack nobody is standing near is a sound nobody
/// makes.
pub fn crack_for(
    shooter: Uuid,
    class: crate::weapon::WeaponClass,
    prev: DVec3,
    next: DVec3,
    speed_mps: f64,
    listener: Option<DVec3>,
) -> Option<Crack> {
    if !greater(speed_mps, SPEED_OF_SOUND_MPS) {
        return None;
    }
    let ear = listener?;
    let (miss_m, at) = point_to_segment_m(ear, prev, next);
    if miss_m > CRACK_RADIUS_M {
        return None;
    }
    Some(Crack {
        class,
        shooter,
        at,
        speed_mps,
        miss_m,
    })
}

/// **The `AudioSource` a supersonic crack plays** — one Ring-0 description, on
/// `crate::weapon::report_source`'s own terms and for its reason.
///
/// The clip is the class's own N-wave ([`crate::weapon::ReportClip::Crack`]) —
/// the same two milliseconds the report's fourth layer plays, because they are
/// the same physical event heard from two places: the fourth layer is the shot
/// heard from far away, and this is the round heard from beside its path.
pub fn crack_source(class: crate::weapon::WeaponClass) -> AudioSource {
    AudioSource {
        clip: Some(crate::weapon::report_clip(
            class,
            crate::weapon::ReportClip::Crack,
        )),
        bus: crate::weapon::REPORT_BUS.to_string(),
        volume: CRACK_VOLUME,
        pitch: 1.0,
        looping: false,
        spatial: true,
        min_distance: CRACK_MIN_M,
        max_distance: CRACK_MAX_M,
        distance_model: DistanceModel::Inverse,
        rolloff: crate::weapon::REPORT_ROLLOFF,
        occlusion: false,
        autoplay: false,
    }
}

/// **The source key a shooter's cracks play on** — salted off the shooter, so a
/// burst going past somebody is one voice that restarts rather than forty.
pub fn crack_source_key(shooter_key: u64) -> u64 {
    shooter_key ^ CRACK_SALT
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The curve is flat by default**, which is what keeps every pre-WPN2a
    /// trace byte-identical.
    #[test]
    fn a_weapon_that_names_no_ranges_does_the_same_damage_everywhere() {
        let d = WeaponDef::default();
        for at in [0.0, 1.0, 50.0, 400.0, 20_000.0] {
            assert_eq!(
                d.damage_at(at, false),
                d.damage_j,
                "the default curve bent at {at} m"
            );
        }
    }

    /// **The smoothstep really is the doc's**, at five distances against the
    /// closed form written out by hand.
    #[test]
    fn the_curve_is_the_hermite_smoothstep_between_the_two_ranges() {
        let (base, min_frac, eff, max) = (600.0f64, 0.6f64, 35.0f64, 80.0f64);
        for at in [35.0, 46.25, 57.5, 68.75, 80.0] {
            let t = ((at - eff) / (max - eff)).clamp(0.0, 1.0);
            let want = base + (t * t * (3.0 - 2.0 * t)) * (base * min_frac - base);
            let got = damage_curve_j(base, min_frac, eff, max, at);
            assert!(
                (got - want).abs() < 1e-9,
                "at {at} m the curve gave {got} and the closed form {want}"
            );
        }
        // The two ends, exactly.
        assert_eq!(damage_curve_j(base, min_frac, eff, max, 10.0), base);
        assert_eq!(damage_curve_j(base, min_frac, eff, max, 500.0), base * 0.6);
    }

    /// **A headshot multiplies and a body shot does not** — the door, not the
    /// world (that arm is `wpn2a_gate`'s).
    #[test]
    fn the_head_multiplier_applies_only_to_a_head() {
        let mut d = WeaponDef {
            damage_j: 600.0,
            headshot_mult: 1.5,
            ..Default::default()
        };
        assert_eq!(d.damage_at(5.0, true), 900.0);
        assert_eq!(d.damage_at(5.0, false), 600.0);
        // …and it multiplies what the CURVE gave, not the base.
        d.effective_range_m = 10.0;
        d.max_range_m = 20.0;
        d.min_damage_frac = 0.5;
        assert_eq!(d.damage_at(20.0, true), 300.0 * 1.5);
    }

    /// **The head band has a height and a width, and neither is a guess.**
    #[test]
    fn the_head_band_is_twelve_centimetres_tall_and_a_body_wide() {
        let head = DVec3::new(1.0, 1.7, 2.0);
        let r = 0.30;
        assert!(is_headshot(head, head, r));
        // Vertically: the band, and one millimetre outside it.
        assert!(is_headshot(head + DVec3::Y * 0.119, head, r));
        assert!(!is_headshot(head + DVec3::Y * 0.121, head, r));
        // Horizontally: the CAPSULE's surface is in, and half a metre out is
        // not — which is what makes the test reachable at all.
        assert!(is_headshot(head + DVec3::X * 0.30, head, r));
        assert!(!is_headshot(head + DVec3::X * 0.50, head, r));
        // The pelvis of a 1.8 m character is nowhere near it.
        assert!(!is_headshot(head - DVec3::Y * 0.7, head, r));
        // A body with no radius at all still has a head.
        assert!(is_headshot(head, head, 0.0));
    }

    /// **The integrator falls like the parabola** when there is no drag — the
    /// closed form, at 200 m of a flat 900 m/s shot.
    #[test]
    fn a_dragless_round_drops_by_one_half_g_t_squared() {
        let def = WeaponDef {
            drag_k: 0.0,
            gravity_scale: 1.0,
            ..Default::default()
        };
        let (mut at, mut v) = (DVec3::ZERO, DVec3::new(0.0, 0.0, 900.0));
        let sub_dt = 1.0 / 60.0 / f64::from(PROJECTILE_SUB_STEPS);
        let mut t = 0.0;
        while at.z < 200.0 {
            let (n_at, n_v) = advance_round(at, v, &def, sub_dt);
            at = n_at;
            v = n_v;
            t += sub_dt;
        }
        // **The DISCRETE closed form, exactly.** Semi-implicit Euler updates
        // the velocity before the position, so after n sub-steps of h the drop
        // is `g·h²·n(n+1)/2` = `½·g·t·(t+h)` — not `½·g·t²`, which is the
        // continuous parabola and is what the round would follow at h → 0. The
        // arm asserts the form the integrator actually implements to 1e-9 and
        // then states the gap to the continuous one, because a tolerance wide
        // enough to hide the difference is a tolerance wide enough to hide a
        // wrong integrator.
        let discrete = -0.5 * PROJECTILE_GRAVITY_MPS2 * t * (t + sub_dt);
        let continuous = -0.5 * PROJECTILE_GRAVITY_MPS2 * t * t;
        println!(
            "at {:.3} m after {t:.4} s: y {:.6}, discrete {discrete:.6}, continuous {continuous:.6} (gap {:.2} mm)",
            at.z,
            at.y,
            (at.y - continuous).abs() * 1000.0
        );
        assert!(
            (at.y - discrete).abs() < 1.0e-9,
            "at {:.3} m the round is at y {:.9} and the discrete form says {discrete:.9}",
            at.z,
            at.y
        );
        assert!(
            (at.y - continuous).abs() < 1.0e-2,
            "the discretization is worth {:.4} m at 200 m, which is not a bullet drop",
            (at.y - continuous).abs()
        );
    }

    /// **Drag takes speed off, in the direction of travel and nowhere else.**
    #[test]
    fn drag_slows_a_round_and_does_not_turn_it() {
        let def = WeaponDef {
            drag_k: 0.0003,
            gravity_scale: 0.0,
            ..Default::default()
        };
        let (mut at, mut v) = (DVec3::ZERO, DVec3::new(0.0, 0.0, 900.0));
        let sub_dt = 1.0 / 60.0 / f64::from(PROJECTILE_SUB_STEPS);
        while at.z < 200.0 {
            let (n_at, n_v) = advance_round(at, v, &def, sub_dt);
            at = n_at;
            v = n_v;
        }
        assert!(v.z < 900.0 && v.z > 800.0, "200 m left it at {} m/s", v.z);
        assert_eq!((v.x, v.y), (0.0, 0.0), "drag steered the round");
    }

    /// **A round at rest does not produce a NaN.** The `v/|v|` the doc writes is
    /// exactly where one comes from, and a NaN position is a round that never
    /// hits anything again.
    #[test]
    fn a_stationary_round_stays_finite() {
        let def = WeaponDef::default();
        let (at, v) = advance_round(DVec3::ZERO, DVec3::ZERO, &def, 1.0 / 240.0);
        assert!(at.is_finite() && v.is_finite());
    }

    /// **The pool refuses past its bound, with a value** — and the bound is
    /// derived from the ray ceiling rather than chosen beside it.
    #[test]
    fn the_pool_refuses_the_sixty_fifth_round_and_counts_it() {
        assert_eq!(
            MAX_ROUNDS_IN_FLIGHT * PROJECTILE_SUB_STEPS as usize,
            MAX_SHOT_RAYS_PER_STEP
        );
        let mut w = EcsWorld::new();
        assert!(round_pool(&w).is_none(), "a quiet world has no pool");
        let r = Round {
            shooter: Uuid::from_u128(1),
            at: DVec3::ZERO,
            velocity: DVec3::Z * 900.0,
            travelled_m: 25.0,
            age_s: 0.0,
            first_segment: true,
            cracked: false,
            def: WeaponDef::default(),
        };
        for i in 0..MAX_ROUNDS_IN_FLIGHT {
            assert!(spawn_round(&mut w, r, 0), "round {i} was refused early");
        }
        assert!(!spawn_round(&mut w, r, 0), "the pool took one too many");
        let pool = round_pool(&w).expect("a pool");
        assert_eq!(pool.rounds.len(), MAX_ROUNDS_IN_FLIGHT);
        assert_eq!(pool.spawned, MAX_ROUNDS_IN_FLIGHT as u64);
        assert_eq!(pool.refused, 1, "the refusal was not counted");
    }

    /// **The ceiling counts the hitscan half too.**
    #[test]
    fn a_step_that_already_spent_its_rays_refuses_the_spawn() {
        let mut w = EcsWorld::new();
        let r = Round {
            shooter: Uuid::from_u128(1),
            at: DVec3::ZERO,
            velocity: DVec3::Z * 900.0,
            travelled_m: 25.0,
            age_s: 0.0,
            first_segment: true,
            cracked: false,
            def: WeaponDef::default(),
        };
        assert!(!spawn_round(&mut w, r, MAX_SHOT_RAYS_PER_STEP));
        assert_eq!(round_pool(&w).expect("a pool").refused, 1);
    }

    /// **The bytes are empty on a quiet level and grow at 81 a round** — the
    /// property every pre-wave trace depends on.
    #[test]
    fn the_trace_section_is_empty_until_something_is_in_the_air() {
        let mut w = EcsWorld::new();
        assert!(round_state_bytes(&w).is_empty());
        let r = Round {
            shooter: Uuid::from_u128(7),
            at: DVec3::new(1.0, 2.0, 3.0),
            velocity: DVec3::Z * 900.0,
            travelled_m: 25.0,
            age_s: 0.0,
            first_segment: true,
            cracked: false,
            def: WeaponDef::default(),
        };
        assert!(spawn_round(&mut w, r, 0));
        assert_eq!(round_state_bytes(&w).len(), ROUND_TRACE_BYTES);
        // …and a pool that has emptied folds nothing again.
        w.world_mut().resource_mut::<RoundPool>().rounds.clear();
        assert!(
            round_state_bytes(&w).is_empty(),
            "a landed round left bytes behind"
        );
        clear_rounds(&mut w);
        assert!(round_pool(&w).is_none());
    }

    /// **The bytes see a round MOVE.** A section that folded a constant would
    /// pass every arm above and see nothing.
    #[test]
    fn the_bytes_change_when_a_round_does() {
        let mut w = EcsWorld::new();
        let r = Round {
            shooter: Uuid::from_u128(7),
            at: DVec3::ZERO,
            velocity: DVec3::Z * 900.0,
            travelled_m: 25.0,
            age_s: 0.0,
            first_segment: true,
            cracked: false,
            def: WeaponDef::default(),
        };
        assert!(spawn_round(&mut w, r, 0));
        let before = round_state_bytes(&w);
        w.world_mut().resource_mut::<RoundPool>().rounds[0].at = DVec3::new(0.0, 0.0, 1.0);
        assert_ne!(before, round_state_bytes(&w));
    }
}
