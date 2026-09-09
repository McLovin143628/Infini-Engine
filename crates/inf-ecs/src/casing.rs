//! **The brass** (wave WPN2c) — a bounded ring of shell casings, spawned at a
//! weapon's ejection port, integrated with gravity and one bounce, and drawn.
//!
//! # The shape, and why it is this one
//!
//! [`crate::ballistics::RoundPool`]'s doctrine, one system over: a bevy
//! resource, never serialized, absent on a level that has never fired, wiped by
//! `clear_casings` on the way into and out of a Simulate session, and folded at
//! the **TAIL** of `RuntimeSim::state_bytes` — **empty when the pool is**, which
//! is what keeps every trace committed before this wave byte-identical.
//!
//! # A RING, not a refusal — and the difference is the point
//!
//! `spawn_round` REFUSES when the pool is full and counts the refusal, because
//! a round that was never minted is a shot the player fired that hurt nobody,
//! and that has to be accountable. A casing is the other way round: it is
//! decoration with a physics body, and the honest failure mode for the
//! hundred-and-twenty-ninth one is that the oldest disappears. So this pool
//! **recycles**, counts it, and never refuses. The research doc asks for
//! exactly this — *"do not allocate dynamic memory per shell; pre-allocate a
//! ring buffer"*.
//!
//! # What it costs
//!
//! One raycast per **airborne** casing per fixed step. A settled one costs an
//! age increment and nothing else, which is what makes an eight-second lifetime
//! affordable: at eight shooters and 600 rpm the pool is full of brass and
//! almost all of it is on the ground.

use bevy_ecs::prelude::Resource;
use glam::DVec3;
use uuid::Uuid;

use crate::weapon::{WeaponClass, WeaponDef, CASING_CLIP};
use crate::EcsWorld;

/// **How many casings exist at once.**
///
/// A hundred and twenty-eight — twice [`crate::ballistics::MAX_ROUNDS_IN_FLIGHT`]
/// and for the opposite reason. Rounds are bounded by the RAY budget, because
/// every one of them costs four casts a step for its whole life; casings are
/// bounded by what a person can see, because a settled one costs no ray at all.
/// At eight shooters and 600 rpm the pool fills in 1.6 s and then recycles, so
/// the floor of a firefight carries about a second and a half of brass, which
/// is what the reference footage shows.
pub const MAX_CASINGS_LIVE: usize = 128;

/// **How long a casing lives**, seconds — the brief's own eight, and
/// [`crate::ballistics::MAX_ROUND_LIFETIME_S`]'s twin.
pub const CASING_LIFETIME_S: f64 = 8.0;

/// **How much of its speed a casing keeps in a bounce** — the research doc
/// section 3's own `e ~= 0.3`.
///
/// Brass on concrete really is about this: it hops once, audibly, and then
/// skitters. The second contact stops it (see [`CASING_SETTLE_CONTACTS`])
/// rather than a third and fourth hop being simulated, which is the honest
/// trade for a thing that is decoration with a physics body.
pub const CASING_RESTITUTION: f64 = 0.3;

/// **How many contacts settle a casing** — two.
///
/// The first is the bounce and makes the noise; the second is where it stops.
/// A real case bounces three or four times over about a second, and the third
/// hop is 3 % of the original energy and inaudible; simulating it would cost a
/// ray a step for every casing on the floor of a firefight, which is the whole
/// cost this pool is built to avoid.
pub const CASING_SETTLE_CONTACTS: u8 = 2;

/// Gravity on a casing, m/s² — [`crate::ballistics::PROJECTILE_GRAVITY_MPS2`]'s
/// own number, named separately because a designer who wanted low-gravity brass
/// should not have to change what a bullet does.
pub const CASING_GRAVITY_MPS2: f64 = 9.81;

/// **How far above the horizontal the port throws**, degrees.
///
/// Twenty-five. One constant for every weapon, because no gun in the research
/// doc's seven tables ejects anywhere but up and out, and a per-weapon rise
/// would be three numbers on a registry row that nobody could check against
/// anything. The AZIMUTH is per-weapon ([`WeaponDef::eject_dir_deg`]) because
/// that one really does differ — a left-handed AR ejects the other way.
pub const EJECT_RISE_DEG: f64 = 25.0;

/// **How fast a casing tumbles**, degrees per second, at most.
///
/// The doc's *"strong rotational torque"*. Seven hundred and twenty is two
/// turns a second, which reads as tumbling rather than as spinning, and the
/// actual rate per axis comes off the counter hash so two casings do not tumble
/// alike.
pub const CASING_SPIN_DEG_S: f64 = 720.0;

/// How long a shell is, metres — a 9 mm case. The draw's `Z` scale.
pub const CASING_LENGTH_M: f64 = 0.019;

/// How fat a shell is, metres. The draw's `X`/`Y` scale.
pub const CASING_RADIUS_M: f64 = 0.005;

/// **How many bytes one casing folds into the trace** — 16 for the shooter's
/// guid, thirteen f64 for the three vectors plus the age, and one for the
/// contact count. Named so the fold and the arm that measures it cannot
/// disagree about the arithmetic, which they did on this module's first draft
/// (the doc said 89 and the fold wrote 121).
pub const CASING_TRACE_BYTES: usize = 16 + 13 * 8 + 1;

/// **How far a casing's landing pitch may wander**, as a fraction.
///
/// A quarter, either way — the doc section 3's *"pitch-randomized metallic"*
/// drop, which is much wider than a gunshot body's two per cent because two
/// cases landing at the same pitch is the thing that makes brass sound fake.
/// Drawn from [`crate::weapon::shot_uniforms`], so there is no RNG in a fixed
/// step and a replay drops the same brass at the same notes.
pub const CASING_PITCH_SPREAD: f64 = 0.25;

/// **The volume a casing lands at.** Quiet: it is a thing on the floor, heard
/// beside a gunshot that is nine tenths of full scale.
pub const CASING_VOLUME: f64 = 0.35;

/// Metres inside which a landing casing is at full volume.
pub const CASING_MIN_M: f64 = 1.0;

/// **Metres past which a landing casing is silent.** Twelve — a room, and not
/// a street. The gunshot that threw it carries two hundred and fifty.
pub const CASING_MAX_M: f64 = 12.0;

/// **The salt a casing's audio source key carries** — the fifth member of
/// `crate::weapon::LAYER_SALTS`' family, and for its reason: a casing's own
/// entity is a thing in the world that could carry an `AudioSource`, and its
/// landing must not take that voice.
pub const CASING_SALT: u64 = 0x5750_4e32_0000_00f0;

/// **The marker on a casing's drawn entity** (wave WPN2c).
///
/// A runtime component, never serialized, so no schema moves. It exists because
/// the draw pass has to answer *"which entities are casings that no longer
/// exist"* every step, and the alternative — a second list of guids in a second
/// resource — is a second thing that can disagree with the pool.
#[derive(bevy_ecs::prelude::Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CasingMark;

/// **How far off a surface a bounced casing is placed**, metres.
///
/// Five millimetres — one casing radius. A contact point left exactly on the
/// surface starts the next step's ray inside the thing it just hit, which reads
/// as a casing that fell through the floor.
pub const CASING_CONTACT_EPS_M: f64 = 0.005;

/// **One casing.**
///
/// It carries its own [`WeaponClass`] rather than a reference to the weapon that
/// threw it, on [`crate::ballistics::Round`]'s argument verbatim: by the time
/// this lands the shooter may have scrolled, and a 12-gauge hull that turned
/// into a 9 mm case in mid-air would be a defect nobody could explain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Casing {
    /// Who ejected it.
    pub shooter: Uuid,
    /// Where it is, world metres.
    pub at: DVec3,
    /// How fast it is going, m/s. Exactly zero once settled.
    pub velocity: DVec3,
    /// How fast it tumbles, degrees per second about each world axis.
    pub spin_deg_s: DVec3,
    /// How far it has tumbled, degrees — what the draw's rotation is.
    pub angle_deg: DVec3,
    /// How long it has existed, seconds.
    pub age_s: f64,
    /// How many surfaces it has touched. [`CASING_SETTLE_CONTACTS`] stops it.
    pub contacts: u8,
    /// What kind of gun threw it.
    pub class: WeaponClass,
    /// **Its ordinal in this session** — the counter its guid, its tumble and
    /// its landing pitch are all derived from, so none of the three needs a
    /// random number in a fixed step.
    pub seq: u64,
}

impl Casing {
    /// Whether this casing has come to rest.
    pub fn settled(&self) -> bool {
        self.contacts >= CASING_SETTLE_CONTACTS
    }
}

/// **Every casing on the floor and in the air** — the pool, a bevy resource.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct CasingPool {
    /// The live casings, in spawn order. **The only field
    /// [`casing_state_bytes`] folds.**
    pub casings: Vec<Casing>,
    /// How many have ever been ejected in this session — and the counter every
    /// casing's `seq` comes from, so it must never go backwards.
    pub spawned: u64,
    /// **How many were RECYCLED** because the ring was full. Not a refusal: the
    /// casing was ejected, and the oldest one on the floor vanished instead.
    pub recycled: u64,
    /// How many first contacts have happened — the number of landing sounds.
    pub bounced: u64,
    /// How many have come to rest.
    pub settled: u64,
    /// How many aged out.
    pub expired: u64,
}

/// The pool, if this world has ever ejected a casing.
pub fn casing_pool(world: &EcsWorld) -> Option<&CasingPool> {
    world.world().get_resource::<CasingPool>()
}

/// **How many casings exist right now.** `0` on a world that has never fired.
pub fn casings_live(world: &EcsWorld) -> usize {
    casing_pool(world).map(|p| p.casings.len()).unwrap_or(0)
}

/// **Throw a casing out of a barrel** — the one door the fire path uses.
///
/// It builds the [`Casing`] rather than taking one, because two of its fields
/// are functions of an ordinal the pool has not assigned yet: the tumble comes
/// off the counter hash and so does the note it will land on. A caller that
/// built the struct itself would have to read `CasingPool::spawned` first, which
/// is the pool's business and not the fire path's.
pub fn eject_casing(
    world: &mut EcsWorld,
    shooter: Uuid,
    class: WeaponClass,
    at: DVec3,
    velocity: DVec3,
) -> Option<u64> {
    spawn_casing(
        world,
        Casing {
            shooter,
            at,
            velocity,
            // Filled by `spawn_casing` from the ordinal it assigns.
            spin_deg_s: DVec3::ZERO,
            angle_deg: DVec3::ZERO,
            age_s: 0.0,
            contacts: 0,
            class,
            seq: 0,
        },
    )
}

/// **Eject one casing.** Always succeeds; when the ring is full the OLDEST is
/// recycled and counted.
///
/// Answers the casing's own `seq`, which the caller needs for its guid. The
/// **tumble is assigned here** rather than by the caller, because it is a
/// function of that ordinal — see [`eject_casing`].
pub fn spawn_casing(world: &mut EcsWorld, mut casing: Casing) -> Option<u64> {
    if !casing.at.is_finite() || !casing.velocity.is_finite() {
        return None;
    }
    let w = world.world_mut();
    if w.get_resource::<CasingPool>().is_none() {
        w.insert_resource(CasingPool::default());
    }
    let mut pool = w.resource_mut::<CasingPool>();
    let seq = pool.spawned;
    casing.seq = seq;
    if casing.spin_deg_s == DVec3::ZERO {
        casing.spin_deg_s = eject_spin(seq);
    }
    if pool.casings.len() >= MAX_CASINGS_LIVE {
        pool.casings.remove(0);
        pool.recycled += 1;
    }
    pool.casings.push(casing);
    pool.spawned += 1;
    Some(seq)
}

/// **Every entity currently drawn as brass**, in `Guid` order (wave WPN2c).
///
/// The draw pass's other half: it knows which casings the POOL holds and needs
/// to know which entities EXIST, so that the ones the pool has recycled or aged
/// out can be despawned. The query lives here rather than in
/// `inf_physics::d3::gameplay` because a `With<...>` filter is a `bevy_ecs`
/// type and this crate is the only one allowed to name one (architecture rule
/// 1); the caller gets guids.
///
/// `None` from the query — a world whose archetypes have never held the marker
/// — is an empty list, which is the right answer and not an error.
pub fn drawn_casings(world: &EcsWorld) -> Vec<Uuid> {
    let w = world.world();
    let Some(mut q) =
        w.try_query_filtered::<&crate::components::Guid, bevy_ecs::prelude::With<CasingMark>>()
    else {
        return Vec::new();
    };
    let mut out: Vec<Uuid> = q.iter(w).map(|g| g.0).collect();
    out.sort_unstable();
    out
}

/// **Forget every casing** — the Simulate session's door, on
/// [`crate::ballistics::clear_rounds`]' terms exactly: run 2 of a session must
/// begin where run 1 did, and brass on the floor when the author pressed stop
/// is run 1's.
pub fn clear_casings(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<CasingPool>();
}

/// **The casings' trace bytes** — 121 a casing (16 guid + thirteen f64 + one
/// flag), in spawn order, and **empty when the pool is**.
///
/// Empty is the load-bearing half, exactly as it is for
/// [`crate::ballistics::round_state_bytes`]: a level that has never fired folds
/// nothing, and so does one whose brass has all aged out. The counters above are
/// not in here for the same reason they are not in there.
pub fn casing_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(pool) = casing_pool(world) else {
        return Vec::new();
    };
    if pool.casings.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pool.casings.len() * CASING_TRACE_BYTES);
    for c in &pool.casings {
        out.extend_from_slice(c.shooter.as_bytes());
        for v in [
            c.at.x,
            c.at.y,
            c.at.z,
            c.velocity.x,
            c.velocity.y,
            c.velocity.z,
            c.spin_deg_s.x,
            c.spin_deg_s.y,
            c.spin_deg_s.z,
            c.angle_deg.x,
            c.angle_deg.y,
            c.angle_deg.z,
            c.age_s,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.push(c.contacts);
    }
    out
}

/// **The entity a casing is drawn as**, derived from its shooter and its
/// ordinal.
///
/// Content-derived rather than minted, which is `equipped_weapon_guid`'s own
/// rule and the law's: a random guid in a fixed step is two hosts spawning
/// different entities on the same step. The splitmix64 fold is
/// [`crate::weapon::shot_uniforms`]' own mixer, over the shooter's own bits.
pub fn casing_guid(shooter: Uuid, seq: u64) -> Uuid {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    };
    let lo = (shooter.as_u128() as u64) ^ seq.wrapping_mul(0x2545_f491_4f6c_dd1d);
    let hi = ((shooter.as_u128() >> 64) as u64) ^ 0x5750_4e32_b8a5_0000;
    Uuid::from_u64_pair(mix(hi), mix(lo))
}

/// **Where the ejection port is**, world metres — the muzzle, moved to the
/// right of the aim line by the weapon's own offset.
///
/// Right is `yaw + 90` in this engine's compass convention (`+Z` at zero, `+X`
/// at `+90` — see [`crate::weapon::aim_forward`]), and the port is at the
/// muzzle's height rather than at the breech's because the muzzle is the one
/// point on a weapon this engine actually knows (`muzzle_of`). A real ejection
/// port is 20-30 cm behind it, which is a socket wave WPN2d's real weapon
/// meshes bring and this cannot invent.
pub fn eject_point(muzzle: DVec3, def: &WeaponDef, yaw_deg: f64) -> DVec3 {
    let right = crate::weapon::aim_forward(yaw_deg + 90.0, 0.0);
    muzzle + right * def.eject_offset_m.clamp(0.0, 1.0)
}

/// **Which way and how fast the brass leaves**, m/s.
///
/// The doc section 3's *"initial outward linear impulse"*: along the weapon's
/// own azimuth right of the aim ([`WeaponDef::eject_dir_deg`]), raised by
/// [`EJECT_RISE_DEG`], at the weapon's own speed. Portable trigonometry
/// throughout — this reaches `casing_state_bytes` and therefore the replay
/// trace, and std trig is not bit-portable (the P14 law).
pub fn eject_velocity(def: &WeaponDef, yaw_deg: f64) -> DVec3 {
    crate::weapon::aim_forward(yaw_deg + def.eject_dir_deg, EJECT_RISE_DEG)
        * def.eject_speed_mps.clamp(0.0, 20.0)
}

/// **How a casing tumbles**, degrees per second per axis, from its ordinal.
///
/// Two draws from [`crate::weapon::shot_uniforms`] make three axes by using the
/// first for `x`, the second for `y` and their difference for `z` — one hash
/// call rather than two, and the third axis is as uncorrelated as it needs to
/// be for a thing that is spinning too fast to look at.
pub fn eject_spin(seq: u64) -> DVec3 {
    let (a, b) = crate::weapon::shot_uniforms(CASING_SALT, seq);
    DVec3::new(
        (a * 2.0 - 1.0) * CASING_SPIN_DEG_S,
        (b * 2.0 - 1.0) * CASING_SPIN_DEG_S,
        ((a - b) * 2.0) * CASING_SPIN_DEG_S,
    )
}

/// **What note a casing lands on** — the doc's pitch randomisation, from the
/// counter hash.
///
/// `1 +/- CASING_PITCH_SPREAD`, and it is a **function of the ordinal alone**,
/// so the seventeenth case of a session lands on the same note in a replay, in
/// a PIE preview and in a shipped build.
pub fn bounce_pitch(seq: u64) -> f64 {
    let (_, u) = crate::weapon::shot_uniforms(CASING_SALT ^ 0x9e37, seq);
    1.0 + (u - 0.5) * 2.0 * CASING_PITCH_SPREAD
}

/// **The `AudioSource` a landing casing plays** — one Ring-0 description, on
/// `crate::weapon::report_source`'s own terms and for its reason.
///
/// **One clip for every class today.** The `class` argument is taken and
/// deliberately not read: a 12-gauge hull and a 9 mm case sound different, and
/// the difference is content (a per-calibre clip) rather than engineering. The
/// argument is here so the day the clips exist the signature does not move, and
/// the PITCH — which is per casing and is a quarter either way — is what makes
/// a floor of brass sound like brass rather than like one sample.
///
/// A twelve-metre reach: brass on concrete is a room-sized sound, and the
/// gunshot that threw it carries two hundred and fifty.
pub fn casing_source(_class: WeaponClass) -> crate::components::AudioSource {
    crate::components::AudioSource {
        clip: Some(CASING_CLIP),
        bus: crate::weapon::REPORT_BUS.to_string(),
        volume: CASING_VOLUME,
        pitch: 1.0,
        looping: false,
        spatial: true,
        min_distance: CASING_MIN_M,
        max_distance: CASING_MAX_M,
        distance_model: crate::components::DistanceModel::Inverse,
        rolloff: crate::weapon::REPORT_ROLLOFF,
        occlusion: false,
        autoplay: false,
    }
}

/// **The source key a casing's landing plays on** — its own entity's key,
/// salted, so a casing that is also an emitter keeps its own voice.
pub fn casing_source_key(casing: Uuid) -> u64 {
    (casing.as_u128() as u64) ^ CASING_SALT
}

/// **One step of a casing's fall**, as a pure function: `(next position, next
/// velocity)`.
///
/// Semi-implicit Euler with the velocity updated first, which is
/// [`crate::ballistics::advance_round`]'s own integrator and for its reason:
/// the position uses the velocity it will leave with. No drag — a brass case is
/// dense and slow and the term would be four significant figures below the
/// gravity it is added to.
pub fn advance_casing(at: DVec3, velocity: DVec3, dt: f64) -> (DVec3, DVec3) {
    let v = velocity + DVec3::new(0.0, -CASING_GRAVITY_MPS2, 0.0) * dt;
    (at + v * dt, v)
}

/// **What a casing does when it hits something** — the reflected, damped
/// velocity.
///
/// `v - 2(v.n)n` scaled by [`CASING_RESTITUTION`], with a normal that is
/// normalised here rather than trusted, and a zero-length normal answering a
/// dead stop rather than a NaN.
pub fn bounce_velocity(velocity: DVec3, normal: DVec3) -> DVec3 {
    let n = normal.normalize_or_zero();
    if n == DVec3::ZERO {
        return DVec3::ZERO;
    }
    (velocity - n * (2.0 * velocity.dot(n))) * CASING_RESTITUTION
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_casing(seq: u64) -> Casing {
        Casing {
            shooter: Uuid::from_u128(0x11),
            at: DVec3::new(0.0, 1.5, 0.0),
            velocity: DVec3::new(2.0, 1.0, 0.0),
            spin_deg_s: eject_spin(seq),
            angle_deg: DVec3::ZERO,
            age_s: 0.0,
            contacts: 0,
            class: WeaponClass::Ar,
            seq,
        }
    }

    /// **A RING, NOT A REFUSAL** — and the counters say which happened.
    #[test]
    fn the_pool_recycles_its_oldest_rather_than_refusing_a_casing() {
        let mut w = EcsWorld::new();
        assert_eq!(casings_live(&w), 0);
        assert!(casing_state_bytes(&w).is_empty());
        for i in 0..MAX_CASINGS_LIVE as u64 {
            assert_eq!(spawn_casing(&mut w, a_casing(i)), Some(i));
        }
        assert_eq!(casings_live(&w), MAX_CASINGS_LIVE);
        assert_eq!(casing_pool(&w).unwrap().recycled, 0);
        // The one over the ceiling recycles the oldest; it is never refused.
        let over = spawn_casing(&mut w, a_casing(999)).expect("a casing");
        assert_eq!(over, MAX_CASINGS_LIVE as u64);
        assert_eq!(casings_live(&w), MAX_CASINGS_LIVE);
        assert_eq!(casing_pool(&w).unwrap().recycled, 1);
        // …and the oldest is the one that went: sequence 0 is gone.
        assert!(casing_pool(&w)
            .unwrap()
            .casings
            .iter()
            .all(|c| c.seq != 0));
        // The bytes are 121 a casing and empty again once the pool is cleared.
        assert_eq!(
            casing_state_bytes(&w).len(),
            MAX_CASINGS_LIVE * CASING_TRACE_BYTES
        );
        clear_casings(&mut w);
        assert!(casing_state_bytes(&w).is_empty());
        assert_eq!(casings_live(&w), 0);
    }

    /// A guid is a function of the shooter and the ordinal — never of a clock,
    /// a counter in a host, or anything else two hosts could disagree about.
    #[test]
    fn a_casings_identity_is_derived_and_never_minted() {
        let a = Uuid::from_u128(0xAA);
        let b = Uuid::from_u128(0xBB);
        assert_eq!(casing_guid(a, 7), casing_guid(a, 7));
        assert_ne!(casing_guid(a, 7), casing_guid(a, 8));
        assert_ne!(casing_guid(a, 7), casing_guid(b, 7));
        // Two hundred casings from two shooters: no collisions.
        let mut seen = std::collections::BTreeSet::new();
        for s in [a, b] {
            for i in 0..100u64 {
                assert!(seen.insert(casing_guid(s, i)));
            }
        }
        // …and its audio key is not its own entity key.
        let g = casing_guid(a, 3);
        assert_ne!(casing_source_key(g), g.as_u128() as u64);
    }

    /// **THE PORT AND THE THROW** — right of the aim, up by the rise, at the
    /// weapon's own speed, and all of it portable trigonometry.
    #[test]
    fn the_brass_leaves_to_the_right_and_upward() {
        let def = WeaponDef {
            eject_offset_m: 0.06,
            eject_dir_deg: 80.0,
            eject_speed_mps: 2.4,
            ..Default::default()
        };
        // Facing +Z (yaw 0): right is +X.
        let port = eject_point(DVec3::ZERO, &def, 0.0);
        assert!(port.x > 0.05 && port.x < 0.07, "{port:?}");
        assert!(port.z.abs() < 1e-9);
        let v = eject_velocity(&def, 0.0);
        assert!(v.x > 0.0, "the brass went left: {v:?}");
        assert!(v.y > 0.0, "the brass went down: {v:?}");
        // A micronewton of slack: `aim_forward` is `psin64`/`pcos64`, whose
        // Pythagorean identity holds to about a part in a trillion rather than
        // exactly — which is the price of being bit-portable and is what this
        // vector is built out of on purpose.
        assert!((v.length() - def.eject_speed_mps).abs() < 1e-6);
        // Facing +X (yaw 90): right is -Z, and the whole thing rotates with it.
        let v90 = eject_velocity(&def, 90.0);
        assert!(v90.z < 0.0, "{v90:?}");
        assert!((v90.y - v.y).abs() < 1e-9, "the rise is not a function of yaw");
        // A left-handed port throws the other way.
        let left = WeaponDef {
            eject_dir_deg: -80.0,
            ..def
        };
        assert!(eject_velocity(&left, 0.0).x < 0.0);
    }

    /// The integrator falls, the bounce damps, and a casing settles on its
    /// second contact rather than for ever.
    #[test]
    fn a_casing_falls_bounces_at_three_tenths_and_settles_on_the_second_contact() {
        let (p, v) = advance_casing(DVec3::new(0.0, 2.0, 0.0), DVec3::ZERO, 1.0 / 60.0);
        assert!(v.y < 0.0 && p.y < 2.0);
        // A vertical drop onto a flat floor comes back up at a third.
        let up = bounce_velocity(DVec3::new(0.0, -4.0, 0.0), DVec3::Y);
        assert!((up.y - 4.0 * CASING_RESTITUTION).abs() < 1e-12, "{up:?}");
        // A degenerate normal is a dead stop, not a NaN.
        assert_eq!(bounce_velocity(DVec3::new(1.0, -1.0, 0.0), DVec3::ZERO), DVec3::ZERO);
        let mut c = a_casing(0);
        assert!(!c.settled());
        c.contacts = 1;
        assert!(!c.settled());
        c.contacts = CASING_SETTLE_CONTACTS;
        assert!(c.settled());
    }

    /// **THE PITCH IS THE HASH** — wide, deterministic, and different for every
    /// casing, which is what stops a floor of brass sounding like one sample.
    #[test]
    fn every_casing_lands_on_its_own_note_and_the_same_one_in_a_replay() {
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..128u64 {
            let p = bounce_pitch(i);
            assert!(
                (p - 1.0).abs() <= CASING_PITCH_SPREAD + 1e-12,
                "casing {i} landed at {p}"
            );
            seen.insert(p.to_bits());
        }
        assert!(seen.len() > 120, "only {} distinct notes", seen.len());
        assert_eq!(bounce_pitch(42), bounce_pitch(42));
        // The tumble is drawn from the same counter and is bounded.
        for i in 0..64u64 {
            let s = eject_spin(i);
            assert!(s.is_finite());
            assert!(s.abs().max_element() <= CASING_SPIN_DEG_S * 2.0 + 1e-9);
        }
        assert_ne!(eject_spin(1), eject_spin(2));
    }
}
