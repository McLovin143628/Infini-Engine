//! **The gameplay fixed step** (island wave I6): doors swing, weapons fire,
//! bodies stop working.
//!
//! One function ([`step_gameplay`]) that both hosts call, in the slot between
//! `step_character_movement` and the solver — the `step_pose_evaluation` shape
//! once more, so the editor's Simulate and the shipped player cannot resolve one
//! trigger pull differently.
//!
//! # What it does not do
//!
//! It does not spend energy at a **destructible**. A shot that hits a wall comes
//! back in [`GameplayReport::destruct`] and the host spends it through its own
//! `runtime_destruct_damage` wrapper — the one the `destruct.apply_damage` node
//! already goes through, which is where the `runtime_destruct` permission flag
//! is read and where the near-miss line is logged. Reaching past it would put a
//! second door on P22's damage, which is the exact defect the P22 "one door for
//! three paths" ruling exists to prevent.
//!
//! # The kick, and why it is on the notify
//!
//! `attack` on a locked door in reach arms a P29-style one-shot
//! ([`inf_ecs::weapon::KICK_TRIGGER`]) and **nothing else happens**. The impulse
//! lands when the animation says it does
//! ([`inf_ecs::weapon::KICK_NOTIFY`], consumed through
//! `inf_ecs::anim_bridge::consume_anim_notify`) — because a kick that broke the
//! lock the instant the button went down would break it before the leg moved,
//! and a notify seam that gameplay routes around is a notify seam nobody can
//! trust.
//!
//! A character with no rig has no notify, so the kick would never land: the
//! **pending kick has a fuse** ([`KICK_FUSE_S`]), and it fires on the fuse when
//! nothing animated it. Both paths are armed, and the report says which one ran.

use std::collections::BTreeSet;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{CharacterMovement, Transform};
use inf_ecs::door::{self, DoorSide, PendingKick};
use inf_ecs::feel::{self, ShotStance, WeaponFeel};
use inf_ecs::item::{self, Inventory};
use inf_ecs::weapon::{self, FireVerdict, Health, ShotKind, WeaponDef, WeaponState};
use inf_ecs::world::EcsWorld;

use super::ecs::PhysicsBridge3D;

/// How long a kick waits for its animation before landing anyway, seconds.
///
/// A ceiling, not a schedule: a rigged character's kick lands on
/// [`inf_ecs::weapon::KICK_NOTIFY`] and this never fires. It exists because a
/// headless run, an NPC with no state machine and every level in this repository
/// committed before I6 have no animation to wait for, and a verb that silently
/// did nothing on those would be the dead-key defect I5 spent a wave on.
pub const KICK_FUSE_S: f64 = 0.35;

/// **How wide the box a melee swing sweeps is**, metres (half-extent) -- wave
/// WPN2d's line-of-sight probe.
///
/// Twelve centimetres, so a 24 cm box: a fist is about 10 cm across and a knife
/// held in one is about that with the hand round it. It is a bound on what can
/// slip THROUGH: a ray between two capsule axes threads a half-open door that an
/// arm does not fit through, and this is the width that stops it.
pub const MELEE_BOX_HALF_M: f64 = 0.12;

/// **How many bodies one blast may spend joules on**, per explosion.
///
/// Thirty-two. `MAX_PANIC_SOURCES`' own kind of number: a cost bound, so a
/// rocket into a crowd costs a constant rather than a function of how many
/// people were standing there. It is four times the eight-shooter firefight the
/// audio eviction table prices, and what it refuses is COUNTED
/// (`RoundReport::blast_targets_refused`).
pub const MAX_BLAST_TARGETS: usize = 32;

/// How far a hitscan shot may reach before the engine stops looking, metres —
/// the bound on `WeaponDef::range_m`, applied at the cast.
pub const SHOT_MAX_RANGE_M: f64 = weapon::MAX_RANGE_M;

/// Where a shot leaves a character **that has no weapon to read a muzzle off**,
/// metres above its feet.
///
/// # It used to be *the* muzzle, and SK1b is when it stopped being
///
/// Until this wave an equipped weapon was an inventory slot id and nothing else:
/// no entity, no transform, nothing in the world at all, so a shot had to start
/// at a height somebody picked. Chest height on a default capsule, so a shot
/// fired along the aim does not begin inside the ground on a downward pitch.
///
/// A character with a **rig** and an equipped weapon now carries that weapon as a
/// real entity attached to its `hand_r` socket, and `muzzle_of` reads the shot's
/// origin off the weapon's own muzzle. This is what is left: the answer for a
/// bare capsule — every level committed before this wave, the whole
/// `phase30-gameplay` fixture, and every test rig that steps gameplay without
/// stepping the pose. `the_new_muzzle_agrees_with_the_old_one_on_a_capsule_hero`
/// is the control that pins the two together, because "no gameplay regression"
/// is a claim about a number and not a feeling.
pub const MUZZLE_HEIGHT_M: f64 = 1.4;

/// The socket an equipped weapon hangs from.
///
/// The engine's own name for the right hand, published by every rig this engine
/// generates — [`inf_anim::manny`]'s twelve sockets include it under both the
/// engine spelling and ALS's `hand_r_socket`, and the twenty-joint template has
/// carried it since P24.1. A rig that does not publish it gets an attachment at
/// its entity origin (`inf_ecs::attach`'s documented fallback) and a muzzle from
/// the capsule rule, which is the same answer it got before this wave.
pub const WEAPON_SOCKET: &str = "hand_r";

/// **The socket a headshot is measured from** (wave WPN2a).
///
/// [`WEAPON_SOCKET`]'s sentence one joint along: `inf_anim::manny` publishes
/// `head` under both this engine's spelling and ALS's `head_socket`, and the
/// twenty-joint template has carried it since P24.1. A rig that does not publish
/// it sends `head_point` to the capsule rule and is COUNTED there
/// ([`RoundReport::heads_without_a_socket`]), which is the discipline
/// `muzzles_without_a_socket` already applies to the other end of the character.
pub const HEAD_SOCKET: &str = "head";

/// The salt that carves equipped weapons' GUID space out of the scene's own —
/// `item::dropped_item_guid`'s shape, with its own constant.
const EQUIPPED_WEAPON_SALT: u128 = 0x5745_4150_4f4e_5f45_5155_4950_5045_4421;

/// **The guid of the entity that IS a character's equipped weapon.**
///
/// Content-derived from the owner, the P22 idiom: a fixed step may not mint a
/// random guid, because two hosts stepping the same sim have to produce the same
/// entity or the trace forks on the first frame anything is equipped. One weapon
/// entity per character, re-used as the character switches weapons — a rifle and
/// a pistol are the same slot in the same hand.
pub fn equipped_weapon_guid(owner: Uuid) -> Uuid {
    let mut x = owner.as_u128() ^ EQUIPPED_WEAPON_SALT;
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

/// **One shot that landed** — the record a gate and the tracer both read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeaponHit {
    /// Who fired.
    pub shooter: Uuid,
    /// What was hit, if anything the bridge could name.
    pub target: Option<Uuid>,
    /// Where the shot started, world metres.
    pub from: DVec3,
    /// Where it ended — the hit, or the end of its range.
    pub to: DVec3,
    /// The energy it carried, joules.
    pub energy_j: f64,
    /// Whether the target absorbed it as **health** (a character) rather than
    /// as structure.
    pub on_flesh: bool,
    /// **Whether this record is a round ARRIVING rather than a trigger being
    /// pulled** (wave WPN2a).
    ///
    /// A hitscan is one record: the shot and the impact are the same event. A
    /// projectile is two — the pull (loud, at the muzzle, on the step the
    /// trigger went down) and the arrival (quiet, at the target, up to eight
    /// seconds later) — and a reader that could not tell them apart would file
    /// the second as a second crime. `step_witness`'s quiet-crime filter is the
    /// reader that needed it: `!loud && on_flesh` is a PUNCH by wave WPN1's own
    /// definition, and a bullet landing is not an assault by hand — it is the
    /// `Shot` act already on the record finally reaching somebody.
    pub arrived: bool,
    /// **Whether the hit point was on the target's head** (wave WPN2a) — the
    /// sphere test in `head_point`'s own doc, already applied to
    /// [`energy_j`](Self::energy_j).
    ///
    /// A record rather than a re-derivation: by the time a HUD, a gate or a
    /// witness reads this the pose has moved on, and asking again would answer
    /// about a head that is somewhere else. `false` for a swing, for a miss and
    /// for everything that is not flesh.
    pub headshot: bool,
    /// **How far this weapon's report carries**, metres — the per-weapon
    /// [`inf_ecs::weapon::WeaponDef::report_max_m`], travelling on the shot for
    /// [`loud`](Self::loud)'s reason exactly: what made the noise is a property
    /// of the shot and not of whatever is in the hand when it lands.
    ///
    /// Both hosts read it inside the `weapon_report` MIRROR fence. Every weapon
    /// authored before wave WPN2a carries `REPORT_MAX_M`, so every committed
    /// audio command stream is byte-identical.
    pub report_max_m: f64,
    /// **How loud this weapon's report is** (wave WPN2d), as the multiplier
    /// `inf_ecs::weapon::report_layers` applies to every layer's volume — the
    /// per-weapon `WeaponDef::report_gain`, travelling on the shot for
    /// [`report_max_m`](Self::report_max_m)'s reason verbatim.
    ///
    /// It is what makes a **suppressor** audible as a difference rather than as
    /// a table entry: the attachment fold multiplies this and `report_max_m` by
    /// the same `loudness_mult`, and both hosts read it inside the
    /// `weapon_report` MIRROR fence. `1.0` for everything the catalogue authors,
    /// so every committed audio command stream is byte-identical.
    pub report_gain: f64,
    /// **Whether this attack made a noise** (wave WPN1) — `true` for a round
    /// leaving a barrel, `false` for a swing.
    ///
    /// Two consumers, and neither could be written without it. Each host's
    /// `fire_weapon_audio` queues the gunshot report only for a loud attack, so
    /// a punch does not fire a rifle's clip; and the crowd's panic takes its
    /// sources only from loud attacks, which is what the reference frames
    /// actually show — an encampment **brawl** draws bystanders who stand a metre
    /// away and watch, while a gunshot is what empties a street.
    ///
    /// A field rather than a re-lookup of the shooter's equipped weapon, because
    /// by the time either consumer runs the shooter may have scrolled: what made
    /// the noise is a property of the shot and not of whatever is in the hand
    /// afterwards.
    pub loud: bool,
    /// **What kind of gun made the noise** (wave WPN2c) — the five clips a
    /// report is built from are chosen by it.
    ///
    /// On the shot for [`loud`](Self::loud)'s reason, third time: what a gunshot
    /// SOUNDS like is a property of the shot, and by the time the audio fence
    /// runs the shooter may have scrolled to a pistol.
    pub class: inf_ecs::weapon::WeaponClass,
    /// **Whether the muzzle was inside** (wave WPN2c) — the enclosure probe's
    /// verdict, taken at the muzzle on the step the trigger went down.
    ///
    /// It decides which of the two tail clips layer 3 plays. It is taken HERE
    /// and not in the hosts because it costs six raycasts against a physics
    /// world, and a value two hosts each computed for themselves is a value they
    /// can disagree about. See `super::audio::enclosure_at`.
    pub indoors: bool,
    /// **How far the listener was from the muzzle**, metres, or `f64::INFINITY`
    /// on a level with nobody listening (wave WPN2c).
    ///
    /// The distant layer's volume is a function of it. Sim-side for `indoors`'
    /// reason: each host has its own `active_listener`, and the two agreeing was
    /// a coincidence rather than a fence — see `inf_ecs::audio`.
    pub listener_m: f64,
    /// **Which round of the magazine this was** (wave WPN2c) — the counter the
    /// body layer's pitch jitter is drawn from.
    ///
    /// The doc's `rand_pitch(0.98, 1.02)` with no RNG behind it: a fixed step
    /// holds no random state, so the n-th round's pitch comes off
    /// [`inf_ecs::weapon::shot_uniforms`] and is the same in a replay.
    pub shot_index: u64,
}

/// **What the projectile pool did in one fixed step** (wave WPN2a).
///
/// Engagement counters on `CrowdDoorReport`'s own terms: without them "the
/// flight pass ran" and "a round moved" are the same fact, and a pool that
/// silently spawned nothing would look exactly like one that worked.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RoundReport {
    /// Rounds minted this step — shots that missed inside their weapon's
    /// hitscan threshold and had range left to fly through.
    pub spawned: u32,
    /// **Spawns REFUSED this step**, because the pool was full or the ray
    /// ceiling would have been crossed. The value the law asks for: a round
    /// dropped without a number is a shot the player fired and nobody can
    /// account for.
    pub refused: u32,
    /// Rounds in flight after this step's advance.
    pub in_flight: u32,
    /// **Segment casts this step's flight spent** — the number
    /// `inf_ecs::ballistics::MAX_SHOT_RAYS_PER_STEP` bounds, so the ceiling is
    /// a measurement rather than an assertion about arithmetic.
    pub rays: u32,
    /// Rounds that ended on a hit this step.
    pub impacts: u32,
    /// Rounds that reached their weapon's `range_m` or aged out.
    pub expired: u32,
    /// Rounds that left the active partition.
    pub left_band: u32,
    /// **Head hits this step**, from either half of the hybrid.
    pub headshots: u32,
    /// **Head tests that fell back to the capsule rule** — a character that
    /// publishes a pose whose rig authors no `head` socket. The muzzle's own
    /// `muzzles_without_a_socket` tripwire, one joint along: without it a rig
    /// that quietly lost its head socket is indistinguishable from a rig that
    /// never had one, and every headshot on it would be tested against a
    /// number instead of a bone.
    pub heads_without_a_socket: u32,
    /// **Casts the enclosure probe spent this step** (wave WPN2c) — six per
    /// loud trigger pull, and the reason the ray ceiling is priced at seven a
    /// shot rather than one.
    ///
    /// Its own counter beside [`rays`](Self::rays) because the two are bounded
    /// by the same ceiling and spent by different things: a gate that saw only
    /// the total could not say which half was about to cross it.
    pub probe_rays: u32,
    /// **Loud shots the probe called INDOORS this step** — an engagement
    /// counter, on `crowd_doors`' terms: "the probe ran" and "somebody fired
    /// inside a building" are different facts, and a gate that could not tell
    /// them apart would certify a probe that always answered `false`.
    pub indoor_shots: u32,
    /// **Casts the SHOT half spent this step** (wave WPN2d) — one per pellet
    /// plus its pull's enclosure probe, counted as they are spent rather than
    /// estimated from `shots`.
    ///
    /// It exists because a shotgun broke the estimate: a pull is one `shots` and
    /// up to `MAX_PELLETS` casts, so `shots × 7` was wrong by the pattern's own
    /// size — and this is the number the pellet loop refuses against and the
    /// number `ballistics::spawn_round` prices a round's flight against.
    pub shot_rays: u32,
    /// **Pellets thrown this step** (wave WPN2d) — the engagement counter that
    /// tells "a shotgun fired" from "a rifle fired".
    pub pellets: u32,
    /// **Pellets REFUSED this step** because the pull would have crossed
    /// `inf_ecs::ballistics::MAX_SHOT_RAYS_PER_STEP` — the value the law asks
    /// for, on `refused`'s own terms. A silently shortened pattern is a shot
    /// the player fired and nobody can account for.
    pub pellets_refused: u32,
    /// **Casts a blast's line-of-sight sweep spent** this step (wave WPN2d) —
    /// one per candidate inside the radius that is more than a millimetre from
    /// the epicentre.
    pub blast_rays: u32,
    /// **Bodies a wall saved** this step — candidates inside a blast's radius
    /// whose line of sight the sweep found blocked. The engagement counter that
    /// tells a blast which honours cover from one that does not.
    pub blast_shadowed: u32,
    /// **Bounces thrown bodies made** this step (wave WPN2d).
    pub bounces: u32,
    /// **Rounds that went off on their FUSE** this step, rather than on impact.
    pub fused: u32,
    /// **Sub-steps a guided round steered on** this step — the engagement
    /// counter on the guidance, so "a missile flew" and "a missile followed
    /// something" are different facts.
    pub guided: u32,
    /// **Thrown bodies that came to REST** this step (wave WPN2d) — out of
    /// bounces, with a fuse still running. The counter that tells a grenade
    /// lying on a pavement from one that has gone off.
    pub settled: u32,
}

/// **What the brass did in one fixed step** (wave WPN2c).
///
/// Engagement counters on [`RoundReport`]'s own terms, plus one list: a bounce
/// is a SOUND, and the hosts need to know where and at what pitch, so it is
/// carried rather than counted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CasingReport {
    /// Casings ejected this step — one per round that left a barrel.
    pub ejected: u32,
    /// Casings alive after this step's fall.
    pub live: u32,
    /// **Casts the fall spent** — one per AIRBORNE casing. A settled one costs
    /// nothing, which is the whole reason the ring can be a hundred and
    /// twenty-eight deep.
    pub rays: u32,
    /// Casings that came to rest this step.
    pub settled: u32,
    /// Casings that aged out this step.
    pub expired: u32,
    /// **Casings recycled over the session** because the ring was full — the
    /// pool's own running total, surfaced so a gate can say the ring wrapped.
    pub recycled: u64,
    /// **First contacts this step** — one landing sound each, built into a
    /// `Play` by both hosts inside the `casing_bounce` MIRROR fence.
    pub bounces: Vec<CasingBounce>,
}

/// **One explosion**, as the hosts have to hear it (wave WPN2d).
///
/// Carried rather than counted, on `CasingBounce`'s own terms: a blast is a
/// SOUND at a place, and the `weapon_blast` MIRROR fence in both hosts turns
/// this into one `Play` at `at`. The joules it spent are already gone through
/// `apply_hit` by the time a host sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlastEvent {
    /// Who fired the thing that went off — the source key the boom is salted
    /// off, so two rockets landing together are two voices.
    pub shooter: Uuid,
    /// Where it went off, world metres.
    pub at: DVec3,
    /// How far it reached, metres — the weapon's own `blast_radius_m`, carried
    /// so a debug draw and a gate read the radius that was actually spent
    /// rather than looking the weapon up again.
    pub radius_m: f64,
    /// **Bodies it spent joules on** — the engagement counter, so "a rocket
    /// landed" and "a rocket hurt somebody" are different facts.
    pub hurt: u32,
}

/// **One melee blow landing** (wave WPN2d) — what a host turns into a
/// surface-chosen contact one-shot.
///
/// `CasingBounce`'s shape exactly, and for its reason: which clip a blow plays
/// is a question about the WORLD (was it a body?) and the answer is decided in
/// the fixed step, where both hosts get it from the same place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeleeImpact {
    /// Who swung.
    pub shooter: Uuid,
    /// Where the blow landed, world metres.
    pub at: DVec3,
    /// What it landed on — the two-way stub
    /// (`inf_ecs::weapon::ImpactSurface`); VEH3a's per-surface table extends it.
    pub surface: inf_ecs::weapon::ImpactSurface,
}

/// **One casing hitting the ground for the first time** — what a host turns
/// into a pitch-randomised metallic one-shot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CasingBounce {
    /// The casing's own derived entity guid — the source key its voice is
    /// salted off, so two casings landing together are two voices.
    pub casing: Uuid,
    /// Where it hit, world metres.
    pub at: DVec3,
    /// The note it landed on, from the counter hash
    /// (`inf_ecs::casing::bounce_pitch`).
    pub pitch: f64,
    /// What kind of gun threw it. Carried for the day the clip is per-calibre;
    /// today one clip serves every class and the pitch is the difference.
    pub class: inf_ecs::weapon::WeaponClass,
}

/// What one fixed step of gameplay did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GameplayReport {
    /// The door system's own numbers.
    pub doors: super::door::DoorReport,
    /// **What the crowd did to the doors** this step (island wave NPC1c) — an
    /// engagement counter, because "the pass ran" and "an NPC opened
    /// something" are different facts and a gate that cannot tell them apart
    /// certifies a no-op.
    pub crowd_doors: CrowdDoorReport,
    /// **What the NPC cover pass did** (wave COV1) — every field an engagement
    /// counter, and `probes` zero on a step with no gunfire.
    pub npc_cover: super::cover::NpcCoverReport,
    /// Rounds fired this step.
    pub shots: u32,
    /// **What the brass did** this step (wave WPN2c).
    pub casings: CasingReport,
    /// **Rounds that went supersonically past the listener** this step (wave
    /// WPN2c) — one per round, ever, because [`inf_ecs::ballistics::Round`]
    /// latches it.
    ///
    /// A list rather than a count for [`CasingReport::bounces`]' reason: each
    /// one is a `Play` at a place, and the place is the closest point on the
    /// round's own segment to the ear rather than the muzzle it left.
    pub cracks: Vec<inf_ecs::ballistics::Crack>,
    /// **What the projectile pool did this step** (wave WPN2a) — every field an
    /// engagement counter, and every one of them zero on a level that has never
    /// fired a round, which is what tells "the pass ran" from "something flew".
    pub rounds: RoundReport,
    /// Reloads that finished this step.
    pub reloads: u32,
    /// Kicks that landed this step.
    pub kicks: u32,
    /// Doors whose lock a kick broke this step.
    pub locks_broken: u32,
    /// Characters that stopped working this step and were handed to the ragdoll.
    pub kills: u32,
    /// **What this step's gunfire did to the crowd** (wave WPN1) — an
    /// engagement counter, on [`crowd_doors`](Self::crowd_doors)' own terms.
    pub panic: PanicReport,
    /// **Acts recorded into the witness log** this step (wave WPN1) — an
    /// engagement counter for a seed nothing reads yet, which is exactly the
    /// case where one is worth most: without it "the witness pass runs" and
    /// "somebody saw something" are indistinguishable, and a seed that silently
    /// recorded nothing would look identical to a seed that worked.
    pub witnessed: u32,
    /// **Melee swings thrown** this step (wave WPN1) — the subset of
    /// [`shots`](Self::shots) that were an arc rather than a ray.
    ///
    /// Its own counter because the two are the same verb through the same
    /// trigger and the same clock: without it a gate cannot tell a course that
    /// punched from one that fired, and "the attack button has three consumers"
    /// is a claim about which one ran.
    pub swings: u32,
    /// **Non-fatal blows that landed on a body** this step (wave WPN1) — every
    /// one of them armed a hit reaction.
    ///
    /// An engagement counter, on `crowd_doors`' own terms: "a round was fired"
    /// and "a round hurt somebody" are different facts, and a gate that cannot
    /// tell them apart certifies a course where every shot missed.
    pub staggers: u32,
    /// **Struck bystanders who did NOT leave** this step (wave WPN1) — the
    /// resist half of the draw.
    ///
    /// Its own counter beside [`PanicReport::fled`], because a course where it
    /// is zero and a course where it is everything look identical from the flee
    /// count alone, and the whole point of a draw is that both happen.
    pub stood_their_ground: u32,
    /// **Blows heavy enough to put a body on the floor** this step — the subset
    /// of [`staggers`](Self::staggers) that also took a mode.
    ///
    /// Its own number rather than a flag on the one above, because the two
    /// answer different questions: `staggers` says the reaction seam is wired
    /// and `knockdowns` says the mode table let go — and a course where they are
    /// equal is a course where every punch is a rifle round.
    pub knockdowns: u32,
    /// **Explosions this step** (wave WPN2d), in the order they went off — the
    /// list both hosts' `weapon_blast` MIRROR fence turns into `Play`s.
    pub blasts: Vec<BlastEvent>,
    /// **Melee blows that landed** this step (wave WPN2d) — the list both hosts'
    /// `melee_impact` MIRROR fence turns into a surface-chosen `Play`.
    pub melee_impacts: Vec<MeleeImpact>,
    /// **Swings a wall stopped** this step (wave WPN2d) — the engagement
    /// counter on the box cast, so "the LOS test ran" and "somebody was refused
    /// a punch through a wall" are different facts. Wave WPN1's carried "no line
    /// of sight" defect is what it measures being closed.
    pub swings_blocked: u32,
    /// **Casts the melee LOS test spent** this step — one shape cast per swing
    /// that found a body in reach, and none at all for a swing that found
    /// nobody.
    pub melee_casts: u32,
    /// **Throws released this step** (wave WPN2d) — a body actually left a hand,
    /// which is a different fact from a throw animation starting.
    pub throws: u32,
    /// **Locks held this step** (wave WPN2d) — shooters whose launcher had a
    /// target in its cone, whether or not the hold is complete yet.
    pub locks_held: u32,
    /// **Locks that completed** this step — `lock_held_s` reached the weapon's
    /// own `lock_s`. The subset of [`locks_held`](Self::locks_held) a HUD draws
    /// as locked and a round leaves guided on.
    pub locks_complete: u32,
    /// Every shot that landed, in `Guid` order of the shooter.
    pub hits: Vec<WeaponHit>,
    /// Energy owed to the P22 damage door: `(destructible entity, joules)`.
    /// **The host spends this**, through its own wrapper. See the module header.
    pub destruct: Vec<(Uuid, f64)>,
    /// **What the hand pass asked for** this step, `(weapon holds, grabs)`
    /// (SK1c) — engagement counters, because "the hand step ran" and "a hand was
    /// asked to do something" are different facts and a gate that cannot tell
    /// them apart certifies a no-op.
    pub hands: (u32, u32),
    /// **Shots this step that fell back to [`MUZZLE_HEIGHT_M`] although the
    /// shooter publishes a pose** (SK1b audit) — the tripwire on the muzzle's
    /// silent half.
    ///
    /// `muzzle_of` has two answers: the weapon entity's own muzzle, and a height
    /// above the character's feet. The second is right for a **rig-less** hero —
    /// every level committed before SK1b, the whole `phase30-gameplay` fixture —
    /// and those are not counted, because they have no pose at all.
    ///
    /// What *is* counted is a character that has a rig and still took the capsule
    /// rule: its skeleton does not author [`WEAPON_SOCKET`], or its weapon entity
    /// has not been placed. Both put every shot back at 1.4 m above the feet
    /// **in silence**, which is a half-metre error on a crouched character and a
    /// shot through the floor on a prone one. A rigged course asserts this is
    /// zero; nothing else could tell the difference between the fallback working
    /// as designed and a rig that quietly lost its hand.
    pub muzzles_without_a_socket: u32,
}

/// **The gameplay fixed step.** Both hosts call it, between the character step
/// and the solver.
///
/// Inert on a level with no door, no weapon and no health: three
/// `try_query_filtered`s that answer `None`.
pub fn step_gameplay(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
) -> GameplayReport {
    let mut report = GameplayReport::default();
    if !dt.is_finite() || dt <= 0.0 {
        return report;
    }
    // 0. **ONE band and ONE placement gather for the whole phase** (NPC1c
    //    audit, closing the wave's own carried item 2). `placements_near` visits
    //    every `DoorwaySlot` a level plans — 19 790 on the shipped city — to
    //    keep the 234 the band admits, and NPC1c gave it a second per-step
    //    caller in `step_crowd_doors`. Gathering twice is the same walk twice;
    //    the placements are a function of the level's geometry and the band, and
    //    nothing between the two calls moves either.
    let band = bridge.sim_band(world);
    let places = super::door::placements_near(world, &band);
    // 0a. **An NPC opens the door in its way** (island wave NPC1c). Before the
    //    leaves move, so a crowd agent's press has the same immediacy the
    //    player's E already has -- that one is consumed in
    //    `step_character_movement`, a phase earlier than this whole function.
    //    Inert on every level with no crowd: one absent-resource read.
    report.crowd_doors = step_crowd_doors(world, &places);
    // 1. The doors move first, because a kick armed on a previous step lands in
    //    step 3 and must find the leaf where this step's solver will.
    let doors = super::door::step_doors_with(world, bridge, dt, places);
    report.doors = doors;
    // 1b. **Every round already in the air** (wave WPN2a) — the far half of the
    //     hybrid, integrated BEFORE this step's triggers.
    //
    //     The order is the one every physics loop uses and it is MEASURED rather
    //     than assumed: advancing the pool AFTER the trigger flew a round minted
    //     this step for a whole step immediately, so a rifle whose threshold is
    //     25 m put its round at **39.96 m** on the step it was fired — the
    //     instant ray's 25 m and a full 15 m of flight, double-counted.
    //     Integrating first means a round leaves the muzzle on the step the
    //     trigger went down and MOVES on the next, which is what a body leaving
    //     a barrel does.
    //
    //     It is still inside `step_gameplay` and still ABOVE the panic, the
    //     deaths and the witness pass, so a projectile kill is filed on the step
    //     it happened rather than one later — see `step_rounds` for why that
    //     ordering is the reason this is not a thirtieth `STEP_PHASES` row.
    //
    //     Inert on every level that has never fired a round: one absent-resource
    //     read.
    step_rounds(world, bridge, &band, dt, &mut report);
    // 1c. **Every casing already on the way down** (wave WPN2c), on 1b's
    //     argument verbatim: a case ejected this step leaves the port on the
    //     step the trigger went down and MOVES on the next, which is what a
    //     body leaving a port does. Inert on every level that has never fired.
    step_casings(world, bridge, dt, &mut report);
    // 1d. **What every launcher is pointing at** (wave WPN2d). BEFORE the
    //     trigger, because a round fired this step leaves on the lock the
    //     player was holding when they pulled — and a lock resolved after the
    //     shot would give the round a target it did not have. Inert on every
    //     level with no vehicles and on every character without a launcher: one
    //     absent-query read.
    step_locks(world, bridge, dt, &mut report);
    // 2. Every character with a weapon: the trigger, the reload, the clocks.
    step_weapons(world, bridge, dt, &mut report);
    // 2b. **The throw** (wave WPN2d) — the press, and the release on the clip's
    //     own notify. AFTER `step_weapons` because it shares the magazine and
    //     the clock that step installs, and because a character who scrolled to
    //     a grenade this step must have its `WeaponState` before it can spend
    //     one. Inert for every character with nothing throwable equipped.
    step_throws(world, &mut report);
    // 3. Every pending kick: the notify, or the fuse.
    step_kicks(world, dt, &mut report);
    // 3b. **The equipped weapon is an entity** (SK1b) — spawned, moved by the
    //     attachment pass below the pose, despawned when nothing is equipped.
    //     After the weapon step, because that is where a scroll wheel changes
    //     what is equipped.
    step_equipped_weapons(world);
    // 3b-ii. **The brass, as things you can see** (wave WPN2c). Beside the
    //     equipped weapon and on its doctrine: a derived guid, a runtime
    //     entity, no schema. AFTER the weapon step so a case ejected this step
    //     is drawn on the step it was thrown.
    step_casing_entities(world);
    // 3c. **The hands** (SK1c) — one request per character, composed from what
    //     it is holding and what it just pressed E on. After the weapon entity
    //     exists (so a hold and a spawn cannot disagree about the same step) and
    //     before the pose step reads it, which is the ordering the
    //     gameplay < pose < attachments pin already covers.
    report.hands = step_hand_ik(world, dt);
    // 3d. **The street hears it** (wave WPN1) — the crowd's panic, from this
    //     step's own gunfire. After the weapons, because it reads their hits;
    //     before the deaths, so a body that is about to be handed to the ragdoll
    //     is not also given a route to walk. `O(agents)` with a bounded inner
    //     loop and inert on a level with no gunfire — see `step_panic` for the
    //     cost and for the one-step latency the crowd's phase ordering implies.
    //
    //     **MERGED, not assigned** — and the arm that found this is
    //     `a_struck_bystander_either_runs_or_stands_its_ground`. A struck
    //     bystander's own flee is counted in `apply_hit`, several passes
    //     earlier; `report.panic = step_panic(…)` erased it, so the latch said
    //     an agent was running and the counter said nobody had. A counter that
    //     disagrees with the world is worse than no counter.
    let panic = step_panic(world, &report.hits, dt);
    report.panic.sources = panic.sources;
    report.panic.considered = panic.considered;
    report.panic.fled += panic.fled;
    report.panic.exempt += panic.exempt;
    // 3b. **The officers who did NOT rout take cover instead** (wave COV1,
    //     clause 5). Straight after the panic, on the same sources it just
    //     coalesced, because they are the same fact seen from two sides: the
    //     agents `flee_from` refuses to scatter are exactly the responders under
    //     fire. Sharing the coalesced list is also what makes the cost bound
    //     one bound rather than two — `panic_sources` is already capped at
    //     `MAX_PANIC_SOURCES`.
    //
    //     Inert on every step nothing was fired on: the source list is empty,
    //     the pass does not enter its loop, and `probes` is zero.
    report.npc_cover = super::cover::step_npc_cover(
        world,
        bridge,
        &panic_sources(&report.hits),
        PANIC_RADIUS_M,
        inf_ecs::traffic::steps(world),
        dt,
    );
    // 4. Every body that stopped working goes to the ragdoll — the P29.4
    //    bridge's own door, whose doc has named "a damage system" as its
    //    intended caller since it was written.
    let killed = step_deaths(world, bridge, &mut report);
    // 5. **Who saw it** (wave WPN1) — the witnessed-act seed. LAST, because a
    //    death outranks a shot in the record and the deaths are only known
    //    once the pass above has run. Inert on every step nothing happened on.
    report.witnessed = step_witness(
        world,
        bridge,
        &report.hits,
        &killed,
        inf_ecs::traffic::steps(world),
    );
    report
}

/// What [`step_crowd_doors`] did this step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CrowdDoorReport {
    /// Blocked agents this pass looked at.
    pub considered: usize,
    /// Presses made -- one per agent that found a shut door in reach.
    pub pressed: usize,
    /// Presses that moved a leaf. `pressed - opened` is the locked ones, which
    /// is the number a designer wants when a district stops working.
    pub opened: usize,
}

/// **A crowd agent opens the door it is standing against** -- clause 3's door
/// verb, through the same [`super::door::use_door`] the interact button and the
/// `door.use` node dispatch to.
///
/// # Why "blocked" is the trigger, and what it costs in seconds
///
/// The alternative -- every agent asking every step whether a door is in front
/// of it -- is `O(agents x doorways)` over a city that plans 19 790 doorways,
/// and 288 near agents would pay it every fixed step for the handful of doors
/// anybody is actually at. So the trigger is the crowd's own
/// [`CrowdAgent::blocked`](inf_ecs::crowd::CrowdAgent::blocked) verdict: an
/// agent whose body has fallen [`BLOCKED_LAG_M`] behind its own route clock.
///
/// [`BLOCKED_LAG_M`]: inf_ecs::crowd::BLOCKED_LAG_M
///
/// That is 2 m of lag, which at a 1.65 m/s walk is about **1.2 seconds** of
/// standing at the door before the handle turns. It is stated rather than
/// hidden, and it reads as a pause rather than as a bug -- a person does pause
/// at a door -- but it is a tuning constant and not a design: lowering the lag
/// shortens the pause and widens this pass's subject set in the same move.
///
/// A **locked** door is pressed and refuses, exactly as it refuses a player, and
/// the agent goes on trying the handle for as long as it stays blocked. That is
/// deliberate rather than unfinished: remembering which doors an agent has
/// already tried is per-agent state, and per-agent state on this path has to
/// ride the crowd's own trace section or two hosts diverge. The cost is one
/// `door::toggle` refusal per blocked agent per step, which is a state lookup;
/// the benefit is that the counters say `pressed` without `opened`, which is
/// the number a designer wants when a district stops working.
///
/// `placements` is the phase's **one** gather — [`step_gameplay`] takes it once
/// and hands the same list to this pass and to `step_doors` (the NPC1c audit;
/// the wave gathered twice and carried the fix by name). It is the same door
/// `candidates` and the player's prompt read, so an NPC cannot reach a door the
/// player is told is out of reach.
pub fn step_crowd_doors(
    world: &mut EcsWorld,
    placements: &[inf_ecs::door::DoorPlacement],
) -> CrowdDoorReport {
    let mut report = CrowdDoorReport::default();
    let blocked = inf_ecs::crowd::blocked_agents(world);
    if blocked.is_empty() {
        return report;
    }
    report.considered = blocked.len();
    if placements.is_empty() {
        return report;
    }
    for guid in blocked {
        let Some(feet) = feet_of(world, guid) else {
            continue;
        };
        let field = door::door_field(world);
        // The nearest SHUT door within reach. Ties break on the door's guid,
        // through `placements_near`'s own ascending order, so two agents at one
        // threshold press the same leaf.
        let mut best: Option<(f64, Uuid)> = None;
        for p in placements {
            let state = field
                .map(|f| f.get(p.guid, &p.spec))
                .unwrap_or_else(|| door::DoorState::fresh(&p.spec));
            // Shut **and at rest**. Without the second half an agent presses
            // again on the next step while the leaf is still swinging, and
            // `use_door` toggles -- measured, thirteen presses to open one door
            // and it shut itself twice on the way.
            if state.is_open(&p.spec) || !state.is_at_rest() {
                continue;
            }
            let d = (door::prompt_position(p) - feet).length();
            if d > door::DOOR_REACH_M {
                continue;
            }
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, p.guid));
            }
        }
        let Some((_, door_guid)) = best else {
            continue;
        };
        report.pressed += 1;
        if super::door::use_door(world, door_guid, feet).moved() {
            report.opened += 1;
        }
    }
    report
}

/// **Where a character's feet are**, world metres.
///
/// The same arithmetic `step_one` and the prompt already use, and it is here
/// once rather than three times because a kick measured from a different point
/// than the prompt would let a player kick a door the prompt says is out of
/// reach.
pub fn feet_of(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    let entity = world.entity_of(guid)?;
    let w = world.world();
    let cm = w.get::<CharacterMovement>(entity)?;
    let t = w.get::<Transform>(entity)?;
    let radius = w
        .get::<inf_ecs::components::Collider3D>(entity)
        .map(|c| c.radius)
        .unwrap_or(0.3);
    Some(t.translation.to_dvec3() - DVec3::Y * (cm.half_height_for(cm.mode) + radius))
}

/// **The character's muzzle**, world metres, and where it is looking.
///
/// # Two answers, and which one applies
///
/// 1. **The weapon's own muzzle.** A character with a rig carries its equipped
///    weapon as an entity attached to [`WEAPON_SOCKET`]; the shot leaves that
///    entity's barrel, [`WeaponDef::muzzle_forward_m`] along its local `+Z`. The
///    weapon's placement is required to be *finite and settled* — it is only
///    settled once `update_attachments` has run, which is why the pose is asked
///    for as well: a character that publishes no pose has an attachment sitting
///    at its entity origin, which is its capsule centre, and a muzzle there would
///    be a silent half-metre regression on every unrigged level in the tree.
/// 2. **A height above the feet**, [`MUZZLE_HEIGHT_M`] — everything else.
///
/// # The one-step latency, stated
///
/// `step_gameplay` runs **before** `step_pose_evaluation` and
/// `update_attachments` in both hosts, so the weapon transform this reads is the
/// one the previous fixed step settled. At 60 Hz that is 16.7 ms of lag between
/// where the hand is and where the shot starts, and it is the same lag in both
/// hosts — so PIE == shipping is unaffected and the trace cannot see it. Moving
/// the gameplay step below the pose would fix it and would move every committed
/// trace in the tree; it is named here rather than done quietly.
/// The fourth element is **which of the two answers this is**: `true` for the
/// weapon's own muzzle, `false` for the capsule rule. The caller counts the
/// second one when it happens to a *posed* character — see
/// [`GameplayReport::muzzles_without_a_socket`].
fn muzzle_of(world: &EcsWorld, guid: Uuid) -> Option<(DVec3, f64, f64, bool)> {
    let entity = world.entity_of(guid)?;
    let cm = world.world().get::<CharacterMovement>(entity)?;
    let (yaw, pitch) = (cm.runtime.aim_yaw_deg, cm.runtime.aim_pitch_deg);
    if let Some(from) = weapon_muzzle(world, guid) {
        return Some((from, yaw, pitch, true));
    }
    let feet = feet_of(world, guid)?;
    Some((feet + DVec3::Y * MUZZLE_HEIGHT_M, yaw, pitch, false))
}

/// The muzzle of `guid`'s **weapon entity**, if there is one and it is attached
/// to a real socket. `None` sends `muzzle_of` to the capsule rule.
fn weapon_muzzle(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    use inf_ecs::components::GlobalTransform;
    // The socket has to exist on the rig AND have been resolved, or the
    // attachment is sitting at the character's own origin.
    inf_ecs::pose::evaluated_pose(world, guid)?.socket(WEAPON_SOCKET)?;
    let (_, def) = equipped_weapon(world, guid)?;
    let e = world.entity_of(equipped_weapon_guid(guid))?;
    let g = world.world().get::<GlobalTransform>(e)?.0;
    let forward = def
        .muzzle_forward_m
        .clamp(0.0, weapon::MAX_MUZZLE_FORWARD_M);
    let at = g.transform_point3(DVec3::new(0.0, 0.0, forward));
    at.is_finite().then_some(at)
}

/// **Keep every character's equipped weapon a real entity in the world**
/// (SK1b) — spawned when something is equipped, moved by the attachment pass,
/// despawned when nothing is.
///
/// # Why a whole entity
///
/// Before this an "equipped weapon" was a slot index. Nothing drew it, nothing
/// could be attached to it, and the shot it fired started at a hard-coded height
/// above the character's feet — the scout's risk 14, whole. An entity is what a
/// socket can carry, what a projector can draw and what a muzzle can be read off,
/// and it costs no schema: `AttachedTo` is already a scene component and this one
/// is never saved.
///
/// # Deterministic by construction
///
/// The guid is [`equipped_weapon_guid`], derived from the owner, so both hosts
/// spawn the same entity on the same step. The entity **appears in the trace** as
/// a transform row, which is the honest cost and is why it is spawned from the
/// one Ring-0 rule both hosts call rather than from either of them.
fn step_equipped_weapons(world: &mut EcsWorld) {
    use inf_ecs::components::{AttachedTo, MeshRef, Name, Primitive, Transform, Visibility};

    for guid in gunners(world) {
        // **WHAT IT DRAWS AS** (wave WPN2d). Three things now rather than one:
        // the id, the barrel length the placeholder is scaled to, and the mesh
        // asset — `inf_ecs::weapon::weapon_mesh_of`, which is the `ItemDef`'s own
        // override if it names one and the class -> art table's derived identity
        // otherwise. `None` is a shotgun, a launcher, or a project whose art is
        // not on this machine, and it draws the primitive it always drew.
        // **A THROWN GRENADE IS NOT IN THE HAND** (wave WPN2d) — the entity
        // leaves the world for the frames the body is in the air, which is what
        // `WeaponState::in_hand` says and what stops a grenade being drawn in a
        // palm and flying through the air at the same time.
        let empty_hand = world.entity_of(guid).is_some_and(|e| {
            let w = world.world();
            match (
                w.get::<weapon::WeaponState>(e),
                equipped_weapon(world, guid),
            ) {
                (Some(st), Some((id, def))) => st.item_id == id && !st.in_hand(&def),
                _ => false,
            }
        });
        let want = (!empty_hand)
            .then(|| equipped_weapon_item(world, guid))
            .flatten()
            .map(|(id, item)| {
                let forward = item
                    .weapon
                    .as_ref()
                    .map(|w| w.muzzle_forward_m)
                    .unwrap_or_default();
                (id, forward, inf_ecs::weapon::weapon_mesh_of(&item))
            });
        let weapon_guid = equipped_weapon_guid(guid);
        let existing = world.entity_of(weapon_guid);
        match (want, existing) {
            (None, Some(e)) => {
                // Nothing equipped: the weapon leaves the world, so a holstered
                // character is byte-identical to one that never had a weapon.
                // Its accessories go with it — a scope with no rifle under it
                // would hang in the air on the derived guid for ever.
                despawn_accessories(world, weapon_guid);
                world.despawn(e);
            }
            (None, None) => {}
            (Some((id, forward, mesh)), existing) => {
                let e = match existing {
                    Some(e) => e,
                    None => world.spawn_with_guid(weapon_guid, &format!("Weapon: {id}"), None),
                };
                let w = world.world_mut();
                // The name follows the equipped item, so switching weapons is
                // visible in the Outliner rather than silently the same row.
                if let Some(mut n) = w.get_mut::<Name>(e) {
                    let label = format!("Weapon: {id}");
                    if n.0 != label {
                        n.0 = label;
                    }
                }
                // A placeholder primitive, scaled to the weapon's own length —
                // the `item::spawn_pickup` precedent, and the same honest bound:
                // an `ItemDef` that names a mesh asset is the next field and is
                // not in this wave.
                //
                // **The size survives the attachment pass** (SK1b audit), and it
                // did not when this was written: `update_attachments` composed
                // the *target's* scale onto its follower, which erased this line
                // one pass later and drew a **1 m cube** in the character's hand.
                // That pass leaves a follower's own size alone now. The two arms
                // are `an_attachment_places_a_follower_without_resizing_it` and
                // `the_equipped_weapon_is_an_entity_attached_to_the_hand_socket`.
                let mut t = w.get_mut::<Transform>(e).map(|t| *t).unwrap_or_default();
                let len = forward.clamp(0.05, weapon::MAX_MUZZLE_FORWARD_M);
                // **A REAL MESH IS DRAWN AT ITS OWN SIZE** (wave WPN2d), and the
                // placeholder is still scaled to the barrel.
                //
                // The inversion is deliberate and it is the honest one. A
                // placeholder box has no size of its own, so it is stretched to
                // `muzzle_forward_m` — that number IS the barrel, and the box is
                // a stand-in for it. A real weapon mesh is modelled in metres and
                // already IS the length its class says it is, so stretching it
                // would distort a rifle to fit a number that was authored to
                // describe that rifle. The fixed step cannot measure a mesh (it
                // has no asset database — that is the editor's and the pack's),
                // so the choice is between scaling art by a number nobody has
                // checked against it and drawing art at 1:1 and letting the
                // registry's own barrel lengths be what a shot's muzzle offset
                // is measured along. This takes the second, and `wpn2d_gate`
                // measures the consequence: the hand holds the mesh at its
                // origin and the muzzle is `muzzle_forward_m` along its `+Z`.
                t.scale = if mesh.is_some() {
                    inf_ecs::math::Vec3d::ONE
                } else {
                    inf_ecs::math::Vec3d::new(0.06, 0.06, len)
                };
                w.entity_mut(e).insert((
                    t,
                    MeshRef {
                        primitive: Primitive::Cube,
                        asset: mesh,
                    },
                    Visibility::default(),
                    // Zero offset: the weapon sits AT the hand socket, which is
                    // where a grip's palm frame is. A `GripAffordance`'s palm
                    // transform is the refinement, and it needs the rig — which
                    // this step does not have and the pose step does.
                    AttachedTo::new(guid, WEAPON_SOCKET, inf_ecs::math::Vec3d::ZERO),
                    // **The marker** (wave WPN2d): what makes this entity
                    // askable-about without re-deriving its guid — the
                    // first-person fade rule's door
                    // (`inf_ecs::weapon::subject_fade_for`) and the gate's.
                    inf_ecs::weapon::EquippedWeapon { owner: guid },
                ));
                step_accessories(world, guid, weapon_guid, len);
            }
        }
    }
}

/// **What is bolted onto the weapon, as things you can SEE** (wave WPN2d).
///
/// `step_casing_entities`' doctrine exactly: a derived guid per slot, a runtime
/// entity reconciled against the state every step, no schema. An accessory is
/// `AttachedTo` the WEAPON with an **empty socket name**, which
/// `inf_ecs::attach::update_attachments` reads as *ride the entity, not a rig* —
/// so a scope follows the rifle the rifle's own attachment already put in the
/// hand, one pass later and with no second rule.
///
/// The three offsets are in the weapon's own local frame (`+Z` down the barrel),
/// and each is the place that part of a gun physically is:
///
/// * an **optic** sits on top of the receiver, a little forward of the grip;
/// * a **muzzle device** sits at the muzzle, which is `muzzle_forward_m` along
///   the barrel — the same number the shot leaves from, so a suppressor is
///   drawn exactly where the bullet comes out;
/// * an **underbarrel** grip sits under the handguard at the FORE-GRIP point,
///   which is [`fore_grip_m`]'s own two-thirds-of-barrel rule — the same number
///   the off hand reaches for, so a hand and a grip cannot end up in two places.
///
/// A checkout without the accessory art draws nothing at all rather than a box:
/// a floating cube on a rifle is worse than an unadorned rifle, and the
/// attachment's EFFECT (the fold) is what the wave is really about.
fn step_accessories(world: &mut EcsWorld, owner: Uuid, weapon_guid: Uuid, barrel_m: f64) {
    use inf_ecs::attachment::{AttachmentArt, AttachmentSlot};
    use inf_ecs::components::{AttachedTo, MeshRef, Name, Primitive, Transform, Visibility};
    let Some(entity) = world.entity_of(owner) else {
        return;
    };
    let equipped = world
        .world()
        .get::<weapon::WeaponState>(entity)
        .map(|s| s.attach);
    let art: Vec<(AttachmentSlot, AttachmentArt)> = match equipped {
        Some(a) => inf_ecs::attachment::equipped_art(&a),
        None => Vec::new(),
    };
    let mut want: BTreeSet<Uuid> = BTreeSet::new();
    for (slot, kind) in &art {
        let Some(asset) = inf_ecs::weapon::attachment_mesh_guid(*kind) else {
            continue;
        };
        let g = accessory_guid(weapon_guid, *slot);
        want.insert(g);
        let offset = accessory_offset(*slot, barrel_m);
        let e = match world.entity_of(g) {
            Some(e) => e,
            None => world.spawn_with_guid(g, &format!("Attachment: {}", slot.name()), None),
        };
        let mut t = world
            .world()
            .get::<Transform>(e)
            .copied()
            .unwrap_or_default();
        t.scale = inf_ecs::math::Vec3d::ONE;
        world.world_mut().entity_mut(e).insert((
            t,
            MeshRef {
                primitive: Primitive::Cube,
                asset: Some(asset),
            },
            Visibility::default(),
            AttachedTo::new(weapon_guid, "", offset),
            inf_ecs::weapon::AccessoryMark {
                weapon: weapon_guid,
            },
        ));
        if let Some(mut n) = world.world_mut().get_mut::<Name>(e) {
            let label = format!("Attachment: {}", slot.name());
            if n.0 != label {
                n.0 = label;
            }
        }
    }
    // Anything on this weapon's rail that is no longer bolted on is gone.
    for g in drawn_accessories(world, weapon_guid) {
        if !want.contains(&g) {
            if let Some(e) = world.entity_of(g) {
                world.despawn(e);
            }
        }
    }
}

/// Every accessory currently drawn on this weapon, in `Guid` order.
fn drawn_accessories(world: &EcsWorld, weapon_guid: Uuid) -> Vec<Uuid> {
    use inf_ecs::components::Guid;
    let w = world.world();
    let Some(mut q) = w.try_query::<(&Guid, &inf_ecs::weapon::AccessoryMark)>() else {
        return Vec::new();
    };
    let mut out: Vec<Uuid> = q
        .iter(w)
        .filter(|(_, m)| m.weapon == weapon_guid)
        .map(|(g, _)| g.0)
        .collect();
    out.sort();
    out
}

/// Take every accessory off a weapon that is leaving the world.
fn despawn_accessories(world: &mut EcsWorld, weapon_guid: Uuid) {
    for g in drawn_accessories(world, weapon_guid) {
        if let Some(e) = world.entity_of(g) {
            world.despawn(e);
        }
    }
}

/// **The identity of one accessory on one weapon** — content-derived, so a fixed
/// step mints nothing random (`equipped_weapon_guid`'s own law).
fn accessory_guid(weapon_guid: Uuid, slot: inf_ecs::attachment::AttachmentSlot) -> Uuid {
    let mut n = weapon_guid.as_u128();
    n ^= ACCESSORY_SALT;
    n = n.rotate_left(u32::from(slot.index() as u8) * 7 + 1);
    Uuid::from_u128(n)
}

/// The salt accessory identities are derived against — `equipped_weapon_guid`'s
/// own construction, one level down.
const ACCESSORY_SALT: u128 = 0x4143_4345_5353_4f52_595f_5750_4e32_4421;

/// **Where an accessory sits on the weapon**, in the weapon's own local frame.
/// See [`step_accessories`] for why each is where it is.
fn accessory_offset(
    slot: inf_ecs::attachment::AttachmentSlot,
    barrel_m: f64,
) -> inf_ecs::math::Vec3d {
    use inf_ecs::attachment::AttachmentSlot;
    match slot {
        // On top of the receiver, a third of the way down it.
        AttachmentSlot::Optic => inf_ecs::math::Vec3d::new(0.0, 0.055, barrel_m * 0.33),
        // At the muzzle — the same point the shot leaves from.
        AttachmentSlot::Muzzle => inf_ecs::math::Vec3d::new(0.0, 0.0, barrel_m),
        // Under the handguard, at the fore-grip the off hand reaches for.
        AttachmentSlot::Underbarrel => {
            inf_ecs::math::Vec3d::new(0.0, -0.045, f64::from(fore_grip_len(barrel_m)))
        }
        // Everything else changes numbers and not silhouettes; nothing is drawn
        // for it and `equipped_art` never answers one.
        _ => inf_ecs::math::Vec3d::ZERO,
    }
}

/// **The equipped item, whole** (wave WPN2d) — what `equipped_weapon` answers
/// plus the `ItemDef` around it, because the mesh is a property of the ITEM and
/// the numbers are a property of the weapon inside it.
fn equipped_weapon_item(world: &EcsWorld, guid: Uuid) -> Option<(String, inf_ecs::item::ItemDef)> {
    let entity = world.entity_of(guid)?;
    let inv = world.world().get::<inf_ecs::item::Inventory>(entity)?;
    let id = inv.equipped_id()?.to_string();
    let item = inf_ecs::item::item_defs(world)?.get(&id)?.clone();
    item.weapon.as_ref()?;
    Some((id, item))
}

/// **Where a two-handed weapon's fore-grip is**, metres along the barrel from
/// the holding hand.
///
/// A fraction of the weapon's own length rather than a constant: a pistol has no
/// fore-grip to reach and a rifle's is most of the way down it. Two thirds is
/// where a hand sits on a rifle's handguard, and clamping the result keeps a
/// 5 cm weapon from asking the off hand to occupy the same space as the on hand.
fn fore_grip_m(def: &WeaponDef) -> f32 {
    fore_grip_len(
        def.muzzle_forward_m
            .clamp(0.05, weapon::MAX_MUZZLE_FORWARD_M),
    )
}

/// **The two-thirds rule itself**, over a barrel length (wave WPN2d).
///
/// Split out of [`fore_grip_m`] with nothing changed, because two things read it
/// now: the off HAND, which is what it was written for, and the underbarrel
/// GRIP an attachment draws. A hand and the grip it is holding must be in the
/// same place, and one function is how.
fn fore_grip_len(barrel_m: f64) -> f32 {
    (barrel_m * 0.66).clamp(0.12, 0.60) as f32
}

/// How far in front of the character's chest an AIMED weapon is brought,
/// metres.
///
/// **A CEILING since the WPN2b audit, not a fixed distance.** A rigged character
/// holds its weapon at the reach its own animation is holding it at — see
/// `aim_hold_point` — and this is the furthest that reach may be stretched to.
/// It stays the fixed distance for a rig-less character, which is what it was
/// written for at wave WPN1.
pub const AIM_REACH_M: f64 = 0.42;

/// **The smallest the aimed reach may be pulled to**, metres (wave WPN2b).
///
/// Five centimetres. The recoil spring pulls the hold point back along the aim
/// by up to twenty centimetres at a launcher's `recoil_intensity`, and a reach
/// that went to zero would put the hand inside the chest — so the pull is
/// clamped here rather than in the profile, because the floor is a property of
/// the BODY and the pull is a property of the weapon.
///
/// It replaces wave WPN1's `RECOIL_PULL_M` / `RECOIL_RISE_DEG` pair, whose whole
/// job was to scale a derived `recoil_fraction` that no longer exists: the
/// spring's own `vm_kick_back` and `vm_kick_up` are the numbers now, they are
/// per-weapon rather than global, and they are metres of measured displacement
/// rather than a fraction of a fire interval.
pub const AIM_REACH_MIN_M: f64 = 0.05;

/// The fraction of a character's height its shoulder line sits at.
///
/// The same proportion `inf_anim::BodyParams` uses for a default biped, spelled
/// here because this step has a capsule and not a rig: it reads the movement
/// component's own stand height, which is what a character IS to the mover.
const SHOULDER_OF_HEIGHT: f64 = 0.82;

/// **THE HAND PASS** (SK1c) — one `HandIk` per character, from everything this
/// step knows.
///
/// # Why one producer and not two
///
/// A weapon wants both hands and a grab wants one, and both write into the same
/// two-slot array. Two producers would race for the same slot every step and the
/// winner would be whichever ran last — so they are composed here, once, and the
/// rule between them is written down rather than emergent: **the weapon owns the
/// hand it is in, and a grab takes the other one.** A character reaching for a
/// door handle with a rifle in its right hand reaches with its left, which is
/// what a person does.
///
/// **The off hand is on loan** (SK1c audit, H1). The weapon owns the hand it is
/// *in*; the other one is supporting it, and a grab takes it back — so while a
/// grab is live the `GunGrip` hold is weighted by its complement and the
/// `rifle_fore` grip is not asked for at all. Without that the gun solve, which
/// runs *after* the reaches inside `apply_hand_ik`, overwrote the grab's reach
/// every step: an armed character's E-press moved neither wrist and the only
/// thing it did was spring the support hand open.
///
/// # What each half asks for
///
/// * **A weapon** puts a [`GunGrip`] hold on the rig — the `ik_hand_gun` path,
///   whose whole purpose is that the off hand is carried *by the weapon* rather
///   than aimed at a point in space — and closes both hands on their own
///   affordances (`rifle` in the holding hand, `rifle_fore` in the other unless a
///   grab has taken it; the trigger finger is left straight by the catalogue, not
///   by this code).
/// * **Aiming** adds a reach for the holding hand, and only then. A weapon at
///   rest hangs where the animation puts it; RMB brings it up to a point on the
///   aim line at shoulder height, which is the difference between carrying a
///   rifle and pointing one.
/// * **A grab** reaches the free hand to the interaction's own point and closes
///   it on the affordance the interactable named.
///
/// Everything is absent unless asked for, so a level nobody has armed and nobody
/// has pressed E in publishes no resource and poses exactly the bytes it did
/// before this wave.
fn step_hand_ik(world: &mut EcsWorld, dt: f64) -> (u32, u32) {
    use inf_ecs::pose::{GunGrip, HandGrip, HandIk, HandReach};

    // Age the grabs first: a grab that ended this step must not also be asked
    // for this step, or a released hand would be one step late in one host and
    // not the other if the two ever aged it in different places.
    inf_ecs::interact::step_grabs(world, dt);

    let mut holds = 0u32;
    let mut grabs = 0u32;
    for guid in gunners(world) {
        let mut req = HandIk::default();

        // **The grab is read FIRST**, because it decides whether the off hand is
        // still on the weapon (SK1c audit, H1). A live grab is one with a
        // non-zero amount; a finished one is removed by `step_grabs` above.
        let grab = inf_ecs::interact::hand_grab(world, guid)
            .map(|g| (g.amount(), g.at, g.grip.clone()))
            .filter(|(amount, _, _)| *amount > 0.0);

        // -- the weapon --
        //
        // **A THROWN GRENADE IS NOT IN THE HAND** (wave WPN2d): a character who
        // has just thrown their last one is holding nothing until they take
        // another out, and posing a hand around a body that is in the air would
        // be the same grenade in two places. `WeaponState::in_hand` is the one
        // door `step_equipped_weapons` asks the same question through, so the
        // drawn entity and the pose cannot disagree.
        let armed = equipped_weapon(world, guid).filter(|(id, def)| {
            world
                .entity_of(guid)
                .and_then(|e| world.world().get::<WeaponState>(e))
                .map(|st| st.item_id != *id || st.in_hand(def))
                .unwrap_or(true)
        });
        if let Some((_, def)) = armed.as_ref() {
            req.grip[1] = Some(HandGrip {
                name: inf_anim::GRIP_RIFLE.to_string(),
                amount: 1.0,
            });
            holds += 1;
            // **The off hand is on LOAN from the weapon, and a grab takes it
            // back.** The weapon owns the hand it is *in* — the right one, the
            // one `WEAPON_SOCKET` hangs off — and the other is merely supporting
            // it, which is the hand a person takes off a rifle to open a door.
            //
            // The gun hold's weight is the complement of the grab's, so the arm
            // crosses over continuously instead of snapping between the
            // fore-grip and the handle on the step the grab starts and the step
            // it ends. At `amount == 1` the hold is weightless and
            // `apply_hand_ik` skips the off-hand solve entirely.
            //
            // Before this, the `GunGrip` solve ran unconditionally and — because
            // it runs AFTER the reaches inside `apply_hand_ik` — overwrote the
            // grab's reach every step: an armed character's E-press moved
            // neither wrist by a single millimetre while `hands.1` counted it.
            let amount = grab.as_ref().map(|(a, _, _)| *a).unwrap_or(0.0);
            req.gun = Some(GunGrip {
                holding: inf_anim::BoneSide::Right,
                off_hand_offset: [0.0, 0.0, fore_grip_m(def)],
                weight: 1.0 - amount,
            });
            if grab.is_none() {
                req.grip[0] = Some(HandGrip {
                    name: inf_anim::GRIP_RIFLE_FORE.to_string(),
                    amount: 1.0,
                });
            }
            // -- and the aim, which is what MOVES it --
            if let Some(target) = aim_hold_point(world, guid) {
                req.reach[1] = Some(HandReach {
                    target: inf_ecs::math::Vec3d::new(target.x, target.y, target.z),
                    weight: 1.0,
                });
            }
        }

        // -- the grab, in whichever hand is free --
        if let Some((amount, at, grip)) = grab {
            // The weapon is in the right hand, so a grab goes to the left; an
            // unarmed character reaches with its right, which is the hand every
            // affordance in a default catalogue but `rifle_fore` is on. The
            // *slot* is what decides which hand closes — `apply_hand_ik` reads
            // it, and the affordance supplies the aperture and the curl set.
            //
            // **Honest bound**: the off hand's fingers do not cross-fade. They
            // let go of the fore-grip on the step the grab begins and close on
            // the new affordance over its ease, because one slot carries one
            // grip. A hand releasing a weapon before it takes hold of something
            // else is the right picture; doing it in one fixed step is the
            // approximation.
            let side = usize::from(armed.is_none());
            req.reach[side] = Some(HandReach {
                target: at,
                weight: amount,
            });
            req.grip[side] = Some(HandGrip { name: grip, amount });
            grabs += 1;
        }

        inf_ecs::pose::set_hand_ik(world, guid, req);
    }
    (holds, grabs)
}

/// **Where an aiming character holds its weapon**, world metres — a point on the
/// aim line at shoulder height, in front of the chest.
///
/// `None` when the character is not aiming, which is what leaves the weapon
/// wherever the animation is carrying it.
///
/// The direction goes through [`weapon::aim_forward`], which is the door the
/// shot's own direction takes — a second copy of that arithmetic would be a hand
/// that points somewhere the bullet does not. Portable for the shot's own
/// reason: this number is folded into `pose_state_bytes` and compared between
/// two machines (the P14 law).
///
/// # THE RECOIL IS HERE, and since wave WPN2b it is a SPRING
///
/// A shot deposits an impulse on [`inf_ecs::feel::WeaponFeel::vm`] — a
/// critically damped spring in the aim frame, `x` right, `y` up, `z` forward —
/// and this reads its position. The spring is advanced once a step by the
/// movement runtime's look integrator, one phase earlier, so what lands here is
/// this step's own displacement; it reaches `HandIk::reach`, then
/// `apply_hand_ik`, then `pose_state_bytes`, so the two hosts are compared on it
/// byte for byte and a replay reproduces it.
///
/// What it replaces is wave WPN1's `recoil_fraction`: the weapon's own fire
/// cycle, scaled by two engine-wide constants, which meant every weapon in the
/// game kicked the same distance and a 450 rpm pistol kicked for LONGER than a
/// 900 rpm rifle because its interval was longer. The spring's numbers come off
/// the registry's `recoil_intensity`, per row.
///
/// The **sway** ([`inf_ecs::feel::sway_offset`]) is added in the same frame and
/// the same units: the doc's *"blend this with procedural recoil so the gun
/// model feels organic"*, which is one addition because both are offsets of the
/// same point.
///
/// # This still does NOT move the aim, and now something else does
///
/// `cm.runtime.aim_*` is what the bullet leaves along and what the camera
/// chases; this reads those two numbers and writes neither. The aim's OWN
/// recoil — wave WPN1's named honest form, and this wave's clause 2 — is a
/// second spring applied inside `inf_physics::d3::movement`'s look integrator,
/// where it is sim state that the camera then follows. So the reticle stays
/// true: it is drawn at the centre of the screen, the camera's pitch tracks
/// `aim_pitch_deg`, and the round leaves along `aim_pitch_deg`.
///
/// # Why there is still no CAMERA recoil, stated as a ruling
///
/// The obvious companion — kick the camera's pitch and let it settle — was
/// priced and refused at wave WPN1, and the refusal has two halves that meet:
///
/// * a camera kick that does **not** move the aim makes the reticle **lie**. The
///   shot leaves along `aim_pitch_deg`, the reticle is drawn at the centre of the
///   screen, and the camera's pitch is the entire mapping between the two; kick
///   one and not the other and the crosshair stops being where the rounds go,
///   permanently under sustained fire.
/// * a camera kick that **does** move the aim is a camera → sim write, which is
///   the one thing `d3::camera`'s Ruling 4 exists to forbid ("`ViewMode` never
///   crosses the sim wire, and there is no camera → sim path at all"), and which
///   `phase29_gate` pins by running the same course under two different cameras
///   and comparing the sim trace.
///
/// Wave WPN2b honours it by moving the AIM. `camera.rs` and `camera.toml` gain
/// no per-shot input, and `wpn2b_gate` greps both for one.
fn aim_hold_point(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    let entity = world.entity_of(guid)?;
    let cm = world.world().get::<CharacterMovement>(entity)?;
    if cm.rotation_mode != inf_ecs::components::RotationMode::Aiming {
        return None;
    }
    // The recoil spring and the sway, both in the AIM FRAME and both in metres.
    // Exactly zero — and every byte below identical to what it was before this
    // wave — for a character with no `WeaponFeel`, which is every character that
    // has never held a weapon.
    let offset = inf_ecs::feel::feel_of(world, guid)
        .map(|f| f.vm.position + f.sway)
        .unwrap_or(DVec3::ZERO);
    let yaw = cm.runtime.aim_yaw_deg;
    let dir = weapon::aim_forward(yaw, cm.runtime.aim_pitch_deg);
    // The frame: `right` is horizontal by construction and `up` completes it —
    // `weapon::shot_direction_with`'s own two lines, because a hold point that
    // used a different basis from the bullet would drift away from the aim as
    // the character pitched.
    let (sy, cy) = {
        let r = yaw.to_radians();
        (inf_math::psin64(r), inf_math::pcos64(r))
    };
    let right = DVec3::new(cy, 0.0, -sy);
    let up = right.cross(dir);
    // **THE ANCHOR AND THE REACH ARE THE RIG'S OWN** (WPN2b audit, carried 218
    // and 227), and the capsule rule below is the fallback.
    let (anchor, span) = match arm_anchor(world, guid) {
        Some(pair) => pair,
        None => {
            let feet = feet_of(world, guid)?;
            // The stand height is what the capsule was built from, so this
            // tracks a 1.2 m character and a 2.4 m one without a second opinion
            // about either.
            let height = (cm.stand_half_height_m * 2.0).max(0.4);
            (feet + DVec3::Y * (height * SHOULDER_OF_HEIGHT), AIM_REACH_M)
        }
    };
    let reach = (span + offset.z).max(AIM_REACH_MIN_M);
    let at = anchor + dir * reach + right * offset.x + up * offset.y;
    at.is_finite().then_some(at)
}

/// **Where this character's weapon arm hangs from, and how far it is already
/// reaching** — `(shoulder in world metres, reach in metres)`, or `None` for a
/// character with no rig.
///
/// # The two carried items this closes, and why they are one item
///
/// Wave WPN2b measured both halves of the same defect and carried them
/// separately. **227**: the hand had never arrived at the hold point — 126.74 mm
/// short at a level aim and 274.65 mm at a 35-degree downward one — because
/// [`AIM_REACH_M`] was measured forward from a shoulder LINE derived from the
/// movement capsule (`SHOULDER_OF_HEIGHT` of the stand height) and a rig's own
/// shoulder joint is neither at that height nor on that line, so the requested
/// point sat outside the arm's reach envelope and `solve_arm` stopped where it
/// could. **218**: an ALS prop pose set was invisible on an armed character,
/// because a hand sent to a world point the arm cannot reach is an arm thrown
/// straight at it and nothing the animation said about that arm survives.
///
/// They are one item because the second is what the first LOOKS like. An arm
/// stretched at an unreachable point has one configuration and it is not a
/// pose; an arm sent somewhere it can go keeps its own fold.
///
/// # The two numbers, and why each comes from where it does
///
/// * **The anchor is the arm chain's own first joint** — the upper arm, whose
///   position is a function of the CLAVICLE and is therefore *not* written by
///   `solve_arm` (`inf_anim::arm_chain` is upper arm / forearm / hand). So the
///   overlay, which poses the clavicle, moves the anchor; the solver cannot.
///   That is what makes this non-recursive.
/// * **The reach is the base pose's own** — the distance from that joint to the
///   hand **before any solve ran**, published by `apply_hand_ik` as
///   [`inf_ecs::pose::HandIkReport::base_hand`]. It is a distance the animation
///   has just demonstrated the arm can achieve, so a point at that distance is
///   inside the envelope by construction — which is the whole of 227 — and it
///   is the pose's own extension, which is the whole of 218: a carry pose that
///   holds the weapon close keeps the elbow it authored, and one that holds it
///   out keeps that.
///
/// Read off the SOLVED pose either number would be the solver measuring its own
/// output: the reach would gain the recoil's pull-back every shot and never give
/// it back, and an arm would curl up over a magazine.
///
/// The reach is clamped to `[AIM_REACH_MIN_M, AIM_REACH_M]`, which is what
/// [`AIM_REACH_M`] is now: a **ceiling**. A character whose arms are hanging at
/// its sides has a base reach of nearly its whole arm, and bringing a weapon up
/// to an aim is not an excuse to lock the elbow.
fn arm_anchor(world: &EcsWorld, guid: Uuid) -> Option<(DVec3, f64)> {
    let report = inf_ecs::pose::hand_ik_report(world, guid)?;
    let shoulder = report.shoulder[1]?.to_dvec3();
    let hand = report.base_hand[1]?.to_dvec3();
    let span = (hand - shoulder).length();
    (shoulder.is_finite() && span.is_finite())
        .then(|| (shoulder, span.clamp(AIM_REACH_MIN_M, AIM_REACH_M)))
}

/// **The posture a shot is fired from** (wave WPN2b) — `(stance, planar speed,
/// ADS blend)`, the three things that are true of the SHOOTER rather than of the
/// weapon and that the cone is scaled by.
///
/// A character with no `CharacterMovement` is standing, still and hip-firing,
/// which is what every rig-less fixture in the tree is and what keeps their
/// patterns identical to their pre-wave selves.
fn shot_posture(world: &EcsWorld, entity: inf_ecs::Entity) -> (ShotStance, f64, f64) {
    let Some(cm) = world.world().get::<CharacterMovement>(entity) else {
        return (ShotStance::Standing, 0.0, 0.0);
    };
    let stance = match cm.mode {
        inf_ecs::components::MovementMode::Crouch => ShotStance::Crouched,
        inf_ecs::components::MovementMode::Prone => ShotStance::Prone,
        _ => ShotStance::Standing,
    };
    let v = cm.runtime.velocity.to_dvec3();
    let speed = DVec3::new(v.x, 0.0, v.z).length();
    let ads = world
        .world()
        .get::<WeaponFeel>(entity)
        .map(|f| f.ads_blend)
        .unwrap_or(0.0);
    (stance, speed, ads)
}

/// **Give a character's feel back** (wave WPN2b) — the removal half of the
/// install beside `WeaponState`.
///
/// It is its own function and not a `remove::<WeaponFeel>()` at the call site
/// because there is a second thing owed: the camera rig's `state_blend_speed`,
/// which the ADS blend borrowed and which has to go back exactly as it was
/// (`WeaponFeel::blend_speed_prior`). One door, so a weapon put away can never
/// leave a character's camera settling at a rifle's rate for the rest of the
/// session.
fn release_feel(world: &mut EcsWorld, guid: Uuid, entity: inf_ecs::Entity) {
    let Some(feel) = world.world().get::<WeaponFeel>(entity).copied() else {
        return;
    };
    // **THE AIM IS OWED WHATEVER THE SPRING HAD NOT GIVEN BACK YET**, and this
    // is the whole reason the removal is a function.
    //
    // The centre-recovery is the DELTA the look integrator adds each step
    // (`inf_physics::d3::movement::step_weapon_feel`), so an aim spring that is
    // still 0.02 deg from home when the weapon is put away has 0.02 deg of the
    // player's aim in it — and dropping the component would stranded that offset
    // on the character for the rest of the level. Measured on
    // `weapon_hands_gate`'s own course: without this, an unarmed hero after a
    // whole weapon course did not pose what it posed before it picked anything
    // up, and the difference was one shot's unpaid residual.
    if feel.applied_pitch_deg != 0.0 || feel.applied_yaw_deg != 0.0 {
        if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(entity) {
            cm.runtime.aim_pitch_deg =
                (cm.runtime.aim_pitch_deg - feel.applied_pitch_deg).clamp(-89.0, 89.0);
            cm.runtime.aim_yaw_deg =
                inf_ecs::movement::wrap_deg(cm.runtime.aim_yaw_deg - feel.applied_yaw_deg);
        }
    }
    let prior = Some(feel.blend_speed_prior);
    if let Some(prior) = prior {
        if prior.is_finite() {
            inf_ecs::camera::set_camera_rig_value(
                world,
                guid,
                inf_ecs::feel::ADS_BLEND_RIG_KEY,
                prior,
            );
        }
    }
    world.world_mut().entity_mut(entity).remove::<WeaponFeel>();
}

/// The equipped weapon's definition, if the character has one equipped and the
/// catalogue knows it.
///
/// **One door** (wave WPN2a): the lookup itself is
/// [`inf_ecs::weapon::equipped_def`], because the movement step now asks the
/// same question — how fast does this character move for what it is carrying —
/// and two spellings of "what is in this hand" is the shape of defect this
/// repository has paid for at five separate seams. This is the name the fire
/// path has always called it by.
fn equipped_weapon(world: &EcsWorld, guid: Uuid) -> Option<(String, WeaponDef)> {
    weapon::equipped_def(world, guid)
}

/// Every character the weapon step visits, in `Guid` order — `O(characters)`.
fn gunners(world: &EcsWorld) -> Vec<Uuid> {
    inf_ecs::movement::movement_targets(world)
}

fn step_weapons(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
    report: &mut GameplayReport,
) {
    for guid in gunners(world) {
        // The equipped weapon decides everything, and an unarmed character costs
        // one map probe.
        let Some(entity) = world.entity_of(guid) else {
            continue;
        };
        // **The edges are TAKEN here, whether or not they are honoured** — the
        // law `step_one`'s own interact edge follows, for the same reason: an
        // edge a path could not use must not survive into a path that can, or a
        // press made in mid-air fires a kick on landing (the P29.7 A1 class).
        // Taken **before** the equipped-weapon check, because an unarmed
        // character still kicks doors and still scrolls.
        let (held_attack, press_attack, want_reload, want_switch) = {
            let w = world.world_mut();
            match w.get_mut::<CharacterMovement>(entity) {
                Some(mut cm) => {
                    let out = (
                        cm.runtime.want_attack,
                        cm.runtime.press_attack,
                        cm.runtime.press_reload,
                        cm.runtime.weapon_switch,
                    );
                    cm.runtime.press_attack = false;
                    cm.runtime.press_reload = false;
                    cm.runtime.weapon_switch = 0;
                    out
                }
                None => (false, false, false, 0),
            }
        };
        // The scroll wheel, before the trigger: a player who scrolls and fires
        // on one step fires the weapon they scrolled to, which is what they
        // asked for.
        if want_switch != 0 {
            let defs = item::item_defs(world).cloned().unwrap_or_default();
            let w = world.world_mut();
            if let Some(mut inv) = w.get_mut::<Inventory>(entity) {
                inv.cycle_equipped(&defs, want_switch, |d| d.is_weapon());
            }
        }
        // **THE ATTACK BUTTON'S THREE VERBS**, arbitrated once, here.
        //
        // A locked door in kicking reach takes the press; anything else lets it
        // through to the trigger; and since wave WPN1 an empty hand is a trigger
        // too. It is decided on the **edge** and not the level, because a kick is
        // a press — and while a kick is pending the weapon does not fire, so
        // holding the button against a door kicks it rather than kicking and
        // then shooting it.
        //
        // The order is the arbitration and it is not arbitrary: a kick beats a
        // punch, because a player standing at a locked gate pressing the attack
        // button wants the gate open and not a bruised hand. `try_kick` refuses
        // every door that is not locked and in reach, so the punch is what the
        // press means everywhere else.
        let kicking = if press_attack {
            let feet = feet_of(world, guid);
            let yaw = world
                .world()
                .get::<CharacterMovement>(entity)
                .map(|cm| cm.runtime.aim_yaw_deg)
                .unwrap_or(0.0);
            match feet {
                Some(feet) => {
                    let band = bridge.sim_band(world);
                    try_kick(world, &band, guid, feet, yaw)
                }
                None => false,
            }
        } else {
            false
        };
        let pending_kick = world.world().get::<PendingKick>(entity).is_some();
        let want_fire = held_attack && !kicking && !pending_kick;
        // Now the weapon, if there is one — **or the fists, if there is not and
        // the button is down** (wave WPN1).
        //
        // A pair of hands is the third consumer of the edge and it goes through
        // the same `try_fire`, the same cooldown and the same
        // `weapon_state_bytes` a rifle does. The one thing it does not do is
        // arrive uninvited: the clock is installed on the first press and not at
        // spawn, so an unarmed character that has never thrown a punch carries
        // no `WeaponState` at all and every trace committed before this wave is
        // byte-identical. Once installed it stays, because that clock is what
        // stops the second punch arriving before the first has landed.
        let armed = equipped_weapon(world, guid);
        let punching = armed.is_none()
            && (want_fire
                || world
                    .world()
                    .get::<WeaponState>(entity)
                    .is_some_and(|s| s.item_id == weapon::FIST_ITEM));
        let Some((item_id, def)) =
            armed.or_else(|| punching.then(|| (weapon::FIST_ITEM.to_string(), weapon::fist_def())))
        else {
            // A character who put their weapon away keeps no ammunition clock:
            // a stale state is a magazine two weapons would share. The FEEL goes
            // with it (wave WPN2b) for the same reason and one more: a spring
            // left behind would go on folding trace bytes for a character with
            // nothing in its hands, and `feel_state_bytes`' empty-when-at-rest
            // rule is what keeps every pre-wave trace byte-identical.
            release_feel(world, guid, entity);
            world.world_mut().entity_mut(entity).remove::<WeaponState>();
            continue;
        };
        // Install (or replace) the ammunition clock. Replacing on an id change
        // is what stops two weapons sharing one magazine.
        let stale = world
            .world()
            .get::<WeaponState>(entity)
            .map(|s| s.item_id != item_id)
            .unwrap_or(true);
        if stale {
            world
                .world_mut()
                .entity_mut(entity)
                .insert(WeaponState::full(&item_id, &def));
        }
        // **The feel** (wave WPN2b) — installed beside the clock and replaced
        // with it, so a spring never carries one weapon's kick into another's.
        //
        // **A MELEE WEAPON GETS NONE**, and a pair of fists is a melee weapon.
        // `FIST_RPM`'s doc has said since the WPN1 audit that a punch moves no
        // bone, and `weapon_hands_gate`'s last arm asserts it: giving the fists a
        // recoil spring gave them a gun's aim kick as well (the profile's `stat
        // = 0` is 0.03 rad of pitch, not zero), and the punch started moving the
        // hero's look. A melee recoil is a real thing and it is not this wave's;
        // when it arrives it is its own profile, not a rifle's.
        if def.is_melee() {
            release_feel(world, guid, entity);
        } else if stale || world.world().get::<WeaponFeel>(entity).is_none() {
            // **The CAMERA's borrowed blend speed survives the replace.** The
            // springs and the blend are this weapon's and are reset with it, but
            // `blend_speed_prior` is what the rig had before any weapon touched
            // it — and a fresh `WeaponFeel` would capture the LAST weapon's ADS
            // speed as the "prior" and hand that back when the character
            // eventually disarms, leaving an authored camera permanently at a
            // rifle's rate.
            let prior = world
                .world()
                .get::<WeaponFeel>(entity)
                .map(|f| f.blend_speed_prior)
                .filter(|p| p.is_finite());
            let mut fresh = WeaponFeel::new();
            if let Some(prior) = prior {
                fresh.blend_speed_prior = prior;
            }
            world.world_mut().entity_mut(entity).insert(fresh);
        }
        // The animation's own reload notify, taken exactly once — the P29.4
        // seam, and the reason the fixed step asks rather than the animation
        // pushing: a notify is consumed by whoever gets there first, and there
        // must be exactly one consumer of a reload.
        let notified =
            inf_ecs::anim_bridge::consume_anim_notify(world, guid, weapon::RELOAD_NOTIFY);
        // **What this shot's cone is made of** (wave WPN2b), gathered before the
        // trigger because a shot fired on the step a character stands up is
        // fired from the stance it was in. `speed_planar` and not `speed`: a
        // character in a lift is not moving its weapon.
        let (stance, speed_mps, ads) = shot_posture(world, entity);
        let mut fired = false;
        let mut reloaded = false;
        let mut cone_deg = def.spread_deg;
        let (shot_index, aim) = {
            let mut aim = None;
            let mut shot = 0u64;
            let w = world.world_mut();
            if let Some(mut state) = w.get_mut::<WeaponState>(entity) {
                if notified {
                    reloaded |= weapon::finish_reload(&def, &mut state);
                }
                reloaded |= weapon::advance(&def, &mut state, dt);
                // The bloom decays whether or not this weapon fires — it is the
                // half that makes a burst worse than five aimed shots.
                feel::decay_bloom(&def, &mut state, dt);
                if want_reload {
                    weapon::try_reload(&def, &mut state);
                }
                if weapon::try_fire(&def, &mut state, want_fire) == FireVerdict::Fired {
                    fired = true;
                    shot = state.shots;
                    // THE CONE THIS ROUND LEAVES THROUGH, taken BEFORE this
                    // round's own bloom is added: the first round of a burst is
                    // as accurate as the weapon is, and the second one pays.
                    cone_deg = feel::resolved_cone_deg(
                        &def,
                        state.spread_bloom_deg,
                        stance,
                        speed_mps,
                        ads,
                    );
                    feel::bloom_on_shot(&def, &mut state);
                }
            }
            if fired {
                aim = muzzle_of(world, guid);
            }
            (shot, aim)
        };
        if fired {
            // **THE RECOIL IMPULSE** — deposited on both springs at once, from
            // the weapon's own `recoil_intensity` through the doc's mapping. The
            // viewmodel half is read by `aim_hold_point` and the aim half is
            // spent by the movement runtime's look integrator on the NEXT step,
            // which is the right way round: the round that caused the kick left
            // along the aim the player had.
            let profile = feel::RecoilProfile::of(&def);
            if let Some(mut f) = world.world_mut().get_mut::<WeaponFeel>(entity) {
                f.fire(&profile, def.spread_seed, shot_index, dt);
            }
        }
        if reloaded {
            report.reloads += 1;
        }
        if want_reload {
            // The animation follows the decision rather than gating it, exactly
            // as the ragdoll's does: an unrigged character still reloads.
            inf_ecs::anim_bridge::set_anim_trigger(world, guid, weapon::RELOAD_TRIGGER);
        }
        if !fired {
            continue;
        }
        // The animation, and it is a different one for a swing: a rig that
        // played `weapon_fire` when somebody threw a punch would be firing an
        // empty hand.
        inf_ecs::anim_bridge::set_anim_trigger(
            world,
            guid,
            if def.is_melee() {
                weapon::MELEE_TRIGGER
            } else {
                weapon::FIRE_TRIGGER
            },
        );
        report.shots += 1;
        if def.is_melee() {
            report.swings += 1;
        }
        let Some((from, yaw, pitch, from_weapon)) = aim else {
            continue;
        };
        // **The fallback, counted** (SK1b audit). A character with no pose at all
        // is the legitimate capsule case — every level committed before SK1b —
        // and is not counted. A character that *is* posed and still took the
        // capsule rule is a rig that does not publish `WEAPON_SOCKET`, or a
        // weapon entity that has not been placed yet, and it puts every shot back
        // at 1.4 m in silence. See `GameplayReport::muzzles_without_a_socket`.
        //
        // **A punch is not counted either** (wave WPN1), and it would have been:
        // a fist has no weapon entity to hang off a socket, so a rigged
        // character throwing one takes the capsule rule *correctly* and would
        // have tripped the tripwire on every swing. A gate that asserts this is
        // zero on a rigged course has to be able to punch on it.
        if !def.is_melee() && !from_weapon && inf_ecs::pose::evaluated_pose(world, guid).is_some() {
            report.muzzles_without_a_socket += 1;
        }
        let dir = weapon::shot_direction_with(&def, yaw, pitch, shot_index, cone_deg);
        // **THE BRASS** (wave WPN2c) — one case per round that leaves a barrel,
        // thrown out of the weapon's own ejection port. A melee weapon ejects
        // nothing (a fist has no port), and neither does one whose port speed is
        // zero, which is how a definition opts out without a flag.
        //
        // **One case per PULL, not per pellet** (wave WPN2d): a shotgun throws
        // eight pellets out of one shell and ejects one shell.
        if !def.is_melee() && def.eject_speed_mps > 0.0 {
            let port = inf_ecs::casing::eject_point(from, &def, yaw);
            let throw = inf_ecs::casing::eject_velocity(&def, yaw);
            if inf_ecs::casing::eject_casing(world, guid, def.audio_class(), port, throw).is_some()
            {
                report.casings.ejected += 1;
            }
        }
        if def.is_melee() {
            let hit = resolve_swing(world, bridge, guid, &def, from, dir, yaw, report);
            // **THE CONTACT** (wave WPN2d) — only when the blow LANDED, and by
            // what it landed on. A swing at thin air makes no noise, which is
            // what `hit.target` being `None` means.
            if hit.target.is_some() {
                report.melee_impacts.push(MeleeImpact {
                    shooter: guid,
                    at: hit.to,
                    surface: inf_ecs::weapon::ImpactSurface::of(hit.on_flesh),
                });
            }
            apply_hit(world, &hit, dt, report);
            report.hits.push(hit);
            continue;
        }
        // **THE PATTERN** (wave WPN2d) — the doc section 5's *"multi-raycast
        // cone spread"*. `pellets` is 1 for every weapon that is not a shotgun,
        // so this loop runs once and casts exactly the ray it cast before this
        // wave, along exactly the direction `shot_direction_with` gave it above.
        //
        // Above one it is N casts through the weapon's OWN `cone_deg` at N
        // consecutive counter indices, each carrying `damage_j / pellets` — so
        // the doc's per-pellet table comes back out of the registry's
        // whole-pull figure by division rather than by a second column.
        // **THE PULL'S OWN CONTEXT**, resolved once for the whole pattern.
        let ctx = {
            let exclude = shot_exclusions(world, bridge, guid);
            shot_context(world, bridge, from, &exclude, report)
        };
        let pellets = def.pellet_count();
        if pellets == 1 {
            let hit = resolve_shot(
                world, bridge, guid, &def, from, dir, shot_index, &ctx, report,
            );
            if hit.loud {
                inf_ecs::casing::note_shot_room(world, hit.indoors);
            }
            apply_hit(world, &hit, dt, report);
            report.hits.push(hit);
            continue;
        }
        // A pellet is the weapon with its joules divided. Building a def rather
        // than passing a scale keeps ONE damage door: `resolve_shot` asks
        // `damage_at`, `damage_at` asks the curve, and the curve is evaluated at
        // the pellet's own distance exactly as it is for a bullet.
        let mut pellet_def = def;
        pellet_def.damage_j = def.pellet_damage_j();
        // The pattern's own cone. A row that names none takes the resolved pull
        // cone, which is what a slug is: one projectile, no pattern.
        let pattern_deg = if def.cone_deg > 0.0 {
            def.cone_deg
        } else {
            cone_deg
        };
        for p in 0..pellets {
            // **THE CEILING, REFUSED WITH A VALUE** (the law). A pull that would
            // cross `MAX_SHOT_RAYS_PER_STEP` stops here and COUNTS what it did
            // not throw: a silently shortened pattern is a shot the player fired
            // and nobody can account for. The bill is the step's own running
            // one, which `resolve_shot` keeps.
            let bill = report.rounds.shot_rays as usize + 1;
            if bill > inf_ecs::ballistics::MAX_SHOT_RAYS_PER_STEP {
                report.rounds.pellets_refused += pellets - p;
                break;
            }
            // Consecutive counter indices, strided by the pellet bound so two
            // pulls of a 64-pellet weapon can never draw the same pair of
            // uniforms — the counter hash is a function of (seed, index) and
            // nothing else, which is what makes a pattern replay.
            let index = shot_index
                .wrapping_mul(u64::from(weapon::MAX_PELLETS))
                .wrapping_add(u64::from(p));
            let pdir = weapon::shot_direction_with(&pellet_def, yaw, pitch, index, pattern_deg);
            let mut hit = resolve_shot(
                world,
                bridge,
                guid,
                &pellet_def,
                from,
                pdir,
                index,
                &ctx,
                report,
            );
            // **ONE BANG PER PULL.** Only the first pellet is `loud`, because
            // `loud` is what both hosts' `weapon_report` fence queues four
            // layers off and what `panic_sources` coalesces on: eight loud
            // pellets would be thirty-two commands and eight witness sources for
            // one trigger pull.
            hit.loud = p == 0;
            if hit.loud {
                inf_ecs::casing::note_shot_room(world, hit.indoors);
            }
            report.rounds.pellets += 1;
            apply_hit(world, &hit, dt, report);
            report.hits.push(hit);
        }
    }
}

/// Cast one shot and answer where it landed — **the near half of the hybrid**
/// (wave WPN2a).
///
/// The cast reaches [`WeaponDef::hitscan_reach_m`], which for a
/// [`ShotKind::Hitscan`] is the weapon's whole range (so every level committed
/// before this wave resolves exactly as it did) and for a
/// [`ShotKind::Projectile`] is the smaller of its range and its
/// `hitscan_threshold_m`. A projectile that missed inside the threshold and has
/// range left to fly through **mints a round** here, at the threshold point,
/// carrying `muzzle_speed_mps` along the same spread direction — the doc's
/// *"instantiate a bullet struct at the 20-meter point along the vector with
/// initial velocity v"*.
///
/// The `WeaponHit` it answers with in that case is a **miss** whose `to` is the
/// spawn point, and that is deliberate: the shot happened, it made a noise, and
/// the street should panic from the muzzle now rather than in half a second when
/// the round lands. The impact, if there is one, comes back later as its own
/// quiet hit from [`step_rounds`].
///
/// The step's own report comes in rather than a ray count and a round report,
/// because the spawn refusal has to cover the WHOLE step's ray bill: one cast
/// per shot fired so far (`report.shots`) plus what the flight will spend — see
/// `inf_ecs::ballistics::MAX_SHOT_RAYS_PER_STEP`.
/// **Everything the SHOOTER is**, as colliders a shot must not stop on (wave
/// WPN2a, closed by its audit) — the one exclusion set both halves of the
/// hybrid build.
///
/// It exists because the shot's cast class changed. Until this audit both
/// halves went through `cast_ray_excluding`, which is
/// [`CastTargets::Fixed`](super::CastTargets::Fixed), so a bullet passed
/// through a car, a crate, a fractured chunk and a **ragdoll** — and the
/// shooter exclusion was belt to a brace, because the only dynamic body it
/// could ever have named was already invisible. The casts ask
/// [`CastTargets::AllSolid`](super::CastTargets::AllSolid) now, and the moment
/// they can see a dynamic body the exclusion has three jobs rather than one:
///
/// 1. **the shooter's own capsule**, which is where it started;
/// 2. **the chassis it is sitting in** — `MovementRuntime::seat` names it, and a
///    driver firing through a windscreen must not shoot the car out from under
///    itself on segment 0. This is the half the wave could not measure and
///    `a_round_does_not_hit_its_own_shooter_or_the_chassis_it_is_sitting_in`
///    now can;
/// 3. **its own ragdoll's limbs**, if it has any. A ragdolling character cannot
///    pull a trigger today, so this is the cheap half of "the set is what the
///    shooter IS" rather than a behaviour anybody can see — and it costs one
///    map lookup on a map that is empty on every level with nobody on the floor.
///
/// Sensors are not in it, and do not need to be: `AllSolid` is
/// `QueryFilter::default().exclude_sensors()`, so a trigger volume cannot stop
/// a bullet.
fn shot_exclusions(
    world: &EcsWorld,
    bridge: &PhysicsBridge3D,
    shooter: Uuid,
) -> BTreeSet<super::ColliderId3D> {
    let mut exclude = BTreeSet::new();
    if let Some(c) = bridge.collider_of(shooter) {
        exclude.insert(c);
    }
    if let Some(r) = bridge.ragdoll_of(shooter) {
        exclude.extend(r.colliders.iter().copied());
    }
    let seated = world
        .entity_of(shooter)
        .and_then(|e| world.world().get::<CharacterMovement>(e))
        .map(|cm| cm.runtime.seat.vehicle)
        .filter(|v| !v.is_nil());
    if let Some(c) = seated.and_then(|v| bridge.collider_of(v)) {
        exclude.insert(c);
    }
    exclude
}

/// **Whose body a shot's cast just hit** (wave WPN2a, closed by its audit) —
/// [`PhysicsBridge3D::guid_of_collider`], and then the ragdolls.
///
/// A ragdolling character's own capsule is *disabled* and its limbs are
/// articulated bodies the bridge attached itself, so they are not in the
/// collider→guid index a document entity is in. Without this door a round that
/// hit somebody lying on the floor named nobody, `is_flesh` answered `false`
/// and the joules went nowhere: a body on the ground was bulletproof.
///
/// The scan is over `ragdoll_count()` entries and runs only when the index
/// misses, which is a hit on terrain, a structure or a limb — and the map is
/// empty on every level where nobody is down.
fn hit_owner(bridge: &PhysicsBridge3D, collider: super::ColliderId3D) -> Option<Uuid> {
    bridge
        .guid_of_collider(collider)
        .or_else(|| bridge.guid_of_ragdoll_collider(collider))
}

/// **What a TRIGGER PULL knows that a projectile does not** (wave WPN2d) — the
/// room it was fired in, and how far the ear is.
///
/// Both are questions about where the MUZZLE is, and a shotgun's eight pellets
/// leave one muzzle: computing them per pellet is six extra rays each for an
/// identical answer, and it counts one pull as eight indoor shots. Measured by
/// the gate's own cost arm before this existed: an eight-pellet pull spent **56
/// casts** where it needs 14.
///
/// So it is resolved ONCE, by the fire path, and handed to every pellet. For a
/// single-projectile weapon that is exactly the work `resolve_shot` did before
/// this wave, in the same order, against the same exclusion set.
struct ShotContext {
    /// What the six-ray probe said about the muzzle's surroundings.
    enclosure: super::audio::Enclosure,
    /// How far the active listener is from the muzzle, or `INFINITY` when
    /// nobody is listening — which makes the distant layer silent, the honest
    /// answer rather than a zero that would make it loudest.
    listener_m: f64,
}

/// Resolve a pull's [`ShotContext`], spending the probe's rays and counting
/// them.
fn shot_context(
    world: &EcsWorld,
    bridge: &mut PhysicsBridge3D,
    from: DVec3,
    exclude: &BTreeSet<super::ColliderId3D>,
    report: &mut GameplayReport,
) -> ShotContext {
    let enclosure = super::audio::enclosure_at(bridge.world_mut(), from, exclude);
    report.rounds.probe_rays += super::audio::ENCLOSURE_PROBE_RAYS as u32;
    report.rounds.shot_rays += super::audio::ENCLOSURE_PROBE_RAYS as u32;
    if enclosure.indoors {
        report.rounds.indoor_shots += 1;
    }
    let listener_m = inf_ecs::audio::active_listener_position(world)
        .map(|p| (p - from).length())
        .unwrap_or(f64::INFINITY);
    ShotContext {
        enclosure,
        listener_m,
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_shot(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    shooter: Uuid,
    def: &WeaponDef,
    from: DVec3,
    dir: DVec3,
    shot_index: u64,
    ctx: &ShotContext,
    report: &mut GameplayReport,
) -> WeaponHit {
    // The step's ray bill so far. The rounds already in the air are deliberately
    // NOT counted — `spawn_round` counts them itself, off the pool it is about
    // to push into. Counting them in both places halved the effective bound:
    // measured, at eight shooters and 900 rpm the pool peaked at **32** rounds
    // against a stated ceiling of 64 and refused 301 spawns.
    //
    // **Seven, not one** (wave WPN2c). The enclosure probe below casts
    // `ENCLOSURE_PROBE_RAYS` more per shot, and a ceiling that did not know
    // about them would be a ceiling on a sixth of the real bill.
    //
    // **It is a RUNNING COUNT since wave WPN2d, not `shots × 7`.** A shotgun
    // pull is one shot and up to sixty-four casts, so the product was an
    // estimate that a pattern makes wrong by a factor of the pellet count — and
    // the ceiling is the thing the pellet loop refuses against. `shot_rays` is
    // incremented here, once, by exactly what THIS call is about to spend — one
    // cast — and the pull's own probe is counted by [`shot_context`], once, for
    // the whole pattern.
    report.rounds.shot_rays += 1;
    let rays_already = report.rounds.shot_rays as usize;
    let range = def.range_m.clamp(0.1, SHOT_MAX_RANGE_M);
    let reach = def.hitscan_reach_m().clamp(0.0, range);
    let exclude = shot_exclusions(world, bridge, shooter);
    // **THE ROOM AND THE EAR** (wave WPN2c) come in on the pull's own context
    // since wave WPN2d: both are questions about where the MUZZLE is, and eight
    // pellets leave one muzzle. See [`ShotContext`].
    let enclosure = &ctx.enclosure;
    let listener_m = ctx.listener_m;
    // A launcher's threshold is zero, and a zero-length cast is not a cast: skip
    // it rather than clamping it up to 0.1 m, which would put a rocket's first
    // ten centimetres inside a rule that has nothing to say about them.
    //
    // **`AllSolid`, not `Fixed`** (the WPN2a audit's closure of carried 200): a
    // round must stop at a parked car and must hit a body on the floor. Sensors
    // stay out, so a trigger volume is still not a wall.
    let landed = (reach > 0.0)
        .then(|| {
            bridge.world_mut().cast_ray_where(
                from,
                dir,
                reach,
                &exclude,
                super::CastTargets::AllSolid,
            )
        })
        .flatten();
    match landed {
        Some(h) => {
            let target = hit_owner(bridge, h.collider);
            let on_flesh = target.is_some_and(|g| is_flesh(world, g));
            let point = from + dir * h.toi;
            let headshot = on_flesh
                && target.is_some_and(|g| {
                    head_hit(world, g, point, &mut report.rounds.heads_without_a_socket)
                });
            if headshot {
                report.rounds.headshots += 1;
            }
            WeaponHit {
                shooter,
                target,
                from,
                to: point,
                // ONE damage door (wave WPN2a): the curve and the head
                // multiplier, evaluated at the cast's own distance. A weapon
                // that named no ranges answers `damage_j` at every distance,
                // which is the I6 number.
                energy_j: def.damage_at(h.toi, headshot),
                on_flesh,
                loud: true,
                arrived: false,
                headshot,
                report_max_m: def.report_max_m,
                report_gain: def.report_gain,
                class: def.audio_class(),
                indoors: enclosure.indoors,
                listener_m,
                shot_index,
            }
        }
        None => {
            // **The far half.** Nothing inside the threshold; if this weapon
            // flies, a round leaves here.
            if def.spawns_a_round() {
                // **WHAT KIND OF BODY** (wave WPN2d). A weapon with a motor or a
                // blast is a ROCKET — the two are the same thing from the pool's
                // side, an accelerating body that goes off — and everything else
                // is the bullet wave WPN2a minted. A THROWN body never comes
                // through here at all: `step_throws` is its door, because it
                // leaves a hand rather than a barrel.
                let kind = if def.accel_mps2 > 0.0 || def.has_blast() {
                    inf_ecs::ballistics::RoundKind::Rocket
                } else {
                    inf_ecs::ballistics::RoundKind::Bullet
                };
                // **THE LOCK IT LEAVES ON.** Read once, at the muzzle, and never
                // re-acquired: the lock a player earned is the one that gets
                // spent, and a missile that could pick a new target mid-air is a
                // different weapon.
                let guide = world
                    .entity_of(shooter)
                    .and_then(|e| world.world().get::<weapon::WeaponState>(e))
                    .and_then(|st| st.locked_on(def))
                    .unwrap_or_else(Uuid::nil);
                let round = inf_ecs::ballistics::Round {
                    shooter,
                    at: from + dir * reach,
                    velocity: dir * def.muzzle_speed_mps.max(1.0),
                    travelled_m: reach,
                    age_s: 0.0,
                    first_segment: true,
                    cracked: false,
                    kind,
                    fuse_left_s: def.fuse_s,
                    bounces: 0,
                    guide,
                    def: *def,
                };
                if inf_ecs::ballistics::spawn_round(world, round, rays_already) {
                    report.rounds.spawned += 1;
                } else {
                    report.rounds.refused += 1;
                }
            }
            WeaponHit {
                shooter,
                target: None,
                from,
                to: from + dir * reach.max(range.min(reach)),
                energy_j: def.damage_at(reach, false),
                on_flesh: false,
                loud: true,
                arrived: false,
                headshot: false,
                report_max_m: def.report_max_m,
                report_gain: def.report_gain,
                class: def.audio_class(),
                indoors: enclosure.indoors,
                listener_m,
                shot_index,
            }
        }
    }
}

/// **Where a character's head is**, world metres — the one door a headshot is
/// tested against, with two answers exactly as [`muzzle_of`] has.
///
/// 1. **The rig's own `head` socket.** `inf_anim::manny` publishes it (under
///    both this engine's spelling and ALS's `head_socket`), the pose step
///    resolves it into model space every step, and `pose::model_to_world` is the
///    door that lifts a point on the rig into the world — the same one
///    `update_attachments` uses, so a head and a weapon are placed by one rule.
/// 2. **A height above the feet**, for a character with no pose at all — every
///    level committed before SK1b, the whole `phase30-gameplay` fixture, and any
///    crowd agent the sim has tiered out of posing. The height is the CAPSULE's
///    own and not a constant somebody chose: a capsule stands with its centre at
///    `feet + h + r` and its top sphere's centre `h` above that, so the head is
///    at `feet + 2h + r` — 1.50 m on the shipped default, which is the neck and
///    shoulder line of a 1.80 m body.
///
/// A **posed** character whose rig authors no `head` socket takes answer 2 and
/// is **counted** ([`RoundReport::heads_without_a_socket`]), which is
/// `muzzles_without_a_socket`'s discipline one joint along.
fn head_point(world: &EcsWorld, guid: Uuid, no_socket: &mut u32) -> Option<DVec3> {
    let entity = world.entity_of(guid)?;
    if let Some(pose) = inf_ecs::pose::evaluated_pose(world, guid) {
        if let Some(m) = pose.socket(HEAD_SOCKET) {
            let to_world = inf_ecs::pose::model_to_world(world, entity);
            let local = glam::DAffine3::from_mat4(m.as_dmat4());
            let at = (to_world * local).translation;
            if at.is_finite() {
                return Some(at);
            }
        } else {
            *no_socket += 1;
        }
    }
    let cm = world.world().get::<CharacterMovement>(entity)?;
    let radius = world
        .world()
        .get::<inf_ecs::components::Collider3D>(entity)
        .map(|c| c.radius)
        .unwrap_or(0.3);
    let feet = feet_of(world, guid)?;
    Some(feet + DVec3::Y * (2.0 * cm.half_height_for(cm.mode) + radius))
}

/// Whether `point` arrived at `target`'s head — [`head_point`] plus
/// [`inf_ecs::ballistics::is_headshot`]'s band, over the target's OWN collider
/// radius, and nothing else.
fn head_hit(world: &EcsWorld, target: Uuid, point: DVec3, no_socket: &mut u32) -> bool {
    let radius = world
        .entity_of(target)
        .and_then(|e| world.world().get::<inf_ecs::components::Collider3D>(e))
        .map(|c| c.radius)
        .unwrap_or(0.0);
    head_point(world, target, no_socket)
        .is_some_and(|head| inf_ecs::ballistics::is_headshot(point, head, radius))
}

/// **Fly every round in the pool**, one fixed step (wave WPN2a).
///
/// # Where this runs, and why it is not a `STEP_PHASES` row
///
/// Inside [`step_gameplay`], between `step_weapons` and `step_kicks`. The
/// alternative was a thirtieth `STEP_PHASES` row on the `vehicle` precedent —
/// attribution, a mirrored call site in both hosts and a `fixed_step_mirror`
/// fence — and it is refused for a reason that is about correctness rather than
/// about cost: **a round's impact has to reach this step's own witness, panic
/// and death passes**. Those run at the bottom of `step_gameplay` over
/// `report.hits`; a phase after gameplay would file every projectile kill one
/// step late (and `step_deaths` would hand a body to the ragdoll a step after
/// the blow), and a phase before it would advance a round on the step before the
/// shot that minted it. Hoisting the four passes out to meet the new row is a
/// bigger edit to a more load-bearing thing than a budget row is worth.
///
/// What that costs is stated rather than hidden: [`WEAPON_STEP_BUDGET_MS`] is a
/// ceiling on the **`gameplay` phase**, which holds the doors, the kicks, the
/// hands and the cover pass as well — so the arm that asserts it also reports
/// the DELTA between a step with rounds in the air and one without, which is the
/// pool's own cost with nothing else in it.
///
/// [`WEAPON_STEP_BUDGET_MS`]: inf_player::budget::WEAPON_STEP_BUDGET_MS
///
/// # The segment cast
///
/// **Every casing in the air** (wave WPN2c) — gravity, one bounce, and a
/// settle.
///
/// # Where the cost is, and where it is not
///
/// One raycast per **airborne** casing per step and nothing at all for one that
/// has settled: a floor covered in brass is an age increment each. That is the
/// whole reason the lifetime can be eight seconds and the ring a hundred and
/// twenty-eight — at eight shooters and 600 rpm the pool is full and almost all
/// of it is already down.
///
/// # The shooter is excluded for the casing's whole flight
///
/// A casing is born at the muzzle, which on a character with no rig is *inside*
/// its own capsule (`muzzle_of`'s height fallback), so a first ray that could
/// see the shooter would bounce every case off its owner's chest. The exclusion
/// set is built once per shooter per step rather than once per casing, because
/// eight shooters can own a hundred and twenty-eight casings between them.
///
/// Inert on every level that has never fired: one absent-resource read.
fn step_casings(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
    report: &mut GameplayReport,
) {
    use inf_ecs::casing::{
        advance_casing, bounce_pitch, bounce_velocity, casing_guid, CasingPool,
        CASING_CONTACT_EPS_M, CASING_LIFETIME_S,
    };
    if world.world().get_resource::<CasingPool>().is_none() {
        return;
    }
    let live: Vec<inf_ecs::casing::Casing> =
        world.world().resource::<CasingPool>().casings.to_vec();
    if live.is_empty() {
        return;
    }
    let mut excludes: std::collections::BTreeMap<Uuid, BTreeSet<super::ColliderId3D>> =
        std::collections::BTreeMap::new();
    let mut survivors: Vec<inf_ecs::casing::Casing> = Vec::with_capacity(live.len());
    for mut c in live {
        c.age_s += dt;
        if c.age_s >= CASING_LIFETIME_S || !c.at.is_finite() {
            report.casings.expired += 1;
            continue;
        }
        if !c.settled() {
            let prev = c.at;
            let (next, v) = advance_casing(c.at, c.velocity, dt);
            let seg = next - prev;
            let len = seg.length();
            let mut landed = false;
            if len > 1e-9 {
                let exclude = excludes
                    .entry(c.shooter)
                    .or_insert_with(|| shot_exclusions(world, bridge, c.shooter));
                report.casings.rays += 1;
                if let Some(h) = bridge.world_mut().cast_ray_where(
                    prev,
                    seg / len,
                    len,
                    exclude,
                    super::CastTargets::AllSolid,
                ) {
                    landed = true;
                    c.contacts = c.contacts.saturating_add(1);
                    c.at = h.point + h.normal.normalize_or_zero() * CASING_CONTACT_EPS_M;
                    c.velocity = bounce_velocity(v, h.normal);
                    // **ONE SOUND PER CASING**, on its FIRST contact. The second
                    // is where it stops and is not a bounce; a third would be
                    // the skitter this pool deliberately does not simulate.
                    if c.contacts == 1 {
                        report.casings.bounces.push(CasingBounce {
                            casing: casing_guid(c.shooter, c.seq),
                            at: h.point,
                            pitch: bounce_pitch(c.seq),
                            class: c.class,
                        });
                    }
                    if c.settled() {
                        c.velocity = DVec3::ZERO;
                        c.spin_deg_s = DVec3::ZERO;
                        report.casings.settled += 1;
                    }
                }
            }
            if !landed {
                c.at = next;
                c.velocity = v;
            }
            c.angle_deg += c.spin_deg_s * dt;
        }
        survivors.push(c);
    }
    let mut pool = world.world_mut().resource_mut::<CasingPool>();
    pool.casings = survivors;
    pool.bounced += report.casings.bounces.len() as u64;
    pool.settled += u64::from(report.casings.settled);
    pool.expired += u64::from(report.casings.expired);
    report.casings.live = pool.casings.len() as u32;
    report.casings.recycled = pool.recycled;
}

/// **The brass, as things you can see** (wave WPN2c) — one entity per live
/// casing, on `step_equipped_weapons`' doctrine exactly.
///
/// # Why entities and not a scatter batch
///
/// The P22.4 GPU scatter path keys a batch on a **content hash of its packed
/// instances**, so a batch whose instances move re-uploads every step — which
/// is why the rubble it was built for is frozen at its rest pose. Tumbling
/// brass has no rest pose. The per-`MeshRef` instanced path does what is
/// wanted for free: the renderer buckets by `(blend, primitive)` and issues one
/// `draw_indexed` per bucket, so a hundred and twenty-eight cylinders are one
/// draw call.
///
/// The guid is DERIVED (`casing_guid`), so both hosts spawn the same entity for
/// the same casing on the same step; the entity appears in the trace as a
/// transform row, which is the honest cost and is the same one the equipped
/// weapon pays.
///
/// Inert on every level that has never fired: one absent-resource read.
fn step_casing_entities(world: &mut EcsWorld) {
    use inf_ecs::casing::{
        casing_guid, CasingMark, CasingPool, CASING_COLOR, CASING_LENGTH_M, CASING_RADIUS_M,
        CASING_ROUGHNESS,
    };
    use inf_ecs::components::{Material, MeshRef, Primitive, Transform, Visibility};
    if world.world().get_resource::<CasingPool>().is_none() {
        return;
    }
    let live: Vec<(Uuid, DVec3, DVec3)> = world
        .world()
        .resource::<CasingPool>()
        .casings
        .iter()
        .map(|c| (casing_guid(c.shooter, c.seq), c.at, c.angle_deg))
        .collect();
    let want: BTreeSet<Uuid> = live.iter().map(|(g, _, _)| *g).collect();
    // Anything marked as brass that the pool no longer holds is gone.
    let stale: Vec<Uuid> = inf_ecs::casing::drawn_casings(world)
        .into_iter()
        .filter(|g| !want.contains(g))
        .collect();
    for g in stale {
        if let Some(e) = world.entity_of(g) {
            world.despawn(e);
        }
    }
    for (g, at, angle) in live {
        let e = match world.entity_of(g) {
            Some(e) => e,
            None => world.spawn_with_guid(g, "Brass", None),
        };
        let mut t = Transform::IDENTITY;
        t.translation = inf_ecs::math::Vec3d::new(at.x, at.y, at.z);
        t.rotation = inf_ecs::math::Vec3d::new(angle.x, angle.y, angle.z);
        t.scale = inf_ecs::math::Vec3d::new(
            CASING_RADIUS_M * 2.0,
            CASING_RADIUS_M * 2.0,
            CASING_LENGTH_M,
        );
        world.world_mut().entity_mut(e).insert((
            t,
            MeshRef {
                // A tiny cylinder is the committed fallback the brief names.
                // The UE `SM_Shell_*_Empty` art is LOCAL-ONLY and reaches this
                // through `MeshRef::asset`, which is one field and no schema.
                primitive: Primitive::Cylinder,
                asset: None,
            },
            // **AND IT IS BRASS** (the wave's audit). Without a `Material` the
            // projector's default is a white dielectric, and a white cylinder is
            // the one thing a shell casing is not -- measured by rendering the
            // shipped projector at half a metre and looking at it. See
            // `inf_ecs::casing::CASING_COLOR`.
            Material {
                base_color: inf_ecs::math::Color::new(
                    CASING_COLOR[0],
                    CASING_COLOR[1],
                    CASING_COLOR[2],
                    1.0,
                ),
                metallic: 1.0,
                roughness: CASING_ROUGHNESS,
                ..Default::default()
            },
            Visibility::default(),
            CasingMark,
        ));
    }
}

/// [`inf_ecs::ballistics::PROJECTILE_SUB_STEPS`] per fixed step, and each
/// sub-step is a cast from the previous position to the next one through the
/// **same** `cast_ray_excluding` door the instant ray uses — never a test of the
/// endpoint. The shooter is excluded on the first segment only. A hit resolves
/// through the **same** `apply_hit` door with the round's remaining energy, so a
/// projectile kill spends its joules, staggers, panics the street and is
/// witnessed exactly as a hitscan kill is.
/// **Where a guided round should be aiming**, for anything (wave WPN2d).
///
/// [`strike_point`] is a CHARACTER's — it is `feet_of` plus a height, and
/// `feet_of` reads `CharacterMovement`. The only things this engine's lock can
/// acquire are VEHICLES, which have none, so a door that asked `strike_point`
/// answered `None` for every target a lock can hold and a guided missile flew
/// exactly as straight as a dumb one. Measured, by the arm that found it: 5.11 m
/// from a stationary car either way.
///
/// So: a character's own strike point when it has one, and the entity's world
/// transform otherwise — which is a car's centre, a helicopter's, and a crate's.
fn target_point(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    if let Some(p) = strike_point(world, guid) {
        return Some(p);
    }
    let e = world.entity_of(guid)?;
    world
        .world()
        .get::<inf_ecs::components::GlobalTransform>(e)
        .map(|g| g.translation())
        .filter(|p| p.is_finite())
}

fn step_rounds(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    band: &inf_ecs::band::SimBand,
    dt: f64,
    report: &mut GameplayReport,
) {
    use inf_ecs::ballistics::{RoundPool, MAX_ROUND_LIFETIME_S, PROJECTILE_SUB_STEPS};
    if world.world().get_resource::<RoundPool>().is_none() {
        return;
    }
    let live: Vec<inf_ecs::ballistics::Round> =
        world.world().resource::<RoundPool>().rounds.to_vec();
    if live.is_empty() {
        return;
    }
    let sub_dt = dt / f64::from(PROJECTILE_SUB_STEPS);
    // **WHERE EVERY GUIDE IS** (wave WPN2d), resolved ONCE for the whole pool
    // rather than per round per sub-step — the ear's own argument: a target
    // cannot move inside a fixed step, and up to 256 world walks for a position
    // that cannot change is 256 walks too many. Empty on every level where
    // nothing in the air is guided, which is every level before this wave.
    let guides: std::collections::BTreeMap<Uuid, DVec3> = live
        .iter()
        .filter(|r| !r.guide.is_nil())
        .filter_map(|r| target_point(world, r.guide).map(|p| (r.guide, p)))
        .collect();
    // **WHERE THE EAR IS** (wave WPN2c) — resolved ONCE for the whole pool
    // rather than per round per sub-step, which is up to 256 walks of the world
    // for a number that cannot change inside a fixed step.
    let ear = inf_ecs::audio::active_listener_position(world);
    let mut survivors: Vec<inf_ecs::ballistics::Round> = Vec::with_capacity(live.len());
    let mut landed: Vec<(WeaponHit, f64)> = Vec::new();
    // **What went off, and where** (wave WPN2d) — collected inside the flight
    // loop and spent below it, because the pool's borrow is live in there and
    // `apply_blast` writes to the world.
    let mut blasts: Vec<(Uuid, DVec3, WeaponDef)> = Vec::new();
    for mut r in live {
        let mut alive = true;
        for _ in 0..PROJECTILE_SUB_STEPS {
            let prev = r.at;
            // **THE FUSE** (wave WPN2d) — the `KICK_FUSE_S` pattern: a clock that
            // counts DOWN and fires when it reaches zero, so a body with no fuse
            // carries a zero and this is one comparison.
            //
            // It is at the TOP of the sub-step, before the cast, and that is a
            // correction rather than a taste: at the bottom it was skipped by
            // every `continue` the loop has — a bounce and a settle — so a
            // grenade that came to rest on a pavement STOPPED COUNTING. Measured
            // by the arm that found it: a three-second fuse ran 3.317 s, which
            // is three seconds plus the time the body spent lying still.
            //
            // A grenade whose fuse runs out in mid-air goes off in mid-air, and
            // one that has landed goes off where it landed, because `r.at` is
            // wherever the last sub-step left it.
            if r.def.fuse_s > 0.0 {
                r.fuse_left_s -= sub_dt;
                if r.fuse_left_s <= 0.0 {
                    report.rounds.fused += 1;
                    if r.def.has_blast() {
                        blasts.push((r.shooter, r.at, r.def));
                    }
                    alive = false;
                    break;
                }
            }
            // **THE GUIDANCE** (wave WPN2d), before the integrator, because what
            // it changes is the velocity the integrator is about to use. A round
            // with no guide is one map lookup that misses on an empty map.
            if let Some(target) = guides.get(&r.guide) {
                let bearing =
                    inf_ecs::ballistics::guidance_bearing(r.at, *target, r.def.top_attack);
                r.velocity = inf_ecs::ballistics::guide_velocity(
                    r.velocity,
                    bearing,
                    inf_ecs::ballistics::GUIDANCE_TURN_DPS,
                    sub_dt,
                );
                report.rounds.guided += 1;
            }
            let (next, v) = inf_ecs::ballistics::advance_round(r.at, r.velocity, &r.def, sub_dt);
            let seg = next - prev;
            let len = seg.length();
            // **THE SUPERSONIC CRACK** (wave WPN2c) — the doc section 4's own
            // rule, on the SEGMENT rather than on the point: a 900 m/s round
            // covers fifteen metres in a fixed step, so a point test would miss
            // almost every pass. Latched on the round, so four sub-steps inside
            // four metres of an ear is one noise.
            //
            // It is decided here rather than host-side because it needs the
            // round's own segment, which only exists inside this loop, and
            // because a value two hosts each computed for themselves is a value
            // they can disagree about.
            if !r.cracked {
                if let Some(c) = inf_ecs::ballistics::crack_for(
                    r.shooter,
                    r.def.audio_class(),
                    prev,
                    next,
                    r.velocity.length(),
                    ear,
                ) {
                    report.cracks.push(c);
                    r.cracked = true;
                }
            }
            report.rounds.rays += 1;
            if len > 1e-6 {
                // **Segment 0 only.** A round leaves a hand's breadth from the
                // body that fired it, so its first segment must not stop on its
                // own shooter — nor on the chassis the shooter is sitting in,
                // which is what `shot_exclusions` adds and what only means
                // something now the cast can SEE a dynamic body. After that the
                // exclusion is dropped, because a round that came back at its
                // shooter should hit them.
                let exclude = if r.first_segment {
                    shot_exclusions(world, bridge, r.shooter)
                } else {
                    BTreeSet::new()
                };
                let hit = bridge.world_mut().cast_ray_where(
                    prev,
                    seg / len,
                    len,
                    &exclude,
                    super::CastTargets::AllSolid,
                );
                if let Some(h) = hit {
                    let point = prev + (seg / len) * h.toi;
                    // **A THROWN BODY BOUNCES** (wave WPN2d) instead of ending
                    // here — the doc section 5's *"bounce elasticity"*. The
                    // normal half of its velocity is reflected and scaled by
                    // `restitution`, the tangent half is scrubbed by
                    // `bounce_friction`, and it carries on from the contact
                    // point pushed a hair off the surface so the next segment
                    // does not start inside what it just hit.
                    //
                    // It settles when it has bounced `MAX_BOUNCES` times or is
                    // slower than `SETTLE_SPEED_MPS`, which is a body rolling
                    // rather than bouncing — and this engine has no rolling.
                    // **A GRENADE THAT HAS STOPPED ROLLING WAITS FOR ITS
                    // FUSE.** A thrown body out of bounces is not a body that
                    // explodes on its next contact — it is a body lying on the
                    // ground, and what makes it go off is the clock. Measured by
                    // the arm that found it: a G67 with a three-second fuse went
                    // off at 2.45 s, on its fifth contact with a pavement.
                    //
                    // A thrown body with NO fuse (the thrown knife) falls
                    // through to the impact below, which is what a knife does.
                    if r.kind.bounces()
                        && r.bounces >= inf_ecs::ballistics::MAX_BOUNCES
                        && r.def.fuse_s > 0.0
                    {
                        r.at = point + h.normal * BOUNCE_OFFSET_M;
                        r.velocity = DVec3::ZERO;
                        r.travelled_m += h.toi;
                        r.age_s += sub_dt;
                        r.first_segment = false;
                        report.rounds.settled += 1;
                        continue;
                    }
                    if r.kind.bounces() && r.bounces < inf_ecs::ballistics::MAX_BOUNCES {
                        let after = inf_ecs::ballistics::bounce_velocity(
                            r.velocity,
                            h.normal,
                            r.def.restitution,
                            r.def.bounce_friction,
                        );
                        r.bounces += 1;
                        report.rounds.bounces += 1;
                        r.at = point + h.normal * BOUNCE_OFFSET_M;
                        r.velocity = if after.length() < inf_ecs::ballistics::SETTLE_SPEED_MPS {
                            DVec3::ZERO
                        } else {
                            after
                        };
                        r.travelled_m += h.toi;
                        r.age_s += sub_dt;
                        r.first_segment = r.age_s < inf_ecs::ballistics::SHOOTER_CLEARANCE_S;
                        continue;
                    }
                    let flight = r.travelled_m + h.toi;
                    let target = hit_owner(bridge, h.collider);
                    let on_flesh = target.is_some_and(|g| is_flesh(world, g));
                    let headshot = on_flesh
                        && target.is_some_and(|g| {
                            head_hit(world, g, point, &mut report.rounds.heads_without_a_socket)
                        });
                    if headshot {
                        report.rounds.headshots += 1;
                    }
                    landed.push((
                        WeaponHit {
                            shooter: r.shooter,
                            target,
                            from: prev,
                            to: point,
                            energy_j: r.def.damage_at(flight, headshot),
                            on_flesh,
                            // **QUIET.** The bang happened at the muzzle, half a
                            // second ago, and was queued then; a loud impact
                            // would fire a second gunshot clip from wherever the
                            // round landed and panic the street a second time
                            // from the wrong place. The target's own emitter
                            // still sounds the impact — that half of
                            // `fire_weapon_audio` reads `hit.target`, not
                            // `hit.loud`.
                            loud: false,
                            arrived: true,
                            headshot,
                            report_max_m: r.def.report_max_m,
                            report_gain: r.def.report_gain,
                            // **A quiet hit reaches no report layer**, so these
                            // four are carried for completeness rather than
                            // read: `fire_weapon_audio` builds a stack only
                            // under `hit.loud`. The class is still the round's
                            // OWN weapon's, because the day an arrival makes a
                            // noise it will be that weapon's noise.
                            class: r.def.audio_class(),
                            indoors: false,
                            listener_m: f64::INFINITY,
                            shot_index: 0,
                        },
                        flight,
                    ));
                    report.rounds.impacts += 1;
                    // **THE BLAST**, at the point the body arrived (wave WPN2d).
                    // Recorded here and spent below the loop, because
                    // `apply_blast` needs `&mut EcsWorld` and the pool's own
                    // borrow is live inside it.
                    if r.def.has_blast() {
                        blasts.push((r.shooter, point, r.def));
                    }
                    alive = false;
                    break;
                }
            }
            r.at = next;
            r.velocity = v;
            r.travelled_m += len;
            r.age_s += sub_dt;
            // **Still clearing the thing that launched it?** A TIME and not a
            // sub-step count since wave WPN2d — see `Round::first_segment` for
            // the grenade that detonated on its own thrower.
            r.first_segment = r.age_s < inf_ecs::ballistics::SHOOTER_CLEARANCE_S;
            if r.travelled_m >= r.def.range_m.clamp(0.1, SHOT_MAX_RANGE_M)
                || r.age_s >= MAX_ROUND_LIFETIME_S
                || !r.at.is_finite()
            {
                report.rounds.expired += 1;
                alive = false;
                break;
            }
            if band.tier(r.at, DVec3::ZERO, glam::DQuat::IDENTITY) == inf_math::Tier::Out {
                // A round outside the active partition is flying through
                // geometry that is not in the physics world, so every segment
                // from here on would report a miss it did not earn.
                report.rounds.left_band += 1;
                alive = false;
                break;
            }
        }
        if alive {
            survivors.push(r);
        }
    }
    {
        let mut pool = world.world_mut().resource_mut::<RoundPool>();
        pool.rounds = survivors;
        pool.impacts += u64::from(report.rounds.impacts);
        pool.expired += u64::from(report.rounds.expired);
        pool.left_band += u64::from(report.rounds.left_band);
        if let Some((_, flight)) = landed.last() {
            pool.last_flight_m = *flight;
        }
        report.rounds.in_flight = pool.rounds.len() as u32;
    }
    for (hit, _) in landed {
        apply_hit(world, &hit, dt, report);
        report.hits.push(hit);
    }
    // **THE BLASTS**, after the direct impacts, so a rocket's direct joules are
    // spent on what it struck before its radius damage reaches the same body —
    // the doc's *"direct + blast split"*, in the order the two halves happen.
    for (shooter, at, def) in blasts {
        apply_blast(world, bridge, shooter, at, &def, dt, report);
    }
}

/// **How far off a surface a bounced body restarts**, metres.
///
/// A millimetre. The contact point is ON the collider, so the next segment would
/// begin inside it and stop at zero distance for ever; pushing off along the
/// contact normal by the smallest distance that is not a rounding error is what
/// makes a bounce a bounce rather than a stall. It is deliberately not the
/// body's own radius, because a thrown body in this engine has no radius: it is
/// a segment, and this is the segment's start.
pub const BOUNCE_OFFSET_M: f64 = 0.001;

/// **How far away an act can be seen**, metres.
///
/// A hundred and twenty — wider than the panic radius, because seeing something
/// and running from it are different distances in the other direction too: the
/// person across the square who did not run is still the person who can describe
/// you. It is the [`weapon::REPORT_MAX_M`] / [`PANIC_RADIUS_M`] pair's third
/// number and it sits between them on purpose.
pub const WITNESS_RADIUS_M: f64 = 120.0;

/// **How many acts one step may record.**
///
/// Four, and it is a cost bound: each act is one `O(agents)` walk plus at most
/// [`inf_ecs::witness::MAX_OBSERVERS`] rays, so this is what keeps a firefight's
/// bookkeeping a constant multiple of a walk rather than a function of how many
/// people are shooting. A step that produced more records the first four; the
/// rest are simply not written, which for a seed nothing reads yet is the honest
/// trade and is stated rather than hidden.
pub const MAX_ACTS_PER_STEP: usize = 4;

/// **How far a gunshot scatters a crowd**, metres.
///
/// Forty-five. Deliberately much smaller than a gunshot's own audible reach
/// ([`weapon::REPORT_MAX_M`], 250 m), because hearing a shot and running from one
/// are different distances: a person three streets away turns their head, and a
/// person on the same corner leaves. The two numbers are related on purpose —
/// see [`weapon::REPORT_MAX_M`]'s own doc — so a designer who widens one can see
/// what it is being compared against.
pub const PANIC_RADIUS_M: f64 = 45.0;

/// **How far a frightened bystander walks**, metres.
///
/// Half again as far as a carjacked driver goes ([`inf_ecs::crowd::FLEE_M`]),
/// because a gunshot is worth more distance than an argument, and short enough
/// that the route is still one straight leg rather than a plan.
pub const PANIC_FLEE_M: f64 = 60.0;

/// **How many distinct places one step may panic a crowd from.**
///
/// The pass is `O(agents × sources)` and this is what makes the second factor a
/// constant rather than a firefight's shooter count: shots inside half a panic
/// radius of each other are one source. Eight is more than a street fight has
/// distinct corners, and it bounds the walk at 8 × 1 000 agents = 8 000 squared
/// distances a step against `NPC_STEP_BUDGET_MS`'s own 1 000-agent figure.
///
/// **Past the cap a shot is DROPPED, not folded** (the WPN1 audit's correction —
/// this said "folded into the nearest", which is what the *coalescing* does and
/// is not what the cap does). A ninth distinct place, more than half a radius
/// from all eight, frightens nobody on that step. That is the right trade for a
/// bound — the ninth corner of a firefight is not where the interesting people
/// are standing — but it is a refusal and it is said as one.
pub const MAX_PANIC_SOURCES: usize = 8;

/// **The distinct places this step's gunfire came from**, coalesced and capped.
///
/// The half of [`step_panic`] that decides its cost, hoisted so a gate can
/// measure it without a crowd: shots inside half a [`PANIC_RADIUS_M`] of each
/// other frighten the same people and are one source, and past
/// [`MAX_PANIC_SOURCES`] the rest of the step's shots are **dropped** — which is
/// what keeps the pass's inner loop a constant rather than a function of how
/// many people are shooting, and which is a refusal rather than a fold (see
/// [`MAX_PANIC_SOURCES`]).
///
/// A **brawl is not a source**: see [`WeaponHit::loud`] and the reference frames
/// it cites.
fn panic_sources(hits: &[WeaponHit]) -> Vec<DVec3> {
    let mut sources: Vec<DVec3> = Vec::new();
    for hit in hits.iter().filter(|h| h.loud && h.from.is_finite()) {
        if sources.len() >= MAX_PANIC_SOURCES {
            break;
        }
        if sources
            .iter()
            .any(|s| (*s - hit.from).length() < PANIC_RADIUS_M * 0.5)
        {
            continue;
        }
        sources.push(hit.from);
    }
    sources
}

/// **How many distinct places this step's gunfire came from** — `panic_sources`
/// counted, for a gate that wants the cost bound without a population.
///
/// (A code span rather than a link: the function it names is private, and a
/// public doc that links to one is a rustdoc warning for a reference no reader
/// of the public API could follow anyway.)
pub fn panic_sources_for(hits: &[WeaponHit]) -> usize {
    panic_sources(hits).len()
}

/// What one `step_panic` did — a code span for `panic_sources_for`'s reason:
/// the pass is private and this type is not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PanicReport {
    /// Distinct places this step's gunfire came from, after coalescing.
    pub sources: usize,
    /// Agents the pass looked at — the whole population, which is the cost.
    pub considered: usize,
    /// Agents that started running on this step. `considered - fled` is mostly
    /// distance, and it is the number that says the radius is doing work.
    pub fled: usize,
    /// **Responders inside the radius that the flee door refused** (wave EMS2) —
    /// the engagement counter behind *an officer under fire does not rout*.
    ///
    /// The rule itself lives at [`inf_ecs::crowd::flee_from`], where it holds for
    /// every caller; this is the pass reading
    /// [`inf_ecs::dispatch::is_responder`] back for the people the door said no
    /// to, so a gate can tell **the officers held** from **no officer was ever in
    /// the radius**. A zero here on a scene with police at it is the second
    /// thing, and a gate that could not tell them apart would pass on a
    /// dispatcher that never sent anybody.
    pub exempt: usize,
}

/// **A GUNSHOT SCATTERS THE STREET** (wave WPN1) — the crowd's panic, through
/// the one flee door.
///
/// # The cost, stated, because it is the whole design
///
/// The obvious shape is "for each shot, for each agent, how far apart are they",
/// which is `O(shots × agents)` — and at 600 rpm with several shooters that is a
/// per-step cost that grows with how exciting the scene is, measured against an
/// `inf_player::budget::NPC_STEP_BUDGET_MS` that was set at a thousand *walking*
/// agents. (Named in a code span rather than linked: `inf-physics` does not
/// depend on `inf-player`, so an intra-doc link there resolves to nothing and
/// costs a rustdoc warning for a reference a reader can follow by name.)
///
/// So this is **one walk over the population** with a bounded inner loop: the
/// step's loud shots are coalesced into at most [`MAX_PANIC_SOURCES`] places
/// first (two shots inside half a radius of each other frighten the same people),
/// and the agents are read off `CrowdPopulationRes`'s own records — the
/// `blocked_agents` shape, `O(agents)` and allocation-free on a level with no
/// crowd, rather than `O(entities)` over a furnished town.
///
/// # The one-step latency, stated
///
/// The crowd steers in phase 5 (`crowd`) and this runs in phase 14 (`gameplay`),
/// so an agent frightened here starts moving on the **next** fixed step: 16.7 ms
/// at 60 Hz. It is the same lag in both hosts — the pass is one Ring-0 rule they
/// both call from the same slot — so PIE == shipping is unaffected and no trace
/// can see it. Running the panic before the crowd would fix it and would put the
/// gunfire pass before the shots that feed it, which is a full step of lag the
/// other way. The sentence is `muzzle_of`'s own, one system along.
///
/// Inert on a level with no gunfire and on a level with no population: two early
/// returns and no allocation.
fn step_panic(world: &mut EcsWorld, hits: &[WeaponHit], dt: f64) -> PanicReport {
    let mut report = PanicReport::default();
    // 1. The sources, coalesced.
    let sources = panic_sources(hits);
    if sources.is_empty() {
        return report;
    }
    report.sources = sources.len();
    // 2. ONE walk over the population, reading where each agent stood at the end
    //    of the crowd's own step.
    let mut want: Vec<(Uuid, DVec3, DVec3)> = Vec::new();
    {
        let Some(pop) = world
            .world()
            .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        else {
            return report;
        };
        report.considered = pop.records.len();
        for (guid, rec) in &pop.records {
            let here = rec.last;
            let mut best: Option<(f64, DVec3)> = None;
            for s in &sources {
                let d = (*s - here).length();
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, *s));
                }
            }
            let Some((d, at)) = best else { continue };
            if d > PANIC_RADIUS_M {
                continue;
            }
            want.push((*guid, here, at));
        }
    }
    // 3. …and the flee itself, through the one door, in `Guid` order (the
    //    `BTreeMap` walk above), so two hosts scatter the same people.
    for (guid, here, away_from) in want {
        if inf_ecs::crowd::flee_from(world, guid, here, away_from, dt, PANIC_FLEE_M) {
            report.fled += 1;
        } else if inf_ecs::dispatch::is_responder(world, guid) {
            // **The exemption, counted where it was applied** (wave EMS2). The
            // door above is where the rule lives — it refuses a responder for
            // every caller, not only this one — and this reads the named
            // predicate back for the people it said no to. A door whose guard
            // was mutated away answers `true` here instead, so `fled` rises and
            // this stays at zero: the counter cannot agree with a broken rule.
            report.exempt += 1;
        }
    }
    report
}

/// **WHO SAW IT** (wave WPN1) — the witnessed-act seed for the EMS arc.
///
/// One record per act, at most [`MAX_ACTS_PER_STEP`] a step, each carrying the
/// nearest few crowd agents that have a clear line to it. Nothing reads it yet;
/// see [`inf_ecs::witness`] for why it is written now anyway.
///
/// # The line of sight, and what it costs
///
/// One `cast_ray_excluding` per candidate observer, from the observer's own eye
/// to the act — which is the same query the audio occlusion path makes and the
/// same one `resolve_shot` makes, so it is the engine's existing answer to "is
/// there something between these two points" rather than a fourth. Bounded at
/// `MAX_ACTS_PER_STEP × MAX_OBSERVERS` = 32 rays a step in the worst case, and
/// **zero** on every step nothing happened on.
///
/// A `Dormant` observer has no collider to exclude and the ray simply runs
/// without one, which is right: an agent with no body cannot be in its own way.
///
/// Returns how many acts were recorded — an engagement counter, because "the
/// pass ran" and "somebody saw something" are different facts.
fn step_witness(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    hits: &[WeaponHit],
    killed: &[Uuid],
    step: u64,
) -> u32 {
    use inf_ecs::witness::{ActKind, WitnessedAct};
    // The acts, in the order they happened: a death outranks a shot, because a
    // dispatcher asked to name one thing about a street should name the body.
    // `(kind, actor, at, subject)` — and **`subject` is whose body the act's
    // position is measured on**, which is not always the actor and is what the
    // sight ray has to be allowed through (wave EMS3 audit). A death and a punch
    // are both recorded AT the victim's chest, which is a point *inside* the
    // victim's own capsule: without excluding it every ray to one of them stops
    // a capsule-radius short of its target and the act is witnessed by nobody.
    // `Uuid::nil()` when the position is not on a body.
    let mut acts: Vec<(ActKind, Uuid, DVec3, Uuid)> = Vec::new();
    for guid in killed {
        if acts.len() >= MAX_ACTS_PER_STEP {
            break;
        }
        let Some(at) = strike_point(world, *guid) else {
            continue;
        };
        // **A KILLING IS FILED AGAINST THE KILLER** (wave EMS3 audit).
        //
        // `killed` is the list of bodies that stopped working, so the guid in
        // hand is the **victim** — and `WitnessedAct::actor` is *who did it*.
        // Nothing read the field for two waves; EMS3 made two things read it,
        // and both read it as the culprit: `crime::report_act` opens a profile
        // keyed on it and `d3::dispatch::open_incidents` files the search under
        // it. Recording the dead man here opened a **three-heat file on the
        // person who was murdered**, wearing the murdered man's description — a
        // `MultiUnit` response sent after a corpse — and left the killer on the
        // two heat their `Shot` earned.
        //
        // The blow that did it is in **this same step's** hits, because
        // `weapon::damage_entity` is the only thing that sets `Health::dead` and
        // `apply_hit` is its only per-step caller. A death nobody in this step's
        // hits accounts for — the two hosts' scripted `damage_entity` door is
        // the one that can produce it — is recorded with a **nil** actor: the
        // call still opens a crime scene where the body is, and
        // `crime::report_act` refuses to put it on anybody's file.
        let killer = hits
            .iter()
            .find(|h| h.target == Some(*guid))
            .map(|h| h.shooter)
            .unwrap_or_else(Uuid::nil);
        acts.push((ActKind::Killed, killer, at, *guid));
    }
    for hit in hits.iter().filter(|h| h.loud && h.from.is_finite()) {
        if acts.len() >= MAX_ACTS_PER_STEP {
            break;
        }
        acts.push((ActKind::Shot, hit.shooter, hit.from, hit.shooter));
    }
    // **THE QUIET CRIME** (wave EMS3) — a swing or a kick that landed on
    // somebody. `loud` is false for a fist by WPN1's own definition, so an
    // assault is exactly the attack the gunshot filter above drops, and
    // `on_flesh` is what tells hitting a person from hitting a wall. Nobody flees
    // from it (that is `step_panic`'s loud-only rule, unchanged) — the only
    // people who know about a punch are the ones who saw it, which is what makes
    // the observer list the whole of the evidence.
    // **…and NOT a round arriving** (wave WPN2a). A projectile is two records —
    // the pull, loud, at the muzzle, and the arrival, quiet, at the target, up
    // to eight seconds later — and without `arrived` the second one lands in
    // this bucket and files an ASSAULT against a shooter whose `Shot` is
    // already on the record. A bullet reaching somebody is not a beating; it is
    // the shot that was already witnessed finally arriving.
    for hit in hits
        .iter()
        .filter(|h| !h.loud && h.on_flesh && !h.arrived && h.from.is_finite())
    {
        if acts.len() >= MAX_ACTS_PER_STEP {
            break;
        }
        acts.push((
            ActKind::Assault,
            hit.shooter,
            hit.to,
            hit.target.unwrap_or_else(Uuid::nil),
        ));
    }
    // **…and everything a phase earlier in this step raised** (wave EMS3) — the
    // carjack, applied in `character move` where there is no collision world to
    // ask about sight lines. Drained unconditionally, BEFORE the emptiness check
    // below, so a queue can never survive a step it was raised on: an act the
    // ray budget refuses is an act nobody saw, which is the honest outcome and
    // not a backlog.
    for (kind, actor, at) in inf_ecs::witness::take_raised(world) {
        if acts.len() >= MAX_ACTS_PER_STEP {
            break;
        }
        // A raised act names a PLACE and not a body — `carjack::door_point` is
        // beside a car rather than inside anybody — so there is no third
        // collider to let the ray through.
        acts.push((kind, actor, at, Uuid::nil()));
    }
    if acts.is_empty() {
        return 0;
    }
    let mut recorded = 0u32;
    for (kind, actor, at, subject) in acts {
        let candidates = inf_ecs::witness::candidates_near(world, at, WITNESS_RADIUS_M);
        let mut observers: Vec<Uuid> = Vec::new();
        for (guid, feet) in candidates {
            // **Nobody who died this step saw anything** (wave EMS3 audit). The
            // actor exclusion used to cover the victim for free, because a death
            // named the dead man as its own actor; now that it names the killer,
            // a crowd agent shot in an empty alley would otherwise be the sole
            // witness to its own murder — and one observer is all
            // `crime::report_act` needs to open a file.
            if guid == actor || killed.contains(&guid) {
                continue;
            }
            let eye = feet + DVec3::Y * MUZZLE_HEIGHT_M;
            let to = at - eye;
            let d = to.length();
            if !d.is_finite() {
                continue;
            }
            if d > 1.0e-6 {
                let mut exclude = BTreeSet::new();
                if let Some(c) = bridge.collider_of(guid) {
                    exclude.insert(c);
                }
                if let Some(c) = bridge.collider_of(actor) {
                    exclude.insert(c);
                }
                // …and the body the act is ON — see the `acts` declaration.
                if !subject.is_nil() {
                    if let Some(c) = bridge.collider_of(subject) {
                        exclude.insert(c);
                    }
                }
                // Anything in the way at all: the ray is stopped short of the
                // act, so the observer cannot see it. The tolerance is a
                // centimetre, which is the wall a shot leaves through.
                if bridge
                    .world_mut()
                    .cast_ray_excluding(eye, to / d, d, &exclude)
                    .is_some_and(|h| h.toi < d - 0.01)
                {
                    continue;
                }
            }
            observers.push(guid);
        }
        inf_ecs::witness::record_act(
            world,
            WitnessedAct {
                kind,
                actor,
                at,
                step,
                observers,
                actor_look: inf_ecs::witness::look_digest(world, actor),
                actor_vehicle: inf_ecs::witness::actor_vehicle(world, actor),
            },
        );
        recorded += 1;
    }
    recorded
}

/// **Where a swing lands on a body**, world metres — the point a punch is aimed
/// at and measured to.
///
/// [`MUZZLE_HEIGHT_M`] above the feet, which is the same height a rig-less
/// character's own shot leaves from. Both ends of a swing are therefore measured
/// at chest height, so the vertical term cancels between two characters of the
/// same size and a punch at a metre is a punch at a metre rather than
/// `sqrt(1² + 1.4²)`.
fn strike_point(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    Some(feet_of(world, guid)? + DVec3::Y * MUZZLE_HEIGHT_M)
}

/// **Resolve a SWING** — a reach and an arc, not a ray (wave WPN1).
///
/// # One door, and it is the interaction rule's
///
/// The reach and the cone go through [`inf_ecs::interact::resolve`], which is
/// what the E-key prompt, the door press and `try_kick` already resolve through.
/// Writing the arithmetic a second time here is exactly the defect `try_kick`'s
/// own doc names — *"spelling it a second way would let a player kick a door the
/// prompt says is out of reach"* — one verb along: a punch that could land on
/// somebody the prompt calls unreachable is a punch through a wall.
///
/// It also buys the portable trigonometry for free: the cone test goes through
/// `inf_math::patan2_64` and the boundary epsilon that exists because of it (the
/// P14 law), so a swing lands identically on two machines.
///
/// # THE LINE OF SIGHT (wave WPN2d) — one box cast, and only when there is
/// something to hit
///
/// Wave WPN1 carried *"no line of sight: a body on the far side of a shut door
/// within reach is hit"* by name. This closes it, and the shape of the fix is
/// the whole point: the reach and the cone are STILL resolved by
/// `inf_ecs::interact::resolve` — one door, unchanged, so a punch still cannot
/// land on somebody the E-key prompt calls unreachable — and what this adds is a
/// second question asked only of the body that door already chose. *Is there a
/// wall in the way?*
///
/// The probe is the doc section 5's own *"short box-cast or sphere-cast"*: a
/// [`MELEE_BOX_HALF_M`] box swept from the strike point to the target's own,
/// through `CastTargets::AllSolid` minus everything the shooter is
/// ([`shot_exclusions`]). If the first thing the box meets is not the target,
/// the swing is a MISS and [`GameplayReport::swings_blocked`] counts it.
///
/// A BOX rather than a ray because a fist is not a point: a ray between two
/// capsule axes threads a door frame that an arm does not fit through, and the
/// half-extent is the width of what is swinging.
///
/// It costs **one cast per swing that found a body**, and nothing at all for a
/// swing that found nobody — which is the reason wave WPN1 gave for not
/// spending one per candidate, honoured: the cast is downstream of the
/// resolution, not inside it.
///
/// # What it still does NOT do
///
/// * **No cleave**, and that is a decision rather than an omission (wave WPN2d
///   restates it deliberately). The nearest body in the arc takes the blow and
///   nobody else does, which is `resolve`'s own rule (*"the first of two equals
///   wins"*). A swing that hit everything in its cone needs a `WeaponDef` field
///   to say so, a second resolution that answers a LIST rather than a body, and
///   a cast per body it answers — three changes to buy a verb no weapon in the
///   registry has. The M9 knife is a single-target weapon and says so.
///
/// `O(characters)`, over the same walk [`gunners`] already makes — and only on
/// the steps a swing actually leaves, which at [`weapon::FIST_RPM`] is at most
/// one and a half a second.
#[allow(clippy::too_many_arguments)]
fn resolve_swing(
    world: &EcsWorld,
    bridge: &mut PhysicsBridge3D,
    shooter: Uuid,
    def: &WeaponDef,
    from: DVec3,
    dir: DVec3,
    yaw_deg: f64,
    report: &mut GameplayReport,
) -> WeaponHit {
    use inf_ecs::interact::{InteractCandidate, InteractVerb};
    let reach = def.reach_m();
    let arc = def.melee_arc_deg.clamp(0.0, 360.0);
    let mut candidates: Vec<InteractCandidate> = Vec::new();
    for guid in gunners(world) {
        if guid == shooter {
            continue;
        }
        let Some(position) = strike_point(world, guid) else {
            continue;
        };
        candidates.push(InteractCandidate {
            guid,
            // The verb is not read by anything downstream — this resolution
            // answers "which body" and nothing else — so it carries the neutral
            // one `try_kick` gives a door rather than inventing a fourth.
            verb: InteractVerb::Use,
            label: String::new(),
            position,
            range_m: reach,
            view_cone_deg: arc,
            grip: None,
        });
    }
    // `gunners` is already `Guid`-ordered, so ties break on the guid — two
    // bodies at exactly one distance answer the same one on both hosts.
    // **THE LINE OF SIGHT** (wave WPN2d). Only for the body the shared door
    // already chose, and only when it chose one.
    let chosen = inf_ecs::interact::resolve(&candidates, from, yaw_deg).filter(|hit| {
        report.melee_casts += 1;
        let to = hit.position - from;
        let span = to.length();
        if span <= 1e-6 {
            return true;
        }
        let exclude = shot_exclusions(world, bridge, shooter);
        let blocker = bridge.world_mut().cast_shape_where(
            &super::ColliderShape3D::Box {
                half_extents: DVec3::splat(MELEE_BOX_HALF_M),
            },
            from,
            glam::DQuat::IDENTITY,
            to / span,
            span,
            &exclude,
            super::CastTargets::AllSolid,
        );
        match blocker {
            // The box starts inside something the shooter is not — a swing from
            // inside a wall — which is not a hit on the target and is not a
            // reason to refuse one either: `started_penetrating`'s own doc says
            // the witness point is unreliable, so the honest answer is to let
            // the reach-and-cone door's verdict stand.
            Some(h) if h.started_penetrating => true,
            Some(h) => {
                let who = hit_owner(bridge, h.collider);
                let clear = who == Some(hit.guid);
                if !clear {
                    report.swings_blocked += 1;
                }
                clear
            }
            None => true,
        }
    });
    match chosen {
        Some(hit) => WeaponHit {
            shooter,
            target: Some(hit.guid),
            from,
            to: hit.position,
            energy_j: def.damage_j,
            // Every candidate here IS a character, which is what `is_flesh`
            // answers `true` for — so this is a fact rather than an assumption.
            on_flesh: true,
            loud: false,
            arrived: false,
            // A swing has no hit POINT on the target — `interact::resolve`
            // answers a body and a position on its capsule axis, not a place a
            // ray arrived at — so there is nothing to test against a head
            // sphere and a punch never multiplies. Wave WPN2d's box cast is
            // where a melee hit grows a point.
            headshot: false,
            report_max_m: def.report_max_m,
            report_gain: def.report_gain,
            // A swing is never loud (see `loud` above), so no report layer ever
            // reads these. A fist names no class and its band answers `Pistol`
            // on a zero-length barrel, which is as meaningless as it is
            // harmless: nothing plays it.
            class: def.audio_class(),
            indoors: false,
            listener_m: f64::INFINITY,
            shot_index: 0,
        },
        None => WeaponHit {
            shooter,
            target: None,
            from,
            // A miss ends at the end of the reach along the aim, so a tracer and
            // a debug line draw the swing rather than nothing.
            to: from + dir * reach,
            energy_j: def.damage_j,
            on_flesh: false,
            loud: false,
            arrived: false,
            headshot: false,
            report_max_m: def.report_max_m,
            report_gain: def.report_gain,
            class: def.audio_class(),
            indoors: false,
            listener_m: f64::INFINITY,
            shot_index: 0,
        },
    }
}

/// **SPEND A BLAST** (wave WPN2d) — the doc section 5's radius damage, through
/// the one door every other joule in this engine leaves by.
///
/// # It is a sweep over BODIES and DESTRUCTIBLES, and it costs one ray each
///
/// Every character (`gunners`) and every entity carrying `Destructible` within
/// `blast_radius_m` of `at` is a candidate, in `Guid` order. Each one gets:
///
/// 1. a **falloff** — `inf_ecs::ballistics::blast_damage_j`, which is
///    `(1 − d/r)²` of `blast_damage_j`, exact at both ends;
/// 2. a **line of sight** — one `cast_ray_where` from the epicentre toward the
///    candidate through `CastTargets::AllSolid`. If the first thing the ray
///    meets is not the candidate, the wall took the blast and the body did not.
///    That is `resolve_swing`'s own rule one verb along, and it is what stops a
///    grenade in a stairwell killing everybody in the building;
/// 3. the joules, through **`apply_hit`** — the same door a bullet spends
///    through, so a blast kill staggers, panics the street, is witnessed and
///    reaches the P22 destructible door exactly as a rifle round does.
///
/// Bounded by [`MAX_BLAST_TARGETS`], which is a COST bound: the candidates are
/// already inside the radius, so what it refuses is the thirty-third body in a
/// crowd around one rocket, and it is counted.
///
/// # The direct hit is NOT here
///
/// A rocket that struck a wall has already spent `damage_j` on that wall through
/// `apply_hit`, in the caller — the doc's *"direct + blast split"*. This is the
/// second half, and it never double-spends on the thing that was struck: the
/// impact point is at distance zero from itself only if the struck entity is
/// also a blast candidate, and it is, deliberately, because a body a rocket hits
/// squarely should take both. `WeaponHit::energy_j` and the blast's own joules
/// are two different quantities from two different fields.
fn apply_blast(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    shooter: Uuid,
    at: DVec3,
    def: &WeaponDef,
    dt: f64,
    report: &mut GameplayReport,
) {
    if !def.has_blast() || !at.is_finite() {
        return;
    }
    let radius = def.blast_radius_m;
    // The candidates, in `Guid` order so two hosts spend the joules in one
    // order: every character, then every destructible.
    let mut candidates: Vec<(Uuid, DVec3)> = Vec::new();
    for guid in gunners(world) {
        if let Some(p) = target_point(world, guid) {
            if (p - at).length() <= radius {
                candidates.push((guid, p));
            }
        }
    }
    candidates.extend(
        weapon::destructible_positions(world)
            .into_iter()
            .filter(|(_, p)| (*p - at).length() <= radius),
    );
    candidates.sort_by_key(|(g, _)| *g);
    candidates.dedup_by_key(|(g, _)| *g);
    let mut hurt = 0u32;
    for (target, point) in candidates.into_iter().take(MAX_BLAST_TARGETS) {
        let span = point - at;
        let distance = span.length();
        let joules = inf_ecs::ballistics::blast_damage_j(def, distance);
        if joules <= 0.0 {
            continue;
        }
        // The line of sight. A candidate a hand's breadth from the epicentre is
        // exposed by construction; anything further is asked.
        if distance > 1e-3 {
            report.rounds.blast_rays += 1;
            let exclude = std::collections::BTreeSet::new();
            let seen = bridge.world_mut().cast_ray_where(
                at,
                span / distance,
                distance,
                &exclude,
                super::CastTargets::AllSolid,
            );
            if let Some(h) = seen {
                if hit_owner(bridge, h.collider) != Some(target) {
                    report.rounds.blast_shadowed += 1;
                    continue;
                }
            }
        }
        let on_flesh = is_flesh(world, target);
        let hit = WeaponHit {
            shooter,
            target: Some(target),
            from: at,
            to: point,
            energy_j: joules,
            on_flesh,
            // **QUIET.** The bang is the `BlastEvent` below, played once by both
            // hosts; a loud hit per body would be one gunshot clip per person in
            // the radius, from the wrong place, panicking the street N times.
            loud: false,
            arrived: true,
            headshot: false,
            report_max_m: def.report_max_m,
            report_gain: def.report_gain,
            class: def.audio_class(),
            indoors: false,
            listener_m: f64::INFINITY,
            shot_index: 0,
        };
        apply_hit(world, &hit, dt, report);
        report.hits.push(hit);
        hurt += 1;
    }
    report.blasts.push(BlastEvent {
        shooter,
        at,
        radius_m: radius,
        hurt,
    });
}

/// **Spend a blast, for a gate** (wave WPN2d) — [`apply_blast`], by a public
/// name.
///
/// The pass itself runs inside `step_rounds`, where the pool's borrow is live,
/// so a gate that wanted to measure a blast's falloff would otherwise have to
/// fly a rocket at a wall and infer the epicentre from where it stopped. This
/// door lets it name the point and read the joules — which is the difference
/// between a gate that measures the CURVE and one that measures a landing.
///
/// It is the same function, not a copy: there is exactly one blast sweep in this
/// engine and this is a `pub` alias for it.
pub fn blast_for_test(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    shooter: Uuid,
    at: DVec3,
    def: &WeaponDef,
    dt: f64,
    report: &mut GameplayReport,
) {
    apply_blast(world, bridge, shooter, at, def, dt, report);
}

/// **HOLD A LOCK** (wave WPN2d) — the launcher's target selection, once per
/// armed character per step.
///
/// # It is SIM STATE, and it lives on the weapon
///
/// `WeaponState::lock_target` and `lock_held_s` are what a lock IS: the target,
/// and how long it has been inside the cone. Both are folded into
/// `weapon_state_bytes` (only while a lock is being held), because a lock
/// decides whether the round that leaves is guided — so two hosts that
/// disagreed about it would fire two different missiles.
///
/// # The rule
///
/// A weapon that [`WeaponDef::can_lock`] scans for the nearest entity carrying
/// `inf_ecs::components::Vehicle` inside `lock_cone_deg` of the aim line and
/// inside `range_m`. `lock_air_only` (the Stinger's rule) refuses anything that
/// is not airborne — measured as `MIN_AIRBORNE_M` above the shooter's own feet,
/// because this engine has no "is flying" flag and altitude is the honest
/// question. The hold accumulates while the SAME target stays in the cone and
/// **resets the instant it leaves**, which is the arm a gate mutates.
fn step_locks(
    world: &mut EcsWorld,
    bridge: &PhysicsBridge3D,
    dt: f64,
    report: &mut GameplayReport,
) {
    use inf_ecs::components::{CharacterMovement, GlobalTransform};
    // **`PhysicsBridge3D::vehicle_guids` is the door**, and it is the same one
    // the E-key prompt asks (`d3::interact::vehicle_candidates`): what counts as
    // a vehicle in this engine is what the bridge built a rig for, and a second
    // spelling of it here would be a lock that could acquire something the game
    // does not think is a car. Empty on every level with no vehicles, which is
    // most of them, and the whole pass then costs one allocation.
    let mut targets: Vec<(Uuid, DVec3)> = bridge
        .vehicle_guids()
        .into_iter()
        .filter_map(|chassis| {
            let e = world.entity_of(chassis)?;
            let t = world.world().get::<GlobalTransform>(e)?;
            Some((chassis, t.translation()))
        })
        .collect();
    targets.sort_by_key(|(g, _)| *g);
    if targets.is_empty() {
        // Nothing to lock onto. A launcher already holding a lock keeps it for
        // this step rather than being cleared by an empty world -- which cannot
        // happen, because a target that despawned is a target the sweep below
        // would not find either. Stated so the early return is a decision.
        return;
    }
    for guid in gunners(world) {
        let Some(entity) = world.entity_of(guid) else {
            continue;
        };
        let Some((_, def)) = weapon::equipped_def(world, guid) else {
            continue;
        };
        if !def.can_lock() {
            // A weapon that cannot lock must not leave a stale one behind: the
            // bytes are folded whenever the target is non-nil, so a launcher
            // put away with a lock on it would go on folding it for ever.
            if let Some(mut st) = world.world_mut().get_mut::<weapon::WeaponState>(entity) {
                if !st.lock_target.is_nil() || st.lock_held_s != 0.0 {
                    st.lock_target = Uuid::nil();
                    st.lock_held_s = 0.0;
                }
            }
            continue;
        }
        let (yaw, pitch) = {
            let w = world.world();
            match w.get::<CharacterMovement>(entity) {
                Some(cm) => (cm.runtime.aim_yaw_deg, cm.runtime.aim_pitch_deg),
                None => continue,
            }
        };
        let Some(feet) = feet_of(world, guid) else {
            continue;
        };
        let eye = feet + DVec3::Y * MUZZLE_HEIGHT_M;
        let aim = weapon::aim_forward(yaw, pitch);
        let half = (def.lock_cone_deg * 0.5).clamp(0.0, 180.0);
        let reach = def.reach_m();
        // The nearest thing in the cone. `pacos64` for the P14 reason: this
        // answer reaches `weapon_state_bytes`.
        let mut best: Option<(Uuid, f64)> = None;
        for (target, at) in &targets {
            let span = *at - eye;
            let distance = span.length();
            if distance <= 1e-6 || distance > reach {
                continue;
            }
            if def.lock_air_only && at.y - feet.y < MIN_AIRBORNE_M {
                continue;
            }
            let cos = (span / distance).dot(aim).clamp(-1.0, 1.0);
            let off = inf_math::pacos64(cos).to_degrees();
            if off > half {
                continue;
            }
            if best.is_none_or(|(_, d)| distance < d) {
                best = Some((*target, distance));
            }
        }
        let Some(mut st) = world.world_mut().get_mut::<weapon::WeaponState>(entity) else {
            continue;
        };
        match best {
            Some((target, _)) => {
                if st.lock_target == target {
                    st.lock_held_s = (st.lock_held_s + dt).min(def.lock_s);
                } else {
                    st.lock_target = target;
                    st.lock_held_s = 0.0;
                }
                report.locks_held += 1;
                if st.lock_held_s >= def.lock_s {
                    report.locks_complete += 1;
                }
            }
            None => {
                // **It releases outside the cone**, and it releases to NOTHING
                // rather than decaying: a lock is a fact about what the player
                // is pointing at, and a half-remembered one would fire a
                // missile at a car that has driven behind a building.
                st.lock_target = Uuid::nil();
                st.lock_held_s = 0.0;
            }
        }
    }
}

/// **How high above its own feet a thing has to be for a Stinger to see it**,
/// metres.
///
/// Six. This engine has no "is airborne" flag — `Vehicle` is a car, a boat and a
/// helicopter — and altitude is the honest question a shoulder-fired
/// anti-aircraft launcher asks. Six metres is above a lorry and below any
/// helicopter that is flying rather than parked.
pub const MIN_AIRBORNE_M: f64 = 6.0;

/// **THROW SOMETHING** (wave WPN2d) — the throw verb, the clip's own notify, and
/// the body that leaves the hand.
///
/// # Two steps, and the notify between them
///
/// 1. **The press.** A character holding a `throwable` weapon whose throw key
///    went down starts the CHAR1b.2 additive through
///    `inf_ecs::anim_bridge::start_throw` — overhand or underhand by the aim
///    pitch, because a grenade lobbed at a roof and one rolled under a car are
///    two different animations and the player has already said which by where
///    they are pointing.
/// 2. **The release**, on the clip's own notify (`weapon::THROW_NOTIFY`, fired
///    by the pose step when the additive crosses
///    `inf_anim::THROW_RELEASE_FRAC`). The pose step runs at `STEP_PHASES` 21
///    and this at 15, so the notify is consumed on the step AFTER it fires —
///    ONE fixed step, 16.7 ms, which is the same one-step latency `muzzle_of`
///    states and is inside the two-frame budget the brief asks for.
///
/// The body leaves from the hand's own socket when the rig publishes one and
/// from the muzzle rule when it does not — `muzzle_of`'s two answers, reused.
///
/// A throw spends a round from the magazine through `weapon::try_fire`, so a
/// character with no grenades left throws nothing and the readout says why.
fn step_throws(world: &mut EcsWorld, report: &mut GameplayReport) {
    use inf_ecs::components::CharacterMovement;
    for guid in gunners(world) {
        let Some(entity) = world.entity_of(guid) else {
            continue;
        };
        // **The edge is TAKEN whether or not it is honoured** — `step_weapons`'
        // own law, for its own reason: a press made with a rifle in hand must
        // not survive into the step a grenade is equipped.
        let pressed = {
            let w = world.world_mut();
            match w.get_mut::<CharacterMovement>(entity) {
                Some(mut cm) => {
                    let out = cm.runtime.press_throw;
                    cm.runtime.press_throw = false;
                    out
                }
                None => false,
            }
        };
        let Some((item_id, def)) = weapon::equipped_def(world, guid) else {
            continue;
        };
        if !def.throwable {
            continue;
        }
        if pressed {
            let (pitch, throwing) = {
                let w = world.world();
                match w.get::<CharacterMovement>(entity) {
                    Some(cm) => (cm.runtime.aim_pitch_deg, cm.runtime.throw_s > 0.0),
                    None => (0.0, false),
                }
            };
            // One throw at a time: a second press mid-animation is a press the
            // arm cannot honour.
            if !throwing {
                // The magazine is spent HERE, on the press, rather than at the
                // release — a pin pulled is a grenade gone, and a character who
                // died mid-throw has still used one.
                let spent = {
                    let w = world.world_mut();
                    match w.get_mut::<weapon::WeaponState>(entity) {
                        Some(mut st) if st.item_id == item_id => {
                            weapon::try_fire(&def, &mut st, true) == weapon::FireVerdict::Fired
                        }
                        _ => false,
                    }
                };
                if spent {
                    // **Overhand above the horizon, underhand below it.** The
                    // split is at −10° rather than at 0 because a flat throw is
                    // an overhand one: an underhand lob is something you do at
                    // your own feet.
                    let overhand = pitch > UNDERHAND_PITCH_DEG;
                    let seconds = if overhand {
                        inf_anim::THROW_OVER_S
                    } else {
                        inf_anim::THROW_UNDER_S
                    };
                    inf_ecs::anim_bridge::start_throw(world, guid, overhand, seconds);
                    inf_ecs::anim_bridge::set_anim_trigger(world, guid, weapon::THROW_TRIGGER);
                }
            }
        }
        // **THE RELEASE.** Two paths, both armed, exactly as a reload's are
        // (`WeaponState::reload_left_s`: *"the ceiling, not the schedule"*).
        //
        // 1. **The clip's own notify** — fired by the pose step when the throw
        //    additive crosses `inf_anim::THROW_RELEASE_FRAC`, consumed here
        //    exactly once. This is the authority whenever there IS a clip.
        // 2. **The character's own clock**, for a character with no rig — every
        //    headless run, every crowd agent the sim has tiered out of posing,
        //    and every level committed before CHAR1b.2 imported the additives.
        //    A throw that only released on a notify would be a verb that did
        //    nothing at all on those, which is the reader-that-lies this door's
        //    own `start_throw` doc refused to ship.
        //
        // The clock is the CEILING and not the schedule: it fires
        // `THROW_RELEASE_GRACE_S` AFTER the fraction the notify fires at, so on
        // a rigged character the notify always gets there first and the clock
        // never runs. That is what keeps path 1 from being decoration.
        let notified = inf_ecs::anim_bridge::consume_anim_notify(world, guid, weapon::THROW_NOTIFY);
        let by_clock = !notified && {
            let w = world.world();
            w.get::<CharacterMovement>(entity).is_some_and(|cm| {
                let total = cm.runtime.throw_total_s;
                total > 0.0
                    && cm.runtime.throw_s > 0.0
                    && (total - cm.runtime.throw_s)
                        >= total * inf_anim::THROW_RELEASE_FRAC + THROW_RELEASE_GRACE_S
            })
        };
        if !notified && !by_clock {
            continue;
        }
        let Some((from, yaw, pitch, _)) = muzzle_of(world, guid) else {
            continue;
        };
        let dir = weapon::aim_forward(yaw, pitch);
        let round = inf_ecs::ballistics::Round {
            shooter: guid,
            at: from,
            velocity: dir * def.muzzle_speed_mps.max(0.1),
            travelled_m: 0.0,
            age_s: 0.0,
            first_segment: true,
            cracked: false,
            kind: inf_ecs::ballistics::RoundKind::Thrown,
            fuse_left_s: def.fuse_s,
            bounces: 0,
            guide: Uuid::nil(),
            def,
        };
        if inf_ecs::ballistics::spawn_round(world, round, report.rounds.shot_rays as usize) {
            report.rounds.spawned += 1;
            report.throws += 1;
            // **THE HAND LETS GO.** The weapon entity leaves the world on the
            // next `step_equipped_weapons` when the magazine empties; what has
            // to happen THIS step is that the hand stops holding it, or the IK
            // pass drags a grenade that is already in the air back onto the
            // palm. `set_hand_ik` with no weapon hold is the door.
            inf_ecs::pose::set_hand_ik(world, guid, inf_ecs::pose::HandIk::default());
        } else {
            report.rounds.refused += 1;
        }
        // The clock is stopped either way: an animation that went on playing
        // after the body left would be a hand throwing nothing.
        if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(entity) {
            cm.runtime.throw_s = 0.0;
            cm.runtime.throw_total_s = 0.0;
        }
    }
}

/// **How long after the clip's release point a rig-less throw lets go**,
/// seconds (wave WPN2d).
///
/// Two fixed steps, 33 ms. It is a GRACE and not a delay: the notify path fires
/// one step after the crossing (the pose step runs after the gameplay step), so
/// two steps is the first moment at which "the notify did not come" is a fact
/// rather than a race — and it keeps the whole release inside the two-frame
/// budget either way.
pub const THROW_RELEASE_GRACE_S: f64 = 2.0 / 60.0;

/// **The aim pitch below which a throw goes underhand**, degrees.
///
/// Minus ten. A flat throw is an overhand one — an underhand lob is what you do
/// at your own feet — so the split is a little below the horizon rather than on
/// it.
pub const UNDERHAND_PITCH_DEG: f64 = -10.0;

/// **Is this thing a body?** — the question a round asks about what it hit
/// (wave WPN1).
///
/// # It used to be "does it have a `Health` component", and that was the silent
/// shot
///
/// I6 gave the hero a body from its own Blueprint and gave **nothing else** one.
/// So a round into an NPC, a crowd agent or any other character was `on_flesh ==
/// false`, went to the destructible branch, and was owed to an entity with no
/// `Destructible` — where the host logged a `NoDestructible` refusal, once per
/// round, ten times a second on a held trigger. The person was unhurt, the log
/// was full, and the only visible symptom was the flood.
///
/// So the question is now **"is it a character"**, and [`apply_hit`] gives it a
/// body on the first hit that lands. `CharacterMovement` is the one component in
/// this engine that means *a person*: the crowd puts it on every materialized
/// agent, the movement step visits exactly the entities that carry it, and
/// nothing else in the tree has one.
///
/// A body that already has [`Health`] still answers `true` through the first arm,
/// so a level that authored one — the gameplay fixture's hero, every
/// `health.set` a Blueprint calls — is byte-identical to what it was.
fn is_flesh(world: &EcsWorld, guid: Uuid) -> bool {
    let Some(e) = world.entity_of(guid) else {
        return false;
    };
    let w = world.world();
    w.get::<Health>(e).is_some() || w.get::<CharacterMovement>(e).is_some()
}

/// Spend a hit's joules: on a body's health here, on a destructible through the
/// host's own wrapper.
///
/// # Lazy health, and what it buys (wave WPN1)
///
/// A character with no [`Health`] is given one **on the first hit that lands on
/// it**, at [`weapon::DEFAULT_VITALITY_J`]. Two alternatives were available and
/// both are worse:
///
/// * giving every character a body at spawn puts 33 bytes an agent per step into
///   `health_state_bytes` — **33 kB a step at the thousand agents
///   `NPC_BUDGET_AGENTS` measures** — and moves every committed trace in the tree
///   for levels that have a crowd and no combat, which is all of them;
/// * giving one to every *materialized* agent makes the trace a function of the
///   crowd BAND, which is the tier-dependent-component trap NPC1a's own
///   `crowd_state_bytes` exists to keep out: an agent would enter and leave the
///   health section as the player walked towards and away from it, and two hosts
///   that tiered it a step apart would diverge for a reason that has nothing to
///   do with anybody's health.
///
/// Lazily, the section is empty until something is shot, which keeps every
/// pre-WPN1 trace byte-identical.
///
/// # WHERE THE BODY STOPS (the WPN1 audit's correction)
///
/// This used to end *"and once a body is in it it stays there however the band
/// moves"*, and the last clause is not true at the bottom of the ladder. The
/// tier owns the components: `Full → Near → Far` removes the collider, the
/// controller and `CharacterMovement` and **leaves the entity**, so a granted
/// [`Health`] survives all three — but `→ Dormant` **despawns the entity**
/// (`crowd::step_crowd`), and the health goes with it. `CrowdRecord` carries the
/// position, the route phase and a pose digest; it does not carry joules.
///
/// So the honest statement is: a wounded crowd agent that leaves the world at
/// `crowd::DEFAULT_CROWD_FAR_M` (**512 m**) comes back **whole**, and for such an
/// agent the health section really is a function of the band. Two things keep
/// that a design bound rather than a divergence: both hosts tier from the same
/// rule on the same step, so they lose it together and no trace comparison can
/// see a disagreement; and a `Far` agent has no collider, so nothing between
/// 96 m and 512 m can be shot at all — the window in which a body is wounded
/// *and* pageable is a walk away from somebody you have already shot.
///
/// Persisting it is a field on `CrowdRecord`, which moves `crowd_state_bytes`
/// and the `AGENT_TRACE_BYTES` ratio quoted against it — a wave, not a doc fix.
/// On this wave's carried list by name.
fn apply_hit(world: &mut EcsWorld, hit: &WeaponHit, dt: f64, report: &mut GameplayReport) {
    let Some(target) = hit.target else {
        return;
    };
    if hit.on_flesh {
        // The body, if this is the first thing that has ever hurt it.
        if weapon::health_of(world, target).is_none() {
            weapon::give_health(world, target, weapon::DEFAULT_VITALITY_J);
        }
        let before = weapon::health_of(world, target)
            .map(|h| h.joules)
            .unwrap_or(0.0);
        // One door (`weapon::damage_entity`), shared with the `health.damage`
        // verb, so a bullet and a script spend joules the same way.
        let Some(r) = weapon::damage_entity(world, target, hit.energy_j) else {
            return;
        };
        if r.was_dead || r.killed {
            // A corpse does not stagger and a kill is the ragdoll's business —
            // `step_deaths` hands it over on this same step, and a hit reaction
            // armed on the way out would be an animation fighting a ragdoll for
            // the same skeleton.
            return;
        }
        stagger(world, hit, target, r.absorbed_j, before, dt, report);
        return;
    }
    // Not flesh. **Only a destructible is owed anything** (wave WPN1): a round
    // into a lamp post, a kerb or the ground reached this list before, and the
    // host answered every one of them with a `NoDestructible` refusal in the log.
    // Owing energy to something with no door is not owing.
    if world
        .entity_of(target)
        .and_then(|e| world.world().get::<inf_ecs::components::Destructible>(e))
        .is_none()
    {
        return;
    }
    // The host spends it at the P22 door. Coalesced by entity so one
    // burst on one wall is one blow — which matters, because **damage is not
    // banked**: three small blows on one step are not a big one, and pretending
    // otherwise here would make the rate of fire a hidden multiplier on damage.
    if let Some(slot) = report.destruct.iter_mut().find(|(g, _)| *g == target) {
        slot.1 += hit.energy_j;
    } else {
        report.destruct.push((target, hit.energy_j));
    }
}

/// **WHAT A PERSON DOES ABOUT BEING HIT** (wave WPN1) — the resist draw, and
/// the one flee door.
///
/// # Why the same draw the carjack uses
///
/// `carjack::RESIST_CHANCE` already answers *"does this person fight you off
/// this time"*, drawn per attempt from the victim's own guid and the sim step —
/// a function of who they are and when you tried, agreed by both hosts, and
/// deliberately **not** a stored counter, because a counter of how many times
/// somebody has resisted is a second copy of what the seed already answers. A
/// punch is the same question with a different verb, so it takes the same draw
/// against a salt of its own: a quarter of the time somebody who is hit stands
/// their ground, and the rest of the time they leave.
///
/// It is a draw and not a certainty because the reference frames show both —
/// the encampment brawl (`frames/police-bike/0033`) has a bystander standing a
/// metre from a fight watching it, and it also has people who are plainly not
/// there any more. A rule that always fled would empty a brawl of everybody but
/// the two people in it.
///
/// # What it is NOT
///
/// It is not a fight-back: an NPC that resists simply stays, and the day one
/// swings back is the day `npc_aim_at` grows a policy — which is EMS3's, for
/// the reason that function's own doc gives. And it reaches
/// [`inf_ecs::crowd::flee_from`], so a person who is not in the population is
/// not made one:
/// the hero, a scripted actor and a shopkeeper with no crowd record are all
/// refused by that door, and the honest answer for them is that being hit does
/// not give them somewhere to be.
fn struck_reaction(
    world: &mut EcsWorld,
    hit: &WeaponHit,
    target: Uuid,
    dt: f64,
    report: &mut GameplayReport,
) {
    // Only somebody the crowd knows about: `flee_from` refuses the rest, and
    // asking first is what keeps this `O(1)` on a hit against the hero.
    if !inf_ecs::crowd::is_in_population(world, target)
        || inf_ecs::crowd::is_panicked(world, target)
    {
        return;
    }
    let tick = inf_ecs::traffic::steps(world);
    if inf_ecs::crowd::agent_unit(target, tick, SALT_STRUCK) < super::carjack::RESIST_CHANCE {
        report.stood_their_ground += 1;
        return;
    }
    let Some(from) = feet_of(world, target) else {
        return;
    };
    // Away from where the blow came from — the attacker's own muzzle, which is
    // the one point in a `WeaponHit` that is always the attacker's.
    if inf_ecs::crowd::flee_from(world, target, from, hit.from, dt, PANIC_FLEE_M) {
        report.panic.fled += 1;
    } else if inf_ecs::dispatch::is_responder(world, target) {
        // **A responder shot at point blank does not leave either** (wave EMS2).
        // The rule is the flee door's and holds here without a line; what this
        // adds is the count, on `step_panic`'s own terms — an officer that was
        // hit is exactly the officer a gate wants to see stay.
        report.panic.exempt += 1;
    }
}

/// Salts the "does being hit make you leave" draw — `carjack::SALT_RESIST`'s
/// shape, with its own constant so a person who resisted a carjack is not
/// thereby the person who stands their ground when punched.
const SALT_STRUCK: u64 = 0x5354_5255_434b_0001;

/// **A hit that hurts is a hit that shows** (wave WPN1) — the one-shot reaction,
/// and the blow that puts a body on the floor.
///
/// Two things, and the second one is a mode:
///
/// * the animation trigger ([`weapon::STAGGER_TRIGGER`]) is armed on **every**
///   non-fatal blow, through the same `set_anim_trigger` seam the fire and the
///   reload use. A character with no state machine plays nothing and still takes
///   the damage — the reload's rule verbatim;
/// * a blow that takes [`weapon::STAGGER_FRACTION`] of what the body had left
///   also **puts it off its feet**, into `MovementMode::FallControlled`. That is
///   the carjack's own eject verbatim (`carjack.rs`: *"being pulled out of a car
///   is a fact about your body and not a choice"*) and it is what
///   `transition_is_legal`'s own doc has been describing since P29.3.
///
/// The mode is asked of the table rather than assigned: a swimmer, a ragdoll and
/// a driver all refuse it, and a refusal is a value.
fn stagger(
    world: &mut EcsWorld,
    hit: &WeaponHit,
    target: Uuid,
    absorbed_j: f64,
    before_j: f64,
    dt: f64,
    report: &mut GameplayReport,
) {
    inf_ecs::anim_bridge::set_anim_trigger(world, target, weapon::STAGGER_TRIGGER);
    report.staggers += 1;
    // **…and whoever it happened to may decide to leave** (wave WPN1). The
    // draw, and the flee, are `struck_reaction`'s.
    struck_reaction(world, hit, target, dt, report);
    if !weapon::is_staggering(absorbed_j, before_j) {
        return;
    }
    let Some(entity) = world.entity_of(target) else {
        return;
    };
    let Some(mut cm) = world
        .world_mut()
        .get_mut::<inf_ecs::components::CharacterMovement>(entity)
    else {
        return;
    };
    if cm.runtime.seat.is_seated()
        || !inf_ecs::movement::transition_is_legal(
            cm.mode,
            inf_ecs::components::MovementMode::FallControlled,
        )
    {
        return;
    }
    cm.mode = inf_ecs::components::MovementMode::FallControlled;
    cm.runtime.time_in_mode_s = 0.0;
    report.knockdowns += 1;
}

/// **Arm a kick** at the door in front of `character`, if there is one and the
/// character is close enough and facing it.
///
/// The attack button's door consumer. Returns whether a kick was armed, which is
/// what stops the same press also firing a weapon at the door.
pub fn try_kick(
    world: &mut EcsWorld,
    band: &inf_ecs::band::SimBand,
    character: Uuid,
    feet: DVec3,
    aim_yaw_deg: f64,
) -> bool {
    if world.entity_of(character).is_none() {
        return false;
    }
    if door::pending_kick(world, character).is_some() {
        return false;
    }
    // The nearest LOCKED door in kicking reach, by the interaction rule's own
    // arithmetic — a kick is a reach and a cone, exactly as a prompt is, and
    // spelling it a second way would let a player kick a door the prompt says
    // is out of reach.
    let field = inf_ecs::door::door_field(world);
    let mut candidates: Vec<inf_ecs::interact::InteractCandidate> = Vec::new();
    for p in super::door::placements_near(world, band) {
        let state = field
            .map(|f| f.get(p.guid, &p.spec))
            .unwrap_or_else(|| inf_ecs::door::DoorState::fresh(&p.spec));
        if !state.locked || state.lock_broken {
            continue;
        }
        candidates.push(inf_ecs::interact::InteractCandidate {
            guid: p.guid,
            verb: inf_ecs::interact::InteractVerb::Use,
            label: p.label.clone(),
            position: inf_ecs::door::prompt_position(&p),
            range_m: inf_ecs::door::KICK_REACH_M,
            view_cone_deg: inf_ecs::door::KICK_CONE_DEG,
            // A kick is a leg, not a hand.
            grip: None,
        });
    }
    candidates.sort_by_key(|c| c.guid);
    let Some(hit) = inf_ecs::interact::resolve(&candidates, feet, aim_yaw_deg) else {
        return false;
    };
    door::set_pending_kick(
        world,
        character,
        PendingKick {
            door: hit.guid,
            fuse_s: KICK_FUSE_S,
        },
    );
    inf_ecs::anim_bridge::set_anim_trigger(world, character, weapon::KICK_TRIGGER);
    true
}

fn step_kicks(world: &mut EcsWorld, dt: f64, report: &mut GameplayReport) {
    for (guid, mut kick) in door::pending_kicks(world) {
        let landed = inf_ecs::anim_bridge::consume_anim_notify(world, guid, weapon::KICK_NOTIFY);
        kick.fuse_s -= dt;
        if !landed && kick.fuse_s > 0.0 {
            door::set_pending_kick(world, guid, kick);
            continue;
        }
        door::clear_pending_kick(world, guid);
        report.kicks += 1;
        // **The one energy door.** A kick is a mass and a speed; what it is
        // compared against is the lock's own P22 bond energy. The crash-through
        // in `d3::movement` calls the same `strike_door` with the character's
        // own kinetic energy.
        let verdict = super::door::strike_door(world, kick.door, inf_ecs::door::kick_energy_j());
        if verdict.broke {
            report.locks_broken += 1;
        }
    }
}

/// **Every body that stopped working goes to the ragdoll** — and nothing gets up
/// again.
///
/// # RULING: what a respawn would be, and why it is not here (I6 item 7)
///
/// The hero can be killed — `phase30_gameplay_gate` proves a round takes 1 700 J
/// off a 2 000 J body and `weapon::Downed` latches the handoff — and when it is,
/// the level has a ragdoll in it and no player. The mandate asks for a respawn.
/// Named rather than built, because the shape matters more than the code:
///
/// **The simplest honest form is a re-seat, not a reload.** On the step a
/// `player_controlled` body is handed to the ragdoll, a host would: end the
/// ragdoll through `ragdoll_bridge`'s own door (the table already permits
/// `(Ragdoll, Grounded)`), restore [`Health`] to its capacity, place the body at
/// the level's own start — the `StreamingSource`-carrying spawn the scene
/// already names — and clear the movement runtime's edges the way `clear_edges`
/// does at a seat. Nothing else. **The world keeps everything that happened**:
/// the doors stay where they were kicked, the bag keeps what it held, the debris
/// stays on the floor, the crowd stays scattered.
///
/// That is deliberate rather than lazy, and it is the reason there is no save
/// container. Restoring a *world* means a snapshot of it, and this engine's one
/// snapshot format is `.inf_lvl` — **the author's document**, which P21's own
/// ruling forbids the runtime to write ("in the editor the render store IS the
/// save's staging source"). A respawn that rolled the world back would need a
/// second, runtime-owned container with its own schema, its own migration and
/// its own gate; a respawn that does not is four calls into doors that already
/// exist. The second is a game, and the first is a wave.
///
/// It is not written here because *where* it goes is a decision this function
/// cannot make: reviving the camera subject belongs to whoever owns the camera
/// subject, and today that is each host. Carried by name.
///
/// Answers **who stopped working on this step**, in `Guid` order — the list the
/// witness pass records a death from, and the reason this returns anything at
/// all: `report.kills` counts the ones the ragdoll *took*, and a body the
/// ragdoll refused is still a body somebody saw fall.
fn step_deaths(
    world: &mut EcsWorld,
    _bridge: &mut PhysicsBridge3D,
    report: &mut GameplayReport,
) -> Vec<Uuid> {
    let dead = weapon::newly_dead(world);
    for guid in dead.iter().copied() {
        // **The latch goes down FIRST**, whether or not the ragdoll takes it.
        // A body whose ragdoll is refused — no `CharacterMovement`, a mode the
        // table will not leave — must not be offered again on the next step; see
        // `weapon::Downed` for the measurement.
        weapon::mark_downed(world, guid);
        // **The P29.4 door, unchanged.** `start_ragdoll`'s own doc has named "a
        // damage system" as its intended caller since P29.4; this is that
        // caller, and it does not reach past the door into the rig.
        if super::ragdoll_bridge::start_ragdoll(world, guid) {
            report.kills += 1;
        }
    }
    dead
}

/// Which side of `door` a character standing at `feet` is on — re-exported for
/// the hosts' prompt, so the press and the prompt read one function.
pub fn side_of(world: &EcsWorld, door_guid: Uuid, feet: DVec3) -> Option<DoorSide> {
    let p = super::door::placement_of(world, door_guid)?;
    Some(p.spec.side_of(p.hinge, feet))
}

/// **AN ARMED NPC AIMS AND FIRES** (wave WPN1) — the one door an intent that is
/// not the local player's crosses into a character.
///
/// # Why a door at all
///
/// `inf_ecs::movement::apply_intent` writes every field this writes — and it
/// writes them **only onto `player_controlled` characters**, which is the line
/// that makes an NPC's body the crowd's business and not the input layer's. So
/// an armed NPC had no way to pull a trigger at all: nothing in the tree could
/// write `want_attack` onto one.
///
/// This is the complement, and it is deliberately the same shape as VEH2b's
/// `drive_intent` door: **one function, refusing the other half of the world.**
/// `apply_intent` refuses everything that is not player-controlled; this refuses
/// everything that is. Between them every character's intent has exactly one
/// author, which is the property that makes a divergence findable.
///
/// # What it does NOT do
///
/// No cover, no squad, no target selection, no reaction time, no leading a moving
/// target, no decision to *stop*. Those are EMS3's, and the reason to keep them
/// out is that each one is a policy: a policy in this function would be a policy
/// two hosts have to agree about, written where nobody would look for it. What
/// this answers is the mechanical question — *can an armed NPC point a weapon at
/// somebody and pull the trigger* — and the answer is now yes.
///
/// The **spread is free and already deterministic**: `shot_direction` folds the
/// weapon's own `spread_seed` with its shot index, so two NPCs firing the same
/// rifle at the same target do not put their rounds in the same hole, and a
/// replay reproduces every one of them.
///
/// `hold_trigger` is the trigger's **level**, exactly as the player's is, so a
/// semi-automatic weapon in an NPC's hands fires once per press through
/// `try_fire`'s own edge rule and needs no second mechanism. It does **not**
/// write `press_attack`: that edge is the door-kick's, and an NPC that kicked
/// every locked door it happened to face is not what "an armed NPC can fire" is
/// asking for.
///
/// Answers `false` for a shooter that is not there, is player-controlled, or has
/// no target to aim at — refusals as values, all the way down.
pub fn npc_aim_at(world: &mut EcsWorld, shooter: Uuid, target: Uuid, hold_trigger: bool) -> bool {
    let Some(entity) = world.entity_of(shooter) else {
        return false;
    };
    if world
        .world()
        .get::<CharacterMovement>(entity)
        .is_none_or(|cm| cm.player_controlled)
    {
        return false;
    }
    // From the shooter's own muzzle to the point a swing would land on — the
    // same `strike_point` melee aims at, so a rifle and a fist agree about where
    // a person is.
    let Some((from, _, _, _)) = muzzle_of(world, shooter) else {
        return false;
    };
    let Some(at) = strike_point(world, target) else {
        return false;
    };
    let to = at - from;
    let planar = (to.x * to.x + to.z * to.z).sqrt();
    if !to.is_finite() || (planar <= 1.0e-9 && to.y.abs() <= 1.0e-9) {
        return false;
    }
    // **Portable trigonometry** (the P14 law): these two numbers reach
    // `shot_direction`, whose output reaches the ray cast, whose hit reaches the
    // damage door, which reaches the trace.
    let yaw = inf_ecs::movement::planar_yaw_deg(inf_ecs::math::Vec2d::new(to.x, to.z));
    let pitch = if planar <= 1.0e-9 {
        // Straight up or straight down: the yaw is whatever it was and the pitch
        // is the limit `aim_forward` clamps to anyway.
        if to.y > 0.0 {
            89.9
        } else {
            -89.9
        }
    } else {
        // `asin(y/|to|)` written as `90 - acos`, because `pacos64` is the
        // portable inverse this engine has and `f64::asin` is not bit-portable.
        90.0 - inf_math::pacos64((to.y / to.length()).clamp(-1.0, 1.0)).to_degrees()
    };
    let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(entity) else {
        return false;
    };
    cm.runtime.aim_yaw_deg = yaw;
    cm.runtime.aim_pitch_deg = pitch.clamp(-89.9, 89.9);
    // The LEVEL, assigned — `apply_intent`'s own treatment of this field, and the
    // reason a semi-automatic weapon needs nothing else.
    cm.runtime.want_attack = hold_trigger;
    true
}

/// Give `character` an equipped weapon by item id — the door a Blueprint and a
/// gate both use, so a weapon cannot be equipped without its ammunition clock.
pub fn equip_weapon(world: &mut EcsWorld, character: Uuid, item_id: &str) -> bool {
    let Some(entity) = world.entity_of(character) else {
        return false;
    };
    let id = item::canonical_id(item_id);
    let slot = {
        let Some(inv) = world.world().get::<Inventory>(entity) else {
            return false;
        };
        inv.slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| s.id == id))
    };
    let Some(slot) = slot else {
        return false;
    };
    let Some(def) = item::item_defs(world)
        .and_then(|d| d.get(&id))
        .and_then(|d| d.weapon)
    else {
        return false;
    };
    {
        let w = world.world_mut();
        if let Some(mut inv) = w.get_mut::<Inventory>(entity) {
            inv.equip(slot);
        }
    }
    world
        .world_mut()
        .entity_mut(entity)
        .insert(WeaponState::full(&id, &def));
    true
}

/// The kind a shot of the equipped weapon is — what a tracer's speed reads.
pub fn equipped_shot_kind(world: &EcsWorld, character: Uuid) -> Option<ShotKind> {
    equipped_weapon(world, character).map(|(_, d)| d.kind)
}
