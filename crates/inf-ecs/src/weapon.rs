//! **Weapons and what they hurt** (island wave I6): a name-keyed definition, the
//! ammunition clock on a character, and a health component whose unit is the
//! engine's one unit.
//!
//! # HEALTH IS JOULES, and that is not a stylistic choice
//!
//! `docs/memos/p22-strength.md` §1 is the argument and it was made about walls:
//! this engine has **no damage numbers**. A blow is a mass and a speed, a bond
//! is a strength and an area, and what they are compared in is joules, because
//! `Pa · m² · m = N · m = J` with no invented conversion. A health component
//! measured in "hit points" would put a conversion constant back — per weapon,
//! per material, exactly the table that memo refuses — and it would sit in the
//! one place where a bullet, a kick, a fall and a collapsing wall all meet.
//!
//! So [`Health`] holds **joules**: how much energy this body can absorb before
//! it stops working. It is the "hp" the mandate asked for, in the unit
//! everything else already speaks, and it means the *same* number that breaks a
//! lock and the *same* number that detaches a chunk also drops a character —
//! through one comparison, with nothing to tune between them.
//!
//! # The definition is tunable BY NAME
//!
//! [`WeaponDef::set`] / [`WeaponDef::names`] are the `VehicleTuning` /
//! `CameraTuning` door verbatim, so every weapon parameter is live-tunable
//! through P29.5's queue and a UI can enumerate the door rather than restate it
//! (the P29.6 audit's A14).
//!
//! # Spread is deterministic
//!
//! A shot's scatter is a **counter-based hash of the shot's index**, not a
//! stateful RNG — so a replay, a PIE preview and a shipped build fire the same
//! shot in the same direction, and a gate can assert where a bullet went. The
//! trigonometry is `psin64`/`pcos64` for the P14 reason: the direction reaches a
//! ray cast, whose hit reaches the damage door, which reaches the trace.

use bevy_ecs::prelude::{Component, With};
use glam::DVec3;
use uuid::Uuid;

use crate::components::{AudioSource, DistanceModel, Guid};
use crate::world::EcsWorld;

// ── the numbers ─────────────────────────────────────────────────────────────

/// What a body can absorb before it stops working, joules.
///
/// A rifle round carries on the order of 1 700 J, so two of them is a
/// stop — which is the behaviour the number exists to produce, arrived at the
/// same way `CRACK_OPENING_M`'s sanity check is: name the real quantities and
/// see whether the outcome is the one everybody expects.
pub const DEFAULT_VITALITY_J: f64 = 2000.0;

/// The most rounds a magazine may hold. A bound on hostile content, not a
/// design limit.
pub const MAX_MAGAZINE: u32 = 10_000;
/// The fastest a weapon may cycle, rounds per minute.
pub const MAX_RPM: f64 = 6000.0;
/// The widest cone a weapon may scatter into, degrees (total).
pub const MAX_SPREAD_DEG: f64 = 45.0;
/// The furthest a hitscan shot may reach, metres.
pub const MAX_RANGE_M: f64 = 20_000.0;

/// How far a muzzle may sit from a weapon's own origin, metres — the bound on
/// [`WeaponDef::muzzle_forward_m`]. A barrel longer than this is a vehicle.
pub const MAX_MUZZLE_FORWARD_M: f64 = 3.0;

/// How far a melee weapon may reach, metres — the bound on a melee
/// [`WeaponDef::range_m`].
///
/// Two and a half. A halberd is at the top of it and anything longer is a
/// vehicle, on [`MAX_MUZZLE_FORWARD_M`]'s own reasoning: the bound exists to
/// stop hostile content, not to express a design.
pub const MAX_MELEE_REACH_M: f64 = 2.5;

/// How a shot travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ShotKind {
    /// Instant: a ray, resolved the step the trigger is pulled.
    #[default]
    Hitscan,
    /// A body in flight, resolved when it arrives.
    Projectile,
    /// **Not a shot at all** (wave WPN1): a reach and an arc, resolved against
    /// the bodies in front of the swinger.
    ///
    /// # Why this is a `WeaponDef` and not a system beside it
    ///
    /// Everything a punch needs, a rifle already has: a rate
    /// ([`WeaponDef::rounds_per_minute`] is how fast you can swing), a damage in
    /// joules, an automatic/semi flag (a held button either keeps swinging or it
    /// does not), an ammunition clock that stops the second swing arriving before
    /// the first has finished, and a trigger arbitration that already decides
    /// between kicking a door and using the weapon. A parallel melee system would
    /// have re-derived every one of them, and the attack button would then have
    /// had two arbitrations to agree about.
    ///
    /// What melee does **not** have is a ray, and that is the whole of the
    /// difference: [`WeaponDef::range_m`] is a *reach* (bounded by
    /// [`MAX_MELEE_REACH_M`]), [`WeaponDef::melee_arc_deg`] is the cone it
    /// sweeps, and the resolution is the interaction rule's own reach-and-cone
    /// (`inf_ecs::interact::resolve`) rather than a cast. A magazine is
    /// meaningless, so a melee definition simply carries a large one and never
    /// runs out — stated rather than special-cased, because a special case here
    /// is a second `try_fire`.
    Melee,
}

/// **What a weapon IS.**
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeaponDef {
    /// Hitscan or projectile.
    pub kind: ShotKind,
    /// Whether holding the trigger keeps firing.
    pub automatic: bool,
    /// The energy one shot delivers, **joules** — through the P22 door for a
    /// destructible and against [`Health`] for a character. One number, two
    /// consumers, no conversion.
    pub damage_j: f64,
    /// How fast it cycles, rounds per minute.
    pub rounds_per_minute: f64,
    /// The total cone a shot may scatter into, degrees.
    pub spread_deg: f64,
    /// How many rounds a magazine holds.
    pub magazine: u32,
    /// How many rounds the character starts carrying beyond the magazine.
    pub reserve: u32,
    /// How long a reload takes, seconds — the ceiling. The actual reload is
    /// gated on the animation's own notify when there is one; see
    /// [`WeaponState::reload_left_s`].
    pub reload_s: f64,
    /// How far a hitscan shot reaches, metres.
    pub range_m: f64,
    /// How fast a projectile leaves the muzzle, m/s. Ignored by a hitscan.
    pub muzzle_speed_mps: f64,
    /// The seed the deterministic spread is folded against, so two weapons
    /// firing their first shot do not scatter identically.
    pub spread_seed: u64,
    /// **How far the muzzle is from the weapon's own origin**, metres, along its
    /// barrel (the weapon's local `+Z` — model space faces `+Z` in this engine).
    ///
    /// This is the weapon's *muzzle socket*, expressed as the one number a
    /// straight barrel needs. It is what a shot's origin is read off once the
    /// weapon is a real entity attached to a hand — see
    /// `inf_physics::d3::gameplay::muzzle_of`, which falls back to a height above
    /// the character's feet for a character that has no rig to hang a weapon on.
    ///
    /// This type carries **no** `Serialize`, rides no wire and is built from TOML
    /// by `from_toml_table`, so adding a field costs no schema bump — the one
    /// place in this arc where that is true, and the reason the muzzle is
    /// described here rather than in a new component.
    pub muzzle_forward_m: f64,
    /// **The cone a melee swing sweeps**, degrees (total) — ignored by every
    /// other [`ShotKind`].
    ///
    /// A swing is a reach and an arc; this is the arc, and it goes straight into
    /// `InteractCandidate::view_cone_deg`, so a punch and a door prompt are
    /// refused by the same rule and a player who is told a thing is out of reach
    /// cannot hit it.
    pub melee_arc_deg: f64,
}

impl Default for WeaponDef {
    fn default() -> Self {
        Self {
            kind: ShotKind::Hitscan,
            automatic: true,
            // A rifle round: about 1 700 J at the muzzle.
            damage_j: 1700.0,
            rounds_per_minute: 600.0,
            spread_deg: 1.5,
            magazine: 30,
            reserve: 120,
            reload_s: 2.0,
            range_m: 400.0,
            muzzle_speed_mps: 900.0,
            spread_seed: 0,
            // A rifle: the muzzle is 45 cm along the barrel from the grip.
            muzzle_forward_m: 0.45,
            // A rifle does not swing; this is what a melee definition would use.
            melee_arc_deg: FIST_ARC_DEG,
        }
    }
}

/// **The item id an unarmed character's fists carry** (wave WPN1).
///
/// It is deliberately **not** in [`crate::item::ItemDefs`] and cannot be picked
/// up, dropped, equipped or seen in a bag: a pair of hands is not an item, and
/// putting one in the catalogue would make it a thing a level could take away.
/// What the id is for is [`WeaponState::item_id`], which is the field that
/// decides whether an ammunition clock is stale — so a character that punches
/// and then equips a rifle gets a fresh magazine rather than the fist's.
///
/// The colon is the same namespace mark `BlueprintClass::new("act:…")` uses.
/// **It is a convention with one mechanical leg, not a construction** (the WPN1
/// audit's correction, which read "by construction"): [`crate::item::
/// canonical_id`] only trims and lower-cases, so nothing in the catalogue
/// *rejects* a colon — what a TOML catalogue rejects is the **bare key**, and an
/// author reaching this id has to quote it deliberately (`["engine:fists"]`).
/// A level that did would get an item the ammunition readout refuses to count
/// (see [`carries_ammunition`]); refusing colons at
/// [`crate::item::ItemDefs::insert`] would make the sentence true and is a
/// content-visible refusal rather than an audit fix. Carried by name.
pub const FIST_ITEM: &str = "engine:fists";

/// **What a punch carries**, joules.
///
/// A hundred and fifty. A trained fist arrives at 6–9 m/s carrying an effective
/// mass of about 4 kg, which is `0.5 · 4 · 8²` ≈ 130 J — the same "name the real
/// quantities" arithmetic [`DEFAULT_VITALITY_J`] is arrived at by. Against a
/// 2 000 J body that is thirteen punches, which is a fist-fight; against the
/// same body after a rifle round it is two, which is why the stagger threshold is
/// a proportion.
pub const FIST_DAMAGE_J: f64 = 150.0;

/// **How far a punch reaches**, metres — from the swinger's own feet to the body
/// it lands on, which is the same measurement `interact::resolve` makes for a
/// door prompt.
///
/// One metre two. `DOOR_REACH_M`'s neighbourhood on purpose: a player who can
/// open a door at this distance can hit a person at it, and a reach a player
/// cannot see the edge of reads as a broken control (`ENTER_REACH_M`'s own note).
pub const FIST_REACH_M: f64 = 1.2;

/// **The cone a punch sweeps**, degrees (total).
///
/// A hundred, which is wider than a rifle's aim and narrower than a shove: you
/// can hit somebody a little off to the side, and you cannot hit somebody beside
/// you.
pub const FIST_ARC_DEG: f64 = 100.0;

/// **How fast a person can throw punches**, "rounds" per minute.
///
/// Ninety — two thirds of a second a swing, which is a jab-and-recover rather
/// than a flurry.
///
/// It paces the swings and **nothing else** (the WPN1 audit's correction: this
/// used to add *"and through [`recoil_fraction`] is also how long the body
/// carries the swing's own pose"*, which the wave's own carried list already
/// contradicted). A fist is not *equipped* — it is not in the catalogue at all —
/// so `d3::gameplay`'s `recoil_of` answers `0.0` for it and a punch moves no
/// bone. `weapon_hands_gate` asserts exactly that, so the day it changes the arm
/// fails.
pub const FIST_RPM: f64 = 90.0;

/// **A pair of hands, as a weapon** (wave WPN1) — what an unarmed character's
/// attack button reaches.
///
/// Semi-automatic on purpose: a held button throws one punch, and the next one
/// needs the button released. Holding a trigger down is what an automatic weapon
/// is for; a person who holds the button down is not punching continuously, and
/// `try_fire`'s edge rule expresses that without a second mechanism.
pub fn fist_def() -> WeaponDef {
    WeaponDef {
        kind: ShotKind::Melee,
        automatic: false,
        damage_j: FIST_DAMAGE_J,
        rounds_per_minute: FIST_RPM,
        spread_deg: 0.0,
        // A magazine is meaningless for a fist and it must never run out, so it
        // carries the bound rather than a special case in `try_fire`.
        magazine: MAX_MAGAZINE,
        reserve: MAX_MAGAZINE,
        reload_s: 0.05,
        range_m: FIST_REACH_M,
        muzzle_speed_mps: 1.0,
        spread_seed: 0,
        muzzle_forward_m: 0.0,
        melee_arc_deg: FIST_ARC_DEG,
    }
}

impl WeaponDef {
    /// **The tuning door, by name.** `false` for an unknown name or a value
    /// that is not finite — a refusal, never a failure, which is
    /// `VehicleTuning::set`'s rule verbatim.
    ///
    /// Ranges are clamped rather than refused, on the same reasoning
    /// `CameraTuning` gives: a designer dragging a slider past a bound wants the
    /// bound, not a silently ignored edit.
    pub fn set(&mut self, name: &str, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match name {
            "damage_j" => self.damage_j = value.max(0.0),
            "rounds_per_minute" => self.rounds_per_minute = value.clamp(1.0, MAX_RPM),
            "spread_deg" => self.spread_deg = value.clamp(0.0, MAX_SPREAD_DEG),
            "magazine" => self.magazine = value.clamp(1.0, f64::from(MAX_MAGAZINE)) as u32,
            "reserve" => self.reserve = value.clamp(0.0, f64::from(MAX_MAGAZINE)) as u32,
            "reload_s" => self.reload_s = value.clamp(0.05, 60.0),
            "range_m" => self.range_m = value.clamp(0.1, MAX_RANGE_M),
            "muzzle_speed_mps" => self.muzzle_speed_mps = value.clamp(1.0, 10_000.0),
            "muzzle_forward_m" => self.muzzle_forward_m = value.clamp(0.0, MAX_MUZZLE_FORWARD_M),
            "melee_arc_deg" => self.melee_arc_deg = value.clamp(0.0, 360.0),
            // Booleans and the kind come across the same door as numbers,
            // because the door is one `(name, f64)` pair and a second door for
            // three flags would be a second thing to keep in step.
            "automatic" => self.automatic = value != 0.0,
            "projectile" => {
                self.kind = if value != 0.0 {
                    ShotKind::Projectile
                } else {
                    ShotKind::Hitscan
                }
            }
            // The third kind across the same `(name, f64)` door, for
            // `projectile`'s reason: a second door for three flags would be a
            // second thing to keep in step. Turning melee OFF answers `Hitscan`,
            // which is the default and is what `projectile` does.
            "melee" => {
                self.kind = if value != 0.0 {
                    ShotKind::Melee
                } else {
                    ShotKind::Hitscan
                }
            }
            _ => return false,
        }
        true
    }

    /// Every settable name, sorted — so a UI and a test enumerate the door
    /// rather than restate it.
    pub fn names() -> &'static [&'static str] {
        &[
            "automatic",
            "damage_j",
            "magazine",
            "melee",
            "melee_arc_deg",
            "muzzle_forward_m",
            "muzzle_speed_mps",
            "projectile",
            "range_m",
            "reload_s",
            "reserve",
            "rounds_per_minute",
            "spread_deg",
        ]
    }

    /// **Whether this weapon is swung rather than fired.**
    ///
    /// One reader-facing question rather than a `match` at every call site: the
    /// gameplay step asks it to choose between a cast and an arc, and the
    /// difference between "is a melee" and "is not a hitscan" is a projectile.
    pub fn is_melee(&self) -> bool {
        self.kind == ShotKind::Melee
    }

    /// **How far this weapon reaches**, metres — the range a cast is given, or
    /// the reach an arc is resolved over.
    ///
    /// One door, because the two bounds differ by an order of magnitude
    /// ([`MAX_RANGE_M`] against [`MAX_MELEE_REACH_M`]) and a call site that
    /// clamped a swing to a rifle's bound would let a 20 km punch through.
    pub fn reach_m(&self) -> f64 {
        if self.is_melee() {
            self.range_m.clamp(0.1, MAX_MELEE_REACH_M)
        } else {
            self.range_m.clamp(0.1, MAX_RANGE_M)
        }
    }

    /// How long one round takes, seconds.
    pub fn fire_interval_s(&self) -> f64 {
        if !self.rounds_per_minute.is_finite() || self.rounds_per_minute <= 0.0 {
            return f64::INFINITY;
        }
        60.0 / self.rounds_per_minute
    }

    /// Read the weapon half of an item's TOML table, if it has one.
    ///
    /// A table with **no `[…].weapon`** sub-table is not a weapon and answers
    /// `None`; a malformed one is an error, because a weapon whose damage was
    /// silently dropped would fire blanks and say nothing.
    pub fn from_toml_table(
        t: &toml::map::Map<String, toml::Value>,
    ) -> Result<Option<Self>, String> {
        let Some(w) = t.get("weapon") else {
            return Ok(None);
        };
        let w = w
            .as_table()
            .ok_or_else(|| "a weapon is a table".to_string())?;
        let mut def = WeaponDef::default();
        for (k, v) in w {
            let n = match v {
                toml::Value::Float(f) => *f,
                toml::Value::Integer(i) => *i as f64,
                toml::Value::Boolean(b) => f64::from(u8::from(*b)),
                toml::Value::String(s) => {
                    // The one string key: `kind = "projectile" | "hitscan"`.
                    if k == "kind" {
                        def.kind = match s.trim().to_ascii_lowercase().as_str() {
                            "projectile" => ShotKind::Projectile,
                            "hitscan" => ShotKind::Hitscan,
                            "melee" => ShotKind::Melee,
                            other => return Err(format!("unknown weapon kind {other}")),
                        };
                        continue;
                    }
                    return Err(format!("weapon key {k} is not a number"));
                }
                _ => return Err(format!("weapon key {k} is not a number")),
            };
            if !def.set(k, n) {
                return Err(format!("unknown weapon key {k}"));
            }
        }
        Ok(Some(def))
    }
}

/// **The ammunition clock on a character** — a runtime component, inserted when
/// a weapon is equipped and replaced when a different one is.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct WeaponState {
    /// The item id this state is about. A state whose id has stopped matching
    /// the equipped item is stale and is replaced rather than reused: two
    /// weapons sharing one magazine is not a feature.
    pub item_id: String,
    /// Rounds in the magazine.
    pub magazine: u32,
    /// Rounds carried beyond it.
    pub reserve: u32,
    /// Seconds until the next shot may be fired.
    pub cooldown_s: f64,
    /// Seconds left of a reload, or `0.0` when none is running.
    ///
    /// **The ceiling, not the schedule.** A reload finishes when the animation's
    /// own notify fires ([`RELOAD_NOTIFY`]); this is what finishes it when there
    /// is no animation, which is every headless run and every character without
    /// a rig. A gate that only ever ran the clock would certify a reload the
    /// notify seam never touched, so both paths exist and both are armed.
    pub reload_left_s: f64,
    /// How many rounds this weapon has fired — the spread's counter, and the
    /// number that makes a shot's direction a pure function of sim state.
    pub shots: u64,
    /// Whether the trigger was down last step, so a semi-automatic weapon fires
    /// once per press.
    pub trigger_held: bool,
}

/// The animation notify a reload finishes on.
///
/// One name, exported, because the state machine authors it and the fixed step
/// consumes it — and a notify spelled twice is a reload that never completes on
/// exactly the rigs that spell it the other way.
pub const RELOAD_NOTIFY: &str = "weapon_reload_done";

/// The animation trigger a shot arms.
pub const FIRE_TRIGGER: &str = "weapon_fire";

/// The animation trigger a reload arms.
pub const RELOAD_TRIGGER: &str = "weapon_reload";

/// **The animation trigger a melee swing arms** (wave WPN1).
///
/// Its own name rather than [`FIRE_TRIGGER`]: a rig that played `weapon_fire`
/// when somebody threw a punch would be firing an empty hand, and a state
/// machine cannot tell the two apart from a trigger alone.
pub const MELEE_TRIGGER: &str = "melee_swing";

/// The animation trigger a door kick arms — a P29-style one-shot, reached
/// through `inf_ecs::anim_bridge::set_anim_trigger` exactly as the ragdoll's is.
pub const KICK_TRIGGER: &str = "door_kick";

/// **The animation trigger a hit reaction arms** (wave WPN1).
///
/// A P29-style one-shot, reached through
/// [`crate::anim_bridge::set_anim_trigger`] exactly as [`FIRE_TRIGGER`] and the
/// ragdoll's are — and, exactly as those are, it is armed whether or not
/// anything is listening: a character with no state machine takes the damage and
/// plays nothing, which is the "the animation follows the decision rather than
/// gating it" rule the reload already follows.
pub const STAGGER_TRIGGER: &str = "hit_react";

/// **The fraction of a body's remaining capacity one blow has to take to put it
/// off its feet.**
///
/// A third. Not a joule count, because a joule count would be a second damage
/// table — the same defect `DEFAULT_VITALITY_J`'s doc refuses — and because the
/// interesting quantity is *proportion*: a rifle round is a third of a fresh
/// 5 100 J body and the whole of a hurt one, and the second is the hit that
/// should drop somebody.
///
/// Measured against the engine's own numbers: a 1 700 J rifle round against the
/// default 2 000 J body is 0.85 of it and staggers; a 150 J punch against the
/// same body is 0.075 and does not, which is the whole difference between being
/// shot and being hit.
pub const STAGGER_FRACTION: f64 = 1.0 / 3.0;

/// Whether a blow of `absorbed_j` on a body that had `before_j` left is one that
/// takes it off its feet.
///
/// The denominator is what the body **had**, not what it started with: the third
/// punch of a fight lands on a body with less to give than the first did, and
/// measuring against the capacity would make a beating feel identical from
/// beginning to end.
pub fn is_staggering(absorbed_j: f64, before_j: f64) -> bool {
    if !absorbed_j.is_finite() || !before_j.is_finite() || before_j <= 0.0 {
        return false;
    }
    absorbed_j >= before_j * STAGGER_FRACTION
}

/// The animation notify a door kick lands on.
///
/// **The impulse is on the notify, not on the press.** A kick that broke the
/// lock the instant the button went down would break it before the leg moved,
/// which is the whole reason P29.4 built a notify seam.
pub const KICK_NOTIFY: &str = "door_kick_impact";

// ── the report ──────────────────────────────────────────────────────────────

/// **The engine's committed gunshot** — the `.inf_audio` a round leaving a
/// barrel plays.
///
/// A fixed GUID, on [`crate::venue::VENUE_MUSIC_CLIP`]'s own terms: an asset a
/// weapon names by id must have the same id every time or the committed bytes
/// are a different set of files on every build. The clip itself is committed
/// beside the gameplay fixture (`samples/phase30-gameplay/Report.inf_audio`).
///
/// A host that has not loaded it resolves nothing and plays silence; the
/// **command** is issued either way, which is the Phase-12 doctrine's own
/// observable — the command stream, not the audible output, is the contract.
pub const WEAPON_REPORT_CLIP: Uuid = Uuid::from_u128(0x5750_4e31_0000_0001);

/// The bus a gunshot plays on. `sfx`, so a player who turns the effects down
/// turns the shooting down and the music stays where they put it.
pub const REPORT_BUS: &str = "sfx";

/// The base linear volume of a gunshot.
///
/// Just under unity, [`crate::venue::VENUE_MUSIC_VOLUME`]'s reasoning inverted:
/// this is the loudest *transient* in the engine and it is heard at the muzzle,
/// so the headroom is left for the mixer rather than spent here.
pub const REPORT_VOLUME: f64 = 0.9;

/// Metres inside which a gunshot is at full volume.
///
/// Three — an arm's length and a barrel. Inside that the shooter is the shooter,
/// and the spatial model has nothing useful to say about half a metre.
pub const REPORT_MIN_M: f64 = 3.0;

/// Metres past which a gunshot is silent.
///
/// **Two hundred and fifty**, which is deliberately much further than anything
/// else this engine emits (a venue's music stops at forty). A gunshot in a town
/// is heard three streets away and that is the whole point of the sound: it is
/// the one emitter whose *range* is the gameplay. It is also the number the
/// crowd's panic radius is set against — see
/// `inf_physics::d3::gameplay::PANIC_RADIUS_M`, which is deliberately smaller,
/// because hearing a shot and running from it are different distances.
pub const REPORT_MAX_M: f64 = 250.0;

/// The rolloff exponent of a gunshot. `1.0` is the inverse-distance default;
/// the model is [`DistanceModel::Inverse`].
pub const REPORT_ROLLOFF: f64 = 1.0;

/// **The `AudioSource` one round leaving a barrel plays** (wave WPN1).
///
/// One place, so the two hosts cannot describe the same shot differently — the
/// [`crate::venue::venue_music_source`] shape exactly, and for its reason: this
/// used to be the class of thing that gets written twice in two host-side loops
/// with a constant beside each copy.
///
/// # It is NOT an entity, and it is NOT occluded, and the two facts are one fact
///
/// A venue's music is an entity with an `AudioSource` on it, because a speaker
/// is a thing in a room. A gunshot is not: it happens at a point in the air for
/// three hundredths of a second and there is nothing left of it afterwards, so
/// it is a **command** built here and pushed straight onto the queue.
///
/// That decides the occlusion, rather than this flag doing it. The one-shot
/// occlusion pass in each host walks the queued `Play`s and looks each one's
/// source key up in the **Blueprint entity map**, then asks the world for that
/// entity's own `AudioSource::occlusion`. A gunshot's source key is its
/// shooter's, a shooter is not an audio emitter, and the lookup answers `None` —
/// so a report is unoccluded whatever this field says. It is set `false` because
/// that is what is true, not because setting it `true` would have done anything.
///
/// **The honest bound, stated:** a shot fired inside a building is heard at full
/// spatial gain by a listener outside it. Muffling it needs the *looping*
/// source's path — the per-step `SetOcclusion` the doorway model drives — and a
/// one-shot has no voice to keep re-evaluating. The fix is to give one-shots the
/// doorway model at `Play` time, which is a change to `portal_gain`'s call site
/// in both hosts and is named on this wave's carried list rather than done
/// quietly.
///
/// # The other honest bound: the SOURCE KEY is shared (WPN1 audit)
///
/// Each host queues this under `guid_source_key(hit.shooter)`, and
/// `AudioEngine::apply_play` is *one voice per source: replace any existing*.
/// That is exactly what makes a barrel one voice — and it is the **first** use
/// of that key namespace for a source which is not the entity's own emitter.
/// Every other `Play` in both hosts keys an entity's own `AudioSource`, so a
/// shooter which is *itself* an emitter (a character carrying an autoplay
/// `AudioSource` — nothing in the committed tree does) has its voice replaced by
/// the gunshot, and the autoplay walk starts a source **once**, so it never
/// comes back. Closing it is a salt on the report's key, which moves the audio
/// command stream both gates compare; carried rather than done inside an audit.
pub fn report_source() -> AudioSource {
    AudioSource {
        clip: Some(WEAPON_REPORT_CLIP),
        bus: REPORT_BUS.to_string(),
        volume: REPORT_VOLUME,
        pitch: 1.0,
        looping: false,
        spatial: true,
        min_distance: REPORT_MIN_M,
        max_distance: REPORT_MAX_M,
        distance_model: DistanceModel::Inverse,
        rolloff: REPORT_ROLLOFF,
        occlusion: false,
        autoplay: false,
    }
}

impl WeaponState {
    /// A full magazine of `def`.
    pub fn full(item_id: &str, def: &WeaponDef) -> Self {
        Self {
            item_id: item_id.to_string(),
            magazine: def.magazine,
            reserve: def.reserve,
            cooldown_s: 0.0,
            reload_left_s: 0.0,
            shots: 0,
            trigger_held: false,
        }
    }

    /// Whether a reload is running.
    pub fn reloading(&self) -> bool {
        self.reload_left_s > 0.0
    }
}

/// **How much of a weapon's recoil is still on it**, `[0, 1]` (wave WPN1).
///
/// # It is DERIVED, and that is the whole design
///
/// A recoil field on [`WeaponState`] would be a second copy of a number the
/// state already answers: `cooldown_s` is set to [`WeaponDef::fire_interval_s`]
/// the instant a round leaves and counts down to zero as the weapon cycles, so
/// *"how far through its own cycle is this weapon"* is already sim state, is
/// already in [`weapon_state_bytes`], and is already identical on both hosts.
/// The copy is the thing that drifts (`CrowdRecord::speed_of`'s own argument),
/// and it would have cost eight bytes an armed character in every trace in the
/// tree for a number that was already there.
///
/// So the recoil **is** the cycle: `1.0` on the step the shot is fired, falling
/// linearly to `0.0` as the action closes. At 600 rpm that is a tenth of a
/// second of kick per round and the next round arrives exactly as the last one
/// finishes, which is what an automatic weapon looks like; at a pistol's 400 rpm
/// it is a hundred and fifty milliseconds and a visible settle between shots.
///
/// **A reload is not a recoil.** `advance` runs the cooldown and the reload
/// clock independently, and this reads only the first.
///
/// Answers `0.0` for a weapon whose rate makes the interval non-finite, which is
/// the same refusal-as-a-value `fire_interval_s` makes.
pub fn recoil_fraction(def: &WeaponDef, state: &WeaponState) -> f64 {
    let interval = def.fire_interval_s();
    if !interval.is_finite() || interval <= 0.0 {
        return 0.0;
    }
    (state.cooldown_s / interval).clamp(0.0, 1.0)
}

/// **What a shooter's own instruments say** — the magazine and the reserve, as
/// one line (wave WPN1).
///
/// The whole of this wave's ammunition HUD, and it is in Ring 0 rather than in
/// the player's window for [`crate::vehicle::drive_readout`]'s reason verbatim:
/// the window cannot be tested and this can. What the window does is read the
/// two numbers and hand them here.
///
/// `"12 / 120"`. Four spaces would have matched the driver's readout and are
/// deliberately not used: a gear and a speed are two unrelated facts, so they are
/// spaced apart, and a magazine and a reserve are one fact counted twice, so they
/// are joined by the slash a player already reads as "of".
///
/// **A reloading weapon shows the magazine it HAS**, not the one it is about to
/// have. The alternative — blanking it, or showing the target — is a readout
/// that lies for the two seconds a player most wants it to be honest.
pub fn ammo_readout(magazine: u32, reserve: u32) -> String {
    format!("{magazine} / {reserve}")
}

/// **Is this clock a magazine, or is it a pair of hands?** (wave WPN1 audit).
///
/// [`WeaponState`] is the ammunition clock, and since this wave's melee it is
/// also what paces a punch — [`fist_def`] carries [`MAX_MAGAZINE`] rounds
/// precisely so a fist never runs out, and [`try_fire`] decrements it like any
/// other. So a character who has thrown one punch carries a perfectly valid
/// clock reading **9 999 / 10 000**, and anything that renders a `WeaponState`
/// without asking this question puts that on the screen.
///
/// It is in Ring 0 rather than in the player's window for [`ammo_readout`]'s own
/// reason verbatim — the window cannot be tested and this can — and it is one
/// predicate rather than two `item_id` comparisons, because the ammunition
/// readout and the reticle are the same question asked by two callers.
///
/// The fists are the only answer today; the rule is *"an ammunition clock a
/// player can count down"*, not *"anything with a magazine field"*.
pub fn carries_ammunition(state: &WeaponState) -> bool {
    state.item_id != FIST_ITEM
}

/// What pulling the trigger did. **A refusal is a value.**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireVerdict {
    /// A round left the barrel.
    Fired,
    /// The weapon is still cycling.
    Cooling,
    /// The magazine is empty.
    Empty,
    /// A reload is running.
    Reloading,
    /// The trigger is still down and the weapon is not automatic.
    NotReleased,
    /// Nothing is equipped, or what is equipped is not a weapon.
    NoWeapon,
}

/// **Pull the trigger.**
///
/// `held` is the trigger's *level*, not its edge: a semi-automatic weapon needs
/// to know the trigger came up, and an automatic one needs to know it is still
/// down. Reading the level here rather than an edge at the call site is what
/// lets both live in one rule — and it is the edges-consumed law's other half,
/// because an edge made in one mode must not fire in another.
pub fn try_fire(def: &WeaponDef, state: &mut WeaponState, held: bool) -> FireVerdict {
    let was_held = state.trigger_held;
    state.trigger_held = held;
    if !held {
        return FireVerdict::NotReleased;
    }
    if !def.automatic && was_held {
        return FireVerdict::NotReleased;
    }
    if state.reloading() {
        return FireVerdict::Reloading;
    }
    if state.cooldown_s > 0.0 {
        return FireVerdict::Cooling;
    }
    if state.magazine == 0 {
        return FireVerdict::Empty;
    }
    state.magazine -= 1;
    state.shots = state.shots.saturating_add(1);
    state.cooldown_s = def.fire_interval_s();
    FireVerdict::Fired
}

/// What asking for a reload did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReloadVerdict {
    /// It started.
    Started,
    /// One is already running.
    Already,
    /// The magazine is already full.
    Full,
    /// There is nothing left to load.
    NoReserve,
}

/// **Ask for a reload.**
pub fn try_reload(def: &WeaponDef, state: &mut WeaponState) -> ReloadVerdict {
    if state.reloading() {
        return ReloadVerdict::Already;
    }
    if state.magazine >= def.magazine {
        return ReloadVerdict::Full;
    }
    if state.reserve == 0 {
        return ReloadVerdict::NoReserve;
    }
    state.reload_left_s = def.reload_s.max(0.0);
    ReloadVerdict::Started
}

/// **Finish a reload** — what the animation's notify calls, and what the clock
/// calls when it runs out.
///
/// Moves as much of the reserve into the magazine as fits. Idempotent: calling
/// it on a weapon that is not reloading does nothing, so a notify that fires
/// twice loads one magazine.
pub fn finish_reload(def: &WeaponDef, state: &mut WeaponState) -> bool {
    if !state.reloading() {
        return false;
    }
    state.reload_left_s = 0.0;
    let want = def.magazine.saturating_sub(state.magazine);
    let take = want.min(state.reserve);
    state.magazine += take;
    state.reserve -= take;
    true
}

/// **One fixed step of a weapon's clocks.**
///
/// Returns whether the reload finished on this step *by the clock*, which is
/// what a host reports and what distinguishes the fallback path from the notify
/// one.
pub fn advance(def: &WeaponDef, state: &mut WeaponState, dt: f64) -> bool {
    if !dt.is_finite() || dt <= 0.0 {
        return false;
    }
    state.cooldown_s = (state.cooldown_s - dt).max(0.0);
    if state.reload_left_s > 0.0 {
        state.reload_left_s -= dt;
        if state.reload_left_s <= 0.0 {
            state.reload_left_s = f64::MIN_POSITIVE;
            finish_reload(def, state);
            return true;
        }
    }
    false
}

// ── the shot's direction ────────────────────────────────────────────────────

/// Two uniforms in `[0, 1)` from a counter, by splitmix64.
///
/// Counter-based, not stateful: the `n`-th shot of a weapon scatters the same
/// way in a replay, in a PIE preview and in a shipped build, and a gate can name
/// where a bullet went. Pure integer arithmetic, so it is bit-portable by
/// construction.
fn shot_uniforms(seed: u64, shot: u64) -> (f64, f64) {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    };
    let a = mix(seed ^ shot.wrapping_mul(0x2545_f491_4f6c_dd1d));
    let b = mix(a ^ 0x1234_5678_9abc_def0);
    // 53 bits into a double, the standard construction: exact and portable.
    let to_unit = |x: u64| ((x >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0);
    (to_unit(a), to_unit(b))
}

/// **Which way a character is pointing**, as a unit vector — the aim line
/// itself, before any spread is folded in (SK1c).
///
/// One door, because two things read it now: the shot leaves along it, and an
/// aiming character's hands are brought up onto it
/// (`inf_physics::d3::gameplay::aim_hold_point`). A second copy of these four
/// trig calls would be a hand that points somewhere the bullet does not.
///
/// `psin64`/`pcos64`, not `f64::sin`/`cos`: this reaches `pose_state_bytes`
/// through the hand pass and the replay trace through the shot, and std trig is
/// not bit-portable (the P14 law).
///
/// Non-finite inputs answer `+Z`, and pitch is clamped just short of vertical so
/// the horizontal component never collapses.
pub fn aim_forward(yaw_deg: f64, pitch_deg: f64) -> DVec3 {
    let yaw = if yaw_deg.is_finite() { yaw_deg } else { 0.0 };
    let pitch = if pitch_deg.is_finite() {
        pitch_deg.clamp(-89.9, 89.9)
    } else {
        0.0
    };
    let (sy, cy) = {
        let r = yaw.to_radians();
        (inf_math::psin64(r), inf_math::pcos64(r))
    };
    let (sp, cp) = {
        let r = pitch.to_radians();
        (inf_math::psin64(r), inf_math::pcos64(r))
    };
    DVec3::new(sy * cp, sp, cy * cp)
}

/// **Where a shot goes**, from an aim and a shot index.
///
/// `yaw_deg` is the compass yaw (`+Z` at zero, `+X` at `+90`) and `pitch_deg` is
/// positive upward — the movement runtime's own two numbers. The scatter is
/// uniform **over the cone's disc** (`theta = half · sqrt(u)`), not uniform in
/// the angle, because the second concentrates shots at the middle in a way a
/// player reads as the weapon being more accurate than its number says.
pub fn shot_direction(def: &WeaponDef, yaw_deg: f64, pitch_deg: f64, shot: u64) -> DVec3 {
    let yaw = if yaw_deg.is_finite() { yaw_deg } else { 0.0 };
    let pitch = if pitch_deg.is_finite() {
        pitch_deg.clamp(-89.9, 89.9)
    } else {
        0.0
    };
    let (sy, cy) = {
        let r = yaw.to_radians();
        (inf_math::psin64(r), inf_math::pcos64(r))
    };
    let forward = aim_forward(yaw, pitch);
    let half = if def.spread_deg.is_finite() {
        (def.spread_deg * 0.5).clamp(0.0, MAX_SPREAD_DEG)
    } else {
        0.0
    };
    if half <= 0.0 {
        return forward.normalize_or_zero();
    }
    // The aim frame: `right` is horizontal by construction, `up` completes it.
    let right = DVec3::new(cy, 0.0, -sy);
    let up = right.cross(forward);
    let (u1, u2) = shot_uniforms(def.spread_seed, shot);
    let theta = (half * u1.sqrt()).to_radians();
    let phi = (u2 * 360.0).to_radians();
    let (st, ct) = (inf_math::psin64(theta), inf_math::pcos64(theta));
    let (sph, cph) = (inf_math::psin64(phi), inf_math::pcos64(phi));
    (forward * ct + (right * cph + up * sph) * st).normalize_or_zero()
}

// ── health ──────────────────────────────────────────────────────────────────

/// **What a body can still absorb** — a runtime component, in joules.
///
/// See the module header for why the unit is joules and not hit points.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Health {
    /// What is left, joules.
    pub joules: f64,
    /// What it started with, joules — the denominator of a UI bar.
    pub capacity_j: f64,
    /// Whether it has stopped working. A latch, so the death handoff fires
    /// exactly once however many bullets arrive on the same step.
    pub dead: bool,
}

impl Default for Health {
    fn default() -> Self {
        Self::new(DEFAULT_VITALITY_J)
    }
}

impl Health {
    /// A body with `capacity_j` joules to give.
    pub fn new(capacity_j: f64) -> Self {
        let c = if capacity_j.is_finite() && capacity_j > 0.0 {
            capacity_j
        } else {
            DEFAULT_VITALITY_J
        };
        Self {
            joules: c,
            capacity_j: c,
            dead: false,
        }
    }

    /// What is left, `[0, 1]`.
    pub fn fraction(&self) -> f64 {
        if self.capacity_j <= 0.0 {
            return 0.0;
        }
        (self.joules / self.capacity_j).clamp(0.0, 1.0)
    }
}

/// What a hit did to a body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HealthReport {
    /// Joules the body actually absorbed — never more than it had.
    pub absorbed_j: f64,
    /// Whether this hit was the one that killed it.
    pub killed: bool,
    /// Whether it was already dead. A hit on a corpse absorbs nothing.
    pub was_dead: bool,
}

/// **Take `energy_j` off a body.**
///
/// The character-side twin of `PhysicsBridge3D::runtime_destruct`, and
/// deliberately the same shape: energy in, a report out, refusals as values, and
/// **no banking** — what a body cannot absorb is spent, not stored.
pub fn damage(health: &mut Health, energy_j: f64) -> HealthReport {
    if health.dead {
        return HealthReport {
            absorbed_j: 0.0,
            killed: false,
            was_dead: true,
        };
    }
    let e = if energy_j.is_finite() && energy_j > 0.0 {
        energy_j
    } else {
        0.0
    };
    let absorbed = e.min(health.joules.max(0.0));
    health.joules -= absorbed;
    let killed = health.joules <= 0.0 && e > 0.0;
    if killed {
        health.joules = 0.0;
        health.dead = true;
    }
    HealthReport {
        absorbed_j: absorbed,
        killed,
        was_dead: false,
    }
}

/// Give `entity` a body worth `capacity_j` joules.
pub fn give_health(world: &mut EcsWorld, guid: Uuid, capacity_j: f64) -> bool {
    let Some(entity) = world.entity_of(guid) else {
        return false;
    };
    world
        .world_mut()
        .entity_mut(entity)
        .insert(Health::new(capacity_j));
    true
}

/// **This body has been handed to the ragdoll** — a runtime marker, and the
/// latch that makes the handoff happen exactly once.
///
/// A `bool` on [`Health`] would have done, and it would have been the wrong
/// shape: "this body has stopped working" and "something has already dealt with
/// that" are two facts, and the second one is the gameplay step's business
/// rather than the damage model's.
///
/// **It is a latch and not a mode test, and the difference is measured.** The
/// obvious rule — "dead and not currently a ragdoll" — re-fires on every step a
/// ragdoll is not running, and a character with no skeleton never gets one: the
/// rig arrives through a one-step command queue that the animation side answers,
/// so a headless body, an NPC with no `.inf_skel` and every level committed
/// before I6 would be handed over again, and again, for ever. Measured on the
/// first fixture that killed something: **two handoffs in thirty steps** where
/// there should be one, and it would have been thirty in three hundred.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Downed;

/// **Every character that has stopped working and has not been handed over
/// yet**, in `Guid` order.
///
/// Here rather than at the gameplay step because this crate is the only one that
/// may name `bevy_ecs` (the facade rule). `O(bodies)`, and `O(1)` on a level
/// where nothing can be hurt.
pub fn newly_dead(world: &EcsWorld) -> Vec<Uuid> {
    let w = world.world();
    let Some(mut q) =
        w.try_query_filtered::<(&Guid, &Health, bevy_ecs::prelude::Entity), With<Health>>()
    else {
        return Vec::new();
    };
    // **The latch is read per entity, not as a `Without<Downed>` filter.**
    // `try_query_filtered` answers `None` when any component it names has never
    // been inserted in this world — which is the `O(1)` fast path the whole
    // codebase relies on, and which for a *negative* filter is exactly backwards:
    // a world where nobody has died yet has no `Downed` anywhere, so the query
    // answered `None` and **nothing was ever handed to the ragdoll at all**.
    // Measured: 0 handoffs where there should be 1.
    let mut out: Vec<Uuid> = q
        .iter(w)
        .filter(|(_, h, e)| h.dead && w.get::<Downed>(*e).is_none())
        .map(|(g, _, _)| g.0)
        .collect();
    out.sort_unstable();
    out
}

/// Latch the handoff. `false` for an entity that is not there.
pub fn mark_downed(world: &mut EcsWorld, guid: Uuid) -> bool {
    let Some(entity) = world.entity_of(guid) else {
        return false;
    };
    world.world_mut().entity_mut(entity).insert(Downed);
    true
}

/// Whether this body has already been handed over.
pub fn is_downed(world: &EcsWorld, guid: Uuid) -> bool {
    world
        .entity_of(guid)
        .is_some_and(|e| world.world().get::<Downed>(e).is_some())
}

/// **Every body on the ground**, in `Guid` order — a CENSUS, not an event
/// (wave EMS2).
///
/// [`newly_dead`] is a one-shot: it answers the bodies that have not been handed
/// to the ragdoll *yet*, and it answers each of them exactly once because the
/// pass that consumes it latches [`Downed`]. That is right for a handoff and
/// wrong for a dispatcher, which asks *"is there anybody who needs an
/// ambulance"* — a question whose answer must stay `yes` while the body is still
/// lying there. Reading `newly_dead` would have meant an ambulance was only ever
/// called on the single step somebody died, and missing that step (a saturated
/// incident table, a station with no free unit) meant nobody ever came.
///
/// So this is the latch read the other way round, and the two doors are one
/// component apart rather than two rules.
///
/// `O(bodies)`, and `O(1)` on a level where nothing has been hurt — the negative
/// filter is read per entity for [`newly_dead`]'s own measured reason.
pub fn downed(world: &EcsWorld) -> Vec<Uuid> {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<(&Guid, bevy_ecs::prelude::Entity), With<Health>>()
    else {
        return Vec::new();
    };
    let mut out: Vec<Uuid> = q
        .iter(w)
        .filter(|(_, e)| w.get::<Downed>(*e).is_some())
        .map(|(g, _)| g.0)
        .collect();
    out.sort_unstable();
    out
}

/// **Spend energy on one body, by `Guid`** — the world-level door beside
/// [`damage`]'s component-level one.
///
/// `None` when the entity does not exist or has no [`Health`], which is the
/// honest answer to "how much did that hurt" for something that cannot be hurt.
///
/// It does **not** mark the body [`Downed`]; the fixed step's own pass does that
/// from [`newly_dead`], and doing it here would put the transition in two places
/// that have to agree about when it happens.
///
/// Added in wave SCRIPT2 for the `health.damage` verb, and immediately given a
/// second caller: `inf_physics::d3::gameplay::apply_hit` did this by hand, and
/// two spellings of "take joules out of a body" is the shape this house has paid
/// for repeatedly.
pub fn damage_entity(world: &mut EcsWorld, guid: Uuid, energy_j: f64) -> Option<HealthReport> {
    let entity = world.entity_of(guid)?;
    let mut health = world.world_mut().get_mut::<Health>(entity)?;
    Some(damage(&mut health, energy_j))
}

/// Read one body's health.
pub fn health_of(world: &EcsWorld, guid: Uuid) -> Option<Health> {
    let entity = world.entity_of(guid)?;
    world.world().get::<Health>(entity).copied()
}

/// **The trace bytes for health**, in `Guid` order.
///
/// Empty on a level with nothing that can be hurt, which keeps every pre-I6
/// trace byte-identical.
pub fn health_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<(&Guid, &Health), With<Health>>() else {
        return Vec::new();
    };
    let mut rows: Vec<(Uuid, Health)> = q.iter(w).map(|(g, h)| (g.0, *h)).collect();
    if rows.is_empty() {
        return Vec::new();
    }
    rows.sort_by_key(|(g, _)| *g);
    let mut out = Vec::with_capacity(rows.len() * 33);
    for (guid, h) in rows {
        out.extend_from_slice(guid.as_bytes());
        out.extend_from_slice(&h.joules.to_bits().to_le_bytes());
        out.extend_from_slice(&h.capacity_j.to_bits().to_le_bytes());
        out.push(u8::from(h.dead));
    }
    out
}

/// **The weapon trace bytes**, in `Guid` order — the ammunition clock is sim
/// state and a PIE-versus-shipping gate has to see it.
pub fn weapon_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let w = world.world();
    let Some(mut q) = w.try_query_filtered::<(&Guid, &WeaponState), With<WeaponState>>() else {
        return Vec::new();
    };
    let mut rows: Vec<(Uuid, &WeaponState)> = q.iter(w).map(|(g, s)| (g.0, s)).collect();
    if rows.is_empty() {
        return Vec::new();
    }
    rows.sort_by_key(|(g, _)| *g);
    let mut out = Vec::new();
    for (guid, s) in rows {
        out.extend_from_slice(guid.as_bytes());
        out.extend_from_slice(&(s.item_id.len() as u32).to_le_bytes());
        out.extend_from_slice(s.item_id.as_bytes());
        out.extend_from_slice(&s.magazine.to_le_bytes());
        out.extend_from_slice(&s.reserve.to_le_bytes());
        out.extend_from_slice(&s.cooldown_s.to_bits().to_le_bytes());
        out.extend_from_slice(&s.reload_left_s.to_bits().to_le_bytes());
        out.extend_from_slice(&s.shots.to_le_bytes());
        out.push(u8::from(s.trigger_held));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1.0 / 60.0;

    /// **The tuning door is by name and every refusal is a value** — the
    /// `VehicleTuning` arm, at a weapon.
    #[test]
    fn the_tuning_door_is_by_name_and_refuses_as_a_value() {
        let mut d = WeaponDef::default();
        for name in WeaponDef::names() {
            assert!(d.set(name, 1.0), "the door does not know {name}");
            assert!(!d.set(name, f64::NAN), "{name} took a NaN");
            assert!(!d.set(name, f64::INFINITY), "{name} took an infinity");
        }
        assert!(!d.set("no_such_knob", 1.0));
        // Sorted and complete: a name the door knows and the list does not is
        // a knob no UI can offer.
        let mut sorted = WeaponDef::names().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, WeaponDef::names());
        // Clamps rather than refusals, and they bite.
        let mut e = WeaponDef::default();
        assert!(e.set("rounds_per_minute", 1.0e9));
        assert_eq!(e.rounds_per_minute, MAX_RPM);
        assert!(e.set("spread_deg", -5.0));
        assert_eq!(e.spread_deg, 0.0);
        assert!(e.set("magazine", 0.0));
        assert_eq!(e.magazine, 1);
        // The three non-numeric knobs come across the same door.
        assert!(e.set("projectile", 1.0));
        assert_eq!(e.kind, ShotKind::Projectile);
        assert!(e.set("automatic", 0.0));
        assert!(!e.automatic);
    }

    /// **The fire rate is a clock**, and a semi-automatic weapon fires once per
    /// press.
    #[test]
    fn a_semi_automatic_weapon_fires_once_per_press_and_an_automatic_one_does_not() {
        let d = WeaponDef {
            automatic: false,
            rounds_per_minute: 600.0,
            ..Default::default()
        };
        assert!((d.fire_interval_s() - 0.1).abs() < 1e-12);
        let mut s = WeaponState::full("rifle", &d);
        assert_eq!(try_fire(&d, &mut s, true), FireVerdict::Fired);
        assert_eq!(s.magazine, 29);
        // Held: refused, and NOT because of the cooldown — the trigger is the
        // reason, and the two must not be confused.
        assert_eq!(try_fire(&d, &mut s, true), FireVerdict::NotReleased);
        // Release, wait out the cooldown, press again.
        assert_eq!(try_fire(&d, &mut s, false), FireVerdict::NotReleased);
        assert_eq!(try_fire(&d, &mut s, true), FireVerdict::Cooling);
        for _ in 0..7 {
            advance(&d, &mut s, DT);
        }
        assert_eq!(try_fire(&d, &mut s, false), FireVerdict::NotReleased);
        assert_eq!(try_fire(&d, &mut s, true), FireVerdict::Fired);
        assert_eq!(s.magazine, 28);
        // Automatic: held is enough, and the rate is what bounds it.
        let a = WeaponDef {
            automatic: true,
            ..Default::default()
        };
        let mut t = WeaponState::full("rifle", &a);
        let mut fired = 0;
        for _ in 0..60 {
            if try_fire(&a, &mut t, true) == FireVerdict::Fired {
                fired += 1;
            }
            advance(&a, &mut t, DT);
        }
        println!("600 rpm held for one second fired {fired} rounds");
        assert!(
            (9..=11).contains(&fired),
            "{fired} rounds in a second at 600 rpm"
        );
    }

    /// **The reload has two finishers and both work** — the notify and the
    /// clock.
    #[test]
    fn a_reload_finishes_on_its_notify_or_on_its_clock() {
        let d = WeaponDef::default();
        let mut s = WeaponState::full("rifle", &d);
        assert_eq!(try_reload(&d, &mut s), ReloadVerdict::Full);
        // Spend some rounds.
        for _ in 0..5 {
            s.magazine -= 1;
        }
        assert_eq!(try_reload(&d, &mut s), ReloadVerdict::Started);
        assert_eq!(try_reload(&d, &mut s), ReloadVerdict::Already);
        // Firing is refused while it runs, and for the reload's reason.
        assert_eq!(try_fire(&d, &mut s, true), FireVerdict::Reloading);
        // The NOTIFY path: it finishes early and the clock stops.
        assert!(finish_reload(&d, &mut s));
        assert_eq!(s.magazine, d.magazine);
        assert_eq!(s.reserve, d.reserve - 5);
        assert!(!s.reloading());
        // Idempotent: a notify that fires twice loads one magazine.
        assert!(!finish_reload(&d, &mut s));
        assert_eq!(s.reserve, d.reserve - 5);
        // The CLOCK path, on a weapon nothing is animating.
        let mut c = WeaponState::full("rifle", &d);
        c.magazine = 0;
        assert_eq!(try_reload(&d, &mut c), ReloadVerdict::Started);
        let mut steps = 0;
        while c.reloading() && steps < 600 {
            advance(&d, &mut c, DT);
            steps += 1;
        }
        println!("a {} s reload took {steps} fixed steps", d.reload_s);
        assert_eq!(c.magazine, d.magazine);
        assert!((steps as f64 * DT - d.reload_s).abs() < 0.02);
        // Nothing left to load is a refusal, not an empty reload.
        let mut z = WeaponState::full("rifle", &d);
        z.magazine = 0;
        z.reserve = 0;
        assert_eq!(try_reload(&d, &mut z), ReloadVerdict::NoReserve);
        // An empty magazine is a distinct refusal from a cooling one.
        assert_eq!(try_fire(&d, &mut z, true), FireVerdict::Empty);
    }

    /// **Spread is a pure function of the shot's index**, and it really
    /// scatters.
    #[test]
    fn a_shots_direction_is_deterministic_and_lands_inside_its_own_cone() {
        let d = WeaponDef {
            spread_deg: 4.0,
            ..Default::default()
        };
        let centre = shot_direction(
            &WeaponDef {
                spread_deg: 0.0,
                ..d
            },
            30.0,
            10.0,
            1,
        );
        let mut worst = 0.0_f64;
        let mut best = 180.0_f64;
        for shot in 0..500u64 {
            let dir = shot_direction(&d, 30.0, 10.0, shot);
            assert!(dir.is_finite() && (dir.length() - 1.0).abs() < 1e-9);
            // The same shot index always answers the same direction.
            assert_eq!(dir, shot_direction(&d, 30.0, 10.0, shot));
            let deg = inf_math::pacos64(dir.dot(centre).clamp(-1.0, 1.0)).to_degrees();
            worst = worst.max(deg);
            best = best.min(deg);
        }
        println!(
            "500 shots of a 4.0 degree cone landed between {best} and {worst} degrees off centre"
        );
        assert!(
            worst <= 2.0 + 1e-3,
            "a shot left its own cone at {worst} degrees"
        );
        assert!(worst > 1.8, "the cone is not being filled");
        assert!(best < 0.5, "no shot went near the middle");
        // A weapon with no spread fires down the aim, and two weapons with
        // different seeds do not fire the same shot.
        let tight = WeaponDef {
            spread_deg: 0.0,
            ..d
        };
        assert_eq!(shot_direction(&tight, 30.0, 10.0, 0), centre);
        let other = WeaponDef {
            spread_seed: 99,
            ..d
        };
        assert_ne!(
            shot_direction(&d, 30.0, 10.0, 0),
            shot_direction(&other, 30.0, 10.0, 0)
        );
        // Hostile aim is a refusal, not a NaN direction.
        assert!(shot_direction(&d, f64::NAN, f64::NAN, 0).is_finite());
    }

    /// **Health is joules, damage does not bank, and a corpse absorbs nothing.**
    #[test]
    fn a_body_absorbs_joules_until_it_stops_and_then_absorbs_none() {
        let mut h = Health::new(DEFAULT_VITALITY_J);
        assert!((h.fraction() - 1.0).abs() < 1e-12);
        // A rifle round is 1 700 J against a 2 000 J body.
        let r = damage(&mut h, WeaponDef::default().damage_j);
        assert!((r.absorbed_j - 1700.0).abs() < 1e-12);
        assert!(!r.killed && !r.was_dead);
        println!("one rifle round left {} J of {}", h.joules, h.capacity_j);
        assert!((h.joules - 300.0).abs() < 1e-12);
        assert!((h.fraction() - 0.15).abs() < 1e-12);
        // The second one stops it, and absorbs only what was left.
        let r = damage(&mut h, 1700.0);
        assert!(
            (r.absorbed_j - 300.0).abs() < 1e-12,
            "a corpse over-absorbed"
        );
        assert!(r.killed);
        assert!(h.dead && h.joules == 0.0);
        // A third absorbs nothing and does not kill it twice — the latch.
        let r = damage(&mut h, 1700.0);
        assert_eq!(r.absorbed_j, 0.0);
        assert!(!r.killed && r.was_dead);
        // Hostile energy is a refusal.
        let mut g = Health::new(10.0);
        assert_eq!(damage(&mut g, f64::NAN).absorbed_j, 0.0);
        assert_eq!(damage(&mut g, -5.0).absorbed_j, 0.0);
        assert!(!g.dead, "a NaN killed something");
        // A hostile capacity takes the default rather than becoming a body
        // nothing can hurt.
        assert_eq!(Health::new(f64::NAN).capacity_j, DEFAULT_VITALITY_J);
        assert_eq!(Health::new(0.0).capacity_j, DEFAULT_VITALITY_J);
    }

    /// **A weapon TOML is the item TOML's own sub-table**, and a malformed one
    /// is an error rather than blanks.
    #[test]
    fn a_weapon_is_read_out_of_its_items_table_and_refuses_nonsense() {
        let doc: toml::Value = toml::from_str(
            r#"
[rifle.weapon]
kind = "hitscan"
damage_j = 1900.0
rounds_per_minute = 750
magazine = 30
automatic = true
"#,
        )
        .expect("a document");
        let t = doc["rifle"].as_table().expect("a table");
        let def = WeaponDef::from_toml_table(t)
            .expect("a weapon")
            .expect("it is a weapon");
        assert_eq!(def.kind, ShotKind::Hitscan);
        assert!((def.damage_j - 1900.0).abs() < 1e-12);
        assert!((def.rounds_per_minute - 750.0).abs() < 1e-12);
        assert!(def.automatic);
        // **The third kind reads out of the same table** (wave WPN1), and a
        // melee reach is bounded by a different number from a rifle's range.
        let doc: toml::Value = toml::from_str(
            "[bat.weapon]\nkind = \"melee\"\ndamage_j = 900.0\nrange_m = 2.0\n\
             melee_arc_deg = 120.0\n",
        )
        .expect("a document");
        let bat = WeaponDef::from_toml_table(doc["bat"].as_table().expect("a table"))
            .expect("a weapon")
            .expect("it is a weapon");
        assert!(bat.is_melee());
        assert!((bat.reach_m() - 2.0).abs() < 1e-12);
        assert!((bat.melee_arc_deg - 120.0).abs() < 1e-12);
        // A melee reach is clamped by `MAX_MELEE_REACH_M` and a rifle's range by
        // `MAX_RANGE_M`, which differ by four orders of magnitude — a call site
        // that used one bound for both would let a 20 km punch through.
        let long = WeaponDef {
            range_m: 4000.0,
            ..bat
        };
        assert_eq!(long.reach_m(), MAX_MELEE_REACH_M);
        let rifle = WeaponDef {
            range_m: 4000.0,
            ..WeaponDef::default()
        };
        assert!(!rifle.is_melee());
        assert!((rifle.reach_m() - 4000.0).abs() < 1e-12);
        // The fists are a melee definition and they never run out.
        let fist = fist_def();
        assert!(fist.is_melee() && !fist.automatic);
        assert!((fist.reach_m() - FIST_REACH_M).abs() < 1e-12);
        assert!((fist.damage_j - FIST_DAMAGE_J).abs() < 1e-12);
        assert_eq!(
            try_reload(&fist, &mut WeaponState::full(FIST_ITEM, &fist)),
            ReloadVerdict::Full,
            "a pair of hands can be reloaded"
        );
        // No sub-table is not a weapon.
        let plain: toml::Value = toml::from_str("[bandage]\nlabel = \"x\"\n").expect("a document");
        assert!(
            WeaponDef::from_toml_table(plain["bandage"].as_table().expect("a table"))
                .expect("no error")
                .is_none()
        );
        // An unknown key and an unknown kind are errors, not silent blanks.
        let bad: toml::Value = toml::from_str("[x.weapon]\nwobble = 3\n").expect("a document");
        assert!(WeaponDef::from_toml_table(bad["x"].as_table().expect("a table")).is_err());
        let worse: toml::Value =
            toml::from_str("[x.weapon]\nkind = \"beam\"\n").expect("a document");
        assert!(WeaponDef::from_toml_table(worse["x"].as_table().expect("a table")).is_err());
    }

    /// **The gunshot is one description, and it is a ONE-SHOT that is not
    /// occluded** (wave WPN1).
    ///
    /// Every field here is load-bearing somewhere and each one is asserted where
    /// it is: `looping: false` because a report that looped would be a siren;
    /// `spatial: true` because a shot has a place and that place is the whole of
    /// what a listener learns from it; `occlusion: false` because the one-shot
    /// occlusion pass keys on the Blueprint entity map and a gunshot has no
    /// entity to be found in it — see [`report_source`]'s own doc.
    #[test]
    fn a_gunshot_is_a_placed_one_shot_that_carries_much_further_than_anything_else() {
        let s = report_source();
        assert_eq!(s.clip, Some(WEAPON_REPORT_CLIP));
        assert_eq!(s.bus, REPORT_BUS);
        assert!(!s.looping, "a gunshot that loops is a siren");
        assert!(
            s.spatial,
            "a gunshot with no place tells a listener nothing"
        );
        assert!(
            !s.autoplay,
            "a report is issued by a trigger, not by a level"
        );
        assert!(
            !s.occlusion,
            "the one-shot occlusion pass looks its source up in the BLUEPRINT \
             entity map and a gunshot has no entity there, so `true` here would \
             be a claim the engine cannot honour"
        );
        // The reach is the claim: a gunshot is heard streets away, which is what
        // makes it the one emitter whose range is the gameplay.
        println!(
            "a gunshot reaches {} m; a venue's music reaches {} m",
            s.max_distance,
            crate::venue::VENUE_MUSIC_MAX_M
        );
        assert!(
            s.max_distance > crate::venue::VENUE_MUSIC_MAX_M * 4.0,
            "a gunshot does not carry meaningfully further than a nightclub"
        );
        assert!(s.min_distance < s.max_distance && s.min_distance > 0.0);
    }

    /// **A stagger is a PROPORTION of what the body had left** (wave WPN1), so
    /// the same punch that bounces off a fresh body drops a hurt one.
    #[test]
    fn a_blow_staggers_by_what_it_takes_of_what_was_left() {
        let rifle = WeaponDef::default().damage_j;
        // A rifle round is 0.85 of a fresh default body: it drops it.
        assert!(is_staggering(rifle, DEFAULT_VITALITY_J));
        // …and a tenth of a big one does not.
        assert!(!is_staggering(500.0, 5000.0));
        // THE POINT: the same 500 J on a body with 1 200 J left is 0.42 and
        // does. A threshold measured against the CAPACITY could not tell these
        // two apart, and a beating would feel identical from beginning to end.
        assert!(is_staggering(500.0, 1200.0));
        // The boundary is inclusive, so a blow worth exactly a third counts.
        assert!(is_staggering(1000.0, 3000.0));
        assert!(!is_staggering(999.0, 3000.0));
        // Refusals are values: a corpse and a NaN stagger nothing.
        assert!(!is_staggering(rifle, 0.0));
        assert!(!is_staggering(f64::NAN, 100.0));
        assert!(!is_staggering(10.0, f64::INFINITY));
    }

    /// **The recoil IS the weapon's own cycle** (wave WPN1) — full on the step
    /// the round leaves, gone by the time the next one may.
    ///
    /// The mutation this kills: a recoil derived from `shots` alone would be
    /// `1.0` for ever after the first round, and a recoil derived from
    /// `reload_left_s` would fire on a reload and never on a shot.
    #[test]
    fn the_recoil_is_full_on_the_shot_and_gone_when_the_action_closes() {
        let d = WeaponDef::default();
        let mut s = WeaponState::full("rifle", &d);
        assert_eq!(recoil_fraction(&d, &s), 0.0, "a rested weapon has recoil");
        assert_eq!(try_fire(&d, &mut s, true), FireVerdict::Fired);
        assert!(
            (recoil_fraction(&d, &s) - 1.0).abs() < 1e-12,
            "the shot did not kick: {}",
            recoil_fraction(&d, &s)
        );
        // It falls, monotonically, and is spent exactly when the weapon may fire
        // again — which is the property that makes it the cycle rather than a
        // second clock beside it.
        let mut last = 1.0_f64;
        let mut steps = 0;
        while recoil_fraction(&d, &s) > 0.0 && steps < 600 {
            advance(&d, &mut s, DT);
            let now = recoil_fraction(&d, &s);
            assert!(now <= last, "the recoil went back up: {last} -> {now}");
            last = now;
            steps += 1;
        }
        println!(
            "a {} rpm weapon's kick lasted {steps} steps ({:.4} s)",
            d.rounds_per_minute,
            steps as f64 * DT
        );
        assert_eq!(try_fire(&d, &mut s, true), FireVerdict::Fired);
        // A RELOAD is not a recoil: the two clocks are independent and this
        // reads one of them.
        let mut r = WeaponState::full("rifle", &d);
        r.magazine = 1;
        assert_eq!(try_reload(&d, &mut r), ReloadVerdict::Started);
        assert!(r.reloading());
        assert_eq!(
            recoil_fraction(&d, &r),
            0.0,
            "a reload registered as a recoil"
        );
        // A weapon whose rate cannot make an interval refuses as a value.
        let dead = WeaponDef {
            rounds_per_minute: 0.0,
            ..d
        };
        assert_eq!(recoil_fraction(&dead, &s), 0.0);
    }

    /// **The ammunition readout is the two numbers and nothing else** (wave
    /// WPN1) — and it does not lie during a reload.
    #[test]
    fn the_ammo_readout_says_the_magazine_it_has_and_the_reserve_behind_it() {
        let d = WeaponDef::default();
        let mut s = WeaponState::full("rifle", &d);
        assert_eq!(ammo_readout(s.magazine, s.reserve), "30 / 120");
        for _ in 0..18 {
            s.magazine -= 1;
        }
        assert_eq!(ammo_readout(s.magazine, s.reserve), "12 / 120");
        // **A reload in flight reads what the magazine HOLDS**, which is the
        // number that decides whether the next trigger pull does anything.
        assert_eq!(try_reload(&d, &mut s), ReloadVerdict::Started);
        assert_eq!(ammo_readout(s.magazine, s.reserve), "12 / 120");
        assert!(finish_reload(&d, &mut s));
        assert_eq!(ammo_readout(s.magazine, s.reserve), "30 / 102");
        // Empty is a readout, not a blank: a player has to be able to tell
        // "no rounds" from "no weapon", and the second one draws nothing at all.
        assert_eq!(ammo_readout(0, 0), "0 / 0");
    }

    /// **A PAIR OF HANDS IS NOT A MAGAZINE** (wave WPN1 audit).
    ///
    /// The defect this closes was on the screen: `fist_def` carries
    /// [`MAX_MAGAZINE`] rounds so a fist never runs out, `try_fire` decrements
    /// it like any other clock, and the HUD read the clock it found — so one
    /// punch put **"9999 / 10000"** at the bottom of the viewport and left it
    /// there for the rest of the level, on a character holding nothing.
    ///
    /// Measured as the string, because the string is what a player sees.
    #[test]
    fn a_pair_of_hands_is_not_an_ammunition_clock() {
        let rifle = WeaponDef::default();
        let armed = WeaponState::full("rifle", &rifle);
        assert!(carries_ammunition(&armed), "a rifle has no magazine");

        let fists = fist_def();
        let mut hands = WeaponState::full(FIST_ITEM, &fists);
        assert!(
            !carries_ammunition(&hands),
            "an empty hand is being counted as ammunition"
        );
        // …and this is what it would have said. One punch, and the clock is a
        // perfectly valid one — which is the whole reason the question has to
        // be asked about the ITEM rather than about the numbers.
        assert_eq!(try_fire(&fists, &mut hands, true), FireVerdict::Fired);
        println!(
            "one punch leaves the fists' clock reading \"{}\"",
            ammo_readout(hands.magazine, hands.reserve)
        );
        assert_eq!(
            ammo_readout(hands.magazine, hands.reserve),
            format!("{} / {}", MAX_MAGAZINE - 1, MAX_MAGAZINE),
            "the fists' clock stopped being the thing this predicate exists for"
        );
        assert!(!carries_ammunition(&hands));
    }

    /// **The traces are empty until something exists**, and move when it does.
    #[test]
    fn the_combat_traces_cost_a_level_without_combat_nothing() {
        let mut w = EcsWorld::new();
        assert!(health_state_bytes(&w).is_empty());
        assert!(weapon_state_bytes(&w).is_empty());
        let hero = Uuid::from_u128(1);
        w.spawn_with_guid(hero, "Hero", None);
        assert!(give_health(&mut w, hero, 500.0));
        let before = health_state_bytes(&w);
        assert_eq!(before.len(), 16 + 8 + 8 + 1);
        let e = w.entity_of(hero).expect("the hero");
        {
            let mut h = w.world_mut().get_mut::<Health>(e).expect("health");
            damage(&mut h, 100.0);
        }
        assert_ne!(before, health_state_bytes(&w));
        w.world_mut()
            .entity_mut(e)
            .insert(WeaponState::full("rifle", &WeaponDef::default()));
        assert!(!weapon_state_bytes(&w).is_empty());
        assert_eq!(health_of(&w, hero).expect("health").joules, 400.0);
    }
}
