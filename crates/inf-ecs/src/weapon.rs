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

/// How a shot travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ShotKind {
    /// Instant: a ray, resolved the step the trigger is pulled.
    #[default]
    Hitscan,
    /// A body in flight, resolved when it arrives.
    Projectile,
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
        }
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

/// The animation trigger a door kick arms — a P29-style one-shot, reached
/// through `inf_ecs::anim_bridge::set_anim_trigger` exactly as the ragdoll's is.
pub const KICK_TRIGGER: &str = "door_kick";

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
