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

/// **The steepest quadratic drag a round may carry**, 1/m — the bound on
/// [`WeaponDef::drag_k`] (wave WPN2a).
///
/// One hundredth. The doc's rifle figure is 0.0003–0.0004 /m and this is
/// twenty-five times the top of it, which is a bound on hostile content rather
/// than a design limit — `MAX_MUZZLE_FORWARD_M`'s own rule. At 0.01 /m a 900 m/s
/// round loses 8 100 m/s² and stops inside a metre, which is as far as a
/// coefficient can be pushed before the flight is not a flight.
pub const MAX_DRAG_K: f64 = 0.01;

/// **The largest head bonus a weapon may carry** — the bound on
/// [`WeaponDef::headshot_mult`] (wave WPN2a).
///
/// Ten. The steepest ratio in the doc's own tables is the Barrett M82's
/// 250/110 = 2.27; ten is four times that and is, again, a bound on content.
/// The **floor** is 1.0: a head is never worth less than a chest, and a
/// multiplier below one would make a headshot a way to survive.
pub const MAX_HEADSHOT_MULT: f64 = 10.0;

/// **The slowest a weapon may make its carrier**, as a multiplier on the
/// character's own speeds — the floor on [`WeaponDef::move_speed_mult`].
///
/// A quarter. The doc's slowest row is the Javelin at 0.72; a quarter is a
/// bound on hostile content, and zero is deliberately not it — a weapon that
/// pinned a character to the floor would look like a broken control rather than
/// a heavy gun.
pub const MIN_MOVE_SPEED_MULT: f64 = 0.25;

/// **What one hit point of the research doc's tables is worth**, joules.
///
/// Twenty. [`DEFAULT_VITALITY_J`] is 2 000 J and the doc's tables assume a
/// **100 HP** body, so the conversion is forced rather than chosen — and the
/// arithmetic checks out against a row that was authored before anybody thought
/// about it: island wave I6's pistol is 600 J, and the doc's Glock 17 is 30 HP.
/// `docs/memos/p22-strength.md` §1's refusal of a per-weapon conversion table
/// stands: this is **one** number for the whole registry, applied once when the
/// rows are authored, and nothing in the engine reads a hit point.
pub const JOULES_PER_HIT_POINT: f64 = DEFAULT_VITALITY_J / 100.0;

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

/// **What KIND of gun a weapon is** (wave WPN2c) — the seven the research doc's
/// own tables enumerate, and the one thing a gunshot's *sound* is chosen by.
///
/// # Why the sound needs a class and the ballistics did not
///
/// Wave WPN2a authored eighty-five rows without one, because every number a
/// round needs is on the row itself: a muzzle velocity is a muzzle velocity and
/// the class is a way of grouping the authoring, not a thing the flight reads.
/// A REPORT is the other way round. There are five clips per class
/// ([`ReportClip`]) and thirty-five in the tree, and a shot has to name one —
/// so the grouping stops being presentation and becomes the key.
///
/// P22 §5's refusal of a per-weapon *clip* slot is untouched by this and is in
/// fact what forces the shape: a weapon does not name its own gunshot file, it
/// names what kind of gun it is, and the engine owns the sound of each kind.
/// Adding a weapon is still a row of numbers; adding a *class* is a wave.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WeaponClass {
    /// A handgun. The doc's table 1.
    Pistol,
    /// A submachine gun. Table 2.
    Smg,
    /// An assault rifle. Table 3.
    Ar,
    /// A marksman rifle. Table 4.
    Dmr,
    /// A sniper rifle. Table 5.
    Sniper,
    /// A shotgun. Table 6.
    Shotgun,
    /// A rocket launcher. Table 7.
    Launcher,
}

impl WeaponClass {
    /// Every class, in the doc's own table order — so a generator, a test and a
    /// UI enumerate the list rather than restate it.
    pub const ALL: [WeaponClass; 7] = [
        WeaponClass::Pistol,
        WeaponClass::Smg,
        WeaponClass::Ar,
        WeaponClass::Dmr,
        WeaponClass::Sniper,
        WeaponClass::Shotgun,
        WeaponClass::Launcher,
    ];

    /// The TOML spelling — the string a registry row's `class = "…"` carries.
    pub fn name(self) -> &'static str {
        match self {
            WeaponClass::Pistol => "pistol",
            WeaponClass::Smg => "smg",
            WeaponClass::Ar => "ar",
            WeaponClass::Dmr => "dmr",
            WeaponClass::Sniper => "sniper",
            WeaponClass::Shotgun => "shotgun",
            WeaponClass::Launcher => "launcher",
        }
    }

    /// Read a class by name; `None` for anything else, so a typo in a registry
    /// row is refused BY NAME rather than silently defaulted (the
    /// [`WEAPON_SUB_TABLES`] rule one level down).
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|c| c.name() == s.trim().to_ascii_lowercase())
    }

    /// **Its index**, `0..7` — the offset a clip GUID and a salt are built from.
    pub fn index(self) -> u8 {
        Self::ALL
            .iter()
            .position(|c| *c == self)
            .expect("ALL contains every variant") as u8
    }

    /// **The class a weapon that names none is**, from the one number that
    /// happens to separate all seven: how far the muzzle is along the barrel.
    ///
    /// This is the SECOND answer, in `inf_physics::d3::gameplay::muzzle_of`'s
    /// own shape — a door with two answers, the second for content authored
    /// before the field existed. The registry's own per-class table gives the
    /// seven values (`weapons.toml`, the PER-CLASS RULES block): 0.12 / 0.25 /
    /// 0.45 / 0.50 / 0.55 / 0.65 / 0.70, which are distinct, so the bands below
    /// are the midpoints between neighbours and every one of the eighty-five
    /// rows lands on the class it names. `weapons_registry_rows_name_the_class_
    /// their_barrel_implies` is the arm that says so, and it is not circular:
    /// the class name and the muzzle offset are two independent authorings of
    /// the same fact.
    ///
    /// A weapon authored outside the registry — a gate fixture, a mod's row —
    /// gets the nearest band. That is a guess and it is stated as one; what it
    /// buys is that no weapon is ever silent for want of a field.
    pub fn from_muzzle_forward_m(m: f64) -> Self {
        if m <= 0.185 {
            WeaponClass::Pistol
        } else if m <= 0.35 {
            WeaponClass::Smg
        } else if m <= 0.475 {
            WeaponClass::Ar
        } else if m <= 0.525 {
            WeaponClass::Shotgun
        } else if m <= 0.60 {
            WeaponClass::Dmr
        } else if m <= 0.675 {
            WeaponClass::Sniper
        } else {
            WeaponClass::Launcher
        }
    }
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

    // ── the hybrid (wave WPN2a) ─────────────────────────────────────────────
    //
    // Eleven fields, and every one of them is free: this type carries no
    // `Serialize`, rides no wire and is built from TOML by `from_toml_table`, so
    // the whole of wave WPN2a costs **zero schema** — scene v27 and
    // `ScenePayload` 13 are untouched. Every default below is chosen so a
    // `WeaponDef` that names none of them behaves exactly as it did at island
    // wave I6, which is what keeps the committed catalogues byte-identical.
    /// **How far the instant ray reaches before a round takes over**, metres —
    /// the hybrid's switch, and it applies to [`ShotKind::Projectile`] only.
    ///
    /// The research doc's own thresholds, by class: pistols ~30 m ("pure
    /// hitscan" is a pistol's whole envelope), SMGs 15 m, assault rifles 25 m,
    /// marksman rifles 15 m, snipers ≤ 10 m, shotguns 10 m, launchers **0** (a
    /// rocket is a body from the muzzle). Inside it, a shot is resolved by the
    /// one cast `resolve_shot` has always made; beyond it,
    /// `inf_physics::d3::gameplay::step_rounds` flies a
    /// [`crate::ballistics::Round`].
    ///
    /// A [`ShotKind::Hitscan`] ignores this completely, which is why every
    /// committed row keeps its behaviour: all three of them are hitscans.
    pub hitscan_threshold_m: f64,
    /// **The whole of the quadratic drag, as one coefficient**, units **1/m**.
    ///
    /// The doc writes drag as `−½ρv²C_dA·v̂` and then collapses it in its own
    /// Rust to one `drag_coefficient` multiplying `speed_sq`; so does this.
    /// `drag_k · v²` must come out in m/s², and `(1/m)·(m/s)² = m/s²`, so the
    /// unit is one over metres. The doc's rifle figure is 0.0003–0.0004 /m.
    /// Four separate fields would be four numbers an author cannot check
    /// against anything and one they can.
    pub drag_k: f64,
    /// **How much of [`crate::ballistics::PROJECTILE_GRAVITY_MPS2`] a round
    /// feels**, dimensionless. `1.0` is 9.81 m/s²; `0.0` is a flat trajectory.
    pub gravity_scale: f64,
    /// **What a hit on the head is worth**, as a multiplier on whatever the
    /// damage curve gave. `1.0` — no head bonus at all — is the default, which
    /// is the I6 behaviour.
    pub headshot_mult: f64,
    /// **The distance inside which a shot does full damage**, metres. See
    /// [`crate::ballistics::damage_curve_j`].
    pub effective_range_m: f64,
    /// **The distance at which a shot is down to
    /// [`min_damage_frac`](Self::min_damage_frac)**, metres.
    ///
    /// `0.0` — the default — means **no curve at all**: `max_range_m >
    /// effective_range_m` is the test, and a weapon that fails it does the same
    /// damage at every distance, which is what shipped before this wave.
    pub max_range_m: f64,
    /// **What is left of the damage at [`max_range_m`](Self::max_range_m)**, as
    /// a fraction of [`damage_j`](Self::damage_j). The doc's M4A1 row is
    /// 18 of 30, i.e. 0.6.
    pub min_damage_frac: f64,
    /// **The doc's 1–10 recoil stat.** Nothing in this wave reads it: it is
    /// what `RecoilProfile::from_recoil_stat` takes in **wave WPN2b**, and it
    /// is carried in the registry now so the 85 rows are authored once.
    pub recoil_intensity: f64,
    /// **How long aiming down the sights takes**, milliseconds. **Wave WPN2b**
    /// drives the camera's `[aiming]` blend with it; nothing reads it here.
    pub ads_time_ms: f64,
    /// **How fast a character moves while this is equipped**, as a multiplier on
    /// its own walk/run/sprint speeds. `1.0` is unarmed.
    ///
    /// **This one has a consumer in this wave.**
    /// [`crate::movement::settings_for`] multiplies the resolved target speed by
    /// it, through [`equipped_move_speed_scale`], so a sniper really is slower
    /// than an empty pair of hands and `wpn2a_gate` measures the difference on
    /// the island's own hero.
    pub move_speed_mult: f64,
    /// **Metres past which this weapon's report is silent** — the per-weapon
    /// [`REPORT_MAX_M`].
    ///
    /// The doc's `[audio]` sub-table asks for per-weapon audio, and this is the
    /// one field of it that does not contradict a standing ruling: P22 §5
    /// refuses an impact-*clip* slot on a weapon ("a shot into concrete and a
    /// shot into glass should differ by what was hit rather than by what
    /// fired"), and `report_source`'s doc extends that to the report's clip. A
    /// report's *range* is not a clip — it is, in [`REPORT_MAX_M`]'s own words,
    /// "the one emitter whose range is the gameplay", and a .50 BMG and a
    /// suppressed .380 empty different numbers of streets. It travels on
    /// `WeaponHit` for `WeaponHit::loud`'s reason: what made the noise is a
    /// property of the shot, not of whatever is in the hand when it lands.
    pub report_max_m: f64,

    // ── the sound and the brass (wave WPN2c) ────────────────────────
    //
    // Four more free fields, on the eleven above's argument verbatim: this type
    // carries no `Serialize`, so wave WPN2c also costs **zero schema** — scene
    // v27 and `ScenePayload` 13 are untouched.
    /// **What kind of gun this is** (wave WPN2c), or `None` for a definition
    /// that does not say.
    ///
    /// Read through [`audio_class`](Self::audio_class), never directly, because
    /// the answer for `None` is a rule and not a default — see
    /// [`WeaponClass::from_muzzle_forward_m`]. Every one of the eighty-five
    /// registry rows names it; the fists do not, and a fist is a
    /// [`ShotKind::Melee`] that never reports at all.
    ///
    /// It comes across `from_toml_table` as the string `class = "ar"`, on
    /// `kind`'s own terms: it is the second string key in the reader and the
    /// numeric [`set`](Self::set) door does not carry it, because a class is a
    /// name and encoding it as an integer would put a magic number in
    /// eighty-five rows.
    pub class: Option<WeaponClass>,
    /// **How far right of the aim line the ejection port sits**, metres.
    ///
    /// The doc §3's *"ejection port socket"*, expressed as the one number a
    /// side-ejecting weapon needs — exactly what
    /// [`muzzle_forward_m`](Self::muzzle_forward_m) is for the muzzle, and it
    /// sits beside it for that reason. The brief asked for this on
    /// [`crate::item::ItemDef`]; it is here instead, one field in, because the
    /// `ItemDef` is the wrapper and the *weapon* is what has a barrel: putting
    /// it on the wrapper would have needed a second TOML reader and a second
    /// clamp for a number this one already takes by name.
    pub eject_offset_m: f64,
    /// **Which way the brass leaves**, degrees right of the aim line
    /// (`90` is straight out to the right; `0` would be down range).
    ///
    /// The doc's *"outward linear impulse"* as a bearing rather than a vector,
    /// so a registry row is three numbers rather than six and an author can
    /// read it. The rise is not a field: see [`crate::casing::EJECT_RISE_DEG`],
    /// which is one constant for every weapon because no gun in the doc's seven
    /// tables ejects anywhere but up and out.
    pub eject_dir_deg: f64,
    /// **How fast the brass leaves**, m/s. The doc's impulse, as a speed.
    pub eject_speed_mps: f64,
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
            // WPN2a. Every one of these is chosen so a definition that names
            // none of them is the I6 weapon it was: the threshold is unread by
            // a hitscan, the curve is FLAT because `max_range_m` is not greater
            // than `effective_range_m`, the head bonus is 1.0, the move scale is
            // 1.0 and the report reaches exactly `REPORT_MAX_M`.
            hitscan_threshold_m: 25.0,
            drag_k: 0.0003,
            gravity_scale: 1.0,
            headshot_mult: 1.0,
            effective_range_m: 0.0,
            max_range_m: 0.0,
            min_damage_frac: 1.0,
            recoil_intensity: 3.0,
            ads_time_ms: 200.0,
            move_speed_mult: 1.0,
            report_max_m: REPORT_MAX_M,
            // WPN2c. `None` is the honest default: a definition that does not
            // name a class gets the band rule rather than somebody's favourite
            // gun. The three ejection numbers are a rifle's, which is what
            // every other default on this struct is.
            class: None,
            eject_offset_m: 0.06,
            eject_dir_deg: 80.0,
            eject_speed_mps: 2.4,
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
/// used to add *"and is also how long the body carries the swing's own pose"*,
/// which the wave's own carried list already contradicted). A fist is not
/// *equipped* — it is not in the catalogue at all — so a punch installs no
/// [`crate::feel::WeaponFeel`], the hold point stays where the animation put it
/// and a punch moves no bone. `weapon_hands_gate` asserts exactly that, so the
/// day it changes the arm fails.
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
        // WPN2a: a fist is a `Melee`, so it never mints a round, never asks the
        // curve for anything but its base and never lets go of a report. Every
        // one of these is the default and is spelled out rather than
        // `..Default::default()`-ed, because that spread would silently give a
        // punch whatever a later wave changes a rifle's default to.
        hitscan_threshold_m: 0.0,
        drag_k: 0.0,
        gravity_scale: 1.0,
        headshot_mult: 1.0,
        effective_range_m: 0.0,
        max_range_m: 0.0,
        min_damage_frac: 1.0,
        recoil_intensity: 0.0,
        ads_time_ms: 0.0,
        move_speed_mult: 1.0,
        report_max_m: REPORT_MAX_M,
        // WPN2c: a fist is silent (`WeaponHit::loud` is false for a swing), so
        // it never reaches a report layer and never names a class; and it
        // ejects nothing, which is what a zero speed means here.
        class: None,
        eject_offset_m: 0.0,
        eject_dir_deg: 0.0,
        eject_speed_mps: 0.0,
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
            // WPN2a. Clamped rather than refused, exactly as the ten above are.
            "hitscan_threshold_m" => self.hitscan_threshold_m = value.clamp(0.0, MAX_RANGE_M),
            "drag_k" => self.drag_k = value.clamp(0.0, MAX_DRAG_K),
            "gravity_scale" => self.gravity_scale = value.clamp(0.0, 100.0),
            "headshot_mult" => self.headshot_mult = value.clamp(1.0, MAX_HEADSHOT_MULT),
            "effective_range_m" => self.effective_range_m = value.clamp(0.0, MAX_RANGE_M),
            "max_range_m" => self.max_range_m = value.clamp(0.0, MAX_RANGE_M),
            "min_damage_frac" => self.min_damage_frac = value.clamp(0.0, 1.0),
            "recoil_intensity" => self.recoil_intensity = value.clamp(0.0, 10.0),
            "ads_time_ms" => self.ads_time_ms = value.clamp(0.0, 5000.0),
            "move_speed_mult" => self.move_speed_mult = value.clamp(MIN_MOVE_SPEED_MULT, 2.0),
            "report_max_m" => self.report_max_m = value.clamp(1.0, MAX_RANGE_M),
            // WPN2c, clamped exactly as everything above it is. The CLASS is
            // not here: it is a name, and it comes across the reader's string
            // branch beside `kind`.
            "eject_offset_m" => self.eject_offset_m = value.clamp(0.0, 1.0),
            "eject_dir_deg" => self.eject_dir_deg = value.clamp(-180.0, 180.0),
            "eject_speed_mps" => self.eject_speed_mps = value.clamp(0.0, 20.0),
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
            "ads_time_ms",
            "automatic",
            "damage_j",
            "drag_k",
            "effective_range_m",
            "eject_dir_deg",
            "eject_offset_m",
            "eject_speed_mps",
            "gravity_scale",
            "headshot_mult",
            "hitscan_threshold_m",
            "magazine",
            "max_range_m",
            "melee",
            "melee_arc_deg",
            "min_damage_frac",
            "move_speed_mult",
            "muzzle_forward_m",
            "muzzle_speed_mps",
            "projectile",
            "range_m",
            "recoil_intensity",
            "reload_s",
            "report_max_m",
            "reserve",
            "rounds_per_minute",
            "spread_deg",
        ]
    }

    /// **What one shot is worth at a distance**, joules — the one door both
    /// halves of the hybrid spend through (wave WPN2a).
    ///
    /// The instant ray asks it with the cast's own `toi`; a round asks it with
    /// [`crate::ballistics::Round::travelled_m`]. There is exactly one damage
    /// computation in this engine and this is it — `apply_hit` then spends the
    /// answer through [`damage_entity`], which is still the one door joules
    /// leave by.
    ///
    /// # The joule scale, stated
    ///
    /// The research doc's tables are in **HP against a 100 HP body**; this
    /// engine's unit is joules against [`DEFAULT_VITALITY_J`] = 2 000 J. So
    /// **1 HP = 20 J**, and the registry's `damage_j` is the doc's torso figure
    /// times twenty — the Glock 17's 30 HP is 600 J, which is the number the
    /// pistol row in `GAMEPLAY_ITEMS_TOML` has carried since island wave I6
    /// without anybody choosing the scale on purpose. See
    /// [`JOULES_PER_HIT_POINT`].
    pub fn damage_at(&self, distance_m: f64, headshot: bool) -> f64 {
        let base = crate::ballistics::damage_curve_j(
            self.damage_j,
            self.min_damage_frac,
            self.effective_range_m,
            self.max_range_m,
            distance_m,
        );
        if headshot {
            base * self.headshot_mult.max(1.0)
        } else {
            base
        }
    }

    /// **How far the instant ray of this weapon reaches**, metres — the hybrid's
    /// switch, resolved against the weapon's own range.
    ///
    /// A [`ShotKind::Hitscan`] answers its whole [`reach_m`](Self::reach_m),
    /// which is what makes every level committed before wave WPN2a
    /// byte-identical. A [`ShotKind::Projectile`] answers the smaller of the
    /// threshold and the range: a round is only worth minting where there is
    /// range left for it to fly through.
    pub fn hitscan_reach_m(&self) -> f64 {
        let reach = self.reach_m();
        if self.kind == ShotKind::Projectile {
            reach.min(self.hitscan_threshold_m.max(0.0))
        } else {
            reach
        }
    }

    /// **Whether a shot that missed inside
    /// [`hitscan_reach_m`](Self::hitscan_reach_m) should mint a round.**
    ///
    /// Only a projectile, and only when the threshold left something to fly
    /// through — a pistol whose 30 m threshold covers its own range never mints
    /// one, which is the doc's "pure hitscan" for that class expressed as a
    /// consequence rather than as a second kind.
    pub fn spawns_a_round(&self) -> bool {
        self.kind == ShotKind::Projectile && self.reach_m() > self.hitscan_reach_m()
    }

    /// **What kind of gun this is, always** (wave WPN2c) — the one door a
    /// report's clips are chosen through.
    ///
    /// Two answers, in `inf_physics::d3::gameplay::muzzle_of`'s own shape: the
    /// class the definition NAMES, or, for one that names none, the band its
    /// barrel implies ([`WeaponClass::from_muzzle_forward_m`]). Never read
    /// [`class`](Self::class) directly — a call site that did would have to
    /// decide what `None` sounds like, and then there would be two rules.
    pub fn audio_class(&self) -> WeaponClass {
        self.class
            .unwrap_or_else(|| WeaponClass::from_muzzle_forward_m(self.muzzle_forward_m))
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
            // **The doc's sub-tables** (wave WPN2a). The research doc's schema
            // groups a weapon's numbers under `[ballistics]`, `[damage_curve]`,
            // `[recoil]` and `[audio]`; an 85-row registry authored as one flat
            // table is a wall of keys nobody can read. So a sub-table is read
            // through **the same by-name door** its parent is — one `set` per
            // key, one place where a range is clamped — and the grouping is
            // presentation. A sub-table this reader does not know is refused
            // BY NAME rather than skipped, which is the whole point: a
            // `[ballisitcs]` typo that silently dropped a muzzle velocity would
            // fire the default at a designer who had authored a number.
            if let toml::Value::Table(sub) = v {
                if !WEAPON_SUB_TABLES.contains(&k.as_str()) {
                    return Err(format!(
                        "unknown weapon sub-table [{k}] (known: {})",
                        WEAPON_SUB_TABLES.join(", ")
                    ));
                }
                for (sk, sv) in sub {
                    let n = match sv {
                        toml::Value::Float(f) => *f,
                        toml::Value::Integer(i) => *i as f64,
                        toml::Value::Boolean(b) => f64::from(u8::from(*b)),
                        _ => return Err(format!("weapon key {k}.{sk} is not a number")),
                    };
                    if !def.set(sk, n) {
                        return Err(format!("unknown weapon key {k}.{sk}"));
                    }
                }
                continue;
            }
            let n = match v {
                toml::Value::Float(f) => *f,
                toml::Value::Integer(i) => *i as f64,
                toml::Value::Boolean(b) => f64::from(u8::from(*b)),
                toml::Value::String(s) => {
                    // **The second string key** (wave WPN2c): `class = "ar"`.
                    // Refused BY NAME on a typo, which is the rule the sub-table
                    // reader above already follows and for the same reason: a
                    // `class = "assault_rifle"` that was silently ignored would
                    // give an assault rifle a pistol's gunshot and say nothing.
                    if k == "class" {
                        def.class = Some(WeaponClass::from_name(s).ok_or_else(|| {
                            format!(
                                "unknown weapon class {s} (known: {})",
                                WeaponClass::ALL
                                    .iter()
                                    .map(|c| c.name())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        })?);
                        continue;
                    }
                    // The first string key: `kind = "projectile" | "hitscan"`.
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

/// **The sub-tables a `[<item>.weapon]` table may carry** (wave WPN2a) — the
/// research doc's own four, enumerated beside the reader that takes them so an
/// author reads the list rather than guessing it (the P29.6 A14 rule, one level
/// down).
///
/// Every key inside every one of them goes through [`WeaponDef::set`]; the
/// grouping carries no meaning to the engine and exists so eighty-five rows are
/// legible. `[audio]` is on the list and carries exactly one field today
/// ([`WeaponDef::report_max_m`]) — the four-layer stack is **wave WPN2c's**, and
/// a per-weapon report *clip* is refused by P22 §5's own reasoning; see
/// `report_source`.
pub const WEAPON_SUB_TABLES: [&str; 5] = ["ballistics", "damage_curve", "recoil", "audio", "eject"];

/// **The eighty-five-row weapon registry** (wave WPN2a) — the research doc's
/// tables, as the TOML the `item.define` node takes.
///
/// # Why an `include_str!` and not an asset kind
///
/// [`crate::item::ItemDefs`]' own module header priced this: an `items.toml`
/// beside a level reaches **one of three boot paths**, so a catalogue there is
/// present in a dev run and absent in a build. A runtime-*editable* registry is
/// a new asset kind with an entity `Uuid` field — a scene bump — and that window
/// belongs to VEH3a, not here. `GAMEPLAY_ITEMS_TOML` is the precedent and this
/// is the same route one order of magnitude larger: the bytes ride the
/// Blueprint's own `item.define` call, so Simulate, PIE and a cooked pack all
/// see exactly the same catalogue.
///
/// It lives in **Ring 0** rather than beside the editor's sample generators
/// because the gates, the island generator and the fixture all want the same
/// eighty-five rows and a second copy of them is a second set of numbers.
/// [`MAX_ITEM_DEFS`](crate::item::MAX_ITEM_DEFS) is 4 096, so the cap does not
/// bind.
///
/// # Real names
///
/// The rows carry the manufacturers' names because the doc does. **A firearm's
/// name used to identify that firearm in a work of fiction is nominative use,
/// not trademark use**: it does not indicate the source of this software, no
/// endorsement is claimed or implied, and courts have consistently treated the
/// depiction of real objects in expressive works this way. The decision is
/// recorded here rather than assumed, and swapping in lore names is a search and
/// replace over one file if a publisher ever wants one.
pub const WEAPON_REGISTRY_TOML: &str = include_str!("weapons.toml");

/// **The equipped weapon's definition**, if the character has one equipped and
/// the catalogue knows it — `(item id, definition)`.
///
/// The **one** lookup: `inf_physics::d3::gameplay` asks it to decide what a
/// trigger does and [`equipped_move_speed_scale`] asks it to decide how fast the
/// carrier walks, and two spellings of "what is in this character's hand" is
/// exactly the defect this repository has paid for at five seams.
pub fn equipped_def(world: &EcsWorld, guid: Uuid) -> Option<(String, WeaponDef)> {
    let entity = world.entity_of(guid)?;
    let inv = world.world().get::<crate::item::Inventory>(entity)?;
    let id = inv.equipped_id()?.to_string();
    let def = *crate::item::item_defs(world)?.get(&id)?.weapon.as_ref()?;
    Some((id, def))
}

/// **How fast this character moves for what it is carrying** — the doc's
/// "Move Speed: relative movement multiplier (1.0 = Base/Unarmed speed)".
///
/// `1.0` for an unarmed character, for one carrying something the catalogue does
/// not know, and for every weapon authored before wave WPN2a — which is what
/// keeps every committed movement trace in the tree byte-identical.
///
/// Read by [`crate::movement::settings_for`], which is the one place a target
/// speed is resolved, so a sprint, a walk, a crouch and a swim all scale by the
/// same number and none of them needed a branch.
pub fn equipped_move_speed_scale(world: &EcsWorld, guid: Uuid) -> f64 {
    equipped_def(world, guid)
        .map(|(_, d)| d.move_speed_mult.clamp(MIN_MOVE_SPEED_MULT, 2.0))
        .unwrap_or(1.0)
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
    /// **How much cone this magazine has bloomed**, degrees (wave WPN2b).
    ///
    /// The doc's *"subtle bullet vector drift"*: every round adds
    /// [`crate::feel::bloom_per_shot_deg`] and every step takes
    /// [`crate::feel::bloom_decay_dps`] back off, so the thirtieth round of a
    /// magazine really does land somewhere the first did not.
    ///
    /// It lives HERE and not on `crate::feel::WeaponFeel` because the bloom
    /// belongs to the magazine: putting a weapon away and taking it out again is
    /// what a player does to reset it in every game that has one, and this
    /// struct is already replaced when the equipped id changes.
    ///
    /// **`0.0` is not folded** by [`weapon_state_bytes`] — see its doc. That is
    /// what keeps every trace of a weapon that is merely being CARRIED
    /// byte-identical to its pre-WPN2b self.
    pub spread_bloom_deg: f64,
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

// ── the four layers (wave WPN2c) ────────────────────────────────────────────

/// **The five clips one weapon class owns** — the research doc §4's stack, as
/// files.
///
/// Four LAYERS play per shot ([`ReportLayerKind`]) and there are five clips,
/// because the third layer is a *room*: a shot indoors and the same shot on the
/// street are the same layer with different bytes, chosen by the enclosure
/// probe (`inf_physics::d3::audio::enclosure_at`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReportClip {
    /// The mechanical snap — bolt, hammer, action. Five milliseconds or less.
    Transient,
    /// The explosive core, 40-80 Hz. The loudest thing this engine makes.
    Body,
    /// A tight room's reflection, 0.3 s.
    IndoorTail,
    /// A street's reflection, 1.5 s.
    OutdoorTail,
    /// The distant N-wave, 2 ms — both the fourth layer and the supersonic
    /// crack a round makes going past an ear.
    Crack,
}

impl ReportClip {
    /// Every clip kind, in the order the GUID table is built in.
    pub const ALL: [ReportClip; 5] = [
        ReportClip::Transient,
        ReportClip::Body,
        ReportClip::IndoorTail,
        ReportClip::OutdoorTail,
        ReportClip::Crack,
    ];

    /// Its index, zero to four — the low nibble of its GUID.
    pub fn index(self) -> u8 {
        Self::ALL
            .iter()
            .position(|c| *c == self)
            .expect("ALL contains every variant") as u8
    }

    /// The file stem a generated clip is committed under, as in
    /// `Report_Ar_Body.inf_audio`.
    pub fn file_stem(self) -> &'static str {
        match self {
            ReportClip::Transient => "Transient",
            ReportClip::Body => "Body",
            ReportClip::IndoorTail => "IndoorTail",
            ReportClip::OutdoorTail => "OutdoorTail",
            ReportClip::Crack => "Crack",
        }
    }
}

/// **The base of the thirty-five generated clip GUIDs** (wave WPN2c).
///
/// `0x5750_4e32` is `"WPN2"`, one after [`WEAPON_REPORT_CLIP`]'s `"WPN1"`, on
/// exactly its argument: an asset an engine constant names by id must have the
/// same id every time or the committed bytes are a different set of files on
/// every build.
pub const REPORT_CLIP_BASE: u128 = 0x5750_4e32_0000_0000;

/// **Which `.inf_audio` a class plays for a layer** — the thirty-five-entry
/// table, as arithmetic rather than as thirty-five constants.
///
/// `base | class << 8 | clip`, so a GUID is readable: `…0201` is class 2, the
/// assault rifle, clip 1, the body.
///
/// # The one fixed point
///
/// Wave WPN1's [`WEAPON_REPORT_CLIP`] **is** the assault rifle's body layer.
/// It is the clip this engine has named since WPN1, the one `.inf_audio` the
/// `samples/phase30-gameplay` fixture commits and the one that fixture's own
/// rifle fires — so it keeps its GUID and the table is built around it rather
/// than orphaning a committed file. Everything else is derived. The special
/// case lives here, in the one function that knows the table, rather than at
/// the call sites that ask it.
pub fn report_clip(class: WeaponClass, clip: ReportClip) -> Uuid {
    if class == WeaponClass::Ar && clip == ReportClip::Body {
        return WEAPON_REPORT_CLIP;
    }
    Uuid::from_u128(REPORT_CLIP_BASE | (u128::from(class.index()) << 8) | u128::from(clip.index()))
}

/// **The metal one-shot a shell casing makes when it lands** (wave WPN2c) —
/// the doc section 3's `casing_drop_metal_01.wav`, pitch-randomised per casing.
///
/// One clip for every calibre: what a brass case sounds like on concrete is a
/// property of the concrete, and the pitch hash is what makes two of them
/// different. It sits outside the class table because it is not a layer of a
/// report.
pub const CASING_CLIP: Uuid = Uuid::from_u128(REPORT_CLIP_BASE | 0x00f0);

/// **Which of the four layers a `Play` is** (wave WPN2c) — the doc section 4's
/// stack, in the order the queue carries them.
///
/// The order is the contract: a queue is ordered, and a gate that compares two
/// command streams cannot tell an ordering it never asserted. It is also the
/// order a person hears them in — the snap before the boom before the room.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReportLayerKind {
    /// Layer 1 — the mechanical punch.
    Transient,
    /// Layer 2 — the explosive body.
    Body,
    /// Layer 3 — the environment tail, indoor or outdoor.
    Tail,
    /// Layer 4 — the distant crack, silent until the listener is far enough
    /// away for it to be the thing they hear.
    Distant,
}

impl ReportLayerKind {
    /// The four, in queue order.
    pub const ALL: [ReportLayerKind; 4] = [
        ReportLayerKind::Transient,
        ReportLayerKind::Body,
        ReportLayerKind::Tail,
        ReportLayerKind::Distant,
    ];

    /// Its index, zero to three.
    pub fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|k| *k == self)
            .expect("ALL contains every variant")
    }
}

/// **The four salts a shooter's source keys are spread over** (wave WPN2c).
///
/// # What this closes
///
/// Since wave WPN1 a report has been keyed on `guid_source_key(shooter)`, and
/// [`report_source`]'s own doc has carried the defect that follows: that is the
/// **shooter's own emitter namespace**, so a character carrying an autoplay
/// `AudioSource` would have its voice replaced by its own gunshot — and the
/// autoplay walk starts a source once, so it would never come back. Nothing in
/// the committed tree does it, which is why it was carried rather than fixed.
///
/// Four layers need four keys anyway — one voice per key, replace on re-`Play`,
/// so four layers that shared a key would be one layer — and the cheapest four
/// keys that do not collide with the shooter are four salts. So the carried
/// item is closed by the change that needed it closed.
///
/// The values are not arbitrary: each is the same `"WPN2"` stamp the clip GUIDs
/// carry, so a key seen in a log says where it came from; and no entity guid's
/// low 64 bits can collide with one more often than with any other 64-bit
/// value, which is the bound `guid_source_key` itself has always had.
pub const LAYER_SALTS: [u64; 4] = [
    0x5750_4e32_0000_0001,
    0x5750_4e32_0000_0002,
    0x5750_4e32_0000_0003,
    0x5750_4e32_0000_0004,
];

/// **The source key one layer of one shooter's report plays on** (wave WPN2c).
///
/// The shooter's key exclusive-or'd with the layer's salt, so the four are
/// distinct from each other and from the shooter's own emitter key — and a
/// barrel is still ONE voice per layer: a second round restarts each of the
/// four rather than stacking, which is what keeps a 600 rpm burst from being
/// forty live voices a second.
pub fn layer_source_key(shooter_key: u64, kind: ReportLayerKind) -> u64 {
    shooter_key ^ LAYER_SALTS[kind.index()]
}

/// **Metres past which the DISTANT layer is the one you hear.**
///
/// A hundred and fifty — the research doc's own `dist / 150.0` in
/// `process_gunshot`. Inside it the distant layer is silent, its volume
/// literally `0.0`, and the command still goes out: the command stream is the
/// contract, and a layer that vanished from the queue at 149 m and appeared at
/// 151 m would make the count a function of where the player is standing.
pub const DISTANT_ONSET_M: f64 = 150.0;

/// **Metres at which the DISTANT layer is at full volume.**
///
/// Three hundred — twice the onset, so the ramp is one onset wide and a shot
/// heard across a square swells rather than switching on.
pub const DISTANT_FULL_M: f64 = 300.0;

/// **The cutoff air puts on a gunshot heard from far away**, hertz.
///
/// Seven hundred. This is the wave's own consumer for the low-pass that has
/// been modelled-not-applied since P12: a rifle at four hundred metres is a
/// *thump*, and the reason is that a few hundred metres of air is a low-pass.
/// It rides the emitter's own `lowpass_hz` rather than a bus effect, because it
/// is a property of one voice and not of the mix.
pub const DISTANT_LOWPASS_HZ: f64 = 700.0;

/// **The cutoff a room puts on its own tail**, hertz.
///
/// Three thousand five hundred — a concrete room's reflection has lost its top
/// end by the time it comes back, which is most of the difference between
/// "inside" and "outside" once the two decay lengths already differ.
pub const INDOOR_TAIL_LOWPASS_HZ: f64 = 3500.0;

/// The transient's base volume — just under the body's, because a bolt is not
/// the bang.
pub const TRANSIENT_VOLUME: f64 = 0.85;

/// **How much of a weapon's report range the TRANSIENT carries**, as a
/// fraction.
///
/// Twelve per cent. A mechanical snap is a near-field sound: you hear the
/// action of a rifle fired beside you and you do not hear it three streets
/// away, where the same shot is still perfectly audible as a bang. So the four
/// layers do not share a reach, and this is the number that makes a gunshot
/// change SHAPE with distance rather than merely get quieter.
pub const TRANSIENT_REACH_FRACTION: f64 = 0.12;

/// The indoor tail's base volume — the doc section 4's own `0.8`.
pub const INDOOR_TAIL_VOLUME: f64 = 0.8;

/// **How much of a weapon's report range the INDOOR tail carries.**
///
/// A quarter. A room's reflection belongs to the room, and it does not leave
/// through the walls at the range the shot itself does.
pub const INDOOR_TAIL_REACH_FRACTION: f64 = 0.25;

/// The outdoor tail's base volume — the doc section 4's own `0.7`.
pub const OUTDOOR_TAIL_VOLUME: f64 = 0.7;

/// **How far the body layer's pitch may wander**, as a fraction.
///
/// Two per cent — the doc's `rand_pitch(0.98, 1.02)`, drawn from
/// [`shot_uniforms`] rather than from an RNG because a fixed step holds no
/// random state (the P14 law). The n-th round of a magazine has the same pitch
/// in a replay, in a PIE preview and in a shipped build.
pub const BODY_PITCH_JITTER: f64 = 0.02;

/// **How loud the DISTANT layer is** for a listener this far from the muzzle —
/// zero inside [`DISTANT_ONSET_M`], full at [`DISTANT_FULL_M`], linear between.
pub fn distant_gain(listener_m: f64) -> f64 {
    if !listener_m.is_finite() {
        return 0.0;
    }
    let t = (listener_m - DISTANT_ONSET_M) / (DISTANT_FULL_M - DISTANT_ONSET_M);
    t.clamp(0.0, 1.0)
}

/// **One layer of a shot**, ready to be queued.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportLayer {
    /// Which of the four this is — the salt its source key is taken from.
    pub kind: ReportLayerKind,
    /// The emitter description, on [`report_source`]'s own terms.
    pub source: AudioSource,
    /// **The cutoff this layer is born with**, hertz, or `None`.
    ///
    /// It is HERE and not on [`crate::components::AudioSource`], and the reason
    /// is a house law rather than a preference: `AudioSource` is a scene
    /// component and the `.inf_lvl` payload is **bincode**, which is
    /// positional, so growing it is a wire-format change and a schema bump.
    /// This wave moves no schema. The cutoff rides the LAYER, which is a Ring-0
    /// description nothing serializes, and reaches the device through
    /// `inf_audio::PlayCommand::lowpass_hz`.
    pub lowpass_hz: Option<f64>,
}

/// **THE FOUR LAYERS OF ONE GUNSHOT** (wave WPN2c) — the research doc section
/// 4's stack, as the four `Play`s both hosts queue inside the `weapon_report`
/// MIRROR fence.
///
/// One place, for [`report_source`]'s reason exactly and one more: four layers
/// written out twice in two host-side loops is four clips, four volumes, four
/// reaches and four keys that have to be compared character for character to
/// stay in step, and the fence is what proves they are.
///
/// # What each argument decides
///
/// * `class` — which five clips ([`report_clip`]). It is the weapon's own
///   [`WeaponDef::audio_class`], carried on the shot rather than looked up
///   afterwards, for `WeaponHit::loud`'s reason: by the time this runs the
///   shooter may have scrolled.
/// * `indoors` — whether layer 3 is the room's tail or the street's. The
///   enclosure probe's verdict, taken at the muzzle on the step the trigger
///   went down (`inf_physics::d3::audio::enclosure_at`).
/// * `report_max_m` — wave WPN2a's per-weapon reach. The layers take
///   *fractions* of it rather than sharing it, which is what makes a gunshot
///   change shape with distance instead of only getting quieter.
/// * `listener_m` — how far the listener is from the muzzle, so layer 4 knows
///   whether it is the thing being heard. Computed sim-side, so both hosts get
///   the same number from the same world rather than each asking its own engine.
/// * `shot_index` — the round's own counter, for the body's pitch jitter.
///
/// # Every layer is a command, always
///
/// Four `Play`s per loud shot, whatever the numbers say. A layer whose volume
/// is zero is still queued, because the command stream is the contract and a
/// count that changed with where the player stood would make every gate that
/// reads it a claim about a camera.
pub fn report_layers(
    class: WeaponClass,
    indoors: bool,
    report_max_m: f64,
    listener_m: f64,
    shot_index: u64,
) -> [ReportLayer; 4] {
    let max = report_max_m.clamp(1.0, MAX_RANGE_M);
    // **Every layer is a [`report_source`]** with three fields moved. That is
    // deliberate rather than tidy: the bus, the spatialisation, the near field,
    // the distance model, the rolloff and the two flags are what "a gunshot's
    // emitter" MEANS, they are argued for one at a time in that function's own
    // doc, and four literals here would be four places for them to drift.
    let base = |clip: Uuid, volume: f64, max_distance: f64| {
        let mut s = report_source();
        s.clip = Some(clip);
        s.volume = volume;
        s.max_distance = max_distance;
        s
    };
    // The body's pitch, from the counter hash rather than from an RNG.
    let (u, _) = shot_uniforms(LAYER_SALTS[1], shot_index);
    let pitch = 1.0 + (u - 0.5) * 2.0 * BODY_PITCH_JITTER;

    let transient = base(
        report_clip(class, ReportClip::Transient),
        TRANSIENT_VOLUME,
        (max * TRANSIENT_REACH_FRACTION).max(REPORT_MIN_M + 1.0),
    );
    let mut body = base(report_clip(class, ReportClip::Body), REPORT_VOLUME, max);
    body.pitch = pitch;
    let (tail, tail_cutoff) = if indoors {
        (
            base(
                report_clip(class, ReportClip::IndoorTail),
                INDOOR_TAIL_VOLUME,
                (max * INDOOR_TAIL_REACH_FRACTION).max(REPORT_MIN_M + 1.0),
            ),
            Some(INDOOR_TAIL_LOWPASS_HZ),
        )
    } else {
        (
            base(
                report_clip(class, ReportClip::OutdoorTail),
                OUTDOOR_TAIL_VOLUME,
                max,
            ),
            None,
        )
    };
    let mut distant = base(
        report_clip(class, ReportClip::Crack),
        REPORT_VOLUME * distant_gain(listener_m),
        max,
    );
    // The distant layer is at full volume until the onset and then falls off,
    // which is the opposite way round from the other three: what it models is
    // the sound that is LEFT at a distance, so its near field is the distance
    // at which it starts to exist at all.
    distant.min_distance = DISTANT_ONSET_M;
    [
        ReportLayer {
            kind: ReportLayerKind::Transient,
            source: transient,
            lowpass_hz: None,
        },
        ReportLayer {
            kind: ReportLayerKind::Body,
            source: body,
            lowpass_hz: None,
        },
        ReportLayer {
            kind: ReportLayerKind::Tail,
            source: tail,
            lowpass_hz: tail_cutoff,
        },
        ReportLayer {
            kind: ReportLayerKind::Distant,
            source: distant,
            lowpass_hz: Some(DISTANT_LOWPASS_HZ),
        },
    ]
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
            spread_bloom_deg: 0.0,
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
///
/// **Public since wave WPN2b**, because the recoil's own noise is drawn here
/// too — the doc's `rand_val()` with no RNG behind it. It is the ONE
/// counter-hash door in this engine's gunplay and a second one would be a second
/// thing to keep replay-safe; `feel::RecoilProfile::peaks` salts the seed so a
/// shot that scattered left does not also kick left for ever.
pub fn shot_uniforms(seed: u64, shot: u64) -> (f64, f64) {
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
    shot_direction_with(def, yaw_deg, pitch_deg, shot, def.spread_deg)
}

/// **Where a shot goes, through a cone this caller resolved** (wave WPN2b).
///
/// [`shot_direction`] with the weapon's own `spread_deg` replaced by
/// `cone_deg`, which is what the fire path computes each shot from the bloom on
/// [`WeaponState::spread_bloom_deg`] and the shooter's stance, speed and aim
/// (`crate::feel::resolved_cone_deg`). One function, one scatter, one seed: a
/// second copy of the disc sampling would be a bullet that went somewhere the
/// pattern arm never looked.
///
/// It is also the door **wave WPN2d's shotgun** takes — N pellets through one
/// `cone_deg` at N consecutive shot indices — which is why the parameter is a
/// cone and not a multiplier.
pub fn shot_direction_with(
    def: &WeaponDef,
    yaw_deg: f64,
    pitch_deg: f64,
    shot: u64,
    cone_deg: f64,
) -> DVec3 {
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
    let half = if cone_deg.is_finite() {
        (cone_deg * 0.5).clamp(0.0, MAX_SPREAD_DEG)
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
///
/// # The bloom is folded only when it is NOT zero (wave WPN2b)
///
/// This buffer is **hashed, never decoded** (`RuntimeSim::state_bytes`' own
/// words), so a row may be two lengths and no reader has to tell them apart.
/// That buys the thing a fixed-width column could not: a weapon that is being
/// carried, or that fired long enough ago for the bloom to have decayed to
/// zero, folds exactly the bytes it folded before this wave — so every
/// committed trace with a resting weapon in it stays byte-identical, and only a
/// trace where somebody has ACTUALLY JUST FIRED moves.
///
/// `crate::ballistics::round_state_bytes`' empty-when-nothing-is-flying rule,
/// one level down: the same argument, applied per row instead of per section.
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
        if s.spread_bloom_deg != 0.0 {
            out.extend_from_slice(&s.spread_bloom_deg.to_bits().to_le_bytes());
        }
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

    /// **THE THIRTY-FIVE CLIPS** (wave WPN2c) — one GUID per class per layer,
    /// all distinct, and wave WPN1's own clip is still one of them.
    #[test]
    fn every_class_names_five_distinct_clips_and_wpn1s_gunshot_is_one_of_them() {
        let mut seen = std::collections::BTreeSet::new();
        for c in WeaponClass::ALL {
            for k in ReportClip::ALL {
                assert!(
                    seen.insert(report_clip(c, k)),
                    "{} {} collides with another clip",
                    c.name(),
                    k.file_stem()
                );
            }
        }
        assert_eq!(seen.len(), 35);
        // The one fixed point: WPN1's committed gunshot IS the rifle's body.
        assert_eq!(
            report_clip(WeaponClass::Ar, ReportClip::Body),
            WEAPON_REPORT_CLIP
        );
        // …and the casing's own clip is outside the table.
        assert!(!seen.contains(&CASING_CLIP));
    }

    /// **THE FOUR SALTED KEYS** (wave WPN2c) — the close of wave WPN1's carried
    /// emitter-namespace collision.
    ///
    /// Four layers on one key would be one layer (one voice per key, replace on
    /// re-`Play`), and the shooter's own key is its emitter namespace.
    #[test]
    fn the_four_layers_key_off_the_shooter_without_colliding_with_it() {
        let shooter = 0x1234_5678_9abc_def0u64;
        let keys: Vec<u64> = ReportLayerKind::ALL
            .into_iter()
            .map(|k| layer_source_key(shooter, k))
            .collect();
        let uniq: std::collections::BTreeSet<u64> = keys.iter().copied().collect();
        assert_eq!(uniq.len(), 4, "two layers share a voice: {keys:?}");
        assert!(
            !uniq.contains(&shooter),
            "a layer took the shooter's own emitter key back"
        );
        // The salt is recoverable, which is what makes a key in a log readable.
        for k in ReportLayerKind::ALL {
            assert_eq!(
                layer_source_key(shooter, k) ^ shooter,
                LAYER_SALTS[k.index()]
            );
        }
        assert_eq!(ReportLayerKind::ALL.len(), LAYER_SALTS.len());
    }

    /// **FOUR LAYERS PER SHOT, IN ORDER** (wave WPN2c) — and the third one is
    /// the room.
    #[test]
    fn a_gunshot_is_four_layers_and_the_third_is_the_room() {
        let out = report_layers(WeaponClass::Ar, false, 320.0, 10.0, 0);
        assert_eq!(
            out.iter().map(|l| l.kind).collect::<Vec<_>>(),
            ReportLayerKind::ALL.to_vec(),
            "the queue order is the contract"
        );
        // Outdoors: the long tail, unfiltered, at the weapon's own reach.
        assert_eq!(
            out[2].source.clip,
            Some(report_clip(WeaponClass::Ar, ReportClip::OutdoorTail))
        );
        assert_eq!(out[2].lowpass_hz, None);
        assert!((out[2].source.max_distance - 320.0).abs() < 1e-9);
        // Indoors: the short tail, filtered, at a quarter of the reach.
        let inside = report_layers(WeaponClass::Ar, true, 320.0, 10.0, 0);
        assert_eq!(
            inside[2].source.clip,
            Some(report_clip(WeaponClass::Ar, ReportClip::IndoorTail))
        );
        assert_eq!(inside[2].lowpass_hz, Some(INDOOR_TAIL_LOWPASS_HZ));
        assert!((inside[2].source.max_distance - 320.0 * 0.25).abs() < 1e-9);
        // The other three layers do not notice the room at all.
        for i in [0usize, 1, 3] {
            assert_eq!(out[i].source, inside[i].source, "layer {i} changed indoors");
        }
        // The transient is a near-field sound and the body is not: the stack
        // changes SHAPE with distance rather than merely getting quieter.
        assert!(out[0].source.max_distance < out[1].source.max_distance / 4.0);
        assert!((out[1].source.max_distance - 320.0).abs() < 1e-9);
        // Every layer is spatial, on `sfx`, and one-shot.
        for l in &out {
            assert!(l.source.spatial && !l.source.looping);
            assert_eq!(l.source.bus, REPORT_BUS);
            assert!(!l.source.occlusion && !l.source.autoplay);
        }
    }

    /// **THE DISTANT LAYER IS SILENT NEXT TO YOU AND LOUD ACROSS A SQUARE** —
    /// and it is a command either way.
    #[test]
    fn the_distant_layer_is_a_function_of_where_the_listener_is() {
        for (m, want) in [
            (0.0, 0.0),
            (10.0, 0.0),
            (DISTANT_ONSET_M, 0.0),
            (225.0, 0.5),
            (DISTANT_FULL_M, 1.0),
            (5000.0, 1.0),
        ] {
            assert!(
                (distant_gain(m) - want).abs() < 1e-12,
                "{m} m gave {}",
                distant_gain(m)
            );
        }
        assert_eq!(distant_gain(f64::NAN), 0.0);
        // A layer whose volume is zero is STILL a command: the count must not
        // be a function of where the player is standing.
        let near = report_layers(WeaponClass::Sniper, false, 600.0, 5.0, 0);
        let far = report_layers(WeaponClass::Sniper, false, 600.0, 600.0, 0);
        assert_eq!(near.len(), far.len());
        assert_eq!(near[3].source.volume, 0.0);
        assert!((far[3].source.volume - REPORT_VOLUME).abs() < 1e-12);
        // …and it is filtered, which is what makes it a thump.
        assert_eq!(far[3].lowpass_hz, Some(DISTANT_LOWPASS_HZ));
        assert!((far[3].source.min_distance - DISTANT_ONSET_M).abs() < 1e-12);
    }

    /// **THE BODY'S PITCH IS THE COUNTER HASH** (wave WPN2c) — the doc's
    /// `rand_pitch(0.98, 1.02)` with no RNG behind it.
    #[test]
    fn the_body_layers_pitch_wanders_by_the_shot_index_and_nothing_else() {
        let mut seen = std::collections::BTreeSet::new();
        for shot in 0..64u64 {
            let l = report_layers(WeaponClass::Smg, false, 240.0, 10.0, shot);
            let p = l[1].source.pitch;
            assert!(
                (p - 1.0).abs() <= BODY_PITCH_JITTER + 1e-12,
                "shot {shot} pitched {p}"
            );
            // The other three layers are unpitched: a bolt and a room do not
            // change note with the round count.
            for i in [0usize, 2, 3] {
                assert!((l[i].source.pitch - 1.0).abs() < 1e-12);
            }
            seen.insert(p.to_bits());
        }
        assert!(
            seen.len() > 50,
            "the pitch barely moves: {} values",
            seen.len()
        );
        // Deterministic: the same round is the same note in a replay.
        assert_eq!(
            report_layers(WeaponClass::Smg, false, 240.0, 10.0, 17)[1]
                .source
                .pitch,
            report_layers(WeaponClass::Smg, false, 240.0, 10.0, 17)[1]
                .source
                .pitch
        );
    }

    /// **THE CLASS DOOR** (wave WPN2c) — a string key beside `kind`, refused by
    /// name on a typo, and a definition that names none answers the band its
    /// barrel implies.
    #[test]
    fn a_weapon_names_its_class_or_takes_the_one_its_barrel_implies() {
        let read = |s: &str| -> Result<WeaponDef, String> {
            let doc: toml::Value = toml::from_str(s).expect("a document");
            Ok(
                WeaponDef::from_toml_table(doc["w"].as_table().expect("a table"))?
                    .expect("it is a weapon"),
            )
        };
        let d = read("[w.weapon]\nclass = \"sniper\"\n").expect("a weapon");
        assert_eq!(d.class, Some(WeaponClass::Sniper));
        assert_eq!(d.audio_class(), WeaponClass::Sniper);
        // Case and padding are the reader's to absorb, exactly as `kind`'s are.
        assert_eq!(
            read("[w.weapon]\nclass = \" Shotgun \"\n")
                .expect("a weapon")
                .audio_class(),
            WeaponClass::Shotgun
        );
        // **A typo is refused BY NAME.** Silently defaulting would give an
        // assault rifle a pistol's gunshot and say nothing.
        let e = read("[w.weapon]\nclass = \"assault_rifle\"\n").expect_err("refused");
        assert!(e.contains("assault_rifle") && e.contains("launcher"), "{e}");
        // …and a definition that names none takes the band. The seven values
        // are the registry's own per-class muzzle table.
        for (m, want) in [
            (0.12, WeaponClass::Pistol),
            (0.25, WeaponClass::Smg),
            (0.45, WeaponClass::Ar),
            (0.50, WeaponClass::Shotgun),
            (0.55, WeaponClass::Dmr),
            (0.65, WeaponClass::Sniper),
            (0.70, WeaponClass::Launcher),
        ] {
            let d = WeaponDef {
                muzzle_forward_m: m,
                ..Default::default()
            };
            assert_eq!(d.class, None);
            assert_eq!(d.audio_class(), want, "muzzle {m} m");
            assert_eq!(WeaponClass::from_muzzle_forward_m(m), want);
        }
        // The seven names round-trip and the indices are the enumeration's own.
        for (i, c) in WeaponClass::ALL.into_iter().enumerate() {
            assert_eq!(WeaponClass::from_name(c.name()), Some(c));
            assert_eq!(c.index() as usize, i);
        }
        assert_eq!(WeaponClass::from_name("carbine"), None);
    }

    /// **THE EJECTION PORT** (wave WPN2c) — three more numbers across the same
    /// by-name door, clamped rather than refused, and enumerated by `names()`.
    #[test]
    fn the_ejection_port_is_three_numbers_on_the_tuning_door() {
        let mut d = WeaponDef::default();
        assert!(d.set("eject_offset_m", 0.11));
        assert!(d.set("eject_dir_deg", 95.0));
        assert!(d.set("eject_speed_mps", 3.5));
        assert!((d.eject_offset_m - 0.11).abs() < 1e-12);
        assert!((d.eject_dir_deg - 95.0).abs() < 1e-12);
        assert!((d.eject_speed_mps - 3.5).abs() < 1e-12);
        // Clamped, on `CameraTuning`'s rule: a slider dragged past a bound wants
        // the bound.
        assert!(d.set("eject_speed_mps", 1e9));
        assert!((d.eject_speed_mps - 20.0).abs() < 1e-12);
        assert!(!d.set("eject_speed_mps", f64::NAN));
        for n in ["eject_offset_m", "eject_dir_deg", "eject_speed_mps"] {
            assert!(WeaponDef::names().contains(&n), "{n} is not enumerated");
        }
        // …and the sub-table they are authored under is on the reader's list.
        assert!(WEAPON_SUB_TABLES.contains(&"eject"));
        let doc: toml::Value =
            toml::from_str("[w.weapon.eject]\neject_speed_mps = 4.0\n").expect("a document");
        let d = WeaponDef::from_toml_table(doc["w"].as_table().expect("a table"))
            .expect("a weapon")
            .expect("it is a weapon");
        assert!((d.eject_speed_mps - 4.0).abs() < 1e-12);
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
