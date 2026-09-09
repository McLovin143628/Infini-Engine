//! **Modular weapon attachments** (wave WPN2d): the ten slots, the
//! multiplicative modifier fold, and the 530-row catalogue behind them.
//!
//! # It is the doc's own design, and the fold is the doc's own fold
//!
//! The user's research doc closes its attachment chapter with a Rust sketch —
//! `AttachmentSlot` (ten variants), `StatModifiers` (multiplicative on damage,
//! range, recoil, ADS, move and velocity, with a magazine override) and
//! `calculate_effective_stats`, which walks the equipped attachments and
//! multiplies. [`AttachmentSlot`] is that enum verbatim, [`StatModifiers`] is
//! that struct plus **one** field this engine needs and the doc's audio chapter
//! keeps somewhere else ([`StatModifiers::loudness_mult`] — see its doc), and
//! [`StatModifiers::compose`] is that fold.
//!
//! # THE FOLD PRODUCES A `WeaponDef`, WHICH IS THE WHOLE POINT
//!
//! There is no second copy of a rule here. The fold does not decide what a
//! suppressed rifle sounds like or how fast a bipodded DMR walks: it produces a
//! [`crate::weapon::WeaponDef`] with different numbers in it, and every rule in
//! the engine — `try_fire`, `damage_at`, `shot_direction`, `report_layers`,
//! `RecoilProfile::of`, `equipped_move_speed_scale`, `feel::ads_speed`,
//! `casing::eject_velocity` — reads that def exactly as it read the base one.
//! [`crate::weapon::equipped_def`] is the single door and it applies the fold
//! **inside itself**, so a caller cannot get the unmodified numbers by accident
//! and a wave that adds a rule gets attachments for free.
//!
//! The one-door law's own test is `wpn2d_gate::
//! the_attachment_fold_has_one_door_and_the_rules_are_not_copied`.
//!
//! # It costs ZERO schema
//!
//! Nothing in this module derives `Serialize`. The catalogue is an
//! `include_str!` of a generated table ([`ATTACHMENTS_TOML`]) on
//! `crate::weapon::WEAPON_REGISTRY_TOML`'s own terms, and what a *character* has
//! equipped is [`crate::weapon::WeaponState::attach`] — a runtime component.
//! Scene v27 and `ScenePayload` 13 are untouched by this wave.
//!
//! # The equipped set is SIM STATE
//!
//! An attachment changes where a bullet goes, how loud the street thinks the
//! shot was and how many rounds are left, so two hosts that disagreed about it
//! have diverged. [`crate::weapon::weapon_state_bytes`] folds the equipped
//! indices at the **tail** of its row and only when something is equipped, which
//! is `crate::ballistics::round_state_bytes`' empty-when-nothing-is-flying rule
//! one level down: a weapon with a bare rail folds exactly the bytes it folded
//! before this wave.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::weapon::{WeaponClass, WeaponDef};

/// **The catalogue**, as a generated table — 530 rows, the research doc's own
/// per-class attachment lists.
///
/// `crate::weapon::WEAPON_REGISTRY_TOML`'s route verbatim: a `&'static str` in
/// Ring 0, so the gates, the island generator, the inventory panel and the fixed
/// step all read one copy of one table and no schema moves. The file's own
/// header states the slot mapping, the row format and the keyword rule every
/// number was derived by.
pub const ATTACHMENTS_TOML: &str = include_str!("attachments.toml");

/// **How many slots a weapon has** — the doc's enum has ten variants and this is
/// the count, named so an array and a fold cannot disagree about it.
pub const ATTACHMENT_SLOTS: usize = 10;

/// **The most attachments the catalogue may hold.**
///
/// Four thousand, `crate::item::MAX_ITEM_DEFS`' own number and its own reason: a
/// bound on hostile content rather than a design limit. The shipped table is
/// 530 rows, so it does not bind — and the index type is `u16`, which this keeps
/// inside.
pub const MAX_ATTACHMENT_DEFS: usize = 4096;

/// **What an attachment goes on** — the research doc's `AttachmentSlot`, all ten
/// variants, in the doc's own order.
///
/// Not a wire enum: it never reaches a file, so the freeze-pin law has nothing
/// to say about it. What it does reach is `weapon_state_bytes`, as
/// [`index`](Self::index) — which is why the order is pinned by
/// [`ALL`](Self::ALL) and by a gate arm rather than left to the declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttachmentSlot {
    /// The muzzle device: a suppressor, a brake, a flash hider, a choke.
    Muzzle,
    /// The barrel, and a launcher's tube shroud.
    Barrel,
    /// The sight — irons, a red dot, a scope, a thermal.
    Optic,
    /// The stock, a DMR's cheek riser and a launcher's shoulder pad.
    Stock,
    /// What is under the handguard: a grip, a handstop, a bipod.
    Underbarrel,
    /// The magazine, and a shotgun's tube.
    Magazine,
    /// The pistol grip the trigger hand holds.
    RearGrip,
    /// The round itself, and a launcher's warhead.
    Ammunition,
    /// A laser or a tactical light.
    Laser,
    /// The trigger group, a sniper's bolt and a launcher's fire-control unit.
    TriggerAction,
}

impl AttachmentSlot {
    /// Every slot, in the doc's order. **The order is load-bearing**: it is what
    /// [`index`](Self::index) answers and what the trace folds.
    pub const ALL: [AttachmentSlot; ATTACHMENT_SLOTS] = [
        AttachmentSlot::Muzzle,
        AttachmentSlot::Barrel,
        AttachmentSlot::Optic,
        AttachmentSlot::Stock,
        AttachmentSlot::Underbarrel,
        AttachmentSlot::Magazine,
        AttachmentSlot::RearGrip,
        AttachmentSlot::Ammunition,
        AttachmentSlot::Laser,
        AttachmentSlot::TriggerAction,
    ];

    /// The name the catalogue's TOML spells it with.
    pub fn name(self) -> &'static str {
        match self {
            AttachmentSlot::Muzzle => "muzzle",
            AttachmentSlot::Barrel => "barrel",
            AttachmentSlot::Optic => "optic",
            AttachmentSlot::Stock => "stock",
            AttachmentSlot::Underbarrel => "underbarrel",
            AttachmentSlot::Magazine => "magazine",
            AttachmentSlot::RearGrip => "rear_grip",
            AttachmentSlot::Ammunition => "ammunition",
            AttachmentSlot::Laser => "laser",
            AttachmentSlot::TriggerAction => "trigger_action",
        }
    }

    /// The slot that name spells, or `None` — a refusal is a value.
    pub fn from_name(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|c| c.name() == s)
    }

    /// Its position in [`ALL`](Self::ALL) — the number the trace folds and the
    /// index into [`WeaponState::attach`](crate::weapon::WeaponState::attach).
    pub fn index(self) -> usize {
        match self {
            AttachmentSlot::Muzzle => 0,
            AttachmentSlot::Barrel => 1,
            AttachmentSlot::Optic => 2,
            AttachmentSlot::Stock => 3,
            AttachmentSlot::Underbarrel => 4,
            AttachmentSlot::Magazine => 5,
            AttachmentSlot::RearGrip => 6,
            AttachmentSlot::Ammunition => 7,
            AttachmentSlot::Laser => 8,
            AttachmentSlot::TriggerAction => 9,
        }
    }

    /// The slot at that position, or `None` past the end.
    pub fn from_index(i: usize) -> Option<Self> {
        Self::ALL.get(i).copied()
    }
}

/// **Which accessory mesh an attachment draws**, and where.
///
/// Four, because the FPS Weapon Bundle's `Accessories/` folder holds exactly
/// four static meshes (`SM_Scope_25x56_X`, `SM_T4_Sight`, `SM_Suppressor5`,
/// `SM_Vertgrip`) and a fifth name would be a row pointing at nothing. An
/// attachment with no art is [`None`](Self::None), which is most of them: a
/// hollow-point round and a stippled grip change numbers, not silhouettes.
///
/// The art is **local-only**. The ids are committed; the meshes are the user's
/// licensed Unreal content and are bound at runtime through
/// `crate::weapon::WeaponMeshes`, so a checkout without them draws the weapon
/// and no accessory rather than failing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum AttachmentArt {
    /// Nothing is drawn.
    #[default]
    None,
    /// A magnified optic, on top of the receiver.
    Scope,
    /// A reflex or micro sight, on top of the receiver.
    Sight,
    /// A can at the muzzle.
    Suppressor,
    /// A vertical grip under the handguard.
    Vertgrip,
}

impl AttachmentArt {
    /// Every art key the catalogue may name, in declaration order.
    pub const ALL: [AttachmentArt; 5] = [
        AttachmentArt::None,
        AttachmentArt::Scope,
        AttachmentArt::Sight,
        AttachmentArt::Suppressor,
        AttachmentArt::Vertgrip,
    ];

    /// The name the catalogue spells it with; `"-"` for [`None`](Self::None).
    pub fn name(self) -> &'static str {
        match self {
            AttachmentArt::None => "-",
            AttachmentArt::Scope => "scope",
            AttachmentArt::Sight => "sight",
            AttachmentArt::Suppressor => "suppressor",
            AttachmentArt::Vertgrip => "vertgrip",
        }
    }

    /// The art that name spells, or `None` on an unknown key — which the parser
    /// turns into a refusal BY NAME rather than a silent [`None`](Self::None),
    /// because a typo that dropped a scope would be a row with no reader.
    pub fn from_name(s: &str) -> Option<Self> {
        let s = s.trim();
        Self::ALL.into_iter().find(|a| a.name() == s)
    }
}

/// **What one attachment does to a weapon** — the doc's `StatModifiers`.
///
/// Every field but [`ads_time_delta_ms`](Self::ads_time_delta_ms) and
/// [`mag_capacity_override`](Self::mag_capacity_override) is a **multiplier**,
/// which is the doc's own design and the reason the fold is order-free: three
/// attachments compose to the same weapon whichever order they were bolted on
/// in, which is not true of additive deltas and is the property a player expects
/// from a gunsmith bench.
///
/// `f64` and not the doc's `f32`: these numbers reach `damage_at`, which reaches
/// `state_bytes`, and the P14 law wants one width on a trace path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StatModifiers {
    /// On [`WeaponDef::damage_j`].
    pub damage_mult: f64,
    /// On [`WeaponDef::range_m`], [`WeaponDef::effective_range_m`] and
    /// [`WeaponDef::max_range_m`] together — a barrel that reaches further
    /// reaches further at every point of the curve, and scaling only one of the
    /// three would bend the curve as a side effect of lengthening it.
    pub range_mult: f64,
    /// On [`WeaponDef::recoil_intensity`], which is what
    /// `crate::feel::RecoilProfile::of` is built from — so a compensator moves
    /// both springs and the aim kick through the one mapping rather than through
    /// a second copy of it.
    pub recoil_mult: f64,
    /// **Added** to [`WeaponDef::ads_time_ms`], milliseconds. The doc's own
    /// `ads_time_delta_ms`, and the one non-multiplicative field in its struct:
    /// a heavy optic costs a fixed number of milliseconds to bring up rather
    /// than a percentage of whatever the base was.
    pub ads_time_delta_ms: f64,
    /// On [`WeaponDef::move_speed_mult`].
    pub move_speed_mult: f64,
    /// On [`WeaponDef::muzzle_speed_mps`].
    pub velocity_mult: f64,
    /// **On [`WeaponDef::report_max_m`] and on the report's own volume** — this
    /// engine's eleventh field, and the one that makes a suppressor mean
    /// something.
    ///
    /// The doc's `StatModifiers` has no loudness because its audio work is a
    /// separate chapter, and a `Low-Profile Suppressor` row that changed six
    /// numbers and left the gunshot alone would be exactly the defect this
    /// repository keeps paying for: a row with no reader. `report_max_m` is the
    /// per-weapon reach `crate::weapon::report_layers` builds four `Play`
    /// commands out of, and the layer volumes are the other half, so ONE number
    /// scaling both is "how loud this weapon is" expressed once.
    pub loudness_mult: f64,
    /// Replaces [`WeaponDef::magazine`] outright. The doc's own
    /// `mag_capacity_override: Option<u32>`: a 60-round drum is not 2× a
    /// 30-round stick, it is sixty.
    pub mag_capacity_override: Option<u32>,
}

impl Default for StatModifiers {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl StatModifiers {
    /// **The identity** — what an empty rail does, and what
    /// [`compose`](Self::compose) starts from.
    pub const IDENTITY: StatModifiers = StatModifiers {
        damage_mult: 1.0,
        range_mult: 1.0,
        recoil_mult: 1.0,
        ads_time_delta_ms: 0.0,
        move_speed_mult: 1.0,
        velocity_mult: 1.0,
        loudness_mult: 1.0,
        mag_capacity_override: None,
    };

    /// The floor and ceiling every multiplier is held between.
    pub const MULT_MIN: f64 = 0.50;
    /// See [`MULT_MIN`](Self::MULT_MIN).
    pub const MULT_MAX: f64 = 2.00;
    /// The floor on [`move_speed_mult`](Self::move_speed_mult) — tighter than
    /// the rest, because a stock that made a character 50 % slower would read as
    /// a broken control rather than as a heavy stock.
    pub const MOVE_MIN: f64 = 0.80;
    /// See [`MOVE_MIN`](Self::MOVE_MIN).
    pub const MOVE_MAX: f64 = 1.20;
    /// The floor on [`loudness_mult`](Self::loudness_mult). A fifth: a
    /// suppressed .22 is quiet, not silent, and a zero would put a gunshot
    /// outside every witness rule in the engine.
    pub const LOUD_MIN: f64 = 0.20;
    /// See [`LOUD_MIN`](Self::LOUD_MIN).
    pub const LOUD_MAX: f64 = 2.00;
    /// The bound on [`ads_time_delta_ms`](Self::ads_time_delta_ms), each way.
    pub const ADS_DELTA_MAX_MS: f64 = 150.0;

    /// **Hold every field inside its bound**, and turn a non-finite into the
    /// identity's value rather than into a NaN that would poison a weapon.
    ///
    /// One function, applied by the parser to a hand-edited row AND by the
    /// generator's own arithmetic, so the file and the reader cannot disagree
    /// about what is representable.
    pub fn clamped(self) -> Self {
        let m = |v: f64, lo: f64, hi: f64| if v.is_finite() { v.clamp(lo, hi) } else { 1.0 };
        Self {
            damage_mult: m(self.damage_mult, Self::MULT_MIN, Self::MULT_MAX),
            range_mult: m(self.range_mult, Self::MULT_MIN, Self::MULT_MAX),
            recoil_mult: m(self.recoil_mult, Self::MULT_MIN, Self::MULT_MAX),
            ads_time_delta_ms: if self.ads_time_delta_ms.is_finite() {
                self.ads_time_delta_ms
                    .clamp(-Self::ADS_DELTA_MAX_MS, Self::ADS_DELTA_MAX_MS)
            } else {
                0.0
            },
            move_speed_mult: m(self.move_speed_mult, Self::MOVE_MIN, Self::MOVE_MAX),
            velocity_mult: m(self.velocity_mult, Self::MULT_MIN, Self::MULT_MAX),
            loudness_mult: m(self.loudness_mult, Self::LOUD_MIN, Self::LOUD_MAX),
            mag_capacity_override: self
                .mag_capacity_override
                .map(|n| n.clamp(1, crate::weapon::MAX_MAGAZINE)),
        }
    }

    /// **The doc's fold, for one more attachment** — multiplicative on the six,
    /// additive on the ADS delta, last-writer-wins on the magazine.
    ///
    /// The magazine is the one field that cannot compose: two overrides are two
    /// capacities and there is no arithmetic that turns them into one. The rule
    /// is that the LAST one folded wins, and the fold walks
    /// [`AttachmentSlot::ALL`] in order, so "last" is a fact about the slot
    /// order rather than about which slot a player filled first — which is what
    /// keeps the answer the same on two hosts.
    pub fn compose(self, other: &StatModifiers) -> Self {
        Self {
            damage_mult: self.damage_mult * other.damage_mult,
            range_mult: self.range_mult * other.range_mult,
            recoil_mult: self.recoil_mult * other.recoil_mult,
            ads_time_delta_ms: self.ads_time_delta_ms + other.ads_time_delta_ms,
            move_speed_mult: self.move_speed_mult * other.move_speed_mult,
            velocity_mult: self.velocity_mult * other.velocity_mult,
            loudness_mult: self.loudness_mult * other.loudness_mult,
            mag_capacity_override: other.mag_capacity_override.or(self.mag_capacity_override),
        }
    }

    /// Whether this is the identity — the question the fold asks to decide
    /// whether it has to touch a [`WeaponDef`] at all.
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// **Apply the fold to a definition** — the doc's `calculate_effective_stats`
    /// in this engine's own vocabulary.
    ///
    /// Every product goes back through [`WeaponDef::set`], which is the door
    /// that clamps: a range multiplier cannot push a barrel past
    /// `MAX_RANGE_M`, a magazine override cannot exceed `MAX_MAGAZINE` and a
    /// recoil multiplier cannot leave the 1–10 scale. Writing the fields
    /// directly would be a second set of bounds.
    pub fn apply(&self, def: &WeaponDef) -> WeaponDef {
        if self.is_identity() {
            return *def;
        }
        let m = self.clamped();
        let mut out = *def;
        out.set("damage_j", def.damage_j * m.damage_mult);
        out.set("range_m", def.range_m * m.range_mult);
        // A zero stays a zero: `max_range_m == 0` is what makes a curve FLAT
        // (`ballistics::damage_curve_j`), and scaling a zero by 1.15 is still a
        // zero, so no branch is needed and none is written.
        out.set("effective_range_m", def.effective_range_m * m.range_mult);
        out.set("max_range_m", def.max_range_m * m.range_mult);
        out.set("recoil_intensity", def.recoil_intensity * m.recoil_mult);
        out.set("ads_time_ms", def.ads_time_ms + m.ads_time_delta_ms);
        out.set("move_speed_mult", def.move_speed_mult * m.move_speed_mult);
        out.set("muzzle_speed_mps", def.muzzle_speed_mps * m.velocity_mult);
        out.set("report_max_m", def.report_max_m * m.loudness_mult);
        out.set("report_gain", def.report_gain * m.loudness_mult);
        if let Some(n) = m.mag_capacity_override {
            out.set("magazine", f64::from(n));
        }
        out
    }
}

/// **One attachment.**
#[derive(Clone, Debug, PartialEq)]
pub struct AttachmentDef {
    /// The catalogue key — the label, slugged. Unique within a class and a slot.
    pub id: String,
    /// What a panel calls it — the research doc's own name.
    pub label: String,
    /// Which class of weapon it fits. An attachment is class-specific in the
    /// doc's own tables (a pistol compensator is not an AR compensator), and
    /// refusing a cross-class fit is what stops a 100-round C-Mag on a revolver.
    pub class: WeaponClass,
    /// Where it goes.
    pub slot: AttachmentSlot,
    /// What it does.
    pub modifiers: StatModifiers,
    /// What it draws, if anything.
    pub art: AttachmentArt,
}

/// **The parsed catalogue.**
///
/// Indexed by a `u16` because that is what [`crate::weapon::WeaponState`] folds
/// into the trace: 530 rows today and [`MAX_ATTACHMENT_DEFS`] rows at most, both
/// inside `u16`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Attachments {
    rows: Vec<AttachmentDef>,
    by_key: BTreeMap<(WeaponClass, AttachmentSlot, String), u16>,
}

/// The sentinel [`crate::weapon::WeaponState::attach`] carries for an empty
/// slot.
///
/// `u16::MAX`, which [`MAX_ATTACHMENT_DEFS`] keeps a real index away from.
pub const NO_ATTACHMENT: u16 = u16::MAX;

impl Attachments {
    /// **Parse the catalogue.** A refusal is a value: an unknown class, an
    /// unknown slot, an unknown art key, a row with the wrong number of fields
    /// or a number that will not parse comes back as an error naming the row,
    /// because a silently skipped row is a suppressor nobody can fit.
    pub fn parse(text: &str) -> Result<Self, String> {
        let doc: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
        let table = doc
            .as_table()
            .ok_or_else(|| "an attachment catalogue is a table of classes".to_string())?;
        let mut out = Attachments::default();
        for (class_name, class_val) in table {
            let class = WeaponClass::from_name(class_name)
                .ok_or_else(|| format!("unknown weapon class [{class_name}]"))?;
            let slots = class_val
                .as_table()
                .ok_or_else(|| format!("[{class_name}] is not a table of slots"))?;
            for (slot_name, slot_val) in slots {
                let slot = AttachmentSlot::from_name(slot_name)
                    .ok_or_else(|| format!("unknown slot [{class_name}.{slot_name}]"))?;
                let rows = slot_val
                    .as_table()
                    .and_then(|t| t.get("rows"))
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("[{class_name}.{slot_name}] has no `rows` array"))?;
                for row in rows {
                    let line = row.as_str().ok_or_else(|| {
                        format!("[{class_name}.{slot_name}] holds a row that is not a string")
                    })?;
                    let def = parse_row(line, class, slot)?;
                    if out.rows.len() >= MAX_ATTACHMENT_DEFS {
                        return Err(format!(
                            "more than {MAX_ATTACHMENT_DEFS} attachments; {} was the last taken",
                            def.id
                        ));
                    }
                    let key = (class, slot, def.id.clone());
                    if out.by_key.contains_key(&key) {
                        return Err(format!(
                            "duplicate attachment {} in [{class_name}.{slot_name}]",
                            def.id
                        ));
                    }
                    out.by_key.insert(key, out.rows.len() as u16);
                    out.rows.push(def);
                }
            }
        }
        Ok(out)
    }

    /// How many rows it holds.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The row at an index, or `None` — including for [`NO_ATTACHMENT`].
    pub fn get(&self, index: u16) -> Option<&AttachmentDef> {
        self.rows.get(index as usize)
    }

    /// **Find an attachment by class, slot and id** — the equip verb's door.
    pub fn find(&self, class: WeaponClass, slot: AttachmentSlot, id: &str) -> Option<u16> {
        self.by_key
            .get(&(class, slot, id.trim().to_ascii_lowercase()))
            .copied()
    }

    /// **Find one by id alone**, searching every class and slot in catalogue
    /// order — what a panel and a Blueprint have, because a player picks a
    /// *thing* rather than a (class, slot, id) triple.
    ///
    /// The first match in catalogue order wins, which is deterministic because
    /// the catalogue is a file.
    pub fn find_any(&self, id: &str) -> Option<u16> {
        let id = id.trim().to_ascii_lowercase();
        self.rows
            .iter()
            .position(|r| r.id == id)
            .map(|i| i as u16)
    }

    /// Every row that fits this class and slot, in catalogue order.
    pub fn for_slot(&self, class: WeaponClass, slot: AttachmentSlot) -> Vec<u16> {
        self.rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.class == class && r.slot == slot)
            .map(|(i, _)| i as u16)
            .collect()
    }

    /// How many rows this class carries.
    pub fn count_for(&self, class: WeaponClass) -> usize {
        self.rows.iter().filter(|r| r.class == class).count()
    }
}

fn parse_row(
    line: &str,
    class: WeaponClass,
    slot: AttachmentSlot,
) -> Result<AttachmentDef, String> {
    const FIELDS: usize = 11;
    let parts: Vec<&str> = line.split('|').collect();
    if parts.len() != FIELDS {
        return Err(format!(
            "attachment row {line:?} has {} fields, not {FIELDS}",
            parts.len()
        ));
    }
    let num = |i: usize| -> Result<f64, String> {
        parts[i]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("attachment row {line:?} field {i}: {e}"))
    };
    let mag = match parts[9].trim() {
        "-" => None,
        n => Some(
            n.parse::<u32>()
                .map_err(|e| format!("attachment row {line:?} magazine: {e}"))?,
        ),
    };
    let art = AttachmentArt::from_name(parts[10])
        .ok_or_else(|| format!("attachment row {line:?} names an unknown art {:?}", parts[10]))?;
    let id = parts[0].trim().to_ascii_lowercase();
    if id.is_empty() {
        return Err(format!("attachment row {line:?} has no id"));
    }
    Ok(AttachmentDef {
        id,
        label: parts[1].trim().to_string(),
        class,
        slot,
        modifiers: StatModifiers {
            damage_mult: num(2)?,
            range_mult: num(3)?,
            recoil_mult: num(4)?,
            ads_time_delta_ms: num(5)?,
            move_speed_mult: num(6)?,
            velocity_mult: num(7)?,
            loudness_mult: num(8)?,
            mag_capacity_override: mag,
        }
        .clamped(),
        art,
    })
}

static CATALOGUE: OnceLock<Attachments> = OnceLock::new();

/// **The shipped catalogue**, parsed once.
///
/// A `OnceLock` and not a bevy resource, on `inf_core::job`'s own precedent: it
/// is a pure function of a `&'static str` compiled into the binary, so there is
/// nothing per-world about it and a resource would be one more thing a Simulate
/// session could forget to install. A catalogue that fails to parse is an
/// **empty** one and the failure is stated by
/// [`catalogue_error`] rather than by a panic in a fixed step — the refusal-is-
/// a-value law, applied to a file that ships inside the executable and therefore
/// cannot be wrong at runtime unless this crate is.
pub fn catalogue() -> &'static Attachments {
    CATALOGUE.get_or_init(|| Attachments::parse(ATTACHMENTS_TOML).unwrap_or_default())
}

/// The parse error, if the shipped catalogue does not parse — `None` when it
/// does, which is what the gate asserts.
pub fn catalogue_error() -> Option<String> {
    Attachments::parse(ATTACHMENTS_TOML).err()
}

/// **The fold, over one weapon's equipped set** — the doc's
/// `calculate_effective_stats`.
///
/// Walks [`AttachmentSlot::ALL`] in order (see
/// [`StatModifiers::compose`] for why the order matters) and composes every row
/// the catalogue knows. An index the catalogue does not know is skipped, which
/// is what a save from a build with a bigger table looks like.
pub fn fold(equipped: &[u16; ATTACHMENT_SLOTS]) -> StatModifiers {
    let cat = catalogue();
    let mut out = StatModifiers::IDENTITY;
    for slot in AttachmentSlot::ALL {
        let idx = equipped[slot.index()];
        if idx == NO_ATTACHMENT {
            continue;
        }
        if let Some(def) = cat.get(idx) {
            out = out.compose(&def.modifiers);
        }
    }
    out
}

/// **What the equipped set draws** — the art keys, in slot order, for the
/// accessory meshes `inf_physics::d3::gameplay::step_equipped_weapons` hangs off
/// the weapon.
pub fn equipped_art(equipped: &[u16; ATTACHMENT_SLOTS]) -> Vec<(AttachmentSlot, AttachmentArt)> {
    let cat = catalogue();
    let mut out = Vec::new();
    for slot in AttachmentSlot::ALL {
        let idx = equipped[slot.index()];
        if idx == NO_ATTACHMENT {
            continue;
        }
        if let Some(def) = cat.get(idx) {
            if def.art != AttachmentArt::None {
                out.push((slot, def.art));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_catalogue_parses_and_is_the_docs_own_lists() {
        assert_eq!(catalogue_error(), None, "the shipped catalogue must parse");
        let cat = catalogue();
        assert_eq!(cat.len(), 530, "the doc's own lists are 530 rows");
        // Per class, the doc's own counts. The AR's header names a `Lasers`
        // slot it never fills, so it is 80 like every other rifle class.
        for (class, want) in [
            (WeaponClass::Pistol, 80),
            (WeaponClass::Smg, 80),
            (WeaponClass::Ar, 80),
            (WeaponClass::Dmr, 80),
            (WeaponClass::Sniper, 80),
            (WeaponClass::Shotgun, 80),
            (WeaponClass::Launcher, 50),
        ] {
            assert_eq!(cat.count_for(class), want, "{class:?} row count");
        }
    }

    #[test]
    fn the_slot_order_is_the_index_and_the_names_round_trip() {
        for (i, s) in AttachmentSlot::ALL.into_iter().enumerate() {
            assert_eq!(s.index(), i, "{s:?} index");
            assert_eq!(AttachmentSlot::from_index(i), Some(s));
            assert_eq!(AttachmentSlot::from_name(s.name()), Some(s));
        }
        assert_eq!(AttachmentSlot::from_name("no_such_slot"), None);
        assert_eq!(AttachmentSlot::from_index(ATTACHMENT_SLOTS), None);
    }

    #[test]
    fn the_fold_is_order_free_and_the_identity_changes_nothing() {
        let cat = catalogue();
        let supp = cat
            .find(
                WeaponClass::Ar,
                AttachmentSlot::Muzzle,
                "tactical_monolithic_suppressor",
            )
            .expect("the AR's monolithic suppressor");
        let mag = cat
            .find(WeaponClass::Ar, AttachmentSlot::Magazine, "45_round_extended_mag")
            .expect("the AR's 45-round mag");
        let mut a = [NO_ATTACHMENT; ATTACHMENT_SLOTS];
        a[AttachmentSlot::Muzzle.index()] = supp;
        a[AttachmentSlot::Magazine.index()] = mag;
        let folded = fold(&a);
        assert!(folded.loudness_mult < 0.5, "a suppressor is quieter");
        assert_eq!(folded.mag_capacity_override, Some(45));
        // The identity leaves a definition alone, byte for byte.
        let base = WeaponDef::default();
        assert_eq!(StatModifiers::IDENTITY.apply(&base), base);
        assert_eq!(fold(&[NO_ATTACHMENT; ATTACHMENT_SLOTS]), StatModifiers::IDENTITY);
    }

    #[test]
    fn a_broken_row_is_refused_by_name() {
        assert!(Attachments::parse("[ar.muzzle]\nrows = [\"too|few\"]\n").is_err());
        assert!(Attachments::parse(
            "[nosuch.muzzle]\nrows = [\"a|A|1|1|1|0|1|1|1|-|-\"]\n"
        )
        .is_err());
        assert!(Attachments::parse(
            "[ar.nosuch]\nrows = [\"a|A|1|1|1|0|1|1|1|-|-\"]\n"
        )
        .is_err());
        assert!(Attachments::parse(
            "[ar.muzzle]\nrows = [\"a|A|1|1|1|0|1|1|1|-|hat\"]\n"
        )
        .is_err());
        // And a good one is taken.
        let ok = Attachments::parse("[ar.muzzle]\nrows = [\"a|A|1|1|1|0|1|1|1|-|scope\"]\n")
            .expect("a well-formed row");
        assert_eq!(ok.len(), 1);
        assert_eq!(ok.get(0).unwrap().art, AttachmentArt::Scope);
    }

    #[test]
    fn every_modifier_is_clamped_into_its_own_bound() {
        let wild = StatModifiers {
            damage_mult: 1.0e9,
            range_mult: -4.0,
            recoil_mult: f64::NAN,
            ads_time_delta_ms: 9_999.0,
            move_speed_mult: 0.0,
            velocity_mult: f64::INFINITY,
            loudness_mult: 0.0,
            mag_capacity_override: Some(0),
        }
        .clamped();
        assert_eq!(wild.damage_mult, StatModifiers::MULT_MAX);
        assert_eq!(wild.range_mult, StatModifiers::MULT_MIN);
        assert_eq!(wild.recoil_mult, 1.0, "a NaN takes the identity");
        assert_eq!(wild.ads_time_delta_ms, StatModifiers::ADS_DELTA_MAX_MS);
        assert_eq!(wild.move_speed_mult, StatModifiers::MOVE_MIN);
        assert_eq!(wild.velocity_mult, 1.0, "an infinity takes the identity");
        assert_eq!(wild.loudness_mult, StatModifiers::LOUD_MIN);
        assert_eq!(wild.mag_capacity_override, Some(1));
    }
}
