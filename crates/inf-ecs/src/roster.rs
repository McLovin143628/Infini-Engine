//! **THE ROSTER** (wave VEH3f) -- the research doc's 153 lore-named vehicles over
//! its eighteen classes, their class DEFAULTS, and the two trailers the
//! articulated rig tows.
//!
//! # What this module IS, and why it is Ring 0
//!
//! Two committed TOML files, [`ROSTER_TOML`] and [`ROSTER_CLASSES_TOML`], read
//! through the one parser every catalogue row has used since VEH1a
//! ([`VehicleDef::from_toml_table_with`]). They live in `inf-ecs` rather than in
//! the editor crate beside `ISLAND_VEHICLES_TOML` because TRAFFIC draws from
//! them: `traffic::catalogue_row` runs inside the fixed step on both hosts, and
//! a Ring-0 step cannot reach a Ring-1 `&str`. The island's eleven VEH1a-VEH2c
//! rows stay where they were and parse exactly as they did (no `class` key, no
//! class defaults) -- the roster is ADDED beside them, never merged over them.
//!
//! # The route VEH3a ruled: no asset kind, no schema
//!
//! `VehicleDef` derives no `Serialize`; nothing here reaches a file the engine
//! writes. A level that wants the roster at RUNTIME (an author's own fleet, after
//! cook) calls the `vehicle.define` Blueprint node, which is `item.define`'s
//! shape exactly -- the TOML rides the class's `.inf_act` bytes into Simulate, a
//! PIE payload and a cooked pack with no scene or payload move -- and merges it
//! into the world's [`VehicleDefsRes`]. Scene v28, payload 13.
//!
//! # The class DEFAULTS are the doc's handling-profile table
//!
//! The doc's table has four columns -- CoM, suspension, friction, acoustic --
//! and five rows (coupe/sedan; emergency/SUV/truck; military/heavy cargo;
//! air; boats). Each of this module's classes writes those four columns as
//! numbers (`cog_height_m`; `travel_m` plus the damping ratio its rows' dampers
//! are cut at; `tyre_*_peak_slip`/`tyre_slide_frac`/`tyre_load_sensitivity`;
//! `cylinders`/`engine_voice_kind`/`firing_order_variant`/`turbo_boost_max`), and
//! a row applies its own keys over them.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::vehicle::{VehicleDef, VehicleDefs};

/// The roster's rows -- 153 lore rows and two trailers, generated from the
/// research doc's roster and committed as authored content.
pub const ROSTER_TOML: &str = include_str!("roster.toml");

/// The roster's class defaults -- the doc's handling-profile table as data.
pub const ROSTER_CLASSES_TOML: &str = include_str!("roster_classes.toml");

/// **The doc's eighteen classes**, plus the trailer the articulated rig tows
/// (which is not one of them, and is counted apart).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RosterClass {
    /// Two-door coupes and sports cars.
    Coupe,
    /// Four-door sedans and commuters.
    Sedan,
    /// Four-door SUVs.
    Suv,
    /// Pickups, two-door and four-door.
    Truck,
    /// Four-door jeeps and off-roaders.
    Jeep,
    /// Hummers and heavy SUV platforms.
    Hummer,
    /// Emergency vehicles.
    Emergency,
    /// Military vehicles.
    Military,
    /// Construction plant.
    Construction,
    /// Utility vehicles.
    Utility,
    /// Freight tractors.
    Freight,
    /// Cargo trucks.
    Cargo,
    /// Buses.
    Bus,
    /// Taxis and rideshares.
    Service,
    /// Vans.
    Van,
    /// Airplanes -- DORMANT until VEH3g.
    Airplane,
    /// Helicopters.
    Helicopter,
    /// Boats and ships.
    Marine,
    /// **Not a doc class**: the articulated rig's towed half.
    Trailer,
}

impl RosterClass {
    /// Every class, the eighteen in the doc's order and then the trailer.
    pub const ALL: [RosterClass; 19] = [
        RosterClass::Coupe,
        RosterClass::Sedan,
        RosterClass::Suv,
        RosterClass::Truck,
        RosterClass::Jeep,
        RosterClass::Hummer,
        RosterClass::Emergency,
        RosterClass::Military,
        RosterClass::Construction,
        RosterClass::Utility,
        RosterClass::Freight,
        RosterClass::Cargo,
        RosterClass::Bus,
        RosterClass::Service,
        RosterClass::Van,
        RosterClass::Airplane,
        RosterClass::Helicopter,
        RosterClass::Marine,
        RosterClass::Trailer,
    ];

    /// The stable name a row's `class = "..."` spells.
    pub fn name(self) -> &'static str {
        match self {
            RosterClass::Coupe => "coupe",
            RosterClass::Sedan => "sedan",
            RosterClass::Suv => "suv",
            RosterClass::Truck => "truck",
            RosterClass::Jeep => "jeep",
            RosterClass::Hummer => "hummer",
            RosterClass::Emergency => "emergency",
            RosterClass::Military => "military",
            RosterClass::Construction => "construction",
            RosterClass::Utility => "utility",
            RosterClass::Freight => "freight",
            RosterClass::Cargo => "cargo",
            RosterClass::Bus => "bus",
            RosterClass::Service => "service",
            RosterClass::Van => "van",
            RosterClass::Airplane => "airplane",
            RosterClass::Helicopter => "helicopter",
            RosterClass::Marine => "marine",
            RosterClass::Trailer => "trailer",
        }
    }

    /// The class a name means, or `None` -- a refusal, never a silent sedan.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == name)
    }

    /// **How many rows the research doc gives this class** -- the count the
    /// roster is held to (`0` for the trailer, which the doc does not name).
    pub fn doc_count(self) -> usize {
        match self {
            RosterClass::Coupe | RosterClass::Sedan | RosterClass::Suv | RosterClass::Truck => 20,
            RosterClass::Marine => 8,
            RosterClass::Trailer => 0,
            _ => 5,
        }
    }

    /// **Whether a row of this class drives on WHEELS in this wave.** Air and
    /// sea rows parse, draw and (the helicopters and boats) fly or float on
    /// VEH2c's classes; they are never a road car and never traffic.
    pub fn road(self) -> bool {
        !matches!(
            self,
            RosterClass::Airplane | RosterClass::Helicopter | RosterClass::Marine
        )
    }

    /// **What is dormant about this class until VEH3g**, or `None`.
    pub fn dormant(self) -> Option<&'static str> {
        match self {
            RosterClass::Airplane => {
                Some("no lift, stall or control surfaces: rolls on its gear (VEH3g)")
            }
            RosterClass::Helicopter => Some(
                "flies on VEH2c's RotorVehicle; the winch and the blade-slap voice are VEH3g's",
            ),
            RosterClass::Marine => {
                Some("floats on VEH2c's HullVehicle; planing, sail and hull voices are VEH3g's")
            }
            _ => None,
        }
    }

    /// **The kerb's weight** for this class -- how often a parked car at a kerb
    /// slot is one of these (wave VEH3f). Zero is the KERB TRAP closed per
    /// class: no bus, semi, dozer, cruiser or aeroplane is ever parked at a kerb.
    pub fn parked_weight(self) -> f64 {
        match self {
            RosterClass::Sedan => 26.0,
            RosterClass::Suv => 18.0,
            RosterClass::Truck => 12.0,
            RosterClass::Coupe => 8.0,
            RosterClass::Van => 8.0,
            RosterClass::Service => 6.0,
            RosterClass::Jeep => 4.0,
            RosterClass::Hummer => 2.0,
            _ => 0.0,
        }
    }

    /// **The road's weight** for this class -- how often a car that DRIVES a
    /// circuit (the day and night circuits; commuters keep the kerb's mix,
    /// because a commuter parks at a kerb at both ends of its day) is one of
    /// these. Buses, cargo and utility trucks appear here and never at a kerb.
    pub fn circuit_weight(self) -> f64 {
        match self {
            RosterClass::Sedan => 22.0,
            RosterClass::Suv => 15.0,
            RosterClass::Truck => 10.0,
            RosterClass::Service => 10.0,
            RosterClass::Van => 9.0,
            RosterClass::Coupe => 6.0,
            RosterClass::Bus => 6.0,
            RosterClass::Cargo => 5.0,
            RosterClass::Utility => 3.0,
            RosterClass::Jeep => 3.0,
            RosterClass::Hummer => 1.0,
            _ => 0.0,
        }
    }
}

/// **The class defaults, by class** -- one TOML table of vehicle keys each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClassProfiles(pub BTreeMap<RosterClass, toml::map::Map<String, toml::Value>>);

impl ClassProfiles {
    /// Parse a class-defaults document: every top-level table is a class name
    /// with a `[<class>.vehicle]` sub-table. An unknown class name is refused.
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let doc: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
        let table = doc
            .as_table()
            .ok_or_else(|| "class defaults are a table of classes".to_string())?;
        let mut out = BTreeMap::new();
        for (name, value) in table {
            let class = RosterClass::from_name(name)
                .ok_or_else(|| format!("unknown roster class `{name}`"))?;
            let vehicle = value
                .get("vehicle")
                .and_then(|v| v.as_table())
                .ok_or_else(|| format!("class `{name}` has no `[{name}.vehicle]` table"))?;
            // Every key must be one the row parser takes: probe it through
            // the SAME door, so a typo in a default is a build-time refusal and
            // not a class that silently keeps `VehicleDef::default()`.
            VehicleDef::default()
                .apply_table(vehicle)
                .map_err(|e| format!("class `{name}`: {e}"))?;
            out.insert(class, vehicle.clone());
        }
        Ok(Self(out))
    }

    /// One class's defaults.
    pub fn get(&self, class: RosterClass) -> Option<&toml::map::Map<String, toml::Value>> {
        self.0.get(&class)
    }
}

/// **The committed class defaults**, parsed once.
///
/// Panics on a malformed file, which is right for a committed `include_str!`:
/// a class table that does not parse is a build error, and
/// `the_roster_parses_and_holds_the_docs_counts` is the arm that says it does.
pub fn class_profiles() -> &'static ClassProfiles {
    static CLASSES: OnceLock<ClassProfiles> = OnceLock::new();
    CLASSES.get_or_init(|| {
        ClassProfiles::from_toml(ROSTER_CLASSES_TOML).expect("the roster's class defaults parse")
    })
}

/// **The committed roster**, parsed once, in id order.
pub fn roster() -> &'static VehicleDefs {
    static ROSTER: OnceLock<VehicleDefs> = OnceLock::new();
    ROSTER.get_or_init(|| {
        let mut defs = VehicleDefs::default();
        defs.merge_toml(ROSTER_TOML)
            .expect("the committed roster parses");
        defs
    })
}

/// **A row's display label** -- the lore name the doc gives it ("Vapid
/// Dominator GT"), read from the row's own `label` key.
pub fn roster_label(id: &str) -> Option<&'static str> {
    static LABELS: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    LABELS
        .get_or_init(|| {
            let doc: toml::Value = toml::from_str(ROSTER_TOML).unwrap_or(toml::Value::Integer(0));
            doc.as_table()
                .map(|t| {
                    t.iter()
                        .filter_map(|(id, v)| {
                            v.get("label")
                                .and_then(|l| l.as_str())
                                .map(|l| (id.clone(), l.to_string()))
                        })
                        .collect()
                })
                .unwrap_or_default()
        })
        .get(id)
        .map(String::as_str)
}

/// **Every row id of one class**, in id order.
pub fn rows_of(class: RosterClass) -> Vec<&'static str> {
    roster()
        .0
        .iter()
        .filter(|(_, d)| d.roster_class == Some(class))
        .map(|(id, _)| id.as_str())
        .collect()
}

/// **A deterministic weighted draw** over the roster -- first a class by
/// `weight`, then a row of that class uniformly, among the rows `fits` accepts.
///
/// `unit` is a draw in `[0, 1)` from a counter-hash (`crowd::agent_unit`), never
/// an RNG: the same car gets the same row on both hosts and in every run. The
/// class draw uses the high part of `unit` and the row draw the fractional
/// remainder, so one draw decides both without a second salt.
pub fn pick_row(
    unit: f64,
    weight: impl Fn(RosterClass) -> f64,
    fits: impl Fn(&VehicleDef) -> bool,
) -> Option<(&'static str, &'static VehicleDef)> {
    let classes: Vec<(RosterClass, f64, Vec<(&'static str, &'static VehicleDef)>)> =
        RosterClass::ALL
            .into_iter()
            .filter_map(|c| {
                let w = weight(c);
                if !(w > 0.0) {
                    return None;
                }
                let rows: Vec<(&'static str, &'static VehicleDef)> = roster()
                    .0
                    .iter()
                    .filter(|(_, d)| d.roster_class == Some(c) && fits(d))
                    .map(|(id, d)| (id.as_str(), d))
                    .collect();
                (!rows.is_empty()).then_some((c, w, rows))
            })
            .collect();
    let total: f64 = classes.iter().map(|(_, w, _)| w).sum();
    if !(total > 0.0) {
        return None;
    }
    let u = if unit.is_finite() {
        unit.clamp(0.0, 1.0 - 1e-12)
    } else {
        0.0
    };
    let mut at = u * total;
    for (_, w, rows) in &classes {
        if at < *w {
            let frac = (at / w).clamp(0.0, 1.0 - 1e-12);
            let i = ((frac * rows.len() as f64) as usize).min(rows.len() - 1);
            return Some(rows[i]);
        }
        at -= w;
    }
    classes.last().and_then(|(_, _, rows)| rows.last().copied())
}

/// **The world's vehicle catalogue** (wave VEH3f) -- what the `vehicle.define`
/// Blueprint node merges into, on `item.define`'s shape.
///
/// A resource rather than a component, and never persisted: a level that wants
/// it says so at `BeginPlay`, so the rows ride the class's own bytes into every
/// host with no scene or payload move.
#[derive(bevy_ecs::prelude::Resource, Clone, Debug, Default, PartialEq)]
pub struct VehicleDefsRes(pub VehicleDefs);

/// The world's catalogue, created empty on first use.
pub fn vehicle_defs_mut(world: &mut crate::world::EcsWorld) -> &mut VehicleDefs {
    let w = world.world_mut();
    if !w.contains_resource::<VehicleDefsRes>() {
        w.insert_resource(VehicleDefsRes::default());
    }
    &mut w
        .get_resource_mut::<VehicleDefsRes>()
        .expect("just inserted")
        .into_inner()
        .0
}

/// The world's catalogue, if a `vehicle.define` ever ran.
pub fn vehicle_defs(world: &crate::world::EcsWorld) -> Option<&VehicleDefs> {
    world.world().get_resource::<VehicleDefsRes>().map(|r| &r.0)
}

/// **Which construction-pack body a roster row draws when its art is on this
/// machine** (wave VEH3f) -- the machine key the UE bridge imports under
/// `Content/UE/Vehicles/<key>` and the per-body TOML sidecar names. `None` for
/// every row that has no art anywhere (all of them but seven).
pub fn art_of(id: &str) -> Option<&'static str> {
    Some(match id {
        "hvy_dozer" => "dozer",
        "hvy_cutter" => "excavator",
        "hvy_dump_truck" => "dump_truck",
        "hvy_mixer" => "mixer",
        "hvy_forklift" => "forklift",
        "hvy_flatbed" => "crane_truck",
        "mtl_packer" => "freight_tractor",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The roster parses and holds the doc's counts** -- 153 rows over the
    /// eighteen classes at 20/20/20/20/5x10/5/5/8, plus two trailers, every id
    /// distinct and labelled, under `MAX_VEHICLE_DEFS`.
    #[test]
    fn the_roster_parses_and_holds_the_docs_counts() {
        let defs = roster();
        let mut doc_rows = 0usize;
        for class in RosterClass::ALL {
            let n = rows_of(class).len();
            if class == RosterClass::Trailer {
                assert_eq!(n, 2, "the articulated rig tows two trailers");
                continue;
            }
            assert_eq!(n, class.doc_count(), "{} has {n} rows", class.name());
            doc_rows += n;
        }
        assert_eq!(doc_rows, 153);
        assert_eq!(defs.0.len(), 155);
        assert!(defs.0.len() <= crate::vehicle::MAX_VEHICLE_DEFS);
        for id in defs.0.keys() {
            assert!(roster_label(id).is_some(), "`{id}` has no label");
        }
        assert_eq!(
            roster_label("vapid_dominator_gt"),
            Some("Vapid Dominator GT")
        );
    }

    /// **The kerb draw is deterministic and weighted** -- the same unit gives
    /// the same row, and a class with no kerb weight is never drawn.
    #[test]
    fn a_draw_is_a_pure_function_of_its_unit() {
        let fits = |d: &VehicleDef| d.body.wheeled();
        for k in 0..64 {
            let u = k as f64 / 64.0;
            let a = pick_row(u, RosterClass::parked_weight, fits);
            let b = pick_row(u, RosterClass::parked_weight, fits);
            assert_eq!(a.map(|r| r.0), b.map(|r| r.0));
            let (_, d) = a.expect("a row");
            assert!(d.roster_class.is_some_and(|c| c.parked_weight() > 0.0));
        }
    }
}
