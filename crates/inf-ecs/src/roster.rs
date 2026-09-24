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

/// One class in a weighted draw: the class, its weight, and its rows.
type ClassRows = (RosterClass, f64, Vec<(&'static str, &'static VehicleDef)>);

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
    let classes: Vec<ClassRows> = RosterClass::ALL
        .into_iter()
        .filter_map(|c| {
            let w = weight(c);
            if w.is_nan() || w <= 0.0 {
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
    if total.is_nan() || total <= 0.0 {
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

/// **A defined vehicle's guid** -- a pure function of its row id and where it
/// was put, `item::authored_pickup_guid`'s rule, so two hosts running one graph
/// put the same entity in the same place and a spawn keyed on a counter cannot
/// depend on how many times the graph ran.
pub fn defined_vehicle_guid(id: &str, at: glam::DVec3) -> uuid::Uuid {
    let mut x: u128 = 0x5645_4833_4644_4546_494e_4544_5645_4843;
    for b in id.as_bytes() {
        x = x.rotate_left(9) ^ (*b as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    for v in [at.x, at.y, at.z] {
        x = x.rotate_left(21) ^ (v.to_bits() as u128).wrapping_mul(0xff51_afd7_ed55_8ccd);
    }
    uuid::Builder::from_random_bytes(x.to_be_bytes()).into_uuid()
}

/// **Spawn a catalogue row into a world** -- the `vehicle.spawn` node's door
/// (wave VEH3f). The row is looked up in the world's own catalogue first (what
/// `vehicle.define` merged) and then in the committed roster, so a level may
/// spawn a lore car by id without defining anything. `at` is the chassis origin;
/// `None` for an id neither knows. The car is a whole rig (wheels, parts,
/// class, an engine voice) and is the physics bridge's on its next sync.
pub fn spawn_defined(
    world: &mut crate::world::EcsWorld,
    id: &str,
    at: glam::DVec3,
    yaw_deg: f64,
) -> Option<uuid::Uuid> {
    let def = vehicle_defs(world)
        .and_then(|d| d.get(id).copied())
        .or_else(|| roster().get(id).copied())?;
    let guid = defined_vehicle_guid(id, at);
    let name = roster_label(id).unwrap_or(id).to_string();
    crate::vehicle::spawn_rig(
        world,
        guid,
        &def,
        &crate::vehicle::RigSpawn {
            name,
            at,
            yaw_deg,
            paint: crate::traffic::car_paint(guid),
            clip: None,
            engine_voice: true,
            livery: None,
        },
    );
    Some(guid)
}

/// **An imported vehicle body** (wave VEH3f) -- one machine of the
/// `ConstructionVehiclesPack1` the UE bridge carries, by the key its files and
/// its GUIDs are derived from.
///
/// # THE ART RULE: one committed identity, two possible payloads
///
/// A row that names `art = "excavator"` draws ONE body mesh
/// ([`art_body_guid`]) on an `art_body` part and one wheel mesh per rig wheel
/// ([`art_wheel_guid`]) on its tyres -- never the family's panels. What sits
/// at those GUIDs depends on the machine: the engine repository commits a
/// FALLBACK (the family's own panels baked into one mesh, and a tyre -- ours,
/// generated, `samples/vehicle-art/`), and `inf-import --vehicles` overwrites
/// them in a LOCAL project with the machine's real art, split off the fused
/// Unreal mesh by material slot and connected piece. The starter character's
/// arrangement (a committed body at the MetaHuman's GUIDs), one content kind
/// over -- so CI, which never has the art, draws the fallback, and nothing from
/// Unreal is ever committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArtKey {
    /// `SM_ExcavatorTracks`.
    Excavator,
    /// `SM_TrackLoader` -- the pack has no bulldozer; the track loader is the
    /// nearest machine.
    Dozer,
    /// `SM_AmericanDumpTruck`.
    DumpTruck,
    /// `SM_AmericanMixerTruck`.
    Mixer,
    /// `SM_Forklift`.
    Forklift,
    /// `SM_MobileCrane`.
    CraneTruck,
    /// `SM_AmericanTruck` -- a tractor unit.
    FreightTractor,
}

impl ArtKey {
    /// Every machine, in key order.
    pub const ALL: [ArtKey; 7] = [
        ArtKey::Excavator,
        ArtKey::Dozer,
        ArtKey::DumpTruck,
        ArtKey::Mixer,
        ArtKey::Forklift,
        ArtKey::CraneTruck,
        ArtKey::FreightTractor,
    ];

    /// The stable key a row's `art = "..."` spells and the files are named by.
    pub fn name(self) -> &'static str {
        match self {
            ArtKey::Excavator => "excavator",
            ArtKey::Dozer => "dozer",
            ArtKey::DumpTruck => "dump_truck",
            ArtKey::Mixer => "mixer",
            ArtKey::Forklift => "forklift",
            ArtKey::CraneTruck => "crane_truck",
            ArtKey::FreightTractor => "freight_tractor",
        }
    }

    /// The key a name means, or `None`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// The Unreal static mesh this key is split from.
    pub fn source_mesh(self) -> &'static str {
        match self {
            ArtKey::Excavator => "SM_ExcavatorTracks",
            ArtKey::Dozer => "SM_TrackLoader",
            ArtKey::DumpTruck => "SM_AmericanDumpTruck",
            ArtKey::Mixer => "SM_AmericanMixerTruck",
            ArtKey::Forklift => "SM_Forklift",
            ArtKey::CraneTruck => "SM_MobileCrane",
            ArtKey::FreightTractor => "SM_AmericanTruck",
        }
    }

    /// Whether the machine runs on tracks -- its "wheels" are track frames,
    /// which stay in the body mesh and do not spin.
    pub fn tracked(self) -> bool {
        matches!(self, ArtKey::Excavator | ArtKey::Dozer)
    }
}

/// The salt of the art guids -- `"VEH3FARTBODYMESH"` in ASCII.
const ART_SALT: u128 = 0x5645_4833_4641_5254_424f_4459_4d45_5348;

fn art_guid(key: ArtKey, part: &str) -> uuid::Uuid {
    let mut x = ART_SALT;
    for b in key
        .name()
        .as_bytes()
        .iter()
        .chain(b"/")
        .chain(part.as_bytes())
    {
        x = x.rotate_left(7) ^ (*b as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    x = x.rotate_left(33) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    uuid::Builder::from_random_bytes(x.to_be_bytes()).into_uuid()
}

/// **The body mesh a machine's rows draw**, a pure function of its key.
pub fn art_body_guid(key: ArtKey) -> uuid::Uuid {
    art_guid(key, "body")
}

/// **The mesh rig wheel `i` of a machine draws** (front left, front right,
/// rear left, rear right -- `VehicleDef::wheel_mounts`' order).
pub fn art_wheel_guid(key: ArtKey, i: usize) -> uuid::Uuid {
    art_guid(key, &format!("wheel{i}"))
}

/// **Which construction-pack body a roster row draws** (wave VEH3f), or `None`
/// for every row that has no art anywhere (all of them but seven).
pub fn art_of(id: &str) -> Option<ArtKey> {
    roster().get(id).and_then(|d| d.art)
}

// ── the instruments' questions (wave VEH3f) ─────────────────────────────────

/// **The HUD's roster row** for the car a player sits in: its class, its lore
/// name, what its body is drawn with, its hitch angle when it tows and its yaw
/// rate when it skids -- or empty for a car that is neither a roster row nor
/// wears an authored body (the row would say nothing a driver needs).
pub fn roster_readout(
    world: &crate::world::EcsWorld,
    chassis: uuid::Uuid,
    track_yaw_rad_s: Option<f64>,
) -> String {
    let row = row_of(world, chassis);
    let body = body_kind(world, chassis);
    if row.is_none() && body == "primitive" {
        return String::new();
    }
    let def = row.and_then(|id| roster().get(id));
    let mut s = format!(
        "ROSTER {} {} [{}]",
        def.and_then(|d| d.roster_class)
            .map(|c| c.name())
            .unwrap_or("-"),
        row.and_then(roster_label).unwrap_or("-"),
        body
    );
    if let Some(a) = hitch_angle_deg(world, chassis) {
        s.push_str(&format!("  HITCH {a:+.1} deg"));
    }
    if let Some(w) = track_yaw_rad_s {
        s.push_str(&format!("  TRACKS {w:+.2} rad/s"));
    }
    s
}

/// **Which roster row a chassis in the world is**, read off the WORLD: a
/// traffic car by its own record's guid (`traffic::catalogue_row_id`, the draw
/// that built it), anything else by the lore label its entity is named (every
/// roster spawn names its chassis so). `None` for a vehicle that is not a
/// roster row -- the eleven island rows that predate it, a sample's car.
pub fn row_of(world: &crate::world::EcsWorld, chassis: uuid::Uuid) -> Option<&'static str> {
    if crate::traffic::traffic_of(world).is_some_and(|t| t.records.contains_key(&chassis)) {
        return crate::traffic::catalogue_row_id(chassis);
    }
    let name = world.name_of(world.entity_of(chassis)?)?;
    roster()
        .0
        .keys()
        .find(|id| roster_label(id) == Some(name))
        .map(String::as_str)
}

/// **What a chassis is drawn with** (wave VEH3f): `"imported"` when it hangs a
/// machine's art body (`art_body` -- the pack's art where this project has it,
/// the committed fallback where not), `"dcc"` when any of its panels hangs a
/// DCC hero mesh, `"primitive"` otherwise. Read off the chassis's own children,
/// never off a table.
pub fn body_kind(world: &crate::world::EcsWorld, chassis: uuid::Uuid) -> &'static str {
    let Some(e) = world.entity_of(chassis) else {
        return "-";
    };
    let mut kind = "primitive";
    for child in world.children_of(e) {
        let Some(m) = world.world().get::<crate::components::MeshRef>(child) else {
            continue;
        };
        if m.asset.is_none() {
            continue;
        }
        if world.name_of(child) == Some(crate::vehicle::ART_BODY_PART) {
            return "imported";
        }
        if world.name_of(child) != Some("Tyre") {
            kind = "dcc";
        }
    }
    kind
}

/// **The hitch angle** of an articulated rig, degrees: the tractor's heading
/// less its trailer's, for the chassis `tractor` (or `None` when nothing is
/// hitched to it). Read off the two chassis' own transforms and the trailer's
/// `Joint3D` -- `O(entities)`, for an instrument that ticks a few times a
/// second, never the step.
pub fn hitch_angle_deg(world: &crate::world::EcsWorld, tractor: uuid::Uuid) -> Option<f64> {
    let heading = |e: bevy_ecs::entity::Entity| -> Option<f64> {
        let t = world.world().get::<crate::components::Transform>(e)?;
        let f = t.quat() * glam::DVec3::Z;
        Some(inf_math::patan2_64(f.x, f.z).to_degrees())
    };
    let te = world.entity_of(tractor)?;
    for e in world.world().iter_entities() {
        let Some(j) = e.get::<crate::components::Joint3D>() else {
            continue;
        };
        if j.other.get() != Some(tractor)
            || j.kind != crate::components::JointKind3D::Spherical
            || e.get::<crate::components::VehicleClass>().is_none()
        {
            continue;
        }
        let a = heading(te)? - heading(e.id())?;
        // Wrapped into (-180, 180].
        let w = a - 360.0 * ((a + 180.0) / 360.0).floor();
        return Some(if w == -180.0 { 180.0 } else { w });
    }
    None
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
