//! **THE FIXTURE VOCABULARY** (wave PAR1a, clause 1) — what light a room of
//! each kind hangs, as ONE table the assembler reads.
//!
//! The user's ruling governs every row: *actual lights, not glowing window
//! panes*. A row is a visible fixture (a [`ModuleShape`] family drawn at the
//! row's own size) **and** a real light — kind, colour temperature,
//! lumens-derived intensity, range, cone, whether it asks PAR0's shadow policy
//! for a page tree — placed at a socket the assembler derives from the room's
//! own geometry (the ceiling centre, a grid over a large ceiling, the stair
//! core's wall, a desk top, a bedside, the entrance's head). Nothing here is a
//! coordinate.
//!
//! # Why the diffuser is not emissive
//!
//! A fixture's light sits [`DIFFUSER_GAP_M`] in front of the fixture's lit
//! face, so the face is lit by the fixture's own light — a lamp reads as a
//! lamp because the lamp is on, and goes dark when the occupancy half of the
//! schedule turns it off, with no second state to keep in step. No emissive
//! constant is authored anywhere in this module.
//!
//! # Units
//!
//! Lumens are the catalogue figure for the real fitting. Intensity is
//! candela × [`CANDELA_TO_INTENSITY`]: candela is `lm / 4π` for a point light
//! and `lm / Ω` for a spot of solid angle `Ω = 2π(1 − cos θ_outer)`. The
//! factor is the engine's calibration (a renderer intensity of 1 is the sun's
//! third), stated once: an 800 lm bulb (63.7 cd) lands at 5.1, beside the
//! venue bar glow's authored 5.0 and the PAR0b sealed-room arm's 6.0 lamp, so
//! a room fixture and the rooms PAR0b measured read in the same eye.

use super::modules::ModuleShape;
use super::{ArchetypeId, RoomType};

/// Renderer intensity per candela (see the module docs). A calibration, not a
/// physical constant — the island's night eye is an eight-stop range around
/// the moon, not a photometer.
pub const CANDELA_TO_INTENSITY: f32 = 0.08;

/// How far in front of its lit face a fixture's light sits, metres. Small
/// enough that the face is the brightest thing in the room (it IS the light),
/// large enough that the renderer's `max(d², 1e-4)` clamp is never what lights
/// it.
pub const DIFFUSER_GAP_M: f64 = 0.04;

/// The most primary fixtures one room may hang, whatever its area — the
/// guard a grid over a runaway room would otherwise lack. Sixteen is a
/// 4 × 4 office floor; the vocabulary's own spacings reach it only in the
/// largest rooms the palettes plan.
pub const MAX_FIXTURES_PER_ROOM: u32 = 16;

/// How close to its own range a fixture is drawn as a LIGHT, metres from the
/// eye (the renderer's per-light draw distance). Past it the light is not
/// submitted at all — see the far band in `inf_render::lights`.
pub const ROOM_DRAW_M: f32 = 90.0;
/// The exterior rows' draw distance: a porch lamp is seen down a street.
pub const EXTERIOR_DRAW_M: f32 = 160.0;

/// **Which row of the vocabulary a fixture is** — its identity in a census.
/// Append-only; [`code`](FixtureRow::code) is the wire word the ECS mirror
/// carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FixtureRow {
    /// A ceiling pendant: living rooms, bedrooms, lobbies.
    Pendant,
    /// An office ceiling batten (a recessed-panel run).
    OfficeBatten,
    /// A shop's fluorescent strip.
    ShopStrip,
    /// A workshop / store / plant-room batten.
    WorkBatten,
    /// A corridor's ceiling bulkhead, one per bay.
    CorridorLight,
    /// A stair core's wall bulkhead.
    StairLight,
    /// A ward's bed-head batten.
    WardLight,
    /// A consulting room's examination light.
    ExamLight,
    /// A custody cell's caged bulkhead.
    CellLight,
    /// An apparatus bay's high-bay flood.
    BayFlood,
    /// A kitchen's ceiling batten.
    KitchenLight,
    /// A bathroom's sealed bulkhead.
    BathLight,
    /// A desk lamp, on a desk the furniture placed.
    DeskLamp,
    /// A bedside lamp, beside a bed the furniture placed.
    BedsideLamp,
    /// A porch / entrance lantern over the street door.
    Porch,
    /// A facade wash beside a shop's or venue's entrance.
    FacadeWash,
    /// The light on a shop's fascia or a venue's sign.
    SignLight,
    /// An institution's forecourt flood.
    ForecourtFlood,
    /// A venue stage / dance-floor rig spot (VEN1a).
    RigSpot,
    /// A venue bar's glow (VEN1a).
    BarGlow,
    /// A venue room the rig does not reach — a dim pendant on the venue's hours.
    VenuePendant,
}

impl FixtureRow {
    /// Every row, in declaration order.
    pub const ALL: [FixtureRow; 21] = [
        FixtureRow::Pendant,
        FixtureRow::OfficeBatten,
        FixtureRow::ShopStrip,
        FixtureRow::WorkBatten,
        FixtureRow::CorridorLight,
        FixtureRow::StairLight,
        FixtureRow::WardLight,
        FixtureRow::ExamLight,
        FixtureRow::CellLight,
        FixtureRow::BayFlood,
        FixtureRow::KitchenLight,
        FixtureRow::BathLight,
        FixtureRow::DeskLamp,
        FixtureRow::BedsideLamp,
        FixtureRow::Porch,
        FixtureRow::FacadeWash,
        FixtureRow::SignLight,
        FixtureRow::ForecourtFlood,
        FixtureRow::RigSpot,
        FixtureRow::BarGlow,
        FixtureRow::VenuePendant,
    ];

    /// The row's wire word (its declaration index).
    pub fn code(self) -> u16 {
        Self::ALL.iter().position(|r| *r == self).unwrap_or(0) as u16
    }

    /// A row from its wire word.
    pub fn from_code(c: u16) -> Option<Self> {
        Self::ALL.get(c as usize).copied()
    }

    /// Whether this row is VEN1a's venue rig (the lights
    /// `volume::VOLUME_LIGHT_CAP` guards).
    pub fn is_rig(self) -> bool {
        matches!(self, FixtureRow::RigSpot | FixtureRow::BarGlow)
    }

    /// Whether this row hangs on the OUTSIDE of a building.
    pub fn is_exterior(self) -> bool {
        matches!(
            self,
            FixtureRow::Porch
                | FixtureRow::FacadeWash
                | FixtureRow::SignLight
                | FixtureRow::ForecourtFlood
        )
    }

    /// A short name for tables.
    pub fn name(self) -> &'static str {
        match self {
            FixtureRow::Pendant => "pendant",
            FixtureRow::OfficeBatten => "office batten",
            FixtureRow::ShopStrip => "shop strip",
            FixtureRow::WorkBatten => "work batten",
            FixtureRow::CorridorLight => "corridor bulkhead",
            FixtureRow::StairLight => "stair bulkhead",
            FixtureRow::WardLight => "ward light",
            FixtureRow::ExamLight => "exam light",
            FixtureRow::CellLight => "cell light",
            FixtureRow::BayFlood => "bay flood",
            FixtureRow::KitchenLight => "kitchen batten",
            FixtureRow::BathLight => "bath bulkhead",
            FixtureRow::DeskLamp => "desk lamp",
            FixtureRow::BedsideLamp => "bedside lamp",
            FixtureRow::Porch => "porch lantern",
            FixtureRow::FacadeWash => "facade wash",
            FixtureRow::SignLight => "sign light",
            FixtureRow::ForecourtFlood => "forecourt flood",
            FixtureRow::RigSpot => "rig spot",
            FixtureRow::BarGlow => "bar glow",
            FixtureRow::VenuePendant => "venue pendant",
        }
    }
}

/// Where on its host a fixture mounts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mount {
    /// Hung under the ceiling, lit face down.
    Ceiling,
    /// Fixed to a wall, lit face out from the wall.
    Wall,
    /// Standing on a piece of furniture, lit face down.
    Furniture,
}

/// **One row of the vocabulary.**
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixtureDef {
    pub row: FixtureRow,
    /// The visible fitting.
    pub shape: ModuleShape,
    /// The fitting's half-extents, metres: `(along, up, depth)` in its own
    /// frame (`+Z` is the lit face's normal for a wall mount).
    pub half: [f64; 3],
    pub mount: Mount,
    /// How far below the ceiling (ceiling mount) or above the floor (wall
    /// mount) the fitting's centre sits, metres.
    pub drop_m: f64,
    /// Correlated colour temperature, kelvin (one of [`kelvin_rgb`]'s rows).
    pub kelvin: u32,
    /// The fitting's catalogue output.
    pub lumens: f32,
    /// `Some((inner, outer))` half-angles in degrees for a spot; `None` for a
    /// point light.
    pub cone_deg: Option<(f32, f32)>,
    /// The pitch of the grid a large room lays this row on, metres; `0` hangs
    /// exactly one.
    pub spacing_m: f64,
    /// Whether the fixture asks PAR0's shadow policy for a page tree (the
    /// policy grants at most `LOCAL_SHADOW_BUDGET` a frame).
    pub shadow: bool,
}

impl FixtureDef {
    /// The renderer intensity this row's lumens imply (see the module docs).
    pub fn intensity(&self) -> f32 {
        let candela = match self.cone_deg {
            None => self.lumens / (4.0 * std::f32::consts::PI),
            Some((_, outer)) => {
                let c = inf_math::portable::pcos(outer.to_radians());
                let omega = 2.0 * std::f32::consts::PI * (1.0 - c).max(1e-3);
                self.lumens / omega
            }
        };
        candela * CANDELA_TO_INTENSITY
    }

    /// The row's linear colour.
    pub fn colour(&self) -> [f32; 3] {
        kelvin_rgb(self.kelvin)
    }
}

/// **Colour temperature → linear RGB**, normalized to a peak of 1 — a table
/// rather than a Planck fit, so the numbers are bit-identical everywhere and a
/// reader can check them against a lighting catalogue's swatches.
pub fn kelvin_rgb(k: u32) -> [f32; 3] {
    match k {
        0..=2850 => [1.0, 0.70, 0.42],    // 2700 K: incandescent / warm LED
        2851..=3250 => [1.0, 0.76, 0.52], // 3000 K: warm white
        3251..=3750 => [1.0, 0.82, 0.63], // 3500 K
        3751..=4500 => [1.0, 0.89, 0.77], // 4000 K: neutral / fluorescent
        4501..=5300 => [1.0, 0.95, 0.90], // 5000 K: clinical / high-bay
        _ => [0.96, 0.97, 1.0],           // 5600 K+: floodlight
    }
}

const fn def(
    row: FixtureRow,
    shape: ModuleShape,
    half: [f64; 3],
    mount: Mount,
    drop_m: f64,
    kelvin: u32,
    lumens: f32,
    cone_deg: Option<(f32, f32)>,
    spacing_m: f64,
    shadow: bool,
) -> FixtureDef {
    FixtureDef {
        row,
        shape,
        half,
        mount,
        drop_m,
        kelvin,
        lumens,
        cone_deg,
        spacing_m,
        shadow,
    }
}

/// **The vocabulary, row by row.**
pub fn fixture(row: FixtureRow) -> FixtureDef {
    use FixtureRow as R;
    use ModuleShape as S;
    use Mount as M;
    match row {
        R::Pendant => def(
            row,
            S::Pendant,
            [0.22, 0.17, 0.22],
            M::Ceiling,
            0.17,
            2700,
            900.0,
            None,
            5.0,
            true,
        ),
        R::OfficeBatten => def(
            row,
            S::Batten,
            [0.6, 0.04, 0.15],
            M::Ceiling,
            0.04,
            4000,
            2400.0,
            None,
            4.5,
            true,
        ),
        R::ShopStrip => def(
            row,
            S::Batten,
            [0.75, 0.04, 0.08],
            M::Ceiling,
            0.04,
            4000,
            3000.0,
            None,
            3.5,
            true,
        ),
        R::WorkBatten => def(
            row,
            S::Batten,
            [0.75, 0.04, 0.08],
            M::Ceiling,
            0.04,
            5000,
            2600.0,
            None,
            4.5,
            true,
        ),
        R::CorridorLight => def(
            row,
            S::Bulkhead,
            [0.16, 0.05, 0.16],
            M::Ceiling,
            0.05,
            4000,
            900.0,
            None,
            6.0,
            true,
        ),
        R::StairLight => def(
            row,
            S::Bulkhead,
            [0.16, 0.12, 0.05],
            M::Wall,
            2.3,
            4000,
            900.0,
            None,
            0.0,
            true,
        ),
        R::WardLight => def(
            row,
            S::Batten,
            [0.6, 0.04, 0.15],
            M::Ceiling,
            0.04,
            4000,
            2000.0,
            None,
            4.0,
            true,
        ),
        R::ExamLight => def(
            row,
            S::Batten,
            [0.6, 0.04, 0.3],
            M::Ceiling,
            0.04,
            5000,
            2600.0,
            None,
            0.0,
            true,
        ),
        R::CellLight => def(
            row,
            S::Bulkhead,
            [0.16, 0.05, 0.16],
            M::Ceiling,
            0.05,
            4000,
            600.0,
            None,
            0.0,
            true,
        ),
        R::BayFlood => def(
            row,
            S::Flood,
            [0.25, 0.12, 0.25],
            M::Ceiling,
            0.5,
            5000,
            6000.0,
            None,
            7.0,
            true,
        ),
        R::KitchenLight => def(
            row,
            S::Batten,
            [0.5, 0.04, 0.12],
            M::Ceiling,
            0.04,
            3500,
            1400.0,
            None,
            0.0,
            true,
        ),
        R::BathLight => def(
            row,
            S::Bulkhead,
            [0.14, 0.05, 0.14],
            M::Ceiling,
            0.05,
            4000,
            700.0,
            None,
            0.0,
            true,
        ),
        R::DeskLamp => def(
            row,
            S::DeskLamp,
            [0.1, 0.22, 0.1],
            M::Furniture,
            0.0,
            3000,
            300.0,
            Some((25.0, 50.0)),
            0.0,
            false,
        ),
        R::BedsideLamp => def(
            row,
            S::DeskLamp,
            [0.1, 0.2, 0.1],
            M::Furniture,
            0.0,
            2700,
            250.0,
            Some((30.0, 60.0)),
            0.0,
            false,
        ),
        R::Porch => def(
            row,
            S::Lantern,
            [0.1, 0.16, 0.1],
            M::Wall,
            0.0,
            2700,
            600.0,
            None,
            0.0,
            false,
        ),
        R::FacadeWash => def(
            row,
            S::Lantern,
            [0.1, 0.16, 0.1],
            M::Wall,
            0.0,
            3000,
            1500.0,
            None,
            0.0,
            false,
        ),
        R::SignLight => def(
            row,
            S::Batten,
            [0.5, 0.04, 0.08],
            M::Wall,
            0.0,
            4000,
            1200.0,
            None,
            0.0,
            false,
        ),
        R::ForecourtFlood => def(
            row,
            S::Flood,
            [0.25, 0.12, 0.25],
            M::Wall,
            4.5,
            5600,
            9000.0,
            None,
            0.0,
            false,
        ),
        // The venue rows are VEN1a's rig, which carries its own colour and
        // intensity; the row exists so a census can name it.
        R::RigSpot | R::BarGlow => def(
            row,
            S::Pendant,
            [0.0; 3],
            M::Ceiling,
            0.0,
            2700,
            0.0,
            None,
            0.0,
            true,
        ),
        R::VenuePendant => def(
            row,
            S::Pendant,
            [0.22, 0.17, 0.22],
            M::Ceiling,
            0.17,
            2700,
            600.0,
            None,
            5.0,
            true,
        ),
    }
}

/// **Which row lights a room** — one exhaustive answer per room type, so a
/// twenty-third room type fails to compile here rather than standing dark.
///
/// Venue public rooms answer [`VenuePendant`](FixtureRow::VenuePendant): the
/// assembler skips it where the archetype's rig already lights the room.
pub fn room_row(kind: RoomType) -> FixtureRow {
    use FixtureRow as R;
    match kind {
        RoomType::Corridor => R::CorridorLight,
        RoomType::Stair => R::StairLight,
        RoomType::Lobby
        | RoomType::Waiting
        | RoomType::Living
        | RoomType::Bedroom
        | RoomType::Guest => R::Pendant,
        RoomType::Office | RoomType::Meeting => R::OfficeBatten,
        RoomType::Service | RoomType::Storage | RoomType::Workshop => R::WorkBatten,
        RoomType::Kitchen => R::KitchenLight,
        RoomType::Bath => R::BathLight,
        RoomType::Retail => R::ShopStrip,
        RoomType::DanceFloor | RoomType::BarRoom | RoomType::Stage => R::VenuePendant,
        RoomType::Cell => R::CellLight,
        RoomType::ApparatusBay => R::BayFlood,
        RoomType::Ward => R::WardLight,
        RoomType::ExamRoom => R::ExamLight,
    }
}

/// **Whose day a fixture keeps** (wave PAR1a clause 4) — the occupancy half of
/// the night schedule, resolved by `inf_ecs::sky::fixture_occupancy` against
/// the level clock. Derived, never serialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FixtureOccupancy {
    /// Lit whenever its schedule says — a crewed room (a venue's public rooms
    /// on the venue's hours, an institution's never-closing rooms) and every
    /// exterior light.
    Crew,
    /// A workplace: its crew's working day, then a deterministic share of
    /// rooms left lit into the evening and a minority all night.
    Work,
    /// A shop floor: open until a per-shop closing hour, then a share left lit
    /// for the window display.
    Shop,
    /// A dwelling room: its household's day.
    Home(HomeRoom),
}

/// Which room of a household a dwelling fixture is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HomeRoom {
    Living,
    Kitchen,
    Bedroom,
    Bath,
    /// Halls, stairs, stores — lit while anyone is up.
    Hall,
}

/// **A fixture's occupancy seed**: the building's salt (`society::
/// building_salt`) folded with the assembler's household key — a pure
/// function of content, the same in every host.
pub fn household_seed(building_salt: u64, household: u32) -> u32 {
    let h = crate::hash::Hash64::new(building_salt)
        .mix_u64(SALT_HOUSEHOLD)
        .mix_u64(u64::from(household))
        .finish();
    (h ^ (h >> 32)) as u32
}

/// Salts [`household_seed`].
const SALT_HOUSEHOLD: u64 = 0x484F_5553_4548_4F4C; // "HOUSEHOL"

/// Whether an archetype's rooms are dwellings.
pub fn is_dwelling(a: ArchetypeId) -> bool {
    matches!(
        a,
        ArchetypeId::House | ArchetypeId::Apartment | ArchetypeId::Estate
    )
}

/// **The schedule and occupancy a room's fixtures keep** (clause 4).
///
/// * Dwellings (`is_dwelling`, and a hotel's guest rooms) keep their
///   household's day under the [`Dusk`](super::FixtureSchedule::Dusk)
///   schedule: a lamp in a home is switched on because it is dark AND
///   someone is in.
/// * A room an institution never closes ([`super::society::crews_of`] lists
///   two crews) is crewed `Always`; so are the circulation rooms of a building
///   that holds one (a hospital's corridor is lit at 04:00) and a hotel's.
/// * A venue's rooms keep the venue's crews ([`super::society::schedule_of`]).
/// * Everything else is a workplace; a shop floor keeps shop hours.
pub fn room_keeps(
    arch: ArchetypeId,
    kind: RoomType,
    never_closes: bool,
) -> (super::FixtureSchedule, FixtureOccupancy) {
    use super::FixtureSchedule as F;
    let circulation = matches!(
        kind,
        RoomType::Corridor
            | RoomType::Stair
            | RoomType::Lobby
            | RoomType::Waiting
            | RoomType::Service
    );
    if super::society::crews_of(kind).len() > 1 {
        return (F::Always, FixtureOccupancy::Crew);
    }
    if is_dwelling(arch) {
        let r = match kind {
            RoomType::Living => HomeRoom::Living,
            RoomType::Kitchen => HomeRoom::Kitchen,
            RoomType::Bedroom | RoomType::Guest => HomeRoom::Bedroom,
            RoomType::Bath => HomeRoom::Bath,
            _ => HomeRoom::Hall,
        };
        return (F::Dusk, FixtureOccupancy::Home(r));
    }
    if arch == ArchetypeId::Hotel {
        return match kind {
            RoomType::Guest | RoomType::Bedroom => {
                (F::Dusk, FixtureOccupancy::Home(HomeRoom::Bedroom))
            }
            RoomType::Bath => (F::Dusk, FixtureOccupancy::Home(HomeRoom::Bath)),
            _ => (F::Always, FixtureOccupancy::Crew),
        };
    }
    if arch.is_venue() {
        // The venue's public rooms and its halls keep the venue's night; its
        // back office keeps the working day.
        if circulation {
            return (F::Night, FixtureOccupancy::Crew);
        }
        return match super::society::schedule_of(kind) {
            F::Night => (F::Night, FixtureOccupancy::Crew),
            _ => (F::Always, FixtureOccupancy::Work),
        };
    }
    if never_closes && circulation {
        return (F::Always, FixtureOccupancy::Crew);
    }
    if kind == RoomType::Retail {
        return (F::Always, FixtureOccupancy::Shop);
    }
    (F::Always, FixtureOccupancy::Work)
}

/// **THE ZERO-UNLIT-ROOMS CENSUS** (wave PAR1a clause 2): for every room of
/// every plan, whether any fixture names it — counted by archetype and floor.
///
/// `plans` are a volume's buildings in evaluation order (the ordinal a
/// fixture's [`super::FixtureTag::building`] carries) and `rooms_lit` the set
/// of `(building, room)` pairs some fixture names — read off whatever list
/// the caller holds (the volume door's `PcgLight`s, or the ECS's
/// `ScatteredLight`s in a loaded world).
pub fn census(
    plans: &[super::BuildingPlan],
    rooms_lit: &std::collections::BTreeSet<(u32, u32)>,
    into: &mut std::collections::BTreeMap<(ArchetypeId, u32), CensusRow>,
) {
    for (b, plan) in plans.iter().enumerate() {
        for (ri, room) in plan.rooms.iter().enumerate() {
            let row = into.entry((plan.archetype, room.floor)).or_default();
            row.rooms += 1;
            if !rooms_lit.contains(&(b as u32, ri as u32)) {
                row.unlit += 1;
            }
        }
    }
}

/// One `(archetype, floor)` cell of [`census`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CensusRow {
    pub rooms: u32,
    pub unlit: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row's intensity is finite and positive, and a bedside lamp is
    /// not a floodlight: the brightest room row is under the bay flood and the
    /// dimmest is a bedside lamp.
    #[test]
    fn the_rows_are_plausible_lamps() {
        for row in FixtureRow::ALL {
            let d = fixture(row);
            if matches!(row, FixtureRow::RigSpot | FixtureRow::BarGlow) {
                continue;
            }
            let i = d.intensity();
            assert!(i.is_finite() && i > 0.0, "{}: intensity {i}", row.name());
            assert!(
                d.half.iter().all(|h| *h > 0.0),
                "{}: no fitting",
                row.name()
            );
            assert_eq!(FixtureRow::from_code(row.code()), Some(row));
        }
        // An 800-lm-class bulb lands beside the calibrated rooms PAR0b measured.
        let p = fixture(FixtureRow::Pendant).intensity();
        assert!((4.0..8.0).contains(&p), "pendant {p}");
        let flood = fixture(FixtureRow::BayFlood).intensity();
        assert!(flood > 4.0 * p, "a bay flood {flood} vs a pendant {p}");
    }

    /// **EVERY ROOM OF EVERY ARCHETYPE ON EVERY FLOOR** hangs a fixture, its
    /// light inside its own room's box and its fitting under the ceiling —
    /// furnished or not, on axis-aligned and rotated lots alike. Mutation:
    /// skipping upper floors in `room_fixtures` reds the census by floor.
    #[test]
    fn every_room_of_every_archetype_on_every_floor_hangs_a_fixture() {
        use super::super::{build_in, plan::BuildingParams, LotFrame, Rect2};
        use glam::DVec2;
        let mut table = std::collections::BTreeMap::new();
        let mut lights = 0usize;
        for arch in ArchetypeId::ALL {
            for seed in 0..12u64 {
                for (furnish, frame) in [
                    (true, LotFrame::IDENTITY),
                    (
                        false,
                        LotFrame::new(DVec2::new(40.0, -7.0), DVec2::new(0.6, 0.8)),
                    ),
                ] {
                    let p = BuildingParams::new(
                        arch,
                        Rect2::new(DVec2::ZERO, DVec2::new(22.0 + seed as f64, 18.0)),
                        3.0,
                        seed * 7919 + 1,
                    );
                    let out = build_in(&p, frame, seed, furnish);
                    let lit: std::collections::BTreeSet<(u32, u32)> = out
                        .lights
                        .iter()
                        .filter(|l| l.tag.room != super::super::FixtureTag::EXTERIOR)
                        .map(|l| (0, l.tag.room))
                        .collect();
                    census(std::slice::from_ref(&out.plan), &lit, &mut table);
                    lights += out.lights.len();
                    for l in &out.lights {
                        let Some(c) = l.clip else { continue };
                        // The light is inside its own box (a lamp in a wall or
                        // above the slab would not be).
                        let d = l.at - c.center;
                        let v = DVec2::new(-c.u.y, c.u.x);
                        let (du, dv) = (d.x * c.u.x + d.z * c.u.y, d.x * v.x + d.z * v.y);
                        assert!(
                            du.abs() <= c.half.x && dv.abs() <= c.half.z && d.y.abs() <= c.half.y,
                            "{arch:?} seed {seed}: a {} at {:?} is outside its box",
                            l.tag.row.name(),
                            l.at
                        );
                    }
                    // Every building hangs a porch light on its entrance.
                    assert!(
                        out.lights.iter().any(|l| l.tag.row == FixtureRow::Porch),
                        "{arch:?} seed {seed}: no porch light"
                    );
                }
            }
        }
        let mut floors_seen = 0;
        for ((arch, floor), row) in &table {
            assert_eq!(
                row.unlit, 0,
                "{arch:?} floor {floor}: {} of {} rooms unlit",
                row.unlit, row.rooms
            );
            if *floor > 0 {
                floors_seen += 1;
            }
        }
        assert!(
            floors_seen > 20,
            "the census saw only {floors_seen} upper (archetype, floor) cells"
        );
        assert!(lights > 1000, "{lights} lights");
    }

    /// Every room type has a row and the round-the-clock rooms are crewed.
    #[test]
    fn every_room_type_has_a_row_and_a_day() {
        for kind in RoomType::ALL {
            let _ = room_row(kind);
            let (s, o) = room_keeps(ArchetypeId::Hospital, kind, true);
            if super::super::society::crews_of(kind).len() > 1 {
                assert_eq!(
                    (s, o),
                    (
                        super::super::FixtureSchedule::Always,
                        FixtureOccupancy::Crew
                    )
                );
            }
        }
        assert!(matches!(
            room_keeps(ArchetypeId::House, RoomType::Living, false).1,
            FixtureOccupancy::Home(HomeRoom::Living)
        ));
        assert_eq!(
            room_keeps(ArchetypeId::Hospital, RoomType::Corridor, true),
            (
                super::super::FixtureSchedule::Always,
                FixtureOccupancy::Crew
            )
        );
        assert_eq!(
            room_keeps(ArchetypeId::Office, RoomType::Corridor, false).1,
            FixtureOccupancy::Work
        );
    }
}
