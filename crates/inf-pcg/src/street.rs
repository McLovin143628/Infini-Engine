//! **STREET FURNITURE** (wave PAR1b) — what stands on a settlement's
//! pavements: lamp posts, traffic signals, utility poles and the cables they
//! carry, road signs, benches, bins, hydrants and mailboxes.
//!
//! # A function of the streets, nothing else
//!
//! Every piece is derived from the level's own street centrelines (the
//! `inf_ecs::traffic::Street`s both hosts already recover from the blocks), the
//! signalised junctions `inf_ecs::traffic::signal_junctions` picks among them,
//! the blocks' exterior doors and the ground. Nothing is authored and nothing
//! reaches a level's bytes: a piece sits on a **world lattice** along its
//! street (`kerb_slots`' rule, so a street that grows keeps the pieces it had),
//! the optional kinds are thinned by a **hash** of their own lattice slot, and a
//! piece's identity is its quantized foot ([`piece_id`]).
//!
//! # Where on the pavement
//!
//! Two lines per side of a street (wave PAR1b.2). The POSTS — lamps, signal
//! masts, utility poles, signs, hydrants, parking meters — stand at the kerb
//! ([`post_line_m`]: [`KERB_POST_OFFSET_M`] behind the kerb face, as
//! `steal-car/0022` and `0028` show them; behind a parked car's door where the
//! reserve parks at its kerb), and so does a bus shelter's screen (the PAR1b.2
//! audit: a shelter against the frontage stood in a window band). The BROAD
//! pieces — benches, bins, mailboxes — stand against the frontage
//! ([`furniture_line_m`], [`FURNITURE_BACK_M`] inside the footway's back
//! edge), facing the street.
//! The crowd walks a ring `PAVEMENT_M` outside each block, between the two
//! lines, and walks PAST the posts: `inf_ecs::furniture::clear_of_furniture`
//! keeps every agent its radius clear of every piece's footprint (PAR1b
//! placed the posts at the back line because the ring had no such term).
//!
//! # What it never does
//!
//! No piece stands in a carriageway (the line is behind the kerb by
//! construction), in a crossing (every lattice slot within the crossing's far
//! edge of a junction plus [`JUNCTION_CLEAR_M`] is refused), in a doorway (a
//! slot within [`DOOR_CLEAR_M`] of an exterior door is refused) or in a
//! parking slot (`kerb_slots` parks in the carriageway). The arms that say so
//! live in `inf_ecs`'s gate, measured over the whole island.

use glam::{DQuat, DVec2, DVec3};

use crate::building::fixtures::{self, FixtureOccupancy, FixtureRow};
use crate::building::modules::{module_mesh_guid, ModuleShape};
use crate::building::{FixtureSchedule, FixtureTag, PcgLight};
use crate::height::HeightProvider;
use crate::scatter::{PcgCollider, PcgInstance, PcgSurface};

/// A kerb stone's width, metres — `inf_gis::KERB_WIDTH_M` by value (this crate
/// names neither `inf-gis` nor `inf-ecs`; the PAR1b gate pins the equality).
pub const KERB_WIDTH_M: f64 = 0.30;
/// A kerb's upstand, metres — `inf_gis::KERB_HEIGHT_M` by value.
pub const KERB_HEIGHT_M: f64 = 0.15;
/// The footway's width, metres — `inf_gis::PAVEMENT_M` by value.
pub const PAVEMENT_M: f64 = 2.0;
/// The road ribbon's lift over its ground, metres — `inf_gis::DEFAULT_ROAD_LIFT_M`.
pub const ROAD_LIFT_M: f64 = 0.02;
/// The footway's cross-fall back from the kerb — `inf_gis::PAVEMENT_FALL`.
pub const PAVEMENT_FALL: f64 = 0.02;
/// One lane, metres — `inf_ecs::traffic::DEFAULT_LANE_WIDTH_M`.
pub const LANE_WIDTH_M: f64 = 3.5;

/// **How far inside the footway's back edge the furniture line is**, metres —
/// where the BROAD pieces stand (benches, bins, mailboxes, bus shelters):
/// against the frontage, facing the street, out of the walking line.
pub const FURNITURE_BACK_M: f64 = 0.35;

/// **How far behind the kerb FACE the posts stand**, metres (wave PAR1b.2) —
/// read off `steal-car/0022` (the hydrant, the sign post and the signal mast
/// at the near-left corner, each about half a metre in from the kerb face) and
/// `0028` (the pole and the sign on the verge at the kerb). Lamps, signal
/// masts, utility poles, signs, hydrants and parking meters stand on this
/// line; the crowd walks past them (`inf_ecs::furniture::clear_of_furniture`).
pub const KERB_POST_OFFSET_M: f64 = 0.5;

/// `inf_ecs::traffic::KERB_PARK_OFFSET_M` by value — where a kerb slot's car
/// stands from the centreline (the PAR1b.2 gate pins the equality).
pub const KERB_PARK_OFFSET_M: f64 = 5.0;
/// `inf_ecs::traffic::KERB_SLOT_M` by value — the kerb slots' world lattice
/// pitch (the PAR1b.2 gate pins the equality).
pub const KERB_SLOT_M: f64 = 14.0;
/// **How far from the centreline a post must stand to clear a parked car**,
/// metres: the slot's offset, a saloon's half-width (0.92), a door's swing
/// clearance of 0.15 and a post's half-width. On a 16 m town street the kerb
/// face is at 5.25 and a car parked at 5.0 overhangs it, so the post line is
/// this rather than the kerb's half metre.
pub const PARKED_CLEAR_M: f64 = KERB_PARK_OFFSET_M + 0.92 + 0.15 + 0.15;

/// **Lamp spacing**, metres — read off `driving/0006` (the brief's ~30 m: four
/// heads over the left verge between the near pole and the junction).
pub const LAMP_SPACING_M: f64 = 30.0;
/// The luminaire's mounting height, metres (a 9 m column, `steal-car/0028`'s
/// cobra head beside the far signal reads ~1.5 x the 5.6 m signal heads).
pub const LAMP_HEIGHT_M: f64 = 9.0;
/// How far the luminaire hangs out over the carriageway past the kerb face,
/// metres.
pub const LAMP_OVERHANG_M: f64 = 0.5;
/// The lamp shaft's half-width, metres.
pub const LAMP_SHAFT_HALF_M: f64 = 0.09;
/// A lamp's light range, metres (audit PAR1b: 28 -> 16). The first cut's 28 m
/// and 75-degree cone laid ONE pool over the whole street — the measured
/// asphalt midway between two lamps was lit 1.46x less than under one, and the
/// shipped kerb eyes closed from x15.6 / x51.5 / x73.8 to x5.7 / x6.3 / x10.6
/// under it (the street a lamp did not reach went dark). At 16 m, with the
/// 55-degree cone and the throw tilted toward the road, a pool ends before the
/// next lamp's begins — a CHAIN of pools, which is what `steal-car/0035`'s
/// night street is (its pool reads 2.2x the road beside it, 8-bit) — and the
/// clustered pass touches a third of the froxels.
pub const LAMP_RANGE_M: f32 = 16.0;
/// **How far the lamp's beam leans toward the carriageway** from straight
/// down: 20 degrees, as its cosine and sine (literals, not a libm call — the
/// P14 law: derived content two hosts compare). A real cobra head throws its
/// light street-side; leaning the spot reproduces that, puts the pool on the
/// near lane, and leaves the lamp's own column at the back edge of its cone.
pub const LAMP_TILT_COS: f64 = 0.939_692_620_785_908_4;
/// See [`LAMP_TILT_COS`].
pub const LAMP_TILT_SIN: f64 = 0.342_020_143_325_668_7;
/// How far from the eye a street lamp is a LIGHT, metres; past it PAR0's
/// distance door has dropped it and the post itself still draws to
/// `modules::STREET_FURNITURE_LOD_M`.
pub const LAMP_DRAW_M: f32 = 110.0;
/// **The luminaire's emission**, authored, before the Dusk schedule and
/// `inf_render::POWERED_RADIANCE_SCALE` — the brightest venue sign's own 3.0,
/// calibrated against the same eight-stop ceiling: at the most open night eye
/// the island allows (x256) it tonemaps to 0.877, under clipping
/// (`par0b_night_gate::a_closed_venues_emitters_are_off_and_an_open_ones_sit_under_clipping`
/// measured the first cut's 24.0 at 1.013 — clipping). At the shipped kerb
/// cameras' x15 - x73 it reads 0.09 - 0.43: a lit opal head, not a lamp-shaped
/// white hole.
pub const LAMP_HEAD_EMISSIVE: f32 = 3.0;

/// **Utility pole spacing**, metres — `driving/0006`: the near pole (left) and
/// the next one down the road, ~40 m on the verge.
pub const POLE_SPACING_M: f64 = 40.0;
/// The pole's height, metres (a class-4 distribution pole, 11.5 m above ground;
/// `driving/0006`'s near pole is ~1.3 x the 9 m lamp columns behind it).
pub const POLE_HEIGHT_M: f64 = 11.5;
/// The pole's half-width, metres.
pub const POLE_HALF_M: f64 = 0.13;
/// **The cables' sag as a fraction of their span** — `driving/0006` shows the
/// top conductors dipping ~1 m over the ~40 m span: 2.5 %.
pub const CABLE_SAG_FRAC: f64 = 0.025;
/// Straight segments one span's catenary is drawn with.
pub const CABLE_SEGMENTS: usize = 6;
/// The longest span the line carries across a junction, metres.
pub const MAX_SPAN_M: f64 = 2.3 * POLE_SPACING_M;

/// **The signal head's height**, metres — `steal-car/0028`: the two red heads
/// hang ~5.6 m over the far side of the junction.
pub const SIGNAL_HEAD_HEIGHT_M: f64 = 5.6;
/// The signal mast's height, metres (its arm leaves the top).
pub const SIGNAL_MAST_M: f64 = 6.4;
/// The mast's half-width, metres.
pub const SIGNAL_MAST_HALF_M: f64 = 0.12;
/// The signal head light's reach, metres.
pub const SIGNAL_RANGE_M: f32 = 30.0;

/// The crossing's far edge from its junction node, metres —
/// `inf_ecs::traffic::CROSSWALK_FAR_M`.
pub const CROSSWALK_FAR_M: f64 = 9.0;
/// **How far past a junction's crossing (or its block corner, whichever is
/// further) the lattice starts**, metres.
pub const JUNCTION_CLEAR_M: f64 = 2.5;
/// **How near an exterior door a piece may stand**, metres (plan distance from
/// the door's hinge).
pub const DOOR_CLEAR_M: f64 = 3.0;
/// The least distance between two pieces on one side line, metres.
pub const MIN_SEPARATION_M: f64 = 1.6;

/// **One street, as this module needs it** — a centreline along X or along Z,
/// and its reserve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreetLine {
    /// One end, world XZ.
    pub a: DVec2,
    /// The other, world XZ.
    pub b: DVec2,
    /// The reserve, metres (block edge to block edge).
    pub gap_m: f64,
}

impl StreetLine {
    /// Whether it runs along world X.
    pub fn along_x(&self) -> bool {
        self.a.y == self.b.y
    }
    /// `(lo, hi)` along its own axis.
    fn span(&self) -> (f64, f64) {
        if self.along_x() {
            (self.a.x.min(self.b.x), self.a.x.max(self.b.x))
        } else {
            (self.a.y.min(self.b.y), self.a.y.max(self.b.y))
        }
    }
    /// Its constant coordinate.
    fn perp(&self) -> f64 {
        if self.along_x() {
            self.a.y
        } else {
            self.a.x
        }
    }
    /// The world point `along` this line's axis, `lateral` to its +side (the
    /// side `+Z` for a line along X, `+X` for a line along Z).
    fn at(&self, along: f64, lateral: f64) -> DVec2 {
        if self.along_x() {
            DVec2::new(along, self.perp() + lateral)
        } else {
            DVec2::new(self.perp() + lateral, along)
        }
    }
    /// The unit world direction of its own axis.
    fn axis(&self) -> DVec3 {
        if self.along_x() {
            DVec3::X
        } else {
            DVec3::Z
        }
    }
    /// The unit world direction of its +side.
    fn side_dir(&self) -> DVec3 {
        if self.along_x() {
            DVec3::Z
        } else {
            DVec3::X
        }
    }
}

/// **One signalised junction** — `inf_ecs::traffic::SignalJunction`'s plan
/// half, mapped by the host.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignalSite {
    /// The crossing, world XZ.
    pub centre: DVec2,
    /// The reserve of the street along X, metres.
    pub gap_x: f64,
    /// The reserve of the street along Z, metres.
    pub gap_z: f64,
    /// The controller's phase offset, whole seconds.
    pub offset_s: u32,
}

/// **What a piece of furniture is.**
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PieceKind {
    /// A lamp post: shaft, arm, luminaire and a real light.
    LampPost,
    /// A signal mast: shaft, mast arm, head, three lenses (drawn by the
    /// projector from the head's light) and a real light.
    SignalPost,
    /// A utility pole and its cross-arm.
    UtilityPole,
    /// The cables from one pole to the next.
    Span,
    /// A stop sign on a post.
    Sign,
    /// A bench.
    Bench,
    /// A litter bin.
    Bin,
    /// A fire hydrant.
    Hydrant,
    /// A mailbox.
    Mailbox,
    /// A bus shelter (wave PAR1b.2): back and end panels and a roof, against
    /// the frontage on an arterial.
    BusShelter,
    /// A parking meter (wave PAR1b.2): a post with a head, on the post line
    /// between kerb slots.
    ParkingMeter,
}

impl PieceKind {
    /// Every kind, in catalogue order (the PAR1b brief's order; PAR1b.2's
    /// kinds appended so every earlier code is unchanged).
    pub const ALL: [PieceKind; 11] = [
        PieceKind::LampPost,
        PieceKind::SignalPost,
        PieceKind::UtilityPole,
        PieceKind::Span,
        PieceKind::Sign,
        PieceKind::Bench,
        PieceKind::Bin,
        PieceKind::Hydrant,
        PieceKind::Mailbox,
        PieceKind::BusShelter,
        PieceKind::ParkingMeter,
    ];
    /// A short name for census tables.
    pub fn name(self) -> &'static str {
        match self {
            PieceKind::LampPost => "lamp post",
            PieceKind::SignalPost => "signal post",
            PieceKind::UtilityPole => "utility pole",
            PieceKind::Span => "cable span",
            PieceKind::Sign => "road sign",
            PieceKind::Bench => "bench",
            PieceKind::Bin => "bin",
            PieceKind::Hydrant => "hydrant",
            PieceKind::Mailbox => "mailbox",
            PieceKind::BusShelter => "bus shelter",
            PieceKind::ParkingMeter => "parking meter",
        }
    }
    /// The census code (declaration index).
    pub fn code(self) -> u8 {
        Self::ALL.iter().position(|k| *k == self).unwrap_or(0) as u8
    }
}

/// **One placed piece** — where its foot stands and everything it puts into a
/// block's population.
#[derive(Debug, Clone, PartialEq)]
pub struct FurniturePiece {
    /// What it is.
    pub kind: PieceKind,
    /// Its foot on the pavement surface, world metres (a span's foot is its
    /// first pole's).
    pub foot: DVec3,
    /// Its content-addressed identity ([`piece_id`]).
    pub id: u64,
    /// The drawn parts, `ModuleShape::Street` prisms.
    pub instances: Vec<PcgInstance>,
    /// The solid boxes a car and a body meet.
    pub colliders: Vec<PcgCollider>,
    /// Its real lights.
    pub lights: Vec<PcgLight>,
}

/// **What [`furnish`] answers.**
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Furnished {
    /// Every piece, in derivation order (street order, then side, then along).
    pub pieces: Vec<FurniturePiece>,
    /// Lattice slots refused because the ground under them has not paged in.
    pub groundless: usize,
    /// Lattice slots refused at a junction / crossing.
    pub at_junction: usize,
    /// Lattice slots refused at a doorway.
    pub at_door: usize,
}

/// **A street's kerb face**, metres from its centreline — what
/// `inf_ecs::traffic::street_kerb_offset_m` answers, by value (the lane count a
/// reserve implies, rounded, times half a lane).
pub fn kerb_offset_m(gap_m: f64) -> f64 {
    let half = (gap_m * 0.5 - PAVEMENT_M - KERB_WIDTH_M).max(LANE_WIDTH_M);
    let want = half * 2.0 / LANE_WIDTH_M;
    let lanes = if want.is_finite() {
        (want.round() as i64).clamp(2, 8) as f64
    } else {
        2.0
    };
    lanes * LANE_WIDTH_M * 0.5
}

/// **The furniture line**, metres from a street's centreline.
pub fn furniture_line_m(gap_m: f64) -> f64 {
    // A reserve whose lane count rounds UP (24 m: six lanes, a 10.5 m kerb)
    // draws its footway past the block's own edge; the line stays inside the
    // reserve, `FURNITURE_BACK_M` short of the frontage.
    (kerb_offset_m(gap_m) + KERB_WIDTH_M + PAVEMENT_M - FURNITURE_BACK_M)
        .min(gap_m * 0.5 - FURNITURE_BACK_M)
}

/// Whether a reserve parks cars at its kerb — `inf_ecs::traffic::kerb_fits`
/// by value: the footway leaves a slot and a metre.
pub fn parks_at_kerb(gap_m: f64) -> bool {
    gap_m * 0.5 - PAVEMENT_M >= KERB_PARK_OFFSET_M + 1.0
}

/// **The post line**, metres from a street's centreline (wave PAR1b.2):
/// [`KERB_POST_OFFSET_M`] behind the kerb face, behind a parked car's door
/// where the reserve parks at its kerb ([`PARKED_CLEAR_M`]), and never behind
/// the broad pieces' line.
pub fn post_line_m(gap_m: f64) -> f64 {
    let kerb = kerb_offset_m(gap_m) + KERB_POST_OFFSET_M;
    let line = if parks_at_kerb(gap_m) {
        kerb.max(PARKED_CLEAR_M)
    } else {
        kerb
    };
    line.min(furniture_line_m(gap_m))
}

/// **The pavement's surface** at `lateral` metres from a street's centreline,
/// world Y — ROAD1's own cross-section over the ground at the kerb face beside
/// it: the ribbon's lift, the kerb's upstand, then the footway's 2 % fall
/// rising toward its back. `None` when the ground has not paged in.
pub fn pavement_y(
    street: &StreetLine,
    along: f64,
    side: f64,
    lateral: f64,
    ground: &dyn HeightProvider,
) -> Option<f64> {
    let kerb = kerb_offset_m(street.gap_m);
    let face = street.at(along, side * kerb);
    let g = ground.height(face.x, face.y)?;
    if !g.is_finite() {
        return None;
    }
    let up = (lateral - kerb - KERB_WIDTH_M).max(0.0) * PAVEMENT_FALL;
    let paved = g + ROAD_LIFT_M + KERB_HEIGHT_M + up;
    // **Where the ground across the footway climbs above the slab** (a street
    // cut into a slope), the slab's back is under the grass the terrain draws,
    // and what a foot stands on is that ground — measured on the CI island, 8
    // of 33 pieces stood up to 0.4 m inside the bank before this.
    let here = street.at(along, side * lateral);
    match ground.height(here.x, here.y) {
        Some(h) if h.is_finite() && h > paved => Some(h),
        _ => Some(paved),
    }
}

/// **A piece's content-addressed identity** — its kind and its foot quantized
/// to a centimetre in plan (the height is the ground's, and moves when a tile
/// pages; `parked_car_guid`'s reason).
pub fn piece_id(kind: PieceKind, foot: DVec3) -> u64 {
    let q = |v: f64| {
        if v.is_finite() {
            (v / 0.01).round() as i64 as u64
        } else {
            0
        }
    };
    let mut x = 0x5041_5231_4246_5552_u64 ^ u64::from(kind.code());
    for lane in [q(foot.x), q(foot.z)] {
        x ^= lane.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x = x.rotate_left(29) ^ x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    }
    x ^ (x >> 31)
}

/// A unit-interval draw from a lattice slot, for the optional kinds' hash
/// occupancy.
fn slot_unit(kind: PieceKind, street_perp: f64, side: f64, k: i64) -> f64 {
    let q = (street_perp / 0.01).round() as i64 as u64;
    let mut x = 0x4f43_4355_5059_0001_u64 ^ (u64::from(kind.code()) << 56);
    for lane in [q, k as u64, side.to_bits()] {
        x ^= lane.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x = x.rotate_left(31) ^ x.wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// The `cosh` of `x` as its Taylor series — exact to the last bit that
/// matters for `|x| < 2` and free of any libm call (the P14 law: a catenary
/// is content two hosts compare).
fn cosh_series(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..14 {
        let k = (2 * n) as f64;
        term *= x2 / (k * (k - 1.0));
        sum += term;
    }
    sum
}

/// The `sinh` of `x` as its Taylor series (see [`cosh_series`]).
fn sinh_series(x: f64) -> f64 {
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..14 {
        let k = (2 * n + 1) as f64;
        term *= x2 / (k * (k - 1.0));
        sum += term;
    }
    sum
}

/// **The catenary parameter `a`** for a level span of `span_m` that sags
/// `sag_m` at mid-span: `a (cosh(L / 2a) - 1) = s`, solved by Newton from the
/// parabola's `L² / 8s`.
pub fn catenary_a(span_m: f64, sag_m: f64) -> f64 {
    if !(span_m > 0.0 && sag_m > 0.0) {
        return f64::INFINITY;
    }
    let mut a = span_m * span_m / (8.0 * sag_m);
    for _ in 0..12 {
        let u = span_m / (2.0 * a);
        let f = a * (cosh_series(u) - 1.0) - sag_m;
        let df = cosh_series(u) - 1.0 - u * sinh_series(u);
        if df.abs() < 1e-15 {
            break;
        }
        let next = a - f / df;
        if !(next.is_finite() && next > 0.0) {
            break;
        }
        a = next;
    }
    a
}

/// **The drop below the chord** at `x` along a span of `span_m` with catenary
/// parameter `a` — `a (cosh(L/2a) - cosh((x - L/2)/a))`, zero at both ends.
pub fn catenary_drop(span_m: f64, a: f64, x: f64) -> f64 {
    if !a.is_finite() {
        return 0.0;
    }
    let h = span_m * 0.5;
    a * (cosh_series(h / a) - cosh_series((x - h) / a))
}

/// A tinted `Street` prism instance: centre, unit axis its length runs along,
/// half-length, and half-width across.
fn prism(centre: DVec3, axis: DVec3, half_len: f64, half_w: f64, tint: [f32; 4]) -> PcgInstance {
    let rot = if axis.y > 0.9999 {
        DQuat::IDENTITY
    } else {
        DQuat::from_rotation_arc(DVec3::Y, axis.normalize_or(DVec3::Y))
    };
    PcgInstance {
        pos: centre,
        rotation: rot,
        scale: 1.0,
        kind_index: 0,
        mesh: Some(module_mesh_guid(ModuleShape::Street)),
        extent: Some([half_w as f32, half_len as f32, half_w as f32]),
        glow: 0.0,
        surface: PcgSurface {
            tint: Some(tint),
            ..PcgSurface::DEFAULT
        },
    }
}

/// A box collider, axis-aligned in plan (every settlement street is).
fn solid(centre: DVec3, half: DVec3) -> PcgCollider {
    PcgCollider {
        center: centre,
        half_extents: half,
        rotation: DQuat::IDENTITY,
    }
}

/// Galvanised steel.
const STEEL: [f32; 4] = [0.34, 0.35, 0.36, 1.0];
/// A signal's black housing.
const HOUSING: [f32; 4] = [0.05, 0.05, 0.05, 1.0];
/// A creosoted timber pole.
const TIMBER: [f32; 4] = [0.27, 0.20, 0.14, 1.0];
/// Cable.
const CABLE: [f32; 4] = [0.04, 0.04, 0.04, 1.0];
/// Bench slats.
const WOOD: [f32; 4] = [0.42, 0.30, 0.18, 1.0];
/// A bin's green.
const BIN_GREEN: [f32; 4] = [0.10, 0.18, 0.12, 1.0];
/// A hydrant's red.
const HYDRANT_RED: [f32; 4] = [0.62, 0.10, 0.05, 1.0];
/// A mailbox's blue.
const MAIL_BLUE: [f32; 4] = [0.08, 0.15, 0.40, 1.0];
/// A stop sign's red.
const SIGN_RED: [f32; 4] = [0.55, 0.04, 0.04, 1.0];

/// The light tag every street light carries: no building, no room.
fn street_tag(row: FixtureRow) -> FixtureTag {
    FixtureTag {
        building: u32::MAX,
        floor: 0,
        room: FixtureTag::EXTERIOR,
        row,
    }
}

/// **A lamp post** standing at `foot`, its arm reaching `reach` metres along
/// `toward` (the unit plan direction to the carriageway).
fn lamp_post(foot: DVec3, toward: DVec3, reach: f64) -> FurniturePiece {
    let h = LAMP_HEIGHT_M;
    let shaft_top = foot + DVec3::Y * h;
    let head = shaft_top + toward * reach - DVec3::Y * 0.12;
    let def = fixtures::fixture(FixtureRow::StreetLamp);
    let mut head_part = prism(
        head,
        toward,
        def.half[0],
        def.half[2],
        [0.70, 0.70, 0.66, 1.0],
    );
    head_part.surface.emissive = {
        let c = def.colour();
        [
            c[0] * LAMP_HEAD_EMISSIVE,
            c[1] * LAMP_HEAD_EMISSIVE,
            c[2] * LAMP_HEAD_EMISSIVE,
        ]
    };
    head_part.surface.schedule = Some(FixtureSchedule::Dusk);
    head_part.surface.roughness = 0.5;
    let colour = def.colour();
    // The throw: tilted LAMP_TILT toward the carriageway (a cobra head's
    // street-side distribution), so the pool lies on the road and its own
    // column stands at the back edge of the cone, not lit full height by it.
    let beam = -DVec3::Y * LAMP_TILT_COS + toward * LAMP_TILT_SIN;
    FurniturePiece {
        kind: PieceKind::LampPost,
        foot,
        id: piece_id(PieceKind::LampPost, foot),
        instances: vec![
            prism(
                foot + DVec3::Y * (h * 0.5),
                DVec3::Y,
                h * 0.5,
                LAMP_SHAFT_HALF_M,
                STEEL,
            ),
            prism(
                shaft_top + toward * (reach * 0.5) - DVec3::Y * 0.04,
                toward,
                reach * 0.5,
                0.05,
                STEEL,
            ),
            head_part,
        ],
        colliders: vec![solid(
            foot + DVec3::Y * (h * 0.5),
            DVec3::new(LAMP_SHAFT_HALF_M, h * 0.5, LAMP_SHAFT_HALF_M),
        )],
        lights: vec![PcgLight {
            at: head - DVec3::Y * fixtures::DIFFUSER_GAP_M,
            dir: beam,
            sweep: (colour, colour),
            intensity: def.intensity(),
            range_m: LAMP_RANGE_M,
            inner_deg: def.cone_deg.map_or(180.0, |c| c.0),
            outer_deg: def.cone_deg.map_or(180.0, |c| c.1),
            cycle_hz: 0.0,
            phase: 0,
            phases: 1,
            schedule: FixtureSchedule::Dusk,
            occupancy: FixtureOccupancy::Crew,
            seed: 0,
            tag: street_tag(FixtureRow::StreetLamp),
            clip: None,
            draw_m: LAMP_DRAW_M,
            shadow: false,
        }],
    }
}

/// **The signal head light's aspect colours**, linear — red, amber, green
/// LED lenses.
pub fn aspect_rgb(aspect: u8) -> [f32; 3] {
    match aspect {
        0 => [0.10, 1.0, 0.45],
        1 => [1.0, 0.55, 0.05],
        _ => [1.0, 0.06, 0.03],
    }
}

/// **A signal mast** at `foot` serving the approach that travels along `d`: its
/// arm reaches `reach` metres along `toward`, the head hangs at
/// [`SIGNAL_HEAD_HEIGHT_M`] facing back down the approach (`-d`).
fn signal_post(
    foot: DVec3,
    d: DVec3,
    toward: DVec3,
    reach: f64,
    site: &SignalSite,
) -> FurniturePiece {
    let top = foot + DVec3::Y * SIGNAL_MAST_M;
    let head = foot + toward * reach + DVec3::Y * SIGNAL_HEAD_HEIGHT_M;
    let def = fixtures::fixture(FixtureRow::SignalHead);
    // The housing stands upright; its depth runs along the approach.
    let housing = PcgInstance {
        extent: Some(if d.x.abs() > 0.5 {
            [0.15, 0.5, 0.17]
        } else {
            [0.17, 0.5, 0.15]
        }),
        ..prism(head, DVec3::Y, 0.5, 0.17, HOUSING)
    };
    // The beam: from the lit lens down the approach to its stop line, which
    // is a junction's width back from the head.
    let aim = (-d * (2.0 * reach.max(8.0) + 2.0)) - DVec3::Y * SIGNAL_HEAD_HEIGHT_M;
    let along_x = d.x.abs() > 0.5;
    FurniturePiece {
        kind: PieceKind::SignalPost,
        foot,
        id: piece_id(PieceKind::SignalPost, foot),
        instances: vec![
            prism(
                foot + DVec3::Y * (SIGNAL_MAST_M * 0.5),
                DVec3::Y,
                SIGNAL_MAST_M * 0.5,
                SIGNAL_MAST_HALF_M,
                STEEL,
            ),
            prism(
                top + toward * (reach * 0.5) - DVec3::Y * 0.2,
                toward,
                reach * 0.5,
                0.07,
                STEEL,
            ),
            prism(
                foot + toward * reach
                    + DVec3::Y * ((SIGNAL_MAST_M - 0.2 + SIGNAL_HEAD_HEIGHT_M + 0.5) * 0.5),
                DVec3::Y,
                (SIGNAL_MAST_M - 0.2 - SIGNAL_HEAD_HEIGHT_M - 0.5).max(0.05) * 0.5,
                0.03,
                STEEL,
            ),
            housing,
        ],
        colliders: vec![solid(
            foot + DVec3::Y * (SIGNAL_MAST_M * 0.5),
            DVec3::new(SIGNAL_MAST_HALF_M, SIGNAL_MAST_M * 0.5, SIGNAL_MAST_HALF_M),
        )],
        lights: vec![PcgLight {
            at: head - d * 0.25,
            dir: aim.normalize_or(-DVec3::Y),
            sweep: (aspect_rgb(2), aspect_rgb(2)),
            intensity: def.intensity(),
            range_m: SIGNAL_RANGE_M,
            inner_deg: def.cone_deg.map_or(180.0, |c| c.0),
            outer_deg: def.cone_deg.map_or(180.0, |c| c.1),
            cycle_hz: 0.0,
            // **The phase key**: `phase` is the approach's axis (0 = along X,
            // 1 = along Z), `phases` = 2 marks a signal, and `seed` is the
            // controller's offset — what `inf_ecs::traffic::aspect_at` reads.
            phase: u32::from(!along_x),
            phases: 2,
            schedule: FixtureSchedule::Always,
            occupancy: FixtureOccupancy::Crew,
            seed: site.offset_s,
            tag: street_tag(FixtureRow::SignalHead),
            clip: None,
            draw_m: LAMP_DRAW_M,
            shadow: false,
        }],
    }
}

/// The two conductors' and the telecom line's `(lateral, height, radius)` on a
/// pole, metres — lateral toward the carriageway from the pole's side line.
/// Three lines, not four: each line is [`CABLE_SEGMENTS`] instances a span, and
/// the fourth cost more frame than it read (the PAR1b frame delta).
const CABLES: [(f64, f64, f64); 3] = [
    (-0.75, POLE_HEIGHT_M - 0.45, 0.018),
    (0.4, POLE_HEIGHT_M - 0.45, 0.018),
    (0.3, 7.5, 0.028),
];

/// **A utility pole** at `foot`, its cross-arm across the street (`across` is
/// the unit plan direction toward the carriageway).
fn utility_pole(foot: DVec3, across: DVec3) -> FurniturePiece {
    let h = POLE_HEIGHT_M;
    FurniturePiece {
        kind: PieceKind::UtilityPole,
        foot,
        id: piece_id(PieceKind::UtilityPole, foot),
        instances: vec![
            prism(
                foot + DVec3::Y * (h * 0.5),
                DVec3::Y,
                h * 0.5,
                POLE_HALF_M,
                TIMBER,
            ),
            prism(
                foot + DVec3::Y * (h - 0.35) + across * 0.175,
                across,
                0.95,
                0.06,
                TIMBER,
            ),
            prism(
                foot + DVec3::Y * 7.5 - across * 0.3,
                across,
                0.12,
                0.05,
                STEEL,
            ),
        ],
        colliders: vec![solid(
            foot + DVec3::Y * (h * 0.5),
            DVec3::new(POLE_HALF_M, h * 0.5, POLE_HALF_M),
        )],
        lights: Vec::new(),
    }
}

/// **The cables from one pole to the next** — every line a true catenary
/// ([`catenary_drop`]) sagging [`CABLE_SAG_FRAC`] of its span, drawn as
/// [`CABLE_SEGMENTS`] thin prisms.
fn span(p0: DVec3, p1: DVec3, across: DVec3) -> FurniturePiece {
    let mut instances = Vec::with_capacity(CABLES.len() * CABLE_SEGMENTS);
    let plan = DVec3::new(p1.x - p0.x, 0.0, p1.z - p0.z);
    let len = plan.length();
    let a = catenary_a(len, CABLE_SAG_FRAC * len);
    // "Lateral" on the pole is toward the carriageway; a cable at +lateral
    // hangs on the street side of the pole.
    for (lat, height, r) in CABLES {
        let at = |t: f64| -> DVec3 {
            let base = p0 + (p1 - p0) * t + DVec3::Y * height + across * lat;
            base - DVec3::Y * catenary_drop(len, a, t * len)
        };
        for k in 0..CABLE_SEGMENTS {
            let (q0, q1) = (
                at(k as f64 / CABLE_SEGMENTS as f64),
                at((k + 1) as f64 / CABLE_SEGMENTS as f64),
            );
            let d = q1 - q0;
            let l = d.length();
            if l <= 0.0 {
                continue;
            }
            instances.push(prism((q0 + q1) * 0.5, d / l, l * 0.5 + r, r, CABLE));
        }
    }
    FurniturePiece {
        kind: PieceKind::Span,
        foot: p0,
        id: piece_id(PieceKind::Span, p0),
        instances,
        colliders: Vec::new(),
        lights: Vec::new(),
    }
}

/// **A stop sign** on a post at `foot`, its face toward `-d` (the approach).
fn stop_sign(foot: DVec3, d: DVec3) -> FurniturePiece {
    FurniturePiece {
        kind: PieceKind::Sign,
        foot,
        id: piece_id(PieceKind::Sign, foot),
        instances: vec![
            prism(foot + DVec3::Y * 1.25, DVec3::Y, 1.25, 0.04, STEEL),
            prism(foot + DVec3::Y * 2.2 - d * 0.05, -d, 0.02, 0.32, SIGN_RED),
        ],
        colliders: vec![solid(foot + DVec3::Y * 1.25, DVec3::new(0.05, 1.25, 0.05))],
        lights: Vec::new(),
    }
}

/// **A bench** at `foot`, running along `axis`, its back toward `-toward`.
fn bench(foot: DVec3, axis: DVec3, toward: DVec3) -> FurniturePiece {
    FurniturePiece {
        kind: PieceKind::Bench,
        foot,
        id: piece_id(PieceKind::Bench, foot),
        instances: vec![
            prism(foot + DVec3::Y * 0.45, axis, 0.9, 0.22, WOOD),
            prism(foot + DVec3::Y * 0.8 - toward * 0.2, axis, 0.9, 0.09, WOOD),
            prism(
                foot + DVec3::Y * 0.2 + axis * 0.7,
                DVec3::Y,
                0.2,
                0.05,
                STEEL,
            ),
            prism(
                foot + DVec3::Y * 0.2 - axis * 0.7,
                DVec3::Y,
                0.2,
                0.05,
                STEEL,
            ),
        ],
        colliders: vec![solid(
            foot + DVec3::Y * 0.45 - toward * 0.05,
            if axis.x.abs() > 0.5 {
                DVec3::new(0.9, 0.45, 0.3)
            } else {
                DVec3::new(0.3, 0.45, 0.9)
            },
        )],
        lights: Vec::new(),
    }
}

/// An upright drum: a bin, a hydrant, a mailbox.
fn drum(kind: PieceKind, foot: DVec3, half_h: f64, r: f64, tint: [f32; 4]) -> FurniturePiece {
    let mut instances = vec![prism(foot + DVec3::Y * half_h, DVec3::Y, half_h, r, tint)];
    if kind == PieceKind::Hydrant {
        instances.push(prism(
            foot + DVec3::Y * (2.0 * half_h + 0.05),
            DVec3::Y,
            0.05,
            r * 0.7,
            tint,
        ));
        instances.push(prism(
            foot + DVec3::Y * (half_h * 1.3),
            DVec3::X,
            r * 1.6,
            0.05,
            tint,
        ));
    }
    FurniturePiece {
        kind,
        foot,
        id: piece_id(kind, foot),
        instances,
        colliders: vec![solid(foot + DVec3::Y * half_h, DVec3::new(r, half_h, r))],
        lights: Vec::new(),
    }
}

/// A shelter panel's glazing (smoked, opaque in this renderer's opaque pass).
const GLAZING: [f32; 4] = [0.22, 0.27, 0.30, 1.0];
/// A meter's head.
const METER_GREY: [f32; 4] = [0.30, 0.32, 0.30, 1.0];

/// **A bus shelter's length along the street**, metres (a two-bay steel
/// shelter: 3.2 m).
pub const SHELTER_LEN_M: f64 = 3.2;
/// Its depth across the footway, metres.
pub const SHELTER_DEPTH_M: f64 = 1.2;
/// Its eave height, metres.
pub const SHELTER_HEIGHT_M: f64 = 2.4;
/// **Bus shelter spacing**, metres, on each side of an arterial (a 20 m+
/// city street) — a stop every 240 m, the two sides offset by half, there
/// being no bus layer in the society to place them at stops.
pub const SHELTER_SPACING_M: f64 = 240.0;
/// **Kerb slots per parking meter** — one meter for every three spaces.
pub const SLOTS_PER_METER: f64 = 3.0;

/// A panel instance: centre, plan unit axis its LENGTH runs along, and its
/// half-length / half-height / half-thickness.
fn panel(
    centre: DVec3,
    along: DVec3,
    half_len: f64,
    half_h: f64,
    half_t: f64,
    tint: [f32; 4],
) -> PcgInstance {
    let x_long = along.x.abs() > 0.5;
    PcgInstance {
        extent: Some(if x_long {
            [half_len as f32, half_h as f32, half_t as f32]
        } else {
            [half_t as f32, half_h as f32, half_len as f32]
        }),
        ..prism(centre, DVec3::Y, half_h, half_t, tint)
    }
}

/// **A bus shelter** (the PAR1b.2 audit's placement): a kerbside canopy whose
/// glazed screen stands on the POST line at `kerb_side` (the line's point at
/// the shelter's middle), facing the traffic, with its roof cantilevered back
/// over the footway away from `toward` (the carriageway) and a lean rail on the
/// screen's footway face, running along `axis`.
///
/// # Why at the kerb, open to the footway
///
/// The wave first stood the shelter against the frontage, its back panel 5 cm
/// off the building line — and on `phase19_gate`'s shipped town that panel
/// stood in an Office's ground-floor window band (the arm's red). A 2.3 m
/// footway has no room for a three-sided 1.2 m shelter anywhere a crowd walks,
/// so this is the narrow-footway shelter real streets use: one screen at the
/// kerb (0.5 m behind the kerb face, clear of a parked car's door), the roof
/// over the walking line, NO end panels (the crowd ring passes under the roof,
/// 0.42 m clear of the screen's face on a 20 m street), nothing against any
/// frontage — so no window band and no doorway is ever behind one.
fn bus_shelter(kerb_side: DVec3, axis: DVec3, toward: DVec3) -> FurniturePiece {
    let (hl, hd, h) = (SHELTER_LEN_M * 0.5, SHELTER_DEPTH_M * 0.5, SHELTER_HEIGHT_M);
    let t = 0.04;
    let away = -toward;
    let mid = kerb_side + away * hd;
    let screen_c = kerb_side + away * t + DVec3::Y * (h * 0.5);
    let instances = vec![
        // The screen.
        panel(screen_c, axis, hl, h * 0.5, t, GLAZING),
        // The roof, over the footway, overhanging the screen by 0.1 m.
        panel(
            mid + toward * 0.05 + DVec3::Y * (h + 0.05),
            axis,
            hl + 0.1,
            0.05,
            hd + 0.05,
            STEEL,
        ),
        // The lean rail on the screen's footway face (drawn; the screen is
        // the solid a body meets).
        panel(
            kerb_side + away * (2.0 * t + 0.025) + DVec3::Y * 0.8,
            axis,
            hl - 0.3,
            0.03,
            0.025,
            STEEL,
        ),
    ];
    let colliders = vec![solid(
        screen_c,
        if axis.x.abs() > 0.5 {
            DVec3::new(hl, h * 0.5, t)
        } else {
            DVec3::new(t, h * 0.5, hl)
        },
    )];
    let foot = DVec3::new(mid.x, kerb_side.y, mid.z);
    FurniturePiece {
        kind: PieceKind::BusShelter,
        foot,
        id: piece_id(PieceKind::BusShelter, foot),
        instances,
        colliders,
        lights: Vec::new(),
    }
}

/// **A parking meter**: a 1.3 m post with its head.
fn parking_meter(foot: DVec3) -> FurniturePiece {
    FurniturePiece {
        kind: PieceKind::ParkingMeter,
        foot,
        id: piece_id(PieceKind::ParkingMeter, foot),
        instances: vec![
            prism(foot + DVec3::Y * 0.55, DVec3::Y, 0.55, 0.04, STEEL),
            prism(foot + DVec3::Y * 1.2, DVec3::Y, 0.14, 0.09, METER_GREY),
        ],
        colliders: vec![solid(foot + DVec3::Y * 0.65, DVec3::new(0.07, 0.65, 0.07))],
        lights: Vec::new(),
    }
}

/// **The junction clearance** of an along-coordinate on `street`: the nearest
/// crossing street's centre is closer than its own half reserve or the
/// crossing's far edge (whichever is further) plus [`JUNCTION_CLEAR_M`].
fn near_junction(streets: &[StreetLine], street: &StreetLine, along: f64) -> bool {
    let perp = street.perp();
    let (lo, hi) = street.span();
    for o in streets {
        if o.along_x() == street.along_x() {
            continue;
        }
        let (olo, ohi) = o.span();
        let op = o.perp();
        // Does `o` meet this line (cross it or end on it)?
        let meets =
            op >= lo - 1.0 && op <= hi + 1.0 && perp >= olo - o.gap_m && perp <= ohi + o.gap_m;
        if !meets {
            continue;
        }
        let clear = (o.gap_m * 0.5).max(CROSSWALK_FAR_M) + JUNCTION_CLEAR_M;
        if (along - op).abs() < clear {
            return true;
        }
    }
    false
}

/// **Every piece of street furniture the streets imply** (wave PAR1b) — the
/// whole derivation, a pure function of its four inputs.
///
/// `streets` are the level's street centrelines; `signals` the signalised
/// junctions among their crossings; `doors` the blocks' exterior doors (plan
/// positions of their hinges); `ground` the terrain.
pub fn furnish(
    streets: &[StreetLine],
    signals: &[SignalSite],
    doors: &[DVec2],
    ground: &dyn HeightProvider,
) -> Furnished {
    let mut out = Furnished::default();
    let door_near = |p: DVec2| doors.iter().any(|d| (*d - p).length() < DOOR_CLEAR_M);
    for street in streets {
        if !(street.gap_m.is_finite() && street.gap_m > 0.0) {
            continue;
        }
        let (lo, hi) = street.span();
        if !(lo.is_finite() && hi.is_finite() && hi - lo > 1.0) {
            continue;
        }
        let kerb = kerb_offset_m(street.gap_m);
        let line = furniture_line_m(street.gap_m);
        let post = post_line_m(street.gap_m);
        let wide = street.gap_m >= 20.0 - 1e-9;
        for side in [1.0f64, -1.0] {
            // Every piece placed on this side so far, by along-coordinate.
            let mut taken: Vec<f64> = Vec::new();
            let toward = -street.side_dir() * side;
            let axis = street.axis();
            // One lattice walk per kind, in catalogue priority, on the
            // lateral line the kind stands on (`post` or `line`).
            let place = |kind: PieceKind,
                         spacing: f64,
                         phase: f64,
                         share: f64,
                         line: f64,
                         taken: &mut Vec<f64>,
                         out: &mut Furnished|
             -> Vec<(f64, DVec3)> {
                let mut placed = Vec::new();
                let k0 = ((lo - phase) / spacing).ceil() as i64;
                let k1 = ((hi - phase) / spacing).floor() as i64;
                for k in k0..=k1 {
                    let along = k as f64 * spacing + phase;
                    if along < lo + 1.0 || along > hi - 1.0 {
                        continue;
                    }
                    if share < 1.0 && slot_unit(kind, street.perp(), side, k) >= share {
                        continue;
                    }
                    if near_junction(streets, street, along) {
                        out.at_junction += 1;
                        continue;
                    }
                    if taken.iter().any(|t| (t - along).abs() < MIN_SEPARATION_M) {
                        continue;
                    }
                    let p = street.at(along, side * line);
                    if door_near(p) {
                        out.at_door += 1;
                        continue;
                    }
                    let Some(y) = pavement_y(street, along, side, line, ground) else {
                        out.groundless += 1;
                        continue;
                    };
                    taken.push(along);
                    placed.push((along, DVec3::new(p.x, y, p.y)));
                }
                placed
            };
            // ── lamps: opposite on a city street, staggered on a town one.
            let lamp_phase = if wide || side > 0.0 {
                0.0
            } else {
                LAMP_SPACING_M * 0.5
            };
            let reach = post - (kerb - LAMP_OVERHANG_M);
            for (_, foot) in place(
                PieceKind::LampPost,
                LAMP_SPACING_M,
                lamp_phase,
                1.0,
                post,
                &mut taken,
                &mut out,
            ) {
                out.pieces.push(lamp_post(foot, toward, reach));
            }
            // ── poles + cables: one pole line per street, on its -side.
            if side < 0.0 {
                // The pole lattice sits 5 m off every lamp the same side line
                // carries: the lamps are at 0 mod 30 (opposite) or 15 mod 30
                // (staggered), and 40 j + 25 / 40 j + 20 is never nearer either
                // than 5 m — so a pole never yields its slot to a lamp and the
                // line keeps its spacing.
                let pole_phase = if wide { 25.0 } else { 20.0 };
                let poles = place(
                    PieceKind::UtilityPole,
                    POLE_SPACING_M,
                    pole_phase,
                    1.0,
                    post,
                    &mut taken,
                    &mut out,
                );
                for (_, foot) in &poles {
                    out.pieces.push(utility_pole(*foot, toward));
                }
                for w in poles.windows(2) {
                    if w[1].0 - w[0].0 <= MAX_SPAN_M {
                        out.pieces.push(span(w[0].1, w[1].1, toward));
                    }
                }
            }
            // ── benches + bins: on the +side, between lamps, a share of slots.
            // ── bus shelters (PAR1b.2): on an arterial, every
            //    `SHELTER_SPACING_M` a side, the sides offset by half; the
            //    screen on the post line, open to the footway (the PAR1b.2
            //    audit moved it off the frontage — see `bus_shelter`).
            //    Placed before the meters and benches so a shelter's length
            //    is clear.
            if wide {
                let shelter_phase = if side > 0.0 {
                    100.0
                } else {
                    100.0 + SHELTER_SPACING_M * 0.5
                };
                let back = post;
                let k0 = ((lo - shelter_phase) / SHELTER_SPACING_M).ceil() as i64;
                let k1 = ((hi - shelter_phase) / SHELTER_SPACING_M).floor() as i64;
                for k in k0..=k1 {
                    let along = k as f64 * SHELTER_SPACING_M + shelter_phase;
                    let hl = SHELTER_LEN_M * 0.5;
                    if along - hl < lo + 1.0 || along + hl > hi - 1.0 {
                        continue;
                    }
                    if near_junction(streets, street, along - hl)
                        || near_junction(streets, street, along + hl)
                    {
                        out.at_junction += 1;
                        continue;
                    }
                    if taken
                        .iter()
                        .any(|t| (t - along).abs() < hl + MIN_SEPARATION_M)
                    {
                        continue;
                    }
                    let p = street.at(along, side * back);
                    if doors.iter().any(|d| (*d - p).length() < DOOR_CLEAR_M + hl) {
                        out.at_door += 1;
                        continue;
                    }
                    let Some(y) = pavement_y(street, along, side, back, ground) else {
                        out.groundless += 1;
                        continue;
                    };
                    taken.push(along - hl);
                    taken.push(along);
                    taken.push(along + hl);
                    out.pieces
                        .push(bus_shelter(DVec3::new(p.x, y, p.y), axis, toward));
                }
            }
            // ── parking meters (PAR1b.2): on the post line, one for every
            //    `SLOTS_PER_METER` kerb slots, at the boundary between two
            //    slots on their world lattice (`kerb_slots`' own pitch), where
            //    the reserve parks at its kerb.
            if parks_at_kerb(street.gap_m) {
                let pitch = KERB_SLOT_M * SLOTS_PER_METER;
                let phase = KERB_SLOT_M * 0.5 + if side > 0.0 { 0.0 } else { KERB_SLOT_M };
                for (_, foot) in place(
                    PieceKind::ParkingMeter,
                    pitch,
                    phase,
                    1.0,
                    post,
                    &mut taken,
                    &mut out,
                ) {
                    out.pieces.push(parking_meter(foot));
                }
            }
            if side > 0.0 {
                for (along, foot) in place(
                    PieceKind::Bench,
                    60.0,
                    15.0,
                    0.75,
                    line,
                    &mut taken,
                    &mut out,
                ) {
                    out.pieces.push(bench(foot, axis, toward));
                    // The bin beside it, when the line has room.
                    let b_along = along + 1.9;
                    let p = street.at(b_along, side * line);
                    if !near_junction(streets, street, b_along)
                        && !taken
                            .iter()
                            .any(|t| (t - b_along).abs() < MIN_SEPARATION_M * 0.5)
                        && !door_near(p)
                    {
                        if let Some(y) = pavement_y(street, b_along, side, line, ground) {
                            taken.push(b_along);
                            out.pieces.push(drum(
                                PieceKind::Bin,
                                DVec3::new(p.x, y, p.y),
                                0.5,
                                0.28,
                                BIN_GREEN,
                            ));
                        }
                    }
                }
                for (_, foot) in place(
                    PieceKind::Mailbox,
                    120.0,
                    47.0,
                    0.6,
                    line,
                    &mut taken,
                    &mut out,
                ) {
                    out.pieces
                        .push(drum(PieceKind::Mailbox, foot, 0.6, 0.26, MAIL_BLUE));
                }
            } else {
                for (_, foot) in place(
                    PieceKind::Hydrant,
                    90.0,
                    53.0,
                    0.8,
                    post,
                    &mut taken,
                    &mut out,
                ) {
                    out.pieces
                        .push(drum(PieceKind::Hydrant, foot, 0.36, 0.15, HYDRANT_RED));
                }
            }
        }
    }
    // ── the signals: a mast at the far-right corner of every approach of every
    //    signalised junction, its arm over that approach's lanes.
    for site in signals {
        for d in [DVec3::X, -DVec3::X, DVec3::Z, -DVec3::Z] {
            let along_x = d.x.abs() > 0.5;
            let (own_gap, cross_gap) = if along_x {
                (site.gap_x, site.gap_z)
            } else {
                (site.gap_z, site.gap_x)
            };
            // The right of travel, `inf_nav::lane::right_of`'s sense.
            let right = DVec3::new(d.z, 0.0, -d.x);
            // PAR1b.2: the mast stands on the post line, at the kerb.
            let line = post_line_m(own_gap);
            let kerb = kerb_offset_m(own_gap);
            let past = (cross_gap * 0.5).max(CROSSWALK_FAR_M) + 0.5;
            let c = DVec3::new(site.centre.x, 0.0, site.centre.y);
            let p = c + d * past + right * line;
            // The approach's own street, as a stub through the centre — all
            // `pavement_y` needs is its axis, its perpendicular and its reserve.
            let (sa, sb) = if along_x {
                (
                    DVec2::new(site.centre.x - 1.0, site.centre.y),
                    DVec2::new(site.centre.x + 1.0, site.centre.y),
                )
            } else {
                (
                    DVec2::new(site.centre.x, site.centre.y - 1.0),
                    DVec2::new(site.centre.x, site.centre.y + 1.0),
                )
            };
            let street = StreetLine {
                a: sa,
                b: sb,
                gap_m: own_gap,
            };
            let along = if along_x { p.x } else { p.z };
            let side = if right.dot(street.side_dir()) > 0.0 {
                1.0
            } else {
                -1.0
            };
            if door_near(DVec2::new(p.x, p.z)) {
                out.at_door += 1;
                continue;
            }
            let Some(y) = pavement_y(&street, along, side, line, ground) else {
                out.groundless += 1;
                continue;
            };
            // The head hangs over the approach's own half, mid-way to the kerb.
            let reach = line - kerb * 0.5;
            out.pieces
                .push(signal_post(DVec3::new(p.x, y, p.z), d, -right, reach, site));
        }
    }
    // ── stop signs: the approaches of every crossing that is NOT signalised,
    //    on the near-right corner, on the narrower street (both when equal).
    for sx in streets.iter().filter(|s| s.along_x()) {
        for sz in streets.iter().filter(|s| !s.along_x()) {
            let (xl, xh) = sx.span();
            let (zl, zh) = sz.span();
            let (x, z) = (sz.perp(), sx.perp());
            if !(x >= xl - 0.5 && x <= xh + 0.5 && z >= zl - 0.5 && z <= zh + 0.5) {
                continue;
            }
            let centre = DVec2::new(x, z);
            if signals.iter().any(|s| (s.centre - centre).length() < 1.0) {
                continue;
            }
            let ds: Vec<DVec3> = [DVec3::X, -DVec3::X, DVec3::Z, -DVec3::Z]
                .into_iter()
                .filter(|d| {
                    let along_x = d.x.abs() > 0.5;
                    let (own, cross) = if along_x { (sx, sz) } else { (sz, sx) };
                    // The approach exists (its arm runs back from the centre)…
                    let (lo, hi) = own.span();
                    let c = if along_x { x } else { z };
                    let back = if along_x { -d.x } else { -d.z };
                    let arm = if back > 0.0 { hi - c } else { c - lo };
                    // …and it is the narrower street's (or they are equal).
                    arm > cross.gap_m * 0.5 + 6.0 && own.gap_m <= cross.gap_m + 1e-9
                })
                .collect();
            for d in ds {
                let along_x = d.x.abs() > 0.5;
                let (own, cross) = if along_x { (sx, sz) } else { (sz, sx) };
                let right = DVec3::new(d.z, 0.0, -d.x);
                // PAR1b.2: the sign post stands on the post line.
                let line = post_line_m(own.gap_m);
                let back = (cross.gap_m * 0.5).max(CROSSWALK_FAR_M) + 0.8;
                let c = DVec3::new(x, 0.0, z);
                let p = c - d * back + right * line;
                let along = if along_x { p.x } else { p.z };
                let side = if right.dot(own.side_dir()) > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                if door_near(DVec2::new(p.x, p.z)) {
                    out.at_door += 1;
                    continue;
                }
                let Some(y) = pavement_y(own, along, side, line, ground) else {
                    out.groundless += 1;
                    continue;
                };
                out.pieces.push(stop_sign(DVec3::new(p.x, y, p.z), d));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::height::FnHeight;

    fn flat() -> FnHeight<impl Fn(f64, f64) -> Option<f64> + Send + Sync> {
        FnHeight::new(|_, _| Some(10.0))
    }

    /// **A catenary is a catenary**: the parameter reproduces the stated sag,
    /// the drop is zero at both poles and largest at mid-span, and it is not
    /// the parabola (the two differ by a measurable amount on a steep sag).
    #[test]
    fn the_span_sags_as_a_catenary() {
        let (l, s) = (40.0, 40.0 * CABLE_SAG_FRAC);
        let a = catenary_a(l, s);
        assert!(
            (catenary_drop(l, a, l * 0.5) - s).abs() < 1e-9,
            "sag {}",
            catenary_drop(l, a, 20.0)
        );
        assert!(catenary_drop(l, a, 0.0).abs() < 1e-12);
        assert!(catenary_drop(l, a, l).abs() < 1e-12);
        assert!(catenary_drop(l, a, 10.0) < s && catenary_drop(l, a, 10.0) > 0.0);
        // cosh vs the parabola, on a deep sag where they part.
        let a2 = catenary_a(40.0, 10.0);
        let parabola_a = 40.0 * 40.0 / 80.0;
        assert!(
            (a2 - parabola_a).abs() > 0.5,
            "a {a2} is the parabola's {parabola_a}"
        );
        // The series matches the identity cosh^2 - sinh^2 = 1.
        for x in [0.0, 0.3, 1.0, 1.9] {
            let (c, sh) = (cosh_series(x), sinh_series(x));
            assert!((c * c - sh * sh - 1.0).abs() < 1e-12, "x {x}");
        }
    }

    /// **The kerb is the one `inf_ecs` draws**: 16 m → 5.25, 20 m → 7.00, and
    /// the furniture line is behind the crowd's ring on both.
    #[test]
    fn the_line_is_behind_the_ring_on_every_reserve() {
        assert!((kerb_offset_m(16.0) - 5.25).abs() < 1e-12);
        assert!((kerb_offset_m(20.0) - 7.0).abs() < 1e-12);
        for gap in [16.0, 18.0, 20.0, 24.0, 32.0] {
            let ring = gap * 0.5 - PAVEMENT_M;
            let line = furniture_line_m(gap);
            let kerb_back = kerb_offset_m(gap) + KERB_WIDTH_M;
            assert!(
                line > kerb_back + 0.5,
                "{gap}: the line is on the kerb stone"
            );
            assert!(
                line - ring >= 0.9,
                "{gap}: line {line} is {} from the ring {ring}",
                line - ring
            );
            assert!(line < gap * 0.5, "{gap}: the line is inside the block");
        }
    }

    fn grid() -> Vec<StreetLine> {
        // Two 20 m streets crossing at the origin, 400 m long each.
        vec![
            StreetLine {
                a: DVec2::new(-200.0, 0.0),
                b: DVec2::new(200.0, 0.0),
                gap_m: 20.0,
            },
            StreetLine {
                a: DVec2::new(0.0, -200.0),
                b: DVec2::new(0.0, 200.0),
                gap_m: 20.0,
            },
        ]
    }

    /// **Lamps at the stated spacing, both sides, none at the junction, every
    /// one a real Dusk light, every one on the pavement surface.**
    #[test]
    fn a_city_street_carries_lamps_at_thirty_metres_on_both_sides() {
        let streets = grid();
        let site = SignalSite {
            centre: DVec2::ZERO,
            gap_x: 20.0,
            gap_z: 20.0,
            offset_s: 7,
        };
        let f = furnish(&streets, &[site], &[], &flat());
        let lamps: Vec<&FurniturePiece> = f
            .pieces
            .iter()
            .filter(|p| p.kind == PieceKind::LampPost)
            .collect();
        // 400 m a street, both sides, less the junction's ±12.5 m: per side
        // 13 + 13 slots on the lattice -> 2 streets x 2 sides x 12..13.
        assert!(lamps.len() >= 4 * 12, "lamps {}", lamps.len());
        for l in &lamps {
            assert_eq!(l.lights.len(), 1);
            let lt = &l.lights[0];
            assert_eq!(lt.schedule, FixtureSchedule::Dusk);
            assert!(lt.clip.is_none(), "a street lamp is unboxed");
            assert!(lt.intensity > 10.0);
            let line = post_line_m(20.0);
            let lateral = if (l.foot.z.abs() - line).abs() < 1e-9 {
                l.foot.z
            } else {
                l.foot.x
            };
            assert!(
                (lateral.abs() - line).abs() < 1e-9,
                "a lamp off the line at {:?}",
                l.foot
            );
            let want_y =
                10.0 + ROAD_LIFT_M + KERB_HEIGHT_M + (line - 7.0 - KERB_WIDTH_M) * PAVEMENT_FALL;
            assert!((l.foot.y - want_y).abs() < 1e-9);
            let along = if l.foot.z.abs() > l.foot.x.abs() {
                l.foot.z
            } else {
                l.foot.x
            };
            assert!(
                along.abs() >= 12.5 - 1e-9,
                "a lamp in the junction at {along}"
            );
            assert_eq!(
                l.colliders.len(),
                1,
                "a lamp post with no collider is a half-kind"
            );
        }
        // The four signal masts and the cables.
        let masts = f
            .pieces
            .iter()
            .filter(|p| p.kind == PieceKind::SignalPost)
            .count();
        assert_eq!(masts, 4);
        let spans = f
            .pieces
            .iter()
            .filter(|p| p.kind == PieceKind::Span)
            .count();
        assert!(spans >= 2 * 8, "spans {spans}");
        // No stop sign at a signalised crossing.
        assert_eq!(
            f.pieces
                .iter()
                .filter(|p| p.kind == PieceKind::Sign)
                .count(),
            0
        );
    }

    /// **A door keeps its threshold clear**: a door on the line refuses the
    /// lattice slot it would have taken. Mutation: dropping `door_near` from
    /// `place` puts a lamp on it.
    #[test]
    fn a_doorway_refuses_the_slot_in_front_of_it() {
        let streets = vec![StreetLine {
            a: DVec2::new(-200.0, 0.0),
            b: DVec2::new(200.0, 0.0),
            gap_m: 20.0,
        }];
        let door = DVec2::new(60.0, furniture_line_m(20.0) + 0.5);
        let f = furnish(&streets, &[], &[door], &flat());
        assert!(f.at_door >= 1);
        for p in &f.pieces {
            if p.kind == PieceKind::Span {
                continue;
            }
            let d = (DVec2::new(p.foot.x, p.foot.z) - door).length();
            assert!(d >= DOOR_CLEAR_M, "{} {d} m from the door", p.kind.name());
        }
    }

    /// **The posts stand at the kerb, the broad pieces at the frontage**
    /// (wave PAR1b.2): on a 20 m city street the post line is half a metre
    /// behind the kerb face (7.5 m, `steal-car/0022`'s offset); on a 16 m town street, whose
    /// parked cars overhang the 5.25 m kerb, it clears a parked car's door
    /// (6.22 m); both are behind the kerb stone; the broad line is behind the
    /// post line. A city street carries bus shelters against the frontage on
    /// both sides and parking meters on the post line; every shelter's span is
    /// clear of the junction and of every lamp. Mutation: `KERB_POST_OFFSET_M`
    /// 0.5 -> 0.0 (posts on the kerb face) reds the first.
    #[test]
    fn posts_stand_at_the_kerb_and_shelters_and_meters_line_a_city_street() {
        assert!(
            (post_line_m(20.0) - 7.5).abs() < 1e-12,
            "{}",
            post_line_m(20.0)
        );
        assert!((post_line_m(16.0) - PARKED_CLEAR_M).abs() < 1e-12);
        for gap in [16.0, 18.0, 20.0, 24.0, 32.0] {
            let post = post_line_m(gap);
            assert!(
                post - 0.15 > kerb_offset_m(gap) + KERB_WIDTH_M,
                "{gap}: on the kerb stone"
            );
            assert!(
                post <= furniture_line_m(gap),
                "{gap}: the posts behind the benches"
            );
        }
        let f = furnish(&grid(), &[], &[], &flat());
        let shelters: Vec<&FurniturePiece> = f
            .pieces
            .iter()
            .filter(|p| p.kind == PieceKind::BusShelter)
            .collect();
        let meters = f
            .pieces
            .iter()
            .filter(|p| p.kind == PieceKind::ParkingMeter)
            .count();
        // 2 streets x 400 m x 2 sides at 240 m: at least one shelter a side.
        assert!(shelters.len() >= 4, "shelters {}", shelters.len());
        assert!(meters >= 8, "meters {meters}");
        for s in &shelters {
            // The PAR1b.2 audit's kerbside shelter: ONE screen a body meets,
            // its roadside face on the post line (half a metre behind the
            // kerb face), its footway face clear of the crowd's walking ring
            // by an agent's radius and the furniture clearance, and nothing
            // of it within a metre of the frontage (no window band, no
            // doorway is ever behind it).
            assert_eq!(s.colliders.len(), 1, "a shelter is one screen");
            let c = &s.colliders[0];
            let (lat, half) = if s.foot.z.abs() > s.foot.x.abs() {
                (c.center.x.abs(), c.half_extents.x)
            } else {
                (c.center.z.abs(), c.half_extents.z)
            };
            let road_face = lat - half;
            assert!(
                (road_face - post_line_m(20.0)).abs() < 1e-9
                    && road_face - kerb_offset_m(20.0) > 0.45,
                "the screen's roadside face is at {road_face}"
            );
            let ring = 10.0 - PAVEMENT_M;
            assert!(
                ring - (lat + half) >= 0.30 + 0.05,
                "the screen stands {} m from the crowd ring",
                ring - (lat + half)
            );
            assert!(
                (kerb_offset_m(20.0) + KERB_WIDTH_M + PAVEMENT_M) - (lat + half) > 1.0,
                "the screen is against the frontage"
            );
            let along = if s.foot.z.abs() > s.foot.x.abs() {
                s.foot.z
            } else {
                s.foot.x
            };
            assert!(
                along.abs() - SHELTER_LEN_M * 0.5 >= 12.5,
                "a shelter at the junction"
            );
            for l in f.pieces.iter().filter(|p| p.kind == PieceKind::LampPost) {
                assert!(
                    (DVec2::new(l.foot.x, l.foot.z) - DVec2::new(s.foot.x, s.foot.z)).length()
                        > SHELTER_LEN_M * 0.5,
                    "a lamp inside a shelter"
                );
            }
        }
    }

    /// **No ground, no furniture.**
    #[test]
    fn groundless_slots_place_nothing() {
        let f = furnish(&grid(), &[], &[], &FnHeight::new(|_, _| None));
        assert!(f.pieces.is_empty());
        assert!(f.groundless > 0);
    }
}
