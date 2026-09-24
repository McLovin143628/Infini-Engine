//! **The roster's body families** (wave VEH3f) -- sixteen more silhouettes in
//! the fractions-of-the-hull convention [`BodyPart`] has used since VEH1a.
//!
//! # Why a sibling module and not more of `vehicle.rs`
//!
//! Only size. The seven families that predate the roster are written out one
//! struct literal at a time in `vehicle.rs`, and sixteen more at that density
//! would add four thousand lines of `BodyPart {` to a file that is already the
//! largest in the crate. The tables here are the same type, read by the same
//! [`VehicleBody::parts`](crate::vehicle::VehicleBody::parts), held to the
//! same arms (`every_body_family_is_a_silhouette_inside_its_own_hull`,
//! `every_authored_part_is_recognised_as_the_kind_it_declares`); they are
//! written through seven `const fn` constructors so that a part's KIND and its
//! NAME cannot disagree about what a hinge is.
//!
//! # What each constructor derives, and what it cannot
//!
//! A door's hinge, a bonnet's and a boot lid's are DERIVED from the part's own
//! box by [`door_hinge`], [`hood_hinge`] and [`trunk_hinge`] -- the VEH3c
//! ruling, so a family that moves a door moves its hinge with it. The side a
//! door opens on is a PARAMETER (`door_left` / `door_right`): a `const fn` may
//! not branch on the sign of a float, and the recogniser that reads it back off
//! the world (`BodyPartKind::of`) is held to the declaration by the arm above.
//!
//! # The SEAT is a part (wave VEH3f)
//!
//! Every family that predates the roster seats its driver at fixed fractions of
//! the hull (`inf_ecs::boarding::SEAT_FRONT_FRAC_Z` and its siblings), which is
//! right for a saloon and wrong for a bus -- whose driver sits over the front
//! axle, eleven metres from where a fraction of the hull would put him. So a
//! roster family DRAWS its seats as [`BodyPartKind::Seat`] cushions, and the
//! driver's cushion gives the seat its PLAN position (`x`, `z`):
//! `boarding::sockets_of` seats the driver there, puts the pedals and the wheel
//! ahead of it by the hull's own fractions (capped in metres), and every
//! seat-reading door -- the rig's `seat_local`, the physics bridge's, the
//! posture pass -- reads the same cushion. The HEIGHTS stay the hull's
//! fractions (every cushion's top face is at `SEAT_CUSHION_FRAC_Y`), because
//! the seat step, the pelvis drop and the foot-well floor are one arithmetic
//! of the hull height, and moving the cushion's height alone was measured to
//! leave a crew-cab driver 0.21 m under it. A family with no seat part keeps
//! the fractions, which is every family that shipped before this wave, byte
//! for byte.
//!
//! The tables themselves are generated from a compact spec (the VEH3f scratch
//! `gen_families.py`) that asserts, before it writes a line, the same three
//! silhouette clauses the arm asserts after: inside the hull, filling it on
//! every axis, under seven of its eight units of volume, with a narrow roof.

use crate::math::Vec3d;
use crate::vehicle::{
    door_hinge, hood_hinge, trunk_hinge, BodyPart, BodyPartKind, PartSide, DOOR_OPEN_DEG,
};

/// A drawn body panel -- a sill, a greenhouse, a track frame, a mast.
const fn panel(name: &'static str, centre: Vec3d, half: Vec3d) -> BodyPart {
    BodyPart {
        name,
        centre,
        half,
        primitive: crate::components::Primitive::Cube,
        kind: BodyPartKind::Panel,
    }
}

/// A door on the near (`-X`) flank, swinging anticlockwise seen from above.
const fn door_left(name: &'static str, centre: Vec3d, half: Vec3d) -> BodyPart {
    BodyPart {
        name,
        centre,
        half,
        primitive: crate::components::Primitive::Cube,
        kind: BodyPartKind::Door {
            hinge: door_hinge(centre, half, DOOR_OPEN_DEG),
            side: PartSide::Left,
        },
    }
}

/// A door on the off (`+X`, the driver's) flank.
const fn door_right(name: &'static str, centre: Vec3d, half: Vec3d) -> BodyPart {
    BodyPart {
        name,
        centre,
        half,
        primitive: crate::components::Primitive::Cube,
        kind: BodyPartKind::Door {
            hinge: door_hinge(centre, half, -DOOR_OPEN_DEG),
            side: PartSide::Right,
        },
    }
}

/// A bonnet (or an engine hatch), hinged at its rear edge.
const fn hood(name: &'static str, centre: Vec3d, half: Vec3d) -> BodyPart {
    BodyPart {
        name,
        centre,
        half,
        primitive: crate::components::Primitive::Cube,
        kind: BodyPartKind::Hood {
            hinge: hood_hinge(centre, half),
        },
    }
}

/// A boot lid, a tailgate or a rear door set, hinged at its forward edge.
const fn trunk(name: &'static str, centre: Vec3d, half: Vec3d) -> BodyPart {
    BodyPart {
        name,
        centre,
        half,
        primitive: crate::components::Primitive::Cube,
        kind: BodyPartKind::Trunk {
            hinge: trunk_hinge(centre, half),
        },
    }
}

/// A bumper.
const fn bumper(name: &'static str, centre: Vec3d, half: Vec3d) -> BodyPart {
    BodyPart {
        name,
        centre,
        half,
        primitive: crate::components::Primitive::Cube,
        kind: BodyPartKind::Bumper,
    }
}

/// A pane of glass.
const fn glass(name: &'static str, centre: Vec3d, half: Vec3d) -> BodyPart {
    BodyPart {
        name,
        centre,
        half,
        primitive: crate::components::Primitive::Cube,
        kind: BodyPartKind::Glass,
    }
}

/// A seat cushion -- its TOP face's centre is the H-point `sockets_of` seats a
/// body on.
const fn seat(name: &'static str, centre: Vec3d, half: Vec3d) -> BodyPart {
    BodyPart {
        name,
        centre,
        half,
        primitive: crate::components::Primitive::Cube,
        kind: BodyPartKind::Seat,
    }
}

/// A long-bonnet two-door grand tourer: a greenhouse set well back, LONG doors (the VEH3d inner-handle rule reads a door this long from the seat, not from its centre) and a short boot.
pub(crate) const COUPE_PARTS: &[BodyPart] = &[
    panel(
        "lower",
        Vec3d::new(0.0, -0.5, 0.0),
        Vec3d::new(1.0, 0.5, 1.0),
    ),
    panel(
        "cabin",
        Vec3d::new(0.0, 0.5, -0.18),
        Vec3d::new(0.82, 0.5, 0.38),
    ),
    hood(
        "bonnet",
        Vec3d::new(0.0, 0.1, 0.6),
        Vec3d::new(0.95, 0.1, 0.38),
    ),
    trunk(
        "boot",
        Vec3d::new(0.0, 0.12, -0.76),
        Vec3d::new(0.94, 0.12, 0.22),
    ),
    door_left(
        "door_l",
        Vec3d::new(-0.955, -0.4, 0.0),
        Vec3d::new(0.045, 0.36, 0.42),
    ),
    door_right(
        "door_r",
        Vec3d::new(0.955, -0.4, 0.0),
        Vec3d::new(0.045, 0.36, 0.42),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.5, 0.985),
        Vec3d::new(0.96, 0.15, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.5, -0.985),
        Vec3d::new(0.96, 0.15, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.5, 0.22),
        Vec3d::new(0.76, 0.36, 0.04),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.5, -0.54),
        Vec3d::new(0.76, 0.3, 0.04),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.84, 0.5, -0.18),
        Vec3d::new(0.04, 0.32, 0.34),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.84, 0.5, -0.18),
        Vec3d::new(0.04, 0.32, 0.34),
    ),
    seat(
        "seat_l",
        Vec3d::new(-0.42, -0.59, -0.08),
        Vec3d::new(0.3, 0.14, 0.16),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.42, -0.59, -0.08),
        Vec3d::new(0.3, 0.14, 0.16),
    ),
];

/// A short five-door hatchback: the greenhouse runs to the tail and the tailgate IS the rear of the car.
pub(crate) const HATCH_PARTS: &[BodyPart] = &[
    panel(
        "lower",
        Vec3d::new(0.0, -0.5, 0.0),
        Vec3d::new(1.0, 0.5, 1.0),
    ),
    panel(
        "cabin",
        Vec3d::new(0.0, 0.5, -0.2),
        Vec3d::new(0.86, 0.5, 0.56),
    ),
    hood(
        "bonnet",
        Vec3d::new(0.0, 0.12, 0.66),
        Vec3d::new(0.94, 0.12, 0.32),
    ),
    trunk(
        "tailgate",
        Vec3d::new(0.0, 0.3, -0.97),
        Vec3d::new(0.84, 0.4, 0.03),
    ),
    door_left(
        "door_fl",
        Vec3d::new(-0.955, -0.42, 0.2),
        Vec3d::new(0.045, 0.34, 0.24),
    ),
    door_right(
        "door_fr",
        Vec3d::new(0.955, -0.42, 0.2),
        Vec3d::new(0.045, 0.34, 0.24),
    ),
    door_left(
        "door_rl",
        Vec3d::new(-0.955, -0.42, -0.32),
        Vec3d::new(0.045, 0.32, 0.22),
    ),
    door_right(
        "door_rr",
        Vec3d::new(0.955, -0.42, -0.32),
        Vec3d::new(0.045, 0.32, 0.22),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.52, 0.985),
        Vec3d::new(0.96, 0.15, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.52, -0.985),
        Vec3d::new(0.96, 0.15, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.5, 0.32),
        Vec3d::new(0.78, 0.34, 0.04),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.56, -0.9),
        Vec3d::new(0.76, 0.26, 0.03),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.9, 0.5, -0.2),
        Vec3d::new(0.04, 0.3, 0.5),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.9, 0.5, -0.2),
        Vec3d::new(0.04, 0.3, 0.5),
    ),
    seat(
        "seat_l",
        Vec3d::new(-0.42, -0.59, 0.02),
        Vec3d::new(0.3, 0.14, 0.16),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.42, -0.59, 0.02),
        Vec3d::new(0.3, 0.14, 0.16),
    ),
];

/// An estate: a long flat roof with rails over a load bay, a tailgate, four doors.
pub(crate) const WAGON_PARTS: &[BodyPart] = &[
    panel(
        "lower",
        Vec3d::new(0.0, -0.5, 0.0),
        Vec3d::new(1.0, 0.5, 1.0),
    ),
    panel(
        "cabin",
        Vec3d::new(0.0, 0.44, -0.28),
        Vec3d::new(0.86, 0.44, 0.58),
    ),
    hood(
        "bonnet",
        Vec3d::new(0.0, 0.12, 0.64),
        Vec3d::new(0.94, 0.12, 0.34),
    ),
    trunk(
        "tailgate",
        Vec3d::new(0.0, 0.2, -0.97),
        Vec3d::new(0.86, 0.32, 0.03),
    ),
    panel(
        "rail_left",
        Vec3d::new(-0.72, 0.93, -0.28),
        Vec3d::new(0.06, 0.05, 0.5),
    ),
    panel(
        "rail_right",
        Vec3d::new(0.72, 0.93, -0.28),
        Vec3d::new(0.06, 0.05, 0.5),
    ),
    door_left(
        "door_fl",
        Vec3d::new(-0.955, -0.42, 0.18),
        Vec3d::new(0.045, 0.34, 0.25),
    ),
    door_right(
        "door_fr",
        Vec3d::new(0.955, -0.42, 0.18),
        Vec3d::new(0.045, 0.34, 0.25),
    ),
    door_left(
        "door_rl",
        Vec3d::new(-0.955, -0.42, -0.34),
        Vec3d::new(0.045, 0.33, 0.24),
    ),
    door_right(
        "door_rr",
        Vec3d::new(0.955, -0.42, -0.34),
        Vec3d::new(0.045, 0.33, 0.24),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.52, 0.985),
        Vec3d::new(0.96, 0.15, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.52, -0.985),
        Vec3d::new(0.96, 0.15, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.46, 0.28),
        Vec3d::new(0.8, 0.32, 0.04),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.46, -0.88),
        Vec3d::new(0.8, 0.28, 0.03),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.9, 0.46, -0.28),
        Vec3d::new(0.04, 0.3, 0.54),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.9, 0.46, -0.28),
        Vec3d::new(0.04, 0.3, 0.54),
    ),
    seat(
        "seat_l",
        Vec3d::new(-0.42, -0.59, 0.0),
        Vec3d::new(0.3, 0.14, 0.16),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.42, -0.59, 0.0),
        Vec3d::new(0.3, 0.14, 0.16),
    ),
];

/// A crew-cab pickup: four doors on a cab that stops at the bed, a short bonnet, an open bed with sides and a tailgate.
pub(crate) const PICKUP4_PARTS: &[BodyPart] = &[
    panel(
        "lower",
        Vec3d::new(0.0, -0.6, 0.0),
        Vec3d::new(1.0, 0.4, 1.0),
    ),
    panel(
        "cab",
        Vec3d::new(0.0, 0.45, 0.2),
        Vec3d::new(0.94, 0.55, 0.4),
    ),
    hood(
        "bonnet",
        Vec3d::new(0.0, 0.1, 0.8),
        Vec3d::new(0.95, 0.14, 0.19),
    ),
    panel(
        "bed",
        Vec3d::new(0.0, -0.1, -0.58),
        Vec3d::new(0.96, 0.1, 0.4),
    ),
    panel(
        "bed_left",
        Vec3d::new(-0.88, 0.15, -0.58),
        Vec3d::new(0.09, 0.35, 0.4),
    ),
    panel(
        "bed_right",
        Vec3d::new(0.88, 0.15, -0.58),
        Vec3d::new(0.09, 0.35, 0.4),
    ),
    trunk(
        "tailgate",
        Vec3d::new(0.0, 0.15, -0.97),
        Vec3d::new(0.92, 0.3, 0.03),
    ),
    door_left(
        "door_fl",
        Vec3d::new(-0.955, 0.05, 0.38),
        Vec3d::new(0.045, 0.42, 0.2),
    ),
    door_right(
        "door_fr",
        Vec3d::new(0.955, 0.05, 0.38),
        Vec3d::new(0.045, 0.42, 0.2),
    ),
    door_left(
        "door_rl",
        Vec3d::new(-0.955, 0.05, -0.02),
        Vec3d::new(0.045, 0.42, 0.18),
    ),
    door_right(
        "door_rr",
        Vec3d::new(0.955, 0.05, -0.02),
        Vec3d::new(0.045, 0.42, 0.18),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.72, 0.985),
        Vec3d::new(0.96, 0.16, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.72, -0.985),
        Vec3d::new(0.96, 0.16, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.5, 0.58),
        Vec3d::new(0.86, 0.38, 0.035),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.5, -0.18),
        Vec3d::new(0.86, 0.36, 0.035),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.9, 0.5, 0.2),
        Vec3d::new(0.04, 0.36, 0.36),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.9, 0.5, 0.2),
        Vec3d::new(0.04, 0.36, 0.36),
    ),
    seat(
        "seat_l",
        Vec3d::new(-0.42, -0.57, 0.34),
        Vec3d::new(0.3, 0.12, 0.14),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.42, -0.57, 0.34),
        Vec3d::new(0.3, 0.12, 0.14),
    ),
];

/// A short upright off-roader: a boxy greenhouse behind an upright screen, a flat bonnet, a snorkel, chunky bumpers.
pub(crate) const JEEP_PARTS: &[BodyPart] = &[
    panel(
        "lower",
        Vec3d::new(0.0, -0.55, 0.0),
        Vec3d::new(1.0, 0.45, 1.0),
    ),
    panel(
        "cabin",
        Vec3d::new(0.0, 0.45, -0.25),
        Vec3d::new(0.9, 0.55, 0.55),
    ),
    hood(
        "bonnet",
        Vec3d::new(0.0, 0.1, 0.64),
        Vec3d::new(0.9, 0.12, 0.34),
    ),
    trunk(
        "tailgate",
        Vec3d::new(0.0, 0.1, -0.96),
        Vec3d::new(0.8, 0.4, 0.03),
    ),
    panel(
        "snorkel",
        Vec3d::new(0.85, 0.52, 0.34),
        Vec3d::new(0.05, 0.36, 0.04),
    ),
    door_left(
        "door_fl",
        Vec3d::new(-0.955, -0.2, 0.12),
        Vec3d::new(0.045, 0.36, 0.2),
    ),
    door_right(
        "door_fr",
        Vec3d::new(0.955, -0.2, 0.12),
        Vec3d::new(0.045, 0.36, 0.2),
    ),
    door_left(
        "door_rl",
        Vec3d::new(-0.955, -0.2, -0.3),
        Vec3d::new(0.045, 0.36, 0.2),
    ),
    door_right(
        "door_rr",
        Vec3d::new(0.955, -0.2, -0.3),
        Vec3d::new(0.045, 0.36, 0.2),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.62, 0.97),
        Vec3d::new(1.0, 0.2, 0.03),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.62, -0.97),
        Vec3d::new(1.0, 0.2, 0.03),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.55, 0.32),
        Vec3d::new(0.84, 0.3, 0.03),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.55, -0.82),
        Vec3d::new(0.8, 0.3, 0.03),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.92, 0.55, -0.25),
        Vec3d::new(0.04, 0.3, 0.5),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.92, 0.55, -0.25),
        Vec3d::new(0.04, 0.3, 0.5),
    ),
    seat(
        "seat_l",
        Vec3d::new(-0.42, -0.57, 0.02),
        Vec3d::new(0.3, 0.12, 0.16),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.42, -0.57, 0.02),
        Vec3d::new(0.3, 0.12, 0.16),
    ),
];

/// A wide military-derived four-door: slit windows over a tall body, a broad flat bonnet, a wide track.
pub(crate) const HUMMER_PARTS: &[BodyPart] = &[
    panel(
        "lower",
        Vec3d::new(0.0, -0.5, 0.0),
        Vec3d::new(1.0, 0.5, 1.0),
    ),
    panel(
        "cabin",
        Vec3d::new(0.0, 0.5, -0.2),
        Vec3d::new(0.92, 0.5, 0.5),
    ),
    hood(
        "bonnet",
        Vec3d::new(0.0, 0.2, 0.66),
        Vec3d::new(0.96, 0.14, 0.32),
    ),
    trunk(
        "tailgate",
        Vec3d::new(0.0, 0.2, -0.97),
        Vec3d::new(0.86, 0.3, 0.03),
    ),
    door_left(
        "door_fl",
        Vec3d::new(-0.955, -0.1, 0.18),
        Vec3d::new(0.045, 0.4, 0.22),
    ),
    door_right(
        "door_fr",
        Vec3d::new(0.955, -0.1, 0.18),
        Vec3d::new(0.045, 0.4, 0.22),
    ),
    door_left(
        "door_rl",
        Vec3d::new(-0.955, -0.1, -0.32),
        Vec3d::new(0.045, 0.4, 0.22),
    ),
    door_right(
        "door_rr",
        Vec3d::new(0.955, -0.1, -0.32),
        Vec3d::new(0.045, 0.4, 0.22),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.72, 0.98),
        Vec3d::new(1.0, 0.16, 0.02),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.72, -0.98),
        Vec3d::new(1.0, 0.16, 0.02),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.62, 0.37),
        Vec3d::new(0.86, 0.2, 0.03),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.62, -0.77),
        Vec3d::new(0.8, 0.18, 0.03),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.93, 0.62, -0.2),
        Vec3d::new(0.04, 0.2, 0.5),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.93, 0.62, -0.2),
        Vec3d::new(0.04, 0.2, 0.5),
    ),
    seat(
        "seat_l",
        Vec3d::new(-0.42, -0.57, 0.04),
        Vec3d::new(0.3, 0.12, 0.16),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.42, -0.57, 0.04),
        Vec3d::new(0.3, 0.12, 0.16),
    ),
];

/// A single-deck transit bus: a box body on a skirt, a big screen, a FOLDING DOOR forward on the driver's flank (the VEH3d inner-handle rule reads the seat, and this family's seat is at the front), a second door amidships, an engine hatch at the rear and an air-conditioning pod on the roof.
pub(crate) const BUS_PARTS: &[BodyPart] = &[
    panel(
        "skirt",
        Vec3d::new(0.0, -0.8, 0.0),
        Vec3d::new(1.0, 0.2, 0.8),
    ),
    panel(
        "body",
        Vec3d::new(0.0, 0.1, 0.0),
        Vec3d::new(0.97, 0.7, 0.97),
    ),
    panel(
        "roof_unit",
        Vec3d::new(0.0, 0.9, -0.2),
        Vec3d::new(0.6, 0.1, 0.25),
    ),
    trunk(
        "boot_engine",
        Vec3d::new(0.0, -0.35, -0.985),
        Vec3d::new(0.7, 0.25, 0.015),
    ),
    door_right(
        "door_front",
        Vec3d::new(0.975, -0.1, 0.84),
        Vec3d::new(0.025, 0.5, 0.1),
    ),
    door_right(
        "door_rear",
        Vec3d::new(0.975, -0.1, -0.3),
        Vec3d::new(0.025, 0.5, 0.1),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.72, 0.985),
        Vec3d::new(0.96, 0.14, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.72, -0.985),
        Vec3d::new(0.96, 0.14, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.3, 0.975),
        Vec3d::new(0.9, 0.4, 0.02),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.4, -0.975),
        Vec3d::new(0.85, 0.28, 0.02),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.975, 0.36, 0.0),
        Vec3d::new(0.02, 0.24, 0.8),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.975, 0.36, -0.12),
        Vec3d::new(0.02, 0.24, 0.66),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.5, -0.53, 0.8),
        Vec3d::new(0.22, 0.08, 0.08),
    ),
];

/// A conventional long-nose semi tractor: a frame, a tall cab over a sleeper, a long bonnet, the FIFTH WHEEL at the rear (where a trailer's kingpin hitches), saddle tanks, exhaust stacks and a HIGH STEP up to the cab.
pub(crate) const SEMI_TRACTOR_PARTS: &[BodyPart] = &[
    panel(
        "frame",
        Vec3d::new(0.0, -0.62, -0.1),
        Vec3d::new(0.5, 0.1, 0.9),
    ),
    panel(
        "cab",
        Vec3d::new(0.0, 0.25, 0.1),
        Vec3d::new(0.94, 0.5, 0.24),
    ),
    panel(
        "sleeper",
        Vec3d::new(0.0, 0.3, -0.3),
        Vec3d::new(0.94, 0.62, 0.16),
    ),
    hood(
        "bonnet",
        Vec3d::new(0.0, -0.05, 0.64),
        Vec3d::new(0.8, 0.28, 0.34),
    ),
    panel(
        "fifth_wheel",
        Vec3d::new(0.0, -0.45, -0.7),
        Vec3d::new(0.4, 0.04, 0.14),
    ),
    panel(
        "fender_l",
        Vec3d::new(-0.85, -0.55, 0.62),
        Vec3d::new(0.14, 0.12, 0.2),
    ),
    panel(
        "fender_r",
        Vec3d::new(0.85, -0.55, 0.62),
        Vec3d::new(0.14, 0.12, 0.2),
    ),
    panel(
        "tank_l",
        Vec3d::new(-0.78, -0.6, -0.05),
        Vec3d::new(0.16, 0.14, 0.18),
    ),
    panel(
        "tank_r",
        Vec3d::new(0.78, -0.6, -0.05),
        Vec3d::new(0.16, 0.14, 0.18),
    ),
    panel(
        "stack_l",
        Vec3d::new(-0.88, 0.45, -0.12),
        Vec3d::new(0.05, 0.55, 0.03),
    ),
    panel(
        "stack_r",
        Vec3d::new(0.88, 0.45, -0.12),
        Vec3d::new(0.05, 0.55, 0.03),
    ),
    panel(
        "step_l",
        Vec3d::new(-0.9, -0.9, 0.12),
        Vec3d::new(0.1, 0.08, 0.12),
    ),
    panel(
        "step_r",
        Vec3d::new(0.9, -0.9, 0.12),
        Vec3d::new(0.1, 0.08, 0.12),
    ),
    door_left(
        "door_l",
        Vec3d::new(-0.955, 0.1, 0.1),
        Vec3d::new(0.045, 0.4, 0.22),
    ),
    door_right(
        "door_r",
        Vec3d::new(0.955, 0.1, 0.1),
        Vec3d::new(0.045, 0.4, 0.22),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.62, 0.985),
        Vec3d::new(0.9, 0.12, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.62, -0.985),
        Vec3d::new(0.6, 0.08, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.52, 0.33),
        Vec3d::new(0.86, 0.2, 0.02),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.55, -0.47),
        Vec3d::new(0.7, 0.2, 0.02),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.93, 0.5, 0.1),
        Vec3d::new(0.02, 0.18, 0.2),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.93, 0.5, 0.1),
        Vec3d::new(0.02, 0.18, 0.2),
    ),
    seat(
        "seat_l",
        Vec3d::new(-0.45, -0.51, 0.12),
        Vec3d::new(0.22, 0.06, 0.1),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.45, -0.51, 0.12),
        Vec3d::new(0.22, 0.06, 0.1),
    ),
];

/// A box semi-trailer, the first articulated rig: a kingpin at the front (hitched to a tractor's fifth wheel by a spherical joint), a tandem axle at the rear, landing legs, roller doors at the back and an underride guard. It has no seat and no side door, because nobody boards a trailer.
pub(crate) const TRAILER_PARTS: &[BodyPart] = &[
    panel(
        "box",
        Vec3d::new(0.0, 0.15, 0.0),
        Vec3d::new(0.98, 0.83, 0.97),
    ),
    panel(
        "nose",
        Vec3d::new(0.0, 0.14, 0.985),
        Vec3d::new(0.9, 0.8, 0.015),
    ),
    panel(
        "frame",
        Vec3d::new(0.0, -0.8, 0.0),
        Vec3d::new(0.4, 0.1, 0.95),
    ),
    panel(
        "landing_gear",
        Vec3d::new(0.0, -0.85, 0.55),
        Vec3d::new(0.5, 0.13, 0.03),
    ),
    panel(
        "roof_vent",
        Vec3d::new(0.0, 0.99, 0.3),
        Vec3d::new(0.3, 0.01, 0.2),
    ),
    trunk(
        "tailgate",
        Vec3d::new(0.0, 0.14, -0.985),
        Vec3d::new(0.94, 0.8, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.8, -0.985),
        Vec3d::new(0.9, 0.06, 0.015),
    ),
];

/// A medium-duty flatbed: a cab-over-axle cab, a headboard, a flat deck with stake rails, a beacon, a cab step.
pub(crate) const FLATBED_PARTS: &[BodyPart] = &[
    panel(
        "frame",
        Vec3d::new(0.0, -0.72, 0.0),
        Vec3d::new(0.5, 0.1, 0.98),
    ),
    panel(
        "cab",
        Vec3d::new(0.0, 0.3, 0.7),
        Vec3d::new(0.94, 0.62, 0.26),
    ),
    panel(
        "deck",
        Vec3d::new(0.0, -0.45, -0.28),
        Vec3d::new(1.0, 0.07, 0.7),
    ),
    panel(
        "headboard",
        Vec3d::new(0.0, 0.0, 0.4),
        Vec3d::new(0.94, 0.38, 0.03),
    ),
    panel(
        "stake_l",
        Vec3d::new(-0.96, -0.25, -0.28),
        Vec3d::new(0.03, 0.15, 0.7),
    ),
    panel(
        "stake_r",
        Vec3d::new(0.96, -0.25, -0.28),
        Vec3d::new(0.03, 0.15, 0.7),
    ),
    panel(
        "beacon",
        Vec3d::new(0.0, 0.97, 0.7),
        Vec3d::new(0.3, 0.03, 0.05),
    ),
    panel(
        "step_l",
        Vec3d::new(-0.9, -0.9, 0.66),
        Vec3d::new(0.08, 0.08, 0.1),
    ),
    panel(
        "step_r",
        Vec3d::new(0.9, -0.9, 0.66),
        Vec3d::new(0.08, 0.08, 0.1),
    ),
    door_left(
        "door_l",
        Vec3d::new(-0.955, 0.2, 0.72),
        Vec3d::new(0.045, 0.45, 0.2),
    ),
    door_right(
        "door_r",
        Vec3d::new(0.955, 0.2, 0.72),
        Vec3d::new(0.045, 0.45, 0.2),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.75, 0.985),
        Vec3d::new(0.95, 0.12, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.72, -0.985),
        Vec3d::new(0.9, 0.1, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.55, 0.965),
        Vec3d::new(0.86, 0.28, 0.02),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.93, 0.55, 0.72),
        Vec3d::new(0.04, 0.22, 0.18),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.93, 0.55, 0.72),
        Vec3d::new(0.04, 0.22, 0.18),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.45, -0.5, 0.72),
        Vec3d::new(0.22, 0.05, 0.1),
    ),
];

/// A rigid tank truck, and the mixer's silhouette: a cab ahead of a TANK drawn as two crossed boxes (a lying cylinder needs a part rotation this table does not carry), a walkway and a beacon.
pub(crate) const TANKER_PARTS: &[BodyPart] = &[
    panel(
        "frame",
        Vec3d::new(0.0, -0.72, 0.0),
        Vec3d::new(0.5, 0.1, 0.98),
    ),
    panel(
        "cab",
        Vec3d::new(0.0, 0.25, 0.72),
        Vec3d::new(0.94, 0.6, 0.24),
    ),
    panel(
        "tank_core",
        Vec3d::new(0.0, 0.12, -0.28),
        Vec3d::new(0.8, 0.62, 0.68),
    ),
    panel(
        "tank_wide",
        Vec3d::new(0.0, 0.12, -0.28),
        Vec3d::new(0.96, 0.42, 0.68),
    ),
    panel(
        "walkway",
        Vec3d::new(0.0, 0.8, -0.28),
        Vec3d::new(0.2, 0.06, 0.5),
    ),
    panel(
        "beacon",
        Vec3d::new(0.0, 0.94, 0.72),
        Vec3d::new(0.3, 0.05, 0.05),
    ),
    panel(
        "step_l",
        Vec3d::new(-0.9, -0.9, 0.66),
        Vec3d::new(0.08, 0.08, 0.1),
    ),
    panel(
        "step_r",
        Vec3d::new(0.9, -0.9, 0.66),
        Vec3d::new(0.08, 0.08, 0.1),
    ),
    door_left(
        "door_l",
        Vec3d::new(-0.955, 0.15, 0.72),
        Vec3d::new(0.045, 0.45, 0.2),
    ),
    door_right(
        "door_r",
        Vec3d::new(0.955, 0.15, 0.72),
        Vec3d::new(0.045, 0.45, 0.2),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.75, 0.985),
        Vec3d::new(0.95, 0.12, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.72, -0.985),
        Vec3d::new(0.9, 0.1, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.5, 0.965),
        Vec3d::new(0.86, 0.28, 0.02),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.93, 0.5, 0.72),
        Vec3d::new(0.04, 0.22, 0.18),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.93, 0.5, 0.72),
        Vec3d::new(0.04, 0.22, 0.18),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.45, -0.5, 0.72),
        Vec3d::new(0.22, 0.05, 0.1),
    ),
];

/// An eight-wheeled armoured personnel carrier on four drawn wheels: a slab hull, a sloped glacis, a turret with a gun, a roof hatch, troop doors, vision blocks.
pub(crate) const APC_PARTS: &[BodyPart] = &[
    panel(
        "hull_lower",
        Vec3d::new(0.0, -0.5, 0.0),
        Vec3d::new(1.0, 0.48, 1.0),
    ),
    panel(
        "hull_upper",
        Vec3d::new(0.0, 0.3, -0.05),
        Vec3d::new(0.9, 0.32, 0.88),
    ),
    panel(
        "glacis",
        Vec3d::new(0.0, 0.2, 0.92),
        Vec3d::new(0.9, 0.2, 0.07),
    ),
    panel(
        "turret",
        Vec3d::new(0.0, 0.78, 0.1),
        Vec3d::new(0.35, 0.2, 0.3),
    ),
    panel(
        "gun",
        Vec3d::new(0.0, 0.82, 0.6),
        Vec3d::new(0.04, 0.04, 0.3),
    ),
    hood(
        "hood_hatch",
        Vec3d::new(0.0, 0.64, -0.4),
        Vec3d::new(0.4, 0.02, 0.2),
    ),
    door_left(
        "door_l",
        Vec3d::new(-0.955, -0.1, -0.35),
        Vec3d::new(0.045, 0.3, 0.18),
    ),
    door_right(
        "door_r",
        Vec3d::new(0.955, -0.1, -0.35),
        Vec3d::new(0.045, 0.3, 0.18),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.6, 0.985),
        Vec3d::new(0.9, 0.14, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.6, -0.985),
        Vec3d::new(0.9, 0.14, 0.015),
    ),
    glass(
        "glass_vision",
        Vec3d::new(0.0, 0.45, 0.72),
        Vec3d::new(0.5, 0.06, 0.02),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.9, 0.4, 0.3),
        Vec3d::new(0.02, 0.05, 0.12),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.9, 0.4, 0.3),
        Vec3d::new(0.02, 0.05, 0.12),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.4, -0.51, 0.55),
        Vec3d::new(0.22, 0.06, 0.1),
    ),
];

/// A TRACKED bulldozer: two track frames (the rig's wheel row runs inside them, and it steers by SKID rather than by a rack), an engine housing, a ROPS cab, a front blade and a rear ripper. The operator's seat is on the centreline.
pub(crate) const DOZER_PARTS: &[BodyPart] = &[
    panel(
        "track_l",
        Vec3d::new(-0.78, -0.62, 0.0),
        Vec3d::new(0.22, 0.36, 0.9),
    ),
    panel(
        "track_r",
        Vec3d::new(0.78, -0.62, 0.0),
        Vec3d::new(0.22, 0.36, 0.9),
    ),
    panel(
        "housing",
        Vec3d::new(0.0, -0.2, 0.35),
        Vec3d::new(0.52, 0.36, 0.5),
    ),
    panel(
        "cab",
        Vec3d::new(0.0, 0.45, -0.35),
        Vec3d::new(0.5, 0.53, 0.32),
    ),
    panel(
        "blade",
        Vec3d::new(0.0, -0.5, 0.96),
        Vec3d::new(0.98, 0.42, 0.04),
    ),
    panel(
        "ripper",
        Vec3d::new(0.0, -0.6, -0.9),
        Vec3d::new(0.1, 0.3, 0.08),
    ),
    hood(
        "hood_engine",
        Vec3d::new(0.0, 0.18, 0.35),
        Vec3d::new(0.5, 0.02, 0.48),
    ),
    door_left(
        "door_l",
        Vec3d::new(-0.53, 0.35, -0.35),
        Vec3d::new(0.03, 0.35, 0.18),
    ),
    door_right(
        "door_r",
        Vec3d::new(0.53, 0.35, -0.35),
        Vec3d::new(0.03, 0.35, 0.18),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.55, -0.03),
        Vec3d::new(0.44, 0.3, 0.02),
    ),
    glass(
        "glass_rear",
        Vec3d::new(0.0, 0.55, -0.67),
        Vec3d::new(0.44, 0.3, 0.02),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.51, 0.62, -0.35),
        Vec3d::new(0.02, 0.2, 0.28),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.51, 0.62, -0.35),
        Vec3d::new(0.02, 0.2, 0.28),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.0, -0.5, -0.4),
        Vec3d::new(0.2, 0.05, 0.12),
    ),
];

/// A counterbalanced forklift: a body, a heavy counterweight at the tail, an overhead guard on four posts, a MAST with a carriage and two forks at the nose, and no doors. The seat is on the centreline under the guard.
pub(crate) const FORKLIFT_PARTS: &[BodyPart] = &[
    panel(
        "body",
        Vec3d::new(0.0, -0.45, -0.1),
        Vec3d::new(0.9, 0.4, 0.6),
    ),
    panel(
        "counterweight",
        Vec3d::new(0.0, -0.35, -0.84),
        Vec3d::new(0.95, 0.5, 0.16),
    ),
    panel(
        "guard_roof",
        Vec3d::new(0.0, 0.9, -0.1),
        Vec3d::new(0.8, 0.04, 0.45),
    ),
    panel(
        "post_fl",
        Vec3d::new(-0.72, 0.45, 0.3),
        Vec3d::new(0.03, 0.45, 0.03),
    ),
    panel(
        "post_fr",
        Vec3d::new(0.72, 0.45, 0.3),
        Vec3d::new(0.03, 0.45, 0.03),
    ),
    panel(
        "post_rl",
        Vec3d::new(-0.72, 0.45, -0.5),
        Vec3d::new(0.03, 0.45, 0.03),
    ),
    panel(
        "post_rr",
        Vec3d::new(0.72, 0.45, -0.5),
        Vec3d::new(0.03, 0.45, 0.03),
    ),
    panel(
        "mast_l",
        Vec3d::new(-0.55, 0.1, 0.72),
        Vec3d::new(0.06, 0.9, 0.05),
    ),
    panel(
        "mast_r",
        Vec3d::new(0.55, 0.1, 0.72),
        Vec3d::new(0.06, 0.9, 0.05),
    ),
    panel(
        "carriage",
        Vec3d::new(0.0, -0.3, 0.8),
        Vec3d::new(0.6, 0.3, 0.03),
    ),
    panel(
        "fork_l",
        Vec3d::new(-0.35, -0.95, 0.9),
        Vec3d::new(0.08, 0.03, 0.1),
    ),
    panel(
        "fork_r",
        Vec3d::new(0.35, -0.95, 0.9),
        Vec3d::new(0.08, 0.03, 0.1),
    ),
    panel(
        "fender_l",
        Vec3d::new(-0.95, -0.6, 0.3),
        Vec3d::new(0.05, 0.2, 0.18),
    ),
    panel(
        "fender_r",
        Vec3d::new(0.95, -0.6, 0.3),
        Vec3d::new(0.05, 0.2, 0.18),
    ),
    hood(
        "hood_engine",
        Vec3d::new(0.0, -0.03, -0.4),
        Vec3d::new(0.8, 0.02, 0.28),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.0, -0.51, -0.1),
        Vec3d::new(0.35, 0.06, 0.18),
    ),
];

/// A wrecker: a short-bonnet cab, a tool body, a BOOM (a post and an arm reaching aft) and a hook over the tail.
pub(crate) const TOW_TRUCK_PARTS: &[BodyPart] = &[
    panel(
        "frame",
        Vec3d::new(0.0, -0.72, 0.0),
        Vec3d::new(0.5, 0.1, 0.98),
    ),
    panel(
        "cab",
        Vec3d::new(0.0, 0.25, 0.5),
        Vec3d::new(0.94, 0.58, 0.26),
    ),
    hood(
        "bonnet",
        Vec3d::new(0.0, -0.02, 0.88),
        Vec3d::new(0.9, 0.26, 0.1),
    ),
    panel(
        "body",
        Vec3d::new(0.0, -0.2, -0.4),
        Vec3d::new(0.96, 0.35, 0.5),
    ),
    panel(
        "boom_post",
        Vec3d::new(0.0, 0.3, -0.2),
        Vec3d::new(0.12, 0.62, 0.08),
    ),
    panel(
        "boom_arm",
        Vec3d::new(0.0, 0.9, -0.55),
        Vec3d::new(0.08, 0.08, 0.4),
    ),
    panel(
        "hook",
        Vec3d::new(0.0, 0.45, -0.93),
        Vec3d::new(0.03, 0.4, 0.03),
    ),
    panel(
        "step_l",
        Vec3d::new(-0.9, -0.9, 0.46),
        Vec3d::new(0.08, 0.08, 0.1),
    ),
    panel(
        "step_r",
        Vec3d::new(0.9, -0.9, 0.46),
        Vec3d::new(0.08, 0.08, 0.1),
    ),
    door_left(
        "door_l",
        Vec3d::new(-0.955, 0.15, 0.5),
        Vec3d::new(0.045, 0.45, 0.2),
    ),
    door_right(
        "door_r",
        Vec3d::new(0.955, 0.15, 0.5),
        Vec3d::new(0.045, 0.45, 0.2),
    ),
    bumper(
        "bumper_front",
        Vec3d::new(0.0, -0.75, 0.985),
        Vec3d::new(0.95, 0.12, 0.015),
    ),
    bumper(
        "bumper_rear",
        Vec3d::new(0.0, -0.72, -0.985),
        Vec3d::new(0.9, 0.1, 0.015),
    ),
    glass(
        "glass_windscreen",
        Vec3d::new(0.0, 0.5, 0.765),
        Vec3d::new(0.86, 0.25, 0.02),
    ),
    glass(
        "glass_side_l",
        Vec3d::new(-0.93, 0.5, 0.5),
        Vec3d::new(0.04, 0.22, 0.18),
    ),
    glass(
        "glass_side_r",
        Vec3d::new(0.93, 0.5, 0.5),
        Vec3d::new(0.04, 0.22, 0.18),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.45, -0.5, 0.5),
        Vec3d::new(0.22, 0.05, 0.1),
    ),
];

/// A light single-engine aeroplane, DORMANT until VEH3g: a fuselage, a high wing, a tailplane and fin, a spinner and a tricycle gear drawn over the rig's four wheels. It rolls on its gear; lift, stall and control surfaces are VEH3g's.
pub(crate) const AIRCRAFT_PARTS: &[BodyPart] = &[
    panel(
        "fuselage",
        Vec3d::new(0.0, -0.1, 0.05),
        Vec3d::new(0.12, 0.4, 0.95),
    ),
    panel(
        "wing",
        Vec3d::new(0.0, 0.25, 0.15),
        Vec3d::new(1.0, 0.04, 0.2),
    ),
    panel(
        "tailplane",
        Vec3d::new(0.0, 0.15, -0.88),
        Vec3d::new(0.35, 0.03, 0.1),
    ),
    panel(
        "fin",
        Vec3d::new(0.0, 0.58, -0.88),
        Vec3d::new(0.02, 0.4, 0.1),
    ),
    panel(
        "spinner",
        Vec3d::new(0.0, -0.1, 0.99),
        Vec3d::new(0.05, 0.05, 0.01),
    ),
    panel(
        "gear_l",
        Vec3d::new(-0.3, -0.7, 0.25),
        Vec3d::new(0.02, 0.28, 0.02),
    ),
    panel(
        "gear_r",
        Vec3d::new(0.3, -0.7, 0.25),
        Vec3d::new(0.02, 0.28, 0.02),
    ),
    panel(
        "gear_nose",
        Vec3d::new(0.0, -0.7, 0.85),
        Vec3d::new(0.02, 0.28, 0.02),
    ),
    glass(
        "glass_canopy",
        Vec3d::new(0.0, 0.28, 0.35),
        Vec3d::new(0.1, 0.1, 0.18),
    ),
    seat(
        "seat_r",
        Vec3d::new(0.0, -0.48, 0.35),
        Vec3d::new(0.08, 0.03, 0.08),
    ),
];
