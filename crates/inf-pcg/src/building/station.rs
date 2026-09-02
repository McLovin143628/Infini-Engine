//! **What a room offers a BODY** (wave VEN1b) — the affordances a venue's own
//! furniture implies.
//!
//! [`society`](super::society) answers *"how many people does this building
//! hold"* out of the **plan**: a bedroom is a home, an office holds workers by
//! area. That is the right question for a house and the wrong one for a bar,
//! because the thing a patron does in a bar is `sit on that stool`, `stand on
//! that piece of floor`, `pour drinks behind that counter` — and none of those
//! is a property of a rectangle. They are properties of the furniture the
//! **assembler** put in the rectangle.
//!
//! So a station is derived where the furniture is placed, in
//! [`assemble`](mod@super::assemble), and carried out on
//! [`GrammarOutput::stations`](crate::grammar::GrammarOutput::stations) beside
//! the doorways. It is `#[serde(skip)]` all the way down for the reason a
//! doorway and a slot are:
//! it is a pure function of the plan, the palette and the seed, so every host
//! re-derives it and **no schema moves**.
//!
//! # A station is a place, a posture and a facing
//!
//! ```text
//!   Stool     ──▶ Seat    ── a patron, sitting, facing the room
//!   Bench     ──▶ Seat×n  ── patrons at the stage edge, facing the catwalk
//!   BarRun    ──▶ Tend    ── the keeper, BEHIND the counter, facing out
//!   Stage     ──▶ Perform ── the act, ON the deck, at the pole
//!   (floor)   ──▶ Mingle  ── the dance floor, on a spaced lattice
//!   (door)    ──▶ Guard   ── the bouncer, outside the entrance
//!   (room)    ──▶ Music   ── not a person: the music bus's emitter
//! ```
//!
//! # THE SPACING IS THE OCCUPANCY (the "a crowd is a wall" answer)
//!
//! A kinematic `Full` crowd agent does not part for another
//! (`inf_ecs::crowd::steer_agent` writes a pursuit intent and no separation
//! term), and a dance floor is the densest interior this engine has. The wave
//! carried two candidate answers — *fix the avoidance* or *cap the occupancy* —
//! and the measurement that picks one is in [`MINGLE_PITCH_M`]: an agent
//! standing at a dance station has **arrived**, and an arrived agent's steering
//! wish is `ZERO`. Avoidance can separate two agents that are *walking*; it can
//! do nothing at all about two agents standing on one point. The overlap on a
//! dance floor is a **destination** collision, so the fix belongs at the
//! destination.
//!
//! Every station of a room is therefore at least one body-diameter from every
//! other, by construction, and a room offers exactly as many as fit.
//!
//! # Portable math
//!
//! A lattice is a `floor` of a division, a facing is one of four axis-aligned
//! normals rotated by the lot's own quaternion, and a distance is a `sqrt` of a
//! sum of products. No trigonometry: a station's metres land on an NPC's
//! `Transform` and therefore in the replay trace (P14).

use glam::{DVec2, DVec3};

use super::{Rect2, RoomType};

/// **What a body does at a station.**
///
/// Six kinds and not a boolean, because the vocabulary is what a schedule is
/// written in downstream: [`Seat`](StationUse::Seat) and
/// [`Mingle`](StationUse::Mingle) become patron slots, the other three become
/// night-shift jobs, and [`Music`](StationUse::Music) is not a person at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StationUse {
    /// A seat on a placed piece of furniture — a stool, or one place on a bench.
    Seat,
    /// A patch of floor with room to stand and move on it, on a spaced lattice.
    Mingle,
    /// Behind a counter: where the keeper stands to serve the run in front of
    /// them.
    Tend,
    /// On a raised deck: where the act performs.
    Perform,
    /// Outside an entrance: where the door is watched from.
    Guard,
    /// **Not a person** — where a room's music comes from.
    Music,
}

impl StationUse {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            StationUse::Seat => "seat",
            StationUse::Mingle => "mingle",
            StationUse::Tend => "tend",
            StationUse::Perform => "perform",
            StationUse::Guard => "guard",
            StationUse::Music => "music",
        }
    }

    /// Whether a body stands here at all — everything but
    /// [`Music`](StationUse::Music).
    ///
    /// One door, because the two consumers of a station list ask exactly
    /// complementary questions and a station that answered neither (or both)
    /// would be a silent hole in one of them.
    pub fn is_occupied_by_a_person(self) -> bool {
        !matches!(self, StationUse::Music)
    }
}

/// **One place one body can be, in a room, doing one thing.**
///
/// Located in world metres like a [`PcgDoorway`](super::PcgDoorway) and naming
/// no index, so concatenating two buildings' stations needs no re-basing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgStation {
    /// What happens here.
    pub use_kind: StationUse,
    /// Where the body's **feet** go, world metres — on the walking surface, or
    /// on the deck for a [`Perform`](StationUse::Perform), or in the air for a
    /// [`Music`](StationUse::Music) emitter.
    pub at: DVec3,
    /// **A unit direction the body faces**, in the plan's XZ plane.
    ///
    /// A direction and not an angle, deliberately: a lot frame is applied as a
    /// quaternion (`assemble::place_in_frame`), so rotating a vector is exact
    /// while rotating a *degree* would need an `atan2` at PCG time — and P14
    /// bans libm on anything that reaches a `Transform`. A consumer that wants
    /// a yaw pays one `inf_math::patan2_64` for it, once, when a body arrives.
    ///
    /// `DVec3::ZERO` for a station with no opinion (every
    /// [`Music`](StationUse::Music) emitter).
    pub face: DVec3,
    /// Index into the plan's [`rooms`](super::BuildingPlan::rooms).
    pub room: u32,
    /// The storey, 0-based.
    pub floor: u32,
}

/// The pitch of a dance floor's standing lattice, metres.
///
/// **Two and a third body diameters.** `CrowdArchetype::humanoid` is a 0.3 m
/// capsule radius, so two agents interpenetrate below 0.6 m and a lattice at
/// exactly that would put every patron permanently in contact with four
/// neighbours. 1.4 m leaves 0.8 m of clear floor between two bodies, which is
/// the spacing a dance floor reads at in `venues/0060` — near enough to be a
/// crowd, far enough that nobody is standing inside anybody.
pub const MINGLE_PITCH_M: f64 = 1.4;

/// The most standing stations one room offers, whatever its area.
///
/// A ceiling on the same terms as
/// [`MAX_SLOTS_PER_ROOM`](super::society::MAX_SLOTS_PER_ROOM), and needed for
/// the same reason: a nightclub's `max_room_area` is 260 m², which at
/// [`MINGLE_PITCH_M`] is 132 lattice points, and a level whose whole population
/// ceiling is a thousand should not spend an eighth of it on one room's floor.
/// Sixteen is a busy floor in a shot and is what
/// `a_dance_floor_offers_spaced_standing_room` measures against.
pub const MAX_MINGLE_PER_ROOM: usize = 16;

/// How far a seat sits along a bench, metres — one place per this much length.
///
/// A person on a bench takes about half a metre of it; 0.6 m is that plus the
/// gap nobody sits in. A 3 m bench therefore seats five, which is what
/// `venues/0028` shows along the catwalk edge.
pub const BENCH_SEAT_PITCH_M: f64 = 0.6;

/// How far behind a counter its keeper stands, metres, measured from the
/// counter's own back face.
///
/// Half a metre: inside the service side, out of the counter's own collider,
/// and near enough that the body reads as *at* the bar rather than against the
/// back wall.
pub const KEEPER_STAND_M: f64 = 0.5;

/// How far outside an entrance its guard stands, metres, measured from the
/// door leaf.
///
/// A metre and a half is one pace clear of the swing (`DEFAULT_DOOR_WIDTH_M` is
/// 0.9 and a leaf reaches a metre out from its wall), which is where the
/// bouncer stands in `venues/0060` — beside the door and not in it.
pub const GUARD_STAND_M: f64 = 1.5;

/// How far above the walking surface a room's music emitter hangs, as a
/// fraction of the storey height.
///
/// Three quarters: above head height, below the rig, which is where a club's
/// speakers are and — more usefully — is a point the doorway rule's ray can
/// reach from a listener standing outside without grazing the floor.
pub const MUSIC_HANG: f64 = 0.75;

/// **How many bodies a named module seats, and how they are spaced along it.**
///
/// Keyed on the module NAME rather than on
/// [`ModuleShape`](super::modules::ModuleShape), and that is the load-bearing
/// choice here: `shape_of` maps `"Desk" | "Table" | "Bench"` all onto
/// `ModuleShape::Legged`, so a shape test would seat four patrons on the office
/// desk in the venue's back room. A name is what the palette author wrote and
/// is what a seat is a property of.
///
/// `None` for every module nobody sits on, which is every module in the twelve
/// pre-venue palettes.
pub fn seats_of(module: &str) -> Option<SeatSpec> {
    Some(match module {
        // A stool is one seat, on top of itself.
        "Stool" => SeatSpec {
            pitch_m: None,
            rise: 1.0,
        },
        // A bench is a RUN of seats along its own long axis, and the whole
        // point of the family: `venues/0028` is five people on one bench.
        "Bench" => SeatSpec {
            pitch_m: Some(BENCH_SEAT_PITCH_M),
            rise: 1.0,
        },
        _ => return None,
    })
}

/// How a named module is sat on. See [`seats_of`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeatSpec {
    /// Metres of the piece's long axis one seat takes, or `None` for a piece
    /// that is one seat however long it is.
    pub pitch_m: Option<f64>,
    /// Where the seat surface is, as a fraction of the piece's own height above
    /// its base — `1.0` is the top.
    pub rise: f64,
}

/// **Whether a named module is a counter somebody works behind.**
///
/// One module today. A function rather than a `==` so the answer is in one
/// place when the second one arrives, exactly as
/// [`seats_of`] is.
pub fn tends_of(module: &str) -> bool {
    module == "BarRun"
}

/// **Whether a named module is a deck somebody performs on.**
pub fn performs_of(module: &str) -> bool {
    matches!(module, "Stage" | "Catwalk")
}

/// **Whether a room is one people go to be IN**, rather than one they work or
/// sleep in — the gate on the whole derivation.
///
/// # This exists because a factory has benches
///
/// Measured, at the first run of `a_venue_offers_a_countable_number_of_places_to_be`
/// and before it was gated: the **Factory / warehouse** palette puts a `Bench`
/// in its workshop, and [`seats_of`] duly offered **111 seats** on one. That is
/// not a harmless extra — a slot reaches `inf_ecs::society`, so a hundred and
/// eleven leisure places would appear on an industrial estate, the town's night
/// out would be planned onto a workshop bench, and every level that predates
/// this wave would stop being byte-identical to itself.
///
/// So the derivation is gated on the ROOM and not on the archetype. A rule
/// about the archetype would say "venues only" and would be wrong the day a
/// café gets a `BarRoom`; a rule about the room says *a bench in a bar room
/// seats a patron and a bench in a workshop is where you put your tools*, which
/// is the true sentence and needs no amendment when the eleventh archetype
/// arrives.
///
/// The three rooms are exactly the three
/// [`shift_of`](super::society::shift_of) calls `Night`, and
/// `the_social_rooms_are_the_night_rooms` is the arm that keeps the two lists
/// from drifting apart.
pub fn is_social(kind: RoomType) -> bool {
    matches!(
        kind,
        RoomType::BarRoom | RoomType::DanceFloor | RoomType::Stage
    )
}

/// **Whether a building watches its own door** (wave VEN1b audit) — whether any
/// of its rooms is one people go to be in.
///
/// # The bouncer was gated on the NEON SIGN
///
/// A `Guard` station is emitted from `assemble::street_face`, which is where the
/// assembler has resolved which wall is the street and which way is out — the
/// right place, and the reason it is there. But `street_face` returns early on
/// `BuildingArchetype::entrance_sign`, so until this function existed the rule
/// *"a venue's door is watched at night"* was really *"a building with a lit
/// sign over its door is watched at night"*. It measured correctly — the three
/// venues are exactly the three archetypes that declare a sign — and it was
/// correct for a reason that has nothing to do with venues. A shop given a
/// signboard would have grown a **night job**: a bouncer on a bakery, and an
/// agent planned onto a night shift there.
///
/// So the gate is a rule about the ROOMS, which is what
/// [`is_social`] already is and what the wave's own law says outlives a rule
/// about an archetype. Behaviour today is unchanged to the station: all three
/// venues have a `BarRoom`, a `DanceFloor` or a `Stage`, and the seven that
/// predate them have none — which is what
/// `a_venue_offers_a_countable_number_of_places_to_be`'s guard column already
/// measures.
pub fn watches_its_door(kinds: impl IntoIterator<Item = RoomType>) -> bool {
    kinds.into_iter().any(is_social)
}

/// **Whether a room's floor is standing room** — the dance floor, and nothing
/// else.
///
/// A `BarRoom`'s floor is *also* standing room in a real bar; it is excluded
/// deliberately, because a bar room already offers stools and a counter and
/// filling the rest of it with a lattice would put a body between the keeper
/// and every one of their customers. The dance floor is the room whose whole
/// purpose is people standing on it.
pub fn is_standing_room(kind: RoomType) -> bool {
    matches!(kind, RoomType::DanceFloor)
}

/// **The standing lattice of one room** — every point at least
/// [`MINGLE_PITCH_M`] from every other and clear of everything already placed.
///
/// `placed` is the assembler's own occupied list, `(centre, reach)` pairs in
/// the plan's coordinates: a centred stage registers its diagonal reach there,
/// which is exactly how the lattice comes out as *the floor around the stage*
/// rather than *the floor including the stage*.
///
/// Deterministic and hash-free: the lattice is laid from the room's own inset
/// rectangle, walked in row-major order, and truncated at
/// [`MAX_MINGLE_PER_ROOM`]. Two hosts lay the same floor because a `floor` of a
/// division is exact on every target.
pub fn mingle_points(inner: &Rect2, placed: &[(DVec2, f64)]) -> Vec<DVec2> {
    let mut out = Vec::new();
    if !inner.is_positive() {
        return out;
    }
    let (sx, sz) = (inner.size_x(), inner.size_z());
    let nx = (sx / MINGLE_PITCH_M).floor() as i64;
    let nz = (sz / MINGLE_PITCH_M).floor() as i64;
    if nx < 1 || nz < 1 {
        return out;
    }
    // Centred in the room rather than pinned to its `min` corner, so the border
    // the lattice leaves is even and a floor with room for three rows does not
    // put all three against one wall.
    let (ox, oz) = (
        inner.min.x + (sx - (nx - 1) as f64 * MINGLE_PITCH_M) * 0.5,
        inner.min.y + (sz - (nz - 1) as f64 * MINGLE_PITCH_M) * 0.5,
    );
    for j in 0..nz {
        for i in 0..nx {
            if out.len() >= MAX_MINGLE_PER_ROOM {
                return out;
            }
            let p = DVec2::new(
                ox + i as f64 * MINGLE_PITCH_M,
                oz + j as f64 * MINGLE_PITCH_M,
            );
            // Clear of the stage, the pole and every bench already registered.
            // `reach` is the piece's own diagonal, so this is "outside the
            // circle that contains it" and never "outside its bounding box",
            // which would let a body stand on a corner of the catwalk.
            if placed
                .iter()
                .any(|(c, r)| (p - *c).length() < r + MINGLE_PITCH_M * 0.5)
            {
                continue;
            }
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: f64, h: f64) -> Rect2 {
        Rect2::new(DVec2::new(-w * 0.5, -h * 0.5), DVec2::new(w * 0.5, h * 0.5))
    }

    /// **THE SPACING CLAUSE.** Every pair of standing stations is at least a
    /// pitch apart, which is what "the occupancy is capped to the spacing"
    /// means as a number rather than as a hope.
    #[test]
    fn a_standing_lattice_never_puts_two_bodies_inside_one_another() {
        let pts = mingle_points(&rect(9.0, 7.0), &[]);
        assert!(pts.len() > 4, "a 63 m2 floor offered {} spots", pts.len());
        for (i, a) in pts.iter().enumerate() {
            for b in pts.iter().skip(i + 1) {
                let d = (*a - *b).length();
                assert!(
                    d >= MINGLE_PITCH_M - 1e-9,
                    "two dance stations are {d:.3} m apart against a pitch of \
                     {MINGLE_PITCH_M}"
                );
            }
        }
    }

    /// The ceiling fires rather than minting a crowd out of one rectangle —
    /// `MAX_SLOTS_PER_ROOM`'s own argument, applied to the floor.
    #[test]
    fn a_huge_floor_is_capped_rather_than_filled() {
        let pts = mingle_points(&rect(60.0, 60.0), &[]);
        assert_eq!(pts.len(), MAX_MINGLE_PER_ROOM);
    }

    /// A stage in the middle of the room takes its own floor out of the
    /// lattice — the claim that makes a dancer's deck a deck rather than a
    /// place four patrons stand.
    ///
    /// **The room is deliberately small enough that the CAP does not bind.**
    /// The first spelling used a 12 × 10 m floor, whose bare lattice is already
    /// 16 points — [`MAX_MINGLE_PER_ROOM`] — so clearing the middle merely let
    /// four outer points in and the count did not move. A 7 × 7 m one is 25
    /// points and does the same. A ceiling hides a difference; the arm has to
    /// be taken below it, which for a 1.4 m pitch means a room under 5.6 m.
    #[test]
    fn a_placed_piece_clears_the_floor_it_stands_on() {
        const REACH: f64 = 0.8;
        let inner = rect(5.5, 5.5);
        let bare = mingle_points(&inner, &[]);
        assert!(
            bare.len() < MAX_MINGLE_PER_ROOM,
            "the fixture room is at the cap ({}), so this arm would measure the \
             ceiling rather than the stage",
            bare.len()
        );
        let with_stage = mingle_points(&inner, &[(DVec2::ZERO, REACH)]);
        assert!(
            with_stage.len() < bare.len(),
            "a piece of reach {REACH} removed no standing room from a {} spot \
             floor",
            bare.len()
        );
        assert!(!with_stage.is_empty(), "…and it removed the whole floor");
        // The rule, exactly as `mingle_points` states it: a body stands clear of
        // the piece's own reach plus half a pitch of its own.
        for p in &with_stage {
            assert!(
                p.length() >= REACH + MINGLE_PITCH_M * 0.5,
                "a body stands {:.3} m from a piece whose reach is {REACH}",
                p.length()
            );
        }
    }

    /// **The social rooms ARE the night rooms.** Two lists that name the same
    /// three room types, in two modules, is exactly the shape of drift this
    /// tree has paid for before (`slots_of`'s two `== Retail` tests, VEN1a) —
    /// so the arm is the pin.
    #[test]
    fn the_social_rooms_are_the_night_rooms() {
        for kind in RoomType::ALL {
            assert_eq!(
                is_social(kind),
                super::super::society::shift_of(kind) == super::super::society::SlotShift::Night,
                "{} is social {} and worked at night {}",
                kind.name(),
                is_social(kind),
                super::super::society::shift_of(kind) == super::super::society::SlotShift::Night
            );
        }
    }

    /// A room too small to hold one body at the pitch offers nothing, rather
    /// than one point at the wall.
    #[test]
    fn a_cupboard_is_not_a_dance_floor() {
        assert!(mingle_points(&rect(1.0, 1.0), &[]).is_empty());
        assert!(mingle_points(&rect(0.0, 0.0), &[]).is_empty());
    }

    /// **The name table, both ways.** A shape test would seat patrons on the
    /// office desk, because `shape_of` maps `Desk`, `Table` and `Bench` onto
    /// one family.
    #[test]
    fn a_seat_is_named_and_not_shaped() {
        assert!(seats_of("Stool").is_some());
        assert!(seats_of("Bench").is_some());
        for other in ["Desk", "Table", "Sofa", "BarRun", "Stage", "Wall"] {
            assert!(
                seats_of(other).is_none(),
                "{other} was offered to somebody to sit on"
            );
        }
        // …and the three that *are* the same family as `Bench` prove the point.
        assert_eq!(
            super::super::modules::shape_of("Bench"),
            super::super::modules::shape_of("Desk"),
            "the test this table exists to defeat no longer applies"
        );
        assert!(tends_of("BarRun") && !tends_of("Counter"));
        assert!(performs_of("Stage") && performs_of("Catwalk") && !performs_of("Deck"));
        assert!(is_standing_room(RoomType::DanceFloor));
        assert!(!is_standing_room(RoomType::BarRoom) && !is_standing_room(RoomType::Bedroom));
    }

    /// **A door is watched because of the ROOMS behind it**, not because of the
    /// sign over it (wave VEN1b audit).
    ///
    /// The arm on [`watches_its_door`] itself, because the thing it replaces —
    /// `street_face`'s early return on `entrance_sign` — cannot be measured
    /// through the archetype table: no archetype in the tree has a sign and no
    /// social room, which is exactly why the coupling was invisible. Here the
    /// two cases are both spellable.
    #[test]
    fn a_door_is_watched_for_the_rooms_behind_it() {
        assert!(watches_its_door([RoomType::Lobby, RoomType::BarRoom]));
        assert!(watches_its_door([RoomType::DanceFloor]));
        assert!(watches_its_door([RoomType::Corridor, RoomType::Stage]));
        // A shop with a signboard over its door is still a shop.
        assert!(!watches_its_door([
            RoomType::Retail,
            RoomType::Storage,
            RoomType::Lobby,
        ]));
        assert!(!watches_its_door([]));
        // …and it is the same rule `is_social` is, rather than a second list.
        for kind in RoomType::ALL {
            assert_eq!(watches_its_door([kind]), is_social(kind), "{}", kind.name());
        }
    }

    /// The uses split cleanly into "a person is here" and "a person is not".
    #[test]
    fn every_station_use_is_a_person_or_an_emitter() {
        for u in [
            StationUse::Seat,
            StationUse::Mingle,
            StationUse::Tend,
            StationUse::Perform,
            StationUse::Guard,
        ] {
            assert!(u.is_occupied_by_a_person(), "{} holds nobody", u.name());
        }
        assert!(!StationUse::Music.is_occupied_by_a_person());
    }
}
