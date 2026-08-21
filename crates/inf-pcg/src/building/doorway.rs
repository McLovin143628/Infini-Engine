//! **Where a building's doors go** (island wave I6).
//!
//! The grammar has always known where its doorways are — `Opening { kind: Door }`
//! on a wall, an interval along the run and a band in Y — and has always emitted
//! them as *holes*: the wall runs stop at the jamb and start again past it, so a
//! doorway is a place no module covers. That is exactly right for geometry and
//! it leaves nothing for a **door** to be hinged on.
//!
//! This module turns each of those openings into a [`PcgDoorway`]: a hinge, a
//! facing, a size and which side is inside. It is pure — no world, no entities,
//! no physics — and it is the only place the derivation is written, so the cook,
//! the shipped player and the editor's Simulate cannot disagree about where a
//! building's front door is.
//!
//! # Why a doorway is derived and not stored
//!
//! `BuildingPlan` is not retained at runtime: `evaluate_buildings_in` keeps the
//! instances, the colliders and the groups and throws the plan away. So a
//! doorway has to be produced **in the same pass that has the plan in hand**,
//! and ride the population out with everything else — which is why
//! [`crate::VolumeOutput`] carries a `doorways` list beside its `colliders`.
//!
//! # The frame
//!
//! Everything here is computed in the **lot's** coordinates, exactly as the rest
//! of the plan is (the IB-6 ruling: plan in the lot's frame, place in the
//! world's), and `assemble::place_in_frame`'s twin
//! [`place_doorways_in_frame`] turns the finished list into the world. The
//! identity frame is skipped by an exact comparison, so nothing an axis-aligned
//! lot produces moves by a bit.

use glam::{DVec2, DVec3};

use super::{palettes, BuildingPlan, OpeningKind};

/// **One doorway a building offers**, in whatever frame the plan was built in.
///
/// The fields are exactly what `inf_ecs::door::DoorSpec` needs plus the hinge —
/// deliberately a *mirror* rather than the type itself, because `inf-pcg` does
/// not depend on `inf-ecs` (the P19.5 dependency-light mirror ruling, which
/// `PcgCollider`/`ScatteredSolid` and `PcgInstance`/`ScatteredInstance` already
/// follow). The translation is written once per host and pinned by the same
/// `population_of` mirror gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgDoorway {
    /// The hinge, at the leaf's **mid-height**.
    pub hinge: DVec3,
    /// The compass yaw from the hinge toward the free edge when shut, degrees.
    pub closed_yaw_deg: f64,
    /// The leaf's width, metres — the opening's own.
    pub width_m: f64,
    /// The leaf's height, metres.
    pub height_m: f64,
    /// The leaf's thickness, metres — the wall's own.
    pub thickness_m: f64,
    /// The compass yaw of the **inside** face's outward normal, degrees.
    pub inside_yaw_deg: f64,
    /// Whether this is the building's one exterior door.
    pub exterior: bool,
    /// Which storey it is on — carried so a caller can band or filter by floor
    /// without re-deriving it from the height.
    pub floor: u32,
}

/// The compass yaw of a planar direction, degrees — `+Z` at zero, `+X` at `+90`.
///
/// `patan2_64`, not `f64::atan2`: this reaches a door's collider pose and
/// therefore the replay trace, and the P14 law is that std trig is not
/// bit-portable.
fn planar_yaw_deg(v: DVec2) -> f64 {
    if v.x.abs() < 1e-12 && v.y.abs() < 1e-12 {
        return 0.0;
    }
    inf_math::patan2_64(v.x, v.y).to_degrees()
}

/// **Every door this plan wants**, in plan order.
///
/// Only `OpeningKind::Door` — a window is an opening you look through, and
/// hanging a leaf in one would be a door in a window frame.
///
/// The **hinge is at the opening's `start` end** and the **inside face is the
/// side `Wall::inside` names**, which is what makes a grammar door swing the way
/// a building's own plan says it should: a room's door opens into the room.
pub fn doorways_of(plan: &BuildingPlan) -> Vec<PcgDoorway> {
    let arch = palettes::archetype(plan.archetype);
    let mut out = Vec::new();
    for o in &plan.openings {
        if o.kind != OpeningKind::Door {
            continue;
        }
        let Some(w) = plan.walls.get(o.wall) else {
            continue;
        };
        let width = o.width();
        let height = o.head - o.sill;
        if !(width.is_finite() && width > 0.0 && height.is_finite() && height > 0.0) {
            continue;
        }
        let dir = w.direction();
        let hinge_xz = w.point_at(o.start);
        let floor_y = plan.floor_y(w.floor);
        let mid_y = floor_y + (o.sill + o.head) * 0.5;
        // The wall's normal, and which of its two signs points into the room the
        // wall calls `inside`. A door swings into the room it serves, and an
        // exterior door swings into the building — which is the same rule, since
        // `Wall::inside` is the interior room for an exterior wall.
        let normal = DVec2::new(-dir.y, dir.x);
        let inside = plan
            .rooms
            .get(w.inside)
            .map(|r| r.rect.center())
            .unwrap_or(hinge_xz + normal);
        let signed = if (inside - hinge_xz).dot(normal) >= 0.0 {
            normal
        } else {
            -normal
        };
        out.push(PcgDoorway {
            hinge: DVec3::new(hinge_xz.x, mid_y, hinge_xz.y),
            closed_yaw_deg: planar_yaw_deg(dir),
            width_m: width,
            height_m: height,
            thickness_m: arch.wall_thickness,
            inside_yaw_deg: planar_yaw_deg(signed),
            exterior: plan.entrance == Some(o.wall) && w.is_exterior(),
            floor: w.floor,
        });
    }
    out
}

/// **Place a doorway list in the world**, the twin of
/// `assemble::place_in_frame` and for the same reason: everything upstream is
/// computed in the lot's own frame, where every existing rule is already
/// correct.
///
/// The identity frame is skipped by an **exact** comparison, so an axis-aligned
/// lot's doorways are bit-identical to what the plan produced — which is what
/// keeps every committed level's population unmoved.
pub fn place_doorways_in_frame(out: &mut [PcgDoorway], frame: super::LotFrame) {
    if frame.is_identity() {
        return;
    }
    // The frame's own yaw, as an angle, so a doorway's facings rotate with it.
    // `LotFrame::v()` is the frame's local `+Z` in world XZ, and a yaw is
    // measured from `+Z` — so the frame's yaw is `planar_yaw_deg(v)`.
    let yaw = planar_yaw_deg(frame.v());
    for d in out.iter_mut() {
        let xz = frame.to_world(DVec2::new(d.hinge.x, d.hinge.z));
        d.hinge = DVec3::new(xz.x, d.hinge.y, xz.y);
        d.closed_yaw_deg = wrap_deg(d.closed_yaw_deg + yaw);
        d.inside_yaw_deg = wrap_deg(d.inside_yaw_deg + yaw);
    }
}

/// Fold an angle into `(-180, 180]`.
fn wrap_deg(a: f64) -> f64 {
    if !a.is_finite() {
        return 0.0;
    }
    let mut x = a % 360.0;
    if x > 180.0 {
        x -= 360.0;
    }
    if x <= -180.0 {
        x += 360.0;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::{plan_building, plan_building_in, ArchetypeId, BuildingParams, Rect2};

    fn params(arch: ArchetypeId) -> BuildingParams {
        BuildingParams {
            archetype: arch,
            footprint: Rect2::new(DVec2::new(0.0, 0.0), DVec2::new(18.0, 12.0)),
            base_y: 0.0,
            seed: 7,
            floors: 2,
        }
    }

    /// **Every enterable archetype offers doors**, and the count is the plan's
    /// own door openings — not a guess.
    #[test]
    fn every_archetype_emits_one_doorway_per_door_opening() {
        for arch in ArchetypeId::ALL {
            let plan = plan_building(&params(arch));
            let doors = plan
                .openings
                .iter()
                .filter(|o| o.kind == OpeningKind::Door)
                .count();
            let ways = doorways_of(&plan);
            println!(
                "{:?}: {} rooms, {} walls, {doors} door openings -> {} doorways",
                arch,
                plan.rooms.len(),
                plan.walls.len(),
                ways.len()
            );
            assert_eq!(ways.len(), doors, "{arch:?}");
            assert!(doors > 0, "{arch:?} plans no doors at all");
            // Exactly one of them is the entrance.
            let exterior = ways.iter().filter(|d| d.exterior).count();
            assert_eq!(exterior, 1, "{arch:?} has {exterior} front doors");
            // A window is never a door.
            for d in &ways {
                assert!(d.width_m > 0.0 && d.height_m > 0.0);
                assert!(d.hinge.is_finite());
                assert!(d.closed_yaw_deg.is_finite() && d.inside_yaw_deg.is_finite());
                assert!(d.thickness_m > 0.0);
            }
        }
    }

    /// **A doorway's leaf spans its own opening**, hinge to free edge, and its
    /// mid-height is inside the opening's Y band.
    #[test]
    fn a_doorway_is_hinged_at_its_openings_start_and_as_wide_as_it_is() {
        let plan = plan_building(&params(ArchetypeId::House));
        let ways = doorways_of(&plan);
        let doors: Vec<_> = plan
            .openings
            .iter()
            .filter(|o| o.kind == OpeningKind::Door)
            .collect();
        assert_eq!(ways.len(), doors.len());
        let mut worst = 0.0_f64;
        for (d, o) in ways.iter().zip(doors.iter()) {
            let w = &plan.walls[o.wall];
            let start = w.point_at(o.start);
            let end = w.point_at(o.end);
            worst = worst.max((DVec2::new(d.hinge.x, d.hinge.z) - start).length());
            assert!((d.width_m - (end - start).length()).abs() < 1e-9);
            let floor_y = plan.floor_y(w.floor);
            assert!(d.hinge.y > floor_y + o.sill - 1e-9);
            assert!(d.hinge.y < floor_y + o.head + 1e-9);
        }
        println!(
            "the worst hinge placement error over {} doorways is {worst:e} m",
            ways.len()
        );
        assert!(worst < 1e-9);
    }

    /// **The leaf swings INTO the room the wall serves**, which is what a door
    /// hung the other way round would not do.
    #[test]
    fn a_doorways_inside_normal_points_at_its_own_room() {
        let plan = plan_building(&params(ArchetypeId::Apartment));
        let ways = doorways_of(&plan);
        let doors: Vec<_> = plan
            .openings
            .iter()
            .filter(|o| o.kind == OpeningKind::Door)
            .collect();
        let mut checked = 0;
        for (d, o) in ways.iter().zip(doors.iter()) {
            let w = &plan.walls[o.wall];
            let Some(room) = plan.rooms.get(w.inside) else {
                continue;
            };
            let hinge = DVec2::new(d.hinge.x, d.hinge.z);
            let toward = room.rect.center() - hinge;
            let r = d.inside_yaw_deg.to_radians();
            let normal = DVec2::new(inf_math::psin64(r), inf_math::pcos64(r));
            assert!(
                toward.dot(normal) >= 0.0,
                "doorway {d:?} faces away from room {}",
                w.inside
            );
            checked += 1;
        }
        println!("{checked} doorways face the rooms their walls serve");
        assert!(checked > 0, "the arm checked nothing");
    }

    /// **The lot frame moves a doorway and the identity frame does not** — the
    /// IB-6 rule, at a door.
    #[test]
    fn a_rotated_lot_carries_its_doorways_and_an_axis_aligned_one_moves_nothing() {
        let p = params(ArchetypeId::House);
        let straight = doorways_of(&plan_building(&p));
        // The identity frame is a no-op, bit for bit.
        let mut same = straight.clone();
        place_doorways_in_frame(&mut same, super::super::LotFrame::IDENTITY);
        assert_eq!(same, straight, "the identity frame moved a doorway");

        // A rotated lot: the plan is the same in its own frame and different in
        // the world's.
        let frame = super::super::LotFrame {
            origin: DVec2::new(100.0, -40.0),
            u: DVec2::new(0.0, 1.0),
        };
        let rotated_plan = plan_building_in(&p, frame);
        let mut rotated = doorways_of(&rotated_plan);
        assert_eq!(
            rotated.len(),
            straight.len(),
            "rotating the lot changed how many doors the building has"
        );
        place_doorways_in_frame(&mut rotated, frame);
        assert_ne!(rotated, straight, "a rotated lot's doorways did not move");
        // …and every one of them really is inside the rotated footprint's own
        // world position, which a frame applied the wrong way round would not be.
        let centre = frame.to_world(rotated_plan.footprint.center());
        let ext = rotated_plan.footprint.max - rotated_plan.footprint.min;
        let reach = ext.x.max(ext.y);
        let mut worst = 0.0_f64;
        for d in &rotated {
            worst = worst.max((DVec2::new(d.hinge.x, d.hinge.z) - centre).length());
        }
        println!(
            "the furthest doorway of a lot centred at {centre:?} is {worst} m out; the footprint's own reach is {reach} m"
        );
        assert!(worst <= reach, "a doorway landed outside its own building");
    }
}
