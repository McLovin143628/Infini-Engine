//! **Structure LOD** (IB-2b): the one rule that decides whether a building is
//! its parts, its shell, or nothing.
//!
//! A grammar building is ~1 000–2 000 boxes. At the certification's measured
//! **0.363 µs per collider per step** that is 0.5–0.7 ms of physics *each*, and a
//! thousand of them is half a second — so a city cannot simulate, or draw, every
//! building as its parts. It does not need to: a building you cannot reach is a
//! silhouette, and a silhouette is one box.
//!
//! # The three tiers, and why they are ONE rule
//!
//! ```text
//!   |<-- near_m -->|<--------- far_m --------->|
//!   [   parts      ][        shell             ][ nothing ]
//!     ~1500 boxes        1 box                    0 boxes
//! ```
//!
//! [`tier_at`] is called by two consumers with two different pairs of radii —
//! the physics bridge bands **colliders** off the sim's own streaming sources,
//! and each host's projection bands **instances** off the camera. Writing the
//! three-way comparison twice is exactly how the two would come to disagree about
//! what "far" means, and the day they do, a building draws as a shell you can
//! still walk into (or as parts you fall through). One function, two callers.
//!
//! # Which input each consumer may use, and why they differ
//!
//! The standing law is *visibility filters what is DRAWN, never what is
//! SIMULATED*. The collider band therefore takes its anchors from
//! [`inf_ecs::StreamingSource`] entities — the same sim-side positions P16's cell
//! activation reads, never a camera and never a frame counter — so the active
//! collider set is a pure function of the world's contents and PIE == shipping
//! holds on a drive-through. The render band takes the camera, because a draw
//! set is *allowed* to be a function of the camera and nothing else may read it.
//! The two share this arithmetic and nothing else.
//!
//! # The shell is derived, never authored
//!
//! [`group_shell`] folds a building's own boxes into the smallest box **in the
//! lot's frame** that contains them. In the lot's frame, because a building's
//! parts are square to its lot by construction (the IB-6 property, asserted by
//! `a_rotated_lot_builds_a_rotated_building`) — so the lot-frame bound is tight,
//! while a world-axis bound of a block turned 37° is up to 2.6× too large and
//! would put an invisible wall in the street.

use glam::{DQuat, DVec2, DVec3};

use super::LotFrame;
use crate::scatter::PcgCollider;

/// Metres beyond which a building draws (and simulates) as its shell.
///
/// **Not a taste value**: at the certification's 0.363 µs/collider and a 4.0 ms
/// streamed step budget the whole engine affords ~11 000 static colliders, and a
/// dense downtown holds roughly one building per 700 m² of ground. 96 m of
/// radius is ~29 000 m² and ~40 buildings — already past the budget at full
/// parts, which is why the *near* radius the city fixture achieves is smaller
/// still and is stated as a measurement rather than guessed here. This is the
/// default a level inherits when it says nothing; the measured number lives in
/// the arm that prints it.
pub const DEFAULT_STRUCTURE_LOD_M: f64 = 96.0;

/// Metres beyond which a building is neither drawn nor simulated.
pub const DEFAULT_STRUCTURE_CULL_M: f64 = 4_000.0;

/// What a building is, at a given distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureTier {
    /// Full grammar geometry: every part drawn, every solid simulated, every
    /// door a real gap. A building in this tier is enterable exactly as it was
    /// before IB-2b existed.
    Near,
    /// The building's shell: one instance, one collider. Solid, so a body cannot
    /// pass through it; opaque, so the skyline is right.
    Far,
    /// Out of range: no instance, no collider.
    Out,
}

impl StructureTier {
    /// `true` when this tier carries the building's own parts.
    #[inline]
    pub fn is_near(self) -> bool {
        matches!(self, Self::Near)
    }
}

/// The tier a building at `distance_m` belongs to.
///
/// **Half-open on the near boundary** (`distance <= near_m` is Near), so a
/// building exactly on the line is the *richer* tier — a boundary case that
/// resolves towards "enterable" cannot strand a character inside a shell.
///
/// A non-finite distance is [`StructureTier::Out`]: a NaN compares false against
/// every bound, and letting it fall through to `Near` would put a
/// nowhere-in-particular building into the physics world (the NaN-at-doors law).
#[inline]
pub fn tier_at(distance_m: f64, near_m: f64, far_m: f64) -> StructureTier {
    if !distance_m.is_finite() {
        return StructureTier::Out;
    }
    if distance_m <= near_m {
        StructureTier::Near
    } else if distance_m <= far_m {
        StructureTier::Far
    } else {
        StructureTier::Out
    }
}

/// One structural group — one building — inside a volume's flat solid list.
///
/// # Why a RANGE and not a group id per solid
///
/// [`inf_ecs::ScatteredSolid`] is a hot, derived, 80-byte record and a city holds
/// a million of them; a `u32` on each is 4 MB of tag to express what is already
/// true of the layout. `evaluate_buildings_in` concatenates one building's output
/// at a time, so a building's solids are **already contiguous**, and a range says
/// so in 8 bytes per building instead of 4 per box. It also keeps
/// `ScatteredSolid` byte-identical, which is what the physics mirror test pins.
///
/// Solids covered by no group — a fence, a `grammar.expand` pass, a scatter — are
/// banded individually against their own box. A group is an optimisation and an
/// LOD source, never a requirement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructureGroup {
    /// The group's shell: the smallest oriented box containing its solids.
    pub shell: PcgCollider,
    /// First **collider** index.
    pub start: u32,
    /// Collider count.
    pub len: u32,
    /// First **instance** index.
    pub inst_start: u32,
    /// Instance count.
    pub inst_len: u32,
}

impl StructureGroup {
    /// The half-open collider range this group covers.
    #[inline]
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start as usize..(self.start as usize + self.len as usize)
    }

    /// The half-open instance range this group covers.
    ///
    /// **Carried separately from [`range`](Self::range) rather than assumed
    /// equal to it.** For a building the two lists really are element-for-element
    /// aligned (`assemble::Ctx::boxed` pushes both at every site, which is the
    /// property `bake` depends on) — but the two lists are concatenated with
    /// *different* prefixes on the way to a volume (a scatter contributes
    /// instances and no colliders), so one range cannot index both once the
    /// composition has happened. Two ranges cost eight bytes a building and
    /// remove an invariant a caller would otherwise have to maintain.
    #[inline]
    pub fn instance_range(&self) -> std::ops::Range<usize> {
        self.inst_start as usize..(self.inst_start as usize + self.inst_len as usize)
    }
}

/// The smallest box, **in `frame`'s own basis**, containing every collider.
///
/// `None` for an empty group — an empty shell would be a zero-size collider at
/// the origin, which is a body-trap rather than a building.
pub fn group_shell(frame: LotFrame, colliders: &[PcgCollider]) -> Option<PcgCollider> {
    if colliders.is_empty() {
        return None;
    }
    let (u, v) = (frame.u, frame.v());
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for c in colliders {
        // Exact support bound: project each of the box's own axes onto the
        // frame's, which is tight for a box square to the frame and correct for
        // one that is not. No trigonometry — the rotation is applied as a
        // quaternion product, which is the P14-portable spelling.
        let e0 = c.rotation * DVec3::X;
        let e1 = c.rotation * DVec3::Z;
        let h = c.half_extents;
        let u3 = DVec3::new(u.x, 0.0, u.y);
        let v3 = DVec3::new(v.x, 0.0, v.y);
        let hu = (e0.dot(u3)).abs() * h.x + (e1.dot(u3)).abs() * h.z;
        let hv = (e0.dot(v3)).abs() * h.x + (e1.dot(v3)).abs() * h.z;
        // Y is shared by every `LotFrame` (a lot frame is a yaw), so the world Y
        // axis IS the frame's — the same support bound, projected onto it.
        let ey = c.rotation * DVec3::Y;
        let hy = e0.y.abs() * h.x + ey.y.abs() * h.y + e1.y.abs() * h.z;
        let local = frame.to_local(DVec2::new(c.center.x, c.center.z));
        let p = DVec3::new(local.x, c.center.y, local.y);
        lo = lo.min(p - DVec3::new(hu, hy, hv));
        hi = hi.max(p + DVec3::new(hu, hy, hv));
    }
    if !(lo.x.is_finite() && lo.y.is_finite() && lo.z.is_finite()) {
        return None;
    }
    let mid = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    let world = frame.to_world(DVec2::new(mid.x, mid.z));
    Some(PcgCollider {
        center: DVec3::new(world.x, mid.y, world.y),
        half_extents: half,
        rotation: frame.yaw(),
    })
}

/// Distance from `p` to an oriented box, measured on the **XZ plane only**.
///
/// Zero inside the box's footprint. Y is deliberately ignored: a character
/// twelve metres below a tower's slab is at the tower, and a band that measured
/// the full 3-D distance would drop a skyscraper's colliders the moment the
/// player walked into its basement.
///
/// Trig-free: one quaternion inverse-rotation and two clamped subtractions.
pub fn distance_xz_to_box(p: DVec3, center: DVec3, half: DVec3, rotation: DQuat) -> f64 {
    let rel = rotation.inverse() * (p - center);
    let dx = (rel.x.abs() - half.x).max(0.0);
    let dz = (rel.z.abs() - half.z).max(0.0);
    (dx * dx + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(center: DVec3, half: DVec3, rotation: DQuat) -> PcgCollider {
        PcgCollider {
            center,
            half_extents: half,
            rotation,
        }
    }

    #[test]
    fn the_tier_rule_is_three_way_and_nan_is_out() {
        assert_eq!(tier_at(0.0, 10.0, 100.0), StructureTier::Near);
        assert_eq!(tier_at(10.0, 10.0, 100.0), StructureTier::Near);
        assert_eq!(tier_at(10.000_1, 10.0, 100.0), StructureTier::Far);
        assert_eq!(tier_at(100.0, 10.0, 100.0), StructureTier::Far);
        assert_eq!(tier_at(100.1, 10.0, 100.0), StructureTier::Out);
        assert_eq!(tier_at(f64::NAN, 10.0, 100.0), StructureTier::Out);
        assert_eq!(tier_at(f64::INFINITY, 10.0, 100.0), StructureTier::Out);
        // A degenerate pair still answers: near_m > far_m means nothing is Far.
        assert_eq!(tier_at(50.0, 100.0, 10.0), StructureTier::Near);
    }

    /// **The shell is tight in the LOT's frame** — and the world-axis
    /// alternative is priced in the same arm.
    #[test]
    fn a_rotated_buildings_shell_is_tight_in_its_own_frame() {
        let (c, s) = (0.8f64, 0.6f64);
        let frame = LotFrame::new(DVec2::new(100.0, -40.0), DVec2::new(c, s));
        let yaw = frame.yaw();
        // Four wall slabs around a 24 x 12 lot, each square to the lot.
        let mut parts = Vec::new();
        for (lx, lz, hx, hz) in [
            (0.0, -6.0, 12.0, 0.2),
            (0.0, 6.0, 12.0, 0.2),
            (-12.0, 0.0, 0.2, 6.0),
            (12.0, 0.0, 0.2, 6.0),
        ] {
            let w = frame.to_world(DVec2::new(lx, lz));
            parts.push(boxed(
                DVec3::new(w.x, 3.0, w.y),
                DVec3::new(hx, 3.0, hz),
                yaw,
            ));
        }
        let shell = group_shell(frame, &parts).expect("four walls make a shell");
        assert!(
            (shell.half_extents.x - 12.2).abs() < 1e-9,
            "{:?}",
            shell.half_extents
        );
        assert!((shell.half_extents.z - 6.2).abs() < 1e-9);
        assert!((shell.half_extents.y - 3.0).abs() < 1e-9);
        assert!((shell.center - DVec3::new(100.0, 3.0, -40.0)).length() < 1e-9);

        // THE ALTERNATIVE, PRICED: the same parts bounded on the WORLD axes.
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in &parts {
            let (hx, hz) = p.xz_half_extents();
            lo = lo.min(p.center - DVec3::new(hx, p.half_extents.y, hz));
            hi = hi.max(p.center + DVec3::new(hx, p.half_extents.y, hz));
        }
        let world_area = (hi.x - lo.x) * (hi.z - lo.z);
        let lot_area = 4.0 * shell.half_extents.x * shell.half_extents.z;
        assert!(
            world_area > lot_area * 1.5,
            "world-axis {world_area:.1} m2 vs lot-frame {lot_area:.1} — if that \
             ratio is small this fixture is not rotated"
        );
        println!("IB-2b shell: lot-frame {lot_area:.1} m2 vs world-axis {world_area:.1} m2");

        // An empty group is no shell, not a zero box at the origin.
        assert!(group_shell(frame, &[]).is_none());
    }

    #[test]
    fn the_distance_is_to_the_box_on_the_xz_plane_and_ignores_height() {
        let b = boxed(DVec3::ZERO, DVec3::new(10.0, 20.0, 5.0), DQuat::IDENTITY);
        // Inside the footprint, at any height.
        assert_eq!(
            distance_xz_to_box(
                DVec3::new(0.0, 900.0, 0.0),
                b.center,
                b.half_extents,
                b.rotation
            ),
            0.0
        );
        assert_eq!(
            distance_xz_to_box(
                DVec3::new(0.0, -900.0, 0.0),
                b.center,
                b.half_extents,
                b.rotation
            ),
            0.0
        );
        // Straight out along X.
        let d = distance_xz_to_box(
            DVec3::new(13.0, 0.0, 0.0),
            b.center,
            b.half_extents,
            b.rotation,
        );
        assert!((d - 3.0).abs() < 1e-12, "{d}");
        // Diagonally off a corner: 3-4-5.
        let d = distance_xz_to_box(
            DVec3::new(13.0, 0.0, 9.0),
            b.center,
            b.half_extents,
            b.rotation,
        );
        assert!((d - 5.0).abs() < 1e-12, "{d}");
        // Rotated: the same point relative to the box's own axes.
        let yaw = DQuat::from_xyzw(0.0, 0.6, 0.0, 0.8).normalize();
        let rot = boxed(DVec3::ZERO, DVec3::new(10.0, 20.0, 5.0), yaw);
        let p = yaw * DVec3::new(13.0, 0.0, 9.0);
        let d = distance_xz_to_box(p, rot.center, rot.half_extents, rot.rotation);
        assert!((d - 5.0).abs() < 1e-9, "{d}");
    }
}
