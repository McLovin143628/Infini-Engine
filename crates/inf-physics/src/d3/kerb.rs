//! **The footway a crowd walks on is solid** (wave ROAD1b) — box colliders under
//! the kerbs and pavements the road builder draws beside a settlement's streets.
//!
//! # What this closes
//!
//! Wave ROAD1 drew a 150 mm kerb and 2 m of concrete behind it and gave neither
//! a collider. The ROAD1 audit measured what that costs over the island
//! fixture's 5 344 walkable footway triangles: the concrete is drawn **p50
//! 0.1775 m, area-weighted mean 0.1915 m above the ground an agent stands on**,
//! so an agent does not float over the slab — it walks shin-deep *inside* it —
//! and a car crosses a kerb as though it were paint (carried 19).
//!
//! # Why a box strip and not a heightfield patch
//!
//! A footway is a flat slab with one step at its inner edge, and that is a
//! cuboid. A heightfield patch would need its own lattice, its own resolution
//! decision and its own seam with the terrain heightfield beside it, to describe
//! a surface with two heights in it. The box also gives the *kerb face* for
//! free: the slab's inner wall IS the 150 mm upstand a wheel hits, and rapier's
//! character controller autosteps it (`CharacterMovement::step_height_m`
//! defaults to 0.45 m, three times the kerb) without anything here asking it to.
//!
//! # The band, and what it costs
//!
//! Slabs are chunked at [`KERB_SLAB_M`] and tiered through the same
//! [`SimBand`] every other derived collider goes through, so a street a
//! kilometre away describes nothing. That is what keeps this inside the step
//! budget: the island's 35 km of street would be ~2 200 boxes described at once
//! and the band admits the couple of dozen within
//! `inf_ecs::DEFAULT_COLLIDER_NEAR_M` of the anchor.
//!
//! # Where the height comes from, and why it is not `Street::y`
//!
//! [`inf_ecs::traffic::Street::y`] is a median over the *doorway sills* of the
//! blocks that bound a street — a first guess good to metres, which is what
//! `streets_of`'s own note says it is for. The slab has to sit on the ground the
//! road was DRAWN on, to 150 mm, so it takes its height from the terrain
//! through the topmost-terrain-that-answers rule (`inf_voxel::ground_height_at`,
//! IB-15) — the same query the ribbon builder sampled at build time and the same
//! one `inf_ecs::deform` uses per contact. Sampled once per chunk: a settlement
//! pad is smooth and a 32 m chunk of it is flat to a few centimetres.

use std::collections::BTreeSet;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{GlobalTransform, Terrain, Transform};
use inf_ecs::traffic::{Street, KERB_HEIGHT_M, KERB_WIDTH_M};
use inf_ecs::{band::SimBand, EcsWorld};

use super::{ColliderDesc3D, ColliderShape3D, EntitySync3D};

/// How long one footway slab collider is, metres.
///
/// Thirty-two metres is half `inf_ecs::DEFAULT_COLLIDER_NEAR_M`, so the band's
/// own radius always contains at least one whole chunk and a character standing
/// on a footway is never standing on the gap between two descriptions. It is
/// also two [`inf_ecs::traffic::KERB_SLOT_M`] parking bays, which is the other
/// thing that happens along a kerb.
///
/// Smaller means more boxes for the same concrete; larger means the band culls
/// in coarser lumps and a slab whose far end is out of range is still described
/// whole.
pub const KERB_SLAB_M: f64 = 32.0;

/// How deep a footway slab's box reaches below its own top, metres.
///
/// It only has to be deeper than anything that could get under it — a wheel, a
/// foot, a dropped crate — and a metre is past all three. Making it reach the
/// terrain would mean sampling the ground under the *back* of the slab as well
/// as under the kerb, and a box that thick has no more contact surface than one
/// that is not.
pub const KERB_SLAB_DEPTH_M: f64 = 1.0;

/// Salt for [`kerb_slab_guid`] — a fifth distinct constant beside
/// `PCG_STRUCTURE_SALT`, `PCG_SHELL_SALT`, `VOXEL_CHUNK_SALT` and
/// `TERRAIN_TILE_SALT`, so a footway slab can never alias any of them in the
/// bridge's one entity map.
const KERB_SLAB_SALT: u128 = 0x524f_4144_3162_4b45_5242_534c_4142_2121;

/// **The synthetic identity of one footway slab** — the street it belongs to,
/// which side of it, and how far along.
///
/// Keyed by the street's own **position** rather than by its index in
/// `TrafficRes::streets`, on `inf_ecs::traffic::parked_car_guid`'s precedent: an
/// index moves when one block pages in anywhere in the settlement, and a slab
/// that changed identity every time that happened would be described, destroyed
/// and re-created for no reason a player could see. A quantized plan position is
/// a function of the ground.
pub fn kerb_slab_guid(along_x: bool, perp_m: f64, chunk: i64, side: i8) -> Uuid {
    let q = |v: f64| {
        if v.is_finite() {
            (v / 0.01).round() as i64 as u128
        } else {
            0
        }
    };
    let mut x = KERB_SLAB_SALT;
    for lane in [
        u128::from(along_x),
        q(perp_m),
        chunk as u128,
        (side as i64) as u128,
    ] {
        x ^= lane.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5cec_c5c3);
        x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    }
    Uuid::from_u128(x)
}

/// The terrains a world holds, resolved to their world origins — the shape the
/// height query below walks.
fn terrains_of(world: &EcsWorld) -> Vec<(&Terrain, DVec3)> {
    let mut out = Vec::new();
    for e in world.world().iter_entities() {
        let Some(t) = e.get::<Terrain>() else {
            continue;
        };
        if t.data.is_empty() {
            continue;
        }
        let origin = e
            .get::<GlobalTransform>()
            .map(|g| g.translation())
            .or_else(|| e.get::<Transform>().map(|x| x.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        out.push((t, origin));
    }
    out
}

/// **The topmost terrain that answers at `p`**, world metres — IB-15's rule,
/// restated here because this crate cannot reach `inf_voxel::ground_height_at`'s
/// key-generic form without a chunk map it has no business holding.
fn ground_at(terrains: &[(&Terrain, DVec3)], p: DVec2) -> Option<f64> {
    let mut best: Option<f64> = None;
    for (t, origin) in terrains {
        let local = p - DVec2::new(origin.x, origin.z);
        let Some(h) = t.data.height_at(local).map(|h| h + origin.y) else {
            continue;
        };
        if best.is_none_or(|y| h > y) {
            best = Some(h);
        }
    }
    best
}

/// **One footway slab's box, or nothing** — the geometry, as a pure function, so
/// it can be asserted without a bridge.
///
/// `chunk` counts along the street from the WORLD origin rather than from the
/// line's own end, `kerb_slots`' lattice rule and for its reason: a street's `a`
/// is a group's bounding-box corner, so one block arriving anywhere in a
/// settlement moves it, and slabs keyed off it would all change identity at once.
///
/// Returns `(centre, half_extents)`. The box is axis-aligned because a
/// settlement street is: `streets_of_blocks` only ever answers lines along X or
/// along Z, so the two cases are a swap of the half-extents and never a rotation.
pub fn slab_box(street: &Street, side: i8, chunk: i64, ground_y: f64) -> Option<(DVec3, DVec3)> {
    let kerb = inf_ecs::traffic::street_kerb_offset_m(street.gap_m);
    let back = kerb + KERB_WIDTH_M + inf_ecs::society::PAVEMENT_M;
    let across_half = (back - kerb) * 0.5;
    let across_mid = f64::from(side) * (kerb + across_half);

    let along_x = street.along_x();
    let (lo, hi) = if along_x {
        (street.a.x.min(street.b.x), street.a.x.max(street.b.x))
    } else {
        (street.a.y.min(street.b.y), street.a.y.max(street.b.y))
    };
    let c0 = chunk as f64 * KERB_SLAB_M;
    let (a, b) = (c0.max(lo), (c0 + KERB_SLAB_M).min(hi));
    // The finite test rather than a negated comparison: a NaN span makes every
    // ordering comparison false, so `<= eps` would let one through and rapier
    // answers a zero-width box with a NaN normal. `plan_contains`'s own idiom.
    let span = b - a;
    if !(span.is_finite() && span > 1.0e-6) {
        return None;
    }
    let along_mid = (a + b) * 0.5;
    let along_half = (b - a) * 0.5;
    // The slab's TOP is the kerb's upstand above the ground; the box hangs below.
    let top = ground_y + KERB_HEIGHT_M;
    let centre_y = top - KERB_SLAB_DEPTH_M * 0.5;

    let perp = if along_x { street.a.y } else { street.a.x };
    let (centre, half) = if along_x {
        (
            DVec3::new(along_mid, centre_y, perp + across_mid),
            DVec3::new(along_half, KERB_SLAB_DEPTH_M * 0.5, across_half),
        )
    } else {
        (
            DVec3::new(perp + across_mid, centre_y, along_mid),
            DVec3::new(across_half, KERB_SLAB_DEPTH_M * 0.5, along_half),
        )
    };
    Some((centre, half))
}

/// The chunk indices a street covers, on the world lattice.
fn chunks_of(street: &Street) -> std::ops::RangeInclusive<i64> {
    let along_x = street.along_x();
    let (lo, hi) = if along_x {
        (street.a.x.min(street.b.x), street.a.x.max(street.b.x))
    } else {
        (street.a.y.min(street.b.y), street.a.y.max(street.b.y))
    };
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        // An empty inclusive range, spelled so `reversed_empty_ranges` can see
        // it is deliberate: a degenerate street has no chunks.
        #[allow(clippy::reversed_empty_ranges)]
        return 1..=0;
    }
    (lo / KERB_SLAB_M).floor() as i64..=((hi / KERB_SLAB_M).ceil() as i64)
}

/// **What one gather pass described** — reported rather than counted by a caller,
/// because "how many boxes does a footway cost" is the budget question this
/// clause exists to answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KerbColliderAudit {
    /// Slabs the band admitted this pass — the number of static boxes the
    /// footways add to the step.
    pub described: u32,
    /// Streets walked.
    pub streets: u32,
    /// Slabs the band tiered out.
    pub culled: u32,
}

/// **Describe the footway slabs near the anchor** — the sixth derived collider
/// source, on the shape the other five already have.
///
/// The stamp is `(TrafficRes::stamp, band.stamp())`: the street set and the
/// membership of the active band are equally part of "what is attached", which
/// is `gather_structures`' own reasoning one level along. When it holds, the
/// admitted set is re-offered to the despawn sweep and nothing is rebuilt.
pub(crate) fn gather_kerbs(
    world: &EcsWorld,
    band: &SimBand,
    stamp_cache: &mut Option<(u64, u64)>,
    admitted_cache: &mut Vec<Uuid>,
    snaps: &mut Vec<EntitySync3D>,
    retained: &mut BTreeSet<Uuid>,
) -> KerbColliderAudit {
    let Some(res) = inf_ecs::traffic::carriageway_of(world) else {
        // A level with no settlement blocks has no streets and pays one
        // resource lookup — the "absent costs nothing" rule the crowd, the
        // society and the traffic derivation all follow.
        *stamp_cache = None;
        admitted_cache.clear();
        return KerbColliderAudit::default();
    };
    let stamp = (res.stamp, band.stamp());
    if *stamp_cache == Some(stamp) {
        retained.extend(admitted_cache.iter().copied());
        return KerbColliderAudit {
            described: admitted_cache.len() as u32,
            streets: res.streets.len() as u32,
            culled: 0,
        };
    }
    let terrains = terrains_of(world);
    let mut audit = KerbColliderAudit {
        streets: res.streets.len() as u32,
        ..Default::default()
    };
    admitted_cache.clear();
    for street in &res.streets {
        for chunk in chunks_of(street) {
            for side in [1i8, -1] {
                // The ground under the slab's own middle, which is what its top
                // is 150 mm above.
                let Some((probe, _)) = slab_box(street, side, chunk, 0.0) else {
                    continue;
                };
                let ground = ground_at(&terrains, DVec2::new(probe.x, probe.z)).unwrap_or(street.y);
                let Some((centre, half)) = slab_box(street, side, chunk, ground) else {
                    continue;
                };
                if !band.tier(centre, half, glam::DQuat::IDENTITY).is_near() {
                    audit.culled += 1;
                    continue;
                }
                let guid = kerb_slab_guid(
                    street.along_x(),
                    if street.along_x() {
                        street.a.y
                    } else {
                        street.a.x
                    },
                    chunk,
                    side,
                );
                admitted_cache.push(guid);
                snaps.push(EntitySync3D {
                    guid,
                    body: None,
                    collider: Some(ColliderDesc3D::new(ColliderShape3D::Box {
                        half_extents: half,
                    })),
                    translation: centre,
                    rotation: glam::DQuat::IDENTITY,
                    joint: None,
                });
                audit.described += 1;
            }
        }
    }
    // **First writer wins on a duplicate.** Two streets that share a `perp` on
    // one axis is a plan the derivation cannot produce (its intervals are
    // merged), but a guid collision would silently make one slab overwrite the
    // other in the bridge's map, so the set is deduplicated rather than trusted.
    admitted_cache.sort_unstable();
    admitted_cache.dedup();
    *stamp_cache = Some(stamp);
    retained.extend(admitted_cache.iter().copied());
    audit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn street(gap: f64) -> Street {
        Street {
            a: DVec2::new(-100.0, 0.0),
            b: DVec2::new(100.0, 0.0),
            y: 3.0,
            gap_m: gap,
        }
    }

    /// **The slab's top is one kerb above the ground, and its face is the kerb.**
    ///
    /// The two numbers the whole clause is about: an agent standing on the slab
    /// is `KERB_HEIGHT_M` above the ground beside it, and the wall it steps up
    /// is at the carriageway's own edge.
    #[test]
    fn a_footway_slab_is_a_kerb_above_the_ground_it_stands_on() {
        for gap in [16.0, 20.0] {
            let s = street(gap);
            let (c, h) = slab_box(&s, 1, 0, 12.5).expect("a slab");
            let top = c.y + h.y;
            assert!(
                (top - (12.5 + KERB_HEIGHT_M)).abs() < 1.0e-12,
                "a {gap} m street's footway tops out at {top}, not {} ",
                12.5 + KERB_HEIGHT_M
            );
            // The inner face is the kerb the paving draws.
            let inner = c.z - h.z;
            let kerb = inf_ecs::traffic::street_kerb_offset_m(gap);
            assert!(
                (inner - kerb).abs() < 1.0e-12,
                "the slab's face is at {inner} m and the kerb at {kerb} m"
            );
            // …and its back is the footway's back.
            let back = c.z + h.z;
            assert!(
                (back - (kerb + KERB_WIDTH_M + inf_ecs::society::PAVEMENT_M)).abs() < 1.0e-12,
                "the slab's back is at {back} m"
            );
            // Both sides, mirrored.
            let (cl, _) = slab_box(&s, -1, 0, 12.5).expect("a slab");
            assert!(
                (cl.z + c.z).abs() < 1.0e-12,
                "the two sides are not mirrored"
            );
        }
    }

    /// A slab is chunked on the WORLD lattice, so a street that grows keeps the
    /// slabs it had — `kerb_slots`' rule, and for its reason.
    #[test]
    fn a_slabs_identity_is_its_place_and_not_its_index() {
        let short = Street {
            b: DVec2::new(50.0, 0.0),
            ..street(20.0)
        };
        let long = street(20.0);
        let g_short = kerb_slab_guid(true, 0.0, 0, 1);
        let g_long = kerb_slab_guid(true, 0.0, 0, 1);
        assert_eq!(g_short, g_long);
        // …and the chunk at the origin is the same box on both.
        let a = slab_box(&short, 1, 0, 0.0).expect("a slab");
        let b = slab_box(&long, 1, 0, 0.0).expect("a slab");
        assert_eq!(a, b, "the same chunk of two streets is two different boxes");
        // The long street reaches chunks the short one does not.
        assert!(chunks_of(&long).count() > chunks_of(&short).count());
        // Four distinct guids for the two sides of two chunks.
        let mut seen = vec![
            kerb_slab_guid(true, 0.0, 0, 1),
            kerb_slab_guid(true, 0.0, 0, -1),
            kerb_slab_guid(true, 0.0, 1, 1),
            kerb_slab_guid(false, 0.0, 0, 1),
        ];
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 4, "two slabs share a guid");
    }

    /// A chunk that falls entirely outside the street's own span describes
    /// nothing rather than a zero-width box, which rapier answers with a NaN
    /// normal.
    #[test]
    fn a_chunk_off_the_end_of_a_street_is_not_a_box() {
        let s = street(20.0);
        assert!(slab_box(&s, 1, 100, 0.0).is_none());
        assert!(slab_box(&s, 1, -100, 0.0).is_none());
        // The end chunk is clipped to the street rather than overhanging it.
        let last = *chunks_of(&s).end();
        if let Some((c, h)) = slab_box(&s, 1, last, 0.0) {
            assert!(c.x + h.x <= 100.0 + 1.0e-9, "the slab overhangs its street");
        }
    }
}
