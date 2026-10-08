//! **Street furniture, the ECS half** (wave PAR1b).
//!
//! What a level's lamp posts, signals, poles, cables and kerbside pieces are
//! derived FROM — the streets the blocks imply, the signalised junctions among
//! them, the blocks' exterior doors and the ground — and the one door they are
//! written back THROUGH: each piece is owned by the block whose frontage it
//! stands in front of, and appended to that block's population with
//! `PcgVolume::set_furniture`. From there every existing door carries it: both
//! projectors draw it as scatter and push its lights, the physics bridge makes
//! its solids static colliders (ungrouped, `Near`-only, dormant like any other
//! scenery), and none of it reaches a level's bytes.
//!
//! The derivation between the two halves is `inf_pcg::street::furnish`, which
//! this crate cannot name; both hosts join the three calls inside one fenced
//! body (`furnish_streets`, compared character for character by
//! `projector_mirror`).
//!
//! # When it runs
//!
//! Whenever a host has just changed which blocks are resident (a load, a cell
//! activation, an editor stream tick). [`furnish_inputs`] answers `None` when
//! every resident block is already furnished against the current
//! [`furniture_key`], so a
//! call that has nothing to do costs one walk.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use crate::components::{
    GlobalTransform, Guid, PcgVolume, ScatteredInstance, ScatteredLight, ScatteredSolid, Terrain,
    Transform,
};
use crate::traffic::{signal_junctions, streets_of, SignalJunction, Street};
use crate::world::EcsWorld;

/// **How far a piece may stand from the block that owns it**, metres: half the
/// widest reserve the derivation admits (`MAX_STREET_GAP_M` is 40) less a
/// metre — a piece on the furniture line is at most `gap / 2` from its block's
/// edge, and a piece further than this from every block stands in front of no
/// frontage.
pub const OWNER_REACH_M: f64 = 19.0;

/// One resident block, as ownership needs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockSlot {
    /// The volume's guid.
    pub guid: Uuid,
    /// Its centre, world XZ.
    pub centre: DVec2,
    /// Its half-extent, world XZ.
    pub half: DVec2,
}

/// **Everything the furniture is derived from**, read out of a world before
/// anything is mutated.
pub struct FurnishInputs<'w> {
    /// The fold every block will be stamped with — see [`furniture_key`].
    pub key: u64,
    /// The streets, in `streets_of` order.
    pub streets: Vec<Street>,
    /// The signalised junctions among them.
    pub junctions: Vec<SignalJunction>,
    /// Every exterior door's hinge, plan position.
    pub doors: Vec<DVec2>,
    /// Every resident block, in `Guid` order.
    pub blocks: Vec<BlockSlot>,
    terrains: Vec<(&'w Terrain, DVec3)>,
}

impl FurnishInputs<'_> {
    /// **The ground at `(x, z)`** — the topmost terrain that answers (IB-15's
    /// rule, `inf_physics::d3::kerb`'s own), or `None` before it pages in.
    pub fn ground_at(&self, x: f64, z: f64) -> Option<f64> {
        let mut best: Option<f64> = None;
        for (t, origin) in &self.terrains {
            let local = DVec2::new(x - origin.x, z - origin.z);
            let Some(h) = t.data.height_at(local).map(|h| h + origin.y) else {
                continue;
            };
            if best.is_none_or(|y| h > y) {
                best = Some(h);
            }
        }
        best
    }

    /// **The block whose frontage `foot` stands in front of** — the nearest
    /// resident block rectangle within [`OWNER_REACH_M`], the first in `Guid`
    /// order on a tie.
    pub fn owner_of(&self, foot: DVec3) -> Option<Uuid> {
        let mut best: Option<(f64, Uuid)> = None;
        for b in &self.blocks {
            let dx = ((foot.x - b.centre.x).abs() - b.half.x).max(0.0);
            let dz = ((foot.z - b.centre.y).abs() - b.half.y).max(0.0);
            let d = (dx * dx + dz * dz).sqrt();
            if d <= OWNER_REACH_M && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, b.guid));
            }
        }
        best.map(|(_, g)| g)
    }
}

/// Every resident block (a volume that offers a resident — the set
/// `streets_of` derives the streets from), in `Guid` order.
fn resident_blocks(world: &EcsWorld) -> Vec<(BlockSlot, u64)> {
    let mut out = Vec::new();
    for e in world.world().iter_entities() {
        let (Some(g), Some(v)) = (e.get::<Guid>(), e.get::<PcgVolume>()) else {
            continue;
        };
        if v.residents.is_empty() {
            continue;
        }
        let c = e
            .get::<GlobalTransform>()
            .map(|t| t.translation())
            .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        if !c.is_finite() {
            continue;
        }
        out.push((
            BlockSlot {
                guid: g.0,
                centre: DVec2::new(c.x, c.z),
                half: DVec2::new(v.extent.x, v.extent.y),
            },
            v.furniture.key,
        ));
    }
    out.sort_by_key(|(b, _)| b.guid);
    out
}

/// **The fold a block's furniture is current against**: the block set (the
/// street derivation's own stamp), every exterior door, and how much terrain
/// is resident — the four things a piece's existence and height depend on.
/// Never `0`, which is "never furnished".
pub fn furniture_key(world: &EcsWorld) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ crate::traffic::block_stamp(world);
    let mut fold = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for e in world.world().iter_entities() {
        if let Some(v) = e.get::<PcgVolume>() {
            if v.residents.is_empty() {
                continue;
            }
            fold(v.doorways.iter().filter(|d| d.exterior).count() as u64);
        }
        if let Some(t) = e.get::<Terrain>() {
            fold(t.data.tile_count() as u64);
            fold(t.data.coarse_tile_count() as u64);
        }
    }
    h.max(1)
}

/// **The inputs, or `None` when every resident block is already furnished
/// against the current key** — the cheap answer a host gets on a call that has
/// nothing to do.
pub fn furnish_inputs(world: &EcsWorld) -> Option<FurnishInputs<'_>> {
    // **The measuring door** (wave PAR1b): with `INF_NO_STREET_FURNITURE`
    // set, no block is ever furnished — the same binary with and without the
    // wave's furniture, which is how the instrument prices it (a frame row, a
    // triangle count, an exposure). Read here, in the one Ring-0 door both
    // hosts pass, so neither host can be measured differently from the other.
    if std::env::var_os("INF_NO_STREET_FURNITURE").is_some() {
        return None;
    }
    let blocks = resident_blocks(world);
    if blocks.is_empty() {
        return None;
    }
    let key = furniture_key(world);
    if blocks.iter().all(|(_, k)| *k == key) {
        return None;
    }
    let streets = streets_of(world);
    let junctions = signal_junctions(&streets);
    let mut doors = Vec::new();
    let mut terrains = Vec::new();
    for e in world.world().iter_entities() {
        if let Some(v) = e.get::<PcgVolume>() {
            if !v.residents.is_empty() {
                doors.extend(
                    v.doorways
                        .iter()
                        .filter(|d| d.exterior && d.hinge.is_finite())
                        .map(|d| DVec2::new(d.hinge.x, d.hinge.z)),
                );
            }
        }
        if let Some(t) = e.get::<Terrain>() {
            if t.data.is_empty() {
                continue;
            }
            let origin = e
                .get::<GlobalTransform>()
                .map(|g| g.translation())
                .or_else(|| e.get::<Transform>().map(|x| x.translation.to_dvec3()))
                .unwrap_or(DVec3::ZERO);
            terrains.push((t, origin));
        }
    }
    Some(FurnishInputs {
        key,
        streets,
        junctions,
        doors,
        blocks: blocks.into_iter().map(|(b, _)| b).collect(),
        terrains,
    })
}

/// One block's furniture, in the ECS's mirror types.
pub type BlockFurniture = (
    Vec<ScatteredInstance>,
    Vec<ScatteredSolid>,
    Vec<ScatteredLight>,
);

/// **Write the furniture back** — every resident block gets its own pieces (or
/// none) through `PcgVolume::set_furniture`, stamped with `key`. Answers how
/// many blocks' populations actually changed.
pub fn apply_furniture(
    world: &mut EcsWorld,
    key: u64,
    mut per: BTreeMap<Uuid, BlockFurniture>,
) -> usize {
    let targets: Vec<(Uuid, crate::Entity)> = world
        .world()
        .iter_entities()
        .filter_map(|e| {
            let g = e.get::<Guid>()?;
            let v = e.get::<PcgVolume>()?;
            (!v.residents.is_empty()).then_some((g.0, e.id()))
        })
        .collect();
    let mut changed = 0usize;
    let mut sorted = targets;
    sorted.sort_by_key(|(g, _)| *g);
    let mut footprints = FurnitureFootprints::default();
    for (g, e) in sorted {
        let (inst, solids, lights) = per.remove(&g).unwrap_or_default();
        if let Some(mut vol) = world.world_mut().get_mut::<PcgVolume>(e) {
            let before = vol.structures_gen;
            vol.set_furniture(key, inst, solids, lights);
            changed += usize::from(vol.structures_gen != before);
            for s in tail(&vol).1 {
                footprints.push(s);
            }
        }
    }
    // PAR1b.2: the crowd's obstacle term reads the pieces' plan footprints
    // from one index rebuilt here, the one door every furnishing passes.
    if footprints.boxes.is_empty() {
        world.world_mut().remove_resource::<FurnitureFootprints>();
    } else {
        world.world_mut().insert_resource(footprints);
    }
    if changed > 0 {
        world.mark_dirty();
    }
    changed
}

/// **How far clear of a piece of furniture a crowd agent keeps**, metres,
/// beyond its own radius (wave PAR1b.2) — a hand's breadth.
pub const FURNITURE_CLEARANCE_M: f64 = 0.05;

/// The footprint index's cell, metres.
const FOOTPRINT_CELL_M: f64 = 8.0;

/// **Every furniture solid's plan footprint, indexed** (wave PAR1b.2) — what
/// [`clear_of_furniture`] reads. Rebuilt by [`apply_furniture`] in block-`Guid`
/// order, so both hosts hold the same index; absent on a level with no
/// furniture, and a bevy resource, so nothing of it reaches a level's bytes.
#[derive(bevy_ecs::prelude::Resource, Debug, Clone, Default, PartialEq)]
pub struct FurnitureFootprints {
    /// `(centre x, centre z, half x, half z)` per solid (every street piece's
    /// solids are axis-aligned in plan).
    pub boxes: Vec<[f64; 4]>,
    grid: BTreeMap<(i64, i64), Vec<u32>>,
}

impl FurnitureFootprints {
    fn push(&mut self, s: &ScatteredSolid) {
        let (c, h) = (s.center, s.half_extents);
        if !(c.is_finite() && h.is_finite()) {
            return;
        }
        let i = self.boxes.len() as u32;
        self.boxes.push([c.x, c.z, h.x.abs(), h.z.abs()]);
        let cell = |v: f64| (v / FOOTPRINT_CELL_M).floor() as i64;
        for gx in cell(c.x - h.x.abs())..=cell(c.x + h.x.abs()) {
            for gz in cell(c.z - h.z.abs())..=cell(c.z + h.z.abs()) {
                self.grid.entry((gx, gz)).or_default().push(i);
            }
        }
    }

    /// The boxes whose cells a disc of `reach` at `(x, z)` touches, each once,
    /// in index order.
    fn near(&self, x: f64, z: f64, reach: f64) -> Vec<u32> {
        let cell = |v: f64| (v / FOOTPRINT_CELL_M).floor() as i64;
        let mut out: Vec<u32> = Vec::new();
        for gx in cell(x - reach)..=cell(x + reach) {
            for gz in cell(z - reach)..=cell(z + reach) {
                if let Some(v) = self.grid.get(&(gx, gz)) {
                    out.extend_from_slice(v);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// **Where an agent of `radius` at `p` stands clear of every piece of street
/// furniture** (wave PAR1b.2, the crowd's obstacle term): pushed straight out
/// of each footprint it would overlap (radius + [`FURNITURE_CLEARANCE_M`])
/// from the footprint's nearest point — a disc grazing a post slides round it,
/// a disc inside a box (a clock-placed agent whose route runs through a bus
/// shelter's panel) leaves through the nearest face. Two passes, so a push off
/// one piece into another is answered. A pure function of `p` and the index:
/// `p` itself on a level with no furniture (bit for bit), and only `x`/`z`
/// move.
pub fn clear_of_furniture(world: &EcsWorld, p: DVec3, radius: f64) -> DVec3 {
    let Some(f) = world.world().get_resource::<FurnitureFootprints>() else {
        return p;
    };
    if !p.is_finite() {
        return p;
    }
    let need = radius.max(0.0) + FURNITURE_CLEARANCE_M;
    let mut q = p;
    for _ in 0..2 {
        let mut moved = false;
        for i in f.near(q.x, q.z, need + 2.0) {
            let [cx, cz, hx, hz] = f.boxes[i as usize];
            let (dx, dz) = (q.x - cx, q.z - cz);
            let (nx, nz) = (dx.clamp(-hx, hx), dz.clamp(-hz, hz));
            let (ox, oz) = (dx - nx, dz - nz);
            let d2 = ox * ox + oz * oz;
            if d2 >= need * need {
                continue;
            }
            moved = true;
            if d2 > 1e-18 {
                let d = d2.sqrt();
                q.x += ox / d * (need - d);
                q.z += oz / d * (need - d);
            } else {
                // Inside the box: out through the nearest face.
                let (px, pz) = (hx - dx.abs(), hz - dz.abs());
                if px < pz {
                    q.x = cx + dx.signum() * (hx + need);
                    if dx == 0.0 {
                        q.x = cx + hx + need;
                    }
                } else {
                    q.z = cz + dz.signum() * (hz + need);
                    if dz == 0.0 {
                        q.z = cz + hz + need;
                    }
                }
            }
        }
        if !moved {
            break;
        }
    }
    q
}

/// **The furniture a world carries, counted** — a world-side census over every
/// block's furniture tail (wave PAR1b): `(blocks, instances, solids, lights)`.
pub fn census(world: &EcsWorld) -> (usize, usize, usize, usize) {
    let mut out = (0, 0, 0, 0);
    for e in world.world().iter_entities() {
        if let Some(v) = e.get::<PcgVolume>() {
            if v.furniture.key == 0 {
                continue;
            }
            out.0 += 1;
            out.1 += v.furniture.instances;
            out.2 += v.furniture.solids;
            out.3 += v.furniture.lights;
        }
    }
    out
}

/// The furniture tail of one volume — the trailing slices
/// [`PcgVolume::furniture`] counts.
pub fn tail(v: &PcgVolume) -> (&[ScatteredInstance], &[ScatteredSolid], &[ScatteredLight]) {
    let f = v.furniture;
    (
        &v.evaluated[v.evaluated.len().saturating_sub(f.instances)..],
        &v.structures[v.structures.len().saturating_sub(f.solids)..],
        &v.lights[v.lights.len().saturating_sub(f.lights)..],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn post(x: f64, z: f64) -> ScatteredSolid {
        ScatteredSolid {
            center: DVec3::new(x, 4.5, z),
            half_extents: DVec3::new(0.13, 4.5, 0.13),
            rotation: glam::DQuat::IDENTITY,
        }
    }

    /// **An agent walking a line through a post walks round it** (wave
    /// PAR1b.2): a clock-placed agent stepped along a line that runs 0.2 m
    /// from a post's centre (the 16 m street's ring past its post line) stays
    /// its radius + `FURNITURE_CLEARANCE_M` clear of the footprint at every
    /// step, passes within 0.6 m of it, and is moved no more than the overlap
    /// asks; a world with no furniture moves nobody, bit for bit. Mutation:
    /// `clear_of_furniture` answering `p` -> the closest sample is inside the
    /// post's radius, red.
    #[test]
    fn an_agent_walking_through_a_post_walks_round_it() {
        let mut world = EcsWorld::new();
        let p0 = DVec3::new(3.0, 1.0, 2.0);
        assert_eq!(clear_of_furniture(&world, p0, 0.3), p0);
        let mut f = FurnitureFootprints::default();
        f.push(&post(0.0, 6.22));
        world.world_mut().insert_resource(f);
        let r = 0.3;
        let need = r + FURNITURE_CLEARANCE_M;
        let mut nearest = f64::INFINITY;
        for i in -40..=40 {
            let raw = DVec3::new(f64::from(i) * 0.05, 1.0, 6.0);
            let q = clear_of_furniture(&world, raw, r);
            let dx = (q.x.abs() - 0.13).max(0.0);
            let dz = ((q.z - 6.22).abs() - 0.13).max(0.0);
            let d = (dx * dx + dz * dz).sqrt();
            nearest = nearest.min(d);
            assert!(
                d >= need - 1e-9,
                "at x {:.2}: {d:.3} m from the post",
                raw.x
            );
            assert!(
                (q - raw).length() <= need + 0.2,
                "moved {:.3} m",
                (q - raw).length()
            );
            assert_eq!(q.y, raw.y, "the height moved");
        }
        assert!(
            nearest < 0.6,
            "the agent never came near the post ({nearest:.2})"
        );
    }
}
