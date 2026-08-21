//! **Doors, where they meet the world** (island wave I6): the leaf's collider,
//! the swing's own blocking probe, and the door half of the interaction
//! candidate list.
//!
//! # Why this is here and not in `inf-ecs`
//!
//! The same reason `d3::interact` is: two of the three things a door needs from
//! the world are physics. Whether a swing is **blocked** is a shape cast, and
//! the leaf's collider is a rapier body. The *rules* — where the leaf is, what
//! the lock costs, what a kick or a crash carries — are all in
//! [`inf_ecs::door`], pure, and both an authored door and a grammar doorway are
//! ranked and swung by that one set.
//!
//! # The leaf is a SYNTHETIC body, for authored doors too
//!
//! A door's entity `Transform` is its **hinge** and never moves. The leaf is a
//! kinematic box under [`door_leaf_guid`], created by [`gather_doors`] in the
//! same walk `gather_structures` runs in and posed by [`step_doors`].
//!
//! An author could instead have put a `Collider3D` on the door entity and had
//! the step write its transform — and that is exactly the design this refuses,
//! because a grammar doorway has no entity to carry one. Two mechanisms would be
//! two answers to "where is this door's collider", and the second one would only
//! exist for half the doors in a level. One synthetic body, both kinds.
//!
//! **An authored `Door` should therefore carry no `Collider3D` of its own**; if
//! it does, that collider sits at the hinge and is nothing to do with the leaf.

use std::collections::BTreeSet;

use glam::{DQuat, DVec3};
use uuid::Uuid;

use inf_ecs::band::SimBand;
use inf_ecs::components::MovementMode;
use inf_ecs::door::{self, Door, DoorPlacement, DoorSide, DoorState};
use inf_ecs::interact::{InteractCandidate, InteractVerb};
use inf_ecs::world::EcsWorld;
use inf_math::Tier;

use super::ecs::{BodyDesc3D, EntitySync3D, PhysicsBridge3D};
use super::world::{BodyKind3D, ColliderDesc3D, ColliderShape3D};
use super::ColliderId3D;

/// Salt for [`door_leaf_guid`]. A different constant from every other synthetic
/// space in this bridge, because a leaf that answered to a structure's guid
/// would make one of them an *update* of the other.
const DOOR_LEAF_SALT: u128 = 0x6006_0100_444f_4f52_4c45_4146_5f42_4f58;

/// The synthetic identity of `door`'s swinging leaf.
///
/// The [`super::ecs::pcg_structure_guid`] rule with its own salt, and stated
/// once so a debug view or a save hook that ever names one names it the same
/// way.
pub fn door_leaf_guid(door: Uuid) -> Uuid {
    let mut x = door.as_u128() ^ DOOR_LEAF_SALT;
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

/// What the prompt calls the verb on a door, by what the door is doing.
///
/// Built here rather than at the call site so the wording exists once — the
/// [`inf_ecs::interact::prompt_text`] discipline, one level down.
pub fn door_label(placement: &DoorPlacement, state: &DoorState, from: DoorSide) -> String {
    if state.locked && !state.lock_broken {
        // **"Locked" is the label, not a missing prompt.** A door that offered
        // nothing would let the prompt fall through to whatever is behind it,
        // and the player would have no way to tell a locked door from a wall.
        return format!("{} (Locked)", placement.label);
    }
    if from == placement.spec.lock_side && !state.lock_broken {
        // The lock verb is offered on ONE face — the owner's "lockable from the
        // inside" — and it is offered alongside the open/close rather than
        // instead of it, because a door you can lock is still a door you can
        // walk through.
        let word = if state.locked { "unlock" } else { "lock" };
        return format!("{} ({word} with the same key)", placement.label);
    }
    placement.label.clone()
}

/// **Every door in the world that a character could act on**, in `Guid` order.
///
/// Authored [`Door`] entities today; the building grammar's doorways join this
/// list through the same function when a volume carries them.
///
/// `O(doors)`, and `O(1)` on a level with none.
pub fn placements(world: &EcsWorld) -> Vec<DoorPlacement> {
    door::doors_in_world(world)
}

/// **The door half of the interaction candidate list.**
///
/// Every door is a candidate whatever its lock is doing — a locked one carries
/// the "(Locked)" label rather than being hidden, which is
/// `Interactable::enabled`'s own lesson: a resolver that skipped it would let
/// the prompt fall through to whatever is behind it.
///
/// `feet` decides which face the character is on and therefore whether the lock
/// verb is in the label at all.
pub fn candidates(world: &EcsWorld, feet: DVec3) -> Vec<InteractCandidate> {
    let field = door::door_field(world);
    let mut out: Vec<InteractCandidate> = placements(world)
        .into_iter()
        .map(|p| {
            let state = field
                .map(|f| f.get(p.guid, &p.spec))
                .unwrap_or_else(|| DoorState::fresh(&p.spec));
            let side = p.spec.side_of(p.hinge, feet);
            InteractCandidate {
                guid: p.guid,
                verb: InteractVerb::Use,
                label: door_label(&p, &state, side),
                position: door::prompt_position(&p),
                range_m: door::DOOR_REACH_M,
                view_cone_deg: door::DOOR_VIEW_CONE_DEG,
            }
        })
        .collect();
    out.sort_by_key(|c| c.guid);
    out
}

/// The one door whose guid is `guid`, if the world has it.
pub fn placement_of(world: &EcsWorld, guid: Uuid) -> Option<DoorPlacement> {
    placements(world).into_iter().find(|p| p.guid == guid)
}

/// **The E key on a door** — the consumer the one interaction site calls when
/// the hit it resolved is a door.
///
/// Returns the verdict, which is a value in every case: an unusable door, a
/// locked one, a lock toggled, a leaf opening. The caller does not need to know
/// which; it needs to know whether anything happened, and the verdict says.
///
/// **What the press does is decided by the same `lock_side` the prompt read**,
/// so what the player is told and what the press does cannot come apart — the
/// property I5 built the one resolution site for.
pub fn use_door(world: &mut EcsWorld, guid: Uuid, feet: DVec3) -> door::DoorVerdict {
    let Some(p) = placement_of(world, guid) else {
        return door::DoorVerdict::Unusable;
    };
    let side = p.spec.side_of(p.hinge, feet);
    let field = door::door_field_mut(world);
    let state = field.entry(guid, &p.spec);
    // The lock verb wins on its own face while the door is shut and locked:
    // that is what "locked from the inside" has to mean for the person who
    // locked it, or they could never get out. On the other face, or once the
    // lock is broken, the same press is the ordinary open/close.
    if side == p.spec.lock_side && !state.lock_broken {
        if state.locked {
            return door::set_locked(&p.spec, state, side, false);
        }
        if !state.is_open(&p.spec) {
            return door::set_locked(&p.spec, state, side, true);
        }
    }
    door::toggle(&p.spec, state)
}

/// **A blow against a door** — the kick and the crash, through one function.
///
/// `energy_j` is kinetic energy in joules, produced by
/// [`inf_ecs::door::kick_energy_j`] or [`inf_ecs::door::breach_energy_j`]; what
/// it is compared against is the lock's own [`inf_ecs::door::DoorSpec::lock_energy_j`],
/// which is the P22 bond rule. **This is the one door for both**, and the reason
/// it can be: neither the kick nor the crash has a damage number of its own —
/// both have a mass and a speed.
pub fn strike_door(world: &mut EcsWorld, guid: Uuid, energy_j: f64) -> door::BreakVerdict {
    let Some(p) = placement_of(world, guid) else {
        return door::BreakVerdict {
            broke: false,
            absorbed_j: 0.0,
            remainder_j: 0.0,
            required_j: 0.0,
        };
    };
    let field = door::door_field_mut(world);
    let state = field.entry(guid, &p.spec);
    let verdict = door::try_break(&p.spec, state, energy_j);
    door::apply_break(&p.spec, state, &verdict);
    verdict
}

/// How close a body must be to a closed door to breach it, metres.
///
/// A sprint covers 0.108 m in a fixed step, so this is about nine steps of
/// opportunity — wide enough that the breach cannot be missed between two
/// frames, narrow enough that it is a door the character is arriving at rather
/// than one across the room.
pub const BREACH_REACH_M: f64 = 1.2;

/// How closely the body's direction of travel must line up with the door,
/// as a cosine. `0.5` is sixty degrees either side: running past a door at a
/// glancing angle is running past it.
pub const BREACH_ALIGNMENT: f64 = 0.5;

/// **What a crash-through did.**
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreachOutcome {
    /// The door.
    pub door: Uuid,
    /// The speed the body arrived at, m/s.
    pub speed_in_mps: f64,
    /// The speed it left with, m/s.
    pub speed_out_mps: f64,
    /// Joules the lock absorbed.
    pub absorbed_j: f64,
    /// Joules the lock needed.
    pub required_j: f64,
    /// Whether it went through.
    pub broke: bool,
}

/// **The crash-through** — a sprinting, sliding or diving body against a closed
/// door.
///
/// The decision is a **pure function of sim state**: the mode, the speed, the
/// geometry, and the lock's own P22 bond energy. There is no random term and no
/// clock in it, which is what lets a breach trace be byte-identical between a
/// cooked pack and a PIE payload.
///
/// Two gates, and they are different questions:
///
/// * the **speed** gate ([`inf_ecs::door::BREACH_SPEED_MPS`]) is whether the
///   engine treats the contact as an attempt at all — the owner's
///   "sprinting/sliding/diving" expressed as a number, sitting above the run
///   speed and below the sprint;
/// * the **energy** gate is whether the lock gives, and it is the same
///   comparison [`strike_door`] makes for a kick. A stouter lock refuses a
///   sprint; that is the lock strength deciding, exactly as the mandate says.
///
/// A body below the speed gate, or facing the wrong way, or in a mode that
/// cannot breach, answers `None` and collides with the leaf normally — because
/// the leaf is a real collider and nothing here removed it.
pub fn try_breach(
    world: &mut EcsWorld,
    feet: DVec3,
    velocity: DVec3,
    mode: MovementMode,
    mass_kg: f64,
) -> Option<BreachOutcome> {
    if !matches!(
        mode,
        MovementMode::Grounded
            | MovementMode::Crouch
            | MovementMode::Slide
            | MovementMode::Roll
            | MovementMode::Dive
    ) {
        return None;
    }
    let planar = DVec3::new(velocity.x, 0.0, velocity.z);
    let speed = planar.length();
    if !speed.is_finite() || speed < door::BREACH_SPEED_MPS {
        return None;
    }
    let heading = planar / speed;
    // The nearest closed door in the way, by the same distance arithmetic every
    // other reach in this engine uses.
    let field = door::door_field(world);
    let mut best: Option<(f64, DoorPlacement)> = None;
    for p in placements(world) {
        let state = field
            .map(|f| f.get(p.guid, &p.spec))
            .unwrap_or_else(|| DoorState::fresh(&p.spec));
        if state.is_open(&p.spec) {
            continue;
        }
        let at = door::prompt_position(&p);
        let to = at - feet;
        let flat = DVec3::new(to.x, 0.0, to.z);
        let d = flat.length();
        if !d.is_finite() || d > BREACH_REACH_M {
            continue;
        }
        // Straight through it, not past it. A zero-length offset means the body
        // is already in the doorway, which counts.
        if d > 1e-9 && (flat / d).dot(heading) < BREACH_ALIGNMENT {
            continue;
        }
        // Strict `<` over a `Guid`-ordered walk — `placements` is sorted, so two
        // doors at the same distance resolve the same way in both hosts.
        if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
            best = Some((d, p));
        }
    }
    let (_, p) = best?;
    let energy = door::breach_energy_j(mass_kg, speed);
    let verdict = strike_door(world, p.guid, energy);
    // A door that is merely shut (not locked) opens for free and the body keeps
    // all of its speed; a locked one costs the lock's own joules.
    let out = if verdict.broke {
        // The leaf is now swinging; open it the rest of the way under power too,
        // so a body that breached at the very edge of the threshold does not
        // arrive at a leaf that has coasted to a stop half way.
        if let Some(f) = world.world_mut().get_resource_mut::<door::DoorField>() {
            if let Some(s) = f.into_inner().0.get_mut(&p.guid) {
                s.target_deg = p.spec.open_limit_deg;
            }
        }
        door::breach_exit_speed_mps(mass_kg, speed, verdict.absorbed_j)
    } else {
        speed
    };
    Some(BreachOutcome {
        door: p.guid,
        speed_in_mps: speed,
        speed_out_mps: out,
        absorbed_j: verdict.absorbed_j,
        required_j: verdict.required_j,
        broke: verdict.broke,
    })
}

/// What one fixed step of the door system did — the numbers a gate asserts on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DoorReport {
    /// How many doors the world holds.
    pub doors: u32,
    /// How many of them moved this step.
    pub moved: u32,
    /// How many were stopped by something in the way.
    pub blocked: u32,
    /// How many leaves have a live collider.
    pub leaves: u32,
}

/// How far ahead of the leaf's current pose the blocking probe sweeps, as a
/// multiple of the arc it will travel this step.
///
/// Slightly more than one, so a leaf about to arrive inside something stops on
/// the step before rather than on the step it is already overlapping — which is
/// the state a shape cast reports as `started_penetrating` and cannot measure a
/// distance out of.
const BLOCK_LOOKAHEAD: f64 = 1.35;

/// The smallest arc worth sweeping, degrees. Below this the leaf has effectively
/// stopped and a cast would be a query about nothing.
const BLOCK_MIN_ARC_DEG: f64 = 0.05;

/// **One fixed step of every door.**
///
/// Runs after the character movement step (so a door the E key opened *this*
/// step starts moving this step) and before the solver (so the leaf's collider
/// is where the state says it is when the solver runs). Both hosts call it; it
/// is the `step_character_movement` shape one system along.
///
/// Inert on a level with no doors: one `try_query_filtered` that answers `None`.
pub fn step_doors(world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, dt: f64) -> DoorReport {
    let places = placements(world);
    let mut report = DoorReport {
        doors: places.len() as u32,
        ..Default::default()
    };
    if places.is_empty() {
        return report;
    }
    // Which doors are blocked is asked BEFORE anything moves, against the world
    // as this step found it — so two doors swinging into each other get the same
    // answer whichever order the walk arrives in. The alternative (probe, move,
    // probe, move) makes the outcome a function of `Guid` order in a way the
    // player can see.
    let mut blocked: Vec<bool> = Vec::with_capacity(places.len());
    for p in &places {
        let state = door::door_field(world)
            .map(|f| f.get(p.guid, &p.spec))
            .unwrap_or_else(|| DoorState::fresh(&p.spec));
        blocked.push(is_blocked(bridge, p, &state, dt));
    }
    let mut poses: Vec<(Uuid, DVec3, DQuat)> = Vec::new();
    {
        let field = door::door_field_mut(world);
        for (p, blocked) in places.iter().zip(blocked.iter().copied()) {
            let state = field.entry(p.guid, &p.spec);
            if blocked {
                report.blocked += 1;
            }
            if door::advance(&p.spec, state, dt, blocked) {
                report.moved += 1;
                let (centre, yaw, _) = door::leaf_pose(p, state.open_deg);
                poses.push((
                    door_leaf_guid(p.guid),
                    centre,
                    DQuat::from_rotation_y(yaw.to_radians()),
                ));
            }
        }
    }
    // The pose write goes straight into the bridge rather than waiting for the
    // next sync, so the solver that runs immediately after this sees the leaf
    // where the state says it is. `gather_doors` computes the same pose from the
    // same state through the same function next step, so the two cannot
    // disagree — and `set_body_pose_if_moved` makes the second write free.
    for (leaf, centre, rot) in poses {
        if let Some(body) = bridge.body_of(leaf) {
            bridge.world_mut().set_body_pose_if_moved(body, centre, rot);
            report.leaves += 1;
        }
    }
    report
}

/// Whether something is in the way of this leaf's next arc.
///
/// A **sweep of the leaf's own box** along the tangent at its free edge, which
/// is the direction the most of the leaf is going. Not an overlap test: a leaf
/// that has already reached something is not the case that matters — the case
/// that matters is a leaf about to.
fn is_blocked(
    bridge: &mut PhysicsBridge3D,
    p: &DoorPlacement,
    state: &DoorState,
    dt: f64,
) -> bool {
    if !p.spec.is_usable() || !dt.is_finite() || dt <= 0.0 {
        return false;
    }
    // How far it will turn this step, whichever kind of swing it is on.
    let arc_deg = if state.powered {
        let delta = state.target_deg - state.open_deg;
        delta.abs().min(door::DOOR_DRIVE_DPS * dt) * delta.signum()
    } else {
        state.vel_dps * dt
    };
    if !arc_deg.is_finite() || arc_deg.abs() < BLOCK_MIN_ARC_DEG {
        return false;
    }
    let (centre, yaw, half) = door::leaf_pose(p, state.open_deg);
    // The tangent at the leaf's centre: perpendicular to the leaf, in the
    // direction of travel.
    let r = yaw.to_radians();
    let along = DVec3::new(inf_math::psin64(r), 0.0, inf_math::pcos64(r));
    let tangent = DVec3::new(along.z, 0.0, -along.x) * arc_deg.signum();
    // Arc length at the leaf's centre — half the width from the hinge.
    let reach = (arc_deg.abs().to_radians() * p.spec.width_m * 0.5) * BLOCK_LOOKAHEAD;
    let shape = ColliderShape3D::Box { half_extents: half };
    let rot = DQuat::from_rotation_y(r);
    let mut exclude: BTreeSet<ColliderId3D> = BTreeSet::new();
    if let Some(c) = bridge.collider_of(door_leaf_guid(p.guid)) {
        exclude.insert(c);
    }
    // The door's own hinge entity, if it has a collider at all, is not something
    // its leaf can be blocked by.
    if let Some(c) = bridge.collider_of(p.guid) {
        exclude.insert(c);
    }
    bridge
        .world_mut()
        .cast_shape(&shape, centre, rot, tangent, reach, &exclude)
        .is_some()
}

/// **Every door's leaf collider**, appended to the sync's snapshot.
///
/// Called from `sync_from_world_sim`'s one walk, beside `gather_structures`, and
/// on the same rule: a leaf that has not moved is **retained** rather than
/// re-described, so a level full of shut doors costs the sync nothing per step.
///
/// The band is the same one the structural colliders use — a door thirty
/// kilometres away is not something a character can walk into — and it fails
/// open exactly as the band does.
pub fn gather_doors(
    world: &EcsWorld,
    band: &SimBand,
    stamps: &mut std::collections::BTreeMap<Uuid, u64>,
    snaps: &mut Vec<EntitySync3D>,
    retained: &mut BTreeSet<Uuid>,
) {
    // The off path: no door in the world and none tracked is one compare.
    if stamps.is_empty() && !world.has_component::<Door>() {
        return;
    }
    let field = door::door_field(world);
    let mut live: BTreeSet<Uuid> = BTreeSet::new();
    for p in placements(world) {
        let state = field
            .map(|f| f.get(p.guid, &p.spec))
            .unwrap_or_else(|| DoorState::fresh(&p.spec));
        let (centre, yaw, half) = door::leaf_pose(&p, state.open_deg);
        if !band.tier(centre, half, DQuat::IDENTITY).is_near() {
            continue;
        }
        let leaf = door_leaf_guid(p.guid);
        live.insert(leaf);
        // The stamp is the leaf's own pose bits plus its size, so a door that is
        // standing still is retained and a door that is swinging is re-described
        // — which is the `gather_structures` rule with the moving case made the
        // exception rather than the rule.
        let stamp = pose_stamp(centre, yaw, half);
        if stamps.get(&leaf) == Some(&stamp) {
            retained.insert(leaf);
            continue;
        }
        stamps.insert(leaf, stamp);
        snaps.push(EntitySync3D {
            guid: leaf,
            // **Kinematic, not static.** A static body cannot be moved without
            // the broad phase treating it as a teleport, and a door is a thing
            // that moves under its own authority — which is exactly what
            // kinematic means.
            body: Some(BodyDesc3D {
                kind: BodyKind3D::Kinematic,
                ..Default::default()
            }),
            collider: Some(ColliderDesc3D::new(ColliderShape3D::Box {
                half_extents: half,
            })),
            translation: centre,
            rotation: DQuat::from_rotation_y(yaw.to_radians()),
            joint: None,
        });
    }
    // A door that left the band (or the world) drops its stamp, so coming back
    // re-describes it rather than retaining a body that is not there.
    stamps.retain(|leaf, _| live.contains(leaf));
}

/// A leaf pose folded into one `u64` — the change stamp.
///
/// Bit patterns, not rounded decimals: a stamp that agreed within a tolerance
/// would let a leaf drift by less than the tolerance for ever without the sync
/// noticing.
fn pose_stamp(centre: DVec3, yaw: f64, half: DVec3) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in [centre.x, centre.y, centre.z, yaw, half.x, half.y, half.z] {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// The tier a door at `centre` falls in — exposed so a gate can measure the
/// band's effect on doors the way `city_collider_band` measures it on walls.
pub fn tier_of(band: &SimBand, placement: &DoorPlacement, open_deg: f64) -> Tier {
    let (centre, _, half) = door::leaf_pose(placement, open_deg);
    band.tier(centre, half, DQuat::IDENTITY)
}
