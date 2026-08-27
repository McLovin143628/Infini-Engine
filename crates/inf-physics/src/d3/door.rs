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
        // **BOTH VERBS, and this sentence is true now** (island wave I8b,
        // closing the I8a audit's MED-5). It used to say the lock was offered
        // "alongside" the open while [`use_door`] meant *instead*: from the lock
        // side a shut leaf took the lock verb whatever its bolt was doing, so
        // pressing E from inside cycled lock → unlock → lock and never opened,
        // and a character who shut the door behind them could not open it again.
        //
        // The two verbs have separate controls now — E is [`use_door`] and
        // always opens or closes, on either face; the bolt is [`lock_door`] and
        // is offered on the owner's face alone — so the prompt names a pair a
        // player can actually press.
        let word = if state.locked { "unlock" } else { "lock" };
        return format!("{} (open, or {word})", placement.label);
    }
    placement.label.clone()
}

/// The salt that carves a PCG volume's doorways out of the scene's own GUID
/// space. Its own constant, so a derived door can never alias a structure, a
/// shell, a voxel chunk or a leaf.
const PCG_DOORWAY_SALT: u128 = 0x6006_0300_5043_4744_4f4f_5257_4159_5321;

/// The synthetic identity of doorway `index` inside the volume on entity
/// `volume` — [`super::ecs::pcg_structure_guid`]'s rule with its own salt.
pub fn pcg_doorway_guid(volume: Uuid, index: usize) -> Uuid {
    let mut x = volume.as_u128() ^ PCG_DOORWAY_SALT;
    x ^= (index as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5cec_c5c3);
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

/// What a derived doorway's prompt calls it.
pub const GRAMMAR_DOOR_LABEL: &str = "door";

/// **Every door in the world that a character could act on**, in `Guid` order —
/// authored [`Door`] entities **and** the building grammar's own doorways.
///
/// The two are flattened onto one [`DoorPlacement`] list for exactly the reason
/// I5 flattened a vehicle seat and an authored item onto one
/// `InteractCandidate`: what makes this ONE door is that both are swung, ranked
/// and broken by one set of rules, not that they came from one place.
///
/// `O(doors + doorways)`, and `O(1)` on a level with neither.
///
/// # This is the UNBANDED list, and almost nothing should use it
///
/// A city plans on the order of twenty thousand doorways, and walking all of
/// them per fixed step — twice, once to describe colliders and once to swing —
/// is a cost a load-time budget never sees. Every per-step caller takes
/// [`placements_near`] instead; this one exists for a **guid lookup**
/// ([`placement_of`]), which is rare and must answer for a door wherever it is.
pub fn placements(world: &EcsWorld) -> Vec<DoorPlacement> {
    let mut out = door::doors_in_world(world);
    out.extend(volume_doorways(world));
    // One sorted walk: the resolver's tie-break is only deterministic over one,
    // and an authored door and a grammar doorway at the same distance must
    // resolve the same way in both hosts.
    out.sort_by_key(|p| p.guid);
    out
}

/// **Every door close enough to matter this step**, in `Guid` order.
///
/// The same band the structural colliders use — `PhysicsBridge3D::sim_band`,
/// with the same radii — because a level whose doors and walls were solid at
/// different distances would have doorways with leaves in buildings with no
/// walls. It **fails open** exactly as the band does: no streaming source means
/// no banding, which is every level committed before the island and every unit
/// fixture.
///
/// The cost this buys is stated rather than implied: an unbanded door does not
/// swing, so a leaf a player left moving four kilometres away freezes where it
/// is until they come back. That is the same bargain a distant wall's collider
/// makes, and it is the reason the band's own doc says the cells decide what
/// EXISTS and the band decides what is SOLID.
pub fn placements_near(world: &EcsWorld, band: &SimBand) -> Vec<DoorPlacement> {
    let mut out = door::doors_in_world(world);
    out.retain(|p| in_band(p, band));
    // **The band is checked BEFORE the placement is built**, and that is the
    // whole point of the visitor: the shipped city plans 19 790 doorways and the
    // band keeps 234, so collecting all of them to throw 98.8 % away would copy
    // about 2 MB and allocate a label string per door — three or four times a
    // fixed step, on a path whose per-step budget is 1.25 ms.
    door::for_each_volume_doorway(world, |volume, i, slot| {
        let Some(spec) = derived_spec(slot) else {
            return;
        };
        let p = DoorPlacement {
            guid: pcg_doorway_guid(volume, i),
            hinge: slot.hinge,
            spec,
            // The label is the last thing built, because it is the only
            // allocation in this closure.
            label: String::new(),
        };
        if !in_band(&p, band) {
            return;
        }
        out.push(DoorPlacement {
            label: GRAMMAR_DOOR_LABEL.to_string(),
            ..p
        });
    });
    out.sort_by_key(|p| p.guid);
    out
}

/// **The spec a derived doorway gets**, from the grammar's own slot.
///
/// One function, so the banded walk and the unbanded one cannot disagree about
/// what a grammar door IS — which they would the first time one of them grew a
/// field. `None` for a slot whose numbers are unusable.
fn derived_spec(d: &inf_ecs::components::DoorwaySlot) -> Option<door::DoorSpec> {
    let spec = door::DoorSpec {
        closed_yaw_deg: d.closed_yaw_deg,
        width_m: d.width_m,
        height_m: d.height_m,
        thickness_m: d.thickness_m,
        // **A grammar door swings INTO the room its wall serves**, which is what
        // the plan's own `Wall::inside` means — so the limit's sign follows the
        // inside normal rather than being a constant. A door that opened the
        // other way would open into the corridor it came from, and every
        // interior door in a building would block the one opposite it.
        open_limit_deg: -door::DEFAULT_OPEN_LIMIT_DEG,
        inside_yaw_deg: d.inside_yaw_deg,
        lock_strength_pa: door::DEFAULT_LOCK_STRENGTH_PA,
        lock_area_m2: door::DEFAULT_LOCK_AREA_M2,
        lock_side: door::DoorSide::Inside,
        // **Nothing the grammar builds starts locked.** A city whose every
        // interior door was bolted would be a city nobody can walk through, and
        // there is no authored intent to read one from. Locking is a verb a
        // player (or a Blueprint) uses.
        locked_at_spawn: false,
    };
    spec.is_usable().then_some(spec)
}

/// Whether a placement's SHUT leaf is inside the band.
fn in_band(p: &DoorPlacement, band: &SimBand) -> bool {
    let (centre, yaw, half) = door::leaf_pose(p, 0.0);
    // The band is asked with the leaf's REAL rotation, not the identity: a box's
    // tier is computed from its oriented extent, and a door lying along its own
    // local `+Z` is not the same shape rotated ninety degrees.
    band.tier(centre, half, DQuat::from_rotation_y(yaw.to_radians()))
        .is_near()
}

/// The grammar's half of the door list, in `Guid` order — **unbanded**.
///
/// `O(doorways)`, and `O(1)` on a level with no `PcgVolume`. Only
/// [`placements`] uses it; every per-step caller takes [`placements_near`],
/// which band-checks before it allocates.
fn volume_doorways(world: &EcsWorld) -> Vec<DoorPlacement> {
    let mut out = Vec::new();
    door::for_each_volume_doorway(world, |volume, i, slot| {
        if let Some(spec) = derived_spec(slot) {
            out.push(DoorPlacement {
                guid: pcg_doorway_guid(volume, i),
                hinge: slot.hinge,
                spec,
                label: GRAMMAR_DOOR_LABEL.to_string(),
            });
        }
    });
    out.sort_by_key(|p| p.guid);
    out
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
pub fn candidates(world: &EcsWorld, band: &SimBand, feet: DVec3) -> Vec<InteractCandidate> {
    let field = door::door_field(world);
    let mut out: Vec<InteractCandidate> = placements_near(world, band)
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
                // **A door is opened by its handle** (SK1c). The generator that
                // writes a rig's grip catalogue names this one `handle`, and it
                // is what makes the near hand close on a door leaf instead of
                // opening it by telekinesis. A rig with no such affordance --
                // the canonical twenty-joint biped, which has no fingers -- just
                // does not close a hand, which is the honest answer.
                grip: Some(inf_anim::GRIP_HANDLE.to_string()),
            }
        })
        .collect();
    out.sort_by_key(|c| c.guid);
    out
}

/// The one door whose guid is `guid`, if the world has it.
///
/// Through the **visitor**, not through [`placements`], for the reason that
/// function's own doc gives and the I6 audit measured: a city plans 19 790
/// doorways, and collecting them to find one would allocate a label string for
/// every door in the world so that 19 789 of them could be dropped. This is
/// `O(doorways)` pointer bumps and **one** allocation.
pub fn placement_of(world: &EcsWorld, guid: Uuid) -> Option<DoorPlacement> {
    if let Some(p) = door::doors_in_world(world)
        .into_iter()
        .find(|p| p.guid == guid)
    {
        return Some(p);
    }
    let mut found: Option<DoorPlacement> = None;
    door::for_each_volume_doorway(world, |volume, i, slot| {
        if found.is_some() || pcg_doorway_guid(volume, i) != guid {
            return;
        }
        if let Some(spec) = derived_spec(slot) {
            found = Some(DoorPlacement {
                guid,
                hinge: slot.hinge,
                spec,
                label: GRAMMAR_DOOR_LABEL.to_string(),
            });
        }
    });
    found
}

/// How close a point has to be to a door for [`is_open_near`] to be about it,
/// metres — a doorway's own reach plus a stride.
pub const NEAR_DOOR_M: f64 = 3.0;

/// **Is the door nearest `at` open** — the Blueprint kit's `door.is_open`.
///
/// A place rather than an entity, because half the doors in this engine have no
/// entity: a grammar doorway is a record on a volume. `false` when there is no
/// door within [`NEAR_DOOR_M`], which is the honest answer for "is the door
/// here open" when there is no door here.
/// How far a point is from a door's prompt, or `None` when that is not a
/// number or is out of [`NEAR_DOOR_M`].
fn reach_to(p: &DoorPlacement, at: DVec3) -> Option<f64> {
    let d = (door::prompt_position(p) - at).length();
    (d.is_finite() && d <= NEAR_DOOR_M).then_some(d)
}

pub fn is_open_near(world: &EcsWorld, at: DVec3) -> bool {
    // **The reach is checked BEFORE a placement is built** — the same discipline
    // `placements_near` applies with the band, and for the same measurement:
    // this is a Blueprint node an author may put on `Tick`, and the shipped city
    // plans 19 790 doorways, so collecting them all would allocate a label
    // string per door and sort twenty thousand records three times a second to
    // answer a question about one. `String::new()` does not allocate, so the
    // probe below costs nothing until a door is actually in reach. (I6 audit.)
    let mut near: Vec<DoorPlacement> = door::doors_in_world(world)
        .into_iter()
        .filter(|p| reach_to(p, at).is_some())
        .collect();
    door::for_each_volume_doorway(world, |volume, i, slot| {
        let Some(spec) = derived_spec(slot) else {
            return;
        };
        let p = DoorPlacement {
            guid: pcg_doorway_guid(volume, i),
            hinge: slot.hinge,
            spec,
            label: String::new(),
        };
        if reach_to(&p, at).is_some() {
            near.push(DoorPlacement {
                label: GRAMMAR_DOOR_LABEL.to_string(),
                ..p
            });
        }
    });

    // The `Guid` order the tie-break rests on, over the doors that are in reach
    // rather than over every door in the world.
    near.sort_by_key(|p| p.guid);
    let field = door::door_field(world);
    let mut best: Option<(f64, DoorPlacement)> = None;
    for p in near {
        let Some(d) = reach_to(&p, at) else {
            continue;
        };
        // Strict `<` over the `Guid`-ordered walk, so two doors at the same
        // distance answer the same way in both hosts.
        if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
            best = Some((d, p));
        }
    }
    match best {
        Some((_, p)) => field
            .map(|f| f.get(p.guid, &p.spec))
            .unwrap_or_else(|| DoorState::fresh(&p.spec))
            .is_open(&p.spec),
        None => false,
    }
}

/// **The E key on a door** — the consumer the one interaction site calls when
/// the hit it resolved is a door.
///
/// Returns the verdict, which is a value in every case: an unusable door, a
/// locked one, a lock toggled, a leaf opening. The caller does not need to know
/// which; it needs to know whether anything happened, and the verdict says.
///
/// **E ALWAYS MEANS OPEN OR CLOSE, ON EITHER FACE** (island wave I8b).
///
/// It did not used to. From the lock side a **shut** leaf took the *lock* verb
/// whatever its bolt was doing, so the press cycled lock → unlock → lock and a
/// character who shut the door behind them could never open it again — 2 070
/// doorways at a time on the shipped island. That was deliberate ("locked from
/// the inside" has to mean something for whoever locked it) and it was wrong,
/// because one control cannot carry two verbs that are both wanted in the same
/// state. The bolt has its own control now ([`lock_door`]), [`door_label`]
/// names both, and this function does the one thing its name says.
///
/// `feet` is unused for the verb and kept in the signature because every caller
/// has it in hand and because a side-dependent open direction would read it
/// here.
pub fn use_door(world: &mut EcsWorld, guid: Uuid, feet: DVec3) -> door::DoorVerdict {
    let _ = feet;
    let Some(p) = placement_of(world, guid) else {
        return door::DoorVerdict::Unusable;
    };
    // A locked door answers `Locked` and does not move — `door::toggle`'s own
    // rule, unchanged, and what the prompt already reads.
    with_state(world, &p, door::toggle)
}

/// **The bolt's own control** (island wave I8b) — the second half of the pair
/// [`door_label`] promises.
///
/// Offered on the owner's face alone ([`door::set_locked`] answers `WrongSide`
/// from the other one, and for a lock that has been broken), and **refused on an
/// open leaf**: a door standing open with its bolt thrown is a lock nobody can
/// see, which is the one part of the old single-button behaviour worth keeping.
///
/// It toggles rather than taking a target state, because the control it is
/// reached from is one key: a player who can lock a door can unlock it with the
/// same press, which is what a key in a hand does.
pub fn lock_door(world: &mut EcsWorld, guid: Uuid, feet: DVec3) -> door::DoorVerdict {
    let Some(p) = placement_of(world, guid) else {
        return door::DoorVerdict::Unusable;
    };
    let side = p.spec.side_of(p.hinge, feet);
    with_state(world, &p, |spec, state| {
        if state.is_open(spec) {
            return door::DoorVerdict::WrongSide;
        }
        door::set_locked(spec, state, side, !state.locked)
    })
}

/// Read a door's state, run `f` over it, and **write it back only if it
/// changed**.
///
/// The one write door into [`door::DoorField`], and the reason it exists is
/// the map's sparseness: absent means closed, so a *refused* verb — a locked
/// door that would not open, a breach too slow to matter — must leave no entry
/// behind. A first draft materialised on every press and made a level's trace
/// bytes a function of how many doors a player had walked past.
fn with_state<R>(
    world: &mut EcsWorld,
    p: &DoorPlacement,
    f: impl FnOnce(&door::DoorSpec, &mut DoorState) -> R,
) -> R {
    let field = door::door_field_mut(world);
    let before = field.get(p.guid, &p.spec);
    let mut state = before;
    let out = f(&p.spec, &mut state);
    if state != before {
        *field.entry(p.guid, &p.spec) = state;
    }
    out
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
    with_state(world, &p, |spec, state| {
        let verdict = door::try_break(spec, state, energy_j);
        door::apply_break(spec, state, &verdict);
        verdict
    })
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
    band: &SimBand,
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
    for p in placements_near(world, band) {
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
    // **The leaf reaches its stop on the free swing alone**, so there is nothing
    // to do here but price the exit (the I6 audit). A first draft re-wrote
    // `target_deg` at this point, claiming to "open it the rest of the way under
    // power" — which it did not do (`apply_break` leaves `powered` false and had
    // already set the same `target_deg`, so the write was a no-op that removed
    // itself under mutation) and did not need to: `apply_break`'s own floor of
    // `0.35 × BREACH_SWING_DPS` is 182 deg/s, and `DOOR_FREE_DRAG_PER_S` of 1.1
    // carries that about 160 degrees before the settle threshold stops it,
    // against a 95-degree limit. The weakest breach this door can price still
    // reaches the frame.
    let out = if verdict.broke {
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

/// How far the blocking probe's box is inset from the leaf's own, metres — a
/// **skin**, the same thing `CharacterController3D::offset` is and for the same
/// reason.
///
/// A leaf stands ON the floor and BETWEEN two wall boxes: its own geometry is in
/// resting contact with three solids at all times. A sweep of the leaf's exact
/// box therefore begins penetrating on the very first step, and parry reports
/// that as a hit with `toi == 0` — which read as "blocked" and meant **no door
/// in the engine could ever open**. Measured on the first fixture that had a
/// floor in it: 0 of 90 steps moved, every one of them reported blocked.
///
/// Two centimetres, matching the character mover's own skin, and the cost is
/// stated: a leaf can swing 2 cm into something before the probe sees it, which
/// is under the leaf's own thickness.
const BLOCK_SKIN_M: f64 = 0.02;

/// **One fixed step of every door.**
///
/// Runs after the character movement step (so a door the E key opened *this*
/// step starts moving this step) and before the solver (so the leaf's collider
/// is where the state says it is when the solver runs). Both hosts call it; it
/// is the `step_character_movement` shape one system along.
///
/// Inert on a level with no doors: one `try_query_filtered` that answers `None`.
pub fn step_doors(world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, dt: f64) -> DoorReport {
    let band = bridge.sim_band(world);
    let places = placements_near(world, &band);
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
        // **The field is only written for a door that CHANGED**, which is what
        // keeps it sparse — and sparse is not an optimisation here, it is the
        // claim that a level whose doors nobody has touched contributes nothing
        // to `state_bytes` and therefore leaves every pre-I6 trace
        // byte-identical. A first draft called `entry` unconditionally and
        // materialised every door in the world on its first step; the arm that
        // caught it asserts the trace bytes are empty after ten steps of a world
        // that has a door in it.
        let field = door::door_field_mut(world);
        for (p, blocked) in places.iter().zip(blocked.iter().copied()) {
            if blocked {
                report.blocked += 1;
            }
            let before = field.get(p.guid, &p.spec);
            let mut state = before;
            let moved = door::advance(&p.spec, &mut state, dt, blocked);
            if state != before {
                *field.entry(p.guid, &p.spec) = state;
            }
            if moved {
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
fn is_blocked(bridge: &mut PhysicsBridge3D, p: &DoorPlacement, state: &DoorState, dt: f64) -> bool {
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
    // The probe's box is the leaf's, inset by a skin — see `BLOCK_SKIN_M` for
    // the measurement that made it necessary.
    let shape = ColliderShape3D::Box {
        half_extents: DVec3::new(
            (half.x * 0.5).max(1e-3),
            (half.y - BLOCK_SKIN_M).max(1e-3),
            (half.z - BLOCK_SKIN_M).max(1e-3),
        ),
    };
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
    // A sweep that **starts** penetrating is not a block: it is the leaf resting
    // against its own frame or standing on the floor, which is where a door
    // lives. What blocks a door is something it is about to *reach*.
    bridge
        .world_mut()
        .cast_shape(&shape, centre, rot, tangent, reach, &exclude)
        .is_some_and(|h| !h.started_penetrating)
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
    // The off path: no authored door, no volume that could carry a doorway, and
    // nothing tracked is one compare.
    //
    // **`has_component::<Door>()` alone is not the off path**, and the arm that
    // says so is the grammar house: a derived doorway is a `DoorwaySlot` on a
    // `PcgVolume`, not a `Door` component, so a level whose only doors came from
    // the grammar took the fast path and **got no leaf at all** - eight doors
    // reported and zero colliders built.
    if stamps.is_empty()
        && !world.has_component::<Door>()
        && !world.has_component::<inf_ecs::components::PcgVolume>()
    {
        return;
    }
    let places = placements_near(world, band);
    let field = door::door_field(world);
    let mut live: BTreeSet<Uuid> = BTreeSet::new();
    for p in places {
        let state = field
            .map(|f| f.get(p.guid, &p.spec))
            .unwrap_or_else(|| DoorState::fresh(&p.spec));
        let (centre, yaw, half) = door::leaf_pose(&p, state.open_deg);
        let rot = DQuat::from_rotation_y(yaw.to_radians());
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
            rotation: rot,
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
