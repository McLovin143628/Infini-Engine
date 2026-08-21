//! **The gameplay fixed step** (island wave I6): doors swing, weapons fire,
//! bodies stop working.
//!
//! One function ([`step_gameplay`]) that both hosts call, in the slot between
//! `step_character_movement` and the solver — the `step_pose_evaluation` shape
//! once more, so the editor's Simulate and the shipped player cannot resolve one
//! trigger pull differently.
//!
//! # What it does not do
//!
//! It does not spend energy at a **destructible**. A shot that hits a wall comes
//! back in [`GameplayReport::destruct`] and the host spends it through its own
//! `runtime_destruct_damage` wrapper — the one the `destruct.apply_damage` node
//! already goes through, which is where the `runtime_destruct` permission flag
//! is read and where the near-miss line is logged. Reaching past it would put a
//! second door on P22's damage, which is the exact defect the P22 "one door for
//! three paths" ruling exists to prevent.
//!
//! # The kick, and why it is on the notify
//!
//! `attack` on a locked door in reach arms a P29-style one-shot
//! ([`inf_ecs::weapon::KICK_TRIGGER`]) and **nothing else happens**. The impulse
//! lands when the animation says it does
//! ([`inf_ecs::weapon::KICK_NOTIFY`], consumed through
//! `inf_ecs::anim_bridge::consume_anim_notify`) — because a kick that broke the
//! lock the instant the button went down would break it before the leg moved,
//! and a notify seam that gameplay routes around is a notify seam nobody can
//! trust.
//!
//! A character with no rig has no notify, so the kick would never land: the
//! **pending kick has a fuse** ([`KICK_FUSE_S`]), and it fires on the fuse when
//! nothing animated it. Both paths are armed, and the report says which one ran.

use std::collections::BTreeSet;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{CharacterMovement, Transform};
use inf_ecs::door::{self, DoorSide, PendingKick};
use inf_ecs::item::{self, Inventory};
use inf_ecs::weapon::{self, FireVerdict, Health, ShotKind, WeaponDef, WeaponState};
use inf_ecs::world::EcsWorld;

use super::ecs::PhysicsBridge3D;

/// How long a kick waits for its animation before landing anyway, seconds.
///
/// A ceiling, not a schedule: a rigged character's kick lands on
/// [`inf_ecs::weapon::KICK_NOTIFY`] and this never fires. It exists because a
/// headless run, an NPC with no state machine and every level in this repository
/// committed before I6 have no animation to wait for, and a verb that silently
/// did nothing on those would be the dead-key defect I5 spent a wave on.
pub const KICK_FUSE_S: f64 = 0.35;

/// How far a hitscan shot may reach before the engine stops looking, metres —
/// the bound on `WeaponDef::range_m`, applied at the cast.
pub const SHOT_MAX_RANGE_M: f64 = weapon::MAX_RANGE_M;

/// Where a shot leaves the character, metres above its feet.
///
/// Chest height on a default capsule, so a shot fired along the aim does not
/// begin inside the ground on a downward pitch.
pub const MUZZLE_HEIGHT_M: f64 = 1.4;

/// **One shot that landed** — the record a gate and the tracer both read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeaponHit {
    /// Who fired.
    pub shooter: Uuid,
    /// What was hit, if anything the bridge could name.
    pub target: Option<Uuid>,
    /// Where the shot started, world metres.
    pub from: DVec3,
    /// Where it ended — the hit, or the end of its range.
    pub to: DVec3,
    /// The energy it carried, joules.
    pub energy_j: f64,
    /// Whether the target absorbed it as **health** (a character) rather than
    /// as structure.
    pub on_flesh: bool,
}

/// What one fixed step of gameplay did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GameplayReport {
    /// The door system's own numbers.
    pub doors: super::door::DoorReport,
    /// Rounds fired this step.
    pub shots: u32,
    /// Reloads that finished this step.
    pub reloads: u32,
    /// Kicks that landed this step.
    pub kicks: u32,
    /// Doors whose lock a kick broke this step.
    pub locks_broken: u32,
    /// Characters that stopped working this step and were handed to the ragdoll.
    pub kills: u32,
    /// Every shot that landed, in `Guid` order of the shooter.
    pub hits: Vec<WeaponHit>,
    /// Energy owed to the P22 damage door: `(destructible entity, joules)`.
    /// **The host spends this**, through its own wrapper. See the module header.
    pub destruct: Vec<(Uuid, f64)>,
}

/// **The gameplay fixed step.** Both hosts call it, between the character step
/// and the solver.
///
/// Inert on a level with no door, no weapon and no health: three
/// `try_query_filtered`s that answer `None`.
pub fn step_gameplay(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
) -> GameplayReport {
    let mut report = GameplayReport::default();
    if !dt.is_finite() || dt <= 0.0 {
        return report;
    }
    // 1. The doors move first, because a kick armed on a previous step lands in
    //    step 3 and must find the leaf where this step's solver will.
    let doors = super::door::step_doors(world, bridge, dt);
    report.doors = doors;
    // 2. Every character with a weapon: the trigger, the reload, the clocks.
    step_weapons(world, bridge, dt, &mut report);
    // 3. Every pending kick: the notify, or the fuse.
    step_kicks(world, dt, &mut report);
    // 4. Every body that stopped working goes to the ragdoll — the P29.4
    //    bridge's own door, whose doc has named "a damage system" as its
    //    intended caller since it was written.
    step_deaths(world, bridge, &mut report);
    report
}

/// **Where a character's feet are**, world metres.
///
/// The same arithmetic `step_one` and the prompt already use, and it is here
/// once rather than three times because a kick measured from a different point
/// than the prompt would let a player kick a door the prompt says is out of
/// reach.
pub fn feet_of(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    let entity = world.entity_of(guid)?;
    let w = world.world();
    let cm = w.get::<CharacterMovement>(entity)?;
    let t = w.get::<Transform>(entity)?;
    let radius = w
        .get::<inf_ecs::components::Collider3D>(entity)
        .map(|c| c.radius)
        .unwrap_or(0.3);
    Some(t.translation.to_dvec3() - DVec3::Y * (cm.half_height_for(cm.mode) + radius))
}

/// The character's muzzle, world metres, and where it is looking.
fn muzzle_of(world: &EcsWorld, guid: Uuid) -> Option<(DVec3, f64, f64)> {
    let feet = feet_of(world, guid)?;
    let entity = world.entity_of(guid)?;
    let cm = world.world().get::<CharacterMovement>(entity)?;
    Some((
        feet + DVec3::Y * MUZZLE_HEIGHT_M,
        cm.runtime.aim_yaw_deg,
        cm.runtime.aim_pitch_deg,
    ))
}

/// The equipped weapon's definition, if the character has one equipped and the
/// catalogue knows it.
fn equipped_weapon(world: &EcsWorld, guid: Uuid) -> Option<(String, WeaponDef)> {
    let entity = world.entity_of(guid)?;
    let inv = world.world().get::<Inventory>(entity)?;
    let id = inv.equipped_id()?.to_string();
    let def = *item::item_defs(world)?.get(&id)?.weapon.as_ref()?;
    Some((id, def))
}

/// Every character the weapon step visits, in `Guid` order — `O(characters)`.
fn gunners(world: &EcsWorld) -> Vec<Uuid> {
    inf_ecs::movement::movement_targets(world)
}

fn step_weapons(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
    report: &mut GameplayReport,
) {
    for guid in gunners(world) {
        // The equipped weapon decides everything, and an unarmed character costs
        // one map probe.
        let Some(entity) = world.entity_of(guid) else {
            continue;
        };
        // **The edges are TAKEN here, whether or not they are honoured** — the
        // law `step_one`'s own interact edge follows, for the same reason: an
        // edge a path could not use must not survive into a path that can, or a
        // press made in mid-air fires a kick on landing (the P29.7 A1 class).
        // Taken **before** the equipped-weapon check, because an unarmed
        // character still kicks doors and still scrolls.
        let (held_attack, press_attack, want_reload, want_switch) = {
            let w = world.world_mut();
            match w.get_mut::<CharacterMovement>(entity) {
                Some(mut cm) => {
                    let out = (
                        cm.runtime.want_attack,
                        cm.runtime.press_attack,
                        cm.runtime.press_reload,
                        cm.runtime.weapon_switch,
                    );
                    cm.runtime.press_attack = false;
                    cm.runtime.press_reload = false;
                    cm.runtime.weapon_switch = 0;
                    out
                }
                None => (false, false, false, 0),
            }
        };
        // The scroll wheel, before the trigger: a player who scrolls and fires
        // on one step fires the weapon they scrolled to, which is what they
        // asked for.
        if want_switch != 0 {
            let defs = item::item_defs(world).cloned().unwrap_or_default();
            let w = world.world_mut();
            if let Some(mut inv) = w.get_mut::<Inventory>(entity) {
                inv.cycle_equipped(&defs, want_switch, |d| d.is_weapon());
            }
        }
        // **THE ATTACK BUTTON'S TWO VERBS**, arbitrated once, here.
        //
        // A locked door in kicking reach takes the press; anything else lets it
        // through to the trigger. It is decided on the **edge** and not the
        // level, because a kick is a press — and while a kick is pending the
        // weapon does not fire, so holding the button against a door kicks it
        // rather than kicking and then shooting it.
        let kicking = if press_attack {
            let feet = feet_of(world, guid);
            let yaw = world
                .world()
                .get::<CharacterMovement>(entity)
                .map(|cm| cm.runtime.aim_yaw_deg)
                .unwrap_or(0.0);
            match feet {
                Some(feet) => try_kick(world, guid, feet, yaw),
                None => false,
            }
        } else {
            false
        };
        let pending_kick = world.world().get::<PendingKick>(entity).is_some();
        let want_fire = held_attack && !kicking && !pending_kick;
        // Now the weapon, if there is one.
        let Some((item_id, def)) = equipped_weapon(world, guid) else {
            // A character who put their weapon away keeps no ammunition clock:
            // a stale state is a magazine two weapons would share.
            world.world_mut().entity_mut(entity).remove::<WeaponState>();
            continue;
        };
        // Install (or replace) the ammunition clock. Replacing on an id change
        // is what stops two weapons sharing one magazine.
        let stale = world
            .world()
            .get::<WeaponState>(entity)
            .map(|s| s.item_id != item_id)
            .unwrap_or(true);
        if stale {
            world
                .world_mut()
                .entity_mut(entity)
                .insert(WeaponState::full(&item_id, &def));
        }
        // The animation's own reload notify, taken exactly once — the P29.4
        // seam, and the reason the fixed step asks rather than the animation
        // pushing: a notify is consumed by whoever gets there first, and there
        // must be exactly one consumer of a reload.
        let notified =
            inf_ecs::anim_bridge::consume_anim_notify(world, guid, weapon::RELOAD_NOTIFY);
        let mut fired = false;
        let mut reloaded = false;
        let (shot_index, aim) = {
            let mut aim = None;
            let mut shot = 0u64;
            let w = world.world_mut();
            if let Some(mut state) = w.get_mut::<WeaponState>(entity) {
                if notified {
                    reloaded |= weapon::finish_reload(&def, &mut state);
                }
                reloaded |= weapon::advance(&def, &mut state, dt);
                if want_reload {
                    weapon::try_reload(&def, &mut state);
                }
                if weapon::try_fire(&def, &mut state, want_fire) == FireVerdict::Fired {
                    fired = true;
                    shot = state.shots;
                }
            }
            if fired {
                aim = muzzle_of(world, guid);
            }
            (shot, aim)
        };
        if reloaded {
            report.reloads += 1;
        }
        if want_reload {
            // The animation follows the decision rather than gating it, exactly
            // as the ragdoll's does: an unrigged character still reloads.
            inf_ecs::anim_bridge::set_anim_trigger(world, guid, weapon::RELOAD_TRIGGER);
        }
        if !fired {
            continue;
        }
        inf_ecs::anim_bridge::set_anim_trigger(world, guid, weapon::FIRE_TRIGGER);
        report.shots += 1;
        let Some((from, yaw, pitch)) = aim else {
            continue;
        };
        let dir = weapon::shot_direction(&def, yaw, pitch, shot_index);
        let hit = resolve_shot(world, bridge, guid, &def, from, dir);
        apply_hit(world, &hit, report);
        report.hits.push(hit);
    }
}

/// Cast one shot and answer where it landed.
///
/// A **projectile** is resolved by the same cast today, at the same place a
/// hitscan lands — the flight time is not simulated. That is an honest v1 and it
/// is stated rather than implied: what `ShotKind::Projectile` changes in I6 is
/// the tracer's speed, and closing the rest means a body in flight, which is a
/// wave of its own.
fn resolve_shot(
    world: &EcsWorld,
    bridge: &mut PhysicsBridge3D,
    shooter: Uuid,
    def: &WeaponDef,
    from: DVec3,
    dir: DVec3,
) -> WeaponHit {
    let range = def.range_m.clamp(0.1, SHOT_MAX_RANGE_M);
    let mut exclude = BTreeSet::new();
    if let Some(c) = bridge.collider_of(shooter) {
        exclude.insert(c);
    }
    let landed = bridge
        .world_mut()
        .cast_ray_excluding(from, dir, range, &exclude);
    match landed {
        Some(h) => {
            let target = bridge.guid_of_collider(h.collider);
            let on_flesh = target
                .and_then(|g| world.entity_of(g))
                .map(|e| world.world().get::<Health>(e).is_some())
                .unwrap_or(false);
            WeaponHit {
                shooter,
                target,
                from,
                to: from + dir * h.toi,
                energy_j: def.damage_j,
                on_flesh,
            }
        }
        None => WeaponHit {
            shooter,
            target: None,
            from,
            to: from + dir * range,
            energy_j: def.damage_j,
            on_flesh: false,
        },
    }
}

/// Spend a hit's joules: on a body's health here, on a destructible through the
/// host's own wrapper.
fn apply_hit(world: &mut EcsWorld, hit: &WeaponHit, report: &mut GameplayReport) {
    let Some(target) = hit.target else {
        return;
    };
    if hit.on_flesh {
        if let Some(entity) = world.entity_of(target) {
            if let Some(mut h) = world.world_mut().get_mut::<Health>(entity) {
                weapon::damage(&mut h, hit.energy_j);
            }
        }
        return;
    }
    // Not flesh: the host spends it at the P22 door. Coalesced by entity so one
    // burst on one wall is one blow — which matters, because **damage is not
    // banked**: three small blows on one step are not a big one, and pretending
    // otherwise here would make the rate of fire a hidden multiplier on damage.
    if let Some(slot) = report.destruct.iter_mut().find(|(g, _)| *g == target) {
        slot.1 += hit.energy_j;
    } else {
        report.destruct.push((target, hit.energy_j));
    }
}

/// **Arm a kick** at the door in front of `character`, if there is one and the
/// character is close enough and facing it.
///
/// The attack button's door consumer. Returns whether a kick was armed, which is
/// what stops the same press also firing a weapon at the door.
pub fn try_kick(world: &mut EcsWorld, character: Uuid, feet: DVec3, aim_yaw_deg: f64) -> bool {
    if world.entity_of(character).is_none() {
        return false;
    }
    if door::pending_kick(world, character).is_some() {
        return false;
    }
    // The nearest LOCKED door in kicking reach, by the interaction rule's own
    // arithmetic — a kick is a reach and a cone, exactly as a prompt is, and
    // spelling it a second way would let a player kick a door the prompt says
    // is out of reach.
    let field = inf_ecs::door::door_field(world);
    let mut candidates: Vec<inf_ecs::interact::InteractCandidate> = Vec::new();
    for p in super::door::placements(world) {
        let state = field
            .map(|f| f.get(p.guid, &p.spec))
            .unwrap_or_else(|| inf_ecs::door::DoorState::fresh(&p.spec));
        if !state.locked || state.lock_broken {
            continue;
        }
        candidates.push(inf_ecs::interact::InteractCandidate {
            guid: p.guid,
            verb: inf_ecs::interact::InteractVerb::Use,
            label: p.label.clone(),
            position: inf_ecs::door::prompt_position(&p),
            range_m: inf_ecs::door::KICK_REACH_M,
            view_cone_deg: inf_ecs::door::KICK_CONE_DEG,
        });
    }
    candidates.sort_by_key(|c| c.guid);
    let Some(hit) = inf_ecs::interact::resolve(&candidates, feet, aim_yaw_deg) else {
        return false;
    };
    door::set_pending_kick(
        world,
        character,
        PendingKick {
            door: hit.guid,
            fuse_s: KICK_FUSE_S,
        },
    );
    inf_ecs::anim_bridge::set_anim_trigger(world, character, weapon::KICK_TRIGGER);
    true
}

fn step_kicks(world: &mut EcsWorld, dt: f64, report: &mut GameplayReport) {
    for (guid, mut kick) in door::pending_kicks(world) {
        let landed = inf_ecs::anim_bridge::consume_anim_notify(world, guid, weapon::KICK_NOTIFY);
        kick.fuse_s -= dt;
        if !landed && kick.fuse_s > 0.0 {
            door::set_pending_kick(world, guid, kick);
            continue;
        }
        door::clear_pending_kick(world, guid);
        report.kicks += 1;
        // **The one energy door.** A kick is a mass and a speed; what it is
        // compared against is the lock's own P22 bond energy. The crash-through
        // in `d3::movement` calls the same `strike_door` with the character's
        // own kinetic energy.
        let verdict = super::door::strike_door(world, kick.door, inf_ecs::door::kick_energy_j());
        if verdict.broke {
            report.locks_broken += 1;
        }
    }
}

fn step_deaths(world: &mut EcsWorld, _bridge: &mut PhysicsBridge3D, report: &mut GameplayReport) {
    for guid in weapon::newly_dead(world) {
        // **The P29.4 door, unchanged.** `start_ragdoll`'s own doc has named "a
        // damage system" as its intended caller since P29.4; this is that
        // caller, and it does not reach past the door into the rig.
        if super::ragdoll_bridge::start_ragdoll(world, guid) {
            report.kills += 1;
        }
    }
}

/// Which side of `door` a character standing at `feet` is on — re-exported for
/// the hosts' prompt, so the press and the prompt read one function.
pub fn side_of(world: &EcsWorld, door_guid: Uuid, feet: DVec3) -> Option<DoorSide> {
    let p = super::door::placement_of(world, door_guid)?;
    Some(p.spec.side_of(p.hinge, feet))
}

/// Give `character` an equipped weapon by item id — the door a Blueprint and a
/// gate both use, so a weapon cannot be equipped without its ammunition clock.
pub fn equip_weapon(world: &mut EcsWorld, character: Uuid, item_id: &str) -> bool {
    let Some(entity) = world.entity_of(character) else {
        return false;
    };
    let id = item::canonical_id(item_id);
    let slot = {
        let Some(inv) = world.world().get::<Inventory>(entity) else {
            return false;
        };
        inv.slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| s.id == id))
    };
    let Some(slot) = slot else {
        return false;
    };
    let Some(def) = item::item_defs(world)
        .and_then(|d| d.get(&id))
        .and_then(|d| d.weapon)
    else {
        return false;
    };
    {
        let w = world.world_mut();
        if let Some(mut inv) = w.get_mut::<Inventory>(entity) {
            inv.equip(slot);
        }
    }
    world
        .world_mut()
        .entity_mut(entity)
        .insert(WeaponState::full(&id, &def));
    true
}

/// The kind a shot of the equipped weapon is — what a tracer's speed reads.
pub fn equipped_shot_kind(world: &EcsWorld, character: Uuid) -> Option<ShotKind> {
    equipped_weapon(world, character).map(|(_, d)| d.kind)
}
