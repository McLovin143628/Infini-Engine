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

/// Where a shot leaves a character **that has no weapon to read a muzzle off**,
/// metres above its feet.
///
/// # It used to be *the* muzzle, and SK1b is when it stopped being
///
/// Until this wave an equipped weapon was an inventory slot id and nothing else:
/// no entity, no transform, nothing in the world at all, so a shot had to start
/// at a height somebody picked. Chest height on a default capsule, so a shot
/// fired along the aim does not begin inside the ground on a downward pitch.
///
/// A character with a **rig** and an equipped weapon now carries that weapon as a
/// real entity attached to its `hand_r` socket, and `muzzle_of` reads the shot's
/// origin off the weapon's own muzzle. This is what is left: the answer for a
/// bare capsule — every level committed before this wave, the whole
/// `phase30-gameplay` fixture, and every test rig that steps gameplay without
/// stepping the pose. `the_new_muzzle_agrees_with_the_old_one_on_a_capsule_hero`
/// is the control that pins the two together, because "no gameplay regression"
/// is a claim about a number and not a feeling.
pub const MUZZLE_HEIGHT_M: f64 = 1.4;

/// The socket an equipped weapon hangs from.
///
/// The engine's own name for the right hand, published by every rig this engine
/// generates — [`inf_anim::manny`]'s twelve sockets include it under both the
/// engine spelling and ALS's `hand_r_socket`, and the twenty-joint template has
/// carried it since P24.1. A rig that does not publish it gets an attachment at
/// its entity origin (`inf_ecs::attach`'s documented fallback) and a muzzle from
/// the capsule rule, which is the same answer it got before this wave.
pub const WEAPON_SOCKET: &str = "hand_r";

/// The salt that carves equipped weapons' GUID space out of the scene's own —
/// `item::dropped_item_guid`'s shape, with its own constant.
const EQUIPPED_WEAPON_SALT: u128 = 0x5745_4150_4f4e_5f45_5155_4950_5045_4421;

/// **The guid of the entity that IS a character's equipped weapon.**
///
/// Content-derived from the owner, the P22 idiom: a fixed step may not mint a
/// random guid, because two hosts stepping the same sim have to produce the same
/// entity or the trace forks on the first frame anything is equipped. One weapon
/// entity per character, re-used as the character switches weapons — a rifle and
/// a pistol are the same slot in the same hand.
pub fn equipped_weapon_guid(owner: Uuid) -> Uuid {
    let mut x = owner.as_u128() ^ EQUIPPED_WEAPON_SALT;
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

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
    /// **What the crowd did to the doors** this step (island wave NPC1c) — an
    /// engagement counter, because "the pass ran" and "an NPC opened
    /// something" are different facts and a gate that cannot tell them apart
    /// certifies a no-op.
    pub crowd_doors: CrowdDoorReport,
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
    /// **What the hand pass asked for** this step, `(weapon holds, grabs)`
    /// (SK1c) — engagement counters, because "the hand step ran" and "a hand was
    /// asked to do something" are different facts and a gate that cannot tell
    /// them apart certifies a no-op.
    pub hands: (u32, u32),
    /// **Shots this step that fell back to [`MUZZLE_HEIGHT_M`] although the
    /// shooter publishes a pose** (SK1b audit) — the tripwire on the muzzle's
    /// silent half.
    ///
    /// `muzzle_of` has two answers: the weapon entity's own muzzle, and a height
    /// above the character's feet. The second is right for a **rig-less** hero —
    /// every level committed before SK1b, the whole `phase30-gameplay` fixture —
    /// and those are not counted, because they have no pose at all.
    ///
    /// What *is* counted is a character that has a rig and still took the capsule
    /// rule: its skeleton does not author [`WEAPON_SOCKET`], or its weapon entity
    /// has not been placed. Both put every shot back at 1.4 m above the feet
    /// **in silence**, which is a half-metre error on a crouched character and a
    /// shot through the floor on a prone one. A rigged course asserts this is
    /// zero; nothing else could tell the difference between the fallback working
    /// as designed and a rig that quietly lost its hand.
    pub muzzles_without_a_socket: u32,
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
    // 0. **An NPC opens the door in its way** (island wave NPC1c). Before the
    //    leaves move, so a crowd agent's press has the same immediacy the
    //    player's E already has -- that one is consumed in
    //    `step_character_movement`, a phase earlier than this whole function.
    //    Inert on every level with no crowd: one absent-resource read.
    report.crowd_doors = step_crowd_doors(world, bridge);
    // 1. The doors move first, because a kick armed on a previous step lands in
    //    step 3 and must find the leaf where this step's solver will.
    let doors = super::door::step_doors(world, bridge, dt);
    report.doors = doors;
    // 2. Every character with a weapon: the trigger, the reload, the clocks.
    step_weapons(world, bridge, dt, &mut report);
    // 3. Every pending kick: the notify, or the fuse.
    step_kicks(world, dt, &mut report);
    // 3b. **The equipped weapon is an entity** (SK1b) — spawned, moved by the
    //     attachment pass below the pose, despawned when nothing is equipped.
    //     After the weapon step, because that is where a scroll wheel changes
    //     what is equipped.
    step_equipped_weapons(world);
    // 3c. **The hands** (SK1c) — one request per character, composed from what
    //     it is holding and what it just pressed E on. After the weapon entity
    //     exists (so a hold and a spawn cannot disagree about the same step) and
    //     before the pose step reads it, which is the ordering the
    //     gameplay < pose < attachments pin already covers.
    report.hands = step_hand_ik(world, dt);
    // 4. Every body that stopped working goes to the ragdoll — the P29.4
    //    bridge's own door, whose doc has named "a damage system" as its
    //    intended caller since it was written.
    step_deaths(world, bridge, &mut report);
    report
}

/// What [`step_crowd_doors`] did this step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CrowdDoorReport {
    /// Blocked agents this pass looked at.
    pub considered: usize,
    /// Presses made -- one per agent that found a shut door in reach.
    pub pressed: usize,
    /// Presses that moved a leaf. `pressed - opened` is the locked ones, which
    /// is the number a designer wants when a district stops working.
    pub opened: usize,
}

/// **A crowd agent opens the door it is standing against** -- clause 3's door
/// verb, through the same [`super::door::use_door`] the interact button and the
/// `door.use` node dispatch to.
///
/// # Why "blocked" is the trigger, and what it costs in seconds
///
/// The alternative -- every agent asking every step whether a door is in front
/// of it -- is `O(agents x doorways)` over a city that plans 19 790 doorways,
/// and 288 near agents would pay it every fixed step for the handful of doors
/// anybody is actually at. So the trigger is the crowd's own
/// [`CrowdAgent::blocked`](inf_ecs::crowd::CrowdAgent::blocked) verdict: an
/// agent whose body has fallen [`BLOCKED_LAG_M`] behind its own route clock.
///
/// [`BLOCKED_LAG_M`]: inf_ecs::crowd::BLOCKED_LAG_M
///
/// That is 2 m of lag, which at a 1.65 m/s walk is about **1.2 seconds** of
/// standing at the door before the handle turns. It is stated rather than
/// hidden, and it reads as a pause rather than as a bug -- a person does pause
/// at a door -- but it is a tuning constant and not a design: lowering the lag
/// shortens the pause and widens this pass's subject set in the same move.
///
/// A **locked** door is pressed and refuses, exactly as it refuses a player, and
/// the agent goes on trying the handle for as long as it stays blocked. That is
/// deliberate rather than unfinished: remembering which doors an agent has
/// already tried is per-agent state, and per-agent state on this path has to
/// ride the crowd's own trace section or two hosts diverge. The cost is one
/// `door::toggle` refusal per blocked agent per step, which is a state lookup;
/// the benefit is that the counters say `pressed` without `opened`, which is
/// the number a designer wants when a district stops working.
pub fn step_crowd_doors(world: &mut EcsWorld, bridge: &PhysicsBridge3D) -> CrowdDoorReport {
    let mut report = CrowdDoorReport::default();
    let blocked = inf_ecs::crowd::blocked_agents(world);
    if blocked.is_empty() {
        return report;
    }
    report.considered = blocked.len();
    // ONE band and ONE placement gather for the whole pass. `placements_near`
    // is the same door `candidates` and the player's prompt read, so an NPC
    // cannot reach a door the player is told is out of reach.
    let band = bridge.sim_band(world);
    let placements = super::door::placements_near(world, &band);
    if placements.is_empty() {
        return report;
    }
    for guid in blocked {
        let Some(feet) = feet_of(world, guid) else {
            continue;
        };
        let field = door::door_field(world);
        // The nearest SHUT door within reach. Ties break on the door's guid,
        // through `placements_near`'s own ascending order, so two agents at one
        // threshold press the same leaf.
        let mut best: Option<(f64, Uuid)> = None;
        for p in &placements {
            let state = field
                .map(|f| f.get(p.guid, &p.spec))
                .unwrap_or_else(|| door::DoorState::fresh(&p.spec));
            // Shut **and at rest**. Without the second half an agent presses
            // again on the next step while the leaf is still swinging, and
            // `use_door` toggles -- measured, thirteen presses to open one door
            // and it shut itself twice on the way.
            if state.is_open(&p.spec) || !state.is_at_rest() {
                continue;
            }
            let d = (door::prompt_position(p) - feet).length();
            if d > door::DOOR_REACH_M {
                continue;
            }
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, p.guid));
            }
        }
        let Some((_, door_guid)) = best else {
            continue;
        };
        report.pressed += 1;
        if super::door::use_door(world, door_guid, feet).moved() {
            report.opened += 1;
        }
    }
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

/// **The character's muzzle**, world metres, and where it is looking.
///
/// # Two answers, and which one applies
///
/// 1. **The weapon's own muzzle.** A character with a rig carries its equipped
///    weapon as an entity attached to [`WEAPON_SOCKET`]; the shot leaves that
///    entity's barrel, [`WeaponDef::muzzle_forward_m`] along its local `+Z`. The
///    weapon's placement is required to be *finite and settled* — it is only
///    settled once `update_attachments` has run, which is why the pose is asked
///    for as well: a character that publishes no pose has an attachment sitting
///    at its entity origin, which is its capsule centre, and a muzzle there would
///    be a silent half-metre regression on every unrigged level in the tree.
/// 2. **A height above the feet**, [`MUZZLE_HEIGHT_M`] — everything else.
///
/// # The one-step latency, stated
///
/// `step_gameplay` runs **before** `step_pose_evaluation` and
/// `update_attachments` in both hosts, so the weapon transform this reads is the
/// one the previous fixed step settled. At 60 Hz that is 16.7 ms of lag between
/// where the hand is and where the shot starts, and it is the same lag in both
/// hosts — so PIE == shipping is unaffected and the trace cannot see it. Moving
/// the gameplay step below the pose would fix it and would move every committed
/// trace in the tree; it is named here rather than done quietly.
/// The fourth element is **which of the two answers this is**: `true` for the
/// weapon's own muzzle, `false` for the capsule rule. The caller counts the
/// second one when it happens to a *posed* character — see
/// [`GameplayReport::muzzles_without_a_socket`].
fn muzzle_of(world: &EcsWorld, guid: Uuid) -> Option<(DVec3, f64, f64, bool)> {
    let entity = world.entity_of(guid)?;
    let cm = world.world().get::<CharacterMovement>(entity)?;
    let (yaw, pitch) = (cm.runtime.aim_yaw_deg, cm.runtime.aim_pitch_deg);
    if let Some(from) = weapon_muzzle(world, guid) {
        return Some((from, yaw, pitch, true));
    }
    let feet = feet_of(world, guid)?;
    Some((feet + DVec3::Y * MUZZLE_HEIGHT_M, yaw, pitch, false))
}

/// The muzzle of `guid`'s **weapon entity**, if there is one and it is attached
/// to a real socket. `None` sends `muzzle_of` to the capsule rule.
fn weapon_muzzle(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    use inf_ecs::components::GlobalTransform;
    // The socket has to exist on the rig AND have been resolved, or the
    // attachment is sitting at the character's own origin.
    inf_ecs::pose::evaluated_pose(world, guid)?.socket(WEAPON_SOCKET)?;
    let (_, def) = equipped_weapon(world, guid)?;
    let e = world.entity_of(equipped_weapon_guid(guid))?;
    let g = world.world().get::<GlobalTransform>(e)?.0;
    let forward = def
        .muzzle_forward_m
        .clamp(0.0, weapon::MAX_MUZZLE_FORWARD_M);
    let at = g.transform_point3(DVec3::new(0.0, 0.0, forward));
    at.is_finite().then_some(at)
}

/// **Keep every character's equipped weapon a real entity in the world**
/// (SK1b) — spawned when something is equipped, moved by the attachment pass,
/// despawned when nothing is.
///
/// # Why a whole entity
///
/// Before this an "equipped weapon" was a slot index. Nothing drew it, nothing
/// could be attached to it, and the shot it fired started at a hard-coded height
/// above the character's feet — the scout's risk 14, whole. An entity is what a
/// socket can carry, what a projector can draw and what a muzzle can be read off,
/// and it costs no schema: `AttachedTo` is already a scene component and this one
/// is never saved.
///
/// # Deterministic by construction
///
/// The guid is [`equipped_weapon_guid`], derived from the owner, so both hosts
/// spawn the same entity on the same step. The entity **appears in the trace** as
/// a transform row, which is the honest cost and is why it is spawned from the
/// one Ring-0 rule both hosts call rather than from either of them.
fn step_equipped_weapons(world: &mut EcsWorld) {
    use inf_ecs::components::{AttachedTo, MeshRef, Name, Primitive, Transform, Visibility};

    for guid in gunners(world) {
        let want = equipped_weapon(world, guid).map(|(id, def)| (id, def.muzzle_forward_m));
        let weapon_guid = equipped_weapon_guid(guid);
        let existing = world.entity_of(weapon_guid);
        match (want, existing) {
            (None, Some(e)) => {
                // Nothing equipped: the weapon leaves the world, so a holstered
                // character is byte-identical to one that never had a weapon.
                world.despawn(e);
            }
            (None, None) => {}
            (Some((id, forward)), existing) => {
                let e = match existing {
                    Some(e) => e,
                    None => world.spawn_with_guid(weapon_guid, &format!("Weapon: {id}"), None),
                };
                let w = world.world_mut();
                // The name follows the equipped item, so switching weapons is
                // visible in the Outliner rather than silently the same row.
                if let Some(mut n) = w.get_mut::<Name>(e) {
                    let label = format!("Weapon: {id}");
                    if n.0 != label {
                        n.0 = label;
                    }
                }
                // A placeholder primitive, scaled to the weapon's own length —
                // the `item::spawn_pickup` precedent, and the same honest bound:
                // an `ItemDef` that names a mesh asset is the next field and is
                // not in this wave.
                //
                // **The size survives the attachment pass** (SK1b audit), and it
                // did not when this was written: `update_attachments` composed
                // the *target's* scale onto its follower, which erased this line
                // one pass later and drew a **1 m cube** in the character's hand.
                // That pass leaves a follower's own size alone now. The two arms
                // are `an_attachment_places_a_follower_without_resizing_it` and
                // `the_equipped_weapon_is_an_entity_attached_to_the_hand_socket`.
                let mut t = w.get_mut::<Transform>(e).map(|t| *t).unwrap_or_default();
                let len = forward.clamp(0.05, weapon::MAX_MUZZLE_FORWARD_M);
                t.scale = inf_ecs::math::Vec3d::new(0.06, 0.06, len);
                w.entity_mut(e).insert((
                    t,
                    MeshRef {
                        primitive: Primitive::Cube,
                        asset: None,
                    },
                    Visibility::default(),
                    // Zero offset: the weapon sits AT the hand socket, which is
                    // where a grip's palm frame is. A `GripAffordance`'s palm
                    // transform is the refinement, and it needs the rig — which
                    // this step does not have and the pose step does.
                    AttachedTo::new(guid, WEAPON_SOCKET, inf_ecs::math::Vec3d::ZERO),
                ));
            }
        }
    }
}

/// **Where a two-handed weapon's fore-grip is**, metres along the barrel from
/// the holding hand.
///
/// A fraction of the weapon's own length rather than a constant: a pistol has no
/// fore-grip to reach and a rifle's is most of the way down it. Two thirds is
/// where a hand sits on a rifle's handguard, and clamping the result keeps a
/// 5 cm weapon from asking the off hand to occupy the same space as the on hand.
fn fore_grip_m(def: &WeaponDef) -> f32 {
    let len = def
        .muzzle_forward_m
        .clamp(0.05, weapon::MAX_MUZZLE_FORWARD_M);
    (len * 0.66).clamp(0.12, 0.60) as f32
}

/// How far in front of the character's chest an AIMED weapon is brought,
/// metres.
pub const AIM_REACH_M: f64 = 0.42;

/// The fraction of a character's height its shoulder line sits at.
///
/// The same proportion `inf_anim::BodyParams` uses for a default biped, spelled
/// here because this step has a capsule and not a rig: it reads the movement
/// component's own stand height, which is what a character IS to the mover.
const SHOULDER_OF_HEIGHT: f64 = 0.82;

/// **THE HAND PASS** (SK1c) — one `HandIk` per character, from everything this
/// step knows.
///
/// # Why one producer and not two
///
/// A weapon wants both hands and a grab wants one, and both write into the same
/// two-slot array. Two producers would race for the same slot every step and the
/// winner would be whichever ran last — so they are composed here, once, and the
/// rule between them is written down rather than emergent: **the weapon owns the
/// hand it is in, and a grab takes the other one.** A character reaching for a
/// door handle with a rifle in its right hand reaches with its left, which is
/// what a person does.
///
/// **The off hand is on loan** (SK1c audit, H1). The weapon owns the hand it is
/// *in*; the other one is supporting it, and a grab takes it back — so while a
/// grab is live the `GunGrip` hold is weighted by its complement and the
/// `rifle_fore` grip is not asked for at all. Without that the gun solve, which
/// runs *after* the reaches inside `apply_hand_ik`, overwrote the grab's reach
/// every step: an armed character's E-press moved neither wrist and the only
/// thing it did was spring the support hand open.
///
/// # What each half asks for
///
/// * **A weapon** puts a [`GunGrip`] hold on the rig — the `ik_hand_gun` path,
///   whose whole purpose is that the off hand is carried *by the weapon* rather
///   than aimed at a point in space — and closes both hands on their own
///   affordances (`rifle` in the holding hand, `rifle_fore` in the other unless a
///   grab has taken it; the trigger finger is left straight by the catalogue, not
///   by this code).
/// * **Aiming** adds a reach for the holding hand, and only then. A weapon at
///   rest hangs where the animation puts it; RMB brings it up to a point on the
///   aim line at shoulder height, which is the difference between carrying a
///   rifle and pointing one.
/// * **A grab** reaches the free hand to the interaction's own point and closes
///   it on the affordance the interactable named.
///
/// Everything is absent unless asked for, so a level nobody has armed and nobody
/// has pressed E in publishes no resource and poses exactly the bytes it did
/// before this wave.
fn step_hand_ik(world: &mut EcsWorld, dt: f64) -> (u32, u32) {
    use inf_ecs::pose::{GunGrip, HandGrip, HandIk, HandReach};

    // Age the grabs first: a grab that ended this step must not also be asked
    // for this step, or a released hand would be one step late in one host and
    // not the other if the two ever aged it in different places.
    inf_ecs::interact::step_grabs(world, dt);

    let mut holds = 0u32;
    let mut grabs = 0u32;
    for guid in gunners(world) {
        let mut req = HandIk::default();

        // **The grab is read FIRST**, because it decides whether the off hand is
        // still on the weapon (SK1c audit, H1). A live grab is one with a
        // non-zero amount; a finished one is removed by `step_grabs` above.
        let grab = inf_ecs::interact::hand_grab(world, guid)
            .map(|g| (g.amount(), g.at, g.grip.clone()))
            .filter(|(amount, _, _)| *amount > 0.0);

        // -- the weapon --
        let armed = equipped_weapon(world, guid);
        if let Some((_, def)) = armed.as_ref() {
            req.grip[1] = Some(HandGrip {
                name: inf_anim::GRIP_RIFLE.to_string(),
                amount: 1.0,
            });
            holds += 1;
            // **The off hand is on LOAN from the weapon, and a grab takes it
            // back.** The weapon owns the hand it is *in* — the right one, the
            // one `WEAPON_SOCKET` hangs off — and the other is merely supporting
            // it, which is the hand a person takes off a rifle to open a door.
            //
            // The gun hold's weight is the complement of the grab's, so the arm
            // crosses over continuously instead of snapping between the
            // fore-grip and the handle on the step the grab starts and the step
            // it ends. At `amount == 1` the hold is weightless and
            // `apply_hand_ik` skips the off-hand solve entirely.
            //
            // Before this, the `GunGrip` solve ran unconditionally and — because
            // it runs AFTER the reaches inside `apply_hand_ik` — overwrote the
            // grab's reach every step: an armed character's E-press moved
            // neither wrist by a single millimetre while `hands.1` counted it.
            let amount = grab.as_ref().map(|(a, _, _)| *a).unwrap_or(0.0);
            req.gun = Some(GunGrip {
                holding: inf_anim::BoneSide::Right,
                off_hand_offset: [0.0, 0.0, fore_grip_m(def)],
                weight: 1.0 - amount,
            });
            if grab.is_none() {
                req.grip[0] = Some(HandGrip {
                    name: inf_anim::GRIP_RIFLE_FORE.to_string(),
                    amount: 1.0,
                });
            }
            // -- and the aim, which is what MOVES it --
            if let Some(target) = aim_hold_point(world, guid) {
                req.reach[1] = Some(HandReach {
                    target: inf_ecs::math::Vec3d::new(target.x, target.y, target.z),
                    weight: 1.0,
                });
            }
        }

        // -- the grab, in whichever hand is free --
        if let Some((amount, at, grip)) = grab {
            // The weapon is in the right hand, so a grab goes to the left; an
            // unarmed character reaches with its right, which is the hand every
            // affordance in a default catalogue but `rifle_fore` is on. The
            // *slot* is what decides which hand closes — `apply_hand_ik` reads
            // it, and the affordance supplies the aperture and the curl set.
            //
            // **Honest bound**: the off hand's fingers do not cross-fade. They
            // let go of the fore-grip on the step the grab begins and close on
            // the new affordance over its ease, because one slot carries one
            // grip. A hand releasing a weapon before it takes hold of something
            // else is the right picture; doing it in one fixed step is the
            // approximation.
            let side = usize::from(armed.is_none());
            req.reach[side] = Some(HandReach {
                target: at,
                weight: amount,
            });
            req.grip[side] = Some(HandGrip { name: grip, amount });
            grabs += 1;
        }

        inf_ecs::pose::set_hand_ik(world, guid, req);
    }
    (holds, grabs)
}

/// **Where an aiming character holds its weapon**, world metres — a point on the
/// aim line at shoulder height, in front of the chest.
///
/// `None` when the character is not aiming, which is what leaves the weapon
/// wherever the animation is carrying it.
///
/// The direction goes through [`weapon::aim_forward`], which is the door the
/// shot's own direction takes — a second copy of that arithmetic would be a hand
/// that points somewhere the bullet does not. Portable for the shot's own
/// reason: this number is folded into `pose_state_bytes` and compared between
/// two machines (the P14 law).
fn aim_hold_point(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    let entity = world.entity_of(guid)?;
    let cm = world.world().get::<CharacterMovement>(entity)?;
    if cm.rotation_mode != inf_ecs::components::RotationMode::Aiming {
        return None;
    }
    let feet = feet_of(world, guid)?;
    // The stand height is what the capsule was built from, so this tracks a
    // 1.2 m character and a 2.4 m one without a second opinion about either.
    let height = (cm.stand_half_height_m * 2.0).max(0.4);
    let dir = weapon::aim_forward(cm.runtime.aim_yaw_deg, cm.runtime.aim_pitch_deg);
    let at = feet + DVec3::Y * (height * SHOULDER_OF_HEIGHT) + dir * AIM_REACH_M;
    at.is_finite().then_some(at)
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
                Some(feet) => {
                    let band = bridge.sim_band(world);
                    try_kick(world, &band, guid, feet, yaw)
                }
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
        let Some((from, yaw, pitch, from_weapon)) = aim else {
            continue;
        };
        // **The fallback, counted** (SK1b audit). A character with no pose at all
        // is the legitimate capsule case — every level committed before SK1b —
        // and is not counted. A character that *is* posed and still took the
        // capsule rule is a rig that does not publish `WEAPON_SOCKET`, or a
        // weapon entity that has not been placed yet, and it puts every shot back
        // at 1.4 m in silence. See `GameplayReport::muzzles_without_a_socket`.
        if !from_weapon && inf_ecs::pose::evaluated_pose(world, guid).is_some() {
            report.muzzles_without_a_socket += 1;
        }
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
pub fn try_kick(
    world: &mut EcsWorld,
    band: &inf_ecs::band::SimBand,
    character: Uuid,
    feet: DVec3,
    aim_yaw_deg: f64,
) -> bool {
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
    for p in super::door::placements_near(world, band) {
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
            // A kick is a leg, not a hand.
            grip: None,
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
        // **The latch goes down FIRST**, whether or not the ragdoll takes it.
        // A body whose ragdoll is refused — no `CharacterMovement`, a mode the
        // table will not leave — must not be offered again on the next step; see
        // `weapon::Downed` for the measurement.
        weapon::mark_downed(world, guid);
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
