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
    /// **Whether this attack made a noise** (wave WPN1) — `true` for a round
    /// leaving a barrel, `false` for a swing.
    ///
    /// Two consumers, and neither could be written without it. Each host's
    /// `fire_weapon_audio` queues the gunshot report only for a loud attack, so
    /// a punch does not fire a rifle's clip; and the crowd's panic takes its
    /// sources only from loud attacks, which is what the reference frames
    /// actually show — an encampment **brawl** draws bystanders who stand a metre
    /// away and watch, while a gunshot is what empties a street.
    ///
    /// A field rather than a re-lookup of the shooter's equipped weapon, because
    /// by the time either consumer runs the shooter may have scrolled: what made
    /// the noise is a property of the shot and not of whatever is in the hand
    /// afterwards.
    pub loud: bool,
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
    /// **What this step's gunfire did to the crowd** (wave WPN1) — an
    /// engagement counter, on [`crowd_doors`](Self::crowd_doors)' own terms.
    pub panic: PanicReport,
    /// **Acts recorded into the witness log** this step (wave WPN1) — an
    /// engagement counter for a seed nothing reads yet, which is exactly the
    /// case where one is worth most: without it "the witness pass runs" and
    /// "somebody saw something" are indistinguishable, and a seed that silently
    /// recorded nothing would look identical to a seed that worked.
    pub witnessed: u32,
    /// **Melee swings thrown** this step (wave WPN1) — the subset of
    /// [`shots`](Self::shots) that were an arc rather than a ray.
    ///
    /// Its own counter because the two are the same verb through the same
    /// trigger and the same clock: without it a gate cannot tell a course that
    /// punched from one that fired, and "the attack button has three consumers"
    /// is a claim about which one ran.
    pub swings: u32,
    /// **Non-fatal blows that landed on a body** this step (wave WPN1) — every
    /// one of them armed a hit reaction.
    ///
    /// An engagement counter, on `crowd_doors`' own terms: "a round was fired"
    /// and "a round hurt somebody" are different facts, and a gate that cannot
    /// tell them apart certifies a course where every shot missed.
    pub staggers: u32,
    /// **Struck bystanders who did NOT leave** this step (wave WPN1) — the
    /// resist half of the draw.
    ///
    /// Its own counter beside [`PanicReport::fled`], because a course where it
    /// is zero and a course where it is everything look identical from the flee
    /// count alone, and the whole point of a draw is that both happen.
    pub stood_their_ground: u32,
    /// **Blows heavy enough to put a body on the floor** this step — the subset
    /// of [`staggers`](Self::staggers) that also took a mode.
    ///
    /// Its own number rather than a flag on the one above, because the two
    /// answer different questions: `staggers` says the reaction seam is wired
    /// and `knockdowns` says the mode table let go — and a course where they are
    /// equal is a course where every punch is a rifle round.
    pub knockdowns: u32,
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
    // 0. **ONE band and ONE placement gather for the whole phase** (NPC1c
    //    audit, closing the wave's own carried item 2). `placements_near` visits
    //    every `DoorwaySlot` a level plans — 19 790 on the shipped city — to
    //    keep the 234 the band admits, and NPC1c gave it a second per-step
    //    caller in `step_crowd_doors`. Gathering twice is the same walk twice;
    //    the placements are a function of the level's geometry and the band, and
    //    nothing between the two calls moves either.
    let band = bridge.sim_band(world);
    let places = super::door::placements_near(world, &band);
    // 0a. **An NPC opens the door in its way** (island wave NPC1c). Before the
    //    leaves move, so a crowd agent's press has the same immediacy the
    //    player's E already has -- that one is consumed in
    //    `step_character_movement`, a phase earlier than this whole function.
    //    Inert on every level with no crowd: one absent-resource read.
    report.crowd_doors = step_crowd_doors(world, &places);
    // 1. The doors move first, because a kick armed on a previous step lands in
    //    step 3 and must find the leaf where this step's solver will.
    let doors = super::door::step_doors_with(world, bridge, dt, places);
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
    // 3d. **The street hears it** (wave WPN1) — the crowd's panic, from this
    //     step's own gunfire. After the weapons, because it reads their hits;
    //     before the deaths, so a body that is about to be handed to the ragdoll
    //     is not also given a route to walk. `O(agents)` with a bounded inner
    //     loop and inert on a level with no gunfire — see `step_panic` for the
    //     cost and for the one-step latency the crowd's phase ordering implies.
    //
    //     **MERGED, not assigned** — and the arm that found this is
    //     `a_struck_bystander_either_runs_or_stands_its_ground`. A struck
    //     bystander's own flee is counted in `apply_hit`, several passes
    //     earlier; `report.panic = step_panic(…)` erased it, so the latch said
    //     an agent was running and the counter said nobody had. A counter that
    //     disagrees with the world is worse than no counter.
    let panic = step_panic(world, &report.hits, dt);
    report.panic.sources = panic.sources;
    report.panic.considered = panic.considered;
    report.panic.fled += panic.fled;
    // 4. Every body that stopped working goes to the ragdoll — the P29.4
    //    bridge's own door, whose doc has named "a damage system" as its
    //    intended caller since it was written.
    let killed = step_deaths(world, bridge, &mut report);
    // 5. **Who saw it** (wave WPN1) — the witnessed-act seed. LAST, because a
    //    death outranks a shot in the record and the deaths are only known
    //    once the pass above has run. Inert on every step nothing happened on.
    report.witnessed = step_witness(
        world,
        bridge,
        &report.hits,
        &killed,
        inf_ecs::traffic::steps(world),
    );
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
///
/// `placements` is the phase's **one** gather — [`step_gameplay`] takes it once
/// and hands the same list to this pass and to `step_doors` (the NPC1c audit;
/// the wave gathered twice and carried the fix by name). It is the same door
/// `candidates` and the player's prompt read, so an NPC cannot reach a door the
/// player is told is out of reach.
pub fn step_crowd_doors(
    world: &mut EcsWorld,
    placements: &[inf_ecs::door::DoorPlacement],
) -> CrowdDoorReport {
    let mut report = CrowdDoorReport::default();
    let blocked = inf_ecs::crowd::blocked_agents(world);
    if blocked.is_empty() {
        return report;
    }
    report.considered = blocked.len();
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
        for p in placements {
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

/// **How far a shot drives the weapon back into the shoulder**, metres (wave
/// WPN1).
///
/// Six centimetres against [`AIM_REACH_M`]'s forty-two — a seventh of the reach,
/// which is enough to read as a shove at 60 Hz and small enough that the hand
/// stays in front of the chest rather than passing through it.
pub const RECOIL_PULL_M: f64 = 0.06;

/// **How far a shot lifts the muzzle**, degrees.
///
/// Four. The hold point is on the aim line at shoulder height, so lifting the
/// *point* is what lifts the weapon; four degrees at forty-two centimetres is
/// three centimetres of rise, which is a muzzle climbing rather than a weapon
/// being thrown over a shoulder.
pub const RECOIL_RISE_DEG: f64 = 4.0;

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
///
/// # THE RECOIL IS HERE, and it is a POSE (wave WPN1)
///
/// A shot lifts this point by [`RECOIL_RISE_DEG`] and pulls it back by
/// [`RECOIL_PULL_M`], scaled by [`weapon::recoil_fraction`] — which is the
/// weapon's own cycle and therefore already sim state on both hosts. It reaches
/// `HandIk::reach`, then `apply_hand_ik`, then `pose_state_bytes`, so the two
/// hosts are compared on it byte for byte and a replay reproduces it.
///
/// **It does NOT move the aim, and that is deliberate.** `cm.runtime.aim_*` is
/// what the bullet leaves along and what the camera chases; this reads those two
/// numbers and writes neither. A shot fired during a burst goes exactly where the
/// player is pointing while the weapon visibly climbs, which is a game with a
/// climbing weapon and honest aim rather than one with recoil.
///
/// # Why there is no CAMERA recoil, stated as a ruling
///
/// The obvious companion — kick the camera's pitch and let it settle — was
/// priced and refused, and the refusal has two halves that meet:
///
/// * a camera kick that does **not** move the aim makes the reticle **lie**. The
///   shot leaves along `aim_pitch_deg`, the reticle is drawn at the centre of the
///   screen, and the camera's pitch is the entire mapping between the two; kick
///   one and not the other and the crosshair stops being where the rounds go,
///   permanently under sustained fire.
/// * a camera kick that **does** move the aim is a camera → sim write, which is
///   the one thing `d3::camera`'s Ruling 4 exists to forbid ("`ViewMode` never
///   crosses the sim wire, and there is no camera → sim path at all"), and which
///   `phase29_gate` pins by running the same course under two different cameras
///   and comparing the sim trace.
///
/// The honest form is an **aim** recoil: the movement step's own aim integrator
/// gaining a per-shot impulse and a recovery, so the aim really moves, the camera
/// follows it because it always did, and the reticle stays true. That is a change
/// to `step_character_movement`'s look integration and it moves every committed
/// aim in the tree; it is named on this wave's carried list rather than done
/// quietly at the end of a different clause.
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
    // The recoil, read off the weapon's own cycle. Zero — and every byte below
    // identical to what it was before this wave — for a character whose weapon
    // is not cooling, which is every step but the ten a second a rifle fires on.
    let kick = recoil_of(world, guid);
    let dir = weapon::aim_forward(
        cm.runtime.aim_yaw_deg,
        cm.runtime.aim_pitch_deg + RECOIL_RISE_DEG * kick,
    );
    let reach = (AIM_REACH_M - RECOIL_PULL_M * kick).max(0.05);
    let at = feet + DVec3::Y * (height * SHOULDER_OF_HEIGHT) + dir * reach;
    at.is_finite().then_some(at)
}

/// **How much recoil is on this character's weapon**, `[0, 1]` — the equipped
/// definition and the live ammunition clock, through the one Ring-0 rule.
///
/// `0.0` for a character with nothing equipped or no clock, which is the answer
/// that leaves [`aim_hold_point`] byte-identical to its pre-WPN1 self.
fn recoil_of(world: &EcsWorld, guid: Uuid) -> f64 {
    let Some(entity) = world.entity_of(guid) else {
        return 0.0;
    };
    let Some((_, def)) = equipped_weapon(world, guid) else {
        return 0.0;
    };
    world
        .world()
        .get::<WeaponState>(entity)
        .map(|s| weapon::recoil_fraction(&def, s))
        .unwrap_or(0.0)
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
        // **THE ATTACK BUTTON'S THREE VERBS**, arbitrated once, here.
        //
        // A locked door in kicking reach takes the press; anything else lets it
        // through to the trigger; and since wave WPN1 an empty hand is a trigger
        // too. It is decided on the **edge** and not the level, because a kick is
        // a press — and while a kick is pending the weapon does not fire, so
        // holding the button against a door kicks it rather than kicking and
        // then shooting it.
        //
        // The order is the arbitration and it is not arbitrary: a kick beats a
        // punch, because a player standing at a locked gate pressing the attack
        // button wants the gate open and not a bruised hand. `try_kick` refuses
        // every door that is not locked and in reach, so the punch is what the
        // press means everywhere else.
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
        // Now the weapon, if there is one — **or the fists, if there is not and
        // the button is down** (wave WPN1).
        //
        // A pair of hands is the third consumer of the edge and it goes through
        // the same `try_fire`, the same cooldown and the same
        // `weapon_state_bytes` a rifle does. The one thing it does not do is
        // arrive uninvited: the clock is installed on the first press and not at
        // spawn, so an unarmed character that has never thrown a punch carries
        // no `WeaponState` at all and every trace committed before this wave is
        // byte-identical. Once installed it stays, because that clock is what
        // stops the second punch arriving before the first has landed.
        let armed = equipped_weapon(world, guid);
        let punching = armed.is_none()
            && (want_fire
                || world
                    .world()
                    .get::<WeaponState>(entity)
                    .is_some_and(|s| s.item_id == weapon::FIST_ITEM));
        let Some((item_id, def)) =
            armed.or_else(|| punching.then(|| (weapon::FIST_ITEM.to_string(), weapon::fist_def())))
        else {
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
        // The animation, and it is a different one for a swing: a rig that
        // played `weapon_fire` when somebody threw a punch would be firing an
        // empty hand.
        inf_ecs::anim_bridge::set_anim_trigger(
            world,
            guid,
            if def.is_melee() {
                weapon::MELEE_TRIGGER
            } else {
                weapon::FIRE_TRIGGER
            },
        );
        report.shots += 1;
        if def.is_melee() {
            report.swings += 1;
        }
        let Some((from, yaw, pitch, from_weapon)) = aim else {
            continue;
        };
        // **The fallback, counted** (SK1b audit). A character with no pose at all
        // is the legitimate capsule case — every level committed before SK1b —
        // and is not counted. A character that *is* posed and still took the
        // capsule rule is a rig that does not publish `WEAPON_SOCKET`, or a
        // weapon entity that has not been placed yet, and it puts every shot back
        // at 1.4 m in silence. See `GameplayReport::muzzles_without_a_socket`.
        //
        // **A punch is not counted either** (wave WPN1), and it would have been:
        // a fist has no weapon entity to hang off a socket, so a rigged
        // character throwing one takes the capsule rule *correctly* and would
        // have tripped the tripwire on every swing. A gate that asserts this is
        // zero on a rigged course has to be able to punch on it.
        if !def.is_melee() && !from_weapon && inf_ecs::pose::evaluated_pose(world, guid).is_some() {
            report.muzzles_without_a_socket += 1;
        }
        let dir = weapon::shot_direction(&def, yaw, pitch, shot_index);
        let hit = if def.is_melee() {
            resolve_swing(world, guid, &def, from, dir, yaw)
        } else {
            resolve_shot(world, bridge, guid, &def, from, dir)
        };
        apply_hit(world, &hit, dt, report);
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
            let on_flesh = target.is_some_and(|g| is_flesh(world, g));
            WeaponHit {
                shooter,
                target,
                from,
                to: from + dir * h.toi,
                energy_j: def.damage_j,
                on_flesh,
                loud: true,
            }
        }
        None => WeaponHit {
            shooter,
            target: None,
            from,
            to: from + dir * range,
            energy_j: def.damage_j,
            on_flesh: false,
            loud: true,
        },
    }
}

/// **How far away an act can be seen**, metres.
///
/// A hundred and twenty — wider than the panic radius, because seeing something
/// and running from it are different distances in the other direction too: the
/// person across the square who did not run is still the person who can describe
/// you. It is the [`weapon::REPORT_MAX_M`] / [`PANIC_RADIUS_M`] pair's third
/// number and it sits between them on purpose.
pub const WITNESS_RADIUS_M: f64 = 120.0;

/// **How many acts one step may record.**
///
/// Four, and it is a cost bound: each act is one `O(agents)` walk plus at most
/// [`inf_ecs::witness::MAX_OBSERVERS`] rays, so this is what keeps a firefight's
/// bookkeeping a constant multiple of a walk rather than a function of how many
/// people are shooting. A step that produced more records the first four; the
/// rest are simply not written, which for a seed nothing reads yet is the honest
/// trade and is stated rather than hidden.
pub const MAX_ACTS_PER_STEP: usize = 4;

/// **How far a gunshot scatters a crowd**, metres.
///
/// Forty-five. Deliberately much smaller than a gunshot's own audible reach
/// ([`weapon::REPORT_MAX_M`], 250 m), because hearing a shot and running from one
/// are different distances: a person three streets away turns their head, and a
/// person on the same corner leaves. The two numbers are related on purpose —
/// see [`weapon::REPORT_MAX_M`]'s own doc — so a designer who widens one can see
/// what it is being compared against.
pub const PANIC_RADIUS_M: f64 = 45.0;

/// **How far a frightened bystander walks**, metres.
///
/// Half again as far as a carjacked driver goes ([`inf_ecs::crowd::FLEE_M`]),
/// because a gunshot is worth more distance than an argument, and short enough
/// that the route is still one straight leg rather than a plan.
pub const PANIC_FLEE_M: f64 = 60.0;

/// **How many distinct places one step may panic a crowd from.**
///
/// The pass is `O(agents × sources)` and this is what makes the second factor a
/// constant rather than a firefight's shooter count: shots inside half a panic
/// radius of each other are one source, and past this many the rest of the step's
/// shots are folded into the nearest. Eight is more than a street fight has
/// distinct corners, and it bounds the walk at 8 × 1 000 agents = 8 000 squared
/// distances a step against `NPC_STEP_BUDGET_MS`'s own 1 000-agent figure.
pub const MAX_PANIC_SOURCES: usize = 8;

/// **The distinct places this step's gunfire came from**, coalesced and capped.
///
/// The half of [`step_panic`] that decides its cost, hoisted so a gate can
/// measure it without a crowd: shots inside half a [`PANIC_RADIUS_M`] of each
/// other frighten the same people and are one source, and past
/// [`MAX_PANIC_SOURCES`] the rest of the step's shots are simply not distinct
/// places — which is what keeps the pass's inner loop a constant rather than a
/// function of how many people are shooting.
///
/// A **brawl is not a source**: see [`WeaponHit::loud`] and the reference frames
/// it cites.
fn panic_sources(hits: &[WeaponHit]) -> Vec<DVec3> {
    let mut sources: Vec<DVec3> = Vec::new();
    for hit in hits.iter().filter(|h| h.loud && h.from.is_finite()) {
        if sources.len() >= MAX_PANIC_SOURCES {
            break;
        }
        if sources
            .iter()
            .any(|s| (*s - hit.from).length() < PANIC_RADIUS_M * 0.5)
        {
            continue;
        }
        sources.push(hit.from);
    }
    sources
}

/// **How many distinct places this step's gunfire came from** — `panic_sources`
/// counted, for a gate that wants the cost bound without a population.
///
/// (A code span rather than a link: the function it names is private, and a
/// public doc that links to one is a rustdoc warning for a reference no reader
/// of the public API could follow anyway.)
pub fn panic_sources_for(hits: &[WeaponHit]) -> usize {
    panic_sources(hits).len()
}

/// What one `step_panic` did — a code span for `panic_sources_for`'s reason:
/// the pass is private and this type is not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PanicReport {
    /// Distinct places this step's gunfire came from, after coalescing.
    pub sources: usize,
    /// Agents the pass looked at — the whole population, which is the cost.
    pub considered: usize,
    /// Agents that started running on this step. `considered - fled` is mostly
    /// distance, and it is the number that says the radius is doing work.
    pub fled: usize,
}

/// **A GUNSHOT SCATTERS THE STREET** (wave WPN1) — the crowd's panic, through
/// the one flee door.
///
/// # The cost, stated, because it is the whole design
///
/// The obvious shape is "for each shot, for each agent, how far apart are they",
/// which is `O(shots × agents)` — and at 600 rpm with several shooters that is a
/// per-step cost that grows with how exciting the scene is, measured against an
/// `inf_player::budget::NPC_STEP_BUDGET_MS` that was set at a thousand *walking*
/// agents. (Named in a code span rather than linked: `inf-physics` does not
/// depend on `inf-player`, so an intra-doc link there resolves to nothing and
/// costs a rustdoc warning for a reference a reader can follow by name.)
///
/// So this is **one walk over the population** with a bounded inner loop: the
/// step's loud shots are coalesced into at most [`MAX_PANIC_SOURCES`] places
/// first (two shots inside half a radius of each other frighten the same people),
/// and the agents are read off `CrowdPopulationRes`'s own records — the
/// `blocked_agents` shape, `O(agents)` and allocation-free on a level with no
/// crowd, rather than `O(entities)` over a furnished town.
///
/// # The one-step latency, stated
///
/// The crowd steers in phase 5 (`crowd`) and this runs in phase 14 (`gameplay`),
/// so an agent frightened here starts moving on the **next** fixed step: 16.7 ms
/// at 60 Hz. It is the same lag in both hosts — the pass is one Ring-0 rule they
/// both call from the same slot — so PIE == shipping is unaffected and no trace
/// can see it. Running the panic before the crowd would fix it and would put the
/// gunfire pass before the shots that feed it, which is a full step of lag the
/// other way. The sentence is `muzzle_of`'s own, one system along.
///
/// Inert on a level with no gunfire and on a level with no population: two early
/// returns and no allocation.
fn step_panic(world: &mut EcsWorld, hits: &[WeaponHit], dt: f64) -> PanicReport {
    let mut report = PanicReport::default();
    // 1. The sources, coalesced.
    let sources = panic_sources(hits);
    if sources.is_empty() {
        return report;
    }
    report.sources = sources.len();
    // 2. ONE walk over the population, reading where each agent stood at the end
    //    of the crowd's own step.
    let mut want: Vec<(Uuid, DVec3, DVec3)> = Vec::new();
    {
        let Some(pop) = world
            .world()
            .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
        else {
            return report;
        };
        report.considered = pop.records.len();
        for (guid, rec) in &pop.records {
            let here = rec.last;
            let mut best: Option<(f64, DVec3)> = None;
            for s in &sources {
                let d = (*s - here).length();
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, *s));
                }
            }
            let Some((d, at)) = best else { continue };
            if d > PANIC_RADIUS_M {
                continue;
            }
            want.push((*guid, here, at));
        }
    }
    // 3. …and the flee itself, through the one door, in `Guid` order (the
    //    `BTreeMap` walk above), so two hosts scatter the same people.
    for (guid, here, away_from) in want {
        if inf_ecs::crowd::flee_from(world, guid, here, away_from, dt, PANIC_FLEE_M) {
            report.fled += 1;
        }
    }
    report
}

/// **WHO SAW IT** (wave WPN1) — the witnessed-act seed for the EMS arc.
///
/// One record per act, at most [`MAX_ACTS_PER_STEP`] a step, each carrying the
/// nearest few crowd agents that have a clear line to it. Nothing reads it yet;
/// see [`inf_ecs::witness`] for why it is written now anyway.
///
/// # The line of sight, and what it costs
///
/// One `cast_ray_excluding` per candidate observer, from the observer's own eye
/// to the act — which is the same query the audio occlusion path makes and the
/// same one `resolve_shot` makes, so it is the engine's existing answer to "is
/// there something between these two points" rather than a fourth. Bounded at
/// `MAX_ACTS_PER_STEP × MAX_OBSERVERS` = 32 rays a step in the worst case, and
/// **zero** on every step nothing happened on.
///
/// A `Dormant` observer has no collider to exclude and the ray simply runs
/// without one, which is right: an agent with no body cannot be in its own way.
///
/// Returns how many acts were recorded — an engagement counter, because "the
/// pass ran" and "somebody saw something" are different facts.
fn step_witness(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    hits: &[WeaponHit],
    killed: &[Uuid],
    step: u64,
) -> u32 {
    use inf_ecs::witness::{ActKind, WitnessedAct};
    // The acts, in the order they happened: a death outranks a shot, because a
    // dispatcher asked to name one thing about a street should name the body.
    let mut acts: Vec<(ActKind, Uuid, DVec3)> = Vec::new();
    for guid in killed {
        if acts.len() >= MAX_ACTS_PER_STEP {
            break;
        }
        let Some(at) = strike_point(world, *guid) else {
            continue;
        };
        acts.push((ActKind::Killed, *guid, at));
    }
    for hit in hits.iter().filter(|h| h.loud && h.from.is_finite()) {
        if acts.len() >= MAX_ACTS_PER_STEP {
            break;
        }
        acts.push((ActKind::Shot, hit.shooter, hit.from));
    }
    if acts.is_empty() {
        return 0;
    }
    let mut recorded = 0u32;
    for (kind, actor, at) in acts {
        let candidates = inf_ecs::witness::candidates_near(world, at, WITNESS_RADIUS_M);
        let mut observers: Vec<Uuid> = Vec::new();
        for (guid, feet) in candidates {
            if guid == actor {
                continue;
            }
            let eye = feet + DVec3::Y * MUZZLE_HEIGHT_M;
            let to = at - eye;
            let d = to.length();
            if !d.is_finite() {
                continue;
            }
            if d > 1.0e-6 {
                let mut exclude = BTreeSet::new();
                if let Some(c) = bridge.collider_of(guid) {
                    exclude.insert(c);
                }
                if let Some(c) = bridge.collider_of(actor) {
                    exclude.insert(c);
                }
                // Anything in the way at all: the ray is stopped short of the
                // act, so the observer cannot see it. The tolerance is a
                // centimetre, which is the wall a shot leaves through.
                if bridge
                    .world_mut()
                    .cast_ray_excluding(eye, to / d, d, &exclude)
                    .is_some_and(|h| h.toi < d - 0.01)
                {
                    continue;
                }
            }
            observers.push(guid);
        }
        inf_ecs::witness::record_act(
            world,
            WitnessedAct {
                kind,
                actor,
                at,
                step,
                observers,
                actor_look: inf_ecs::witness::look_digest(actor),
                actor_vehicle: inf_ecs::witness::actor_vehicle(world, actor),
            },
        );
        recorded += 1;
    }
    recorded
}

/// **Where a swing lands on a body**, world metres — the point a punch is aimed
/// at and measured to.
///
/// [`MUZZLE_HEIGHT_M`] above the feet, which is the same height a rig-less
/// character's own shot leaves from. Both ends of a swing are therefore measured
/// at chest height, so the vertical term cancels between two characters of the
/// same size and a punch at a metre is a punch at a metre rather than
/// `sqrt(1² + 1.4²)`.
fn strike_point(world: &EcsWorld, guid: Uuid) -> Option<DVec3> {
    Some(feet_of(world, guid)? + DVec3::Y * MUZZLE_HEIGHT_M)
}

/// **Resolve a SWING** — a reach and an arc, not a ray (wave WPN1).
///
/// # One door, and it is the interaction rule's
///
/// The reach and the cone go through [`inf_ecs::interact::resolve`], which is
/// what the E-key prompt, the door press and `try_kick` already resolve through.
/// Writing the arithmetic a second time here is exactly the defect `try_kick`'s
/// own doc names — *"spelling it a second way would let a player kick a door the
/// prompt says is out of reach"* — one verb along: a punch that could land on
/// somebody the prompt calls unreachable is a punch through a wall.
///
/// It also buys the portable trigonometry for free: the cone test goes through
/// `inf_math::patan2_64` and the boundary epsilon that exists because of it (the
/// P14 law), so a swing lands identically on two machines.
///
/// # What it does NOT do
///
/// * **No line of sight.** A body on the far side of a shut door within reach is
///   hit. The reach is 1.2 m and a leaf is 5 cm thick, so this is reachable in
///   principle, and closing it is one `cast_ray_excluding` per candidate — which
///   this function deliberately does not spend on a press that resolves at most
///   one target. Carried by name.
/// * **No cleave.** The nearest body in the arc takes the blow and nobody else
///   does, which is `resolve`'s own rule (*"the first of two equals wins"*). A
///   swing that hit everything in its cone is a different weapon and would want
///   its own `WeaponDef` field.
///
/// `O(characters)`, over the same walk [`gunners`] already makes — and only on
/// the steps a swing actually leaves, which at [`weapon::FIST_RPM`] is at most
/// one and a half a second.
fn resolve_swing(
    world: &EcsWorld,
    shooter: Uuid,
    def: &WeaponDef,
    from: DVec3,
    dir: DVec3,
    yaw_deg: f64,
) -> WeaponHit {
    use inf_ecs::interact::{InteractCandidate, InteractVerb};
    let reach = def.reach_m();
    let arc = def.melee_arc_deg.clamp(0.0, 360.0);
    let mut candidates: Vec<InteractCandidate> = Vec::new();
    for guid in gunners(world) {
        if guid == shooter {
            continue;
        }
        let Some(position) = strike_point(world, guid) else {
            continue;
        };
        candidates.push(InteractCandidate {
            guid,
            // The verb is not read by anything downstream — this resolution
            // answers "which body" and nothing else — so it carries the neutral
            // one `try_kick` gives a door rather than inventing a fourth.
            verb: InteractVerb::Use,
            label: String::new(),
            position,
            range_m: reach,
            view_cone_deg: arc,
            grip: None,
        });
    }
    // `gunners` is already `Guid`-ordered, so ties break on the guid — two
    // bodies at exactly one distance answer the same one on both hosts.
    match inf_ecs::interact::resolve(&candidates, from, yaw_deg) {
        Some(hit) => WeaponHit {
            shooter,
            target: Some(hit.guid),
            from,
            to: hit.position,
            energy_j: def.damage_j,
            // Every candidate here IS a character, which is what `is_flesh`
            // answers `true` for — so this is a fact rather than an assumption.
            on_flesh: true,
            loud: false,
        },
        None => WeaponHit {
            shooter,
            target: None,
            from,
            // A miss ends at the end of the reach along the aim, so a tracer and
            // a debug line draw the swing rather than nothing.
            to: from + dir * reach,
            energy_j: def.damage_j,
            on_flesh: false,
            loud: false,
        },
    }
}

/// **Is this thing a body?** — the question a round asks about what it hit
/// (wave WPN1).
///
/// # It used to be "does it have a `Health` component", and that was the silent
/// shot
///
/// I6 gave the hero a body from its own Blueprint and gave **nothing else** one.
/// So a round into an NPC, a crowd agent or any other character was `on_flesh ==
/// false`, went to the destructible branch, and was owed to an entity with no
/// `Destructible` — where the host logged a `NoDestructible` refusal, once per
/// round, ten times a second on a held trigger. The person was unhurt, the log
/// was full, and the only visible symptom was the flood.
///
/// So the question is now **"is it a character"**, and [`apply_hit`] gives it a
/// body on the first hit that lands. `CharacterMovement` is the one component in
/// this engine that means *a person*: the crowd puts it on every materialized
/// agent, the movement step visits exactly the entities that carry it, and
/// nothing else in the tree has one.
///
/// A body that already has [`Health`] still answers `true` through the first arm,
/// so a level that authored one — the gameplay fixture's hero, every
/// `health.set` a Blueprint calls — is byte-identical to what it was.
fn is_flesh(world: &EcsWorld, guid: Uuid) -> bool {
    let Some(e) = world.entity_of(guid) else {
        return false;
    };
    let w = world.world();
    w.get::<Health>(e).is_some() || w.get::<CharacterMovement>(e).is_some()
}

/// Spend a hit's joules: on a body's health here, on a destructible through the
/// host's own wrapper.
///
/// # Lazy health, and what it buys (wave WPN1)
///
/// A character with no [`Health`] is given one **on the first hit that lands on
/// it**, at [`weapon::DEFAULT_VITALITY_J`]. Two alternatives were available and
/// both are worse:
///
/// * giving every character a body at spawn puts 33 bytes an agent per step into
///   `health_state_bytes` — **33 kB a step at the thousand agents
///   `NPC_BUDGET_AGENTS` measures** — and moves every committed trace in the tree
///   for levels that have a crowd and no combat, which is all of them;
/// * giving one to every *materialized* agent makes the trace a function of the
///   crowd BAND, which is the tier-dependent-component trap NPC1a's own
///   `crowd_state_bytes` exists to keep out: an agent would enter and leave the
///   health section as the player walked towards and away from it, and two hosts
///   that tiered it a step apart would diverge for a reason that has nothing to
///   do with anybody's health.
///
/// Lazily, the section is empty until something is shot, which keeps every
/// pre-WPN1 trace byte-identical, and once a body is in it it stays there
/// however the band moves.
fn apply_hit(world: &mut EcsWorld, hit: &WeaponHit, dt: f64, report: &mut GameplayReport) {
    let Some(target) = hit.target else {
        return;
    };
    if hit.on_flesh {
        // The body, if this is the first thing that has ever hurt it.
        if weapon::health_of(world, target).is_none() {
            weapon::give_health(world, target, weapon::DEFAULT_VITALITY_J);
        }
        let before = weapon::health_of(world, target)
            .map(|h| h.joules)
            .unwrap_or(0.0);
        // One door (`weapon::damage_entity`), shared with the `health.damage`
        // verb, so a bullet and a script spend joules the same way.
        let Some(r) = weapon::damage_entity(world, target, hit.energy_j) else {
            return;
        };
        if r.was_dead || r.killed {
            // A corpse does not stagger and a kill is the ragdoll's business —
            // `step_deaths` hands it over on this same step, and a hit reaction
            // armed on the way out would be an animation fighting a ragdoll for
            // the same skeleton.
            return;
        }
        stagger(world, hit, target, r.absorbed_j, before, dt, report);
        return;
    }
    // Not flesh. **Only a destructible is owed anything** (wave WPN1): a round
    // into a lamp post, a kerb or the ground reached this list before, and the
    // host answered every one of them with a `NoDestructible` refusal in the log.
    // Owing energy to something with no door is not owing.
    if world
        .entity_of(target)
        .and_then(|e| world.world().get::<inf_ecs::components::Destructible>(e))
        .is_none()
    {
        return;
    }
    // The host spends it at the P22 door. Coalesced by entity so one
    // burst on one wall is one blow — which matters, because **damage is not
    // banked**: three small blows on one step are not a big one, and pretending
    // otherwise here would make the rate of fire a hidden multiplier on damage.
    if let Some(slot) = report.destruct.iter_mut().find(|(g, _)| *g == target) {
        slot.1 += hit.energy_j;
    } else {
        report.destruct.push((target, hit.energy_j));
    }
}

/// **WHAT A PERSON DOES ABOUT BEING HIT** (wave WPN1) — the resist draw, and
/// the one flee door.
///
/// # Why the same draw the carjack uses
///
/// `carjack::RESIST_CHANCE` already answers *"does this person fight you off
/// this time"*, drawn per attempt from the victim's own guid and the sim step —
/// a function of who they are and when you tried, agreed by both hosts, and
/// deliberately **not** a stored counter, because a counter of how many times
/// somebody has resisted is a second copy of what the seed already answers. A
/// punch is the same question with a different verb, so it takes the same draw
/// against a salt of its own: a quarter of the time somebody who is hit stands
/// their ground, and the rest of the time they leave.
///
/// It is a draw and not a certainty because the reference frames show both —
/// the encampment brawl (`frames/police-bike/0033`) has a bystander standing a
/// metre from a fight watching it, and it also has people who are plainly not
/// there any more. A rule that always fled would empty a brawl of everybody but
/// the two people in it.
///
/// # What it is NOT
///
/// It is not a fight-back: an NPC that resists simply stays, and the day one
/// swings back is the day `npc_aim_at` grows a policy — which is EMS3's, for
/// the reason that function's own doc gives. And it reaches
/// [`inf_ecs::crowd::flee_from`], so a person who is not in the population is
/// not made one:
/// the hero, a scripted actor and a shopkeeper with no crowd record are all
/// refused by that door, and the honest answer for them is that being hit does
/// not give them somewhere to be.
fn struck_reaction(
    world: &mut EcsWorld,
    hit: &WeaponHit,
    target: Uuid,
    dt: f64,
    report: &mut GameplayReport,
) {
    // Only somebody the crowd knows about: `flee_from` refuses the rest, and
    // asking first is what keeps this `O(1)` on a hit against the hero.
    if !inf_ecs::crowd::is_in_population(world, target)
        || inf_ecs::crowd::is_panicked(world, target)
    {
        return;
    }
    let tick = inf_ecs::traffic::steps(world);
    if inf_ecs::crowd::agent_unit(target, tick, SALT_STRUCK) < super::carjack::RESIST_CHANCE {
        report.stood_their_ground += 1;
        return;
    }
    let Some(from) = feet_of(world, target) else {
        return;
    };
    // Away from where the blow came from — the attacker's own muzzle, which is
    // the one point in a `WeaponHit` that is always the attacker's.
    if inf_ecs::crowd::flee_from(world, target, from, hit.from, dt, PANIC_FLEE_M) {
        report.panic.fled += 1;
    }
}

/// Salts the "does being hit make you leave" draw — `carjack::SALT_RESIST`'s
/// shape, with its own constant so a person who resisted a carjack is not
/// thereby the person who stands their ground when punched.
const SALT_STRUCK: u64 = 0x5354_5255_434b_0001;

/// **A hit that hurts is a hit that shows** (wave WPN1) — the one-shot reaction,
/// and the blow that puts a body on the floor.
///
/// Two things, and the second one is a mode:
///
/// * the animation trigger ([`weapon::STAGGER_TRIGGER`]) is armed on **every**
///   non-fatal blow, through the same `set_anim_trigger` seam the fire and the
///   reload use. A character with no state machine plays nothing and still takes
///   the damage — the reload's rule verbatim;
/// * a blow that takes [`weapon::STAGGER_FRACTION`] of what the body had left
///   also **puts it off its feet**, into `MovementMode::FallControlled`. That is
///   the carjack's own eject verbatim (`carjack.rs`: *"being pulled out of a car
///   is a fact about your body and not a choice"*) and it is what
///   `transition_is_legal`'s own doc has been describing since P29.3.
///
/// The mode is asked of the table rather than assigned: a swimmer, a ragdoll and
/// a driver all refuse it, and a refusal is a value.
fn stagger(
    world: &mut EcsWorld,
    hit: &WeaponHit,
    target: Uuid,
    absorbed_j: f64,
    before_j: f64,
    dt: f64,
    report: &mut GameplayReport,
) {
    inf_ecs::anim_bridge::set_anim_trigger(world, target, weapon::STAGGER_TRIGGER);
    report.staggers += 1;
    // **…and whoever it happened to may decide to leave** (wave WPN1). The
    // draw, and the flee, are `struck_reaction`'s.
    struck_reaction(world, hit, target, dt, report);
    if !weapon::is_staggering(absorbed_j, before_j) {
        return;
    }
    let Some(entity) = world.entity_of(target) else {
        return;
    };
    let Some(mut cm) = world
        .world_mut()
        .get_mut::<inf_ecs::components::CharacterMovement>(entity)
    else {
        return;
    };
    if cm.runtime.seat.is_seated()
        || !inf_ecs::movement::transition_is_legal(
            cm.mode,
            inf_ecs::components::MovementMode::FallControlled,
        )
    {
        return;
    }
    cm.mode = inf_ecs::components::MovementMode::FallControlled;
    cm.runtime.time_in_mode_s = 0.0;
    report.knockdowns += 1;
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

/// **Every body that stopped working goes to the ragdoll** — and nothing gets up
/// again.
///
/// # RULING: what a respawn would be, and why it is not here (I6 item 7)
///
/// The hero can be killed — `phase30_gameplay_gate` proves a round takes 1 700 J
/// off a 2 000 J body and `weapon::Downed` latches the handoff — and when it is,
/// the level has a ragdoll in it and no player. The mandate asks for a respawn.
/// Named rather than built, because the shape matters more than the code:
///
/// **The simplest honest form is a re-seat, not a reload.** On the step a
/// `player_controlled` body is handed to the ragdoll, a host would: end the
/// ragdoll through `ragdoll_bridge`'s own door (the table already permits
/// `(Ragdoll, Grounded)`), restore [`Health`] to its capacity, place the body at
/// the level's own start — the `StreamingSource`-carrying spawn the scene
/// already names — and clear the movement runtime's edges the way `clear_edges`
/// does at a seat. Nothing else. **The world keeps everything that happened**:
/// the doors stay where they were kicked, the bag keeps what it held, the debris
/// stays on the floor, the crowd stays scattered.
///
/// That is deliberate rather than lazy, and it is the reason there is no save
/// container. Restoring a *world* means a snapshot of it, and this engine's one
/// snapshot format is `.inf_lvl` — **the author's document**, which P21's own
/// ruling forbids the runtime to write ("in the editor the render store IS the
/// save's staging source"). A respawn that rolled the world back would need a
/// second, runtime-owned container with its own schema, its own migration and
/// its own gate; a respawn that does not is four calls into doors that already
/// exist. The second is a game, and the first is a wave.
///
/// It is not written here because *where* it goes is a decision this function
/// cannot make: reviving the camera subject belongs to whoever owns the camera
/// subject, and today that is each host. Carried by name.
///
/// Answers **who stopped working on this step**, in `Guid` order — the list the
/// witness pass records a death from, and the reason this returns anything at
/// all: `report.kills` counts the ones the ragdoll *took*, and a body the
/// ragdoll refused is still a body somebody saw fall.
fn step_deaths(
    world: &mut EcsWorld,
    _bridge: &mut PhysicsBridge3D,
    report: &mut GameplayReport,
) -> Vec<Uuid> {
    let dead = weapon::newly_dead(world);
    for guid in dead.iter().copied() {
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
    dead
}

/// Which side of `door` a character standing at `feet` is on — re-exported for
/// the hosts' prompt, so the press and the prompt read one function.
pub fn side_of(world: &EcsWorld, door_guid: Uuid, feet: DVec3) -> Option<DoorSide> {
    let p = super::door::placement_of(world, door_guid)?;
    Some(p.spec.side_of(p.hinge, feet))
}

/// **AN ARMED NPC AIMS AND FIRES** (wave WPN1) — the one door an intent that is
/// not the local player's crosses into a character.
///
/// # Why a door at all
///
/// `inf_ecs::movement::apply_intent` writes every field this writes — and it
/// writes them **only onto `player_controlled` characters**, which is the line
/// that makes an NPC's body the crowd's business and not the input layer's. So
/// an armed NPC had no way to pull a trigger at all: nothing in the tree could
/// write `want_attack` onto one.
///
/// This is the complement, and it is deliberately the same shape as VEH2b's
/// `drive_intent` door: **one function, refusing the other half of the world.**
/// `apply_intent` refuses everything that is not player-controlled; this refuses
/// everything that is. Between them every character's intent has exactly one
/// author, which is the property that makes a divergence findable.
///
/// # What it does NOT do
///
/// No cover, no squad, no target selection, no reaction time, no leading a moving
/// target, no decision to *stop*. Those are EMS3's, and the reason to keep them
/// out is that each one is a policy: a policy in this function would be a policy
/// two hosts have to agree about, written where nobody would look for it. What
/// this answers is the mechanical question — *can an armed NPC point a weapon at
/// somebody and pull the trigger* — and the answer is now yes.
///
/// The **spread is free and already deterministic**: `shot_direction` folds the
/// weapon's own `spread_seed` with its shot index, so two NPCs firing the same
/// rifle at the same target do not put their rounds in the same hole, and a
/// replay reproduces every one of them.
///
/// `hold_trigger` is the trigger's **level**, exactly as the player's is, so a
/// semi-automatic weapon in an NPC's hands fires once per press through
/// `try_fire`'s own edge rule and needs no second mechanism. It does **not**
/// write `press_attack`: that edge is the door-kick's, and an NPC that kicked
/// every locked door it happened to face is not what "an armed NPC can fire" is
/// asking for.
///
/// Answers `false` for a shooter that is not there, is player-controlled, or has
/// no target to aim at — refusals as values, all the way down.
pub fn npc_aim_at(world: &mut EcsWorld, shooter: Uuid, target: Uuid, hold_trigger: bool) -> bool {
    let Some(entity) = world.entity_of(shooter) else {
        return false;
    };
    if world
        .world()
        .get::<CharacterMovement>(entity)
        .is_none_or(|cm| cm.player_controlled)
    {
        return false;
    }
    // From the shooter's own muzzle to the point a swing would land on — the
    // same `strike_point` melee aims at, so a rifle and a fist agree about where
    // a person is.
    let Some((from, _, _, _)) = muzzle_of(world, shooter) else {
        return false;
    };
    let Some(at) = strike_point(world, target) else {
        return false;
    };
    let to = at - from;
    let planar = (to.x * to.x + to.z * to.z).sqrt();
    if !to.is_finite() || (planar <= 1.0e-9 && to.y.abs() <= 1.0e-9) {
        return false;
    }
    // **Portable trigonometry** (the P14 law): these two numbers reach
    // `shot_direction`, whose output reaches the ray cast, whose hit reaches the
    // damage door, which reaches the trace.
    let yaw = inf_ecs::movement::planar_yaw_deg(inf_ecs::math::Vec2d::new(to.x, to.z));
    let pitch = if planar <= 1.0e-9 {
        // Straight up or straight down: the yaw is whatever it was and the pitch
        // is the limit `aim_forward` clamps to anyway.
        if to.y > 0.0 {
            89.9
        } else {
            -89.9
        }
    } else {
        // `asin(y/|to|)` written as `90 - acos`, because `pacos64` is the
        // portable inverse this engine has and `f64::asin` is not bit-portable.
        90.0 - inf_math::pacos64((to.y / to.length()).clamp(-1.0, 1.0)).to_degrees()
    };
    let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(entity) else {
        return false;
    };
    cm.runtime.aim_yaw_deg = yaw;
    cm.runtime.aim_pitch_deg = pitch.clamp(-89.9, 89.9);
    // The LEVEL, assigned — `apply_intent`'s own treatment of this field, and the
    // reason a semi-automatic weapon needs nothing else.
    cm.runtime.want_attack = hold_trigger;
    true
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
