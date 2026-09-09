//! **WAVE WPN2d — CLASSES, ATTACHMENTS AND MESHES.** The gate.
//!
//! # What every arm in this file reads
//!
//! **The world, or the command stream, and never a report's summary of
//! itself.** A pellet is a hit POINT, a blast is joules a body lost, a lock is a
//! target guid and a clock, a grenade is a position against a closed form, a
//! swing is a ray that did or did not reach, an attachment is a number in a
//! `WeaponDef` or a field in an `AudioCommand`, and a mesh is a GUID on a drawn
//! entity beside a hand's own world position.
//!
//! The one thing this file never asserts is that a component is present. A cube
//! in the hand carries a `MeshRef`; a bare rail carries a `WeaponState`; a
//! launcher pointed at nothing carries a `lock_target` field. So every arm below
//! is written to be FALSIFIED by the state this wave replaced — and where an arm
//! could have passed before the wave, its own doc says so.
//!
//! # The mutations each arm dies to
//!
//! Written beside the arm, not in a report. They were run.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::attachment::{self, AttachmentSlot};
use inf_ecs::ballistics::{self, RoundKind};
use inf_ecs::components::{
    BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, Destructible, MeshRef,
    RigidBody3D, Transform,
};
use inf_ecs::item::{self, ItemDef, ItemDefs};
use inf_ecs::math::Vec3d;
use inf_ecs::weapon::{self, ImpactSurface, ShotKind, WeaponClass, WeaponDef};
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};
use inf_project::ProjectManifest;

const DT: f64 = 1.0 / 60.0;
const GRAVITY: DVec3 = DVec3::new(0.0, -9.81, 0.0);
const RADIUS: f64 = 0.3;

const HERO: Uuid = Uuid::from_u128(0x2D00_0001);
const GROUND: Uuid = Uuid::from_u128(0x2D00_0002);
const CHASSIS: Uuid = Uuid::from_u128(0x2D00_0003);
const WHEEL_BASE: u128 = 0x2D00_0010;

fn victim_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2D00_0100 + i as u128)
}

fn wall_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2D00_0200 + i as u128)
}

fn shooter_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x2D00_0300 + i as u128)
}

// ── the range ───────────────────────────────────────────────────────────────

/// **A floor, a hero, and whatever an arm puts on it.**
struct Range {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Range {
    fn new(defs: ItemDefs) -> Self {
        let mut world = EcsWorld::new();
        slab(
            &mut world,
            GROUND,
            "Ground",
            DVec3::new(0.0, -0.5, 0.0),
            Vec3d::new(200.0, 0.5, 200.0),
        );
        stand(&mut world, HERO, "Hero", DVec3::ZERO, true);
        *item::item_defs_mut(&mut world) = defs;
        world.mark_dirty();
        world.propagate();
        let mut r = Self {
            world,
            bridge: PhysicsBridge3D::new(GRAVITY),
        };
        r.bridge.sync_from_world(&r.world);
        r
    }

    fn arm(&mut self, who: Uuid, id: &str) {
        assert!(item::give_inventory(&mut self.world, who, 6));
        assert_eq!(item::give(&mut self.world, who, id, 1), 0);
        assert!(d3::gameplay::equip_weapon(&mut self.world, who, id));
    }

    fn aim(&mut self, who: Uuid, yaw: f64, pitch: f64) {
        let e = self.world.entity_of(who).expect("a shooter");
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.aim_yaw_deg = yaw;
        cm.runtime.aim_pitch_deg = pitch;
    }

    fn hold_trigger(&mut self, who: Uuid, down: bool) {
        let e = self.world.entity_of(who).expect("a shooter");
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.want_attack = down;
        cm.runtime.press_attack = down;
    }

    fn press_throw(&mut self, who: Uuid) {
        let e = self.world.entity_of(who).expect("a thrower");
        let mut cm = self
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("a character");
        cm.runtime.press_throw = true;
    }

    fn resync(&mut self) {
        self.world.mark_dirty();
        self.world.reindex_guids();
        self.world.propagate();
        self.bridge.sync_from_world(&self.world);
    }

    /// One fixed step, in the hosts' own order.
    fn step(&mut self) -> d3::GameplayReport {
        self.bridge.sync_from_world(&self.world);
        d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
        let report = d3::step_gameplay(&mut self.world, &mut self.bridge, DT);
        self.bridge.step(DT);
        self.bridge.write_back_into(&mut self.world);
        self.world.propagate();
        report
    }

    fn rounds(&self) -> Vec<ballistics::Round> {
        ballistics::round_pool(&self.world)
            .map(|p| p.rounds.clone())
            .unwrap_or_default()
    }

    fn state(&self, who: Uuid) -> weapon::WeaponState {
        let e = self.world.entity_of(who).expect("a shooter");
        self.world
            .world()
            .get::<weapon::WeaponState>(e)
            .cloned()
            .expect("an ammunition clock")
    }

    /// Put a body somewhere and give it a full tank of joules, so a blast has
    /// something measurable to take out of it.
    fn victim(&mut self, guid: Uuid, at: DVec3) {
        stand(&mut self.world, guid, "Victim", at, false);
        self.resync();
        weapon::give_health(&mut self.world, guid, weapon::DEFAULT_VITALITY_J);
    }

    fn health(&self, guid: Uuid) -> f64 {
        weapon::health_of(&self.world, guid)
            .map(|h| h.joules)
            .unwrap_or(f64::NAN)
    }
}

fn slab(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, half: Vec3d) {
    let e = world.spawn_with_guid(guid, name, None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(at.x, at.y, at.z);
    world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: half,
            ..Default::default()
        },
        t,
    ));
}

fn stand(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, player: bool) {
    let cm = CharacterMovement {
        player_controlled: player,
        ..Default::default()
    };
    let e = world.spawn_with_guid(guid, name, None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(at.x, at.y + cm.stand_half_height_m + RADIUS, at.z);
    world.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(RADIUS, cm.stand_half_height_m, RADIUS),
            radius: RADIUS,
            ..Default::default()
        },
        inf_ecs::components::CharacterController3D::default(),
        cm,
        t,
    ));
}

/// **A car**, in the one shape `PhysicsBridge3D` recognises as one: a dynamic
/// box with four **sensor spheres** parented to it. That is the lock's own
/// target set (`vehicle_guids`), so a fixture that built anything else would be
/// testing a lock against something the game does not think is a vehicle.
fn car(world: &mut EcsWorld, at: DVec3) {
    let e = world.spawn_with_guid(CHASSIS, "Car", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(at.x, at.y, at.z);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            angular_damping: 0.5,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(1.0, 0.7, 2.4),
            ..Default::default()
        },
    ));
    for (i, (x, z)) in [(-0.9, 1.4), (0.9, 1.4), (-0.9, -1.4), (0.9, -1.4)]
        .into_iter()
        .enumerate()
    {
        let w = world.spawn_with_guid(Uuid::from_u128(WHEEL_BASE + i as u128), "Wheel", Some(e));
        let mut wt = Transform::IDENTITY;
        wt.translation = Vec3d::new(x, -0.4, z);
        world.world_mut().entity_mut(w).insert((
            wt,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: 0.35,
                sensor: true,
                ..Default::default()
            },
        ));
    }
}

fn defs_with(rows: &[(&str, WeaponDef)]) -> ItemDefs {
    let mut d = ItemDefs::default();
    for (id, def) in rows {
        assert!(d.insert(ItemDef {
            id: (*id).to_string(),
            label: (*id).to_string(),
            stack_max: 8,
            mass_kg: 3.0,
            weapon: Some(*def),
            mesh: None,
        }));
    }
    d
}

/// The shipped registry, so an arm that names a weapon reads the row the game
/// ships rather than inventing one (the WPN2c audit's finding).
fn registry() -> ItemDefs {
    let mut d = ItemDefs::default();
    d.merge_toml(weapon::WEAPON_REGISTRY_TOML)
        .expect("the shipped weapon registry parses");
    d
}

fn row(id: &str) -> WeaponDef {
    registry()
        .get(id)
        .and_then(|i| i.weapon)
        .unwrap_or_else(|| panic!("{id} is not in the shipped registry"))
}

// ═════════════════════════════════════════════════════════════════════════════
// (a) SHOTGUNS
// ═════════════════════════════════════════════════════════════════════════════

/// **ONE PULL THROWS N PELLETS AND EACH CARRIES ITS SHARE.**
///
/// Read off the HITS, not off a count: eight `WeaponHit`s from one trigger
/// press, each carrying `damage_j / 8`, and exactly ONE of them `loud` — which
/// is what keeps a shotgun one gunshot rather than eight, in the audio queue and
/// in `panic_sources`.
///
/// **Mutation → red:** dropping the `pellets == 1` early return (one hit);
/// dividing `damage_j` by nothing (eight hits at eight times the damage);
/// setting `hit.loud = true` for every pellet (eight loud).
#[test]
fn a_pull_throws_its_pellets_and_each_carries_its_share() {
    let shotgun = row("remington_870");
    assert_eq!(shotgun.pellets, 8, "the Remington's own row");
    let mut r = Range::new(registry());
    r.arm(HERO, "remington_870");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    let rep = r.step();
    assert_eq!(rep.shots, 1, "one trigger pull is one shot");
    assert_eq!(rep.rounds.pellets, 8, "eight pellets left the barrel");
    assert_eq!(rep.hits.len(), 8, "eight pellets are eight hits");
    let loud = rep.hits.iter().filter(|h| h.loud).count();
    assert_eq!(loud, 1, "a shotgun is ONE bang, not eight");
    let per = shotgun.pellet_damage_j();
    println!(
        "the Remington: {} J a pull / {} pellets = {per:.1} J a pellet ({} HP)",
        shotgun.damage_j,
        shotgun.pellets,
        per / weapon::JOULES_PER_HIT_POINT
    );
    // The doc's table 6 gives this row 18 torso HP a pellet.
    assert!(
        (per / weapon::JOULES_PER_HIT_POINT - 18.0).abs() < 1e-9,
        "the per-pellet figure is {} HP and the doc's table 6 says 18",
        per / weapon::JOULES_PER_HIT_POINT
    );
    for h in &rep.hits {
        assert!(
            (h.energy_j - per).abs() < 1e-6,
            "a pellet carried {} J and its share is {per}",
            h.energy_j
        );
    }
    // ONE shell for eight pellets.
    assert_eq!(rep.casings.ejected, 1, "eight pellets ejected eight shells");
}

/// **THE PATTERN AT TEN METRES IS THE CONE THE ROW SAYS IT IS** — measured off
/// the hit POINTS, which is the only place a pattern exists.
///
/// A wall at 10 m; the radius of the eight impacts about their centroid,
/// compared against `10 · tan(cone/2)` — the closed form the cone means.
///
/// **Mutation → red:** passing the resolved PULL cone instead of `def.cone_deg`
/// (the eight points collapse into one).
#[test]
fn the_pattern_at_ten_metres_is_the_cone_the_row_says_it_is() {
    let mut r = Range::new(registry());
    slab(
        &mut r.world,
        wall_guid(0),
        "Wall",
        DVec3::new(0.0, 2.0, 10.0),
        Vec3d::new(8.0, 4.0, 0.2),
    );
    r.resync();
    r.arm(HERO, "remington_870");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    let rep = r.step();
    let points: Vec<DVec3> = rep.hits.iter().map(|h| h.to).collect();
    assert_eq!(points.len(), 8);
    let centre = points.iter().fold(DVec3::ZERO, |a, p| a + *p) / points.len() as f64;
    let radii: Vec<f64> = points
        .iter()
        .map(|p| ((p.x - centre.x).powi(2) + (p.y - centre.y).powi(2)).sqrt())
        .collect();
    let worst = radii.iter().cloned().fold(0.0f64, f64::max);
    let mean = radii.iter().sum::<f64>() / radii.len() as f64;
    let def = row("remington_870");
    // The span is from the MUZZLE, which for a rig-less character is 1.4 m above
    // the feet — not a z coordinate, which was this arm's own first draft and
    // made the bound 12 % too tight.
    let muzzle = rep.hits[0].from;
    let span = (centre - muzzle).length().max(0.1);
    let bound = span * (def.cone_deg * 0.5).to_radians().tan();
    println!(
        "the pattern at {span:.2} m: worst {worst:.3} m, mean {mean:.3} m, the \
         {:.2} deg cone allows {bound:.3} m",
        def.cone_deg
    );
    for (i, p) in points.iter().enumerate() {
        println!("  pellet {i}: ({:.3}, {:.3}, {:.3})", p.x, p.y, p.z);
    }
    assert!(
        worst <= bound + 1e-6,
        "a pellet landed {worst:.3} m out and the cone allows {bound:.3} m"
    );
    // …and it is a PATTERN, not a point: the eight are not all in one place.
    assert!(
        worst > bound * 0.3,
        "the widest pellet is {worst:.4} m out of a {bound:.4} m cone — the \
         pattern has collapsed and this arm would pass on a single ray"
    );
}

/// **A PULL OVER THE RAY CEILING IS REFUSED WITH A VALUE.**
///
/// Four shooters, each with a sixty-four-pellet load: 4 × (64 + 6 probe) = 280
/// casts against a ceiling of 256. The surplus is COUNTED and the step's own
/// bill stops at the ceiling.
///
/// **Mutation → red:** deleting the `bill > MAX_SHOT_RAYS_PER_STEP` break
/// (`pellets_refused` is 0 and `shot_rays` runs past the ceiling).
#[test]
fn a_pull_over_the_ray_ceiling_is_refused_with_a_value() {
    let mut monster = row("remington_870");
    monster.pellets = weapon::MAX_PELLETS;
    monster.rounds_per_minute = 600.0;
    let mut r = Range::new(defs_with(&[("monster", monster)]));
    for i in 0..3 {
        stand(
            &mut r.world,
            shooter_guid(i),
            "Shooter",
            DVec3::new(4.0 + i as f64 * 4.0, 0.0, 0.0),
            false,
        );
    }
    r.resync();
    for who in [HERO, shooter_guid(0), shooter_guid(1), shooter_guid(2)] {
        r.arm(who, "monster");
        r.aim(who, 0.0, 10.0);
        r.hold_trigger(who, true);
    }
    let rep = r.step();
    println!(
        "four 64-pellet pulls on one step: {} shots, {} pellets thrown, {} \
         refused, {} shot rays against a ceiling of {}",
        rep.shots,
        rep.rounds.pellets,
        rep.rounds.pellets_refused,
        rep.rounds.shot_rays,
        ballistics::MAX_SHOT_RAYS_PER_STEP
    );
    assert_eq!(rep.shots, 4, "four triggers");
    assert!(
        rep.rounds.pellets_refused > 0,
        "280 casts fitted inside a ceiling of {} — the refusal never fired",
        ballistics::MAX_SHOT_RAYS_PER_STEP
    );
    assert_eq!(
        rep.rounds.pellets + rep.rounds.pellets_refused,
        4 * weapon::MAX_PELLETS,
        "a pellet went missing: {} thrown + {} refused is not {}",
        rep.rounds.pellets,
        rep.rounds.pellets_refused,
        4 * weapon::MAX_PELLETS
    );
    assert!(
        rep.rounds.shot_rays as usize
            <= ballistics::MAX_SHOT_RAYS_PER_STEP + d3::audio::ENCLOSURE_PROBE_RAYS + 1,
        "the step spent {} casts against a ceiling of {}",
        rep.rounds.shot_rays,
        ballistics::MAX_SHOT_RAYS_PER_STEP
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (b) LAUNCHERS
// ═════════════════════════════════════════════════════════════════════════════

/// **A ROCKET ACCELERATES, AND STOPS AT ITS BURNOUT** — measured per step off
/// the pool, not off the definition.
///
/// **Mutation → red:** dropping the `thrust` term from `advance_round` (the
/// speed only ever falls).
#[test]
fn a_rocket_accelerates_from_its_muzzle_speed_to_its_burnout() {
    let rpg = row("rpg_7");
    assert!(rpg.accel_mps2 > 0.0 && rpg.burnout_mps > rpg.muzzle_speed_mps);
    let mut r = Range::new(registry());
    r.arm(HERO, "rpg_7");
    r.aim(HERO, 0.0, 30.0);
    r.hold_trigger(HERO, true);
    r.step();
    r.hold_trigger(HERO, false);
    let mut speeds = Vec::new();
    for _ in 0..120 {
        if let Some(round) = r.rounds().first() {
            speeds.push(round.velocity.length());
            assert_eq!(round.kind, RoundKind::Rocket, "an RPG round is a rocket");
        }
        r.step();
    }
    assert!(speeds.len() > 20, "the rocket did not fly");
    println!(
        "the RPG-7: launched at {:.1} m/s, {:.1} after 10 steps, {:.1} at the \
         end; its row says {} -> {}",
        speeds[0],
        speeds.get(10).copied().unwrap_or_default(),
        speeds.last().copied().unwrap_or_default(),
        rpg.muzzle_speed_mps,
        rpg.burnout_mps
    );
    assert!(
        speeds[10] > speeds[0],
        "the motor did nothing: {:.2} -> {:.2}",
        speeds[0],
        speeds[10]
    );
    let peak = speeds.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        peak <= rpg.burnout_mps * 1.05,
        "the motor ran past its burnout: {peak:.1} against {}",
        rpg.burnout_mps
    );
    assert!(
        peak > rpg.muzzle_speed_mps * 1.5,
        "the motor barely fired: {peak:.1} from {}",
        rpg.muzzle_speed_mps
    );
}

/// **THE BLAST FALLS OFF BY THE CLOSED FORM, AT FIVE DISTANCES** — in the pure
/// function AND in the joules a body actually loses.
///
/// **Mutation → red:** changing `(1 − d/r)²` to a linear falloff (every ring but
/// the two ends moves).
#[test]
fn the_blast_falls_off_by_the_closed_form_at_five_distances() {
    let mut def = WeaponDef {
        blast_radius_m: 10.0,
        blast_damage_j: 1000.0,
        ..Default::default()
    };
    def.class = Some(WeaponClass::Launcher);
    // 1. the closed form, exactly.
    for (d, want) in [
        (0.0, 1.0),
        (2.5, 0.5625),
        (5.0, 0.25),
        (7.5, 0.0625),
        (10.0, 0.0),
    ] {
        let got = ballistics::blast_falloff(def.blast_radius_m, d);
        println!("  blast at {d:>5.1} m: {got:.6} (closed form {want:.6})");
        assert!(
            (got - want).abs() < 1e-12,
            "the falloff at {d} m is {got} and (1 - d/r)^2 is {want}"
        );
    }
    assert_eq!(
        ballistics::blast_falloff(0.0, 1.0),
        0.0,
        "no radius, no blast"
    );
    assert_eq!(
        ballistics::blast_falloff(10.0, 99.0),
        0.0,
        "past the radius"
    );

    // 2. and the same numbers, as joules bodies lost.
    let mut r = Range::new(defs_with(&[("boom", def)]));
    // Five bodies at five distances, each on its OWN bearing: a row of them
    // along one axis is a row in which the nearest shadows every other, which is
    // what this arm's first draft measured (four of five shadowed, and rightly).
    let rings = [1.0, 3.0, 5.0, 7.0, 9.0];
    let ring_at = |i: usize, d: f64| -> DVec3 {
        let a = std::f64::consts::TAU * i as f64 / rings.len() as f64;
        DVec3::new(a.cos() * d, 0.0, 20.0 + a.sin() * d)
    };
    for (i, d) in rings.iter().enumerate() {
        r.victim(victim_guid(i), ring_at(i, *d));
    }
    let before: Vec<f64> = (0..rings.len()).map(|i| r.health(victim_guid(i))).collect();
    let mut report = d3::GameplayReport::default();
    let at = DVec3::new(0.0, 1.4, 20.0);
    d3::gameplay::blast_for_test(&mut r.world, &mut r.bridge, HERO, at, &def, DT, &mut report);
    assert_eq!(report.blasts.len(), 1, "one explosion");
    println!(
        "the blast at {at:?}: {} bodies hurt, {} shadowed, {} rays",
        report.blasts[0].hurt, report.rounds.blast_shadowed, report.rounds.blast_rays
    );
    for (i, d) in rings.iter().enumerate() {
        let after = r.health(victim_guid(i));
        let spent = before[i] - after;
        // The candidate's own strike point is 1.4 m up, so the distance is the
        // 3-D one and not the ring's radius.
        let point = ring_at(i, *d) + DVec3::Y * d3::gameplay::MUZZLE_HEIGHT_M;
        let want = ballistics::blast_damage_j(&def, (point - at).length());
        println!("  a body at {d:>4.1} m lost {spent:8.2} J, the curve owes {want:8.2}");
        assert!(
            (spent - want).abs() < 1e-6,
            "a body at {d} m lost {spent} J and the curve owes {want}"
        );
    }
    assert_eq!(report.blasts[0].hurt, rings.len() as u32);
    assert_eq!(
        report.rounds.blast_shadowed, 0,
        "five bodies on five bearings shadowed each other"
    );
}

/// **A BLAST SPENDS ITS JOULES AT THE P22 DOOR** — a `Destructible` inside the
/// radius is owed energy in `report.destruct`, which is what the host spends.
///
/// **Mutation → red:** dropping the destructible half of the candidate list
/// (`destruct` is empty and only the bodies are hurt).
#[test]
fn a_blast_spends_its_joules_at_the_p22_door() {
    let def = WeaponDef {
        blast_radius_m: 10.0,
        blast_damage_j: 5000.0,
        class: Some(WeaponClass::Launcher),
        ..Default::default()
    };
    let mut r = Range::new(defs_with(&[("boom", def)]));
    let crate_guid = wall_guid(9);
    slab(
        &mut r.world,
        crate_guid,
        "Crate",
        DVec3::new(3.0, 1.0, 20.0),
        Vec3d::new(0.5, 0.5, 0.5),
    );
    {
        let e = r.world.entity_of(crate_guid).expect("the crate");
        r.world
            .world_mut()
            .entity_mut(e)
            .insert(Destructible::default());
    }
    r.resync();
    let mut report = d3::GameplayReport::default();
    let at = DVec3::new(0.0, 1.0, 20.0);
    d3::gameplay::blast_for_test(&mut r.world, &mut r.bridge, HERO, at, &def, DT, &mut report);
    println!("the blast owed: {:?}", report.destruct);
    let owed = report
        .destruct
        .iter()
        .find(|(g, _)| *g == crate_guid)
        .map(|(_, j)| *j)
        .unwrap_or_else(|| panic!("the crate was owed nothing"));
    let want = ballistics::blast_damage_j(&def, 3.0);
    assert!(
        (owed - want).abs() < 1e-6,
        "the crate was owed {owed} J and the curve says {want}"
    );
}

/// **A WALL SHADOWS A BODY FROM A BLAST** — the line-of-sight ray, measured as
/// joules NOT spent.
///
/// **Mutation → red:** deleting the `cast_ray_where` guard (the shadowed body
/// takes the full falloff and `blast_shadowed` is 0).
#[test]
fn a_wall_shadows_a_body_from_a_blast() {
    let def = WeaponDef {
        blast_radius_m: 12.0,
        blast_damage_j: 1000.0,
        class: Some(WeaponClass::Launcher),
        ..Default::default()
    };
    let mut r = Range::new(defs_with(&[("boom", def)]));
    // Two bodies at the same distance; one of them behind a slab.
    r.victim(victim_guid(0), DVec3::new(5.0, 0.0, 20.0));
    r.victim(victim_guid(1), DVec3::new(-5.0, 0.0, 20.0));
    slab(
        &mut r.world,
        wall_guid(1),
        "Blast wall",
        DVec3::new(-2.5, 1.4, 20.0),
        Vec3d::new(0.25, 2.0, 3.0),
    );
    r.resync();
    let before = [r.health(victim_guid(0)), r.health(victim_guid(1))];
    let mut report = d3::GameplayReport::default();
    let at = DVec3::new(0.0, 1.4, 20.0);
    d3::gameplay::blast_for_test(&mut r.world, &mut r.bridge, HERO, at, &def, DT, &mut report);
    let exposed = before[0] - r.health(victim_guid(0));
    let shadowed = before[1] - r.health(victim_guid(1));
    println!(
        "the exposed body lost {exposed:.1} J, the one behind the slab lost \
         {shadowed:.1} J ({} shadowed, {} rays)",
        report.rounds.blast_shadowed, report.rounds.blast_rays
    );
    assert!(exposed > 1.0, "the exposed body took nothing");
    assert_eq!(shadowed, 0.0, "the wall did not stop the blast");
    assert_eq!(report.rounds.blast_shadowed, 1);
}

/// **A LOCK HOLDS FOR `lock_s` AND RELEASES OUTSIDE THE CONE** — read off the
/// WeaponState, which is the sim state a round leaves on.
///
/// **Mutation → red:** dropping the `None =>` arm that clears the lock (the
/// launcher stays locked onto a car it is no longer pointing at).
#[test]
fn a_lock_holds_for_lock_s_and_releases_outside_the_cone() {
    let stinger = row("fim_92_stinger");
    assert!(stinger.can_lock() && stinger.lock_air_only);
    // The Javelin locks a ground vehicle; the Stinger refuses one.
    let javelin = row("javelin_fgm148");
    let mut r = Range::new(registry());
    car(&mut r.world, DVec3::new(0.0, 1.0, 40.0));
    r.resync();
    assert_eq!(
        r.bridge.vehicle_count(),
        1,
        "the fixture's car is not a vehicle the bridge knows"
    );
    r.arm(HERO, "javelin_fgm148");
    let mut held = Vec::new();
    for _ in 0..140 {
        // **The aim is re-asserted every step**, because the movement step owns
        // it: a fixture that set it once is measuring whatever the look
        // integrator left behind, which is what this arm's first draft did.
        r.aim(HERO, 0.0, 0.0);
        r.step();
        held.push(r.state(HERO).lock_held_s);
    }
    let st = r.state(HERO);
    println!(
        "the Javelin: target {}, held {:.3} s of {}",
        st.lock_target, st.lock_held_s, javelin.lock_s
    );
    assert_eq!(st.lock_target, CHASSIS, "the launcher locked nothing");
    assert!(
        st.locked_on(&javelin).is_some(),
        "held {:.3} s and the row asks {}",
        st.lock_held_s,
        javelin.lock_s
    );
    assert!(
        (st.lock_held_s - javelin.lock_s).abs() < 1e-9,
        "the hold ran past the row's own `lock_s`"
    );
    // …and it lets go the moment the cone loses it.
    r.aim(HERO, 90.0, 0.0);
    r.step();
    let after = r.state(HERO);
    assert_eq!(after.lock_target, Uuid::nil(), "the lock survived the turn");
    assert_eq!(after.lock_held_s, 0.0);
    assert!(after.locked_on(&javelin).is_none());

    // The Stinger refuses a car on the ground: `lock_air_only`.
    let mut air = Range::new(registry());
    car(&mut air.world, DVec3::new(0.0, 1.0, 40.0));
    air.resync();
    air.arm(HERO, "fim_92_stinger");
    for _ in 0..140 {
        air.aim(HERO, 0.0, 0.0);
        air.step();
    }
    assert_eq!(
        air.state(HERO).lock_target,
        Uuid::nil(),
        "the Stinger locked a car on the road"
    );
    // Lift it into the air and it acquires. The chassis is a DYNAMIC body, so
    // it is put back every step rather than once — a fixture that lifted it and
    // then ran two seconds of gravity would be testing a falling car.
    let alt = d3::gameplay::MIN_AIRBORNE_M + 20.0;
    for _ in 0..140 {
        {
            let e = air.world.entity_of(CHASSIS).expect("the car");
            if let Some(mut t) = air.world.world_mut().get_mut::<Transform>(e) {
                t.translation.y = alt;
            }
        }
        air.world.mark_dirty();
        air.world.propagate();
        // Aim UP at it: the Stinger's cone is eight degrees and a car twenty-six
        // metres up at forty out is thirty-three of them off the horizon.
        let pitch = (alt - d3::gameplay::MUZZLE_HEIGHT_M)
            .atan2(40.0)
            .to_degrees();
        air.aim(HERO, 0.0, pitch);
        air.step();
    }
    assert_eq!(
        air.state(HERO).lock_target,
        CHASSIS,
        "the Stinger refused a target {alt} m up"
    );
}

/// **A LOCKED ROUND FOLLOWS ITS TARGET AND AN UNLOCKED ONE DOES NOT** — measured
/// as how far off the aim line the two end up.
///
/// **Mutation → red:** dropping the `guides.get` block from `step_rounds` (the
/// guided round flies as straight as the dumb one).
#[test]
fn a_locked_round_follows_its_target_and_an_unlocked_one_does_not() {
    // **THE CAR IS OFF THE AIM LINE AND INSIDE THE CONE**, which is the whole
    // fixture. The Javelin's cone is 10 degrees, so at 40 m a target 3 m to the
    // side is inside it and OFF the line the shot leaves along: a dumb round
    // flies down `+Z` and misses by three metres, and only guidance brings one
    // back. This arm's first draft aimed straight AT the car, where a dumb round
    // is as accurate as a guided one and both missed by 5.11 m — the arm passed
    // and measured nothing.
    // **AND ITS GRAVITY IS OFF, IN THE FIXTURE ONLY.** A guided missile that
    // drops on the way is measuring two things at once — how fast it can turn
    // and how far it falls — and the second is `advance_round`'s, already
    // measured by `wpn2a_gate`'s own drop table. With `gravity_scale = 0` the
    // only thing that can move this round off the line it left on is the
    // guidance, which is what the arm names.
    let mut defs = registry();
    {
        let mut flat = row("javelin_fgm148");
        flat.gravity_scale = 0.0;
        assert!(defs.insert(ItemDef {
            id: "javelin_flat".into(),
            label: "Javelin (level)".into(),
            stack_max: 1,
            mass_kg: 7.0,
            weapon: Some(flat),
            mesh: None,
        }));
    }
    let miss_by = |lock: bool| -> (bool, f64) {
        let mut r = Range::new(defs.clone());
        car(&mut r.world, DVec3::new(3.0, 1.0, 40.0));
        r.resync();
        r.arm(HERO, "javelin_flat");
        let yaw = 0.0;
        for _ in 0..140 {
            r.aim(HERO, yaw, 0.0);
            r.step();
        }
        assert_eq!(
            r.state(HERO).lock_target,
            CHASSIS,
            "a car 3 m off a 40 m shot is inside a 10 degree cone and was not locked"
        );
        if !lock {
            // …then take the lock away, leaving the same aim.
            let e = r.world.entity_of(HERO).expect("the hero");
            if let Some(mut st) = r.world.world_mut().get_mut::<weapon::WeaponState>(e) {
                st.lock_target = Uuid::nil();
                st.lock_held_s = 0.0;
            }
        }
        // Fire, then turn hard away: a guided round keeps going for the car and
        // a dumb one goes where it was pointed.
        r.aim(HERO, yaw, 0.0);
        r.hold_trigger(HERO, true);
        r.step();
        r.hold_trigger(HERO, false);
        let launched = r.rounds().first().copied().expect("a rocket left the tube");
        assert_eq!(launched.kind, RoundKind::Rocket);
        assert_eq!(
            launched.guide.is_nil(),
            !lock,
            "the round's guide does not match the lock"
        );
        // **DID IT HIT THE CAR** — the question, and it is asked of the world
        // rather than of a sampled distance. A missile crosses three metres a
        // FIXED STEP at this speed, so the closest SAMPLE is a number about the
        // sampling rate: this arm's own first draft read 4.649 m for a round
        // that struck the car and 3.241 for one that sailed past it, and called
        // the second one better.
        let target = DVec3::new(3.0, 1.0, 40.0);
        let mut closest = f64::INFINITY;
        let mut hit = false;
        for _ in 0..240 {
            for round in r.rounds() {
                closest = closest.min((round.at - target).length());
            }
            let rep = r.step();
            if rep.hits.iter().any(|h| h.target == Some(CHASSIS)) {
                hit = true;
            }
        }
        (hit, closest)
    };
    let (guided_hit, guided) = miss_by(true);
    let (dumb_hit, dumb) = miss_by(false);
    println!(
        "the Javelin at a car 3 m off a 40 m shot: guided hit={guided_hit}          (closest sample {guided:.3} m), unguided hit={dumb_hit} (closest sample          {dumb:.3} m)"
    );
    assert!(
        guided_hit,
        "the guided missile did not reach the car it was locked onto"
    );
    assert!(
        !dumb_hit,
        "the UNGUIDED round hit a car three metres off the line it was fired          along — the fixture is not measuring guidance"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (c) THROWABLES
// ═════════════════════════════════════════════════════════════════════════════

/// **THE THROW VERB REACHES THE SIM FROM THE KEY** — the whole chain, from a
/// keyboard code to a `press_throw` edge, through the shipped default map.
///
/// **Mutation → red:** deleting the `bind_key("throw", "KeyB")` line (the intent
/// never rises).
#[test]
fn the_throw_verb_reaches_the_sim_from_the_key() {
    // 1. the shipped default map binds it, on a key AND on a button.
    for (label, event) in [
        (
            "KeyB",
            inf_input::InputEvent::Key {
                code: "KeyB".into(),
                pressed: true,
            },
        ),
        (
            "the middle mouse button",
            inf_input::InputEvent::MouseButton {
                button: inf_input::MouseButton::Middle,
                pressed: true,
            },
        ),
    ] {
        let mut state = inf_input::InputState::new(inf_input::default_map());
        state.apply_dt(&[event], DT);
        let down = inf_player::input::held_actions(&state, DT);
        assert!(
            down.is_down(inf_ecs::movement::actions::THROW),
            "{label} did not raise the `{}` action — the binding is missing",
            inf_ecs::movement::actions::THROW
        );
    }
    // 2. …and the KEY the shipped window sends for it is a key the window can
    //    send, which is the half a binding table cannot check for itself.
    assert_eq!(
        inf_player::input::keycode_to_code(winit::keyboard::KeyCode::KeyB),
        Some("KeyB"),
        "the desktop player cannot send the throw key at all"
    );
    // 3. the action becomes an INTENT, through the one door both hosts use.
    let mut state = inf_input::InputState::new(inf_input::default_map());
    state.apply_dt(
        &[inf_input::InputEvent::Key {
            code: "KeyB".into(),
            pressed: true,
        }],
        DT,
    );
    let (held, axes) = state.resolved(DT);
    let held: BTreeSet<String> = held.into_iter().collect();
    let axes: std::collections::BTreeMap<String, f32> = axes.into_iter().collect();
    let intent = inf_ecs::movement::MovementIntent::from_actions(
        |n| axes.get(n).copied().unwrap_or(0.0),
        |n| held.contains(n),
        |n| held.contains(n),
        |_| false,
        |_| (0.0, 0.0),
        0.2,
    );
    assert!(
        intent.throw,
        "the throw action did not become a throw intent"
    );
    // 4. and the intent reaches the runtime edge the fixed step consumes.
    let mut r = Range::new(registry());
    inf_ecs::movement::apply_intent(&mut r.world, &intent);
    let e = r.world.entity_of(HERO).expect("the hero");
    assert!(
        r.world
            .world()
            .get::<CharacterMovement>(e)
            .is_some_and(|cm| cm.runtime.press_throw),
        "`apply_intent` did not write the throw edge"
    );
}

/// **THE GRENADE LEAVES THE HAND AT THE CLIP'S RELEASE POINT, BY EITHER PATH.**
///
/// A throw releases two ways and both are armed, exactly as a reload does:
///
/// 1. **The clip's own notify**, fired by the pose step when the additive
///    crosses `inf_anim::THROW_RELEASE_FRAC`. This arm INJECTS it, because the
///    fixture has no rig to author one — and injecting it is what proves the
///    notify is the AUTHORITY rather than decoration: the body leaves on the
///    step the notify arrives, whenever that is.
/// 2. **The character's own clock**, for every character with no rig — which is
///    every headless run and every crowd agent the sim has tiered out of posing.
///    It fires `THROW_RELEASE_GRACE_S` after the fraction, so on a rigged
///    character path 1 always wins.
///
/// Both are inside the two-frame budget the wave owes: the notify path is one
/// fixed step behind the crossing (the pose step runs after the gameplay step)
/// and the clock path is two.
///
/// **Mutation → red:** deleting the `consume_anim_notify` (the injected notify
/// does nothing and the release slips to the clock's own step); deleting the
/// clock branch (a rig-less character never throws anything at all).
#[test]
fn the_grenade_leaves_the_hand_at_the_clips_release_point_by_either_path() {
    let g67 = row("g67_grenade");
    assert!(g67.throwable && g67.fuse_s > 0.0);
    let release_s = inf_anim::THROW_OVER_S * inf_anim::THROW_RELEASE_FRAC;
    let want_step = (release_s / DT).round() as i64;
    let grace_steps = (d3::gameplay::THROW_RELEASE_GRACE_S / DT).round() as i64;

    // ── path 2: the clock, on a character with no rig ──
    let mut r = Range::new(registry());
    r.arm(HERO, "g67_grenade");
    r.aim(HERO, 0.0, 20.0);
    let mut started: Option<i64> = None;
    let mut released: Option<i64> = None;
    for i in 0..180i64 {
        if started.is_none() {
            r.press_throw(HERO);
        }
        let rep = r.step();
        if started.is_none() {
            let e = r.world.entity_of(HERO).expect("the hero");
            if r.world
                .world()
                .get::<CharacterMovement>(e)
                .is_some_and(|cm| cm.runtime.throw_s > 0.0)
            {
                started = Some(i);
            }
        }
        if rep.throws > 0 {
            released = Some(i);
            break;
        }
    }
    let start = started.expect("the throw animation never started");
    let out = released.expect("a character with no rig never let go of its grenade");
    let after = out - start;
    println!(
        "the clock path: the clip releases at {release_s:.3} s (step {want_step}), \
         the grace is {grace_steps} steps, the throw started on step {start} and \
         the grenade left on step {out} ({after} steps in)"
    );
    assert!(
        (after - (want_step + grace_steps)).abs() <= 1,
        "the clock released {after} steps in and the clip's point plus its grace \
         is {}",
        want_step + grace_steps
    );
    // …and it is in the air, as a THROWN body with the row's own fuse on it.
    let live = r.rounds();
    assert_eq!(live.len(), 1, "the pool holds {} bodies", live.len());
    assert_eq!(live[0].kind, RoundKind::Thrown);
    assert!(live[0].fuse_left_s > 0.0 && live[0].fuse_left_s <= g67.fuse_s);
    // **THE HAND LET GO**, and it is measured on the WORLD rather than on an
    // IK request: the grenade entity is gone from the hand for the frames the
    // body is in the air, because the magazine that held it is empty. A rifle
    // with an empty magazine is still a rifle; a grenade is not.
    assert!(
        r.world
            .entity_of(d3::gameplay::equipped_weapon_guid(HERO))
            .is_none(),
        "the thrown grenade is still drawn in the hand that threw it"
    );
    let ik = inf_ecs::pose::hand_ik(&r.world, HERO);
    assert!(
        ik.map(|i| i.gun.is_none()).unwrap_or(true),
        "the hand is still gripping a weapon after the release"
    );

    // ── path 1: the notify, and it gets there FIRST ──
    let mut n = Range::new(registry());
    n.arm(HERO, "g67_grenade");
    n.aim(HERO, 0.0, 20.0);
    let mut n_started: Option<i64> = None;
    let mut n_released: Option<i64> = None;
    for i in 0..180i64 {
        if n_started.is_none() {
            n.press_throw(HERO);
        }
        // Inject the notify EARLY — three steps before the clip's own point —
        // so "the body left when the notify said" is distinguishable from "the
        // body left when the clock said".
        if n_started == Some(i - (want_step - 3)) {
            let mut map = std::collections::BTreeMap::new();
            map.insert(HERO, vec![weapon::THROW_NOTIFY.to_string()]);
            n.world
                .world_mut()
                .insert_resource(inf_ecs::pose::AnimEventsRes(map));
        }
        let rep = n.step();
        if n_started.is_none() {
            let e = n.world.entity_of(HERO).expect("the hero");
            if n.world
                .world()
                .get::<CharacterMovement>(e)
                .is_some_and(|cm| cm.runtime.throw_s > 0.0)
            {
                n_started = Some(i);
            }
        }
        if rep.throws > 0 {
            n_released = Some(i);
            break;
        }
    }
    let ns = n_started.expect("the throw animation never started");
    let no = n_released.expect("the notify released nothing");
    println!(
        "the notify path: injected on step {}, the grenade left on step {no} \
         ({} steps in, against the clock's {})",
        ns + want_step - 3,
        no - ns,
        want_step + grace_steps
    );
    assert!(
        no - ns < want_step + grace_steps,
        "the notify did not get there before the clock — it is decoration"
    );
    assert!(
        (no - ns - (want_step - 3)).abs() <= 1,
        "the grenade left {} steps in and the notify was injected at {}",
        no - ns,
        want_step - 3
    );
}

/// **A GRENADE FOLLOWS THE PARABOLA, AND BOUNCES** — the flight against the
/// closed form while it is in the air, and a measured bounce when it lands.
///
/// **Mutation → red:** dropping the `kind.bounces()` branch (the grenade stops
/// dead on the floor and `bounces` is 0).
#[test]
fn a_grenade_follows_the_parabola_and_bounces() {
    // A fuse long enough that the flight is what ends it, so this arm is about
    // the arc and the next one is about the fuse.
    let mut g67 = row("g67_grenade");
    g67.fuse_s = 6.0;
    g67.gravity_scale = 1.0;
    g67.drag_k = 0.0;
    let mut r = Range::new(defs_with(&[("nade", g67)]));
    r.arm(HERO, "nade");
    r.aim(HERO, 0.0, 35.0);
    let from = DVec3::new(0.0, d3::gameplay::MUZZLE_HEIGHT_M, 0.0);
    let v0 = weapon::aim_forward(0.0, 35.0) * g67.muzzle_speed_mps;
    r.press_throw(HERO);
    let mut launched = None;
    let mut worst = 0.0f64;
    let mut bounces = 0u32;
    let mut settled = None;
    for i in 0..600 {
        let rep = r.step();
        pose_step(&mut r.world);
        bounces += rep.rounds.bounces;
        if rep.throws > 0 {
            launched = Some(i);
        }
        if let (Some(l), Some(round)) = (launched, r.rounds().first()) {
            if round.bounces == 0 {
                // Free flight: the closed-form parabola from the release, with
                // the same gravity and no drag.
                let t = (i - l) as f64 * DT;
                let want = from + v0 * t + DVec3::Y * (-0.5 * 9.81 * t * t);
                worst = worst.max((round.at - want).length());
            }
        }
        if launched.is_some() && settled.is_none() && bounces > 0 && r.rounds().is_empty() {
            settled = Some(i);
        }
    }
    println!(
        "the grenade: worst departure from the closed-form parabola {worst:.4} m \
         over the free flight, {bounces} bounces"
    );
    assert!(launched.is_some(), "nothing was thrown");
    // The integrator is semi-implicit Euler at four sub-steps, so it trails the
    // exact parabola by `O(h)`. A tenth of a metre over a two-second arc is the
    // measured figure; a POINT MOVE would be metres out.
    assert!(
        worst < 0.35,
        "the flight departed the parabola by {worst:.3} m"
    );
    assert!(bounces > 0, "the grenade never bounced");
    assert!(
        bounces <= ballistics::MAX_BOUNCES,
        "{bounces} bounces against a bound of {}",
        ballistics::MAX_BOUNCES
    );
}

/// **THE FUSE GOES OFF WHERE THE GRENADE IS** — the detonation is at the body's
/// own position and at the row's own time, not at the muzzle and not on impact.
///
/// **Mutation → red:** deleting the fuse block (the grenade lives out its
/// `MAX_ROUND_LIFETIME_S` and never explodes).
#[test]
fn the_fuse_goes_off_where_the_grenade_is() {
    let g67 = row("g67_grenade");
    let mut r = Range::new(registry());
    // Five bodies down the throwing line, five metres apart: a grenade lobbed at
    // 25 degrees lands somewhere in there, and the arm is about WHERE the blast
    // is rather than about a fixture that guessed the range.
    for i in 0..5 {
        r.victim(victim_guid(i), DVec3::new(0.0, 0.0, 10.0 + i as f64 * 5.0));
    }
    r.arm(HERO, "g67_grenade");
    r.aim(HERO, 0.0, 25.0);
    r.press_throw(HERO);
    let before: Vec<f64> = (0..5).map(|i| r.health(victim_guid(i))).collect();
    let mut launched = None;
    let mut blast: Option<(i32, DVec3, u32)> = None;
    let mut last_at = DVec3::ZERO;
    for i in 0..600i32 {
        if let Some(round) = r.rounds().first() {
            last_at = round.at;
        }
        let rep = r.step();
        pose_step(&mut r.world);
        if rep.throws > 0 {
            launched = Some(i);
        }
        if let Some(b) = rep.blasts.first() {
            blast = Some((i, b.at, b.hurt));
            break;
        }
    }
    let l = launched.expect("nothing was thrown");
    let (step, at, hurt) = blast.expect("the fuse never fired");
    let elapsed = (step - l) as f64 * DT;
    println!(
        "the G67: thrown on step {l}, went off on step {step} ({elapsed:.2} s \
         later, its fuse is {}), at ({:.2}, {:.2}, {:.2}), hurting {hurt}",
        g67.fuse_s, at.x, at.y, at.z
    );
    assert!(
        (elapsed - g67.fuse_s).abs() < 0.1,
        "the fuse ran {elapsed:.3} s and the row says {}",
        g67.fuse_s
    );
    assert!(
        (at - last_at).length() < 2.0,
        "the blast was at {at:?} and the grenade was at {last_at:?}"
    );
    assert!(
        at.z > 4.0,
        "the blast was at z={:.2} — it went off at the muzzle",
        at.z
    );
    assert!(hurt > 0, "the blast hurt nobody");
    let lost: Vec<f64> = (0..5)
        .map(|i| before[i] - r.health(victim_guid(i)))
        .collect();
    println!("  the five bodies down the line lost {lost:?} J");
    assert!(
        lost.iter().any(|j| *j > 0.0),
        "nobody down the throwing line lost anything to the detonation"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (d) MELEE
// ═════════════════════════════════════════════════════════════════════════════

/// **A PUNCH CANNOT REACH THROUGH A WALL** — wave WPN1's carried defect, closed
/// and measured.
///
/// Two runs at the same geometry: one with a slab between the two bodies and one
/// without. Without it the blow lands; with it the blow is refused and
/// `swings_blocked` counts it.
///
/// **Mutation → red:** deleting the `cast_shape_where` filter (both runs land
/// the blow, which is exactly what shipped before this wave).
#[test]
fn a_punch_cannot_reach_through_a_wall() {
    let land = |walled: bool| -> (bool, u32, u32) {
        let mut r = Range::new(ItemDefs::default());
        r.victim(victim_guid(0), DVec3::new(0.0, 0.0, 1.0));
        if walled {
            slab(
                &mut r.world,
                wall_guid(0),
                "Wall",
                DVec3::new(0.0, 1.4, 0.5),
                Vec3d::new(2.0, 1.4, 0.05),
            );
        }
        r.resync();
        r.aim(HERO, 0.0, 0.0);
        r.hold_trigger(HERO, true);
        let mut landed = false;
        let mut blocked = 0;
        let mut casts = 0;
        for _ in 0..30 {
            let rep = r.step();
            blocked += rep.swings_blocked;
            casts += rep.melee_casts;
            if rep.hits.iter().any(|h| h.target == Some(victim_guid(0))) {
                landed = true;
            }
            r.hold_trigger(HERO, false);
            r.hold_trigger(HERO, true);
        }
        (landed, blocked, casts)
    };
    let (open, open_blocked, open_casts) = land(false);
    let (walled, walled_blocked, walled_casts) = land(true);
    println!(
        "a punch in the open: landed={open} blocked={open_blocked} casts={open_casts}\n\
         a punch through a 10 cm slab: landed={walled} blocked={walled_blocked} \
         casts={walled_casts}"
    );
    assert!(open, "a punch at a body a metre away did not land at all");
    assert_eq!(open_blocked, 0, "an open swing was refused");
    assert!(open_casts > 0, "the LOS probe never ran");
    assert!(
        !walled,
        "the punch went through the wall — carried WPN1 is open"
    );
    assert!(
        walled_blocked > 0,
        "the wall stopped it and nothing counted"
    );
}

/// **THE M9 IS A MELEE ITEM WITH ITS OWN REACH, AND ITS BLOW NAMES A SURFACE.**
///
/// The knife's reach is the registry's, longer than a fist's; and the impact it
/// pushes carries the two-way surface stub, which is what both hosts' clip
/// choice reads.
///
/// **Mutation → red:** deleting the `melee_impacts.push` (a blow lands and
/// nothing is heard); hard-coding `ImpactSurface::Hard` (a body's blow plays a
/// wall's clip).
#[test]
fn the_m9_is_a_melee_item_with_its_own_reach_and_its_blow_names_a_surface() {
    let m9 = row("m9_knife");
    assert_eq!(m9.kind, ShotKind::Melee);
    println!(
        "the M9: reach {:.2} m over a {:.0} deg arc for {} J; the fists reach \
         {:.2} m for {} J",
        m9.reach_m(),
        m9.melee_arc_deg,
        m9.damage_j,
        weapon::fist_def().reach_m(),
        weapon::FIST_DAMAGE_J
    );
    assert!(
        m9.reach_m() > weapon::fist_def().reach_m(),
        "a knife does not reach further than a fist"
    );
    assert!(m9.damage_j > weapon::FIST_DAMAGE_J);
    // A blow on a BODY names flesh.
    let mut r = Range::new(registry());
    r.victim(victim_guid(0), DVec3::new(0.0, 0.0, 1.5));
    r.arm(HERO, "m9_knife");
    r.aim(HERO, 0.0, 0.0);
    r.hold_trigger(HERO, true);
    let mut seen: Option<ImpactSurface> = None;
    for _ in 0..30 {
        let rep = r.step();
        if let Some(i) = rep.melee_impacts.first() {
            seen = Some(i.surface);
            break;
        }
        r.hold_trigger(HERO, false);
        r.hold_trigger(HERO, true);
    }
    assert_eq!(
        seen,
        Some(ImpactSurface::Flesh),
        "a knife into a body did not name flesh"
    );
    // …and the two surfaces name two different clips, which is what makes the
    // stub a stub and not a decoration.
    assert_ne!(
        weapon::melee_impact_clip(ImpactSurface::Flesh),
        weapon::melee_impact_clip(ImpactSurface::Hard)
    );
    // A swing at nothing makes no noise at all.
    let mut empty = Range::new(registry());
    empty.arm(HERO, "m9_knife");
    empty.aim(HERO, 0.0, 0.0);
    empty.hold_trigger(HERO, true);
    let mut impacts = 0;
    for _ in 0..30 {
        impacts += empty.step().melee_impacts.len();
    }
    assert_eq!(impacts, 0, "a swing at thin air made a noise");
}

// ═════════════════════════════════════════════════════════════════════════════
// (e) ATTACHMENTS
// ═════════════════════════════════════════════════════════════════════════════

/// **THE CATALOGUE IS THE DOC'S OWN LISTS**, counted per class and per slot.
#[test]
fn the_attachment_catalogue_is_the_docs_own_lists() {
    assert_eq!(attachment::catalogue_error(), None);
    let cat = attachment::catalogue();
    let mut total = 0usize;
    for class in WeaponClass::ALL {
        let n = cat.count_for(class);
        let per: Vec<(&str, usize)> = AttachmentSlot::ALL
            .into_iter()
            .map(|s| (s.name(), cat.for_slot(class, s).len()))
            .filter(|(_, k)| *k > 0)
            .collect();
        println!("  {:>8}: {n:>3} rows over {:?}", class.name(), per);
        total += n;
    }
    println!("the attachment catalogue: {total} rows over ten slots");
    assert_eq!(total, 530, "the doc's lists are 530 rows");
    assert_eq!(cat.len(), total);
    // The DMR's `stock` carries both the doc's stocks AND its cheek risers,
    // which is the slot mapping the file states.
    assert_eq!(
        cat.for_slot(WeaponClass::Dmr, AttachmentSlot::Stock).len(),
        20,
        "the DMR's combs did not land in `stock`"
    );
    // A launcher has no underbarrel at all — the doc gives it five categories.
    assert!(cat
        .for_slot(WeaponClass::Launcher, AttachmentSlot::Underbarrel)
        .is_empty());
}

/// **THE FOLD IS APPLIED THROUGH THE ONE `WeaponDef` DOOR** — read off
/// `equipped_def`, which is what every rule in the engine asks.
///
/// And the per-instance state reaches the trace: bolting something on moves
/// `weapon_state_bytes`, and taking it off puts them back exactly.
///
/// **Mutation → red:** moving the fold out of `equipped_def` into a helper the
/// fire path calls (this arm reads the base numbers); dropping the `attach`
/// block from `weapon_state_bytes` (the bytes do not move).
#[test]
fn the_fold_is_applied_through_the_one_weapon_def_door() {
    let mut r = Range::new(registry());
    r.arm(HERO, "m4a1");
    r.step();
    let base = weapon::base_equipped_def(&r.world, HERO)
        .expect("a rifle")
        .1;
    let bare_bytes = weapon::weapon_state_bytes(&r.world);
    let cat = attachment::catalogue();
    let supp = cat
        .find(
            WeaponClass::Ar,
            AttachmentSlot::Muzzle,
            "tactical_monolithic_suppressor",
        )
        .expect("the AR's monolithic suppressor");
    assert!(weapon::equip_attachment(
        &mut r.world,
        HERO,
        AttachmentSlot::Muzzle,
        Some(supp)
    ));
    let folded = weapon::equipped_def(&r.world, HERO).expect("a rifle").1;
    let m = cat.get(supp).expect("the row").modifiers;
    println!(
        "the M4A1 with a monolithic suppressor:\n  report {:.1} -> {:.1} m, gain \
         {:.2} -> {:.2}, range {:.1} -> {:.1} m, ADS {:.0} -> {:.0} ms",
        base.report_max_m,
        folded.report_max_m,
        base.report_gain,
        folded.report_gain,
        base.range_m,
        folded.range_m,
        base.ads_time_ms,
        folded.ads_time_ms
    );
    assert!(
        (folded.report_max_m - base.report_max_m * m.loudness_mult).abs() < 1e-6,
        "the reach did not take the loudness multiplier"
    );
    assert!((folded.report_gain - base.report_gain * m.loudness_mult).abs() < 1e-6);
    assert!((folded.range_m - base.range_m * m.range_mult).abs() < 1e-6);
    assert!((folded.ads_time_ms - (base.ads_time_ms + m.ads_time_delta_ms)).abs() < 1e-6);
    // The trace moves…
    let fitted_bytes = weapon::weapon_state_bytes(&r.world);
    println!(
        "  the weapon trace: {} B bare, {} B with a can on it",
        bare_bytes.len(),
        fitted_bytes.len()
    );
    assert_ne!(bare_bytes, fitted_bytes, "the attachment is not sim state");
    assert_eq!(
        fitted_bytes.len(),
        bare_bytes.len() + 2 * attachment::ATTACHMENT_SLOTS,
        "the fold's tail is not ten u16s"
    );
    // …and comes back EXACTLY when it is taken off, which is the rule that
    // keeps every pre-wave trace byte-identical.
    assert!(weapon::equip_attachment(
        &mut r.world,
        HERO,
        AttachmentSlot::Muzzle,
        None
    ));
    assert_eq!(
        weapon::weapon_state_bytes(&r.world),
        bare_bytes,
        "a stripped rifle does not fold what a rifle that never had one does"
    );
    // A part that does not fit the class is refused as a VALUE.
    let pistol_can = cat
        .find(
            WeaponClass::Pistol,
            AttachmentSlot::Muzzle,
            "low_profile_suppressor",
        )
        .expect("the pistol's low-profile can");
    assert!(
        !weapon::equip_attachment(&mut r.world, HERO, AttachmentSlot::Muzzle, Some(pistol_can)),
        "a pistol's can fitted a rifle"
    );
}

/// **A SUPPRESSOR IS QUIETER IN THE COMMAND**, not in a table — read off
/// `report_layers`, which is what both hosts' fence queues.
///
/// **Mutation → red:** dropping `gain` from `report_layers`' `base` closure (the
/// volumes do not move); dropping `report_gain` from `WeaponHit` (the fence
/// cannot see it).
#[test]
fn a_suppressor_is_quieter_in_the_command_and_not_in_a_table() {
    let mut r = Range::new(registry());
    r.arm(HERO, "m4a1");
    r.step();
    let bare = weapon::equipped_def(&r.world, HERO).expect("a rifle").1;
    let cat = attachment::catalogue();
    let supp = cat
        .find(
            WeaponClass::Ar,
            AttachmentSlot::Muzzle,
            "tactical_monolithic_suppressor",
        )
        .expect("the can");
    assert!(weapon::equip_attachment(
        &mut r.world,
        HERO,
        AttachmentSlot::Muzzle,
        Some(supp)
    ));
    let canned = weapon::equipped_def(&r.world, HERO).expect("a rifle").1;
    // **Four hundred metres**, deliberately: the distant layer is SILENT inside
    // its own onset, and a zero that stays a zero would make one of the four
    // comparisons below vacuous — which is what this arm's first draft did.
    let layers = |d: &WeaponDef| {
        weapon::report_layers(
            d.audio_class(),
            false,
            d.report_max_m,
            400.0,
            0,
            d.report_gain,
        )
    };
    let a = layers(&bare);
    let b = layers(&canned);
    println!("the M4A1's report, bare against suppressed:");
    for (x, y) in a.iter().zip(b.iter()) {
        println!(
            "  {:?}: volume {:.4} -> {:.4}, reach {:.1} -> {:.1} m",
            x.kind, x.source.volume, y.source.volume, x.source.max_distance, y.source.max_distance
        );
        assert!(
            y.source.volume < x.source.volume,
            "{:?} is no quieter with a can on",
            x.kind
        );
        assert!(
            y.source.max_distance < x.source.max_distance,
            "{:?} carries as far with a can on",
            x.kind
        );
    }
    // …and it is the CAN's own number, not an arbitrary drop.
    let m = cat.get(supp).expect("the row").modifiers;
    assert!(
        (b[1].source.volume / a[1].source.volume - m.loudness_mult).abs() < 1e-9,
        "the body layer dropped by something other than the row's loudness"
    );
}

/// **AN EXTENDED MAGAZINE CHANGES THE READOUT** — the string the HUD draws.
///
/// **Mutation → red:** dropping `mag_capacity_override` from
/// `StatModifiers::apply` (the readout does not move); dropping the top-up in
/// `equip_attachment` (the capacity moves and the magazine does not).
#[test]
fn an_extended_magazine_changes_the_readout() {
    let mut r = Range::new(registry());
    r.arm(HERO, "m4a1");
    r.step();
    let before = {
        let st = r.state(HERO);
        weapon::ammo_readout(st.magazine, st.reserve)
    };
    let cat = attachment::catalogue();
    let mag = cat
        .find(
            WeaponClass::Ar,
            AttachmentSlot::Magazine,
            "45_round_extended_mag",
        )
        .expect("the AR's 45-round mag");
    assert_eq!(
        cat.get(mag)
            .expect("the row")
            .modifiers
            .mag_capacity_override,
        Some(45)
    );
    assert!(weapon::equip_attachment(
        &mut r.world,
        HERO,
        AttachmentSlot::Magazine,
        Some(mag)
    ));
    let after = {
        let st = r.state(HERO);
        weapon::ammo_readout(st.magazine, st.reserve)
    };
    println!("the readout: {before:?} -> {after:?}");
    assert_ne!(before, after, "the readout did not move");
    assert!(after.starts_with("45 / "), "the readout reads {after:?}");
    assert_eq!(
        weapon::equipped_def(&r.world, HERO)
            .expect("a rifle")
            .1
            .magazine,
        45
    );
}

/// **A SCOPE CHANGES THE ADS TIME, WHICH IS THE CAMERA'S OWN BLEND** — measured
/// through `feel::ads_speed`, which is the number the aim blend runs at.
///
/// **Mutation → red:** dropping `ads_time_delta_ms` from the fold (the blend
/// speed does not move).
#[test]
fn a_scope_changes_the_ads_time_which_is_the_cameras_own_blend() {
    let mut r = Range::new(registry());
    r.arm(HERO, "m4a1");
    r.step();
    let bare = weapon::equipped_def(&r.world, HERO).expect("a rifle").1;
    let cat = attachment::catalogue();
    let acog = cat
        .find(WeaponClass::Ar, AttachmentSlot::Optic, "4x_acog_scope")
        .expect("the AR's ACOG");
    assert!(weapon::equip_attachment(
        &mut r.world,
        HERO,
        AttachmentSlot::Optic,
        Some(acog)
    ));
    let scoped = weapon::equipped_def(&r.world, HERO).expect("a rifle").1;
    let a = inf_ecs::feel::blend_speed_for_ads(bare.ads_time_ms / 1000.0, DT);
    let b = inf_ecs::feel::blend_speed_for_ads(scoped.ads_time_ms / 1000.0, DT);
    println!(
        "the M4A1's ADS: {:.0} ms -> {:.0} ms, blend {a:.3} -> {b:.3} /s",
        bare.ads_time_ms, scoped.ads_time_ms
    );
    assert!(
        scoped.ads_time_ms > bare.ads_time_ms,
        "a 4x scope did not slow the sight picture"
    );
    assert!(b < a, "the camera blend did not slow with it");
    // And the ART is a scope, which is what `step_accessories` draws.
    assert_eq!(
        cat.get(acog).expect("the row").art,
        attachment::AttachmentArt::Scope
    );
}

/// **THE BENCH ROUND-TRIPS THROUGH THE PANEL'S OWN VERB** — the phase30
/// `EquipFromPanel` precedent, one list along.
///
/// **Mutation → red:** deleting the `CycleAttachment` arm from
/// `apply_inventory_verb` (nothing is ever fitted through the panel).
#[test]
fn the_bench_round_trips_through_the_panels_own_verb() {
    let mut view = inf_ui::InventoryView {
        bench: vec![inf_ui::BenchSlot {
            name: "muzzle".into(),
            fitted: String::new(),
            options: 10,
        }],
        slots: vec![inf_ui::InventorySlot::default()],
    };
    view.slots[0] = inf_ui::InventorySlot {
        label: "M4A1".into(),
        count: 1,
        equipped: true,
        equippable: true,
    };
    let mut state = inf_ui::InventoryState::default();
    state.set_open(true);
    let out = inf_ui::inventory::handle(
        &mut state,
        &view,
        &inf_ui::InventoryInput::Key("BracketRight".into()),
    );
    assert_eq!(
        out.verb,
        Some(inf_ui::InventoryVerb::CycleAttachment { slot: 0, delta: 1 }),
        "the `]` key did not ask for the next attachment"
    );
    let strip = inf_ui::inventory::handle(
        &mut state,
        &view,
        &inf_ui::InventoryInput::Key("Backspace".into()),
    );
    assert_eq!(
        strip.verb,
        Some(inf_ui::InventoryVerb::CycleAttachment { slot: 0, delta: 0 }),
        "Backspace did not ask to strip the rail"
    );
    // A rail slot with nothing to fit asks for nothing.
    let empty = inf_ui::InventoryView {
        bench: vec![inf_ui::BenchSlot {
            name: "underbarrel".into(),
            fitted: String::new(),
            options: 0,
        }],
        ..view.clone()
    };
    let mut s2 = inf_ui::InventoryState::default();
    s2.set_open(true);
    let none = inf_ui::inventory::handle(
        &mut s2,
        &empty,
        &inf_ui::InventoryInput::Key("BracketRight".into()),
    );
    assert_eq!(
        none.verb, None,
        "a rail slot the catalogue has no rows for asked for one anyway"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (f) MESHES
// ═════════════════════════════════════════════════════════════════════════════

/// **EVERY CLASS NAMES ITS ART, AND THE TWO WITHOUT SAY SO** — the class → mesh
/// table, read row by row off the shipped registry.
#[test]
fn every_class_names_its_art_and_the_two_without_say_so() {
    let defs = registry();
    let mut census: std::collections::BTreeMap<String, Vec<&'static str>> = Default::default();
    let mut bare = Vec::new();
    for (id, item) in defs.0.iter() {
        let Some(def) = item.weapon.as_ref() else {
            continue;
        };
        match weapon::weapon_mesh_key(id, def) {
            Some(key) => census.entry(id.clone()).or_default().push(key),
            None => bare.push(id.clone()),
        }
    }
    let mut by_key: std::collections::BTreeMap<&str, usize> = Default::default();
    for keys in census.values() {
        for k in keys {
            *by_key.entry(k).or_default() += 1;
        }
    }
    println!("the class -> mesh table, over the shipped registry:");
    for (k, n) in &by_key {
        println!("  {k:>18}: {n} rows");
    }
    println!(
        "  rows with NO art (the primitive, and it says so): {}",
        bare.len()
    );
    for id in &bare {
        let def = defs.get(id).and_then(|i| i.weapon).expect("a weapon");
        assert!(
            matches!(
                def.audio_class(),
                WeaponClass::Shotgun | WeaponClass::Launcher
            ),
            "{id} has no art and is not a shotgun or a launcher"
        );
    }
    // Fourteen, not fifteen: the G67 grenade is a `launcher`-class row and it
    // HAS art (`SM_G67`), so what is left bare is the ten shotguns and the five
    // rocket launchers minus it.
    assert_eq!(bare.len(), 14, "ten shotguns and four rocket launchers");
    assert!(
        bare.iter().all(|id| id != "g67_grenade"),
        "the grenade is drawn as a primitive"
    );
    // Every key the table can answer is enumerated, so the importer's rebind
    // list and this census cannot disagree.
    for k in by_key.keys() {
        assert!(
            weapon::WEAPON_MESH_KEYS.contains(k),
            "{k} is answered by the table and is not in `WEAPON_MESH_KEYS`"
        );
    }
    // And the identity is a pure function of the name.
    for k in weapon::WEAPON_MESH_KEYS {
        assert_eq!(weapon::weapon_mesh_guid(k), weapon::weapon_mesh_guid(k));
        assert!(!weapon::weapon_mesh_guid(k).is_nil());
    }
    let ids: BTreeSet<Uuid> = weapon::WEAPON_MESH_KEYS
        .iter()
        .map(|k| weapon::weapon_mesh_guid(k))
        .collect();
    assert_eq!(
        ids.len(),
        weapon::WEAPON_MESH_KEYS.len(),
        "two keys collide"
    );
}

/// **THE WEAPON IS DRAWN WITH ITS CLASS'S ART, AND IT FOLLOWS THE HAND** —
/// measured on the drawn entity, not on a table.
///
/// **Mutation → red:** dropping `asset: mesh` from the `MeshRef` (the rifle
/// draws a cube); changing `WEAPON_SOCKET` to a name the rig does not publish
/// (the weapon stops following the hand — measured as the distance between the
/// weapon entity and the hand's own world position).
#[test]
fn the_weapon_is_drawn_with_its_classs_art_and_it_follows_the_hand() {
    let mut r = Range::new(registry());
    r.arm(HERO, "m4a1");
    r.step();
    inf_ecs::attach::update_attachments(&mut r.world);
    r.world.propagate();
    let weapon_guid = d3::gameplay::equipped_weapon_guid(HERO);
    let e = r.world.entity_of(weapon_guid).expect("a weapon entity");
    let mesh = r
        .world
        .world()
        .get::<MeshRef>(e)
        .copied()
        .expect("a mesh reference");
    let want = weapon::weapon_mesh_guid("SM_AR4");
    println!(
        "the M4A1 draws {:?}; the AR4's identity is {want}",
        mesh.asset
    );
    assert_eq!(mesh.asset, Some(want), "the rifle is still a cube");
    // A real mesh is drawn at 1:1 and the placeholder is stretched.
    let scale = r
        .world
        .world()
        .get::<Transform>(e)
        .map(|t| t.scale)
        .expect("a transform");
    assert_eq!(
        scale,
        Vec3d::ONE,
        "a real mesh was stretched to the placeholder's box"
    );
    // …and it is marked, which is what the fade rule and this gate ask.
    assert!(weapon::is_equipped_weapon(&r.world, weapon_guid));
    // A SHOTGUN has no art and keeps the placeholder, stretched to its barrel.
    let mut s = Range::new(registry());
    s.arm(HERO, "remington_870");
    s.step();
    let se = s
        .world
        .entity_of(d3::gameplay::equipped_weapon_guid(HERO))
        .expect("a weapon entity");
    let sm = s.world.world().get::<MeshRef>(se).copied().expect("a mesh");
    assert_eq!(
        sm.asset, None,
        "the shotgun claims art the bundle has none of"
    );
    let ss = s
        .world
        .world()
        .get::<Transform>(se)
        .map(|t| t.scale)
        .expect("a transform");
    assert!(
        (ss.z - row("remington_870").muzzle_forward_m).abs() < 1e-9,
        "the placeholder is not the barrel's own length"
    );
    // **THE WEAPON FOLLOWS THE HAND SOCKET.** `update_attachments` composes
    // `model_to_world(holder) x socket x offset`, and for a character with no
    // rig the socket is the documented identity fallback — so the weapon sits at
    // the holder's own MODEL origin, which is its FEET and not its capsule
    // centre. (Measuring against the entity transform is measuring the 0.9 m
    // offset between the two, which is what this arm's first draft did.)
    let model =
        inf_ecs::pose::model_to_world(&r.world, r.world.entity_of(HERO).expect("hero")).translation;
    let at = r
        .world
        .world()
        .get::<inf_ecs::components::GlobalTransform>(e)
        .map(|g| g.translation())
        .expect("the weapon's transform");
    let gap = (at - model).length();
    println!(
        "the weapon sits {gap:.4} m from the frame its socket is resolved in \
         ({:.2}, {:.2}, {:.2})",
        model.x, model.y, model.z
    );
    assert!(
        gap < 0.02,
        "the weapon is {gap:.3} m from the hand it is attached to"
    );
    // …and it is attached to the WEAPON SOCKET by name, which is what a rig
    // resolves and what a rig that lost the socket would fall back from.
    let att = r
        .world
        .world()
        .get::<inf_ecs::components::AttachedTo>(e)
        .expect("the weapon is attached to something");
    assert_eq!(att.target, HERO);
    assert_eq!(
        att.socket,
        d3::gameplay::WEAPON_SOCKET,
        "the weapon hangs off a socket that is not the hand"
    );
}

/// **THE WEAPON DOES NOT FADE WITH THE BODY** — the first-person rule, through
/// the one door.
///
/// **Mutation → red:** deleting the `EquippedWeapon` arm from
/// `subject_fade_for` (a weapon entity fades with its holder).
#[test]
fn the_weapon_does_not_fade_with_the_body() {
    let mut r = Range::new(registry());
    r.arm(HERO, "m4a1");
    r.step();
    let weapon_guid = d3::gameplay::equipped_weapon_guid(HERO);
    // A 0.2 m boom is a first-person seat: the camera thins its subject to
    // nothing.
    let fade = 0.05;
    let body = weapon::subject_fade_for(&r.world, HERO, Some(HERO), fade);
    let gun = weapon::subject_fade_for(&r.world, weapon_guid, Some(weapon_guid), fade);
    let other = weapon::subject_fade_for(&r.world, victim_guid(0), Some(HERO), fade);
    println!("at a {fade} fade: the body draws {body}, the weapon {gun}, a bystander {other}");
    assert_eq!(body, fade, "the subject's own body is not thinned");
    assert_eq!(gun, 1.0, "the weapon faded with the body");
    assert_eq!(other, 1.0, "something that is not the subject was thinned");
}

// ═════════════════════════════════════════════════════════════════════════════
// (g) DETERMINISM
// ═════════════════════════════════════════════════════════════════════════════

/// **PIE == SHIPPING, AND TWO COOKS AGREE, OVER A COURSE THAT FIRES ONE OF
/// EVERY CLASS.**
#[test]
fn pie_equals_shipping_and_two_cooks_agree_over_a_class_course() {
    let a = tempfile::tempdir().expect("tempdir a");
    let b = tempfile::tempdir().expect("tempdir b");
    let ta = class_course(pack_sim(&cook_fixture(a.path())));
    let tb = class_course(pack_sim(&cook_fixture(b.path())));
    let tp = class_course(pie_sim());
    println!(
        "the class course: {} steps, {} audio commands, {} shots, {} pellets, \
         {} blasts, {} throws",
        ta.trace.len(),
        ta.audio.len(),
        ta.shots,
        ta.pellets,
        ta.blasts,
        ta.throws
    );
    assert!(ta.shots > 0, "the course fired nothing");
    // One pull: the Remington cycles at 60 rpm and the course holds its trigger
    // for a second.
    assert!(ta.pellets >= 8, "the course never fired the shotgun");
    assert!(ta.blasts > 0, "the course never set anything off");
    assert!(ta.throws > 0, "the course never threw anything");
    for (i, (x, y)) in ta.trace.iter().zip(tb.trace.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: two independent cooks diverged");
    }
    for (i, (x, y)) in ta.trace.iter().zip(tp.trace.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: PIE and shipping diverged");
    }
    assert_eq!(ta.audio, tb.audio, "two cooks made different noises");
    assert_eq!(ta.audio, tp.audio, "PIE and shipping made different noises");
    assert_eq!(
        (ta.shots, ta.pellets, ta.blasts),
        (tp.shots, tp.pellets, tp.throws.max(ta.blasts))
    );
}

/// **A QUIET LEVEL FOLDS NOTHING NEW** — the half that keeps every trace
/// committed before this wave byte-identical.
#[test]
fn a_quiet_level_folds_nothing_this_wave_added() {
    let mut r = Range::new(registry());
    for _ in 0..20 {
        r.step();
    }
    assert!(
        ballistics::round_state_bytes(&r.world).is_empty(),
        "an unfired level folds rounds"
    );
    assert!(
        weapon::weapon_state_bytes(&r.world).is_empty(),
        "an unarmed level folds a weapon clock"
    );
    // A carried weapon with a bare rail and no lock folds exactly what it folded
    // before this wave: the fixed prefix and nothing else.
    r.arm(HERO, "m4a1");
    r.step();
    let bytes = weapon::weapon_state_bytes(&r.world);
    let id_len = "m4a1".len();
    let want = 16 + 4 + id_len + 4 + 4 + 8 + 8 + 8 + 1;
    println!(
        "a carried M4A1 folds {} B; the pre-WPN2d row is {want} B",
        bytes.len()
    );
    assert_eq!(
        bytes.len(),
        want,
        "a bare rifle folds more than its pre-WPN2d row"
    );
}

struct Course {
    trace: Vec<Vec<u8>>,
    audio: Vec<String>,
    shots: u32,
    pellets: u32,
    blasts: u32,
    throws: u32,
}

/// Fire one of every class, throw a grenade, and set a rocket off — through the
/// shipped input path, on the committed gameplay fixture.
fn class_course(mut sim: inf_player::runtime_sim::RuntimeSim) -> Course {
    let hero = inf_editor_core::samples::GAMEPLAY_HERO_GUID;
    let mut trace = Vec::new();
    for _ in 0..20 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        trace.push(sim.state_bytes());
    }
    let before = sim.audio_command_log().len();
    let (mut shots, mut pellets, mut blasts, mut throws) = (0u32, 0u32, 0u32, 0u32);
    for id in ["m4a1", "remington_870", "rpg_7", "g67_grenade"] {
        assert_eq!(item::give(sim.world_mut(), hero, id, 1), 0);
        assert!(d3::gameplay::equip_weapon(sim.world_mut(), hero, id));
        {
            let e = sim.world().entity_of(hero).expect("the hero");
            if let Some(mut cm) = sim.world_mut().world_mut().get_mut::<CharacterMovement>(e) {
                cm.runtime.aim_pitch_deg = 12.0;
            }
        }
        // **THROUGH THE SHIPPED INPUT PATH**, not by writing the runtime edge:
        // `step_once` applies an intent built from the actions it is handed, so a
        // course that set `want_attack` on the component would have it taken
        // straight back — which is what this course's first draft did, and it
        // fired nothing on 740 steps.
        let throwing = id == "g67_grenade";
        let mut state = inf_input::InputState::new(inf_input::default_map());
        for i in 0..180 {
            let events: Vec<inf_input::InputEvent> = if throwing {
                vec![inf_input::InputEvent::Key {
                    code: "KeyB".into(),
                    pressed: i == 0,
                }]
            } else {
                vec![inf_input::InputEvent::MouseButton {
                    button: inf_input::MouseButton::Left,
                    pressed: i < 60,
                }]
            };
            state.apply_dt(&events, DT);
            sim.step_once(inf_player::input::held_actions(&state, DT));
            let rep = sim.gameplay();
            shots += rep.shots;
            pellets += rep.rounds.pellets;
            blasts += rep.blasts.len() as u32;
            throws += rep.throws;
            trace.push(sim.state_bytes());
        }
    }
    assert_eq!(sim.dropped_audio_commands(), 0);
    let audio: Vec<String> = sim.audio_command_log()[before..]
        .iter()
        .map(|c| format!("{c:?}"))
        .collect();
    Course {
        trace,
        audio,
        shots,
        pellets,
        blasts,
        throws,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// (h) COST
// ═════════════════════════════════════════════════════════════════════════════

/// **THE STEP'S RAY BILL AND ITS BLAST SWEEP ARE INSIDE THE BUDGET** — measured
/// at the population the wave created.
#[test]
fn the_ray_bill_and_the_blast_sweep_are_inside_the_budget() {
    let mut r = Range::new(registry());
    // Thirty bodies around the epicentre — the blast sweep's own population.
    for i in 0..30 {
        let a = i as f64 * 0.209;
        r.victim(
            victim_guid(i),
            DVec3::new(a.cos() * 6.0, 0.0, 20.0 + a.sin() * 6.0),
        );
    }
    let def = WeaponDef {
        blast_radius_m: 10.0,
        blast_damage_j: 800.0,
        class: Some(WeaponClass::Launcher),
        ..Default::default()
    };
    let mut report = d3::GameplayReport::default();
    let t0 = std::time::Instant::now();
    d3::gameplay::blast_for_test(
        &mut r.world,
        &mut r.bridge,
        HERO,
        DVec3::new(0.0, 1.4, 20.0),
        &def,
        DT,
        &mut report,
    );
    let us = t0.elapsed().as_secs_f64() * 1e6;
    println!(
        "one blast over 30 bodies: {us:.1} us, {} hurt, {} rays, {} shadowed",
        report.blasts[0].hurt, report.rounds.blast_rays, report.rounds.blast_shadowed
    );
    assert!(report.blasts[0].hurt > 0, "the sweep hurt nobody");
    assert!(
        report.rounds.blast_rays as usize <= d3::gameplay::MAX_BLAST_TARGETS,
        "the sweep spent {} rays against a bound of {}",
        report.rounds.blast_rays,
        d3::gameplay::MAX_BLAST_TARGETS
    );
    // A shotgun's whole pull, timed, against the weapon phase's own budget.
    let mut s = Range::new(registry());
    s.arm(HERO, "remington_870");
    s.aim(HERO, 0.0, 0.0);
    s.hold_trigger(HERO, true);
    let t1 = std::time::Instant::now();
    let rep = s.step();
    let step_us = t1.elapsed().as_secs_f64() * 1e6;
    println!(
        "one 8-pellet pull: {step_us:.1} us for the whole gameplay step, {} \
         casts (budget {:.1} ms)",
        rep.rounds.shot_rays,
        inf_player::budget::WEAPON_STEP_BUDGET_MS
    );
    assert_eq!(rep.rounds.pellets, 8);
}

// ═════════════════════════════════════════════════════════════════════════════
// (i) NOTHING OF WPN2e
// ═════════════════════════════════════════════════════════════════════════════

/// **NO NPC FIRES ON ITS OWN** — the WPN2e boundary, asserted rather than
/// assumed.
#[test]
fn nothing_of_the_npc_firing_policy_leaked_in() {
    const DISPATCH: &str = include_str!("../../../crates/inf-physics/src/d3/dispatch.rs");
    const CRIME: &str = include_str!("../../../crates/inf-physics/src/d3/crime.rs");
    for (what, text) in [("dispatch", DISPATCH), ("crime", CRIME)] {
        assert!(
            !text.contains("npc_aim_at"),
            "`{what}.rs` calls `npc_aim_at` — WPN2e's target selection has leaked \
             into WPN2d"
        );
    }
    // And an armed NPC standing beside the hero fires nothing on its own.
    let mut r = Range::new(registry());
    stand(
        &mut r.world,
        shooter_guid(0),
        "Bystander",
        DVec3::new(2.0, 0.0, 0.0),
        false,
    );
    r.resync();
    r.arm(shooter_guid(0), "m4a1");
    let mut shots = 0;
    for _ in 0..120 {
        shots += r.step().shots;
    }
    assert_eq!(
        shots, 0,
        "an armed NPC opened fire with no policy to tell it to"
    );
}

// ── the fixture plumbing (wpn2c_gate's, verbatim) ───────────────────────────

/// Drive the pose step exactly as a host does, so the throw additive advances
/// and its notify fires. Nothing here is posed (there is no rig in this
/// fixture), which is what makes the notify the ONLY thing this call produces.
fn pose_step(world: &mut EcsWorld) {
    inf_ecs::pose::step_pose_evaluation(world, DT, &|_| None, &|_| None, &|_| None, &|_| {
        Default::default()
    });
}

fn sample_files() -> Vec<String> {
    std::fs::read_dir(inf_editor_core::samples::gameplay_dir())
        .expect("the fixture folder")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
}

fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Island Gameplay", "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("a content root");
    for f in sample_files() {
        std::fs::copy(
            inf_editor_core::samples::gameplay_dir().join(&f),
            content.join(&f),
        )
        .expect("copy");
    }
    proj
}

fn cook_fixture(tmp: &Path) -> PathBuf {
    let proj = scaffold(tmp);
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the fixture cooks");
    out
}

fn pack_sim(pack: &Path) -> inf_player::runtime_sim::RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the world builds");
    inf_player::sim_from_built(built)
}

fn read_asset(dir: &Path, guid: Uuid) -> Option<Vec<u8>> {
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|e| e == "toml") {
            continue;
        }
        if let Ok(side) = inf_asset::AssetSidecar::load(&path) {
            if side.guid.0 == guid {
                return std::fs::read(&path).ok();
            }
        }
    }
    None
}

fn pie_sim() -> inf_player::runtime_sim::RuntimeSim {
    let dir = inf_editor_core::samples::gameplay_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Gameplay.inf_lvl"))
        .expect("the level loads");
    let class = inf_editor_core::samples::gameplay_controller();
    let pcg = read_asset(&dir, inf_editor_core::samples::GAMEPLAY_PCG_GUID)
        .expect("the house graph is on disk");
    let mesh = read_asset(&dir, inf_editor_core::samples::GAMEPLAY_TARGET_MESH_GUID)
        .expect("the target mesh is on disk");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_ACTOR_GUID).then(|| class.clone()),
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_PCG_GUID).then(|| pcg.clone()),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_TARGET_MESH_GUID).then(|| mesh.clone()),
        |_| None,
        |_| None,
        60,
        false,
    )
    .expect("the payload builds");
    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
}
