//! **The vehicle fixed-step door** (P29.7): cast the wheel rays, apply what the
//! model asks for, write the wheels back.
//!
//! The `inf_physics` half of `inf_ecs::vehicle`, and the same split as
//! [`super::camera`] / `inf_ecs::camera` and [`super::movement`] /
//! `inf_ecs::movement`: everything here touches rapier and nothing here decides
//! anything. The arithmetic — the spring, the engine curve, the friction circle
//! — is on the other side of that wall and is unit-tested without a world.
//!
//! # One door, and since island wave VEH1a its own phase
//!
//! [`step_vehicles`] used to be the **last statement** of
//! [`super::movement::step_character_movement`]: P29.7 put it there because *"a
//! sibling function called separately by each host would be a hand-maintained
//! mirror"*, and that phase's own ledger records two defects of exactly that
//! shape (the intent feed and the publish order).
//!
//! It is now called by each host directly, immediately after the movement door
//! returns — **the same slot, one function up** — and it has a `STEP_PHASES` row
//! of its own (`vehicle`, index 12). The ordering argument is untouched: a
//! driver's controls are written by the character step above, from the same
//! intent, and the forces must land before `bridge.step`.
//!
//! What was bought is attribution. On the island a car is not a corner of the
//! character step — it casts four rays into a streamed heightfield sixty times a
//! second — and *"a step that cannot say where its milliseconds went"* is the
//! defect wave I4b existed to remove. What it costs is a real mirror, and the
//! mirror is **paid for rather than avoided**: both call sites live inside
//! `// MIRROR-BEGIN vehicle_step` fences and are pinned character-for-character
//! by `inf-editor-core`'s `tests/fixed_step_mirror.rs`. That is a stronger guard
//! than the old arrangement had, which pinned the *call* and left nothing at all
//! pinning the two hosts agreeing about **when**.
//!
//! `O(vehicles)`, never `O(entities)`: the set is discovered inside the entity
//! walk `sync_from_world_sim` already makes, and this function walks the map.
//!
//! # Force ownership, which is not obvious
//!
//! A rapier force **persists** until `reset_forces` (P20.2's law), so a pass
//! that applies one must own the clear. The water pass resets and re-applies
//! every buoyant body at fixed-step stage 8, and this door runs at stage 12 — so
//! for a vehicle that is *also* buoyant, water's reset is already the clear and
//! resetting again here would delete this step's buoyancy before the solver ever
//! saw it. The rule is therefore one line with a real reason:
//! **reset iff the water pass does not already own this body**
//! ([`PhysicsBridge3D::is_buoyant`]).

use std::collections::BTreeSet;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{Collider3D, Terrain, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::{
    ChassisState, Footprint, SubstepAdvance, SurfaceClass, WheelContact, WheelForce, MAX_SUBSTEPS,
};
use inf_ecs::EcsWorld;

use super::query::CastTargets;
use super::{ColliderId3D, PhysicsBridge3D};

/// What one vehicle's step did — returned so a test can assert on the
/// *decisions*, while the arms that matter assert on the WORLD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleOutcome {
    /// The chassis entity.
    pub chassis: Uuid,
    /// How many of its wheels found ground this step.
    pub wheels_grounded: usize,
    /// The suspension load summed over those wheels, newtons — a rig at rest on
    /// flat ground reports its own weight, which is the cheapest possible check
    /// that the springs are doing their job.
    pub load_n: f64,
    /// Forward speed, m/s, signed (negative is reversing).
    pub forward_mps: f64,
    /// How fast the engine is turning over, `[0, 1]` — the class's own
    /// [`Vehicle::engine_state`], island wave VEH1a.
    ///
    /// [`Vehicle::engine_state`]: inf_ecs::vehicle::Vehicle::engine_state
    pub revs: f64,
    /// How hard the driver is asking, `[0, 1]` — the other half of the same
    /// answer.
    ///
    /// Published on the outcome rather than read back off the trait by the audio
    /// step, so the engine's sound is a function of **this step's** decision and
    /// not of whatever the vehicle map holds when a later phase gets around to
    /// asking. That is the P12 doctrine's own shape: the command stream is a
    /// pure function of sim state, and a report published at the moment the
    /// state was decided is the cheapest way to keep it one.
    pub load: f64,
}

/// **Advance every vehicle one fixed step.**
///
/// Called by each host immediately after
/// [`super::movement::step_character_movement`] returns — after every character
/// has moved (so a driver's controls are this step's) and before the solver.
pub fn step_vehicles(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
) -> Vec<VehicleOutcome> {
    if !dt.is_finite() || dt <= 0.0 {
        return Vec::new();
    }
    let guids = bridge.vehicle_guids();
    if guids.is_empty() {
        // **A level with no vehicle may still have a bumper lying in the road**
        // (wave VEH3c). Debris nothing reaps is a leak with no deadline, and the
        // last car on a level can be despawned while its own door is on the
        // floor -- so the bodywork step runs on the empty case too, where it is
        // one `is_none` on a level that never had a car.
        super::bodywork::step_bodywork(world, bridge, dt);
        return Vec::new();
    }
    // **THE GROUND'S OWN SURFACE** (wave VEH3a), derived once a step and cheap
    // when nothing moved — `traffic::sync_carriageway`'s pattern, and it is here
    // rather than at level load for that door's reason: a level whose terrain or
    // blocks change during a session (the editor's whole job) gets a map that
    // followed them, and one that did not pays a stamp walk.
    inf_ecs::vehicle::sync_surface_map(world);
    // The weather, read ONCE for every vehicle in the level rather than once per
    // wheel: it is the same sky.
    let (wetness, ambient_c) = inf_ecs::vehicle::weather_at(world);
    let mut out = Vec::with_capacity(guids.len());
    let mut forces: Vec<WheelForce> = Vec::new();
    // **What every drivetrain did, gathered for the trace** (wave VEH3b). The
    // crank, the clutch and the turbo live inside a `dyn Vehicle` in this
    // bridge, and `state_bytes` is handed a WORLD -- so the walk that already
    // visits every vehicle every step is where the answer crosses. Rebuilt
    // whole, so a car that was despawned leaves nothing behind.
    let mut drivetrains: Vec<(Uuid, inf_ecs::vehicle::DrivetrainState, f64)> =
        Vec::with_capacity(guids.len());
    for chassis in guids {
        if let Some(o) = step_one(world, bridge, chassis, dt, wetness, ambient_c, &mut forces) {
            if let Some(v) = bridge.vehicle_of(chassis) {
                if let Some(state) = v.drivetrain() {
                    drivetrains.push((chassis, state, v.idle_rpm()));
                }
            }
            out.push(o);
        }
    }
    inf_ecs::vehicle::publish_drivetrains(world, drivetrains);
    // **The bodywork, last** (wave VEH3c) -- the crash that tears a bumper off,
    // the hinge that swings a door, the pane, the dent, the fire and the reap.
    // Here rather than in a sibling each host calls for `step_vehicles`' own
    // reason, one level down: a sibling would be a hand-maintained mirror, and
    // this rides the `vehicle_step` fence both hosts already carry.
    super::bodywork::step_bodywork(world, bridge, dt);
    out
}

/// How many rays one tyre's footprint is sampled with (wave VEH3a).
///
/// Four, the bottom of the research doc's *4–8 raycasts per tire footprint*
/// band. The cost is linear and it is paid every step by every simulated wheel,
/// so the number that matters is the measured one in `VEHICLE_STEP_BUDGET_MS`'s
/// own arm rather than the one the doc suggests.
pub const FOOTPRINT_SAMPLES: usize = 4;

/// **What surface a hit entity is** (wave VEH3a), from the field that has been
/// on `Collider3D` since P12.1 and that the tyre model has never read.
///
/// # A friction is not a surface identity, and mapping it as one cost a wave
///
/// The first cut nearest-matched `Collider3D::friction` against
/// [`SurfaceClass::dry_mu`], which reads as the obvious thing to do and is a
/// category error: that field is the **solver's** Coulomb coefficient for
/// box-on-box contact, chosen by whoever wanted a crate to stop sliding, and it
/// is not a tyre's µ. Measured immediately — the feel-table fixture's ground
/// carries `friction: 0.9`, which nearest-matches to concrete (0.95) and cost
/// the sports row **five per cent** of its grip. That five per cent took its
/// 0–100 km/h from **3.98 s to 10.70**, because a car whose drive force sits
/// just inside its traction limit falls out of it and its traction control then
/// throttles the launch.
///
/// So the mapping is BANDED, and the band that matters is the top one:
///
/// * **`friction >= 0.85` is a SEALED surface** — asphalt, µ 1.0. Both values in
///   committed content land here (the Ring-0 default 0.5 is exempt below, and
///   the ground fixtures use 0.9), so no level this repository ships changes
///   grip at all.
/// * below that, the value IS read as the surface's own µ and nearest-matched,
///   so an author says *this is gravel* with `friction = 0.6` and *this is mud*
///   with `0.35` — through a field that already exists, at zero schema cost,
///   which is what the VEH3a price ruled.
/// * the Ring-0 **default is exempt**, and that is load-bearing:
///   `Collider3D::default().friction` is 0.5, which nearest-matches to sand
///   (0.45), so without the branch every kerb, bridge and building in every
///   committed level would become a beach.
///
/// What this leaves for a later wave, by name: the TERRAIN heightfield answers
/// asphalt like everything else, so the island's grass verges are not soft yet.
/// The honest door for that is the P19 biome map read at the contact point, and
/// it is a lookup this bridge cannot reach today.
fn surface_under(world: &EcsWorld, guid: Uuid, at: DVec3) -> SurfaceClass {
    let Some(entity) = world.entity_of(guid) else {
        return SurfaceClass::Asphalt;
    };
    // **THE TERRAIN ANSWERS FROM ITS MAP**, not from its collider's friction.
    // A heightfield is ONE collider over a whole island, so a friction on it
    // could only ever say one thing about fifty square kilometres of road,
    // verge, beach and forest floor. `SurfaceMap` is the level's own splat and
    // its own carriageway, sampled where the wheel actually is.
    if world.world().get::<Terrain>(entity).is_some() {
        if let Some(class) = inf_ecs::vehicle::surface_map_of(world)
            .and_then(|res| res.maps.get(&guid))
            .and_then(|map| map.at(at.x, at.z))
        {
            return class;
        }
    }
    let Some(col) = world.world().get::<Collider3D>(entity) else {
        return SurfaceClass::Asphalt;
    };
    if !col.friction.is_finite()
        || col.friction >= SEALED_SURFACE_FRICTION
        || (col.friction - DEFAULT_COLLIDER_FRICTION).abs() < 1e-9
    {
        return SurfaceClass::Asphalt;
    }
    let mut best = SurfaceClass::Asphalt;
    let mut best_err = f64::INFINITY;
    for s in SurfaceClass::all() {
        let err = (s.dry_mu() - col.friction).abs();
        if err < best_err {
            best_err = err;
            best = s;
        }
    }
    best
}

/// At or above this, a collider's friction says *sealed road* rather than a µ.
const SEALED_SURFACE_FRICTION: f64 = 0.85;

/// `Collider3D::default().friction`, restated here because
/// [`surface_under`] must recognise *the author said nothing* and a value read
/// from the type it is comparing against would agree with anything.
const DEFAULT_COLLIDER_FRICTION: f64 = 0.5;

fn step_one(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    chassis: Uuid,
    dt: f64,
    wetness: f64,
    ambient_c: f64,
    forces: &mut Vec<WheelForce>,
) -> Option<VehicleOutcome> {
    let body = bridge.body_of(chassis)?;
    let position = bridge.world().body_translation(body)?;
    // Wave VEH2c: the sea under this vehicle, from the SAME index and the same
    // sim clock the buoyancy pass reads at stage 8 — never a wall clock and
    // never a second opinion, so PIE and the shipped player cannot disagree
    // about where the water is. `O(bodies over the cell)`, and `O(1)` on a level
    // with no water, which is what keeps a wheeled vehicle from paying for it.
    let water_y = bridge.water_surface_at(glam::DVec2::new(position.x, position.z));
    let state = {
        let w = bridge.world();
        ChassisState {
            position,
            rotation: w.body_rotation(body)?,
            linvel: w.body_linvel(body).unwrap_or(DVec3::ZERO),
            angvel: w.body_angvel(body).unwrap_or(DVec3::ZERO),
            mass_kg: w.body_mass(body).unwrap_or(0.0),
            water_y,
        }
    };
    let (_, _, up) = state.basis();
    // A wheel ray starts inside the vehicle it belongs to, so the exclusion is
    // structural rather than optional. The wheels themselves are NOT in this set
    // because they are not in rapier at all — the bridge consumes a wheel rather
    // than mirroring it (see `inf_ecs::vehicle`'s module docs).
    let mut exclude: BTreeSet<ColliderId3D> = BTreeSet::new();
    if let Some(c) = bridge.collider_of(chassis) {
        exclude.insert(c);
    }

    // ── 1. the rays. FOUR per wheel since wave VEH3a, `O(4 · wheels)`.
    //
    // The research doc asks for *4-8 raycasts per tire footprint*, and the reason
    // is the kerb: one ray at the wheel's centre sees a kerb as a step change —
    // the wheel is either fully on it or fully off it — while four at the corners
    // of the contact patch see it arrive under one edge first, which is what a
    // real tyre does. The wheel rides on the HIGHEST ground under its patch (the
    // shortest ray), and the surface it works against is the four normals'
    // average, which is the bilinear smoothing the 15.69° snap on open DTM relief
    // has been waiting for since P29.7.
    let (fwd_b, right_b, _) = state.basis();
    let footprint = world
        .world()
        .get_resource::<Footprint>()
        .copied()
        .unwrap_or(Footprint::SHIPPED);
    let rays: Vec<[(DVec3, f64); FOOTPRINT_SAMPLES]> = {
        let v = bridge.vehicle_of(chassis)?;
        let rest = v.suspension_rest_m();
        v.rig()
            .wheels
            .iter()
            .map(|wm| {
                // The authored child transform is the wheel CENTRE at full
                // extension; the strut anchor is `rest` above it, and the ray
                // reaches one radius past the extended centre.
                let anchor_local = wm.mount_local.to_dvec3() + DVec3::Y * rest;
                let anchor = state.position + state.rotation * anchor_local;
                // The patch, in the CHASSIS frame rather than the steered wheel's:
                // the corners of a footprint are a property of the wheel's own
                // geometry and the steering angle would rotate them by at most a
                // few degrees at the speeds a kerb is met at. Half a tyre's width
                // across, and a contact patch's half-length along.
                let half_w = wm.radius_m * footprint.half_width_frac;
                let half_l = wm.radius_m * footprint.half_length_frac;
                let max_toi = rest + wm.radius_m;
                [
                    (anchor - right_b * half_w - fwd_b * half_l, max_toi),
                    (anchor + right_b * half_w - fwd_b * half_l, max_toi),
                    (anchor - right_b * half_w + fwd_b * half_l, max_toi),
                    (anchor + right_b * half_w + fwd_b * half_l, max_toi),
                ]
            })
            .collect()
    };
    let hits: Vec<Option<(WheelContact, Uuid)>> = rays
        .iter()
        .map(|corners| {
            let mut point = DVec3::ZERO;
            let mut normal = DVec3::ZERO;
            let mut nearest = f64::INFINITY;
            let mut found = 0usize;
            let mut nearest_guid: Option<Uuid> = None;
            for (origin, max_toi) in corners.iter() {
                let hit = bridge
                    .world_mut()
                    // **Everything solid**, not just static geometry: a car drives
                    // over a crate and up a fractured chunk, and a suspension ray
                    // that skipped dynamic bodies would put the wheel through
                    // them.
                    //
                    // `AllSolid` and not `All` (island wave VEH1a): P29.7 shipped
                    // this with `All`, which includes sensors, and carried the
                    // consequence as a named bound — *"a car crossing a trigger
                    // volume would ride on it"*. A trigger is a description of a
                    // region and exerts no force, so a suspension pushing off one
                    // is a car floating on a checkpoint. Measured before the
                    // filter: `a_wheel_does_not_ride_on_a_trigger_volume`.
                    .cast_ray_where(*origin, -up, *max_toi, &exclude, CastTargets::AllSolid);
                let Some(hit) = hit else { continue };
                found += 1;
                point += hit.point;
                normal += hit.normal;
                if hit.toi < nearest {
                    nearest = hit.toi;
                    nearest_guid = bridge.guid_of_collider(hit.collider);
                }
            }
            if found == 0 {
                return None;
            }
            let n = found as f64;
            // The BILINEAR normal: the mean of what the corners found, normalized.
            // A patch straddling a kerb edge gets the average of the road's normal
            // and the kerb face's, which is the direction a tyre actually pushes
            // against there.
            let blended = (normal / n).normalize_or_zero();
            Some((
                WheelContact {
                    point: point / n,
                    // The wheel rests on the HIGHEST ground under its patch, so
                    // the suspension takes the shortest ray. Averaging the
                    // distances instead would let a wheel sink half-way into a
                    // kerb before the spring noticed it.
                    normal: if blended == DVec3::ZERO { up } else { blended },
                    distance_m: nearest,
                },
                nearest_guid.unwrap_or_else(Uuid::nil),
            ))
        })
        .collect();

    // ── 1b. WHAT EACH WHEEL IS STANDING ON (wave VEH3a) ──────────────────────
    //
    // The lookup happens HERE, once per wheel per step, because this is the only
    // place the hit collider and the world are both in scope. What reaches the
    // model is one number per wheel (`WheelState::mu_surface`), so the tyre never
    // learns what a collider is and `surface_mu` has exactly one caller.
    // The bridge publishes WHAT THE GROUND IS and stops there. What that ground
    // is WORTH is the car's own business — its compound row is a tunable, live
    // on `self.tuning`, and a µ computed out here would silently ignore a tune
    // made through the live-tuning door. So `surface_mu` is called by the model,
    // once, with both halves in scope.
    let surfaces: Vec<SurfaceClass> = hits
        .iter()
        .map(|h| match h {
            Some((c, guid)) => surface_under(world, *guid, c.point),
            None => SurfaceClass::Asphalt,
        })
        .collect();

    // ── 2. the model.
    forces.clear();
    let (grounded, load_n, poses) = {
        let v = bridge.vehicle_mut(chassis)?;
        for ((state, hit), class) in v.wheels_mut().iter_mut().zip(hits).zip(surfaces) {
            state.contact = hit.map(|(c, _)| c);
            state.surface = class;
            state.wetness = wetness;
            state.ambient_c = ambient_c;
        }
        // ── THE INNER LOOP (wave VEH3a clause 5) ────────────────────────
        //
        // The research doc asks for the tyre solve at 300–400 Hz where this
        // engine's stick/slip split is stable at 60 without one. `tyre_substeps`
        // is that N, per class, clamped to `1..=8` by
        // `VehicleTuning::substeps`.
        //
        // **What sub-steps and what does not.** The tyre and the suspension
        // solve N times at `dt / N`, so a wheel's angular velocity, its slip and
        // its temperature are integrated at 240 Hz by default. The CASTS are not
        // repeated: a contact patch moves at most a few centimetres inside one
        // 16.7 ms step, and re-casting would multiply the wave's four rays a
        // wheel by N again — 64 rays a car a step — for a contact that has
        // barely moved. The chassis body is integrated ONCE, by rapier, from the
        // averaged force, which is what keeps the solver's own contract intact.
        //
        // Between sub-steps the chassis state is advanced LOCALLY by the force
        // the previous sub-step produced, so the second sub-step sees the
        // velocity the first one earned rather than the one the step began with.
        //
        // What that is worth was MEASURED, not asserted (VEH3a's audit): over
        // six hundred steps of a full-throttle turn at N = 4 it moves the car
        // **1.894 m** against holding the chassis at the state the fixed step
        // began with. And the thing the wave's prose said — that without it the
        // loop is N identical solves buying nothing — is false: the sub-steps
        // still solve at `dt/4` and average, which is its own trajectory. Both
        // halves are `veh3a_gate::the_substep_loop_runs_and_one_is_what_ships`,
        // through `SubstepAdvance`.
        let substeps = v.substeps().clamp(1, MAX_SUBSTEPS);
        // The measurement door, `Footprint`'s own idiom — see `SubstepAdvance`.
        let advance = world
            .world()
            .get_resource::<SubstepAdvance>()
            .copied()
            .unwrap_or_default();
        if substeps == 1 {
            v.solve(state, dt, forces);
        } else {
            let sub_dt = dt / substeps as f64;
            let inv_mass = if state.mass_kg > 0.0 {
                1.0 / state.mass_kg
            } else {
                0.0
            };
            let mut running = state;
            let mut sub: Vec<WheelForce> = Vec::new();
            let mut sum: Vec<WheelForce> = Vec::new();
            for i in 0..substeps {
                sub.clear();
                v.solve(running, sub_dt, &mut sub);
                if i == 0 {
                    sum = sub.clone();
                } else {
                    // The forces of the i-th sub-step are added at the i-th
                    // sub-step's own contact points; the average below is over
                    // however many each sub-step produced, so a wheel that left
                    // the ground part-way through contributes for the part it was
                    // on it. Lengths agree by construction (one force per
                    // grounded wheel, and the contacts are fixed for the step).
                    for (a, b) in sum.iter_mut().zip(sub.iter()) {
                        a.force += b.force;
                    }
                }
                // Advance the LOCAL chassis by what this sub-step earned. Linear
                // only: the angular half needs the body's inertia tensor, which
                // lives in rapier and is not on this side of the seam, and the
                // yaw a car develops inside 4 ms is small next to the linear
                // velocity change. Stated rather than hidden.
                if i + 1 < substeps && advance == SubstepAdvance::Shipped {
                    let net: DVec3 = sub.iter().map(|f| f.force).sum();
                    running.linvel += net * inv_mass * sub_dt;
                    running.position += running.linvel * sub_dt;
                }
            }
            let inv_n = 1.0 / substeps as f64;
            forces.clear();
            forces.extend(sum.into_iter().map(|f| WheelForce {
                point: f.point,
                force: f.force * inv_n,
            }));
        }
        let grounded = v.wheels().iter().filter(|w| w.contact.is_some()).count();
        let load_n = v.wheels().iter().map(|w| w.load_n).sum::<f64>();
        let poses: Vec<(Uuid, Vec3d, f64, f64, f64)> = v
            .rig()
            .wheels
            .iter()
            .enumerate()
            .filter_map(|(i, wm)| {
                let (length, steer, spin) = v.wheel_pose(i)?;
                Some((wm.guid, wm.mount_local, length, steer, spin))
            })
            .collect();
        (grounded, load_n, poses)
    };

    // ── 3. the forces. The clear first — see this module's ownership note.
    if !bridge.is_buoyant(chassis) {
        bridge.world_mut().reset_forces(body);
    }
    for f in forces.iter() {
        if f.force != DVec3::ZERO {
            bridge
                .world_mut()
                .apply_force_at_point(body, f.force, f.point);
        }
    }

    // ── 3b. the parts, as they are drawn (wave VEH2c). The same visual-only
    //    write the wheels get below, and for the same reason: a spinning rotor
    //    is a rotation on a part's own transform and nothing reads it back.
    let part_poses: Vec<(Uuid, Vec3d)> = {
        let v = bridge.vehicle_of(chassis)?;
        v.rig()
            .parts
            .iter()
            .enumerate()
            .filter_map(|(i, p)| v.part_pose(i).map(|rot| (p.guid, rot)))
            .collect()
    };

    // ── 4. the wheels, as they are drawn. A visual write only: nothing reads
    //    these back (the mount is taken once — see `reconcile_vehicles`), so a
    //    rig with no wheel meshes simulates identically to one with them.
    let rest = bridge.vehicle_of(chassis)?.suspension_rest_m();
    let mut moved = false;
    for (guid, mount_local, length, steer, spin) in poses {
        let Some(entity) = world.entity_of(guid) else {
            continue;
        };
        let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) else {
            continue;
        };
        t.translation = Vec3d::new(
            mount_local.x,
            mount_local.y + (rest - length),
            mount_local.z,
        );
        // Euler YXZ degrees, the `Transform` convention: yaw is the steer and
        // pitch is the roll of the wheel about its own axle.
        t.rotation = Vec3d::new(spin, steer, 0.0);
        moved = true;
    }
    for (guid, rotation) in part_poses {
        let Some(entity) = world.entity_of(guid) else {
            continue;
        };
        let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) else {
            continue;
        };
        // The translation is NOT written: a part does not travel on a
        // suspension, so its authored mount is where it stays and there is
        // nothing for a later reconcile to read back as a moved mount.
        t.rotation = rotation;
        moved = true;
    }
    if moved {
        world.mark_dirty();
    }

    let forward_mps = state.linvel.dot(state.basis().0);
    let (revs, load) = bridge.vehicle_of(chassis)?.engine_state(forward_mps);
    // ── 5. **INPUT IS PER STEP** (wave VEH2c). Every commander — the movement
    //    door, traffic's controller, dispatch's — writes its `VehicleControls`
    //    BEFORE this phase runs, so clearing them after the solve means a
    //    vehicle nobody spoke to this step hears silence rather than whatever
    //    it was last told.
    //
    //    It was not always so, and nothing noticed: a car whose driver got out
    //    kept the last throttle it was given for ever, which is invisible in the
    //    one thing anybody had ever got out of — a car that was already
    //    stopping. It is very visible in a helicopter, whose neutral collective
    //    is a hover (see `VehicleControls::occupied`).
    //
    //    After `engine_state`, which reads the controls this step was solved
    //    with: the sound a vehicle makes is a function of the decision that was
    //    just taken, which is P12's own doctrine.
    if let Some(v) = bridge.vehicle_mut(chassis) {
        v.control(inf_ecs::vehicle::VehicleControls::default());
    }
    Some(VehicleOutcome {
        chassis,
        wheels_grounded: grounded,
        load_n,
        forward_mps,
        revs,
        load,
    })
}

// ── the seat: enter, drive, exit (P29.7) ────────────────────────────────────

/// How close a character's feet must be to a seat to climb into it, metres.
///
/// Generous rather than exact: a door is not a pixel, and a refusal an author
/// cannot see the edge of reads as a broken control. `try_enter` answers the
/// nearest seat inside this radius and refuses everything else as a value.
pub const ENTER_REACH_M: f64 = 3.0;

/// How far to the side of the seat an exiting character is placed, in chassis
/// half-widths plus this, metres.
pub const EXIT_CLEARANCE_M: f64 = 0.6;

/// Where a vehicle's seat is in the world, the chassis rotation it is in, and
/// the chassis's own velocity — the three numbers the seat step needs.
pub fn seat_pose(bridge: &PhysicsBridge3D, chassis: Uuid) -> Option<(DVec3, glam::DQuat, DVec3)> {
    let body = bridge.body_of(chassis)?;
    let v = bridge.vehicle_of(chassis)?;
    let w = bridge.world();
    let pos = w.body_translation(body)?;
    let rot = w.body_rotation(body)?;
    let seat = pos + rot * v.rig().seat_local.to_dvec3();
    Some((seat, rot, w.body_linvel(body).unwrap_or(DVec3::ZERO)))
}

/// **Try to climb into the nearest vehicle** (P29.7).
///
/// Returns the chassis `Guid` if one was in reach. `O(vehicles)`, walked in
/// `Guid` order with a strict `<` on the distance, so a tie between two seats
/// at the same distance is broken by the lower guid rather than by a bevy
/// archetype layout.
///
/// A refusal is a value: no vehicle, none in reach, or one already occupied all
/// answer `None`, and the character does whatever it was going to do instead.
/// **Migrated onto the one interaction door** (island wave I5): this is now
/// `super::interact::nearest_seat`, which builds seat candidates and ranks them
/// with `inf_ecs::interact::resolve` — the same rule an authored
/// `Interactable` is ranked by.
///
/// Every property above is preserved *exactly*: the reach is still
/// [`ENTER_REACH_M`], the walk is still `Guid`-ordered with a strict `<`,
/// occupied seats are still skipped, and there is still **no view test** (the
/// seat candidate carries `NO_VIEW_TEST_DEG`, so the rule's view half is skipped
/// for it). What changed is that there is one implementation of "nearest thing
/// in reach" instead of two, so the day the reach or the tie-break moves, it
/// moves for both.
pub fn try_enter(bridge: &PhysicsBridge3D, feet: DVec3, occupied: &BTreeSet<Uuid>) -> Option<Uuid> {
    super::interact::nearest_seat(bridge, feet, occupied)
}

/// Park (or restore) the character's own collider while it is in a seat.
///
/// The **same door the ragdoll uses**, for the same reason: a character riding
/// on a chassis with its capsule still in the world is a capsule permanently
/// intersecting a dynamic body, which is a depenetration force with nowhere to
/// go. `set_collider_enabled` is a physics-world operation, so both hosts do it
/// identically and PIE parity sees it — which is what the brief asks of this
/// seam.
pub fn park_collider(bridge: &mut PhysicsBridge3D, guid: Uuid, parked: bool) {
    if let Some(c) = bridge.collider_of(guid) {
        bridge.world_mut().set_collider_enabled(c, !parked);
    }
}
