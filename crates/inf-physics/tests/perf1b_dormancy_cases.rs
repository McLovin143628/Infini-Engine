//! **DORMANCY IS INVISIBLE TO EVERY PHYSICS CONSUMER** (the PERF1b audit).
//!
//! `perf1b_dormancy.rs` holds the implementer's two arms (a crate shoved along
//! a street and a ball thrown at a wall; a ray at a sleeping wall). Those
//! movers exist from the first step and arrive at the scenery by moving. The
//! cases below are the ones that reach dormant scenery any OTHER way -- a body
//! that appears, a body that is put somewhere, a static that becomes dynamic,
//! a collider re-described under a resting body, a trigger, a jointed door, a
//! ragdoll, a collapse -- each run twice, the dormancy on and off, from the
//! same script. `off` is the base tree's physics world exactly: the update
//! returns before touching anything, the query doors and `collider_enabled`
//! read no dormancy state, and the hooks only fill bookkeeping nobody reads.
//!
//! Every arm compares EVERY step: each tracked body's translation (within
//! [`TOL_M`]) and the contact-start / contact-stop events by collider pair.
//! Every arm also proves it is not vacuous -- the scenery the case reaches was
//! ASLEEP in the dormant run on the step before the case began.
//!
//! The audit's finding these arms carry: the dormancy measured a mover's reach
//! from its COLLIDERS' poses, and rapier re-poses a body's colliders only
//! inside the pipeline. A body put somewhere through `set_body_translation`
//! (a teleport, a dispatcher placement, every kinematic body the bridge
//! re-poses) was measured where it WAS, so the scenery where it now IS stayed
//! asleep for the first step there. `a_teleported_body_meets_the_scenery_it_lands_in`
//! is the arm that caught it (the dormancy reverted to collider poses: RED).

use glam::{DQuat, DVec3};
use inf_physics::d3::{
    BodyId3D, BodyKind3D, ColliderDesc3D, ColliderId3D, ColliderShape3D, JointDesc3D, JointKind3D,
    PhysicsWorld3D,
};
use inf_physics::ContactPhase;

const DT: f64 = 1.0 / 60.0;

/// Per-step translation tolerance floor, metres. The two runs hold different
/// broad-phase pair sets, so rapier's pair order differs, and a stack in
/// friction amplifies that; each case MEASURES that floor with a control (the
/// awake world with its tiles inserted in reverse) and the tolerance is the
/// larger of this and four times the control's deviation. A woken-too-late
/// static is a millimetre-scale difference on its first step (1.6 mm measured
/// on the teleport defect) -- above every case's tolerance but the stack's.
const TOL_M: f64 = 1e-4;

/// The tail tolerance floor, metres: past the case's first second, where
/// only pair-order bits amplified by friction can separate the runs.
const TAIL_TOL_M: f64 = 1e-3;

/// The step every case begins on; the dormant region has slept since step 1.
const CASE_STEP: usize = 30;

fn static_box(w: &mut PhysicsWorld3D, at: DVec3, half: DVec3) -> (BodyId3D, ColliderId3D) {
    let b = w.add_body(BodyKind3D::Static, at, DQuat::IDENTITY);
    let c = w
        .add_collider(
            b,
            ColliderDesc3D::new(ColliderShape3D::Box { half_extents: half }),
        )
        .expect("a static box attaches");
    (b, c)
}

fn dynamic(w: &mut PhysicsWorld3D, at: DVec3, shape: ColliderShape3D) -> BodyId3D {
    let b = w.add_body(BodyKind3D::Dynamic, at, DQuat::IDENTITY);
    w.add_collider(b, ColliderDesc3D::new(shape))
        .expect("a dynamic body attaches");
    b
}

/// The town: a hero (a kinematic capsule) standing at the origin, which keeps
/// its own neighbourhood awake, and 100 m away a street of overlapping floor
/// tiles (x 100..140) with a building wall across it at x = 125 and a kerb
/// along its +z side -- all of it out of every mover's reach, so asleep.
fn town(dormant: bool, reversed: bool) -> PhysicsWorld3D {
    let mut w = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    w.set_static_dormancy(dormant);
    let hero = w.add_body(
        BodyKind3D::Kinematic,
        DVec3::new(0.0, 1.0, 0.0),
        DQuat::IDENTITY,
    );
    w.add_collider(
        hero,
        ColliderDesc3D::new(ColliderShape3D::Capsule {
            half_height: 0.5,
            radius: 0.3,
        }),
    )
    .expect("the hero attaches");
    for i in -3..=3 {
        static_box(
            &mut w,
            DVec3::new(f64::from(i) * 2.0, -0.5, 0.0),
            DVec3::new(1.05, 0.5, 1.05),
        );
    }
    let mut tiles: Vec<(i32, i32)> = (0..21)
        .flat_map(|i| (-2..=2).map(move |j| (i, j)))
        .collect();
    if reversed {
        tiles.reverse();
    }
    for (i, j) in tiles {
        static_box(
            &mut w,
            DVec3::new(100.0 + f64::from(i) * 2.0, -0.5, f64::from(j) * 2.0),
            DVec3::new(1.05, 0.5, 1.05),
        );
    }
    // The building wall across the street.
    static_box(
        &mut w,
        DVec3::new(125.0, 2.0, 0.0),
        DVec3::new(0.5, 2.0, 5.0),
    );
    // The kerb along the street's +z side.
    static_box(
        &mut w,
        DVec3::new(115.0, 0.1, 2.6),
        DVec3::new(10.0, 0.1, 0.15),
    );
    w
}

/// A case's script: called before every step with the step index, the
/// tracked bodies, and whether this is the reversed-order control run.
type Act<'a> = dyn Fn(&mut PhysicsWorld3D, usize, &mut Vec<BodyId3D>, bool) + 'a;
type ActMut<'a> = dyn FnMut(&mut PhysicsWorld3D, usize, &mut Vec<BodyId3D>, bool) + 'a;

/// One run's record: per step, every tracked body's translation and the
/// contact events by (pair, phase).
type Record = Vec<(Vec<DVec3>, Vec<(ColliderId3D, ColliderId3D, bool, bool)>)>;

/// Run a case `steps` steps with the dormancy `dormant`. `setup` builds extra
/// scenery before the first step; `act` is called before every step with the
/// step index and returns the bodies to track (it may add them on
/// [`CASE_STEP`]). Returns the record and the asleep count on the step before
/// the case began.
fn run(
    dormant: bool,
    reversed: bool,
    steps: usize,
    setup: &dyn Fn(&mut PhysicsWorld3D),
    act: &mut ActMut<'_>,
) -> (Record, usize) {
    let mut w = town(dormant, reversed);
    setup(&mut w);
    let mut tracked: Vec<BodyId3D> = Vec::new();
    let mut rec: Record = Vec::new();
    let mut asleep_before = 0usize;
    for step in 0..steps {
        if step == CASE_STEP {
            asleep_before = w.dormancy_stats().asleep;
        }
        act(&mut w, step, &mut tracked, reversed);
        w.step(DT);
        let poses: Vec<DVec3> = tracked
            .iter()
            .map(|b| w.body_translation(*b).expect("a tracked body"))
            .collect();
        let mut events: Vec<(ColliderId3D, ColliderId3D, bool, bool)> = w
            .drain_contact_events()
            .iter()
            .map(|e| {
                (
                    e.collider_a,
                    e.collider_b,
                    e.phase == ContactPhase::Started,
                    e.sensor,
                )
            })
            .collect();
        events.sort_unstable();
        rec.push((poses, events));
    }
    (rec, asleep_before)
}

/// Compare the two runs step by step; returns (worst deviation, contact
/// starts in the awake run).
fn compare(name: &str, awake: &Record, asleep: &Record, noise: f64) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut starts = 0usize;
    for (step, (a, s)) in awake.iter().zip(asleep).enumerate() {
        assert_eq!(
            a.0.len(),
            s.0.len(),
            "{name}: step {step} tracks different bodies"
        );
        // The first second of the case -- where a static woken late shows,
        // 1.6 mm on its first step -- at the floor; the tail, where a stack in
        // friction amplifies pair-order bits, at the case's measured tolerance.
        let floor = if step < CASE_STEP + 60 {
            TOL_M
        } else {
            TAIL_TOL_M
        };
        let tol = floor.max(4.0 * noise);
        for (k, (pa, ps)) in a.0.iter().zip(&s.0).enumerate() {
            let d = (*pa - *ps).length();
            worst = worst.max(d);
            assert!(
                d <= tol,
                "{name}: step {step}, body {k}: {pa:?} with the statics awake, {ps:?} with them asleep ({d:.6} m)"
            );
        }
        assert_eq!(
            a.1, s.1,
            "{name}: step {step}: the contact events differ (awake left, dormant right)"
        );
        starts += a.1.iter().filter(|e| e.2).count();
    }
    (worst, starts)
}

fn case(
    name: &str,
    steps: usize,
    setup: &dyn Fn(&mut PhysicsWorld3D),
    act: &Act<'_>,
    min_starts: usize,
) {
    let (awake, _) = run(false, false, steps, setup, &mut |w, s, t, r| {
        act(w, s, t, r)
    });
    let (asleep, asleep_before) = run(true, false, steps, setup, &mut |w, s, t, r| act(w, s, t, r));
    // **The ordering-noise floor**, measured rather than assumed: the same
    // awake world with the street's tiles (and, where a case says so, its own
    // bodies) inserted in the reverse order -- no dormancy anywhere, only a
    // different pair order in rapier's graph.
    let (control, _) = run(false, true, steps, setup, &mut |w, s, t, r| act(w, s, t, r));
    let noise = deviation(&awake, &control);
    let tol = TAIL_TOL_M.max(4.0 * noise);
    let (worst, starts) = compare(name, &awake, &asleep, noise);
    println!(
        "{name}: worst deviation {worst:.2e} m over {steps} steps (ordering-noise floor {noise:.2e}, tolerance {tol:.2e}), {starts} contact starts, {asleep_before} statics asleep when it began"
    );
    assert!(
        tol <= 5e-3,
        "{name}: the ordering-noise floor {noise:.2e} m is too loose to test anything"
    );
    assert!(
        asleep_before >= 50,
        "{name}: only {asleep_before} statics were asleep when the case began -- the arm is vacuous"
    );
    assert!(
        starts >= min_starts,
        "{name}: {starts} contact starts -- the case never reached the scenery"
    );
}

/// The worst per-step translation difference between two runs.
fn deviation(a: &Record, b: &Record) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.0.iter().zip(&y.0).map(|(p, q)| (*p - *q).length()))
        .fold(0.0, f64::max)
}

/// **A BODY SPAWNED, NOT MOVED, INSIDE THE DORMANT STREET** -- a dropped
/// weapon, a shed bumper: on its first step it meets the floor it was put on,
/// and a second body spawned moving at 12 m/s meets the wall.
#[test]
fn a_body_spawned_in_the_dormant_street_meets_it_on_its_first_step() {
    case(
        "spawned",
        120,
        &|_| {},
        &|w, step, tracked, _rev| {
            if step == CASE_STEP {
                tracked.push(dynamic(
                    w,
                    DVec3::new(110.0, 0.4005, 0.0),
                    ColliderShape3D::Box {
                        half_extents: DVec3::splat(0.4),
                    },
                ));
                let ball = dynamic(
                    w,
                    DVec3::new(123.9, 0.3005, 0.0),
                    ColliderShape3D::Sphere { radius: 0.3 },
                );
                w.set_body_linvel(ball, DVec3::new(12.0, 0.0, 0.0));
                tracked.push(ball);
            }
        },
        3,
    );
}

/// **A BODY PUT SOMEWHERE** -- the dispatcher's teleport, a debug placement,
/// a car the parking hold re-poses: a ball resting by the hero is moved into
/// the dormant street right against the wall, moving at it at 20 m/s. The
/// floor and the wall where it lands must meet it on the first step there.
#[test]
fn a_teleported_body_meets_the_scenery_it_lands_in() {
    case(
        "teleported",
        120,
        &|_| {},
        &|w, step, tracked, _rev| {
            if step == 0 {
                tracked.push(dynamic(
                    w,
                    DVec3::new(3.0, 0.3005, 0.0),
                    ColliderShape3D::Sphere { radius: 0.3 },
                ));
            }
            if step == CASE_STEP {
                let b = tracked[0];
                w.set_body_translation(b, DVec3::new(124.05, 0.3002, 0.0));
                w.set_body_linvel(b, DVec3::new(20.0, 0.0, 0.0));
            }
        },
        3,
    );
}

/// **A KINEMATIC BODY RE-POSED EVERY STEP** -- the bridge pushes every
/// kinematic pose through `set_body_translation`: a kinematic plough driven
/// at 75 m/s (1.25 m a step) down the dormant street into a crate resting
/// against the wall. The crate it shoves must meet the wall as it would.
#[test]
fn a_kinematic_body_reposed_fast_wakes_what_it_reaches() {
    case(
        "kinematic",
        120,
        &|_| {},
        &|w, step, tracked, _rev| {
            if step == CASE_STEP {
                let plough = w.add_body(
                    BodyKind3D::Kinematic,
                    DVec3::new(106.0, 0.6, 0.0),
                    DQuat::IDENTITY,
                );
                w.add_collider(
                    plough,
                    ColliderDesc3D::new(ColliderShape3D::Box {
                        half_extents: DVec3::new(0.3, 0.5, 0.8),
                    }),
                )
                .expect("the plough attaches");
                tracked.push(plough);
                tracked.push(dynamic(
                    w,
                    DVec3::new(122.0, 0.4005, 0.0),
                    ColliderShape3D::Box {
                        half_extents: DVec3::splat(0.4),
                    },
                ));
            }
            if step > CASE_STEP && step <= CASE_STEP + 14 {
                let p = w.body_translation(tracked[0]).expect("the plough");
                w.set_body_translation(tracked[0], p + DVec3::new(1.25, 0.0, 0.0));
            }
        },
        2,
    );
}

/// **A STATIC THAT BECOMES DYNAMIC, AND A COLLAPSE** -- P22's fracture swap
/// and structural collapse change a fixed body's kind in place. A column of
/// three fixed blocks stands in the dormant street beside a fourth; the top
/// block is released at the case step and the middle one fifteen steps later.
/// They fall, strike the fourth block, each other and the floor -- all asleep
/// until then -- and their masses (which rapier computes from ENABLED
/// colliders) decide the dynamic-dynamic contacts.
#[test]
fn a_static_released_into_a_dynamic_falls_and_collides_as_before() {
    let blocks = |w: &mut PhysicsWorld3D, rev: bool| -> Vec<BodyId3D> {
        let mut order = vec![0usize, 1, 2, 3];
        if rev {
            order.reverse();
        }
        let mut out = vec![None; 4];
        for k in order {
            let at = if k == 3 {
                DVec3::new(113.2, 0.5, 0.0)
            } else {
                DVec3::new(112.0, 0.5 + k as f64, 0.0)
            };
            let (b, _) = static_box(w, at, DVec3::new(0.5, 0.5, 0.5));
            out[k] = Some(b);
        }
        out.into_iter()
            .take(3)
            .map(|b| b.expect("a block"))
            .collect()
    };
    case(
        "released",
        180,
        &|_| {},
        &|w, step, tracked, rev| {
            if step == 0 {
                let b = blocks(w, rev);
                tracked.extend(b);
            }
            if step == CASE_STEP {
                w.set_body_kind(tracked[2], BodyKind3D::Dynamic);
                w.set_body_angvel(tracked[2], DVec3::new(0.0, 0.0, -2.0));
                w.set_body_linvel(tracked[2], DVec3::new(1.5, 0.0, 0.0));
            }
            if step == CASE_STEP + 15 {
                w.set_body_kind(tracked[1], BodyKind3D::Dynamic);
                w.set_body_linvel(tracked[1], DVec3::new(-1.0, 0.0, 0.0));
            }
        },
        2,
    );
}

/// **A COLLIDER RE-DESCRIBED UNDER A RESTING BODY** -- a voxel carve or a
/// terrain tile re-cut replaces a static collider in place (the bridge's
/// `col_changed` path: remove, attach). A crate rests on a slab in the dormant
/// street; the slab is replaced by a lower one, and a dormant slab across the
/// street is replaced too with nothing near it. The crate falls onto the new
/// slab exactly as it would.
#[test]
fn a_collider_replaced_under_a_resting_body_is_met_as_before() {
    case(
        "replaced",
        150,
        &|_| {},
        &|w, step, tracked, _rev| {
            if step == 0 {
                let (slab, col) =
                    static_box(w, DVec3::new(108.0, 0.6, 0.0), DVec3::new(0.8, 0.1, 0.8));
                let (far, far_col) =
                    static_box(w, DVec3::new(108.0, 0.6, -3.0), DVec3::new(0.8, 0.1, 0.8));
                tracked.push(dynamic(
                    w,
                    DVec3::new(108.0, 1.1005, 0.0),
                    ColliderShape3D::Box {
                        half_extents: DVec3::splat(0.4),
                    },
                ));
                // Stash the slab handles in the world's own order: the case
                // closure is called identically in both runs, so the handles
                // are the same, and they are recovered below by pose.
                let _ = (slab, col, far, far_col);
            }
            if step == CASE_STEP + 10 {
                // The slab under the crate and the far one, found by their
                // place, re-described lower: the crate's support drops 0.3 m.
                let hits =
                    w.intersect_aabb(DVec3::new(107.5, 0.55, -3.5), DVec3::new(108.5, 0.65, 0.5));
                let mut slabs: Vec<ColliderId3D> = hits
                    .into_iter()
                    .filter(|c| w.collider_parent(*c).is_some_and(|b| !tracked.contains(&b)))
                    .collect();
                slabs.sort();
                assert_eq!(slabs.len(), 2, "both slabs are found");
                for c in slabs {
                    let body = w.collider_parent(c).expect("a slab's body");
                    w.remove_collider(c);
                    w.add_collider(
                        body,
                        ColliderDesc3D::new(ColliderShape3D::Box {
                            half_extents: DVec3::new(0.8, 0.1, 0.8),
                        })
                        .local_translation(DVec3::new(0.0, -0.3, 0.0)),
                    )
                    .expect("the re-cut slab attaches");
                }
            }
        },
        2,
    );
}

/// **A TRIGGER ON DORMANT SCENERY** -- a venue door's volume, an on-scene
/// zone: a fixed sensor in the dormant street over the floor, and a ball
/// rolled through it. The sensor's enter / leave events and the ball's path
/// match.
#[test]
fn a_trigger_in_the_dormant_street_reports_as_before() {
    case(
        "trigger",
        150,
        &|w| {
            let b = w.add_body(
                BodyKind3D::Static,
                DVec3::new(116.0, 1.0, 0.0),
                DQuat::IDENTITY,
            );
            w.add_collider(
                b,
                ColliderDesc3D::new(ColliderShape3D::Box {
                    half_extents: DVec3::new(1.0, 1.0, 1.0),
                })
                .sensor(true),
            )
            .expect("the trigger attaches");
        },
        &|w, step, tracked, _rev| {
            if step == CASE_STEP {
                let ball = dynamic(
                    w,
                    DVec3::new(110.0, 0.3005, 0.0),
                    ColliderShape3D::Sphere { radius: 0.3 },
                );
                w.set_body_linvel(ball, DVec3::new(6.0, 0.0, 0.0));
                tracked.push(ball);
            }
        },
        2,
    );
}

/// **A DOOR ON ITS HINGE BESIDE A DORMANT KERB** -- VEH3d's boarding door:
/// a parked car body (dynamic, resting) with a door leaf on a revolute joint,
/// swung open hard into the kerb.
#[test]
fn a_jointed_door_swings_into_a_dormant_kerb_as_before() {
    case(
        "door",
        150,
        &|_| {},
        &|w, step, tracked, rev| {
            if step == CASE_STEP {
                let make_car = |w: &mut PhysicsWorld3D| {
                    dynamic(
                        w,
                        DVec3::new(115.0, 0.6005, 1.0),
                        ColliderShape3D::Box {
                            half_extents: DVec3::new(2.0, 0.6, 0.9),
                        },
                    )
                };
                let make_door = |w: &mut PhysicsWorld3D| {
                    dynamic(
                        w,
                        DVec3::new(115.5, 0.7, 1.95),
                        ColliderShape3D::Box {
                            half_extents: DVec3::new(0.5, 0.4, 0.04),
                        },
                    )
                };
                let (car, door) = if rev {
                    let d = make_door(w);
                    (make_car(w), d)
                } else {
                    let c = make_car(w);
                    (c, make_door(w))
                };
                w.add_joint(
                    car,
                    door,
                    JointDesc3D::new(JointKind3D::Revolute {
                        axis: DVec3::Y,
                        limits: None,
                        motor: None,
                    })
                    .local_anchor1(DVec3::new(-0.0, 0.1, 0.95))
                    .local_anchor2(DVec3::new(-0.5, 0.0, 0.0))
                    .without_contacts(),
                )
                .expect("the hinge attaches");
                w.set_body_angvel(door, DVec3::new(0.0, -6.0, 0.0));
                tracked.push(car);
                tracked.push(door);
            }
        },
        2,
    );
}

/// **A RAGDOLL THROWN INTO A DORMANT BUILDING** -- five capsules on
/// spherical joints, appearing at 25 m/s three metres from the wall.
#[test]
fn a_ragdoll_thrown_into_a_dormant_wall_lands_as_before() {
    case(
        "ragdoll",
        150,
        &|_| {},
        &|w, step, tracked, _rev| {
            if step == CASE_STEP {
                let mut prev: Option<BodyId3D> = None;
                for k in 0..5 {
                    let b = dynamic(
                        w,
                        DVec3::new(121.0, 0.4 + f64::from(k) * 0.42, 0.0),
                        ColliderShape3D::Capsule {
                            half_height: 0.1,
                            radius: 0.1,
                        },
                    );
                    w.set_body_linvel(b, DVec3::new(25.0, 2.0, 0.0));
                    if let Some(p) = prev {
                        w.add_joint(
                            p,
                            b,
                            JointDesc3D::new(JointKind3D::Spherical)
                                .local_anchor1(DVec3::new(0.0, 0.21, 0.0))
                                .local_anchor2(DVec3::new(0.0, -0.21, 0.0))
                                .without_contacts(),
                        )
                        .expect("a ragdoll joint attaches");
                    }
                    prev = Some(b);
                    tracked.push(b);
                }
            }
        },
        3,
    );
}

/// **A FAST BODY WITH CCD** -- 240 m/s (4 m a step) at the wall from 30 m
/// out, continuous collision on. Its reach is two steps of travel, so the
/// wall is awake before the sweep that meets it.
#[test]
fn a_fast_ccd_body_is_stopped_by_a_dormant_wall_as_before() {
    case(
        "ccd",
        90,
        &|_| {},
        &|w, step, tracked, _rev| {
            if step == CASE_STEP {
                let b = dynamic(
                    w,
                    DVec3::new(94.0, 1.5, 0.0),
                    ColliderShape3D::Sphere { radius: 0.1 },
                );
                w.set_body_ccd(b, true);
                w.set_body_linvel(b, DVec3::new(240.0, 0.0, 0.0));
                tracked.push(b);
            }
        },
        1,
    );
}
