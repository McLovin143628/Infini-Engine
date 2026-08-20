//! **What a fixed step over a CITY was paying for** (island wave I4b), and the
//! two rules that stopped it paying.
//!
//! Wave I4's fps instrument found the frame CPU-bound with the fixed step its
//! dearest part — 13.0–14.9 ms over the phase-30 city — and could not say where
//! the milliseconds went. I4b's phase clock named them, and two of the names
//! were surprises:
//!
//! * **72.9 % was `bridge3d.step`** — rapier itself, over a world whose only
//!   moving thing is one character. Cause: every collider asked for
//!   `ActiveCollisionTypes::all()`, which includes `FIXED_FIXED`, so every
//!   banded building box resting on a streamed terrain **heightfield** had a
//!   contact manifold computed and re-computed at 60 Hz for a pair no solver
//!   can ever act on.
//! * **17.6 % was the P29.6 locomotion camera** — one sphere sweep. Cause: the
//!   query BVH was rebuilt from scratch on the first query after every step,
//!   because `step` declared the whole tree stale unconditionally.
//!
//! This file arms both rules. The **shapes and counts** are asserted everywhere,
//! because a contact pair is a contact pair on every machine; the **clocks** are
//! printed, per the standing rule.

use glam::{DQuat, DVec3};
use inf_physics::d3::{BodyKind3D, ColliderDesc3D, ColliderShape3D, PhysicsWorld3D};

const DT: f64 = 1.0 / 60.0;

fn world() -> PhysicsWorld3D {
    PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0))
}

fn box_at(w: &mut PhysicsWorld3D, kind: BodyKind3D, at: DVec3, half: f64, sensor: bool) {
    let b = w.add_body(kind, at, DQuat::IDENTITY);
    let mut desc = ColliderDesc3D::new(ColliderShape3D::Box {
        half_extents: DVec3::splat(half),
    });
    desc.sensor = sensor;
    w.add_collider(b, desc).expect("the collider attaches");
}

// ── rule 1: a solid fixed ↔ fixed pair is not a manifold ─────────────────────

/// **Two static solids that overlap cost the narrow phase nothing.**
///
/// The pair is still *tracked* — the broad phase found it, and dropping it there
/// would need collision groups the author owns — but no manifold is computed,
/// which is what `touching` counts. The **control** in the same arm makes one of
/// them dynamic and the same overlap becomes a real contact, so this cannot pass
/// by the two boxes simply not overlapping.
#[test]
fn two_static_solids_that_overlap_compute_no_manifold() {
    let mut w = world();
    box_at(&mut w, BodyKind3D::Static, DVec3::ZERO, 1.0, false);
    box_at(
        &mut w,
        BodyKind3D::Static,
        DVec3::new(0.5, 0.0, 0.0),
        1.0,
        false,
    );
    for _ in 0..4 {
        w.step(DT);
    }
    let (tracked, touching) = w.contact_pair_counts();
    println!("static solid ↔ static solid: {tracked} tracked, {touching} touching");
    assert_eq!(tracked, 1, "the broad phase must still find the pair");
    assert_eq!(
        touching, 0,
        "a fixed ↔ fixed SOLID pair computed a manifold — `FIXED_FIXED` is back \
         on for solids, and the city's fixed step is paying 9 ms for manifolds \
         no solver can act on"
    );

    // The control: the same overlap, one body dynamic.
    let mut c = world();
    box_at(&mut c, BodyKind3D::Static, DVec3::ZERO, 1.0, false);
    box_at(
        &mut c,
        BodyKind3D::Dynamic,
        DVec3::new(0.5, 0.0, 0.0),
        1.0,
        false,
    );
    for _ in 0..4 {
        c.step(DT);
    }
    let (ctracked, ctouching) = c.contact_pair_counts();
    println!("static solid ↔ DYNAMIC solid: {ctracked} tracked, {ctouching} touching");
    assert_eq!(
        ctouching, 1,
        "the control's dynamic box computed no manifold either — this arm is \
         measuring two boxes that do not overlap, not a collision-type rule"
    );
}

/// **…and a static SENSOR over static scenery still reports**, which is the case
/// the engine widened the flags for in the first place.
///
/// `ActiveCollisionTypes` is tested as the union of the pair's two colliders, so
/// the sensor's own `all()` carries the pair across the line the solid half no
/// longer crosses. A trigger volume an author places over a static wall fires
/// exactly as it did.
#[test]
fn a_static_sensor_over_static_scenery_still_reports_its_overlap() {
    let mut w = world();
    box_at(&mut w, BodyKind3D::Static, DVec3::ZERO, 1.0, false);
    box_at(
        &mut w,
        BodyKind3D::Static,
        DVec3::new(0.5, 0.0, 0.0),
        1.0,
        true,
    );
    let mut sensor_events = 0usize;
    for _ in 0..4 {
        w.step(DT);
        sensor_events += w
            .drain_contact_events()
            .into_iter()
            .filter(|e| e.sensor)
            .count();
    }
    println!("static sensor ↔ static solid: {sensor_events} sensor events");
    assert!(
        sensor_events > 0,
        "a static trigger volume over static scenery reported nothing — the \
         `FIXED_FIXED` narrowing took the sensor case with it, which is the one \
         case the engine widened the flags for"
    );
}

// ── rule 2: the query tree is maintained, not rebuilt ────────────────────────

/// A field of static boxes plus one falling ball — small enough to be a unit
/// test, shaped like the city.
fn field(boxes: i32) -> (PhysicsWorld3D, inf_physics::d3::BodyId3D) {
    let mut w = world();
    for i in 0..boxes {
        for j in 0..boxes {
            box_at(
                &mut w,
                BodyKind3D::Static,
                DVec3::new(f64::from(i) * 3.0, 0.0, f64::from(j) * 3.0),
                1.0,
                false,
            );
        }
    }
    let b = w.add_body(
        BodyKind3D::Dynamic,
        DVec3::new(4.5, 12.0, 4.5),
        DQuat::IDENTITY,
    );
    w.add_collider(
        b,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.5 }),
    )
    .expect("the ball's collider attaches");
    (w, b)
}

/// **THE EQUIVALENCE GATE.** A query tree maintained incrementally and one built
/// from scratch answer the same question the same way.
///
/// Two worlds, stepped identically: one whose tree is only ever *maintained*, and
/// a control whose tree is thrown away and rebuilt before every question. At each
/// of 180 steps both are asked the same four questions, of four different kinds,
/// at the moving body. If the incremental path ever misses a leaf that moved, or
/// holds one at a stale AABB, the two answers diverge on the step it happened.
///
/// # Two worlds, because one world could not accumulate staleness (the I4b audit)
///
/// The first cut of this gate stepped **one** world and called
/// [`force_query_rebuild`](PhysicsWorld3D::force_query_rebuild) between the two
/// halves of every iteration — which means the "incremental" tree it questioned
/// at step *n* was the tree the control rebuilt at step *n − 1*, i.e. **exactly
/// one step stale**. One step of a falling ball is centimetres and the leaf is
/// the ball's own half-metre AABB, so the two halves agreed by construction.
/// Measured: deleting the marking **entirely** — both
/// `query_moved_bodies.extend(islands.active_bodies())` calls removed from
/// `step` — left that gate green over all 540 answers. Here the incremental world
/// is never rebuilt, so 180 steps of drift accumulate in it, and the same
/// mutation dies.
///
/// # …and a point query, because a long ray cannot see a stale leaf either
///
/// The BVH's leaf AABB only decides what the narrow phase is *offered*; the
/// narrow phase then reads the collider's real pose. So a 40 m ray cast from 6 m
/// above traverses a leaf left at the body's spawn just as happily as a fresh
/// one, and answers correctly. `intersect_point` at the body's own centre is the
/// question that reaches the leaf **where the leaf says the body is**, and it is
/// what makes the other three mean anything.
///
/// # What it does NOT arm, measured
///
/// The two `active_bodies()` extends are armed as a **pair**, not individually:
/// deleting either one alone leaves this gate green, because a body that is awake
/// on both sides of a step is marked by whichever half survives, and the AABB is
/// recomputed lazily at query time from the collider's current pose. Each half
/// earns its place on a case the other misses — the before-half a body that fell
/// asleep *during* the step (rapier sleeps at the end of one, so the after-half
/// no longer names it) and the after-half a body woken *by* the step — and
/// neither case is expressible without reaching into rapier's sleep state, which
/// this facade does not expose. Written down rather than implied.
#[test]
fn the_incremental_query_tree_answers_what_a_rebuilt_one_does() {
    let (mut inc, ball) = field(8);
    let (mut reb, control_ball) = field(8);
    let mut compared = 0usize;
    let mut hits = 0usize;
    for step in 0..180 {
        inc.step(DT);
        reb.step(DT);
        let p = inc.body_translation(ball).expect("the ball exists");
        let q = reb
            .body_translation(control_ball)
            .expect("the control's ball exists");
        assert_eq!(
            p, q,
            "step {step}: the two worlds stopped being the same simulation, so \
             comparing their query answers compares two different scenes"
        );
        // The control's tree is thrown away before every question; the
        // incremental one is never rebuilt for the whole run.
        reb.force_query_rebuild();
        let exclude: std::collections::BTreeSet<inf_physics::d3::ColliderId3D> =
            std::collections::BTreeSet::new();
        let sphere = ColliderShape3D::Sphere { radius: 0.4 };
        let ray = inc.cast_ray(p + DVec3::Y * 6.0, -DVec3::Y, 40.0);
        let sweep = inc.cast_shape(
            &sphere,
            p + DVec3::Y * 6.0,
            DQuat::IDENTITY,
            -DVec3::Y,
            40.0,
            &exclude,
        );
        let aabb = inc.intersect_aabb(p - DVec3::splat(2.0), p + DVec3::splat(2.0));
        let at = inc.intersect_point(p);
        let ray2 = reb.cast_ray(q + DVec3::Y * 6.0, -DVec3::Y, 40.0);
        let sweep2 = reb.cast_shape(
            &sphere,
            q + DVec3::Y * 6.0,
            DQuat::IDENTITY,
            -DVec3::Y,
            40.0,
            &exclude,
        );
        let aabb2 = reb.intersect_aabb(q - DVec3::splat(2.0), q + DVec3::splat(2.0));
        let at2 = reb.intersect_point(q);
        compared += 4;
        hits += usize::from(ray.is_some()) + usize::from(sweep.is_some()) + at.len();
        assert_eq!(
            at, at2,
            "step {step}: a POINT query at the moving body's own centre answered \
             differently from the rebuilt control — the incremental tree still \
             has the body where it used to be"
        );
        assert_eq!(
            ray.map(|h| (h.collider, h.toi.to_bits())),
            ray2.map(|h| (h.collider, h.toi.to_bits())),
            "step {step}: the ray answered differently from the rebuilt control \
             — the incremental tree has drifted from the colliders"
        );
        assert_eq!(
            sweep.map(|h| (h.collider, h.toi.to_bits())),
            sweep2.map(|h| (h.collider, h.toi.to_bits())),
            "step {step}: the sweep answered differently from the rebuilt control"
        );
        assert_eq!(
            aabb, aabb2,
            "step {step}: the AABB query answered differently from the rebuilt \
             control"
        );
    }
    println!(
        "{compared} query answers compared across 180 steps, {hits} of them hits \
         — a never-rebuilt incremental tree and a rebuilt-every-step control \
         agree on every one"
    );
    // Anti-vacuity: a run where nothing ever hit anything would compare `None`
    // against `None` 720 times and prove nothing.
    assert!(
        hits > 180,
        "only {hits} of the queries hit anything — this gate is comparing \
         emptiness with emptiness"
    );
}

/// **A BODY THE CALLER TELEPORTS ANSWERS WHERE IT LANDED, NOT WHERE IT WAS**
/// (the I4b audit).
///
/// `set_body_translation` / `set_body_rotation` / `set_body_kind` mark their body
/// stale because the solver never sees the move — and the shipped hosts' one
/// per-step path for a **kinematic or static** body's pose,
/// `set_body_pose_if_moved`, goes straight through the first two. A moving
/// platform whose query leaf is never refreshed is a platform the character mover
/// and the P29.6 camera sweep collide with where it *used to be*.
///
/// Measured blind before this arm existed: deleting
/// `query_moved_bodies.push(body.0)` from `set_body_translation` left **every
/// test in this crate green**. The equivalence gate above could not see it
/// either — it moves its body with the *solver*.
///
/// # A FIXED body, and a `step` between the write and the query
///
/// Both halves of that shape are load-bearing and neither is a preference:
///
/// * **Fixed**, because a *kinematic* body is in `islands.active_bodies()` the
///   moment it is touched, so the step's own union marks it and the explicit
///   push is redundant. `set_body_pose_if_moved`'s other caller — a static
///   collider an author, a gizmo or a Blueprint moved — is the case only the
///   explicit push covers.
/// * **A step between**, because rapier propagates a body's pose onto its
///   colliders inside `PhysicsPipeline::step`; `Collider::compute_aabb` reads
///   the collider's own position, so between a pose write and the next step the
///   query pipeline answers at the OLD pose whatever the tree does. That is
///   true of the from-scratch rebuild this incremental path replaced, word for
///   word, and it is a property of the dependency rather than of this wave —
///   written down here because an arm that queried without stepping would fail
///   on both, and read as this wave's regression.
#[test]
fn a_teleported_body_answers_where_it_landed_and_not_where_it_was() {
    let mut w = world();
    let b = w.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    w.add_collider(
        b,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(1.0),
        }),
    )
    .expect("the platform's collider attaches");
    w.step(DT);
    assert_eq!(
        w.intersect_point(DVec3::ZERO).len(),
        1,
        "the platform answers where it was built"
    );

    // A teleport far past its own AABB — the case a refit cannot paper over.
    assert!(w.set_body_translation(b, DVec3::new(50.0, 0.0, 0.0)));
    w.step(DT);
    let there = w.intersect_point(DVec3::new(50.0, 0.0, 0.0));
    let back = w.intersect_point(DVec3::ZERO);
    println!(
        "after a 50 m teleport: {} there, {} behind",
        there.len(),
        back.len()
    );
    assert_eq!(
        there.len(),
        1,
        "a body teleported 50 m does not answer at its new position — its query \
         leaf was never marked stale, so the mover and the camera sweep still see \
         it where it was"
    );
    assert!(
        back.is_empty(),
        "the teleported body still answers at its OLD position — the leaf held \
         both places at once"
    );

    // …and a rotation, which moves the AABB of a long box without moving its
    // centre at all — `set_body_rotation` is the second half of the bridge's
    // one pose-write door.
    let mut r = world();
    let rb = r.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    r.add_collider(
        rb,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(8.0, 0.5, 0.5),
        }),
    )
    .expect("the beam's collider attaches");
    r.step(DT);
    let probe = DVec3::new(0.0, 0.0, 6.0);
    assert!(
        r.intersect_point(probe).is_empty(),
        "the beam must not reach the probe before it turns"
    );
    assert!(r.set_body_rotation(rb, DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2)));
    r.step(DT);
    let turned = r.intersect_point(probe);
    println!("after a quarter turn: {} at the probe", turned.len());
    assert_eq!(
        turned.len(),
        1,
        "a rotated beam does not answer along its new axis — `set_body_rotation` \
         did not mark its colliders' AABBs stale"
    );
}

/// **A REMOVED BODY LEAVES THE QUERY TREE**, and not only a removed *collider*.
///
/// `remove_body` takes its colliders with it, so it has to force the same fresh
/// build `remove_collider` does. This is the path a despawn takes: a streamed-out
/// partition cell, a `Destroyed` actor, a Blueprint despawn — and the sequence
/// here is the awkward one, **moved and then removed**, which leaves a stale mark
/// behind a dead handle that the next query has to drop rather than trip over.
///
/// Like its collider twin above, this arm asserts the *behaviour* and not the
/// flag: deleting `query_rebuild = true` from `remove_body` leaves it green, for
/// the reason `PhysicsWorld3D::query_rebuild`'s doc gives. What it would catch is
/// the incremental path panicking or answering on a dead handle, which is the
/// failure mode a `Vec` of handles surviving its arena invites.
#[test]
fn a_removed_body_leaves_the_query_tree() {
    let mut w = world();
    box_at(
        &mut w,
        BodyKind3D::Static,
        DVec3::new(9.0, 0.0, 0.0),
        1.0,
        false,
    );
    let victim = w.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    w.add_collider(
        victim,
        ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(1.0),
        }),
    )
    .expect("the victim's collider attaches");
    w.step(DT);
    assert_eq!(
        w.intersect_point(DVec3::ZERO).len(),
        1,
        "the body answers before it is removed"
    );

    // **Moved, THEN removed** — the sequence that leaves a stale mark behind a
    // dead handle, which the rebuild has to drop rather than trip over.
    assert!(w.set_body_translation(victim, DVec3::new(0.0, 0.0, 4.0)));
    assert!(w.remove_body(victim), "the body removes");
    let (bodies, colliders) = w.pending_query_marks();
    let gone = w.intersect_point(DVec3::new(0.0, 0.0, 4.0));
    let old = w.intersect_point(DVec3::ZERO);
    println!(
        "after moving then removing: {} at the new place, {} at the old, marks \
         pending before the query were ({bodies}, {colliders})",
        gone.len(),
        old.len()
    );
    assert!(
        gone.is_empty() && old.is_empty(),
        "a removed BODY still answers a point query — the removal did not force \
         a rebuild, and `BroadPhaseBvh` cannot drop a leaf any other way"
    );
    // The control: the rest of the world is untouched by the rebuild.
    assert_eq!(
        w.intersect_point(DVec3::new(9.0, 0.0, 0.0)).len(),
        1,
        "the rebuild dropped a body it was not asked to drop"
    );
}

/// **A removed collider stops answering**, and the rebuild it forces is correct.
///
/// `BroadPhaseBvh` has no removal, so a leaf can only leave by a fresh build —
/// which is a property of the dependency and therefore a property this file
/// states out loud.
///
/// **What this arm does NOT say** (the I4b audit): it does not say the
/// `query_rebuild` flag is load-bearing, and the first write-up's claim that
/// "without it the tree would answer with a collider that no longer exists" is
/// wrong. Mutation-measured: deleting `query_rebuild = true` from
/// `remove_collider` leaves this arm — and every other test in this crate —
/// green, because the query pipeline resolves a leaf through
/// `ColliderSet::get_unknown_gen` and a dead index yields nothing. The flag's job
/// is the "one leaf per live collider" invariant, which this type exposes no way
/// to observe. See `PhysicsWorld3D::query_rebuild`'s own doc.
#[test]
fn a_removed_collider_leaves_the_query_tree() {
    let mut w = world();
    box_at(&mut w, BodyKind3D::Static, DVec3::ZERO, 1.0, false);
    w.step(DT);
    let before = w.intersect_point(DVec3::ZERO);
    assert_eq!(before.len(), 1, "the box answers before it is removed");
    let victim = before[0];
    assert!(w.remove_collider(victim), "the collider removes");
    w.step(DT);
    let after = w.intersect_point(DVec3::ZERO);
    println!("after removal: {} colliders at the origin", after.len());
    assert!(
        after.is_empty(),
        "a removed collider still answers a point query — the removal did not \
         force a rebuild, and `BroadPhaseBvh` cannot drop a leaf any other way"
    );
}

/// **A WORLD THAT NEVER QUERIES DOES NOT GROW A LIST FOREVER.**
///
/// The incremental query tree's pending marks are drained by the next *query*,
/// and a level may never make one: a physics playground with falling props, no
/// character, no camera subject and no gameplay cast steps at 60 Hz and asks
/// nothing. Left as a plain accumulator that is "a pin with no release is a leak
/// with a deadline" one phase on, so `step` deduplicates the body list and drops
/// the collider list once it is longer than the world's own collider count —
/// past which a fresh build is cheaper than re-inserting anyway.
///
/// This arm steps such a world 600 times and reads the lists' own bound, then
/// **checks the world still answers correctly afterwards**, because a bound that
/// works by forgetting is only a bound if what it forgot was recoverable.
#[test]
fn a_world_that_never_queries_does_not_accumulate_pending_marks() {
    let (mut w, ball) = field(4);
    for _ in 0..600 {
        w.step(DT);
    }
    let (bodies, colliders) = w.pending_query_marks();
    println!(
        "600 steps with no query: {bodies} pending bodies, {colliders} pending \
         colliders, over a world of {} colliders",
        4 * 4 + 1
    );
    assert!(
        bodies <= 4 * 4 + 1,
        "{bodies} pending body marks after 600 steps of a {} body world — the \
         list is accumulating per step rather than per distinct body",
        4 * 4 + 1
    );
    assert!(
        colliders <= 4 * 4 + 1,
        "{colliders} pending collider marks after 600 steps — the ceiling that \
         converts a long pending list into a rebuild is not firing"
    );
    // …and the world is still right. The first query after all that drains
    // whatever is pending, and it has to answer what a rebuilt tree would.
    let p = w.body_translation(ball).expect("the ball exists");
    let ray = w.cast_ray(p + DVec3::Y * 6.0, -DVec3::Y, 40.0);
    w.force_query_rebuild();
    let ray2 = w.cast_ray(p + DVec3::Y * 6.0, -DVec3::Y, 40.0);
    assert_eq!(
        ray.map(|h| (h.collider, h.toi.to_bits())),
        ray2.map(|h| (h.collider, h.toi.to_bits())),
        "after 600 unqueried steps the incremental tree disagrees with a rebuilt \
         one — the bound dropped a mark it needed"
    );
    assert!(ray.is_some(), "the ray must hit the floor it is aimed at");
}

/// **A collider attached between two steps is queryable immediately.**
///
/// The reason the query tree is a second structure at all rather than the
/// pipeline's own broad phase: the fixed step attaches colliders (the collider
/// band admits a building) and then queries (the character moves, the camera
/// sweeps) *before* the next `step` propagates anything. A tree that only
/// learned about geometry at step time would let a character walk through a
/// building the band had just admitted, for one step.
#[test]
fn a_collider_attached_between_steps_answers_the_next_query() {
    let mut w = world();
    box_at(&mut w, BodyKind3D::Static, DVec3::ZERO, 1.0, false);
    w.step(DT);
    assert_eq!(w.intersect_point(DVec3::new(9.0, 0.0, 0.0)).len(), 0);
    box_at(
        &mut w,
        BodyKind3D::Static,
        DVec3::new(9.0, 0.0, 0.0),
        1.0,
        false,
    );
    // No step in between — this is the fixed step's own order.
    let found = w.intersect_point(DVec3::new(9.0, 0.0, 0.0));
    println!(
        "attached and queried with no step between: {} found",
        found.len()
    );
    assert_eq!(
        found.len(),
        1,
        "a collider attached since the last step is invisible to queries — the \
         band would admit a building the character can walk through for a step"
    );
}
