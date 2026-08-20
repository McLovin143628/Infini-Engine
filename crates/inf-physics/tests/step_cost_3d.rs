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
/// Built to falsify: the sequence steps a world with a body moving through a
/// field of static boxes, and at every step it asks the same three questions
/// twice — once of the incrementally-maintained tree, and once immediately after
/// [`PhysicsWorld3D::force_query_rebuild`]. If the incremental path ever missed
/// a leaf that moved, or held a leaf at a stale AABB, the two answers diverge on
/// the step it happened.
///
/// The mutation this is aimed at: deleting either half of the pair
/// `query_moved_bodies.extend(islands.active_bodies())` in `step` (the
/// before-the-step half covers a body that fell asleep during it, the after
/// covers one that woke).
#[test]
fn the_incremental_query_tree_answers_what_a_rebuilt_one_does() {
    let (mut w, ball) = field(8);
    let mut compared = 0usize;
    let mut hits = 0usize;
    for step in 0..180 {
        w.step(DT);
        let p = w.body_translation(ball).expect("the ball exists");
        let exclude: std::collections::BTreeSet<inf_physics::d3::ColliderId3D> =
            std::collections::BTreeSet::new();
        // Three questions, of three different kinds, at the moving body.
        let ray = w.cast_ray(p + DVec3::Y * 6.0, -DVec3::Y, 40.0);
        let sweep = w.cast_shape(
            &ColliderShape3D::Sphere { radius: 0.4 },
            p + DVec3::Y * 6.0,
            DQuat::IDENTITY,
            -DVec3::Y,
            40.0,
            &exclude,
        );
        let aabb = w.intersect_aabb(p - DVec3::splat(2.0), p + DVec3::splat(2.0));
        w.force_query_rebuild();
        let ray2 = w.cast_ray(p + DVec3::Y * 6.0, -DVec3::Y, 40.0);
        let sweep2 = w.cast_shape(
            &ColliderShape3D::Sphere { radius: 0.4 },
            p + DVec3::Y * 6.0,
            DQuat::IDENTITY,
            -DVec3::Y,
            40.0,
            &exclude,
        );
        let aabb2 = w.intersect_aabb(p - DVec3::splat(2.0), p + DVec3::splat(2.0));
        compared += 3;
        hits += usize::from(ray.is_some()) + usize::from(sweep.is_some());
        assert_eq!(
            ray.map(|h| (h.collider, h.toi.to_bits())),
            ray2.map(|h| (h.collider, h.toi.to_bits())),
            "step {step}: the ray answered differently before and after a forced \
             rebuild — the incremental tree has drifted from the colliders"
        );
        assert_eq!(
            sweep.map(|h| (h.collider, h.toi.to_bits())),
            sweep2.map(|h| (h.collider, h.toi.to_bits())),
            "step {step}: the sweep answered differently before and after a \
             forced rebuild"
        );
        assert_eq!(
            aabb, aabb2,
            "step {step}: the AABB query answered differently before and after a \
             forced rebuild"
        );
    }
    println!(
        "{compared} query answers compared across 180 steps, {hits} of them hits \
         — incremental and rebuilt agree on every one"
    );
    // Anti-vacuity: a run where nothing ever hit anything would compare `None`
    // against `None` 540 times and prove nothing.
    assert!(
        hits > 180,
        "only {hits} of the queries hit anything — this gate is comparing \
         emptiness with emptiness"
    );
}

/// **A removal still rebuilds, and the rebuild is correct.**
///
/// `BroadPhaseBvh` has no removal, so a leaf can only leave by a fresh build —
/// which is a property of the dependency and therefore a property this file
/// states out loud. Without the `query_rebuild` flag on the removal paths the
/// tree would answer with a collider that no longer exists.
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
