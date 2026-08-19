//! The pure ragdoll-setup helper (P12.1 seed).

use glam::DVec3;
use inf_physics::d3::{BodyKind3D, ColliderShape3D, JointKind3D, PhysicsWorld3D};
use inf_physics::ragdoll::{build_ragdoll, classify, BoneRole, RagdollBone, RagdollConfig};

/// A character-demo-style humanoid: hips (root) + spine + chest, a left arm
/// (upper→lower), and a left leg (thigh→shin) — 7 bones ⇒ 6 joints.
fn demo_skeleton() -> Vec<RagdollBone> {
    vec![
        RagdollBone::new("Hips", DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, 1.2, 0.0)),
        RagdollBone::new(
            "Spine",
            DVec3::new(0.0, 1.2, 0.0),
            DVec3::new(0.0, 1.5, 0.0),
        ),
        RagdollBone::new(
            "Chest",
            DVec3::new(0.0, 1.5, 0.0),
            DVec3::new(0.0, 1.8, 0.0),
        ),
        RagdollBone::new(
            "UpperArm.L",
            DVec3::new(0.2, 1.75, 0.0),
            DVec3::new(0.5, 1.5, 0.0),
        ),
        RagdollBone::new(
            "LowerArm.L",
            DVec3::new(0.5, 1.5, 0.0),
            DVec3::new(0.75, 1.25, 0.0),
        ),
        RagdollBone::new(
            "Thigh.L",
            DVec3::new(0.1, 1.0, 0.0),
            DVec3::new(0.12, 0.5, 0.0),
        ),
        RagdollBone::new(
            "Shin.L",
            DVec3::new(0.12, 0.5, 0.0),
            DVec3::new(0.13, 0.05, 0.0),
        ),
    ]
}

#[test]
fn classify_recognizes_common_conventions() {
    assert_eq!(classify("Hips"), Some(BoneRole::Hips));
    assert_eq!(classify("pelvis"), Some(BoneRole::Hips));
    assert_eq!(classify("Spine"), Some(BoneRole::Spine));
    assert_eq!(classify("UpperArm.L"), Some(BoneRole::UpperArmL));
    assert_eq!(classify("lowerarm_r"), Some(BoneRole::LowerArmR));
    assert_eq!(classify("LeftForeArm"), Some(BoneRole::LowerArmL));
    assert_eq!(classify("calf_r"), Some(BoneRole::ShinR));
    assert_eq!(classify("thigh.L"), Some(BoneRole::ThighL));
    assert_eq!(classify("weapon_socket"), None);
}

#[test]
fn build_ragdoll_produces_six_joints_one_root() {
    let parts = build_ragdoll(&demo_skeleton(), RagdollConfig::default());
    assert_eq!(parts.len(), 7, "one body per classified bone");

    let roots: Vec<_> = parts.iter().filter(|p| p.joint.is_none()).collect();
    assert_eq!(roots.len(), 1, "exactly one free root");
    assert_eq!(roots[0].role, BoneRole::Hips, "the root is the hips");

    let joints: Vec<_> = parts.iter().filter_map(|p| p.joint.as_ref()).collect();
    assert_eq!(joints.len(), 6, "6 joints for a 7-bone chain");

    // Parents precede children (a valid spawn order).
    for (i, p) in parts.iter().enumerate() {
        if let Some(j) = &p.joint {
            assert!(
                j.parent < i,
                "part {i} references a later parent {}",
                j.parent
            );
        }
    }

    // Elbow + knee are revolute hinges; the other four joints are spherical.
    let hinges = joints
        .iter()
        .filter(|j| matches!(j.desc.kind, JointKind3D::Revolute { .. }))
        .count();
    let spherical = joints
        .iter()
        .filter(|j| matches!(j.desc.kind, JointKind3D::Spherical))
        .count();
    assert_eq!(hinges, 2, "elbow + knee hinge");
    assert_eq!(spherical, 4, "spine, chest, shoulder, hip are ball joints");

    // Every body is dynamic with a capsule collider.
    for p in &parts {
        assert_eq!(p.body.kind, BodyKind3D::Dynamic);
        assert!(matches!(p.collider.shape, ColliderShape3D::Capsule { .. }));
    }
}

/// **The two anchors of a joint name ONE point in the world** (P29.6 audit, A1).
///
/// `local_anchor1` belongs to body 1 and `local_anchor2` to body 2, and every
/// consumer in this repository spawns a ragdoll joint as
/// `add_joint(parent, child, desc)` — so slot 1 is the **parent's** frame. From
/// P12.1 until P29.6 the builder filled them the other way round, and nothing in
/// the tree could see it: `build_ragdoll` had no runtime consumer for seven
/// phases, and every fixture that did spawn one spawned it **at rest on flat
/// ground**, where a mis-anchored joint still settles into a heap — just the
/// wrong one. `ragdoll_spawns_and_settles_without_exploding` below passes with
/// the anchors swapped; so does `phase12_gate`'s playground.
///
/// This arm is the direct one. Both anchors resolve, through their own body's
/// pose, to the child bone's **head** — the point the two bones share — which is
/// what an anchor pair means. Swap the two fields and it fails on the first
/// joint, exactly, with no physics stepped and no tolerance to argue about.
#[test]
fn the_joint_anchors_name_one_world_point() {
    let skeleton = demo_skeleton();
    let parts = build_ragdoll(&skeleton, RagdollConfig::default());
    let head_of = |name: &str| -> DVec3 {
        skeleton
            .iter()
            .find(|b| b.name == name)
            .expect("the fixture names this bone")
            .head
    };
    let to_world =
        |i: usize, local: DVec3| -> DVec3 { parts[i].rotation * local + parts[i].position };

    let mut checked = 0;
    for (i, part) in parts.iter().enumerate() {
        let Some(j) = &part.joint else { continue };
        let shared = head_of(&part.name);
        let on_parent = to_world(j.parent, j.desc.local_anchor1);
        let on_child = to_world(i, j.desc.local_anchor2);
        assert!(
            (on_parent - shared).length() < 1e-9,
            "`{}`'s joint: local_anchor1 resolves in the PARENT (`{}`) to {on_parent:?}, \
             which is not the shared point {shared:?} — the anchors are the wrong way \
             round, and `add_joint(parent, child, desc)` puts the parent in slot 1",
            part.name,
            parts[j.parent].name
        );
        assert!(
            (on_child - shared).length() < 1e-9,
            "`{}`'s joint: local_anchor2 resolves in the CHILD to {on_child:?}, not {shared:?}",
            part.name
        );
        // The two frames are genuinely different, so the arm is not satisfied by
        // a builder that wrote the same vector into both slots.
        assert_ne!(
            j.desc.local_anchor1, j.desc.local_anchor2,
            "`{}`'s two anchors are the same vector, so this comparison proves nothing",
            part.name
        );
        // A ragdoll's adjacent limbs overlap by construction (they share this
        // very point), so jointed pairs must not push each other apart.
        assert!(
            !j.desc.contacts,
            "`{}`'s joint still collides with its parent",
            part.name
        );
        checked += 1;
    }
    assert_eq!(checked, 6, "the sweep visited the wrong number of joints");
}

/// **Every limb is in the ragdoll layer and ignores its own kind**, and the
/// **bound** of that answer is asserted rather than described (P29.6 audit, A1).
///
/// A ragdoll's limb capsules overlap by construction, so two of them pushing
/// each other apart is a depenetration force with nowhere to go. Turning
/// contacts off between *jointed* pairs handles the adjacent ones; the rest need
/// a layer. This arm is the layer's only falsifier: the dynamic arm below does
/// not fail without it, because this file's seven-bone fixture is too sparse for
/// a non-adjacent pair to overlap — which is exactly why the property is
/// asserted directly instead of hoped for through a simulation.
///
/// The bound the ledger states — *two different ragdolls pass through each
/// other* — is the second half. It is a consequence of one shared bit, and
/// writing it down as an assertion is what makes it fail the day somebody gives
/// each ragdoll its own group.
#[test]
fn every_limb_carries_the_ragdoll_layer_and_ignores_its_own_kind() {
    use inf_physics::ragdoll::{ragdoll_layers, RAGDOLL_LAYER_BIT};
    let parts = build_ragdoll(&demo_skeleton(), RagdollConfig::default());
    assert_eq!(parts.len(), 7);
    for p in &parts {
        assert_eq!(
            p.collider.layers,
            ragdoll_layers(),
            "`{}` is not in the ragdoll layer, so it collides with the limb it \
             shares an endpoint with",
            p.name
        );
        assert!(
            p.collider.layers.memberships & RAGDOLL_LAYER_BIT != 0,
            "`{}` is not a member of the ragdoll bit",
            p.name
        );
        assert!(
            p.collider.layers.filter & RAGDOLL_LAYER_BIT == 0,
            "`{}` still filters IN the ragdoll bit",
            p.name
        );
        // …and it still meets the world. A mask that filtered everything would
        // satisfy the two lines above and drop the ragdoll through the floor.
        let world_layers = inf_physics::CollisionLayers::default();
        assert!(
            p.collider.layers.filter & world_layers.memberships != 0
                && world_layers.filter & p.collider.layers.memberships != 0,
            "`{}` cannot collide with an ordinary collider — it would fall \
             through the floor",
            p.name
        );
    }
    // **The bound, asserted.** One bit for every ragdoll means two ragdolls do
    // not collide with each other. The day that is fixed with a per-body group
    // id, this line is what says the ledger has to be rewritten.
    let a = ragdoll_layers();
    let b = ragdoll_layers();
    assert!(
        a.filter & b.memberships == 0,
        "two different ragdolls now collide — the P29.6 bound is closed and its \
         ledger entry is stale"
    );
}

/// **A ragdoll given a real impact does not gain energy** (P29.6 audit, A1).
///
/// The arm the tree did not have, and the reason three defects lived in
/// `build_ragdoll` for seventeen sub-phases: every ragdoll fixture in this
/// repository spawned one **from rest**, and all three defects are invisible at
/// rest. The P29.6 course was the first thing ever to hand a ragdoll a 10.7 m/s
/// landing, and it found backwards joint anchors, limbs depenetrating against
/// each other, and a density of 1 kg/m³.
///
/// So: drop the whole rig onto a floor at 11 m/s and require that it **slows
/// down**. One number, no tolerance to tune, and it dies on any of the three:
/// swap the anchors and the solver puts metres per second of sideways velocity
/// into the pelvis; re-enable contacts between jointed pairs, or drop the layer
/// mask, and the overlapping capsules push each other apart; set the density
/// back to 1.0 and six-gram limbs are launched by their own depenetration.
#[test]
fn a_ragdoll_landing_at_speed_loses_energy_rather_than_gaining_it() {
    const IMPACT: f64 = -11.0;
    let parts = build_ragdoll(&demo_skeleton(), RagdollConfig::default());
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let floor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, -0.5, 0.0),
        glam::DQuat::IDENTITY,
    );
    world.add_collider(
        floor,
        inf_physics::d3::ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(20.0, 0.5, 20.0),
        }),
    );
    let mut bodies = Vec::with_capacity(parts.len());
    for p in &parts {
        let b = world.add_body(p.body.kind, p.position, p.rotation);
        world.add_collider(b, p.collider.clone());
        // The whole rig arrives together, which is what a character that just
        // fell five metres actually does.
        world.set_body_linvel(b, DVec3::new(0.0, IMPACT, 0.0));
        bodies.push(b);
    }
    for (i, p) in parts.iter().enumerate() {
        if let Some(j) = &p.joint {
            world
                .add_joint(bodies[j.parent], bodies[i], j.desc)
                .expect("ragdoll joint should build");
        }
    }
    // A limb weighs what a limb weighs — the fourth catch of the placeholder
    // law, asserted rather than described.
    for b in &bodies {
        let m = world.body_mass(*b).expect("a dynamic body has a mass");
        assert!(
            m > 0.5,
            "a limb weighs {m} kg — `RagdollConfig::density` is back at rapier's placeholder"
        );
    }

    let fastest = |w: &PhysicsWorld3D| -> f64 {
        bodies
            .iter()
            .map(|b| w.body_linvel(*b).unwrap_or(DVec3::ZERO).length())
            .fold(0.0, f64::max)
    };
    let start = fastest(&world);
    assert!(
        start >= 10.0,
        "the fixture did not arrive at speed: {start}"
    );
    let mut worst: f64 = 0.0;
    for _ in 0..240 {
        world.step(1.0 / 60.0);
        worst = worst.max(fastest(&world));
    }
    // Nothing may end up going FASTER than it arrived. Gravity over four seconds
    // could add 39 m/s to a body in free fall, so the bound is the impact speed
    // itself plus that budget — and a settled ragdoll is nowhere near either.
    assert!(
        worst <= start + 1.0,
        "a limb reached {worst:.2} m/s from a {start:.2} m/s landing — the rig is \
         being pushed by its own constraints, not by the world"
    );
    let end = fastest(&world);
    assert!(
        end < 1.0,
        "the ragdoll never settled: {end:.3} m/s after four seconds"
    );
    for b in &bodies {
        let p = world.body_translation(*b).unwrap();
        assert!(
            p.is_finite() && p.length() < 20.0,
            "a limb flew away: {p:?}"
        );
    }
}

#[test]
fn ragdoll_spawns_and_settles_without_exploding() {
    // Spawn the descriptors into a real world and step it: the ragdoll should
    // fall under gravity and stay finite/bounded (no solver blow-up).
    let parts = build_ragdoll(&demo_skeleton(), RagdollConfig::default());
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));

    // A floor to catch it.
    let floor = world.add_body(
        BodyKind3D::Static,
        DVec3::new(0.0, -0.5, 0.0),
        glam::DQuat::IDENTITY,
    );
    world.add_collider(
        floor,
        inf_physics::d3::ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::new(20.0, 0.5, 20.0),
        }),
    );

    // Spawn bodies + colliders, remembering each part's body handle.
    let mut bodies = Vec::with_capacity(parts.len());
    for p in &parts {
        let b = world.add_body(p.body.kind, p.position, p.rotation);
        world.add_collider(b, p.collider.clone());
        bodies.push(b);
    }
    // Wire joints (parents precede children, so the parent body already exists).
    for (i, p) in parts.iter().enumerate() {
        if let Some(j) = &p.joint {
            world
                .add_joint(bodies[j.parent], bodies[i], j.desc)
                .expect("ragdoll joint should build");
        }
    }
    assert_eq!(world.joint_ids().len(), 6);

    for _ in 0..300 {
        world.step(1.0 / 60.0);
    }
    for b in &bodies {
        let p = world.body_translation(*b).unwrap();
        assert!(p.is_finite(), "a ragdoll body went non-finite: {p:?}");
        assert!(p.length() < 100.0, "a ragdoll body flew away: {p:?}");
    }
}
