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
