//! **The grip gate** (SK1b): a character takes hold of three different things,
//! its fingers conform to each of them, and the editor's PIE and the shipped
//! player agree about the resulting pose byte for byte.
//!
//! # What this is a gate on
//!
//! SK1a shipped `GripAffordance` with an empty table on every rig and wrote down
//! that it had no consumer. This wave gave it one — `inf_anim::grip`, invoked
//! from `inf_ecs::pose::step_pose_evaluation` through the `HandIkRes` resource —
//! and this is the arm that says the whole path holds together **across two
//! processes' worth of hosts**, which is the only thing that ever catches a pose
//! writer that exists in one fixed step and not the other.
//!
//! Three grips, deliberately, because the claim is *per-affordance conformance*
//! and one grip cannot demonstrate it:
//!
//! * a **handle** — one hand, a 4.5 cm bar;
//! * a **rifle**, two-handed, with the off hand carried by the weapon through
//!   `ik_hand_gun` (the `GunGrip` path);
//! * a **thrown prop** — a 9 cm ball, gripped and then *released*, because a
//!   release is the half a grip solver most easily gets wrong: `apply_grip` sets
//!   a pose rather than accumulating a delta, so letting go must return the hand
//!   exactly to where an ungripped one poses, to the bit.
//!
//! # The trace
//!
//! `inf_ecs::pose::pose_state_bytes` — the same fold `pose_parity` compares, over
//! a 161-bone rig, so a divergence anywhere in the drive pass, the hand IK, the
//! finger solver or the correction re-drive lands here.

use std::collections::BTreeMap;

use glam::DVec2;
use uuid::Uuid;

use inf_anim::{
    AnimClip, BoneSide, Interpolation, JointTrack, QuatTrack, SkeletonAsset, SmState, SmTransition,
    StateMachine,
};
use inf_ecs::components::{AnimStateMachine, SkeletalMesh};
use inf_ecs::math::Vec3d;
use inf_ecs::pose::{GunGrip, HandGrip, HandIk, HandReach};
use inf_ecs::world::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;
/// Long enough to walk the whole sequence below and settle after it.
const STEPS: u32 = 24;

const HERO: Uuid = Uuid::from_u128(0x5B10_0000_0000_0000_0000_0000_0000_0001);
const SM: Uuid = Uuid::from_u128(0x5B10_0000_0000_0000_0000_0000_0000_0002);
const SKEL: Uuid = Uuid::from_u128(0x5B10_0000_0000_0000_0000_0000_0000_0003);
const MESH: Uuid = Uuid::from_u128(0x5B10_0000_0000_0000_0000_0000_0000_0004);
const IDLE: Uuid = Uuid::from_u128(0x5B10_0000_0000_0000_0000_0000_0000_0010);

// ── the fixture ─────────────────────────────────────────────────────────────

/// The 161-bone mannequin, **with the catalogue it now generates for itself**.
///
/// SK1b authored the four affordances here by hand and said why: a grip
/// catalogue is content, and the wave that ships one was not that one. SK1c is,
/// and the four numbers moved into `inf_anim::grip::grip_catalogue` unchanged —
/// so every measurement this gate committed is the same measurement, and the
/// thing that changed is who wrote the table. That is the point: the arms below
/// now run against what a rig arrives with rather than against a fixture, and
/// `a_generated_rig_is_the_catalogue_this_gate_takes` is what keeps the two the
/// same.
fn rig() -> SkeletonAsset {
    inf_anim::build_manny(&inf_anim::BodyParams::default()).expect("a mannequin")
}

/// A clip that holds one spine joint at a constant angle, so the machine poses
/// something and the pose is a constant the assertions can name.
///
/// One key, `Step` — `pose_parity`'s finding, kept: two identical keys under
/// `Linear` drift by a ULP as the play-head moves, which is invisible to a human
/// and fatal to a byte comparison.
fn hold(joint: u16, deg: f32) -> AnimClip {
    let half = (deg as f64 * 0.5).to_radians();
    let q = [
        0.0,
        0.0,
        inf_math::psin64(half) as f32,
        inf_math::pcos64(half) as f32,
    ];
    AnimClip::new(
        "hold",
        vec![JointTrack {
            joint,
            translation: None,
            rotation: Some(QuatTrack::new(vec![0.0], vec![q], Interpolation::Step)),
            scale: None,
        }],
    )
}

fn machine() -> StateMachine {
    StateMachine {
        states: vec![SmState::clip("idle", *IDLE.as_bytes())],
        transitions: Vec::<SmTransition>::new(),
        entry: 0,
        ..Default::default()
    }
}

fn skeletons() -> BTreeMap<Uuid, SkeletonAsset> {
    BTreeMap::from([(SKEL, rig())])
}

fn machines() -> BTreeMap<Uuid, StateMachine> {
    BTreeMap::from([(SM, machine())])
}

fn clips() -> BTreeMap<Uuid, AnimClip> {
    // Joint 3 is `spine_02` on the mannequin — a bone the grip solver never
    // touches, so "the pose changed" always means the hands changed.
    BTreeMap::from([(IDLE, hold(3, 12.0))])
}

fn character() -> (AnimStateMachine, SkeletalMesh) {
    (
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
        },
        SkeletalMesh {
            mesh: Some(MESH),
            skeleton: Some(SKEL),
        },
    )
}

// ── the sequence, written once and driven by both hosts ─────────────────────

/// Where the right hand reaches for the thing it is about to take hold of.
const HOLD_AT: Vec3d = Vec3d::new(0.22, 1.18, 0.42);

/// **The grip/release sequence.** One function, so the two hosts cannot drive
/// different courses — the `pose_parity` discipline, and the reason a trace
/// comparison means anything at all.
///
/// | steps | what the hands are doing |
/// |---|---|
/// | 0–3 | nothing: the animation alone |
/// | 4–8 | the right hand reaches a handle and closes on it |
/// | 9–14 | both hands on a rifle, the off hand carried by `ik_hand_gun` |
/// | 15–19 | the right hand on a thrown prop, easing in |
/// | 20–23 | the prop is **released** — the hand must return to the open pose |
fn plan(step: u32) -> Option<HandIk> {
    let reach = |t: Vec3d| {
        Some(HandReach {
            target: t,
            weight: 1.0,
        })
    };
    match step {
        0..=3 => None,
        4..=8 => Some(HandIk {
            reach: [None, reach(HOLD_AT)],
            gun: None,
            grip: [
                None,
                Some(HandGrip {
                    name: "handle".into(),
                    // Eased in over the five steps, so the trace has to carry a
                    // *changing* pose rather than one step of a new constant.
                    amount: (step - 3) as f32 / 5.0,
                }),
            ],
        }),
        9..=14 => Some(HandIk {
            reach: [None, reach(Vec3d::new(0.10, 1.30, 0.36))],
            gun: Some(GunGrip {
                holding: BoneSide::Right,
                // 30 cm along the barrel: a rifle's fore-grip.
                off_hand_offset: [0.0, 0.0, 0.30],
                weight: 1.0,
            }),
            grip: [
                Some(HandGrip {
                    name: "rifle_fore".into(),
                    amount: 1.0,
                }),
                Some(HandGrip {
                    name: "rifle".into(),
                    amount: 1.0,
                }),
            ],
        }),
        15..=19 => Some(HandIk {
            reach: [None, reach(HOLD_AT)],
            gun: None,
            grip: [
                None,
                Some(HandGrip {
                    name: "prop".into(),
                    amount: (step - 14) as f32 / 5.0,
                }),
            ],
        }),
        // The release: the hands go back to doing nothing at all, which must be
        // byte-identical to steps 0–3.
        _ => None,
    }
}

// ── the two hosts ───────────────────────────────────────────────────────────

/// One step's record: the pose bytes, and what the hand pass said it did.
#[derive(Debug, Clone, PartialEq)]
struct Step {
    pose: Vec<u8>,
    /// `(arms solved, finger bones written, cone clamps, bones re-driven)`.
    engagement: (u32, u32, u32, usize),
}

fn engagement(world: &EcsWorld) -> (u32, u32, u32, usize) {
    let Some(r) = inf_ecs::pose::hand_ik_report(world, HERO) else {
        return (0, 0, 0, 0);
    };
    let solved = r
        .reach
        .iter()
        .chain(std::iter::once(&r.gun))
        .filter(|o| matches!(o, Some(inf_ecs::pose::IkOutcome::Solved(_))))
        .count() as u32;
    (
        solved,
        r.grip.iter().map(|g| g.joints).sum(),
        r.grip.iter().map(|g| g.clamped).sum(),
        r.redriven,
    )
}

fn player_trace() -> Vec<Step> {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(character());
    world.mark_dirty();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, HZ);
    sim.set_state_machines(machines());
    sim.set_skeletons(skeletons());
    sim.set_pose_clips(clips());
    (0..STEPS)
        .map(|s| {
            inf_ecs::pose::set_hand_ik(sim.world_mut(), HERO, plan(s).unwrap_or_default());
            sim.step_once(RuntimeInput::default());
            Step {
                pose: inf_ecs::pose::pose_state_bytes(sim.world()),
                engagement: engagement(sim.world()),
            }
        })
        .collect()
}

fn editor_trace() -> Vec<Step> {
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(character());
    doc.world_mut().mark_dirty();
    let mut session = SimSession::enter(&mut doc, Vec::new(), DVec2::ZERO, HZ);
    session.set_state_machines(machines());
    session.set_skeletons(skeletons());
    session.set_pose_clips(clips());
    let out = (0..STEPS)
        .map(|s| {
            inf_ecs::pose::set_hand_ik(doc.world_mut(), HERO, plan(s).unwrap_or_default());
            session.step_once(&mut doc, SimInput::default());
            Step {
                pose: inf_ecs::pose::pose_state_bytes(doc.world()),
                engagement: engagement(doc.world()),
            }
        })
        .collect();
    session.exit(&mut doc);
    out
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **ANTI-VACUITY**, and it is most of this gate.
///
/// Two empty traces are equal; so are two traces of a character that never
/// gripped anything. Every claim below is about a step where something is
/// supposed to have *happened*, asserted against a step where it is not.
fn assert_not_vacuous(t: &[Step]) {
    assert_eq!(t.len() as u32, STEPS);
    assert!(!t[0].pose.is_empty(), "step 0 published no pose at all");
    // **The course really is a course** (SK1b audit). The distinct-pose count is
    // the headline number this gate prints, and printing it is not asserting it:
    // a solver that collapsed every grip onto one pose would still satisfy the
    // handful of `assert_ne!` pairs below if it happened to keep those apart.
    // Twelve of twenty-four, pinned, and the byte length with it — 6476 is the
    // 161-bone trace SK1a priced (36 B header + 40 B per joint), so a rig that
    // silently lost its side tables would be caught here rather than looking
    // like a quieter grip.
    let mut distinct: Vec<&Vec<u8>> = t.iter().map(|s| &s.pose).collect();
    distinct.sort();
    distinct.dedup();
    assert_eq!(
        (distinct.len(), t[0].pose.len()),
        (12, 6476),
        "the grip course posed {} distinct poses of {} bytes",
        distinct.len(),
        t[0].pose.len()
    );
    // Nothing is asked for over 0..=3, so those steps are one settled pose.
    assert_eq!(t[0].pose, t[3].pose, "the idle pose is not settled");
    assert_eq!(t[0].engagement, (0, 0, 0, 0), "{:?}", t[0].engagement);

    // The handle: the arm solves, the fingers close, and closing FURTHER moves
    // the pose again — an eased grip that snapped to its end state on the first
    // step would satisfy "the pose changed" and would not be an ease.
    assert_ne!(t[3].pose, t[5].pose, "taking hold moved nothing");
    assert_ne!(t[5].pose, t[7].pose, "the grip did not tighten");
    assert_eq!(t[5].engagement.0, 1, "one arm should have solved");
    assert!(t[5].engagement.1 >= 15, "{:?}", t[5].engagement);
    assert!(
        t[5].engagement.3 > 0,
        "the correction re-drive did not run over a corrected pose"
    );

    // The rifle: TWO arms solve, and both hands carry fingers.
    assert_eq!(
        t[12].engagement.0, 2,
        "a two-handed hold solves both arms: {:?}",
        t[12].engagement
    );
    assert!(
        t[12].engagement.1 > t[5].engagement.1,
        "two hands wrote no more finger bones than one ({:?} vs {:?})",
        t[12].engagement,
        t[5].engagement
    );

    // The prop: a different affordance produces a different pose from the
    // handle, at the same reach and the same amount.
    assert_ne!(
        t[8].pose, t[19].pose,
        "a 9 cm ball and a 4.5 cm bar posed the same hand — the aperture is doing nothing"
    );

    // **The release.** Letting go returns the hand to the pose it had before it
    // ever took hold — byte for byte, which is the claim `apply_grip`'s "a curl
    // is a pose, not a delta" rests on.
    assert_eq!(
        t[0].pose, t[23].pose,
        "a released hand did not return to the open pose"
    );
    assert_eq!(t[23].engagement, (0, 0, 0, 0));
}

/// **THE GATE: PIE == shipping over a grip/release sequence.**
#[test]
fn pie_equals_shipping_over_a_grip_and_release() {
    let player = player_trace();
    let editor = editor_trace();
    assert_not_vacuous(&player);
    assert_not_vacuous(&editor);
    println!(
        "the grip course: {} steps, {} distinct poses, {} bytes a step",
        player.len(),
        {
            let mut d: Vec<&Vec<u8>> = player.iter().map(|s| &s.pose).collect();
            d.sort();
            d.dedup();
            d.len()
        },
        player[0].pose.len()
    );
    for (i, (a, b)) in player.iter().zip(editor.iter()).enumerate() {
        assert_eq!(
            a.engagement, b.engagement,
            "step {i}: the two hosts' hand passes did different amounts of work"
        );
        assert_eq!(
            a.pose, b.pose,
            "step {i}: the shipped player and the editor Simulate posed the same \
             character differently — a grip would look one way in the preview and \
             another in the shipped build"
        );
    }
}

/// **The fingers conform to the affordance**, and the affordance is the thing
/// that decides — measured on the hand itself, in metres, rather than inferred
/// from the fact that two poses differ.
#[test]
fn each_grip_closes_the_hand_by_its_own_aperture() {
    let asset = rig();
    let sk = &asset.skeleton;
    let hands = inf_anim::hands_of(sk, asset.role_index());
    let hand = hands[1].clone().expect("a right hand");
    let wrist = hand.joint as usize;
    let tip = *hand
        .finger(inf_anim::Digit::Middle)
        .expect("a middle finger")
        .joints
        .last()
        .expect("a tip") as usize;

    let span = |name: &str| -> (f32, inf_anim::GripReport) {
        let grip = asset
            .grips
            .iter()
            .find(|g| g.name == name)
            .unwrap_or_else(|| panic!("the rig authors a `{name}` grip"));
        let mut pose = inf_anim::Pose::rest(sk);
        let report = inf_anim::apply_grip(sk, &mut pose, &hand, &asset.limits, grip, 1.0);
        let g = inf_anim::global_transforms(sk, &pose);
        (
            (g[tip].w_axis.truncate() - g[wrist].w_axis.truncate()).length(),
            report,
        )
    };

    let open = {
        let g = inf_anim::global_transforms(sk, &inf_anim::Pose::rest(sk));
        (g[tip].w_axis.truncate() - g[wrist].w_axis.truncate()).length()
    };
    let (handle, hr) = span("handle");
    let (rifle, rr) = span("rifle");
    let (prop, pr) = span("prop");
    println!(
        "fingertip to wrist: open {open:.4} m, handle {handle:.4}, rifle {rifle:.4}, prop {prop:.4}"
    );
    // Every grip closes the hand.
    for (name, s) in [("handle", handle), ("rifle", rifle), ("prop", prop)] {
        assert!(s < open, "the `{name}` grip did not close the hand");
    }
    // **A thicker thing is held in a more open hand**, which is the whole content
    // of `aperture_m`: the 9 cm prop leaves the fingers straighter than the
    // 4.5 cm handle, which leaves them straighter than the 3.2 cm rifle grip.
    assert!(
        prop > handle && handle > rifle,
        "the aperture ordering does not hold: prop {prop}, handle {handle}, rifle {rifle}"
    );
    // …and the report says so per digit, so this is a counter and not a picture.
    for d in inf_anim::Digit::ALL {
        assert!(
            pr.closure[d.slot()] < hr.closure[d.slot()],
            "{d:?} closed further on a 9 cm ball than on a 4.5 cm bar"
        );
        assert!(hr.closure[d.slot()] < rr.closure[d.slot()], "{d:?}");
    }
    // The rifle's trigger finger is authored **straight** and the solver honours
    // it — the one place a per-finger curl target says something a per-hand
    // aperture cannot.
    let index = rr.closure[inf_anim::Digit::Index.slot()];
    assert!(
        index > 0.0,
        "the aperture closed the index finger to nothing"
    );
    let mut pose = inf_anim::Pose::rest(sk);
    let rifle_grip = asset.grips.iter().find(|g| g.name == "rifle").unwrap();
    inf_anim::apply_grip(sk, &mut pose, &hand, &asset.limits, rifle_grip, 1.0);
    for &j in &hand.finger(inf_anim::Digit::Index).unwrap().joints {
        assert_eq!(
            pose.locals[j as usize].rotation,
            inf_anim::Pose::rest(sk).locals[j as usize].rotation,
            "the trigger finger was curled by a grip that asks for zero curl on it"
        );
    }
}

/// **The catalogue this gate exercises is the one a rig arrives with** (SK1c).
///
/// The four affordances used to be authored in this file. They are generated
/// now, and every measurement above is unchanged — which is a claim worth an arm
/// rather than a sentence: if `grip_catalogue` ever moved an aperture, the
/// aperture-ordering test would still pass (it asserts an *order*) and
/// `pie_equals_shipping_over_a_grip_and_release` would still pass (it compares
/// two hosts, and both would move together). The numbers are what the gate's
/// committed readings rest on, so the numbers are pinned.
#[test]
fn a_generated_rig_is_the_catalogue_this_gate_takes() {
    let rig = rig();
    let roles = rig.role_index();
    let hand = |side| {
        roles
            .first(inf_anim::BoneRoleKind::Hand, side)
            .expect("a hand")
    };
    let want = [
        (inf_anim::GRIP_HANDLE, hand(BoneSide::Right), 0.045f32),
        (inf_anim::GRIP_RIFLE, hand(BoneSide::Right), 0.032),
        (inf_anim::GRIP_RIFLE_FORE, hand(BoneSide::Left), 0.045),
        (inf_anim::GRIP_PROP, hand(BoneSide::Right), 0.09),
    ];
    assert_eq!(rig.grips.len(), want.len(), "{:?}", rig.grips);
    for (name, joint, aperture) in want {
        let g = rig
            .grips
            .iter()
            .find(|g| g.name == name)
            .unwrap_or_else(|| panic!("the generated rig has no `{name}`"));
        assert_eq!(g.hand, joint, "`{name}` is on the wrong hand");
        assert_eq!(
            g.aperture_m, aperture,
            "`{name}`'s aperture moved — every closure this file prints was \
             measured against the old one"
        );
    }
    // …and the affordance every name in `plan` reaches really resolves, which is
    // the silent failure `apply_hand_ik` has: a `HandGrip` naming a grip the rig
    // does not carry writes nothing and reports nothing.
    for name in [
        inf_anim::GRIP_HANDLE,
        inf_anim::GRIP_RIFLE,
        inf_anim::GRIP_RIFLE_FORE,
        inf_anim::GRIP_PROP,
    ] {
        assert!(
            rig.grips.iter().any(|g| g.name == name),
            "the plan names `{name}` and the rig does not carry it"
        );
    }
}
