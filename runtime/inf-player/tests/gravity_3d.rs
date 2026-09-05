//! **P29.7: a level's `gravity_3d` reaches the 3D solver, in both hosts.**
//!
//! # The defect
//!
//! `LevelSettings` has carried `gravity_2d` and `gravity_3d` since the scene
//! schema had a version number, and until this wave both hosts built their 3D
//! world with `DVec3::new(0.0, gravity_2d.y, 0.0)`. So `gravity_3d` was authored
//! in the Details panel, serialized, round-tripped by every schema ladder — and
//! read by nothing that simulates. Worse, `LevelSettings::default()` pairs a
//! `gravity_2d` of **zero** with a `gravity_3d` of −9.81, so every level that
//! never touched either field had **no 3D gravity at all** while its panel read
//! −9.81. Two committed samples were in that state.
//!
//! And the editor was one step further out: `commands/sim.rs` passed a literal
//! `DVec2::ZERO`, under a comment about characters applying their own gravity.
//! That is true of a *character* — `CharacterMovement::gravity_mps2` is
//! per-character — and false of every dynamic body, so Simulate floated what the
//! shipped player dropped, in every level, always.
//!
//! # Why nothing could see it
//!
//! Because the two halves cancelled: the samples with dynamic bodies are exactly
//! the samples whose authors typed `gravity_2d.y = -9.81` by hand to make them
//! fall, and no committed level with a dynamic body had ever been played in
//! **both** hosts and compared. This file is that comparison, and it asserts the
//! **world** (how far a body fell, in metres) rather than a report.
//!
//! Mutation-measured: reverting either host to `DVec3::new(0.0, gravity_2d.y,
//! 0.0)` reds `a_dynamic_body_falls_at_the_authored_gravity_3d`; reverting the
//! studio's `gravity_of` to `DVec2::ZERO` is the editor half of the same arm.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_editor_core::scene::serialize::LevelSettings;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_physics::WorldGravity;
use inf_player::runtime_sim::RuntimeInput;

const HZ: f64 = 60.0;
/// Long enough for the difference between two gravities to be metres rather
/// than millimetres, and short enough that the crate is still in free fall.
const STEPS: u32 = 60;
const CRATE: Uuid = Uuid::from_u128(0x2907_0001);
const START_Y: f64 = 50.0;

/// A document with one dynamic crate in the air and nothing to land on.
///
/// Nothing else: the claim is about the number the solver integrates, and a
/// floor, a character or a second body would put a contact between the arm and
/// the thing it measures.
fn crate_doc(settings: LevelSettings) -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_settings(settings);
    let e = doc.create_with_guid(CRATE, inf_editor_core::ipc::SpawnKind::Empty, "Crate", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, START_Y, 0.0);
    doc.world_mut().world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(0.5),
            // Authored, not rapier's 1 kg/m³ placeholder — this arm does not
            // depend on the mass, and a wrong number that looks right is worse
            // than one that is obviously a placeholder.
            density: 500.0,
            ..Default::default()
        },
    ));
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

/// How far the crate fell in the **editor's** Simulate, metres.
fn editor_fall(settings: LevelSettings) -> f64 {
    let mut doc = crate_doc(settings);
    // The studio command's own two lines, which is the point: an arm that built
    // its gravity by hand would pass over a host that ignores the document.
    let gravity = SimSession::gravity_of(&doc);
    let mut session = SimSession::enter_with_gravity(&mut doc, Vec::new(), gravity, HZ);
    for _ in 0..STEPS {
        session.step_once(&mut doc, SimInput::default());
    }
    let y = crate_y(doc.world());
    session.exit(&mut doc);
    START_Y - y
}

/// How far the crate fell in the **shipped player**, metres — through
/// `sim_from_payload`, which is the one PIE boot seam.
fn player_fall(settings: LevelSettings) -> f64 {
    let doc = crate_doc(settings);
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        HZ as u32,
        false,
    )
    .expect("the payload builds");
    let mut sim = inf_player::sim_from_payload(&payload)
        .expect("the payload world builds")
        .sim;
    for _ in 0..STEPS {
        sim.step_once(RuntimeInput::default());
    }
    START_Y - crate_y(sim.world())
}

fn crate_y(world: &inf_ecs::EcsWorld) -> f64 {
    let e = world.entity_of(CRATE).expect("the crate survives the run");
    world
        .world()
        .get::<Transform>(e)
        .expect("…with a transform")
        .translation
        .y
}

/// The settings a level authors: `y2` into `gravity_2d`, `y3` into `gravity_3d`.
fn settings(y2: f64, y3: f64) -> LevelSettings {
    LevelSettings {
        gravity_2d: Vec2d::new(0.0, y2),
        gravity_3d: Vec3d::new(0.0, y3, 0.0),
        ..LevelSettings::default()
    }
}

/// **The arm.** A level authors −3.0 in `gravity_3d` and −9.81 in `gravity_2d`;
/// the crate must fall at −3.0 in both hosts.
///
/// The two numbers are deliberately both non-zero and both plausible: a fixture
/// that paired a real `gravity_3d` with a zero `gravity_2d` would pass over a
/// host that reads neither.
///
/// # Why the claim is a ratio and not a closed form
///
/// A free fall with no damping and no contact is **exactly linear in `g`** —
/// whatever rapier's integrator does to the first step, it does the same thing
/// scaled. So `fall(g₁)·g₂ == fall(g₂)·g₁` is a claim about the solver's input
/// that needs no model of its interior, and the first draft of this arm, which
/// hand-derived `g·dt²·n(n+1)/2`, was wrong by 1.2 % against a rapier that had
/// read the right number. A tolerance wide enough to hide that would have been
/// wide enough to hide a wrong gravity.
#[test]
fn a_dynamic_body_falls_at_the_authored_gravity_3d() {
    let slow = player_fall(settings(-9.81, -3.0));
    let fast = player_fall(settings(-9.81, -9.81));
    assert!(
        fast > 4.0,
        "the fixture must actually fall, or every ratio below is 0/0: {fast} m"
    );
    assert!(
        fast - slow > 3.0,
        "…and the two gravities must be far enough apart to tell apart: {fast} vs {slow}"
    );
    assert!(
        (slow * 9.81 - fast * 3.0).abs() < 1e-9,
        "the shipped player fell {slow} m under an authored -3.0 and {fast} m under -9.81; \
         those are not the same gravity scaled, so the solver did not read `gravity_3d`"
    );

    let editor = editor_fall(settings(-9.81, -3.0));
    assert_eq!(
        editor.to_bits(),
        slow.to_bits(),
        "the editor's Simulate fell {editor} m and the shipped player {slow} m — \
         the two hosts must read the same field and agree bit for bit"
    );
}

/// **The defect the P29.6 audit named**: a level that touched neither field had
/// no 3D gravity at all, in a world whose Details panel read −9.81.
///
/// The claim is an equality against the level that authors −9.81 explicitly, so
/// it needs no model of the integrator either.
#[test]
fn a_level_that_touched_neither_field_now_falls_at_earth_gravity() {
    let untouched = player_fall(LevelSettings::default());
    let explicit = player_fall(settings(-9.81, -9.81));
    assert!(explicit > 4.0, "the fixture must fall: {explicit} m");
    assert_eq!(
        untouched.to_bits(),
        explicit.to_bits(),
        "an untouched level's crate fell {untouched} m and an explicitly-Earth one {explicit} m; \
         before P29.7 the first number was 0"
    );
    let editor = editor_fall(LevelSettings::default());
    assert_eq!(
        editor.to_bits(),
        explicit.to_bits(),
        "…and the editor's Simulate fell {editor} m (it used to pass DVec2::ZERO)"
    );
}

/// A level that authors **no** 3D gravity gets none — the field is an authored
/// number and not a suggestion, so the fix must not have replaced one hard-coded
/// vector with another.
#[test]
fn a_level_that_authors_zero_3d_gravity_gets_none() {
    let settings = settings(-9.81, 0.0);
    assert_eq!(
        player_fall(settings),
        0.0,
        "a zero `gravity_3d` must float the crate even though `gravity_2d.y` is -9.81"
    );
    assert_eq!(editor_fall(settings), 0.0);
}

/// The 2D solver keeps reading its own field — the change is a split, not a
/// rename.
#[test]
fn the_pair_keeps_the_two_solvers_apart() {
    let doc = crate_doc(settings(-20.0, -3.0));
    let g = SimSession::gravity_of(&doc);
    assert_eq!(g.d2, DVec2::new(0.0, -20.0));
    assert_eq!(g.d3, DVec3::new(0.0, -3.0, 0.0));
    // And the legacy door still means what it always meant.
    assert_eq!(
        WorldGravity::from_2d(DVec2::new(0.0, -20.0)).d3,
        DVec3::new(0.0, -20.0, 0.0)
    );
}
