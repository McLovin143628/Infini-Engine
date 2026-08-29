//! **The iteration claim, measured on real committed content** (SCRIPT3 clause 4).
//!
//! `script_hot_reload` proved the mechanism on a fixture: a four-line counter in
//! a temp directory. This is the claim the whole arc exists to make, on the
//! thing a designer would actually be editing — **the Harbour Heist mission**,
//! 5.6 KB of committed `.infini` with two handlers and six functions, running in
//! a real `SimSession` over its own committed level, mid-mission.
//!
//! The designer's edit is a balance change: the bank's staff hit for `12.0` and
//! the designer decides that is not enough. One number, one save, and the run
//! that was going to get away clean is now going to get caught — **without
//! leaving the session, and with zero `rustc`.**
//!
//! # What is printed and what is asserted
//!
//! The **timings are printed, never asserted** (the house rule: no wall-clock
//! assertions). What is asserted is the *behaviour*: the mission that is running
//! after the save is the edited one, and the state it had before the save
//! survived it.
//!
//! The plumbing half of "edit → running" — the editor's watch debounce and its
//! drain tick — is `script_hot_reload`'s claim and is measured there, out of Ring
//! 2's own source. Repeating the source read here would be a second owner for one
//! number, which is the shape the SCRIPT1b audit charged for.

use std::time::Instant;

use glam::{DVec2, DVec3};

use inf_blueprint::Value;
use inf_ecs::components::Transform;
use inf_ecs::math::Vec3d;
use inf_editor_core::heist::{
    heist_dir, heist_scene, heist_script, HEIST_HERO_GUID, HEIST_HERO_HALF_H, HEIST_HERO_RADIUS,
    HEIST_HERO_START, HEIST_SCRIPT_GUID, HEIST_VAULT_AT,
};
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

/// The number the designer changes: what being watched costs, per fixed step.
const AS_SHIPPED: &str = "health.damage(entity, 12.0)";
/// …and what they change it to.
const THE_EDIT: &str = "health.damage(entity, 40.0)";

fn stand(at: (f64, f64, f64)) -> DVec3 {
    DVec3::new(at.0, at.1 + HEIST_HERO_HALF_H + HEIST_HERO_RADIUS, at.2)
}

fn place(doc: &mut SceneDoc, at: DVec3) {
    let world = doc.world_mut();
    let e = world.entity_of(HEIST_HERO_GUID).expect("the hero");
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::new(at.x, at.y, at.z);
    }
    world.mark_dirty();
}

fn var_f(session: &SimSession, name: &str) -> f64 {
    match session.actor_var(HEIST_HERO_GUID, name) {
        Some(Value::Float(f)) => *f,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn var_i(session: &SimSession, name: &str) -> i64 {
    match session.actor_var(HEIST_HERO_GUID, name) {
        Some(Value::Int(i)) => *i,
        other => panic!("`{name}` is {other:?}"),
    }
}

/// **The bank's staff, installed by hand — and the reason is a fixture fact
/// worth writing down rather than working around.**
///
/// In a cooked pack and in a PIE payload the level's own population grows from
/// `PcgVolume::residents`, which `inf_player::level::population_of` derives when
/// it builds the world (`harbour_heist_gate` measures 28 agents from the bank
/// and the flats). In the **editor** those slots are derived by the Ring-2 PCG
/// evaluation, which a Ring-1 test does not run — so a `SimSession` entered over
/// a freshly generated document has a volume with no residents in it and
/// therefore nobody at all.
///
/// `SimSession::set_crowd_population` is the documented door for a test that
/// wants a crowd (`SimSession::enter` deliberately clears one, so a Simulate
/// never leaves agents standing in the author's document), and this is
/// `script_api_surface`'s arrangement for `script_api_surface`'s reason. Three
/// people, standing well clear of the vault box so they cannot become a fact
/// about `zone.contains`.
fn install_the_staff(doc: &mut SceneDoc, session: &mut SimSession) {
    let records: std::collections::BTreeMap<uuid::Uuid, inf_ecs::crowd::CrowdRecord> = (0..3)
        .map(|i| {
            (
                uuid::Uuid::from_u128(0x5C13_7000 + i as u128),
                inf_ecs::crowd::CrowdRecord::standing(
                    inf_ecs::crowd::CrowdArchetype::default(),
                    DVec3::new(60.0 + i as f64 * 4.0, 0.0, 0.0),
                ),
            )
        })
        .collect();
    session.set_crowd_population(doc, records);
}

/// # THE ARM: a save changes a running mission.
#[test]
fn editing_the_mission_changes_the_running_simulate() {
    // The designer's working copy. **Never the committed file**: this arm edits
    // a script, and a test that edited `samples/` would leave the repository
    // dirty and the next run measuring its own last edit.
    let tmp = tempfile::tempdir().expect("a content root");
    let path = tmp.path().join("HarbourHeist.infini");
    let shipped = heist_script().expect("the mission is committed");
    std::fs::write(&path, &shipped).expect("the working copy is written");

    // Compile it the way the editor does — through the ONE file door.
    let opened = Instant::now();
    let (class, warnings) = inf_script::compile_path(&path, format!("script:{HEIST_SCRIPT_GUID}"))
        .unwrap_or_else(|d| panic!("{}", inf_script::render(&d)));
    let first_compile = opened.elapsed();
    assert!(warnings.is_empty());

    let mut doc = heist_scene();
    let mut session = SimSession::enter(
        &mut doc,
        vec![(HEIST_HERO_GUID, class)],
        DVec2::ZERO,
        SIM_HZ,
    );
    install_the_staff(&mut doc, &mut session);

    // Walk in, exactly as the gate's clean route does, and let the mission take
    // hold: the door bolts, the alarm goes down, the shelf starts paying.
    for i in 0..40 {
        place(
            &mut doc,
            stand(if i < 5 {
                HEIST_HERO_START
            } else {
                HEIST_VAULT_AT
            }),
        );
        session.step_once(&mut doc, SimInput::default());
    }
    assert_eq!(var_i(&session, "phase"), 1, "the mission is in the vault");
    assert!(
        var_i(&session, "witnesses") > 0,
        "the bank's staff have to be there, or the number the designer is about \
         to tune does nothing"
    );

    // Three steps of the shipped balance, measured off the world.
    let before = var_f(&session, "condition");
    for _ in 0..3 {
        place(&mut doc, stand(HEIST_VAULT_AT));
        session.step_once(&mut doc, SimInput::default());
    }
    let shipped_drain = (before - var_f(&session, "condition")) / 3.0;

    // ── THE EDIT ────────────────────────────────────────────────────────────
    let edited = String::from_utf8(shipped.clone())
        .expect("the mission is UTF-8")
        .replace(AS_SHIPPED, THE_EDIT);
    assert!(
        edited.contains(THE_EDIT),
        "the mission no longer spells `{AS_SHIPPED}`; move this arm's edit with it"
    );
    let bars_before = var_i(&session, "bars");
    let clock_before = var_f(&session, "clock");
    std::fs::write(&path, &edited).expect("save the edit");

    // ── the engine's half: file door → compile → swap → one fixed step ──────
    let door = Instant::now();
    let (new_class, _) = inf_script::compile_path(&path, format!("script:{HEIST_SCRIPT_GUID}"))
        .expect("the edit compiles");
    let compiled = door.elapsed();
    session.reload_class(HEIST_SCRIPT_GUID, new_class);
    assert_eq!(
        session.pending_classes(),
        1,
        "a swap is QUEUED — a save must not land inside a fixed step"
    );
    let swapped_at = Instant::now();
    place(&mut doc, stand(HEIST_VAULT_AT));
    session.step_once(&mut doc, SimInput::default());
    let swap_and_step = swapped_at.elapsed();
    let engine = door.elapsed();

    let swap = session.last_swap();
    assert_eq!(swap.units, 1);
    assert_eq!(swap.swapped, 1, "the hero took the new mission");
    assert_eq!(session.pending_classes(), 0);

    // ── the mission that is running now is the EDITED one ───────────────────
    let after = var_f(&session, "condition");
    for _ in 0..3 {
        place(&mut doc, stand(HEIST_VAULT_AT));
        session.step_once(&mut doc, SimInput::default());
    }
    let edited_drain = (after - var_f(&session, "condition")) / 3.0;
    assert!(
        edited_drain > shipped_drain * 2.5,
        "the running mission did not take the edit: it was costing {shipped_drain} \
         of condition a step and is costing {edited_drain}"
    );

    // …and the run it was in the middle of SURVIVED the save. This is the half
    // that separates hot reload from a restart: the bullion already in the bag,
    // the clock already spent, the phase already reached.
    assert_eq!(
        var_i(&session, "bars"),
        bars_before,
        "the bullion already taken must survive the save"
    );
    assert!(
        var_f(&session, "clock") < clock_before,
        "the vault clock must keep running from where it was, not restart"
    );
    assert_eq!(
        var_i(&session, "phase"),
        1,
        "the mission is still mid-heist"
    );

    println!(
        "THE ITERATION SHOWCASE, on {} bytes of committed mission ({} handlers, \
         {} functions):\n  \
         first compile (door + parse + lower): {:.2} ms\n  \
         the edit: compile {:.2} ms, swap + one fixed step {:.2} ms, ENGINE TOTAL \
         {:.2} ms, zero rustc\n  \
         the running mission changed: {:.5} -> {:.5} of condition per step, with \
         {} bars and {:.2} s of clock carried across the save",
        shipped.len(),
        2,
        6,
        first_compile.as_secs_f64() * 1000.0,
        compiled.as_secs_f64() * 1000.0,
        swap_and_step.as_secs_f64() * 1000.0,
        engine.as_secs_f64() * 1000.0,
        shipped_drain,
        edited_drain,
        var_i(&session, "bars"),
        var_f(&session, "clock"),
    );
}

/// **The committed mission is what this arm thinks it is.**
///
/// Both arms above are written against the mission's own text (`AS_SHIPPED`, and
/// the handler/function counts in the printed line). A mission edited without
/// them would leave two arms measuring a program that no longer exists, and the
/// first would still pass — it replaces a string that is not there and swaps in
/// an identical class.
#[test]
fn the_arms_above_are_about_the_committed_mission() {
    let src = String::from_utf8(heist_script().expect("committed")).expect("UTF-8");
    assert_eq!(
        src.matches(AS_SHIPPED).count(),
        1,
        "the edit this arm makes must have exactly one site in the mission"
    );
    assert!(
        heist_dir().join("HarbourHeist.infini").exists(),
        "the mission is committed content, not a fixture"
    );
}
