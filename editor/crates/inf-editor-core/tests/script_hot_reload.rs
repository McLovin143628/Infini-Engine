//! **Hot reload** (SCRIPT1b clause 2): edit a `.infini` while Simulate is
//! running and the new program takes over on the next fixed step — no `rustc`,
//! no restart, live variables kept.
//!
//! This is the P6 deferred item, landing at last, through the **interpreter**
//! rather than the dylib: *"compile-on-save hot swap in Simulate (mechanism
//! proven in `inf-hotreload`, wired at the P9 Simulate/PIE loop)"*.
//!
//! # What each arm holds
//!
//! | | |
//! |---|---|
//! | the whole loop, through a **real `AssetWatcher`** | a save on disk becomes new behaviour, and the wall clock is **printed** |
//! | queued, not applied | a swap lands at a step boundary and never inside one |
//! | state survives | live member variables keep their values across the swap |
//! | a new variable is seeded | or the first Tick after adding one dies at `vars::get` |
//! | **failure containment** | a broken edit leaves the PREVIOUS program running, and nothing is half-swapped |
//! | atomic per unit | every actor bound to the script takes the new code in one step |

use std::path::Path;
use std::time::{Duration, Instant};

use glam::DVec2;
use inf_asset::AssetWatcher;
use inf_blueprint::Value;
use inf_ecs::components::{ActorClass, Transform};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use uuid::Uuid;

/// The script a designer starts with: one member variable, counted up every
/// tick by a rate the script sets.
const V1: &str = "\
actor \"Counter\"

var count: float = 0.0
var rate: float = 1.0

on tick(dt)
  count = count + rate
end
";

/// The edit: a different rate, and a **new member variable** the surviving
/// instance has never heard of.
const V2: &str = "\
actor \"Counter\"

var count: float = 0.0
var rate: float = 10.0
var bonus: float = 100.0

on tick(dt)
  count = count + rate + bonus
end
";

/// The edit that does not compile.
const BROKEN: &str = "\
actor \"Counter\"

var count: float = 0.0

on tick(dt)
  count = count +
end
";

const ACTOR_A: Uuid = Uuid::from_u128(0x5C21_0001);
const ACTOR_B: Uuid = Uuid::from_u128(0x5C21_0002);
const SCRIPT_ASSET: Uuid = Uuid::from_u128(0x5C21_9999);

/// A document with **two** entities bound to the same script asset, so "atomic
/// per unit" is a claim over more than one actor.
fn doc_with_two_actors() -> SceneDoc {
    let mut doc = SceneDoc::new();
    for (guid, label) in [(ACTOR_A, "A"), (ACTOR_B, "B")] {
        bind(&mut doc, guid, label, SCRIPT_ASSET);
    }
    doc
}

/// Spawn an entity with a stable GUID and bind it to a script asset — the
/// persisted `ActorClass` link a level really carries (P9.5).
fn bind(doc: &mut SceneDoc, guid: Uuid, label: &str, asset: Uuid) {
    doc.create_with_guid(guid, SpawnKind::Empty, label, None);
    let e = doc.entity_of(guid).expect("the entity spawned");
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(Transform::default());
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(ActorClass(asset));
    doc.world_mut().mark_dirty();
}

fn compile(src: &str) -> inf_blueprint::BlueprintClass {
    let (class, warnings) = inf_script::compile(src, format!("script:{SCRIPT_ASSET}"))
        .unwrap_or_else(|d| panic!("{}", inf_script::render(&d)));
    assert!(warnings.is_empty(), "{warnings:?}");
    class
}

fn enter(class: &inf_blueprint::BlueprintClass) -> (SceneDoc, SimSession) {
    let mut doc = doc_with_two_actors();
    let actors = vec![(ACTOR_A, class.clone()), (ACTOR_B, class.clone())];
    let session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    (doc, session)
}

fn count(session: &SimSession, guid: Uuid) -> f64 {
    match session.actor_var(guid, "count") {
        Some(Value::Float(f)) => *f,
        other => panic!("count is {other:?}"),
    }
}

/// **The clause, end to end, through a real file watcher**: write a new
/// `.infini` over the old one while Simulate is running, and the running world
/// changes behaviour.
///
/// The edit-to-running time is **printed, never asserted** — the house rule. The
/// number a wall clock can give here is dominated by the watcher's own debounce,
/// which is a constant somebody chose, so the two halves are printed separately:
/// what the *engine* costs (door → compile → swap → step) and what the
/// *plumbing* costs on top.
#[test]
fn a_save_becomes_new_behaviour_in_a_running_simulate() {
    let tmp = tempfile::tempdir().expect("a content root");
    let path = tmp.path().join("Counter.infini");
    std::fs::write(&path, V1).expect("write v1");

    let (mut doc, mut session) = enter(&compile(V1));
    for _ in 0..5 {
        session.step_once(&mut doc, SimInput::default());
    }
    assert_eq!(count(&session, ACTOR_A), 5.0, "v1 counts by 1");

    // A real watcher on a real directory, with the editor's own debounce.
    const DEBOUNCE: Duration = Duration::from_millis(120);
    let watcher = AssetWatcher::watch(tmp.path(), DEBOUNCE).expect("the watcher starts");

    let saved = Instant::now();
    std::fs::write(&path, V2).expect("save the edit");

    // Wait for the watcher, stepping the sim the whole time — which is what an
    // editor does, and which is also what makes "the OLD program kept running"
    // observable rather than assumed.
    let mut seen = None;
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        let changes = watcher.drain();
        if let Some(c) = changes.iter().find(|c| c.path() == path) {
            seen = Some(c.path().to_path_buf());
            break;
        }
        session.step_once(&mut doc, SimInput::default());
        std::thread::yield_now();
    }
    let watched = saved.elapsed();
    let observed = seen.expect("the watcher saw the save");
    let before = count(&session, ACTOR_A);

    // ── the engine's half: the file door, the compile, the swap, one step ──
    let engine_start = Instant::now();
    let (class, _) = inf_script::compile_path(&observed, format!("script:{SCRIPT_ASSET}"))
        .expect("the edit compiles");
    session.reload_class(SCRIPT_ASSET, class);
    assert_eq!(
        session.pending_classes(),
        1,
        "a swap is QUEUED, not applied — a filesystem event must not land inside a fixed step"
    );
    assert_eq!(
        count(&session, ACTOR_A),
        before,
        "and nothing changed before the step"
    );
    session.step_once(&mut doc, SimInput::default());
    let engine = engine_start.elapsed();

    let swap = session.last_swap();
    println!(
        "edit -> running: watcher {:.0} ms (debounce {} ms), engine {:.2} ms \
         (door + compile + swap + one step); swap {swap:?}",
        watched.as_secs_f64() * 1000.0,
        DEBOUNCE.as_millis(),
        engine.as_secs_f64() * 1000.0
    );

    // Counted shapes, which is what is asserted.
    assert_eq!(swap.units, 1);
    assert_eq!(swap.swapped, 2, "both bound actors took the new code");
    assert_eq!(swap.seeded_vars, 2, "`bonus` seeded on both instances");
    assert_eq!(session.pending_classes(), 0);

    // **The new program ran, and it ran with the LIVE state.** `+101`, not
    // `+110`: the edit's new `bonus` (100) is seeded because the instance had no
    // such variable, while `rate` keeps its live `1.0` and ignores the edit's new
    // default of `10.0`.
    //
    // That asymmetry is the honest semantics of hot reload and a designer has to
    // know it: **changing a variable's DEFAULT does not change a running
    // instance.** The alternative — re-seeding every variable from the class —
    // would throw away exactly the state the feature exists to preserve.
    assert_eq!(count(&session, ACTOR_A), before + 101.0);
    assert_eq!(count(&session, ACTOR_B), before + 101.0);
    // …and the LIVE variable survived rather than resetting to its default.
    assert!(
        before >= 5.0,
        "the pre-swap count must be non-zero or 'state survived' proves nothing"
    );
    assert_eq!(
        session.actor_var(ACTOR_A, "bonus"),
        Some(&Value::Float(100.0)),
        "a variable the edit ADDED must be seeded, or the first Tick after it dies at vars::get"
    );
    assert_eq!(
        session.actor_var(ACTOR_A, "rate"),
        Some(&Value::Float(1.0)),
        "a variable that already existed keeps its LIVE value — the whole point of hot reload"
    );
}

/// **Failure containment**, at the bound the memo states: a broken edit never
/// becomes code, so the previous good program keeps running and no actor is
/// half-swapped.
#[test]
fn a_broken_edit_leaves_the_previous_program_running() {
    let tmp = tempfile::tempdir().expect("a content root");
    let path = tmp.path().join("Counter.infini");
    std::fs::write(&path, V1).expect("write v1");

    let (mut doc, mut session) = enter(&compile(V1));
    for _ in 0..3 {
        session.step_once(&mut doc, SimInput::default());
    }
    assert_eq!(count(&session, ACTOR_A), 3.0);

    // The edit that does not compile, through the same door the watcher uses.
    std::fs::write(&path, BROKEN).expect("save the broken edit");
    let diags = inf_script::compile_path(&path, "script:x").expect_err("it does not compile");
    let rendered = inf_script::render(&diags);
    println!("the editor would print:\n{rendered}");
    assert!(rendered.contains("error:"), "{rendered}");
    let place = rendered.split(':').take(2).collect::<Vec<_>>();
    assert!(
        place[0].parse::<u32>().is_ok() && place[1].parse::<u32>().is_ok(),
        "the diagnostic must open with a line and a column the designer can \
         look at: {rendered}"
    );
    assert!(
        place[0].parse::<u32>().unwrap() >= 6,
        "…and it must point at the edit, not at line 1: {rendered}"
    );

    // Nothing was queued — a class is what `reload_class` takes, and a broken
    // edit never produces one. The sim does not learn there was an edit.
    assert_eq!(session.pending_classes(), 0);
    for _ in 0..3 {
        session.step_once(&mut doc, SimInput::default());
    }
    assert_eq!(
        count(&session, ACTOR_A),
        6.0,
        "the PREVIOUS good program kept running"
    );
    assert_eq!(
        session.last_swap().swapped,
        0,
        "and nothing was half-swapped"
    );
}

/// A swap reaches only the actors bound to **that** asset.
///
/// The vacuity this closes: a swap that replaced every class in the session
/// would pass every arm above, because every actor above is bound to one script.
#[test]
fn a_swap_reaches_only_the_actors_bound_to_that_asset() {
    const OTHER: Uuid = Uuid::from_u128(0x5C21_0003);
    const OTHER_ASSET: Uuid = Uuid::from_u128(0x5C21_8888);

    let mut doc = doc_with_two_actors();
    bind(&mut doc, OTHER, "Other", OTHER_ASSET);

    let v1 = compile(V1);
    let actors = vec![
        (ACTOR_A, v1.clone()),
        (ACTOR_B, v1.clone()),
        (OTHER, v1.clone()),
    ];
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    for _ in 0..4 {
        session.step_once(&mut doc, SimInput::default());
    }

    session.reload_class(SCRIPT_ASSET, compile(V2));
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(session.last_swap().swapped, 2, "not three");
    assert_eq!(count(&session, ACTOR_A), 4.0 + 101.0);
    assert_eq!(
        count(&session, OTHER),
        5.0,
        "the actor bound to a different asset still runs v1"
    );
}

/// The watcher recognises a `.infini` by the same door the compiler does — so
/// the editor and the cook cannot disagree about which files are programs.
#[test]
fn the_watcher_and_the_compiler_agree_about_what_a_script_is() {
    assert!(inf_script::is_script_path(Path::new(
        "Content/Scripts/A.infini"
    )));
    assert_eq!(
        inf_asset::AssetKind::from_path(Path::new("Content/Scripts/A.infini")),
        inf_asset::AssetKind::Script
    );
}
