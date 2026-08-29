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
//! | a **removed** or **retyped** variable | the other two edits, which the wave stated nothing about — measured and sentenced by the SCRIPT1b audit |
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

/// The editor's **real** watcher constants, read out of Ring 2's own source.
///
/// # The SCRIPT1b audit's finding, and why this is read rather than written
///
/// The first version of this arm chose `Duration::from_millis(120)` and called
/// it *"the editor's own debounce"*, and the wave's ledger, `infiniscript-
/// direction.md` §8 and the ROADMAP all repeated **"121 ms, of which 120 ms is
/// the watcher's own debounce"**. Both halves of that sentence were wrong about
/// the editor:
///
/// * `commands/assets.rs` sets `WATCH_DEBOUNCE` to **250 ms**, not 120;
/// * **120 ms is `TICK`** — the interval at which the asset thread *drains* the
///   watcher — so the two constants had been conflated.
///
/// The honest editor number is therefore the debounce **plus up to one drain
/// tick**: `WATCH_DEBOUNCE ..= WATCH_DEBOUNCE + TICK`. The engine's half —
/// door, compile, swap, one fixed step — is the part this repository controls
/// and the only part worth a sub-millisecond figure.
///
/// Read from the file instead of duplicated so the printed number cannot go
/// stale the day somebody tunes either constant. Ring 2 is not a dependency of
/// this crate and must not become one; this is a **source read**, the same
/// device `player_has_no_tuning_door` uses to make a claim about a tree it does
/// not link.
fn editor_watch_constants() -> (Duration, Duration) {
    let src =
        std::fs::read_to_string(repo_root().join("editor/studio/src-tauri/src/commands/assets.rs"))
            .expect("the editor's asset command module is committed");
    (ms_const(&src, "WATCH_DEBOUNCE"), ms_const(&src, "TICK"))
}

fn repo_root() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        p.pop();
    }
    p
}

fn ms_const(src: &str, name: &str) -> Duration {
    let needle = format!("const {name}: Duration");
    let line = src
        .lines()
        .find(|l| l.trim_start().starts_with(&needle))
        .unwrap_or_else(|| {
            panic!("`{name}` is no longer a `Duration` constant in commands/assets.rs")
        });
    let ms: u64 = line
        .split("from_millis(")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or_else(|| panic!("could not read `{name}` out of: {line}"));
    Duration::from_millis(ms)
}

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
///
/// The plumbing half is the **editor's own** constants, read out of Ring 2's
/// source ([`editor_watch_constants`]) rather than invented here — the SCRIPT1b
/// audit's finding: a number this arm chooses is a fact about this arm, and the
/// wave reported it as a fact about the editor.
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

    // A real watcher on a real directory, with the editor's own debounce — the
    // value `commands/assets.rs` really passes, read out of it.
    let (debounce, drain_tick) = editor_watch_constants();
    let watcher = AssetWatcher::watch(tmp.path(), debounce).expect("the watcher starts");

    let saved = Instant::now();
    std::fs::write(&path, V2).expect("save the edit");

    // Wait for the watcher, stepping the sim the whole time — which is what an
    // editor does, and which is also what makes "the OLD program kept running"
    // observable rather than assumed.
    //
    // **Both sides canonicalized, at one place** (the SCRIPT1b CI-red fix; the
    // same exposure reddened the queue's own watcher arm on macOS, see
    // `assets::queue`'s `a_changed_script_reaches_the_tick_outcome_with_its_guid`):
    // a tempdir handed out as `/var/folders/…` is reported
    // by the watcher as `/private/var/folders/…` through macOS's `/var` →
    // `/private/var` symlink, so matching the raw tempdir path finds nothing
    // there and this arm would time out at its 20 s deadline. It is the file
    // that has to match, not the spelling.
    let want = std::fs::canonicalize(&path).expect("the saved file is on disk");
    let mut seen = None;
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        let changes = watcher.drain();
        if let Some(c) = changes
            .iter()
            .find(|c| std::fs::canonicalize(c.path()).is_ok_and(|p| p == want))
        {
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
        "edit -> seen by this arm: {:.0} ms at the editor's WATCH_DEBOUNCE of {} ms. \
         In the editor the asset thread DRAINS the watcher every TICK = {} ms, so \
         edit -> seen there is {}..={} ms; the swap then lands on the next fixed step \
         (<= {:.1} ms at {SIM_HZ} Hz). ENGINE half (door + compile + swap + one step): \
         {:.2} ms, zero rustc. swap {swap:?}",
        watched.as_secs_f64() * 1000.0,
        debounce.as_millis(),
        drain_tick.as_millis(),
        debounce.as_millis(),
        debounce.as_millis() + drain_tick.as_millis(),
        1000.0 / SIM_HZ,
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

/// **The other two edits, which the wave stated nothing about** (SCRIPT1b audit).
///
/// The seeding rule is one-directional: a variable the edit **added** is seeded,
/// and nothing else is touched. The wave armed and sentenced that half and left
/// the other two shapes an author can reach in one keystroke unmeasured, so
/// here they are, as behaviour rather than as an intention.
///
/// **A variable the edit REMOVED keeps its live value on the instance.** The
/// class stops declaring it, the map keeps it, and nothing prunes it. That is
/// deliberate — pruning is the same operation as discarding live state, which is
/// what hot reload exists not to do — but it has a consequence a designer will
/// meet: *remove a variable, save, put it back with a different default, and the
/// instance still holds the old value*, because by then it is no longer "a
/// variable the edit added". The remedy is the one the feature already has:
/// leave Simulate and re-enter, which rebuilds instances from the class.
///
/// **A variable whose TYPE changed is not migrated.** The instance keeps the
/// `Value` it holds, so the first handler that reads it under the new type gets
/// a `RunError::Type` — and that lands exactly on §4's stated bound: the handler
/// dies, on that actor, for that tick, and the sim keeps running. It is *not*
/// silent: the actor simply stops progressing, which is the observable this arm
/// pins so the day somebody adds coercion the arm says so.
#[test]
fn a_removed_variable_lingers_and_a_retyped_one_stops_its_handler() {
    // ── removed ──────────────────────────────────────────────────────────────
    const NO_RATE: &str = "\
actor \"Counter\"

var count: float = 0.0

on tick(dt)
  count = count + 2.0
end
";
    let (mut doc, mut session) = enter(&compile(V1));
    for _ in 0..3 {
        session.step_once(&mut doc, SimInput::default());
    }
    session.reload_class(SCRIPT_ASSET, compile(NO_RATE));
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(session.last_swap().swapped, 2);
    assert_eq!(session.last_swap().seeded_vars, 0, "nothing was ADDED");
    assert_eq!(count(&session, ACTOR_A), 5.0, "the new program runs");
    assert_eq!(
        session.actor_var(ACTOR_A, "rate"),
        Some(&Value::Float(1.0)),
        "a variable the edit REMOVED lingers on the instance with its live \
         value — hot reload prunes nothing, because pruning is discarding state"
    );

    // …and putting it back with a NEW default does not re-seed it, because by
    // then the instance already has it. This is the consequence worth telling a
    // designer about rather than letting them find.
    const RATE_BACK: &str = "\
actor \"Counter\"

var count: float = 0.0
var rate: float = 99.0

on tick(dt)
  count = count + rate
end
";
    session.reload_class(SCRIPT_ASSET, compile(RATE_BACK));
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(session.last_swap().seeded_vars, 0, "it was never missing");
    assert_eq!(
        count(&session, ACTOR_A),
        6.0,
        "the lingering 1.0 wins over the re-declared default of 99.0"
    );

    // ── retyped ──────────────────────────────────────────────────────────────
    const RETYPED: &str = "\
actor \"Counter\"

var count: bool = false
var rate: float = 1.0

on tick(dt)
  if count then
    rate = rate + 1.0
  end
end
";
    let before_rate = session.actor_var(ACTOR_A, "rate").cloned();
    session.reload_class(SCRIPT_ASSET, compile(RETYPED));
    for _ in 0..3 {
        session.step_once(&mut doc, SimInput::default());
    }
    assert_eq!(
        session.last_swap().seeded_vars,
        0,
        "`count` is present, so a TYPE change seeds nothing — the map is untyped"
    );
    assert!(
        matches!(session.actor_var(ACTOR_A, "count"), Some(Value::Float(_))),
        "the instance still holds the OLD type: {:?}",
        session.actor_var(ACTOR_A, "count")
    );
    assert_eq!(
        session.actor_var(ACTOR_A, "rate").cloned(),
        before_rate,
        "the handler died at `if count` (RunError::Type) every tick, so nothing \
         after it ran — §4's bound, on the actor, for the tick. If this ever \
         changes, a type migration was added and this ledger entry is retired"
    );
    // …and the sim itself is fine: the other actor's handler dies the same way
    // and the session keeps stepping, which is the containment claim.
    assert_eq!(session.last_swap().swapped, 2);
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
