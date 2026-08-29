//! **The `engine.*` kit, driven from text at a real world** (wave SCRIPT3).
//!
//! `both_hosts_reach_the_same_ring_0_rule_for_the_engine_kit` reads source and
//! proves the arms are written and named; `inf_ecs::prefab`'s own tests prove
//! the rule. Neither can say that *dispatching* the verb from a script does what
//! the verb's description promises, and this house has a law about exactly that:
//! **a gate must aim at the thing it names**.
//!
//! So the three verbs are driven from a `.infini` script through
//! `inf_script::compile` into a real [`SimSession`], and what is asserted is the
//! **world** — an entity that is there, at a place, with a name, and then gone —
//! never a log line and never the registry that was already checked.
//!
//! The shipped host's half of the same claim is `script_spawn_gate.rs`, which
//! compares the two hosts step for step; this file is the editor's half and the
//! one that can name a `Guid`.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::Value;
use inf_ecs::components::{MeshRef, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::prefab::{authored_spawn_guid, spawn_entity_id};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

const CONJURER: u128 = 0x0005_C193_0001;
const SUICIDE: u128 = 0x0005_C193_0002;

/// Where the conjurer stands. **Not the origin**, deliberately: a spawn that
/// ignored the acting actor's place and used `Vec3d::ZERO` would pass against it.
const AT: DVec3 = DVec3::new(10.0, 2.0, -5.0);

/// The prefab name. A file stem rather than a GUID, so the placeholder half of
/// the verb's contract is what is measured here; the GUID half is
/// `inf_ecs::prefab`'s own arm and the cook's closure is `script_spawn_gate`'s.
const PREFAB: &str = "Crate";

/// **The script.** Spawn once on `BeginPlay`, ask for the same thing again every
/// tick (the idempotence claim), turn the conjurer, and banish on a key.
const CONJURE: &str = "\
actor \"Conjurer\"

var entity: int = 0
var summoned: int = 0
var again: int = 0
var ticks: float = 0.0

on begin_play()
    summoned = engine.spawn(\"Crate\")
    engine.set_rotation(90.0)
end

on tick(dt)
    ticks = ticks + 1.0
end

on input \"conjure\"(pressed)
    if pressed then
        again = engine.spawn(\"Crate\")
    end
end

on input \"banish\"(pressed)
    if pressed then
        engine.destroy(summoned)
    end
end
";

/// An actor that removes its own entity, to measure what happens to the handler
/// it did it from and to the handlers after it.
const SUICIDE_SCRIPT: &str = "\
actor \"Ghost\"

var entity: int = 0
var ticks: float = 0.0

on begin_play()
    engine.destroy(entity)
    debug.print(\"the statement after the destroy\")
end

on tick(dt)
    ticks = ticks + 1.0
end
";

fn doc_with(actors: &[(u128, &str)]) -> SceneDoc {
    let mut doc = SceneDoc::new();
    for (guid, name) in actors {
        let g = Uuid::from_u128(*guid);
        doc.create_with_guid(g, SpawnKind::Empty, name, None);
        let e = doc.entity_of(g).expect("the actor spawned");
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(AT));
    }
    doc.world_mut().propagate();
    doc
}

fn compile(src: &str, label: &str) -> inf_blueprint::BlueprintClass {
    let (class, warnings) = inf_script::compile(src, label.to_string())
        .unwrap_or_else(|d| panic!("{}", inf_script::render(&d)));
    assert!(warnings.is_empty(), "{}", inf_script::render(&warnings));
    class
}

fn var(session: &SimSession, guid: u128, name: &str) -> Value {
    session
        .actor_var(Uuid::from_u128(guid), name)
        .unwrap_or_else(|| panic!("no `{name}` on {guid:#x}"))
        .clone()
}

fn var_i(session: &SimSession, guid: u128, name: &str) -> i64 {
    match var(session, guid, name) {
        Value::Int(i) => i,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn tick(doc: &mut SceneDoc, session: &mut SimSession, input: SimInput) {
    session.tick(doc, 1.0 / SIM_HZ, input);
}

/// Press and release, so the next press is an edge again (the `script_api_surface`
/// helper, for the same reason).
fn press(doc: &mut SceneDoc, session: &mut SimSession, key: &str) {
    tick(doc, session, SimInput::with_down([key]));
    tick(doc, session, SimInput::default());
}

/// # THE ARM: a script puts a thing in the world, turns itself, and takes the
/// thing away again.
#[test]
fn the_engine_kit_spawns_turns_and_destroys_a_real_entity() {
    let mut doc = doc_with(&[(CONJURER, "Conjurer")]);
    let actors = vec![(
        Uuid::from_u128(CONJURER),
        compile(CONJURE, "script:conjure"),
    )];
    let before = doc.world_mut().entities().len();
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);

    tick(&mut doc, &mut session, SimInput::default());

    // ── the spawn ────────────────────────────────────────────────────────
    //
    // The identity is derived here rather than searched for: the arm asserts
    // that the entity in the world is the one the CONTENT names, so a spawn that
    // minted a fresh `Uuid::new_v4()` (which would look identical from a count)
    // fails.
    let guid = authored_spawn_guid(PREFAB, Vec3d::from(AT));
    let spawned = doc
        .world()
        .entity_of(guid)
        .expect("the script's spawn is in the world, under its content-derived guid");
    assert_eq!(doc.world().name_of(spawned), Some(PREFAB));
    assert_eq!(
        doc.world()
            .world()
            .get::<Transform>(spawned)
            .expect("a transform")
            .translation,
        Vec3d::from(AT),
        "a spawn lands at the ACTING actor's place, not at the origin"
    );
    assert!(
        doc.world().world().get::<MeshRef>(spawned).is_some(),
        "a spawned thing must be drawable, or a designer spawns something \
         invisible and thinks the verb did nothing"
    );
    assert_eq!(
        var_i(&session, CONJURER, "summoned"),
        spawn_entity_id(guid),
        "the handle a script holds is folded from the identity, so both hosts \
         hand the same program the same number"
    );
    assert!(
        var_i(&session, CONJURER, "summoned") > 1_000_000,
        "a spawned handle must not look like an authored actor's 1..=n id"
    );

    // ── the turn ─────────────────────────────────────────────────────────
    let actor = doc
        .world()
        .entity_of(Uuid::from_u128(CONJURER))
        .expect("actor");
    assert_eq!(
        doc.world()
            .world()
            .get::<Transform>(actor)
            .expect("a transform")
            .rotation,
        Vec3d::new(0.0, 90.0, 0.0),
        "`engine.set_rotation` writes the ACTING entity's yaw in degrees"
    );

    // ── idempotence ──────────────────────────────────────────────────────
    //
    // Ask for the same prefab at the same place five more times. Two spawns of
    // one thing in one place are ONE entity — the pickup kit's ruling — so the
    // world must not grow, and the handle must not move.
    for _ in 0..5 {
        press(&mut doc, &mut session, "conjure");
    }
    assert_eq!(
        var_i(&session, CONJURER, "again"),
        spawn_entity_id(guid),
        "a repeated spawn answers the same handle"
    );
    assert_eq!(
        doc.world_mut().entities().len(),
        before + 1,
        "six spawns of one prefab at one place made more than one entity"
    );

    // ── the banish ───────────────────────────────────────────────────────
    press(&mut doc, &mut session, "banish");
    assert!(
        doc.world().entity_of(guid).is_none(),
        "`engine.destroy` must take the entity out of the world"
    );
    assert_eq!(doc.world_mut().entities().len(), before);

    // …and the same content spawns the same entity back: one guid, not a second
    // one beside it, which is what a spawn keyed on a counter would have made.
    press(&mut doc, &mut session, "conjure");
    assert_eq!(doc.world_mut().entities().len(), before + 1);
    assert!(doc.world().entity_of(guid).is_some());
    assert_eq!(var_i(&session, CONJURER, "again"), spawn_entity_id(guid));
}

/// **An actor that destroys itself finishes its handler and then stops.**
///
/// Two claims, and the pair is the point. Containment (appendix A.7) says a
/// handler runs to its end, so the statement after the `engine.destroy` must
/// still have run — it is a `debug.print`, because a member variable written
/// after it would be unreadable for the same reason the actor is gone.
/// Lifecycle says an entity that is gone stops being ticked, so the `Tick`
/// handler must never run at all.
#[test]
fn an_actor_that_destroys_itself_finishes_the_handler_and_stops() {
    let mut doc = doc_with(&[(SUICIDE, "Ghost")]);
    let actors = vec![(
        Uuid::from_u128(SUICIDE),
        compile(SUICIDE_SCRIPT, "script:ghost"),
    )];
    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);

    for _ in 0..4 {
        tick(&mut doc, &mut session, SimInput::default());
    }

    assert!(
        doc.world().entity_of(Uuid::from_u128(SUICIDE)).is_none(),
        "the ghost's own entity must be gone"
    );
    assert!(
        session
            .logs()
            .iter()
            .any(|l| l == "the statement after the destroy"),
        "the handler must run to its end — a destroy that aborted the handler \
         would be a containment rule this arc does not have: {:?}",
        session.logs()
    );
    assert!(
        session
            .actor_var(Uuid::from_u128(SUICIDE), "ticks")
            .is_none(),
        "a destroyed actor must stop being an actor; it is still in the session's \
         map, so its `Tick` is still running against a world that no longer has it"
    );
}
