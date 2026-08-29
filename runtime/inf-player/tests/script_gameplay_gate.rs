//! **PIE == shipping, from a `.infini`** (SCRIPT1b clause 5).
//!
//! Real gameplay — an item catalogue defined on `BeginPlay` and a pickup handed
//! out on a timer — authored as **text**, cooked, and run twice: once off a
//! `.inf_pack` the way a shipped build boots, and once off the `ScenePayload`
//! the editor really builds for PIE. The two traces must be identical step for
//! step.
//!
//! # Why this arm exists rather than being assumed
//!
//! `.infini` adds a **third producer** of the one IR, and the memo's law is
//! explicit that the arc owes this: *"`PIE == shipping` over a trace whose
//! gameplay runs from a `.infini` script is a SCRIPT1 gate arm, not a SCRIPT3
//! aspiration."* The failure it exists to catch is a divergence in the *door*
//! rather than in the program: the cook lowers a script and the PIE payload
//! resolves a class, and if those two ever stop being the same lowering, both
//! sides go on working and disagree.
//!
//! # What "shipping" reads, and what it deliberately does not
//!
//! The shipped player **never reads a `.infini`**. The cook lowers it and packs
//! the `BlueprintClass` under `AssetKind::Script`, so the player decodes a class
//! exactly as it does for a `.inf_act` — no lexer, no parser, no `inf-script` in
//! the shipped binary. That is the ship decision `cook_script.rs` states, and
//! this gate is where it is *proven* rather than described.
//!
//! It follows that there is no loose-directory `.infini` path to compare
//! against: `load_actor_classes_by_guid_from_dir` reads `.inf_act` and only
//! `.inf_act`. Adding a parser to the player to make a "loose" arm possible
//! would falsify the sentence above to gate it, which is the wrong trade.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_ecs::components::{ActorClass, Transform};
use inf_player::runtime_sim::RuntimeSim;
use inf_project::ProjectManifest;
use uuid::Uuid;

const HZ: u32 = 60;
const STEPS: usize = 90;

const ACTOR_GUID: Uuid = Uuid::from_u128(0x5C15_0001);
const SCRIPT_GUID: Uuid = Uuid::from_u128(0x5C15_0002);
const LEVEL_GUID: Uuid = Uuid::from_u128(0x5C15_0003);

/// **The gameplay, in text.** A catalogue defined once, a pickup put on the
/// ground, and a crate handed to the actor every quarter second — spawn-on-
/// trigger, with the trigger being the script's own clock.
///
/// Every effect here lands in the **world** rather than in a log: the catalogue
/// is a resource, the pickup is an entity, the inventory is a component. So the
/// two hosts are compared on what they simulated, not on what they reported —
/// P21's *assert the WORLD, not the report*.
const SCRIPT: &str = "\
actor \"Quartermaster\"

var elapsed: float = 0.0
var handed: float = 0.0
var carried: float = 0.0

on begin_play()
  item.define(\"[crate]\\nlabel = \\\"Crate\\\"\\nstack_max = 9\\nmass_kg = 1.0\\n\")
  item.spawn_pickup(\"crate\", 2.0, 0.5, 3.0, 2)
end

on tick(dt)
  elapsed = elapsed + dt
  if elapsed > 0.25 then
    elapsed = 0.0
    handed = handed + 1.0
    item.give(var.get(\"entity\"), \"crate\", 1)
  end
  carried = math.to_float(item.count(var.get(\"entity\"), \"crate\"))
  health.set(var.get(\"entity\"), 100.0 - handed * 7.0)
end
";

/// The level: one entity, bound to the SCRIPT by GUID through the persisted
/// [`ActorClass`] link (P9.5) — the same binding a `.inf_act` uses, which is the
/// point.
fn scene() -> inf_editor_core::scene::SceneDoc {
    use inf_editor_core::ipc::SpawnKind;
    let mut doc = inf_editor_core::scene::SceneDoc::new();
    doc.create_with_guid(ACTOR_GUID, SpawnKind::Empty, "Quartermaster", None);
    let e = doc.entity_of(ACTOR_GUID).expect("the actor spawned");
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(Transform::default());
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(ActorClass(SCRIPT_GUID));
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

/// A project on disk: the manifest, the boot level under `Content/Levels/`, and
/// the script under `Content/Scripts/` — the SCRIPT1b layout ruling, written the
/// way `inf new` writes it.
fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Quartermaster", "blank-3d")
        .save(&proj)
        .expect("the manifest saves");
    let content = proj.join("Content");
    std::fs::create_dir_all(content.join("Levels")).expect("levels root");
    std::fs::create_dir_all(content.join("Scripts")).expect("scripts root");

    inf_editor_core::scene::serialize::save(
        &scene(),
        &content.join("Levels/Main.inf_lvl"),
        Some(LEVEL_GUID),
    )
    .expect("the level saves");

    // The script, with a sidecar pinning its GUID — the one the level's
    // `ActorClass` names. Without it the database would synthesise a
    // content-hash GUID and the binding would resolve to nothing.
    let path = content.join("Scripts/Quartermaster.infini");
    std::fs::write(&path, SCRIPT).expect("the script saves");
    let mut side = AssetSidecar::new(
        AssetId(SCRIPT_GUID),
        AssetKind::Script,
        ContentHash::of(SCRIPT.as_bytes()),
    );
    side.normalize();
    side.save(&path).expect("the sidecar saves");
    proj
}

fn cook(proj: &Path, out: &Path) -> inf_packager::CookReport {
    inf_packager::cook(proj, out, &inf_packager::CookOptions::default()).expect("the fixture cooks")
}

/// **The shipping side**: a sim off the cooked pack, the way `--pack` boots.
fn pack_sim(out: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(out).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the world builds");
    inf_player::sim_from_built(built)
}

/// **The PIE side**: the payload the editor really builds, resolving the script
/// asset through the SAME class closure a `.inf_act` uses — which is the whole
/// reason `build_scene_payload` needed no new parameter.
fn pie_sim() -> RuntimeSim {
    let doc = scene();
    let (class, warnings) =
        inf_script::compile(SCRIPT, format!("script:{}", AssetId(SCRIPT_GUID))).expect("compiles");
    assert!(warnings.is_empty(), "{warnings:?}");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == SCRIPT_GUID).then(|| class.clone()),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        HZ,
        false,
    )
    .expect("the payload builds");
    // Non-vacuity at the payload, before anything is compared: a payload with no
    // class boots a world where nothing was authored, and the two hosts would
    // agree perfectly about it.
    assert_eq!(
        payload.classes.len(),
        1,
        "the script's class must ride the wire"
    );
    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
}

/// One step of the trace: the whole sim state, plus the three member variables
/// as **bit patterns**.
///
/// `state_bytes` covers the world — the pickup entity, the inventory component,
/// the catalogue — and Blueprint `vars` are deliberately **not** in it, so a
/// script's own state has to be traced explicitly (the phase22 gate's
/// `var_bits`, met again for the reason it was written).
#[derive(Clone, PartialEq, Debug)]
struct Frame {
    state: u64,
    vars: [u64; 3],
}

fn var_bits(sim: &RuntimeSim, name: &str) -> u64 {
    match sim.actor_var(ACTOR_GUID, name) {
        Some(inf_blueprint::Value::Float(f)) => f.to_bits(),
        Some(inf_blueprint::Value::Int(i)) => *i as u64,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn run_trace(sim: &mut RuntimeSim) -> Vec<Frame> {
    (0..STEPS)
        .map(|_| {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            let state = inf_player::step_state_hash(sim);
            Frame {
                state,
                vars: [
                    var_bits(sim, "elapsed"),
                    var_bits(sim, "handed"),
                    var_bits(sim, "carried"),
                ],
            }
        })
        .collect()
}

/// **Anti-vacuity**: two worlds where nothing happened are identical. The trace
/// has to *move*, and it has to move in the way the script says.
fn assert_not_vacuous(trace: &[Frame]) {
    let states: BTreeSet<u64> = trace.iter().map(|f| f.state).collect();
    let vars: BTreeSet<[u64; 3]> = trace.iter().map(|f| f.vars).collect();
    println!(
        "trace: {} distinct world states, {} distinct variable tuples, over {STEPS} steps",
        states.len(),
        vars.len()
    );
    // **Two different claims, and both are needed.** The variables move every
    // step because the script's clock does; the WORLD moves only when the
    // script writes to it, and a gate that only watched the variables would
    // certify a script whose gameplay verbs had all become no-ops.
    assert!(
        vars.len() > STEPS / 2,
        "only {} distinct variable tuples — the script is not running",
        vars.len()
    );
    assert!(
        states.len() >= 6,
        "only {} distinct world states — the script ran but changed NOTHING in          the world, which is the vacuous shape this gate exists to refuse",
        states.len()
    );
    let handed = f64::from_bits(trace.last().expect("a trace").vars[1]);
    let carried = f64::from_bits(trace.last().expect("a trace").vars[2]);
    // 90 steps at 60 Hz is 1.5 s; a hand-out every 0.25 s is five of them.
    assert_eq!(handed, 5.0, "the script handed out {handed} crates, not 5");
    assert_eq!(
        carried, handed,
        "the actor is carrying {carried} of the {handed} it was handed — the \
         inventory is not the world the script thinks it is"
    );
}

/// # THE ARM: `PIE == shipping`, over a trace whose gameplay is a `.infini`.
#[test]
fn pie_equals_shipping_on_a_script_driven_trace() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = scaffold(tmp.path());
    let out = tmp.path().join("out");
    let report = cook(&proj, &out);
    println!("{}", report.render());
    assert!(!report.has_blocking(), "{:?}", report.blocking);
    assert_eq!(report.kinds.get("script"), Some(&1), "{:?}", report.kinds);

    let mut ship = pack_sim(&out);
    let mut pie = pie_sim();
    let ship_trace = run_trace(&mut ship);
    let pie_trace = run_trace(&mut pie);

    assert_not_vacuous(&ship_trace);
    assert_not_vacuous(&pie_trace);

    // **Per step, not one comparison of two vectors.** The failure this exists
    // to catch is a divergence POINT, and "which step" is most of the diagnosis
    // (the phase22 gate's rule).
    for (i, (s, p)) in ship_trace.iter().zip(pie_trace.iter()).enumerate() {
        assert_eq!(s.state, p.state, "state hash diverged at step {i}");
        assert_eq!(s.vars, p.vars, "script variables diverged at step {i}");
    }
    assert_eq!(ship_trace.len(), pie_trace.len());
    println!(
        "PIE == shipping over {STEPS} steps of script-driven gameplay; final handed={} carried={}",
        f64::from_bits(ship_trace[STEPS - 1].vars[1]),
        f64::from_bits(ship_trace[STEPS - 1].vars[2])
    );
}

/// **The cooked artifact is the same on every host** — the cross-host lowering
/// hash, extended from the IR to the bytes that ship.
///
/// A pinned digest, not a comparison of two cooks: CI runs this on three
/// operating systems, and a lowering that depended on the host reddens exactly
/// one leg with a number rather than a mystery. Hand-rolled FNV-1a for the
/// reason `inf-script`'s own determinism gate gives — `DefaultHasher` is
/// documented as unspecified, so a constant pinned against it says something
/// about a toolchain and nothing about a host.
#[test]
fn the_cooked_script_artifact_is_byte_identical_on_every_host() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = scaffold(tmp.path());
    let out = tmp.path().join("out");
    cook(&proj, &out);

    let reader = inf_asset::PackReader::open(&out.join(inf_packager::DEFAULT_PACK_NAME))
        .expect("the pack opens");
    let entry = reader
        .index()
        .find(|e| e.kind == AssetKind::Script)
        .expect("a script entry");
    assert_eq!(entry.guid.0, SCRIPT_GUID, "the sidecar's GUID must survive");
    let bytes = reader.read(entry.guid).expect("the blob reads");

    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in &bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    println!("cooked script: {} bytes, FNV-1a {h:#018x}", bytes.len());
    assert_eq!(
        h, 0xe349_9f72_abc1_1769,
        "the cooked artifact moved. If the LANGUAGE changed this is expected and \
         the pin is re-blessed with the reason; if only the HOST changed, the \
         lowering is not host-independent and that is the bug."
    );
}

/// **`engine.spawn` is the only asset-naming verb, and neither host implements
/// it** — a tripwire under the PIE payload's dependency closure.
///
/// The cook walks a script's asset references (`inf_blueprint::asset_refs`) and
/// pulls what it names into the pack. `build_scene_payload` does **not** — it is
/// a hand-maintained mirror of `asset_deps`, and this edge is not in it. That is
/// safe today for a measurable reason and not for a hopeful one: `engine.spawn`
/// reaches neither host's `Host::call`, so a script cannot spawn anything in
/// either PIE or shipping, so there is nothing for a payload to be missing.
///
/// The day somebody implements it, this arm goes red and says what it owes.
#[test]
fn nothing_a_script_names_can_reach_a_world_yet() {
    let assets = inf_blueprint::assetrefs::STR_PORTS
        .iter()
        .filter(|(_, _, r)| *r == inf_blueprint::StrRole::Asset)
        .map(|(t, p, _)| format!("{t}.{p}"))
        .collect::<Vec<_>>();
    assert_eq!(
        assets,
        ["engine.spawn.prefab"],
        "a new asset-naming verb arrived. `inf_packager::asset_deps` walks it; \
         `inf_editor_core::pie::build_scene_payload` does NOT, and a PIE payload \
         that is missing what a cooked pack carries is precisely the divergence \
         this file exists to prevent. Mirror the edge, or factor one Ring-0 walk."
    );

    // …and the verb is inert in the shipped host: a class that calls it runs,
    // logs the unknown path and changes nothing.
    let mut sim = pie_sim();
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    assert!(
        !sim.logs().iter().any(|l| l == "engine::spawn"),
        "the fixture does not call it"
    );
}
