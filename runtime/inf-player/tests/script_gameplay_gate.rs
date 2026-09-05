//! **PIE == shipping, from a `.infini`** (SCRIPT1b clause 5).
//!
//! Real gameplay — an item catalogue defined on `BeginPlay` and a pickup handed
//! out on a timer — authored as **text**, cooked, and run twice: once off a
//! `.ipack` the way a shipped build boots, and once off the `ScenePayload`
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
  health.set(var.get(\"entity\"), remaining(handed))
end

-- **The call form** (wave SCRIPT2), on the ship path. A unit-local function is
-- a one-segment `Expr::Call` and needed no IR change, so nothing about the cook
-- or the pack had to learn it — which is a claim worth *crossing* the cook
-- rather than asserting, because the failure it would have is silent: a class
-- that lowered one way in the editor and another in the pack still runs on both
-- sides and disagrees.
function remaining(given: float) -> float
  return math.max(100.0 - given * 7.0, 0.0)
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
        "only {} distinct world states — the script ran but changed NOTHING in \
         the world, which is the vacuous shape this gate exists to refuse",
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
    // Re-blessed once, in wave SCRIPT2, for a stated reason: the fixture grew a
    // `function remaining(given) -> float` and a call to it, so the class the
    // cook packs holds one more `BlueprintFn` and one fewer inline expression.
    // The LANGUAGE changed, which is the half of this assertion's own message
    // that expects a new number; a move with the fixture unchanged would mean
    // the lowering had started depending on the host.
    assert_eq!(
        h, 0xd269_f486_f2b4_3cf4,
        "the cooked artifact moved. If the LANGUAGE changed this is expected and \
         the pin is re-blessed with the reason; if only the HOST changed, the \
         lowering is not host-independent and that is the bug."
    );
}

/// **`engine.spawn` is the only asset-naming verb — and since wave SCRIPT3 both
/// hosts implement it, so the PIE payload's asset edge fell due.**
///
/// This arm was a **tripwire** under `build_scene_payload`'s dependency closure:
/// the cook walks a script's `inf_blueprint::asset_refs` and pulls what it names
/// into the pack, the payload builder is a hand-maintained mirror of that walk
/// and did not, and the whole thing was safe for one measured reason — *no host
/// implemented `engine.spawn`*, so a script could not put anything in a world in
/// either PIE or shipping and there was nothing for a payload to be missing. The
/// arm's message said what to do the day that changed: *"mirror the edge, or
/// factor one Ring-0 walk."*
///
/// SCRIPT3 changed it, the edge is mirrored (`pie.rs`, the SCRIPT3 block beside
/// the skeletal refs), and the arm is turned round to face the other way. What it
/// measures now:
///
/// 1. **`engine.spawn.prefab` is still the only `StrRole::Asset` port** — a new
///    one arriving is a new edge, and the payload mirror would owe it too;
/// 2. **the verb is LIVE in the shipped host**: a script that calls it puts an
///    entity in the world, at its own place, under the content-derived GUID;
/// 3. **the handler runs past it**, which was true when the call was inert and
///    has to stay true now that it is not;
/// 4. **and the payload really walks the edge** — added by the SCRIPT3 audit,
///    because points 1–3 are all about the *world* and this arm's name promises
///    the payload too. `build_scene_payload`'s SCRIPT3 walk resolves only a
///    **GUID-spelled** prefab (a `PackEntry` carries no name), every fixture in
///    the tree spells a **stem**, and so the walk's resolver had never been
///    called by any test: deleting the loop reddened nothing anywhere. The
///    fourth block below spells the GUID and asserts the mesh rides.
#[test]
fn what_a_script_names_reaches_the_world_and_the_payload_walks_it() {
    let assets = inf_blueprint::assetrefs::STR_PORTS
        .iter()
        .filter(|(_, _, r)| *r == inf_blueprint::StrRole::Asset)
        .map(|(t, p, _)| format!("{t}.{p}"))
        .collect::<Vec<_>>();
    assert_eq!(
        assets,
        ["engine.spawn.prefab"],
        "a new asset-naming verb arrived. `inf_packager::asset_deps` walks it and \
         so does `inf_editor_core::pie::build_scene_payload` — check the second \
         one covers this port too, or a PIE payload is missing what a cooked pack \
         carries, which is precisely the divergence this file exists to prevent."
    );

    const SPAWNER: &str = "\
on begin_play()
  engine.spawn(\"Coyote\")
  debug.print(\"the statement after the spawn\")
end
";
    let doc = scene();
    let (class, _) = inf_script::compile(SPAWNER, format!("script:{}", AssetId(SCRIPT_GUID)))
        .expect("the spawner compiles");
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
        |_| None,
        HZ,
        false,
    )
    .expect("the payload builds");
    let mut sim = inf_player::sim_from_payload(&payload)
        .expect("the world builds")
        .sim;
    // `BeginPlay` runs when the sim is BUILT, not on the first step -- so the
    // spawn has already happened here, and the count below is the whole world
    // rather than a delta. (Measuring a delta across `step_once` would have read
    // zero and looked like a verb that does nothing, which is the answer this
    // arm used to assert.)
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    println!("the spawner's host log: {:?}", sim.logs());

    // The actor sits at the origin (the fixture's `Transform::default()`), so
    // that is where its spawn lands — and the identity is derived here rather
    // than searched for, so a spawn that minted a fresh GUID fails.
    let spawned = inf_ecs::prefab::authored_spawn_guid("Coyote", inf_ecs::math::Vec3d::ZERO);
    assert!(
        sim.world_mut().entity_of(spawned).is_some(),
        "`engine.spawn` is implemented in the shipped host now; a script that \
         names a prefab must put THAT entity in the world: {:?}",
        sim.logs()
    );
    assert_eq!(
        sim.world_mut().entities().len(),
        2,
        "the world is the actor plus its one spawn: {:?}",
        sim.world_mut()
            .entities()
            .into_iter()
            .map(|e| sim.world_mut().name_of(e).unwrap_or("?").to_string())
            .collect::<Vec<_>>()
    );
    assert!(
        !sim.logs().iter().any(|l| l == "engine::spawn"),
        "the verb reached the unknown-call logger, so it is dispatched by nobody \
         again: {:?}",
        sim.logs()
    );
    // Anti-vacuity, unchanged in purpose from when the call was inert: the
    // handler did not stop at the spawn.
    assert!(
        sim.logs()
            .iter()
            .any(|l| l == "the statement after the spawn"),
        "the handler stopped at the spawn: {:?}",
        sim.logs()
    );

    // ── (4) …AND THE PAYLOAD WALKS IT ───────────────────────────────────────
    //
    // **The half this arm's NAME promised and did not measure** (the SCRIPT3
    // audit). `build_scene_payload`'s SCRIPT3 walk resolves a class's
    // `asset_refs` and pulls what they name into `meshes`, mirroring
    // `inf_packager::asset_deps`. But **only a GUID-spelled prefab resolves** —
    // a `PackEntry` carries no name — and every fixture in the tree spells a
    // STEM: the spawner above, and the committed Harbour Heist mission. So the
    // walk's collector ran and its resolver never did, in any test, and deleting
    // the whole loop reddened nothing. This is the fixture that spells the GUID.
    const MESH_GUID: Uuid = Uuid::from_u128(0x5C15_0004);
    let named_src = format!("on begin_play()\n  engine.spawn(\"{MESH_GUID}\")\nend\n");
    let (named_class, _) =
        inf_script::compile(&named_src, format!("script:{}", AssetId(SCRIPT_GUID)))
            .expect("the GUID-spelling spawner compiles");
    let mesh_bytes = inf_asset::encode(&one_triangle()).expect("the mesh encodes");
    let mut asked: Vec<Uuid> = Vec::new();
    let payload = inf_editor_core::pie::build_scene_payload(
        &scene(),
        |guid| (guid == SCRIPT_GUID).then(|| named_class.clone()),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |guid| {
            asked.push(guid);
            (guid == MESH_GUID).then(|| mesh_bytes.clone())
        },
        |_| None,
        |_| None,
        HZ,
        false,
    )
    .expect("the payload builds");
    assert_eq!(
        payload.meshes.iter().map(|(g, _)| *g).collect::<Vec<_>>(),
        vec![MESH_GUID],
        "`build_scene_payload` must pull what a PROGRAM names into the payload, the \
         way `asset_deps` pulls it into the pack. A PIE session missing what a cooked \
         pack carries is precisely the divergence this file exists to prevent. The \
         mesh resolver was asked for: {asked:?}"
    );
    // …and the spawned entity binds it, so the bytes have an entity to be about.
    let mut bound = inf_player::sim_from_payload(&payload)
        .expect("the world builds")
        .sim;
    let g =
        inf_ecs::prefab::authored_spawn_guid(&MESH_GUID.to_string(), inf_ecs::math::Vec3d::ZERO);
    let e = bound
        .world_mut()
        .entity_of(g)
        .expect("the GUID-spelled spawn is in the world");
    assert_eq!(
        bound
            .world_mut()
            .world()
            .get::<inf_ecs::components::MeshRef>(e)
            .expect("a mesh ref")
            .asset,
        Some(MESH_GUID),
        "a GUID-spelled prefab binds its asset — that is the half of the verb's \
         contract the payload edge exists to serve"
    );
}

/// The smallest valid `.inf_mesh` an asset edge can be about: one triangle.
///
/// The claim being measured is a **closure edge**, and an edge does not care how
/// many triangles are on the other end of it (`inf_editor_core::heist`'s alarm
/// mesh makes the same argument about the committed sample).
fn one_triangle() -> inf_mesh::MeshAsset {
    let v = |p: [f32; 3]| inf_mesh::MeshVertex {
        position: p,
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    inf_mesh::MeshAsset::new(
        vec![inf_mesh::SubMesh {
            name: "tri".into(),
            vertices: vec![v([0.0, 0.0, 0.0]), v([1.0, 0.0, 0.0]), v([0.0, 0.0, 1.0])],
            indices: vec![0, 1, 2],
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec!["mat:tri".into()],
    )
}
