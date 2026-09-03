//! **Every committed level's sidecar declares what it binds** (P26.5).
//!
//! The P26.4 audit's remainder, in its own words:
//!
//! > **The delete guard is inert on every committed sample.** All sixteen
//! > `.inf_lvl.toml` files under `samples/` and `templates/` predate
//! > `dependencies` and nothing in this batch rewrites them, so `has_referrers`
//! > stays blind to level → material for shipped content until each level is
//! > re-saved through the editor. The migration is correct and now has its arm;
//! > the *effect* is retroactively absent.
//!
//! So the sixteen are re-saved — and this is the arm that keeps them saved. It
//! is not a one-shot script: a level committed tomorrow with a stale sidecar is
//! the same defect, and the guard it disarms is the one that stands between an
//! author and deleting an asset their level needs.
//!
//! # Two claims, and the second is what makes the first safe
//!
//! 1. **The declared dependencies are the ones the payload implies** — recomputed
//!    here from the level's own document through `serialize::level_dependencies`,
//!    the same function the editor's save calls.
//! 2. **The payload bytes do NOT move.** A re-bless that rewrote the `.inf_lvl`
//!    would be a content change wearing a metadata change's clothes: every
//!    determinism trace, every cook and every gate that folds these samples reads
//!    the payload, and none of them reads the sidecar. The sidecar is the only
//!    file this batch is allowed to touch, and "allowed" has to be checked rather
//!    than intended.
//!
//! Re-bless with `INF_BLESS_SIDECARS=1`, the golden harness's own pattern.

use std::path::{Path, PathBuf};

use inf_editor_core::scene::serialize;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

/// Every committed `.inf_lvl` under `samples/` and `templates/`, path-sorted.
///
/// Discovered rather than listed: a list is a thing that goes stale silently,
/// and the count is asserted separately so a *disappearance* is still loud.
fn committed_levels() -> Vec<PathBuf> {
    let root = workspace_root();
    let mut out = Vec::new();
    for dir in ["samples", "templates"] {
        let base = root.join(dir);
        let Ok(entries) = std::fs::read_dir(&base) else {
            continue;
        };
        let mut subs: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        subs.sort();
        for sub in subs {
            let Ok(files) = std::fs::read_dir(&sub) else {
                continue;
            };
            let mut levels: Vec<PathBuf> = files
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "inf_lvl"))
                .collect();
            levels.sort();
            out.extend(levels);
        }
    }
    out.sort();
    out
}

/// The sixteen the P26.4 audit counted, plus P29.6's locomotion course, the two
/// starter levels the island phase's IB-7 ruling owes the templates that had none
/// (`templates/blank-3d/Blank.inf_lvl`, `templates/2d-platformer/Platformer.inf_lvl`),
/// **plus wave I3's thousand-building city** (`samples/phase30-city/City.inf_lvl`)
/// **and wave I6's gameplay fixture**
/// (`samples/phase30-gameplay/Gameplay.inf_lvl`), **and wave I7's two islands**
/// (`samples/island/VancouverIsland.inf_lvl`,
/// `samples/island-fixture/IslandFixture.inf_lvl`).
///
/// The islands are levels whose **terrain is not committed** — 549.9 MB and
/// 4.6 MB respectively (the first was 342.7 MB before wave TER2b's detail band;
/// re-measured by the I8a audit), built by `inf island build` from the recipe
/// beside each.
/// The level itself is authored from the committed design alone, which is
/// exactly what lets it be counted here.
///
/// …**plus wave SCRIPT3's mission** (`samples/harbour-heist/HarbourHeist.inf_lvl`),
/// which is the twenty-fourth and the first committed level whose gameplay is a
/// `.infini` rather than a `.inf_act`: a quayside slab, a grammar-built vault
/// and one hero whose `ActorClass` names the script beside it.
///
/// Exact, not `>= 1`: this arm's whole subject is *shipped content*, and a walk
/// that quietly found one file would pass every assertion below.
const EXPECTED_LEVELS: usize = 24;

#[test]
fn every_committed_level_sidecar_declares_its_bindings() {
    let bless = std::env::var("INF_BLESS_SIDECARS").is_ok();
    let levels = committed_levels();
    assert_eq!(
        levels.len(),
        EXPECTED_LEVELS,
        "found {} committed levels, not {EXPECTED_LEVELS} — if content was added or \
         removed deliberately, move this constant in the same commit",
        levels.len()
    );

    let mut with_deps = 0usize;
    let mut stale: Vec<String> = Vec::new();
    for path in &levels {
        let payload_before = std::fs::read(path).expect("read the payload");
        let doc = serialize::load(path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));

        let side_path = serialize::sidecar_path(path);
        let committed: serialize::Sidecar = toml::from_str(
            &std::fs::read_to_string(&side_path)
                .unwrap_or_else(|e| panic!("read {}: {e}", side_path.display())),
        )
        .unwrap_or_else(|e| panic!("parse {}: {e}", side_path.display()));

        // Re-encode through the SHIPPED writer, at the level's own GUID.
        let enc = serialize::encode_scene(&doc, Some(committed.guid))
            .unwrap_or_else(|e| panic!("encode {}: {e}", path.display()));

        // (2) THE PAYLOAD DOES NOT MOVE. Asserted before anything is written,
        // and asserted even under `--bless`: a re-bless that had to rewrite the
        // payload is not a re-bless, it is a content edit, and it must fail here
        // rather than land quietly in `samples/`.
        assert_eq!(
            enc.payload,
            payload_before,
            "re-encoding {} changed its payload bytes ({} → {}); the sidecar may \
             be re-blessed, the level may not",
            path.display(),
            payload_before.len(),
            enc.payload.len()
        );

        let want = serialize::level_dependencies(&doc);
        if !want.is_empty() {
            with_deps += 1;
        }
        if committed.dependencies != want {
            if bless {
                std::fs::write(&side_path, &enc.sidecar_toml).expect("write sidecar");
            } else {
                stale.push(format!(
                    "{}: declares {:?}, binds {:?}",
                    side_path.display(),
                    committed.dependencies,
                    want
                ));
            }
        }
    }

    assert!(
        stale.is_empty(),
        "{} committed level sidecar(s) do not declare what their level binds, so \
         `AssetDb::has_referrers` — the delete-with-references guard — is blind to \
         them. Re-bless with INF_BLESS_SIDECARS=1.\n{}",
        stale.len(),
        stale.join("\n")
    );

    // ANTI-VACUITY: the shipped content actually HAS bindings. Sixteen levels
    // that all bind nothing would satisfy every equality above, and the guard
    // would still be inert — which is precisely the state this arm exists to end.
    assert!(
        with_deps >= 8,
        "only {with_deps} of {EXPECTED_LEVELS} committed levels bind any asset at \
         all, so this arm is a statement about empty lists"
    );
}

/// **…and the guard actually fires on shipped content.**
///
/// The claim the re-bless was for. `phase26_gate` pins the mechanism on a
/// synthetic fixture; this pins the *effect* on the files in the repository,
/// which is the difference the audit's remainder was about — *"the migration is
/// correct and now has its arm; the effect is retroactively absent."*
///
/// Scanned as a real `AssetDb` over each sample's own folder, because that is
/// the door the Content Drawer's delete goes through.
#[test]
fn the_delete_guard_sees_a_shipped_levels_bindings() {
    let mut guarded = 0usize;
    let mut scanned = 0usize;
    for path in committed_levels() {
        let dir = path.parent().expect("a level lives in a folder");
        let side_path = serialize::sidecar_path(&path);
        let side: serialize::Sidecar =
            toml::from_str(&std::fs::read_to_string(&side_path).expect("read sidecar"))
                .expect("parse sidecar");
        if side.dependencies.is_empty() {
            continue;
        }
        scanned += 1;

        let mut db = inf_asset::AssetDb::new(dir);
        db.scan().expect("scan the sample folder");
        // Only the bindings whose target actually lives beside the level can be
        // checked here — a sample that names an asset from elsewhere in the
        // workspace is not this arm's business.
        for dep in &side.dependencies {
            let id = inf_asset::AssetId(*dep);
            if db.get(id).is_none() {
                continue;
            }
            assert!(
                db.has_referrers(id),
                "{}: {dep} is in the folder and the level declares it, but the \
                 delete guard sees no referrer",
                path.display()
            );
            assert!(
                db.referenced_by(id).iter().any(|r| r.uuid() == side.guid),
                "{}: the referrer of {dep} is not the level",
                path.display()
            );
            guarded += 1;
        }
    }
    assert!(scanned > 0, "no committed level declares a dependency");
    // ANTI-VACUITY: at least one declared binding really resolved to an asset
    // sitting beside its level, or every loop above ran zero times and this
    // test is a statement about `continue`.
    assert!(
        guarded > 0,
        "{scanned} level(s) declare dependencies and NONE of them names an asset \
         in its own folder, so nothing here exercised the guard"
    );
}

/// **THE SEAM'S REGRESSION SURFACE, MEASURED ON SHIPPED CONTENT** (wave VEH2c;
/// written by that wave's audit, which found the citation and not the arm).
///
/// `inf_ecs::vehicle::part_of`'s own doc names this test by this name and says
/// it "measures it on the committed island". At `4c69d3b5` it did not exist —
/// the citation was the only thing that did, which is the one failure mode a
/// cited gate has that no gate at all does not.
///
/// # What is actually at risk
///
/// Wave VEH2c opened the vehicle recogniser to a chassis with **no wheels**: a
/// dynamic body whose child is a **box** or **capsule** sensor now derives a
/// [`VehicleRig`](inf_ecs::vehicle::VehicleRig) where before it derived nothing.
/// `WHEELS WIN` bounds that — a chassis that has wheels keeps its box children
/// as ordinary triggers — but it bounds it only for **wheeled** bodies. A
/// wheel-less dynamic body with a box sensor child that some level authored for
/// another reason entirely would have become a vehicle on the day this wave
/// landed, silently, and the only place that shows up is in shipped content.
///
/// So the claim is measured over **every committed level**, not over a fixture:
/// the only wheel-less craft in this repository's content are the two the island
/// recipe places, and every other rig in every other level is a wheeled one.
#[test]
fn the_islands_only_wheel_less_vehicles_are_the_ones_it_placed() {
    let levels = committed_levels();
    assert_eq!(
        levels.len(),
        EXPECTED_LEVELS,
        "found {} committed levels, not {EXPECTED_LEVELS}",
        levels.len()
    );

    // (level, chassis name, wheels, parts) for every rig in every committed level.
    let mut wheeled: Vec<String> = Vec::new();
    let mut craft: Vec<String> = Vec::new();
    for path in &levels {
        let doc = serialize::load(path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
        let world = doc.world();
        // Every guid in the level, in the order the world walks them; `rig_of`
        // answers `None` for everything that is not a chassis, which is the same
        // door the physics bridge asks.
        let guids: Vec<uuid::Uuid> = world
            .world()
            .iter_entities()
            .filter_map(|e| e.get::<inf_ecs::Guid>().map(|g| g.0))
            .collect();
        let level = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        for guid in guids {
            let Some(rig) = inf_ecs::vehicle::rig_of(world, guid) else {
                continue;
            };
            let name = world
                .entity_of(guid)
                .and_then(|e| world.world().get::<inf_ecs::components::Name>(e))
                .map(|n| n.0.clone())
                .unwrap_or_else(|| guid.to_string());
            let row = format!(
                "{level}: {name} — {} wheel(s), {} part(s)",
                rig.wheels.len(),
                rig.parts.len()
            );
            if rig.wheels.is_empty() {
                craft.push(row);
            } else {
                // WHEELS WIN as a tripwire over shipped content, and named as
                // one: no committed level authors a box or capsule sensor under
                // a *wheeled* chassis today, so this clause cannot fail on the
                // content as it stands. The rule's own mutation-verified arm is
                // Ring 0's `wheels_win_so_an_existing_cars_trigger_child_stays_a_
                // trigger`, which reds when `rig_of`'s `parts.clear()` is
                // dropped; this one exists so the day a level does author that
                // shape, it is caught in the content rather than in a fixture.
                assert!(
                    rig.parts.is_empty(),
                    "a wheeled rig came back carrying parts, which is the one \
                     thing `rig_of`'s compatibility rule forbids — {row}"
                );
                wheeled.push(row);
            }
        }
    }

    println!("WHEELED RIGS IN COMMITTED CONTENT ({}):", wheeled.len());
    for r in &wheeled {
        println!("  {r}");
    }
    println!("WHEEL-LESS CRAFT IN COMMITTED CONTENT ({}):", craft.len());
    for r in &craft {
        println!("  {r}");
    }

    // ANTI-VACUITY, and it is the first thing checked: a walk that found no rig
    // at all would satisfy every claim below. The committed content really does
    // ship wheeled vehicles, and this arm really does see them.
    assert!(
        wheeled.len() >= 4,
        "only {} wheeled rig(s) in {EXPECTED_LEVELS} committed levels — this arm \
         is looking at the wrong thing",
        wheeled.len()
    );

    // THE CLAIM. Every wheel-less craft in shipped content belongs to the island,
    // and the island's are the two its own recipe places.
    let strays: Vec<&String> = craft
        .iter()
        .filter(|r| !r.starts_with("VancouverIsland.inf_lvl:"))
        .collect();
    assert!(
        strays.is_empty(),
        "wave VEH2c's recogniser turned {} authored entity(s) OUTSIDE the island \
         into vehicles — a level that shipped a box sensor under a wheel-less \
         dynamic body for some other reason now has a boat in it:\n  {}",
        strays.len(),
        strays
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    assert_eq!(
        craft.len(),
        2,
        "the island ships {} wheel-less craft, not the launch and the \
         helicopter:\n  {}",
        craft.len(),
        craft.join("\n  ")
    );
}
