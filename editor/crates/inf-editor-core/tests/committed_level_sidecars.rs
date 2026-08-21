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
/// (`samples/phase30-gameplay/Gameplay.inf_lvl`).
///
/// Exact, not `>= 1`: this arm's whole subject is *shipped content*, and a walk
/// that quietly found one file would pass every assertion below.
const EXPECTED_LEVELS: usize = 21;

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
