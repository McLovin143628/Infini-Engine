//! Simulate integration for animation state machines (P11.2).
//!
//! Proves the tick wiring: an entity carrying an `AnimStateMachine` whose `sm`
//! GUID resolves in the session's registered machines is stepped each fixed step
//! (after `advance_anim_players`), its runtime advancing deterministically. The
//! machine's condition/param lookups read the actor's Blueprint variables through
//! the `SmContext` seam; an entity with no actor gets an empty variable set, so
//! params default to `0` — exercised here (the entity is not a blueprint actor).
//!
//! Pose evaluation → skinning palette **used to be** render-time (the same
//! placeholder gap as `SkeletalMesh`), and P24.1 closed that: the same fixed step
//! now evaluates the machine's pose and publishes it for both projectors
//! (`inf_ecs::pose`). This file still asserts only the *runtime state*, which is
//! the narrower claim it was written for; the pose half is
//! `runtime/inf-player/tests/pose_parity.rs`.

mod support;

use std::collections::BTreeMap;

use glam::DVec2;
use uuid::Uuid;

use inf_anim::state_machine::{CmpOp, SmCondition, SmState, SmTransition, StateMachine};
use inf_ecs::components::AnimStateMachine;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

fn spawn_sm_entity(doc: &mut SceneDoc, guid: Uuid, sm: Uuid) -> inf_ecs::Entity {
    let e = doc.world_mut().spawn_with_guid(guid, "hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(AnimStateMachine {
            sm: Some(sm),
            params_from_vars: true,
            ..Default::default()
        });
    e
}

#[test]
fn unconditional_machine_advances_in_the_tick() {
    let mut doc = SceneDoc::new();
    let entity_guid = Uuid::from_u128(0xA0);
    let sm_guid = Uuid::from_u128(0x5A);
    let e = spawn_sm_entity(&mut doc, entity_guid, sm_guid);

    // idle → walk, unconditional (no conditions, no exit-time) → fires at once.
    let machine = StateMachine {
        states: vec![
            SmState::clip("idle", [1; 16]),
            SmState::clip("walk", [2; 16]),
        ],
        transitions: vec![SmTransition {
            from: 0,
            to: 1,
            duration: 0.0,
            conditions: vec![],
            exit_time: None,
        }],
        entry: 0,
    };
    let mut machines = BTreeMap::new();
    machines.insert(sm_guid, machine);

    // No actors: the SM entity is not a blueprint → params default 0, but the
    // machine (with no conditions) still steps.
    let mut session = SimSession::enter(&mut doc, vec![], DVec2::ZERO, SIM_HZ);
    session.set_state_machines(machines);
    session.step_once(&mut doc, SimInput::default());

    let asm = doc.world().world().get::<AnimStateMachine>(e).unwrap();
    assert!(asm.runtime.started, "runtime must be entered");
    assert_eq!(
        asm.runtime.current, 1,
        "unconditional transition should fire"
    );
}

#[test]
fn unresolved_machine_is_left_untouched() {
    // An `AnimStateMachine` whose `sm` GUID is NOT registered must not advance.
    let mut doc = SceneDoc::new();
    let e = spawn_sm_entity(&mut doc, Uuid::from_u128(0xB0), Uuid::from_u128(0x99));

    let mut session = SimSession::enter(&mut doc, vec![], DVec2::ZERO, SIM_HZ);
    session.set_state_machines(BTreeMap::new()); // nothing resolvable
    session.step_once(&mut doc, SimInput::default());

    let asm = doc.world().world().get::<AnimStateMachine>(e).unwrap();
    assert!(!asm.runtime.started, "an unresolved machine must not step");
}

#[test]
fn condition_machine_holds_until_its_variable_flips() {
    // A machine gated on `moving > 0.5`. With no actor the variable is absent
    // (reads 0), so it must stay in idle across steps.
    let mut doc = SceneDoc::new();
    let entity_guid = Uuid::from_u128(0xC0);
    let sm_guid = Uuid::from_u128(0x77);
    let e = spawn_sm_entity(&mut doc, entity_guid, sm_guid);

    let machine = StateMachine {
        states: vec![
            SmState::clip("idle", [1; 16]),
            SmState::clip("walk", [2; 16]),
        ],
        transitions: vec![SmTransition {
            from: 0,
            to: 1,
            duration: 0.2,
            conditions: vec![SmCondition {
                var: "moving".into(),
                op: CmpOp::Gt,
                value: 0.5,
            }],
            exit_time: None,
        }],
        entry: 0,
    };
    let mut machines = BTreeMap::new();
    machines.insert(sm_guid, machine);

    let mut session = SimSession::enter(&mut doc, vec![], DVec2::ZERO, SIM_HZ);
    session.set_state_machines(machines);
    for _ in 0..10 {
        session.step_once(&mut doc, SimInput::default());
    }

    let asm = doc.world().world().get::<AnimStateMachine>(e).unwrap();
    assert_eq!(asm.runtime.current, 0, "condition never met → stays idle");
}
// ── P24.1 audit B1 / re-audit F1+F2: no decode on the sim path is silent ─────

/// The **one** door a fallible asset decode may flow through in `simulate.rs`.
///
/// An ALLOWLIST, not a ban list. The re-audit's finding was that banning `.ok()`
/// enumerates the spellings whoever wrote the ban happened to think of: the same
/// silent decode passes as `.map_or(None, Some)`, as `if let Ok(x) = …`, as
/// `let Ok(x) = … else`, or laundered through a one-line helper elsewhere in the
/// module. What all of those share is the **call** — so the gate asks which
/// function is called, and requires the answer to be this one.
const REPORTING_DOOR: &str = "decode_anim";

/// Where the gate looks: the **whole module**, not one item.
///
/// A per-item scope would still miss the laundering case (a helper defined
/// outside the item that swallows the error inside itself), which is exactly the
/// respelling that motivated this rebuild. `decode_anim` is the module's single
/// asset-decode call site; everything else must route through it. (The P23 law:
/// read a SCOPE, ban the MODULE.)
fn simulate_module_code() -> String {
    // Comments and string literals are blanked FIRST. A gate that matches raw
    // source reads prose as code — the previous ban would have fired on a doc
    // comment quoting `.ok()` while explaining why `.ok()` is wrong, which is
    // the sentence such a gate is most likely to be written next to.
    support::strip_comments_and_strings(&support::read_crate_source("src/simulate.rs"))
}

/// **Every asset decode on the Simulate path reports its failure** (P24.1 B1).
///
/// This was `inf_asset::decode::<T>(&b).ok()`, five times, and the `.ok()` was
/// the whole defect. A `.inf_skel` written before the v2 `limits` tail cannot be
/// decoded (bincode is positional), so the `None` propagated, the skeleton
/// registry stayed empty, the machine advanced while publishing no pose, and the
/// character silently stopped animating with its sockets back at the origin —
/// P24.1's own defect, reintroduced through a discarded error. Every project that
/// imported a glTF before this batch has v1 `.inf_skel` files.
#[test]
fn every_asset_decode_in_simulate_flows_through_the_reporting_door() {
    let code = simulate_module_code();
    let sites = support::decode_call_sites(&code);

    // Not vacuous: the module really does decode assets, and really does route
    // them. If this shrank to nothing the allowlist below would be satisfied by
    // an empty set.
    let through_door = sites.iter().filter(|(_, p)| p == REPORTING_DOOR).count();
    assert!(
        through_door >= 5,
        "expected at least five decodes through `{REPORTING_DOOR}` (the \
         `SkeletalMesh` skeleton, the machine, the `AnimPlayer` clip, that clip's \
         own skeleton, each clip the machine's states play, and the `.inf_audio` \
         clips); found {through_door} in {sites:?}"
    );

    // The door's own line span. Anchored through `support::item_start`, so a
    // mention of the signature in a doc comment cannot move it — the F2 finding,
    // which is what this line exists to be immune to.
    let raw = support::read_crate_source("src/simulate.rs");
    let door_text = support::item_text(&raw, REPORTING_DOOR);
    let door_at = raw
        .find(&door_text)
        .expect("the extracted item is a substring of its file");
    let door_first = raw[..door_at].bytes().filter(|c| *c == b'\n').count() + 1;
    let door_last = door_first + door_text.bytes().filter(|c| *c == b'\n').count();

    // The allowlist: a decode call site is legal iff it calls the door, OR it IS
    // the one `inf_asset::decode` inside the door's own body.
    let stray: Vec<&(usize, String)> = sites
        .iter()
        .filter(|(line, path)| {
            let inside_door = (door_first..=door_last).contains(line);
            !(path == REPORTING_DOOR || (path == "inf_asset::decode" && inside_door))
        })
        .collect();
    assert!(
        stray.is_empty(),
        "an asset decode in `simulate.rs` does not go through `{REPORTING_DOOR}` \
         (lines {door_first}..={door_last}) — however its result is spelled, a \
         stale asset would then stop a character animating (or an emitter \
         sounding) with no message at all, which is the exact failure P24.1's \
         audit blocked on. Offending call sites (line, callee): {stray:?}"
    );
    // …and the door really does decode, exactly once. Two stubs are identical.
    assert_eq!(
        sites
            .iter()
            .filter(|(line, path)| path == "inf_asset::decode"
                && (door_first..=door_last).contains(line))
            .count(),
        1,
        "`{REPORTING_DOOR}` (lines {door_first}..={door_last}) no longer calls \
         `inf_asset::decode` exactly once — it is the module's single decode site \
         by construction"
    );
}

/// …and the reporting door really is the loud one: a v1 `.inf_skel` resolves to
/// **nothing**, which is why the message is the only signal a user gets.
///
/// The behaviour is deliberate — one stale rig must not stop a session starting —
/// so this pins the shape of the honest outcome rather than pretending the asset
/// loads.
#[test]
fn a_stale_skeleton_resolves_to_nothing_and_is_refused_by_name() {
    use inf_anim::{Joint, JointTransform, Skeleton};
    use inf_ecs::components::SkeletalMesh;

    const HERO: Uuid = Uuid::from_u128(0x2401_B101);
    const SKEL: Uuid = Uuid::from_u128(0x2401_B102);

    let skeleton = Skeleton::new(vec![Joint {
        name: "root".into(),
        parent: None,
        inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
        local_bind: JointTransform::IDENTITY,
    }])
    .unwrap();

    // A v1 payload: the pre-P24.1 shape, spelled out rather than derived from the
    // live encoder.
    #[derive(serde::Serialize)]
    struct SkeletonAssetV1 {
        schema_version: u32,
        skeleton: Skeleton,
        sockets: Vec<inf_anim::Socket>,
    }
    let v1 = bincode::serde::encode_to_vec(
        &SkeletonAssetV1 {
            schema_version: 1,
            skeleton,
            sockets: vec![],
        },
        inf_asset::bincode_config(),
    )
    .unwrap();

    // The refusal is NAMED and carries the remedy — this is the text that reaches
    // the Output Log through `decode_anim`.
    let err = inf_asset::decode::<inf_anim::SkeletonAsset>(&v1).unwrap_err();
    assert!(matches!(
        err,
        inf_asset::AssetError::SchemaTooOld {
            kind: "skeleton",
            ..
        }
    ));
    assert!(
        err.to_string().contains("re-import the source model"),
        "{err}"
    );

    // And the honest outcome: the resolver yields no skeleton, so the character
    // steps its machine and poses nothing.
    let mut doc = SceneDoc::new();
    let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(SkeletalMesh {
            mesh: Some(Uuid::from_u128(9)),
            skeleton: Some(SKEL),
        });
    let (_machines, _root, skeletons, _clips) =
        inf_editor_core::simulate::resolve_anim_assets(&doc, |g| (g == SKEL).then(|| v1.clone()));
    assert!(
        skeletons.is_empty(),
        "a v1 `.inf_skel` must not resolve — if it did, the reader invented a \
         limits table out of whatever followed"
    );
}
