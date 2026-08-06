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
// ── P24.1 audit B1: no error on the sim path is discarded ────────────────────

/// The `resolve_anim_assets` item's source text, signature line through its
/// closing brace at column 0.
fn resolve_anim_assets_src() -> String {
    // Normalized: `core.autocrlf = true` checks `.rs` out CRLF on Windows.
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/simulate.rs"),
    )
    .expect("simulate.rs is readable")
    .replace("\r\n", "\n");
    let at = src
        .find("pub fn resolve_anim_assets<H>(")
        .expect("`resolve_anim_assets` is still an item");
    let rest = &src[at..];
    let end = rest.find("\n}\n").expect("it terminates at column 0") + 3;
    rest[..end].to_string()
}

/// **Every decode on the Simulate path reports its failure** (P24.1 audit B1).
///
/// This was `inf_asset::decode::<T>(&b).ok()`, four times, and the `.ok()` was the
/// whole defect. A `.inf_skel` written before the v2 `limits` tail cannot be
/// decoded (bincode is positional), so the `None` propagated, the skeleton
/// registry stayed empty, the machine advanced while publishing no pose, and the
/// character silently stopped animating with its sockets back at the origin —
/// P24.1's own defect, reintroduced through a discarded error. Every project that
/// imported a glTF before this batch has v1 `.inf_skel` files.
///
/// A SCOPE, not a spelling: the ban is over `resolve_anim_assets`'s whole item, so
/// a fifth decode added tomorrow has to go through the reporting door too. (The
/// P23 law — a byte pin cannot see a semantic change; read a scope, ban the
/// module.)
#[test]
fn no_decode_on_the_simulate_path_discards_its_error() {
    let body = resolve_anim_assets_src();
    assert!(
        !body.contains(".ok()"),
        "a decode inside `resolve_anim_assets` discards its error — a stale \
         `.inf_skel` would then stop a character animating with no message at \
         all, which is the exact failure P24.1's audit blocked on:\n{body}"
    );
    // …and it is not a stub: every one of the five asset lookups goes through the
    // reporting door. FIVE, measured — this was written as four and the test said
    // otherwise: the transitive machine-clip walk lives inside this item too, not
    // outside it.
    assert_eq!(
        body.matches("decode_anim::<").count(),
        5,
        "expected five reported decodes — the `SkeletalMesh` skeleton, the machine, \
         the `AnimPlayer` clip, that clip's own skeleton, and each clip the \
         machine's states play:\n{body}"
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
