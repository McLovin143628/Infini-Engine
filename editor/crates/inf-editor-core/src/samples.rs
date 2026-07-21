//! Committed sample content (P8.4): the **2D platformer** the Phase-8 gate plays
//! in-viewport via the interpreter.
//!
//! The sample is defined by the generators here (the source of truth) and
//! committed under `samples/platformer-2d/` as the exact bytes they produce:
//!
//! * `Platformer.inf_lvl` (+ `.toml` sidecar) — a tilemap ground strip, a static
//!   ground collider that ends in a **ledge**, a floating platform, a 2D light,
//!   and the **player** (Sprite + kinematic `RigidBody2D` + capsule `Collider2D`
//!   + `CharacterController2D`).
//! * `Coyote.inf_act` (+ `.json`) — the player's [`BlueprintClass`]: the
//!   **coyote-time jump** handler. It is authored directly as [`BlueprintFn`] IR
//!   (the on-disk `.inf_act` stores the lowered IR, not the visual graph) —
//!   richer than a hand-wired graph would be, and the same IR the interpreter and
//!   the transpiler consume.
//!
//! The coyote-time rule: horizontal move from left/right input, gravity applied
//! to a `vy` velocity var, and a jump that is allowed while `is_grounded` **or**
//! within a short **coyote window** — a `coyote` float reset when grounded and
//! decremented every tick — so a jump pressed a few frames *after* walking off a
//! ledge still fires.

use std::path::PathBuf;

use glam::DVec3;
use uuid::Uuid;

use inf_blueprint::{
    BinOp, Binding, BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, LocalId,
    Param, Stmt, Ty, Variable,
};
use inf_ecs::components::{
    ActorClass, BillboardMode, BodyKind2D, CharacterController2D, Collider2D, ColliderShape2DKind,
    Light2D, RigidBody2D, Sprite, Tilemap, Transform,
};
use inf_ecs::math::{Color, Vec2d};

use crate::ipc::SpawnKind;
use crate::scene::SceneDoc;

// ── Stable identities (so the committed sample is byte-reproducible) ──────────
pub const GROUND_GUID: Uuid = Uuid::from_u128(0x8401_0001);
pub const GROUND_TILES_GUID: Uuid = Uuid::from_u128(0x8401_0002);
pub const PLATFORM_GUID: Uuid = Uuid::from_u128(0x8401_0003);
pub const PLAYER_GUID: Uuid = Uuid::from_u128(0x8401_0004);
pub const LIGHT_GUID: Uuid = Uuid::from_u128(0x8401_0005);
/// The fixed level GUID stamped into the committed sidecar.
pub const LEVEL_GUID: Uuid = Uuid::from_u128(0x8401_0000);
/// The coyote actor class id.
pub const COYOTE_CLASS_ID: &str = "act:coyote_player";
/// The **asset** GUID of the committed `Coyote.inf_act` (its inf_asset sidecar,
/// P9.5). Stable so the level's persisted `actor` binding (the [`ActorClass`] on
/// the player) resolves to this blueprint through the AssetDb / cooked pack.
pub const COYOTE_ASSET_GUID: Uuid = Uuid::from_u128(0x8401_00AC);

// ── Coyote-time tuning (world units, seconds) ────────────────────────────────
/// Downward acceleration applied to `vy` each tick.
pub const GRAVITY: f64 = 30.0;
/// Upward velocity a jump imparts.
pub const JUMP_SPEED: f64 = 9.0;
/// Horizontal run speed.
pub const MOVE_SPEED: f64 = 5.0;
/// How long after leaving the ground a jump is still allowed (the coyote window).
pub const COYOTE_TIME: f64 = 0.12;

// ── IR builders (keep the handler readable) ──────────────────────────────────
fn str_lit(v: &str) -> Expr {
    Expr::Lit(Lit::Str(v.to_string()))
}
fn float_lit(v: f64) -> Expr {
    Expr::Lit(Lit::Float(v))
}
fn local(id: u32) -> Expr {
    Expr::Local(LocalId(id))
}
fn call(path: &[&str], args: Vec<Expr>) -> Expr {
    Expr::Call {
        path: path.iter().map(|s| s.to_string()).collect(),
        args,
    }
}
fn get_var(name: &str) -> Expr {
    call(&["vars", "get"], vec![str_lit(name)])
}
fn set_var(name: &str, value: Expr) -> Stmt {
    Stmt::ExprStmt(call(&["vars", "set"], vec![str_lit(name), value]))
}
fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Binary(op, Box::new(a), Box::new(b))
}
fn let_named(id: u32, name: &str, mutable: bool, value: Expr) -> Stmt {
    Stmt::Let {
        id: LocalId(id),
        binding: Binding::Named(name.to_string()),
        ty: None,
        mutable,
        value,
    }
}
fn if_then(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
    Stmt::If {
        cond,
        then_body,
        else_body,
    }
}

/// The coyote-time **Tick** handler as `BlueprintFn` IR. `dt` is the fixed step.
///
/// Locals: `n1=entity`, `n2=grounded`, `n3=vx`, `n4=grounded_after`.
fn coyote_tick_fn() -> BlueprintFn {
    let e = || local(1);
    let dt = || Expr::Param("dt".to_string());
    let grounded = || local(2);

    let body = vec![
        // let entity = vars::get("entity")   (seeded by the Simulate session)
        let_named(1, "entity", false, get_var("entity")),
        // let grounded = physics2d::is_grounded(entity)
        let_named(
            2,
            "grounded",
            false,
            call(&["physics2d", "is_grounded"], vec![e()]),
        ),
        // coyote timer: reset to COYOTE_TIME when grounded, else count down by dt.
        if_then(
            grounded(),
            vec![set_var("coyote", float_lit(COYOTE_TIME))],
            vec![set_var("coyote", bin(BinOp::Sub, get_var("coyote"), dt()))],
        ),
        // gravity: vy -= GRAVITY * dt
        set_var(
            "vy",
            bin(
                BinOp::Sub,
                get_var("vy"),
                bin(BinOp::Mul, float_lit(GRAVITY), dt()),
            ),
        ),
        // jump: if just_pressed("jump") && (grounded || coyote > 0) { vy = JUMP; coyote = 0 }
        if_then(
            bin(
                BinOp::And,
                call(&["input", "just_pressed"], vec![str_lit("jump")]),
                bin(
                    BinOp::Or,
                    grounded(),
                    bin(BinOp::Gt, get_var("coyote"), float_lit(0.0)),
                ),
            ),
            vec![
                set_var("vy", float_lit(JUMP_SPEED)),
                set_var("coyote", float_lit(0.0)),
            ],
            vec![],
        ),
        // horizontal: vx from left/right held state.
        let_named(3, "vx", true, float_lit(0.0)),
        if_then(
            call(&["input", "is_down"], vec![str_lit("right")]),
            vec![Stmt::Assign {
                target: LocalId(3),
                value: bin(BinOp::Add, local(3), float_lit(MOVE_SPEED)),
            }],
            vec![],
        ),
        if_then(
            call(&["input", "is_down"], vec![str_lit("left")]),
            vec![Stmt::Assign {
                target: LocalId(3),
                value: bin(BinOp::Sub, local(3), float_lit(MOVE_SPEED)),
            }],
            vec![],
        ),
        // move_and_slide(entity, vx*dt, vy*dt) → grounded_after
        let_named(
            4,
            "grounded_after",
            false,
            call(
                &["physics2d", "move_and_slide"],
                vec![
                    e(),
                    bin(BinOp::Mul, local(3), dt()),
                    bin(BinOp::Mul, get_var("vy"), dt()),
                ],
            ),
        ),
        // Landed while moving down → cancel the downward velocity.
        if_then(
            bin(
                BinOp::And,
                local(4),
                bin(BinOp::Lt, get_var("vy"), float_lit(0.0)),
            ),
            vec![set_var("vy", float_lit(0.0))],
            vec![],
        ),
    ];

    BlueprintFn {
        id: EventKind::Tick.key(),
        name: EventKind::Tick.key(),
        params: vec![Param {
            name: "dt".to_string(),
            ty: Ty::Float,
        }],
        ret: Ty::Unit,
        body,
    }
}

/// A `BeginPlay` handler that zeroes the motion state (so a re-enter is clean).
fn coyote_begin_play_fn() -> BlueprintFn {
    BlueprintFn {
        id: EventKind::BeginPlay.key(),
        name: EventKind::BeginPlay.key(),
        params: vec![],
        ret: Ty::Unit,
        body: vec![
            set_var("vy", float_lit(0.0)),
            set_var("coyote", float_lit(0.0)),
        ],
    }
}

/// The player's coyote-time [`BlueprintClass`] (the `.inf_act`).
pub fn coyote_class() -> BlueprintClass {
    let mut class = BlueprintClass::new(COYOTE_CLASS_ID, "Coyote Player");
    class.variables = vec![
        // `entity` is the opaque blueprint id; the Simulate session seeds it.
        Variable {
            name: "entity".into(),
            ty: Ty::Int,
            default: Lit::Int(0),
            exposed: false,
        },
        Variable {
            name: "vy".into(),
            ty: Ty::Float,
            default: Lit::Float(0.0),
            exposed: true,
        },
        Variable {
            name: "coyote".into(),
            ty: Ty::Float,
            default: Lit::Float(0.0),
            exposed: true,
        },
    ];
    class.events = vec![
        EventBinding {
            event: EventKind::BeginPlay,
            body: coyote_begin_play_fn(),
        },
        EventBinding {
            event: EventKind::Tick,
            body: coyote_tick_fn(),
        },
    ];
    class
}

// ── The scene ────────────────────────────────────────────────────────────────

/// Insert a bundle onto `guid`'s entity (dirties the doc), mirroring the pattern
/// the scene serialize tests use — this crate doesn't name `bevy_ecs::Bundle`.
macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// Build the committed platformer [`SceneDoc`].
///
/// Layout (side view, +Y up): a static ground box spanning world x ∈ [-3, 3] with
/// its top at y = 0, a painted tilemap strip beneath it, a floating platform to
/// the right, a soft 2D light, and the player standing on the ground at x = 1.5
/// — so running **right** walks off the ledge at x = 3.
pub fn platformer_scene() -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.set_title("Platformer 2D");

    // Ground collider (static box): centre (0,-0.5), half (3.0,0.5) → top y=0.
    doc.create_with_guid(GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        GROUND_GUID,
        Transform::from_translation(DVec3::new(0.0, -0.5, 0.0))
    );
    insert!(
        doc,
        GROUND_GUID,
        RigidBody2D {
            kind: BodyKind2D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        GROUND_GUID,
        Collider2D {
            shape_kind: ColliderShape2DKind::Box,
            half_extents: Vec2d::new(3.0, 0.5),
            ..Default::default()
        }
    );

    // Visual tilemap ground strip (painted via the chunk API) under the collider.
    doc.create_with_guid(GROUND_TILES_GUID, SpawnKind::Empty, "Ground Tiles", None);
    insert!(
        doc,
        GROUND_TILES_GUID,
        Transform::from_translation(DVec3::new(-3.0, -1.0, 0.0))
    );
    let mut tm = Tilemap {
        tile_size: Vec2d::new(1.0, 1.0),
        atlas_cols: 1,
        atlas_rows: 1,
        tint: Color::new(0.35, 0.30, 0.25, 1.0),
        ..Default::default()
    };
    // A 6-wide, 1-tall row of ground tiles beneath the collider.
    for gx in 0..6i32 {
        tm.set_tile(gx, 0, 1);
    }
    insert!(doc, GROUND_TILES_GUID, tm);

    // Floating platform (static box) up and to the right.
    doc.create_with_guid(PLATFORM_GUID, SpawnKind::Empty, "Platform", None);
    insert!(
        doc,
        PLATFORM_GUID,
        Transform::from_translation(DVec3::new(5.0, 1.5, 0.0))
    );
    insert!(
        doc,
        PLATFORM_GUID,
        RigidBody2D {
            kind: BodyKind2D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLATFORM_GUID,
        Collider2D {
            shape_kind: ColliderShape2DKind::Box,
            half_extents: Vec2d::new(1.25, 0.25),
            ..Default::default()
        }
    );

    // A soft 2D light so the sprites read.
    doc.create_with_guid(LIGHT_GUID, SpawnKind::Empty, "Sun 2D", None);
    insert!(
        doc,
        LIGHT_GUID,
        Transform::from_translation(DVec3::new(0.0, 3.0, 0.0))
    );
    insert!(
        doc,
        LIGHT_GUID,
        Light2D {
            color: Color::new(1.0, 0.95, 0.8, 1.0),
            intensity: 1.2,
            radius: 12.0,
        }
    );

    // The player: sprite + kinematic body + capsule + character controller.
    doc.create_with_guid(PLAYER_GUID, SpawnKind::Empty, "Player", None);
    insert!(
        doc,
        PLAYER_GUID,
        Transform::from_translation(DVec3::new(1.5, 0.8, 0.0))
    );
    insert!(
        doc,
        PLAYER_GUID,
        Sprite {
            size: Vec2d::new(0.8, 1.2),
            color: Color::new(0.9, 0.4, 0.3, 1.0),
            billboard: BillboardMode::None,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYER_GUID,
        RigidBody2D {
            kind: BodyKind2D::Kinematic,
            fixed_rotation: true,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYER_GUID,
        Collider2D {
            shape_kind: ColliderShape2DKind::Capsule,
            half_extents: Vec2d::new(0.3, 0.35),
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYER_GUID,
        CharacterController2D {
            max_slope_deg: 46.0,
            snap_to_ground: 0.3,
            offset: 0.02,
        }
    );
    // Bind the player to the Coyote blueprint class (P9.5 persisted actor link):
    // the level now carries its own gameplay binding — no CC2D heuristic needed.
    insert!(doc, PLAYER_GUID, ActorClass(COYOTE_ASSET_GUID));

    // Level settings: the platformer keeps the character-self-gravity convention
    // (2D world gravity ZERO) at 60 Hz — i.e. the schema-v3 defaults, now made
    // explicit + persisted instead of the player's old hard-coded constants.
    doc.set_settings(crate::scene::serialize::LevelSettings::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The `(guid, class)` actor list for [`crate::simulate::SimSession::enter`].
pub fn platformer_actors() -> Vec<(Uuid, BlueprintClass)> {
    vec![(PLAYER_GUID, coyote_class())]
}

/// Resolve the in-editor Simulate actor list, **preferring the level's persisted
/// [`ActorClass`] bindings** (P9.5): each entity carrying an `ActorClass` is run
/// with the blueprint class its asset GUID resolves to (via `resolve`, which the
/// caller backs with the project's asset DB). Falls back to the legacy
/// [`character_actors`] heuristic when the scene carries **no** bindings at all
/// (kept for scenes authored before v3). Guid order.
pub fn bound_actors<F>(doc: &SceneDoc, mut resolve: F) -> Vec<(Uuid, BlueprintClass)>
where
    F: FnMut(Uuid) -> Option<BlueprintClass>,
{
    let mut out = Vec::new();
    let mut any_binding = false;
    let w = doc.world();
    for &guid in doc.order() {
        if let Some(e) = w.entity_of(guid) {
            if let Some(ac) = w.world().get::<ActorClass>(e) {
                any_binding = true;
                if let Some(class) = resolve(ac.0) {
                    out.push((guid, class));
                }
            }
        }
    }
    if any_binding && !out.is_empty() {
        out
    } else {
        // No bindings (or none resolved) → legacy CC2D heuristic.
        character_actors(doc)
    }
}

/// Discover controllable actors in an arbitrary scene for Simulate: every entity
/// carrying a `CharacterController2D` gets the coyote-time class (the legacy
/// pre-v3 heuristic, kept as the fallback for [`bound_actors`]). Guid order.
pub fn character_actors(doc: &SceneDoc) -> Vec<(Uuid, BlueprintClass)> {
    let mut out = Vec::new();
    let w = doc.world();
    for &guid in doc.order() {
        if let Some(e) = w.entity_of(guid) {
            if w.world().get::<CharacterController2D>(e).is_some() {
                out.push((guid, coyote_class()));
            }
        }
    }
    out
}

// ── Hybrid 2.5D template scene (P8.4b) ───────────────────────────────────────
pub const HYBRID_GROUND_GUID: Uuid = Uuid::from_u128(0x8402_0001);
pub const HYBRID_SUN_GUID: Uuid = Uuid::from_u128(0x8402_0002);
pub const HYBRID_LIGHT2D_GUID: Uuid = Uuid::from_u128(0x8402_0003);
pub const HYBRID_SPRITE_SPHERE_GUID: Uuid = Uuid::from_u128(0x8402_0004);
pub const HYBRID_SPRITE_CYL_GUID: Uuid = Uuid::from_u128(0x8402_0005);
pub const HYBRID_LEVEL_GUID: Uuid = Uuid::from_u128(0x8402_0000);

/// The minimal **hybrid 2.5D** starter scene: a 3D ground **plane mesh**, a
/// directional sun, a soft 2D light, and two **billboarded** sprites standing on
/// the plane (one spherical, one cylindrical) — the 2.5D idiom (2D cards in a 3D
/// world). Scaffolded by `inf new --template hybrid-2.5d`.
pub fn hybrid_scene() -> SceneDoc {
    use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive};

    let mut doc = SceneDoc::new();
    doc.set_title("Hybrid 2.5D");

    // 3D ground plane.
    doc.create_with_guid(HYBRID_GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        HYBRID_GROUND_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::ZERO,
            rotation: inf_ecs::math::Vec3d::ZERO,
            scale: inf_ecs::math::Vec3d::new(20.0, 1.0, 20.0),
        }
    );
    insert!(
        doc,
        HYBRID_GROUND_GUID,
        MeshRef {
            primitive: Primitive::Plane,
        }
    );
    insert!(
        doc,
        HYBRID_GROUND_GUID,
        Material {
            base_color: Color::new(0.30, 0.34, 0.30, 1.0),
            ..Default::default()
        }
    );

    // A directional sun (3D lighting).
    doc.create_with_guid(HYBRID_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        HYBRID_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 2.0,
        }
    );

    // A soft 2D light for the sprites.
    doc.create_with_guid(HYBRID_LIGHT2D_GUID, SpawnKind::Empty, "Fill 2D", None);
    insert!(
        doc,
        HYBRID_LIGHT2D_GUID,
        Transform::from_translation(DVec3::new(0.0, 2.0, 0.0))
    );
    insert!(
        doc,
        HYBRID_LIGHT2D_GUID,
        Light2D {
            color: Color::new(0.9, 0.9, 1.0, 1.0),
            intensity: 0.8,
            radius: 10.0,
        }
    );

    // Two billboarded sprites standing on the plane.
    for (guid, name, x, mode, tint) in [
        (
            HYBRID_SPRITE_SPHERE_GUID,
            "Billboard (Spherical)",
            -1.5,
            BillboardMode::Spherical,
            Color::new(0.9, 0.5, 0.3, 1.0),
        ),
        (
            HYBRID_SPRITE_CYL_GUID,
            "Tree (Cylindrical)",
            1.5,
            BillboardMode::Cylindrical,
            Color::new(0.4, 0.8, 0.4, 1.0),
        ),
    ] {
        doc.create_with_guid(guid, SpawnKind::Empty, name, None);
        insert!(
            doc,
            guid,
            Transform::from_translation(DVec3::new(x, 1.0, 0.0))
        );
        insert!(
            doc,
            guid,
            Sprite {
                size: Vec2d::new(1.4, 2.0),
                color: tint,
                billboard: mode,
                ..Default::default()
            }
        );
    }

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `templates/hybrid-2.5d/` directory (committed starter scene).
pub fn hybrid_template_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../templates/hybrid-2.5d")
}

/// Write the committed hybrid-2.5D template scene from [`hybrid_scene`].
pub fn write_hybrid_template() -> Result<(), String> {
    let dir = hybrid_template_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let doc = hybrid_scene();
    crate::scene::serialize::save(&doc, &dir.join("Hybrid.inf_lvl"), Some(HYBRID_LEVEL_GUID))?;
    std::fs::write(dir.join("README.md"), HYBRID_README).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

const HYBRID_README: &str = "# Hybrid 2.5D template\n\n\
The starter scene `inf new --template hybrid-2.5d` scaffolds: a 3D ground plane,\n\
a directional sun, a soft 2D light, and two **billboarded** sprites (spherical +\n\
cylindrical) — 2D cards standing in a 3D world.\n\n\
Generated by `inf_editor_core::samples::hybrid_scene`. Regenerate with\n\
`INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Committed files (fixture discipline) ─────────────────────────────────────

/// The repo-root `samples/platformer-2d/` directory.
pub fn sample_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/platformer-2d")
}

/// Encode a [`BlueprintClass`] to the deterministic `.inf_act` payload.
///
/// The `.inf_act` is stored as **pretty JSON**, not bincode: `BlueprintClass`
/// carries `#[serde(skip_serializing_if)]` fields (e.g. `parent`) that a
/// non-self-describing bincode stream cannot round-trip (the omitted field
/// desynchronizes the decoder). JSON is self-describing, deterministic (ordered
/// `Vec`s + `BTreeMap`s), and human-diffable — the right fit for a blueprint
/// document. (A future asset-DB integration for blueprints would either drop the
/// `skip_serializing_if` or keep JSON; see the P8.4 notes.)
pub fn encode_actor(class: &BlueprintClass) -> Result<Vec<u8>, String> {
    serde_json::to_vec_pretty(class).map_err(|e| format!("encode actor: {e}"))
}

/// Decode a `.inf_act` JSON payload.
pub fn decode_actor(bytes: &[u8]) -> Result<BlueprintClass, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("decode actor: {e}"))
}

/// Write the committed sample files from the generators (regeneration path).
/// Used by the blessed regeneration test; also handy for tooling.
pub fn write_sample() -> Result<(), String> {
    let dir = sample_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    // Scene (payload + sidecar) with a fixed level GUID.
    let doc = platformer_scene();
    crate::scene::serialize::save(&doc, &dir.join("Platformer.inf_lvl"), Some(LEVEL_GUID))?;

    // Actor: JSON payload (see `encode_actor` — bincode can't round-trip it).
    let class = coyote_class();
    let act_bytes = encode_actor(&class)?;
    let act_path = dir.join("Coyote.inf_act");
    std::fs::write(&act_path, &act_bytes).map_err(|e| format!("write actor: {e}"))?;

    // Its inf_asset sidecar with the **stable** [`COYOTE_ASSET_GUID`], so the
    // level's persisted `actor` binding resolves to this blueprint through the
    // AssetDb + cooked pack (P9.5 dependency edge level→blueprint).
    let side = inf_asset::AssetSidecar::new(
        inf_asset::AssetId(COYOTE_ASSET_GUID),
        inf_asset::AssetKind::Blueprint,
        inf_asset::ContentHash::of(&act_bytes),
    );
    side.save(&act_path)
        .map_err(|e| format!("write actor sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), SAMPLE_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const SAMPLE_README: &str = "# 2D Platformer sample\n\n\
Generated by `inf_editor_core::samples` — the Phase-8 gate scene. A small\n\
platformer with a **Blueprint coyote-time jump** that plays in-viewport via the\n\
interpreter.\n\n\
- `Platformer.inf_lvl` — the scene (tilemap ground + collider ledge + platform +\n\
  a kinematic character player).\n\
- `Coyote.inf_act` — the player's blueprint class (BeginPlay + Tick coyote-time\n\
  handler), stored as JSON.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coyote_class_round_trips() {
        let class = coyote_class();
        // The committed `.inf_act` encoding round-trips exactly.
        let bytes = encode_actor(&class).unwrap();
        assert_eq!(decode_actor(&bytes).unwrap(), class);
        // The handlers the Simulate loop fires are present.
        assert!(class.handler(&EventKind::Tick).is_some());
        assert!(class.handler(&EventKind::BeginPlay).is_some());
    }

    #[test]
    fn platformer_scene_saves_and_reloads_byte_identical() {
        // The P3 discipline applied to the full 2D content: save→load→save is
        // byte-identical.
        let doc = platformer_scene();
        let file1 = crate::scene::serialize::to_scene_file(&doc);
        let bytes1 = crate::scene::serialize::encode(&file1).unwrap();

        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "platformer scene must round-trip byte-identically"
        );

        // The player carries every physics + sprite component.
        let file = crate::scene::serialize::to_scene_file(&doc2);
        let player = file
            .entities
            .iter()
            .find(|r| r.guid == PLAYER_GUID)
            .expect("player present");
        assert!(player.sprite.is_some());
        // The reloaded scene keeps the tilemap ground strip.
        let tiles = file
            .entities
            .iter()
            .find(|r| r.guid == GROUND_TILES_GUID)
            .unwrap();
        assert_eq!(tiles.tilemap.as_ref().unwrap().get_tile(0, 0), 1);
    }

    #[test]
    fn hybrid_scene_saves_and_reloads_byte_identical() {
        let doc = hybrid_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        let mut doc2 = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut doc2,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc2))
                .unwrap();
        assert_eq!(
            bytes1, bytes2,
            "hybrid template scene must round-trip byte-identically"
        );
        // The two billboard sprites survive with their modes.
        let file = crate::scene::serialize::to_scene_file(&doc2);
        let sph = file
            .entities
            .iter()
            .find(|r| r.guid == HYBRID_SPRITE_SPHERE_GUID)
            .unwrap();
        assert_eq!(
            sph.sprite.as_ref().unwrap().billboard,
            BillboardMode::Spherical
        );
    }

    /// Regenerate the committed files under `INF_BLESS_SAMPLES=1`; otherwise
    /// assert the committed bytes still match the generators (fixture lock).
    #[test]
    fn committed_sample_matches_generators() {
        if std::env::var("INF_BLESS_SAMPLES").is_ok() {
            write_sample().expect("regenerate sample");
            write_hybrid_template().expect("regenerate hybrid template");
            eprintln!("samples: regenerated {}", sample_dir().display());
            return;
        }
        let dir = sample_dir();
        let lvl = dir.join("Platformer.inf_lvl");
        let act = dir.join("Coyote.inf_act");
        if !lvl.exists() || !act.exists() {
            // First run before blessing: don't fail CI spuriously.
            eprintln!("SKIP: committed sample not present yet ({})", dir.display());
            return;
        }
        let want_lvl = crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(
            &platformer_scene(),
        ))
        .unwrap();
        let got_lvl = std::fs::read(&lvl).unwrap();
        assert_eq!(
            got_lvl, want_lvl,
            "committed .inf_lvl drifted from the generator"
        );

        let want_act = encode_actor(&coyote_class()).unwrap();
        let got_act = std::fs::read(&act).unwrap();
        assert_eq!(
            got_act, want_act,
            "committed .inf_act drifted from the generator"
        );
    }
}
