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

// ── First-person template GUIDs (0x8406 block) ──
pub const FP_LEVEL_GUID: Uuid = Uuid::from_u128(0x8406_0000);
pub const FP_GROUND_GUID: Uuid = Uuid::from_u128(0x8406_0001);
pub const FP_SUN_GUID: Uuid = Uuid::from_u128(0x8406_0002);
pub const FP_PLAYER_GUID: Uuid = Uuid::from_u128(0x8406_0003);
pub const FP_CAMERA_GUID: Uuid = Uuid::from_u128(0x8406_0004);

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
            asset: None,
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
            ..Default::default()
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

/// The minimal **first-person** starter scene: a 3D ground plane, a directional
/// sun, a kinematic **Player** capsule carrying a `CharacterController3D`, and a
/// first-person **Camera** at eye height. Scaffolded by
/// `inf new --template first-person`.
///
/// Scene-only (like the hybrid template): the player + controller + camera are
/// authored, but an input-driven look/move Blueprint is a documented follow-up.
/// A keyboard mover can reuse the character demo's existing host seams
/// (`input.is_down`, `input.just_pressed`, `physics3d.move_and_slide`); true
/// mouse-look additionally needs new input host seams (mouse-delta → camera
/// yaw/pitch) that don't exist in the node kit yet — see the README.
pub fn firstperson_scene() -> SceneDoc {
    use inf_ecs::components::{
        BodyKind3D, Camera, CharacterController3D, Collider3D, ColliderShape3DKind, Light,
        LightKind, Material, MeshRef, Primitive, RigidBody3D,
    };
    use inf_ecs::math::Vec3d;

    let mut doc = SceneDoc::new();
    doc.set_title("First Person");

    // 3D ground plane.
    doc.create_with_guid(FP_GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        FP_GROUND_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::ZERO,
            scale: Vec3d::new(40.0, 1.0, 40.0),
        }
    );
    insert!(
        doc,
        FP_GROUND_GUID,
        MeshRef {
            primitive: Primitive::Plane,
            asset: None,
        }
    );
    insert!(
        doc,
        FP_GROUND_GUID,
        Material {
            base_color: Color::new(0.28, 0.30, 0.34, 1.0),
            ..Default::default()
        }
    );

    // A directional sun.
    doc.create_with_guid(FP_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        FP_SUN_GUID,
        Transform {
            translation: Vec3d::new(0.0, 30.0, 0.0),
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        FP_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // The player: a kinematic capsule with a 3D character controller.
    doc.create_with_guid(FP_PLAYER_GUID, SpawnKind::Empty, "Player", None);
    insert!(
        doc,
        FP_PLAYER_GUID,
        Transform::from_translation(DVec3::new(0.0, 1.0, 0.0))
    );
    insert!(
        doc,
        FP_PLAYER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        FP_PLAYER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.9, 0.3),
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(doc, FP_PLAYER_GUID, CharacterController3D::default());

    // A first-person camera at eye height.
    doc.create_with_guid(
        FP_CAMERA_GUID,
        SpawnKind::Empty,
        "First-Person Camera",
        None,
    );
    insert!(
        doc,
        FP_CAMERA_GUID,
        Transform::from_translation(DVec3::new(0.0, 1.7, 0.0))
    );
    insert!(doc, FP_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `templates/first-person/` directory (committed starter scene).
pub fn firstperson_template_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../templates/first-person")
}

/// Write the committed first-person template scene from [`firstperson_scene`].
pub fn write_firstperson_template() -> Result<(), String> {
    let dir = firstperson_template_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    let doc = firstperson_scene();
    crate::scene::serialize::save(&doc, &dir.join("FirstPerson.inf_lvl"), Some(FP_LEVEL_GUID))?;
    std::fs::write(dir.join("README.md"), FIRSTPERSON_README).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

const FIRSTPERSON_README: &str = "# First Person template\n\n\
The starter scene `inf new --template first-person` scaffolds: a 3D ground\n\
plane, a directional sun, a kinematic **Player** capsule with a\n\
`CharacterController3D`, and a first-person **Camera** at eye height.\n\n\
Generated by `inf_editor_core::samples::firstperson_scene`. Regenerate with\n\
`INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n\n\
## Wiring up movement (follow-up)\n\n\
The scene is authored but has no gameplay Blueprint yet. A keyboard mover can\n\
reuse the **character-demo**'s existing host seams — `input.is_down` /\n\
`input.just_pressed` for WASD + jump and `physics3d.move_and_slide` to drive the\n\
controller — via an `.inf_act` on the Player (see `character_demo_class`). True\n\
**mouse-look** additionally needs new input host seams (a mouse-delta node and a\n\
way to write the camera's yaw/pitch), which the node kit does not expose yet;\n\
that is the tracked engine follow-up for a fully playable first-person starter.\n";

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

// ── Terrain-demo gate scene (P10.6) ──────────────────────────────────────────
//
// The Phase-10-closing gate scene: a multi-tile sculpted + splat-painted terrain,
// a PCG scatter volume (noise+slope rule) referencing a committed `.inf_pcg`
// graph, a directional sun, and a camera. Committed as v4 `.inf_lvl` bytes + the
// `.inf_pcg` sidecar under `samples/terrain-demo/`. The terrain heights come from
// [`terrain_demo_height`] so the runtime gate can probe `height_at` against the
// exact generator function.

pub const TERRAIN_DEMO_LEVEL_GUID: Uuid = Uuid::from_u128(0x8403_0000);
pub const TERRAIN_DEMO_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8403_0001);
pub const TERRAIN_DEMO_PCG_GUID: Uuid = Uuid::from_u128(0x8403_0002);
pub const TERRAIN_DEMO_SUN_GUID: Uuid = Uuid::from_u128(0x8403_0003);
pub const TERRAIN_DEMO_CAMERA_GUID: Uuid = Uuid::from_u128(0x8403_0004);
/// The **asset** GUID of the committed `Scatter.inf_pcg` (its inf_asset sidecar).
/// Stable so the level's `PcgVolume.graph` ref resolves through the AssetDb /
/// cooked pack, and the cook's level→pcg dep edge ships the graph.
pub const TERRAIN_DEMO_PCG_ASSET_GUID: Uuid = Uuid::from_u128(0x8403_00AA);

/// Terrain samples-per-tile side + world spacing for the demo (small, so the
/// committed payload stays compact while genuinely multi-tile).
pub const TERRAIN_DEMO_RESOLUTION: u32 = 16;
pub const TERRAIN_DEMO_MPS: f64 = 2.0;
/// World XZ span authored (a 2×2 tile block at the above resolution/spacing).
pub const TERRAIN_DEMO_SPAN: f64 = 64.0;

/// The demo's analytic terrain height at world `(x, z)` — a gentle sine hill. The
/// runtime gate probes `TerrainData::height_at` at a grid point against this.
pub fn terrain_demo_height(x: f64, z: f64) -> f64 {
    6.0 * (x * 0.08).sin() * (z * 0.08).cos()
}

/// Build the demo's [`inf_pcg::PcgDocument`]: one layer, one rule scattering on a
/// noise-modulated gentle-slope band — a few hundred instances over the terrain.
/// Two weighted kinds so a multi-kind scatter reads as varied placeholder content.
pub fn terrain_demo_pcg_document() -> inf_pcg::PcgDocument {
    use inf_pcg::{PcgKind, PcgRule, SamplerDef};
    let sampler = SamplerDef::Multiply(
        Box::new(SamplerDef::Noise(inf_pcg::ValueNoise {
            seed: 1337,
            frequency: 0.05,
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
        })),
        Box::new(SamplerDef::Slope {
            min_deg: 0.0,
            max_deg: 32.0,
            feather_deg: 6.0,
        }),
    );
    let rule = PcgRule {
        name: "vegetation".into(),
        sampler,
        scatter: inf_pcg::ScatterParams {
            seed: 2026_0721,
            cell_size: 8.0,
            base_density: 0.5,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (0.8, 1.4),
            rotation: inf_pcg::RotationMode::RandomYaw,
            altitude_offset: 0.0,
        },
        kinds: vec![
            PcgKind {
                mesh: None,
                weight: 3.0,
            },
            PcgKind {
                mesh: None,
                weight: 1.0,
            },
        ],
    };
    inf_pcg::PcgDocument::single_layer("ground", vec![rule])
}

/// The committed `.inf_pcg` payload for the demo (document-only envelope — the
/// player evaluates from its stored lowered document).
pub fn terrain_demo_pcg_payload() -> inf_pcg::PcgAssetPayload {
    inf_pcg::PcgAssetPayload::new(terrain_demo_pcg_document())
}

/// Build the terrain-demo [`SceneDoc`]: a sculpted + painted heightfield terrain,
/// a PCG scatter volume referencing the committed graph, a directional sun, and a
/// camera framing the terrain.
pub fn terrain_demo_scene() -> SceneDoc {
    use inf_ecs::components::{Camera, Light, LightKind, PcgVolume, Terrain};

    let mut doc = SceneDoc::new();
    doc.set_title("Terrain Demo");

    // ── Terrain: a multi-tile sine hill (sculpt-level detail via write_region) +
    //    two splat-painted bands (materialized weights on some tiles, defaults on
    //    others — the sparse/materialized mix). ──
    doc.create_with_guid(TERRAIN_DEMO_TERRAIN_GUID, SpawnKind::Empty, "Terrain", None);
    // Terrain entity sits at the origin, so world XZ == terrain-local XZ (the
    // height probe + PCG height seam are then the bare generator function).
    insert!(
        doc,
        TERRAIN_DEMO_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(TERRAIN_DEMO_RESOLUTION, TERRAIN_DEMO_MPS);
        terrain.data.write_region(
            glam::DVec2::ZERO,
            glam::DVec2::splat(TERRAIN_DEMO_SPAN),
            terrain_demo_height,
        );
        // Splat band A: rock (layer 1) over the low-left quadrant.
        let _ = inf_terrain::apply_paint(
            &mut terrain.data,
            1,
            inf_terrain::BrushParams {
                center: glam::DVec2::new(16.0, 16.0),
                radius: 14.0,
                strength: 1.0,
                falloff: inf_terrain::Falloff::Plateau(0.5),
            },
        );
        // Splat band B: dirt (layer 2) over an upper strip.
        let _ = inf_terrain::apply_paint(
            &mut terrain.data,
            2,
            inf_terrain::BrushParams {
                center: glam::DVec2::new(16.0, 48.0),
                radius: 12.0,
                strength: 1.0,
                falloff: inf_terrain::Falloff::Smooth,
            },
        );
        terrain.macro_variation = 0.2;
        insert!(doc, TERRAIN_DEMO_TERRAIN_GUID, terrain);
    }

    // ── PCG scatter volume: references the committed graph; centered over the
    //    terrain so its region covers the authored footprint. ──
    doc.create_with_guid(
        TERRAIN_DEMO_PCG_GUID,
        SpawnKind::Empty,
        "Scatter Volume",
        None,
    );
    insert!(
        doc,
        TERRAIN_DEMO_PCG_GUID,
        Transform::from_translation(DVec3::new(
            TERRAIN_DEMO_SPAN * 0.5,
            0.0,
            TERRAIN_DEMO_SPAN * 0.5
        ))
    );
    insert!(
        doc,
        TERRAIN_DEMO_PCG_GUID,
        PcgVolume {
            graph: Some(TERRAIN_DEMO_PCG_ASSET_GUID),
            extent: Vec2d::new(TERRAIN_DEMO_SPAN * 0.5, TERRAIN_DEMO_SPAN * 0.5),
            seed: 0,
            ..Default::default()
        }
    );

    // ── A directional sun. ──
    doc.create_with_guid(TERRAIN_DEMO_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        TERRAIN_DEMO_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 40.0, 0.0),
            // Angle the sun so the hills cast readable shading.
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        TERRAIN_DEMO_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // ── A camera framing the terrain (the "camera note"). ──
    doc.create_with_guid(TERRAIN_DEMO_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        TERRAIN_DEMO_CAMERA_GUID,
        Transform::from_translation(DVec3::new(TERRAIN_DEMO_SPAN * 0.5, 30.0, -20.0))
    );
    insert!(doc, TERRAIN_DEMO_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/terrain-demo/` directory.
pub fn terrain_demo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/terrain-demo")
}

/// Write the committed terrain-demo files from the generators (regeneration path):
/// the v4 `.inf_lvl` (+ sidecar), the `.inf_pcg` graph (+ its inf_asset sidecar so
/// the `PcgVolume.graph` ref resolves through the AssetDb / cooked pack), + README.
pub fn write_terrain_demo() -> Result<(), String> {
    let dir = terrain_demo_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    // Level (payload + sidecar) with a fixed level GUID.
    let doc = terrain_demo_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("TerrainDemo.inf_lvl"),
        Some(TERRAIN_DEMO_LEVEL_GUID),
    )?;

    // The `.inf_pcg` graph payload + its inf_asset sidecar with the STABLE asset
    // GUID the level's PcgVolume.graph points at.
    let pcg_bytes = terrain_demo_pcg_payload()
        .encode()
        .map_err(|e| format!("encode pcg: {e}"))?;
    let pcg_path = dir.join("Scatter.inf_pcg");
    std::fs::write(&pcg_path, &pcg_bytes).map_err(|e| format!("write pcg: {e}"))?;
    let side = inf_asset::AssetSidecar::new(
        inf_asset::AssetId(TERRAIN_DEMO_PCG_ASSET_GUID),
        inf_asset::AssetKind::Pcg,
        inf_asset::ContentHash::of(&pcg_bytes),
    );
    side.save(&pcg_path)
        .map_err(|e| format!("write pcg sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), TERRAIN_DEMO_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const TERRAIN_DEMO_README: &str = "# Terrain Demo (Phase-10 gate scene)\n\n\
Generated by `inf_editor_core::samples::terrain_demo_scene` — the P10.6 gate\n\
scene: a multi-tile **sculpted + splat-painted** heightfield terrain, a **PCG\n\
scatter volume** (noise+slope rule) referencing `Scatter.inf_pcg`, a directional\n\
sun, and a camera.\n\n\
- `TerrainDemo.inf_lvl` — the scene as schema-v4 `.inf_lvl` bytes (terrain +\n\
  PcgVolume persist).\n\
- `Scatter.inf_pcg` — the scatter graph the volume evaluates on load (its\n\
  instances are a derived cache, never persisted in the level).\n\n\
The terrain heights are `terrain_demo_height(x, z)`; the runtime gate probes\n\
`TerrainData::height_at` against it. The PCG volume's `evaluated` cache is\n\
re-computed on load (editor `pcg_evaluate`, shipped/PIE player `evaluate_pcg_volumes`).\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

const SAMPLE_README: &str = "# 2D Platformer sample\n\n\
Generated by `inf_editor_core::samples` — the Phase-8 gate scene. A small\n\
platformer with a **Blueprint coyote-time jump** that plays in-viewport via the\n\
interpreter.\n\n\
- `Platformer.inf_lvl` — the scene (tilemap ground + collider ledge + platform +\n\
  a kinematic character player).\n\
- `Coyote.inf_act` — the player's blueprint class (BeginPlay + Tick coyote-time\n\
  handler), stored as JSON.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Character-demo gate scene (P11.4) ────────────────────────────────────────
//
// The Phase-11-closing gate scene: a sine-hill terrain (P10-style) and a
// **character** driven by a Blueprint across it. The character carries the full
// P11 animation/character component set — `SkeletalMesh` + `AnimStateMachine` +
// `RootMotion` + a 3D `CharacterController3D`/`Collider3D`/`RigidBody3D` — plus an
// actor blueprint that reads left/right input, moves via `physics3d.move_and_slide`
// and tracks terrain height via the `terrain.height_at` host seam, jumping on the
// `jump` action. A procedural 6-joint humanoid-ish skeleton, three programmatic
// clips (idle bob / run forward via `straight_line_clip` / jump arc), and a
// state machine (idle→run on speed>0.1, run→idle on ≤0.1, any→jump on
// jump_pressed>0.5 with exit back) are committed as `.inf_skel` / `.inf_anim` /
// `.inf_sm` sidecars. Committed as v5 `.inf_lvl` bytes under
// `samples/character-demo/`.
//
// Root motion note: the run clip is authored with forward root translation (via
// `straight_line_clip`, honest placeholder), but the entity is **state-machine
// driven** (no `AnimPlayer`), so locomotion comes from the Blueprint's
// `physics3d.move_and_slide` — root-motion *extraction* (which reads `AnimPlayer`)
// is inert this session; the `RootMotion` component persists for future use.

pub const CHARACTER_DEMO_LEVEL_GUID: Uuid = Uuid::from_u128(0x8404_0000);
pub const CHARACTER_DEMO_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8404_0001);
pub const CHARACTER_DEMO_CHARACTER_GUID: Uuid = Uuid::from_u128(0x8404_0002);
pub const CHARACTER_DEMO_SUN_GUID: Uuid = Uuid::from_u128(0x8404_0003);
pub const CHARACTER_DEMO_CAMERA_GUID: Uuid = Uuid::from_u128(0x8404_0004);
/// The committed anim asset GUIDs the level's components reference (stable so the
/// refs resolve through the AssetDb / cooked pack and the cook's dep edges ship them).
pub const CHARACTER_DEMO_SKELETON_GUID: Uuid = Uuid::from_u128(0x8404_00A0);
pub const CHARACTER_DEMO_IDLE_CLIP_GUID: Uuid = Uuid::from_u128(0x8404_00A1);
pub const CHARACTER_DEMO_RUN_CLIP_GUID: Uuid = Uuid::from_u128(0x8404_00A2);
pub const CHARACTER_DEMO_JUMP_CLIP_GUID: Uuid = Uuid::from_u128(0x8404_00A3);
pub const CHARACTER_DEMO_SM_GUID: Uuid = Uuid::from_u128(0x8404_00A4);
pub const CHARACTER_DEMO_ACTOR_GUID: Uuid = Uuid::from_u128(0x8404_00AC);

/// The actor class id string.
pub const CHARACTER_DEMO_CLASS_ID: &str = "act:character_demo";

// Terrain + tuning (world units, seconds).
pub const CHARACTER_DEMO_RESOLUTION: u32 = 16;
pub const CHARACTER_DEMO_MPS: f64 = 2.0;
/// Horizontal run speed (m/s) from left/right input.
pub const CHAR_MOVE_SPEED: f64 = 4.0;
/// Downward acceleration applied to `vy` each tick.
pub const CHAR_GRAVITY: f64 = 20.0;
/// Upward velocity a jump imparts.
pub const CHAR_JUMP_SPEED: f64 = 7.0;
/// Height the capsule centre stands above the sampled terrain height.
pub const CHAR_STAND: f64 = 0.9;
/// Start position (world X); the character begins at the terrain origin where the
/// height is exactly `0`, so its start Y is exactly `CHAR_STAND` (grounded).
pub const CHAR_START_X: f64 = 0.0;
/// Start Y = terrain height at the start (0) + the stand offset. Kept a const so
/// the spawn `Transform` and the Blueprint's `BeginPlay` position seed agree
/// exactly — the invariant that makes `transform == tracked position` hold each
/// tick (`move_and_slide` deltas telescope).
pub const CHAR_START_Y: f64 = CHAR_STAND;

/// The demo's analytic terrain height at world `(x, z)` — a gentle sine hill along
/// X (flat in Z along the character's z=0 path). `height(0,0) == 0`, so the
/// character starts grounded at `CHAR_START_Y`. The runtime gate probes
/// `TerrainData::height_at` + the character's Y against this.
pub fn character_demo_height(x: f64, z: f64) -> f64 {
    3.0 * (x * 0.08).sin() * (z * 0.08).cos()
}

/// A procedural 6-joint humanoid-ish skeleton (honest placeholder): hips (root) →
/// spine → head, plus two upper arms and a foot. Bind transforms are simple local
/// offsets; inverse-binds are identity (a placeholder rig, not an imported mesh).
pub fn character_demo_skeleton() -> inf_anim::Skeleton {
    use glam::{Mat4, Quat, Vec3};
    use inf_anim::{Joint, JointTransform};
    let joint = |name: &str, parent: Option<u16>, t: Vec3| Joint {
        name: name.into(),
        parent,
        inverse_bind: Mat4::IDENTITY.to_cols_array(),
        local_bind: JointTransform::from_trs(t, Quat::IDENTITY, Vec3::ONE),
    };
    inf_anim::Skeleton::new(vec![
        joint("hips", None, Vec3::new(0.0, 1.0, 0.0)),
        joint("spine", Some(0), Vec3::new(0.0, 0.4, 0.0)),
        joint("head", Some(1), Vec3::new(0.0, 0.4, 0.0)),
        joint("upper_arm_l", Some(1), Vec3::new(-0.25, 0.3, 0.0)),
        joint("upper_arm_r", Some(1), Vec3::new(0.25, 0.3, 0.0)),
        joint("foot", Some(0), Vec3::new(0.0, -0.9, 0.0)),
    ])
    .expect("valid procedural skeleton")
}

/// The **idle** clip: a subtle vertical bob of the hips (joint 0) over 2 s, looping.
pub fn character_demo_idle_clip() -> inf_anim::AnimClip {
    use inf_anim::{AnimClip, Interpolation, JointTrack, Vec3Track};
    let mut jt = JointTrack::new(0);
    jt.translation = Some(Vec3Track::new(
        vec![0.0, 1.0, 2.0],
        vec![[0.0, 1.0, 0.0], [0.0, 1.05, 0.0], [0.0, 1.0, 0.0]],
        Interpolation::Linear,
    ));
    AnimClip::new("idle", vec![jt])
}

/// The **run** clip: forward root motion via the `straight_line_clip` helper
/// (hips translate +X over the loop). Authored honestly even though locomotion is
/// Blueprint-driven this session (see the module note).
pub fn character_demo_run_clip() -> inf_anim::AnimClip {
    inf_anim::root_motion::straight_line_clip("run", glam::Vec3::X, 2.0, 0.6)
}

/// The **jump** clip: a vertical arc on the hips (joint 0) over 0.5 s, non-looping.
pub fn character_demo_jump_clip() -> inf_anim::AnimClip {
    use inf_anim::{AnimClip, Interpolation, JointTrack, Vec3Track};
    let mut jt = JointTrack::new(0);
    jt.translation = Some(Vec3Track::new(
        vec![0.0, 0.25, 0.5],
        vec![[0.0, 1.0, 0.0], [0.0, 1.6, 0.0], [0.0, 1.0, 0.0]],
        Interpolation::Linear,
    ));
    AnimClip::new("jump", vec![jt])
}

/// The state machine: idle(0) / run(1) / jump(2). Jump transitions are declared
/// **first** so a jump pressed while moving wins over the run transition. Reads
/// the actor's `speed` + `jump` Blueprint variables via the [`SmContext`] seam.
pub fn character_demo_state_machine() -> inf_anim::StateMachine {
    use inf_anim::state_machine::{CmpOp, Motion, SmCondition, SmState, SmTransition};
    let clip_ref = |g: Uuid| *g.as_bytes();
    let cond = |var: &str, op: CmpOp, value: f64| SmCondition {
        var: var.into(),
        op,
        value,
    };
    let tr = |from: usize, to: usize, c: SmCondition| SmTransition {
        from,
        to,
        duration: 0.15,
        conditions: vec![c],
        exit_time: None,
    };
    inf_anim::StateMachine {
        states: vec![
            SmState {
                name: "idle".into(),
                motion: Motion::Clip(clip_ref(CHARACTER_DEMO_IDLE_CLIP_GUID)),
                looping: true,
                speed: 1.0,
                position: (0.0, 0.0),
            },
            SmState {
                name: "run".into(),
                motion: Motion::Clip(clip_ref(CHARACTER_DEMO_RUN_CLIP_GUID)),
                looping: true,
                speed: 1.0,
                position: (240.0, 0.0),
            },
            SmState {
                name: "jump".into(),
                motion: Motion::Clip(clip_ref(CHARACTER_DEMO_JUMP_CLIP_GUID)),
                looping: false,
                speed: 1.0,
                position: (120.0, -160.0),
            },
        ],
        transitions: vec![
            // any→jump (declared first so jump wins over run when both hold).
            tr(0, 2, cond("jump", CmpOp::Gt, 0.5)),
            tr(1, 2, cond("jump", CmpOp::Gt, 0.5)),
            // locomotion.
            tr(0, 1, cond("speed", CmpOp::Gt, 0.1)),
            tr(1, 0, cond("speed", CmpOp::Le, 0.1)),
            // exit jump back to run (moving) or idle (stopped).
            tr(2, 1, cond("speed", CmpOp::Gt, 0.1)),
            tr(2, 0, cond("speed", CmpOp::Le, 0.1)),
        ],
        entry: 0,
    }
}

/// The character's **Tick** handler as `BlueprintFn` IR. Reads left/right + jump
/// input, integrates a var-tracked position, clamps Y to `terrain.height_at + STAND`
/// (gravity + grounding), sets the `speed`/`jump` vars the state machine reads, and
/// drives the entity with `physics3d.move_and_slide` deltas.
///
/// Locals: `n1=entity`, `n2=vx` (mut), `n3=old_py`, `n4=ground`.
fn character_tick_fn() -> BlueprintFn {
    let e = || local(1);
    let dt = || Expr::Param("dt".to_string());
    let vx = || local(2);
    let old_py = || local(3);
    let ground = || local(4);

    let body = vec![
        let_named(1, "entity", false, get_var("entity")),
        // Horizontal velocity from held input.
        let_named(2, "vx", true, float_lit(0.0)),
        if_then(
            call(&["input", "is_down"], vec![str_lit("right")]),
            vec![Stmt::Assign {
                target: LocalId(2),
                value: bin(BinOp::Add, vx(), float_lit(CHAR_MOVE_SPEED)),
            }],
            vec![],
        ),
        if_then(
            call(&["input", "is_down"], vec![str_lit("left")]),
            vec![Stmt::Assign {
                target: LocalId(2),
                value: bin(BinOp::Sub, vx(), float_lit(CHAR_MOVE_SPEED)),
            }],
            vec![],
        ),
        // speed = |vx| → drives idle↔run.
        if_then(
            bin(BinOp::Lt, vx(), float_lit(0.0)),
            vec![set_var("speed", bin(BinOp::Sub, float_lit(0.0), vx()))],
            vec![set_var("speed", vx())],
        ),
        // Jump on the rising edge while grounded → seed vy + the jump var.
        if_then(
            bin(
                BinOp::And,
                call(&["input", "just_pressed"], vec![str_lit("jump")]),
                bin(BinOp::Gt, get_var("grounded"), float_lit(0.5)),
            ),
            vec![
                set_var("vy", float_lit(CHAR_JUMP_SPEED)),
                set_var("jump", float_lit(1.0)),
            ],
            vec![set_var("jump", float_lit(0.0))],
        ),
        // Gravity.
        set_var(
            "vy",
            bin(
                BinOp::Sub,
                get_var("vy"),
                bin(BinOp::Mul, float_lit(CHAR_GRAVITY), dt()),
            ),
        ),
        // Integrate the var-tracked position.
        let_named(3, "old_py", false, get_var("py")),
        set_var(
            "px",
            bin(BinOp::Add, get_var("px"), bin(BinOp::Mul, vx(), dt())),
        ),
        set_var(
            "py",
            bin(BinOp::Add, old_py(), bin(BinOp::Mul, get_var("vy"), dt())),
        ),
        // Ground = terrain height under the character + the stand offset.
        let_named(
            4,
            "ground",
            false,
            bin(
                BinOp::Add,
                call(
                    &["terrain", "height_at"],
                    vec![get_var("px"), get_var("pz")],
                ),
                float_lit(CHAR_STAND),
            ),
        ),
        // Clamp to the ground when at/under it; else airborne.
        if_then(
            bin(BinOp::Le, get_var("py"), ground()),
            vec![
                set_var("py", ground()),
                set_var("vy", float_lit(0.0)),
                set_var("grounded", float_lit(1.0)),
            ],
            vec![set_var("grounded", float_lit(0.0))],
        ),
        // Move the entity by this tick's delta (x + the y delta toward the target).
        Stmt::ExprStmt(call(
            &["physics3d", "move_and_slide"],
            vec![
                e(),
                bin(BinOp::Mul, vx(), dt()),
                bin(BinOp::Sub, get_var("py"), old_py()),
                float_lit(0.0),
            ],
        )),
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

/// A `BeginPlay` handler seeding the var-tracked position to the spawn position
/// (so `transform == tracked position` holds), grounded and at rest.
fn character_begin_play_fn() -> BlueprintFn {
    BlueprintFn {
        id: EventKind::BeginPlay.key(),
        name: EventKind::BeginPlay.key(),
        params: vec![],
        ret: Ty::Unit,
        body: vec![
            set_var("px", float_lit(CHAR_START_X)),
            set_var("py", float_lit(CHAR_START_Y)),
            set_var("pz", float_lit(0.0)),
            set_var("vy", float_lit(0.0)),
            set_var("speed", float_lit(0.0)),
            set_var("jump", float_lit(0.0)),
            set_var("grounded", float_lit(1.0)),
        ],
    }
}

/// The character's [`BlueprintClass`] (the `.inf_act`).
pub fn character_demo_class() -> BlueprintClass {
    let mut class = BlueprintClass::new(CHARACTER_DEMO_CLASS_ID, "Character Demo");
    let fvar = |name: &str| Variable {
        name: name.into(),
        ty: Ty::Float,
        default: Lit::Float(0.0),
        exposed: true,
    };
    class.variables = vec![
        Variable {
            name: "entity".into(),
            ty: Ty::Int,
            default: Lit::Int(0),
            exposed: false,
        },
        fvar("px"),
        fvar("py"),
        fvar("pz"),
        fvar("vy"),
        fvar("speed"),
        fvar("jump"),
        fvar("grounded"),
    ];
    class.events = vec![
        EventBinding {
            event: EventKind::BeginPlay,
            body: character_begin_play_fn(),
        },
        EventBinding {
            event: EventKind::Tick,
            body: character_tick_fn(),
        },
    ];
    class
}

/// Build the character-demo [`SceneDoc`]: a sine-hill terrain, a character entity
/// carrying the full P11 animation/character component set + the actor binding, a
/// directional sun, and a camera.
pub fn character_demo_scene() -> SceneDoc {
    use inf_ecs::components::{
        AnimStateMachine, BodyKind3D, Camera, CharacterController3D, Collider3D,
        ColliderShape3DKind, Light, LightKind, RigidBody3D, RootMotion, SkeletalMesh, Terrain,
    };

    let mut doc = SceneDoc::new();
    doc.set_title("Character Demo");

    // ── Terrain: a sine hill at the origin (world XZ == terrain-local XZ, so the
    //    height probe + the character's terrain.height_at seam are the bare
    //    generator function). Authored over the character's path. ──
    doc.create_with_guid(
        CHARACTER_DEMO_TERRAIN_GUID,
        SpawnKind::Empty,
        "Terrain",
        None,
    );
    insert!(
        doc,
        CHARACTER_DEMO_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(CHARACTER_DEMO_RESOLUTION, CHARACTER_DEMO_MPS);
        terrain.data.write_region(
            glam::DVec2::new(-16.0, -16.0),
            glam::DVec2::new(48.0, 16.0),
            character_demo_height,
        );
        terrain.macro_variation = 0.15;
        insert!(doc, CHARACTER_DEMO_TERRAIN_GUID, terrain);
    }

    // ── The character: SkeletalMesh + AnimStateMachine + RootMotion + a 3D
    //    kinematic character controller, standing at the origin (grounded). ──
    doc.create_with_guid(
        CHARACTER_DEMO_CHARACTER_GUID,
        SpawnKind::Empty,
        "Character",
        None,
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        Transform::from_translation(DVec3::new(CHAR_START_X, CHAR_START_Y, 0.0))
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        SkeletalMesh {
            mesh: None,
            skeleton: Some(CHARACTER_DEMO_SKELETON_GUID),
        }
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        AnimStateMachine {
            sm: Some(CHARACTER_DEMO_SM_GUID),
            params_from_vars: true,
            ..Default::default()
        }
    );
    insert!(doc, CHARACTER_DEMO_CHARACTER_GUID, RootMotion::apply());
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: inf_ecs::math::Vec3d::new(0.3, 0.5, 0.3),
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        CharacterController3D::default()
    );
    insert!(
        doc,
        CHARACTER_DEMO_CHARACTER_GUID,
        ActorClass(CHARACTER_DEMO_ACTOR_GUID)
    );

    // ── A directional sun. ──
    doc.create_with_guid(CHARACTER_DEMO_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        CHARACTER_DEMO_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 30.0, 0.0),
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        CHARACTER_DEMO_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // ── A camera framing the character. ──
    doc.create_with_guid(CHARACTER_DEMO_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        CHARACTER_DEMO_CAMERA_GUID,
        Transform::from_translation(DVec3::new(6.0, 4.0, -8.0))
    );
    insert!(doc, CHARACTER_DEMO_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The `(guid, class)` actor list for a headless Simulate of the character demo.
pub fn character_demo_actors() -> Vec<(Uuid, BlueprintClass)> {
    vec![(CHARACTER_DEMO_CHARACTER_GUID, character_demo_class())]
}

/// The repo-root `samples/character-demo/` directory.
pub fn character_demo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/character-demo")
}

/// Encode an [`AssetPayload`](inf_asset::AssetPayload) + write it with an inf_asset
/// sidecar stamped with a stable GUID (the character-demo asset-writing helper).
fn write_anim_asset<T: inf_asset::AssetPayload>(
    dir: &std::path::Path,
    file: &str,
    guid: Uuid,
    kind: inf_asset::AssetKind,
    payload: &T,
) -> Result<(), String> {
    let bytes = inf_asset::encode(payload).map_err(|e| format!("encode {file}: {e}"))?;
    let path = dir.join(file);
    std::fs::write(&path, &bytes).map_err(|e| format!("write {file}: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(guid),
        kind,
        inf_asset::ContentHash::of(&bytes),
    )
    .save(&path)
    .map_err(|e| format!("write {file} sidecar: {e}"))
}

/// Write the committed character-demo files from the generators (regeneration
/// path): the v5 `.inf_lvl` (+ sidecar), the `.inf_skel` / three `.inf_anim` /
/// `.inf_sm` anim assets (+ sidecars with their stable GUIDs), the `.inf_act`
/// actor (+ sidecar), and a README.
pub fn write_character_demo() -> Result<(), String> {
    use inf_anim::{AnimClipAsset, SkeletonAsset, StateMachineAsset};
    use inf_asset::AssetKind;

    let dir = character_demo_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    // Level (payload + sidecar).
    let doc = character_demo_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("Character.inf_lvl"),
        Some(CHARACTER_DEMO_LEVEL_GUID),
    )?;

    // Skeleton.
    let skel_bytes_ref = *CHARACTER_DEMO_SKELETON_GUID.as_bytes();
    write_anim_asset(
        &dir,
        "Character.inf_skel",
        CHARACTER_DEMO_SKELETON_GUID,
        AssetKind::Skeleton,
        &SkeletonAsset::new(character_demo_skeleton()),
    )?;

    // Three clips (each bound to the skeleton GUID — a dep edge).
    for (file, guid, clip) in [
        (
            "Idle.inf_anim",
            CHARACTER_DEMO_IDLE_CLIP_GUID,
            character_demo_idle_clip(),
        ),
        (
            "Run.inf_anim",
            CHARACTER_DEMO_RUN_CLIP_GUID,
            character_demo_run_clip(),
        ),
        (
            "Jump.inf_anim",
            CHARACTER_DEMO_JUMP_CLIP_GUID,
            character_demo_jump_clip(),
        ),
    ] {
        write_anim_asset(
            &dir,
            file,
            guid,
            AssetKind::AnimClip,
            &AnimClipAsset::new(clip, Some(skel_bytes_ref)),
        )?;
    }

    // State machine (bound to the skeleton GUID; references the clip GUIDs).
    write_anim_asset(
        &dir,
        "Locomotion.inf_sm",
        CHARACTER_DEMO_SM_GUID,
        AssetKind::StateMachine,
        &StateMachineAsset::new(character_demo_state_machine(), Some(skel_bytes_ref)),
    )?;

    // Actor blueprint (JSON, like Coyote) + its inf_asset sidecar.
    let class = character_demo_class();
    let act_bytes = encode_actor(&class)?;
    let act_path = dir.join("Character.inf_act");
    std::fs::write(&act_path, &act_bytes).map_err(|e| format!("write actor: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(CHARACTER_DEMO_ACTOR_GUID),
        AssetKind::Blueprint,
        inf_asset::ContentHash::of(&act_bytes),
    )
    .save(&act_path)
    .map_err(|e| format!("write actor sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), CHARACTER_DEMO_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const CHARACTER_DEMO_README: &str = "# Character Demo (Phase-11 gate scene)\n\n\
Generated by `inf_editor_core::samples::character_demo_scene` — the P11.4 gate\n\
scene: an idle/run/jump **state-machine character** driven by a Blueprint across a\n\
sine-hill terrain (the P10→P11 capstone).\n\n\
- `Character.inf_lvl` — the scene as schema-v5 `.inf_lvl` bytes (SkeletalMesh +\n\
  AnimStateMachine + RootMotion + a 3D character controller persist).\n\
- `Character.inf_skel` — a procedural 6-joint humanoid-ish skeleton.\n\
- `Idle/Run/Jump.inf_anim` — three programmatic clips (bob / forward root motion /\n\
  vertical arc).\n\
- `Locomotion.inf_sm` — the state machine (idle→run on speed>0.1, run→idle on ≤0.1,\n\
  any→jump on jump>0.5 with exit back).\n\
- `Character.inf_act` — the actor blueprint (left/right → move_and_slide across the\n\
  terrain via `terrain.height_at`; jump on the rising edge).\n\n\
The terrain heights are `character_demo_height(x, z)`; the runtime gate scripts\n\
input and asserts the character crosses the terrain (x advances, Y tracks the\n\
height), jumps (Y rises then returns), and its state machine transitions\n\
idle→run→jump. PIE == shipping (identical trace/probes on both paths).\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Physics-playground gate scene (P12.4) ────────────────────────────────────
//
// The Phase-12-closing gate scene: a committed 3D physics playground composing
// every P12 feature at once — box stacks, a motorized revolute spinner, a
// distance-rope pendulum, a prismatic slider, a CCD bullet vs a thin wall, a
// collision-layer ghost pair, a sensor plate, and a small **ragdoll**
// (`inf_physics::ragdoll::build_ragdoll` output, its descs mapped to `Joint3D`
// components) — plus TWO spatial `AudioSource`s (one autoplay-looping on the
// spinner with occlusion, one on the sensor plate) and an `AudioListener` on a
// camera. All persisted as schema-v6 `.inf_lvl` bytes under
// `samples/physics-playground/`, with the two `.inf_audio` clips beside it.
//
// The in-code determinism guarantee is `inf-physics`'s
// `playground_determinism.rs`; this is the same composition as **committed
// content**, run through the cook/PIE pipeline by the runtime gate
// (`runtime/inf-player/tests/physics_demo.rs`): 300 fixed steps twice yields a
// byte-identical pose trace + identical audio command stream, PIE == shipping.
//
// Joints/audio persist because P12.4 bumped the `.inf_lvl` schema to v6 (the
// `joint_2d`/`joint_3d`/`audio_source`/`audio_listener` slots); the physics bridge
// reconciles the joints from the ECS components each step, so authoring
// `RigidBody3D` + `Collider3D` + `Joint3D` is sufficient to simulate them.

pub const PLAYGROUND_LEVEL_GUID: Uuid = Uuid::from_u128(0x8405_0000);
pub const PLAYGROUND_GROUND_GUID: Uuid = Uuid::from_u128(0x8405_0001);
pub const PLAYGROUND_SPINNER_HUB_GUID: Uuid = Uuid::from_u128(0x8405_0020);
pub const PLAYGROUND_SPINNER_WHEEL_GUID: Uuid = Uuid::from_u128(0x8405_0021);
pub const PLAYGROUND_PENDULUM_ANCHOR_GUID: Uuid = Uuid::from_u128(0x8405_0030);
pub const PLAYGROUND_PENDULUM_BOB_GUID: Uuid = Uuid::from_u128(0x8405_0031);
pub const PLAYGROUND_SLIDER_RAIL_GUID: Uuid = Uuid::from_u128(0x8405_0040);
pub const PLAYGROUND_SLIDER_GUID: Uuid = Uuid::from_u128(0x8405_0041);
pub const PLAYGROUND_BULLET_WALL_GUID: Uuid = Uuid::from_u128(0x8405_0050);
pub const PLAYGROUND_BULLET_GUID: Uuid = Uuid::from_u128(0x8405_0051);
pub const PLAYGROUND_GHOST_A_GUID: Uuid = Uuid::from_u128(0x8405_0060);
pub const PLAYGROUND_GHOST_B_GUID: Uuid = Uuid::from_u128(0x8405_0061);
pub const PLAYGROUND_SENSOR_GUID: Uuid = Uuid::from_u128(0x8405_0070);
pub const PLAYGROUND_SENSOR_PROBE_GUID: Uuid = Uuid::from_u128(0x8405_0071);
pub const PLAYGROUND_CAMERA_GUID: Uuid = Uuid::from_u128(0x8405_0090);
/// First box of the stack (subsequent boxes are `+ i`).
pub const PLAYGROUND_STACK_BASE_GUID: u128 = 0x8405_0010;
/// First ragdoll part (subsequent parts are `+ i`, in `build_ragdoll` order).
pub const PLAYGROUND_RAGDOLL_BASE_GUID: u128 = 0x8405_0080;
/// The two committed `.inf_audio` clip asset GUIDs (stable so the AudioSource
/// `clip` refs resolve through the AssetDb / cooked pack, and the cook's
/// level→audio dep edge ships them).
pub const PLAYGROUND_SPINNER_CLIP_GUID: Uuid = Uuid::from_u128(0x8405_00A0);
pub const PLAYGROUND_SENSOR_CLIP_GUID: Uuid = Uuid::from_u128(0x8405_00A1);

/// The number of dynamic boxes in the settling stack.
pub const PLAYGROUND_STACK_COUNT: usize = 5;

/// Map a ragdoll [`inf_physics::d3::JointDesc3D`] onto a persisted [`Joint3D`]
/// component (the "descs mapped to components" step). The other body is `other`.
fn joint3d_from_ragdoll(
    other: Uuid,
    desc: inf_physics::d3::JointDesc3D,
) -> inf_ecs::components::Joint3D {
    use inf_ecs::components::{Joint3D, JointKind3D as EK};
    use inf_physics::d3::JointKind3D as PK;
    let mut j = Joint3D {
        other: inf_ecs::EntityRef::new(other),
        local_anchor: desc.local_anchor1.into(),
        other_anchor: desc.local_anchor2.into(),
        ..Default::default()
    };
    match desc.kind {
        PK::Fixed => j.kind = EK::Fixed,
        PK::Spherical => j.kind = EK::Spherical,
        PK::Distance { max_distance } => {
            j.kind = EK::Distance;
            j.max_distance = max_distance;
        }
        PK::Revolute {
            axis,
            limits,
            motor,
        } => {
            j.kind = EK::Revolute;
            j.axis = axis.into();
            if let Some([lo, hi]) = limits {
                j.limits_enabled = true;
                j.limit_min = lo;
                j.limit_max = hi;
            }
            if let Some(m) = motor {
                j.motor_enabled = true;
                j.motor_target_pos = m.target_pos;
                j.motor_target_vel = m.target_vel;
                j.motor_stiffness = m.stiffness;
                j.motor_damping = m.damping;
                j.motor_max_force = m.max_force;
            }
        }
        PK::Prismatic {
            axis,
            limits,
            motor,
        } => {
            j.kind = EK::Prismatic;
            j.axis = axis.into();
            if let Some([lo, hi]) = limits {
                j.limits_enabled = true;
                j.limit_min = lo;
                j.limit_max = hi;
            }
            if let Some(m) = motor {
                j.motor_enabled = true;
                j.motor_target_pos = m.target_pos;
                j.motor_target_vel = m.target_vel;
                j.motor_stiffness = m.stiffness;
                j.motor_damping = m.damping;
                j.motor_max_force = m.max_force;
            }
        }
    }
    j
}

/// The small humanoid skeleton fed to [`build_ragdoll`] (world-space bone
/// endpoints of a figure standing at world x = 30). Names classify to Hips /
/// Spine / Chest / Head / UpperArm{L,R} / Thigh{L,R} → 8 bodies + 7 joints.
pub fn playground_ragdoll_skeleton() -> Vec<inf_physics::ragdoll::RagdollBone> {
    use inf_physics::ragdoll::RagdollBone;
    let x = 30.0;
    vec![
        RagdollBone::new("hips", DVec3::new(x, 2.0, 0.0), DVec3::new(x, 2.3, 0.0)),
        RagdollBone::new("spine", DVec3::new(x, 2.3, 0.0), DVec3::new(x, 2.7, 0.0)),
        RagdollBone::new("chest", DVec3::new(x, 2.7, 0.0), DVec3::new(x, 3.1, 0.0)),
        RagdollBone::new("head", DVec3::new(x, 3.1, 0.0), DVec3::new(x, 3.5, 0.0)),
        RagdollBone::new(
            "upperarm_l",
            DVec3::new(x - 0.1, 3.0, 0.0),
            DVec3::new(x - 0.6, 3.0, 0.0),
        ),
        RagdollBone::new(
            "upperarm_r",
            DVec3::new(x + 0.1, 3.0, 0.0),
            DVec3::new(x + 0.6, 3.0, 0.0),
        ),
        RagdollBone::new(
            "thigh_l",
            DVec3::new(x - 0.15, 2.0, 0.0),
            DVec3::new(x - 0.15, 1.3, 0.0),
        ),
        RagdollBone::new(
            "thigh_r",
            DVec3::new(x + 0.15, 2.0, 0.0),
            DVec3::new(x + 0.15, 1.3, 0.0),
        ),
    ]
}

/// Build the physics-playground [`SceneDoc`]. See the module note for the layout.
pub fn physics_playground_scene() -> SceneDoc {
    use inf_ecs::components::{
        AudioListener, AudioSource, BodyKind3D, Camera, Collider3D, ColliderShape3DKind,
        DistanceModel, Joint3D, JointKind3D, Light, LightKind, RigidBody3D,
    };
    use inf_ecs::math::Vec3d;
    use inf_physics::ragdoll::{build_ragdoll, RagdollConfig};

    let mut doc = SceneDoc::new();
    doc.set_title("Physics Playground");

    // Helpers to cut the boilerplate.
    let static_body = || RigidBody3D {
        kind: BodyKind3D::Static,
        ..Default::default()
    };
    let box_collider = |half: Vec3d| Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: half,
        ..Default::default()
    };

    // ── Ground slab (static box; top at y = 0). ──
    doc.create_with_guid(PLAYGROUND_GROUND_GUID, SpawnKind::Empty, "Ground", None);
    insert!(
        doc,
        PLAYGROUND_GROUND_GUID,
        Transform::from_translation(DVec3::new(0.0, -0.5, 0.0))
    );
    insert!(doc, PLAYGROUND_GROUND_GUID, static_body());
    insert!(
        doc,
        PLAYGROUND_GROUND_GUID,
        box_collider(Vec3d::new(48.0, 0.5, 48.0))
    );

    // ── A settling box stack (5 dynamic boxes at x = 0). ──
    for i in 0..PLAYGROUND_STACK_COUNT {
        let guid = Uuid::from_u128(PLAYGROUND_STACK_BASE_GUID + i as u128);
        doc.create_with_guid(guid, SpawnKind::Empty, &format!("Box {i}"), None);
        insert!(
            doc,
            guid,
            Transform::from_translation(DVec3::new(0.0, 0.5 + i as f64 * 1.02, 0.0))
        );
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            }
        );
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::splat(0.5),
                friction: 0.7,
                ..Default::default()
            }
        );
    }

    // ── A motorized revolute spinner (hub static, wheel dynamic, at x = 8). The
    //    wheel also carries the autoplay-looping, occluded spatial AudioSource. ──
    doc.create_with_guid(
        PLAYGROUND_SPINNER_HUB_GUID,
        SpawnKind::Empty,
        "Spinner Hub",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_HUB_GUID,
        Transform::from_translation(DVec3::new(8.0, 4.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_HUB_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    doc.create_with_guid(
        PLAYGROUND_SPINNER_WHEEL_GUID,
        SpawnKind::Empty,
        "Spinner Wheel",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        Transform::from_translation(DVec3::new(8.0, 4.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(0.4),
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        Joint3D {
            other: inf_ecs::EntityRef::new(PLAYGROUND_SPINNER_HUB_GUID),
            kind: JointKind3D::Revolute,
            axis: Vec3d::new(0.0, 0.0, 1.0),
            motor_enabled: true,
            motor_target_vel: 8.0,
            motor_damping: 1.0,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SPINNER_WHEEL_GUID,
        AudioSource {
            clip: Some(PLAYGROUND_SPINNER_CLIP_GUID),
            bus: "sfx".to_string(),
            volume: 0.8,
            pitch: 1.0,
            looping: true,
            spatial: true,
            min_distance: 2.0,
            max_distance: 40.0,
            distance_model: DistanceModel::Inverse,
            rolloff: 1.0,
            occlusion: true,
            autoplay: true,
        }
    );

    // ── A distance-rope pendulum (anchor static, bob dynamic, at x = -8). ──
    doc.create_with_guid(
        PLAYGROUND_PENDULUM_ANCHOR_GUID,
        SpawnKind::Empty,
        "Rope Anchor",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_ANCHOR_GUID,
        Transform::from_translation(DVec3::new(-8.0, 6.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_ANCHOR_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    doc.create_with_guid(
        PLAYGROUND_PENDULUM_BOB_GUID,
        SpawnKind::Empty,
        "Rope Bob",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_BOB_GUID,
        // Offset horizontally so the taut rope swings (a real pendulum).
        Transform::from_translation(DVec3::new(-7.0, 4.8, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_BOB_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_BOB_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_PENDULUM_BOB_GUID,
        Joint3D {
            other: inf_ecs::EntityRef::new(PLAYGROUND_PENDULUM_ANCHOR_GUID),
            kind: JointKind3D::Distance,
            max_distance: 1.5,
            ..Default::default()
        }
    );

    // ── A prismatic slider under gravity with limits (at x = 12). ──
    doc.create_with_guid(
        PLAYGROUND_SLIDER_RAIL_GUID,
        SpawnKind::Empty,
        "Slider Rail",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_RAIL_GUID,
        Transform::from_translation(DVec3::new(12.0, 6.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_RAIL_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    doc.create_with_guid(PLAYGROUND_SLIDER_GUID, SpawnKind::Empty, "Slider", None);
    insert!(
        doc,
        PLAYGROUND_SLIDER_GUID,
        Transform::from_translation(DVec3::new(12.0, 6.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SLIDER_GUID,
        Joint3D {
            other: inf_ecs::EntityRef::new(PLAYGROUND_SLIDER_RAIL_GUID),
            kind: JointKind3D::Prismatic,
            axis: Vec3d::new(0.0, 1.0, 0.0),
            limits_enabled: true,
            limit_min: -2.0,
            limit_max: 0.0,
            ..Default::default()
        }
    );

    // ── A CCD bullet aimed (by fast gravity) at a thin horizontal wall (x = 24).
    //    Without CCD the fast body tunnels the 0.04-thick plate; with it, it stops. ──
    doc.create_with_guid(
        PLAYGROUND_BULLET_WALL_GUID,
        SpawnKind::Empty,
        "Thin Wall",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_WALL_GUID,
        Transform::from_translation(DVec3::new(24.0, 5.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_WALL_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_WALL_GUID,
        box_collider(Vec3d::new(2.0, 0.02, 2.0))
    );
    doc.create_with_guid(PLAYGROUND_BULLET_GUID, SpawnKind::Empty, "CCD Bullet", None);
    insert!(
        doc,
        PLAYGROUND_BULLET_GUID,
        Transform::from_translation(DVec3::new(24.0, 22.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            // A heavy gravity scale accelerates it to a tunnelling speed fast.
            gravity_scale: 12.0,
            ccd_enabled: true,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_BULLET_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.15,
            ..Default::default()
        }
    );

    // ── A collision-layer ghost PAIR: two dynamic spheres at the SAME point, each
    //    with an empty collision filter (interact with nothing) → they free-fall in
    //    lockstep, perfectly co-located, interpenetrating unimpeded (the layer
    //    proof: were the filters non-empty the contact solver would shove them
    //    apart, and the floor would stop them). ──
    for (guid, name) in [
        (PLAYGROUND_GHOST_A_GUID, "Ghost A"),
        (PLAYGROUND_GHOST_B_GUID, "Ghost B"),
    ] {
        doc.create_with_guid(guid, SpawnKind::Empty, name, None);
        insert!(
            doc,
            guid,
            Transform::from_translation(DVec3::new(-16.0, 8.0, 0.0))
        );
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            }
        );
        insert!(
            doc,
            guid,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: 0.4,
                // Membership present but an EMPTY filter → collides with nothing.
                collision_memberships: 0b10,
                collision_filter: 0,
                ..Default::default()
            }
        );
    }

    // ── A sensor plate (static trigger volume, x = 16) with the second AudioSource,
    //    plus a probe ball that falls THROUGH it (a sensor generates no force) and
    //    lands on the ground — proving the plate is non-blocking. ──
    doc.create_with_guid(
        PLAYGROUND_SENSOR_GUID,
        SpawnKind::Empty,
        "Sensor Plate",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_GUID,
        Transform::from_translation(DVec3::new(16.0, 1.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(1.5, 0.5, 1.5),
            sensor: true,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_GUID,
        AudioSource {
            clip: Some(PLAYGROUND_SENSOR_CLIP_GUID),
            bus: "sfx".to_string(),
            volume: 0.6,
            pitch: 1.0,
            looping: false,
            spatial: true,
            min_distance: 1.0,
            max_distance: 30.0,
            distance_model: DistanceModel::Inverse,
            rolloff: 1.0,
            occlusion: false,
            autoplay: true,
        }
    );
    doc.create_with_guid(
        PLAYGROUND_SENSOR_PROBE_GUID,
        SpawnKind::Empty,
        "Sensor Probe",
        None,
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_PROBE_GUID,
        Transform::from_translation(DVec3::new(16.0, 5.0, 0.0))
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_PROBE_GUID,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        PLAYGROUND_SENSOR_PROBE_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.25,
            ..Default::default()
        }
    );

    // ── A small ragdoll from `build_ragdoll` (its descs mapped to components). ──
    let parts = build_ragdoll(&playground_ragdoll_skeleton(), RagdollConfig::default());
    // Stable per-part GUIDs (index order == build_ragdoll's parents-first order).
    let part_guid = |i: usize| Uuid::from_u128(PLAYGROUND_RAGDOLL_BASE_GUID + i as u128);
    for (i, part) in parts.iter().enumerate() {
        let guid = part_guid(i);
        doc.create_with_guid(
            guid,
            SpawnKind::Empty,
            &format!("Ragdoll {}", part.name),
            None,
        );
        let mut t = Transform::from_translation(part.position);
        t.set_quat(part.rotation);
        insert!(doc, guid, t);
        insert!(
            doc,
            guid,
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..Default::default()
            }
        );
        // The capsule spanning the bone (build_ragdoll always emits a Capsule).
        if let inf_physics::ColliderShape3D::Capsule {
            half_height,
            radius,
        } = part.collider.shape
        {
            insert!(
                doc,
                guid,
                Collider3D {
                    shape_kind: ColliderShape3DKind::Capsule,
                    half_extents: Vec3d::new(radius, half_height, radius),
                    radius,
                    density: part.collider.density,
                    friction: part.collider.friction,
                    ..Default::default()
                }
            );
        }
        // The joint to the parent part (root has none).
        if let Some(rj) = &part.joint {
            let other = part_guid(rj.parent);
            insert!(doc, guid, joint3d_from_ragdoll(other, rj.desc));
        }
    }

    // ── A camera carrying the active AudioListener (the sim reads its pose). ──
    doc.create_with_guid(PLAYGROUND_CAMERA_GUID, SpawnKind::Empty, "Camera", None);
    insert!(
        doc,
        PLAYGROUND_CAMERA_GUID,
        Transform::from_translation(DVec3::new(0.0, 6.0, -15.0))
    );
    insert!(doc, PLAYGROUND_CAMERA_GUID, Camera::default());
    insert!(doc, PLAYGROUND_CAMERA_GUID, AudioListener { active: true });

    // A directional sun so the playground reads (rendering is human-verified).
    let sun = Uuid::from_u128(0x8405_0002);
    doc.create_with_guid(sun, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        sun,
        Transform {
            translation: Vec3d::new(0.0, 30.0, 0.0),
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        sun,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // The 3D physics gravity flows from `gravity_2d.y` (the runtime sim wires the
    // 3D bridge to it), so the playground makes it explicit real-world down.
    doc.set_settings(crate::scene::serialize::LevelSettings {
        gravity_2d: Vec2d::new(0.0, -9.81),
        gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
        sim_hz: 60.0,
        ..Default::default()
    });

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/physics-playground/` directory.
pub fn physics_playground_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/physics-playground")
}

/// A minimal valid 16-bit mono PCM WAV (decodable by kira headlessly), so the
/// committed `.inf_audio` clips need no binary fixture. Ported from the
/// `inf-audio` payload test's `tone_wav`.
fn tone_wav(samples: usize, sample_rate: u32) -> Vec<u8> {
    let bits = 16u16;
    let channels = 1u16;
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * block_align as u32;
    let data_len = samples as u32 * block_align as u32;
    let mut w = Vec::new();
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&channels.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&byte_rate.to_le_bytes());
    w.extend_from_slice(&block_align.to_le_bytes());
    w.extend_from_slice(&bits.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..samples {
        let s = (i as i16).wrapping_mul(64);
        w.extend_from_slice(&s.to_le_bytes());
    }
    w
}

/// The committed `AudioAsset` for the given clip (a short deterministic tone).
pub fn playground_audio_asset() -> inf_audio::AudioAsset {
    inf_audio::AudioAsset::from_encoded(tone_wav(4000, 8000), inf_audio::AudioFormat::Wav)
        .expect("tone wav decodes")
}

/// Write the committed physics-playground files from the generators (regeneration
/// path): the v6 `.inf_lvl` (+ sidecar), the two `.inf_audio` clips (+ inf_asset
/// sidecars with their stable GUIDs so the AudioSource `clip` refs resolve through
/// the AssetDb / cooked pack), + a README.
pub fn write_physics_playground() -> Result<(), String> {
    let dir = physics_playground_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    // Level (payload + sidecar) with a fixed level GUID.
    let doc = physics_playground_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("Playground.inf_lvl"),
        Some(PLAYGROUND_LEVEL_GUID),
    )?;

    // The two committed `.inf_audio` clips (same tone; distinct stable GUIDs).
    let audio = playground_audio_asset();
    for (file, guid) in [
        ("Spinner.inf_audio", PLAYGROUND_SPINNER_CLIP_GUID),
        ("Sensor.inf_audio", PLAYGROUND_SENSOR_CLIP_GUID),
    ] {
        write_anim_asset(&dir, file, guid, inf_asset::AssetKind::Audio, &audio)?;
    }

    std::fs::write(dir.join("README.md"), PLAYGROUND_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PLAYGROUND_README: &str = "# Physics Playground (Phase-12 gate scene)\n\n\
Generated by `inf_editor_core::samples::physics_playground_scene` — the P12.4 gate\n\
scene: a committed 3D physics playground composing every P12 feature at once — a\n\
settling **box stack**, a **motorized revolute spinner**, a **distance-rope\n\
pendulum**, a **prismatic slider**, a **CCD bullet** vs a thin wall, a\n\
**collision-layer ghost pair** (overlapping, non-interacting via filters), a\n\
**sensor plate**, and a small **ragdoll** (`inf_physics::ragdoll::build_ragdoll`\n\
output, its joint descs mapped to `Joint3D` components) — plus two spatial\n\
**AudioSource**s (one autoplay-looping on the spinner with occlusion, one on the\n\
sensor) and an **AudioListener** on a camera.\n\n\
- `Playground.inf_lvl` — the scene as schema-**v6** `.inf_lvl` bytes (joints +\n\
  audio persist).\n\
- `Spinner.inf_audio` / `Sensor.inf_audio` — the two clips the AudioSources\n\
  reference (a short deterministic tone; shipped via the cook's level→audio edge).\n\n\
Determinism is asserted by `runtime/inf-player/tests/physics_demo.rs`: 300 fixed\n\
steps twice yield a byte-identical pose trace (xxh3 over Guid-sorted transforms)\n\
AND an identical audio command stream; the ragdoll settles bounded, the CCD bullet\n\
is stopped, the ghost pair interpenetrates unimpeded, PIE == shipping.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// ── Phase 13 gate: the virtualized-geometry demo (P13.4) ─────────────────────
//
// The generator builds ONE dense displaced mesh asset (~33k triangles) and
// instances it `GRID × GRID` times across an XZ plane, so the **source** triangle
// count across instances exceeds 10M while the committed `.inf_lvl` stays tiny
// (every instance references the single `.inf_mesh` by GUID). The cook derives the
// `.inf_vmesh` meshlet DAG from that mesh (via the P13.4 `MeshRef.asset` →
// dependency-closure edge); the player renders it through the GPU meshlet path
// (vgeom on) or the classic discrete-LOD fallback (vgeom off / a lower render
// tier). The gate lives in `runtime/inf-player/tests/vgeom_gate.rs`.

/// Stable GUID of the vgeom-demo level.
pub const VGEOM_DEMO_LEVEL_GUID: Uuid = Uuid::from_u128(0x8406_0001_0000_0000_0000_0000_0000_0001);
/// Stable GUID of the shared dense `.inf_mesh` asset every instance references.
pub const VGEOM_DEMO_MESH_GUID: Uuid = Uuid::from_u128(0x8406_0002_0000_0000_0000_0000_0000_0002);
/// Stable GUID of the directional sun.
pub const VGEOM_DEMO_SUN_GUID: Uuid = Uuid::from_u128(0x8406_0003_0000_0000_0000_0000_0000_0003);
/// Base GUID for the grid instance entities (add the flat index).
const VGEOM_DEMO_INSTANCE_BASE: u128 = 0x8406_0100_0000_0000_0000_0000_0000_0000;

/// Grid subdivisions of the dense mesh — `2·N²` triangles (128 → 32 768 tris,
/// well over the cook's 2048 `min_triangles` vmesh threshold).
pub const VGEOM_DEMO_MESH_N: usize = 128;
/// Instance grid side: `GRID × GRID` placed copies of the dense mesh.
pub const VGEOM_DEMO_GRID: usize = 18;

/// Total **source** triangles across every instance (`2·N² · GRID²`). Exceeds the
/// phase gate's 10M requirement (128/18 → 10 616 832).
pub const fn vgeom_demo_source_triangles() -> u64 {
    (2 * VGEOM_DEMO_MESH_N * VGEOM_DEMO_MESH_N * VGEOM_DEMO_GRID * VGEOM_DEMO_GRID) as u64
}

/// A byte-portable sine: std `f32::sin` is NOT bit-identical across libms (MSVC
/// vs glibc diverged on the Ubuntu CI runner — the generator-lock caught it), so
/// the displacement uses this pure-arithmetic minimax polynomial instead. IEEE
/// f32 add/mul/floor are exactly specified, so the committed mesh bytes are
/// identical on every platform.
fn psin(x: f32) -> f32 {
    use std::f32::consts::TAU;
    // Range-reduce to [-π, π] (floor is exact; inputs here are small, no
    // catastrophic cancellation at the scales the generator uses).
    let x = x - (x / TAU + 0.5).floor() * TAU;
    // Odd 7th-order minimax on [-π, π] (~1e-4 abs error — far below visual or
    // meshlet-build significance, and perfectly reproducible).
    let x2 = x * x;
    x * (0.987_862 + x2 * (-0.155_271 + x2 * (0.005_641_12 - x2 * 0.000_060_461_2)))
}

/// Byte-portable cosine via [`psin`].
fn pcos(x: f32) -> f32 {
    psin(x + std::f32::consts::FRAC_PI_2)
}

/// One dense displaced-grid [`inf_mesh::MeshAsset`] (`2·N²` triangles) — the shared
/// asset every instance references. A deterministic function of `N` built on
/// byte-portable arithmetic ([`psin`]/[`pcos`]), so the committed `.inf_mesh` is
/// reproducible on every platform.
pub fn vgeom_demo_mesh() -> inf_mesh::MeshAsset {
    let n = VGEOM_DEMO_MESH_N;
    let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            let x = (u - 0.5) * 2.0;
            let z = (v - 0.5) * 2.0;
            let y = 0.3 * psin(x * 3.0) * pcos(z * 3.0);
            let nrm = glam::Vec3::new(
                -0.9 * pcos(x * 3.0) * pcos(z * 3.0),
                1.0,
                0.9 * psin(x * 3.0) * psin(z * 3.0),
            )
            .normalize();
            vertices.push(inf_mesh::MeshVertex {
                position: [x, y, z],
                normal: nrm.to_array(),
                uv: [u, v],
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
        }
    }
    let stride = (n + 1) as u32;
    let mut indices = Vec::with_capacity(n * n * 6);
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * stride + i;
            indices.extend_from_slice(&[a, a + stride, a + 1, a + 1, a + stride, a + stride + 1]);
        }
    }
    let submesh = inf_mesh::SubMesh {
        name: "dense".into(),
        vertices,
        indices,
        material_slot: Some(0),
        skin: Vec::new(),
    };
    inf_mesh::MeshAsset::new(vec![submesh], vec!["Default".into()])
}

/// The vgeom-demo scene: `GRID × GRID` instances of the dense mesh asset spread
/// across an XZ plane, each an entity with a [`MeshRef`] whose `asset` points at
/// [`VGEOM_DEMO_MESH_GUID`] (plus a placeholder `Cube` primitive for the editor
/// viewport, which cannot render asset geometry yet), + a sun. The `.inf_lvl`
/// stays small (one asset, many light instance records).
pub fn vgeom_demo_scene() -> SceneDoc {
    use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive, Transform};
    use inf_ecs::math::{Color, Vec3d};

    let mut doc = SceneDoc::new();
    doc.set_title("Vgeom Demo");

    // Sun.
    doc.create_with_guid(VGEOM_DEMO_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        VGEOM_DEMO_SUN_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::new(-50.0, -30.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        VGEOM_DEMO_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 3.0,
            ..Default::default()
        }
    );

    // The instance grid. `spacing` tiles the 2-unit-wide meshes edge-to-edge with a
    // small gap, spreading them over the XZ plane so a ground-level camera sees only
    // a small fraction (frustum + LOD) — the cull-ratio gate.
    let grid = VGEOM_DEMO_GRID;
    let spacing = 2.4f64;
    let offset = (grid as f64 - 1.0) * 0.5 * spacing;
    for j in 0..grid {
        for i in 0..grid {
            let idx = (j * grid + i) as u128;
            let guid = Uuid::from_u128(VGEOM_DEMO_INSTANCE_BASE + idx);
            let name = format!("Tile {i}x{j}");
            doc.create_with_guid(guid, SpawnKind::Empty, &name, None);
            insert!(
                doc,
                guid,
                Transform {
                    translation: Vec3d::new(
                        i as f64 * spacing - offset,
                        0.0,
                        j as f64 * spacing - offset,
                    ),
                    rotation: Vec3d::ZERO,
                    scale: Vec3d::ONE,
                }
            );
            insert!(
                doc,
                guid,
                MeshRef {
                    primitive: Primitive::Cube,
                    asset: Some(VGEOM_DEMO_MESH_GUID),
                }
            );
            // A subtle per-tile tint so the instances read as distinct content.
            let t = idx as f32 / (grid * grid) as f32;
            insert!(
                doc,
                guid,
                Material {
                    base_color: Color::new(0.45 + 0.3 * t, 0.5, 0.65 - 0.3 * t, 1.0),
                    metallic: 0.0,
                    roughness: 0.7,
                    emissive: Color::new(0.0, 0.0, 0.0, 1.0),
                    ..Default::default()
                }
            );
        }
    }

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/vgeom-demo/` directory.
pub fn vgeom_demo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/vgeom-demo")
}

/// Write the committed vgeom-demo files from the generators (regeneration path):
/// the `.inf_lvl` (+ sidecar), the dense `.inf_mesh` (+ its inf_asset sidecar so
/// the `MeshRef.asset` refs resolve through the AssetDb / cooked pack), + README.
pub fn write_vgeom_demo() -> Result<(), String> {
    let dir = vgeom_demo_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    let doc = vgeom_demo_scene();
    crate::scene::serialize::save(
        &doc,
        &dir.join("VgeomDemo.inf_lvl"),
        Some(VGEOM_DEMO_LEVEL_GUID),
    )?;

    // The dense `.inf_mesh` payload + its inf_asset sidecar (STABLE mesh GUID the
    // level's MeshRef.asset points at; the cook derives its `.inf_vmesh` beside it).
    let mesh_bytes =
        inf_asset::encode(&vgeom_demo_mesh()).map_err(|e| format!("encode mesh: {e}"))?;
    let mesh_path = dir.join("Dense.inf_mesh");
    std::fs::write(&mesh_path, &mesh_bytes).map_err(|e| format!("write mesh: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(VGEOM_DEMO_MESH_GUID),
        inf_asset::AssetKind::Mesh,
        inf_asset::ContentHash::of(&mesh_bytes),
    )
    .save(&mesh_path)
    .map_err(|e| format!("write mesh sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), VGEOM_DEMO_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const VGEOM_DEMO_README: &str = "# Vgeom Demo (Phase-13 gate scene)\n\n\
Generated by `inf_editor_core::samples::vgeom_demo_scene` — the P13.4 gate scene: a\n\
grid of `GRID × GRID` instances of ONE dense displaced mesh asset, so the SOURCE\n\
triangle count across instances exceeds 10M while the `.inf_lvl` stays tiny (every\n\
instance references the single `Dense.inf_mesh` by GUID).\n\n\
- `VgeomDemo.inf_lvl` — the scene (schema v7): instance transforms + `MeshRef.asset`\n\
  refs + materials + a sun.\n\
- `Dense.inf_mesh` — the shared ~33k-triangle displaced grid. The cook derives its\n\
  `.inf_vmesh` meshlet DAG (via the `MeshRef.asset` dependency-closure edge) and\n\
  ships both in the pack.\n\n\
The gate (`runtime/inf-player/tests/vgeom_gate.rs`): byte-identical save/reload; a\n\
cooked load where the total source triangles ≥ 10M and the GPU meshlet cull leaves\n\
only a small fraction of meshlets visible from a ground-level camera (deterministic\n\
across runs); the SAME pack with vgeom OFF renders through the classic discrete-LOD\n\
fallback (a far camera picks a coarser level than a near one); and the auto-tier\n\
disables vgeom on the Low tier. GPU parts skip cleanly with no adapter.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// -- Streamed-terrain gate scene (P16.3b2) -----------------------------------
//
// The camera-driven-streaming gate scene: a terrain that lives ENTIRELY in a
// `.inf_terrain` asset (the level carries an empty working set plus the asset
// ref), a "Walker" entity the sim scripts across it, a sun and a camera.
//
// The `.inf_lvl` is committed; the `.inf_terrain` is **generated into the
// fixture's content directory** by [`write_streamed_terrain_asset`] rather than
// committed, because it is ~100 KB of derived bytes a pure generator reproduces
// exactly (the same reasoning that keeps the vgeom demo's mesh small).

pub const STREAMED_TERRAIN_LEVEL_GUID: Uuid = Uuid::from_u128(0x8416_0000);
pub const STREAMED_TERRAIN_TERRAIN_GUID: Uuid = Uuid::from_u128(0x8416_0001);
pub const STREAMED_TERRAIN_WALKER_GUID: Uuid = Uuid::from_u128(0x8416_0002);
pub const STREAMED_TERRAIN_SUN_GUID: Uuid = Uuid::from_u128(0x8416_0003);
pub const STREAMED_TERRAIN_CAMERA_GUID: Uuid = Uuid::from_u128(0x8416_0004);
/// The **asset** GUID of the generated `World.inf_terrain`, stamped into its
/// inf_asset sidecar. Stable so the level's `Terrain.asset` ref resolves through
/// the AssetDb / the cooked pack, and the cook's level -> terrain edge ships it.
pub const STREAMED_TERRAIN_ASSET_GUID: Uuid = Uuid::from_u128(0x8416_00AA);

/// Samples per tile side. Small enough that 256 pages stay ~100 KB, large enough
/// that a page is a real page.
pub const STREAMED_TERRAIN_RESOLUTION: u32 = 9;
/// Level-0 metres per sample => a 16 m tile span.
pub const STREAMED_TERRAIN_MPS: f64 = 2.0;
/// Level-0 tiles per side: a 16 x 16 grid => **256 m of world**, far wider than
/// any single render-wants radius (`RENDER_LOD0_RADIUS_TILES` x 16 m = 40 m), so
/// the camera genuinely pages tiles in and out as it moves. 16 is a power of two,
/// so the pyramid closes cleanly: 256 -> 64 -> 16 -> 4 pages, i.e. **three coarse
/// levels** (the gate needs at least two).
pub const STREAMED_TERRAIN_TILES: i32 = 16;

/// World edge length of the generated terrain (metres).
pub fn streamed_terrain_world_size() -> f64 {
    (STREAMED_TERRAIN_RESOLUTION as f64 - 1.0)
        * STREAMED_TERRAIN_MPS
        * STREAMED_TERRAIN_TILES as f64
}

/// The generated terrain's analytic height at world `(x, z)`.
///
/// Built from [`inf_math::psin64`] / [`inf_math::pcos64`], never `std` trig: the
/// P14 law -- `std` transcendentals are not bit-portable, and this function's
/// output ends up in bytes a cook on one OS and a run on another must agree about.
pub fn streamed_terrain_height(x: f64, z: f64) -> f64 {
    8.0 * inf_math::psin64(x * 0.04) * inf_math::pcos64(z * 0.035)
        + 2.0 * inf_math::psin64((x + z) * 0.11)
}

/// The authored level-0 heightfield the `.inf_terrain` is built from.
pub fn streamed_terrain_data() -> inf_terrain::TerrainData {
    let mut t = inf_terrain::TerrainData::new(STREAMED_TERRAIN_RESOLUTION, STREAMED_TERRAIN_MPS);
    for tz in 0..STREAMED_TERRAIN_TILES {
        for tx in 0..STREAMED_TERRAIN_TILES {
            t.author_tile((tx, tz), streamed_terrain_height);
        }
    }
    t
}

/// The `.inf_terrain` payload: the level-0 grid plus its full LOD pyramid.
///
/// A pure function of the generators, so two builds are byte-identical (the
/// `.inf_terrain` layout is deterministic by construction -- see
/// `inf_terrain::asset`).
pub fn streamed_terrain_asset() -> inf_terrain::TerrainAsset {
    let data = streamed_terrain_data();
    let pyramid = inf_terrain::build_pyramid(&data, inf_terrain::PyramidOptions::default());
    inf_terrain::build_terrain_asset(&data, &pyramid).expect("streamed-terrain asset builds")
}

/// Write `World.inf_terrain` (+ its inf_asset sidecar with the stable asset GUID)
/// into `dir` -- the gate's **fixture setup**, and the P16.4 import wizard's model.
///
/// Goes through [`inf_terrain::write_terrain_asset`], the one sanctioned writer:
/// the bytes on disk are the raw payload image, never a framed `inf_asset::encode`
/// (which would knock every tile off its 16-byte boundary). The sidecar hashes
/// exactly the bytes written, so the cook packs them verbatim.
pub fn write_streamed_terrain_asset(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let asset = streamed_terrain_asset();
    let path = dir.join("World.inf_terrain");
    let bytes = inf_terrain::write_terrain_asset(&path, &asset)
        .map_err(|e| format!("write terrain asset: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(STREAMED_TERRAIN_ASSET_GUID),
        inf_asset::AssetKind::Terrain,
        inf_asset::ContentHash::of(bytes),
    )
    .save(&path)
    .map_err(|e| format!("write terrain sidecar: {e}"))
}

/// Build the streamed-terrain [`SceneDoc`].
///
/// The `Terrain` component carries **no tiles**: its `data` is an empty working
/// set configured on the asset's grid, and `asset` points at the `.inf_terrain`.
/// That is the whole point -- the level stays kilobytes while the world is
/// 256 m x 256 m of paged heightfield.
pub fn streamed_terrain_scene() -> SceneDoc {
    use inf_ecs::components::{Camera, Light, LightKind, Terrain};

    let mut doc = SceneDoc::new();
    doc.set_title("Streamed Terrain");
    let world_size = streamed_terrain_world_size();

    // -- The streamed terrain, anchored at the world origin (so world XZ ==
    //    terrain-local XZ and the height probe is the bare generator). --
    doc.create_with_guid(
        STREAMED_TERRAIN_TERRAIN_GUID,
        SpawnKind::Empty,
        "Terrain",
        None,
    );
    insert!(
        doc,
        STREAMED_TERRAIN_TERRAIN_GUID,
        Transform::from_translation(DVec3::ZERO)
    );
    {
        let mut terrain = Terrain::configured(STREAMED_TERRAIN_RESOLUTION, STREAMED_TERRAIN_MPS);
        terrain.asset = Some(STREAMED_TERRAIN_ASSET_GUID);
        terrain.macro_variation = 0.2;
        debug_assert!(terrain.data.is_empty(), "a streamed terrain ships no tiles");
        insert!(doc, STREAMED_TERRAIN_TERRAIN_GUID, terrain);
    }

    // -- The walker: the entity the SIM scripts across the terrain. Its position
    //    is what `sim_wants` derives level-0 residency from, and the gate probes
    //    `terrain.height_at` under it. --
    doc.create_with_guid(
        STREAMED_TERRAIN_WALKER_GUID,
        SpawnKind::Empty,
        "Walker",
        None,
    );
    insert!(
        doc,
        STREAMED_TERRAIN_WALKER_GUID,
        Transform::from_translation(streamed_terrain_walk_point(0))
    );
    // A character controller is what makes the Walker a **terrain observer**
    // (`inf_player::terrain_stream::observes_terrain`): sim residency follows the
    // things that walk on the ground, not everything with a transform. With no
    // `RigidBody3D`/`Collider3D` beside it the physics bridge skips the entity
    // entirely, so this marks intent without simulating anything.
    insert!(
        doc,
        STREAMED_TERRAIN_WALKER_GUID,
        inf_ecs::components::CharacterController3D::default()
    );

    // -- A directional sun. --
    doc.create_with_guid(STREAMED_TERRAIN_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        STREAMED_TERRAIN_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 80.0, 0.0),
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        STREAMED_TERRAIN_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );

    // -- A camera over the middle of the world. --
    doc.create_with_guid(
        STREAMED_TERRAIN_CAMERA_GUID,
        SpawnKind::Empty,
        "Camera",
        None,
    );
    insert!(
        doc,
        STREAMED_TERRAIN_CAMERA_GUID,
        Transform::from_translation(DVec3::new(world_size * 0.5, 60.0, world_size * 0.5))
    );
    insert!(doc, STREAMED_TERRAIN_CAMERA_GUID, Camera::default());

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The scripted **sim** walk: step `i`'s world position for the Walker entity.
///
/// A diagonal crossing of the whole 256 m world, so the sim's level-0
/// neighbourhood really does page in and out. Deterministic and camera-free --
/// this is the trace the "sim determinism vs camera" gate compares.
pub fn streamed_terrain_walk_point(step: usize) -> DVec3 {
    let world_size = streamed_terrain_world_size();
    let t = step as f64 * 1.5;
    let x = (t % world_size).clamp(0.0, world_size);
    let z = ((t * 0.7) % world_size).clamp(0.0, world_size);
    DVec3::new(x, streamed_terrain_height(x, z), z)
}

/// Scripted **camera** path A: a straight sweep along +X at mid-Z.
pub fn streamed_terrain_camera_a(step: usize) -> DVec3 {
    let world_size = streamed_terrain_world_size();
    let x = (step as f64 * 3.0) % world_size;
    DVec3::new(x, 40.0, world_size * 0.5)
}

/// Scripted **camera** path B: an orbit around the world centre -- deliberately
/// nothing like [`streamed_terrain_camera_a`], so "the sim ignores the camera" is
/// tested against a genuinely different residency history, not a variation of the
/// same one.
pub fn streamed_terrain_camera_b(step: usize) -> DVec3 {
    let world_size = streamed_terrain_world_size();
    let a = step as f64 * 0.21;
    let r = world_size * 0.35;
    DVec3::new(
        world_size * 0.5 + r * inf_math::pcos64(a),
        40.0,
        world_size * 0.5 + r * inf_math::psin64(a),
    )
}

/// The repo-root `samples/streamed-terrain/` directory.
pub fn streamed_terrain_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/streamed-terrain")
}

/// Write the committed streamed-terrain files (regeneration path): the `.inf_lvl`
/// (+ sidecar) and the README. The `.inf_terrain` itself is **not** committed --
/// see [`write_streamed_terrain_asset`].
pub fn write_streamed_terrain() -> Result<(), String> {
    let dir = streamed_terrain_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    crate::scene::serialize::save(
        &streamed_terrain_scene(),
        &dir.join("StreamedTerrain.inf_lvl"),
        Some(STREAMED_TERRAIN_LEVEL_GUID),
    )?;
    std::fs::write(dir.join("README.md"), STREAMED_TERRAIN_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const STREAMED_TERRAIN_README: &str = "# Streamed Terrain (P16.3 gate scene)\n\n\
Generated by `inf_editor_core::samples::streamed_terrain_scene` -- the P16.3b2 gate\n\
scene for **camera-driven terrain streaming**.\n\n\
- `StreamedTerrain.inf_lvl` -- the scene (schema v9). Its `Terrain` carries an EMPTY\n\
  working set plus `asset = World.inf_terrain`, so the level stays kilobytes while\n\
  the world is 256 m x 256 m of paged heightfield. A `Walker` entity is what the\n\
  gate scripts across the terrain.\n\
- `World.inf_terrain` -- NOT committed. It is ~100 KB of derived bytes that\n\
  `samples::write_streamed_terrain_asset` reproduces exactly, so the gate generates\n\
  it into the fixture's Content directory (16x16 level-0 pages of 9^2 samples at\n\
  2 m, plus three coarse pyramid levels: 256 -> 64 -> 16 -> 4).\n\n\
## The doctrine this scene exists to pin\n\n\
The fixed-step sim's results must never depend on camera-driven residency. Sim\n\
wants (level-0 pages around the sim's own entities) load synchronously at the\n\
fixed-step boundary into the ECS `Terrain`'s data; render wants (the camera's\n\
quadtree cut) load into a separate working set inside the streamer that no entity\n\
references.\n\n\
The gate (`runtime/inf-player/tests/streamed_terrain.rs`):\n\n\
1. the cook ships the `.inf_terrain` through the level->terrain edge, UNCOMPRESSED\n\
   (streaming-class), so tiles page zero-copy out of the mapping;\n\
2. a headless run over a scripted camera path reproduces a byte-identical\n\
   resident-set trace AND rendered-frame (projected terrain) trace across two runs;\n\
3. the SAME scripted sim under two COMPLETELY different camera paths produces a\n\
   byte-identical sim trace -- the doctrine, as an executable assertion;\n\
4. PIE == shipping: the cooked-pack path and the editor-doc path stream the same\n\
   terrain to the same sim trace and the same resident set.\n\n\
Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.\n";

// -- Partitioned-world gate scene (P16.5) ------------------------------------
//
// The world-partition gate scene: a 4 x 4 grid of 128 m cells, one authored prop
// per cell, a persistent manager + sun, and a "Walker" carrying a
// `StreamingSource` that the gate scripts along the +X row. Everything is
// committed as ONE `.inf_lvl` (the editor stays single-document); the `.inf_part`
// is DERIVED by the cook, exactly as `.inf_vmesh` is.

pub const PARTITIONED_LEVEL_GUID: Uuid = Uuid::from_u128(0x8416_0500);
pub const PARTITIONED_WALKER_GUID: Uuid = Uuid::from_u128(0x8416_0501);
pub const PARTITIONED_MANAGER_GUID: Uuid = Uuid::from_u128(0x8416_0502);
pub const PARTITIONED_SUN_GUID: Uuid = Uuid::from_u128(0x8416_0503);
/// A prop parented under the cell-(3,3) prop — proves a hierarchy streams as one
/// unit even though its own transform sits nowhere near its parent's cell.
pub const PARTITIONED_CHILD_GUID: Uuid = Uuid::from_u128(0x8416_0504);

/// Cell edge length (metres). Small enough that a scripted walk crosses several
/// cells in a short run; large enough to be a plausible authoring unit.
pub const PARTITIONED_CELL_SIZE_M: f64 = 128.0;
/// Cells per side: 4 x 4 = 16, comfortably over the gate's "at least 3 x 3".
pub const PARTITIONED_GRID: i32 = 4;
/// Level activation radius (metres) — under one cell, so exactly the cells the
/// walker is standing on/next to activate and the trace has real churn.
pub const PARTITIONED_ACTIVATION_RADIUS_M: f64 = 32.0;
/// Level prefetch margin (metres). Cells within `activation + margin` may be
/// decoded ahead of need; it can never change WHICH cells activate.
pub const PARTITIONED_PREFETCH_MARGIN_M: f64 = 192.0;

/// World edge length of the partitioned grid (metres).
pub fn partitioned_world_size() -> f64 {
    PARTITIONED_CELL_SIZE_M * PARTITIONED_GRID as f64
}

/// The stable GUID of the prop authored in cell `(cx, cz)`.
pub fn partitioned_prop_guid(cx: i32, cz: i32) -> Uuid {
    Uuid::from_u128(0x8416_0600 + (cz * PARTITIONED_GRID + cx) as u128)
}

/// The world position of the prop authored in cell `(cx, cz)` — the cell's
/// centre, so it is unambiguously inside exactly one cell.
pub fn partitioned_prop_position(cx: i32, cz: i32) -> DVec3 {
    DVec3::new(
        (cx as f64 + 0.5) * PARTITIONED_CELL_SIZE_M,
        0.0,
        (cz as f64 + 0.5) * PARTITIONED_CELL_SIZE_M,
    )
}

/// The scripted **sim** walk: step `i`'s world position for the Walker.
///
/// Straight along +X through the centres of row `z = 0`, one third of a cell per
/// step, so the walk crosses every cell in the row and the activation trace has
/// real transitions. Deterministic and camera-free — this is the trace the gates
/// compare, and (by the doctrine) the only thing residency may depend on.
pub fn partitioned_walk_point(step: usize) -> DVec3 {
    let span = partitioned_world_size();
    let x = (step as f64 * (PARTITIONED_CELL_SIZE_M / 3.0)) % span;
    DVec3::new(x, 0.0, PARTITIONED_CELL_SIZE_M * 0.5)
}

/// Build the partitioned-world [`SceneDoc`].
///
/// The document is a plain, single, unpartitioned-looking level — because that is
/// what the editor authors. What makes it partitioned is one settings block; the
/// cook is what splits it, and the player is what streams it.
pub fn partitioned_world_scene() -> SceneDoc {
    use crate::scene::serialize::{LevelSettings, PartitionSettings};
    use inf_ecs::components::{AlwaysLoaded, Light, LightKind, StreamingSource};

    let mut doc = SceneDoc::new();
    doc.set_title("Partitioned World");
    doc.set_settings(LevelSettings {
        partition: PartitionSettings {
            enabled: true,
            cell_size_m: PARTITIONED_CELL_SIZE_M,
            activation_radius_m: PARTITIONED_ACTIVATION_RADIUS_M,
            prefetch_margin_m: PARTITIONED_PREFETCH_MARGIN_M,
        },
        ..LevelSettings::default()
    });

    // -- The persistent cell --
    //
    // A manager with no spatial component at all (the `Unplaced` rule), and a sun
    // explicitly marked `AlwaysLoaded` (a Light DOES occupy space, so without the
    // marker it would stream out and the world would go dark). Those two entities
    // are the whole "what is a persistent cell for" story, in the scene.
    doc.create_with_guid(PARTITIONED_MANAGER_GUID, SpawnKind::Empty, "GameMode", None);
    insert!(
        doc,
        PARTITIONED_MANAGER_GUID,
        Transform::from_translation(DVec3::ZERO)
    );

    doc.create_with_guid(PARTITIONED_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        PARTITIONED_SUN_GUID,
        Transform {
            translation: inf_ecs::math::Vec3d::new(0.0, 80.0, 0.0),
            rotation: inf_ecs::math::Vec3d::new(-50.0, -30.0, 0.0),
            scale: inf_ecs::math::Vec3d::new(1.0, 1.0, 1.0),
        }
    );
    insert!(
        doc,
        PARTITIONED_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::new(1.0, 0.97, 0.9, 1.0),
            intensity: 2.5,
            ..Default::default()
        }
    );
    insert!(doc, PARTITIONED_SUN_GUID, AlwaysLoaded);

    // -- The streaming source: the entity residency is derived FROM. --
    //
    // It carries a `StreamingSource` (which is also what makes it persistent —
    // a source that could stream itself out is a bootstrap paradox), and
    // `radius_m: 0` so the LEVEL's activation radius is what governs; the gate
    // then has one knob to reason about.
    doc.create_with_guid(PARTITIONED_WALKER_GUID, SpawnKind::Empty, "Walker", None);
    insert!(
        doc,
        PARTITIONED_WALKER_GUID,
        Transform::from_translation(partitioned_walk_point(0))
    );
    insert!(
        doc,
        PARTITIONED_WALKER_GUID,
        StreamingSource { radius_m: 0.0 }
    );

    // -- One authored prop per cell (a cube at the cell centre). --
    for cz in 0..PARTITIONED_GRID {
        for cx in 0..PARTITIONED_GRID {
            let guid = partitioned_prop_guid(cx, cz);
            doc.create_with_guid(guid, SpawnKind::Cube, &format!("Prop {cx},{cz}"), None);
            insert!(
                doc,
                guid,
                Transform::from_translation(partitioned_prop_position(cx, cz))
            );
        }
    }

    // -- A child of the far-corner prop, authored a whole world away. --
    //
    // Its own transform would bin it into cell (0,0); the partitioner assigns it
    // its ROOT's cell instead, so the pair never splits. The gate asserts it
    // appears and disappears together with its parent.
    doc.create_with_guid(
        PARTITIONED_CHILD_GUID,
        SpawnKind::Cube,
        "Far Child",
        Some(partitioned_prop_guid(
            PARTITIONED_GRID - 1,
            PARTITIONED_GRID - 1,
        )),
    );
    insert!(
        doc,
        PARTITIONED_CHILD_GUID,
        Transform::from_translation(DVec3::new(-2000.0, 0.0, -2000.0))
    );

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// The repo-root `samples/partitioned-world/` directory.
pub fn partitioned_world_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/partitioned-world")
}

/// Write the committed partitioned-world files (regeneration path).
pub fn write_partitioned_world() -> Result<(), String> {
    let dir = partitioned_world_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;
    crate::scene::serialize::save(
        &partitioned_world_scene(),
        &dir.join("PartitionedWorld.inf_lvl"),
        Some(PARTITIONED_LEVEL_GUID),
    )?;
    std::fs::write(dir.join("README.md"), PARTITIONED_WORLD_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

const PARTITIONED_WORLD_README: &str = "# Partitioned World (P16.5 gate scene)\n\n\
Generated by `inf_editor_core::samples::partitioned_world_scene` -- the P16.5 gate\n\
scene for **world partition / level streaming**.\n\n\
- `PartitionedWorld.inf_lvl` -- the scene (schema v10). ONE document, as the editor\n\
  always authors: a 4x4 grid of cubes at the centres of 128 m cells, a `GameMode`\n\
  with no spatial component, a sun marked `AlwaysLoaded`, a `Walker` carrying a\n\
  `StreamingSource`, and a child of the far-corner prop authored a whole world\n\
  away from it. What makes it partitioned is one settings block\n\
  (`partition.enabled`), nothing else.\n\
- `PartitionedWorld.inf_part` -- NOT committed. It is DERIVED by the cook (like\n\
  `.inf_vmesh`): the cook bins the entities into a persistent cell + 16 grid cells\n\
  and writes them to a pack entry whose GUID is a pure function of the level's.\n\n\
## The doctrine this scene exists to pin\n\n\
Cell streaming decides which entities EXIST, so residency must be a function of sim\n\
state alone. Wants come from `StreamingSource` entities read at the fixed-step\n\
boundary -- never a camera -- and activation/deactivation happen only at that sync\n\
point, in ascending cell order. Loading may run ahead (the prefetch margin); a cell\n\
that reaches its activation step unloaded blocks the step. So the margin buys\n\
latency and can never move a result.\n\n\
## The v1 boundaries, stated rather than discovered\n\n\
- The PERSISTENT cell is the world at step 0: the level builder spawns it BEFORE\n\
  blueprint actors bind, so a persistent entity's `ActorClass` ticks normally. An\n\
  entity that streams IN does NOT gain a ticking blueprint in v1 (the actor map is\n\
  fixed at `RuntimeSim` construction). Mark such an entity `AlwaysLoaded`.\n\
- Runtime-spawned entities are never despawned by streaming; a statically-placed\n\
  one is evicted with its BIRTH cell, wherever a script has since moved it.\n\
- A cook-time reference from one cell to another is a cook WARNING, never a\n\
  promotion: residency must not depend on the reference graph.\n\
- The editor's in-process Simulate runs the whole document unpartitioned (single\n\
  document in v1); PIE and a shipped build both stream, and those two are what the\n\
  parity gate compares.\n\n\
The gate (`runtime/inf-player/tests/partitioned_world.rs`):\n\n\
1. the cook emits ONE `.inf_part` entry, UNCOMPRESSED, deterministic across\n\
   rebuilds, and the cooked `.inf_lvl` carries no entities;\n\
2. two headless runs of the scripted walk produce an identical activation trace;\n\
3. the SAME walk under two DIFFERENT prefetch margins produces a byte-identical\n\
   sim trace -- the doctrine, as an executable assertion;\n\
4. PIE == shipping: the cooked-pack path and the editor-document path stream the\n\
   same cells to the same sim trace;\n\
5. non-partitioned regression: the platformer sample's pack is byte-identical\n\
   whether or not partitioning exists;\n\
6. an entity in a far cell does not exist until the walker approaches -- then it\n\
   exists with its authored transform.\n\n\
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
    fn firstperson_scene_saves_and_reloads_byte_identical() {
        let doc = firstperson_scene();
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
            "first-person template scene must round-trip byte-identically"
        );
        // The player, camera, ground, and sun all survive (4 entities).
        assert_eq!(
            crate::scene::serialize::to_scene_file(&doc2).entities.len(),
            4
        );
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

    #[test]
    fn vgeom_demo_saves_and_reloads_byte_identical() {
        let doc = vgeom_demo_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0],
            crate::scene::serialize::SCHEMA_VERSION as u8,
            "vgeom-demo writes at the current schema"
        );

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
            "vgeom-demo scene must round-trip byte-identically"
        );

        // Every instance carries a MeshRef.asset pointing at the shared mesh GUID.
        let file = crate::scene::serialize::to_scene_file(&doc2);
        let inst: Vec<_> = file
            .entities
            .iter()
            .filter(|r| r.mesh.as_ref().and_then(|m| m.asset).is_some())
            .collect();
        assert_eq!(inst.len(), VGEOM_DEMO_GRID * VGEOM_DEMO_GRID);
        assert!(inst
            .iter()
            .all(|r| r.mesh.as_ref().unwrap().asset == Some(VGEOM_DEMO_MESH_GUID)));
    }

    #[test]
    fn vgeom_demo_exceeds_10m_source_triangles() {
        // The dense mesh's triangle count times the instance grid is the source
        // triangle budget the phase gate demands (≥ 10M).
        let mesh = vgeom_demo_mesh();
        let per = mesh.triangle_count() as u64;
        assert_eq!(per, (2 * VGEOM_DEMO_MESH_N * VGEOM_DEMO_MESH_N) as u64);
        let total = per * (VGEOM_DEMO_GRID * VGEOM_DEMO_GRID) as u64;
        assert_eq!(total, vgeom_demo_source_triangles());
        assert!(
            total >= 10_000_000,
            "gate needs 10M+ source triangles, got {total}"
        );
        // Above the cook's default vmesh derivation threshold.
        assert!(per >= 2048);
    }

    /// Regenerate the committed files under `INF_BLESS_SAMPLES=1`; otherwise
    /// assert the committed bytes still match the generators (fixture lock).
    #[test]
    fn committed_sample_matches_generators() {
        if std::env::var("INF_BLESS_SAMPLES").is_ok() {
            write_sample().expect("regenerate sample");
            write_hybrid_template().expect("regenerate hybrid template");
            write_firstperson_template().expect("regenerate first-person template");
            write_terrain_demo().expect("regenerate terrain demo");
            write_character_demo().expect("regenerate character demo");
            write_physics_playground().expect("regenerate physics playground");
            write_vgeom_demo().expect("regenerate vgeom demo");
            write_streamed_terrain().expect("regenerate streamed terrain");
            write_partitioned_world().expect("regenerate partitioned world");
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

        // First-person template lock: the committed `.inf_lvl` still matches the
        // generator (skips gracefully before the first bless).
        let fpdir = firstperson_template_dir();
        let fplvl = fpdir.join("FirstPerson.inf_lvl");
        if fplvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&firstperson_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&fplvl).unwrap(),
                want_lvl,
                "committed first-person .inf_lvl drifted from the generator"
            );
        }

        // Terrain-demo lock: the committed `.inf_lvl` + `.inf_pcg` still match the
        // generators (skips gracefully before the first bless).
        let tdir = terrain_demo_dir();
        let tlvl = tdir.join("TerrainDemo.inf_lvl");
        let tpcg = tdir.join("Scatter.inf_pcg");
        if tlvl.exists() && tpcg.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&terrain_demo_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&tlvl).unwrap(),
                want_lvl,
                "committed terrain-demo .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(&tpcg).unwrap(),
                terrain_demo_pcg_payload().encode().unwrap(),
                "committed terrain-demo .inf_pcg drifted from the generator"
            );
        }

        // Character-demo lock: the committed `.inf_lvl` + anim assets still match
        // the generators (skips gracefully before the first bless).
        let cdir = character_demo_dir();
        let clvl = cdir.join("Character.inf_lvl");
        if clvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&character_demo_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&clvl).unwrap(),
                want_lvl,
                "committed character-demo .inf_lvl drifted from the generator"
            );
            assert_eq!(
                std::fs::read(cdir.join("Locomotion.inf_sm")).unwrap(),
                inf_asset::encode(&inf_anim::StateMachineAsset::new(
                    character_demo_state_machine(),
                    Some(*CHARACTER_DEMO_SKELETON_GUID.as_bytes()),
                ))
                .unwrap(),
                "committed character-demo .inf_sm drifted from the generator"
            );
        }

        // Physics-playground lock: the committed v6 `.inf_lvl` + the two
        // `.inf_audio` clips still match the generators (skips before first bless).
        let pdir = physics_playground_dir();
        let plvl = pdir.join("Playground.inf_lvl");
        if plvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&physics_playground_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&plvl).unwrap(),
                want_lvl,
                "committed physics-playground .inf_lvl drifted from the generator"
            );
            let want_audio = inf_asset::encode(&playground_audio_asset()).unwrap();
            assert_eq!(
                std::fs::read(pdir.join("Spinner.inf_audio")).unwrap(),
                want_audio,
                "committed Spinner.inf_audio drifted from the generator"
            );
            assert_eq!(
                std::fs::read(pdir.join("Sensor.inf_audio")).unwrap(),
                want_audio,
                "committed Sensor.inf_audio drifted from the generator"
            );
        }

        // Vgeom-demo lock: the committed `.inf_lvl` + dense `.inf_mesh` still match
        // the generators (skips gracefully before the first bless).
        let vdir = vgeom_demo_dir();
        let vlvl = vdir.join("VgeomDemo.inf_lvl");
        let vmesh = vdir.join("Dense.inf_mesh");
        if vlvl.exists() && vmesh.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&vgeom_demo_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&vlvl).unwrap(),
                want_lvl,
                "committed vgeom-demo .inf_lvl drifted from the generator"
            );
            let want_mesh = inf_asset::encode(&vgeom_demo_mesh()).unwrap();
            assert_eq!(
                std::fs::read(&vmesh).unwrap(),
                want_mesh,
                "committed Dense.inf_mesh drifted from the generator"
            );
        }

        // Partitioned-world lock (P16.5): only the `.inf_lvl` is committed — the
        // `.inf_part` is DERIVED by the cook (like `.inf_vmesh`).
        let pwdir = partitioned_world_dir();
        let pwlvl = pwdir.join("PartitionedWorld.inf_lvl");
        if pwlvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&partitioned_world_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&pwlvl).unwrap(),
                want_lvl,
                "committed partitioned-world .inf_lvl drifted from the generator"
            );
        }

        // Streamed-terrain lock (P16.3b2): only the `.inf_lvl` is committed — the
        // `.inf_terrain` is generated into a fixture's Content dir by the gate.
        let sdir = streamed_terrain_dir();
        let slvl = sdir.join("StreamedTerrain.inf_lvl");
        if slvl.exists() {
            let want_lvl = crate::scene::serialize::encode(
                &crate::scene::serialize::to_scene_file(&streamed_terrain_scene()),
            )
            .unwrap();
            assert_eq!(
                std::fs::read(&slvl).unwrap(),
                want_lvl,
                "committed streamed-terrain .inf_lvl drifted from the generator"
            );
        }
    }

    // ── Streamed-terrain sample shape (P16.3b2) ────────────────────────────

    /// The generated `.inf_terrain` really is what the gate needs: at least two
    /// coarse pyramid levels, and a level-0 grid wider than any single render
    /// wants radius — otherwise "the camera pages tiles" would never be exercised.
    #[test]
    fn streamed_terrain_asset_has_a_pyramid_and_outgrows_the_wants_radius() {
        let asset = streamed_terrain_asset();
        let reader = asset.reader();
        assert!(
            reader.lod_levels() >= 3,
            "need level 0 + at least two coarse levels, got {}",
            reader.lod_levels()
        );
        let level0 = reader.keys().filter(|k| k.is_lod0()).count();
        assert_eq!(
            level0,
            (STREAMED_TERRAIN_TILES * STREAMED_TERRAIN_TILES) as usize
        );
        assert_eq!(reader.tile_resolution(), STREAMED_TERRAIN_RESOLUTION);
        assert_eq!(reader.meters_per_sample(), STREAMED_TERRAIN_MPS);

        // The world is far wider than the finest streaming radius, so a cut over
        // it is genuinely partial.
        let span = (STREAMED_TERRAIN_RESOLUTION as f64 - 1.0) * STREAMED_TERRAIN_MPS;
        assert!(streamed_terrain_world_size() > 4.0 * span);

        // The payload is a pure function of the generators (the cook, and the
        // fixture setup, must be able to reproduce it byte for byte).
        assert_eq!(asset.as_bytes(), streamed_terrain_asset().as_bytes());

        // The level ships NO tiles — the whole point of the asset ref.
        let doc = streamed_terrain_scene();
        let (data, _) = doc
            .terrain_data_and_origin(STREAMED_TERRAIN_TERRAIN_GUID)
            .expect("terrain entity present");
        assert!(data.is_empty(), "a streamed level must ship no tiles");
    }

    /// The two scripted camera paths must really be different, or gate (c) would
    /// prove nothing.
    /// The partitioned-world scene survives the P3 discipline (save→load→save is
    /// byte-identical) **and** carries the partition settings + the two v10
    /// component markers the gate depends on.
    #[test]
    fn partitioned_world_saves_and_reloads_byte_identical() {
        use inf_ecs::components::{AlwaysLoaded, StreamingSource};
        let doc = partitioned_world_scene();
        let file = crate::scene::serialize::to_scene_file(&doc);
        let bytes1 = crate::scene::serialize::encode(&file).unwrap();
        let mut back = SceneDoc::new();
        crate::scene::serialize::apply_to_doc(
            &mut back,
            &crate::scene::serialize::decode(&bytes1).unwrap(),
        );
        let bytes2 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&back))
                .unwrap();
        assert_eq!(bytes1, bytes2, "save→load→save must be byte-identical");

        // The settings block is what makes this a partitioned level.
        let settings = back.settings();
        assert!(settings.partition.enabled);
        assert_eq!(settings.partition.cell_size_m, PARTITIONED_CELL_SIZE_M);
        assert_eq!(
            settings.partition.activation_radius_m,
            PARTITIONED_ACTIVATION_RADIUS_M
        );

        // One prop per cell, plus the far child, plus the three persistent-ish
        // entities (manager / sun / walker).
        let n = (PARTITIONED_GRID * PARTITIONED_GRID) as usize;
        assert_eq!(file.entities.len(), n + 4);

        let w = back.world();
        let src = w.entity_of(PARTITIONED_WALKER_GUID).expect("walker");
        assert!(w.world().get::<StreamingSource>(src).is_some());
        let sun = w.entity_of(PARTITIONED_SUN_GUID).expect("sun");
        assert!(w.world().get::<AlwaysLoaded>(sun).is_some());
        // The far child really is parented to the far-corner prop.
        let child = w.entity_of(PARTITIONED_CHILD_GUID).expect("child");
        let parent = w
            .entity_of(partitioned_prop_guid(
                PARTITIONED_GRID - 1,
                PARTITIONED_GRID - 1,
            ))
            .expect("far prop");
        assert_eq!(w.parent_of(child), Some(parent));
    }

    /// The scripted walk really crosses cells — otherwise the streaming gate
    /// would be asserting over a world that never streams.
    #[test]
    fn partitioned_walk_crosses_every_cell_in_its_row() {
        use std::collections::BTreeSet;
        let seen: BTreeSet<i32> = (0..PARTITIONED_GRID as usize * 3)
            .map(|i| {
                let p = partitioned_walk_point(i);
                (p.x / PARTITIONED_CELL_SIZE_M).floor() as i32
            })
            .collect();
        assert_eq!(
            seen,
            (0..PARTITIONED_GRID).collect::<BTreeSet<i32>>(),
            "the walk must visit every cell of row z=0"
        );
        // …and each prop sits unambiguously inside exactly one cell.
        for cz in 0..PARTITIONED_GRID {
            for cx in 0..PARTITIONED_GRID {
                let p = partitioned_prop_position(cx, cz);
                assert_eq!((p.x / PARTITIONED_CELL_SIZE_M).floor() as i32, cx);
                assert_eq!((p.z / PARTITIONED_CELL_SIZE_M).floor() as i32, cz);
            }
        }
    }

    #[test]
    fn streamed_terrain_camera_paths_diverge() {
        let a: Vec<_> = (0..40).map(streamed_terrain_camera_a).collect();
        let b: Vec<_> = (0..40).map(streamed_terrain_camera_b).collect();
        assert_ne!(a, b);
        let far = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (*p - *q).length())
            .fold(0.0f64, f64::max);
        assert!(far > streamed_terrain_world_size() * 0.2, "paths too close");
        // And the walk crosses the world (so sim residency really slides).
        let walk: Vec<_> = (0..120).map(streamed_terrain_walk_point).collect();
        let dx = walk.last().unwrap().x - walk[0].x;
        assert!(
            dx.abs() > streamed_terrain_world_size() * 0.5,
            "walk too short"
        );
    }

    // ── Character-demo gate test (a): byte-identical save/reload ────────────

    /// GATE (a) — the P3 discipline applied to the character demo: save → load →
    /// save is byte-identical (a genuine schema-v5 payload), and the reloaded doc
    /// keeps the full P11 animation/character component set on the character.
    #[test]
    fn character_demo_saves_and_reloads_byte_identical() {
        use inf_ecs::components::{
            AnimStateMachine, CharacterController3D, Collider3D, RigidBody3D, RootMotion,
            SkeletalMesh,
        };

        let doc = character_demo_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0],
            crate::scene::serialize::SCHEMA_VERSION as u8,
            "character-demo writes at the current schema"
        );

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
            "character-demo save→load→save must be byte-identical"
        );

        // The character keeps every persisted anim/character component + its refs.
        let ce = doc2.entity_of(CHARACTER_DEMO_CHARACTER_GUID).unwrap();
        let w = doc2.world().world();
        assert_eq!(
            w.get::<SkeletalMesh>(ce).unwrap().skeleton,
            Some(CHARACTER_DEMO_SKELETON_GUID)
        );
        assert_eq!(
            w.get::<AnimStateMachine>(ce).unwrap().sm,
            Some(CHARACTER_DEMO_SM_GUID)
        );
        assert!(w.get::<RootMotion>(ce).is_some());
        assert!(w.get::<CharacterController3D>(ce).is_some());
        assert!(w.get::<Collider3D>(ce).is_some());
        assert!(w.get::<RigidBody3D>(ce).is_some());
    }

    /// The committed anim assets decode + the state machine references the three
    /// committed clip GUIDs (the cook's SM→clip closure walks exactly these).
    #[test]
    fn character_demo_state_machine_references_its_clips() {
        use inf_anim::state_machine::Motion;
        let sm = character_demo_state_machine();
        let clip_of = |i: usize| match &sm.states[i].motion {
            Motion::Clip(c) => uuid::Uuid::from_bytes(*c),
            _ => panic!("expected a clip motion"),
        };
        assert_eq!(clip_of(0), CHARACTER_DEMO_IDLE_CLIP_GUID);
        assert_eq!(clip_of(1), CHARACTER_DEMO_RUN_CLIP_GUID);
        assert_eq!(clip_of(2), CHARACTER_DEMO_JUMP_CLIP_GUID);
        // The actor blueprint round-trips through its committed encoding.
        let class = character_demo_class();
        assert_eq!(decode_actor(&encode_actor(&class).unwrap()).unwrap(), class);
    }

    // ── Terrain-demo gate test (a): byte-identical save/reload ─────────────

    /// GATE (a) — the P3 discipline applied to the terrain-demo: save → load →
    /// save is byte-identical, and the reloaded doc keeps the terrain (heights +
    /// materialized splat weights) and the PCG volume's graph ref.
    #[test]
    fn terrain_demo_saves_and_reloads_byte_identical() {
        use inf_ecs::components::{PcgVolume, Terrain};

        let doc = terrain_demo_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0],
            crate::scene::serialize::SCHEMA_VERSION as u8,
            "terrain-demo writes at the current schema"
        );

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
            "terrain-demo save→load→save must be byte-identical"
        );

        // Terrain survives with a value probe matching the generator function.
        let te = doc2.entity_of(TERRAIN_DEMO_TERRAIN_GUID).unwrap();
        let terrain = doc2.world().world().get::<Terrain>(te).unwrap();
        assert!(
            terrain.data.tile_count() >= 4,
            "multi-tile terrain persists"
        );
        let probe = terrain
            .data
            .height_at(glam::DVec2::new(16.0, 16.0))
            .unwrap();
        assert!(
            (probe - terrain_demo_height(16.0, 16.0)).abs() < 1e-3,
            "height probe {probe} matches the generator function"
        );
        assert!(
            terrain.data.tiles().any(|(_, t)| !t.weights_are_default()),
            "painted (materialized) splat weights persist"
        );

        // The PCG volume keeps its graph ref (its evaluated cache is not persisted).
        let pe = doc2.entity_of(TERRAIN_DEMO_PCG_GUID).unwrap();
        let vol = doc2.world().world().get::<PcgVolume>(pe).unwrap();
        assert_eq!(vol.graph, Some(TERRAIN_DEMO_PCG_ASSET_GUID));
        assert!(vol.evaluated.is_empty());
    }

    /// The demo's PCG graph, evaluated over the demo terrain, places a few hundred
    /// instances — the in-editor reference the runtime gate matches against.
    #[test]
    fn terrain_demo_pcg_scatters_a_few_hundred_instances() {
        use inf_pcg::height::FnHeight;
        use inf_pcg::Region;

        let doc = terrain_demo_scene();
        let te = doc.entity_of(TERRAIN_DEMO_TERRAIN_GUID).unwrap();
        let data = doc
            .world()
            .world()
            .get::<inf_ecs::components::Terrain>(te)
            .unwrap()
            .data
            .clone();
        let provider = FnHeight::new(move |x, z| data.height_at(glam::DVec2::new(x, z)));
        let region = Region::from_xz(0.0, 0.0, TERRAIN_DEMO_SPAN, TERRAIN_DEMO_SPAN);
        let insts = inf_pcg::evaluate(&terrain_demo_pcg_document(), &provider, region);
        assert!(
            insts.len() > 100,
            "expected a few hundred instances, got {}",
            insts.len()
        );
        // Deterministic across two evaluations.
        let insts2 = inf_pcg::evaluate(&terrain_demo_pcg_document(), &provider, region);
        assert_eq!(insts, insts2);
    }

    // ── Physics-playground gate scene (P12.4) ──────────────────────────────

    /// The P3 discipline applied to the playground: save → load → save is
    /// byte-identical (a genuine schema-v6 payload), and the reloaded doc keeps the
    /// joints (incl. the `other` entity refs), the audio sources (incl. `clip`
    /// refs), the listener, and the collision-layer / CCD collider fields.
    #[test]
    fn physics_playground_saves_and_reloads_byte_identical() {
        use inf_ecs::components::{
            AudioListener, AudioSource, Collider3D, Joint3D, JointKind3D, RigidBody3D,
        };

        let doc = physics_playground_scene();
        let bytes1 =
            crate::scene::serialize::encode(&crate::scene::serialize::to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0],
            crate::scene::serialize::SCHEMA_VERSION as u8,
            "physics-playground writes at the current schema"
        );

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
            "physics-playground save→load→save must be byte-identical"
        );

        let w = doc2.world().world();
        // The motorized spinner joint persists with its motor + other ref.
        let we = doc2.entity_of(PLAYGROUND_SPINNER_WHEEL_GUID).unwrap();
        let sj = w.get::<Joint3D>(we).expect("spinner joint persists");
        assert_eq!(sj.kind, JointKind3D::Revolute);
        assert_eq!(
            sj.other,
            inf_ecs::EntityRef::new(PLAYGROUND_SPINNER_HUB_GUID)
        );
        assert!(sj.motor_enabled);
        assert_eq!(sj.motor_target_vel, 8.0);
        // The spinner's autoplay/looping/occluded AudioSource persists.
        let src = w.get::<AudioSource>(we).expect("spinner audio persists");
        assert_eq!(src.clip, Some(PLAYGROUND_SPINNER_CLIP_GUID));
        assert!(src.autoplay && src.looping && src.occlusion && src.spatial);
        // The CCD bullet's collider/body fields persist.
        let be = doc2.entity_of(PLAYGROUND_BULLET_GUID).unwrap();
        assert!(w.get::<RigidBody3D>(be).unwrap().ccd_enabled);
        // The ghost pair's collision-layer filter persists (empty filter).
        let ge = doc2.entity_of(PLAYGROUND_GHOST_A_GUID).unwrap();
        assert_eq!(w.get::<Collider3D>(ge).unwrap().collision_filter, 0);
        // The sensor plate is a persisted trigger volume.
        let se = doc2.entity_of(PLAYGROUND_SENSOR_GUID).unwrap();
        assert!(w.get::<Collider3D>(se).unwrap().sensor);
        // The camera carries the active listener.
        let ce = doc2.entity_of(PLAYGROUND_CAMERA_GUID).unwrap();
        assert!(w.get::<AudioListener>(ce).unwrap().active);
        // The ragdoll produced 8 bodies + 7 joints (its descs mapped to components).
        let ragdoll_joints = (0..8)
            .filter_map(|i| doc2.entity_of(Uuid::from_u128(PLAYGROUND_RAGDOLL_BASE_GUID + i)))
            .filter(|&e| w.get::<Joint3D>(e).is_some())
            .count();
        assert_eq!(ragdoll_joints, 7, "ragdoll wires 7 parent joints");
    }
}
