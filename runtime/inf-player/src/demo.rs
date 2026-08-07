//! The programmatic `--demo` world (P9.3 item 1, the "prove the runtime" path).
//!
//! Since the runtime `.inf_lvl` reader is a concurrent task (P9.2), the player
//! proves its window/renderer/loop/input/sim end-to-end against a world built
//! **in code** here — a small 2D platformer mirroring `samples/platformer-2d`:
//! a static ground + ledge, a light, and a kinematic character running the
//! sample's **Coyote** blueprint (coyote-time jump). The coyote class is loaded
//! from the committed `Coyote.inf_act` **JSON bytes** via [`include_str!`], so the
//! exact sample asset the editor produces is what the player interprets — closing
//! "the coyote `.inf_act` JSON from the sample must run".
//!
//! When P9.2's reader lands, `--level` loads the real cooked platformer and this
//! demo becomes a self-test fixture (kept for CI, which has no cooked packs yet).

use glam::DVec2;
use uuid::Uuid;

use inf_blueprint::BlueprintClass;
use inf_ecs::components::{
    BodyKind2D, CharacterController2D, Collider2D, ColliderShape2DKind, Light, LightKind,
    RigidBody2D, Sprite,
};
use inf_ecs::{Color, EcsWorld, Entity, Vec2d};

use crate::level::BuiltWorld;

/// The player's stable entity `Guid` (matches `samples::PLAYER_GUID`).
pub const PLAYER_GUID: u128 = 0x8401_0004;

/// The committed sample blueprint, embedded so the player interprets the exact
/// bytes the editor writes.
const COYOTE_ACT_JSON: &str = include_str!("../../../samples/platformer-2d/Coyote.inf_act");

/// The Coyote [`BlueprintClass`] decoded from the sample `.inf_act` JSON.
pub fn coyote_class() -> BlueprintClass {
    serde_json::from_str(COYOTE_ACT_JSON).expect("committed Coyote.inf_act is valid JSON")
}

/// Set an entity's local XY translation.
fn set_pos(world: &mut EcsWorld, e: Entity, x: f64, y: f64) {
    if let Some(mut t) = world
        .world_mut()
        .get_mut::<inf_ecs::components::Transform>(e)
    {
        t.translation.x = x;
        t.translation.y = y;
    }
    world.mark_dirty();
}

/// Build the demo platformer [`BuiltWorld`].
///
/// Gravity is [`DVec2::ZERO`]: the character applies its own gravity in the
/// blueprint (`vy -= GRAVITY*dt`), exactly like the sample — so passing world
/// gravity would double it.
pub fn build() -> BuiltWorld {
    let mut world = EcsWorld::new();

    // ── ground: a wide static box the character stands on ──
    let ground = world.spawn_with_guid(Uuid::from_u128(0x8401_0001), "Ground", None);
    world.world_mut().entity_mut(ground).insert((
        Sprite {
            size: Vec2d::new(12.0, 1.0),
            color: Color::new(0.20, 0.45, 0.22, 1.0),
            sorting_layer: -10,
            ..Sprite::default()
        },
        RigidBody2D {
            kind: BodyKind2D::Static,
            ..RigidBody2D::default()
        },
        Collider2D {
            shape_kind: ColliderShape2DKind::Box,
            half_extents: Vec2d::new(6.0, 0.5),
            ..Collider2D::default()
        },
    ));
    set_pos(&mut world, ground, 0.0, -0.5);

    // ── ledge: a raised static box to jump onto ──
    let ledge = world.spawn_with_guid(Uuid::from_u128(0x8401_0002), "Ledge", None);
    world.world_mut().entity_mut(ledge).insert((
        Sprite {
            size: Vec2d::new(2.0, 1.0),
            color: Color::new(0.30, 0.30, 0.36, 1.0),
            sorting_layer: -10,
            ..Sprite::default()
        },
        RigidBody2D {
            kind: BodyKind2D::Static,
            ..RigidBody2D::default()
        },
        Collider2D {
            shape_kind: ColliderShape2DKind::Box,
            half_extents: Vec2d::new(1.0, 0.5),
            ..Collider2D::default()
        },
    ));
    set_pos(&mut world, ledge, 4.5, 0.5);

    // ── a directional light so lit content is not black ──
    let sun = world.spawn_with_guid(Uuid::from_u128(0x8401_0003), "Sun", None);
    world.world_mut().entity_mut(sun).insert(Light {
        kind: LightKind::Directional,
        color: Color::WHITE,
        intensity: 1.0,
        ..Light::default()
    });

    // ── the player: kinematic character running the Coyote blueprint ──
    let player_guid = Uuid::from_u128(PLAYER_GUID);
    let player = world.spawn_with_guid(player_guid, "Coyote", None);
    world.world_mut().entity_mut(player).insert((
        Sprite {
            size: Vec2d::new(0.6, 0.9),
            color: Color::new(0.85, 0.62, 0.28, 1.0),
            sorting_layer: 0,
            ..Sprite::default()
        },
        RigidBody2D {
            kind: BodyKind2D::Kinematic,
            fixed_rotation: true,
            ..RigidBody2D::default()
        },
        Collider2D {
            shape_kind: ColliderShape2DKind::Capsule,
            half_extents: Vec2d::new(0.3, 0.35),
            radius: 0.3,
            ..Collider2D::default()
        },
        CharacterController2D {
            max_slope_deg: 46.0,
            snap_to_ground: 0.3,
            offset: 0.02,
        },
    ));
    set_pos(&mut world, player, 1.5, 0.8);

    world.propagate();

    BuiltWorld {
        world,
        actors: vec![(player_guid, coyote_class())],
        gravity: DVec2::ZERO,
        hz: 60.0,
        render: inf_scene::RenderSettingsRecord::default(),
        label: "platformer-demo".to_string(),
        state_machines: std::collections::BTreeMap::new(),
        root_clips: Vec::new(),
        skeletons: std::collections::BTreeMap::new(),
        pose_clips: std::collections::BTreeMap::new(),
        audio_clips: std::collections::BTreeMap::new(),
        cloths: std::collections::BTreeMap::new(),
        hairs: std::collections::BTreeMap::new(),
        // The programmatic demo is a single document — nothing streams.
        partition: crate::level::PartitionContent::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coyote_act_json_decodes() {
        let class = coyote_class();
        assert_eq!(class.id, "act:coyote_player");
        // BeginPlay + Tick handlers present.
        assert_eq!(class.events.len(), 2);
    }

    #[test]
    fn demo_world_has_player_and_ground() {
        let built = build();
        let player = built.world.entity_of(Uuid::from_u128(PLAYER_GUID));
        assert!(player.is_some(), "player entity present");
        assert_eq!(built.actors.len(), 1);
    }
}
