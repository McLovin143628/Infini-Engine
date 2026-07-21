//! Skeletal-animation systems (P11.1).
//!
//! [`advance_anim_players`] is the per-fixed-step system that ticks every
//! [`AnimPlayer`]'s clip play-head. It is called from **both** the editor's
//! Simulate loop (`inf_editor_core::simulate::SimSession`) and the runtime sim
//! (`inf_player::runtime_sim::RuntimeSim`) so preview == shipped: one clip-time
//! integration, run in `Guid`-independent (per-entity, order-free) fashion so it
//! stays deterministic under any iteration order (§2.5).
//!
//! Pose *evaluation* (clip → skinning palette) is not done here — it happens at
//! render projection time, where the skeleton/clip assets are available. This
//! system only advances `t`; the wrap/clamp math lives on [`AnimPlayer::advance`].

use crate::components::AnimPlayer;
use crate::world::EcsWorld;

/// Advance every [`AnimPlayer`] in the world by one fixed step of `dt` seconds.
/// Paused players are skipped (see [`AnimPlayer::advance`]). Order-independent
/// (each player integrates only its own `t`), so determinism holds regardless of
/// archetype iteration order.
pub fn advance_anim_players(world: &mut EcsWorld, dt: f64) {
    let w = world.world_mut();
    let mut query = w.query::<&mut AnimPlayer>();
    for mut player in query.iter_mut(w) {
        player.advance(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advances_playing_players_and_skips_paused() {
        let mut world = EcsWorld::new();
        let playing = world
            .world_mut()
            .spawn(AnimPlayer {
                t: 0.0,
                speed: 1.0,
                duration: 10.0,
                ..Default::default()
            })
            .id();
        let paused = world
            .world_mut()
            .spawn(AnimPlayer {
                t: 0.5,
                playing: false,
                ..Default::default()
            })
            .id();

        advance_anim_players(&mut world, 0.25);

        let t_playing = world.world().get::<AnimPlayer>(playing).unwrap().t;
        let t_paused = world.world().get::<AnimPlayer>(paused).unwrap().t;
        assert!((t_playing - 0.25).abs() < 1e-9);
        assert_eq!(t_paused, 0.5, "paused player did not advance");
    }
}
