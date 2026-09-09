//! **Who is listening** (wave WPN2c) — the one Ring-0 answer to a question two
//! hosts and one sim step were all asking separately.
//!
//! # What this closes
//!
//! `runtime_sim.rs` and `simulate.rs` each carried a private `active_listener`
//! that walked every entity for an active [`AudioListener`] and took the lowest
//! [`Guid`]. Two copies of one rule is the class of thing the `weapon_report`
//! MIRROR fence exists to catch, and these two were never fenced — they only
//! happened to agree.
//!
//! Wave WPN2c needs a **third** reader, and it is not a host: the report's
//! distant layer is a function of how far the listener is from the muzzle, and
//! the supersonic crack is a function of how close a round passes to it. Both
//! are decided inside the fixed step, where neither host's private function can
//! be called. A third copy would have been the one that drifted.
//!
//! So the rule lives here, both hosts call it, and the sim calls it — and a
//! shot's distance to the listener is now a number the trace can be compared on
//! rather than a number each process computed for itself.

use glam::DVec3;
use uuid::Uuid;

use crate::components::{AudioListener, GlobalTransform, Guid, Transform};
use crate::EcsWorld;

/// **The active listener**, as `(guid, world position)`, or `None` on a level
/// that has none.
///
/// "The first active [`AudioListener`] in `Guid` order" — the rule both hosts
/// have followed since P12.3, moved rather than changed, so no committed audio
/// command stream moves with it.
///
/// A level with no listener answers `None`, and each caller decides what that
/// means: a host leaves the engine's listener at its default pose (the world
/// origin — see `inf_editor_core::island`'s own note about what that cost the
/// island before VEN1b put an ear on the hero), and the sim treats a shot as
/// having no listener at all, which makes the distant layer silent and the
/// supersonic crack impossible. Both are the honest answer to "nobody is
/// listening".
///
/// `O(entities)` with one component probe each, on `crate::movement::
/// camera_subject`'s own terms: it is walked once per step per caller, and a
/// level with no listener pays one archetype scan.
pub fn active_listener_pose(world: &EcsWorld) -> Option<(Uuid, DVec3)> {
    let mut best: Option<(Uuid, DVec3)> = None;
    for e in world.world().iter_entities() {
        let Some(al) = e.get::<AudioListener>() else {
            continue;
        };
        if !al.active {
            continue;
        }
        let guid = e.get::<Guid>().map(|g| g.0).unwrap_or_else(Uuid::nil);
        let pos = e
            .get::<GlobalTransform>()
            .map(|g| g.translation())
            .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        if best.as_ref().map(|(g, _)| guid < *g).unwrap_or(true) {
            best = Some((guid, pos));
        }
    }
    best
}

/// **Where the active listener is**, or `None` — [`active_listener_pose`]
/// without the identity, for the callers that only want the distance.
pub fn active_listener_position(world: &EcsWorld) -> Option<DVec3> {
    active_listener_pose(world).map(|(_, p)| p)
}

/// **Every clip this ENGINE plays without a level naming it** (wave WPN2c) —
/// the door island-progress carried item 43 prescribes, built at last.
///
/// # The defect, in one sentence
///
/// `runtime_packager::cook::asset_deps` closes a pack over the assets a level's
/// entities REFERENCE, and the PIE payload builder walks `doc.order()` for
/// `AudioSource.clip` — so a sound that a *fixed step* decides to play, naming
/// its clip by a Ring-0 constant, is invisible to both. Measured at island wave
/// I8b and written up as carried 43: **the cooked island pack contains no
/// `.inf_audio` at all**, so the venue's music is silent in a shipped build and
/// nobody noticed, because a `Play` whose clip does not resolve is silence with
/// no error.
///
/// Wave WPN2c would have made it thirty-six times worse: every gunshot is four
/// commands naming four engine constants, and every one of them would have been
/// silent in a cooked build while sounding perfectly in the editor.
///
/// # The rule
///
/// A clip belongs on this list when the SIM names it and no component does.
/// That is the whole test, and it is why the list is here rather than in a
/// manifest: the constants are Ring-0, the systems that play them are Ring-0,
/// and a wire field would be a fourth place for the same fact.
///
/// Ordered and deduplicated, so a pack's dependency closure is deterministic.
pub fn engine_spawned_clips() -> Vec<Uuid> {
    let mut out = vec![
        // Island wave VEN1b: a venue's music is played by an emitter
        // `sync_venue_audio` SPAWNS from a `PcgVolume`, so no authored entity
        // ever carries the clip.
        crate::venue::VENUE_MUSIC_CLIP,
        // Wave WPN2c: the brass.
        crate::weapon::CASING_CLIP,
    ];
    // Wave WPN1 + WPN2c: thirty-six report clips, five per class, built by
    // `report_clip` — including WPN1's own `WEAPON_REPORT_CLIP`, which is the
    // assault rifle's body layer.
    for class in crate::weapon::WeaponClass::ALL {
        for clip in crate::weapon::ReportClip::ALL {
            out.push(crate::weapon::report_clip(class, clip));
        }
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3d;

    /// **CARRIED 43'S DOOR** — every clip the engine plays without a level
    /// naming it, so a cook and a PIE payload can close over them.
    #[test]
    fn the_engine_names_thirty_seven_clips_no_component_ever_references() {
        let clips = active_test_clips();
        assert_eq!(
            clips.len(),
            37,
            "thirty-six report clips plus the casing plus the venue loop"
        );
        let uniq: std::collections::BTreeSet<Uuid> = clips.iter().copied().collect();
        assert_eq!(uniq.len(), clips.len(), "the list has a duplicate");
        // Sorted, so a pack's dependency closure is deterministic.
        let mut sorted = clips.clone();
        sorted.sort();
        assert_eq!(clips, sorted);
        // The two the tree has committed the longest are on it.
        assert!(clips.contains(&crate::weapon::WEAPON_REPORT_CLIP));
        assert!(clips.contains(&crate::venue::VENUE_MUSIC_CLIP));
        assert!(clips.contains(&crate::weapon::CASING_CLIP));
    }

    fn active_test_clips() -> Vec<Uuid> {
        engine_spawned_clips()
    }

    #[test]
    fn the_listener_is_the_lowest_guid_that_is_active_and_nothing_otherwise() {
        let mut w = EcsWorld::new();
        assert_eq!(active_listener_pose(&w), None);

        let hi = Uuid::from_u128(0x22);
        let lo = Uuid::from_u128(0x11);
        for (g, x) in [(hi, 10.0), (lo, -4.0)] {
            let e = w.spawn_with_guid(g, "Ear", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(x, 1.0, 0.0);
            w.world_mut()
                .entity_mut(e)
                .insert((t, AudioListener { active: true }));
        }
        // **The pose comes off `GlobalTransform` when there is one**, and a
        // freshly spawned entity has an identity one until the transform pass
        // has run — measured on this test's own first draft, which read the ear
        // at the origin. That is the host behaviour too, and it is why this is
        // called after the propagate step in both of them.
        crate::transform::propagate(w.world_mut());
        let (g, p) = active_listener_pose(&w).expect("an ear");
        assert_eq!(g, lo);
        assert!((p.x + 4.0).abs() < 1e-12);

        // An inactive listener is not one, however low its guid.
        let e = w.entity_of(lo).expect("the low ear");
        w.world_mut()
            .entity_mut(e)
            .insert(AudioListener { active: false });
        assert_eq!(active_listener_pose(&w).map(|(g, _)| g), Some(hi));
    }
}
