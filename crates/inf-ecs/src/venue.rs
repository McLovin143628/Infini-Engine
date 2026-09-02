//! **A venue makes a noise** (island wave VEN1b) — the music emitter a level's
//! own buildings imply, derived beside the people they imply.
//!
//! `inf_pcg`'s assembler put a [`Music`](inf_pcg-doc) station over the middle of
//! every venue's main room and the mirror carried it across as
//! [`PcgVolume::emitters`](crate::components::PcgVolume::emitters). This module
//! is the half that needs an ECS: an emitter is a *place*, and the thing that
//! plays is an **entity with an [`AudioSource`]** — which has been a complete,
//! persisted component since scene v6, so placing one moves no schema.
//!
//! # Derived, and therefore never saved
//!
//! [`sync_venue_audio`] is [`crate::society::sync_society`]'s shape exactly, one
//! system smaller: it reconciles a set of synthetic entities against what the
//! resident volumes say, minting each `Guid` from the level's own content
//! ([`venue_music_guid`]) so two hosts spawn the same emitters without talking.
//! A volume that streams out takes its emitter with it, on the same step.
//!
//! The entities are **runtime** entities, on `crate::crowd::materialize`'s
//! terms: an editor Simulate session restores its `SceneDoc` snapshot when it
//! stops, so nothing here can reach an author's `.inf_lvl`, and
//! [`clear_venue_audio`] is the twin of `clear_crowd` for the same reason.
//!
//! # Why an ENTITY and not another special case in `audio_step`
//!
//! The vehicle engine loop is a special case: `audio_step` walks
//! `self.vehicles` and synthesizes its commands. It could have been the pattern
//! here and deliberately is not, because a real `AudioSource` buys the two
//! things a synthesized command cannot:
//!
//! * every reader of `AudioSource` sees it — the autoplay walk, the despawn
//!   sweep that stops a voice when its emitter goes, and (the one this wave
//!   needs) the **occlusion path**, which now re-evaluates a looping spatial
//!   source every step;
//! * and it is *inspectable*: a venue's music is an entity in the world with a
//!   bus, a volume and a rolloff on it, rather than four constants in a
//!   host-side loop that two hosts have to be compared character for character
//!   to keep in step.
//!
//! [`AudioSource`]: crate::components::AudioSource
//! [inf_pcg-doc]: crate::components::AudioEmitterSlot

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::Component;
use glam::DVec3;
use uuid::Uuid;

use crate::components::{AudioSource, DistanceModel, Guid, PcgVolume, Transform, Visibility};
use crate::math::Vec3d;
use crate::world::EcsWorld;

/// **The engine's committed club loop** — the `.inf_audio` a venue plays.
///
/// A fixed GUID, on `inf_editor_core::samples::starter_character_ids`' own
/// terms: an asset a level names by id must have the same id every time or the
/// committed bytes are a different set of files on every build.
///
/// A host that has not loaded the clip resolves nothing and plays silence; the
/// **command** is issued either way, which is the Phase-12 doctrine's own
/// observable (`inf_audio::command`: *"the command stream — not the audible
/// output — is the observable contract"*).
pub const VENUE_MUSIC_CLIP: Uuid = Uuid::from_u128(0x5645_4E31_0000_0001);

/// The bus a venue's music plays on. `music`, and not `sfx`, so a player who
/// turns the music down turns the club down.
pub const VENUE_MUSIC_BUS: &str = "music";

/// The base volume of a venue's music loop, linear.
///
/// Below unity on purpose: this is the loudest continuous thing in a
/// settlement, and it is heard *through* a doorway most of the time.
pub const VENUE_MUSIC_VOLUME: f64 = 0.85;

/// Metres inside which a venue's music is at full volume before occlusion.
///
/// Two metres — a body's own reach. Inside a dance floor this is the whole
/// difference between standing at the speaker and standing across the room.
pub const VENUE_MUSIC_MIN_M: f64 = 2.0;

/// Metres past which a venue's music is silent.
///
/// Forty. A settlement's blocks are about a hundred and twenty metres apart, so
/// this reaches the street outside and the pavement opposite, and never the
/// next block — which is what stops three venues in one city being audible at
/// once from anywhere.
pub const VENUE_MUSIC_MAX_M: f64 = 40.0;

/// The rolloff exponent of a venue's music. `1.0` is the inverse-distance
/// default; the model is [`DistanceModel::Inverse`].
pub const VENUE_MUSIC_ROLLOFF: f64 = 1.0;

/// Salts a venue emitter's derived `Guid`. See [`venue_music_guid`].
const SALT_MUSIC: u64 = 0x4d55_5349_4300_0001;

/// The tag every derived emitter `Guid` carries in its top sixteen bits —
/// `"MU"`.
///
/// `crate::society::AGENT_TAG`'s convention and its disclaimer: not a namespace
/// guarantee, there so a guid in a trace is recognizable. What guarantees an
/// emitter never overwrites a level entity is the refusal in
/// [`sync_venue_audio`], which asks the world.
const MUSIC_TAG: u128 = 0x4d55;

/// **The `Guid` of the emitter in one venue's main room** — a hash of the
/// level's own content, so two hosts mint the same one without talking.
///
/// `(volume, index)` is the level's name for that speaker.
/// [`crate::society::agent_guid`]'s shape and its argument: nothing about it
/// depends on iteration order, on when the volume streamed in, or on how many
/// emitters have been minted already.
pub fn venue_music_guid(volume: Uuid, index: u32) -> Uuid {
    let b = volume.as_u128();
    let hi = crate::society::mix64((b as u64) ^ SALT_MUSIC ^ u64::from(index));
    let lo = crate::society::mix64(((b >> 64) as u64) ^ hi);
    let raw = (u128::from(hi) << 64) | u128::from(lo);
    Uuid::from_u128((MUSIC_TAG << 112) | (raw & ((1u128 << 112) - 1)))
}

/// **This entity is a venue's music**, and which volume's.
///
/// A component the scene serializer does not know about, on
/// [`crate::crowd::CrowdAgent`]'s terms — so nothing here can be saved and
/// **scene v27 does not move**. It is what makes the reconcile below able to
/// find its own emitters without a registry that could drift from the world.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct VenueMusic {
    /// The `PcgVolume` whose building this speaker hangs in.
    pub volume: Uuid,
}

/// **The `AudioSource` a venue's main room carries.**
///
/// One place, so the two hosts cannot describe the same speaker differently —
/// which they would, because this used to be four constants in a host-side loop
/// in the shape `vehicle_engine_audio` still has.
///
/// `occlusion: true` is the load-bearing field: it is what puts this source on
/// the doorway path (`inf_physics::d3::audio`), and it is the reason that path
/// can be an upgrade rather than a rewrite — **no committed content sets it**
/// (`AudioSource::default` is `false`), so nothing that predates this wave
/// changes by a byte.
pub fn venue_music_source() -> AudioSource {
    AudioSource {
        clip: Some(VENUE_MUSIC_CLIP),
        bus: VENUE_MUSIC_BUS.to_string(),
        volume: VENUE_MUSIC_VOLUME,
        pitch: 1.0,
        looping: true,
        spatial: true,
        min_distance: VENUE_MUSIC_MIN_M,
        max_distance: VENUE_MUSIC_MAX_M,
        distance_model: DistanceModel::Inverse,
        rolloff: VENUE_MUSIC_ROLLOFF,
        occlusion: true,
        autoplay: true,
    }
}

/// What one [`sync_venue_audio`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VenueAudioStats {
    /// Emitters the resident volumes offer.
    pub wanted: usize,
    /// Emitter entities in the world after this sync.
    pub live: usize,
    /// Emitters spawned on this step.
    pub spawned: usize,
    /// Emitters despawned on this step — a volume streamed out and took its
    /// speaker with it.
    pub despawned: usize,
    /// **Emitter `Guid`s the world already held as something else**, and which
    /// were therefore refused. `crate::crowd::add_agents`' own counter, for its
    /// own reason: a synthetic guid that collided with a level entity would
    /// make that entity unreachable.
    pub refused: usize,
}

/// **Reconcile a level's venue music against its resident volumes** — the one
/// Ring-0 door both hosts call, once per fixed step, in the society phase.
///
/// Cheap when nothing changed: one walk over the entities, two `BTreeSet`
/// differences, and nothing else on a step where the same venues are resident.
/// A level with no venue never grows the sets and never spawns anything, which
/// is why every level that predates this wave is byte-identical.
pub fn sync_venue_audio(world: &mut EcsWorld) -> VenueAudioStats {
    let mut stats = VenueAudioStats::default();
    // ── what the level WANTS ────────────────────────────────────────────────
    let mut want: BTreeMap<Uuid, (Uuid, DVec3)> = BTreeMap::new();
    let mut have: BTreeMap<Uuid, crate::Entity> = BTreeMap::new();
    {
        let w = world.world();
        for e in w.iter_entities() {
            let Some(g) = e.get::<Guid>() else {
                continue;
            };
            if e.get::<VenueMusic>().is_some() {
                have.insert(g.0, e.id());
            }
            let Some(v) = e.get::<PcgVolume>() else {
                continue;
            };
            if v.emitters.is_empty() {
                continue;
            }
            // The volume's own origin, because `PcgVolume::emitters` are in the
            // world already (`place_in_frame` put them there beside the
            // colliders) — this is the same reading `society::volume_sites`
            // takes and it is here only to refuse a volume with no transform.
            if e.get::<crate::components::GlobalTransform>().is_none() {
                continue;
            }
            for (i, m) in v.emitters.iter().enumerate() {
                if !m.at.is_finite() {
                    continue;
                }
                want.insert(venue_music_guid(g.0, i as u32), (g.0, m.at));
            }
        }
    }
    stats.wanted = want.len();

    // ── the ones that have gone ─────────────────────────────────────────────
    let stale: Vec<crate::Entity> = have
        .iter()
        .filter(|(g, _)| !want.contains_key(*g))
        .map(|(_, e)| *e)
        .collect();
    for e in stale {
        world.despawn(e);
        stats.despawned += 1;
    }

    // ── and the ones that are new ───────────────────────────────────────────
    let existing: BTreeSet<Uuid> = have.keys().copied().collect();
    for (guid, (volume, at)) in &want {
        if existing.contains(guid) {
            continue;
        }
        // **The refusal that matters** (`add_agents`' own): a synthetic guid the
        // world already holds belongs to something else, and overwriting it
        // would make a level entity unreachable.
        if world.entity_of(*guid).is_some() {
            stats.refused += 1;
            continue;
        }
        let e = world.spawn_with_guid(*guid, "Venue Music", None);
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: Vec3d::new(at.x, at.y, at.z),
                ..Transform::IDENTITY
            },
            Visibility::default(),
            venue_music_source(),
            VenueMusic { volume: *volume },
        ));
        stats.spawned += 1;
    }
    stats.live = stats.wanted - stats.refused;
    stats
}

/// **Forget a level's venue music** — the twin of
/// [`crate::crowd::clear_crowd`], called beside it for the same reason: a
/// `SceneDoc` snapshot restores entities and touches no resource, so a stopped
/// Simulate session's speakers would otherwise outlive the run that spawned
/// them.
pub fn clear_venue_audio(world: &mut EcsWorld) {
    let doomed: Vec<crate::Entity> = world
        .world()
        .iter_entities()
        .filter(|e| e.get::<VenueMusic>().is_some())
        .map(|e| e.id())
        .collect();
    for e in doomed {
        world.despawn(e);
    }
}

/// **Every venue emitter in the world**, by `Guid`, in `Guid` order — the read
/// a gate and an instrument take.
pub fn venue_emitters(world: &EcsWorld) -> Vec<(Uuid, DVec3)> {
    let mut out: Vec<(Uuid, DVec3)> = world
        .world()
        .iter_entities()
        .filter(|e| e.get::<VenueMusic>().is_some())
        .filter_map(|e| {
            let g = e.get::<Guid>()?.0;
            let t = e.get::<Transform>()?;
            Some((g, t.translation.to_dvec3()))
        })
        .collect();
    out.sort_by_key(|(g, _)| *g);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{AudioEmitterSlot, GlobalTransform};

    fn volume_with(world: &mut EcsWorld, guid: Uuid, at: DVec3, emitters: usize) {
        world.spawn_with_guid(guid, "block", None);
        let e = world.entity_of(guid).expect("the block");
        let mut vol = PcgVolume::default();
        vol.set_population(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Default::default(),
            Vec::new(),
            (0..emitters)
                .map(|i| AudioEmitterSlot {
                    at: at + DVec3::new(i as f64, 0.0, 0.0),
                    room: i as u32,
                })
                .collect(),
        );
        world.world_mut().entity_mut(e).insert((
            Transform::IDENTITY,
            GlobalTransform(glam::DAffine3::IDENTITY),
            vol,
        ));
    }

    /// **A venue gets a speaker, and it goes when the venue does** — the whole
    /// of the reconcile, both directions, because a sync that only ever spawns
    /// leaks a voice per streamed-out block for the life of a session.
    #[test]
    fn a_venue_gets_a_speaker_and_loses_it_when_the_block_streams_out() {
        let mut world = EcsWorld::new();
        assert_eq!(
            sync_venue_audio(&mut world),
            VenueAudioStats::default(),
            "a level with no venue spawned something"
        );

        let v = Uuid::from_u128(0x5E_1);
        volume_with(&mut world, v, DVec3::new(10.0, 3.0, 4.0), 1);
        let a = sync_venue_audio(&mut world);
        assert_eq!((a.wanted, a.spawned, a.despawned, a.live), (1, 1, 0, 1));
        // Idempotent: a settled step spawns nothing.
        let b = sync_venue_audio(&mut world);
        assert_eq!((b.wanted, b.spawned, b.despawned), (1, 0, 0));

        // …and it plays what a venue plays, at where the assembler put it.
        let emitters = venue_emitters(&world);
        assert_eq!(emitters.len(), 1);
        assert_eq!(emitters[0].0, venue_music_guid(v, 0));
        assert_eq!(emitters[0].1, DVec3::new(10.0, 3.0, 4.0));
        let e = world.entity_of(emitters[0].0).expect("the speaker");
        let src = world
            .world()
            .get::<AudioSource>(e)
            .expect("a speaker with no AudioSource on it plays nothing");
        assert!(src.looping && src.spatial && src.occlusion && src.autoplay);
        assert_eq!(src.bus, VENUE_MUSIC_BUS);
        assert_eq!(src.clip, Some(VENUE_MUSIC_CLIP));

        // The block streams out: the volume entity goes, and the speaker with
        // it on the very next sync.
        let block = world.entity_of(v).expect("the block");
        world.despawn(block);
        let c = sync_venue_audio(&mut world);
        assert_eq!((c.wanted, c.spawned, c.despawned, c.live), (0, 0, 1, 0));
        assert!(venue_emitters(&world).is_empty());
    }

    /// Two emitters in one volume are two speakers, and their ids are the
    /// LEVEL's rather than the order's.
    #[test]
    fn an_emitters_id_is_the_levels_own_and_not_the_orders() {
        let v = Uuid::from_u128(0x5E_2);
        let mut a = EcsWorld::new();
        volume_with(&mut a, v, DVec3::ZERO, 2);
        sync_venue_audio(&mut a);
        let mut b = EcsWorld::new();
        volume_with(
            &mut b,
            Uuid::from_u128(0x5E_3),
            DVec3::new(99.0, 0.0, 0.0),
            1,
        );
        volume_with(&mut b, v, DVec3::ZERO, 2);
        sync_venue_audio(&mut b);

        let ka: Vec<Uuid> = venue_emitters(&a).into_iter().map(|(g, _)| g).collect();
        let kb: Vec<Uuid> = venue_emitters(&b)
            .into_iter()
            .map(|(g, _)| g)
            .filter(|g| ka.contains(g))
            .collect();
        assert_eq!(ka.len(), 2, "two emitters made {} speakers", ka.len());
        assert_eq!(ka, kb, "the same venue minted different speakers");
        assert_ne!(ka[0], ka[1], "two emitters of one volume share an id");
    }

    /// The clear door really clears, which is what stops a stopped Simulate
    /// session's speakers outliving it.
    #[test]
    fn the_clear_door_takes_every_speaker() {
        let mut world = EcsWorld::new();
        volume_with(&mut world, Uuid::from_u128(9), DVec3::ZERO, 2);
        sync_venue_audio(&mut world);
        assert_eq!(venue_emitters(&world).len(), 2);
        clear_venue_audio(&mut world);
        assert!(venue_emitters(&world).is_empty());
        // …and the volume itself is untouched: this clears speakers, not blocks.
        assert!(world.entity_of(Uuid::from_u128(9)).is_some());
    }
}
