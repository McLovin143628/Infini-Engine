//! Level loading seams (P9.3 item 1 · wired to real content in P9.5).
//!
//! The player loads a world through two narrow traits so the P9.2 pieces (the
//! pack format + the runtime `.inf_lvl` decode in `inf-scene`) slot in cleanly:
//!
//! * [`LevelSource`] — produces the **raw serialized level bytes**.
//!   [`DevDirLevelSource`] reads an `.inf_lvl` file straight off disk (the
//!   `--level` dev-dir path); [`PackLevelSource`] opens a cooked
//!   `content.inf_pack` (+ `manifest.toml`) and returns the root level's bytes
//!   (the `--pack` / exported-game path).
//! * [`WorldBuilder`] — **decodes those bytes into a populated [`BuiltWorld`]**
//!   (an ECS world + the blueprint actors to tick + gravity/rate).
//!   [`InfSceneWorldBuilder`] is the real, P9.2-backed decoder;
//!   [`StubWorldBuilder`] is kept only for the "reader not wired" error surface.
//!
//! ## What the level format persists (and what it does not)
//!
//! `.inf_lvl` (the frozen `inf_scene` schema v2, mirroring the editor's
//! `EntityRecord`) persists per entity: `guid`, `name`, `parent`, `transform`,
//! `visible`, and the renderable/authoring components `mesh` / `material` /
//! `light` / `camera` / `sprite` / `tilemap` / `nine_slice` / `text2d` /
//! `light_2d`. [`InfSceneWorldBuilder`] instantiates **all** of them, so a cooked
//! level renders exactly as the editor viewport shows it.
//!
//! As of **schema v3** (P9.5) it **also** persists per entity the 2D + 3D
//! **physics** components (`RigidBody2D` / `Collider2D` /
//! `CharacterController2D` and the 3D trio) and a per-entity **blueprint-class
//! binding** ([`ActorClass`](inf_ecs::components::ActorClass), the `actor` slot),
//! plus a file-level settings record (gravity + rate). [`InfSceneWorldBuilder`]
//! instantiates them all, so a cooked level is fully runnable:
//!
//! * gravity/rate come from the level's own settings (a v1/v2 level lifts to the
//!   [`DEFAULT_GRAVITY`] / [`DEFAULT_HZ`] fallback, matching the
//!   platformer/`--demo` convention where the character applies its own gravity);
//! * actors bind from the **persisted `ActorClass` links** ([`resolve_bound_actors`]):
//!   each entity's `actor` GUID resolves — through the pack index / dev-dir
//!   sidecars — to a `BlueprintClass`. The legacy **`CharacterController2D`
//!   heuristic** ([`resolve_actors`]) is kept as a documented fallback for levels
//!   authored before v3 (kinder for hand-rolled levels).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_anim::{AnimClip, AnimClipAsset, Skeleton, SkeletonAsset, StateMachine, StateMachineAsset};
use inf_asset::{AssetId, AssetKind, PackReader};
use inf_blueprint::BlueprintClass;
use inf_ecs::components::{
    CharacterController2D, GlobalTransform, PcgVolume, ScatteredInstance, Spline, Terrain,
    Transform,
};
use inf_ecs::{EcsWorld, Guid};
use inf_pcg::height::FnHeight;
use inf_pcg::{GrammarPass, PcgAssetPayload, Region};
use inf_scene::partition::PartitionSettings;
use inf_scene::{RenderSettingsRecord, RuntimeEntity};

use crate::terrain_stream::TerrainSource;

/// The cook's default pack file name (kept in sync with
/// `inf_packager::DEFAULT_PACK_NAME`; duplicated here so the shipped player does
/// not depend on the cook pipeline).
pub const PACK_FILE: &str = "content.inf_pack";

/// The cook's manifest file name (kept in sync with `inf_packager::MANIFEST_FILE`).
pub const MANIFEST_FILE: &str = "manifest.toml";

/// Default world gravity for a loaded level (a level-settings record is the
/// follow-up). [`DVec2::ZERO`] matches the platformer/`--demo` convention: the
/// character applies its own gravity in the blueprint, so a nonzero world gravity
/// would double it.
pub const DEFAULT_GRAVITY: DVec2 = DVec2::ZERO;

/// Default fixed update rate (Hz) for a loaded level.
pub const DEFAULT_HZ: f64 = 60.0;

/// A populated world ready to hand to [`RuntimeSim`](crate::runtime_sim::RuntimeSim).
pub struct BuiltWorld {
    /// The ECS world (entities + components).
    pub world: EcsWorld,
    /// The blueprint actors to tick: `(entity Guid, class)`.
    pub actors: Vec<(Uuid, BlueprintClass)>,
    /// World gravity for the 2D physics bridge (the platformer uses `ZERO`).
    pub gravity: DVec2,
    /// Fixed update rate (Hz).
    pub hz: f64,
    /// The level's scene-persisted render block (R-P4 · schema v8): post /
    /// exposure / lighting the render host maps onto the live `RenderSettings`
    /// (see `crate::render::apply_record`). Defaults for a settings-less level.
    pub render: RenderSettingsRecord,
    /// A human label for logs / the window title.
    pub label: String,
    /// Resolved `.inf_sm` state machines keyed by asset GUID (P11.4) — the map
    /// the caller seeds [`RuntimeSim::set_state_machines`](crate::runtime_sim::RuntimeSim::set_state_machines)
    /// with so an `AnimStateMachine` entity steps like the editor Simulate.
    pub state_machines: BTreeMap<Uuid, StateMachine>,
    /// Resolved root-motion clips: `(clip asset GUID, skeleton, clip)` (P11.4) —
    /// the caller registers each via
    /// [`RuntimeSim::register_root_motion_clip`](crate::runtime_sim::RuntimeSim::register_root_motion_clip).
    /// A clip whose skeleton ref doesn't resolve is dropped (root motion needs it).
    pub root_clips: Vec<(Uuid, Skeleton, AnimClip)>,
    /// Resolved `.inf_skel` assets keyed by asset GUID (P24.1) — the caller seeds
    /// [`RuntimeSim::set_skeletons`](crate::runtime_sim::RuntimeSim::set_skeletons)
    /// so a machine-driven character is posed by its machine rather than drawn at
    /// rest. Sockets ride along (the attachment system reads them).
    pub skeletons: BTreeMap<Uuid, inf_anim::SkeletonAsset>,
    /// Resolved `.inf_anim` clips a state machine's states play, keyed by asset
    /// GUID (P24.1) — the caller seeds
    /// [`RuntimeSim::set_pose_clips`](crate::runtime_sim::RuntimeSim::set_pose_clips).
    pub pose_clips: BTreeMap<Uuid, AnimClip>,
    /// Resolved `.inf_cloth` garments keyed by asset GUID (P24.4) — the caller
    /// seeds
    /// [`RuntimeSim::set_cloths`](crate::runtime_sim::RuntimeSim::set_cloths) so a
    /// `ClothSim` wearer simulates in the shipped build exactly as it does in the
    /// editor's Simulate.
    pub cloths: BTreeMap<Uuid, inf_anim::ClothAsset>,
    /// Resolved `.inf_hair` hairstyles keyed by asset GUID (P24.4) - the caller
    /// seeds
    /// [`RuntimeSim::set_hairs`](crate::runtime_sim::RuntimeSim::set_hairs).
    pub hairs: BTreeMap<Uuid, inf_anim::HairAsset>,
    /// Resolved `.inf_audio` clips keyed by asset GUID (P12.3) — the caller seeds
    /// [`RuntimeSim::set_audio_clips`](crate::runtime_sim::RuntimeSim::set_audio_clips)
    /// so a scene `AudioSource` plays the same clip as in the editor Simulate.
    pub audio_clips: BTreeMap<Uuid, inf_audio::AudioAsset>,
    /// Where this world's streamed **partition cells** come from (P16.5).
    /// [`PartitionContent::None`] for every unpartitioned level, in which case
    /// `world` already holds every entity and nothing streams.
    pub partition: PartitionContent,
}

impl BuiltWorld {
    /// Take the partition source out of the built world (it is moved into the
    /// [`CellStreaming`](crate::cell_stream::CellStreaming) manager, and
    /// [`sim_from_built`](crate::sim_from_built) consumes the rest).
    pub fn take_partition(&mut self) -> PartitionContent {
        std::mem::replace(&mut self.partition, PartitionContent::None)
    }
}

/// A partitioned level's **resolved** cell source (P16.5) — the cell mirror of
/// [`crate::TerrainContent`].
///
/// Resolved eagerly by [`InfSceneWorldBuilder::build`], because the persistent
/// cell is not an afterthought: it *is* the level's world at step 0, and it has
/// to be spawned before actor binding runs, or a partitioned level would boot
/// with no blueprint bound to anything.
///
/// Two sources, one store trait, and that is the PIE-==-shipping seam. A cooked
/// level ships its entities in a `.inf_part` sliced out of the pack mapping; a
/// loose / PIE level still carries them inline, so the runtime bins them with the
/// **same Ring-0 function the cook used**. Same cells, same order, same
/// activation timeline.
#[derive(Default, Clone)]
pub enum PartitionContent {
    /// The level is not partitioned (or has nothing to stream).
    #[default]
    None,
    /// A resolved cell store plus the level's partition settings.
    Streamed(Arc<dyn crate::cell_stream::CellStore>, PartitionSettings),
}

impl PartitionContent {
    /// Whether anything streams.
    pub fn is_none(&self) -> bool {
        matches!(self, PartitionContent::None)
    }

    /// The level's partition settings (defaults when nothing streams).
    pub fn settings(&self) -> PartitionSettings {
        match self {
            PartitionContent::None => PartitionSettings::default(),
            PartitionContent::Streamed(_, s) => *s,
        }
    }

    /// The resolved cell store, if this level streams.
    pub fn store(&self) -> Option<&Arc<dyn crate::cell_stream::CellStore>> {
        match self {
            PartitionContent::None => None,
            PartitionContent::Streamed(s, _) => Some(s),
        }
    }
}

/// Resolve a partitioned level's cell store.
///
/// # The discriminator is "is there a pack?", not "is the level empty?"
///
/// Those two look interchangeable — a cooked level's entities moved into the
/// `.inf_part`, so it *is* empty — but they diverge on a case that is trivially
/// reachable and reads as a bug: an author ticks **Enabled** in World Settings on
/// a brand-new, still-empty scratch level and hits Play. There is no pack, and
/// nothing to stream, and the honest answer is "an empty partitioned world, which
/// is empty". Discriminating on emptiness instead would send that level down the
/// pack arm and fail it with a message about a `.inf_part` the user has never
/// heard of and could not have produced.
///
/// So: **no pack → bin whatever the level carries in memory** (zero entities
/// included; the loose `.inf_lvl`, the PIE handoff, an uncooked project). **Pack
/// → the cooked `.inf_part`**, and a missing one there is a genuine hard error —
/// a cooked partitioned level ships no entities at all, so degrading to "run it
/// unpartitioned" would silently boot an empty world, the one failure worse than
/// not booting.
fn resolve_cell_store(
    entities: &[RuntimeEntity],
    settings: &PartitionSettings,
    pack: Option<&(Arc<PackReader>, AssetId)>,
) -> Result<Arc<dyn crate::cell_stream::CellStore>, String> {
    let Some((reader, level_id)) = pack else {
        return Ok(Arc::new(
            crate::cell_stream::MemoryCellStore::from_entities(entities, settings),
        ));
    };
    let id = AssetId(crate::cell_stream::derived_partition_id(level_id.uuid()));
    if !reader.contains(id) {
        return Err(format!(
            "level is partitioned but its .inf_part ({id}) is not in the pack — the cooked \
             level ships no entities of its own, so this world would boot empty"
        ));
    }
    Ok(Arc::new(crate::cell_stream::PackCellStore::open(
        reader.clone(),
        id,
    )?))
}

/// Produces the raw serialized bytes of a level (an `.inf_lvl` payload). The
/// dev-dir implementation reads a file; the pack-backed one (P9.2) reads a pack
/// entry.
pub trait LevelSource {
    /// The level's raw bytes.
    fn level_bytes(&self) -> Result<Vec<u8>, String>;
    /// A human label for logs / the window title.
    fn label(&self) -> String;
    /// Resolve a blueprint **class asset** by its GUID — the persisted `actor`
    /// binding a v3 `.inf_lvl` carries (P9.5). Returns `None` when the source has
    /// no such asset (the default; legacy sources without a GUID index fall back
    /// to the [`resolve_actors`] heuristic).
    fn blueprint_by_guid(&self, _guid: Uuid) -> Option<BlueprintClass> {
        None
    }
}

/// Decodes raw level bytes into a populated [`BuiltWorld`]. The runtime reader
/// (`inf-scene`, P9.2) provides the real implementation.
pub trait WorldBuilder {
    fn build(&self, level_bytes: &[u8]) -> Result<BuiltWorld, String>;
}

/// v1 [`LevelSource`]: read an `.inf_lvl` file straight off disk.
pub struct DevDirLevelSource {
    path: PathBuf,
}

impl DevDirLevelSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl LevelSource for DevDirLevelSource {
    fn level_bytes(&self) -> Result<Vec<u8>, String> {
        std::fs::read(&self.path).map_err(|e| format!("read level {}: {e}", self.path.display()))
    }

    fn label(&self) -> String {
        self.path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("level")
            .to_string()
    }
}

/// The placeholder [`WorldBuilder`] kept only to surface a clear "reader not
/// wired" error (unused on the real paths, which use [`InfSceneWorldBuilder`]).
pub struct StubWorldBuilder;

impl WorldBuilder for StubWorldBuilder {
    fn build(&self, _level_bytes: &[u8]) -> Result<BuiltWorld, String> {
        Err("no world builder wired (use InfSceneWorldBuilder)".to_string())
    }
}

/// The real [`WorldBuilder`]: decode `.inf_lvl` bytes with the Ring-0
/// [`inf_scene`] reader and instantiate a populated [`EcsWorld`], then bind the
/// discovered blueprint actor classes to `CharacterController2D` entities.
///
/// Holds the actor classes discovered beside the level (dev-dir) or in the pack
/// so [`WorldBuilder::build`] — which only sees the level bytes — can attach them.
pub struct InfSceneWorldBuilder {
    /// Fallback classes for the [`resolve_actors`] heuristic — used only when a
    /// level carries **no** persisted `actor` bindings (a legacy / hand-rolled
    /// level). Kept because it is kinder for levels authored before v3.
    fallback: Vec<BlueprintClass>,
    /// Persisted-binding resolution: blueprint **asset GUID** → its class. Built
    /// from the [`LevelSource`] (pack index / dev-dir sidecars).
    by_guid: HashMap<Uuid, BlueprintClass>,
    /// `.inf_pcg` graph payloads keyed by asset GUID — the map used to evaluate a
    /// [`PcgVolume`]'s scatter **on load** (its `evaluated` cache is never
    /// persisted; see [`evaluate_pcg_volumes`]). Built from the [`LevelSource`]
    /// (pack index / dev-dir sidecars) or the streamed PIE payload.
    pcgs: HashMap<Uuid, PcgAssetPayload>,
    /// `.inf_biomes` payloads keyed by asset GUID (P19.3) — the level's biome
    /// vocabularies, resolved so a terrain's painted ids can dispatch their
    /// graphs on load (see [`evaluate_biome_bindings`]).
    biome_sets: HashMap<Uuid, inf_terrain::BiomeSet>,
    /// `.inf_skel` payloads keyed by asset GUID (P11.4) — resolves the skeleton a
    /// clip / skeletal mesh references.
    skeletons: HashMap<Uuid, SkeletonAsset>,
    /// `.inf_anim` clip payloads keyed by asset GUID (P11.4).
    clips: HashMap<Uuid, AnimClipAsset>,
    /// `.inf_sm` state-machine payloads keyed by asset GUID (P11.4).
    machines: HashMap<Uuid, StateMachineAsset>,
    /// `.inf_audio` clip payloads keyed by asset GUID (P12.3).
    audio: HashMap<Uuid, inf_audio::AudioAsset>,
    /// `.inf_cloth` garment payloads keyed by asset GUID (P24.4).
    cloths: HashMap<Uuid, inf_anim::ClothAsset>,
    /// `.inf_hair` hairstyle payloads keyed by asset GUID (P24.4).
    hairs: HashMap<Uuid, inf_anim::HairAsset>,
    /// Gravity/rate used only when the level predates settings (v1/v2) — a v3
    /// level's own [`LevelSettings`](inf_scene::RuntimeSettings) always wins.
    gravity: DVec2,
    hz: f64,
    /// The cooked pack + this level's asset id, so a **partitioned** level can be
    /// resolved to its derived `.inf_part` (P16.5). `None` on the loose /
    /// PIE path, where a partitioned level is binned in memory instead.
    partition_pack: Option<(Arc<PackReader>, AssetId)>,
}

impl InfSceneWorldBuilder {
    /// Build with explicit fallback gravity/rate (used for pre-settings levels).
    pub fn new(fallback: Vec<BlueprintClass>, gravity: DVec2, hz: f64) -> Self {
        Self {
            fallback,
            by_guid: HashMap::new(),
            pcgs: HashMap::new(),
            biome_sets: HashMap::new(),
            skeletons: HashMap::new(),
            clips: HashMap::new(),
            machines: HashMap::new(),
            audio: HashMap::new(),
            cloths: HashMap::new(),
            hairs: HashMap::new(),
            gravity,
            hz,
            partition_pack: None,
        }
    }

    /// Build with the documented defaults ([`DEFAULT_GRAVITY`] / [`DEFAULT_HZ`]).
    pub fn with_defaults(fallback: Vec<BlueprintClass>) -> Self {
        Self::new(fallback, DEFAULT_GRAVITY, DEFAULT_HZ)
    }

    /// Attach the persisted-binding resolution map (asset GUID → class). Builder
    /// style so `build_world` can wire it from the source's blueprint index.
    pub fn with_bindings(mut self, by_guid: HashMap<Uuid, BlueprintClass>) -> Self {
        self.by_guid = by_guid;
        self
    }

    /// Attach the `.inf_pcg` payloads (asset GUID → graph payload) used to evaluate
    /// [`PcgVolume`] scatter on load. Builder style so `build_world` can wire it
    /// from the source's PCG index (or a PIE payload).
    pub fn with_pcgs(mut self, pcgs: HashMap<Uuid, PcgAssetPayload>) -> Self {
        self.pcgs = pcgs;
        self
    }

    /// Attach the `.inf_biomes` payloads (asset GUID → biome set) the P19.3
    /// biome→PCG binding dispatches from. Builder style, exactly like
    /// [`with_pcgs`](Self::with_pcgs), so `build_world` wires it from the
    /// source's biome-set index (or a PIE payload).
    pub fn with_biome_sets(mut self, biome_sets: HashMap<Uuid, inf_terrain::BiomeSet>) -> Self {
        self.biome_sets = biome_sets;
        self
    }

    /// Attach the P11 animation asset maps (asset GUID → payload) used to seed the
    /// [`RuntimeSim`](crate::runtime_sim::RuntimeSim)'s state machines + root-motion
    /// clips. Builder style so `build_world` can wire them from the source's anim
    /// index (or a PIE payload).
    pub fn with_anim_assets(
        mut self,
        skeletons: HashMap<Uuid, SkeletonAsset>,
        clips: HashMap<Uuid, AnimClipAsset>,
        machines: HashMap<Uuid, StateMachineAsset>,
    ) -> Self {
        self.skeletons = skeletons;
        self.clips = clips;
        self.machines = machines;
        self
    }

    /// Attach the `.inf_cloth` garments (asset GUID → payload) used to seed the
    /// [`RuntimeSim`](crate::runtime_sim::RuntimeSim)'s cloth registry (P24.4).
    /// Builder style, wired by `build_world` from the source's cloth index.
    pub fn with_cloth_assets(mut self, cloths: HashMap<Uuid, inf_anim::ClothAsset>) -> Self {
        self.cloths = cloths;
        self
    }

    /// Attach the `.inf_hair` hairstyles (asset GUID -> payload) used to seed the
    /// [`RuntimeSim`](crate::runtime_sim::RuntimeSim)'s hair registry (P24.4).
    pub fn with_hair_assets(mut self, hairs: HashMap<Uuid, inf_anim::HairAsset>) -> Self {
        self.hairs = hairs;
        self
    }

    /// Attach the cooked pack + this level's asset id so a partitioned level
    /// resolves to its derived `.inf_part` (P16.5). Builder style, wired by
    /// [`build_world_from_pack`](crate::build_world_from_pack). Without it a
    /// partitioned level is binned in memory from its own entities — which is
    /// exactly right for the loose / PIE path, and produces the identical cells.
    pub fn with_partition_pack(mut self, reader: Arc<PackReader>, level: AssetId) -> Self {
        self.partition_pack = Some((reader, level));
        self
    }

    /// Attach the `.inf_audio` payloads (asset GUID → payload) used to seed the
    /// [`RuntimeSim`](crate::runtime_sim::RuntimeSim)'s audio clips (P12.3). Builder
    /// style, wired by `build_world` from the source's audio index.
    pub fn with_audio(mut self, audio: HashMap<Uuid, inf_audio::AudioAsset>) -> Self {
        self.audio = audio;
        self
    }

    /// Resolve the `.inf_audio` payloads into the deterministic `Guid → AudioAsset`
    /// map the runtime sim seeds (P12.3).
    fn resolve_audio_clips(&self) -> BTreeMap<Uuid, inf_audio::AudioAsset> {
        self.audio.iter().map(|(g, a)| (*g, a.clone())).collect()
    }

    /// Resolve the `.inf_sm` machines into the `Guid → StateMachine` map the
    /// [`RuntimeSim`](crate::runtime_sim::RuntimeSim) steps against (P11.4).
    fn resolve_state_machines(&self) -> BTreeMap<Uuid, StateMachine> {
        self.machines
            .iter()
            .map(|(g, a)| (*g, a.machine.clone()))
            .collect()
    }

    /// Resolve each `.inf_anim` clip into `(clip GUID, skeleton, clip)` for
    /// root-motion registration, joining the clip's skeleton ref to a loaded
    /// `.inf_skel` (P11.4). Clips whose skeleton doesn't resolve are dropped (root
    /// motion needs a skeleton to sample the root joint). Deterministic (Guid order).
    fn resolve_root_clips(&self) -> Vec<(Uuid, Skeleton, AnimClip)> {
        let mut out: Vec<(Uuid, Skeleton, AnimClip)> = self
            .clips
            .iter()
            .filter_map(|(g, ca)| {
                let skel_guid = Uuid::from_bytes(ca.skeleton?);
                let sk = self.skeletons.get(&skel_guid)?;
                Some((*g, sk.skeleton.clone(), ca.clip.clone()))
            })
            .collect();
        out.sort_by_key(|(g, _, _)| *g);
        out
    }

    /// The loaded `.inf_skel` assets, keyed by asset GUID (P24.1) — the map the
    /// [`RuntimeSim`](crate::runtime_sim::RuntimeSim) poses machine-driven
    /// characters against.
    ///
    /// Every indexed skeleton, not only the ones a `SkeletalMesh` names: the
    /// source's anim index already resolved exactly the level's referenced set
    /// (pack index / dev-dir sidecars / PIE payload), so filtering again here
    /// would only risk dropping one.
    fn resolve_skeletons(&self) -> BTreeMap<Uuid, SkeletonAsset> {
        self.skeletons
            .iter()
            .map(|(g, a)| (*g, a.clone()))
            .collect()
    }

    /// The loaded `.inf_anim` clips, keyed by asset GUID (P24.1) — the clips a
    /// state machine's states play.
    ///
    /// Unlike [`resolve_root_clips`](Self::resolve_root_clips) this keeps clips
    /// whose `skeleton` ref is `None` or unresolvable: the pose path takes its
    /// skeleton from the *entity*, so a skeleton-agnostic clip is perfectly
    /// playable — it is only root motion that needs the join.
    fn resolve_pose_clips(&self) -> BTreeMap<Uuid, AnimClip> {
        self.clips
            .iter()
            .map(|(g, ca)| (*g, ca.clip.clone()))
            .collect()
    }
}

impl WorldBuilder for InfSceneWorldBuilder {
    fn build(&self, level_bytes: &[u8]) -> Result<BuiltWorld, String> {
        let level = inf_scene::decode(level_bytes).map_err(|e| format!("decode level: {e}"))?;
        let title = level.title;
        let settings = level.settings;

        // ── P16.5 world partition ──
        //
        // Three shapes, one rule ("the level's entities live wherever the
        // partition says"):
        //
        // * not partitioned → every entity spawns now, exactly as before;
        // * partitioned, no pack (a loose `.inf_lvl`, the PIE handoff, an
        //   uncooked project) → bin the level's own entities here with the same
        //   Ring-0 function the cook uses;
        // * partitioned, cooked pack → they are in the derived `.inf_part`.
        //
        // In BOTH partitioned shapes the **persistent cell is spawned right here**,
        // through the normal population path — not by `CellStreaming::attach`, and
        // that ordering is load-bearing rather than incidental. `RuntimeSim`
        // assigns blueprint actor ids ONCE, at construction, from the entities
        // present then; the streamer is attached after the sim exists, so spawning
        // the persistent cell there would hand `RuntimeSim::new` an empty world and
        // bind ZERO actors — silently, because an unbound actor is not an error,
        // it is just a prop that never moves. Spawning here means
        // `resolve_bound_actors` below sees the persistent cell and a partitioned
        // level's blueprints bind exactly as an unpartitioned level's do.
        // (Grid cells spawn later, at the streamer's sync points — and an entity
        // that arrives that way genuinely does NOT get a ticking blueprint in v1;
        // the cook warns about it, and the negative test in
        // `runtime/inf-player/tests/partitioned_world.rs` pins it.)
        let (entities, partition) = if settings.partition.enabled {
            let store = resolve_cell_store(
                &level.entities,
                &settings.partition,
                self.partition_pack.as_ref(),
            )?;
            let persistent = store.persistent()?;
            (
                persistent,
                PartitionContent::Streamed(store, settings.partition),
            )
        } else {
            (level.entities, PartitionContent::None)
        };

        let mut world = populate_world(entities);
        world.propagate();
        // Evaluate PCG scatter volumes on load: their `evaluated` cache is never
        // persisted (`#[serde(skip)]`), so the player recomputes it from the
        // referenced `.inf_pcg` graph against the level's terrain (P10.6).
        evaluate_pcg_volumes(&mut world, &self.pcgs);
        // …and the terrain-level sibling: each painted biome's graph over the
        // region its id owns (P19.3). Same reason, same never-persisted cache.
        evaluate_biome_bindings(&mut world, &self.biome_sets, &self.pcgs);
        // Prefer the level's persisted per-entity actor bindings; fall back to the
        // CC2D heuristic only when the level carries none (legacy levels).
        let actors = resolve_bound_actors(&world, &self.by_guid);
        let actors = if actors.is_empty() {
            resolve_actors(&world, &self.fallback)
        } else {
            actors
        };
        // v3 levels carry their own gravity/rate; v1/v2 lift to defaults, which
        // equal the constructor's fallback for the platformer convention.
        let gravity = DVec2::new(settings.gravity_2d.x, settings.gravity_2d.y);
        let hz = settings.sim_hz;
        tracing::info!(
            "inf-player: built '{}' — {} actor(s) bound (gravity {:?}, {} Hz)",
            if title.is_empty() { "level" } else { &title },
            actors.len(),
            gravity,
            hz
        );
        let _ = (self.gravity, self.hz); // retained for pre-settings call sites
        Ok(BuiltWorld {
            world,
            actors,
            gravity,
            hz,
            render: settings.render,
            label: if title.is_empty() {
                "level".to_string()
            } else {
                title
            },
            state_machines: self.resolve_state_machines(),
            root_clips: self.resolve_root_clips(),
            skeletons: self.resolve_skeletons(),
            pose_clips: self.resolve_pose_clips(),
            audio_clips: self.resolve_audio_clips(),
            cloths: self.cloths.iter().map(|(g, c)| (*g, c.clone())).collect(),
            hairs: self.hairs.iter().map(|(g, h)| (*g, h.clone())).collect(),
            partition,
        })
    }
}

/// Instantiate an [`EcsWorld`] from a decoded level's entities: spawn each with
/// its stable `Guid` + name, insert every persisted component, then rebuild the
/// hierarchy. Entities arrive parents-first (the `inf_scene` invariant), but the
/// reparent is a deliberate second pass so it is robust to order.
pub fn populate_world(entities: Vec<RuntimeEntity>) -> EcsWorld {
    let mut world = EcsWorld::new();
    spawn_entities(&mut world, entities);
    world
}

/// Spawn `entities` **into an existing world**, returning their `Guid`s in the
/// order they were spawned.
///
/// This is [`populate_world`]'s body, lifted so world-partition cell streaming
/// (P16.5) can spawn one cell at a time into a world that is already running —
/// the same machinery, the same component set, the same two-pass reparent, so a
/// streamed cell is indistinguishable from the same entities having been in the
/// level all along. That equivalence is what the PIE-==-shipping gate rests on;
/// a second, parallel spawn path would be a place for the two to drift.
///
/// Parent links resolve **within this batch** (plus any entity already in the
/// world carrying that `Guid`), which is exactly why the partitioner keeps a
/// hierarchy in one cell — see `inf_scene::partition::partition_entities`.
pub fn spawn_entities(world: &mut EcsWorld, entities: Vec<RuntimeEntity>) -> Vec<Uuid> {
    let mut by_guid: HashMap<Uuid, inf_ecs::Entity> = HashMap::new();
    let mut pending_parents: Vec<(inf_ecs::Entity, Uuid)> = Vec::new();
    let mut spawned: Vec<Uuid> = Vec::with_capacity(entities.len());

    for e in entities {
        let RuntimeEntity {
            guid,
            name,
            parent,
            transform,
            visible,
            mesh,
            material,
            light,
            camera,
            sprite,
            tilemap,
            nine_slice,
            text2d,
            light_2d,
            rigid_body_2d,
            collider_2d,
            character_controller_2d,
            rigid_body_3d,
            collider_3d,
            character_controller_3d,
            actor,
            terrain,
            pcg_volume,
            skeletal_mesh,
            anim_player,
            anim_state_machine,
            root_motion,
            attached_to,
            joint_2d,
            joint_3d,
            audio_source,
            audio_listener,
            decal,
            volume,
            spline,
            foliage,
            streaming_source,
            always_loaded,
            time_of_day,
            sky_atmosphere,
            water_body,
            buoyancy,
            voxel_volume,
            destructible,
            ik_target,
            cloth_sim,
            hair_guides,
        } = e;

        let entity = world.spawn_with_guid(guid, &name, None);
        by_guid.insert(guid, entity);
        spawned.push(guid);
        if !visible {
            world.set_visible(entity, false);
        }
        {
            // Overwrite the identity transform and add each present component.
            let mut em = world.world_mut().entity_mut(entity);
            em.insert(transform);
            if let Some(c) = mesh {
                em.insert(c);
            }
            if let Some(c) = material {
                em.insert(c);
            }
            if let Some(c) = light {
                em.insert(c);
            }
            if let Some(c) = camera {
                em.insert(c);
            }
            if let Some(c) = sprite {
                em.insert(c);
            }
            if let Some(c) = tilemap {
                em.insert(c);
            }
            if let Some(c) = nine_slice {
                em.insert(c);
            }
            if let Some(c) = text2d {
                em.insert(c);
            }
            if let Some(c) = light_2d {
                em.insert(c);
            }
            // ── v3 physics components + actor binding ──
            if let Some(c) = rigid_body_2d {
                em.insert(c);
            }
            if let Some(c) = collider_2d {
                em.insert(c);
            }
            if let Some(c) = character_controller_2d {
                em.insert(c);
            }
            if let Some(c) = rigid_body_3d {
                em.insert(c);
            }
            if let Some(c) = collider_3d {
                em.insert(c);
            }
            if let Some(c) = character_controller_3d {
                em.insert(c);
            }
            if let Some(a) = actor {
                em.insert(inf_ecs::components::ActorClass(a));
            }
            // ── v4 world components (terrain + PCG volume) ──
            if let Some(c) = terrain {
                em.insert(c);
            }
            if let Some(c) = pcg_volume {
                em.insert(c);
            }
            // ── v5 animation / character components ──
            if let Some(c) = skeletal_mesh {
                em.insert(c);
            }
            if let Some(c) = anim_player {
                em.insert(c);
            }
            if let Some(c) = anim_state_machine {
                em.insert(c);
            }
            if let Some(c) = root_motion {
                em.insert(c);
            }
            if let Some(c) = attached_to {
                em.insert(c);
            }
            // ── v6 joints / spatial-audio components ──
            if let Some(c) = joint_2d {
                em.insert(c);
            }
            if let Some(c) = joint_3d {
                em.insert(c);
            }
            if let Some(c) = audio_source {
                em.insert(c);
            }
            if let Some(c) = audio_listener {
                em.insert(c);
            }
            // ── v8 world components (decal slot + volumes + splines + foliage) ──
            if let Some(c) = decal {
                em.insert(c);
            }
            if let Some(c) = volume {
                em.insert(c);
            }
            if let Some(c) = spline {
                em.insert(c);
            }
            if let Some(c) = foliage {
                em.insert(c);
            }
            // ── v11 sky authority (P17.1) ──
            if let Some(c) = time_of_day {
                em.insert(c);
            }
            if let Some(c) = sky_atmosphere {
                em.insert(c);
            }
            // ── v10 world-partition components (P16.5) ──
            if let Some(c) = streaming_source {
                em.insert(c);
            }
            if let Some(c) = always_loaded {
                em.insert(c);
            }
            // ── v17 water (P20.1) ──
            //
            // A `River` additionally needs the `spline` slot above, which is on
            // the same entity and is therefore already inserted by the time this
            // runs — component composition, not a reference to resolve.
            if let Some(c) = water_body {
                em.insert(c);
            }
            // ── v18 buoyancy (P20.2) ──
            //
            // The authored marker only; nothing here binds it to a body of
            // water. Which water covers it is resolved per step, so a level that
            // spawns a floating crate before its lake still floats it.
            if let Some(c) = buoyancy {
                em.insert(c);
            }
            // ── v19 volumetric terrain (P21.1) ──
            //
            // The reference + its two authored knobs only; the chunks live in the
            // `.inf_voxel` and are paged by the host, exactly as a streamed
            // terrain's tiles are. A volume whose asset the pack does not carry
            // spawns and draws as nothing, which is what "no chunks" means.
            if let Some(c) = voxel_volume {
                em.insert(c);
            }
            // ── v20 destruction (P22.2) ──
            //
            // The marker plus its five numbers only; the chunk set lives in the
            // `.inf_fracture` DERIVED from this entity's own mesh, which P22.3
            // loads when the asset actually breaks. Spawning it here costs
            // nothing and means the component is on the entity before any
            // gameplay can ask about it.
            if let Some(c) = destructible {
                em.insert(c);
            }
            // ── v21 character components (P24.3) ──
            //
            // `IkTarget` is the one with a reader today: `step_pose_evaluation`
            // walks it every fixed step, which is what makes a real `--pie`
            // subprocess engage IK through the door it always used rather than
            // through a test-only injection hook. `ClothSim` joined it at P24.4
            // (`inf_ecs::cloth::step_cloth_simulation` folds the garment named by
            // its `asset`, resolved out of this level's `.inf_cloth` entries);
            // `HairGuides` gets its reader in the same batch. Spawning them here
            // costs nothing and means the components are on the entity before
            // anything can ask.
            if let Some(c) = ik_target {
                em.insert(c);
            }
            if let Some(c) = cloth_sim {
                em.insert(c);
            }
            if let Some(c) = hair_guides {
                em.insert(c);
            }
        }
        world.mark_dirty();
        if let Some(p) = parent {
            pending_parents.push((entity, p));
        }
    }

    for (child, parent_guid) in pending_parents {
        // Prefer a parent from this batch; fall back to one already in the world
        // (a streamed cell can never need this — hierarchies do not split — but a
        // caller spawning a fragment gets the kinder behaviour rather than a
        // silently orphaned child).
        if let Some(pe) = by_guid
            .get(&parent_guid)
            .copied()
            .or_else(|| world.entity_of(parent_guid))
        {
            world.reparent(child, Some(pe));
        }
    }
    spawned
}

/// Evaluate every [`PcgVolume`] whose `graph` ref resolves in `pcgs`, refreshing
/// its `evaluated` instance cache from the scatter graph over the level's terrain
/// (P10.6). This is the runtime twin of the editor's `pcg_evaluate` command: the
/// volume's instances are never persisted in `.inf_lvl` (they are a derived
/// cache), so the shipped/PIE player must recompute them on load, exactly as the
/// editor does on demand — keeping preview == shipping.
///
/// ## v1 simplifications (documented)
///
/// * The **first** non-empty [`Terrain`] in `Guid` order drives height/slope;
///   with no terrain a flat `y = 0` plane is used (mirrors the command).
/// * The height seam is an [`FnHeight`] closure shifted by the terrain entity's
///   world origin — the general form of the [`inf_pcg::height::TerrainHeight`]
///   bridge (which is exactly `TerrainHeight(data)` when that origin is zero).
/// * Evaluation runs **once at load** (terrain is static in sim v1); a moving
///   camera / streaming re-eval is a documented follow-up.
///
/// ## P19.4 grammar passes and P19.5 buildings
///
/// A graph may also author grammar passes and building passes, which lower
/// beside the document rather than into it. They evaluate here, on the same
/// volume, against the same height provider, and their instances are appended
/// **after** the scatter instances — grammars first, then buildings, a fixed
/// order, so the cache is a pure function of the content. Everything downstream
/// of the resolved spans is [`inf_pcg::evaluate_grammars`] /
/// [`inf_pcg::evaluate_buildings`], shared verbatim with the editor's
/// `pcg_evaluate`; all this function owns is the fetch.
///
/// The **solid** half of that evaluation lands on
/// [`PcgVolume::structures`](inf_ecs::components::PcgVolume::structures), which
/// the physics bridge turns into static box colliders — the reason a
/// grammar-built building is enterable rather than merely drawn. Like
/// `evaluated`, it is derived state and is never serialized.
pub fn evaluate_pcg_volumes(world: &mut EcsWorld, pcgs: &HashMap<Uuid, PcgAssetPayload>) {
    if pcgs.is_empty() {
        return;
    }

    // Deterministic Guid-sorted entity list (parents-first order is irrelevant
    // here; sorting keeps terrain-pick + volume order stable across loads).
    let ents: Vec<(Uuid, inf_ecs::Entity)> = {
        let w = world.world();
        let mut v: Vec<(Uuid, inf_ecs::Entity)> = w
            .iter_entities()
            .filter_map(|e| e.get::<Guid>().map(|g| (g.0, e.id())))
            .collect();
        v.sort_by_key(|(g, _)| *g);
        v
    };

    // The first non-empty terrain (its paged data + world origin) drives scatter.
    let terrain: Option<(inf_terrain::TerrainData, DVec3)> = {
        let w = world.world();
        ents.iter().find_map(|&(_, e)| {
            let t = w.get::<Terrain>(e)?;
            if t.data.is_empty() {
                return None;
            }
            let origin = w
                .get::<GlobalTransform>(e)
                .map(|g| g.translation())
                .unwrap_or(DVec3::ZERO);
            Some((t.data.clone(), origin))
        })
    };

    // The world's splines, in world space — the P19.4 `grammar.spline` seam.
    // MIRROR: `commands::pcg::collect_spline_paths` in the editor.
    let splines = collect_spline_paths(world);

    // Gather the volumes to evaluate: (entity, program, center, extent, seed).
    struct Job {
        entity: inf_ecs::Entity,
        guid: Uuid,
        document: inf_pcg::PcgDocument,
        grammars: Vec<GrammarPass>,
        buildings: Vec<inf_pcg::BuildingPass>,
        center: DVec3,
        extent: inf_ecs::math::Vec2d,
        seed: u32,
    }
    let jobs: Vec<Job> = {
        let w = world.world();
        ents.iter()
            .filter_map(|&(guid, e)| {
                let vol = w.get::<PcgVolume>(e)?;
                let graph_guid = vol.graph?;
                let payload = pcgs.get(&graph_guid)?;
                let (document, grammars, buildings) = lowered_program(payload);
                let center = w
                    .get::<GlobalTransform>(e)
                    .map(|g| g.translation())
                    .or_else(|| w.get::<Transform>(e).map(|t| t.translation.to_dvec3()))
                    .unwrap_or(DVec3::ZERO);
                Some(Job {
                    entity: e,
                    guid,
                    document,
                    grammars,
                    buildings,
                    center,
                    extent: vol.extent,
                    seed: vol.seed,
                })
            })
            .collect()
    };

    for mut job in jobs {
        // Fold the volume seed into every rule so distinct volumes differ.
        for layer in &mut job.document.layers {
            for rule in &mut layer.rules {
                rule.scatter.seed = rule.scatter.seed.wrapping_add(job.seed as u64);
            }
        }
        let region = Region::from_xz(
            job.center.x - job.extent.x,
            job.center.z - job.extent.y,
            job.center.x + job.extent.x,
            job.center.z + job.extent.y,
        );
        let cx = inf_pcg::GrammarContext {
            entity: Some(job.guid),
            center: job.center,
            extent: glam::DVec2::new(job.extent.x, job.extent.y),
            seed_offset: job.seed as u64,
        };
        // One closure per branch so the scatter and the grammar see the SAME
        // height provider — a grammar snapping to a different ground than the
        // scatter beside it would be invisible until somebody walked the level.
        let (mut raw, solids) = match &terrain {
            Some((data, o)) => {
                let data = data.clone();
                let o = *o;
                let provider = FnHeight::new(move |x, z| {
                    data.height_at(DVec2::new(x - o.x, z - o.z))
                        .map(|h| h + o.y)
                });
                let mut v = inf_pcg::evaluate(&job.document, &provider, region);
                let mut g = inf_pcg::evaluate_grammars(&job.grammars, &splines, &provider, &cx);
                g.extend(inf_pcg::evaluate_buildings(
                    &job.buildings,
                    &splines,
                    &provider,
                    &cx,
                ));
                v.extend(g.instances);
                (v, g.colliders)
            }
            None => {
                let provider = FnHeight::new(|_, _| Some(0.0));
                let mut v = inf_pcg::evaluate(&job.document, &provider, region);
                let mut g = inf_pcg::evaluate_grammars(&job.grammars, &splines, &provider, &cx);
                g.extend(inf_pcg::evaluate_buildings(
                    &job.buildings,
                    &splines,
                    &provider,
                    &cx,
                ));
                v.extend(g.instances);
                (v, g.colliders)
            }
        };
        let baked: Vec<ScatteredInstance> = raw
            .drain(..)
            .map(|i| ScatteredInstance {
                position: i.pos,
                rotation: i.rotation,
                scale: i.scale,
                kind: i.kind_index,
            })
            .collect();
        let solid: Vec<inf_ecs::components::ScatteredSolid> = solids
            .iter()
            .map(|s| inf_ecs::components::ScatteredSolid {
                center: s.center,
                half_extents: s.half_extents,
                rotation: s.rotation,
            })
            .collect();

        if let Some(mut vol) = world.world_mut().get_mut::<PcgVolume>(job.entity) {
            vol.evaluated = baked;
            vol.set_structures(solid);
        }
    }
}

/// Every entity's [`Spline`] resolved into **world space**, keyed by its stable
/// [`Guid`] — the P19.4 `grammar.spline` seam's fetch.
///
/// Unlike `mask.image`, this resolves identically in the editor, in PIE and in a
/// shipped build: a `Spline` is a persisted scene component in both codecs, so
/// every host has already built the entity by the time PCG evaluates. There is
/// no preview/shipping divergence here and therefore no cook advisory for it.
///
/// MIRROR: identical in `inf_player::level` and the editor's `commands::pcg`,
/// pinned by `inf-editor-core`'s `tests/grammar_span_mirror.rs`.
pub fn collect_spline_paths(
    world: &inf_ecs::EcsWorld,
) -> std::collections::HashMap<Uuid, inf_pcg::SplinePath> {
    let w = world.world();
    let mut out = std::collections::HashMap::new();
    for e in w.iter_entities() {
        let (Some(guid), Some(spline)) = (e.get::<inf_ecs::Guid>(), e.get::<Spline>()) else {
            continue;
        };
        let to_world = e
            .get::<GlobalTransform>()
            .map(|g| g.0)
            .unwrap_or(glam::DAffine3::IDENTITY);
        out.insert(
            guid.0,
            inf_pcg::SplinePath::from_local(
                spline.points.iter().map(|p| p.to_dvec3()),
                spline.closed,
                match spline.interp {
                    inf_ecs::components::SplineInterp::Linear => inf_pcg::SplineInterp::Linear,
                    inf_ecs::components::SplineInterp::CatmullRom => {
                        inf_pcg::SplineInterp::CatmullRom
                    }
                },
                to_world,
            ),
        );
    }
    out
}

/// Evaluate the **biome→PCG binding** for every terrain that carries a
/// [`BiomeSet`](inf_terrain::BiomeSet) (P19.3), refreshing its
/// `biome_population` from the graphs its painted biomes reference.
///
/// This is the terrain-level **sibling** of [`evaluate_pcg_volumes`], not a
/// replacement: a volume scatters one graph over a box an author placed, a
/// binding scatters many graphs over the regions an author *painted*. Both run
/// here, in that order, and neither reads the other's output.
///
/// Like the volume cache, `biome_population` is `#[serde(skip)]`, so the shipped
/// and PIE players recompute it on load exactly as the editor's evaluate command
/// does — which is what makes the two comparable and is the whole content of the
/// P19.3 parity gate.
///
/// Everything downstream of a GUID — which biomes dispatch, in what order, under
/// which feather — is [`inf_pcg::BiomeBinding::from_set`], shared verbatim with
/// the editor. All this function owns is the fetch.
pub fn evaluate_biome_bindings(
    world: &mut EcsWorld,
    biome_sets: &HashMap<Uuid, inf_terrain::BiomeSet>,
    pcgs: &HashMap<Uuid, PcgAssetPayload>,
) {
    if biome_sets.is_empty() || pcgs.is_empty() {
        return;
    }

    // Deterministic Guid-sorted terrain list, exactly as the volume pass sorts.
    let terrains: Vec<(inf_ecs::Entity, Uuid, DVec3)> = {
        let w = world.world();
        let mut v: Vec<(Uuid, inf_ecs::Entity)> = w
            .iter_entities()
            .filter_map(|e| e.get::<Guid>().map(|g| (g.0, e.id())))
            .collect();
        v.sort_by_key(|(g, _)| *g);
        v.into_iter()
            .filter_map(|(_, e)| {
                let t = w.get::<Terrain>(e)?;
                let set = t.biome_set?;
                if t.data.is_empty() {
                    return None;
                }
                let origin = w
                    .get::<GlobalTransform>(e)
                    .map(|g| g.translation())
                    .unwrap_or(DVec3::ZERO);
                Some((e, set, origin))
            })
            .collect()
    };

    for (entity, set_guid, origin) in terrains {
        let Some(set) = biome_sets.get(&set_guid) else {
            // A dangling biome-set reference is the cook's advisory, not a load
            // failure: the level is valid, its ids just resolve to nothing.
            continue;
        };
        let binding =
            inf_pcg::BiomeBinding::from_set(set, inf_pcg::DEFAULT_BIOME_FEATHER, |guid| {
                lowered_document(pcgs.get(&guid)?)
            });
        if binding.is_empty() {
            continue;
        }
        // The terrain's own authored extent, in world space, is the region.
        let data = {
            let w = world.world();
            match w.get::<Terrain>(entity) {
                Some(t) => t.data.clone(),
                None => continue,
            }
        };
        let Some((min, max)) = data.xz_bounds() else {
            continue;
        };
        let region = Region::from_xz(
            min.x + origin.x,
            min.y + origin.z,
            max.x + origin.x,
            max.y + origin.z,
        );
        let fields = inf_pcg::OffsetTerrain::new(&data, origin);
        let provider = FnHeight::new(|x, z| fields.height_at(x, z));
        let baked: Vec<ScatteredInstance> = binding
            .evaluate(&provider, &fields, region)
            .iter()
            .map(|i| ScatteredInstance {
                position: i.pos,
                rotation: i.rotation,
                scale: i.scale,
                kind: i.kind_index,
            })
            .collect();
        if let Some(mut t) = world.world_mut().get_mut::<Terrain>(entity) {
            t.biome_population = baked;
        }
    }
}

/// The runtime document a `.inf_pcg` payload evaluates as: the stored authored
/// graph re-lowered when there is one (parity with the editor, which lowers on
/// demand), else the stored lowered mirror (all a v1 payload carries).
///
/// Stated once so the volume pass and the biome pass cannot disagree about which
/// half of the payload is the source of truth.
fn lowered_document(payload: &PcgAssetPayload) -> Option<inf_pcg::PcgDocument> {
    Some(lowered_program(payload).0)
}

/// The full lowered program: the scatter document **and** the P19.4 grammar
/// passes.
///
/// A payload with no stored graph (a v1, document-only `.inf_pcg`) carries no
/// grammar — the passes live in the authored graph, which is the source of
/// truth, and the document mirror deliberately never grew a field for them.
fn lowered_program(
    payload: &PcgAssetPayload,
) -> (
    inf_pcg::PcgDocument,
    Vec<GrammarPass>,
    Vec<inf_pcg::BuildingPass>,
) {
    match payload.graph() {
        Some(g) => {
            let lowered = inf_pcg::lower_graph(&g, &inf_pcg::pcg_registry());
            (lowered.document, lowered.grammars, lowered.buildings)
        }
        None => (payload.document.clone(), Vec::new(), Vec::new()),
    }
}

/// Read every `.inf_biomes` payload in `dir` (non-recursive) **keyed by its asset
/// GUID** — the dev-dir half of the P19.3 binding's fetch. A plain
/// [`inf_asset::AssetPayload`], so it rides the shared sidecar-keyed loader
/// rather than growing a fourth copy of it.
pub fn load_biome_sets_by_guid_from_dir(dir: &Path) -> HashMap<Uuid, inf_terrain::BiomeSet> {
    load_anim_assets_by_guid_from_dir::<inf_terrain::BiomeSet>(dir, "inf_biomes")
}

/// Read every `.inf_pcg` payload in `dir` (non-recursive) **keyed by its asset
/// GUID** (from the sibling inf_asset `.toml` sidecar) — the map the player uses
/// to evaluate a level's [`PcgVolume`] scatter on load (dev-dir path). Files
/// without a readable sidecar/GUID or a decodable payload are skipped.
/// Deterministic (path-sorted) iteration.
pub fn load_pcg_payloads_by_guid_from_dir(dir: &Path) -> HashMap<Uuid, PcgAssetPayload> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("inf_pcg"))
            .collect(),
        Err(_) => return HashMap::new(),
    };
    files.sort();
    let mut out = HashMap::new();
    for p in files {
        let Ok(side) = inf_asset::AssetSidecar::load(&p) else {
            continue;
        };
        match std::fs::read(&p).map(|b| PcgAssetPayload::decode(&b)) {
            Ok(Ok(payload)) => {
                out.insert(side.guid.uuid(), payload);
            }
            _ => tracing::warn!("inf-player: bad .inf_pcg {}", p.display()),
        }
    }
    out
}

/// Read every P11 animation asset of extension `ext` in `dir` (non-recursive)
/// **keyed by its asset GUID** (from the sibling inf_asset `.toml` sidecar) — the
/// dev-dir twin of the pack anim loaders (P11.4). Files without a readable
/// sidecar/GUID or a decodable payload are skipped. Deterministic (path-sorted).
pub fn load_anim_assets_by_guid_from_dir<T: inf_asset::AssetPayload>(
    dir: &Path,
    ext: &str,
) -> HashMap<Uuid, T> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some(ext))
            .collect(),
        Err(_) => return HashMap::new(),
    };
    files.sort();
    let mut out = HashMap::new();
    for p in files {
        let Ok(side) = inf_asset::AssetSidecar::load(&p) else {
            continue;
        };
        match std::fs::read(&p).map(|b| inf_asset::decode::<T>(&b)) {
            Ok(Ok(payload)) => {
                out.insert(side.guid.uuid(), payload);
            }
            _ => tracing::warn!("inf-player: bad .{ext} {}", p.display()),
        }
    }
    out
}

/// The three dev-dir anim-asset maps (`.inf_skel` / `.inf_anim` / `.inf_sm`),
/// keyed by asset GUID (P11.4).
pub fn load_anim_assets_from_dir(
    dir: &Path,
) -> (
    HashMap<Uuid, SkeletonAsset>,
    HashMap<Uuid, AnimClipAsset>,
    HashMap<Uuid, StateMachineAsset>,
) {
    (
        load_anim_assets_by_guid_from_dir(dir, "inf_skel"),
        load_anim_assets_by_guid_from_dir(dir, "inf_anim"),
        load_anim_assets_by_guid_from_dir(dir, "inf_sm"),
    )
}

/// The dev-dir `.inf_audio` payload map keyed by asset GUID (P12.3) — the audio
/// mirror of [`load_anim_assets_from_dir`], seeded into the runtime sim so a scene
/// `AudioSource` resolves the same clip it does in the editor Simulate.
pub fn load_audio_assets_from_dir(dir: &Path) -> HashMap<Uuid, inf_audio::AudioAsset> {
    load_anim_assets_by_guid_from_dir(dir, "inf_audio")
}

/// The dev-dir `.inf_cloth` payload map keyed by asset GUID (P24.4) — the cloth
/// mirror of [`load_anim_assets_from_dir`], seeded into the runtime sim so a
/// `ClothSim` wearer simulates the same garment it does in the editor Simulate.
pub fn load_cloth_assets_from_dir(dir: &Path) -> HashMap<Uuid, inf_anim::ClothAsset> {
    load_anim_assets_by_guid_from_dir(dir, "inf_cloth")
}

/// The dev-dir `.inf_hair` payload map keyed by asset GUID (P24.4) - the hair
/// twin of [`load_cloth_assets_from_dir`].
pub fn load_hair_assets_from_dir(dir: &Path) -> HashMap<Uuid, inf_anim::HairAsset> {
    load_anim_assets_by_guid_from_dir(dir, "inf_hair")
}

/// Bind actor classes to controllable entities (the P8/P9 heuristic mirrored from
/// the editor's `samples::character_actors`): every entity carrying a
/// `CharacterController2D` — in `Guid` order — is ticked with the first discovered
/// actor class. Empty when there are no classes or no such entities.
///
/// Per-entity blueprint-class binding (so different actors run on different
/// entities) is the documented follow-up; until the binding is persisted in
/// `.inf_lvl`, one class drives every character (exactly what the sample needs).
pub fn resolve_actors(world: &EcsWorld, classes: &[BlueprintClass]) -> Vec<(Uuid, BlueprintClass)> {
    let Some(class) = classes.first() else {
        return Vec::new();
    };
    let w = world.world();
    let mut guids: Vec<Uuid> = w
        .iter_entities()
        .filter(|e| e.contains::<CharacterController2D>())
        .filter_map(|e| e.get::<Guid>().map(|g| g.0))
        .collect();
    guids.sort();
    guids.into_iter().map(|g| (g, class.clone())).collect()
}

/// Bind actors from the level's **persisted** `ActorClass` links (P9.5): each
/// entity carrying an [`ActorClass`] whose asset GUID resolves in `by_guid` is
/// ticked with that class, in entity-`Guid` order. Empty when the level carries
/// no bindings (or none resolve) — the caller then falls back to
/// [`resolve_actors`]. This is the real per-entity class binding the CC2D
/// heuristic stood in for.
pub fn resolve_bound_actors(
    world: &EcsWorld,
    by_guid: &HashMap<Uuid, BlueprintClass>,
) -> Vec<(Uuid, BlueprintClass)> {
    let w = world.world();
    let mut bound: Vec<(Uuid, BlueprintClass)> = w
        .iter_entities()
        .filter_map(|e| {
            let entity_guid = e.get::<Guid>()?.0;
            let actor_asset = e.get::<inf_ecs::components::ActorClass>()?.0;
            let class = by_guid.get(&actor_asset)?.clone();
            Some((entity_guid, class))
        })
        .collect();
    bound.sort_by_key(|(g, _)| *g);
    bound
}

/// Read and decode every `.inf_act` blueprint class in `dir` (non-recursive),
/// sorted by path for a deterministic order. Malformed files are logged + skipped.
pub fn load_actor_classes_from_dir(dir: &Path) -> Vec<BlueprintClass> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("inf_act"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    files.sort();
    let mut out = Vec::new();
    for p in files {
        match std::fs::read(&p) {
            Ok(bytes) => match serde_json::from_slice::<BlueprintClass>(&bytes) {
                Ok(c) => out.push(c),
                Err(e) => tracing::warn!("inf-player: bad .inf_act {}: {e}", p.display()),
            },
            Err(e) => tracing::warn!("inf-player: read {}: {e}", p.display()),
        }
    }
    out
}

/// Read every `.inf_act` in `dir` **keyed by its asset GUID** (from the sibling
/// inf_asset `.toml` sidecar) — the map the [`InfSceneWorldBuilder`] uses to
/// resolve a level's persisted `actor` bindings on the dev-dir path (P9.5). Files
/// without a readable sidecar/GUID are skipped (they can still bind via the CC2D
/// heuristic). Deterministic (path-sorted) iteration.
pub fn load_actor_classes_by_guid_from_dir(dir: &Path) -> HashMap<Uuid, BlueprintClass> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("inf_act"))
            .collect(),
        Err(_) => return HashMap::new(),
    };
    files.sort();
    let mut out = HashMap::new();
    for p in files {
        let Ok(side) = inf_asset::AssetSidecar::load(&p) else {
            continue;
        };
        match std::fs::read(&p).map(|b| serde_json::from_slice::<BlueprintClass>(&b)) {
            Ok(Ok(class)) => {
                out.insert(side.guid.uuid(), class);
            }
            _ => tracing::warn!("inf-player: bad .inf_act {}", p.display()),
        }
    }
    out
}

/// The minimal slice of the cook `manifest.toml` the player boots from. Unknown
/// fields are ignored (toml) so this stays decoupled from `inf_packager`'s full
/// `CookManifest`.
#[derive(Debug, Default, serde::Deserialize)]
struct BootManifest {
    #[serde(default)]
    project_name: String,
    #[serde(default)]
    packs: Vec<String>,
    #[serde(default)]
    root_level: Option<Uuid>,
}

/// A [`LevelSource`] backed by a cooked `content.inf_pack` (+ optional
/// `manifest.toml`). Opens the pack, resolves the root level GUID, and reads its
/// bytes; `.inf_act` classes are read straight out of the pack too.
pub struct PackLevelSource {
    /// `Arc` so a streaming store can hold the mapping open beyond the source's
    /// own lifetime (P16.3b2): a [`PackTileStore`](inf_terrain::PackTileStore)
    /// slices tiles straight out of it for as long as the world runs.
    reader: Arc<PackReader>,
    root_level: AssetId,
    label: String,
}

impl PackLevelSource {
    /// Open a pack given either the **directory** holding `content.inf_pack` +
    /// `manifest.toml`, or the **pack file** itself (its sibling `manifest.toml`
    /// is used when present).
    pub fn open(path: &Path) -> Result<Self, String> {
        let (default_pack, manifest_path) = if path.is_dir() {
            (path.join(PACK_FILE), path.join(MANIFEST_FILE))
        } else {
            (path.to_path_buf(), path.with_file_name(MANIFEST_FILE))
        };

        let manifest: Option<BootManifest> = std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|t| toml::from_str(&t).ok());

        // A manifest may name a non-default pack file (only meaningful for a dir).
        let pack_path = match (&manifest, path.is_dir()) {
            (Some(m), true) if !m.packs.is_empty() => path.join(&m.packs[0]),
            _ => default_pack,
        };

        let reader = PackReader::open(&pack_path)
            .map_err(|e| format!("open pack {}: {e}", pack_path.display()))?;

        // Root level: the manifest's, else the lowest-GUID level entry in the pack.
        let root_level = manifest
            .as_ref()
            .and_then(|m| m.root_level)
            .map(AssetId)
            .filter(|id| reader.contains(*id))
            .or_else(|| {
                reader
                    .index()
                    .find(|e| e.kind == AssetKind::Level)
                    .map(|e| e.guid)
            })
            .ok_or_else(|| "pack has no level to boot".to_string())?;

        let label = manifest
            .as_ref()
            .map(|m| m.project_name.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "pack".to_string());

        Ok(Self {
            reader: Arc::new(reader),
            root_level,
            label,
        })
    }

    /// Build a pack source from an already-loaded [`PackReader`] — the path the
    /// **web player** takes: it fetches the pack bytes over HTTP (no filesystem),
    /// parses them with [`PackReader::from_bytes`], and boots the lowest-GUID
    /// `Level` entry (there is no sibling `manifest.toml` to name a root, so v1
    /// uses that deterministic fallback — the same rule `open` falls back to).
    /// `label` names the world (the window title).
    pub fn from_reader(reader: PackReader, label: impl Into<String>) -> Result<Self, String> {
        let root_level = reader
            .index()
            .find(|e| e.kind == AssetKind::Level)
            .map(|e| e.guid)
            .ok_or_else(|| "pack has no level to boot".to_string())?;
        Ok(Self {
            reader: Arc::new(reader),
            root_level,
            label: label.into(),
        })
    }

    /// Decode every `.inf_act` blueprint class stored in the pack (GUID order).
    pub fn actor_classes(&self) -> Result<Vec<BlueprintClass>, String> {
        let mut out = Vec::new();
        for e in self.reader.index() {
            if e.kind != AssetKind::Blueprint {
                continue;
            }
            let bytes = self
                .reader
                .read(e.guid)
                .map_err(|err| format!("read actor {}: {err}", e.guid))?;
            let class = serde_json::from_slice::<BlueprintClass>(&bytes)
                .map_err(|err| format!("decode actor {}: {err}", e.guid))?;
            out.push(class);
        }
        Ok(out)
    }

    /// Every blueprint class in the pack **keyed by its asset GUID** — the map
    /// resolving a level's persisted `actor` bindings (P9.5).
    pub fn blueprint_classes_by_guid(&self) -> Result<HashMap<Uuid, BlueprintClass>, String> {
        let mut out = HashMap::new();
        for e in self.reader.index() {
            if e.kind != AssetKind::Blueprint {
                continue;
            }
            let bytes = self
                .reader
                .read(e.guid)
                .map_err(|err| format!("read actor {}: {err}", e.guid))?;
            let class = serde_json::from_slice::<BlueprintClass>(&bytes)
                .map_err(|err| format!("decode actor {}: {err}", e.guid))?;
            out.insert(e.guid.uuid(), class);
        }
        Ok(out)
    }

    /// Every `.inf_pcg` graph payload in the pack **keyed by its asset GUID** — the
    /// map the player uses to evaluate a level's [`PcgVolume`] scatter on load
    /// (P10.6). The cook joins a level→`PcgVolume.graph` edge into the dep closure,
    /// so the referenced `.inf_pcg` ships in the pack.
    pub fn pcg_payloads_by_guid(&self) -> Result<HashMap<Uuid, PcgAssetPayload>, String> {
        let mut out = HashMap::new();
        for e in self.reader.index() {
            if e.kind != AssetKind::Pcg {
                continue;
            }
            let bytes = self
                .reader
                .read(e.guid)
                .map_err(|err| format!("read pcg {}: {err}", e.guid))?;
            let payload = PcgAssetPayload::decode(&bytes)
                .map_err(|err| format!("decode pcg {}: {err}", e.guid))?;
            out.insert(e.guid.uuid(), payload);
        }
        Ok(out)
    }

    /// Every anim asset of `kind` in the pack **keyed by its asset GUID**, decoded
    /// as `T` (P11.4). The cook joins the level→anim-component edges (and the
    /// state-machine→clip edge) into the dep closure, so the referenced
    /// `.inf_skel` / `.inf_anim` / `.inf_sm` all ship in the pack.
    pub fn anim_assets_by_guid<T: inf_asset::AssetPayload>(
        &self,
        kind: AssetKind,
    ) -> Result<HashMap<Uuid, T>, String> {
        let mut out = HashMap::new();
        for e in self.reader.index() {
            if e.kind != kind {
                continue;
            }
            let bytes = self
                .reader
                .read(e.guid)
                .map_err(|err| format!("read anim {}: {err}", e.guid))?;
            let payload = inf_asset::decode::<T>(&bytes)
                .map_err(|err| format!("decode anim {}: {err}", e.guid))?;
            out.insert(e.guid.uuid(), payload);
        }
        Ok(out)
    }

    /// The three pack anim-asset maps (`.inf_skel` / `.inf_anim` / `.inf_sm`),
    /// keyed by asset GUID (P11.4).
    #[allow(clippy::type_complexity)]
    pub fn anim_assets(
        &self,
    ) -> Result<
        (
            HashMap<Uuid, SkeletonAsset>,
            HashMap<Uuid, AnimClipAsset>,
            HashMap<Uuid, StateMachineAsset>,
        ),
        String,
    > {
        Ok((
            self.anim_assets_by_guid(AssetKind::Skeleton)?,
            self.anim_assets_by_guid(AssetKind::AnimClip)?,
            self.anim_assets_by_guid(AssetKind::StateMachine)?,
        ))
    }

    /// Every `.inf_audio` payload in the pack keyed by asset GUID (P12.3) — the
    /// audio mirror of [`anim_assets`](Self::anim_assets). The cook's dep closure
    /// ships every clip a scene `AudioSource` references.
    pub fn audio_assets(&self) -> Result<HashMap<Uuid, inf_audio::AudioAsset>, String> {
        self.anim_assets_by_guid(AssetKind::Audio)
    }

    /// Every `.inf_cloth` payload in the pack keyed by asset GUID (P24.4) — the
    /// cloth mirror of [`anim_assets`](Self::anim_assets). The cook's dep closure
    /// ships every garment a scene `ClothSim` references.
    pub fn cloth_assets(&self) -> Result<HashMap<Uuid, inf_anim::ClothAsset>, String> {
        self.anim_assets_by_guid(AssetKind::Cloth)
    }

    /// Every `.inf_hair` payload in the pack keyed by asset GUID (P24.4) - the
    /// hair twin of [`cloth_assets`](Self::cloth_assets).
    pub fn hair_assets(&self) -> Result<HashMap<Uuid, inf_anim::HairAsset>, String> {
        self.anim_assets_by_guid(AssetKind::Hair)
    }

    /// Every `.inf_biomes` payload in the pack keyed by asset GUID (P19.3) — what
    /// the biome→PCG binding dispatches from.
    ///
    /// The cook closes the whole chain: level → `Terrain.biome_set` →
    /// `.inf_biomes` → each `BiomeDef.pcg_graph` → `.inf_pcg`. So a pack that
    /// holds the set also holds every graph its biomes name, and this map plus
    /// [`pcg_payloads_by_guid`](Self::pcg_payloads_by_guid) is everything the
    /// binding needs.
    pub fn biome_sets_by_guid(&self) -> Result<HashMap<Uuid, inf_terrain::BiomeSet>, String> {
        self.anim_assets_by_guid(AssetKind::BiomeSet)
    }

    /// The pack mapping, shared. A streaming store holds this open for the life of
    /// the world and slices tiles (P16.3b2) / partition cells (P16.5) out of it.
    pub fn reader(&self) -> &Arc<PackReader> {
        &self.reader
    }

    /// The root level's asset id — the key a partitioned level's derived
    /// `.inf_part` is computed from (P16.5).
    pub fn root_level(&self) -> AssetId {
        self.root_level
    }

    /// Resolve a `Terrain.asset` GUID to a **zero-copy** streaming source over the
    /// pack's `.inf_terrain` entry (P16.3b2).
    ///
    /// `None` when the pack has no such entry (a dangling ref — the cook already
    /// warns; the level's inline data stays authoritative). `Err` only for a
    /// corrupt payload, which must be loud rather than a silently flat world.
    pub fn terrain_source(&self, guid: Uuid) -> Result<Option<TerrainSource>, String> {
        let id = AssetId(guid);
        if !self.reader.contains(id) {
            return Ok(None);
        }
        let store = inf_terrain::PackTileStore::open(self.reader.clone(), id)?;
        let header = *store.header();
        Ok(Some(TerrainSource {
            store: Arc::new(store),
            tile_resolution: header.tile_resolution,
            meters_per_sample: header.meters_per_sample,
        }))
    }
}

/// Index every loose `.inf_terrain` in `dir` (non-recursive) **by its asset
/// GUID** (from the sibling inf_asset `.toml` sidecar) — the dev-dir twin of
/// [`PackLevelSource::terrain_source`] (P16.3b2). Deterministic (path-sorted);
/// files without a readable sidecar are skipped.
pub fn terrain_paths_by_guid_from_dir(dir: &Path) -> HashMap<Uuid, PathBuf> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("inf_terrain"))
            .collect(),
        Err(_) => return HashMap::new(),
    };
    files.sort();
    let mut out = HashMap::new();
    for p in files {
        match inf_asset::AssetSidecar::load(&p) {
            Ok(side) => {
                out.insert(side.guid.uuid(), p);
            }
            Err(_) => tracing::warn!("inf-player: .inf_terrain without a sidecar {}", p.display()),
        }
    }
    out
}

/// Open an **in-memory** `.inf_terrain` payload as a streaming source (P21.4) —
/// the PIE twin of [`terrain_source_from_file`], for bytes that arrived over the
/// wire rather than off a disk the player cannot see.
///
/// The dev-dir path already reads the whole payload once and slices tiles out of
/// it (`TerrainAssetReader<Vec<u8>>`), so a source over bytes is the same store
/// with the read already done. Nothing about residency, tiling or the hole mask
/// differs — which is the point: PIE must stream the terrain the shipped build
/// streams, not a second decoding of it.
pub fn terrain_source_from_bytes(bytes: Vec<u8>) -> Result<TerrainSource, String> {
    let store = inf_terrain::TerrainAssetReader::new(bytes)
        .map_err(|e| format!("decode .inf_terrain payload: {e}"))?;
    let header = *store.header();
    Ok(TerrainSource {
        store: Arc::new(store),
        tile_resolution: header.tile_resolution,
        meters_per_sample: header.meters_per_sample,
    })
}

/// Open a loose `.inf_terrain` as a streaming source (the `--level` dev-dir
/// path). The whole payload is read once; tiles are then sliced out of it.
pub fn terrain_source_from_file(path: &Path) -> Result<TerrainSource, String> {
    let store = inf_terrain::open_file_tile_store(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let header = *store.header();
    Ok(TerrainSource {
        store: Arc::new(store),
        tile_resolution: header.tile_resolution,
        meters_per_sample: header.meters_per_sample,
    })
}

impl LevelSource for PackLevelSource {
    fn level_bytes(&self) -> Result<Vec<u8>, String> {
        self.reader
            .read(self.root_level)
            .map_err(|e| format!("read level {}: {e}", self.root_level))
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    fn blueprint_by_guid(&self, guid: Uuid) -> Option<BlueprintClass> {
        let id = AssetId(guid);
        if !self.reader.contains(id) {
            return None;
        }
        let bytes = self.reader.read(id).ok()?;
        serde_json::from_slice::<BlueprintClass>(&bytes).ok()
    }
}

/// Load a world by piping a [`LevelSource`]'s bytes through a [`WorldBuilder`].
pub fn load(source: &dyn LevelSource, builder: &dyn WorldBuilder) -> Result<BuiltWorld, String> {
    let bytes = source.level_bytes()?;
    tracing::info!(
        "inf-player: loaded {} level byte(s) from '{}'",
        bytes.len(),
        source.label()
    );
    builder.build(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::components::Sprite;
    use inf_ecs::Vec2d;

    /// A hand-built world with one `CharacterController2D` entity proves the
    /// binding heuristic (the level format cannot carry a CC2D, so this is the
    /// only way to unit-test attachment).
    #[test]
    fn resolve_actors_binds_the_class_to_cc2d_entities() {
        let mut world = EcsWorld::new();
        let g = Uuid::from_u128(0x11);
        let e = world.spawn_with_guid(g, "Hero", None);
        world
            .world_mut()
            .entity_mut(e)
            .insert(CharacterController2D {
                max_slope_deg: 46.0,
                snap_to_ground: 0.3,
                offset: 0.02,
            });
        // A second entity with no controller must not bind.
        world.spawn_with_guid(Uuid::from_u128(0x22), "Prop", None);

        let class = BlueprintClass::new("act:x", "X");
        let bound = resolve_actors(&world, std::slice::from_ref(&class));
        assert_eq!(bound.len(), 1);
        assert_eq!(bound[0].0, g);
    }

    #[test]
    fn resolve_actors_is_empty_without_classes_or_controllers() {
        let mut world = EcsWorld::new();
        world.spawn_with_guid(Uuid::from_u128(1), "A", None);
        assert!(resolve_actors(&world, &[]).is_empty());
        let class = BlueprintClass::new("act:x", "X");
        // No CC2D entity → still empty even with a class available.
        assert!(resolve_actors(&world, std::slice::from_ref(&class)).is_empty());
    }

    /// `populate_world` instantiates components + hierarchy from decoded entities.
    #[test]
    fn populate_world_inserts_components_and_parenting() {
        let parent_guid = Uuid::from_u128(0xA0);
        let child_guid = Uuid::from_u128(0xA1);
        let mut parent = RuntimeEntity {
            guid: parent_guid,
            name: "Parent".into(),
            parent: None,
            transform: inf_ecs::components::Transform::from_translation(glam::DVec3::new(
                1.0, 2.0, 0.0,
            )),
            visible: true,
            mesh: None,
            material: None,
            light: None,
            camera: None,
            sprite: None,
            tilemap: None,
            nine_slice: None,
            text2d: None,
            light_2d: None,
            rigid_body_2d: None,
            collider_2d: None,
            character_controller_2d: None,
            rigid_body_3d: None,
            collider_3d: None,
            character_controller_3d: None,
            actor: None,
            terrain: None,
            pcg_volume: None,
            skeletal_mesh: None,
            anim_player: None,
            anim_state_machine: None,
            root_motion: None,
            attached_to: None,
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
        };
        parent.sprite = Some(Sprite {
            size: Vec2d::new(1.0, 1.0),
            ..Sprite::default()
        });
        let child = RuntimeEntity {
            guid: child_guid,
            name: "Child".into(),
            parent: Some(parent_guid),
            visible: false,
            ..parent.clone()
        };

        let mut world = populate_world(vec![parent, child]);
        world.propagate();

        let pe = world.entity_of(parent_guid).unwrap();
        let ce = world.entity_of(child_guid).unwrap();
        assert!(world.world().get::<Sprite>(pe).is_some());
        assert_eq!(world.parent_of(ce), Some(pe));
        // The child's own visibility toggle propagated.
        assert!(
            !world
                .world()
                .get::<inf_ecs::components::ComputedVisibility>(ce)
                .unwrap()
                .0
        );
    }
}
