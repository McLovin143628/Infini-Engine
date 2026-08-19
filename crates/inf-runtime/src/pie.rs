//! The PIE (play-in-editor) wire protocol between the editor and an
//! `inf-player` subprocess.
//!
//! Transport: the player's **stdin/stdout pipes** — the "local channel" of
//! the roadmap. In `--pie` mode stdout carries *only* protocol frames
//! (human logs go to stderr). Subprocess-first is the whole spike: a
//! panicking script kills the player process, never the editor.
//!
//! # Frame framing (versioned, little-endian) — P9.4
//!
//! Each frame is a small **versioned little-endian header** followed by a
//! bincode payload:
//!
//! ```text
//! ┌────────────┬──────────────┬────────────┬──────────────┐
//! │ magic u32  │ frame_ver u16 │ len u32     │ payload[len]  │
//! │ 0x0050_4945│ PIE_FRAME_VER │ (bincode)   │  bincode      │
//! └────────────┴──────────────┴────────────┴──────────────┘
//! ```
//!
//! The magic + version self-describe the stream so a desynced or mismatched
//! peer fails cleanly (a wrong first byte is an error, not garbage bincode).
//! A cleanly closed pipe (peer exit) surfaces as `UnexpectedEof` on the first
//! header read, which the reader loops treat as end-of-stream.
//!
//! # Content handoff (P9.4)
//!
//! Spike D handed over a toy [`CookedSnapshot`]. P9.4 adds [`ScenePayload`]:
//! the editor streams the **real** live scene as v3 `.inf_lvl` bytes plus the
//! bound blueprint classes as `(guid, json)`, and the player builds the world
//! exactly like the shipping pack path (`InfSceneWorldBuilder::with_bindings`)
//! — so previewing never diverges from shipping.

use std::io::{self, Read, Write};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::sim::ActorState;
use crate::snapshot::CookedSnapshot;

/// Protocol version, checked at handshake. Bumped to 2 in P9.4 (the real
/// [`ScenePayload`] content handoff + Step/Eject/SetViewport control frames +
/// Window/State reports on top of the Spike D v1 surface).
pub const PIE_PROTOCOL_VERSION: u32 = 2;

/// Frame-header magic (little-endian `b"PIE\0"`).
pub const PIE_FRAME_MAGIC: u32 = 0x0050_4945;

/// Frame-header version (the framing itself, distinct from the protocol
/// version carried in the `Ready` handshake).
pub const PIE_FRAME_VERSION: u16 = 1;

/// Schema version of a [`ScenePayload`] — **the PIE envelope's own version**, not
/// a scene or asset schema. It versions this wire struct alone; the `.inf_lvl`
/// bytes inside carry the scene schema version independently, and each attached
/// asset payload carries its own. Bumping it is therefore *not* a content schema
/// bump: nothing on disk changes, and no golden is re-blessed.
///
/// * v1 — level bytes + bound blueprint classes.
/// * v2 — appended `pcgs`: the `.inf_pcg` graph payloads a v4 level's
///   [`PcgVolume`]s reference, so the PIE player evaluates scatter exactly like
///   the shipping pack path.
/// * v3 — appended the P11 animation assets a v5 level's components reference:
///   `skeletons` (`.inf_skel`), `clips` (`.inf_anim`) and `machines` (`.inf_sm`),
///   so the PIE player resolves state machines / root-motion clips exactly like
///   the shipping pack path (PIE == shipping for animation).
/// * v4 — appended `biome_sets`: the `.inf_biomes` payloads a v16 level's
///   `Terrain.biome_set` refs, so the PIE player runs the biome→PCG binding
///   exactly like the shipping pack path.
/// * **v6** — `fractures` (P22.3): the DERIVED `.inf_fracture` chunk sets, so a
///   PIE session breaks the same walls into the same pieces the shipped build
///   does. The first payload entry that is *computed* rather than *resolved*.
/// * v5 — appended `voxels` **and** `terrains` (P21.4), the two byte sources the
///   PIE player had no route to at all.
///
///   `voxels` are the `.inf_voxel` payloads a v19 level's `VoxelVolume.asset`
///   refs name. Before this rung a carved cave answered `terrain.height_at` with
///   the seam's `0.0` in preview and with the cave floor in the shipped build,
///   and the PIE == shipping gate compared **two empty maps agreeing**, which is
///   not a comparison.
///
///   `terrains` are the `.inf_terrain` payloads a v10 level's `Terrain.asset`
///   refs name — the **P16.3b2 deferral**, closed here because P21.2's hole mask
///   only persists on an asset-backed terrain (scene schema v19 pins
///   `TerrainTileFrozenV3`, which has no hole rows), so "a level with carved cave
///   mouths" and "a level PIE can preview" were mutually exclusive until now.
///   `serialize::strip_streamed_terrain` still blanks the inline working set, so
///   PIE streams from the asset exactly as the shipped build does.
/// * **v7** — `meshes` (P24.1): the `.inf_mesh` payloads a level's
///   `SkeletalMesh.mesh` refs name. Skeletons, clips and machines have ridden
///   this envelope since v3, so a PIE session has always *simulated* its
///   characters correctly — and it drew every one of them as a placeholder cube,
///   because the mesh bytes the skinned pass needs were the one part that never
///   crossed. See [`ScenePayload::meshes`].
///
/// **The only payload bump of P24.1**, and deliberately audited against the rest
/// of the phase before freezing.
///
/// # That audit's prediction about P24.4 was wrong, and v8 was owed — P26.3b pays it
///
/// It read: *"P24.4's cloth is sim state over a garment that is itself an
/// `.inf_mesh`, so it needs a GUID added to the **collector** and nothing added
/// to the wire; hair guides that ride a mesh or a skeleton are the same. A
/// distinct `.inf_hair` asset kind — which P24.4 has not chosen — would be a v8,
/// and would be the honest reason for one."*
///
/// P24.4 **did** choose distinct `.inf_cloth` and `.inf_hair` kinds (pack kind
/// codes 22 and 23), because a garment is self-contained for the sim and the draw
/// — rest positions, inverse masses, precomputed edge and hinge lists — and none
/// of that is in the mesh. So by that note's own words the honest reason for a v8
/// existed, and P24.4's schema budget was already spent on scene v21: **a
/// character previewed in PIE wore nothing** while the same character in Simulate
/// and in a cooked pack wore both. The gap was measured rather than described —
/// `phase24_gate::the_pie_payload_carries_no_garment_and_that_is_measured` read
/// this struct's own declaration and was written to fail the day it grew
/// `cloths` / `hairs`.
///
/// **It fired, and this is the batch that fired it.** The arm is retired and
/// replaced by the positive ones (`phase26_gate`).
///
/// * **v8** — P26.3b, **one bump carrying four slots** (the P24.3 precedent, at
///   the payload instead of the scene):
///   * [`cloths`](ScenePayload::cloths) / [`hairs`](ScenePayload::hairs) — the
///     `.inf_cloth` / `.inf_hair` payloads a v21 level's `ClothSim.asset` /
///     `HairGuides.asset` name. The debt above.
///   * [`materials`](ScenePayload::materials) — the **derived** material records
///     (`inf_asset::DerivedMaterial`) a v22 level's `Material.asset` bindings
///     imply, keyed by `inf_asset::derived_material_id`, exactly as `fractures`
///     are keyed by `inf_mesh::derived_fracture_id` and for the same reason: one
///     lookup rule for the pack path and the payload path rather than two.
///   * [`textures`](ScenePayload::textures) — the `.inf_tex` v2 container bytes
///     those records reference, which the player pages through
///     `inf_vt::TiledTextureReader`.
///
///   Four slots and one bump because a phase gets one payload move and all four
///   are the same wiring: without `materials` the level names materials nothing
///   can resolve, and without `textures` the derived records name textures with
///   no bytes behind them — the "two empty maps agreeing" hazard, twice.
///
/// * **v9** — P29.3, and it carries **no new slot at all**. The envelope's
///   version is the *build contract* between the two processes, and this bump
///   records that `level_bytes` is now a scene **v23** payload (the movement
///   component).
///
///   That is a real refusal rather than bookkeeping. Without it, a v23 editor
///   handing a stale v8 player a level would pass `check_version` -- the
///   envelope is unchanged, after all -- and fail several layers deeper, in
///   `inf_scene::decode`, with a message about a scene schema. The player's own
///   refusal ("rebuild both from the same commit") is the one that names the
///   actual fix, and it only fires if the envelope moves when the contract does.
///   A version with no field behind it is exactly what an envelope is for.
///
/// * **v10** — Wave G, and the same shape of bump as v9 for the same reason:
///   **no new slot**, recording that `level_bytes` is now a scene **v24**
///   payload (the geo-anchor and `TerrainLayer::material`).
///
///   The anchor needs no envelope field of its own because it rides inside
///   `level_bytes` — the scene file record carries it, and this envelope's whole
///   job is to say which scene contract those bytes obey.
pub const SCENE_PAYLOAD_VERSION: u32 = 10;

/// Upper bound on a single frame; anything larger means a desynced or
/// corrupt stream and is treated as an error rather than an allocation. A
/// live scene payload (level bytes + blueprint JSON) fits comfortably.
pub const MAX_FRAME_LEN: usize = 256 * 1024 * 1024;

/// The real content the editor streams to the player: the live scene as v3
/// `.inf_lvl` bytes plus the set of bound blueprint classes. The player builds
/// its world from these exactly like the cooked-pack path, so PIE == shipping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenePayload {
    /// Envelope schema version ([`SCENE_PAYLOAD_VERSION`]).
    pub schema_version: u32,
    /// Human label for logs / the window title.
    pub label: String,
    /// The v3 `.inf_lvl` bincode payload of the *live* (unsaved-included) doc.
    pub level_bytes: Vec<u8>,
    /// Bound blueprint classes: `(asset guid, .inf_act JSON bytes)`. The guid
    /// keys the level's persisted `ActorClass` bindings.
    pub classes: Vec<(Uuid, Vec<u8>)>,
    /// Referenced PCG graphs: `(asset guid, .inf_pcg bincode bytes)`. The guid
    /// keys a v4 level's `PcgVolume.graph` refs; the player evaluates scatter from
    /// these on load (schema v2). `#[serde(default)]` so a v1 payload decodes.
    #[serde(default)]
    pub pcgs: Vec<(Uuid, Vec<u8>)>,
    /// Referenced skeletons: `(asset guid, .inf_skel bincode bytes)` — keyed by a
    /// v5 level's `SkeletalMesh.skeleton` / clip-skeleton refs (schema v3).
    #[serde(default)]
    pub skeletons: Vec<(Uuid, Vec<u8>)>,
    /// Referenced clips: `(asset guid, .inf_anim bincode bytes)` — keyed by a v5
    /// level's `AnimPlayer.clip` refs (schema v3).
    #[serde(default)]
    pub clips: Vec<(Uuid, Vec<u8>)>,
    /// Referenced state machines: `(asset guid, .inf_sm bincode bytes)` — keyed by
    /// a v5 level's `AnimStateMachine.sm` refs (schema v3).
    #[serde(default)]
    pub machines: Vec<(Uuid, Vec<u8>)>,
    /// Referenced biome sets: `(asset guid, .inf_biomes bincode bytes)` — keyed by
    /// a v16 level's `Terrain.biome_set` refs (schema v4). The player dispatches
    /// each painted biome's `pcg_graph` from these against the `pcgs` above, so a
    /// PIE session grows the same world a cooked pack does.
    #[serde(default)]
    pub biome_sets: Vec<(Uuid, Vec<u8>)>,
    /// Fixed update rate (Hz) the player ticks at.
    pub tick_hz: u32,
    /// Open a real window (`true`, the embedded / new-window PIE path) vs run
    /// headless + step-driven (`false`, the CI / determinism path).
    pub windowed: bool,

    // ── APPEND-ONLY TAIL ────────────────────────────────────────────────────
    //
    // **New fields go HERE, after `windowed`, and nowhere else.** bincode is
    // positional: `#[serde(default)]` buys nothing, and a field inserted in the
    // middle re-interprets every byte after it. v5 was first written with
    // `voxels` and `terrains` between `biome_sets` and `tick_hz`, where two empty
    // `Vec`s decode in a stale player as a perfectly valid `u32` + `bool` — so
    // Play started, reported success, and silently ran at a `tick_hz` of zero. A
    // crash would have been the kind outcome.
    //
    // At the tail a stale reader gets every field it knows about correct and
    // stops; `check_version` below is what turns "stops" into a refusal rather
    // than a silent half-load. See `SCENE_PAYLOAD_VERSION`.
    /// Referenced voxel volumes: `(asset guid, .inf_voxel bincode bytes)` — keyed
    /// by a v19 level's `VoxelVolume.asset` refs (schema v5, P21.4).
    ///
    /// The player resolves these through the **same** `resolve_voxel_volumes` the
    /// pack path uses, so a Blueprint standing on a cave floor reads the same
    /// metre in preview and in the shipped build. Without them the sim's voxel map
    /// is empty and every combined ground query over a hole falls back to `0.0`.
    #[serde(default)]
    pub voxels: Vec<(Uuid, Vec<u8>)>,
    /// Referenced streamed terrains: `(asset guid, .inf_terrain bytes)` — keyed by
    /// a v10 level's `Terrain.asset` refs (schema v5, P21.4 / the P16.3b2
    /// deferral).
    ///
    /// The level bytes above carry a **blanked** working set for any asset-backed
    /// terrain (`strip_streamed_terrain`), so without these a PIE session over a
    /// streamed terrain runs on no ground at all — and, since P21.2, with none of
    /// the hole mask that makes a cave mouth a cave mouth.
    #[serde(default)]
    pub terrains: Vec<(Uuid, Vec<u8>)>,
    /// `.inf_fracture` payloads (P22.3), keyed by the **derived** fracture id —
    /// `inf_mesh::derived_fracture_id(mesh)` (not linked: `inf-runtime` does not
    /// depend on `inf-mesh`, and this crate carries bytes rather than the types
    /// they decode to), exactly the id the cooked pack stores them under, so the
    /// player looks a fracture up the same way on both paths and there is one
    /// lookup rule rather than two.
    ///
    /// A `.inf_fracture` is *derived*, not authored: it does not exist in the
    /// editor's content root at all, so unlike every other entry above these
    /// bytes are produced at payload time by running the **same**
    /// `inf_mesh::fracture_mesh` the cook runs, with the same
    /// `Destructible::{fracture_seed, chunk_count}` parameters. That is what makes
    /// PIE == shipping here: the same deterministic function over the same mesh
    /// with the same seed cannot produce two different chunk sets.
    ///
    /// Empty when nothing in the level is destructible — **which is also what a
    /// caller that does not resolve meshes produces**, and the two are
    /// indistinguishable from here.
    ///
    /// That **was** a live hazard, and P22.4 closed it. The obligation this
    /// paragraph used to record — *"P22.4 owes a `phase22_gate` whose PIE arm
    /// asserts this vector is non-empty before it compares anything"* — is now
    /// discharged: `runtime/inf-player/tests/phase22_gate.rs`'s
    /// `playground_payload` asserts `payload.fractures.len() == 2` (the flagship
    /// sample's two destructible meshes) **before** any comparison runs, so a
    /// payload that carried no chunk sets fails there with a message about
    /// unbreakable buildings rather than passing as two intact worlds agreeing.
    ///
    /// The P21.4 lesson stands, and the shape of the guard is what it teaches: an
    /// exact expected count, taken from the fixture, asserted at the seam that
    /// produces the payload. A `!is_empty()` would still pass on a level whose
    /// second destructible mesh had silently stopped resolving.
    #[serde(default)]
    pub fractures: Vec<(Uuid, Vec<u8>)>,
    /// Referenced skeletal meshes: `(asset guid, .inf_mesh bincode bytes)` — keyed
    /// by a level's `SkeletalMesh.mesh` refs (schema v7, P24.1).
    ///
    /// The one part of a character that never crossed this wire. `skeletons`,
    /// `clips` and `machines` have ridden it since v3, so a PIE session stepped
    /// its characters exactly like the shipped build and then drew every one of
    /// them as a **placeholder cube** — the `SkinnedRegistry` the windowed PIE
    /// player was handed had no content at all, so `resolve_skinned` missed on
    /// the mesh and the projector fell through to its slate placeholder.
    ///
    /// Keyed by ASSET, deduplicated, and resolved through the caller's `.inf_mesh`
    /// door — the same one P22.3's fracture derivation reads. Unlike `fractures`
    /// these bytes are carried, not computed: the skinned pass's bind-space
    /// rebuild (`skinned_mesh_data`) is a MIRROR pair, so running it here would
    /// put a third copy of it in the tree.
    ///
    /// Empty on a level with no `SkeletalMesh`, **which is also what a caller that
    /// does not resolve meshes produces** — the P21.4 "two empty maps agreeing"
    /// hazard, and the reason `pie.rs`'s payload pin asserts an exact expected
    /// count taken from its fixture before anything is compared.
    #[serde(default)]
    pub meshes: Vec<(Uuid, Vec<u8>)>,
    /// Referenced garments: `(asset guid, .inf_cloth bincode bytes)` — keyed by a
    /// v21 level's `ClothSim.asset` refs (schema v8, P26.3b).
    ///
    /// **The debt this envelope owed since P24.4.** A character previewed in PIE
    /// wore nothing while the same character in the editor's Simulate and in a
    /// cooked pack wore both, because `build_world_from_payload` called
    /// `with_anim_assets` and neither cloth door — and the reason it called
    /// neither is that there was nothing here to hand them.
    ///
    /// Carried, not computed: a `.inf_cloth` is authored (the Model Editor's
    /// cloth door tunes stiffness and pins hems), so re-deriving it here would
    /// throw the tuning away and give PIE a different garment from the shipped
    /// build — the exact divergence this file exists to prevent.
    #[serde(default)]
    pub cloths: Vec<(Uuid, Vec<u8>)>,
    /// Referenced hairstyles: `(asset guid, .inf_hair bincode bytes)` — the
    /// [`cloths`](Self::cloths) twin, keyed by `HairGuides.asset`.
    #[serde(default)]
    pub hairs: Vec<(Uuid, Vec<u8>)>,
    /// **Derived** material records: `(derived guid, .inf_matd bincode bytes)`,
    /// keyed by `inf_asset::derived_material_id` of the `.inf_mat` — exactly the
    /// id a cooked pack stores them under, so the player looks a material up the
    /// same way on both paths and there is one lookup rule rather than two
    /// (schema v8, P26.3b · scene v22's `Material.asset`).
    ///
    /// **Computed here rather than resolved**, like [`fractures`](Self::fractures)
    /// and for a sharper reason: the player *cannot decode a `.inf_mat`*.
    /// `MaterialAsset` lives in `inf-material`, which owns the `image` decoders
    /// and the naga-validating material compiler, and the P26.2 dependency
    /// inversion exists to keep both out of a shipped build. So the editor runs
    /// the **same** `inf_material::derive_material` the cook runs and ships the
    /// answer. One deterministic function over the same bytes cannot give the
    /// preview a different surface from the shipped one.
    ///
    /// Empty when nothing in the level binds a material — **which is also what a
    /// caller that does not resolve materials produces**, the P21.4 "two empty
    /// maps agreeing" hazard, and the reason `phase26_gate` asserts an exact
    /// expected count before it compares anything.
    #[serde(default)]
    pub materials: Vec<(Uuid, Vec<u8>)>,
    /// Referenced textures: `(asset guid, .inf_tex v2 container bytes)` — the
    /// payloads the [`materials`](Self::materials) above name (schema v8,
    /// P26.3b).
    ///
    /// **Raw container bytes, not a decoded image.** The player reads tiles out
    /// of them through `inf_vt::TiledTextureReader`, which is the same door the
    /// pack path takes one indirection later (a pack-backed host hands the
    /// registry a slice of its mmap; a PIE host hands it a slice of this
    /// vector). Nothing decodes a whole texture on either path.
    ///
    /// Keyed by the TEXTURE asset, deduplicated: two materials naming one map
    /// carry it once, and the registry dedupes again by GUID so the atlas holds
    /// one copy.
    #[serde(default)]
    pub textures: Vec<(Uuid, Vec<u8>)>,
}

impl ScenePayload {
    /// Build a payload from encoded scene bytes + bound classes.
    pub fn new(
        label: impl Into<String>,
        level_bytes: Vec<u8>,
        classes: Vec<(Uuid, Vec<u8>)>,
        tick_hz: u32,
        windowed: bool,
    ) -> Self {
        Self {
            schema_version: SCENE_PAYLOAD_VERSION,
            label: label.into(),
            level_bytes,
            classes,
            pcgs: Vec::new(),
            skeletons: Vec::new(),
            clips: Vec::new(),
            machines: Vec::new(),
            biome_sets: Vec::new(),
            tick_hz,
            windowed,
            voxels: Vec::new(),
            terrains: Vec::new(),
            fractures: Vec::new(),
            meshes: Vec::new(),
            cloths: Vec::new(),
            hairs: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
        }
    }

    /// Refuse a payload this build cannot read (P21.4).
    ///
    /// Nothing read `schema_version` before v5, which is half of why the
    /// mid-struct insertion the tail comment describes was survivable long enough
    /// to ship: a stale player decoded nonsense and reported `Loaded`. Checked at
    /// the one place every consumer goes through
    /// (`inf_player::build_world_from_payload`), so a version the reader does not
    /// understand is a **loud refusal** rather than a world quietly missing its
    /// caves.
    ///
    /// Exact equality, not a floor: the envelope is positional, so an older
    /// payload is as unreadable as a newer one, and the editor and the player it
    /// spawns are built together.
    pub fn check_version(&self) -> Result<(), String> {
        if self.schema_version == SCENE_PAYLOAD_VERSION {
            return Ok(());
        }
        Err(format!(
            "PIE scene payload schema v{} but this build speaks v{SCENE_PAYLOAD_VERSION} — \
             the editor and the player must be built together (the envelope is \
             positional bincode, so a mismatch decodes as garbage rather than failing)",
            self.schema_version
        ))
    }

    /// Attach the referenced `.inf_pcg` graph payloads (`(asset guid, bytes)`).
    /// Builder-style so [`Self::new`]'s signature stays stable.
    pub fn with_pcgs(mut self, pcgs: Vec<(Uuid, Vec<u8>)>) -> Self {
        self.pcgs = pcgs;
        self
    }

    /// Attach the referenced `.inf_biomes` payloads (`(asset guid, bytes)`) a
    /// level's `Terrain.biome_set` refs name (P19.3). Builder-style, exactly like
    /// [`with_pcgs`](Self::with_pcgs) — the binding needs both.
    pub fn with_biome_sets(mut self, biome_sets: Vec<(Uuid, Vec<u8>)>) -> Self {
        self.biome_sets = biome_sets;
        self
    }

    /// Attach the referenced `.inf_voxel` payloads (`(asset guid, bytes)`) a
    /// level's `VoxelVolume.asset` refs name (P21.4). Builder-style, exactly like
    /// [`with_pcgs`](Self::with_pcgs).
    pub fn with_voxels(mut self, voxels: Vec<(Uuid, Vec<u8>)>) -> Self {
        self.voxels = voxels;
        self
    }

    /// Attach the derived `.inf_fracture` payloads (P22.3), keyed by the derived
    /// fracture id (`inf_mesh::derived_fracture_id`) — the same id the cooked
    /// pack stores them under. Builder-style, exactly like
    /// [`with_pcgs`](Self::with_pcgs).
    pub fn with_fractures(mut self, fractures: Vec<(Uuid, Vec<u8>)>) -> Self {
        self.fractures = fractures;
        self
    }

    /// Attach the referenced `.inf_terrain` payloads (`(asset guid, bytes)`) a
    /// level's `Terrain.asset` refs name (P21.4). Builder-style, exactly like
    /// [`with_pcgs`](Self::with_pcgs).
    pub fn with_terrains(mut self, terrains: Vec<(Uuid, Vec<u8>)>) -> Self {
        self.terrains = terrains;
        self
    }

    /// Attach the referenced `.inf_mesh` payloads (`(asset guid, bytes)`) a
    /// level's `SkeletalMesh.mesh` refs name (P24.1). Builder-style, exactly like
    /// [`with_pcgs`](Self::with_pcgs).
    pub fn with_meshes(mut self, meshes: Vec<(Uuid, Vec<u8>)>) -> Self {
        self.meshes = meshes;
        self
    }

    /// Attach the referenced `.inf_cloth` / `.inf_hair` payloads (P26.3b) a v21
    /// level's `ClothSim.asset` / `HairGuides.asset` refs name. Builder-style,
    /// exactly like [`with_pcgs`](Self::with_pcgs).
    ///
    /// **The two travel together on purpose.** A wearer usually has both, they
    /// are resolved by one walk of the document, and the P24.4 mutation matrix
    /// found that severing one alone still left the anti-vacuity arm passing on
    /// the other. One door makes "the garments crossed" one fact.
    pub fn with_garments(
        mut self,
        cloths: Vec<(Uuid, Vec<u8>)>,
        hairs: Vec<(Uuid, Vec<u8>)>,
    ) -> Self {
        self.cloths = cloths;
        self.hairs = hairs;
        self
    }

    /// Attach the **derived** material records and the `.inf_tex` containers they
    /// name (P26.3b). Builder-style, exactly like [`with_pcgs`](Self::with_pcgs).
    ///
    /// **The two travel together for the reason the doc on
    /// [`materials`](Self::materials) states**: a record naming textures with no
    /// bytes behind them previews as an untextured surface while the shipped
    /// build is textured, and the comparison between two such worlds passes.
    pub fn with_materials(
        mut self,
        materials: Vec<(Uuid, Vec<u8>)>,
        textures: Vec<(Uuid, Vec<u8>)>,
    ) -> Self {
        self.materials = materials;
        self.textures = textures;
        self
    }

    /// Attach the referenced P11 animation assets (`(asset guid, bytes)` for each
    /// of `.inf_skel` / `.inf_anim` / `.inf_sm`). Builder-style so [`Self::new`]'s
    /// signature stays stable.
    pub fn with_anim_assets(
        mut self,
        skeletons: Vec<(Uuid, Vec<u8>)>,
        clips: Vec<(Uuid, Vec<u8>)>,
        machines: Vec<(Uuid, Vec<u8>)>,
    ) -> Self {
        self.skeletons = skeletons;
        self.clips = clips;
        self.machines = machines;
        self
    }
}

/// A viewport rectangle forwarded to an embedded player (physical pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewportRectMsg {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditorToPlayer {
    /// Hand over the Spike-D toy snapshot (kept for the determinism smoke +
    /// the crash/pause/reparent drills that don't need real content).
    Load(CookedSnapshot),
    /// Hand over the **real** live scene (P9.4). Sent once after `Ready`.
    ///
    /// **Boxed** (P22.3). `ScenePayload` grew past clippy's `large_enum_variant`
    /// threshold when v6 appended `fractures`, and the lint is right on the
    /// merits: every `EditorToPlayer` value would otherwise cost the largest
    /// variant, so the cheap `Pause`/`Resume`/`Step` messages this protocol sends
    /// every frame would each carry a scene payload's worth of stack. The box is
    /// **wire-invisible** — serde encodes `Box<T>` exactly as `T` — so the frame
    /// bytes, the protocol version and every golden are untouched.
    LoadScene(Box<ScenePayload>),
    Pause,
    Resume,
    /// Advance exactly `count` fixed steps (works while paused) — the
    /// deterministic step control the PIE==shipping test drives.
    Step {
        count: u32,
    },
    /// Graceful shutdown; the player answers `Stopped` and exits 0.
    Stop,
    /// Release input possession back to the editor (v1: stops the player
    /// possessing input; camera possession is a documented follow-up). The
    /// player keeps running until `Stop`.
    Eject,
    /// Forward a viewport rect change to an embedded player.
    SetViewport(ViewportRectMsg),
    /// Test/QA hook: make the player panic on its main loop — the
    /// crash-isolation drill.
    InjectPanic,
}

/// A running/paused status report the player pushes to the editor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlayerState {
    pub running: bool,
    pub paused: bool,
    pub frame: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlayerToEditor {
    /// First message after spawn; the editor refuses a version mismatch.
    Ready {
        protocol: u32,
    },
    Loaded {
        level: String,
        actor_count: usize,
    },
    /// The player created its window; `handle` is the native handle (HWND as
    /// `isize` on Windows, `0` where not applicable) the editor reparents into
    /// the viewport slot for embedded PIE.
    Window {
        handle: i64,
    },
    Frame {
        frame: u64,
        state_hash: u64,
        actors: Vec<ActorState>,
    },
    /// A running/paused status report (frame count, last error).
    State(PlayerState),
    Paused,
    Resumed,
    Stopped,
    Ejected,
    /// A recoverable error the editor surfaces (a fatal one is a process exit).
    Error {
        message: String,
    },
}

/// Write one length-prefixed, versioned bincode frame.
pub fn write_msg<T: Serialize>(writer: &mut impl Write, msg: &T) -> io::Result<()> {
    let bytes = bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    // **Refuse an oversized frame; never truncate its length** (P21.4).
    //
    // `bytes.len() as u32` wrapped silently past 4 GiB, and past `MAX_FRAME_LEN`
    // it produced a header the reader rejects *after* the writer has committed to
    // sending the body — a desynced stream either way. Since P21.4 a scene payload
    // carries whole `.inf_voxel` and `.inf_terrain` assets **uncompressed**, so
    // frame size is now a function of the author's content rather than of the
    // protocol, and this is a bound a real level can reach. Compressing them (or
    // streaming them out of band) is the open question the ROADMAP carries.
    let len = u32::try_from(bytes.len())
        .ok()
        .filter(|_| bytes.len() <= MAX_FRAME_LEN);
    let Some(len) = len else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "PIE frame of {} bytes exceeds the {MAX_FRAME_LEN}-byte cap",
                bytes.len()
            ),
        ));
    };
    writer.write_all(&PIE_FRAME_MAGIC.to_le_bytes())?;
    writer.write_all(&PIE_FRAME_VERSION.to_le_bytes())?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.flush()
}

/// Read one versioned bincode frame. `Err(UnexpectedEof)` on a cleanly closed
/// pipe (the peer exited) — surfaced from the very first header read so reader
/// loops end cleanly.
pub fn read_msg<T: DeserializeOwned>(reader: &mut impl Read) -> io::Result<T> {
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    if u32::from_le_bytes(magic_bytes) != PIE_FRAME_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad PIE frame magic (desynced stream)",
        ));
    }
    let mut ver_bytes = [0u8; 2];
    reader.read_exact(&mut ver_bytes)?;
    let ver = u16::from_le_bytes(ver_bytes);
    if ver != PIE_FRAME_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("PIE frame version {ver} unsupported (host speaks {PIE_FRAME_VERSION})"),
        ));
    }
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds MAX_FRAME_LEN"),
        ));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let (msg, _): (T, usize) = bincode::serde::decode_from_slice(&buf, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn framing_round_trip() {
        let messages = vec![
            EditorToPlayer::Load(CookedSnapshot::demo()),
            EditorToPlayer::LoadScene(Box::new(ScenePayload::new(
                "level",
                vec![1, 2, 3, 4],
                vec![(Uuid::from_u128(0xAC), vec![b'{', b'}'])],
                60,
                false,
            ))),
            EditorToPlayer::Pause,
            EditorToPlayer::Resume,
            EditorToPlayer::Step { count: 3 },
            EditorToPlayer::Eject,
            EditorToPlayer::SetViewport(ViewportRectMsg {
                x: 1,
                y: 2,
                width: 320,
                height: 240,
            }),
            EditorToPlayer::Stop,
            EditorToPlayer::InjectPanic,
        ];
        let mut wire = Vec::new();
        for msg in &messages {
            write_msg(&mut wire, msg).unwrap();
        }
        let mut cursor = std::io::Cursor::new(wire);
        for expected in &messages {
            let got: EditorToPlayer = read_msg(&mut cursor).unwrap();
            assert_eq!(&got, expected);
        }
        // Stream exhausted → clean EOF on the first header read.
        let eof = read_msg::<EditorToPlayer>(&mut cursor).unwrap_err();
        assert_eq!(eof.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn player_to_editor_frames_round_trip() {
        let messages = vec![
            PlayerToEditor::Ready {
                protocol: PIE_PROTOCOL_VERSION,
            },
            PlayerToEditor::Loaded {
                level: "L".into(),
                actor_count: 2,
            },
            PlayerToEditor::Window { handle: 0x1234 },
            PlayerToEditor::State(PlayerState {
                running: true,
                paused: false,
                frame: 7,
                last_error: None,
            }),
            PlayerToEditor::Ejected,
            PlayerToEditor::Error {
                message: "oops".into(),
            },
        ];
        let mut wire = Vec::new();
        for msg in &messages {
            write_msg(&mut wire, msg).unwrap();
        }
        let mut cursor = std::io::Cursor::new(wire);
        for expected in &messages {
            let got: PlayerToEditor = read_msg(&mut cursor).unwrap();
            assert_eq!(&got, expected);
        }
    }

    #[test]
    fn oversized_frame_is_rejected_without_allocating() {
        let mut wire = Vec::new();
        wire.extend_from_slice(&PIE_FRAME_MAGIC.to_le_bytes());
        wire.extend_from_slice(&PIE_FRAME_VERSION.to_le_bytes());
        wire.extend_from_slice(&(u32::MAX).to_le_bytes());
        let err = read_msg::<EditorToPlayer>(&mut std::io::Cursor::new(wire)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut wire = Vec::new();
        wire.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let err = read_msg::<EditorToPlayer>(&mut std::io::Cursor::new(wire)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    // ── the v5 tail, and the law it was written for (P21.4) ─────────────────

    fn payload_with_assets() -> ScenePayload {
        ScenePayload::new(
            "Cavern",
            vec![9, 8, 7],
            vec![(Uuid::from_u128(1), vec![1, 2, 3])],
            60,
            true,
        )
        .with_pcgs(vec![(Uuid::from_u128(2), vec![4, 5])])
        .with_biome_sets(vec![(Uuid::from_u128(3), vec![6])])
        .with_anim_assets(
            vec![(Uuid::from_u128(4), vec![7])],
            vec![(Uuid::from_u128(5), vec![8])],
            vec![(Uuid::from_u128(6), vec![9])],
        )
        .with_voxels(vec![
            (Uuid::from_u128(0x21_04_01), vec![0xAA; 64]),
            (Uuid::from_u128(0x21_04_02), vec![0xBB; 32]),
        ])
        .with_terrains(vec![(Uuid::from_u128(0x21_04_03), vec![0xCC; 128])])
        // P22.3 — non-empty for the same reason, and it was NOT until the P24.1
        // audit: with `fractures` left empty it and `meshes` were two empty
        // vectors, which agree with each other no matter which order they are in.
        .with_fractures(vec![(Uuid::from_u128(0x22_03_01), vec![0xEE; 200])])
        // v7 (P24.1) — NON-EMPTY, for the reason the round-trip test below
        // records: an empty new field cannot distinguish a tail append from a
        // mid-struct insertion.
        .with_meshes(vec![(Uuid::from_u128(0x24_01_01), vec![0xDD; 96])])
        // v8 (P26.3b) — all four NON-EMPTY and all four a DIFFERENT length, for
        // the reason the P24.1 audit found the hard way: two empty vectors agree
        // with each other however they are ordered, and so do two of equal
        // length.
        .with_garments(
            vec![(Uuid::from_u128(0x26_03_01), vec![0x11; 40])],
            vec![(Uuid::from_u128(0x26_03_02), vec![0x22; 48])],
        )
        .with_materials(
            vec![(Uuid::from_u128(0x26_03_03), vec![0x33; 56])],
            vec![(Uuid::from_u128(0x26_03_04), vec![0x44; 72])],
        )
    }

    /// **The round trip the v5 fields never had.** Every earlier envelope test
    /// used a payload whose new `Vec`s were EMPTY, which is exactly the shape that
    /// cannot tell a mid-struct insertion from a tail one: two empty vectors are
    /// two zero bytes, and two zero bytes are a perfectly good `u32` + `bool`.
    #[test]
    fn a_payload_with_non_empty_voxels_and_terrains_round_trips() {
        let want = payload_with_assets();
        assert!(!want.voxels.is_empty() && !want.terrains.is_empty());

        let mut wire = Vec::new();
        write_msg(
            &mut wire,
            &EditorToPlayer::LoadScene(Box::new(want.clone())),
        )
        .unwrap();
        let got: EditorToPlayer = read_msg(&mut std::io::Cursor::new(wire)).unwrap();
        let EditorToPlayer::LoadScene(got) = got else {
            panic!("not a LoadScene");
        };
        assert_eq!(*got, want);
        // The fields a stale reader would have mis-decoded are the load-bearing
        // ones, so they are named rather than left to the struct equality.
        assert_eq!(got.tick_hz, 60);
        assert!(got.windowed);
        assert_eq!(got.voxels.len(), 2);
        assert_eq!(got.terrains[0].1.len(), 128);
        assert_eq!(got.fractures[0].1.len(), 200);
        // v7 (P24.1): the newest tail field, likewise named rather than left to
        // the struct equality.
        assert_eq!(got.meshes.len(), 1);
        assert_eq!(got.meshes[0].1.len(), 96);
    }

    /// **The WHOLE wire order is pinned, all fifteen fields** (P24.1 audit M5).
    ///
    /// The stale-reader test below models a *pre-v5* build and therefore stops at
    /// `windowed` — correctly, that is its whole point. But it left the four tail
    /// fields (`voxels`, `terrains`, `fractures`, `meshes`) unpinned in ORDER, and
    /// the round trip above cannot see an order change either, because it compares
    /// a payload with itself. `meshes` inserted between `voxels` and `terrains`
    /// passed both.
    ///
    /// This decodes a real encoding through a positional twin listing every field
    /// in the canonical order, then compares field for field. Two things make it
    /// able to fail:
    ///
    ///  * the fixture gives every tail field **distinct, non-empty** content, so a
    ///    swapped pair mismatches instead of two empty vectors agreeing (the exact
    ///    hazard `ScenePayload::meshes`' own doc cites);
    ///  * the decode must consume **every byte**, so a field appended without a
    ///    version bump leaves a remainder here.
    #[test]
    fn the_wire_order_is_pinned_field_for_field() {
        #[derive(Deserialize)]
        struct WireOrder {
            schema_version: u32,
            label: String,
            level_bytes: Vec<u8>,
            classes: Vec<(Uuid, Vec<u8>)>,
            pcgs: Vec<(Uuid, Vec<u8>)>,
            skeletons: Vec<(Uuid, Vec<u8>)>,
            clips: Vec<(Uuid, Vec<u8>)>,
            machines: Vec<(Uuid, Vec<u8>)>,
            biome_sets: Vec<(Uuid, Vec<u8>)>,
            tick_hz: u32,
            windowed: bool,
            // ── the append-only tail ────────────────────────────────────────
            voxels: Vec<(Uuid, Vec<u8>)>,
            terrains: Vec<(Uuid, Vec<u8>)>,
            fractures: Vec<(Uuid, Vec<u8>)>,
            meshes: Vec<(Uuid, Vec<u8>)>,
            cloths: Vec<(Uuid, Vec<u8>)>,
            hairs: Vec<(Uuid, Vec<u8>)>,
            materials: Vec<(Uuid, Vec<u8>)>,
            textures: Vec<(Uuid, Vec<u8>)>,
        }

        let want = payload_with_assets();
        // Every tail field carries different bytes AND a different length, so no
        // two of them can compare equal by accident.
        let tail_lens: Vec<usize> = [
            &want.voxels,
            &want.terrains,
            &want.fractures,
            &want.meshes,
            &want.cloths,
            &want.hairs,
            &want.materials,
            &want.textures,
        ]
        .iter()
        .map(|v| v[0].1.len())
        .collect();
        let mut uniq = tail_lens.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            tail_lens.len(),
            "two tail fields carry payloads of the same length, so swapping them \
             would be invisible to this gate: {tail_lens:?}"
        );

        let bytes = bincode::serde::encode_to_vec(&want, bincode::config::standard()).unwrap();
        let (wire, consumed): (WireOrder, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .expect("the canonical 19-field order decodes the wire");
        assert_eq!(
            consumed,
            bytes.len(),
            "the encoding carries bytes the pinned 19-field order does not account \
             for — a field was added to `ScenePayload` without extending this pin"
        );

        assert_eq!(wire.schema_version, SCENE_PAYLOAD_VERSION);
        assert_eq!(wire.label, want.label);
        assert_eq!(wire.level_bytes, want.level_bytes);
        assert_eq!(wire.classes, want.classes);
        assert_eq!(wire.pcgs, want.pcgs);
        assert_eq!(wire.skeletons, want.skeletons);
        assert_eq!(wire.clips, want.clips);
        assert_eq!(wire.machines, want.machines);
        assert_eq!(wire.biome_sets, want.biome_sets);
        assert_eq!(wire.tick_hz, want.tick_hz);
        assert_eq!(wire.windowed, want.windowed);
        assert_eq!(wire.voxels, want.voxels, "`voxels` is not wire field 12");
        assert_eq!(
            wire.terrains, want.terrains,
            "`terrains` is not wire field 13"
        );
        assert_eq!(
            wire.fractures, want.fractures,
            "`fractures` is not wire field 14"
        );
        assert_eq!(wire.meshes, want.meshes, "`meshes` is not wire field 15");
        assert_eq!(wire.cloths, want.cloths, "`cloths` is not wire field 16");
        assert_eq!(wire.hairs, want.hairs, "`hairs` is not wire field 17");
        assert_eq!(
            wire.materials, want.materials,
            "`materials` is not wire field 18"
        );
        assert_eq!(
            wire.textures, want.textures,
            "`textures` is not wire field 19"
        );
    }

    /// **The tail is positional, and it is at the tail.** A reader that knows only
    /// the pre-v5 field order decodes `tick_hz` and `windowed` **correctly** from a
    /// v5 payload and simply stops early — which is what makes the version check a
    /// refusal instead of a corruption. (Modelled by decoding into a struct with
    /// the tail truncated, which is exactly what an older build's `ScenePayload`
    /// is.)
    #[test]
    fn a_stale_reader_still_decodes_every_field_it_knows() {
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct PreV5 {
            schema_version: u32,
            label: String,
            level_bytes: Vec<u8>,
            classes: Vec<(Uuid, Vec<u8>)>,
            pcgs: Vec<(Uuid, Vec<u8>)>,
            skeletons: Vec<(Uuid, Vec<u8>)>,
            clips: Vec<(Uuid, Vec<u8>)>,
            machines: Vec<(Uuid, Vec<u8>)>,
            biome_sets: Vec<(Uuid, Vec<u8>)>,
            tick_hz: u32,
            windowed: bool,
        }

        let want = payload_with_assets();
        let bytes = bincode::serde::encode_to_vec(&want, bincode::config::standard()).unwrap();
        let (old, _): (PreV5, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(old.schema_version, SCENE_PAYLOAD_VERSION);
        assert_eq!(old.tick_hz, want.tick_hz, "the tail moved a known field");
        assert_eq!(old.windowed, want.windowed, "the tail moved a known field");
        assert_eq!(old.label, want.label);
    }

    /// …and the version check is what stops it running with the caves missing.
    ///
    /// **Both directions.** The envelope is positional, so a *newer* payload is as
    /// unreadable as an older one — `check_version` is exact equality and not a
    /// floor, and this asserts the property rather than the implementation.
    #[test]
    fn a_version_mismatch_is_refused_loudly() {
        let mut p = payload_with_assets();
        assert!(p.check_version().is_ok());
        for v in [SCENE_PAYLOAD_VERSION - 1, SCENE_PAYLOAD_VERSION + 1] {
            p.schema_version = v;
            let err = p
                .check_version()
                .expect_err("a mismatched payload must be refused");
            assert!(err.contains("schema"), "{err}");
        }
    }

    /// A frame past the cap is a **real error**, not a truncated length header
    /// that desyncs the stream after the body is already committed.
    #[test]
    fn an_oversized_frame_is_refused_rather_than_truncated() {
        // A type whose *encoding* is enormous but whose value is not: serde is
        // told there are `MAX_FRAME_LEN + 16` bytes and yields them lazily, so the
        // check is exercised without a 256 MiB allocation in a unit test.
        struct HugeBlob;
        impl Serialize for HugeBlob {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeSeq;
                let n = MAX_FRAME_LEN + 16;
                let mut seq = s.serialize_seq(Some(n))?;
                for _ in 0..n {
                    seq.serialize_element(&0u8)?;
                }
                seq.end()
            }
        }

        let mut wire = Vec::new();
        let err = write_msg(&mut wire, &HugeBlob).expect_err("an oversized frame must be refused");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            wire.is_empty(),
            "the header was written before the refusal, which desyncs the stream"
        );

        // …and a frame just under the cap is accepted, so the bound is a bound and
        // not a blanket refusal.
        let ok = ScenePayload::new("Small", vec![0u8; 1024], vec![], 60, false);
        let mut wire = Vec::new();
        write_msg(&mut wire, &EditorToPlayer::LoadScene(Box::new(ok)))
            .expect("a normal frame writes");
        assert!(!wire.is_empty());
    }
}
