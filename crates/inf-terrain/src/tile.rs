//! One page of the heightfield: a square block of `f32` sample heights measured
//! against an `f64` world anchor (the precision doctrine — tile-local `f32`
//! against an `f64` origin, so a planetary-scale terrain never loses vertical
//! resolution to `f32` range).

use glam::DVec3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A tile's identity inside a terrain: its grid coordinate **plus its LOD level**
/// (P16.3).
///
/// Level `0` is the authored, full-resolution level — the one [`super::TerrainData`]'s
/// height/normal queries, the sculpt brushes and the `.inf_lvl` inline form all
/// mean when they say "tile `(tx, tz)`". Level `n` covers `2ⁿ ×` the world span at
/// the same sample count (metres-per-sample doubles each level), so a level-`n`
/// tile `(TX, TZ)` is the 2×2 block of level-`(n−1)` tiles
/// `(2TX+a, 2TZ+b), a,b ∈ {0,1}` decimated 2:1 (see [`crate::pyramid`]).
///
/// `Ord` sorts by **`lod` first, then `tx`, then `tz`** — so a `BTreeMap`/
/// `BTreeSet` of keys groups a whole LOD level contiguously (a clipmap ring reads
/// one contiguous run of the asset's tile directory) and every iteration order is
/// deterministic, which is what makes the `.inf_terrain` layout byte-stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TileKey {
    /// LOD level; `0` is the authored full-resolution level.
    pub lod: u32,
    /// Tile grid coordinate `(tx, tz)` **within that level**.
    pub coord: (i32, i32),
}

impl TileKey {
    /// A level-0 (authored, full-resolution) key.
    #[inline]
    pub const fn lod0(coord: (i32, i32)) -> Self {
        Self { lod: 0, coord }
    }

    /// A key at an explicit level.
    #[inline]
    pub const fn new(lod: u32, coord: (i32, i32)) -> Self {
        Self { lod, coord }
    }

    /// `true` for the authored full-resolution level.
    #[inline]
    pub const fn is_lod0(self) -> bool {
        self.lod == 0
    }

    /// The coarser key one level up that contains this tile (`lod + 1`, coordinate
    /// halved with floor semantics so negative coordinates group correctly).
    #[inline]
    pub const fn parent(self) -> Self {
        Self {
            lod: self.lod + 1,
            coord: (self.coord.0.div_euclid(2), self.coord.1.div_euclid(2)),
        }
    }

    /// The four finer keys one level down whose footprints tile this one
    /// (`lod − 1`, coordinate doubled), in the fixed order
    /// `(2x, 2z), (2x+1, 2z), (2x, 2z+1), (2x+1, 2z+1)`.
    ///
    /// The exact inverse of [`parent`](Self::parent) — `k.parent().children()`
    /// contains `k` for every key, including negative coordinates, because the
    /// halving floors. At level 0 (nothing is finer) the level saturates and the
    /// result is meaningless; callers gate on `lod > 0`.
    ///
    /// MIRROR of `inf_render::TerrainTileKey::children` — the two must agree, or
    /// the streamer's cut and the renderer's `fully_subdivided` test would
    /// disagree about what "all four children" means.
    #[inline]
    pub const fn children(self) -> [Self; 4] {
        let lod = self.lod.saturating_sub(1);
        let (x, z) = (
            self.coord.0.saturating_mul(2),
            self.coord.1.saturating_mul(2),
        );
        [
            Self::new(lod, (x, z)),
            Self::new(lod, (x.saturating_add(1), z)),
            Self::new(lod, (x, z.saturating_add(1))),
            Self::new(lod, (x.saturating_add(1), z.saturating_add(1))),
        ]
    }
}

/// The default per-sample splat weight: **100 % layer 0** (`[255, 0, 0, 0]`), so
/// an unpainted terrain shades entirely as its first [`TerrainLayer`]. Channels
/// are `[layer0, layer1, layer2, layer3]` and are kept normalized to sum ≈ 255.
pub const DEFAULT_WEIGHT: [u8; 4] = [255, 0, 0, 0];

/// Number of **erosion data-map** channels stored per sample (P19.1): flow,
/// deposition, wear — see [`DataMapKind`].
pub const DATA_MAP_CHANNELS: usize = 3;

/// The per-sample erosion data-map default: **never eroded** (all three
/// accumulators at zero). A tile that has never been eroded stores no data maps
/// at all — see [`TerrainTile::maps`].
pub const DEFAULT_DATA_MAP: [f32; DATA_MAP_CHANNELS] = [0.0; DATA_MAP_CHANNELS];

/// The **reserved** biome id meaning *this sample belongs to no named biome*
/// (P19.2).
///
/// Id `0` is not definable in a [`BiomeSet`](crate::BiomeSet) — it is the sparse
/// default every sample starts at, so "unassigned" needs no storage and a
/// never-painted tile costs exactly one length byte. Everything downstream reads
/// it as *absent*: the overlay draws it neutral, and P19.3's per-biome dispatch
/// evaluates nothing for it.
pub const UNASSIGNED_BIOME: u8 = 0;

/// The per-sample biome default — [`UNASSIGNED_BIOME`]. A tile nobody has painted
/// a biome onto stores no biome ids at all; see [`TerrainTile::biomes`].
pub const DEFAULT_BIOME: u8 = UNASSIGNED_BIOME;

/// The largest number of biomes a [`BiomeSet`](crate::BiomeSet) can define: ids
/// are `u8` and `0` is [reserved](UNASSIGNED_BIOME), so `1..=255` are definable.
pub const MAX_BIOMES: usize = 255;

/// Which **erosion data map** (P19.1) a sample or export refers to.
///
/// All three are **raw, monotone, non-negative accumulators** over every erosion
/// step a tile has ever been through — never normalized on the way in, so a
/// second bake adds to the first's totals and a normalized *view* (the PNG
/// export, a PCG mask) is always derived, never stored. Units follow the
/// SI doctrine (`docs/memos/units-doctrine.md`): 1 world unit = 1 metre.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataMapKind {
    /// **Flow** — `Σ_steps dt · Σ_pipes outflow`, the time-integrated water flux
    /// leaving the cell over the whole run. Units **m³** (cubic metres of water),
    /// because the virtual-pipe flux is a volume *rate* (m³·s⁻¹) and `dt` is
    /// seconds. This is the classic flow-accumulation map: it peaks in the
    /// channels the water carved.
    Flow,
    /// **Deposition** — `Σ_steps` material settled out of suspension onto the
    /// cell. Units **metres** of terrain height gained (multiply by the cell area
    /// `l²` for a volume).
    Deposition,
    /// **Wear** — `Σ_steps` material dissolved off the cell into suspension.
    /// Units **metres** of terrain height lost (multiply by `l²` for a volume).
    Wear,
}

impl DataMapKind {
    /// Every kind, in channel order — the order the per-sample `[f32; 3]` is
    /// stored in, so `ALL[k].channel() == k`.
    pub const ALL: [DataMapKind; DATA_MAP_CHANNELS] = [
        DataMapKind::Flow,
        DataMapKind::Deposition,
        DataMapKind::Wear,
    ];

    /// Index of this kind inside a per-sample `[f32; DATA_MAP_CHANNELS]`.
    #[inline]
    pub const fn channel(self) -> usize {
        match self {
            DataMapKind::Flow => 0,
            DataMapKind::Deposition => 1,
            DataMapKind::Wear => 2,
        }
    }

    /// Lowercase wire/UI name (`"flow"`, `"deposition"`, `"wear"`) — the string
    /// the editor command and the exported file name use.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            DataMapKind::Flow => "flow",
            DataMapKind::Deposition => "deposition",
            DataMapKind::Wear => "wear",
        }
    }

    /// SI unit of this map's accumulator, for diagnostics and UI.
    #[inline]
    pub const fn unit(self) -> &'static str {
        match self {
            // Flux is a volume rate; integrating it over time gives a volume.
            DataMapKind::Flow => "m^3",
            DataMapKind::Deposition | DataMapKind::Wear => "m",
        }
    }

    /// Parse a [`label`](Self::label) (case-insensitive). `None` for anything else
    /// — the IPC boundary rejects rather than guesses.
    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "flow" => Some(DataMapKind::Flow),
            "deposition" => Some(DataMapKind::Deposition),
            "wear" => Some(DataMapKind::Wear),
            _ => None,
        }
    }
}

/// A single terrain tile: `resolution × resolution` height samples (row-major,
/// `j * resolution + i`, `i` along +X, `j` along +Z) stored as `f32` offsets
/// from the tile's `f64` [`origin`](TerrainTile::origin).
///
/// The tile owns its `f64` world origin (the world position of sample `(0, 0)`):
/// horizontal `x`/`z` are set from the tile grid coordinate by [`super::TerrainData`],
/// and `y` is the anchor the `f32` heights are relative to. Absolute world height
/// of sample `(i, j)` is therefore `origin.y + heights[j*res + i] as f64` — the
/// `f32` only ever carries a tile-local delta.
///
/// ## Splat weights (P10.4)
///
/// Beside the heights, a tile carries a parallel **splat weight** layer
/// ([`weights`](TerrainTile::weights)): one `[u8; 4]` per sample, the RGBA-packed
/// blend weights of the four [`TerrainLayer`]s (normalized to sum ≈ 255). To keep
/// an unpainted terrain free — and, crucially, **byte-identical to a pre-P10.4
/// serialized tile** — the buffer is stored *sparsely*: an empty `Vec` means
/// "every sample is [`DEFAULT_WEIGHT`]" (uniform layer 0). Painting the tile
/// [materializes](TerrainTile::ensure_weights) the full `resolution²` buffer.
///
/// ## Erosion data maps (P19.1)
///
/// A third per-sample layer ([`maps`](TerrainTile::maps)) rides beside the heights
/// and the weights: one `[f32; 3]` per sample holding the **flow / deposition /
/// wear** accumulators an erosion bake wrote (see [`DataMapKind`] for the exact
/// definitions and units). It is stored **sparsely** on the same rule as the splat
/// weights — an empty `Vec` means "every sample is [`DEFAULT_DATA_MAP`]", i.e.
/// this tile has never been eroded — so a never-eroded tile costs **zero
/// per-sample bytes** (and, in bincode, exactly one length byte).
///
/// ## Biome ids (P19.2)
///
/// A fourth per-sample layer ([`biomes`](TerrainTile::biomes)): one `u8` per
/// sample naming which of the level's [`BiomeSet`](crate::BiomeSet) biomes owns
/// the sample. Same sparse rule again — an empty `Vec` means "every sample is
/// [`UNASSIGNED_BIOME`]" (id `0`, the reserved *no biome* value), so an unpainted
/// tile costs zero per-sample bytes.
///
/// Unlike the weights, this layer is **categorical**: an id names a biome, it
/// does not blend with its neighbours. That is why it is a separate `u8` layer
/// rather than a fifth splat channel, and why the paint brush writes a crisp
/// boundary (see [`crate::paint_biome`]).
///
/// Serde: `origin` (as a portable `[f64; 3]`, since the workspace `glam` pin has
/// no `serde` feature) + a flat `heights` sequence, and — **only when non-empty**
/// (`skip_serializing_if`) — flat `weights`, `maps` and `biomes` sequences. An old
/// tile (no `weights`/`maps`/`biomes` field) decodes with the defaults, and an
/// unpainted, un-eroded new tile serializes without any of them, so existing
/// human-readable bytes round-trip unchanged. `resolution` is not stored on the
/// tile (it is a terrain-wide constant on [`super::TerrainData`]); the terrain
/// validates `heights.len() == resolution²` and, when present, `weights.len()` /
/// `maps.len()` / `biomes.len() == resolution²` on load.
///
/// **The bincode form is versioned at the container, not the tile.** bincode is
/// positional, so appending a layer is a wire-format change: a stream written
/// before P19.2 has four fields where this build reads five. Each historical
/// layout is therefore frozen — [`TerrainTileFrozenV1`] (pre-P19.1: no maps, no
/// biomes) and [`TerrainTileFrozenV2`] (P19.1: maps, no biomes) — and selected by
/// the *container's* schema version. See [`crate::asset::decode_tile_at`] and the
/// generation table on [`TerrainTileFrozenV1`].
///
/// [`TerrainLayer`]: the ECS `inf_ecs::components::TerrainLayer` (the layer
/// definitions live on the `Terrain` component; the tile only stores per-sample
/// weights into them).
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainTile {
    /// World position of sample `(0, 0)` (`f64` anchor for the `f32` heights).
    pub origin: DVec3,
    /// `resolution²` height offsets (metres) from `origin.y`, row-major.
    heights: Vec<f32>,
    /// `resolution²` row-major RGBA splat weights, **or empty** for the uniform
    /// [`DEFAULT_WEIGHT`] (layer 0) — see the type docs.
    weights: Vec<[u8; 4]>,
    /// `resolution²` row-major erosion data maps (`[flow, deposition, wear]`),
    /// **or empty** for the never-eroded [`DEFAULT_DATA_MAP`] — see the type docs.
    maps: Vec<[f32; DATA_MAP_CHANNELS]>,
    /// `resolution²` row-major biome ids, **or empty** for the unpainted
    /// [`DEFAULT_BIOME`] ([`UNASSIGNED_BIOME`]) — see the type docs.
    biomes: Vec<u8>,
}

/// Serde wire form for **human-readable** formats (JSON/TOML): `origin` as
/// `[f64; 3]` (glam `DVec3` isn't serde-derivable without enabling glam's `serde`
/// feature workspace-wide). `weights` (P10.4), `maps` (P19.1) and `biomes`
/// (P19.2) are appended and skipped when empty, so pre-P10.4 tiles — and
/// unpainted, un-eroded new tiles — encode byte-identically to the two-field form.
#[derive(Serialize, Deserialize)]
struct TerrainTileRaw {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    weights: Vec<[u8; 4]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    maps: Vec<[f32; DATA_MAP_CHANNELS]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    biomes: Vec<u8>,
}

/// Serde wire form for **non-self-describing** formats (bincode — the `.inf_lvl`
/// terrain persistence path, P10.6, and the `.inf_terrain` tile blob, P16.3).
/// `weights`, `maps` and `biomes` are **always** encoded (as plain
/// length-prefixed sequences — a 0-length count for an unpainted / never-eroded
/// tile), because a `skip_serializing_if` field desyncs a non-self-describing
/// stream (the engine-wide bincode constraint; the same reason
/// `.inf_pcg`/`.inf_act` store their skip-heavy models as JSON strings). An
/// untouched tile still costs only a single length byte per layer, so terrain
/// stays compact and byte-stable in bincode too.
#[derive(Serialize, Deserialize)]
struct TerrainTileBin {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default)]
    weights: Vec<[u8; 4]>,
    #[serde(default)]
    maps: Vec<[f32; DATA_MAP_CHANNELS]>,
    #[serde(default)]
    biomes: Vec<u8>,
}

/// **Frozen tile wire layout, generation 1** — `origin + heights + weights`, with
/// no erosion data maps and no biome ids. The shape every terrain tile had before
/// P19.1.
///
/// # THE GENERATION TABLE
///
/// A tile is written into **two** containers that version themselves
/// independently, and neither owns the other. Every row here is load-bearing —
/// this is the table the tripwires below assert, and the one
/// [`crate::asset::decode_tile_at`] implements:
///
/// | tile generation | layout | `.inf_lvl` (scene schema) | `.inf_terrain` (asset header) |
/// |---|---|---|---|
/// | [`TerrainTileFrozenV1`] | origin + heights + weights | v1 … **v14** | v1, **v2** |
/// | [`TerrainTileFrozenV2`] | + erosion data maps (P19.1) | **v15** | **v3** |
/// | [`TerrainTile`] (live) | + biome ids (P19.2) | v16 | v4 |
///
/// Naming generation 1 `TerrainTileV14` — as the first cut did — leaks the
/// *scene* codec's numbering into `inf-terrain`'s public API and silently implies
/// the asset container agrees, which it does not (it calls the same bytes "v2").
/// The generation counter belongs to the **tile**: each tile-layout change adds
/// one row here and **both** container columns gain a version.
/// `frozen_tile_generations_are_pinned_to_both_ladders` is the tripwire that fails
/// if either container bumps past its row without a new generation.
///
/// bincode is positional, so these bytes cannot be fed to the grown
/// [`TerrainTile`] — the decoder would read past the end of the tile and into
/// whatever follows it. Each container picks the wire type from *its own* version
/// stamp and lifts through [`into_current`](Self::into_current); the reverse
/// projection ([`from_current`](Self::from_current)) is the downgrade-bless
/// direction, used when a frozen record has to be *written* in a test fixture.
///
/// Like [`TerrainTile`] it is format-aware, so the frozen shape is exact in both
/// the human-readable and the bincode codec.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainTileFrozenV1 {
    /// World position of sample `(0, 0)`.
    pub origin: DVec3,
    /// `resolution²` height offsets, row-major.
    pub heights: Vec<f32>,
    /// `resolution²` row-major splat weights, or empty for the sparse default.
    pub weights: Vec<[u8; 4]>,
}

/// Human-readable wire form of [`TerrainTileFrozenV1`] — the exact pre-P19.1
/// [`TerrainTileRaw`].
#[derive(Serialize, Deserialize)]
struct TerrainTileFrozenV1Raw {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    weights: Vec<[u8; 4]>,
}

/// bincode wire form of [`TerrainTileFrozenV1`] — the exact pre-P19.1
/// [`TerrainTileBin`] (three positional fields, no `maps`).
#[derive(Serialize, Deserialize)]
struct TerrainTileFrozenV1Bin {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default)]
    weights: Vec<[u8; 4]>,
}

impl TerrainTileFrozenV1 {
    /// Lift a frozen tile to the live one: the data maps are **empty** (never
    /// eroded) and the biome ids are **empty** (unassigned) — which is exactly
    /// what a pre-P19.1 tile meant.
    pub fn into_current(self) -> TerrainTile {
        TerrainTile {
            origin: self.origin,
            heights: self.heights,
            weights: self.weights,
            maps: Vec::new(),
            biomes: Vec::new(),
        }
    }

    /// Project a live tile onto the frozen shape, **dropping** its data maps and
    /// biome ids (the downgrade-bless direction — a pre-P19.1 container cannot
    /// carry either).
    pub fn from_current(tile: &TerrainTile) -> Self {
        Self {
            origin: tile.origin,
            heights: tile.heights.clone(),
            weights: tile.weights.clone(),
        }
    }

    /// Lift one generation, to [`TerrainTileFrozenV2`] with empty data maps.
    ///
    /// The editor codec's ladder is *chained* (v1 → v2 → … → current), so it needs
    /// the single-step hop rather than a jump to the live type; going via
    /// `into_current` would work but would clone every buffer twice and would stop
    /// being obviously lossless.
    pub fn into_v2(self) -> TerrainTileFrozenV2 {
        TerrainTileFrozenV2 {
            origin: self.origin,
            heights: self.heights,
            weights: self.weights,
            maps: Vec::new(),
        }
    }
}

/// **Frozen tile wire layout, generation 2** — `origin + heights + weights +
/// maps`, with no biome ids. The shape every terrain tile had between P19.1 and
/// P19.2 (`.inf_lvl` v15, `.inf_terrain` header v3).
///
/// See [`TerrainTileFrozenV1`]'s generation table for why the name counts tile
/// generations rather than container versions, and why one exists at all: bincode
/// is positional, so P19.2's `biomes` layer means a v15 payload has four fields
/// where the live [`TerrainTile`] reads five.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainTileFrozenV2 {
    /// World position of sample `(0, 0)`.
    pub origin: DVec3,
    /// `resolution²` height offsets, row-major.
    pub heights: Vec<f32>,
    /// `resolution²` row-major splat weights, or empty for the sparse default.
    pub weights: Vec<[u8; 4]>,
    /// `resolution²` row-major erosion data maps, or empty for never-eroded.
    pub maps: Vec<[f32; DATA_MAP_CHANNELS]>,
}

/// Human-readable wire form of [`TerrainTileFrozenV2`] — the exact pre-P19.2
/// [`TerrainTileRaw`].
#[derive(Serialize, Deserialize)]
struct TerrainTileFrozenV2Raw {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    weights: Vec<[u8; 4]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    maps: Vec<[f32; DATA_MAP_CHANNELS]>,
}

/// bincode wire form of [`TerrainTileFrozenV2`] — the exact pre-P19.2
/// [`TerrainTileBin`] (four positional fields, no `biomes`).
#[derive(Serialize, Deserialize)]
struct TerrainTileFrozenV2Bin {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default)]
    weights: Vec<[u8; 4]>,
    #[serde(default)]
    maps: Vec<[f32; DATA_MAP_CHANNELS]>,
}

impl TerrainTileFrozenV2 {
    /// Lift to the live tile: the biome ids come up **empty**, i.e. every sample
    /// is [`UNASSIGNED_BIOME`] — exactly what a pre-P19.2 tile meant.
    pub fn into_current(self) -> TerrainTile {
        TerrainTile {
            origin: self.origin,
            heights: self.heights,
            weights: self.weights,
            maps: self.maps,
            biomes: Vec::new(),
        }
    }

    /// Project a live tile onto the frozen shape, **dropping** its biome ids (the
    /// downgrade-bless direction).
    pub fn from_current(tile: &TerrainTile) -> Self {
        Self {
            origin: tile.origin,
            heights: tile.heights.clone(),
            weights: tile.weights.clone(),
            maps: tile.maps.clone(),
        }
    }
}

impl Serialize for TerrainTileFrozenV2 {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            TerrainTileFrozenV2Raw {
                origin: self.origin.to_array(),
                heights: self.heights.clone(),
                weights: self.weights.clone(),
                maps: self.maps.clone(),
            }
            .serialize(s)
        } else {
            TerrainTileFrozenV2Bin {
                origin: self.origin.to_array(),
                heights: self.heights.clone(),
                weights: self.weights.clone(),
                maps: self.maps.clone(),
            }
            .serialize(s)
        }
    }
}

impl<'de> Deserialize<'de> for TerrainTileFrozenV2 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            let raw = TerrainTileFrozenV2Raw::deserialize(d)?;
            Ok(Self {
                origin: DVec3::from_array(raw.origin),
                heights: raw.heights,
                weights: raw.weights,
                maps: raw.maps,
            })
        } else {
            let raw = TerrainTileFrozenV2Bin::deserialize(d)?;
            Ok(Self {
                origin: DVec3::from_array(raw.origin),
                heights: raw.heights,
                weights: raw.weights,
                maps: raw.maps,
            })
        }
    }
}

impl Serialize for TerrainTileFrozenV1 {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            TerrainTileFrozenV1Raw {
                origin: self.origin.to_array(),
                heights: self.heights.clone(),
                weights: self.weights.clone(),
            }
            .serialize(s)
        } else {
            TerrainTileFrozenV1Bin {
                origin: self.origin.to_array(),
                heights: self.heights.clone(),
                weights: self.weights.clone(),
            }
            .serialize(s)
        }
    }
}

impl<'de> Deserialize<'de> for TerrainTileFrozenV1 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            let raw = TerrainTileFrozenV1Raw::deserialize(d)?;
            Ok(Self {
                origin: DVec3::from_array(raw.origin),
                heights: raw.heights,
                weights: raw.weights,
            })
        } else {
            let raw = TerrainTileFrozenV1Bin::deserialize(d)?;
            Ok(Self {
                origin: DVec3::from_array(raw.origin),
                heights: raw.heights,
                weights: raw.weights,
            })
        }
    }
}

impl Serialize for TerrainTile {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Human-readable formats keep the sparse (skip-empty) form so JSON/TOML
        // stay byte-stable; bincode uses the always-present form so it never
        // desyncs (see [`TerrainTileBin`]).
        if s.is_human_readable() {
            TerrainTileRaw {
                origin: self.origin.to_array(),
                heights: self.heights.clone(),
                weights: self.weights.clone(),
                maps: self.maps.clone(),
                biomes: self.biomes.clone(),
            }
            .serialize(s)
        } else {
            TerrainTileBin {
                origin: self.origin.to_array(),
                heights: self.heights.clone(),
                weights: self.weights.clone(),
                maps: self.maps.clone(),
                biomes: self.biomes.clone(),
            }
            .serialize(s)
        }
    }
}

impl<'de> Deserialize<'de> for TerrainTile {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            let raw = TerrainTileRaw::deserialize(d)?;
            Ok(Self {
                origin: DVec3::from_array(raw.origin),
                heights: raw.heights,
                weights: raw.weights,
                maps: raw.maps,
                biomes: raw.biomes,
            })
        } else {
            let raw = TerrainTileBin::deserialize(d)?;
            Ok(Self {
                origin: DVec3::from_array(raw.origin),
                heights: raw.heights,
                weights: raw.weights,
                maps: raw.maps,
                biomes: raw.biomes,
            })
        }
    }
}

impl TerrainTile {
    /// A flat tile at `origin` with every sample `0.0` (height == `origin.y`) and
    /// uniform default splat weights (layer 0).
    pub fn flat(resolution: u32, origin: DVec3) -> Self {
        Self {
            origin,
            heights: vec![0.0; (resolution * resolution) as usize],
            weights: Vec::new(),
            maps: Vec::new(),
            biomes: Vec::new(),
        }
    }

    /// Build a tile from an explicit height buffer (row-major, length
    /// `resolution²`). Returns `None` on a length mismatch. Splat weights start
    /// uniform (layer 0).
    pub fn from_heights(resolution: u32, origin: DVec3, heights: Vec<f32>) -> Option<Self> {
        (heights.len() == (resolution * resolution) as usize).then_some(Self {
            origin,
            heights,
            weights: Vec::new(),
            maps: Vec::new(),
            biomes: Vec::new(),
        })
    }

    /// The raw row-major height buffer (`f32` offsets from `origin.y`).
    #[inline]
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Mutable access to the raw buffer (bulk edits — the P10.2 brush seam).
    #[inline]
    pub fn heights_mut(&mut self) -> &mut [f32] {
        &mut self.heights
    }

    /// The `f32` height offset at sample `(i, j)` (`0`-based, `i` = column/+X,
    /// `j` = row/+Z). Out-of-range indices clamp to the edge.
    #[inline]
    pub fn sample(&self, resolution: u32, i: u32, j: u32) -> f32 {
        let r = resolution.max(1);
        let i = i.min(r - 1);
        let j = j.min(r - 1);
        self.heights[(j * r + i) as usize]
    }

    /// The absolute world height of sample `(i, j)` (`origin.y + offset`).
    #[inline]
    pub fn world_height(&self, resolution: u32, i: u32, j: u32) -> f64 {
        self.origin.y + self.sample(resolution, i, j) as f64
    }

    /// Write the `f32` offset at sample `(i, j)`. Out-of-range indices are ignored.
    #[inline]
    pub fn set_sample(&mut self, resolution: u32, i: u32, j: u32, height: f32) {
        let r = resolution.max(1);
        if i < r && j < r {
            self.heights[(j * r + i) as usize] = height;
        }
    }

    // ── splat weights (P10.4) ───────────────────────────────────────────────

    /// The raw splat-weight buffer: either empty (uniform [`DEFAULT_WEIGHT`],
    /// layer 0) or `resolution²` row-major `[u8; 4]`. Prefer
    /// [`weight_sample`](TerrainTile::weight_sample) for a value that already
    /// resolves the empty case.
    #[inline]
    pub fn weights(&self) -> &[[u8; 4]] {
        &self.weights
    }

    /// `true` when the tile stores no per-sample weights — i.e. it is uniform
    /// layer 0 ([`DEFAULT_WEIGHT`] everywhere). This is the byte-stable default
    /// (an unpainted tile serializes without a `weights` field).
    #[inline]
    pub fn weights_are_default(&self) -> bool {
        self.weights.is_empty()
    }

    /// The splat weight at sample `(i, j)`, resolving the sparse default: an
    /// unpainted tile returns [`DEFAULT_WEIGHT`] for every sample. Out-of-range
    /// indices clamp to the edge.
    #[inline]
    pub fn weight_sample(&self, resolution: u32, i: u32, j: u32) -> [u8; 4] {
        if self.weights.is_empty() {
            return DEFAULT_WEIGHT;
        }
        let r = resolution.max(1);
        let i = i.min(r - 1);
        let j = j.min(r - 1);
        self.weights[(j * r + i) as usize]
    }

    /// Materialize the full `resolution²` weight buffer (filled with
    /// [`DEFAULT_WEIGHT`]) if the tile is still on the sparse default, then return
    /// a mutable handle to it. Painting calls this before writing samples.
    #[inline]
    pub fn ensure_weights(&mut self, resolution: u32) -> &mut [[u8; 4]] {
        if self.weights.is_empty() {
            self.weights = vec![DEFAULT_WEIGHT; (resolution * resolution) as usize];
        }
        &mut self.weights
    }

    /// Reset the tile to the sparse uniform default (drops any painted weights),
    /// so it re-serializes byte-identically to a never-painted tile. Used by
    /// splat-paint **undo** when a stroke had materialized the buffer.
    #[inline]
    pub fn clear_weights(&mut self) {
        self.weights = Vec::new();
    }

    /// Write the splat weight at sample `(i, j)`, materializing the buffer first.
    /// Out-of-range indices are ignored.
    #[inline]
    pub fn set_weight_sample(&mut self, resolution: u32, i: u32, j: u32, weight: [u8; 4]) {
        let r = resolution.max(1);
        if i < r && j < r {
            self.ensure_weights(resolution)[(j * r + i) as usize] = weight;
        }
    }

    /// Length of the stored weight buffer (`0` for the sparse default). Used by
    /// the terrain's serde length validation.
    #[inline]
    pub fn weights_len(&self) -> usize {
        self.weights.len()
    }

    // ── erosion data maps (P19.1) ───────────────────────────────────────────

    /// The raw data-map buffer: either empty (never eroded — every sample is
    /// [`DEFAULT_DATA_MAP`]) or `resolution²` row-major
    /// `[flow, deposition, wear]`. Prefer
    /// [`map_sample`](TerrainTile::map_sample) for a value that already resolves
    /// the empty case.
    #[inline]
    pub fn maps(&self) -> &[[f32; DATA_MAP_CHANNELS]] {
        &self.maps
    }

    /// `true` when the tile stores no data maps — i.e. it has **never been
    /// eroded**. This is the byte-stable default (such a tile serializes without
    /// a `maps` field at all in JSON/TOML, and as a single zero-length count in
    /// bincode).
    #[inline]
    pub fn maps_are_default(&self) -> bool {
        self.maps.is_empty()
    }

    /// One data-map channel at sample `(i, j)`, resolving the sparse default (a
    /// never-eroded tile reads `0.0` everywhere). Out-of-range indices clamp to
    /// the edge.
    #[inline]
    pub fn map_sample(&self, resolution: u32, kind: DataMapKind, i: u32, j: u32) -> f32 {
        if self.maps.is_empty() {
            return DEFAULT_DATA_MAP[kind.channel()];
        }
        let r = resolution.max(1);
        let i = i.min(r - 1);
        let j = j.min(r - 1);
        self.maps[(j * r + i) as usize][kind.channel()]
    }

    /// All three channels at sample `(i, j)`, resolving the sparse default.
    #[inline]
    pub fn map_texel(&self, resolution: u32, i: u32, j: u32) -> [f32; DATA_MAP_CHANNELS] {
        if self.maps.is_empty() {
            return DEFAULT_DATA_MAP;
        }
        let r = resolution.max(1);
        let i = i.min(r - 1);
        let j = j.min(r - 1);
        self.maps[(j * r + i) as usize]
    }

    /// Materialize the full `resolution²` data-map buffer (filled with
    /// [`DEFAULT_DATA_MAP`]) if the tile is still on the sparse default, then
    /// return a mutable handle. An erosion write-back calls this before writing.
    #[inline]
    pub fn ensure_maps(&mut self, resolution: u32) -> &mut [[f32; DATA_MAP_CHANNELS]] {
        if self.maps.is_empty() {
            self.maps = vec![DEFAULT_DATA_MAP; (resolution * resolution) as usize];
        }
        &mut self.maps
    }

    /// Reset the tile to the never-eroded sparse default (drops any accumulated
    /// data maps), so it re-serializes byte-identically to a tile erosion never
    /// touched. Used by erosion **undo** when a bake had materialized the buffer.
    #[inline]
    pub fn clear_maps(&mut self) {
        self.maps = Vec::new();
    }

    /// Write all three data-map channels at sample `(i, j)`, materializing the
    /// buffer first. Out-of-range indices are ignored.
    #[inline]
    pub fn set_map_texel(
        &mut self,
        resolution: u32,
        i: u32,
        j: u32,
        texel: [f32; DATA_MAP_CHANNELS],
    ) {
        let r = resolution.max(1);
        if i < r && j < r {
            self.ensure_maps(resolution)[(j * r + i) as usize] = texel;
        }
    }

    /// Length of the stored data-map buffer (`0` for the sparse default). Used by
    /// the terrain's serde length validation.
    #[inline]
    pub fn maps_len(&self) -> usize {
        self.maps.len()
    }

    // ── biome ids (P19.2) ───────────────────────────────────────────────────

    /// The raw biome-id buffer: either empty (unpainted — every sample is
    /// [`UNASSIGNED_BIOME`]) or `resolution²` row-major `u8`. Prefer
    /// [`biome_sample`](TerrainTile::biome_sample) for a value that already
    /// resolves the empty case.
    #[inline]
    pub fn biomes(&self) -> &[u8] {
        &self.biomes
    }

    /// `true` when the tile stores no biome ids — i.e. **nothing has been
    /// painted**. This is the byte-stable default (such a tile serializes without
    /// a `biomes` field at all in JSON/TOML, and as a single zero-length count in
    /// bincode).
    #[inline]
    pub fn biomes_are_default(&self) -> bool {
        self.biomes.is_empty()
    }

    /// The biome id at sample `(i, j)`, resolving the sparse default (an unpainted
    /// tile reads [`UNASSIGNED_BIOME`] everywhere). Out-of-range indices clamp to
    /// the edge.
    #[inline]
    pub fn biome_sample(&self, resolution: u32, i: u32, j: u32) -> u8 {
        if self.biomes.is_empty() {
            return DEFAULT_BIOME;
        }
        let r = resolution.max(1);
        let i = i.min(r - 1);
        let j = j.min(r - 1);
        self.biomes[(j * r + i) as usize]
    }

    /// Materialize the full `resolution²` biome buffer (filled with
    /// [`DEFAULT_BIOME`]) if the tile is still on the sparse default, then return
    /// a mutable handle. Biome painting calls this before writing samples.
    #[inline]
    pub fn ensure_biomes(&mut self, resolution: u32) -> &mut [u8] {
        if self.biomes.is_empty() {
            self.biomes = vec![DEFAULT_BIOME; (resolution * resolution) as usize];
        }
        &mut self.biomes
    }

    /// Reset the tile to the unpainted sparse default (drops any painted biome
    /// ids), so it re-serializes byte-identically to a tile the biome brush never
    /// touched. Used by biome-paint **undo** when a stroke had materialized the
    /// buffer.
    #[inline]
    pub fn clear_biomes(&mut self) {
        self.biomes = Vec::new();
    }

    /// Write the biome id at sample `(i, j)`, materializing the buffer first.
    /// Out-of-range indices are ignored.
    #[inline]
    pub fn set_biome_sample(&mut self, resolution: u32, i: u32, j: u32, biome: u8) {
        let r = resolution.max(1);
        if i < r && j < r {
            self.ensure_biomes(resolution)[(j * r + i) as usize] = biome;
        }
    }

    /// Length of the stored biome buffer (`0` for the sparse default). Used by the
    /// terrain's serde length validation.
    #[inline]
    pub fn biomes_len(&self) -> usize {
        self.biomes.len()
    }

    /// Inclusive `(min, max)` of the tile's `f32` offsets (for AABB culling and
    /// height-texture range normalization). An empty buffer yields `(0, 0)`.
    pub fn height_bounds(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &h in &self.heights {
            lo = lo.min(h);
            hi = hi.max(h);
        }
        if lo.is_finite() {
            (lo, hi)
        } else {
            (0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpainted_tile_serializes_without_weights_field() {
        // Byte-stability: a tile that has never been painted must encode exactly
        // like the pre-P10.4 two-field form (no `weights` key at all).
        let tile = TerrainTile::flat(4, DVec3::ZERO);
        let json = serde_json::to_string(&tile).unwrap();
        assert!(
            !json.contains("weights"),
            "unpainted tile leaked a weights field: {json}"
        );
    }

    #[test]
    fn old_bytes_decode_with_default_weights() {
        // A pre-P10.4 serialized tile (origin + heights only) decodes with the
        // sparse default weights, and re-serializes byte-identically (round-trip
        // both directions).
        let old = r#"{"origin":[0.0,0.0,0.0],"heights":[0.0,1.0,2.0,3.0]}"#;
        let tile: TerrainTile = serde_json::from_str(old).unwrap();
        assert!(tile.weights_are_default());
        assert_eq!(tile.weight_sample(2, 0, 0), DEFAULT_WEIGHT);
        assert_eq!(tile.weight_sample(2, 1, 1), DEFAULT_WEIGHT);
        // Re-serialization matches the old form byte-for-byte.
        assert_eq!(serde_json::to_string(&tile).unwrap(), old);
    }

    #[test]
    fn painted_tile_round_trips_weights() {
        // A materialized/painted tile appends the weights field and round-trips.
        let mut tile = TerrainTile::flat(2, DVec3::new(1.0, 2.0, 3.0));
        tile.set_weight_sample(2, 0, 0, [10, 200, 45, 0]);
        tile.set_weight_sample(2, 1, 1, [0, 0, 128, 127]);
        assert!(!tile.weights_are_default());
        let json = serde_json::to_string(&tile).unwrap();
        assert!(json.contains("weights"));
        let back: TerrainTile = serde_json::from_str(&json).unwrap();
        assert_eq!(tile, back);
        assert_eq!(back.weight_sample(2, 0, 0), [10, 200, 45, 0]);
        assert_eq!(back.weight_sample(2, 1, 1), [0, 0, 128, 127]);
        // Idempotent re-encode.
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
    }

    #[test]
    fn clear_weights_restores_sparse_default() {
        let mut tile = TerrainTile::flat(2, DVec3::ZERO);
        tile.set_weight_sample(2, 0, 0, [0, 255, 0, 0]);
        assert!(!tile.weights_are_default());
        tile.clear_weights();
        assert!(tile.weights_are_default());
        assert!(!serde_json::to_string(&tile).unwrap().contains("weights"));
    }

    /// bincode is **non-self-describing**, so the skip-empty JSON form would
    /// desync it (a `.inf_lvl` terrain persists via bincode, P10.6). The
    /// format-aware serde must round-trip both an unpainted and a painted tile
    /// through bincode, byte-identically on re-encode.
    #[test]
    fn tile_round_trips_through_bincode() {
        let cfg = bincode::config::standard();
        for mut tile in [TerrainTile::flat(2, DVec3::new(1.0, 2.0, 3.0)), {
            let mut t = TerrainTile::flat(2, DVec3::ZERO);
            t.set_weight_sample(2, 0, 0, [10, 200, 45, 0]);
            t.set_weight_sample(2, 1, 1, [0, 0, 128, 127]);
            t
        }] {
            // Give the unpainted tile some heights so both cases carry data.
            tile.set_sample(2, 0, 0, 5.0);
            let bytes = bincode::serde::encode_to_vec(&tile, cfg).unwrap();
            let (back, _): (TerrainTile, usize) =
                bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
            assert_eq!(tile, back, "bincode round trip must preserve the tile");
            let bytes2 = bincode::serde::encode_to_vec(&back, cfg).unwrap();
            assert_eq!(bytes, bytes2, "bincode re-encode is byte-identical");
        }
    }

    // ── the frozen generation-1 wire type (P19.1) ────────────────────────────

    use crate::data::{TerrainData, TerrainDataFrozenV1};

    /// A frozen heightfield with one well-formed 2×2 tile.
    fn frozen_ok() -> TerrainDataFrozenV1 {
        TerrainDataFrozenV1 {
            tile_resolution: 2,
            meters_per_sample: 1.0,
            tiles: vec![(
                0,
                0,
                TerrainTileFrozenV1 {
                    origin: DVec3::ZERO,
                    heights: vec![1.0, 2.0, 3.0, 4.0],
                    weights: Vec::new(),
                },
            )],
        }
    }

    fn bincode_cfg() -> impl bincode::config::Config {
        bincode::config::standard()
    }

    /// **A corrupt legacy height buffer is a decode ERROR, not a hole.**
    ///
    /// The live [`TerrainData`] decoder has always rejected a tile whose height
    /// buffer does not match the declared resolution. The frozen record is the
    /// path every pre-P19.1 payload now takes, so it has to reject it too — the
    /// alternative is a corrupt file that loads as a terrain full of holes and
    /// then gets *saved back* over the original.
    #[test]
    fn a_frozen_tile_with_wrong_height_count_fails_to_decode() {
        let mut bad = frozen_ok();
        bad.tiles[0].2.heights.pop(); // 3 of 4
        for label in ["bincode", "json"] {
            let err = if label == "bincode" {
                let bytes = bincode::serde::encode_to_vec(&bad, bincode_cfg()).unwrap();
                bincode::serde::decode_from_slice::<TerrainDataFrozenV1, _>(&bytes, bincode_cfg())
                    .err()
                    .map(|e| e.to_string())
            } else {
                let json = serde_json::to_string(&bad).unwrap();
                serde_json::from_str::<TerrainDataFrozenV1>(&json)
                    .err()
                    .map(|e| e.to_string())
            };
            let err =
                err.unwrap_or_else(|| panic!("{label}: a short height buffer must not decode"));
            assert!(
                err.contains("expected 4"),
                "{label}: unhelpful rejection: {err}"
            );
        }
        // The live decoder rejects the equivalent bytes for the same reason —
        // the frozen record is not allowed to be more permissive than the type
        // it stands in for.
        let live = serde_json::to_string(&serde_json::json!({
            "tile_resolution": 2,
            "meters_per_sample": 1.0,
            "tiles": [[0, 0, {"origin": [0.0, 0.0, 0.0], "heights": [1.0, 2.0, 3.0]}]],
        }))
        .unwrap();
        assert!(serde_json::from_str::<TerrainData>(&live).is_err());
    }

    /// **A short-but-non-empty legacy weight buffer is a decode ERROR, not a
    /// latent panic.**
    ///
    /// This is the sharp one. [`TerrainTile::weight_sample`] resolves only the
    /// *empty* case and indexes otherwise, so a weight buffer with 1 ≤ len < res²
    /// is an out-of-bounds index the first time anything reads or paints the
    /// tile. Rejecting it at the door is what keeps that unreachable — see
    /// [`a_decoded_terrain_can_always_be_painted`].
    #[test]
    fn a_frozen_tile_with_short_weights_fails_to_decode() {
        let mut bad = frozen_ok();
        bad.tiles[0].2.weights = vec![DEFAULT_WEIGHT; 2]; // 2 of 4 — the OOB shape
        for label in ["bincode", "json"] {
            let err = if label == "bincode" {
                let bytes = bincode::serde::encode_to_vec(&bad, bincode_cfg()).unwrap();
                bincode::serde::decode_from_slice::<TerrainDataFrozenV1, _>(&bytes, bincode_cfg())
                    .err()
                    .map(|e| e.to_string())
            } else {
                let json = serde_json::to_string(&bad).unwrap();
                serde_json::from_str::<TerrainDataFrozenV1>(&json)
                    .err()
                    .map(|e| e.to_string())
            };
            let err =
                err.unwrap_or_else(|| panic!("{label}: a short weight buffer must not decode"));
            assert!(
                err.contains("weight samples"),
                "{label}: unhelpful rejection: {err}"
            );
        }

        // An EMPTY weight buffer is the sparse default and must still decode —
        // the check is "0 or exactly res²", not "always res²".
        let ok = frozen_ok();
        let bytes = bincode::serde::encode_to_vec(&ok, bincode_cfg()).unwrap();
        let (back, _): (TerrainDataFrozenV1, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_cfg()).unwrap();
        assert_eq!(back, ok);
    }

    /// Run the exact calls the splat brush makes over every sample of every tile.
    /// Panics on an out-of-bounds weight index — which is the point.
    fn sweep_the_paint_path(data: &mut TerrainData) {
        let res = data.tile_resolution();
        let coords: Vec<(i32, i32)> = data.tiles().map(|(&c, _)| c).collect();
        for c in coords {
            for j in 0..res {
                for i in 0..res {
                    // Read (the brush's gather half) …
                    let w = data.get_tile(c).unwrap().weight_sample(res, i, j);
                    // … and write (the brush's commit half, which materializes).
                    data.get_tile_mut(c)
                        .unwrap()
                        .set_weight_sample(res, i, j, w);
                }
            }
        }
    }

    /// **The hazard, pinned.** A tile whose weight buffer is short but non-empty
    /// indexes off the end the first time the paint path touches it —
    /// [`TerrainTile::weight_sample`] resolves only the *empty* case. This is
    /// what the frozen record's `Deserialize` check exists to keep unreachable,
    /// and it is asserted here so the check can never be "simplified away" as
    /// belt-and-braces.
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn a_short_weight_buffer_would_index_out_of_bounds() {
        let mut data = TerrainData::new(2, 1.0);
        // Built by hand, bypassing every door — exactly the state the decoder is
        // responsible for never producing.
        let tile = TerrainTileFrozenV1 {
            origin: DVec3::ZERO,
            heights: vec![0.0; 4],
            weights: vec![DEFAULT_WEIGHT; 2],
        }
        .into_current();
        let _ = data.insert_tile((0, 0), tile);
        sweep_the_paint_path(&mut data);
    }

    /// **…and it is unreachable through the decoder.** Every weight buffer a
    /// legacy payload can carry either fails to decode (the short case above) or
    /// lifts to a terrain the paint path sweeps without panicking.
    ///
    /// Mutation-verified: deleting the weight-length check in
    /// `TerrainDataFrozenV1::deserialize` makes the short case decode, and this
    /// test then fails with the index panic instead of the expected decode error.
    #[test]
    fn a_decoded_terrain_can_always_be_painted() {
        let cases: [(&str, Vec<[u8; 4]>, bool); 3] = [
            ("sparse default", Vec::new(), true),
            ("materialized", vec![[10, 20, 30, 195]; 4], true),
            ("short — must be rejected", vec![DEFAULT_WEIGHT; 2], false),
        ];
        for (label, weights, decodes) in cases {
            let mut src = frozen_ok();
            src.tiles[0].2.weights = weights;
            let bytes = bincode::serde::encode_to_vec(&src, bincode_cfg()).unwrap();
            let decoded =
                bincode::serde::decode_from_slice::<TerrainDataFrozenV1, _>(&bytes, bincode_cfg());
            assert_eq!(decoded.is_ok(), decodes, "{label}: wrong decode verdict");
            let Ok((frozen, _)) = decoded else { continue };
            let mut data = frozen.into_current();
            sweep_the_paint_path(&mut data);
            assert_eq!(
                data.get_tile((0, 0)).unwrap().weights_len(),
                4,
                "{label}: the sweep must have materialized the buffer"
            );
        }
    }

    /// Lifting a legacy terrain preserves **every** tile: no silent drops, and
    /// nothing is dirtied (a load is not an edit).
    #[test]
    fn lifting_a_legacy_terrain_keeps_every_tile() {
        let mut src = frozen_ok();
        src.tiles.push((
            3,
            -2,
            TerrainTileFrozenV1 {
                origin: DVec3::new(3.0, 7.5, -2.0),
                heights: vec![9.0; 4],
                weights: vec![DEFAULT_WEIGHT; 4],
            },
        ));
        let data = src.clone().into_current();
        assert_eq!(data.tile_count(), 2, "a lift must not drop pages");
        assert!(!data.has_dirty_tiles(), "loading is not an edit");
        // The `f64` height anchor rides through untouched (never re-snapped).
        assert_eq!(data.get_tile((3, -2)).unwrap().origin.y, 7.5);
        // …and the maps come up empty — never eroded, what a legacy level meant —
        // as do the biome ids (nothing painted).
        assert!(data.data_maps_are_default());
        assert!(data.biomes_are_default());
        // Round trip back down is lossless for everything v1 could express.
        assert_eq!(TerrainDataFrozenV1::from_current(&data), src);
    }

    // ── the biome layer's length contract (P19.2) ────────────────────────────
    //
    // The P19.2 review found the biome check in `TerrainData::deserialize` was
    // load-bearing but UNGUARDED: deleting it left all 285 tests green. These
    // three are the biome twins of the weight trio above, and they exist for the
    // identical reason — `biome_sample` resolves only the *empty* case and indexes
    // otherwise, so a buffer with 1 ≤ len < res² is an out-of-bounds index the
    // first time anything reads or paints the tile.
    //
    // The bypass door differs, and that difference is the point. A weight buffer
    // can be smuggled in through `TerrainTileFrozenV1`'s public fields; biome ids
    // have no frozen record (no historical container could carry them), so the way
    // in is a **hand-built tile** — which is exactly what a corrupt payload
    // decodes to, because a `TerrainTile` cannot check its own buffers (it does
    // not know the terrain's resolution). `TerrainData`'s `Deserialize` is
    // therefore the ONLY door, on both codecs.

    /// A tile built by hand, bypassing every door — exactly the state the decoder
    /// is responsible for never producing. Only reachable from inside this module,
    /// which is why the check has to live at the terrain level.
    fn tile_with_biomes(biomes: Vec<u8>) -> TerrainTile {
        TerrainTile {
            origin: DVec3::ZERO,
            heights: vec![0.0; 4],
            weights: Vec::new(),
            maps: Vec::new(),
            biomes,
        }
    }

    /// Run the exact calls the biome brush makes over every sample of every tile.
    /// Panics on an out-of-bounds biome index — which is the point.
    fn sweep_the_biome_paint_path(data: &mut TerrainData) {
        let res = data.tile_resolution();
        let coords: Vec<(i32, i32)> = data.tiles().map(|(&c, _)| c).collect();
        for c in coords {
            for j in 0..res {
                for i in 0..res {
                    // Read (the brush's gather half) …
                    let b = data.get_tile(c).unwrap().biome_sample(res, i, j);
                    // … and write (the brush's commit half, which materializes).
                    data.get_tile_mut(c).unwrap().set_biome_sample(res, i, j, b);
                }
            }
        }
    }

    /// **The hazard, pinned.** A tile whose biome buffer is short but non-empty
    /// indexes off the end the first time the paint path touches it. This is what
    /// the terrain decoder's biome-length check exists to keep unreachable, and it
    /// is asserted here so the check can never be "simplified away" as
    /// belt-and-braces.
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn a_short_biome_buffer_would_index_out_of_bounds() {
        let mut data = TerrainData::new(2, 1.0);
        let _ = data.insert_tile((0, 0), tile_with_biomes(vec![0; 2]));
        sweep_the_biome_paint_path(&mut data);
    }

    /// **…and it is unreachable through the decoder**, on BOTH codecs. Every
    /// biome buffer a payload can carry either fails to decode (the short case)
    /// or lifts to a terrain the biome paint path sweeps without panicking.
    ///
    /// Mutation-verified: deleting the `biomes_len` check in
    /// `TerrainData::deserialize` makes the short case decode, and this test then
    /// fails **with the index panic** — the sweep runs before the verdict is
    /// asserted precisely so the mutation surfaces as the real consequence
    /// (out-of-bounds on the paint path) rather than as a bare verdict mismatch.
    ///
    /// [`a_short_biome_buffer_would_index_out_of_bounds`] keeps passing under that
    /// mutation, by construction: it bypasses the decoder entirely, which is its
    /// whole job. The pair is what covers the contract — one shows the hazard is
    /// real, this one shows the decoder is the only door to it.
    #[test]
    fn a_decoded_terrain_can_always_be_biome_painted() {
        let cases: [(&str, Vec<u8>, bool); 3] = [
            ("sparse default", Vec::new(), true),
            ("materialized", vec![7; 4], true),
            ("short — must be rejected", vec![0; 2], false),
        ];
        for (label, biomes, decodes) in cases {
            let mut src = TerrainData::new(2, 1.0);
            // Serialization has no validation, so this is exactly the shape a
            // corrupt file on disk has.
            let _ = src.insert_tile((0, 0), tile_with_biomes(biomes.clone()));

            for codec in ["bincode", "json"] {
                let decoded: Result<TerrainData, String> = if codec == "bincode" {
                    let bytes = bincode::serde::encode_to_vec(&src, bincode_cfg()).unwrap();
                    bincode::serde::decode_from_slice::<TerrainData, _>(&bytes, bincode_cfg())
                        .map(|(d, _)| d)
                        .map_err(|e| e.to_string())
                } else {
                    let json = serde_json::to_string(&src).unwrap();
                    serde_json::from_str::<TerrainData>(&json).map_err(|e| e.to_string())
                };

                match &decoded {
                    // Sweep FIRST — see the mutation note above.
                    Ok(data) => {
                        let mut data = data.clone();
                        sweep_the_biome_paint_path(&mut data);
                        assert_eq!(
                            data.get_tile((0, 0)).unwrap().biomes_len(),
                            4,
                            "{label} / {codec}: the sweep must have materialized the buffer"
                        );
                    }
                    Err(err) => assert!(
                        err.contains("biome samples"),
                        "{label} / {codec}: unhelpful rejection: {err}"
                    ),
                }
                assert_eq!(
                    decoded.is_ok(),
                    decodes,
                    "{label} / {codec}: wrong decode verdict"
                );
            }
        }
    }

    /// **The two-ladder tripwire.** Each frozen generation stands in for a tile
    /// layout that TWO independently-versioned containers point at (see
    /// [`TerrainTileFrozenV1`]'s generation table). If either container bumps past
    /// the version in that table without a new frozen generation, its old payloads
    /// start decoding through the wrong wire type — silently, positionally, into
    /// the next record's bytes.
    ///
    /// `inf-terrain` can only see its own half; `inf-scene` and the editor codec
    /// each carry the mirror assertion for the scene half.
    #[test]
    fn frozen_tile_generations_are_pinned_to_both_ladders() {
        assert_eq!(
            crate::asset::TERRAIN_ASSET_SCHEMA_VERSION,
            4,
            "the .inf_terrain header moved. Generation-1 frozen tiles cover header \
             versions 1..=2, generation-2 covers 3, and the current TerrainTile covers 4. \
             If the tile layout changed again, add TerrainTileFrozenV3 and extend \
             `decode_tile_at`; if only the header changed, update this pin and \
             TerrainTileFrozenV1's generation table."
        );
        // The version→wire-type mapping the pin is really about. Each generation
        // is exactly the previous one plus its own sparse (zero-length) layer, so
        // the byte deltas below also price "sparse is free" at the wire.
        let tile = TerrainTile::flat(2, DVec3::ZERO);
        let gen1 =
            bincode::serde::encode_to_vec(TerrainTileFrozenV1::from_current(&tile), bincode_cfg())
                .unwrap();
        let gen2 =
            bincode::serde::encode_to_vec(TerrainTileFrozenV2::from_current(&tile), bincode_cfg())
                .unwrap();
        let current = crate::asset::encode_tile(&tile).unwrap();
        assert_eq!(
            gen2.len(),
            gen1.len() + 1,
            "generation 2 is generation 1 plus exactly the sparse maps count"
        );
        assert_eq!(
            current.len(),
            gen2.len() + 1,
            "the current tile is generation 2 plus exactly the sparse biomes count"
        );
        for (bytes, versions, label) in [
            (&gen1, &[0u32, 1, 2][..], "generation-1"),
            (&gen2, &[3][..], "generation-2"),
            (&current, &[4][..], "current"),
        ] {
            for &v in versions {
                let got = crate::asset::decode_tile_at(bytes, v)
                    .unwrap_or_else(|e| panic!("header v{v} must decode {label} bytes: {e}"));
                assert_eq!(
                    got, tile,
                    "header v{v} decoded {label} bytes to a wrong tile"
                );
            }
        }
    }

    /// **Generation 2 is a real, distinct wire shape.** A v15/v3 payload's bytes
    /// must not be fed to the live tile — the biome layer would read off the end —
    /// and a v15 payload must lift with the maps intact but the biomes empty.
    #[test]
    fn generation_two_carries_maps_and_lifts_with_empty_biomes() {
        let mut tile = TerrainTile::flat(2, DVec3::new(1.0, 2.0, 3.0));
        tile.set_map_texel(2, 0, 0, [1.5, 2.5, 3.5]);
        tile.set_biome_sample(2, 1, 1, 7);

        let frozen = TerrainTileFrozenV2::from_current(&tile);
        assert_eq!(frozen.maps.len(), 4, "the maps ride generation 2");
        let lifted = frozen.clone().into_current();
        assert!(
            lifted.biomes_are_default(),
            "a pre-P19.2 tile means: nothing painted"
        );
        assert_eq!(lifted.map_texel(2, 0, 0), [1.5, 2.5, 3.5]);

        // Round trip through both codecs, byte-identical on re-encode.
        for human in [true, false] {
            let (bytes, back) = if human {
                let s = serde_json::to_string(&frozen).unwrap();
                let back: TerrainTileFrozenV2 = serde_json::from_str(&s).unwrap();
                (s.into_bytes(), back)
            } else {
                let b = bincode::serde::encode_to_vec(&frozen, bincode_cfg()).unwrap();
                let (back, _): (TerrainTileFrozenV2, usize) =
                    bincode::serde::decode_from_slice(&b, bincode_cfg()).unwrap();
                (b, back)
            };
            assert_eq!(back, frozen);
            let again = if human {
                serde_json::to_string(&back).unwrap().into_bytes()
            } else {
                bincode::serde::encode_to_vec(&back, bincode_cfg()).unwrap()
            };
            assert_eq!(bytes, again, "re-encode must be byte-identical");
        }
    }

    // ── biome ids (P19.2) ────────────────────────────────────────────────────

    /// An unpainted tile leaks no `biomes` field, and re-serializes exactly like
    /// the pre-P19.2 form.
    #[test]
    fn unpainted_tile_serializes_without_biomes_field() {
        let tile = TerrainTile::flat(4, DVec3::ZERO);
        let json = serde_json::to_string(&tile).unwrap();
        assert!(
            !json.contains("biomes"),
            "unpainted tile leaked a biomes field: {json}"
        );
    }

    /// Painting → round-trip → clearing gets back to byte-identical sparse space.
    #[test]
    fn painted_biomes_round_trip_and_clear_restores_the_sparse_default() {
        let mut tile = TerrainTile::flat(2, DVec3::ZERO);
        assert!(tile.biomes_are_default());
        assert_eq!(tile.biome_sample(2, 1, 1), UNASSIGNED_BIOME);

        tile.set_biome_sample(2, 0, 0, 3);
        tile.set_biome_sample(2, 1, 1, 255);
        assert!(!tile.biomes_are_default());
        assert_eq!(tile.biomes_len(), 4);
        assert_eq!(tile.biomes(), &[3, 0, 0, 255]);

        let json = serde_json::to_string(&tile).unwrap();
        assert!(json.contains("biomes"));
        let back: TerrainTile = serde_json::from_str(&json).unwrap();
        assert_eq!(tile, back);
        assert_eq!(back.biome_sample(2, 1, 1), 255);

        let cfg = bincode_cfg();
        let bytes = bincode::serde::encode_to_vec(&tile, cfg).unwrap();
        let (back, _): (TerrainTile, usize) =
            bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(tile, back);

        let mut cleared = tile.clone();
        cleared.clear_biomes();
        assert!(cleared.biomes_are_default());
        assert_eq!(
            serde_json::to_string(&cleared).unwrap(),
            serde_json::to_string(&TerrainTile::flat(2, DVec3::ZERO)).unwrap()
        );
    }

    /// Out-of-range writes are ignored (they never materialize a buffer either)
    /// and out-of-range reads clamp to the edge — the same contract the other
    /// three layers keep.
    #[test]
    fn biome_indices_clamp_and_ignore_like_the_other_layers() {
        let mut tile = TerrainTile::flat(2, DVec3::ZERO);
        tile.set_biome_sample(2, 9, 9, 42);
        assert!(
            tile.biomes_are_default(),
            "an out-of-range write must not materialize the buffer"
        );
        tile.set_biome_sample(2, 1, 1, 42);
        assert_eq!(tile.biome_sample(2, 99, 99), 42, "reads clamp to the edge");
    }
}
