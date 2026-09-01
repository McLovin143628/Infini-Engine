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
///
/// # THE ORDERING LAW: new kinds go at the **end**, always
///
/// This enum is **nested inside a persisted wire form** — `inf_pcg::SamplerDef::DataMap`
/// carries it, and a `.inf_pcg` is bincode, which encodes an externally-tagged
/// enum as its **declaration index**. Inserting a kind here (the deferred fourth
/// *thermal* channel is the obvious candidate — see the P19.1 remainders) would
/// silently renumber every kind after it and turn a committed `mask.wear` node
/// into a `mask.deposition`. Appending costs nothing;
/// [`channel`](Self::channel) already states the storage order separately, so the
/// two never have to agree by accident. Pinned by
/// `inf_pcg::rules::sampler_variant_discriminants_are_frozen`.
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

    /// Whether this channel is **extensive** — its value scales with the area it
    /// covers — as opposed to *intensive* (a per-area value that does not).
    ///
    /// This is the one place the dimension is stated, and it is what every
    /// area-changing reduction must branch on. [`Flow`](Self::Flow) is a volume
    /// (m³): a region twice the size shipped twice the water, so combining cells
    /// **sums**. [`Deposition`](Self::Deposition) and [`Wear`](Self::Wear) are
    /// metres of height moved: a region twice the size did not gain twice the
    /// height, so combining cells **averages** — which also preserves the volume
    /// integral, since `mean(h) · ΣA == Σ(h · A)` for equal-area children.
    ///
    /// The first caller is the LOD pyramid's per-layer reduction
    /// ([`crate::pyramid`]); a mip/resample path or a coarse-resolution PCG pass
    /// needs exactly the same branch, which is why it is a property of the kind
    /// rather than a rule written inside one reducer.
    #[inline]
    pub const fn is_extensive(self) -> bool {
        match self {
            DataMapKind::Flow => true,
            DataMapKind::Deposition | DataMapKind::Wear => false,
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
/// ## Hole mask (P21.2)
///
/// A fifth per-sample layer ([`holes`](TerrainTile::holes)): **one bit** per
/// sample saying "there is no heightfield here". A holed sample has no surface at
/// all — the clipmap discards it, [`height_at`](super::TerrainData::height_at)
/// returns `None` through it, and what a camera or a capsule finds below is
/// whatever the P21.1 voxel volume put there. It is the mechanism that lets a
/// cave mouth open in ground that is otherwise a heightfield.
///
/// Same sparse rule as every layer above it — an empty `Vec` means "no sample is
/// holed", so an un-carved tile costs zero per-sample bytes.
///
/// ### Why bits, not one `u8` per sample
///
/// The other four layers store a *value* per sample (a weight quad, three
/// accumulators, an id); this one stores a **predicate**, and a predicate packs.
/// At the default 257² tile:
///
/// * **one `u8` per sample** (the `biomes` shape): 66 049 B per holed tile, and
///   the same 66 049 B in the renderer's per-tile upload;
/// * **one bit per sample** (this): 8 257 B, an 8× cut in *both* places.
///
/// The layer is also unusually likely to be *present on many tiles at once* — a
/// cave system that crosses four tiles materializes four masks even though each
/// holds a few hundred set bits — so the multiplier lands on the common case, not
/// a corner. And the packed form is exactly what the GPU wants: the terrain pass
/// uploads it verbatim as `u32` words and tests one bit per fragment
/// (`inf_render::passes::terrain`), where a byte-per-sample layer would have paid
/// 8× the bandwidth to carry seven zero bits per sample.
///
/// The cost is that indexing is not `buf[j * res + i]`; nothing outside this
/// module does that arithmetic, because [`is_hole`](TerrainTile::is_hole) /
/// [`set_hole`](TerrainTile::set_hole) are the only accessors and the raw buffer
/// is documented as opaque packed bits. Undo does **not** inherit the packing:
/// [`HoleDelta`](crate::HoleDelta) patches are bounded by a brush footprint, so
/// they store a plain byte per sample and stay trivially diffable.
///
/// Serde: `origin` (as a portable `[f64; 3]`, since the workspace `glam` pin has
/// no `serde` feature) + a flat `heights` sequence, and — **only when non-empty**
/// (`skip_serializing_if`) — flat `weights`, `maps`, `biomes` and `holes`
/// sequences. An old tile (no `weights`/`maps`/`biomes`/`holes` field) decodes
/// with the defaults, and an unpainted, un-eroded, un-carved new tile serializes
/// without any of them, so existing human-readable bytes round-trip unchanged.
/// `resolution` is not stored on the tile (it is a terrain-wide constant on
/// [`super::TerrainData`]); the terrain validates `heights.len() == resolution²`
/// and, when present, `weights.len()` / `maps.len()` / `biomes.len() ==
/// resolution²` and `holes.len() == `[`hole_mask_bytes`]` on load.
///
/// **The bincode form is versioned at the container, not the tile.** bincode is
/// positional, so appending a layer is a wire-format change: a stream written
/// before P21.2 has five fields where this build reads six. Each historical
/// layout is therefore frozen — [`TerrainTileFrozenV1`] (pre-P19.1: no maps, no
/// biomes, no holes), [`TerrainTileFrozenV2`] (P19.1: maps only) and
/// [`TerrainTileFrozenV3`] (P19.2: maps + biomes, no holes) — and selected by
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
    /// [`hole_mask_bytes(resolution)`](hole_mask_bytes) **packed bits** (bit
    /// `n = j * resolution + i`, LSB-first within each byte), **or empty** for
    /// the un-carved default (no sample is holed) — see the type docs. Opaque
    /// outside this module: read it with [`is_hole`](TerrainTile::is_hole).
    holes: Vec<u8>,
}

/// Re-pack a tile's hole mask into **row-aligned `u32` words** — the layout a GPU
/// wants — returning an empty `Vec` for a tile with no holes.
///
/// The tile packs bit `j * resolution + i`, which is dense but does not align a
/// row to a word boundary unless the resolution happens to be a multiple of 32.
/// A fragment shader would then need a division by a non-constant to find its
/// word. Row-aligned, the index is `(i >> 5, j)` — free — at a cost of at most
/// 31 wasted bits per row on a carved tile, which is nothing against the
/// `resolution²` bits it is carrying.
///
/// Lives here, beside the packing it inverts, so the two cannot drift; the
/// renderer only ever sees the `Vec<u32>` and never learns the tile's own
/// layout. Empty out means "nothing is holed", the same sparse default the layer
/// itself carries.
pub fn pack_hole_rows(tile: &TerrainTile, resolution: u32) -> Vec<u32> {
    if !tile.has_holes() {
        return Vec::new();
    }
    let res = resolution.max(1);
    let words = res.div_ceil(32) as usize;
    let mut out = vec![0u32; words * res as usize];
    for j in 0..res {
        for i in 0..res {
            if tile.is_hole(res, i, j) {
                out[j as usize * words + (i >> 5) as usize] |= 1u32 << (i & 31);
            }
        }
    }
    out
}

/// Bytes a packed hole mask occupies for a `resolution × resolution` tile:
/// `ceil(resolution² / 8)`.
///
/// The bit for sample `(i, j)` is `n = j * resolution + i`, living in byte
/// `n / 8` at bit `n % 8` (LSB-first). Bits past `resolution²` in the final byte
/// are always zero — [`TerrainTile::set_hole`] never addresses them — so the
/// buffer is byte-stable and two tiles with the same holes serialize identically.
#[inline]
pub const fn hole_mask_bytes(resolution: u32) -> usize {
    let n = (resolution as usize) * (resolution as usize);
    n.div_ceil(8)
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    holes: Vec<u8>,
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
    #[serde(default)]
    holes: Vec<u8>,
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
/// | [`TerrainTileFrozenV3`] | + biome ids (P19.2) | **v16 …** | **v4** |
/// | [`TerrainTile`] (live) | + hole mask (P21.2) | *(none yet)* | **v5** |
///
/// Naming generation 1 `TerrainTileV14` — as the first cut did — leaks the
/// *scene* codec's numbering into `inf-terrain`'s public API and silently implies
/// the asset container agrees, which it does not (it calls the same bytes "v2").
/// The generation counter belongs to the **tile**: each tile-layout change adds
/// one row here and **both** container columns gain a version.
/// `frozen_tile_generations_are_pinned_to_both_ladders` is the tripwire that fails
/// if either container bumps past its row without a new generation.
///
/// # THE EMPTY CELL: generation 4 has no `.inf_lvl` column (P21.2)
///
/// Every earlier row advanced both containers at once. Generation 4 advances only
/// the asset one, because the scene schema was **already frozen at v19** when the
/// hole layer landed (Phase 21 spent its single scene bump on the P21.1
/// `VoxelVolume` component), and bincode is positional: a sixth tile field in the
/// `.inf_lvl` stream is a v20, full stop.
///
/// So [`super::TerrainData`]'s wire form — the *only* path a tile takes into an
/// `.inf_lvl` — is pinned at **generation 3**: it serializes every tile through
/// [`TerrainTileFrozenV3`] and lifts back with an empty hole mask. `.inf_lvl` v19
/// bytes are therefore byte-identical to v16's for the same terrain, which is the
/// point. `.inf_terrain` encodes tiles *individually*
/// ([`crate::asset::encode_tile`]), never through `TerrainData`, so its v5 blobs
/// carry the holes.
///
/// The consequence, stated plainly rather than discovered later: **an
/// `.inf_lvl`-inline (non-asset-backed) terrain does not persist its holes.**
/// Carving one is a live, undoable edit that survives until the level is written,
/// and then the mask is gone. Holes persist on a terrain backed by a
/// `.inf_terrain` asset — which is the streaming path every carve tool targets,
/// and the container the cook's P21.2 advisories read. The next scene bump fills
/// the empty cell in; `terrain_data_wire_is_pinned_at_generation_three` is the
/// tripwire that fails when someone tries to fill it early.
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
    /// eroded), the biome ids are **empty** (unassigned) and the hole mask is
    /// **empty** (nothing carved) — which is exactly what a pre-P19.1 tile meant.
    pub fn into_current(self) -> TerrainTile {
        TerrainTile {
            origin: self.origin,
            heights: self.heights,
            weights: self.weights,
            maps: Vec::new(),
            biomes: Vec::new(),
            holes: Vec::new(),
        }
    }

    /// Project a live tile onto the frozen shape, **dropping** its data maps,
    /// biome ids and hole mask (the downgrade-bless direction — a pre-P19.1
    /// container cannot carry any of them).
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
    /// is [`UNASSIGNED_BIOME`], and so does the hole mask — exactly what a
    /// pre-P19.2 tile meant.
    pub fn into_current(self) -> TerrainTile {
        TerrainTile {
            origin: self.origin,
            heights: self.heights,
            weights: self.weights,
            maps: self.maps,
            biomes: Vec::new(),
            holes: Vec::new(),
        }
    }

    /// Project a live tile onto the frozen shape, **dropping** its biome ids and
    /// hole mask (the downgrade-bless direction).
    pub fn from_current(tile: &TerrainTile) -> Self {
        Self {
            origin: tile.origin,
            heights: tile.heights.clone(),
            weights: tile.weights.clone(),
            maps: tile.maps.clone(),
        }
    }

    /// Lift one generation, to [`TerrainTileFrozenV3`] with empty biome ids.
    ///
    /// The chained-ladder hop, for exactly the reason
    /// [`TerrainTileFrozenV1::into_v2`] gives: a codec that walks its versions one
    /// rung at a time needs the single step, not a jump to the live type.
    pub fn into_v3(self) -> TerrainTileFrozenV3 {
        TerrainTileFrozenV3 {
            origin: self.origin,
            heights: self.heights,
            weights: self.weights,
            maps: self.maps,
            biomes: Vec::new(),
        }
    }
}

/// **Frozen tile wire layout, generation 3** — `origin + heights + weights +
/// maps + biomes`, with no hole mask. The shape every terrain tile had between
/// P19.2 and P21.2 (`.inf_lvl` v16+, `.inf_terrain` header v4).
///
/// See [`TerrainTileFrozenV1`]'s generation table for why the name counts tile
/// generations rather than container versions — and, for this rung in
/// particular, for **why its `.inf_lvl` column has no end**: the scene schema was
/// frozen at v19 when P21.2's hole mask landed, so `TerrainData`'s wire form is
/// pinned here and every `.inf_lvl` written by this build still holds generation-3
/// tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrainTileFrozenV3 {
    /// World position of sample `(0, 0)`.
    pub origin: DVec3,
    /// `resolution²` height offsets, row-major.
    pub heights: Vec<f32>,
    /// `resolution²` row-major splat weights, or empty for the sparse default.
    pub weights: Vec<[u8; 4]>,
    /// `resolution²` row-major erosion data maps, or empty for never-eroded.
    pub maps: Vec<[f32; DATA_MAP_CHANNELS]>,
    /// `resolution²` row-major biome ids, or empty for unpainted.
    pub biomes: Vec<u8>,
}

/// Human-readable wire form of [`TerrainTileFrozenV3`] — the exact pre-P21.2
/// [`TerrainTileRaw`].
#[derive(Serialize, Deserialize)]
struct TerrainTileFrozenV3Raw {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    weights: Vec<[u8; 4]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    maps: Vec<[f32; DATA_MAP_CHANNELS]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    biomes: Vec<u8>,
}

/// bincode wire form of [`TerrainTileFrozenV3`] — the exact pre-P21.2
/// [`TerrainTileBin`] (five positional fields, no `holes`).
#[derive(Serialize, Deserialize)]
struct TerrainTileFrozenV3Bin {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default)]
    weights: Vec<[u8; 4]>,
    #[serde(default)]
    maps: Vec<[f32; DATA_MAP_CHANNELS]>,
    #[serde(default)]
    biomes: Vec<u8>,
}

impl TerrainTileFrozenV3 {
    /// Lift to the live tile: the hole mask comes up **empty** (no sample is
    /// holed) — exactly what a pre-P21.2 tile meant.
    pub fn into_current(self) -> TerrainTile {
        TerrainTile {
            origin: self.origin,
            heights: self.heights,
            weights: self.weights,
            maps: self.maps,
            biomes: self.biomes,
            holes: Vec::new(),
        }
    }

    /// Project a live tile onto the frozen shape, **dropping** its hole mask.
    ///
    /// Unlike the earlier `from_current`s this is **not** test-only: it is the
    /// production write path for every `.inf_lvl`, because `TerrainData`'s wire
    /// form is pinned at this generation until the next scene bump (see the
    /// generation table). That is the one place holes are lost, and it is stated
    /// there.
    pub fn from_current(tile: &TerrainTile) -> Self {
        Self {
            origin: tile.origin,
            heights: tile.heights.clone(),
            weights: tile.weights.clone(),
            maps: tile.maps.clone(),
            biomes: tile.biomes.clone(),
        }
    }

    /// Length of the stored hole buffer this generation cannot carry: always `0`.
    /// Spelled out so the length-validation loops read the same for every layer.
    #[inline]
    pub fn holes_len(&self) -> usize {
        0
    }

    /// Length of the stored weight buffer (`0` for the sparse default), for the
    /// terrain's serde length validation — the frozen twin of
    /// [`TerrainTile::weights_len`].
    #[inline]
    pub fn weights_len(&self) -> usize {
        self.weights.len()
    }

    /// Length of the stored data-map buffer, for the same validation.
    #[inline]
    pub fn maps_len(&self) -> usize {
        self.maps.len()
    }

    /// Length of the stored biome buffer, for the same validation.
    #[inline]
    pub fn biomes_len(&self) -> usize {
        self.biomes.len()
    }

    /// The raw row-major height buffer, for the same validation.
    #[inline]
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }
}

impl Serialize for TerrainTileFrozenV3 {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            TerrainTileFrozenV3Raw {
                origin: self.origin.to_array(),
                heights: self.heights.clone(),
                weights: self.weights.clone(),
                maps: self.maps.clone(),
                biomes: self.biomes.clone(),
            }
            .serialize(s)
        } else {
            TerrainTileFrozenV3Bin {
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

impl<'de> Deserialize<'de> for TerrainTileFrozenV3 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            let raw = TerrainTileFrozenV3Raw::deserialize(d)?;
            Ok(Self {
                origin: DVec3::from_array(raw.origin),
                heights: raw.heights,
                weights: raw.weights,
                maps: raw.maps,
                biomes: raw.biomes,
            })
        } else {
            let raw = TerrainTileFrozenV3Bin::deserialize(d)?;
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
                holes: self.holes.clone(),
            }
            .serialize(s)
        } else {
            TerrainTileBin {
                origin: self.origin.to_array(),
                heights: self.heights.clone(),
                weights: self.weights.clone(),
                maps: self.maps.clone(),
                biomes: self.biomes.clone(),
                holes: self.holes.clone(),
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
                holes: raw.holes,
            })
        } else {
            let raw = TerrainTileBin::deserialize(d)?;
            Ok(Self {
                origin: DVec3::from_array(raw.origin),
                heights: raw.heights,
                weights: raw.weights,
                maps: raw.maps,
                biomes: raw.biomes,
                holes: raw.holes,
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
            holes: Vec::new(),
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
            holes: Vec::new(),
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
    ///
    /// **A non-finite height is dropped, not stored** — the same rule as its
    /// sibling [`HeightRegion::set_height`](crate::region::HeightRegion::set_height)
    /// (C4-35), which this door did not have (round-2 finding B3).
    ///
    /// The refusal that matters for an *imported* heightmap is at
    /// [`decode_rows`](crate::import::decode_rows), where it can name the
    /// offending pixel. This is the check that stands in front of the seven
    /// other writers — the brush, the delta replay, the pyramid fold, the
    /// analytic `from_fn` generators — none of which cross that door, and any
    /// of which can produce `inf - inf` on a tile a saturated edit left
    /// infinite. `encode_tile` bincodes whatever is in this buffer with no
    /// finiteness check of its own, so this is the last place before a
    /// committed `.inf_terrain`.
    #[inline]
    pub fn set_sample(&mut self, resolution: u32, i: u32, j: u32, height: f32) {
        if !height.is_finite() {
            return;
        }
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

    // ── hole mask (P21.2) ───────────────────────────────────────────────────

    /// The raw hole mask: either empty (nothing carved) or
    /// [`hole_mask_bytes(resolution)`](hole_mask_bytes) of **packed bits**. The
    /// packing is documented on [`hole_mask_bytes`], and this accessor exists for
    /// the two consumers that legitimately want the bits whole — the renderer's
    /// per-tile GPU upload and the serde length check. Everything else asks
    /// [`is_hole`](TerrainTile::is_hole).
    #[inline]
    pub fn holes(&self) -> &[u8] {
        &self.holes
    }

    /// `true` when the tile stores no hole mask — i.e. **nothing has been
    /// carved**. This is the byte-stable default (such a tile serializes without
    /// a `holes` field at all in JSON/TOML, and as a single zero-length count in
    /// bincode).
    ///
    /// Note the asymmetry with [`has_holes`](TerrainTile::has_holes): a tile whose
    /// mask was materialized and then cleared bit by bit is **not** on the default
    /// (it still pays its buffer) even though no sample is holed. Persistence
    /// cares about the former, queries about the latter, and healing a carve calls
    /// [`clear_holes`](TerrainTile::clear_holes) to collapse one into the other.
    #[inline]
    pub fn holes_are_default(&self) -> bool {
        self.holes.is_empty()
    }

    /// `true` when **some** sample of this tile is holed. `false` for a tile on
    /// the sparse default, and also for one whose materialized mask is all zeros.
    ///
    /// This is the predicate a renderer, a query or a cook advisory wants: it
    /// answers "is there a hole here" without caring how the buffer got that way.
    #[inline]
    pub fn has_holes(&self) -> bool {
        self.holes.iter().any(|&b| b != 0)
    }

    /// `true` when sample `(i, j)` is holed — there is no heightfield surface
    /// there. An un-carved tile reads `false` everywhere. Out-of-range indices
    /// clamp to the edge, matching every other per-sample accessor.
    #[inline]
    pub fn is_hole(&self, resolution: u32, i: u32, j: u32) -> bool {
        if self.holes.is_empty() {
            return false;
        }
        let r = resolution.max(1);
        let i = i.min(r - 1);
        let j = j.min(r - 1);
        let n = (j * r + i) as usize;
        match self.holes.get(n / 8) {
            Some(byte) => byte & (1u8 << (n % 8)) != 0,
            // A mask shorter than the resolution demands cannot happen through
            // serde (the length check rejects it) or through `ensure_holes`; read
            // it as unholed rather than panicking on a corrupt in-memory tile.
            None => false,
        }
    }

    /// Materialize the full packed hole mask (all bits clear — nothing holed) if
    /// the tile is still on the sparse default, then return a mutable handle to
    /// the **packed bytes**. A carve calls this before setting bits.
    #[inline]
    pub fn ensure_holes(&mut self, resolution: u32) -> &mut [u8] {
        if self.holes.is_empty() {
            self.holes = vec![0u8; hole_mask_bytes(resolution)];
        }
        &mut self.holes
    }

    /// Reset the tile to the un-carved sparse default (drops the mask entirely),
    /// so it re-serializes byte-identically to a tile no carve ever touched. Used
    /// by carve **undo**, and by [`heal_holes`](TerrainTile::heal_holes) when the
    /// last hole closes.
    #[inline]
    pub fn clear_holes(&mut self) {
        self.holes = Vec::new();
    }

    /// Set or clear the hole bit at sample `(i, j)`, materializing the mask first
    /// when opening one. Out-of-range indices are ignored.
    ///
    /// Clearing on a tile that is still on the sparse default is a **no-op that
    /// allocates nothing** — closing a hole that was never open must not cost the
    /// tile its byte-stable default.
    #[inline]
    pub fn set_hole(&mut self, resolution: u32, i: u32, j: u32, hole: bool) {
        let r = resolution.max(1);
        if i >= r || j >= r {
            return;
        }
        if !hole && self.holes.is_empty() {
            return;
        }
        let n = (j * r + i) as usize;
        let mask = 1u8 << (n % 8);
        let buf = self.ensure_holes(resolution);
        if let Some(byte) = buf.get_mut(n / 8) {
            if hole {
                *byte |= mask;
            } else {
                *byte &= !mask;
            }
        }
    }

    /// Collapse an all-zero materialized mask back to the sparse default, and
    /// report whether it collapsed.
    ///
    /// The **inverse** half of the P21.2 carve↔fill round trip: carving
    /// materializes a mask, filling clears its bits, and this is what turns the
    /// resulting all-zero buffer back into bytes indistinguishable from a tile
    /// that was never carved. Without it a carve→fill cycle would leave a
    /// permanent `resolution²/8` scar in every container.
    #[inline]
    pub fn heal_holes(&mut self) -> bool {
        if !self.holes.is_empty() && !self.has_holes() {
            self.holes = Vec::new();
            true
        } else {
            false
        }
    }

    /// Length of the stored hole mask (`0` for the sparse default). Used by the
    /// terrain's serde length validation.
    #[inline]
    pub fn holes_len(&self) -> usize {
        self.holes.len()
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
    // reach the same door through `TerrainTileFrozenV3` (P21.2 froze the layout
    // that carries them, and `TerrainData`'s wire form now goes through it), and
    // the door that predates both is a **hand-built tile** — which is exactly what
    // a corrupt payload decodes to, because a `TerrainTile` cannot check its own
    // buffers (it does not know the terrain's resolution). `TerrainData`'s
    // `Deserialize` is therefore the ONLY check, on both codecs, for every layer.

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
            holes: Vec::new(),
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
            6,
            "the .inf_terrain header moved. Generation-1 frozen tiles cover header \
             versions 1..=2, generation-2 covers 3, generation-3 covers 4, and the \
             current TerrainTile covers 5 AND 6 — IASSET1's v6 changed the \
             CONTAINER's directory (a per-tile codec byte), not the tile blob, \
             which is the one kind of bump that needs no new frozen generation \
             and is exactly the distinction this pin exists to make someone \
             check. If the tile LAYOUT changed again, add TerrainTileFrozenV4 \
             and extend `decode_tile_at`; if only the header or the directory \
             changed, update this pin and TerrainTileFrozenV1's generation table."
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
        let gen3 =
            bincode::serde::encode_to_vec(TerrainTileFrozenV3::from_current(&tile), bincode_cfg())
                .unwrap();
        let current = crate::asset::encode_tile(&tile).unwrap();
        assert_eq!(
            gen2.len(),
            gen1.len() + 1,
            "generation 2 is generation 1 plus exactly the sparse maps count"
        );
        assert_eq!(
            gen3.len(),
            gen2.len() + 1,
            "generation 3 is generation 2 plus exactly the sparse biomes count"
        );
        assert_eq!(
            current.len(),
            gen3.len() + 1,
            "the current tile is generation 3 plus exactly the sparse holes count"
        );
        for (bytes, versions, label) in [
            (&gen1, &[0u32, 1, 2][..], "generation-1"),
            (&gen2, &[3][..], "generation-2"),
            (&gen3, &[4][..], "generation-3"),
            (&current, &[5][..], "current"),
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
