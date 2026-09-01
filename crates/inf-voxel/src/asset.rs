//! The `.inf_voxel` **streaming asset** (P21.1): a voxel volume's chunks in a
//! random-access, page-at-a-time container.
//!
//! ```text
//! ┌ header (v1: 64 B, little-endian) ─────────────────────────────────┐
//! │  magic         [u8; 8]   b"INFVOXEL"                              │
//! │  schema_ver    u32       VOXEL_ASSET_SCHEMA_VERSION               │
//! │  chunk_dim     u32       samples per chunk side (CHUNK_DIM)       │
//! │  voxel_size    f64       world metres per voxel                   │
//! │  origin        [f64; 3]  world anchor of global sample (0,0,0)    │
//! │  chunk_count   u32       directory entries                        │
//! │  reserved      u32       zero (room for v2 without a re-length)   │
//! │  blob_base     u64       absolute offset of the blob section      │
//! ├ chunk directory (chunk_count × 32 B, sorted by ChunkKey) ─────────┤
//! │  cx i32 · cy i32 · cz i32 · codec u8 + rsv[3] · offset u64 ·      │
//! │  stored_len u64                                                   │
//! ├ blob section (each chunk's stored bytes, 16-byte-aligned) ────────┤
//! │  … chunk blobs, zero-padded up to each 16-byte boundary …         │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Schema v2: per-chunk compression (IASSET1)
//!
//! v2 spends the directory entry's reserved word on a
//! [`BlockCodec`](inf_asset::BlockCodec) byte, exactly as `.inf_terrain` v6 does
//! and for the same reason: a chunk blob has always been `bincode`-decoded on
//! page-in, never cast, so a decompress in front of that decode is an addition to
//! an existing cost rather than the destruction of a zero-copy path.
//!
//! **The measurement that justified the bump.** SDF chunks are the most
//! compressible thing in this engine — a distance field over mostly-empty or
//! mostly-solid space. `samples/phase21-cavern/Cavern.inf_voxel`'s eight chunks:
//! **131 104 B → 11 619 B, ratio 0.089**, all eight compressed. (Terrain's is
//! 0.35; the island's `.inf_part` is 0.198 but only 60 kB in absolute terms,
//! which is why *it* did not get this treatment — see the IASSET1 memo.)
//!
//! **The downgrade story.** A v2 payload whose chunks are all
//! [`Raw`](inf_asset::BlockCodec::Raw) differs from the v1 image of the same
//! chunks in exactly the four bytes of `schema_version`, because the codec byte
//! lands where v1 already wrote a reserved zero. v1 payloads load for ever, and
//! `parse` keeps the *strict* reserved-word check on the v1 branch — which is
//! precisely the "claimed by a version bump" discipline the section below
//! promised, being exercised for the first time.
//!
//! # Why this shape
//!
//! It is `inf_terrain`'s `.inf_terrain` layout with a third coordinate, and the
//! deliberate sameness is the point: one set of habits, one set of hazards, one
//! set of lessons already paid for.
//!
//! * **Random access without decoding the volume.** The directory is a flat,
//!   sorted array — a consumer binary-searches a [`ChunkKey`] and gets an
//!   `(offset, len)` into the payload. A cooked-pack consumer takes
//!   [`PackReader::read_ref`](inf_asset::PackReader::read_ref) (the entry is
//!   **uncompressed** by `PackWriter::compresses_kind`, so it is a borrowed slice
//!   of the mapping) and sub-slices it; a loose-file consumer seeks to the very
//!   same offsets. One layout, both paths.
//! * **16-byte-aligned chunks.** Blob offsets are multiples of [`CHUNK_ALIGN`] —
//!   the same constant and the same reasoning as [`inf_asset::BLOB_ALIGN`]: a pack
//!   v2 blob starts 16-byte aligned *and* every pack backing is 16-byte aligned at
//!   its base, so a chunk slice sub-sliced out of one is aligned by address too.
//! * **Byte-deterministic.** The directory is emitted in `BTreeMap` order, padding
//!   is deterministic zeros, and each chunk blob is the *same* bincode a chunk
//!   writes anywhere else — so two builds of one volume are byte-identical.
//!
//! # The reserved bytes are **validated zero**, not merely written zero
//!
//! Two fields in the layout above carry no data: the header word at 52..56 and
//! each directory entry's word at `base + 12 .. base + 16`. They exist so a
//! future version can grow *without* re-lengthing a fixed-size record, which is
//! the same room [`inf_terrain`]'s v2 header bought itself with 56 bytes and
//! spent nothing of across three tile-layout bumps.
//!
//! Reserved room is only worth anything if it is **claimable exactly once, by a
//! version bump**. So [`parse`] rejects a payload whose reserved words are
//! non-zero, and the rejection is the point: without it, a future writer could
//! start putting meaning in those bytes and every reader in the wild would keep
//! loading the file, silently ignoring a field it does not understand — the
//! precise failure the version ladder exists to prevent, arriving through the one
//! hole the ladder does not cover. With the check, that writer's files fail to
//! load until it stamps the version that describes them, which is the correct and
//! only outcome.
//!
//! It costs two comparisons per payload and one per directory entry, catches
//! truncate-and-refill corruption on the way past, and constrains nothing this
//! build can legitimately produce: [`write_voxel_asset`] is the only writer and
//! it writes zeros. **A v2+ reader keeps this check for the v1 branch and defines
//! its own meaning for the bytes at its own version** — that is what "claimed by
//! a version bump" means operationally.
//!
//! (`inf_terrain`'s reader does not do this. Its reserved tail predates the rule
//! and adding the check there would be a compatibility decision about files that
//! already exist, not a free one; `.inf_voxel` is new, so it starts strict.)
//!
//! # The version selects the wire type
//!
//! bincode is **positional**, so any future growth of the chunk blob (a second
//! material channel, a per-sample flags layer) is a wire-format change even if the
//! new field is `#[serde(default)]`. [`decode_chunk_at`] is therefore the one
//! place the version→layout mapping lives, and it exists at v1 with a single row
//! precisely so the second row has an obvious home — the shape
//! `inf_terrain::decode_tile_at` grew into across two phases, adopted up front.
//!
//! # There is exactly one writer: [`write_voxel_asset`]
//!
//! The bytes on disk — and in a `.ipack` — are the **raw image**
//! ([`VoxelAsset::as_bytes`]). They are *not* `inf_asset::encode` output: a
//! bincode length prefix would shift every chunk off its 16-byte boundary and
//! defeat the entire point of the layout, silently, in a way only a GPU upload on
//! some other machine would notice.
//!
//! So the generic door is **closed at compile time**: [`VoxelAsset`] deliberately
//! implements neither [`AssetPayload`](inf_asset::AssetPayload) nor
//! `Serialize`/`Deserialize`, which makes `inf_asset::encode`,
//! `Project::write_asset` and `Project::rewrite_payload` reject it with a type
//! error rather than write a subtly wrong file. The kind and schema version it
//! still owes the database live as inherent consts ([`VoxelAsset::KIND`],
//! [`VoxelAsset::SCHEMA_VERSION`]).

use std::borrow::Cow;
use std::collections::BTreeMap;

use glam::DVec3;
use inf_asset::block::{decode_block, encode_block, BlockCodec};
use inf_asset::AssetKind;

use crate::chunk::{ChunkKey, VoxelChunk, CHUNK_DIM};
use crate::data::VoxelData;

/// Magic at the head of every `.inf_voxel` payload.
pub const VOXEL_ASSET_MAGIC: [u8; 8] = *b"INFVOXEL";

/// Current `.inf_voxel` payload schema version (**2** since IASSET1 — the chunk
/// directory carries a per-chunk [`BlockCodec`]).
pub const VOXEL_ASSET_SCHEMA_VERSION: u32 = 2;

/// The first schema version whose directory entries carry a codec byte.
///
/// Below it, entry byte 12 is v1's reserved word and **must be zero** (the module
/// docs' claimed-by-a-version-bump rule) — which is what makes "a v1 asset has no
/// codec" a checked fact rather than an assumption.
pub const FIRST_CODEC_SCHEMA_VERSION: u32 = 2;

/// Chunk blobs start on multiples of this many bytes (see the module docs).
pub const CHUNK_ALIGN: u64 = 16;

/// Bytes of the **v1** fixed header. A multiple of [`CHUNK_ALIGN`], so the
/// directory and every blob stay aligned without padding.
pub const HEADER_LEN_V1: u64 = 64;

/// Bytes of the **v2** fixed header. IASSET1's per-chunk codec lives in the
/// *directory entry*, in a byte v1 already reserved, so the header does not move.
pub const HEADER_LEN_V2: u64 = HEADER_LEN_V1;

/// Bytes of the fixed header **this build writes**.
pub const HEADER_LEN: u64 = HEADER_LEN_V2;

/// Bytes of the fixed header of a payload at `schema_version`.
///
/// The one place the version→length mapping lives, so the writer, the parser and
/// the `blob_base` validation cannot disagree about where the directory starts.
/// An unknown (future) version is rejected before this is ever asked.
#[inline]
pub const fn header_len(schema_version: u32) -> u64 {
    match schema_version {
        1 => HEADER_LEN_V1,
        2 => HEADER_LEN_V2,
        // Unreachable in practice: `parse` rejects an unknown version before this
        // is ever asked. Written as a table row rather than a bare constant so the
        // v3 row has an obvious, single home.
        _ => HEADER_LEN_V2,
    }
}

/// A ceiling on how many bytes one chunk blob can decompress to.
///
/// A compressed block's declared length is written by whoever made the file, so
/// it is bounded before it becomes an allocation. A [`VoxelChunk`] is a fixed
/// `CHUNK_DIM³` grid — the one thing this container knows exactly — so the bound
/// is arithmetic rather than a guess: 64 bytes per sample is better than 3×
/// headroom over the widest sample a chunk can hold, plus slack for `bincode`'s
/// framing.
#[inline]
pub const fn chunk_raw_ceiling() -> usize {
    const BYTES_PER_SAMPLE: usize = 64;
    const SLACK: usize = 64 * 1024;
    (CHUNK_DIM as usize).pow(3) * BYTES_PER_SAMPLE + SLACK
}

/// **The codec the cook transcodes `.inf_voxel` chunks to** (IASSET1).
///
/// The same `Zstd` `.inf_terrain` chose, for the same measured reasons — see
/// [`inf_terrain::COOK_TILE_CODEC`]'s table. One codec across the streaming
/// containers so a build has one decode profile, not three.
pub const COOK_CHUNK_CODEC: BlockCodec = BlockCodec::Zstd;

/// Bytes of one chunk-directory entry.
pub const DIR_ENTRY_LEN: u64 = 32;

/// A failure building or reading a `.inf_voxel` payload.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VoxelAssetError {
    #[error("voxel asset is shorter than its fixed header")]
    TooShort,
    #[error("not an .inf_voxel payload (bad magic)")]
    BadMagic,
    #[error("voxel asset schema v{found} is newer than this build (v{current})")]
    SchemaTooNew { found: u32, current: u32 },
    #[error("voxel asset is malformed: {0}")]
    Malformed(String),
    #[error("voxel chunk ({x},{y},{z}) failed to decode: {message}")]
    ChunkDecode {
        x: i32,
        y: i32,
        z: i32,
        message: String,
    },
    #[error("voxel chunk ({x},{y},{z}) added twice")]
    DuplicateChunk { x: i32, y: i32, z: i32 },
}

type Result<T> = std::result::Result<T, VoxelAssetError>;

/// Round `n` up to the next multiple of [`CHUNK_ALIGN`].
#[inline]
fn align_up(n: u64) -> u64 {
    n.next_multiple_of(CHUNK_ALIGN)
}

/// The bincode configuration chunk blobs use — the **same** `standard()` every
/// other payload in the engine uses, so a chunk's bytes are identical wherever
/// they land.
fn chunk_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// Encode one chunk to its canonical blob bytes.
pub fn encode_chunk(chunk: &VoxelChunk) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(chunk, chunk_config())
        .map_err(|e| VoxelAssetError::Malformed(format!("chunk encode: {e}")))
}

/// Decode one chunk from its canonical blob bytes, at the **current** schema.
///
/// Prefer [`decode_chunk_at`] anywhere the payload's own version is in hand.
pub fn decode_chunk(bytes: &[u8]) -> std::result::Result<VoxelChunk, String> {
    decode_chunk_at(bytes, VOXEL_ASSET_SCHEMA_VERSION)
}

/// Decode one chunk blob written at `schema_version`.
///
/// The version is the **only** thing that says which chunk wire layout the bytes
/// hold (bincode carries no field names or count), so this is the single place the
/// mapping lives:
///
/// | payload schema | chunk layout |
/// |---|---|
/// | 1 | [`VoxelChunk`] — sdf + sparse materials |
///
/// One row today. It is a table rather than a call because the *second* row is
/// what breaks a build that never had one: `inf_terrain` grew from one row to
/// three across two phases, and each growth was a wire-format change that a
/// `#[serde(default)]` could not rescue.
pub fn decode_chunk_at(
    bytes: &[u8],
    schema_version: u32,
) -> std::result::Result<VoxelChunk, String> {
    debug_assert!(
        schema_version <= VOXEL_ASSET_SCHEMA_VERSION,
        "a payload newer than this build must be rejected at parse, not decoded"
    );
    bincode::serde::decode_from_slice(bytes, chunk_config())
        .map(|(c, _)| c)
        .map_err(|e| e.to_string())
}

// ── header ──────────────────────────────────────────────────────────────────

/// The decoded fixed header of a `.inf_voxel` payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelAssetHeader {
    /// Payload schema version.
    pub schema_version: u32,
    /// Samples per chunk side. Recorded rather than assumed: a payload written by
    /// a build with a different [`CHUNK_DIM`] is *readable metadata*, not a
    /// mystery, and the parser rejects a mismatch with a message that says the two
    /// numbers instead of failing later on a short buffer.
    pub chunk_dim: u32,
    /// World metres per voxel.
    pub voxel_size_m: f64,
    /// World anchor of global sample `(0, 0, 0)`.
    pub origin: DVec3,
    /// Number of chunk-directory entries.
    pub chunk_count: u32,
    /// Absolute offset of the blob section within the payload.
    pub blob_base: u64,
}

/// One chunk-directory entry: where a chunk's blob lives inside the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkDirEntry {
    /// Which chunk this is.
    pub key: ChunkKey,
    /// Absolute offset of the blob **within the payload** (a multiple of
    /// [`CHUNK_ALIGN`]) — usable directly as a `seek` target on a loose file.
    pub offset: u64,
    /// **Stored** blob length in bytes (unpadded) — the bytes on disk, which for
    /// a compressed chunk is smaller than what it decodes to.
    pub len: u64,
    /// How the blob is stored (schema v2+; always [`BlockCodec::Raw`] below it).
    pub codec: BlockCodec,
}

// ── builder ─────────────────────────────────────────────────────────────────

/// Assembles a `.inf_voxel` payload chunk by chunk.
///
/// Chunks land in a `BTreeMap` keyed by [`ChunkKey`], so insertion order cannot
/// affect the output — [`build`](Self::build) is a pure function of the chunk set.
#[derive(Debug, Clone)]
pub struct VoxelAssetBuilder {
    voxel_size_m: f64,
    origin: DVec3,
    codec: BlockCodec,
    chunks: BTreeMap<ChunkKey, Vec<u8>>,
}

impl VoxelAssetBuilder {
    /// A builder for a volume at the given voxel scale (metres), anchored at the
    /// world origin.
    pub fn new(voxel_size_m: f64) -> Self {
        Self {
            voxel_size_m: if voxel_size_m.is_finite() && voxel_size_m > 0.0 {
                voxel_size_m
            } else {
                0.5
            },
            origin: DVec3::ZERO,
            // RAW by default: this builder writes the LOOSE asset an author
            // carves, whose chunks are rewritten on every write-back.
            // Compression belongs to the cook (`recompress_voxel_asset`).
            codec: BlockCodec::Raw,
            chunks: BTreeMap::new(),
        }
    }

    /// Compress every staged chunk under `codec` (IASSET1, schema v2).
    ///
    /// Per-chunk and independent: a chunk that would inflate is stored
    /// [`Raw`](BlockCodec::Raw) and its directory entry says so, so the payload
    /// can never be larger than the uncompressed one. The default is
    /// [`Raw`](BlockCodec::Raw) — see [`new`](Self::new).
    pub fn with_codec(mut self, codec: BlockCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Anchor the chunk grid at an explicit world position.
    pub fn with_origin(mut self, origin: DVec3) -> Self {
        self.origin = if origin.is_finite() {
            origin
        } else {
            DVec3::ZERO
        };
        self
    }

    /// Add a chunk, encoding it to its canonical blob. Errors on a duplicate key.
    pub fn insert(&mut self, key: ChunkKey, chunk: &VoxelChunk) -> Result<()> {
        self.insert_bytes(key, encode_chunk(chunk)?)
    }

    /// Add a chunk from already-encoded blob bytes — the **verbatim** path a
    /// re-pack takes, which never decodes.
    pub fn insert_bytes(&mut self, key: ChunkKey, bytes: Vec<u8>) -> Result<()> {
        if self.chunks.insert(key, bytes).is_some() {
            return Err(VoxelAssetError::DuplicateChunk {
                x: key.x,
                y: key.y,
                z: key.z,
            });
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Serialize the payload image.
    pub fn build(self) -> Result<VoxelAsset> {
        let count = u32::try_from(self.chunks.len())
            .map_err(|_| VoxelAssetError::Malformed("more than u32::MAX chunks".into()))?;

        // Compress each chunk independently (IASSET1). `encode_block` reports
        // the codec it actually used, which is `Raw` for a chunk compression
        // would inflate — so the directory always states what happened.
        let stored: Vec<(BlockCodec, Vec<u8>)> = self
            .chunks
            .values()
            .map(|blob| encode_block(self.codec, blob))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e: inf_asset::AssetError| {
                VoxelAssetError::Malformed(format!("chunk compression: {e}"))
            })?;

        // Offsets first: the blob section starts after the header + full directory
        // (both already multiples of CHUNK_ALIGN, so no padding is ever needed
        // there — `align_up` states the invariant rather than relying on it), and
        // every chunk after it starts on the next aligned boundary.
        let blob_base = align_up(HEADER_LEN + DIR_ENTRY_LEN * count as u64);
        let mut offsets = Vec::with_capacity(self.chunks.len());
        let mut offset = blob_base;
        for (_, blob) in &stored {
            offsets.push(offset);
            offset = align_up(offset + blob.len() as u64);
        }
        let total = offset;

        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&VOXEL_ASSET_MAGIC);
        out.extend_from_slice(&VOXEL_ASSET_SCHEMA_VERSION.to_le_bytes());
        out.extend_from_slice(&CHUNK_DIM.to_le_bytes());
        out.extend_from_slice(&self.voxel_size_m.to_le_bytes());
        for v in self.origin.to_array() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved, always zero
        out.extend_from_slice(&blob_base.to_le_bytes());
        debug_assert_eq!(out.len() as u64, HEADER_LEN);

        for ((key, (codec, blob)), &off) in self.chunks.keys().zip(&stored).zip(&offsets) {
            out.extend_from_slice(&key.x.to_le_bytes());
            out.extend_from_slice(&key.y.to_le_bytes());
            out.extend_from_slice(&key.z.to_le_bytes());
            // v2: the codec byte, then three bytes v1 reserved. A raw chunk
            // writes 0 here, which is what makes an all-raw v2 image
            // byte-identical to the v1 one.
            out.push(codec.code());
            out.extend_from_slice(&[0u8; 3]);
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        }
        debug_assert_eq!(out.len() as u64, HEADER_LEN + DIR_ENTRY_LEN * count as u64);

        // Blob section, zero-padded up to each offset (and to the end, so the
        // payload length is itself a multiple of CHUNK_ALIGN).
        for ((_, blob), &off) in stored.iter().zip(&offsets) {
            out.resize(off as usize, 0);
            out.extend_from_slice(blob);
        }
        out.resize(total as usize, 0);

        VoxelAsset::from_bytes(out)
    }
}

/// Build a `.inf_voxel` payload from a volume's resident chunks.
pub fn build_voxel_asset(data: &VoxelData) -> Result<VoxelAsset> {
    let mut b = VoxelAssetBuilder::new(data.voxel_size_m()).with_origin(data.origin());
    for (&key, chunk) in data.chunks() {
        b.insert(key, chunk)?;
    }
    b.build()
}

// ── the payload ─────────────────────────────────────────────────────────────

/// A validated `.inf_voxel` payload image.
///
/// Owns the bytes; [`reader`](Self::reader) borrows a random-access view over
/// them. See the module docs for why [`as_bytes`](Self::as_bytes) — not
/// `inf_asset::encode` — is what goes on disk and into a pack.
#[derive(Clone, PartialEq)]
pub struct VoxelAsset {
    bytes: Vec<u8>,
    header: VoxelAssetHeader,
}

impl VoxelAsset {
    /// The asset kind this payload is written as. An inherent const, not an
    /// [`AssetPayload`](inf_asset::AssetPayload) impl — see the module docs.
    pub const KIND: AssetKind = AssetKind::VoxelVolume;

    /// The current payload schema version (mirrors
    /// [`VOXEL_ASSET_SCHEMA_VERSION`], published beside [`KIND`](Self::KIND)).
    pub const SCHEMA_VERSION: u32 = VOXEL_ASSET_SCHEMA_VERSION;

    /// Validate and take ownership of a payload image.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let header = parse(&bytes)?.0;
        Ok(Self { bytes, header })
    }

    /// The canonical payload bytes — what is written to `*.inf_voxel` and packed.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume into the payload bytes.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// The decoded header.
    #[inline]
    pub fn header(&self) -> &VoxelAssetHeader {
        &self.header
    }

    /// A random-access view over the payload.
    pub fn reader(&self) -> VoxelAssetReader<&[u8]> {
        // Already validated in `from_bytes`; re-parsing cannot fail.
        VoxelAssetReader::new(self.bytes.as_slice()).expect("validated payload")
    }
}

impl std::fmt::Debug for VoxelAsset {
    /// Summarizes; never dumps the (possibly gigabyte) chunk blobs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoxelAsset")
            .field("schema_version", &self.header.schema_version)
            .field("chunk_dim", &self.header.chunk_dim)
            .field("voxel_size_m", &self.header.voxel_size_m)
            .field("origin", &self.header.origin)
            .field("chunk_count", &self.header.chunk_count)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

/// Write a `.inf_voxel` payload to `path` — **the one sanctioned loose-file
/// writer** (see the module docs).
///
/// Writes the raw image, never a framed encoding, and does it **atomically**: the
/// bytes go to a sibling temp file that is then renamed over `path`, exactly like
/// `inf_terrain::write_terrain_asset` and for the same reason — a re-save
/// routinely targets a file some other process may still have mapped, and
/// truncating it under a live mapping is the mutation an mmap's safety contract
/// forbids. A failed write cleans up its temp file rather than leaving litter.
///
/// The mechanism comes from [`inf_asset::write_atomically`] rather than being
/// re-spelled here, which also gives this door the *counter* half of the Spike C
/// temp-name law: the name used to be pid-only, so two threads of one process
/// writing two volumes at once could pick one temp path and interleave into each
/// other's rename.
///
/// Returns the bytes written, so a caller can hash them into the sidecar without
/// re-reading the file.
pub fn write_voxel_asset<'a>(
    path: &std::path::Path,
    asset: &'a VoxelAsset,
) -> std::io::Result<&'a [u8]> {
    let bytes = asset.as_bytes();
    inf_asset::write_atomically(path, bytes)?;
    Ok(bytes)
}

/// What one [`recompress_voxel_asset`] pass did — the cook's per-asset row in
/// the before/after table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecompressReport {
    /// Chunks in the payload.
    pub chunks: usize,
    /// Chunks that actually ended up compressed.
    pub chunks_compressed: usize,
    /// Payload bytes before.
    pub bytes_before: u64,
    /// Payload bytes after.
    pub bytes_after: u64,
}

impl RecompressReport {
    /// `after / before` — 1.0 when nothing was won.
    pub fn ratio(&self) -> f64 {
        if self.bytes_before == 0 {
            1.0
        } else {
            self.bytes_after as f64 / self.bytes_before as f64
        }
    }
}

/// **Transcode a `.inf_voxel` payload to per-chunk `codec`** — the cook's
/// compile step for a streaming volume (IASSET1).
///
/// The `.inf_terrain` door's twin ([`inf_terrain::recompress_terrain_asset`]),
/// and lossless for the same construction: chunk blobs are carried through as
/// **opaque bytes**, never decoded and re-encoded, so this cannot change an SDF
/// sample even if [`VoxelChunk`]'s wire form were wrong. A v1 payload is lifted
/// to v2, which is legal precisely because the *chunk blob* layout is untouched.
pub fn recompress_voxel_asset(
    payload: &[u8],
    codec: BlockCodec,
) -> Result<(Vec<u8>, RecompressReport)> {
    let reader = VoxelAssetReader::new(payload)?;
    let mut b = VoxelAssetBuilder::new(reader.voxel_size_m())
        .with_origin(reader.origin())
        .with_codec(codec);
    for e in reader.directory() {
        let blob = reader
            .chunk_bytes(e.key)
            .ok_or_else(|| VoxelAssetError::ChunkDecode {
                x: e.key.x,
                y: e.key.y,
                z: e.key.z,
                message: "stored block did not decode".into(),
            })?;
        b.insert_bytes(e.key, blob.into_owned())?;
    }
    let asset = b.build()?;
    let after = VoxelAssetReader::new(asset.as_bytes())?;
    let report = RecompressReport {
        chunks: after.chunk_count(),
        chunks_compressed: after
            .directory()
            .iter()
            .filter(|e| e.codec != BlockCodec::Raw)
            .count(),
        bytes_before: payload.len() as u64,
        bytes_after: asset.as_bytes().len() as u64,
    };
    Ok((asset.into_bytes(), report))
}

/// Read a `.inf_voxel` payload from `path`, validating it.
pub fn read_voxel_asset(path: &std::path::Path) -> std::io::Result<VoxelAsset> {
    let bytes = std::fs::read(path)?;
    VoxelAsset::from_bytes(bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ── reader ──────────────────────────────────────────────────────────────────

/// Random access over a `.inf_voxel` payload.
///
/// Generic over the byte source so the *same* reader serves every backing: an
/// owned `Vec<u8>` (a loose file read), a `&[u8]` borrowed from a
/// [`PackReader::read_ref`](inf_asset::PackReader::read_ref) mapping, or a `Cow`
/// straight off it. [`VoxelAssetView`] is the borrowed shorthand.
///
/// The header + directory are parsed and validated once at construction; every
/// [`chunk_bytes`](Self::chunk_bytes) after that is a binary search plus a slice —
/// no decode, no allocation.
#[derive(Debug, Clone)]
pub struct VoxelAssetReader<B> {
    bytes: B,
    header: VoxelAssetHeader,
    dir: Vec<ChunkDirEntry>,
}

/// A [`VoxelAssetReader`] borrowing its bytes (the pack-mapping case).
pub type VoxelAssetView<'a> = VoxelAssetReader<&'a [u8]>;

impl<B: AsRef<[u8]>> VoxelAssetReader<B> {
    /// Parse + validate a payload. Rejects a bad magic, a newer schema, a
    /// foreign chunk dimension, an out-of-bounds, overlapping or misaligned blob,
    /// and an out-of-order directory (which would break both the binary search and
    /// the byte-determinism guarantee).
    pub fn new(bytes: B) -> Result<Self> {
        let (header, dir) = parse(bytes.as_ref())?;
        Ok(Self { bytes, header, dir })
    }

    /// The whole payload image.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// The decoded header.
    #[inline]
    pub fn header(&self) -> &VoxelAssetHeader {
        &self.header
    }

    /// World metres per voxel.
    #[inline]
    pub fn voxel_size_m(&self) -> f64 {
        self.header.voxel_size_m
    }

    /// World anchor of global sample `(0, 0, 0)`.
    #[inline]
    pub fn origin(&self) -> DVec3 {
        self.header.origin
    }

    /// Number of chunks in the payload.
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.dir.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dir.is_empty()
    }

    /// The chunk directory, in [`ChunkKey`] order — the seam a loose-file consumer
    /// seeks with.
    #[inline]
    pub fn directory(&self) -> &[ChunkDirEntry] {
        &self.dir
    }

    /// Every key present, in directory order.
    pub fn keys(&self) -> impl Iterator<Item = ChunkKey> + '_ {
        self.dir.iter().map(|e| e.key)
    }

    /// The directory entry for `key`.
    pub fn entry(&self, key: ChunkKey) -> Option<&ChunkDirEntry> {
        self.dir
            .binary_search_by(|e| e.key.cmp(&key))
            .ok()
            .map(|i| &self.dir[i])
    }

    /// A chunk's **stored** bytes, exactly as they sit in the payload — borrowed,
    /// 16-byte aligned within it, and still compressed if the directory says so.
    ///
    /// The re-pack seam ([`recompress_voxel_asset`], a write-back carrying a
    /// chunk through untouched) and the seam a size report measures. A consumer
    /// that wants the *chunk* wants [`chunk_bytes`](Self::chunk_bytes).
    pub fn stored_bytes(&self, key: ChunkKey) -> Option<&[u8]> {
        let e = self.entry(key)?;
        // Bounds + alignment were validated in `new`.
        Some(&self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize])
    }

    /// A chunk's **canonical bincode blob** — borrowed straight out of the
    /// payload for a [`Raw`](BlockCodec::Raw) chunk, decompressed into an owned
    /// buffer for a compressed one (schema v2, IASSET1).
    ///
    /// `None` for an absent chunk, and `None` for a corrupt block: a directory
    /// that lies is an absent chunk, never silent geometry. The caller that wants
    /// the reason uses [`chunk`](Self::chunk).
    pub fn chunk_bytes(&self, key: ChunkKey) -> Option<Cow<'_, [u8]>> {
        let e = self.entry(key)?;
        let stored = &self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize];
        decode_block(e.codec, stored, chunk_raw_ceiling()).ok()
    }

    /// Decode a chunk, through **this payload's own** schema version.
    pub fn chunk(&self, key: ChunkKey) -> Result<Option<VoxelChunk>> {
        let Some(e) = self.entry(key) else {
            return Ok(None);
        };
        let stored = &self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize];
        let bytes = decode_block(e.codec, stored, chunk_raw_ceiling()).map_err(|message| {
            VoxelAssetError::ChunkDecode {
                x: key.x,
                y: key.y,
                z: key.z,
                message: message.to_string(),
            }
        })?;
        decode_chunk_at(&bytes, self.header.schema_version)
            .map(Some)
            .map_err(|message| VoxelAssetError::ChunkDecode {
                x: key.x,
                y: key.y,
                z: key.z,
                message,
            })
    }

    /// Rebuild a fully-resident [`VoxelData`] from the payload.
    ///
    /// The whole-volume door, for tools and tests. A *streaming* consumer pages
    /// through the residency layer instead ([`crate::residency`]) and never calls
    /// this — that is the difference between a cave system that fits in memory and
    /// one that does not.
    pub fn to_voxel_data(&self) -> Result<VoxelData> {
        let mut data = VoxelData::new(self.voxel_size_m()).with_origin(self.origin());
        for e in &self.dir {
            let chunk = self.chunk(e.key)?.expect("directory entry resolves");
            data.insert_resident_chunk(e.key, chunk);
        }
        data.clear_dirty();
        Ok(data)
    }
}

/// Parse + validate the header and directory of a payload image.
fn parse(data: &[u8]) -> Result<(VoxelAssetHeader, Vec<ChunkDirEntry>)> {
    if (data.len() as u64) < HEADER_LEN_V1 {
        return Err(VoxelAssetError::TooShort);
    }
    if data[0..8] != VOXEL_ASSET_MAGIC {
        return Err(VoxelAssetError::BadMagic);
    }
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
    let f64_at = |o: usize| f64::from_le_bytes(data[o..o + 8].try_into().unwrap());

    let schema_version = u32_at(8);
    if schema_version == 0 || schema_version > VOXEL_ASSET_SCHEMA_VERSION {
        return Err(VoxelAssetError::SchemaTooNew {
            found: schema_version,
            current: VOXEL_ASSET_SCHEMA_VERSION,
        });
    }
    let chunk_dim = u32_at(12);
    if chunk_dim != CHUNK_DIM {
        // Recorded, not assumed: a payload from a build with a different chunk
        // dimension fails HERE, saying both numbers, rather than decoding into a
        // buffer of the wrong length and surfacing as geometry that is wrong by a
        // factor of two.
        return Err(VoxelAssetError::Malformed(format!(
            "chunk_dim {chunk_dim} is not this build's {CHUNK_DIM}"
        )));
    }
    let voxel_size_m = f64_at(16);
    if !(voxel_size_m.is_finite() && voxel_size_m > 0.0) {
        return Err(VoxelAssetError::Malformed(format!(
            "voxel_size_m {voxel_size_m} is not a positive finite length"
        )));
    }
    let origin = DVec3::new(f64_at(24), f64_at(32), f64_at(40));
    if !origin.is_finite() {
        return Err(VoxelAssetError::Malformed("origin is not finite".into()));
    }
    let chunk_count = u32_at(48);
    // Reserved: claimable only by a version bump — see the module docs.
    if u32_at(52) != 0 {
        return Err(VoxelAssetError::Malformed(
            "reserved header word is not zero; a v1 payload may not use it".into(),
        ));
    }
    let blob_base = u64_at(56);

    let hlen = header_len(schema_version);
    let dir_end = hlen + DIR_ENTRY_LEN * chunk_count as u64;
    if (data.len() as u64) < dir_end {
        return Err(VoxelAssetError::Malformed(
            "truncated in the chunk directory".into(),
        ));
    }
    if blob_base != align_up(dir_end) || blob_base > data.len() as u64 {
        return Err(VoxelAssetError::Malformed(format!(
            "blob_base {blob_base} does not follow the directory"
        )));
    }

    let mut dir = Vec::with_capacity(chunk_count as usize);
    let mut prev: Option<ChunkKey> = None;
    // Blobs are laid out in directory order, so each must start at or after the
    // previous one's end. This is the **overlap** check: two entries sharing bytes
    // would let one chunk's edit corrupt another, and a reader has no way to notice.
    let mut prev_end = blob_base;
    for i in 0..chunk_count as u64 {
        let base = (hlen + DIR_ENTRY_LEN * i) as usize;
        let key = ChunkKey {
            x: i32::from_le_bytes(data[base..base + 4].try_into().unwrap()),
            y: i32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()),
            z: i32::from_le_bytes(data[base + 8..base + 12].try_into().unwrap()),
        };
        // Bytes 12..16: v1's reserved word, half of which v2 spends on a codec
        // byte. Below v2 the WHOLE word must still be zero — the module docs'
        // claimed-by-a-version-bump rule, and what makes "a v1 asset has no
        // codec" a checked fact rather than an assumption.
        let codec = if schema_version >= FIRST_CODEC_SCHEMA_VERSION {
            if data[base + 13..base + 16] != [0, 0, 0] {
                return Err(VoxelAssetError::Malformed(format!(
                    "chunk directory entry {i} has a non-zero reserved tail"
                )));
            }
            BlockCodec::from_code(data[base + 12]).ok_or_else(|| {
                VoxelAssetError::Malformed(format!(
                    "chunk block codec {} is not one this build knows",
                    data[base + 12]
                ))
            })?
        } else {
            if u32::from_le_bytes(data[base + 12..base + 16].try_into().unwrap()) != 0 {
                return Err(VoxelAssetError::Malformed(format!(
                    "chunk directory entry {i} has a non-zero reserved word; a v1 \
                     payload may not use it"
                )));
            }
            BlockCodec::Raw
        };
        let offset = u64::from_le_bytes(data[base + 16..base + 24].try_into().unwrap());
        let len = u64::from_le_bytes(data[base + 24..base + 32].try_into().unwrap());
        if let Some(p) = prev {
            if key <= p {
                return Err(VoxelAssetError::Malformed(
                    "chunk directory is not strictly ascending".into(),
                ));
            }
        }
        prev = Some(key);
        if offset % CHUNK_ALIGN != 0 {
            return Err(VoxelAssetError::Malformed(format!(
                "chunk blob at {offset} is not {CHUNK_ALIGN}-byte aligned"
            )));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| VoxelAssetError::Malformed("chunk length overflow".into()))?;
        if offset < blob_base || end > data.len() as u64 {
            return Err(VoxelAssetError::Malformed(format!(
                "chunk blob [{offset}, {end}) out of bounds"
            )));
        }
        if offset < prev_end {
            return Err(VoxelAssetError::Malformed(format!(
                "chunk blob [{offset}, {end}) overlaps the previous blob (ends at {prev_end})"
            )));
        }
        prev_end = end;
        dir.push(ChunkDirEntry {
            key,
            offset,
            len,
            codec,
        });
    }

    Ok((
        VoxelAssetHeader {
            schema_version,
            chunk_dim,
            voxel_size_m,
            origin,
            chunk_count,
            blob_base,
        },
        dir,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::chunk_range;

    fn sample_volume() -> VoxelData {
        let mut v = VoxelData::new(0.5).with_origin(DVec3::new(10.0, -4.0, 2.0));
        for key in chunk_range(ChunkKey::new(-1, 0, 0), ChunkKey::new(1, 1, 0)) {
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|i, j, k| {
                    let p = DVec3::new(
                        (b[0] + i as i32) as f64,
                        (b[1] + j as i32) as f64,
                        (b[2] + k as i32) as f64,
                    );
                    p.distance(DVec3::new(4.0, 8.0, 8.0)) - 7.0
                }),
            );
        }
        // One painted chunk, so the blobs exercise both the sparse and the
        // materialized form.
        v.get_chunk_mut(ChunkKey::new(0, 0, 0))
            .unwrap()
            .set_material(3, 3, 3, 2);
        v.clear_dirty();
        v
    }

    // ── schema v2: per-chunk block compression (IASSET1) ────────────────────

    /// **The downgrade story, as bytes.** A v2 payload with every chunk
    /// [`BlockCodec::Raw`] differs from the v1 image of the same chunks in
    /// exactly the four bytes of `schema_version` — the codec byte lands where v1
    /// already wrote a reserved zero — and the restamped image parses as v1
    /// through the strict branch.
    #[test]
    fn v2_all_raw_is_v1_plus_four_bytes() {
        let asset = build_voxel_asset(&sample_volume()).unwrap();
        let image = asset.as_bytes();
        assert_eq!(image[8..12], 2u32.to_le_bytes(), "this build writes v2");
        for e in asset.reader().directory() {
            assert_eq!(e.codec, BlockCodec::Raw, "the loose writer stays raw");
        }
        let mut as_v1 = image.to_vec();
        as_v1[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(image[..8], as_v1[..8]);
        assert_eq!(image[12..], as_v1[12..]);
        let v1 = VoxelAssetReader::new(as_v1.as_slice()).unwrap();
        assert_eq!(v1.header().schema_version, 1);
        for e in asset.reader().directory() {
            assert_eq!(
                v1.chunk_bytes(e.key).unwrap(),
                asset.reader().chunk_bytes(e.key).unwrap()
            );
        }
    }

    /// **The lossless claim, per codec**, and the ratio SDF chunks actually give.
    #[test]
    fn recompress_round_trips_every_codec_byte_for_byte() {
        let asset = build_voxel_asset(&sample_volume()).unwrap();
        let raw_image = asset.as_bytes().to_vec();
        for codec in BlockCodec::ALL {
            let (packed, report) = recompress_voxel_asset(&raw_image, codec).unwrap();
            assert_eq!(report.chunks, asset.reader().chunk_count());
            assert!(
                report.bytes_after <= report.bytes_before,
                "{codec:?} inflated"
            );
            let r = VoxelAssetReader::new(packed.as_slice()).unwrap();
            assert_eq!(r.header().schema_version, VOXEL_ASSET_SCHEMA_VERSION);
            for e in asset.reader().directory() {
                assert_eq!(
                    r.chunk(e.key).unwrap(),
                    asset.reader().chunk(e.key).unwrap(),
                    "{codec:?} changed chunk {:?}",
                    e.key
                );
                assert_eq!(
                    r.chunk_bytes(e.key).unwrap(),
                    asset.reader().chunk_bytes(e.key).unwrap()
                );
            }
            // …and back. The inverse transcode is the byte-level arm.
            let (back, _) = recompress_voxel_asset(&packed, BlockCodec::Raw).unwrap();
            assert_eq!(back, raw_image, "{codec:?} did not round-trip to raw");
            // Deterministic: same input, same codec, same bytes.
            let (again, _) = recompress_voxel_asset(&raw_image, codec).unwrap();
            assert_eq!(again, packed, "{codec:?} is not byte-deterministic");
        }
    }

    /// An SDF chunk is the most compressible thing this engine ships, and the
    /// **cook's chosen codec** must actually take that win here — an assertion
    /// that would pass vacuously if `chunks_compressed` were zero.
    #[test]
    fn the_cook_codec_compresses_sdf_chunks_hard() {
        let asset = build_voxel_asset(&sample_volume()).unwrap();
        let (_, report) = recompress_voxel_asset(asset.as_bytes(), COOK_CHUNK_CODEC).unwrap();
        assert_eq!(
            report.chunks_compressed, report.chunks,
            "every SDF chunk should compress"
        );
        assert!(
            report.ratio() < 0.5,
            "an SDF volume compressed to {:.3} — the corpus or the codec changed",
            report.ratio()
        );
    }

    /// The store seam borrows a raw chunk and owns a compressed one.
    #[test]
    fn the_chunk_store_seam_borrows_raw_and_owns_compressed() {
        use crate::residency::ChunkStore;
        let asset = build_voxel_asset(&sample_volume()).unwrap();
        let key = asset.reader().directory()[0].key;
        assert!(matches!(
            ChunkStore::chunk_bytes(&asset.reader(), key).unwrap(),
            Cow::Borrowed(_)
        ));

        let (packed, _) = recompress_voxel_asset(asset.as_bytes(), COOK_CHUNK_CODEC).unwrap();
        let r = VoxelAssetReader::new(packed).unwrap();
        let compressed = r
            .directory()
            .iter()
            .find(|e| e.codec != BlockCodec::Raw)
            .expect("the sample volume has a compressible chunk");
        assert!(matches!(
            ChunkStore::chunk_bytes(&r, compressed.key).unwrap(),
            Cow::Owned(_)
        ));
        assert_eq!(
            r.load_chunk(compressed.key).unwrap().unwrap(),
            asset.reader().chunk(compressed.key).unwrap().unwrap()
        );
    }

    #[test]
    fn a_payload_round_trips_and_is_byte_deterministic() {
        let v = sample_volume();
        let a = build_voxel_asset(&v).unwrap();
        let b = build_voxel_asset(&v).unwrap();
        assert_eq!(a.as_bytes(), b.as_bytes(), "two builds must be identical");

        let r = a.reader();
        assert_eq!(r.header().schema_version, VOXEL_ASSET_SCHEMA_VERSION);
        assert_eq!(r.header().chunk_dim, CHUNK_DIM);
        assert_eq!(r.voxel_size_m(), 0.5);
        assert_eq!(r.origin(), DVec3::new(10.0, -4.0, 2.0));
        assert_eq!(r.chunk_count(), v.chunk_count());
        assert!(!r.is_empty());

        for (&key, chunk) in v.chunks() {
            assert_eq!(&r.chunk(key).unwrap().unwrap(), chunk, "{key:?}");
        }
        assert!(r.chunk(ChunkKey::new(9, 9, 9)).unwrap().is_none());
        assert!(r.chunk_bytes(ChunkKey::new(9, 9, 9)).is_none());

        // The whole-volume door reproduces the authored state exactly.
        let back = r.to_voxel_data().unwrap();
        assert_eq!(back, v);
        assert!(!back.has_dirty_chunks(), "a load is not an edit");
        assert_eq!(build_voxel_asset(&back).unwrap().as_bytes(), a.as_bytes());

        // …and insertion order cannot reach the bytes.
        let mut shuffled = VoxelData::new(0.5).with_origin(v.origin());
        for (&key, chunk) in v.chunks().collect::<Vec<_>>().into_iter().rev() {
            shuffled.insert_chunk(key, chunk.clone());
        }
        assert_eq!(
            build_voxel_asset(&shuffled).unwrap().as_bytes(),
            a.as_bytes()
        );
    }

    #[test]
    fn the_directory_is_ascending_aligned_and_non_overlapping() {
        let v = sample_volume();
        let a = build_voxel_asset(&v).unwrap();
        let r = a.reader();

        let keys: Vec<ChunkKey> = r.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "the directory must be ChunkKey-ascending");

        let mut prev_end = r.header().blob_base;
        for e in r.directory() {
            assert_eq!(e.offset % CHUNK_ALIGN, 0, "{e:?} is misaligned");
            assert!(e.offset >= prev_end, "{e:?} overlaps its predecessor");
            prev_end = e.offset + e.len;
            // The borrowed slice IS the canonical blob.
            assert_eq!(
                r.chunk_bytes(e.key).unwrap(),
                encode_chunk(v.get_chunk(e.key).unwrap()).unwrap()
            );
        }
        assert_eq!(
            r.header().blob_base,
            (HEADER_LEN + DIR_ENTRY_LEN * r.chunk_count() as u64).next_multiple_of(CHUNK_ALIGN)
        );
        assert_eq!(a.as_bytes().len() as u64 % CHUNK_ALIGN, 0);
    }

    #[test]
    fn an_empty_volume_is_a_valid_payload() {
        let a = build_voxel_asset(&VoxelData::new(1.0)).unwrap();
        let r = a.reader();
        assert!(r.is_empty());
        assert_eq!(r.chunk_count(), 0);
        assert_eq!(r.header().blob_base, HEADER_LEN);
        assert_eq!(a.as_bytes().len() as u64, HEADER_LEN);
        assert!(r.to_voxel_data().unwrap().is_empty());
    }

    #[test]
    fn a_duplicate_chunk_is_an_error() {
        let mut b = VoxelAssetBuilder::new(0.5);
        let key = ChunkKey::new(1, 2, 3);
        b.insert(key, &VoxelChunk::empty()).unwrap();
        assert_eq!(
            b.insert(key, &VoxelChunk::empty()),
            Err(VoxelAssetError::DuplicateChunk { x: 1, y: 2, z: 3 })
        );
        assert_eq!(b.len(), 1);
        assert!(!b.is_empty());
    }

    #[test]
    fn malformed_payloads_are_rejected() {
        let v = sample_volume();
        let good = build_voxel_asset(&v).unwrap().into_bytes();

        assert_eq!(
            VoxelAsset::from_bytes(good[..32].to_vec()),
            Err(VoxelAssetError::TooShort)
        );

        let mut bad = good.clone();
        bad[0] = b'X';
        assert_eq!(VoxelAsset::from_bytes(bad), Err(VoxelAssetError::BadMagic));

        let mut newer = good.clone();
        newer[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            VoxelAsset::from_bytes(newer),
            Err(VoxelAssetError::SchemaTooNew {
                found: 99,
                current: VOXEL_ASSET_SCHEMA_VERSION
            })
        );

        // A foreign chunk dimension fails HERE, naming both numbers.
        let mut dim = good.clone();
        dim[12..16].copy_from_slice(&32u32.to_le_bytes());
        let err = VoxelAsset::from_bytes(dim).unwrap_err().to_string();
        assert!(
            err.contains("32") && err.contains(&CHUNK_DIM.to_string()),
            "{err}"
        );

        // A degenerate voxel scale.
        let mut scale = good.clone();
        scale[16..24].copy_from_slice(&0f64.to_le_bytes());
        assert!(VoxelAsset::from_bytes(scale).is_err());

        // The reserved words are validated zero, not merely written zero — both
        // of them, so a future writer cannot claim either without a version bump.
        // (See the module docs for why the alternative is silent misreading.)
        let mut hdr_reserved = good.clone();
        hdr_reserved[52..56].copy_from_slice(&1u32.to_le_bytes());
        let err = VoxelAsset::from_bytes(hdr_reserved)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reserved header word"), "{err}");

        // IASSET1 (v2) SPLIT the directory entry's reserved word: byte 12 is the
        // block codec now, bytes 13..16 are still reserved. So the strict check
        // is in three parts, and each one is asserted where it lives.
        //
        // (a) At v2, an unknown codec byte is refused **by name** — never read as
        //     raw, which would hand a compressed frame to `bincode` and blame it.
        let r = HEADER_LEN as usize + 12;
        let mut bad_codec = good.clone();
        bad_codec[r] = 200;
        let err = VoxelAsset::from_bytes(bad_codec).unwrap_err().to_string();
        assert!(err.contains("codec 200"), "{err}");

        // (b) At v2, the remaining three reserved bytes are still claimable only
        //     by a version bump.
        let mut dir_reserved = good.clone();
        dir_reserved[r + 1] = 7;
        let err = VoxelAsset::from_bytes(dir_reserved)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reserved tail"), "{err}");

        // (c) Below v2 the WHOLE word must be zero — the original rule, kept for
        //     the v1 branch. This is what makes "a v1 asset has no codec" checked
        //     rather than assumed: a v1 image with a byte in that word does not
        //     load as a compressed one, it does not load at all.
        let mut v1_reserved = good.clone();
        v1_reserved[8..12].copy_from_slice(&1u32.to_le_bytes());
        v1_reserved[r..r + 4].copy_from_slice(&7u32.to_le_bytes());
        let err = VoxelAsset::from_bytes(v1_reserved).unwrap_err().to_string();
        assert!(err.contains("reserved word"), "{err}");

        // … and the good payload really does have zeros there, so the two checks
        // above are not passing on a coincidence.
        assert_eq!(&good[52..56], &[0, 0, 0, 0]);
        assert_eq!(&good[r..r + 4], &[0, 0, 0, 0]);

        // A misaligned blob offset in the directory.
        let mut misaligned = good.clone();
        let off = HEADER_LEN as usize + 16;
        let cur = u64::from_le_bytes(misaligned[off..off + 8].try_into().unwrap());
        misaligned[off..off + 8].copy_from_slice(&(cur + 1).to_le_bytes());
        let err = VoxelAsset::from_bytes(misaligned).unwrap_err().to_string();
        assert!(err.contains("aligned"), "{err}");

        // An out-of-order directory (which would break the binary search).
        let mut unsorted = good.clone();
        let a = HEADER_LEN as usize;
        let b = a + DIR_ENTRY_LEN as usize;
        for i in 0..12 {
            unsorted.swap(a + i, b + i);
        }
        let err = VoxelAsset::from_bytes(unsorted).unwrap_err().to_string();
        assert!(err.contains("ascending"), "{err}");

        // An overlapping blob — and it must trip the **overlap** arm, not the
        // bounds arm. The first entry's length is stretched to reach one byte past
        // its neighbour's start while staying comfortably inside the payload, so
        // the only rule it breaks is the one under test. (An `||` over both arms
        // would have let a bounds failure stand in for an overlap failure forever,
        // and the overlap check is the one nothing else can catch: two entries
        // sharing bytes means one chunk's edit corrupts another, invisibly.)
        let r = VoxelAsset::from_bytes(good.clone()).unwrap();
        let dir: Vec<_> = r.reader().directory().to_vec();
        assert!(dir.len() >= 2, "the fixture needs two blobs to overlap");
        let reach = dir[1].offset + 1 - dir[0].offset;
        assert!(
            dir[0].offset + reach < good.len() as u64,
            "the mutation must stay in bounds or it proves nothing"
        );
        let mut overlap = good.clone();
        let len_at = HEADER_LEN as usize + 24;
        overlap[len_at..len_at + 8].copy_from_slice(&reach.to_le_bytes());
        let err = VoxelAsset::from_bytes(overlap).unwrap_err().to_string();
        assert!(
            err.contains("overlaps"),
            "the in-bounds overlap must trip the OVERLAP arm, got: {err}"
        );
    }

    #[test]
    fn a_payload_writes_and_reads_from_a_loose_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deep").join("Cave.inf_voxel");
        let v = sample_volume();
        let asset = build_voxel_asset(&v).unwrap();
        let written = write_voxel_asset(&path, &asset).unwrap();
        assert_eq!(written, asset.as_bytes());
        assert!(path.exists());
        let back = read_voxel_asset(&path).unwrap();
        assert_eq!(back.as_bytes(), asset.as_bytes());
        assert_eq!(back.reader().to_voxel_data().unwrap(), v);
        // No temp litter beside the asset.
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The kind + schema the database is owed live as inherent consts, because
    /// the generic `AssetPayload` door is deliberately closed (see the module
    /// docs — a bincode length prefix would misalign every chunk).
    #[test]
    fn the_asset_publishes_its_kind_without_the_generic_door() {
        assert_eq!(VoxelAsset::KIND, AssetKind::VoxelVolume);
        assert_eq!(VoxelAsset::KIND.extension(), Some("inf_voxel"));
        assert_eq!(VoxelAsset::SCHEMA_VERSION, VOXEL_ASSET_SCHEMA_VERSION);
        // Streaming-class: it must NOT compress, or `read_ref` could never hand
        // back a borrowed, aligned slice.
        assert!(!inf_asset::PackWriter::compresses_kind(VoxelAsset::KIND));
    }
}
