//! The `.inf_pack` archive: a single-file, content-addressed asset pack (P9.2).
//!
//! A pack is what a **shipped game loads** instead of a loose Content directory:
//! one file holding every cooked asset's payload, keyed by [`AssetId`], with a
//! sorted index and zstd-compressed blobs. It is the cook pipeline's output
//! (`runtime/inf-packager`) and the player's input (`runtime/inf-player`).
//!
//! # On-disk layout (little-endian throughout)
//!
//! ```text
//! ┌ header (16 bytes) ────────────────────────────────────────────────┐
//! │  magic         [u8; 8]   b"INFPACK\0"                              │
//! │  format_ver    u32       = PACK_FORMAT_VERSION                     │
//! │  entry_count   u32                                                 │
//! ├ index (entry_count × 60 bytes, SORTED BY GUID) ───────────────────┤
//! │  guid          [u8; 16]  Uuid big-endian bytes                     │
//! │  kind_code     u16       stable AssetKind code (see kind_code)     │
//! │  flags         u16       bit0 = payload is zstd-compressed         │
//! │  content_hash  u128      xxh3-128 of the UNCOMPRESSED payload      │
//! │  offset        u64       absolute file offset of this blob         │
//! │  stored_len    u64       bytes on disk (compressed size if bit0)   │
//! │  uncompressed  u64       decompressed payload length               │
//! ├ blob section (blobs at offsets, zero-padded to BLOB_ALIGN) ───────┤
//! │  … blob 0 … pad … blob 1 …                                         │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Determinism (the P9.2 gate)
//!
//! The same inputs always produce a **byte-identical** pack: the index is sorted
//! by GUID, blob offsets are assigned in that order, compression uses a fixed
//! level with no timestamp/dictionary, alignment padding is deterministic zeros,
//! and nothing wall-clock enters the bytes.
//!
//! # Zero-copy reads (P16.1)
//!
//! [`PackReader::open`] **memory-maps** the file (native targets) instead of
//! reading it whole, and [`read_ref`](PackReader::read_ref) hands back a
//! `Cow::Borrowed(&[u8])` pointing straight into the mapping for an uncompressed
//! entry — no allocation, no copy, and only the touched pages fault in. That is
//! what lets a streaming runtime (P16.3 terrain tiles, P18.2 meshlets) page a
//! multi-GB ship pack under a bounded residency budget. Compressed entries still
//! decode into an owned `Vec` (`Cow::Owned`).
//!
//! Blob *offsets* are 16-byte aligned from `format_ver` 2 onward, and every
//! backing this reader builds has a 16-byte-aligned *base* (an mmap is
//! page-aligned; the owned buffer is over-allocated to match). Address
//! alignment is the conjunction of the two, so on a v2 pack a borrowed slice
//! can go straight to a GPU upload or a `[f32]`/`[u32]` view with no realigning
//! copy. **Neither half alone is enough** — a v1 pack's blobs sit at arbitrary
//! offsets, so slices borrowed out of one carry no alignment guarantee.
//!
//! # Integrity (verify-once)
//!
//! Every entry's xxh3-128 is checked against the index's `content_hash` on its
//! **first** read through a given [`PackReader`], and the success is recorded in a
//! per-entry atomic mark; later reads of that entry skip the rehash. A corrupt
//! blob therefore still fails on first access (and keeps failing, since the mark
//! is only set on success) without taxing every subsequent read — the difference
//! between "hash 4 GB once per streamed tile" and "hash it every frame".
//!
//! # Schema-version discipline (ROADMAP §3)
//!
//! `format_ver` gates the whole container. A reader rejects a pack whose
//! `format_ver` is newer than [`PACK_FORMAT_VERSION`]; future revisions add a
//! decode arm keyed on the version rather than reinterpreting the bytes.
//!
//! - **v1** (P9.2): blobs concatenated with no padding.
//! - **v2** (P16.1): blobs padded to [`BLOB_ALIGN`]-byte offsets. The reader
//!   still loads v1 packs verbatim — the index carries absolute offsets either
//!   way, so the only version-keyed behaviour is that v2 *enforces* alignment.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use uuid::Uuid;

use crate::error::{AssetError, Result};
use crate::hash::ContentHash;
use crate::id::AssetId;
use crate::kind::AssetKind;

/// Magic bytes at the start of every `.inf_pack`.
pub const PACK_MAGIC: [u8; 8] = *b"INFPACK\0";

/// The current container format version (bump on a breaking layout change).
///
/// - v1 (P9.2): unpadded blob section.
/// - v2 (P16.1): blobs start on [`BLOB_ALIGN`]-byte offsets.
pub const PACK_FORMAT_VERSION: u32 = 2;

/// The oldest `format_ver` this build still reads.
pub const PACK_MIN_READ_VERSION: u32 = 1;

/// Bytes of the fixed header.
const HEADER_LEN: u64 = 16;

/// Bytes of one index entry.
const ENTRY_LEN: u64 = 60;

/// Blob offsets are multiples of this from `format_ver` 2 onward.
///
/// 16 bytes covers every alignment the engine hands a borrowed pack slice to:
/// `f32x4`/`u32x4` views, GPU buffer uploads (WebGPU requires 4, most drivers
/// prefer 16), and the meshlet/terrain-tile headers P18.2 / P16.3 will read
/// in place. The padding is deterministic zeros, so the byte-identical rebuild
/// gate is untouched.
///
/// This is an *offset* alignment; a borrowed slice's **address** is aligned only
/// because every [`PackBacking`] is 16-byte-aligned at its base too.
pub const BLOB_ALIGN: u64 = 16;

/// Round `n` up to the next multiple of [`BLOB_ALIGN`].
fn align_up(n: u64) -> u64 {
    n.next_multiple_of(BLOB_ALIGN)
}

/// zstd compression level used for cooked blobs. Cook is an offline step, so a
/// high level trades build time for a smaller ship. Determinism is independent
/// of the level; this is a pure size/speed knob (tunable later). Only the native
/// encoder uses it — the wasm build decodes only (see `zstd_decode`).
#[cfg(not(target_arch = "wasm32"))]
const ZSTD_LEVEL: i32 = 19;

/// Payloads at least this large are considered for compression (below it the
/// zstd frame overhead usually loses; such blobs are stored raw).
const COMPRESS_THRESHOLD: usize = 64;

/// A stable numeric code for an [`AssetKind`] in the pack index.
///
/// The match is exhaustive on purpose: adding a new `AssetKind` forces a
/// deliberate code assignment here (a compile error otherwise), so codes never
/// silently shift.
fn kind_code(kind: AssetKind) -> u16 {
    match kind {
        AssetKind::Unknown => 0,
        AssetKind::Level => 1,
        AssetKind::Mesh => 2,
        AssetKind::Texture => 3,
        AssetKind::Material => 4,
        AssetKind::MaterialInstance => 5,
        AssetKind::Blueprint => 6,
        AssetKind::FunctionLib => 7,
        AssetKind::Struct => 8,
        AssetKind::Enum => 9,
        AssetKind::Table => 10,
        AssetKind::Audio => 11,
        AssetKind::Pcg => 12,
        AssetKind::Skeleton => 13,
        AssetKind::AnimClip => 14,
        AssetKind::StateMachine => 15,
        AssetKind::MeshletMesh => 16,
        AssetKind::Terrain => 17,
        AssetKind::Partition => 18,
        AssetKind::BiomeSet => 19,
        AssetKind::VoxelVolume => 20,
        // P22.2. Appended, never inserted: a kind code is a WIRE value and every
        // pack ever cooked reads its index by it.
        AssetKind::Fracture => 21,
        // P24.4. Appended, never inserted — same rule, same reason.
        AssetKind::Cloth => 22,
        AssetKind::Hair => 23,
        // P26.3b. Appended, never inserted — same rule, same reason.
        AssetKind::DerivedMaterial => 24,
    }
}

/// Inverse of [`kind_code`]; unknown codes map to [`AssetKind::Unknown`].
fn kind_from_code(code: u16) -> AssetKind {
    match code {
        1 => AssetKind::Level,
        2 => AssetKind::Mesh,
        3 => AssetKind::Texture,
        4 => AssetKind::Material,
        5 => AssetKind::MaterialInstance,
        6 => AssetKind::Blueprint,
        7 => AssetKind::FunctionLib,
        8 => AssetKind::Struct,
        9 => AssetKind::Enum,
        10 => AssetKind::Table,
        11 => AssetKind::Audio,
        12 => AssetKind::Pcg,
        13 => AssetKind::Skeleton,
        14 => AssetKind::AnimClip,
        15 => AssetKind::StateMachine,
        16 => AssetKind::MeshletMesh,
        17 => AssetKind::Terrain,
        18 => AssetKind::Partition,
        19 => AssetKind::BiomeSet,
        20 => AssetKind::VoxelVolume,
        21 => AssetKind::Fracture,
        22 => AssetKind::Cloth,
        23 => AssetKind::Hair,
        24 => AssetKind::DerivedMaterial,
        _ => AssetKind::Unknown,
    }
}

/// One entry in a pack index (metadata only; the payload lives in the blob
/// section and is fetched by [`PackReader::read`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackEntry {
    /// The asset's stable GUID (the lookup key).
    pub guid: AssetId,
    /// What kind of asset this is.
    pub kind: AssetKind,
    /// xxh3-128 of the **uncompressed** payload — the integrity signal.
    pub content_hash: ContentHash,
    /// Absolute file offset of this blob.
    pub offset: u64,
    /// Bytes stored on disk (the compressed size when [`compressed`](Self::compressed)).
    pub stored_len: u64,
    /// Decompressed payload length.
    pub uncompressed_len: u64,
    /// Whether the blob is zstd-compressed.
    pub compressed: bool,
}

/// Accumulates assets and writes a deterministic `.inf_pack`.
///
/// Add payloads with [`add_bytes`](Self::add_bytes) or straight from an
/// [`AssetEntry`](crate::AssetEntry) with [`add_entry`](Self::add_entry); the
/// writer compresses each blob up front and keeps entries sorted by GUID (via
/// the `BTreeMap`), so [`write`](Self::write) is a single deterministic pass.
#[derive(Default)]
pub struct PackWriter {
    items: BTreeMap<AssetId, PackItem>,
}

struct PackItem {
    kind: AssetKind,
    content_hash: ContentHash,
    uncompressed_len: u64,
    /// The bytes actually stored (compressed or raw).
    stored: Vec<u8>,
    compressed: bool,
}

impl PackWriter {
    /// A fresh, empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of assets added.
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// True once `guid` has been added.
    pub fn contains(&self, guid: AssetId) -> bool {
        self.items.contains_key(&guid)
    }

    /// Whether blobs of `kind` are considered for compression — the per-kind
    /// compression policy seam (P16.1).
    ///
    /// **Streaming-class** kinds cook *uncompressed* so a runtime can page them
    /// straight out of an mmap'd pack as borrowed slices
    /// ([`PackReader::read_ref`]) with no decode step and no allocation: a tile
    /// or meshlet page that must land within a frame budget cannot afford a zstd
    /// pass, and a compressed blob can never be a zero-copy read. The trade is
    /// ship size for streaming latency, and it is deliberate.
    ///
    /// Today that is [`AssetKind::MeshletMesh`] (`.inf_vmesh`, read via mmap
    /// slices in P18.2), [`AssetKind::Terrain`] (`.inf_terrain`, P16.3) and
    /// [`AssetKind::Partition`] (`.inf_part`, P16.5): each is a header +
    /// directory + **16-byte-aligned per-page blobs**, and the whole point of that
    /// layout is that a streamer slices one page out of the mapping without
    /// touching the rest. zstd would force a whole-asset decode — hundreds of MB
    /// to page one tile, or every cell in the world to spawn one — so they stay
    /// raw.
    ///
    /// **[`AssetKind::Level`] deliberately stays compressed**, which is exactly
    /// why P16.5's partition could not ride inside the level entry: a compressed
    /// entry can never be `read_ref`'d, and flipping `Level` to raw would move the
    /// bytes of every pack ever cooked.
    ///
    /// Everything else keeps the P9.2 behaviour: zstd above
    /// [`COMPRESS_THRESHOLD`] bytes, stored raw when compression would inflate.
    ///
    /// The match is exhaustive on purpose, exactly like [`kind_code`]: a new
    /// [`AssetKind`] must not silently inherit "compress me" — it fails to
    /// compile until someone decides whether it is streaming-class.
    pub fn compresses_kind(kind: AssetKind) -> bool {
        match kind {
            // Streaming-class: paged out of the mapping as borrowed slices.
            AssetKind::MeshletMesh
            | AssetKind::Terrain
            | AssetKind::Partition
            // P21.1: a `.inf_voxel` is paged chunk-at-a-time out of the mapping,
            // exactly like a `.inf_terrain` tile — 16-byte-aligned blobs a reader
            // sub-slices with no decode and no copy. Compressing it would defeat
            // the entire container layout.
            | AssetKind::VoxelVolume
            // P26.1: a `.inf_tex` is a **tiled** container now — header +
            // directories + 16-byte-aligned 136² tile blobs — and streaming
            // virtual texturing pages ONE of those tiles per request, inside a
            // frame. A zstd frame would force a whole-texture decode to reach one
            // tile, which is precisely the CPU-decompression bottleneck the SVT
            // direction memo says to design out of existence rather than optimise.
            //
            // The trade is ship size, it is deliberate, and it is bigger than the
            // lost zstd frame — which is the part that is easy to say and is not
            // where the bytes go. A v2 payload is larger than the v1 record of the
            // same source before any pack touches it, because every tile stores a
            // 4-texel border ring on each side (136²/128² = 1.13) and every tail
            // mip occupies a whole 136² tile however small it is. Measured in
            // `inf_material::tiles`: **6.8×** at 128² BC1, 2.55× at 256², 1.22× at
            // 1024², 1.16× at 2048² — and an uncompressed-format texture grows by
            // the same factor as a BC one, so "BC payloads barely compress" does
            // not describe the cost either. It is a small-texture cost, and
            // packing the tail mips into one shared page is the named follow-up.
            //
            // A v1 (bincode) `.inf_tex` from an older cook rides along raw too.
            // That costs a little size and nothing else — it is read whole either
            // way, through `TextureAsset::from_payload`.
            | AssetKind::Texture => false,
            AssetKind::Unknown
            | AssetKind::Level
            | AssetKind::Mesh
            | AssetKind::Material
            | AssetKind::MaterialInstance
            | AssetKind::Blueprint
            | AssetKind::FunctionLib
            | AssetKind::Struct
            | AssetKind::Enum
            | AssetKind::Table
            | AssetKind::Audio
            | AssetKind::Pcg
            | AssetKind::Skeleton
            | AssetKind::AnimClip
            | AssetKind::StateMachine
            // P19.2: a biome set is a short list of names + colours — authored,
            // never paged, so it compresses like every other data asset.
            | AssetKind::BiomeSet
            // P22.2: a `.inf_fracture` COMPRESSES, and the decision is worth
            // stating because it is the first *derived* kind that does.
            //
            // The other three derived/streamed containers are raw because a
            // runtime pages ONE unit out of them under a frame budget — one
            // terrain tile, one meshlet page, one partition cell — and a zstd
            // frame would force a whole-asset decode to reach it. A fracture has
            // no such access pattern: it is read exactly once, in full, at the
            // instant its mesh breaks, and a chunk set missing chunks is not a
            // cheaper load but a hole in a wall. There is nothing to sub-slice,
            // so the raw-blob trade (ship size for streaming latency) would be
            // paid for nothing. It stays compressed.
            | AssetKind::Fracture
            // P24.4: a `.inf_cloth` / `.inf_hair` is read WHOLE, once, when its
            // wearer spawns — the fracture argument above, met again. Half a
            // constraint set is not a cheaper load but a coat that tears, and
            // half a guide set is a bald patch, so there is nothing to sub-slice
            // and no streaming latency to buy. They compress.
            | AssetKind::Cloth
            | AssetKind::Hair
            // P26.3b: a `.inf_matd` is a handful of GUIDs and scalars, read whole
            // when a level loads. There is no unit to page and nothing to
            // sub-slice; the `.inf_tex` payloads it POINTS AT are the streaming
            // half, and those are already raw.
            | AssetKind::DerivedMaterial => true,
        }
    }

    /// Add an asset from raw payload bytes. Compresses eagerly (subject to
    /// [`compresses_kind`](Self::compresses_kind)); a duplicate GUID is an error
    /// (the cook resolves ids before packing).
    pub fn add_bytes(&mut self, guid: AssetId, kind: AssetKind, payload: &[u8]) -> Result<()> {
        if self.items.contains_key(&guid) {
            return Err(AssetError::Pack(format!("duplicate guid {guid} in pack")));
        }
        let content_hash = ContentHash::of(payload);
        let (stored, compressed) = if Self::compresses_kind(kind) {
            maybe_compress(payload)?
        } else {
            (payload.to_vec(), false)
        };
        self.items.insert(
            guid,
            PackItem {
                kind,
                content_hash,
                uncompressed_len: payload.len() as u64,
                stored,
                compressed,
            },
        );
        Ok(())
    }

    /// Add an asset by reading its payload from an [`AssetEntry`](crate::AssetEntry).
    pub fn add_entry(&mut self, entry: &crate::AssetEntry) -> Result<()> {
        let bytes = std::fs::read(&entry.path)?;
        self.add_bytes(entry.id(), entry.kind(), &bytes)
    }

    /// Serialize the whole pack to `w`.
    pub fn write<W: Write>(&self, w: &mut W) -> Result<()> {
        let count = self.items.len() as u32;
        let mut header = Vec::with_capacity(HEADER_LEN as usize);
        header.extend_from_slice(&PACK_MAGIC);
        header.extend_from_slice(&PACK_FORMAT_VERSION.to_le_bytes());
        header.extend_from_slice(&count.to_le_bytes());
        w.write_all(&header)?;

        // Blob offsets, assigned in GUID order: the section starts after the
        // header + full index, rounded up to BLOB_ALIGN, and every blob after it
        // starts on the next aligned boundary (format v2). Zero-length blobs
        // consume no space, so two of them can legitimately share an offset.
        let mut offsets = Vec::with_capacity(self.items.len());
        let mut offset = align_up(HEADER_LEN + ENTRY_LEN * count as u64);
        for item in self.items.values() {
            offsets.push(offset);
            offset = align_up(offset + item.stored.len() as u64);
        }

        // Index (already GUID-sorted by the BTreeMap).
        for ((guid, item), &offset) in self.items.iter().zip(&offsets) {
            let mut e = Vec::with_capacity(ENTRY_LEN as usize);
            e.extend_from_slice(guid.uuid().as_bytes()); // 16
            e.extend_from_slice(&kind_code(item.kind).to_le_bytes()); // 2
            e.extend_from_slice(&(item.compressed as u16).to_le_bytes()); // 2
            e.extend_from_slice(&item.content_hash.0.to_le_bytes()); // 16
            e.extend_from_slice(&offset.to_le_bytes()); // 8
            e.extend_from_slice(&(item.stored.len() as u64).to_le_bytes()); // 8
            e.extend_from_slice(&item.uncompressed_len.to_le_bytes()); // 8
            debug_assert_eq!(e.len() as u64, ENTRY_LEN);
            w.write_all(&e)?;
        }

        // Blob section, in the same GUID order, zero-padded up to each offset.
        // The padding is a deterministic constant, so two cooks of the same
        // inputs stay byte-identical (the P9.2 gate).
        const PAD: [u8; BLOB_ALIGN as usize] = [0u8; BLOB_ALIGN as usize];
        let mut pos = HEADER_LEN + ENTRY_LEN * count as u64;
        for (item, &offset) in self.items.values().zip(&offsets) {
            debug_assert!(pos <= offset && offset - pos < BLOB_ALIGN);
            w.write_all(&PAD[..(offset - pos) as usize])?;
            w.write_all(&item.stored)?;
            pos = offset + item.stored.len() as u64;
        }
        Ok(())
    }

    /// Write the pack to a file (creating parent directories), **atomically**.
    ///
    /// The bytes go to a sibling temp file that is then `rename`d over `path`,
    /// never written in place. That is not just crash-safety: a re-cook
    /// routinely targets a fixed `Build/` directory while a player process may
    /// still hold a live mapping of the pack it is running
    /// ([`PackReader::open`]). Truncating that file under the mapping is exactly
    /// the mutation the reader's SAFETY contract forbids — a SIGBUS on Unix, an
    /// opaque `os error 1224` on Windows. `rename` instead unlinks the old name
    /// and leaves the old inode intact, so an existing reader keeps seeing the
    /// bytes it verified while new readers get the new pack.
    ///
    /// (Windows reaches the same outcome by a different route: `std::fs::rename`
    /// first tries `MoveFileExW(MOVEFILE_REPLACE_EXISTING)`, and only if that
    /// comes back `ERROR_ACCESS_DENIED` — which is what an active mapping on the
    /// destination provokes — does it retry with `FileRenameInfoEx` +
    /// `POSIX_SEMANTICS`, which unlinks the name and leaves the mapping valid.
    /// `rewriting_a_pack_under_a_live_reader_is_safe` covers it. If both routes
    /// fail the caller simply gets an error and the original pack is untouched,
    /// never truncated — the safe failure.)
    ///
    /// Durability is deliberately *not* guaranteed: there is no fsync before the
    /// rename, because a pack is a rebuildable build artifact, not a document.
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let mut buf = Vec::new();
        self.write(&mut buf)?;
        // The mechanism this doc describes now lives in one place —
        // [`crate::atomic::write_atomically`] — which is where every other
        // document door in the engine reaches it too. Never leave temp litter
        // next to a ship artifact: a mid-cook failure (disk full is the
        // realistic one) cleans up after itself just as a failed rename does.
        Ok(crate::atomic::write_atomically(path, &buf)?)
    }

    /// Serialize the pack to an in-memory buffer.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.write(&mut buf)?;
        Ok(buf)
    }
}

/// Compress a payload if it is worth it; returns `(stored_bytes, compressed?)`.
fn maybe_compress(payload: &[u8]) -> Result<(Vec<u8>, bool)> {
    if payload.len() < COMPRESS_THRESHOLD {
        return Ok((payload.to_vec(), false));
    }
    let packed = zstd_encode(payload)?;
    if packed.len() < payload.len() {
        Ok((packed, true))
    } else {
        // Incompressible: store raw so we never inflate.
        Ok((payload.to_vec(), false))
    }
}

/// zstd encode — native uses the C `zstd` at [`ZSTD_LEVEL`]; wasm cannot encode
/// (packs are written by the cook, which never runs in a browser), so this is an
/// error there. Keeping it a function (not an inline call) is what lets the
/// wasm build compile without pulling zstd-sys.
#[cfg(not(target_arch = "wasm32"))]
fn zstd_encode(payload: &[u8]) -> Result<Vec<u8>> {
    zstd::encode_all(payload, ZSTD_LEVEL).map_err(|e| AssetError::Pack(format!("zstd: {e}")))
}

#[cfg(target_arch = "wasm32")]
fn zstd_encode(_payload: &[u8]) -> Result<Vec<u8>> {
    Err(AssetError::Pack(
        "pack writing (zstd encode) is not supported on wasm — packs are cooked on desktop".into(),
    ))
}

/// zstd decode — native uses the C `zstd`; wasm uses the pure-Rust `ruzstd`
/// streaming decoder (the browser player reads packs it fetched over HTTP). Both
/// produce identical bytes, so the content-hash integrity check downstream is
/// backend-agnostic.
#[cfg(not(target_arch = "wasm32"))]
fn zstd_decode(stored: &[u8]) -> Result<Vec<u8>> {
    zstd::decode_all(stored).map_err(|e| AssetError::Pack(format!("zstd: {e}")))
}

#[cfg(target_arch = "wasm32")]
fn zstd_decode(stored: &[u8]) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(stored)
        .map_err(|e| AssetError::Pack(format!("ruzstd init: {e}")))?;
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| AssetError::Pack(format!("ruzstd: {e}")))?;
    Ok(out)
}

/// A 16-byte block — the alignment carrier for [`AlignedBytes`].
///
/// `repr(align(16))` states the requirement in the language, where it cannot
/// drift: it is the type's alignment by definition, on every target and every
/// release. A `Vec<u128>` would happen to work on the targets we ship today,
/// but only as a consequence of `u128`'s ABI alignment — which is a platform
/// detail Rust has already changed once (it was raised to 16 in 1.77). This
/// buffer is load-bearing for a documented format guarantee, so it should not
/// rest on a fact that a future toolchain is free to revise.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Align16([u8; BLOB_ALIGN as usize]);

/// An owned byte buffer whose **base address** is 16-byte aligned.
///
/// A plain `Vec<u8>` is 1-byte aligned, so `base + aligned_offset` says nothing
/// about the resulting address. Since `Owned` is the *only* backing on wasm
/// (and the native mmap fallback), a consumer that trusted the format's
/// alignment promise and did `bytemuck::cast_slice::<_, u32>(blob)` would be
/// at the mercy of whatever base the allocator happened to hand out — and would
/// fail in the browser player and nowhere else. Carrying the alignment in the
/// buffer type makes the promise uniform across both backings.
///
/// **Public since P18.2**: every container that ships aligned sections and may be
/// held *owned* rather than mapped — `.inf_vmesh`'s lifted-v1 and loose-file
/// backings, `.inf_terrain`'s — needs the same carrier, and re-deriving it per
/// crate is how one of them ends up with a bare `Vec<u8>` and a
/// `bytemuck::cast_slice` that panics only in the browser.
pub struct AlignedBytes {
    /// Storage, over-allocated to a whole number of 16-byte blocks.
    blocks: Vec<Align16>,
    /// The meaningful prefix length in bytes.
    len: usize,
}

impl AlignedBytes {
    /// Copy `bytes` into a fresh 16-byte-aligned allocation.
    pub fn copy_from(bytes: &[u8]) -> Self {
        let mut blocks =
            vec![Align16([0u8; BLOB_ALIGN as usize]); bytes.len().div_ceil(BLOB_ALIGN as usize)];
        // SAFETY: `blocks` holds ceil(len/16)*16 >= bytes.len() initialized
        // bytes, and `Align16` is a plain byte array, so writing bytes through
        // the pointer keeps every value valid. The regions cannot overlap —
        // `blocks` was just allocated.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                blocks.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        Self {
            blocks,
            len: bytes.len(),
        }
    }

    /// The bytes, at a 16-byte-aligned base address.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation covers ceil(len/16)*16 >= len initialized bytes
        // and lives as long as `self`; `Align16` is `[u8; 16]` underneath, so
        // reading it as bytes is well-defined. An empty `Vec` still yields the
        // type's dangling-but-aligned pointer, which is valid for a 0-length
        // slice.
        unsafe { std::slice::from_raw_parts(self.blocks.as_ptr().cast::<u8>(), self.len) }
    }

    /// Bytes held.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl std::ops::Deref for AlignedBytes {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for AlignedBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::fmt::Debug for AlignedBytes {
    /// Summarizes; never dumps a payload that may be hundreds of megabytes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBytes")
            .field("len", &self.len)
            .finish()
    }
}

/// Where a [`PackReader`]'s bytes live.
///
/// The two arms are the whole point of P16.1: a shipped pack opened from disk is
/// **mapped**, so reading an uncompressed entry touches only its pages and hands
/// out a borrowed slice; a pack that arrived as bytes (the wasm player's HTTP
/// fetch, an in-memory test) keeps an owned — and equally aligned — buffer.
/// Everything above this enum sees one `&[u8]` whose base is a multiple of
/// [`BLOB_ALIGN`], which is what turns v2's aligned *offsets* into aligned
/// *addresses*.
enum PackBacking {
    /// A memory-mapped file (native only). Page-aligned by construction.
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(memmap2::Mmap),
    /// An owned buffer: `from_bytes`, wasm, or the mmap fallback.
    Owned(AlignedBytes),
}

impl PackBacking {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            PackBacking::Mmap(map) => map,
            PackBacking::Owned(buf) => buf.as_slice(),
        }
    }
}

/// A read-only view over a `.inf_pack`.
///
/// `open` memory-maps the file (native) and `from_bytes` takes ownership of a
/// buffer; either way the header + index are parsed once up front.
/// [`read_ref`](Self::read_ref) then fetches one payload on demand — borrowed
/// straight out of the backing for an uncompressed entry, decoded into an owned
/// `Vec` for a compressed one — and [`read`](Self::read) is the owning
/// convenience wrapper over it.
///
/// Integrity is **verify-once per entry per reader** (see the module docs): the
/// first successful read of an entry hashes it, later reads trust the mark.
pub struct PackReader {
    data: PackBacking,
    format_version: u32,
    /// Index entries in file order.
    entries: Vec<PackEntry>,
    /// GUID → slot in `entries`, which also gives the GUID-ordered iteration.
    by_guid: BTreeMap<AssetId, usize>,
    /// Per-entry "already integrity-checked" marks, parallel to `entries`.
    ///
    /// Atomics (not a `Mutex`/`RefCell`) keep [`read_ref`](Self::read_ref) a
    /// `&self` method that many streaming threads can call at once. A benign
    /// race just hashes the same entry twice; the mark is only ever set on a
    /// *successful* verification, so corruption can never be marked clean.
    ///
    /// **Granularity is one whole entry**: the first touch hashes the entire
    /// blob, however large. That is right for today's access pattern (a pack
    /// entry is read in full), but a P16.3+ streamer that range-reads a slice of
    /// one multi-hundred-MB terrain or meshlet entry would pay the whole hash to
    /// read a page of it. Sub-entry (page/chunk-granular) verification — a
    /// per-chunk hash table in the index — is the follow-up that unlocks that,
    /// and is deliberately out of scope here.
    verified: Vec<AtomicBool>,
    /// How many integrity hashes this reader has actually computed.
    verify_calls: AtomicU64,
}

impl PackReader {
    /// Open a pack from a file, memory-mapping it when possible.
    ///
    /// Falls back to a whole-file read if the mapping fails — an exotic or
    /// network filesystem, a sandbox that refuses `mmap`, exhausted address
    /// space on a 32-bit host — so `open` is never *less* capable than the P9.2
    /// reader it replaces. (An empty file is **not** such a case: memmap2
    /// returns a valid empty mapping for it, which then fails the header parse
    /// like any other truncated pack.)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: &Path) -> Result<Self> {
        // SAFETY: `Mmap::map` is unsafe because the mapped bytes are only sound
        // for as long as nobody else mutates or truncates the file underneath
        // us. That contract is upheld on our side by [`PackWriter::write_to_file`],
        // which never writes a pack in place: it writes a sibling temp file and
        // `rename`s over the target, so a re-cook unlinks the old name while the
        // inode an existing mapping refers to stays alive and unchanged. A
        // `.inf_pack` is otherwise an immutable ship artifact. (The verify-once
        // hash catches corrupt *content*; it cannot catch a file another program
        // truncates under a live mapping — that is a SIGBUS on Unix, and no
        // mmap-based asset runtime defends against it.)
        Self::open_with(path, |file| unsafe { memmap2::Mmap::map(file) })
    }

    /// [`open`](Self::open) with an injectable mapper — the seam that lets tests
    /// exercise the fallback path (a real mapping failure is not something a
    /// test can provoke portably).
    #[cfg(not(target_arch = "wasm32"))]
    fn open_with(
        path: &Path,
        map: impl FnOnce(&std::fs::File) -> std::io::Result<memmap2::Mmap>,
    ) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        match map(&file) {
            Ok(mapped) => Self::from_backing(PackBacking::Mmap(mapped)),
            Err(err) => {
                tracing::debug!(
                    path = %path.display(),
                    error = %err,
                    "mmap failed; falling back to a whole-file pack read"
                );
                Self::from_backing(PackBacking::Owned(AlignedBytes::copy_from(&std::fs::read(
                    path,
                )?)))
            }
        }
    }

    /// Open a pack from a file (wasm: whole-file read — no mmap in a browser).
    #[cfg(target_arch = "wasm32")]
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Parse a pack already in memory (the wasm player's fetched bytes, tests).
    ///
    /// Costs one memcpy: the bytes are re-homed into a 16-byte-aligned buffer so
    /// borrowed blob slices keep the same address-alignment guarantee the mmap
    /// backing gives for free. `data` is dropped straight after, so the doubled
    /// footprint is transient.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        Self::from_backing(PackBacking::Owned(AlignedBytes::copy_from(&data)))
    }

    /// Parse the header + index over whichever backing the bytes live in.
    fn from_backing(backing: PackBacking) -> Result<Self> {
        let data = backing.as_slice();
        if data.len() < HEADER_LEN as usize {
            return Err(AssetError::Pack("pack shorter than header".into()));
        }
        if data[0..8] != PACK_MAGIC {
            return Err(AssetError::Pack("bad magic (not an .inf_pack)".into()));
        }
        let format_version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if format_version > PACK_FORMAT_VERSION {
            return Err(AssetError::Pack(format!(
                "pack format v{format_version} is newer than this build (v{PACK_FORMAT_VERSION})"
            )));
        }
        if format_version < PACK_MIN_READ_VERSION {
            return Err(AssetError::Pack(format!(
                "pack format v{format_version} is older than this build reads (v{PACK_MIN_READ_VERSION})"
            )));
        }
        let count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as u64;
        let index_end = HEADER_LEN + ENTRY_LEN * count;
        if (data.len() as u64) < index_end {
            return Err(AssetError::Pack("pack truncated in index".into()));
        }

        let mut entries = Vec::with_capacity(count as usize);
        let mut by_guid = BTreeMap::new();
        for i in 0..count {
            let base = (HEADER_LEN + ENTRY_LEN * i) as usize;
            let e = &data[base..base + ENTRY_LEN as usize];
            let guid = AssetId(Uuid::from_bytes(e[0..16].try_into().unwrap()));
            let kind = kind_from_code(u16::from_le_bytes(e[16..18].try_into().unwrap()));
            let compressed = u16::from_le_bytes(e[18..20].try_into().unwrap()) & 1 == 1;
            let content_hash = ContentHash(u128::from_le_bytes(e[20..36].try_into().unwrap()));
            let offset = u64::from_le_bytes(e[36..44].try_into().unwrap());
            let stored_len = u64::from_le_bytes(e[44..52].try_into().unwrap());
            let uncompressed_len = u64::from_le_bytes(e[52..60].try_into().unwrap());
            // Bounds-check the blob so `read_ref` can slice without re-validating.
            let end = offset
                .checked_add(stored_len)
                .ok_or_else(|| AssetError::Pack("blob length overflow".into()))?;
            if offset < index_end || end > data.len() as u64 {
                return Err(AssetError::Pack(format!("blob for {guid} out of bounds")));
            }
            // v2 promises 16-byte blob alignment; enforce it so a consumer may
            // rely on a borrowed slice's alignment purely from the version.
            // (v1 packs are unaligned by construction and stay readable.)
            if format_version >= 2 && offset % BLOB_ALIGN != 0 {
                return Err(AssetError::Pack(format!(
                    "blob for {guid} at {offset} is not {BLOB_ALIGN}-byte aligned (v2 pack)"
                )));
            }
            by_guid.insert(guid, entries.len());
            entries.push(PackEntry {
                guid,
                kind,
                content_hash,
                offset,
                stored_len,
                uncompressed_len,
                compressed,
            });
        }
        let verified = entries.iter().map(|_| AtomicBool::new(false)).collect();
        Ok(Self {
            data: backing,
            format_version,
            entries,
            by_guid,
            verified,
            verify_calls: AtomicU64::new(0),
        })
    }

    /// The container format version this pack was written with.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// True if the pack is memory-mapped rather than buffered in memory.
    pub fn is_mmapped(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            matches!(self.data, PackBacking::Mmap(_))
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    /// Number of assets in the pack.
    pub fn len(&self) -> usize {
        self.by_guid.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_guid.is_empty()
    }

    /// True if `guid` is present.
    pub fn contains(&self, guid: AssetId) -> bool {
        self.by_guid.contains_key(&guid)
    }

    /// The index entry for `guid`, if present.
    pub fn entry(&self, guid: AssetId) -> Option<&PackEntry> {
        self.by_guid.get(&guid).map(|&slot| &self.entries[slot])
    }

    /// Iterate the index in GUID order.
    pub fn index(&self) -> impl Iterator<Item = &PackEntry> {
        self.by_guid.values().map(|&slot| &self.entries[slot])
    }

    /// How many integrity hashes this reader has computed since it was opened.
    ///
    /// The observable half of verify-once: reading the same entry N times leaves
    /// this at 1 (it only ever rises on a *first* verification of an entry, or on
    /// a benign race between threads). Useful for tests and cook/CLI diagnostics.
    pub fn verify_count(&self) -> u64 {
        self.verify_calls.load(Ordering::Relaxed)
    }

    /// Read a payload by GUID **without copying when possible** (P16.1).
    ///
    /// An uncompressed entry comes back as `Cow::Borrowed` — a slice straight
    /// into the mmap/buffer, valid for as long as the reader — while a
    /// compressed one is decoded into a `Cow::Owned`. The first read of an entry
    /// integrity-checks it (see the module docs); later reads skip the rehash.
    pub fn read_ref(&self, guid: AssetId) -> Result<Cow<'_, [u8]>> {
        let &slot = self
            .by_guid
            .get(&guid)
            .ok_or(AssetError::UnknownAsset(guid))?;
        let e = &self.entries[slot];
        let start = e.offset as usize;
        let stored = &self.data.as_slice()[start..start + e.stored_len as usize];
        let payload: Cow<'_, [u8]> = if e.compressed {
            Cow::Owned(zstd_decode(stored).map_err(|err| {
                AssetError::Pack(format!("zstd decode {guid}: {err} (corrupt pack?)"))
            })?)
        } else {
            Cow::Borrowed(stored)
        };
        // Integrity, once per entry: the payload must hash to what the index
        // promised. `Relaxed` is enough — the mark guards no other memory (the
        // backing bytes are immutable), and a lost race only rehashes.
        if !self.verified[slot].load(Ordering::Relaxed) {
            self.verify_calls.fetch_add(1, Ordering::Relaxed);
            if ContentHash::of(&payload) != e.content_hash {
                return Err(AssetError::Pack(format!(
                    "content hash mismatch for {guid} (corrupt pack)"
                )));
            }
            // **`uncompressed_len` was parsed, published, and never verified**
            // (L5.F12). The content hash catches wrong *bytes*, so this is a
            // truth-of-metadata question rather than a corruption one — but the
            // field is `pub`, it reads as authoritative, and it costs one
            // comparison inside a branch that is already hashing the payload.
            // A header that says something the file does not is a header nobody
            // can plan against.
            if payload.len() as u64 != e.uncompressed_len {
                return Err(AssetError::Pack(format!(
                    "{guid} declares {} uncompressed bytes and yields {} (corrupt pack)",
                    e.uncompressed_len,
                    payload.len()
                )));
            }
            self.verified[slot].store(true, Ordering::Relaxed);
        }
        Ok(payload)
    }

    /// Read, decompress, and integrity-verify a payload by GUID, as an owned
    /// `Vec`.
    ///
    /// The compatibility face of [`read_ref`](Self::read_ref) — identical bytes,
    /// one copy for an uncompressed entry. Prefer `read_ref` on hot streaming
    /// paths.
    pub fn read(&self, guid: AssetId) -> Result<Vec<u8>> {
        Ok(self.read_ref(guid)?.into_owned())
    }
}

impl std::fmt::Debug for PackReader {
    /// Summarizes the pack; never dumps the (possibly gigabyte) backing bytes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackReader")
            .field("format_version", &self.format_version)
            .field("entries", &self.entries.len())
            .field("bytes", &self.data.as_slice().len())
            .field("mmapped", &self.is_mmapped())
            .field("verify_calls", &self.verify_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guid(n: u128) -> AssetId {
        AssetId(Uuid::from_u128(n))
    }

    /// A payload large + repetitive enough to actually compress.
    fn big(seed: u8) -> Vec<u8> {
        (0..4096u32).map(|i| (i as u8).wrapping_add(seed)).collect()
    }

    #[test]
    fn round_trips_payloads_and_index() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(2), AssetKind::Level, b"level-bytes")
            .unwrap();
        w.add_bytes(guid(1), AssetKind::Blueprint, &big(7)).unwrap();
        w.add_bytes(guid(3), AssetKind::Texture, b"").unwrap(); // empty payload
        let bytes = w.to_bytes().unwrap();

        let r = PackReader::from_bytes(bytes).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r.format_version(), PACK_FORMAT_VERSION);
        assert!(r.contains(guid(1)));
        assert_eq!(r.read(guid(2)).unwrap(), b"level-bytes");
        assert_eq!(r.read(guid(1)).unwrap(), big(7));
        assert_eq!(r.read(guid(3)).unwrap(), b"");
        // Index is GUID-sorted.
        let ids: Vec<_> = r.index().map(|e| e.guid).collect();
        assert_eq!(ids, vec![guid(1), guid(2), guid(3)]);
        // The big blueprint blob actually compressed; the tiny/empty ones didn't.
        assert!(r.entry(guid(1)).unwrap().compressed);
        assert!(!r.entry(guid(2)).unwrap().compressed);
        assert_eq!(r.entry(guid(1)).unwrap().kind, AssetKind::Blueprint);
        assert_eq!(r.entry(guid(1)).unwrap().uncompressed_len, 4096);
    }

    /// **L5.F12 — `uncompressed_len` was parsed, published as a `pub` field,
    /// and never verified.**
    ///
    /// The content hash catches wrong *bytes*, so this is a truth-of-metadata
    /// question rather than a corruption one — but the field reads as
    /// authoritative and it is the number a consumer would plan an allocation
    /// against. The check costs one comparison inside the branch that is already
    /// hashing.
    ///
    /// Un-fix mutation: delete the comparison in `read_ref`'s verify block and
    /// the doctored read below succeeds.
    #[test]
    fn a_pack_entry_that_lies_about_its_uncompressed_length_is_refused() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Level, b"eleven-byte")
            .unwrap();
        let good = w.to_bytes().unwrap();
        // The control: the honest pack reads.
        let r = PackReader::from_bytes(good.clone()).unwrap();
        assert_eq!(r.read(guid(1)).unwrap(), b"eleven-byte");
        assert_eq!(r.entry(guid(1)).unwrap().uncompressed_len, 11);

        // Doctor entry 0's `uncompressed_len` (bytes 52..60 of the entry)
        // without touching the blob or its hash: the payload still hashes
        // correctly, so only this check can catch it.
        let at = HEADER_LEN as usize + 52;
        assert_eq!(
            u64::from_le_bytes(good[at..at + 8].try_into().unwrap()),
            11,
            "the fixture must be pointing at the length field"
        );
        let mut lying = good;
        lying[at..at + 8].copy_from_slice(&99u64.to_le_bytes());
        let r = PackReader::from_bytes(lying).expect("the header still parses");
        assert_eq!(r.entry(guid(1)).unwrap().uncompressed_len, 99);
        let e = r.read(guid(1)).unwrap_err().to_string();
        assert!(
            e.contains("declares 99 uncompressed bytes and yields 11"),
            "{e}"
        );
    }

    #[test]
    fn build_is_byte_identical_regardless_of_insertion_order() {
        let mut a = PackWriter::new();
        a.add_bytes(guid(10), AssetKind::Mesh, &big(1)).unwrap();
        a.add_bytes(guid(20), AssetKind::Texture, &big(2)).unwrap();
        a.add_bytes(guid(30), AssetKind::Material, b"m").unwrap();

        let mut b = PackWriter::new();
        // Reverse insertion order → must still produce identical bytes.
        b.add_bytes(guid(30), AssetKind::Material, b"m").unwrap();
        b.add_bytes(guid(20), AssetKind::Texture, &big(2)).unwrap();
        b.add_bytes(guid(10), AssetKind::Mesh, &big(1)).unwrap();

        assert_eq!(a.to_bytes().unwrap(), b.to_bytes().unwrap());
    }

    #[test]
    fn hash_verification_catches_a_flipped_byte() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Level, &big(3)).unwrap();
        let mut bytes = w.to_bytes().unwrap();
        // Flip a byte in the last position (inside the blob section).
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let r = PackReader::from_bytes(bytes).unwrap();
        // Either zstd fails to decode or the hash mismatches — both are errors.
        assert!(r.read(guid(1)).is_err());
        // Verify-once must never launder corruption: the mark is only set on a
        // successful check, so every later read fails the same way.
        assert!(r.read(guid(1)).is_err());
        assert!(r.read_ref(guid(1)).is_err());
    }

    #[test]
    fn hash_verification_catches_a_flipped_byte_in_an_uncompressed_entry() {
        // The zero-copy path has no zstd frame to fail first — the xxh3 check is
        // the only thing standing between a bit rot and a borrowed slice.
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::MeshletMesh, &big(3))
            .unwrap();
        let r0 = PackReader::from_bytes(w.to_bytes().unwrap()).unwrap();
        assert!(!r0.entry(guid(1)).unwrap().compressed, "vmesh stays raw");

        let mut bytes = w.to_bytes().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let r = PackReader::from_bytes(bytes).unwrap();
        assert!(r.read_ref(guid(1)).is_err());
        assert!(r.read(guid(1)).is_err());
    }

    #[test]
    fn missing_guid_is_an_error() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Level, b"x").unwrap();
        let r = PackReader::from_bytes(w.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            r.read(guid(999)),
            Err(AssetError::UnknownAsset(_))
        ));
        assert!(!r.contains(guid(999)));
    }

    #[test]
    fn empty_pack_round_trips() {
        let w = PackWriter::new();
        assert!(w.is_empty());
        let bytes = w.to_bytes().unwrap();
        let r = PackReader::from_bytes(bytes).unwrap();
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
        assert_eq!(r.index().count(), 0);
    }

    #[test]
    fn duplicate_guid_rejected() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Level, b"a").unwrap();
        assert!(w.add_bytes(guid(1), AssetKind::Level, b"b").is_err());
    }

    #[test]
    fn rejects_bad_magic_and_newer_format() {
        assert!(PackReader::from_bytes(vec![0u8; 32]).is_err());
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Level, b"x").unwrap();
        let mut bytes = w.to_bytes().unwrap();
        // Bump the format version past what we understand.
        bytes[8..12].copy_from_slice(&(PACK_FORMAT_VERSION + 1).to_le_bytes());
        assert!(PackReader::from_bytes(bytes).is_err());
    }

    #[test]
    fn all_kinds_round_trip_through_codes() {
        for &k in AssetKind::all() {
            assert_eq!(kind_from_code(kind_code(k)), k, "{k:?}");
        }
        // Codes are a permanent on-disk contract — every already-shipped pack
        // decodes its index through them, so they may only ever be APPENDED to.
        // Pinning them here turns an accidental renumber into a test failure
        // rather than a silently mis-typed asset in an old build.
        assert_eq!(kind_code(AssetKind::Unknown), 0);
        assert_eq!(kind_code(AssetKind::Level), 1);
        assert_eq!(kind_code(AssetKind::MeshletMesh), 16);
        assert_eq!(kind_code(AssetKind::Terrain), 17, "P16.3 appended 17");
        assert_eq!(kind_code(AssetKind::Partition), 18, "P16.5 appended 18");
        assert_eq!(kind_code(AssetKind::BiomeSet), 19, "P19.2 appended 19");
        assert_eq!(kind_code(AssetKind::VoxelVolume), 20, "P21.1 appended 20");
        // …and an unknown future code degrades to `Unknown` rather than erroring,
        // so a newer pack's extra kinds never break an older reader's index parse.
        assert_eq!(kind_from_code(9999), AssetKind::Unknown);
    }

    // ── P16.1: alignment, zero-copy reads, verify-once, mmap ────────────────

    /// Every blob in a v2 pack starts on a 16-byte boundary — the promise a
    /// borrowed slice's alignment rests on.
    #[test]
    fn v2_blobs_are_16_byte_aligned() {
        let mut w = PackWriter::new();
        // Sizes chosen so unpadded offsets would land off-boundary: 11, 4096
        // (compressed to an odd length), 0, and a 3-byte tail.
        w.add_bytes(guid(1), AssetKind::Level, b"level-bytes")
            .unwrap();
        w.add_bytes(guid(2), AssetKind::Blueprint, &big(7)).unwrap();
        w.add_bytes(guid(3), AssetKind::Texture, b"").unwrap();
        w.add_bytes(guid(4), AssetKind::Mesh, b"abc").unwrap();
        w.add_bytes(guid(5), AssetKind::MeshletMesh, &big(9))
            .unwrap();
        let r = PackReader::from_bytes(w.to_bytes().unwrap()).unwrap();

        assert_eq!(r.format_version(), 2);
        for e in r.index() {
            assert_eq!(
                e.offset % BLOB_ALIGN,
                0,
                "blob {} at {} is unaligned",
                e.guid,
                e.offset
            );
        }
        // The first blob sits past the index, padded up (16 + 5*60 = 316 → 320).
        assert_eq!(r.entry(guid(1)).unwrap().offset, 320);
        // Payloads survive the padding.
        assert_eq!(r.read(guid(1)).unwrap(), b"level-bytes");
        assert_eq!(r.read(guid(2)).unwrap(), big(7));
        assert_eq!(r.read(guid(3)).unwrap(), b"");
        assert_eq!(r.read(guid(4)).unwrap(), b"abc");
        assert_eq!(r.read(guid(5)).unwrap(), big(9));
    }

    /// Padding is deterministic zeros, so the P9.2 byte-identity gate survives
    /// the v2 layout.
    #[test]
    fn v2_build_is_byte_identical_across_rebuilds() {
        let build = || {
            let mut w = PackWriter::new();
            w.add_bytes(guid(30), AssetKind::Material, b"m").unwrap();
            w.add_bytes(guid(10), AssetKind::Mesh, &big(1)).unwrap();
            w.add_bytes(guid(40), AssetKind::MeshletMesh, &big(4))
                .unwrap();
            w.add_bytes(guid(20), AssetKind::Texture, b"tex").unwrap();
            w.add_bytes(guid(50), AssetKind::Level, b"lvl").unwrap();
            w.to_bytes().unwrap()
        };
        let a = build();
        let b = build();
        assert_eq!(a, b, "two builds of the same inputs → byte-identical pack");
        // And the padding really is zeros (no allocator garbage leaking in):
        // 16 + 5*60 = 316 of header+index, so 4 bytes of pad precede blob 0.
        let first = PackReader::from_bytes(a.clone())
            .unwrap()
            .index()
            .next()
            .unwrap()
            .offset as usize;
        let index_end = (HEADER_LEN + ENTRY_LEN * 5) as usize;
        assert!(first > index_end, "there is padding to inspect");
        assert!(a[index_end..first].iter().all(|&b| b == 0));
    }

    /// `read_ref` returns exactly what `read` does, borrowing where it can.
    #[test]
    fn read_ref_matches_read_and_borrows_uncompressed() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Blueprint, &big(7)).unwrap(); // compressed
        w.add_bytes(guid(2), AssetKind::Level, b"level-bytes")
            .unwrap(); // raw
        w.add_bytes(guid(3), AssetKind::Texture, b"").unwrap(); // raw, empty
        w.add_bytes(guid(4), AssetKind::MeshletMesh, &big(9))
            .unwrap(); // raw by policy
        let r = PackReader::from_bytes(w.to_bytes().unwrap()).unwrap();

        for id in [guid(1), guid(2), guid(3), guid(4)] {
            assert_eq!(
                r.read_ref(id).unwrap().as_ref(),
                r.read(id).unwrap().as_slice(),
                "{id}"
            );
        }
        // Uncompressed entries borrow straight out of the backing; the
        // compressed one must own its decoded bytes.
        assert!(matches!(r.read_ref(guid(2)).unwrap(), Cow::Borrowed(_)));
        assert!(matches!(r.read_ref(guid(3)).unwrap(), Cow::Borrowed(_)));
        assert!(matches!(r.read_ref(guid(4)).unwrap(), Cow::Borrowed(_)));
        assert!(matches!(r.read_ref(guid(1)).unwrap(), Cow::Owned(_)));

        // A borrowed slice really points into the pack bytes, not a copy.
        let e = r.entry(guid(2)).unwrap();
        let slice = match r.read_ref(guid(2)).unwrap() {
            Cow::Borrowed(s) => s.as_ptr() as usize,
            Cow::Owned(_) => unreachable!(),
        };
        let base = r.data.as_slice().as_ptr() as usize;
        assert_eq!(slice - base, e.offset as usize);
    }

    /// The xxh3 check runs once per entry per reader, not once per read.
    #[test]
    fn integrity_is_verified_once_per_entry() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Blueprint, &big(7)).unwrap();
        w.add_bytes(guid(2), AssetKind::MeshletMesh, &big(9))
            .unwrap();
        let r = PackReader::from_bytes(w.to_bytes().unwrap()).unwrap();

        assert_eq!(r.verify_count(), 0, "nothing hashed before the first read");
        for _ in 0..8 {
            r.read_ref(guid(1)).unwrap();
            r.read(guid(1)).unwrap();
        }
        assert_eq!(r.verify_count(), 1, "one entry, one hash");
        assert!(r.verified[r.by_guid[&guid(1)]].load(Ordering::Relaxed));
        assert!(!r.verified[r.by_guid[&guid(2)]].load(Ordering::Relaxed));

        for _ in 0..8 {
            r.read(guid(2)).unwrap();
        }
        assert_eq!(r.verify_count(), 2, "second entry adds exactly one hash");

        // A fresh reader over the same bytes starts clean (marks are per-reader).
        let r2 = PackReader::from_bytes(w.to_bytes().unwrap()).unwrap();
        assert_eq!(r2.verify_count(), 0);
        r2.read(guid(1)).unwrap();
        assert_eq!(r2.verify_count(), 1);
    }

    /// Streaming-class kinds cook uncompressed; everything else is unchanged.
    #[test]
    fn streaming_class_kinds_are_stored_uncompressed() {
        /// The full streaming-class list; a new kind joins it only deliberately.
        const STREAMING: &[AssetKind] = &[
            AssetKind::MeshletMesh,
            AssetKind::Terrain,
            AssetKind::Partition,
            // P21.1: a `.inf_voxel` is paged chunk-at-a-time out of the mapping,
            // exactly like a `.inf_terrain` tile.
            AssetKind::VoxelVolume,
            // P26.1: a `.inf_tex` is a tiled container paged one 136² tile at a
            // time by streaming virtual texturing.
            AssetKind::Texture,
        ];
        for &k in STREAMING {
            assert!(!PackWriter::compresses_kind(k), "{k:?} must stay raw");
        }
        for &k in AssetKind::all() {
            if !STREAMING.contains(&k) {
                assert!(PackWriter::compresses_kind(k), "{k:?}");
            }
        }
        // P22.2, called out because it is the counter-example: `Fracture` is a
        // DERIVED kind like `MeshletMesh` and `Partition`, and it still
        // compresses. Derived-ness is not what makes a kind streaming-class —
        // being paged one unit at a time is, and a chunk set is read whole or
        // not at all.
        assert!(
            PackWriter::compresses_kind(AssetKind::Fracture),
            "a .inf_fracture is loaded whole; there is nothing to sub-slice"
        );

        let payload = big(5); // compressible — a Mesh would shrink
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::MeshletMesh, &payload)
            .unwrap();
        w.add_bytes(guid(2), AssetKind::Mesh, &payload).unwrap();
        w.add_bytes(guid(3), AssetKind::Terrain, &payload).unwrap();
        w.add_bytes(guid(4), AssetKind::VoxelVolume, &payload)
            .unwrap();
        let r = PackReader::from_bytes(w.to_bytes().unwrap()).unwrap();

        for g in [guid(1), guid(3), guid(4)] {
            let e = r.entry(g).unwrap();
            assert!(!e.compressed, "{:?} blobs must stay mmap-readable", e.kind);
            assert_eq!(e.stored_len, payload.len() as u64);
            assert_eq!(e.stored_len, e.uncompressed_len);
            assert_eq!(r.read(g).unwrap(), payload);
            // Borrowed, not decoded — the zero-copy streaming promise.
            assert!(matches!(r.read_ref(g).unwrap(), Cow::Borrowed(_)));
        }
        assert!(r.entry(guid(2)).unwrap().compressed, "meshes still zstd");
    }

    /// `open` maps the file and reads through the mapping.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn open_memory_maps_the_pack() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/ship.inf_pack");
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Blueprint, &big(7)).unwrap();
        w.add_bytes(guid(2), AssetKind::MeshletMesh, &big(9))
            .unwrap();
        w.write_to_file(&path).unwrap();

        let r = PackReader::open(&path).unwrap();
        assert!(r.is_mmapped(), "a real pack file must map");
        assert_eq!(r.format_version(), 2);
        assert_eq!(r.read(guid(1)).unwrap(), big(7));
        assert!(matches!(r.read_ref(guid(2)).unwrap(), Cow::Borrowed(_)));
        assert_eq!(r.read_ref(guid(2)).unwrap().as_ref(), big(9).as_slice());
        assert_eq!(r.verify_count(), 2);
    }

    /// An empty file maps fine (memmap2 hands back an empty mapping) and then
    /// fails the header parse like any other truncated pack — the caller sees a
    /// pack error, never a raw mmap errno.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn an_empty_pack_file_is_a_parse_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.inf_pack");
        std::fs::write(&path, []).unwrap();
        let err = PackReader::open(&path).unwrap_err();
        assert!(
            matches!(&err, AssetError::Pack(m) if m.contains("shorter than header")),
            "{err}"
        );
    }

    /// When the mapping itself fails, `open` degrades to a whole-file read and
    /// the pack is still fully readable. A real mapping failure can't be
    /// provoked portably, so the mapper is injected.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn open_falls_back_to_a_whole_file_read_when_mapping_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ship.inf_pack");
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Blueprint, &big(7)).unwrap();
        w.add_bytes(guid(2), AssetKind::MeshletMesh, &big(9))
            .unwrap();
        w.write_to_file(&path).unwrap();

        let r = PackReader::open_with(&path, |_| {
            Err(std::io::Error::other("injected mmap failure"))
        })
        .expect("fallback path still opens the pack");

        assert!(!r.is_mmapped(), "must be the owned backing");
        assert_eq!(r.read(guid(1)).unwrap(), big(7));
        assert_eq!(r.read_ref(guid(2)).unwrap().as_ref(), big(9).as_slice());
        // The fallback keeps the zero-copy + alignment contract intact.
        assert!(matches!(r.read_ref(guid(2)).unwrap(), Cow::Borrowed(_)));
        assert_eq!(r.data.as_slice().as_ptr() as usize % BLOB_ALIGN as usize, 0);
    }

    /// Both backings put their base on a 16-byte boundary, so v2's aligned
    /// offsets become aligned *addresses* — the promise P18.2 will cast on.
    #[test]
    fn borrowed_blobs_are_16_byte_aligned_in_memory() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Level, b"level-bytes")
            .unwrap();
        w.add_bytes(guid(2), AssetKind::MeshletMesh, &big(9))
            .unwrap();
        w.add_bytes(guid(3), AssetKind::Mesh, b"abc").unwrap();

        let r = PackReader::from_bytes(w.to_bytes().unwrap()).unwrap();
        assert_eq!(
            r.data.as_slice().as_ptr() as usize % BLOB_ALIGN as usize,
            0,
            "owned backing base must be aligned"
        );
        for e in r.index() {
            if let Cow::Borrowed(slice) = r.read_ref(e.guid).unwrap() {
                assert_eq!(
                    slice.as_ptr() as usize % BLOB_ALIGN as usize,
                    0,
                    "blob {} borrowed at an unaligned address",
                    e.guid
                );
            }
        }
    }

    /// The alignment carrier holds for odd and empty lengths too.
    #[test]
    fn aligned_bytes_are_aligned_at_every_length() {
        for len in [0usize, 1, 15, 16, 17, 4095, 4096] {
            let src: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let a = AlignedBytes::copy_from(&src);
            assert_eq!(
                a.as_slice().as_ptr() as usize % BLOB_ALIGN as usize,
                0,
                "len {len}"
            );
            assert_eq!(a.as_slice(), &src[..], "len {len}");
        }
    }

    /// Re-cooking over a path someone is already reading must never corrupt the
    /// live reader. `write_to_file` renames a sibling temp over the target, so
    /// the old inode survives for existing mappings.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn rewriting_a_pack_under_a_live_reader_is_safe() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ship.inf_pack");

        let mut first = PackWriter::new();
        first
            .add_bytes(guid(1), AssetKind::MeshletMesh, &big(1))
            .unwrap();
        first.write_to_file(&path).unwrap();

        let live = PackReader::open(&path).unwrap();
        assert_eq!(live.read(guid(1)).unwrap(), big(1));

        // Re-cook different content over the same path while the map is live.
        let mut second = PackWriter::new();
        second
            .add_bytes(guid(1), AssetKind::MeshletMesh, &big(2))
            .unwrap();
        let rewrite = second.write_to_file(&path);

        // Whatever happened, the live reader still sees the bytes it verified.
        assert_eq!(
            live.read_ref(guid(1)).unwrap().as_ref(),
            big(1).as_slice(),
            "a live mapping must not observe a re-cook"
        );

        // And the file on disk is a valid pack either way. The rename is
        // expected to land on Unix *and* on modern Windows (std asks for POSIX
        // rename semantics); if a platform ever refuses the replace instead,
        // that must leave the original intact rather than truncated.
        let expected = if rewrite.is_ok() { big(2) } else { big(1) };
        if cfg!(unix) {
            rewrite.expect("rename over a mapped file succeeds on unix");
        }
        let reopened = PackReader::open(&path).unwrap();
        assert_eq!(reopened.read(guid(1)).unwrap(), expected);

        // No temp litter left beside the artifact.
        let strays: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(strays.is_empty(), "left temp files behind: {strays:?}");
    }

    /// A hand-made v2 pack whose blobs are NOT aligned is rejected: the version
    /// is a promise consumers get to rely on.
    #[test]
    fn v2_rejects_unaligned_blobs() {
        let mut w = PackWriter::new();
        w.add_bytes(guid(1), AssetKind::Level, b"level-bytes")
            .unwrap();
        let mut bytes = w.to_bytes().unwrap();
        // Entry 0's offset field lives at header + 36.
        let off = (HEADER_LEN + 36) as usize;
        let orig = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        bytes[off..off + 8].copy_from_slice(&(orig - 4).to_le_bytes());
        let err = PackReader::from_bytes(bytes).unwrap_err();
        assert!(
            matches!(&err, AssetError::Pack(m) if m.contains("aligned")),
            "{err}"
        );
    }

    /// The frozen v1 pack (written by the P9.2 code, committed byte-for-byte)
    /// must load forever: unaligned blobs, and a MeshletMesh that was still
    /// zstd-compressed under the old policy.
    ///
    /// **Provenance:** produced by the pre-v2 writer — i.e. this file's own
    /// `PackWriter` as of the commit before P16.1 — fed exactly the five entries
    /// asserted below (`big(7)`, `b"level-bytes"`, `b""`, `big(9)`, `big(11)`)
    /// through `write_to_file`. To regenerate it, check out that parent commit,
    /// run the same five `add_bytes` calls, and commit the bytes; it must never
    /// be re-blessed from the current writer, which emits v2.
    #[test]
    fn loads_the_frozen_v1_pack_fixture() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pack_v1.inf_pack");
        let bytes = std::fs::read(&path).expect("committed v1 fixture present");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            1,
            "fixture is a genuine v1 pack"
        );

        // Both doors — the mmap `open` and the owned `from_bytes` — read it.
        for r in [
            PackReader::open(&path).unwrap(),
            PackReader::from_bytes(bytes).unwrap(),
        ] {
            assert_eq!(r.format_version(), 1);
            assert_eq!(r.len(), 5);
            // v1 packed blobs back-to-back straight after the index: 16 + 5*60.
            assert_eq!(r.entry(guid(1)).unwrap().offset, 316);
            assert_ne!(r.entry(guid(1)).unwrap().offset % BLOB_ALIGN, 0);

            assert_eq!(r.read(guid(1)).unwrap(), big(7));
            assert_eq!(r.read(guid(2)).unwrap(), b"level-bytes");
            assert_eq!(r.read(guid(3)).unwrap(), b"");
            assert_eq!(r.read(guid(4)).unwrap(), big(9));
            assert_eq!(r.read(guid(5)).unwrap(), big(11));
            // read_ref works over v1 too (borrowing the unaligned raw entries).
            assert_eq!(r.read_ref(guid(2)).unwrap().as_ref(), b"level-bytes");
            assert!(matches!(r.read_ref(guid(2)).unwrap(), Cow::Borrowed(_)));

            // The old compression policy is preserved on read: this vmesh blob
            // is compressed, which today's writer would store raw.
            let vmesh = r.entry(guid(4)).unwrap();
            assert_eq!(vmesh.kind, AssetKind::MeshletMesh);
            assert!(vmesh.compressed, "v1 compressed vmeshes stay readable");
            assert!(!PackWriter::compresses_kind(AssetKind::MeshletMesh));
        }
    }
}
