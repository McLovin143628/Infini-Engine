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
//! ├ blob section (concatenated payloads at their offsets) ────────────┤
//! │  … blob 0 … blob 1 …                                               │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Determinism (the P9.2 gate)
//!
//! The same inputs always produce a **byte-identical** pack: the index is sorted
//! by GUID, blob offsets are assigned in that order, compression uses a fixed
//! level with no timestamp/dictionary, and nothing wall-clock enters the bytes.
//!
//! # Integrity
//!
//! Every [`PackReader::read`] recomputes the xxh3-128 of the decompressed payload
//! and compares it to the stored `content_hash`; a flipped byte (in the blob or a
//! corrupted zstd frame) fails to decode or mismatches and returns an error.
//!
//! # Schema-version discipline (ROADMAP §3)
//!
//! `format_ver` gates the whole container. A reader rejects a pack whose
//! `format_ver` is newer than [`PACK_FORMAT_VERSION`]; future revisions add a
//! decode arm keyed on the version rather than reinterpreting the bytes.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use uuid::Uuid;

use crate::error::{AssetError, Result};
use crate::hash::ContentHash;
use crate::id::AssetId;
use crate::kind::AssetKind;

/// Magic bytes at the start of every `.inf_pack`.
pub const PACK_MAGIC: [u8; 8] = *b"INFPACK\0";

/// The current container format version (bump on a breaking layout change).
pub const PACK_FORMAT_VERSION: u32 = 1;

/// Bytes of the fixed header.
const HEADER_LEN: u64 = 16;

/// Bytes of one index entry.
const ENTRY_LEN: u64 = 60;

/// zstd compression level used for cooked blobs. Cook is an offline step, so a
/// high level trades build time for a smaller ship. Determinism is independent
/// of the level; this is a pure size/speed knob (tunable later).
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

    /// Add an asset from raw payload bytes. Compresses eagerly; a duplicate GUID
    /// is an error (the cook resolves ids before packing).
    pub fn add_bytes(&mut self, guid: AssetId, kind: AssetKind, payload: &[u8]) -> Result<()> {
        if self.items.contains_key(&guid) {
            return Err(AssetError::Pack(format!("duplicate guid {guid} in pack")));
        }
        let content_hash = ContentHash::of(payload);
        let (stored, compressed) = maybe_compress(payload)?;
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

        // Blobs start after the header + full index.
        let mut offset = HEADER_LEN + ENTRY_LEN * count as u64;

        // Index (already GUID-sorted by the BTreeMap).
        for (guid, item) in &self.items {
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
            offset += item.stored.len() as u64;
        }

        // Blob section, in the same GUID order.
        for item in self.items.values() {
            w.write_all(&item.stored)?;
        }
        Ok(())
    }

    /// Write the pack to a file (creating parent directories).
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut buf = Vec::new();
        self.write(&mut buf)?;
        std::fs::write(path, &buf)?;
        Ok(())
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
    let packed = zstd::encode_all(payload, ZSTD_LEVEL)
        .map_err(|e| AssetError::Pack(format!("zstd: {e}")))?;
    if packed.len() < payload.len() {
        Ok((packed, true))
    } else {
        // Incompressible: store raw so we never inflate.
        Ok((payload.to_vec(), false))
    }
}

/// A read-only view over a `.inf_pack` (whole-file buffered).
///
/// `open`/`from_bytes` parse the header + index once; [`read`](Self::read)
/// fetches and verifies a single payload on demand. (Whole-file buffering keeps
/// the reader dependency-free; an `mmap` backing for very large ship packs is a
/// documented follow-up.)
pub struct PackReader {
    data: Vec<u8>,
    format_version: u32,
    index: BTreeMap<AssetId, PackEntry>,
}

impl PackReader {
    /// Open a pack from a file.
    pub fn open(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_bytes(data)
    }

    /// Parse a pack already in memory.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
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
        let count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as u64;
        let index_end = HEADER_LEN + ENTRY_LEN * count;
        if (data.len() as u64) < index_end {
            return Err(AssetError::Pack("pack truncated in index".into()));
        }

        let mut index = BTreeMap::new();
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
            // Bounds-check the blob so `read` can slice without re-validating.
            let end = offset
                .checked_add(stored_len)
                .ok_or_else(|| AssetError::Pack("blob length overflow".into()))?;
            if offset < index_end || end > data.len() as u64 {
                return Err(AssetError::Pack(format!("blob for {guid} out of bounds")));
            }
            index.insert(
                guid,
                PackEntry {
                    guid,
                    kind,
                    content_hash,
                    offset,
                    stored_len,
                    uncompressed_len,
                    compressed,
                },
            );
        }
        Ok(Self {
            data,
            format_version,
            index,
        })
    }

    /// The container format version this pack was written with.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Number of assets in the pack.
    pub fn len(&self) -> usize {
        self.index.len()
    }
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// True if `guid` is present.
    pub fn contains(&self, guid: AssetId) -> bool {
        self.index.contains_key(&guid)
    }

    /// The index entry for `guid`, if present.
    pub fn entry(&self, guid: AssetId) -> Option<&PackEntry> {
        self.index.get(&guid)
    }

    /// Iterate the index in GUID order.
    pub fn index(&self) -> impl Iterator<Item = &PackEntry> {
        self.index.values()
    }

    /// Read, decompress, and integrity-verify a payload by GUID.
    pub fn read(&self, guid: AssetId) -> Result<Vec<u8>> {
        let e = self
            .index
            .get(&guid)
            .ok_or(AssetError::UnknownAsset(guid))?;
        let start = e.offset as usize;
        let stored = &self.data[start..start + e.stored_len as usize];
        let payload = if e.compressed {
            zstd::decode_all(stored).map_err(|err| {
                AssetError::Pack(format!("zstd decode {guid}: {err} (corrupt pack?)"))
            })?
        } else {
            stored.to_vec()
        };
        // Integrity: the decompressed bytes must hash to what the index promised.
        let got = ContentHash::of(&payload);
        if got != e.content_hash {
            return Err(AssetError::Pack(format!(
                "content hash mismatch for {guid} (corrupt pack)"
            )));
        }
        Ok(payload)
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
    }
}
