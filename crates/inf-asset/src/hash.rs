//! Content-addressed hashing.
//!
//! The whole asset pipeline is keyed on **content hashes**, not timestamps:
//! imports are cached by the hash of their source bytes + settings, thumbnails
//! by the hash of the imported payload, and change detection compares hashes
//! rather than mtimes (which lie across checkouts and syncs).
//!
//! We use xxh3-128 — fast, non-cryptographic, and wide enough that accidental
//! collisions across a project's asset set are not a practical concern.

use std::fmt;

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_128;

/// A 128-bit content hash, rendered as lowercase hex.
///
/// Serializes as the hex string so sidecars stay human-readable and diffable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash(pub u128);

impl ContentHash {
    /// Hash a byte slice.
    pub fn of(bytes: &[u8]) -> Self {
        Self(xxh3_128(bytes))
    }

    /// Combine several byte slices in order (e.g. source bytes + serialized
    /// import settings) into one hash. Order-sensitive by construction.
    ///
    /// **Streamed, not concatenated** (Hardening Wave E). This used to build a
    /// scratch `Vec` holding every part and hash that — so the import path,
    /// whose first part is the whole `fs::read` of a glTF/GLB/EXR, allocated and
    /// memcpy'd the entire source file a second time **even on a cache hit**,
    /// for an eight-byte length prefix. Measured on this machine: 6.8 ms at
    /// 16 MiB, 26.5 ms at 64 MiB, 96.2 ms at 200 MiB, with 2x peak RSS to match.
    /// Feeding the same byte sequence through `Xxh3` incrementally is the same
    /// digest — which
    /// [`streaming_of_parts_matches_the_concatenation`](#) pins byte for byte,
    /// because this value is **persisted**: it keys the import cache manifest and
    /// every sidecar's `content_hash`, so a digest that moved would silently
    /// invalidate every cache in every project.
    pub fn of_parts(parts: &[&[u8]]) -> Self {
        let mut h = xxhash_rust::xxh3::Xxh3::new();
        for p in parts {
            // Length-prefix each part so `["ab","c"]` and `["a","bc"]` differ.
            h.update(&(p.len() as u64).to_le_bytes());
            h.update(p);
        }
        Self(h.digest128())
    }

    /// The zero hash (sentinel for "not yet hashed").
    pub const ZERO: Self = Self(0);

    /// Lowercase 32-char hex.
    pub fn to_hex(self) -> String {
        format!("{:032x}", self.0)
    }

    /// A short 12-char prefix, for cache filenames and UI.
    pub fn short(self) -> String {
        let full = self.to_hex();
        full[..12].to_string()
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl std::str::FromStr for ContentHash {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(u128::from_str_radix(s, 16)?))
    }
}

impl Serialize for ContentHash {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips() {
        let h = ContentHash::of(b"hello world");
        let s = h.to_hex();
        assert_eq!(s.len(), 32);
        assert_eq!(s.parse::<ContentHash>().unwrap(), h);
    }

    #[test]
    fn parts_are_order_and_boundary_sensitive() {
        assert_ne!(
            ContentHash::of_parts(&[b"ab", b"c"]),
            ContentHash::of_parts(&[b"a", b"bc"]),
        );
        assert_ne!(
            ContentHash::of_parts(&[b"a", b"b"]),
            ContentHash::of_parts(&[b"b", b"a"]),
        );
    }

    #[test]
    fn same_bytes_same_hash() {
        assert_eq!(ContentHash::of(b"x"), ContentHash::of(b"x"));
        assert_ne!(ContentHash::of(b"x"), ContentHash::of(b"y"));
    }

    #[test]
    fn serde_is_the_hex_string() {
        let h = ContentHash::of(b"abc");
        let json = serde_json::to_string(&h).unwrap();
        assert_eq!(json, format!("\"{}\"", h.to_hex()));
        let back: ContentHash = serde_json::from_str(&json).unwrap();
        assert_eq!(back, h);
    }

    /// **The digest may not move.** `of_parts` is persisted — it keys the import
    /// cache manifest and every sidecar's `content_hash` — so streaming it
    /// instead of concatenating has to produce the identical 128 bits, not
    /// merely an equally good hash. Checked against the concatenation this
    /// function used to build, over splits chosen to cross xxh3's internal
    /// block boundaries (it switches strategy at 16, 128 and 240 bytes and
    /// buffers in 1 KiB stripes).
    #[test]
    fn streaming_of_parts_matches_the_concatenation() {
        let concat = |parts: &[&[u8]]| {
            let mut buf = Vec::new();
            for p in parts {
                buf.extend_from_slice(&(p.len() as u64).to_le_bytes());
                buf.extend_from_slice(p);
            }
            ContentHash::of(&buf)
        };
        let blob: Vec<u8> = (0..5000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        for split in [
            0usize, 1, 7, 15, 16, 17, 63, 127, 128, 129, 239, 240, 241, 1023, 1024, 1025, 4999,
            5000,
        ] {
            let (a, b) = blob.split_at(split);
            assert_eq!(
                ContentHash::of_parts(&[a, b]),
                concat(&[a, b]),
                "streaming diverged from the concatenation at split {split}"
            );
        }
        // Three parts, an empty part, and the no-parts case.
        assert_eq!(ContentHash::of_parts(&[]), concat(&[]));
        assert_eq!(ContentHash::of_parts(&[b""]), concat(&[b""]));
        assert_eq!(
            ContentHash::of_parts(&[&blob[..100], b"", &blob[100..]]),
            concat(&[&blob[..100], b"", &blob[100..]])
        );
        // …and the property the length prefix exists for still holds.
        assert_ne!(
            ContentHash::of_parts(&[b"ab", b"c"]),
            ContentHash::of_parts(&[b"a", b"bc"])
        );
    }
}
