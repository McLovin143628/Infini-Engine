//! One page of the heightfield: a square block of `f32` sample heights measured
//! against an `f64` world anchor (the precision doctrine — tile-local `f32`
//! against an `f64` origin, so a planetary-scale terrain never loses vertical
//! resolution to `f32` range).

use glam::DVec3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The default per-sample splat weight: **100 % layer 0** (`[255, 0, 0, 0]`), so
/// an unpainted terrain shades entirely as its first [`TerrainLayer`]. Channels
/// are `[layer0, layer1, layer2, layer3]` and are kept normalized to sum ≈ 255.
pub const DEFAULT_WEIGHT: [u8; 4] = [255, 0, 0, 0];

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
/// Serde: `origin` (as a portable `[f64; 3]`, since the workspace `glam` pin has
/// no `serde` feature) + a flat `heights` sequence, and — **only when non-empty**
/// (`skip_serializing_if`) — a flat `weights` sequence. An old tile (no `weights`
/// field) decodes with the default (empty) weights, and an unpainted new tile
/// serializes without a `weights` field, so existing bytes round-trip unchanged;
/// a painted tile appends the `weights` field. `resolution` is not stored on the
/// tile (it is a terrain-wide constant on [`super::TerrainData`]); the terrain
/// validates `heights.len() == resolution²` and, when present,
/// `weights.len() == resolution²` on load.
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
}

/// Serde wire form: `origin` as `[f64; 3]` (glam `DVec3` isn't serde-derivable
/// without enabling glam's `serde` feature workspace-wide). `weights` is appended
/// (P10.4) and skipped when empty, so pre-P10.4 tiles — and unpainted new tiles —
/// encode byte-identically to the two-field form.
#[derive(Serialize, Deserialize)]
struct TerrainTileRaw {
    origin: [f64; 3],
    heights: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    weights: Vec<[u8; 4]>,
}

impl Serialize for TerrainTile {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        TerrainTileRaw {
            origin: self.origin.to_array(),
            heights: self.heights.clone(),
            weights: self.weights.clone(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for TerrainTile {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = TerrainTileRaw::deserialize(d)?;
        Ok(Self {
            origin: DVec3::from_array(raw.origin),
            heights: raw.heights,
            weights: raw.weights,
        })
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
}
