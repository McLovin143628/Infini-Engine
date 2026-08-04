//! [`VoxelChunk`]: one dense block of the signed-distance field, and the
//! [`ChunkKey`] that names it.

use glam::DVec3;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Samples per chunk side. **16**, and the reasoning is a paging argument, not a
/// taste one — the same shape of argument `crate::delta`'s module docs make about
/// dense-vs-sparse patches.
///
/// A sample costs 5 bytes (an `f32` distance + a `u8` material), so:
///
/// | dim | samples | bytes/chunk | world span @ 0.5 m |
/// |---|---|---|---|
/// | 16³ | 4 096 | ~20 KB | 8 m |
/// | 32³ | 32 768 | ~160 KB | 16 m |
///
/// Three things pushed this to 16:
///
/// * **A chunk is the unit of re-meshing.** A carve re-meshes every chunk it
///   touched, and meshing is `O(dim³)`. At 16³ a dab re-meshes a few thousand
///   cells; at 32³ the same dab re-meshes eight times that for geometry that did
///   not change. P21.4 runs this at *runtime*, inside a frame, which is the
///   constraint that actually binds.
/// * **A chunk is the unit of undo.** [`crate::VoxelDelta`] stores a dense
///   sub-box per touched chunk; a smaller chunk means a tighter box and less
///   `before`/`after` that is a no-op.
/// * **A chunk is the unit of paging.** 20 KB is a comfortable read; 160 KB is a
///   hitch. The cost is more directory entries and more draw calls — a 100 m cave
///   system is ~12 chunks per axis rather than ~6 — and that is the trade that was
///   taken, because directory entries are 32 bytes and draw calls are the cheapest
///   thing in this list to batch later.
///
/// Anything larger also makes the mesher's padded gather ([`crate::mesh`]) grow as
/// `(dim + 2)³`, which at 32³ is 39 304 samples of scratch per chunk.
pub const CHUNK_DIM: u32 = 16;

/// Samples in one chunk (`CHUNK_DIM³`).
pub const CHUNK_VOXELS: usize = (CHUNK_DIM * CHUNK_DIM * CHUNK_DIM) as usize;

/// Splat material channels a voxel surface can carry — **4**, deliberately the
/// same count as the terrain's splat layers and with the **same indices**.
///
/// The alignment is the whole point: a cave mouth is a place where voxel surface
/// and heightfield surface meet, and if material 1 meant "rock" on one side and
/// "snow" on the other, every opening in the world would show a seam that no
/// amount of normal blending could hide. So a voxel material index is *the same
/// index* into the `Terrain` component's `layers`, the projectors hand the
/// renderer the same `RenderTerrainLayer` array for both, and P21.2's seam
/// blending has one palette to blend rather than two to reconcile.
pub const MATERIAL_COUNT: usize = 4;

/// The material a never-painted (and every empty) sample carries.
pub const DEFAULT_MATERIAL: u8 = 0;

/// The distance band, in **voxels**, that stored signed distances are clamped to.
///
/// The field only has to be accurate near the surface: the mesher reads it within
/// one cell of a sign change, and the ops only ever push a sample toward or away
/// from a surface. Clamping the far field to a constant is what makes an
/// untouched region *uniform* — every empty sample is exactly [`EMPTY_SDF`] and
/// every deep-solid one exactly `-SDF_BAND` — so two volumes with the same
/// geometry have the same bytes however they were carved, and a chunk full of
/// rock costs the same as a chunk full of air.
///
/// 4 voxels is wide enough that the gradient at a surface is never clipped (the
/// mesher's central difference reads one voxel either side) and narrow enough
/// that the band is genuinely reached.
pub const SDF_BAND: f32 = 4.0;

/// The signed distance of a sample that is nowhere near anything solid — and the
/// value every sample of an absent chunk reads as. Equal to [`SDF_BAND`] by
/// construction, so "clamped far field" and "empty" are the same number and an
/// all-empty chunk is uniform.
pub const EMPTY_SDF: f32 = SDF_BAND;

/// A chunk's coordinate on the volume's chunk grid.
///
/// Chunk `(x, y, z)` owns global sample coordinates `x·CHUNK_DIM + i` for
/// `i ∈ [0, CHUNK_DIM)` on each axis — a **partition** of the global sample grid
/// with no shared plane. (Terrain tiles deliberately *share* their edge samples;
/// voxel chunks deliberately do not, because a shared sample would have two
/// authorities and a carve would have to write both. The mesher gets its
/// cross-chunk continuity from a padded gather instead — see [`crate::mesh`].)
///
/// `Ord` is derived, so keys sort lexicographically by `(x, y, z)`. Every
/// directory, every mesh walk and every residency report is in that order, which
/// is what makes them byte-deterministic.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct ChunkKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkKey {
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// The chunk containing global sample `(gx, gy, gz)` (floored, so negative
    /// coordinates group correctly — sample −1 belongs to chunk −1, not chunk 0).
    #[inline]
    pub fn of_sample(gx: i32, gy: i32, gz: i32) -> Self {
        let n = CHUNK_DIM as i32;
        Self {
            x: gx.div_euclid(n),
            y: gy.div_euclid(n),
            z: gz.div_euclid(n),
        }
    }

    /// This chunk's lowest owned global sample coordinate.
    #[inline]
    pub fn base_sample(self) -> [i32; 3] {
        let n = CHUNK_DIM as i32;
        [self.x * n, self.y * n, self.z * n]
    }

    /// This key offset by `(dx, dy, dz)` chunks (saturating; a key that would
    /// overflow `i32` cannot name a real chunk and simply never matches one).
    #[inline]
    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self {
            x: self.x.saturating_add(dx),
            y: self.y.saturating_add(dy),
            z: self.z.saturating_add(dz),
        }
    }
}

/// A dense block of the signed-distance field plus its per-sample material.
///
/// ## Layout
///
/// `sdf` is always `CHUNK_VOXELS` long, row-major with **x fastest, then y, then
/// z** ([`index`](VoxelChunk::index)). Values are signed distances in **voxels**,
/// negative inside solid, clamped to `±`[`SDF_BAND`].
///
/// `materials` is the sparse half: `CHUNK_VOXELS` long **or empty**, where empty
/// means every sample carries [`DEFAULT_MATERIAL`]. A chunk carved out of a
/// single-material volume — which is most of them — pays one length byte instead
/// of 4 KB. Same trick, same reason, as `inf_terrain::TerrainTile`'s weights.
///
/// ## The serde form is FORMAT-AWARE, and that is load-bearing
///
/// Human-readable formats (JSON/TOML) skip an empty `materials`; **bincode always
/// encodes it**. A `skip_serializing_if` field desyncs a non-self-describing
/// stream — the engine-wide law, paid for three times before this crate existed —
/// because bincode carries no field names and a reader that expected four fields
/// and found three walks off into the next chunk's bytes. So there are two wire
/// structs and one hand-written codec pair, exactly as
/// `inf_terrain::TerrainTile` has.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxelChunk {
    /// `CHUNK_VOXELS` signed distances in voxels, row-major (x fastest).
    sdf: Vec<f32>,
    /// `CHUNK_VOXELS` material indices, **or empty** for the uniform
    /// [`DEFAULT_MATERIAL`].
    materials: Vec<u8>,
}

/// Serde wire form for **human-readable** formats (JSON/TOML): `materials` is
/// skipped when empty, so an unpainted chunk encodes as the one-field form.
#[derive(Serialize, Deserialize)]
struct VoxelChunkRaw {
    sdf: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    materials: Vec<u8>,
}

/// Serde wire form for **non-self-describing** formats (bincode — the
/// `.inf_voxel` chunk blob). `materials` is **always** encoded, as a plain
/// length-prefixed sequence (a 0-length count for an unpainted chunk), because a
/// skipped field desyncs the stream. See the type docs.
#[derive(Serialize, Deserialize)]
struct VoxelChunkBin {
    sdf: Vec<f32>,
    #[serde(default)]
    materials: Vec<u8>,
}

impl Serialize for VoxelChunk {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if s.is_human_readable() {
            VoxelChunkRaw {
                sdf: self.sdf.clone(),
                materials: self.materials.clone(),
            }
            .serialize(s)
        } else {
            VoxelChunkBin {
                sdf: self.sdf.clone(),
                materials: self.materials.clone(),
            }
            .serialize(s)
        }
    }
}

impl<'de> Deserialize<'de> for VoxelChunk {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (sdf, materials) = if d.is_human_readable() {
            let raw = VoxelChunkRaw::deserialize(d)?;
            (raw.sdf, raw.materials)
        } else {
            let raw = VoxelChunkBin::deserialize(d)?;
            (raw.sdf, raw.materials)
        };
        if sdf.len() != CHUNK_VOXELS {
            return Err(serde::de::Error::custom(format!(
                "voxel chunk carries {} samples, expected {CHUNK_VOXELS}",
                sdf.len()
            )));
        }
        if !materials.is_empty() && materials.len() != CHUNK_VOXELS {
            return Err(serde::de::Error::custom(format!(
                "voxel chunk carries {} material indices, expected 0 or {CHUNK_VOXELS}",
                materials.len()
            )));
        }
        Ok(Self { sdf, materials })
    }
}

impl Default for VoxelChunk {
    fn default() -> Self {
        Self::empty()
    }
}

impl VoxelChunk {
    /// A chunk of nothing: every sample [`EMPTY_SDF`], every material default.
    /// Byte-identical to what an absent chunk reads as.
    pub fn empty() -> Self {
        Self {
            sdf: vec![EMPTY_SDF; CHUNK_VOXELS],
            materials: Vec::new(),
        }
    }

    /// A chunk of solid rock: every sample `-SDF_BAND`, material `material`.
    pub fn solid(material: u8) -> Self {
        let mut c = Self {
            sdf: vec![-SDF_BAND; CHUNK_VOXELS],
            materials: Vec::new(),
        };
        if material != DEFAULT_MATERIAL {
            c.materials = vec![material.min(MATERIAL_COUNT as u8 - 1); CHUNK_VOXELS];
        }
        c
    }

    /// Build a chunk from an explicit sample buffer (row-major, x fastest).
    /// `None` on a length mismatch. Values are clamped to the band.
    pub fn from_sdf(sdf: Vec<f32>) -> Option<Self> {
        (sdf.len() == CHUNK_VOXELS).then(|| Self {
            sdf: sdf.into_iter().map(clamp_sdf).collect(),
            materials: Vec::new(),
        })
    }

    /// Author every sample from a function of its **chunk-local** integer sample
    /// coordinate, returning a signed distance in voxels (clamped to the band).
    ///
    /// A gather: one output per iteration step, in a fixed order, so two runs
    /// produce byte-identical chunks.
    pub fn from_fn(mut f: impl FnMut(u32, u32, u32) -> f64) -> Self {
        let n = CHUNK_DIM;
        let mut sdf = Vec::with_capacity(CHUNK_VOXELS);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    sdf.push(clamp_sdf(f(i, j, k) as f32));
                }
            }
        }
        Self {
            sdf,
            materials: Vec::new(),
        }
    }

    /// Row-major index of chunk-local sample `(i, j, k)` — x fastest, then y,
    /// then z. Out-of-range coordinates wrap into range with `%`, which cannot
    /// happen through the checked accessors and keeps this branch-free.
    #[inline]
    pub fn index(i: u32, j: u32, k: u32) -> usize {
        let n = CHUNK_DIM;
        (((k % n) * n + (j % n)) * n + (i % n)) as usize
    }

    /// The raw sample buffer.
    #[inline]
    pub fn sdf(&self) -> &[f32] {
        &self.sdf
    }

    /// The raw material buffer — **empty** for a chunk that is uniformly
    /// [`DEFAULT_MATERIAL`] (see [`materials_are_default`](Self::materials_are_default)).
    #[inline]
    pub fn materials(&self) -> &[u8] {
        &self.materials
    }

    /// `true` when no sample has been given a material (the sparse form).
    #[inline]
    pub fn materials_are_default(&self) -> bool {
        self.materials.is_empty()
    }

    /// Signed distance at chunk-local sample `(i, j, k)`, in voxels.
    #[inline]
    pub fn sample(&self, i: u32, j: u32, k: u32) -> f32 {
        self.sdf[Self::index(i, j, k)]
    }

    /// Material index at chunk-local sample `(i, j, k)`.
    #[inline]
    pub fn material(&self, i: u32, j: u32, k: u32) -> u8 {
        if self.materials.is_empty() {
            DEFAULT_MATERIAL
        } else {
            self.materials[Self::index(i, j, k)]
        }
    }

    /// Write a signed distance (clamped to `±`[`SDF_BAND`]).
    #[inline]
    pub fn set_sample(&mut self, i: u32, j: u32, k: u32, v: f32) {
        let idx = Self::index(i, j, k);
        self.sdf[idx] = clamp_sdf(v);
    }

    /// Write a material index, materializing the sparse buffer on first use and
    /// clamping to the [`MATERIAL_COUNT`] channels the palette actually has.
    #[inline]
    pub fn set_material(&mut self, i: u32, j: u32, k: u32, m: u8) {
        let m = m.min(MATERIAL_COUNT as u8 - 1);
        if self.materials.is_empty() {
            if m == DEFAULT_MATERIAL {
                return;
            }
            self.materials = vec![DEFAULT_MATERIAL; CHUNK_VOXELS];
        }
        let idx = Self::index(i, j, k);
        self.materials[idx] = m;
    }

    /// Drop the material buffer back to its sparse form when every entry is the
    /// default — so a chunk whose paint was carved away costs what an unpainted
    /// one costs, and two histories reaching the same geometry reach the same
    /// bytes.
    pub fn compact_materials(&mut self) {
        if !self.materials.is_empty() && self.materials.iter().all(|&m| m == DEFAULT_MATERIAL) {
            self.materials.clear();
        }
    }

    /// `true` when no sample is inside solid — nothing to mesh from this chunk's
    /// own samples (its seam cells may still be active; see [`crate::mesh`]).
    pub fn is_empty_space(&self) -> bool {
        self.sdf.iter().all(|&v| v >= 0.0)
    }

    /// `true` when every sample is inside solid.
    pub fn is_solid(&self) -> bool {
        self.sdf.iter().all(|&v| v < 0.0)
    }

    /// Inclusive `(min, max)` of the sample values.
    pub fn sdf_bounds(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in &self.sdf {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if self.sdf.is_empty() {
            (0.0, 0.0)
        } else {
            (lo, hi)
        }
    }

    /// **Trilinearly** interpolated signed distance at a fractional *chunk-local*
    /// position, in voxels.
    ///
    /// `p` is in chunk-local sample units (`(0,0,0)` is sample `(0,0,0)`,
    /// `(1.5, 0, 0)` is halfway between samples 0 and 1 on x). Positions are
    /// clamped into `[0, CHUNK_DIM − 1]`, so this reads this chunk's own samples
    /// and **never a neighbour's** — it is the single-chunk probe. The
    /// volume-wide one that does cross chunk boundaries is
    /// [`VoxelData::sdf_at`](crate::VoxelData::sdf_at).
    ///
    /// Trilinear interpolation is *exact* for a field that is linear in each
    /// axis, which is what the analytic test pins.
    pub fn sdf_at_local(&self, p: DVec3) -> f64 {
        let last = (CHUNK_DIM - 1) as f64;
        let x = p.x.clamp(0.0, last);
        let y = p.y.clamp(0.0, last);
        let z = p.z.clamp(0.0, last);
        let (i0, fx) = split(x, last);
        let (j0, fy) = split(y, last);
        let (k0, fz) = split(z, last);
        let i1 = (i0 + 1).min(CHUNK_DIM - 1);
        let j1 = (j0 + 1).min(CHUNK_DIM - 1);
        let k1 = (k0 + 1).min(CHUNK_DIM - 1);
        let s = |i: u32, j: u32, k: u32| self.sample(i, j, k) as f64;
        let c00 = s(i0, j0, k0) * (1.0 - fx) + s(i1, j0, k0) * fx;
        let c10 = s(i0, j1, k0) * (1.0 - fx) + s(i1, j1, k0) * fx;
        let c01 = s(i0, j0, k1) * (1.0 - fx) + s(i1, j0, k1) * fx;
        let c11 = s(i0, j1, k1) * (1.0 - fx) + s(i1, j1, k1) * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }
}

/// Split a clamped coordinate into its base sample index and fraction.
#[inline]
fn split(v: f64, last: f64) -> (u32, f64) {
    let f = v.floor().min(last - 1.0).max(0.0);
    (f as u32, v - f)
}

/// Clamp a signed distance into the stored band (see [`SDF_BAND`]). A NaN — which
/// can only arrive from a degenerate authored field — normalizes to "empty"
/// rather than propagating into every vertex position downstream.
#[inline]
pub(crate) fn clamp_sdf(v: f32) -> f32 {
    if v.is_nan() {
        return EMPTY_SDF;
    }
    v.clamp(-SDF_BAND, SDF_BAND)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin() -> impl bincode::config::Config {
        bincode::config::standard()
    }

    #[test]
    fn an_empty_chunk_is_uniform_and_matches_an_absent_one() {
        let c = VoxelChunk::empty();
        assert_eq!(c.sdf().len(), CHUNK_VOXELS);
        assert!(c.materials_are_default());
        assert!(c.is_empty_space());
        assert!(!c.is_solid());
        assert!(c.sdf().iter().all(|&v| v == EMPTY_SDF));
        assert_eq!(c.sdf_bounds(), (EMPTY_SDF, EMPTY_SDF));
        assert_eq!(EMPTY_SDF, SDF_BAND, "empty IS the clamped far field");
    }

    #[test]
    fn indexing_is_x_fastest_then_y_then_z() {
        assert_eq!(VoxelChunk::index(0, 0, 0), 0);
        assert_eq!(VoxelChunk::index(1, 0, 0), 1);
        assert_eq!(VoxelChunk::index(0, 1, 0), CHUNK_DIM as usize);
        assert_eq!(VoxelChunk::index(0, 0, 1), (CHUNK_DIM * CHUNK_DIM) as usize);
        assert_eq!(
            VoxelChunk::index(CHUNK_DIM - 1, CHUNK_DIM - 1, CHUNK_DIM - 1),
            CHUNK_VOXELS - 1
        );
    }

    #[test]
    fn chunk_keys_partition_the_global_sample_grid() {
        let n = CHUNK_DIM as i32;
        assert_eq!(ChunkKey::of_sample(0, 0, 0), ChunkKey::new(0, 0, 0));
        assert_eq!(ChunkKey::of_sample(n - 1, 0, 0), ChunkKey::new(0, 0, 0));
        assert_eq!(ChunkKey::of_sample(n, 0, 0), ChunkKey::new(1, 0, 0));
        // Negative coordinates floor, so sample −1 belongs to chunk −1 (a `/`
        // here would put it in chunk 0 and overlap the two).
        assert_eq!(ChunkKey::of_sample(-1, 0, 0), ChunkKey::new(-1, 0, 0));
        assert_eq!(ChunkKey::of_sample(-n, 0, 0), ChunkKey::new(-1, 0, 0));
        assert_eq!(ChunkKey::of_sample(-n - 1, 0, 0), ChunkKey::new(-2, 0, 0));
        assert_eq!(ChunkKey::new(-1, 2, 0).base_sample(), [-n, 2 * n, 0]);
        assert_eq!(
            ChunkKey::new(1, 1, 1).offset(-1, 0, 2),
            ChunkKey::new(0, 1, 3)
        );
    }

    /// The **format-aware serde gate**: an unpainted chunk skips its material
    /// buffer in JSON and still encodes a zero-length count in bincode. A shared
    /// codec would either bloat the JSON or desync the bincode stream — the law
    /// this repo has now paid for four times.
    #[test]
    fn chunk_serde_is_format_aware_and_round_trips_byte_identically() {
        let mut c =
            VoxelChunk::from_fn(|i, j, k| (i as f64 + j as f64 * 2.0 + k as f64 * 3.0) - 8.0);

        // ── unpainted ──
        let json = serde_json::to_string(&c).unwrap();
        assert!(
            !json.contains("materials"),
            "an unpainted chunk must not spell out its default material buffer: {json}"
        );
        let back: VoxelChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
        assert_eq!(serde_json::to_string(&back).unwrap(), json);

        let bytes = bincode::serde::encode_to_vec(&c, bin()).unwrap();
        let (rt, _): (VoxelChunk, usize) =
            bincode::serde::decode_from_slice(&bytes, bin()).unwrap();
        assert_eq!(rt, c);
        assert_eq!(
            bincode::serde::encode_to_vec(&rt, bin()).unwrap(),
            bytes,
            "bincode must be byte-identical across a round trip"
        );
        let unpainted_len = bytes.len();

        // ── painted ──
        c.set_material(1, 2, 3, 2);
        assert!(!c.materials_are_default());
        assert_eq!(c.material(1, 2, 3), 2);
        assert_eq!(c.material(0, 0, 0), DEFAULT_MATERIAL);
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("materials"));
        assert_eq!(serde_json::from_str::<VoxelChunk>(&json).unwrap(), c);

        let painted = bincode::serde::encode_to_vec(&c, bin()).unwrap();
        let (rt, _): (VoxelChunk, usize) =
            bincode::serde::decode_from_slice(&painted, bin()).unwrap();
        assert_eq!(rt, c);
        assert_eq!(bincode::serde::encode_to_vec(&rt, bin()).unwrap(), painted);
        // The price of the sparse form, stated exactly: an unpainted chunk pays
        // one zero-length byte; a painted one pays bincode's 3-byte varint for a
        // 4096-element count plus the dense buffer itself.
        assert_eq!(painted.len(), unpainted_len - 1 + 3 + CHUNK_VOXELS);

        // Carving the paint back out returns the chunk to its unpainted bytes.
        c.set_material(1, 2, 3, DEFAULT_MATERIAL);
        c.compact_materials();
        assert!(c.materials_are_default());
        assert_eq!(
            bincode::serde::encode_to_vec(&c, bin()).unwrap().len(),
            unpainted_len
        );
    }

    /// A chunk blob whose sample count is wrong is rejected at decode, not
    /// silently accepted with a short buffer that would panic on the first
    /// `sample()`.
    #[test]
    fn a_malformed_chunk_is_rejected_at_decode() {
        let short = serde_json::json!({ "sdf": [0.0, 1.0, 2.0] }).to_string();
        let err = serde_json::from_str::<VoxelChunk>(&short).unwrap_err();
        assert!(err.to_string().contains("expected"), "{err}");

        let bad_mat = serde_json::json!({ "sdf": vec![0.0f32; CHUNK_VOXELS], "materials": [1, 2] })
            .to_string();
        assert!(serde_json::from_str::<VoxelChunk>(&bad_mat).is_err());
    }

    #[test]
    fn values_are_clamped_into_the_band_and_nan_reads_as_empty() {
        let c = VoxelChunk::from_sdf(vec![0.0; CHUNK_VOXELS]).unwrap();
        assert_eq!(c.sample(0, 0, 0), 0.0);
        assert!(VoxelChunk::from_sdf(vec![0.0; 3]).is_none());

        let mut c = VoxelChunk::empty();
        c.set_sample(0, 0, 0, 1e9);
        c.set_sample(1, 0, 0, -1e9);
        c.set_sample(2, 0, 0, f32::NAN);
        assert_eq!(c.sample(0, 0, 0), SDF_BAND);
        assert_eq!(c.sample(1, 0, 0), -SDF_BAND);
        assert_eq!(
            c.sample(2, 0, 0),
            EMPTY_SDF,
            "a NaN must normalize rather than propagate into every vertex downstream"
        );
        // A material past the palette clamps rather than indexing out of it.
        c.set_material(0, 0, 0, 200);
        assert_eq!(c.material(0, 0, 0), MATERIAL_COUNT as u8 - 1);
    }

    /// **The analytic trilinear gate**: on a field that is linear in each axis,
    /// trilinear interpolation is exact, so `sdf_at_local` must reproduce the
    /// closed form at fractional positions — not merely at the samples, which any
    /// nearest-neighbour lookup would also pass.
    #[test]
    fn trilinear_sdf_matches_the_analytic_field_it_was_built_from() {
        // f(x, y, z) = 0.25x − 0.5y + 0.125z − 2  (inside the band everywhere in
        // a 16³ chunk, so nothing is clipped).
        let f = |x: f64, y: f64, z: f64| 0.25 * x - 0.5 * y + 0.125 * z - 2.0;
        let c = VoxelChunk::from_fn(|i, j, k| f(i as f64, j as f64, k as f64));

        for &(x, y, z) in &[
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.25, 2.75, 3.5),
            (7.5, 7.5, 7.5),
            (14.9, 0.1, 9.3),
            (15.0, 15.0, 15.0),
        ] {
            let got = c.sdf_at_local(DVec3::new(x, y, z));
            let want = f(x, y, z);
            assert!(
                (got - want).abs() < 1e-5,
                "trilinear at ({x},{y},{z}) gave {got}, analytic is {want}"
            );
        }

        // Out of range clamps to the boundary sample rather than reading a
        // neighbour (that is `VoxelData::sdf_at`'s job) or panicking.
        assert!((c.sdf_at_local(DVec3::new(-5.0, -5.0, -5.0)) - f(0.0, 0.0, 0.0)).abs() < 1e-5);
        assert!((c.sdf_at_local(DVec3::new(99.0, 99.0, 99.0)) - f(15.0, 15.0, 15.0)).abs() < 1e-5);
    }

    #[test]
    fn solid_chunks_carry_their_material_densely() {
        let s = VoxelChunk::solid(2);
        assert!(s.is_solid());
        assert!(!s.is_empty_space());
        assert_eq!(s.material(5, 5, 5), 2);
        assert!(!s.materials_are_default());
        // Material 0 stays sparse — the common case pays nothing.
        let plain = VoxelChunk::solid(DEFAULT_MATERIAL);
        assert!(plain.materials_are_default());
        assert!(plain.is_solid());
    }
}
