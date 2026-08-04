//! Terrain: a paged `f64` world-space heightfield — the CPU foundation of the
//! engine's massive-terrain technology (P10.1).
//!
//! A [`TerrainData`] is a sparse grid of [`TerrainTile`] pages (`f32` heights
//! against an `f64` per-tile anchor — the precision doctrine), keyed by tile
//! coordinate in a `BTreeMap` so only authored regions cost memory and every
//! query/iteration/serialization is deterministic. It answers `f64` world-space
//! height/normal queries ([`TerrainData::height_at`] / [`TerrainData::normal_at`])
//! and imports/exports 16-bit heightmaps ([`TerrainData::from_height_image`] /
//! [`TerrainData::export_png16`]).
//!
//! The [`HeightSource`] trait is the seam downstream systems (PCG scatter,
//! P10.5; sculpt brushes, P10.2) code against — a minimal `height`/`normal`
//! lookup that hides the paging. The clipmap-LOD renderer lives in `inf-render`;
//! it consumes a projection of this data.
//!
//! ## Streaming (P16.3)
//!
//! Terrain also lives outside the level as a [`TerrainAsset`] (`.inf_terrain`): a
//! header + tile directory + 16-byte-aligned per-tile blobs across an LOD
//! [`pyramid`], cooked **uncompressed** so a runtime pages one tile straight out
//! of an mmap'd `.inf_pack` with no decode and no copy. [`TerrainData`] is then
//! the *resident* working set — grown and shrunk against a [`TileStore`] by an
//! explicit set of wanted `(coord, lod)` keys, with per-tile versions and a dirty
//! set for write-back.
//!
//! ## Biomes (P19.2)
//!
//! A terrain also carries a per-sample **biome id** layer beside its heights,
//! splat weights and erosion data maps — sparse on the same rule, so an unpainted
//! terrain costs nothing. The ids name entries in a [`BiomeSet`] (`.inf_biomes`),
//! the asset that defines a level's biomes; [`biomepaint`] is the brush, and its
//! module docs state why the boundary is deliberately **hard-edged**.
//!
//! [`residency`] executes wants; [`wants`] **computes** them (pure functions: a
//! camera-driven quadtree cut with hysteresis, and the sim's level-0
//! neighbourhood); [`stream`] holds the state that drives them — the cold stores
//! ([`PackTileStore`] over a cooked pack, [`FileTileStore`] over a loose file) and
//! the [`TerrainStreamer`].
//!
//! ### The determinism doctrine
//!
//! The fixed-step sim's results must **never** depend on camera-driven residency.
//! Sim wants ([`sim_wants`]) are computed from sim state alone and loaded
//! synchronously at the fixed-step boundary into the `TerrainData` the world
//! holds; render wants ([`render_wants`]) are camera-driven and land in a
//! **second** working set inside the [`TerrainStreamer`] that the world has no
//! reference to. See [`stream`] for why that separation is structural rather than
//! conventional.

pub mod asset;
pub mod biome;
pub mod biomepaint;
mod brush;
pub mod chunked;
mod data;
mod delta;
pub mod erosion;
pub mod import;
pub mod maps;
pub use erosion::{erode, erode_terrain, erode_with, ErosionParams, ErosionStats};
mod noise;
pub mod pyramid;
mod raycast;
mod region;
pub mod residency;
mod splat;
pub mod stream;
mod tile;
pub mod wants;
pub mod writeback;

use glam::DVec3;

pub use asset::{
    build_terrain_asset, header_len, read_terrain_asset, write_terrain_asset, TerrainAsset,
    TerrainAssetBuilder, TerrainAssetError, TerrainAssetHeader, TerrainAssetReader,
    TerrainAssetView, TileDirEntry, HEADER_LEN_V1, HEADER_LEN_V2, HEADER_LEN_V3, HEADER_LEN_V4,
    HEADER_LEN_V5, TERRAIN_ASSET_SCHEMA_VERSION, TILE_ALIGN,
};
pub use biome::{
    BiomeDef, BiomeSet, BiomeSetError, BIOME_SET_SCHEMA_VERSION, UNASSIGNED_BIOME_COLOR,
};
pub use biomepaint::{apply_paint_biome, biome_claims, BiomeDelta, BiomePatch, BiomeStroke};
pub use brush::{apply_brush, dab_positions, BrushOp, BrushParams, Falloff, FlattenTarget, Stroke};
pub use chunked::{
    import_heightmap, import_heightmap_in, import_heightmap_reader, import_heightmap_reader_in,
    ChunkedImportOptions, ImportProgress, ImportReport,
};
pub use data::{
    TerrainData, TerrainDataFrozenV1, TerrainDataFrozenV2, DEFAULT_METERS_PER_SAMPLE,
    DEFAULT_TILE_RESOLUTION,
};
pub use delta::{HeightDelta, HoleDelta, HoleDeltaBuilder, HolePatch, TilePatch};
pub use import::{
    encode_png16, probe_heightmap, probe_heightmap_bytes, HeightImage, HeightMode, HeightmapFormat,
    HeightmapGrid, HeightmapImport, HeightmapProbe, TerrainError,
};
pub use maps::{DataMapDelta, DataMapPatch};
pub use noise::fbm_signed;
pub use pyramid::{
    build_pyramid, coarsen_coords, downsample_block, plan_pyramid, PyramidLevel, PyramidOptions,
    PyramidPlanLevel,
};
pub use raycast::{raycast_terrain, TerrainHit};
pub use region::HeightRegion;
pub use residency::{tile_range, MemoryTileStore, ResidencyReport, TileStore};
pub use splat::{apply_paint, paint_weight, SplatDelta, SplatPatch, SplatStroke, SPLAT_LAYERS};
pub use stream::{
    open_file_tile_store, FileTileStore, PackTileStore, StreamBudget, TerrainStreamStats,
    TerrainStreamer,
};
pub use tile::{
    hole_mask_bytes, DataMapKind, TerrainTile, TerrainTileFrozenV1, TerrainTileFrozenV2,
    TerrainTileFrozenV3, TileKey, DATA_MAP_CHANNELS, DEFAULT_BIOME, DEFAULT_DATA_MAP,
    DEFAULT_WEIGHT, MAX_BIOMES, UNASSIGNED_BIOME,
};
pub use wants::{
    advance_cut, brush_wants, clamp_cut, render_wants, sim_wants, RenderWantsParams, TileCatalog,
    TileGrid, TileIndex, DEFAULT_HYSTERESIS,
};
pub use writeback::{rewrite_terrain_asset, TerrainEdits};

/// The minimal world-space height query seam downstream systems code against
/// (PCG scatter, sculpt brushes). Kept deliberately small: a paged heightfield,
/// a procedural noise field, or a flat plane can all satisfy it without exposing
/// their storage.
pub trait HeightSource {
    /// Absolute world height at world `(x, z)`, or `None` outside the source's
    /// authored/defined domain.
    fn height(&self, x: f64, z: f64) -> Option<f64>;

    /// Unit surface normal (+Y up) at world `(x, z)`, or `None` outside the
    /// domain.
    fn normal(&self, x: f64, z: f64) -> Option<DVec3>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec2, DVec3};

    /// A terrain whose two adjacent tiles are authored from one global height
    /// function `h(x,z) = sin(x*0.1) + cos(z*0.13)` — seamless by construction.
    fn sine_terrain(res: u32, mps: f64) -> TerrainData {
        let mut t = TerrainData::new(res, mps);
        let f = |x: f64, z: f64| (x * 0.1).sin() + (z * 0.13).cos();
        t.author_tile((0, 0), f);
        t.author_tile((1, 0), f);
        t
    }

    #[test]
    fn bilinear_exact_at_sample_points() {
        let res = 8;
        let mps = 2.0;
        let mut t = TerrainData::new(res, mps);
        // Deterministic per-sample values via the sample index.
        t.author_tile((0, 0), |x, z| x * 0.5 - z * 0.25 + 3.0);
        let o = t.tile_origin_xz((0, 0));
        for j in 0..res {
            for i in 0..res {
                let wx = o.x + i as f64 * mps;
                let wz = o.y + j as f64 * mps;
                let expect = wx * 0.5 - wz * 0.25 + 3.0;
                let got = t.height_at(DVec2::new(wx, wz)).unwrap();
                assert!(
                    (got - expect).abs() < 1e-4,
                    "sample ({i},{j}) got {got} expect {expect}"
                );
            }
        }
    }

    #[test]
    fn bilinear_midpoint_is_average_on_a_plane() {
        // On a linear height field, the bilinear value at a cell midpoint is the
        // exact plane value (linear interpolation reproduces linear data).
        let res = 5;
        let mps = 4.0;
        let mut t = TerrainData::new(res, mps);
        let f = |x: f64, z: f64| 2.0 * x + 3.0 * z + 1.0;
        t.author_tile((0, 0), f);
        // A point at the centre of cell (0,0)-(1,1).
        let o = t.tile_origin_xz((0, 0));
        let p = DVec2::new(o.x + mps * 0.5, o.y + mps * 0.5);
        let got = t.height_at(p).unwrap();
        assert!((got - f(p.x, p.y)).abs() < 1e-6, "midpoint {got}");
    }

    #[test]
    fn normal_on_known_slope() {
        // A plane tilted only along +X: h = 0.5*x. dh/dx = 0.5, dh/dz = 0.
        // Normal ∝ (-0.5, 1, 0), normalized.
        let mut t = TerrainData::new(6, 1.0);
        t.author_tile((0, 0), |x, _z| 0.5 * x);
        let o = t.tile_origin_xz((0, 0));
        let n = t.normal_at(DVec2::new(o.x + 2.0, o.y + 2.0)).unwrap();
        let expect = DVec3::new(-0.5, 1.0, 0.0).normalize();
        assert!((n - expect).length() < 1e-6, "normal {n:?}");
    }

    #[test]
    fn tile_boundary_is_continuous() {
        // Two tiles authored from one function share their edge exactly, so the
        // height is continuous across the boundary (no seam).
        let res = 8;
        let mps = 1.0;
        let t = sine_terrain(res, mps);
        let span = t.tile_span();
        let f = |x: f64, z: f64| (x * 0.1).sin() + (z * 0.13).cos();
        // Sweep across the tile-0 / tile-1 boundary (z within a tile's Z range,
        // [0, span)). The seam guarantee: the height just left of the edge equals
        // the height just right of it.
        for k in 0..((res - 1) * 2) {
            let z = k as f64 * 0.5;
            let just_left = t.height_at(DVec2::new(span - 1e-6, z)).unwrap();
            let just_right = t.height_at(DVec2::new(span + 1e-6, z)).unwrap();
            assert!(
                (just_left - just_right).abs() < 1e-3,
                "seam at z={z}: {just_left} vs {just_right}"
            );
        }
        // Exactly on a shared sample (integer z), both tiles reproduce the source
        // function exactly (no interpolation error to mask a seam).
        for j in 0..(res - 1) {
            let z = j as f64 * mps;
            let on_edge = t.height_at(DVec2::new(span, z)).unwrap();
            assert!(
                (on_edge - f(span, z)).abs() < 1e-4,
                "edge sample z={z}: {on_edge} vs {}",
                f(span, z)
            );
        }
    }

    #[test]
    fn queries_outside_authored_tiles_are_none() {
        let t = sine_terrain(8, 1.0);
        // Far away, no tile authored.
        assert!(t.height_at(DVec2::new(1e6, 1e6)).is_none());
        assert!(t.normal_at(DVec2::new(1e6, 1e6)).is_none());
    }

    #[test]
    fn write_region_authors_seamlessly_across_tiles() {
        let mut t = TerrainData::new(8, 1.0);
        let f = |x: f64, z: f64| (x + z) * 0.01;
        let span = t.tile_span();
        // Region spans both tile (0,0) and (1,0).
        t.write_region(DVec2::new(0.0, 0.0), DVec2::new(span * 2.0, span), f);
        assert!(t.has_tile((0, 0)) && t.has_tile((1, 0)));
        let mid = t.height_at(DVec2::new(span, span * 0.5)).unwrap();
        assert!((mid - f(span, span * 0.5)).abs() < 1e-3);
    }

    #[test]
    fn height_source_trait_matches_inherent_queries() {
        let t = sine_terrain(8, 1.0);
        let p = DVec2::new(3.5, 2.25);
        assert_eq!(HeightSource::height(&t, p.x, p.y), t.height_at(p));
        let n_trait = HeightSource::normal(&t, p.x, p.y).unwrap();
        let n_inh = t.normal_at(p).unwrap();
        assert!((n_trait - n_inh).length() < 1e-12);
    }

    #[test]
    fn serde_round_trip_is_deterministic() {
        let t = sine_terrain(8, 1.5);
        let a = serde_json::to_string(&t).unwrap();
        let back: TerrainData = serde_json::from_str(&a).unwrap();
        assert_eq!(t, back);
        // Re-serialization is byte-identical (BTreeMap order is stable).
        let b = serde_json::to_string(&back).unwrap();
        assert_eq!(a, b);
    }

    /// A **resident tile always holds a stamp** (P16.3): loading a terrain draws
    /// one per tile, so a GPU tile cache sees a stable, non-zero version and
    /// uploads each page once — instead of treating every tile as unstamped and
    /// re-uploading the whole heightfield every frame. Loading is not an edit, so
    /// nothing is marked dirty.
    #[test]
    fn deserialized_tiles_carry_change_stamps() {
        let t = sine_terrain(8, 1.5);
        let json = serde_json::to_string(&t).unwrap();
        let back: TerrainData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.version_ledger_len(), back.tile_count());
        for (&coord, _) in back.tiles() {
            assert!(
                back.tile_version(TileKey::lod0(coord)) > 0,
                "loaded tile {coord:?} must be stamped"
            );
        }
        assert!(
            !back.has_dirty_tiles(),
            "a fresh load schedules no write-back"
        );
        // Stamps are runtime state, not content: equality and the bytes are
        // unaffected.
        assert_eq!(t, back);
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
    }

    #[test]
    fn serde_rejects_wrong_length_tile() {
        // A tile whose height buffer doesn't match resolution² is refused on load.
        let bad = r#"{
            "tile_resolution": 4,
            "meters_per_sample": 1.0,
            "tiles": [[0, 0, {"origin":[0.0,0.0,0.0],"heights":[1.0,2.0,3.0]}]]
        }"#;
        assert!(serde_json::from_str::<TerrainData>(bad).is_err());
    }

    #[test]
    fn heightmap_png16_round_trips_exactly() {
        // Author a terrain, export to PNG16 over the full 16-bit range (so the
        // f32 offsets are exact integers), re-import, and compare pixels + heights.
        let res = 5;
        let import = HeightmapImport {
            tile_resolution: res,
            meters_per_sample: 1.0,
            min_height: 0.0,
            max_height: 65535.0,
            ..Default::default()
        };
        // A 9×9 grid = 2×2 tiles of resolution 5 (shared edges), exact fit.
        let mut src = TerrainData::new(res, 1.0);
        let f = |x: f64, z: f64| ((x * 900.0).rem_euclid(60000.0)).floor() + (z * 300.0).floor();
        src.author_tile((0, 0), f);
        src.author_tile((1, 0), f);
        src.author_tile((0, 1), f);
        src.author_tile((1, 1), f);

        let png = src.export_png16(0.0, 65535.0).unwrap();
        let back = TerrainData::from_height_image(&png, import).unwrap();

        // Re-export both and compare the 16-bit images byte-for-byte.
        let img_a = src.to_height_image(0.0, 65535.0).unwrap();
        let img_b = back.to_height_image(0.0, 65535.0).unwrap();
        assert_eq!(img_a.width, img_b.width);
        assert_eq!(img_a.height, img_b.height);
        assert_eq!(img_a.samples, img_b.samples);
    }

    #[test]
    fn import_maps_min_max_range() {
        // A 2×2 image with corner values 0 and 65535 maps to min/max heights.
        let img = HeightImage {
            width: 2,
            height: 2,
            samples: vec![0, 65535, 32768, 16384],
        };
        let png = encode_png16(&img).unwrap();
        let import = HeightmapImport {
            tile_resolution: 2,
            meters_per_sample: 10.0,
            min_height: -100.0,
            max_height: 100.0,
            ..Default::default()
        };
        let t = TerrainData::from_height_image(&png, import).unwrap();
        // Sample (0,0) = 0 → min; (1,0) = 65535 → max.
        assert!((t.height_at(DVec2::new(0.0, 0.0)).unwrap() - -100.0).abs() < 1e-2);
        assert!((t.height_at(DVec2::new(10.0, 0.0)).unwrap() - 100.0).abs() < 1e-2);
    }
}
