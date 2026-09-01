//! `.inf_terrain` through a **real cooked pack** (P16.3).
//!
//! The unit tests in `asset.rs` prove the payload layout in isolation. This is the
//! end-to-end claim the layout exists for: build a terrain asset, cook it into a
//! `.ipack` on disk, `mmap` that pack, take the **borrowed** `read_ref` slice,
//! and page individual tiles out of the mapping — no decode of the pack entry, no
//! decode of the other tiles, and every tile blob 16-byte aligned *by address*.

use std::collections::BTreeSet;

use inf_asset::{AssetId, AssetKind, PackReader, PackWriter};
use inf_terrain::{
    build_pyramid, build_terrain_asset, PyramidOptions, TerrainAssetReader, TerrainData, TileKey,
    TILE_ALIGN,
};

/// A bit-portable height field — polynomial only, never `std` trig (the P14 law).
fn height_fn(x: f64, z: f64) -> f64 {
    x * 0.25 - z * 0.125 + x * z * 0.001 + 5.0
}

fn terrain(n: i32) -> TerrainData {
    let mut t = TerrainData::new(9, 2.0);
    for tz in 0..n {
        for tx in 0..n {
            t.author_tile((tx, tz), height_fn);
        }
    }
    // One painted tile, so the pack carries both weight forms.
    t.get_tile_mut((1, 1))
        .unwrap()
        .set_weight_sample(9, 3, 4, [10, 20, 30, 195]);
    t
}

#[test]
fn terrain_asset_pages_out_of_a_cooked_pack_without_decoding_it() {
    let src = terrain(4);
    let pyramid = build_pyramid(&src, PyramidOptions::default());
    let asset = build_terrain_asset(&src, &pyramid, PyramidOptions::default()).unwrap();
    let guid = AssetId(uuid::Uuid::from_u128(0x1603_0001));

    // Cook: the payload bytes go into the pack verbatim (streaming-class kinds are
    // stored uncompressed, which is what makes the borrowed read possible).
    let dir = tempfile::tempdir().unwrap();
    let pack_path = dir.path().join("content.ipack");
    let mut w = PackWriter::new();
    w.add_bytes(guid, AssetKind::Terrain, asset.as_bytes())
        .unwrap();
    w.write_to_file(&pack_path).unwrap();

    let reader = PackReader::open(&pack_path).unwrap();
    let entry = reader.entry(guid).unwrap();
    assert!(!entry.compressed, "terrain must cook uncompressed");
    assert_eq!(entry.stored_len, asset.as_bytes().len() as u64);

    let payload = reader.read_ref(guid).unwrap();
    assert!(
        matches!(payload, std::borrow::Cow::Borrowed(_)),
        "the pack entry is borrowed, not decoded"
    );
    assert_eq!(&*payload, asset.as_bytes(), "pack bytes == payload bytes");

    // Slice tiles straight out of the mapping.
    let view = TerrainAssetReader::new(&*payload).unwrap();
    assert_eq!(view.tile_resolution(), 9);
    assert_eq!(view.lod_levels(), 1 + pyramid.len() as u32);

    for (&coord, tile) in src.tiles() {
        let key = TileKey::lod0(coord);
        let blob = view.tile_bytes(key).expect("tile in the mapping");
        // Aligned by ADDRESS, not just by offset: the pack's base is 16-byte
        // aligned and v2 blob offsets are too, so a sub-slice of a sub-slice keeps
        // the promise a GPU upload / `[f32]` view relies on.
        assert_eq!(
            blob.as_ptr() as usize % TILE_ALIGN as usize,
            0,
            "tile {coord:?} blob address is not {TILE_ALIGN}-byte aligned"
        );
        assert_eq!(&view.tile(key).unwrap().unwrap(), tile);
    }
    for level in &pyramid {
        for (&coord, tile) in &level.tiles {
            let key = TileKey::new(level.lod, coord);
            assert_eq!(&view.tile(key).unwrap().unwrap(), tile, "lod {}", level.lod);
        }
    }

    // Reading one tile never hashed or decoded anything beyond the single
    // verify-once integrity pass over the entry.
    assert_eq!(reader.verify_count(), 1);
}

#[test]
fn residency_streams_a_window_from_the_packed_asset() {
    let src = terrain(4);
    let pyramid = build_pyramid(&src, PyramidOptions::default());
    let asset = build_terrain_asset(&src, &pyramid, PyramidOptions::default()).unwrap();
    let guid = AssetId(uuid::Uuid::from_u128(0x1603_0002));

    let dir = tempfile::tempdir().unwrap();
    let pack_path = dir.path().join("content.ipack");
    let mut w = PackWriter::new();
    w.add_bytes(guid, AssetKind::Terrain, asset.as_bytes())
        .unwrap();
    w.write_to_file(&pack_path).unwrap();
    let pack = PackReader::open(&pack_path).unwrap();
    let payload = pack.read_ref(guid).unwrap();
    let store = TerrainAssetReader::new(&*payload).unwrap();

    // A caller-computed window (no camera in Ring 0) pages in, then slides.
    let mut live = TerrainData::new(9, 2.0);
    let wants: BTreeSet<TileKey> = inf_terrain::tile_range(0, (0, 0), (1, 1));
    let report = live.sync_residency(&wants, &store);
    assert_eq!(report.loaded.len(), 4);
    assert_eq!(live.tile_count(), 4);
    let first_stamp = live.tile_version(TileKey::lod0((0, 0)));
    assert!(first_stamp > 0, "a streamed-in tile carries a stamp");
    assert!(live.height_at(glam::DVec2::new(1.0, 1.0)).is_some());
    assert!(
        live.height_at(glam::DVec2::new(50.0, 50.0)).is_none(),
        "a non-resident tile answers exactly like an unauthored one"
    );

    let next: BTreeSet<TileKey> = inf_terrain::tile_range(0, (2, 2), (3, 3));
    let report = live.sync_residency(&next, &store);
    assert_eq!(report.loaded.len(), 4);
    assert_eq!(report.evicted.len(), 4);
    assert_eq!(live.tile_count(), 4);
    assert!(report.retained_dirty.is_empty(), "streaming is not editing");
    assert!(!live.has_dirty_tiles());

    // Every streamed tile is bit-identical to the authored one.
    for tz in 2..4 {
        for tx in 2..4 {
            assert_eq!(live.get_tile((tx, tz)), src.get_tile((tx, tz)));
        }
    }

    // Stamps are strictly increasing across the evict → reload cycle, and the
    // ledger stays bounded by residency rather than by everything ever streamed.
    let key = TileKey::lod0((0, 0));
    assert_eq!(live.tile_version(key), 0, "evicted ⇒ stamp pruned");
    assert_eq!(live.version_ledger_len(), 4, "one stamp per resident tile");
    live.sync_residency(&wants, &store);
    assert!(
        live.tile_version(key) > 0,
        "a re-paged tile takes a fresh stamp"
    );
    assert!(
        live.tile_version(key) > first_stamp,
        "and it is strictly greater than any stamp it ever held"
    );

    // And the asset itself round-trips to an equal authored terrain.
    assert_eq!(store.to_terrain_data().unwrap(), src);
}
