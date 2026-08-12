//! **The VT mirror's readback arms** (P26.2): a transaction applied to a real
//! atlas and a real indirection buffer, and both read back byte for byte.
//!
//! The residency decides *what*; these arms check that the *where* survives the
//! trip to the GPU. Three things can go wrong between an `inf-vt` transaction and
//! a texel, and none of them is visible without a readback:
//!
//! * the slot origin (a transposed `slot % slots_x` puts every page in the wrong
//!   column and every sample is off by a tile);
//! * the row pitch (272 B is not 256-aligned, and getting this wrong shears a
//!   page rather than losing it);
//! * the table bytes (a block written at the wrong word offset points a whole
//!   texture at another texture's pages).
//!
//! # Two adapter tiers, one door (the P25.2 doctrine)
//!
//! `TEXTURE_COMPRESSION_BC` is optional and probed, so a runner without it must
//! still exercise the path rather than skip it — that is precisely the adapter
//! class the transcode fallback exists for, and a fallback nobody runs is a
//! fallback nobody knows about. The pool's format comes from
//! `inf_render::vt::pool_format` against the **clamped** settings, exactly as a
//! host computes it, and the fetch closure switches between
//! `TiledTextureReader::tile` and `::tile_rgba8` accordingly — the same door, one
//! format decision earlier. Both tiers assert exact byte equality, which is legal
//! on any adapter: a texture→buffer copy is a copy, not a filter.
//!
//! Skips cleanly, and says so, when the machine has no adapter at all.

use std::borrow::Cow;
use std::collections::BTreeSet;

use inf_material::tiles::TiledTextureImage;
use inf_material::{TextureCompression, TextureImportSettings};
use inf_render::vt::{pool_format, VtPools};
use inf_render::{GpuContext, RenderSettings};
use inf_vt::{PageFormat, TileCoord, VtAdmit, VtPoolConfig, VtResidency, VtWant};

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available for {what} ({e})");
            None
        }
    }
}

/// Four distinct corner quadrants plus a prime-period diagonal ramp, so a
/// transposed slot origin, a mirrored row order or a sheared row pitch cannot
/// pass by symmetry — and so no two tiles carry the same bytes.
fn corners(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let base: [u8; 3] = match (x * 2 < w, y * 2 < h) {
                (true, true) => [220, 20, 20],
                (false, true) => [20, 200, 20],
                (true, false) => [20, 20, 210],
                (false, false) => [230, 210, 30],
            };
            let g = ((x * 3 + y * 5) % 61) as u8;
            v.extend_from_slice(&[
                base[0].saturating_sub(g),
                base[1].saturating_sub(g),
                base[2].saturating_sub(g),
                255,
            ]);
        }
    }
    v
}

/// A real `.inf_tex` v2 container — the bytes a cook writes, not a hand-built
/// block soup that could agree with nothing.
fn container(w: u32, h: u32) -> TiledTextureImage {
    inf_material::build_tiled_texture(
        corners(w, h),
        w,
        h,
        TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression: TextureCompression::Bc1,
        },
    )
    .expect("the fixture tiles")
}

/// Read one atlas slot back, tightly packed.
fn read_slot(gpu: &GpuContext, pools: &VtPools, slot: u32) -> Vec<u8> {
    let g = pools.geometry();
    let (ox, oy) = g.slot_origin(slot).expect("slot in range");
    let format = pools.format();
    let (bw, bh) = format.block_dimensions();
    let block = format.block_copy_size(None).expect("colour format");
    let side = g.stored_tile_size;
    let blocks_x = side.div_ceil(bw);
    let blocks_y = side.div_ceil(bh);
    let tight = (blocks_x * block) as usize;
    // THIS copy is the one the 256-byte rule binds — `copy_texture_to_buffer`
    // takes a `GPUBuffer`, unlike `queue.write_texture`.
    let padded = tight.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vt-slot-readback"),
        size: (padded * blocks_y as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vt-slot-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: pools.atlas(),
            mip_level: 0,
            origin: wgpu::Origin3d { x: ox, y: oy, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(blocks_y),
            },
        },
        wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));
    map_and_take(gpu, &readback, |mapped| {
        let mut out = Vec::with_capacity(tight * blocks_y as usize);
        for row in mapped.chunks(padded) {
            out.extend_from_slice(&row[..tight]);
        }
        out
    })
}

/// Read the whole indirection buffer back as words.
fn read_table(gpu: &GpuContext, pools: &VtPools) -> Vec<u32> {
    let size = pools.table().size();
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vt-table-readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vt-table-readback"),
        });
    encoder.copy_buffer_to_buffer(pools.table(), 0, &readback, 0, size);
    gpu.queue.submit(Some(encoder.finish()));
    map_and_take(gpu, &readback, |mapped| {
        mapped
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    })
}

fn map_and_take<T>(gpu: &GpuContext, buffer: &wgpu::Buffer, f: impl FnOnce(&[u8]) -> T) -> T {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map_async dropped").expect("map_async");
    let mapped = slice.get_mapped_range().expect("get_mapped_range");
    let out = f(&mapped);
    drop(mapped);
    buffer.unmap();
    out
}

/// The residency + pools a host would build for one container, with the format
/// decided the way a host decides it.
struct Harness {
    residency: VtResidency,
    pools: VtPools,
    handle: inf_vt::VtTextureHandle,
    bc: bool,
}

fn harness(gpu: &GpuContext, image: &TiledTextureImage, pages: u64) -> Harness {
    harness_as(gpu, image, pages, false)
}

/// `force_rgba8` is the BC-less adapter, simulated on an adapter that has BC.
///
/// **The two-tier doctrine needs this door.** Deciding the tier from the adapter
/// alone means the transcode arm runs only on a machine without
/// `TEXTURE_COMPRESSION_BC` — which is no development machine and no GPU CI
/// runner we have — so the 8×-bytes fallback would ship untested everywhere it is
/// ever looked at. `bc_tiles` is a *setting* that the adapter clamps, so a caller
/// may also just turn it off: that is the same door one decision earlier, which
/// is the whole premise, and it makes the fallback tier reachable on every leg.
fn harness_as(
    gpu: &GpuContext,
    image: &TiledTextureImage,
    pages: u64,
    force_rgba8: bool,
) -> Harness {
    let reader = image.reader();
    let mut settings = inf_render::caps::detect_and_clamp(gpu, RenderSettings::default());
    if force_rgba8 {
        settings.vt.bc_tiles = false;
    }
    let stored: PageFormat = reader.header().format;
    let format = pool_format(&settings, stored);
    let bc = format.needs_bc();
    assert_eq!(
        bc,
        gpu.supports_texture_compression_bc() && !force_rgba8,
        "the pool format must follow the adapter through the clamp, not around it"
    );
    let desc = reader.vt_desc();
    let stored_side = desc.stored_tile_size();
    let (mut residency, advisories) = VtResidency::new(VtPoolConfig {
        format,
        stored_tile_size: stored_side,
        budget_bytes: format.page_bytes(stored_side) * pages,
        max_texture_dim: gpu.device.limits().max_texture_dimension_2d,
    });
    assert!(advisories.is_empty(), "{advisories:?}");
    let handle = residency.register_texture(desc).expect("the floor fits");
    let pools = VtPools::new(&gpu.device, &gpu.queue, &residency, reader.header().srgb);
    Harness {
        residency,
        pools,
        handle,
        bc,
    }
}

/// Apply one transaction, paging out of the container through whichever door the
/// pool's format calls for.
fn apply(
    gpu: &GpuContext,
    h: &mut Harness,
    image: &TiledTextureImage,
    wants: &[VtWant],
) -> (inf_vt::VtTransaction, inf_render::vt::VtApplyReport) {
    apply_multi(gpu, h, &[image], wants)
}

/// The same, with one container per registered texture — so an admit's bytes come
/// from the texture it names rather than from whichever container the arm happened
/// to have in hand.
fn apply_multi(
    gpu: &GpuContext,
    h: &mut Harness,
    images: &[&TiledTextureImage],
    wants: &[VtWant],
) -> (inf_vt::VtTransaction, inf_render::vt::VtApplyReport) {
    let readers: Vec<_> = images.iter().map(|i| i.reader()).collect();
    let txn = h.residency.apply_wants(wants);
    let bc = h.bc;
    let report = h.pools.apply(
        &gpu.device,
        &gpu.queue,
        &h.residency,
        &txn,
        |a: &VtAdmit| -> Option<Cow<'_, [u8]>> {
            let r = readers.get(a.texture.index())?;
            if bc {
                r.tile_at(a.tile).map(Cow::Borrowed)
            } else {
                r.tile_rgba8(a.tile.mip, a.tile.x, a.tile.y).map(Cow::Owned)
            }
        },
    );
    assert!(report.missing.is_empty(), "{:?}", report.missing);
    (txn, report)
}

/// What the caller would have handed the mirror for `admit` — the arm's own copy
/// of the fetch, so a readback compares against the container rather than against
/// the closure that filled the atlas.
fn expected_page(h: &Harness, image: &TiledTextureImage, admit: &VtAdmit) -> Vec<u8> {
    let reader = image.reader();
    if h.bc {
        reader.tile_at(admit.tile).expect("tile").to_vec()
    } else {
        reader
            .tile_rgba8(admit.tile.mip, admit.tile.x, admit.tile.y)
            .expect("tile")
    }
}

/// **The page round trip**, into a slot that is not slot 0 and not in the atlas's
/// first row — the two addresses a transposed or truncated origin cannot reach.
#[test]
fn an_admitted_page_lands_in_its_slot_byte_for_byte() {
    let Some(gpu) = gpu_or_skip("the VT page round trip") else {
        return;
    };
    let image = container(320, 192);
    // Six slots: the root plus mip 0's 3×2 grid, laid out 2×3, so the last admit
    // is at slot 5 = column 1, row 2.
    let mut h = harness(&gpu, &image, 6);
    assert_eq!(
        (h.pools.geometry().slots_x, h.pools.geometry().slots_y),
        (2, 3)
    );
    apply(&gpu, &mut h, &image, &[]);

    let tiles: Vec<TileCoord> = (0..2)
        .flat_map(|y| (0..3).map(move |x| TileCoord::new(0, x, y)))
        .collect();
    let wants: Vec<VtWant> = tiles.iter().map(|t| VtWant::new(h.handle, *t)).collect();
    let (txn, report) = apply(&gpu, &mut h, &image, &wants);
    assert_eq!(txn.admits.len(), 5, "five free slots after the root");
    assert_eq!(report.pages, 5);

    let mut checked = 0;
    for admit in &txn.admits {
        let expect = expected_page(&h, &image, admit);
        // Anti-vacuity: a uniform page survives any round trip, including one
        // that uploaded nothing. Counted in 16-byte chunks because a BC page is
        // structured — hundreds of distinct BLOCKS out of a dozen byte values.
        let distinct: BTreeSet<&[u8]> = expect.chunks(16).collect();
        assert!(
            distinct.len() > 64,
            "slot {} got a nearly uniform page ({} distinct chunks)",
            admit.slot,
            distinct.len()
        );
        let back = read_slot(&gpu, &h.pools, admit.slot);
        assert!(
            back == expect,
            "slot {} (origin {:?}) came back changed: {} of {} bytes differ",
            admit.slot,
            h.pools.geometry().slot_origin(admit.slot),
            back.iter().zip(&expect).filter(|(a, b)| a != b).count(),
            expect.len()
        );
        checked += 1;
    }
    assert_eq!(checked, 5);

    // …and the pages are genuinely different from each other, so "every slot
    // holds the same correct bytes" cannot be what just passed.
    let a = read_slot(&gpu, &h.pools, txn.admits[0].slot);
    let b = read_slot(&gpu, &h.pools, txn.admits[4].slot);
    assert_ne!(a, b, "two different tiles read back identical");
    eprintln!(
        "P26.2 atlas round trip: bc={} format={:?} {}×{} slots, {} B/page",
        h.bc,
        h.pools.format(),
        h.pools.geometry().slots_x,
        h.pools.geometry().slots_y,
        h.residency.page_bytes()
    );
}

/// **The transcode tier, on an adapter that does not need it.**
///
/// The BC arm above runs whenever the machine has `TEXTURE_COMPRESSION_BC`, which
/// is every machine we build on — so left to the adapter, the RGBA8 fallback runs
/// nowhere and the two-tier doctrine is a sentence rather than a test. Turning
/// `bc_tiles` off is the same door the clamp uses, so this is the fallback exactly
/// as a BC-less adapter reaches it: `tile_rgba8` instead of `tile`, 8× the page
/// bytes for BC1, a 544 B row pitch instead of 272 B — and neither is a multiple
/// of 256, which is the point of `write_texture`.
#[test]
fn the_rgba8_fallback_pages_the_same_tiles_at_eight_times_the_bytes() {
    let Some(gpu) = gpu_or_skip("the VT transcode fallback") else {
        return;
    };
    let image = container(320, 192);
    let bc_pool = harness(&gpu, &image, 6);
    let mut h = harness_as(&gpu, &image, 6, true);
    assert!(!h.bc, "the forced pool is the fallback tier");
    assert_eq!(h.pools.format(), wgpu::TextureFormat::Rgba8UnormSrgb);
    assert_eq!(
        h.residency.page_bytes(),
        bc_pool.residency.page_bytes() * 8,
        "the measured cost of the fallback (P26.1: 8× for BC1, not 4×)"
    );
    assert_eq!(h.residency.page_bytes(), 136 * 136 * 4);

    apply(&gpu, &mut h, &image, &[]);
    let tiles: Vec<TileCoord> = (0..2)
        .flat_map(|y| (0..3).map(move |x| TileCoord::new(0, x, y)))
        .collect();
    let wants: Vec<VtWant> = tiles.iter().map(|t| VtWant::new(h.handle, *t)).collect();
    let (txn, report) = apply(&gpu, &mut h, &image, &wants);
    assert_eq!(txn.admits.len(), 5);
    assert_eq!(report.page_bytes, 5 * 73_984);

    for admit in &txn.admits {
        let expect = expected_page(&h, &image, admit);
        assert_eq!(expect.len(), 73_984, "a transcoded page is RGBA8 136²");
        // Anti-vacuity: a decoded page that came back all-zero, or uniform, would
        // survive a round trip that uploaded nothing.
        // Anti-vacuity, in texels rather than blocks: the fixture's ramp has a
        // period of 61, so a tile inside one colour quadrant carries about that
        // many distinct texels and no more.
        let distinct: BTreeSet<&[u8]> = expect.chunks(4).collect();
        assert!(distinct.len() > 32, "slot {} is nearly uniform", admit.slot);
        assert!(
            read_slot(&gpu, &h.pools, admit.slot) == expect,
            "slot {} (origin {:?}) came back changed",
            admit.slot,
            h.pools.geometry().slot_origin(admit.slot)
        );
    }
    // The table is the same table: the format decision is one layer below it.
    assert_eq!(read_table(&gpu, &h.pools), h.residency.table_words());
}

/// **The indirection round trip**: after a burst of transactions the buffer holds
/// exactly the CPU's table, word for word — including the blocks that were
/// patched rather than rewritten.
#[test]
fn the_indirection_buffer_matches_the_cpu_table_after_a_burst() {
    let Some(gpu) = gpu_or_skip("the VT indirection round trip") else {
        return;
    };
    let image = container(320, 192);
    let mut h = harness(&gpu, &image, 4);
    apply(&gpu, &mut h, &image, &[]);
    assert_eq!(
        read_table(&gpu, &h.pools),
        h.residency.table_words(),
        "the layout write did not land"
    );

    // A burst that thrashes three cache slots across six tiles, so most rounds
    // patch a block rather than rewriting the layout.
    let tiles: Vec<TileCoord> = (0..2)
        .flat_map(|y| (0..3).map(move |x| TileCoord::new(0, x, y)))
        .collect();
    let mut patched = 0;
    for round in 0..12usize {
        let wants: Vec<VtWant> = (0..2)
            .map(|k| VtWant::new(h.handle, tiles[(round * 2 + k) % tiles.len()]))
            .collect();
        let (txn, report) = apply(&gpu, &mut h, &image, &wants);
        assert!(!report.table_recreated, "no texture was registered");
        patched += usize::from(!txn.tables.is_empty());
        assert_eq!(
            read_table(&gpu, &h.pools),
            h.residency.table_words(),
            "round {round}: the GPU table drifted from the CPU's"
        );
    }
    assert!(
        patched >= 8,
        "only {patched} of 12 rounds patched a block — the burst never moved the table"
    );

    // Anti-vacuity: the table is not a field of zeros, and its entries genuinely
    // differ (a table that pointed everything at slot 0 would match a CPU table
    // that did the same, and both would be wrong).
    let words = read_table(&gpu, &h.pools);
    assert_eq!(words[0], inf_vt::TABLE_MAGIC);
    let desc = h.residency.desc(h.handle).expect("registered").clone();
    let entries: BTreeSet<u32> = (0..desc.mips[0].tiles_y)
        .flat_map(|y| (0..desc.mips[0].tiles_x).map(move |x| TileCoord::new(0, x, y)))
        .map(|at| words[entry_word_index(&h.residency, h.handle, at)])
        .collect();
    assert!(
        entries.len() > 1,
        "every mip-0 entry resolves to the same page: {entries:?}"
    );
    // And every entry the GPU holds — every tile of every level, not mip 0 —
    // decodes to what the CPU resolves.
    let checked = assert_gpu_table_resolves_like_the_cpu(&words, &h.residency);
    assert_eq!(checked, desc.tile_count() as usize, "the whole pyramid");
}

/// The word index of `at`'s entry, computed the way the shader's arithmetic will:
/// block base, past the header and the mip records, plus the tile's directory
/// index. (`entry_index` is the same number the `.inf_tex` tile directory uses.)
fn entry_word_index(
    residency: &VtResidency,
    handle: inf_vt::VtTextureHandle,
    at: TileCoord,
) -> usize {
    let desc = residency.desc(handle).expect("registered");
    let (base, block) = residency.table_block(handle).expect("registered");
    let w = base
        + inf_vt::TABLE_TEXTURE_HEADER_WORDS
        + inf_vt::TABLE_MIP_REC_WORDS * desc.mips.len()
        + desc.entry_index(at).expect("in grid") as usize;
    assert!(w < base + block.len(), "{at:?} is outside its own block");
    w
}

/// Every entry the GPU holds, against `inf_vt::resolved_table` — the residency's
/// own answer for **every tile of every level of every texture**. Returns how many
/// were compared, so a caller can refuse a vacuous zero.
fn assert_gpu_table_resolves_like_the_cpu(words: &[u32], residency: &VtResidency) -> usize {
    let resolved = inf_vt::resolved_table(residency);
    for (&(tex, at), r) in &resolved {
        let handle = inf_vt::VtTextureHandle(tex);
        let e = inf_vt::unpack_entry(words[entry_word_index(residency, handle, at)]);
        assert_eq!(
            (e.slot, e.mip),
            (r.slot, r.tile.mip),
            "texture {tex} {at:?}: the GPU entry and the CPU resolve disagree"
        );
    }
    resolved.len()
}

/// A page the caller cannot produce becomes a **defined** page, not a stale one.
///
/// The slot already has an indirection entry pointing at it by the time `apply`
/// runs, so "leave it alone" means sampling whatever the previous occupant left
/// — another texture's texels, or another mip's. Zero is reported and readable.
#[test]
fn a_page_that_will_not_load_is_zeroed_and_named() {
    let Some(gpu) = gpu_or_skip("the missing-page fill") else {
        return;
    };
    let image = container(320, 192);
    let mut h = harness(&gpu, &image, 4);
    apply(&gpu, &mut h, &image, &[]);

    // Seat a real page first, so the slot holds something recognisable.
    let handle = h.handle;
    let a = TileCoord::new(0, 0, 0);
    let (txn, _) = apply(&gpu, &mut h, &image, &[VtWant::new(handle, a)]);
    let slot = txn.admits[0].slot;
    assert!(read_slot(&gpu, &h.pools, slot).iter().any(|&b| b != 0));

    // Now churn `a` out and admit a tile whose bytes the caller "loses".
    let b = TileCoord::new(0, 1, 0);
    let c = TileCoord::new(0, 2, 0);
    let d = TileCoord::new(0, 0, 1);
    apply(
        &gpu,
        &mut h,
        &image,
        &[VtWant::new(handle, b), VtWant::new(handle, c)],
    );
    let txn = h.residency.apply_wants(&[VtWant::new(handle, d)]);
    assert_eq!(txn.admits.len(), 1);
    let reused = txn.admits[0].slot;
    let report = h.pools.apply(
        &gpu.device,
        &gpu.queue,
        &h.residency,
        &txn,
        |_: &VtAdmit| -> Option<Cow<'_, [u8]>> { None },
    );
    assert_eq!(report.missing, txn.admits, "the loss must be named");
    assert_eq!(report.pages, 1, "a zero page is still a page written");
    let back = read_slot(&gpu, &h.pools, reused);
    assert!(
        back.iter().all(|&x| x == 0),
        "a missing page left {} non-zero bytes of the previous occupant behind",
        back.iter().filter(|&&x| x != 0).count()
    );
}

/// Registering a second texture moves every block after it, so the buffer is
/// re-created and written whole — and the mirror must notice without being told
/// twice.
#[test]
fn a_registration_re_creates_the_indirection_buffer() {
    let Some(gpu) = gpu_or_skip("the VT layout rebuild") else {
        return;
    };
    let image = container(320, 192);
    let second = container(256, 256);
    let mut h = harness(&gpu, &image, 6);
    apply(&gpu, &mut h, &image, &[]);
    let before = h.pools.table().size();

    let other = h
        .residency
        .register_texture(second.reader().vt_desc())
        .expect("the floor fits");
    let (txn, report) = apply_multi(&gpu, &mut h, &[&image, &second], &[]);
    assert!(txn.layout_rebuilt);
    assert!(report.table_recreated);
    assert!(
        h.pools.table().size() > before,
        "the buffer did not grow for a second texture"
    );
    assert_eq!(read_table(&gpu, &h.pools), h.residency.table_words());
    // The first texture's block moved, and its entries moved with it.
    let (base, _) = h.residency.table_block(other).expect("registered");
    assert!(base > 0);
    assert!(h
        .residency
        .resolve(other, TileCoord::new(0, 1, 1))
        .is_some());

    // **And then the blocks are PATCHED, at a base that is no longer zero.**
    //
    // Every write after a rebuild goes to `base * 4` — a word offset scaled into
    // bytes — and with one texture `base` is 5 and the whole class of byte-vs-word
    // mistakes is either a `wgpu` alignment refusal or invisible. With two
    // textures the second block sits hundreds of words in, so writing it at the
    // wrong offset lands inside the FIRST texture's entries and points a whole
    // texture at another texture's pages. That is the readback this arm exists
    // for, and it needs a burst that touches both.
    // Two pinned roots leave four cache slots against six wants a round, so the
    // set never settles and both blocks keep moving.
    let mut patched_second = 0;
    for round in 0..10usize {
        let r = round as u32;
        let wants = vec![
            VtWant::new(h.handle, TileCoord::new(0, r % 3, 0)),
            VtWant::new(h.handle, TileCoord::new(0, (r + 1) % 3, 1)),
            VtWant::new(h.handle, TileCoord::new(0, (r + 2) % 3, 0)),
            VtWant::new(other, TileCoord::new(0, r % 2, 0)),
            VtWant::new(other, TileCoord::new(0, (r + 1) % 2, 1)),
            VtWant::new(other, TileCoord::new(1, 0, 0)),
        ];
        let (txn, report) = apply_multi(&gpu, &mut h, &[&image, &second], &wants);
        assert!(!report.table_recreated, "round {round}: nothing registered");
        patched_second += usize::from(txn.tables.contains(&other));
        let words = read_table(&gpu, &h.pools);
        assert_eq!(
            words,
            h.residency.table_words(),
            "round {round}: the GPU table drifted from the CPU's"
        );
        assert!(assert_gpu_table_resolves_like_the_cpu(&words, &h.residency) > 0);
    }
    assert!(
        patched_second >= 4,
        "only {patched_second} of 10 rounds patched the SECOND texture's block — \
         the burst never wrote at a non-trivial base"
    );

    // The pages behind both textures are the pages their own containers hold —
    // with two readers in play, a fetch that ignored `admit.texture` would fill
    // one texture's slots from the other's file and every assertion above would
    // still pass, since a table says nothing about texels.
    let settle: Vec<VtWant> = [h.handle, other]
        .iter()
        .map(|t| VtWant::new(*t, TileCoord::new(0, 0, 1)))
        .collect();
    apply_multi(&gpu, &mut h, &[&image, &second], &settle);
    for (handle, img) in [(h.handle, &image), (other, &second)] {
        let at = TileCoord::new(0, 0, 1);
        let r = h.residency.resolve(handle, at).expect("resolves");
        assert_eq!(r.tile, at, "texture {handle:?} did not settle on {at:?}");
        let admit = VtAdmit {
            slot: r.slot,
            texture: handle,
            tile: at,
        };
        assert_eq!(
            read_slot(&gpu, &h.pools, r.slot),
            expected_page(&h, img, &admit),
            "texture {handle:?} slot {} holds another container's bytes",
            r.slot
        );
    }
}

/// The three bindings P26.3 folds into the environment group **agree with each
/// other**: `wgpu` validates a bind group against its layout, so creating one
/// from the pair is the only check that a swapped sampler and buffer, or a
/// mismatched `base`, cannot pass.
///
/// Nothing binds VT yet, which is exactly why this is worth a test now: the pair
/// is the surface P26.3 builds on, and a mismatch would otherwise first be seen
/// as a validation error inside a much larger change.
#[test]
fn the_bind_group_layout_and_its_entries_are_the_same_three_bindings() {
    let Some(gpu) = gpu_or_skip("the VT bind-group pair") else {
        return;
    };
    let image = container(320, 192);
    let h = harness(&gpu, &image, 4);
    let entries = VtPools::bind_group_layout_entries(9);
    assert_eq!(
        entries.len() as u32,
        inf_render::vt::VT_BINDING_COUNT,
        "the constant P26.3 budgets against is not the number of entries"
    );
    assert_eq!(
        entries.iter().map(|e| e.binding).collect::<Vec<_>>(),
        vec![9, 10, 11],
        "the entries must be consecutive from `base`"
    );
    let layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vt-test-layout"),
            entries: &entries,
        });
    let _ = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vt-test-group"),
        layout: &layout,
        entries: &h.pools.bind_group_entries(9),
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
}
