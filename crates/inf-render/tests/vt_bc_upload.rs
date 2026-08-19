//! **The BC upload proof** (P26.1): one tile out of a real `.inf_tex` v2
//! container, into a GPU texture, and back — byte for byte.
//!
//! # What this proves, and what it deliberately does not
//!
//! The whole premise of the tiled container is that a resident page is a
//! *memcpy*: the cooked bytes are already GPU-native BC blocks, so the format,
//! the row pitch and the 4×4 block granularity have to line up with what wgpu
//! expects with nothing in between. That is a claim about plumbing, and it is
//! exactly what this asserts.
//!
//! It is **not** a render: nothing is sampled, nothing is bound, no pass runs.
//! Virtual sampling arrives in P26.3 and the residency door that decides *which*
//! tile to upload arrives in P26.2. This test exists so that when those land,
//! the format/pitch/alignment layer underneath them is already known-good.
//!
//! # Two adapters, one call shape (the two-tier doctrine)
//!
//! `TEXTURE_COMPRESSION_BC` is optional and probed
//! ([`GpuContext::supports_texture_compression_bc`]), so a runner without it
//! must still exercise the path rather than skip it — that is precisely the
//! adapter class the CPU-transcode fallback exists for, and a fallback nobody
//! runs is a fallback nobody knows about. So both tiers go through the *same*
//! [`upload_and_read_back`], which derives its block dimensions and row pitch
//! from `wgpu::TextureFormat` itself:
//!
//! * BC present → the stored tile blob uploads as `Bc1RgbaUnorm` / `Bc3RgbaUnorm`;
//! * BC absent  → `TiledTextureReader::tile_rgba8` transcodes the same tile and
//!   it uploads as `Rgba8Unorm`.
//!
//! Both assert **exact** byte equality, which is legitimate on any adapter: a
//! texture→buffer copy is a copy, not a filter or a conversion. (The one
//! platform-dependent thing here would be the BC *encoder*, and it is not on
//! this path at all — `inf_material::bc` is hand-rolled integer arithmetic with
//! no floating point anywhere, so its output is the same bytes on every target
//! that runs it.)
//!
//! Skips cleanly, and says so, when the machine has no adapter at all.

use inf_material::tiles::{TiledTextureImage, STORED_TILE_SIZE};
use inf_material::{TextureCompression, TextureImportSettings};
use inf_render::{caps::AdapterCaps, GpuContext};

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available for {what} ({e})");
            None
        }
    }
}

/// The fixture: four distinct corner quadrants plus a per-texel gradient, so a
/// transposed tile lookup, a mirrored row order or a sheared row pitch cannot
/// pass by symmetry. 320×192 gives mip 0 a 3×2 tile grid, which is what makes
/// "tile (2, 1)" a meaningful address rather than the only one.
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
            // A prime-period diagonal ramp, not an 8-texel repeat: BC blocks
            // that repeat produce a page with a handful of distinct bytes, and a
            // page like that would survive a round trip that did nothing.
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

fn container(compression: TextureCompression) -> TiledTextureImage {
    inf_material::build_tiled_texture(
        corners(320, 192),
        320,
        192,
        TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression,
            hdr: false,
        },
    )
    .expect("the fixture tiles")
}

/// The wgpu format a stored tile uploads as — **the renderer's own mapping**.
///
/// P26.1 parked a copy here for want of a crate to own it; P26.2 built the crate
/// and `inf_render::vt::page_format` is the mapping. This is the call, not a
/// second spelling of it: two matches over six format pairs are two things to
/// keep in step, and the one in a test is the one that would drift unnoticed.
///
/// P26.3 note: the container's header now carries an `inf_vt::PageFormat`
/// directly (the reader moved into `inf-vt`, which knows no `TextureFormat`), so
/// this takes what the header holds and the `.into()` that used to sit here is
/// gone rather than moved.
fn page_format(stored: inf_vt::PageFormat, srgb: bool) -> wgpu::TextureFormat {
    inf_render::vt::page_format(stored, srgb)
}

/// Upload `bytes` as one `size × size` texture of `format` and read them back.
///
/// One function for both tiers. Everything that differs between a block format
/// and a linear one — how many blocks a row holds, how many bytes a block is,
/// how many rows the copy walks — is asked of `wgpu::TextureFormat`, so there is
/// no second arithmetic to be wrong about.
fn upload_and_read_back(
    gpu: &GpuContext,
    format: wgpu::TextureFormat,
    size: u32,
    bytes: &[u8],
) -> Vec<u8> {
    let (bw, bh) = format.block_dimensions();
    let block = format
        .block_copy_size(None)
        .expect("a colour format has a block size");
    let blocks_x = size.div_ceil(bw);
    let blocks_y = size.div_ceil(bh);
    let tight = (blocks_x * block) as usize;
    assert_eq!(
        bytes.len(),
        tight * blocks_y as usize,
        "the blob is not a whole {format:?} image of {size}²"
    );
    // The COPY-BACK is what needs the 256-byte row alignment; `write_texture`
    // takes the tight pitch, which is the point — a cooked tile uploads as it
    // lies on disk, with no repacking.
    let padded = tight.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;

    let extent = wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
    };
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vt-page-proof"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(tight as u32),
            rows_per_image: Some(blocks_y),
        },
        extent,
    );

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vt-page-readback"),
        size: (padded * blocks_y as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vt-page-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
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
        extent,
    );
    gpu.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map_async dropped").expect("map_async");
    let mapped = slice.get_mapped_range().expect("get_mapped_range");
    let mut out = Vec::with_capacity(tight * blocks_y as usize);
    for row in mapped.chunks(padded) {
        out.extend_from_slice(&row[..tight]);
    }
    drop(mapped);
    readback.unmap();
    out
}

/// The probe and the device agree: whatever `AdapterCaps` reports is what
/// `request_device` was actually granted.
///
/// Not a tautology — the adapter *exposes* a feature and the device is created
/// with a mask, and the bug this catches is a mask that forgot to include it, at
/// which point every BC path silently degrades on a machine that supports it.
#[test]
fn the_probe_and_the_created_device_agree_about_bc() {
    let Some(gpu) = gpu_or_skip("the BC probe") else {
        return;
    };
    let caps = AdapterCaps::probe(&gpu);
    assert_eq!(
        caps.texture_compression_bc,
        gpu.supports_texture_compression_bc(),
        "the adapter exposes BC but the device was not created with it (or vice versa)"
    );
    // And the clamp agrees with both, on the live adapter.
    let clamped = inf_render::caps::detect_and_clamp(&gpu, inf_render::RenderSettings::default());
    assert_eq!(clamped.vt.bc_tiles, caps.texture_compression_bc);
    eprintln!(
        "P26.1 BC probe: adapter={:?} texture_compression_bc={}",
        gpu.adapter.get_info().name,
        caps.texture_compression_bc
    );
}

/// **The proof.** A tile of a real container survives the round trip
/// unmodified, through whichever of the two page formats this adapter can take.
#[test]
fn one_tile_uploads_and_reads_back_byte_for_byte() {
    let Some(gpu) = gpu_or_skip("the BC upload proof") else {
        return;
    };
    let bc = gpu.supports_texture_compression_bc();

    for compression in [TextureCompression::Bc1, TextureCompression::Bc3] {
        let image = container(compression);
        let r = image.reader();
        assert_eq!((r.mips()[0].tiles_x, r.mips()[0].tiles_y), (3, 2));

        // The bottom-right tile of mip 0, deliberately: (2, 1) is the address a
        // transposed index cannot reach, and its content (the yellow quadrant)
        // differs from every other tile's.
        let blob = r.tile(0, 2, 1).expect("tile (0, 2, 1)");
        assert_eq!(blob.len(), r.header().tile_bytes as usize);

        let (format, upload) = if bc {
            (
                page_format(r.header().format, r.header().srgb),
                blob.to_vec(),
            )
        } else {
            // The documented fallback: the SAME tile, transcoded on the CPU,
            // through the SAME call below.
            (
                page_format(inf_vt::PageFormat::Rgba8, r.header().srgb),
                r.tile_rgba8(0, 2, 1).expect("the tile transcodes"),
            )
        };

        // Anti-vacuity: an all-zero (or block-repeating) page would survive any
        // round trip, including one that uploaded nothing at all. Counted in
        // 16-byte chunks rather than in bytes, because a BC page is structured —
        // it can carry hundreds of distinct BLOCKS out of a dozen distinct byte
        // values, and it is the blocks that make the round trip falsifiable.
        let distinct: std::collections::BTreeSet<&[u8]> = upload.chunks(16).collect();
        assert!(
            distinct.len() > 64,
            "the page is nearly uniform ({} distinct 16-byte chunks) — the round \
             trip would prove nothing",
            distinct.len()
        );

        let back = upload_and_read_back(&gpu, format, STORED_TILE_SIZE, &upload);
        assert_eq!(
            back.len(),
            upload.len(),
            "{compression:?}: readback length differs"
        );
        assert!(
            back == upload,
            "{compression:?}: the page came back changed ({} of {} bytes differ)",
            back.iter().zip(&upload).filter(|(a, b)| a != b).count(),
            upload.len()
        );
        eprintln!(
            "P26.1 page round trip: {compression:?} -> {format:?}, {} B, {} distinct chunks, bc={bc}",
            upload.len(),
            distinct.len()
        );
    }
}

/// The transcode fallback is exercised on **every** adapter, not only on the
/// ones that lack BC — otherwise the path that keeps a BC-less machine running
/// would be dead code everywhere it is tested.
///
/// Uploading transcoded RGBA8 needs no optional feature at all, so this runs
/// wherever there is an adapter.
#[test]
fn the_transcode_fallback_uploads_on_any_adapter() {
    let Some(gpu) = gpu_or_skip("the transcode fallback") else {
        return;
    };
    let image = container(TextureCompression::Bc1);
    let r = image.reader();
    let rgba = r.tile_rgba8(0, 2, 1).expect("the tile transcodes");
    assert_eq!(
        rgba.len(),
        (STORED_TILE_SIZE * STORED_TILE_SIZE * 4) as usize,
        "a transcoded page is the stored tile, border ring and all"
    );
    let back = upload_and_read_back(
        &gpu,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        STORED_TILE_SIZE,
        &rgba,
    );
    assert!(back == rgba, "the transcoded page came back changed");

    // It is 8× the BC1 page — the cost of the fallback, measured rather than
    // asserted in a comment. (BC1 is 4 bits per texel against RGBA8's 32, so the
    // ratio is 8:1; BC3's is 8 bits and 4:1. The prose in `caps.rs`,
    // `settings.rs` and `gpu.rs` said 4× for both, contradicted by this line.)
    let bc_bytes = r.header().tile_bytes as usize;
    assert_eq!(rgba.len(), bc_bytes * 8, "BC1 is 8:1 against RGBA8");
}
