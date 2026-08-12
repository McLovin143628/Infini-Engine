//! **The VT sampling gate** (P26.3): the shader's walk, against the CPU's.
//!
//! # What this aims at
//!
//! `vt_sample.wgsl` re-implements, in WGSL, arithmetic that `inf-vt` already owns
//! in Rust. Two implementations of one function is exactly the shape the house
//! keeps finding drift in, and this one has a known, *measured* way to be wrong:
//! the P26.2 audit's F1. The sketch the shader was first written against derived
//! the tile from `uv` at the RESOLVED mip (`floor(gp / tile_size)`), and on an
//! odd-halving pyramid that is a different function from the tile the indirection
//! entry names — 1 322 addresses of a 4095² pyramid land a whole tile, 128
//! texels, from where they belong, at exactly one column of one tile boundary.
//!
//! So the arms below are built to catch that specific class:
//!
//! * [`the_shader_walk_lands_on_the_tile_the_entry_names`] — a **line-for-line
//!   Rust twin** of the WGSL walk, swept over the extents where the two
//!   derivations part, asserted against [`VtTextureDesc::ancestor`]. It is
//!   anti-vacuous in two directions: the sweep must contain a disagreement with
//!   the forbidden derivation, and `local` must go negative, or the border ring
//!   is a statement about nothing.
//! * [`the_wgsl_implements_the_twin_above`] — a source gate over
//!   `vt_sample.wgsl`, because a twin that agrees with a shader nobody checked is
//!   a twin that agrees with itself. It bans the forbidden spelling by MODULE
//!   (the P23 law: a byte pin cannot see a semantic change, so read a scope) and
//!   requires the tile-tree loop.
//! * the residency arms below — the fallback landing on the finest resident
//!   ancestor asserted as **state**, and the registration/warm/floor rules —
//!   plus a GPU arm that pages a real container into a real atlas and reads the
//!   pages back out of it.
//!
//! Every GPU arm skips cleanly, and says so, when the machine has no adapter.

use std::sync::Arc;

use inf_render::vt::{pool_format, VtPools};
use inf_render::vt_library::{VtTextures, VT_FLOOR_LEVELS};
use inf_render::GpuContext;
use inf_vt::{
    full_pyramid, unpack_entry, PageFormat, TileCoord, VtPoolConfig, VtTextureDesc,
    VtTextureHandle, STORED_TILE_SIZE, TABLE_HEADER_WORDS, TABLE_MIP_REC_WORDS,
    TABLE_TEXTURE_HEADER_WORDS,
};

/// The WGSL walk, transliterated. **Every line here has a counterpart in
/// `vt_sample.wgsl` and the source gate below pins that they stay together.**
///
/// Returns `(page slot, resolved mip, local offset in texels)` — the three
/// things the atlas address is built from.
fn shader_walk(words: &[u32], tex: u32, uv: (f32, f32), want_mip: u32) -> (u32, u32, (f32, f32)) {
    let b = words[TABLE_HEADER_WORDS + tex as usize] as usize;
    let mip_count = words[b];
    let tsz = words[b + 1];
    let tile_sz = tsz as f32;
    // Spelled as the WGSL spells it — `(h + tsz - 1u) / tsz` — because this
    // function is a transliteration and `div_ceil` has no WGSL counterpart. The
    // clippy lint is allowed HERE and nowhere else for exactly that reason.
    #[allow(clippy::manual_div_ceil)]
    let tiles_y = |h: u32| ((h + tsz - 1) / tsz).max(1);

    let w_uv = (uv.0 - uv.0.floor(), uv.1 - uv.1.floor());
    let m = want_mip.min(mip_count - 1);
    let rec = b + TABLE_TEXTURE_HEADER_WORDS + m as usize * TABLE_MIP_REC_WORDS;
    let (mw, mh) = (words[rec], words[rec + 1]);
    let tiles_x = words[rec + 2];
    let tx = ((w_uv.0 * mw as f32 / tile_sz) as u32).min(tiles_x - 1);
    let ty = ((w_uv.1 * mh as f32 / tile_sz) as u32).min(tiles_y(mh) - 1);
    let entry = words[words[rec + 3] as usize + (ty * tiles_x + tx) as usize];
    let e = unpack_entry(entry);

    let (mut gtx, mut gty) = (tx, ty);
    for lv in m..e.mip {
        let r = b + TABLE_TEXTURE_HEADER_WORDS + (lv as usize + 1) * TABLE_MIP_REC_WORDS;
        gtx = (gtx / 2).min(words[r + 2] - 1);
        gty = (gty / 2).min(tiles_y(words[r + 1]) - 1);
    }

    let grec = b + TABLE_TEXTURE_HEADER_WORDS + e.mip as usize * TABLE_MIP_REC_WORDS;
    let gp = (w_uv.0 * words[grec] as f32, w_uv.1 * words[grec + 1] as f32);
    let local = (gp.0 - gtx as f32 * tile_sz, gp.1 - gty as f32 * tile_sz);
    (e.slot, e.mip, local)
}

fn pool_cfg(format: PageFormat, pages: u64) -> VtPoolConfig {
    VtPoolConfig {
        format,
        stored_tile_size: STORED_TILE_SIZE,
        budget_bytes: format.page_bytes(STORED_TILE_SIZE) * pages,
        max_texture_dim: 8192,
    }
}

/// A real v2 container, built by the one writer, whose texels are a **prime-period
/// diagonal ramp per channel** — the fixture-asymmetry law: a transposed tile
/// lookup, a mirrored row order or an off-by-a-tile address all read a different
/// value, and no symmetry can rescue them.
fn container(w: u32, h: u32, srgb: bool, compression: inf_material::TextureCompression) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&[
                ((x * 7 + y * 3) % 251) as u8,
                ((x * 11 + y * 29) % 241) as u8,
                ((x * 5 + y * 17) % 239) as u8,
                255,
            ]);
        }
    }
    inf_material::build_tiled_texture(
        rgba,
        w,
        h,
        inf_material::TextureImportSettings {
            srgb,
            generate_mips: true,
            compression,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

fn library(w: u32, h: u32) -> VtTextures {
    let (mut lib, _) = VtTextures::new(pool_cfg(PageFormat::Rgba8, 512));
    lib.register_or_record(
        1,
        Arc::new(container(
            w,
            h,
            true,
            inf_material::TextureCompression::None,
        )),
    )
    .expect("the fixture registers");
    lib
}

// ── (a) the contract arm: the shader walk lands where the entry points ──────

/// **The F1 regression arm.** The transliterated shader walk must land on the
/// tile [`VtTextureDesc::ancestor`] names, for every tile of every level of every
/// pyramid in the sweep — including the odd-halving extents where the forbidden
/// derivation is off by a whole tile.
#[test]
fn the_shader_walk_lands_on_the_tile_the_entry_names() {
    // The extents the P26.2 audit measured the disagreement on, plus the shapes
    // that stress the sliver and the strip. 4095² is the one that produced 1 322
    // disagreements; it is here in full.
    let extents = [
        (511u32, 3u32),
        (1023, 1023),
        (2047, 511),
        (4095, 4095),
        (320, 192),
        (257, 257),
        (8192, 4),
    ];
    let mut disagreements = 0u64;
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    let mut walked = 0u64;

    for (w, h) in extents {
        let desc = full_pyramid(w, h, 128, 4, false);
        // **Mip 1 resident, mip 0 not.** This is the residency the F1 class needs
        // and the one a roots-only pool cannot produce: with only the roots
        // admitted every address resolves to the coarsest level, which is ONE
        // tile, and `min(x / 2, tiles - 1)` clamps everything to tile 0 — the
        // disagreement is real and invisible, which is exactly how the P26.2
        // sketch survived its first gate. Resolving one level up, onto a grid
        // that has more than one tile, is where the two derivations part.
        let (mut res, _) = inf_vt::VtResidency::new(pool_cfg(PageFormat::Bc1, 512));
        let handle = res.register_texture(desc.clone()).expect("registers");
        let m1 = desc.mips[1];
        let wants: Vec<_> = (0..m1.tiles_y)
            .flat_map(|y| {
                (0..m1.tiles_x).map(move |x| inf_vt::VtWant::new(handle, TileCoord::new(1, x, y)))
            })
            .collect();
        let txn = res.apply_wants(&wants);
        assert_eq!(
            txn.deferred, 0,
            "{w}x{h}: the sweep pool could not hold mip 1"
        );
        let words = res.table_words().to_vec();

        // Mip 0 only: those are the addresses that resolve UP into mip 1.
        for mip in 0..1 {
            let m = desc.mips[mip as usize];
            for ty in 0..m.tiles_y {
                for tx in 0..m.tiles_x {
                    let at = TileCoord::new(mip, tx, ty);
                    // Two corners of the tile, in the level's own texel space —
                    // `tile_at_texel` is monotone, so an interior texel cannot
                    // leave a range both ends of which are inside.
                    for (px, py) in [
                        (tx * 128, ty * 128),
                        (
                            ((tx + 1) * 128 - 1).min(m.width - 1),
                            ((ty + 1) * 128 - 1).min(m.height - 1),
                        ),
                    ] {
                        let uv = (
                            (px as f32 + 0.5) / m.width as f32,
                            (py as f32 + 0.5) / m.height as f32,
                        );
                        let (slot, got, local) = shader_walk(&words, handle.0, uv, mip);
                        walked += 1;

                        // The resolved tile the shader's walk implies, read back
                        // out of the WORLD: the slot it landed on holds exactly
                        // the tile `ancestor` names.
                        let anc = desc.ancestor(at, got).expect("an ancestor exists");
                        assert_eq!(
                            res.slot_occupant(slot),
                            Some((handle, anc)),
                            "{w}×{h} {at:?} → mip {got}: the page the shader addressed does \
                             not hold the tile the entry names"
                        );

                        // …and the offset inside it stays within the payload
                        // EXTENDED BY THE BORDER RING. Never more than one texel
                        // out, which is what `+ border` in the shader rests on.
                        for l in [local.0, local.1] {
                            lo = lo.min(l);
                            hi = hi.max(l);
                            assert!(
                                (-(desc.border as f32)..(128.0 + desc.border as f32)).contains(&l),
                                "{w}×{h} {at:?} → mip {got}: {l} texels into a 128-texel tile \
                                 with a {}-texel border",
                                desc.border
                            );
                        }

                        // The FORBIDDEN derivation, measured rather than argued.
                        let t = desc.mips[got as usize];
                        let qx = ((uv.0 * t.width as f32) as u32).min(t.width - 1);
                        let qy = ((uv.1 * t.height as f32) as u32).min(t.height - 1);
                        disagreements += u64::from(desc.tile_at_texel(got, qx, qy) != Some(anc));
                    }
                }
            }
        }
    }

    // Anti-vacuity, three ways. Without these the arm is satisfied perfectly by a
    // sweep that never reaches the case, which is exactly what the first version
    // of `inf-vt`'s own version of this was.
    // Measured on this sweep, not guessed: seven pyramids, every tile of every
    // level, two corners each. The floor is a tripwire for an extent list that
    // quietly shrank, so it sits under the real number rather than at it.
    assert!(walked > 2_000, "the sweep walked only {walked} addresses");
    assert!(
        disagreements > 0,
        "no address in the sweep disagreed with the uv derivation — the extents \
         stopped reaching the class this arm exists for"
    );
    assert!(
        lo < 0.0,
        "the offset never went negative ({lo}..={hi}), so the border ring was never \
         load-bearing and this arm proves nothing about it"
    );
}

/// **The twin has to be a twin.** A Rust transliteration that agrees with a
/// shader nobody read is a Rust function that agrees with itself.
///
/// Scoped by MODULE rather than by byte (the P23 law — a `pub fn recompute`
/// wrapper once defeated a gate that banned one spelling): the whole shader is
/// read, comments stripped, and the two structural facts asserted.
#[test]
fn the_wgsl_implements_the_twin_above() {
    let src = include_str!("../src/shaders/vt_sample.wgsl");
    // Comments carry the explanation of the forbidden form, so a gate that did
    // not strip them would be satisfied by the prose that warns against it —
    // the P26.1 audit's comment-blindness finding, applied before it bites.
    let code: String = src
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        code.contains("for (var lv = m; lv < got; lv = lv + 1u)"),
        "the tile-tree walk is gone from vt_sample.wgsl — the shader is now free \
         to disagree with `VtTextureDesc::ancestor` on every odd-halving pyramid"
    );
    assert!(
        code.contains("gtx = min(gtx / 2u,") && code.contains("gty = min(gty / 2u,"),
        "the clamped parent step is not `min(t / 2, tiles - 1)` any more"
    );
    for banned in ["floor(gp", "gp / tile_sz", "gp / f32(tsz)"] {
        assert!(
            !code.contains(banned),
            "`{banned}` is back in vt_sample.wgsl: that re-derives the tile from \
             uv at the RESOLVED mip, which is off by a whole tile on 1 322 \
             addresses of a 4095² pyramid"
        );
    }
    // **The border offset**, the other half of the walk's contract. The twin
    // above proves `local` reaches -1 on a real pyramid; that is only safe
    // because the shader adds the border ring back before it addresses the
    // atlas. Dropping `+ vec2<f32>(border)` was measured invisible to every
    // other arm in this crate — a negative `local` then addresses the slot
    // BEFORE this one, i.e. another texture's texels.
    assert!(
        code.contains("origin + vec2<f32>(border) + local"),
        "the border offset is gone from vt_sample.wgsl's atlas address — a \
         sample whose `local` is negative now reads the previous slot"
    );
    // Anti-vacuity: the stripper left the code, not only the comments.
    assert!(
        code.contains("fn vt_sample(") && code.contains("textureSampleLevel(vt_atlas"),
        "the comment stripper ate the shader"
    );
}

// ── (b) residency: the fallback lands on the finest resident ancestor ───────

/// **Assert the WORLD, not the report.** Evicting a tile must re-point it — and
/// its whole non-resident subtree — at the finest ancestor that is still
/// resident, and the shader walk must then land on that ancestor's page.
#[test]
fn an_evicted_tile_resolves_through_its_finest_resident_ancestor() {
    let desc = full_pyramid(1023, 1023, 128, 4, false);
    // Twelve pages: the mandatory floor is one, so eleven are evictable and the
    // pressure below cannot fail to displace the fine tile. A pool with room to
    // spare would make this arm certify a no-op.
    let (mut res, _) = inf_vt::VtResidency::new(pool_cfg(PageFormat::Bc1, 12));
    let h = res.register_texture(desc.clone()).expect("registers");

    // Want a fine tile and every ancestor of it.
    let target = TileCoord::new(0, 3, 5);
    let chain: Vec<TileCoord> = (0..desc.mip_count())
        .map(|m| desc.ancestor(target, m).expect("ancestor"))
        .collect();
    let wants: Vec<_> = chain.iter().map(|t| inf_vt::VtWant::new(h, *t)).collect();
    let _ = res.apply_wants(&wants);
    assert!(res.is_resident(h, target), "the fine tile was not admitted");

    let resolved = res.resolve(h, target).expect("resolves");
    assert_eq!(resolved.tile, target, "a resident tile resolves to itself");

    // Now squeeze it out: want a large disjoint set, twice, so the LRU takes the
    // fine tile. Its parent is still wanted, so the fallback has somewhere to go.
    let mut pressure: Vec<_> = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            pressure.push(inf_vt::VtWant::new(h, TileCoord::new(0, x + 4, y)));
        }
    }
    pressure.push(inf_vt::VtWant::new(h, chain[1]));
    let _ = res.apply_wants(&pressure);
    let _ = res.apply_wants(&pressure);

    if res.is_resident(h, target) {
        // The pool was large enough that nothing had to go; the arm would then be
        // asserting nothing at all, which is the failure mode it is built against.
        panic!("the fine tile was never evicted — this arm measured nothing");
    }
    let after = res.resolve(h, target).expect("still resolves");
    assert_ne!(
        after.tile, target,
        "an evicted tile still resolves to itself"
    );
    assert!(
        res.is_resident(h, after.tile),
        "the fallback landed on a tile that is not resident — a hole, which the \
         first law forbids"
    );
    assert_eq!(
        after.tile,
        desc.ancestor(target, after.tile.mip).expect("ancestor"),
        "the fallback is not on the target's own ancestor chain"
    );
    // …and the finest such: every strictly finer ancestor is gone.
    for m in target.mip..after.tile.mip {
        let a = desc.ancestor(target, m).expect("ancestor");
        assert!(
            !res.is_resident(h, a),
            "{a:?} is resident and finer than the {:?} the fallback chose",
            after.tile
        );
    }
    // And the slot the SHADER would address holds exactly that ancestor.
    let words = res.table_words().to_vec();
    let m0 = desc.mips[0];
    let uv = (
        (target.x as f32 * 128.0 + 0.5) / m0.width as f32,
        (target.y as f32 * 128.0 + 0.5) / m0.height as f32,
    );
    let (slot, got, _) = shader_walk(&words, h.0, uv, 0);
    assert_eq!(got, after.tile.mip);
    assert_eq!(res.slot_occupant(slot), Some((h, after.tile)));
}

// ── (c) the door: registration, the warm gate, the floor ────────────────────

/// The floor a scene load admits is deterministic, bounded, and enough to make a
/// surface visibly textured rather than visibly one colour.
#[test]
fn the_static_floor_admits_the_coarsest_levels_of_every_texture() {
    let mut lib = library(1024, 1024);
    lib.register_or_record(
        2,
        Arc::new(container(
            512,
            256,
            false,
            inf_material::TextureCompression::None,
        )),
    )
    .expect("a second texture registers");

    let wants = lib.want_floor();
    assert!(!wants.is_empty());
    // Bounded: at most 21 pages per texture, whatever the extents.
    assert!(
        wants.len() <= 42,
        "the floor asked for {} pages for two textures",
        wants.len()
    );
    // Every want is inside its texture's grid, and inside the coarsest N levels.
    for w in &wants {
        let desc = lib.residency().desc(w.texture).expect("registered");
        assert!(desc.contains(w.tile));
        assert!(desc.coarsest_mip() - w.tile.mip < VT_FLOOR_LEVELS);
    }
    // …and the same registry built twice asks for the same thing, byte for byte.
    let again = library(1024, 1024);
    assert_eq!(
        lib.want_floor()[..again.want_floor().len()],
        again.want_floor()[..]
    );
}

// ── (d) the GPU: a real container pages into a real atlas ───────────────────

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available for {what} ({e})");
            None
        }
    }
}

/// **The pages in the atlas are the container's own bytes.** Registration →
/// floor → `sync` → read the atlas back and compare every admitted page against
/// the tile it claims to hold, texel for texel.
///
/// Runs on **both** adapter tiers through one call shape: with
/// `TEXTURE_COMPRESSION_BC` the page is the stored BC blob, without it the pool
/// is RGBA8 and the same tile arrives through `tile_rgba8` — the same door, one
/// format decision earlier. The comparison is exact either way: a
/// texture→buffer copy is a copy.
#[test]
fn every_admitted_page_holds_the_tile_it_names() {
    let Some(gpu) = gpu_or_skip("the VT page upload") else {
        return;
    };
    let bc = gpu.supports_texture_compression_bc();
    let mut settings = inf_render::RenderSettings::default();
    settings.vt.bc_tiles = bc;
    let stored = if bc {
        PageFormat::Bc1
    } else {
        PageFormat::Rgba8
    };
    let format = pool_format(&settings, stored);
    let compression = if bc {
        inf_material::TextureCompression::Bc1
    } else {
        inf_material::TextureCompression::None
    };

    let bytes = container(320, 192, true, compression);
    let (mut lib, advisories) = VtTextures::new(pool_cfg(format, 128));
    assert!(advisories.is_empty(), "{advisories:?}");
    lib.register_or_record(1, Arc::new(bytes.clone()))
        .expect("registers");

    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    // The floor plus one mip-0 tile. The floor alone is the pyramid's TAIL —
    // levels of a handful of texels, padded to 136 square, which BC1 encodes as
    // a near-flat page — so a round trip over the floor alone would be a round
    // trip over pages that a no-op could reproduce. One detailed page makes the
    // comparison mean something, and the anti-vacuity counter below is what says
    // so out loud.
    let mut wants = lib.want_floor();
    wants.push(inf_vt::VtWant::new(
        VtTextureHandle(0),
        TileCoord::new(0, 1, 0),
    ));
    let (txn, report) = lib.sync(&gpu.device, &gpu.queue, &mut pools, &wants);
    assert!(
        report.missing.is_empty(),
        "the fetch could not produce {} pages",
        report.missing.len()
    );
    assert!(
        report.pages > 0,
        "nothing was admitted — the arm measured nothing"
    );
    assert_eq!(report.pages, txn.admits.len());

    // Read the whole atlas back and compare each admitted slot against the
    // container's own tile bytes.
    let reader = inf_vt::TiledTextureReader::new(bytes.as_slice()).expect("the fixture reads back");
    let page_bytes = lib.residency().page_bytes() as usize;
    let mut varied = 0u32;

    for admit in &txn.admits {
        let expect: Vec<u8> = if format == PageFormat::Rgba8 && stored != PageFormat::Rgba8 {
            reader
                .tile_rgba8(admit.tile.mip, admit.tile.x, admit.tile.y)
                .expect("tile transcodes")
        } else {
            reader.tile_at(admit.tile).expect("tile exists").to_vec()
        };
        assert_eq!(expect.len(), page_bytes);
        // Anti-vacuity, counted across the batch rather than per page: the tail
        // mips of any pyramid ARE one flat colour (a 1x1 level padded to 136
        // square), so demanding variety of every page would be demanding it of a
        // page that cannot have it. What must not happen is that EVERY page is
        // uniform, because then the comparison below would survive a round trip
        // that did nothing.
        varied += u32::from(
            expect
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 8,
        );
        assert_eq!(
            read_page(&gpu, &pools, admit.slot),
            expect,
            "slot {} does not hold {:?}'s bytes",
            admit.slot,
            admit.tile
        );
    }
    assert!(
        varied > 0,
        "every admitted page was a flat colour — the round trip proved nothing"
    );
}

/// One physical page, read back out of the atlas, tightly packed.
///
/// Copied at the SLOT's own origin rather than gathered out of a whole-atlas
/// readback: the slot origin is the one piece of arithmetic the mirror already
/// owns (`VtPoolGeometry::slot_origin`), so asking `wgpu` for exactly that
/// sub-rectangle leaves this arm with no second layout of its own to be wrong
/// about.
fn read_page(gpu: &GpuContext, pools: &VtPools, slot: u32) -> Vec<u8> {
    let geometry = pools.geometry();
    let fmt = pools.format();
    let (bw, bh) = fmt.block_dimensions();
    let block = fmt.block_copy_size(None).expect("colour format");
    let side = geometry.stored_tile_size;
    let blocks_x = side.div_ceil(bw);
    let blocks_y = side.div_ceil(bh);
    let tight = (blocks_x * block) as usize;
    let padded = tight.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let (ox, oy) = geometry.slot_origin(slot).expect("slot in atlas");

    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vt-page-readback"),
        size: (padded * blocks_y as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: pools.atlas(),
            mip_level: 0,
            origin: wgpu::Origin3d { x: ox, y: oy, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
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
    gpu.queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("get_mapped_range");
    let mut out = Vec::with_capacity(tight * blocks_y as usize);
    for r in 0..blocks_y as usize {
        out.extend_from_slice(&mapped[r * padded..r * padded + tight]);
    }
    drop(mapped);
    buffer.unmap();
    out
}

// ── (e) the lit path: a virtual texture reaches a PIXEL ─────────────────────
//
// Everything above proves the bytes reach the atlas and that the address
// arithmetic agrees with `inf-vt`. Nothing above renders a fragment, and the
// audit measured what that costs: severing the sRGB decode, swapping the albedo
// and ORM words in `InstanceRaw::pack`, collapsing `vt_box_uv` to a constant and
// dropping the border offset were all invisible to the whole `inf-render` test
// suite. The arms below close that: they draw a real frame through the real
// `EngineRenderer` and read the colour back.

const FW: u32 = 320;
const FH: u32 = 180;

/// One cube, scaled ×2 at the origin, whose +Z face fills the frame at the view
/// below — so every pixel read back is a fragment of one textured surface and no
/// arm needs a mask to find it.
fn face_scene(set: inf_render::VtTextureSet) -> inf_render::RenderScene {
    let mut scene = inf_render::RenderScene::default();
    scene.instances.push(inf_render::MeshInstance {
        vt: set,
        translation: glam::DVec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::splat(2.0),
        color: [1.0, 1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 1,
        mesh: inf_render::PrimMesh::Cube,
        blend: 0,
        cutoff: 0.5,
    });
    scene.mark_dirty();
    scene
}

/// Head-on at the +Z face, close enough that the face overfills 320×180 on both
/// axes. `vt_box_uv` maps that face onto uv ≈ [0.09, 0.91] × [0.27, 0.73], and
/// the footprint works out under one texel of a 256² texture per pixel, so the
/// gradient mip resolves to **mip 0** — the level the arms below make resident.
fn face_view() -> inf_render::RenderView {
    inf_render::RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.0, 0.0, 1.8),
        forward: glam::Vec3::new(0.0, 0.0, -1.0),
        up: glam::Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: FW,
        height: FH,
        ortho: None,
    }
}

/// Draw one frame of [`face_scene`] and return `(rgba, engaged frames)`.
fn render_face(
    gpu: &GpuContext,
    pools: Option<VtPools>,
    set: inf_render::VtTextureSet,
) -> (Vec<u8>, u64) {
    let target = inf_render::HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    renderer.set_vt_pools(pools);
    renderer.render(gpu, &face_scene(set), &face_view(), &target.view, (FW, FH));
    (
        target.read_rgba(gpu).expect("readback"),
        renderer.vt_engaged_frames(),
    )
}

/// The red channel's `(min, max, mean)` over a frame.
fn red_stats(rgba: &[u8]) -> (u8, u8, f64) {
    let r: Vec<u8> = rgba.chunks_exact(4).map(|p| p[0]).collect();
    let sum: u64 = r.iter().map(|&v| u64::from(v)).sum();
    (
        *r.iter().min().expect("a frame"),
        *r.iter().max().expect("a frame"),
        sum as f64 / r.len() as f64,
    )
}

/// A registry holding one 256² texture whose **whole pyramid** is resident, so a
/// mip-0 sample resolves to mip 0 and the frame shows the texels rather than the
/// pyramid's tail. Returns the library, its synced pools, and the warm set.
fn resident_library(
    gpu: &GpuContext,
    srgb: bool,
) -> (VtTextures, VtPools, inf_render::VtTextureSet) {
    let (mut lib, _) = VtTextures::new(pool_cfg(PageFormat::Rgba8, 64));
    lib.register_or_record(1, Arc::new(ramp_container(256, srgb)))
        .expect("the fixture registers");
    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    // Every tile of every level — the floor alone stops three levels from the
    // top, which on a 256² pyramid is a 4×4 texel image and would make every
    // assertion below a statement about a blur.
    let desc = lib
        .residency()
        .desc(VtTextureHandle(0))
        .expect("registered");
    let wants: Vec<_> = (0..desc.mip_count())
        .flat_map(|m| {
            let g = desc.mips[m as usize];
            (0..g.tiles_y).flat_map(move |y| {
                (0..g.tiles_x)
                    .map(move |x| inf_vt::VtWant::new(VtTextureHandle(0), TileCoord::new(m, x, y)))
            })
        })
        .collect();
    let (txn, report) = lib.sync(&gpu.device, &gpu.queue, &mut pools, &wants);
    assert_eq!(txn.deferred, 0, "the fixture pyramid did not fit");
    assert!(
        report.missing.is_empty(),
        "{} pages missing",
        report.missing.len()
    );
    let set = lib.set_for(Some(1), None, None);
    assert!(!set.is_none(), "the fixture never went warm");
    (lib, pools, set)
}

/// A square texture whose **red channel ramps left to right** and whose green and
/// blue are constant. The ramp is what makes "the uv varies across the surface" a
/// measurable claim rather than a hope: a projection collapsed to a constant, or
/// an address that lands on one texel, flattens the frame.
fn ramp_container(n: u32, srgb: bool) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((n * n * 4) as usize);
    for _y in 0..n {
        for x in 0..n {
            rgba.extend_from_slice(&[(x * 255 / (n - 1)) as u8, 40, 200, 255]);
        }
    }
    inf_material::build_tiled_texture(
        rgba,
        n,
        n,
        inf_material::TextureImportSettings {
            srgb,
            generate_mips: true,
            compression: inf_material::TextureCompression::None,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

/// **A virtual texture reaches a pixel, and the engagement counter says so.**
///
/// The claim the batch's headline makes — "material textures appear in the
/// interactive renderer for the first time" — asserted on the frame rather than
/// on the atlas. Three things have to hold at once, and each falsifies a
/// different way of shipping a feature that does nothing:
///
/// * the textured frame differs from the textureless one (something sampled);
/// * the textured frame **varies across the surface** while the textureless one
///   does not (the uv is a projection, not a constant — a `vt_box_uv` that
///   returns one value passes every other arm in this file);
/// * `vt_engaged_frames` is 1 after one frame with a pool and **0** after one
///   frame without, which is what makes "the 50 goldens did not change" a
///   measurement of the command stream rather than an expectation.
#[test]
fn a_virtual_texture_reaches_the_lit_pixel() {
    let Some(gpu) = gpu_or_skip("the VT lit path") else {
        return;
    };
    let (_lib, pools, set) = resident_library(&gpu, false);

    let (bare, bare_engaged) = render_face(&gpu, None, inf_render::VtTextureSet::NONE);
    let (textured, engaged) = render_face(&gpu, Some(pools), set);

    assert_eq!(
        bare_engaged, 0,
        "a frame drawn with no pool engaged virtual texturing"
    );
    assert_eq!(engaged, 1, "the frame drawn with a pool did not engage");

    let (blo, bhi, bmean) = red_stats(&bare);
    let (tlo, thi, tmean) = red_stats(&textured);
    let bare_spread = u32::from(bhi) - u32::from(blo);
    let tex_spread = u32::from(thi) - u32::from(tlo);

    // Anti-vacuity first: the untextured face must be ON SCREEN and lit, or both
    // frames are a comparison between two skies.
    assert!(
        bmean > 20.0,
        "the untextured face never reached the frame (mean red {bmean:.1})"
    );
    assert_ne!(bare, textured, "the texture changed no pixel at all");
    // The ramp spans the face; a flat white one does not. Floors under the
    // measured numbers, not at them.
    assert!(
        tex_spread > 100,
        "the textured face spans only {tex_spread} of red — the uv is not varying \
         across the surface (a constant projection looks exactly like this)"
    );
    assert!(
        bare_spread < tex_spread / 2,
        "the untextured face already spans {bare_spread} of red against the \
         texture's {tex_spread}, so the spread test proves nothing"
    );
    assert!(
        (tmean - bmean).abs() > 5.0,
        "textured mean {tmean:.1} vs untextured {bmean:.1}"
    );
}

/// **The three slots are three different slots, and the sRGB flag is read.**
///
/// Two mutations this arm exists for, both measured invisible to every other arm
/// in the crate before it: swapping the albedo and ORM words in
/// `InstanceRaw::pack` (occlusion would silently become base colour), and
/// deleting the `vt_srgb_to_linear` call in `vt_sample_color` (a base colour
/// would ship gamma-encoded).
#[test]
fn the_slot_routing_and_the_srgb_flag_reach_the_pixel() {
    let Some(gpu) = gpu_or_skip("the VT slot routing") else {
        return;
    };
    // Same texels, same handle, different SLOT.
    let (_lin, lin_pools, lin_set) = resident_library(&gpu, false);
    let (albedo, _) = render_face(&gpu, Some(lin_pools), lin_set);

    let (_orm_lib, orm_pools, warm) = resident_library(&gpu, false);
    let orm_set = inf_render::VtTextureSet {
        albedo: 0,
        normal: 0,
        orm: warm.albedo,
    };
    let (orm, _) = render_face(&gpu, Some(orm_pools), orm_set);
    assert_ne!(
        albedo, orm,
        "the same texture in the albedo slot and the ORM slot rendered the same \
         frame — the three per-instance words are not three different maps"
    );

    // Same texels, same slot, different SRGB FLAG. The atlas bytes are identical
    // (srgb is a flag in the container header, not a transform on the texels), so
    // any difference here is the shader's transfer function and nothing else.
    let (_srgb_lib, srgb_pools, srgb_set) = resident_library(&gpu, true);
    let (decoded, _) = render_face(&gpu, Some(srgb_pools), srgb_set);
    let (_, _, lin_mean) = red_stats(&albedo);
    let (_, _, srgb_mean) = red_stats(&decoded);
    assert!(
        (lin_mean - srgb_mean).abs() > 5.0,
        "an sRGB base colour and a linear one rendered the same mean red \
         ({lin_mean:.1} vs {srgb_mean:.1}) — the per-texture transfer function \
         in `vt_sample_color` is not being applied"
    );
    assert!(
        srgb_mean < lin_mean,
        "sRGB→linear must DARKEN a mid-tone ramp: {srgb_mean:.1} vs {lin_mean:.1}"
    );
}

/// A texture-set is a pure function of `(registry state, three GUIDs)`, so two
/// hosts that registered the same payloads in the same order resolve a material
/// identically — the mirror property, asserted on the RULE rather than on two
/// call sites hoped to agree.
#[test]
fn two_registries_resolve_one_material_identically() {
    let build = || {
        let (mut lib, _) = VtTextures::new(pool_cfg(PageFormat::Rgba8, 256));
        for (guid, w, h, srgb) in [(11u128, 256u32, 256u32, true), (22, 128, 64, false)] {
            lib.register_or_record(
                guid,
                Arc::new(container(
                    w,
                    h,
                    srgb,
                    inf_material::TextureCompression::None,
                )),
            )
            .expect("registers");
        }
        let wants = lib.want_floor();
        let _ = lib.residency_mut().apply_wants(&wants);
        lib
    };
    let (editor, player) = (build(), build());
    for material in [
        (Some(11u128), Some(22u128), None),
        (None, None, Some(11)),
        (Some(99), None, None),
        (None, None, None),
    ] {
        let a = editor.set_for(material.0, material.1, material.2);
        let b = player.set_for(material.0, material.1, material.2);
        assert_eq!(a, b, "the two registries disagree about {material:?}");
    }
    // Anti-vacuity: the interesting case is not all zeros.
    assert_ne!(
        editor.set_for(Some(11), Some(22), None),
        inf_render::VtTextureSet::NONE,
        "the registry resolved nothing — this arm compared two absences"
    );
}

/// A handle is `slot - 1`, and the encoding leaves 0 free for "nothing". Pinned
/// because the shader reads it and a change here is silent everywhere else.
#[test]
fn the_slot_encoding_is_the_handle_plus_one() {
    let mut lib = library(128, 128);
    let wants = lib.want_floor();
    let _ = lib.residency_mut().apply_wants(&wants);
    let set = lib.set_for(Some(1), None, None);
    assert_eq!(set.albedo, VtTextureHandle(0).0 + 1);
    assert_eq!(set.slots(), [1, 0, 0]);
    assert!(!set.is_none());
    assert!(inf_render::VtTextureSet::NONE.is_none());
}

/// A descriptor whose stored tile does not match the pool's is refused **by
/// name** at the door, rather than paging garbage into a slot.
#[test]
fn a_geometry_mismatch_is_refused_at_the_door() {
    let (mut res, _) = inf_vt::VtResidency::new(pool_cfg(PageFormat::Bc1, 32));
    let wrong = VtTextureDesc {
        tile_size: 64,
        border: 4,
        srgb: false,
        mips: full_pyramid(64, 64, 64, 4, false).mips,
    };
    assert!(matches!(
        res.register_texture(wrong),
        Err(inf_vt::VtError::PoolGeometryMismatch { .. })
    ));
}
