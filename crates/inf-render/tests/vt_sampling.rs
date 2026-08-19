//! **The VT sampling gate** (P26.3, extended P26.5): the shader's walk against
//! the CPU's — and, since P26.5, the residency heat-map's pixels and the
//! downlevel tier's.
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
        trilinear: false,
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
            hdr: false,
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
/// Written to run on **either** adapter tier through one call shape: with
/// `TEXTURE_COMPRESSION_BC` the page is the stored BC blob, without it the pool
/// is RGBA8 and the same tile arrives through `tile_rgba8` — the same door, one
/// format decision earlier. The comparison is exact either way: a
/// texture→buffer copy is a copy.
///
/// It takes the tier the *machine* has, so one run exercises one of them. The
/// transcode tier is covered on every adapter by `vt_pools.rs`'s
/// `the_transcode_fallback_uploads_on_any_adapter`, which forces `bc_tiles` off
/// rather than waiting for a BC-less machine.
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
// and ORM words in `InstanceRaw::pack`, collapsing the uv to a constant and
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
/// axes. The cube's uv stream (P26.5) maps that face onto the whole unit square,
/// so the framing above reaches uv ≈ [0.09, 0.91] × [0.27, 0.73] of it — and the
/// footprint works out under one texel of a 256² texture per pixel, so the
/// gradient mip resolves to **mip 0**, the level the arms below make resident.
///
/// The numbers are the ones the retired box projection produced, because a unit
/// cube is the shape it was exactly right for: `the_cube_uv_is_the_projection_it_replaces`
/// pins that identity, which is what keeps these arms measuring what they
/// measured before the stream landed.
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
    resident_library_of(gpu, ramp_container(256, srgb))
}

/// [`resident_library`] over any container — so an arm can hold one fixture's
/// *shading* fixed and vary only the texture's own content (P26.5).
fn resident_library_of(
    gpu: &GpuContext,
    bytes: Vec<u8>,
) -> (VtTextures, VtPools, inf_render::VtTextureSet) {
    let (mut lib, _) = VtTextures::new(pool_cfg(PageFormat::Rgba8, 64));
    lib.register_or_record(1, Arc::new(bytes))
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
            hdr: false,
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
///   does not (the uv varies across the face — a stream, or the projection it
///   replaced, that returns one value passes every other arm in this file);
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
        detail: 0,
        detail_scale_q8: 0,
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

/// **A handle is a per-registry index, and two projectors must compare by GUID.**
///
/// The arm above builds both registries in the SAME order, which is the case the
/// two hosts do not have. The P16.6 mirror law is explicit that **the editor
/// walks document order and the player walks `Guid` order** (see the MIRROR block
/// on `project_sky` in both hosts), so the day a projector registers textures as
/// it meets them, the same level mints DIFFERENT handles on the two sides — and
/// `VtTextureSet` is handles, so the two sets are unequal integers naming one
/// texture.
///
/// That is not a defect, and this arm exists so nobody fixes it into one. A
/// handle indexes the registry that minted it; each host's own indirection table
/// maps its own handle to the same content. What must hold — and what is asserted
/// here — is that **a GUID resolves to the same texture in both**, and what must
/// never be written is a cross-host comparison of handle NUMBERS.
///
/// Nothing registers today (no projector calls `VtTextures::register` outside
/// tests — the persisted material binding is P26.4's), so this is a pin placed
/// ahead of the wiring rather than a bug found behind it.
#[test]
fn two_projectors_in_different_orders_agree_by_guid_and_not_by_handle() {
    let build = |order: [u128; 2]| {
        let (mut lib, _) = VtTextures::new(pool_cfg(PageFormat::Rgba8, 256));
        for guid in order {
            // Extents differ per GUID, so a texture is identifiable by its own
            // descriptor and a mixed-up registry cannot pass by symmetry.
            let (w, h) = if guid == 11 { (256, 256) } else { (128, 64) };
            lib.register_or_record(
                guid,
                Arc::new(container(
                    w,
                    h,
                    guid == 11,
                    inf_material::TextureCompression::None,
                )),
            )
            .expect("registers");
        }
        let wants = lib.want_floor();
        let _ = lib.residency_mut().apply_wants(&wants);
        lib
    };
    let editor = build([11, 22]);
    let player = build([22, 11]);

    // The handles genuinely differ — without this the assertions below are the
    // same-order arm again under another name.
    assert_eq!(editor.handle(11).expect("registered").0, 0);
    assert_eq!(player.handle(11).expect("registered").0, 1);
    assert_ne!(
        editor.set_for(Some(11), None, None),
        player.set_for(Some(11), None, None),
        "the two orders minted the same handle — this arm is not testing two \
         orders any more, and the comment above it is now wrong"
    );

    // …and yet each names the same TEXTURE, which is the invariant that matters.
    for guid in [11u128, 22] {
        let (ha, hb) = (
            editor.handle(guid).expect("registered"),
            player.handle(guid).expect("registered"),
        );
        assert_eq!(
            editor.residency().desc(ha),
            player.residency().desc(hb),
            "GUID {guid} resolves to different textures in the two projectors"
        );
    }
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
        reconstruct_z: false,
        mips: full_pyramid(64, 64, 64, 4, false).mips,
    };
    assert!(matches!(
        res.register_texture(wrong),
        Err(inf_vt::VtError::PoolGeometryMismatch { .. })
    ));
}

// ── (f) THE MISSING LEG: the WGSL walk, on the GPU, with a non-zero trip count

/// **The fallback-residency GPU arm** — the P26.3 ledger's named remainder,
/// carried into P26.4 and closed here.
///
/// That ledger is explicit about what was missing and why it mattered: *"the
/// batch's claim that this 'runs the correct walk on the GPU' is not what
/// happens … the twin is CPU, the WGSL is pinned to the twin's spelling by the
/// source gate, and no test executes the WGSL tile-tree loop with a non-zero
/// trip count on a GPU (the lit-pixel arms make the whole pyramid resident, so
/// `got == m` and the loop body never runs). A fallback-residency GPU arm —
/// floor-only, every address resolving several levels up — is the missing leg."*
///
/// This runs `vt_resolve` — the **shipped function**, composed out of
/// `vt_sample.wgsl` exactly as the lit passes compose it, not a copy — over
/// every tile of mip 0 of a **1023² pyramid**, under a residency in which mip 0
/// is entirely absent. Every address therefore resolves several levels up, the
/// loop body runs on every thread, and the answer is compared with
/// `VtTextureDesc::ancestor` address by address.
///
/// 1023² is the extent, not 1024², because that is where the two derivations
/// PART: the container halves with `w / 2`, so 1023 sits over 511 sits over 255,
/// and `uv × 255` is not `texel / 4`. On a power-of-two pyramid the forbidden
/// derivation and the correct one agree everywhere and the arm would certify
/// nothing. The P26.2 audit measured 50 disagreeing addresses on 1023².
#[test]
fn the_wgsl_walk_runs_on_the_gpu_with_a_non_zero_trip_count() {
    let Some(gpu) = gpu_or_skip("the VT fallback walk") else {
        return;
    };
    // A pool with room for the coarse levels and nothing like enough for mip 0
    // (a 1023² pyramid's mip 0 is 8×8 = 64 tiles).
    let desc = full_pyramid(1023, 1023, 128, 4, false);
    let (mut res, _) = inf_vt::VtResidency::new(pool_cfg(PageFormat::Bc1, 24));
    let h = res.register_texture(desc.clone()).expect("registers");
    // Resident: levels 2 and coarser. Level 2 is 255 texels — TWO tiles a side —
    // so the walk lands on one of four tiles rather than being clamped to tile 0
    // by a one-tile level, which is the shape that hides every disagreement.
    let mut wants = Vec::new();
    for mip in 2..desc.mip_count() {
        let m = desc.mips[mip as usize];
        for y in 0..m.tiles_y {
            for x in 0..m.tiles_x {
                wants.push(inf_vt::VtWant::new(h, TileCoord::new(mip, x, y)));
            }
        }
    }
    let txn = res.apply_wants(&wants);
    assert_eq!(
        txn.deferred,
        0,
        "the coarse set did not fit: {}",
        txn.trace()
    );
    assert!(
        desc.mips[2].tiles_x > 1,
        "the resolved level is one tile wide, so the clamp hides every disagreement and this arm certifies nothing"
    );
    for y in 0..desc.mips[0].tiles_y {
        for x in 0..desc.mips[0].tiles_x {
            assert!(
                !res.is_resident(h, TileCoord::new(0, x, y)),
                "mip 0 is resident, so `got == m` and the loop body never runs"
            );
        }
    }

    // ── the shader: the shipped `vt_resolve`, composed the way a lit pass
    // composes it, plus a compute entry that writes what it returned.
    let vt = include_str!("../src/shaders/vt_sample.wgsl").replace("GROUP_ENV", "0");
    let src = format!(
        "{vt}
struct Probe {{ uvx: f32, uvy: f32, m: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> probes: array<Probe>;
@group(0) @binding(1) var<storage, read_write> walked: array<vec4<u32>>;
@compute @workgroup_size(64)
fn cs_probe(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&probes)) {{ return; }}
    let p = probes[i];
    let b = vt_table[VT_HEADER_WORDS + 0u];
    let r = vt_resolve(b, vec2<f32>(p.uvx, p.uvy), p.m);
    walked[i] = vec4<u32>(r.page, r.got, r.gtx, r.gty);
}}
"
    );

    // Every tile of mip 0, probed at its own centre.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Probe {
        uvx: f32,
        uvy: f32,
        m: u32,
        pad: u32,
    }
    let m0 = desc.mips[0];
    let mut probes = Vec::new();
    let mut addrs = Vec::new();
    // Three texels per tile per axis: its FIRST, its centre and its LAST.
    //
    // The first texel is the one that matters and the centre alone is not
    // enough — measured, in this arm's own first draft. The two derivations
    // agree comfortably inside a tile and part at its EDGE: mip 0's texel 512 is
    // the first texel of tile 4, and at the resolved level (255 texels) it is
    // 127.75, i.e. tile 0, while the clamped chain walks 4 → 2 → 1. A sweep of
    // centres alone passes with the forbidden derivation in place, which is
    // exactly the vacuity this arm exists to avoid.
    const WITHIN: [u32; 3] = [0, 64, 127];
    for y in 0..m0.tiles_y {
        for x in 0..m0.tiles_x {
            for oy in WITHIN {
                for ox in WITHIN {
                    let px = (x * 128 + ox).min(m0.width - 1) as f32 + 0.5;
                    let py = (y * 128 + oy).min(m0.height - 1) as f32 + 0.5;
                    probes.push(Probe {
                        uvx: px / m0.width as f32,
                        uvy: py / m0.height as f32,
                        m: 0,
                        pad: 0,
                    });
                    addrs.push(TileCoord::new(0, x, y));
                }
            }
        }
    }

    let pools = VtPools::new(&gpu.device, &gpu.queue, &res, false);
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vt-walk-probe"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let probe_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probes"),
        size: (probes.len() * std::mem::size_of::<Probe>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&probe_buf, 0, bytemuck::cast_slice(&probes));
    let out_size = (probes.len() * 16) as u64;
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("walk-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("walk-rb"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(true),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(false),
            count: None,
        },
    ];
    // The atlas / sampler / table at 14/15/16 — the same three bindings the env
    // group carries, visible to COMPUTE here.
    for mut e in VtPools::bind_group_layout_entries(14) {
        e.visibility = wgpu::ShaderStages::COMPUTE;
        entries.push(e);
    }
    let bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vt-walk-probe"),
            entries: &entries,
        });
    let mut bg_entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: probe_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: out_buf.as_entire_binding(),
        },
    ];
    bg_entries.extend(pools.bind_group_entries(14));
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vt-walk-probe"),
        layout: &bgl,
        entries: &bg_entries,
    });
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vt-walk-probe"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vt-walk-probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("cs_probe"),
            compilation_options: Default::default(),
            cache: None,
        });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vt-walk-probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((probes.len() as u32).div_ceil(64).max(1), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &rb, 0, out_size);
    gpu.queue.submit([enc.finish()]);
    let slice = rb.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("map walk readback");
    let got: Vec<[u32; 4]> = bytemuck::cast_slice::<u8, [u32; 4]>(&mapped).to_vec();
    drop(mapped);
    rb.unmap();

    // ── the comparison: every address, against `ancestor` and the residency.
    let mut trips = 0u32;
    for (i, at) in addrs.iter().enumerate() {
        let [page, resolved_mip, gtx, gty] = got[i];
        assert!(
            resolved_mip > 0,
            "{at:?}: the GPU resolved to mip 0, which is not resident"
        );
        trips += resolved_mip;
        let want = desc
            .ancestor(*at, resolved_mip)
            .unwrap_or_else(|| panic!("{at:?} has no ancestor at {resolved_mip}"));
        assert_eq!(
            (gtx, gty),
            (want.x, want.y),
            "{at:?} resolved at mip {resolved_mip}: the GPU walked to tile ({gtx},{gty}) and `VtTextureDesc::ancestor` says ({},{})",
            want.x,
            want.y
        );
        // …and the page it named is the page the residency put that tile in, so
        // the walk and the table agree about the atlas as well as the tile.
        let r = res.resolve(h, *at).expect("resolves");
        assert_eq!(page, r.slot, "{at:?}: wrong page");
        assert_eq!(want, r.tile, "{at:?}: the CPU twin disagrees with itself");
    }
    // ANTI-VACUITY, and the whole point of the arm: the loop body really ran,
    // several times per address. Zero here is the state the P26.3 ledger
    // described — the loop present and never executed.
    assert!(
        trips >= 2 * addrs.len() as u32,
        "the tile-tree loop ran {trips} times over {} addresses — it is not being exercised",
        addrs.len()
    );
}

// ── (g) THE DOOR ITSELF: `build_vt_level`, which no test had ever executed ───

/// **The registration door decides the pool's page format from the level, and
/// the mixed case really transcodes** (P26.4 audit).
///
/// `inf_render::build_vt_level` is clause 0. Both hosts call it, the P26.4
/// ledger spends a paragraph on its three-branch pool-format ruling, and until
/// this arm **nothing executed it**: `projector_mirror` greps its *name* out of
/// two source files, and `phase26_gate` deliberately spells the format decision
/// by hand ("the same decision `build_vt_level` makes from the payload's own
/// header, spelled here because this arm builds the registry without a device").
///
/// Measured before it was written: a mutation making a MIXED level take a BC
/// pool, and one making every level take RGBA8, both survived the entire
/// workspace. `inf_vt::stored_page_format` — the ruling's only input — shipped
/// with one caller and no arm at all.
///
/// The ruling matters because one pool slot is one size for every tile of every
/// texture in it: a BC1 pool cannot hold a BC3 tile, the fetch is the wrong
/// length, `VtPools::apply` writes its deterministic zero page and the surface is
/// silently wrong. So the third assertion here is not about the reported format
/// at all — it is that every page of a mixed level's floor actually LANDS.
///
/// Both adapter tiers, on the P25.2 doctrine: a runner without
/// `TEXTURE_COMPRESSION_BC` collapses branches 1 and 2 onto RGBA8, which is
/// exactly the transcode tier this ruling exists to serve, so the discriminating
/// assertion is guarded and says so rather than being skipped.
#[test]
fn the_registration_door_decides_the_pool_format_from_the_level() {
    let Some(gpu) = gpu_or_skip("the VT registration door") else {
        return;
    };
    let settings = inf_render::RenderSettings::default();
    let has_bc = pool_format(&settings, PageFormat::Bc1) == PageFormat::Bc1;

    // The ruling's input, which had no arm: the header's format, without a reader.
    for (compression, want) in [
        (inf_material::TextureCompression::None, PageFormat::Rgba8),
        (inf_material::TextureCompression::Bc1, PageFormat::Bc1),
        (inf_material::TextureCompression::Bc3, PageFormat::Bc3),
    ] {
        assert_eq!(
            inf_vt::stored_page_format(&container(96, 96, true, compression)),
            Some(want),
            "a {compression:?} container does not declare {want:?}"
        );
    }
    // Anything that is not a readable v2 container is `None` — the same answer
    // `TiledTextureReader::new` gives the registration a moment later.
    assert_eq!(inf_vt::stored_page_format(&[]), None);
    assert_eq!(inf_vt::stored_page_format(&[0u8; 64]), None);

    let level = |maps: Vec<(u128, Vec<u8>)>| {
        let mut mats: std::collections::BTreeMap<u128, inf_render::VtMaterialMaps> =
            std::collections::BTreeMap::new();
        for (i, (g, _)) in maps.iter().enumerate() {
            mats.insert(
                100 + i as u128,
                inf_render::VtMaterialMaps {
                    albedo: Some(*g),
                    normal: None,
                    orm: None,
                    detail: None,
                    detail_scale_q8: 0,
                },
            );
        }
        let by_guid: std::collections::BTreeMap<u128, Vec<u8>> = maps.into_iter().collect();
        inf_render::build_vt_level(
            &gpu.device,
            &gpu.queue,
            &settings,
            inf_vt::DEFAULT_VT_BUDGET_BYTES,
            &mats,
            |g| {
                by_guid
                    .get(&g)
                    .cloned()
                    .map(|b| Arc::new(b) as Arc<dyn inf_render::VtTileSource>)
            },
        )
    };

    // 1. ONE BC format across the level → that format, clamped by the adapter.
    let (_uniform, _uniform_pools, uniform) = level(vec![
        (
            1,
            container(256, 256, true, inf_material::TextureCompression::Bc1),
        ),
        (
            2,
            container(128, 128, false, inf_material::TextureCompression::Bc1),
        ),
    ])
    .expect("a bound level builds");
    assert_eq!(uniform.textures, 2, "the door registered the wrong count");
    assert_eq!(uniform.refused, 0, "a bound texture was refused");
    assert_eq!(
        uniform.pool_format,
        Some(pool_format(&settings, PageFormat::Bc1)),
        "a uniformly-BC1 level did not take the BC1 pool the adapter allows"
    );

    // 2. A level that MIXES BC1 and BC3 → RGBA8, on EVERY adapter.
    let (mut mixed_lib, mut mixed_pools, mixed) = level(vec![
        (
            1,
            container(256, 256, true, inf_material::TextureCompression::Bc1),
        ),
        (
            2,
            container(128, 128, true, inf_material::TextureCompression::Bc3),
        ),
    ])
    .expect("a mixed level builds");
    assert_eq!(
        mixed.pool_format,
        Some(PageFormat::Rgba8),
        "a level mixing BC1 and BC3 took a BC pool: one slot is one size, so the \
         other format's fetch is the wrong length and the mirror writes a zero page"
    );
    assert_eq!(mixed.textures, 2);
    assert_eq!(
        mixed.refused, 0,
        "the transcode tier refused a bound texture"
    );

    // …and ASSERT THE WORLD: the floor pages and NOT ONE page is missing, which
    // is what "RGBA8 is the format every container transcodes into" has to mean.
    let floor = mixed_lib.want_floor();
    assert!(!floor.is_empty(), "the mixed level's floor wanted nothing");
    let (txn, report) = mixed_lib.sync(&gpu.device, &gpu.queue, &mut mixed_pools, &floor);
    assert_eq!(txn.deferred, 0, "the floor did not fit: {}", txn.trace());
    assert!(
        report.missing.is_empty(),
        "{} page(s) of a mixed level did not transcode into the RGBA8 pool",
        report.missing.len()
    );
    assert!(report.pages > 0, "the mixed level admitted nothing");

    // ANTI-VACUITY: on a BC adapter branches 1 and 2 give DIFFERENT answers, so
    // the mixed ruling is not branch 1 wearing another name.
    if has_bc {
        assert_ne!(
            uniform.pool_format, mixed.pool_format,
            "the two branches agree, so the mixed ruling asserts nothing here"
        );
    } else {
        eprintln!(
            "NOTE: no TEXTURE_COMPRESSION_BC on this adapter, so both branches are \
             RGBA8; the mixed ruling's transcode is still asserted above"
        );
    }

    // 3. A level whose every bound texture is unreadable is `None` — not a panic,
    //    and not an empty registry pretending to be a level.
    assert!(level(vec![(1, vec![0u8; 16])]).is_none());
    // …and a level that binds nothing at all is `None` too, which is the
    // textureless command stream all 50 committed goldens recorded.
    assert!(level(vec![]).is_none());
}

// -- (e) the residency heat-map + the tier budget (P26.5) -------------------

/// Draw one frame of [`face_scene`] in `mode` and return its rgba.
///
/// `set_vt_pools` rather than `set_vt_level` on purpose: it binds the atlas and
/// the table and leaves `vt_textures` unset, so `EngineRenderer::vt_stream`
/// returns immediately and the residency the heat-map paints is **exactly the
/// one this fixture built**. A streaming loop running under the arm would
/// re-admit mip 0 the moment the camera justified it, and the fallback fixture
/// below would quietly become the resident one.
fn render_face_mode(
    gpu: &GpuContext,
    pools: Option<VtPools>,
    set: inf_render::VtTextureSet,
    mode: inf_render::ViewMode,
) -> Vec<u8> {
    let target = inf_render::HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    renderer.set_vt_pools(pools);
    renderer.set_view_mode(mode);
    assert_eq!(renderer.view_mode(), mode, "the view mode was degraded");
    renderer.render(gpu, &face_scene(set), &face_view(), &target.view, (FW, FH));
    target.read_rgba(gpu).expect("readback")
}

/// A registry holding one 256 texture with **only its coarsest levels**
/// resident -- the analytic floor standing alone, which is what a pool under
/// pressure looks like and what the heat-map exists to show.
fn floor_only_library(gpu: &GpuContext) -> (VtTextures, VtPools, inf_render::VtTextureSet) {
    let (mut lib, _) = VtTextures::new(pool_cfg(PageFormat::Rgba8, 64));
    lib.register_or_record(1, Arc::new(ramp_container(256, false)))
        .expect("the fixture registers");
    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let floor = lib.want_floor();
    let (txn, report) = lib.sync(&gpu.device, &gpu.queue, &mut pools, &floor);
    assert_eq!(txn.deferred, 0, "the floor did not fit");
    assert!(report.missing.is_empty());
    // ANTI-VACUITY: mip 0 really is ABSENT, or the "fallback" fixture is the
    // resident one under another name.
    let desc = lib
        .residency()
        .desc(VtTextureHandle(0))
        .expect("registered")
        .clone();
    assert!(
        !lib.residency()
            .is_resident(VtTextureHandle(0), TileCoord::new(0, 0, 0)),
        "the floor admitted mip 0"
    );
    assert!(
        desc.mip_count() >= 4,
        "the pyramid is too short to fall back"
    );
    let set = lib.set_for(Some(1), None, None);
    assert!(!set.is_none(), "the fixture never went warm");
    (lib, pools, set)
}

/// The `(mean red, mean green)` of a frame.
fn rg_means(rgba: &[u8]) -> (f64, f64) {
    let n = (rgba.len() / 4) as f64;
    let r: u64 = rgba.chunks_exact(4).map(|p| u64::from(p[0])).sum();
    let g: u64 = rgba.chunks_exact(4).map(|p| u64::from(p[1])).sum();
    (r as f64 / n, g as f64 / n)
}

/// **The residency heat-map paints what is resident** (P26.5, ROADMAP clause 1).
///
/// The debug view mode `VtPopIn` and `VtStats::summary` were declared for in
/// P26.2 and P26.4 -- *"ships before its caller, deliberately: P26.5's residency
/// heat-map ... is what reads it"* -- asserted where a debug view can actually
/// be wrong: on the FRAME.
///
/// Three fixtures over one scene, one camera and one texture, differing only in
/// what is resident:
///
/// * whole pyramid resident => the face is **green** (`got == m`);
/// * the analytic floor alone, mip 0 absent => the face is **red/orange** (the
///   walk resolves several levels up);
/// * no pool at all => **grey**, which is the reading that tells an author a
///   surface they believed was textured is not bound -- a fact no amount of
///   staring at a lit frame distinguishes from a texture that happens to look
///   like its scalar colour.
///
/// The channels are compared against each other rather than against constants:
/// the ramp is written in linear space and the frame comes back through ACES and
/// an sRGB target, so the *values* are the tonemapper's and the *ordering* is the
/// claim.
#[test]
fn the_residency_heat_map_paints_what_is_resident() {
    let Some(gpu) = gpu_or_skip("the VT residency heat-map") else {
        return;
    };
    let (_lib, resident_pools, set) = resident_library(&gpu, false);
    let (_lib2, floor_pools, floor_set) = floor_only_library(&gpu);

    let hot = render_face_mode(
        &gpu,
        Some(resident_pools),
        set,
        inf_render::ViewMode::VtResidency,
    );
    let cold = render_face_mode(
        &gpu,
        Some(floor_pools),
        floor_set,
        inf_render::ViewMode::VtResidency,
    );
    let bare = render_face_mode(
        &gpu,
        None,
        inf_render::VtTextureSet::NONE,
        inf_render::ViewMode::VtResidency,
    );

    let (hot_r, hot_g) = rg_means(&hot);
    let (cold_r, cold_g) = rg_means(&cold);
    let (bare_r, bare_g) = rg_means(&bare);

    assert!(
        hot_g > hot_r * 2.0,
        "a fully resident surface is not green (r {hot_r:.1}, g {hot_g:.1})"
    );
    assert!(
        cold_r > cold_g * 2.0,
        "a surface at the floor is not red (r {cold_r:.1}, g {cold_g:.1}) -- the \
         heat-map is not reading the residency"
    );
    // Grey: darker than either verdict in that verdict's own channel, and with
    // no channel dominating. A heat-map that painted an unbound surface green
    // would say "everything is fine" about a surface that samples nothing at
    // all. Compared against the two measured frames rather than against a
    // constant, because the ramp is authored in linear space and read back
    // through ACES and an sRGB target -- 0.06 linear arrives at ~69/255, and a
    // literal here would be a pin on the tonemapper.
    assert!(
        bare_g * 2.0 < hot_g,
        "an unbound surface is as green as a resident one (bare g {bare_g:.1}, \
         resident g {hot_g:.1})"
    );
    assert!(
        bare_r * 2.0 < cold_r,
        "an unbound surface is as red as one at the floor (bare r {bare_r:.1}, \
         floor r {cold_r:.1})"
    );
    assert!(
        (bare_r - bare_g).abs() < 8.0,
        "an unbound surface is tinted (r {bare_r:.1}, g {bare_g:.1})"
    );

    // ANTI-VACUITY: the mode does something. The same fixture in `Lit` must NOT
    // produce this picture -- otherwise the ordering above could be a property
    // of the ramp texture rather than of the overlay.
    let lit = render_face_mode(
        &gpu,
        None,
        inf_render::VtTextureSet::NONE,
        inf_render::ViewMode::Lit,
    );
    assert_ne!(lit, bare, "VtResidency rendered the same frame as Lit");
}

/// **The one line a host logs** (P26.5) -- `VtStats::summary` and
/// `VtPopIn::summary`'s non-gate reader, which both shipped promising this.
///
/// `None` on a textureless renderer is the load-bearing half: a host that prints
/// nothing says *"this level has no virtual textures"*, and one that prints
/// zeros says *"it has some and they are doing nothing"*. Those are different
/// facts and an author reading an Output Log is entitled to both.
#[test]
fn the_vt_summary_is_a_line_a_host_can_log() {
    let Some(gpu) = gpu_or_skip("the VT summary line") else {
        return;
    };
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    assert!(
        renderer.vt_summary().is_none(),
        "a renderer with no VT level produced a summary"
    );

    // A COLD registry: registered, nothing paged. The renderer's own streaming
    // loop is what admits the floor, so `admits` below is a measurement of the
    // loop rather than of this fixture — the same distinction
    // `the_renderer_runs_the_streaming_loop_every_frame` rests on.
    let (mut lib, _) = VtTextures::new(pool_cfg(PageFormat::Rgba8, 64));
    lib.register_or_record(1, Arc::new(ramp_container(256, false)))
        .expect("the fixture registers");
    let pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    assert_eq!(
        lib.residency().stats().admits,
        0,
        "the fixture paged something before the renderer did"
    );
    renderer.set_vt_level(Some((lib, pools)));
    // Two frames: the first warms the registry, the second draws a textured
    // surface through it — which is what puts a non-zero coverage list into the
    // loop rather than only the camera-free floor.
    for _ in 0..2 {
        let set = renderer
            .vt_textures()
            .map(|l| l.set_for(Some(1), None, None))
            .unwrap_or(inf_render::VtTextureSet::NONE);
        renderer.render(&gpu, &face_scene(set), &face_view(), &target.view, (FW, FH));
    }
    let line = renderer.vt_summary().expect("a level has a summary");
    // It names BOTH halves -- the residency's state and the streamer's history --
    // because a pool that is full and a streamer that is thrashing look
    // identical in either one alone.
    for word in [
        "vt residency:",
        "slots resident",
        "vt streaming:",
        "admits",
        "feedback",
    ] {
        assert!(
            line.contains(word),
            "the summary does not say {word:?}: {line}"
        );
    }
    // ...and the streamer really ran, so the numbers are a measurement.
    assert_eq!(
        renderer.vt_pop_in().frames,
        2,
        "the streaming loop did not run once per frame, so the line is a row of \
         zeros"
    );
    assert!(
        renderer.vt_pop_in().admits > 0,
        "nothing was admitted -- the anti-vacuity number `VtPopIn` carries for \
         exactly this reason"
    );
}

// -- (f) the DOWNLEVEL tier's pixels (P26.5, the P26.4 audit's remainder) ----

/// The vgeom fixture: one dense grid quad, stood up in world XY so it faces the
/// camera and fills the frame. Its `VgeomVertex`es carry a real authored uv
/// (`inf_vgeom::test_support` writes one per grid corner), which is what the
/// classic tier now reads -- P26.5 -- and what the meshlet tier has read since
/// P13.1b.
fn vgeom_face_scene(
    mesh: &std::sync::Arc<inf_render::VgeomMesh>,
    set: inf_render::VtTextureSet,
) -> inf_render::RenderScene {
    const ASSET: u128 = 0x2605_0000_0dc1_0000;
    let mut scene = inf_render::RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![inf_render::VgeomAsset::from_mesh(ASSET, mesh).expect("index the vmesh")],
        ..Default::default()
    };
    let mut inst = inf_render::VgeomInstance::lit(
        ASSET,
        glam::DVec3::ZERO,
        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        glam::Vec3::splat(8.0),
        [1.0, 1.0, 1.0, 1.0],
        1,
    );
    inst.vt = set;
    scene.vgeom_instances.push(inst);
    scene.mark_dirty();
    scene
}

/// Draw one frame of the vgeom fixture with the meshlet path on or off.
fn render_vgeom_face(
    gpu: &GpuContext,
    mesh: &std::sync::Arc<inf_render::VgeomMesh>,
    pools: Option<VtPools>,
    set: inf_render::VtTextureSet,
    meshlets: bool,
) -> Vec<u8> {
    let target = inf_render::HeadlessTarget::new(gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(gpu, inf_render::HEADLESS_FORMAT);
    let mut settings = inf_render::RenderSettings::default();
    // The exact complement the two nodes are gated on: exactly one of them draws
    // this content, so "the classic frame" and "the meshlet frame" are two
    // rasterizers over one scene rather than two scenes.
    settings.vgeom.enabled = meshlets;
    settings.vgeom.occlusion = false;
    settings.vgeom.two_pass = false;
    renderer.set_settings(settings);
    renderer.set_vt_pools(pools);
    renderer.render(
        gpu,
        &vgeom_face_scene(mesh, set),
        &face_view(),
        &target.view,
        (FW, FH),
    );
    target.read_rgba(gpu).expect("readback")
}

/// A flat container of one colour — the CONTROL for "does the uv vary".
fn flat_container(n: u32, red: u8) -> Vec<u8> {
    inf_material::build_tiled_texture(
        std::iter::repeat_n([red, 40, 200, 255], (n * n) as usize)
            .flatten()
            .collect(),
        n,
        n,
        inf_material::TextureImportSettings {
            srgb: false,
            generate_mips: true,
            compression: inf_material::TextureCompression::None,
            hdr: false,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

/// The spread of the per-pixel red **RATIO** `a / b`, in 1/256ths, over the
/// pixels where `b` carries a surface at all.
///
/// A texture modulates a lit surface **multiplicatively**, so the *difference*
/// between a textured and an untextured frame varies wherever the SHADING
/// varies — even for a texture that is one flat colour. Measured: a
/// `uv: [0.0; 2]` mutation in `classic_vgeom` — which collapses a whole mesh
/// onto one texel — left a delta spread of 74 and passed a difference-based arm
/// unharmed. A ratio divides the shading out, so what is left is the texture's
/// own spatial variation, which is the thing a uv stream is FOR.
///
/// Pixels where the reference is near black are skipped: a ratio against nothing
/// is noise, and the sky is not the surface under test.
fn red_ratio_spread(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len());
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    let mut n = 0u32;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if y[0] < 24 {
            continue;
        }
        let r = (u32::from(x[0]) * 256) / u32::from(y[0]);
        lo = lo.min(r);
        hi = hi.max(r);
        n += 1;
    }
    assert!(n > 100, "only {n} pixels carried a surface to divide by");
    hi - lo
}

/// **The DOWNLEVEL tier draws a textured mesh, in texels** (P26.5).
///
/// The P26.4 audit's remainder, verbatim: *"The downlevel tier has no pixel arm.
/// `pack_vgeom_instance`'s dropped texture set is fixed and pinned on
/// `InstanceRaw`'s words; nothing renders a textured vgeom surface with
/// `RenderSettings.vgeom.enabled = false`, so 'the two tiers sample the same
/// thing' is asserted on instance bytes rather than on texels."*
///
/// This is that arm, and it now carries a second claim the audit could not have
/// asked for: P26.5 gave `MeshVertex` a uv and `classic_vgeom` stopped
/// **dropping** `VgeomVertex::uv` on the way into it. So the classic tier
/// samples the artist's parametrization rather than a box projection, and the
/// two tiers can be compared on the picture rather than on the intent.
///
/// Four frames, one scene, one camera, one lighting:
///
/// * classic (`vgeom.enabled = false`) with a **ramp** texture bound;
/// * classic with a **flat** texture of the same size, format and residency —
///   the control that separates "a texture reached the pixels" from "the uv
///   varies across them";
/// * classic with `VtTextureSet::NONE` — the reference to divide the shading out
///   against;
/// * meshlet (`vgeom.enabled = true`) with the ramp — textured too, which is the
///   "both tiers sample the same thing" half.
///
/// **Measured as a ratio, not a difference**, and the flat control is why. A
/// texture modulates a lit surface multiplicatively, so `textured − untextured`
/// varies wherever the *shading* varies — for a flat texture as much as for a
/// ramp. Measured: a `uv: [0.0; 2]` mutation in `classic_vgeom`, which collapses
/// the whole mesh onto one texel, left a difference-spread of 74 and survived a
/// difference-based arm untouched. Dividing by the untextured frame removes the
/// shading and leaves the texture's own spatial variation, which is the thing a
/// uv stream exists for.
///
/// The two tiers' frames are **not** compared pixel for pixel: they rasterize
/// different LOD cuts of one DAG, which is the whole point of having two. What
/// is compared is that each one carries the texture's own variation.
#[test]
fn the_downlevel_tier_draws_a_textured_mesh_in_texels() {
    let Some(gpu) = gpu_or_skip("the downlevel VT pixel arm") else {
        return;
    };
    // A FLAT grid (amplitude 0), deliberately: a displaced one is lit
    // differently at every texel, and a tonemapped, sRGB-encoded frame is not a
    // linear function of radiance — so neither a difference nor a ratio against
    // an untextured frame divides that shading out. Measured, twice: a
    // `uv: [0.0; 2]` mutation survived a difference-based arm (delta spread 74),
    // and a FLAT texture scored 81/256 of ratio spread against a ramp's 112,
    // which is not a measurement of anything. With uniform shading the frame's
    // own spread IS the texture's, and the mesh still carries the same real
    // authored uv per grid corner.
    let mesh = std::sync::Arc::new(inf_vgeom::test_support::build_grid(
        24,
        0.0,
        inf_vgeom::test_support::GridNormals::Analytic,
    ));
    assert!(
        mesh.vertices.iter().any(|v| v.uv != [0.0, 0.0]),
        "the fixture mesh carries no uv, so nothing below can measure one"
    );

    let (_lib, pools, set) = resident_library(&gpu, false);
    let classic = render_vgeom_face(&gpu, &mesh, Some(pools), set, false);
    let (_flib, fpools, fset) = resident_library_of(&gpu, flat_container(256, 128));
    let flat = render_vgeom_face(&gpu, &mesh, Some(fpools), fset, false);
    let bare = render_vgeom_face(&gpu, &mesh, None, inf_render::VtTextureSet::NONE, false);

    let (_, _, bmean) = red_stats(&bare);
    let (_, _, cmean) = red_stats(&classic);
    let ramp_ratio = red_ratio_spread(&classic, &bare);
    let flat_ratio = red_ratio_spread(&flat, &bare);
    println!(
        "downlevel VT: bare mean {bmean:.1}, classic mean {cmean:.1}, ratio spread \
         ramp {ramp_ratio} vs flat {flat_ratio} (1/256ths)"
    );

    // ANTI-VACUITY FIRST: the untextured mesh is on screen and lit, or every
    // comparison below is between two skies.
    assert!(
        bmean > 20.0,
        "the untextured vgeom surface never reached the frame (mean red {bmean:.1})"
    );
    assert_ne!(
        bare, classic,
        "the texture changed no pixel on the classic tier"
    );
    // THE CLAIM: the ramp's modulation varies across the surface and the flat
    // texture's does not. A dropped texture set, and a uv collapsed to a
    // constant, both land on the right of this inequality.
    assert!(
        ramp_ratio > flat_ratio * 3,
        "the classic tier's ramp modulates the surface by {ramp_ratio}/256 across \
         it and a FLAT texture by {flat_ratio}/256 — the uv is not varying, which \
         is what a dropped `VgeomVertex::uv` looks like from here"
    );
    // …and the flat control really is nearly flat, so the inequality above is a
    // measurement rather than a comparison of two large numbers.
    assert!(
        flat_ratio < 24,
        "a FLAT texture modulates the surface by {flat_ratio}/256 across it, so \
         the ratio is not dividing the shading out and this arm measures nothing"
    );
    // **And the sample did not COLLAPSE onto one texel.** A ratio is a poor
    // instrument near zero: with the uv dead, every fragment reads texel (0, 0)
    // of a left-to-right ramp — which is black — and the surviving handful of
    // near-black pixels quantize into a wide-looking ratio (measured: mean 5.5
    // against the untextured 189.1, ratio spread 46, which passed the
    // inequality above). A ramp averages to something like half its surface, so
    // a textured frame that is 3 % of the untextured one is not a texture, it is
    // one texel.
    assert!(
        cmean > bmean * 0.15,
        "the textured surface is {cmean:.1} against the untextured {bmean:.1} — \
         every fragment is reading ONE texel, which is what a uv stream that \
         never arrived looks like"
    );

    // …and the MESHLET tier, over the same scene, is textured too. This is the
    // half `projector_mirror` cannot see: the third spelling of the texture set
    // lived inside `inf-render`, downstream of both hosts.
    let (_lib2, pools2, set2) = resident_library(&gpu, false);
    let meshlet = render_vgeom_face(&gpu, &mesh, Some(pools2), set2, true);
    let (_, _, mmean) = red_stats(&meshlet);
    let meshlet_ratio = red_ratio_spread(&meshlet, &bare);
    if mmean <= 20.0 {
        // The meshlet path needs compute + indirect draws; an adapter that
        // cannot run it draws nothing here, and that is a capability fact
        // rather than a failure. Said out loud rather than skipped silently.
        eprintln!(
            "NOTE: the meshlet tier drew nothing on this adapter (mean red \
             {mmean:.1}); the classic half above still ran"
        );
        return;
    }
    assert!(
        meshlet_ratio > flat_ratio * 3,
        "the meshlet tier's texture modulation spans only {meshlet_ratio}/256 \
         against a flat texture's {flat_ratio} while the classic tier's spans \
         {ramp_ratio} — the two tiers do not sample the \
         same thing"
    );
}

// ── (g) the shipped seam: a host that commits a camera (P28.4 audit) ─────────

/// **THE SHIPPED PREDICTIVE PATH, ARMED** (P28.4 audit) — the first arm anywhere
/// that runs `commit_camera → camera_history → prediction() → speculative_wants`
/// through `EngineRenderer` rather than around it.
///
/// P28.4 proved its two producers as *functions*: `whip_pan.rs` calls
/// `inf_math::dead_reckon` and `inf_render::speculative_wants` directly and
/// never builds a renderer, and `commit_camera` had exactly one caller in the
/// tree (`inf_player::window`) and no test at all. Measured in the audit's
/// mutation matrix: **`EngineRenderer::prediction()` hard-wired to `None` —
/// which turns the whole feature off in both consumers — left all twenty-two of
/// this crate's test binaries green.** The gate proved the parts and nothing
/// proved the wiring.
///
/// So this asserts the wiring and only the wiring, against the one counter that
/// can see it: `VtPopIn::predict_wants`, which until now had no reader outside
/// the module that writes it. The camera path is a slow yaw at the fixed step —
/// it does not need to be a whip-pan, because what is being measured is that a
/// committed pose reaches the want set, and `whip_pan.rs` owns the question of
/// what the want set is worth.
///
/// **The `set_for` warm rule is why the surface is re-read every frame.** A
/// texture is not in a `VtTextureSet` until a transaction has warmed it, so a
/// set sampled before the first render is `NONE`, the coverage list is empty and
/// the camera-driven half of every want class — floor and speculation alike — is
/// empty with it. A fixture that takes the set once measures the camera-free
/// floor and calls it a scene.
#[test]
fn a_committed_camera_reaches_the_shipped_streaming_loop() {
    let Some(gpu) = gpu_or_skip("the committed-camera seam") else {
        return;
    };
    let target = inf_render::HeadlessTarget::new(&gpu, FW, FH);
    let mut renderer = inf_render::EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    let (mut lib, _) = VtTextures::new(pool_cfg(PageFormat::Rgba8, 64));
    lib.register_or_record(1, Arc::new(ramp_container(256, false)))
        .expect("the fixture registers");
    let pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    renderer.set_vt_level(Some((lib, pools)));

    // An empty history is the real enable flag, and it fails safe: this is the
    // editor viewport's state, and every gate that existed before P28.4.
    assert!(
        renderer.prediction().is_none(),
        "a renderer nobody committed a camera to predicted something"
    );
    assert!(renderer.camera_history().is_empty());

    let view_at = |tick: u64| {
        let a = 0.02 * tick as f64;
        inf_render::RenderView {
            forward: glam::Vec3::new(inf_math::psin64(a) as f32, 0.0, -inf_math::pcos64(a) as f32)
                .normalize(),
            ..face_view()
        }
    };
    let draw = |r: &mut inf_render::EngineRenderer, v: &inf_render::RenderView| {
        let set = r
            .vt_textures()
            .map(|l| l.set_for(Some(1), None, None))
            .unwrap_or(inf_render::VtTextureSet::NONE);
        r.render(&gpu, &face_scene(set), v, &target.view, (FW, FH));
        !set.is_none()
    };

    // Three frames with no committed camera. The control, and the anti-vacuity
    // for the count below: the surface has to be *in* the coverage list by the
    // end of it, or "no speculative want" is a statement about an empty scene.
    let mut textured = false;
    for tick in 0..3u64 {
        textured |= draw(&mut renderer, &view_at(tick));
    }
    assert!(
        textured,
        "the fixture never warmed its texture, so nothing was ever covered"
    );
    assert!(
        renderer.vt_pop_in().floor_wants > 3,
        "the camera-driven floor is empty — the coverage list never had the \
         surface in it and this fixture cannot see a speculation either"
    );
    assert_eq!(
        renderer.vt_pop_in().predict_wants,
        0,
        "the loop speculated with an empty history — `None` does not mean `None`"
    );

    // …and now a host that commits at its fixed step, the way `inf_player`'s
    // window does: the SIM's step count, once per frame.
    for tick in 0..8u64 {
        let v = view_at(tick);
        assert!(
            renderer.commit_camera(tick, &v),
            "tick {tick} was refused by a history that should have taken it"
        );
        draw(&mut renderer, &v);
    }
    assert_eq!(
        renderer.camera_history().len(),
        inf_math::PREDICT_HISTORY,
        "the window did not fill or did not bound"
    );
    assert!(renderer.prediction().is_some());
    assert!(
        renderer.vt_pop_in().predict_wants > 0,
        "a committed camera produced no speculative want — `commit_camera` and \
         the streaming loop are not connected"
    );

    // **The refusal reaches a host**, which is the one thing
    // `CameraHistory::refused` is for: a frame that ran no fixed step commits
    // nothing, and says so rather than appending twice for one step.
    let seen = renderer.vt_pop_in().predict_wants;
    assert!(
        !renderer.commit_camera(7, &view_at(7)),
        "a tick that did not advance was taken"
    );
    assert_eq!(renderer.camera_history().refused(), 1);

    // **The knob is the knob.** With the predictor cleared the loop goes back to
    // exactly the pre-P28.4 want set while the history stays full — so this is
    // the settings gate, tested apart from the empty-history one above.
    let mut off = *renderer.settings();
    off.stream.predict.enabled = false;
    renderer.set_settings(off);
    assert!(renderer.prediction().is_none());
    draw(&mut renderer, &view_at(8));
    assert_eq!(
        renderer.vt_pop_in().predict_wants,
        seen,
        "a disabled predictor still offered wants"
    );

    // …and a cut clears the window rather than spanning it.
    renderer.reset_camera_history();
    assert!(renderer.camera_history().is_empty());
    assert!(renderer.prediction().is_none());
}

// ── (h) WAVE T: the two sampling channels, executed on the GPU ──────────────

/// A pool config with Wave T's trilinear flag.
fn pool_cfg_tri(format: PageFormat, pages: u64, trilinear: bool) -> VtPoolConfig {
    VtPoolConfig {
        trilinear,
        ..pool_cfg(format, pages)
    }
}

/// A tiled container from raw RGBA8, uncompressed, with a full mip chain.
fn raw_container(n: u32, mut texel: impl FnMut(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((n * n * 4) as usize);
    for y in 0..n {
        for x in 0..n {
            rgba.extend_from_slice(&texel(x, y));
        }
    }
    inf_material::build_tiled_texture(
        rgba,
        n,
        n,
        inf_material::TextureImportSettings {
            srgb: false,
            generate_mips: true,
            compression: inf_material::TextureCompression::None,
            hdr: false,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

/// Register `payloads` (GUID = index + 1) and make **every tile of every level**
/// resident, so nothing below is a statement about the pyramid's tail.
fn resident_many(
    gpu: &GpuContext,
    payloads: Vec<Vec<u8>>,
    trilinear: bool,
) -> (VtTextures, VtPools) {
    let (mut lib, _) = VtTextures::new(pool_cfg_tri(PageFormat::Rgba8, 256, trilinear));
    for (i, bytes) in payloads.into_iter().enumerate() {
        lib.register_or_record(i as u128 + 1, Arc::new(bytes))
            .expect("the fixture registers");
    }
    let mut pools = VtPools::new(&gpu.device, &gpu.queue, lib.residency(), false);
    let mut wants = Vec::new();
    for t in 0..lib.residency().texture_count() {
        let h = VtTextureHandle(t as u32);
        let desc = lib.residency().desc(h).expect("registered").clone();
        for m in 0..desc.mip_count() {
            let g = desc.mips[m as usize];
            for y in 0..g.tiles_y {
                for x in 0..g.tiles_x {
                    wants.push(inf_vt::VtWant::new(h, TileCoord::new(m, x, y)));
                }
            }
        }
    }
    let (txn, report) = lib.sync(&gpu.device, &gpu.queue, &mut pools, &wants);
    assert_eq!(txn.deferred, 0, "the fixture pyramids did not fit");
    assert!(
        report.missing.is_empty(),
        "{} pages missing",
        report.missing.len()
    );
    (lib, pools)
}

/// One `vt_surface` probe: the three map words, a uv, and the two derivatives.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceProbe {
    maps: [u32; 4],
    uv: [f32; 2],
    dd: [f32; 2],
}

/// Run the **shipped** `vt_surface` on the GPU for every probe and read back
/// `(normal_ts, roughness)` and `(albedo, alpha)`.
///
/// The `vt_sample.wgsl` in this shader is the file the lit passes compose, not a
/// transliteration — the P26.4 rule that a twin agreeing with a shader nobody
/// ran is a twin agreeing with itself.
fn run_surface_probe(
    gpu: &GpuContext,
    pools: &VtPools,
    probes: &[SurfaceProbe],
) -> Vec<([f32; 4], [f32; 4])> {
    let vt = include_str!("../src/shaders/vt_sample.wgsl").replace("GROUP_ENV", "0");
    let src = format!(
        "{vt}
struct SProbe {{ maps: vec4<u32>, uv: vec2<f32>, dd: vec2<f32> }};
@group(0) @binding(0) var<storage, read> probes: array<SProbe>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;
@compute @workgroup_size(64)
fn cs_surface(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&probes)) {{ return; }}
    let p = probes[i];
    let s = vt_surface(p.maps.xyz, p.uv,
                       vec2<f32>(p.dd.x, 0.0), vec2<f32>(0.0, p.dd.y),
                       vec3<f32>(1.0, 1.0, 1.0), 1.0, 0.0, 0.5);
    outp[i * 2u] = vec4<f32>(s.normal_ts, s.roughness);
    outp[i * 2u + 1u] = vec4<f32>(s.albedo, s.alpha);
}}
"
    );
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vt-surface-probe"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
    let in_size = std::mem::size_of_val(probes) as u64;
    let probe_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sprobes"),
        size: in_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&probe_buf, 0, bytemuck::cast_slice(probes));
    let out_size = (probes.len() * 2 * 16) as u64;
    let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface-rb"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(true),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(false),
            count: None,
        },
    ];
    for mut e in VtPools::bind_group_layout_entries(14) {
        e.visibility = wgpu::ShaderStages::COMPUTE;
        entries.push(e);
    }
    let bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vt-surface-probe"),
            entries: &entries,
        });
    let mut bg_entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: probe_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: out_buf.as_entire_binding(),
        },
    ];
    bg_entries.extend(pools.bind_group_entries(14));
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vt-surface-probe"),
        layout: &bgl,
        entries: &bg_entries,
    });
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vt-surface-probe"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vt-surface-probe"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("cs_surface"),
            compilation_options: Default::default(),
            cache: None,
        });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vt-surface-probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((probes.len() as u32).div_ceil(64).max(1), 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &rb, 0, out_size);
    gpu.queue.submit([enc.finish()]);
    let slice = rb.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("map surface readback");
    let raw: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec();
    drop(mapped);
    rb.unmap();
    raw.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

/// **WAVE T: the detail map reaches the surface, and is INERT without one**
/// (the texture document's section 3 A).
///
/// A feature that ships behind a default-off lane has two claims to prove and
/// they falsify opposite mistakes:
///
/// 1. **inert** — with the detail lane zero, `vt_surface` returns exactly what
///    it returned before Wave T. That is what makes "the 54 goldens did not
///    move" a measurement rather than a hope, and it is asserted as equality,
///    not as a tolerance.
/// 2. **engaged** — with the lane set, the tangent-space normal moves in the
///    direction the detail map's XY says and the roughness is multiplied by its
///    alpha. Without this the whole channel could be dead code and every other
///    arm in the file would still pass.
///
/// A third case is the one a shipping bug lives in: the slot set and the SCALE
/// zero must be inert, because the two fields are one decision and a surface
/// that sampled a detail map at 0x would tile it once across the whole object.
#[test]
fn the_detail_map_reaches_the_surface_and_is_inert_without_one() {
    let Some(gpu) = gpu_or_skip("Wave T detail maps") else {
        return;
    };
    // Texture 1: the base normal map — flat (0,0,1) everywhere, so any XY in the
    // result came from the detail map and nowhere else.
    let base = raw_container(128, |_, _| [128, 128, 255, 255]);
    // Texture 2: the detail map — a constant, strongly non-neutral normal whose
    // X is +1 and Y is -1, with alpha 0.5 so the roughness term is visible too.
    let detail = raw_container(64, |_, _| [255, 0, 128, 128]);
    let (lib, pools) = resident_many(&gpu, vec![base, detail], false);
    let base_slot = lib.set_for(Some(1), Some(1), None).normal;
    let detail_slot = lib.set_for(Some(2), None, None).albedo;
    assert!(
        base_slot != 0 && detail_slot != 0,
        "the fixture is not warm"
    );

    // The three map words: albedo unbound, normal = the base map, ORM unbound.
    // The detail slot rides the ALBEDO word's top half and the scale the NORMAL
    // word's, exactly as `VtTextureSet::slots` packs them.
    let plain = [0u32, base_slot, 0, 0];
    let with_detail = [
        detail_slot << 16,
        base_slot | (u32::from(inf_render::detail_scale_q8(4.0)) << 16),
        0,
        0,
    ];
    let scale_zero = [detail_slot << 16, base_slot, 0, 0];

    // A tiny derivative, so the detail texture resolves near mip 0 and its fade
    // weight is 1 — the arm is about the blend, not about the ramp.
    let dd = [1e-5f32, 1e-5];
    let probes: Vec<SurfaceProbe> = [plain, with_detail, scale_zero]
        .into_iter()
        .flat_map(|maps| {
            [(0.25f32, 0.25f32), (0.5, 0.5), (0.75, 0.6)]
                .into_iter()
                .map(move |(u, v)| SurfaceProbe {
                    maps,
                    uv: [u, v],
                    dd,
                })
        })
        .collect();
    let got = run_surface_probe(&gpu, &pools, &probes);

    for k in 0..3 {
        let (no_detail, _) = got[k];
        let (detailed, _) = got[3 + k];
        let (zero_scale, _) = got[6 + k];

        // (1) INERT — both ways of saying "no detail" give the same surface.
        assert_eq!(
            no_detail, zero_scale,
            "probe {k}: a detail slot with scale 0 changed the surface; the slot \
             and the scale are one decision"
        );
        // The base map really is flat, so the un-detailed normal is +Z.
        assert!(
            no_detail[0].abs() < 0.02 && no_detail[1].abs() < 0.02 && no_detail[2] > 0.9,
            "probe {k}: the base normal is not flat: {no_detail:?} — the arm below \
             cannot attribute a change to the detail map"
        );
        assert!(
            (no_detail[3] - 0.5).abs() < 1e-6,
            "probe {k}: the scalar roughness did not pass through: {}",
            no_detail[3]
        );

        // (2) ENGAGED — and in the direction the detail map says.
        assert!(
            detailed[0] > 0.4,
            "probe {k}: the detail map's +X did not reach the normal: {detailed:?}"
        );
        assert!(
            detailed[1] < -0.4,
            "probe {k}: the detail map's -Y did not reach the normal: {detailed:?}"
        );
        assert!(
            (detailed[3] - 0.25).abs() < 0.02,
            "probe {k}: roughness is {} — the detail alpha (about 0.5) did not \
             multiply the scalar 0.5",
            detailed[3]
        );
        // The result is still a unit vector: a blend that forgot to normalize
        // reads as a lighting error nobody traces back to here.
        let len =
            (detailed[0] * detailed[0] + detailed[1] * detailed[1] + detailed[2] * detailed[2])
                .sqrt();
        assert!((len - 1.0).abs() < 1e-3, "probe {k}: |n| = {len}");
    }
}

/// **WAVE T: trilinear blends two levels, and is OFF by default** (T47).
///
/// The same two claims, on the other channel. The derivative is picked so the
/// level of detail lands at **1.5**, i.e. the worst case for a truncating
/// sampler: half a level of error, which is exactly the mip band a camera dolly
/// walks through.
///
/// # The fixture, and the one this arm started with
///
/// **A 32-texel stripe pattern does not work, and it is worth writing down why**
/// — it was this arm's first fixture and it measured *zero* of 24 probes
/// changing. A power-of-two-aligned stripe is very nearly **scale-invariant
/// under a box downsample**: every pair of texels averaged lies inside the same
/// stripe, so mip 1 is mip 0 at half the resolution with the same values, and
/// blending two levels of it changes nothing anywhere except within one texel of
/// an edge. The arm would have failed for a good reason and read as a broken
/// feature.
///
/// A **1-in-6 bright pattern** is not aligned to the halving, so each level
/// genuinely averages differently from the one above it, and "the two levels
/// disagree at this uv" is true almost everywhere. The final assertion measures
/// that property directly rather than trusting this paragraph.
#[test]
fn trilinear_blends_two_levels_and_is_off_by_default() {
    let Some(gpu) = gpu_or_skip("Wave T trilinear") else {
        return;
    };
    let stripes = || {
        raw_container(256, |x, _| {
            let v = if x % 6 == 0 { 250 } else { 10 };
            [v, v, v, 255]
        })
    };
    let (lib_off, pools_off) = resident_many(&gpu, vec![stripes()], false);
    let (lib_on, pools_on) = resident_many(&gpu, vec![stripes()], true);
    let slot_off = lib_off.set_for(Some(1), None, None).albedo;
    let slot_on = lib_on.set_for(Some(1), None, None).albedo;
    assert!(slot_off != 0 && slot_on != 0, "the fixture is not warm");

    // rho = 2^1.5 texels of a 256-texel level, so lod = 1.5 — the maximum
    // possible distance from an integer level.
    let dd = [2f32.powf(1.5) / 256.0, 2f32.powf(1.5) / 256.0];
    // …and an integer level, where the two paths must agree exactly.
    let dd_int = [2.0f32 / 256.0, 2.0 / 256.0];

    let sweep: Vec<(f32, f32)> = (0..24).map(|i| (i as f32 / 24.0 + 0.01, 0.37)).collect();
    let build = |slot: u32, dd: [f32; 2]| -> Vec<SurfaceProbe> {
        sweep
            .iter()
            .map(|&(u, v)| SurfaceProbe {
                maps: [slot, 0, 0, 0],
                uv: [u, v],
                dd,
            })
            .collect()
    };

    let off = run_surface_probe(&gpu, &pools_off, &build(slot_off, dd));
    let on = run_surface_probe(&gpu, &pools_on, &build(slot_on, dd));
    let off_int = run_surface_probe(&gpu, &pools_off, &build(slot_off, dd_int));
    let on_int = run_surface_probe(&gpu, &pools_on, &build(slot_on, dd_int));

    // (2) ENGAGED at a fractional level.
    let differing = off
        .iter()
        .zip(&on)
        .filter(|(a, b)| (a.1[0] - b.1[0]).abs() > 1.0 / 255.0)
        .count();
    assert!(
        differing >= sweep.len() / 4,
        "only {differing} of {} probes changed with trilinear on — the second tap \
         is not reaching the result",
        sweep.len()
    );

    // (1) INERT at an integer level: `blend < 0.01` takes the early-out, so the
    // flag costs nothing and returns the same value.
    for (i, (a, b)) in off_int.iter().zip(&on_int).enumerate() {
        assert_eq!(
            a.1, b.1,
            "probe {i}: trilinear changed a sample at an INTEGER level of detail, \
             where the document's own early-out says it must not"
        );
    }
    // ANTI-VACUITY, and the finding this arm's first fixture produced: the two
    // levels being blended must actually DISAGREE. A pattern aligned to the
    // halving (a 32-texel stripe) is scale-invariant under a box downsample, so
    // mip 1 and mip 2 are the same image at the same uv and a correct trilinear
    // sampler changes nothing — measured as 0 of 24 probes, which reads exactly
    // like a dead feature. Asserted here on the levels themselves rather than
    // trusted from the doc comment.
    let spread = off
        .iter()
        .map(|(_, c)| c[0])
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
    assert!(
        spread.1 - spread.0 > 0.05,
        "the fixture is flat at this level of detail ({spread:?}) — nothing here \
         could have differed"
    );
    let coarser = run_surface_probe(
        &gpu,
        &pools_off,
        &build(slot_off, [dd_int[0] * 2.0, dd_int[1] * 2.0]),
    );
    let level_gap = off_int
        .iter()
        .zip(&coarser)
        .filter(|(a, b)| (a.1[0] - b.1[0]).abs() > 1.0 / 255.0)
        .count();
    assert!(
        level_gap >= sweep.len() / 2,
        "mip 1 and mip 2 of the fixture agree at {} of {} probes — a blend \
         between them could not change anything, so the arm above certifies \
         nothing",
        sweep.len() - level_gap,
        sweep.len()
    );
}
