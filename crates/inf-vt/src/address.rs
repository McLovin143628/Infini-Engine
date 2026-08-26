//! The **virtual address space**: where a tile is, who its parent is, and which
//! entry of the indirection table describes it.
//!
//! Every piece of arithmetic in this module is shared by three consumers that
//! must agree exactly or the picture is wrong:
//!
//! * the residency ([`crate::residency`]), which walks *up* the pyramid to
//!   resolve a fallback;
//! * the container (`inf_material::tiles`), whose tile-directory index is
//!   [`VtTextureDesc::entry_index`] — the same number, so the table entry that
//!   describes a tile and the directory entry that locates its bytes are at the
//!   same index;
//! * the fragment shader (P26.3), which walks *down* from a `uv` and a mip.
//!
//! The third one gets a `uv` and a mip and must land on the tile the *table
//! entry* names. It must therefore walk the tile tree down from the address it
//! asked for — [`VtTextureDesc::ancestor`]'s walk, `min(t / 2, tiles - 1)` per
//! level — and **must not re-derive the tile from `uv` at the resolved mip**. The
//! two are not the same function: the container halves an extent with `w / 2`, so
//! on a level whose extent is odd `uv × w_coarse` is not `texel / 2`, and at a
//! tile boundary the two land in different tiles (measured: 1 322 addresses of a
//! 4 095² pyramid). What *is* true, and what the shader's `+ border` rests on, is
//! that the resolved level's texel sits inside the named tile's payload extended
//! by the border ring — never more than one texel out. That is the **sampling
//! contract**, asserted over a swept set of extents by
//! [`tests::the_offset_inside_the_resolved_tile_is_covered_by_the_border`], with
//! the forbidden alternative pinned by
//! [`tests::the_shader_must_not_re_derive_the_tile_from_uv`].

use std::ops::Range;

/// Address of one tile in a virtual texture — the key a residency table is built
/// on, and the key `.inf_tex`'s tile directory is sorted by.
///
/// **Its derived `Ord` is `(mip, x, y)`; the tile directory is sorted
/// `(mip, y, x)`.** The two orders are different and neither is wrong, but a
/// `BTreeSet<TileCoord>` iterates in *its* order rather than the file's, so a
/// residency pass that wants to walk a request set in payload order — which is
/// the point of a sorted directory, and the difference between one sequential
/// read and a scatter — has to sort on [`payload_order`](Self::payload_order)
/// itself. [`crate::VtResidency::apply_wants`] does.
///
/// (Found by the P26.1 audit, when this type lived in `inf_material::tiles`. It
/// moved here with the crate that consumes it; `inf_material::tiles` re-exports
/// it, so every existing path still names the same type.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileCoord {
    pub mip: u32,
    pub x: u32,
    pub y: u32,
}

impl TileCoord {
    pub const fn new(mip: u32, x: u32, y: u32) -> Self {
        Self { mip, x, y }
    }

    /// This address as the tile directory sorts it — the key to order a request
    /// set by, when the order that matters is where the bytes are.
    pub const fn payload_order(&self) -> (u32, u32, u32) {
        (self.mip, self.y, self.x)
    }
}

/// One mip level's grid, as the container's `TexMipEntry` describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtMipDesc {
    /// The level's virtual extent, in texels.
    pub width: u32,
    pub height: u32,
    /// Tiles across / down (`ceil(extent / tile_size)`, at least 1).
    pub tiles_x: u32,
    pub tiles_y: u32,
}

impl VtMipDesc {
    #[inline]
    pub fn tile_count(&self) -> u32 {
        self.tiles_x * self.tiles_y
    }
}

/// The maximum mip count a virtual texture may declare.
///
/// A 32 768-texel level chain is 16 levels; 32 is generous headroom and it is
/// what keeps the packed table entry's 8-bit mip field ([`crate::table`]) unable
/// to overflow — a bound the encoder can rely on instead of checking.
pub const MAX_VT_MIPS: usize = 32;

/// The widest level a virtual texture may declare, in texels.
///
/// **This is the door D-22 said had nothing behind it.** Until wave TER2a
/// [`VtTextureDesc::validate`] had no extent rule at all: what kept a pyramid
/// inside an `f32` uv was that no content this project could produce came near
/// the limit, and the arm that watched it
/// (`table::tests::an_f32_uv_still_addresses_every_texel_of_the_largest_legal_pyramid`)
/// watches the **resident** side — the atlas constants — and is explicit that
/// "it does not fire on a registration, and it cannot". TER2a is the wave that
/// authors real ground materials for a 51 km² world, which is exactly the
/// content D-22 said to re-measure before, so the gap is closed rather than
/// carried again.
///
/// `1 << 23` is the `f32` mantissa's integer limit: a shader resolves a texel
/// inside one tiling period as `uv · width`, and past `2^23` consecutive
/// integers stop being representable, so two distinct texels start rounding onto
/// one address. It is the same number the resident-side arm compares against,
/// stated on the other half of the same address chain.
///
/// For scale: the largest texture this engine's own importer will produce is
/// bounded by `VtPoolConfig::max_texture_dim` (8 192) on the atlas side, and the
/// ground sets TER2a authors are 1 024² — **8 192 times** under this ceiling.
/// The guard exists so that a hand-built descriptor, a future importer, or a
/// photogrammetry bake that grew cannot walk past it silently.
pub const MAX_VT_EXTENT: u32 = 1 << 23;

/// One virtual texture's address space: the tile geometry and the pyramid.
///
/// Built from a container by `inf_material::tiles::TiledTextureReader::vt_desc`,
/// or by hand in a test. Deliberately holds **no bytes and no asset id** — this
/// is the shape of the address space, and the residency keys textures by the
/// handle it hands back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VtTextureDesc {
    /// Payload texels per tile side (the container's `TILE_SIZE`).
    pub tile_size: u32,
    /// Border texels baked on **each** side (the container's `TILE_BORDER`).
    pub border: u32,
    /// Whether the stored data is sRGB-encoded. Carried through to the atlas
    /// format decision and to the table, never interpreted here.
    pub srgb: bool,
    /// Whether the stored data is **two-channel**, so a sampler has to rebuild
    /// the third as `z = sqrt(1 − x² − y²)` (Wave T's BC5 normal maps).
    ///
    /// Rides beside `srgb` and for the same reason: one atlas holds base colours
    /// and normal maps at once, so neither can be a pool-wide decision. Like
    /// `srgb` it is carried, never interpreted here — the table hands it to the
    /// shader as a flag bit and the shader is the only reader.
    ///
    /// `false` for every format that predates Wave T, which is what makes a
    /// pre-Wave-T texture shade exactly as it did.
    pub reconstruct_z: bool,
    /// The pyramid, **finest first** — exactly the container's mip directory.
    pub mips: Vec<VtMipDesc>,
}

impl VtTextureDesc {
    /// Texels per side of a stored tile: `border + tile_size + border`.
    #[inline]
    pub fn stored_tile_size(&self) -> u32 {
        self.tile_size + 2 * self.border
    }

    /// Levels in the pyramid.
    #[inline]
    pub fn mip_count(&self) -> u32 {
        self.mips.len() as u32
    }

    /// The **coarsest** level — the one that is always fully resident (law 1).
    #[inline]
    pub fn coarsest_mip(&self) -> u32 {
        self.mip_count().saturating_sub(1)
    }

    /// Tiles in the whole pyramid — the number of indirection entries this
    /// texture needs, and the number of tiles its container stores.
    pub fn tile_count(&self) -> u32 {
        self.mips.iter().map(|m| m.tile_count()).sum()
    }

    /// The **root set**: every tile of the coarsest level, in payload order.
    ///
    /// One tile for a full pyramid (the level has shrunk to `1×1` texels, which
    /// is one tile). For a texture imported with `generate_mips: false` the
    /// coarsest level *is* mip 0, so the root set is the whole texture — which is
    /// the honest reading of "the coarsest level is always resident" and the
    /// reason the floor is a *set* rather than a single page.
    pub fn root_tiles(&self) -> Vec<TileCoord> {
        let mip = self.coarsest_mip();
        let Some(m) = self.mips.get(mip as usize) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(m.tile_count() as usize);
        for y in 0..m.tiles_y {
            for x in 0..m.tiles_x {
                out.push(TileCoord::new(mip, x, y));
            }
        }
        out
    }

    /// Whether `at` names a tile that exists.
    pub fn contains(&self, at: TileCoord) -> bool {
        self.mips
            .get(at.mip as usize)
            .is_some_and(|m| at.x < m.tiles_x && at.y < m.tiles_y)
    }

    /// **The index of `at` in this texture's tile order** — identical to the
    /// index of its entry in the `.inf_tex` v2 tile directory, so the table entry
    /// that describes a tile and the directory entry that locates its bytes sit
    /// at the same number.
    ///
    /// Pinned against a real container in `inf_material::tiles`'s tests.
    pub fn entry_index(&self, at: TileCoord) -> Option<u32> {
        if !self.contains(at) {
            return None;
        }
        let mut base = 0u32;
        for m in &self.mips[..at.mip as usize] {
            base += m.tile_count();
        }
        let m = &self.mips[at.mip as usize];
        Some(base + at.y * m.tiles_x + at.x)
    }

    /// The index of mip `mip`'s first tile (its `first_tile` in the container).
    pub fn mip_first_entry(&self, mip: u32) -> Option<u32> {
        if mip as usize > self.mips.len() {
            return None;
        }
        Some(
            self.mips[..mip as usize]
                .iter()
                .map(|m| m.tile_count())
                .sum(),
        )
    }

    /// The tile at the next-coarser level whose footprint contains `at`'s, or
    /// `None` when `at` is already at the coarsest level.
    ///
    /// # Why the clamp is not defensive
    ///
    /// The obvious answer is `(x/2, y/2)`, and it is wrong at exactly one shape:
    /// a level whose extent leaves a **sliver tile**. A 257-texel level is
    /// `ceil(257/128) = 3` tiles, the third covering a single texel; the level
    /// below it is 128 texels, i.e. **one** tile — and `2/2 = 1` is off the end
    /// of it. Clamping to the last tile is not a rescue, it is the right answer:
    /// that sliver's texel maps to texel 128 of a 128-texel level, which
    /// clamp-to-edge sampling reads as texel 127, and texel 127 is in tile 0.
    /// The container's border ring is built with the same `clamp(x, 0, w-1)`
    /// rule, so the two agree by construction.
    pub fn parent(&self, at: TileCoord) -> Option<TileCoord> {
        let up = at.mip + 1;
        let m = self.mips.get(up as usize)?;
        Some(TileCoord::new(
            up,
            (at.x / 2).min(m.tiles_x - 1),
            (at.y / 2).min(m.tiles_y - 1),
        ))
    }

    /// The ancestor of `at` at level `target` (`target >= at.mip`), following the
    /// same clamped chain [`parent`](Self::parent) walks.
    ///
    /// **This is the tile an indirection entry names**, and the CPU twin of the
    /// walk the fragment shader must do when the table hands it back a coarser
    /// mip than it asked for. It is *not* [`tile_at_texel`](Self::tile_at_texel)
    /// at the coarser level — see the sampling contract in the module docs.
    pub fn ancestor(&self, at: TileCoord, target: u32) -> Option<TileCoord> {
        if !self.contains(at) || target < at.mip || target as usize >= self.mips.len() {
            return None;
        }
        let mut cur = at;
        while cur.mip < target {
            cur = self.parent(cur)?;
        }
        Some(cur)
    }

    /// The tile of level `mip` that texel `(px, py)` of **that level** falls in.
    ///
    /// The arithmetic for the mip a sample *asks* for, where the texel and the
    /// tile come from the same level and the answer is exact. It is **not** how a
    /// fallback is addressed: applied at a coarser level than the one that was
    /// asked for it disagrees with [`ancestor`](Self::ancestor) by a whole tile on
    /// an odd chain, which
    /// [`tests::the_shader_must_not_re_derive_the_tile_from_uv`] measures.
    pub fn tile_at_texel(&self, mip: u32, px: u32, py: u32) -> Option<TileCoord> {
        let m = self.mips.get(mip as usize)?;
        Some(TileCoord::new(
            mip,
            (px / self.tile_size).min(m.tiles_x - 1),
            (py / self.tile_size).min(m.tiles_y - 1),
        ))
    }

    /// The half-open range of child columns of parent column `x` at level `mip`.
    ///
    /// The inverse of [`parent`](Self::parent)'s clamp: an ordinary parent has
    /// children `{2x, 2x+1}`, and the **last** parent additionally adopts every
    /// sliver column beyond `2x+1` — because that is where the clamp sent them.
    pub fn child_x_range(&self, mip: u32, x: u32) -> Range<u32> {
        child_range(
            self.mips.get(mip as usize).map_or(0, |m| m.tiles_x),
            mip.checked_sub(1)
                .and_then(|c| self.mips.get(c as usize))
                .map_or(0, |m| m.tiles_x),
            x,
        )
    }

    /// **The quadrant path from an ancestor down to `at`** (P26.5), coarsest
    /// first — the inverse of [`ancestor`](Self::ancestor), as the sequence of
    /// half-selections [`crate::fill::fill_from_ancestor`] needs to reconstruct a
    /// page.
    ///
    /// Each step is `(qx, qy)` in `{0, 1}²`: which half of the parent tile's
    /// payload the child covers. On the clamped chain a "child" may be a sliver
    /// several tiles past `2·parent + 1` — the same case
    /// [`child_x_range`](Self::child_x_range) documents — and it takes the LAST
    /// half, which is where the clamp sent it and therefore where its texels
    /// actually live.
    ///
    /// `None` unless `from` really is `at`'s ancestor at its own level, so a
    /// caller cannot reconstruct a page out of an unrelated tile: the chain is
    /// re-walked here rather than trusted.
    pub fn descent(&self, at: TileCoord, from: TileCoord) -> Option<Vec<(u32, u32)>> {
        if self.ancestor(at, from.mip)? != from {
            return None;
        }
        let mut steps = Vec::with_capacity((from.mip - at.mip) as usize);
        let mut cur = at;
        while cur.mip < from.mip {
            let up = self.parent(cur)?;
            steps.push(((cur.x - up.x * 2).min(1), (cur.y - up.y * 2).min(1)));
            cur = up;
        }
        steps.reverse();
        Some(steps)
    }

    /// The half-open range of child rows of parent row `y` at level `mip`.
    pub fn child_y_range(&self, mip: u32, y: u32) -> Range<u32> {
        child_range(
            self.mips.get(mip as usize).map_or(0, |m| m.tiles_y),
            mip.checked_sub(1)
                .and_then(|c| self.mips.get(c as usize))
                .map_or(0, |m| m.tiles_y),
            y,
        )
    }

    /// Validate the shape. Called by [`crate::VtResidency::register_texture`], so
    /// a hand-built descriptor that is not a real pyramid is refused at the door
    /// rather than mis-resolved for the rest of the session.
    pub fn validate(&self) -> Result<(), DescError> {
        if self.mips.is_empty() {
            return Err(DescError::EmptyPyramid);
        }
        if self.mips.len() > MAX_VT_MIPS {
            return Err(DescError::TooManyMips {
                mips: self.mips.len(),
                max: MAX_VT_MIPS,
            });
        }
        if self.tile_size == 0
            || !self.tile_size.is_multiple_of(4)
            || !self.border.is_multiple_of(4)
        {
            return Err(DescError::TileGeometry {
                tile_size: self.tile_size,
                border: self.border,
            });
        }
        for (i, m) in self.mips.iter().enumerate() {
            if m.width == 0 || m.height == 0 {
                return Err(DescError::MipExtent { mip: i as u32 });
            }
            // TER2a (D-22): the extent rule this door did not have. See
            // [`MAX_VT_EXTENT`] — a level past the f32 mantissa's integer limit
            // cannot be addressed to the texel by `uv · width`, and two texels
            // would silently round onto one.
            if m.width > MAX_VT_EXTENT || m.height > MAX_VT_EXTENT {
                return Err(DescError::MipTooLarge {
                    mip: i as u32,
                    width: m.width,
                    height: m.height,
                    max: MAX_VT_EXTENT,
                });
            }
            if m.tiles_x != m.width.div_ceil(self.tile_size).max(1)
                || m.tiles_y != m.height.div_ceil(self.tile_size).max(1)
            {
                return Err(DescError::MipGrid { mip: i as u32 });
            }
            // The chain halves, exactly as `rgba_mip_chain` builds it
            // (`(w / 2).max(1)`). Pinning it is what makes the clamped parent
            // chain a geometric statement rather than a convention: a descriptor
            // whose levels are not successive halvings would resolve fallbacks
            // to tiles that do not cover the footprint they claim to.
            if i > 0 {
                let p = &self.mips[i - 1];
                if m.width != (p.width / 2).max(1) || m.height != (p.height / 2).max(1) {
                    return Err(DescError::MipChain { mip: i as u32 });
                }
            }
        }
        Ok(())
    }
}

/// `parent_count` tiles at the parent level, `child_count` at the child level;
/// which children belong to parent `p`.
fn child_range(parent_count: u32, child_count: u32, p: u32) -> Range<u32> {
    let lo = p.saturating_mul(2);
    if lo >= child_count || parent_count == 0 {
        return lo..lo;
    }
    let hi = if p + 1 == parent_count {
        child_count
    } else {
        (lo + 2).min(child_count)
    };
    lo..hi
}

/// A descriptor that is not a virtual texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DescError {
    #[error("a virtual texture needs at least one mip level")]
    EmptyPyramid,
    #[error("{mips} mip levels is past the {max}-level ceiling")]
    TooManyMips { mips: usize, max: usize },
    #[error("tile geometry {tile_size}+2×{border} is not a whole number of BC blocks")]
    TileGeometry { tile_size: u32, border: u32 },
    #[error("mip {mip} has a zero extent")]
    MipExtent { mip: u32 },
    #[error(
        "mip {mip} is {width}×{height}, past the {max}-texel ceiling an f32 uv \
         can address to the texel"
    )]
    MipTooLarge {
        mip: u32,
        width: u32,
        height: u32,
        max: u32,
    },
    #[error("mip {mip}'s grid does not tile its extent")]
    MipGrid { mip: u32 },
    #[error("mip {mip} is not half of mip {} — this is not a mip chain", mip - 1)]
    MipChain { mip: u32 },
}

/// Build a descriptor for a `width × height` texture with a full mip chain, at
/// the given tile geometry.
///
/// The test-side twin of what the container writes. Ships (rather than living in
/// a test) because `inf-render`'s mirror tests and P26.3's fixtures need the same
/// pyramid without linking the importer.
pub fn full_pyramid(
    width: u32,
    height: u32,
    tile_size: u32,
    border: u32,
    srgb: bool,
) -> VtTextureDesc {
    let mut mips = Vec::new();
    let (mut w, mut h) = (width.max(1), height.max(1));
    loop {
        mips.push(VtMipDesc {
            width: w,
            height: h,
            tiles_x: w.div_ceil(tile_size).max(1),
            tiles_y: h.div_ceil(tile_size).max(1),
        });
        if w == 1 && h == 1 {
            break;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    VtTextureDesc {
        tile_size,
        border,
        srgb,
        // A test helper builds a pyramid, not a normal map: the two-channel
        // rebuild is a property of the stored FORMAT, which this door does not
        // take, so it is false here and set from the container in `vt_desc`.
        reconstruct_z: false,
        mips,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The extent guard D-22 asked for, and the island-scale exposure it was
    /// asked about** (wave TER2a, clause 1).
    ///
    /// Two halves, because the "32 K f32-uv tripwire" is really two independent
    /// exposures on one address chain and only one of them had a door:
    ///
    /// 1. **the pyramid's own extent** — `uv · width` inside one tiling period.
    ///    This is the half `validate` had no rule for at all;
    ///    [`MAX_VT_EXTENT`] is now that rule, and this arm fires it.
    /// 2. **the world position the uv is derived from** — for terrain,
    ///    `(world.xz − origin.xz) / tex_scale` in `terrain.wgsl`, which is an
    ///    **f32 world coordinate**. This one has no door and needs none; it is a
    ///    precision budget, and the numbers below are the measurement TER2a took
    ///    before authoring the ground sets it authors.
    #[test]
    fn the_extent_guard_fires_and_the_island_sits_far_under_it() {
        // ── half 1: the guard ────────────────────────────────────────────────
        let ok = full_pyramid(1024, 1024, 128, 4, true);
        assert!(ok.validate().is_ok(), "a 1024² ground set must register");
        // A hand-built descriptor one texel past the ceiling is refused, by name.
        let mut over = full_pyramid(1024, 1024, 128, 4, true);
        over.mips[0].width = MAX_VT_EXTENT + 1;
        over.mips[0].tiles_x = over.mips[0].width.div_ceil(over.tile_size).max(1);
        match over.validate() {
            Err(DescError::MipTooLarge {
                mip, width, max, ..
            }) => {
                assert_eq!(mip, 0);
                assert_eq!(width, MAX_VT_EXTENT + 1);
                assert_eq!(max, MAX_VT_EXTENT);
            }
            other => panic!("an over-wide pyramid was not refused: {other:?}"),
        }
        // ANTI-VACUITY: exactly at the ceiling is legal, so the guard is a
        // ceiling and not an off-by-one that moved the real limit.
        let mut at = full_pyramid(1024, 1024, 128, 4, true);
        at.mips[0].width = MAX_VT_EXTENT;
        at.mips[0].tiles_x = at.mips[0].width.div_ceil(at.tile_size).max(1);
        assert!(
            matches!(at.validate(), Err(DescError::MipChain { .. })) || at.validate().is_ok(),
            "the ceiling itself must not be refused for being too large"
        );

        // ── half 2: the island's own numbers ─────────────────────────────────
        //
        // The Vancouver island is 28 × 28 tiles of 257 samples at 1 m, centred
        // on the world origin: 7 168 m square, so the furthest ground from the
        // origin is the corner at 3 584 m on each axis. `terrain.wgsl` rebuilds
        // an ABSOLUTE world XZ in f32 there (it must — a render-local uv would
        // slide the material every time the floating origin snaps).
        const ISLAND_HALF_EXTENT_M: f32 = 3584.0;
        // The f32 quantum at that magnitude: 2^exp · 2^-23.
        let ulp_m = f32::from_bits(ISLAND_HALF_EXTENT_M.to_bits() + 1) - ISLAND_HALF_EXTENT_M;
        // The finest ground TER2a authors: a 1 024² albedo tiled every 1.5 m
        // (sand — the tightest of the five).
        const FINEST_EXTENT: f32 = 1024.0;
        const TIGHTEST_TEX_SCALE_M: f32 = 1.5;
        let texel_m = TIGHTEST_TEX_SCALE_M / FINEST_EXTENT;
        println!(
            "VT EXTENT (TER2a clause 1): guard {MAX_VT_EXTENT} texels; the \
             island's ground sets are {FINEST_EXTENT}² — {}x under it. \
             f32 world quantum at the island's far corner ({ISLAND_HALF_EXTENT_M} m) \
             is {:.6} mm; the finest texel is {:.4} mm at tex_scale \
             {TIGHTEST_TEX_SCALE_M} m, so a texel is {:.1} distinct positions wide.",
            MAX_VT_EXTENT / FINEST_EXTENT as u32,
            ulp_m * 1000.0,
            texel_m * 1000.0,
            texel_m / ulp_m
        );
        assert!(
            texel_m / ulp_m >= 4.0,
            "an f32 world position quantises the island's finest ground texel: \
             {:.6} mm quantum against a {:.6} mm texel. The two-u32 uv path (T44) \
             is what this measurement exists to trigger.",
            ulp_m * 1000.0,
            texel_m * 1000.0
        );
        assert!(
            FINEST_EXTENT as u32 <= MAX_VT_EXTENT,
            "the content this wave authors would not register"
        );
    }

    #[test]
    fn the_derived_order_is_not_the_payload_order() {
        // The P26.1 audit's finding, pinned where the type now lives. `(mip, x,
        // y)` and `(mip, y, x)` disagree the moment x and y differ, so a table
        // that sorts one way and a file laid out the other way walk apart.
        let a = TileCoord::new(0, 0, 1);
        let b = TileCoord::new(0, 1, 0);
        assert!(a < b, "derived Ord compares x before y");
        assert!(
            b.payload_order() < a.payload_order(),
            "payload order compares y before x — the OPPOSITE verdict"
        );
    }

    #[test]
    fn the_pyramid_indexes_exactly_as_the_container_does() {
        let d = full_pyramid(320, 192, 128, 4, true);
        // 320,192 → 160,96 → 80,48 → 40,24 → 20,12 → 10,6 → 5,3 → 2,1 → 1,1
        assert_eq!(d.mips.len(), 9);
        assert_eq!((d.mips[0].tiles_x, d.mips[0].tiles_y), (3, 2));
        assert_eq!((d.mips[1].tiles_x, d.mips[1].tiles_y), (2, 1));
        assert_eq!(d.tile_count(), 6 + 2 + 7);
        // The index walk is (mip, y, x).
        let mut n = 0;
        for (mip, m) in d.mips.iter().enumerate() {
            for y in 0..m.tiles_y {
                for x in 0..m.tiles_x {
                    let at = TileCoord::new(mip as u32, x, y);
                    assert_eq!(d.entry_index(at), Some(n), "{at:?}");
                    n += 1;
                }
            }
        }
        assert_eq!(n, d.tile_count());
        assert_eq!(d.entry_index(TileCoord::new(0, 3, 0)), None);
        assert_eq!(d.entry_index(TileCoord::new(9, 0, 0)), None);
        d.validate().expect("a real pyramid validates");
    }

    /// **The sliver.** 257 texels is three tiles whose last one is a single
    /// texel, over a 128-texel level that is one tile — `2/2 = 1` is off the end.
    #[test]
    fn the_sliver_tiles_parent_is_clamped_not_invented() {
        let d = full_pyramid(257, 257, 128, 4, false);
        assert_eq!((d.mips[0].tiles_x, d.mips[0].tiles_y), (3, 3));
        assert_eq!((d.mips[1].tiles_x, d.mips[1].tiles_y), (1, 1));
        assert_eq!(
            d.parent(TileCoord::new(0, 2, 2)),
            Some(TileCoord::new(1, 0, 0))
        );
        // …and the inverse relation agrees: mip 1's only tile adopts all three
        // columns, including the sliver.
        assert_eq!(d.child_x_range(1, 0), 0..3);
        assert_eq!(d.child_y_range(1, 0), 0..3);
    }

    /// **`descent` names the half of the parent the child's TEXELS are in**
    /// (P26.5) — the clamp included, which is where the naive answer is wrong.
    ///
    /// A descent step is not `x & 1`. On an ordinary chain the two agree; on a
    /// **clamped** one they do not, and the clamped one is the case the P26.2
    /// audit's whole finding lives on. A 257-texel level is three tiles, the
    /// third covering one texel, over a 128-texel level that is **one** tile —
    /// so tile 2's parent is tile 0, `x & 1` says "the left half of it", and the
    /// texel says otherwise: child texel 256 maps to `min(128, 127) = 127` of
    /// the parent, which is the **right** end of its payload. Reconstruct that
    /// sliver's page out of the left half and it is a page of the wrong content
    /// with no error anywhere.
    ///
    /// So the sweep asserts each step against the texel map — `min(t / 2, w - 1)`,
    /// the same clamp the container bakes its border ring with — rather than
    /// against a re-walk, which cannot tell the two apart precisely where they
    /// differ (every child of a clamped parent is in that parent's subtree, so a
    /// round-trip check passes on both answers; measured).
    #[test]
    fn the_descent_names_the_half_the_childs_texels_are_in() {
        let mut clamped = 0u32;
        let mut checked = 0u32;
        for (w, h) in [
            (511u32, 3u32),
            (1023, 1023),
            (2047, 511),
            (257, 257),
            (4095, 4095),
        ] {
            let d = full_pyramid(w, h, 128, 4, false);
            let ts = d.tile_size;
            for mip in 0..d.mip_count() {
                let m = d.mips[mip as usize];
                for ty in 0..m.tiles_y {
                    for tx in 0..m.tiles_x {
                        let at = TileCoord::new(mip, tx, ty);
                        for target in mip..d.mip_count() {
                            let anc = d.ancestor(at, target).expect("an ancestor");
                            let steps = d.descent(at, anc).expect("a descent");
                            assert_eq!(steps.len(), (target - mip) as usize);
                            // Walk the chain again, and check every step against
                            // where the tile's own first texel lands in its
                            // parent's payload.
                            let mut cur = at;
                            for &(qx, qy) in steps.iter().rev() {
                                let up = d.parent(cur).expect("a parent");
                                let pm = d.mips[up.mip as usize];
                                let px = (cur.x * ts / 2).min(pm.width - 1);
                                let py = (cur.y * ts / 2).min(pm.height - 1);
                                let want = (
                                    u32::from(px - up.x * ts >= ts / 2),
                                    u32::from(py - up.y * ts >= ts / 2),
                                );
                                assert_eq!(
                                    (qx, qy),
                                    want,
                                    "{w}x{h}: {cur:?} -> {up:?} descends as {:?}, but its \
                                     first texel lands at ({px}, {py}) of the parent level",
                                    (qx, qy)
                                );
                                if d.child_x_range(up.mip, up.x).len() > 2
                                    || d.child_y_range(up.mip, up.y).len() > 2
                                {
                                    clamped += 1;
                                }
                                cur = up;
                            }
                            assert_eq!(cur, anc, "the chain did not arrive");
                            checked += 1;
                        }
                    }
                }
            }
        }
        // ANTI-VACUITY, both halves: the sweep is not empty, and it really did
        // reach the clamped chains the odd extents exist for — without them this
        // is a statement about power-of-two pyramids, where `x & 1` is right.
        assert!(checked > 1000, "the sweep visited only {checked} pairs");
        assert!(
            clamped > 0,
            "no clamped parent was reached, so this arm never exercised the case \
             it names"
        );

        // A tile that is NOT a descendant is refused rather than reconstructed
        // out of an unrelated subtree.
        let d = full_pyramid(1023, 1023, 128, 4, false);
        assert!(d
            .descent(TileCoord::new(0, 7, 7), TileCoord::new(1, 0, 0))
            .is_none());
        // …and the zero-step case is the identity, which is what makes "no
        // fallback" not a special case at the call site.
        let self_ = TileCoord::new(2, 1, 1);
        assert_eq!(d.descent(self_, self_), Some(Vec::new()));
    }

    #[test]
    fn every_child_range_inverts_the_parent_clamp() {
        for (w, h) in [
            (320u32, 192u32),
            (257, 257),
            (300, 260),
            (129, 3),
            (8192, 4),
            (4, 8192),
            (255, 255),
            (511, 3),
            (128, 128),
            (1, 1),
        ] {
            let d = full_pyramid(w, h, 128, 4, false);
            for mip in 1..d.mip_count() {
                let m = d.mips[mip as usize];
                let child = d.mips[mip as usize - 1];
                // Every child is adopted exactly once…
                let mut seen = vec![0u32; (child.tiles_x * child.tiles_y) as usize];
                for y in 0..m.tiles_y {
                    for x in 0..m.tiles_x {
                        for cy in d.child_y_range(mip, y) {
                            for cx in d.child_x_range(mip, x) {
                                seen[(cy * child.tiles_x + cx) as usize] += 1;
                                assert_eq!(
                                    d.parent(TileCoord::new(mip - 1, cx, cy)),
                                    Some(TileCoord::new(mip, x, y)),
                                    "{w}×{h} mip {mip}: ({cx},{cy}) is not ({x},{y})'s child"
                                );
                            }
                        }
                    }
                }
                assert!(
                    seen.iter().all(|&n| n == 1),
                    "{w}×{h} mip {mip}: a child was adopted {:?} times",
                    seen.iter().collect::<std::collections::BTreeSet<_>>()
                );
            }
        }
    }

    /// **The sampling contract**, on the CPU — the arithmetic P26.3's shader has
    /// to implement, and the one it must not.
    ///
    /// The tile an entry names is [`ancestor`](VtTextureDesc::ancestor)'s, walked
    /// down the tile tree from the address that was asked for. The *offset inside
    /// that tile* is then the resolved level's texel minus that tile's origin —
    /// and this arm's real content is that the offset stays inside the tile's
    /// payload **extended by the border ring**, because on an odd chain it does
    /// not stay inside the payload.
    ///
    /// See [`the_shader_must_not_re_derive_the_tile_from_uv`] for the measurement
    /// that forbids the obvious alternative.
    #[test]
    fn the_offset_inside_the_resolved_tile_is_covered_by_the_border() {
        // The awkward extents, swept rather than sampled: a one-texel sliver
        // (257), a one-texel-SHORT level (255, whose last tile is 127 texels), a
        // 14-level strip in both orientations (8192×4, 4×8192 — the only shapes
        // where `tiles_x != tiles_y` for most of the chain), and five pyramids
        // whose levels are ODD and wide enough to have more than one parent tile,
        // which is the only shape where the tile tree and the uv derivation part.
        let mut disagreements = 0u64;
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for (w, h) in [
            (320u32, 192u32),
            (257, 257),
            (255, 255),
            (300, 260),
            (129, 129),
            (511, 3),
            (8192, 4),
            (4, 8192),
            (1000, 1),
            (1023, 1023),
            (2047, 511),
            (777, 333),
        ] {
            let d = full_pyramid(w, h, 128, 4, false);
            let (tile, border) = (d.tile_size as i64, d.border as i64);
            for mip in 0..d.mip_count() {
                let m = d.mips[mip as usize];
                for ty in 0..m.tiles_y {
                    for tx in 0..m.tiles_x {
                        let at = TileCoord::new(mip, tx, ty);
                        // A texel inside this tile, in the level's own space, at
                        // both the low corner and the last texel it owns. Two
                        // corners suffice: `tile_at_texel` is monotone in the
                        // texel, so an interior texel cannot leave a range both
                        // ends of which are inside it.
                        for (px, py) in [
                            (tx * d.tile_size, ty * d.tile_size),
                            (
                                ((tx + 1) * d.tile_size - 1).min(m.width - 1),
                                ((ty + 1) * d.tile_size - 1).min(m.height - 1),
                            ),
                        ] {
                            let u = (px as f64 + 0.5) / m.width as f64;
                            let v = (py as f64 + 0.5) / m.height as f64;
                            for target in mip..d.mip_count() {
                                let t = d.mips[target as usize];
                                let qx = ((u * t.width as f64) as u32).min(t.width - 1);
                                let qy = ((v * t.height as f64) as u32).min(t.height - 1);
                                let anc = d.ancestor(at, target).expect("an ancestor exists");
                                disagreements +=
                                    u64::from(Some(anc) != d.tile_at_texel(target, qx, qy));
                                for (q, a) in [(qx, anc.x), (qy, anc.y)] {
                                    let local = q as i64 - a as i64 * tile;
                                    lo = lo.min(local);
                                    hi = hi.max(local);
                                    assert!(
                                        (-border..tile + border).contains(&local),
                                        "{w}×{h}: {at:?} → mip {target} lands {local} texels \
                                         into a {tile}-texel tile with a {border}-texel border \
                                         — outside the page {anc:?} occupies"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        // Anti-vacuity, both halves. The bound above is satisfied trivially by a
        // sweep where the two derivations never part, and that is exactly what
        // the first version of this arm was: every odd extent it swept had a
        // ONE-TILE parent level, where the clamp hides the disagreement.
        assert!(
            disagreements > 0,
            "no address in the sweep disagreed with its uv derivation — the extents \
             stopped reaching the case the border is here for"
        );
        assert!(
            lo < 0,
            "the offset never went negative ({lo}..={hi}), so the border was never load-bearing"
        );
    }

    /// **The shader may not re-derive the tile from `uv` at the resolved mip.**
    ///
    /// The tempting line is `floor(gp / tile_size)`, and it is wrong wherever a
    /// level's extent is odd: the container halves with `w / 2`, dropping the odd
    /// texel, so `uv × w_coarse` is not `texel / 2` and the two land in different
    /// tiles at a tile boundary. The tile tree has no answer to that — a tile of
    /// a 511-texel level straddles two tiles of the 255-texel level above it, so
    /// it cannot have two parents — which is precisely why the offset is taken
    /// against the tile the *entry* names and the border ring absorbs the texel
    /// of slack.
    ///
    /// Measured, so the class is a number rather than an anecdote: 1 address of a
    /// 511×3 pyramid, 50 of 1023², 51 of 2047×511 and **1 322 of 4095²**. Each one
    /// is a sample fetched a whole tile — 128 texels — from where it belongs.
    #[test]
    fn the_shader_must_not_re_derive_the_tile_from_uv() {
        let d = full_pyramid(511, 3, 128, 4, false);
        assert_eq!((d.mips[0].tiles_x, d.mips[1].tiles_x), (4, 2));
        assert_eq!(d.mips[1].width, 255, "511 halves to 255, not to 256");

        let at = TileCoord::new(0, 2, 0);
        // The tile tree: tile 2 of mip 0 hangs under tile 1 of mip 1.
        assert_eq!(d.ancestor(at, 1), Some(TileCoord::new(1, 1, 0)));
        // The uv derivation, at this tile's very first texel: mip 0's texel 256
        // is 127.999 of mip 1 — tile 0.
        let u = (256.0 + 0.5) / 511.0;
        let qx = (u * 255.0) as u32;
        assert_eq!(qx, 127);
        assert_eq!(d.tile_at_texel(1, qx, 0), Some(TileCoord::new(1, 0, 0)));
        // One texel of slack against the tile the entry names — inside the border
        // ring, which the container fills from this level's own texels under the
        // same clamp. A whole tile of slack against the tile uv would have named.
        let local = |tile_x: u32| qx as i64 - tile_x as i64 * d.tile_size as i64;
        assert_eq!(local(1), -1, "against the tile the entry names");
        assert_eq!(local(0), 127, "against the tile uv would have named");
    }

    #[test]
    fn the_root_set_is_the_coarsest_level() {
        let full = full_pyramid(320, 192, 128, 4, false);
        assert_eq!(full.root_tiles(), vec![TileCoord::new(8, 0, 0)]);
        // A texture imported without mips has ONE level, and it is the coarsest —
        // so the whole grid is the mandatory floor.
        let flat = VtTextureDesc {
            tile_size: 128,
            border: 4,
            srgb: false,
            reconstruct_z: false,
            mips: vec![VtMipDesc {
                width: 320,
                height: 192,
                tiles_x: 3,
                tiles_y: 2,
            }],
        };
        flat.validate().expect("one level is a legal pyramid");
        assert_eq!(flat.root_tiles().len(), 6);
        assert_eq!(flat.root_tiles()[1], TileCoord::new(0, 1, 0));
    }

    #[test]
    fn a_descriptor_that_is_not_a_pyramid_is_refused_by_name() {
        let mut d = full_pyramid(320, 192, 128, 4, false);
        d.mips[1].tiles_x = 9;
        assert_eq!(d.validate(), Err(DescError::MipGrid { mip: 1 }));

        let mut d = full_pyramid(320, 192, 128, 4, false);
        d.mips.remove(1); // 320×192 straight to 80×48 — not a chain
        assert_eq!(d.validate(), Err(DescError::MipChain { mip: 1 }));

        let mut d = full_pyramid(320, 192, 128, 4, false);
        d.border = 3;
        assert_eq!(
            d.validate(),
            Err(DescError::TileGeometry {
                tile_size: 128,
                border: 3
            })
        );

        let d = VtTextureDesc {
            tile_size: 128,
            border: 4,
            srgb: false,
            reconstruct_z: false,
            mips: Vec::new(),
        };
        assert_eq!(d.validate(), Err(DescError::EmptyPyramid));
    }
}
