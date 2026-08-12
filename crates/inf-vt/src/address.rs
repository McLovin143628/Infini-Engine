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
//! The third one never sees a [`TileCoord`] at all: it computes the tile from the
//! `uv` directly at whatever mip the table hands it back. That the two agree —
//! that the clamped parent chain lands on exactly the tile a `uv` lands in — is
//! the **sampling contract**, and
//! [`tests::the_parent_chain_agrees_with_direct_texel_addressing`] asserts it
//! over a swept set of extents rather than assuming it.

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
    /// This is the CPU twin of what the fragment shader does when the table hands
    /// it back a coarser mip than it asked for — see the sampling contract in the
    /// module docs.
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
    /// The shader's arithmetic, on the CPU: it never walks a parent chain, it
    /// converts a `uv` to the level's texel coordinates and divides. The
    /// sampling-contract gate asserts this agrees with [`ancestor`](Self::ancestor)
    /// for every tile of every level of a swept set of extents.
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
        mips,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn every_child_range_inverts_the_parent_clamp() {
        for (w, h) in [
            (320u32, 192u32),
            (257, 257),
            (300, 260),
            (129, 3),
            (8192, 4),
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

    /// **The sampling contract**, on the CPU: the clamped parent chain lands on
    /// exactly the tile a texel of the coarser level falls in. P26.3's shader
    /// computes the right-hand side; this crate computes the left-hand side.
    #[test]
    fn the_parent_chain_agrees_with_direct_texel_addressing() {
        for (w, h) in [
            (320u32, 192u32),
            (257, 257),
            (300, 260),
            (129, 129),
            (8192, 4),
            (1000, 1),
        ] {
            let d = full_pyramid(w, h, 128, 4, false);
            for mip in 0..d.mip_count() {
                let m = d.mips[mip as usize];
                for ty in 0..m.tiles_y {
                    for tx in 0..m.tiles_x {
                        let at = TileCoord::new(mip, tx, ty);
                        // A texel inside this tile, in the level's own space, at
                        // both the low corner and the last texel it owns.
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
                                assert_eq!(
                                    d.ancestor(at, target),
                                    d.tile_at_texel(target, qx, qy),
                                    "{w}×{h}: {at:?} → mip {target} disagrees with uv ({u}, {v})"
                                );
                            }
                        }
                    }
                }
            }
        }
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
            mips: Vec::new(),
        };
        assert_eq!(d.validate(), Err(DescError::EmptyPyramid));
    }
}
