//! **The registration door** (P26.3): the one place a `.inf_tex` v2 payload
//! becomes a virtual texture, shared by the editor viewport and the shipped
//! player.
//!
//! # Why it is one door and not two
//!
//! The P16.6 mirror law, applied to texturing. Everything that decides *what a
//! surface samples* lives here: which payloads are registered, in what order,
//! which handle each gets, what the deterministic want floor is, and when a
//! texture is safe to name (warm). Both hosts call the same functions in the
//! same order, so "PIE == shipping" for VT is a property of the code rather than
//! of two projectors agreeing by inspection.
//!
//! It lives in `inf-render` rather than `inf-vt` because it needs the GPU mirror
//! ([`VtPools`](crate::vt::VtPools)) on one side and the container reader on the
//! other, and `inf-vt` is deliberately GPU-free. It needs **no asset crate**: a
//! host hands it bytes through [`VtTileSource`], so the pack/mmap path and the
//! loose-file path are the same door one indirection earlier — the arrangement
//! `VtPools::apply`'s `Cow` already encodes for tiles.
//!
//! # What it does NOT do
//!
//! It does not decide what a *material* is. A host resolves `.inf_mat` texture
//! GUIDs — from its asset database or from its pack — and asks
//! [`VtTextures::set_for`] for the three slots; the rule for turning three GUIDs
//! into a [`VtTextureSet`], including the warm gate, is here so it cannot be
//! written twice.
//!
//! It does not stream. The want set is [`VtTextures::want_floor`]'s — a
//! deterministic, camera-free, conservative floor computed from the pyramid
//! alone. P26.4's feedback ring refines it and can never regress below it; a
//! dropped feedback frame degrades to exactly this.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

use inf_vt::{
    PageFormat, TileCoord, TiledTextureReader, VtAdvisory, VtError, VtPoolConfig, VtResidency,
    VtTextureHandle, VtTransaction, VtWant,
};

use crate::scene::VtTextureSet;
use crate::vt::{VtApplyReport, VtPools};

/// How many of the **coarsest** levels of every registered texture the
/// deterministic floor admits.
///
/// Three, and the number is bounded by construction rather than tuned: the
/// coarsest level is one tile, the one below it at most four, the one below that
/// at most sixteen — so a texture costs at most 21 pages however large it is,
/// and 24 MiB of BC1 pages (2 714 slots) holds the floor for 129 textures before
/// anything is deferred. The coarsest level is *already* mandatory and pinned by
/// `inf-vt`'s first law; this adds the two below it so a surface is visibly
/// textured rather than visibly one colour, which is what makes P26.3 shippable
/// with no feedback at all.
pub const VT_FLOOR_LEVELS: u32 = 3;

/// A host's handle on one `.inf_tex` v2 payload.
///
/// The whole seam between "where the bytes live" and "which tile the residency
/// wants". A pack-backed host returns a slice of its mmap; a loose-file host
/// returns a slice of a `Vec` it read once. Neither is named here, which is why
/// this module needs no asset crate and the shipped player pays for no decoder.
pub trait VtTileSource: Send + Sync {
    /// The whole `.inf_tex` v2 payload.
    fn payload(&self) -> &[u8];
}

/// An owned payload — the loose-file / test door.
impl VtTileSource for Vec<u8> {
    fn payload(&self) -> &[u8] {
        self
    }
}

/// Adapter that lets a [`VtTileSource`] back a [`TiledTextureReader`].
#[derive(Clone)]
pub struct VtBytes(Arc<dyn VtTileSource>);

impl AsRef<[u8]> for VtBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.payload()
    }
}

/// Why a payload did not become a virtual texture. Counted and reported rather
/// than logged, on `inf-vt`'s advisory doctrine: a host decides whether a texture
/// it cannot page is a warning, a fatal, or a line in an import report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VtRefusal {
    /// The payload is not a readable v2 container.
    Container(String),
    /// The residency refused it — a geometry mismatch, or a mandatory floor that
    /// does not fit the budget.
    Residency(VtError),
}

/// The registry: every virtual texture this level has, and the residency behind
/// them.
pub struct VtTextures {
    residency: VtResidency,
    /// Asset GUID (as `u128`) → handle. **The dedupe**: two materials naming one
    /// texture register it once and share a handle, so the atlas holds one copy.
    by_guid: BTreeMap<u128, VtTextureHandle>,
    /// Parsed readers, indexed by handle. Parallel to the residency's textures.
    readers: Vec<TiledTextureReader<VtBytes>>,
    /// The pool's page format — `Bc1`/`Bc3` when the adapter has BC, else
    /// `Rgba8` and every fetch transcodes through the same door.
    pool_format: PageFormat,
    refusals: Vec<(u128, VtRefusal)>,
}

impl VtTextures {
    /// A registry over a pool planned from `cfg`.
    ///
    /// `cfg.format` is the **pool's** format, which a caller derives from the
    /// container's format and the adapter's clamp through
    /// [`crate::vt::pool_format`] — the same door, one decision earlier.
    pub fn new(cfg: VtPoolConfig) -> (Self, Vec<VtAdvisory>) {
        let (residency, advisories) = VtResidency::new(cfg);
        (
            Self {
                residency,
                by_guid: BTreeMap::new(),
                readers: Vec::new(),
                pool_format: cfg.format,
                refusals: Vec::new(),
            },
            advisories,
        )
    }

    /// **Register a `.inf_tex` v2 payload, idempotently by GUID.**
    ///
    /// Returns the handle it already had when it is already registered, so a
    /// host may call this for every material it projects without keeping its own
    /// side index, and so one texture costs one atlas copy however many materials
    /// name it.
    ///
    /// **A handle is a per-registry index, not a shared name.** It is the order
    /// of first sight, and the two projectors do not share that order: the P16.6
    /// mirror law has the editor walking DOCUMENT order and the player walking
    /// `Guid` order, so one level mints different handles on the two sides. That
    /// is correct and must stay correct — each host's own indirection table maps
    /// its own handle to the same texture. What it forbids is a cross-host
    /// comparison of [`VtTextureSet`] numbers: **two projectors compare by GUID**,
    /// which is what [`handle`](Self::handle) is for, and
    /// `two_projectors_in_different_orders_agree_by_guid_and_not_by_handle` pins.
    ///
    /// The same caveat reaches [`want_floor`](Self::want_floor): it is a pure
    /// function of the registration *sequence*, so two orders emit the same wants
    /// in different orders. Slot assignment then differs — harmless, since the
    /// table is per host — but a budget too small for the whole floor would defer
    /// a different tail on each side. P26.4's wiring is where that gets a rule.
    pub fn register(
        &mut self,
        guid: u128,
        bytes: Arc<dyn VtTileSource>,
    ) -> Result<VtTextureHandle, VtRefusal> {
        if let Some(h) = self.by_guid.get(&guid) {
            return Ok(*h);
        }
        let reader = TiledTextureReader::new(VtBytes(bytes))
            .map_err(|e| VtRefusal::Container(e.to_string()))?;
        let handle = self
            .residency
            .register_texture(reader.vt_desc())
            .map_err(VtRefusal::Residency)?;
        debug_assert_eq!(handle.index(), self.readers.len());
        self.readers.push(reader);
        self.by_guid.insert(guid, handle);
        Ok(handle)
    }

    /// [`register`](Self::register), recording a refusal instead of returning it
    /// — the shape a projector wants, since a texture it cannot page must not
    /// take a frame down.
    pub fn register_or_record(
        &mut self,
        guid: u128,
        bytes: Arc<dyn VtTileSource>,
    ) -> Option<VtTextureHandle> {
        match self.register(guid, bytes) {
            Ok(h) => Some(h),
            Err(e) => {
                self.refusals.push((guid, e));
                None
            }
        }
    }

    /// Payloads that did not become virtual textures, in the order they were
    /// refused.
    pub fn refusals(&self) -> &[(u128, VtRefusal)] {
        &self.refusals
    }

    /// The handle a GUID registered under, if any.
    pub fn handle(&self, guid: u128) -> Option<VtTextureHandle> {
        self.by_guid.get(&guid).copied()
    }

    /// **The material rule**: three `.inf_mat` texture GUIDs → the per-instance
    /// [`VtTextureSet`] the shader reads.
    ///
    /// Written once so the editor viewport and the shipped player cannot resolve
    /// a material differently — the thing the mirror gate compares.
    ///
    /// A GUID that is not registered, or whose texture is **not yet warm**,
    /// contributes `0` and the instance falls back to its scalar attribute for
    /// that map. Warmth is not a nicety: a texture's indirection table is
    /// complete from registration, but the *pages* behind it exist only once the
    /// transaction carrying them has been applied, so naming it a frame early
    /// would sample an atlas slot holding some other texture's texels. Never a
    /// hole, never a stall, and never last texture's colour.
    pub fn set_for(
        &self,
        albedo: Option<u128>,
        normal: Option<u128>,
        orm: Option<u128>,
    ) -> VtTextureSet {
        let slot = |g: Option<u128>| {
            g.and_then(|g| self.by_guid.get(&g))
                .filter(|h| self.residency.is_warm(**h))
                .map_or(0, |h| h.0 + 1)
        };
        VtTextureSet {
            albedo: slot(albedo),
            normal: slot(normal),
            orm: slot(orm),
        }
    }

    /// **The deterministic want floor**: the coarsest [`VT_FLOOR_LEVELS`] levels
    /// of every registered texture, in registration order and payload order
    /// within a texture.
    ///
    /// Camera-free and history-free on purpose. It is a pure function of the
    /// registration *sequence*, so two runs of one scene **that registered in one
    /// order** produce one want sequence and therefore one residency trace — the
    /// property the phase gate pins — and P26.4's feedback becomes an addition to
    /// it rather than a replacement for it. Two different orders are the caveat
    /// on [`register`](Self::register), not a second guarantee.
    pub fn want_floor(&self) -> Vec<VtWant> {
        let mut out = Vec::new();
        for t in 0..self.residency.texture_count() {
            let handle = VtTextureHandle(t as u32);
            let Some(desc) = self.residency.desc(handle) else {
                continue;
            };
            let coarsest = desc.coarsest_mip();
            let finest = coarsest.saturating_sub(VT_FLOOR_LEVELS.saturating_sub(1));
            for mip in (finest..=coarsest).rev() {
                let m = desc.mips[mip as usize];
                for y in 0..m.tiles_y {
                    for x in 0..m.tiles_x {
                        out.push(VtWant::new(handle, TileCoord::new(mip, x, y)));
                    }
                }
            }
        }
        out
    }

    /// Advance residency toward `wants` and apply the result to the GPU mirror.
    ///
    /// **Call at the frame's sync point, before the frame's `submit`** — the
    /// ordering contract `crate::vt`'s module docs state, unchanged.
    pub fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pools: &mut VtPools,
        wants: &[VtWant],
    ) -> (VtTransaction, VtApplyReport) {
        let txn = self.residency.apply_wants(wants);
        // The mutable borrow above has ended; the fetch below needs only `&`.
        let readers = &self.readers;
        let format = self.pool_format;
        let report = pools.apply(device, queue, &self.residency, &txn, |admit| {
            let r = readers.get(admit.texture.index())?;
            let at = admit.tile;
            if format == PageFormat::Rgba8 && r.header().format != PageFormat::Rgba8 {
                // The transcode tier: the SAME tile, one format decision later.
                r.tile_rgba8(at.mip, at.x, at.y).map(Cow::Owned)
            } else {
                r.tile_at(at).map(Cow::Borrowed)
            }
        });
        (txn, report)
    }

    /// The residency, for a gate that wants to assert the WORLD.
    #[inline]
    pub fn residency(&self) -> &VtResidency {
        &self.residency
    }

    /// The residency, for a gate that wants to force an eviction.
    #[inline]
    pub fn residency_mut(&mut self) -> &mut VtResidency {
        &mut self.residency
    }

    /// How many textures are registered.
    #[inline]
    pub fn len(&self) -> usize {
        self.readers.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.readers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_vt::{full_pyramid, DEFAULT_VT_BUDGET_BYTES, STORED_TILE_SIZE};

    fn cfg() -> VtPoolConfig {
        VtPoolConfig {
            format: PageFormat::Bc1,
            stored_tile_size: STORED_TILE_SIZE,
            budget_bytes: DEFAULT_VT_BUDGET_BYTES,
            max_texture_dim: 8192,
        }
    }

    /// The floor is **bounded by the pyramid's tail, not by the texture's size**
    /// — which is the whole reason it can be admitted unconditionally at load.
    #[test]
    fn the_floor_is_at_most_twenty_one_pages_however_large_the_texture() {
        for (w, h) in [(4096u32, 4096u32), (320, 192), (8192, 4), (1, 1)] {
            let desc = full_pyramid(w, h, 128, 4, true);
            let coarsest = desc.coarsest_mip();
            let finest = coarsest.saturating_sub(VT_FLOOR_LEVELS - 1);
            let pages: u32 = (finest..=coarsest)
                .map(|m| desc.mips[m as usize].tile_count())
                .sum();
            assert!(
                pages <= 21,
                "{w}×{h}: a {VT_FLOOR_LEVELS}-level floor is {pages} pages"
            );
        }
    }

    /// A GUID registers **once**: two materials naming one texture share a
    /// handle, so the atlas holds one copy and the two projectors cannot disagree
    /// about which handle it is.
    #[test]
    fn a_guid_registers_once_and_the_handle_is_first_sight_order() {
        let (mut lib, _) = VtTextures::new(cfg());
        let a = tiled(64, 64, true);
        let b = tiled(32, 32, false);
        let ha = lib.register(7, Arc::new(a.clone())).expect("a registers");
        let hb = lib.register(9, Arc::new(b)).expect("b registers");
        assert_eq!((ha.0, hb.0), (0, 1), "handles follow first sight");
        assert_eq!(
            lib.register(7, Arc::new(a)).expect("idempotent"),
            ha,
            "a second registration of one GUID must not mint a second texture"
        );
        assert_eq!(lib.len(), 2);
        assert_eq!(lib.handle(7), Some(ha));
        assert_eq!(lib.handle(11), None);
    }

    /// **The warm gate.** A texture's table is complete from registration but its
    /// pages are not, so `set_for` must refuse to name it until a transaction has
    /// carried them — otherwise the shader samples a slot holding some other
    /// texture's texels.
    #[test]
    fn a_set_names_nothing_until_the_texture_is_warm() {
        let (mut lib, _) = VtTextures::new(cfg());
        let h = lib
            .register(7, Arc::new(tiled(64, 64, true)))
            .expect("registers");
        assert!(
            lib.set_for(Some(7), None, None).is_none(),
            "a registered-but-cold texture must not be named"
        );
        // The transaction that carries its root pages.
        let _ = lib.residency_mut().apply_wants(&[]);
        assert!(lib.residency().is_warm(h));
        assert_eq!(
            lib.set_for(Some(7), None, None),
            VtTextureSet {
                albedo: h.0 + 1,
                normal: 0,
                orm: 0
            },
            "a warm texture is named as handle + 1, and the absent maps as 0"
        );
        // An unregistered GUID is 0, not a panic and not a wrong handle.
        assert_eq!(lib.set_for(Some(99), Some(7), None).albedo, 0);
    }

    /// The floor is a pure function of what is registered — asserted as byte
    /// equality between two independently built registries, which is what makes
    /// the residency trace comparable between two hosts.
    #[test]
    fn the_want_floor_is_a_pure_function_of_the_registration_order() {
        let build = || {
            let (mut lib, _) = VtTextures::new(cfg());
            lib.register(7, Arc::new(tiled(320, 192, true))).unwrap();
            lib.register(9, Arc::new(tiled(64, 64, false))).unwrap();
            lib
        };
        let (a, b) = (build(), build());
        assert_eq!(a.want_floor(), b.want_floor());
        assert!(!a.want_floor().is_empty(), "the floor wanted nothing");
        // Coarse-to-fine within a texture: the first want of each texture is its
        // coarsest level, so an admit burst walks the pyramid the way a fallback
        // resolves it.
        let floor = a.want_floor();
        let first = floor[0];
        assert_eq!(
            first.tile.mip,
            a.residency().desc(first.texture).unwrap().coarsest_mip()
        );
    }

    /// A payload that is not a v2 container is refused **by name** and recorded,
    /// not panicked on and not silently skipped.
    #[test]
    fn a_bad_payload_is_refused_by_name() {
        let (mut lib, _) = VtTextures::new(cfg());
        assert!(matches!(
            lib.register(1, Arc::new(vec![0u8; 16])),
            Err(VtRefusal::Container(_))
        ));
        assert!(lib.register_or_record(1, Arc::new(vec![0u8; 16])).is_none());
        assert_eq!(lib.refusals().len(), 1);
        assert_eq!(lib.refusals()[0].0, 1);
        assert!(lib.is_empty());
    }

    /// A real v2 container of `w × h`, built by the one writer.
    fn tiled(w: u32, h: u32, srgb: bool) -> Vec<u8> {
        inf_material::build_tiled_texture(
            vec![200u8; (w * h * 4) as usize],
            w,
            h,
            inf_material::TextureImportSettings {
                srgb,
                generate_mips: true,
                compression: inf_material::TextureCompression::Bc1,
            },
        )
        .expect("the fixture tiles")
        .into_bytes()
    }
}
