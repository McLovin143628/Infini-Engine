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
    /// The host could not produce bytes for a GUID a bound material names
    /// (P26.4). Its own variant because it is the only one that is *not* about
    /// the payload: the pack or the content root simply did not have it, which
    /// is the state the cook's `dangling_material_refs` advisory warns about at
    /// build time and this is what it looks like at run time.
    NoBytes,
}

/// **The three texture GUIDs one material binds** — the shape both hosts hand
/// the registry, and deliberately not `inf_asset::DerivedMaterial`.
///
/// `inf-render` names no asset crate (the whole P26.2 dependency inversion), so
/// the record it registers from is three `Option<u128>`s and nothing else. The
/// player converts a `DerivedMaterial` into one of these at the boundary; the
/// editor converts a resolved `.inf_mat` into one at its own; and the *order*
/// rule below is then written once, here, where both of them link it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VtMaterialMaps {
    pub albedo: Option<u128>,
    pub normal: Option<u128>,
    pub orm: Option<u128>,
}

impl VtMaterialMaps {
    /// The GUIDs this material names, in the **fixed slot order** albedo →
    /// normal → ORM, with absent slots skipped rather than shifting the next
    /// one into their place.
    pub fn texture_guids(&self) -> impl Iterator<Item = u128> + '_ {
        [self.albedo, self.normal, self.orm].into_iter().flatten()
    }

    /// Whether this material names any texture at all — `false` is the
    /// scalars-only surface, the permanent no-texture path.
    pub fn is_empty(&self) -> bool {
        self.albedo.is_none() && self.normal.is_none() && self.orm.is_none()
    }
}

/// **THE registration order.** Materials by GUID, then each one's albedo →
/// normal → ORM, deduplicated, first sight wins.
///
/// One function, in the crate both hosts link, because the order *is* the
/// residency: [`VtTextures::want_floor`] is a pure function of the registration
/// **sequence**, so two hosts walking one level's materials in two orders page
/// different tiles from the same pack — silently, and only under a pool small
/// enough for the tail to be deferred. The P26.3 LAW says a *handle* is a
/// per-registry index and may differ between hosts; it explicitly does not
/// license the pages to.
///
/// A `BTreeMap` rather than a sorted `Vec` at the call site so the ordering is
/// structural: the P26.3b audit's finding was a `HashMap` walk that produced two
/// answers on two machines and looked correct in review.
pub fn registration_order(materials: &BTreeMap<u128, VtMaterialMaps>) -> Vec<u128> {
    let mut out: Vec<u128> = Vec::new();
    for maps in materials.values() {
        for tex in maps.texture_guids() {
            if !out.contains(&tex) {
                out.push(tex);
            }
        }
    }
    out
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
    /// `.inf_mat` GUID → its three maps, for [`VtTextures::set_for_material`].
    /// A `BTreeMap` because [`registration_order`] walks it and the walk order is
    /// the residency.
    materials: BTreeMap<u128, VtMaterialMaps>,
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
                materials: BTreeMap::new(),
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

    /// **Register a whole level's material content, in the one order** (P26.4).
    ///
    /// This is the door the P26.3b ledger named as the missing glue: *"both
    /// projectors still fill `vt: Default::default()` and no non-test code calls
    /// `VtTextures::register`"*. `materials` is what the level BINDS —
    /// `Material.asset` → a flattened record — not what its pack happens to
    /// hold, which is the distinction that audit measured (an imported `.inf_mesh`
    /// drags its own materials into the closure through sidecar edges, and one
    /// extra material moves every page after it).
    ///
    /// `fetch` resolves a texture GUID to its `.inf_tex` v2 bytes: a slice of a
    /// pack mmap on one host, a file read on the other. A GUID it cannot serve is
    /// **counted as a refusal** rather than skipped, so "the level is textureless"
    /// and "the level's textures did not load" are different observable states.
    ///
    /// Returns how many textures were registered (existing ones included — the
    /// registration is idempotent by GUID).
    pub fn register_materials(
        &mut self,
        materials: &BTreeMap<u128, VtMaterialMaps>,
        mut fetch: impl FnMut(u128) -> Option<Arc<dyn VtTileSource>>,
    ) -> usize {
        let mut registered = 0usize;
        for guid in registration_order(materials) {
            if self.by_guid.contains_key(&guid) {
                registered += 1;
                continue;
            }
            match fetch(guid) {
                Some(bytes) => {
                    if self.register_or_record(guid, bytes).is_some() {
                        registered += 1;
                    }
                }
                None => self.refusals.push((guid, VtRefusal::NoBytes)),
            }
        }
        for (mat, maps) in materials {
            self.materials.insert(*mat, *maps);
        }
        registered
    }

    /// **The per-instance set for a surface bound to `material`.**
    ///
    /// The projector's whole job, once [`register_materials`](Self::register_materials)
    /// has run: an entity's `Material.asset` is a `.inf_mat` GUID, and this is
    /// what its `MeshInstance::vt` becomes. A material this registry never saw —
    /// an unresolvable binding, a pre-v22 level, a `None` — is
    /// [`VtTextureSet::NONE`], which is the scalar surface and is exactly what
    /// the same entity rendered as before P26.
    ///
    /// Routed through [`set_for`](Self::set_for) rather than reimplementing it,
    /// so the warm gate is not written twice.
    pub fn set_for_material(&self, material: u128) -> VtTextureSet {
        match self.materials.get(&material) {
            Some(m) => self.set_for(m.albedo, m.normal, m.orm),
            None => VtTextureSet::NONE,
        }
    }

    /// The materials this registry was built from, in GUID order.
    #[inline]
    pub fn materials(&self) -> &BTreeMap<u128, VtMaterialMaps> {
        &self.materials
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

/// **The per-instance set for an optional material binding** — the one
/// expression both projectors spell (P26.4, clause 0).
///
/// A free function rather than two `map_or` chains because `projector_mirror`
/// compares the two projections *token for token*: a rule written twice is a
/// rule that eventually differs, and this one decides whether a surface is
/// textured. `None` for the library (no VT level) and `None` for the material
/// (an unbound surface, i.e. every pre-v22 level) both give
/// [`VtTextureSet::NONE`], which is the scalar path.
#[inline]
pub fn vt_set_for(lib: Option<&VtTextures>, material: Option<u128>) -> VtTextureSet {
    match (lib, material) {
        (Some(l), Some(m)) => l.set_for_material(m),
        _ => VtTextureSet::NONE,
    }
}

/// What building a level's virtual-texture surface produced — everything a host
/// wants to log or a gate wants to assert, so neither has to reach inside.
#[derive(Debug, Clone, Default)]
pub struct VtLevelReport {
    /// Textures registered.
    pub textures: usize,
    /// GUIDs a bound material named that did not become a virtual texture.
    pub refused: usize,
    /// The page format the pool took.
    pub pool_format: Option<PageFormat>,
    /// Advisories the pool planner raised (`inf-vt` returns rather than logs).
    pub advisories: Vec<VtAdvisory>,
}

/// **Build a level's virtual-texture registry and its GPU mirror** — the one
/// call both projectors make (P26.4, clause 0).
///
/// `None` when the level binds no material with any texture at all. That is not
/// a failure and it is not a branch a host has to remember: a `None` pool means
/// `EngineRenderer` binds the empty VT surface, `vt_active()` is false in every
/// lit shader, and the command stream is the one all 50 goldens recorded. The
/// textureless path stays *structural*.
///
/// # One pool holds one page size
///
/// A slot is one fixed size for every tile of every texture (P26.1's uniform
/// page), so a pool that is BC1 cannot hold a BC3 tile — the fetch would be the
/// wrong length, `VtPools::apply` would report it missing and write a black
/// page, and the surface would be silently wrong. So the format is decided here,
/// once, from what the level actually stores:
///
/// * the adapter has no BC (`vt.bc_tiles` clamped off) → **RGBA8**, every tile
///   transcoded through `tile_rgba8` — the P26.1 fallback tier, unchanged;
/// * every bound texture stores the same BC format → **that** format, and a tile
///   uploads exactly as it lies in the mmap;
/// * the level **mixes** formats (a BC1 albedo and a BC3 mask, which an importer
///   writes routinely — BC3 is chosen per texture by whether it has alpha) →
///   **RGBA8**, because that is the one format every container can transcode
///   into. It costs 8×/4× the page bytes and it is the honest answer: the
///   alternative is refusing half a level's textures at registration.
///
/// The mixed case is reported through [`VtLevelReport::pool_format`] rather than
/// only inferred, because it is a real cost a project can avoid by importing its
/// masks the same way.
pub fn build_vt_level(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    settings: &crate::settings::RenderSettings,
    budget_bytes: u64,
    materials: &BTreeMap<u128, VtMaterialMaps>,
    mut fetch: impl FnMut(u128) -> Option<Arc<dyn VtTileSource>>,
) -> Option<(VtTextures, VtPools, VtLevelReport)> {
    // Resolve once, in the registration order, so `fetch` is called exactly once
    // per GUID and the bytes are held for the pool-format decision AND the
    // registration that follows it.
    let resolved: Vec<(u128, Option<Arc<dyn VtTileSource>>)> = registration_order(materials)
        .into_iter()
        .map(|g| (g, fetch(g)))
        .collect();
    if resolved.is_empty() {
        return None;
    }

    // The stored formats present, in registration order. A payload that does not
    // parse contributes nothing here and is refused by name a moment later.
    let mut stored: Option<PageFormat> = None;
    let mut mixed = false;
    for (_, bytes) in &resolved {
        let Some(b) = bytes else { continue };
        let Some(f) = inf_vt::stored_page_format(b.payload()) else {
            continue;
        };
        match stored {
            None => stored = Some(f),
            Some(s) if s != f => mixed = true,
            Some(_) => {}
        }
    }
    let pool_format = match stored {
        Some(f) if !mixed => crate::vt::pool_format(settings, f),
        // Mixed, or nothing parsed: RGBA8 is the format every container
        // transcodes into, so it is the total answer.
        _ => PageFormat::Rgba8,
    };

    let (mut lib, advisories) = VtTextures::new(VtPoolConfig {
        format: pool_format,
        stored_tile_size: inf_vt::STORED_TILE_SIZE,
        budget_bytes,
        max_texture_dim: device.limits().max_texture_dimension_2d,
    });
    let mut by_guid: BTreeMap<u128, Arc<dyn VtTileSource>> = BTreeMap::new();
    for (g, bytes) in resolved {
        if let Some(b) = bytes {
            by_guid.insert(g, b);
        }
    }
    let textures = lib.register_materials(materials, |g| by_guid.get(&g).cloned());
    if lib.is_empty() {
        // Every bound texture was refused. The report is lost with the registry,
        // so it is logged here — this is the shape of "a project that predates
        // P26.1 has v1 `.inf_tex` payloads", which the cook already advises about
        // and which renders as a perfectly ordinary untextured level.
        tracing::warn!(
            "inf-render: {} bound texture(s) did not become virtual textures; every \
             surface renders off its scalar attributes",
            lib.refusals().len()
        );
        return None;
    }
    let pools = VtPools::new(device, queue, lib.residency(), false);
    let report = VtLevelReport {
        textures,
        refused: lib.refusals().len(),
        pool_format: Some(pool_format),
        advisories,
    };
    Some((lib, pools, report))
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

    // ── the registration door (P26.4, clause 0) ─────────────────────────────

    fn maps(a: Option<u128>, n: Option<u128>, o: Option<u128>) -> VtMaterialMaps {
        VtMaterialMaps {
            albedo: a,
            normal: n,
            orm: o,
        }
    }

    /// **The registration order is a function of the SET**, not of how the
    /// caller assembled it: materials by GUID, then albedo → normal → ORM, with
    /// an absent slot skipped rather than shifting the next one into its place.
    ///
    /// The P26.3b audit found this defect in the player's copy of the walk (a
    /// `HashMap` iteration, which is seeded per map and therefore genuinely
    /// differs between two builds of the same content). Moving the rule here is
    /// what stops the editor growing a second copy of it, so the rule itself is
    /// pinned here rather than only at the two call sites.
    #[test]
    fn the_registration_order_is_materials_then_fixed_slots() {
        let mut m = BTreeMap::new();
        // Inserted back to front, and with a hole in the middle slot.
        m.insert(30u128, maps(Some(300), None, Some(301)));
        m.insert(10, maps(Some(100), Some(101), Some(102)));
        m.insert(20, maps(None, None, Some(200)));
        assert_eq!(
            registration_order(&m),
            vec![100, 101, 102, 200, 300, 301],
            "the order is not (material GUID, albedo, normal, ORM)"
        );

        // A texture two materials share registers ONCE, at first sight.
        let mut shared = BTreeMap::new();
        shared.insert(2u128, maps(Some(9), None, None));
        shared.insert(1, maps(Some(9), Some(8), None));
        assert_eq!(registration_order(&shared), vec![9, 8]);

        // Anti-vacuity: the answer is not the insertion order, so the equality
        // above is a statement about the rule and not about a constant.
        let mut reversed: Vec<u128> = registration_order(&m);
        reversed.reverse();
        assert_ne!(registration_order(&m), reversed);
    }

    /// **The door registers what the level BINDS, in that order, and counts what
    /// it could not fetch** — the P26.3b remainder, closed.
    #[test]
    fn register_materials_walks_the_order_and_names_what_it_could_not_fetch() {
        let (mut lib, _) = VtTextures::new(cfg());
        let mut m = BTreeMap::new();
        m.insert(2u128, maps(Some(70), None, None));
        m.insert(1, maps(Some(50), Some(60), None));
        // GUID 60 is bound and unavailable — a dangling `.inf_tex`, which is the
        // state the cook's advisory warns about at build time.
        let bytes: BTreeMap<u128, Vec<u8>> =
            [(50u128, tiled(64, 64, true)), (70, tiled(32, 32, false))]
                .into_iter()
                .collect();
        let n = lib.register_materials(&m, |g| {
            bytes
                .get(&g)
                .cloned()
                .map(|b| Arc::new(b) as Arc<dyn VtTileSource>)
        });

        assert_eq!(n, 2);
        assert_eq!(lib.len(), 2);
        // Handles follow the registration order — material 1's albedo first.
        assert_eq!(lib.handle(50).map(|h| h.0), Some(0));
        assert_eq!(lib.handle(70).map(|h| h.0), Some(1));
        assert_eq!(lib.handle(60), None);
        assert_eq!(
            lib.refusals(),
            &[(60u128, VtRefusal::NoBytes)],
            "a bound texture the host could not serve was skipped in silence"
        );

        // The per-instance sets come out of the same registry: warm first (the
        // transaction that carries the root pages), then the sets are real.
        assert!(
            lib.set_for_material(1).is_none(),
            "a cold registry must not name a texture"
        );
        let _ = lib.residency_mut().apply_wants(&[]);
        assert_eq!(
            lib.set_for_material(1),
            VtTextureSet {
                albedo: 1,
                normal: 0,
                orm: 0
            },
            "the bound albedo is handle 0 + 1, and the unfetchable normal is 0"
        );
        assert_eq!(
            lib.set_for_material(2),
            VtTextureSet {
                albedo: 2,
                normal: 0,
                orm: 0
            }
        );
        // A material this registry never saw is the SCALAR surface, not a panic
        // and not another material's textures.
        assert_eq!(lib.set_for_material(999), VtTextureSet::NONE);
    }

    /// Two hosts that resolve one level's materials **in different orders** page
    /// the same tiles, because the door sorts rather than trusting its caller.
    ///
    /// The P26.3 handle LAW says the two hosts mint different handles and compare
    /// by GUID; it does not license them to page different sets. This is the
    /// property that makes the two claims consistent, asserted on the want floor
    /// projected back onto GUIDs.
    #[test]
    fn two_hosts_registering_in_opposite_orders_page_the_same_tiles() {
        let build = |rev: bool| {
            let (mut lib, _) = VtTextures::new(cfg());
            let mut m = BTreeMap::new();
            let mut ids: Vec<u128> = (0..4).collect();
            if rev {
                ids.reverse();
            }
            for i in ids {
                m.insert(10 + i, maps(Some(100 + i), None, None));
            }
            let n = lib.register_materials(&m, |g| {
                Some(Arc::new(tiled(64 + (g % 4) as u32 * 32, 64, true)) as Arc<dyn VtTileSource>)
            });
            assert_eq!(n, 4);
            lib
        };
        let (a, b) = (build(false), build(true));
        // The floor is a sequence of (handle, tile); projected back onto GUIDs it
        // must be the same sequence on both sides.
        let guids = |lib: &VtTextures| -> Vec<(u128, inf_vt::TileCoord)> {
            lib.want_floor()
                .into_iter()
                .map(|w| {
                    let g = *lib
                        .materials()
                        .values()
                        .flat_map(|m| m.texture_guids())
                        .collect::<Vec<_>>()
                        .iter()
                        .find(|g| lib.handle(**g) == Some(w.texture))
                        .expect("every floor want names a registered texture");
                    (g, w.tile)
                })
                .collect()
        };
        assert_eq!(guids(&a), guids(&b));
        assert!(!guids(&a).is_empty(), "the floor wanted nothing");
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
